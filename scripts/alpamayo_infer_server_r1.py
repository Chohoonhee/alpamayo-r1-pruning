"""
alpamayo_infer_server.py  –  Python 3.12 / Alpamayo env
==========================================================
ZMQ REPLY server that loads Alpamayo-R1-10B once and serves
trajectory inference requests from the CARLA bridge agent.

Protocol (JSON over ZMQ REP-REQ):
  Request:
    {
      "frames":   [[C,H,W] uint8 list], # 4 frames
      "ego_xyz":  [[[x,y,z]…×16]],     # (1,1,16,3)
      "ego_rot":  [[[[r00…r22]…]×16]], # (1,1,16,3,3)
      "nav":      "Turn left…"
    }
  Response:
    {"ok": true,  "waypoints": [[x,y,z]×64]}
    {"ok": false, "error": "…"}

Usage:
  conda activate alpamayo_b2d
  python alpamayo_infer_server.py [--model <hf_id_or_path>] [--port <zmq_port>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import zmq

# Alpamayo
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.helper import create_message, to_device

# Depth Anything V2 for depth-aware CoT
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL  = os.environ.get("ALPAMAYO_MODEL",
                                "/home/irteam/ws/alpamayo_bench2drive/alpamayo_weights")
DEFAULT_PORT   = int(os.environ.get("INFER_PORT", "5556"))
N_TRAJ_SAMPLES = int(os.environ.get("ALPAMAYO_TRAJ_SAMPLES", "4"))  # 4 samples for nav-based selection
DEVICE         = os.environ.get("ALPAMAYO_DEVICE", "cuda:0")
DTYPE          = torch.bfloat16
USE_DEPTH      = os.environ.get("USE_DEPTH", "0") == "1"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str, drop_layers_json: str | None = None) -> AlpamayoR1:
    print(f"[InferServer] Loading Alpamayo from: {model_name}")
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    model.eval()

    if drop_layers_json:
        import json as _json
        with open(drop_layers_json) as _f:
            meta = _json.load(_f)
        drop_idx = sorted(set(meta["dropped_layers"]))
        layers = model.vlm.language_model.layers
        n_orig = len(layers)
        print(f"[InferServer] Runtime pruning (identity bypass): "
              f"drop {len(drop_idx)}/{n_orig} text layers")
        print(f"[InferServer]   dropped (bypassed as identity): {drop_idx}")

        # Patch forward of dropped layers to return input unchanged
        # Qwen3VL decoder layer returns hidden_states (or tuple)
        def _make_identity_forward(original_layer):
            def _identity_fwd(hidden_states, *args, **kwargs):
                # Mirror the return signature of the original layer
                return hidden_states
            return _identity_fwd

        for idx in drop_idx:
            layers[idx].forward = _make_identity_forward(layers[idx])
            # Zero out the parameters so memory can be freed conceptually
            # (can't free without removing module, but at least mark as unused)

        n_keep = n_orig - len(drop_idx)
        print(f"[InferServer] After pruning: {n_keep} active text layers "
              f"({len(drop_idx)} bypassed), VRAM not reduced (identity bypass)")

    torch.cuda.empty_cache()
    print(f"[InferServer] Model loaded in {time.time()-t0:.1f}s  "
          f"(GPU: {torch.cuda.get_device_name(0)})")
    return model


def load_depth_model():
    """Load Depth Anything V2 Small (~200MB GPU memory)."""
    print(f"[InferServer] Loading Depth Anything V2 Small...")
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
    depth_model.eval()
    print(f"[InferServer] Depth model loaded.")
    return depth_model, depth_processor


def estimate_depth_info(depth_model, depth_processor, front_image_np: np.ndarray) -> str:
    """Estimate depth from front camera and return a natural language description.

    Analyzes: (1) left/right asymmetry (lane centering)
              (2) forward obstacles
              (3) road depth/clearance

    Args:
        front_image_np: (3, H, W) uint8 numpy array (RGB, CHW format)

    Returns:
        Natural language description for CoT prefix injection.
    """
    # Convert CHW → HWC for PIL
    img_hwc = front_image_np.transpose(1, 2, 0)
    pil_img = PILImage.fromarray(img_hwc)

    inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth  # (1, H, W) relative depth

    depth_np = depth[0].cpu().float().numpy()  # (H, W)
    h, w = depth_np.shape

    parts = []

    # --- 1. Left/Right asymmetry → lane centering ---
    # Lower half: compare left vs right proximity
    # Higher depth value = closer object (inverse depth)
    left_region  = depth_np[h//2:, :w//3]       # left third
    right_region = depth_np[h//2:, 2*w//3:]     # right third
    left_prox  = float(left_region.mean())
    right_prox = float(right_region.mean())

    asym_ratio = left_prox / (right_prox + 1e-6)
    if asym_ratio > 1.3:
        parts.append(
            "Objects are closer on my left side, I may be drifting left. "
            "I should steer slightly right to stay centered in my lane."
        )
    elif asym_ratio < 0.7:
        parts.append(
            "Objects are closer on my right side, I may be drifting right. "
            "I should steer slightly left to stay centered in my lane."
        )
    else:
        parts.append("I am well-centered in my lane.")

    # --- 2. Forward obstacle detection ---
    # Immediate area (bottom center)
    immediate = depth_np[3*h//4:, 2*w//5:3*w//5]
    imm_max = float(immediate.max())
    # Far road (upper center)
    far_road = depth_np[h//3:h//2, w//3:2*w//3]
    far_mean = float(far_road.mean())

    if imm_max > far_mean * 2.0:
        parts.append(
            "A vehicle or obstacle is directly ahead at close range. "
            "I should slow down and keep safe distance."
        )
    else:
        parts.append(
            "The road ahead is clear. "
            "I should maintain speed and drive forward smoothly."
        )

    # --- 3. Road depth ---
    center_depth = float(np.median(depth_np[h//3:2*h//3, w//3:2*w//3]))
    overall_median = float(np.median(depth_np))
    if center_depth < overall_median * 0.8:
        parts.append("The road extends far into the distance.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: AlpamayoR1,
    frames_np: np.ndarray,   # (N, 3, H, W) uint8
    ego_xyz_np: np.ndarray,  # (1, 1, 16, 3)
    ego_rot_np: np.ndarray,  # (1, 1, 16, 3, 3)
    nav_text: str,
    depth_info: str = "",
) -> np.ndarray:              # (64, 3)

    img_tensor = torch.from_numpy(frames_np).to(DEVICE)  # (4, 3, H, W) uint8

    ego_xyz = torch.tensor(ego_xyz_np, dtype=DTYPE, device=DEVICE)
    ego_rot = torch.tensor(ego_rot_np, dtype=DTYPE, device=DEVICE)

    messages = create_message(img_tensor)
    # Inject navigation text into the user prompt (same as run17)
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            msg["content"] = nav_text + " " + content
        else:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = nav_text + " " + item["text"]
                    break

    # Inject depth info as CoT prefix if available
    if depth_info:
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = "<|cot_start|>" + depth_info
                    break

    tokenized = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move tokenized data to device. Integer tensors (input_ids, attention_mask,
    # token_type_ids, etc.) must stay as Long — only float tensors are cast to DTYPE.
    integer_keys = {"input_ids", "attention_mask", "token_type_ids",
                    "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor):
            if k in integer_keys or not v.is_floating_point():
                tokenized_data[k] = v.long().to(device=DEVICE)
            else:
                tokenized_data[k] = v.to(device=DEVICE, dtype=DTYPE)
        else:
            tokenized_data[k] = v

    data = {
        "tokenized_data":  tokenized_data,
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }

    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        pred_xyz, _pred_rot, extra = \
            model.sample_trajectories_from_data_with_vlm_rollout(
                data,
                num_traj_samples = N_TRAJ_SAMPLES,
                num_traj_sets    = 1,
                return_extra     = True,
                top_p            = 0.98,
                temperature      = 0.6,
                max_new_tokens   = 256,
            )

    # Log CoT reasoning
    cot_texts = extra.get("cot", None)
    cot_str_full = ""
    if cot_texts is not None and len(cot_texts) > 0:
        cot_str_full = str(cot_texts[0])
        print(f"[InferServer] CoT: {cot_str_full[:200]}")

    # pred_xyz: (1, 1, S, 64, 3) → nav-based rule selection
    candidates = pred_xyz[0, 0].cpu().float().numpy()  # (S, 64, 3)
    if candidates.shape[0] == 1:
        return candidates[0], cot_str_full

    # Score based on navigation command:
    #   straight/follow lane → min lateral deviation (proven in run17)
    #   turn left  → prefer positive y (leftward)
    #   turn right → prefer negative y (rightward)
    lat_mean = candidates[:, :20, 1].mean(axis=1)  # mean y of first 20 waypoints

    if "Turn left" in nav_text or "left" in nav_text.lower():
        best = int(np.argmax(lat_mean))   # most leftward
        print(f"[InferServer] NAV=LEFT  → picked sample {best} (y_mean={lat_mean[best]:.2f})")
    elif "Turn right" in nav_text or "right" in nav_text.lower():
        best = int(np.argmin(lat_mean))   # most rightward
        print(f"[InferServer] NAV=RIGHT → picked sample {best} (y_mean={lat_mean[best]:.2f})")
    else:
        # straight / follow lane → min absolute lateral deviation (run17 proven)
        best = int(np.argmin(np.abs(lat_mean)))
        print(f"[InferServer] NAV=STRAIGHT → picked sample {best} (|y|={abs(lat_mean[best]):.2f})")

    return candidates[best], cot_str_full


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------
def serve(model_name: str, zmq_port: int, drop_layers_json: str | None = None) -> None:
    model = load_model(model_name, drop_layers_json=drop_layers_json)

    # Load Depth Anything V2 if enabled
    depth_model, depth_processor = None, None
    if USE_DEPTH:
        depth_model, depth_processor = load_depth_model()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{zmq_port}")
    print(f"[InferServer] Listening on tcp://127.0.0.1:{zmq_port}")
    print(f"[InferServer] Depth estimation: {'ON' if USE_DEPTH else 'OFF'}")
    print("[InferServer] Ready to accept requests.")

    req_count = 0
    while True:
        raw = sock.recv()
        t0 = time.time()
        try:
            req = json.loads(raw)
            frames_np  = np.array(req["frames"],  dtype=np.uint8)   # (N,3,H,W)
            ego_xyz_np = np.array(req["ego_xyz"], dtype=np.float32) # (1,1,16,3)
            ego_rot_np = np.array(req["ego_rot"], dtype=np.float32) # (1,1,16,3,3)
            nav_text   = req.get("nav", "Follow the road ahead.")

            # Depth estimation on front-wide camera (index 1, frames 4-7)
            depth_info = ""
            if depth_model is not None and len(frames_np) >= 8:
                front_frame = frames_np[7]  # latest front-wide frame (cam1, frame3)
                depth_info = estimate_depth_info(depth_model, depth_processor, front_frame)

            waypoints, cot_text = run_inference(model, frames_np, ego_xyz_np, ego_rot_np, nav_text, depth_info)
            resp = json.dumps({"ok": True, "waypoints": waypoints.tolist(), "cot": cot_text})
            req_count += 1
            dt = time.time() - t0
            depth_tag = f" D:[{depth_info[:40]}]" if depth_info else ""
            print(f"[InferServer] #{req_count:04d}  {dt:.2f}s  "
                  f"wp[0]=({waypoints[0,0]:.2f},{waypoints[0,1]:.2f}){depth_tag}")
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[InferServer] ERROR:\n{tb}")
            resp = json.dumps({"ok": False, "error": tb[:500]})

        sock.send(resp.encode())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Alpamayo ZMQ Inference Server")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="ZMQ REP port (default: 5556)")
    parser.add_argument("--drop_layers_json", default=None,
                        help="Path to pruning_meta.json; prunes at runtime "
                             "(bypasses broken save_pretrained path).")
    args = parser.parse_args()

    serve(args.model, args.port, drop_layers_json=args.drop_layers_json)
