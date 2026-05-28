
"""
Inference server for Alpamayo 1.5.
Same protocol as v1 infer server but uses alpamayo1_5 module.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(v15=True)  # was: sys.path.insert(1.5 src)

import argparse, json, os, sys, time, traceback
import numpy as np
import torch
import zmq

# Add 1.5 source path

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


DEFAULT_MODEL  = os.environ.get("ALPAMAYO_MODEL",
                                str(ALPAMAYO_15_WEIGHTS))
DEFAULT_PORT   = int(os.environ.get("INFER_PORT", "5556"))
N_TRAJ_SAMPLES = int(os.environ.get("ALPAMAYO_TRAJ_SAMPLES", "4"))
DEVICE         = os.environ.get("ALPAMAYO_DEVICE", "cuda:0")
DTYPE          = torch.bfloat16


def load_model(model_name: str, drop_layers_json: str | None = None):
    print(f"[V15 InferServer] Loading Alpamayo 1.5 from: {model_name}")
    t0 = time.time()
    model = Alpamayo1_5.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    model.eval()

    if drop_layers_json:
        import json as _json

        with open(drop_layers_json) as _f:
            meta = _json.load(_f)
        drop_idx = sorted(set(meta["dropped_layers"]))
        layers = model.vlm.language_model.layers
        n_orig = len(layers)
        print(f"[V15 InferServer] Runtime pruning (identity bypass): "
              f"drop {len(drop_idx)}/{n_orig} text layers")
        print(f"[V15 InferServer]   dropped: {drop_idx}")

        def _make_identity_forward():
            def _fwd(hidden_states, *args, **kwargs):
                return hidden_states
            return _fwd
        for idx in drop_idx:
            layers[idx].forward = _make_identity_forward()
        n_keep = n_orig - len(drop_idx)
        print(f"[V15 InferServer] After pruning: {n_keep} active text layers")

    torch.cuda.empty_cache()
    print(f"[V15 InferServer] Loaded in {time.time()-t0:.1f}s")
    return model


def run_inference(model, frames_np, ego_xyz_np, ego_rot_np, nav_text, camera_indices=None):
    """Run 1.5 inference and return (waypoints, cot_text)."""
    img_tensor = torch.from_numpy(frames_np).to(DEVICE)

    # camera_indices: default 0,1,2 for 3-camera setup
    if camera_indices is None:
        n_cam = frames_np.shape[0] // 4   # assume 4 frames per cam
        camera_indices = torch.arange(n_cam, dtype=torch.long)
    cam_idx = torch.as_tensor(camera_indices, dtype=torch.long).to(DEVICE)

    ego_xyz = torch.tensor(ego_xyz_np, dtype=DTYPE, device=DEVICE)
    ego_rot = torch.tensor(ego_rot_np, dtype=DTYPE, device=DEVICE)

    # Build message — v1.5 natively supports nav_text via <|route_start|>...<|route_end|>
    messages = helper.create_message(
        frames=img_tensor,
        camera_indices=cam_idx,
        nav_text=nav_text,   # ← properly wrapped by helper
    )

    processor = helper.get_processor(model.tokenizer)
    tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    integer_keys = {"input_ids", "attention_mask", "token_type_ids",
                    "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor):
            if k in integer_keys or not v.is_floating_point():
                tokenized_data[k] = v.long().to(DEVICE)
            else:
                tokenized_data[k] = v.to(DEVICE, dtype=DTYPE)
        else:
            tokenized_data[k] = v

    data = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }

    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        pred_xyz, _pred_rot, extra = \
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=data,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=N_TRAJ_SAMPLES,
                max_generation_length=128,
                return_extra=True,
            )

    cot_texts = extra.get("cot", None)
    cot_str_full = ""
    if cot_texts is not None and len(cot_texts) > 0:
        cot_str_full = str(cot_texts[0])
        print(f"[V15 InferServer] CoT: {cot_str_full[:200]}")

    # pred_xyz shape: (B, 1, S, 64, 3)
    candidates = pred_xyz[0, 0].cpu().float().numpy()
    if candidates.shape[0] == 1:
        return candidates[0], cot_str_full

    # Score by nav_text (same rule as v1)
    lat_mean = candidates[:, :20, 1].mean(axis=1)
    if "Turn left" in nav_text or "left" in nav_text.lower():
        best = int(np.argmax(lat_mean))
    elif "Turn right" in nav_text or "right" in nav_text.lower():
        best = int(np.argmin(lat_mean))
    else:
        best = int(np.argmin(np.abs(lat_mean)))
    return candidates[best], cot_str_full


def serve(model_name: str, zmq_port: int, drop_layers_json: str | None = None) -> None:
    model = load_model(model_name, drop_layers_json=drop_layers_json)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{zmq_port}")
    print(f"[V15 InferServer] Listening on tcp://127.0.0.1:{zmq_port}")
    print("[V15 InferServer] Ready to accept requests.")

    req_count = 0
    while True:
        raw = sock.recv()
        t0 = time.time()
        try:
            req = json.loads(raw)
            frames_np  = np.array(req["frames"],  dtype=np.uint8)
            ego_xyz_np = np.array(req["ego_xyz"], dtype=np.float32)
            ego_rot_np = np.array(req["ego_rot"], dtype=np.float32)
            nav_text   = req.get("nav", "Follow the road ahead.")
            cam_idx    = req.get("camera_indices", None)

            waypoints, cot_text = run_inference(
                model, frames_np, ego_xyz_np, ego_rot_np, nav_text, cam_idx
            )
            resp = json.dumps({
                "ok": True, "waypoints": waypoints.tolist(), "cot": cot_text
            })
            req_count += 1
            dt = time.time() - t0
            print(f"[V15 InferServer] #{req_count:04d} {dt:.2f}s "
                  f"wp[0]=({waypoints[0,0]:.2f},{waypoints[0,1]:.2f})")
        except Exception:
            tb = traceback.format_exc()
            print(f"[V15 InferServer] ERROR:\n{tb}")
            resp = json.dumps({"ok": False, "error": tb[:500]})
        sock.send(resp.encode())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--drop_layers_json", default=None,
                        help="Path to pruning_meta.json for runtime identity-bypass pruning.")
    args = parser.parse_args()
    serve(args.model, args.port, drop_layers_json=args.drop_layers_json)
