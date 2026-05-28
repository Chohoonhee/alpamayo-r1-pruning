
"""Layer coupling probe for Alpamayo VLA (R1 or 1.5).

Tests the hypothesis: in Alpamayo architecture, VLM layer k and Expert layer k
form a 1-to-1 coupled pair (expert layer k cross-attends to VLM layer k's KV).

Protocol:
  For each layer index k in [0, n_layers):
    - Run inference with VLM[k] identity-bypassed       → L2_vlm[k]
    - Run inference with expert[k] identity-bypassed    → L2_expert[k]
    - Run inference with both VLM[k] and expert[k] bypassed → L2_both[k]
  Plot: (L2_vlm, L2_expert, L2_both) per layer index.

If the pair is coupled/redundant: L2_both[k] ≈ max(L2_vlm[k], L2_expert[k]).
If independent: L2_both[k] ≈ L2_vlm[k] + L2_expert[k] (additive).

Runs a small nuScenes subset (N=10 default) to keep it fast.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    ALPAMAYO_R1_WEIGHTS,
    NUSC_ROOT,
    OUTPUTS_DIR,
    add_alpamayo_to_syspath,
)
import argparse
import json
import os
import sys
import math

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

NUSC_ROOT = str(NUSC_ROOT)
VERSION = "v1.0-trainval"


# ── Data helpers ─────────────────────────────────────────────────────────────

def _get_ego_in_world(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ep["translation"]), Quaternion(ep["rotation"])

def _world_to_ego(pos_world, ego_trans, ego_rot):
    return ego_rot.inverse.rotate(pos_world - ego_trans)

def get_past_history(nusc, sample_token, n_hist=16, target_hz=10, nusc_hz=2):
    ego_t0_trans, ego_t0_rot = _get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]
    sparse_xyz, sparse_yaw, sparse_t = [], [], []
    cur = nusc.get("sample", sample_token)
    dt_nusc = 1.0 / nusc_hz
    n_back = int(math.ceil((n_hist / target_hz) / dt_nusc)) + 2
    for k in range(n_back):
        tr, q = _get_ego_in_world(nusc, cur["token"])
        ego_xyz = _world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        rel_yaw = q.yaw_pitch_roll[0] - t0_yaw
        sparse_xyz.insert(0, [ego_xyz[0], ego_xyz[1], 0.0])
        sparse_yaw.insert(0, rel_yaw)
        sparse_t.insert(0, -k * dt_nusc)
        if not cur["prev"]:
            break
        cur = nusc.get("sample", cur["prev"])
    sparse_xyz = np.array(sparse_xyz, dtype=np.float64)
    sparse_yaw = np.array(sparse_yaw, dtype=np.float64)
    sparse_t = np.array(sparse_t, dtype=np.float64)
    tgt_t = np.arange(-(n_hist - 1), 1) * (1.0 / target_hz)
    out_xyz = np.zeros((n_hist, 3), dtype=np.float32)
    for dim in range(3):
        out_xyz[:, dim] = np.interp(tgt_t, sparse_t, sparse_xyz[:, dim],
                                    left=sparse_xyz[0, dim], right=sparse_xyz[-1, dim])
    out_xyz[-1] = [0, 0, 0]
    yaw_interp = np.interp(tgt_t, sparse_t, sparse_yaw,
                           left=sparse_yaw[0], right=sparse_yaw[-1])
    out_rot = np.zeros((n_hist, 3, 3), dtype=np.float32)
    for i in range(n_hist):
        y = float(yaw_interp[i])
        out_rot[i] = [[math.cos(y), -math.sin(y), 0],
                      [math.sin(y),  math.cos(y), 0],
                      [0, 0, 1]]
    out_rot[-1] = np.eye(3)
    return out_xyz, out_rot

def extract_front_cams(nusc, sample_token, resize=(512, 320), n_temporal=4):
    sample = nusc.get("sample", sample_token)
    cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    frames = []
    for cn in cam_names:
        cur_sd = nusc.get("sample_data", sample["data"][cn])
        cam_frames = []
        for _ in range(n_temporal):
            img_path = os.path.join(nusc.dataroot, cur_sd["filename"])
            arr = np.array(Image.open(img_path).resize(resize)).transpose(2, 0, 1).astype(np.uint8)
            cam_frames.insert(0, arr)
            if cur_sd["prev"]:
                cur_sd = nusc.get("sample_data", cur_sd["prev"])
        while len(cam_frames) < n_temporal:
            cam_frames.insert(0, cam_frames[0].copy())
        frames.extend(cam_frames)
    return np.stack(frames)

def get_gt_future(nusc, sample_token, n_future=6, dt=0.5):
    ego_t0_trans, ego_t0_rot = _get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]
    pts = []
    cur = nusc.get("sample", sample_token)
    for _ in range(n_future):
        if not cur["next"]: break
        cur = nusc.get("sample", cur["next"])
        tr, _ = _get_ego_in_world(nusc, cur["token"])
        pos = _world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        pts.append([pos[0], pos[1]])
    while len(pts) < n_future:
        pts.append(pts[-1] if pts else [0, 0])
    return np.array(pts)


# ── Inference helpers (variant-aware) ─────────────────────────────────────────

def load_model_and_helper(variant, weights_path, device):
    if variant == "1.5":
        add_alpamayo_to_syspath(v15=True)  # was: sys.path.insert(1.5 src)
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5 as ModelCls
        from alpamayo1_5 import helper as H
    else:  # r1
        add_alpamayo_to_syspath(r1=True)  # was: sys.path.insert(R1 src)
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1 as ModelCls
        from alpamayo_r1 import helper as H
    model = ModelCls.from_pretrained(weights_path, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = H.get_processor(model.tokenizer)
    return model, H, processor


def run_waypoints(model, helper_mod, processor, sample, device, variant):
    """Return (2,) waypoint at horizon 3s for this sample using the model.

    Simple: take (front_left, front, front_right) + 4 temporal, ego history, nav text,
    run 1 trajectory sample, take waypoint at 3s (index 29 of 64 @10Hz).
    """
    frames = torch.from_numpy(sample["frames"])  # (12, 3, H, W)
    n_cam = 3
    camera_indices = torch.tensor([0, 1, 2], dtype=torch.long) if variant == "1.5" else None

    try:
        if variant == "1.5":
            messages = helper_mod.create_message(
                frames=frames, camera_indices=camera_indices,
                nav_text="Follow the road ahead.",
            )
        else:
            try:
                messages = helper_mod.create_message(frames, camera_indices=camera_indices)
            except TypeError:
                messages = helper_mod.create_message(frames)
    except TypeError:
        messages = helper_mod.create_message(frames)

    tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )

    # Mirror v15 server: manually preserve integer dtypes (to_device helper
    # coerces attention_mask to float which breaks get_rope_index).
    integer_keys = {"input_ids", "attention_mask", "token_type_ids",
                    "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor):
            if k in integer_keys or not v.is_floating_point():
                tokenized_data[k] = v.long().to(device)
            else:
                tokenized_data[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            tokenized_data[k] = v

    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": torch.from_numpy(sample["hist_xyz"])[None, None].float().to(device),
        "ego_history_rot": torch.from_numpy(sample["hist_rot"])[None, None].float().to(device),
    }

    with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=96,
            return_extra=False,
        )
    pred_xyz = out[0].detach().float().cpu().numpy()
    pred_xyz_flat = pred_xyz.reshape(-1, pred_xyz.shape[-2], pred_xyz.shape[-1])
    way64 = pred_xyz_flat[0]   # (64, 3)
    # 2s, 3s waypoints (indices 19, 29 at 10Hz)
    return way64[19, :2], way64[29, :2]


# ── Layer bypass context ──────────────────────────────────────────────────────

class BypassLayer:
    """Monkey-patch a layer's forward to identity; restore on exit."""
    def __init__(self, layer):
        self.layer = layer
        self.orig = layer.forward
    def __enter__(self):
        def _identity(hidden_states, *args, **kwargs):
            # Qwen3VL decoder layer returns a single hidden_states tensor (when
            # output_attentions / output_hidden_states are False).
            return hidden_states
        self.layer.forward = _identity
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.layer.forward = self.orig


# ── Main probe ────────────────────────────────────────────────────────────────

def probe(variant, weights, n_samples, device, out_path):
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val_scene_names = set(splits.get("val", []))
    val_tokens = []
    for scene in nusc.scene:
        if scene["name"] in val_scene_names:
            tok = scene["first_sample_token"]
            while tok:
                val_tokens.append(tok)
                tok = nusc.get("sample", tok)["next"]
    stride = max(1, len(val_tokens) // n_samples)
    selected = val_tokens[::stride][:n_samples]
    print(f"Probing {variant} on {len(selected)} nuScenes samples")

    # Prepare sample data once
    samples = []
    for tok in selected:
        try:
            frames = extract_front_cams(nusc, tok)
            hist_xyz, hist_rot = get_past_history(nusc, tok)
            gt = get_gt_future(nusc, tok)
            samples.append({
                "tok": tok, "frames": frames,
                "hist_xyz": hist_xyz, "hist_rot": hist_rot,
                "gt_3s": gt[-1],   # last of 6 @ 2Hz = 3s
                "gt_2s": gt[3],
            })
        except Exception as e:
            print(f"skip {tok[:8]}: {e}")
    print(f"Prepared {len(samples)} samples")

    model, helper_mod, processor = load_model_and_helper(variant, weights, device)
    vlm_layers = model.vlm.language_model.layers
    expert_layers = model.expert.layers
    n_layers = len(vlm_layers)
    assert len(expert_layers) == n_layers
    print(f"VLM/Expert both have {n_layers} layers")

    # --- Stage 1: unperturbed baseline ---
    base_preds_2s = []
    base_preds_3s = []
    for i, s in enumerate(samples):
        p2, p3 = run_waypoints(model, helper_mod, processor, s, device, variant)
        base_preds_2s.append(p2)
        base_preds_3s.append(p3)
        print(f"  [base {i+1}/{len(samples)}] pred_3s=({p3[0]:.2f},{p3[1]:.2f})", flush=True)
    base_preds_2s = np.stack(base_preds_2s)
    base_preds_3s = np.stack(base_preds_3s)

    # --- Stage 2: per-layer ablation ---
    results = {
        "variant": variant,
        "n_layers": n_layers,
        "n_samples": len(samples),
        "base_pred_2s": base_preds_2s.tolist(),
        "base_pred_3s": base_preds_3s.tolist(),
        "l2_vlm_only": [],      # VLM[k] ablated
        "l2_expert_only": [],   # Expert[k] ablated
        "l2_both": [],          # Both ablated
    }

    # Skip layers known to cause KV-cache crashes on identity bypass
    # (layer 0: causes position_id issues; layer n-1: attention to incomplete cache)
    CRASH_LAYERS = {0, n_layers - 1}
    print(f"Skipping ablation on critical layers: {sorted(CRASH_LAYERS)} "
          f"(angular score very high → not drop candidates anyway)")

    for k in range(n_layers):
        if k in CRASH_LAYERS:
            print(f"  [layer {k:2d}] skipped (critical)")
            continue
        for mode in ["vlm_only", "expert_only", "both"]:
            deltas_2s = []
            deltas_3s = []
            n_ok = 0
            for i, s in enumerate(samples):
                cms = []
                if mode in ("vlm_only", "both"):
                    cms.append(BypassLayer(vlm_layers[k]))
                if mode in ("expert_only", "both"):
                    cms.append(BypassLayer(expert_layers[k]))
                # nested contexts with per-sample failure handling
                try:
                    for cm in cms: cm.__enter__()
                    p2, p3 = run_waypoints(model, helper_mod, processor, s, device, variant)
                    d3 = float(np.linalg.norm(p3 - base_preds_3s[i]))
                    d2 = float(np.linalg.norm(p2 - base_preds_2s[i]))
                    deltas_2s.append(d2)
                    deltas_3s.append(d3)
                    n_ok += 1
                except Exception as e:
                    print(f"    sample {i} failed on layer {k}/{mode}: "
                          f"{type(e).__name__}", flush=True)
                finally:
                    for cm in reversed(cms): cm.__exit__(None, None, None)
                    torch.cuda.empty_cache()
            if n_ok == 0:
                mean_3s = float("nan")
                mean_2s = float("nan")
            else:
                mean_3s = float(np.mean(deltas_3s))
                mean_2s = float(np.mean(deltas_2s))
            key = f"l2_{mode}"
            results[key].append({"layer": k, "mean_d3s": mean_3s,
                                 "mean_d2s": mean_2s, "n_ok": n_ok})
            print(f"  [layer {k:2d} / {mode:11s}] Δ3s={mean_3s:.4f} m "
                  f"(n_ok={n_ok}/{len(samples)})", flush=True)
        # Incremental save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["r1", "1.5"], required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.weights is None:
        args.weights = (
            str(ALPAMAYO_15_WEIGHTS) if args.variant == "1.5"
            else str(ALPAMAYO_R1_WEIGHTS)
        )
    if args.out is None:
        args.out = str(OUTPUTS_DIR / f'coupling_probe_{args.variant}.json')

    probe(args.variant, args.weights, args.n_samples, args.device, args.out)


if __name__ == "__main__":
    main()
