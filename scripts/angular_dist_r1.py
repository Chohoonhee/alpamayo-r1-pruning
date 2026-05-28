
"""Angular-distance layer importance scoring for Alpamayo R1.

For each of the 36 Qwen3VL text layers computes:
    score_ℓ = 1 - cosine_sim(h_ℓ_in, h_ℓ_out)   (averaged over tokens & samples)

Low score → layer barely transforms its input → safe to drop.

Usage:
    conda run -n alpamayo_b2d python angular_dist_r1.py \
        --weights $ALPAMAYO_R1_WEIGHTS \
        --n_samples 100 \
        --out angular_scores_r1.json
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_R1_WEIGHTS,
    NUSC_ROOT,
    OUTPUTS_DIR,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(r1=True)  # was: sys.path.insert(R1 src)
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ── nuScenes imports ───────────────────────────────────────────────────────────
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

NUSC_ROOT = str(NUSC_ROOT)
VERSION = "v1.0-trainval"
DEVICE = "cuda:0"
N_LAYERS = 36


# ── Data helpers (from nuscenes_zero_shot.py) ─────────────────────────────────

def get_ego_in_world(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ep["translation"]), Quaternion(ep["rotation"])


def world_to_ego(pos_world, ego_trans, ego_rot):
    return ego_rot.inverse.rotate(pos_world - ego_trans)


def get_past_history(nusc, sample_token, n_hist=16, target_hz=10, nusc_hz=2):
    ego_t0_trans, ego_t0_rot = get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]
    sparse_xyz, sparse_yaw, sparse_t = [], [], []
    cur = nusc.get("sample", sample_token)
    dt_nusc = 1.0 / nusc_hz
    n_back = int(math.ceil((n_hist / target_hz) / dt_nusc)) + 2
    for k in range(n_back):
        tr, q = get_ego_in_world(nusc, cur["token"])
        ego_xyz = world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
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
    dt_target = 1.0 / target_hz
    tgt_t = np.arange(-(n_hist - 1), 1) * dt_target
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
    return np.stack(frames)  # (12, 3, H, W)


def build_model_inputs(model, processor, frames_np, hist_xyz, hist_rot, nav_text):
    """Prepare model inputs exactly as alpamayo_server.py does."""
    camera_indices = torch.tensor([0]*4 + [1]*4 + [2]*4, dtype=torch.long)
    frames_t = torch.from_numpy(frames_np)  # (12, 3, H, W)

    try:
        messages = helper.create_message(frames_t, camera_indices=camera_indices, nav_text=nav_text)
    except TypeError:
        try:
            messages = helper.create_message(frames_t, camera_indices=camera_indices)
        except TypeError:
            messages = helper.create_message(frames_t)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )

    ego_xyz = torch.from_numpy(hist_xyz)[None, None].float()   # (1,1,16,3)
    ego_rot = torch.from_numpy(hist_rot)[None, None].float()   # (1,1,16,3,3)

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }
    return helper.to_device(model_inputs, DEVICE)


# ── Angular distance hook logic ────────────────────────────────────────────────

class LayerHook:
    """Captures pre/post hidden states for one decoder layer."""
    def __init__(self):
        self.h_in: torch.Tensor | None = None
        self.h_out: torch.Tensor | None = None
        self._handle = None

    def register(self, layer_module):
        def _hook(module, args, output):
            # args[0] is the hidden state input to the layer
            self.h_in = args[0].detach().float().cpu()
            # output is (hidden_state, ...) or just hidden_state
            h_out = output[0] if isinstance(output, tuple) else output
            self.h_out = h_out.detach().float().cpu()
        self._handle = layer_module.register_forward_hook(_hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


def angular_distance_batch(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """Mean angular distance over [batch, seq, hidden] tensors."""
    # Flatten to (N, D)
    h_in_flat = h_in.reshape(-1, h_in.shape[-1])
    h_out_flat = h_out.reshape(-1, h_out.shape[-1])
    cos_sim = torch.nn.functional.cosine_similarity(h_in_flat, h_out_flat, dim=-1)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    return float((1.0 - cos_sim).mean())


def score_layers(model, processor, samples_data: list[dict],
                 n_samples: int) -> list[float]:
    """Run n_samples forward passes, return per-layer angular distances."""
    # Access the 36 text decoder layers
    layers = model.vlm.language_model.layers  # ModuleList, len=36
    assert len(layers) == N_LAYERS, f"Expected {N_LAYERS} layers, got {len(layers)}"

    hooks = [LayerHook().register(layer) for layer in layers]
    accum = [0.0] * N_LAYERS
    count = 0

    try:
        for i, sample in enumerate(samples_data[:n_samples]):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_samples}] layer scores so far (first 5): "
                      f"{[f'{accum[j]/(count or 1):.4f}' for j in range(5)]}", flush=True)
            try:
                model_inputs = build_model_inputs(
                    model, processor,
                    sample["frames"], sample["hist_xyz"], sample["hist_rot"],
                    sample["nav_text"],
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Only need the encoder forward pass (no generation needed)
                    tokenized = model_inputs["tokenized_data"]
                    ego_xyz = model_inputs["ego_history_xyz"]
                    ego_rot = model_inputs["ego_history_rot"]
                    # Call the VLM forward directly for efficiency (no trajectory sampling)
                    _ = model.vlm(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized.get("attention_mask"),
                        pixel_values=tokenized.get("pixel_values"),
                        pixel_values_videos=tokenized.get("pixel_values_videos"),
                        image_grid_thw=tokenized.get("image_grid_thw"),
                        video_grid_thw=tokenized.get("video_grid_thw"),
                    )
                for j in range(N_LAYERS):
                    if hooks[j].h_in is not None and hooks[j].h_out is not None:
                        accum[j] += angular_distance_batch(hooks[j].h_in, hooks[j].h_out)
                count += 1
            except Exception as e:
                print(f"  [skip sample {i}] {e}", flush=True)
                continue
    finally:
        for h in hooks:
            h.remove()

    if count == 0:
        raise RuntimeError("No samples processed successfully")
    scores = [accum[j] / count for j in range(N_LAYERS)]
    return scores


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ALPAMAYO_R1_WEIGHTS))
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out", default=str(OUTPUTS_DIR / 'angular_scores_r1.json'))
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
    DEVICE = args.device

    print(f"Loading nuScenes {VERSION} ...", flush=True)
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
    print(f"Val samples: {len(val_tokens)}", flush=True)

    # Stride-sample for broad coverage
    n = args.n_samples
    stride = max(1, len(val_tokens) // n)
    selected = val_tokens[::stride][:n]
    print(f"Loading {len(selected)} samples for scoring ...", flush=True)

    samples_data = []
    for tok in selected:
        try:
            frames = extract_front_cams(nusc, tok)
            hist_xyz, hist_rot = get_past_history(nusc, tok)
            # Simple nav: straight ahead (avoid complex GT-based detection for efficiency)
            nav_text = "Continue straight ahead."
            samples_data.append({
                "frames": frames,
                "hist_xyz": hist_xyz,
                "hist_rot": hist_rot,
                "nav_text": nav_text,
            })
        except Exception as e:
            print(f"  [skip load {tok[:8]}] {e}", flush=True)
    print(f"Prepared {len(samples_data)} samples", flush=True)

    print(f"\nLoading model from {args.weights} ...", flush=True)
    model = AlpamayoR1.from_pretrained(args.weights, dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.", flush=True)

    print(f"\nScoring {N_LAYERS} layers on {len(samples_data)} samples ...", flush=True)
    scores = score_layers(model, processor, samples_data, n_samples=len(samples_data))

    # Sort by score (ascending = least important first)
    ranked = sorted(range(N_LAYERS), key=lambda i: scores[i])

    result = {
        "scores": scores,
        "ranked_layers": ranked,   # index 0 = least important
        "n_samples": len(samples_data),
        "weights": args.weights,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}", flush=True)

    print("\n── Layer scores (layer: score) ──────────────────────────────────")
    for j, score in enumerate(scores):
        print(f"  Layer {j:2d}: {score:.6f}")
    print("\n── Bottom-13 layers (drop candidates) ──────────────────────────")
    for rank, layer_idx in enumerate(ranked[:13]):
        print(f"  rank {rank+1:2d}: layer {layer_idx:2d}  score={scores[layer_idx]:.6f}")


if __name__ == "__main__":
    main()
