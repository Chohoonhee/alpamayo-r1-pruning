"""Evaluate Phase B SFT'd LoRA model on nuScenes.

Properly loads: base 1.5 → runtime prune → re-inject LoRA → load saved state.
Then runs inference (same as nuscenes_zero_shot.py but in-process).
"""
from __future__ import annotations
import argparse, os, sys, json, math, time

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from safetensors.torch import load_file

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from sft_phase_b import apply_joint_prune, apply_lora_both
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, "/home/irteam/ws/vipe_test")
from nuscenes_zero_shot import (
    PlanningMetric, get_ego_in_world, get_gt_future, get_past_history,
    extract_front_cams, get_agent_boxes_in_ego, get_nav_text,
    alpamayo_to_nuscenes_traj, NUSC_ROOT, VERSION,
)


def load_trained_model(orig_weights, drop_pairs_json, lora_checkpoint,
                        lora_r, lora_alpha, device):
    """Load base → prune → re-inject LoRA → load saved state dict."""
    print(f"[load] base from {orig_weights}", flush=True)
    m = Alpamayo1_5.from_pretrained(orig_weights, dtype=torch.bfloat16).to(device)
    print(f"[load] applying runtime prune from {drop_pairs_json}", flush=True)
    drop_idx = apply_joint_prune(m, drop_pairs_json)
    for p in m.parameters(): p.requires_grad = False
    print(f"[load] injecting LoRA (r={lora_r}, a={lora_alpha})", flush=True)
    m = apply_lora_both(m, drop_idx, r=lora_r, alpha=lora_alpha)

    # Collect saved state dict from all safetensors shards
    print(f"[load] loading state dict from {lora_checkpoint}", flush=True)
    sd = {}
    import glob
    for shard in sorted(glob.glob(f"{lora_checkpoint}/model-*.safetensors")):
        sd.update(load_file(shard, device=device))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[load] first 3 missing: {missing[:3]}")
    if unexpected:
        print(f"[load] first 3 unexpected: {unexpected[:3]}")
    m.eval()
    return m


def run_inference_trajectory(model, processor, frames_np, ego_xyz, ego_rot,
                              nav_text, cam_idx_np, device):
    """Run 1.5 inference, return 64-waypoint trajectory."""
    frames = torch.from_numpy(frames_np).to(device)
    cam_idx = torch.tensor(cam_idx_np, dtype=torch.long).to(device) if cam_idx_np is not None else None
    messages = helper.create_message(
        frames=frames, camera_indices=cam_idx, nav_text=nav_text,
    )
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    int_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            if k in int_keys or not v.is_floating_point():
                tokenized_data[k] = v.long().to(device)
            else:
                tokenized_data[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            tokenized_data[k] = v
    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": torch.from_numpy(ego_xyz).float().to(device),
        "ego_history_rot": torch.from_numpy(ego_rot).float().to(device),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=96, return_extra=False,
        )
    pred_xyz = out[0].detach().float().cpu().numpy()
    pred_xyz_flat = pred_xyz.reshape(-1, pred_xyz.shape[-2], pred_xyz.shape[-1])
    way64 = pred_xyz_flat[0, :, :2]
    return way64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_pairs_json", required=True)
    ap.add_argument("--lora_checkpoint", required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    m = load_trained_model(
        args.orig_weights, args.drop_pairs_json, args.lora_checkpoint,
        args.lora_r, args.lora_alpha, args.device,
    )
    processor = helper.get_processor(m.tokenizer)

    # nuScenes val sampling
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val_scene_names = set(splits.get("val", []))
    samples = []
    for scene in nusc.scene:
        if scene["name"] in val_scene_names:
            t = scene["first_sample_token"]
            while t:
                samples.append(t)
                t = nusc.get("sample", t)["next"]
    stride = max(1, len(samples) // args.n_samples)
    samples = samples[::stride][:args.n_samples]
    print(f"[eval] using {len(samples)} samples", flush=True)

    metric = PlanningMetric()
    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)
    results = []
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            nav = get_nav_text(nusc, tok)
            ego_xyz = hist[None, None]; ego_rot = hist_rot[None, None]
            way = run_inference_trajectory(
                m, processor, frames, ego_xyz, ego_rot, nav, cam_idx_np, args.device
            )
            pred = alpamayo_to_nuscenes_traj(way)
            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)
            l2 = metric.l2(pred, gt)
            col = metric.collision(pred, boxes)
            results.append({"sample_token": tok, **l2, **col})
            if (i + 1) % 20 == 0:
                avg_l2 = np.mean([(r["L2_1s"]+r["L2_2s"]+r["L2_3s"])/3 for r in results])
                print(f"[{i+1}/{len(samples)}]  avg L2 = {avg_l2:.3f}m", flush=True)
        except Exception as e:
            print(f"  [skip] {tok[:8]}: {type(e).__name__}: {e}", flush=True)
            continue

    if not results:
        print("NO RESULTS", flush=True); return

    n = len(results)
    l2 = [sum(r[k] for r in results)/n for k in ["L2_1s", "L2_2s", "L2_3s"]]
    col = [sum(r[k] for r in results)/n * 100 for k in ["Col_1s", "Col_2s", "Col_3s"]]
    print(f"\n=== Final over {n} samples ===")
    print(f"L2 1s/2s/3s:  {l2[0]:.3f} / {l2[1]:.3f} / {l2[2]:.3f}  (avg {sum(l2)/3:.3f})")
    print(f"Col 1s/2s/3s: {col[0]:.2f} / {col[1]:.2f} / {col[2]:.2f} %")
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
