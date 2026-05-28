"""Eval Phase D: VLM-only prune + Expert-only LoRA checkpoint."""
from __future__ import annotations
import argparse, os, sys, json, glob

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")
sys.path.insert(0, "/home/irteam/ws/vipe_test")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from sft_phase_c import apply_vlm_only_prune
from sft_phase_d import apply_lora_expert_only
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    PlanningMetric, get_past_history, extract_front_cams,
    get_gt_future, get_agent_boxes_in_ego, get_nav_text,
    alpamayo_to_nuscenes_traj, NUSC_ROOT, VERSION,
)
from eval_sft_lora import run_inference_trajectory


def load_trained_model_d(orig_weights, drop_layers_json, lora_checkpoint,
                         lora_r, lora_alpha, device):
    m = Alpamayo1_5.from_pretrained(orig_weights, dtype=torch.bfloat16).to(device)
    apply_vlm_only_prune(m, drop_layers_json)
    for p in m.parameters(): p.requires_grad = False
    m = apply_lora_expert_only(m, r=lora_r, alpha=lora_alpha)
    sd = {}
    for shard in sorted(glob.glob(f"{lora_checkpoint}/model-*.safetensors")):
        sd.update(load_file(shard, device=device))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--lora_checkpoint", required=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    m = load_trained_model_d(
        args.orig_weights, args.drop_layers_json, args.lora_checkpoint,
        args.lora_r, args.lora_alpha, args.device,
    )
    processor = helper.get_processor(m.tokenizer)

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

    metric = PlanningMetric()
    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)
    results = []
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            nav = get_nav_text(nusc, tok)
            ego_xyz = hist[None, None]; ego_rot = hist_rot[None, None]
            way = run_inference_trajectory(m, processor, frames, ego_xyz, ego_rot,
                                           nav, cam_idx_np, args.device)
            pred = alpamayo_to_nuscenes_traj(way)
            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)
            l2 = metric.l2(pred, gt); col = metric.collision(pred, boxes)
            results.append({"sample_token": tok, **l2, **col})
            if (i + 1) % 20 == 0:
                avg_l2 = np.mean([(r["L2_1s"]+r["L2_2s"]+r["L2_3s"])/3 for r in results])
                print(f"[{i+1}/{len(samples)}]  avg L2 = {avg_l2:.3f}m", flush=True)
        except Exception as e:
            print(f"  [skip] {tok[:8]}: {type(e).__name__}: {e}")
            continue

    n = len(results)
    l2 = [sum(r[k] for r in results)/n for k in ["L2_1s","L2_2s","L2_3s"]]
    col = [sum(r[k] for r in results)/n * 100 for k in ["Col_1s","Col_2s","Col_3s"]]
    print(f"\n=== n={n} ===")
    print(f"L2 1s/2s/3s:  {l2[0]:.3f}/{l2[1]:.3f}/{l2[2]:.3f}  (avg {sum(l2)/3:.3f})")
    print(f"Col 1s/2s/3s: {col[0]:.2f}/{col[1]:.2f}/{col[2]:.2f} %")
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
