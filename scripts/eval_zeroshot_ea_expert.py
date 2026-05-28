"""Zero-shot eval: ea VLM pruning + Expert layer pruning."""
from __future__ import annotations
import argparse, sys, json, os

import numpy as np
import torch

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")
sys.path.insert(0, "/home/irteam/ws/vipe_test")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from sft_phase_c import apply_vlm_only_prune
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    PlanningMetric, get_past_history, extract_front_cams,
    get_gt_future, get_agent_boxes_in_ego, get_nav_text,
    alpamayo_to_nuscenes_traj, NUSC_ROOT, VERSION,
)
from eval_sft_lora import run_inference_trajectory


def apply_expert_prune(model, expert_scores_json, n_drop):
    with open(expert_scores_json) as f:
        meta = json.load(f)
    ranked = meta["ranked_layers"]
    drop_idx = sorted(ranked[:n_drop])

    def _id_fwd():
        def _f(hidden_states, *a, **kw):
            return hidden_states
        return _f

    for i in drop_idx:
        model.expert.layers[i].forward = _id_fwd()
    print(f"[prune] Expert drop {n_drop}/{len(model.expert.layers)}: {drop_idx}", flush=True)
    return drop_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--vlm_drop_json", required=True)
    ap.add_argument("--expert_scores_json", required=True)
    ap.add_argument("--n_expert_drop", type=int, required=True)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = Alpamayo1_5.from_pretrained(args.orig_weights, dtype=torch.bfloat16).to(args.device)
    apply_vlm_only_prune(model, args.vlm_drop_json)
    apply_expert_prune(model, args.expert_scores_json, args.n_expert_drop)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val_tokens = []
    for scene in nusc.scene:
        if scene["name"] in set(splits.get("val", [])):
            t = scene["first_sample_token"]
            while t:
                val_tokens.append(t)
                t = nusc.get("sample", t)["next"]
    stride = max(1, len(val_tokens) // args.n_samples)
    samples = val_tokens[::stride][:args.n_samples]

    metric = PlanningMetric()
    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)
    results = []
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            nav = get_nav_text(nusc, tok)
            ego_xyz = hist[None, None]; ego_rot = hist_rot[None, None]
            way = run_inference_trajectory(model, processor, frames, ego_xyz, ego_rot,
                                           nav, cam_idx_np, args.device)
            pred = alpamayo_to_nuscenes_traj(way)
            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)
            l2 = metric.l2(pred, gt); col = metric.collision(pred, boxes)
            results.append({"sample_token": tok, **l2, **col})
            if (i + 1) % 20 == 0:
                avg_l2 = np.mean([(r["L2_1s"]+r["L2_2s"]+r["L2_3s"])/3 for r in results])
                print(f"[{i+1}/{len(samples)}] avg L2={avg_l2:.3f}m", flush=True)
        except Exception as e:
            print(f"  [skip] {tok[:8]}: {e}")

    n = len(results)
    l2 = [sum(r[k] for r in results)/n for k in ["L2_1s","L2_2s","L2_3s"]]
    col = [sum(r[k] for r in results)/n*100 for k in ["Col_1s","Col_2s","Col_3s"]]
    print(f"\n=== ea_vlm28 + expert_drop{args.n_expert_drop} (n={n}) ===")
    print(f"L2 1s/2s/3s:  {l2[0]:.3f}/{l2[1]:.3f}/{l2[2]:.3f}  (avg {sum(l2)/3:.3f})")
    print(f"Col 1s/2s/3s: {col[0]:.2f}/{col[1]:.2f}/{col[2]:.2f} %")
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
