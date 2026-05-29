"""R1 version of eval_zeroshot_alignment.py.

Same methodology — applies VLM-only identity bypass per policy meta,
runs nuScenes val L2 metric AND CoT-Action alignment match rate.
Differences: AlpamayoR1 import, R1 helper (no nav_text, processor on model).
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(r1=True)

import argparse
import json
import time

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.helper import create_message
from sft_phase_c import apply_vlm_only_prune
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    PlanningMetric, get_past_history, extract_front_cams,
    get_gt_future, get_agent_boxes_in_ego,
    alpamayo_to_nuscenes_traj, NUSC_ROOT, VERSION,
)
from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


INTEGER_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}


def run_inference_full(model, frames_np, hist_xyz, hist_rot, device):
    frames = torch.from_numpy(frames_np).to(device)
    messages = create_message(frames)
    tok = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    td = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            if k in INTEGER_KEYS or not v.is_floating_point():
                td[k] = v.long().to(device)
            else:
                td[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            td[k] = v
    ego_xyz = torch.tensor(hist_xyz[None, None], dtype=torch.bfloat16, device=device)
    ego_rot = torch.tensor(hist_rot[None, None], dtype=torch.bfloat16, device=device)
    data = {"tokenized_data": td, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data, num_traj_samples=1, num_traj_sets=1,
            return_extra=True, top_p=0.98, temperature=0.6, max_new_tokens=256,
        )
    action64 = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
    cot_text = str(extra["cot"][0, 0, 0])
    return cot_text, action64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default=str(ALPAMAYO_R1_WEIGHTS))
    ap.add_argument("--policy_meta", default=None)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[load] {args.orig_weights}", flush=True)
    model = AlpamayoR1.from_pretrained(args.orig_weights, dtype=torch.bfloat16).to(args.device)
    if args.policy_meta:
        drop_idx = apply_vlm_only_prune(model, args.policy_meta)
        print(f"[prune] applied {len(drop_idx)} layer bypasses from {args.policy_meta}", flush=True)
    else:
        drop_idx = []
    model.eval()

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
    print(f"[eval] {len(samples)} val tokens, stride={stride}", flush=True)

    metric = PlanningMetric()
    rows = []
    t0 = time.time()
    align_sum = 0
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            cot, action64 = run_inference_full(model, frames, hist, hist_rot, args.device)
            cot_label = classify_cot_rule(cot)
            action_label = classify_traj_4class(action64.tolist())
            match = alignment_match(cot_label, action_label)
            align_sum += match
            pred = alpamayo_to_nuscenes_traj(action64)
            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)
            l2 = metric.l2(pred, gt)
            col = metric.collision(pred, boxes)
            rows.append({
                "sample_token": tok, **l2, **col,
                "cot_label": cot_label, "action_label": action_label, "match": match,
            })
            if (i + 1) % 20 == 0:
                avg_l2 = np.mean([(r["L2_1s"]+r["L2_2s"]+r["L2_3s"])/3 for r in rows])
                align = align_sum / (i + 1)
                print(f"  [{i+1}/{len(samples)}] avg_L2={avg_l2:.3f}m  "
                      f"align={align:.3f}  elapsed={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [skip {tok[:8]}] {type(e).__name__}: {e}", flush=True)

    n = len(rows)
    if n == 0:
        print("[ERR] no successful samples"); return
    l2 = [sum(r[k] for r in rows)/n for k in ["L2_1s","L2_2s","L2_3s"]]
    col = [sum(r[k] for r in rows)/n*100 for k in ["Col_1s","Col_2s","Col_3s"]]
    align = sum(r["match"] for r in rows) / n

    summary = {
        "policy_meta": args.policy_meta, "n_dropped": len(drop_idx),
        "dropped_layers": drop_idx, "n_samples": n,
        "L2_1s": l2[0], "L2_2s": l2[1], "L2_3s": l2[2], "L2_avg": sum(l2)/3,
        "Col_1s": col[0], "Col_2s": col[1], "Col_3s": col[2],
        "alignment_match_rate": align, "rows": rows,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f)
    print(f"\n=== {args.policy_meta or 'R1 baseline'} (n={n}, drop={len(drop_idx)}) ===")
    print(f"L2 1s/2s/3s:  {l2[0]:.3f} / {l2[1]:.3f} / {l2[2]:.3f}  (avg {sum(l2)/3:.3f})")
    print(f"Col 1s/2s/3s: {col[0]:.2f} / {col[1]:.2f} / {col[2]:.2f} %")
    print(f"Alignment:    {align:.3f}")
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
