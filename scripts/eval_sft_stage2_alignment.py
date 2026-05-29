"""Eval Stage 2 v2: base 1.5 + VLM bypass + LoRA on Expert + L2 + alignment.

Like eval_sft_stage2.py but also reports CoT-Action alignment match rate.
For evaluating the Stage 2 v2 model trained on plus_harmful drop set.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(v15=True)

import argparse
import glob
import json
import time

import numpy as np
import torch
from safetensors.torch import load_file

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from sft_phase_c import apply_vlm_only_prune
# For Stage 2 v2 (Expert-only LoRA), use the expert-only injector
from sft_stage2_expert_only import apply_lora_vlm_expert
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    PlanningMetric, get_past_history, extract_front_cams,
    get_gt_future, get_agent_boxes_in_ego, get_nav_text,
    alpamayo_to_nuscenes_traj, NUSC_ROOT, VERSION,
)
from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


def load_model(orig_weights, drop_layers_json, lora_checkpoint, lora_r,
               lora_alpha, device):
    m = Alpamayo1_5.from_pretrained(orig_weights, dtype=torch.bfloat16).to(device)
    apply_vlm_only_prune(m, drop_layers_json)
    for p in m.parameters():
        p.requires_grad = False
    m = apply_lora_vlm_expert(m, r=lora_r, alpha=lora_alpha)
    sd = {}
    for shard in sorted(glob.glob(f"{lora_checkpoint}/model-*.safetensors")):
        sd.update(load_file(shard, device=device))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    m.eval()
    return m


def run_inference_with_cot(model, processor, frames_np, ego_xyz, ego_rot,
                            nav_text, cam_idx_np, device):
    frames = torch.from_numpy(frames_np).to(device)
    cam_idx = torch.tensor(cam_idx_np, dtype=torch.long).to(device) if cam_idx_np is not None else None
    messages = helper.create_message(frames=frames, camera_indices=cam_idx, nav_text=nav_text)
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    int_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}
    td = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            if k in int_keys or not v.is_floating_point():
                td[k] = v.long().to(device)
            else:
                td[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            td[k] = v
    data = {"tokenized_data": td,
            "ego_history_xyz": torch.from_numpy(ego_xyz).float().to(device),
            "ego_history_rot": torch.from_numpy(ego_rot).float().to(device)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True,
        )
    action64 = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
    cot_text = str(extra["cot"][0, 0, 0])
    return cot_text, action64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--lora_checkpoint", required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    m = load_model(args.orig_weights, args.drop_layers_json, args.lora_checkpoint,
                   args.lora_r, args.lora_alpha, args.device)
    processor = helper.get_processor(m.tokenizer)

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
    rows = []
    t0 = time.time()
    align_sum = 0
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            nav = get_nav_text(nusc, tok)
            ego_xyz = hist[None, None]; ego_rot = hist_rot[None, None]
            cot, action64 = run_inference_with_cot(
                m, processor, frames, ego_xyz, ego_rot, nav, cam_idx_np, args.device,
            )
            cot_label = classify_cot_rule(cot)
            action_label = classify_traj_4class(action64.tolist())
            match = alignment_match(cot_label, action_label)
            align_sum += match
            pred = alpamayo_to_nuscenes_traj(action64)
            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)
            l2 = metric.l2(pred, gt); col = metric.collision(pred, boxes)
            rows.append({"sample_token": tok, **l2, **col,
                         "cot_label": cot_label, "action_label": action_label,
                         "match": match})
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
        "drop_layers_json": args.drop_layers_json,
        "lora_checkpoint": args.lora_checkpoint,
        "n_samples": n,
        "L2_1s": l2[0], "L2_2s": l2[1], "L2_3s": l2[2], "L2_avg": sum(l2)/3,
        "Col_1s": col[0], "Col_2s": col[1], "Col_3s": col[2],
        "alignment_match_rate": align,
        "rows": rows,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f)
    print(f"\n=== Stage 2 v2 (n={n}) ===")
    print(f"L2 1s/2s/3s:  {l2[0]:.3f} / {l2[1]:.3f} / {l2[2]:.3f}  (avg {sum(l2)/3:.3f})")
    print(f"Col 1s/2s/3s: {col[0]:.2f} / {col[1]:.2f} / {col[2]:.2f} %")
    print(f"Alignment:    {align:.3f}")
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
