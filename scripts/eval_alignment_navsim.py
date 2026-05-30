"""Apply a pruning policy + measure CoT-Action alignment on NAVSIM samples.

Reads the same pickle format as run_iterative_greedy_navsim.py.

Usage:
    python eval_alignment_navsim.py \\
        --backbone r1 \\
        --samples_pkl logs/navsim_samples_100.pkl \\
        --sample_slice "50:100" \\
        --policy_meta logs/greedyR1_navsim_meta.json \\
        --out_json logs/evalR1_navsim_greedy_holdout.json
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS, ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)

import argparse
import json
import pickle
import time
from contextlib import contextmanager

import numpy as np
import torch

from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


@contextmanager
def bypass_layers(model, layer_indices):
    layers = model.vlm.language_model.layers
    originals = {i: layers[i].forward for i in layer_indices}

    def _identity(hidden_states, *a, **kw):
        return hidden_states

    for i in layer_indices:
        layers[i].forward = _identity
    try:
        yield
    finally:
        for i, orig in originals.items():
            layers[i].forward = orig


INTEGER_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}


def run_inference(model, processor, sample, device, backbone):
    if backbone == "15":
        from alpamayo1_5 import helper as h15
        frames = torch.from_numpy(sample["image_frames"]).to(device)
        n_cam, n_frames = frames.shape[0], frames.shape[1]
        flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
        cam_idx = torch.tensor(sample["camera_indices"], dtype=torch.long).to(device)
        messages = h15.create_message(frames=flat, camera_indices=cam_idx,
                                       nav_text=(sample["nav_text"] or None))
        tok = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
    else:
        from alpamayo_r1.helper import create_message as cmR1
        frames = torch.from_numpy(sample["image_frames"]).to(device)
        n_cam, n_frames = frames.shape[0], frames.shape[1]
        flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
        messages = cmR1(flat)
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
    ego_xyz = torch.from_numpy(sample["ego_history_xyz"]).float().to(device)
    ego_rot = torch.from_numpy(sample["ego_history_rot"]).float().to(device)
    data = {"tokenized_data": td, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if backbone == "15":
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=data, top_p=0.98, temperature=0.6,
                num_traj_samples=1, max_generation_length=256, return_extra=True,
            )
        else:
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data, num_traj_samples=1, num_traj_sets=1,
                return_extra=True, top_p=0.98, temperature=0.6, max_new_tokens=256,
            )
    action64 = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
    cot_text = str(extra["cot"][0, 0, 0])
    return cot_text, action64


def parse_slice(spec, n):
    a, _, b = spec.partition(":")
    a = int(a) if a else 0
    b = int(b) if b else n
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15", "r1"], required=True)
    ap.add_argument("--samples_pkl", required=True)
    ap.add_argument("--sample_slice", default="0:",
                    help="Python-slice on the pickled list, e.g. '50:100' or '0:50'")
    ap.add_argument("--policy_meta", default=None,
                    help="meta json with dropped_layers list. Omit for baseline.")
    ap.add_argument("--orig_weights", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if args.backbone == "15":
        add_alpamayo_to_syspath(v15=True)
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper as h15
        weights = args.orig_weights or str(ALPAMAYO_15_WEIGHTS)
        print(f"[load] 1.5: {weights}", flush=True)
        model = Alpamayo1_5.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = h15.get_processor(model.tokenizer)
    else:
        add_alpamayo_to_syspath(r1=True)
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        weights = args.orig_weights or str(ALPAMAYO_R1_WEIGHTS)
        print(f"[load] R1: {weights}", flush=True)
        model = AlpamayoR1.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = None
    model.eval()

    drop_set = []
    if args.policy_meta:
        with open(args.policy_meta) as f:
            drop_set = sorted(set(json.load(f)["dropped_layers"]))
        print(f"[prune] applied drop_set {drop_set}", flush=True)

    with open(args.samples_pkl, "rb") as f:
        all_samples = pickle.load(f)
    a, b = parse_slice(args.sample_slice, len(all_samples))
    samples = all_samples[a:b]
    print(f"[cal] {len(samples)} samples ({args.sample_slice} of {len(all_samples)})", flush=True)

    rows = []
    t0 = time.time()
    with bypass_layers(model, drop_set):
        for i, s in enumerate(samples):
            try:
                cot, act = run_inference(model, processor, s, args.device, args.backbone)
                cot_l = classify_cot_rule(cot)
                act_l = classify_traj_4class(act.tolist())
                m = alignment_match(cot_l, act_l)
                rows.append({"token": s.get("token", "?"), "cot_label": cot_l,
                             "action_label": act_l, "match": m,
                             "cot": cot, "action_xy0": act[0].tolist(),
                             "action_xy_last": act[-1].tolist()})
            except Exception as e:
                rows.append({"token": s.get("token", "?"), "match": 0,
                             "error": f"{type(e).__name__}: {e}"})
            if (i + 1) % 25 == 0:
                rate = sum(r["match"] for r in rows) / len(rows)
                print(f"  [{i+1}/{len(samples)}] align={rate:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

    align = sum(r["match"] for r in rows) / max(1, len(rows))
    summary = {
        "samples_pkl": args.samples_pkl,
        "sample_slice": args.sample_slice,
        "policy_meta": args.policy_meta,
        "drop_set": drop_set,
        "n_samples": len(rows),
        "alignment_match_rate": align,
        "rows": rows,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f)
    print(f"\n=== align={align:.4f} drop={len(drop_set)} n={len(rows)} ===")
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
