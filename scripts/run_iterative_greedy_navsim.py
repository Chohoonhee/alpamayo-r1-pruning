"""Iterative-greedy with arbitrary pickled calibration samples (NAVSIM, or
mixed). Mirrors run_iterative_greedy.py but reads samples from a pickle
produced by extract_navsim_samples.py instead of loading nuScenes via
SceneLoader.

Why a separate script: navsim_venv (Python 3.9) does NAVSIM loading;
alpamayo_b2d (Python 3.12) runs the model. We bridge via pickle.

Usage:
    python run_iterative_greedy_navsim.py \\
        --backbone r1 --samples_pkl logs/navsim_samples.pkl \\
        --rounds 12 --out_json logs/greedyR1_navsim.json
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


def _build_15_messages(processor, frames_np, cam_idx_np, nav_text, device):
    add_alpamayo_to_syspath(v15=True)
    from alpamayo1_5 import helper as h15
    # frames: (N_cam, num_frames, 3, H, W) uint8
    frames = torch.from_numpy(frames_np).to(device)
    # collapse (N_cam, T, ...) -> (N_cam*T, ...) — matches helper.create_message expectation
    n_cam, n_frames = frames.shape[0], frames.shape[1]
    flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
    cam_idx = torch.tensor(cam_idx_np, dtype=torch.long).to(device)
    messages = h15.create_message(frames=flat, camera_indices=cam_idx,
                                  nav_text=(nav_text or None))
    return messages


def _build_r1_messages(processor_unused, frames_np, cam_idx_np_unused, nav_text_unused, device):
    add_alpamayo_to_syspath(r1=True)
    from alpamayo_r1.helper import create_message as cmR1
    frames = torch.from_numpy(frames_np).to(device)
    n_cam, n_frames = frames.shape[0], frames.shape[1]
    flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
    return cmR1(flat)


def run_inference(model, processor, sample, device, backbone):
    """Run inference and return (cot_text, action64_xy ndarray)."""
    if backbone == "15":
        messages = _build_15_messages(processor, sample["image_frames"],
                                       sample["camera_indices"], sample["nav_text"], device)
        tok = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
    else:
        messages = _build_r1_messages(processor, sample["image_frames"],
                                       sample["camera_indices"], sample["nav_text"], device)
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


def measure_alignment(model, processor, samples, device, backbone, drop_set):
    matches = 0
    n = len(samples)
    with bypass_layers(model, drop_set):
        for s in samples:
            try:
                cot, act = run_inference(model, processor, s, device, backbone)
                cot_l = classify_cot_rule(cot)
                act_l = classify_traj_4class(act.tolist())
                matches += alignment_match(cot_l, act_l)
            except Exception:
                pass
    return matches / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15", "r1"], required=True)
    ap.add_argument("--orig_weights", default=None)
    ap.add_argument("--samples_pkl", required=True)
    ap.add_argument("--n_samples", type=int, default=0,
                    help="0 = use all from pickle; else slice first N")
    ap.add_argument("--rounds", type=int, default=12)
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
    n_layers = len(model.vlm.language_model.layers)
    print(f"[load] {n_layers} VLM layers detected", flush=True)

    with open(args.samples_pkl, "rb") as f:
        samples = pickle.load(f)
    if args.n_samples > 0:
        samples = samples[:args.n_samples]
    print(f"[cal] {len(samples)} calibration samples loaded from {args.samples_pkl}",
          flush=True)

    base_align = measure_alignment(model, processor, samples, args.device, args.backbone, set())
    print(f"[baseline] align={base_align:.4f}  drop=∅", flush=True)

    drop_set = []
    history = [{"round": 0, "drop_set": [], "best_layer": None, "best_align": base_align}]

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        candidates = sorted(set(range(n_layers)) - set(drop_set))
        scores = {}
        for ci, c in enumerate(candidates):
            trial = drop_set + [c]
            a = measure_alignment(model, processor, samples, args.device, args.backbone, set(trial))
            scores[c] = a
            print(f"  [r{r} {ci+1}/{len(candidates)} ℓ={c}] align={a:.3f}", flush=True)

        best = max(scores, key=scores.get)
        best_a = scores[best]
        drop_set.append(best)
        history.append({
            "round": r, "drop_set": list(drop_set),
            "best_layer": best, "best_align": best_a,
            "all_scores": scores, "round_elapsed_s": time.time() - t0,
        })
        print(f"\n[round {r}] +ℓ={best} → drop={drop_set}  align={best_a:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out_json, "w") as f:
        json.dump({
            "backbone": args.backbone, "weights": weights,
            "samples_pkl": args.samples_pkl,
            "n_calibration": len(samples), "rounds": args.rounds,
            "baseline_align": base_align,
            "final_drop_set": drop_set, "history": history,
        }, f, indent=2)
    print(f"\nSaved → {args.out_json}")

    meta_path = args.out_json.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dropped_layers": sorted(drop_set),
            "policy": "iterative_greedy_navsim",
            "backbone": args.backbone,
            "source": args.out_json,
        }, f, indent=2)
    print(f"Saved meta → {meta_path}")


if __name__ == "__main__":
    main()
