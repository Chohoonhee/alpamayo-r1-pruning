"""Iterative-greedy layer drop: at each round, with the current drop set
D fixed, evaluate alignment when ADDITIONALLY bypassing each remaining
layer one-by-one. Pick the layer whose addition gives the HIGHEST
remaining alignment (or doesn't lower it). Add to D, repeat.

This directly searches for a compatible drop set, rather than relying on
the single-layer importance ranking (which we showed doesn't compose).

Loads the model ONCE, applies bypasses per candidate inline. Single-GPU.

Usage:
    python run_iterative_greedy.py \\
        --backbone 15 \\
        --n_samples 50 \\
        --rounds 10 \\
        --out_json logs/greedy15.json
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS, ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)

import argparse
import json
import time
from contextlib import contextmanager

import numpy as np
import torch

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    extract_front_cams, get_past_history, get_nav_text,
    NUSC_ROOT, VERSION,
)
from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


def _add_15_paths():
    add_alpamayo_to_syspath(v15=True)


def _add_r1_paths():
    add_alpamayo_to_syspath(r1=True)


# ── Bypass multiple layers simultaneously ─────────────────────────────────
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


# ── Inference (backbone-specific) ─────────────────────────────────────────
INTEGER_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}


def _infer_15(model, processor, frames_np, ego_xyz, ego_rot, nav_text, cam_idx_np, device):
    from alpamayo1_5 import helper as h15
    frames = torch.from_numpy(frames_np).to(device)
    cam_idx = torch.tensor(cam_idx_np, dtype=torch.long).to(device) if cam_idx_np is not None else None
    messages = h15.create_message(frames=frames, camera_indices=cam_idx, nav_text=nav_text)
    tok = processor.apply_chat_template(
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


def _infer_r1(model, frames_np, hist_xyz, hist_rot, device):
    from alpamayo_r1.helper import create_message as cmR1
    frames = torch.from_numpy(frames_np).to(device)
    messages = cmR1(frames)
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


def measure_alignment(model, processor, nusc, tokens, cam_idx_np, device,
                       backbone, drop_set):
    """Run inference on all tokens with `drop_set` layers bypassed.
    Returns mean alignment match rate."""
    matches = 0
    with bypass_layers(model, drop_set):
        for tok in tokens:
            try:
                frames = extract_front_cams(nusc, tok)
                hist, hist_rot = get_past_history(nusc, tok)
                if backbone == "15":
                    nav = get_nav_text(nusc, tok)
                    ego_xyz = hist[None, None]
                    ego_rot = hist_rot[None, None]
                    cot, act_xy = _infer_15(model, processor, frames, ego_xyz, ego_rot,
                                            nav, cam_idx_np, device)
                else:
                    cot, act_xy = _infer_r1(model, frames, hist, hist_rot, device)
                cot_label = classify_cot_rule(cot)
                act_label = classify_traj_4class(act_xy.tolist())
                matches += alignment_match(cot_label, act_label)
            except Exception as e:
                # Failed inference counts as non-match (model broken)
                pass
    return matches / max(1, len(tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15", "r1"], required=True)
    ap.add_argument("--orig_weights", default=None,
                    help="defaults: 1.5→ALPAMAYO_15_WEIGHTS, r1→ALPAMAYO_R1_WEIGHTS")
    ap.add_argument("--n_samples", type=int, default=50,
                    help="smaller calibration set for greedy speed")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if args.backbone == "15":
        _add_15_paths()
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper as h15
        weights = args.orig_weights or str(ALPAMAYO_15_WEIGHTS)
        print(f"[load] 1.5: {weights}", flush=True)
        model = Alpamayo1_5.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = h15.get_processor(model.tokenizer)
    else:
        _add_r1_paths()
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        weights = args.orig_weights or str(ALPAMAYO_R1_WEIGHTS)
        print(f"[load] R1: {weights}", flush=True)
        model = AlpamayoR1.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = None
    model.eval()
    n_layers = len(model.vlm.language_model.layers)
    print(f"[load] {n_layers} VLM layers detected", flush=True)

    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val = []
    for scene in nusc.scene:
        if scene["name"] in set(splits.get("val", [])):
            t = scene["first_sample_token"]
            while t:
                val.append(t)
                t = nusc.get("sample", t)["next"]
    stride = max(1, len(val) // args.n_samples)
    tokens = val[::stride][:args.n_samples]
    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)
    print(f"[cal] {len(tokens)} calibration tokens, stride={stride}", flush=True)

    base_align = measure_alignment(model, processor, nusc, tokens, cam_idx_np,
                                   args.device, args.backbone, set())
    print(f"[baseline] align={base_align:.4f}  drop=∅", flush=True)

    drop_set = []
    history = [{"round": 0, "drop_set": [], "best_layer": None, "best_align": base_align}]

    for r in range(1, args.rounds + 1):
        round_t0 = time.time()
        candidates = sorted(set(range(n_layers)) - set(drop_set))
        scores = {}
        for ci, c in enumerate(candidates):
            trial = drop_set + [c]
            a = measure_alignment(model, processor, nusc, tokens, cam_idx_np,
                                  args.device, args.backbone, set(trial))
            scores[c] = a
            print(f"  [r{r} cand {ci+1}/{len(candidates)} ℓ={c}] "
                  f"align_with_drop={a:.3f}", flush=True)

        best_layer = max(scores, key=scores.get)
        best_align = scores[best_layer]
        drop_set.append(best_layer)
        history.append({
            "round": r, "drop_set": list(drop_set),
            "best_layer": best_layer, "best_align": best_align,
            "all_scores": scores,
            "round_elapsed_s": time.time() - round_t0,
        })
        print(f"\n[round {r}] +ℓ={best_layer} → drop_set={drop_set}  "
              f"align={best_align:.4f}  ({time.time()-round_t0:.0f}s)", flush=True)

    with open(args.out_json, "w") as f:
        json.dump({
            "backbone": args.backbone, "weights": weights,
            "n_calibration": len(tokens), "rounds": args.rounds,
            "baseline_align": base_align,
            "final_drop_set": drop_set,
            "history": history,
        }, f, indent=2)
    print(f"\nSaved → {args.out_json}")

    # Build pruning_meta.json from the greedy result
    meta_path = args.out_json.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dropped_layers": sorted(drop_set),
            "policy": "iterative_greedy",
            "backbone": args.backbone,
            "source": args.out_json,
        }, f, indent=2)
    print(f"Saved meta → {meta_path}")


if __name__ == "__main__":
    main()
