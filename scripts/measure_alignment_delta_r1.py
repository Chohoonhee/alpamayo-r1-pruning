"""Per-layer CoT-Action alignment delta — R1 backbone variant.

Same methodology as measure_alignment_delta.py (1.5 version), adapted for
Alpamayo-R1-10B:
  - imports `alpamayo_r1.models.alpamayo_r1.AlpamayoR1` + `alpamayo_r1.helper`
  - R1's `helper.create_message(frames)` takes no nav_text / camera_indices
  - processor is attached to model (`model.processor`) rather than rebuilt

See measure_alignment_delta.py for the methodology comments.
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
from contextlib import contextmanager

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.helper import create_message
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    extract_front_cams, get_past_history,
    NUSC_ROOT, VERSION,
)

from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


# ── Layer bypass ─────────────────────────────────────────────────────────
@contextmanager
def bypass_layer(model, layer_idx: int):
    """Same as 1.5: temporary identity on model.vlm.language_model.layers[ℓ]."""
    layers = model.vlm.language_model.layers
    original = layers[layer_idx].forward

    def _identity(hidden_states, *a, **kw):
        return hidden_states

    layers[layer_idx].forward = _identity
    try:
        yield
    finally:
        layers[layer_idx].forward = original


# ── Inference returning both CoT and action ──────────────────────────────
INTEGER_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}


def run_inference_with_cot(model, frames_np, hist_xyz, hist_rot, device):
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
            return_extra=True, top_p=0.98, temperature=0.6,
            max_new_tokens=256,
        )
    action_xy = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
    cot_text = str(extra["cot"][0, 0, 0])
    return cot_text, action_xy


# ── Calibration set ──────────────────────────────────────────────────────
def select_calibration_tokens(n: int):
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val = []
    for scene in nusc.scene:
        if scene["name"] in set(splits.get("val", [])):
            t = scene["first_sample_token"]
            while t:
                val.append(t)
                t = nusc.get("sample", t)["next"]
    stride = max(1, len(val) // n)
    return val[::stride][:n], nusc


def classify_one(model, nusc, token, device):
    frames = extract_front_cams(nusc, token)
    hist, hist_rot = get_past_history(nusc, token)
    cot, act_xy = run_inference_with_cot(model, frames, hist, hist_rot, device)
    cot_label = classify_cot_rule(cot)
    act_label = classify_traj_4class(act_xy.tolist())
    return cot_label, act_label, alignment_match(cot_label, act_label)


def measure_pass(model, nusc, tokens, device, tag):
    rows = []
    t0 = time.time()
    for i, tok in enumerate(tokens):
        try:
            cot_l, act_l, m = classify_one(model, nusc, tok, device)
            rows.append({"token": tok, "cot": cot_l, "action": act_l, "match": m})
        except Exception as e:
            rows.append({"token": tok, "cot": None, "action": None, "match": 0,
                         "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 25 == 0:
            rate = sum(r["match"] for r in rows) / len(rows)
            print(f"  [{tag} {i+1}/{len(tokens)}] match_rate={rate:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    rate = sum(r["match"] for r in rows) / max(1, len(rows))
    return rate, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default=str(ALPAMAYO_R1_WEIGHTS))
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[load] {args.orig_weights}", flush=True)
    model = AlpamayoR1.from_pretrained(args.orig_weights, dtype=torch.bfloat16).to(args.device)
    model.eval()
    n_layers = len(model.vlm.language_model.layers)
    print(f"[load] {n_layers} VLM layers detected", flush=True)

    target_layers = (
        sorted(int(x) for x in args.layers.split(","))
        if args.layers else list(range(n_layers))
    )

    tokens, nusc = select_calibration_tokens(args.n_samples)
    print(f"[cal] {len(tokens)} calibration tokens (val stride-sampled)", flush=True)

    print(f"[baseline] full model alignment …", flush=True)
    baseline_rate, baseline_rows = measure_pass(model, nusc, tokens, args.device, "base")
    print(f"[baseline] match_rate={baseline_rate:.4f}", flush=True)

    per_layer = []
    for li in target_layers:
        print(f"\n[bypass ℓ={li}/{n_layers-1}]", flush=True)
        with bypass_layer(model, li):
            rate, rows = measure_pass(model, nusc, tokens, args.device, f"ℓ{li}")
        delta = baseline_rate - rate
        print(f"[bypass ℓ={li}]  rate={rate:.4f}  importance={delta:+.4f}", flush=True)
        per_layer.append({
            "layer": li, "bypassed_rate": rate, "importance": delta, "rows": rows,
        })

    out = {
        "weights": str(args.orig_weights),
        "n_calibration": len(tokens),
        "baseline_match_rate": baseline_rate,
        "baseline_rows": baseline_rows,
        "per_layer": per_layer,
        "n_vlm_layers": n_layers,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"\nSaved → {args.out_json}", flush=True)

    eps = 0.005
    helpful  = [r["layer"] for r in per_layer if r["importance"] >  eps]
    neutral  = [r["layer"] for r in per_layer if abs(r["importance"]) <= eps]
    harmful  = [r["layer"] for r in per_layer if r["importance"] < -eps]
    print(f"\n=== Policy preview (ε={eps}) ===")
    print(f"  KEEP   ({len(helpful)}): {helpful}")
    print(f"  PRUNE neutral ({len(neutral)}): {neutral}")
    print(f"  PRUNE harmful ({len(harmful)}): {harmful}")


if __name__ == "__main__":
    main()
