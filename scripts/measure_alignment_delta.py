"""Per-layer CoT-Action alignment delta — the scoring core of the
alignment-grounded pruning paper.

For each VLM layer ℓ ∈ {0..N-1}:
    importance(ℓ) = mean_s [ align(model, s) - align(model_ℓ_bypassed, s) ]

  + importance > +ε → layer HELPS alignment (KEEP)
  ≈ 0 within noise   → layer is NEUTRAL (PRUNE for compression)
  − importance < -ε → layer HURTS alignment (PRUNE → model improves)

Outputs `alignment_scores_{backbone}.json` with the per-layer score plus
the raw match-rate tables, for downstream policy decisions and paper Tab 1.

NOTE: Scaffold. Wired up against Alpamayo 1.5 (Cosmos-Reason2). For R1
swap `Alpamayo1_5` for the R1 model class. Both share the
`model.vlm.language_model.layers` path so the bypass mechanism is identical.

Open items (mark TODO):
  - CoT extraction: `sample_trajectories_from_data_with_vlm_rollout(...,
    return_extra=True)` should expose the generated assistant tokens. If
    that surface doesn't work as expected, fall back to running VLM
    `.generate()` separately and stitching the trajectory after — slower
    but unambiguous. The ViPE project's `alpamayo_infer_server_noprefix.py`
    has the reference implementation for the clean CoT generation path.
  - Calibration set: this script accepts an arbitrary nuScenes val token
    list. Default is a stride-sampled 500 val tokens. For the paper, lock
    the seed + token list and check it into `results/calibration_set.json`.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(v15=True)

import argparse
import json
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    extract_front_cams, get_past_history, get_nav_text,
    NUSC_ROOT, VERSION,
)

from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


# ── Layer bypass ─────────────────────────────────────────────────────────
@contextmanager
def bypass_layer(model, layer_idx: int):
    """Temporarily make `model.vlm.language_model.layers[layer_idx]` an
    identity, restoring the original forward on exit.

    Identity-bypass (not physical removal) so weights stay loaded and the
    HybridCache layer_idx slots stay correct — required for any forward
    that uses the cache (training or inference)."""
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
def run_inference_with_cot(model, processor, frames_np, ego_xyz, ego_rot,
                           nav_text, cam_idx_np, device):
    """Run a single forward + sample and return (cot_text, action_xy_64).

    TODO: confirm that `sample_trajectories_from_data_with_vlm_rollout(...,
    return_extra=True)` exposes generated tokens (`extra['generated_ids']`
    or similar). If it does, decode via the tokenizer to get the CoT text.
    Fallback path: call `helper.run_inference(...)` (or analogous) for the
    CoT text, then `sample_trajectories_from_data` for the action.
    """
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
            num_traj_samples=1, max_generation_length=96, return_extra=True,
        )
    # `out` shape varies; with return_extra=True it should be
    # (waypoints, extras) where extras contains generated ids.
    if isinstance(out, tuple) and len(out) == 2:
        pred_xyz, extras = out
    else:
        pred_xyz, extras = out, {}

    pred_xyz_np = pred_xyz[0].detach().float().cpu().numpy()
    pred_flat = pred_xyz_np.reshape(-1, pred_xyz_np.shape[-2], pred_xyz_np.shape[-1])
    action_xy = pred_flat[0, :, :2]

    # TODO: replace this with the real CoT extraction once `extras` surface
    # is confirmed. For now, return empty string → CoT class falls back to
    # 'cruise'. This makes the scaffold runnable end-to-end but the score
    # only reflects action-vs-action drift until CoT extraction lands.
    cot_text = ""
    if isinstance(extras, dict):
        ids = extras.get("generated_ids") or extras.get("vlm_tokens")
        if ids is not None:
            try:
                cot_text = processor.tokenizer.decode(
                    ids[0] if hasattr(ids, "__getitem__") else ids,
                    skip_special_tokens=True,
                )
            except Exception:
                cot_text = ""

    return cot_text, action_xy


# ── Calibration set ──────────────────────────────────────────────────────
def select_calibration_tokens(n: int) -> list[str]:
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


# ── Per-sample evaluation harness ────────────────────────────────────────
def classify_one(model, processor, nusc, token, cam_idx_np, device):
    frames = extract_front_cams(nusc, token)
    hist, hist_rot = get_past_history(nusc, token)
    nav = get_nav_text(nusc, token)
    ego_xyz = hist[None, None]
    ego_rot = hist_rot[None, None]
    cot, act_xy = run_inference_with_cot(
        model, processor, frames, ego_xyz, ego_rot, nav, cam_idx_np, device,
    )
    cot_label = classify_cot_rule(cot)
    act_label = classify_traj_4class(act_xy.tolist())
    return cot_label, act_label, alignment_match(cot_label, act_label)


def measure_pass(model, processor, nusc, tokens, cam_idx_np, device, tag):
    rows = []
    t0 = time.time()
    for i, tok in enumerate(tokens):
        try:
            cot_l, act_l, m = classify_one(model, processor, nusc, tok, cam_idx_np, device)
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


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--n_samples", type=int, default=500,
                    help="size of calibration set")
    ap.add_argument("--layers", type=str, default=None,
                    help="comma-separated layer indices to bypass. "
                         "Default: every VLM layer.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[load] {args.orig_weights}", flush=True)
    model = Alpamayo1_5.from_pretrained(args.orig_weights, dtype=torch.bfloat16).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    n_layers = len(model.vlm.language_model.layers)
    print(f"[load] {n_layers} VLM layers detected", flush=True)

    target_layers = (
        sorted(int(x) for x in args.layers.split(","))
        if args.layers else list(range(n_layers))
    )

    tokens, nusc = select_calibration_tokens(args.n_samples)
    print(f"[cal] {len(tokens)} calibration tokens (val stride-sampled)", flush=True)
    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)

    # Baseline (no bypass) — single pass, expensive enough to cache.
    print(f"[baseline] full model alignment …", flush=True)
    baseline_rate, baseline_rows = measure_pass(
        model, processor, nusc, tokens, cam_idx_np, args.device, "base",
    )
    print(f"[baseline] match_rate={baseline_rate:.4f}", flush=True)

    # Per-layer bypass.
    per_layer = []
    for li in target_layers:
        print(f"\n[bypass ℓ={li}/{n_layers-1}]", flush=True)
        with bypass_layer(model, li):
            rate, rows = measure_pass(
                model, processor, nusc, tokens, cam_idx_np, args.device, f"ℓ{li}",
            )
        delta = baseline_rate - rate  # +: layer helps, -: layer hurts
        print(f"[bypass ℓ={li}]  rate={rate:.4f}  importance={delta:+.4f}",
              flush=True)
        per_layer.append({
            "layer": li,
            "bypassed_rate": rate,
            "importance": delta,
            "rows": rows,
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

    # Quick summary banner — what would the policy say.
    eps = 0.005  # tweak in paper based on noise band
    helpful  = [r["layer"] for r in per_layer if r["importance"] >  eps]
    neutral  = [r["layer"] for r in per_layer if abs(r["importance"]) <= eps]
    harmful  = [r["layer"] for r in per_layer if r["importance"] < -eps]
    print(f"\n=== Policy preview (ε={eps}) ===")
    print(f"  KEEP   ({len(helpful)}): {helpful}")
    print(f"  PRUNE neutral ({len(neutral)}): {neutral}")
    print(f"  PRUNE harmful ({len(harmful)}): {harmful}")


if __name__ == "__main__":
    main()
