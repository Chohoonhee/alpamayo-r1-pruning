"""Apply RANDOM layer drop policy + measure alignment + L2 on nuScenes val.

Generates K random drop sets of size n_drop with different seeds,
evaluates each, reports mean ± std. This is the control we need to
claim our alignment-grounded greedy beats a random baseline.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS, ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)

import argparse
import json
import random
import time

import numpy as np
import torch


def _id_fwd():
    def _f(hidden_states, *a, **kw):
        return hidden_states
    return _f


def bypass(model, drop_idx):
    vlm = model.vlm.language_model.layers
    saved = {i: vlm[i].forward for i in drop_idx}
    for i in drop_idx:
        vlm[i].forward = _id_fwd()
    return saved


def restore(model, saved):
    vlm = model.vlm.language_model.layers
    for i, orig in saved.items():
        vlm[i].forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15", "r1"], required=True)
    ap.add_argument("--n_drop", type=int, required=True)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    # Build N random drop sets
    drop_sets = []
    rng = random.Random(0)
    for s in range(args.n_seeds):
        r = random.Random(s)
        idx = sorted(r.sample(range(36), args.n_drop))
        drop_sets.append({"seed": s, "dropped": idx})

    # Write per-seed meta files and call eval script via subprocess
    import subprocess, os
    results = []
    for ds in drop_sets:
        meta_path = args.out_json.replace(".json", f"_seed{ds['seed']}.meta.json")
        eval_path = args.out_json.replace(".json", f"_seed{ds['seed']}.eval.json")
        with open(meta_path, "w") as f:
            json.dump({
                "dropped_layers": ds["dropped"],
                "policy": "random_baseline",
                "backbone": args.backbone, "seed": ds["seed"],
            }, f, indent=2)
        eval_script = ("eval_zeroshot_alignment.py" if args.backbone == "15"
                       else "eval_zeroshot_alignment_r1.py")
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        cmd = [
            "python", eval_script,
            "--policy_meta", meta_path,
            "--n_samples", str(args.n_samples),
            "--out_json", eval_path,
            "--device", args.device,
        ]
        print(f"\n[run] seed={ds['seed']} drop={ds['dropped']}", flush=True)
        subprocess.run(cmd, env=env, check=False)
        with open(eval_path) as f:
            r = json.load(f)
        results.append({
            "seed": ds["seed"], "dropped": ds["dropped"],
            "L2_avg": r["L2_avg"], "L2_3s": r["L2_3s"],
            "alignment": r["alignment_match_rate"],
        })

    summary = {
        "backbone": args.backbone, "n_drop": args.n_drop,
        "n_seeds": args.n_seeds, "n_samples": args.n_samples,
        "results": results,
        "L2_avg_mean": float(np.mean([r["L2_avg"] for r in results])),
        "L2_avg_std": float(np.std([r["L2_avg"] for r in results])),
        "alignment_mean": float(np.mean([r["alignment"] for r in results])),
        "alignment_std": float(np.std([r["alignment"] for r in results])),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Random baseline {args.backbone} n_drop={args.n_drop} ({args.n_seeds} seeds) ===")
    print(f"L2 avg:    {summary['L2_avg_mean']:.3f} ± {summary['L2_avg_std']:.3f}")
    print(f"Alignment: {summary['alignment_mean']:.3f} ± {summary['alignment_std']:.3f}")
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
