"""Merge per-layer alignment-delta shards into one ranked report.

Each shard JSON has its own `baseline_match_rate` (since shards are
independent processes that each re-computed baseline on the same
calibration set). Sanity-check: baselines should agree to within
sampling noise. If they diverge by >0.05 something is off.

Usage:
    python merge_alignment_shards.py \\
        logs/pilot15_shard0.json logs/pilot15_shard1.json ... \\
        --out logs/pilot15_merged.json --csv logs/pilot15_merged.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="per-shard JSON files")
    ap.add_argument("--out", required=True, help="merged JSON output")
    ap.add_argument("--csv", default=None, help="optional CSV per-layer report")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="neutral band: |importance| < eps")
    args = ap.parse_args()

    shards = []
    for p in args.shards:
        with open(p) as f:
            shards.append(json.load(f))

    baselines = [s["baseline_match_rate"] for s in shards]
    base_mean = statistics.mean(baselines)
    base_spread = max(baselines) - min(baselines)
    print(f"[baseline] shards={baselines}")
    print(f"[baseline] mean={base_mean:.4f}  spread={base_spread:.4f}")
    if base_spread > 0.05:
        print(f"[WARN] baseline spread > 0.05 — shards may be using different "
              f"calibration tokens. Verify --n_samples matched.")

    # Per-layer rows, ordered by layer index
    per_layer = {}
    for s in shards:
        for entry in s["per_layer"]:
            li = entry["layer"]
            if li in per_layer:
                print(f"[WARN] layer {li} appears in >1 shard — overwriting")
            per_layer[li] = entry

    rows = sorted(per_layer.values(), key=lambda r: r["layer"])
    if len(rows) != shards[0]["n_vlm_layers"]:
        missing = sorted(set(range(shards[0]["n_vlm_layers"])) -
                         set(per_layer.keys()))
        print(f"[WARN] {len(missing)} layers missing: {missing}")

    eps = args.eps
    helpful = [r for r in rows if r["importance"] >  eps]
    neutral = [r for r in rows if abs(r["importance"]) <= eps]
    harmful = [r for r in rows if r["importance"] < -eps]

    print(f"\n=== Per-layer importance (ε={eps}) ===")
    print(f"  {'ℓ':>3}  {'bypassed_rate':>14}  {'importance':>11}  bucket")
    for r in rows:
        imp = r["importance"]
        if   imp >  eps:  bucket = "KEEP"
        elif imp < -eps:  bucket = "HARMFUL → PRUNE"
        else:             bucket = "neutral → prune"
        print(f"  {r['layer']:>3}  {r['bypassed_rate']:>14.4f}  {imp:>+11.4f}  {bucket}")
    print(f"\nKEEP    ({len(helpful):2d}): {[r['layer'] for r in helpful]}")
    print(f"NEUTRAL ({len(neutral):2d}): {[r['layer'] for r in neutral]}")
    print(f"HARMFUL ({len(harmful):2d}): {[r['layer'] for r in harmful]}")

    out = {
        "shards": args.shards,
        "baselines": baselines,
        "baseline_mean": base_mean,
        "baseline_spread": base_spread,
        "n_vlm_layers": shards[0]["n_vlm_layers"],
        "eps": eps,
        "per_layer": rows,
        "policy": {
            "keep":    [r["layer"] for r in helpful],
            "neutral": [r["layer"] for r in neutral],
            "harmful": [r["layer"] for r in harmful],
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "bypassed_rate", "importance", "bucket"])
            for r in rows:
                imp = r["importance"]
                bucket = ("KEEP" if imp > eps
                          else "HARMFUL" if imp < -eps else "NEUTRAL")
                w.writerow([r["layer"], f"{r['bypassed_rate']:.4f}",
                            f"{imp:+.4f}", bucket])
        print(f"Saved → {args.csv}")


if __name__ == "__main__":
    main()
