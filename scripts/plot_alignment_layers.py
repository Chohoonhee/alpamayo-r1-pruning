"""Plot per-layer alignment importance as a bar chart.

Reads the merged JSON from merge_alignment_shards.py and produces:
  - A colored bar chart (green=KEEP, gray=neutral, red=HARMFUL) saved as PNG.
  - A short text summary printed to stdout.

Usage:
    python plot_alignment_layers.py logs/pilot15_merged.json \\
        --out logs/pilot15_per_layer.png \\
        --title "Alpamayo 1.5 — per-layer CoT-Action alignment importance"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_json")
    ap.add_argument("--out", required=True, help="PNG output path")
    ap.add_argument("--title", default="Per-layer CoT-Action alignment importance")
    args = ap.parse_args()

    with open(args.merged_json) as f:
        data = json.load(f)

    eps = data["eps"]
    rows = sorted(data["per_layer"], key=lambda r: r["layer"])
    layers = [r["layer"] for r in rows]
    imp = np.array([r["importance"] for r in rows])

    colors = [
        "#1b9e77" if v > eps else "#d62728" if v < -eps else "#888888"
        for v in imp
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layers, imp, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline( eps, ls="--", lw=0.6, color="black", alpha=0.5)
    ax.axhline(-eps, ls="--", lw=0.6, color="black", alpha=0.5)
    ax.axhline(0,    ls="-",  lw=0.6, color="black")
    ax.set_xlabel("VLM layer index")
    ax.set_ylabel("importance = baseline_match - bypassed_match")
    ax.set_title(f"{args.title}\n"
                 f"(baseline match rate = {data['baseline_mean']:.3f}, "
                 f"N_cal = {data['per_layer'][0].get('bypassed_rate', None) and ''}"
                 f"ε = {eps})")
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)

    keep = data["policy"]["keep"]
    neutral = data["policy"]["neutral"]
    harmful = data["policy"]["harmful"]
    legend_text = (f"KEEP    ({len(keep):2d}): {keep}\n"
                   f"NEUTRAL ({len(neutral):2d}): {neutral}\n"
                   f"HARMFUL ({len(harmful):2d}): {harmful}")
    ax.text(0.01, -0.18, legend_text, transform=ax.transAxes,
            family="monospace", fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")

    print(f"\nbaseline_mean = {data['baseline_mean']:.4f}  "
          f"baseline_spread = {data['baseline_spread']:.4f}")
    print(f"KEEP    {len(keep):2d}: {keep}")
    print(f"NEUTRAL {len(neutral):2d}: {neutral}")
    print(f"HARMFUL {len(harmful):2d}: {harmful}")


if __name__ == "__main__":
    main()
