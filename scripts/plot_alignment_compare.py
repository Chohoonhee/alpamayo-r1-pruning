"""Side-by-side per-layer importance: R1 vs 1.5 backbone.

The killer figure for the paper. Two bar charts on one canvas, identical
y-axis, layer index on x. Color: green=KEEP, gray=neutral, red=HARMFUL.

Usage:
    python plot_alignment_compare.py \\
        --r1   logs/pilotR1_merged.json \\
        --v15  logs/pilot15_merged.json \\
        --out  logs/compare_r1_v15_per_layer.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_merged(path):
    with open(path) as f:
        data = json.load(f)
    rows = sorted(data["per_layer"], key=lambda r: r["layer"])
    layers = [r["layer"] for r in rows]
    imp = np.array([r["importance"] for r in rows])
    return data, layers, imp


def bar_axes(ax, layers, imp, eps, title, baseline_match):
    colors = [
        "#1b9e77" if v > eps else "#d62728" if v < -eps else "#888888"
        for v in imp
    ]
    ax.bar(layers, imp, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline( eps, ls="--", lw=0.6, color="black", alpha=0.5)
    ax.axhline(-eps, ls="--", lw=0.6, color="black", alpha=0.5)
    ax.axhline(0,    ls="-",  lw=0.6, color="black")
    ax.set_xlabel("VLM layer index")
    ax.set_ylabel("alignment importance\n(baseline − bypassed)")
    ax.set_title(f"{title}  (baseline match = {baseline_match:.3f})")
    ax.set_xticks(layers[::2])
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)

    n_keep    = sum(v >  eps for v in imp)
    n_neutral = sum(abs(v) <= eps for v in imp)
    n_harmful = sum(v < -eps for v in imp)
    summary = (f"KEEP={n_keep:2d}  NEUTRAL={n_neutral:2d}  HARMFUL={n_harmful:2d}"
               f"   → compressible: {n_neutral + n_harmful}/{len(imp)}")
    ax.text(0.01, 0.97, summary, transform=ax.transAxes,
            ha="left", va="top", family="monospace", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1",  required=True)
    ap.add_argument("--v15", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d_r1, ly_r1, imp_r1 = load_merged(args.r1)
    d_15, ly_15, imp_15 = load_merged(args.v15)

    ymin = float(np.floor(min(imp_r1.min(), imp_15.min()) * 20) / 20)
    ymax = float(np.ceil( max(imp_r1.max(), imp_15.max()) * 20) / 20)

    fig, (ax_r1, ax_15) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    bar_axes(ax_r1, ly_r1, imp_r1, d_r1["eps"],
             "Alpamayo-R1-10B (vanilla Qwen3VL backbone)",
             d_r1["baseline_mean"])
    bar_axes(ax_15, ly_15, imp_15, d_15["eps"],
             "Alpamayo-1.5-10B (Cosmos-Reason2 backbone)",
             d_15["baseline_mean"])
    ax_r1.set_ylim(ymin, ymax)
    ax_15.set_ylim(ymin, ymax)
    fig.suptitle("Per-layer CoT-Action alignment importance — R1 vs 1.5",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
