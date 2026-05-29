"""Progressive sorted drop: drop layers in order of their single-layer
importance (most harmful / least helpful first), evaluating L2 +
alignment at each k = 1, 2, ..., K_max.

Tests whether the per-layer ranking is informative for progressive
multi-layer drop. If the curve degrades smoothly (L2 ↑ as k ↑), the
ranking is OK and we just need to pick the right cutoff. If the curve
is non-monotonic or worse than baseline immediately, single-layer
ranking doesn't compose, supporting the negative finding.

Usage:
    python run_progressive_sorted.py \\
        --backbone 15 --merged_json logs/pilot15_merged.json \\
        --k_max 15 --n_samples 100 \\
        --out_json logs/progsort15.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15", "r1"], required=True)
    ap.add_argument("--merged_json", required=True,
                    help="output of merge_alignment_shards.py")
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    with open(args.merged_json) as f:
        merged = json.load(f)

    rows = sorted(merged["per_layer"], key=lambda r: r["importance"])
    drop_order = [r["layer"] for r in rows]
    print(f"[plan] drop order (ascending importance): {drop_order}")
    print(f"[plan] importance values: {[round(r['importance'], 3) for r in rows]}")

    eval_script = ("eval_zeroshot_alignment.py" if args.backbone == "15"
                   else "eval_zeroshot_alignment_r1.py")
    log_dir = Path("logs")

    results = []
    for k in range(1, args.k_max + 1):
        dropped = sorted(drop_order[:k])
        meta_path = log_dir / f"progsort{args.backbone}_k{k}.meta.json"
        out_path  = log_dir / f"progsort{args.backbone}_k{k}.eval.json"
        with open(meta_path, "w") as f:
            json.dump({
                "dropped_layers": dropped,
                "policy": "progressive_sorted",
                "backbone": args.backbone,
                "k": k,
                "source": args.merged_json,
            }, f, indent=2)
        print(f"\n=== k={k} drop={dropped} ===", flush=True)
        cmd = [
            sys.executable, eval_script,
            "--policy_meta", str(meta_path),
            "--n_samples", str(args.n_samples),
            "--out_json", str(out_path),
            "--device", args.device,
        ]
        subprocess.run(cmd, check=True)

        with open(out_path) as f:
            eval_out = json.load(f)
        results.append({
            "k": k,
            "dropped": dropped,
            "L2_avg": eval_out["L2_avg"],
            "L2_1s": eval_out["L2_1s"],
            "L2_2s": eval_out["L2_2s"],
            "L2_3s": eval_out["L2_3s"],
            "alignment": eval_out["alignment_match_rate"],
        })

    print(f"\n=== Progressive sorted summary ({args.backbone}) ===")
    print(f"  {'k':>2}  {'L2 avg':>7}  {'L2_3s':>7}  {'align':>5}  dropped")
    for r in results:
        print(f"  {r['k']:>2}  {r['L2_avg']:>7.3f}  {r['L2_3s']:>7.3f}  "
              f"{r['alignment']:>5.3f}  {r['dropped']}")

    with open(args.out_json, "w") as f:
        json.dump({"backbone": args.backbone, "k_max": args.k_max,
                   "n_samples": args.n_samples, "results": results}, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
