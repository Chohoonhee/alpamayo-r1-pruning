"""Aggregate sharded full-set eval JSONs into a single result.

Each shard has {L2_1s..3s, Col_1s..3s, alignment_match_rate, rows:[...]}.
We re-aggregate by concatenating rows and recomputing means weighted by
shard size — equivalent to running the full set in one shot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="per-shard JSON paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []
    base_keys = None
    for p in args.shards:
        d = json.load(open(p))
        all_rows.extend(d.get("rows", []))
        if base_keys is None:
            base_keys = {
                "policy_meta": d.get("policy_meta"),
                "dropped_layers": d.get("dropped_layers", []),
                "n_dropped": d.get("n_dropped", 0),
            }

    n = len(all_rows)
    if n == 0:
        print("[err] no rows aggregated"); return

    def avg(key, scale=1.0):
        vals = [r[key] for r in all_rows if key in r]
        return sum(vals) / max(1, len(vals)) * scale

    summary = {
        **base_keys,
        "n_samples": n,
        "n_shards": len(args.shards),
        "L2_1s": avg("L2_1s"),
        "L2_2s": avg("L2_2s"),
        "L2_3s": avg("L2_3s"),
        "Col_1s": avg("Col_1s", 100),
        "Col_2s": avg("Col_2s", 100),
        "Col_3s": avg("Col_3s", 100),
        "alignment_match_rate": avg("match"),
        "rows": all_rows,
    }
    summary["L2_avg"] = (summary["L2_1s"] + summary["L2_2s"] + summary["L2_3s"]) / 3
    with open(args.out, "w") as f:
        json.dump(summary, f)
    print(f"\n=== aggregated full-set (n={n}, {len(args.shards)} shards) ===")
    print(f"L2 1s/2s/3s:  {summary['L2_1s']:.3f} / {summary['L2_2s']:.3f} / {summary['L2_3s']:.3f}  "
          f"(avg {summary['L2_avg']:.3f})")
    print(f"Col 1s/2s/3s: {summary['Col_1s']:.2f} / {summary['Col_2s']:.2f} / {summary['Col_3s']:.2f} %")
    print(f"Alignment:    {summary['alignment_match_rate']:.3f}")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
