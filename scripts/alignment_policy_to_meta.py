"""Convert an alignment-grounded merged report into a `pruning_meta.json`
compatible with `apply_vlm_only_prune` (sft_phase_c.py).

The downstream scripts (eval_zeroshot_ea_expert, sft_phase_c, ...) all read
`pruning_meta.json` containing `{"dropped_layers": [...]}`. This script
takes a merged alignment JSON and emits two such files: one dropping ONLY
the neutral layers (lossless compression), one dropping neutral + harmful
(compression + alignment improvement).

Usage:
    python alignment_policy_to_meta.py logs/pilot15_merged.json \\
        --neutral_out  logs/policy15_neutral.json \\
        --plus_harmful_out logs/policy15_plus_harmful.json \\
        --backbone 15
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_json")
    ap.add_argument("--neutral_out", required=True,
                    help="meta with only neutral layers dropped (lossless)")
    ap.add_argument("--plus_harmful_out", required=True,
                    help="meta with neutral + harmful layers dropped (alignment +)")
    ap.add_argument("--backbone", choices=["15", "r1"], required=True,
                    help="for the source-meta annotation only")
    args = ap.parse_args()

    with open(args.merged_json) as f:
        data = json.load(f)

    neutral = sorted(data["policy"]["neutral"])
    harmful = sorted(data["policy"]["harmful"])
    keep    = sorted(data["policy"]["keep"])

    print(f"[input] {args.merged_json}")
    print(f"  baseline_match = {data['baseline_mean']:.4f}  ε = {data['eps']}")
    print(f"  KEEP    ({len(keep):2d}): {keep}")
    print(f"  NEUTRAL ({len(neutral):2d}): {neutral}")
    print(f"  HARMFUL ({len(harmful):2d}): {harmful}")

    # Policy A: drop only neutral — lossless compression
    meta_neutral = {
        "dropped_layers": neutral,
        "policy": "alignment_grounded_neutral_only",
        "backbone": args.backbone,
        "source": args.merged_json,
        "baseline_match_rate": data["baseline_mean"],
        "epsilon": data["eps"],
        "rationale": "Drop layers whose CoT-Action alignment importance is "
                     "within ±ε of zero (no measurable effect on coherence).",
    }
    with open(args.neutral_out, "w") as f:
        json.dump(meta_neutral, f, indent=2)
    print(f"\nSaved → {args.neutral_out}  ({len(neutral)} layers dropped)")

    # Policy B: drop neutral + harmful — alignment improvement expected
    plus = sorted(set(neutral) | set(harmful))
    meta_plus = {
        "dropped_layers": plus,
        "policy": "alignment_grounded_neutral_plus_harmful",
        "backbone": args.backbone,
        "source": args.merged_json,
        "baseline_match_rate": data["baseline_mean"],
        "epsilon": data["eps"],
        "rationale": "Drop neutral (importance≈0) AND harmful (importance<-ε) "
                     "layers. Expectation: compression + improved alignment.",
    }
    with open(args.plus_harmful_out, "w") as f:
        json.dump(meta_plus, f, indent=2)
    print(f"Saved → {args.plus_harmful_out}  ({len(plus)} layers dropped)")


if __name__ == "__main__":
    main()
