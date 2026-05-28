
"""Prune Alpamayo R1 by dropping text layers, save a new model checkpoint.

Strategies:
  - angular:  drop layers ranked by angular_dist_r1.py (least important first)
  - last:     drop the last N layers
  - random:   randomly drop N layers (fixed seed)

Usage:
    source $ALPAMAYO_15_SRC/a1_5_venv/bin/activate
    python prune_r1.py \
        --scores angular_scores_r1.json \
        --strategy angular --n_drop 13 \
        --out $ALPAMAYO_WEIGHTS_DIR/Alpamayo-R1-10B-pruned-angular13

    # Or without scores file (last / random):
    python prune_r1.py --strategy last --n_drop 13 \
        --out $ALPAMAYO_WEIGHTS_DIR/Alpamayo-R1-10B-pruned-last13
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(r1=True)  # was: sys.path.insert(R1 src)
import argparse
import json
import os
import random
import shutil
import sys

import torch


from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

WEIGHTS_PATH = str(ALPAMAYO_R1_WEIGHTS)
DEVICE = "cuda:0"
N_LAYERS = 36


def load_scores(scores_path: str) -> list[int]:
    """Return layer indices sorted by importance ascending (drop first = index 0)."""
    with open(scores_path) as f:
        data = json.load(f)
    return data["ranked_layers"]


def drop_layers(model, drop_indices: list[int]) -> None:
    """Remove decoder layers at drop_indices in-place."""
    layers = model.vlm.language_model.layers
    keep = sorted(set(range(len(layers))) - set(drop_indices))
    print(f"Keeping layers: {keep}", flush=True)
    print(f"Dropping layers: {sorted(drop_indices)}", flush=True)
    new_layers = torch.nn.ModuleList([layers[i] for i in keep])
    # Update config so the model knows about new depth
    model.vlm.language_model.layers = new_layers
    n_new = len(keep)
    # Patch config attributes
    for cfg_obj in [
        model.vlm.config,
        getattr(model.vlm.config, "text_config", None),
        getattr(model.vlm, "config", None),
    ]:
        if cfg_obj is not None and hasattr(cfg_obj, "num_hidden_layers"):
            cfg_obj.num_hidden_layers = n_new
    print(f"Model now has {n_new} text layers.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_PATH)
    ap.add_argument("--scores", default=None,
                    help="Path to angular_scores_r1.json (required for --strategy angular)")
    ap.add_argument("--strategy", choices=["angular", "last", "random"], default="angular")
    ap.add_argument("--n_drop", type=int, default=13,
                    help="Number of layers to drop (default 13 → ~50%% of text params)")
    ap.add_argument("--out", required=True, help="Output directory for pruned model")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --strategy random")
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    # Determine which layers to drop
    if args.strategy == "angular":
        if not args.scores:
            ap.error("--scores is required for --strategy angular")
        ranked = load_scores(args.scores)
        drop_indices = ranked[: args.n_drop]
    elif args.strategy == "last":
        drop_indices = list(range(N_LAYERS - args.n_drop, N_LAYERS))
    else:  # random
        rng = random.Random(args.seed)
        drop_indices = rng.sample(range(N_LAYERS), args.n_drop)

    print(f"Strategy: {args.strategy}", flush=True)
    print(f"Dropping {args.n_drop}/{N_LAYERS} layers: {sorted(drop_indices)}", flush=True)

    print(f"\nLoading model from {args.weights} ...", flush=True)
    model = AlpamayoR1.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)
    model.eval()

    drop_layers(model, drop_indices)

    os.makedirs(args.out, exist_ok=True)

    # Copy tokenizer / config files from original weights
    for fname in os.listdir(args.weights):
        if fname.endswith((".json", ".tiktoken", ".model", "LICENSE", "README.md")):
            shutil.copy(os.path.join(args.weights, fname), os.path.join(args.out, fname))

    # Save the pruned model using HF save_pretrained
    print(f"\nSaving pruned model to {args.out} ...", flush=True)
    model.save_pretrained(args.out)
    print("Done.", flush=True)

    # Save a record of what was dropped
    meta = {
        "strategy": args.strategy,
        "n_drop": args.n_drop,
        "n_keep": N_LAYERS - args.n_drop,
        "dropped_layers": sorted(drop_indices),
        "source_weights": args.weights,
        "scores_file": args.scores,
    }
    with open(os.path.join(args.out, "pruning_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Pruning metadata saved to {args.out}/pruning_meta.json", flush=True)


if __name__ == "__main__":
    main()
