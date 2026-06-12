"""Split navtest's 12146 tokens into N shards, write one Hydra yaml per shard.

Each yaml is a SceneFilter override with a tokens list pinning that shard's scenes.
Used by run_pdms_condition_shard.sh to evaluate one condition in parallel.
"""
import csv
import os
from pathlib import Path

META_CSV = "/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp/metric_cache/metadata/metric_cache_metadata_node_0.csv"
OUT_DIR = Path("/home/irteam/ws/alpamayo_pruning_share/scripts/shard_yamls")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def main(n_shards: int = 8):
    tokens = []
    with open(META_CSV) as f:
        for r in csv.DictReader(f):
            # path: .../metric_cache/<log_name>/unknown/<token>/metric_cache.pkl
            parts = r["file_name"].split("/")
            tokens.append(parts[-2])
    print(f"loaded {len(tokens)} tokens")
    for s in range(n_shards):
        shard_tokens = tokens[s::n_shards]
        path = OUT_DIR / f"navtest_shard{s}of{n_shards}.yaml"
        with open(path, "w") as f:
            f.write("# @package train_test_split.scene_filter\n")
            f.write("_target_: navsim.common.dataclasses.SceneFilter\n")
            f.write("_convert_: 'all'\n")
            f.write("num_history_frames: 4\n")
            f.write("num_future_frames: 10\n")
            f.write("frame_interval: 1\n")
            f.write("has_route: true\n")
            f.write("max_scenes: null\n")
            f.write("log_names: null\n")
            f.write("tokens:\n")
            for t in shard_tokens:
                f.write(f"  - {t}\n")
        print(f"  wrote {path}  n={len(shard_tokens)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(n)
