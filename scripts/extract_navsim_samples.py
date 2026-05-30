"""Extract N NAVSIM scenes into the Alpamayo request format and pickle.

Runs in navsim_venv (Python 3.9). The extracted file is then consumed by
the alpamayo_b2d greedy/eval scripts via a simple pickle load — no NAVSIM
deps needed downstream.

Usage:
    python extract_navsim_samples.py -n 100 --out logs/navsim_samples.pkl
"""
from __future__ import annotations

import os
os.environ.setdefault("NAVSIM_DEVKIT_ROOT",
    "/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim")
os.environ.setdefault("OPENSCENE_DATA_ROOT",
    "/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT",
    "/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT",
    "/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import argparse
import pickle
import sys
from pathlib import Path

# Allow importing the project's NAVSIM agent feature builder
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning_share/scripts")

import numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

from alpamayo_navsim_agent import AlpamayoFeatureBuilder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_scenes", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="mini",
                    help="NAVSIM split: mini or navtest")
    args = ap.parse_args()

    if args.split == "mini":
        logs_dir = Path("/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset/navsim_logs/mini")
        sensor_dir = Path("/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset/sensor_blobs/mini")
    else:
        # navhard_two_stage path may apply; adapt if you need a different split
        logs_dir = Path("/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset/navhard_two_stage")
        sensor_dir = Path("/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset/sensor_blobs/navtest")

    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=8, has_route=False)
    sensor_cfg = SensorConfig(
        cam_f0=True, cam_l0=True, cam_r0=True,
        cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False,
        lidar_pc=False,
    )
    print(f"[load] SceneLoader split={args.split} logs={logs_dir}", flush=True)
    loader = SceneLoader(
        data_path=logs_dir,
        original_sensor_path=sensor_dir,
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )
    tokens = loader.tokens[:args.num_scenes]
    print(f"[load] {len(tokens)} scenes", flush=True)

    fb = AlpamayoFeatureBuilder(use_tele_crop=True)
    extracted = []
    for i, tok in enumerate(tokens):
        try:
            scene = loader.get_scene_from_token(tok)
            agent_input = scene.get_agent_input()
            req = fb.compute_features(agent_input)
            # Convert tensors to numpy for portable pickle
            entry = {
                "token": tok,
                "image_frames": req["image_frames"].numpy(),
                "camera_indices": req["camera_indices"].numpy(),
                "ego_history_xyz": req["ego_history_xyz"].numpy(),
                "ego_history_rot": req["ego_history_rot"].numpy(),
                "nav_text": req["nav_text"],
            }
            extracted.append(entry)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(tokens)}]", flush=True)
        except Exception as e:
            print(f"  [skip {tok[:8]}] {type(e).__name__}: {e}", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(extracted, f)
    sz_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nSaved {len(extracted)} scenes → {args.out} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
