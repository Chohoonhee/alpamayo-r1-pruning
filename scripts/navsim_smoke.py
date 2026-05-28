"""NAVSIM smoke test: load one mini scene, run AlpamayoNAVSIMAgent, print trajectory.

Runs in navsim_venv. Assumes alpamayo_server.py is already running on port 5556.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Set NAVSIM env vars before importing navsim
WS = "/home/irteam/ws/alpamayo_pruning/navsim_workspace"
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", f"{WS}/navsim")
os.environ.setdefault("OPENSCENE_DATA_ROOT", f"{WS}/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", f"{WS}/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{WS}/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

# Make scripts importable
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

import numpy as np

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataclasses import SceneFilter


def load_one_scene():
    logs_dir = Path(f"{WS}/dataset/navsim_logs/mini")
    sensor_dir = Path(f"{WS}/dataset/sensor_blobs/mini")
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=8, has_route=False)

    loader = SceneLoader(
        data_path=logs_dir,
        original_sensor_path=sensor_dir,
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True, cam_l0=True, cam_r0=True,
            cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        ),
    )
    tokens = loader.tokens
    print(f"[smoke] scene loader ready: {len(tokens)} tokens", flush=True)
    # find a token whose sensor data is actually present on disk
    for tok in tokens[:50]:
        try:
            agent_input = loader.get_agent_input_from_token(tok)
            return tok, agent_input
        except Exception as e:
            print(f"  skip {tok[:12]}: {type(e).__name__}", flush=True)
            continue
    raise RuntimeError("no scene with available sensor data found in first 50 tokens")


def main():
    from alpamayo_navsim_agent import AlpamayoNAVSIMAgent

    print("[smoke] loading first available NAVSIM mini scene ...", flush=True)
    t0 = time.time()
    tok, agent_input = load_one_scene()
    print(f"[smoke] scene loaded in {time.time()-t0:.1f}s, token={tok[:16]}...",
          flush=True)
    print(f"  num history frames: {len(agent_input.ego_statuses)} / {len(agent_input.cameras)}",
          flush=True)

    agent = AlpamayoNAVSIMAgent(server_addr="tcp://127.0.0.1:5557",
                                num_traj_samples=1, max_generation_length=128)
    agent.initialize()
    print("[smoke] agent connected to server, running inference ...", flush=True)
    t0 = time.time()
    trajectory = agent.compute_trajectory(agent_input)
    print(f"[smoke] inference done in {time.time()-t0:.1f}s", flush=True)
    print(f"[smoke] trajectory shape: {trajectory.poses.shape}", flush=True)
    print(f"[smoke] first 3 poses (x, y, yaw):")
    for p in trajectory.poses[:3]:
        print(f"  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})", flush=True)
    print(f"[smoke] CoT: {getattr(agent, '_last_cot', '')[:200]}", flush=True)
    print("[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
