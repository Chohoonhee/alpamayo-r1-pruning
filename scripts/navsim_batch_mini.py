"""Batch-run AlpamayoNAVSIMAgent over N scenes from NAVSIM mini and log results.

Produces `scripts/batch_mini_results.json` with per-scene:
  - token, trajectory (8x3), cot, latency, error(if any)

Also prints a summary: #success, mean latency, trajectory sanity (monotonic x, etc).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

WS = "/home/irteam/ws/alpamayo_pruning/navsim_workspace"
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", f"{WS}/navsim")
os.environ.setdefault("OPENSCENE_DATA_ROOT", f"{WS}/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", f"{WS}/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{WS}/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

import numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig


def build_loader():
    logs_dir = Path(f"{WS}/dataset/navsim_logs/mini")
    sensor_dir = Path(f"{WS}/dataset/sensor_blobs/mini")
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=8, has_route=False)
    return SceneLoader(
        data_path=logs_dir,
        original_sensor_path=sensor_dir,
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True, cam_l0=True, cam_r0=True,
            cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        ),
    )


def trajectory_sanity(traj: np.ndarray) -> dict:
    """Basic trajectory health checks."""
    x = traj[:, 0]
    y = traj[:, 1]
    yaw = traj[:, 2]
    return {
        "x_min": float(x.min()), "x_max": float(x.max()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "x_monotonic_forward": bool(np.all(np.diff(x) >= -0.05)),
        "dx_total": float(x[-1] - x[0]),
        "yaw_max_abs": float(np.max(np.abs(yaw))),
        "any_large_jump": bool(np.max(np.abs(np.diff(x))) > 5.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_scenes", type=int, default=50)
    ap.add_argument("--server", default="tcp://127.0.0.1:5557")
    ap.add_argument("--out", default="/home/irteam/ws/alpamayo_pruning/scripts/batch_mini_results.json")
    args = ap.parse_args()

    from alpamayo_navsim_agent import AlpamayoNAVSIMAgent

    print(f"[batch] loading NAVSIM mini scene loader ...", flush=True)
    loader = build_loader()
    tokens = loader.tokens
    print(f"[batch] {len(tokens)} tokens available, sampling first {args.num_scenes}", flush=True)

    agent = AlpamayoNAVSIMAgent(server_addr=args.server,
                                num_traj_samples=1, max_generation_length=128)
    agent.initialize()

    results = []
    t_start = time.time()
    for i, tok in enumerate(tokens[: args.num_scenes]):
        entry = {"idx": i, "token": tok}
        try:
            ai = loader.get_agent_input_from_token(tok)
            t0 = time.time()
            traj = agent.compute_trajectory(ai)
            entry["latency"] = round(time.time() - t0, 2)
            entry["trajectory"] = traj.poses.tolist()
            entry["cot"] = getattr(agent, "_last_cot", "")
            entry["sanity"] = trajectory_sanity(traj.poses)
            entry["ok"] = True
            print(f"[{i:3d}] ok {entry['latency']:.1f}s | dx={entry['sanity']['dx_total']:+.2f}m "
                  f"| cot={entry['cot'][:80]}", flush=True)
        except Exception as e:
            traceback.print_exc()
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i:3d}] FAIL {entry['error']}", flush=True)
        results.append(entry)

    total_t = time.time() - t_start
    ok_list = [r for r in results if r.get("ok")]
    print(f"\n[batch] {len(ok_list)}/{len(results)} succeeded in {total_t:.1f}s "
          f"(avg {total_t/len(results):.1f}s/scene)", flush=True)
    if ok_list:
        lat = np.array([r["latency"] for r in ok_list])
        dxs = np.array([r["sanity"]["dx_total"] for r in ok_list])
        mono_fwd = sum(1 for r in ok_list if r["sanity"]["x_monotonic_forward"])
        big_jumps = sum(1 for r in ok_list if r["sanity"]["any_large_jump"])
        print(f"[batch] latency: mean={lat.mean():.2f}s p50={np.median(lat):.2f}s p95={np.percentile(lat,95):.2f}s",
              flush=True)
        print(f"[batch] trajectory: dx mean={dxs.mean():+.2f}m, monotonic forward={mono_fwd}/{len(ok_list)}, "
              f"large jumps={big_jumps}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"total_s": total_t, "results": results}, f, indent=2)
    print(f"[batch] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
