
"""nuScenes → Alpamayo R1 SFT dataset adapter.

Produces the same sample dict as Bench2DriveDataset so it can plug
directly into the existing train_hf.py / ReasoningVLA_Trainer pipeline.

Sample format (mirrors bench2drive_dataset.py):
  image_frames:        (N_cam, N_frame, 3, H, W) uint8 tensor
  camera_indices:      (N_cam * N_frame,) int64 tensor
  relative_timestamps: (N_cam, N_frame) float32 (dummy zeros)
  absolute_timestamps: (N_cam, N_frame) int64   (dummy zeros)
  ego_history_xyz:     (1, 16, 3) float32
  ego_history_rot:     (1, 16, 3, 3) float32
  ego_future_xyz:      (1, 64, 3) float32
  ego_future_rot:      (1, 64, 3, 3) float32
  [tokenized_data]:    output of vla_preprocess_func if provided

nuScenes notes:
  - Samples are 2 Hz; history interpolated to 10 Hz (16 steps = 1.6s back)
  - Future GT: next 6 nuScenes samples @ 2 Hz (3s) → interpolate to 10 Hz (64 steps = 6.4s)
    beyond 3s we extrapolate linearly using the last two valid GT points
  - Cameras: FRONT_LEFT (idx=0), FRONT (idx=1), FRONT_RIGHT (idx=2)
  - 4 temporal frames per camera using sweep data (oldest → newest)
"""
from __future__ import annotations

from paths import (
    NUSC_ROOT,
)

import math
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

# nuScenes devkit must be installed in this venv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

NUSC_ROOT = str(NUSC_ROOT)
VERSION = "v1.0-trainval"

# Alpamayo / nuScenes camera slot mapping (same as bench2drive)
CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAM_INDICES = [0, 1, 2]   # Alpamayo slot indices (cross_left, front_wide, cross_right)

NUM_HISTORY = 16
NUM_FUTURE = 64
HISTORY_HZ = 10
FUTURE_HZ = 10
NUSC_HZ = 2
N_TEMPORAL = 4    # temporal frames per camera
CAM_W, CAM_H = 512, 320


# ── Helpers (adapted from nuscenes_zero_shot.py) ──────────────────────────────

def _get_ego_in_world(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ep["translation"]), Quaternion(ep["rotation"])


def _world_to_ego(pos_world, ego_trans, ego_rot):
    return ego_rot.inverse.rotate(pos_world - ego_trans)


def _get_past_history(nusc, sample_token):
    """Return (16,3) xyz and (16,3,3) rotation in t0 ego frame, 10 Hz."""
    ego_t0_trans, ego_t0_rot = _get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]

    sparse_xyz, sparse_yaw, sparse_t = [], [], []
    cur = nusc.get("sample", sample_token)
    dt_nusc = 1.0 / NUSC_HZ
    n_back = int(math.ceil((NUM_HISTORY / HISTORY_HZ) / dt_nusc)) + 2
    for k in range(n_back):
        tr, q = _get_ego_in_world(nusc, cur["token"])
        ego_xyz = _world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        rel_yaw = q.yaw_pitch_roll[0] - t0_yaw
        sparse_xyz.insert(0, [ego_xyz[0], ego_xyz[1], 0.0])
        sparse_yaw.insert(0, rel_yaw)
        sparse_t.insert(0, -k * dt_nusc)
        if not cur["prev"]:
            break
        cur = nusc.get("sample", cur["prev"])

    sparse_xyz = np.array(sparse_xyz, dtype=np.float64)
    sparse_yaw = np.array(sparse_yaw, dtype=np.float64)
    sparse_t = np.array(sparse_t, dtype=np.float64)

    tgt_t = np.arange(-(NUM_HISTORY - 1), 1) * (1.0 / HISTORY_HZ)
    out_xyz = np.zeros((NUM_HISTORY, 3), dtype=np.float32)
    for dim in range(3):
        out_xyz[:, dim] = np.interp(tgt_t, sparse_t, sparse_xyz[:, dim],
                                    left=sparse_xyz[0, dim], right=sparse_xyz[-1, dim])
    out_xyz[-1] = [0, 0, 0]

    yaw_interp = np.interp(tgt_t, sparse_t, sparse_yaw,
                           left=sparse_yaw[0], right=sparse_yaw[-1])
    out_rot = np.zeros((NUM_HISTORY, 3, 3), dtype=np.float32)
    for i in range(NUM_HISTORY):
        y = float(yaw_interp[i])
        out_rot[i] = [[math.cos(y), -math.sin(y), 0],
                      [math.sin(y),  math.cos(y), 0],
                      [0, 0, 1]]
    out_rot[-1] = np.eye(3)
    return out_xyz, out_rot


def _get_future_trajectory(nusc, sample_token, n_future_nusc=6):
    """GT future 64 waypoints @ 10 Hz in t0 ego frame.

    nuScenes only has 6 GT steps at 2 Hz (3s).  We interpolate to 10 Hz up
    to 3s, then linearly extrapolate the last segment to reach 6.4s (64 steps).
    """
    ego_t0_trans, ego_t0_rot = _get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]

    # Collect future samples
    positions = [[0.0, 0.0, 0.0]]  # t=0 (current)
    yaws = [0.0]
    times = [0.0]
    cur = nusc.get("sample", sample_token)
    for k in range(1, n_future_nusc + 1):
        if not cur["next"]:
            break
        cur = nusc.get("sample", cur["next"])
        tr, q = _get_ego_in_world(nusc, cur["token"])
        pos = _world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        rel_yaw = q.yaw_pitch_roll[0] - t0_yaw
        positions.append([float(pos[0]), float(pos[1]), 0.0])
        yaws.append(rel_yaw)
        times.append(k * (1.0 / NUSC_HZ))

    positions = np.array(positions, dtype=np.float64)
    yaws = np.array(yaws, dtype=np.float64)
    times = np.array(times, dtype=np.float64)

    # Target: 64 steps at 10 Hz (0.1 .. 6.4s)
    tgt_t = np.arange(1, NUM_FUTURE + 1) * (1.0 / FUTURE_HZ)
    t_max_gt = times[-1]

    out_xyz = np.zeros((NUM_FUTURE, 3), dtype=np.float32)
    for dim in range(2):   # x, y only; z=0
        out_xyz[:, dim] = np.interp(tgt_t, times, positions[:, dim],
                                    left=positions[0, dim], right=positions[-1, dim])
    # Linear extrapolation beyond GT range
    if t_max_gt < tgt_t[-1] and len(times) >= 2:
        dx = (positions[-1, 0] - positions[-2, 0]) / (times[-1] - times[-2] + 1e-8)
        dy = (positions[-1, 1] - positions[-2, 1]) / (times[-1] - times[-2] + 1e-8)
        for i, t in enumerate(tgt_t):
            if t > t_max_gt:
                dt_ext = t - t_max_gt
                out_xyz[i, 0] = float(positions[-1, 0]) + dx * dt_ext
                out_xyz[i, 1] = float(positions[-1, 1]) + dy * dt_ext

    yaw_interp = np.interp(tgt_t, times, yaws, left=yaws[0], right=yaws[-1])
    out_rot = np.zeros((NUM_FUTURE, 3, 3), dtype=np.float32)
    for i in range(NUM_FUTURE):
        y = float(yaw_interp[i])
        out_rot[i] = [[math.cos(y), -math.sin(y), 0],
                      [math.sin(y),  math.cos(y), 0],
                      [0, 0, 1]]
    return out_xyz, out_rot


def _extract_cams(nusc, sample_token):
    """(N_cam, N_frame, 3, H, W) uint8 array, camera_indices list."""
    sample = nusc.get("sample", sample_token)
    per_cam = []
    for cn in CAM_NAMES:
        cur_sd = nusc.get("sample_data", sample["data"][cn])
        frames = []
        for _ in range(N_TEMPORAL):
            img_path = os.path.join(nusc.dataroot, cur_sd["filename"])
            arr = np.array(Image.open(img_path).resize((CAM_W, CAM_H))).transpose(2, 0, 1).astype(np.uint8)
            frames.insert(0, arr)
            if cur_sd["prev"]:
                cur_sd = nusc.get("sample_data", cur_sd["prev"])
        while len(frames) < N_TEMPORAL:
            frames.insert(0, frames[0].copy())
        per_cam.append(np.stack(frames, axis=0))  # (N_frame, 3, H, W)
    return np.stack(per_cam, axis=0)  # (N_cam, N_frame, 3, H, W)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NuScenesSFTDataset(Dataset):
    """nuScenes train/val → Alpamayo R1 SFT sample dict."""

    def __init__(
        self,
        split: str = "train",
        n_samples: Optional[int] = None,
        nusc_root: str = NUSC_ROOT,
        version: str = VERSION,
        stride: int = 1,
        vla_preprocess_args=None,
        model_config=None,
        **kwargs,
    ):
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        splits = create_splits_scenes()
        scene_names = set(splits.get(split, []))

        tokens = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_names:
                tok = scene["first_sample_token"]
                while tok:
                    tokens.append(tok)
                    tok = self.nusc.get("sample", tok)["next"]

        if stride > 1:
            tokens = tokens[::stride]
        if n_samples is not None:
            stride2 = max(1, len(tokens) // n_samples)
            tokens = tokens[::stride2][:n_samples]

        # Filter out samples that have no future GT
        self.tokens = []
        for tok in tokens:
            s = self.nusc.get("sample", tok)
            if s["next"]:
                self.tokens.append(tok)

        print(f"[NuScenesSFTDataset] split={split}, samples={len(self.tokens)}", flush=True)

        self.vla_preprocess_func = None
        if model_config is not None and vla_preprocess_args is not None:
            try:
                from alpamayo_r1.processor.qwen_processor import get_preprocess_data_fn_from_model_config
                self.vla_preprocess_func = get_preprocess_data_fn_from_model_config(
                    model_config=model_config, **vla_preprocess_args
                )
                print("[NuScenesSFTDataset] VLA preprocessing enabled", flush=True)
            except Exception as e:
                print(f"[NuScenesSFTDataset] VLA preprocessing not available: {e}", flush=True)

        self._error_count = 0

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = self.tokens[idx]
        try:
            image_frames_np = _extract_cams(self.nusc, tok)  # (3,4,3,H,W)
            hist_xyz, hist_rot = _get_past_history(self.nusc, tok)
            fut_xyz, fut_rot = _get_future_trajectory(self.nusc, tok)

            image_frames = torch.from_numpy(image_frames_np)                    # (3,4,3,H,W)
            camera_indices = torch.tensor(CAM_INDICES, dtype=torch.long)        # (3,)
            ego_history_xyz = torch.from_numpy(hist_xyz).unsqueeze(0)           # (1,16,3)
            ego_history_rot = torch.from_numpy(hist_rot).unsqueeze(0)           # (1,16,3,3)
            ego_future_xyz = torch.from_numpy(fut_xyz).unsqueeze(0)             # (1,64,3)
            ego_future_rot = torch.from_numpy(fut_rot).unsqueeze(0)             # (1,64,3,3)

            n_cam, n_frame = image_frames.shape[:2]
            relative_timestamps = torch.zeros(n_cam, n_frame, dtype=torch.float32)
            absolute_timestamps = torch.zeros(n_cam, n_frame, dtype=torch.long)

            sample = {
                "image_frames": image_frames,
                "camera_indices": camera_indices,
                "relative_timestamps": relative_timestamps,
                "absolute_timestamps": absolute_timestamps,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
                "ego_future_xyz": ego_future_xyz,
                "ego_future_rot": ego_future_rot,
            }

            if self.vla_preprocess_func is not None:
                sample["tokenized_data"] = self.vla_preprocess_func(data=sample)

            return sample

        except Exception as e:
            self._error_count += 1
            if self._error_count % 50 == 1:
                print(f"[NuScenesSFTDataset] Error at idx={idx} tok={tok[:8]}: {e}", flush=True)
            next_idx = (idx + 7) % len(self)
            if next_idx == idx:
                next_idx = 0
            return self[next_idx]


if __name__ == "__main__":
    # Quick smoke test
    ds = NuScenesSFTDataset(split="val", n_samples=5)
    sample = ds[0]
    print("Keys:", list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
