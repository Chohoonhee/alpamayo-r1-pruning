"""NAVSIM trainval SFT dataset.

Mirrors NuScenesSFTDataset interface so existing trainers (sft_stage2_safe.py,
sft_stage2_token_only.py, sft_stage2_expert_lastN_full.py) can swap source data
by changing only the import.

Per-sample returns the same dict shape as NuScenesSFTDataset:
    image_frames     : (3, 4, 3, H, W) uint8 — 3 cams (FL, F, FR), 4 frames hist
    camera_indices   : (3,) long  [0,1,2]
    relative_timestamps : (3,4) float — zeros
    absolute_timestamps : (3,4) long — zeros
    ego_history_xyz  : (1,16,3) — past 16 waypoints in t0 ego frame
    ego_history_rot  : (1,16,3,3)
    ego_future_xyz   : (1,64,3) — future 64 waypoints (6.4s @ 10Hz) in t0 ego frame
    ego_future_rot   : (1,64,3,3)

NAVSIM trainval data is at 10 Hz natively, so no temporal interpolation needed.
"""
from __future__ import annotations

import math
import os
import pickle
from glob import glob

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset


NAVSIM_DATA_ROOT = "/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset"
NAVSIM_TRAINVAL_LOGS = os.path.join(NAVSIM_DATA_ROOT, "navsim_logs", "trainval")
NAVSIM_TRAINVAL_BLOBS = os.path.join(NAVSIM_DATA_ROOT, "sensor_blobs", "trainval")

CAM_NAMES = ["CAM_L0", "CAM_F0", "CAM_R0"]   # left wide, front, right wide
CAM_INDICES = [0, 1, 2]
CAM_W, CAM_H = 512, 320

NUM_HISTORY = 16
NUM_FUTURE = 64
HISTORY_HZ = 10
FUTURE_HZ = 10
NAVSIM_HZ = 10
N_TEMPORAL = 4   # 4 cam frames history at 10Hz native


def _quat_to_yaw(q):
    """q = [w, x, y, z] -> yaw (rad)."""
    return Quaternion(q).yaw_pitch_roll[0]


def _world_to_ego(p_world, ego_translation, ego_rotation_quat):
    """Transform world-frame xyz to ego-frame at t0."""
    rel = np.asarray(p_world, dtype=np.float64) - np.asarray(ego_translation, dtype=np.float64)
    yaw = _quat_to_yaw(ego_rotation_quat)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return rot @ rel


def _extract_traj(frames, i0, t0_trans, t0_rot_q, frame_offsets, default_at_end="extrapolate"):
    """Extract trajectory at given frame_offsets relative to i0 (ego-frame xyz, yaws)."""
    t0_yaw = _quat_to_yaw(t0_rot_q)
    n_frames = len(frames)
    xyz = np.zeros((len(frame_offsets), 3), dtype=np.float32)
    rots = np.zeros((len(frame_offsets), 3, 3), dtype=np.float32)
    for k, off in enumerate(frame_offsets):
        i = i0 + off
        if i < 0:
            i = 0
        if i >= n_frames:
            if default_at_end == "extrapolate" and i0 + frame_offsets[k - 1] < n_frames:
                # Linear extrapolation from last available
                last_off = max(o for o in frame_offsets[:k] if i0 + o < n_frames)
                last_i = i0 + last_off
                prev_i = max(0, last_i - 1)
                pos_last = _world_to_ego(frames[last_i]['ego2global_translation'], t0_trans, t0_rot_q)
                pos_prev = _world_to_ego(frames[prev_i]['ego2global_translation'], t0_trans, t0_rot_q)
                dt = (off - last_off) / NAVSIM_HZ
                vel_dt = 1.0 / NAVSIM_HZ
                dx = (pos_last[0] - pos_prev[0]) / vel_dt
                dy = (pos_last[1] - pos_prev[1]) / vel_dt
                xyz[k] = [float(pos_last[0] + dx * dt), float(pos_last[1] + dy * dt), 0.0]
                # Yaw: keep last yaw
                y = _quat_to_yaw(frames[last_i]['ego2global_rotation']) - t0_yaw
                rots[k] = [[math.cos(y), -math.sin(y), 0],
                           [math.sin(y),  math.cos(y), 0],
                           [0, 0, 1]]
            else:
                # Clamp to last frame
                i = n_frames - 1
        if 0 <= i < n_frames and (default_at_end != "extrapolate" or i < n_frames):
            pos = _world_to_ego(frames[i]['ego2global_translation'], t0_trans, t0_rot_q)
            xyz[k] = [float(pos[0]), float(pos[1]), 0.0]
            y = _quat_to_yaw(frames[i]['ego2global_rotation']) - t0_yaw
            rots[k] = [[math.cos(y), -math.sin(y), 0],
                       [math.sin(y),  math.cos(y), 0],
                       [0, 0, 1]]
    return xyz, rots


def _load_cam_frames(frames, i0):
    """(3, 4, 3, H, W) uint8 cam stack — 3 cameras × 4 history frames."""
    per_cam = []
    for cn in CAM_NAMES:
        cam_imgs = []
        for back in range(N_TEMPORAL - 1, -1, -1):  # newest last in original; insert(0) reversal below
            i = max(0, i0 - back)
            data_path = frames[i]['cams'][cn]['data_path']
            full = os.path.join(NAVSIM_TRAINVAL_BLOBS, data_path)
            try:
                arr = np.array(Image.open(full).resize((CAM_W, CAM_H))).transpose(2, 0, 1).astype(np.uint8)
            except Exception:
                # Image missing — use zeros
                arr = np.zeros((3, CAM_H, CAM_W), dtype=np.uint8)
            cam_imgs.append(arr)
        per_cam.append(np.stack(cam_imgs, axis=0))   # (N_TEMPORAL, 3, H, W)
    return np.stack(per_cam, axis=0)  # (3, 4, 3, H, W)


class NavsimSFTDataset(Dataset):
    def __init__(
        self,
        split: str = "train",   # ignored — uses trainval pkl list
        n_samples: int | None = None,
        vla_preprocess_func=None,
        skip_first_n: int = 16,      # need at least 16 frames of history
        skip_last_n: int = 64,       # need at least 64 future frames
        stride: int = 5,             # subsample frames to reduce density (2Hz)
        nusc_root=None,              # API-compatible w/ NuScenes (ignored)
        version=None,                # API-compatible (ignored)
    ):
        self.vla_preprocess_func = vla_preprocess_func
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n

        # Index: list of (log_pkl_path, frame_idx)
        # Cache loaded pkl per-worker via lazy load
        self._pkl_cache: dict[str, list] = {}

        INDEX_CACHE = os.path.join(os.path.dirname(__file__), "navsim_trainval_index_stride1.json"); import json as _json
        # Load cached index (built once by build_index.py) to avoid loading all pkls
        import json
        index_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navsim_trainval_index_stride1.json")
        with open(index_cache) as f:
            self.index = [(os.path.join(NAVSIM_TRAINVAL_LOGS, fname), i) for fname, i in json.load(f)]
        pkl_files = list({p for p, _ in self.index})

        if n_samples is not None and len(self.index) > n_samples:
            # Deterministic subsample: every k-th
            k = max(1, len(self.index) // n_samples)
            self.index = self.index[::k][:n_samples]

        self._error_count = 0
        print(f"[NavsimSFTDataset] from {len(pkl_files)} logs, samples={len(self.index)}", flush=True)

    def __len__(self):
        return len(self.index)

    def _load_log(self, pkl_path):
        if pkl_path not in self._pkl_cache:
            with open(pkl_path, "rb") as f:
                self._pkl_cache[pkl_path] = pickle.load(f)
            # Keep cache bounded
            if len(self._pkl_cache) > 8:
                # Evict oldest
                old = next(iter(self._pkl_cache))
                if old != pkl_path:
                    del self._pkl_cache[old]
        return self._pkl_cache[pkl_path]

    def __getitem__(self, idx):
        pkl_path, i0 = self.index[idx]
        try:
            frames = self._load_log(pkl_path)
            t0 = frames[i0]
            t0_trans = t0['ego2global_translation']
            t0_rot = t0['ego2global_rotation']

            # Past 16 frames at 10Hz: offsets [-15, -14, ..., 0]
            past_offsets = list(range(-(NUM_HISTORY - 1), 1))
            hist_xyz, hist_rot = _extract_traj(frames, i0, t0_trans, t0_rot, past_offsets, default_at_end="clamp")
            hist_xyz[-1] = [0, 0, 0]
            hist_rot[-1] = np.eye(3, dtype=np.float32)

            # Future 64 frames at 10Hz: offsets [1, 2, ..., 64]
            fut_offsets = list(range(1, NUM_FUTURE + 1))
            fut_xyz, fut_rot = _extract_traj(frames, i0, t0_trans, t0_rot, fut_offsets, default_at_end="extrapolate")

            # Camera frames
            image_frames_np = _load_cam_frames(frames, i0)

            image_frames = torch.from_numpy(image_frames_np)
            camera_indices = torch.tensor(CAM_INDICES, dtype=torch.long)
            ego_history_xyz = torch.from_numpy(hist_xyz).unsqueeze(0)
            ego_history_rot = torch.from_numpy(hist_rot).unsqueeze(0)
            ego_future_xyz = torch.from_numpy(fut_xyz).unsqueeze(0)
            ego_future_rot = torch.from_numpy(fut_rot).unsqueeze(0)

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
                print(f"[NavsimSFTDataset] Error at idx={idx} pkl={os.path.basename(pkl_path)} frame={i0}: {e}", flush=True)
            next_idx = (idx + 7) % len(self)
            if next_idx == idx:
                next_idx = 0
            return self[next_idx]


if __name__ == "__main__":
    ds = NavsimSFTDataset(n_samples=10)
    print("dataset len:", len(ds))
    s = ds[0]
    print("keys:", list(s.keys()))
    for k, v in s.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
