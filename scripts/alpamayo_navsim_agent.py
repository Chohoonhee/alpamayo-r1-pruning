"""NAVSIM AbstractAgent wrapper for Alpamayo R1 via ZMQ client.

Runs in navsim_venv. Sends pickled multi-camera request to Alpamayo server,
receives trajectory + CoT, converts to NAVSIM Trajectory format.

NAVSIM output contract: Trajectory with poses (8 @ 0.5s = 4s horizon) in (x, y, yaw).
Alpamayo outputs 64 waypoints @ 0.1s (6.4s horizon) in (x, y, z).

Conversion: pick every 5th waypoint starting at index 4 → 0.5,1.0,...,4.0s → 8 waypoints.
Heading = atan2(dy, dx) from consecutive waypoints (last heading duplicates previous).
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import zmq

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


# Alpamayo camera slot indices:
# 0=cross_left, 1=front_wide, 2=cross_right, 3=rear_left,
# 4=rear_tele,  5=rear_right, 6=front_tele
# NAVSIM -> Alpamayo mapping (we use 4 slots when all available):
#   CAM_L0 -> 0 (cross_left)
#   CAM_F0 -> 1 (front_wide)
#   CAM_R0 -> 2 (cross_right)
#   center-crop(CAM_F0) -> 6 (front_tele)
ALPAMAYO_CAM_SLOT = {
    "cam_l0": 0,
    "cam_f0": 1,
    "cam_r0": 2,
    "cam_f0_tele": 6,  # synthesized from CAM_F0 center crop
}

TARGET_H, TARGET_W = 320, 576
NUM_FRAMES_PER_CAM = 4
NUM_HISTORY = 16  # Alpamayo expects 16 history waypoints @ 10Hz (1.5s back)

# NAVSIM driving_command: 4-dim one-hot [STRAIGHT, LEFT, RIGHT, UTURN]
_NAV_TEXT_MAP = {0: "Go straight", 1: "Turn left", 2: "Turn right", 3: "Make a U-turn"}


def _image_to_chw(img_hwc: np.ndarray, h: int = TARGET_H, w: int = TARGET_W) -> np.ndarray:
    """Resize HWC uint8 image to (3, h, w) uint8 via simple PIL resize."""
    from PIL import Image as PIL
    if img_hwc.dtype != np.uint8:
        img_hwc = img_hwc.astype(np.uint8)
    pil = PIL.fromarray(img_hwc).resize((w, h), PIL.BILINEAR)
    arr = np.array(pil)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.transpose(2, 0, 1).copy()  # (3, h, w)


def _center_crop_tele(img_hwc: np.ndarray) -> np.ndarray:
    """Synthesize a tele (~30FOV) approximation from a wide frame via center crop."""
    h, w = img_hwc.shape[:2]
    # Crop central 1/3 area, then resize
    ch, cw = h // 3, w // 3
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img_hwc[y0:y0 + ch, x0:x0 + cw]


class AlpamayoFeatureBuilder(AbstractFeatureBuilder):
    """Build multi-camera + ego history tensors matching Alpamayo data format."""

    def __init__(self, use_tele_crop: bool = True):
        self.use_tele_crop = use_tele_crop

    def get_unique_name(self) -> str:
        return "alpamayo_multicam_feature_builder"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cam_hist: List[Any] = agent_input.cameras
        n_hist = min(len(cam_hist), NUM_FRAMES_PER_CAM)
        # Take the latest num_frames
        cam_hist = cam_hist[-n_hist:]
        # Pad with earliest if fewer frames than needed
        while len(cam_hist) < NUM_FRAMES_PER_CAM:
            cam_hist.insert(0, cam_hist[0])

        slots: List[str] = ["cam_l0", "cam_f0", "cam_r0"]
        if self.use_tele_crop:
            slots.append("cam_f0_tele")

        per_slot_frames: Dict[str, List[np.ndarray]] = {s: [] for s in slots}
        for cams in cam_hist:
            l0 = cams.cam_l0.image
            f0 = cams.cam_f0.image
            r0 = cams.cam_r0.image
            per_slot_frames["cam_l0"].append(_image_to_chw(l0))
            per_slot_frames["cam_f0"].append(_image_to_chw(f0))
            per_slot_frames["cam_r0"].append(_image_to_chw(r0))
            if self.use_tele_crop:
                per_slot_frames["cam_f0_tele"].append(_image_to_chw(_center_crop_tele(f0)))

        image_frames = np.stack(
            [np.stack(per_slot_frames[s], axis=0) for s in slots], axis=0
        )  # (N_cam, num_frames, 3, H, W) uint8
        camera_indices = np.array([ALPAMAYO_CAM_SLOT[s] for s in slots], dtype=np.int64)
        sort_order = np.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]

        # Ego history: NAVSIM gives `num_history_frames` at 2Hz, we need 16 at 10Hz.
        # Interpolate 3D (x, y, yaw) and lift to (1,1,16,3) / (1,1,16,3,3).
        statuses = agent_input.ego_statuses  # list, len = num_history_frames
        # Stack in chronological order: oldest -> newest
        raw = np.stack([s.ego_pose for s in statuses], axis=0)  # (K, 3) (x,y,yaw)
        K = raw.shape[0]
        t_src = np.linspace(-(K - 1) * 0.5, 0.0, K)
        t_tgt = np.linspace(-(NUM_HISTORY - 1) * 0.1, 0.0, NUM_HISTORY)
        x_interp = np.interp(t_tgt, t_src, raw[:, 0])
        y_interp = np.interp(t_tgt, t_src, raw[:, 1])
        yaw_interp = np.interp(t_tgt, t_src, np.unwrap(raw[:, 2]))

        # Relative to pose at t0 (newest): origin at (0,0,0), yaw=0
        x0, y0, yaw0 = x_interp[-1], y_interp[-1], yaw_interp[-1]
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        dx = x_interp - x0
        dy = y_interp - y0
        x_loc = c * dx - s * dy
        y_loc = s * dx + c * dy
        yaw_loc = yaw_interp - yaw0
        ego_history_xyz = np.stack([x_loc, y_loc, np.zeros_like(x_loc)], axis=-1).astype(np.float32)
        rots = np.zeros((NUM_HISTORY, 3, 3), dtype=np.float32)
        for i in range(NUM_HISTORY):
            cy, sy = np.cos(yaw_loc[i]), np.sin(yaw_loc[i])
            rots[i] = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ego_history_xyz = ego_history_xyz[None, None]  # (1,1,16,3)
        ego_history_rot = rots[None, None]  # (1,1,16,3,3)

        # Driving command from the most recent frame → navigation text for Alpamayo
        dc = np.asarray(agent_input.ego_statuses[-1].driving_command)
        dc_idx = int(np.argmax(dc)) if dc.sum() > 0 else -1
        nav_text = _NAV_TEXT_MAP.get(dc_idx, "")

        return {
            "image_frames": torch.from_numpy(image_frames),
            "camera_indices": torch.from_numpy(camera_indices),
            "ego_history_xyz": torch.from_numpy(ego_history_xyz),
            "ego_history_rot": torch.from_numpy(ego_history_rot),
            "nav_text": nav_text,  # str, not tensor — passed through as-is
        }


class AlpamayoNAVSIMAgent(AbstractAgent):
    """Wraps Alpamayo via ZMQ. Server must be running separately (alpamayo_server.py)."""

    def __init__(self, server_addr: str = "tcp://127.0.0.1:5556",
                 num_traj_samples: int = 1, max_generation_length: int = 256,
                 timeout_s: int = 300,
                 trajectory_sampling: TrajectorySampling = None):
        if trajectory_sampling is None:
            trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
        super().__init__(trajectory_sampling=trajectory_sampling)
        # Accept comma-separated list of server addresses for round-robin.
        if isinstance(server_addr, str) and "," in server_addr:
            self.server_addrs = [s.strip() for s in server_addr.split(",") if s.strip()]
        else:
            self.server_addrs = [server_addr]
        self.server_addr = self.server_addrs[0]
        self.num_traj_samples = num_traj_samples
        self.max_generation_length = max_generation_length
        self.timeout_s = timeout_s
        self._ctx = None
        self._sock = None
        # Per-worker-instance idx for deterministic assignment
        import os, random
        self._assign_idx = (os.getpid() + random.randint(0, 10000)) % len(self.server_addrs)

    def name(self) -> str:
        return "alpamayo_r1_zmq"

    def get_sensor_config(self) -> SensorConfig:
        # Include F0, L0, R0 only; history=4 frames at 2Hz
        return SensorConfig(
            cam_f0=True, cam_l0=True, cam_r0=True,
            cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        )

    def initialize(self) -> None:
        if self._sock is not None:
            return
        # Assign this worker instance to one server address (sticky round-robin).
        addr = self.server_addrs[self._assign_idx % len(self.server_addrs)]
        self.server_addr = addr
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_s * 1000)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_s * 1000)
        self._sock.connect(addr)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [AlpamayoFeatureBuilder(use_tele_crop=True)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Features come with batch dim = 1. Strip it for the server."""
        nav_text = features.get("nav_text", "")
        if isinstance(nav_text, (list, tuple)):
            nav_text = nav_text[0]
        req = {
            "image_frames": features["image_frames"][0].numpy(),
            "camera_indices": features["camera_indices"][0].numpy(),
            "ego_history_xyz": features["ego_history_xyz"][0].numpy(),
            "ego_history_rot": features["ego_history_rot"][0].numpy(),
            "num_traj_samples": self.num_traj_samples,
            "max_generation_length": self.max_generation_length,
            "nav_text": nav_text,
        }
        if self._sock is None:
            self.initialize()
        self._sock.send(pickle.dumps(req))
        resp = pickle.loads(self._sock.recv())
        if not resp.get("ok"):
            raise RuntimeError(f"Alpamayo server error: {resp.get('error')}")
        pred_xyz = np.asarray(resp["pred_xyz_list"], dtype=np.float32)  # (N, 64, 3)
        # Average across samples to reduce frame-to-frame jitter (comfort metric).
        way64 = pred_xyz.mean(axis=0)  # (64, 3)
        idx = np.arange(4, 64, 5)[:8]  # [4, 9, 14, 19, 24, 29, 34, 39] -> 0.5..4.0s
        xy = way64[idx, :2]
        # heading from finite differences
        dx = np.diff(xy[:, 0], prepend=xy[0, 0])
        dy = np.diff(xy[:, 1], prepend=xy[0, 1])
        yaw = np.arctan2(dy, dx)
        trajectory = np.stack([xy[:, 0], xy[:, 1], yaw], axis=-1).astype(np.float32)
        return {
            "trajectory": torch.from_numpy(trajectory).unsqueeze(0),  # (1, 8, 3)
            "cot": resp.get("cot", ""),
        }

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        # Inline override so we keep CoT accessible via self._last_cot
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
        with torch.no_grad():
            predictions = self.forward(features)
        poses = predictions["trajectory"].squeeze(0).numpy()
        self._last_cot = predictions.get("cot", "")
        return Trajectory(poses)
