"""Smoke test: load Alpamayo 1.5 from local weights and run inference with dummy inputs."""
import os
import sys
import time

# Pin GPU with most free memory before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper


WEIGHTS = "/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B"


def build_dummy_data(n_cameras: int = 3,
                      num_frames: int = 4,
                      img_h: int = 320, img_w: int = 576,
                      num_history: int = 16, num_future: int = 64) -> dict:
    """Construct a dummy input dict matching load_physical_aiavdataset format.

    Camera slots: 0=cross_left, 1=front_wide, 2=cross_right, 3=rear_left,
                  4=rear_tele, 5=rear_right, 6=front_tele
    """
    camera_indices = torch.tensor([0, 1, 2][:n_cameras], dtype=torch.int64)

    image_frames = torch.randint(
        0, 255, (n_cameras, num_frames, 3, img_h, img_w), dtype=torch.uint8
    )

    # Ego frame-of-reference at t0: start at origin, tiny linear motion
    history_xyz = torch.zeros(1, 1, num_history, 3)
    history_xyz[0, 0, :, 0] = torch.linspace(-1.5, 0.0, num_history)
    history_rot = torch.eye(3).expand(1, 1, num_history, 3, 3).clone()

    future_xyz = torch.zeros(1, 1, num_future, 3)
    future_xyz[0, 0, :, 0] = torch.linspace(0.1, 6.4, num_future)
    future_rot = torch.eye(3).expand(1, 1, num_future, 3, 3).clone()

    # relative_timestamps: 4 frames at 0.1s spacing per camera
    rel_ts = torch.arange(num_frames, dtype=torch.float32) * 0.1
    relative_timestamps = rel_ts.unsqueeze(0).expand(n_cameras, -1).contiguous()
    absolute_timestamps = (relative_timestamps * 1_000_000).to(torch.int64)

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": history_xyz,
        "ego_history_rot": history_rot,
        "ego_future_xyz": future_xyz,
        "ego_future_rot": future_rot,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "t0_us": 5_100_000,
        "clip_id": "dummy",
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] loading model from {WEIGHTS} onto {device} ...")
    t0 = time.time()
    model = Alpamayo1_5.from_pretrained(WEIGHTS, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s")
    print(f"[smoke] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    data = build_dummy_data(n_cameras=3, num_frames=4)
    data = helper.to_device(data, device) if hasattr(helper, "to_device") else {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()
    }

    print("[smoke] input keys:", list(data.keys()))
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    print("[smoke] running sample_trajectories_from_data_with_vlm_rollout ...")
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=64,
            return_extra=True,
        )
    print(f"[smoke] inference took {time.time()-t0:.1f}s")
    # Returns tuple per docstring; unpack defensively
    if isinstance(out, tuple) and len(out) >= 2:
        pred_xyz, pred_rot = out[0], out[1]
        extra = out[-1] if len(out) > 2 and isinstance(out[-1], dict) else None
    else:
        pred_xyz = out; pred_rot = None; extra = None
    print(f"[smoke] pred_xyz shape: {tuple(pred_xyz.shape)}")
    if extra is not None and "cot" in extra:
        print("[smoke] CoT sample[0]:", str(extra["cot"][0])[:200])
    print("[smoke] DONE")


if __name__ == "__main__":
    main()
