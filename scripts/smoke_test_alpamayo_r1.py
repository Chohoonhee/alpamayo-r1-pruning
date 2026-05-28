
"""Smoke test: load Alpamayo R1 from local weights and run inference with dummy inputs.

Falls back to R1 because Alpamayo 1.5 requires gated Cosmos-Reason2-8B processor.
Local R1 weights: $ALPAMAYO_R1_WEIGHTS/
"""

from paths import (
    ALPAMAYO_R1_WEIGHTS,
)
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import time
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

WEIGHTS = str(ALPAMAYO_R1_WEIGHTS)


def build_dummy_data(n_cameras: int = 4,
                      num_frames: int = 4,
                      img_h: int = 320, img_w: int = 576,
                      num_history: int = 16, num_future: int = 64) -> dict:
    """Mimic load_physical_aiavdataset output with random tensors."""
    camera_indices = torch.tensor([0, 1, 2, 6][:n_cameras], dtype=torch.int64)
    image_frames = torch.randint(
        0, 255, (n_cameras, num_frames, 3, img_h, img_w), dtype=torch.uint8
    )

    history_xyz = torch.zeros(1, 1, num_history, 3)
    history_xyz[0, 0, :, 0] = torch.linspace(-1.5, 0.0, num_history)
    history_rot = torch.eye(3).expand(1, 1, num_history, 3, 3).clone()

    future_xyz = torch.zeros(1, 1, num_future, 3)
    future_xyz[0, 0, :, 0] = torch.linspace(0.1, 6.4, num_future)
    future_rot = torch.eye(3).expand(1, 1, num_future, 3, 3).clone()

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
    device = "cuda:0"
    print(f"[smoke] loading model from {WEIGHTS} onto {device} ...", flush=True)
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(WEIGHTS, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"[smoke] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    data = build_dummy_data(n_cameras=4, num_frames=4)
    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    print("[smoke] prepared model_inputs", flush=True)

    torch.cuda.manual_seed_all(42)
    print("[smoke] running sample_trajectories_from_data_with_vlm_rollout ...", flush=True)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=1,
            max_generation_length=64,
            return_extra=True,
        )
    print(f"[smoke] inference took {time.time()-t0:.1f}s", flush=True)

    if isinstance(out, tuple):
        pred_xyz = out[0]
        extra = out[-1] if isinstance(out[-1], dict) else None
    else:
        pred_xyz = out; extra = None
    print(f"[smoke] pred_xyz shape: {tuple(pred_xyz.shape)}", flush=True)
    if extra is not None and "cot" in extra:
        try:
            cot0 = extra["cot"][0]
            print("[smoke] CoT[0]:", str(cot0)[:300], flush=True)
        except Exception as e:
            print("[smoke] extra keys:", list(extra.keys()), flush=True)
    print("[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
