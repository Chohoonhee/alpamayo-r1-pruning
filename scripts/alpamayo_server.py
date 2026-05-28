
"""ZMQ REP server: Alpamayo 1.5 (or R1) inference for NAVSIM agent.

Model is selected via ALPAMAYO_VARIANT env var: "1.5" (default) or "r1".
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    ALPAMAYO_R1_WEIGHTS,
)
import argparse
import os
import pickle
import time
import traceback

import numpy as np
import torch
import zmq

VARIANT = os.environ.get("ALPAMAYO_VARIANT", "1.5").lower()
if VARIANT in ("1.5", "a1.5", "alpamayo1.5"):
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5 as ModelCls
    from alpamayo1_5 import helper
    DEFAULT_WEIGHTS = str(ALPAMAYO_15_WEIGHTS)
else:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1 as ModelCls
    from alpamayo_r1 import helper

    DEFAULT_WEIGHTS = str(ALPAMAYO_R1_WEIGHTS)

DEFAULT_PORT = int(os.environ.get("ALPAMAYO_PORT", "5557"))
DEVICE = os.environ.get("ALPAMAYO_DEVICE", "cuda:0")


def run_inference(model, processor, req: dict) -> dict:
    image_frames = torch.from_numpy(req["image_frames"])
    camera_indices = torch.from_numpy(req["camera_indices"])
    ego_xyz = torch.from_numpy(req["ego_history_xyz"]).float()
    ego_rot = torch.from_numpy(req["ego_history_rot"]).float()
    # nav_text: driving command string from NAVSIM (e.g. "Turn left"), empty = not provided
    nav_text = req.get("nav_text", "") or None

    flat_frames = image_frames.flatten(0, 1)
    # Build message: 1.5 supports camera_indices + nav_text; R1 supports neither
    try:
        messages = helper.create_message(flat_frames, camera_indices=camera_indices, nav_text=nav_text)
    except TypeError:
        try:
            messages = helper.create_message(flat_frames, camera_indices=camera_indices)
        except TypeError:
            messages = helper.create_message(flat_frames)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }
    model_inputs = helper.to_device(model_inputs, DEVICE)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=int(req.get("num_traj_samples", 1)),
            max_generation_length=int(req.get("max_generation_length", 256)),
            return_extra=True,
        )
    pred_xyz = out[0].detach().float().cpu().numpy()  # (B,S,N,64,3)
    extra = out[-1] if isinstance(out[-1], dict) else {}
    cot = extra.get("cot", [[""]])
    try:
        cot_str = str(cot[0][0] if isinstance(cot[0], (list, tuple)) else cot[0])
    except Exception:
        cot_str = str(cot)
    pred_xyz_flat = pred_xyz.reshape(-1, pred_xyz.shape[-2], pred_xyz.shape[-1])
    return {
        "ok": True,
        "pred_xyz_list": pred_xyz_flat.tolist(),
        "pred_xyz_shape": list(pred_xyz_flat.shape),
        "cot": cot_str,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    print(f"[server] variant={VARIANT} loading {args.weights} on {DEVICE} ...", flush=True)
    t0 = time.time()
    model = ModelCls.from_pretrained(args.weights, dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print(f"[server] loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[server] ready on tcp://127.0.0.1:{args.port}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except Exception:
            traceback.print_exc()
            sock.send(pickle.dumps({"ok": False, "error": "bad request"}))
            continue
        t0 = time.time()
        try:
            resp = run_inference(model, processor, req)
            resp["latency_s"] = time.time() - t0
            print(f"[server] inference ok {resp['latency_s']:.1f}s", flush=True)
        except Exception as e:
            traceback.print_exc()
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(pickle.dumps(resp))


if __name__ == "__main__":
    main()
