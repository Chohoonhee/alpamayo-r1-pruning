"""Angular distance scoring for Expert layers on ea-pruned VLM model.

Loads ea_vlm-pruned model, scores Expert layers by angular distance.
Low score = layer barely transforms input = candidate for zero-shot drop.
"""
from __future__ import annotations
import argparse, json, os, sys, math

import numpy as np
import torch

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")
sys.path.insert(0, "/home/irteam/ws/vipe_test")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from sft_phase_c import apply_vlm_only_prune
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes_zero_shot import (
    get_past_history, extract_front_cams, get_nav_text, NUSC_ROOT, VERSION,
)

N_EXPERT_LAYERS = 36


class LayerHook:
    def __init__(self):
        self.h_in = None
        self.h_out = None
        self._handle = None

    def register(self, layer):
        def _hook(module, args, output):
            self.h_in = args[0].detach().float().cpu()
            h_out = output[0] if isinstance(output, tuple) else output
            self.h_out = h_out.detach().float().cpu()
        self._handle = layer.register_forward_hook(_hook)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()


def angular_distance(h_in, h_out):
    h_in_f = h_in.reshape(-1, h_in.shape[-1])
    h_out_f = h_out.reshape(-1, h_out.shape[-1])
    cos = torch.nn.functional.cosine_similarity(h_in_f, h_out_f, dim=-1).clamp(-1, 1)
    return float((1 - cos).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_layers_json", required=True, help="ea VLM pruning meta json")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("[load] model ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)
    apply_vlm_only_prune(model, args.drop_layers_json)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    print("[data] nuScenes val ...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    val_tokens = []
    for scene in nusc.scene:
        if scene["name"] in set(splits.get("val", [])):
            t = scene["first_sample_token"]
            while t:
                val_tokens.append(t)
                t = nusc.get("sample", t)["next"]
    stride = max(1, len(val_tokens) // args.n_samples)
    tokens = val_tokens[::stride][:args.n_samples]

    expert_layers = model.expert.layers
    assert len(expert_layers) == N_EXPERT_LAYERS
    hooks = [LayerHook().register(l) for l in expert_layers]
    accum = [0.0] * N_EXPERT_LAYERS
    count = 0

    cam_idx_np = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)

    try:
        for i, tok in enumerate(tokens):
            try:
                frames = extract_front_cams(nusc, tok)
                hist, hist_rot = get_past_history(nusc, tok)
                nav = get_nav_text(nusc, tok)

                frames_t = torch.from_numpy(frames).to(args.device)
                cam_idx = torch.tensor(cam_idx_np, dtype=torch.long).to(args.device)
                messages = helper.create_message(frames=frames_t, camera_indices=cam_idx, nav_text=nav)
                tok_out = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, return_dict=True, return_tensors="pt",
                )
                int_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}
                tokenized = {}
                for k, v in tok_out.items():
                    if isinstance(v, torch.Tensor):
                        tokenized[k] = v.long().to(args.device) if (k in int_keys or not v.is_floating_point()) else v.to(args.device, dtype=torch.bfloat16)
                    else:
                        tokenized[k] = v

                ego_xyz = torch.from_numpy(hist[None, None]).float().to(args.device)
                ego_rot = torch.from_numpy(hist_rot[None, None]).float().to(args.device)
                model_inputs = {
                    "tokenized_data": tokenized,
                    "ego_history_xyz": ego_xyz,
                    "ego_history_rot": ego_rot,
                }
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs, top_p=0.98, temperature=0.6,
                        num_traj_samples=1, max_generation_length=96, return_extra=False,
                    )
                for j in range(N_EXPERT_LAYERS):
                    if hooks[j].h_in is not None and hooks[j].h_out is not None:
                        accum[j] += angular_distance(hooks[j].h_in, hooks[j].h_out)
                count += 1
                if (i + 1) % 20 == 0:
                    print(f"[{i+1}/{len(tokens)}] count={count}", flush=True)
            except Exception as e:
                print(f"  [skip] {tok[:8]}: {e}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    scores = [accum[j] / count for j in range(N_EXPERT_LAYERS)]
    ranked = sorted(range(N_EXPERT_LAYERS), key=lambda i: scores[i])

    result = {"scores": scores, "ranked_layers": ranked, "n_samples": count,
              "drop_layers_json": args.drop_layers_json}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Expert layer scores (on ea-pruned VLM) ===")
    for j, s in enumerate(scores):
        print(f"  Expert layer {j:2d}: {s:.6f}")
    print(f"\n── Bottom-10 (drop candidates) ──")
    for rank, idx in enumerate(ranked[:10]):
        print(f"  rank {rank+1:2d}: layer {idx:2d}  score={scores[idx]:.6f}")
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
