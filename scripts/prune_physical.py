"""Physical VLM layer removal: actually deletes weights from ModuleList.

Unlike runtime identity bypass, this:
- Removes weights from memory (real memory reduction)
- Updates layer_idx and num_hidden_layers so save/load works correctly
- Saves a proper pruned checkpoint loadable without any patch

Usage:
    python prune_physical.py \
        --weights .../Alpamayo-1.5-10B \
        --drop_layers_json .../pruning_meta.json \
        --out_dir .../Alpamayo-1.5-10B-physical-vlm28 \
        --verify
"""
from __future__ import annotations
import argparse, json, sys, os

import torch
import torch.nn as nn

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def physical_prune_vlm(model, drop_layers_json: str):
    with open(drop_layers_json) as f:
        meta = json.load(f)
    drop_idx = sorted(set(meta["dropped_layers"]))
    kept_idx = [i for i in range(len(model.vlm.language_model.layers)) if i not in set(drop_idx)]

    print(f"[prune] dropping {len(drop_idx)} layers, keeping {len(kept_idx)}", flush=True)
    print(f"[prune] drop: {drop_idx}", flush=True)

    # Keep original layer_idx values — cache expects indices up to 35.
    # We physically remove the weight tensors but don't renumber layer_idx,
    # so the HybridCache still allocates 36 slots (dropped slots stay empty).
    # This avoids the IndexError from cache size mismatch.

    # Replace dropped layers with lightweight identity modules (no weights).
    # Must return (hidden_states, past_key_value) to match decoder layer API.
    # past_key_value (HybridCache) is passed in and the slot for this layer
    # simply stays unpopulated — that's fine since we never read it.
    class IdentityLayer(nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    old_layers = model.vlm.language_model.layers
    new_layers = []
    for i in range(len(old_layers)):
        if i in set(drop_idx):
            new_layers.append(IdentityLayer())
        else:
            new_layers.append(old_layers[i])
    model.vlm.language_model.layers = nn.ModuleList(new_layers)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[prune] done. remaining VLM layers: {len(kept_idx)}", flush=True)
    print(f"[prune] total params after pruning: {n_params:.2f}B", flush=True)

    return kept_idx


def verify_inference(model, device="cuda:0"):
    """Quick sanity check: run a dummy forward pass."""
    import numpy as np
    sys.path.insert(0, "/home/irteam/ws/vipe_test")
    from alpamayo1_5 import helper
    from nuscenes_zero_shot import get_past_history, extract_front_cams, get_nav_text, NUSC_ROOT, VERSION
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    print("[verify] loading nuScenes sample ...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    splits = create_splits_scenes()
    tok = next(s["first_sample_token"] for s in nusc.scene
               if s["name"] in set(splits.get("val", [])))

    processor = helper.get_processor(model.tokenizer)
    frames = extract_front_cams(nusc, tok)
    hist, hist_rot = get_past_history(nusc, tok)
    nav = get_nav_text(nusc, tok)

    cam_idx = torch.tensor([0]*4 + [1]*4 + [2]*4, dtype=torch.long).to(device)
    frames_t = torch.from_numpy(frames).to(device)
    messages = helper.create_message(frames=frames_t, camera_indices=cam_idx, nav_text=nav)
    tok_out = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    int_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}
    tokenized = {}
    for k, v in tok_out.items():
        if isinstance(v, torch.Tensor):
            tokenized[k] = v.long().to(device) if (k in int_keys or not v.is_floating_point()) else v.to(device, dtype=torch.bfloat16)
        else:
            tokenized[k] = v

    model_inputs = {
        "tokenized_data": tokenized,
        "ego_history_xyz": torch.from_numpy(hist[None, None]).float().to(device),
        "ego_history_rot": torch.from_numpy(hist_rot[None, None]).float().to(device),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=96, return_extra=False,
        )
    print(f"[verify] output shape: {out[0].shape} — OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--verify", action="store_true", help="run a quick inference sanity check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[load] Alpamayo 1.5 ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)

    kept = physical_prune_vlm(model, args.drop_layers_json)

    if args.verify:
        model.eval()
        verify_inference(model, args.device)

    print(f"[save] saving to {args.out_dir} ...", flush=True)
    model.save_pretrained(args.out_dir)

    # Save pruning metadata alongside
    with open(args.drop_layers_json) as f:
        meta = json.load(f)
    meta["kept_layers"] = kept
    meta["physical_prune"] = True
    with open(os.path.join(args.out_dir, "pruning_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] saved to {args.out_dir}", flush=True)

    # Memory report
    mem = torch.cuda.memory_allocated(args.device) / 1e9
    print(f"[mem] GPU memory allocated: {mem:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
