"""Phase A SFT: runtime prune + LoRA on expert + nuScenes full trainset.

Fine-tunes Alpamayo 1.5 after identity-bypassing `drop_layers_json` VLM layers.
Only LoRA adapters on expert decoder are trained (original weights frozen).

Uses next-token loss on trajectory tokens (cheaper and proven path, same as
Alpamayo official TrainableReasoningVLA).

Usage:
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
        /home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python \\
        scripts/sft_phase_a.py \\
        --drop_layers_json .../Alpamayo-1.5-10B-pruned-expertaware_vlm22/pruning_meta.json \\
        --train_samples 28000 \\
        --epochs 3 \\
        --out_dir /home/irteam/ws/alpamayo_pruning/weights/sft_ea_vlm22_fullset
"""
from __future__ import annotations
import argparse, math, os, sys, time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Paths ---
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from nuscenes_sft_dataset import NuScenesSFTDataset

# --- Constants mirrored from base_model for loss computation ---
IGNORE_INDEX = -100


def apply_runtime_prune(model, drop_layers_json: str):
    import json as _json
    with open(drop_layers_json) as f:
        meta = _json.load(f)
    drop_idx = sorted(set(meta["dropped_layers"]))
    target = meta.get("drop_target", "vlm")
    vlm = model.vlm.language_model.layers
    exp = model.expert.layers

    def _id_fwd():
        def _f(hidden_states, *a, **kw):
            return hidden_states
        return _f

    for i in drop_idx:
        if target in ("vlm", "both"):
            vlm[i].forward = _id_fwd()
        if target in ("expert", "both"):
            exp[i].forward = _id_fwd()
    print(f"[prune] target={target} drop {len(drop_idx)}/{len(vlm)}: {drop_idx}", flush=True)


def apply_lora(model, target_module="vlm_text", r=16, alpha=32, dropout=0.05,
               drop_layers_json=None):
    """Attach LoRA adapters to specific linear projections.

    target_module:
      - "vlm_text": self-attn proj of all *kept* VLM text layers (if drop_layers_json
        given, excludes dropped layers; else all 36)
      - "expert": self-attn proj of all expert layers
      - "expert_mlp": mlp proj of expert
    """
    from peft import LoraConfig, inject_adapter_in_model

    # Enumerate module names explicitly to avoid pattern-matching bugs
    real_targets = []
    dropped = set()
    if drop_layers_json is not None:
        import json as _json
        with open(drop_layers_json) as f:
            dropped = set(_json.load(f)["dropped_layers"])

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if target_module == "vlm_text":
            # name like "vlm.model.language_model.layers.5.self_attn.q_proj"
            if ".language_model.layers." in name:
                try:
                    layer_idx = int(name.split(".language_model.layers.")[1].split(".")[0])
                except ValueError:
                    continue
                if layer_idx in dropped:
                    continue
                if name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                    real_targets.append(name)
        elif target_module == "expert":
            if name.startswith("expert.layers.") or ".expert.layers." in name:
                if name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                    real_targets.append(name)
        elif target_module == "expert_mlp":
            if name.startswith("expert.layers.") or ".expert.layers." in name:
                if name.endswith((".gate_proj", ".up_proj", ".down_proj")):
                    real_targets.append(name)
        else:
            raise ValueError(target_module)

    print(f"[lora] found {len(real_targets)} target linear layers ({target_module})",
          flush=True)
    if real_targets:
        print(f"[lora] e.g. {real_targets[0]}, ...,  {real_targets[-1]}", flush=True)

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none",
        target_modules=real_targets,
    )
    model = inject_adapter_in_model(cfg, model)
    return model


def collate(batch):
    """Batch size 1 collate.  Keeps tensors as-is (no batch dim wrapping)
    because the NuScenesSFTDataset already returns in per-sample format
    matching Alpamayo's expected shapes."""
    return batch[0]


def train_step(model, tokenizer, processor, sample, device):
    """One training step with next-token trajectory loss.

    Mirrors TrainableReasoningVLA.forward but minimal.
    """
    # NuScenesSFTDataset returns shapes without batch dim:
    #   image_frames:     (N_cam, N_frame, 3, H, W)
    #   camera_indices:   (N_cam,)
    #   ego_history_xyz:  (1, 16, 3)
    #   ego_history_rot:  (1, 16, 3, 3)
    #   ego_future_xyz:   (1, 64, 3)
    #   ego_future_rot:   (1, 64, 3, 3)
    frames = sample["image_frames"].to(device)            # (N_cam, N_frame, 3, H, W)
    frames_flat = frames.flatten(0, 1)                    # (N_cam*N_frame, 3, H, W)
    cam_idx = sample["camera_indices"].to(device)         # (N_cam,)
    messages = helper.create_message(
        frames=frames_flat,
        camera_indices=cam_idx,
    )
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    int_keys = {"input_ids", "attention_mask", "token_type_ids",
                "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            if k in int_keys or not v.is_floating_point():
                tokenized_data[k] = v.long().to(device)
            else:
                tokenized_data[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            tokenized_data[k] = v

    input_ids = tokenized_data.pop("input_ids")
    # fuse_traj_tokens requires [B, n_traj, T, ...] 4-D / 5-D shape, but the
    # dataset returns (1, T, ...).  Add batch dim -> (1, 1, T, ...).
    def _add_batch(t):
        t = t.to(device).float()
        return t.unsqueeze(0) if t.dim() < (4 if "xyz" in "xyz" else 5) else t
    traj = {
        "ego_history_xyz": sample["ego_history_xyz"].to(device).float().unsqueeze(0),  # (1,1,T,3)
        "ego_history_rot": sample["ego_history_rot"].to(device).float().unsqueeze(0),  # (1,1,T,3,3)
        "ego_future_xyz":  sample["ego_future_xyz"].to(device).float().unsqueeze(0),
        "ego_future_rot":  sample["ego_future_rot"].to(device).float().unsqueeze(0),
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj)

    # 2) Labels: predict trajectory tokens only
    labels = input_ids.clone()

    # 3) VLM forward
    vlm_out = model.vlm(input_ids=input_ids, labels=labels, **tokenized_data)

    # 4) Loss: trajectory next-token only
    traj_start = model.config.traj_token_ids["future_start"]
    traj_end = model.config.traj_token_ids["future_end"]
    traj_vocab = model.config.traj_vocab_size
    traj_offset = model.config.traj_token_start_idx

    # Identify trajectory-related labels
    traj_mask = (
        ((labels >= traj_offset) & (labels < traj_offset + traj_vocab))
        | (labels == traj_start)
        | (labels == traj_end)
    )

    # Compute loss only on trajectory tokens
    logits = vlm_out.logits       # (B, L, V)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = traj_mask[:, 1:].contiguous()
    masked_labels = torch.where(shift_mask, shift_labels, torch.full_like(shift_labels, IGNORE_INDEX))
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        masked_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--lora_target", default="vlm_text",
                    choices=["expert", "expert_mlp", "vlm_text"],
                    help="vlm_text = remaining (un-pruned) VLM layers. This is what "
                         "receives the gradient from VLM next-token loss, so it trains. "
                         "Expert LoRA would require a separate diffusion loss path.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_samples", type=int, default=None,
                    help="None = full nuScenes train set (~28K)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[load] Alpamayo 1.5 ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)
    apply_runtime_prune(model, args.drop_layers_json)
    model.eval()   # freeze dropout; we'll enable training for LoRA below

    # Freeze ALL base weights
    for p in model.parameters():
        p.requires_grad = False

    # Add LoRA (trainable) adapters — pass drop_layers_json so we only put LoRA
    # on kept VLM layers (no point LoRA-ing a pruned layer)
    model = apply_lora(model, target_module=args.lora_target,
                       r=args.lora_r, alpha=args.lora_alpha,
                       drop_layers_json=args.drop_layers_json)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {n_trainable/1e6:.2f}M / total {n_total/1e9:.2f}B "
          f"({100*n_trainable/n_total:.3f}%)", flush=True)

    processor = helper.get_processor(model.tokenizer)

    print("[data] loading nuScenes train ...", flush=True)
    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
    print(f"[data] dataset size: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(loader) // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=args.lr * 0.1)

    print(f"[train] epochs={args.epochs} steps/epoch={len(loader)//args.grad_accum} "
          f"total_steps={total_steps}", flush=True)

    model.train()  # LoRA layers go into train mode; base stays in eval (no_grad anyway)

    global_step = 0
    accum_loss = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = train_step(model, model.tokenizer, processor, batch, args.device)
                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()
                accum_loss += float(loss.detach())
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                    global_step += 1
                    if global_step % args.log_every == 0:
                        lr = sched.get_last_lr()[0]
                        tps = (time.time() - t0) / max(1, global_step)
                        print(f"[epoch {epoch+1}/{args.epochs}] step {global_step}/{total_steps} "
                              f"loss={accum_loss/args.grad_accum/args.log_every:.4f} "
                              f"lr={lr:.2e} t/step={tps:.1f}s", flush=True)
                        accum_loss = 0.0
                    if global_step % args.save_every == 0:
                        save_path = os.path.join(args.out_dir, f"lora_step_{global_step}")
                        model.save_pretrained(save_path)
                        print(f"[save] {save_path}", flush=True)
            except Exception as e:
                print(f"[skip] sample {i}: {type(e).__name__}: {e}", flush=True)
                optim.zero_grad()
                torch.cuda.empty_cache()
                continue

    # Final save
    save_path = os.path.join(args.out_dir, "lora_final")
    model.save_pretrained(save_path)
    print(f"[done] saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
