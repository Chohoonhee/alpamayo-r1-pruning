"""Stage 2 SAFE FT: minimal trainable params to prevent VLM breakage.

Key insight: prior Stage 2 v2/v3 LoRA approaches break VLM's ability to
generate <traj_future_start> token, even with VLM weights frozen. Hypothesis:
Expert LoRA + diffusion loss subtly degrades end-to-end generation pathway.

SAFE recipe: train ONLY action_in_proj and action_out_proj (tiny MLPs that
translate between trajectory space and embedding space, OUTSIDE the VLM and
Expert decoder layers). This is the smallest possible adjustment to the
trajectory generation pathway.

Usage (DDP, 8 GPUs):
    torchrun --nproc_per_node=8 sft_stage2_safe.py \\
        --drop_layers_json logs/greedy15_navsim_earlystop_meta.json \\
        --train_samples 1000 --epochs 1 --lr 1e-4 --out_dir /path --ddp
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(v15=True)

import argparse, os, sys, time, json

import einops
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from navsim_sft_dataset import NavsimSFTDataset as NuScenesSFTDataset
from sft_phase_b import collate, _process_position_ids_qwen
from sft_phase_c import apply_vlm_only_prune
from sft_stage2 import train_step


def apply_safe_freeze(model):
    """Freeze EVERYTHING except action_in_proj and action_out_proj."""
    for p in model.parameters():
        p.requires_grad = False
    train_count = 0
    for name, p in model.named_parameters():
        if name.startswith("action_in_proj.") or name.startswith("action_out_proj."):
            p.requires_grad = True
            train_count += p.numel()
    return train_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda:0")
        world_size = 1; rank = 0
    is_main = rank == 0

    if is_main:
        print(f"[load] Alpamayo 1.5 base + runtime VLM identity bypass ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(device)

    drop_idx = apply_vlm_only_prune(model, args.drop_layers_json)

    n_t = apply_safe_freeze(model)
    n_tot = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[safe] trainable {n_t/1e6:.2f}M / total {n_tot/1e9:.2f}B ({100*n_t/n_tot:.4f}%)", flush=True)
        print(f"[safe] training ONLY action_in_proj + action_out_proj (no LoRA, no VLM, no Expert layer FT)", flush=True)

    processor = helper.get_processor(model.tokenizer)

    if is_main: print("[data] loading nuScenes train ...", flush=True)
    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
    if is_main: print(f"[data] size {len(ds)}", flush=True)

    sampler = DistributedSampler(ds) if args.ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=2, collate_fn=collate)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

    raw_model = model.module if args.ddp else model
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(loader) // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=args.lr * 0.1)

    global_step = 0
    accum_loss = accum_vlm = accum_diff = 0.0
    n = 0
    for ep in range(args.epochs):
        if sampler: sampler.set_epoch(ep)
        for i, batch in enumerate(loader):
            try:
                loss, vlm_l, diff_l = train_step(raw_model, processor, batch, device)
                (loss / args.grad_accum).backward()
                accum_loss += float(loss.detach()); accum_vlm += float(vlm_l); accum_diff += float(diff_l)
                n += 1
                if (i + 1) % args.grad_accum == 0:
                    optim.step(); sched.step(); optim.zero_grad()
                    global_step += 1
                    if is_main and global_step % args.log_every == 0:
                        cur_lr = sched.get_last_lr()[0]
                        print(f"[ep{ep+1}/{args.epochs}] step {global_step}/{total_steps} "
                              f"loss={accum_loss/n:.4f} (vlm={accum_vlm/n:.4f} diff={accum_diff/n:.4f}) "
                              f"lr={cur_lr:.2e}", flush=True)
                        accum_loss = accum_vlm = accum_diff = 0.0; n = 0
                    if is_main and global_step % args.save_every == 0:
                        path = os.path.join(args.out_dir, f"step_{global_step}")
                        raw_model.save_pretrained(path)
                        print(f"[save] {path}", flush=True)
            except Exception as e:
                print(f"[skip] {type(e).__name__}: {e}", flush=True)
                continue

    if is_main:
        path = os.path.join(args.out_dir, "final")
        raw_model.save_pretrained(path)
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
