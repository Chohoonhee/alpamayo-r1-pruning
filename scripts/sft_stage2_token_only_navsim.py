"""Token-only restoration FT.

Hypothesis: All prior FT recipes break PDMS by drifting trajectory generation.
This recipe makes ONLY the LM-head rows for special tokens trainable.
Everything else (VLM, Expert, action_proj) is frozen.

Trainable rows: lm_head.weight[token_id] for token_id in [<|cot_start|>,
<|cot_end|>, <|traj_future_start|>, <|traj_future|>, <|traj_future_end|>].

Goal: restore <|traj_future_start|> generation without touching trajectory quality.
Best case PDMS ceiling = zero-shot pruned ~0.64 (since trajectory un-changed).

Usage (DDP, 8 GPUs):
    torchrun --nproc_per_node=8 sft_stage2_token_only.py \\
        --drop_layers_json logs/greedy15_navsim_earlystop_meta.json \\
        --train_samples 500 --epochs 1 --lr 1e-3 --out_dir /path --ddp
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

SPECIAL_TOKENS = ["<|cot_start|>", "<|cot_end|>", "<|traj_future_start|>",
                  "<|traj_future|>", "<|traj_future_end|>"]


class TokenRowGradMask:
    """Hook that zeros out lm_head.weight gradient for all rows EXCEPT specials."""
    def __init__(self, lm_head_weight: torch.Tensor, allowed_ids: list[int]):
        self.allowed_ids = torch.tensor(allowed_ids, device=lm_head_weight.device)
        self.mask = torch.zeros_like(lm_head_weight)
        self.mask[self.allowed_ids] = 1.0
        self.handle = lm_head_weight.register_hook(self._hook)

    def _hook(self, grad):
        return grad * self.mask


def apply_token_only_freeze(model, tokenizer):
    """Freeze everything. Enable grad ONLY on lm_head + apply row mask via hook."""
    for p in model.parameters():
        p.requires_grad = False

    # find lm_head and embed_tokens
    lm_head = None
    embed = None
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            lm_head = mod
        if name.endswith("embed_tokens"):
            embed = mod
    if lm_head is None:
        raise RuntimeError("lm_head not found")

    allowed_ids = []
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            allowed_ids.append(ids[0])
        else:
            print(f"[warn] tokenizer.encode({tok!r}) -> {ids}; using first id", flush=True)
            allowed_ids.append(ids[0])

    lm_head.weight.requires_grad = True
    mask = TokenRowGradMask(lm_head.weight, allowed_ids)

    return allowed_ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--train_samples", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every", type=int, default=9999)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda:0"); rank = 0
    is_main = rank == 0

    if is_main:
        print(f"[load] Alpamayo 1.5 base + runtime VLM identity bypass ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(device)
    drop_idx = apply_vlm_only_prune(model, args.drop_layers_json)

    allowed_ids, _grad_mask = apply_token_only_freeze(model, model.tokenizer)
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[token-only] trainable {n_t/1e6:.2f}M / total {n_tot/1e9:.2f}B "
              f"({100*n_t/n_tot:.4f}%) row-masked to {len(allowed_ids)} special tokens", flush=True)

    processor = helper.get_processor(model.tokenizer)

    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
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
    total_steps = max(1, args.epochs * len(loader) // args.grad_accum)
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
            except Exception as e:
                print(f"[skip] {type(e).__name__}: {e}", flush=True)
                continue

    if is_main:
        path = os.path.join(args.out_dir, "final")
        raw_model.save_pretrained(path)
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
