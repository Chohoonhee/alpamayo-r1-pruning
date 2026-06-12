"""SAFE + masked-placeholder CE with early-stop bracket checkpoints.

Saves at user-chosen optimizer steps (default 8, 16, 30, 60, 120) so the
harness can eval each and pick the best. Avoids running 4 independent
training jobs.

Usage (DDP, 8 GPUs):
    torchrun --nproc_per_node=8 sft_stage2_safe_earlystop.py \
        --drop_layers_json logs/greedy15_navsim_earlystop_meta.json \
        --train_samples 1000 --epochs 1 --lr 1e-5 \
        --save_at_steps 8,16,30,60,120 \
        --out_dir /path --ddp
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
from nuscenes_sft_dataset import NuScenesSFTDataset
from sft_phase_b import collate, _process_position_ids_qwen
from sft_phase_c import apply_vlm_only_prune
from sft_stage2_safe_mask_placeholders import apply_safe_freeze, train_step_masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--train_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--save_at_steps", default="8,16,30,60,120")
    ap.add_argument("--save_every", type=int, default=9999, help="sweep-harness compat; ignored")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    save_steps = sorted({int(x.strip()) for x in args.save_at_steps.split(",") if x.strip()})
    if not save_steps:
        save_steps = [60]
    max_save = max(save_steps)

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
    apply_vlm_only_prune(model, args.drop_layers_json)

    n_t = apply_safe_freeze(model)
    n_tot = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[safe+earlystop] trainable {n_t/1e6:.2f}M / total {n_tot/1e9:.2f}B "
              f"({100*n_t/n_tot:.4f}%) save_at={save_steps}", flush=True)

    processor = helper.get_processor(model.tokenizer)

    if is_main: print("[data] loading nuScenes train ...", flush=True)
    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
    if is_main: print(f"[data] size {len(ds)}", flush=True)

    sampler = DistributedSampler(ds) if args.ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=2, collate_fn=collate, drop_last=True)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

    raw_model = model.module if args.ddp else model
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)  # flat lr; tight bracket

    model.train()
    global_step = 0
    accum_loss = accum_vlm = accum_diff = 0.0
    n = 0
    t0 = time.time()
    done = False
    for ep in range(args.epochs):
        if done: break
        if sampler: sampler.set_epoch(ep)
        for i, batch in enumerate(loader):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, vlm_l, diff_l = train_step_masked(raw_model, processor, batch, device)
                (loss / args.grad_accum).backward()
                accum_loss += float(loss.detach()); accum_vlm += float(vlm_l); accum_diff += float(diff_l)
                n += 1
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                    optim.step(); sched.step(); optim.zero_grad()
                    global_step += 1
                    if is_main and global_step % args.log_every == 0:
                        tps = (time.time() - t0) / max(1, global_step)
                        print(f"[step {global_step}/{max_save}] loss={accum_loss/n:.4f} "
                              f"(vlm={accum_vlm/n:.4f} diff={accum_diff/n:.4f}) "
                              f"t/step={tps:.1f}s", flush=True)
                        accum_loss = accum_vlm = accum_diff = 0.0; n = 0
                    if is_main and global_step in save_steps:
                        path = os.path.join(args.out_dir, f"step_{global_step}")
                        raw_model.save_pretrained(path)
                        print(f"[bracket-save] {path}", flush=True)
                    if global_step >= max_save:
                        done = True
                        break
            except Exception as e:
                print(f"[skip] {i}: {type(e).__name__}: {e}", flush=True)
                optim.zero_grad(); torch.cuda.empty_cache()

    if is_main:
        path = os.path.join(args.out_dir, "final")
        raw_model.save_pretrained(path)
        print(f"[done] {path}", flush=True)

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
