"""Phase C SFT: VLM-only runtime prune + LoRA on remaining VLM + all Expert
+ flow-matching diffusion loss (Alpamayo-native training objective).

Rationale: in our ablations, VLM-only pruning (ea_vlm) strictly beats joint
pruning (jp) at matched compression for zero-shot. This script adds SFT on top
of ea to push compression further while recovering any degradation.

Differences from Phase B:
- drop_target = "vlm" (no Expert drop, Expert is safety-critical)
- LoRA on remaining VLM text layers + ALL Expert layers (both get gradient)
- Everything else (flow-matching training forward, loader, etc.) identical
"""
from __future__ import annotations
import argparse, os, sys, time, json

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/alpamayo1.5/src")
sys.path.insert(0, "/home/irteam/ws/alpamayo_pruning/scripts")

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from nuscenes_sft_dataset import NuScenesSFTDataset
from sft_phase_b import train_step_b, collate  # re-use training step & collate


def apply_vlm_only_prune(model, drop_layers_json: str):
    with open(drop_layers_json) as f:
        meta = json.load(f)
    drop_idx = sorted(set(meta["dropped_layers"]))
    vlm = model.vlm.language_model.layers
    def _id_fwd():
        def _f(hidden_states, *a, **kw):
            return hidden_states
        return _f
    for i in drop_idx:
        vlm[i].forward = _id_fwd()
    print(f"[prune] VLM-only drop {len(drop_idx)}/{len(vlm)}: {drop_idx}", flush=True)
    return drop_idx


def apply_lora_vlm_kept_plus_expert(model, drop_idx, r=8, alpha=16, dropout=0.05):
    """LoRA on remaining (un-pruned) VLM layers + ALL Expert layers."""
    from peft import LoraConfig, inject_adapter_in_model
    dropped = set(drop_idx)
    real_targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
            continue
        if ".language_model.layers." in name:
            try:
                k = int(name.split(".language_model.layers.")[1].split(".")[0])
            except ValueError:
                continue
            if k in dropped:
                continue
            real_targets.append(name)
        elif name.startswith("expert.layers.") or ".expert.layers." in name:
            real_targets.append(name)
    print(f"[lora] {len(real_targets)} target projections (kept VLM + all Expert)", flush=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=real_targets)
    model = inject_adapter_in_model(cfg, model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_layers_json", required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[load] Alpamayo 1.5 ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)

    drop_idx = apply_vlm_only_prune(model, args.drop_layers_json)

    for p in model.parameters():
        p.requires_grad = False

    model = apply_lora_vlm_kept_plus_expert(model, drop_idx,
                                            r=args.lora_r, alpha=args.lora_alpha)

    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_t/1e6:.2f}M / total {n_tot/1e9:.2f}B "
          f"({100*n_t/n_tot:.3f}%)", flush=True)

    processor = helper.get_processor(model.tokenizer)

    print("[data] loading nuScenes train ...", flush=True)
    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
    print(f"[data] size {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(loader) // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps,
                                                       eta_min=args.lr * 0.1)

    model.train()
    global_step = 0
    accum_loss = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = train_step_b(model, processor, batch, args.device)
                (loss / args.grad_accum).backward()
                accum_loss += float(loss.detach())
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    optim.step(); sched.step(); optim.zero_grad()
                    global_step += 1
                    if global_step % args.log_every == 0:
                        lr = sched.get_last_lr()[0]
                        tps = (time.time() - t0) / max(1, global_step)
                        print(f"[epoch {epoch+1}/{args.epochs}] step {global_step}/{total_steps} "
                              f"loss={accum_loss/args.grad_accum/args.log_every:.4f} "
                              f"lr={lr:.2e} t/step={tps:.1f}s", flush=True)
                        accum_loss = 0.0
                    if global_step % args.save_every == 0:
                        sp = os.path.join(args.out_dir, f"lora_step_{global_step}")
                        model.save_pretrained(sp)
                        print(f"[save] {sp}", flush=True)
            except Exception as e:
                print(f"[skip] {i}: {type(e).__name__}: {e}", flush=True)
                optim.zero_grad()
                torch.cuda.empty_cache()
                continue

    sp = os.path.join(args.out_dir, "lora_final")
    model.save_pretrained(sp)
    print(f"[done] {sp}", flush=True)


if __name__ == "__main__":
    main()
