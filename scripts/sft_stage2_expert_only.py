
"""Stage 2 SFT: physically-pruned VLM + LoRA on VLM(kept) + Expert + joint loss.

Joint loss = VLM CE loss + diffusion flow-matching MSE loss
  - CE loss keeps VLM token generation intact
  - diffusion loss trains Expert for trajectory recovery

Supports single-GPU and DDP (torchrun) via --ddp flag.

Usage (single GPU):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=X python sft_stage2.py \
        --weights .../Alpamayo-1.5-10B-physical-vlm28 \
        --train_samples 2000 --epochs 1 --out_dir .../sft_stage2_test

Usage (DDP, 2 GPUs):
    HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 sft_stage2.py \
        --weights .../Alpamayo-1.5-10B-physical-vlm28 \
        --train_samples 28000 --epochs 3 --out_dir .../sft_stage2_full --ddp
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS,
    add_alpamayo_to_syspath,
)
add_alpamayo_to_syspath(v15=True)  # was: sys.path.insert(1.5 src)
import argparse, os, sys, time, json

import einops
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from nuscenes_sft_dataset import NuScenesSFTDataset
from sft_phase_b import collate, _process_position_ids_qwen
from sft_phase_c import apply_vlm_only_prune


def apply_lora_vlm_expert(model, r=16, alpha=32, dropout=0.05):
    """Expert-only LoRA (NO VLM LoRA). Pruning project finding: Stage 2 v1
    with joint VLM+Expert LoRA regressed below zero-shot — VLM LoRA on
    already-good kept layers added noise. v2 = Expert-only."""
    from peft import LoraConfig, inject_adapter_in_model
    from torch.nn import Linear

    real_targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, Linear):
            continue
        if not name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
            continue
        if name.startswith("expert.layers.") or ".expert.layers." in name:
            real_targets.append(name)

    print(f"[lora] {len(real_targets)} target projections (Expert-only)", flush=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     bias="none", target_modules=real_targets)
    return inject_adapter_in_model(cfg, model)


def train_step(model, processor, sample, device):
    """Joint loss: VLM CE + diffusion flow-matching MSE."""
    frames = sample["image_frames"].to(device)
    frames_flat = frames.flatten(0, 1)
    cam_idx = sample["camera_indices"].to(device)
    messages = helper.create_message(frames=frames_flat, camera_indices=cam_idx)

    num_fut = model.config.tokens_per_future_traj
    placeholder = (
        "<|traj_future_start|>"
        + "<|traj_future|>" * num_fut
        + "<|traj_future_end|>"
    )
    messages[-1]["content"][0]["text"] += placeholder

    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    int_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "position_ids"}
    tokenized_data = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            tokenized_data[k] = v.long().to(device) if (k in int_keys or not v.is_floating_point()) else v.to(device, dtype=torch.bfloat16)
        else:
            tokenized_data[k] = v

    input_ids = tokenized_data.pop("input_ids")
    traj = {
        "ego_history_xyz": sample["ego_history_xyz"].to(device).float().unsqueeze(0),
        "ego_history_rot": sample["ego_history_rot"].to(device).float().unsqueeze(0),
        "ego_future_xyz":  sample["ego_future_xyz"].to(device).float().unsqueeze(0),
        "ego_future_rot":  sample["ego_future_rot"].to(device).float().unsqueeze(0),
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj)
    batch_size = input_ids.shape[0]

    # VLM forward with CE loss
    labels = input_ids.clone()
    vlm_outputs = model.vlm(
        input_ids=input_ids,
        labels=labels,
        use_cache=True,
        **tokenized_data,
    )
    vlm_loss = vlm_outputs.loss

    # Locate <traj_future_start>
    future_start_tok = model.config.traj_token_ids["future_start"]
    fs_positions = (input_ids == future_start_tok).nonzero(as_tuple=False)
    if fs_positions.numel() == 0:
        raise RuntimeError("No <traj_future_start> token found")
    last_fs = fs_positions[-1, 1] + 1

    # Flow-matching diffusion loss
    x_1 = model.action_space.traj_to_action(
        traj_history_xyz=traj["ego_history_xyz"],
        traj_history_rot=traj["ego_history_rot"],
        traj_future_xyz=traj["ego_future_xyz"],
        traj_future_rot=traj["ego_future_rot"],
    ).reshape(-1, *model.action_space.get_action_space_dims())

    proj_dtype = next(model.action_in_proj.parameters()).dtype
    x_0 = torch.randn_like(x_1).to(proj_dtype)
    x_1 = x_1.to(proj_dtype)
    t = torch.rand(x_1.shape[0], device=device, dtype=proj_dtype)
    t_b = t
    while t_b.dim() < x_1.dim():
        t_b = t_b.unsqueeze(-1)
    x_t = (1 - t_b) * x_0 + t_b * x_1
    v_target = x_1 - x_0

    action_embeds = model.action_in_proj(x_t, t)

    kv_cache = vlm_outputs.past_key_values
    kv_cache.crop(last_fs)
    position_ids = _process_position_ids_qwen(
        vlm_outputs, batch_size, action_embeds.shape[1], device
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    expert_outputs = model.expert(
        inputs_embeds=action_embeds,
        position_ids=position_ids,
        past_key_values=kv_cache,
        attention_mask=None,
        use_cache=True,
        **forward_kwargs,
    )
    diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1]:]
    pred = model.action_out_proj(diffusion_out).view(-1, *model.action_space.get_action_space_dims())
    diffusion_loss = ((pred - v_target) ** 2).mean()

    return vlm_loss + diffusion_loss, vlm_loss.detach(), diffusion_loss.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ALPAMAYO_15_WEIGHTS))
    ap.add_argument("--drop_layers_json", required=True, help="ea VLM pruning meta json")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    # DDP setup
    if args.ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        is_main = local_rank == 0
    else:
        device = "cuda:0"
        is_main = True

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)

    if is_main:
        print("[load] Alpamayo 1.5 base + runtime VLM identity bypass ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(device)
    apply_vlm_only_prune(model, args.drop_layers_json)

    for p in model.parameters():
        p.requires_grad = False

    model = apply_lora_vlm_expert(model, r=args.lora_r, alpha=args.lora_alpha)

    if is_main:
        n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in model.parameters())
        print(f"[lora] trainable {n_t/1e6:.2f}M / total {n_tot/1e9:.2f}B ({100*n_t/n_tot:.3f}%)", flush=True)

    processor = helper.get_processor(model.tokenizer)

    if is_main:
        print("[data] loading nuScenes train ...", flush=True)
    ds = NuScenesSFTDataset(split="train", n_samples=args.train_samples)
    if is_main:
        print(f"[data] size {len(ds)}", flush=True)

    sampler = DistributedSampler(ds, shuffle=True) if args.ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=0, collate_fn=collate, drop_last=True)

    if args.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if args.ddp else model
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(loader) // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=args.lr * 0.1)

    model.train()
    global_step = 0
    accum_loss = accum_vlm = accum_diff = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, vlm_l, diff_l = train_step(raw_model, processor, batch, device)
                (loss / args.grad_accum).backward()
                accum_loss += float(loss.detach())
                accum_vlm += float(vlm_l); accum_diff += float(diff_l)

                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                    global_step += 1

                    if is_main and global_step % args.log_every == 0:
                        n = args.grad_accum * args.log_every
                        lr = sched.get_last_lr()[0]
                        tps = (time.time() - t0) / max(1, global_step)
                        print(f"[ep{epoch+1}/{args.epochs}] step {global_step}/{total_steps} "
                              f"loss={accum_loss/n:.4f} "
                              f"(vlm={accum_vlm/n:.4f} diff={accum_diff/n:.4f}) "
                              f"lr={lr:.2e} t/step={tps:.1f}s", flush=True)
                        accum_loss = accum_vlm = accum_diff = 0.0

                    if is_main and global_step % args.save_every == 0:
                        sp = os.path.join(args.out_dir, f"lora_step_{global_step}")
                        raw_model.save_pretrained(sp)
                        print(f"[save] {sp}", flush=True)
            except Exception as e:
                print(f"[skip] {i}: {type(e).__name__}: {e}", flush=True)
                optim.zero_grad(); torch.cuda.empty_cache()

    if is_main:
        sp = os.path.join(args.out_dir, "lora_final")
        raw_model.save_pretrained(sp)
        print(f"[done] {sp}", flush=True)

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
