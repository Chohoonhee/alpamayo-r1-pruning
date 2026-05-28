"""Phase B SFT: joint VLM + Expert runtime prune + LoRA on both + diffusion loss.

Step-2 of the paper: prove that we can drop BOTH VLM and Expert layers (at the
same paired indices) and recover performance with LoRA adapters on the remaining
components — using a proper diffusion-based training loss (not just next-token
on trajectory tokens).

This is heavier than Phase A but gives a stronger claim:
> "VLM + Expert joint pruning is viable when paired with LoRA compensation"

Design notes:
* Runtime identity-bypass on VLM[k] and Expert[k] for k in drop_pairs
* LoRA adapters on remaining VLM text layers AND remaining Expert layers
* Forward = Alpamayo's training forward (VLM → action_in_proj → expert →
  action_out_proj → flow-matching loss) — mirrors TrainableAlpamayoR1.forward

Usage (same CLI as Phase A):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
        /home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python \\
        scripts/sft_phase_b.py \\
        --drop_pairs_json .../pairs22.json \\
        --train_samples 28000 --epochs 3 --out_dir .../sft_s2_22pair/
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


IGNORE_INDEX = -100


# ── Runtime pruning of both sides ─────────────────────────────────────────────
def apply_joint_prune(model, pairs_json: str):
    """Drop VLM[k] and Expert[k] for every k in `pairs_json['dropped_layers']`."""
    with open(pairs_json) as f:
        meta = json.load(f)
    drop_idx = sorted(set(meta["dropped_layers"]))
    vlm = model.vlm.language_model.layers
    exp = model.expert.layers

    def _id_fwd():
        def _f(hidden_states, *a, **kw):
            return hidden_states
        return _f

    for i in drop_idx:
        vlm[i].forward = _id_fwd()
        exp[i].forward = _id_fwd()
    print(f"[prune] joint drop {len(drop_idx)}/{len(vlm)} pairs: {drop_idx}", flush=True)
    return drop_idx


# ── LoRA on both remaining VLM text layers and remaining Expert layers ────────
def apply_lora_both(model, drop_idx: list[int], r=16, alpha=32, dropout=0.05):
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
            if k not in dropped:
                real_targets.append(name)
        elif name.startswith("expert.layers.") or ".expert.layers." in name:
            try:
                k = int(name.split("expert.layers.")[1].split(".")[0])
            except ValueError:
                continue
            if k not in dropped:
                real_targets.append(name)

    print(f"[lora] {len(real_targets)} target projections", flush=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     bias="none", target_modules=real_targets)
    model = inject_adapter_in_model(cfg, model)
    return model


# ── Phase-B training forward (mirrors TrainableAlpamayoR1.forward) ────────────
def _process_position_ids_qwen(vlm_outputs, batch_size, num_expert_tokens, device):
    position_ids = torch.arange(num_expert_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
    delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
    position_ids += delta.to(position_ids.device)
    return position_ids


def train_step_b(model, processor, sample, device):
    # --- Build prompt ----------------------------------------------------------
    frames = sample["image_frames"].to(device)        # (N_cam, N_frame, 3, H, W)
    frames_flat = frames.flatten(0, 1)                # (N_cam*N_frame, 3, H, W)
    cam_idx = sample["camera_indices"].to(device)     # (N_cam,)
    messages = helper.create_message(frames=frames_flat, camera_indices=cam_idx)

    # For TRAINING, the assistant message needs to include placeholder tokens
    # for the future trajectory so that `fuse_traj_tokens` can see a
    # <|traj_future_start|> ... <|traj_future_end|> block to replace with
    # continuous action embeddings.
    num_fut = model.config.tokens_per_future_traj  # 128
    placeholder = (
        "<|traj_future_start|>"
        + "<|traj_future|>" * num_fut
        + "<|traj_future_end|>"
    )
    # Append after the existing <|cot_start|> in the assistant turn
    messages[-1]["content"][0]["text"] = (
        messages[-1]["content"][0]["text"] + placeholder
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
    traj = {
        "ego_history_xyz": sample["ego_history_xyz"].to(device).float().unsqueeze(0),
        "ego_history_rot": sample["ego_history_rot"].to(device).float().unsqueeze(0),
        "ego_future_xyz":  sample["ego_future_xyz"].to(device).float().unsqueeze(0),
        "ego_future_rot":  sample["ego_future_rot"].to(device).float().unsqueeze(0),
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj)
    batch_size = input_ids.shape[0]

    # --- VLM forward (need cache for expert cross-attn) -----------------------
    vlm_outputs = model.vlm(
        input_ids=input_ids,
        use_cache=True,
        **tokenized_data,
    )

    # --- Locate <traj_future_start> in tokens ---------------------------------
    future_start_tok = model.config.traj_token_ids["future_start"]
    fs_positions = (input_ids == future_start_tok).nonzero(as_tuple=False)
    if fs_positions.numel() == 0:
        raise RuntimeError("No <traj_future_start> token found")
    last_fs = fs_positions[-1, 1] + 1  # position right after

    # --- Prepare noisy future trajectory (flow-matching training) ------------
    # Alpamayo 1.5's FlowMatching exposes only `sample()`. We implement the
    # standard flow-matching training loss manually here:
    #   x_0 ~ N(0, I),  t ~ U(0, 1)
    #   x_t = (1 - t) * x_0 + t * x_1   (x_1 = GT action)
    #   target velocity v* = x_1 - x_0
    #   loss = MSE(model(x_t, t), v*)
    ego_history_xyz = traj["ego_history_xyz"]
    ego_history_rot = traj["ego_history_rot"]
    ego_future_xyz = traj["ego_future_xyz"]
    ego_future_rot = traj["ego_future_rot"]
    x_1 = model.action_space.traj_to_action(
        traj_history_xyz=ego_history_xyz,
        traj_history_rot=ego_history_rot,
        traj_future_xyz=ego_future_xyz,
        traj_future_rot=ego_future_rot,
    )
    x_1 = x_1.reshape(-1, *model.action_space.get_action_space_dims())

    B_flat = x_1.shape[0]
    proj_dtype = next(model.action_in_proj.parameters()).dtype
    x_0 = torch.randn_like(x_1).to(proj_dtype)
    x_1 = x_1.to(proj_dtype)
    t = torch.rand(B_flat, device=device, dtype=proj_dtype)

    # Broadcast t to x_1 shape by unsqueezing trailing dims
    t_b = t
    while t_b.dim() < x_1.dim():
        t_b = t_b.unsqueeze(-1)
    x_t = (1 - t_b) * x_0 + t_b * x_1
    v_target = x_1 - x_0

    action_embeds = model.action_in_proj(x_t, t)

    # --- Expert forward -------------------------------------------------------
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
    pred = model.action_out_proj(diffusion_out)
    pred = pred.view(-1, *model.action_space.get_action_space_dims())

    # --- Flow-matching MSE loss -----------------------------------------------
    loss = ((pred - v_target) ** 2).mean()
    return loss


# ── Dataloader / collate ──────────────────────────────────────────────────────
def collate(batch):
    return batch[0]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B")
    ap.add_argument("--drop_pairs_json", required=True,
                    help="pruning_meta.json listing the layer indices to drop "
                         "from BOTH VLM and Expert (paired drop)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[load] Alpamayo 1.5 ...", flush=True)
    model = Alpamayo1_5.from_pretrained(args.weights, dtype=torch.bfloat16).to(args.device)

    drop_idx = apply_joint_prune(model, args.drop_pairs_json)

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    # LoRA on both remaining sides
    model = apply_lora_both(model, drop_idx=drop_idx, r=args.lora_r, alpha=args.lora_alpha)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_trainable/1e6:.2f}M / total {n_total/1e9:.2f}B "
          f"({100*n_trainable/n_total:.3f}%)", flush=True)

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
