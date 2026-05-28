# Alpamayo 1.5 VLM Pruning — Project Notes

NeurIPS 2026 target. Goal: prune Alpamayo 1.5 (10B VLA) via (1) zero-shot VLM layer drop + (2) SFT recovery + (3) Expert layer drop.

---

## Architecture (Alpamayo 1.5)

```
Alpamayo1_5
├── vlm              : Qwen3VL, 36 decoder layers (language_model.layers.0..35)
├── expert           : trajectory decoder, 36 layers (expert.layers.0..35)
├── action_in_proj   : noisy traj -> embed
├── action_out_proj  : embed -> velocity
├── action_space     : traj <-> normalized action
└── diffusion        : FlowMatching (velocity = x1 - x0, MSE loss)
```

Sequence: frames -> VLM -> `<traj_future_start>` -> Expert takes over using VLM KV-cache -> diffusion denoise -> waypoints.

---

## Key Findings

### 1. ea_vlm (Expert-Aware VLM pruning) is the winner
- Score = angular distance of VLM layer output when conditioned by Expert gradients
- **ea_vlm-28 (drop 28/36 VLM layers) zero-shot: L2 1.461 vs baseline 1.465** — 78% VLM param reduction with no loss.
- Angular-24 zero-shot is actually 1.443 (slightly BETTER than baseline). Regularization effect.

### 2. Expert pruning MUST have SFT
- Even drop-8 Expert layers zero-shot hurts: L2 1.672 vs baseline 1.465
- Expert is tightly coupled to VLM, can't be sliced without retraining

### 3. Loss = MSE is correct for flow-matching
- DDPM uses MSE on **noise**; Flow-matching uses MSE on **velocity v = x1 - x0**
- Alpamayo uses flow-matching → our MSE loss matches original

### 4. SFT phase C (diffusion-only) was WRONG
- Only diffusion MSE → VLM lost ability to generate `<traj_future_start>` token
- Result: worse than zero-shot

### 5. Stage 2 (joint CE+diffusion, LoRA on VLM+Expert) also hurt performance
- L2 1.622 > ea_vlm-28 zero-shot 1.461
- Suspected causes: lr=1e-4 too high, VLM LoRA unnecessary (zero-shot already good), overfitting

---

## Pitfalls (Hard-won lessons)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Physical prune breaks on reload | IdentityLayer has no weights → dropped positions reinit on load | Use base model + runtime identity bypass for training |
| HybridCache IndexError after renumbering `layer_idx` | Cache pre-allocates `num_hidden_layers` slots | Keep original `layer_idx`, don't renumber |
| IdentityLayer must return `hidden_states` only | Returning tuple breaks `_deepstack_process` | `def forward(self, h, *a, **kw): return h` |
| `einops`/`peft` missing | Default python has neither | Use `/home/irteam/miniconda/envs/alpamayo_b2d/bin/python` |
| HF gated repo 403 (Cosmos-Reason2-8B) | AutoProcessor tries to fetch config online | `HF_HUB_OFFLINE=1` env var |
| "No `<traj_future_start>` token found" warnings | VLM generation degraded after bad SFT | Joint loss with proper lr; verify CE loss stays low |

---

## File Index

### Scoring (choose what to drop)
- `angular_dist_15.py` — Alpamayo 1.5 VLM angular distance per layer
- `angular_dist_r1.py` — Alpamayo R1 VLM angular distance
- `angular_dist_expert.py` — Expert angular distance on ea-pruned VLM
- `layer_coupling_probe.py` — Expert↔VLM coupling analysis

### Pruning
- `prune_r1.py` — R1 VLM pruning variants (angular / ea / random / last / both)
- `prune_physical.py` — physically remove VLM weights (inference only, DON'T use for training)

### SFT training
- `sft_phase_a.py` — zero-shot pruning pipeline scaffolding
- `sft_phase_b.py` — joint VLM+Expert LoRA + diffusion-only loss (BROKEN)
- `sft_phase_c.py` — VLM-only prune + LoRA on kept VLM + Expert, diffusion-only loss (BROKEN)
- `sft_phase_d.py` — Expert-only LoRA (not used)
- `sft_stage2.py` — ★ current: joint CE + diffusion loss, DDP support

### Evaluation
- `eval_sft_lora.py` — shared `run_inference_trajectory` helper
- `eval_sft_lora_c.py` — eval Phase C checkpoint
- `eval_sft_lora_d.py` — eval Phase D checkpoint
- `eval_sft_stage2.py` — ★ eval Stage 2 checkpoint
- `eval_zeroshot_ea_expert.py` — VLM + Expert zero-shot pruning eval

### Dataset
- `nuscenes_sft_dataset.py` — nuScenes SFT dataset loader
- `nuscenes_zero_shot.py` — eval utils (lives in `/home/irteam/ws/vipe_test/`)

---

## Commands

Env: `/home/irteam/miniconda/envs/alpamayo_b2d/bin/python` (has torch, peft, einops, transformers).

### Zero-shot VLM angular scoring
```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=X \
  /home/irteam/miniconda/envs/alpamayo_b2d/bin/python \
  angular_dist_15.py \
  --weights /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B \
  --n_samples 100 --out angular_scores_15.json
```

### Expert-aware VLM pruning eval (zero-shot)
```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=X \
  /home/irteam/miniconda/envs/alpamayo_b2d/bin/python \
  eval_zeroshot_ea_expert.py \
  --vlm_drop_json .../ea_vlm28/pruning_meta.json \
  --expert_scores_json .../angular_scores_expert_ea28.json \
  --n_expert_drop 0 --n_samples 100 \
  --out_json .../results_ea28.json
```

### Stage 2 SFT (4-GPU DDP, full train)
```bash
cd /home/irteam/ws/alpamayo_pruning/scripts && \
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4,5,1,3 \
  nohup /home/irteam/miniconda/envs/alpamayo_b2d/bin/torchrun \
  --nproc_per_node=4 --master_port=29502 \
  sft_stage2.py \
  --weights /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B \
  --drop_layers_json /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B-pruned-expertaware_vlm28/pruning_meta.json \
  --train_samples 28000 --epochs 3 --ddp \
  --lr 1e-4 --grad_accum 8 \
  --out_dir /home/irteam/ws/alpamayo_pruning/weights/sft_stage2_full \
  > sft_stage2_full_train.log 2>&1 &
```
Runtime: ~9.5s/step × 2559 steps × 3 epochs = **~20 hours on 4 GPUs**.

### Stage 2 eval (single GPU)
```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=X \
  /home/irteam/miniconda/envs/alpamayo_b2d/bin/python \
  eval_sft_stage2.py \
  --orig_weights /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B \
  --drop_layers_json .../ea_vlm28/pruning_meta.json \
  --lora_checkpoint .../sft_stage2_full/lora_final \
  --lora_r 16 --lora_alpha 32 --n_samples 100 \
  --out_json .../eval_stage2_result.json
```

---

## Results Summary

| Method | L2 avg | Col 3s % | Note |
|--------|:-:|:-:|-----|
| Baseline | 1.465 | 2.17 | no prune |
| Angular-24 zs | **1.443** | 2.00 | ★ best zero-shot |
| ea_vlm-28 zs | 1.461 | 2.67 | ★ 78% VLM reduction, lossless |
| ea_vlm28 + Expert-drop8 zs | 1.672 | 3.17 | Expert needs SFT |
| ea_vlm28 + SFT-C (diff only) | 1.575 | 2.50 | missing VLM CE loss |
| ea_vlm28 + Stage2 (joint) | 1.622 | 2.50 | lr too high + VLM LoRA unnecessary |

---

## Next Steps (planned)

1. **Stage 2 v2 (pilot)**: Expert-only LoRA, lr=5e-5, 1 epoch, ~2K samples. Verify direction.
2. **Stage 2 v2 (full)**: if pilot OK, 28K × 2 epoch on 4 GPUs.
3. **Stage 3**: Re-score Expert after Stage 2, drop lowest-scoring Expert layers.
4. **Ablation table for paper**: zero-shot variants × pruning ratios × SFT.

---

## Paths

```
weights/
├── Alpamayo-1.5-10B/                            # base model
├── Alpamayo-1.5-10B-pruned-expertaware_vlm28/   # ea_vlm-28 meta (drop list)
├── Alpamayo-1.5-10B-physical-vlm28/             # physical prune (inference only)
├── sft_stage2_full/                             # Stage 2 LoRA checkpoints
│   ├── lora_step_500, lora_step_1000, ..., lora_final
└── eval_stage2_result.json                      # Stage 2 eval (100 samples)
```

NUSC_ROOT, VERSION are defined in `/home/irteam/ws/vipe_test/nuscenes_zero_shot.py`.
