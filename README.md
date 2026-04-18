# Alpamayo R1 — Layer Pruning + Fast Domain Adaptation

Research experiments on pruning NVIDIA Alpamayo R1 (10B VLA) via angular-distance
layer importance scoring and fine-tuning on small in-domain data (nuScenes).

**Goal:** reduce Alpamayo R1's text backbone parameters without GT labels,
then fine-tune efficiently on a small target-domain dataset, to close the
OOD CoT-Action gap observed between Alpamayo's training distribution and
autonomous driving benchmarks (NAVSIM, nuScenes).

## TL;DR results

- Alpamayo 1.5 outperforms R1 on NAVSIM zero-shot (PDMS 0.70 vs 0.49 at 1 sample/500 scenes)
- R1's text backbone is highly redundant: angular distance scoring reveals 18 of 36 layers
  contribute almost no transformation to the residual stream
- Zero-shot nuScenes L2 is preserved after dropping 18 layers (+0.027m only vs original)
- Random pruning at 13 layers is nearly identical to angular → R1 is surprisingly
  robust to layer drop in zero-shot evaluation; the real differentiator is SFT recovery

## Repository contents

| Path | Purpose |
|------|---------|
| `scripts/angular_dist_r1.py` | Score 36 Qwen3VL text layers via `1 - cos(h_in, h_out)` on 100 nuScenes val samples (no GT needed) |
| `scripts/prune_r1.py` | Drop N layers by strategy (`angular` / `last` / `random`) and save HF checkpoint |
| `scripts/nuscenes_sft_dataset.py` | nuScenes → Alpamayo R1 SFT sample dict (images + ego history + GT future traj) |
| `scripts/alpamayo_server.py` | ZMQ REP (pickle) inference server used by NAVSIM agent |
| `scripts/alpamayo_navsim_agent.py` | NAVSIM `AbstractAgent` wrapper (ZMQ client) with driving_command → nav_text mapping |
| `scripts/run_zeroshot_compare.sh` | Sequential zero-shot eval of 4 R1 variants on nuScenes 100 samples |
| `scripts/run_pruning_experiment.sh` | End-to-end driver: score → prune (3 strategies) → print next SFT commands |
| `scripts/angular_scores_r1.json` | Scored 36-layer importance, drop-ranking for R1 |
| `configs/sft_nuscenes.yaml` | Hydra config for fine-tuning a pruned R1 on nuScenes via Alpamayo's `train_hf.py` |
| `notes/log.md` | Session notes / running PDMS and L2 tables |

## Key findings

### 1. NAVSIM navtest zero-shot (Alpamayo 1.5)
- Full 12,147 scenes: **PDMS 0.6162**
- 500 scene sample (8 traj samples): 0.7286
- 500 scene sample (1 traj sample): 0.7002 → sampling count is minor factor

### 2. NAVSIM sample500 comparison (1 traj sample, same 500 scenes)
| Model | PDMS |
|-------|------|
| 1.5 (8 samples) | 0.7286 |
| 1.5 (1 sample)  | 0.7002 |
| **R1 (1 sample)** | **0.4901** |

R1 shows a large drop in `time_to_collision` (-0.25) and `no_at_fault_collisions`
(-0.20) — evidence of **OOD mismatch** with NAVSIM domain, not a sampling artefact.

### 3. Angular distance layer importance (R1 text layers on nuScenes)
Per-layer `1 - cos(h_in, h_out)`:

```
Layer  0: 0.193  ← important (post-embedding)
Layer  1: 0.068  ← important
Layers 2-11: 0.004-0.008  ← mostly residual-pass
Layer 16: 0.257  ← modality fusion spike
Layers 25-28: 0.007-0.011  ← also low
Layer 35: 0.071  ← important (pre-output)
```

The ranked drop list is saved in `scripts/angular_scores_r1.json`.

### 4. Zero-shot nuScenes L2 (100 val samples)
| Variant | Text layers | Total params Δ | Avg L2 |
|---------|-------------|----------------|--------|
| Original R1 | 36 | — | 1.458 m |
| angular-13 | 23 | -22.6% | 1.478 m |
| last-13    | 23 | -22.6% | 1.458 m |
| random-13  | 23 | -22.6% | 1.445 m |
| **angular-18** | **18** | **-31.4%** | **1.485 m** |

**18 of 36 layers can be dropped with +0.027 m L2 change.** This is a strong
signal that R1's text backbone is heavily redundant in zero-shot on OOD data.

## Reproduction

### Setup
- Angular scoring / pruning / SFT: `conda activate alpamayo_b2d`
- NAVSIM evaluation: use the `navsim_venv` inside `navsim_workspace/navsim/`
- Model weights: Alpamayo-R1-10B (NVIDIA) — not included

### Score + prune
```bash
conda activate alpamayo_b2d

# 1. Score all 36 text layers on 100 nuScenes val samples
python scripts/angular_dist_r1.py \
    --weights /path/to/Alpamayo-R1-10B \
    --n_samples 100 \
    --out scripts/angular_scores_r1.json

# 2. Prune (pick one strategy; "angular" uses the scores file)
python scripts/prune_r1.py \
    --scores scripts/angular_scores_r1.json \
    --strategy angular --n_drop 13 \
    --out /path/to/Alpamayo-R1-10B-pruned-angular13
```

### Zero-shot evaluate
Start one `alpamayo_infer_server` per variant (see `alpamayo_bench2drive`
repo for the server definition — JSON ZMQ protocol on port 5571/5572/...),
then:
```bash
bash scripts/run_zeroshot_compare.sh 100
```

### Fine-tune on nuScenes
The `configs/sft_nuscenes.yaml` plugs into Alpamayo's official `train_hf.py`
pipeline (Hydra + HF Trainer). Override `model.checkpoint_path` to point at
the pruned weights and `data.train_dataset.n_samples` for data efficiency
experiments.

## Sister project

- `alpamayo_pruning` — this repo (pruning + fast SFT)
- `alpamayo_vipe` — GT-free RL fine-tuning via ViPE pseudo-GT

## License

Code here is for research. Alpamayo weights and their accompanying licenses
remain property of NVIDIA. Do not redistribute model weights with this repo.
