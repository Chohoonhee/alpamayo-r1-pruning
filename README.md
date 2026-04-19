# Alpamayo R1 / 1.5 — Layer Pruning + Backbone Comparison

Research experiments on **pruning NVIDIA Alpamayo VLA models** (R1 &
1.5, each 10 B) via angular-distance layer importance scoring, revealing
that **pruning sensitivity depends on the VLM backbone's training objective**.

**Key finding.** Identical 36-layer Qwen3VL topology, but:
- **R1** (vanilla Qwen3VL base) — text pathway essentially dormant on OOD
  nuScenes. Drop 13 / 28 layers in any way → Avg L2 changes by only ≤ 0.06 m.
- **1.5** (NVIDIA **Cosmos-Reason2-8B** base, same arch, reasoning-fine-tuned)
  — text pathway actively used. Dropping the wrong layer 1 → +0.226 m L2.

Angular-distance scoring is weak signal on R1 but **strongly discriminative
on 1.5** (angular-13 = +0.004 m, random-13 = +0.226 m → 55× gap).

See [docs/KEY_FINDING.md](docs/KEY_FINDING.md) for the full write-up.

## TL;DR

- Alpamayo 1.5 beats R1 on NAVSIM zero-shot by **0.21 PDMS** at equal settings
- Angular-distance scoring flags **layers 2-11 and 25-28** as low-information
- **Dropping 13-18 text layers (22-31% of total params) adds ≤ 0.03 m to
  nuScenes zero-shot L2** — R1's text backbone is dramatically redundant
- At the 13-drop level, **random pruning is indistinguishable from angular**
  — the discriminator will be post-SFT recovery, not zero-shot retention

## Quick results

### NAVSIM 500-scene subset
| Model                | PDMS |
|----------------------|:------:|
| Alpamayo 1.5 (8 samples) | 0.7286 |
| Alpamayo 1.5 (1 sample)  | 0.7002 |
| Alpamayo R1 (1 sample)   | **0.4901** |

### nuScenes zero-shot L2 (100 val samples, runtime identity-bypass pruning)

| variant        | R1 AvgL2 (Δ) | 1.5 AvgL2 (Δ) |
|----------------|:------------:|:-------------:|
| orig (36L)     | 1.465        | 1.476         |
| angular-13     | 1.509 (+0.045) | **1.480 (+0.004)** |
| random-13      | 1.525 (+0.061) | **1.701 (+0.226)** |
| angular-28     | 1.467 (+0.002) | 1.477 (+0.001) |
| random_safe-28 | 1.460 (−0.004) | 1.482 (+0.006) |

### ⚠️ Important bug discovered and corrected

The first pass of results (now in `results/nuscenes_zeroshot/`) used
`prune_r1.py` with `save_pretrained()`. This silently broke pruning: the
checkpoint wrote only the kept layers (re-indexed 0…n-1) but the model
config still said `num_hidden_layers = 36`. `from_pretrained` re-created
the architecture at 36 layers and **randomly initialised** the dropped
positions.

All original "pruned" numbers were therefore `(kept weights + random
decoder layers)`, not a true smaller model. The random layers behaved
roughly as identity → the old numbers showed suspicious preservation at
28-drop. Fixed by switching to **runtime identity-bypass** in the server
(`--drop_layers_json` in `scripts/alpamayo_infer_server_*.py`).

The corrected numbers are the "runtime" row above.

## Repository map

```
.
├── README.md                           ← you are here
├── scripts/
│   ├── angular_dist_r1.py              # score 36 text layers, no GT needed
│   ├── prune_r1.py                     # drop N layers by strategy, save ckpt
│   ├── nuscenes_sft_dataset.py         # nuScenes → Alpamayo SFT sample dict
│   ├── alpamayo_server.py              # NAVSIM pickle-ZMQ inference server
│   ├── alpamayo_navsim_agent.py        # NAVSIM AbstractAgent wrapper
│   ├── run_pruning_experiment.sh       # end-to-end driver
│   ├── run_zeroshot_compare.sh         # 4-variant zero-shot at 13 drop
│   ├── run_zeroshot_aggressive.sh      # 4-variant zero-shot at 24/28 drop
│   ├── run_sample500_15_1sample.sh     # NAVSIM 1.5 1-sample run
│   ├── run_sample500_r1.sh             # NAVSIM R1 1-sample run
│   └── angular_scores_r1.json          # frozen layer-importance scores
├── configs/
│   └── sft_nuscenes.yaml               # Hydra SFT config (plugs into Alpamayo train_hf.py)
├── results/
│   ├── nuscenes_zeroshot/              # results_<variant>.json + eval logs
│   ├── navsim_sample500/               # PDMS CSVs for 1.5 + R1 runs
│   └── pruning_meta/                   # pruning_meta.json per checkpoint
├── docs/
│   ├── CONVERSATION_LOG.md             # session narrative, what happened when
│   ├── EXPERIMENT_NOTES.md             # technical details + design decisions
│   └── REPRODUCE.md                    # step-by-step rebuild instructions
├── notes/log.md                        # short experiment log
└── .gitignore
```

## How to reproduce

See **[docs/REPRODUCE.md](docs/REPRODUCE.md)** for the full step-by-step
guide. Short version:

```bash
# 1. Score layers (no GT needed, ~10 min)
conda activate alpamayo_b2d
python scripts/angular_dist_r1.py \
    --weights /path/to/Alpamayo-R1-10B \
    --n_samples 100 \
    --out scripts/angular_scores_r1.json

# 2. Prune (repeat for strategies / N values)
python scripts/prune_r1.py \
    --scores scripts/angular_scores_r1.json \
    --strategy angular --n_drop 13 \
    --out /path/to/Alpamayo-R1-10B-pruned-angular13

# 3. Zero-shot nuScenes eval (per-variant inference server + zero_shot harness)
bash scripts/run_zeroshot_compare.sh 100

# 4. SFT (config-only, not yet run end-to-end)
cd /path/to/alpamayo-r1-source
python -m finetune.sft.train_hf --config-path finetune/sft/configs \
    --config-name sft_nuscenes \
    model.checkpoint_path=/path/to/pruned/weights
```

## Research questions left open

1. **Does angular distance actually pick better layers than random?**
   The 24- and 28-drop comparisons (in progress at repo snapshot) will
   tell us — `results/nuscenes_zeroshot/results_{angular,random}{24,28}.json`.
2. **Post-SFT recovery curve.** Do pruned variants close the gap to the
   full model with N ∈ {100, 500, 1000} nuScenes training samples? Is
   angular faster than random?
3. **Gradient-based importance** as a stronger signal. Skipped in this
   session in favour of getting the end-to-end pipeline working first.

## Sister project

- **alpamayo_vipe** — GT-free RL fine-tuning via ViPE pseudo-GT
  (complementary: our pruning + small-data SFT approach assumes GT exists
  but is scarce; ViPE trains on unlabelled data by using pseudo-labels).

## License & credits

- Research code MIT-licensed
- Alpamayo weights and benchmarks: NVIDIA (do **not** redistribute weights)
- nuScenes: Aptiv / Motional (CC BY-NC-SA 4.0)
- NAVSIM: University of Tübingen + Motional
