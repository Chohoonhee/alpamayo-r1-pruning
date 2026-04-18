# Alpamayo R1 — Layer Pruning + Fast Domain Adaptation

Research experiments on **pruning NVIDIA Alpamayo R1** (10 B VLA) via
angular-distance layer importance, and preparing for rapid fine-tuning on
small in-domain data (nuScenes).

**Motivation.** We observed a large OOD gap for Alpamayo R1 on NAVSIM
benchmarks (PDMS 0.49 vs Alpamayo 1.5 at 0.70). R1's text backbone appears
redundant on OOD data, so we ask: can we drop a large fraction of text
layers without catastrophic loss, then recover any gap with a small SFT
pass?

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

### nuScenes zero-shot L2 (100 val samples)
| Variant    | Keep | Δparams | Avg L2 | Δ vs orig |
|------------|:----:|:-------:|:------:|:---------:|
| R1 original | 36 | — | **1.458 m** | — |
| angular-13 | 23 | −22.6 % | 1.478 m | +0.020 |
| last-13    | 23 | −22.6 % | 1.458 m | 0.000 |
| random-13  | 23 | −22.6 % | 1.445 m | −0.013 |
| angular-18 | 18 | −31.4 % | 1.485 m | +0.027 |
| angular-24 | 12 | −41.8 % | 1.443 m | −0.015 |
| random-24  | 12 | −41.8 % | 1.452 m | −0.006 |
| **angular-28** | **8** | **−48.7 %** | **1.455 m** | **−0.003** |
| random-28  |  8 | −48.7 % | 1.465 m | +0.007 |

**Shocking finding:** dropping **28 of 36 text layers (78 %)** changes zero-shot
L2 by −0.003 m. Zero-shot on nuScenes cannot discriminate between pruning
strategies. R1's text pathway appears to contribute almost nothing when the
input domain is OOD — the trajectory head + ego history carries essentially
all the predictive signal. The decisive experiment is SFT recovery on the
target domain, not zero-shot retention.

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
