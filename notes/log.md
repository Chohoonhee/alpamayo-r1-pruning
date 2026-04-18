# Experiment log

## 2026-04-17 / 2026-04-18

### NAVSIM zero-shot (Alpamayo 1.5)
- Full navtest (12,147 scenes): PDMS **0.6162**
- Sample500 v1 (initial, with temporal camera bug): 0.6990
- Sample500 v2 (temporal fix): **0.7286**
- Sample500 w/ nav_text (driving_command → "Turn left"/"Turn right"/...):
  0.7110  — **slight regression**; Alpamayo 1.5 likely expects richer nav
  strings ("Turn left onto Main St in 40m") than one-word commands.

### NAVSIM sample500 R1 vs 1.5 comparison (num_traj_samples=1)
| Model | PDMS |
|-------|------|
| 1.5 (8 samples)        | 0.7286 |
| 1.5 (1 sample)         | 0.7002 |
| R1 (1 sample)          | **0.4901** |

Delta 1.5→R1 at equal settings: **-0.210**. Largest sub-metric drops for R1:
- `time_to_collision_within_bound` −0.250
- `no_at_fault_collisions` −0.199
- `lane_keeping` −0.128

Interpretation: R1 is trained on Alpamayo's in-domain data with CoT reasoning
and is more fragile to OOD camera/ego-pose conventions (nuScenes/NAVSIM) than
1.5 which has richer multi-modal conditioning.

### Angular distance pruning (R1, 100 nuScenes val samples)
Pipeline:
1. `angular_dist_r1.py` — register forward hooks on each of 36 Qwen3VL text
   decoder layers; compute `1 − cos(h_in, h_out)` averaged over tokens × samples.
2. `prune_r1.py` — drop N layers by the ranked score (or `last` / `random` for
   baselines), save the new weights.

Per-layer angular distance (R1):
```
Layer  0: 0.193
Layer  1: 0.068
Layer  2: 0.007
Layer  3: 0.005
Layer  4: 0.018
Layer  5: 0.004   ← min
Layer  6: 0.006
...
Layer 11: 0.008
Layer 12: 0.019
...
Layer 16: 0.257   ← max (fusion?)
...
Layer 25: 0.013
Layer 26: 0.007
...
Layer 35: 0.071
```

Bottom-13 drop targets: `[2, 3, 5, 6, 7, 8, 9, 10, 11, 25, 26, 27, 28]`
Bottom-18 drop targets (for angular-18): adds `[4, 12, 29, 30, 34]`
(one layer skipped between blocks due to small score differences).

### Zero-shot nuScenes L2 after pruning (100 samples)
| variant  | Keep | Δparams | L2_1s | L2_2s | L2_3s | Avg L2 |
|----------|:----:|:-------:|:-----:|:-----:|:-----:|:------:|
| orig     | 36   | —       | 0.645 | 1.375 | 2.354 | 1.458 |
| angular-13 | 23 | −22.6%  | 0.659 | 1.408 | 2.367 | 1.478 |
| last-13    | 23 | −22.6%  | 0.659 | 1.387 | 2.327 | 1.458 |
| random-13  | 23 | −22.6%  | 0.649 | 1.381 | 2.305 | 1.445 |
| angular-18 | 18 | −31.4%  | 0.662 | 1.413 | 2.380 | **1.485** |

Observations:
- All 3 strategies (angular / last / random) retain zero-shot L2 at 13-drop
  within ±0.02 m of the original. R1 is extraordinarily robust to layer
  removal in zero-shot on OOD data.
- Pushing to 18 layers (50% of text layers, 31.4% total params) still only
  adds +0.027 m — still within noise.
- Random ≈ angular in zero-shot is surprising and suggests the discriminator
  among strategies will be **SFT recovery speed / final quality**, not
  zero-shot degradation.

### Next steps
- SFT each pruned variant on a small nuScenes train subset (100 / 500 / 1000
  samples) with `configs/sft_nuscenes.yaml` + Alpamayo's `train_hf.py`.
- Measure L2 / collision after SFT → the gap between angular vs random
  should widen here if angular truly preserves more informative capacity.
- Compare against full-R1 + SFT baseline (no pruning) for the data-efficiency
  trade-off.
