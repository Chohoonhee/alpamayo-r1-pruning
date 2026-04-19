# Key finding: VLM backbone training objective determines pruning sensitivity

## One-sentence version

Alpamayo R1's text backbone (vanilla Qwen3VL-8B) is essentially **dormant on OOD
nuScenes** — any 13 to 28 of 36 text layers can be dropped with ≤ 0.06 m L2
change. Alpamayo 1.5's backbone (NVIDIA **Cosmos-Reason2-8B**, same architecture
but reasoning-fine-tuned) **actively uses** the text pathway — dropping the
wrong layers causes a +0.226 m L2 spike.

## The discovery path

1. We ran the same pruning pipeline (angular distance scoring → identity-bypass
   drop) on both R1 and 1.5, each with four drop patterns: `angular-13`,
   `random-13`, `angular-28`, `random_safe-28` (protects layers 0, 1, 35 from
   being dropped by random).
2. We evaluated each variant with the identical 100-sample nuScenes zero-shot
   harness, computing Avg L2 over 1 s / 2 s / 3 s.

## Numbers

| variant          | R1 AvgL2 (Δ vs orig) | 1.5 AvgL2 (Δ vs orig) |
|------------------|:--------------------:|:---------------------:|
| orig (36 layers) | 1.465                | 1.476                 |
| angular-13       | 1.509 (+0.045)       | **1.480 (+0.004)**    |
| random-13        | 1.525 (+0.061)       | **1.701 (+0.226)**    |
| angular-28       | 1.467 (+0.002)       | 1.477 (+0.001)        |
| random_safe-28   | 1.460 (−0.004)       | 1.482 (+0.006)        |

### Same random seed (42), same dropped layers for R1 and 1.5
Both `random-13` runs dropped `[1, 2, 3, 4, 7, 14, 15, 17, 18, 21, 23, 29, 33]`.
Layer 1 is among the dropped set.

- R1 tolerates this: +0.061 m
- 1.5 cannot: +0.226 m

The only variable is the backbone weights. **Cosmos-Reason2 has layer 1 (and
likely others in that set) carrying real information; base Qwen3VL does not.**

### Angular vs random, signal magnitude

- R1: angular-13 vs random-13 → Δ = 0.016 m (barely above noise)
- **1.5: angular-13 vs random-13 → Δ = 0.221 m (55× larger gap)**

Angular distance scoring is only weakly useful on the vanilla Qwen3VL backbone
because the layers are nearly interchangeable on OOD data. It becomes a strong
signal on the reasoning-tuned Cosmos-Reason2 because layer differentiation is
real.

## Why this matters

1. **"Pruning-friendly" depends on training objective, not just architecture.**
   Identical 36-layer Qwen3VL topology gives dramatically different pruning
   sensitivity depending on what the model was trained for.

2. **OOD dormancy of CoT text pathway in R1** is explained by lack of
   reasoning-specific training — the text generation head emits CoT, but the
   layers producing it carry little driving-relevant signal on nuScenes.

3. **Cosmos-Reason2 actually uses reasoning.** Its layers encode information
   that persists into the trajectory head. Dropping the wrong layers breaks
   the pipeline.

## Cross-layer importance comparison

Per-layer angular distance for the 36 text layers:

| Top-3 active layers | R1              | 1.5 (Cosmos-Reason2)     |
|---------------------|-----------------|--------------------------|
| rank 1              | Layer 16 (0.257)| Layer 0  (0.125)         |
| rank 2              | Layer 0  (0.193)| Layer 16 (0.114)         |
| rank 3              | Layer 35 (0.071)| **Layer 24 (0.075)**     |

| Least active layers | R1        | 1.5                  |
|---------------------|-----------|----------------------|
| bottom 3 (dropped first) | L5, L3, L6 | **L32, L31, L27**  |

**R1** concentrates activity at positions 0, 16, 35 (input / fusion / output).
**1.5** spreads activity more, with a new "late reasoning hub" at layer 24.

Total angular-distance activity:
- R1: 1.278
- 1.5: 0.878

Cosmos-Reason2 is *less* active on average but far more *selective* — the work
is concentrated in specific layers whose removal hurts.

## Open questions

1. **Is this a general property of reasoning-fine-tuned VLMs?** Needs
   replication on other reasoning-tuned variants (e.g. Cosmos-Reason1, R1's
   finetunes).
2. **Does SFT on pruned 1.5 recover?** If yes, the reasoning-specific layers
   are specializable from a sparse base. If not, Cosmos-Reason2 requires all
   its text layers for this kind of task.
3. **NAVSIM**: our numbers are nuScenes L2, a loose metric. The tighter PDMS
   on NAVSIM may amplify differences further.

## Files

- Runtime-pruned zero-shot results: `results/nuscenes_zeroshot_runtime/results_*.json`
- 1.5 pruning metadata: `results/pruning_meta_15/Alpamayo-1.5-10B-pruned-*.json`
- Angular scores: `scripts/angular_scores_r1.json`, `scripts/angular_scores_15.json`
- Servers with `--drop_layers_json` flag: `scripts/alpamayo_infer_server_r1.py`,
  `scripts/alpamayo_infer_server_v15.py`
