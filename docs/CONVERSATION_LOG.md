# Session Log — 2026-04-17 / 2026-04-18

Narrative transcript of the Claude Code session that produced this repo.
Preserved so the research decisions can be reconstructed if data is lost.

---

## Starting point

Prior session had established:
- Alpamayo 1.5 NAVSIM navtest baseline: **PDMS 0.6162** over 12,147 scenes
  (see `results/navsim_sample500/15_navtest_full.csv`)
- Two project repos in parallel:
  - `alpamayo_vipe` — GT-free RL fine-tune via ViPE pseudo-GT
  - `alpamayo_pruning` — **this project**: layer pruning + fast SFT for rapid
    domain adaptation of Alpamayo R1 to OOD benchmarks (nuScenes/NAVSIM)

Infrastructure already set up in previous sessions:
- NAVSIM agent wrapper + ZMQ pickle server (`scripts/alpamayo_navsim_agent.py`,
  `scripts/alpamayo_server.py`) supporting both variant=1.5 and variant=r1
- nuScenes zero-shot harness at `/home/irteam/ws/vipe_test/nuscenes_zero_shot.py`
  (JSON-ZMQ client against Alpamayo infer servers)
- Alpamayo 1.5 venv (`a1_5_venv` inside `alpamayo_pruning/alpamayo1.5/`) for
  the 1.5 stack; conda env `alpamayo_b2d` for R1 + nuScenes tooling

## 1 — Navigation text injection into Alpamayo 1.5 (the nav_text patch)

NAVSIM's `agent_input.ego_statuses[-1].driving_command` is a 4-dim one-hot
[STRAIGHT, LEFT, RIGHT, UTURN]. Alpamayo 1.5 was never getting this signal.
Added a mapping in `scripts/alpamayo_navsim_agent.py`:

```python
_NAV_TEXT_MAP = {0: "Go straight", 1: "Turn left", 2: "Turn right", 3: "Make a U-turn"}
```

and pass-through through the ZMQ request. Server already attempts
`helper.create_message(..., nav_text=...)` with cascading try/except so the
1.5 path uses it.

### Result — 500 scene NAVSIM comparison (Alpamayo 1.5)

| Run                              | PDMS   |
|----------------------------------|--------|
| sample500 v1 (pre temporal fix)  | 0.6990 |
| sample500 v2 (temporal fix)      | **0.7286** |
| sample500 w/ nav_text            | 0.7110 |

Adding nav_text **slightly degraded** performance (−0.018) — main drops:
`lane_keeping` −0.016, `no_at_fault_collisions` −0.009. Hypothesis: Alpamayo
1.5 expects richer navigation instructions ("Turn left onto Main St in 40m")
rather than simple commands, so the coarse mapping occasionally overrides
correct behaviour.

Raw CSVs: `results/navsim_sample500/15_*.csv`.

## 2 — Alpamayo R1 on NAVSIM (same 500 scenes, same settings)

Switched the server variant to R1 and re-ran the 500-scene evaluation with
`num_traj_samples=1` (dropping the default 8 because R1's long CoT caused
GPU-1 OOM when serving four parallel workers with 8 samples each).

For apples-to-apples, also re-ran 1.5 with `num_traj_samples=1`.

| Model                | PDMS   |
|----------------------|--------|
| 1.5 (8 samples)      | 0.7286 |
| 1.5 (1 sample)       | 0.7002 |
| **R1 (1 sample)**    | **0.4901** |

Sampling count change is minor (−0.028). The R1 → 1.5 gap at equal settings
is **−0.210**, largest in:
- `time_to_collision_within_bound` −0.250
- `no_at_fault_collisions` −0.199
- `lane_keeping` −0.128

R1 is markedly weaker on NAVSIM. Interpretation: R1's CoT reasoning overfits
to its native Alpamayo training domain; OOD camera/ego-pose conventions
degrade its collision-avoidance behaviour disproportionately. 1.5 carries
richer multi-modal conditioning (camera_indices, nav_text) and absorbs
domain shift better.

This is the **motivation** for pruning + fast SFT: R1 has the most
headroom for domain adaptation.

## 3 — Angular-distance layer importance (no GT needed)

Implementation in `scripts/angular_dist_r1.py`:

1. Load Alpamayo R1, register forward hooks on each of 36 Qwen3VL text
   decoder layers (path: `model.vlm.language_model.layers`)
2. For N=100 nuScenes val samples, capture `(h_in, h_out)` per layer
3. Score `= 1 − cos(h_in.mean, h_out.mean)` averaged across tokens & samples
4. Rank layers ascending → drop candidates (lower = more redundant)

Full per-layer scores in `scripts/angular_scores_r1.json`. Highlights:

```
Layer  0: 0.193  ← important (post-embedding)
Layer  1: 0.068  ← important
Layers 2-11:   0.004 - 0.018  (mostly residual-pass)
Layer 16:     0.257  ← sharp spike (modality fusion?)
Layers 25-28: 0.007 - 0.011  (also low)
Layer 35:     0.071  ← important (pre-output)
```

Low-score layers form two bands: early transformer block 2-11, and
mid-to-late block 25-28. Layer 16 sits between them as a fusion bottleneck.

## 4 — Pruning strategies (`scripts/prune_r1.py`)

Three strategies, all drop N layers and save a new HF checkpoint:

| Strategy | Selection |
|----------|-----------|
| `angular` | Lowest N scores from `angular_scores_r1.json` |
| `last`    | Last N layers (indices `36-N … 35`) |
| `random`  | Random N with fixed seed (42) |

Seven checkpoints produced (metadata in `results/pruning_meta/`):

| Variant    | # drop | # keep | text-layer drop % | total-param drop % |
|------------|:------:|:------:|:-----------------:|:------------------:|
| angular13  | 13 | 23 | 36% | 22.6% |
| last13     | 13 | 23 | 36% | 22.6% |
| random13   | 13 | 23 | 36% | 22.6% |
| angular18  | 18 | 18 | 50% | 31.4% |
| angular24  | 24 | 12 | 67% | 41.8% |
| angular28  | 28 |  8 | 78% | 48.7% |
| random24   | 24 | 12 | 67% | 41.8% |
| random28   | 28 |  8 | 78% | 48.7% |

## 5 — Zero-shot nuScenes evaluation (100 val samples)

Same harness, same 100 samples, one inference server per variant
(4 GPUs in parallel). Per-variant raw JSON in
`results/nuscenes_zeroshot/results_*.json`.

| Variant    | Keep | L2_1s | L2_2s | L2_3s | **Avg L2** | Δ vs orig |
|------------|:----:|:-----:|:-----:|:-----:|:----------:|:---------:|
| orig       | 36   | 0.645 | 1.375 | 2.354 | **1.458**  | — |
| angular-13 | 23   | 0.659 | 1.408 | 2.367 | 1.478      | +0.020 |
| last-13    | 23   | 0.659 | 1.387 | 2.327 | 1.458      | +0.000 |
| random-13  | 23   | 0.649 | 1.381 | 2.305 | 1.445      | −0.013 |
| angular-18 | 18   | 0.662 | 1.413 | 2.380 | 1.485      | +0.027 |

**Key observation:** at the 13-drop level, **random performed
indistinguishably from angular**, and even slightly better (−0.013 m).
Angular-18 added only +0.027 m over the original. R1's text backbone is
extraordinarily redundant when serving an OOD domain like nuScenes.

### Does angular distance actually pick the right layers?

At the 13-drop level, no — random is within noise. Hypotheses:

1. R1's CoT pathway may be effectively dormant on OOD nuScenes inputs
   (the trajectory head is doing most of the work), so any 13 layers you
   remove leaves enough residual capacity.
2. Redundancy margin is so large that 13 / 36 drops haven't crossed the
   cliff — need aggressive pruning to see the signal.
3. Angular distance is a weaker-than-gradient importance metric.

Launched angular-24/28 + random-24/28 to test hypothesis (2). Results
in progress at time of repo snapshot; see `results/nuscenes_zeroshot/`
for latest JSONs.

## 6 — SFT infrastructure

Created but **not yet run**:
- `scripts/nuscenes_sft_dataset.py` — nuScenes → Alpamayo R1 sample dict,
  mirroring `bench2drive_dataset.py` so it plugs into the existing
  `ReasoningVLA_Trainer` pipeline without modification.
  - Future traj: 6 nuScenes samples (3s @ 2Hz) → linearly extrapolated to
    64 points @ 10Hz to match Alpamayo's 6.4s output horizon.
- `configs/sft_nuscenes.yaml` — Hydra config for `train_hf.py`. Freezes the
  vision encoder (`lr_multiplier.vlm.model.visual: 0.0`), keeps text-layer
  LR full.

Remaining experiment plan:
- SFT each pruned variant + the original at N ∈ {100, 500, 1000} nuScenes
  samples; compare post-SFT L2 / collision. Angular's advantage (if any)
  should emerge here, not zero-shot.

## 7 — Repo snapshot

Everything in this repo:
- `scripts/` — all code
- `configs/` — SFT config
- `results/nuscenes_zeroshot/` — 100-sample JSON results per variant
- `results/navsim_sample500/` — NAVSIM CSVs (1.5, 1.5+nav, R1)
- `results/pruning_meta/` — each pruned model's `pruning_meta.json`
- `scripts/angular_scores_r1.json` — the full 36-layer score vector used
  for deterministic re-pruning

**Not in this repo:** Alpamayo weights (NVIDIA), nuScenes raw data
(trainval ~70 GB), NAVSIM dataset. Paths to those sit in
`docs/REPRODUCE.md`.
