# Briefing for Future Claude

You're picking up the **Alpamayo VLM Pruning** project on a new machine.
Read this file fully before doing anything. It has the goal, the headline
finding, current state, the active question to answer next, and decisions
that should not be relitigated.

**Also read [PAPER_DIRECTIONS.md](PAPER_DIRECTIONS.md) after this file.**
That document carries forward cross-project synthesis with the sister ViPE
project (`alpamayo-vipe-research`), paper-shape options, and concrete
next-step recommendations that are NOT in CLAUDE.md / KEY_FINDING.md /
NOTES.md. Both repos have an identical copy.

After reading both, confirm you've absorbed the state, surface the active
blocker if relevant, and ask the user what they want to work on. Don't
propose huge plans uninvited.

---

## What this project is

**Layer-pruning Alpamayo VLA models (R1 + 1.5, each ~10B) for fast OOD
domain adaptation.** Hypothesis: a sizeable fraction of the VLM text
backbone is dormant or actively harmful on out-of-domain driving data, and
identifying + removing those layers gives (1) a smaller model and (2) a
faster fine-tune cycle on the new domain.

Standalone from `alpamayo-vipe-research`. **Do not reuse** ViPE pseudo-GT,
VLM judge, or ANCHOR/CORRECTION classification from that project.

Target venue: NeurIPS 2026 / CoRL 2026.

---

## Headline finding (the paper hook)

**VLM backbone training objective determines pruning sensitivity.**

Same 36-layer Qwen3VL topology in R1 and 1.5, but the backbones differ:
R1 uses vanilla Qwen3VL-8B; 1.5 uses NVIDIA Cosmos-Reason2-8B
(reasoning-fine-tuned, same arch).

| variant            | R1 AvgL2 (Δ vs orig) | 1.5 AvgL2 (Δ vs orig) |
|--------------------|:--------------------:|:---------------------:|
| orig (36 layers)   | 1.465                | 1.476                 |
| angular-13         | 1.509 (+0.045)       | **1.480 (+0.004)**    |
| random-13          | 1.525 (+0.061)       | **1.701 (+0.226)**    |
| angular-28         | 1.467 (+0.002)       | 1.477 (+0.001)        |
| random_safe-28     | 1.460 (−0.004)       | 1.482 (+0.006)        |

- R1's text pathway is essentially **dormant on OOD nuScenes** — any 13 of
  36 layers can be dropped (random or angular) for ≤0.06m L2 change.
- 1.5's pathway is **actively used** — dropping the wrong layer 1 spikes
  L2 by +0.226m. Angular scoring discriminates 55× better than random on 1.5
  (+0.004m vs +0.226m at the 13-drop level).

Best zero-shot variants:
- **Angular-24 on R1**: AvgL2 **1.443** (slightly *better* than baseline — regularization).
- **ea_vlm-28 on 1.5**: AvgL2 1.461, **78% VLM param reduction lossless**.

Full table + write-up in [`docs/KEY_FINDING.md`](docs/KEY_FINDING.md).

NAVSIM sample500 (PDMS, higher is better):
- 1.5 8-sample: 0.7286
- 1.5 1-sample: 0.7002
- **R1 1-sample: 0.4901** — R1 generalizes much worse OOD than 1.5.

---

## Current pipeline state

```
  base 1.5 (Cosmos-Reason2)                                  base R1 (Qwen3VL)
        │                                                          │
        ▼ angular distance per VLM layer (angular_dist_15.py)      ▼ same (angular_dist_r1.py)
  angular_scores_15.json                                     angular_scores_r1.json
        │                                                          │
        ▼ expert-aware re-scoring (angular_dist_expert.py)         ▼ —
  expertaware_vlm{13,18,22,26,28,30}/pruning_meta.json
        │
        ▼ runtime identity bypass (eval_zeroshot_ea_expert.py)
  zero-shot results → ea_vlm28 winner (L2 1.461)               [DONE]
        │
        ▼ Stage 2 SFT: joint CE + diffusion, LoRA on VLM + Expert
  sft_stage2_full/lora_final → L2 1.622                  [WORSE than zs ✗]
        │
        ▼ Stage 2 v2 (Expert-only LoRA, lower lr)         [PLANNED, not run]
```

Phases A / B / C / D of SFT were broken in earlier attempts (see
`scripts/NOTES.md`). Stage 2 is the current SFT path.

---

## Active question — start here

**Why does Stage 2 SFT make things worse than zero-shot pruning?**

Stage 2 with `ea_vlm28 + joint CE+diffusion + LoRA on both heads` gives
L2 1.622 vs zero-shot 1.461. Suspected causes (in order):

1. **lr=1e-4 too high** — pilot ran with no warmup. Standard LoRA SFT
   numbers in this regime are 5e-5 or below.
2. **VLM LoRA is unnecessary** — zero-shot already shows the VLM is fine
   after pruning; adding LoRA on the kept VLM layers just disturbs an
   already-good model.
3. **Possible overfit** on the SFT subset (28k samples × 3 epochs).

**Planned next experiment (not yet run):**
- **Stage 2 v2 pilot**: Expert-only LoRA (no VLM LoRA), lr=5e-5,
  1 epoch, ~2K samples. Goal: confirm direction.
- **If pilot recovers ≥ zero-shot**: scale to 28K × 2 epochs on 4 GPUs.
- **Stage 3 (after that)**: re-score Expert layers post-Stage-2 and prune
  the lowest-scoring Expert layers (the Expert head is currently untouched).

---

## Decisions made — do not relitigate

These are settled. If you find yourself proposing one of these alternatives,
stop — they were tried or considered and rejected.

- **Use runtime identity-bypass, not physical pruning, for training.**
  Physical pruning (`prune_physical.py`) reinit dropped positions on reload
  and breaks. For inference-only zero-shot evaluation, physical or runtime
  both work; for SFT, always runtime.
- **Don't renumber `layer_idx` after dropping.** Qwen3VL's `HybridCache`
  pre-allocates `num_hidden_layers` slots; renumbering causes `IndexError`.
  Keep original layer indices, just skip in forward.
- **`IdentityLayer.forward` must return `hidden_states` only.** Returning
  a tuple breaks `_deepstack_process`. Pattern:
  `def forward(self, h, *a, **kw): return h`.
- **Loss = MSE on velocity (flow-matching) is correct.** Alpamayo uses
  flow-matching (v = x1 - x0), so MSE on velocity matches the original
  objective. Don't try to switch to DDPM-style noise MSE.
- **Diffusion-only loss = WRONG.** Phase C tried diffusion-only and broke
  the VLM's ability to generate `<traj_future_start>`. Stage 2 uses
  **joint CE + diffusion**, which is the right setup.
- **Angular distance is the chosen importance metric for v1.** Gradient /
  Taylor scoring is more principled but needs GT + backward pass. Angular
  was the MVP and it works (especially on 1.5).
- **Two backbones to study side-by-side.** R1 (vanilla Qwen3VL) and 1.5
  (Cosmos-Reason2). The contrast is the paper. Don't drop one.

---

## How to behave in this project

- User wants short, direct answers. No tables of options unless asked.
- They've already decided most of the design; don't re-propose alternatives
  unless something genuinely new is on the table.
- Shared GPU server — always `nvidia-smi` for free memory before launching.
  Alpamayo bf16 inference needs ~22GB.
- **Always set `HF_HUB_OFFLINE=1`** for any Alpamayo 1.5 command. Cosmos-Reason2
  is a gated HF repo and the processor tries to fetch online by default → 403.
- Two Python environments — don't mix them:
  - `alpamayo_b2d` (conda, R1 + nuScenes tooling, has peft / einops / transformers)
  - `a1_5_venv` (uv venv inside `alpamayo_pruning/alpamayo1.5/`) — only for
    1.5-source workflows. The `alpamayo_b2d` env can still run 1.5 weights
    if you import paths correctly.
- The Korean user prefers technical jargon in English even within Korean
  sentences. Don't translate `LoRA`, `flow-matching`, `pruning`, `Expert`, etc.

---

## Where things live

| Path | Purpose |
|---|---|
| `/home/irteam/ws/alpamayo_pruning/` | active 1.6 TB working dir (NOT in git) |
| `alpamayo_pruning/scripts/` | all training/eval/scoring scripts + `NOTES.md` |
| `alpamayo_pruning/weights/Alpamayo-{R1,1.5}-10B/` | base models |
| `alpamayo_pruning/weights/Alpamayo-1.5-10B-pruned-*/` | each variant has `pruning_meta.json` |
| `alpamayo_pruning/weights/sft_stage2_full/lora_final` | last Stage 2 LoRA checkpoint (21 GB; also on `gdrive:alpamayo-pruning-artifacts/sft_stage2_full/lora_final`) |
| `alpamayo_pruning/weights/eval_stage2_result.json` | Stage 2 eval (100 nuScenes val samples) |
| `alpamayo_pruning/alpamayo1.5/` | Alpamayo 1.5 source (cloned), has `a1_5_venv` |
| `alpamayo_pruning/navsim_workspace/` | NAVSIM devkit fork (separate Python 3.9 venv) |
| `alpamayo_pruning_share/` | **this directory** — sharing copy of code/docs/configs/results |
| `github.com/Chohoonhee/alpamayo-r1-pruning` | this repo (branch: `master`) |
| `gdrive:alpamayo-conversation-transcripts/` | raw + alpamayo-filtered transcript of the Claude session covering both this project and the ViPE one (for audit / nostalgia — not required to resume) |
| `/home/irteam/ws/vipe_test/nuscenes_zero_shot.py` | eval harness shared with the ViPE project — keep in mind it has the `NUSC_ROOT`/`VERSION` constants |
| `/home/irteam/ws/nuscenes/raw_extracted/` | nuScenes data |

[REPRODUCE.md](docs/REPRODUCE.md) has the full install + dataset + run
commands. [PATHS.md](PATHS.md) documents the env-var contract — every
script in `scripts/` reads paths from there via the local `paths.py`
helper, so no `/home/irteam/...` is hardcoded any more. Run
`python scripts/paths.py` on a fresh machine to verify your env vars
before launching anything.

---

## Glossary, quickly

- **Alpamayo R1** (renamed to "Alpamayo 1"): NVIDIA driving VLA. VLM
  (Qwen3VL-8B, 36 layers) + Expert action head (36 layers) + flow-matching
  diffusion trajectory head.
- **Alpamayo 1.5**: successor, same arch but VLM backbone is
  **Cosmos-Reason2-8B** (NVIDIA reasoning-fine-tuned Qwen3VL).
- **VLM-Action gap**: OOD failure mode where the VLM over-specializes to
  source-domain priors and pushes the Expert toward wrong actions.
- **Angular distance** (ShortGPT): `1 − cos(h_in, h_out)` per layer. No GT
  needed. Measures how much a layer transforms its input.
- **ea (expert-aware) scoring**: re-rank VLM layers by how much they
  affect Expert outputs (uses Expert gradients).
- **Stage 2 SFT**: joint CE + diffusion loss with LoRA on VLM + Expert.
  Currently underperforming zero-shot (see "Active question").
- **NAVSIM**: a closed-loop driving benchmark; we use its `navtest` split.
  PDMS = Predictive Driver Model Score (higher = better).
- **Runtime identity bypass**: in `LayerListWithDropping`, dropped layers
  become a pass-through during forward; weights stay loaded but unused.
  Required for SFT (physical removal breaks reload).
