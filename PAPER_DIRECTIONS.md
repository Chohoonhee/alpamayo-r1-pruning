# Paper directions & cross-project synthesis

**Status:** Synthesis as of 2026-05-28. Reflects the state of both
`alpamayo-vipe-research` and `alpamayo-r1-pruning` repos at that date.
Treat this as a **working hypothesis + recommendations**, not a frozen plan
— if you find a stronger angle, override.

This document carries forward analysis that doesn't live in CLAUDE.md /
KEY_FINDING.md / NOTES.md — namely the **cross-project narrative** and the
**paper-shape options**. A new Claude session picking up either project
should read this after CLAUDE.md.

---

## 1. The unifying problem (both projects)

Both projects attack the same underlying pathology in driving VLA models:

> **The VLM/CoT pathway is mis-aligned with the action head's output on
> out-of-distribution data.** The model "얼추 되지만 틀림" — looks right,
> but the trajectory it emits doesn't match the reasoning it just produced.

This shows up empirically as:
- Alpamayo's action head is 85% "cruise" on nuScenes val regardless of scene.
- Its CoT mentions left/right/stop ~50% of the time (when measured without
  the inference-server prefix bias).
- Pruning experiments show the VLM is largely **dormant on OOD** in R1 — so
  the VLM is "expensive but unused" rather than "actively helpful".

The two projects approach this from opposite angles:
- **ViPE project**: fix the **action** first using pseudo-GT trajectories
  from raw video, then realign CoT to match the corrected action.
- **Pruning project**: surgically **remove** the parts of the VLM that
  contribute to the misalignment, then fast-fine-tune what remains.

---

## 2. ViPE project — concrete insights

### Confirmed by experiment

1. **CoT-Action gap is quantitatively real.** With the no-prefix inference
   server, CoT distribution is `{cruise 50%, stop 15%, left 12%, right 10%}`
   while action distribution is `{cruise 85%, stop 13%, left 2%, right 0.3%}`
   over 10,591 nuScenes samples. The model says "turn" but drives straight.

2. **The default inference server's CoT prefix is a measurement confound.**
   With `alpamayo_infer_server_clean.py`'s injected prefix ("I should follow
   the road ahead..."), CoT becomes 93% cruise. Switching to
   `alpamayo_infer_server_noprefix.py` (prefix removed) drops it to 50% — the
   gap was being **hidden by the prompt template**. Any future measurement
   of CoT-Action alignment must use the no-prefix server.

3. **ViPE on nuScenes is near-metric (scale 0.93-0.99).** Unlike B2D (1.27×)
   or KITTI (1.25×). This means no per-dataset scale calibration is needed
   — pseudo-GT trajectories from ViPE are usable as-is for L2-style training.

4. **ViPE ATE@3s on nuScenes is 0.2-0.7m.** This is roughly 4× better than
   Alpamayo's zero-shot L2 (2.144m at 3s). i.e. ViPE is a **good enough**
   teacher signal — the supervision is more accurate than the student.

5. **Rule-based 4-class CoT classification matches Qwen3-VL judge at 88%.**
   For Step 1 filtering, VLM judges are unnecessary overhead. The 4 classes
   (cruise / left_any / right_any / stop) lump nudge with turn because the
   action head treats them similarly and 6-class had too few samples per bin.

6. **Step 1 reliable filter yields 5,051 / 10,591 (47.7%) samples** where
   rule-CoT and ViPE-trajectory agree on maneuver. Breakdown:
   cruise 3,847, stop 708, right 324, left 172. Non-cruise share = 24%,
   enough to teach maneuvers.

### Failed or learned negatively

7. **MSE loss on `sample_trajectories_from_data_with_vlm_rollout` doesn't
   train the model.** That call path is for inference (samples through
   diffusion); using MSE on its output ignores the Flow Matching velocity
   target. Use `TrainableAlpamayoR1.forward()` which computes the official
   FM diffusion loss.

8. **Flow-matching loss is very sensitive to lr at the start.** Pilot
   training had `avg_loss = 29.04` (FM loss should be O(0.1-1.0)). lr=1e-5
   with clip_grad_norm=1.0 was not enough; all 144 LoRA tensors went NaN
   within the first epoch. **The pilot loss magnitude was already an early
   warning that should not have been ignored.**

### Open / not yet verified

- Step 1 training has never succeeded; the L2 improvement claim is
  unverified. lr=1e-6 + warmup hasn't been tried.
- Possible scale/coordinate mismatch between ViPE-derived `ego_future_xyz`
  and Alpamayo's action-space normalization — would explain the loss
  magnitude even if training were stable.
- Step 2 (Alpamayo_1-action ≈ ViPE filter, joint VLM+Expert LoRA) is
  designed but not run.
- Step 3 (Qwen rewrites CoT conditioned on the model's own action) is
  designed but not run. **It is independent of Step 1's success** — you
  could run Step 3 on the baseline model directly to test the CoT rewrite
  idea before the action curriculum is fixed.

---

## 3. Pruning project — concrete insights

### Confirmed by experiment (★ = paper-strength)

1. **★ Backbone training objective determines pruning sensitivity.** Same
   36-layer Qwen3VL topology in R1 (vanilla) and 1.5 (Cosmos-Reason2,
   reasoning-fine-tuned). Identical drop pattern (`random-13`, seed 42,
   layers [1,2,3,4,7,14,15,17,18,21,23,29,33]) causes:
   - R1 zero-shot Avg L2: +0.061m (essentially tolerated)
   - 1.5 zero-shot Avg L2: **+0.226m** (clearly broken)
   The only variable is the backbone weights. **R1's text pathway is
   dormant on OOD nuScenes; 1.5's is actively used.** This is the
   strongest single finding in either project.

2. **★ ea_vlm-28 on 1.5 is lossless** — 78% of VLM params dropped, AvgL2
   stays at 1.461 vs baseline 1.476. The VLM is not just "compressible",
   it is **mostly redundant** for this domain.

3. **★ R1 angular-24 outperforms baseline** (AvgL2 1.443 < 1.465).
   Slightly. Best interpretation: **regularization effect** — overactive
   text layers were nudging the action head in wrong directions, and
   removing them helps. This is a small but interesting positive ablation.

4. **Random ≈ angular on R1 at low drop counts.** R1 random-13 = +0.061m,
   angular-13 = +0.045m. The text pathway is so uniformly redundant that
   pruning method barely matters. **The discriminator between methods
   will be post-SFT recovery, not zero-shot retention.**

5. **NAVSIM PDMS: 1.5 at 0.7286 vs R1 at 0.4901** (sample500, equal
   num_traj_samples=1). The R1→1.5 generational delta is ~0.21 PDMS at
   matched settings. R1 generalizes much worse OOD than 1.5 — again
   consistent with the "vanilla backbone is dormant on OOD" interpretation.

### Failed or learned negatively (architectural lessons)

6. **Diffusion-only SFT loss broke VLM token generation** (Phase B and C).
   The VLM lost the ability to emit `<traj_future_start>` because the loss
   only flowed through the expert head. **Joint CE + diffusion is
   essential.** This is a non-obvious design lesson — anyone repeating
   this work would hit the same trap.

7. **Stage 2 SFT regressed below zero-shot.** ea_vlm28 + joint loss + LoRA
   on both heads → L2 1.622 vs ea_vlm28 zero-shot L2 1.461. Suspected:
   lr=1e-4 too high + VLM LoRA on already-good model causes drift.
   **No SFT recipe has yet beaten ea_vlm-28 zero-shot.** This is the
   project's current open question.

### Pruning-specific implementation traps

8. **Runtime identity-bypass, not physical pruning, for training.**
   Physical removal reinit dropped positions on reload. Identity-bypass
   keeps weights loaded but skips them in forward. Required for any
   training path.

9. **Don't renumber `layer_idx`.** Qwen3VL's `HybridCache` pre-allocates
   `num_hidden_layers` slots; renumbering causes `IndexError`. Keep
   original indices, skip in forward.

10. **`IdentityLayer.forward` must return `hidden_states` only.** Returning
    a tuple breaks `_deepstack_process`.

### Open / not yet executed (from project's pilot plan)

- Hidden-state linear probing per layer (`hidden → correct action intent`).
  Would give per-layer evidence for the "dormant on OOD" claim beyond the
  current angular-distance proxy.
- Causal layer ablation (replace layer ℓ with identity, measure delta).
  Would identify which specific layers are dead vs alive on each backbone.
- Expert-side pruning after Stage 2 succeeds. Currently the Expert head is
  untouched.

---

## 4. Cross-project synthesis

### Two halves of the same story

| Aspect | ViPE project | Pruning project |
|---|---|---|
| Diagnosis | Action head is wrong; CoT is more right | VLM text path is dormant / harmful |
| Remedy | Pseudo-GT trajectory teaches action | Remove the dormant VLM layers |
| Bottleneck right now | Step 1 training NaN | Stage 2 SFT regression |
| Status | Designed end-to-end, blocked on first training | Strong zero-shot results, SFT recovery unsolved |

The two are **complementary**, not redundant:
- Pruning removes the structural cause of the gap.
- ViPE pseudo-GT provides the supervision signal to teach the remaining
  Expert head.

A combined pipeline would be: prune VLM → use ViPE pseudo-GT to fine-tune
the remaining (now-light) model on a target OOD domain. This is the natural
"future work" for either paper.

### Methodology lessons across both

- **Inference-time prompt artifacts can hide the model's true behavior.**
  ViPE project found this with the CoT prefix; any reviewer-credible
  measurement of CoT-Action alignment must control for it.
- **Loss-magnitude is the first thing to inspect on a new objective.** Both
  projects had silent failures (NaN explosion in ViPE Step 1, Stage 2
  regression in Pruning) that an early loss-scale check would have flagged.
- **Joint CE + diffusion loss is the correct training recipe** for
  VLM+Expert fine-tuning. Diffusion-only loses VLM generative ability;
  CE-only ignores the trajectory.

---

## 5. Paper direction options

### Option A — Pruning solo (recommended)

**Title candidates:**
- "Backbone Training Objective Determines VLM Pruning Sensitivity in
  Driving VLAs"
- "When Half the VLM is Dormant: OOD Pruning of Driving VLAs"
- "Diagnostic VLM Pruning for Fast Domain Adaptation of Driving VLAs"

**Why:** Has the strongest concrete results today.
- R1 vs 1.5 sensitivity contrast is genuinely surprising.
- ea_vlm-28 lossless 78% is a clean engineering win.
- The "random ≈ angular on R1" is a counterintuitive finding that frames
  why deeper investigation is needed.

**Risks:**
- Stage 2 SFT regression is honest but reads as "we haven't fully solved
  this yet". Either pivot to honest limitations section, or solve it
  before submission (Stage 2 v2 plan in CLAUDE.md).
- The "OOD" claim needs to be defended — nuScenes and NAVSIM are arguably
  in-distribution for what NVIDIA trained on. Best framing: "OOD from the
  prompt-tuning + dataset-curation distribution", not "OOD from any driving".

**Target venues:** NeurIPS 2026, CoRL 2026.

### Option B — ViPE solo (would need to unblock first)

**Title candidates:**
- "GT-Free Domain Adaptation of Driving VLAs via Visual SLAM Pseudo-GT"
- "Triangulating Action, CoT, and Pose: Self-Supervised Refinement of
  Driving VLAs"

**Why:** Bigger practical claim ("any dashcam video can fine-tune your
VLA"), which is what the field actually wants.

**Risks:**
- **No L2 improvement yet.** Step 1 training is blocked. Paper requires
  showing the curriculum actually moves L2 from 2.144m toward 1.0m.
- 3-step curriculum is complex; reviewers will ask for ablations of each
  step. Each step adds 4-8 weeks of work.

**Target venues:** ICCV 2026, CVPR 2027, RAL.

### Option C — Combined paper

**Why:** The unifying narrative is genuine; both projects diagnose the
VLM-Action gap and offer complementary remedies.

**Risks:** Massive scope. Either half alone is paper-worthy. Reviewers will
ask why this isn't two papers. Timeline risk is severe — both halves need
to be fully worked out.

---

## 6. Recommended approach

**Option A (Pruning paper) first, ViPE follows as a sequel.**

### Why Option A first
1. **Current evidence is strongest.** The R1/1.5 contrast is the kind of
   surprise reviewers remember. ViPE doesn't have its headline result yet.
2. **Less dependency on unblocked work.** Pruning needs Stage 2 v2 to
   succeed; ViPE needs Step 1 to even start running. The shorter critical
   path is Pruning.
3. **Sets up ViPE as a natural follow-up.** "We pruned the VLM; the
   bottleneck is now the SFT recovery → here's GT-free SFT via ViPE."

### Pruning paper outline (concrete)

```
1. Introduction
   - Driving VLAs (Alpamayo) work in-distribution, fail OOD
   - Hypothesis: the VLM-Action gap widens OOD because the VLM
     over-specializes to source-domain priors
   - We show this is largely structural — much of the VLM is dormant
     on OOD — and remove it surgically

2. Background
   - Driving VLA architecture (VLM + diffusion Expert head)
   - Layer-pruning history (ShortGPT, SmolVLA)
   - Cosmos-Reason2 vs Qwen3VL backbones

3. Method
   - Angular distance scoring per layer
   - Expert-aware re-scoring (ea_vlm): weight the angular signal by
     Expert gradients
   - Runtime identity-bypass for training
   - Joint CE + diffusion SFT loss (rationale: Phase C diffusion-only
     broke VLM generation)

4. Headline result
   - R1 vs 1.5 sensitivity contrast (Table 1, the killer result)
   - ea_vlm-28 lossless on 1.5: 78% VLM param reduction (Table 2)
   - R1 angular-24 outperforms baseline (regularization evidence)

5. Ablations
   - Pruning ratio sweep (13/18/24/28 drop)
   - Random vs angular vs ea (R1 and 1.5)
   - Importance metric ablation
   - NAVSIM eval to confirm the OOD generalization gap is real

6. Limitations
   - Stage 2 SFT regression: post-SFT recovery is an open question
   - "OOD" is narrowly defined (nuScenes/NAVSIM vs NVIDIA in-domain data)
   - Expert head pruning untested

7. Discussion + future work
   - Why R1's text path is dormant — speculation: vanilla Qwen3VL never
     learned to use vision-driving-specific reasoning cues
   - Why 1.5 is more sensitive — Cosmos-Reason2 fine-tune created
     real text-pathway computations
   - Future: combine with GT-free SFT (ViPE pseudo-GT curriculum) to make
     the pruned + lightweight model train fast on any target domain
```

### What needs to happen for Pruning paper

| Item | Effort | Status |
|---|---|---|
| Stage 2 v2 pilot (Expert-only LoRA, lr=5e-6) | ~1 day | Not started |
| If pilot OK → Stage 2 v2 full (28k × 2 epochs, 4 GPU) | ~20h compute | Not started |
| Hidden-state probing per layer (independent evidence) | ~3 days | Optional but strong |
| Causal layer ablation | ~1 week | Optional but strong |
| Honest limitations write-up | ~1 day | Trivial |
| Paper draft | ~2 weeks | Not started |

### What ViPE follow-up looks like

Once Pruning paper is submitted, ViPE story becomes:
> "We have a pruned Alpamayo. To adapt it to a new target domain, we don't
> have GT trajectories. ViPE pseudo-GT + Step 1/2/3 curriculum lets us
> fine-tune the pruned model on raw video. Combined throughput: X frames /
> second / GPU, vs Y for full Alpamayo with GT."

This is naturally a stronger paper than ViPE-solo because (a) the pruned
model is smaller so the curriculum is cheaper to run, and (b) the "why
GT-free matters" motivation is much sharper after the pruning paper's
"we have a small VLM ready to deploy" framing.

---

## 7. Open questions worth thinking about

These are not answered above and may shape the paper differently than
expected:

1. **Is the R1 dormancy a property of Qwen3VL or of NVIDIA's R1 fine-tune?**
   If it's the latter, the "backbone training objective" framing weakens
   — it could be "training data choice" instead. Hidden-state probing of
   raw Qwen3VL-8B (without R1 fine-tune) would disambiguate.

2. **Is ea_vlm-28 lossless on 1.5 OOD-specific, or does it also hold on
   in-distribution NVIDIA data?** If it holds even in-distribution, that's
   a much stronger compression claim. If it only holds OOD, the framing
   has to stay tightly OOD-focused.

3. **Why is Stage 2 SFT specifically worse than zero-shot?** Hypothesis is
   lr-too-high, but it might be:
   - VLM LoRA on dormant layers is just noise
   - Joint loss balancing CE vs diffusion is wrong (no scale weight)
   - Training data subset is too small (28k samples)
   Each implies a different fix. Worth instrumenting before re-running.

4. **Does the CoT rewrite (ViPE Step 3) work standalone, without Steps 1 +
   2?** This is a low-cost experiment — Qwen rewrites CoT conditioned on
   the BASELINE Alpamayo's action, train VLM LoRA. If yes, the ViPE paper
   has a much simpler v1 story ("forget the curriculum, just rewrite CoT").

5. **What's the relationship between ea_vlm-28 (heavy pruning) and the
   CoT-Action gap?** If the gap shrinks after pruning even without SFT,
   that's evidence pruning addresses the same pathology ViPE is addressing.
   Easy to measure: run the same CoT/Action distribution analysis on
   ea_vlm-28 zero-shot. If non-cruise CoT goes from 50% → higher, that's
   smoking-gun evidence for the cross-project narrative.

---

## 8. Things I noticed but couldn't fully chase

- The fact that the ViPE project's reliable filter retains only 47.7% of
  samples means **the model and ViPE disagree on the maneuver in 52% of
  cases**. The 47.7% "reliable" pool is itself biased toward cruise
  (76%). Step 1 training might therefore overfit to cruise even when the
  Step 1 filter is "working as designed".
- ViPE Step 1's NaN explosion happened immediately at high loss (~29).
  The Flow Matching diffusion loss scale is typically O(0.1-1.0). A 30x
  loss magnitude means either the input data is mis-scaled or the action
  space normalization is mismatched. **Before more training attempts:
  compare ViPE-future_xyz statistics to nuScenes GT-future_xyz statistics
  on a few samples.** If magnitudes differ by 10x, that's the bug.
- The R1 zero-shot baseline L2 (1.465) is suspiciously close to the
  trivial "predict constant velocity" baseline. Worth checking what a
  constant-velocity model scores on the same 100-sample subset — if 1.465
  ≈ constant-velocity, then the "lossless pruning" story is weakened
  (you can't lose what wasn't there).
- NAVSIM nav_text experiment (driving_command → "Turn left") slightly
  regressed PDMS (0.7286 → 0.7110). This suggests Alpamayo 1.5 expects
  richer nav strings than one-word commands. Worth knowing for any future
  closed-loop deployment.

---

## 9. What I would do next if I were running this

In priority order:

1. **Cross-project measurement: re-run the CoT/Action distribution
   analysis (ViPE project's measurement) on the ea_vlm-28 zero-shot
   pruned model.** Tiny experiment, possibly huge insight. If pruning
   shrinks the gap without any SFT, that's the bridge for the combined
   paper.

2. **Sanity-check ViPE pseudo-GT scale** against nuScenes GT ego_pose
   on a handful of samples. If the magnitudes differ, that's the NaN
   root cause and unblocks the entire ViPE project in an afternoon.

3. **Stage 2 v2 pilot** on the Pruning side with the lower lr.
   ~2K samples, 1 epoch, Expert-only LoRA. Fast.

4. **Hidden-state linear probe per layer on both R1 and 1.5.** Strong
   independent evidence for the headline R1/1.5 contrast.

5. **Start the Pruning paper draft** with whatever you have after the
   above. Honest limitations are fine; the headline result holds.

---

## 10. Competitive landscape — who else publishes in this space

**Caveat: this section was written with a 2026-01 knowledge cutoff.**
Real publication situation may have shifted by the time you read this. Do
a fresh arxiv / venue search before committing to a paper angle.

### Groups to actively watch

- **OpenDriveLab (Shanghai AI Lab)** — most prolific driving-VLM group.
  Track record: UniAD (CVPR 2023 best paper), DriveLM (CVPR 2024,
  benchmark + dataset), OmniDrive (3D occupancy + VLM), VAD-v2,
  BEV-Planner. Pattern: **they propose new models or benchmarks**, not
  analysis of existing models. They probably haven't done the R1 vs 1.5
  controlled comparison because they don't focus on NVIDIA's Alpamayo
  specifically.
- **UCLA / Berkeley / CMU driving labs** — less concentrated than
  OpenDriveLab but consistent output. Bolei Zhou (MetaDrive,
  ScenarioNet) is simulator-side; others vary.
- **NVIDIA themselves** — they own Alpamayo and could publish their own
  ablations + extensions at any time. Highest scoop risk if they do
  pruning analysis internally.

### Realistic scenario distribution (subjective, pre-search)

| Scenario | P (subjective) | Impact on our paper |
|---|---|---|
| Some group already did "driving VLA + layer pruning" generic ablation | 60% | Cite + extend. Our R1 vs 1.5 controlled contrast still unique. |
| OpenDriveLab released a paper with identical R1 vs 1.5 finding | 10% | Scoop. Need to pivot. |
| Nobody has done a controlled two-backbone comparison on driving VLA | 70% | Our headline result survives. |
| Someone published "VLM is mostly dormant in driving VLA" claim | 30% | We become "we confirm + explain why". Still publishable. |
| NVIDIA themselves released Alpamayo follow-up with pruning | 15% | Scoop on the model side, but our analysis angle survives. |

(Probabilities are subjective and not mutually exclusive — they're
event-likelihoods, not partition probabilities.)

### Why our R1 vs 1.5 contrast is probably still safe

- Both backbones (vanilla Qwen3VL fine-tuned for R1, Cosmos-Reason2
  fine-tuned for 1.5) must be **on the same hardware with the same
  evaluation harness** to do the controlled comparison.
- Cosmos-Reason2 is NVIDIA-internal-flavored; not every lab can or
  bothers to set both up.
- Most groups propose new models; ablation-only papers are rarer.
- Even if pruning has been done, doing it **across two backbones with
  intentional contrast** is a specific methodological angle.

### What would scoop us

- A paper saying "X% of VLA VLM layers are redundant for driving"
  (regardless of method) — that's the same headline category. Need to
  find this before submitting.
- An Alpamayo follow-up by NVIDIA that includes pruning ablations.
- Any paper that contrasts reasoning-tuned vs vanilla backbones for
  ANY VLA task (manipulation or driving) — would force us to position
  as "we extend this principle to driving".

### Defensive moves (do these THIS WEEK)

1. **Literature reconnaissance** via arxiv + Google Scholar with these queries:
   - `"driving VLA" pruning`
   - `"vision-language-action" compression driving`
   - `Alpamayo` (any follow-up)
   - `"Cosmos-Reason"` ablation
   - `driving VLM` `layer importance` OR `redundancy`
   - OpenDriveLab Github publications list (last 12 months)
   - NVIDIA driving research / Alpamayo-related arxiv submissions
2. **Identify 3-5 most-related published papers** and decide:
   - Cite + extend? (most likely outcome)
   - Pivot angle? (if direct overlap exists)
3. **Decide submission venue and deadline before doing more experiments**
   so the experimental priorities follow the venue requirements.

### Fall-back if scooped

- **Cross-domain angle**: re-run the same pruning analysis on a
  manipulation VLA (OpenVLA, π0). If the "OOD VLM dormancy" pattern
  also holds in manipulation, the contribution becomes "general
  principle across VLA domains" — much higher value, less domain-specific.
  This is genuinely the strongest fall-back, leveraging the user's
  ongoing work in manipulation VLA (MemVLA, VLA-Adapter).
- **Theoretical angle**: explain via attention-flow / hidden-state
  analysis why reasoning fine-tuning makes a backbone less prunable.
  Less "engineering paper", more "analysis paper" — fits NeurIPS
  analysis-track better than method-track.

### Venue success probability (subjective, conditional on no scoop)

| Venue | P (accept) | What pushes it up |
|---|---|---|
| Workshop @ NeurIPS/CoRL (efficient VLA / OOD generalization) | 90% | Current results are enough |
| CoRL 2026 main | 55% | Stage 2 v2 SFT recovery succeeds |
| NeurIPS 2026 main (analysis track) | 30% | Hidden-state probing + 1 additional backbone comparison |
| ICLR 2027 main | 25% | Theoretical explanation + multi-domain (manipulation) extension |
| CVPR/ICCV | 8% | Vision conferences less receptive to driving-VLA niche |

These are honest, somewhat-pessimistic estimates. Reality usually lower
than the writer's gut feel.

---

## 11. How to evolve this document

- If new experiments confirm or refute an item above, edit in-place with
  a `[2026-MM-DD]` annotation.
- If the recommended Option A pivots (e.g. ViPE unblocks before Pruning
  Stage 2 v2), rewrite section 5-6 and leave the old text under a
  `## Superseded analysis (DATE)` heading so the history is preserved.
- Don't let this doc become a graveyard — if you've decided some item
  here is wrong, delete it and note in CLAUDE.md why.
