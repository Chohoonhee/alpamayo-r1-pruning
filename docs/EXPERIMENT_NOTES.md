# Experiment notes

Technical details and design decisions behind the numbers in the README.

## Model architecture reminders

Alpamayo R1 = 11.08 B parameters total:
- **Vision encoder** (SigLIP-ish): 0.58 B (5.2 %)
- **Action decoder / expert** (trajectory head): 2.28 B (20.6 %)
- **Text backbone (Qwen3VL-8B language_model)**: 36 layers × 192.9 M = 6.95 B (62.7 %)
- Embeddings & misc: remainder

Action decoder + vision encoder together are 25.8 % of the model. Those are
untouched by our pruning. The only freely prunable pool is the 36 text
layers. Per-layer size: 192.9 M params → 2.78 % of total model each.

Pruning N layers removes `N × 2.78 %` of total params. So:
- 13 drop = 36.1 % of text, 22.6 % of total
- 18 drop = 50.0 % of text, 31.4 % of total
- 24 drop = 66.7 % of text, 41.8 % of total
- 28 drop = 77.8 % of text, 48.7 % of total

## Why angular distance specifically

Three alternatives considered:

1. **Angular distance** (ShortGPT, 2024): `1 − cos(h_in, h_out)`. No GT
   required, single forward pass, cheap. Measures how much a layer
   transforms its input on average.
2. **Gradient magnitude / Taylor importance**: requires GT + backward
   pass. We don't yet have enough nuScenes training pipeline integration
   to compute this cheaply.
3. **Attention entropy / head-level importance**: finer-grained but more
   involved implementation.

Chose (1) as the minimum viable first step — the goal of the session was
to prove the infrastructure end-to-end, not to find the optimal
importance metric.

## Why the hook logic works

From `scripts/angular_dist_r1.py`:

```python
def _hook(module, args, output):
    self.h_in  = args[0].detach().float().cpu()
    h_out = output[0] if isinstance(output, tuple) else output
    self.h_out = h_out.detach().float().cpu()
layer_module.register_forward_hook(_hook)
```

Qwen3VL decoder layers return `(hidden_state, self_attn_weights, …)` when
called with `output_attentions=True`, or just `hidden_state` otherwise. The
tuple check handles both paths. `args[0]` is always the input hidden state.

## Driving the server path

The R1 inference server (`alpamayo_bench2drive/alpamayo_infer_server.py`)
is a JSON-REQ/REP server on a user-chosen port. Requests include:
- `frames` (N_cam × N_temporal × 3 × H × W uint8)
- `ego_xyz`, `ego_rot`, `nav` text
- optional `camera_indices`

Our `alpamayo_server.py` is a **different** server (pickle-REQ/REP)
tailored to the NAVSIM pickle protocol. Both are in this repo under
`scripts/`.

## Caveats of the zero-shot nuScenes numbers

- **100 sample variance**: Avg L2 differences under ≈ 0.03 m are within
  noise at this sample size.
- **Stride-sample across val**: we evenly subsample ~150 per-scene samples
  to 100, which covers different scenes but not uniformly weighted. Good
  enough for trend detection.
- **Interpolation tails**: our adapter extrapolates GT beyond 3 s to
  match Alpamayo's 6.4 s horizon. L2 is measured at 1/2/3 s only, so the
  extrapolation doesn't affect the reported metric, but does affect any
  downstream SFT loss.

## Why random might be the near-equal baseline at 13 drop

Two complementary reasons:

1. **Residual-stream redundancy.** With 23 of 36 layers remaining, the
   residual stream still has enough capacity that the missing layers'
   work gets absorbed by adjacent ones. This is well documented for LLMs
   (Gromov et al. 2024). R1 is no exception.
2. **OOD dormancy.** R1's text pathway is trained to produce CoT in
   Alpamayo's native style. On nuScenes inputs, the CoT is either
   irrelevant or structurally broken, meaning the text layers contribute
   less to the final trajectory than they would in-domain. Any 13 layers
   removed from a mostly-idle path produce similar outcomes.

The decisive test is SFT recovery. If angular-13 and random-13 reach
similar final L2 after SFT on the same nuScenes subset, angular distance
wasn't the right signal. If angular-13 reaches lower L2 or converges
faster, it was.

## Seeds and non-determinism

- `prune_r1.py --strategy random` uses `random.Random(seed=42)` for a
  reproducible random drop set (documented in each model's
  `pruning_meta.json`).
- `angular_dist_r1.py` uses no seed — autocast(bfloat16) is not bit-exact.
  In practice the ranking is stable to within one position at the
  top/bottom of the list.

## GPU memory planning

R1 loads in ~20 GB VRAM. Serving with `num_traj_samples ≥ 8` peaks at
~75 GB because of accumulated KV cache across 8 parallel generations,
each up to 128 tokens. NAVSIM with 4 parallel workers times 8 samples
per inference explodes GPU-1 if other tenants are also on the card.

Mitigations we used:
- **Reduce `num_traj_samples` to 1** for NAVSIM comparison runs —
  acceptable because comparison is apples-to-apples (we ran 1.5 with 1
  sample too).
- **Distribute servers across GPUs** with ≥ 76 GB free each, preferring
  GPUs 3, 4, 5, 7 on our machine.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if memory is
  fragmented.

## Why NAVSIM and nuScenes disagree in difficulty

- NAVSIM measures PDM Score, a composite of 9 sub-metrics (collision
  avoidance, lane keeping, progress, comfort…). Heavily scenario-based.
  Sub-metrics amplify small trajectory errors into scoring cliffs.
- nuScenes measures L2 + collision at 1/2/3 s. Smoother, more
  error-proportional.

Same R1 that loses 0.21 PDMS on NAVSIM (0.70 → 0.49) only loses 0.02 m
L2 on nuScenes zero-shot. Different sensitivities to OOD drift.
