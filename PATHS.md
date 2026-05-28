# Paths & environment variables

All scripts in this repo resolve absolute paths through
[`scripts/paths.py`](scripts/paths.py), which reads a small set of
environment variables. **Set them once in your shell and every script picks
them up.**

Defaults match the project's original `/home/irteam/ws/` layout, so existing
runs on the dev server keep working with no env-var changes.

---

## Required variables

| Variable | Default | What it points to |
|---|---|---|
| `ALPAMAYO_R1_SRC` | `/home/irteam/ws/alpamayo_bench2drive/alpamayo` | Clone of [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) — must contain `src/alpamayo_r1/` |
| `ALPAMAYO_15_SRC` | `/home/irteam/ws/alpamayo_pruning/alpamayo1.5` | Alpamayo 1.5 source dir — must contain `src/alpamayo1_5/` and a working `a1_5_venv/` |
| `ALPAMAYO_WEIGHTS_DIR` | `/home/irteam/ws/alpamayo_pruning/weights` | Parent dir holding `Alpamayo-R1-10B/`, `Alpamayo-1.5-10B/`, all `*-pruned-*/` variants, and `sft_*` LoRA outputs |
| `NUSC_ROOT` | `/home/irteam/ws/nuscenes/raw_extracted` | nuScenes data root (parent of `v1.0-trainval/`, `samples/`, `sweeps/`) |
| `NAVSIM_WORKSPACE` | `/home/irteam/ws/alpamayo_pruning/navsim_workspace` | NAVSIM data + devkit dir |

## Optional variables

| Variable | Default | Purpose |
|---|---|---|
| `NUSC_VERSION` | `v1.0-trainval` | nuScenes split |
| `OUTPUTS_DIR` | `<repo>/scripts/` | Where scripts dump JSON / log output |

## Quick verification

```bash
cd scripts/
python paths.py
```

Prints every resolved path and whether it exists. Fix any `exists=False`
before running training/eval.

---

## Example: fresh machine

```bash
# 1. Set env vars in your shell rc or a project .env
cat >> ~/.bashrc <<'EOF'
export ALPAMAYO_R1_SRC=$HOME/repos/alpamayo
export ALPAMAYO_15_SRC=$HOME/repos/alpamayo1.5
export ALPAMAYO_WEIGHTS_DIR=/data/alpamayo_weights
export NUSC_ROOT=/data/nuscenes
export NAVSIM_WORKSPACE=$HOME/navsim_ws
export OUTPUTS_DIR=$HOME/alpamayo_outputs
EOF
source ~/.bashrc

# 2. Verify
cd alpamayo-r1-pruning/scripts
python paths.py
# Should print your paths with exists=True everywhere

# 3. Run any script — no edits needed
python angular_dist_r1.py --n_samples 100
```

---

## Migrating an existing run

If you previously ran scripts pinned to `/home/irteam/...`, nothing changes
on that machine. The defaults in `paths.py` reproduce the old layout. The
env-var system is additive: opt in by exporting variables, opt out by doing
nothing.

## External artifacts (not in git)

A few large items live on Google Drive instead of the repo:

| Artifact | Size | Drive path |
|---|---|---|
| `sft_stage2_full/lora_final` (Stage 2 LoRA, 21 GB merged shards) | 21 GB | `gdrive:alpamayo-pruning-artifacts/sft_stage2_full/lora_final` |
| `eval_stage2_result.json` + train/eval logs | < 50 KB | `gdrive:alpamayo-pruning-artifacts/` |

Fetch them with `rclone` once the `gdrive` remote is configured against the
project account. The same commands are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) under "Stage 2 SFT checkpoint".

Alpamayo 1.5 base weights (`nvidia/Alpamayo-1.5-10B`) and Cosmos-Reason2-8B
are gated HuggingFace repos — not redistributable via gdrive. See REPRODUCE.md
for the access-request flow.

## Vendored helper modules

These were originally in `/home/irteam/ws/vipe_test/` (the sister
ViPE project) and are now vendored locally so this repo is self-contained:

- `scripts/nuscenes_zero_shot.py` — nuScenes val L2/collision evaluator
- `scripts/alpamayo_client.py` — ZMQ client for the inference server

They consume `NUSC_ROOT` / `NUSC_VERSION` / `OUTPUTS_DIR` from
`paths.py` just like the rest.
