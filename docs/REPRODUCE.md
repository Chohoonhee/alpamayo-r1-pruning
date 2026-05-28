# Reproduction guide

Step-by-step to rebuild every result in this repo from scratch.

> **Portable paths.** Every script reads paths from environment variables
> (see [`PATHS.md`](../PATHS.md) for the full contract). The commands below
> use those env vars so they work on any machine, not just the dev server.

## 0 — Expected environment

- Linux, CUDA-capable GPU(s) with ≥ 22 GB free VRAM per R1 server
- conda + uv (or pip)
- Alpamayo source repos (see step 1)
- External datasets:
  - nuScenes v1.0-trainval
  - NAVSIM navtest split

Pick a root and export it once — every other path can derive from it:

```bash
export ALPAMAYO_ROOT="$HOME/alpamayo_workspace"   # or wherever
mkdir -p "$ALPAMAYO_ROOT"

# These point the repo's scripts at the right places. Defaults match the
# dev server's /home/irteam/ws/ layout, so on that machine you can skip them.
export ALPAMAYO_R1_SRC="$ALPAMAYO_ROOT/alpamayo"
export ALPAMAYO_15_SRC="$ALPAMAYO_ROOT/alpamayo1.5"
export ALPAMAYO_WEIGHTS_DIR="$ALPAMAYO_ROOT/weights"
export NUSC_ROOT="$ALPAMAYO_ROOT/nuscenes/raw_extracted"
export NAVSIM_WORKSPACE="$ALPAMAYO_ROOT/navsim_workspace"
export OUTPUTS_DIR="$ALPAMAYO_ROOT/outputs"
```

Verify with `python scripts/paths.py` — every line should show
`exists=True` once you've completed the downloads in steps 1–2.

## 1 — Clone Alpamayo repos

```bash
# Alpamayo R1 source (has alpamayo_r1 python package + ZMQ servers)
git clone https://github.com/NVlabs/alpamayo "$ALPAMAYO_R1_SRC"

# Alpamayo 1.5 source (private — request access from NVIDIA, then clone into:)
# "$ALPAMAYO_15_SRC"

# NAVSIM devkit fork (customised to use alpamayo_agent)
# (project's internal fork — clone into "$NAVSIM_WORKSPACE/navsim")
```

### Base model weights

```bash
mkdir -p "$ALPAMAYO_WEIGHTS_DIR"

# Alpamayo R1 — public HF repo
huggingface-cli download nvidia/Alpamayo-R1-10B \
    --local-dir "$ALPAMAYO_WEIGHTS_DIR/Alpamayo-R1-10B"
```

#### Alpamayo 1.5 weights (gated)

`nvidia/Alpamayo-1.5-10B` is a **gated** HuggingFace repo and the supporting
processor pulls `nvidia/Cosmos-Reason2-8B` which is also gated. Both need a
manual access request before the download succeeds.

1. Visit https://huggingface.co/nvidia/Alpamayo-1.5-10B and
   https://huggingface.co/nvidia/Cosmos-Reason2-8B and click **"Request access"**
   on each. Approval is usually same-day if your HF account has a real name + org.
2. Once approved, log in locally:
   ```bash
   huggingface-cli login   # paste an HF token with `read` scope
   ```
3. Download:
   ```bash
   huggingface-cli download nvidia/Alpamayo-1.5-10B \
       --local-dir "$ALPAMAYO_WEIGHTS_DIR/Alpamayo-1.5-10B"
   ```
4. **Run with `HF_HUB_OFFLINE=1`** every time afterward so the processor
   doesn't try to re-fetch Cosmos-Reason2 at import time (which causes
   403 errors even when you have access):
   ```bash
   export HF_HUB_OFFLINE=1
   ```

If you only need to reproduce the R1 results (angular vs random pruning,
~half the entries in `docs/KEY_FINDING.md`), you can skip the 1.5 download
entirely.

### Stage 2 SFT checkpoint (optional, for skipping the 20h train)

The current Stage 2 SFT result (`ea_vlm28 + joint CE+diffusion LoRA`,
L2 1.622 — see `weights/eval_stage2_result.json` in this repo) was produced
by `sft_stage2_full/lora_final`. That's a 21 GB merged-model checkpoint
(safetensors shards) and is not in git.

Download it from the project's Google Drive (if you have the `gdrive` rclone
remote configured):

```bash
mkdir -p "$ALPAMAYO_WEIGHTS_DIR/sft_stage2_full"
rclone copy --progress --transfers 4 \
    gdrive:alpamayo-pruning-artifacts/sft_stage2_full/lora_final \
    "$ALPAMAYO_WEIGHTS_DIR/sft_stage2_full/lora_final"

# Also fetch the eval result + train logs (small)
rclone copy gdrive:alpamayo-pruning-artifacts/ \
    "$ALPAMAYO_WEIGHTS_DIR/" \
    --include "{eval_stage2_result.json,sft_stage2_eval.log,sft_stage2_full_train.log}"
```

Don't have the gdrive remote? Either run `rclone config` to create a Google
Drive remote named `gdrive` and point it at the account that owns the
artifacts, or re-train (see "Stage 2 SFT" in `scripts/NOTES.md` —
roughly 20h on 4 GPUs).

Intermediate checkpoints `lora_step_{500,1000,1500,2000,2500}` are **not**
uploaded; they exist only for training-resume and aren't needed for eval.

## 2 — Python environments

Two isolated envs:

```bash
# (a) R1 + nuScenes tooling
conda create -n alpamayo_b2d python=3.12 -y
conda activate alpamayo_b2d
pip install -e "$ALPAMAYO_R1_SRC"   # installs alpamayo_r1
pip install nuscenes-devkit pyquaternion zmq torch transformers peft einops

# (b) Alpamayo 1.5 (has its own uv-managed venv)
cd "$ALPAMAYO_15_SRC"
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv pip install -e .

# (c) NAVSIM venv (Python 3.9)
cd "$NAVSIM_WORKSPACE/navsim"
python3.9 -m venv navsim_venv
source navsim_venv/bin/activate
pip install -e .
```

## 3 — Install NAVSIM agent wrapper

Copy `scripts/alpamayo_navsim_agent.py` to
`navsim_workspace/navsim/navsim/agents/alpamayo_agent.py` and register in
the agent config `navsim/planning/script/config/common/agent/alpamayo_agent.yaml`:

```yaml
_target_: navsim.agents.alpamayo_agent.AlpamayoNAVSIMAgent
_convert_: 'all'
server_addr: "tcp://127.0.0.1:5557,tcp://127.0.0.1:5558,tcp://127.0.0.1:5559,tcp://127.0.0.1:5560"
num_traj_samples: 1    # set to 8 for original config (needs more VRAM)
max_generation_length: 128
timeout_s: 600
trajectory_sampling:
  _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  _convert_: 'all'
  time_horizon: 4
  interval_length: 0.5
```

## 4 — Run NAVSIM evaluations

All NAVSIM scripts launch 4 servers on ports 5557-5560 (different GPUs),
then run `run_pdm_score_one_stage.py`.

```bash
# Alpamayo 1.5 sample500 (reproduces PDMS 0.7002 with num_traj_samples=1)
bash scripts/run_sample500_15_1sample.sh

# Alpamayo R1 sample500 (reproduces PDMS 0.4901)
bash scripts/run_sample500_r1.sh
```

Results land in `$NAVSIM_EXP_ROOT/alpamayo_sample500_*/<timestamp>/<timestamp>.csv`.
CSVs from our run are copied into `results/navsim_sample500/` for reference.

## 5 — Angular distance scoring

```bash
conda activate alpamayo_b2d
python scripts/angular_dist_r1.py \
    --weights /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-R1-10B \
    --n_samples 100 \
    --out scripts/angular_scores_r1.json \
    --device cuda:4
```

Our output shipped as `scripts/angular_scores_r1.json`.

## 6 — Prune checkpoints

```bash
# Angular 13 (36% text-layer drop, 22.6% total params)
python scripts/prune_r1.py \
    --scores scripts/angular_scores_r1.json \
    --strategy angular --n_drop 13 \
    --out /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-R1-10B-pruned-angular13

# Last / random baselines (need no --scores)
python scripts/prune_r1.py --strategy last   --n_drop 13 --out ...-pruned-last13
python scripts/prune_r1.py --strategy random --n_drop 13 --seed 42 --out ...-pruned-random13

# Aggressive variants
python scripts/prune_r1.py --scores scripts/angular_scores_r1.json --strategy angular --n_drop 18 --out ...-pruned-angular18
python scripts/prune_r1.py --scores scripts/angular_scores_r1.json --strategy angular --n_drop 24 --out ...-pruned-angular24
python scripts/prune_r1.py --scores scripts/angular_scores_r1.json --strategy angular --n_drop 28 --out ...-pruned-angular28
python scripts/prune_r1.py --strategy random --n_drop 24 --seed 42 --out ...-pruned-random24
python scripts/prune_r1.py --strategy random --n_drop 28 --seed 42 --out ...-pruned-random28
```

Produced metadata files are in `results/pruning_meta/`.

## 7 — Zero-shot nuScenes eval

Serve each variant on its own port (runs in parallel, 1 GPU each):

```bash
cd /home/irteam/ws/alpamayo_bench2drive
for variant in orig angular13 last13 random13 angular18 angular24 angular28 random24 random28; do
  case $variant in
    orig)      MODEL=/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-R1-10B ;;
    *)         MODEL=/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-R1-10B-pruned-${variant} ;;
  esac
  GPU=$(get_free_gpu)     # pick a GPU with ≥ 22 GB free
  PORT=$(free_port)
  CUDA_VISIBLE_DEVICES=$GPU nohup conda run -n alpamayo_b2d --no-capture-output \
      python alpamayo_infer_server.py --model $MODEL --port $PORT \
      > srv_${variant}.log 2>&1 &
done
```

Then run the comparison scripts (they hard-code ports 5571-5575; edit as needed):

```bash
bash scripts/run_zeroshot_compare.sh 100     # orig, angular13, last13, random13
bash scripts/run_zeroshot_aggressive.sh 100  # angular24, angular28, random24, random28
```

Each run writes `results_<variant>.json` to `zeroshot_logs/`. We copied the
final JSONs into `results/nuscenes_zeroshot/`.

## 8 — SFT (not yet executed in this snapshot)

Configured but unvalidated:

```bash
cd /home/irteam/ws/alpamayo_bench2drive
python -m finetune.sft.train_hf \
    --config-path finetune/sft/configs \
    --config-name sft_nuscenes \
    model.checkpoint_path=/path/to/pruned/weights \
    data.train_dataset.n_samples=500 \
    paths.output_dir=output_nuscenes_angular13_n500
```

Config is at `configs/sft_nuscenes.yaml`; drop it into the Alpamayo SFT
config directory before running.

## 9 — Optional: angular_dist_r1 deterministic reproduction

The scoring run is not bit-exact (torch autocast non-determinism) but the
ranking is stable to within ±1 position in most layers. Fix seeds if you
need byte-reproducible results:

```python
import torch, numpy as np, random
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True)
```
