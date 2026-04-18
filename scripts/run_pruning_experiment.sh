#!/usr/bin/env bash
# Full angular-distance pruning + nuScenes SFT experiment pipeline for R1.
#
# Steps:
#   1. Score all 36 text layers via angular distance on 100 nuScenes val samples
#   2. Prune: angular-guided drop of 13 layers → save checkpoint
#   3. (Optionally) prune: last-13 and random-13 for ablation
#   4. Fine-tune each pruned model on N nuScenes train samples
#   5. Evaluate each variant on nuScenes val
#
# Usage:
#   conda activate alpamayo_b2d
#   cd /home/irteam/ws/alpamayo_pruning/scripts
#   bash run_pruning_experiment.sh [N_SAMPLES] [GPU]
#
# N_SAMPLES: number of nuScenes train samples for SFT (default 500)
# GPU: CUDA device (default 0)

set -euo pipefail

N_SAMPLES=${1:-500}
GPU=${2:-0}
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_BASE="/home/irteam/ws/alpamayo_pruning/weights"
B2D_DIR="/home/irteam/ws/alpamayo_bench2drive"

export CUDA_VISIBLE_DEVICES=${GPU}

echo "============================================================"
echo " Alpamayo R1 Pruning Experiment"
echo " N_SAMPLES=${N_SAMPLES}  GPU=${GPU}"
echo "============================================================"

# ── Step 1: Angular distance scoring ─────────────────────────────────────────
SCORES_FILE="${SCRIPTS_DIR}/angular_scores_r1.json"
if [ -f "${SCORES_FILE}" ]; then
    echo "[SKIP] Scores already exist: ${SCORES_FILE}"
else
    echo ""
    echo "[1/4] Scoring layers ..."
    python "${SCRIPTS_DIR}/angular_dist_r1.py" \
        --weights "${WEIGHTS_BASE}/Alpamayo-R1-10B" \
        --n_samples 100 \
        --out "${SCORES_FILE}" \
        --device "cuda:0"
    echo "[1/4] Done."
fi

# ── Step 2: Prune (angular) ───────────────────────────────────────────────────
PRUNED_ANGULAR="${WEIGHTS_BASE}/Alpamayo-R1-10B-pruned-angular13"
if [ -d "${PRUNED_ANGULAR}" ]; then
    echo "[SKIP] Pruned model exists: ${PRUNED_ANGULAR}"
else
    echo ""
    echo "[2/4] Pruning (angular, drop 13) ..."
    python "${SCRIPTS_DIR}/prune_r1.py" \
        --scores "${SCORES_FILE}" \
        --strategy angular --n_drop 13 \
        --out "${PRUNED_ANGULAR}"
    echo "[2/4] Done."
fi

# Optional: last-13 and random-13 ablations
PRUNED_LAST="${WEIGHTS_BASE}/Alpamayo-R1-10B-pruned-last13"
if [ ! -d "${PRUNED_LAST}" ]; then
    echo ""
    echo "[2b] Pruning (last-13) ..."
    python "${SCRIPTS_DIR}/prune_r1.py" \
        --strategy last --n_drop 13 \
        --out "${PRUNED_LAST}"
fi

PRUNED_RANDOM="${WEIGHTS_BASE}/Alpamayo-R1-10B-pruned-random13"
if [ ! -d "${PRUNED_RANDOM}" ]; then
    echo ""
    echo "[2c] Pruning (random-13, seed=42) ..."
    python "${SCRIPTS_DIR}/prune_r1.py" \
        --strategy random --n_drop 13 --seed 42 \
        --out "${PRUNED_RANDOM}"
fi

echo ""
echo "[3/4] SFT fine-tuning ..."
echo "  (run manually or integrate with train_hf.py — see sft_nuscenes.yaml)"
echo ""
echo "  Example commands:"
echo "    cd ${B2D_DIR}"
echo "    # Angular pruned, N=${N_SAMPLES} samples:"
echo "    python -m finetune.sft.train_hf \\"
echo "        --config-path finetune/sft/configs --config-name sft_nuscenes \\"
echo "        model.checkpoint_path=${PRUNED_ANGULAR} \\"
echo "        data.train_dataset.n_samples=${N_SAMPLES} \\"
echo "        paths.output_dir=output_nuscenes_angular13_n${N_SAMPLES}"
echo ""
echo "    # Baseline (full R1, no pruning), N=${N_SAMPLES} samples:"
echo "    python -m finetune.sft.train_hf \\"
echo "        --config-path finetune/sft/configs --config-name sft_nuscenes \\"
echo "        model.checkpoint_path=${WEIGHTS_BASE}/Alpamayo-R1-10B \\"
echo "        data.train_dataset.n_samples=${N_SAMPLES} \\"
echo "        paths.output_dir=output_nuscenes_full_n${N_SAMPLES}"

echo ""
echo "[4/4] Evaluation:"
echo "  After SFT, start each checkpoint's server and run nuscenes_zero_shot.py"
echo "  to compare L2/collision metrics."
echo ""
echo "Done."
