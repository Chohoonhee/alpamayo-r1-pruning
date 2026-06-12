#!/bin/bash
# SAFE FT: only action_in_proj and action_out_proj trainable.
# Goal: prove FT can be done without breaking <traj_future_start> generation.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
OUT_DIR=/home/irteam/ws/alpamayo_pruning/weights/sft_safe_15_k2

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

QLOG=$LOGS/sft_safe_15.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== SAFE FT: 1.5 + K=2 [23,31] + ONLY action_proj MLPs + 1k samples + 1 epoch ==="

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

cd $SCRIPTS
torchrun --nproc_per_node=8 --master_port=29504 \
    sft_stage2_safe.py \
    --drop_layers_json $LOGS/greedy15_navsim_earlystop_meta.json \
    --train_samples 1000 \
    --epochs 1 \
    --lr 1e-4 \
    --batch_size 1 \
    --grad_accum 8 \
    --out_dir $OUT_DIR \
    --ddp \
    --log_every 5 \
    --save_every 50 2>&1 | tee -a $QLOG

log "=== SAFE FT training DONE ==="
