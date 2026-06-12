#!/bin/bash
# PoC: 1.5 + greedy K=2 [23,31] runtime-pruned + Expert LoRA + 1k samples + 1 epoch
# Goal: Does light fine-tune recover PDMS 0.32 → 0.40+?

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
OUT_DIR=/home/irteam/ws/alpamayo_pruning/weights/sft_poc_15_k2

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

QLOG=$LOGS/sft_poc_15.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== PoC fine-tune: 1.5 + K=2 [23,31] + 1k samples + 1 epoch ==="
log "  weights: ALPAMAYO_15_WEIGHTS"
log "  drop_layers: greedy15_navsim_earlystop_meta.json (drops [23,31])"
log "  lr: 5e-6 (Stage 2 v2 recipe)"
log "  DDP 8 GPUs"

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

cd $SCRIPTS
torchrun --nproc_per_node=8 --master_port=29501 \
    sft_stage2_expert_only.py \
    --drop_layers_json $LOGS/greedy15_navsim_earlystop_meta.json \
    --train_samples 1000 \
    --epochs 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lr 5e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --out_dir $OUT_DIR \
    --ddp \
    --save_every 50 2>&1 | tee -a $QLOG

log "=== PoC training DONE ==="
log "LoRA checkpoint dir: $OUT_DIR/lora_final"
