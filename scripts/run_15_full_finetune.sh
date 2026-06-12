#!/bin/bash
# FULL recipe-changed fine-tune: VLM kept + Expert LoRA, lr 5e-5, 28k samples, 3 epochs
# Apply to: 1.5 + greedy K=2 [23,31]

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
OUT_DIR=/home/irteam/ws/alpamayo_pruning/weights/sft_full_15_k2_v3

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

QLOG=$LOGS/sft_full_15.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== FULL FT v3: 1.5 + K=2 [23,31] + VLM kept+Expert LoRA + lr 5e-5 + 28k samples + 3 epochs ==="

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

cd $SCRIPTS
torchrun --nproc_per_node=8 --master_port=29502 \
    sft_stage2.py \
    --drop_layers_json $LOGS/greedy15_navsim_earlystop_meta.json \
    --train_samples 28000 \
    --epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lr 5e-5 \
    --batch_size 1 \
    --grad_accum 8 \
    --out_dir $OUT_DIR \
    --ddp \
    --log_every 10 \
    --save_every 200 2>&1 | tee -a $QLOG

log "=== FULL FT training DONE ==="
log "LoRA checkpoint: $OUT_DIR/lora_final"
