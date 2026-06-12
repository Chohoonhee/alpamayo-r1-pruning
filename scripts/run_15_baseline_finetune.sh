#!/bin/bash
# SANITY CHECK: Fine-tune 1.5 baseline (NO PRUNE) with same recipe as v3
# If PDMS drops from 0.72 → ~0.4: fine-tune is hurting, not helping
# If PDMS stays ~0.72: fine-tune is OK, pruning damage is unrecoverable

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
OUT_DIR=/home/irteam/ws/alpamayo_pruning/weights/sft_baseline_15_v3

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

QLOG=$LOGS/sft_baseline_15.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== SANITY: 1.5 BASELINE (no prune) + VLM kept+Expert LoRA + lr 5e-5 + 1k samples + 1 epoch ==="
log "  If FT works: PDMS stays near 0.72. If FT itself breaks model: PDMS drops to ~0.4."

# Create empty policy for "no drop"
cat > $LOGS/empty_meta.json <<'EOF'
{"dropped_layers": [], "policy": "no_prune_baseline", "backbone": "15", "K": 0}
EOF

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

cd $SCRIPTS
torchrun --nproc_per_node=8 --master_port=29503 \
    sft_stage2.py \
    --drop_layers_json $LOGS/empty_meta.json \
    --train_samples 1000 \
    --epochs 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lr 5e-5 \
    --batch_size 1 \
    --grad_accum 8 \
    --out_dir $OUT_DIR \
    --ddp \
    --log_every 10 \
    --save_every 100 2>&1 | tee -a $QLOG

log "=== DONE ==="
