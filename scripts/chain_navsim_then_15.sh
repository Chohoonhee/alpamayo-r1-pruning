#!/bin/bash
# After R1 NAVSIM k=7 6-shard finishes, aggregate and chain 4 missing
# 1.5 full-set conditions (sharded 6-way each on GPU 2-7, sequential).
set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
QLOG=$LOGS/chain_navsim_then_15.log
cd $SCRIPTS

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# ---- Stage 1: wait for R1 NAVSIM 6 shards, then aggregate ----
TAG_R1=r1_greedy_k7_navsim
log "stage1: waiting for R1 NAVSIM 6 shards"
while true; do
    done=0
    for s in 0 1 2 3 4 5; do
        [ -f $LOGS/fullset_${TAG_R1}_shard${s}.json ] && done=$((done+1))
    done
    if [ $done -eq 6 ]; then break; fi
    sleep 60
done
log "stage1: all 6 shards done, aggregating"
python aggregate_shards.py $LOGS/fullset_${TAG_R1}_shard{0,1,2,3,4,5}.json \
    --out $LOGS/fullset_${TAG_R1}.json 2>&1 | tail -10 | tee -a $QLOG
python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
python $SCRIPTS/generate_analysis_report.py 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "full-set R1 NAVSIM k=7 cross-domain (6-shard) DONE" 2>&1 | tail -2 | tee -a $QLOG

# ---- Stage 2: 4 missing 1.5 conditions, sharded sequentially ----
declare -a TASKS_15=(
    "15_greedy_k3_nusc|$LOGS/ksweep_15_nusc_k3.meta.json"
    "15_greedy_k5_nusc|$LOGS/ksweep_15_nusc_k5.meta.json"
    "15_greedy_k7_nusc|$LOGS/ksweep_15_nusc_k7.meta.json"
    "15_greedy_k7_navsim|$LOGS/ksweep_15_navsim_k7.meta.json"
)
for entry in "${TASKS_15[@]}"; do
    IFS='|' read -r TAG META <<<"$entry"
    if [ -f $LOGS/fullset_${TAG}.json ]; then
        log "[skip] fullset_${TAG}.json exists"; continue
    fi
    if [ ! -f "$META" ]; then
        log "[err] meta $META not found, skipping"; continue
    fi
    log "stage2: launching 6 shards of $TAG"
    pids=()
    for s in 0 1 2 3 4 5; do
        GPU=$((2+s))
        OUT=$LOGS/fullset_${TAG}_shard${s}.json
        LOG=$LOGS/fullset_${TAG}_shard${s}.log
        if [ -f $OUT ]; then continue; fi
        nohup bash -c "export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$GPU; \
            python eval_zeroshot_alignment.py --full_set --shard_idx $s --n_shards 6 \
            --out_json $OUT --device cuda:0 --policy_meta $META" > $LOG 2>&1 &
        pids+=($!)
    done
    log "[launched] ${#pids[@]} shards for $TAG"
    for pid in "${pids[@]}"; do wait $pid; done
    log "stage2: $TAG all shards done, aggregating"
    python aggregate_shards.py $LOGS/fullset_${TAG}_shard{0,1,2,3,4,5}.json \
        --out $LOGS/fullset_${TAG}.json 2>&1 | tail -10 | tee -a $QLOG
    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    python $SCRIPTS/generate_analysis_report.py 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "full-set $TAG (6-shard) DONE" 2>&1 | tail -2 | tee -a $QLOG
done

log "=== CHAIN COMPLETE ==="
bash $COMMIT "chain complete: R1 NAVSIM + 4× 1.5 full-set conditions" 2>&1 | tail -2 | tee -a $QLOG
