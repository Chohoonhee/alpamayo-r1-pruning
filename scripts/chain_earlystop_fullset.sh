#!/bin/bash
# Waits for current chain to finish (1.5 NAVSIM k=7), then evaluates the
# early-stop greedy policies (R1 k=4 drop=[7,17,21,32], 1.5 k=2 drop=[23,31])
# on full nuScenes val. 6-shard each, sequential to avoid GPU oversubscription.
set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
QLOG=$LOGS/chain_earlystop_fullset.log
cd $SCRIPTS

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "waiting for 1.5 NAVSIM k=7 full-set (last step of previous chain) ..."
while [ ! -f $LOGS/fullset_15_greedy_k7_navsim.json ]; do sleep 60; done
log "previous chain done, starting early-stop full-set evals"

declare -a TASKS=(
    "r1_greedy_earlystop|$LOGS/greedyR1_navsim_earlystop_meta.json|eval_zeroshot_alignment_r1.py"
    "15_greedy_earlystop|$LOGS/greedy15_navsim_earlystop_meta.json|eval_zeroshot_alignment.py"
)
for entry in "${TASKS[@]}"; do
    IFS='|' read -r TAG META SCRIPT <<<"$entry"
    if [ -f $LOGS/fullset_${TAG}.json ]; then
        log "[skip] fullset_${TAG}.json exists"; continue
    fi
    log "launching 6 shards of $TAG ($SCRIPT)"
    pids=()
    for s in 0 1 2 3 4 5; do
        GPU=$((2+s))
        OUT=$LOGS/fullset_${TAG}_shard${s}.json
        LOG_=$LOGS/fullset_${TAG}_shard${s}.log
        if [ -f $OUT ]; then continue; fi
        nohup bash -c "export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$GPU; \
            python $SCRIPT --full_set --shard_idx $s --n_shards 6 \
            --out_json $OUT --device cuda:0 --policy_meta $META" > $LOG_ 2>&1 &
        pids+=($!)
    done
    log "[launched] ${#pids[@]} shards for $TAG"
    for pid in "${pids[@]}"; do wait $pid; done
    log "$TAG: all shards done, aggregating"
    python aggregate_shards.py $LOGS/fullset_${TAG}_shard{0,1,2,3,4,5}.json \
        --out $LOGS/fullset_${TAG}.json 2>&1 | tail -10 | tee -a $QLOG
    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    python $SCRIPTS/generate_analysis_report.py 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "full-set $TAG (early-stop policy, 6-shard) DONE" 2>&1 | tail -2 | tee -a $QLOG
done

log "=== EARLY-STOP FULLSET CHAIN COMPLETE ==="
bash $COMMIT "early-stop full-set chain complete (R1 k=4 + 1.5 k=2)" 2>&1 | tail -2 | tee -a $QLOG
