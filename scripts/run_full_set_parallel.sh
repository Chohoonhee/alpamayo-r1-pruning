#!/bin/bash
# 6 critical full-set conditions in parallel on GPU 2-7.
# Each condition uses 1 GPU, processes all 6019 nuScenes val samples.
# Wall clock: ~150min (vs ~3h sequential). Stability stays on GPU 0,1.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
QLOG=$LOGS/fullset_parallel_queue.log

cd $SCRIPTS

echo "[$(date +%H:%M:%S)] === FULL-SET PARALLEL START ===" | tee -a $QLOG

# (gpu | tag | backbone | meta_or_empty | script_name) — exactly 5 fields
declare -a TASKS=(
    "2|r1_baseline|r1||eval_zeroshot_alignment_r1.py"
    "3|r1_greedy_k5_nusc|r1|$LOGS/ksweep_r1_nusc_k5.meta.json|eval_zeroshot_alignment_r1.py"
    "4|r1_greedy_k7_nusc|r1|$LOGS/ksweep_r1_nusc_k7.meta.json|eval_zeroshot_alignment_r1.py"
    "5|r1_greedy_k7_navsim|r1|$LOGS/ksweep_r1_navsim_k7.meta.json|eval_zeroshot_alignment_r1.py"
    "6|r1_greedy_k3_nusc|r1|$LOGS/ksweep_r1_nusc_k3.meta.json|eval_zeroshot_alignment_r1.py"
    "7|15_baseline|15||eval_zeroshot_alignment.py"
)

pids=()
for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU TAG BB META SCRIPT <<<"$entry"
    OUT=$LOGS/fullset_${TAG}.json
    LOG=$LOGS/fullset_${TAG}.log
    if [ -f $OUT ]; then
        echo "[skip] $OUT exists" | tee -a $QLOG
        continue
    fi
    ARGS="--full_set --shard_idx 0 --n_shards 1 --out_json $OUT --device cuda:0"
    [ -n "$META" ] && ARGS="$ARGS --policy_meta $META"
    echo "[launch] gpu=$GPU tag=$TAG script=$SCRIPT meta=${META:-baseline}" | tee -a $QLOG
    nohup bash -c "export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$GPU; \
        python $SCRIPT $ARGS" > $LOG 2>&1 &
    pids+=($!)
done

echo "[$(date +%H:%M:%S)] launched ${#pids[@]} parallel full-set evals" | tee -a $QLOG
echo "  PIDs: ${pids[*]}" | tee -a $QLOG

# Wait for all and periodically commit
while true; do
    alive=0
    for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then alive=$((alive+1)); fi
    done
    if [ $alive -eq 0 ]; then break; fi
    # Check for newly-completed outputs to commit
    sleep 300  # 5 min
    python $STATUS 2>&1 | tail -1 | tee -a $QLOG
    python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "full-set progress (${alive} still running)" 2>&1 | tail -1 | tee -a $QLOG
done

echo "[$(date +%H:%M:%S)] all evals done" | tee -a $QLOG
python $STATUS 2>&1 | tail -1 | tee -a $QLOG
python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "full-set parallel eval COMPLETE (6 conditions × 6019 samples)" 2>&1 | tail -2 | tee -a $QLOG

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === FULL-SET PARALLEL DONE ===" | tee -a $QLOG
