#!/bin/bash
# Full-set nuScenes val eval (6019 samples) for critical R1 conditions.
# Sharded across 6 GPUs (2-7, leaving 0/1 for stability). Per shard
# ~1000 samples × 1.5s = ~25min. Total per condition: ~30min wall clock.
# Conditions:
#   R1 baseline
#   R1 greedy k=3 nuS
#   R1 greedy k=5 nuS
#   R1 greedy k=7 nuS
#   R1 greedy k=7 NAVSIM (cross-domain)
#   1.5 baseline (reference)

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
QLOG=$LOGS/fullset_queue.log

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

N_SHARDS=6

eval_one_condition() {
    local tag=$1        # short name for the condition
    local bb=$2         # r1 or 15
    local meta=$3       # path to policy_meta or empty for baseline
    local script_name=$4  # eval_zeroshot_alignment_r1.py or eval_zeroshot_alignment.py

    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] === FULL-SET $tag ===" | tee -a $QLOG

    # Launch N_SHARDS shards in parallel on GPUs 2..7
    pids=()
    for s in $(seq 0 $((N_SHARDS - 1))); do
        GPU=$((2 + s))
        OUT=$LOGS/fullset_${tag}_shard${s}.json
        LOG=$LOGS/fullset_${tag}_shard${s}.log
        if [ -f $OUT ]; then
            echo "  [skip] $OUT exists" | tee -a $QLOG
            continue
        fi
        ARGS="--full_set --shard_idx $s --n_shards $N_SHARDS --out_json $OUT --device cuda:0"
        [ -n "$meta" ] && ARGS="$ARGS --policy_meta $meta"
        nohup bash -c "export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$GPU; \
            python $script_name $ARGS" > $LOG 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "[$(date +%H:%M:%S)] all shards done" | tee -a $QLOG

    # Aggregate
    SHARDS=""
    for s in $(seq 0 $((N_SHARDS - 1))); do
        SHARDS="$SHARDS $LOGS/fullset_${tag}_shard${s}.json"
    done
    python aggregate_shards.py $SHARDS --out $LOGS/fullset_${tag}.json 2>&1 | tail -5 | tee -a $QLOG

    # Commit + status
    python $STATUS 2>&1 | tail -1 | tee -a $QLOG
    python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "full-set eval: $tag" 2>&1 | tail -2 | tee -a $QLOG
}

echo "[$(date +%H:%M:%S)] === FULL-SET QUEUE START ===" | tee -a $QLOG

# Wait for fill-in queue to finish so GPUs are free
echo "[$(date +%H:%M:%S)] waiting for fill-in to finish ..." | tee -a $QLOG
while ! grep -q "FILL-IN DONE" $LOGS/fill_in_queue.log 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] fill-in done" | tee -a $QLOG

eval_one_condition "r1_baseline"        r1 ""                                       eval_zeroshot_alignment_r1.py
eval_one_condition "r1_greedy_k3_nusc"  r1 "$LOGS/ksweep_r1_nusc_k3.meta.json"      eval_zeroshot_alignment_r1.py
eval_one_condition "r1_greedy_k5_nusc"  r1 "$LOGS/ksweep_r1_nusc_k5.meta.json"      eval_zeroshot_alignment_r1.py
eval_one_condition "r1_greedy_k7_nusc"  r1 "$LOGS/ksweep_r1_nusc_k7.meta.json"      eval_zeroshot_alignment_r1.py
eval_one_condition "r1_greedy_k7_navsim" r1 "$LOGS/ksweep_r1_navsim_k7.meta.json"   eval_zeroshot_alignment_r1.py
eval_one_condition "15_baseline"        15 ""                                       eval_zeroshot_alignment.py

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === FULL-SET QUEUE DONE ===" | tee -a $QLOG
bash $COMMIT "full-set eval complete — 6 conditions × 6019 samples" 2>&1 | tail -2 | tee -a $QLOG
