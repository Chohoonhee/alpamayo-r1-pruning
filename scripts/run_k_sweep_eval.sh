#!/bin/bash
# Backfill K-sweep evals: meta files were generated, run all eval combos
# in waves of 6 (GPUs 2..7, leaving 0/1 for stability).

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
QLOG=$LOGS/k_sweep_eval_queue.log

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

echo "[$(date +%H:%M:%S)] === K-sweep EVAL backfill START ===" | tee -a $QLOG

declare -a TASKS=()
for bb in r1 15; do
    for calib in nusc navsim; do
        for k in 1 2 3 5 7 9 12; do
            META="$LOGS/ksweep_${bb}_${calib}_k${k}.meta.json"
            [ ! -f $META ] && continue
            TASKS+=("$bb|$META|nusc|ksweep_${bb}_${calib}_k${k}_on_nusc")
            TASKS+=("$bb|$META|navsim|ksweep_${bb}_${calib}_k${k}_on_navsim")
        done
    done
done
echo "[$(date +%H:%M:%S)] queued ${#TASKS[@]} evals" | tee -a $QLOG

GPUS=(2 3 4 5 6 7)
WAVE=6
N=${#TASKS[@]}
for ((i=0; i<N; i+=WAVE)); do
    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] wave $((i/WAVE+1)) tasks $i..$((i+WAVE-1))" | tee -a $QLOG
    pids=()
    for ((j=0; j<WAVE && i+j<N; j++)); do
        IFS='|' read -r BB META DOMAIN OUTN <<<"${TASKS[$((i+j))]}"
        GPU=${GPUS[$j]}
        LOG_FILE=$LOGS/run_${OUTN}.log
        OUT_FILE=$LOGS/${OUTN}.json
        if [ -f $OUT_FILE ]; then
            echo "  [skip $((i+j))] $OUT_FILE exists" | tee -a $QLOG
            continue
        fi
        nohup bash /tmp/eval_peak_one.sh $GPU $BB $META $DOMAIN $OUTN > $LOG_FILE 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "[$(date +%H:%M:%S)] wave done" | tee -a $QLOG
    python $STATUS 2>&1 | tail -1 | tee -a $QLOG
    python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "K-sweep eval wave $((i/WAVE+1))" 2>&1 | tail -1 | tee -a $QLOG
done

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === K-sweep EVAL DONE ===" | tee -a $QLOG
bash $COMMIT "K-sweep eval complete" 2>&1 | tail -1 | tee -a $QLOG
