#!/bin/bash
# Priority 1: K-sweep over greedy histories.
#
# For each greedy run × each k ∈ {1,2,3,5,7,9,12}: extract drop_set,
# build meta JSON, eval on both nuScenes val (100) and NAVSIM holdout (50).
# Produces a per-K alignment curve to find the true optimal K per backbone
# × calibration domain combination — the paper Figure 2 candidate.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
QLOG=$LOGS/k_sweep_queue.log

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

echo "[$(date +%H:%M:%S)] === K-sweep START ===" | tee -a $QLOG

# 1. Generate meta files for each (greedy, k) combination
python -c "
import json
GREEDY_FILES = [
    ('r1', 'nusc',   'logs/greedyR1.json'),
    ('r1', 'navsim', 'logs/greedyR1_navsim.json'),
    ('15', 'nusc',   'logs/greedy15.json'),
    ('15', 'navsim', 'logs/greedy15_navsim.json'),
]
KS = [1, 2, 3, 5, 7, 9, 12]
written = []
for bb, calib, path in GREEDY_FILES:
    try:
        d = json.load(open(path))
        history = [h for h in d['history'] if h['round'] > 0]
        for k in KS:
            if k > len(history):
                continue
            entry = history[k-1]  # round k = history[k-1]
            assert entry['round'] == k
            drop = sorted(entry['drop_set'])
            meta = {
                'dropped_layers': drop,
                'policy': f'greedy_k{k}',
                'backbone': bb,
                'calibration_domain': calib,
                'source_greedy': path,
            }
            p = f'logs/ksweep_{bb}_{calib}_k{k}.meta.json'
            json.dump(meta, open(p, 'w'), indent=2)
            written.append((bb, calib, k, p, drop))
print(f'wrote {len(written)} meta files')
" | tee -a $QLOG

# 2. Launch evals in parallel batches (8 at a time)
# Define eval tasks: (gpu, backbone, meta, eval_domain, out_name)
declare -a TASKS=()
for bb in r1 15; do
    for calib in nusc navsim; do
        for k in 1 2 3 5 7 9 12; do
            META="$LOGS/ksweep_${bb}_${calib}_k${k}.meta.json"
            if [ ! -f $META ]; then continue; fi
            # eval on nuScenes val
            TASKS+=("$bb|$META|nusc|ksweep_${bb}_${calib}_k${k}_on_nusc")
            # eval on NAVSIM holdout
            TASKS+=("$bb|$META|navsim|ksweep_${bb}_${calib}_k${k}_on_navsim")
        done
    done
done
echo "[$(date +%H:%M:%S)] queued ${#TASKS[@]} K-sweep evals" | tee -a $QLOG

# 3. Run in waves of 8 (one per GPU)
GPUS=(0 1 2 3 4 5 6 7)
WAVE_SIZE=8
N=${#TASKS[@]}
for ((i=0; i<N; i+=WAVE_SIZE)); do
    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] wave $((i/WAVE_SIZE+1)) (tasks $i..$((i+WAVE_SIZE-1)))" | tee -a $QLOG
    pids=()
    for ((j=0; j<WAVE_SIZE && i+j<N; j++)); do
        IFS='|' read -r BB META DOMAIN OUTN <<<"${TASKS[$((i+j))]}"
        GPU=${GPUS[$j]}
        LOG_FILE=$LOGS/run_${OUTN}.log
        OUT_FILE=$LOGS/${OUTN}.json
        nohup bash /tmp/eval_peak_one.sh $GPU $BB $META $DOMAIN $OUTN > $LOG_FILE 2>&1 &
        pids+=($!)
        echo "  task $((i+j)) gpu=$GPU bb=$BB domain=$DOMAIN → $OUT_FILE" | tee -a $QLOG
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "[$(date +%H:%M:%S)] wave done" | tee -a $QLOG
    # Refresh + commit after each wave
    python $STATUS 2>&1 | tail -1 | tee -a $QLOG
    python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "K-sweep wave $((i/WAVE_SIZE+1))" 2>&1 | tail -2 | tee -a $QLOG
done

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === K-sweep DONE ===" | tee -a $QLOG
python $TRANSCRIPT 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "K-sweep complete — per-K alignment curves" 2>&1 | tail -2 | tee -a $QLOG
