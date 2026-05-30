#!/bin/bash
# Runs after the NAVSIM-followup matrix completes. Adds:
#   B1. Random baseline (R1, drop=12, 3 seeds) — control
#   B2. Random baseline (1.5, drop=12, 3 seeds) — control
#   C.  Sample efficiency: R1 NAVSIM greedy with N=10 calibration
#   D.  Final analysis report regeneration + push.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
QLOG=$LOGS/extended_queue.log

echo "[$(date +%H:%M:%S)] extended queue: waiting for NAVSIM followup ..." | tee -a $QLOG
while ! grep -q "NAVSIM FOLLOWUP DONE" $LOGS/navsim_followup_queue.log 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] followup done, launching extended experiments" | tee -a $QLOG

PICK_GPU() {
    for i in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$used" -lt 5000 ]; then echo $i; return; fi
    done
    echo 0
}

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

step() {
    local desc=$1; local cmd=$2; local log=$3
    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] STEP: $desc" | tee -a $QLOG
    bash -c "$cmd" > $log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        tail -3 $log | tee -a $QLOG
    else
        echo "[FAIL rc=$rc] see $log" | tee -a $QLOG
        tail -10 $log | tee -a $QLOG
    fi
    # Refresh report + commit after each step so user sees progress
    python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
    python $STATUS 2>&1 | tail -1 | tee -a $QLOG
    bash $COMMIT "extended: $desc" 2>&1 | tail -2 | tee -a $QLOG
}

# B1. Random baseline R1 drop=12 — control for greedy
G=$(PICK_GPU)
step "B1: R1 random baseline (drop=12, 3 seeds)" \
"CUDA_VISIBLE_DEVICES=$G python eval_random_baseline.py --backbone r1 \
    --n_drop 12 --n_seeds 3 --n_samples 100 \
    --device cuda:0 \
    --out_json $LOGS/random_baseline_r1_drop12.json" \
$LOGS/random_r1_drop12.log

# B2. Random baseline 1.5 drop=12 — same control on the other backbone
G=$(PICK_GPU)
step "B2: 1.5 random baseline (drop=12, 3 seeds)" \
"CUDA_VISIBLE_DEVICES=$G python eval_random_baseline.py --backbone 15 \
    --n_drop 12 --n_seeds 3 --n_samples 100 \
    --device cuda:0 \
    --out_json $LOGS/random_baseline_15_drop12.json" \
$LOGS/random_15_drop12.log

# C. Sample efficiency: R1 NAVSIM-calibrated greedy with N=10
G=$(PICK_GPU)
step "C: R1 NAVSIM greedy with N=10 calibration (sample efficiency)" \
"CUDA_VISIBLE_DEVICES=$G python run_iterative_greedy_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --n_samples 10 --rounds 12 \
    --device cuda:0 \
    --out_json $LOGS/greedyR1_navsim_N10.json" \
$LOGS/greedyR1_navsim_N10.log

# D. Eval the N=10 policy on the same holdout for cross-N comparison
G=$(PICK_GPU)
step "D: eval R1 NAVSIM-greedy(N=10) on NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$G python eval_alignment_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --policy_meta $LOGS/greedyR1_navsim_N10_meta.json \
    --out_json $LOGS/evalR1_navsimgreedy_N10_on_navsim.json --device cuda:0" \
$LOGS/eval_navsim_greedy_N10.log

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === EXTENDED QUEUE DONE ===" | tee -a $QLOG
python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
python $TRANSCRIPT 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "extended queue complete — random baselines + sample efficiency" 2>&1 | tail -3 | tee -a $QLOG
