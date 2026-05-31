#!/bin/bash
# Priority 2: Greedy stability — re-run R1 nuScenes greedy with 2 different
# calibration seeds. Tests if k=5 peak (0.95 align) is reproducible or lucky.
#
# Modifies run_iterative_greedy.py invocation to use different sample slices.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
QLOG=$LOGS/stability_queue.log

# Wait for K-sweep done
echo "[$(date +%H:%M:%S)] stability: waiting for K-sweep ..." | tee -a $QLOG
while ! grep -q "K-sweep DONE" $LOGS/k_sweep_queue.log 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] K-sweep done, launching stability runs" | tee -a $QLOG

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

# Stability: 3 greedy runs with different cal subsets (need a stride-based
# offset in run_iterative_greedy.py). For simplicity, we re-run R1
# NAVSIM greedy with samples 25:75 and 50:100 (different 50-sample windows).
# This tests whether k=3 peak [12,6,13] reproduces or shifts.

NAVSIM_PKL=$LOGS/navsim_samples_100.pkl

PICK_GPU() {
    for i in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$used" -lt 5000 ]; then echo $i; return; fi
    done
    echo 0
}

# Need new script entry that supports --sample_offset, but ours doesn't.
# Quick workaround: build subset pickle on the fly per seed.
build_subset_pkl() {
    local start=$1
    local end=$2
    local out=$3
    python -c "
import pickle
with open('$NAVSIM_PKL','rb') as f: d = pickle.load(f)
sub = d[$start:$end]
with open('$out','wb') as f: pickle.dump(sub, f)
print(f'wrote $out with {len(sub)} samples')
"
}

build_subset_pkl 25 75 $LOGS/navsim_samples_window_25_75.pkl
build_subset_pkl 0 25 $LOGS/navsim_samples_window_0_25.pkl  # smaller for N=25 stability

# Run 2 R1 NAVSIM-greedy seeds in parallel
G1=$(PICK_GPU); sleep 2; G2=$(PICK_GPU)
[ "$G2" = "$G1" ] && G2=$((G1 + 1))

nohup bash -c "
export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$G1
python run_iterative_greedy_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_window_25_75.pkl --n_samples 50 \
    --rounds 8 --device cuda:0 \
    --out_json $LOGS/greedyR1_navsim_seed1.json
" > $LOGS/greedyR1_navsim_seed1.log 2>&1 &
P1=$!
nohup bash -c "
export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$G2
python run_iterative_greedy_navsim.py --backbone 15 \
    --samples_pkl $LOGS/navsim_samples_window_25_75.pkl --n_samples 50 \
    --rounds 8 --device cuda:0 \
    --out_json $LOGS/greedy15_navsim_seed1.json
" > $LOGS/greedy15_navsim_seed1.log 2>&1 &
P2=$!

echo "stability seed1 R1 pid=$P1 gpu=$G1" | tee -a $QLOG
echo "stability seed1 1.5 pid=$P2 gpu=$G2" | tee -a $QLOG

wait $P1 $P2
echo "[$(date +%H:%M:%S)] stability seed1 done" | tee -a $QLOG

python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
python $STATUS 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "stability seed1: R1+1.5 NAVSIM-greedy on window 25:75" 2>&1 | tail -2 | tee -a $QLOG

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === STABILITY DONE ===" | tee -a $QLOG
