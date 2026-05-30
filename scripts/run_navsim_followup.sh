#!/bin/bash
# Waits for both NAVSIM-greedy runs to produce meta JSONs, then fires off
# the cross-domain evaluation matrix:
#   R1 navsim-greedy  → NAVSIM holdout (50:100)
#   R1 navsim-greedy  → nuScenes 100 val
#   1.5 navsim-greedy → NAVSIM holdout
#   1.5 navsim-greedy → nuScenes 100 val
# Plus baselines + nuScenes-greedy → NAVSIM holdout for symmetry.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
QLOG=$LOGS/navsim_followup_queue.log

R1_GREEDY_META=$LOGS/greedyR1_navsim_meta.json
V15_GREEDY_META=$LOGS/greedy15_navsim_meta.json

echo "[$(date +%H:%M:%S)] follow-up: waiting for NAVSIM-greedy meta files ..." | tee -a $QLOG
while [ ! -f $R1_GREEDY_META ] || [ ! -f $V15_GREEDY_META ]; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] both metas found, launching evals" | tee -a $QLOG

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

eval_one() {
    local desc=$1
    local cmd=$2
    local log=$3
    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] eval: $desc" | tee -a $QLOG
    bash -c "$cmd" > $log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        tail -5 $log | tee -a $QLOG
    else
        echo "[FAIL rc=$rc] see $log" | tee -a $QLOG
        tail -10 $log | tee -a $QLOG
    fi
}

GPU0=$(PICK_GPU)
GPU1=$(PICK_GPU)
[ "$GPU1" = "$GPU0" ] && GPU1=$((GPU0 + 1))

eval_one "R1 navsim-greedy → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU0 python eval_alignment_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --policy_meta $R1_GREEDY_META \
    --out_json $LOGS/evalR1_navsimgreedy_on_navsim.json --device cuda:0" \
$LOGS/eval_r1_navsim_on_navsim.log

eval_one "1.5 navsim-greedy → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU1 python eval_alignment_navsim.py --backbone 15 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --policy_meta $V15_GREEDY_META \
    --out_json $LOGS/eval15_navsimgreedy_on_navsim.json --device cuda:0" \
$LOGS/eval_15_navsim_on_navsim.log

eval_one "R1 navsim-greedy → nuScenes val" \
"CUDA_VISIBLE_DEVICES=$GPU0 python eval_zeroshot_alignment_r1.py \
    --policy_meta $R1_GREEDY_META --n_samples 100 \
    --out_json $LOGS/evalR1_navsimgreedy_on_nuscenes.json --device cuda:0" \
$LOGS/eval_r1_navsim_on_nusc.log

eval_one "1.5 navsim-greedy → nuScenes val" \
"CUDA_VISIBLE_DEVICES=$GPU1 python eval_zeroshot_alignment.py \
    --policy_meta $V15_GREEDY_META --n_samples 100 \
    --out_json $LOGS/eval15_navsimgreedy_on_nuscenes.json --device cuda:0" \
$LOGS/eval_15_navsim_on_nusc.log

NUSC_R1_GREEDY=$LOGS/greedyR1_meta.json
NUSC_V15_GREEDY=$LOGS/greedy15_meta.json
[ -f $NUSC_R1_GREEDY ] && eval_one "R1 nuScenes-greedy → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU0 python eval_alignment_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --policy_meta $NUSC_R1_GREEDY \
    --out_json $LOGS/evalR1_nuscgreedy_on_navsim.json --device cuda:0" \
$LOGS/eval_r1_nusc_on_navsim.log
[ -f $NUSC_V15_GREEDY ] && eval_one "1.5 nuScenes-greedy → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU1 python eval_alignment_navsim.py --backbone 15 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --policy_meta $NUSC_V15_GREEDY \
    --out_json $LOGS/eval15_nuscgreedy_on_navsim.json --device cuda:0" \
$LOGS/eval_15_nusc_on_navsim.log

eval_one "R1 baseline → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU0 python eval_alignment_navsim.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --out_json $LOGS/evalR1_baseline_on_navsim_holdout.json --device cuda:0" \
$LOGS/eval_r1_baseline_on_navsim.log
eval_one "1.5 baseline → NAVSIM holdout" \
"CUDA_VISIBLE_DEVICES=$GPU1 python eval_alignment_navsim.py --backbone 15 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --sample_slice 50:100 \
    --out_json $LOGS/eval15_baseline_on_navsim_holdout.json --device cuda:0" \
$LOGS/eval_15_baseline_on_navsim.log

python $STATUS 2>&1 | tail -2 | tee -a $QLOG
python $TRANSCRIPT 2>&1 | tail -2 | tee -a $QLOG
bash $COMMIT "cross-domain matrix: NAVSIM/nuScenes × calibrated-on-{NAVSIM,nuScenes}" 2>&1 | tail -3 | tee -a $QLOG

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === NAVSIM FOLLOWUP DONE ===" | tee -a $QLOG
