#!/bin/bash
# Long sequential experiment queue. Runs while user is away.
# Each step:
#   1. runs the experiment
#   2. calls update_status.py
#   3. calls auto_commit.sh with a label
#   4. continues to next step regardless of failure (best-effort)
#
# Designed to be robust: any single failure logs but doesn't kill the queue.

set -uo pipefail  # NOT -e (we want to continue past failures)
source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
QLOG=$LOGS/sequential_queue.log

echo "[$(date +%H:%M:%S)] queue start" | tee -a $QLOG

step() {
    local desc="$1"
    echo "" | tee -a $QLOG
    echo "=================================================================" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] STEP: $desc" | tee -a $QLOG
    echo "=================================================================" | tee -a $QLOG
}

finalize() {
    local desc="$1"
    python $STATUS 2>&1 | tail -2 | tee -a $QLOG
    python $TRANSCRIPT 2>&1 | tail -2 | tee -a $QLOG
    bash $COMMIT "$desc" 2>&1 | tail -3 | tee -a $QLOG
}

cd $SCRIPTS

# ───── PHASE 0: initial commit ──────────────────────────────────────────────
step "phase 0: initial status + transcript"
finalize "queue started — initial status snapshot"

# ───── PHASE 1: ea_vlm28 alignment eval (compare against ours) ─────────────
# Existing pruning method on 1.5: ea_vlm28 drops 28 layers. We want to know
# its alignment for direct comparison with our methods.
EA28_DROP=$REPO/scripts/logs/policy15_ea28.json
python -c "
import json
m = json.load(open('/home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B-pruned-expertaware_vlm28/pruning_meta.json'))
out = {'dropped_layers': sorted(m['dropped_layers']),
       'policy': 'ea_vlm_existing', 'backbone': '15',
       'source': 'ea_vlm28 weight dir',
       'rationale': 'Existing ea-vlm pruning method (28 drop). For comparison.'}
json.dump(out, open('$EA28_DROP', 'w'), indent=2)
print(f'ea_vlm28 policy meta written: {len(out[\"dropped_layers\"])} dropped layers')
"
step "phase 1: 1.5 ea_vlm28 zero-shot eval (drop=28)"
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot_alignment.py \
    --policy_meta $EA28_DROP \
    --n_samples 100 \
    --out_json logs/eval15_ea28_baseline_method.json \
    --device cuda:0 2>&1 | tail -5 | tee -a $QLOG
finalize "phase 1: ea_vlm28 alignment eval (1.5, 28 drop)"

# ───── PHASE 2: Stage 2 v2 on R1 plus_harmful ─────────────────────────────
step "phase 2a: Stage 2 v2 train on R1 plus_harmful (Expert-only, DDP)"
OUT_R1=/home/irteam/ws/alpamayo_pruning/weights/sft_stage2_v2_R1_plus_harmful
mkdir -p $OUT_R1
# (Note: sft_stage2_expert_only uses Alpamayo 1.5 imports — would need R1 variant.
#  Skip if not adapted; fall back to 1.5 retry.)
# For now, skip R1 training. Comment in if you write sft_stage2_expert_only_r1.py.
echo "[skip] R1 Stage 2 v2 needs R1-specific sft script; not built yet." | tee -a $QLOG

# ───── PHASE 3: 500-sample re-pilot on 1.5 (sharded across 4 GPUs) ─────────
step "phase 3a: 1.5 per-layer pilot, 500 samples, 4-shard"
L0="0,1,2,3,4,5,6,7,8"
L1="9,10,11,12,13,14,15,16,17"
L2="18,19,20,21,22,23,24,25,26"
L3="27,28,29,30,31,32,33,34,35"

run_pilot_shard() {
    local gpu=$1 layers=$2 tag=$3 backbone=$4 nsamples=$5
    local script="measure_alignment_delta.py"
    [ "$backbone" = "r1" ] && script="measure_alignment_delta_r1.py"
    local out="logs/pilot${backbone}_500_${tag}.json"
    CUDA_VISIBLE_DEVICES=$gpu python $script \
        --n_samples $nsamples --layers "$layers" \
        --out_json $out --device cuda:0
}

# Run shards in parallel for 1.5
run_pilot_shard 0 "$L0" shard0 15 500 &
run_pilot_shard 1 "$L1" shard1 15 500 &
run_pilot_shard 2 "$L2" shard2 15 500 &
run_pilot_shard 3 "$L3" shard3 15 500 &
wait
echo "[$(date +%H:%M:%S)] 1.5 500-sample pilot done" | tee -a $QLOG

# Merge + plot + commit
python merge_alignment_shards.py \
    logs/pilot15_500_shard0.json logs/pilot15_500_shard1.json \
    logs/pilot15_500_shard2.json logs/pilot15_500_shard3.json \
    --out logs/pilot15_500_merged.json \
    --csv logs/pilot15_500_merged.csv --eps 0.02 2>&1 | tail -10 | tee -a $QLOG
python plot_alignment_layers.py logs/pilot15_500_merged.json \
    --out logs/pilot15_500_per_layer.png \
    --title "Alpamayo 1.5 — per-layer importance (500 samples)" 2>&1 | tail -2 | tee -a $QLOG
python alignment_policy_to_meta.py logs/pilot15_500_merged.json \
    --neutral_out logs/policy15_500_neutral.json \
    --plus_harmful_out logs/policy15_500_plus_harmful.json \
    --backbone 15 2>&1 | tail -3 | tee -a $QLOG
finalize "phase 3a: 1.5 500-sample re-pilot"

# ───── PHASE 4: 500-sample R1 pilot ───────────────────────────────────────
step "phase 4: R1 per-layer pilot, 500 samples, 4-shard"
run_pilot_shard 4 "$L0" shard0 R1 500 &
run_pilot_shard 5 "$L1" shard1 R1 500 &
run_pilot_shard 6 "$L2" shard2 R1 500 &
run_pilot_shard 7 "$L3" shard3 R1 500 &
wait
echo "[$(date +%H:%M:%S)] R1 500-sample pilot done" | tee -a $QLOG

python merge_alignment_shards.py \
    logs/pilotR1_500_shard0.json logs/pilotR1_500_shard1.json \
    logs/pilotR1_500_shard2.json logs/pilotR1_500_shard3.json \
    --out logs/pilotR1_500_merged.json \
    --csv logs/pilotR1_500_merged.csv --eps 0.02 2>&1 | tail -10 | tee -a $QLOG
python plot_alignment_layers.py logs/pilotR1_500_merged.json \
    --out logs/pilotR1_500_per_layer.png \
    --title "Alpamayo R1 — per-layer importance (500 samples)" 2>&1 | tail -2 | tee -a $QLOG
python alignment_policy_to_meta.py logs/pilotR1_500_merged.json \
    --neutral_out logs/policyR1_500_neutral.json \
    --plus_harmful_out logs/policyR1_500_plus_harmful.json \
    --backbone r1 2>&1 | tail -3 | tee -a $QLOG
# side-by-side at 500
python plot_alignment_compare.py \
    --r1 logs/pilotR1_500_merged.json \
    --v15 logs/pilot15_500_merged.json \
    --out logs/compare_r1_v15_500_per_layer.png 2>&1 | tail -2 | tee -a $QLOG
finalize "phase 4: R1 500-sample re-pilot + side-by-side plot"

# ───── PHASE 5: eval new 500-sample policies ──────────────────────────────
step "phase 5: eval new 500-sample policies on 100 nuScenes val"
CUDA_VISIBLE_DEVICES=0 python eval_zeroshot_alignment.py \
    --policy_meta logs/policy15_500_neutral.json \
    --n_samples 100 --out_json logs/eval15_500_neutral.json --device cuda:0 \
    2>&1 | tail -5 | tee -a $QLOG &
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot_alignment.py \
    --policy_meta logs/policy15_500_plus_harmful.json \
    --n_samples 100 --out_json logs/eval15_500_plus_harmful.json --device cuda:0 \
    2>&1 | tail -5 | tee -a $QLOG &
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot_alignment_r1.py \
    --policy_meta logs/policyR1_500_neutral.json \
    --n_samples 100 --out_json logs/evalR1_500_neutral.json --device cuda:0 \
    2>&1 | tail -5 | tee -a $QLOG &
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot_alignment_r1.py \
    --policy_meta logs/policyR1_500_plus_harmful.json \
    --n_samples 100 --out_json logs/evalR1_500_plus_harmful.json --device cuda:0 \
    2>&1 | tail -5 | tee -a $QLOG &
wait
finalize "phase 5: 500-pilot-derived policies evaluated"

# ───── PHASE 6: R1 iterative-greedy (since 1.5 greedy was on GPU 7) ───────
step "phase 6: R1 iterative-greedy, 12 rounds, 50 samples"
CUDA_VISIBLE_DEVICES=4 python run_iterative_greedy.py --backbone r1 \
    --n_samples 50 --rounds 12 \
    --out_json logs/greedyR1.json --device cuda:0 2>&1 | tail -20 | tee -a $QLOG
finalize "phase 6: R1 iterative-greedy done"

# ───── PHASE 7: eval greedy outcomes ──────────────────────────────────────
step "phase 7a: eval 1.5 greedy final drop set"
# greedy15.json may not exist if greedy is still running — check
if [ -f logs/greedy15_meta.json ]; then
    CUDA_VISIBLE_DEVICES=0 python eval_zeroshot_alignment.py \
        --policy_meta logs/greedy15_meta.json \
        --n_samples 100 --out_json logs/eval15_greedy.json --device cuda:0 \
        2>&1 | tail -5 | tee -a $QLOG
fi
step "phase 7b: eval R1 greedy final drop set"
if [ -f logs/greedyR1_meta.json ]; then
    CUDA_VISIBLE_DEVICES=1 python eval_zeroshot_alignment_r1.py \
        --policy_meta logs/greedyR1_meta.json \
        --n_samples 100 --out_json logs/evalR1_greedy.json --device cuda:0 \
        2>&1 | tail -5 | tee -a $QLOG
fi
finalize "phase 7: greedy outcome evaluations"

# ───── PHASE 8: final summary ─────────────────────────────────────────────
step "phase 8: final status + transcript snapshot"
finalize "queue complete — final snapshot"

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] ALL PHASES DONE" | tee -a $QLOG
