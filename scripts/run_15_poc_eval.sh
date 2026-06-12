#!/bin/bash
# Evaluate PoC fine-tuned 1.5 checkpoint: PDMS sample500 8-sample in a1_5_venv.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
POC_WEIGHTS=/home/irteam/ws/alpamayo_pruning/weights/sft_poc_15_k2/lora_final

export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
CACHE=$NAVSIM_EXP_ROOT/metric_cache

QLOG=$LOGS/pdms_15_poc_eval.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }
log "=== PoC FT eval: 1.5 + K=[23,31] + Expert LoRA (1k train, 1 epoch) ==="

# Launch server with PoC weights + runtime drop
SLOG=$LOGS/pdms_15_poc_srv.log
ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=0 nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py \
    --weights $POC_WEIGHTS \
    --port 5557 \
    --drop_layers_json $LOGS/greedy15_navsim_earlystop_meta.json > $SLOG 2>&1 &
SP=$!
log "server PID=$SP"

waited=0
while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
    sleep 5; waited=$((waited+5))
    if [ $waited -gt 600 ]; then log "timeout"; kill $SP 2>/dev/null; exit 1; fi
    if ! kill -0 $SP 2>/dev/null; then log "server died"; tail -20 $SLOG | tee -a $QLOG; exit 1; fi
done
log "server ready in ${waited}s"

# Use 8-sample agent yaml (already configured for single server)
cd $NAVSIM_ROOT
$NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent_8sample \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=500 \
    metric_cache_path=$CACHE \
    experiment_name=alpamayo_15_poc_eval \
    worker=single_machine_thread_pool \
    worker.max_workers=2 2>&1 | tee -a $QLOG

kill $SP 2>/dev/null
sleep 5
CSV=$(ls -1t $NAVSIM_EXP_ROOT/alpamayo_15_poc_eval/*/202*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && cp "$CSV" $LOGS/pdms_15_poc_eval_result.csv && log "RESULT: $(tail -1 $CSV)"
log "=== DONE ==="
