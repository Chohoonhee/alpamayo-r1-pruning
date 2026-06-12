#!/bin/bash
# 1.5 ANGULAR distance K-sweep in a1_5_venv.
# Test: does angular-distance pruning preserve PDMS better than greedy?

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim

export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
CACHE=$NAVSIM_EXP_ROOT/metric_cache

QLOG=$LOGS/pdms_15_angular_chain.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# Wait for current K=1 chain to free GPU
while pgrep -f "alpamayo_server.py --port 5557" > /dev/null; do
    log "waiting for current chain..."; sleep 60
done

run_one() {
    local LABEL=$1 POLICY=$2
    if [ -f $LOGS/pdms_15_${LABEL}_a1_5_result.csv ]; then
        log "[skip] $LABEL done"; return
    fi
    log "=== $LABEL ==="
    SLOG=$LOGS/pdms_15_${LABEL}_a1_5_srv.log
    ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=0 nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py \
        --port 5557 --drop_layers_json $POLICY > $SLOG 2>&1 &
    SP=$!
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 600 ] || ! kill -0 $SP 2>/dev/null; then
            log "  server fail"; kill $SP 2>/dev/null; return 1
        fi
    done
    log "  server ready ${waited}s"
    EXPNAME=alpamayo_15_${LABEL}_a1_5
    cd $NAVSIM_ROOT
    $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=alpamayo_agent_8sample \
        train_test_split=navtest \
        train_test_split.scene_filter.max_scenes=500 \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=2 2>&1 | tee -a $QLOG
    kill $SP 2>/dev/null; sleep 5
    CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    [ -n "$CSV" ] && cp "$CSV" $LOGS/pdms_15_${LABEL}_a1_5_result.csv && log "  RESULT: $(tail -1 $CSV)"
    log "--- $LABEL done ---"
}

log "=== 1.5 ANGULAR K-sweep ==="
run_one "angular_k1" "$LOGS/angular_15_k1_layer32_meta.json"      # [32]
run_one "angular_k2" "$LOGS/angular_15_k2_layers32_31_meta.json"  # [32,31]
run_one "angular_k3" "$LOGS/angular_15_k3_layers32_31_27_meta.json" # [32,31,27]
run_one "angular_k5" "$LOGS/angular_15_k5_meta.json"               # [32,31,27,33,26]
log "=== angular chain DONE ==="
