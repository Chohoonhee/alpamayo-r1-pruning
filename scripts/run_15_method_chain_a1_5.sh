#!/bin/bash
# 1.5 method comparison chain using a1_5_venv server (RESTORED setup).
# Conditions (sample500 8-sample, single server per condition):
#   - baseline (already have 0.7197)
#   - greedy k=2 [23,31] (early-stop policy)
#   - greedy k=3 nuS
#   - random k=2 seed0/1/2
# Sequential; each ~17min wallclock.

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

QLOG=$LOGS/pdms_15_method_chain.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

run_condition() {
    local LABEL=$1 POLICY=$2
    if [ -f $LOGS/pdms_15_${LABEL}_a1_5_result.csv ]; then
        log "[skip] $LABEL already done"; return
    fi
    log "=== launching $LABEL ==="

    SLOG=$LOGS/pdms_15_${LABEL}_a1_5_srv.log
    EXTRA=""
    [ -n "$POLICY" ] && EXTRA="--drop_layers_json $POLICY"

    ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=0 nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py \
        --port 5557 $EXTRA > $SLOG 2>&1 &
    SP=$!
    log "  server PID=$SP"

    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 600 ]; then log "  timeout"; kill $SP 2>/dev/null; return 1; fi
        if ! kill -0 $SP 2>/dev/null; then log "  server died"; tail -10 $SLOG | tee -a $QLOG; return 1; fi
    done
    log "  server ready in ${waited}s"

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

    kill $SP 2>/dev/null
    sleep 5
    CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        cp "$CSV" $LOGS/pdms_15_${LABEL}_a1_5_result.csv
        log "  RESULT: $(tail -1 $CSV)"
    fi
    log "--- $LABEL done ---"

    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    bash $SCRIPTS/auto_commit.sh "1.5 a1_5_venv: $LABEL DONE" 2>&1 | tail -2 | tee -a $QLOG
}

log "=== 1.5 method chain in a1_5_venv (8-sample, sample500) ==="

run_condition "greedy_k2"     "$LOGS/greedy15_navsim_earlystop_meta.json"
run_condition "greedy_k3_nusc" "$LOGS/ksweep_15_nusc_k3.meta.json"
run_condition "random_k2_seed0" "$LOGS/random_baseline_15_drop2_seed0.meta.json"
run_condition "random_k2_seed1" "$LOGS/random_baseline_15_drop2_seed1.meta.json"
run_condition "random_k2_seed2" "$LOGS/random_baseline_15_drop2_seed2.meta.json"

log "=== 1.5 method chain DONE ==="
bash $SCRIPTS/auto_commit.sh "1.5 method chain a1_5_venv COMPLETE" 2>&1 | tail -2 | tee -a $QLOG
