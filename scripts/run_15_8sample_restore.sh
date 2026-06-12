#!/bin/bash
# Restore 1.5 8-sample baseline (April 0.7286). 4 servers GPU 0-3, ports 5557-5560.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
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

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

QLOG=$LOGS/pdms_15_8sample_restore.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== 1.5 8-SAMPLE RESTORE TEST ==="
log "  target: April reference 0.7286"

# Launch 4 servers (1.5 baseline, no pruning)
SERVER_PIDS=()
for i in 0 1 2 3; do
    GPU=$i; PORT=$((5557+i))
    SLOG=$LOGS/pdms_15_8sample_srv_${PORT}.log
    ALPAMAYO_VARIANT=15 CUDA_VISIBLE_DEVICES=$GPU nohup python $SCRIPTS/alpamayo_server.py \
        --port $PORT > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
done
log "launched 4 servers, waiting for ready..."

for i in 0 1 2 3; do
    PORT=$((5557+i))
    SLOG=$LOGS/pdms_15_8sample_srv_${PORT}.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 300 ]; then
            log "server $PORT timeout"; for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done; exit 1
        fi
    done
done
log "all 4 servers ready"

EXPNAME=alpamayo_15_8sample_restore
cd $NAVSIM_ROOT
$NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent_8sample \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=500 \
    metric_cache_path=$CACHE \
    experiment_name=$EXPNAME \
    worker=single_machine_thread_pool \
    worker.max_workers=8 \
    2>&1 | tee -a $QLOG

# Kill servers
for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
sleep 5

# Find result
CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    log "result CSV: $CSV"
    log "aggregate row:"
    tail -1 "$CSV" | tee -a $QLOG
    cp "$CSV" $LOGS/pdms_15_8sample_restore_result.csv
fi
log "=== DONE ==="
