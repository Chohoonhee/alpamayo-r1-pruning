#!/bin/bash
# Run NAVSIM PDMS scoring for one R1 condition.
# Args:
#   $1 = label (e.g. "greedy_k3_nusc")
#   $2 = policy_meta JSON path (empty string "" for baseline)
#   $3 = first GPU index (uses 4 consecutive GPUs starting here)
#   $4 = first server port (uses 4 consecutive ports starting here)
#   $5 = max_scenes (default 500)
#
# Pattern: 4 alpamayo_server instances (one per GPU) on 4 ports,
# then NAVSIM PDMS scorer talks to all 4 via round-robin.

set -uo pipefail
LABEL=$1
POLICY=${2:-}
GPU0=${3:-0}
PORT0=${4:-5557}
N=${5:-500}
VARIANT=${6:-r1}

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

QLOG=$LOGS/pdms_${LABEL}.log
EXPNAME=alpamayo_pdms_${VARIANT}_${LABEL}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== PDMS run: $LABEL  policy=${POLICY:-baseline}  gpus=$GPU0..$((GPU0+3))  ports=$PORT0..$((PORT0+3))  max_scenes=$N ==="

# 1. Launch 4 servers
SERVER_PIDS=()
for i in 0 1 2 3; do
    GPU=$((GPU0+i))
    PORT=$((PORT0+i))
    SLOG=$LOGS/pdms_${LABEL}_server_${PORT}.log
    EXTRA=""
    [ -n "$POLICY" ] && EXTRA="--drop_layers_json $POLICY"
    log "  server port=$PORT gpu=$GPU policy=${POLICY:-none}"
    ALPAMAYO_VARIANT=$VARIANT CUDA_VISIBLE_DEVICES=$GPU nohup python $SCRIPTS/alpamayo_server.py \
        --port $PORT $EXTRA > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
done
log "  server PIDs: ${SERVER_PIDS[*]}"

# 2. Wait for all servers ready (max 5 min)
log "  waiting for servers to be ready..."
for i in 0 1 2 3; do
    PORT=$((PORT0+i))
    SLOG=$LOGS/pdms_${LABEL}_server_${PORT}.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 300 ]; then
            log "  server $PORT timeout — see $SLOG"
            for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null; done
            exit 1
        fi
        if ! kill -0 ${SERVER_PIDS[$i]} 2>/dev/null; then
            log "  server $PORT died — see $SLOG"
            for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null; done
            exit 1
        fi
    done
    log "  server $PORT ready (${waited}s)"
done

# 3. Run PDMS scorer via navsim_venv
SERVERS="tcp://127.0.0.1:$PORT0,tcp://127.0.0.1:$((PORT0+1)),tcp://127.0.0.1:$((PORT0+2)),tcp://127.0.0.1:$((PORT0+3))"
log "  starting PDMS scorer (servers=$SERVERS)"

cd $NAVSIM_ROOT
$NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent \
    "agent.server_addr='$SERVERS'" \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=$N \
    metric_cache_path=$CACHE \
    experiment_name=$EXPNAME \
    worker=single_machine_thread_pool \
    worker.max_workers=8 \
    2>&1 | tee -a $QLOG
RC=$?
log "  PDMS scorer rc=$RC"

# 4. Kill servers
for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null; done
sleep 5

# 5. Find and report the result CSV
CSV=$(find $NAVSIM_EXP_ROOT/$EXPNAME -name "*.csv" -newer $QLOG 2>/dev/null | head -1)
if [ -z "$CSV" ]; then
    CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
fi
if [ -n "$CSV" ] && [ -f "$CSV" ]; then
    log "  result CSV: $CSV"
    log "  aggregate row:"
    tail -1 "$CSV" | tee -a $QLOG
    cp "$CSV" $LOGS/pdms_${LABEL}_result.csv
    log "  saved to $LOGS/pdms_${LABEL}_result.csv"
else
    log "  WARN: no result CSV found under $NAVSIM_EXP_ROOT/$EXPNAME"
fi

log "=== PDMS DONE: $LABEL ==="
