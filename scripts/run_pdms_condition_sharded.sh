#!/bin/bash
# Run NAVSIM PDMS for one R1 condition, sharded 8-way across GPU 0-7.
# Each shard uses 1 GPU + 1 server + 1/8 of navtest (~1518 scenes).
# Wallclock ~35 min vs ~2h single-GPU.
#
# Args:
#   $1 = label
#   $2 = policy_meta JSON path ("" for baseline)
#   $3 = N_SHARDS (default 8)

set -uo pipefail
LABEL=$1
POLICY=${2:-}
N_SHARDS=${3:-8}

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

QLOG=$LOGS/pdmsfull_${LABEL}.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== FULL PDMS run: $LABEL  policy=${POLICY:-baseline}  ${N_SHARDS}-way shard ==="

# Phase 1: launch one server per GPU (8 servers on ports 6000..6007)
SERVER_PIDS=()
PORT_BASE=6000
for i in $(seq 0 $((N_SHARDS-1))); do
    GPU=$i
    PORT=$((PORT_BASE + i))
    SLOG=$LOGS/pdmsfull_${LABEL}_srv_${PORT}.log
    EXTRA=""
    [ -n "$POLICY" ] && EXTRA="--drop_layers_json $POLICY"
    ALPAMAYO_VARIANT=r1 CUDA_VISIBLE_DEVICES=$GPU nohup python $SCRIPTS/alpamayo_server.py \
        --port $PORT $EXTRA > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
done
log "launched ${#SERVER_PIDS[@]} servers on ports $PORT_BASE..$((PORT_BASE+N_SHARDS-1))"

# Phase 2: wait for all ready
for i in $(seq 0 $((N_SHARDS-1))); do
    PORT=$((PORT_BASE + i))
    SLOG=$LOGS/pdmsfull_${LABEL}_srv_${PORT}.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 300 ]; then
            log "server $PORT timeout"; for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done; exit 1
        fi
        if ! kill -0 ${SERVER_PIDS[$i]} 2>/dev/null; then
            log "server $PORT died"; for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done; exit 1
        fi
    done
done
log "all $N_SHARDS servers ready"

# Phase 3: launch N_SHARDS Navsim PDMS scorers in parallel
# Each scorer talks to ONE server (its own dedicated one) via single addr.
SCORER_PIDS=()
for i in $(seq 0 $((N_SHARDS-1))); do
    PORT=$((PORT_BASE + i))
    SHARD_YAML=navtest_shard${i}of${N_SHARDS}
    EXPNAME=alpamayo_pdmsfull_r1_${LABEL}_shard${i}
    SCORER_LOG=$LOGS/pdmsfull_${LABEL}_shard${i}.log
    log "  scorer shard $i → port $PORT  yaml=$SHARD_YAML"
    cd $NAVSIM_ROOT
    nohup $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=alpamayo_agent \
        "agent.server_addr='tcp://127.0.0.1:$PORT'" \
        train_test_split=navtest \
        train_test_split/scene_filter=$SHARD_YAML \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=4 \
        > $SCORER_LOG 2>&1 &
    SCORER_PIDS+=($!)
done
log "launched ${#SCORER_PIDS[@]} parallel scorers"

# Phase 4: wait for all scorers
for pid in "${SCORER_PIDS[@]}"; do wait $pid; done
log "all scorers done"

# Phase 5: kill servers
for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
sleep 5

# Phase 6: collect per-shard CSVs and aggregate
SHARD_CSVS=()
for i in $(seq 0 $((N_SHARDS-1))); do
    EXPNAME=alpamayo_pdmsfull_r1_${LABEL}_shard${i}
    CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        cp "$CSV" $LOGS/pdmsfull_${LABEL}_shard${i}.csv
        SHARD_CSVS+=("$LOGS/pdmsfull_${LABEL}_shard${i}.csv")
    else
        log "  WARN: no CSV for shard $i (exp=$EXPNAME)"
    fi
done
log "collected ${#SHARD_CSVS[@]} shard CSVs, aggregating"
python3 $SCRIPTS/aggregate_pdms_shards.py "${SHARD_CSVS[@]}" \
    --out $LOGS/pdmsfull_${LABEL}_result.csv 2>&1 | tail -20 | tee -a $QLOG
log "=== FULL PDMS DONE: $LABEL ==="
