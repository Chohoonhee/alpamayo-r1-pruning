#!/bin/bash
# Maximum parallelism PDMS runner.
# 16 servers (2 per GPU × 8 GPUs), max_workers=24.
# ~3min per condition × sample500.
#
# Args:
#   $1 = label
#   $2 = policy_meta JSON path ("" for baseline)
#   $3 = variant (r1 or 15)
#   $4 = num_traj_samples (1 or 8)
#   $5 = N scenes (default 500)

set -uo pipefail
LABEL=$1
POLICY=${2:-}
VARIANT=${3:-15}
NTRAJ=${4:-8}
N=${5:-500}

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
CFG_DIR=$NAVSIM_ROOT/navsim/planning/script/config/common/agent

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

QLOG=$LOGS/pdmsmax_${LABEL}.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== MAX PARALLEL PDMS: $LABEL  variant=$VARIANT  ntraj=$NTRAJ  N=$N ==="

# Phase 1: 16 servers (2 per GPU × 8 GPUs), ports 6100..6115
PORT_BASE=6100
SERVER_PIDS=()
for i in $(seq 0 15); do
    GPU=$((i / 2))   # 2 servers per GPU
    PORT=$((PORT_BASE + i))
    SLOG=$LOGS/pdmsmax_${LABEL}_srv_${PORT}.log
    EXTRA=""
    [ -n "$POLICY" ] && EXTRA="--drop_layers_json $POLICY"
    ALPAMAYO_VARIANT=$VARIANT CUDA_VISIBLE_DEVICES=$GPU nohup python $SCRIPTS/alpamayo_server.py \
        --port $PORT $EXTRA > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
done
log "launched 16 servers (2/GPU × 8 GPUs), ports $PORT_BASE-$((PORT_BASE+15))"

# Wait for all ready
ready_count=0
for i in $(seq 0 15); do
    PORT=$((PORT_BASE + i))
    SLOG=$LOGS/pdmsmax_${LABEL}_srv_${PORT}.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 600 ]; then
            log "server $PORT timeout"
            for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done; exit 1
        fi
    done
    ready_count=$((ready_count+1))
done
log "all $ready_count servers ready"

# Build server_addr list
SERVERS=""
for i in $(seq 0 15); do
    PORT=$((PORT_BASE + i))
    SERVERS="${SERVERS}tcp://127.0.0.1:$PORT,"
done
SERVERS=${SERVERS%,}  # trim trailing comma

# Generate dynamic agent yaml
DYN_YAML=alpamayo_agent_max_${NTRAJ}sample
cat > $CFG_DIR/${DYN_YAML}.yaml <<EOF
_target_: navsim.agents.alpamayo_agent.AlpamayoNAVSIMAgent
_convert_: 'all'

server_addr: "$SERVERS"
num_traj_samples: $NTRAJ
max_generation_length: 128
timeout_s: 900

trajectory_sampling:
  _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  _convert_: 'all'
  time_horizon: 4
  interval_length: 0.5
EOF

EXPNAME=alpamayo_pdmsmax_${VARIANT}_${LABEL}
cd $NAVSIM_ROOT
$NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=$DYN_YAML \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=$N \
    metric_cache_path=$CACHE \
    experiment_name=$EXPNAME \
    worker=single_machine_thread_pool \
    worker.max_workers=24 \
    2>&1 | tee -a $QLOG

# Cleanup
for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
sleep 5

CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    log "result CSV: $CSV"
    log "aggregate:"
    tail -1 "$CSV" | tee -a $QLOG
    cp "$CSV" $LOGS/pdmsmax_${LABEL}_result.csv
fi
log "=== MAX PARALLEL DONE: $LABEL ==="
