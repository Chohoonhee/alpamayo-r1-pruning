#!/bin/bash
# Parallel launch: remaining 1.5 K=1 + angular conditions, each on a separate GPU.
# Currently GPU 0 = k1_layer23 still running. Use GPU 1-5 for 5 parallel conditions.

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

QLOG=$LOGS/pdms_15_parallel.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# (GPU | port | label | policy_meta | yaml_to_use)
declare -a TASKS=(
    "1|5558|k1_layer31|$LOGS/single_layer31_meta.json"
    "2|5559|angular_k1|$LOGS/angular_15_k1_layer32_meta.json"
    "3|5560|angular_k2|$LOGS/angular_15_k2_layers32_31_meta.json"
    "4|5561|angular_k3|$LOGS/angular_15_k3_layers32_31_27_meta.json"
    "5|5562|angular_k5|$LOGS/angular_15_k5_meta.json"
)

# 1) Launch all 5 servers in parallel
log "=== launching 5 parallel servers (GPU 1-5, ports 5558-5562) ==="
SERVER_PIDS=()
for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU PORT LABEL POLICY <<<"$entry"
    SLOG=$LOGS/pdms_15_${LABEL}_parallel_srv.log
    ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=$GPU nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py \
        --port $PORT --drop_layers_json $POLICY > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
    log "  $LABEL on GPU $GPU port $PORT pid=${SERVER_PIDS[-1]}"
done

# 2) Wait for all ready
log "waiting for all servers ready..."
for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU PORT LABEL POLICY <<<"$entry"
    SLOG=$LOGS/pdms_15_${LABEL}_parallel_srv.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 600 ]; then log "  $LABEL TIMEOUT"; exit 1; fi
    done
    log "  $LABEL ready (${waited}s)"
done

# 3) Launch 5 scorers in parallel, each with its own agent yaml
# We need 5 different agent yamls pointing to different ports
log "=== launching 5 parallel scorers ==="
SCORER_PIDS=()
for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU PORT LABEL POLICY <<<"$entry"
    YAML_NAME=alpamayo_agent_8sample_${LABEL}
    cat > $NAVSIM_ROOT/navsim/planning/script/config/common/agent/${YAML_NAME}.yaml <<EOF
_target_: navsim.agents.alpamayo_agent.AlpamayoNAVSIMAgent
_convert_: 'all'
server_addr: "tcp://127.0.0.1:$PORT"
num_traj_samples: 8
max_generation_length: 128
timeout_s: 900
trajectory_sampling:
  _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  _convert_: 'all'
  time_horizon: 4
  interval_length: 0.5
EOF

    EXPNAME=alpamayo_15_${LABEL}_a1_5
    SLG=$LOGS/pdms_15_${LABEL}_parallel.log
    cd $NAVSIM_ROOT
    nohup $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=$YAML_NAME \
        train_test_split=navtest \
        train_test_split.scene_filter.max_scenes=500 \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=2 > $SLG 2>&1 &
    SCORER_PIDS+=($!)
    log "  $LABEL scorer pid=${SCORER_PIDS[-1]}"
done

# 4) Wait all scorers
log "waiting for all scorers..."
for pid in "${SCORER_PIDS[@]}"; do wait $pid; done
log "all scorers done"

# 5) Kill servers, collect results
for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null; done
sleep 5

for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU PORT LABEL POLICY <<<"$entry"
    EXPNAME=alpamayo_15_${LABEL}_a1_5
    CSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    [ -n "$CSV" ] && cp "$CSV" $LOGS/pdms_15_${LABEL}_a1_5_result.csv && log "RESULT $LABEL: $(tail -1 $CSV)"
done
log "=== parallel chain DONE ==="
