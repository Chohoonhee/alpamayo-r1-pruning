#!/bin/bash
# PoC eval parallelized: 4 servers on GPU 0-3, ports 5557-5560, max_workers=8

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
POC_WEIGHTS=/home/irteam/ws/alpamayo_pruning/weights/sft_full_15_k2_v3/lora_step_400

export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
CACHE=$NAVSIM_EXP_ROOT/metric_cache

QLOG=$LOGS/pdms_15_poc_parallel.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }
log "=== PoC PARALLEL eval (4 GPU, 4 servers) ==="

SERVER_PIDS=()
for i in 0 1 2 3; do
    GPU=$i
    PORT=$((5557+i))
    SLOG=$LOGS/pdms_15_poc_parallel_srv_${PORT}.log
    ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=$GPU nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py \
        --weights $POC_WEIGHTS \
        --port $PORT \
        --drop_layers_json $LOGS/greedy15_navsim_earlystop_meta.json > $SLOG 2>&1 &
    SERVER_PIDS+=($!)
    log "  GPU $GPU port $PORT pid=${SERVER_PIDS[-1]}"
done
log "waiting for all servers ready..."
for i in 0 1 2 3; do
    PORT=$((5557+i))
    SLOG=$LOGS/pdms_15_poc_parallel_srv_${PORT}.log
    waited=0
    while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 600 ]; then log "  $PORT TIMEOUT"; exit 1; fi
    done
done
log "all 4 servers ready"

# Update agent yaml for 4 servers
cat > $NAVSIM_ROOT/navsim/planning/script/config/common/agent/alpamayo_agent_8sample.yaml <<'EOF'
_target_: navsim.agents.alpamayo_agent.AlpamayoNAVSIMAgent
_convert_: 'all'
server_addr: "tcp://127.0.0.1:5557,tcp://127.0.0.1:5558,tcp://127.0.0.1:5559,tcp://127.0.0.1:5560"
num_traj_samples: 8
max_generation_length: 128
timeout_s: 900
trajectory_sampling:
  _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  _convert_: 'all'
  time_horizon: 4
  interval_length: 0.5
EOF

cd $NAVSIM_ROOT
$NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent_8sample \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=500 \
    metric_cache_path=$CACHE \
    experiment_name=alpamayo_15_poc_parallel \
    worker=single_machine_thread_pool \
    worker.max_workers=8 2>&1 | tee -a $QLOG

for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null; done
sleep 5
CSV=$(ls -1t $NAVSIM_EXP_ROOT/alpamayo_15_poc_parallel/*/202*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && cp "$CSV" $LOGS/pdms_15_poc_eval_result.csv && log "RESULT: $(tail -1 $CSV)"
log "=== DONE ==="
