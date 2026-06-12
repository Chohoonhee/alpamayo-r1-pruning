#!/bin/bash
# R1 baseline sample500 1-sample PDMS in a1_5_venv (same env as 1.5 restoration).
# Compare to alpamayo_b2d R1 (April: 0.4901, Now: 0.4795).

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

QLOG=$LOGS/pdms_r1_a1_5_venv.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }
log "=== R1 baseline sample500 1-sample in a1_5_venv ==="
log "April R1 1-sample: 0.4901, Now (alpamayo_b2d): 0.4795"

SLOG=$LOGS/pdms_r1_a1_5_srv.log
ALPAMAYO_VARIANT=r1 CUDA_VISIBLE_DEVICES=0 nohup $A1_5_PYTHON $SCRIPTS/alpamayo_server.py --port 5557 > $SLOG 2>&1 &
SP=$!
log "server PID=$SP"

waited=0
while ! grep -q "\[server\] ready" $SLOG 2>/dev/null; do
    sleep 5; waited=$((waited+5))
    if [ $waited -gt 600 ]; then log "timeout"; kill $SP 2>/dev/null; exit 1; fi
    if ! kill -0 $SP 2>/dev/null; then log "server died"; tail -20 $SLOG | tee -a $QLOG; exit 1; fi
done
log "server ready in ${waited}s"

# Use single-server 1-sample agent
cat > $NAVSIM_ROOT/navsim/planning/script/config/common/agent/alpamayo_agent.yaml <<'EOF'
_target_: navsim.agents.alpamayo_agent.AlpamayoNAVSIMAgent
_convert_: 'all'
server_addr: "tcp://127.0.0.1:5557"
num_traj_samples: 1
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
    agent=alpamayo_agent \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=500 \
    metric_cache_path=$CACHE \
    experiment_name=alpamayo_r1_a1_5_venv \
    worker=single_machine_thread_pool \
    worker.max_workers=2 2>&1 | tee -a $QLOG

kill $SP 2>/dev/null
sleep 5
CSV=$(ls -1t $NAVSIM_EXP_ROOT/alpamayo_r1_a1_5_venv/*/202*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && cp "$CSV" $LOGS/pdms_r1_a1_5_venv_result.csv && log "Result: $(tail -1 $CSV)"
log "=== DONE ==="
