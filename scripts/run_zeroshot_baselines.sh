#!/bin/bash
# Zero-shot baselines: no FT, just pure 1.5 base evaluated on 200 navtest scenes.
# Two reference points:
#   1) baseline_15_noprune    — 1.5 base, no pruning, no FT (expected ~0.7286)
#   2) baseline_15_k2          — 1.5 base + K=2 [23,31] runtime prune, no FT (the true floor)

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
WEIGHTS_BASE=/home/irteam/ws/alpamayo_pruning/weights
POLICY_K2=$LOGS/greedy15_navsim_earlystop_meta.json
POLICY_EMPTY=$LOGS/empty_meta.json

# Empty policy for no-prune baseline
cat > $POLICY_EMPTY <<'EOF'
{"dropped_layers": [], "policy": "no_prune_baseline", "backbone": "15", "K": 0}
EOF

export HF_HUB_OFFLINE=1
export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1

CACHE=$NAVSIM_EXP_ROOT/metric_cache
QLOG=$LOGS/zeroshot_baselines.log
CSV=$LOGS/ft_sweep_results.csv
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

eval_baseline() {
    local RECIPE=$1 POLICY=$2 WTS=$3
    if grep -q "^${RECIPE}," $CSV; then log "  [skip] $RECIPE already in CSV"; return; fi

    log "=== $RECIPE ==="
    SERVER_PIDS=()
    for i in 0 1 2 3; do
        GPU=$i; PORT=$((5557+i))
        SLOG=$LOGS/baseline_${RECIPE}_srv_${PORT}.log
        ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=$GPU nohup $A1_5_PYTHON \
            $SCRIPTS/alpamayo_server.py \
            --weights $WTS --port $PORT --drop_layers_json $POLICY > $SLOG 2>&1 &
        SERVER_PIDS+=($!)
    done
    waited=0
    while true; do
        ready=0
        for i in 0 1 2 3; do
            PORT=$((5557+i)); SLOG=$LOGS/baseline_${RECIPE}_srv_${PORT}.log
            grep -q "\[server\] ready" $SLOG 2>/dev/null && ready=$((ready+1))
        done
        [ $ready -eq 4 ] && break
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 300 ]; then
            log "  server timeout for $RECIPE"
            for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
            return
        fi
    done

    cat > $NAVSIM_ROOT/navsim/planning/script/config/common/agent/alpamayo_agent_8sample.yaml <<EOF
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

    EXPNAME=alpamayo_baseline_${RECIPE}
    cd $NAVSIM_ROOT
    timeout 2700 $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=alpamayo_agent_8sample \
        train_test_split=navtest \
        train_test_split.scene_filter.max_scenes=200 \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=8 > $LOGS/baseline_${RECIPE}_eval.log 2>&1

    for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
    sleep 5

    EVCSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    if [ -z "$EVCSV" ]; then log "  no eval CSV"; return; fi
    TOKEN_OK="yes"
    grep -q "No <traj_future_start>" $LOGS/baseline_${RECIPE}_srv_5557.log 2>/dev/null && TOKEN_OK="no"
    PDMS=$(tail -1 "$EVCSV" | awk -F',' '{print $NF}')
    NCC=$(tail -1 "$EVCSV" | awk -F',' '{print $4}')
    DAC=$(tail -1 "$EVCSV" | awk -F',' '{print $5}')
    TTC=$(tail -1 "$EVCSV" | awk -F',' '{print $9}')
    EP=$(tail -1 "$EVCSV" | awk -F',' '{print $8}')
    C2=$(tail -1 "$EVCSV" | awk -F',' '{print $12}')
    echo "$RECIPE,baseline_noFT,0,none,none,none,$TOKEN_OK,$PDMS,$NCC,$DAC,$TTC,$EP,$C2" >> $CSV
    log "  $RECIPE: PDMS=$PDMS token_ok=$TOKEN_OK"
}

log "=== Zero-shot baselines (no FT) START ==="

# K=2 pruned base 1.5 — the true floor for all FT recipes
eval_baseline "baseline_15_k2"     $POLICY_K2     /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B

# No prune, no FT — the absolute reference
eval_baseline "baseline_15_noprune" $POLICY_EMPTY /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B

log "=== Zero-shot baselines DONE ==="
