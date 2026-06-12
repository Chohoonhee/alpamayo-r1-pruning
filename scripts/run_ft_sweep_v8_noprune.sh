#!/bin/bash
# Sweep v8: NO PRUNE + FT.
# Isolates the FT effect from the prune effect.
# Same recipes as v7 best (token lr=1e-2 and lr=1e-1, SAFE lr=1e-5)
# but with empty drop policy (K=0, no layers removed).
# Comparison axis: does FT itself help, hurt, or do nothing without prune.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
WEIGHTS_BASE=/home/irteam/ws/alpamayo_pruning/weights
POLICY=$LOGS/empty_meta.json   # NO PRUNE

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1

source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d

CACHE=$NAVSIM_EXP_ROOT/metric_cache
QLOG=$LOGS/ft_sweep_v8.log
CSV=$LOGS/ft_sweep_results.csv
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

train_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    local OUT=$WEIGHTS_BASE/sft_sweep_${RECIPE}
    if [ -d $OUT/final ] || [ -d $OUT/lora_final ]; then
        log "  [skip-train] $RECIPE exists"; return
    fi
    local TOTAL_SAMPLES=$((NTRAIN * NEPOCH))
    local TO=$((TOTAL_SAMPLES * 4 + 1800))
    log "  TRAIN $RECIPE: $TRAINER lr=$LR ntrain=$NTRAIN epoch=$NEPOCH (NO PRUNE) timeout=${TO}s"
    cd $SCRIPTS
    timeout $TO torchrun --nproc_per_node=8 --master_port=29512 $TRAINER \
        --drop_layers_json $POLICY \
        --train_samples $NTRAIN --epochs $NEPOCH \
        --lr $LR --batch_size 1 --grad_accum 8 \
        --out_dir $OUT --ddp \
        --log_every 20 --save_every 9999 2>&1 | tail -80 | tee -a $QLOG
    sleep 5
}

eval_recipe() {
    local RECIPE=$1
    local WTS=$WEIGHTS_BASE/sft_sweep_${RECIPE}/final
    [ ! -d $WTS ] && WTS=$WEIGHTS_BASE/sft_sweep_${RECIPE}/lora_final
    if [ ! -d $WTS ]; then log "  [skip-eval] no $WTS"; return; fi
    if grep -q "^${RECIPE}," $CSV; then log "  [skip-eval] $RECIPE already in CSV"; return; fi

    log "  EVAL $RECIPE (no prune)"
    SERVER_PIDS=()
    for i in 0 1 2 3; do
        GPU=$i; PORT=$((5557+i))
        SLOG=$LOGS/sweep_${RECIPE}_srv_${PORT}.log
        ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=$GPU nohup $A1_5_PYTHON \
            $SCRIPTS/alpamayo_server.py \
            --weights $WTS --port $PORT --drop_layers_json $POLICY > $SLOG 2>&1 &
        SERVER_PIDS+=($!)
    done
    waited=0
    while true; do
        ready=0
        for i in 0 1 2 3; do
            PORT=$((5557+i)); SLOG=$LOGS/sweep_${RECIPE}_srv_${PORT}.log
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

    EXPNAME=alpamayo_sweep_${RECIPE}
    cd $NAVSIM_ROOT
    timeout 2700 $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=alpamayo_agent_8sample \
        train_test_split=navtest \
        train_test_split.scene_filter.max_scenes=200 \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=8 > $LOGS/sweep_${RECIPE}_eval.log 2>&1

    for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
    sleep 5

    EVCSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    if [ -z "$EVCSV" ]; then log "  eval no CSV for $RECIPE"; return; fi
    TOKEN_OK="yes"
    grep -q "No <traj_future_start>" $LOGS/sweep_${RECIPE}_srv_5557.log 2>/dev/null && TOKEN_OK="no"
    PDMS=$(tail -1 "$EVCSV" | awk -F',' '{print $NF}')
    NCC=$(tail -1 "$EVCSV" | awk -F',' '{print $4}')
    DAC=$(tail -1 "$EVCSV" | awk -F',' '{print $5}')
    TTC=$(tail -1 "$EVCSV" | awk -F',' '{print $9}')
    EP=$(tail -1 "$EVCSV" | awk -F',' '{print $8}')
    C2=$(tail -1 "$EVCSV" | awk -F',' '{print $12}')
    echo "$RECIPE,navsim_noprune,?,navsim,navsim,navsim,$TOKEN_OK,$PDMS,$NCC,$DAC,$TTC,$EP,$C2" >> $CSV
    log "  $RECIPE: PDMS=$PDMS token_ok=$TOKEN_OK NCC=$NCC TTC=$TTC"
}

run_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    log "=== $RECIPE ==="
    train_recipe "$RECIPE" "$TRAINER" "$LR" "$NTRAIN" "$NEPOCH"
    eval_recipe "$RECIPE"
}

log "=== FT recipe sweep_v8 START (NO PRUNE + FT, NAVSIM data) ==="

# 1) Best v7 NAVSIM recipe: token lr=1e-1 — does FT help on un-pruned model?
run_recipe "v8_noprune_token_10k_lr1e1"  sft_stage2_token_only_navsim.py  1e-1  52000  1

# 2) Best v6 recipe: token lr=1e-2 full set — most informative comparison
run_recipe "v8_noprune_token_10k_lr1e2"  sft_stage2_token_only_navsim.py  1e-2  52000  1

# 3) SAFE lr=1e-5 — does action_proj FT help without prune?
run_recipe "v8_noprune_safe_10k_lr1e5"   sft_stage2_safe_navsim.py        1e-5  52000  1

# 4) Higher-stakes SAFE
run_recipe "v8_noprune_safe_10k_lr1e4"   sft_stage2_safe_navsim.py        1e-4  52000  1

log "=== FT recipe sweep_v8 DONE ==="
log "results: $CSV"
