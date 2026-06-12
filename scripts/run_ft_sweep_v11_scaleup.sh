#!/bin/bash
# Sweep v11: scale-up around current best (SAFE+token combo lr=1e-4).
#   - lr neighborhood scan (5e-5, 2e-4, 5e-4) — does lower/higher lr help?
#   - data scale-up: stride=1 = 260k samples (5x more data)
#   - 5 epoch on best
# All no-prune NAVSIM full.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
WEIGHTS_BASE=/home/irteam/ws/alpamayo_pruning/weights
POLICY=$LOGS/empty_meta.json

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
QLOG=$LOGS/ft_sweep_v11.log
CSV=$LOGS/ft_sweep_results.csv
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

train_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    local OUT=$WEIGHTS_BASE/sft_sweep_${RECIPE}
    if [ -d $OUT/final ] || [ -d $OUT/lora_final ]; then
        log "  [skip-train] $RECIPE exists"; return
    fi
    local TOTAL_SAMPLES=$((NTRAIN * NEPOCH))
    local TO=$((TOTAL_SAMPLES * 4 + 3600))
    log "  TRAIN $RECIPE: $TRAINER lr=$LR ntrain=$NTRAIN epoch=$NEPOCH (NO PRUNE) timeout=${TO}s"
    cd $SCRIPTS
    timeout $TO torchrun --nproc_per_node=8 --master_port=29515 $TRAINER \
        --drop_layers_json $POLICY \
        --train_samples $NTRAIN --epochs $NEPOCH \
        --lr $LR --batch_size 1 --grad_accum 8 \
        --out_dir $OUT --ddp \
        --log_every 50 --save_every 9999 2>&1 | tail -80 | tee -a $QLOG
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
    echo "$RECIPE,scaleup,?,navsim,navsim,navsim,$TOKEN_OK,$PDMS,$NCC,$DAC,$TTC,$EP,$C2" >> $CSV
    log "  $RECIPE: PDMS=$PDMS token_ok=$TOKEN_OK NCC=$NCC TTC=$TTC"
}

run_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    log "=== $RECIPE ==="
    train_recipe "$RECIPE" "$TRAINER" "$LR" "$NTRAIN" "$NEPOCH"
    eval_recipe "$RECIPE"
}

log "=== FT recipe sweep_v11 START (scaleup around SAFE+token lr=1e-4 best) ==="

# Lr neighborhood scan around best (52k same)
run_recipe "v11_safetoken_52k_lr5e5"   sft_stage2_safe_plus_token_navsim.py  5e-5   52000  1
run_recipe "v11_safetoken_52k_lr2e4"   sft_stage2_safe_plus_token_navsim.py  2e-4   52000  1
run_recipe "v11_safetoken_52k_lr5e4"   sft_stage2_safe_plus_token_navsim.py  5e-4   52000  1

# Data scale: 260k samples (stride=1 = 5x data), same lr
run_recipe "v11_safetoken_260k_lr1e4"  sft_stage2_safe_plus_token_navsim_stride1.py  1e-4  260000  1

# 5 epoch on best
run_recipe "v11_safetoken_52k_lr1e4_5ep"  sft_stage2_safe_plus_token_navsim.py  1e-4  52000  5

log "=== FT recipe sweep_v11 DONE ==="
log "results: $CSV"

# Final summary
cd $SCRIPTS
bash build_sweep_summary.sh > $LOGS/FINAL_SUMMARY.txt 2>&1
log "Final summary at $LOGS/FINAL_SUMMARY.txt"
