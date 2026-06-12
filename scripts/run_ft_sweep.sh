#!/bin/bash
# Systematic FT recipe sweep on 1.5 + K=2 [23,31].
# Goal: find recipe that doesn't break <traj_future_start> generation, ideally recovers PDMS.
# Logs everything to logs/ft_sweep_results.csv.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
A1_5_PYTHON=/home/irteam/ws/alpamayo_pruning/alpamayo1.5/a1_5_venv/bin/python
NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv
NAVSIM_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
WEIGHTS_BASE=/home/irteam/ws/alpamayo_pruning/weights
POLICY=$LOGS/greedy15_navsim_earlystop_meta.json

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT

# CRITICAL: activate alpamayo_b2d so torchrun resolves to env with einops/peft/transformers.
# Without this, system /usr/bin/python3 is used and all recipes fail with ModuleNotFoundError: einops.
source /home/irteam/miniconda/etc/profile.d/conda.sh
conda activate alpamayo_b2d
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1
CACHE=$NAVSIM_EXP_ROOT/metric_cache

QLOG=$LOGS/ft_sweep.log
CSV=$LOGS/ft_sweep_results.csv
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }
[ ! -f $CSV ] && echo "recipe,trainer,trainable_M,lr,n_train,n_epoch,token_ok,pdms,ncc,dac,ttc,ep,comfort_2f" > $CSV

# Helper: train one recipe
train_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    local OUT=$WEIGHTS_BASE/sft_sweep_${RECIPE}
    if [ -d $OUT/final ] || [ -d $OUT/lora_final ]; then
        log "  [skip-train] $RECIPE exists"; return
    fi
    # timeout scales with NTRAIN*NEPOCH: rough 90s/sample/epoch with overhead. Floor 1800s.
    local TOTAL_SAMPLES=$((NTRAIN * NEPOCH))
    local TO=$((TOTAL_SAMPLES * 3 + 1800))
    log "  TRAIN $RECIPE: $TRAINER lr=$LR ntrain=$NTRAIN epoch=$NEPOCH timeout=${TO}s"
    cd $SCRIPTS
    timeout $TO torchrun --nproc_per_node=8 --master_port=29505 $TRAINER \
        --drop_layers_json $POLICY \
        --train_samples $NTRAIN --epochs $NEPOCH \
        --lr $LR --batch_size 1 --grad_accum 8 \
        --out_dir $OUT --ddp \
        --log_every 5 --save_every 9999 2>&1 | tail -80 | tee -a $QLOG
    sleep 5
}

# Helper: eval one recipe with 4 server parallel, 200 scenes (faster)
eval_recipe() {
    local RECIPE=$1
    local WTS=$WEIGHTS_BASE/sft_sweep_${RECIPE}/final
    [ ! -d $WTS ] && WTS=$WEIGHTS_BASE/sft_sweep_${RECIPE}/lora_final
    if [ ! -d $WTS ]; then log "  [skip-eval] no $WTS"; return; fi
    if grep -q "^${RECIPE}," $CSV; then log "  [skip-eval] $RECIPE already in CSV"; return; fi

    log "  EVAL $RECIPE"
    # Launch 4 servers GPU 0-3, ports 5557-5560
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

    # Update agent yaml for 4 servers
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
    timeout 1500 $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
        agent=alpamayo_agent_8sample \
        train_test_split=navtest \
        train_test_split.scene_filter.max_scenes=200 \
        metric_cache_path=$CACHE \
        experiment_name=$EXPNAME \
        worker=single_machine_thread_pool \
        worker.max_workers=8 > $LOGS/sweep_${RECIPE}_eval.log 2>&1

    for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
    sleep 5

    # Parse result
    EVCSV=$(ls -1t $NAVSIM_EXP_ROOT/$EXPNAME/*/202*.csv 2>/dev/null | head -1)
    if [ -z "$EVCSV" ]; then log "  eval no CSV for $RECIPE"; return; fi

    # Check token preservation from server log
    TOKEN_OK="yes"
    grep -q "No <traj_future_start>" $LOGS/sweep_${RECIPE}_srv_5557.log 2>/dev/null && TOKEN_OK="no"

    # Extract metrics
    PDMS=$(tail -1 "$EVCSV" | awk -F',' '{print $NF}')
    NCC=$(tail -1 "$EVCSV" | awk -F',' '{print $4}')
    DAC=$(tail -1 "$EVCSV" | awk -F',' '{print $5}')
    TTC=$(tail -1 "$EVCSV" | awk -F',' '{print $9}')
    EP=$(tail -1 "$EVCSV" | awk -F',' '{print $8}')
    C2=$(tail -1 "$EVCSV" | awk -F',' '{print $12}')

    # Get trainable count from training log
    TRAIN_M=$(grep -E "trainable.*M.*total" $LOGS/ft_sweep.log 2>/dev/null | tail -1 | sed -E 's/.*trainable ([0-9.]+)M.*/\1/')

    echo "$RECIPE,$TRAINER,$TRAIN_M,$LR,$NTRAIN,$NEPOCH,$TOKEN_OK,$PDMS,$NCC,$DAC,$TTC,$EP,$C2" >> $CSV
    log "  $RECIPE: PDMS=$PDMS token_ok=$TOKEN_OK NCC=$NCC TTC=$TTC"
    bash $SCRIPTS/auto_commit.sh "sweep recipe $RECIPE done" 2>&1 | tail -1 | tee -a $QLOG
}

run_recipe() {
    local RECIPE=$1 TRAINER=$2 LR=$3 NTRAIN=$4 NEPOCH=$5
    log "=== $RECIPE ==="
    train_recipe "$RECIPE" "$TRAINER" "$LR" "$NTRAIN" "$NEPOCH"
    eval_recipe "$RECIPE"
}

log "=== FT recipe sweep START ==="
# SAFE FT @ 1k @ lr1e-4 already validated: 5-scene PDMS 0.7415 ≥ baseline 0.7286, token preservation OK.
# Sweep now maps out: lr sensitivity (SAFE), trainable scope (SAFE vs Expert-only vs vlmexp),
# scale (1k→5k→10k SAFE), and lower lr for vlmexp to find non-breaking threshold.

# Phase 1: SAFE lr scan @ 1k (anchor + extreme lr)
run_recipe "safe_lr1e4_1k"    sft_stage2_safe.py  1e-4  1000  1
run_recipe "safe_lr5e4_1k"    sft_stage2_safe.py  5e-4  1000  1
run_recipe "safe_lr1e5_1k"    sft_stage2_safe.py  1e-5  1000  1
run_recipe "safe_lr5e5_1k"    sft_stage2_safe.py  5e-5  1000  1
run_recipe "safe_lr1e3_1k"    sft_stage2_safe.py  1e-3  1000  1

# Phase 2: Expert-only LoRA lr scan @ 1k (matches prior Stage2 v2 attempt)
run_recipe "exp_lr5e5_1k"     sft_stage2_expert_only.py  5e-5  1000  1
run_recipe "exp_lr1e5_1k"     sft_stage2_expert_only.py  1e-5  1000  1
run_recipe "exp_lr5e6_1k"     sft_stage2_expert_only.py  5e-6  1000  1
run_recipe "exp_lr1e6_1k"     sft_stage2_expert_only.py  1e-6  1000  1

# Phase 3: VLM+Expert LoRA lr scan @ 1k (find threshold where breakage starts)
run_recipe "vlmexp_lr1e5_1k"  sft_stage2.py  1e-5  1000  1
run_recipe "vlmexp_lr1e6_1k"  sft_stage2.py  1e-6  1000  1
run_recipe "vlmexp_lr1e7_1k"  sft_stage2.py  1e-7  1000  1

# Phase 4: SAFE scale @ best lr (presumably 1e-4) — does more data help or hurt?
run_recipe "safe_lr1e4_2k"    sft_stage2_safe.py  1e-4  2000  1
run_recipe "safe_lr1e4_5k"    sft_stage2_safe.py  1e-4  5000  1
run_recipe "safe_lr1e4_10k"   sft_stage2_safe.py  1e-4  10000 1

# Phase 5: SAFE 2-epoch on small data — does revisiting help?
run_recipe "safe_lr1e4_1k_2ep"  sft_stage2_safe.py  1e-4  1000  2
run_recipe "safe_lr5e5_1k_2ep"  sft_stage2_safe.py  5e-5  1000  2

log "=== FT recipe sweep DONE ==="
log "results: $CSV"
cat $CSV | tee -a $QLOG
