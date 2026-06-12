#!/bin/bash
# v10: pick best v8/v9 recipe from CSV, scale it to 3 epochs full set.
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
QLOG=$LOGS/ft_sweep_v10.log
CSV=$LOGS/ft_sweep_results.csv
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# Find best v8/v9 recipe
BEST=$(grep -E "^(v8|v9)_noprune" $CSV | sort -t',' -k8 -gr | head -1)
BEST_NAME=$(echo "$BEST" | cut -d',' -f1)
BEST_PDMS=$(echo "$BEST" | cut -d',' -f8)
log "best so far: $BEST_NAME PDMS=$BEST_PDMS"

# Pick trainer based on best recipe name
case "$BEST_NAME" in
  *token*) TRAINER=sft_stage2_token_only_navsim.py; LR=1e-1 ;;
  *safetoken*) TRAINER=sft_stage2_safe_plus_token_navsim.py; LR=1e-4 ;;
  *vlmexp*) TRAINER=sft_stage2_navsim.py; LR=1e-5 ;;
  *exp*) TRAINER=sft_stage2_expert_only_navsim.py; LR=5e-5 ;;
  *safe*) TRAINER=sft_stage2_safe_navsim.py; LR=1e-5 ;;
  *) TRAINER=sft_stage2_token_only_navsim.py; LR=1e-1 ;;
esac

log "scaling $BEST_NAME family with $TRAINER, lr=$LR, 3 epochs full set"

RECIPE=v10_noprune_best_52k_3ep
OUT=$WEIGHTS_BASE/sft_sweep_${RECIPE}
[ -d $OUT/final ] && { log "$RECIPE exists, skip train"; } || {
    cd $SCRIPTS
    timeout 86400 torchrun --nproc_per_node=8 --master_port=29514 $TRAINER \
        --drop_layers_json $POLICY \
        --train_samples 52000 --epochs 3 \
        --lr $LR --batch_size 1 --grad_accum 8 \
        --out_dir $OUT --ddp \
        --log_every 50 --save_every 9999 2>&1 | tail -100 | tee -a $QLOG
}

# Eval
WTS=$OUT/final
[ ! -d $WTS ] && WTS=$OUT/lora_final
[ -d $WTS ] || { log "no checkpoint"; exit 1; }

log "EVAL $RECIPE"
SERVER_PIDS=()
for i in 0 1 2 3; do
    GPU=$i; PORT=$((5557+i))
    SLOG=$LOGS/sweep_${RECIPE}_srv_${PORT}.log
    ALPAMAYO_VARIANT=1.5 CUDA_VISIBLE_DEVICES=$GPU nohup $A1_5_PYTHON \
        $SCRIPTS/alpamayo_server.py --weights $WTS --port $PORT --drop_layers_json $POLICY > $SLOG 2>&1 &
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
    [ $waited -gt 300 ] && { log "server timeout"; for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done; exit 1; }
done

cat > $NAVSIM_ROOT/navsim/planning/script/config/common/agent/alpamayo_agent_8sample.yaml <<YAML
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
YAML

cd $NAVSIM_ROOT
timeout 2700 $NAVSIM_VENV/bin/python navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent_8sample train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=200 metric_cache_path=$CACHE \
    experiment_name=alpamayo_sweep_${RECIPE} worker=single_machine_thread_pool worker.max_workers=8 \
    > $LOGS/sweep_${RECIPE}_eval.log 2>&1
for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null; done
sleep 5

EVCSV=$(ls -1t $NAVSIM_EXP_ROOT/alpamayo_sweep_${RECIPE}/*/202*.csv 2>/dev/null | head -1)
[ -z "$EVCSV" ] && { log "no eval CSV"; exit 1; }
TOKEN_OK="yes"; grep -q "No <traj_future_start>" $LOGS/sweep_${RECIPE}_srv_5557.log 2>/dev/null && TOKEN_OK="no"
PDMS=$(tail -1 "$EVCSV" | awk -F',' '{print $NF}')
NCC=$(tail -1 "$EVCSV" | awk -F',' '{print $4}')
TTC=$(tail -1 "$EVCSV" | awk -F',' '{print $9}')
echo "$RECIPE,scaled_best,?,$LR,52000,3,$TOKEN_OK,$PDMS,$NCC,?,$TTC,?,?" >> $CSV
log "$RECIPE: PDMS=$PDMS token_ok=$TOKEN_OK"
log "=== v10 DONE ==="
cd $SCRIPTS
bash build_sweep_summary.sh > $LOGS/FINAL_SUMMARY.txt 2>&1
