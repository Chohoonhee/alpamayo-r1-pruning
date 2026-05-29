#!/bin/bash
# NAVSIM cross-domain eval queue.
#
# Waits for the main sequential queue to finish (looks for "ALL PHASES DONE"
# in sequential_queue.log), THEN runs NAVSIM PDMS eval on the
# alignment-grounded pruned models — the cross-domain robustness story.
#
# Per condition:
#   1. start alpamayo_server with the given policy_meta in background
#   2. wait for "[server] ready" line
#   3. run navsim_batch_mini.py against it (in navsim_venv, Python 3.9)
#   4. kill server, free GPU
#
# Conditions:
#   1.5 baseline | 1.5 plus_harmful (13 drop)
#   R1  baseline | R1  neutral (8 drop, the hero)
#   R1  plus_harmful (16 drop)

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
QLOG=$LOGS/navsim_queue.log

NAVSIM_VENV=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv

echo "[$(date +%H:%M:%S)] navsim queue: waiting for main queue to finish ..." | tee -a $QLOG

# Poll for main queue completion (or skip wait if already done)
while ! grep -q "ALL PHASES DONE" $LOGS/sequential_queue.log 2>/dev/null; do
    sleep 120
done
echo "[$(date +%H:%M:%S)] main queue done, starting NAVSIM" | tee -a $QLOG

# After main queue: pick a free GPU. Greedy may still hold one — use 0 first.
PICK_GPU() {
    for i in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$used" -lt 5000 ]; then
            echo $i; return
        fi
    done
    echo 0  # fall back to GPU 0 even if busy
}

run_navsim_condition() {
    local backbone=$1   # "1.5" or "r1"
    local label=$2
    local policy_meta=$3  # "" for baseline, path otherwise
    local gpu=$(PICK_GPU)
    local server_log=$LOGS/navsim_${backbone}_${label}_server.log
    local agent_log=$LOGS/navsim_${backbone}_${label}_agent.log
    local out_json=$LOGS/navsim_${backbone}_${label}_results.json

    echo "" | tee -a $QLOG
    echo "[$(date +%H:%M:%S)] === NAVSIM: $backbone/$label  gpu=$gpu  policy=${policy_meta:-none}" | tee -a $QLOG

    # 1. Start the server (in alpamayo_b2d env)
    conda activate alpamayo_b2d
    export HF_HUB_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=$gpu
    export ALPAMAYO_VARIANT=$backbone
    export ALPAMAYO_PORT=5557
    export ALPAMAYO_DEVICE=cuda:0
    cd $SCRIPTS

    local extra_args=""
    if [ -n "$policy_meta" ]; then
        extra_args="--drop_layers_json $policy_meta"
    fi
    nohup python alpamayo_server.py $extra_args > $server_log 2>&1 &
    local SERVER_PID=$!
    echo "[server-pid] $SERVER_PID" | tee -a $QLOG

    # 2. Wait for server ready (max 5 minutes)
    local waited=0
    while ! grep -q "\[server\] ready" $server_log 2>/dev/null; do
        sleep 5; waited=$((waited + 5))
        if [ $waited -gt 300 ]; then
            echo "[$(date +%H:%M:%S)] server startup timeout, killing" | tee -a $QLOG
            kill $SERVER_PID 2>/dev/null
            sleep 5
            return 1
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] server died during startup; see $server_log" | tee -a $QLOG
            return 1
        fi
    done
    echo "[$(date +%H:%M:%S)] server ready after ${waited}s" | tee -a $QLOG
    conda deactivate

    # 3. Run NAVSIM batch (in navsim_venv)
    source $NAVSIM_VENV/bin/activate
    python $SCRIPTS/navsim_batch_mini.py \
        -n 50 \
        --server tcp://127.0.0.1:5557 \
        --out $out_json > $agent_log 2>&1
    local AGENT_RC=$?
    deactivate
    echo "[$(date +%H:%M:%S)] navsim agent done rc=$AGENT_RC  out=$out_json" | tee -a $QLOG

    # 4. Kill server
    kill $SERVER_PID 2>/dev/null
    sleep 5

    # 5. Status + commit
    python $STATUS 2>&1 | tail -2 | tee -a $QLOG
    python $TRANSCRIPT 2>&1 | tail -2 | tee -a $QLOG
    bash $COMMIT "navsim: $backbone/$label" $out_json $server_log $agent_log 2>&1 | tail -3 | tee -a $QLOG
}

# Sequence of conditions
run_navsim_condition "1.5" "baseline"            ""
run_navsim_condition "1.5" "plus_harmful"        "$LOGS/policy15_plus_harmful.json"
run_navsim_condition "r1"  "baseline"            ""
run_navsim_condition "r1"  "neutral_8"           "$LOGS/policyR1_neutral.json"
run_navsim_condition "r1"  "plus_harmful_16"     "$LOGS/policyR1_plus_harmful.json"

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === NAVSIM QUEUE DONE ===" | tee -a $QLOG
bash $COMMIT "navsim queue complete — cross-domain PDMS battery" 2>&1 | tail -3 | tee -a $QLOG
