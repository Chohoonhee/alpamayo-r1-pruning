#!/bin/bash
# 1.5 PDMS trial chain on sample500 (~4 conditions × ~5 min each = ~20 min).
# Quick test: does method beat random on 1.5 backbone?

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
QLOG=$LOGS/pdms_chain_15.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# 1.5 backbone — use ports 5557-5560 + GPU 0-3
declare -a TASKS=(
    "baseline_15|"
    "early_stop_k2|$LOGS/greedy15_navsim_earlystop_meta.json"
    "greedy_k2_nusc|$LOGS/ksweep_15_nusc_k2.meta.json"
    "random_k2_seed0|$LOGS/random_baseline_15_drop2_seed0.meta.json"
    "random_k2_seed1|$LOGS/random_baseline_15_drop2_seed1.meta.json"
    "random_k2_seed2|$LOGS/random_baseline_15_drop2_seed2.meta.json"
)

log "=== 1.5 PDMS sample500 chain start (6 conditions × ~5min) ==="
for entry in "${TASKS[@]}"; do
    IFS='|' read -r LABEL POLICY <<<"$entry"
    if [ -f $LOGS/pdms_${LABEL}_result.csv ]; then
        log "[skip] pdms_${LABEL}_result.csv exists"; continue
    fi
    log "--- launching $LABEL ($POLICY) ---"
    bash $SCRIPTS/run_pdms_condition.sh "$LABEL" "$POLICY" 0 5557 500 15 2>&1 | tail -30
    log "--- $LABEL done ---"
    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    bash $SCRIPTS/auto_commit.sh "1.5 PDMS sample500: $LABEL DONE" 2>&1 | tail -2 | tee -a $QLOG
done
log "=== 1.5 chain COMPLETE ==="
