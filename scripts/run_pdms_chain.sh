#!/bin/bash
# Run NAVSIM PDMS for 5 R1 conditions sequentially.
# Each uses GPU 0-3 + ports 5557-5560. ~12 min per condition. Total ~1h.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
QLOG=$LOGS/pdms_chain.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# (label, policy_meta or "")
declare -a TASKS=(
    "baseline_r1|"
    "greedy_k3_nusc|$LOGS/ksweep_r1_nusc_k3.meta.json"
    "early_stop_k4|$LOGS/greedyR1_navsim_earlystop_meta.json"
    "greedy_k7_nusc|$LOGS/ksweep_r1_nusc_k7.meta.json"
    "random_k3_seed0|$LOGS/random_baseline_r1_drop3_seed0.meta.json"
)

log "=== PDMS chain start (5 conditions × 500 scenes, ~1h total) ==="
for entry in "${TASKS[@]}"; do
    IFS='|' read -r LABEL POLICY <<<"$entry"
    if [ -f $LOGS/pdms_${LABEL}_result.csv ]; then
        log "[skip] pdms_${LABEL}_result.csv exists"
        continue
    fi
    log "--- launching $LABEL (policy=${POLICY:-baseline}) ---"
    bash $SCRIPTS/run_pdms_condition.sh "$LABEL" "$POLICY" 0 5557 500 2>&1 | tail -50
    log "--- $LABEL done ---"
    # Auto-commit + status
    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    bash $SCRIPTS/auto_commit.sh "PDMS $LABEL DONE" 2>&1 | tail -2 | tee -a $QLOG
done

log "=== PDMS chain COMPLETE ==="
bash $SCRIPTS/auto_commit.sh "PDMS chain complete (5 R1 conditions)" 2>&1 | tail -2 | tee -a $QLOG
