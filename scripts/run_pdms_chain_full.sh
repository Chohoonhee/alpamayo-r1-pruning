#!/bin/bash
# Full NAVSIM PDMS chain: 9 R1 conditions × full 12146 navtest scenes, 8-way shard.
# ~35 min per condition × 9 = ~5h sequential.
# Skips conditions where result already exists.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
QLOG=$LOGS/pdmsfull_chain.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

# (label, policy_meta or "")
declare -a TASKS=(
    "baseline|"
    "greedy_k1_nusc|$LOGS/ksweep_r1_nusc_k1.meta.json"
    "greedy_k2_nusc|$LOGS/ksweep_r1_nusc_k2.meta.json"
    "greedy_k3_nusc|$LOGS/ksweep_r1_nusc_k3.meta.json"
    "early_stop_k4|$LOGS/greedyR1_navsim_earlystop_meta.json"
    "greedy_k7_nusc|$LOGS/ksweep_r1_nusc_k7.meta.json"
    "random_k3_seed0|$LOGS/random_baseline_r1_drop3_seed0.meta.json"
    "random_k3_seed1|$LOGS/random_baseline_r1_drop3_seed1.meta.json"
    "random_k3_seed2|$LOGS/random_baseline_r1_drop3_seed2.meta.json"
)

log "=== FULL PDMS CHAIN START (9 conditions × 12146 scenes, ~5h) ==="

for entry in "${TASKS[@]}"; do
    IFS='|' read -r LABEL POLICY <<<"$entry"
    if [ -f $LOGS/pdmsfull_${LABEL}_result.csv ]; then
        log "[skip] pdmsfull_${LABEL}_result.csv exists"
        continue
    fi
    # Special: re-use smoke_full_baseline as baseline if present
    if [ "$LABEL" = "baseline" ] && [ -f $LOGS/pdmsfull_smoke_full_baseline_result.csv ]; then
        log "[reuse] smoke_full_baseline → baseline"
        cp $LOGS/pdmsfull_smoke_full_baseline_result.csv $LOGS/pdmsfull_baseline_result.csv
        for s in 0 1 2 3 4 5 6 7; do
            [ -f $LOGS/pdmsfull_smoke_full_baseline_shard${s}.csv ] && \
                cp $LOGS/pdmsfull_smoke_full_baseline_shard${s}.csv $LOGS/pdmsfull_baseline_shard${s}.csv
        done
        continue
    fi
    log "--- launching $LABEL (policy=${POLICY:-baseline}) ---"
    bash $SCRIPTS/run_pdms_condition_sharded.sh "$LABEL" "$POLICY" 8 2>&1 | tail -30
    log "--- $LABEL done ---"
    python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
    bash $SCRIPTS/auto_commit.sh "PDMS full-navtest $LABEL DONE" 2>&1 | tail -2 | tee -a $QLOG
done

log "=== FULL PDMS CHAIN COMPLETE ==="
bash $SCRIPTS/auto_commit.sh "Full-navtest PDMS chain complete (9 conditions × 12146 scenes)" 2>&1 | tail -2 | tee -a $QLOG
