#!/bin/bash
# Run 2 PDMS conditions in PARALLEL on full navtest (no max_scenes).
# Condition A uses GPUs 0-3, ports 5557-5560.
# Condition B uses GPUs 4-7, ports 5561-5564.
# Each ~2h; both finish in ~2h wallclock.
#
# Args:
#   $1 = labelA, $2 = policyA (or "")
#   $3 = labelB, $4 = policyB (or "")
#   $5 = VARIANT (r1 or 15)

set -uo pipefail
LABEL_A=$1; POLICY_A=${2:-}
LABEL_B=$3; POLICY_B=${4:-}
VARIANT=${5:-r1}

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs

QLOG=$LOGS/pdmsfull2_${LABEL_A}_${LABEL_B}.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }
log "=== FULL PDMS pair: A=$LABEL_A  B=$LABEL_B  variant=$VARIANT ==="

# Modified condition runner: takes GPU range + port + label + policy
run_one() {
    local LBL=$1 POL=$2 GPU0=$3 PORT0=$4
    if [ -f $LOGS/pdmsfull_${LBL}_result.csv ]; then
        log "[skip] $LBL exists"; return
    fi
    bash $SCRIPTS/run_pdms_condition.sh "fullnav_${LBL}" "$POL" $GPU0 $PORT0 99999 $VARIANT 2>&1 | tail -25
}

run_one "$LABEL_A" "$POLICY_A" 0 5557 &
PID_A=$!
sleep 30  # stagger so server logs don't collide
run_one "$LABEL_B" "$POLICY_B" 4 5561 &
PID_B=$!
log "launched A pid=$PID_A  B pid=$PID_B"

wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
log "A rc=$RC_A  B rc=$RC_B"
log "=== FULL PDMS pair DONE ==="
