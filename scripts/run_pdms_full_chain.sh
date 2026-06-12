#!/bin/bash
# R1 full navtest PDMS chain via 4-server pattern (proven from sample500).
# 3 sequential pairs of conditions (2 parallel per pair, 4 GPUs each).
# Total ~6h.

set -uo pipefail
REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
QLOG=$LOGS/pdmsfullnav_chain.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $QLOG; }

log "=== R1 FULL NAVTEST PDMS CHAIN START (3 pairs × ~2h) ==="

# Pair 1: paper headline (baseline vs greedy k=3)
bash $SCRIPTS/run_pdms_full_2cond.sh \
    "baseline" "" \
    "greedy_k3_nusc" "$LOGS/ksweep_r1_nusc_k3.meta.json" \
    r1 2>&1 | tail -30
python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
bash $SCRIPTS/auto_commit.sh "PDMS full navtest: baseline + greedy_k3 DONE" 2>&1 | tail -2 | tee -a $QLOG

# Pair 2: random variance
bash $SCRIPTS/run_pdms_full_2cond.sh \
    "random_k3_seed0" "$LOGS/random_baseline_r1_drop3_seed0.meta.json" \
    "random_k3_seed1" "$LOGS/random_baseline_r1_drop3_seed1.meta.json" \
    r1 2>&1 | tail -30
python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
bash $SCRIPTS/auto_commit.sh "PDMS full navtest: random seed0/1 DONE" 2>&1 | tail -2 | tee -a $QLOG

# Pair 3: 3rd random + early-stop k=4
bash $SCRIPTS/run_pdms_full_2cond.sh \
    "random_k3_seed2" "$LOGS/random_baseline_r1_drop3_seed2.meta.json" \
    "early_stop_k4" "$LOGS/greedyR1_navsim_earlystop_meta.json" \
    r1 2>&1 | tail -30
python $SCRIPTS/update_status.py 2>&1 | tail -1 | tee -a $QLOG
bash $SCRIPTS/auto_commit.sh "PDMS full navtest: random seed2 + early_stop_k4 DONE" 2>&1 | tail -2 | tee -a $QLOG

log "=== R1 FULL NAVTEST PDMS CHAIN COMPLETE ==="
bash $SCRIPTS/auto_commit.sh "R1 full navtest PDMS chain complete (6 conditions × 12146 scenes)" 2>&1 | tail -2 | tee -a $QLOG
