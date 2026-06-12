#!/bin/bash
# Watch for sweep_v3 completion, then decide next sweep based on results.
# Decision tree:
#   best PDMS >= 0.65 → stop (good enough)
#   best PDMS 0.55-0.65 → still try v4 token-only to confirm hypothesis
#   best PDMS < 0.55 → launch v4 token-only immediately
LOG_V3=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v3.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/auto_decide.log
CSV=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_results.csv

while true; do
    if grep -q "sweep_v3 DONE" $LOG_V3 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep_v3 DONE detected" >> $TRIG
        # Parse best PDMS from all wfv* rows
        BEST=$(grep -E "^wfv" $CSV 2>/dev/null | awk -F',' '{print $8}' | sort -gr | head -1)
        echo "[$(date +%H:%M:%S)] Best wfv* PDMS = $BEST" >> $TRIG
        # Always launch v4 token-only as next experiment regardless (informative)
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v4_token.sh > logs/ft_sweep_v4_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] sweep_v4 token-only launched PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
