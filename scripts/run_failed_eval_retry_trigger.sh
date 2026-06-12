#!/bin/bash
LOG_V4=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v4.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/failed_eval_retry.log
while true; do
    if grep -q "sweep_v4 DONE" $LOG_V4 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep_v4 DONE; running failed eval retries" >> $TRIG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        bash retry_failed_evals.sh >> $TRIG 2>&1
        echo "[$(date +%H:%M:%S)] retries done" >> $TRIG
        exit 0
    fi
    sleep 60
done
