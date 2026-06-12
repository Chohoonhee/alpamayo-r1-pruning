#!/bin/bash
LOG_RETRY=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v2_retry_trigger.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/auto_decide.log
while true; do
    if grep -q "earlystop retry done" $LOG_RETRY 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] retry done; launching sweep_v4 token-only" >> $TRIG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v4_token.sh > logs/ft_sweep_v4_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] sweep_v4 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
