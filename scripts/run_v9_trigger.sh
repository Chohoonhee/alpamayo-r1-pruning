#!/bin/bash
LOG_V8=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v8.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v9_trigger.log
while true; do
    if grep -q "sweep_v8 DONE" $LOG_V8 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] v8 DONE; launching v9" >> $TRIG
        sleep 60
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v9_noprune_aggressive.sh > logs/ft_sweep_v9_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v9 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
