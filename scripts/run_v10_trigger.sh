#!/bin/bash
LOG_V9=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v9.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v10_trigger.log
while true; do
    if grep -q "sweep_v9 DONE" $LOG_V9 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] v9 DONE; launching v10 (scale best 3 epoch)" >> $TRIG
        sleep 60
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_v10_scale_best.sh > logs/ft_sweep_v10_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v10 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
