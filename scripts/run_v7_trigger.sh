#!/bin/bash
LOG_V6=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v6.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v7_trigger.log
while true; do
    if grep -q "sweep_v6 DONE" $LOG_V6 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep_v6 DONE; launching v7" >> $TRIG
        sleep 30
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v7_navsim_explore.sh > logs/ft_sweep_v7_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v7 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
