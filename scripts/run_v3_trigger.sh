#!/bin/bash
LOG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v2.log
TRIG_LOG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v3_trigger.log

while true; do
    if grep -q "sweep_v2 DONE" $LOG 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep_v2 DONE detected; launching sweep_v3" >> $TRIG_LOG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v3.sh > logs/ft_sweep_v3_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] sweep_v3 PID=$!" >> $TRIG_LOG
        exit 0
    fi
    sleep 60
done
