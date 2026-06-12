#!/bin/bash
LOG_V10=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v10.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v11_trigger.log
while true; do
    if grep -q "v10 DONE" $LOG_V10 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] v10 DONE; launching v11 scaleup" >> $TRIG
        sleep 60
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v11_scaleup.sh > logs/ft_sweep_v11_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v11 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
