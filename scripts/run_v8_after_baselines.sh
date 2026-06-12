#!/bin/bash
LOG_BL=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/zeroshot_baselines.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v8_after_baselines.log
while true; do
    if grep -q "Zero-shot baselines (no FT) DONE\|baselines DONE" $LOG_BL 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] baselines DONE, launching v8" >> $TRIG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v8_noprune.sh > logs/ft_sweep_v8_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v8 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 60
done
