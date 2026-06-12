#!/bin/bash
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v11_resume.log
echo "[$(date +%H:%M:%S)] watcher started, waiting for GPUs 1-7 to free" >> $TRIG
while true; do
    # Check if any non-our process holds significant memory on GPUs 1-7
    busy=0
    for i in 1 2 3 4 5 6 7; do
        used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used" -gt 30000 ]; then
            busy=1
            break
        fi
    done
    if [ $busy -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] GPUs 1-7 free (<30GB each), resuming v11" >> $TRIG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        nohup bash run_ft_sweep_v11_scaleup.sh > logs/ft_sweep_v11_stdout.log 2>&1 &
        echo "[$(date +%H:%M:%S)] v11 PID=$!" >> $TRIG
        exit 0
    fi
    sleep 120
done
