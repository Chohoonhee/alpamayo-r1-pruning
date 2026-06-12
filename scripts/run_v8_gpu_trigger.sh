#!/bin/bash
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v8_trigger.log
LAUNCHED_BASELINES=0
LAUNCHED_V8=0
while true; do
    # Check GPU 0-7 memory usage. Wait until all < 8GB used (our own previous runs hold ~24GB).
    # Actually just check if there are non-our processes on any GPU
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | wc -l)
    if [ $busy -eq 0 ]; then
        if [ $LAUNCHED_BASELINES -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] GPUs free, launching baselines first" >> $TRIG
            cd /home/irteam/ws/alpamayo_pruning_share/scripts
            bash run_zeroshot_baselines.sh >> $TRIG 2>&1
            LAUNCHED_BASELINES=1
            echo "[$(date +%H:%M:%S)] baselines done" >> $TRIG
        fi
        if [ $LAUNCHED_V8 -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] launching sweep_v8 no-prune FT" >> $TRIG
            nohup bash run_ft_sweep_v8_noprune.sh > logs/ft_sweep_v8_stdout.log 2>&1 &
            echo "[$(date +%H:%M:%S)] v8 PID=$!" >> $TRIG
            LAUNCHED_V8=1
            exit 0
        fi
    fi
    sleep 120
done
