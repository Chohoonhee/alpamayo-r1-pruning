#!/bin/bash
TRIG_LOG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v6_trigger.log
# Wait until all alpamayo_server processes for v5_navsim_safe_lr1e5_1k are gone
while pgrep -f "v5_navsim_safe_lr1e5_1k" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] v5 recipe 2 eval finished; launching v6_full" >> $TRIG_LOG
sleep 30  # let GPU clear
cd /home/irteam/ws/alpamayo_pruning_share/scripts
nohup bash run_ft_sweep_v6_navsim_full.sh > logs/ft_sweep_v6_stdout.log 2>&1 &
echo "[$(date +%H:%M:%S)] sweep_v6 PID=$!" >> $TRIG_LOG
