#!/bin/bash
# Watch for sweep_v3 done; first run sweep_v2 retry (earlystop), then auto_decide handles v4.
LOG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_v3.log
TRIG=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/v2_retry_trigger.log
while true; do
    if grep -q "sweep_v3 DONE" $LOG 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] sweep_v3 DONE; running earlystop retry" >> $TRIG
        cd /home/irteam/ws/alpamayo_pruning_share/scripts
        bash run_ft_sweep_v2_retry_earlystop.sh >> $TRIG 2>&1
        echo "[$(date +%H:%M:%S)] earlystop retry done" >> $TRIG
        exit 0
    fi
    sleep 60
done
