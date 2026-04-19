#!/bin/bash
set -u
N_SAMPLES=${1:-100}
OUT_DIR=/home/irteam/ws/alpamayo_pruning/scripts/zeroshot_logs
declare -A PORT=([orig]=5571 [angular13_rt]=5572 [random13_rt]=5573 [angular28_rt]=5574)

for variant in orig angular13_rt random13_rt angular28_rt; do
  port=${PORT[$variant]}
  echo "=== ${variant} (port ${port}, ${N_SAMPLES} samples) ==="
  conda run -n alpamayo_b2d python /home/irteam/ws/vipe_test/nuscenes_zero_shot.py \
    --n_samples $N_SAMPLES --port $port 2>&1 | tee ${OUT_DIR}/eval_${variant}.log
  if [ -f /home/irteam/ws/nuscenes/zero_shot_results.json ]; then
    mv /home/irteam/ws/nuscenes/zero_shot_results.json ${OUT_DIR}/results_${variant}.json
  fi
done

echo ""
echo "======= SUMMARY (runtime-pruned; true pruning) ======="
for variant in orig angular13_rt random13_rt angular28_rt; do
  echo "--- ${variant} ---"
  grep -E "Avg L2|Avg Collision" ${OUT_DIR}/eval_${variant}.log
done
