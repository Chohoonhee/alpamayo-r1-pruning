#!/bin/bash
set -u
N_SAMPLES=${1:-100}
OUT_DIR=/home/irteam/ws/alpamayo_pruning/scripts/zeroshot_logs
declare -A PORT=([angular24]=5572 [angular28]=5573 [random24]=5574 [random28]=5575)

for variant in angular24 angular28 random24 random28; do
  port=${PORT[$variant]}
  echo "=== ${variant} (port ${port}, ${N_SAMPLES} samples) ==="
  conda run -n alpamayo_b2d python /home/irteam/ws/vipe_test/nuscenes_zero_shot.py \
    --n_samples $N_SAMPLES --port $port 2>&1 | tee ${OUT_DIR}/eval_${variant}.log
  if [ -f /home/irteam/ws/nuscenes/zero_shot_results.json ]; then
    mv /home/irteam/ws/nuscenes/zero_shot_results.json ${OUT_DIR}/results_${variant}.json
  fi
done

echo ""
echo "======= SUMMARY ======="
for variant in angular24 angular28 random24 random28; do
  echo "--- ${variant} ---"
  grep -E "Avg L2|Avg Collision" ${OUT_DIR}/eval_${variant}.log
done
