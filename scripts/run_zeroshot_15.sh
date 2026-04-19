#!/bin/bash
set -u
N_SAMPLES=${1:-100}
OUT_DIR=/home/irteam/ws/alpamayo_pruning/scripts/zeroshot_logs
declare -A PORT=(
  [orig15]=5571
  [angular13_15]=5572
  [random13_15]=5573
  [angular28_15]=5574
  [random_safe28_15]=5575
)

for variant in orig15 angular13_15 random13_15 angular28_15 random_safe28_15; do
  port=${PORT[$variant]}
  echo "=== ${variant} (port ${port}, ${N_SAMPLES} samples) ==="
  conda run -n alpamayo_b2d python /home/irteam/ws/vipe_test/nuscenes_zero_shot.py \
    --n_samples $N_SAMPLES --port $port --camera_indices 2>&1 | tee ${OUT_DIR}/eval_${variant}.log
  if [ -f /home/irteam/ws/nuscenes/zero_shot_results.json ]; then
    mv /home/irteam/ws/nuscenes/zero_shot_results.json ${OUT_DIR}/results_${variant}.json
  fi
done

echo ""
echo "======= 1.5 SUMMARY ======="
for variant in orig15 angular13_15 random13_15 angular28_15 random_safe28_15; do
  echo "--- ${variant} ---"
  grep -E "Avg L2|Avg Collision" ${OUT_DIR}/eval_${variant}.log
done
