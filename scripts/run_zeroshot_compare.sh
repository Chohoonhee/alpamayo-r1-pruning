#!/bin/bash
# Zero-shot nuScenes eval for 4 R1 variants: orig + angular/last/random pruned
set -u
N_SAMPLES=${1:-100}
OUT_DIR=/home/irteam/ws/alpamayo_pruning/scripts/zeroshot_logs
mkdir -p $OUT_DIR

declare -A PORT=([orig]=5571 [angular]=5572 [last]=5573 [random]=5574)

for variant in orig angular last random; do
  port=${PORT[$variant]}
  echo "=== ${variant} (port ${port}, ${N_SAMPLES} samples) ==="
  conda run -n alpamayo_b2d python /home/irteam/ws/vipe_test/nuscenes_zero_shot.py \
    --n_samples $N_SAMPLES --port $port 2>&1 | tee ${OUT_DIR}/eval_${variant}.log
  # Move result file before next variant overwrites it
  if [ -f /home/irteam/ws/nuscenes/zero_shot_results.json ]; then
    mv /home/irteam/ws/nuscenes/zero_shot_results.json ${OUT_DIR}/results_${variant}.json
  fi
done

echo ""
echo "==========================================================="
echo "  Summary:"
echo "==========================================================="
for variant in orig angular last random; do
  echo "--- ${variant} ---"
  grep -E "Avg L2|Avg Collision" ${OUT_DIR}/eval_${variant}.log || echo "(no result)"
done
