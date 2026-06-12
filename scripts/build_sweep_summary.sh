#!/bin/bash
# Build final summary table from sweep CSV + server logs.
CSV=/home/irteam/ws/alpamayo_pruning_share/scripts/logs/ft_sweep_results.csv
LOGS=/home/irteam/ws/alpamayo_pruning_share/scripts/logs

echo "================================================================================"
echo "FT RECIPE SWEEP SUMMARY (generated $(date))"
echo "================================================================================"
echo
echo "Baseline (no FT, no prune): PDMS = 0.7286"
echo "Zero-shot K=2 [23,31]:      PDMS = 0.64 (approx, from prior eval)"
echo
echo "--- Results (sorted by PDMS desc) ---"
echo
{
  head -1 $CSV
  tail -n +2 $CSV | sort -t',' -k8 -gr
} | column -t -s','
echo
echo "--- Token-preservation details ---"
for srv_log in $LOGS/sweep_*_srv_5557.log; do
    recipe=$(basename $srv_log | sed -E 's/sweep_(.*)_srv_5557.log/\1/')
    fails=$(grep -c "No <traj_future_start>" $srv_log 2>/dev/null)
    ok=$(grep -c "inference ok" $srv_log 2>/dev/null)
    [ "$ok" -gt 0 ] && echo "  $recipe: $fails / $((fails+ok)) failures ($(awk "BEGIN{printf \"%.1f\", 100*$fails/($fails+$ok)}")%)"
done
echo
echo "--- Best recipe ---"
BEST=$(tail -n +2 $CSV | sort -t',' -k8 -gr | head -1)
echo "  $BEST"
echo
echo "--- Recipes still training/incomplete ---"
ls /home/irteam/ws/alpamayo_pruning/weights/sft_sweep_*/ 2>/dev/null | grep -E "^/" | while read d; do
    r=$(basename $(dirname $d))
    grep -q "^$r," $CSV || echo "  $r (no CSV row)"
done
echo
echo "================================================================================"
