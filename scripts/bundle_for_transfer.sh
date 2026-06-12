#!/bin/bash
# Creates a transfer manifest of all files needed to resume on another machine.
# Doesn't bundle the huge sensor_blobs (do those via rsync separately).
#
# Usage:
#   ./bundle_for_transfer.sh DEST_HOST DEST_BASE_PATH
# Example:
#   ./bundle_for_transfer.sh user@newmachine /data/alpamayo

set -uo pipefail

DEST_HOST=${1:-}
DEST_BASE=${2:-}

if [ -z "$DEST_HOST" ] || [ -z "$DEST_BASE" ]; then
    echo "Usage: $0 DEST_HOST DEST_BASE_PATH"
    echo "Example: $0 user@newmachine /data/alpamayo"
    echo ""
    echo "Or set DRY_RUN=1 to just print the rsync commands:"
    echo "  DRY_RUN=1 $0 user@newmachine /data/alpamayo"
    exit 1
fi

DRY=${DRY_RUN:-0}
RSYNC="rsync -av --progress --partial"
[ "$DRY" = "1" ] && RSYNC="echo $RSYNC"

# Source paths on this machine
SRC_SHARE=/home/irteam/ws/alpamayo_pruning_share
SRC_BASE=/home/irteam/ws/alpamayo_pruning

echo "=== Transfer to $DEST_HOST:$DEST_BASE ==="
echo "(dry run = $DRY)"
echo

# 1. Code repo (small but critical)
echo "[1/5] Code repo + results CSV ..."
$RSYNC $SRC_SHARE/ ${DEST_HOST}:${DEST_BASE}/alpamayo_pruning_share/

# 2. Base model + best checkpoints
echo "[2/5] Base model + 2 best checkpoints ..."
$RSYNC \
    $SRC_BASE/weights/Alpamayo-1.5-10B \
    $SRC_BASE/weights/sft_sweep_v9_noprune_safetoken_52k_lr1e4 \
    $SRC_BASE/weights/sft_sweep_v9_noprune_safetoken_52k_lr1e4_2ep \
    ${DEST_HOST}:${DEST_BASE}/weights/

# 3. NAVSIM dataset (LARGEST)
echo "[3/5] NAVSIM dataset (~580GB) ..."
$RSYNC \
    $SRC_BASE/navsim_workspace/dataset/ \
    ${DEST_HOST}:${DEST_BASE}/navsim_workspace/dataset/

# 4. NAVSIM devkit + exp dir (metric cache)
echo "[4/5] NAVSIM devkit + exp ..."
$RSYNC \
    $SRC_BASE/navsim_workspace/navsim/ \
    ${DEST_HOST}:${DEST_BASE}/navsim_workspace/navsim/
$RSYNC \
    $SRC_BASE/navsim_workspace/exp/metric_cache/ \
    ${DEST_HOST}:${DEST_BASE}/navsim_workspace/exp/metric_cache/

# 5. Alpamayo 1.5 source + a1_5_venv
echo "[5/5] Alpamayo 1.5 source ..."
$RSYNC \
    $SRC_BASE/alpamayo1.5/ \
    ${DEST_HOST}:${DEST_BASE}/alpamayo1.5/

echo
echo "=== Done. On destination, run: ==="
echo "  cd ${DEST_BASE}/alpamayo_pruning_share/scripts"
echo "  source ../env.sh   # set env vars (create from TRANSFER.md template)"
echo "  bash setup_destination.sh"
