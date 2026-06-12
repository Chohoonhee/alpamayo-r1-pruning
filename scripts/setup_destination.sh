#!/bin/bash
# Run on the DESTINATION machine after transfer is complete.
# Verifies paths + can rebuild Python envs.
#
# Required env vars (set BEFORE running this):
#   DEST_BASE                  base path holding alpamayo_pruning_share/, weights/, navsim_workspace/, alpamayo1.5/
#   ALPAMAYO_15_SRC            = $DEST_BASE/alpamayo1.5
#   ALPAMAYO_WEIGHTS_DIR       = $DEST_BASE/weights
#   NAVSIM_WORKSPACE           = $DEST_BASE/navsim_workspace

set -uo pipefail

: "${DEST_BASE:?Need to set DEST_BASE}"
: "${ALPAMAYO_15_SRC:=$DEST_BASE/alpamayo1.5}"
: "${ALPAMAYO_WEIGHTS_DIR:=$DEST_BASE/weights}"
: "${NAVSIM_WORKSPACE:=$DEST_BASE/navsim_workspace}"

export ALPAMAYO_15_SRC ALPAMAYO_WEIGHTS_DIR NAVSIM_WORKSPACE
export NAVSIM_DEVKIT_ROOT=$NAVSIM_WORKSPACE/navsim
export OPENSCENE_DATA_ROOT=$NAVSIM_WORKSPACE/dataset
export NAVSIM_EXP_ROOT=$NAVSIM_WORKSPACE/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HF_HUB_OFFLINE=1

echo "=== verifying transferred files ==="
ALL_OK=1
check() {
    if [ -e "$1" ]; then
        echo "  ✓ $1"
    else
        echo "  ✗ MISSING: $1"
        ALL_OK=0
    fi
}

check "$ALPAMAYO_WEIGHTS_DIR/Alpamayo-1.5-10B/config.json"
check "$ALPAMAYO_WEIGHTS_DIR/sft_sweep_v9_noprune_safetoken_52k_lr1e4/final/config.json"
check "$NAVSIM_WORKSPACE/dataset/navsim_logs/trainval"
check "$NAVSIM_WORKSPACE/dataset/navsim_logs/test"
check "$NAVSIM_WORKSPACE/dataset/sensor_blobs/trainval"
check "$NAVSIM_WORKSPACE/dataset/sensor_blobs/test"
check "$NAVSIM_WORKSPACE/dataset/maps"
check "$NAVSIM_WORKSPACE/exp/metric_cache"
check "$NAVSIM_WORKSPACE/navsim/navsim/agents/alpamayo_agent.py"
check "$ALPAMAYO_15_SRC/src/alpamayo1_5/models/alpamayo1_5.py"
check "$DEST_BASE/alpamayo_pruning_share/scripts/run_ft_sweep_v11_scaleup.sh"
check "$DEST_BASE/alpamayo_pruning_share/scripts/navsim_trainval_index.json"

if [ $ALL_OK -eq 0 ]; then
    echo
    echo "✗ Some files missing. Re-run rsync for missing components."
    exit 1
fi
echo
echo "✓ All required files present."
echo

echo "=== Python envs check ==="
if command -v conda &> /dev/null; then
    if conda env list | grep -q alpamayo_b2d; then
        echo "  ✓ conda env 'alpamayo_b2d' exists"
    else
        echo "  ✗ conda env 'alpamayo_b2d' not found"
        echo "    Run: bash $DEST_BASE/alpamayo_pruning_share/scripts/setup_destination_envs.sh"
        ALL_OK=0
    fi
else
    echo "  ✗ conda not installed"
fi

if [ -f "$ALPAMAYO_15_SRC/a1_5_venv/bin/python" ]; then
    echo "  ✓ a1_5_venv exists"
else
    echo "  ✗ a1_5_venv not found at $ALPAMAYO_15_SRC/a1_5_venv"
    ALL_OK=0
fi

if [ -f "$NAVSIM_WORKSPACE/navsim/navsim_venv/bin/python" ]; then
    echo "  ✓ navsim_venv exists"
else
    echo "  ✗ navsim_venv not found"
    ALL_OK=0
fi

[ $ALL_OK -eq 0 ] && { echo; echo "Run setup_destination_envs.sh first."; exit 1; }

echo
echo "=== Functional test ==="
cd $DEST_BASE/alpamayo_pruning_share/scripts
source $(conda info --base)/etc/profile.d/conda.sh && conda activate alpamayo_b2d
python paths.py 2>&1 | head -20 || true
python -c "from navsim_sft_dataset import NavsimSFTDataset; ds = NavsimSFTDataset(n_samples=5); print('NAVSIM dataset loads OK:', len(ds), 'samples')"

echo
echo "=== Ready to resume training ==="
echo "  cd $DEST_BASE/alpamayo_pruning_share/scripts"
echo "  nohup bash run_ft_sweep_v11_scaleup.sh > logs/v11_stdout.log 2>&1 &"
echo
echo "  Current best: v9_noprune_safetoken_52k_lr1e4_2ep = PDMS 0.8214"
echo "  v11 will run: lr neighborhood (5e-5, 2e-4, 5e-4) + 260k stride1 + 5 epoch"
echo "  v11 auto-skips already-completed recipes (recipe 1 done if you transferred its weights)"
