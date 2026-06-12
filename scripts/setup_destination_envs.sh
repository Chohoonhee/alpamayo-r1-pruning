#!/bin/bash
# Build the 3 Python envs needed on the destination machine.
# Run AFTER transfer is complete + before setup_destination.sh

set -uo pipefail
: "${DEST_BASE:?Need to set DEST_BASE}"
: "${ALPAMAYO_15_SRC:=$DEST_BASE/alpamayo1.5}"
: "${NAVSIM_WORKSPACE:=$DEST_BASE/navsim_workspace}"

echo "=== [1/3] alpamayo_b2d conda env (training + general) ==="
if ! conda env list | grep -q alpamayo_b2d; then
    conda create -n alpamayo_b2d python=3.12 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate alpamayo_b2d
    pip install --upgrade pip
    # Torch first (specific version that matches base server)
    pip install torch==2.8.0 numpy==1.26.4
    # Core deps
    pip install transformers peft einops accelerate safetensors pillow pyquaternion
    pip install nuscenes-devkit
    # flash_attn last, requires torch
    pip install flash-attn==2.8.3 --no-build-isolation
    echo "  ✓ alpamayo_b2d ready"
else
    echo "  (already exists)"
fi

echo
echo "=== [2/3] a1_5_venv (Alpamayo 1.5 inference server) ==="
if [ ! -f "$ALPAMAYO_15_SRC/a1_5_venv/bin/python" ]; then
    cd $ALPAMAYO_15_SRC
    # The a1_5_venv folder might have been transferred via rsync — check first.
    # If not, recreate with uv.
    if ! command -v uv &> /dev/null; then
        echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    uv venv a1_5_venv --python 3.10
    uv pip install -e . --python a1_5_venv/bin/python
    uv pip install flash-attn==2.8.3 --python a1_5_venv/bin/python --no-build-isolation
    echo "  ✓ a1_5_venv ready"
else
    echo "  (already exists)"
fi

echo
echo "=== [3/3] navsim_venv (NAVSIM eval, Python 3.9) ==="
if [ ! -f "$NAVSIM_WORKSPACE/navsim/navsim_venv/bin/python" ]; then
    cd $NAVSIM_WORKSPACE/navsim
    if ! command -v python3.9 &> /dev/null; then
        echo "Install python3.9 first (e.g. via pyenv or apt)"
        exit 1
    fi
    python3.9 -m venv navsim_venv
    source navsim_venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    echo "  ✓ navsim_venv ready"
else
    echo "  (already exists)"
fi

echo
echo "=== All 3 envs ready. Next: bash setup_destination.sh ==="
