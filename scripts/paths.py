"""
Portable path configuration for the Alpamayo pruning repo.

All paths can be overridden via environment variables. Defaults match the
original `/home/irteam/ws/` layout on the project's dev server, so existing
runs there keep working without changes. On any other machine, set the env
vars in your shell (or a `.env` file) before launching scripts.

Env vars (path-style):
  ALPAMAYO_R1_SRC          dir of the `alpamayo_r1` package (clone of NVlabs/alpamayo)
  ALPAMAYO_15_SRC          dir of the `alpamayo1_5` package (1.5 source)
  ALPAMAYO_WEIGHTS_DIR     dir containing Alpamayo-R1-10B/, Alpamayo-1.5-10B/ and pruned variants
  NUSC_ROOT                nuScenes data root (parent of v1.0-trainval/, samples/, sweeps/)
  NAVSIM_WORKSPACE         NAVSIM workspace dir (contains navsim/, dataset/, exp/ …)
  OUTPUTS_DIR              where scripts write JSON/log output (default: this scripts/ dir)

String-style:
  NUSC_VERSION             default "v1.0-trainval"

Usage from a script:

    from paths import (
        ALPAMAYO_R1_SRC, ALPAMAYO_15_SRC,
        ALPAMAYO_R1_WEIGHTS, ALPAMAYO_15_WEIGHTS,
        NUSC_ROOT, NUSC_VERSION, OUTPUTS_DIR,
        add_alpamayo_to_syspath,
    )

    add_alpamayo_to_syspath(r1=True, v15=True)   # makes `import alpamayo_r1` / `alpamayo1_5` work

    ap.add_argument("--weights", default=str(ALPAMAYO_15_WEIGHTS))
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser()


# ---- Source repos -----------------------------------------------------------
ALPAMAYO_R1_SRC = _env_path(
    "ALPAMAYO_R1_SRC", "/home/irteam/ws/alpamayo_bench2drive/alpamayo"
)
ALPAMAYO_15_SRC = _env_path(
    "ALPAMAYO_15_SRC", "/home/irteam/ws/alpamayo_pruning/alpamayo1.5"
)

# ---- Weights ----------------------------------------------------------------
ALPAMAYO_WEIGHTS_DIR = _env_path(
    "ALPAMAYO_WEIGHTS_DIR", "/home/irteam/ws/alpamayo_pruning/weights"
)
ALPAMAYO_R1_WEIGHTS = ALPAMAYO_WEIGHTS_DIR / "Alpamayo-R1-10B"
ALPAMAYO_15_WEIGHTS = ALPAMAYO_WEIGHTS_DIR / "Alpamayo-1.5-10B"

# ---- Data -------------------------------------------------------------------
NUSC_ROOT    = _env_path("NUSC_ROOT", "/home/irteam/ws/nuscenes/raw_extracted")
NUSC_VERSION = os.environ.get("NUSC_VERSION", "v1.0-trainval")

# ---- Workspaces / outputs ---------------------------------------------------
NAVSIM_WORKSPACE = _env_path(
    "NAVSIM_WORKSPACE", "/home/irteam/ws/alpamayo_pruning/navsim_workspace"
)
OUTPUTS_DIR = _env_path("OUTPUTS_DIR", str(REPO_ROOT / "scripts"))


# ---- Helpers ----------------------------------------------------------------
def add_alpamayo_to_syspath(r1: bool = True, v15: bool = False) -> None:
    """Insert Alpamayo source paths at the front of sys.path.

    Also makes the scripts/ dir importable so peer modules (paths, helpers)
    resolve regardless of CWD.
    """
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    if r1:
        p = str(ALPAMAYO_R1_SRC / "src")
        if p not in sys.path:
            sys.path.insert(0, p)
    if v15:
        p = str(ALPAMAYO_15_SRC / "src")
        if p not in sys.path:
            sys.path.insert(0, p)


def pruned_weights(name: str) -> Path:
    """Path to a pruned variant under the weights dir.

    Example: pruned_weights("expertaware_vlm28") ->
             $ALPAMAYO_WEIGHTS_DIR/Alpamayo-1.5-10B-pruned-expertaware_vlm28
    """
    return ALPAMAYO_WEIGHTS_DIR / f"Alpamayo-1.5-10B-pruned-{name}"


def output_path(filename: str) -> Path:
    """Resolve an output file inside OUTPUTS_DIR."""
    return OUTPUTS_DIR / filename


if __name__ == "__main__":
    # Lightweight diagnostic
    print(f"REPO_ROOT             = {REPO_ROOT}")
    print(f"ALPAMAYO_R1_SRC       = {ALPAMAYO_R1_SRC}")
    print(f"ALPAMAYO_15_SRC       = {ALPAMAYO_15_SRC}")
    print(f"ALPAMAYO_WEIGHTS_DIR  = {ALPAMAYO_WEIGHTS_DIR}")
    print(f"  R1 weights          = {ALPAMAYO_R1_WEIGHTS}  exists={ALPAMAYO_R1_WEIGHTS.exists()}")
    print(f"  1.5 weights         = {ALPAMAYO_15_WEIGHTS}  exists={ALPAMAYO_15_WEIGHTS.exists()}")
    print(f"NUSC_ROOT             = {NUSC_ROOT}  exists={NUSC_ROOT.exists()}")
    print(f"NUSC_VERSION          = {NUSC_VERSION}")
    print(f"NAVSIM_WORKSPACE      = {NAVSIM_WORKSPACE}")
    print(f"OUTPUTS_DIR           = {OUTPUTS_DIR}")
