#!/usr/bin/env bash
# One-command setup for Linux / macOS / WSL.
#   bash install.sh            # GPU build (NVIDIA CUDA) if you have an NVIDIA card
#   bash install.sh --cpu      # force the CPU-only build
#
# Creates a local .venv, installs PyTorch + the few dependencies, and runs a
# quick environment check. Re-run any time; it is safe.
set -e
cd "$(dirname "$0")"

CUDA_URL="https://download.pytorch.org/whl/cu124"   # RTX 40-series (Ada) friendly
MODE="gpu"
[ "$1" = "--cpu" ] && MODE="cpu"

echo "=============================================="
echo " Embodied Agent - installer ($MODE build)"
echo "=============================================="

PY="python3"; command -v python3 >/dev/null 2>&1 || PY="python"
echo "[1/4] Creating virtual environment (.venv) ..."
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

echo "[2/4] Installing PyTorch ..."
if [ "$MODE" = "gpu" ]; then
  echo "      (CUDA build from $CUDA_URL - large download, be patient)"
  pip install torch --index-url "$CUDA_URL" || {
    echo "      GPU build failed; falling back to the CPU build."
    pip install torch
  }
else
  pip install torch
fi

echo "[3/4] Installing the rest (numpy, matplotlib, pyyaml, imageio, pytest) ..."
pip install -r requirements.txt

echo "[4/4] Checking your environment ..."
python -m embodied_agent.doctor || true

echo
echo "Done. To watch the agent learn, run:"
echo "    bash run.sh          (opens the dashboard in your browser)"
