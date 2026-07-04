#!/usr/bin/env bash
# Set up a GPU-capable environment for the State Backdoor attack on
# LIBERO-Goal + OpenVLA-OFT. Run this once, on the machine/box that has
# the GPU (this repo's dev sandbox does NOT -- see state_backdoor/README.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENVLA_OFT_DIR="$REPO_ROOT/third_party/openvla-oft"
LIBERO_DIR="$REPO_ROOT/third_party/LIBERO"

echo "== Initializing submodules =="
git -C "$REPO_ROOT" submodule update --init --recursive

echo "== Creating conda env 'state-backdoor' (Python 3.10, per openvla-oft's SETUP.md) =="
conda create -n state-backdoor python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate state-backdoor

echo "== Installing PyTorch (match your CUDA version; see pytorch.org) =="
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "== Installing openvla-oft =="
pip install -e "$OPENVLA_OFT_DIR"
pip install "$OPENVLA_OFT_DIR"[train] 2>/dev/null || true
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

echo "== Installing LIBERO =="
pip install -e "$LIBERO_DIR"
pip install -r "$OPENVLA_OFT_DIR/experiments/robot/libero/libero_requirements.txt"

echo "== Installing state_backdoor's own (light) requirements =="
pip install -r "$REPO_ROOT/state_backdoor/requirements.txt"

echo "== Installing RLDS/TFDS conversion requirements =="
pip install tensorflow tensorflow_datasets tensorflow-graphics apache_beam

echo
echo "Setup complete. Next: state_backdoor/scripts/01_prepare_libero_goal_data.sh"
