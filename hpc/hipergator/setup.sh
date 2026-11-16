#!/bin/bash
# One-time setup, run on a HiPerGator login node:
#   bash hpc/hipergator/setup.sh
# Creates a conda env in /blue with PyTorch (CUDA 12.8 build: works on A100 and B200) and this package.
set -euo pipefail
source "$(dirname "$0")/config.sh"
[[ "$GROUP" == CHANGE_ME ]] && { echo "Set GROUP in hpc/hipergator/config.sh first"; exit 1; }

mkdir -p "$WORK"
set +u; module load conda; eval "$(conda shell.bash hook)"; set -u
[[ -d "$ENV_PREFIX" ]] || conda create -y -p "$ENV_PREFIX" python=3.11
activate_env
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e "$REPO[dev]" gdown
python -c "import torch, monai; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| monai', monai.__version__)"
echo "Setup done. Env: $ENV_PREFIX"
