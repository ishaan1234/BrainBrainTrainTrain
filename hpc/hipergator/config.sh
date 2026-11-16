# Edit these for your HiPerGator account, then: source hpc/hipergator/config.sh
# (every script in this folder sources it automatically)

export GROUP="${GROUP:-gaitor_club}"                    # your HiPerGator group, see: id -gn  or  showAssoc $USER
export QOS="${QOS:-$GROUP}"                               # "$GROUP-b" for burst QOS
export WORK="${WORK:-/blue/$GROUP/$USER/arteryseg}"       # data, cache, runs, conda env live here (not /home)
export REPO="${REPO:-$WORK/BrainBrainTrainTrain}"
export ENV_PREFIX="${ENV_PREFIX:-$WORK/env}"

# GPU: check what your group can use with `sinfo -s` and the UFRC docs.
#   A100 80GB:  GPU_PARTITION=gpu       GPU_GRES=a100:1
#   B200:       GPU_PARTITION=hpg-b200  GPU_GRES=1   (much faster in fp32, if your group has access)
export GPU_PARTITION="${GPU_PARTITION:-hpg-b200}"
export GPU_GRES="${GPU_GRES:-b200:1}"
export CPU_PARTITION="${CPU_PARTITION:-hpg-default}"

# Training: DynUNet, same recipe as the best model (fp32, 160^3 patches, batch 2, augmentation).
export TRAIN_MINUTES="${TRAIN_MINUTES:-210}"    # training time per model (3.5 h); the LR schedule spans exactly this
export SEEDS="${SEEDS:-2026 7}"                 # one model per seed; predict.py averages them
export PATCH="${PATCH:-160}"
export BATCH="${BATCH:-2}"

# Load conda and activate the env (safe under `set -euo pipefail`).
activate_env() {
  set +u
  module load conda
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_PREFIX"
  set -u
}
