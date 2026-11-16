#!/bin/bash
#SBATCH --job-name=artery-predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Segment one scan on a GPU:
#   sbatch --partition=$GPU_PARTITION --gres=gpu:$GPU_GRES --account=$GROUP --qos=$QOS \
#          hpc/hipergator/predict.sbatch /path/to/dicom_folder /path/to/output
set -euo pipefail
source "$REPO/hpc/hipergator/config.sh"
activate_env
MODELS="$WORK/models"
[[ -d "$MODELS" ]] || MODELS="$REPO/models"
python "$REPO/scripts/predict.py" "${1:?input DICOM folder / zip / NIfTI}" --models "$MODELS" --out "${2:?output folder}"
