#!/bin/bash
#SBATCH --job-name=artery-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Resample every case to 0.5 mm and write the training cache. Submitted by run_all.sh.
set -euo pipefail
source "$REPO/hpc/hipergator/config.sh"
activate_env
python "$REPO/scripts/prepare_data.py" --cases "$WORK/cases.json" --out "$WORK/cache" --workers 8
