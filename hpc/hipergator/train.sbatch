#!/bin/bash
#SBATCH --job-name=artery-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96gb
#SBATCH --time=20:00:00
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=%x-%A_%a.log
# One DynUNet per array task (one per seed), on every labeled case. Submitted by run_all.sh.
# Resumable: if the job is stopped (time limit, preemption), sbatch it again with the same
# --array index and it continues from its last state.pt.
set -euo pipefail
source "$REPO/hpc/hipergator/config.sh"
activate_env
read -ra SEED_LIST <<< "$SEEDS"
SEED="${SEED_LIST[$SLURM_ARRAY_TASK_ID]}"
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
srun python -u "$REPO/scripts/train.py" --records "$WORK/cache/records.json" --arch dynunet --train all \
  --minutes "$TRAIN_MINUTES" --patch "$PATCH" --batch "$BATCH" --workers 10 --seed "$SEED" \
  --out "$WORK/runs/dynunet_s$SEED" || rc=$?
if [[ $rc -eq 99 ]]; then   # hit the time limit before the budget was used up: state saved, go again
  echo "requeueing to resume"
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit $rc
