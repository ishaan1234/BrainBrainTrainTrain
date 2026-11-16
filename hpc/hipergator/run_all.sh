#!/bin/bash
# Submit the whole chain: preprocess -> train (one GPU job per seed, in parallel) -> export.
#   bash hpc/hipergator/run_all.sh            # everything
#   bash hpc/hipergator/run_all.sh --skip-prep # cache already built
set -euo pipefail
source "$(dirname "$0")/config.sh"
[[ "$GROUP" == CHANGE_ME ]] && { echo "Set GROUP in hpc/hipergator/config.sh first"; exit 1; }
[[ -f "$WORK/cases.json" ]] || { echo "No $WORK/cases.json: run get_data.sh first"; exit 1; }
cd "$WORK"
export REPO
ACC=(--account="$GROUP" --qos="$QOS")
read -ra SEED_LIST <<< "$SEEDS"
LAST=$(( ${#SEED_LIST[@]} - 1 ))

DEP=()
if [[ "${1:-}" != --skip-prep ]]; then
  PREP=$(sbatch --parsable "${ACC[@]}" --partition="$CPU_PARTITION" --export=ALL "$REPO/hpc/hipergator/prep.sbatch")
  echo "prep   job $PREP"
  DEP=(--dependency=afterok:$PREP)
fi
TRAIN=$(sbatch --parsable "${ACC[@]}" "${DEP[@]}" --partition="$GPU_PARTITION" --gres="gpu:$GPU_GRES" \
  --array=0-$LAST --export=ALL "$REPO/hpc/hipergator/train.sbatch")
echo "train  job $TRAIN (array 0-$LAST, seeds: $SEEDS, $TRAIN_MINUTES min each)"
EXPORT=$(sbatch --parsable "${ACC[@]}" --dependency=afterok:$TRAIN --partition="$GPU_PARTITION" \
  --gres="gpu:$GPU_GRES" --export=ALL "$REPO/hpc/hipergator/export.sbatch")
echo "export job $EXPORT (runs after all training finishes)"
echo "Logs in $WORK. Watch with these two commands:"
echo "  squeue -u $USER"
echo "  tail -f $WORK/artery-train-${TRAIN}_0.log"
