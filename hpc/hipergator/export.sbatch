#!/bin/bash
#SBATCH --job-name=artery-export
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# After training: collect the models into $WORK/models (split for GitHub), score them on every
# case and write an STL per case to $WORK/export. Submitted by run_all.sh.
set -euo pipefail
source "$REPO/hpc/hipergator/config.sh"
activate_env
mkdir -p "$WORK/models"
for run in "$WORK"/runs/dynunet_s*/; do
  name=$(basename "$run")
  cp "$run/final.pt" "$WORK/models/$name.pt"
done
CASES=$(python -c "import json; print(','.join(json.load(open('$WORK/cache/records.json'))))")
python "$REPO/scripts/evaluate.py" --records "$WORK/cache/records.json" --models "$WORK/models" \
  --cases "$CASES" --out "$WORK/export"
python - "$WORK/models" <<'EOF'
import sys
from pathlib import Path
from arteryseg.models import split_checkpoint
for p in sorted(Path(sys.argv[1]).glob("*.pt")):
    print("split", [q.name for q in split_checkpoint(p)])
EOF
echo "Models: $WORK/models   STLs + metrics.json: $WORK/export"
