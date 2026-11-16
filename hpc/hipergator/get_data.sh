#!/bin/bash
# Download a shared Google Drive folder of cases and build cases.json. Run on a login node:
#   bash hpc/hipergator/get_data.sh "https://drive.google.com/drive/folders/<id>"
# Re-running only fetches files that are not there yet.
# Layout expected: one sub-folder (or .zip) per case holding one CT and one 0/1 artery label NIfTI.
set -euo pipefail
source "$(dirname "$0")/config.sh"
URL="${1:?usage: get_data.sh <google-drive-folder-url>}"
activate_env
mkdir -p "$WORK/raw"
python - "$URL" "$WORK/raw" <<'EOF'
import os, sys, gdown
url, out = sys.argv[1], sys.argv[2]
for f in gdown.download_folder(url, skip_download=True, quiet=True):
    if f.path.lower().endswith(".stl"):
        continue  # reference meshes are not needed for training
    dst = os.path.join(out, f.path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print("downloading", f.path, flush=True)
        gdown.download(id=f.id, output=dst, quiet=True)
EOF
python "$REPO/scripts/make_manifest.py" "$WORK/raw" --out "$WORK/cases.json"
echo "Check the case list above, then: bash hpc/hipergator/run_all.sh"
