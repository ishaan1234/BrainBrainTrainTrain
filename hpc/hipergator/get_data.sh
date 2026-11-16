#!/bin/bash
# Download a shared Google Drive folder of cases and build cases.json. Run on a login node:
#   bash hpc/hipergator/get_data.sh "https://drive.google.com/drive/folders/<id>"
# Re-running only fetches files that are not there yet, so if Drive rate-limits you, just run it again.
# Layout expected: one sub-folder (or .zip) per case holding one CT and one 0/1 artery label NIfTI.
set -euo pipefail
source "$(dirname "$0")/config.sh"
URL="${1:?usage: get_data.sh <google-drive-folder-url>}"
activate_env
mkdir -p "$WORK/raw"
python - "$URL" "$WORK/raw" <<'PYEOF'
import os, sys, time, gdown
url, out = sys.argv[1], sys.argv[2]
SKIP = (".stl", ".csv")  # reference meshes and Slicer colour tables are not needed for training
failed = []
for f in gdown.download_folder(url, skip_download=True, quiet=True):
    if f.path.lower().endswith(SKIP):
        continue
    dst = os.path.join(out, f.path)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):  # Drive rate-limits bursts of downloads: back off and retry
        try:
            print("downloading", f.path, flush=True)
            gdown.download(id=f.id, output=dst + ".part", quiet=True)
            os.replace(dst + ".part", dst)
            break
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"  failed ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        failed.append(f.path)
if failed:
    print("\nCould not download these (Drive rate limit). Wait ~1 h and re-run this script:")
    for p in failed:
        print("  ", p)
PYEOF
python "$REPO/scripts/make_manifest.py" "$WORK/raw" --out "$WORK/cases.json"
echo "Check the case list above, then: bash hpc/hipergator/run_all.sh"
