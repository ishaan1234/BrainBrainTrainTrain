#!/usr/bin/env python3
"""CT scan (DICOM folder, .zip, or NIfTI) -> arteries.stl + prediction.nii.gz.

python scripts/predict.py /path/to/dicom_folder --out result/
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arteryseg.pipeline import segment  # noqa: E402

DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="DICOM folder, .zip of one, or .nii/.nii.gz CT")
    ap.add_argument("--models", default=str(DEFAULT_MODELS), help="folder of .pt checkpoints (averaged) or one .pt")
    ap.add_argument("--out", default="result")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap; 0.25 is ~2x faster")
    ap.add_argument("--min-component-mm3", type=float, default=50.0, help="drop disconnected specks smaller than this")
    ap.add_argument("--tta", action="store_true", help="add left-right flip test-time augmentation (2x slower)")
    ap.add_argument("--save-prob", action="store_true", help="also write probability.nii.gz (large)")
    ap.add_argument("--no-crop", action="store_true", help="run on the full field of view, air included")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    a = ap.parse_args()
    segment(a.input, a.models, a.out, threshold=a.threshold, overlap=a.overlap,
            min_component_mm3=a.min_component_mm3, tta=a.tta, device="cpu" if a.cpu else "auto",
            save_prob=a.save_prob, crop=not a.no_crop, threads=a.threads)


if __name__ == "__main__":
    main()
