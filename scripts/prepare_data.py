#!/usr/bin/env python3
"""Preprocess labeled cases for training, in parallel.

python scripts/prepare_data.py --cases cases.json --out data/cache

cases.json maps a case id to its CT and binary artery label (any grid; cropped labels are fine):
  {"case01": {"image": "raw/case01/ct.nii.gz", "label": "raw/case01/arteries.nii"}, ...}
Writes per case: image.npy (3,D,H,W) float16, label.npy uint8, fg.npy / hard.npy sampling indices,
and a records.json index used by train.py and evaluate.py.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arteryseg import preprocess as P  # noqa: E402

HARD_NEGATIVE_HU = 200.0  # bright voxels that are not artery: bone, calcification
HARD_STRIDE = 8           # subsample the hard-negative index list to bound memory


def prepare(job):
    case, img_path, lbl_path, out = job
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = nib.load(img_path)
    hu, affine = P.to_model_grid(nib.Nifti1Image(np.asarray(src.dataobj, dtype=np.float32), src.affine))
    x = P.normalize(hu)
    del hu
    lab = nib.load(lbl_path)
    y = P.label_to_grid(lab, x.shape[1:], affine)
    fg = np.flatnonzero(y.ravel()).astype(np.int64)
    if len(fg) == 0:
        raise ValueError(f"{case}: label is empty on the model grid")
    lo, hi = P.WINDOWS[0]
    bright = x[0].ravel()[::HARD_STRIDE] > (HARD_NEGATIVE_HU - lo) / (hi - lo)
    hard = (np.flatnonzero(bright & (y.ravel()[::HARD_STRIDE] == 0)) * HARD_STRIDE).astype(np.int64)
    for name, arr in (("image", x), ("label", y), ("fg", fg), ("hard", hard)):
        np.save(out / f"{name}.npy", arr)
    src_mm3 = float((np.asarray(lab.dataobj) > 0).sum() * np.prod(lab.header.get_zooms()[:3]))
    rec = dict(case=case, image=str(out / "image.npy"), label=str(out / "label.npy"), fg=str(out / "fg.npy"),
               hard=str(out / "hard.npy"), shape=list(x.shape[1:]), affine=affine.tolist(),
               source_image=str(img_path), native_shape=list(src.shape), native_affine=src.affine.tolist(),
               label_mm3_source=src_mm3, label_mm3_grid=float(len(fg) * P.SPACING ** 3))
    print(f"[{case}] grid {x.shape[1:]}, label {src_mm3:.0f} -> {rec['label_mm3_grid']:.0f} mm3 "
          f"({time.time() - t0:.0f}s)", flush=True)
    return case, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="JSON: {case_id: {image, label}}")
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (0 = one per case)")
    a = ap.parse_args()
    base = Path(a.cases).resolve().parent
    cases = json.loads(Path(a.cases).read_text())
    out = Path(a.out)
    jobs = [(c, str(base / v["image"]), str(base / v["label"]), out / c) for c, v in cases.items()]
    with Pool(a.workers or len(jobs)) as pool:
        records = dict(pool.map(prepare, jobs))
    (out / "records.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {out / 'records.json'} ({len(records)} cases)")


if __name__ == "__main__":
    main()
