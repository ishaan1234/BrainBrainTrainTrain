#!/usr/bin/env python3
"""Build cases.json from a folder with one sub-folder per case.

python scripts/make_manifest.py raw/ --out cases.json

Each sub-folder (or .zip, which is extracted first) should hold one CT and one binary artery label.
Files are told apart by their voxel values, not their names: a NIfTI whose values are all 0/1 is
the label, anything else is the CT. Duplicates (same voxels and grid, e.g. .nii and .nii.gz copies)
are ignored. Folders with no CT, no
label, or more than one distinct CT/label are skipped and reported, so check the summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import nibabel as nib
import numpy as np


def content_hash(p: Path) -> str:
    """Hash of grid + voxel values, so a .nii and its .nii.gz copy compare equal."""
    img = nib.load(str(p))
    h = hashlib.sha1(np.asarray(img.affine, np.float32).round(4).tobytes())
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(np.asarray(img.dataobj)).tobytes())
    return h.hexdigest()


def is_label(p: Path) -> bool:
    img = nib.load(str(p))
    if np.dtype(img.header.get_data_dtype()).kind == "f" and img.header.get_data_dtype() != np.float32:
        return False
    x = np.asarray(img.dataobj[::4, ::4, ::4]) if img.ndim == 3 else np.asarray(img.dataobj)
    return bool(np.isin(np.unique(x), [0, 1]).all())


def case_id(name: str) -> str:
    """'Patient 6' -> pat006, 'case03' -> case003, '084' -> 084, anything else -> sanitized name."""
    m = re.search(r"\d+", name)
    if not m:
        return re.sub(r"\W+", "_", name)
    low = name.lower()
    prefix = "pat" if low.startswith("patient") else "case" if low.startswith("case") else ""
    return f"{prefix}{int(m.group()):03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="cases.json")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for z in sorted(root.glob("*.zip")):
        target = root / z.stem
        if not target.exists():
            with zipfile.ZipFile(z) as zz:
                zz.extractall(root)
    out_dir = Path(a.out).resolve().parent
    cases, problems = {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        niis, seen = [], set()
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.name.lower().endswith((".nii", ".nii.gz")):
                h = content_hash(f)
                if h not in seen:
                    seen.add(h)
                    niis.append(f)
        labels = [f for f in niis if is_label(f)]
        cts = [f for f in niis if f not in labels]
        cid = case_id(d.name)
        if len(cts) != 1 or len(labels) != 1:
            problems.append(f"{d.name}: {len(cts)} CT(s), {len(labels)} label(s): skipped")
            continue
        ct, lb = nib.load(str(cts[0])), nib.load(str(labels[0]))
        if [round(float(z), 3) for z in ct.header.get_zooms()[:3]] != [round(float(z), 3) for z in lb.header.get_zooms()[:3]]:
            problems.append(f"{d.name}: CT and label voxel sizes differ (label is still resampled through its affine)")
        cases[cid] = dict(image=str(Path(cts[0]).relative_to(out_dir) if cts[0].is_relative_to(out_dir) else cts[0]),
                          label=str(Path(labels[0]).relative_to(out_dir) if labels[0].is_relative_to(out_dir) else labels[0]))
        print(f"{cid:8s} CT {cts[0].relative_to(root)}  |  label {labels[0].relative_to(root)}")
    Path(a.out).write_text(json.dumps(cases, indent=2))
    print(f"\n{len(cases)} cases -> {a.out}")
    for p in problems:
        print("  !", p)


if __name__ == "__main__":
    main()
