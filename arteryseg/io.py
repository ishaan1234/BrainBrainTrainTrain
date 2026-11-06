"""Load a CT scan from a NIfTI file, a DICOM folder, or a .zip of either."""
from pathlib import Path
import tempfile
import zipfile

import nibabel as nib
import numpy as np


def _in_memory(img: nib.Nifti1Image) -> nib.Nifti1Image:
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape {img.shape}")
    return nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), img.affine)


def load_dicom_dir(root: Path) -> nib.Nifti1Image:
    """Read the DICOM series with the most slices found anywhere under `root`."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    best = None
    for d in sorted({p.parent for p in Path(root).rglob("*") if p.is_file()}):
        for sid in reader.GetGDCMSeriesIDs(str(d)) or []:
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        raise FileNotFoundError(f"No DICOM series found under {root}")
    print(f"DICOM: using largest series ({len(best)} slices) in {Path(best[0]).parent}", flush=True)
    reader.SetFileNames(best)
    image = reader.Execute()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ct.nii"
        sitk.WriteImage(image, str(out))  # SimpleITK converts DICOM LPS to a NIfTI (RAS) affine
        return _in_memory(nib.load(str(out)))


def load_ct(path) -> nib.Nifti1Image:
    """Return the scan as an in-memory float32 HU volume with its affine."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(p) as z:
                z.extractall(td)
            return load_ct(td)
    if p.is_dir():
        files = [q for q in p.rglob("*") if q.is_file()]
        niis = sorted(q for q in files if q.name.lower().endswith((".nii", ".nii.gz"))
                      and "label" not in q.name.lower() and "seg" not in q.name.lower())
        if niis and len(files) <= 10:  # a folder holding a NIfTI, not a DICOM series
            return load_ct(niis[0])
        return load_dicom_dir(p)
    return _in_memory(nib.load(str(p)))
