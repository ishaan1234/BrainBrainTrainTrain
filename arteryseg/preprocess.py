"""Resampling and intensity normalization shared by training and inference."""
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

SPACING = 0.5  # isotropic model grid, mm
# HU windows, one input channel each: vessel/soft tissue, bone, wide (full useful range)
WINDOWS = ((-100.0, 700.0), (0.0, 2000.0), (-1024.0, 3000.0))
IN_CHANNELS = len(WINDOWS)
AIR_HU = -1024.0


def to_model_grid(img: nib.Nifti1Image):
    """Reorient to RAS and resample linearly to SPACING mm isotropic. Outside the scan is air.
    Returns (hu float32 array, affine)."""
    can = nib.as_closest_canonical(img)
    hu = np.asarray(can.dataobj, dtype=np.float32)
    res = resample_to_output(nib.Nifti1Image(hu, can.affine), voxel_sizes=[SPACING] * 3, order=1, cval=AIR_HU)
    return np.asarray(res.dataobj, dtype=np.float32), res.affine


def normalize(hu: np.ndarray) -> np.ndarray:
    """HU volume (D,H,W) -> (C,D,H,W) float16, each channel one window clipped and scaled to [0, 1]."""
    out = np.empty((IN_CHANNELS,) + hu.shape, np.float16)
    for i, (lo, hi) in enumerate(WINDOWS):
        out[i] = (np.clip(hu, lo, hi) - lo) / (hi - lo)
    return out


def label_to_grid(label_img: nib.Nifti1Image, shape, affine) -> np.ndarray:
    """Resample a binary label onto a target grid purely through the affines, so labels stored on a
    cropped sub-grid (e.g. 3D Slicer segment exports) are placed correctly. Linear interpolation then
    a 0.5 threshold gives smoother edges than nearest neighbour."""
    m = nib.Nifti1Image((np.asarray(label_img.dataobj) > 0).astype(np.float32), label_img.affine)
    r = resample_from_to(m, (tuple(shape), affine), order=1, cval=0)
    return (np.asarray(r.dataobj) >= 0.5).astype(np.uint8)


def body_bbox(x: np.ndarray, margin: int = 16):
    """Bounding box (tuple of slices) of everything denser than -500 HU, plus a margin.
    Arteries are inside the body, so cropping the surrounding air only saves time."""
    lo, hi = WINDOWS[2]
    body = np.asarray(x[2]) > (-500 - lo) / (hi - lo)
    idx = [np.flatnonzero(body.any(axis=tuple(a for a in range(3) if a != d))) for d in range(3)]
    shape = x.shape[1:]
    if any(len(i) == 0 for i in idx):
        return tuple(slice(0, s) for s in shape)
    return tuple(slice(max(0, int(i[0]) - margin), min(s, int(i[-1]) + 1 + margin)) for i, s in zip(idx, shape))
