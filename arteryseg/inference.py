"""Sliding-window inference and export to the original scan grid."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to


@torch.inference_mode()
def predict_prob(models, x: np.ndarray, device, overlap: float = 0.5, sw_batch: int = 4,
                 flip_tta: bool = False) -> np.ndarray:
    """Mean sigmoid probability (D,H,W) float32 over `models` = [(model, roi), ...].

    `x` is the normalized (C,D,H,W) volume; float16 is fine. Each window is cast to float32 before
    it enters a network, so all network math is fp32 while the full volume stays small in memory."""
    from monai.inferers import sliding_window_inference

    arr = np.ascontiguousarray(x)
    if not arr.flags.writeable:  # memory-mapped input: copy once so torch can own it
        arr = arr.copy()
    inp = torch.from_numpy(arr).to(device)[None]
    flips = [()] + ([(2,)] if flip_tta else [])  # left-right only (RAS axis 0), anatomically valid
    acc, n = None, 0
    for model, roi in models:
        def run(window, m=model):
            return m(window.float())

        for f in flips:
            xi = inp.flip(f) if f else inp
            logit = sliding_window_inference(xi, roi_size=tuple(roi), sw_batch_size=sw_batch, predictor=run,
                                             overlap=overlap, mode="gaussian", sw_device=device, device=device,
                                             progress=device.type == "cpu")
            p = torch.sigmoid_(logit.float())
            p = p.flip(f) if f else p
            acc = p if acc is None else acc.add_(p)
            n += 1
            del logit, xi
    return (acc / n)[0, 0].cpu().numpy()


def save_native(prob: np.ndarray, grid_affine, ref_shape, ref_affine, outdir: Path,
                threshold: float = 0.5, save_prob: bool = False) -> np.ndarray:
    """Resample probabilities linearly to the original scan grid, then threshold.
    Writes prediction.nii.gz (and probability.nii.gz if save_prob). Returns the binary mask."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = np.asarray(resample_from_to(nib.Nifti1Image(prob.astype(np.float32, copy=False), grid_affine),
                                    (tuple(ref_shape), ref_affine), order=1, cval=0).dataobj, dtype=np.float32)
    np.clip(p, 0, 1, out=p)
    seg = (p >= threshold).astype(np.uint8)
    if save_prob:
        nib.save(nib.Nifti1Image(p, ref_affine), str(outdir / "probability.nii.gz"))
    nib.save(nib.Nifti1Image(seg, ref_affine), str(outdir / "prediction.nii.gz"))
    return seg
