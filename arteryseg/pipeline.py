"""End-to-end: CT scan -> artery STL + mask on the original grid."""
from pathlib import Path
import time

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
import torch

from . import io, mesh, models as M, preprocess as P
from .inference import predict_prob, save_native


def segment(input_path, model_path, out_dir, threshold: float = 0.5, overlap: float = 0.5,
            min_component_mm3: float = 50.0, tta: bool = False, device: str = "auto",
            save_prob: bool = False, crop: bool = True, threads: int = 0) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    if threads:
        torch.set_num_threads(threads)
    nets = M.load_models(M.find_checkpoints(model_path), dev)

    print("loading", input_path, flush=True)
    src = io.load_ct(input_path)
    ref_shape, ref_affine = src.shape, src.affine
    print(f"  shape {src.shape}, voxel {np.round(src.header.get_zooms()[:3], 3)} mm", flush=True)
    hu, affine = P.to_model_grid(src)
    del src
    x = P.normalize(hu)
    del hu
    shape = x.shape[1:]
    bb = P.body_bbox(x) if crop else tuple(slice(0, s) for s in shape)
    sub = np.ascontiguousarray(x[(slice(None),) + bb])
    del x
    print(f"  model grid {shape} at {P.SPACING} mm, body crop {sub.shape[1:]} ({time.time() - t0:.0f}s)", flush=True)

    p_sub = predict_prob(nets, sub, dev, overlap=overlap, flip_tta=tta, sw_batch=1 if dev.type == "cpu" else 4)
    del sub
    prob = np.zeros(shape, np.float32)
    prob[bb] = p_sub
    del p_sub
    print(f"  inference done on {dev} ({time.time() - t0:.0f}s)", flush=True)

    stl = out / "arteries.stl"
    ntri = mesh.stl_from_prob(prob, affine, stl, P.SPACING, threshold=threshold, min_component_mm3=min_component_mm3)
    save_native(prob, affine, ref_shape, ref_affine, out, threshold=threshold, save_prob=save_prob)
    info = dict(stl=str(stl) if ntri else None, triangles=ntri, artery_ml=float((prob >= threshold).sum() * P.SPACING ** 3 / 1000),
                seconds=round(time.time() - t0, 1), device=str(dev),
                peak_ram_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2) if resource else None)
    print(f"DONE in {info['seconds']:.0f}s: {info['stl'] or 'no arteries found, no STL'} ({ntri:,} triangles), "
          f"artery volume {info['artery_ml']:.1f} mL, "
          f"peak RAM {info['peak_ram_gib']} GiB", flush=True)
    return info
