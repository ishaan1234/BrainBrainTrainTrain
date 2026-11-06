"""Network definitions and checkpoint I/O.

Released checkpoints are stored split into `<name>.pt.part00`, `.part01`, ... (each under GitHub's
100 MB file limit); they are reassembled in memory on load. A plain `<name>.pt` works too."""
import io
from pathlib import Path

import torch

from .preprocess import IN_CHANNELS, SPACING, WINDOWS

ARCHS = ("dynunet", "segresnet", "swinunetr")


def build_model(arch: str) -> torch.nn.Module:
    from monai.networks.nets import DynUNet, SegResNet, SwinUNETR

    if arch == "dynunet":  # nnU-Net style: residual blocks, deep supervision
        return DynUNet(spatial_dims=3, in_channels=IN_CHANNELS, out_channels=1,
                       kernel_size=[3] * 6, strides=[1, 2, 2, 2, 2, 2], upsample_kernel_size=[2] * 5,
                       filters=[32, 64, 128, 256, 320, 320], norm_name="instance",
                       act_name=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
                       deep_supervision=True, deep_supr_num=3, res_block=True)
    if arch == "segresnet":
        return SegResNet(spatial_dims=3, in_channels=IN_CHANNELS, out_channels=1, init_filters=32,
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), norm="instance")
    if arch == "swinunetr":  # Swin-transformer encoder + CNN decoder
        return SwinUNETR(in_channels=IN_CHANNELS, out_channels=1, feature_size=48, use_v2=True,
                         norm_name="instance", spatial_dims=3)
    raise ValueError(f"unknown arch {arch!r}, expected one of {ARCHS}")


def save_checkpoint(path, model: torch.nn.Module, arch: str, roi, **extra):
    torch.save(dict(model=model.state_dict(), arch=arch, roi=list(roi), spacing=SPACING,
                    windows=[list(w) for w in WINDOWS], **extra), path)


PART_BYTES = 45 * 2 ** 20


def _parts(path: Path) -> list:
    return sorted(path.parent.glob(path.name + ".part[0-9][0-9]"))


def split_checkpoint(path, part_bytes: int = PART_BYTES) -> list:
    """Split `path` into numbered parts next to it and delete the original."""
    path = Path(path)
    data = path.read_bytes()
    parts = []
    for i in range(0, len(data), part_bytes):
        q = path.with_name(f"{path.name}.part{i // part_bytes:02d}")
        q.write_bytes(data[i:i + part_bytes])
        parts.append(q)
    path.unlink()
    return parts


def load_checkpoint(path) -> dict:
    path = Path(path)
    if not path.exists() and _parts(path):
        src = io.BytesIO(b"".join(q.read_bytes() for q in _parts(path)))
    else:
        src = str(path)
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if ck.get("spacing", SPACING) != SPACING or [tuple(w) for w in ck.get("windows", WINDOWS)] != list(WINDOWS):
        raise ValueError(f"{path} was trained with different preprocessing (spacing/windows)")
    return ck


def find_checkpoints(path) -> list:
    """Checkpoints in a folder (plain .pt or split .pt.partNN), or a single checkpoint path."""
    p = Path(path)
    if p.is_dir():
        names = {q.name for q in p.glob("*.pt")} | {q.name[:-7] for q in p.glob("*.pt.part00")}
        paths = [p / n for n in sorted(names)]
    else:
        paths = [p]
    if not paths or not all(q.is_file() or _parts(q) for q in paths):
        raise FileNotFoundError(f"no .pt checkpoints at {p}")
    return paths


def load_models(paths, device):
    """Returns [(model, roi), ...] in eval mode on `device`."""
    out = []
    for p in paths:
        ck = load_checkpoint(p)
        m = build_model(ck["arch"])
        m.load_state_dict(ck["model"])
        out.append((m.to(device).eval(), ck["roi"]))
        print(f"loaded {Path(p).name}: {ck['arch']} roi={ck['roi']}", flush=True)
    return out
