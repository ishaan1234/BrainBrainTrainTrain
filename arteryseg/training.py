"""Patch sampling, GPU-side augmentation and loss used by scripts/train.py."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Crops(Dataset):
    """Oversized random crops (headroom for rotation/scaling) from preprocessed cases.

    Centres: `p_fg` on an artery voxel, `p_hard` on a bright non-artery voxel (>200 HU, bone and
    calcification), the rest on random tissue. Each index has its own RNG stream, so sampling is
    reproducible regardless of the number of workers."""

    def __init__(self, records, size: int, seed: int, p_fg: float = 0.5, p_hard: float = 0.2):
        self.records, self.size, self.seed, self.p_fg, self.p_hard = records, size, seed, p_fg, p_hard
        self._arrays = {}

    def __len__(self):
        return 10 ** 8  # effectively infinite; training stops on its time budget

    def _get(self, i):
        if i not in self._arrays:  # lazily memory-map, once per worker
            r = self.records[i]
            self._arrays[i] = tuple(np.load(r[k], mmap_mode="r") for k in ("image", "label", "fg", "hard"))
        return self._arrays[i]

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        x, y, fg, hard = self._get(int(rng.integers(len(self.records))))
        shape, size = np.array(y.shape), np.full(3, self.size)
        u = rng.random()
        if u < self.p_fg:
            c = np.array(np.unravel_index(int(fg[rng.integers(len(fg))]), y.shape))
        elif u < self.p_fg + self.p_hard and len(hard):
            c = np.array(np.unravel_index(int(hard[rng.integers(len(hard))]), y.shape))
        else:
            for _ in range(32):
                c = rng.integers(shape)
                if x[0][tuple(c)] > 0.05:  # inside the body, not air
                    break
        c = c + rng.integers(-8, 9, 3)  # jitter so targets are not always centred
        lo = c - size // 2
        src_lo, src_hi = np.maximum(lo, 0), np.minimum(lo + size, shape)
        dst_lo = src_lo - lo
        dst_hi = dst_lo + (src_hi - src_lo)
        xs = np.zeros((x.shape[0],) + tuple(size), np.float16)  # zero-padding = below every window
        ys = np.zeros(tuple(size), np.uint8)
        s = tuple(slice(int(a), int(b)) for a, b in zip(src_lo, src_hi))
        d = tuple(slice(int(a), int(b)) for a, b in zip(dst_lo, dst_hi))
        xs[(slice(None),) + d] = x[(slice(None),) + s]
        ys[d] = y[s]
        return torch.from_numpy(xs), torch.from_numpy(ys)[None]


def crop_size(patch: int) -> int:
    """Oversized crop edge so ±15° rotation and 0.85x scaling stay (almost) inside real data."""
    return int(math.ceil(patch * 1.375 / 16) * 16)


def rot_matrices(angles: torch.Tensor) -> torch.Tensor:
    a, b, c = angles.unbind(1)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    rx = torch.stack([o, z, z, z, a.cos(), -a.sin(), z, a.sin(), a.cos()], 1).view(-1, 3, 3)
    ry = torch.stack([b.cos(), z, b.sin(), z, o, z, -b.sin(), z, b.cos()], 1).view(-1, 3, 3)
    rz = torch.stack([c.cos(), -c.sin(), z, c.sin(), c.cos(), z, z, z, o], 1).view(-1, 3, 3)
    return rz @ ry @ rx


def gpu_augment(x: torch.Tensor, y: torch.Tensor, patch: int, aug: bool = True):
    """(B,C,S,S,S) fp16 crops + (B,1,S,S,S) uint8 labels -> (B,C,P,P,P), (B,1,P,P,P) float32.

    Random rotation (±15°, p=.7), isotropic scale (0.85-1.15, p=.7), left-right flip, contrast
    scale/shift, gamma on the vessel channel, Gaussian noise, and thick-slice simulation along z."""
    B, dev, S = x.shape[0], x.device, x.shape[-1]
    x, y = x.float(), y.float()
    if aug:
        ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(15) * (torch.rand(B, 1, device=dev) < 0.7)
        sc = torch.where(torch.rand(B, device=dev) < 0.7, torch.empty(B, device=dev).uniform_(0.85, 1.15),
                         torch.ones(B, device=dev))
    else:
        ang, sc = torch.zeros(B, 3, device=dev), torch.ones(B, device=dev)
    A = rot_matrices(ang) * (patch / S) / sc[:, None, None]
    theta = torch.cat([A, torch.zeros(B, 3, 1, device=dev)], 2)
    grid = F.affine_grid(theta, (B, 1, patch, patch, patch), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = (F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False) >= 0.5).float()
    if not aug:
        return x, y

    flip = (torch.rand(B, device=dev) < 0.5)[:, None, None, None, None]  # RAS axis 0 = left-right
    x, y = torch.where(flip, x.flip(2), x), torch.where(flip, y.flip(2), y)
    x = x * torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.9, 1.1) \
        + torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)
    gamma = torch.where(torch.rand(B, device=dev) < 0.3, torch.empty(B, device=dev).uniform_(0.7, 1.5),
                        torch.ones(B, device=dev))
    x[:, 0] = x[:, 0].clamp(0, 1) ** gamma[:, None, None, None]
    sigma = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.3) * torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0, 0.03)
    x = x + torch.randn_like(x) * sigma
    for i in range(B):  # thick slices (1-1.25 mm scans)
        if torch.rand(()) < 0.25:
            k = int(patch / float(torch.empty(()).uniform_(1.5, 2.5)))
            xi = F.interpolate(x[i:i + 1], size=(patch, patch, k), mode="trilinear", align_corners=False)
            x[i:i + 1] = F.interpolate(xi, size=(patch, patch, patch), mode="trilinear", align_corners=False)
    return x.clamp(0, 1), y


def seg_loss(z: torch.Tensor, y: torch.Tensor):
    """Batch soft Dice + BCE. Returns (loss, dice_loss)."""
    p = torch.sigmoid(z)
    dice = 1 - (2 * (p * y).sum() + 1e-5) / (p.sum() + y.sum() + 1e-5)
    return dice + F.binary_cross_entropy_with_logits(z, y), dice


def deep_supervision_loss(z: torch.Tensor, y: torch.Tensor):
    """DynUNet training output (B, heads, C, D, H, W): heads weighted 1, 1/2, 1/4, ... (normalized)."""
    w = torch.tensor([0.5 ** i for i in range(z.shape[1])], device=z.device)
    w = w / w.sum()
    parts = [seg_loss(z[:, i], y) for i in range(z.shape[1])]
    return sum(wi * p[0] for wi, p in zip(w, parts)), parts[0][1]


def cosine_lr(frac: float, lr_max: float, warmup: float = 0.03, lr_min: float = 1e-6) -> float:
    if frac < warmup:
        return lr_max * frac / warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (frac - warmup) / (1 - warmup)))
