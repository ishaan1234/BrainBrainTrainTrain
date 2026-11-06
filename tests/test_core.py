"""CPU tests on synthetic data (no patient data needed)."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from arteryseg import io, mesh, models as M, preprocess as P
from arteryseg.metrics import scores
from arteryseg.training import crop_size, gpu_augment


def lps_affine(spacing, origin):
    """NIfTI affine of a typical CT stored L-P-S like the DICOM-derived scans."""
    return np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]) + np.c_[np.zeros((4, 3)), list(origin) + [0]]


def test_normalize_windows():
    hu = np.array([[[-2000, -100, 300, 700, 2000, 5000]]], np.float32)
    x = P.normalize(hu)
    assert x.shape == (3, 1, 1, 6) and x.dtype == np.float16
    np.testing.assert_allclose(x[0, 0, 0], [0, 0, 0.5, 1, 1, 1], atol=1e-3)
    np.testing.assert_allclose(x[1, 0, 0], [0, 0, 0.15, 0.35, 1, 1], atol=1e-3)


def test_label_on_cropped_subgrid_lands_in_place():
    """3D Slicer exports segments on a cropped grid; resampling must use the affines."""
    full = np.zeros((40, 40, 20), np.uint8)
    full[10:20, 22:30, 5:12] = 1
    aff = lps_affine((0.5, 0.5, 1.0), (10, 20, -5))
    off = np.array([8, 18, 3])
    crop_aff = aff.copy()
    crop_aff[:3, 3] = aff[:3, :3] @ off + aff[:3, 3]
    crop = full[8:30, 18:35, 3:16]
    y = P.label_to_grid(nib.Nifti1Image(crop, crop_aff), full.shape, aff)
    assert scores(y, full)["dice"] == 1.0


def test_model_grid_spacing_and_orientation():
    img = nib.Nifti1Image(np.zeros((20, 24, 10), np.float32), lps_affine((0.45, 0.45, 0.9), (0, 0, 0)))
    hu, aff = P.to_model_grid(img)
    assert nib.aff2axcodes(aff) == ("R", "A", "S")
    np.testing.assert_allclose(np.abs(np.diag(aff)[:3]), P.SPACING)


def test_body_bbox_skips_air():
    hu = np.full((60, 60, 60), P.AIR_HU, np.float32)
    hu[20:30, 25:40, 10:50] = 40
    bb = P.body_bbox(P.normalize(hu), margin=2)
    assert bb == (slice(18, 32), slice(23, 42), slice(8, 52))


def test_stl_of_sphere_is_closed_and_in_lps_mm(tmp_path):
    g = np.mgrid[:40, :40, :40].astype(np.float32)
    vol = (np.sqrt(((g - 20) ** 2).sum(0)) <= 10).astype(np.float32)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])  # RAS, origin 0
    n = mesh.volume_to_stl(vol, aff, tmp_path / "s.stl")
    v = mesh.read_stl_vertices(tmp_path / "s.stl")
    assert n > 1000 and len(v) == 3 * n
    centre = v.mean(0)
    np.testing.assert_allclose(centre, [-10, -10, 10], atol=0.1)  # x, y negated for LPS
    radius = np.linalg.norm(v - centre, axis=1)
    assert 4.5 < radius.mean() < 5.5


def test_small_components_removed():
    m = np.zeros((30, 30, 30), bool)
    m[2:12, 2:12, 2:12] = True
    m[25, 25, 25] = True
    out = mesh.remove_small_components(m, 10)
    assert out.sum() == 1000 and not out[25, 25, 25]


def test_gpu_augment_shapes_and_identity():
    S, Pt = crop_size(32), 32
    x = torch.rand(2, 3, S, S, S).half()
    y = torch.zeros(2, 1, S, S, S, dtype=torch.uint8)
    y[:, :, S // 2 - 4:S // 2 + 4] = 1
    xa, ya = gpu_augment(x, y, Pt, aug=True)
    assert xa.shape == (2, 3, Pt, Pt, Pt) and ya.shape == (2, 1, Pt, Pt, Pt)
    assert 0 <= float(xa.min()) and float(xa.max()) <= 1 and set(ya.unique().tolist()) <= {0.0, 1.0}
    _, y0 = gpu_augment(x, y, Pt, aug=False)  # no aug = centre crop: the slab fills the middle
    assert y0[:, :, Pt // 2].min() == 1 and y0[:, :, 0].max() == 0


@pytest.mark.parametrize("arch", M.ARCHS)
def test_models_forward_and_checkpoint(arch, tmp_path):
    net = M.build_model(arch).eval()
    with torch.no_grad():
        z = net(torch.rand(1, P.IN_CHANNELS, 64, 64, 64))
    assert z.shape == (1, 1, 64, 64, 64)
    M.save_checkpoint(tmp_path / f"{arch}.pt", net, arch, [64] * 3)
    [(net2, roi)] = M.load_models([tmp_path / f"{arch}.pt"], torch.device("cpu"))
    assert roi == [64] * 3
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))


def _phantom():
    hu = np.full((64, 64, 40), P.AIR_HU, np.float32)
    hu[8:56, 8:56, 4:36] = 40           # head-sized block of soft tissue
    hu[30:34, 10:54, 10:30] = 450       # a bright "vessel" slab
    return nib.Nifti1Image(hu, lps_affine((0.6, 0.6, 0.8), (20, 20, -10)))


def _write_dicom(img: nib.Nifti1Image, folder: Path):
    sitk = pytest.importorskip("SimpleITK")
    tmp = folder.parent / "src.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), img.affine), str(tmp))
    vol = sitk.ReadImage(str(tmp))
    folder.mkdir()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    d = vol.GetDirection()
    tags = [("0008|0060", "CT"), ("0020|000e", "1.2.826.0.1.3680043.2.1125.1.99"),
            ("0020|0037", "\\".join(map(str, (d[0], d[3], d[6], d[1], d[4], d[7]))))]
    for i in range(vol.GetDepth()):
        s = vol[:, :, i]
        for k, v in tags:
            s.SetMetaData(k, v)
        s.SetMetaData("0020|0032", "\\".join(map(str, vol.TransformIndexToPhysicalPoint((0, 0, i)))))
        s.SetMetaData("0020|0013", str(i))
        w.SetFileName(str(folder / f"IM{i:04d}.dcm"))
        w.Execute(s)


def test_dicom_and_nifti_load_identically(tmp_path):
    img = _phantom()
    _write_dicom(img, tmp_path / "dcm")
    a = io.load_ct(tmp_path / "src.nii.gz")
    b = io.load_ct(tmp_path / "dcm")
    np.testing.assert_allclose(b.affine, a.affine, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(b.dataobj), np.asarray(a.dataobj))


def test_pipeline_end_to_end(tmp_path):
    from arteryseg.pipeline import segment

    torch.manual_seed(0)
    M.save_checkpoint(tmp_path / "m.pt", M.build_model("dynunet"), "dynunet", [64] * 3)
    img = _phantom()
    nib.save(img, str(tmp_path / "ct.nii.gz"))
    info = segment(tmp_path / "ct.nii.gz", tmp_path / "m.pt", tmp_path / "out", device="cpu",
                   threshold=0.0, min_component_mm3=0)  # random weights: threshold 0 forces a mesh
    pred = nib.load(str(tmp_path / "out" / "prediction.nii.gz"))
    assert pred.shape == img.shape and np.allclose(pred.affine, img.affine)
    assert info["triangles"] > 0 and (tmp_path / "out" / "arteries.stl").stat().st_size > 84


def test_split_checkpoint_roundtrip(tmp_path):
    net = M.build_model("segresnet")
    M.save_checkpoint(tmp_path / "a.pt", net, "segresnet", [64] * 3)
    parts = M.split_checkpoint(tmp_path / "a.pt", part_bytes=10 * 2 ** 20)
    assert len(parts) > 1 and not (tmp_path / "a.pt").exists()
    [path] = M.find_checkpoints(tmp_path)
    [(net2, _)] = M.load_models([path], torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))
