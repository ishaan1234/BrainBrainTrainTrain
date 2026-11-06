"""Surface extraction and binary STL writing."""
import numpy as np


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop 26-connected components smaller than `min_voxels`."""
    if min_voxels <= 0:
        return mask
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_voxels
    keep[0] = False
    return keep[lab]


def write_stl(path, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = verts[faces].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(b"arteryseg mesh".ljust(80, b" "))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def read_stl_vertices(path) -> np.ndarray:
    """(N*3, 3) vertices of a binary STL; used by tests and for quick checks."""
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        rec = np.frombuffer(f.read(n * 50), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return rec["v"].reshape(-1, 3)


def volume_to_stl(vol: np.ndarray, affine: np.ndarray, path, level: float = 0.5, coords: str = "LPS",
                  step: int = 1) -> int:
    """Marching cubes at `level`, vertices in world mm. LPS matches 3D Slicer; RAS matches nibabel.
    Returns the triangle count (0 and no file if nothing reaches `level`)."""
    from skimage.measure import marching_cubes

    vol = np.pad(np.asarray(vol, dtype=np.float32), 1)  # zero border -> closed meshes at the scan edge
    if vol.max() < level:
        print(f"  nothing above {level}, no STL written: {path}", flush=True)
        return 0
    verts, faces, _, _ = marching_cubes(vol, level=level, step_size=step, allow_degenerate=False)
    verts = (verts - 1) @ affine[:3, :3].T + affine[:3, 3]
    if np.linalg.det(affine[:3, :3]) < 0:  # mirrored grid: flip winding to keep normals outward
        faces = faces[:, ::-1]
    if coords == "LPS":  # RAS -> LPS is a reflection too
        verts[:, :2] *= -1
        faces = faces[:, ::-1]
    write_stl(path, verts, np.ascontiguousarray(faces))
    return len(faces)


def stl_from_prob(prob: np.ndarray, affine: np.ndarray, path, spacing: float, threshold: float = 0.5,
                  min_component_mm3: float = 50.0) -> int:
    """Remove small disconnected specks, then mesh the smooth probability map at `threshold`."""
    from scipy import ndimage

    keep = remove_small_components(prob >= threshold, int(min_component_mm3 / spacing ** 3))
    near = ndimage.binary_dilation(keep, iterations=1)  # 1-voxel margin keeps kept surfaces smooth
    p = np.where(near, prob, np.minimum(prob, threshold * 0.99)).astype(np.float32)
    return volume_to_stl(p, affine, path, level=threshold)
