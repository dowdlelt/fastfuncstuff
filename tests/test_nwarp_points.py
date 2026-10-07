"""nwarpforge sampling a point list instead of a grid (the surface route).

The referee is the volume path itself: points placed at the master's voxel centres
must reproduce the volume output through the same chain. A second test pins the
frame: points are true scanner mm, entering through the master's REAL affine, so an
oblique master must still recover a function of scanner position.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.nwarpforge import nwarpforge

DEV = torch.device("cpu")


def _oblique_affine(vox: float, deg: float, origin=(-20.0, -18.0, -16.0)) -> np.ndarray:
    t = np.deg2rad(deg)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    a = np.eye(4)
    a[:3, :3] = rot * vox
    a[:3, 3] = origin
    return a


def _write_aff1d(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(" ".join(f"{v:.8f}" for v in r) + "\n")


def _motion_rows(nt: int) -> list[list[float]]:
    rows = []
    for t in range(nt):
        a = np.deg2rad(1.5 * t)
        rows.append(
            [np.cos(a), -np.sin(a), 0, 0.3 * t, np.sin(a), np.cos(a), 0, -0.2 * t, 0, 0, 1, 0.1 * t]
        )
    return rows


def _chain(tmp_path, n: int, nt: int, aff: np.ndarray):
    rng = np.random.default_rng(0)
    # Smooth enough that the kernel matters but nothing is at the noise floor.
    base = rng.random((n, n, n)).astype(np.float32)
    k = np.ones(3) / 3
    for ax in range(3):
        base = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, base)
    src = np.stack([base * (1 + 0.05 * t) + 1 for t in range(nt)], axis=-1).astype(np.float32)
    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(src, aff).to_filename(str(src_path))

    fmap = tmp_path / "fmap.nii"
    w = np.zeros((n, n, n, 3), dtype=np.float32)
    w[..., 1] = 0.3 * np.sin(np.arange(n, dtype=np.float32) / 3.0)[None, :, None]
    nib.Nifti1Image(w, aff).to_filename(str(fmap))
    motion = tmp_path / "motion.aff12.1D"
    _write_aff1d(motion, _motion_rows(nt))
    return src_path, [str(fmap), str(motion)]


def _voxel_centres_mm(shape, aff):
    """Scanner mm of every master voxel, in the volume's (z, y, x) memory order."""
    nx, ny, nz = shape
    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    ijk = np.c_[ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)]
    return (aff @ ijk.T)[:3].T


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"slice_times": list(np.linspace(0.0, 0.9, 10)), "tr": 1.0},
        {"jac_axis": 1, "jac_match": "fmap"},
    ],
    ids=["chain", "slice_timing", "jac"],
)
def test_points_at_voxel_centres_reproduce_the_volume(tmp_path, extra):
    n, nt = 10, 4
    aff = _oblique_affine(2.0, 12.0)
    src_path, chain = _chain(tmp_path, n, nt, aff)
    common = dict(
        source_path=str(src_path),
        nwarp_specs=chain,
        master_path=str(src_path),
        interp="wsinc5",
        device=DEV,
        verb=0,
        auto_pad=False,
        **extra,
    )
    out = tmp_path / "vol.nii"
    nwarpforge(prefix=str(out), **common)
    vol = np.asarray(nib.load(str(out)).dataobj)  # (x, y, z, t)

    pts = _voxel_centres_mm((n, n, n), aff)
    got = nwarpforge(prefix="", points=pts, **common)
    assert got is not None and tuple(got.shape) == (nt, n**3)
    want = np.moveaxis(vol, -1, 0).transpose(0, 3, 2, 1).reshape(nt, -1)  # (t, z*y*x)
    np.testing.assert_allclose(got.numpy(), want, atol=2e-4 * float(np.abs(want).max()))


def test_points_recover_a_function_of_scanner_position_on_an_oblique_grid(tmp_path):
    # f is linear in TRUE scanner mm and trilinear interpolation is exact for it, so
    # any frame slip (the cardinal-vs-real affine trap) shows up as an error.
    n = 14
    aff = _oblique_affine(2.0, 15.0)
    xyz = _voxel_centres_mm((n, n, n), aff)
    coef = np.array([0.7, -0.4, 1.1])
    vals = (xyz @ coef + 50.0).reshape(n, n, n).transpose(2, 1, 0)  # back to (x, y, z)
    src_path = tmp_path / "f.nii"
    nib.Nifti1Image(vals.astype(np.float32), aff).to_filename(str(src_path))
    ident = tmp_path / "ident.aff12.1D"
    _write_aff1d(ident, [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])

    rng = np.random.default_rng(3)
    ijk = rng.uniform(3, n - 4, size=(200, 3))
    pts = (aff @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T
    got = nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[str(ident)],
        prefix="",
        master_path=str(src_path),
        interp="linear",
        device=DEV,
        verb=0,
        points=pts,
    )
    assert got is not None
    np.testing.assert_allclose(got.numpy(), pts @ coef + 50.0, atol=2e-3)


def test_points_refuse_what_has_no_meaning_off_grid(tmp_path):
    n = 6
    aff = _oblique_affine(2.0, 0.0)
    src_path, chain = _chain(tmp_path, n, 2, aff)
    pts = np.zeros((4, 3))
    kw = dict(source_path=str(src_path), nwarp_specs=chain, prefix="", device=DEV, verb=0)
    with pytest.raises(ValueError, match="AXIS:FIELDMAP"):
        nwarpforge(points=pts, jac_axis=1, **kw)
    with pytest.raises(ValueError, match="dxyz"):
        nwarpforge(points=pts, dxyz=1.0, **kw)
    with pytest.raises(ValueError, match="anatomy"):
        nwarpforge(points=pts, master_path="WARP", **kw)
