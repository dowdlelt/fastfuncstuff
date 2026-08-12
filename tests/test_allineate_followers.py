"""-source_follower rides the solved transform, including onto a different grid.

The bug this guards against is applying the source's *voxel* matrix to a follower
that lives on its own grid: the follower must be re-expressed through its affine
(inv(A_follower) @ A_source @ M), otherwise a follower at a different voxel size
or origin comes out shifted/scaled while looking plausible.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.allineate import main


def _write(path, arr, affine):
    # arr is (nz, ny, nx); NIfTI wants (nx, ny, nz)
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32).transpose(2, 1, 0), affine), str(path))


def _phantom(n=32):
    vol = np.zeros((n, n, n), np.float32)
    vol[8:24, 10:22, 12:26] = 80.0
    return vol


def _run_with_followers(tmp_path, follower_affine, follower_shape_factor):
    src = _phantom()
    base = np.roll(src, 2, axis=0)  # small, recoverable shift
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -32.0

    # The follower is 2x the source everywhere in *world* space, resampled onto
    # its own grid, so a correct warp must reproduce 2x the aligned source.
    n = 32 * follower_shape_factor
    ijk = np.indices((n, n, n)).reshape(3, -1)
    world = follower_affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
    src_vox = (np.linalg.inv(aff) @ world)[:3]
    from scipy.ndimage import map_coordinates

    fol = map_coordinates(src.transpose(2, 1, 0) * 2, src_vox, order=1).reshape(n, n, n)
    fol = fol.transpose(2, 1, 0)

    _write(tmp_path / "base.nii", base, aff)
    _write(tmp_path / "src.nii", src, aff)
    _write(tmp_path / "fol.nii", fol, follower_affine)

    main(
        [
            "-base", str(tmp_path / "base.nii"),
            "-source", str(tmp_path / "src.nii"),
            "-prefix", str(tmp_path / "out.nii"),
            "-source_follower", str(tmp_path / "fol.nii"),
            "-follower_prefix", str(tmp_path / "out_fol.nii"),
            "-device", "cpu",
            "-fast",
            "-verb", "0",
        ]
    )  # fmt: skip

    out = nib.load(str(tmp_path / "out.nii"))
    fout = nib.load(str(tmp_path / "out_fol.nii"))
    return out, fout


def test_follower_same_grid(tmp_path):
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -32.0
    out, fout = _run_with_followers(tmp_path, aff, 1)

    assert fout.shape == out.shape
    assert np.allclose(fout.affine, out.affine)
    o, f = out.get_fdata(), fout.get_fdata()
    m = o > 10
    assert np.allclose(f[m] / o[m], 2.0, atol=1e-3)


def test_follower_on_a_different_grid(tmp_path):
    pytest.importorskip("scipy")
    faff = np.diag([1.0, 1.0, 1.0, 1.0])  # 1mm, same origin, twice the matrix
    faff[:3, 3] = -32.0
    out, fout = _run_with_followers(tmp_path, faff, 2)

    # Output always lands on the base/output grid, never the follower's.
    assert fout.shape == out.shape
    assert np.allclose(fout.affine, out.affine)
    o, f = out.get_fdata(), fout.get_fdata()
    m = o > 10
    assert np.allclose(f[m] / o[m], 2.0, atol=2e-2)


def test_mismatched_follower_prefix_count_errors(tmp_path):
    aff = np.eye(4)
    _write(tmp_path / "base.nii", _phantom(16), aff)
    _write(tmp_path / "src.nii", _phantom(16), aff)
    _write(tmp_path / "fol.nii", _phantom(16), aff)

    with pytest.raises(SystemExit):
        main(
            [
                "-base", str(tmp_path / "base.nii"),
                "-source", str(tmp_path / "src.nii"),
                "-prefix", str(tmp_path / "out.nii"),
                "-source_follower", str(tmp_path / "fol.nii"), str(tmp_path / "fol.nii"),
                "-follower_prefix", str(tmp_path / "out_fol.nii"),
                "-device", "cpu",
                "-verb", "0",
            ]
        )  # fmt: skip
