"""Tests for ffs_util_3dmath — voxelwise reductions + 3dcalc-style expressions."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.util_3dmath import main


def _write(path, arr):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(path))
    return str(path)


def _run(argv, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["ffs_util_3dmath", *argv])
    return main()


def test_mean_across_inputs(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.full((4, 4, 3), 2.0))
    b = _write(tmp_path / "b.nii.gz", np.full((4, 4, 3), 4.0))
    c = _write(tmp_path / "c.nii.gz", np.full((4, 4, 3), 6.0))
    out = tmp_path / "mean.nii.gz"
    assert _run(["-input", a, b, c, "-mean", "-prefix", str(out)], monkeypatch) == 0
    got = nib.load(str(out)).get_fdata()
    np.testing.assert_allclose(got, 4.0)  # (2+4+6)/3


def test_max_and_min(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.array([[[1.0, 5.0]]]))
    b = _write(tmp_path / "b.nii.gz", np.array([[[3.0, 2.0]]]))
    omax = tmp_path / "max.nii.gz"
    omin = tmp_path / "min.nii.gz"
    _run(["-input", a, b, "-max", "-prefix", str(omax)], monkeypatch)
    _run(["-input", a, b, "-min", "-prefix", str(omin)], monkeypatch)
    np.testing.assert_array_equal(nib.load(str(omax)).get_fdata(), [[[3.0, 5.0]]])
    np.testing.assert_array_equal(nib.load(str(omin)).get_fdata(), [[[1.0, 2.0]]])


def test_expr_difference(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.full((3, 3, 2), 10.0))
    b = _write(tmp_path / "b.nii.gz", np.full((3, 3, 2), 4.0))
    out = tmp_path / "diff.nii.gz"
    assert _run(["-input", a, b, "-expr", "a-b", "-prefix", str(out)], monkeypatch) == 0
    np.testing.assert_allclose(nib.load(str(out)).get_fdata(), 6.0)


def test_expr_step_threshold(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.array([[[50.0, 150.0]]]))
    out = tmp_path / "mask.nii.gz"
    _run(["-input", a, "-expr", "step(a-100)", "-prefix", str(out)], monkeypatch)
    np.testing.assert_array_equal(nib.load(str(out)).get_fdata(), [[[0.0, 1.0]]])


def test_shape_mismatch_errors(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.zeros((4, 4, 3)))
    b = _write(tmp_path / "b.nii.gz", np.zeros((4, 4, 2)))
    assert _run(["-input", a, b, "-mean", "-prefix", str(tmp_path / "o.nii.gz")], monkeypatch) == 1


def test_header_affine_preserved(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.ones((4, 4, 3)))
    out = tmp_path / "o.nii.gz"
    _run(["-input", a, "-mean", "-prefix", str(out)], monkeypatch)
    np.testing.assert_allclose(nib.load(str(out)).affine, np.diag([3.0, 3.0, 3.0, 1.0]))


def test_mask_zeros_outside(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.full((2, 2, 2), 5.0))
    mask = _write(tmp_path / "m.nii.gz", np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]))
    out = tmp_path / "o.nii.gz"
    _run(["-input", a, "-mean", "-mask", mask, "-prefix", str(out)], monkeypatch)
    got = nib.load(str(out)).get_fdata()
    m = nib.load(mask).get_fdata()
    assert np.all(got[m == 0] == 0) and np.all(got[m > 0] == 5.0)


def test_reduction_and_expr_mutually_exclusive(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.nii.gz", np.ones((2, 2, 2)))
    with pytest.raises(SystemExit):
        _run(["-input", a, "-mean", "-expr", "a", "-prefix", str(tmp_path / "o.nii.gz")], monkeypatch)
