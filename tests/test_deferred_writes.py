"""io.afni.deferred_writes: background NIfTI writes for batch runs."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from fastfuncstuff.io import afni


def _vol(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((9, 8, 7, 5)).astype(np.float32)


@pytest.mark.parametrize("ext", [".nii", ".nii.gz", ".nii.zst"])
def test_deferred_write_is_identical_to_a_direct_one(tmp_path, ext):
    data = _vol(0)
    afni.save_nifti(data, tmp_path / f"direct{ext}", affine=np.diag([2, 2, 2, 1.0]), tr=1.5)
    with afni.deferred_writes():
        afni.save_nifti(data, tmp_path / f"later{ext}", affine=np.diag([2, 2, 2, 1.0]), tr=1.5)
    a = afni.load_nifti(tmp_path / f"direct{ext}")
    b = afni.load_nifti(tmp_path / f"later{ext}")
    np.testing.assert_array_equal(np.asarray(a.dataobj), np.asarray(b.dataobj))
    np.testing.assert_array_equal(a.affine, b.affine)
    assert a.header["pixdim"][4] == b.header["pixdim"][4] == 1.5


def test_reading_a_pending_write_waits_for_it(tmp_path, monkeypatch):
    gate = threading.Event()
    real = afni._finish_nifti_write

    def slow(*args, **kwargs):
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(afni, "_finish_nifti_write", slow)
    path = tmp_path / "x.nii"
    with afni.deferred_writes():
        afni.save_nifti(_vol(1), path, affine=np.eye(4))
        assert not path.exists() or path.stat().st_size == 0  # still queued
        threading.Timer(0.2, gate.set).start()
        got = np.asarray(afni.load_nifti(path).dataobj)  # must block, not read a stub
    np.testing.assert_array_equal(got, _vol(1))


def test_write_errors_surface_when_the_block_ends(tmp_path):
    bad = tmp_path / "missing_dir" / "x.nii"
    with pytest.raises(OSError, match="deferred write"):
        with afni.deferred_writes():
            afni.save_nifti(_vol(2), bad, affine=np.eye(4))
    assert afni._deferred is None  # never left installed


def test_pending_bytes_are_bounded(tmp_path, monkeypatch):
    peak = []
    real = afni._finish_nifti_write

    def watch(*args, **kwargs):
        peak.append(afni._deferred.bytes if afni._deferred else 0)
        return real(*args, **kwargs)

    monkeypatch.setattr(afni, "_finish_nifti_write", watch)
    one = _vol(3).nbytes
    with afni.deferred_writes(max_bytes=2 * one):
        for i in range(6):
            afni.save_nifti(_vol(3), tmp_path / f"{i}.nii", affine=np.eye(4))
    assert max(peak) <= 2 * one
