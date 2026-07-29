"""
Tests for the threaded run loader in io/afni.py.

Loading runs concurrently is only safe if it is *invisible*: same data, same
order, same run_starts, whatever the thread count. These tests compare the
threaded path against the serial one on the formats we actually ship
(.nii, .nii.gz, and .nii.zst when zstd is installed).
"""

from __future__ import annotations

import shutil
import subprocess

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.io.afni import (
    _peek_run_length,
    load_and_concatenate_runs,
    resolve_load_threads,
)

SHAPE = (5, 6, 4)
RUN_LENS = [7, 9, 5, 8]


def _write_runs(tmp_path, ext=".nii.gz"):
    """One file per run, each with a distinct, checkable pattern."""
    paths = []
    rng = np.random.default_rng(0)
    for i, n_tp in enumerate(RUN_LENS):
        arr = rng.normal(100 + 10 * i, 1, (*SHAPE, n_tp)).astype(np.float32)
        img = nib.Nifti1Image(arr, np.eye(4))
        img.header.set_zooms((2.0, 2.0, 2.0, 2.0))
        base = tmp_path / f"run{i}.nii.gz"
        nib.save(img, base)
        if ext == ".nii.zst":
            plain = tmp_path / f"run{i}.nii"
            nib.save(img, plain)
            out = tmp_path / f"run{i}.nii.zst"
            subprocess.run(["zstd", "-q", "-f", str(plain), "-o", str(out)], check=True)
            plain.unlink()
            paths.append(str(out))
        else:
            paths.append(str(base))
    return paths


@pytest.mark.parametrize("ext", [".nii.gz", ".nii.zst"])
def test_threaded_load_matches_serial(tmp_path, ext):
    if ext == ".nii.zst" and shutil.which("zstd") is None:
        pytest.skip("zstd not installed")
    files = _write_runs(tmp_path, ext)

    serial, starts_serial = load_and_concatenate_runs(files, keep_on_cpu=True, load_threads=1)
    for n in (2, 3, 8):  # more threads than runs must also be fine
        threaded, starts = load_and_concatenate_runs(files, keep_on_cpu=True, load_threads=n)
        assert starts == starts_serial, f"run_starts differ at {n} threads"
        assert torch.equal(serial, threaded), f"data differs at {n} threads"

    assert starts_serial == [0, 7, 16, 21]
    assert serial.shape == (int(np.prod(SHAPE)), sum(RUN_LENS))


def test_threaded_load_preserves_run_order(tmp_path):
    """Each run has a distinct mean; out-of-order assembly would scramble them."""
    files = _write_runs(tmp_path)
    data, starts = load_and_concatenate_runs(files, keep_on_cpu=True, load_threads=4)
    ends = starts[1:] + [data.shape[1]]
    means = [float(data[:, s:e].mean()) for s, e in zip(starts, ends, strict=True)]
    assert means == sorted(means), f"runs are out of order: {means}"
    for i, m in enumerate(means):
        assert abs(m - (100 + 10 * i)) < 1.0


def test_masked_threaded_load_matches_serial(tmp_path):
    files = _write_runs(tmp_path)
    n_vox = int(np.prod(SHAPE))
    mask = np.zeros(n_vox, dtype=bool)
    mask[::3] = True

    serial, _ = load_and_concatenate_runs(files, keep_on_cpu=True, mask_flat=mask, load_threads=1)
    threaded, _ = load_and_concatenate_runs(files, keep_on_cpu=True, mask_flat=mask, load_threads=4)
    assert serial.shape[0] == mask.sum()
    assert torch.equal(serial, threaded)


@pytest.mark.parametrize("ext", [".nii.gz", ".nii.zst"])
def test_peek_run_length_matches_the_real_load(tmp_path, ext):
    """The peek feeds the preallocated (single-copy) path; a wrong answer there
    silently truncates or over-allocates the dataset."""
    if ext == ".nii.zst" and shutil.which("zstd") is None:
        pytest.skip("zstd not installed")
    files = _write_runs(tmp_path, ext)
    assert [_peek_run_length(f) for f in files] == RUN_LENS


def test_peek_returns_none_for_unreadable_file(tmp_path):
    bad = tmp_path / "not_a_nifti.nii.gz"
    bad.write_bytes(b"definitely not a nifti")
    assert _peek_run_length(bad) is None


def test_resolve_load_threads_bounds():
    assert resolve_load_threads(1) == 1  # single run: nothing to overlap
    assert resolve_load_threads(4, requested=1) == 1  # explicit opt-out
    assert resolve_load_threads(2, requested=99) == 2  # never exceeds run count
    assert 1 <= resolve_load_threads(10) <= 8
    # A run that eats the whole RAM budget forces serial loading.
    assert resolve_load_threads(10, bytes_per_run=10**15) == 1


def test_load_threads_respect_the_cpu_budget(monkeypatch):
    """The loader draws from the same budget as compute: a 2-core cap must not
    become 8 loader threads just because there are 20 runs to read."""
    for var in ("FFS_LOAD_THREADS", "FFS_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert resolve_load_threads(20) <= 2
