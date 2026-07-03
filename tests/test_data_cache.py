"""
Tests for data_cache.py — HDF5 caching for preprocessed fMRI data,
called from ffs_reml.

Cache silently returning stale data is the single failure mode this
module exists to prevent. Every test pins one of: round-trip fidelity,
hash-based invalidation, optional-metadata round-trip, or the standalone
check_cache_valid query path.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.data_cache import (
    _compute_file_hash,
    check_cache_valid,
    load_cache,
    save_cache,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_inputs(tmp_path):
    """Create two dummy input files representing 'NIfTI runs'."""
    files = []
    for i in range(2):
        p = tmp_path / f"run-{i + 1}.nii.gz"
        p.write_bytes(b"fake nifti content " + bytes([i]))
        files.append(p)
    return files


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache.h5"


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 200)).astype(np.float32)


# ---------------------------------------------------------------------------
# _compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_deterministic_for_same_inputs(self, dummy_inputs):
        h1 = _compute_file_hash(dummy_inputs)
        h2 = _compute_file_hash(dummy_inputs)
        assert h1 == h2

    def test_order_invariant(self, dummy_inputs):
        """Hash should not depend on input file ordering — implementation
        sorts internally."""
        h1 = _compute_file_hash(dummy_inputs)
        h2 = _compute_file_hash(list(reversed(dummy_inputs)))
        assert h1 == h2

    def test_changes_when_file_mtime_changes(self, dummy_inputs):
        h1 = _compute_file_hash(dummy_inputs)
        # Bump mtime by rewriting
        time.sleep(0.01)
        dummy_inputs[0].write_bytes(b"updated content")
        # Ensure mtime is actually different — some filesystems have coarse
        # mtime granularity, so force it.
        import os

        new_mtime = dummy_inputs[0].stat().st_mtime + 1
        os.utime(dummy_inputs[0], (new_mtime, new_mtime))
        h2 = _compute_file_hash(dummy_inputs)
        assert h1 != h2

    def test_changes_when_paths_change(self, tmp_path, dummy_inputs):
        h1 = _compute_file_hash(dummy_inputs)
        other = tmp_path / "different.nii.gz"
        other.write_bytes(b"x")
        h2 = _compute_file_hash([dummy_inputs[0], other])
        assert h1 != h2

    def test_handles_nonexistent_file(self, tmp_path):
        """Hashing a path that doesn't exist on disk: implementation skips
        mtime and uses the path string only. Should not raise."""
        h = _compute_file_hash([tmp_path / "ghost.nii.gz"])
        assert isinstance(h, str) and len(h) == 32  # md5 hex


# ---------------------------------------------------------------------------
# save_cache / load_cache round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_data_round_trip_exact(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        loaded, meta = load_cache(cache_file, dummy_inputs)
        np.testing.assert_array_equal(loaded, small_data)
        assert meta["n_voxels"] == small_data.shape[0]
        assert meta["n_timepoints"] == small_data.shape[1]

    def test_optional_metadata_round_trip(self, cache_file, dummy_inputs, small_data):
        affine = np.eye(4) * 2.0
        affine[3, 3] = 1.0
        run_starts = [0, 100]
        volume_shape = (5, 5, 2)

        save_cache(
            cache_file,
            small_data,
            dummy_inputs,
            run_starts=run_starts,
            affine=affine,
            volume_shape=volume_shape,
            was_scaled=True,
            original_mean=987.5,
        )
        _, meta = load_cache(cache_file, dummy_inputs)
        np.testing.assert_array_equal(meta["run_starts"], run_starts)
        np.testing.assert_array_equal(meta["affine"], affine)
        assert meta["volume_shape"] == volume_shape
        assert meta["was_scaled"] is np.True_ or meta["was_scaled"] is True
        assert meta["original_mean"] == pytest.approx(987.5)

    def test_input_files_stored_as_absolute_paths(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        _, meta = load_cache(cache_file, dummy_inputs)
        for stored, original in zip(meta["input_files"], dummy_inputs):
            assert Path(stored).is_absolute()
            assert Path(stored) == original.absolute()

    def test_nifti_header_pickle_round_trip(self, cache_file, dummy_inputs, small_data):
        """Headers are saved as a sidecar .header.pkl, not inside the HDF5."""
        # Use a simple picklable stand-in (a dict) — load_cache doesn't
        # introspect header type, just unpickles it.
        fake_header = {"pixdim": [1.0, 2.0, 2.0, 2.0, 1.5], "units": "mm"}
        save_cache(cache_file, small_data, dummy_inputs, nifti_header=fake_header)
        sidecar = Path(str(cache_file) + ".header.pkl")
        assert sidecar.exists(), "header sidecar should be written"

        _, meta = load_cache(cache_file, dummy_inputs)
        assert meta["nifti_header"] == fake_header

    def test_no_header_means_no_sidecar(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        sidecar = Path(str(cache_file) + ".header.pkl")
        assert not sidecar.exists()

        _, meta = load_cache(cache_file, dummy_inputs)
        assert "nifti_header" not in meta


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


class TestInvalidation:
    def test_load_raises_when_inputs_changed(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        # Bump mtime
        import os

        new_mtime = dummy_inputs[0].stat().st_mtime + 100
        os.utime(dummy_inputs[0], (new_mtime, new_mtime))
        with pytest.raises(ValueError, match="stale"):
            load_cache(cache_file, dummy_inputs)

    def test_load_validate_false_skips_check(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        import os

        new_mtime = dummy_inputs[0].stat().st_mtime + 100
        os.utime(dummy_inputs[0], (new_mtime, new_mtime))
        # Should load successfully because validation is disabled
        loaded, _ = load_cache(cache_file, dummy_inputs, validate=False)
        np.testing.assert_array_equal(loaded, small_data)

    def test_load_missing_file_raises(self, cache_file):
        with pytest.raises(FileNotFoundError):
            load_cache(cache_file)

    def test_load_skips_validation_when_input_files_none(
        self, cache_file, dummy_inputs, small_data
    ):
        save_cache(cache_file, small_data, dummy_inputs)
        # No input_files passed → validation skipped even if validate=True
        loaded, _ = load_cache(cache_file, input_files=None, validate=True)
        np.testing.assert_array_equal(loaded, small_data)


# ---------------------------------------------------------------------------
# check_cache_valid
# ---------------------------------------------------------------------------


class TestCheckCacheValid:
    def test_returns_true_for_fresh_cache(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        assert check_cache_valid(cache_file, dummy_inputs) is True

    def test_returns_false_when_file_missing(self, tmp_path, dummy_inputs):
        assert check_cache_valid(tmp_path / "missing.h5", dummy_inputs) is False

    def test_returns_false_for_stale_cache(self, cache_file, dummy_inputs, small_data):
        save_cache(cache_file, small_data, dummy_inputs)
        import os

        new_mtime = dummy_inputs[0].stat().st_mtime + 100
        os.utime(dummy_inputs[0], (new_mtime, new_mtime))
        assert check_cache_valid(cache_file, dummy_inputs) is False

    def test_returns_false_for_corrupt_file(self, cache_file, dummy_inputs):
        """A non-HDF5 file at the cache path should be reported as invalid,
        not raise — callers check this and fall back to recomputing."""
        cache_file.write_bytes(b"not an HDF5 file")
        assert check_cache_valid(cache_file, dummy_inputs) is False
