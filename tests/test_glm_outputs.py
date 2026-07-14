"""Tests for GLM NIfTI export utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.core import fit_glm
from fastfuncstuff.glm.outputs import write_glm_results_nifti


def test_write_glm_results_nifti_roundtrip():
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Simulate a tiny dataset
    n_timepoints = 60
    n_voxels = 8
    n_regressors = 2
    data = torch.randn(n_voxels, n_timepoints, device=device)
    design = torch.randn(n_timepoints, n_regressors, device=device)

    results = fit_glm(data, design, tr=1.5, verbose=False, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = write_glm_results_nifti(
            results,
            tmpdir,
            prefix="unit",
            condition_names=["stim_a", "stim_b"],
            include_beta=True,
            include_tstat=True,
            include_fstat=True,
            include_r2=True,
            include_mean=True,
            include_sigma=True,
            write_residuals=False,
            write_predictions=False,
            volume_shape=(2, 2, 2),
        )

        stats_path = Path(out["stats"])
        assert stats_path.exists(), "Stats NIfTI missing"

        # There should be 4 volumes: beta/tstat pair per condition
        import nibabel as nib

        stats_img = nib.load(str(stats_path))
        assert stats_img.shape == (2, 2, 2, 4), "Unexpected stats volume shape"
        assert np.isclose(stats_img.header["pixdim"][4], 1.5), "TR not encoded"

        meta = json.loads(Path(out["stats_meta"]).read_text())
        assert len(meta["volumes"]) == 4, "Metadata length mismatch"
        assert meta["volumes"][0]["condition"] == "stim_a"
        assert meta["volumes"][0]["metric"] == "beta"
        assert meta["volumes"][1]["condition"] == "stim_b"
        assert meta["volumes"][1]["metric"] == "beta"
        assert meta["volumes"][2]["condition"] == "stim_a"
        assert meta["volumes"][2]["metric"] == "tstat"
        assert meta["volumes"][3]["condition"] == "stim_b"
        assert meta["volumes"][3]["metric"] == "tstat"

        assert Path(out["fstat"]).exists(), "F-stat map missing"
        assert Path(out["r2"]).exists(), "R2 map missing"
        assert Path(out["mean"]).exists(), "Mean volume missing"
        assert Path(out["sigma"]).exists(), "Sigma volume missing"


class TestGLMOutputsComprehensive:
    """Comprehensive tests for GLM output utilities."""

    @pytest.fixture
    def mock_results(self):
        """Create a mock GLMResults object."""
        from fastfuncstuff.glm.core import GLMResults

        results = GLMResults()
        results.betas = torch.randn(10, 4)  # 10 voxels, 4 regressors
        results.tstats = torch.randn(10, 4)
        results.r2 = torch.randn(10)
        results.sigma2 = torch.rand(10)
        results.fstats = torch.rand(10)  # Needed for AFNI bucket
        results.dof = 100
        results.tr = 2.0
        results.original_shape = (5, 2, 1)  # 10 voxels
        results.affine = np.eye(4)
        return results

    def test_slice_glm_results(self, mock_results):
        """Test slicing GLM results."""
        from fastfuncstuff.glm.outputs import slice_glm_results

        # Slice first 2 regressors
        indices = [0, 1]
        sliced = slice_glm_results(mock_results, indices)

        assert sliced.betas.shape == (10, 2)
        assert sliced.tstats.shape == (10, 2)
        assert sliced.r2.shape == (10,)  # Unchanged
        assert torch.allclose(sliced.betas, mock_results.betas[:, :2])

        # Test numpy inputs
        mock_results.betas = mock_results.betas.numpy()
        sliced_np = slice_glm_results(mock_results, indices)
        assert isinstance(sliced_np.betas, np.ndarray)
        assert sliced_np.betas.shape == (10, 2)

    def test_reconstruct_partial_timeseries(self):
        """betas × selected columns reproduces that part of the fitted signal, and
        the -save_clean identity (data - nuisance_fit + mean == task_fit + resid + mean)
        holds exactly."""
        from fastfuncstuff.glm.outputs import reconstruct_partial_timeseries

        rng = np.random.default_rng(0)
        n_time, n_reg, n_vox = 40, 5, 12
        design = rng.standard_normal((n_time, n_reg)).astype(np.float32)
        design[:, 0] = 1.0  # constant (baseline / mean nuisance column)
        betas = rng.standard_normal((n_vox, n_reg)).astype(np.float32)
        resid = rng.standard_normal((n_vox, n_time)).astype(np.float32)

        task_idx = [3, 4]
        nuis_idx = [0, 1, 2]
        data = betas @ design.T + resid  # (n_vox, n_time)

        task_fit = reconstruct_partial_timeseries(betas, design, task_idx)
        nuis_fit = reconstruct_partial_timeseries(betas, design, nuis_idx)

        # Partitioning the columns partitions the fitted signal.
        assert np.allclose(task_fit + nuis_fit, betas @ design.T, atol=1e-4)

        # Clean built from the fit equals data minus nuisance plus per-voxel mean.
        grand_mean = betas @ design.mean(axis=0) + resid.mean(axis=1)
        clean = task_fit + resid + grand_mean[:, None]
        assert np.allclose(clean, data - nuis_fit + data.mean(axis=1)[:, None], atol=1e-3)

        # Empty selection yields an all-zero (n_vox, n_time) block, not an error.
        empty = reconstruct_partial_timeseries(betas, design, [])
        assert empty.shape == (n_vox, n_time)
        assert np.all(empty == 0.0)

        # Exact 3-way partition: data = taskfit + nuisance_fit + residual.
        assert np.allclose(task_fit + nuis_fit + resid, data, atol=1e-4)

    def test_find_baseline_columns(self):
        """Per-run polort-0 (constant-over-support) columns are detected numerically,
        motion/higher-degree columns are not."""
        from fastfuncstuff.glm.outputs import find_baseline_columns

        n_time = 30
        design = np.zeros((n_time, 5), dtype=np.float32)
        # col 0: run-1 constant (first half), col 1: run-2 constant (second half)
        design[: n_time // 2, 0] = 1.0
        design[n_time // 2 :, 1] = 1.0
        # col 2: linear ramp (polort-1) — not constant
        design[:, 2] = np.linspace(-1, 1, n_time)
        # col 3: motion-like — not constant
        design[:, 3] = np.random.default_rng(1).standard_normal(n_time)
        # col 4: a task column (also varies)
        design[:, 4] = np.sin(np.arange(n_time))

        # Restrict the search to the nuisance candidates (0-3); task col 4 excluded.
        found = find_baseline_columns(design, [0, 1, 2, 3])
        assert found == [0, 1]

        # Per-run baseline keeps run offsets; grand mean would not.
        assert design[: n_time // 2, 0].std() == 0.0

    def test_extract_onset_times(self):
        """Test extracting onset times from design matrix."""
        from fastfuncstuff.glm.outputs import extract_onset_times_from_design

        # Design matrix: 10 timepoints, 2 regressors
        # Reg 0: onset at 3
        # Reg 1: onset at 5
        design = np.zeros((10, 2))
        design[3:, 0] = 1
        design[5:, 1] = 1

        onsets = extract_onset_times_from_design(design, [0, 1])
        assert onsets == [3, 5]

        # Test empty column
        design[:, 0] = 0
        onsets_empty = extract_onset_times_from_design(design, [0])
        assert onsets_empty[0] >= 10  # Should be sorted to end

    def test_write_single_trials_output(self, mock_results):
        """Test writing single trial betas."""
        import nibabel as nib

        from fastfuncstuff.glm.outputs import write_single_trials_output

        # Setup design matrix with interleaved onsets
        # Reg 0: onset 5
        # Reg 1: onset 2
        # Reg 2: onset 8
        # Reg 3: onset 1
        design = np.zeros((20, 4))
        design[5, 0] = 1
        design[2, 1] = 1
        design[8, 2] = 1
        design[1, 3] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "single_trials.nii.gz"

            written_path = write_single_trials_output(
                mock_results,
                out_path,
                design,
                stim_indices=[0, 1, 2, 3],
                stim_labels=["d5", "d2", "d8", "d1"],
            )

            assert written_path.exists()
            assert written_path.name.endswith(".nii.gz")

            # Check JSON sidecar
            json_path = written_path.with_suffix("").with_suffix(".json")  # Remove .gz then .nii
            assert json_path.exists()

            meta = json.loads(json_path.read_text())
            # Sorted order should be: d1(1), d2(2), d5(5), d8(8)
            # Original indices: 3, 1, 0, 2
            assert meta["Labels"] == ["d1", "d2", "d5", "d8"]
            assert meta["OnsetTimes"] == [1, 2, 5, 8]

            # Check NIfTI shape
            img = nib.load(str(written_path))
            assert img.shape == (5, 2, 1, 4)

    def test_write_glm_bucket_as_nifti(self, mock_results):
        """Test writing AFNI bucket file."""
        import nibabel as nib

        from fastfuncstuff.glm.outputs import write_glm_bucket_as_nifti

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test writing NIfTI format
            out_path = Path(tmpdir) / "bucket.nii.gz"

            written = write_glm_bucket_as_nifti(
                mock_results,
                out_path,
                condition_names=["A", "B", "C", "D"],
                apply_afni_metadata=False,  # Skip 3drefit check
            )

            assert written.exists()

            img = nib.load(str(written))
            # Shape: 1 (Full_F) + 4 conditions * 2 (Beta+T) = 9 volumes
            assert img.shape[-1] == 9

            # Test AFNI format (HEAD/BRIK) - check for WARNING
            afni_path = Path(tmpdir) / "bucket+tlrc.HEAD"
            with pytest.warns(UserWarning, match="direct writing"):
                written_afni = write_glm_bucket_as_nifti(
                    mock_results,
                    afni_path,
                    condition_names=["A", "B", "C", "D"],
                    apply_afni_metadata=False,
                )

            assert written_afni.exists()
            assert str(written_afni).endswith(".nii.gz")

    def test_write_afni_bucket_deprecation(self, mock_results):
        """Test deprecated write_afni_bucket function."""
        from fastfuncstuff.glm.outputs import write_afni_bucket

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "old.nii.gz"

            with pytest.warns(DeprecationWarning, match="deprecated"):
                written = write_afni_bucket(
                    mock_results,
                    out_path,
                    condition_names=["A", "B", "C", "D"],
                    apply_afni_metadata=False,
                )
            assert written.exists()

    def test_error_cases(self, mock_results):
        """Test error handling."""
        from fastfuncstuff.glm.outputs import (
            _reshape_parameter_map,
            _resolve_shape,
            write_glm_results_nifti,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Condition names mismatch
            with pytest.raises(ValueError, match="condition_names has length"):
                write_glm_results_nifti(
                    mock_results,
                    tmpdir,
                    condition_names=["A", "B"],  # only 2 names for 4 regressors
                )

            # 2. Missing T-stats
            mock_results.tstats = None
            with pytest.raises(ValueError, match="T-statistics requested"):
                write_glm_results_nifti(mock_results, tmpdir, include_tstat=True)

            # 3. Missing spatial info
            mock_results.original_shape = None
            mock_results.full_shape = None
            with pytest.raises(ValueError, match="do not contain spatial"):
                _resolve_shape(mock_results, None)

            # 4. Invalid shape for mask
            mask = np.zeros(10, dtype=bool)
            data_3d = np.zeros((2, 5, 2))  # 3D data instead of flat
            with pytest.raises(ValueError, match="Data must be 1D or 2D"):
                _reshape_parameter_map(data_3d, (10,), mask)

    def test_file_format_detection(self):
        """Test file format detection logic."""
        from fastfuncstuff.glm.outputs import _normalize_output_path

        path, fmt = _normalize_output_path("test.nii")
        assert fmt == "nifti"

        path, fmt = _normalize_output_path("test.nii.gz")
        assert fmt == "nifti_gz"

        path, fmt = _normalize_output_path("test+tlrc.HEAD")
        assert fmt == "afni"
        assert str(path) == "test"  # Stripped extension and view

        path, fmt = _normalize_output_path("test")
        assert fmt == "nifti_gz"
        assert str(path).endswith(".nii.gz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
