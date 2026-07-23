"""
Tests for uncovered functions in fastfuncstuff/utils.py:
- configure_torch_backends
- scale_to_percent_signal
- gaussian_blur_3d
- generate_synthetic_runs
- load_per_run_nuisance_files
- compute_power_spectrum / compute_power_spectra
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.utils import (
    calc_memory_usage,
    compute_power_spectra,
    compute_power_spectrum,
    configure_torch_backends,
    gaussian_blur_3d,
    generate_synthetic_runs,
    load_per_run_nuisance_files,
    scale_to_percent_signal,
)


class TestConfigureTorchBackends:
    def test_sets_matmul_precision(self):
        configure_torch_backends(torch.device("cpu"))
        assert torch.get_float32_matmul_precision() == "high"

    def test_cpu_does_not_crash(self):
        # Should simply not error on CPU
        configure_torch_backends(torch.device("cpu"))


class TestScaleToPercentSignal:
    def test_basic_scaling(self):
        """Mean of each run should be ~100 after scaling."""
        torch.manual_seed(42)
        n_voxels, n_tp = 50, 200
        data = torch.randn(n_voxels, n_tp) + 1000.0  # positive signal
        run_starts = [0, 100]

        scaled, violations, info = scale_to_percent_signal(data.clone(), run_starts, verbose=False)

        # Mean of each run should be ~100 (within tolerance for clipping)
        for _run_idx, (start, end) in enumerate(
            zip(run_starts, run_starts[1:] + [n_tp], strict=False)
        ):
            run_mean = scaled[:, start:end].mean(dim=1)
            # Most voxels should have mean ~100
            assert torch.abs(run_mean.mean() - 100.0) < 5.0

    def test_no_violations_for_stable_signal(self):
        """Stable signal (low variance) should have no ceiling violations."""
        data = torch.ones(10, 100) * 500.0 + torch.randn(10, 100) * 0.1
        _, violations, info = scale_to_percent_signal(data, [0], max_scale=200.0, verbose=False)
        assert info["n_violations"] == 0

    def test_violations_for_extreme_signal(self):
        """Very low mean + high spikes should trigger ceiling violations."""
        data = torch.ones(5, 50) * 1.0  # very low mean
        data[0, 10] = 1000.0  # extreme spike
        _, violations, info = scale_to_percent_signal(
            data.clone(), [0], max_scale=200.0, verbose=False
        )
        assert info["n_violations"] > 0
        assert info["n_voxels_with_violations"] > 0

    def test_zero_mean_voxels(self):
        """Zero-mean voxels should be set to 0, not NaN/Inf."""
        data = torch.zeros(3, 50)
        data[1, :] = 100.0  # only voxel 1 has signal
        scaled, _, _ = scale_to_percent_signal(data.clone(), [0], verbose=False)
        assert torch.isfinite(scaled).all()
        # Zero-mean voxels should be zero
        assert (scaled[0, :] == 0).all()
        assert (scaled[2, :] == 0).all()

    def test_multi_run(self):
        """Each run should be scaled independently."""
        data = torch.ones(5, 200)
        data[:, :100] *= 500.0  # run 1 mean=500
        data[:, 100:] *= 1000.0  # run 2 mean=1000
        scaled, _, info = scale_to_percent_signal(data.clone(), [0, 100], verbose=False)
        assert info["scale_factors"].shape == (5, 2)
        assert info["mean_per_run"].shape == (5, 2)

    def test_scale_info_structure(self):
        """Check that all expected keys are present in scale_info."""
        data = torch.ones(3, 50) * 100.0
        _, _, info = scale_to_percent_signal(data.clone(), [0], verbose=False)
        assert "n_violations" in info
        assert "n_voxels_with_violations" in info
        assert "violation_voxel_indices" in info
        assert "mean_per_run" in info
        assert "scale_factors" in info

    def test_track_violations_false_matches_stats(self):
        """track_violations=False skips the full mask but reports identical stats.

        The whole-dataset reml path relies on this to avoid a tens-of-GB
        (n_voxels, n_timepoints) bool mask it never reads.
        """
        data = torch.ones(5, 50) * 1.0  # low mean -> guaranteed violations
        data[0, 10] = 1000.0
        data[3, 25] = 5000.0

        scaled_full, mask_full, info_full = scale_to_percent_signal(
            data.clone(), [0, 25], max_scale=200.0, verbose=False, track_violations=True
        )
        scaled_none, mask_none, info_none = scale_to_percent_signal(
            data.clone(), [0, 25], max_scale=200.0, verbose=False, track_violations=False
        )

        assert mask_full is not None and mask_full.shape == (5, 50)
        assert mask_none is None
        assert torch.equal(scaled_full, scaled_none)
        assert info_full["n_violations"] == info_none["n_violations"]
        assert info_full["n_voxels_with_violations"] == info_none["n_voxels_with_violations"]
        assert torch.equal(
            info_full["violation_voxel_indices"], info_none["violation_voxel_indices"]
        )
        # The reported count must match the full mask ground truth.
        assert info_none["n_violations"] == int(mask_full.sum().item())


class TestGaussianBlur3D:
    def test_basic_blur(self):
        """Blurring should reduce spatial variance."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 10, 10, 3)).astype(np.float32)
        blurred = gaussian_blur_3d(
            data,
            fwhm_mm=4.0,
            voxel_sizes=(2.0, 2.0, 2.0),
            device=torch.device("cpu"),
            verbose=False,
        )
        assert blurred.shape == data.shape
        # Blurred data should have lower spatial variance
        for t in range(3):
            assert blurred[:, :, :, t].std() < data[:, :, :, t].std()

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((8, 8, 8, 5)).astype(np.float32)
        blurred = gaussian_blur_3d(
            data,
            fwhm_mm=4.0,
            voxel_sizes=(2.0, 2.0, 2.0),
            device=torch.device("cpu"),
            verbose=False,
        )
        assert blurred.shape == data.shape
        assert blurred.dtype == data.dtype

    def test_wrong_ndim_raises(self):
        data = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 4D data"):
            gaussian_blur_3d(
                data,
                fwhm_mm=4.0,
                voxel_sizes=(2.0, 2.0, 2.0),
                device=torch.device("cpu"),
                verbose=False,
            )

    def test_anisotropic_voxels(self):
        """Should handle anisotropic voxels without error."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((6, 8, 10, 2)).astype(np.float32)
        blurred = gaussian_blur_3d(
            data,
            fwhm_mm=6.0,
            voxel_sizes=(1.0, 1.5, 3.0),
            device=torch.device("cpu"),
            verbose=False,
        )
        assert blurred.shape == data.shape


class TestGenerateSyntheticRuns:
    def test_all_synthetic(self):
        """Generate all synthetic runs (no real data)."""
        result = generate_synthetic_runs(
            first_run_data=None,
            n_runs_total=3,
            run_length=50,
            n_voxels=100,
            verbose=False,
        )
        assert result.shape == (100, 150)
        # Values should be in the expected range
        assert result.min() >= 10.0
        assert result.max() <= 100.0

    def test_with_first_run_data(self):
        """First run is real, rest are synthetic."""
        real_data = torch.ones(20, 50) * 75.0
        result = generate_synthetic_runs(
            first_run_data=real_data,
            n_runs_total=3,
            run_length=50,
            verbose=False,
        )
        assert result.shape == (20, 150)
        # First run should be preserved exactly
        torch.testing.assert_close(result[:, :50], real_data)

    def test_reproducible_with_generator(self):
        gen1 = torch.Generator().manual_seed(123)
        r1 = generate_synthetic_runs(None, 2, 30, n_voxels=10, generator=gen1, verbose=False)
        gen2 = torch.Generator().manual_seed(123)
        r2 = generate_synthetic_runs(None, 2, 30, n_voxels=10, generator=gen2, verbose=False)
        torch.testing.assert_close(r1, r2)

    def test_raises_without_n_voxels(self):
        with pytest.raises(ValueError, match="n_voxels must be provided"):
            generate_synthetic_runs(None, 3, 50, verbose=False)


class TestLoadPerRunNuisanceFiles:
    def test_load_basic(self, tmp_path):
        """Load per-run nuisance files with 2D data."""
        for i in range(1, 4):
            data = np.random.randn(100, 3)
            np.savetxt(tmp_path / f"test_run{i:02d}_PCs.txt", data)

        prefix = str(tmp_path / "test")
        result = load_per_run_nuisance_files(prefix, 3, suffix="_PCs.txt")
        assert len(result) == 3
        for arr in result:
            assert arr is not None
            assert arr.shape == (100, 3)

    def test_missing_file_returns_none(self, tmp_path):
        """Missing files should return None for that run."""
        # Only create run 1
        np.savetxt(tmp_path / "test_run01_PCs.txt", np.random.randn(50, 2))

        result = load_per_run_nuisance_files(str(tmp_path / "test"), 2, suffix="_PCs.txt")
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is None

    def test_empty_file_returns_none(self, tmp_path):
        """Empty files should return None."""
        (tmp_path / "test_run01_PCs.txt").write_text("")
        result = load_per_run_nuisance_files(str(tmp_path / "test"), 1, suffix="_PCs.txt")
        assert result[0] is None

    def test_single_column_reshaped(self, tmp_path):
        """Single-column file should be reshaped to (n, 1)."""
        np.savetxt(tmp_path / "test_run01_PCs.txt", np.random.randn(50))
        result = load_per_run_nuisance_files(str(tmp_path / "test"), 1, suffix="_PCs.txt")
        assert result[0].ndim == 2
        assert result[0].shape[1] == 1


class TestComputePowerSpectrum:
    def test_sine_wave_peak(self):
        """A pure sine wave should have a peak at its frequency."""
        fs = 100.0  # 100 Hz sampling rate
        freq = 10.0  # 10 Hz sine wave
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(2 * np.pi * freq * t)

        freqs, amps = compute_power_spectrum(signal, sampling_rate=fs)
        peak_freq = freqs[np.argmax(amps)]
        assert abs(peak_freq - freq) < 1.5  # Within 1.5 Hz

    def test_dc_component(self):
        """Constant signal should have only DC component."""
        signal = np.ones(100) * 5.0
        freqs, amps = compute_power_spectrum(signal, sampling_rate=10.0)
        # DC component should be dominant
        assert amps[0] > amps[1:].max() * 10

    def test_torch_input(self):
        """Should accept torch tensors."""
        signal = torch.randn(100)
        freqs, amps = compute_power_spectrum(signal, sampling_rate=10.0)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(amps, np.ndarray)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            compute_power_spectrum(np.ones((10, 2)), sampling_rate=1.0)


class TestComputePowerSpectra:
    def test_batch_matches_individual(self):
        """Batch result should match individual calls."""
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((5, 100))

        freqs_batch, amps_batch = compute_power_spectra(signals, sampling_rate=10.0)

        for i in range(5):
            freqs_i, amps_i = compute_power_spectrum(signals[i], sampling_rate=10.0)
            np.testing.assert_array_equal(freqs_batch, freqs_i)
            np.testing.assert_allclose(amps_batch[i], amps_i, atol=1e-10)

    def test_torch_input(self):
        signals = torch.randn(3, 50)
        freqs, amps = compute_power_spectra(signals, sampling_rate=5.0)
        assert amps.shape == (3, 26)  # n_freqs = 50//2 + 1

    def test_custom_axis(self):
        """Should work with non-default axis."""
        signals = np.random.randn(100, 4)  # time along axis 0
        freqs, amps = compute_power_spectra(signals, sampling_rate=10.0, axis=0)
        assert amps.shape == (51, 4)  # 100//2 + 1 = 51


class TestCalcMemoryUsageExtended:
    def test_zero_element(self):
        """Shape with zero dimension should give 0 memory."""
        assert calc_memory_usage((0, 100)) == 0.0

    def test_int16(self):
        """int16 = 2 bytes per element."""
        mem = calc_memory_usage((1000,), dtype=torch.int16)
        assert abs(mem - 2e-6) < 1e-10
