"""Tests for fastfuncstuff.decomposition.postprocess and tools (uncovered functions)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.postprocess import (
    auto_mask_from_data,
    component_condition_spectral_correlations,
    preprocess_design_for_correlation,
    save_corr_heatmap,
    save_depth_lag_plot,
    save_score_heatmap,
    save_scree_plot,
)
from fastfuncstuff.decomposition.tools import (
    apply_high_pass_fft,
    apply_polort_projection,
    estimate_ica_component_count,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# postprocess: preprocess_design_for_correlation
# ---------------------------------------------------------------------------


class TestPreprocessDesignForCorrelation:
    def test_basic_shape(self):
        """Output shape matches input shape."""
        design = torch.randn(20, 3, device=DEVICE)
        out = preprocess_design_for_correlation(
            design, tr=2.0, polort=2, high_pass_hz=None, device=DEVICE
        )
        assert out.shape == design.shape

    def test_empty_input(self):
        """Empty tensor returns empty."""
        design = torch.empty(0, device=DEVICE)
        out = preprocess_design_for_correlation(
            design, tr=2.0, polort=2, high_pass_hz=0.01, device=DEVICE
        )
        assert out.numel() == 0

    def test_polort_removes_mean(self):
        """With polort>=0, output columns should be roughly zero-mean."""
        design = torch.randn(30, 2, device=DEVICE) + 5.0
        out = preprocess_design_for_correlation(
            design, tr=1.0, polort=0, high_pass_hz=None, device=DEVICE
        )
        col_means = out.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-4)

    def test_high_pass_applied(self):
        """With high_pass_hz, low-frequency content should be attenuated."""
        t = torch.linspace(0, 10, 100, device=DEVICE)
        # Very slow sine (below cutoff) + fast sine
        slow = torch.sin(2 * np.pi * 0.005 * t)
        fast = torch.sin(2 * np.pi * 0.2 * t)
        design = (slow + fast).unsqueeze(1)
        out = preprocess_design_for_correlation(
            design, tr=0.1, polort=-1, high_pass_hz=0.05, device=DEVICE
        )
        # Slow component should be attenuated
        assert out.abs().mean() < design.abs().mean()


# ---------------------------------------------------------------------------
# postprocess: component_condition_spectral_correlations
# ---------------------------------------------------------------------------


class TestSpectralCorrelations:
    def test_output_shape(self):
        n_time, n_comp, n_cond = 40, 3, 2
        mixing = torch.randn(n_time, n_comp, device=DEVICE)
        design = torch.randn(n_time, n_cond, device=DEVICE)
        out = component_condition_spectral_correlations(mixing, design)
        assert out.shape == (n_comp, n_cond)

    def test_identical_signals_high_corr(self):
        """A component identical to a condition should have high spectral corr."""
        n_time = 64
        sig = torch.randn(n_time, 1, device=DEVICE)
        out = component_condition_spectral_correlations(sig, sig)
        assert out.shape == (1, 1)
        assert out[0, 0] > 0.9

    def test_very_short_timeseries(self):
        """With only 1 timepoint, rfft drops DC -> 0 freq bins -> zeros."""
        mixing = torch.randn(1, 2, device=DEVICE)
        design = torch.randn(1, 3, device=DEVICE)
        out = component_condition_spectral_correlations(mixing, design)
        assert out.shape == (2, 3)
        # Should be all zeros since no frequencies after dropping DC
        assert np.allclose(out, 0.0)


# ---------------------------------------------------------------------------
# postprocess: auto_mask_from_data
# ---------------------------------------------------------------------------


class TestAutoMaskFromData:
    def test_basic_mask(self):
        """High-intensity voxels should be in mask, low ones out."""
        data = np.zeros((5, 5, 5, 10), dtype=np.float32)
        data[1:4, 1:4, 1:4, :] = 1000.0  # "brain"
        mask = auto_mask_from_data(data, verbose=False)
        assert mask.shape == (5, 5, 5)
        assert mask.dtype == bool
        assert mask[2, 2, 2]  # center brain voxel
        assert not mask[0, 0, 0]  # outside

    def test_all_zeros(self):
        """All-zero data produces empty mask."""
        data = np.zeros((3, 3, 3, 5), dtype=np.float32)
        mask = auto_mask_from_data(data)
        assert mask.sum() == 0

    def test_verbose_runs(self, capsys):
        data = np.random.rand(4, 4, 4, 10).astype(np.float32) + 1.0
        mask = auto_mask_from_data(data, verbose=True)
        captured = capsys.readouterr()
        assert "Auto-mask" in captured.out
        assert mask.sum() > 0


# ---------------------------------------------------------------------------
# postprocess: plotting functions
# ---------------------------------------------------------------------------


class TestPlottingFunctions:
    def test_save_scree_plot(self, tmp_path):
        evr = np.array([0.4, 0.3, 0.15, 0.1, 0.05], dtype=np.float32)
        out_png = tmp_path / "scree.png"
        save_scree_plot(evr, out_png, title="Test Scree")
        assert out_png.exists()
        assert out_png.stat().st_size > 100

    def test_save_depth_lag_plot(self, tmp_path):
        lag_matrix = np.random.randn(4, 3).astype(np.float32)
        depth_labels = [1, 2, 3]
        out_png = tmp_path / "depth_lag.png"
        save_depth_lag_plot(lag_matrix, depth_labels, out_png, title="Lag Test")
        assert out_png.exists()

    def test_save_depth_lag_plot_empty(self, tmp_path):
        """Empty data should not create a file."""
        out_png = tmp_path / "empty_lag.png"
        save_depth_lag_plot(np.array([]), [], out_png, title="Empty")
        assert not out_png.exists()

    def test_save_corr_heatmap(self, tmp_path):
        corr = np.random.randn(5, 3).astype(np.float32)
        labels = ["reg1", "reg2", "reg3"]
        out_png = tmp_path / "corr.png"
        save_corr_heatmap(corr, labels, out_png, title="Corr Test")
        assert out_png.exists()

    def test_save_corr_heatmap_empty(self, tmp_path):
        out_png = tmp_path / "empty_corr.png"
        save_corr_heatmap(np.array([]), [], out_png, title="Empty")
        assert not out_png.exists()

    def test_save_score_heatmap(self, tmp_path):
        scores = np.random.rand(4, 2).astype(np.float32)
        labels = ["mask_a", "mask_b"]
        out_png = tmp_path / "scores.png"
        save_score_heatmap(scores, labels, out_png, title="Score Test", cmap="hot")
        assert out_png.exists()

    def test_save_score_heatmap_empty(self, tmp_path):
        out_png = tmp_path / "empty_scores.png"
        save_score_heatmap(np.array([]), [], out_png, title="Empty", cmap="hot")
        assert not out_png.exists()

    def test_save_depth_lag_with_nan(self, tmp_path):
        """Handles NaN values in lag matrix."""
        lag_matrix = np.array([[1.0, np.nan], [np.nan, -0.5]], dtype=np.float32)
        out_png = tmp_path / "nan_lag.png"
        save_depth_lag_plot(lag_matrix, [1, 2], out_png, title="NaN Lag")
        assert out_png.exists()


# ---------------------------------------------------------------------------
# tools: apply_polort_projection
# ---------------------------------------------------------------------------


class TestApplyPolortProjection:
    def test_removes_linear_trend(self):
        """polort=1 should remove a linear ramp from data."""
        n_vox, n_t = 5, 50
        ramp = torch.linspace(0, 1, n_t, device=DEVICE).unsqueeze(0).expand(n_vox, -1)
        data = ramp.clone()
        out = apply_polort_projection(data, polort=1, device=DEVICE)
        assert out.shape == (n_vox, n_t)
        assert out.abs().max() < 0.01

    def test_negative_polort_noop(self):
        """polort=-1 should return data unchanged."""
        data = torch.randn(3, 20, device=DEVICE)
        original = data.clone()
        out = apply_polort_projection(data, polort=-1, device=DEVICE)
        assert torch.allclose(out, original)

    def test_polort_none_noop(self):
        data = torch.randn(3, 20, device=DEVICE)
        original = data.clone()
        out = apply_polort_projection(data, polort=None, device=DEVICE)
        assert torch.allclose(out, original)

    def test_multi_run_block_diagonal(self):
        """With run_starts, each run gets its own polynomial removal."""
        n_vox, n_t = 4, 40
        # Two runs: first has a ramp, second has a different ramp
        data = torch.zeros(n_vox, n_t, device=DEVICE)
        data[:, :20] = torch.linspace(0, 1, 20, device=DEVICE).unsqueeze(0)
        data[:, 20:] = torch.linspace(1, 0, 20, device=DEVICE).unsqueeze(0)
        out = apply_polort_projection(data, polort=1, device=DEVICE, run_starts=[0, 20])
        assert out.abs().max() < 0.01

    def test_single_run_start_same_as_none(self):
        """run_starts=[0] should behave like run_starts=None."""
        data = torch.randn(3, 30, device=DEVICE)
        out1 = apply_polort_projection(data.clone(), polort=2, device=DEVICE, run_starts=None)
        out2 = apply_polort_projection(data.clone(), polort=2, device=DEVICE, run_starts=[0])
        assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# tools: apply_high_pass_fft
# ---------------------------------------------------------------------------


class TestApplyHighPassFFT:
    def test_removes_low_freq(self):
        """Low frequency sinusoid should be attenuated."""
        n_vox, n_t = 3, 200
        tr = 1.0
        t = torch.arange(n_t, dtype=torch.float32, device=DEVICE)
        low_freq = torch.sin(2 * np.pi * 0.005 * t)  # 0.005 Hz
        data = low_freq.unsqueeze(0).expand(n_vox, -1).clone()
        out = apply_high_pass_fft(data, tr=tr, high_pass_hz=0.02)
        # Power should be greatly reduced
        assert out.abs().mean() < data.abs().mean() * 0.3

    def test_preserves_high_freq(self):
        """High frequency sinusoid should be mostly preserved."""
        n_vox, n_t = 2, 200
        tr = 1.0
        t = torch.arange(n_t, dtype=torch.float32, device=DEVICE)
        high_freq = torch.sin(2 * np.pi * 0.2 * t)
        data = high_freq.unsqueeze(0).expand(n_vox, -1).clone()
        out = apply_high_pass_fft(data, tr=tr, high_pass_hz=0.01)
        # Should retain most power
        ratio = out.abs().mean() / data.abs().mean()
        assert ratio > 0.8

    def test_none_hz_noop(self):
        data = torch.randn(2, 30, device=DEVICE)
        original = data.clone()
        out = apply_high_pass_fft(data, tr=1.0, high_pass_hz=None)
        assert torch.allclose(out, original)

    def test_zero_hz_noop(self):
        data = torch.randn(2, 30, device=DEVICE)
        original = data.clone()
        out = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.0)
        assert torch.allclose(out, original)

    def test_multi_run(self):
        """Multi-run filtering should not crash and should produce valid output."""
        n_vox, n_t = 3, 60
        data = torch.randn(n_vox, n_t, device=DEVICE)
        out = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.01, run_starts=[0, 30])
        assert out.shape == (n_vox, n_t)
        assert torch.isfinite(out).all()

    def test_brick_wall_transition(self):
        """transition_width=0 should use brick wall filter."""
        data = torch.randn(2, 100, device=DEVICE)
        out = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.05, transition_width=0.0)
        assert out.shape == data.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# tools: estimate_ica_component_count
# ---------------------------------------------------------------------------


class TestEstimateICAComponentCount:
    @pytest.fixture
    def fake_data(self):
        """Small fake data: 100 voxels x 30 timepoints."""
        rng = np.random.default_rng(123)
        # Create data with some structure (a few signal components + noise)
        n_vox, n_time = 100, 30
        signal = rng.standard_normal((n_vox, 3)) @ rng.standard_normal((3, n_time))
        noise = rng.standard_normal((n_vox, n_time)) * 0.5
        data = torch.tensor(signal + noise, dtype=torch.float32, device=DEVICE)
        return data

    def test_fixed_int(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method=5,
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
        )
        assert k == 5
        assert info["mode"] == "fixed_int"

    def test_float_variance(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method=0.9,
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
        )
        assert 1 <= k <= 30
        assert info["mode"] == "pca_variance_fraction"

    def test_laplace_mode(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method="laplace",
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
        )
        assert 1 <= k <= 20
        assert info["mode"] == "laplace_mp"

    def test_erank_mode(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method="erank",
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
        )
        assert k >= 1
        assert info["mode"] == "effective_rank"

    def test_mp_mode(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method="mp",
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
        )
        assert k >= 1
        assert "mp" in info["mode"]

    def test_invalid_mode_raises(self, fake_data):
        with pytest.raises(ValueError, match="Unsupported"):
            estimate_ica_component_count(
                fake_data,
                method="nonsense",
                max_auto_components=20,
                auto_min_components=2,
                auto_var_threshold=0.95,
                use_mp_prior=False,
                device=DEVICE,
                verbose=False,
            )

    def test_invalid_float_raises(self, fake_data):
        with pytest.raises(ValueError, match="Float"):
            estimate_ica_component_count(
                fake_data,
                method=1.5,
                max_auto_components=20,
                auto_min_components=2,
                auto_var_threshold=0.95,
                use_mp_prior=False,
                device=DEVICE,
                verbose=False,
            )

    def test_verbose_output(self, fake_data, capsys):
        estimate_ica_component_count(
            fake_data,
            method=5,
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Component estimation" in captured.out

    def test_with_n_eff(self, fake_data):
        """n_eff parameter should not crash."""
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method="laplace",
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
            n_eff=50,
        )
        assert k >= 1

    def test_capture_ppca_trace(self, fake_data):
        k, diag, info = estimate_ica_component_count(
            fake_data,
            method="laplace",
            max_auto_components=20,
            auto_min_components=2,
            auto_var_threshold=0.95,
            use_mp_prior=False,
            device=DEVICE,
            verbose=False,
            capture_ppca_trace=True,
        )
        assert "ppca_trace" in diag
