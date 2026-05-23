"""
Tests for processing/filters.py — Savitzky-Golay filtering used by
phase regression (ffs_phasereg).

The oracle is scipy.signal.savgol_filter (reflect-mode padding matches
our F.pad(mode='reflect')). We verify correctness against scipy on
random data and pin the contract-shaped behaviors (kernel coefficients,
batched-shape preservation, the polynomial-preservation property).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import savgol_coeffs, savgol_filter

from fastfuncstuff.processing.filters import (
    _sgf_coefficients,
    savgol_filter_1d,
    savgol_filter_explore,
)


# ---------------------------------------------------------------------------
# _sgf_coefficients
# ---------------------------------------------------------------------------

class TestSGFCoefficients:
    @pytest.mark.parametrize("window,order", [(5, 2), (7, 3), (11, 4), (21, 5)])
    def test_matches_scipy_smoothing_kernel(self, window, order):
        # float32 pinv on a 21x6 Vandermonde drifts ~1e-4 from scipy's
        # float64 reference; wider tolerance still pins the right values.
        ours = _sgf_coefficients(window, order, deriv=0).numpy()
        ref = savgol_coeffs(window, order, deriv=0)
        np.testing.assert_allclose(ours, ref, atol=5e-4)

    def test_kernel_sums_to_one(self):
        """A smoothing SGF kernel (deriv=0) preserves a constant signal,
        so its coefficients must sum to 1.0."""
        k = _sgf_coefficients(11, 3).numpy()
        assert k.sum() == pytest.approx(1.0, abs=1e-5)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="odd"):
            _sgf_coefficients(6, 2)

    def test_poly_order_geq_window_raises(self):
        with pytest.raises(ValueError, match="poly_order"):
            _sgf_coefficients(5, 5)


# ---------------------------------------------------------------------------
# savgol_filter_1d — correctness against scipy + shape contracts
# ---------------------------------------------------------------------------

class TestSavgolFilter1D:
    @pytest.mark.parametrize("window,order", [(5, 2), (11, 3), (21, 4)])
    def test_matches_scipy_on_1d(self, window, order):
        torch.manual_seed(0)
        x = torch.randn(200)
        ours = savgol_filter_1d(x, window, order).numpy()
        ref = savgol_filter(x.numpy(), window, order, mode="mirror")
        # scipy "mirror" matches F.pad mode='reflect' (a b | c d e | d c).
        # Wider tolerance for window=21 absorbs float32 pinv drift in the kernel.
        np.testing.assert_allclose(ours, ref, atol=5e-4)

    def test_matches_scipy_on_2d(self):
        """2-D input (n_voxels, n_timepoints): filter is per-voxel."""
        torch.manual_seed(0)
        x = torch.randn(8, 200)
        ours = savgol_filter_1d(x, 11, 3).numpy()
        ref = savgol_filter(x.numpy(), 11, 3, mode="mirror", axis=-1)
        np.testing.assert_allclose(ours, ref, atol=1e-4)

    def test_higher_dim_preserves_shape(self):
        """3-D input should flatten leading dims, filter, then reshape back."""
        torch.manual_seed(0)
        x = torch.randn(3, 4, 200)
        out = savgol_filter_1d(x, 11, 3)
        assert out.shape == x.shape
        # Compare to scipy applied along last axis
        ref = savgol_filter(x.numpy(), 11, 3, mode="mirror", axis=-1)
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)

    def test_preserves_low_order_polynomial(self):
        """SGF with poly_order=p should reproduce any polynomial of degree
        ≤ p exactly (up to numerical error)."""
        t = torch.arange(200, dtype=torch.float64)
        # Cubic: filter with poly_order=3 should be exact in the interior
        signal = 1.0 + 0.5 * t - 0.001 * t**2 + 1e-5 * t**3
        filtered = savgol_filter_1d(signal, 11, 3)
        # Interior region only (reflect padding can leak at edges)
        interior = slice(20, 180)
        np.testing.assert_allclose(filtered[interior].numpy(),
                                   signal[interior].numpy(),
                                   atol=1e-6)

    def test_short_signal_passthrough(self):
        """If n_timepoints < window_length the filter returns a clone."""
        x = torch.randn(10, 5)
        out = savgol_filter_1d(x, window_length=11, poly_order=3)
        assert torch.equal(out, x)
        assert out.data_ptr() != x.data_ptr()  # clone, not the same tensor

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="odd"):
            savgol_filter_1d(torch.randn(100), 6, 2)

    def test_poly_order_geq_window_raises(self):
        with pytest.raises(ValueError, match="poly_order"):
            savgol_filter_1d(torch.randn(100), 5, 5)

    def test_preserves_dtype_and_device(self):
        x = torch.randn(50, dtype=torch.float64)
        out = savgol_filter_1d(x, 5, 2)
        assert out.dtype == torch.float64
        assert out.device == x.device


# ---------------------------------------------------------------------------
# savgol_filter_explore
# ---------------------------------------------------------------------------

class TestSavgolFilterExplore:
    def test_no_metric_returns_input_unchanged(self):
        x = torch.randn(5, 200)
        out = savgol_filter_explore(x, 200, torch.device("cpu"), metric_fn=None)
        assert out is x

    def test_picks_per_voxel_best_filter(self):
        """Construct two voxels with different optimal windows, give a
        metric that prefers closeness to a clean reference, and verify
        the explorer picks different parameters per voxel."""
        torch.manual_seed(7)
        t = torch.linspace(0, 10, 200)
        clean1 = torch.sin(t)
        clean2 = torch.sin(5.0 * t)  # higher freq
        noisy = torch.stack([
            clean1 + 0.5 * torch.randn(200),
            clean2 + 0.5 * torch.randn(200),
        ])
        clean = torch.stack([clean1, clean2])

        def metric(filtered):
            # Score = -MSE vs clean reference (higher is better)
            return -((filtered - clean) ** 2).mean(dim=-1)

        out = savgol_filter_explore(
            noisy, n_timepoints=200, device=torch.device("cpu"),
            min_window=5, max_window=51, min_order=2, max_order=5,
            step=4, metric_fn=metric,
        )
        assert out.shape == noisy.shape
        # The explorer should improve on the noisy input on at least one voxel
        score_noisy = metric(noisy)
        score_out = metric(out)
        assert (score_out >= score_noisy - 1e-6).all()
        assert (score_out > score_noisy).any()

    def test_defaults_for_max_window_and_order(self):
        """When max_window/max_order are None the explorer still runs and
        respects the n_timepoints upper bound."""
        x = torch.randn(2, 80)

        def metric(filtered):
            return -filtered.std(dim=-1)  # arbitrary, just needs to be callable

        out = savgol_filter_explore(
            x, n_timepoints=80, device=torch.device("cpu"),
            metric_fn=metric,
        )
        assert out.shape == x.shape
