"""Tests for partition-axis inter-echo shift estimation/correction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.shiftcorr import (
    ShiftEstimate,
    apply_shift,
    estimate_pair_shift,
    estimate_shifts,
    search_bounds,
    te_regression,
)


def _phantom(
    n_t: int = 4,
    shape: tuple[int, int, int] = (24, 20, 20),
    seed: int = 0,
    noise: float = 0.01,
) -> torch.Tensor:
    """Smooth structured volumes with a bit of noise — (T, nz, ny, nx)."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(
        *[torch.linspace(-1, 1, s) for s in shape],
        indexing="ij",
    )
    blob = torch.exp(-(zz**2 + yy**2 + xx**2) * 3.0) * (1.0 + 0.4 * torch.sin(4.0 * zz))
    vols = blob.unsqueeze(0).repeat(n_t, 1, 1, 1)
    vols = vols + noise * torch.randn(vols.shape, generator=g)
    return vols


def test_pair_shift_recovers_known_subvoxel_shift():
    """The estimate is the CORRECTION: the shift that undoes the displacement."""
    ref = _phantom(n_t=3)
    displacement = np.array([-0.35, 0.80, -1.25])
    mov = apply_shift(ref, displacement, axis=2)  # axis 2 == tensor dim 1 (nz)

    est, corr = estimate_pair_shift(ref, mov, axis=2, ordering="unknown", max_shift=3.0)

    assert np.allclose(est, -displacement, atol=0.01), f"{est} vs {-displacement}"
    assert (corr > 0.99).all()


def test_apply_shift_is_invertible():
    """A push by d then by -d must return the original (sinc-exact, no smoothing)."""
    # Noiseless: white noise sits at Nyquist, where the replicate pad's own edge
    # discontinuity dominates the round trip. Real image content is band-limited.
    vols = _phantom(n_t=2, noise=0.0)
    d = np.array([0.4, -0.6])
    back = apply_shift(apply_shift(vols, d, axis=2), -d, axis=2)
    # Edge partitions genuinely lose information (content shifted in from the
    # replicated pad cannot come back), so the identity only holds in the interior.
    inner = slice(6, -6)
    assert torch.allclose(back[:, inner], vols[:, inner], atol=1e-3)


def test_ordering_constrains_sign():
    lo, hi = search_bounds("ascending", 5.0)
    assert (lo, hi) == (-5.0, 0.0)
    lo, hi = search_bounds("descending", 5.0)
    assert (lo, hi) == (0.0, 5.0)
    with pytest.raises(ValueError):
        search_bounds("sideways", 5.0)


def test_te_regression_recovers_slope_and_drops_intercept():
    tes = np.array([7.61, 21.71, 35.81])
    slope = np.array([-0.02, -0.015])
    intercept = np.array([0.5, -0.3])
    cum = slope[:, None] * tes[None, :] + intercept[:, None]

    m, b = te_regression(cum, tes)

    assert np.allclose(m, slope)
    assert np.allclose(b, intercept)


def test_estimate_shifts_end_to_end_te_line():
    """Three echoes shifted by m·TE per timepoint: the fit must recover m."""
    tes = np.array([7.61, 21.71, 35.81])
    base = _phantom(n_t=3, seed=1)
    m_true = np.array([-0.03, -0.05, -0.02])  # voxels per ms of TE, per timepoint
    echoes = [apply_shift(base, m_true * te, axis=2) for te in tes]

    est = estimate_shifts(echoes, axis=2, tes=tes, ordering="unknown", max_shift=3.0, verb=0)

    assert isinstance(est, ShiftEstimate)
    # Estimates are corrections, so the recovered slope is minus the applied one.
    assert np.allclose(est.slope, -m_true, atol=2e-3)
    # The correction goes through the origin, so echo 1 is corrected too.
    assert np.allclose(est.applied, -m_true[:, None] * tes[None, :], atol=0.03)
    assert np.abs(est.applied[:, 0]).min() > 0.1
    assert np.allclose(est.frequency_drift_hz, m_true * 1e3, atol=2.0)


def test_estimate_shifts_without_tes_uses_cumulative():
    base = _phantom(n_t=2, seed=2)
    echoes = [base, apply_shift(base, np.array([-0.4, -0.4]), axis=2)]

    est = estimate_shifts(echoes, axis=2, tes=None, ordering="unknown", max_shift=3.0, verb=0)

    assert est.slope is None
    assert np.allclose(est.applied[:, 0], 0.0)  # echo 1 is the reference
    assert np.allclose(est.applied[:, 1], 0.4, atol=0.01)


def test_complex_shift_preserves_phase_structure():
    """Complex input shifts magnitude and phase together, with no global phase roll."""
    # Strictly positive magnitude: a negative real value carries its own pi of
    # phase and would make the constant-phase assertion below meaningless.
    base = _phantom(n_t=1, seed=3).abs() + 1.0
    comp = (base * torch.exp(1j * 0.3 * torch.ones_like(base))).to(torch.complex64)
    out = apply_shift(comp, np.array([1.0]), axis=2)
    inner = slice(4, -4)
    # A whole-voxel shift of a constant-phase volume must leave the phase constant.
    assert torch.allclose(
        torch.angle(out[:, inner]), torch.full_like(base[:, inner], 0.3), atol=1e-3
    )


def test_fold_into_matrices_matches_the_fourier_shift():
    """The folded matrix must move voxels the same way apply_shift does.

    This is the sign-convention gate for -me_3depi: a flipped sign here would
    DOUBLE the inter-echo shift instead of removing it, and nothing downstream
    would catch it.
    """
    from fastfuncstuff.processing.ffs_moco import MocoConfig, resample_timeseries
    from fastfuncstuff.processing.shiftcorr import fold_shift_into_matrices

    vols = _phantom(n_t=2, noise=0.0)
    d = np.array([0.45, -0.30])
    axis = 2  # NIfTI z == the (nt, nz, ny, nx) tensor's dim 1

    identity = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    folded = fold_shift_into_matrices(identity, d, axis)
    config = MocoConfig(device=torch.device("cpu"), final_interp="wsinc5", verb=0)
    via_matrix, _ = resample_timeseries(vols, folded, config, torch.device("cpu"))
    via_fourier = apply_shift(vols, d, axis)

    inner = slice(6, -6)
    assert torch.allclose(via_matrix[:, inner], via_fourier[:, inner], atol=5e-3)
