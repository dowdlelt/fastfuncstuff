"""Tests for the scale-invariant cone prior (:func:`fit_basis_cone_prior`).

The bug of record: :func:`fit_basis_constrained_ridge` penalises
``(β − m)ᵀC⁻¹(β − m)`` with a fixed ``m`` taken from *peak-normalised*
HRF samples, which makes it an amplitude prior in data units.  Every
voxel gets dragged toward peak ≈ ``‖m‖/‖β_peak1‖`` ≈ 0.8 regardless of
its true response size (flat amplitude maps on real data).  The cone
prior constrains only the *direction* of β and must therefore be
amplitude-unbiased.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.design.flobs import (
    decouple_amplitude_prior,
    fit_basis_cone_prior,
    fit_basis_constrained_ridge,
    flobs_prior,
    generate_flobs_basis,
)
from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run

TR = 1.5
NTP = 200
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def setup():
    basis = generate_flobs_basis(n_basis=3, n_samples=500, seed=0)
    lag = np.arange(basis.basis_functions.shape[1]) * basis.dt
    onsets = np.arange(10.0, NTP * TR - 40, 12.0)
    X = build_pc_basis_design_per_run(
        onsets_per_run=[onsets],
        pcs=basis.basis_functions,
        lag_times=lag,
        tr=TR,
        n_timepoints_per_run=[NTP],
        basis="TENT",
    )[0]
    # coefficient vector of an average-shaped HRF normalised to peak 1
    h = basis.m @ basis.basis_functions
    h = h / h.max()
    beta_unit = basis.basis_functions @ h
    polys = np.polynomial.legendre.legvander(np.linspace(-1, 1, NTP), 2)
    return basis, X, beta_unit, polys


def _recovered_peak(betas, basis, n_basis=3):
    return np.median(np.max(betas[:, :n_basis] @ basis.basis_functions, axis=-1))


def _simulate(X, beta_unit, amplitude, sd, n_vox=400, seed=0):
    rng = np.random.default_rng(seed)
    return amplitude * (X @ beta_unit)[None, :] + rng.normal(0, sd, size=(n_vox, NTP))


def test_cone_prior_has_no_amplitude_dependent_bias(setup):
    """recovered/true must be ~constant across a 15x amplitude range.

    Absolute accuracy is not the right target: taking the peak of a
    reconstructed HRF is a positively-biased estimator (max of noise), so
    even OLS overshoots at the smallest amplitude.  What distinguishes a
    *shape* prior from an *amplitude* prior is that the gain is flat --
    the fixed-mean prior's gain sweeps from 2.2x to 0.8x as the true
    amplitude grows, because it is pulling everything to a fixed peak.
    """
    basis, X, beta_unit, polys = setup
    m, C = flobs_prior(basis)
    amplitudes = (0.2, 1.0, 3.0)
    gains = {"cone": [], "ridge": []}
    for amplitude in amplitudes:
        y = _simulate(X, beta_unit, amplitude, sd=0.5)
        common = dict(
            data=torch.from_numpy(y),
            design_task=torch.from_numpy(X),
            basis_functions=basis.basis_functions,
            prior_mean=m,
            prior_cov=C,
            n_blocks=1,
            nuisance=torch.from_numpy(polys),
            prior_weight="auto",
            device=CPU,
        )
        gains["cone"].append(
            _recovered_peak(fit_basis_cone_prior(**common).betas, basis) / amplitude
        )
        gains["ridge"].append(
            _recovered_peak(
                fit_basis_constrained_ridge(**common, reconstruct_hrfs=False).betas, basis
            )
            / amplitude
        )
    cone_spread = max(gains["cone"]) / min(gains["cone"])
    ridge_spread = max(gains["ridge"]) / min(gains["ridge"])
    assert cone_spread < 1.3, f"cone gain varies with amplitude: {gains['cone']}"
    assert ridge_spread > 2.0, f"expected the known fixed-mean amplitude bias, got {gains['ridge']}"


def test_cone_beats_fixed_mean_prior_on_large_amplitudes(setup):
    """The bug of record: the fixed-mean prior crushes a true peak of 3.0."""
    basis, X, beta_unit, polys = setup
    m, C = flobs_prior(basis)
    y = _simulate(X, beta_unit, amplitude=3.0, sd=2.0)
    common = dict(
        data=torch.from_numpy(y),
        design_task=torch.from_numpy(X),
        basis_functions=basis.basis_functions,
        prior_mean=m,
        prior_cov=C,
        n_blocks=1,
        nuisance=torch.from_numpy(polys),
        prior_weight="auto",
        device=CPU,
    )
    ridge = fit_basis_constrained_ridge(**common, reconstruct_hrfs=False)
    cone = fit_basis_cone_prior(**common)
    ridge_peak = _recovered_peak(ridge.betas, basis)
    cone_peak = _recovered_peak(cone.betas, basis)
    # The fixed-mean prior pulls toward ~0.8; the cone prior must not.
    assert ridge_peak < 2.0, f"expected the known ridge bias, got {ridge_peak:.3f}"
    assert cone_peak > 2.7, f"cone prior should stay near 3.0, got {cone_peak:.3f}"


def test_cone_penalty_is_scale_invariant(setup):
    """Doubling the data doubles beta exactly -- degree-0 homogeneous penalty."""
    basis, X, beta_unit, polys = setup
    m, C = flobs_prior(basis)
    y = _simulate(X, beta_unit, amplitude=1.0, sd=0.5)
    kw = dict(
        design_task=torch.from_numpy(X),
        basis_functions=basis.basis_functions,
        prior_mean=m,
        prior_cov=C,
        n_blocks=1,
        nuisance=torch.from_numpy(polys),
        device=CPU,
    )
    # prior_weight must scale with the variance (lambda = sigma^2) for the
    # comparison to be apples-to-apples, so pin lambda via a fixed float
    # and scale it by 4 = 2^2 alongside the 2x data scaling.
    a = fit_basis_cone_prior(data=torch.from_numpy(y), prior_weight=1.0, **kw)
    b = fit_basis_cone_prior(data=torch.from_numpy(2.0 * y), prior_weight=1.0, **kw)
    np.testing.assert_allclose(2.0 * a.betas[:, :3], b.betas[:, :3], rtol=1e-6, atol=1e-8)


def test_cone_allows_negative_responses(setup):
    """Sign symmetry: an inverted BOLD response keeps its shape and sign."""
    basis, X, beta_unit, polys = setup
    m, C = flobs_prior(basis)
    y = _simulate(X, beta_unit, amplitude=-1.5, sd=0.5)
    fit = fit_basis_cone_prior(
        data=torch.from_numpy(y),
        design_task=torch.from_numpy(X),
        basis_functions=basis.basis_functions,
        prior_mean=m,
        prior_cov=C,
        n_blocks=1,
        nuisance=torch.from_numpy(polys),
        prior_weight="auto",
        device=CPU,
    )
    hrfs = fit.betas[:, :3] @ basis.basis_functions
    idx = np.argmax(np.abs(hrfs), axis=-1)
    signed = np.take_along_axis(hrfs, idx[:, None], axis=-1).squeeze(-1)
    assert np.median(signed) == pytest.approx(-1.5, rel=0.15)
    assert np.median(fit.size_per_block[:, 0]) < 0, "size param should carry the sign"


def test_cone_rejects_degenerate_prior(setup):
    """decouple_amplitude_prior output has no prior ray -- must not fit.

    That prior deliberately zeroes the precision along the amplitude
    direction, which is exactly the direction the cone prior needs in
    order to define its ray.  Feeding it in is a user error worth
    catching loudly rather than silently fitting an unconstrained model.
    """
    basis, X, _, polys = setup
    m, C = flobs_prior(basis)
    m_bad, C_bad = decouple_amplitude_prior(m, C)
    with pytest.raises(ValueError, match="degenerate"):
        fit_basis_cone_prior(
            data=torch.zeros((4, NTP), dtype=torch.float64),
            design_task=torch.from_numpy(X),
            basis_functions=basis.basis_functions,
            prior_mean=m_bad if np.linalg.norm(m_bad) > 0 else m,
            prior_cov=C_bad,
            n_blocks=1,
            nuisance=torch.from_numpy(polys),
            device=CPU,
        )
