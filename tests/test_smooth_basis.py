"""Roughness-penalized FIR/TENT with per-voxel REML/GCV smoothing."""

import numpy as np
import pytest
import torch
from scipy.linalg import block_diag
from scipy.stats import gamma

from fastfuncstuff.design.builder import build_per_run_task_designs, legendre_polynomials
from fastfuncstuff.glm.smooth_basis import fit_smooth_basis, roughness_penalty

CPU = torch.device("cpu")
T, RUNS, TR = 300, 2, 1.0
RNG = np.random.default_rng(0)
BASE = [
    np.sort(RNG.choice(np.arange(5, 280), 30, replace=False)).astype(float) for _ in range(RUNS)
]


def _hrf(t):
    t = np.asarray(t, float)
    return np.where(t >= 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0) / 0.18


def _problem(phase, knot_dt=1.0, n_vox=60, amp=1.0):
    onsets = [[b + phase for b in BASE]]
    n_knots = int(round(15 / knot_dt)) + 1
    res = build_per_run_task_designs(
        onsets,
        [T] * RUNS,
        TR,
        basis="TENT",
        fir_window_s=[(0.0, 15.0)],
        tent_n_basis=n_knots,
        device=CPU,
    )
    x = torch.cat(res.per_run, 0).double().numpy()
    design = np.hstack([x, block_diag(*[legendre_polynomials(T, 2)] * RUNS)])
    t = np.arange(T) * TR
    signal = np.concatenate([sum(_hrf(t - o) for o in onsets[0][r]) for r in range(RUNS)])
    y = amp * signal[None, :] + RNG.normal(0, 1.0, (n_vox, T * RUNS)) + 100.0
    truth = amp * _hrf(np.linspace(0, 15, n_knots))
    return torch.tensor(y, dtype=torch.float32), torch.tensor(design), n_knots, truth, res


def test_tiny_fixed_lambda_is_ols_on_an_identifiable_design():
    y, design, k, _, res = _problem(0.0)
    fit = fit_smooth_basis(
        y, design, k, roughness_penalty(res.n_basis_per_condition), method="fixed", lam=1e-9
    )
    ols = np.linalg.lstsq(design.numpy(), y.double().numpy().T, rcond=None)[0][:k].T
    np.testing.assert_allclose(fit.betas.numpy(), ols, atol=2e-3)
    assert fit.edf.mean() == pytest.approx(k, abs=1e-3)


@pytest.mark.parametrize("method", ["reml", "gcv"])
def test_penalty_rescues_the_singular_mid_tr_design(method):
    y, design, k, truth, res = _problem(0.5)
    fit = fit_smooth_basis(
        y, design, k, roughness_penalty(res.n_basis_per_condition), method=method
    )
    assert np.isfinite(fit.betas.numpy()).all()
    ols = np.linalg.lstsq(design.numpy(), y.double().numpy().T, rcond=None)[0][:k].T

    def rmse(b):
        return np.sqrt(np.mean((b - truth) ** 2))

    assert rmse(fit.betas.numpy()) < 0.5 * rmse(ols)


def test_reml_smooths_noise_to_the_unpenalized_line():
    y, design, k, _, res = _problem(0.0, amp=0.0)
    fit = fit_smooth_basis(y, design, k, roughness_penalty(res.n_basis_per_condition))
    # Only the penalty's null space (constant + slope) survives.
    assert float(fit.edf.median()) < 3.0


def test_finer_than_tr_knots_are_solvable_when_timing_supports_them():
    y, design, k, truth, res = _problem(RNG.choice([0.0, 0.5], BASE[0].size), knot_dt=0.5)
    fit = fit_smooth_basis(y, design, k, roughness_penalty(res.n_basis_per_condition))
    assert np.sqrt(np.mean((fit.betas.numpy() - truth) ** 2)) < 0.15


def test_zero_edge_penalty_differences_through_the_pinned_ends():
    p = roughness_penalty([3], order=2, zero_edges=True)
    d = np.diff(np.eye(5), 2, axis=0)[:, 1:-1]
    np.testing.assert_allclose(p, d.T @ d)
    assert np.linalg.matrix_rank(p) == 3  # pinned ends leave no free line


def _loro_ols(y, design, k):
    from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits

    splits = generate_cv_splits(n_runs=RUNS, strategy=1)
    out = compute_xval_r2(
        y,
        design.float(),
        [0, T],
        list(range(k)),
        list(range(k, design.shape[1])),
        splits,
        device=CPU,
        verbose=False,
    )
    return out["r2"].numpy()


def test_loro_at_tiny_lambda_matches_the_ols_cross_validation():
    from fastfuncstuff.glm.smooth_basis import loro_r2_smooth

    y, design, k, _, res = _problem(0.0, amp=2.0)
    pen = roughness_penalty(res.n_basis_per_condition)
    smooth = loro_r2_smooth(y, design, k, pen, [0, T], method="fixed", lam=1e-9).numpy()
    np.testing.assert_allclose(smooth, _loro_ols(y, design, k), atol=2e-3)


def test_smoothing_raises_held_out_r2_where_timing_is_poor():
    from fastfuncstuff.glm.smooth_basis import loro_r2_smooth

    y, design, k, _, res = _problem(RNG.uniform(0.4, 0.6, BASE[0].size), amp=2.0)
    pen = roughness_penalty(res.n_basis_per_condition)
    smooth = loro_r2_smooth(y, design, k, pen, [0, T]).numpy()
    assert np.median(smooth) > np.median(_loro_ols(y, design, k))
