"""Per-run smooth FIR curves (fit_smooth_per_run) against brute-force refits."""

import numpy as np
import pytest
import torch
from scipy.linalg import block_diag
from scipy.stats import gamma

from fastfuncstuff.design.builder import legendre_polynomials
from fastfuncstuff.design.matrices import make_tent_design
from fastfuncstuff.glm.smooth_basis import (
    fit_smooth_basis,
    fit_smooth_per_run,
    roughness_penalty,
)

CPU = torch.device("cpu")
T, RUNS, TR, K = 150, 4, 1.0, 16


def _hrf(t):
    t = np.asarray(t, float)
    return np.where(t >= 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0) / 0.18


def _problem(n_vox=40, amp=1.0, noise=None, seed=0, same_onsets=False):
    rng = np.random.default_rng(seed)
    noise = np.ones(RUNS) if noise is None else noise
    xs = []
    shared = np.sort(rng.uniform(2, T - 20, 14))
    for _ in range(RUNS):
        onsets = shared if same_onsets else np.sort(rng.uniform(2, T - 20, 14))
        xs.append(
            make_tent_design([onsets], 0.0, 15.0, TR, T, n_basis=K, device=CPU).double().numpy()
        )
    poly = legendre_polynomials(T, 2)
    design = np.hstack([np.vstack(xs), block_diag(*[poly] * RUNS)])
    truth = amp * _hrf(np.linspace(0, 15, K))
    y = np.hstack(
        [x @ truth + noise[r] * rng.normal(0, 1, (n_vox, T)) + 50 for r, x in enumerate(xs)]
    )
    run_design = [np.hstack([x, poly]) for x in xs]
    return torch.tensor(y, dtype=torch.float32), torch.tensor(design), run_design, truth


def _proj(d):
    q, _ = np.linalg.qr(d[:, K:])
    x = d[:, :K]
    return x - q @ (q.T @ x), q


def _trace(d):
    x, _ = _proj(d)
    return float((x * x).sum())


STARTS = [r * T for r in range(RUNS)]
PEN = roughness_penalty([K])


def _pooled(y, design, method="reml", lam=None):
    return fit_smooth_basis(y, design.float(), K, PEN, method=method, lam=lam, device=CPU)


def _run_fit(y, run_design, r, method, lam=None):
    return fit_smooth_basis(
        y[:, r * T : (r + 1) * T],
        torch.tensor(run_design[r]).float(),
        K,
        PEN,
        method=method,
        lam=lam,
        device=CPU,
    )


def test_shared_lambda_is_the_pooled_strength_rescaled_to_each_run():
    y, design, run_design, _ = _problem()
    pooled = _pooled(y, design)
    out = fit_smooth_per_run(y, design, K, [PEN], STARTS, pooled.lam, device=CPU)
    traces = np.array([_trace(d) for d in run_design])
    for r in range(RUNS):
        lam_r = pooled.lam.double() * traces.sum() / traces[r]
        ref = _run_fit(y, run_design, r, "fixed", lam_r)
        torch.testing.assert_close(out.betas[:, r], ref.betas, atol=2e-4, rtol=1e-3)
        torch.testing.assert_close(
            out.log10_lambda[:, r], torch.log10(lam_r).float(), atol=1e-4, rtol=0
        )


def test_run_mode_is_each_runs_own_reml_fit():
    y, design, run_design, _ = _problem()
    pooled = _pooled(y, design)
    out = fit_smooth_per_run(y, design, K, [PEN], STARTS, pooled.lam, lambda_mode="run", device=CPU)
    for r in range(RUNS):
        ref = _run_fit(y, run_design, r, "reml")
        torch.testing.assert_close(out.betas[:, r], ref.betas, atol=2e-4, rtol=1e-3)


def _brute_xval(y, run_design, betas_for):
    """COD of fit-run-r-predict-run-j over all pairs; betas_for(r, j) -> (V, K)."""
    y64 = y.double().numpy()
    ss = np.zeros(y.shape[0])
    ss_tot_sq = np.zeros(y.shape[0])
    tot = np.zeros(y.shape[0])
    for j in range(RUNS):
        x, q = _proj(run_design[j])
        yj = y64[:, j * T : (j + 1) * T]
        yj = yj - (yj @ q) @ q.T
        ss_tot_sq += (yj * yj).sum(1)
        tot += yj.sum(1)
        for r in range(RUNS):
            if r != j:
                ss += ((yj - betas_for(r, j) @ x.T) ** 2).sum(1)
    ss_tot = (RUNS - 1) * (ss_tot_sq - tot**2 / (RUNS * T))
    return 1 - ss / ss_tot


def _ols(y, run_design, r):
    x, q = _proj(run_design[r])
    yr = y.double().numpy()[:, r * T : (r + 1) * T]
    return np.linalg.lstsq(x, (yr - (yr @ q) @ q.T).T, rcond=None)[0].T


def test_xval_all_lambda_matches_brute_force_pairs():
    y, design, run_design, _ = _problem()
    pooled = _pooled(y, design)
    out = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, pooled.lam, xval=True, xval_lambda="all", device=CPU
    )
    b = out.betas.double().numpy()
    ref = _brute_xval(y, run_design, lambda r, j: b[:, r])
    np.testing.assert_allclose(out.xval_r2.numpy(), ref, atol=2e-4)
    ref_ols = _brute_xval(y, run_design, lambda r, j: _ols(y, run_design, r))
    np.testing.assert_allclose(out.xval_r2_ols.numpy(), ref_ols, atol=2e-4)
    assert out.xval_lambda_used == "all"


def test_xval_fold_lambda_never_sees_the_scored_run():
    y, design, run_design, _ = _problem()
    pooled = _pooled(y, design)
    out = fit_smooth_per_run(y, design, K, [PEN], STARTS, pooled.lam, xval=True, device=CPU)
    traces = np.array([_trace(d) for d in run_design])
    fold_betas = {}
    for j in range(RUNS):
        keep = [r for r in range(RUNS) if r != j]
        rows = np.concatenate([np.arange(r * T, (r + 1) * T) for r in keep])
        d_m = np.hstack(
            [
                np.vstack([run_design[r][:, :K] for r in keep]),
                block_diag(*[run_design[r][:, K:] for r in keep]),
            ]
        )
        lam_m = _pooled(y[:, rows], torch.tensor(d_m)).lam.double()
        for r in keep:
            lam_r = lam_m * traces[keep].sum() / traces[r]
            fold_betas[r, j] = _run_fit(y, run_design, r, "fixed", lam_r).betas.double().numpy()
    ref = _brute_xval(y, run_design, lambda r, j: fold_betas[r, j])
    np.testing.assert_allclose(out.xval_r2.numpy(), ref, atol=5e-4)
    assert out.xval_lambda_used == "fold"


def test_loro_pooled_rule_falls_back_to_all_run_lambda():
    y, design, _, _ = _problem(n_vox=5)
    out = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, torch.ones(5), rule="loro", xval=True, device=CPU
    )
    assert out.xval_lambda_used == "all"


def test_arma_runs_are_prewhitened_with_the_voxels_pair():
    from fastfuncstuff.glm.arma import build_arma11_covariance

    y, design, run_design, _ = _problem(n_vox=6)
    arma = torch.tensor([[0.4, 0.1]] * 6, dtype=torch.float32)
    out = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, torch.full((6,), 0.5), arma=arma, device=CPU
    )
    chol = torch.linalg.cholesky(build_arma11_covariance(0.4, 0.1, T, CPU, torch.float64))
    whiten = [
        torch.linalg.solve_triangular(chol, torch.tensor(d), upper=False).numpy()
        for d in run_design
    ]
    traces = np.array([_trace(d) for d in whiten])
    r = 2
    y_w = torch.linalg.solve_triangular(chol, y[:, r * T : (r + 1) * T].double().T, upper=False).T
    ref = fit_smooth_basis(
        y_w.float(),
        torch.tensor(whiten[r]).float(),
        K,
        PEN,
        method="fixed",
        lam=0.5 * traces.sum() / traces[r],
        device=CPU,
    )
    torch.testing.assert_close(out.betas[:, r], ref.betas, atol=2e-4, rtol=1e-3)


def test_standard_error_matches_the_spread_over_noise_replicates():
    y, design, _, _ = _problem(n_vox=4000, seed=3)
    out = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, torch.full((4000,), 0.05), se=True, device=CPU
    )
    empirical = out.betas.std(dim=0)
    predicted = out.se.mean(dim=0)
    torch.testing.assert_close(predicted, empirical, rtol=0.08, atol=0)


def test_per_run_lambda_fakes_a_trend_when_noise_rises_and_global_does_not():
    """The bug the shared modes exist to prevent: per-run REML shrinks noisier
    runs harder, so equal true curves come out declining across runs.  Same
    onsets in every run, so only the noise differs."""
    noise = np.linspace(1.0, 3.0, RUNS)
    y, design, _, truth = _problem(n_vox=3000, amp=0.6, noise=noise, seed=5, same_onsets=True)
    pooled = _pooled(y, design)
    peak = int(np.argmax(truth))

    def drop(mode):
        b = (
            fit_smooth_per_run(
                y,
                design,
                K,
                [PEN],
                STARTS,
                pooled.lam,
                lambda_mode=mode,
                device=CPU,
            )
            .betas[:, :, peak]
            .double()
        )
        d = b[:, -1] - b[:, 0]
        return float(d.mean()), float(d.mean() / (d.std() / np.sqrt(d.numel())))

    run_mean, run_z = drop("run")
    shared_mean, _ = drop("shared")
    _, global_z = drop("global")
    assert run_z < -6.0
    assert abs(global_z) < 3.0
    # Per-voxel pooled lambda is chosen from the same noise: small, not zero.
    assert abs(shared_mean) < 0.5 * abs(run_mean)


GRID = np.linspace(-3.0, 5.0, 9)


def _signal_and_noise():
    sig, design, _, _ = _problem(n_vox=60, amp=1.0, seed=2)
    noise, _, _, _ = _problem(n_vox=200, amp=0.0, seed=2)  # same onsets, no response
    return torch.cat([sig, noise]), design


def test_global_lambda_maximizes_the_signal_voxels_median_heldout_r2():
    from fastfuncstuff.glm.smooth_basis import choose_global_lambda, fit_smooth_selected

    y, design = _signal_and_noise()

    def xval(rule, lam=None):
        return fit_smooth_selected(
            y, design, K, [PEN], STARTS, rule=rule, lam=lam, xval=True, device=CPU
        ).xval_r2

    signal = (xval("fixed", 1e-6) > 0.05) | (xval("reml") > 0.05)
    curve = torch.stack([xval("fixed", 10.0**g) for g in GRID], dim=1)
    med = curve[signal].median(dim=0).values.numpy()
    out = choose_global_lambda(y, design, K, PEN, STARTS, log10_grid=GRID, device=CPU)
    assert out.n_signal == int(signal.sum()) and not out.fallback
    assert 50 <= out.n_signal <= 75  # the 60 responsive voxels, not the 200 silent ones
    np.testing.assert_allclose(out.median_curve, med, atol=2e-4)
    assert abs(np.log10(out.lam) - GRID[med.argmax()]) <= GRID[1] - GRID[0]


def test_per_run_global_mode_fits_every_run_with_the_chosen_lambda():
    from fastfuncstuff.glm.smooth_basis import choose_global_lambda

    y, design = _signal_and_noise()
    chosen = choose_global_lambda(y, design, K, PEN, STARTS, device=CPU).lam
    out = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, torch.ones(y.shape[0]), lambda_mode="global", device=CPU
    )
    ref = fit_smooth_per_run(
        y, design, K, [PEN], STARTS, torch.full((y.shape[0],), chosen), device=CPU
    )
    torch.testing.assert_close(out.betas, ref.betas)


def test_global_lambda_falls_back_to_every_voxel_without_signal():
    from fastfuncstuff.glm.smooth_basis import choose_global_lambda

    y, design, _, _ = _problem(n_vox=50, amp=0.0, seed=4)
    out = choose_global_lambda(y, design, K, PEN, STARTS, device=CPU)
    assert out.fallback and out.n_signal == 50


def test_rejects_unknown_modes_and_single_runs():
    y, design, _, _ = _problem(n_vox=2)
    with pytest.raises(ValueError, match="lambda_mode"):
        fit_smooth_per_run(y, design, K, [PEN], STARTS, torch.ones(2), lambda_mode="x")
    with pytest.raises(ValueError, match="two runs"):
        fit_smooth_per_run(y[:, :T], design[:T], K, [PEN], [0], torch.ones(2))


def test_per_run_scores_held_out_runs_on_score_data_when_given():
    y, design, run_design, _ = _problem()
    pooled = _pooled(y, design)
    rough = y + torch.randn_like(y)  # a different series to score against
    out = fit_smooth_per_run(
        y,
        design,
        K,
        [PEN],
        STARTS,
        pooled.lam,
        xval=True,
        xval_lambda="all",
        score_data=rough,
        device=CPU,
    )
    b = out.betas.double().numpy()
    ref = _brute_xval(rough, run_design, lambda r, j: b[:, r])
    np.testing.assert_allclose(out.xval_r2.numpy(), ref, atol=2e-4)
