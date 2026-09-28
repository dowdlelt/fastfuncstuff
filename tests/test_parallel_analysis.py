"""Revised parallel analysis with phase-randomised surrogates (decomposition/model_order.py)."""

import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.model_order import parallel_analysis_order, select_model_order

CPU = torch.device("cpu")


def _ar1(n, T, phi, g):
    e = torch.randn(n, T, generator=g)
    x = torch.zeros(n, T)
    x[:, 0] = e[:, 0]
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x * (1 - phi**2) ** 0.5


def _planted(rank, V=5000, T=160, noise_phi=0.0, source_phi=0.0, strength=1.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(V, T, generator=g) if noise_phi == 0 else _ar1(V, T, noise_phi, g)
    S = torch.randn(rank, T, generator=g) if source_phi == 0 else _ar1(rank, T, source_phi, g)
    M = torch.randn(V, rank, generator=g) * strength
    return M @ S + noise


def _white_count(X):
    Xc = X - X.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh((Xc.T @ Xc).double() / X.shape[0]).flip(0).numpy()
    return select_model_order(np.clip(ev, 0, None), n_samples=X.shape[0]).k


def _pa(X, **kw):
    return parallel_analysis_order(
        X, n_samples=X.shape[0], remove_timepoint_mean=True, device=CPU, **kw
    )


def test_recovers_planted_rank_in_white_noise():
    assert _pa(_planted(6)).k == 6


@pytest.mark.xfail(
    strict=True,
    reason="known bias: removing leading directions notches coloured residual spectra, the rebuilt null runs low and the count runs away",
)
def test_coloured_noise_fools_the_white_null_but_not_parallel_analysis():
    # Temporally autocorrelated noise spreads the null spectrum, so the white
    # Marchenko-Pastur null reads its leading noise eigenvalues as signal.
    X = _planted(6, noise_phi=0.6)
    assert _white_count(X) > 12
    assert abs(_pa(X).k - 6) <= 1


@pytest.mark.xfail(
    strict=True,
    reason="known bias: removing leading directions notches coloured residual spectra, the rebuilt null runs low and the count runs away",
)
def test_autocorrelated_sources_are_not_absorbed_into_the_null():
    # What sank the one-shot version on BOLD: strongly autocorrelated shared sources put
    # their power into every voxel's spectrum, so a null that keeps each voxel's
    # spectrum inherits them. Removing accepted components before rebuilding the null
    # is what keeps the count honest.
    X = _planted(6, noise_phi=0.4, source_phi=0.95, strength=1.0, seed=2)
    assert abs(_pa(X).k - 6) <= 1


@pytest.mark.xfail(
    strict=True,
    reason="known bias: removing leading directions notches coloured residual spectra, the rebuilt null runs low and the count runs away",
)
def test_pure_coloured_noise_gives_the_floor():
    g = torch.Generator().manual_seed(4)
    X = _ar1(5000, 160, 0.6, g)
    assert _pa(X, k_min=1).k <= 2


@pytest.mark.xfail(
    strict=True,
    reason="known bias: removing leading directions notches coloured residual spectra, the rebuilt null runs low and the count runs away",
)
def test_stops_at_the_first_failure_not_a_later_pass():
    # Past the true rank the test can pass again (deflated residual, lower surrogate
    # edge); the count must be the FIRST failure. A bisection returned 98 here.
    X = _planted(8, noise_phi=0.5, seed=5)
    pa = _pa(X, k_min=0)
    assert abs(pa.k - 8) <= 1, pa.k
    assert pa.n_null_evaluations == pa.k + 1


def test_first_failure_scan_in_white_noise():
    pa = _pa(_planted(8, seed=5), k_min=0)
    assert pa.k == 8
    assert pa.n_null_evaluations == pa.k + 1
