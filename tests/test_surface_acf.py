"""Surface ACF estimation, blurring to a smoothness, and SurfClustSim."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.stats.clustsim import ACF, acf_rfunc
from fastfuncstuff.stats.surface_clustsim import (
    MixedNoise,
    _basis_acf,
    max_cluster_areas,
    surface_clustsim,
)
from fastfuncstuff.surface.acf import blur_to_fwhm, detrend, surface_acf
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, surface_fwhm


def _sheet(n: int, s: float):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * s, jj.ravel() * s, np.zeros(n * n)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v, f.astype(np.int64)


def _white(v, f, n, seed):
    # Continuum white noise: variance per vertex inversely proportional to its area.
    area = vertex_areas(v, f)
    return np.random.default_rng(seed).normal(size=(len(v), n)) / np.sqrt(area)[:, None]


@pytest.fixture(scope="module")
def sheet():
    return _sheet(141, 0.6)


def test_acf_of_smoothed_noise_is_the_kernels_own(sheet):
    v, f = sheet
    field = HeatSmoother(v, f, 4.0)(_white(v, f, 80, 0))
    inner = (np.abs(v[:, 0] - 42) < 34) & (np.abs(v[:, 1] - 42) < 34)
    est = surface_acf(field, v, f, inner, n_centres=600)
    r = np.array([1.8, 3.6, 5.4])
    truth = _basis_acf(4.0, 16, (0.0, *r))[1:]  # normalised to r = 0
    got = np.interp(r, est.r, est.curve)
    np.testing.assert_allclose(got, truth, atol=0.04)


def test_mixed_noise_carries_the_requested_tail(sheet):
    v, f = sheet
    acf = ACF(0.5, 2.5, 6.0)
    noise = MixedNoise(v, f, acf)
    assert noise.fit_rms < 0.01
    inner = (np.abs(v[:, 0] - 42) < 30) & (np.abs(v[:, 1] - 42) < 30)  # away from the walls
    est = surface_acf(noise.sample(64, np.random.default_rng(3)), v, f, inner, n_centres=600)
    r = np.array([2.0, 6.0, 10.0, 15.0])
    np.testing.assert_allclose(np.interp(r, est.r, est.curve), acf_rfunc(r, acf), atol=0.035)


def test_blur_to_fwhm_lands_on_the_target_and_does_nothing_when_already_there(sheet):
    v, f = sheet
    inner = (np.abs(v[:, 0] - 42) < 34) & (np.abs(v[:, 1] - 42) < 34)
    base = HeatSmoother(v, f, 3.0)(_white(v, f, 60, 1))
    hs, kernel, achieved = blur_to_fwhm(base, v, f, 6.0, inner)
    assert hs is not None and 0 < kernel < 6.0
    assert achieved == pytest.approx(6.0, rel=0.03)
    # the returned smoother does what the search measured, on all the data
    assert surface_fwhm(hs(base), v, f, inner)[0] == pytest.approx(6.0, rel=0.04)
    none, k0, start = blur_to_fwhm(base, v, f, 2.0, inner)
    assert none is None and k0 == 0.0 and start > 2.0


def test_detrend_removes_per_run_polynomials():
    t = np.linspace(-1, 1, 50)
    runs = np.r_[3 + 2 * t + t**2, -1 + 0.5 * t**3]
    x = np.tile(runs, (4, 1))
    np.testing.assert_allclose(detrend(x, [50, 50], degree=3), 0.0, atol=1e-9)


def test_cluster_areas_by_sidedness():
    v, f = _sheet(30, 1.0)
    area = vertex_areas(v, f)
    from fastfuncstuff.surface.smooth import mesh_edges

    edges = mesh_edges(f)
    z = np.zeros(len(v))
    x, y = v[:, 0], v[:, 1]
    pos = (x >= 5) & (x <= 9) & (y >= 5) & (y <= 9)  # 5 x 5 vertices
    neg = (x >= 10) & (x <= 12) & (y >= 5) & (y <= 9)  # touches pos along x = 9 -> 10
    z[pos], z[neg] = 5.0, -5.0
    thr = np.array([3.0])
    one = max_cluster_areas(z[:, None], edges, area, thr, "1-sided")[0, 0]
    two = max_cluster_areas(z[:, None], edges, area, thr, "2-sided")[0, 0]
    bi = max_cluster_areas(z[:, None], edges, area, thr, "bi-sided")[0, 0]
    assert one == pytest.approx(area[pos].sum())
    assert two == pytest.approx(area[pos | neg].sum())  # opposite signs may join
    assert bi == pytest.approx(max(area[pos].sum(), area[neg].sum()))


def test_clustsim_table_is_its_own_null_quantile_and_is_cached(tmp_path):
    v, f = _sheet(61, 1.0)
    acf = ACF(0.6, 2.0, 4.0)
    kw = dict(niter=400, pthr=(0.01, 0.001), sideds=("2-sided",), batch=200, verb=0)
    res = surface_clustsim(v, f, acf, cache_dir=tmp_path, **kw)
    table = res.table("2-sided", athr=(0.10, 0.05))
    assert table[0, 1] == pytest.approx(np.quantile(res.max_areas["2-sided"][:, 0], 0.95), rel=0.15)
    assert table[0, 1] >= table[0, 0] and table[0, 1] >= table[1, 1]  # monotone both ways
    again = surface_clustsim(v, f, acf, cache_dir=tmp_path, **kw)
    assert again.cached and np.array_equal(again.max_areas["2-sided"], res.max_areas["2-sided"])


@pytest.mark.slow
def test_clustsim_table_holds_its_false_positive_rate():
    # Thresholds from one set of null fields applied to a fresh set: at alpha 0.05 about
    # 5% of fresh fields hold a cluster that large. A jittered mesh, so areas are
    # continuous (a regular grid quantises them and the 95th percentile jumps between
    # whole mm^2), and enough iterations that the tail is not Monte-Carlo noise.
    v, f = _sheet(61, 1.0)
    v[:, :2] += np.random.default_rng(5).uniform(-0.25, 0.25, (len(v), 2))
    acf = ACF(0.6, 2.0, 4.0)
    kw = dict(niter=3000, pthr=(0.01,), sideds=("2-sided",), batch=500, cache_dir=False, verb=0)
    thr = surface_clustsim(v, f, acf, **kw).table("2-sided", athr=(0.05,))[0, 0]
    fresh = surface_clustsim(v, f, acf, seed=99, **kw).max_areas["2-sided"][:, 0]
    assert 0.035 < float((fresh >= thr).mean()) < 0.065
