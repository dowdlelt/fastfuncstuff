"""Heat smoothing along a mesh, and the edge-based smoothness estimator."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, _step_fwhm_ratio, surface_fwhm


def _sheet(n: int, s: float) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * s, jj.ravel() * s, np.zeros(n * n)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v, f.astype(np.int64)


def _half_max_width(row: np.ndarray, s: float) -> float:
    c = int(np.argmax(row))
    r = row[c:]
    half = row[c] / 2
    j = int(np.argmax(r < half))
    return 2 * s * (j - 1 + (r[j - 1] - half) / (r[j - 1] - r[j]))


def test_step_ratio_matches_the_hankel_transform():
    assert _step_fwhm_ratio(8) == pytest.approx(0.8936, abs=2e-3)
    assert _step_fwhm_ratio(32) == pytest.approx(0.974, abs=2e-3)


def test_an_impulse_spreads_to_the_target_fwhm_and_keeps_its_mass():
    n, s = 121, 0.5
    v, f = _sheet(n, s)
    hs = HeatSmoother(v, f, 4.0)
    imp = np.zeros(n * n)
    imp[(n // 2) * n + n // 2] = 1.0
    k = hs(imp)
    assert _half_max_width(k.reshape(n, n)[n // 2], s) == pytest.approx(4.0, rel=0.04)
    area = vertex_areas(v, f)
    assert (k * area).sum() == pytest.approx(area[(n // 2) * n + n // 2], rel=1e-9)


def test_the_estimator_reads_back_the_smoothing():
    n, s = 121, 0.5
    v, f = _sheet(n, s)
    inner = (np.abs(v[:, 0] - 30) < 22) & (np.abs(v[:, 1] - 30) < 22)
    noise = np.random.default_rng(1).normal(size=(n * n, 150))
    for fwhm in (3.0, 5.0):
        g, per = surface_fwhm(HeatSmoother(v, f, fwhm, n_steps=16)(noise), v, f, inner)
        assert g == pytest.approx(fwhm, rel=0.06)
        assert np.nanmedian(per[inner]) == pytest.approx(g, rel=0.03)


def _folded(n: int = 81, s: float = 0.5, fold: float = 20.0):
    """A sheet folded into a U: the far bank sits 1 mm above the near one."""
    v, f = _sheet(n, s)
    x = v[:, 0].copy()
    far = x > fold
    v[far, 0] = 2 * fold - x[far]
    v[far, 2] = 1.0
    return v, f, x, far


def test_nothing_crosses_a_sulcus():
    # The banks are 1 mm apart in space but ~30 mm apart along the cortex: a 4 mm
    # volume blur would mix them, a surface blur must not.
    v, f, x, far = _folded()
    out = HeatSmoother(v, f, 4.0)(np.where(far, 0.0, 100.0))
    near_mid = (~far) & (x > 5) & (x < 6)  # directly under far-bank vertices
    assert out[near_mid].min() > 99.99


def test_the_mask_edge_is_a_wall():
    v, f, x, far = _folded()
    rng = np.random.default_rng(0)
    data = rng.normal(size=len(v))
    mask = np.abs(x - 10.0) >= 0.3  # a cut across the near bank
    out = HeatSmoother(v, f, 4.0, mask=mask)(data)
    np.testing.assert_array_equal(out[~mask], data[~mask])
    area = vertex_areas(v, f)
    isolated = mask & (x < 10)  # the cut leaves this strip on its own: its mass stays
    assert (out[isolated] * area[isolated]).sum() == pytest.approx(
        (data[isolated] * area[isolated]).sum(), abs=1e-6
    )
    assert out[isolated].std() < 0.5 * data[isolated].std()  # and it was smoothed
