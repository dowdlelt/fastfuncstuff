"""MEMA (stats/group_mema.py).

The reference is a line-for-line port of 3dMEMA.R's rmaB (MoM, REML Fisher
scoring, collect with Hartung-Knapp) using explicit N x N projection
matrices, one voxel at a time — independent of the batched trace identities
the module uses.  Parity with 3dMEMA itself (R) was measured 2026-11-12:
~4e-8 on one-group, two-group, covariates, and -no_HKtest, with
reml="shortcut".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.group_mema import fit_mema, mema_design, run_mema, variance_from_t
from fastfuncstuff.stats.group_ttest import TtestDesign


def _rmab(y, v, x, knha=True, shortcut=True):
    """3dMEMA.R rmaB with lapMod=0, resOut=0: returns (b, z, tau2)."""
    n, p = x.shape

    def collect(w):
        W = np.diag(w)
        vb = np.linalg.inv(x.T @ W @ x)
        R = vb @ x.T @ W
        P = W - W @ x @ R
        if knha:
            vb = max(y @ P @ y / (n - p), 0) * vb
        se = np.sqrt(np.diag(vb))
        b = R @ y
        return b, np.where(se > 1e-8, b / np.where(se > 1e-8, se, 1), 0), P

    W0 = np.diag(1 / v)
    P0 = W0 - W0 @ x @ np.linalg.inv(x.T @ W0 @ x) @ x.T @ W0
    qe = y @ P0 @ y
    tau2 = max(0.0, (qe - (n - p)) / np.trace(P0))
    b, z, _ = collect(1 / (v + tau2))
    if shortcut and not np.any(np.abs(z) > 1.3):
        return b, z, tau2
    t2 = max(0.0, np.var(y, ddof=1) - v.mean())
    change, it = 1000.0, 0
    while abs(change) > 0.001:
        it += 1
        if it > 200:
            return b, z, tau2
        W = np.diag(1 / (v + t2))
        P = W - W @ x @ np.linalg.inv(x.T @ W @ x) @ x.T @ W
        py = P @ y
        adj = (py @ py - np.trace(P)) / np.trace(P @ P)
        while t2 + adj < 0 and adj != 0:
            adj /= 2
            if abs(adj) < 1e-300:
                adj = 0.0
        t2 += adj
        change = abs(adj) / (abs(t2) + 1e-8)
    b, z, _ = collect(1 / (v + t2))
    return b, z, t2


def _data(seed, n=12, n_vox=40, two=False, cov=False):
    rng = np.random.default_rng(seed)
    v = 0.2 + rng.gamma(2, 0.4, size=(n_vox, n))
    tau = rng.uniform(0, 1.2, size=(n_vox, 1))
    mu = rng.normal(0, 1, size=(n_vox, 1))
    y = mu + tau * rng.standard_normal((n_vox, n)) + np.sqrt(v) * rng.standard_normal((n_vox, n))
    c = rng.normal(size=(n, 1)) if cov else None
    x = mema_design(n - 5 if two else n, 5 if two else 0, None if c is None else c - c.mean(0))
    return y, v, x


@pytest.mark.parametrize("two,cov,knha", [(False, False, True), (True, False, True),
                                          (False, True, True), (True, True, False)])  # fmt: skip
def test_batched_fit_matches_per_voxel_rmab(two, cov, knha):
    y, v, x = _data(1, two=two, cov=cov)
    fit = fit_mema(torch.tensor(y), torch.tensor(v), x, knha=knha, reml="shortcut")
    for i in range(y.shape[0]):
        b, z, t2 = _rmab(y[i], v[i], x, knha=knha)
        np.testing.assert_allclose(fit.b[i].numpy(), b, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(fit.z[i].numpy(), z, rtol=1e-8, atol=1e-10)
        assert fit.tau2[i].item() == pytest.approx(t2, rel=1e-7, abs=1e-12)


def test_reml_always_equals_reml_where_the_shortcut_used_it():
    y, v, x = _data(2)
    a = fit_mema(torch.tensor(y), torch.tensor(v), x, reml="always")
    s = fit_mema(torch.tensor(y), torch.tensor(v), x, reml="shortcut")
    both = s.reml
    assert both.any() and (~both).any()
    torch.testing.assert_close(a.z[both], s.z[both])
    for i in range(y.shape[0]):
        b, z, t2 = _rmab(y[i], v[i], x, shortcut=False)
        assert a.tau2[i].item() == pytest.approx(t2, rel=1e-7, abs=1e-12)


def test_homogeneous_data_gives_zero_tau_and_fixed_effects_z():
    rng = np.random.default_rng(3)
    n = 15
    v = np.full((5, n), 0.5)
    y = 0.3 + np.sqrt(v) * rng.standard_normal((5, n)) * 1e-3  # far less spread than v says
    fit = fit_mema(torch.tensor(y), torch.tensor(v), np.ones((n, 1)), knha=False)
    assert torch.all(fit.tau2 == 0)
    # Fixed effects: b = mean, se = sqrt(v / n).
    np.testing.assert_allclose(fit.z[:, 0], y.mean(1) / np.sqrt(0.5 / n), rtol=1e-10)


def test_no_data_subjects_get_no_weight():
    b = torch.tensor([[1.0, 0.0, 2.0]])
    t = torch.tensor([[2.0, 0.0, 0.0]])
    v = variance_from_t(b, t)
    assert v[0, 0] == 0.25 and v[0, 1] == 1e8 and v[0, 2] == 1e8


def test_two_set_bricks_report_a_minus_b():
    rng = np.random.default_rng(4)
    ba = torch.tensor(rng.normal(1.0, 0.5, (8, 30)))
    bb = torch.tensor(rng.normal(0.0, 0.5, (7, 30)))
    ta, tb = ba / 0.3, bb / 0.3
    bricks, _ = run_mema(TtestDesign(8, 7, "A", "B"), ba, ta, bb, tb)
    lab = {b.label: b for b in bricks}
    assert list(lab)[:6] == ["A:b", "A:t", "B:b", "B:t", "A-B:b", "A-B:t"]
    torch.testing.assert_close(lab["A-B:b"].data, lab["A:b"].data - lab["B:b"].data)
    assert lab["A-B:t"].stat == (3, (13.0,))
    assert float(lab["A-B:b"].data.mean()) > 0.5
