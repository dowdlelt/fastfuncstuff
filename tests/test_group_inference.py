"""Nulls for every group statistic (stats/group_inference.py and the nulls it builds).

Each null is checked against refitting explicitly relabeled data with the
estimator that produced the observed brick — the shortcut (GEMM for Welch,
batched (iteration x voxel) fits for MEMA) is what could silently differ.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.group_inference import brick_nulls
from fastfuncstuff.stats.group_mema import (
    MemaNull,
    fit_mema,
    fit_mema_unequal,
    mema_design,
    run_mema,
    variance_from_t,
)
from fastfuncstuff.stats.group_ttest import (
    TtestDesign,
    TToZ2D,
    WelchNull,
    run_ttest,
    t_to_z_np,
    welch,
)


def _maps(seed=0, na=9, nb=7, v=30, dt=torch.float64):
    rng = np.random.default_rng(seed)
    ba = torch.tensor(rng.normal(0.4, 1.0, (na, v)), dtype=dt)
    bb = torch.tensor(rng.normal(0.0, 1.6, (nb, v)), dtype=dt)
    ta = ba / torch.tensor(rng.uniform(0.3, 1.0, (na, v)), dtype=dt)
    tb = bb / torch.tensor(rng.uniform(0.3, 1.0, (nb, v)), dtype=dt)
    return ba, ta, bb, tb


DESIGNS = {
    "one": (TtestDesign(9), False),
    "two": (TtestDesign(9, 7), True),
    "welch": (TtestDesign(9, 7, unpooled=True), True),
    "paired": (TtestDesign(7, 7, paired=True), True),
    "cov": (TtestDesign(9, cov_a=np.arange(9.0)[:, None], cov_names=("c",)), False),
}


@pytest.mark.parametrize("model", ["ols", "mema"])
@pytest.mark.parametrize("name", list(DESIGNS))
def test_brick_indices_name_the_right_bricks(model, name):
    des, two = DESIGNS[name]
    ba, ta, bb, tb = _maps()
    if name == "paired":
        ba, ta = ba[:7], ta[:7]
    yb, tbb = (bb, tb) if two else (None, None)
    if model == "ols":
        bricks = run_ttest(des, ba, yb)
    else:
        bricks = run_mema(des, ba, ta, yb, tbb)[0]
    nulls = brick_nulls(des, model=model, ya=ba, yb=yb, ta=ta, tb=tbb, n_iter=4)
    assert sum(n.main for n in nulls) == 1
    for n in nulls:
        est, t = bricks[n.brick], bricks[n.brick + 1]
        assert t.stat is not None and t.stat[0] in (3, 5)
        assert est.label.replace(":b", "").startswith(n.label.split("_")[0])
    main = next(n for n in nulls if n.main)
    assert main.brick == (4 if (model == "mema" and two and name != "paired") else 0)


def test_welch_null_equals_welch_on_flipped_residuals():
    ba, _, bb, _ = _maps(1)
    null = WelchNull(ba.float(), bb.float(), n_iter=5, seed=2)
    t, dof = null.t_maps(0, 5)
    ra, rb = ba - ba.mean(0), bb - bb.mean(0)
    for i in range(5):
        s = torch.tensor(null._signs[i], dtype=torch.float64)
        _, tr, dr = welch(ra * s[:9, None], rb * s[9:, None])
        np.testing.assert_allclose(t[i], tr, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(dof[i], dr, rtol=2e-4)


def test_two_d_t_to_z_table():
    toz = TToZ2D(6, 14, torch.device("cpu"))
    t = torch.linspace(-9, 9, 401).repeat(5)
    dof = torch.linspace(6, 14, 5).repeat_interleave(401)
    np.testing.assert_allclose(toz(t, dof).numpy(), t_to_z_np(t.numpy(), dof.numpy()), atol=3e-4)


@pytest.mark.parametrize("unequal", [False, True])
def test_mema_null_equals_refitting_flipped_residuals(unequal):
    ba, ta, bb, tb = _maps(3)
    y = torch.cat([ba, bb])
    v = torch.cat([variance_from_t(ba, ta), variance_from_t(bb, tb)])
    if unequal:
        null = MemaNull(y, v, None, 2, 14.0, n_iter=3, n_a=9, seed=4)
    else:
        x = mema_design(9, 7, None)
        null = MemaNull(y, v, x, -np.eye(2)[1], 14.0, n_iter=3, seed=4)
    got = null.stat(0, 3)
    for i in range(3):
        s = torch.tensor(null._signs[i])
        ys = (null._r * s).contiguous()
        if unequal:
            f = fit_mema_unequal(ys[:, :9], null._v[:, :9], ys[:, 9:], null._v[:, 9:])
            want = f.z[2]
        else:
            f = fit_mema(ys, null._v, mema_design(9, 7, None))
            se = torch.einsum("p,vpq,q->v", torch.tensor([0.0, -1.0], dtype=torch.float64), f.vb,
                              torch.tensor([0.0, -1.0], dtype=torch.float64)).sqrt()  # fmt: skip
            want = -f.b[:, 1] / se
        torch.testing.assert_close(got[i], want, rtol=1e-9, atol=1e-9)
