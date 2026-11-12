"""Group t-tests (stats/group_ttest.py).

Parametric fits are checked against scipy and an explicit lstsq; parity with
3dttest++ itself (labels, dof, values to 1e-7 on all of one/two-sample,
Welch, paired, covariates DIFF/SAME, -toz) was measured on 2026-11-12 with
AFNI on PATH and is not repeated here.  The null shortcut is checked against
refitting the flipped / permuted data from scratch, which is the bug it
could hide.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from fastfuncstuff.stats.group_ttest import (
    OlsNull,
    TtestDesign,
    TToZ,
    main_test,
    ols_contrasts,
    run_ttest,
    t_to_z_np,
)

CPU = torch.device("cpu")


def _data(seed=0, na=9, nb=7, v=50):
    rng = np.random.default_rng(seed)
    ya = rng.standard_normal((na, v)) + 0.4
    yb = rng.standard_normal((nb, v)) * 1.5
    return torch.tensor(ya), torch.tensor(yb)


def _by_label(bricks):
    return {b.label: b for b in bricks}


def test_one_and_two_sample_match_scipy():
    ya, yb = _data()
    one = _by_label(run_ttest(TtestDesign(9, label_a="A"), ya))
    np.testing.assert_allclose(one["A_Tstat"].data, stats.ttest_1samp(ya, 0).statistic, rtol=1e-10)
    assert one["A_Tstat"].stat == (3, (8.0,))
    two = _by_label(run_ttest(TtestDesign(9, 7, "A", "B"), ya, yb))
    np.testing.assert_allclose(two["A-B_Tstat"].data, stats.ttest_ind(ya, yb).statistic, rtol=1e-10)
    assert two["A-B_Tstat"].stat == (3, (14.0,))


def test_welch_and_paired_match_scipy():
    ya, yb = _data()
    w = _by_label(run_ttest(TtestDesign(9, 7, "A", "B", unpooled=True), ya, yb))
    ref = stats.ttest_ind(ya, yb, equal_var=False)
    # Welch is reported as z with the same two-sided p.
    z_from_p = stats.norm.isf(ref.pvalue / 2) * np.sign(ref.statistic)
    np.testing.assert_allclose(w["A-B_Zscr"].data, z_from_p, rtol=1e-8)
    ya8 = ya[:7]
    p = _by_label(run_ttest(TtestDesign(7, 7, "A", "B", paired=True), ya8, yb))
    np.testing.assert_allclose(p["A-B_Tstat"].data, stats.ttest_rel(ya8, yb).statistic, rtol=1e-10)


def test_two_sample_covariates_are_separate_slopes_per_set():
    ya, yb = _data(1)
    rng = np.random.default_rng(2)
    ca, cb = rng.standard_normal((9, 2)), rng.standard_normal((7, 2))
    des = TtestDesign(9, 7, "A", "B", cov_a=ca, cov_b=cb, cov_names=("age", "iq"))
    got = _by_label(run_ttest(des, ya, yb))
    assert [b.label for b in run_ttest(des, ya, yb)][:6] == [
        "A-B_mean", "A-B_Tstat", "A-B_age", "A-B_age_Tstat", "A-B_iq", "A-B_iq_Tstat",
    ]  # fmt: skip
    # Explicit: each set centred on its own mean, own intercept and slopes.
    ca_c, cb_c = ca - ca.mean(0), cb - cb.mean(0)
    x = np.zeros((16, 6))
    x[:9, 0], x[:9, 1:3], x[9:, 3], x[9:, 4:6] = 1, ca_c, 1, cb_c
    y = np.vstack([ya.numpy(), yb.numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    np.testing.assert_allclose(got["A-B_mean"].data, beta[0] - beta[3], atol=1e-10)
    np.testing.assert_allclose(got["A-B_age"].data, beta[1] - beta[4], atol=1e-10)
    assert got["A-B_Tstat"].stat == (3, (10.0,))
    assert got["A_Tstat"].stat == (3, (6.0,))


def test_center_same_uses_the_pooled_mean():
    rng = np.random.default_rng(3)
    ca, cb = rng.standard_normal((5, 1)) + 3, rng.standard_normal((4, 1))
    des = TtestDesign(5, 4, cov_a=ca, cov_b=cb, cov_names=("c",), center="SAME")
    a, b = des.centered_covariates()
    mu = np.vstack([ca, cb]).mean(0)
    np.testing.assert_allclose(a, ca - mu)
    np.testing.assert_allclose(b, cb - mu)


def test_refusals():
    with pytest.raises(ValueError, match="pooled"):
        TtestDesign(4, 4, unpooled=True, cov_a=np.zeros((4, 1)), cov_b=np.zeros((4, 1)),
                    cov_names=("c",))  # fmt: skip
    with pytest.raises(ValueError, match="equal set sizes"):
        TtestDesign(4, 5, paired=True)
    with pytest.raises(ValueError, match="degrees of freedom"):
        ols_contrasts(torch.zeros(2, 3, dtype=torch.float64), np.ones((2, 2)), np.eye(2))


def test_ttoz_table_matches_scipy():
    toz = TToZ(11, CPU)
    t = torch.linspace(-12, 12, 2001)
    np.testing.assert_allclose(toz(t).numpy(), t_to_z_np(t.numpy(), 11), atol=2e-5)


@pytest.mark.parametrize("kind", ["one", "two_perm", "paired", "two_cov"])
def test_null_shortcut_equals_refitting_the_flipped_data(kind):
    ya, yb = _data(4)
    ya, yb = ya.float(), yb.float()
    rng = np.random.default_rng(5)
    if kind == "one":
        des = TtestDesign(9)
    elif kind == "paired":
        des, yb = TtestDesign(7, 7, paired=True), yb[:7]
        ya = ya[:7]
    elif kind == "two_cov":
        des = TtestDesign(9, 7, cov_a=rng.standard_normal((9, 1)), cov_b=rng.standard_normal((7, 1)),
                          cov_names=("c",))  # fmt: skip
    else:
        des = TtestDesign(9, 7)
    y, x, c = main_test(des, ya, None if not des.two_sets else yb)
    null = OlsNull(y, x, c, n_iter=6, seed=1, permute=(kind == "two_perm"))
    got = null.t_maps(0, 6).numpy()
    # Reference: rebuild each null data set and refit from scratch.
    beta = np.linalg.pinv(x) @ y.numpy()
    r = y.numpy() - x @ beta
    for i in range(6):
        ystar = r * null._signs[i][:, None]
        if null._perm is not None:
            ystar = ystar[null._perm[i]]
        fit = ols_contrasts(torch.tensor(ystar, dtype=torch.float64), x, c[None])
        np.testing.assert_allclose(got[i], fit.t[0].numpy(), rtol=2e-4, atol=2e-4)


def test_null_is_independent_of_batching():
    ya, _ = _data(6)
    y, x, c = main_test(TtestDesign(9), ya.float(), None)
    null = OlsNull(y, x, c, n_iter=10, seed=3)
    whole = null(0, 10)
    parts = torch.cat([null(0, 3), null(3, 4), null(7, 3)])
    torch.testing.assert_close(whole, parts, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["one_cov", "two_cov", "paired"])
def test_per_brick_specs_reproduce_run_ttest(kind):
    from fastfuncstuff.stats.group_ttest import t_tests

    ya, yb = _data(7)
    rng = np.random.default_rng(8)
    if kind == "one_cov":
        des, yb = TtestDesign(9, cov_a=rng.standard_normal((9, 2)), cov_names=("a", "b")), None
    elif kind == "two_cov":
        des = TtestDesign(9, 7, cov_a=rng.standard_normal((9, 1)), cov_b=rng.standard_normal((7, 1)),
                          cov_names=("a",))  # fmt: skip
    else:
        des, ya, yb = TtestDesign(7, 7, paired=True), ya[:7], yb
    bricks = run_ttest(des, ya, yb)
    specs = t_tests(des, ya, yb)
    assert len(specs) * 2 == len(bricks)
    for sp in specs:
        assert bricks[sp.brick].label == sp.label
        fit = ols_contrasts(sp.y, sp.x, sp.c[None])
        np.testing.assert_allclose(fit.t[0], bricks[sp.brick + 1].data, rtol=1e-10)
    y, x, c = main_test(des, ya, yb)
    np.testing.assert_allclose(specs[0].c, c)
