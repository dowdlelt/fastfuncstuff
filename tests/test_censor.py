"""Timepoint censoring primitives (processing/censor.py) and the AFNI-faithful automask.

The motion and outlier rules were cross-checked TR for TR against 1d_tool.py
-censor_motion and 3dToutcount on real and synthetic data; these tests pin the
semantics that check depended on, so a refactor cannot drift from them silently.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from fastfuncstuff.processing import censor as C
from fastfuncstuff.processing.mask import _afni_fillin_once, afni_automask

CPU = torch.device("cpu")


def test_backward_diff_is_zero_at_every_run_start():
    x = np.arange(10, dtype=float)[:, None] ** 2
    d = C.backward_diff_per_run(x, [4, 6])
    assert d[0, 0] == 0 and d[4, 0] == 0  # the between-run jump never appears
    assert d[5, 0] == 25 - 16


def test_enorm_and_fd_units():
    p = np.zeros((3, 6))
    p[1] = [1.0, 0, 0, 0, 0, 2.0]  # 1 degree roll, 2 mm dP
    e = C.motion_enorm(p)
    fd = C.framewise_displacement(p)
    assert e[1] == pytest.approx(np.sqrt(1 + 4))  # degrees and mm summed raw (AFNI)
    assert fd[1] == pytest.approx(2.0 + np.deg2rad(1.0) * 50.0)
    assert e[0] == 0 and fd[0] == 0


def test_censor_from_trace_limit_is_inclusive_and_prev_is_default():
    trace = np.array([0, 0.1, 0.5, 0.51, 0.2, 0.0])
    keep = C.censor_from_trace(trace, 0.5)
    # 0.5 == limit survives; 0.51 is censored together with the TR before it.
    np.testing.assert_array_equal(keep, [1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        C.censor_from_trace(trace, 0.5, censor_prev=False), [1, 1, 1, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        C.censor_from_trace(trace, 0.5, censor_next=True), [1, 1, 0, 0, 0, 1]
    )


def test_censor_extensions_stay_inside_a_run_and_first_trs_does_not_spread():
    trace = np.array([0, 0, 0, 0.0, 9, 0, 0, 0])  # flagged at run 2's first TR
    keep = C.censor_from_trace(trace, 0.5, run_lengths=[4, 4], first_trs=1)
    np.testing.assert_array_equal(keep, [0, 1, 1, 1, 0, 1, 1, 1])


def test_outlier_censor_is_strict_and_skip_first_is_per_run():
    frac = np.array([0.5, 0.1, 0.2, 0.5, 0.1, 0.2])
    keep = C.censor_from_outliers(frac, 0.1, run_lengths=[3, 3], skip_first=1)
    np.testing.assert_array_equal(keep, [1, 1, 0, 1, 1, 0])


def test_spikes_one_column_per_censored_tr_and_none_when_clean():
    s = C.censor_to_spikes(np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(s, [[0, 0], [1, 0], [0, 0], [0, 1]])
    assert C.censor_to_spikes(np.ones(5)).shape == (5, 0)


def test_combine_is_a_product():
    np.testing.assert_array_equal(C.combine_censor([1, 0, 1, 1], [1, 1, 0, 1]), [1, 0, 0, 1])


def test_afni_median_averages_the_middle_pair():
    x = torch.tensor([[1.0], [2.0], [3.0], [10.0]])
    assert float(C.afni_median(x, 0)) == 2.5  # torch.median would say 2.0


def test_default_polort_matches_afni_proc():
    assert C.default_outlier_polort(2.0, 152) == 3  # 1 + floor(304/150)
    assert C.default_outlier_polort(1.0, 100) == 1


def test_l1_detrend_reaches_the_lp_optimum():
    rng = np.random.default_rng(0)
    T, V, p = 60, 5, 3
    X = np.polynomial.legendre.legvander(np.linspace(-1, 1, T), p - 1)
    Y = X @ rng.normal(size=(p, V)) + rng.standard_t(1.5, size=(T, V))  # heavy tails
    r = C.l1_detrend(torch.from_numpy(Y), torch.from_numpy(X)).numpy()
    for v in range(V):
        # min 1'(u+w)  s.t.  X b + u - w = y
        res = linprog(
            np.r_[np.zeros(p), np.ones(2 * T)],
            A_eq=np.hstack([X, np.eye(T), -np.eye(T)]),
            b_eq=Y[:, v],
            bounds=[(None, None)] * p + [(0, None)] * (2 * T),
        )
        assert np.abs(r[:, v]).sum() == pytest.approx(res.fun, rel=1e-8)


def test_outlier_counts_polort0_matches_the_definition():
    rng = np.random.default_rng(1)
    T, V = 40, 200
    y = rng.normal(size=(T, V))
    y[7, :50] += 30.0  # 50 of 200 voxels spike at TR 7
    counts = C.outlier_counts(torch.from_numpy(y).to(CPU), polort=0).numpy()
    med = np.median(y, axis=0)  # T even: numpy averages the middle pair, as AFNI
    r = y - med
    mad = np.median(np.abs(r), axis=0)
    expect = (np.abs(r) > C.outlier_alpha(T) * mad).sum(axis=1)
    np.testing.assert_array_equal(counts, expect)
    assert counts[7] >= 50


def test_outlier_fraction_4d_uses_each_runs_own_polort_and_mask():
    rng = np.random.default_rng(2)
    vol = torch.zeros(12, 12, 12)
    zz, yy, xx = torch.meshgrid(*(torch.arange(12),) * 3, indexing="ij")
    vol[((zz - 6) ** 2 + (yy - 6) ** 2 + (xx - 6) ** 2) < 20] = 100.0
    data = vol[None] + torch.from_numpy(rng.normal(0, 1, (30, 12, 12, 12))).float()
    data[25] += 50.0 * (vol > 0)  # every brain voxel spikes in run 2
    frac, nvox = C.outlier_fraction_4d(data, run_lengths=[15, 15], tr=2.0)
    assert len(nvox) == 2 and frac.shape == (30,)
    assert frac[25] > 0.9 and np.median(frac) < 0.05


def test_fillin_accepts_set_voxels_at_different_distances():
    # Set voxel at +1 on one side and -2 on the other: AFNI fills at nside=2
    # (any within 1..n on each side); a same-distance rule would not.
    m = np.zeros((7, 7, 9), dtype=bool)
    m[3, 3, 5] = True
    m[3, 3, 2] = True
    out, n = _afni_fillin_once(m, 2)
    assert out[3, 3, 4] and n >= 1
    out1, _ = _afni_fillin_once(m, 1)
    assert not out1[3, 3, 4]


def test_afni_automask_keeps_a_solid_head_whole():
    zz, yy, xx = torch.meshgrid(*(torch.arange(32),) * 3, indexing="ij")
    head = (((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 100).float() * 1000.0
    head += torch.rand(32, 32, 32) * 5.0
    m = afni_automask(head)
    inside = head > 500
    # The peel re-dilates: the boundary shell must come back.
    assert int((m & inside).sum()) == int(inside.sum())
