"""Global ETAC (stats/etac.py).

The claim the module rests on: a test iteration is a false positive exactly
when some sub-test's largest null cluster reaches its threshold, so the FPR
can be read off the stored max-FOM matrix instead of re-clustering (what
3dXClustSim does).  ``test_fpr_shortcut_equals_reclustering`` checks that
claim against applying the thresholds to the null maps themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from fastfuncstuff.stats.etac import (
    EtacNull,
    EtacOptions,
    apply,
    build_null,
    fpr_at,
    run_group_etac,
    search,
    thresholds_at,
)
from fastfuncstuff.stats.group_ttest import OlsNull, TtestDesign, main_test

CPU = torch.device("cpu")


def _subjects(n, seed, shape=(16, 16, 12), effect=0.0, smooth=1.2):
    rng = np.random.default_rng(seed)
    g = np.indices(shape).astype(float)
    c = np.array(shape)[:, None, None, None] / 2
    mask = ((g - c) ** 2 / (c * 0.9) ** 2).sum(0) < 1
    blob = np.exp(-((g[0] - 6) ** 2 + (g[1] - 7) ** 2 + (g[2] - 6) ** 2) / 6)
    y = np.stack([
        (gaussian_filter(rng.standard_normal(shape), smooth) * 3 + effect * blob)[mask]
        for _ in range(n)
    ])  # fmt: skip
    return mask, torch.tensor(y, dtype=torch.float32)


def test_thresholds_follow_afni_ranking_rule():
    # One sub-test, 1000 iterations, maxima 1..1000: the threshold at tfrac is
    # near the (1 - tfrac) quantile and interpolates between ranks.
    fom = np.arange(1, 1001, dtype=np.float32)[:, None, None, None]
    null = EtacNull(fom, ("B0.0",), EtacOptions(pthr=(0.01,), hpows=(0,)), n_train=1000)
    t = thresholds_at(null, 0.05)[0, 0, 0]
    assert 945 < t < 955
    assert thresholds_at(null, 0.01)[0, 0, 0] > t  # stricter tail, higher bar


def test_fpr_shortcut_equals_reclustering():
    mask, y = _subjects(12, 0)
    opts = EtacOptions(pthr=(0.01, 0.002), hpows=(0, 2), fpr=(10.0,))
    yy, x, c = main_test(TtestDesign(12), y, None)
    src = OlsNull(yy, x, c, n_iter=300, seed=1)
    null = build_null({"B0.0": src}, 300, mask, opts, CPU, verbose=False)
    res = search(null, 10.0)
    assert 0 < res.fpr <= 10.0
    hits = 0
    for i in range(300):
        union, _ = apply({"B0.0": src(i, 1)[0]}, mask, opts, res)
        hits += bool(union.any())
    assert 100.0 * hits / 300 == pytest.approx(res.fpr)
    assert fpr_at(null, res.thr) == pytest.approx(res.fpr)


def test_null_data_rarely_survives_and_signal_does():
    mask, y0 = _subjects(14, 2)
    opts = EtacOptions(pthr=(0.01, 0.005, 0.001), hpows=(2,))
    r0 = run_group_etac(TtestDesign(14), y0, None, mask, (3, 3, 3), blurs=(0.0, 4.0),
                        opts=opts, n_iter=600, device=CPU, verbose=False)  # fmt: skip
    assert r0.thresholds[0].fpr <= 5.0
    mask, y1 = _subjects(14, 3, effect=2.5)
    r1 = run_group_etac(TtestDesign(14), y1, None, mask, (3, 3, 3), blurs=(0.0, 4.0),
                        opts=opts, n_iter=600, device=CPU, verbose=False)  # fmt: skip
    union, per = r1.masks[(5.0, "2sid")]
    assert union.sum() > 20
    assert per.shape == (2 * 3, int(mask.sum()))


def test_one_sided_gives_positive_and_negative_masks():
    mask, y = _subjects(12, 4, effect=-2.5)
    opts = EtacOptions(pthr=(0.01, 0.002), sided=1)
    r = run_group_etac(TtestDesign(12), y, None, mask, (3, 3, 3), opts=opts, n_iter=400,
                       device=CPU, verbose=False)  # fmt: skip
    assert set(r.masks) == {(5.0, "1pos"), (5.0, "1neg")}
    assert r.masks[(5.0, "1neg")][0].sum() > r.masks[(5.0, "1pos")][0].sum()


# The honest referee: fresh null data sets, not the permutations themselves (~7 s).
def test_familywise_rate_on_fresh_null_datasets():
    opts = EtacOptions(pthr=(0.01, 0.005, 0.002), hpows=(2,), fpr=(10.0,))
    hits = 0
    n_sets = 40
    for s in range(n_sets):
        mask, y = _subjects(12, 100 + s)
        r = run_group_etac(TtestDesign(12), y, None, mask, (3, 3, 3), blurs=(0.0, 4.0),
                           opts=opts, n_iter=500, seed=s, device=CPU, verbose=False)  # fmt: skip
        hits += bool(r.masks[(10.0, "2sid")][0].any())
    # 10% goal: 4 of 40 expected; binomial 99% interval is about 0-10.
    assert hits <= 10


def test_options_are_validated():
    with pytest.raises(ValueError, match="splitfrac"):
        EtacOptions(splitfrac=0.1)
    with pytest.raises(ValueError, match="hpow"):
        EtacOptions(hpows=(3,))
