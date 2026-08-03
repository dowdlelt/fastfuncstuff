"""Per-run validity detection, families, and the censored refit that motivates them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import build_censor_run_info, fit_glm_arma11
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.missing import (
    build_families,
    censor_design,
    detect_run_validity,
    dof_loss_map,
    run_inclusion_map,
    task_survival,
)

NT = 60
NRUN = 3
T = NT * NRUN
RUN_STARTS = [0, NT, 2 * NT]


def _data(n_voxels=8, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(1000.0 + rng.standard_normal((n_voxels, T)).astype(np.float32) * 10.0)


def _design(tr=2.0):
    t = np.arange(NT) * tr
    box = ((t % 30.0) < 10.0).astype(np.float64)
    task = np.tile(box - box.mean(), NRUN)
    polys = [construct_polynomial_matrix(NT, 2, torch.device("cpu")).numpy() for _ in range(NRUN)]
    npoly = polys[0].shape[1]
    P = np.zeros((T, NRUN * npoly))
    for r in range(NRUN):
        P[r * NT : (r + 1) * NT, r * npoly : (r + 1) * npoly] = polys[r]
    return torch.from_numpy(np.column_stack([task, P]).astype(np.float32))


# ── detection ───────────────────────────────────────────────────────────────


def test_clean_data_all_runs_valid():
    v = detect_run_validity(_data(), RUN_STARTS, verbose=False)
    assert bool(v.all_runs_valid.all())
    assert v.valid.shape == (8, NRUN)


def test_all_zero_run_is_invalid_only_for_that_run():
    d = _data()
    d[3, 2 * NT :] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert v.valid[3].tolist() == [True, True, False]
    assert bool(v.constant[3, 2])
    # everyone else untouched
    assert bool(v.valid[torch.arange(8) != 3].all())


def test_mid_run_drop_to_zero_is_caught():
    """The dangerous case: not constant, and a polynomial cannot absorb the step."""
    d = _data()
    d[2, 2 * NT + NT // 2 :] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert v.valid[2].tolist() == [True, True, False]
    assert bool(v.zeros[2, 2])
    assert not bool(v.constant[2, 2])  # genuinely not constant


def test_constant_nonzero_run_is_invalid():
    d = _data()
    d[1, NT : 2 * NT] = 742.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert not bool(v.valid[1, 1])
    assert bool(v.constant[1, 1])


def test_constant_rule_is_scale_invariant():
    """Percent-signal-scaled data must classify identically to raw."""
    d = _data()
    d[4, NT : 2 * NT] = 0.0
    raw = detect_run_validity(d, RUN_STARTS, verbose=False)
    scaled = detect_run_validity(d / 10.0, RUN_STARTS, verbose=False)
    assert torch.equal(raw.valid, scaled.valid)


def test_negative_run_rejected_on_magnitude_data():
    d = _data()
    d[5, NT : 2 * NT] = -50.0 + torch.randn(NT) * 2.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert not bool(v.valid[5, 1])
    assert bool(v.negative[5, 1])
    assert v.negative_rule_active


def test_negative_rule_self_disables_on_centered_data():
    """A detrended input is ~50% negative; the rule must not delete the brain."""
    d = _data() - 1000.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert not v.negative_rule_active
    assert not bool(v.negative.any())
    assert bool(v.all_runs_valid.all())


def test_isolated_negative_samples_do_not_reject():
    """Sinc/wsinc5 ringing puts a few negatives in good cortex."""
    d = _data()
    d[6, NT + 3] = -5.0
    d[6, NT + 17] = -2.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    assert bool(v.valid[6].all())


# ── families ────────────────────────────────────────────────────────────────


def test_families_group_by_pattern():
    d = _data(n_voxels=400)
    d[:100, 2 * NT :] = 0.0  # missing run 2
    d[100:200, NT : 2 * NT] = 0.0  # missing run 1
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, demoted = build_families(v, RUN_STARTS, T, min_family_voxels=10, verbose=False)

    assert len(fams) == 2
    assert {f.n_voxels for f in fams} == {100}
    assert {tuple(f.pattern.tolist()) for f in fams} == {
        (True, True, False),
        (True, False, True),
    }
    assert not bool(demoted.any())
    for f in fams:
        assert len(f.good_list) == 2 * NT


def test_small_families_are_demoted_not_fitted():
    d = _data(n_voxels=200)
    d[:5, 2 * NT :] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, demoted = build_families(v, RUN_STARTS, T, min_family_voxels=50, verbose=False)
    assert fams == []
    assert int(demoted.sum()) == 5


def test_max_families_budget_demotes_overflow():
    d = _data(n_voxels=600)
    d[:200, 2 * NT :] = 0.0
    d[200:400, NT : 2 * NT] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, demoted = build_families(
        v, RUN_STARTS, T, min_family_voxels=10, max_families=1, verbose=False
    )
    assert len(fams) == 1
    assert int(demoted.sum()) == 200


def test_family_rejected_when_a_task_regressor_does_not_survive():
    """Blocked design: run 0 is condition A only, run 1 condition B only."""
    a = np.zeros(T)
    b = np.zeros(T)
    a[:NT] = np.tile([1.0, 0.0], NT // 2)
    b[NT : 2 * NT] = np.tile([1.0, 0.0], NT // 2)
    design = torch.from_numpy(np.column_stack([a, b]).astype(np.float32))

    d = _data(n_voxels=300)
    d[:150, NT : 2 * NT] = 0.0  # lose run 1 -> condition B unobservable
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, demoted = build_families(
        v,
        RUN_STARTS,
        T,
        min_family_voxels=10,
        design=design,
        task_indices=[0, 1],
        verbose=False,
    )
    assert fams == []
    assert int(demoted.sum()) == 150


def test_task_survival_fractions():
    design = _design()
    full = task_survival(design, [0], list(range(T)))
    assert float(full[0]) == pytest.approx(1.0)
    two_thirds = task_survival(design, [0], list(range(2 * NT)))
    assert float(two_thirds[0]) == pytest.approx(2 / 3, abs=1e-6)


def test_censor_design_drops_dead_run_polynomials():
    design = _design()
    good = list(range(2 * NT))
    sub, keep = censor_design(design, good)
    assert sub.shape[0] == 2 * NT
    # run-2's three polynomial columns go all-zero and must be removed
    assert int((~keep).sum()) == 3
    assert sub.shape[1] == design.shape[1] - 3
    assert bool(keep[0])  # the task column survives


# ── bookkeeping artifacts ───────────────────────────────────────────────────


def test_run_inclusion_map_matches_families():
    d = _data(n_voxels=300)
    d[:100, 2 * NT :] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, _ = build_families(v, RUN_STARTS, T, min_family_voxels=10, verbose=False)
    mask = torch.ones(300, dtype=torch.bool)
    inc = run_inclusion_map(v, fams, mask)

    assert inc.shape == (300, NRUN)
    assert inc[0].tolist() == [1.0, 1.0, 0.0]  # family voxel
    assert inc[250].tolist() == [1.0, 1.0, 1.0]  # clean voxel
    # out-of-mask voxels are zeroed
    mask[299] = False
    assert run_inclusion_map(v, fams, mask)[299].tolist() == [0.0, 0.0, 0.0]


def test_dof_loss_map_uses_fitted_dof_not_timepoints():
    """Censoring drops polynomial columns too, so loss < timepoints removed."""
    d = _data(n_voxels=200)
    d[:100, 2 * NT :] = 0.0
    v = detect_run_validity(d, RUN_STARTS, verbose=False)
    fams, _ = build_families(v, RUN_STARTS, T, min_family_voxels=10, verbose=False)
    mask = torch.ones(200, dtype=torch.bool)

    dof_full = T - 10
    loss = dof_loss_map(200, fams, [2 * NT - 7], dof_full, mask)
    assert float(loss[0]) == pytest.approx(dof_full - (2 * NT - 7))
    assert float(loss[0]) < NT  # strictly less than the timepoints removed
    assert float(loss[150]) == 0.0


# ── the thing this all exists for ───────────────────────────────────────────


def test_censored_refit_removes_the_beta_dilution():
    """A dead run left in dilutes beta and deflates sigma2; censoring fixes both."""
    rng = np.random.default_rng(3)
    design = _design()
    X = design.numpy().astype(np.float64)
    task = X[:, 0]
    y = 1000.0 + 1.0 * task + rng.standard_normal(T) * 0.5

    Y = np.repeat(y[None, :], 2, axis=0)
    Y[1, 2 * NT :] = 0.0  # run 2 dead

    def fit(data, dmat, rs, tau=None):
        return fit_glm_arma11(
            torch.from_numpy(np.ascontiguousarray(data)).float(),
            torch.from_numpy(np.ascontiguousarray(dmat)).float(),
            tr=2.0,
            run_starts=rs,
            tau=tau,
            task_indices=[0],
            device=torch.device("cpu"),
            verbose=False,
        )

    truth = fit(Y[:1], X, RUN_STARTS)
    naive = fit(Y[1:2], X, RUN_STARTS)

    good = list(range(2 * NT))
    rs_ret, tau = build_censor_run_info(RUN_STARTS, T, good_list=good)
    Xc, _ = censor_design(design, good)
    fixed = fit(Y[1:2, good], Xc.numpy().astype(np.float64), rs_ret, tau=tau)

    b_true = float(truth.betas[0, 0])
    b_naive = float(naive.betas[0, 0])
    b_fixed = float(fixed.betas[0, 0])

    # Naive loses ~1/3 of the beta; censored recovers most of it.
    assert b_naive < 0.8 * b_true
    assert abs(b_fixed - b_true) < abs(b_naive - b_true)

    # And the deflated noise variance is what hides it: naive sigma2 is far
    # below truth while the censored fit recovers the right scale.
    s_true = float(truth.sigma2[0])
    assert float(naive.sigma2[0]) < 0.8 * s_true
    assert float(fixed.sigma2[0]) > 0.8 * s_true

    # The censored fit honestly reports fewer degrees of freedom.
    assert fixed.dof < truth.dof


def test_accumulator_matches_whole_array_detection():
    """The streaming path ffs_reml uses must agree with the batch path exactly."""
    from fastfuncstuff.glm.missing import RunValidityAccumulator

    d = _data(n_voxels=32, seed=7)
    d[3, 2 * NT :] = 0.0
    d[5, NT : 2 * NT] = 742.0
    d[9, NT + NT // 2 : 2 * NT] = 0.0

    batch = detect_run_validity(d, RUN_STARTS, verbose=False)

    acc = RunValidityAccumulator(32, NRUN)
    for r, s in enumerate(RUN_STARTS):
        e = RUN_STARTS[r + 1] if r + 1 < NRUN else T
        acc.observe_run(d[:, s:e], r)
    streamed = acc.finalize(verbose=False)

    assert torch.equal(batch.valid, streamed.valid)
    assert torch.equal(batch.constant, streamed.constant)
    assert torch.equal(batch.zeros, streamed.zeros)


def test_accumulator_rejects_unobserved_run():
    from fastfuncstuff.glm.missing import RunValidityAccumulator

    acc = RunValidityAccumulator(4, NRUN)
    acc.observe_run(_data(n_voxels=4)[:, :NT], 0)
    with pytest.raises(RuntimeError, match="never observed"):
        acc.finalize(verbose=False)
