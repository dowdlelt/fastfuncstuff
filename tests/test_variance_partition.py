"""Tests for stats/variance_partition.py.

The test that earns its place is ``test_rank1_interaction_recovered_at_three_repeats``:
an implementation that skips per-band shrinkage passes a generous-repeat-count test and
fails this one, because at n=3 the saturated model estimates each cell from 2 trials and
its unregularized CV-R2 collapses. See ../fmri_wiki/concepts/Variance partitioning.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.variance_partition import (
    BALANCE_TOL,
    _build_cell_ops,
    _cellspace_partials,
    _orthonormal_contrasts,
    _partition_stats,
    build_factor_design,
    build_repeat_folds,
    derive_repeat_index,
    partition_variance,
    permutation_test,
)

N_STIM = 20
N_TASK = 21
N_REP = 3


def make_crossed_table(n_stim=N_STIM, n_task=N_TASK, n_rep=N_REP, seed=0):
    """Exhaustively crossed trial table, shuffled into runs so repeats never share a run."""
    rng = np.random.default_rng(seed)
    stim, task, rep = [], [], []
    for r in range(n_rep):
        cells = [(s, t) for s in range(n_stim) for t in range(n_task)]
        rng.shuffle(cells)
        for s, t in cells:
            stim.append(s)
            task.append(t)
            rep.append(r)
    stim = np.array(stim)
    task = np.array(task)
    rep = np.array(rep)
    # One run per repeat block keeps every cell's repeats in distinct runs.
    run = rep.copy()
    return {"stim": stim, "task": task}, rep, run


def synth_betas(factors, *, a=None, b=None, e=None, noise=1.0, seed=0):
    """Cell-mean model plus iid noise, one voxel per row of the supplied effect arrays."""
    rng = np.random.default_rng(seed)
    s, t = factors["stim"], factors["task"]
    n_trials = len(s)
    n_vox = max(x.shape[0] for x in (a, b, e) if x is not None)
    y = np.zeros((n_vox, n_trials))
    if a is not None:
        y += a[:, s]
    if b is not None:
        y += b[:, t]
    if e is not None:
        y += e[:, s, t]
    y += rng.normal(0, noise, size=y.shape)
    return torch.as_tensor(y, dtype=torch.float32)


def centered(x, axis=None):
    if axis is None:
        return x - x.mean()
    return x - x.mean(axis=axis, keepdims=True)


def test_orthonormal_contrasts_span_sum_to_zero():
    c = _orthonormal_contrasts(7, torch.float64)
    assert c.shape == (7, 6)
    torch.testing.assert_close(c.T @ c, torch.eye(6, dtype=torch.float64), atol=1e-10, rtol=0)
    # Orthogonal to the intercept: this is what makes the bands decouple.
    assert c.sum(dim=0).abs().max() < 1e-10


def test_balanced_crossed_design_has_orthogonal_bands():
    factors, _, _ = make_crossed_table()
    design = build_factor_design(factors)
    assert design.balanced
    assert design.max_offdiag < BALANCE_TOL
    assert design.bands["stim"].shape[1] == N_STIM - 1
    assert design.bands["task"].shape[1] == N_TASK - 1
    assert design.bands["stim:task"].shape[1] == (N_STIM - 1) * (N_TASK - 1)
    assert int(design.cell_counts.min()) == N_REP


def test_dropped_trials_break_balance():
    factors, _, _ = make_crossed_table()
    keep = np.ones(len(factors["stim"]), dtype=bool)
    keep[:40] = False  # censor a block of trials
    dropped = {k: v[keep] for k, v in factors.items()}
    design = build_factor_design(dropped)
    assert not design.balanced
    assert design.max_offdiag > BALANCE_TOL


def test_derive_repeat_index_matches_explicit():
    factors, rep, _ = make_crossed_table()
    design = build_factor_design(factors)
    derived = derive_repeat_index(design)
    np.testing.assert_array_equal(derived, rep)


def test_fold_builder_refuses_run_leak():
    factors, rep, run = make_crossed_table()
    _, diag = build_repeat_folds(rep, run)
    assert diag["run_locality_ok"]
    assert diag["n_folds"] == N_REP

    leaky = np.zeros_like(run)  # every trial in one run -> every fold leaks
    _, diag_leak = build_repeat_folds(rep, leaky)
    assert not diag_leak["run_locality_ok"]
    assert diag_leak["run_leaks"]


def test_partition_rejects_run_leak():
    factors, rep, _ = make_crossed_table()
    betas = synth_betas(factors, a=np.zeros((4, N_STIM)), noise=1.0)
    with pytest.raises(ValueError, match="leaks runs"):
        partition_variance(betas, factors, repeat=rep, run=np.zeros_like(rep), verbose=False)


def test_shared_variance_is_near_zero_on_balanced_design():
    """Exhaustive crossing makes S and T orthogonal, so C has nothing to find."""
    rng = np.random.default_rng(1)
    factors, rep, run = make_crossed_table()
    n_vox = 60
    a = rng.normal(0, 1.0, size=(n_vox, N_STIM))
    b = rng.normal(0, 1.0, size=(n_vox, N_TASK))
    betas = synth_betas(factors, a=a, b=b, noise=1.0, seed=2)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    assert res.diagnostics["balanced"]
    assert res.shared.abs().median() < 0.02
    assert res.diagnostics["shared_abs_median"] < 0.02


def test_preference_index_tracks_the_dominant_factor():
    rng = np.random.default_rng(3)
    factors, rep, run = make_crossed_table()
    n_each = 30
    # First block: stimulus-driven. Second block: task-driven.
    a = np.concatenate([rng.normal(0, 1.5, size=(n_each, N_STIM)), np.zeros((n_each, N_STIM))])
    b = np.concatenate([np.zeros((n_each, N_TASK)), rng.normal(0, 1.5, size=(n_each, N_TASK))])
    betas = synth_betas(factors, a=a, b=b, noise=1.0, seed=4)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    stim_pref = res.preference[:n_each]
    task_pref = res.preference[n_each:]
    # preference = (U_task - U_stim) / (U_task + U_stim): negative when stimulus wins.
    assert stim_pref.median() < -0.5
    assert task_pref.median() > 0.5
    assert res.unique["stim"][:n_each].median() > res.unique["task"][:n_each].median()
    assert res.unique["task"][n_each:].median() > res.unique["stim"][n_each:].median()


def test_rank1_interaction_recovered_at_three_repeats():
    """The load-bearing test: a real rank-1 interaction must survive n=3.

    Half the voxels are purely additive, half carry a rank-1 stimulus x task interaction
    at ~0.6x the main-effect scale. With 3 repeats the saturated model estimates each cell
    from 2 trials under leave-one-repeat-out, so an implementation that skips per-band
    shrinkage overfits catastrophically. Measured with gamma forced to 1 (verified across
    6 seeds, amp=0.6, noise=1.0):

        band shrinkage    I(additive)         I(interacting)
        gamma fitted       0.0000              +0.032 .. +0.039
        gamma = 1         -0.149 .. -0.160     -0.046 .. -0.064

    The unshrunk interaction is *negative for both groups* -- a real interaction is not
    merely attenuated, it is undetectable. Both assertions below flip under gamma=1, and
    the amplitude is chosen so they flip with room to spare rather than marginally.
    """
    rng = np.random.default_rng(100)
    factors, rep, run = make_crossed_table()
    n_each = 40
    n_vox = 2 * n_each

    a = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)

    # Rank-1 interaction: a single stimulus profile scaled by a single task profile.
    u = centered(rng.normal(0, 1.0, size=(n_each, N_STIM)), axis=1)
    v = centered(rng.normal(0, 1.0, size=(n_each, N_TASK)), axis=1)
    e = np.zeros((n_vox, N_STIM, N_TASK))
    e[n_each:] = 0.6 * u[:, :, None] * v[:, None, :]

    betas = synth_betas(factors, a=a, b=b, e=e, noise=1.0, seed=200)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    add_i = res.interaction[:n_each]
    int_i = res.interaction[n_each:]

    # 1. A real interaction is positively detected. Fails at -0.046 without shrinkage.
    assert int_i.median() > 0.02, f"interaction not recovered: {int_i.median():.4f}"

    # 2. An absent band costs nothing, because gamma drives it to zero. Fails at -0.149
    #    without shrinkage -- the signature of the saturated model overfitting 2-trial
    #    cell means.
    assert add_i.median() > -0.01, f"absent interaction overfit: {add_i.median():.4f}"

    # 3. gamma_interaction is the direct readout of how much raw interaction survives CV.
    g_add = res.gammas["stim:task"][:n_each]
    g_int = res.gammas["stim:task"][n_each:]
    assert g_add.median() < 0.1
    assert g_int.median() > 0.2

    # 4. Rank is the scientific payoff: additive voxels -> 0, gain voxels -> 1.
    assert res.rank_e[:n_each].float().median() == 0
    assert res.rank_e[n_each:].float().median() == 1


def test_rank_is_undetermined_not_zero_below_the_snr_floor():
    """Low-SNR voxels must not be reported as task-invariant.

    Rank selection errs in one direction only: measured on synthetic 20x21x3 data a
    purely additive truth never produces rank >= 1 at any noise level, while a true
    rank-1 interaction is *missed* more than half the time once ncsnr drops below ~0.75.
    An unmasked rank map therefore paints "additive" over white matter, dropout and
    edges -- a spatial artifact that reads as a finding. Those voxels get -1 instead.
    """
    rng = np.random.default_rng(21)
    factors, rep, run = make_crossed_table()
    n_each = 40
    a = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    u = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)
    v = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    e = 0.8 * u[:, :, None] * v[:, None, :]

    s, t = factors["stim"], factors["task"]
    y = a[:, s] + b[:, t] + e[:, s, t]
    # First block clean, second block swamped.
    noise = np.concatenate([np.full(n_each, 0.5), np.full(n_each, 8.0)])
    y = y + rng.normal(0, 1.0, size=y.shape) * noise[:, None]
    betas = torch.as_tensor(y, dtype=torch.float32)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    assert res.rank_e[:n_each].float().median() == 1, "clean voxels should resolve rank 1"
    # Swamped voxels: flagged undetermined, never silently called additive.
    assert (res.rank_e[n_each:] == -1).float().mean() > 0.9
    assert (res.rank_e[n_each:] == 0).float().mean() < 0.05
    # The raw argmax is still exposed, and it is what would have produced the artifact.
    assert res.rank_e_raw is not None
    assert (res.rank_e_raw[n_each:] == 0).float().mean() > 0.9
    assert res.diagnostics["rank_undetermined_frac"] > 0.4


def test_additive_only_data_selects_rank_zero():
    rng = np.random.default_rng(7)
    factors, rep, run = make_crossed_table()
    n_vox = 40
    a = centered(rng.normal(0, 1.2, size=(n_vox, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.2, size=(n_vox, N_TASK)), axis=1)
    betas = synth_betas(factors, a=a, b=b, noise=1.0, seed=8)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    assert res.rank_e.float().median() == 0
    assert res.interaction.median() < 0.02
    assert res.gammas["stim:task"].median() < 0.3


def test_noise_ceiling_orders_by_snr():
    rng = np.random.default_rng(9)
    factors, rep, run = make_crossed_table()
    n_each = 20
    a = np.tile(centered(rng.normal(0, 1.0, size=(1, N_STIM)), axis=1), (2 * n_each, 1))
    noise = np.concatenate([np.full(n_each, 0.3), np.full(n_each, 3.0)])
    s = factors["stim"]
    y = a[:, s] + rng.normal(0, 1.0, size=(2 * n_each, len(s))) * noise[:, None]

    res = partition_variance(
        torch.as_tensor(y, dtype=torch.float32), factors, repeat=rep, run=run, verbose=False
    )

    assert res.ncsnr[:n_each].median() > res.ncsnr[n_each:].median()
    assert res.noise_ceiling[:n_each].median() > res.noise_ceiling[n_each:].median()
    assert 0.0 <= float(res.noise_ceiling.min()) <= float(res.noise_ceiling.max()) <= 1.0


def test_requires_two_factors():
    factors, _, _ = make_crossed_table()
    with pytest.raises(ValueError, match="at least 2 factors"):
        build_factor_design({"stim": factors["stim"]})


def test_trial_count_mismatch_is_rejected():
    factors, rep, run = make_crossed_table()
    betas = torch.zeros(5, len(factors["stim"]) - 1)
    with pytest.raises(ValueError, match="one row per volume"):
        partition_variance(betas, factors, repeat=rep, run=run, verbose=False)


# ---------------------------------------------------------------------------
# Permutation inference
# ---------------------------------------------------------------------------


def _perm_data(amp, *, n_vox=100, seed=0, with_task=True):
    """Additive main effects plus an optional rank-1 interaction of amplitude ``amp``."""
    factors, rep, run = make_crossed_table()
    rng = np.random.default_rng(seed)
    a = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    b = (
        centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
        if with_task
        else np.zeros((n_vox, N_TASK))
    )
    u = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    v = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
    e = amp * u[:, :, None] * v[:, None, :]
    return synth_betas(factors, a=a, b=b, e=e, noise=1.0, seed=seed + 50), factors, rep, run


def test_cellspace_engine_matches_general_path():
    """The permutation engine must compute the *same* statistic as the reported one.

    The observed value is produced by the cell-space fast path while ``partition_variance``
    uses the general pinv path; if they disagreed, every p-value would be comparing an
    observed statistic against a null drawn from a different estimator.
    """
    betas, factors, rep, run = _perm_data(0.6, n_vox=50, seed=11)
    design = build_factor_design(factors)
    folds, _ = build_repeat_folds(rep, run)
    ops = _build_cell_ops(design, folds, torch.device("cpu"))
    fast = _partition_stats(*_cellspace_partials(betas, ops))

    slow = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    torch.testing.assert_close(fast["unique_a"], slow.unique["stim"], atol=1e-4, rtol=0)
    torch.testing.assert_close(fast["unique_b"], slow.unique["task"], atol=1e-4, rtol=0)
    torch.testing.assert_close(fast["interaction"], slow.interaction, atol=1e-4, rtol=0)
    torch.testing.assert_close(fast["shared"], slow.shared, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        fast["gamma_interaction"], slow.gammas["stim:task"], atol=1e-4, rtol=0
    )


@pytest.mark.slow
def test_null_is_not_degenerate():
    """The null max must be a live distribution, not a point mass at zero.

    Band shrinkage clamps gamma to [0, 1], so an absent effect gives a statistic of
    *exactly* zero. If every permutation also produced exactly zero, then any voxel with a
    trivially positive observed value would score p = 1/(P+1) and the test would
    manufacture false positives. Taking the max over voxels is what keeps the null live.
    """
    betas, factors, rep, run = _perm_data(0.0, n_vox=100, seed=12)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("interaction",),
        n_perms=100,
        seed=1,
        verbose=False,
    )
    null_max = res.null_max["interaction"]
    assert (null_max > 0).all(), "null max collapsed to zero; the test would be vacuous"
    assert null_max.std() > 0


@pytest.mark.slow
def test_interaction_null_controls_type_one_error():
    """Additive-only truth: no voxel should survive FWE, and uncorrected p stays nominal."""
    betas, factors, rep, run = _perm_data(0.0, n_vox=100, seed=13)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("interaction",),
        n_perms=200,
        seed=1,
        verbose=False,
    )
    p_fwe = res.p_fwe["interaction"]
    p_unc = res.p_uncorrected["interaction"]
    # Measured 0.000 FWE / 0.060 uncorrected at nominal 0.05 (100 voxels, SE ~0.022).
    assert (p_fwe < 0.05).float().mean() <= 0.02
    assert (p_unc < 0.05).float().mean() <= 0.15


@pytest.mark.slow
def test_interaction_null_has_power():
    """A real rank-1 interaction must survive FWE correction."""
    betas, factors, rep, run = _perm_data(0.6, n_vox=100, seed=14)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("interaction",),
        n_perms=200,
        seed=1,
        verbose=False,
    )
    assert (res.p_fwe["interaction"] < 0.05).float().mean() > 0.8


@pytest.mark.slow
def test_unique_task_null_controls_type_one_error():
    """Stimulus effect present, task effect absent: unique_b must not be significant."""
    betas, factors, rep, run = _perm_data(0.0, n_vox=100, seed=15, with_task=False)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("unique_b",),
        n_perms=200,
        seed=1,
        verbose=False,
    )
    assert (res.p_fwe["unique_b"] < 0.05).float().mean() <= 0.02
    assert (res.p_uncorrected["unique_b"] < 0.05).float().mean() <= 0.15


def test_permutation_rejects_unbalanced_design():
    betas, factors, rep, run = _perm_data(0.0, n_vox=10, seed=16)
    keep = np.ones(len(factors["stim"]), dtype=bool)
    keep[:40] = False
    with pytest.raises(ValueError, match="requires a balanced crossed design"):
        permutation_test(
            betas[:, keep],
            {k: v[keep] for k, v in factors.items()},
            repeat=rep[keep],
            run=run[keep],
            n_perms=2,
            verbose=False,
        )


def test_permutation_rejects_unknown_statistic():
    betas, factors, rep, run = _perm_data(0.0, n_vox=10, seed=17)
    with pytest.raises(ValueError, match="unknown statistics"):
        permutation_test(
            betas, factors, repeat=rep, run=run, statistics=("bogus",), n_perms=2, verbose=False
        )


def test_permutation_warns_without_run_blocks():
    betas, factors, rep, _ = _perm_data(0.0, n_vox=10, seed=18)
    with pytest.warns(UserWarning, match="anticonservative"):
        permutation_test(
            betas,
            factors,
            repeat=rep,
            run=None,
            statistics=("interaction",),
            n_perms=2,
            verbose=False,
        )


def test_rank_masking_can_be_disabled():
    betas, factors, rep, run = _perm_data(0.0, n_vox=20, seed=22)
    res = partition_variance(
        betas, factors, repeat=rep, run=run, min_ncsnr_for_rank=0.0, verbose=False
    )
    assert (res.rank_e >= 0).all()
    torch.testing.assert_close(res.rank_e, res.rank_e_raw)


# ---------------------------------------------------------------------------
# ROI collapsing
# ---------------------------------------------------------------------------


class TestRoiCollapse:
    def test_label_map_groups_by_value(self):
        from fastfuncstuff.stats.variance_partition import build_roi_weights, collapse_to_rois

        atlas = np.array([[[1, 1, 2, 0]]])  # 1x1x4, one voxel unassigned
        spec, ids, sizes = build_roi_weights(atlas)
        assert ids == [1, 2]
        assert sizes.tolist() == [2.0, 1.0]

        betas = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0], [99.0, 99.0]])
        out = collapse_to_rois(betas, spec, sizes, device=torch.device("cpu"))
        # ROI 1 = mean of first two rows; the label-0 voxel must not contribute.
        torch.testing.assert_close(out[0], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(out[1], torch.tensor([10.0, 20.0]))

    def test_label_map_respects_mask(self):
        from fastfuncstuff.stats.variance_partition import build_roi_weights

        atlas = np.array([[[1, 1, 2, 2]]])
        mask = np.array([[[True, False, True, True]]])
        spec, ids, sizes = build_roi_weights(atlas, mask=mask)
        assert spec.shape[0] == 3  # only in-mask voxels
        assert sizes.tolist() == [1.0, 2.0]

    def test_4d_atlas_allows_overlap_and_weights(self):
        from fastfuncstuff.stats.variance_partition import build_roi_weights, collapse_to_rois

        # two ROIs sharing voxel 1; second ROI is weighted, not binary
        atlas = np.zeros((1, 1, 3, 2), dtype=np.float32)
        atlas[0, 0, :, 0] = [1.0, 1.0, 0.0]
        atlas[0, 0, :, 1] = [0.0, 2.0, 2.0]
        w, ids, sizes = build_roi_weights(atlas)
        assert w.shape == (2, 3)
        assert sizes.tolist() == [2.0, 4.0]

        betas = torch.tensor([[1.0], [3.0], [5.0]])
        out = collapse_to_rois(betas, w, sizes, device=torch.device("cpu"))
        torch.testing.assert_close(out[0], torch.tensor([2.0]))  # (1+3)/2
        torch.testing.assert_close(out[1], torch.tensor([4.0]))  # (2*3+2*5)/4

    def test_rejects_empty_or_wrong_dimensionality(self):
        from fastfuncstuff.stats.variance_partition import build_roi_weights

        with pytest.raises(ValueError, match="no non-zero labels"):
            build_roi_weights(np.zeros((2, 2, 2), dtype=int))
        with pytest.raises(ValueError, match="3-D .* or 4-D"):
            build_roi_weights(np.zeros((2, 2)))

    def test_collapsing_raises_the_noise_ceiling(self):
        """The point of -atlas beyond speed: averaging lifts ncsnr past the rank floor."""
        from fastfuncstuff.stats.variance_partition import build_roi_weights, collapse_to_rois

        rng = np.random.default_rng(31)
        factors, rep, run = make_crossed_table()
        n_vox = 64
        # One shared signal across all voxels, swamped per voxel by independent noise.
        a = np.tile(centered(rng.normal(0, 1.0, size=(1, N_STIM)), axis=1), (n_vox, 1))
        b = np.tile(centered(rng.normal(0, 1.0, size=(1, N_TASK)), axis=1), (n_vox, 1))
        s, t = factors["stim"], factors["task"]
        y = a[:, s] + b[:, t] + rng.normal(0, 6.0, size=(n_vox, len(s)))
        betas = torch.as_tensor(y, dtype=torch.float32)

        vox = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

        atlas = np.ones((4, 4, 4), dtype=int)  # single ROI over all 64 voxels
        spec, _, sizes = build_roi_weights(atlas)
        roi_betas = collapse_to_rois(betas, spec, sizes, device=torch.device("cpu"))
        roi = partition_variance(roi_betas, factors, repeat=rep, run=run, verbose=False)

        assert float(roi.ncsnr[0]) > float(vox.ncsnr.median()) * 3
        # Per voxel the signal is below the rank floor; pooled it is not.
        assert vox.ncsnr.median() < 0.75 <= float(roi.ncsnr[0])


class TestPaintRoisToVoxels:
    def test_label_map_broadcasts_and_fills_unassigned(self):
        from fastfuncstuff.stats.variance_partition import paint_rois_to_voxels

        spec = np.array([0, 0, 1, -1])
        out = paint_rois_to_voxels(np.array([2.5, 7.0]), spec, n_voxels=4)
        assert out.tolist() == [2.5, 2.5, 7.0, 0.0]

    def test_overlap_resolves_by_weighted_average(self):
        from fastfuncstuff.stats.variance_partition import paint_rois_to_voxels

        # voxel 1 belongs to both ROIs (weights 1 and 3), voxel 2 to neither
        w = np.array([[1.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
        out = paint_rois_to_voxels(np.array([4.0, 8.0]), w, n_voxels=3)
        assert out[0] == pytest.approx(4.0)
        assert out[1] == pytest.approx((1 * 4.0 + 3 * 8.0) / 4)
        assert out[2] == 0.0

    def test_disjoint_binary_4d_is_exact(self):
        """Painting must not smear values where coverage is disjoint."""
        from fastfuncstuff.stats.variance_partition import paint_rois_to_voxels

        w = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = paint_rois_to_voxels(np.array([1.5, -2.0]), w, n_voxels=2)
        assert out.tolist() == [1.5, -2.0]

    def test_roundtrip_collapse_then_paint_is_the_roi_mean(self):
        from fastfuncstuff.stats.variance_partition import (
            build_roi_weights,
            collapse_to_rois,
            paint_rois_to_voxels,
        )

        atlas = np.array([[[1, 1, 2, 2]]])
        spec, _, sizes = build_roi_weights(atlas)
        betas = torch.tensor([[1.0], [3.0], [10.0], [20.0]])
        roi = collapse_to_rois(betas, spec, sizes, device=torch.device("cpu"))
        painted = paint_rois_to_voxels(roi[:, 0], spec, n_voxels=4)
        assert painted.tolist() == [2.0, 2.0, 15.0, 15.0]
