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
    _build_nested_fold_solvers,
    _build_trial_ops,
    _cellspace_partials,
    _nested_partials,
    _orthonormal_contrasts,
    _partition_stats,
    _solve_gammas,
    _trialspace_partials,
    build_factor_design,
    build_repeat_folds,
    build_run_exchange,
    build_stat_specs,
    cell_labels,
    derive_repeat_index,
    detect_run_nesting,
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


def test_nested_gamma_never_reads_its_outer_test_responses():
    factors, rep, _ = make_crossed_table(n_stim=3, n_task=3, n_rep=3, seed=4)
    design = build_factor_design(factors)
    band_order = list(design.band_order)
    bands = {name: mat.float() for name, mat in design.bands.items()}
    slices = {}
    offset = 0
    for name in band_order:
        width = bands[name].shape[1]
        slices[name] = slice(offset, offset + width)
        offset += width

    folds_np, _ = build_repeat_folds(rep)
    folds = [(torch.as_tensor(train), torch.as_tensor(test)) for train, test in folds_np]
    inner = _build_nested_fold_solvers(bands, band_order, folds)
    y = torch.randn(5, len(rep), generator=torch.Generator().manual_seed(8))
    active = list(range(len(band_order)))

    before = _solve_gammas(*_nested_partials(y, bands, band_order, slices, inner[0]), active)
    contaminated = y.clone()
    contaminated[:, folds[0][1]] += 1000 * torch.randn_like(contaminated[:, folds[0][1]])
    after = _solve_gammas(
        *_nested_partials(contaminated, bands, band_order, slices, inner[0]), active
    )

    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_fold_builder_flags_run_leak():
    factors, rep, run = make_crossed_table()
    design = build_factor_design(factors)
    cells = cell_labels(design)
    _, diag = build_repeat_folds(rep, run, cell=cells)
    assert diag["run_locality_ok"]
    assert diag["n_folds"] == N_REP

    leaky = np.zeros_like(run)  # every trial in one run -> every cell repeats within it
    _, diag_leak = build_repeat_folds(rep, leaky, cell=cells)
    assert not diag_leak["run_locality_ok"]
    assert diag_leak["run_leaks"]


def test_interleaved_runs_are_not_a_leak():
    """A run holding many cells is the normal design, not a violation.

    Without per-cell identity the check degrades to "does any run appear on both sides",
    which every interleaved design fails by construction even though train and test share
    no cell mean. This is the case that made the guard fire on real data.
    """
    factors, rep, _ = make_crossed_table()
    design = build_factor_design(factors)
    cells = cell_labels(design)
    n = len(rep)
    # Chop each repeat block into 4 runs: every run holds many cells, but no cell's
    # repeats ever share a run.
    per_rep = n // N_REP
    run = np.array([f"{rep[i]}/{(i % per_rep) // (per_rep // 4)}" for i in range(n)])
    _, diag = build_repeat_folds(rep, run, cell=cells)
    assert diag["run_locality_ok"], diag["run_leaks"]


def test_partition_warns_but_proceeds_on_run_leak(capsys):
    """Repeats inside a run are a legitimate design; they inflate R2, they do not abort."""
    factors, rep, _ = make_crossed_table()
    betas = synth_betas(factors, a=np.zeros((4, N_STIM)), noise=1.0)
    res = partition_variance(betas, factors, repeat=rep, run=np.zeros_like(rep), verbose=True)
    assert res.diagnostics["run_locality_ok"] is False
    assert "repeats inside one run" in capsys.readouterr().out


def test_partition_rejects_run_leak_under_strict():
    factors, rep, _ = make_crossed_table()
    betas = synth_betas(factors, a=np.zeros((4, N_STIM)), noise=1.0)
    with pytest.raises(ValueError, match="leaks"):
        partition_variance(
            betas,
            factors,
            repeat=rep,
            run=np.zeros_like(rep),
            strict_run_locality=True,
            verbose=False,
        )


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
    # The unmasked selection is still exposed, and it is what would have produced the
    # artifact: the great majority of these voxels read as a confident "additive" 0.
    # It is not *all* of them, because the rank sweep carries the same per-band shrinkage
    # as the reported models -- so a real rank-1 interaction occasionally survives even at
    # this SNR. That is sensitivity, not invention: these voxels genuinely have a rank-1
    # interaction. The invention case is covered by test_additive_only_data_selects_rank_zero
    # and test_low_snr_additive_truth_does_not_invent_rank.
    assert res.rank_e_raw is not None
    assert (res.rank_e_raw[n_each:] == 0).float().mean() > 0.8
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


def test_low_snr_additive_truth_does_not_invent_rank():
    """The one-sided error property: misses are acceptable, inventions are not.

    Bug of record: once the rank sweep was evaluated under the fitted per-band gammas,
    voxels with no interaction had gamma_interaction shrink to 0, every rank predicted
    identically, and the curve went flat to within float32 noise. A bare argmax over a flat
    curve returned whatever the rounding favoured -- 8/20 additive voxels landed at rank
    2-6 off improvements of 1e-5. The parsimony rule and the detection floor fix that.
    """
    rng = np.random.default_rng(31)
    factors, rep, run = make_crossed_table()
    n_vox = 120
    a = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
    for noise in (0.5, 2.0, 6.0):
        betas = synth_betas(factors, a=a, b=b, noise=noise, seed=32)
        res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
        invented = (res.rank_e > 0).float().mean()
        assert invented == 0.0, f"noise={noise}: invented rank in {invented:.1%} of voxels"

    # The detection floor carries this on its own wherever the SNR mask is switched off,
    # up to the point where the mask is the guard that matters. Below ncsnr ~0.2 the two
    # are not interchangeable, which is why the mask is on by default.
    betas = synth_betas(factors, a=a, b=b, noise=2.0, seed=32)
    unmasked = partition_variance(
        betas, factors, repeat=rep, run=run, min_ncsnr_for_rank=0.0, verbose=False
    )
    assert (unmasked.rank_e > 0).float().mean() < 0.02


def test_rank_curve_endpoint_is_the_full_model():
    """The sweep and the reported models must be the same predictor family.

    Evaluating the rank curve under different shrinkage than the models it is read against
    made the curve systematically noisier than R2(M_full), which is the comparison a reader
    makes by eye. Anchoring the top of the curve to M_full exactly removes that gap.
    """
    rng = np.random.default_rng(33)
    factors, rep, run = make_crossed_table()
    n_vox = 30
    a = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
    u = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    v = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
    betas = synth_betas(
        factors, a=a, b=b, e=0.9 * u[:, :, None] * v[:, None, :], noise=1.0, seed=34
    )

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
    assert res.rank_r2 is not None
    torch.testing.assert_close(res.rank_r2[:, -1], res.r2["M_full"], atol=1e-4, rtol=0)


def test_gain_alignment_separates_gain_from_reorganization():
    """A multiplicative gain is rank 1 ALIGNED with the main effect; reorganisation is not.

    m = mu + a_s*(1 + g_t) leaves the interaction a_s*g_t, whose left singular vector is
    parallel to the stimulus main effect. A rank-1 interaction built from an unrelated
    pattern has no such alignment. Both are "rank 1", so rank_E alone cannot tell them
    apart -- this is the map that can.
    """
    rng = np.random.default_rng(35)
    factors, rep, run = make_crossed_table()
    n_each = 30
    a = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    g = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    w = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)

    e = np.empty((2 * n_each, N_STIM, N_TASK))
    e[:n_each] = 1.2 * a[:n_each, :, None] * g[:n_each, None, :]  # gain on the stim profile
    e[n_each:] = 1.2 * w[n_each:, :, None] * g[n_each:, None, :]  # unrelated pattern
    betas = synth_betas(factors, a=a, b=b, e=e, noise=1.0, seed=36)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
    align = res.gain_alignment["stim"]
    resolved = res.rank_e >= 1
    assert resolved.float().mean() > 0.8, "both blocks are a strong rank-1 interaction"
    gain_block = align[:n_each][resolved[:n_each]]
    reorg_block = align[n_each:][resolved[n_each:]]
    assert gain_block.median() > 0.85
    assert reorg_block.median() < 0.5
    # Zeroed where no interaction was resolved, so the map never shows noise alignment.
    assert (align[~resolved] == 0).all()


def test_nuclear_sweep_tracks_the_interaction_and_stays_nonnegative():
    rng = np.random.default_rng(37)
    factors, rep, run = make_crossed_table()
    n_each = 25
    a = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    u = centered(rng.normal(0, 1.0, size=(2 * n_each, N_STIM)), axis=1)
    v = centered(rng.normal(0, 1.0, size=(2 * n_each, N_TASK)), axis=1)
    e = np.zeros((2 * n_each, N_STIM, N_TASK))
    e[:n_each] = 1.0 * u[:n_each, :, None] * v[:n_each, None, :]  # second block additive
    betas = synth_betas(factors, a=a, b=b, e=e, noise=1.0, seed=38)

    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
    assert res.nuclear_gain is not None and res.nuclear_tau is not None
    assert (res.nuclear_gain >= 0).all(), "referenced to its own no-interaction endpoint"
    assert res.nuclear_gain[:n_each].median() > 0.1
    assert res.nuclear_gain[n_each:].median() < 0.01
    # A strong clean interaction needs little shrinkage; an absent one is thresholded away.
    assert res.nuclear_tau[:n_each].median() < res.nuclear_tau[n_each:].median()

    off = partition_variance(betas, factors, repeat=rep, run=run, n_nuclear_taus=0, verbose=False)
    assert off.nuclear_gain is None


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
    fast = _partition_stats(*_cellspace_partials(betas, ops), build_stat_specs(design))

    slow = partition_variance(
        betas, factors, repeat=rep, run=run, nested_gamma=False, verbose=False
    )

    torch.testing.assert_close(fast["unique_a"], slow.unique["stim"], atol=1e-4, rtol=0)
    torch.testing.assert_close(fast["unique_b"], slow.unique["task"], atol=1e-4, rtol=0)
    torch.testing.assert_close(fast["interaction"], slow.interaction, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        fast["band_stim:task"], slow.band_unique["stim:task"], atol=1e-4, rtol=0
    )


def test_nested_permutation_observed_matches_reported_estimator():
    betas, factors, rep, run = _perm_data(0.5, n_vox=8, seed=19)
    reported = partition_variance(
        betas, factors, repeat=rep, run=run, nested_gamma=True, verbose=False
    )
    perm = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("unique_stim", "unique_task", "interaction"),
        n_perms=1,
        seed=2,
        nested_gamma=True,
        device=torch.device("cpu"),
        verbose=False,
    )

    torch.testing.assert_close(
        perm.observed["unique_stim"], reported.unique["stim"], atol=2e-4, rtol=0
    )
    torch.testing.assert_close(
        perm.observed["unique_task"], reported.unique["task"], atol=2e-4, rtol=0
    )
    torch.testing.assert_close(
        perm.observed["interaction"], reported.interaction, atol=2e-4, rtol=0
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


def test_trialspace_engine_matches_cellspace_when_balanced():
    """The general engine is only trustworthy if it reproduces the fast one where both apply.

    Same folds, same design, same data: the cell-space compression is exact under balance,
    so any disagreement means the trial-space path is fitting a different model.
    """
    betas, factors, rep, run = _perm_data(0.0, n_vox=8, seed=21)
    design = build_factor_design(factors)
    folds, _ = build_repeat_folds(rep, run, cell=cell_labels(design))
    device = torch.device("cpu")

    specs = build_stat_specs(design)
    cell_stats = _partition_stats(
        *_cellspace_partials(betas, _build_cell_ops(design, folds, device)), specs
    )
    trial_stats = _partition_stats(
        *_trialspace_partials(betas, _build_trial_ops(design, folds, device)), specs
    )
    for key in ("unique_a", "unique_b", "interaction", "band_stim", "band_stim:task"):
        torch.testing.assert_close(cell_stats[key], trial_stats[key], atol=2e-4, rtol=2e-3)


def test_permutation_runs_on_unbalanced_design(capsys):
    """Dropping trials leaves unequal repeats; inference falls back, it does not refuse."""
    betas, factors, rep, run = _perm_data(0.0, n_vox=10, seed=16)
    keep = np.ones(len(factors["stim"]), dtype=bool)
    keep[:40] = False
    sub = {k: v[keep] for k, v in factors.items()}
    assert not build_factor_design(sub).balanced

    res = permutation_test(
        betas[:, keep],
        sub,
        repeat=rep[keep],
        run=run[keep],
        n_perms=8,
        seed=3,
        verbose=True,
    )
    assert res.diagnostics["engine"] == "trial-space"
    assert "not balanced" in capsys.readouterr().out
    for key in ("unique_a", "unique_b", "interaction"):
        p = res.p_uncorrected[key]
        assert p.shape == (10,)
        assert bool(((p > 0) & (p <= 1)).all())
        # A degenerate null (every permutation identical) would make the test vacuous.
        assert res.null_max[key].std() > 0


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


@pytest.mark.slow
def test_unbalanced_null_stays_calibrated():
    """The fallback engine has to be a real test, not just one that runs.

    Under a null with no effect at all, uncorrected p must stay near nominal and no unit
    may survive FWE. Measured at 12% of trials randomly dropped (off-diagonal Gram ~8e-3,
    1-3 repeats per cell): 0.050 / 0.050 / 0.035 at alpha 0.05 and FWE 0.000 for the three
    statistics -- the imbalance costs orthogonality, not calibration.
    """
    rng = np.random.default_rng(0)
    factors, rep, run = make_crossed_table()
    keep = rng.random(len(rep)) > 0.12
    sub = {k: v[keep] for k, v in factors.items()}
    assert not build_factor_design(sub).balanced

    y = torch.as_tensor(rng.normal(0, 1, size=(100, int(keep.sum()))), dtype=torch.float32)
    res = permutation_test(
        y, sub, repeat=rep[keep], run=run[keep], n_perms=200, seed=1, verbose=False
    )
    for key in ("unique_a", "unique_b", "interaction"):
        p = res.p_uncorrected[key].numpy()
        assert (p <= 0.05).mean() < 0.12, f"{key} anticonservative: {(p <= 0.05).mean()}"
        assert (res.p_fwe[key].numpy() <= 0.05).mean() < 0.05


# ---------------------------------------------------------------------------
# Run-nested factors
# ---------------------------------------------------------------------------


def make_run_nested_table(n_task=3, n_stim=16, n_rep=3, seed=0):
    """One task per run: task is nested in run, stimulus varies within it.

    The shape Logan's follow-up study takes -- 9 runs, 3 tasks locked to run, 16 stimuli,
    3 repeats. Still exhaustively crossed and balanced; the problem is confounding, not
    the algebra.
    """
    rng = np.random.default_rng(seed)
    task, stim, run, rep = [], [], [], []
    r = 0
    for k in range(n_rep):
        for t in range(n_task):
            for s in rng.permutation(n_stim):  # order randomised so drift does not alias
                task.append(f"T{t}")
                stim.append(f"S{s}")
                run.append(f"run{r:02d}")
                rep.append(k)
            r += 1
    return (
        {"stim": np.array(stim), "task": np.array(task)},
        np.array(rep),
        np.array(run),
    )


def _run_nested_betas(task_effect, run_noise, seed, n_vox=60):
    rng = np.random.default_rng(seed)
    factors, rep, run = make_run_nested_table(seed=seed)
    ti = np.array([int(x[1:]) for x in factors["task"]])
    si = np.array([int(x[1:]) for x in factors["stim"]])
    ri = np.array([int(x[3:]) for x in run])
    n_stim, n_task, n_run = si.max() + 1, ti.max() + 1, ri.max() + 1
    vs = rng.normal(size=(n_vox, n_stim))
    vt = task_effect * rng.normal(size=(n_vox, n_task))
    y = (
        vs[:, si]
        + vt[:, ti]
        + run_noise * rng.normal(size=(n_vox, n_run))[:, ri]
        + 0.5 * rng.normal(size=(n_vox, len(si)))
    )
    return torch.as_tensor(y, dtype=torch.float32), factors, rep, run


def test_run_nesting_is_detected_and_reported(capsys):
    betas, factors, rep, run = _run_nested_betas(0.0, 1.0, seed=3)
    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=True)

    nested = res.diagnostics["factors_nested_in_run"]
    assert set(nested) == {"task"}, "stimulus varies within run; task does not"
    assert set(nested["task"].values()) == {3}, "three runs per task"
    out = capsys.readouterr().out
    assert "NESTED" in out and "unique_task" in out

    # Nothing is nested when both factors vary within a run.
    crossed, c_rep, c_run = make_crossed_table(n_stim=6, n_task=5, n_rep=3)
    assert detect_run_nesting(crossed, c_run) == {}
    assert detect_run_nesting(crossed, None) == {}


def test_additive_run_nuisance_lands_only_on_the_nested_factor():
    """The reason the interaction survives a run-locked design -- and its one caveat.

    A per-run offset is constant across stimulus, so in the additive decomposition it sits
    ENTIRELY in the task main effect: no part of it can reach the stimulus main effect or
    the interaction. That is what makes the interaction family the trustworthy half of a
    run-locked design, and the whole recommendation rests on it.

    The caveat is that these maps are R2, i.e. ratios. The offset inflates total variance,
    so it dilutes every map through the shared denominator even though it contaminates only
    one numerator. Absolute R2 therefore drops for stimulus and interaction alike; what is
    preserved is their RATIO, and everything derived from the interaction's shape.
    """
    rng = np.random.default_rng(6)
    clean, factors, rep, run = _run_nested_betas(0.0, 0.0, seed=5)
    si = np.array([int(x[1:]) for x in factors["stim"]])
    ti = np.array([int(x[1:]) for x in factors["task"]])
    ri = np.array([int(x[3:]) for x in run])
    n_vox = clean.shape[0]
    # Give it a real rank-1 interaction, so there is a ratio worth preserving.
    u = rng.normal(size=(n_vox, si.max() + 1))
    v = rng.normal(size=(n_vox, ti.max() + 1))
    clean = clean + torch.as_tensor(1.2 * u[:, si] * v[:, ti], dtype=torch.float32)

    offsets = torch.as_tensor(
        rng.normal(scale=2.0, size=(n_vox, ri.max() + 1)), dtype=torch.float32
    )
    contaminated = clean + offsets[:, ri]

    a = partition_variance(clean, factors, repeat=rep, run=run, nested_gamma=False, verbose=False)
    b = partition_variance(
        contaminated, factors, repeat=rep, run=run, nested_gamma=False, verbose=False
    )

    # Task soaks up the entire run-level offset...
    assert b.unique["task"].mean() > 2.0 * a.unique["task"].mean()
    # ...and dilutes the others only through the shared denominator, so their ratio holds.
    ratio_a = a.interaction / a.unique["stim"].clamp_min(1e-6)
    ratio_b = b.interaction / b.unique["stim"].clamp_min(1e-6)
    torch.testing.assert_close(ratio_a.median(), ratio_b.median(), atol=0.05, rtol=0)
    # Run nuisance costs SENSITIVITY on the interaction even though it cannot bias it:
    # a cell's repeats live in different runs, so the offset lands in within-cell variance,
    # which is what ncsnr calls noise. The ceiling drops and voxels fall below the rank
    # floor. What it does not do is change the answer where an answer is still available.
    assert (b.rank_e == -1).float().mean() > (a.rank_e == -1).float().mean()
    both = (a.rank_e >= 0) & (b.rank_e >= 0)
    assert both.any()
    assert (a.rank_e[both] == b.rank_e[both]).float().mean() > 0.9
    resolved = (a.rank_e >= 1) & (b.rank_e >= 1)  # alignment is zeroed below rank 1
    torch.testing.assert_close(
        a.gain_alignment["stim"][resolved],
        b.gain_alignment["stim"][resolved],
        atol=0.1,
        rtol=0,
    )


def test_whole_run_permutation_has_power_where_within_run_has_none():
    """Within-run permutation cannot test a run-nested factor -- at all.

    A run-constant effect is exactly invariant under permutation inside a run, so it
    survives into every permuted dataset, the null lands on the observed statistic, and
    the test detects nothing. Switching the exchangeability unit to the run fixes it.
    """
    betas, factors, rep, run = _run_nested_betas(1.0, 0.0, seed=7, n_vox=80)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("unique_b",),
        n_perms=200,
        seed=0,
        verbose=False,
    )
    assert res.diagnostics["permutation_scheme"]["unique_b"] == "whole_run"
    p_whole = res.p_uncorrected["unique_b"]
    assert (p_whole < 0.05).float().mean() > 0.8, "a real task effect must be detectable"

    # The interaction still varies within a run, so its null keeps the within-run scheme.
    res_i = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("interaction",),
        n_perms=50,
        seed=0,
        verbose=False,
    )
    assert res_i.diagnostics["permutation_scheme"]["interaction"] == "within_run"


def test_whole_run_permutation_controls_type_one_error():
    """Run-level noise must not read as a task effect once the error term is right."""
    betas, factors, rep, run = _run_nested_betas(0.0, 1.5, seed=11, n_vox=200)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("unique_b",),
        n_perms=400,
        seed=0,
        verbose=False,
    )
    p = res.p_uncorrected["unique_b"]
    assert (p < 0.05).float().mean() < 0.12, "nominal 5%, allowing Monte-Carlo slack"


def test_whole_run_permutation_needs_matching_runs():
    betas, factors, rep, run = _run_nested_betas(0.0, 0.0, seed=13, n_vox=5)
    keep = np.ones(len(run), dtype=bool)
    keep[0] = False  # one run is now a trial short
    with pytest.raises(ValueError, match="matching trial structure"):
        build_run_exchange(run[keep], factors["stim"][keep])


# ---------------------------------------------------------------------------
# Three or more factors
# ---------------------------------------------------------------------------


def make_three_factor_table(n_task=3, n_stim=16, n_noise=2, n_rep=3, seed=0):
    """Task locked to run, stimulus and noise level crossed within it.

    The shape of Logan's follow-up study: 9 runs, 3 tasks, 16 stimuli at 2 noise levels,
    3 repeats. 96 cells, 288 trials, still exhaustively crossed and balanced.
    """
    rng = np.random.default_rng(seed)
    task, stim, noise, run, rep = [], [], [], [], []
    r = 0
    for k in range(n_rep):
        for t in range(n_task):
            cells = [(s, z) for s in range(n_stim) for z in range(n_noise)]
            rng.shuffle(cells)
            for s, z in cells:
                task.append(f"T{t}")
                stim.append(f"S{s:02d}")
                noise.append(f"N{z}")
                run.append(f"run{r:02d}")
                rep.append(k)
            r += 1
    factors = {"stim": np.array(stim), "task": np.array(task), "noise": np.array(noise)}
    return factors, np.array(rep), np.array(run)


def test_three_factor_design_builds_every_band_orthogonally():
    factors, _, _ = make_three_factor_table()
    design = build_factor_design(factors)
    assert design.band_order == [
        "stim",
        "task",
        "noise",
        "stim:task",
        "stim:noise",
        "task:noise",
        "stim:task:noise",
    ]
    assert design.main_bands == ["stim", "task", "noise"]
    assert design.pair_bands == ["stim:task", "stim:noise", "task:noise"]
    assert design.balanced
    # 2^k - 1 bands, and their widths partition the cell space minus the intercept.
    assert sum(b.shape[1] for b in design.bands.values()) == 16 * 3 * 2 - 1
    assert design.max_offdiag < BALANCE_TOL


def _three_factor_betas(seed=0, n_vox=90, noise_sd=0.6):
    """Three blocks: additive, a noise-level GAIN on the stimulus profile, a task REORG."""
    rng = np.random.default_rng(seed)
    factors, rep, run = make_three_factor_table(seed=seed)
    ti = np.array([int(x[1:]) for x in factors["task"]])
    si = np.array([int(x[1:]) for x in factors["stim"]])
    zi = np.array([int(x[1:]) for x in factors["noise"]])
    n_stim, n_task, n_noise = si.max() + 1, ti.max() + 1, zi.max() + 1
    third = n_vox // 3

    vs = rng.normal(size=(n_vox, n_stim))
    vz = 0.8 * rng.normal(size=(n_vox, n_noise))
    sig = vs[:, si] + vz[:, zi]
    g = rng.normal(size=(n_vox, n_noise))
    sig[third : 2 * third] += 1.2 * (vs[third : 2 * third][:, si] * g[third : 2 * third][:, zi])
    w = rng.normal(size=(n_vox, n_stim))
    g2 = rng.normal(size=(n_vox, n_task))
    sig[2 * third :] += 1.2 * (w[2 * third :][:, si] * g2[2 * third :][:, ti])

    y = sig + noise_sd * rng.normal(size=(n_vox, len(si)))
    return torch.as_tensor(y, dtype=torch.float32), factors, rep, run, third


def test_three_factor_partition_attributes_each_effect_to_its_own_band():
    """The payoff of k factors: a noise-level gain and a task reorganisation separate.

    Collapsed into a single 32-level "stimulus" factor these are indistinguishable -- both
    read as "stimulus interacts with task". Split out, each lands in exactly one band.
    """
    betas, factors, rep, run, third = _three_factor_betas(seed=2)
    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)

    add, gain, reorg = slice(0, third), slice(third, 2 * third), slice(2 * third, None)
    bu = res.band_unique

    # Each planted effect shows up in its own band and nowhere else.
    assert bu["stim:noise"][gain].median() > 0.05
    assert bu["stim:noise"][add].median() < 0.01
    assert bu["stim:noise"][reorg].median() < 0.01
    assert bu["stim:task"][reorg].median() > 0.05
    assert bu["stim:task"][add].median() < 0.01
    assert bu["stim:task"][gain].median() < 0.01
    # No three-way term was planted, and none is invented.
    assert bu["stim:task:noise"].abs().median() < 0.01
    # Orthogonality still holds at k = 3, so there is nothing shared.
    assert res.shared.abs().median() < 0.02


def test_three_factor_gain_alignment_names_the_modulated_factor():
    betas, factors, rep, run, third = _three_factor_betas(seed=3)
    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
    gain = slice(third, 2 * third)

    align = res.pair_gain_alignment["stim:noise"]["stim"]
    resolved = res.pair_rank_e["stim:noise"][gain] >= 1
    assert resolved.float().mean() > 0.5
    # The planted effect IS a gain on the stimulus profile, so the leading singular vector
    # of stim:noise should sit on the stimulus main effect.
    assert align[gain][resolved].median() > 0.7

    # A 2-level factor has one contrast column, so any interaction it takes part in is
    # rank 1 by construction -- worth asserting, because it means rank_E carries no
    # information there and gain_align is the only informative output.
    assert res.diagnostics["max_rank_per_pair"]["stim:noise"] == 1
    assert res.diagnostics["max_rank_per_pair"]["task:noise"] == 1
    assert res.diagnostics["max_rank_per_pair"]["stim:task"] == 2


def test_three_factor_flat_aliases_are_empty_but_pair_dicts_are_not():
    """Above two factors "the" interaction is ambiguous, so the flat fields stay unset."""
    betas, factors, rep, run, _ = _three_factor_betas(seed=4, n_vox=12)
    res = partition_variance(betas, factors, repeat=rep, run=run, verbose=False)
    assert res.rank_e is None and res.rank_r2 is None and res.nuclear_gain is None
    assert res.gain_alignment == {}
    assert set(res.pair_rank_e) == {"stim:task", "stim:noise", "task:noise"}
    assert res.preference is None, "preference is a two-way ratio"

    two = partition_variance(
        betas,
        {k: factors[k] for k in ("stim", "task")},
        repeat=rep,
        run=run,
        verbose=False,
    )
    assert two.rank_e is not None and two.preference is not None
    torch.testing.assert_close(two.rank_e, two.pair_rank_e["stim:task"])


def test_three_factor_permutation_uses_the_right_scheme_per_statistic():
    betas, factors, rep, run, _ = _three_factor_betas(seed=5, n_vox=30)
    res = permutation_test(
        betas,
        factors,
        repeat=rep,
        run=run,
        statistics=("unique_stim", "unique_task", "band_stim:noise"),
        n_perms=30,
        seed=0,
        verbose=False,
    )
    scheme = res.diagnostics["permutation_scheme"]
    # Only the run-nested factor's own statistic changes exchangeability unit.
    assert scheme["unique_task"] == "whole_run"
    assert scheme["unique_stim"] == "within_run"
    assert scheme["band_stim:noise"] == "within_run"
    for key in ("unique_stim", "unique_task", "band_stim:noise"):
        assert res.p_uncorrected[key].shape == (30,)


def test_three_factor_cellspace_and_trialspace_engines_agree():
    betas, factors, rep, run, _ = _three_factor_betas(seed=6, n_vox=8)
    design = build_factor_design(factors)
    folds, _ = build_repeat_folds(rep, run, cell=cell_labels(design))
    device = torch.device("cpu")
    specs = build_stat_specs(design)
    keys = ("unique_stim", "interaction", "band_stim:noise", "band_stim:task:noise")

    cell = _partition_stats(
        *_cellspace_partials(betas, _build_cell_ops(design, folds, device)), specs, keys
    )
    trial = _partition_stats(
        *_trialspace_partials(betas, _build_trial_ops(design, folds, device)), specs, keys
    )
    for key in keys:
        torch.testing.assert_close(cell[key], trial[key], atol=2e-4, rtol=2e-3)
