"""The InstaGLM engine: one run, one events file, every column reported alike.

The invariants here are the ones that would make the teaching wrong rather than
merely slow. A beta has to be the joint-model beta, because the whole promise is
that a motion column and a condition column are read the same way. The extra
sums of squares have to equal what refitting without the column would actually
cost, because that identity is the only reason the partial maps are free. And a
design the user made rank deficient -- polort 0 plus a constant ortvec, which is
one click away -- has to produce a finite map rather than a screenful of
infinities.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import instaglm as ig

CPU = torch.device("cpu")
TR = 2.0
N_TIME = 60
SHAPE = (6, 5, 4)


def _events_tsv(path, onsets_by_condition):
    lines = ["onset\tduration\ttrial_type"]
    for name, onsets in onsets_by_condition.items():
        lines += [f"{o:g}\t2\t{name}" for o in onsets]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def events(tmp_path):
    return _events_tsv(
        tmp_path / "sub-01_events.tsv",
        {"faces": [4, 28, 60, 84], "houses": [16, 44, 72, 100]},
    )


@pytest.fixture
def run():
    """A 4-D run with a real baseline, so percent signal change means something."""
    rng = np.random.default_rng(3)
    data = 100.0 + rng.normal(scale=1.0, size=(*SHAPE, N_TIME)).astype(np.float32)
    return data


@pytest.fixture
def prepared(run):
    return ig.prepare(
        run,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr=TR,
        mask=np.ones(SHAPE, dtype=bool),
        device=CPU,
    )


def _task(events, *, basis="spmg1", **kwargs):
    parsed = ig.read_events(events, n_time=N_TIME, tr=TR)
    curves, suffixes = ig.hrf_bases(basis, microtime_dt=0.1, device=CPU, **kwargs)
    return parsed, ig.task_columns(
        parsed, n_time=N_TIME, tr=TR, curves=curves, suffixes=suffixes, device=CPU
    )


# -- reading timing ---------------------------------------------------------


def test_a_bids_events_file_becomes_conditions(events):
    parsed = ig.read_events(events, n_time=N_TIME, tr=TR)
    assert parsed.labels == ["faces", "houses"]
    assert parsed.n_events == 8
    assert parsed.durations == [2.0, 2.0]


def test_an_afni_timing_file_is_one_condition_named_for_itself(tmp_path):
    path = tmp_path / "faces.1D"
    path.write_text("4 28 60 84\n")
    parsed = ig.read_events(path, n_time=N_TIME, tr=TR)
    assert parsed.labels == ["faces"]
    assert parsed.n_events == 4


def test_onsets_past_the_end_of_the_run_are_refused(tmp_path):
    """The usual cause is a wrong TR, and the symptom downstream is an empty
    map that reads as a bad dataset rather than a bad argument."""
    path = _events_tsv(tmp_path / "late_events.tsv", {"faces": [500, 600]})
    with pytest.raises(ValueError, match="after the end of the run"):
        ig.read_events(path, n_time=N_TIME, tr=TR)


# -- the design -------------------------------------------------------------


def test_columns_are_grouped_and_ordered_task_first(events):
    parsed, (task, labels) = _task(events)
    ort = np.arange(N_TIME, dtype=float)[:, None] / N_TIME
    model = ig.build_model(
        n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2, ort=ort, ort_labels=["roll"]
    )
    assert [c.group for c in model.columns] == [ig.TASK] * 2 + [ig.ORT] + [ig.DRIFT] * 3
    assert model.labels[:3] == ("faces", "houses", "roll")


def test_a_derivative_basis_names_its_columns_with_the_condition(events):
    _parsed, (task, labels) = _task(events, basis="spmg2")
    assert labels == ["faces", "faces'", "houses", "houses'"]
    assert task.shape[1] == 4


def test_ortvec_derivatives_are_backward_differences(events):
    ort = np.cumsum(np.ones((N_TIME, 1)), axis=0)
    model = ig.build_model(n_time=N_TIME, tr=TR, ort=ort, ort_labels=["roll"], ort_derivatives=True)
    deriv = model.matrix[:, model.index_of("roll'")]
    assert deriv[0] == 0.0
    assert np.allclose(deriv[1:], 1.0)


def test_an_empty_model_is_an_error():
    with pytest.raises(ValueError, match="empty model"):
        ig.build_model(n_time=N_TIME, tr=TR, polort=-1)


def test_a_block_of_the_wrong_length_is_an_error():
    with pytest.raises(ValueError, match="but the run has"):
        ig.build_model(n_time=N_TIME, tr=TR, ort=np.zeros((N_TIME + 3, 1)))


# -- the fit ----------------------------------------------------------------


def test_betas_recover_a_planted_response(events, run):
    """The headline: put a known amount of a condition into a voxel and get it
    back, in the units the map is drawn in."""
    parsed, (task, labels) = _task(events)
    truth = 2.5
    data = run.copy()
    data[2, 2, 2, :] += truth * task[:, 0].astype(np.float32)
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2)
    fit = ig.fit_model(prepared, model, device=CPU)

    row = prepared.row((2, 2, 2))
    assert float(fit.betas[0, row]) == pytest.approx(truth, abs=0.2)
    assert float(fit.tstats[0, row]) > 5.0


def test_nuisance_betas_are_the_joint_model_ones_not_marginal(run):
    """The reason this does not call ``glm/core.py:fit_glm``.

    That fit orthogonalises the task block against nuisance first, which hands
    the nuisance columns everything the two share. Here a correlated ortvec and
    a correlated task column must split the shared variance the way OLS does,
    or "how much PSC does motion carry" is answered with the task's share too.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(size=N_TIME)
    b = 0.8 * a + 0.6 * rng.normal(size=N_TIME)  # deliberately collinear
    design = np.column_stack([a, b])
    truth = np.array([3.0, -1.5])
    data = np.zeros((*SHAPE, N_TIME), dtype=np.float32)
    data[...] = 100.0 + (design @ truth).astype(np.float32)

    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(
        n_time=N_TIME,
        tr=TR,
        task=a[:, None],
        task_labels=["cond"],
        ort=b[:, None],
        ort_labels=["roll"],
        polort=0,
    )
    fit = ig.fit_model(prepared, model, device=CPU)
    row = prepared.row((0, 0, 0))
    assert float(fit.betas[0, row]) == pytest.approx(truth[0], abs=1e-3)
    assert float(fit.betas[1, row]) == pytest.approx(truth[1], abs=1e-3)


def test_unique_sums_of_squares_equal_what_refitting_without_the_column_costs(events, run):
    """The identity the free partial-R² maps rest on: dropping column j raises
    RSS by ``b_j² / (X'X)⁻¹_jj``. If this drifts, every partial map is wrong and
    nothing else in the suite would notice."""
    parsed, (task, labels) = _task(events)
    data = run.copy()
    data[1, 1, 1, :] += 3.0 * task[:, 0].astype(np.float32)
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=3)
    fit = ig.fit_model(prepared, model, device=CPU)
    row = prepared.row((1, 1, 1))

    y = prepared.y[row].numpy().astype(np.float64)
    full_rss = float(fit.rss[row])
    for drop in range(model.n_columns):
        keep = [j for j in range(model.n_columns) if j != drop]
        reduced = model.matrix[:, keep]
        resid = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        expected = float(resid @ resid) - full_rss
        assert float(fit.ss_unique[drop, row]) == pytest.approx(expected, rel=1e-3, abs=1e-6)


def test_the_task_block_sum_of_squares_matches_dropping_the_whole_block(events, run):
    parsed, (task, labels) = _task(events)
    data = run.copy()
    data[1, 1, 1, :] += 3.0 * task[:, 0].astype(np.float32)
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=3)
    fit = ig.fit_model(prepared, model, device=CPU)
    row = prepared.row((1, 1, 1))

    y = prepared.y[row].numpy().astype(np.float64)
    nuisance = model.matrix[:, model.indices(ig.DRIFT)]
    resid = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    expected = float(resid @ resid) - float(fit.rss[row])
    assert fit.task_ss is not None
    assert float(fit.task_ss[row]) == pytest.approx(expected, rel=1e-3)


def test_r2_is_about_the_mean_and_a_drift_only_model_explains_the_drift():
    """Stepping polort up is the first thing anyone does in this mode, so the
    thing it is supposed to improve has to actually improve."""
    trend = np.linspace(0.0, 20.0, N_TIME, dtype=np.float32)
    data = np.zeros((*SHAPE, N_TIME), dtype=np.float32) + 100.0 + trend
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    flat = ig.fit_model(prepared, ig.build_model(n_time=N_TIME, tr=TR, polort=0), device=CPU)
    linear = ig.fit_model(prepared, ig.build_model(n_time=N_TIME, tr=TR, polort=1), device=CPU)
    row = prepared.row((0, 0, 0))
    assert float(flat.r2[row]) == pytest.approx(0.0, abs=1e-4)
    assert float(linear.r2[row]) > 0.999


def test_a_rank_deficient_design_still_produces_a_finite_map(run):
    """polort 0 beside a constant ortvec is one click away and exactly the
    combination [[Block-diagonal nuisance]] warns about. It must degrade to a
    minimum-norm answer, not to infinities on screen."""
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU)
    model = ig.build_model(
        n_time=N_TIME, tr=TR, ort=np.ones((N_TIME, 1)), ort_labels=["flat"], polort=0
    )
    fit = ig.fit_model(prepared, model, device=CPU)
    assert fit.rank == 1
    assert fit.dof == N_TIME - 1
    assert np.isfinite(fit.volume("t", column=0)).all()
    assert np.isfinite(fit.volume("beta", column=0)).all()


# -- maps -------------------------------------------------------------------


def test_swing_and_per_unit_psc_differ_by_the_regressors_own_excursion(events, run):
    parsed, (task, labels) = _task(events)
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2)
    fit = ig.fit_model(prepared, model, device=CPU)
    swing = fit.values("beta", column=0, psc="swing")
    per_unit = fit.values("beta", column=0, psc="per unit")
    assert torch.allclose(swing, per_unit * model.columns[0].swing, atol=1e-5)


def test_every_declared_map_is_a_finite_volume_of_the_right_shape(events, run):
    parsed, (task, labels) = _task(events)
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2)
    fit = ig.fit_model(prepared, model, device=CPU)
    for kind in ig.MAPS:
        vol = fit.volume(kind, column=0)
        assert vol.shape == SHAPE, kind
        assert np.isfinite(vol).all(), kind


def test_a_map_is_zero_outside_the_mask(events, run):
    parsed, (task, labels) = _task(events)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:3] = True
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=mask, device=CPU)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2)
    fit = ig.fit_model(prepared, model, device=CPU)
    assert prepared.n_voxels == int(mask.sum())
    assert np.all(fit.volume("R2")[~mask] == 0.0)


def test_an_unknown_map_names_the_ones_that_exist(events, prepared):
    parsed, (task, labels) = _task(events)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=1)
    fit = ig.fit_model(prepared, model, device=CPU)
    with pytest.raises(ValueError, match="unique R2"):
        fit.values("nonsense")


# -- the graph lines --------------------------------------------------------


def test_the_decomposition_adds_back_up_to_the_data(events, run):
    parsed, (task, labels) = _task(events)
    data = run.copy()
    data[2, 2, 2, :] += 2.0 * task[:, 0].astype(np.float32)
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=2)
    fit = ig.fit_model(prepared, model, device=CPU)
    lines = fit.decompose((2, 2, 2), column=0)
    # signal is data minus the nuisance fit, and fit is the same subtraction
    # applied to the model, so their difference is the residual exactly.
    assert np.allclose(lines["signal"] - lines["fit"], lines["resid"], atol=1e-3)


def test_the_nuisance_projected_line_keeps_the_voxels_baseline(events, run):
    """Otherwise it drops to zero and floats below the data it sits on."""
    parsed, (task, labels) = _task(events)
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=3)
    fit = ig.fit_model(prepared, model, device=CPU)
    lines = fit.decompose((1, 2, 3), column=0)
    assert lines["signal"].mean() == pytest.approx(lines["data"].mean(), abs=1e-2)
    assert lines["column"].mean() == pytest.approx(lines["data"].mean(), abs=1e-2)


def test_a_voxel_outside_the_mask_has_no_lines(events, run):
    parsed, (task, labels) = _task(events)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[0, 0, 0] = True
    prepared = ig.prepare(run, affine=np.eye(4), tr=TR, mask=mask, device=CPU)
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=1)
    fit = ig.fit_model(prepared, model, device=CPU)
    assert fit.decompose((3, 3, 3)) == {}


# -- HRF shape --------------------------------------------------------------


def test_stepping_the_library_index_changes_the_shape_not_the_scale():
    early, _ = ig.hrf_bases("library", microtime_dt=0.1, index=0, device=CPU)
    late, _ = ig.hrf_bases("library", microtime_dt=0.1, index=ig.library_size() - 1, device=CPU)
    assert not torch.allclose(early, late)
    assert float(early.abs().max()) == pytest.approx(float(late.abs().max()), rel=1e-6)


def test_the_custom_hrf_peak_follows_the_delay_slider():
    early, _ = ig.hrf_bases("custom", microtime_dt=0.1, delay=4.0, device=CPU)
    late, _ = ig.hrf_bases("custom", microtime_dt=0.1, delay=9.0, device=CPU)
    assert int(late.argmax()) > int(early.argmax())
    assert float(early.max()) == pytest.approx(1.0, rel=1e-6)


def test_the_derivative_bases_come_as_a_set():
    for basis, n in (("spmg1", 1), ("spmg2", 2), ("spmg3", 3)):
        curves, suffixes = ig.hrf_bases(basis, microtime_dt=0.1, device=CPU)
        assert curves.shape[0] == n
        assert len(suffixes) == n


def test_an_unknown_basis_is_an_error():
    with pytest.raises(ValueError, match="unknown HRF basis"):
        ig.hrf_bases("wavelet", microtime_dt=0.1, device=CPU)


# -- noise pool -------------------------------------------------------------


def test_the_noise_pool_is_bright_voxels_the_task_does_not_explain(events):
    parsed, (task, labels) = _task(events)
    rng = np.random.default_rng(5)
    data = rng.normal(scale=1.0, size=(*SHAPE, N_TIME)).astype(np.float32)
    data[:3] += 1000.0  # bright half
    data[3:] += 10.0  # dark half
    data[0, :, :, :] += 30.0 * task[:, 0].astype(np.float32)  # bright and task driven

    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=1)
    fit = ig.fit_model(prepared, model, device=CPU)
    selected = np.zeros(SHAPE, dtype=bool)
    selected[prepared.mask] = ig.noise_pool(fit, brightness=0.5).numpy()

    assert not selected[3:].any(), "dark voxels are air and carry no physiology"
    assert not selected[0].any(), "task-driven voxels must not enter the pool"
    assert selected[1:3].mean() > 0.5


def test_noise_pcs_come_back_as_run_length_columns(events):
    parsed, (task, labels) = _task(events)
    rng = np.random.default_rng(9)
    data = 100.0 + rng.normal(size=(*SHAPE, N_TIME)).astype(np.float32)
    prepared = ig.prepare(
        data, affine=np.eye(4), tr=TR, mask=np.ones(SHAPE, dtype=bool), device=CPU
    )
    model = ig.build_model(n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=1)
    fit = ig.fit_model(prepared, model, device=CPU)
    pool = ig.noise_pool(fit, brightness=0.2, f_ceiling=1e9)
    pcs, ratios = ig.noise_pcs(prepared, pool, model.matrix, 3, device=CPU)
    assert pcs.shape == (N_TIME, 3)
    assert ratios.shape == (3,)

    with_pcs = ig.build_model(
        n_time=N_TIME, tr=TR, task=task, task_labels=labels, polort=1, pcs=pcs
    )
    assert with_pcs.labels[-5:-2] == ("PC#0", "PC#1", "PC#2")


def test_asking_for_no_pcs_returns_no_columns(prepared):
    empty, ratios = ig.noise_pcs(
        prepared,
        torch.ones(prepared.n_voxels, dtype=torch.bool),
        np.zeros((N_TIME, 0)),
        0,
        device=CPU,
    )
    assert empty.shape == (N_TIME, 0)
    assert ratios.size == 0


# -- preparation ------------------------------------------------------------


def test_preparing_a_3d_dataset_is_an_error():
    with pytest.raises(ValueError, match="one 4-D run"):
        ig.prepare(np.zeros(SHAPE, dtype=np.float32), affine=np.eye(4), tr=TR, device=CPU)


def test_an_empty_mask_is_an_error(run):
    with pytest.raises(ValueError, match="mask is empty"):
        ig.prepare(run, affine=np.eye(4), tr=TR, mask=np.zeros(SHAPE, dtype=bool), device=CPU)


def test_a_design_of_the_wrong_length_is_an_error(prepared):
    model = ig.build_model(n_time=N_TIME + 5, tr=TR, polort=2)
    with pytest.raises(ValueError, match="but the run has"):
        ig.fit_model(prepared, model, device=CPU)
