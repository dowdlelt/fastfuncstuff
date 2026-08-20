"""
Tests for continuous TR-locked stimulus vectors (-stim_event_vec / -stim_vec).

The load path mirrors the NuisanceBlock tests (full-length vs per-run file),
but the important tests here are the convolution ones: a stim vector is *not*
an onset list, so it must not go anywhere near the per-event peak-normalising
path in convolve_hrf_microtime, and it must not be resampled.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastfuncstuff.design.hrf import get_spmg1_hrf
from fastfuncstuff.design.matrices import convolve_tr_locked
from fastfuncstuff.design.stim_vec import (
    STIM_VEC_MODS,
    add_stim_vec_arguments,
    apply_stim_vec_mod,
    build_stim_vec_design,
    collect_stim_vec_blocks,
    load_stim_vec_block,
    resolve_stim_vec_hrf,
    split_label_mod,
)

CPU = torch.device("cpu")
TR = 1.5
DT = 0.1
BINS = int(round(TR / DT))


def _hrf():
    return get_spmg1_hrf(microtime_dt=DT, stim_duration=0.0, normalize_peak=True, device=CPU)


def _microtime_reference(x_tr: np.ndarray, hrf: torch.Tensor, n_t: int) -> torch.Tensor:
    """Convolve the same input on the microtime grid as an impulse train.

    This is the operation convolve_tr_locked claims to be exactly equal to, so
    it is written out the long way here rather than reusing any of its code.
    """
    xm = torch.zeros(n_t * BINS, 1)
    xm[::BINS, 0] = torch.as_tensor(x_tr, dtype=torch.float32)
    conv = F.conv1d(xm.T.unsqueeze(0), hrf.flip(0).view(1, 1, -1), padding=len(hrf) - 1)
    return conv[0, :, : n_t * BINS].T


# ---------------------------------------------------------------------------
# convolve_tr_locked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("microtime_onset", [0, 5, BINS - 1])
def test_tr_convolution_equals_microtime_impulse_train(microtime_onset):
    """Decimating the HRF is exact, not an approximation -- at every phase.

    If this drifts, the stim-vector columns are sitting at a different sampling
    phase from the event columns in the same design.
    """
    n_t = 80
    hrf = _hrf()
    x = np.zeros(n_t, dtype=np.float32)
    x[10], x[41] = 1.0, -2.0  # signed, so any rectification shows up here

    got = convolve_tr_locked(
        torch.as_tensor(x).reshape(-1, 1),
        hrf,
        n_t,
        tr=TR,
        microtime_dt=DT,
        microtime_onset=microtime_onset,
        normalize_peak=False,
        device=CPU,
    )
    want = _microtime_reference(x, hrf, n_t)[microtime_onset::BINS][:n_t]
    assert torch.allclose(got, want, atol=1e-6)


def test_sign_changes_are_not_treated_as_separate_events():
    """The reason this primitive exists.

    convolve_hrf_microtime segments a column into contiguous non-zero regions
    and rescales each to peak 1.0. On an oscillating background every zero
    crossing would start a new "event" and each half-cycle would be
    independently renormalised, flattening the amplitude structure. Here a
    doubled input must produce a doubled response.
    """
    n_t = 120
    hrf = _hrf()
    t = np.arange(n_t) * TR
    small = np.sin(2 * np.pi * t / 30.0).astype(np.float32)
    big = 2.0 * small

    kw = dict(tr=TR, microtime_dt=DT, normalize_peak=False, device=CPU)
    y_small = convolve_tr_locked(torch.as_tensor(small).reshape(-1, 1), hrf, n_t, **kw)
    y_big = convolve_tr_locked(torch.as_tensor(big).reshape(-1, 1), hrf, n_t, **kw)
    assert torch.allclose(y_big, 2.0 * y_small, atol=1e-6)
    # And the response still oscillates rather than being clipped to one sign.
    assert y_small.min() < 0 < y_small.max()


def test_hrf_tail_does_not_cross_a_run_boundary():
    n_t, run_starts = 100, [0, 50]
    hrf = _hrf()
    x = np.zeros(n_t, dtype=np.float32)
    x[45] = 1.0  # late in run 1; its tail would reach well into run 2

    y = convolve_tr_locked(
        torch.as_tensor(x).reshape(-1, 1),
        hrf,
        n_t,
        tr=TR,
        microtime_dt=DT,
        run_starts=run_starts,
        normalize_peak=False,
        device=CPU,
    )
    assert y[:50].abs().sum() > 0
    assert y[50:].abs().max() == 0.0


def test_shapes_pass_through_without_interpolation():
    """A delta must stay a delta: no smearing into the neighbouring TR.

    Linear upsampling to microtime would spread a one-TR spike across two TRs;
    the response to a lone spike here must be exactly the decimated HRF.
    """
    n_t = 60
    hrf = _hrf()
    x = np.zeros(n_t, dtype=np.float32)
    x[7] = 1.0
    y = convolve_tr_locked(
        torch.as_tensor(x).reshape(-1, 1),
        hrf,
        n_t,
        tr=TR,
        microtime_dt=DT,
        normalize_peak=False,
        device=CPU,
    )
    hrf_tr = hrf[::BINS]
    n = min(len(hrf_tr), n_t - 7)
    assert torch.allclose(y[7 : 7 + n, 0], hrf_tr[:n], atol=1e-6)
    assert y[:7].abs().max() == 0.0  # strictly causal


def test_peak_normalisation_is_absolute_and_per_column():
    n_t = 100
    hrf = _hrf()
    x = np.zeros((n_t, 2), dtype=np.float32)
    x[10, 0] = 1.0
    x[10, 1] = -5.0  # different scale AND different sign
    y = convolve_tr_locked(torch.as_tensor(x), hrf, n_t, tr=TR, microtime_dt=DT, device=CPU)
    assert pytest.approx(1.0, abs=1e-5) == float(y[:, 0].abs().max())
    assert pytest.approx(1.0, abs=1e-5) == float(y[:, 1].abs().max())
    assert y[:, 1].min() < 0  # sign survives normalisation


def test_multi_column_input_is_one_launch_but_independent_columns():
    n_t = 80
    hrf = _hrf()
    x = np.zeros((n_t, 3), dtype=np.float32)
    x[5, 0] = 1.0
    x[20, 1] = 1.0
    x[40, 2] = 1.0
    kw = dict(tr=TR, microtime_dt=DT, normalize_peak=False, device=CPU)
    y = convolve_tr_locked(torch.as_tensor(x), hrf, n_t, **kw)
    for c in range(3):
        solo = convolve_tr_locked(torch.as_tensor(x[:, c : c + 1]), hrf, n_t, **kw)
        assert torch.allclose(y[:, c : c + 1], solo, atol=1e-6)


def test_rejects_bad_geometry():
    hrf = _hrf()
    with pytest.raises(ValueError, match="expected n_timepoints"):
        convolve_tr_locked(torch.zeros(10, 1), hrf, 20, tr=TR, microtime_dt=DT, device=CPU)
    with pytest.raises(ValueError, match="microtime_onset"):
        convolve_tr_locked(
            torch.zeros(10, 1), hrf, 10, tr=TR, microtime_dt=DT, microtime_onset=BINS, device=CPU
        )


# ---------------------------------------------------------------------------
# Labels and modifiers
# ---------------------------------------------------------------------------


def test_split_label_mod():
    assert split_label_mod("background") == ("background", "none")
    assert split_label_mod("background:abs") == ("background", "abs")
    assert split_label_mod("bg:deriv_abs") == ("bg", "deriv_abs")
    with pytest.raises(ValueError, match="unknown modifier"):
        split_label_mod("background:nonsense")
    with pytest.raises(ValueError, match="no label"):
        split_label_mod(":abs")


def test_all_declared_mods_are_implemented():
    arr = np.array([[1.0], [-2.0], [3.0], [-4.0]])
    for mod in STIM_VEC_MODS:
        out = apply_stim_vec_mod(arr, mod)
        assert out.shape == arr.shape


def test_mod_semantics():
    arr = np.array([[1.0], [-2.0], [3.0]])
    assert np.allclose(apply_stim_vec_mod(arr, "abs"), [[1.0], [2.0], [3.0]])
    # 1d_tool.py backward difference: first row zero.
    assert np.allclose(apply_stim_vec_mod(arr, "deriv"), [[0.0], [-3.0], [5.0]])
    # deriv_abs is |d[t]|, the derivative rectified -- not the derivative of |x|.
    assert np.allclose(apply_stim_vec_mod(arr, "deriv_abs"), [[0.0], [3.0], [5.0]])


def test_derivative_is_taken_per_run(tmp_path):
    """A between-run offset must not become a spike in the regressor."""
    run_starts, n_t = [0, 4], 8
    values = np.concatenate([np.zeros(4), np.full(4, 100.0)])
    path = tmp_path / "step.1D"
    np.savetxt(path, values)
    block = load_stim_vec_block("step:deriv", [path], run_starts, n_t, preconvolved=True, trim=None)
    assert block.values[4, 0] == 0.0, "run 2 must start its own derivative at zero"
    assert np.abs(block.values).max() == 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _write_runs(tmp_path, per_run):
    paths = []
    for i, arr in enumerate(per_run):
        p = tmp_path / f"vec_run{i + 1}.1D"
        np.savetxt(p, arr)
        paths.append(p)
    full = tmp_path / "vec_full.1D"
    np.savetxt(full, np.concatenate(per_run))
    return paths, full


def test_per_run_files_equal_one_full_length_file(tmp_path):
    run_starts, n_t = [0, 30], 70
    rng = np.random.default_rng(0)
    per_run = [rng.normal(size=30), rng.normal(size=40)]
    paths, full = _write_runs(tmp_path, per_run)

    a = load_stim_vec_block("bg", [full], run_starts, n_t, preconvolved=True)
    b = load_stim_vec_block("bg", paths, run_starts, n_t, preconvolved=True)
    assert np.allclose(a.values, b.values)
    assert a.values.shape == (n_t, 1)


def test_wrong_file_count_and_length_are_errors(tmp_path):
    run_starts, n_t = [0, 30], 70
    per_run = [np.zeros(30), np.zeros(40)]
    paths, full = _write_runs(tmp_path, per_run)

    with pytest.raises(ValueError, match="one file per run"):
        load_stim_vec_block("bg", paths[:1] + paths, run_starts, n_t, preconvolved=True)
    with pytest.raises(ValueError, match="rows"):
        load_stim_vec_block("bg", [full], [0, 30], 100, preconvolved=True)


def test_collect_from_args_rejects_duplicate_labels(tmp_path):
    path = tmp_path / "v.1D"
    np.savetxt(path, np.zeros(20))
    parser = argparse.ArgumentParser()
    add_stim_vec_arguments(parser)
    args = parser.parse_args(["-stim_event_vec", "bg", str(path), "-stim_vec", "bg", str(path)])
    with pytest.raises(ValueError, match="more than once"):
        collect_stim_vec_blocks(args, [0], 20)


def test_collect_from_args_round_trip(tmp_path):
    path = tmp_path / "v.1D"
    np.savetxt(path, np.arange(20.0))
    parser = argparse.ArgumentParser()
    add_stim_vec_arguments(parser)
    args = parser.parse_args(
        ["-stim_event_vec", "bg:abs", str(path), "-stim_vec", "pre", str(path)]
    )
    blocks = collect_stim_vec_blocks(args, [0], 20)
    assert [(b.label, b.mod, b.preconvolved) for b in blocks] == [
        ("bg", "abs", False),
        ("pre", "none", True),
    ]


# ---------------------------------------------------------------------------
# Design assembly
# ---------------------------------------------------------------------------


def _block(label, values, preconvolved=False, mod="none"):
    from fastfuncstuff.design.stim_vec import StimVecBlock

    return StimVecBlock(
        label=label,
        mod=mod,
        values=np.asarray(values, dtype=np.float64).reshape(len(values), -1),
        preconvolved=preconvolved,
    )


def test_preconvolved_block_is_used_verbatim():
    n_t = 40
    raw = np.linspace(0, 3, n_t)
    design, labels, groups = build_stim_vec_design(
        [_block("pre", raw, preconvolved=True)],
        n_timepoints=n_t,
        tr=TR,
        microtime_dt=DT,
        device=CPU,
    )
    assert np.allclose(design.numpy()[:, 0], raw)
    assert labels == ["pre#0"]
    assert groups == [("pre", 0, 0)]


def test_spmg2_gives_the_vector_derivative_columns():
    n_t = 60
    x = np.zeros(n_t)
    x[10] = 1.0
    bases, note = resolve_stim_vec_hrf(
        "SPMG2", is_fir_model=False, n_basis=2, microtime_dt=DT, device=CPU
    )
    assert bases.shape[0] == 2 and "derivative" in note
    design, labels, groups = build_stim_vec_design(
        [_block("bg", x)],
        n_timepoints=n_t,
        tr=TR,
        microtime_dt=DT,
        hrf_bases=bases,
        device=CPU,
    )
    assert design.shape == (n_t, 2)
    assert labels == ["bg#0", "bg#1"]
    assert groups == [("bg", 0, 1)]


def test_fir_design_falls_back_to_spmg1_and_says_so():
    bases, note = resolve_stim_vec_hrf(
        "TENT", is_fir_model=True, n_basis=6, microtime_dt=DT, device=CPU
    )
    assert bases.shape[0] == 1
    assert "SPMG1" in note


def test_group_offsets_are_contiguous_across_blocks():
    n_t = 50
    two_col = np.stack([np.arange(n_t), np.arange(n_t)[::-1]], axis=1)
    design, labels, groups = build_stim_vec_design(
        [_block("a", np.arange(n_t), preconvolved=True), _block("b", two_col, preconvolved=True)],
        n_timepoints=n_t,
        tr=TR,
        microtime_dt=DT,
        device=CPU,
    )
    assert design.shape == (n_t, 3)
    assert labels == ["a#0", "b#0", "b#1"]
    assert groups == [("a", 0, 0), ("b", 1, 2)]


def test_event_vec_without_an_hrf_is_an_error():
    with pytest.raises(ValueError, match="needs hrf_bases"):
        build_stim_vec_design(
            [_block("bg", np.zeros(20))], n_timepoints=20, tr=TR, microtime_dt=DT, device=CPU
        )


# ---------------------------------------------------------------------------
# End to end: does the beta come back?
# ---------------------------------------------------------------------------


def test_known_amplitude_is_recovered_by_ols():
    """An oscillating background at a known amplitude, fit alongside drift."""
    n_t, run_starts = 200, [0, 100]
    hrf = _hrf()
    t = np.arange(n_t) * TR
    raw = np.abs(np.sin(2 * np.pi * t / 40.0))

    design, _, _ = build_stim_vec_design(
        [_block("bg", raw)],
        n_timepoints=n_t,
        tr=TR,
        microtime_dt=DT,
        hrf_bases=hrf.unsqueeze(0),
        run_starts=run_starts,
        device=CPU,
    )
    truth = 2.5
    rng = np.random.default_rng(1)
    drift = np.linspace(-1, 1, n_t) * 0.3
    y = truth * design.numpy()[:, 0] + 100.0 + drift + rng.normal(0, 0.01, n_t)

    x = np.column_stack([design.numpy()[:, 0], np.ones(n_t), np.linspace(-1, 1, n_t)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    assert beta[0] == pytest.approx(truth, abs=0.01)


# ---------------------------------------------------------------------------
# Unpenalized (Frisch-Waugh) split used by ffs_ridge
# ---------------------------------------------------------------------------


def test_two_stage_projection_reproduces_the_joint_ols_fit():
    """The unpenalized path must equal fitting everything jointly.

    ffs_ridge does not put the stim vectors in the design; it projects them out
    (stage two, after the per-run nuisance) and fits the trials on what is left.
    That is only legitimate if it reproduces the joint model exactly -- and if
    the back-substitution returns the vector's beta in the vector's own units,
    not in the orthonormalised basis the QR happens to use.
    """
    from fastfuncstuff.design.stim_vec import recover_stim_vec_betas, residualize_stim_vecs
    from fastfuncstuff.glm.xval import project_out_nuisance_per_run

    torch.manual_seed(0)
    n_t, run_starts = 120, [0, 60]
    n_trials, n_vox = 8, 5

    trials = torch.randn(n_t, n_trials)
    vecs = torch.randn(n_t, 2) * 3.0  # deliberately not unit-scaled
    nuisance_per_run = [torch.randn(60, 3), torch.randn(60, 3)]
    nuisance_full = torch.block_diag(*nuisance_per_run)

    joint = torch.cat([trials, vecs, nuisance_full], dim=1)
    truth = torch.randn(joint.shape[1], n_vox)
    data = (joint @ truth).T + 0.01 * torch.randn(n_vox, n_t)

    joint_betas = torch.linalg.lstsq(joint, data.T).solution  # (n_cols, n_vox)

    # Stage one: per-run nuisance out of data, trials AND vectors together.
    data_1, design_1 = project_out_nuisance_per_run(
        data, torch.cat([trials, vecs], dim=1), nuisance_per_run, run_starts, device=CPU
    )
    trials_1, vecs_1 = design_1[:, :n_trials], design_1[:, n_trials:]

    # Stage two: vectors out, globally.
    data_2, trials_2, q, r = residualize_stim_vecs(data_1, trials_1, vecs_1, device=CPU)
    trial_betas = torch.linalg.lstsq(trials_2, data_2.T).solution.T  # (n_vox, n_trials)

    assert torch.allclose(trial_betas.T, joint_betas[:n_trials], atol=1e-3)

    vec_betas = recover_stim_vec_betas(data_1, trials_1, trial_betas, q, r)
    assert torch.allclose(vec_betas.T, joint_betas[n_trials : n_trials + 2], atol=1e-3)
