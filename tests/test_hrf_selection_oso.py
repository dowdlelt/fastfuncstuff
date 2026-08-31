"""End-to-end tests for -oso_mode inside fit_glm_hrf_library_with_xval."""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.builder import create_onset_matrix_microtime
from fastfuncstuff.design.hrf import get_hrf_library
from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval
from fastfuncstuff.design.matrices import build_task_design
from fastfuncstuff.design.onset_offset import expand_events, waveshape_maps

CPU = torch.device("cpu")

TR = 2.0
DT = 0.1
N_RUNS = 4
N_TP_RUN = 100  # 200 s per run
BLOCK = 20.0
ONSETS = [10.0, 70.0, 130.0]  # last block ends at 150 s, well inside the run


@pytest.fixture(scope="module")
def library():
    return get_hrf_library(mode="library", stim_duration=0.0, microtime_dt=DT, device=CPU)


def _timing():
    run_starts = [i * N_TP_RUN for i in range(N_RUNS)]
    events = [[np.asarray(ONSETS, dtype=float) for _ in range(N_RUNS)]]
    onset_matrix = create_onset_matrix_microtime(
        events,
        run_starts,
        TR,
        N_TP_RUN * N_RUNS,
        DT,
        stim_durations=[BLOCK],
        device=CPU,
    )
    return run_starts, events, onset_matrix


def _synth(library, hrf_idx, betas, noise=0.05, n_voxels=40, seed=0):
    """Voxels whose truth is a known SUS/ON/OFF mixture of one library HRF."""
    run_starts, events, onset_matrix = _timing()
    ev, dur, _, _ = expand_events(events, [BLOCK], ["blk"], [True])
    X = build_task_design(
        library[hrf_idx],
        N_TP_RUN * N_RUNS,
        run_starts,
        tr=TR,
        microtime_dt=DT,
        event_onsets=ev,
        durations=dur,
        device=CPU,
    )
    g = torch.Generator().manual_seed(seed)
    signal = (X @ torch.tensor(betas)).unsqueeze(0).repeat(n_voxels, 1)
    data = signal + noise * torch.randn(signal.shape, generator=g)
    return data, run_starts, events, onset_matrix


def _fit(library, data, run_starts, events, onset_matrix, **kw):
    return fit_glm_hrf_library_with_xval(
        data=data,
        onsets=onset_matrix,
        hrf_library=library,
        tr=TR,
        run_starts=run_starts,
        stim_durations=[BLOCK],
        microtime_dt=DT,
        polort=1,
        device=CPU,
        verbose=False,
        condition_labels=["blk"],
        event_onsets=events,
        **kw,
    )


def test_off_mode_is_unchanged(library):
    data, run_starts, events, onset_matrix = _synth(library, 9, [1.0, 0.5, -0.3])
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode="off")
    assert res.final_results.betas.shape[1] == 1
    assert not res.oso_plan.active
    assert res.oso_gain is None


@pytest.mark.parametrize("mode", ["joint", "staged"])
def test_oso_widens_the_final_fit(library, mode):
    data, run_starts, events, onset_matrix = _synth(library, 9, [1.0, 0.5, -0.3])
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode=mode)
    assert res.oso_plan.active
    assert res.oso_plan.labels == ["blk", "blk_ON", "blk_OFF"]
    # Three task betas per condition, and the canonical baseline matches so the
    # two beta sets stay comparable.
    assert res.final_results.betas.shape[1] == 3
    assert res.canonical_results.betas.shape[1] == 3


def test_oso_recovers_the_planted_waveshape(library):
    truth = [1.0, 0.5, -0.3]
    data, run_starts, events, onset_matrix = _synth(library, 9, truth, noise=0.02)
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint")
    betas = res.final_results.betas
    assert torch.allclose(betas.mean(dim=0), torch.tensor(truth), atol=0.05)

    maps = waveshape_maps(betas, res.oso_plan)
    expected_w = truth[0] / (abs(truth[0]) + abs(truth[1]) + abs(truth[2]))
    expected_a = (truth[1] - truth[2]) / (abs(truth[1]) + abs(truth[2]))
    assert float(maps["blk_w"].median()) == pytest.approx(expected_w, abs=0.03)
    assert float(maps["blk_a"].median()) == pytest.approx(expected_a, abs=0.03)


def test_staged_selection_launders_the_transient(library):
    """Why joint is the recommended mode, on planted truth.

    Choosing the HRF from the sustained-only design lets the curve absorb the
    transient: with a real onset response present, staged selects a DIFFERENT
    library entry than the one the data was built from, and the final fit is
    then stuck with it -- beta_ON comes back at 0.19 against a planted 0.5.
    With a purely sustained truth there is nothing to absorb and the two modes
    agree.  This is the laundering effect from the waveshape wiki note, seen
    from the estimator side.
    """
    truth_idx = 9
    transient = [1.0, 0.5, -0.3]
    data, run_starts, events, onset_matrix = _synth(library, truth_idx, transient, noise=0.02)
    joint = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint")
    staged = _fit(library, data, run_starts, events, onset_matrix, oso_mode="staged")
    assert int(joint.hrf_index.mode().values) == truth_idx
    assert int(staged.hrf_index.mode().values) != truth_idx
    assert float(staged.final_results.betas[:, 1].mean()) < 0.5 * transient[1]

    sustained = [1.0, 0.0, 0.0]
    data, run_starts, events, onset_matrix = _synth(library, truth_idx, sustained, noise=0.02)
    joint = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint")
    staged = _fit(library, data, run_starts, events, onset_matrix, oso_mode="staged")
    assert int(joint.hrf_index.mode().values) == truth_idx
    assert int(staged.hrf_index.mode().values) == truth_idx


def test_sustained_only_model_cannot_see_the_transients(library):
    """The reason the columns exist: SUS alone misfits a transient response."""
    data, run_starts, events, onset_matrix = _synth(library, 9, [0.2, 1.0, -1.0], noise=0.02)
    off = _fit(library, data, run_starts, events, onset_matrix, oso_mode="off")
    joint = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint")
    assert float(joint.final_results.r2.median()) > float(off.final_results.r2.median()) + 0.2


def test_oso_gain_is_positive_when_the_truth_has_transients(library):
    data, run_starts, events, onset_matrix = _synth(library, 9, [0.5, 1.0, -0.8], noise=0.05)
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint", oso_gain=True)
    assert res.oso_gain is not None
    assert res.oso_gain.shape == (data.shape[0],)
    assert float(res.oso_gain.median()) > 0


def test_oso_gain_is_negative_when_the_columns_only_cost_dof(library):
    """A purely sustained response pays for the extra columns and gains nothing."""
    data, run_starts, events, onset_matrix = _synth(library, 9, [1.0, 0.0, 0.0], noise=0.3)
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint", oso_gain=True)
    assert float(res.oso_gain.median()) < 0


def test_gain_sign_is_family_relative_not_selection_relative(library):
    """staged selects on the narrow family, so its gain must not flip sign."""
    data, run_starts, events, onset_matrix = _synth(library, 9, [0.5, 1.0, -0.8], noise=0.05)
    joint = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint", oso_gain=True)
    staged = _fit(library, data, run_starts, events, onset_matrix, oso_mode="staged", oso_gain=True)
    assert float(joint.oso_gain.median()) > 0
    assert float(staged.oso_gain.median()) > 0
    assert float(joint.oso_gain.median()) == pytest.approx(
        float(staged.oso_gain.median()), abs=1e-5
    )


def test_short_events_fall_back_to_the_plain_model(library):
    """The gate has to hold at the CLI boundary, not just in the planner."""
    run_starts = [i * N_TP_RUN for i in range(N_RUNS)]
    events = [[np.asarray([10.0, 40.0, 70.0, 100.0, 130.0], dtype=float) for _ in range(N_RUNS)]]
    onset_matrix = create_onset_matrix_microtime(
        events, run_starts, TR, N_TP_RUN * N_RUNS, DT, stim_durations=[0.5], device=CPU
    )
    data = torch.randn(20, N_TP_RUN * N_RUNS, generator=torch.Generator().manual_seed(1))
    res = fit_glm_hrf_library_with_xval(
        data=data,
        onsets=onset_matrix,
        hrf_library=library,
        tr=TR,
        run_starts=run_starts,
        stim_durations=[0.5],
        microtime_dt=DT,
        polort=1,
        device=CPU,
        verbose=False,
        condition_labels=["trial"],
        event_onsets=events,
        oso_mode="joint",
    )
    assert not res.oso_plan.active
    assert res.final_results.betas.shape[1] == 1
    assert "too short" in res.oso_plan.reasons[0]


def test_metadata_records_the_plan(library):
    import json

    data, run_starts, events, onset_matrix = _synth(library, 9, [1.0, 0.4, -0.2])
    res = _fit(library, data, run_starts, events, onset_matrix, oso_mode="joint")
    meta = res.hrf_metadata["oso"]
    assert meta["mode"] == "joint" and meta["enabled"] == [True]
    assert res.hrf_metadata["task_column_labels"] == ["blk", "blk_ON", "blk_OFF"]
    json.dumps(res.hrf_metadata["oso"])
