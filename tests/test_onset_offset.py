"""Tests for the onset/offset (OSO) block-design expansion."""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.hrf import get_hrf_library
from fastfuncstuff.design.matrices import build_task_design
from fastfuncstuff.design.onset_offset import (
    OSOPlan,
    column_vifs,
    expand_events,
    offset_events,
    plan_onset_offset,
    waveshape_maps,
)

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def library():
    return get_hrf_library(mode="library", stim_duration=0.0, microtime_dt=0.1, device=CPU)


def _blocks(onsets, n_runs=2):
    return [np.asarray(onsets, dtype=float) for _ in range(n_runs)]


def _plan(library, duration, tr, n_tp_per_run, onsets, n_runs=2, **kw):
    run_starts = [i * n_tp_per_run for i in range(n_runs)]
    return plan_onset_offset(
        mode="joint",
        event_onsets=[_blocks(onsets, n_runs)],
        durations=[duration],
        condition_labels=["blk"],
        hrf_library=library,
        n_timepoints=n_tp_per_run * n_runs,
        run_starts=run_starts,
        tr=tr,
        microtime_dt=0.1,
        device=CPU,
        verbose=False,
        **kw,
    )


# ---------------------------------------------------------------- expansion


def test_expand_events_inserts_on_and_off():
    events = [_blocks([10.0, 70.0]), _blocks([30.0])]
    ev, dur, labels, groups = expand_events(events, [20.0, 0.5], ["blk", "trial"], [True, False])
    assert labels == ["blk", "blk_ON", "blk_OFF", "trial"]
    assert dur == [20.0, 0.0, 0.0, 0.5]
    assert groups == {0: (0, 1, 2)}
    # OFF events are the onsets shifted by the block duration.
    np.testing.assert_allclose(ev[2][0], np.array([30.0, 90.0]))
    # ON events keep the original onsets but lose the duration.
    np.testing.assert_allclose(ev[1][0], np.array([10.0, 70.0]))


def test_expand_events_disabled_is_identity():
    events = [_blocks([10.0])]
    ev, dur, labels, groups = expand_events(events, [20.0], ["blk"], [False])
    assert labels == ["blk"] and dur == [20.0] and groups == {}
    assert len(ev) == 1


def test_offset_events_shifts_every_run():
    runs = offset_events(_blocks([5.0, 25.0], n_runs=3), 20.0)
    assert len(runs) == 3
    for run in runs:
        np.testing.assert_allclose(run, np.array([25.0, 45.0]))


def test_expand_events_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        expand_events([_blocks([1.0])], [20.0, 20.0], None, [True])
    with pytest.raises(ValueError):
        expand_events([_blocks([1.0])], [20.0], None, [True, True])


# ---------------------------------------------------------------------- VIF


def test_column_vifs_orthogonal_is_one():
    n = 200
    t = torch.arange(n, dtype=torch.float64)
    X = torch.stack([torch.sin(t / 7), torch.cos(t / 7)], dim=1)
    assert torch.allclose(column_vifs(X), torch.ones(2, dtype=torch.float64), atol=0.2)


def test_column_vifs_duplicate_column_is_infinite():
    """Perfect collinearity leaves a ~1e-31 residual, not a zero one."""
    x = torch.randn(100, 1, dtype=torch.float64)
    X = torch.cat([x, x], dim=1)
    assert torch.isinf(column_vifs(X)).all()


def test_column_vifs_constant_column_is_infinite():
    X = torch.cat([torch.randn(50, 1), torch.ones(50, 1)], dim=1)
    assert torch.isinf(column_vifs(X)[1])


# --------------------------------------------------------------------- gate


def test_long_blocks_pass_the_gate(library):
    plan = _plan(library, 20.0, 2.0, 150, [10.0, 70.0, 130.0, 190.0, 250.0])
    assert plan.enabled == [True]
    assert plan.active
    assert plan.max_vif[0] < 2.0
    assert plan.labels == ["blk", "blk_ON", "blk_OFF"]
    assert plan.source == [0, 0, 0]


def test_short_events_are_refused(library):
    plan = _plan(library, 0.5, 1.0, 300, [10.0, 40.0, 70.0, 100.0, 130.0])
    assert plan.enabled == [False]
    assert not plan.active
    assert plan.max_vif[0] > 100
    assert "too short" in plan.reasons[0]
    # A refused condition keeps exactly its one column.
    assert plan.labels == ["blk"] and plan.source == [0]


def test_impulse_duration_is_refused_without_building_a_design(library):
    plan = _plan(library, 0.0, 2.0, 150, [10.0, 70.0])
    assert plan.enabled == [False]
    assert "impulse" in plan.reasons[0]
    assert np.isnan(plan.max_vif[0])


def test_offsets_past_the_run_end_are_refused(library):
    # One 20 s block whose offset lands beyond the (30 s) run: the OFF column is
    # identically zero, which is a silent rank deficiency if it reaches the fit.
    plan = _plan(library, 20.0, 2.0, 15, [14.0], n_runs=2)
    assert plan.enabled == [False]
    assert "empty" in plan.reasons[0]


def test_gate_is_per_condition(library):
    run_starts = [0, 150]
    plan = plan_onset_offset(
        mode="joint",
        event_onsets=[_blocks([10.0, 90.0, 170.0]), _blocks([50.0, 130.0, 210.0])],
        durations=[20.0, 0.5],
        condition_labels=["block", "trial"],
        hrf_library=library,
        n_timepoints=300,
        run_starts=run_starts,
        tr=2.0,
        microtime_dt=0.1,
        device=CPU,
        verbose=False,
    )
    assert plan.enabled == [True, False]
    assert plan.labels == ["block", "block_ON", "block_OFF", "trial"]
    assert plan.source == [0, 0, 0, 1]
    assert plan.groups == {0: (0, 1, 2)}


def test_mode_off_short_circuits(library):
    plan = plan_onset_offset(
        mode="off",
        event_onsets=None,
        durations=[20.0],
        condition_labels=["blk"],
        hrf_library=library,
        n_timepoints=300,
        run_starts=[0],
        tr=2.0,
        microtime_dt=0.1,
        device=CPU,
        verbose=False,
    )
    assert not plan.active and plan.enabled == [False] and plan.labels == ["blk"]


def test_joint_mode_without_events_raises(library):
    with pytest.raises(ValueError, match="event list"):
        plan_onset_offset(
            mode="joint",
            event_onsets=None,
            durations=[20.0],
            condition_labels=["blk"],
            hrf_library=library,
            n_timepoints=300,
            run_starts=[0],
            tr=2.0,
            microtime_dt=0.1,
            device=CPU,
            verbose=False,
        )


# --------------------------------------------------------- design behaviour


def test_expanded_design_columns_are_unit_peak(library):
    """SUS, ON and OFF must share a scale or w is meaningless."""
    events = [_blocks([10.0, 90.0, 170.0], n_runs=1)]
    ev, dur, _, _ = expand_events(events, [20.0], ["blk"], [True])
    design = build_task_design(
        library[9],
        150,
        [0],
        tr=2.0,
        microtime_dt=0.1,
        event_onsets=ev,
        durations=dur,
        device=CPU,
    )
    assert design.shape == (150, 3)
    peaks = design.abs().max(dim=0).values
    assert torch.allclose(peaks, torch.ones(3), atol=0.05)


def test_oso_design_recovers_planted_betas(library):
    """The point of the model: a transient-plus-sustained response separates."""
    events = [_blocks([10.0, 90.0, 170.0, 250.0], n_runs=1)]
    ev, dur, _, groups = expand_events(events, [20.0], ["blk"], [True])
    X = build_task_design(
        library[9], 200, [0], tr=2.0, microtime_dt=0.1, event_onsets=ev, durations=dur, device=CPU
    )
    truth = torch.tensor([1.0, 0.6, -0.35])
    y = X @ truth
    beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    assert torch.allclose(beta, truth, atol=1e-4)

    plan = OSOPlan(mode="joint", enabled=[True], labels=["blk", "blk_ON", "blk_OFF"], groups=groups)
    maps = waveshape_maps(beta.unsqueeze(0), plan)
    w = float(maps["blk_w"])
    a = float(maps["blk_a"])
    assert w == pytest.approx(1.0 / (1.0 + 0.6 + 0.35), abs=1e-5)
    assert a == pytest.approx((0.6 - (-0.35)) / (0.6 + 0.35), abs=1e-5)


# ------------------------------------------------------------ index algebra


def test_waveshape_maps_shapes_and_range():
    groups = {0: (0, 1, 2), 1: (3, 4, 5)}
    plan = OSOPlan(
        mode="joint",
        enabled=[True, True],
        labels=["a", "a_ON", "a_OFF", "b", "b_ON", "b_OFF"],
        groups=groups,
    )
    betas = torch.randn(64, 6)
    maps = waveshape_maps(betas, plan)
    assert set(maps) == {"a_w", "a_a", "b_w", "b_a"}
    for v in maps.values():
        assert v.shape == (64,)
        assert bool((v.abs() <= 1.0 + 1e-6).all())


def test_waveshape_maps_zero_betas_are_zero_not_nan():
    plan = OSOPlan(
        mode="joint", enabled=[True], labels=["a", "a_ON", "a_OFF"], groups={0: (0, 1, 2)}
    )
    maps = waveshape_maps(torch.zeros(8, 3), plan)
    for v in maps.values():
        assert torch.isfinite(v).all() and float(v.abs().max()) == 0.0


def test_waveshape_maps_pure_sustained_is_w_one():
    plan = OSOPlan(
        mode="joint", enabled=[True], labels=["a", "a_ON", "a_OFF"], groups={0: (0, 1, 2)}
    )
    betas = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    maps = waveshape_maps(betas, plan)
    np.testing.assert_allclose(maps["a_w"].numpy(), [1.0, -1.0])


def test_plan_metadata_is_json_safe(library):
    import json

    plan = _plan(library, 20.0, 2.0, 150, [10.0, 70.0, 130.0])
    json.dumps(plan.to_metadata())
