"""Tests for what a tunewarp run leaves behind, and whether it stays interpretable.

A trials table is only useful later if it records what produced it. On 2026-08-16
a fix to the qwarp penalty changed every warp the engine makes, silently
invalidating every table written before it -- with nothing in the file to say so.
These cover the machinery that turns that from a trap into a warning.
"""

from __future__ import annotations

import pytest

from fastfuncstuff.processing.tunestore import (
    BASELINE,
    RunMeta,
    TrialStore,
    comparability_warnings,
    format_guide,
    format_importance,
    format_results_table,
    format_runs,
    headline_metric,
    knob_importance,
)


def _meta(run_id=1, **kw):
    base = dict(
        started="2026-08-16T12:00:00",
        commit="aaaaaaa",
        recipe="MNI_T1",
        contrast="same",
        optimize="lpa",
        panel=["ls", "mi", "lncc"],
        search="adaptive",
        subjects=["s1", "s2"],
        shape=(40, 40, 40),
        voxdims=(1.0, 1.0, 1.0),
        n_mask_voxels=1000,
    )
    base.update(kw)
    return RunMeta(run_id=run_id, **base)


def _store(tmp_path, with_baseline=True):
    s = TrialStore(tmp_path / "t.json")
    s.runs.append(_meta())
    if with_baseline:
        for subj in ("s1", "s2"):
            s.add(BASELINE, subj, {}, [], scores={"ls": 0.60, "lncc": -0.05}, seconds=0.0)
    for reg, ls in ((0.0, 0.30), (1.0, 0.45)):
        for subj in ("s1", "s2"):
            s.add(
                "formwarp",
                subj,
                {"total_var": reg},
                [],
                scores={"ls": ls, "lncc": -0.20 + reg / 10},
                seconds=10.0,
            )
    s.compute_consensus(["ls", "lncc"])
    return s


class TestRunMeta:
    def test_records_what_produced_the_trials(self, tmp_path):
        s = TrialStore(tmp_path / "t.json")
        meta = s.begin_run(device="cuda", recipe="MNI_T1", panel=["ls"])
        assert meta.run_id == 1
        assert meta.started and meta.commit  # captured, not blank
        assert meta.device == "cuda"

    def test_trials_are_tagged_with_their_run(self, tmp_path):
        s = TrialStore(tmp_path / "t.json")
        s.begin_run(recipe="MNI_T1")
        s.add("formwarp", "s1", {}, [])
        assert s.trials[-1].run_id == 1
        s.begin_run(recipe="MNI_T1")
        s.add("formwarp", "s2", {}, [])
        assert s.trials[-1].run_id == 2

    def test_survives_a_round_trip(self, tmp_path):
        s = _store(tmp_path)
        s.save()
        again = TrialStore(tmp_path / "t.json")
        assert len(again.runs) == 1
        assert again.runs[0].shape == (40, 40, 40)
        assert again.runs[0].optimize == "lpa"

    def test_reads_a_table_written_before_run_tracking(self, tmp_path):
        """Old trials.json files have no 'runs' key at all."""
        path = tmp_path / "t.json"
        path.write_text(
            '{"config_ids": {}, "trials": [{"trial_id": 1, "config_id": 1, '
            '"backend": "formwarp", "subject": "s1", "config": {}, "command": []}]}'
        )
        s = TrialStore(path)
        assert s.runs == []
        assert "predates run tracking" in format_runs(s)

    def test_voxel_count_is_derived(self):
        assert _meta().n_voxels == 64000


class TestComparability:
    """Restarts fold new data in; they must never silently pretend it is the same."""

    def test_a_single_run_warns_about_nothing(self):
        assert comparability_warnings([_meta()]) == []

    def test_a_different_commit_is_flagged(self):
        w = comparability_warnings([_meta(1, commit="aaaaaaa"), _meta(2, commit="bbbbbbb")])
        assert any("commit" in x for x in w)

    def test_uncommitted_changes_are_flagged(self):
        w = comparability_warnings([_meta(1), _meta(2, dirty=True)])
        assert any("uncommitted" in x for x in w)

    def test_a_different_panel_is_flagged(self):
        w = comparability_warnings([_meta(1), _meta(2, panel=["ls"])])
        assert any("panel" in x for x in w)

    def test_a_different_optimised_cost_is_flagged(self):
        assert any(
            "optimised" in x for x in comparability_warnings([_meta(1), _meta(2, optimize="lpc")])
        )

    def test_a_different_grid_is_flagged_for_timing(self):
        w = comparability_warnings([_meta(1), _meta(2, shape=(80, 80, 80))])
        assert any("timings" in x for x in w)

    def test_a_stale_run_is_flagged(self):
        w = comparability_warnings([_meta(1, started="2025-01-01T00:00:00"), _meta(2)])
        assert any("days older" in x for x in w)

    def test_warnings_never_block(self, tmp_path):
        """The point is to inform, not to refuse: folding in new subjects is normal."""
        s = TrialStore(tmp_path / "t.json")
        s.runs += [_meta(1, commit="aaaaaaa"), _meta(2, commit="bbbbbbb")]
        s.add("formwarp", "s3", {}, [])  # still accepted
        assert s.warnings()
        assert len(s.trials) == 1


class TestBaseline:
    def test_baseline_is_a_row_like_any_other(self, tmp_path):
        s = _store(tmp_path)
        base = s.baseline()
        assert base is not None and base.is_baseline
        assert base.abs_scores["ls"] == pytest.approx(0.60)

    def test_absent_baseline_is_reported_not_faked(self, tmp_path):
        s = _store(tmp_path, with_baseline=False)
        assert s.baseline() is None
        assert "No baseline row" in format_results_table(s.results())

    def test_table_shows_the_improvement_over_doing_nothing(self, tmp_path):
        text = format_results_table(_store(tmp_path).results())
        assert "vs base" in text
        assert "lncc" in text

    def test_guide_states_whether_nonlinear_helped(self, tmp_path):
        text = format_guide(_store(tmp_path), "MNI_T1")
        assert "Is nonlinear worth it" in text
        assert "a change of" in text

    def test_guide_admits_when_it_cannot_say(self, tmp_path):
        text = format_guide(_store(tmp_path, with_baseline=False), "MNI_T1")
        assert "cannot say whether" in text


class TestAbsoluteScoresAndTiming:
    def test_raw_metric_values_are_aggregated(self, tmp_path):
        r = next(x for x in _store(tmp_path).results() if not x.is_baseline)
        assert set(r.abs_scores) >= {"ls", "lncc"}

    def test_timing_is_normalised_to_problem_size(self, tmp_path):
        """Raw seconds do not transfer to another grid; per-megavoxel does."""
        r = next(x for x in _store(tmp_path).results() if not x.is_baseline)
        assert r.seconds_per_mvox == pytest.approx(10.0 / (64000 / 1e6))

    def test_headline_metric_prefers_a_structural_one(self):
        from fastfuncstuff.processing.tunestore import ConfigResult

        rows = [ConfigResult(1, "f", {}, 1, 0.0, 0.0, "pass", [], 1.0, {"ls": 1.0, "lncc": -1.0})]
        assert headline_metric(rows) == "lncc"


class TestKnobImportance:
    def test_ranks_knobs_by_how_much_they_moved_the_score(self, tmp_path):
        items = knob_importance(_store(tmp_path))
        assert items and items[0].key == "total_var"
        assert 0.0 < items[0].share <= 1.0

    def test_a_pinned_knob_is_not_ranked(self, tmp_path):
        s = TrialStore(tmp_path / "t.json")
        for subj in ("s1", "s2"):
            s.add("formwarp", subj, {"fixed": 1.0}, [], scores={"consensus": 1.0})
        assert knob_importance(s) == []

    def test_report_renders_the_ranking(self, tmp_path):
        text = format_importance(knob_importance(_store(tmp_path)))
        assert "total_var" in text
        assert "share" in text

    def test_says_so_when_nothing_varied(self, tmp_path):
        assert "nothing varied" in format_importance([])
