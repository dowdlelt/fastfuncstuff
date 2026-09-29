"""Described experiments -> event realizations."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.simulation.experiment import (
    ExperimentSpec,
    Interval,
    Unit,
    default_contrasts,
    parse_contrast,
    realize,
)


class TestInterval:
    @pytest.mark.parametrize(
        ("spec", "kind", "mean"),
        [("4", "fixed", 4.0), ("uniform:2,6", "uniform", 4.0), ("exp:4,2,12", "exp", 4.0)],
    )
    def test_parse(self, spec, kind, mean):
        iv = Interval.parse(spec)
        assert iv.kind == kind and iv.mean == mean

    @pytest.mark.parametrize("bad", ["exp:1,2,12", "uniform:3", "gauss:1,2", "x"])
    def test_bad_specs_are_refused(self, bad):
        with pytest.raises(ValueError):
            Interval.parse(bad)

    @pytest.mark.parametrize("spec", ["exp:4,2,12", "uniform:2,6", "poisson:5,2,12"])
    def test_samples_respect_bounds_and_mean(self, spec):
        iv = Interval.parse(spec)
        x = iv.sample(400, np.random.default_rng(0), tr=1.0)
        assert x.min() >= iv.low - 1e-9 and x.max() <= iv.high + 1e-9
        # uniform is not mean-matched, so it carries ordinary sampling error
        assert x.mean() == pytest.approx(iv.mean, rel=0.06 if iv.kind == "uniform" else 0.03)
        assert x.std() > 0.5  # jittered, not collapsed


class TestUnits:
    def test_repeat_syntax(self):
        u = Unit.parse("blk", "A:1x10,B:2", 3)
        assert [it.condition for it in u.items] == ["A"] * 10 + ["B"]

    def test_conflicting_durations_need_distinct_names(self):
        spec = ExperimentSpec(tr=2, units=[Unit.parse("x", "A:2", 2), Unit.parse("y", "A:4", 2)])
        with pytest.raises(ValueError, match="give each its own name"):
            spec.durations()


class TestRealize:
    def _spec(self, **kw):
        units = [
            Unit.parse("A", "A:2", 12),
            Unit.parse("B", "B:2", 8),
            Unit.parse("n", "null:3", 5),
        ]
        base = dict(
            tr=1.25,
            units=units,
            n_runs=2,
            isi=Interval.parse("exp:4,2,12"),
            initial_fix=10,
            post_fix=16,
        )
        base.update(kw)
        return ExperimentSpec(**base)

    def test_counts_timing_and_nulls(self):
        r = realize(self._spec(), seed=3)
        assert r.conditions == ["A", "B"]  # null is time, not a condition
        assert [len(x) for x in r.onsets[0]] == [12, 12]
        assert [len(x) for x in r.onsets[1]] == [8, 8]
        first = min(min(r.onsets[0][0]), min(r.onsets[1][0]))
        assert first == pytest.approx(10.0)
        # 25 units: stimulus time + 24 gaps (mean 4) + fixation
        expected = 10 + 12 * 2 + 8 * 2 + 5 * 3 + 24 * 4 + 16
        assert r.run_durations[0] == pytest.approx(expected, rel=0.02)
        assert r.run_lengths[0] == int(np.ceil(r.run_durations[0] / 1.25))

    def test_gaps_are_at_least_the_minimum(self):
        r = realize(self._spec(), seed=4)
        on = np.sort(np.concatenate([r.onsets[0][0], r.onsets[1][0]]))
        assert np.diff(on).min() >= 2 + 2 - 1e-9  # 2 s event + >= 2 s gap

    def test_seed_reproducible_and_distinct(self):
        a, b, c = realize(self._spec(), 1), realize(self._spec(), 1), realize(self._spec(), 2)
        assert np.array_equal(a.onsets[0][0], b.onsets[0][0])
        assert not np.array_equal(a.onsets[0][0], c.onsets[0][0])

    def test_miniblock_internal_spacing(self):
        spec = ExperimentSpec(
            tr=2,
            units=[Unit.parse("AB", "A:2,B:2", 5)],
            within_isi=Interval.parse(1),
            isi=Interval.parse(10),
        )
        r = realize(spec, 0)
        np.testing.assert_allclose(r.onsets[1][0] - r.onsets[0][0], 3.0)  # 2 s A + 1 s gap
        np.testing.assert_allclose(np.diff(r.onsets[0][0]), 2 + 1 + 2 + 10)


def test_contrasts():
    assert parse_contrast("A-2*B+C", ["A", "B", "C"]).tolist() == [1, -2, 1]
    assert list(default_contrasts(["A", "B", "C"])) == ["A", "B", "C", "A-B", "A-C", "B-C"]
    with pytest.raises(ValueError, match="unknown condition"):
        parse_contrast("A-D", ["A", "B"])


class TestItemOffsets:
    """LABEL:DUR[:OFF][xN]: a per-position gap after each item, jitterable."""

    def test_parse_the_mixed_example(self):
        u = Unit.parse("ABC", "A:0.5:0, B:2:2,C3:3:uniform:2,4", 10)
        assert [(i.condition, i.duration, str(i.off)) for i in u.items] == [
            ("A", 0.5, "0"),
            ("B", 2.0, "2"),
            ("C3", 3.0, "uniform:2,4"),
        ]

    def test_specs_with_commas_stay_with_their_item(self):
        u = Unit.parse("x", "A:2:exp:4,2,12,B:1", 1)
        assert [i.condition for i in u.items] == ["A", "B"]
        assert str(u.items[0].off) == "exp:4,2,12" and u.items[1].off is None

    def test_interval_kinds_cannot_name_conditions(self):
        with pytest.raises(ValueError, match="interval spec"):
            Unit.parse("x", "uniform:2", 1)

    def test_timeline_follows_the_offsets(self):
        u = Unit.parse("ABC", "A:0.5:0, B:2:2, C:3:uniform:2,4", 12)
        r = realize(ExperimentSpec(tr=1, units=[u], isi=Interval.parse(99)), 0)
        a, b, c = (np.asarray(x[0]) for x in r.onsets)
        np.testing.assert_allclose(b - a, 0.5)  # no gap after A
        np.testing.assert_allclose(c - b, 4.0)  # 2 s on + 2 s off
        between = a[1:] - (c[:-1] + 3.0)  # C's OFF overrides -isi 99
        assert between.min() >= 2.0 - 1e-9 and between.max() <= 4.0 + 1e-9
        assert between.std() > 0.2

    def test_jittered_within_gaps_are_actually_jittered(self):
        """Drawn one unit at a time, a mean-matched 1-sample draw is the mean itself."""
        spec = ExperimentSpec(
            tr=2,
            units=[Unit.parse("AB", "A:2,B:2", 20)],
            within_isi=Interval.parse("exp:3,1,8"),
            isi=Interval.parse(8),
        )
        r = realize(spec, 0)
        gaps = np.asarray(r.onsets[1][0]) - np.asarray(r.onsets[0][0]) - 2.0
        assert gaps.std() > 0.5
        assert gaps.mean() == pytest.approx(3.0, rel=0.02)


class TestCountsAndScanTime:
    BASE = dict(tr=1.25, isi=Interval.parse("exp:4,2,12"), initial_fix=10, post_fix=16)

    def test_num_events_turns_counts_into_weights(self):
        spec = ExperimentSpec(
            units=[
                Unit.parse("A", "A:2", 3),
                Unit.parse("B", "B:2", 1),
                Unit.parse("n", "null:2", 1),
            ],
            num_events=40,
            **self.BASE,
        )
        assert spec.resolve_counts() == [30, 10, 10]  # nulls scale with the events

    def test_scan_time_fills_the_run(self):
        spec = ExperimentSpec(
            units=[Unit.parse("A", "A:2", 1), Unit.parse("B", "B:2", 1)], scan_time=300, **self.BASE
        )
        counts = spec.resolve_counts()
        assert counts[0] == counts[1]
        per_unit = 2 + 4  # duration + mean gap
        assert abs(spec.expected_duration([float(c) for c in counts]) - 300) <= per_unit
        r = realize(spec, 0)
        assert r.run_lengths == [240] and r.run_durations == [300]

    def test_a_block_total_is_held_while_events_fill(self):
        spec = ExperimentSpec(
            units=[Unit.parse("A", "A:2", 1), Unit.parse("blk", "B:20", 1, family="block")],
            scan_time=400,
            num_blocks=4,
            **self.BASE,
        )
        counts = spec.resolve_counts()
        assert counts[1] == 4 and counts[0] > 30

    def test_fixed_content_that_cannot_fit_is_refused(self):
        spec = ExperimentSpec(
            units=[Unit.parse("A", "A:2", 1)], scan_time=100, num_events=100, **self.BASE
        )
        with pytest.raises(ValueError, match="more than -scan_time"):
            spec.resolve_counts()

    def test_fixed_counts_under_scan_time_keep_the_volumes(self):
        spec = ExperimentSpec(
            units=[Unit.parse("A", "A:2", 1)], scan_time=300, num_events=10, **self.BASE
        )
        r = realize(spec, 0)
        assert r.counts == [10] and r.run_lengths == [240]
        assert r.n_dropped == 0 and len(r.onsets[0][0]) == 10

    def test_overrun_events_are_dropped_and_counted(self):
        # uniform is not mean-matched: some realizations run long
        spec = ExperimentSpec(
            units=[Unit.parse("A", "A:2", 1)],
            scan_time=200,
            tr=1,
            isi=Interval.parse("uniform:2,10"),
            post_fix=0,
        )
        dropped = [realize(spec, s).n_dropped for s in range(20)]
        assert any(dropped)
        for s in range(20):
            r = realize(spec, s)
            assert len(r.onsets[0][0]) + r.n_dropped == r.counts[0]
            assert max(r.onsets[0][0]) < 200
