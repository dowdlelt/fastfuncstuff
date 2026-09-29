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
