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

    def test_scan_time_never_rounds_past_the_scan(self):
        # 45 s blocks + 10 s gaps fill 330 s with 5.73 blocks: rounding to 6
        # needed 345 s, and the tool refused the count it had chosen itself.
        spec = ExperimentSpec(
            units=[
                Unit.parse("E1", "E1:45", 1, family="block"),
                Unit.parse("E2", "E2:45", 1, family="block"),
            ],
            tr=1.0,
            isi=Interval.parse(10),
            initial_fix=10,
            post_fix=15,
            scan_time=330,
        )
        counts = spec.resolve_counts()
        assert sum(counts) == 5
        assert spec.expected_duration([float(c) for c in counts]) <= 330
        text = spec.describe()
        assert "uneven counts (3/2 per run)" in text and "-num_blocks 4 (runs of 235 s)" in text
        # ... and the run is trimmed to the 5 blocks: padding the leftover 40 s
        # with fixation would understate the design per minute.
        assert "trimmed from -scan_time 330 s" in text
        r = realize(spec, 0)
        assert r.run_lengths == [290] and r.run_durations == [290.0] and r.n_dropped == 0

    def test_a_fixed_scan_keeps_its_final_fixation_and_whole_units(self):
        # uniform jitter is not mean-matched, so some runs run long. Dropping only
        # events that *started* past the scan let them eat the final fixation
        # (3.5 s of a 15 s one); a cycle unit must also never lose half of itself.
        spec = ExperimentSpec(
            units=[Unit.parse("cycle", "E1:0.25:0, E2:0.25:uniform:2,5", 1, "block")],
            tr=1.0,
            n_runs=2,
            initial_fix=10,
            post_fix=15,
            scan_time=330,
        )
        dropped = 0
        for seed in range(30):
            r = realize(spec, seed)
            dropped += r.n_dropped
            for run in range(2):
                end = max(o[run].max() + d for o, d in zip(r.onsets, r.durations, strict=True))
                assert r.run_durations[run] - end >= 15 - 1e-9
                assert len(r.onsets[0][run]) == len(r.onsets[1][run])
        assert dropped > 0 and dropped % 2 == 0  # whole two-event units

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


def test_scan_time_rounds_up_to_whole_volumes():
    spec = ExperimentSpec(
        tr=2.0, units=[Unit.parse("A", "A:2", 10)], isi=Interval.parse(10), post_fix=16,
        scan_time=331,
    )  # fmt: skip
    r = realize(spec, 0)
    assert r.run_lengths == [166] and r.run_durations == [332.0]


def test_tr_lock_puts_every_onset_on_the_grid_and_keeps_the_mean_gap():
    import copy

    from fastfuncstuff.simulation.experiment import assemble, draw_plans
    from fastfuncstuff.simulation.optimize import mutate

    def spec(lock):
        return ExperimentSpec(
            tr=1.5,
            units=[Unit.parse("A", "A:0.5", 1), Unit.parse("B", "B:0.5", 1)],
            isi=Interval.parse("exp:4,1,12"),
            n_runs=2,
            initial_fix=10,
            post_fix=15,
            scan_time=330,
            tr_lock=lock,
        )

    def onsets(r):
        return np.concatenate([np.concatenate(c) for c in r.onsets])

    locked, free = realize(spec(True), 0), realize(spec(False), 0)
    grid = onsets(locked) / 1.5
    assert np.allclose(grid, np.round(grid))
    assert not np.allclose(onsets(free) / 1.5, np.round(onsets(free) / 1.5))
    gap = np.mean(np.diff(np.sort(locked.onsets[0][0])))
    assert gap == pytest.approx(np.mean(np.diff(np.sort(free.onsets[0][0]))), rel=0.1)
    # a design search goes through the same layout: still on the grid
    counts, plans = draw_plans(spec(True), 1)
    plans = copy.deepcopy(plans)
    for _ in range(50):
        mutate(spec(True), plans, np.random.default_rng(0))
    g = onsets(assemble(spec(True), counts, plans)) / 1.5
    assert np.allclose(g, np.round(g))


def test_list_intervals_draw_from_the_values_and_even_balances_them():
    rng = np.random.default_rng(0)
    pick = Interval.parse("uniform:(2, 5, 9)")
    assert str(pick) == "uniform:(2,5,9)" and pick.mean == pytest.approx(16 / 3)
    assert set(pick.sample(200, rng)) == {2.0, 5.0, 9.0}
    even = Interval.parse("even:(2,5,9,15)")
    drawn = even.sample(8, rng)
    assert sorted(drawn) == [2, 2, 5, 5, 9, 9, 15, 15]
    assert list(drawn) != sorted(drawn) or list(even.sample(8, rng)) != sorted(drawn)  # shuffled
    rest = even.sample(6, rng)  # 6 over 4: one of each, two more without replacement
    vals, counts = np.unique(rest, return_counts=True)
    assert len(vals) == 4 and sorted(counts) == [1, 1, 2, 2]
    with pytest.raises(ValueError, match=r"uniform:\(4,9,11\)"):
        Interval.parse("uniform:4,9,11")


def test_items_split_on_spaces_and_a_gap_item_can_open_a_unit():
    u = Unit.parse(
        "c", "isi:4 DP:15:even:(4,9,11) DI:18:uniform:(2, 4, 6) DRV:4:0 DRD:4:1", 1, "block"
    )
    assert [it.condition for it in u.items] == ["null", "DP", "DI", "DRV", "DRD"]
    assert u.items[0].duration == 0 and u.items[0].off == Interval.parse(4)
    assert u.items[1].off.kind == "even" and u.items[2].off.kind == "choice"
    # the old comma form and range specs with commas still parse
    old = Unit.parse("c", "A:0.5:0, B:2:2, C:3:uniform:2,4", 1)
    assert old.items[2].off == Interval.parse("uniform:2,4")
    spaced = Unit.parse("c", "A:1:uniform:3,7 x15 A:1:10", 1, "block")
    assert len(spaced.items) == 16 and spaced.items[-1].off == Interval.parse(10)
    with pytest.raises(ValueError, match="reserved"):
        Unit.parse("c", "isi:2 ISI:3:1", 1)
    spec = ExperimentSpec(tr=1.0, units=[Unit.parse("c", "isi:7 A:1", 3, "block")], initial_fix=10)
    real = realize(spec, 0)
    assert real.onsets[0][0][0] == pytest.approx(17.0)  # 10 s fixation + the 7 s gap


def test_a_named_list_is_one_pool_across_units_and_shifts_keep_their_draw():
    from collections import Counter

    from fastfuncstuff.simulation.experiment import draw_plans

    lists = {"GAP": (4.0, 9.0, 11.0)}
    units = [
        Unit.parse("d", "DP:15:GAP DI:18:2 R:4:1", 1, "block", lists),
        Unit.parse("s", "SP:15:even:GAP SI:18:2 R:4:1", 1, "block", lists),
        Unit.parse("p", "isi:GAP-3 PI:18:2 R:4:1", 1, "block", lists),
    ]
    assert str(units[2].items[0].off) == "even:GAP-3" and units[2].items[0].off.low == 1
    spec = ExperimentSpec(tr=1.0, units=units, n_runs=2, initial_fix=10, post_fix=15, num_blocks=6)
    for seed in range(5):
        _, plans = draw_plans(spec, seed)
        for plan in plans:
            # 6 draws over 3 values: 2 each per run, counted before the -3 shift --
            # per item it would be 2 draws over 3 values, never balanced
            drawn = Counter(g[0] + 3 * (ui == 2) for ui, g, _ in plan.entries)
            assert drawn == {4.0: 2, 9.0: 2, 11.0: 2}
    # the pool fixes the content length, so nothing is dropped from a fixed scan
    spec.scan_time = 330
    assert sum(realize(spec, s).n_dropped for s in range(20)) == 0
    # a condition used by several units is one condition: one column, all onsets
    real = realize(spec, 0)
    assert real.conditions.count("R") == 1
    assert sum(len(r) for r in real.onsets[real.conditions.index("R")]) == 12

    pick = Interval.parse("uniform:GAP+1", lists)
    assert pick.kind == "choice" and not pick.pool
    assert set(pick.sample(100, np.random.default_rng(0))) == {5.0, 10.0, 12.0}
    with pytest.raises(ValueError, match="not an -isi_list"):
        Interval.parse("even:GAPS", lists)
    with pytest.raises(ValueError, match="negative"):
        Interval.parse("GAP-5", lists)


def test_pool_leftovers_rotate_across_runs():
    from fastfuncstuff.simulation.experiment import _deal

    rng, deck = np.random.default_rng(1), []
    runs = [np.bincount(_deal(3, 10, deck, rng), minlength=3) for _ in range(3)]
    for r in runs:  # each run balanced: 3/3/3 and one extra
        assert sorted(r) == [3, 3, 4]
    assert sum(runs).tolist() == [10, 10, 10]  # and so is the experiment


def test_shuffled_items_move_but_gaps_stay_with_their_positions():
    from collections import Counter

    from fastfuncstuff.simulation.experiment import draw_plans

    lists = {"J": (3.0, 5.0, 7.0)}
    items = " ".join(f"T{i}:{i}:Jx2" for i in range(1, 4)) + " isi:0 T4:4:J T4:4:10"
    u = Unit.parse("cond", items, 1, "block", lists)
    u.shuffle = True
    spec = ExperimentSpec(tr=1.0, units=[u, Unit.parse("B", "B:1", 2)], n_runs=3, post_fix=15)
    orders = set()
    for seed in range(4):
        _, plans = draw_plans(spec, seed)
        for plan in plans:
            for ui, gaps, shown in plan.entries:
                if ui != 0:
                    continue
                labels = [u.items[i].condition for i in shown]
                assert Counter(labels) == {"T1": 2, "T2": 2, "T3": 2, "T4": 2, "null": 1}
                assert labels[6] == "null"  # the gap item stays where it was written
                assert gaps[-1] == 10  # the block's own trailing gap, whoever is last
                orders.add(tuple(shown))
    assert len(orders) > 5
    real = realize(spec, 0)
    # durations travel with their condition, whatever slot it lands in
    assert real.durations[real.conditions.index("T3")] == 3


def test_unshuffled_plans_are_unchanged_by_the_item_order_field():
    spec = ExperimentSpec(
        tr=1.0, units=[Unit.parse("M", "A:1:2 B:2:uniform:2,4", 4, "block")],
        isi=Interval.parse("exp:4,2,10"), n_runs=2, post_fix=15,
    )  # fmt: skip
    from fastfuncstuff.simulation.experiment import draw_plans

    for _, _, shown in draw_plans(spec, 0)[1][0].entries:
        assert shown == [0, 1]
