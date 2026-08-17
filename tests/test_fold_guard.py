"""Tests for the local anti-folding guard and the convergence telemetry.

Both engines used to have exactly one fold control — the global smoothing knob —
so the only way to stop a handful of voxels inverting was to blur the whole field.
These cover the local alternative: back the step off *where* it would fold, and
refuse to snapshot an iterate that has folded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.formwarp import (
    LevelStats,
    SynConfig,
    _fold_damping_mask,
    formwarp,
    jacobian_determinant,
)
from fastfuncstuff.processing.optiwarp import OptiwarpConfig, optiwarp

CPU = torch.device("cpu")


def _blob(shape, cx, cy, cz, r, amp=100.0):
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    return (amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * r**2)))).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def pair():
    """A pair deformed hard enough that unguarded flow actually folds.

    The deformation has to be genuinely awkward — a large shift, a size change and
    two internal features moving *differently* — or the premise of these tests does
    not hold and they would pass without testing anything. A gentle pair simply does
    not fold at zero regularization.
    """
    shape = (40, 40, 40)
    base = (
        _blob(shape, 20, 20, 20, 9)
        + _blob(shape, 13, 20, 20, 2.5, 60.0)
        + _blob(shape, 27, 20, 20, 2.5, 60.0)
    )
    moving = (
        _blob(shape, 22.5, 20, 20, 10.5)
        + _blob(shape, 17.0, 20, 20, 2.5, 60.0)
        + _blob(shape, 26.0, 23.0, 20, 2.5, 60.0)
    )
    rng = np.random.default_rng(0)
    moving = moving + rng.normal(0, 1.0, shape).astype(np.float32)
    return torch.from_numpy(base).to(CPU), torch.from_numpy(moving).to(CPU)


def _opti(pair, **kw):
    fixed, moving = pair
    kw.setdefault("update_sigma", 0.5)
    cfg = OptiwarpConfig(
        verb=0,
        force="demons",
        shrink_factors=(1,),
        smoothing_sigmas=(0.0,),
        iterations=(60,),
        convergence_window=0,
        **kw,
    )
    return optiwarp(fixed, moving, config=cfg, device=CPU)


class TestDampingMask:
    def test_covers_the_fold_and_its_neighbourhood(self):
        jac = torch.ones(9, 9, 9)
        jac[4, 4, 4] = -0.5
        mask = _fold_damping_mask(jac, floor=0.05, strength=1.0)
        assert mask[4, 4, 4] > 0.9, "the offending voxel must be strongly damped"
        assert mask[4, 4, 7] > 0, "its neighbourhood should feel some of it"
        # Not exactly zero: the Gaussian has a finite tail. "Untouched" means the far
        # corner keeps >99% of its step, not that the kernel has compact support.
        assert mask[0, 0, 0] < 0.01, "far away must be essentially untouched"

    def test_is_a_backoff_not_a_veto(self):
        """Strength 0.5 halves the local step; it never freezes the region.

        Cancelling the update outright was measurably worse — the level found its
        best legal iterate at 3 and spent 56 more fighting the clamp.
        """
        jac = torch.ones(9, 9, 9)
        jac[4, 4, 4] = -0.5
        mask = _fold_damping_mask(jac, floor=0.05, strength=0.5)
        assert 0.0 < float(mask.max()) <= 0.5

    def test_clean_field_damps_nothing(self):
        mask = _fold_damping_mask(torch.ones(8, 8, 8), floor=0.05, strength=1.0)
        assert float(mask.max()) == pytest.approx(0.0, abs=1e-6)

    def test_isolated_fold_still_gets_full_strength(self):
        """Smoothing alone peaks far below 1 on a single voxel, so the damping
        would have been weakest exactly where the fold is. Dilate-then-smooth."""
        jac = torch.ones(13, 13, 13)
        jac[6, 6, 6] = -1.0
        lone = float(_fold_damping_mask(jac, 0.05, 1.0).max())
        jac[5:8, 5:8, 5:8] = -1.0
        clustered = float(_fold_damping_mask(jac, 0.05, 1.0).max())
        assert lone > 0.9
        assert lone == pytest.approx(clustered, rel=0.1)


class TestOptiwarpFoldGuard:
    def test_unguarded_run_folds_at_zero_regularization(self, pair):
        """The premise. Without this the guard would have nothing to fix."""
        r = _opti(pair, total_sigma=0.0, fold_guard=0.0, fold_aware_best=False)
        assert int((r.jacobian <= 0).sum()) > 0

    def test_guard_removes_the_folding(self, pair):
        r = _opti(pair, total_sigma=0.0)
        assert int((r.jacobian <= 0).sum()) == 0
        assert r.min_jacobian > 0

    def test_guard_is_a_no_op_when_nothing_would_fold(self, pair):
        """At the shipped regularization the guard must not perturb anything —
        otherwise every existing result silently changes."""
        guarded = _opti(pair, total_sigma=1.0, update_sigma=1.0)
        plain = _opti(
            pair, total_sigma=1.0, update_sigma=1.0, fold_guard=0.0, fold_aware_best=False
        )
        assert guarded.cost == pytest.approx(plain.cost, rel=1e-9)
        assert guarded.min_jacobian == pytest.approx(plain.min_jacobian, rel=1e-9)
        assert guarded.levels[0].damped_iters == 0

    def test_guard_opens_up_a_setting_that_was_illegal(self, pair):
        """The point of the whole exercise: low regularization used to be
        unusable because it folded, not because it fit badly."""
        low = _opti(pair, total_sigma=0.0)
        safe = _opti(pair, total_sigma=1.0, update_sigma=1.0)
        assert int((low.jacobian <= 0).sum()) == 0
        assert low.cost < safe.cost, "guarded low regularization should fit better"

    def test_records_that_it_intervened(self, pair):
        assert _opti(pair, total_sigma=0.0).levels[0].damped_iters > 0

    def test_fold_aware_best_rejects_a_folded_snapshot(self, pair):
        """Folding improves the image metric, so an unaware snapshot prefers it."""
        aware = _opti(pair, total_sigma=0.0, fold_guard=0.0, fold_aware_best=True)
        unaware = _opti(pair, total_sigma=0.0, fold_guard=0.0, fold_aware_best=False)
        assert unaware.min_jacobian < aware.min_jacobian


class TestFormwarpFoldGuard:
    def _syn(self, pair, **kw):
        fixed, moving = pair
        cfg = SynConfig(
            verb=0,
            metric="cc",
            shrink_factors=(2, 1),
            smoothing_sigmas=(1.0, 0.0),
            iterations=(20, 20),
            convergence_window=0,
            **kw,
        )
        return formwarp(fixed, moving, config=cfg, device=CPU)

    def test_guard_is_a_no_op_at_default_regularization(self, pair):
        a = self._syn(pair)
        b = self._syn(pair, fold_guard=0.0, fold_aware_best=False)
        assert torch.allclose(a.fwd[0], b.fwd[0], atol=1e-6)

    def test_guarded_run_is_fold_free_under_hard_settings(self, pair):
        r = self._syn(pair, update_var=0.5, total_var=0.0, grad_step=1.0)
        jac = jacobian_determinant(*r.fwd)
        assert int((jac <= 0).sum()) == 0

    def test_reports_level_telemetry(self, pair):
        r = self._syn(pair)
        assert len(r.levels) == 2
        assert all(isinstance(s, LevelStats) for s in r.levels)


class TestLevelStats:
    def test_starved_means_still_improving_at_the_cap(self):
        s = LevelStats(iters_run=40, best_iter=39, n_iter_cap=40, early_stopped=False)
        assert s.hit_cap and s.starved and s.wasted_iters == 0
        assert "STARVED" in s.describe()

    def test_over_running_is_counted_as_waste(self):
        s = LevelStats(iters_run=100, best_iter=4, n_iter_cap=100, early_stopped=False)
        assert not s.starved
        assert s.wasted_iters == 95
        assert "95 wasted" in s.describe()

    def test_early_stop_is_neither_starved_nor_capped(self):
        s = LevelStats(iters_run=12, best_iter=11, n_iter_cap=40, early_stopped=True)
        assert not s.hit_cap and not s.starved

    def test_as_dict_carries_the_derived_flags(self):
        d = LevelStats(iters_run=40, best_iter=39, n_iter_cap=40, early_stopped=False).as_dict()
        assert d["starved"] is True and d["hit_cap"] is True and d["wasted_iters"] == 0

    def test_fold_fallback_is_announced(self):
        s = LevelStats(
            iters_run=10, best_iter=0, n_iter_cap=10, early_stopped=False, fold_fallback=True
        )
        assert "NO LEGAL ITERATE" in s.describe()


class TestConvergenceReport:
    def test_summarises_per_backend_and_level(self, tmp_path):
        from fastfuncstuff.processing.tunestore import TrialStore, format_convergence

        store = TrialStore(tmp_path / "t.json")
        store.add(
            "optiwarp_demons",
            "s1",
            {},
            [],
            levels=[
                LevelStats(40, 39, 40, False).as_dict(),
                LevelStats(40, 2, 40, False, damped_iters=3).as_dict(),
            ],
        )
        text = format_convergence(store)
        assert "optiwarp_demons" in text
        assert "100%" in text, "level 1 was starved on every trial"

    def test_says_so_when_there_is_nothing_to_report(self, tmp_path):
        from fastfuncstuff.processing.tunestore import TrialStore, format_convergence

        assert "no convergence telemetry" in format_convergence(TrialStore(tmp_path / "t.json"))


class TestIterationCeiling:
    """The ceiling is measured, not searched.

    Too few iterations is always wrong and too many is free, so there is no interior
    optimum for a search to find. What is unknown is how many a level actually uses,
    and that is answered by running with the ceiling high and reading it back.
    """

    def _store(self, tmp_path, used, caps, starved):
        from fastfuncstuff.processing.tunestore import TrialStore

        store = TrialStore(tmp_path / "t.json")
        for u, c, st in zip(used, caps, starved, strict=True):
            store.add(
                "formwarp",
                "s1",
                {},
                [],
                levels=[LevelStats(u, u - 1 if st else 3, c, not st).as_dict()],
            )
        return store

    def test_recommends_headroom_over_the_worst_case(self, tmp_path):
        from fastfuncstuff.processing.tunestore import recommend_iterations

        store = self._store(tmp_path, [20, 30, 47], [300, 300, 300], [False] * 3)
        (a,) = recommend_iterations(store, headroom=2.0)
        assert a.max_used == 47
        assert a.recommended == 94, "the hardest subject sets the ceiling, not the mean"
        assert "about right" in a.verdict or "generous" in a.verdict

    def test_starving_is_called_out_as_too_low(self, tmp_path):
        from fastfuncstuff.processing.tunestore import recommend_iterations

        store = self._store(tmp_path, [40, 40, 40], [40, 40, 40], [True] * 3)
        (a,) = recommend_iterations(store)
        assert a.starved_frac == 1.0
        assert "TOO LOW" in a.verdict

    def test_generous_ceiling_is_not_a_complaint(self, tmp_path):
        from fastfuncstuff.processing.tunestore import recommend_iterations

        store = self._store(tmp_path, [12, 14, 15], [300, 300, 300], [False] * 3)
        (a,) = recommend_iterations(store)
        assert a.starved_frac == 0.0
        assert "generous" in a.verdict, "headroom is free; it must not read as a problem"

    def test_report_warns_when_a_measurement_is_only_a_floor(self, tmp_path):
        from fastfuncstuff.processing.tunestore import (
            format_iteration_advice,
            recommend_iterations,
        )

        store = self._store(tmp_path, [40, 40, 40], [40, 40, 40], [True] * 3)
        text = format_iteration_advice(recommend_iterations(store))
        assert "floor, not a measurement" in text


class TestCeilingIsNotSearched:
    def test_no_recipe_ladders_the_iteration_count(self):
        """Searching it asks a question whose direction is known: more."""
        from fastfuncstuff.processing.tunespec import RECIPES

        for r in RECIPES.values():
            assert "formwarp.iters" not in r.tune, r.name
            assert "optiwarp.iters" not in r.tune, r.name

    def test_the_stopping_rule_is_still_searched(self):
        """When to stop *is* a real choice, unlike how long you are allowed to."""
        from fastfuncstuff.processing.tunespec import RECIPES

        for r in RECIPES.values():
            assert "formwarp.conv_threshold" in r.tune, r.name
            assert "optiwarp.conv_window" in r.tune, r.name

    def test_defaults_leave_real_headroom(self):
        """A ceiling only helps if it is comfortably above what levels use."""
        from fastfuncstuff.processing.formwarp import SynConfig
        from fastfuncstuff.processing.optiwarp import OptiwarpConfig

        for cfg in (SynConfig(), OptiwarpConfig()):
            assert cfg.iterations == (300, 210, 120)

    def test_iters_remains_available_to_pin_or_tune(self):
        """Not searched by default is not the same as unreachable."""
        from fastfuncstuff.processing.tunespec import find_param, parse_fix

        assert find_param("formwarp.iters").config_attr == "iterations"
        assert parse_fix(["optiwarp.iters=100x70x40"]) == {"optiwarp.iters": (100, 70, 40)}


class TestCrawlingHint:
    """Early stopping not firing is not a bug — it means the iterations were used.

    The useful response is a bigger step, not only a bigger ceiling: a level that
    needs hundreds of iterations to flatten is stepping too small.
    """

    def _advice(self, backend, used, cap, starved=False):
        import tempfile

        from fastfuncstuff.processing.tunestore import TrialStore, recommend_iterations

        with tempfile.TemporaryDirectory() as d:
            store = TrialStore(f"{d}/t.json")
            store.add(
                backend,
                "s1",
                {},
                [],
                levels=[LevelStats(used, used - 1 if starved else 3, cap, not starved).as_dict()],
            )
            return recommend_iterations(store)[0]

    def test_burning_the_ceiling_suggests_a_bigger_step(self):
        a = self._advice("formwarp", used=250, cap=300)
        assert a.crawling
        assert "-grad_step" in a.verdict

    def test_optiwarp_gets_its_own_step_knob(self):
        assert "-max_step" in self._advice("optiwarp_demons", used=250, cap=300).verdict

    def test_qwarp_has_no_step_knob_to_suggest(self):
        """It optimises shrinking patches, so there is no single step size."""
        a = self._advice("qwarp", used=250, cap=300)
        assert a.step_knob is None
        assert "try a larger" not in a.verdict

    def test_converging_early_is_not_flagged_as_crawling(self):
        a = self._advice("formwarp", used=15, cap=300)
        assert not a.crawling
        assert "try a larger" not in a.verdict

    def test_starved_level_also_gets_the_step_hint(self):
        a = self._advice("formwarp", used=40, cap=40, starved=True)
        assert "TOO LOW" in a.verdict and "-grad_step" in a.verdict
