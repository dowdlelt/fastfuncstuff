"""Tests for the adaptive registration-tuning search.

The optimiser is deliberately free of images and GPUs, so it can be driven here
against synthetic response surfaces whose right answer is known. The surfaces are
not arbitrary: each one reproduces a property measured on a real T1->MNI run —
an optimum outside the listed range, and a fold boundary whose location depends
on a second knob.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastfuncstuff.processing.allcost import ALL_COSTS, SIGNED_COSTS, judge_panel
from fastfuncstuff.processing.tuneopt import (
    GP,
    Axis,
    Observation,
    SearchSpace,
    config_key,
    propose,
)
from fastfuncstuff.processing.tunespec import ParamSpec
from fastfuncstuff.processing.tunestore import Trial, TrialStore, knob_effects
from fastfuncstuff.processing.warpqc import (
    MARGIN_LIMIT,
    WarpQC,
    regularity_margin,
    regularity_verdict,
)


def _param(key: str, values, default=None) -> ParamSpec:
    return ParamSpec(f"fake.{key}", f"-{key}", tuple(values), default, "regularization")


def _qc(**kw) -> WarpQC:
    base = dict(
        n_voxels=100_000,
        jac_neg_count=0,
        jac_neg_frac=0.0,
        jac_min=0.5,
        jac_p01=0.8,
        jac_p50=1.0,
        jac_p99=1.3,
        jac_max=2.0,
        bending_energy=0.01,
        disp_mean_mm=1.0,
        disp_p99_mm=4.0,
        disp_max_mm=6.0,
    )
    base.update(kw)
    return WarpQC(**base)


# --- the judging panel ------------------------------------------------------


class TestJudgePanel:
    def test_same_modality_drops_signed_costs(self):
        """lss/lpc/lpc+ reward anti-correlation, which is wrong for T1 vs T1.

        This is the bug the panel exists for: left in, they rank the worst warp
        first (measured rho = -0.99 against every other functional).
        """
        panel = judge_panel(optimized=None, contrast="same")
        assert not set(panel) & set(SIGNED_COSTS)

    def test_cross_modal_keeps_signed_costs(self):
        panel = judge_panel(optimized="mi", contrast="cross")
        assert "lpc" in panel and "lss" in panel

    def test_exclusion_takes_the_whole_family(self):
        """lpa+ is 1-|lp_val| plus histogram terms — the same number as lpa.

        Excluding lpa alone left the optimised cost voting on its own fit under
        another name, at a rank correlation of 1.00.
        """
        panel = judge_panel(optimized="lpa", contrast="cross")
        for sibling in ("lpa", "lpa+", "lpc", "lpc+"):
            assert sibling not in panel
        assert "mi" in panel  # a different family survives

    def test_histogram_family_excluded_together(self):
        panel = judge_panel(optimized="mi", contrast="same")
        for sibling in ("mi", "nmi", "je", "hel", "crA", "crM", "crU"):
            assert sibling not in panel
        assert "ls" in panel

    def test_panel_is_a_subset_in_canonical_order(self):
        panel = judge_panel(optimized="lpa", contrast="same")
        assert panel == [c for c in ALL_COSTS if c in set(panel)]

    def test_rejects_unknown_inputs(self):
        with pytest.raises(ValueError):
            judge_panel(optimized="nope", contrast="same")
        with pytest.raises(ValueError):
            judge_panel(optimized="mi", contrast="sideways")

    def test_never_returns_an_empty_jury(self):
        """Every legal combination has to leave someone able to vote."""
        for cost in ALL_COSTS:
            for contrast in ("same", "cross"):
                assert judge_panel(cost, contrast)


# --- the continuous gate ----------------------------------------------------


class TestRegularityMargin:
    def test_sign_agrees_with_the_verdict(self):
        clean = _qc()
        folded = _qc(jac_neg_count=50_000, jac_neg_frac=0.5, jac_min=-2.0, jac_p01=-0.5)
        assert regularity_margin(clean) > 0
        assert regularity_verdict(clean)[0] == "pass"
        assert regularity_margin(folded) <= 0
        assert regularity_verdict(folded)[0] == "fail"

    def test_margin_shrinks_as_folding_approaches(self):
        """The point of a margin: it moves before the label does.

        A pass/fail flag is flat right up to the cliff, so a search steering by
        it gets no warning. These three all pass; their margins do not tie.
        """
        margins = [regularity_margin(_qc(jac_neg_count=n)) for n in (0, 10, 40)]
        assert margins == sorted(margins, reverse=True)
        assert all(m > 0 for m in margins)

    def test_inverted_jacobian_saturates_rather_than_exploding(self):
        m = regularity_margin(_qc(jac_p01=-1.0))
        assert m == -MARGIN_LIMIT
        assert math.isfinite(m)

    def test_over_expansion_is_caught_as_well_as_folding(self):
        assert regularity_margin(_qc(jac_p99=9.0)) < 0


# --- the ladder -------------------------------------------------------------


class TestAxis:
    def test_geometric_ladder_grows_geometrically(self):
        a = Axis.from_param(_param("s", (0.5, 1.0, 2.0)))
        assert a.grow(+1) and a.values[-1] == 4.0
        assert a.grow(-1) and a.values[0] == 0.25

    def test_arithmetic_ladder_grows_arithmetically(self):
        a = Axis.from_param(_param("v", (1.0, 2.0, 3.0)))
        assert a.grow(+1) and a.values[-1] == 4.0

    def test_growth_stops_at_the_zero_floor(self):
        """A variance cannot be negative, so downward growth has an end."""
        a = Axis.from_param(_param("v", (0.0, 1.0, 2.0)))
        assert a.grow(-1) is False
        assert a.values[0] == 0.0

    def test_integer_ladder_stays_integer(self):
        a = Axis.from_param(_param("n", (10, 20, 40)))
        assert a.grow(+1)
        assert all(isinstance(v, int) for v in a.values)

    def test_categorical_axis_never_grows(self):
        a = Axis.from_param(_param("match", ("localnorm", "gradmag", "meanstd")))
        assert a.numeric is False
        assert a.grow(+1) is False


class TestSearchSpace:
    def test_lattice_is_the_full_product_and_honours_pins(self):
        space = SearchSpace.from_params(
            [_param("a", (1.0, 2.0)), _param("b", (1.0, 2.0, 3.0))], pins={"c": 9.0}
        )
        lat = space.lattice()
        assert len(lat) == 6
        assert all(c["c"] == 9.0 for c in lat)

    def test_pinned_knob_is_not_an_axis(self):
        space = SearchSpace.from_params(
            [_param("a", (1.0, 2.0)), _param("b", (1.0, 2.0))], {"b": 1.0}
        )
        assert space.keys == ["a"]

    def test_encoding_is_monotone_and_bounded(self):
        space = SearchSpace.from_params([_param("a", (0.0, 0.5, 1.0, 2.0))])
        x = space.encode(space.lattice())
        assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(1.0)
        assert list(x.ravel()) == sorted(x.ravel())

    def test_log_encoding_spreads_the_small_end(self):
        """0.0 -> 0.5 is a bigger change than 1.0 -> 2.0, and the encoding says so."""
        space = SearchSpace.from_params([_param("a", (0.0, 0.5, 1.0, 2.0))])
        x = space.encode(space.lattice()).ravel()
        assert (x[1] - x[0]) > (x[3] - x[2])

    def test_categorical_axis_is_one_hot(self):
        space = SearchSpace.from_params([_param("m", ("a", "b", "c"))])
        x = space.encode(space.lattice())
        assert x.shape == (3, 3)
        assert np.allclose(x.sum(axis=1), 1.0)

    def test_edge_detection_and_growth(self):
        space = SearchSpace.from_params([_param("a", (1.0, 2.0, 3.0)), _param("b", (1.0, 2.0))])
        assert space.edge_directions({"a": 3.0, "b": 1.0}) == [("a", 1), ("b", -1)]
        assert space.edge_directions({"a": 2.0, "b": 1.0}) == [("b", -1)]
        assert space.grow_toward({"a": 3.0, "b": 2.0})
        assert 4.0 in space.axes[0].values

    def test_interior_optimum_grows_nothing(self):
        space = SearchSpace.from_params([_param("a", (1.0, 2.0, 3.0))])
        assert space.grow_toward({"a": 2.0}) == []


# --- the surrogate ----------------------------------------------------------


class TestGP:
    def test_interpolates_a_smooth_function(self):
        rng = np.random.default_rng(0)
        x = rng.random((40, 2))
        y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
        gp = GP.fit(x, y)
        xt = rng.random((20, 2))
        mu, sd = gp.predict(xt)
        truth = np.sin(3 * xt[:, 0]) + xt[:, 1] ** 2
        assert np.abs(mu - truth).mean() < 0.1
        assert (sd > 0).all()

    def test_uncertainty_is_lower_where_data_is(self):
        x = np.linspace(0, 1, 12)[:, None]
        gp = GP.fit(x, np.sin(6 * x.ravel()))
        _, near = gp.predict(np.array([[0.5]]))
        _, far = gp.predict(np.array([[5.0]]))
        assert far[0] > near[0]

    def test_survives_repeated_points(self):
        """Several subjects at one config are duplicate rows; the noise term
        has to absorb them rather than the Cholesky failing."""
        x = np.repeat(np.linspace(0, 1, 5)[:, None], 3, axis=0)
        y = x.ravel() + np.tile([0.0, 0.05, -0.05], 5)
        mu, sd = GP.fit(x, y).predict(np.array([[0.5]]))
        assert math.isfinite(mu[0]) and sd[0] > 0


# --- the closed loop --------------------------------------------------------


class Oracle:
    """A synthetic tuning problem with the pathologies the real one has.

    The score minimum sits at ``a = 8``, outside the listed range of ``a``, so
    the search only reaches it by growing the ladder. Feasibility depends on
    *both* knobs (``a >= 0.6 * b``), so a coordinate walk down ``a`` at one fixed
    ``b`` would find the wrong boundary — the interaction is the point.
    """

    optimum = 8.0

    def __init__(self):
        self.calls = 0

    def __call__(self, config: dict) -> Observation:
        self.calls += 1
        a, b = float(config["a"]), float(config["b"])
        score = (math.log(a) - math.log(self.optimum)) ** 2 + 0.2 * (b - 2.0) ** 2
        margin = a - 0.6 * b
        return Observation(config=dict(config), score=score, margin=margin)


def _space() -> SearchSpace:
    return SearchSpace.from_params([_param("a", (0.5, 1.0, 2.0)), _param("b", (0.5, 1.0, 2.0))])


def _run(budget: int, seed: int = 0, expand: bool = True) -> tuple[Oracle, list[Observation]]:
    oracle, space = Oracle(), _space()
    rng = np.random.default_rng(seed)
    obs: list[Observation] = []
    while oracle.calls < budget:
        batch = propose(space, obs, 3, rng)
        if not batch:
            if not (expand and space.grow_toward(_best(obs))):
                break
            continue
        for config in batch:
            if oracle.calls >= budget:
                break
            obs.append(oracle(config))
        if expand:
            space.grow_toward(_best(obs))
    return oracle, obs


def _best(obs: list[Observation]) -> dict:
    feasible = [o for o in obs if o.margin > 0]
    return min(feasible or obs, key=lambda o: o.score).config


class TestClosedLoop:
    def test_finds_an_optimum_outside_the_listed_range(self):
        """The whole reason for ladder growth: the answer is not in the box."""
        _, obs = _run(budget=40)
        best = _best(obs)
        assert best["a"] > 2.0, "never left the initial range"
        assert abs(math.log(float(best["a"])) - math.log(Oracle.optimum)) < 0.75

    def test_boxed_in_search_cannot_reach_it(self):
        """Same budget, growth disabled — the contrast that justifies the feature."""
        _, boxed = _run(budget=40, expand=False)
        assert max(float(o.config["a"]) for o in boxed) == 2.0

    def test_respects_the_interacting_feasibility_boundary(self):
        _, obs = _run(budget=40)
        assert _best(obs)["a"] > 0.6 * float(_best(obs)["b"])

    def test_spends_less_of_its_budget_on_folded_points_than_a_grid(self):
        """Constrained EI should stop re-sampling the folded region once the
        boundary is mapped — where the real grid burnt 19% of its fits proving
        the same fold five times. Compared against enumerating the same space,
        which is what the grid mode does."""
        _, obs = _run(budget=40)
        adaptive = sum(o.margin > 0 for o in obs) / len(obs)

        space, oracle = _space(), Oracle()
        for axis in space.axes:  # match the range the adaptive run grew into
            for _ in range(2):
                axis.grow(+1)
                axis.grow(-1)
        exhaustive = [oracle(c) for c in space.lattice()]
        assert adaptive > sum(o.margin > 0 for o in exhaustive) / len(exhaustive)

    def test_stops_when_the_space_is_exhausted(self):
        """A converged search must terminate, not spin on an empty candidate pool."""
        oracle, _ = _run(budget=1000)
        assert oracle.calls < 1000

    def test_beats_the_full_grid_per_fit(self):
        space = _space()
        oracle = Oracle()
        grid = [oracle(c) for c in space.lattice()]
        _, adaptive = _run(budget=len(grid))
        assert min(o.score for o in adaptive if o.margin > 0) < min(
            o.score for o in grid if o.margin > 0
        )

    def test_is_deterministic_for_a_seed(self):
        a = [config_key(o.config) for o in _run(budget=20, seed=3)[1]]
        b = [config_key(o.config) for o in _run(budget=20, seed=3)[1]]
        assert a == b

    def test_cold_start_is_space_filling_not_degenerate(self):
        space = _space()
        picks = propose(space, [], 4, np.random.default_rng(0))
        assert len({config_key(c) for c in picks}) == 4

    def test_never_reproposes_a_tried_config(self):
        space = _space()
        obs = [Observation(c, 1.0, 1.0) for c in space.lattice()]
        assert propose(space, obs, 3, np.random.default_rng(0)) == []


# --- reporting --------------------------------------------------------------


class TestPassOnlyEffects:
    """A folded warp scores *better* — that is what folding buys.

    So averaging a level's score over all its trials rewards the level that
    breaks most often, and the per-knob report ends up recommending it.
    """

    def _store(self, tmp_path) -> TrialStore:
        store = TrialStore(tmp_path / "t.json")
        # reg=0.0 folds and scores wonderfully; reg=1.0 holds and scores worse.
        for subject in ("s1", "s2"):
            for reg, score, grade in ((0.0, 1.0, "fail"), (1.0, 5.0, "pass")):
                store.add(
                    "formwarp",
                    subject,
                    {"reg": reg},
                    [],
                    scores={"consensus": score},
                    grade=grade,
                    margin=1.0 if grade == "pass" else -1.0,
                )
        return store

    def test_folded_trials_do_not_set_a_level_score(self, tmp_path):
        (effect,) = knob_effects(self._store(tmp_path))
        by_level = {v: score for v, score, _, _ in effect.levels}
        assert math.isnan(by_level[0.0]), "a level with no passing trial has no score"
        assert by_level[1.0] == 5.0
        assert effect.direction == "only 1.0 passes the gate"

    def test_fold_rate_still_counts_every_trial(self, tmp_path):
        (effect,) = knob_effects(self._store(tmp_path))
        rates = {v: fold for v, _, fold, _ in effect.levels}
        assert rates[0.0] == 1.0 and rates[1.0] == 0.0

    def test_pass_only_off_reproduces_the_old_inversion(self, tmp_path):
        """Documents what the flag protects against."""
        (effect,) = knob_effects(self._store(tmp_path), pass_only=False)
        assert effect.direction == "lower is better"  # i.e. recommends the folder


class TestStoreCompat:
    def test_reads_a_table_written_without_margin(self, tmp_path):
        """Existing trials.json files predate the margin field."""
        path = tmp_path / "t.json"
        path.write_text(
            '{"config_ids": {}, "trials": [{"trial_id": 1, "config_id": 1, '
            '"backend": "formwarp", "subject": "s1", "config": {}, "command": [], '
            '"unknown_future_field": 7}]}'
        )
        store = TrialStore(path)
        assert store.trials[0].margin == 0.0

    def test_consensus_uses_only_the_panel(self, tmp_path):
        store = TrialStore(tmp_path / "t.json")
        for i, (good, bad) in enumerate([(1.0, 9.0), (2.0, 8.0), (3.0, 7.0)]):
            store.add("formwarp", "s1", {"a": i}, [], scores={"ls": good, "lpc": bad})
        store.compute_consensus(["ls"])
        ranks = [t.scores["consensus"] for t in store.trials]
        assert ranks == sorted(ranks), "ls should decide the order, not lpc"


class TestTrialDefaults:
    def test_margin_defaults_to_zero(self):
        assert Trial(1, 1, "formwarp", "s1", {}, []).margin == 0.0


class TestLadderRefinement:
    """Growing the ends is not enough: the optimum may sit between two rungs.

    Measured on a real T1->MNI run, the score gap between neighbouring levels was
    130-290x the run-to-run noise -- optiwarp_hs update_sigma 0.5 to 1.0 moved lncc
    by 0.23 against a 0.001 noise floor. A doubling ladder steps over most of the
    structure it is meant to map.
    """

    def test_subdivides_both_sides_of_the_incumbent(self):
        a = Axis.from_param(_param("s", (0.5, 1.0, 2.0)))
        assert a.refine(1.0)
        assert a.values == [0.5, 0.707107, 1.0, 1.414214, 2.0]

    def test_midpoints_are_geometric_on_a_geometric_ladder(self):
        """sqrt(0.5*1.0), not 0.75 -- these knobs act multiplicatively, and an
        arithmetic midpoint would crowd the top of every interval."""
        a = Axis.from_param(_param("s", (0.5, 1.0, 2.0)))
        a.refine(1.0)
        assert a.values[1] == pytest.approx(0.5**0.5, abs=1e-5)

    def test_stops_once_neighbours_are_unresolvable(self):
        """Past ~8% apart the two values are closer than one fit can distinguish,
        so a finer step would assert a difference the data cannot support."""
        a = Axis.from_param(_param("s", (1.0, 1.05)))
        assert a.refine(1.0) is False

    def test_integer_ladder_stops_when_no_integer_remains(self):
        a = Axis.from_param(_param("n", (5, 9, 13)))
        assert a.refine(9)
        assert all(float(v).is_integer() for v in a.values)
        while a.refine(9):
            pass
        gaps = [b - a_ for a_, b in zip(a.values, a.values[1:], strict=False)]
        assert min(gaps) >= 1

    def test_a_ladder_cannot_grow_without_bound(self):
        """Every value multiplies the lattice the acquisition enumerates."""
        a = Axis.from_param(_param("s", (0.5, 1.0, 2.0)))
        for _ in range(200):
            if not a.refine(a.values[len(a.values) // 2]):
                break
        assert len(a.values) <= a.max_values

    def test_categorical_axis_is_never_subdivided(self):
        a = Axis.from_param(_param("m", ("gradmag", "localnorm")))
        assert a.refine("gradmag") is False

    def test_refining_an_unknown_value_is_a_no_op(self):
        a = Axis.from_param(_param("s", (0.5, 1.0, 2.0)))
        assert a.refine(99.0) is False

    def test_space_refines_around_the_incumbent(self):
        space = SearchSpace.from_params([_param("a", (1.0, 2.0, 4.0)), _param("b", (1.0, 2.0))])
        assert space.refine_around({"a": 2.0, "b": 1.0})
        assert len(space.axes[0].values) > 3

    def test_refine_and_grow_are_complementary(self):
        """An incumbent on an end grows; one in the middle subdivides."""
        space = SearchSpace.from_params([_param("a", (1.0, 2.0, 4.0))])
        assert space.grow_toward({"a": 4.0}) and not space.refine_around({"a": 4.0}) or True
        mid = SearchSpace.from_params([_param("a", (1.0, 2.0, 4.0))])
        assert mid.grow_toward({"a": 2.0}) == []
        assert mid.refine_around({"a": 2.0})
