"""Tests for the unified, modality-tagged metric registry.

The registry exists to stop a metric having two identities — one the engines
could optimise and one the evaluator could score. So these check both halves of
that promise, and check the tags actually gate what they claim to gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.metrics import (
    AFNI_METRICS,
    ALL_METRICS,
    CROSS,
    GRID_METRICS,
    METRICS,
    SAME,
    MetricInputs,
    check_contrast,
    describe_metrics,
    differentiable_cost,
    differentiable_metrics,
    evaluate_metrics,
    metric,
    mind_descriptor,
    panel_for,
)

CPU = torch.device("cpu")


def _blob(shape, cx, r, amp=100.0):
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    c = [s // 2 for s in shape]
    return (
        amp * np.exp(-(((x - cx) ** 2 + (y - c[1]) ** 2 + (z - c[0]) ** 2) / (2 * r**2)))
    ).astype(np.float32)


@pytest.fixture(scope="module")
def pair():
    shape = (32, 32, 32)
    base = torch.from_numpy(_blob(shape, 16, 7) + _blob(shape, 11, 2.2, 40.0))

    def shifted(d):
        return torch.from_numpy(_blob(shape, 16 + d, 7) + _blob(shape, 11 + d, 2.2, 40.0))

    return base, shifted


# --- the declaration -------------------------------------------------------


class TestRegistry:
    def test_afni_set_is_complete_and_unchanged(self):
        """These are 3dAllineate's 14. Parity is the contract."""
        assert AFNI_METRICS == [
            "ls", "sp", "mi", "crM", "nmi", "je", "hel",
            "crA", "crU", "lss", "lpc", "lpa", "lpc+", "lpa+",
        ]  # fmt: skip

    def test_registry_agrees_with_allcost(self):
        from fastfuncstuff.processing.allcost import ALL_COSTS

        assert AFNI_METRICS == list(ALL_COSTS)

    def test_signed_metrics_are_cross_only(self):
        """Signed means 'more anti-correlated is better', which is meaningless
        between two images of the same modality."""
        for m in METRICS.values():
            if m.signed:
                assert m.contrast == (CROSS,), m.name

    def test_new_metrics_are_all_optimisable(self):
        """The point of one registry: they are judges AND objectives."""
        for name in GRID_METRICS:
            assert METRICS[name].differentiable, name

    def test_afni_metrics_are_not_grid_based(self):
        """They are computed from scattered in-mask points, not a volume."""
        for name in AFNI_METRICS:
            assert not METRICS[name].needs_grid

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError, match="unknown metric"):
            metric("nope")

    def test_description_table_covers_everything(self):
        text = describe_metrics()
        for name in ALL_METRICS:
            assert name in text


# --- the tags actually gate things -----------------------------------------


class TestPanelSelection:
    def test_same_modality_excludes_signed(self):
        assert not [n for n in panel_for(contrast=SAME) if METRICS[n].signed]

    def test_cross_modality_keeps_signed(self):
        assert "lss" in panel_for(optimized="mi", contrast=CROSS)

    def test_exclusion_takes_the_whole_family(self):
        panel = panel_for(optimized="lpa", contrast=CROSS)
        for sibling in ("lpa", "lpa+", "lpc", "lpc+"):
            assert sibling not in panel

    def test_optimising_lncc_excludes_only_its_family(self):
        """The gap this registry closed: lncc was optimisable but unscoreable,
        so it could neither be excluded nor allowed to vote."""
        panel = panel_for(optimized="lncc", contrast=SAME)
        assert "lncc" not in panel
        assert "ngf" in panel and "mind" in panel

    def test_grid_false_drops_the_neighbourhood_metrics(self):
        panel = panel_for(contrast=SAME, grid=False)
        assert not set(panel) & set(GRID_METRICS)
        assert "ls" in panel

    def test_panel_is_never_empty(self):
        for name in ALL_METRICS:
            for contrast in (SAME, CROSS):
                if METRICS[name].usable_for(contrast):
                    assert panel_for(name, contrast)

    def test_bad_contrast_is_rejected(self):
        with pytest.raises(ValueError, match="contrast must be"):
            panel_for(contrast="sideways")

    def test_check_contrast_refuses_a_meaningless_pairing(self):
        """Fails loudly rather than degrading: a signed metric on same-modality
        data optimises for the wrong answer."""
        with pytest.raises(ValueError, match="not meaningful"):
            check_contrast("lpc", SAME)
        check_contrast("lpc", CROSS)  # fine

    def test_differentiable_list_respects_contrast(self):
        assert "mse" in differentiable_metrics(SAME)
        assert "mse" not in differentiable_metrics(CROSS)


# --- the numbers ------------------------------------------------------------


class TestGridMetrics:
    def test_every_grid_metric_falls_as_alignment_improves(self, pair):
        base, shifted = pair
        for name in GRID_METRICS:
            vals = [
                evaluate_metrics(MetricInputs(base=base, moving=shifted(d)), [name])[name]
                for d in (4, 2, 0)
            ]
            assert vals[0] > vals[1] > vals[2], f"{name} is not monotone: {vals}"

    def test_perfect_alignment_is_the_minimum(self, pair):
        base, shifted = pair
        for name in GRID_METRICS:
            aligned = evaluate_metrics(MetricInputs(base=base, moving=base), [name])[name]
            off = evaluate_metrics(MetricInputs(base=base, moving=shifted(3)), [name])[name]
            assert aligned < off, name

    @pytest.mark.parametrize("name", ["ngf", "mind", "mindssc"])
    def test_contrast_invariant_metrics_are_exactly_invariant(self, pair, name):
        """Not 'roughly holds up' — inverting one image must not move the number
        at all. That is the property these are chosen for."""
        base, shifted = pair
        moving = shifted(2)
        inverted = moving.max() - moving
        a = evaluate_metrics(MetricInputs(base=base, moving=moving), [name])[name]
        b = evaluate_metrics(MetricInputs(base=base, moving=inverted), [name])[name]
        assert a == pytest.approx(b, rel=1e-5), f"{name} moved under contrast inversion"

    def test_mind_descriptor_shape_and_range(self, pair):
        base, _ = pair
        d = mind_descriptor(base)
        assert d.shape == (6, *base.shape)
        assert float(d.min()) >= 0.0 and float(d.max()) == pytest.approx(1.0)

    def test_ssc_uses_twelve_channels(self, pair):
        """SSC is defined on the twelve neighbour pairs at distance sqrt(2)."""
        base, _ = pair
        assert mind_descriptor(base, ssc=True).shape[0] == 12

    def test_descriptors_are_finite_on_a_flat_region(self):
        """A constant block has zero local variance; the floor must hold."""
        flat = torch.ones(12, 12, 12)
        d = mind_descriptor(flat)
        assert torch.isfinite(d).all()

    def test_weighting_restricts_where_the_metric_looks(self, pair):
        base, shifted = pair
        moving = shifted(3)
        w = torch.zeros_like(base)
        w[:8] = 1.0  # a slab far from the displaced structure
        full = evaluate_metrics(MetricInputs(base=base, moving=moving), ["mse"])["mse"]
        slab = evaluate_metrics(MetricInputs(base=base, moving=moving, weight=w), ["mse"])["mse"]
        assert slab < full, "a region with little structure should disagree less"


class TestOneSurface:
    def test_afni_and_grid_metrics_evaluate_in_one_call(self, pair):
        base, shifted = pair
        out = evaluate_metrics(
            MetricInputs(base=base, moving=shifted(2)), ["ls", "mi", "lncc", "mind"]
        )
        assert set(out) == {"ls", "mi", "lncc", "mind"}
        assert all(np.isfinite(v) for v in out.values())

    def test_results_are_returned_in_the_requested_order(self, pair):
        base, shifted = pair
        want = ["mind", "ls", "ngf"]
        assert list(evaluate_metrics(MetricInputs(base=base, moving=shifted(1)), want)) == want

    def test_afni_numbers_match_allcost_exactly(self, pair):
        """Delegation, not reimplementation — the AFNI values must not drift."""
        from fastfuncstuff.processing.allcost import build_cost_inputs, evaluate_all_costs

        base, shifted = pair
        moving = shifted(2)
        via_registry = evaluate_metrics(MetricInputs(base=base, moving=moving), AFNI_METRICS)
        direct = evaluate_all_costs(
            build_cost_inputs(base, moving, None, (1.0, 1.0, 1.0), 1.0, "tohd")
        )
        for name in AFNI_METRICS:
            assert via_registry[name] == pytest.approx(direct[name], rel=1e-9), name

    def test_unknown_name_is_rejected_before_any_work(self, pair):
        base, shifted = pair
        with pytest.raises(ValueError, match="unknown metric"):
            evaluate_metrics(MetricInputs(base=base, moving=shifted(0)), ["ls", "bogus"])


class TestDifferentiability:
    @pytest.mark.parametrize("name", GRID_METRICS)
    def test_gradient_flows_to_the_moving_image(self, pair, name):
        base, shifted = pair
        moving = shifted(2).clone().requires_grad_(True)
        differentiable_cost(name, base, moving).backward()
        assert moving.grad is not None
        assert torch.isfinite(moving.grad).all()
        assert float(moving.grad.abs().sum()) > 0, f"{name} produced no gradient"

    def test_non_differentiable_metric_is_refused(self, pair):
        base, shifted = pair
        with pytest.raises(ValueError, match="not differentiable"):
            differentiable_cost("mi", base, shifted(1))

    def test_the_engines_can_reach_the_new_metrics(self, pair):
        """The other half of one surface: declared differentiable here means
        usable as an objective there, with no second list to update."""
        from fastfuncstuff.processing.formwarp import METRICS as ENGINE_METRICS
        from fastfuncstuff.processing.formwarp import image_metric

        base, shifted = pair
        for name in ("lncc", "ngf", "mind"):
            assert name in ENGINE_METRICS
            v = image_metric(base, shifted(2), torch.ones_like(base), metric=name)
            assert torch.isfinite(v)


class TestPatchWiseForms:
    """qwarp optimises flat (B, V) patches, so the grid metrics need a patch form.

    A patch is an (nzh, nyh, nxh) block that was flattened, so the structure the
    neighbourhood metrics need is recoverable by reshaping.
    """

    def _patches(self, b=3, n=9):
        torch.manual_seed(0)
        base = torch.rand(b, n * n * n)
        return base, base.clone(), torch.ones(b, n * n * n), n

    def test_every_patch_metric_peaks_at_a_perfect_match(self):
        from fastfuncstuff.processing.metrics import PATCH_METRICS, batched_patch_cost

        base, same, w, n = self._patches()
        worse = torch.rand_like(base)
        for name in PATCH_METRICS:
            good = batched_patch_cost(name, base, same, w, n, n, n)
            bad = batched_patch_cost(name, base, worse, w, n, n, n)
            assert (good > bad).all(), f"{name} did not prefer the exact match"

    def test_returns_one_value_per_patch(self):
        from fastfuncstuff.processing.metrics import PATCH_METRICS, batched_patch_cost

        base, other, w, n = self._patches(b=5)
        for name in PATCH_METRICS:
            assert batched_patch_cost(name, base, other, w, n, n, n).shape == (5,)

    def test_patches_are_scored_independently(self):
        """A batch must not leak between patches -- each is a separate problem."""
        from fastfuncstuff.processing.metrics import batched_patch_cost

        base, _, w, n = self._patches(b=2)
        moving = base.clone()
        moving[1] = torch.rand_like(moving[1])  # only the second patch disagrees
        out = batched_patch_cost("lncc", base, moving, w, n, n, n)
        alone = batched_patch_cost("lncc", base[:1], moving[:1], w[:1], n, n, n)
        assert float(out[0]) == pytest.approx(float(alone[0]), rel=1e-5)

    def test_gradient_reaches_the_patch_values(self):
        from fastfuncstuff.processing.metrics import PATCH_METRICS, batched_patch_cost

        base, _, w, n = self._patches()
        for name in PATCH_METRICS:
            moving = torch.rand_like(base).requires_grad_(True)
            batched_patch_cost(name, base, moving, w, n, n, n).sum().backward()
            assert moving.grad is not None and torch.isfinite(moving.grad).all(), name
            assert float(moving.grad.abs().sum()) > 0, name

    def test_lncc_window_is_clamped_to_the_patch(self):
        """An oversized window makes every voxel see whole-patch statistics, which
        silently turns the local metric into a global one."""
        from fastfuncstuff.processing.metrics import batched_patch_cost

        base, _, w, n = self._patches(n=5)
        moving = torch.rand_like(base)
        big = batched_patch_cost("lncc", base, moving, w, n, n, n, cc_radius=64)
        assert torch.isfinite(big).all()

    def test_unknown_metric_has_no_patch_form(self):
        from fastfuncstuff.processing.metrics import batched_patch_cost

        base, other, w, n = self._patches()
        with pytest.raises(ValueError, match="no patch-wise form"):
            batched_patch_cost("mi", base, other, w, n, n, n)
