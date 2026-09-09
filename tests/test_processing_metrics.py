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


class TestLabelDice:
    """The label-overlap panel: an independent referee, so it has to be exact."""

    @staticmethod
    def _blocks() -> torch.Tensor:
        seg = torch.zeros(12, 12, 12)
        seg[1:5, 1:5, 1:5] = 1
        seg[6:10, 6:10, 6:10] = 2
        seg[1:3, 6:8, 1:3] = 3
        return seg

    def test_identical_segmentations_score_one(self):
        from fastfuncstuff.processing.metrics import label_dice

        seg = self._blocks()
        d = label_dice(seg, seg)
        assert d.numel() == 3
        assert torch.allclose(d, torch.ones_like(d))

    def test_dice_matches_the_definition(self):
        from fastfuncstuff.processing.metrics import label_dice

        a = torch.zeros(6, 6, 6)
        a[0:4, 0, 0] = 1  # 4 voxels
        b = torch.zeros(6, 6, 6)
        b[2:6, 0, 0] = 1  # 4 voxels, 2 shared
        assert float(label_dice(a, b)[0]) == pytest.approx(2 * 2 / (4 + 4))

    def test_labels_absent_from_both_are_not_scored_as_zero(self):
        """Averaging in an undrawn parcel would punish a config for nothing."""
        from fastfuncstuff.processing.metrics import label_dice

        a = torch.zeros(6, 6, 6)
        a[0:2, 0, 0] = 1
        a[0:2, 1, 0] = 7  # nothing uses labels 2..6
        d = label_dice(a, a.clone())
        assert d.numel() == 2

    def test_a_label_in_one_volume_only_scores_zero(self):
        from fastfuncstuff.processing.metrics import label_dice

        a = torch.zeros(6, 6, 6)
        a[0:2, 0, 0] = 1
        a[0:2, 1, 0] = 2
        b = a.clone()
        b[b == 2] = 0
        d = label_dice(a, b)
        assert float(d[0]) == pytest.approx(1.0)
        assert float(d[1]) == pytest.approx(0.0)

    def test_mean_is_over_labels_not_voxels(self):
        """A big parcel and a small one count the same, which is the whole point."""
        from fastfuncstuff.processing.metrics import label_dice_summary

        a = torch.zeros(20, 20, 20)
        a[0:10, :, :] = 1  # 4000 voxels
        a[19, 0, 0:2] = 2  # 2 voxels
        b = a.clone()
        b[b == 2] = 0  # the small parcel is lost entirely
        s = label_dice_summary(a, b)
        assert s["n_labels"] == 2
        assert s["mean"] == pytest.approx(0.5)  # voxel-weighting would give ~1.0

    def test_dice_metrics_are_opt_in_to_a_panel(self):
        from fastfuncstuff.processing.metrics import LABEL_METRICS, panel_for

        assert not set(LABEL_METRICS) & set(panel_for("lpa", SAME))
        assert set(LABEL_METRICS) <= set(panel_for("lpa", SAME, labels=True))

    def test_evaluating_dice_without_labels_says_so(self):
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        inp = MetricInputs(base=torch.rand(6, 6, 6), moving=torch.rand(6, 6, 6))
        with pytest.raises(ValueError, match="segmentation"):
            evaluate_metrics(inp, ["dice"])

    def test_registry_reports_afni_convention(self):
        """Lower is better everywhere, so a perfect overlap must score 0."""
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        seg = self._blocks()
        inp = MetricInputs(
            base=torch.rand(12, 12, 12),
            moving=torch.rand(12, 12, 12),
            base_labels=seg,
            moving_labels=seg.clone(),
        )
        out = evaluate_metrics(inp, ["dice", "dice_q25"])
        assert out["dice"] == pytest.approx(0.0)
        assert out["dice_q25"] == pytest.approx(0.0)


class TestCrossSubjectAgreement:
    """Common-space scoring: agreement among a cohort, and where it fails."""

    @staticmethod
    def _cohort():
        a = torch.zeros(8, 8, 8)
        a[1:5, 1:5, 1:5] = 1
        a[6:8, 6:8, 6:8] = 2
        b = a.clone()
        b[1, 1, 1] = 0
        c = a.clone()
        c[c == 2] = 0  # this tracer did not draw label 2
        return [a, b, c]

    def test_identical_cohort_agrees_perfectly(self):
        from fastfuncstuff.processing.metrics import cross_subject_dice

        a = self._cohort()[0]
        out = cross_subject_dice([a, a.clone(), a.clone()])
        assert out["mean"] == pytest.approx(1.0)
        assert out["n_pairs"] == 3

    def test_every_unordered_pair_is_counted_once(self):
        from fastfuncstuff.processing.metrics import cross_subject_dice

        segs = self._cohort()
        assert cross_subject_dice(segs)["n_pairs"] == 3
        assert cross_subject_dice(segs + [segs[0].clone()])["n_pairs"] == 6

    def test_a_cohort_of_one_is_not_a_cohort(self):
        from fastfuncstuff.processing.metrics import cross_subject_dice

        with pytest.raises(ValueError, match="at least 2"):
            cross_subject_dice([self._cohort()[0]])

    def test_mismatched_grids_are_refused(self):
        from fastfuncstuff.processing.metrics import cross_subject_dice

        with pytest.raises(ValueError, match="common-space grid"):
            cross_subject_dice([torch.zeros(4, 4, 4), torch.zeros(4, 4, 5)])

    def test_a_missing_parcel_is_visible_as_missing(self):
        """The distinction the table exists for: a label the tracers left out
        must not read as a region the warp misplaced."""
        from fastfuncstuff.processing.metrics import cross_subject_detail

        d = cross_subject_detail(self._cohort(), ["s1", "s2", "s3"])
        assert d["labels"] == [1, 2]
        assert [int(x) for x in d["present"]] == [3, 2]  # only 2 subjects drew label 2
        assert float(d["per_label"][0]) > 0.99  # label 1 agrees
        assert float(d["per_label"][1]) < 0.4  # label 2 depressed by the absence

    def test_detail_reports_every_pair_by_every_label(self):
        from fastfuncstuff.processing.metrics import cross_subject_detail

        d = cross_subject_detail(self._cohort(), ["s1", "s2", "s3"])
        assert d["per_pair_label"].shape == (3, 2)
        assert d["pairs"] == [("s1", "s2"), ("s1", "s3"), ("s2", "s3")]
        assert d["volumes"].shape == (3, 2)

    def test_overlap_stack_is_one_frame_per_label(self):
        from fastfuncstuff.processing.metrics import label_overlap_stack

        segs = self._cohort()
        stack = label_overlap_stack(segs, [1, 2])
        assert stack.shape == (2, 8, 8, 8)
        assert float(stack[0, 2, 2, 2]) == pytest.approx(1.0)  # all three agree here
        assert float(stack[0, 1, 1, 1]) == pytest.approx(2 / 3)  # s2 dropped this voxel

    def test_overlap_denominator_excludes_subjects_lacking_the_label(self):
        """Otherwise a parcel two of three subjects drew perfectly reads as 0.67."""
        from fastfuncstuff.processing.metrics import label_overlap_stack

        stack = label_overlap_stack(self._cohort(), [1, 2])
        assert float(stack[1, 7, 7, 7]) == pytest.approx(1.0)

    def test_group_metrics_have_no_per_trial_value(self):
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        inp = MetricInputs(base=torch.rand(4, 4, 4), moving=torch.rand(4, 4, 4))
        with pytest.raises(ValueError, match="whole-cohort"):
            evaluate_metrics(inp, ["xdice"])
