"""Tests for processing/allcost.py — the all-cost report behind ffs_util_cost."""

import pytest
import torch

from fastfuncstuff.processing.allcost import (
    ALL_COSTS,
    build_cost_inputs,
    consensus_rank,
    evaluate_all_costs,
)
from fastfuncstuff.processing.cost import spearman_correlation

DEV = torch.device("cpu")


def _blob(shape=(40, 44, 40), shift=(0, 0, 0), seed=0) -> torch.Tensor:
    """A smooth ellipsoidal 'brain' with texture, optionally translated."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij"
    )
    ctr = [(n - 1) / 2.0 + s for n, s in zip(shape, shift, strict=True)]
    r2 = sum(((c - m) / (n / 3.0)) ** 2 for c, m, n in zip((zz, yy, xx), ctr, shape, strict=True))
    vol = torch.exp(-r2 * 2.0)
    vol = vol + 0.05 * torch.randn(shape, generator=g)
    return (vol * (r2 < 1.0)).clamp(min=0.0).to(DEV)


def _inputs(base, source, n_match=1.0):
    return build_cost_inputs(base, source, None, (1.0, 1.0, 1.0), n_match)


# ── spearman_correlation ──


class TestSpearman:
    def test_identical(self):
        x = torch.randn(5000, device=DEV)
        assert abs(float(spearman_correlation(x, x)) - 1.0) < 1e-4

    def test_monotone_nonlinear_is_still_one(self):
        """The point of a rank correlation: it sees through a monotone warp."""
        x = torch.rand(5000, device=DEV) + 0.1
        y = x.pow(3.0)
        assert abs(float(spearman_correlation(x, y)) - 1.0) < 1e-4
        # Pearson, by contrast, is degraded by the same warp.
        from fastfuncstuff.processing.cost import pearson_correlation

        assert float(pearson_correlation(x, y)) < 0.98

    def test_large_input_does_not_overflow(self):
        """Ranks run to N; squaring them in float32 used to collapse r to 0."""
        n = 3_000_000
        x = torch.randn(n, device=DEV)
        r = float(spearman_correlation(x, x))
        assert abs(r - 1.0) < 1e-3

    def test_ties_are_averaged(self):
        x = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0], device=DEV)
        y = torch.tensor([5.0, 5.0, 5.0, 6.0, 7.0], device=DEV)
        assert abs(float(spearman_correlation(x, y)) - 1.0) < 1e-5


# ── evaluate_all_costs ──


class TestEvaluateAllCosts:
    def test_all_costs_present_and_finite(self):
        base = _blob()
        vals = evaluate_all_costs(_inputs(base, base.clone()))
        assert list(vals) == ALL_COSTS
        assert all(torch.isfinite(torch.tensor(v)) for v in vals.values())

    def test_perfect_match_scores_near_zero_on_ls(self):
        base = _blob()
        vals = evaluate_all_costs(_inputs(base, base.clone()), costs=["ls", "sp", "lss"])
        assert vals["ls"] < 1e-4  # 1 - |r|, r == 1
        assert vals["sp"] < 1e-3
        assert vals["lss"] > 0.99  # signed correlation, reported as-is

    @pytest.mark.parametrize("cost", ["ls", "sp", "mi", "nmi", "je", "crU", "crA", "crM", "lpa"])
    def test_misalignment_is_worse(self, cost):
        """Every functional must prefer the aligned pair to a 6-voxel shift."""
        base = _blob()
        aligned = _blob(seed=1)
        shifted = _blob(shift=(6, 0, 0), seed=1)
        good = evaluate_all_costs(_inputs(base, aligned), costs=[cost])[cost]
        bad = evaluate_all_costs(_inputs(base, shifted), costs=[cost])[cost]
        assert good < bad, f"{cost}: aligned {good} should beat shifted {bad}"

    def test_lpa_needs_coords(self):
        base = _blob()
        from fastfuncstuff.processing.allcost import CostInputs

        inp = CostInputs(base.reshape(-1), base.reshape(-1), None, None, None)
        with pytest.raises(ValueError, match="coords_mm"):
            evaluate_all_costs(inp, costs=["lpa"])

    def test_unknown_cost_rejected(self):
        base = _blob()
        with pytest.raises(ValueError, match="Unknown cost"):
            evaluate_all_costs(_inputs(base, base), costs=["nope"])

    def test_combo_costs_reduce_to_weighted_sum(self):
        """lpc+/lpa+ are their base cost plus the weighted standalone terms."""
        from fastfuncstuff.processing.allcost import MICHO_LPA

        base = _blob()
        src = _blob(shift=(2, 0, 0), seed=1)
        v = evaluate_all_costs(_inputs(base, src))
        w_hel, w_mi, w_nmi, w_cra, _ = MICHO_LPA
        expect = (
            v["lpa"]
            + w_hel * v["hel"]
            + w_mi * v["mi"]
            + w_nmi * v["nmi"]
            + w_cra * v["crA"]
        )
        assert abs(v["lpa+"] - expect) < 1e-5


# ── build_cost_inputs ──


class TestBuildCostInputs:
    def test_restricts_to_the_weight_domain(self):
        """The background must be excluded — it is what dilutes a blok cost."""
        base = _blob()
        weight = (base > 0.05).float()
        inp = build_cost_inputs(base, base.clone(), weight, (1.0, 1.0, 1.0))
        assert inp.base.numel() == int(weight.sum())
        assert inp.base.numel() < base.numel()

    def test_n_match_subsamples(self):
        base = _blob()
        weight = (base > 0.05).float()
        full = build_cost_inputs(base, base, weight, (1.0, 1.0, 1.0), n_match=1.0)
        half = build_cost_inputs(base, base, weight, (1.0, 1.0, 1.0), n_match=0.5)
        assert half.base.numel() == pytest.approx(full.base.numel() // 2, rel=0.01)


# ── consensus_rank ──


class TestConsensusRank:
    def test_ranks_by_mean_position(self):
        cands = {
            "good": {"a": 0.1, "b": 0.1, "c": 0.9},
            "mid": {"a": 0.2, "b": 0.2, "c": 0.1},
            "bad": {"a": 0.3, "b": 0.3, "c": 0.95},
        }
        order = [n for n, _ in consensus_rank(cands)]
        assert order[0] in ("good", "mid")
        assert order[-1] == "bad"

    def test_majority_beats_a_single_dissenting_metric(self):
        """A candidate that wins only the cost it optimised does not win."""
        cands = {
            "self_optimised": {"a": 0.0, "b": 0.9, "c": 0.9, "d": 0.9},
            "broadly_good": {"a": 0.5, "b": 0.1, "c": 0.1, "d": 0.1},
        }
        assert consensus_rank(cands)[0][0] == "broadly_good"

    def test_empty(self):
        assert consensus_rank({}) == []
