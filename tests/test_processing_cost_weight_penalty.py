import torch
import numpy as np
import pytest

from fastfuncstuff.processing.cost import (
    pearson_correlation,
    clipped_pearson_correlation,
    lpa_correlation,
    IncrementalCorrelation,
)
from fastfuncstuff.processing.weight import compute_weight_image
from fastfuncstuff.processing.penalty import (
    compute_jacobian_energy,
    compute_penalty,
    compute_penalty_batched,
)


class TestCostPearson:
    def test_pearson_identical(self):
        a = torch.randn(1000)
        r = pearson_correlation(a, a)
        assert r.item() == pytest.approx(1.0, abs=1e-6)

    def test_pearson_anticorrelated(self):
        a = torch.randn(1000)
        r = pearson_correlation(a, -a)
        assert r.item() == pytest.approx(-1.0, abs=1e-6)

    def test_pearson_independent(self):
        torch.manual_seed(42)
        a = torch.randn(10000)
        b = torch.randn(10000)
        r = pearson_correlation(a, b)
        assert abs(r.item()) < 0.05

    def test_clipped_pearson_identical(self):
        a = torch.randn(1000)
        r = clipped_pearson_correlation(a, a)
        assert r.item() == pytest.approx(1.0, abs=1e-6)


class TestCostLPA:
    def test_lpa_correlation_identical(self):
        a = torch.randn(10, 10, 10)
        r = lpa_correlation(a, a, sigma=2.0)
        assert r.item() > 0.9


class TestCostIncremental:
    def test_incremental_correlation_identical(self):
        a = torch.randn(1000)
        w = torch.ones(1000)
        inc = IncrementalCorrelation(method="pearclp")
        inc.add_fixed(a[:500], a[:500], w[:500])
        r = inc.evaluate(a[500:], a[500:], w[500:])
        assert r > 0.9


class TestWeight:
    def test_weight_range(self):
        base = torch.randn(15, 15, 15)
        w = compute_weight_image(base)
        assert w.min() >= 0.0
        assert w.max() <= 1.0

    def test_weight_edges_zero(self):
        base = torch.randn(15, 15, 15).abs() + 1.0
        w = compute_weight_image(base, gauss_fwhm=0.0)
        xfade = max(2, int(0.04 * 15 + 2))
        assert w[:, :, :xfade].max() == 0.0
        assert w[:, :, -xfade:].max() == 0.0
        assert w[:, :xfade, :].max() == 0.0
        assert w[:, -xfade:, :].max() == 0.0
        assert w[:xfade, :, :].max() == 0.0
        assert w[-xfade:, :, :].max() == 0.0

    def test_weight_nonzero_interior(self):
        base = torch.randn(15, 15, 15).abs() + 1.0
        w = compute_weight_image(base)
        xfade = max(2, int(0.04 * 15 + 2))
        interior = w[xfade:-xfade, xfade:-xfade, xfade:-xfade]
        assert interior.max() > 0.0


class TestPenalty:
    def test_jacobian_energy_zero_displacement(self):
        xd = torch.zeros(5, 5, 5)
        yd = torch.zeros(5, 5, 5)
        zd = torch.zeros(5, 5, 5)
        je, se = compute_jacobian_energy(xd, yd, zd)
        assert je.max().item() == pytest.approx(0.0, abs=1e-6)
        assert se.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_zero_displacement(self):
        xd = torch.zeros(5, 5, 5)
        yd = torch.zeros(5, 5, 5)
        zd = torch.zeros(5, 5, 5)
        p = compute_penalty(xd, yd, zd)
        assert p == 0.0

    def test_penalty_nonzero(self):
        xd = torch.randn(5, 5, 5) * 0.5
        yd = torch.randn(5, 5, 5) * 0.5
        zd = torch.randn(5, 5, 5) * 0.5
        p = compute_penalty(xd, yd, zd)
        assert p > 0.0

    def test_penalty_batched_zero(self):
        B = 3
        xd = torch.zeros(B, 5, 5, 5)
        yd = torch.zeros(B, 5, 5, 5)
        zd = torch.zeros(B, 5, 5, 5)
        ext = torch.zeros(B)
        result = compute_penalty_batched(xd, yd, zd, pen_fac=0.033333, external_sums=ext)
        assert torch.all(result == 0.0).item()

    def test_penalty_batched_consistency(self):
        B = 4
        torch.manual_seed(0)
        xd = torch.randn(B, 5, 5, 5) * 0.3
        yd = torch.randn(B, 5, 5, 5) * 0.3
        zd = torch.randn(B, 5, 5, 5) * 0.3
        pen_fac = 0.05
        ext = torch.zeros(B)

        batched = compute_penalty_batched(xd, yd, zd, pen_fac=pen_fac, external_sums=ext)

        for i in range(B):
            individual = compute_penalty(xd[i], yd[i], zd[i], pen_fac=pen_fac)
            assert batched[i].item() == pytest.approx(individual, abs=1e-5)
