import pytest
import torch

from fastfuncstuff.processing.cost import (
    IncrementalCorrelation,
    clipped_pearson_correlation,
    lpa_correlation,
    pearson_correlation,
)
from fastfuncstuff.processing.penalty import (
    compute_jacobian_energy,
    compute_penalty,
    compute_penalty_batched,
)
from fastfuncstuff.processing.weight import compute_weight_image


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


class TestPenaltyDeadband:
    """The warp penalty must be dormant on benign deformation.

    AFNI excludes voxel-wise energy below ``Hpen_cut`` and raises only the excess
    to the 4th power (mri_nwarp.c:2283). We summed ``je + se`` raw, which taxed
    every voxel of a perfectly sound warp instead of only the misbehaving ones.
    Measured consequence on a real T1->MNI pair: the warp shrank to 22% of AFNI's
    displacement and every pyramid level past the second made the fit *worse*.
    """

    def _fields(self, scale, shape=(16, 16, 16)):
        import torch

        z, y, x = torch.meshgrid(*[torch.linspace(0, 1, s) for s in shape], indexing="ij")
        return scale * torch.sin(3 * x), scale * torch.sin(3 * y), scale * torch.zeros_like(z)

    def test_benign_warp_costs_nothing(self):
        from fastfuncstuff.processing.penalty import compute_jacobian_energy, penalty_energy

        je, se = compute_jacobian_energy(*self._fields(0.05))
        assert float(je.max()) < 1.0, "fixture is not actually benign"
        assert float(penalty_energy(je, se).sum()) == 0.0

    def test_extreme_warp_is_charged(self):
        import torch

        from fastfuncstuff.processing.penalty import penalty_energy

        je = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
        se = torch.zeros_like(je)
        e = penalty_energy(je, se)
        assert list(e[:3]) == [0.0, 0.0, 0.0], "at or below the cut must be free"
        assert float(e[3]) == 1.0  # (2-1)^4
        assert float(e[4]) == 16.0  # (3-1)^4

    def test_growth_is_quartic_in_the_excess(self):
        """Paired with the ^0.25 on the total, a single dominant voxel contributes
        (ev^4)^0.25 = ev -- a soft maximum over the worst excess, not an average."""
        import torch

        from fastfuncstuff.processing.penalty import penalty_energy

        je = torch.tensor([2.0, 3.0])  # excess 1 and 2
        e = penalty_energy(je, torch.zeros_like(je))
        assert float(e[1]) / float(e[0]) == 16.0

    def test_penalty_is_zero_for_a_sound_warp_end_to_end(self):
        from fastfuncstuff.processing.penalty import compute_penalty

        assert compute_penalty(*self._fields(0.05), pen_fac=0.033) == 0.0

    def test_penalty_engages_once_a_voxel_inverts(self):
        import torch

        from fastfuncstuff.processing.penalty import compute_penalty

        # A ramp steep enough to invert the Jacobian somewhere.
        n = 12
        r = torch.zeros(n, n, n)
        r[:, :, n // 2 :] = 4.0
        z = torch.zeros_like(r)
        assert compute_penalty(r, z, z, pen_fac=0.033) > 0.0

    def test_batched_and_serial_agree(self):
        import torch

        from fastfuncstuff.processing.penalty import compute_penalty, compute_penalty_batched

        xd, yd, zd = self._fields(0.9)
        serial = compute_penalty(xd, yd, zd, pen_fac=0.033)
        batched = compute_penalty_batched(xd[None], yd[None], zd[None], 0.033, torch.zeros(1))
        assert float(batched[0]) == pytest.approx(serial, rel=1e-5)
