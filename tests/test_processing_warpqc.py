"""Tests for processing/warpqc.py — deformation-regularity metrics."""

import pytest
import torch

from fastfuncstuff.processing.warpqc import (
    FAIL,
    MARGINAL,
    UNCONSTRAINED_MARGIN,
    gate_margin,
    regularity_cautions,
    PASS,
    regularity_verdict,
    remedies,
    warp_regularity,
)

DEV = torch.device("cpu")
SHAPE = (24, 28, 26)


def _zeros():
    return (torch.zeros(SHAPE, device=DEV) for _ in range(3))


def _ramp(scale: float):
    """A uniform-gradient displacement along x: det(J) is constant at 1+scale."""
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in SHAPE), indexing="ij"
    )
    return xx * scale, torch.zeros(SHAPE), torch.zeros(SHAPE)


class TestIdentityWarp:
    def test_zero_displacement_is_perfectly_regular(self):
        qc = warp_regularity(*_zeros())
        assert qc.jac_neg_frac == 0.0
        assert qc.jac_p50 == pytest.approx(1.0, abs=1e-5)
        assert qc.disp_max_mm == pytest.approx(0.0)
        assert qc.bending_energy == pytest.approx(0.0, abs=1e-9)

    def test_identity_passes(self):
        grade, reasons = regularity_verdict(warp_regularity(*_zeros()))
        assert grade == PASS and reasons == []


class TestJacobian:
    def test_uniform_expansion(self):
        """A constant dx/dx = s gives det(J) = 1 + s everywhere."""
        qc = warp_regularity(*_ramp(0.5))
        assert qc.jac_p50 == pytest.approx(1.5, abs=1e-4)
        assert qc.jac_neg_frac == 0.0

    def test_compression_past_minus_one_folds(self):
        """dx/dx < -1 inverts the map: det(J) < 0 is folding, by definition."""
        qc = warp_regularity(*_ramp(-1.5))
        assert qc.jac_neg_frac > 0.9
        grade, reasons = regularity_verdict(qc)
        assert grade == FAIL
        assert any("folding" in r for r in reasons)

    def test_over_expansion_is_a_caution_not_a_verdict(self):
        """An extreme Jacobian is unusual anatomy, not impossible anatomy.

        Heads differ in shape and ventricles vary enormously, so a percentile
        past an arbitrary bound is worth saying out loud and not worth vetoing —
        unlike folding, which no head does at all.
        """
        qc = warp_regularity(*_ramp(5.0))
        grade, reasons = regularity_verdict(qc)
        assert grade == PASS and reasons == []
        assert any("over-expansion" in c for c in regularity_cautions(qc))

    def test_a_caution_never_masks_a_fold(self):
        """Both tripped at once must still fail: the rule outranks the notice."""
        qc = warp_regularity(*_ramp(-1.5))
        grade, reasons = regularity_verdict(qc)
        assert grade == FAIL
        assert any("folding" in r for r in reasons)
        assert regularity_cautions(qc)  # still reported, just not why it failed

    def test_clearance_ignores_the_cautions_it_reports(self):
        """A warp can be heavily compressed and still be nowhere near failing."""
        qc = warp_regularity(*_ramp(5.0))
        assert regularity_cautions(qc)  # extreme
        assert gate_margin(qc) == UNCONSTRAINED_MARGIN  # ... and nothing folded


class TestNegativeTolerance:
    def test_a_few_grazing_voxels_do_not_disqualify(self):
        """The user-visible requirement: 10 voxels at -0.01 is not a folded warp."""
        xd, yd, zd = (torch.zeros(SHAPE) for _ in range(3))
        qc = warp_regularity(xd, yd, zd)
        n = qc.n_voxels
        qc.jac_neg_count = 5  # a handful out of ~17k
        qc.jac_neg_frac = 5.0 / n
        grade, _ = regularity_verdict(qc)
        assert grade == PASS

    def test_a_real_fold_does_disqualify(self):
        xd, yd, zd = (torch.zeros(SHAPE) for _ in range(3))
        qc = warp_regularity(xd, yd, zd)
        qc.jac_neg_count = int(0.02 * qc.n_voxels)  # 2% of the brain inverted
        qc.jac_neg_frac = 0.02
        grade, reasons = regularity_verdict(qc)
        assert grade == FAIL and reasons

    def test_small_localised_folding_is_marginal_not_fatal(self):
        """The best candidate folding slightly is recoverable, not disqualified."""
        xd, yd, zd = (torch.zeros(SHAPE) for _ in range(3))
        qc = warp_regularity(xd, yd, zd)
        # Past the absolute "handful" budget (64) but well under the 0.5%
        # marginal ceiling -- the window where the fold is real but localised.
        qc.jac_neg_count = 75
        qc.jac_neg_frac = 75 / qc.n_voxels
        grade, reasons = regularity_verdict(qc)
        assert grade == MARGINAL and reasons

    def test_folding_reasons_carry_a_remedy(self):
        """A bad grade is a direction to move, not just a veto."""
        qc = warp_regularity(*_ramp(-1.5))
        _, reasons = regularity_verdict(qc)
        fix = remedies(reasons)
        assert fix and any("regulariz" in f for f in fix)


class TestMask:
    def test_mask_restricts_the_statistics(self):
        xd, yd, zd = _ramp(-1.5)  # folds everywhere
        mask = torch.zeros(SHAPE)
        mask[:4, :4, :4] = 1  # a corner
        qc = warp_regularity(xd, yd, zd, mask=mask)
        assert qc.n_voxels == 64

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="no voxels"):
            warp_regularity(*_zeros(), mask=torch.zeros(SHAPE))


class TestDisplacementUnits:
    def test_voxdims_scale_the_reported_mm(self):
        xd, yd, zd = (torch.zeros(SHAPE) for _ in range(3))
        xd = xd + 2.0  # 2 voxels along x
        one = warp_regularity(xd, yd, zd, voxdims=(1.0, 1.0, 1.0))
        two = warp_regularity(xd, yd, zd, voxdims=(2.0, 1.0, 1.0))
        assert one.disp_mean_mm == pytest.approx(2.0)
        assert two.disp_mean_mm == pytest.approx(4.0)


class TestBendingEnergy:
    def test_linear_field_has_no_bending(self):
        """Bending is a second derivative — a uniform gradient must not register."""
        qc = warp_regularity(*_ramp(0.3))
        assert qc.bending_energy == pytest.approx(0.0, abs=1e-6)

    def test_wiggly_field_has_more_bending_than_smooth(self):
        zz, yy, xx = torch.meshgrid(
            *(torch.arange(n, dtype=torch.float32) for n in SHAPE), indexing="ij"
        )
        zero = torch.zeros(SHAPE)
        smooth = warp_regularity(torch.sin(xx * 0.1), zero, zero)
        wiggly = warp_regularity(torch.sin(xx * 0.8), zero, zero)
        assert wiggly.bending_energy > smooth.bending_energy
