"""Tests for fastfuncstuff.stats.fdr (BH q-values, FDRCURVE)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.fdr import (
    compute_fdr_curve,
    fdr_qvalues,
    stat_to_pvalue,
)


def test_stat_to_pvalue_t_matches_scipy():
    from scipy.stats import t as scipy_t

    rng = np.random.default_rng(0)
    stats = rng.standard_normal(500) * 2.0
    p = stat_to_pvalue(torch.from_numpy(stats), "fitt", dof=30.0).numpy()
    p_ref = 2.0 * scipy_t.sf(np.abs(stats), df=30.0)
    np.testing.assert_allclose(p, p_ref, atol=1e-6)


def test_fdr_qvalues_under_null_no_rejections():
    rng = np.random.default_rng(1)
    # 5000 null t-stats at df=20
    stats = rng.standard_normal(5000)
    q = fdr_qvalues(torch.from_numpy(stats), stat_code="fitt", dof=20.0)
    # No voxel should have q < 0.05 under the null with this n.
    n_rej = int((q < 0.05).sum().item())
    assert n_rej < 100, f"too many false rejections under null: {n_rej}"


def test_fdr_qvalues_signal_recovered():
    rng = np.random.default_rng(2)
    null = rng.standard_normal(4000)
    signal = rng.standard_normal(1000) + 5.0  # strong activation
    stats = np.concatenate([null, signal])
    q = fdr_qvalues(torch.from_numpy(stats), stat_code="fitt", dof=100.0).numpy()
    # Most signal voxels should pass q<0.05
    rej_signal = int((q[4000:] < 0.05).sum())
    assert rej_signal > 800, f"signal recovery too low: {rej_signal}/1000"


def test_fdr_qvalues_mask_excludes_zeros():
    stats = torch.tensor([0.0, 0.0, 3.0, -3.5, 0.0, 4.0])
    q = fdr_qvalues(stats, stat_code="fitt", dof=50.0)
    # Zeros default-masked out → NaN
    assert torch.isnan(q[0]) and torch.isnan(q[1]) and torch.isnan(q[4])
    assert torch.isfinite(q[2]) and torch.isfinite(q[3]) and torch.isfinite(q[5])


def test_compute_fdr_curve_shape_and_monotone():
    rng = np.random.default_rng(3)
    null = rng.standard_normal(2000)
    sig = rng.standard_normal(500) + 4.0
    stats = np.concatenate([null, sig])
    curve = compute_fdr_curve(stats, "fitt", dof=80.0)
    assert curve["z"].shape == (101,)
    assert curve["dx"] > 0
    # Curve in |stat| direction should be non-decreasing (we built cummax).
    # Grid is signed, so we test on |grid| sorted order.
    grid = curve["x0"] + np.arange(101) * curve["dx"]
    abs_order = np.argsort(np.abs(grid))
    z_in_abs = curve["z"][abs_order]
    diffs = np.diff(z_in_abs)
    # Allow tiny float interp wobble
    assert (diffs >= -1e-5).all(), "curve not monotone in |stat|"


def test_fdr_addfdr_writer_roundtrip(tmp_path):
    """Synthetic NIfTI + AFNI extension round-trip."""
    import nibabel as nib

    from fastfuncstuff.stats.fdr import add_fdrcurves_to_nifti

    data = np.zeros((4, 4, 4, 2), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    afni_xml = '<?xml version="1.0" ?>\n<AFNI_attributes self_idcode="ABC">\n</AFNI_attributes>\n'
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, afni_xml.encode("utf-8")))
    out = tmp_path / "test.nii.gz"
    nib.save(img, str(out))

    curves = {
        0: {"x0": -5.0, "dx": 0.1, "z": np.linspace(0, 4, 101).astype(np.float32)},
        1: {"x0": 0.0, "dx": 0.05, "z": np.linspace(0, 3, 101).astype(np.float32)},
    }
    add_fdrcurves_to_nifti(out, curves)

    reloaded = nib.load(str(out))
    afni_ext = next(e for e in reloaded.header.extensions if e.get_code() == 4)
    xml = afni_ext.content.decode("utf-8")
    assert 'atr_name="FDRCURVE_000000"' in xml
    assert 'atr_name="FDRCURVE_000001"' in xml


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ---------------------------------------------------------------------------
# p <-> stat, as one pair
# ---------------------------------------------------------------------------


class TestPValueToStat:
    """The inverse of stat_to_pvalue, and it has to agree with it exactly.

    Two sidedness conventions in one viewer is a bug that presents as a
    rounding error: a threshold typed as p = 0.001 and a threshold reported as
    p = 0.001 would be different cuts.
    """

    def test_round_trips_against_its_own_forward(self):
        from fastfuncstuff.stats.fdr import pvalue_to_stat, stat_value_to_pvalue

        for code, dof in (("fitt", 100.0), ("fizt", None), ("fift", (3.0, 100.0))):
            for p in (0.05, 0.01, 0.001, 1e-5):
                value = pvalue_to_stat(p, code, dof)
                assert stat_value_to_pvalue(value, code, dof) == pytest.approx(p, rel=1e-5)

    def test_the_t_threshold_matches_afni(self):
        """t(120) at two-sided p = 0.001 is 3.3735 in AFNI's own readout."""
        from fastfuncstuff.stats.fdr import pvalue_to_stat

        assert pvalue_to_stat(0.001, "fitt", 120.0) == pytest.approx(3.3735, abs=5e-4)

    def test_f_is_upper_tail_not_halved(self):
        from scipy.stats import f as scipy_f

        from fastfuncstuff.stats.fdr import is_two_sided, pvalue_to_stat

        assert not is_two_sided("fift")
        assert pvalue_to_stat(0.01, "fift", (3.0, 50.0)) == pytest.approx(
            scipy_f.isf(0.01, dfn=3.0, dfd=50.0)
        )

    def test_t_and_z_are_two_tailed(self):
        from scipy.stats import norm

        from fastfuncstuff.stats.fdr import is_two_sided, pvalue_to_stat

        assert is_two_sided("fitt") and is_two_sided("fizt")
        assert pvalue_to_stat(0.05, "fizt") == pytest.approx(norm.isf(0.025))

    def test_a_p_outside_the_open_unit_interval_is_refused(self):
        from fastfuncstuff.stats.fdr import pvalue_to_stat

        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="p must be"):
                pvalue_to_stat(bad, "fizt")

    def test_missing_degrees_of_freedom_are_refused(self):
        from fastfuncstuff.stats.fdr import pvalue_to_stat

        with pytest.raises(ValueError, match="requires scalar dof"):
            pvalue_to_stat(0.01, "fitt")
        with pytest.raises(ValueError, match="requires dof"):
            pvalue_to_stat(0.01, "fift", 50.0)

    def test_an_unknown_code_is_refused(self):
        from fastfuncstuff.stats.fdr import pvalue_to_stat

        with pytest.raises(ValueError, match="Unsupported stat_code"):
            pvalue_to_stat(0.01, "fico", 50.0)
