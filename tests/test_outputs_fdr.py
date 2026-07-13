"""Smoke tests for -add_fdr plumbing in glm/outputs.py."""

from __future__ import annotations

import numpy as np
import pytest


def test_inject_fdr_curves_writes_attrs(tmp_path):
    import nibabel as nib

    from fastfuncstuff.glm.outputs import _inject_fdr_curves

    rng = np.random.default_rng(0)
    # 10×10×10×3 bucket: brick0 = beta, brick1 = t-stat, brick2 = F-stat
    nx, ny, nz = 10, 10, 10
    data = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    data[..., 0] = rng.standard_normal((nx, ny, nz))  # beta
    data[..., 1] = rng.standard_normal((nx, ny, nz)) * 2.0  # t
    # Add some "signal" voxels
    data[2:5, 2:5, 2:5, 1] += 4.0
    data[..., 2] = rng.chisquare(2, (nx, ny, nz)) + 0.5  # F-ish

    img = nib.Nifti1Image(data, affine=np.eye(4))
    afni_xml = '<?xml version="1.0" ?>\n<AFNI_attributes self_idcode="X">\n</AFNI_attributes>\n'
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, afni_xml.encode("utf-8")))
    bucket = tmp_path / "bucket.nii.gz"
    nib.save(img, str(bucket))

    fdr_specs = [
        (1, "fitt", 50.0),  # t-stat sub-brick
        (2, "fift", (3.0, 50.0)),  # F-stat sub-brick
    ]
    _inject_fdr_curves(bucket, fdr_specs)

    reloaded = nib.load(str(bucket))
    afni_ext = next(e for e in reloaded.header.extensions if e.get_code() == 4)
    xml = afni_ext.content.decode("utf-8")
    assert 'atr_name="FDRCURVE_000001"' in xml
    assert 'atr_name="FDRCURVE_000002"' in xml
    # No curve for brick 0 (beta — we didn't ask)
    assert 'atr_name="FDRCURVE_000000"' not in xml


def test_write_glm_bucket_with_add_fdr(tmp_path, monkeypatch):
    """End-to-end check that add_fdr=True populates FDRCURVE attrs.

    Skips the 3drefit step (which requires AFNI binary) by setting
    apply_afni_metadata=False — the FDR injection currently runs inside the
    3drefit branch, so for this test we'll inject directly via the helper.
    """
    # The full write_glm_bucket_as_nifti integration requires AFNI 3drefit and
    # a real GLMResults object. We've already covered _inject_fdr_curves above
    # and the fdr_qvalues math via tests/test_stats_fdr.py. Mark this as a
    # placeholder for a future integration test when ffs_reml is run end-to-end
    # in CI with AFNI installed.
    pytest.skip("End-to-end add_fdr requires AFNI 3drefit; covered by helper tests")


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
