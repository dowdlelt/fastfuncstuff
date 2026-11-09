"""-source_mask / -base_mask: masking for the fit without a 3dcalc step."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from fastfuncstuff.cli.allineate import main


def _save(path, arr, affine):  # arr is (nx, ny, nz), NIfTI order
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), affine), str(path))


def _textured(n=32, seed=1):
    rng = np.random.default_rng(seed)
    vol = gaussian_filter(rng.normal(size=(n, n, n)), 2.0)
    vol = (vol - vol.min()) / np.ptp(vol)
    vol[:4], vol[-4:], vol[:, :4], vol[:, -4:], vol[:, :, :4], vol[:, :, -4:] = 0, 0, 0, 0, 0, 0
    return vol * 100


def test_base_mask_restricts_the_weight_even_from_another_grid(tmp_path):
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    vol = _textured(n=20)
    _save(tmp_path / "base.nii", vol, aff)
    _save(tmp_path / "src.nii", vol, aff)
    # The mask on a 1 mm grid covering the same space: its left half is "in".
    fine = np.zeros((40, 40, 40), np.float32)
    fine[:20] = 1
    _save(tmp_path / "mask.nii", fine, np.diag([1.0, 1.0, 1.0, 1.0]))
    main(["-base", str(tmp_path / "base.nii"), "-source", str(tmp_path / "src.nii"),
          "-prefix", str(tmp_path / "out.nii"), "-base_mask", str(tmp_path / "mask.nii"),
          "-save_weight", str(tmp_path / "w.nii"), "-rigid", "-superfast", "-device", "cpu",
          "-verb", "0"])  # fmt: skip
    w = nib.load(tmp_path / "w.nii").get_fdata()
    assert w[10:].max() == 0  # base x >= 20 mm is outside the mask
    assert w[:10].max() > 0


def test_source_mask_masks_the_fit_but_not_the_output(tmp_path):
    """The mask applies to the fit only (the user's 3dcalc step), never the output."""
    n = 24
    rng = np.random.default_rng(2)
    tex = gaussian_filter(rng.normal(size=(n, n, n)), 1.5)
    tex = (tex - tex.min()) / np.ptp(tex) * 100 + 20
    r = np.linalg.norm(np.indices((n, n, n)) - (n - 1) / 2, axis=0)
    base = tex * (r < 8)
    src = ndshift(base, (1.5, -1.0, 0.0), order=3)
    src[1:6, 1:6, 1:22] = 300.0  # something outside the object, as a VASO edge might be
    for name, arr in (("base", base), ("src", src), ("keep", r < 10)):
        _save(tmp_path / f"{name}.nii", arr, np.eye(4))
    mat = tmp_path / "m.aff12.1D"
    main(["-base", str(tmp_path / "base.nii"), "-source", str(tmp_path / "src.nii"),
          "-source_mask", str(tmp_path / "keep.nii"), "-prefix", str(tmp_path / "out.nii"),
          "-1Dmatrix_save", str(mat), "-rigid", "-cost", "ls", "-nocmass", "-superfast",
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    m = np.loadtxt(mat).reshape(3, 4)
    # DICOM x/y are negated from NIfTI: the 1.5 / -1.0 voxel shift reads -1.5 / +1.0
    np.testing.assert_allclose(m[:, 3], [-1.5, 1.0, 0.0], atol=0.1)
    assert nib.load(tmp_path / "out.nii").get_fdata().max() > 250  # output unmasked
