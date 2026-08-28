"""``ffs_librarian -save_fir_volume``: the per-voxel FIR betas as a 4-D NIfTI.

The R2 map says WHERE the FIR fit worked.  This says WHAT it found there, and
without it there is no way to put a voxel's own measured response next to the
library entry ffs_hrfopt picked for it — deriving a library from these betas
and then discarding them left the tool uncheckable against its own input.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from fastfuncstuff.cli.librarian import write_fir_volume

SHAPE = (4, 5, 6)
N_LAG = 18
LAG = np.arange(0, 36, 2.0)


def _mask_and_betas(seed=0):
    n_vox = int(np.prod(SHAPE))
    mask = np.zeros(n_vox, dtype=bool)
    mask[np.arange(0, n_vox, 2)] = True
    betas = np.random.default_rng(seed).standard_normal((int(mask.sum()), N_LAG))
    return mask, betas


def test_fir_volume_round_trips_the_betas(tmp_path):
    mask, betas = _mask_and_betas()
    path = tmp_path / "sub01_fir_betas.nii.gz"
    write_fir_volume(betas, mask, SHAPE, path, np.eye(4), LAG, 2.0)

    img = nib.load(path)
    assert img.shape == (*SHAPE, N_LAG)
    got = img.get_fdata().reshape(-1, N_LAG)
    np.testing.assert_allclose(got[mask], betas, atol=1e-5)
    # Everything outside the mask must be zero, not left-over memory.
    assert np.allclose(got[~mask], 0.0)


def test_fourth_dimension_is_lag_not_time(tmp_path):
    # The 4th axis is FIR lag, so the header TR is the LAG SPACING -- a viewer's
    # time axis then reads directly in seconds-since-event.  Writing the run's
    # TR here would be right only by coincidence (when the FIR is TR-locked).
    mask, betas = _mask_and_betas()
    path = tmp_path / "sub01_fir_betas.nii.gz"
    write_fir_volume(betas, mask, SHAPE, path, np.eye(4), np.arange(0, 18, 1.0), 2.0)
    assert nib.load(path).header.get_zooms()[3] == 1.0


def test_single_lag_falls_back_to_the_run_tr(tmp_path):
    mask, _ = _mask_and_betas()
    betas = np.zeros((int(mask.sum()), 1))
    path = tmp_path / "sub01_fir_betas.nii.gz"
    write_fir_volume(betas, mask, SHAPE, path, np.eye(4), np.array([0.0]), 2.5)
    img = nib.load(path)
    assert img.shape == (*SHAPE, 1)
    assert img.header.get_zooms()[3] == 2.5


def test_affine_is_preserved(tmp_path):
    mask, betas = _mask_and_betas()
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    path = tmp_path / "sub01_fir_betas.nii.gz"
    write_fir_volume(betas, mask, SHAPE, path, affine, LAG, 2.0)
    np.testing.assert_allclose(nib.load(path).affine, affine, atol=1e-6)


def test_flag_is_registered_and_off_by_default():
    from fastfuncstuff.cli.librarian import create_parser

    args = create_parser().parse_args(["-prefix", "x", "-input", "a.nii"])
    assert args.save_fir_volume is False
    on = create_parser().parse_args(["-prefix", "x", "-input", "a.nii", "-save-fir-volume"])
    assert on.save_fir_volume is True
