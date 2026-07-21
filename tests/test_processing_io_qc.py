"""Tests for the QC-output primitives in ``processing/io.py``.

Covers the first/last, first/last/difference, and temporal-SNR maps shared by
ffs_moco / ffs_locomoco / ffs_nwarp, plus the path-derivation helper they use.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.processing.io import (
    derive_mean_output_path,
    derive_prefixed_output_path,
    save_first_last,
    save_tsnr,
)


def test_derive_prefixed_output_path_preserves_extension():
    assert derive_prefixed_output_path("d/out.nii.gz", "firstlast") == "d/firstlast_out.nii.gz"
    assert derive_prefixed_output_path("/t/out.nii", "tsnr") == "/t/tsnr_out.nii"
    assert derive_prefixed_output_path("out", "mean") == "mean_out"
    # derive_mean_output_path stays a thin wrapper over the generalized helper.
    assert derive_mean_output_path("out.nii.gz") == "mean_out.nii.gz"


def _load(path):
    # save_image writes (nx, ny, nz[, t]); return with time first to match input.
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return arr


def test_save_first_last_two_volumes(tmp_path):
    data = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    hdr = {"affine": np.eye(4), "header": None}
    out = save_first_last(data, str(tmp_path / "out.nii.gz"), hdr, verb=0)
    assert out == str(tmp_path / "firstlast_out.nii.gz")
    arr = _load(out)  # (nx, ny, nz, 2)
    assert arr.shape[-1] == 2
    # Volume 0 == first input frame, volume 1 == last (compare via transpose back).
    np.testing.assert_allclose(arr[..., 0].transpose(2, 1, 0), data[0].numpy())
    np.testing.assert_allclose(arr[..., 1].transpose(2, 1, 0), data[-1].numpy())


def test_save_first_last_diff_is_signed_last_minus_first(tmp_path):
    data = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    hdr = {"affine": np.eye(4), "header": None}
    out = save_first_last(
        data, str(tmp_path / "out.nii.gz"), hdr, include_diff=True, initial=True, verb=0
    )
    assert out == str(tmp_path / "firstlastdiff_initial_out.nii.gz")
    arr = _load(out)
    assert arr.shape[-1] == 3
    np.testing.assert_allclose(arr[..., 2], arr[..., 1] - arr[..., 0])


def test_save_first_last_skips_non_series():
    # 3-D input (no time axis) and a 1-volume series both no-op to None.
    hdr = {"affine": np.eye(4), "header": None}
    assert save_first_last(torch.zeros(3, 4, 4), "x.nii.gz", hdr, verb=0) is None
    assert save_first_last(torch.zeros(1, 3, 4, 4), "x.nii.gz", hdr, verb=0) is None


def test_save_tsnr_mean_over_std_with_zero_std_guard(tmp_path):
    torch.manual_seed(0)
    data = torch.randn(8, 2, 2, 2) * 3.0 + 50.0
    data[:, 0, 0, 0] = 7.0  # constant in time -> std 0 -> guarded to 0
    hdr = {"affine": np.eye(4), "header": None}
    out = save_tsnr(data, str(tmp_path / "mc.nii.gz"), hdr, verb=0)
    assert out == str(tmp_path / "tsnr_mc.nii.gz")
    arr = _load(out)
    assert np.isfinite(arr).all()
    assert arr[0, 0, 0] == 0.0  # zero-std voxel, not inf/nan

    expected = (data.mean(dim=0) / data.std(dim=0)).numpy()
    expected[0, 0, 0] = 0.0
    np.testing.assert_allclose(arr.transpose(2, 1, 0), expected, rtol=1e-5, atol=1e-5)
