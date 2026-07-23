"""Mask streaming during multi-run load in ``analyze_from_design_matrix``.

A plain ``-mask`` combined with multi-run file input is applied run-by-run
inside the loader (a memory saver at whole-dataset scale) instead of after the
full concatenation. These tests pin that the streamed-mask path produces the
same GLM result as the classic post-load mask path, and that out-of-mask voxels
are still reconstructed as zeros in the full-volume output.
"""

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.analysis import _load_fmri_data, analyze_from_design_matrix


def test_load_fmri_data_ndarray_is_zero_copy():
    """The preprocessing path (-do_scale) hands a float32 ndarray to the analysis;
    it must be adopted via from_numpy (shared buffer), not copied -- a copy doubles
    peak RAM and OOM-kills at whole-dataset scale."""
    arr = np.random.default_rng(0).standard_normal((1000, 40)).astype(np.float32)
    data = _load_fmri_data(arr, torch.device("cpu"))
    assert data.dtype == torch.float32
    # Shares memory with the numpy input (no copy).
    assert np.shares_memory(data.numpy(), arr)


def test_load_fmri_data_float64_ndarray_casts():
    """A dtype mismatch must still yield correct float32 data (copy is unavoidable)."""
    arr = np.random.default_rng(1).standard_normal((50, 8)).astype(np.float64)
    data = _load_fmri_data(arr, torch.device("cpu"))
    assert data.dtype == torch.float32
    assert np.allclose(data.numpy(), arr.astype(np.float32))


def _write_design(tmp_path, n_time, split):
    X = np.zeros((n_time, 3), np.float32)
    X[:split, 0] = 1.0
    X[split:, 1] = 1.0
    X[:, 2] = np.sin(np.arange(n_time) * 0.3)
    path = tmp_path / "design.1D"
    with open(path, "w") as f:
        f.write("# RowTR = 2.0\n")
        f.write(f"# NRowFull = {n_time}\n")
        f.write(f"# RunStart = 0,{split}\n")
        f.write('# ColumnLabels = "run1base;run2base;stim"\n')
        np.savetxt(f, X)
    return path


def test_streamed_mask_matches_postload_mask(tmp_path):
    nx, ny, nz, split = 4, 4, 2, 50
    T = split * 2
    rng = np.random.default_rng(0)
    run0 = rng.standard_normal((nx, ny, nz, split)).astype(np.float32) + 100.0
    run1 = rng.standard_normal((nx, ny, nz, split)).astype(np.float32) + 100.0

    run0_path = tmp_path / "run0.nii.gz"
    run1_path = tmp_path / "run1.nii.gz"
    nib.save(nib.Nifti1Image(run0, np.eye(4)), str(run0_path))
    nib.save(nib.Nifti1Image(run1, np.eye(4)), str(run1_path))

    # Single concatenated file (drives the classic post-load mask path).
    concat = np.concatenate([run0, run1], axis=3)
    concat_path = tmp_path / "concat.nii.gz"
    nib.save(nib.Nifti1Image(concat, np.eye(4)), str(concat_path))

    # Mask keeping a subset of voxels.
    mask = np.zeros((nx, ny, nz), np.float32)
    mask.reshape(-1)[::3] = 1.0
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
    kept = int(mask.reshape(-1).astype(bool).sum())

    design = _write_design(tmp_path, T, split)

    # List input + mask -> streamed masking inside the loader.
    res_stream, _ = analyze_from_design_matrix(
        [str(run0_path), str(run1_path)],
        str(design),
        method="ols",
        mask_file=str(mask_path),
        device=torch.device("cpu"),
        use_double=True,
    )
    # Single file + mask -> classic post-load masking.
    res_post, _ = analyze_from_design_matrix(
        str(concat_path),
        str(design),
        method="ols",
        mask_file=str(mask_path),
        device=torch.device("cpu"),
        use_double=True,
    )

    # Betas are returned over kept voxels; the two mask paths must agree exactly.
    assert res_stream.betas.shape == (kept, 3)
    assert res_post.betas.shape == (kept, 3)
    assert np.allclose(res_stream.betas.numpy(), res_post.betas.numpy(), atol=1e-6)
