"""Block-diagonal, per-run -dsort loading through ``analyze_from_design_matrix``.

The GLS math is covered by ``test_arma_dsort.py`` (which feeds a ready-made
tensor to ``fit_glm_arma11``). These tests cover the *loading* layer added on
top of it:
- a -dsort SET is a list of files, concatenated in time to the input length,
- each run gets its OWN zero-padded column (fit per run, like polynomials),
- multiple -dsort flags stack into independent sets,
- a single concatenated file is equivalent to per-run files.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.analysis import _dsort_block_diagonalize, analyze_from_design_matrix


def test_block_diagonalize_zero_pads_per_run():
    n_vox, n_time = 5, 100
    run_starts = [0, 40]
    reg = torch.arange(n_vox * n_time, dtype=torch.float32).reshape(n_vox, n_time) + 1.0
    block, labels = _dsort_block_diagonalize(reg, 0, run_starts, n_time)

    assert block.shape == (n_vox, 2, n_time)
    assert labels == ["dsort0_run1", "dsort0_run2"]
    # Column 0 carries run 1 (frames 0:40) and is zero afterwards.
    assert torch.all(block[:, 0, 40:] == 0)
    assert torch.all(block[:, 0, :40] == reg[:, :40])
    # Column 1 carries run 2 (frames 40:100) and is zero before.
    assert torch.all(block[:, 1, :40] == 0)
    assert torch.all(block[:, 1, 40:] == reg[:, 40:])
    # The two columns partition the regressor with no overlap or loss.
    assert torch.allclose(block.sum(dim=1), reg)


def test_block_diagonalize_single_run_is_one_column():
    reg = torch.randn(3, 50)
    block, labels = _dsort_block_diagonalize(reg, 2, [0], 50)
    assert block.shape == (3, 1, 50)
    assert labels == ["dsort2"]  # no run suffix for a single-run design
    assert torch.allclose(block[:, 0, :], reg)


def _write_two_run_design(tmp_path, n_time=100, split=50):
    """A minimal, full-rank 2-run design: per-run intercepts + one stim column."""
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


def _save_4d(path, arr, affine=None):
    import nibabel as nib

    affine = np.eye(4, dtype=np.float32) if affine is None else affine
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))


@pytest.fixture
def two_run_case(tmp_path):
    rng = np.random.default_rng(0)
    nx, ny, nz, T, split = 4, 4, 2, 100, 50
    data = rng.standard_normal((nx, ny, nz, T)).astype(np.float32) + 100.0
    data_path = tmp_path / "data.nii.gz"
    _save_4d(data_path, data)
    design = _write_two_run_design(tmp_path, T, split)
    return dict(
        tmp_path=tmp_path,
        rng=rng,
        shape=(nx, ny, nz),
        T=T,
        split=split,
        data_path=data_path,
        design=design,
    )


def _dsort_file(case, name, t0, t1):
    nx, ny, nz = case["shape"]
    arr = case["rng"].standard_normal((nx, ny, nz, t1 - t0)).astype(np.float32)
    path = case["tmp_path"] / name
    _save_4d(path, arr)
    return path


def test_ols_dsort_runs_and_matches_direct_lstsq(two_run_case):
    """OLS + dsort must run (it errored before) and equal a direct per-voxel
    augmented least-squares — proving (a,b)=(0,0) reduces the GLS to exact OLS."""
    c = two_run_case
    f1 = _dsort_file(c, "ols_ds0.nii.gz", 0, c["split"])
    f2 = _dsort_file(c, "ols_ds1.nii.gz", c["split"], c["T"])
    res, _ = analyze_from_design_matrix(
        str(c["data_path"]),
        str(c["design"]),
        method="ols",
        dsort_files=[[str(f1), str(f2)]],
        device=torch.device("cpu"),
        use_double=True,
    )
    assert res.dsort_betas.shape == (int(np.prod(c["shape"])), 2)
    assert res.dsort_labels == ["dsort0_run1", "dsort0_run2"]
    assert res.dof == c["T"] - 3 - 2

    # Direct reference: per-voxel OLS with [design | block-diagonal dsort].
    import nibabel as nib

    X = np.loadtxt(str(c["design"])).astype(np.float64)  # (T, 3)
    T, split = c["T"], c["split"]
    d0 = nib.load(str(f1)).get_fdata().reshape(-1, split)
    d1 = nib.load(str(f2)).get_fdata().reshape(-1, split)
    dfull = np.concatenate([d0, d1], axis=1)  # (nvox, T)
    Y = nib.load(str(c["data_path"])).get_fdata().reshape(-1, T).astype(np.float64)
    nvox = Y.shape[0]
    ref_base = np.zeros((nvox, 3))
    ref_dsort = np.zeros((nvox, 2))
    for v in range(nvox):
        blk = np.zeros((T, 2))
        blk[:split, 0] = dfull[v, :split]
        blk[split:, 1] = dfull[v, split:]
        Xv = np.concatenate([X, blk], axis=1)  # (T, 5)
        coef, *_ = np.linalg.lstsq(Xv, Y[v], rcond=None)
        ref_base[v] = coef[:3]
        ref_dsort[v] = coef[3:]
    assert np.allclose(res.betas.numpy(), ref_base, atol=1e-4, rtol=1e-3)
    assert np.allclose(res.dsort_betas.numpy(), ref_dsort, atol=1e-4, rtol=1e-3)


def test_two_files_one_per_run(two_run_case):
    c = two_run_case
    f1 = _dsort_file(c, "ds_run0.nii.gz", 0, c["split"])
    f2 = _dsort_file(c, "ds_run1.nii.gz", c["split"], c["T"])
    res, info = analyze_from_design_matrix(
        str(c["data_path"]),
        str(c["design"]),
        method="arma11",
        dsort_files=[[str(f1), str(f2)]],
        device=torch.device("cpu"),
        use_double=True,
    )
    assert info["run_starts"] == [0, c["split"]]
    # One set × two runs = two voxel-wise columns, fit per run.
    assert res.dsort_betas.shape == (np.prod(c["shape"]), 2)
    assert res.dsort_labels == ["dsort0_run1", "dsort0_run2"]
    # dof = T - 3 base - 2 dsort.
    assert res.dof == c["T"] - 3 - 2


def test_single_concatenated_file_matches_per_run(two_run_case):
    c = two_run_case
    # One 4D file spanning both runs is equivalent to two per-run files.
    whole = _dsort_file(c, "ds_whole.nii.gz", 0, c["T"])
    res, _ = analyze_from_design_matrix(
        str(c["data_path"]),
        str(c["design"]),
        method="arma11",
        dsort_files=[[str(whole)]],
        device=torch.device("cpu"),
        use_double=True,
    )
    assert res.dsort_betas.shape == (np.prod(c["shape"]), 2)
    assert res.dsort_labels == ["dsort0_run1", "dsort0_run2"]
    assert res.dof == c["T"] - 3 - 2


def test_two_dsort_sets_stack(two_run_case):
    c = two_run_case
    a = _dsort_file(c, "a_whole.nii.gz", 0, c["T"])
    b = _dsort_file(c, "b_whole.nii.gz", 0, c["T"])
    res, _ = analyze_from_design_matrix(
        str(c["data_path"]),
        str(c["design"]),
        method="arma11",
        dsort_files=[[str(a)], [str(b)]],
        device=torch.device("cpu"),
        use_double=True,
    )
    # Two sets × two runs = four columns.
    assert res.dsort_betas.shape == (np.prod(c["shape"]), 4)
    assert res.dsort_labels == [
        "dsort0_run1",
        "dsort0_run2",
        "dsort1_run1",
        "dsort1_run2",
    ]
    assert res.dof == c["T"] - 3 - 4


def test_wrong_length_file_raises(two_run_case):
    c = two_run_case
    short = _dsort_file(c, "short.nii.gz", 0, c["T"] - 10)  # 90 TRs, not 100
    with pytest.raises(ValueError, match="concatenate to"):
        analyze_from_design_matrix(
            str(c["data_path"]),
            str(c["design"]),
            method="arma11",
            dsort_files=[[str(short)]],
            device=torch.device("cpu"),
            use_double=True,
        )
