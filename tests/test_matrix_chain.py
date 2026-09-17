"""Tests for affine.load_matrix_chain — composing a stack of .aff12.1D affines."""

import numpy as np
import torch

from fastfuncstuff.processing.affine import (
    dicom_matrix_to_voxel,
    load_matrix_1D,
    load_matrix_chain,
    save_matrix_1D,
)

BASE = np.diag([2.0, 2.0, 2.0, 1.0])
SRC = np.diag([3.0, 3.0, 3.0, 1.0])


def _write_dicom(path, M):
    # save_matrix_1D with no affines writes a raw DICOM 4x4.
    save_matrix_1D(torch.as_tensor(M, dtype=torch.float32), str(path))


def test_single_element_matches_load_matrix_1D(tmp_path):
    M = np.eye(4)
    M[0, 3] = 5.0  # translate in DICOM x
    p = tmp_path / "m.aff12.1D"
    _write_dicom(p, M)
    a = load_matrix_chain([str(p)], base_affine=BASE, source_affine=SRC)
    b = load_matrix_1D(str(p), base_affine=BASE, source_affine=SRC)
    assert torch.allclose(a, b, atol=1e-5)


def test_two_matrix_composition_order(tmp_path):
    # Non-commuting: D1 scales x by 2, D2 translates x by +3 (DICOM mm).
    D1 = np.eye(4)
    D1[0, 0] = 2.0
    D2 = np.eye(4)
    D2[0, 3] = 3.0
    f1, f2 = tmp_path / "d1.aff12.1D", tmp_path / "d2.aff12.1D"
    _write_dicom(f1, D1)
    _write_dicom(f2, D2)

    # Stack [f1, f2] composes base-side→source-side: C_dicom = D2 @ D1.
    got = load_matrix_chain([str(f1), str(f2)], base_affine=BASE, source_affine=SRC)
    expect = dicom_matrix_to_voxel(torch.as_tensor(D2 @ D1, dtype=torch.float32), BASE, SRC)
    assert torch.allclose(got, expect, atol=1e-5)

    # Order matters: reversing the stack gives a different composite.
    other = load_matrix_chain([str(f2), str(f1)], base_affine=BASE, source_affine=SRC)
    assert not torch.allclose(got, other, atol=1e-3)


def test_chain_equals_stepwise_dicom_product(tmp_path):
    rng = np.random.default_rng(0)
    mats = []
    paths = []
    for i in range(3):
        M = np.eye(4)
        M[:3, :3] += 0.1 * rng.standard_normal((3, 3))
        M[:3, 3] = rng.standard_normal(3)
        mats.append(M)
        pth = tmp_path / f"m{i}.aff12.1D"
        _write_dicom(pth, M)
        paths.append(str(pth))
    # C = M_last @ ... @ M_first
    C = np.eye(4)
    for M in mats:
        C = M @ C
    expect = dicom_matrix_to_voxel(torch.as_tensor(C, dtype=torch.float32), BASE, SRC)
    got = load_matrix_chain(paths, base_affine=BASE, source_affine=SRC)
    assert torch.allclose(got, expect, atol=1e-4)


def test_saved_inverse_is_the_transform_with_the_datasets_swapped(tmp_path):
    """-1Dmatrix_save_inv must be usable as-is with -base and -source exchanged.

    The point of solving the cheap way round (hi-res source onto a low-res base)
    is that the inverse is the matrix you actually wanted, so what it has to
    satisfy is: load it with the two datasets swapped and it undoes the forward
    one in voxel space, on their own grids.
    """
    from fastfuncstuff.cli.allineate import _inverse_dicom_matrix
    from fastfuncstuff.processing.affine import identity_params, params_to_matrix

    # Two different grids and centres, as an EPI and an anat would be.
    epi = np.diag([3.0, 3.0, 3.0, 1.0])
    epi[:3, 3] = [-96.0, -90.0, -70.0]
    anat = np.diag([0.9, 0.94, 0.94, 1.0])
    anat[:3, 3] = [-96.5, -90.4, -140.2]

    p = identity_params().clone()
    p[0], p[1], p[3], p[5] = 2.5, -1.75, 4.0, -3.0
    forward = params_to_matrix(p)  # base(epi) voxel -> source(anat) voxel

    fwd_path, inv_path = tmp_path / "f.aff12.1D", tmp_path / "i.aff12.1D"
    save_matrix_1D(forward, str(fwd_path), base_affine=epi, source_affine=anat)
    save_matrix_1D(_inverse_dicom_matrix(forward, epi, anat), str(inv_path))

    # As they would be used: the forward with -base epi, the inverse with -base anat.
    fwd = load_matrix_1D(str(fwd_path), base_affine=epi, source_affine=anat).double()
    inv = load_matrix_1D(str(inv_path), base_affine=anat, source_affine=epi).double()
    torch.testing.assert_close(inv @ fwd, torch.eye(4, dtype=torch.float64), atol=1e-4, rtol=0)

    # And on a concrete voxel, which is what a resample does.
    v = torch.tensor([20.0, 30.0, 25.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(inv @ (fwd @ v), v, atol=1e-4, rtol=0)
