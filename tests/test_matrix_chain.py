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
