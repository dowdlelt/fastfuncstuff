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


def test_load_matrix_1D_reads_afni_commented_file(tmp_path):
    # 3dAllineate -1Dmatrix_save writes a '#' header; this crashed the parser.
    vals = [1, 0, 0, 5, 0, 1, 0, -2, 0, 0, 1, 3]
    p = tmp_path / "afni.aff12.1D"
    p.write_text(
        "# 3dAllineate matrices (DICOM-to-DICOM, row-by-row):\n"
        + " ".join(str(v) for v in vals)
        + "\n"
    )
    M = load_matrix_1D(str(p))
    assert torch.equal(M[:3, :4], torch.tensor(vals, dtype=torch.float32).reshape(3, 4))
    # The 3x4 layout (cat_matvec -P style) still reads too.
    q = tmp_path / "grid.aff12.1D"
    q.write_text("1 0 0 5\n0 1 0 -2\n0 0 1 3\n")
    assert torch.equal(load_matrix_1D(str(q)), M)


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


# --- rebasing: the grid a matrix was FITTED on vs the grid it is read against ----
# [[project_oblique_frame_slip]]. An .aff12.1D is in its base's *cardinal* mm frame
# (AFNI's ijk_to_dicom), and cardinal_affine keeps each grid's own origin. So a crop
# of an oblique base is a different mm frame than its parent, and reading the matrix
# against the parent silently shifts everything. Autoproc did exactly that: the
# matrix's base was stage09.anat_autobox, ffs_segment read it against the TPM on the
# uncropped anat, and a 5.04 deg anat cost 6.59 mm.


def _oblique(angle_deg, origin, vox=1.0):
    """ijk->RAS affine rotated about x by angle_deg (so y/z axes are tilted)."""
    a = np.radians(angle_deg)
    r = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    aff = np.eye(4)
    aff[:3, :3] = r * vox
    aff[:3, 3] = origin
    return aff


def _crop_of(parent, offset_xyz):
    """The affine of a crop starting at offset_xyz voxels — same anatomy, same sform."""
    out = parent.copy()
    out[:3, 3] = parent[:3, 3] + parent[:3, :3] @ np.asarray(offset_xyz, dtype=float)
    return out


def _a_matrix(tmp_path, name="m.aff12.1D"):
    M = np.eye(4)
    M[:3, :3] = _oblique(7.0, [0, 0, 0])[:3, :3]
    M[:3, 3] = [1.5, -2.0, 3.25]
    p = tmp_path / name
    _write_dicom(p, M)
    return str(p)


def test_rebase_defaults_are_bit_identical_to_the_two_argument_call(tmp_path):
    """Every existing caller passes neither fitted affine — that path must not move."""
    m = _a_matrix(tmp_path)
    plain = load_matrix_chain(m, base_affine=BASE, source_affine=SRC)
    explicit = load_matrix_chain(
        m, base_affine=BASE, source_affine=SRC, fitted_base_affine=BASE, fitted_source_affine=SRC
    )
    assert torch.equal(plain, load_matrix_chain(m, base_affine=BASE, source_affine=SRC))
    torch.testing.assert_close(plain, explicit, atol=1e-5, rtol=0)


def test_cardinal_base_crop_needs_no_rebase(tmp_path):
    """The discriminator: on a CARDINAL base, cropping really is free, so naive == rebased."""
    m = _a_matrix(tmp_path)
    parent = _oblique(0.0, [-96.0, -110.0, -70.0], vox=0.8)
    crop = _crop_of(parent, (38, 28, 81))

    naive = load_matrix_chain(m, base_affine=parent, source_affine=SRC)
    rebased = load_matrix_chain(m, base_affine=parent, source_affine=SRC, fitted_base_affine=crop)
    torch.testing.assert_close(naive, rebased, atol=1e-4, rtol=0)


def test_oblique_base_crop_slips_and_the_rebase_removes_it(tmp_path):
    """On an oblique base the same crop is a hard shift, of exactly the predicted size."""
    from fastfuncstuff.io.dsetinfo import cardinal_affine

    m = _a_matrix(tmp_path)
    parent = _oblique(5.04, [-96.0, -110.0, -70.0], vox=0.8)
    crop = _crop_of(parent, (38, 28, 81))

    naive = load_matrix_chain(m, base_affine=parent, source_affine=SRC).double()
    rebased = load_matrix_chain(
        m, base_affine=parent, source_affine=SRC, fitted_base_affine=crop
    ).double()
    assert not torch.allclose(naive, rebased, atol=1e-3)

    # The rebase maps a PARENT voxel exactly where reading the matrix on its own
    # grid maps the SAME physical point, which is the whole claim.
    fitted = load_matrix_chain(m, base_affine=crop, source_affine=SRC).double()
    parent_to_crop = torch.from_numpy(np.linalg.inv(crop) @ parent)
    q = torch.tensor([44.0, 61.0, 97.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(rebased @ q, fitted @ (parent_to_crop @ q), atol=1e-6, rtol=0)

    # And the error the naive read makes IS the cardinal-frame slip between the two
    # grids, independently derived: a pure translation, carried through the matrix's
    # rotation (norm-preserving) and into SRC's 3 mm voxels. The real subject's anat
    # was 5.04 deg with a (38, 28, 81) crop and slipped 6.59 mm; a synthetic grid of
    # the same tilt lands near that, so assert the identity, not the number.
    slip = (
        cardinal_affine(parent)
        @ np.linalg.inv(parent)
        @ crop
        @ np.linalg.inv(cardinal_affine(crop))
    )
    slip_mm = float(np.linalg.norm(slip[:3, 3]))
    assert slip_mm > 1.0, "the synthetic geometry stopped exercising the slip"
    err_mm = float((naive @ q - rebased @ q)[:3].norm()) * 3.0  # SRC voxels are 3 mm
    assert abs(err_mm - slip_mm) < 1e-4


def test_oblique_source_crop_slips_too(tmp_path):
    """The other end: ffs_segment -autobox crops the SOURCE out from under the matrix."""
    m = _a_matrix(tmp_path)
    src_parent = _oblique(9.58, [-105.0, 77.7, -63.6], vox=2.0)
    src_crop = _crop_of(src_parent, (5, 7, 3))

    naive = load_matrix_chain(m, base_affine=BASE, source_affine=src_crop).double()
    rebased = load_matrix_chain(
        m, base_affine=BASE, source_affine=src_crop, fitted_source_affine=src_parent
    ).double()
    assert not torch.allclose(naive, rebased, atol=1e-3)

    fitted = load_matrix_chain(m, base_affine=BASE, source_affine=src_parent).double()
    parent_to_crop = torch.from_numpy(np.linalg.inv(src_crop) @ src_parent)
    q = torch.tensor([7.0, 9.0, 11.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(rebased @ q, parent_to_crop @ (fitted @ q), atol=1e-6, rtol=0)
