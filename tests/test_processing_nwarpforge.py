"""Tests for fastfuncstuff.processing.nwarpforge.

Covers dataclasses, composition functions, utility helpers, and the compose_chain
pipeline using small synthetic tensors on CPU.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.nwarpforge import (
    AffineTransform,
    NonlinearWarp,
    _nifti_mm_to_voxels,
    _regrid_to_dxyz,
    apply_composed_warp,
    compose_chain,
    compose_matrix_then_warp,
    compose_warp_then_matrix,
    compose_warp_then_warp,
    compute_cardinal_affine,
    identify_transform_type,
    load_affine_1D,
    parse_nwarp_string,
    prepare_warp_for_grid,
)

DEVICE = torch.device("cpu")
SHAPE = (8, 8, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_affine(voxel_size: float = 2.0, origin: tuple = (-7.0, -7.0, -7.0)) -> np.ndarray:
    """Create a simple diagonal NIfTI affine."""
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = voxel_size
    aff[1, 1] = voxel_size
    aff[2, 2] = voxel_size
    aff[0, 3] = origin[0]
    aff[1, 3] = origin[1]
    aff[2, 3] = origin[2]
    return aff


def _make_zero_warp(shape=SHAPE, affine=None, units="voxels") -> NonlinearWarp:
    """Create a zero-displacement (identity) warp."""
    if affine is None:
        affine = _identity_affine()
    return NonlinearWarp(
        xd=torch.zeros(shape, dtype=torch.float32, device=DEVICE),
        yd=torch.zeros(shape, dtype=torch.float32, device=DEVICE),
        zd=torch.zeros(shape, dtype=torch.float32, device=DEVICE),
        header_info={"affine": affine.copy()},
        units=units,
    )


def _make_small_warp(shape=SHAPE, scale=0.5, affine=None, units="voxels") -> NonlinearWarp:
    """Create a small smooth displacement field."""
    if affine is None:
        affine = _identity_affine()
    torch.manual_seed(42)
    xd = torch.randn(shape, dtype=torch.float32, device=DEVICE) * scale
    yd = torch.randn(shape, dtype=torch.float32, device=DEVICE) * scale
    zd = torch.randn(shape, dtype=torch.float32, device=DEVICE) * scale
    return NonlinearWarp(
        xd=xd, yd=yd, zd=zd,
        header_info={"affine": affine.copy()},
        units=units,
    )


def _make_affine_transform(matrix_4x4: torch.Tensor | None = None) -> AffineTransform:
    """Create AffineTransform from a single 4x4 matrix (identity by default)."""
    if matrix_4x4 is None:
        matrix_4x4 = torch.eye(4, dtype=torch.float32, device=DEVICE)
    return AffineTransform(
        matrices=matrix_4x4.unsqueeze(0),
        base_affine=_identity_affine(),
        source_affine=_identity_affine(),
    )


# ---------------------------------------------------------------------------
# Tests: compute_cardinal_affine
# ---------------------------------------------------------------------------

class TestComputeCardinalAffine:
    def test_diagonal_affine_unchanged(self):
        aff = _identity_affine(voxel_size=3.0, origin=(-10.0, 5.0, 2.0))
        cardinal = compute_cardinal_affine(aff)
        np.testing.assert_allclose(cardinal, aff, atol=1e-10)

    def test_oblique_affine_cardinalized(self):
        """Oblique affine should produce axis-aligned cardinal affine."""
        aff = np.eye(4, dtype=np.float64)
        # Slight rotation: dominant axis still clear
        aff[0, 0] = 2.0
        aff[0, 1] = 0.1
        aff[1, 1] = 2.0
        aff[1, 0] = -0.1
        aff[2, 2] = 2.0
        aff[:3, 3] = [10.0, 20.0, 30.0]

        cardinal = compute_cardinal_affine(aff)
        # Off-diagonal elements should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert cardinal[i, j] == 0.0, f"cardinal[{i},{j}] should be 0"
        # Origin preserved
        np.testing.assert_allclose(cardinal[:3, 3], aff[:3, 3])
        # Voxel sizes preserved (approximately)
        for col in range(3):
            orig_size = np.sqrt(np.sum(aff[:3, col] ** 2))
            card_size = np.sqrt(np.sum(cardinal[:3, col] ** 2))
            np.testing.assert_allclose(card_size, orig_size, atol=1e-10)

    def test_negative_diagonal(self):
        """Negative voxel spacing (LPI convention) preserved."""
        aff = np.eye(4, dtype=np.float64)
        aff[0, 0] = -2.0
        aff[1, 1] = -2.0
        aff[2, 2] = 2.0
        cardinal = compute_cardinal_affine(aff)
        assert cardinal[0, 0] == -2.0
        assert cardinal[1, 1] == -2.0
        assert cardinal[2, 2] == 2.0


# ---------------------------------------------------------------------------
# Tests: AffineTransform dataclass
# ---------------------------------------------------------------------------

class TestAffineTransform:
    def test_single_time_point(self):
        xform = _make_affine_transform()
        assert not xform.is_time_dependent
        mat = xform.at_time(0)
        assert mat.shape == (4, 4)
        torch.testing.assert_close(mat, torch.eye(4))

    def test_multi_time_point(self):
        mats = torch.stack([torch.eye(4) * (i + 1) for i in range(5)])
        xform = AffineTransform(matrices=mats)
        assert xform.is_time_dependent
        # at_time clamps to last index
        torch.testing.assert_close(xform.at_time(4), mats[4])
        torch.testing.assert_close(xform.at_time(100), mats[4])

    def test_at_time_zero(self):
        mats = torch.stack([torch.eye(4), torch.eye(4) * 2])
        xform = AffineTransform(matrices=mats)
        torch.testing.assert_close(xform.at_time(0), mats[0])


# ---------------------------------------------------------------------------
# Tests: NonlinearWarp dataclass
# ---------------------------------------------------------------------------

class TestNonlinearWarp:
    def test_shape_property(self):
        warp = _make_zero_warp(shape=(10, 12, 14))
        assert warp.shape == (10, 12, 14)

    def test_default_units(self):
        warp = _make_zero_warp()
        assert warp.units == "voxels"


# ---------------------------------------------------------------------------
# Tests: parse_nwarp_string
# ---------------------------------------------------------------------------

class TestParseNwarpString:
    def test_single_path(self):
        assert parse_nwarp_string("warp.nii") == ["warp.nii"]

    def test_multiple_paths(self):
        result = parse_nwarp_string("warp.nii mat.1D other.nii.gz")
        assert result == ["warp.nii", "mat.1D", "other.nii.gz"]

    def test_strips_whitespace(self):
        result = parse_nwarp_string("  warp.nii  mat.1D  ")
        assert result == ["warp.nii", "mat.1D"]


# ---------------------------------------------------------------------------
# Tests: identify_transform_type
# ---------------------------------------------------------------------------

class TestIdentifyTransformType:
    def test_1d_file(self):
        assert identify_transform_type("mat.1D") == "affine"
        assert identify_transform_type("MAT.1d") == "affine"

    def test_txt_file(self):
        assert identify_transform_type("transform.txt") == "affine"
        assert identify_transform_type("TRANSFORM.TXT") == "affine"

    def test_nii_file(self):
        assert identify_transform_type("warp.nii") == "nonlinear"

    def test_nii_gz_file(self):
        assert identify_transform_type("warp.nii.gz") == "nonlinear"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot identify"):
            identify_transform_type("mystery.hdf5")


# ---------------------------------------------------------------------------
# Tests: _nifti_mm_to_voxels
# ---------------------------------------------------------------------------

class TestNiftiMmToVoxels:
    def test_identity_affine(self):
        """With identity affine (1mm iso), mm displacements == voxel displacements."""
        aff = np.eye(4, dtype=np.float64)
        xd = torch.tensor([[[1.0, 2.0]]], device=DEVICE)
        yd = torch.tensor([[[3.0, 4.0]]], device=DEVICE)
        zd = torch.tensor([[[5.0, 6.0]]], device=DEVICE)
        vx, vy, vz = _nifti_mm_to_voxels(xd, yd, zd, aff)
        torch.testing.assert_close(vx, xd)
        torch.testing.assert_close(vy, yd)
        torch.testing.assert_close(vz, zd)

    def test_scaled_affine(self):
        """2mm voxels: 4mm displacement = 2 voxels."""
        aff = np.eye(4, dtype=np.float64) * 2.0
        aff[3, 3] = 1.0
        xd = torch.tensor([[[4.0]]], device=DEVICE)
        yd = torch.tensor([[[0.0]]], device=DEVICE)
        zd = torch.tensor([[[0.0]]], device=DEVICE)
        vx, vy, vz = _nifti_mm_to_voxels(xd, yd, zd, aff)
        assert abs(vx.item() - 2.0) < 1e-5

    def test_negative_diagonal(self):
        """Negative voxel spacing inverts displacement direction."""
        aff = np.diag([-2.0, 2.0, 2.0, 1.0]).astype(np.float64)
        xd = torch.tensor([[[4.0]]], device=DEVICE)
        yd = torch.tensor([[[4.0]]], device=DEVICE)
        zd = torch.tensor([[[0.0]]], device=DEVICE)
        vx, vy, vz = _nifti_mm_to_voxels(xd, yd, zd, aff)
        assert abs(vx.item() - (-2.0)) < 1e-5
        assert abs(vy.item() - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: compose_warp_then_matrix
# ---------------------------------------------------------------------------

class TestComposeWarpThenMatrix:
    def test_identity_matrix_preserves_warp(self):
        warp = _make_small_warp()
        identity = torch.eye(4, dtype=torch.float32, device=DEVICE)
        result = compose_warp_then_matrix(warp, identity)
        torch.testing.assert_close(result.xd, warp.xd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.yd, warp.yd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.zd, warp.zd, atol=1e-5, rtol=1e-5)

    def test_zero_warp_with_translation(self):
        """Zero warp + translation matrix = pure translation displacement."""
        warp = _make_zero_warp()
        mat = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat[0, 3] = 3.0  # translate x by 3 voxels
        result = compose_warp_then_matrix(warp, mat)
        # Every voxel should have xd=3
        torch.testing.assert_close(
            result.xd,
            torch.full(SHAPE, 3.0, device=DEVICE),
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            result.yd,
            torch.zeros(SHAPE, device=DEVICE),
            atol=1e-5, rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Tests: compose_matrix_then_warp
# ---------------------------------------------------------------------------

class TestComposeMatrixThenWarp:
    def test_identity_matrix_preserves_warp(self):
        warp = _make_small_warp()
        identity = torch.eye(4, dtype=torch.float32, device=DEVICE)
        result = compose_matrix_then_warp(identity, warp, DEVICE)
        torch.testing.assert_close(result.xd, warp.xd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.yd, warp.yd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.zd, warp.zd, atol=1e-5, rtol=1e-5)

    def test_zero_warp_with_translation(self):
        """Identity warp + matrix with translation: displacement = translation."""
        warp = _make_zero_warp()
        mat = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat[1, 3] = 2.0  # translate y by 2 voxels
        result = compose_matrix_then_warp(mat, warp, DEVICE)
        # The warp is zero so sampling it at translated coords gives zero displacement.
        # Result displacement = matrix @ x - x = translation
        torch.testing.assert_close(
            result.yd,
            torch.full(SHAPE, 2.0, device=DEVICE),
            atol=1e-5, rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Tests: compose_warp_then_warp
# ---------------------------------------------------------------------------

class TestComposeWarpThenWarp:
    def test_identity_warp_composition(self):
        """Composing with a zero (identity) warp preserves the original."""
        warp_a = _make_small_warp()
        warp_b = _make_zero_warp()
        result = compose_warp_then_warp(warp_a, warp_b)
        # B is zero, so sampling B at warped coords gives zero -> result = A
        torch.testing.assert_close(result.xd, warp_a.xd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.yd, warp_a.yd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.zd, warp_a.zd, atol=1e-5, rtol=1e-5)

    def test_two_zero_warps(self):
        """Composing two identity warps gives identity."""
        warp_a = _make_zero_warp()
        warp_b = _make_zero_warp()
        result = compose_warp_then_warp(warp_a, warp_b)
        torch.testing.assert_close(
            result.xd, torch.zeros(SHAPE, device=DEVICE), atol=1e-6, rtol=0,
        )

    def test_composition_not_commutative(self):
        """Warp composition is generally not commutative."""
        warp_a = _make_small_warp(scale=0.5)
        torch.manual_seed(99)
        warp_b = NonlinearWarp(
            xd=torch.randn(SHAPE, device=DEVICE) * 0.5,
            yd=torch.randn(SHAPE, device=DEVICE) * 0.5,
            zd=torch.randn(SHAPE, device=DEVICE) * 0.5,
            header_info={"affine": _identity_affine()},
        )
        ab = compose_warp_then_warp(warp_a, warp_b)
        ba = compose_warp_then_warp(warp_b, warp_a)
        # They should NOT be equal in general
        assert not torch.allclose(ab.xd, ba.xd, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: compose_chain
# ---------------------------------------------------------------------------

class TestComposeChain:
    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="Empty transform chain"):
            compose_chain([], SHAPE, _identity_affine(), DEVICE)

    def test_single_identity_affine(self):
        xform = _make_affine_transform()
        result = compose_chain([xform], SHAPE, _identity_affine(), DEVICE, verb=0)
        assert result.shape == SHAPE
        # Identity affine -> zero displacement
        torch.testing.assert_close(
            result.xd, torch.zeros(SHAPE, device=DEVICE), atol=1e-5, rtol=0,
        )

    def test_single_zero_warp(self):
        """A single zero warp in voxel units goes through prepare_warp_for_grid (needs nifti_mm)."""
        # For compose_chain, warps need to be in nifti_mm so prepare_warp_for_grid works.
        aff = _identity_affine()
        warp = _make_zero_warp(units="nifti_mm", affine=aff)
        result = compose_chain([warp], SHAPE, aff, DEVICE, verb=0)
        assert result.shape == SHAPE
        # Zero mm displacements -> zero voxel displacements
        assert result.xd.abs().max().item() < 1e-5

    def test_two_affines_compose(self):
        """Two translation affines should add their translations."""
        mat1 = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat1[0, 3] = 1.0
        mat2 = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat2[0, 3] = 2.0
        xform1 = _make_affine_transform(mat1)
        xform2 = _make_affine_transform(mat2)
        result = compose_chain([xform1, xform2], SHAPE, _identity_affine(), DEVICE, verb=0)
        # Net translation should be 3.0 in x
        # Check center voxel
        mid = SHAPE[0] // 2
        assert abs(result.xd[mid, mid, mid].item() - 3.0) < 1e-4

    def test_affine_and_inverse_cancel(self):
        """Composing a translation with its inverse should give near-zero displacement."""
        mat = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat[0, 3] = 2.5
        mat[1, 3] = -1.5
        mat_inv = torch.linalg.inv(mat)
        xform = _make_affine_transform(mat)
        xform_inv = _make_affine_transform(mat_inv)
        result = compose_chain([xform, xform_inv], SHAPE, _identity_affine(), DEVICE, verb=0)
        assert result.xd.abs().max().item() < 1e-4
        assert result.yd.abs().max().item() < 1e-4
        assert result.zd.abs().max().item() < 1e-4

    def test_affine_then_warp(self):
        """Chain of [affine, warp] should compose without error."""
        mat = torch.eye(4, dtype=torch.float32, device=DEVICE)
        mat[2, 3] = 1.0
        xform = _make_affine_transform(mat)
        aff = _identity_affine()
        warp = _make_zero_warp(units="nifti_mm", affine=aff)
        result = compose_chain([xform, warp], SHAPE, aff, DEVICE, verb=0)
        assert result.shape == SHAPE


# ---------------------------------------------------------------------------
# Tests: prepare_warp_for_grid
# ---------------------------------------------------------------------------

class TestPrepareWarpForGrid:
    def test_same_grid_no_resample(self):
        """When warp and target grids match, warp values are converted but not resampled."""
        aff = _identity_affine()
        warp = _make_zero_warp(units="nifti_mm", affine=aff)
        result = prepare_warp_for_grid(warp, SHAPE, aff, DEVICE, verb=0)
        assert result.units == "voxels"
        assert result.shape == SHAPE
        assert result.xd.abs().max().item() < 1e-6

    def test_wrong_units_raises(self):
        warp = _make_zero_warp(units="voxels")
        with pytest.raises(AssertionError, match="NIfTI mm"):
            prepare_warp_for_grid(warp, SHAPE, _identity_affine(), DEVICE)

    def test_different_grid_resamples(self):
        """Resampling to a different-sized grid should produce correct output shape."""
        aff = _identity_affine()
        warp = _make_zero_warp(shape=(8, 8, 8), units="nifti_mm", affine=aff)
        target_shape = (10, 10, 10)
        # Different shape -> needs resample
        result = prepare_warp_for_grid(warp, target_shape, aff, DEVICE, verb=0)
        assert result.shape == target_shape
        assert result.units == "voxels"


# ---------------------------------------------------------------------------
# Tests: apply_composed_warp
# ---------------------------------------------------------------------------

class TestApplyComposedWarp:
    def test_identity_warp_preserves_volume(self):
        """Zero displacement should return the source volume (approximately)."""
        torch.manual_seed(123)
        source = torch.rand(SHAPE, dtype=torch.float32, device=DEVICE)
        warp = _make_zero_warp()
        aff = _identity_affine()
        # Use linear interp for speed in tests
        result = apply_composed_warp(source, warp, aff, aff, interp="linear")
        assert result.shape == SHAPE
        # Interior voxels should match well (edges may have boundary effects)
        interior = slice(2, 6)
        torch.testing.assert_close(
            result[interior, interior, interior],
            source[interior, interior, interior],
            atol=1e-4, rtol=1e-4,
        )

    def test_wsinc5_interp_runs(self):
        """wsinc5 interpolation should run without error."""
        torch.manual_seed(7)
        source = torch.rand(SHAPE, dtype=torch.float32, device=DEVICE)
        warp = _make_zero_warp()
        aff = _identity_affine()
        result = apply_composed_warp(source, warp, aff, aff, interp="wsinc5")
        assert result.shape == SHAPE

    def test_translation_warp_shifts_data(self):
        """A constant displacement should shift the sampled data."""
        source = torch.zeros(SHAPE, dtype=torch.float32, device=DEVICE)
        source[4, 4, 4] = 100.0  # point source
        aff = _identity_affine()
        # Displace by 1 voxel in x
        warp = NonlinearWarp(
            xd=torch.full(SHAPE, 1.0, device=DEVICE),
            yd=torch.zeros(SHAPE, device=DEVICE),
            zd=torch.zeros(SHAPE, device=DEVICE),
            header_info={"affine": aff.copy()},
        )
        result = apply_composed_warp(source, warp, aff, aff, interp="linear")
        # The peak should appear shifted: at (4,4,3) result should pick up the
        # source at (4,4,4+1)=5, so result[4,4,3] should sample source[4,4,4]
        # (pull mapping: result[i] = source[i + disp])
        assert result[4, 4, 3].item() > 50.0


# ---------------------------------------------------------------------------
# Tests: _regrid_to_dxyz
# ---------------------------------------------------------------------------

class TestRegridToDxyz:
    def test_same_voxel_size(self):
        """dxyz matching current voxel size should preserve shape."""
        aff = _identity_affine(voxel_size=2.0)
        new_shape, new_aff = _regrid_to_dxyz(SHAPE, aff, dxyz=2.0)
        assert new_shape == SHAPE

    def test_halving_voxel_size(self):
        """Halving voxel size should roughly double grid dimensions."""
        aff = _identity_affine(voxel_size=2.0)
        new_shape, new_aff = _regrid_to_dxyz(SHAPE, aff, dxyz=1.0)
        # 8 voxels * 2mm = 16mm FOV, at 1mm -> 16 voxels
        assert new_shape == (16, 16, 16)
        # New voxel size should be 1.0
        new_voxel_sizes = np.sqrt((new_aff[:3, :3] ** 2).sum(axis=0))
        np.testing.assert_allclose(new_voxel_sizes, 1.0, atol=1e-10)

    def test_doubling_voxel_size(self):
        """Doubling voxel size should roughly halve grid dimensions."""
        aff = _identity_affine(voxel_size=2.0)
        new_shape, new_aff = _regrid_to_dxyz(SHAPE, aff, dxyz=4.0)
        # 8 voxels * 2mm = 16mm, at 4mm -> 4 voxels
        assert new_shape == (4, 4, 4)

    def test_fov_center_preserved(self):
        """FOV center should be preserved after regridding."""
        aff = _identity_affine(voxel_size=2.0, origin=(-10.0, -10.0, -10.0))
        old_center = aff[:3, 3] + aff[:3, :3] @ ((np.array([8, 8, 8]) - 1) * 0.5)

        new_shape, new_aff = _regrid_to_dxyz(SHAPE, aff, dxyz=1.0)
        new_shape_ijk = np.array([new_shape[2], new_shape[1], new_shape[0]])
        new_center = new_aff[:3, 3] + new_aff[:3, :3] @ ((new_shape_ijk - 1) * 0.5)

        np.testing.assert_allclose(new_center, old_center, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: load_affine_1D
# ---------------------------------------------------------------------------

class TestLoadAffine1D:
    def test_load_identity(self):
        """Loading an identity matrix from a .1D file."""
        content = "1 0 0 0 0 1 0 0 0 0 1 0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".1D", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            aff = _identity_affine()
            result = load_affine_1D(path, aff, device=DEVICE)
            assert isinstance(result, AffineTransform)
            assert result.matrices.shape == (1, 4, 4)
            # Identity DICOM matrix -> identity voxel matrix
            torch.testing.assert_close(
                result.matrices[0], torch.eye(4, dtype=torch.float32), atol=1e-5, rtol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_load_multi_row(self):
        """Loading multiple time points."""
        lines = "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".1D", delete=False) as f:
            f.write(lines)
            f.flush()
            path = f.name
        try:
            result = load_affine_1D(path, _identity_affine(), device=DEVICE)
            assert result.matrices.shape[0] == 2
            assert result.is_time_dependent
        finally:
            os.unlink(path)

    def test_bad_column_count_raises(self):
        """Wrong number of columns should raise ValueError."""
        content = "1 0 0 0 0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".1D", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError, match="Expected 12"):
                load_affine_1D(path, _identity_affine(), device=DEVICE)
        finally:
            os.unlink(path)

    def test_debug_mode_runs(self):
        """Debug mode should run without error."""
        content = "1 0 0 0 0 1 0 0 0 0 1 0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".1D", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = load_affine_1D(path, _identity_affine(), device=DEVICE, debug=True)
            assert result.matrices.shape == (1, 4, 4)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: compose_chain with time-dependent affines
# ---------------------------------------------------------------------------

class TestComposeChainTimeDep:
    def test_time_dependent_affine(self):
        """Time-dependent affine should use the correct time index."""
        mats = torch.stack([torch.eye(4), torch.eye(4)], dim=0).float()
        mats[0, 0, 3] = 1.0  # t=0: translate x by 1
        mats[1, 0, 3] = 5.0  # t=1: translate x by 5
        xform = AffineTransform(matrices=mats, base_affine=_identity_affine())

        r0 = compose_chain([xform], SHAPE, _identity_affine(), DEVICE, time_idx=0, verb=0)
        r1 = compose_chain([xform], SHAPE, _identity_affine(), DEVICE, time_idx=1, verb=0)

        mid = SHAPE[0] // 2
        assert abs(r0.xd[mid, mid, mid].item() - 1.0) < 1e-4
        assert abs(r1.xd[mid, mid, mid].item() - 5.0) < 1e-4
