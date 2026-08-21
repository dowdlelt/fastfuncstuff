"""Grid geometry must reproduce 3dresample's header arithmetic exactly.

Every expected number below was read off ``3dresample`` (AFNI 25.3.03) run on the
synthetic RPI volume built in :func:`_grid` — ``3dinfo -orient -ad3 -o3 -n4`` of
the output. Getting a voxel count or an origin wrong by half a voxel silently
misaligns every dataset resampled afterwards, and nothing downstream complains.

AFNI origins are quoted in its DICOM/LPS frame (x toward L, y toward P, z toward
S), which is what ``3dinfo -o3`` prints.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.cli.util_resample import parse_args as parse_resample_args
from fastfuncstuff.processing import grid as grid_module
from fastfuncstuff.processing.grid import (
    afni_orient_code,
    as_index_map,
    crop_affine,
    grid_extent_rai,
    reorient_grid,
    resample_grid,
    resample_to_grid,
    take_index_map,
    validate_orient,
    voxel_map,
)

# nibabel axcodes ('L','A','S') -> AFNI "RPI"; 2.5 x 2.0 x 3.0 mm, 61 x 73 x 55.
AFFINE = np.array(
    [
        [-2.5, 0.0, 0.0, 75.0],
        [0.0, 2.0, 0.0, -80.0],
        [0.0, 0.0, 3.0, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SHAPE = (55, 73, 61)  # (nz, ny, nx)

_LPS = np.array([-1.0, -1.0, 1.0])


def test_resample_defaults_to_high_fidelity_and_labels_remain_explicit():
    args = parse_resample_args(["-input", "in.nii", "-prefix", "out.nii"])
    assert args.rmode == "wsinc5"


def test_single_volume_high_order_resample_stays_3d(monkeypatch):
    seen = []

    def record(source, sx, sy, sz, interp):
        seen.append((source.shape, interp))
        return torch.zeros_like(sx)

    monkeypatch.setattr(grid_module, "_separable_resample_3d", record)
    source = torch.ones(1, 4, 5, 6)
    z, y, x = torch.meshgrid(
        torch.arange(4, dtype=torch.float32),
        torch.arange(5, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    out = grid_module._sample_batch(source, x + 0.2, y, z, "wsinc5")

    assert seen == [(source.shape[1:], "wsinc5")]
    assert out.shape == source.shape


def _grid(shape, affine):
    """(nx, ny, nz) and the AFNI-frame origin, in the order 3dinfo prints them."""
    return (shape[2], shape[1], shape[0]), affine[:3, 3] * _LPS


def test_orient_code_is_the_opposite_of_the_axcode():
    # nibabel says the axes point ('L','A','S'); AFNI names where they start.
    assert afni_orient_code(AFFINE) == "RPI"


def test_validate_orient_rejects_a_repeated_axis():
    assert validate_orient("asl") == "ASL"
    with pytest.raises(ValueError):
        validate_orient("RRI")
    with pytest.raises(ValueError):
        validate_orient("RP")


def test_fov_bound_matches_3dresample():
    # 3dresample -bound_type FOV -dxyz 1.7 3.1 2.2
    shape, affine = resample_grid(SHAPE, AFFINE, (1.7, 3.1, 2.2), "FOV")
    dims, org = _grid(shape, affine)
    assert dims == (90, 47, 75)
    assert np.allclose(org, [-75.65, 79.3, -60.4], atol=1e-4)


def test_slab_bound_matches_3dresample():
    # 3dresample -bound_type SLAB -dxyz 1.7 3.1 2.2 — one fewer voxel on x than
    # FOV, because SLAB keeps the outer centres instead of the field of view.
    shape, affine = resample_grid(SHAPE, AFFINE, (1.7, 3.1, 2.2), "SLAB")
    dims, org = _grid(shape, affine)
    assert dims == (89, 47, 75)
    assert np.allclose(org, [-74.8, 79.3, -60.4], atol=1e-4)


def test_reorient_matches_3dresample():
    # 3dresample -orient ASL
    shape, affine = reorient_grid(SHAPE, AFFINE, "ASL")
    dims, org = _grid(shape, affine)
    assert afni_orient_code(affine) == "ASL"
    assert dims == (73, 55, 61)
    # -o3 prints per index axis: (A/P axis, S/I axis, L/R axis)
    assert np.allclose([org[1], org[2], org[0]], [-64.0, 102.0, 75.0], atol=1e-4)
    assert np.allclose(np.linalg.norm(affine[:3, :3], axis=0), [2.0, 3.0, 2.5])


def test_reorient_then_dxyz_matches_3dresample():
    # 3dresample -orient ASL -dxyz 1.7 3.1 2.2. dxyz applies to the axes as they
    # stand *after* reorienting, which is why the counts are not a permutation
    # of the dxyz-only result.
    shape, affine = reorient_grid(SHAPE, AFFINE, "ASL")
    shape, affine = resample_grid(shape, affine, (1.7, 3.1, 2.2), "FOV")
    dims, org = _grid(shape, affine)
    assert dims == (86, 53, 69)
    assert np.allclose([org[1], org[2], org[0]], [-64.25, 101.6, 74.8], atol=1e-4)


def test_cent_preserves_voxel_centres_for_an_integer_factor():
    # The point of CENT: downsampling by 2 must land on original voxel centres,
    # so the map back to source indices is exactly (0, 2, 4, ...) with no offset.
    vox = np.linalg.norm(AFFINE[:3, :3], axis=0)
    shape, affine = resample_grid(SHAPE, AFFINE, vox * 2, "CENT")
    m = voxel_map(AFFINE, affine)
    assert np.allclose(m[:3, 3], 0.0, atol=1e-9)
    assert np.allclose(np.diag(m[:3, :3]), 2.0)


def test_cent_and_cent_orig_trim_opposite_ends():
    # Same voxel count, but CENT truncates toward R/A/I while CENT_ORIG
    # truncates toward the origin — so they differ only where a half voxel of
    # slack has to go. Here that is the y axis (negative AFNI delta).
    cent_shape, cent = resample_grid(SHAPE, AFFINE, (3.3, 2.6, 4.1), "CENT")
    orig_shape, orig = resample_grid(SHAPE, AFFINE, (3.3, 2.6, 4.1), "CENT_ORIG")
    assert cent_shape == orig_shape
    assert np.allclose(voxel_map(AFFINE, cent)[:3, 3], [0.0, 0.5, 0.0], atol=1e-9)
    assert np.allclose(voxel_map(AFFINE, orig)[:3, 3], [0.0, 0.0, 0.0], atol=1e-9)


def test_extent_matches_3dautobox():
    # 3dAutobox -extent on the cropped default box of this volume.
    cropped_affine = crop_affine(AFFINE, (18, 11, 14))
    extent = grid_extent_rai((25, 39, 31), cropped_affine)
    assert np.allclose(extent, [-30.0, 45.0, -18.0, 58.0, -18.0, 54.0], atol=1e-4)


# --- exact (interpolation-free) path -------------------------------------


def test_reorient_is_detected_as_an_exact_index_map():
    _, affine = reorient_grid(SHAPE, AFFINE, "ASL")
    assert as_index_map(voxel_map(AFFINE, affine)) is not None


def test_dxyz_resample_is_not_an_exact_index_map():
    _, affine = resample_grid(SHAPE, AFFINE, (1.7, 3.1, 2.2), "FOV")
    assert as_index_map(voxel_map(AFFINE, affine)) is None


def test_reorient_round_trip_is_bit_exact():
    # Two relabellings and back must return the original array untouched — no
    # interpolation may sneak into a pure permutation.
    data = torch.rand(*SHAPE)
    shape_a, aff_a = reorient_grid(SHAPE, AFFINE, "ASL")
    out = resample_to_grid(data, AFFINE, shape_a, aff_a, interp="cubic")
    back = resample_to_grid(out, aff_a, SHAPE, AFFINE, interp="cubic")
    assert torch.equal(back, data)


def test_identical_grid_is_returned_untouched():
    data = torch.rand(*SHAPE)
    out = resample_to_grid(data, AFFINE, SHAPE, AFFINE, interp="linear")
    assert torch.equal(out, data)


def test_take_index_map_zero_pads_outside_the_source():
    data = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # (nz,ny,nx)
    ident, signs = np.arange(3), np.ones(3, dtype=int)
    out = take_index_map(data, ident, signs, np.array([-1, 0, 0]), (2, 3, 6))
    assert out.shape == (2, 3, 6)
    assert torch.equal(out[..., 0], torch.zeros(2, 3))  # padded column
    assert torch.equal(out[..., 1:5], data)
    assert torch.equal(out[..., 5], torch.zeros(2, 3))


def test_take_index_map_handles_4d():
    data = torch.rand(7, 2, 3, 4)
    ident, signs = np.arange(3), np.ones(3, dtype=int)
    out = take_index_map(data, ident, signs, np.array([1, 0, 0]), (2, 3, 2))
    assert out.shape == (7, 2, 3, 2)
    assert torch.equal(out, data[..., 1:3])


# --- interpolation --------------------------------------------------------


@pytest.mark.parametrize("interp", ["nearest", "linear", "blocky", "cubic"])
def test_interpolation_reproduces_the_source_at_grid_nodes(interp):
    # Any kernel evaluated exactly on source voxel centres must return them.
    # An off-by-half in the coordinate map shows up here and nowhere else.
    data = torch.rand(*SHAPE)
    vox = np.linalg.norm(AFFINE[:3, :3], axis=0)
    shape, affine = resample_grid(SHAPE, AFFINE, vox * 2, "CENT")
    out = resample_to_grid(data, AFFINE, shape, affine, interp=interp)
    assert torch.allclose(out, data[::2, ::2, ::2][: shape[0], : shape[1], : shape[2]], atol=1e-5)


def test_blocky_differs_from_linear_off_node():
    # Blocky is linear evaluated at a warped fraction; if the prewarp were a
    # no-op this would silently be plain linear interpolation.
    data = torch.rand(*SHAPE)
    shape, affine = resample_grid(SHAPE, AFFINE, (1.7, 3.1, 2.2), "FOV")
    lin = resample_to_grid(data, AFFINE, shape, affine, interp="linear")
    blk = resample_to_grid(data, AFFINE, shape, affine, interp="blocky")
    assert not torch.allclose(lin, blk, atol=1e-3)


def test_samples_outside_the_source_are_zero():
    data = torch.ones(*SHAPE)
    # Shift the grid far enough that most of it hangs off the source.
    far = crop_affine(AFFINE, (-200, 0, 0))
    out = resample_to_grid(data, AFFINE, SHAPE, far, interp="linear")
    assert float(out[..., :100].abs().max()) == 0.0


def test_4d_resample_preserves_the_time_axis():
    data = torch.rand(9, *SHAPE)
    shape, affine = resample_grid(SHAPE, AFFINE, (1.7, 3.1, 2.2), "FOV")
    out = resample_to_grid(data, AFFINE, shape, affine, interp="linear")
    assert out.shape == (9, *shape)
    # Frames are independent: scaling one frame scales only that output frame.
    single = resample_to_grid(data[3], AFFINE, shape, affine, interp="linear")
    assert torch.allclose(out[3], single)
