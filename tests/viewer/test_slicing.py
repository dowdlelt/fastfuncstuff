"""Resampling a layer into the display grid.

Orientation bugs here are the classic silent ones: a transposed or mirrored
slice still looks like a brain. These tests pin the mapping with asymmetric
volumes and known voxel values so a flip cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.slicing import (
    display_to_layer,
    extract_plane,
    plane_indices,
    plane_layout,
    plane_shape,
    ras_axes,
    sample_volume,
    voxel_value,
)
from fastfuncstuff.viewer.state import DisplayGrid, Plane

_RAS = np.diag([2.0, 3.0, 4.0, 1.0])
_LPI = np.diag([-2.0, -3.0, 4.0, 1.0])
#: Stored (A, S, R) rather than (R, A, S) -- a legal grid a viewer must handle.
_SWAPPED = np.array([[0, 0, 2, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 1]], dtype=float)


def _grid(shape=(6, 8, 10), step=(2.0, 3.0, 4.0), origin=(-5.0, -9.0, -20.0)):
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = step
    a[:3, 3] = origin
    return DisplayGrid(shape=shape, affine=a)


def _ramp(shape=(6, 8, 10)) -> torch.Tensor:
    """A volume whose value encodes its own index, so a flip is detectable."""
    nx, ny, nz = shape
    i = torch.arange(nx).view(nx, 1, 1)
    j = torch.arange(ny).view(1, ny, 1)
    k = torch.arange(nz).view(1, 1, nz)
    return (i * 10_000 + j * 100 + k).to(torch.float32).expand(nx, ny, nz).contiguous()


# ---------------------------------------------------------------------------
# plane geometry
# ---------------------------------------------------------------------------


def test_each_plane_is_normal_to_its_anatomical_axis():
    """A plane is defined by the anatomy it cuts, not by an array index."""
    grid = _grid()
    axes = ras_axes(grid.affine)
    assert plane_layout(grid.affine, Plane.AXIAL).fixed == axes["S"][0]
    assert plane_layout(grid.affine, Plane.CORONAL).fixed == axes["A"][0]
    assert plane_layout(grid.affine, Plane.SAGITTAL).fixed == axes["R"][0]


def test_coronal_puts_superior_at_the_top():
    for affine in (_RAS, _LPI, _SWAPPED):
        layout = plane_layout(affine, Plane.CORONAL)
        assert layout.labels[0] == "S", "superior must be up"
        assert layout.labels[2] == "I"


def test_axial_and_coronal_put_the_subjects_left_on_the_left():
    """Neurological convention: the viewer's left is the subject's left."""
    for affine in (_RAS, _LPI, _SWAPPED):
        for plane in (Plane.AXIAL, Plane.CORONAL):
            assert plane_layout(affine, plane).labels[3] == "L"
            assert plane_layout(affine, plane).labels[1] == "R"


def test_axial_puts_anterior_at_the_top():
    for affine in (_RAS, _LPI, _SWAPPED):
        assert plane_layout(affine, Plane.AXIAL).labels[0] == "A"


def test_sagittal_faces_left_with_superior_up():
    for affine in (_RAS, _LPI, _SWAPPED):
        layout = plane_layout(affine, Plane.SAGITTAL)
        assert layout.labels == ("S", "P", "I", "A")


def test_orientation_holds_when_the_stored_axis_order_changes():
    """The same anatomy must land in the same screen position either way."""
    ras = plane_layout(_RAS, Plane.CORONAL)
    lpi = plane_layout(_LPI, Plane.CORONAL)
    assert ras.labels == lpi.labels
    # LPI runs right-to-left in storage, so the column must be flipped to
    # compensate; RAS does not.
    assert lpi.col_flip != ras.col_flip


def test_ras_axes_reads_the_affine():
    assert ras_axes(_RAS) == {"R": (0, 1), "A": (1, 1), "S": (2, 1)}
    assert ras_axes(_LPI) == {"R": (0, -1), "A": (1, -1), "S": (2, 1)}


def test_ras_axes_handles_a_permuted_grid():
    axes = ras_axes(_SWAPPED)
    assert {a for a, _ in axes.values()} == {0, 1, 2}, "each axis used once"


def test_image_and_ijk_round_trip():
    """The flip arithmetic lives in one place precisely so this holds."""
    grid = _grid((6, 8, 10))
    for affine in (_RAS, _LPI, _SWAPPED):
        for plane in Plane:
            layout = plane_layout(affine, plane)
            ijk = (2, 3, 4)
            row, col = layout.to_image(ijk, grid.shape)
            assert layout.to_ijk(row, col, ijk, grid.shape) == ijk


def test_plane_shape_follows_the_layout():
    grid = _grid((6, 8, 10))
    for plane in Plane:
        layout = plane_layout(grid.affine, plane)
        assert plane_shape(grid, plane) == (grid.shape[layout.row], grid.shape[layout.col])


def test_plane_indices_hold_the_fixed_axis_constant():
    grid = _grid()
    layout = plane_layout(grid.affine, Plane.AXIAL)
    idx = plane_indices(grid, Plane.AXIAL, 3)
    assert torch.all(idx[..., layout.fixed] == 3)


def test_plane_indices_run_backwards_along_a_flipped_axis():
    """A flip is how superior ends up at the top rather than the bottom."""
    grid = _grid((6, 8, 10))
    layout = plane_layout(grid.affine, Plane.CORONAL)
    assert layout.row_flip, "RAS coronal must flip rows to put S up"
    idx = plane_indices(grid, Plane.CORONAL, 0)
    top = float(idx[0, 0, layout.row])
    bottom = float(idx[-1, 0, layout.row])
    assert top > bottom, "row 0 should be the superior end"


# ---------------------------------------------------------------------------
# the affine composition
# ---------------------------------------------------------------------------


def test_identical_affines_are_the_identity_mapping():
    grid = _grid()
    pts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = display_to_layer(pts, grid.affine, grid.affine)
    assert torch.allclose(out, pts, atol=1e-5)


def test_a_shifted_layer_maps_by_the_offset():
    grid = _grid(step=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    layer = np.eye(4)
    layer[:3, 3] = [2.0, 0.0, 0.0]  # layer origin 2 mm along x
    out = display_to_layer(torch.tensor([[2.0, 0.0, 0.0]]), grid.affine, layer)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-5)


def test_a_coarser_layer_maps_by_the_scale():
    grid = _grid(step=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    layer = np.diag([2.0, 2.0, 2.0, 1.0])  # 2 mm voxels
    out = display_to_layer(torch.tensor([[4.0, 6.0, 8.0]]), grid.affine, layer)
    assert torch.allclose(out, torch.tensor([[2.0, 3.0, 4.0]]), atol=1e-5)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_sampling_on_exact_indices_returns_those_voxels():
    """The orientation test: an index-encoding ramp must come back unpermuted."""
    vol = _ramp((6, 8, 10))
    pts = torch.tensor([[0.0, 0.0, 0.0], [5.0, 7.0, 9.0], [2.0, 3.0, 4.0]])
    got = sample_volume(vol, pts)
    assert torch.allclose(got, torch.tensor([0.0, 50709.0, 20304.0]), atol=1e-2)


def test_sampling_interpolates_between_voxels():
    vol = torch.zeros(4, 4, 4)
    vol[1, 1, 1] = 0.0
    vol[2, 1, 1] = 10.0
    got = sample_volume(vol, torch.tensor([[1.5, 1.0, 1.0]]))
    assert abs(float(got[0]) - 5.0) < 1e-4


def test_samples_outside_the_volume_are_zero_not_edge_clamped():
    """A smaller field of view must show its real extent, not smear its border."""
    vol = torch.ones(4, 4, 4)
    got = sample_volume(vol, torch.tensor([[-5.0, 0.0, 0.0], [99.0, 0.0, 0.0]]))
    assert torch.allclose(got, torch.zeros(2), atol=1e-6)


def test_sample_rejects_non_3d_volumes():
    with pytest.raises(ValueError):
        sample_volume(torch.zeros(4, 4), torch.tensor([[0.0, 0.0, 0.0]]))


def test_nearest_mode_returns_an_actual_voxel_value():
    vol = _ramp((4, 4, 4))
    got = sample_volume(vol, torch.tensor([[1.4, 2.4, 3.4]]), mode="nearest")
    assert float(got[0]) == 10203.0


# ---------------------------------------------------------------------------
# plane extraction
# ---------------------------------------------------------------------------


def test_extracted_plane_has_the_planes_shape():
    grid = _grid((6, 8, 10))
    vol = _ramp((6, 8, 10))
    for plane in Plane:
        got = extract_plane(vol, grid, grid.affine, plane, 1)
        assert got.shape == plane_shape(grid, plane), plane


def test_each_pixel_holds_the_voxel_the_layout_says_it_should():
    """The orientation test: an index-encoding ramp must land where claimed."""
    grid = _grid((6, 8, 10))
    vol = _ramp((6, 8, 10))
    for plane in Plane:
        layout = plane_layout(grid.affine, plane)
        got = extract_plane(vol, grid, grid.affine, plane, 2)
        h, w = plane_shape(grid, plane)
        for row, col in ((0, 0), (h - 1, 0), (0, w - 1), (h // 2, w // 2)):
            ijk = layout.to_ijk(row, col, (2, 2, 2), grid.shape)
            expected = ijk[0] * 10_000 + ijk[1] * 100 + ijk[2]
            assert abs(float(got[row, col]) - expected) < 1e-2, (plane, row, col)


def test_a_flipped_grid_shows_the_same_anatomy_the_same_way_up():
    """LPI and RAS storage of the same brain must look identical on screen."""
    shape = (6, 8, 10)
    ras_grid = DisplayGrid(shape=shape, affine=_RAS)
    lpi_grid = DisplayGrid(shape=shape, affine=_LPI)
    vol = _ramp(shape)
    # Mirror the volume along x so it represents the same anatomy under LPI.
    flipped = torch.flip(vol, dims=[0])
    a = extract_plane(vol, ras_grid, _RAS, Plane.CORONAL, 3)
    b = extract_plane(flipped, lpi_grid, _LPI, Plane.CORONAL, 3)
    assert torch.allclose(a, b, atol=1e-3)


def test_a_layer_on_a_coarser_grid_still_lands_in_the_right_place():
    """Resampling across grids is the whole point of dropping view spaces."""
    grid = _grid((8, 8, 8), step=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    layer_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    vol = torch.zeros(4, 4, 4)
    vol[1, 1, 1] = 100.0  # sits at 2,2,2 mm -> display voxel (2,2,2)
    got = extract_plane(vol, grid, layer_affine, Plane.AXIAL, 2)
    layout = plane_layout(grid.affine, Plane.AXIAL)
    row, col = layout.to_image((2, 2, 2), grid.shape)
    assert abs(float(got[row, col]) - 100.0) < 1e-3
    assert float(got[0, 0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


def test_voxel_value_reports_the_underlying_voxel():
    grid = _grid((6, 8, 10))
    vol = _ramp((6, 8, 10))
    assert voxel_value(vol, grid, grid.affine, (2, 3, 4)) == 20304.0


def test_voxel_value_is_none_outside_the_layer():
    grid = _grid((6, 8, 10))
    vol = _ramp((4, 4, 4))
    assert voxel_value(vol, grid, grid.affine, (5, 7, 9)) is None
