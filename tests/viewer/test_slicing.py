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
    plane_axes,
    plane_indices,
    plane_shape,
    sample_volume,
    voxel_value,
)
from fastfuncstuff.viewer.state import DisplayGrid, Plane


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


def test_each_plane_holds_a_different_axis_fixed():
    assert plane_axes(Plane.SAGITTAL)[0] == 0
    assert plane_axes(Plane.CORONAL)[0] == 1
    assert plane_axes(Plane.AXIAL)[0] == 2


def test_plane_shape_matches_the_spanned_axes():
    grid = _grid((6, 8, 10))
    assert plane_shape(grid, Plane.AXIAL) == (6, 8)
    assert plane_shape(grid, Plane.CORONAL) == (6, 10)
    assert plane_shape(grid, Plane.SAGITTAL) == (8, 10)


def test_plane_indices_hold_the_fixed_axis_constant():
    grid = _grid()
    idx = plane_indices(grid, Plane.AXIAL, 3)
    assert torch.all(idx[..., 2] == 3)
    assert idx.shape == (6, 8, 3)


def test_plane_indices_span_the_other_two_axes():
    grid = _grid()
    idx = plane_indices(grid, Plane.AXIAL, 0)
    assert float(idx[0, 0, 0]) == 0.0
    assert float(idx[5, 0, 0]) == 5.0
    assert float(idx[0, 7, 1]) == 7.0


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


def test_axial_plane_is_not_transposed():
    """Rows must follow display i and columns display j, not the reverse."""
    grid = _grid((6, 8, 10))
    vol = _ramp((6, 8, 10))
    got = extract_plane(vol, grid, grid.affine, Plane.AXIAL, 5)
    assert abs(float(got[0, 0]) - 5.0) < 1e-2  # i=0, j=0, k=5
    assert abs(float(got[3, 0]) - 30005.0) < 1e-2  # i=3, j=0, k=5
    assert abs(float(got[0, 2]) - 205.0) < 1e-2  # i=0, j=2, k=5


def test_sagittal_plane_orientation():
    grid = _grid((6, 8, 10))
    vol = _ramp((6, 8, 10))
    got = extract_plane(vol, grid, grid.affine, Plane.SAGITTAL, 2)
    assert abs(float(got[0, 0]) - 20000.0) < 1e-2  # i=2, j=0, k=0
    assert abs(float(got[4, 6]) - 20406.0) < 1e-2  # i=2, j=4, k=6


def test_a_layer_on_a_coarser_grid_still_lands_in_the_right_place():
    """Resampling across grids is the whole point of dropping view spaces."""
    grid = _grid((8, 8, 8), step=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    layer_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    vol = torch.zeros(4, 4, 4)
    vol[1, 1, 1] = 100.0  # sits at 2,2,2 mm -> display voxel (2,2,2)
    got = extract_plane(vol, grid, layer_affine, Plane.AXIAL, 2)
    assert abs(float(got[2, 2]) - 100.0) < 1e-3
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
