"""A display grid finer than the underlay (SET_GRID_RES).

The point is to draw a fine overlay at its own resolution and interpolate the
anatomy instead of the other way round. What breaks silently is position: the
crosshair, seed, pan and parked slices are all held in display-grid indices, so
a regrid that forgets one of them moves it to a different place in the head.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.state import grid_res_choices, resize_grid
from fastfuncstuff.viewer.vocab import (
    Load,
    OpenView,
    SetGridRes,
    SetPan,
    SetSeed,
    SetViewPosition,
)

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
ORIGIN = np.array([-16.0, -20.0, -12.0])


def _edges(shape, affine):
    """Outer voxel *edges* of a grid along each axis, in mm."""
    a = np.asarray(affine, float)
    lo = a[:3, 3] - 0.5 * a[:3, :3].sum(axis=1)
    hi = a[:3, :3] @ (np.asarray(shape) - 0.5) + a[:3, 3]
    return lo, hi


# ---------------------------------------------------------------------------
# resize_grid
# ---------------------------------------------------------------------------


def test_zero_keeps_the_grid():
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    shape, out = resize_grid((10, 12, 8), aff, 0.0)
    assert shape == (10, 12, 8)
    assert np.array_equal(out, aff)


def test_refining_keeps_the_field_of_view_edge_to_edge():
    """Edges, not outer centres: otherwise the refined grid pokes past the underlay."""
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # oblique-ish storage
    aff = np.eye(4)
    aff[:3, :3] = rot @ np.diag([2.0, 3.0, 2.5])
    aff[:3, 3] = (5.0, -7.0, 3.0)
    shape, out = resize_grid((10, 12, 8), aff, 1.0)
    assert shape == (20, 36, 20)
    for a, b in zip(_edges((10, 12, 8), aff), _edges(shape, out), strict=True):
        assert np.allclose(a, b)
    # Same directions, only shorter steps.
    cols = out[:3, :3] / np.linalg.norm(out[:3, :3], axis=0)
    assert np.allclose(cols, rot)


def test_resizing_goes_both_ways_per_axis():
    """A 0.5 mm-thick slab at 1 mm: coarser in-plane, finer through it."""
    shape, out = resize_grid((10, 12, 40), np.diag([2.0, 2.0, 0.5, 1.0]), 1.0)
    assert shape == (20, 24, 20)
    assert np.allclose(np.diag(out)[:3], (1.0, 1.0, 1.0))
    coarse, _ = resize_grid((30, 36, 24), np.diag([1.0, 1.0, 1.0, 1.0]), 3.0)
    assert coarse == (10, 12, 8)


# ---------------------------------------------------------------------------
# in a session
# ---------------------------------------------------------------------------


def _save(path, data, step, origin):
    aff = np.diag([step, step, step, 1.0])
    aff[:3, 3] = origin
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff), str(path))
    return path


@pytest.fixture
def session(tmp_path):
    """A 2 mm anatomy and a 1 mm map covering exactly the same box."""
    rng = np.random.default_rng(7)
    _save(tmp_path / "anat.nii.gz", rng.random((16, 20, 12)) * 100, 2.0, ORIGIN)
    # Half a 1 mm voxel inside the 2 mm grid's first centre: the same outer edge.
    _save(tmp_path / "map.nii.gz", rng.normal(size=(32, 40, 24)), 1.0, ORIGIN - 0.5)
    s = ViewerSession(device=CPU)
    s.do(Load(str(tmp_path / "anat.nii.gz")))
    s.do(Load(str(tmp_path / "map.nii.gz")))
    yield s
    s.close()


def _map(session):
    return next(ly for ly in session.state.layers if ly.name.startswith("map"))


def test_refining_to_the_maps_size_puts_the_map_on_its_own_voxels(session):
    """The whole request: the overlay drawn voxel-for-voxel, not resampled."""
    session.do(SetGridRes(1.0))
    grid = session.state.grid
    assert grid.shape == (32, 40, 24)
    assert np.allclose(grid.affine, _map(session).affine)


def test_auto_interpolates_the_underlay_on_a_finer_grid(session):
    session.do(SetGridRes(1.0))
    anat = session.state.layers.base
    assert session.resample_mode(anat) == "linear"
    assert session.resample_mode(_map(session)) == "linear"


def test_positions_survive_a_round_trip(session):
    st = session.state
    session.do(OpenView("I1", "image", "axial"))
    session.do(SetSeed(3, 4, 5))
    session.do(SetPan("I1", 2.0, -3.0))
    session.do(SetViewPosition("I1", 7))
    before_mm = st.crosshair_mm
    before = (st.crosshair, st.seed, st.viewports.get("I1").pan, st.viewports.get("I1").position)

    session.do(SetGridRes(1.0))
    # On the fine grid: the crosshair sits within half a fine voxel of where it was...
    assert np.allclose(st.crosshair_mm, before_mm, atol=0.5 + 1e-6)
    # ...and a pan of grid voxels doubles, so the picture stays put.
    assert st.viewports.get("I1").pan == (4.0, -6.0)

    session.do(SetGridRes(0.0))
    after = (st.crosshair, st.seed, st.viewports.get("I1").pan, st.viewports.get("I1").position)
    assert after == before


def test_an_unchanged_size_does_nothing(session):
    from fastfuncstuff.viewer.commands import Aspect

    session.do(SetGridRes(1.0))
    assert session.do(SetGridRes(1.0)) is Aspect.NOTHING


def test_set_before_loading_applies_when_the_underlay_arrives(tmp_path):
    """A replayed script sets the size first and loads after."""
    _save(tmp_path / "anat.nii.gz", np.ones((16, 20, 12)), 2.0, ORIGIN)
    s = ViewerSession(device=CPU)
    try:
        s.do(SetGridRes(1.0))
        s.do(Load(str(tmp_path / "anat.nii.gz")))
        assert s.state.grid.shape == (32, 40, 24)
        assert "SET_GRID_RES 1.0" in s.to_script()
    finally:
        s.close()


def test_the_picker_offers_each_layers_own_size(session):
    choices = {mm: label for label, mm in grid_res_choices(session.state)}
    assert choices[0.0].startswith("underlay (2 mm")
    assert choices[1.0].startswith("match map")
    # The underlay's own size is "underlay", not offered a second time as "2 mm".
    assert 2.0 not in choices
    assert 3.0 in choices and 0.5 in choices


def test_a_coarser_grid_draws_the_anatomy_at_the_runs_size(session):
    """The other direction: which layer is finer flips between datasets."""
    session.do(SetGridRes(4.0))
    assert session.state.grid.shape == (8, 10, 6)
