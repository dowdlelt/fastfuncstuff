"""The profile column's model: order, zoomed-out rows, cortex mask, edits."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from fastfuncstuff.io.freesurfer import Hemisphere
from fastfuncstuff.viewer.profilecolumn import build_column, update_column

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def scene():
    vox, n = 0.5, 120
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3], axis=-1)
    img = np.where(r < 21, 110.0, np.where(r < 24, 70.0, 20.0)).astype(np.float32)
    img = gaussian_filter(img, 0.5) + np.random.default_rng(2).normal(0, 4, img.shape)
    k = 3000
    i = np.arange(k) + 0.5
    phi, theta = np.arccos(1 - 2 * i / k), np.pi * (1 + 5**0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    faces = ConvexHull(u).simplices.astype(np.int32)
    return img.astype(np.float32), aff, u, faces


def _hemi(u, faces, cortex=None, pial_r=24.0):
    return Hemisphere(
        name="lh",
        faces=faces,
        states={
            "white": (21.0 * u).astype(np.float32),
            "pial": (pial_r * u).astype(np.float32),
            "sphere": (100.0 * u).astype(np.float32),
        },
        tkr_to_scanner=np.eye(4),
        cortex=cortex,
    )


def test_medial_wall_has_no_rows_and_every_cortex_vertex_has_one(scene):
    img, aff, u, faces = scene
    cortex = u[:, 0] < 0.8  # a "medial wall" cap at +x
    col = build_column({"lh": _hemi(u, faces, cortex)}, img, aff, device=CPU)
    assert col.n_rows == int(cortex.sum())
    assert set(col.row_vertex.tolist()) == set(np.flatnonzero(cortex).tolist())
    rows = col.row_of("lh", np.array([np.flatnonzero(cortex)[0], np.flatnonzero(~cortex)[0]]))
    assert rows[0] >= 0 and rows[1] == -1


def test_zoomed_out_rows_show_the_worst_profile_not_an_average(scene):
    img, aff, u, faces = scene
    pial = 24.0 * u
    bad = int(np.argmax(u[:, 2]))
    pial[bad] = 26.0 * u[bad]  # one pial vertex run out through CSF
    h = _hemi(u, faces)
    h.states["pial"] = pial.astype(np.float32)
    col = build_column({"lh": h}, img, aff, device=CPU)
    bad_row = int(col.row_of("lh", np.array([bad]))[0])
    assert col.scores["worst"][bad_row] > 0.8
    # Squeeze every row into 10 pixels: the bad row must be one of them.
    shown = col.pick_rows(0, col.n_rows, 10, "worst")
    assert bad_row in shown.tolist()
    # Zoomed in, one row per pixel.
    np.testing.assert_array_equal(col.pick_rows(100, 110, 10, "worst"), np.arange(100, 110))
    density = col.flag_density(0, col.n_rows, 10, "worst")
    assert density.shape == (10,) and density.max() > 0


def test_an_edit_resamples_only_what_moved_and_the_flag_clears(scene):
    img, aff, u, faces = scene
    h = _hemi(u, faces)
    bad = int(np.argmax(u[:, 2]))
    h.states["pial"][bad] = (26.0 * u[bad]).astype(np.float32)
    col = build_column({"lh": h}, img, aff, device=CPU)
    row = int(col.row_of("lh", np.array([bad]))[0])
    before = col.profiles.values.copy()
    assert col.scores["pial_out"][row] > 0.8
    h.states["pial"][bad] = (24.0 * u[bad]).astype(np.float32)  # fixed by an edit
    changed = update_column(col, {"lh": h}, img, aff, device=CPU)
    assert changed.tolist() == [row]
    others = np.ones(col.n_rows, bool)
    others[row] = False
    np.testing.assert_array_equal(col.profiles.values[others], before[others])
    assert col.scores["pial_out"][row] < 0.7
    assert col.nearest_row(tuple(0.5 * (21.0 + 24.0) * u[bad])) == row


def test_flags_map_back_to_vertices_with_nan_on_the_medial_wall(scene):
    from fastfuncstuff.viewer import surface3d as s3
    from fastfuncstuff.viewer.profilecolumn import flags_by_vertex

    img, aff, u, faces = scene
    cortex = u[:, 0] < 0.8
    h = _hemi(u, faces, cortex)
    bad = int(np.argmax(np.where(cortex, u[:, 2], -9)))
    h.states["pial"][bad] = (26.0 * u[bad]).astype(np.float32)
    col = build_column({"lh": h}, img, aff, device=CPU)
    flags = flags_by_vertex(col, "worst", {"lh": h.n_vertices})["lh"]
    assert np.isnan(flags[~cortex]).all() and np.isfinite(flags[cortex]).all()
    assert flags[bad] > 0.8
    rgba = s3.vertex_colors(h, "flags", flags=flags)
    assert rgba is not None
    # Flagged: opaque and red; fine: transparent, so the anatomy shows.
    assert rgba[bad, 3] == 255 and rgba[bad, 0] > 150 and rgba[bad, 1] < 100
    fine = cortex & (flags < 0.3)
    assert (rgba[fine, 3] == 0).all()
    assert (rgba[~cortex, 3] == 0).all()
