"""The ribbon tiling behind thick-depth reads, refereed by concentric spheres.

Between a white sphere (radius RW) and a pial one (RP) the cells must add up to the
exact volume between the two polyhedra, equidistant bins are shells of equal
thickness and equivolume bins shells of equal volume -- also when every pial vertex
slides sideways of its white partner, as FreeSurfer's do, because the pairing only
tiles the ribbon and never decides depth.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.surface.slab import patch_faces, ribbon_cells
from fastfuncstuff.surface.volume_depth import RibbonSurfaces
from tests.test_surface_voxelize import icosphere

RW, RP = 15.0, 21.0


def _polyhedron_volume(x, f):
    t = x[f]
    return np.einsum("fx,fx->f", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6


def _sphere(slide=0.0):
    v, f = icosphere(4)
    c, s = np.cos(slide), np.sin(slide)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return RibbonSurfaces("lh", v * RW, (v @ turn.T) * RP, f.astype(np.int64)), v, f


def _bins(cells, k):
    cell, b, share = cells.bin_shares(k)
    vol = np.bincount(b, cells.volume[cell] * share, k)
    r = np.linalg.norm(cells.centre, axis=1)[cell]
    return vol, np.bincount(b, cells.volume[cell] * share * r, k) / vol


@pytest.mark.slow  # 9-11s per case, 4 cases
@pytest.mark.parametrize("slide", [0.0, 0.08])  # 0.08 rad: 1.7 mm at the pial sphere
@pytest.mark.parametrize("method", ["equidistant", "equivolume"])
def test_cells_tile_the_ribbon_and_bins_are_the_shells(method, slide):
    surf, v, f = _sphere(slide)
    k = 4
    cells = ribbon_cells([surf], {0: np.arange(len(f))}, step=1.0, n_depths=k, method=method)
    exact = _polyhedron_volume(v * RP, f) - _polyhedron_volume(v * RW, f)
    np.testing.assert_allclose(cells.volume.sum(), exact, rtol=1e-9)
    assert cells.inverted_mm3 == 0
    t = np.linspace(0, 1, k + 1)
    edges = (
        RW + t * (RP - RW) if method == "equidistant" else (RW**3 + t * (RP**3 - RW**3)) ** (1 / 3)
    )
    vol, mean_r = _bins(cells, k)
    # The polyhedra sit a little inside the spheres, so compare shares, not mm^3.
    np.testing.assert_allclose(
        vol / vol.sum(), np.diff(edges**3) / np.diff(edges**3).sum(), atol=0.004
    )
    np.testing.assert_allclose(mean_r, 0.75 * np.diff(edges**4) / np.diff(edges**3), atol=0.06)


def test_a_cell_across_a_boundary_splits_its_volume():
    surf, v, f = _sphere()
    cells = ribbon_cells([surf], {0: np.arange(len(f))}, step=2.0, n_depths=3)
    cell, b, share = cells.bin_shares(3)
    np.testing.assert_allclose(np.bincount(cell, share, len(cells)), 1.0)
    # 7 sheets through the depth cannot all stop on thirds: some cells are cut.
    assert (np.bincount(cell, minlength=len(cells)) > 1).any()


@pytest.mark.slow  # 14s
def test_a_patch_holds_every_cell_its_columns_own():
    surf, v, f = _sphere(slide=0.08)
    cap = np.flatnonzero(v[:, 2] > 0.6)
    whole = ribbon_cells([surf], {0: np.arange(len(f))}, step=1.0, n_depths=2)
    margin = float(np.linalg.norm(surf.pial - surf.white, axis=1).max()) + 3.0
    patch = ribbon_cells([surf], {0: patch_faces(surf, cap, margin)}, step=1.0, n_depths=2)
    in_cap = np.isin(whole.vertex, cap)
    np.testing.assert_allclose(
        patch.volume[np.isin(patch.vertex, cap)].sum(), whole.volume[in_cap].sum(), rtol=1e-9
    )


def test_equivolume_cells_beat_equidistant_on_bok_truth_with_pial_slide():
    """Folded sphere, parallel surfaces, pial vertices slid ~1 mm; truth is the volume
    fraction below each point, integrated from offset-mesh areas (Bok's constant-volume
    columns). Cells whose volume sits at a few discrete depths made the column CDF a
    staircase and equivolume bins 4-9% off; the error against truth must stay far below
    equidistant's."""
    from fastfuncstuff.surface.mesh import MeshTopology, vertex_areas, vertex_normals
    from fastfuncstuff.surface.slab import equivolume_metric, tile_surface
    from fastfuncstuff.surface.voxelize import MeshDistance

    R, T, K = 12.0, 1.5, 4
    u, f = icosphere(4)
    f = f.astype(np.int64)
    x, y, z = u.T
    bump = np.sin(5.0 * np.arctan2(y, x)) * np.sin(4.0 * np.arccos(np.clip(z, -1, 1)))
    white = u * (R + 0.8 * bump)[:, None]
    n = vertex_normals(white, MeshTopology.from_faces(f, len(white)))
    pial = white + T * n
    ts = np.linspace(0, T, 16)
    area = np.stack([vertex_areas(white + t * n, f, len(white)) for t in ts], 1)
    cum = np.concatenate(
        [np.zeros((len(white), 1)), np.cumsum((area[:, 1:] + area[:, :-1]) / 2 * np.diff(ts), 1)], 1
    )
    cum /= cum[:, -1:]
    slide = np.sin(pial @ np.random.default_rng(0).normal(size=(3, 3)) * 0.5) * 1.2
    slide -= (slide * n).sum(1, keepdims=True) * n
    foot = MeshDistance(pial, f)(pial + slide, torch.device("cpu"))
    slid = np.stack([foot.interpolate(f, pial[:, k]) for k in range(3)], 1)
    surf = RibbonSurfaces("lh", white, slid, f)

    tiling = tile_surface(surf, np.arange(len(f)), 0.5, K)
    t = np.clip(tiling.rho, 0, 1) * (len(ts) - 1)
    i0 = np.minimum(t.astype(int), len(ts) - 2)
    fr = t - i0
    fv = f[tiling.foot_face]
    truth = sum(
        tiling.foot_bary[:, k] * (cum[fv[:, k], i0] * (1 - fr) + cum[fv[:, k], i0 + 1] * fr)
        for k in range(3)
    )
    err_equidist = np.abs(tiling.rho - truth).mean()
    err_equivol = np.abs(equivolume_metric(tiling, surf, 1) - truth).mean()
    assert err_equivol < 0.25 * err_equidist, (err_equivol, err_equidist)


@pytest.mark.slow  # 27s
def test_thicker_sheets_reach_the_tiling_and_the_estimate_is_exact():
    """The read budget relaxes only the sheet thickness; that must change the cells
    actually made, not just the estimate of them."""
    from fastfuncstuff.surface.slab import estimate_cells

    surf, v, f = _sphere()
    faces = np.arange(len(f))
    for through in (0.5, 1.0):
        n = len(ribbon_cells([surf], {0: faces}, step=1.0, n_depths=2, through=through))
        assert n == estimate_cells(surf, faces, 1.0, 2, through=through)
    thin = len(ribbon_cells([surf], {0: faces}, step=1.0, n_depths=2, through=0.5))
    thick = len(ribbon_cells([surf], {0: faces}, step=1.0, n_depths=2, through=1.0))
    assert thick < thin


@pytest.mark.slow  # 17s
def test_float32_vertices_do_not_change_the_tiling():
    surf, v, f = _sphere()
    s32 = RibbonSurfaces("lh", surf.white.astype(np.float32), surf.pial.astype(np.float32), f)
    a = ribbon_cells([surf], {0: np.arange(len(f))}, step=1.0, n_depths=3)
    b = ribbon_cells([s32], {0: np.arange(len(f))}, step=1.0, n_depths=3)
    assert len(a) == len(b)
