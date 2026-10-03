"""Snapping edits on a phantom whose true boundaries are known.

The phantom is concentric: WM inside r=21 mm, GM to r=24, CSF beyond (T1
contrast). The meshes start 1 mm wrong -- white at 20, pial at 23 -- which is
the "the surface did not go far enough" case the editor exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from fastfuncstuff.surface.edit import SnapParams, SurfaceEdit
from fastfuncstuff.surface.mesh import MeshTopology, face_normals, geodesic_ball
from fastfuncstuff.surface.sampling import VolumeSampler

WHITE_TRUE, PIAL_TRUE = 21.0, 24.0


def _sphere(n=6000):
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    f = ConvexHull(u).simplices.astype(np.int64)
    # Outward winding, as FreeSurfer stores it.
    fn = face_normals(u, f)
    flip = np.einsum("ij,ij->i", fn, u[f].mean(1)) < 0
    f[flip] = f[flip][:, ::-1]
    return u, f


@pytest.fixture(scope="module")
def phantom():
    vox = 0.5
    n = 120
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3], axis=-1)
    img = np.where(r < WHITE_TRUE, 110.0, np.where(r < PIAL_TRUE, 70.0, 20.0)).astype(np.float32)
    u, f = _sphere()
    return VolumeSampler(img, aff), u, f, MeshTopology.from_faces(f)


def _top(u):
    return int(np.argmax(u[:, 2]))


def test_drag_out_snaps_white_to_the_true_boundary_and_leaves_the_rest(phantom):
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    c = _top(u)
    edit = SurfaceEdit(white, topo, c, sampler, SnapParams(radius=5.0), role="white", partner=pial)
    res = edit.update(np.array([0.0, 0.0, 0.6]))  # a rough drag, not the exact 1 mm
    r_new = np.linalg.norm(res.positions, axis=1)
    core = edit.weight > 0.8
    assert core.sum() > 5
    np.testing.assert_allclose(r_new[core], WHITE_TRUE, atol=0.15)
    # No kink where the brush meets the untouched mesh: across every edge,
    # including rim-to-outside ones, the displacement changes by well under
    # the 1 mm correction.
    full = np.zeros(len(u))
    full[res.ids] = res.displacement
    jump = np.abs(full[topo.edges[:, 0]] - full[topo.edges[:, 1]])
    assert jump.max() < 0.5
    # Nothing outside the brush is in the result.
    inside, _ = geodesic_ball(white, topo, c, 5.0)
    assert set(res.ids.tolist()) == set(inside.tolist())


def test_drag_back_to_the_start_restores_exactly_with_no_snap(phantom):
    sampler, u, f, topo = phantom
    white = 20.0 * u
    edit = SurfaceEdit(white, topo, _top(u), sampler, SnapParams(snap=0.0))
    edit.update(np.array([0.0, 0.0, 1.0]))
    res = edit.update(np.zeros(3))
    np.testing.assert_allclose(res.positions, white[res.ids], atol=1e-12)


def test_pial_cannot_be_dragged_through_white(phantom):
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    p = SnapParams(radius=4.0, snap=0.0, min_thickness=0.1)
    edit = SurfaceEdit(pial, topo, _top(u), sampler, p, role="pial", partner=white)
    res = edit.update(np.array([0.0, 0.0, -6.0]))
    thickness = np.linalg.norm(res.positions, axis=1) - 20.0
    assert thickness.min() >= 0.1 - 1e-6
    # ...and stops *at* white rather than being thrown anywhere outside it.
    assert thickness.min() == pytest.approx(0.1, abs=0.05)
    # A small inward nudge well clear of white is not clamped at all.
    small = edit.update(np.array([0.0, 0.0, -0.5]))
    assert small.displacement.min() == pytest.approx(-0.5, abs=0.05)


def test_white_pushed_out_pushes_pial_ahead(phantom):
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 20.5 * u
    p = SnapParams(radius=4.0, snap=0.0, min_thickness=0.1)
    edit = SurfaceEdit(white, topo, _top(u), sampler, p, role="white", partner=pial)
    res = edit.update(np.array([0.0, 0.0, 2.0]))
    assert res.partner_ids.size > 0
    new_pial = pial.copy()
    new_pial[res.partner_ids] = res.partner_positions
    new_white = white.copy()
    new_white[res.ids] = res.positions
    gap = np.linalg.norm(new_pial, axis=1) - np.linalg.norm(new_white, axis=1)
    assert gap.min() >= 0.1 - 1e-6


def test_no_face_flips_even_for_a_violent_drag(phantom):
    sampler, u, f, topo = phantom
    white = 20.0 * u
    edit = SurfaceEdit(white, topo, _top(u), sampler, SnapParams(radius=2.0, snap=0.0, smooth=0.0))
    res = edit.update(np.array([0.0, 0.0, -30.0]))
    moved = white.copy()
    moved[res.ids] = res.positions
    before, after = face_normals(white, f), face_normals(moved, f)
    assert np.all(np.einsum("ij,ij->i", before, after) > 0)


def test_brush_does_not_jump_across_a_sulcus():
    # Two parallel sheets 1 mm apart joined only far away: a 3 mm Euclidean
    # ball would take both banks; the geodesic one must take one.
    xs = np.arange(0, 21)
    ys = np.arange(0, 11)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1).astype(float)
    bottom = top + [0, 0, -1.0]
    verts = np.concatenate([top, bottom])
    nx, ny = len(xs), len(ys)

    def grid_faces(offset):
        out = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                a = offset + i * ny + j
                out += [[a, a + ny, a + 1], [a + 1, a + ny, a + ny + 1]]
        return out

    faces = np.array(grid_faces(0) + grid_faces(top.shape[0]), np.int64)
    topo = MeshTopology.from_faces(faces, verts.shape[0])
    centre = 10 * ny + 5
    inside, _ = geodesic_ball(verts, topo, centre, 3.0)
    assert np.all(inside < top.shape[0])


def test_pial_dragged_inward_does_not_snap_onto_the_white_boundary(phantom):
    # Pial at 23 dragged 1.5 mm in: the search band [20, 23] holds the WM/GM
    # edge (r=21), which also darkens outward. It is the wrong boundary; with
    # no GM/CSF crossing in reach the hand's position should stand.
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    edit = SurfaceEdit(
        pial, topo, _top(u), sampler, SnapParams(radius=4.0), role="pial", partner=white
    )
    res = edit.update(np.array([0.0, 0.0, -1.5]))
    core = np.linalg.norm(res.positions[edit.weight > 0.8], axis=1)
    assert np.all(np.abs(core - WHITE_TRUE) > 0.5)


def test_pial_dragged_out_finds_the_csf_boundary(phantom):
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    edit = SurfaceEdit(
        pial, topo, _top(u), sampler, SnapParams(radius=5.0), role="pial", partner=white
    )
    level, contrast = edit.levels
    assert level == pytest.approx(45.0, abs=3) and contrast == pytest.approx(50.0, abs=5)
    res = edit.update(np.array([0.0, 0.0, 0.6]))
    core = np.linalg.norm(res.positions[edit.weight > 0.8], axis=1)
    np.testing.assert_allclose(core, PIAL_TRUE, atol=0.15)


def test_inward_wound_mesh_still_pushes_outward(phantom):
    # Same sphere with every face reversed: normals point in. The edit must
    # still treat white -> pial as outward.
    sampler, u, f, _ = phantom
    flipped = f[:, ::-1].copy()
    topo = MeshTopology.from_faces(flipped)
    white, pial = 20.0 * u, 23.0 * u
    edit = SurfaceEdit(
        white, topo, _top(u), sampler, SnapParams(radius=5.0), role="white", partner=pial
    )
    res = edit.update(np.array([0.0, 0.0, 0.6]))
    core = np.linalg.norm(res.positions[edit.weight > 0.8], axis=1)
    np.testing.assert_allclose(core, WHITE_TRUE, atol=0.15)
    assert res.partner_ids.size == 0
