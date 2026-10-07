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
    # The rim's sqrt fade puts the largest step there: 0.49 mm with raw
    # vertex normals, 0.50 with the smoothed ones a drag now moves along.
    assert jump.max() < 0.55
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
    # Ungated ("edge" mode), the strongest edge wins -- here the wrong one,
    # which is why the gate is the default and "edge" a deliberate choice.
    ungated = SurfaceEdit(
        pial, topo, _top(u), sampler, SnapParams(radius=4.0, gate=False), role="pial", partner=white
    )
    assert ungated.levels is None
    core = np.linalg.norm(
        ungated.update(np.array([0.0, 0.0, -1.5])).positions[ungated.weight > 0.8], axis=1
    )
    assert np.median(np.abs(core - WHITE_TRUE)) < 0.5


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


def _draw_on_slice(u, f, z=15.0, white_r=20.0, target_r=21.0, half_angle=0.6):
    """The viewer's draw gesture on the phantom, in scanner mm.

    Returns the seeds (vertices of the faces the slice cuts between the
    stroke's ends) and the stroke: from the outline, out along the true
    boundary, back to the outline.
    """
    from fastfuncstuff.surface.geometry import SliceIndex, contour_path

    white = white_r * u
    index = SliceIndex(white, f)
    seg, faces, edges = index.segments_with_edges(2, z)
    rho_now = np.sqrt(white_r**2 - z**2)
    rho_new = np.sqrt(target_r**2 - z**2)
    ends = [
        np.array([rho_now * np.cos(a), rho_now * np.sin(a), z]) for a in (-half_angle, half_angle)
    ]
    mids = seg.mean(axis=1)
    a_seg, b_seg = (int(np.argmin(np.linalg.norm(mids - e, axis=1))) for e in ends)
    path = contour_path(edges, a_seg, b_seg)
    assert path is not None
    seeds = np.unique(f[faces[path]])
    arc = np.linspace(-half_angle, half_angle, 40)
    stroke = np.concatenate(
        [
            ends[0][None],
            np.stack([rho_new * np.cos(arc), rho_new * np.sin(arc), np.full_like(arc, z)], 1),
            ends[1][None],
        ]
    )
    return white, seeds, stroke


def test_a_drawn_stretch_moves_to_the_stroke_and_the_surface_around_follows(phantom):
    from fastfuncstuff.surface.edit import StrokeEdit

    sampler, u, f, topo = phantom
    white, seeds, stroke = _draw_on_slice(u, f)
    params = SnapParams(radius=4.0, snap=0.0)
    edit = StrokeEdit(white, topo, seeds, stroke, sampler, params, role="white", partner=23.0 * u)
    res = edit.result()
    r_new = np.linalg.norm(res.positions, axis=1)
    middle = edit.seed & (np.abs(np.arctan2(edit.start[:, 1], edit.start[:, 0])) < 0.3)
    assert middle.sum() >= 3
    # Hand mode: the redrawn stretch sits on the stroke (r = 21 on the slice).
    np.testing.assert_allclose(r_new[middle], 21.0, atol=0.25)
    # One slice-width above, the surface came along -- part of the way.
    above = (
        ~edit.seed
        & (edit.start[:, 2] > 16.5)
        & (np.abs(np.arctan2(edit.start[:, 1], edit.start[:, 0])) < 0.3)
    )
    assert above.any()
    assert np.all((res.displacement[above] > 0.05) & (res.displacement[above] < 1.0))
    # Nothing on the other side of the sphere moves.
    assert np.all(edit.start[:, 0] > 0)


def test_auto_mode_places_the_drawn_stretch_on_the_edge(phantom):
    from fastfuncstuff.surface.edit import StrokeEdit

    sampler, u, f, topo = phantom
    # A sloppy stroke: drawn at r = 21.4, the true boundary is 21.
    white, seeds, stroke = _draw_on_slice(u, f, target_r=21.4)
    hand = StrokeEdit(
        white,
        topo,
        seeds,
        stroke,
        sampler,
        SnapParams(radius=4.0, snap=0.0),
        role="white",
        partner=23.0 * u,
    )
    auto = StrokeEdit(
        white,
        topo,
        seeds,
        stroke,
        sampler,
        SnapParams(radius=4.0, snap=1.0),
        role="white",
        partner=23.0 * u,
    )
    mid = hand.seed & (np.abs(np.arctan2(hand.start[:, 1], hand.start[:, 0])) < 0.3)
    r_hand = np.linalg.norm(hand.result().positions[mid], axis=1)
    r_auto = np.linalg.norm(auto.result().positions[mid], axis=1)
    assert np.abs(r_auto - WHITE_TRUE).mean() < np.abs(r_hand - WHITE_TRUE).mean()
    np.testing.assert_allclose(r_auto, WHITE_TRUE, atol=0.2)


def test_contour_path_takes_the_short_way_and_refuses_separate_pieces():
    from fastfuncstuff.surface.geometry import contour_path

    # A ring of 10 segments: segment i joins edge i to edge i+1.
    ring = np.array([[[i, i + 100], [(i + 1) % 10, (i + 1) % 10 + 100]] for i in range(10)])
    np.testing.assert_array_equal(contour_path(ring, 1, 3), [1, 2, 3])
    np.testing.assert_array_equal(contour_path(ring, 1, 9), [1, 0, 9])
    two = np.concatenate([ring, ring + 1000])
    assert contour_path(two, 1, 12) is None


def test_a_fold_is_damped_where_it_happens_not_across_the_whole_brush(phantom):
    """The global halving cost the vertex under the cursor half its move
    whenever any face in the brush flipped -- on a real subject, the median
    2 mm pial drag. A fold at one spot must leave the centre its full move."""
    sampler, u, f, topo = phantom
    white = 20.0 * u
    c = _top(u)
    edit = SurfaceEdit(white, topo, c, sampler, SnapParams(radius=6.0, snap=0.0, smooth=0.0))
    # A hand-made fold near the rim: a vertex and its ring pushed inward
    # through the sphere's centre, while the rest move a gentle 0.5 mm.
    ids = edit.ids
    k = int(np.argmin(np.abs(edit.weight - 0.15)))
    ring = np.r_[k, edit._adjacency[k].indices]
    along = 0.5 * edit.weight
    along[ring] -= 45.0
    res = edit._finish(along)
    centre = int(np.flatnonzero(ids == c)[0])
    assert edit.fold_damped > 0
    assert res.displacement[centre] == pytest.approx(along[centre], rel=0.05)
    moved = white.copy()
    moved[res.ids] = res.positions
    before, after = face_normals(white, f), face_normals(moved, f)
    assert np.all(np.einsum("ij,ij->i", before, after) > 0)


def test_an_edit_says_why_it_did_not_do_what_was_asked(phantom):
    from fastfuncstuff.surface.edit import explain

    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    c = _top(u)
    # Pial dragged 5 mm in, through white 3 mm below it: held.
    edit = SurfaceEdit(
        pial, topo, c, sampler, SnapParams(radius=4.0, snap=0.0), role="pial", partner=white
    )
    res = edit.update(np.array([0.0, 0.0, -5.0]))
    assert res.held > 0
    assert "held at white" in explain(res)
    # Snap: a rough 0.6 mm drag the edge finishes to 1 mm, outward.
    edit = SurfaceEdit(white, topo, c, sampler, SnapParams(radius=5.0), role="white", partner=pial)
    res = edit.update(np.array([0.0, 0.0, 0.6]))
    assert res.snap_offset == pytest.approx(0.4, abs=0.15)
    assert "snap moved it" in explain(res) and "out" in explain(res)
    # A plain hand drag that nothing holds back says nothing.
    edit = SurfaceEdit(
        white, topo, c, sampler, SnapParams(radius=5.0, snap=0.0), role="white", partner=pial
    )
    assert explain(edit.update(np.array([0.0, 0.0, 0.5]))) == ""


def test_stroke_targets_cast_along_the_outline_not_to_the_nearest_point():
    """A stroke that starts on the outline and swerves 1.5 mm out: a vertex
    1 mm along is nearer the stroke's start than the swerve, and the nearest
    point gave it no shift. Cast along its normal, it reaches the swerve."""
    from fastfuncstuff.surface.edit import closest_on_polyline, stroke_targets

    # Surface: the plane x = 0 seen in the slice z = 0, normal +x. The vertex
    # sits 0.5 mm above the slice, so its own outline point is (0, 1, 0).
    vertex = np.array([[0.0, 1.0, 0.5]])
    normal = np.array([[1.0, 0.0, 0.0]])
    stroke = np.array([[0.0, 0.0, 0.0], [1.5, 0.3, 0.0], [1.5, 4.0, 0.0]])
    near = closest_on_polyline(vertex, stroke)
    cast = stroke_targets(vertex, normal, stroke, np.array([0.0, 0.0, 1.0]))
    assert near[0, 0] < 1.0  # the trap: pulled toward the start
    np.testing.assert_allclose(cast[0], [1.5, 1.0, 0.0], atol=1e-9)
    # Lying in the slice (normal along z) there is no outline point to cast.
    flat = stroke_targets(vertex, np.array([[0.0, 0.0, 1.0]]), stroke, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(flat, near)


def test_a_free_hand_drag_goes_where_it_is_dragged_not_along_the_normal(phantom):
    """At the sphere's top the normal is +z: a sideways drag along normals
    moves nothing, which is the "it will not go where I drag" report."""
    sampler, u, f, topo = phantom
    white, pial = 20.0 * u, 23.0 * u
    c = _top(u)
    drag = np.array([1.0, 0.0, 0.0])
    normal = SurfaceEdit(
        white, topo, c, sampler, SnapParams(radius=4.0, snap=0.0), role="white", partner=pial
    )
    free = SurfaceEdit(
        white,
        topo,
        c,
        sampler,
        SnapParams(radius=4.0, snap=0.0, free=True),
        role="white",
        partner=pial,
    )
    k = int(np.flatnonzero(normal.ids == c)[0])
    assert np.linalg.norm(normal.update(drag).positions[k] - white[c]) < 0.05
    res = free.update(drag)
    np.testing.assert_allclose(res.positions[k] - white[c], drag, atol=1e-6)
    moved = white.copy()
    moved[res.ids] = res.positions
    before, after = face_normals(white, f), face_normals(moved, f)
    assert np.all(np.einsum("ij,ij->i", before, after) > 0)
    # Pial dragged freely down into white is still held outside it.
    edit = SurfaceEdit(
        pial,
        topo,
        c,
        sampler,
        SnapParams(radius=4.0, snap=0.0, free=True),
        role="pial",
        partner=white,
    )
    res = edit.update(np.array([0.0, 0.0, -5.0]))
    assert res.held > 0
    # Along the (smoothed) normal, so within a hair of the 0.1 mm floor radially.
    assert np.linalg.norm(res.positions[k]) >= 20.0 + 0.09


def _sheet(n=41, spacing=0.8):
    xs = (np.arange(n) - n // 2) * spacing
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    v = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + n, a + 1], [a + 1, a + n, a + n + 1]]
    f = np.asarray(faces, np.int64)
    # Normals +z (outward) for this winding.
    if face_normals(v, f)[:, 2].mean() < 0:
        f = f[:, ::-1]
    return v, f


def _flat_sampler():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = -40
    return VolumeSampler(np.zeros((81, 81, 81), np.float32), aff)


def test_flatten_brings_a_spike_down_further_than_the_plateau():
    from fastfuncstuff.surface.edit import HighlightEdit

    v, f = _sheet()
    topo = MeshTopology.from_faces(f)
    centre = len(v) // 2
    v[centre, 2] = 1.0  # one vertex sticking out 1 mm
    near = np.flatnonzero(np.linalg.norm(v[:, :2], axis=1) < 3.0)
    params = SnapParams(radius=2.0, snap=0.0)
    plain = HighlightEdit(v, topo, near, -0.25, _flat_sampler(), params, role="pial").result()
    flat = HighlightEdit(
        v, topo, near, -0.25, _flat_sampler(), params, role="pial", flatten=0.5
    ).result()

    def dz(res, k):
        return res.positions[res.ids == k][0, 2] - v[k, 2]

    other = int(near[near != centre][0])
    assert dz(plain, centre) == pytest.approx(dz(plain, other), abs=1e-4)  # uniform today
    # The spike comes down by the step plus half its height; the plateau by
    # about the step (a little less beside the spike: it sits below its
    # neighbourhood's mean now).
    assert dz(flat, centre) == pytest.approx(-0.25 - 0.5 * 1.0, abs=0.05)
    far = int(near[np.argmax(np.linalg.norm(v[near, :2], axis=1))])
    assert dz(flat, far) == pytest.approx(-0.25, abs=0.02)
    # Relax only: no plateau shift, the spike still comes down.
    relax = HighlightEdit(v, topo, near, 0.0, _flat_sampler(), params, role="pial", flatten=0.5)
    assert dz(relax.result(), centre) < -0.4
    assert abs(dz(relax.result(), far)) < 0.02


def test_flatten_leaves_uniform_curvature_nearly_alone(phantom):
    """Measured over two rings, a smooth sphere is not a spike: real folding survives."""
    from fastfuncstuff.surface.edit import HighlightEdit

    sampler, u, f, topo = phantom
    pial = u * 24.0
    seeds = np.flatnonzero(u[:, 2] > 0.95)
    params = SnapParams(radius=2.0, snap=0.0)
    flat = HighlightEdit(pial, topo, seeds, -0.25, sampler, params, role="pial", flatten=0.6)
    res = flat.result()
    moved = np.linalg.norm(res.positions, axis=1) - 24.0
    seed_moves = moved[np.isin(res.ids, seeds)]
    assert np.all(np.abs(seed_moves + 0.25) < 0.05)


def _crowded_sheet():
    """The sheet with the vertices within 3 mm of the centre crowded toward it (r -> r^2/3)."""
    v, f = _sheet()
    r = np.linalg.norm(v[:, :2], axis=1)
    inner = r < 3.0
    v[inner, :2] *= (r[inner] / 3.0)[:, None]
    return v, f, inner


def test_even_spreads_a_crowded_patch_within_the_surface():
    from fastfuncstuff.surface.edit import HighlightEdit
    from fastfuncstuff.surface.mesh import vertex_areas

    v, f, inner = _crowded_sheet()
    topo = MeshTopology.from_faces(f)
    seeds = np.flatnonzero(np.linalg.norm(v[:, :2], axis=1) < 4.0)
    params = SnapParams(radius=2.0, snap=0.0)
    res = HighlightEdit(v, topo, seeds, 0.0, _flat_sampler(), params, role="pial", even=40).result()
    after = v.copy()
    after[res.ids] = res.positions
    # The centre vertex's area was a tenth of the 0.64 mm^2 every other has.
    before_area = vertex_areas(v, f, len(v))[inner]
    after_area = vertex_areas(after, f, len(v))[inner]
    assert before_area.min() < 0.1
    assert after_area.min() > 0.55
    assert after_area.std() / after_area.mean() < 0.05
    # Within the surface: the sheet stays flat, nothing outside the brush moves,
    # and no triangle turned over.
    assert np.abs(after[:, 2]).max() < 1e-9
    far = np.linalg.norm(v[:, :2], axis=1) > 7.0
    assert np.array_equal(after[far], v[far])
    assert np.all(face_normals(after, f)[:, 2] > 0)


def test_even_off_is_the_plain_push():
    from fastfuncstuff.surface.edit import HighlightEdit

    v, f, _ = _crowded_sheet()
    topo = MeshTopology.from_faces(f)
    seeds = np.flatnonzero(np.linalg.norm(v[:, :2], axis=1) < 4.0)
    params = SnapParams(radius=2.0, snap=0.0)
    plain = HighlightEdit(v, topo, seeds, -0.2, _flat_sampler(), params, role="pial").result()
    both = HighlightEdit(
        v, topo, seeds, -0.2, _flat_sampler(), params, role="pial", even=10
    ).result()
    # The push is the same along the normal; even only adds the slide.
    assert np.allclose(both.positions[:, 2], plain.positions[:, 2], atol=1e-9)
    assert np.abs(both.positions[:, :2] - plain.positions[:, :2]).max() > 0.01
