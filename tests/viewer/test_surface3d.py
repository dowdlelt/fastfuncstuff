"""The 3-D surface view's CPU side: texture frames, cameras, picking, layout."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer import surface3d as s3


def test_texture_frame_puts_voxel_centres_at_half_texels():
    aff = np.array([[-2.0, 0, 0, 90], [0, 0, 2.0, -126], [0, -2.0, 0, 72], [0, 0, 0, 1]])
    shape = (10, 12, 8)
    m = s3.texture_from_mm(shape, aff)
    for ijk in [(0, 0, 0), (9, 11, 7), (3, 5, 2)]:
        mm = aff @ [*ijk, 1.0]
        t = (m @ mm)[:3]
        np.testing.assert_allclose(t, (np.array(ijk) + 0.5) / np.array(shape), atol=1e-12)


def test_texture_data_is_k_slowest_so_texture_xyz_is_array_ijk():
    vol = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    data = s3.texture_data(vol)
    assert data.shape == (4, 3, 2)
    assert data[3, 2, 1] == vol[1, 2, 3]
    assert data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("name", list(s3.VIEWS))
def test_views_are_rotations(name):
    r = s3.VIEWS[name]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_lateral_views_put_anterior_toward_the_face():
    anterior = np.array([0.0, 1.0, 0.0])
    # Screen x of the anterior direction: left lateral looks at the left
    # hemisphere from the left, so the frontal pole is on the viewer's left.
    assert (s3.VIEWS["left lateral"] @ anterior)[0] < 0
    assert (s3.VIEWS["right lateral"] @ anterior)[0] > 0
    # Seen from above, the subject's right is on the viewer's right.
    assert (s3.VIEWS["top"] @ [1.0, 0, 0])[0] > 0


def test_ray_through_the_centre_hits_the_target_and_picks_the_right_face():
    cam = s3.Camera()
    cam.target = np.array([5.0, -3.0, 2.0])
    origin, direction = cam.ray(0.0, 0.0, 1.5)
    # The centre ray passes through the target.
    along = cam.target - origin
    np.testing.assert_allclose(np.cross(along, direction), 0.0, atol=1e-6)
    # Two triangles at different heights under the ray: the nearer one wins.
    tri = np.array([[-10, -10, 0], [10, -10, 0], [0, 10, 0]], np.float64)
    verts = np.concatenate([tri + cam.target, tri + cam.target + [0, 0, 5]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    found = s3.pick(origin, direction, verts, faces)
    assert found is not None
    face, bary, _ = found
    assert face == 1  # the camera looks down from +z, so z = +5 is nearer
    assert bary.sum() == pytest.approx(1.0) and np.all(bary >= 0)
    assert s3.pick(origin, -direction, verts, faces) is None


def test_laid_out_hemispheres_sit_left_and_right_with_a_gap():
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(500, 3)) * [30, 60, 40]
    pos = {"lh": blob.astype(np.float32), "rh": blob.astype(np.float32)}
    off = s3.layout_offsets(pos, "inflated")
    lh, rh = pos["lh"] + off["lh"], pos["rh"] + off["rh"]
    assert lh[:, 0].max() <= -s3.HEMI_GAP / 2 + 1e-4
    assert rh[:, 0].min() >= s3.HEMI_GAP / 2 - 1e-4
    # The anatomy is drawn where the head is.
    assert all(np.all(v == 0) for v in s3.layout_offsets(pos, "mid").values())


def test_uniform_block_is_column_major_sized_and_layered():
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    shade = s3.ShadeParams(lo=-2, hi=3, threshold=1.5, has_data=True, outline=s3.OUTLINE_ONLY)
    labels = s3.ShadeParams(has_data=True, labels=True)
    raw = s3.pack_uniforms(
        m,
        np.eye(4),
        morph=0.25,
        depth=(0.1, 0.9),
        samples=3,
        fold_contrast=0.1,
        layers=[
            s3.LayerUniforms(np.eye(4), np.eye(4), shade, lut_row=0),
            s3.LayerUniforms(np.eye(4) * 2, np.eye(4), labels, lut_row=1, palette_row=3),
        ],
        cross=(1, 2, 3, 4),
        cross_rgb=(0.5, 0.5, 0.5),
    )
    assert len(raw) == s3.UNIFORM_BYTES
    first = np.frombuffer(raw[:64], np.float32)
    # GLSL reads column 0 first: m[0,0], m[1,0], m[2,0], m[3,0].
    np.testing.assert_array_equal(first[:4], m[:, 0])
    assert np.frombuffer(raw[128:144], np.float32)[0] == pytest.approx(0.25)
    head = 2 * 64 + 5 * 16

    def layer(k):
        return raw[head + k * s3.LAYER_BYTES : head + (k + 1) * s3.LAYER_BYTES]

    cmap = np.frombuffer(layer(0)[128:144], np.float32)
    modes = np.frombuffer(layer(0)[144:160], np.float32)
    info = np.frombuffer(layer(0)[160:176], np.float32)
    np.testing.assert_allclose(cmap, [-2, 3, 1.5, 1])
    assert modes[3] == 1.0 and info[2] == s3.OUTLINE_ONLY
    assert np.frombuffer(layer(1)[144:160], np.float32)[3] == 2.0  # labels
    assert np.frombuffer(layer(1)[160:176], np.float32)[1] == 3.0  # palette row
    assert np.frombuffer(layer(1)[:4], np.float32)[0] == 2.0  # its own frame
    # Unused slots are off.
    assert not any(layer(2)) and not any(layer(3))


def test_folding_shades_are_bounded_and_signed():
    from fastfuncstuff.io.freesurfer import Hemisphere

    curv = np.array([-0.4, -0.1, 0.0, 0.2, 0.9], np.float32)
    hemi = Hemisphere(
        name="lh",
        faces=np.array([[0, 1, 2], [2, 3, 4]], np.int32),
        states={"white": np.zeros((5, 3), np.float32)},
        tkr_to_scanner=np.eye(4),
        morph={"curv": curv},
    )
    for mode in s3.FOLDING:
        f = s3.folding_values(hemi, mode)
        assert f.shape == (5,) and np.all(np.abs(f) <= 1)
    np.testing.assert_array_equal(s3.folding_values(hemi, "binary"), np.sign(curv))
    assert not s3.folding_values(hemi, "off").any()
    # No sulc file: falls back to curvature rather than to nothing.
    np.testing.assert_array_equal(s3.folding_values(hemi, "sulc"), s3.folding_values(hemi, "curv"))


def test_depth_fractions():
    np.testing.assert_allclose(s3.depth_fractions((0.3, 0.9), 1), [0.3])
    np.testing.assert_allclose(s3.depth_fractions((0.0, 1.0), 3), [0.0, 0.5, 1.0])


def test_equivolume_depth_encloses_the_asked_volume_fraction():
    from scipy.integrate import quad

    for aw, ap in [(1.0, 2.5), (2.0, 0.6), (1.3, 1.3)]:
        for alpha in (0.0, 0.2, 0.5, 0.9, 1.0):
            rho = float(s3.equivolume_fraction(alpha, aw, ap))

            def area(r, aw=aw, ap=ap):
                return aw + (ap - aw) * r

            total = quad(area, 0, 1)[0]
            assert quad(area, 0, rho)[0] / total == pytest.approx(alpha, abs=1e-9)
    # A gyral crown (pial area > white): outer layers are the thin ones, so the
    # mid-volume surface sits nearer pial. A fundus: nearer white.
    assert s3.equivolume_fraction(0.5, 1.0, 3.0) > 0.5
    assert s3.equivolume_fraction(0.5, 3.0, 1.0) < 0.5


def test_vertex_maps_leave_the_medial_wall_transparent():
    from fastfuncstuff.io.freesurfer import Annotation, Hemisphere

    n = 6
    cortex = np.array([True, True, True, True, False, False])
    hemi = Hemisphere(
        name="lh",
        faces=np.array([[0, 1, 2], [3, 4, 5]], np.int32),
        states={"white": np.zeros((n, 3), np.float32)},
        tkr_to_scanner=np.eye(4),
        morph={"thickness": np.array([1.0, 2.5, 4.5, 9.0, 2.0, 2.0], np.float32)},
        cortex=cortex,
    )
    thick = s3.vertex_colors(hemi, "thickness")
    assert thick is not None
    assert (thick[~cortex, 3] == 0).all() and (thick[cortex, 3] == 255).all()
    # Clipped at 4.5 mm: 4.5 and 9 mm get the same top colour.
    np.testing.assert_array_equal(thick[2], thick[3])
    ann = Annotation(
        labels=np.array([0, 1, 1, 2, 1, -1]),
        names=["unknown", "precentral", "insula"],
        rgba=np.array([[1, 1, 1, 255], [200, 0, 0, 255], [0, 200, 0, 255]], np.uint8),
    )
    rgba = s3.vertex_colors(hemi, "annot", ann)
    assert rgba is not None
    assert rgba[0, 3] == 0  # "unknown" names no region
    assert tuple(rgba[1, :3]) == (200, 0, 0) and rgba[1, 3] == 255
    assert rgba[4, 3] == 0  # medial wall
    assert ann.name_at(3) == "insula" and ann.name_at(0) is None and ann.name_at(5) is None


def test_depth_reductions_pick_the_statistic_from_the_same_depth():
    from fastfuncstuff.viewer.surface3d import DEPTH_STATS, reduce_depth

    v = np.array([[0.0, 3.0, -5.0, 1.0, 0.0]])
    s = np.array([[10.0, 11.0, 12.0, 13.0, 14.0]])
    want = {
        "mean": (-0.2, 12.0),
        "max": (3.0, 11.0),
        "min": (-5.0, 12.0),
        "max_abs": (-5.0, 12.0),
        "nzmean": (-1 / 3, 12.0),
    }
    # Stable sort: -5(12) 0(10) 0(14) 1(13) 3(11); the middle is 0 from depth 4.
    want["median"] = (0.0, 14.0)
    for how in DEPTH_STATS:
        got = reduce_depth(v, s, how)
        assert got[0][0] == pytest.approx(want[how][0]), how
        assert got[1][0] == pytest.approx(want[how][1]), how
    zeros = reduce_depth(np.zeros((1, 3)), np.ones((1, 3)), "nzmean")
    assert zeros[0][0] == 0.0 and zeros[1][0] == 0.0
    with pytest.raises(ValueError, match="unknown depth statistic"):
        reduce_depth(v, s, "mode")


def test_the_depth_statistic_reaches_the_uniform_block_by_index():
    from fastfuncstuff.viewer.surface3d import DEPTH_STATS, pack_uniforms

    for k, how in enumerate(DEPTH_STATS):
        raw = pack_uniforms(
            np.eye(4), np.eye(4), morph=1.0, depth=(0.0, 1.0), samples=4, fold_contrast=0.0,
            layers=[], cross=(0, 0, 0, 0), cross_rgb=(1, 1, 1), depth_stat=how,
        )  # fmt: skip
        extra = np.frombuffer(raw[2 * 64 + 4 * 16 : 2 * 64 + 5 * 16], np.float32)
        assert extra[2] == k


def _box(x0, x1):
    """Corners of a hemisphere-ish box: x in [x0, x1], y (A-P) in [-80, 60]."""
    import itertools

    return np.array(list(itertools.product([x0, x1], [-80.0, 60.0], [-30.0, 50.0])))


def _apply(m, p):
    return p @ m[:3, :3].T + m[:3, 3]


def test_closed_hemispheres_only_split_along_x():
    from fastfuncstuff.viewer.surface3d import hemisphere_models

    pos = {"lh": _box(-70.0, -2.0), "rh": _box(2.0, 70.0)}
    m = hemisphere_models(pos, split=10.0)
    assert np.allclose(_apply(m["lh"], pos["lh"]), pos["lh"] - [5.0, 0, 0])
    assert np.allclose(_apply(m["rh"], pos["rh"]), pos["rh"] + [5.0, 0, 0])


@pytest.mark.parametrize("hinge", [180.0, -180.0])
def test_a_full_hinge_lays_the_hemispheres_end_to_end(hinge):
    """+180: noses meet at the front hinge, occipital poles out at the sides;
    -180: the reverse. Either way the medial walls end up facing one way and
    the lateral surfaces the other, and the hinge edge does not move."""
    from fastfuncstuff.viewer.surface3d import hemisphere_models

    pos = {"lh": _box(-70.0, -2.0), "rh": _box(2.0, 70.0)}
    m = hemisphere_models(pos, hinge=hinge)
    hinge_y = 60.0 if hinge > 0 else -80.0
    for h, medial_x, out in (("lh", -2.0, -1.0), ("rh", 2.0, 1.0)):
        moved = _apply(m[h], pos[h])
        # The pivot edge (medial, at the hinge end) stays put.
        edge = (pos[h][:, 0] == medial_x) & (pos[h][:, 1] == hinge_y)
        assert np.allclose(moved[edge], pos[h][edge])
        # The far end has swung out to the hemisphere's own side; the far end
        # of the medial wall lies in line with the hinge.
        far = pos[h][:, 1] != hinge_y
        assert np.all(np.sign(moved[far, 0] - medial_x) == out)
        far_medial = far & (pos[h][:, 0] == medial_x)
        assert np.allclose(moved[far_medial, 1], hinge_y)
        # The medial wall's normal (+x for lh) now points along y, the same
        # way for both hemispheres.
        n = m[h][:3, :3] @ np.array([-out, 0.0, 0.0])
        assert abs(n[1]) == pytest.approx(1.0)
    n_l = m["lh"][:3, :3] @ np.array([1.0, 0, 0])
    n_r = m["rh"][:3, :3] @ np.array([-1.0, 0, 0])
    assert np.allclose(n_l, n_r)


def test_flat_patches_turn_to_meet_occipital_to_occipital():
    """Each patch file comes out of flattening at its own angle, either one possibly
    mirrored; laid out, both read as lateral views -- superior up, posterior toward
    the other hemisphere -- whatever turn the file had."""
    from fastfuncstuff.io.freesurfer import FlatPatch, Hemisphere

    rng = np.random.default_rng(0)
    yz = rng.uniform(-50, 50, (300, 2))  # anterior-posterior, superior-inferior
    white = np.c_[rng.normal(0, 5, 300), yz]

    def hemi(name, angle, mirror):
        c, s = np.cos(angle), np.sin(angle)
        flat = yz @ np.array([[c, -s], [s, c]]) * [1.0, -1.0 if mirror else 1.0] + [17.0, -9.0]
        coords = np.c_[flat, np.zeros(300)]
        patch = FlatPatch("flat", coords, np.ones(300, bool), np.zeros(300, bool))
        return Hemisphere(name, np.zeros((0, 3), np.int64), {"white": white}, np.eye(4),
                          patches={"flat": patch})  # fmt: skip

    for name, angle, mirror in (("lh", 1.1, False), ("rh", -2.3, True), ("lh", 0.4, True)):
        xy = s3.shape_positions(hemi(name, angle, mirror), "flat")[:, :2]
        toward_rh = 1.0 if name == "lh" else -1.0
        # screen x follows posterior toward the other hemisphere, screen y superior
        np.testing.assert_allclose(xy[:, 0], -toward_rh * (yz[:, 0] - yz[:, 0].mean()), atol=1e-4)
        np.testing.assert_allclose(xy[:, 1], yz[:, 1] - yz[:, 1].mean(), atol=1e-4)
