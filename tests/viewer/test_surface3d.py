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


def test_uniform_block_is_column_major_and_sized():
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    raw = s3.pack_uniforms(
        m,
        np.eye(4),
        np.eye(4),
        np.eye(4),
        morph=0.25,
        depth=(0.1, 0.9),
        samples=3,
        curv_contrast=0.1,
        shade=s3.ShadeParams(),
        cross=(1, 2, 3, 4),
        cross_rgb=(0.5, 0.5, 0.5),
    )
    assert len(raw) == s3.UNIFORM_BYTES
    first = np.frombuffer(raw[:64], np.float32)
    # GLSL reads column 0 first: m[0,0], m[1,0], m[2,0], m[3,0].
    np.testing.assert_array_equal(first[:4], m[:, 0])
    morph = np.frombuffer(raw[256:272], np.float32)
    assert morph[0] == pytest.approx(0.25)


def test_depth_fractions():
    np.testing.assert_allclose(s3.depth_fractions((0.3, 0.9), 1), [0.3])
    np.testing.assert_allclose(s3.depth_fractions((0.0, 1.0), 3), [0.0, 0.5, 1.0])
