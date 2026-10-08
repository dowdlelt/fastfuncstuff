"""build_sampling: which points a surface reads, and how reads fold onto vertices."""

from __future__ import annotations

import numpy as np

from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.projection import _lattice, build_sampling, depth_surfaces


def _sheet(n: int, spacing: float, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A flat n x n vertex grid, each square split along the same diagonal."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * spacing, jj.ravel() * spacing, np.full(n * n, z)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v.astype(np.float64), f.astype(np.int64)


def test_lattice_tiles_the_triangle_into_equal_thirds():
    for level in range(1, 9):
        bary, share = _lattice(level)
        n = 3 if level == 1 else level * level  # level 1: one read per corner's third
        assert bary.shape[0] == n
        np.testing.assert_allclose(bary.sum(1), 1.0)
        assert (bary > 0).all()  # centroids, strictly inside
        np.testing.assert_allclose(share.sum(0), n / 3.0)
        # every read belongs to the corner it is nearest (none to a far corner)
        owner = share.argmax(1)
        assert (bary[np.arange(n), owner] >= bary.max(1) - 1e-12).all()


def test_small_triangles_read_each_corner_third_not_the_centroid():
    """Level 1 used to read only the centroid, shared three ways: a vertex then averaged
    a ring of centroids and never itself. Its third's centroid is 11/18 of the way in."""
    bary, share = _lattice(1)
    np.testing.assert_allclose(np.diag(bary), 11 / 18)
    np.testing.assert_array_equal(share, np.eye(3))


def test_point_mode_reads_each_vertex_once():
    w, f = _sheet(6, 2.0)
    smp = build_sampling(w, w + [0, 0, 2.5], f, fractions=(0.2, 0.5))
    assert smp.points.shape == (2 * 36, 3)
    reads = np.arange(72, dtype=np.float32)
    np.testing.assert_array_equal(smp.fold(reads), reads.reshape(2, 36))
    np.testing.assert_allclose(smp.points[36:, 2], 1.25)


def test_footprint_patch_is_the_vertex_area_and_centred_on_the_vertex():
    w, f = _sheet(9, 3.0)
    smp = build_sampling(w, w + [0, 0, 2.0], f, voxel_face=0.64)
    # every triangle (area 4.5 mm^2) is cut to sub-triangles no bigger than a voxel face
    assert smp.points.shape[0] >= f.shape[0] * 4.5 / 0.64
    np.testing.assert_allclose(np.asarray(smp.operator.sum(1)).ravel(), 1.0, atol=1e-6)
    # each vertex owns a third of each of its triangles: the patch vertex_areas measures
    owned = np.zeros(len(w))
    tri = 0.5 * 3.0 * 3.0
    np.add.at(owned, f.ravel(), tri / 3.0)
    np.testing.assert_allclose(owned, vertex_areas(w, f))
    reads_per_vertex = np.diff(smp.operator.indptr)
    assert (reads_per_vertex[owned > 0] >= np.floor(owned / 0.64)).all()
    # A linear field averages to its value at the vertex wherever the patch is
    # symmetric (interior vertices of this grid).
    coef = np.array([0.3, -0.7, 0.0])
    got = smp.fold(smp.points @ coef + 5.0)[0].reshape(9, 9)
    want = (w @ coef + 5.0).reshape(9, 9)
    np.testing.assert_allclose(got[1:-1, 1:-1], want[1:-1, 1:-1], atol=1e-4)


def test_footprint_reads_every_voxel_where_a_point_aliases():
    # A voxel-scale checkerboard under a mesh four voxels coarse: point sampling lands
    # on whatever square each vertex hits; the footprint averages it out.
    vox = 0.8
    w, f = _sheet(12, 4 * vox + 0.13)  # off-grid spacing so points don't align

    def checker(p):
        return np.sign(np.sin(np.pi * p[:, 0] / vox) * np.sin(np.pi * p[:, 1] / vox))

    pial = w + [0, 0, 2.0]
    point = build_sampling(w, pial, f)
    foot = build_sampling(w, pial, f, voxel_face=vox * vox / 4)
    interior = np.zeros((12, 12), bool)
    interior[1:-1, 1:-1] = True
    p_vals = point.fold(checker(point.points))[0][interior.ravel()]
    f_vals = foot.fold(checker(foot.points))[0][interior.ravel()]
    assert np.abs(p_vals).mean() > 0.9  # aliased: almost every vertex reads +-1
    assert np.abs(f_vals).mean() < 0.15  # the patch averages the pattern out


def test_equivolume_mid_surface_sits_nearer_pial_in_a_crown():
    # pial area > white area (a crown): the outer layers are thin, mid-volume is outer.
    w, f = _sheet(7, 2.0)
    pial = w * [1.4, 1.4, 1.0] + [0, 0, 3.0]
    mid = depth_surfaces(w, pial, f, [0.5])[0]
    lin = depth_surfaces(w, pial, f, [0.5], equivolume=False)[0]
    assert (mid[:, 2] > lin[:, 2] + 0.1).all()


def test_coverage_counts_reads_outside_the_epi_in_any_frame():
    w, f = _sheet(5, 2.0)
    smp = build_sampling(w, w + [0, 0, 2.0], f, voxel_face=0.5)
    reads = np.ones((3, smp.points.shape[0]), np.float32)
    out = smp.points[:, 0] > 4.0
    reads[1, out] = 0.0  # left the FOV in one frame
    cov = smp.coverage(reads)[0].reshape(5, 5)
    assert cov[0, 0] == 1.0 and cov[-1, 0] == 0.0
    assert 0.0 < cov[2, 0] < 1.0  # the patch straddles the edge
