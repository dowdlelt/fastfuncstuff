import numpy as np
import pytest
import torch

from fastfuncstuff.surface.voxelize import MeshDistance, _closest_on_triangles, winding_number

CPU = torch.device("cpu")


def box_mesh(lo, hi):
    """Closed axis-aligned box, outward winding, split along face diagonals."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=np.float64
    )
    # Vertex index = x + 2y + 4z (bits).
    quads = [
        (0, 2, 3, 1),  # z0, normal -z
        (4, 5, 7, 6),  # z1, +z
        (0, 1, 5, 4),  # y0, -y
        (2, 6, 7, 3),  # y1, +y
        (0, 4, 6, 2),  # x0, -x
        (1, 3, 7, 5),  # x1, +x
    ]
    f = []
    for a, b, c, d in quads:
        f += [(a, b, c), (a, c, d)]
    return v, np.array(f, np.int64)


def icosphere(n_sub=3, radius=1.0, centre=(0, 0, 0)):
    t = (1 + 5**0.5) / 2
    v = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    f = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(n_sub):
        cache, nf = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = v[i] + v[j]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v) * radius + np.asarray(centre, float), np.array(f, np.int64)


def test_box_on_grid_lines_is_watertight():
    # Every vertex sits on a column centre, so columns run exactly through the
    # face diagonals, edges and corners -- the cases the top-left rule exists for.
    v, f = box_mesh((2, 2, 2.5), (6, 7, 6.5))
    w = winding_number(v, f, np.eye(4), (10, 10, 10), CPU)
    assert set(np.unique(w)) <= {0, 1}
    assert (w[3:6, 3:7, 3:7] == 1).all()
    assert w[:, :, :3].sum() == 0 and w[:, :, 7:].sum() == 0
    assert w[:2].sum() == 0 and w[7:].sum() == 0
    # A boundary column is in or out as a whole, never half of it.
    col = w[2, 4]
    assert col[3:7].min() == col[3:7].max()


def test_sphere_matches_analytic_and_mirrored_affine():
    v, f = icosphere(4, radius=7.3, centre=(10.2, 9.7, 10.4))
    shape = (21, 20, 22)
    xyz = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    w = winding_number(v, f, np.eye(4), shape, CPU)
    # The mesh is inscribed: inside it by more than the sagitta, the answer is known.
    r = np.linalg.norm(xyz - np.array([10.2, 9.7, 10.4]), axis=-1)
    assert (w[r < 7.2] == 1).all() and (w[r > 7.31] == 0).all()

    # Same world geometry, x-mirrored grid (negative determinant).
    aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    aff[0, 3] = shape[0] - 1
    wm = winding_number(v, f, aff, shape, CPU)
    np.testing.assert_array_equal(wm[::-1], w)


def test_self_overlap_counts_two():
    v1, f1 = box_mesh((1, 1, 1.5), (5, 5, 5.5))
    v2, f2 = box_mesh((3, 3, 3.5), (8, 8, 7.5))
    v = np.concatenate([v1, v2])
    f = np.concatenate([f1, f2 + len(v1)])
    w = winding_number(v, f, np.eye(4), (10, 10, 10), CPU)
    assert w[4, 4, 4] == 2 and w[2, 2, 2] == 1 and w[7, 7, 7] == 1 and w[0, 0, 0] == 0


def test_closest_point_regions():
    a = torch.tensor([0.0, 0, 0])
    b = torch.tensor([1.0, 0, 0])
    c = torch.tensor([0.0, 1, 0])
    cases = {
        (0.2, 0.2, 1.0): (1.0, (0.6, 0.2, 0.2)),  # face interior, above
        (-1.0, -1.0, 0.0): (2.0, (1, 0, 0)),  # vertex a
        (2.0, -1.0, 0.0): (2.0, (0, 1, 0)),  # vertex b
        (0.5, -1.0, 0.0): (1.0, (0.5, 0.5, 0)),  # edge ab
        (1.0, 1.0, 0.0): (0.5, (0, 0.5, 0.5)),  # edge bc
        (-1.0, 0.5, 0.0): (1.0, (0.5, 0, 0.5)),  # edge ca
    }
    for p, (d2, bary) in cases.items():
        got_d2, got_w = _closest_on_triangles(torch.tensor(p), a, b, c)
        assert abs(float(got_d2) - d2) < 1e-6, p
        np.testing.assert_allclose(got_w.numpy(), bary, atol=1e-6, err_msg=str(p))


def test_mesh_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    v, f = icosphere(3, radius=5.0)
    v = v * (1 + 0.05 * rng.standard_normal(v.shape[0]))[:, None]  # lumpy
    # Points in a band around the surface, as ribbon voxels are -- not deep
    # inside, where every face is about equally far and nearest-centroid
    # candidates are a guess.
    d = rng.standard_normal((3000, 3))
    pts = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(4, 6.5, (3000, 1))
    got = MeshDistance(v, f)(pts, CPU)
    p = torch.as_tensor(pts)[:, None, :]
    tri = torch.as_tensor(v)[torch.as_tensor(f)]
    d2, _ = _closest_on_triangles(p, tri[None, :, 0], tri[None, :, 1], tri[None, :, 2])
    ref = d2.min(1).values.sqrt().numpy()
    np.testing.assert_allclose(got.distance, ref, atol=1e-4)
    # The barycentrics reproduce the closest point.
    q = got.interpolate(f, v[:, 0]), got.interpolate(f, v[:, 1]), got.interpolate(f, v[:, 2])
    np.testing.assert_allclose(np.linalg.norm(np.stack(q, 1) - pts, axis=1), ref, atol=1e-4)


@pytest.mark.gpu
def test_brick_search_is_exact_on_cuda():
    """The CUDA brick search against an all-faces scan, on a folded mesh.

    Exact by its bound, so it must match brute force everywhere -- including
    the points near the inside of a fold, where per-brick *candidates* without
    a bound picked the wrong bank.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    from fastfuncstuff.surface import voxelize

    if voxelize.brick_nearest is None:
        pytest.skip("Triton")
    dev = torch.device("cuda")
    u, f = icosphere(4)
    x, y, z = u.T
    v = u * (6 + 0.8 * np.sin(5 * np.arctan2(y, x)) * np.sin(4 * np.arccos(z)))[:, None]
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(*[np.arange(-7.5, 7.5, 0.25)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(g, axis=1)
    pts = g[(r > 4.5) & (r < 7.5)] + rng.uniform(-0.01, 0.01, (1, 3))
    got = MeshDistance(v, f)(pts, dev, brick_mm=1.0)
    tri = torch.as_tensor(v, device=dev)[torch.as_tensor(f, device=dev)]
    ref = []
    for s in range(0, len(pts), 512):
        q = torch.as_tensor(pts[s : s + 512], device=dev)[:, None]
        d2, _ = _closest_on_triangles(q, tri[None, :, 0], tri[None, :, 1], tri[None, :, 2])
        ref.append(d2.min(1).values.sqrt().cpu().numpy())
    ref = np.concatenate(ref)
    np.testing.assert_allclose(got.distance, ref, atol=2e-5)
    q = np.stack([got.interpolate(f, v[:, k]) for k in range(3)], 1)
    np.testing.assert_allclose(np.linalg.norm(q - pts, axis=1), ref, atol=1e-4)
