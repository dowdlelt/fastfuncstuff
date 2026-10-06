import numpy as np
import torch

from fastfuncstuff.surface.volume_depth import (
    RIM_CSF,
    RIM_GM,
    RIM_WM,
    RibbonSurfaces,
    cortical_depth_volumes,
    crop_to_points,
    regrid,
)
from tests.test_surface_voxelize import icosphere

CPU = torch.device("cpu")
R1, R2 = 6.0, 9.0
CENTRE = np.array([12.3, 11.8, 12.1])


def shell(cortex=None, step=0.25):
    v, f = icosphere(5, centre=(0, 0, 0))
    s = RibbonSurfaces("lh", v * R1 + CENTRE, v * R2 + CENTRE, f, cortex)
    n = int(25 / step)
    affine = np.diag([step, step, step, 1.0])
    return s, v, affine, (n, n, n)


def radius(affine, shape):
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    return np.linalg.norm(ijk @ affine[:3, :3].T + affine[:3, 3] - CENTRE, axis=-1)


def test_concentric_spheres_match_closed_form():
    s, _, affine, shape = shell()
    out = cortical_depth_volumes([s], affine, shape, n_layers=3, device=CPU)
    r = radius(affine, shape)
    gm = out.rim == RIM_GM
    # Inscribed polyhedra: stay clear of the facet sagitta at each surface.
    core = gm & (r > R1 + 0.05) & (r < R2 - 0.05)
    assert core.sum() > 1000 and (gm[(r > R1 + 0.05) & (r < R2 - 0.05)]).all()
    np.testing.assert_allclose(out.metric_equidist[core], (r[core] - R1) / (R2 - R1), atol=0.01)
    np.testing.assert_allclose(out.thickness[core], R2 - R1, atol=0.02)
    # Volume-quantile equivolume is the sphere's cubic law, up to counting
    # noise (unbiased: the worst single voxel shrinks as columns grow).
    exact = (r[core] ** 3 - R1**3) / (R2**3 - R1**3)
    err = np.abs(out.metric_equivol[core] - exact)
    assert err.mean() < 0.005 and np.percentile(err, 99) < 0.02
    assert abs((out.metric_equivol[core] - exact).mean()) < 0.001
    # Borders: one voxel thick, WM inside, CSF outside, nothing else labelled.
    assert set(np.unique(out.rim)) == {0, RIM_CSF, RIM_WM, RIM_GM}
    assert (r[out.rim == RIM_WM] < R1 + 0.01).all() and (r[out.rim == RIM_CSF] > R2 - 0.01).all()
    assert out.rim[r < R1 - 0.5].max() == 0 and out.rim[r > R2 + 0.5].max() == 0
    # Layers are the binned metric; borders carry no metric/layer.
    lay = out.layers_equidist
    assert set(np.unique(lay[gm])) == {1, 2, 3} and lay[~gm].max() == 0
    assert (lay[gm & (out.metric_equidist < 1 / 3 - 0.01)] == 1).all()
    assert out.metric_equidist[~gm].max() == 0
    # midGM is a thin sheet at r ~ (R1 + R2) / 2.
    mid = out.mid_gm_equidist.astype(bool)
    assert np.abs(r[mid] - (R1 + R2) / 2).max() < 0.25
    assert out.n_thick == 0


def test_medial_wall_and_thickness_warning():
    s, v, affine, shape = shell(step=0.5)
    s.cortex = v[:, 0] < 0.5  # an x cap is "medial wall"
    out = cortical_depth_volumes([s], affine, shape, thick_limit=2.0, device=CPU)
    x = (np.arange(shape[0]) * 0.5 - CENTRE[0])[:, None, None]
    r = radius(affine, shape)
    ribbon = (r > R1 + 0.3) & (r < R2 - 0.3)
    assert out.n_medial > 0
    assert (out.rim[ribbon & (x / r > 0.6)] == 0).all()
    assert (out.rim[ribbon & (x / r < 0.4)] == RIM_GM).all()
    # Every remaining GM voxel is 3 mm "thick" here, above the 2 mm limit.
    assert out.n_thick == out.n_gm > 0
    clusters = out.thick_clusters(affine, top=None)
    assert sum(n for n, _ in clusters) == out.n_thick
    # One shell minus a cap: its centroid is pulled off-centre away from the cap.
    assert clusters[0][1][0] < CENTRE[0]


def test_regrid_and_crop_keep_world_positions():
    affine = np.array([[-0.7, 0, 0, 90], [0, 0.7, 0, -100], [0, 0, 0.7, -80], [0, 0, 0, 1.0]])
    a2, shape2 = regrid(affine, (100, 120, 80), 0.35)
    assert shape2 == (200, 240, 160)
    np.testing.assert_allclose(np.abs(np.diag(a2)[:3]), 0.35)
    pts = np.array([[60.0, -50, -40], [40, -20, -30]])
    a3, shape3, sl = crop_to_points(a2, shape2, pts, pad_mm=1.0)
    # The crop's first voxel is the original's voxel at the slice starts.
    start = np.array([s.start for s in sl], float)
    np.testing.assert_allclose(a3[:3, 3], a2[:3, :3] @ start + a2[:3, 3])
    inv = np.linalg.inv(a3)
    ijk = pts @ inv[:3, :3].T + inv[:3, 3]
    assert (ijk >= 0).all() and (ijk <= np.array(shape3) - 1).all()


def test_equivolume_survives_freesurfer_sized_vertex_slide():
    """Folded cortex with Bok ground truth; pial vertices slid ~1 mm sideways.

    FreeSurfer's pial vertex sits a median 1 mm (p90 2.4 mm) off its white
    partner's normal. Anything that pairs white and pial vertices to define a
    column -- the vertex-area method this replaced -- did worse than plain
    equidistant here. Geometry alone must not care.
    """
    from fastfuncstuff.surface.mesh import MeshTopology, vertex_areas, vertex_normals
    from fastfuncstuff.surface.voxelize import MeshDistance

    R, T = 12.0, 1.5
    u, f = icosphere(5)
    x, y, z = u.T
    bump = np.sin(5.0 * np.arctan2(y, x)) * np.sin(4.0 * np.arccos(np.clip(z, -1, 1)))
    white = u * (R + 0.8 * bump)[:, None]
    n = vertex_normals(white, MeshTopology.from_faces(f, len(white)))
    pial = white + T * n  # parallel surface: columns are the normals
    # Truth: per-vertex volume below depth t, integrating the offset meshes' areas.
    ts = np.linspace(0, T, 16)
    area = np.stack([vertex_areas(white + t * n, f, len(white)) for t in ts], 1)
    cum = np.concatenate(
        [np.zeros((len(white), 1)), np.cumsum((area[:, 1:] + area[:, :-1]) / 2 * np.diff(ts), 1)], 1
    )
    cum /= cum[:, -1:]
    # Same pial geometry, vertices slid tangentially ~1 mm and put back on it.
    rng = np.random.default_rng(0)
    slide = np.sin(pial @ rng.normal(size=(3, 3)) * 0.5) * 1.2
    slide -= (slide * n).sum(1, keepdims=True) * n
    foot = MeshDistance(pial, f)(pial + slide, CPU)
    slid = np.stack([foot.interpolate(f, pial[:, k]) for k in range(3)], 1)
    assert np.median(np.linalg.norm(slid - pial, axis=1)) > 0.5

    step = 0.3
    lo = slid.min(0) - 1
    shape = tuple(int(v) for v in np.ceil((slid.max(0) + 1 - lo) / step) + 1)
    affine = np.diag([step, step, step, 1.0])
    affine[:3, 3] = lo
    out = cortical_depth_volumes([RibbonSurfaces("lh", white, slid, f)], affine, shape, device=CPU)
    ijk = np.stack(np.unravel_index(out.gm_idx, shape), 1)
    pts = ijk @ affine[:3, :3].T + affine[:3, 3]
    cw = MeshDistance(white, f)(pts, CPU)
    t = np.clip(cw.distance / T, 0, 1) * (len(ts) - 1)
    i0 = np.minimum(t.astype(int), len(ts) - 2)
    fr = t - i0
    fv = f[cw.face]
    truth = sum(
        cw.bary[:, k] * (cum[fv[:, k], i0] * (1 - fr) + cum[fv[:, k], i0 + 1] * fr)
        for k in range(3)
    )
    e_equidist = np.abs(out.rho - truth).mean()
    e_equivol = np.abs(out.equivol - truth).mean()
    assert e_equivol < 0.5 * e_equidist, (e_equivol, e_equidist)
