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
    out = cortical_depth_volumes([s], affine, shape, n_layers=3, area_smooth=0, device=CPU)
    r = radius(affine, shape)
    gm = out.rim == RIM_GM
    # Inscribed polyhedra: stay clear of the facet sagitta at each surface.
    core = gm & (r > R1 + 0.05) & (r < R2 - 0.05)
    assert core.sum() > 1000 and (gm[(r > R1 + 0.05) & (r < R2 - 0.05)]).all()
    np.testing.assert_allclose(out.metric_equidist[core], (r[core] - R1) / (R2 - R1), atol=0.01)
    np.testing.assert_allclose(out.thickness[core], R2 - R1, atol=0.02)
    # Linear-area equivolume is Waehnert's model, not the sphere's cubic law;
    # it still has to land close and sit above equidistant (deep layers thin).
    exact = (r[core] ** 3 - R1**3) / (R2**3 - R1**3)
    np.testing.assert_allclose(out.metric_equivol[core], exact, atol=0.03)
    assert (out.metric_equivol[core] <= out.metric_equidist[core] + 1e-6).all()
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
    out = cortical_depth_volumes([s], affine, shape, area_smooth=0, thick_limit=2.0, device=CPU)
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
