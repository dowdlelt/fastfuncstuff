"""On-the-fly columns: the cover, and reads that equal reading each column as an ROI.

The referee is ``processing.roi_depth``: a column summed from vertex x depth reads
must give what ``build_depth_roi_set`` gives for the same vertices drawn as a label
-- values, volumes and depth PSF -- because it is the same cells summed in another
order.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.columns import (
    column_curves,
    make_columns,
    responsive_vertices,
    trim_to_coverage,
    vertex_depth_reads,
)
from fastfuncstuff.processing.roi_depth import (
    DepthOptions,
    build_depth_roi_set,
    depth_sampling_matrices,
)
from fastfuncstuff.surface.mesh import MeshTopology, geodesic_ball
from fastfuncstuff.surface.volume_depth import RibbonSurfaces
from tests.test_surface_voxelize import icosphere

RW, RP = 15.0, 21.0
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def sphere():
    v, f = icosphere(4)
    surf = RibbonSurfaces("lh", v * RW, v * RP, f.astype(np.int64))
    n = 26
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    r = np.linalg.norm(xyz, axis=1)
    # Radius, and a second "time point" that varies along the surface too.
    data = np.stack([r, r + xyz[:, 0]], 1)
    return surf, affine, (n, n, n), data, v


def _label(tmp_path, ids, name="lh.col.label"):
    path = tmp_path / name
    rows = "\n".join(f"{i}  0 0 0 1" for i in ids)
    path.write_text(f"#!ascii label\n{len(ids)}\n{rows}\n")
    return str(path)


def test_cover_reaches_every_responsive_vertex(sphere):
    surf, *_, v = sphere
    resp = v[:, 2] > 0.3
    cols = make_columns([surf], [resp], radius_mm=6.0, spacing_mm=3.0, seed=1)
    mid = 0.5 * (surf.white + surf.pial)
    topo = MeshTopology.from_faces(surf.faces, len(mid))
    reach = np.zeros(len(mid), bool)
    for c in cols.centre_vertex:
        near, _ = geodesic_ball(mid, topo, int(c), 3.0)
        reach[near] = True
    assert reach[resp].all()
    assert resp[cols.centre_vertex].all()
    # Each column holds its centre, and overlap means most responsive vertices sit
    # in several columns.
    member_keys = cols.vertex_id
    for i, c in enumerate(cols.centre_vertex[:5]):
        row = cols.members[i].indices
        assert c in member_keys[row]
    count = cols.coverage_count()
    inner = np.isin(cols.vertex_id, np.flatnonzero(v[:, 2] > 0.6))
    assert np.median(count[inner]) >= 3
    # Finer spacing, more columns: the data's area and the spacing set the number.
    finer = make_columns([surf], [resp], radius_mm=6.0, spacing_mm=1.5, seed=1)
    assert finer.n_columns > 2 * cols.n_columns


@pytest.mark.parametrize("method", ["equidistant", "equivolume"])
def test_column_read_equals_reading_it_as_an_roi(sphere, tmp_path, method):
    surf, affine, shape, data, v = sphere
    resp = np.zeros(len(v), bool)
    resp[[int(np.argmax(v[:, 2])), int(np.argmax(v[:, 0]))]] = True
    cols = make_columns([surf], [resp], radius_mm=7.0, spacing_mm=0.5)
    assert cols.n_columns == 2
    opts = DepthOptions(n_depths=4, method=method, dxyz=1.0)
    reads = vertex_depth_reads([surf], cols, affine, shape, [4], opts, psf=False, device=CPU)
    cc = column_curves(reads, cols, data[reads.voxels], 4)
    for i in range(2):
        ids = cols.vertex_id[cols.members[i].indices]
        rs = build_depth_roi_set(_label(tmp_path, ids), [surf], affine, shape, opts=opts,
                                 device=CPU)  # fmt: skip
        sel = torch.from_numpy(data[rs.rows.reshape(-1)])
        want = rs.collapse(sel, device=CPU).numpy()
        # The ROI collapse returns float32; measured agreement is its rounding (5e-7).
        np.testing.assert_allclose(cc.values[i], want, rtol=2e-6)
        np.testing.assert_allclose(cc.coverage[i], 1.0)


def test_column_psf_equals_the_roi_psf(sphere, tmp_path):
    surf, affine, shape, data, v = sphere
    resp = np.zeros(len(v), bool)
    resp[int(np.argmax(v[:, 2]))] = True
    cols = make_columns([surf], [resp], radius_mm=7.0, spacing_mm=0.5)
    opts = DepthOptions(n_depths=5, method="equivolume", dxyz=1.0)
    reads = vertex_depth_reads([surf], cols, affine, shape, [5], opts, device=CPU)
    cc = column_curves(reads, cols, data[reads.voxels], 5)
    ids = cols.vertex_id[cols.members[0].indices]
    ((_, _, m),) = depth_sampling_matrices(_label(tmp_path, ids), [surf], affine, shape, opts,
                                            device=CPU)  # fmt: skip
    assert cc.psf is not None
    np.testing.assert_allclose(cc.psf[0], m, rtol=1e-9, atol=1e-12)


def test_mask_shows_up_as_coverage(sphere):
    surf, affine, shape, data, v = sphere
    resp = np.zeros(len(v), bool)
    resp[int(np.argmax(v[:, 2]))] = True
    cols = make_columns([surf], [resp], radius_mm=7.0, spacing_mm=0.5)
    # Cut the outer half of the ribbon out of the analysis mask.
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(ijk @ affine[:3, :3].T + affine[:3, 3], axis=1)
    mask = (r < 0.5 * (RW + RP)).reshape(shape)
    opts = DepthOptions(n_depths=4, dxyz=1.0)
    reads = vertex_depth_reads([surf], cols, affine, shape, [4], opts, mask=mask, psf=False,
                               device=CPU)  # fmt: skip
    cc = column_curves(reads, cols, data[reads.voxels], 4)
    assert cc.coverage[0, 0] > 0.99 and cc.coverage[0, -1] < 0.5


def test_a_region_cut_by_the_slab_keeps_its_covered_part(sphere):
    surf, affine, shape, data, v = sphere
    resp = np.zeros(len(v), bool)
    resp[int(np.argmax(v[:, 2]))] = True
    cols = make_columns([surf], [resp], radius_mm=9.0, spacing_mm=0.5)
    # A slab boundary through the region: x >= 1 mm out of the FOV.
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    x = (ijk @ affine[:3, :3].T + affine[:3, 3])[:, 0]
    mask = (x < 1.0).reshape(shape)
    opts = DepthOptions(n_depths=4, dxyz=1.0)
    reads = vertex_depth_reads([surf], cols, affine, shape, [4], opts, mask=mask, psf=False,
                               device=CPU)  # fmt: skip
    assert column_curves(reads, cols, data[reads.voxels], 4).coverage.min() < 0.9
    trimmed, kept = trim_to_coverage(cols, reads, 0.9)
    cc = column_curves(reads, trimmed, data[reads.voxels], 4)
    assert cc.coverage.min() >= 0.9
    assert 0.3 < kept[0] < 0.8
    ids = trimmed.vertex_id[trimmed.members[0].indices]
    assert (v[ids, 0] * RP < 1.0).all()


def test_responsive_vertices_reads_a_statistic_along_the_column(sphere, tmp_path):
    surf, affine, shape, data, v = sphere
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    # A "t map" that is large only in the outer ribbon of the x > 0 half.
    r = np.linalg.norm(xyz, axis=-1)
    stat = np.where((xyz[..., 0] > 0) & (r > 19.0), 6.0, 0.5)
    path = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(stat.astype(np.float32), affine), path)
    (hit,) = responsive_vertices(str(path), [surf], threshold=4.0)
    assert hit[v[:, 0] > 0.2].mean() > 0.95
    assert not hit[v[:, 0] < -0.2].any()


def test_a_cluster_filter_drops_lone_responsive_voxels(sphere, tmp_path):
    """One suprathreshold voxel is enough to place a column, so it must be filterable.

    The bug this guards: a lone false positive at |z| >= 3 licensed a whole
    -column_radius neighbourhood fit.
    """
    surf, affine, shape, data, v = sphere
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    r = np.linalg.norm(xyz, axis=-1)
    # A real patch on x > 0, plus ONE isolated voxel in the ribbon on the far side.
    stat = np.where((xyz[..., 0] > 5.0) & (r > 19.0), 6.0, 0.5)
    lone = np.argmin(np.where(xyz[..., 0] < -5.0, np.abs(r - 19.5), 1e9).reshape(-1))
    lone = np.unravel_index(lone, r.shape)
    stat[lone] = 6.0
    path = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(stat.astype(np.float32), affine), path)

    (unfiltered,) = responsive_vertices(str(path), [surf], threshold=4.0)
    notes: list[str] = []
    (filtered,) = responsive_vertices(
        str(path), [surf], threshold=4.0, cluster_size=5, cluster_nn=1, notes=notes
    )
    # The lone voxel reached vertices; the filter removed exactly those and kept the patch.
    assert unfiltered[v[:, 0] < -0.2].any()
    assert not filtered[v[:, 0] < -0.2].any()
    assert filtered[v[:, 0] > 0.5].mean() > 0.95
    assert any("cluster filter (NN1" in note for note in notes)

    # NN3 is never stricter than NN1: more neighbours can only merge clusters.
    (nn3,) = responsive_vertices(str(path), [surf], threshold=4.0, cluster_size=5, cluster_nn=3)
    assert nn3.sum() >= filtered.sum()
    # size 1 is a no-op, not a filter.
    (one,) = responsive_vertices(str(path), [surf], threshold=4.0, cluster_size=1)
    assert np.array_equal(one, unfiltered)


def test_responsive_reads_a_surface_map_and_clusters_on_the_mesh(sphere, tmp_path):
    """A .gii -responsive needs no sampling, and its clusters follow mesh edges.

    The surface route must agree with the volume route on the same pattern, which is
    the only check that the two are the same question asked two ways.
    """
    from fastfuncstuff.io.gifti import save_gifti_data

    surf, affine, shape, data, v = sphere
    # Mark the x > 0 cap on the surface directly, plus one isolated vertex opposite.
    stat = np.where(v[:, 0] > 0.3, 6.0, 0.5)
    lone = int(np.argmin(v[:, 0]))
    stat[lone] = 6.0
    path = tmp_path / "resp.lh.func.gii"
    save_gifti_data(str(path), stat.astype(np.float32), {}, time_series=False)

    notes: list[str] = []
    (unfiltered,) = responsive_vertices(str(path), [surf], threshold=4.0, notes=notes)
    assert unfiltered[lone]
    assert unfiltered[v[:, 0] > 0.5].all()
    assert any("surface maps" in n for n in notes)

    (filtered,) = responsive_vertices(str(path), [surf], threshold=4.0, cluster_size=5)
    assert not filtered[lone]
    assert filtered[v[:, 0] > 0.5].all()

    # Same pattern through the volume route: the two must mark the same cap.
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    r = np.linalg.norm(xyz, axis=-1)
    unit = xyz / np.maximum(r[..., None], 1e-9)
    vol = np.where((unit[..., 0] > 0.3) & (r > 17.0) & (r < 22.0), 6.0, 0.5)
    vpath = tmp_path / "resp.nii.gz"
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), vpath)
    (from_vol,) = responsive_vertices(str(vpath), [surf], threshold=4.0)
    core = v[:, 0] > 0.5
    assert from_vol[core].all()
    assert np.mean(from_vol == unfiltered) > 0.9

    # A list of one is the same as the bare path; a .gii that fits no surface is refused.
    (as_list,) = responsive_vertices([str(path)], [surf], threshold=4.0)
    assert np.array_equal(as_list, unfiltered)
    bad = tmp_path / "wrong.lh.func.gii"
    save_gifti_data(str(bad), np.zeros(7, np.float32), {}, time_series=False)
    with pytest.raises(ValueError, match="match no surface"):
        responsive_vertices(str(bad), [surf], threshold=4.0)
    with pytest.raises(ValueError, match="mixed surface and volume"):
        responsive_vertices([str(path), str(vpath)], [surf], threshold=4.0)


def test_regions_from_map_finds_each_surface_cluster(sphere, tmp_path):
    from fastfuncstuff.laminar.columns import regions_from_map

    surf, affine, shape, data, v = sphere
    n = shape[0]
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    unit = xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-9)
    t = np.zeros(shape)
    t[unit[..., 2] > 0.85] = 5.0  # a cap at the north pole
    t[unit[..., 0] > 0.85] = 4.0  # another on +x, well apart
    t[unit[..., 1] > 0.995] = 6.0  # a speck: too small to keep
    path = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(t.astype(np.float32), affine), path)
    regions = regions_from_map([surf], str(path), 3.1, min_area_mm2=30.0)
    assert regions.n_columns == 2
    centres = v[regions.centre_vertex]
    assert sorted(np.argmax(np.abs(centres), axis=1).tolist()) == [0, 2]
    # Members are each cap's vertices (claimed at any depth, closed along the mesh).
    north = regions.vertex_id[regions.members[int(np.argmax(centres[:, 2]))].indices]
    assert (v[north, 2] > 0.75).all() and len(north) > 20


def test_regions_from_atlas_keeps_names_and_hemispheres(sphere, tmp_path):
    from fastfuncstuff.laminar.columns import regions_from_atlas

    surf, affine, shape, data, v = sphere
    a = _label(tmp_path, np.flatnonzero(v[:, 2] > 0.7), "lh.north.label")
    b = _label(tmp_path, np.flatnonzero(v[:, 2] < -0.7), "lh.south.label")
    regions = regions_from_atlas([surf], [a, b])
    assert regions.n_columns == 2 and regions.names == ["lh.north", "lh.south"]
    north = regions.vertex_id[regions.members[0].indices]
    assert set(north) == set(np.flatnonzero(v[:, 2] > 0.7))
    assert v[regions.centre_vertex[0], 2] > 0.9
