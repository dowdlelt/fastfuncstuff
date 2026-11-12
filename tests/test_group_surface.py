"""ffs_group on surfaces: mesh-edge clustering in mm^2, heat-kernel blur, GIfTI tables.

The clustering reference is surface_clustsim's own scipy connected components
over mesh edges (an independent algorithm from the batched GPU labeller that
SurfaceGeometry feeds), so agreement checks the neighbour table and the area
weighting together.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.cli.group import main
from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data, save_gifti_surface
from fastfuncstuff.stats.cluster_inference import SurfaceGeometry, max_fom_null
from fastfuncstuff.stats.surface_clustsim import max_cluster_areas
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, mesh_edges
from fastfuncstuff.surface.statmap import load_surface_data

CPU = torch.device("cpu")


def _icosphere(level: int, radius: float = 60.0):
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t)]
    v += [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4)]
    f += [(11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8)]
    f += [(3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        mid: dict = {}

        def m(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                mid[key] = len(verts) - 1
            return mid[key]

        faces = [g for a, b, c in faces for g in ((a, m(a, b), m(c, a)), (b, m(b, c), m(a, b)),
                                                   (c, m(c, a), m(b, c)), (m(a, b), m(b, c), m(c, a)))]  # fmt: skip
    return np.asarray(verts) * radius, np.asarray(faces, np.int64)


@pytest.fixture(scope="module")
def mesh():
    return _icosphere(4)  # 2562 vertices, ~4.7 mm spacing on a 60 mm sphere


def _smooth_fields(mesh, n, seed, fwhm=10.0):
    v, f = mesh
    rng = np.random.default_rng(seed)
    hs = HeatSmoother(v, f, fwhm)
    x = hs(rng.standard_normal((len(v), n)))
    return (x / x.std(0)).T.astype(np.float32)  # [n, V]


def test_neighbour_table_is_the_mesh_edges(mesh):
    v, f = mesh
    mask = np.ones(len(v), bool)
    mask[:100] = False
    geom = SurfaceGeometry(v, f, mask)
    tab = geom.neighbors(1, CPU).numpy()
    remap = np.full(len(v), -1)
    remap[mask] = np.arange(mask.sum())
    e = mesh_edges(f)
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    want = {(min(a, b), max(a, b)) for a, b in remap[e]}
    got = {(min(i, j), max(i, j)) for i in range(tab.shape[0]) for j in tab[i] if j >= 0}
    assert got == want


@pytest.mark.parametrize("sided", ["1-sided", "2-sided", "bi-sided"])
def test_area_weighted_max_cluster_matches_scipy_reference(mesh, sided):
    v, f = mesh
    mask = np.ones(len(v), bool)
    fields = _smooth_fields(mesh, 6, 1)
    geom = SurfaceGeometry(v, f, mask)
    thr = np.array([2.3, 1.6])
    got = max_fom_null(lambda s, n: torch.from_numpy(fields[s : s + n]), 6, geom, {sided: thr},
                       nns=(1,), sideds=(sided,), device=CPU, verbose=False)  # fmt: skip
    ref = max_cluster_areas(fields.T.astype(np.float64), mesh_edges(f), vertex_areas(v, f),
                            thr, sided)  # fmt: skip
    np.testing.assert_allclose(got[(sided, 1, 0)], ref, rtol=1e-5)


def _write_subjects(tmp_path, mesh, n, effect=0.0, seed=0):
    v, f = mesh
    geo = tmp_path / "lh.midthickness.surf.gii"
    save_gifti_surface(geo, v, f)
    meta = {"mesh_fingerprint": mesh_fingerprint(f, len(v)), "geometry": geo.name}
    blob = np.exp(-np.sum((v - v[0]) ** 2, 1) / (2 * 12.0**2))
    paths = []
    for i, x in enumerate(_smooth_fields(mesh, n, seed)):
        p = tmp_path / f"s{i:02d}.shape.gii"
        save_gifti_data(p, (x + effect * blob).astype(np.float32), meta)
        paths.append(p)
    return paths


def test_cli_surface_clustsim_etac_and_blur(tmp_path, mesh):
    paths = _write_subjects(tmp_path, mesh, 10, effect=1.5)
    out = tmp_path / "grp"
    main(["-setA", *map(str, paths), "-labelA", "vis", "-prefix", str(out), "-device", "cpu",
          "-verb", "0", "-clustsim", "-etac", "-etac_blur", "0", "6", "-etac_pthr", "0.01",
          "0.002", "-niter", "120"])  # fmt: skip
    bucket = tmp_path / "grp.func.gii"
    data = load_surface_data(bucket)
    assert data.labels == ["vis_mean", "vis_Tstat"]
    tab = data.tables["bi-sided"]
    assert np.asarray(tab["area_mm2"]).shape == (len(tab["pthr"]), len(tab["athr"]))
    assert tab["null"] == "permutation"
    # The blur case carries its own (larger) area tables.
    blurred = load_surface_data(tmp_path / "grp.B6.0.func.gii").tables["bi-sided"]
    assert blurred["area_mm2"][0][1] > tab["area_mm2"][0][1]
    union = load_surface_data(tmp_path / "grp.ETACmask.global.2sid.5perc.func.gii").values[:, 0]
    assert union[0] == 1.0  # the blob's centre survives
    assert (tmp_path / "grp.clustsim").is_dir()


def test_surface_input_without_a_mesh_is_refused(tmp_path, mesh, capsys):
    v, f = mesh
    for i in range(4):
        save_gifti_data(tmp_path / f"s{i}.shape.gii", np.random.default_rng(i).standard_normal(len(v)).astype(np.float32))  # fmt: skip
    with pytest.raises(SystemExit):
        main(["-setA", *[str(tmp_path / f"s{i}.shape.gii") for i in range(4)], "-clustsim",
              "-prefix", str(tmp_path / "x"), "-device", "cpu", "-verb", "0"])  # fmt: skip
    assert "-surf_geom" in capsys.readouterr().err


def test_cli_surface_mema_finds_tstat_in_gifti_buckets(tmp_path, mesh):
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import save_nifti_data_as_gifti

    paths = _write_subjects(tmp_path, mesh, 8, effect=1.0)
    rng = np.random.default_rng(3)
    buckets = []
    for i, p in enumerate(paths):
        img = load_nifti(p)
        b = np.asarray(img.dataobj, np.float32).reshape(-1)
        se = rng.uniform(0.4, 1.2, b.shape).astype(np.float32)
        out = tmp_path / f"stats{i}.func.gii"
        save_nifti_data_as_gifti(out, np.stack([b, b / se], -1)[:, None, None, :], img.header,
                                 brick_labels=["vis#0_Coef", "vis#0_Tstat"],
                                 brick_stataux={1: (3, (40.0,))})  # fmt: skip
        buckets.append(out)
    main(["-model", "mema", "-setA", *map(str, buckets), "-brick", "vis#0_Coef", "-prefix",
          str(tmp_path / "m"), "-device", "cpu", "-verb", "0", "-clustsim", "-niter", "60"])  # fmt: skip
    data = load_surface_data(tmp_path / "m.func.gii")
    assert data.labels[:2] == ["SetA:b", "SetA:t"]
    assert "bi-sided" in data.tables
