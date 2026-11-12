"""ffs_group surface results in the viewer: real tables, real cluster reports.

tests/viewer/test_surface_layers.py checks the viewer against hand-written
tables; this runs ffs_group per hemisphere on the same kind of meshes and
checks that what it writes -- permutation ClustSim tables in each GIfTI,
per-test files, ETAC masks -- loads, clusters and reports as a surface layer.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy import stats

from fastfuncstuff.cli.group import main
from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data, save_gifti_surface
from fastfuncstuff.surface.smooth import HeatSmoother
from tests.test_surface_ribbon import _icosphere

N_SUBJ = 10


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grp")
    d, f = _icosphere(4)
    centres = {"lh": np.array([-25.0, 0.0, 0.0]), "rh": np.array([25.0, 0.0, 0.0])}
    rng = np.random.default_rng(0)
    for h, c in centres.items():
        white, pial = c + 20 * d, c + 23 * d
        mid = 0.5 * (white + pial)
        save_gifti_surface(tmp / f"{h}.white.surf.gii", white, f)
        save_gifti_surface(tmp / f"{h}.pial.surf.gii", pial, f)
        save_gifti_surface(tmp / f"{h}.midthickness.surf.gii", mid, f)
        meta = {"mesh_fingerprint": mesh_fingerprint(f, len(d)),
                "geometry": f"{h}.midthickness.surf.gii"}  # fmt: skip
        hs = HeatSmoother(mid, f, 6.0)
        north = (d[:, 2] > 0.8).astype(float)
        paths = []
        for i in range(N_SUBJ):
            x = hs(rng.standard_normal(len(d)))
            x = x / x.std() + 1.5 * north
            p = tmp / f"s{i:02d}.{h}.shape.gii"
            save_gifti_data(p, x.astype(np.float32), meta)
            paths.append(str(p))
        main(["-setA", *paths, "-labelA", "vis", "-prefix", str(tmp / f"grp.{h}"),
              "-device", "cpu", "-verb", "0", "-clustsim", "-etac", "-etac_pthr", "0.01",
              "0.002", "-niter", "300"])  # fmt: skip
    aff = np.eye(4)
    aff[:3, 3] = [-55.0, -30.0, -30.0]
    nib.save(nib.Nifti1Image(np.ones((110, 60, 60), np.float32), aff), str(tmp / "anat.nii.gz"))
    return tmp, d


@pytest.fixture
def session(group):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import LoadMesh

    tmp, _ = group
    s = ViewerSession(device=torch.device("cpu"))
    s.load(str(tmp / "anat.nii.gz"))
    for h in ("lh", "rh"):
        for k in ("white", "pial"):
            s.do(LoadMesh(str(tmp / f"{h}.{k}.surf.gii"), h, k))
    yield s
    s.close()


def _threshold(s, key, p_two_sided):
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    s.do(SetThresholdIndex(key, 1))
    s.do(SetThreshold(key, float(stats.t.isf(p_two_sided / 2, N_SUBJ - 1))))


def test_group_result_clusters_with_corrected_alpha_in_the_viewer(group, session):
    tmp, d = group
    key = session.load(str(tmp / "grp.lh.func.gii"))
    layer = session.state.layers.get(key)
    assert layer.source == "surface:lh,rh"  # the rh result joined by name
    assert layer.labels == ("vis_mean", "vis_Tstat")
    _threshold(session, key, 0.002)
    _, table = session.clusterize(key, min_voxels=1)
    assert len(table) >= 2  # the planted cap on each hemisphere
    big = [c for c in table if c.area_mm2 > 200]
    assert len(big) == 2
    for c in big:
        assert c.alpha is not None and c.alpha <= 0.05  # earned from the permutation table
    assert table.alpha_range == (0.01, 0.1)
    assert table.pthr == pytest.approx(0.002, rel=1e-6)
    assert "mm² of cortex" in table.summary()


def test_alpha_rule_sizes_each_hemisphere_from_its_own_group_table(group, session):
    tmp, _ = group
    key = session.load(str(tmp / "grp.lh.func.gii"))
    _threshold(session, key, 0.002)
    _, table = session.clusterize(key, min_voxels=1, alpha=0.05)
    assert "lh ≥" in table.size_rule and "rh ≥" in table.size_rule
    from fastfuncstuff.stats.clustsim import clustsim_min_size
    from fastfuncstuff.surface.statmap import load_surface_data

    need = {}
    for h in ("lh", "rh"):
        t = load_surface_data(tmp / f"grp.{h}.func.gii").tables["bi-sided"]
        need[h] = clustsim_min_size(t["pthr"], t["athr"], t["area_mm2"], 0.002, 0.05)
    assert len(table) >= 2
    for c in table:  # each survivor meets its own hemisphere's table (lh sits at x < 0)
        assert c.area_mm2 >= need["lh" if c.peak_xyz[0] < 0 else "rh"]


def test_etac_mask_and_per_test_files_open_as_surface_layers(group, session):
    tmp, _ = group
    m = session.state.layers.get(
        session.load(str(tmp / "grp.lh.ETACmask.global.2sid.5perc.func.gii"))
    )
    assert m.source == "surface:lh,rh"
    allm = session.state.layers.get(
        session.load(str(tmp / "grp.lh.ETACmaskALL.global.2sid.5perc.func.gii"))
    )
    assert allm.labels == ("B0.0_p0.01", "B0.0_p0.002")


def test_a_group_result_brings_its_own_mesh_when_none_is_loaded(group):
    """fs_LR-style data: only a midthickness, named in the result's metadata."""
    from fastfuncstuff.viewer.session import ViewerSession

    tmp, d = group
    s = ViewerSession(device=torch.device("cpu"))
    try:
        key = s.load(str(tmp / "grp.lh.func.gii"))
        assert s.state.layers.get(key).source == "surface:lh,rh"
        assert set(s.surfaces.hemis) == {"lh", "rh"}
        _threshold(s, key, 0.002)
        _, table = s.clusterize(key, min_voxels=1)
        big = [c for c in table if c.area_mm2 > 200]
        assert len(big) == 2 and all(c.alpha is not None for c in big)
    finally:
        s.close()


def test_a_subjects_own_mesh_is_never_replaced(group, session, tmp_path):
    """A loaded hemisphere on another mesh keeps it; the result is refused as before."""
    from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data, save_gifti_surface

    d2, f2 = _icosphere(3)
    save_gifti_surface(tmp_path / "lh.midthickness.surf.gii", 20 * d2, f2)
    meta = {
        "mesh_fingerprint": mesh_fingerprint(f2, len(d2)),
        "geometry": "lh.midthickness.surf.gii",
    }
    save_gifti_data(tmp_path / "x.lh.func.gii", np.ones(len(d2), np.float32), meta)
    with pytest.raises(ValueError, match="not on any loaded mesh"):
        session.load(str(tmp_path / "x.lh.func.gii"))
