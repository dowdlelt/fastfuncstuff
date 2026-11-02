"""Surface results as layers: one stack, the same controls, slices and vertices agree."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy import stats

from fastfuncstuff.io.afni import save_nifti
from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_surface, set_surface_meta
from tests.test_surface_ribbon import _icosphere

DOF = 60.0


def _bucket(path, faces, n, t, table=None):
    hdr = nib.Nifti2Header()
    meta = {"mesh_fingerprint": mesh_fingerprint(faces, n)}
    if table is not None:
        meta["ClustSim_bi-sided"] = json.dumps(table)
    set_surface_meta(hdr, meta)
    vals = np.stack([0.5 * t, t], 1).astype(np.float32)[:, None, None, :]
    save_nifti(vals, path, header=hdr, brick_labels=["c#0_Coef", "c#0_Tstat"],
               brick_stataux={1: (3, (DOF,))})  # fmt: skip


@pytest.fixture
def world(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import LoadMesh

    d, f = _icosphere(4)
    centres = {"lh": np.array([-25.0, 0.0, 0.0]), "rh": np.array([25.0, 0.0, 0.0])}
    for h, c in centres.items():
        save_gifti_surface(tmp_path / f"{h}.white.surf.gii", c + 20 * d, f)
        save_gifti_surface(tmp_path / f"{h}.pial.surf.gii", c + 23 * d, f)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = [-55.0, -30.0, -30.0]
    nib.save(
        nib.Nifti1Image(np.ones((110, 60, 60), np.float32), aff), str(tmp_path / "anat.nii.gz")
    )
    # A blob of strong t on each hemisphere's "north", nothing elsewhere.
    t = np.where(d[:, 2] > 0.8, 6.0, 0.5).astype(np.float32)
    table = {
        "pthr": [0.01, 0.001],
        "athr": [0.1, 0.05],
        "area_mm2": [[400.0, 500.0], [100.0, 150.0]],
    }
    for h in centres:
        _bucket(tmp_path / f"s.{h}.func.gii", f, len(d), t, table)
    s = ViewerSession(device=torch.device("cpu"))
    s.load(str(tmp_path / "anat.nii.gz"))
    for h in centres:
        for k in ("white", "pial"):
            s.do(LoadMesh(str(tmp_path / f"{h}.{k}.surf.gii"), h, k))
    yield s, tmp_path, d, f, t
    s.close()


def test_a_gii_loads_as_one_layer_painted_into_both_ribbons(world):
    s, tmp, d, f, t = world
    key = s.load(str(tmp / "s.lh.func.gii"))
    layer = s.state.layers.get(key)
    assert layer.source == "surface:lh,rh" and layer.n_volumes == 2
    assert layer.stataux == {1: (3, (DOF,))} and layer.labels[1] == "c#0_Tstat"
    assert layer.shape == s.state.grid.shape  # on the display grid
    vol = s.display_volume(key, index=1).cpu().numpy()
    painted = vol != 0
    sld = s.surface_layers[key]
    ribbon = np.zeros(vol.size, bool)
    for rm in sld.maps.values():
        ribbon[rm.flat] = True
    assert not (painted.ravel() & ~ribbon).any()  # nothing off the ribbon
    rm = sld.maps["lh"]
    np.testing.assert_array_equal(vol.ravel()[rm.flat], t[rm.vertex])


def test_the_same_threshold_cuts_slices_and_vertices(world):
    from fastfuncstuff.viewer.compose import render_plane
    from fastfuncstuff.viewer.state import Plane
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    s, tmp, d, f, t = world
    key = s.load(str(tmp / "s.lh.func.gii"))
    s.do(SetThresholdIndex(key, 1))
    thr = float(stats.t.isf(0.0005, DOF))
    s.do(SetThreshold(key, thr))
    rgba, layer = s.surface_vertex_rgba("lh")
    assert layer.key == key
    np.testing.assert_array_equal(rgba[:, 3] > 0, t > thr)
    s.state.crosshair = (30, 30, 45)  # an axial plane through the north caps
    assert render_plane(s, Plane.AXIAL) is not None


def test_clusters_are_found_on_the_mesh_with_their_corrected_alpha(world):
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    s, tmp, d, f, t = world
    key = s.load(str(tmp / "s.lh.func.gii"))
    s.do(SetThresholdIndex(key, 1))
    s.do(SetThreshold(key, float(stats.t.isf(0.0005, DOF))))  # p = .001, two-sided
    layer, table = s.clusterize(key, min_voxels=1)
    assert len(table) == 2  # one cap per hemisphere
    north_area = 2 * np.pi * 21.5**2 * (1 - 0.8)  # spherical cap on the midthickness
    for c in table:
        assert c.area_mm2 == pytest.approx(north_area, rel=0.15)
        # Far bigger than the strictest area simulated (150 mm^2): alpha sits at the
        # table's bound, which the cluster window prints as "<0.05".
        assert c.alpha == pytest.approx(0.05)
    assert table.alpha_range == (0.05, 0.1)
    assert set(np.unique(table.labels)) == {0, 1, 2}
    assert table.pthr == pytest.approx(0.001, rel=1e-6)


def test_a_result_with_no_mesh_for_it_says_so(world, tmp_path):
    s, tmp, d, f, t = world
    d2, f2 = _icosphere(3)
    _bucket(tmp_path / "other.lh.func.gii", f2, len(d2), np.ones(len(d2), np.float32))
    with pytest.raises(ValueError, match="not on any loaded mesh"):
        s.load(str(tmp_path / "other.lh.func.gii"))


def test_a_bucket_is_not_a_series_even_with_a_tr_and_a_series_is(world, tmp_path):
    from fastfuncstuff.io.gifti import save_gifti_data

    s, tmp, d, f, t = world
    key = s.load(str(tmp / "s.lh.func.gii"))
    assert not s.state.layers.get(key).time_linked
    fp = mesh_fingerprint(f, len(d))
    series = np.random.default_rng(0).normal(size=(len(d), 5)).astype(np.float32)
    save_gifti_data(
        tmp_path / "run.lh.func.gii", series, {"TR_seconds": "2", "mesh_fingerprint": fp}
    )
    run = s.state.layers.get(s.load(str(tmp_path / "run.lh.func.gii")))
    assert run.time_linked and run.n_volumes == 5 and run.labels == ()


def test_the_cluster_window_sizes_surface_clusters_in_mm2(world):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.clusterwindow import ClusterWindow
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s, tmp, d, f, t = world
    key = s.load(str(tmp / "s.lh.func.gii"))
    s.do(SetThresholdIndex(key, 1))
    s.do(SetThreshold(key, float(stats.t.isf(0.0005, DOF))))
    _, table = s.clusterize(key, min_voxels=1)
    win = ClusterWindow("C1", s, s.do)
    win.show_table(key, table)
    assert win.table.horizontalHeaderItem(2).text() == "mm²"
    area = float(win.table.item(0, 2).text().replace(",", ""))
    assert area == pytest.approx(table.clusters[0].area_mm2, rel=0.01)
    assert "mm² of cortex" in table.summary()
    win.close()
    app.processEvents()


def test_an_roi_saves_as_labels_and_the_cortex_it_spans_and_loads_back(world, tmp_path):
    from fastfuncstuff.viewer.vocab import (
        HighlightSurface,
        HighlightToRoi,
        LoadSurfaceRoi,
        SaveSurfaceRoi,
        encode_ids,
    )

    s, tmp, d, f, t = world
    north = np.flatnonzero(d[:, 2] > 0.8)
    s.do(HighlightSurface("lh", encode_ids(north), "set"))
    s.do(SaveSurfaceRoi(str(tmp_path / "roi")))
    lab = nib.freesurfer.read_label(str(tmp_path / "roi.lh.label"))
    np.testing.assert_array_equal(np.sort(lab), north)
    vol = np.asarray(nib.load(str(tmp_path / "roi.nii.gz")).dataobj) > 0
    # The volume is the ribbon voxels those vertices own, white to pial.
    base = s.state.layers.base
    rm = s.surfaces.ribbon_map("lh", base.shape, base.affine)
    expect = np.zeros(base.shape, bool)
    expect.reshape(-1)[rm.voxels_of(np.isin(np.arange(len(d)), north))] = True
    np.testing.assert_array_equal(vol, expect)
    depths = rm.depth[np.isin(rm.vertex, north)]
    assert depths.min() < 0.15 and depths.max() > 0.85  # it spans the depth
    # the same voxels as the ROI layer made from the highlight
    s.do(HighlightToRoi())
    top = s.state.layers[-1]
    np.testing.assert_array_equal(np.asarray(s.volume(top.key, 0)) > 0, expect)
    # and back
    s.do(HighlightSurface(mode="clear"))
    s.do(LoadSurfaceRoi(str(tmp_path / "roi.lh.label")))
    np.testing.assert_array_equal(s.surfaces.highlighted("lh"), north)


def test_a_shape_drawn_in_the_window_fills_as_one_recorded_highlight(world):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.surfacewindow import SurfaceWindow
    from fastfuncstuff.viewer.vocab import OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s, tmp, d, f, t = world
    s.do(OpenView("S1", "surface", "axial"))
    win = SurfaceWindow("S1", s, s.do)
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ring = [int(np.argmax(d @ np.r_[0.6 * np.cos(a), 0.6 * np.sin(a), 0.8])) for a in ang]
    win._fill_outline("lh", ring, False)
    shown = s.surfaces.highlight["lh"]
    assert shown[d[:, 2] > 0.85].all() and not shown[d[:, 2] < 0.7].any()
    assert "HIGHLIGHT_SURFACE lh" in s.to_script()
    win._fill_outline("lh", ring, True)  # ctrl+shift: the same shape erases
    assert not s.surfaces.highlight["lh"].any()
    win.close()
    app.processEvents()


def test_a_surface_overlay_gets_the_overlay_look(world):
    """-surf_data used to AddLayer, which skips the overlay defaults: a signed map
    arrived gray, unthresholded, on a percentile range. Load gives it what a volume
    stat gets, and AUTO's look is read off the vertices (the painted ribbon counts
    each vertex once per voxel it owns)."""
    from fastfuncstuff.viewer.vocab import Load

    s, tmp, d, f, t = world
    signed = np.where(d[:, 0] > 0, t, -t).astype(np.float32)
    for h in ("lh", "rh"):
        _bucket(tmp / f"g.{h}.func.gii", f, len(d), signed)
    s.do(Load(str(tmp / "g.lh.func.gii"), "S"))
    layer = s.state.layers.get("S")
    # Opens on the coefficient, which has no p: uncut, with p 0.001 held for its t.
    assert layer.threshold == 0 and layer.threshold_p == pytest.approx(1e-3)
    assert layer.colormap == "redblue" and layer.range_lo == -layer.range_hi == -3.0
    look = s.overlay_look("S", 1, colormap=layer.colormap)  # what AUTO applies
    assert look["range_lo"] == -look["range_hi"] and look["range_hi"] == pytest.approx(6.0)


def test_an_alpha_sizes_each_hemisphere_by_its_own_table(world):
    """Each hemisphere was simulated apart: at α 0.05 lh asks for 150 mm², rh (a
    smoother map, say) for 900 -- more than its ~580 mm² cap -- so only lh's stays."""
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    s, tmp, d, f, t = world
    rough = {"pthr": [0.01, 0.001], "athr": [0.1, 0.05], "area_mm2": [[1500.0, 1800.0], [700.0, 900.0]]}  # fmt: skip
    _bucket(tmp / "s.rh.func.gii", f, len(d), t, rough)
    key = s.load(str(tmp / "s.lh.func.gii"))
    s.do(SetThresholdIndex(key, 1))
    s.do(SetThreshold(key, float(stats.t.isf(0.0005, DOF))))
    _, table = s.clusterize(key, min_voxels=1, alpha=0.05)
    assert len(table) == 1 and table.clusters[0].peak_xyz[0] < 0  # lh's cap
    assert "lh ≥ 150 mm²" in table.size_rule and "rh ≥ 900 mm²" in table.size_rule
    assert table.is_surface and table.dropped is not None and table.dropped.any()


def test_ribbon_map_comes_back_from_disk_until_the_mesh_changes(world, monkeypatch):
    import fastfuncstuff.surface.ribbon as ribbon

    s, _, _, _, _ = world
    base = s.state.layers.base
    fresh = s.surfaces.ribbon_maps(["lh", "rh"], base.shape, base.affine)
    s.surfaces._ribbons.clear()  # a new session: nothing in memory

    def no_build(*a, **k):
        raise AssertionError("rebuilt a ribbon map that was on disk")

    monkeypatch.setattr(ribbon, "build_ribbon_map", no_build)
    again = s.surfaces.ribbon_maps(["lh", "rh"], base.shape, base.affine)
    for h in ("lh", "rh"):
        for field in ("flat", "vertex", "depth"):
            np.testing.assert_array_equal(getattr(again[h], field), getattr(fresh[h], field))
        assert again[h].n_vertices == fresh[h].n_vertices

    # An edited pial is a different mesh: the disk entry must not answer for it.
    s.surfaces.hemis["lh"].states["pial"] = s.surfaces.hemis["lh"].states["pial"] * 1.01
    s.surfaces.version["lh"] = s.surfaces.version.get("lh", 0) + 1
    with pytest.raises(AssertionError, match="rebuilt"):
        s.surfaces.ribbon_map("lh", base.shape, base.affine)


def test_launch_command_reopens_what_is_loaded(world):
    import shlex

    from fastfuncstuff.cli.viewer import build_parser
    from fastfuncstuff.viewer.vocab import Load

    s, tmp_path, _, _, _ = world
    s.read_directory(str(tmp_path))
    s.do(Load(str(tmp_path / "s.lh.func.gii")))
    gone = tmp_path / "gone.nii.gz"
    gone.write_bytes((tmp_path / "anat.nii.gz").read_bytes())
    s.do(Load(str(gone)))
    gone.unlink()  # loaded, then deleted: nothing left to name

    line = s.launch_command()
    command, _, note = line.partition("  # not reloaded: ")
    assert note == "gone.nii.gz"
    argv = shlex.split(command)
    assert argv[0] == "ffs_viewer"
    args = build_parser().parse_args(argv[1:])
    assert args.read == [str(tmp_path.resolve())]
    assert args.datasets == [str((tmp_path / "anat.nii.gz").resolve())]
    assert args.surf_data == [str((tmp_path / "s.lh.func.gii").resolve())]
    assert sorted(args.mesh) == sorted(
        str((tmp_path / f"{h}.{k}.surf.gii").resolve())
        for h in ("lh", "rh")
        for k in ("white", "pial")
    )
    assert args.surfaces is None


def test_a_surface_series_is_shown_through_its_time_view(world):
    from fastfuncstuff.io.gifti import save_gifti_data
    from fastfuncstuff.viewer.timeview import build_time_view
    from fastfuncstuff.viewer.vocab import Load, SetIndex, SetLayerTimeView

    s, tmp_path, d, f, _ = world
    rng = np.random.default_rng(1)
    series = (1000 + 5 * np.arange(8.0) + rng.normal(size=(len(d), 8))).astype(np.float32)
    fp = mesh_fingerprint(f, len(d))
    save_gifti_data(
        tmp_path / "r.lh.func.gii", series, meta={"mesh_fingerprint": fp, "TR_seconds": "2"}
    )
    s.do(Load(str(tmp_path / "r.lh.func.gii")))
    layer = list(s.state.layers)[-1]
    assert layer.time_linked and layer.n_volumes == 8
    s.do(SetIndex(5))
    s.do(SetLayerTimeView(layer.key, 1, False))
    layer = s.state.layers.get(layer.key)
    want = build_time_view(series, 1, False).frame(series[:, 5], 5)
    np.testing.assert_allclose(s.vertex_frame(layer, "lh", 5), want, atol=1e-3)
    # The ribbon painted from the viewed values, not the stored ones.
    rm = s.surfaces.ribbon_map("lh", layer.shape, layer.affine)
    vol = s.shown_volume(layer.key, 5)
    np.testing.assert_allclose(vol.reshape(-1)[rm.flat], want[rm.vertex], atol=1e-3)


def test_unticking_the_selected_surface_layer_hands_the_surface_to_the_next(world):
    from fastfuncstuff.viewer.vocab import Load, SelectLayer, SetLayerVisible

    s, tmp_path, *_ = world
    s.do(Load(str(tmp_path / "s.lh.func.gii")))
    below = list(s.state.layers)[-1].key
    s.do(Load(str(tmp_path / "s.lh.func.gii")))
    top = list(s.state.layers)[-1].key
    s.do(SelectLayer(top))
    assert s.surface_vertex_rgba("lh")[1].key == top
    before = s.surface_display_token("lh")
    s.do(SetLayerVisible(top, False))
    # Still selected, but hidden: the surface shows the one below, and the
    # token a 3-D window recolours on has changed with it.
    assert s.surface_vertex_rgba("lh")[1].key == below
    assert s.surface_display_token("lh") != before
