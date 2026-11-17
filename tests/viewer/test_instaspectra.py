"""InstaSpectra: the engine's maps, and the mode laying them along the time axis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats import spectra
from fastfuncstuff.viewer import instaspectra as engine
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import SetIndex, SetMode, SetOverlay, SetUnderlay, SetVolume

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
TR = 0.5
N_TIME = 240
SHAPE = (10, 10, 6)
AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])
#: A "heartbeat" at 0.8 Hz, below Nyquist (1 Hz) at this TR, in one corner.
F_HEART = 0.8
CORNER = (slice(1, 4), slice(1, 4), slice(1, 3))


def _planted(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIME) * TR
    data = rng.normal(5.0, 0.5, (*SHAPE, N_TIME)).astype(np.float32)
    box = np.zeros(SHAPE, bool)
    box[1:9, 1:9, 1:5] = True
    # Bright, drifting, noisy brain.
    data[box] += 1000.0 + np.linspace(0, 40, N_TIME) + rng.normal(0, 2.0, (box.sum(), N_TIME))
    data[CORNER] += 3.0 * np.sin(2 * np.pi * F_HEART * t)
    # A loud voxel with nothing periodic: absolute power would light it up.
    data[7, 7, 3] += rng.normal(0, 30.0, N_TIME)
    return data, box


def test_the_heartbeat_bin_lights_its_corner_and_not_the_loud_voxel():
    data, box = _planted()
    s = engine.compute(data, affine=AFFINE, tr=TR, mask=box, device=CPU)
    k = s.frame_of(F_HEART)
    for kind in engine.MAP_KINDS:
        if kind == "amplitude":
            continue  # amplitude is meant to follow loudness
        vol = s.volume(s.maps(kind)[:, k])
        assert np.nanmin(vol[CORNER][box[CORNER]]) > 3.0, kind
        assert vol[7, 7, 3] < 3.0, kind
    assert np.isnan(s.volume(s.maps()[:, k])[0, 0, 0])


def test_share_is_a_percentage_of_the_power_above_the_floor():
    data, box = _planted(1)
    s = engine.compute(data, affine=AFFINE, tr=TR, mask=box, device=CPU)
    share = s.maps("share", scale="value", floor=0.04)
    above = s.freqs >= 0.04
    np.testing.assert_allclose(share[:, above].sum(axis=1), 100.0, rtol=1e-4)


def test_a_band_sums_its_bins_and_amplitude_is_its_sd_in_percent():
    data, box = _planted(2)
    s = engine.compute(data, affine=AFFINE, tr=TR, mask=box, device=CPU)
    width = 4 * s.df  # two bins either side
    amp = s.maps("amplitude", width=width, scale="value")
    k = s.frame_of(F_HEART)
    v = 0
    band = s.psd[v, k - 2 : k + 3].astype(np.float64).sum() * s.df
    assert amp[v, k] == pytest.approx(100 * np.sqrt(band) / abs(s.mean[v]), rel=1e-4)


def test_robust_z_is_a_median_and_mad_per_frame():
    rng = np.random.default_rng(0)
    m = torch.as_tensor(rng.normal(3.0, 2.0, (5000, 4)), dtype=torch.float32)
    z = engine.robust_z(m)
    assert float(z.median(dim=0).values.abs().max()) < 1e-5
    assert float(z.std(dim=0).mean()) == pytest.approx(1.0, abs=0.05)


def test_the_voxel_spectrum_matches_the_engines_row():
    data, box = _planted(3)
    s = engine.compute(data, affine=AFFINE, tr=TR, mask=box, device=CPU)
    flat = np.flatnonzero(box.reshape(-1))
    i, j, k = np.unravel_index(flat[5], SHAPE)
    one = engine.voxel_psd(data[i, j, k], tr=TR, polort=s.polort)
    np.testing.assert_allclose(one, s.psd[5], rtol=1e-3, atol=1e-6 * float(s.psd[5].max()))


def test_no_tr_falls_back_to_cycles_per_volume():
    data, box = _planted(4)
    s = engine.compute(data, affine=AFFINE, tr=0.0, mask=box, device=CPU)
    assert not s.tr_known and s.unit == "cycles/vol"
    assert s.freqs[-1] == pytest.approx(0.5)


# -- mode -------------------------------------------------------------------


def _write(path, data, tr=0.0):
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), AFFINE)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


@pytest.fixture
def spec_session(tmp_path):
    data, box = _planted(6)
    _write(tmp_path / "anat.nii.gz", data.mean(axis=3))
    _write(tmp_path / "bold.nii.gz", data, tr=TR)
    _write(tmp_path / "mask.nii.gz", box.astype(np.float32))
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    s.tmp = tmp_path
    yield s
    s.close()


def _map_layer(s):
    layer = s.state.layers.find_by_source("mode:instaspectra")
    assert layer is not None
    return layer


def test_the_map_has_one_frame_per_bin_stepped_apart_from_time(spec_session):
    """Its own sub-brick index: stepping frequency must not move the run."""
    s = spec_session
    s.do(SetMode("instaspectra"))
    s.set_mode_param("mask", str(s.tmp / "mask.nii.gz"))
    s.mode.action("go")
    result = s.mode._result
    layer = _map_layer(s)
    assert not layer.time_linked and layer.n_volumes == result.n_frames
    s.do(SetIndex(7))
    k = result.frame_of(F_HEART)
    s.do(SetVolume(layer.key, k))
    shown = s.volume(layer.key, _map_layer(s).volume_index)
    assert np.nanmin(shown[CORNER][result.mask[CORNER]]) > 3.0
    for _ in range(3):
        s.mode.action("next")
    assert _map_layer(s).volume_index == k + 3
    assert s.state.time_index == 7
    assert layer.labels[k].endswith("Hz")
    # A new look at the same spectra keeps the bin.
    s.set_mode_param("width", "0.05")
    assert _map_layer(s).volume_index == k + 3


def test_comma_and_period_step_the_selected_map_not_time(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    layer = _map_layer(s)
    s.state.selected = layer.key
    s.do(SetIndex(4))
    s.do(s.frame_step(1))
    assert _map_layer(s).volume_index == 1 and s.state.time_index == 4


def test_the_spectrum_panel_marks_the_current_bin_and_scrubs_the_map(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    k = s.mode._result.frame_of(F_HEART)
    s.do(SetVolume(_map_layer(s).key, k))
    traces = s.mode.panels()["spectrum"]
    voxel = traces[0]
    assert voxel.marker == k and voxel.scrub and voxel.scrub_key == _map_layer(s).key
    assert voxel.x is not None and voxel.x[k] == pytest.approx(s.mode._result.freqs[k])


def test_voxel_peak_and_alias_jump_the_bin(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    result = s.mode._result
    s.do(SetIndex(5))
    # Into the corner, then ask for that voxel's peak.
    s.state.crosshair = (2, 2, 1)
    s.mode.action("voxel_peak")
    assert _map_layer(s).volume_index == result.frame_of(F_HEART)
    # A rate of 48 bpm is 0.8 Hz, below Nyquist: the alias is itself.
    s.set_mode_param("heart_bpm", "48")
    s.do(SetVolume(_map_layer(s).key, 0))
    s.mode.action("alias")
    assert _map_layer(s).volume_index == result.frame_of(spectra.alias(0.8, TR))
    assert s.state.time_index == 5


def test_display_changes_do_not_recompute_the_spectra(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    first = s.mode._result
    s.set_mode_param("map", "peak")
    s.set_mode_param("width", "0.05")
    s.set_mode_param("scale", "value")
    assert s.mode._result is first


def test_keep_freezes_a_lazy_copy(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    s.mode.action("keep")
    kept = [ly for ly in s.state.layers if ly.source == "kept:mode:instaspectra"]
    assert len(kept) == 1
    live = _map_layer(s)
    assert kept[0].n_volumes == live.n_volumes and not kept[0].time_linked
    np.testing.assert_array_equal(
        np.nan_to_num(s.volume(kept[0].key, 3)), np.nan_to_num(s.volume(live.key, 3))
    )


def test_clicking_the_spectrum_emits_the_nearest_bin():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets

    from fastfuncstuff.viewer.modes.base import Trace
    from fastfuncstuff.viewer.ui.tracewindow import PlotView

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = PlotView()
    view.resize(400, 200)
    freqs = np.linspace(0.01, 1.0, 100, dtype=np.float32)
    view.set_traces([Trace("s", np.ones(100, np.float32), x=freqs, scrub=True)])
    view.grab()  # paint once, so the click has axes to land on
    got: list[int] = []
    view.scrubbed.connect(got.append)
    plot, x0, x1 = view._geom
    px = plot.left() + (0.5 - x0) / (x1 - x0) * plot.width()
    press = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(px, plot.center().y()),
        QtCore.QPointF(px, plot.center().y()),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    assert got and abs(freqs[got[0]] - 0.5) < 0.011


def test_a_click_in_the_spectrum_window_moves_the_map_and_not_time(spec_session):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.tracewindow import TraceWindow
    from fastfuncstuff.viewer.viewports import ViewKind

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = spec_session
    s.do(SetMode("instaspectra"))
    s.open_mode_panels()
    vp = next(v for v in s.state.viewports.of_kind(ViewKind.TRACE) if v.panel == "spectrum")
    win = TraceWindow(vp.id, s, s.do)
    win.apply(vp)
    s.do(SetIndex(6))
    timed: list[int] = []
    win.scrubbed.connect(timed.append)
    win._scrubbed(9)
    assert _map_layer(s).volume_index == 9
    assert s.state.time_index == 6 and not timed
    win.close()


def test_the_3d_window_follows_the_bin(spec_session):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.ui.vox3dwindow import Vox3DWindow
    from fastfuncstuff.viewer.vocab import OpenView

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = spec_session
    s.do(SetMode("instaspectra"))
    s.do(OpenView("X1", "vox3d", "axial"))
    win = Vox3DWindow("X1", s, s.do)
    win.apply(s.state.viewports.get("X1"))
    assert win.chosen_layers()[0].key == _map_layer(s).key
    before = win.canvas.overlay.data.copy()
    k = s.mode._result.frame_of(F_HEART)
    win.refresh(s.do(SetVolume(_map_layer(s).key, k)))
    after = win.canvas.overlay.data
    assert not np.array_equal(before, after)
    # (nz, ny, nx): the heartbeat corner is bright at its bin.
    assert after[1, 2, 2] > 3.0
    win.refresh(Aspect.TIME)  # stepping time does not move it
    assert np.array_equal(win.canvas.overlay.data, after)
    win.close()


def test_the_bin_and_its_band_are_shown_where_you_look(spec_session):
    """Status line, sub-brick label, the spectrum's shaded band and the 3-D
    caption all name the frequency -- and the band's edges once it is wide."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.vox3dwindow import Vox3DWindow
    from fastfuncstuff.viewer.vocab import OpenView

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = spec_session
    s.do(SetMode("instaspectra"))
    r = s.mode._result
    k = r.frame_of(F_HEART)
    s.do(SetVolume(_map_layer(s).key, k))
    here = f"{r.freqs[k]:.4g} Hz"
    assert here in s.mode.status() and "band" not in s.mode.status()
    lo, hi = s.mode.panels()["spectrum"][0].band
    assert hi - lo == pytest.approx(r.df)  # one bin: its own width

    s.set_mode_param("width", str(4 * r.df))  # two bins either side
    lo, hi = r.band_edges(k, 4 * r.df)
    assert (lo, hi) == pytest.approx((r.freqs[k - 2] - r.df / 2, r.freqs[k + 2] + r.df / 2))
    band = f"(band {lo:.3g}-{hi:.3g})"
    assert band in s.mode.status()
    assert _map_layer(s).labels[k].endswith(band)
    assert s.mode.panels()["spectrum"][0].band == pytest.approx((lo, hi))

    s.do(OpenView("X1", "vox3d", "axial"))
    win = Vox3DWindow("X1", s, s.do)
    win.apply(s.state.viewports.get("X1"))
    assert band in win.canvas.caption and here in win.canvas.caption
    win.close()


def test_the_spectrum_window_writes_the_frequency_at_the_marker(spec_session):
    """With two lines the plot shows a legend instead of a title, which hid the
    frequency; the marker's own caption is drawn above the bar regardless."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    from fastfuncstuff.viewer.ui import theme
    from fastfuncstuff.viewer.ui.tracewindow import PlotView

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = spec_session
    s.do(SetMode("instaspectra"))
    r = s.mode._result
    k = r.frame_of(F_HEART)
    s.do(SetVolume(_map_layer(s).key, k))
    traces = s.mode.panels()["spectrum"]
    assert len(traces) == 2
    assert traces[0].marker_label == f"{r.freqs[k]:.4g} Hz"

    def strip(view: PlotView) -> np.ndarray:
        img = view.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        px = np.array(img.constBits()).reshape(img.height(), img.width(), 4)[..., :3]
        plot, x0, x1 = view._geom
        x = int(plot.left() + (r.freqs[k] - x0) / (x1 - x0) * plot.width())
        return px[: int(plot.top()), max(x - 30, 0) : x + 30].astype(int)

    warn = QtGui.QColor(theme.palette().warn)
    want = np.array([warn.red(), warn.green(), warn.blue()])
    with_caption = PlotView()
    with_caption.resize(500, 220)
    with_caption.set_traces(traces)
    with_caption.grab()
    assert (np.abs(strip(with_caption) - want).max(axis=-1) <= 10).sum() > 20
    from dataclasses import replace

    bare = PlotView()
    bare.resize(500, 220)
    bare.set_traces([replace(t, marker_label="") for t in traces])
    bare.grab()
    assert (np.abs(strip(bare) - want).max(axis=-1) <= 10).sum() == 0
