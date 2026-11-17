"""InstaSpectra: the engine's maps, and the mode laying them along the time axis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats import spectra
from fastfuncstuff.viewer import instaspectra as engine
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import SetIndex, SetMode, SetOverlay, SetUnderlay

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


def test_the_map_has_one_frame_per_bin_and_follows_the_time_index(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    s.set_mode_param("mask", str(s.tmp / "mask.nii.gz"))
    s.mode.action("go")
    result = s.mode._result
    layer = _map_layer(s)
    assert layer.time_linked and layer.n_volumes == result.n_frames
    k = result.frame_of(F_HEART)
    s.do(SetIndex(k))
    shown = s.volume(layer.key)
    assert np.nanmin(shown[CORNER][result.mask[CORNER]]) > 3.0
    s.do(SetIndex(1))
    assert not np.array_equal(np.nan_to_num(s.volume(layer.key)), np.nan_to_num(shown))
    assert layer.labels[k].endswith("Hz")


def test_the_spectrum_panel_marks_the_current_bin_and_scrubs(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    k = s.mode._result.frame_of(F_HEART)
    s.do(SetIndex(k))
    traces = s.mode.panels()["spectrum"]
    voxel = traces[0]
    assert voxel.marker == k and voxel.scrub
    assert voxel.x is not None and voxel.x[k] == pytest.approx(s.mode._result.freqs[k])


def test_voxel_peak_and_alias_jump_the_time_index(spec_session):
    s = spec_session
    s.do(SetMode("instaspectra"))
    result = s.mode._result
    # Into the corner, then ask for that voxel's peak.
    s.state.crosshair = (2, 2, 1)
    s.mode.action("voxel_peak")
    assert s.state.time_index == result.frame_of(F_HEART)
    # A rate of 48 bpm is 0.8 Hz, below Nyquist: the alias is itself.
    s.set_mode_param("heart_bpm", "48")
    s.do(SetIndex(0))
    s.mode.action("alias")
    assert s.state.time_index == result.frame_of(spectra.alias(0.8, TR))


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
    assert kept[0].n_volumes == live.n_volumes and kept[0].time_linked
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
