"""Window behaviour that has no way back if it breaks.

Runs Qt offscreen. These are deliberately not pixel tests -- the compositing is
covered headlessly in test_compose.py. What is tested here is the class of bug
that leaves the user stuck: a panel that cannot be reopened, a pane that cannot
be restored, a control that desynchronises from the state it displays.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
pytest.importorskip("PySide6")

# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from fastfuncstuff.viewer.commands import Aspect  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.state import Plane  # noqa: E402
from fastfuncstuff.viewer.ui.window import ViewerWindow  # noqa: E402
from fastfuncstuff.viewer.viewports import ViewKind  # noqa: E402
from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay  # noqa: E402

CPU = torch.device("cpu")


def image_of(win, plane):
    """The image window currently showing ``plane``."""
    for viewport in win.session.state.viewports.images:
        if viewport.plane is plane:
            return win.manager.windows[viewport.id]
    raise AssertionError(f"no image window on {plane}")


def open_graph(win, qapp, plane=Plane.AXIAL):
    """Open a graph window through the manager and return it."""
    vid = win.manager.open(ViewKind.GRAPH, plane)
    win.refresh(Aspect.VIEWPORTS | Aspect.GRAPH)
    qapp.processEvents()
    return win.manager.windows[vid]


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def datadir(tmp_path):
    rng = np.random.default_rng(31)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    for name, data in (
        ("anat.nii.gz", rng.random((10, 12, 8)) * 100),
        ("stats.nii.gz", rng.normal(size=(10, 12, 8))),
    ):
        nib.save(nib.Nifti1Image(np.asarray(data, np.float32), aff), str(tmp_path / name))
    return tmp_path


@pytest.fixture
def win(qapp, datadir):
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.read_directory(datadir)
    w.refresh(session.do(SetUnderlay(str(datadir / "anat.nii.gz"))))
    w.refresh(session.do(SetOverlay(str(datadir / "stats.nii.gz"))))
    w.show()
    qapp.processEvents()
    yield w
    w.close()


# ---------------------------------------------------------------------------
# companion windows
#
# The bug class that used to live here was "a panel with no way back". The
# panel is now the controller window itself, so what replaces it is the same
# question asked of the companions: can every window that can be closed be
# opened again, and does closing one leave the session consistent.
# ---------------------------------------------------------------------------


def test_the_default_layout_is_three_images_and_no_graph(win):
    """Goal zero is an underlay and an overlay; a graph is something you ask for."""
    assert [v.plane for v in win.session.state.viewports.images] == [
        Plane.AXIAL,
        Plane.SAGITTAL,
        Plane.CORONAL,
    ]
    assert win.session.state.viewports.graphs == []


def test_every_viewport_has_a_real_window(win):
    assert set(win.manager.windows) == set(win.session.state.viewports.ids)
    assert all(w.isWindow() for w in win.manager.windows.values())


def test_the_controller_holds_no_image(win):
    """The whole point of the split: the controller is controls, not brains.

    Companion windows are Qt children of the controller for ownership, so the
    check is on what the controller *lays out*, not on what it parents.
    """
    from fastfuncstuff.viewer.ui.panes import ImagePane

    assert win.centralWidget().findChildren(ImagePane) == []


def test_closing_a_window_closes_its_viewport(win, qapp):
    target = win.session.state.viewports.images[1]
    win.manager.windows[target.id].close()
    qapp.processEvents()
    assert target.id not in win.session.state.viewports.ids
    assert target.id not in win.manager.windows


def test_a_closed_window_can_always_be_opened_again(win, qapp):
    for viewport in list(win.session.state.viewports):
        win.manager.windows[viewport.id].close()
    qapp.processEvents()
    assert win.manager.windows == {}
    win._new_image()
    qapp.processEvents()
    assert len(win.manager.windows) == 1


def test_a_new_image_offers_a_plane_that_is_not_already_shown(win, qapp):
    """Opening a fourth wraps; opening onto a free plane comes first."""
    sagittal = image_of(win, Plane.SAGITTAL)
    sagittal.close()
    qapp.processEvents()
    win._new_image()
    qapp.processEvents()
    assert Plane.SAGITTAL in [v.plane for v in win.session.state.viewports.images]


def test_two_windows_of_the_same_plane_are_independent(win, qapp):
    """The assumption the viewport change exists to remove."""
    from fastfuncstuff.viewer.vocab import SetViewSolo

    first = image_of(win, Plane.AXIAL)
    second_id = win.manager.open(ViewKind.IMAGE, Plane.AXIAL)
    win.refresh(Aspect.VIEWPORTS | Aspect.SLICES)
    qapp.processEvents()
    assert second_id != first.vid
    win.refresh(win.session.do(SetViewSolo(second_id, True)))
    qapp.processEvents()
    assert win.manager.windows[second_id].solo_button.isChecked()
    assert not first.solo_button.isChecked()


def test_a_repaint_does_not_write_back_into_the_recording(win, qapp):
    """A sync that dispatches is a repaint that mutates state.

    Selection moved into state, so the layer list has to be told what is
    selected -- and setting a row fires the same signal a click does. Left
    unguarded, every redraw appended a SELECT_LAYER to the recording and could
    recurse through refresh.
    """
    win.session.bus.clear_log()
    win.refresh(Aspect.ALL)
    qapp.processEvents()
    assert [c.name for c in win.session.bus.log] == []


def test_tiling_gives_every_window_a_rectangle(win, qapp):
    before = len(
        [ln for ln in win.session.to_script().splitlines() if ln.startswith("SET_VIEW_GEOM")]
    )
    win._tile()
    qapp.processEvents()
    rects = [v.geometry for v in win.session.state.viewports]
    assert all(r is not None for r in rects)
    # Recorded per window: collapsing on command type alone would leave one.
    # Counted as a delta, because opening a window now records where it was
    # put as well, and those lines are in the script before tiling runs.
    lines = [ln for ln in win.session.to_script().splitlines() if ln.startswith("SET_VIEW_GEOM")]
    assert len(lines) - before == len(rects)


def test_tiling_does_not_cover_the_controller(win, qapp):
    """Tiling over the controller hides the panel the windows are driven from."""
    win.show()
    qapp.processEvents()
    win._tile()
    qapp.processEvents()
    controller = win.frameGeometry()
    for child in win.manager.windows.values():
        assert not controller.intersects(child.geometry())


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def test_a_graph_opens_as_a_floating_window(win, qapp):
    graph = open_graph(win, qapp)
    assert graph.isWindow(), "the graph must not be docked into the controller"
    assert graph.isVisible()


def test_the_graph_grid_steps_rather_than_choosing_a_preset(win, qapp):
    """Plus and minus, not 1/4/9: it is a square that grows."""
    graph = open_graph(win, qapp)
    seen = []
    for _ in range(3):
        graph.step_grid(1)
        qapp.processEvents()
        graph.refresh()
        seen.append(len(graph.graph._cells))
    assert seen == [9, 16, 25]


def test_the_graph_grid_clamps_at_one(win, qapp):
    graph = open_graph(win, qapp)
    for _ in range(5):
        graph.step_grid(-1)
        qapp.processEvents()
    graph.refresh()
    assert len(graph.graph._cells) == 1


# ---------------------------------------------------------------------------
# controls stay in sync with state
# ---------------------------------------------------------------------------


def test_selecting_a_layer_shows_that_layers_settings(win, qapp):
    """The list is top-first, so row 0 is the overlay, not the underlay."""
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    assert win.current_key() == win.session.state.layers.overlay.key
    win.layer_list.setCurrentRow(1)
    qapp.processEvents()
    assert win.current_key() == win.session.state.layers.base.key


def test_the_mode_panel_is_empty_in_view_mode(win):
    assert not win.mode_panel.isVisible() or win.mode_panel._form.rowCount() == 0


def test_switching_to_a_mode_renders_its_declared_controls(win, qapp):
    from fastfuncstuff.viewer.vocab import SetMode

    win.refresh(win.session.do(SetMode("instacorr")))
    qapp.processEvents()
    assert set(win.mode_panel._widgets) == {
        "polort",
        "fbot",
        "ftop",
        "blur",
        "seed_radius",
        "action:keep",
    }


def test_the_threshold_label_follows_the_mode(win, qapp):
    from fastfuncstuff.viewer.vocab import SetMode

    assert win.thr_head.text() == "[T]HRESH"  # the key is written into the label
    win.refresh(win.session.do(SetMode("ica")))
    qapp.processEvents()
    # No decomposition here, so no computed layer -- the label stays generic
    # rather than claiming units it is not showing.
    assert win.thr_head.text() == "[T]HRESH"


# ---------------------------------------------------------------------------
# the mode's input row
# ---------------------------------------------------------------------------


def test_a_mode_that_reads_nothing_shows_no_input_row(win, qapp):
    """View has no input; a box saying "(auto: anat)" would invent a fit."""
    assert win.session.mode.name == "plain"
    assert not win.input_row_host.isVisible()


def test_the_input_row_lists_the_runs_and_names_the_default(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetMode

    win4d.refresh(win4d.session.do(SetMode("instacorr")))
    qapp.processEvents()
    assert win4d.input_row_host.isVisible()
    texts = [win4d.input_box.itemText(i) for i in range(win4d.input_box.count())]
    assert texts[0].startswith("(auto: bold.nii.gz")
    assert any("bold.nii.gz" in t for t in texts[1:])
    # The 3-D anatomical is not offered: InstaCorr cannot correlate it.
    assert not any("anat.nii.gz" in t for t in texts)


def test_picking_an_input_re_points_the_mode(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetMode

    win4d.refresh(win4d.session.do(SetMode("instacorr")))
    qapp.processEvents()
    row = next(i for i in range(1, win4d.input_box.count()) if win4d.input_box.itemData(i))
    key = win4d.input_box.itemData(row)
    win4d.input_box.setCurrentIndex(row)
    win4d.input_box.activated.emit(row)
    qapp.processEvents()
    assert win4d.session.state.input_key == key
    assert f"SET_INPUT {key}" in win4d.session.to_script()


def test_the_input_row_says_so_when_there_is_nothing_to_read(win, qapp):
    """An empty drop-down invites a click that cannot be answered."""
    from fastfuncstuff.viewer.vocab import SetMode

    win.refresh(win.session.do(SetMode("instacorr")))
    qapp.processEvents()
    assert win.input_row_host.isVisible()
    assert not win.input_box.isEnabled()
    assert "load" in win.input_box.currentText()


def test_an_unticked_run_is_still_offered_as_an_input(win4d, qapp):
    """The whole point: hiding a run is about the picture, not the data."""
    from fastfuncstuff.viewer.vocab import SetLayerVisible, SetMode

    run = next(ly for ly in win4d.session.state.layers if ly.n_volumes > 1)
    win4d.refresh(win4d.session.do(SetLayerVisible(run.key, on=False)))
    win4d.refresh(win4d.session.do(SetMode("instacorr")))
    qapp.processEvents()
    keys = [win4d.input_box.itemData(i) for i in range(win4d.input_box.count())]
    assert run.key in keys


# ---------------------------------------------------------------------------
# the pickers must agree with what is on screen
# ---------------------------------------------------------------------------


def test_a_fresh_read_loads_nothing(qapp, datadir):
    """Reading a directory fills the picker; it does not open anything."""
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    try:
        w.read_directory(datadir)
        qapp.processEvents()
        assert w.data_box.currentIndex() == 0
        assert w.data_box.count() == 3  # the prompt plus two datasets
        assert len(session.state.layers) == 0
    finally:
        w.close()


def test_picking_a_dataset_loads_it_on_top(qapp, datadir):
    """One verb. Picking twice stacks two layers rather than replacing one."""
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    try:
        w.read_directory(datadir)
        qapp.processEvents()
        for name in ("anat.nii.gz", "stats.nii.gz"):
            row = next(i for i in range(w.data_box.count()) if name in w.data_box.itemText(i))
            w.data_box.setCurrentIndex(row)
            w.data_box.activated.emit(row)
            qapp.processEvents()
        assert [ly.name for ly in session.state.layers] == ["anat.nii.gz", "stats.nii.gz"]
        # The first one in defines the grid, because it is the bottom of the stack.
        assert session.state.grid.shape == session.state.layers.layers[0].shape
        assert session.state.selected == session.state.layers.layers[-1].key
    finally:
        w.close()


def test_the_picker_offers_only_files_on_disk(win4d, qapp):
    """It says what LOAD will open, so a derived layer has no row in it.

    The old picker mirrored the stack and grew a row per file-less layer, which
    offered to re-open something that was never on disk.
    """
    texts = [win4d.data_box.itemText(i) for i in range(win4d.data_box.count())]
    assert all(t.startswith("(") or ".nii" in t for t in texts)
    assert not any("C1" in t for t in texts)


def test_picker_rows_are_two_columns(win):
    """Dimensions and volume count must be scannable, not truncated."""
    row = win.data_box.itemText(1)
    assert "anat.nii.gz" in row
    assert "10x12x8" in row


def test_pickers_are_wide_enough_for_their_widest_row(win):
    """Qt sizes a combo to its current item, which truncates the rest on macOS."""
    from PySide6 import QtGui

    metrics = QtGui.QFontMetrics(win.data_box.font())
    widest = max(
        metrics.horizontalAdvance(win.data_box.itemText(i)) for i in range(win.data_box.count())
    )
    assert win.data_box.minimumWidth() >= widest


# ---------------------------------------------------------------------------
# range, opacity, colour bar
# ---------------------------------------------------------------------------


def test_min_and_max_are_independently_settable(win, qapp):
    """A stats map is routinely asymmetric; one number must not set both."""
    win.layer_list.setCurrentRow(0)
    win.min_spin.setValue(-2.5)
    win.max_spin.setValue(7.0)
    qapp.processEvents()
    layer = win.session.state.layers.overlay
    assert (layer.range_lo, layer.range_hi) == (-2.5, 7.0)


def test_opacity_slider_applies(win, qapp):
    win.layer_list.setCurrentRow(0)
    win.opacity_slider.setValue(40)
    qapp.processEvents()
    assert win.session.state.layers.overlay.opacity == pytest.approx(0.4)
    assert win.opacity_label.text() == "40%"


def test_autorange_rederives_from_the_data(win, qapp):
    win.layer_list.setCurrentRow(0)
    win.min_spin.setValue(-99.0)
    qapp.processEvents()
    win.autorange_button.click()
    qapp.processEvents()
    assert win.session.state.layers.overlay.range_lo > -99.0


def test_the_colour_bar_tracks_the_selected_layer(win, qapp):
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    layer = win.session.state.layers.overlay
    assert win.colorbar._lut_name == layer.colormap
    assert win.colorbar._threshold == layer.threshold


# ---------------------------------------------------------------------------
# the bar is a view of the layer, so it must follow every aspect that changes it
# ---------------------------------------------------------------------------


def test_the_bar_follows_a_colormap_change(win, qapp):
    """It listened only for LAYERS, so a colour change left it showing the old scale."""
    from fastfuncstuff.viewer.vocab import SetColormap

    win.layer_list.setCurrentRow(0)
    key = win.session.state.layers.overlay.key
    win.refresh(win.session.do(SetColormap(key, "viridis")))
    qapp.processEvents()
    assert win.colorbar._lut_name == "viridis"


def test_the_bar_follows_a_threshold_change(win, qapp):
    from fastfuncstuff.viewer.vocab import SetThreshold

    win.layer_list.setCurrentRow(0)
    key = win.session.state.layers.overlay.key
    win.refresh(win.session.do(SetThreshold(key, 1.25)))
    qapp.processEvents()
    assert win.colorbar._threshold == pytest.approx(1.25)
    assert win.thr_spin.value() == pytest.approx(1.25)


def test_the_bar_follows_a_range_change(win, qapp):
    from fastfuncstuff.viewer.vocab import SetRange

    win.layer_list.setCurrentRow(0)
    key = win.session.state.layers.overlay.key
    win.refresh(win.session.do(SetRange(key, -3.0, 12.0)))
    qapp.processEvents()
    assert win.colorbar._hi == pytest.approx(12.0)
    assert win.max_spin.value() == pytest.approx(12.0)


def test_typing_a_threshold_applies_it(win, qapp):
    win.layer_list.setCurrentRow(0)
    win.thr_spin.setValue(0.8)
    qapp.processEvents()
    assert win.session.state.layers.overlay.threshold == pytest.approx(0.8)


def test_clicking_the_bar_sets_the_threshold(win, qapp):
    """The bar is a control, not just a picture."""
    win.layer_list.setCurrentRow(0)
    win.colorbar.clicked.emit(2.5)
    qapp.processEvents()
    assert win.session.state.layers.overlay.threshold == pytest.approx(2.5)


def test_the_threshold_slider_spans_the_larger_half_of_the_range(win, qapp):
    """A one-sided map must not waste half its travel on values it never shows."""
    win.layer_list.setCurrentRow(0)
    win.min_spin.setValue(-1.0)
    win.max_spin.setValue(10.0)
    qapp.processEvents()
    win.thr_slider.setValue(1000)
    qapp.processEvents()
    assert win.session.state.layers.overlay.threshold == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# preparation must never run on the GUI thread
# ---------------------------------------------------------------------------


def test_the_window_tells_the_session_to_defer_preparation(win):
    assert win.session.defer_mode_preparation


def test_switching_modes_does_not_prepare_inline(win, qapp):
    """The freeze: seconds of filtering inside a click handler."""
    from fastfuncstuff.viewer.vocab import SetMode

    win.refresh(win.session.do(SetMode("instacorr")))
    qapp.processEvents()
    assert win.session.mode.needs_prepare


def test_optional_controls_render_as_a_checkbox_and_a_slider(win, qapp):
    win._switch_mode("instacorr")
    qapp.processEvents()
    row = win.mode_panel._widgets["blur"]
    assert row.findChild(QtWidgets.QCheckBox) is not None
    assert row.findChild(QtWidgets.QSlider) is not None


def test_a_disabled_optional_control_reads_as_off(win, qapp):
    """Off and 'set to zero' must not look the same."""
    win._switch_mode("instacorr")
    qapp.processEvents()
    row = win.mode_panel._widgets["blur"]
    assert not row.findChild(QtWidgets.QCheckBox).isChecked()
    assert not row.findChild(QtWidgets.QSlider).isEnabled()
    assert row.findChild(QtWidgets.QLabel).text() == "off"


def test_the_progress_bar_is_hidden_when_idle(win):
    assert not win.progress.isVisible()


# ---------------------------------------------------------------------------
# sub-brick labels
#
# ffs and 3dDeconvolve both write them, io/headers.py reads them, and until now
# nothing displayed them -- so a stats bucket read as "volume 3" and you had to
# go and run 3dinfo to find out which contrast that was.
# ---------------------------------------------------------------------------


def _labelled_bucket(path, labels):
    """A 4-D NIfTI carrying AFNI BRICK_LABS, the way a stats bucket does."""
    rng = np.random.default_rng(11)
    data = rng.normal(size=(10, 12, 8, len(labels))).astype(np.float32)
    img = nib.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0]))
    payload = ("BRICK_LABS=" + "~".join(labels) + "\x00").encode()
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, payload))
    nib.save(img, str(path))
    return path


@pytest.fixture
def winstats(win, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import SetOverlay

    path = _labelled_bucket(
        tmp_path / "stats.nii.gz", ["Full_Fstat", "Faces#0_Coef", "Faces#0_Tstat"]
    )
    win.refresh(win.session.do(SetOverlay(str(path))))
    qapp.processEvents()
    return win


def test_a_stats_bucket_keeps_its_sub_brick_labels(winstats):
    layer = winstats.session.state.layers.overlay
    assert layer.labels == ("Full_Fstat", "Faces#0_Coef", "Faces#0_Tstat")
    assert layer.sub_brick(1) == "#1 Faces#0_Coef"


def test_the_sub_brick_picker_lists_the_labels(winstats, qapp):
    winstats.layer_list.setCurrentRow(0)
    qapp.processEvents()
    assert winstats.brick_box.isVisible()
    listed = [winstats.brick_box.itemText(i) for i in range(winstats.brick_box.count())]
    assert listed == ["#0 Full_Fstat", "#1 Faces#0_Coef", "#2 Faces#0_Tstat"]


def test_picking_a_sub_brick_changes_what_is_displayed(winstats, qapp):
    winstats.layer_list.setCurrentRow(0)
    qapp.processEvents()
    winstats.brick_box.setCurrentIndex(2)
    winstats.brick_box.activated.emit(2)
    qapp.processEvents()
    assert winstats.session.state.layers.overlay.volume_index == 2


def test_the_threshold_can_read_a_different_sub_brick(winstats, qapp):
    """Colour by the coefficient, threshold on its t -- the stats case."""
    winstats.layer_list.setCurrentRow(0)
    qapp.processEvents()
    winstats.thrbrick_box.setCurrentIndex(2)
    winstats.thrbrick_box.activated.emit(2)
    qapp.processEvents()
    assert winstats.session.state.layers.overlay.threshold_index == 2


def test_the_readout_names_the_sub_brick(winstats, qapp):
    winstats.refresh(Aspect.SLICES)
    qapp.processEvents()
    assert "Full_Fstat" in winstats.value_label.text()


def test_a_time_series_offers_no_sub_brick_picker(win4d, qapp):
    """Its sub-bricks are time points; the T control already steps them."""
    win4d.layer_list.setCurrentRow(0)
    qapp.processEvents()
    assert not win4d.brick_box.isVisible()


# ---------------------------------------------------------------------------
# keyboard help
# ---------------------------------------------------------------------------


def test_h_opens_a_shortcut_list_for_the_window(win, qapp):
    win.help.toggle()
    qapp.processEvents()
    assert win.help._dialog.isVisible()
    assert "nexus" in win.help._dialog.windowTitle()


def test_h_toggles_the_list_closed(win, qapp):
    win.help.toggle()
    qapp.processEvents()
    win.help.toggle()
    qapp.processEvents()
    assert not win.help._dialog.isVisible()


def test_every_listed_key_is_grouped_and_described(win):
    for binding in win.help._bindings:
        assert binding.description, f"{binding.keys} has no description"
        assert binding.group, f"{binding.keys} has no group"


def test_the_help_lists_the_keys_that_are_actually_installed(win):
    """One table drives both, so documented and installed cannot drift apart."""
    from PySide6 import QtGui

    installed = {
        a.shortcut().toString().lower() for a in win.actions() if not a.shortcut().isEmpty()
    }
    for binding in win.help._bindings:
        if binding.action is None:
            continue  # mouse gestures are listed but not shortcuts
        spelling = QtGui.QKeySequence(binding.keys).toString().lower()
        assert spelling in installed, f"{binding.keys} is listed but not installed"


def test_a_graph_window_has_its_own_keys(win, qapp):
    graph = open_graph(win, qapp)
    assert graph.help._bindings
    assert {b.group for b in graph.help._bindings} != {b.group for b in win.help._bindings}


def test_an_image_window_has_its_own_keys(win):
    image = image_of(win, Plane.AXIAL)
    assert image.help._bindings
    assert "o" in {b.keys for b in image.help._bindings}


# ---------------------------------------------------------------------------
# the graph must not lose the trace the map was computed from
# ---------------------------------------------------------------------------


def test_the_graph_keeps_the_source_trace_after_a_seed(win, qapp, tmp_path):
    """The map displaces its input from the stack; the trace must survive."""
    from fastfuncstuff.viewer.vocab import SetIJK, SetMode, SetOverlay, SetSeed

    rng = np.random.default_rng(41)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    img = nib.Nifti1Image(rng.normal(size=(8, 9, 7, 30)).astype(np.float32), aff)
    img.header["pixdim"][4] = 2.0
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "bold.nii.gz"))

    win.refresh(win.session.do(SetOverlay(str(tmp_path / "bold.nii.gz"))))
    win.session.store.ensure_ram(win.session.state.layers.overlay.key)
    graph = open_graph(win, qapp)

    win.refresh(win.session.do(SetMode("instacorr")))
    assert win.session.mode.prepare()
    win.refresh(win.session.do(SetSeed(3, 4, 3)))
    win.refresh(win.session.do(SetIJK(3, 4, 3)))
    qapp.processEvents()

    labels = [t[0] for t in graph.graph._cells[0].traces]
    assert "mode:source" in labels, f"the correlated time course vanished: {labels}"
    assert "mode:prepared" in labels


# ---------------------------------------------------------------------------
# the crosshair decides which slice each pane shows
#
# These pin a cluster that all had one cause: SET_IJK dirties CROSSHAIR, and
# the pane redraw did not listen for it. Only the drawn crosshair lines moved,
# so panes sat on stale slices until something else forced a full redraw --
# which is why stepping time appeared to make everything "jump".
# ---------------------------------------------------------------------------


def _positions(win):
    return {p.value: image_of(win, p).pane.position for p in Plane}


def _click(win, qapp, plane, row, col, *, seed=False):
    image_of(win, plane)._pick(row, col, seed=seed)
    qapp.processEvents()


def _pick_and_expect(win, qapp, plane, row, col):
    """Click a window, then say where the other two should now be sitting."""
    from fastfuncstuff.viewer.slicing import plane_layout

    grid = win.session.state.grid
    layout = plane_layout(grid.affine, plane)
    ijk = layout.to_ijk(row, col, win.session.state.crosshair, grid.shape)
    _click(win, qapp, plane, row, col)
    assert win.session.state.crosshair == ijk
    return {p.value: ijk[plane_layout(grid.affine, p).fixed] for p in Plane}


def test_clicking_one_pane_reslices_the_others(win, qapp):
    expected = _pick_and_expect(win, qapp, Plane.AXIAL, 4, 5)
    assert _positions(win) == expected


def test_a_click_does_not_move_the_pane_that_was_clicked(win, qapp):
    """Clicking axial changes i and j, not k -- that pane's slice is unchanged."""
    before = _positions(win)["axial"]
    _click(win, qapp, Plane.AXIAL, 3, 6)
    assert _positions(win)["axial"] == before


def test_every_pane_can_drive_the_others(win, qapp):
    """The inconsistency was per-pane, so each one needs checking."""
    for plane, row, col in (
        (Plane.SAGITTAL, 6, 3),
        (Plane.CORONAL, 2, 4),
        (Plane.AXIAL, 5, 1),
    ):
        expected = _pick_and_expect(win, qapp, plane, row, col)
        assert _positions(win) == expected, plane


def test_clicks_survive_a_flipped_grid(win, qapp):
    """A click must land on the voxel under the cursor, whatever the storage."""
    from fastfuncstuff.viewer.slicing import plane_layout

    grid = win.session.state.grid
    for plane in Plane:
        layout = plane_layout(grid.affine, plane)
        h, w = grid.shape[layout.row], grid.shape[layout.col]
        _click(win, qapp, plane, 0, 0)
        corner = win.session.state.crosshair
        _click(win, qapp, plane, h - 1, w - 1)
        assert win.session.state.crosshair != corner, plane


def test_scrolling_reslices_the_scrolled_pane(win, qapp):
    before = _positions(win)["axial"]
    image_of(win, Plane.AXIAL)._step(1)
    qapp.processEvents()
    assert _positions(win)["axial"] == before + 1


def test_stepping_time_does_not_move_any_slice(win, qapp, tmp_path):
    """The 'everything jumps' symptom was panes catching up on a forced redraw."""
    from fastfuncstuff.viewer.vocab import SetOverlay

    rng = np.random.default_rng(53)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    img = nib.Nifti1Image(rng.normal(size=(10, 12, 8, 20)).astype(np.float32), aff)
    img.header["pixdim"][4] = 2.0
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "bold.nii.gz"))
    win.refresh(win.session.do(SetOverlay(str(tmp_path / "bold.nii.gz"))))
    _click(win, qapp, Plane.AXIAL, 3, 4)

    before = _positions(win)
    win._step_time(1)
    qapp.processEvents()
    assert _positions(win) == before


def test_a_seed_click_moves_the_crosshair_with_it(win, qapp):
    """Leaving the crosshair behind makes the graph describe another voxel."""
    _click(win, qapp, Plane.CORONAL, 5, 3, seed=True)
    assert win.session.state.seed == win.session.state.crosshair


def test_a_seed_click_records_both_commands(win, qapp):
    """SET_SEED stays a primitive; the UI expresses the gesture as two."""
    win.session.bus.clear_log()
    _click(win, qapp, Plane.AXIAL, 2, 3, seed=True)
    assert [c.name for c in win.session.bus.log] == ["SET_IJK", "SET_SEED"]


# ---------------------------------------------------------------------------
# choosing a timepoint
# ---------------------------------------------------------------------------


@pytest.fixture
def win4d(win, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import SetOverlay

    rng = np.random.default_rng(61)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    img = nib.Nifti1Image(rng.normal(size=(10, 12, 8, 25)).astype(np.float32), aff)
    img.header["pixdim"][4] = 2.0
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "bold.nii.gz"))
    win.refresh(win.session.do(SetOverlay(str(tmp_path / "bold.nii.gz"))))
    win.session.store.ensure_ram(win.session.state.layers.overlay.key)
    qapp.processEvents()
    return win


def test_the_time_readout_is_editable(win4d, qapp):
    win4d.time_spin.setValue(17)
    qapp.processEvents()
    assert win4d.session.state.time_index == 17


def test_the_time_readout_is_bounded_by_the_data(win4d, qapp):
    assert win4d.time_spin.maximum() == 24
    assert win4d.time_spin.isEnabled()


def test_the_time_readout_is_disabled_without_a_time_series(win, qapp):
    assert not win.time_spin.isEnabled()


def test_clicking_the_graph_jumps_to_that_volume(win4d, qapp):
    graph = open_graph(win4d, qapp)
    graph.graph.scrubbed.emit(12)
    qapp.processEvents()
    assert win4d.session.state.time_index == 12
    assert win4d.time_spin.value() == 12


def test_the_time_readout_follows_playback(win4d, qapp):
    win4d._step_time(3)
    qapp.processEvents()
    assert win4d.time_spin.value() == win4d.session.state.time_index


# ---------------------------------------------------------------------------
# shortcuts must not receive QAction's `checked` argument
# ---------------------------------------------------------------------------


def test_a_binding_that_captures_a_loop_variable_keeps_it(win, qapp):
    """QAction.triggered passes a bool to any slot that will take one.

    The natural way to capture a loop variable is `lambda p=plane: ...`, whose
    arity is one -- so PySide6 handed it False and the plane became the string
    "False". Pressing 2 in an image window did nothing at all.

    Triggered through the installed QAction rather than a synthetic key event,
    because that is the object that supplies the spurious argument.
    """
    image = image_of(win, Plane.AXIAL)
    action = next(a for a in image.actions() if a.shortcut().toString() == "2")
    action.trigger()
    qapp.processEvents()
    assert win.session.state.viewports.get(image.vid).plane is Plane.SAGITTAL


# ---------------------------------------------------------------------------
# a window has to be able to get small
# ---------------------------------------------------------------------------


def test_an_image_window_sheds_its_header_as_it_narrows(win, qapp):
    """The header's own width was the floor under the whole window."""
    image = image_of(win, Plane.AXIAL)
    image.show()
    image.resize(420, 420)
    qapp.processEvents()
    assert image.header.isVisible()
    assert image.solo_button.text() == "S[O]LO"

    image.resize(240, 240)
    qapp.processEvents()
    assert image.header.isVisible()
    assert image.solo_button.text() == "[o]", "compact buttons keep only the key"

    image.resize(120, 120)
    qapp.processEvents()
    assert not image.header.isVisible()


def test_an_image_window_can_be_made_tiny(win, qapp):
    """A wall of small images is a real way to look at data.

    The header's layout still *hints* at a wide minimum, and Qt applies that
    hint before any resize happens -- so the window refused to narrow, which
    meant the resize that would have collapsed the header never fired. An
    explicit minimum is what breaks that circle.
    """
    image = image_of(win, Plane.AXIAL)
    image.show()
    qapp.processEvents()
    assert image.minimumSize().width() <= 100
    image.resize(90, 90)
    qapp.processEvents()
    assert image.width() == 90


# ---------------------------------------------------------------------------
# the crosshair says what the graph is reading
# ---------------------------------------------------------------------------


def test_the_crosshair_opens_to_the_graphs_footprint(win, qapp):
    from fastfuncstuff.viewer.vocab import SetViewGrid

    axial = image_of(win, Plane.AXIAL)
    assert axial.pane._coverage == [], "no graph open, no box"

    graph = open_graph(win, qapp, Plane.AXIAL)
    win.refresh(win.session.do(SetViewGrid(graph.vid, 5)))
    qapp.processEvents()
    (row, col, n_rows, n_cols) = axial.pane._coverage[0]
    assert (n_rows, n_cols) == (5, 5)

    # Centred the way the graph walks it: from -half, inclusive.
    cross_row, cross_col = axial.pane._cross
    assert (row, col) == (cross_row - 2, cross_col - 2)


def test_the_footprint_follows_the_grid_size(win, qapp):
    from fastfuncstuff.viewer.vocab import SetViewGrid

    axial = image_of(win, Plane.AXIAL)
    graph = open_graph(win, qapp, Plane.AXIAL)
    for n in (1, 3, 8, 16):
        win.refresh(win.session.do(SetViewGrid(graph.vid, n)))
        qapp.processEvents()
        assert axial.pane._coverage[0][2:] == (n, n)


def test_only_the_graphs_own_plane_gets_a_box(win, qapp):
    """An axial block is one slice as far as sagittal is concerned."""
    open_graph(win, qapp, Plane.AXIAL)
    qapp.processEvents()
    assert image_of(win, Plane.AXIAL).pane._coverage != []
    assert image_of(win, Plane.SAGITTAL).pane._coverage == []


def test_two_graphs_of_different_sizes_read_as_nested_boxes(win, qapp):
    from fastfuncstuff.viewer.vocab import SetViewGrid

    a = open_graph(win, qapp, Plane.AXIAL)
    b = open_graph(win, qapp, Plane.AXIAL)
    win.refresh(win.session.do(SetViewGrid(a.vid, 3)))
    win.refresh(win.session.do(SetViewGrid(b.vid, 9)))
    qapp.processEvents()
    sizes = sorted(box[2] for box in image_of(win, Plane.AXIAL).pane._coverage)
    assert sizes == [3, 9]


# ---------------------------------------------------------------------------
# light mode
# ---------------------------------------------------------------------------


def test_one_switch_flips_the_whole_interface(win, qapp):
    from fastfuncstuff.viewer.ui import theme

    assert win.session.state.theme == "light"
    win._toggle_theme()
    qapp.processEvents()
    assert win.session.state.theme == "dark"
    assert theme.palette().name == "dark"

    # Every window, not just the one the button is on.
    dark = theme.DARK.bg
    assert dark in win.styleSheet()
    assert all(dark in w.styleSheet() for w in win.manager.windows.values())

    win._toggle_theme()
    qapp.processEvents()
    assert theme.palette().name == "light"


def test_the_theme_button_names_where_it_takes_you(win, qapp):
    """A button labelled with the current state reads as a status light."""
    # The key is bracketed inside the word when the word contains it.
    assert win.theme_button.text() == "[D]ARK"
    win._toggle_theme()
    qapp.processEvents()
    assert win.theme_button.text() == "LIGHT [d]"
    win._toggle_theme()
    qapp.processEvents()


def test_a_window_opened_after_the_switch_is_born_dark(win, qapp):
    from fastfuncstuff.viewer.ui import theme

    win._toggle_theme()
    qapp.processEvents()
    win._new_graph()
    qapp.processEvents()
    graph = win.manager.windows[win.session.state.viewports.graphs[0].id]
    assert theme.DARK.bg in graph.styleSheet()
    win._toggle_theme()
    qapp.processEvents()


def test_the_palette_is_recorded_so_a_replay_looks_the_same(win, qapp):
    win.session.bus.clear_log()
    win._toggle_theme()
    qapp.processEvents()
    assert "SET_THEME dark" in win.session.to_script()
    win._toggle_theme()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# denoise, as a mode
# ---------------------------------------------------------------------------


def _denoise_mode(win4d, qapp, polort="2"):
    from fastfuncstuff.viewer.vocab import SetMode

    src = win4d.session.state.layers.overlay.key
    win4d.session.store.ensure_ram(src)
    win4d._switch_mode("denoise")
    win4d._mode_param_changed("polort", polort)
    qapp.processEvents()
    assert win4d.session.mode.name == "denoise" and SetMode
    return src


def _apply(win4d, qapp):
    win4d.mode_panel._widgets["action:apply"].click()
    assert win4d.runner.wait(20_000)
    qapp.processEvents()
    qapp.processEvents()


def test_denoise_is_a_mode_with_an_apply_button(win4d, qapp):
    _denoise_mode(win4d, qapp)
    assert {"matrix", "polort", "keep_mean", "action:apply", "action:carpets"} <= set(
        win4d.mode_panel._widgets
    )
    assert not hasattr(win4d, "denoise_button"), "DERIVE left the controller"


def test_apply_runs_off_the_gui_thread_and_leaves_a_run_and_a_map(win4d, qapp):
    src = _denoise_mode(win4d, qapp)
    _apply(win4d, qapp)
    stack = win4d.session.state.layers
    derived = stack.find_by_source(f"derived:denoise:{src}")
    assert derived is not None and derived.name == "A_DENOISE"
    assert derived.n_volumes == stack.get(src).n_volumes
    vr = stack.find_by_source("mode:denoise")
    assert vr is not None and vr.name == "A_DENOISE_VR"
    assert win4d.session.state.selected == vr.key
    removed = win4d.session.volume(vr.key, 0)
    assert np.all((removed >= 0) & (removed <= 1)) and removed.max() > 0
    assert "MODE_ACTION apply" in win4d.session.to_script()


def test_a_parameter_change_does_not_run_the_projection(win4d, qapp):
    """Seconds of work per spin-box click is the freeze APPLY exists to avoid."""
    _denoise_mode(win4d, qapp)
    before = len(win4d.session.state.layers)
    win4d._mode_param_changed("polort", "3")
    qapp.processEvents()
    assert not win4d.runner.busy
    assert len(win4d.session.state.layers) == before


def test_applying_again_replaces_rather_than_chaining(win4d, qapp):
    """Applying selects the map, and a selected A_DENOISE must not become the
    input of the next APPLY."""
    src = _denoise_mode(win4d, qapp, polort="1")
    _apply(win4d, qapp)
    after_one = len(win4d.session.state.layers)
    derived = win4d.session.state.layers.find_by_source(f"derived:denoise:{src}")
    from fastfuncstuff.viewer.vocab import SelectLayer

    win4d._dispatch(SelectLayer(derived.key))
    win4d._mode_param_changed("polort", "3")
    _apply(win4d, qapp)
    assert len(win4d.session.state.layers) == after_one
    assert win4d.session.state.layers.find_by_source(f"derived:denoise:{derived.key}") is None


def test_carpets_opens_raw_and_denoised_side_by_side(win4d, qapp):
    src = _denoise_mode(win4d, qapp)
    _apply(win4d, qapp)
    win4d.mode_panel._widgets["action:carpets"].click()
    for _ in range(4):
        win4d.runner.wait(20_000)
        qapp.processEvents()
    carpets = win4d.session.state.viewports.carpets
    assert len(carpets) == 2
    derived = win4d.session.state.layers.find_by_source(f"derived:denoise:{src}")
    assert [v.traces for v in carpets] == [(src,), (derived.key,)]
    windows = win4d.manager.carpets()
    assert all(w.view._carpet is not None for w in windows), "the second carpet was never built"


def test_leaving_denoise_keeps_what_it_made(win4d, qapp):
    src = _denoise_mode(win4d, qapp)
    _apply(win4d, qapp)
    win4d._switch_mode("plain")
    qapp.processEvents()
    assert win4d.session.state.layers.find_by_source(f"derived:denoise:{src}") is not None
    assert win4d.session.state.layers.find_by_source("mode:denoise") is not None


# ---------------------------------------------------------------------------
# reordering and removing layers
#
# MOVE_LAYER and REMOVE_LAYER were registered commands with no UI at all, so
# the stack could only grow and a derived layer could not actually be promoted
# to underlay despite that being the point of making it. Neither handler was
# reached by a test either, which is how one of them stayed unregistered
# through a green run.
# ---------------------------------------------------------------------------


def test_raise_and_lower_walk_the_stack(win, qapp):
    before = list(win.session.state.layers.keys)
    win.layer_list.setCurrentRow(0)  # the top layer
    qapp.processEvents()
    win._reorder(-1)
    qapp.processEvents()
    assert win.session.state.layers.keys == before[::-1]
    win._reorder(1)
    qapp.processEvents()
    assert win.session.state.layers.keys == before


def test_reordering_past_either_end_does_nothing(win, qapp):
    before = list(win.session.state.layers.keys)
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    win._reorder(1)  # already on top
    qapp.processEvents()
    assert win.session.state.layers.keys == before


def test_promoting_to_underlay_brings_its_grid_with_it(win, qapp, tmp_path):
    """The underlay defines the display grid, however it got to the bottom."""
    from fastfuncstuff.viewer.vocab import AddOverlay

    aff = np.diag([6.0, 6.0, 6.0, 1.0])
    coarse = tmp_path / "coarse.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 4), np.float32), aff), str(coarse))
    win.refresh(win.session.do(AddOverlay(str(coarse))))
    qapp.processEvents()
    assert win.session.state.grid.shape == (10, 12, 8)  # the anat's

    win.layer_list.setCurrentRow(0)  # the coarse one, on top
    qapp.processEvents()
    win._make_underlay()
    qapp.processEvents()
    assert win.session.state.layers.base.name == "coarse.nii.gz"
    assert win.session.state.grid.shape == (5, 6, 4)


def test_dropping_a_layer_releases_its_voxels(win, qapp):
    key = win.session.state.layers.overlay.key
    assert key in win.session.store.keys()
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    win._drop_layer()
    qapp.processEvents()
    assert win.session.state.layers.find(key) is None
    assert key not in win.session.store.keys()


def test_the_last_layer_cannot_be_dropped(win, qapp):
    """A viewer showing nothing is a state whose only way out is undo."""
    while len(win.session.state.layers) > 1:
        win.layer_list.setCurrentRow(0)
        win._drop_layer()
        qapp.processEvents()
    win.layer_list.setCurrentRow(0)
    win._drop_layer()
    qapp.processEvents()
    assert len(win.session.state.layers) == 1


def test_dropping_the_underlay_regrids_onto_what_is_left(win, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import AddOverlay

    aff = np.diag([6.0, 6.0, 6.0, 1.0])
    coarse = tmp_path / "coarse.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 4), np.float32), aff), str(coarse))
    win.refresh(win.session.do(AddOverlay(str(coarse))))
    win.layer_list.setCurrentRow(0)
    win._make_underlay()
    qapp.processEvents()
    assert win.session.state.grid.shape == (5, 6, 4)

    win.layer_list.setCurrentRow(win.layer_list.count() - 1)  # the underlay
    qapp.processEvents()
    win._drop_layer()
    qapp.processEvents()
    assert win.session.state.grid.shape == win.session.state.layers.base.shape


def test_the_stack_gestures_go_through_the_bus(win, qapp):
    """Recorded, so a rearranged stack comes back from a replay."""
    win.session.bus.clear_log()
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    win._reorder(-1)
    win._drop_layer()
    qapp.processEvents()
    names = [c.name for c in win.session.bus.log]
    assert "MOVE_LAYER" in names and "REMOVE_LAYER" in names


# ---------------------------------------------------------------------------
# zoom and pan, per window
# ---------------------------------------------------------------------------


def test_zooming_crops_the_rendered_image(win, qapp):
    image = image_of(win, Plane.AXIAL)
    image.redraw()
    before = image.pane._image.width()
    image._zoom_by(2.0)
    qapp.processEvents()
    assert image.pane._image.width() < before
    assert image.pane._zoomed


def test_a_click_still_lands_on_the_voxel_under_it_when_zoomed(win, qapp):
    """The hit test has to apply the crop offset the drawing applied."""
    image = image_of(win, Plane.AXIAL)
    image._zoom_by(2.0)
    qapp.processEvents()
    image.redraw()
    row, col = image.pane._cross
    before = win.session.state.crosshair
    image.pane.picked.emit(row, col)
    qapp.processEvents()
    assert win.session.state.crosshair == before


def test_zoom_is_per_window(win, qapp):
    a = image_of(win, Plane.AXIAL)
    b = image_of(win, Plane.CORONAL)
    a._zoom_by(2.0)
    qapp.processEvents()
    assert win.session.state.viewports.get(a.vid).zoom > 1.0
    assert win.session.state.viewports.get(b.vid).zoom == 1.0


def test_reset_brings_the_whole_plane_back(win, qapp):
    image = image_of(win, Plane.AXIAL)
    image._zoom_by(2.0)
    image._pan_by(3.0, 2.0)
    qapp.processEvents()
    image._reset_view()
    qapp.processEvents()
    viewport = win.session.state.viewports.get(image.vid)
    assert (viewport.zoom, viewport.pan) == (1.0, (0.0, 0.0))
    image.redraw()
    assert not image.pane._zoomed


def test_zoom_and_pan_are_recorded(win, qapp):
    win.session.bus.clear_log()
    image = image_of(win, Plane.AXIAL)
    image._zoom_by(2.0)
    image._pan_by(1.0, 1.0)
    qapp.processEvents()
    names = [c.name for c in win.session.bus.log]
    assert "SET_ZOOM" in names and "SET_PAN" in names


# ---------------------------------------------------------------------------
# the carpet window
# ---------------------------------------------------------------------------


def _carpet(win4d, qapp):
    win4d.session.store.ensure_ram(win4d.session.state.layers.overlay.key)
    win4d.layer_list.setCurrentRow(0)
    qapp.processEvents()
    win4d._new_carpet()
    assert win4d.runner.wait(30_000)
    qapp.processEvents()
    return win4d.manager.carpets()[0]


def test_a_carpet_opens_as_its_own_window(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    assert carpet.isWindow()
    assert win4d.session.state.viewports.get(carpet.vid).kind.value == "carpet"
    assert carpet.view._image is not None


def test_the_carpet_draws_one_column_per_volume(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    assert carpet.view._image.width() == win4d.session.state.layers.overlay.n_volumes


def test_changing_the_order_rebuilds_it(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetCarpetOrder

    carpet = _carpet(win4d, qapp)
    before = carpet.info.text()
    win4d._dispatch(SetCarpetOrder(carpet.vid, "voxel"))
    win4d.runner.wait(30_000)
    qapp.processEvents()
    assert carpet.info.text() != before
    assert "acquisition order" in carpet.info.text()


def test_an_unknown_order_is_refused(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetCarpetOrder

    carpet = _carpet(win4d, qapp)
    with pytest.raises(KeyError):
        win4d.session.do(SetCarpetOrder(carpet.vid, "spiral"))


def test_a_layer_change_marks_it_stale_rather_than_rebuilding(win4d, qapp, tmp_path):
    """Seconds of work; rebuilding on a threshold drag would be unusable.

    But a sidebar drawn from an overlay that has since been swapped is a stale
    widget with no numbers on it to contradict, so it has to say so.
    """
    from fastfuncstuff.viewer.vocab import AddOverlay

    carpet = _carpet(win4d, qapp)
    assert "stale" not in carpet.info.text()

    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    other = tmp_path / "other.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 12, 8), np.float32), aff), str(other))
    win4d.refresh(win4d.session.do(AddOverlay(str(other))))
    qapp.processEvents()
    assert "stale" in carpet.info.text()


def test_the_carpet_scrubs_time_like_a_graph(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    carpet.scrubbed.emit(9)
    qapp.processEvents()
    assert win4d.session.state.time_index == 9


def test_the_carpet_is_recorded_and_replayable(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    script = win4d.session.to_script()
    assert f"OPEN_VIEW {carpet.vid} carpet" in script

    replayed = ViewerSession(device=CPU)
    try:
        replayed.run_script(script)
        assert replayed.state.viewports.get(carpet.vid).is_carpet
    finally:
        replayed.close()


def test_the_carpet_sheds_its_controls_when_narrow(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    carpet.resize(700, 400)
    qapp.processEvents()
    assert carpet.controls.isVisible()
    carpet.resize(150, 150)
    qapp.processEvents()
    assert not carpet.controls.isVisible()


# ---------------------------------------------------------------------------
# the keys have to actually fire
#
# Reported as "the shortcuts don't work". Two causes, both invisible in code:
# Qt does not distinguish case in a key sequence, so `d` and `D` registered two
# actions on one key and Qt fired neither; and clicking a layer moved focus
# into a QListWidget, whose type-to-search then ate every letter.
# ---------------------------------------------------------------------------


def test_no_two_shortcuts_share_a_key(win, qapp):
    """`d` and `D` are the same key to Qt. Five pairs were written that way."""
    from PySide6 import QtGui

    windows = [win, *win.manager.windows.values()]
    for window in windows:
        seen: dict[str, str] = {}
        for binding in window.help._bindings:
            if binding.action is None:
                continue
            key = QtGui.QKeySequence(binding.keys).toString()
            assert key not in seen, (
                f"{type(window).__name__}: {key!r} is both "
                f"{seen[key]!r} and {binding.description!r}"
            )
            seen[key] = binding.description


def test_installing_two_bindings_on_one_key_is_refused(qapp):
    from fastfuncstuff.viewer.ui.shortcuts import Binding, install

    widget = QtWidgets.QWidget()
    try:
        with pytest.raises(ValueError, match="two shortcuts"):
            install(
                widget,
                [
                    Binding("d", "dark / light", lambda: None),
                    Binding("D", "denoise", lambda: None),
                ],
            )
    finally:
        widget.deleteLater()


def test_clicking_a_layer_does_not_disable_the_keyboard(win, qapp):
    """The list's type-to-search was eating every letter after a click."""
    from PySide6 import QtCore

    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    assert win.layer_list.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    for box in (win.cmap_box, win.mode_box, win.data_box):
        assert box.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus


def test_text_entry_keeps_its_keys(win, qapp):
    """While you are typing a path, the letters belong to the line edit."""
    from PySide6 import QtCore, QtWidgets

    win._switch_mode("denoise")
    qapp.processEvents()
    edit = win.mode_panel._widgets["matrix"].findChild(QtWidgets.QLineEdit)
    assert edit is not None and edit.focusPolicy() != QtCore.Qt.FocusPolicy.NoFocus


def test_the_theme_key_fires(win, qapp):
    from PySide6 import QtGui

    action = next(a for a in win.actions() if a.shortcut() == QtGui.QKeySequence("d"))
    action.trigger()
    qapp.processEvents()
    assert win.session.state.theme == "dark"
    win._toggle_theme()
    qapp.processEvents()


def test_the_carpet_click_moves_in_space_as_well_as_time(win4d, qapp):
    """A carpet has two axes and both of them mean something outside it.

    Across is the volume, down is a voxel -- and a picture you cannot click
    your way back out of is the one place in the viewer where finding something
    does not tell you where it is.
    """
    carpet = _carpet(win4d, qapp)
    picture = carpet.view._carpet
    assert picture is not None and picture.voxels is not None

    row = picture.shape[0] // 3
    expected = picture.voxel_of(row)
    carpet.view.scrubbed.emit(7)
    carpet.view.rowed.emit(row)
    qapp.processEvents()

    assert win4d.session.state.time_index == 7
    assert win4d.session.state.crosshair == expected


def test_every_carpet_row_points_at_a_voxel_inside_the_mask(win4d, qapp):
    """The mapping is carried through the ordering and the binning, so the
    failure mode is a crosshair that lands *near* the truth rather than on it."""
    carpet = _carpet(win4d, qapp)
    picture = carpet.view._carpet
    nx, ny, nz = win4d.session.state.layers.overlay.shape
    for row in (0, picture.shape[0] // 2, picture.shape[0] - 1):
        i, j, k = picture.voxel_of(row)
        assert 0 <= i < nx and 0 <= j < ny and 0 <= k < nz


def _drag_rows(view, qapp, y0, y1):
    """Press, move and release on a carpet the way a hand would."""
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    x = view.width() // 2
    QTest.mousePress(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(x, y0))
    QTest.mouseMove(view, QtCore.QPoint(x, (y0 + y1) // 2))
    QTest.mouseMove(view, QtCore.QPoint(x, y1))
    QTest.mouseRelease(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(x, y1))
    qapp.processEvents()


def test_dragging_carpet_rows_makes_one_red_overlay_of_their_voxels(win4d, qapp):
    """The selection is a layer like any other, on top, selected, and alone:
    the voxels may be anywhere, and a stat map over them hides where."""
    carpet = _carpet(win4d, qapp)
    carpet.resize(600, 400)
    qapp.processEvents()
    stack = win4d.session.state.layers
    before = len(stack)
    others = [ly.key for ly in stack if ly.key != stack.base.key]

    view = carpet.view
    _drag_rows(view, qapp, 10, view.height() // 3)

    assert len(stack) == before + 1
    selection = stack.layers[-1]
    assert selection.source == f"selection:{carpet.vid}"
    assert selection.colormap == "red"
    assert win4d.session.state.selected == selection.key
    assert all(not stack.get(k).visible for k in others)
    assert stack.base.visible

    first, last = view._selection
    expected = view._carpet.mask_of_rows(first, last)
    shown = win4d.session.volume(selection.key, 0) > 0.5
    assert np.array_equal(shown, expected) and expected.any()


def test_a_new_drag_updates_the_same_overlay(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    carpet.resize(600, 400)
    qapp.processEvents()
    view = carpet.view
    _drag_rows(view, qapp, 10, 60)
    stack = win4d.session.state.layers
    count, key = len(stack), stack.layers[-1].key
    first_voxels = int((win4d.session.volume(key, 0) > 0.5).sum())

    _drag_rows(view, qapp, 10, view.height() - 10)
    assert len(stack) == count
    assert stack.layers[-1].key == key
    assert int((win4d.session.volume(key, 0) > 0.5).sum()) > first_voxels


def test_a_short_press_on_the_carpet_is_still_a_click(win4d, qapp):
    carpet = _carpet(win4d, qapp)
    carpet.resize(600, 400)
    qapp.processEvents()
    before = len(win4d.session.state.layers)
    _drag_rows(carpet.view, qapp, 100, 101)
    assert len(win4d.session.state.layers) == before
    assert carpet.view._selection is None


def test_a_selection_can_be_saved_and_read_back(win4d, qapp, tmp_path):
    carpet = _carpet(win4d, qapp)
    carpet.resize(600, 400)
    qapp.processEvents()
    _drag_rows(carpet.view, qapp, 10, 120)
    layer = win4d.session.state.layers.layers[-1]
    out = win4d.session.save_layer(layer.key, tmp_path / "picked.nii.gz")

    img = nib.load(str(out))
    assert img.shape == layer.shape
    assert np.allclose(img.affine, layer.affine)
    assert np.array_equal(np.asarray(img.dataobj) > 0.5, win4d.session.volume(layer.key, 0) > 0.5)


def test_a_selection_layer_never_enters_the_picker(win4d, qapp):
    """The picker used to grow a row per file-less layer, once per sync."""
    carpet = _carpet(win4d, qapp)
    carpet.resize(600, 400)
    qapp.processEvents()
    _drag_rows(carpet.view, qapp, 10, 120)
    stack = win4d.session.state.layers
    from fastfuncstuff.viewer.vocab import MoveLayer

    win4d._dispatch(MoveLayer(stack.layers[-1].key, 1))
    for _ in range(3):
        win4d.refresh(Aspect.LAYERS)
    name = stack.layers[1].name
    texts = [win4d.data_box.itemText(i) for i in range(win4d.data_box.count())]
    assert texts.count(name) == 0


# ---------------------------------------------------------------------------
# the matrix window
# ---------------------------------------------------------------------------


def _matrix(win4d, qapp):
    win4d.layer_list.setCurrentRow(0)
    qapp.processEvents()
    win4d._new_matrix()
    assert win4d.runner.wait(30_000)
    qapp.processEvents()
    from fastfuncstuff.viewer.ui.matrixwindow import MatrixWindow

    return next(w for w in win4d.manager.windows.values() if isinstance(w, MatrixWindow))


def _add_atlas(win4d, qapp, tmp_path):
    labels = np.zeros((10, 12, 8), dtype=np.float32)
    labels[:5, :6] = 1
    labels[5:, :6] = 2
    labels[:, 6:] = 3
    path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.diag([3.0, 3.0, 3.0, 1.0])), str(path))
    from fastfuncstuff.viewer.vocab import AddOverlay

    win4d.refresh(win4d.session.do(AddOverlay(str(path))))
    qapp.processEvents()
    return win4d.session.state.layers.layers[-1].key


def test_a_matrix_opens_as_its_own_window(win4d, qapp):
    matrix = _matrix(win4d, qapp)
    assert matrix.isWindow()
    assert win4d.session.state.viewports.get(matrix.vid).kind.value == "matrix"
    assert matrix.view._image is not None


def test_without_rois_the_matrix_is_voxel_bins_and_says_so(win4d, qapp):
    matrix = _matrix(win4d, qapp)
    assert "voxel bins" in matrix.info.text()
    assert matrix._matrix is not None and not matrix._matrix.from_rois


def test_dropping_an_atlas_on_the_stack_names_the_rows(win4d, qapp, tmp_path):
    """The point of the ROI layer: one load, and every matrix becomes a
    connectivity matrix without visiting a picker."""
    _add_atlas(win4d, qapp, tmp_path)
    matrix = _matrix(win4d, qapp)
    assert matrix._matrix.from_rois
    assert matrix._matrix.n_nodes == 3
    assert "3 ROIs" in matrix.info.text()


def test_the_triangle_decides_which_end_of_a_pair_a_click_goes_to(win4d, qapp, tmp_path):
    """Every cell is two nodes. Below the diagonal the click goes to the row's
    region, above it to the column's -- so both ends of an edge are one click
    away, on the two mirrored cells."""
    key = _add_atlas(win4d, qapp, tmp_path)
    matrix = _matrix(win4d, qapp)
    rois = win4d.session.roi_set(key)

    matrix.view.picked.emit(2, 0)  # lower triangle: the row
    qapp.processEvents()
    assert win4d.session.state.crosshair == rois.find(matrix._matrix.indices[2]).center_ijk

    matrix.view.picked.emit(0, 1)  # upper triangle: the column
    qapp.processEvents()
    assert win4d.session.state.crosshair == rois.find(matrix._matrix.indices[1]).center_ijk


def test_a_voxel_bin_cell_moves_the_crosshair_too(win4d, qapp):
    """Bins have no name, but they are still somewhere; a click that did
    nothing without an atlas was a dead end."""
    matrix = _matrix(win4d, qapp)
    assert not matrix._matrix.from_rois
    matrix.view.picked.emit(3, 1)
    qapp.processEvents()
    assert win4d.session.state.crosshair == matrix._matrix.location_of(3)


def test_changing_the_node_order_rebuilds_it(win4d, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import SetMatrixOrder

    _add_atlas(win4d, qapp, tmp_path)
    matrix = _matrix(win4d, qapp)
    win4d._dispatch(SetMatrixOrder(matrix.vid, "size"))
    win4d.runner.wait(30_000)
    qapp.processEvents()
    assert "voxel count" in matrix.info.text()


def test_an_unknown_node_order_is_refused(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetMatrixOrder

    matrix = _matrix(win4d, qapp)
    with pytest.raises(KeyError):
        win4d.session.do(SetMatrixOrder(matrix.vid, "spiral"))


def test_the_matrix_is_recorded_and_replayable(win4d, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import SetMatrixOrder

    _add_atlas(win4d, qapp, tmp_path)
    matrix = _matrix(win4d, qapp)
    win4d._dispatch(SetMatrixOrder(matrix.vid, "input"))
    script = win4d.session.to_script()
    assert "OPEN_VIEW" in script and "matrix" in script
    assert "SET_MATRIX_ORDER" in script


# ---------------------------------------------------------------------------
# the cluster window
# ---------------------------------------------------------------------------


def _blobby(win4d, qapp, tmp_path):
    """A stat layer with two blobs of different sizes, selected and thresholded."""
    from fastfuncstuff.viewer.vocab import AddOverlay, SelectLayer, SetThreshold

    v = np.zeros((10, 12, 8), dtype=np.float32)
    v[1:5, 1:5, 1:3] = 4.0
    v[1, 1, 1] = 9.0
    v[8:10, 9:11, 1:2] = -6.0
    path = tmp_path / "blobs.nii.gz"
    nib.save(nib.Nifti1Image(v, np.diag([3.0, 3.0, 3.0, 1.0])), str(path))
    win4d.refresh(win4d.session.do(AddOverlay(str(path))))
    key = win4d.session.state.layers.layers[-1].key
    win4d.refresh(win4d.session.do(SelectLayer(key)))
    win4d.refresh(win4d.session.do(SetThreshold(key, 2.0)))
    qapp.processEvents()
    return key


def _clusters(win4d, qapp, tmp_path):
    """The blob layer's key and its cluster window. One layer, not two -- a
    second AddOverlay of the same file makes the selected layer a different
    one from the key the test then thresholds."""
    key = _blobby(win4d, qapp, tmp_path)
    win4d._new_clusters()
    qapp.processEvents()
    window = win4d.manager.cluster_windows()[0]
    window.min_spin.setValue(1)  # the blobs are smaller than the default minimum
    qapp.processEvents()
    return key, window


def test_clusters_are_listed_biggest_first(win4d, qapp, tmp_path):
    _key, window = _clusters(win4d, qapp, tmp_path)
    assert window.table.rowCount() == 2
    assert window.table.item(0, 1).text() == "32"
    assert window.table.item(1, 1).text() == "4"


def test_clicking_a_cluster_goes_to_its_peak(win4d, qapp, tmp_path):
    _key, window = _clusters(win4d, qapp, tmp_path)
    window.table.setCurrentCell(0, 0)
    qapp.processEvents()
    assert win4d.session.state.crosshair == (1, 1, 1)


def test_the_table_follows_the_threshold_rather_than_going_stale(win4d, qapp, tmp_path):
    """A cluster table computed at a different cut does not describe the
    picture beside it, and two things on screen disagreeing is the worse bug."""
    from fastfuncstuff.viewer.vocab import SetThreshold

    key, window = _clusters(win4d, qapp, tmp_path)
    assert window.table.rowCount() == 2

    # At 5.0 the +4.0 body is gone but its 9.0 peak and the -6.0 blob remain,
    # which is bi-sided clustering doing what it says.
    win4d._dispatch(SetThreshold(key, 5.0))
    qapp.processEvents()
    assert [window.table.item(r, 1).text() for r in range(window.table.rowCount())] == ["4", "1"]

    win4d._dispatch(SetThreshold(key, 7.0))  # only the 9.0 peak survives
    qapp.processEvents()
    assert window.table.rowCount() == 1
    assert window.table.item(0, 1).text() == "1"


def test_an_unthresholded_layer_says_so_instead_of_listing_everything(win4d, qapp, tmp_path):
    from fastfuncstuff.viewer.vocab import SetThreshold

    key, window = _clusters(win4d, qapp, tmp_path)
    win4d._dispatch(SetThreshold(key, 0.0))
    qapp.processEvents()
    assert window.table.rowCount() == 0
    assert "no threshold" in window.info.text()


def test_without_a_clustsim_table_the_alpha_column_is_empty_and_the_note_says_why(
    win4d, qapp, tmp_path
):
    _key, window = _clusters(win4d, qapp, tmp_path)
    assert window.table.item(0, 7).text() == "--"
    assert "no ClustSim table" in window.info.text()


def test_clusters_become_an_roi_layer_that_everything_else_can_use(win4d, qapp, tmp_path):
    """The loop the whole design is for: a threshold produces clusters, the
    clusters become an atlas, the atlas becomes a correlation matrix."""
    _key, window = _clusters(win4d, qapp, tmp_path)
    before = len(win4d.session.state.layers)
    window.rois_button.click()
    qapp.processEvents()

    assert len(win4d.session.state.layers) == before + 1
    added = win4d.session.state.layers.layers[-1]
    assert added.roi
    rois = win4d.session.roi_set(added.key)
    assert [r.name for r in rois] == ["C1", "C2"]

    matrix = _matrix(win4d, qapp)
    assert matrix._matrix.from_rois
    assert matrix._matrix.names == ("C1", "C2") or set(matrix._matrix.names) == {"C1", "C2"}


def test_a_cluster_plots_its_mean_and_not_its_peak_voxel(win4d, qapp, tmp_path):
    """The peak is by definition the most extreme voxel, so its time course is
    the one most selected for -- plotting it flatters the effect."""
    _key, window = _clusters(win4d, qapp, tmp_path)
    window.table.setCurrentCell(0, 0)
    qapp.processEvents()

    series = window.trace._series
    assert series is not None
    run = win4d.session.state.layers.overlay
    picked = window._table.labels == 1
    expected = win4d.session.store.get(run.key).array[picked].mean(0)
    assert np.allclose(series, expected, atol=1e-4)


def test_the_cluster_window_starts_with_a_minimum_that_keeps_out_speckle(win4d, qapp, tmp_path):
    """Opening it on a loose threshold used to list every one-voxel cluster,
    which on a whole brain is tens of thousands of rows and a frozen window."""
    from fastfuncstuff.viewer.ui.clusterwindow import DEFAULT_MIN_VOXELS

    _blobby(win4d, qapp, tmp_path)
    win4d._new_clusters()
    qapp.processEvents()
    window = win4d.manager.cluster_windows()[0]
    assert window.min_voxels == DEFAULT_MIN_VOXELS >= 50
    assert window.table.rowCount() == 0
    assert not window.min_spin.keyboardTracking()


def test_changing_the_cluster_minimum_recomputes_the_table(win4d, qapp, tmp_path):
    _key, window = _clusters(win4d, qapp, tmp_path)
    assert window.table.rowCount() == 2
    window.min_spin.setValue(10)
    qapp.processEvents()
    assert window.table.rowCount() == 1


# ---------------------------------------------------------------------------
# controllers: A, B, ... as tabs
# ---------------------------------------------------------------------------


def _second_controller(win, qapp, datadir, *, underlay="stats.nii.gz"):
    """Open controller B and give it an underlay of its own."""
    ctl = win.new_controller()
    win.refresh(win.session.do(SetUnderlay(str(datadir / underlay))))
    qapp.processEvents()
    return ctl


def test_a_second_controller_is_a_tab_with_its_own_stack(win, qapp, datadir):
    a = win.active
    b = _second_controller(win, qapp, datadir)
    assert [c.letter for c in win.controllers] == ["A", "B"]
    assert win.tabs.count() == 2 and win.active is b
    assert win.session is b.session and win.session is not a.session
    assert len(b.session.state.layers) == 1 and len(a.session.state.layers) == 2
    # The panel is a view of the active tab, so it lists B's stack.
    assert win.layer_list.count() == 1

    win.activate(a)
    qapp.processEvents()
    assert win.layer_list.count() == 2
    assert win.tabs.currentIndex() == 0


def test_every_window_title_says_which_controller_it_belongs_to(win, qapp, datadir):
    _second_controller(win, qapp, datadir)
    for ctl in win.controllers:
        titles = [w.windowTitle() for w in ctl.manager.windows.values()]
        assert titles and all(t.startswith(f"[{ctl.letter}]") for t in titles)


def test_controllers_share_the_crosshair_in_millimetres(win, qapp, datadir):
    from fastfuncstuff.viewer.vocab import SetIJK

    a = win.active
    b = _second_controller(win, qapp, datadir)
    win.activate(a)
    win._dispatch(SetIJK(2, 3, 4))
    qapp.processEvents()
    assert b.session.state.crosshair_mm == pytest.approx(a.session.state.crosshair_mm)
    assert b.session.state.crosshair == (2, 3, 4)


def test_a_click_in_b_s_window_moves_b_and_makes_b_active(win, qapp, datadir):
    """Each controller's windows dispatch into their own session. Routing
    through 'whichever tab is showing' would move A when B's image was clicked."""
    from fastfuncstuff.viewer.vocab import SetIJK

    a = win.active
    b = _second_controller(win, qapp, datadir)
    win.activate(a)
    b_image = next(iter(b.manager.windows.values()))
    b_image._dispatch(SetIJK(5, 6, 1))
    qapp.processEvents()
    assert win.active is b
    assert b.session.state.crosshair == (5, 6, 1)
    assert a.session.state.crosshair == (5, 6, 1)  # followed, being linked


def test_unlinked_controllers_keep_their_own_place(win, qapp, datadir):
    from fastfuncstuff.viewer.vocab import SetIJK

    a = win.active
    b = _second_controller(win, qapp, datadir)
    win.link_check.setChecked(False)
    win.activate(a)
    before = b.session.state.crosshair
    win._dispatch(SetIJK(1, 1, 1))
    qapp.processEvents()
    assert b.session.state.crosshair == before


def test_closing_a_controller_closes_its_windows_and_the_last_one_stays(win, qapp, datadir):
    a = win.active
    b = _second_controller(win, qapp, datadir)
    b_windows = list(b.manager.windows.values())
    assert win.close_controller(b)
    qapp.processEvents()
    assert win.controllers == [a] and win.active is a
    assert all(not w.isVisible() for w in b_windows)
    assert not win.close_controller(a)


def test_tiling_places_every_controller_s_windows(win, qapp, datadir):
    b = _second_controller(win, qapp, datadir)
    win._tile()
    qapp.processEvents()
    for ctl in win.controllers:
        assert all(v.geometry is not None for v in ctl.session.state.viewports)
    assert win.active is b  # arranging windows is not choosing a controller


# ---------------------------------------------------------------------------
# instacorr from a click
# ---------------------------------------------------------------------------


def test_a_mac_ctrl_click_seeds_instacorr_and_draws_a_map(win4d, qapp, monkeypatch):
    """macOS delivers ctrl+click as a right-button press with Meta held, which
    the pane took for the start of a pan -- so the documented gesture never
    produced a map."""
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    from fastfuncstuff.viewer.ui import panes

    monkeypatch.setattr(panes.sys, "platform", "darwin")
    win4d._switch_mode("instacorr")
    assert win4d.runner.wait(20_000)
    qapp.processEvents()

    image = image_of(win4d, Plane.AXIAL)
    pane = image.pane
    image.resize(400, 400)
    qapp.processEvents()
    QTest.mouseClick(
        pane,
        QtCore.Qt.MouseButton.RightButton,
        QtCore.Qt.KeyboardModifier.MetaModifier,
        pane.rect().center(),
    )
    for _ in range(3):
        win4d.runner.wait(20_000)
        qapp.processEvents()

    assert win4d.session.state.seed is not None
    layer = win4d.session.state.layers.find_by_source("mode:instacorr")
    assert layer is not None and layer.name == "A_ICORR"
    assert win4d.session.state.selected == layer.key


def test_keep_from_the_panel_numbers_the_copies(win4d, qapp):
    from fastfuncstuff.viewer.vocab import SetSeed

    win4d._switch_mode("instacorr")
    win4d.runner.wait(20_000)
    qapp.processEvents()
    win4d._dispatch(SetSeed(3, 4, 2))
    win4d.runner.wait(20_000)
    qapp.processEvents()
    for _ in range(2):
        win4d.mode_panel._widgets["action:keep"].click()
        qapp.processEvents()
    names = [ly.name for ly in win4d.session.state.layers if ly.source.startswith("kept:")]
    assert names == ["A_ICORR_1", "A_ICORR_2"]
    assert win4d.layer_list.count() == len(win4d.session.state.layers)


# ---------------------------------------------------------------------------
# ica review
# ---------------------------------------------------------------------------


@pytest.fixture
def ica_folder(tmp_path):
    rng = np.random.default_rng(21)
    out = tmp_path / "ica_out" / "melodic_compat"
    out.mkdir(parents=True)
    maps = rng.normal(size=(10, 12, 8, 4)).astype(np.float32)
    nib.save(nib.Nifti1Image(maps, np.diag([3.0, 3.0, 3.0, 1.0])), str(out / "melodic_IC.nii.gz"))
    t = np.arange(40)
    np.savetxt(out / "melodic_mix", np.column_stack([np.sin(0.1 * f * t) for f in (1, 2, 3, 4)]))
    return tmp_path / "ica_out"


def _ica(win, qapp, folder):
    win._switch_mode("ica")
    win.runner.wait(20_000)
    qapp.processEvents()
    win._mode_param_changed("folder", str(folder))
    win.runner.wait(20_000)
    qapp.processEvents()
    qapp.processEvents()


def test_ica_loads_in_the_window_where_preparation_is_deferred(win, qapp, ica_folder):
    """Its prepare() never cleared the dirty flag, so under the window's
    deferred preparation the decomposition never loaded at all."""
    _ica(win, qapp, ica_folder)
    layer = win.session.state.layers.find_by_source("mode:ica")
    assert layer is not None and layer.name == "A_ICA IC 0"


def test_entering_ica_opens_a_timecourse_and_a_spectrum_window_once(win, qapp, ica_folder):
    from fastfuncstuff.viewer.ui.tracewindow import TraceWindow

    _ica(win, qapp, ica_folder)
    win._switch_mode("plain")
    win._switch_mode("ica")
    qapp.processEvents()
    traces = [w for w in win.manager.windows.values() if isinstance(w, TraceWindow)]
    panels = sorted(win.session.state.viewports.get(w.vid).panel for w in traces)
    assert panels == ["spectrum", "timecourse"]
    timecourse = next(
        w for w in traces if win.session.state.viewports.get(w.vid).panel == "timecourse"
    )
    drawn = timecourse.view._traces
    assert len(drawn) == 1 and drawn[0].values.size == 40
    assert timecourse.windowTitle().startswith("[A] timecourse")


def test_review_keys_in_a_trace_window_label_and_step(win, qapp, ica_folder):
    from fastfuncstuff.viewer.ui.tracewindow import TraceWindow

    _ica(win, qapp, ica_folder)
    window = next(w for w in win.manager.windows.values() if isinstance(w, TraceWindow))
    window.action_requested.emit("noise")
    qapp.processEvents()
    window.action_requested.emit("next")
    qapp.processEvents()
    mode = win.session.mode
    assert mode.labels == {0: "noise"}
    assert mode.params["component"] == 2
    # the panel follows the mode, so the spin box shows where the review is
    spin = win.mode_panel._widgets["component"]
    assert spin.value() == 2
    assert "IC 2" in window.view._traces[0].label


# ---------------------------------------------------------------------------
# the graph legend: tick lines off, colours stay put
# ---------------------------------------------------------------------------


def _ica_graph(win4d, qapp, ica_folder):
    _ica(win4d, qapp, ica_folder)
    graph = open_graph(win4d, qapp)
    win4d.refresh(Aspect.GRAPH | Aspect.LAYERS)
    qapp.processEvents()
    return graph


def test_the_graph_legend_offers_every_layer_and_every_mode_line(win4d, qapp, ica_folder):
    graph = _ica_graph(win4d, qapp, ica_folder)
    run = win4d.session.graph_layers()[0].key
    assert set(graph._trace_checks) == {run, "mode:timecourse", "mode:spectrum"}
    assert all(box.isChecked() for box in graph._trace_checks.values())
    assert graph._trace_checks["mode:spectrum"].text() == "IC spectrum"
    assert graph._trace_checks[run].text() == "bold"


def test_unticking_the_spectrum_takes_it_off_the_graph_and_keeps_colours(win4d, qapp, ica_folder):
    """Colours come from each line's place in the whole list, so hiding one
    must not hand its colour to the next line along."""
    graph = _ica_graph(win4d, qapp, ica_folder)
    before = dict(graph.graph._colors)
    graph._trace_checks["mode:spectrum"].setChecked(False)
    qapp.processEvents()

    idents = {t[0] for cell in graph.graph._cells for t in cell.traces}
    assert "mode:spectrum" not in idents and "mode:timecourse" in idents
    assert graph.graph._colors["mode:timecourse"] == before["mode:timecourse"]
    assert not graph._trace_checks["mode:spectrum"].isChecked()
    assert "SET_VIEW_HIDDEN" in win4d.session.to_script()

    # stepping to another component is a new line with the same identity
    win4d._mode_action("next")
    qapp.processEvents()
    idents = {t[0] for cell in graph.graph._cells for t in cell.traces}
    assert "mode:spectrum" not in idents


def test_a_layer_can_be_ticked_off_and_back_on(win4d, qapp):
    graph = open_graph(win4d, qapp)
    run = win4d.session.graph_layers()[0].key
    graph._trace_checks[run].setChecked(False)
    qapp.processEvents()
    assert not any(cell.traces for cell in graph.graph._cells)
    graph._trace_checks[run].setChecked(True)
    qapp.processEvents()
    assert all(cell.traces for cell in graph.graph._cells)


def _press(window, qapp, key):
    from PySide6 import QtGui

    action = next(a for a in window.actions() if a.shortcut() == QtGui.QKeySequence(key))
    action.trigger()
    qapp.processEvents()


def test_left_right_and_comma_period_step_time_in_a_graph(win4d, qapp):
    graph = open_graph(win4d, qapp)
    st = win4d.session.state
    for key, expected in (("Right", 1), (".", 2), ("Left", 1), (",", 0), (",", 24)):
        _press(graph, qapp, key)
        assert st.time_index == expected, key


def test_comma_and_period_step_time_in_an_image_window(win4d, qapp):
    image = image_of(win4d, Plane.AXIAL)
    _press(image, qapp, ".")
    _press(image, qapp, ".")
    assert win4d.session.state.time_index == 2
    _press(image, qapp, ",")
    assert win4d.session.state.time_index == 1


# ---------------------------------------------------------------------------
# arrow keys in an image window
#
# The controller's arrows move along volume axes, because a controller has no
# picture and "+y" is the only meaning available. An image window does have a
# picture, so its arrows have to mean up/down/left/right *in that picture* --
# a different thing on every plane, and on a flipped axis the opposite thing.
# ---------------------------------------------------------------------------


def _image_window(win):
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow

    return next(w for w in win.manager.windows.values() if isinstance(w, ImageWindow))


@pytest.fixture
def flipped(qapp, tmp_path):
    """A window on a dataset whose x axis runs the other way (LAS-ish)."""
    from fastfuncstuff.viewer.vocab import SetUnderlay

    rng = np.random.default_rng(5)
    aff = np.diag([-3.0, 3.0, 4.0, 1.0])
    aff[:3, 3] = (36.0, -36.0, -20.0)
    nib.save(
        nib.Nifti1Image(rng.random((14, 16, 12)).astype(np.float32) * 100, aff),
        str(tmp_path / "anat.nii.gz"),
    )
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.read_directory(tmp_path)
    w.refresh(session.do(SetUnderlay(str(tmp_path / "anat.nii.gz"))))
    w.show()
    qapp.processEvents()
    yield w
    w.close()


@pytest.mark.parametrize("plane", list(Plane))
@pytest.mark.parametrize(
    ("drow", "dcol"), [(0, -1), (0, 1), (-1, 0), (1, 0)], ids=["left", "right", "up", "down"]
)
def test_arrows_move_the_crosshair_the_way_the_picture_reads(flipped, qapp, plane, drow, dcol):
    from fastfuncstuff.viewer.compose import plane_view
    from fastfuncstuff.viewer.vocab import SetIJK, SetViewPlane

    win = flipped
    image = _image_window(win)
    win.refresh(win.session.do(SetViewPlane(image.vid, str(plane))))
    qapp.processEvents()

    win.session.do(SetIJK(7, 8, 6))  # middle, clear of every edge
    view = plane_view(win.session.state, win.session.state.viewports.find(image.vid))
    before = view.to_image(win.session.state.crosshair)

    image._nudge_in_plane(drow, dcol)

    after = view.to_image(win.session.state.crosshair)
    assert (after[0] - before[0], after[1] - before[1]) == (drow, dcol)


def test_the_edge_stops_rather_than_wrapping_to_the_far_side(flipped, qapp):
    """Converting an out-of-range row through a flipped axis lands at the far
    edge, which would make a held arrow key teleport the crosshair."""
    from fastfuncstuff.viewer.compose import plane_view
    from fastfuncstuff.viewer.vocab import SetIJK, SetViewPlane

    win = flipped
    image = _image_window(win)
    win.refresh(win.session.do(SetViewPlane(image.vid, str(Plane.AXIAL))))
    qapp.processEvents()
    win.session.do(SetIJK(7, 8, 6))

    view = plane_view(win.session.state, win.session.state.viewports.find(image.vid))
    for _ in range(100):
        image._nudge_in_plane(-1, 0)
    assert view.to_image(win.session.state.crosshair)[0] == 0


def test_page_keys_step_the_slice(flipped, qapp):
    from fastfuncstuff.viewer.slicing import plane_layout
    from fastfuncstuff.viewer.vocab import SetIJK, SetViewPlane

    win = flipped
    image = _image_window(win)
    win.refresh(win.session.do(SetViewPlane(image.vid, str(Plane.AXIAL))))
    qapp.processEvents()
    win.session.do(SetIJK(7, 8, 6))

    state = win.session.state
    axis = plane_layout(state.grid.affine, Plane.AXIAL).fixed
    before = state.crosshair[axis]
    image._step(1)
    assert state.crosshair[axis] == before + 1


def test_an_image_window_declares_its_arrows_so_h_can_list_them(flipped):
    """The table is both what is installed and what the help shows."""
    image = _image_window(flipped)
    keys = {b.keys for b in image.help._bindings}
    assert {"Left", "Right", "Up", "Down", "PgUp", "PgDn"} <= keys


# ---------------------------------------------------------------------------
# the graph window's detrend menu
#
# One setting for the window, applied to every line it draws. Per-line would be
# the one thing worse than none at all: the purpose is to make lines
# comparable, and detrending some of them destroys exactly that.
# ---------------------------------------------------------------------------


@pytest.fixture
def graphed(qapp, tmp_path):
    """A window with a graph open on a run whose voxels differ in baseline."""
    from fastfuncstuff.viewer.ui.gridgraph import GraphWindow
    from fastfuncstuff.viewer.vocab import SetIJK, SetUnderlay

    nt = 60
    t = np.arange(nt, dtype=np.float32)
    shared = 4.0 * np.sin(2 * np.pi * t / 15.0)
    data = np.zeros((6, 6, 4, nt), np.float32)
    for i in range(6):
        for j in range(6):
            data[i, j, :] = 300.0 + 50.0 * i + (0.4 + 0.2 * j) * t + shared
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    img = nib.Nifti1Image(data, aff)
    img.header["pixdim"][4] = 2.0
    nib.save(img, str(tmp_path / "run.nii.gz"))

    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.read_directory(tmp_path)
    w.refresh(session.do(SetUnderlay(str(tmp_path / "run.nii.gz"))))
    session.store.ensure_ram(next(iter(session.state.layers)).key)
    w.show()
    qapp.processEvents()
    w._new_graph()
    qapp.processEvents()
    w.refresh(session.do(SetIJK(3, 3, 2)))
    qapp.processEvents()
    graph = next(x for x in w.manager.windows.values() if isinstance(x, GraphWindow))
    yield w, graph, data
    w.close()


def _curves(graph):
    return [np.asarray(v) for cell in graph.graph._cells for _, v in cell.traces]


def test_a_graph_starts_undetrended(graphed):
    """A time course in its own units at its own level is what a graph is for."""
    win, graph, _ = graphed
    assert win.session.state.viewports.find(graph.vid).detrend == -1
    assert graph.detrend_box.currentText() == "none"


def test_a_carpet_still_starts_detrended(qapp, win):
    """The default is per kind: a carpet is unreadable through a ramp."""
    from fastfuncstuff.viewer.viewports import ViewKind

    vid = win.session.open_view(ViewKind.CARPET, Plane.AXIAL)
    assert win.session.state.viewports.find(vid).detrend == 1


def test_the_menu_offers_every_order_up_to_nine(graphed):
    win, graph, _ = graphed
    orders = [graph.detrend_box.itemData(i) for i in range(graph.detrend_box.count())]
    assert orders == list(range(-1, 10))
    assert graph.detrend_box.itemText(0) == "none"
    assert graph.detrend_box.itemText(1) == "mean"
    assert graph.detrend_box.itemText(2) == "linear"


def test_removing_the_mean_stacks_every_curve_at_zero(graphed, qapp):
    from fastfuncstuff.viewer.vocab import SetViewDetrend

    win, graph, _ = graphed
    win.refresh(win.session.do(SetViewDetrend(graph.vid, 0)))
    qapp.processEvents()
    curves = _curves(graph)
    assert len(curves) > 1
    assert max(abs(c.mean()) for c in curves) < 1e-3


def test_removing_the_trend_lays_the_shared_shape_on_itself(graphed, qapp):
    """Baselines and drifts differ per voxel; the response does not."""
    from fastfuncstuff.viewer.vocab import SetViewDetrend

    win, graph, _ = graphed

    win.refresh(win.session.do(SetViewDetrend(graph.vid, -1)))
    qapp.processEvents()
    raw = np.stack(_curves(graph))
    apart = float(np.mean(np.std(raw - raw.mean(axis=1, keepdims=True), axis=0)))

    win.refresh(win.session.do(SetViewDetrend(graph.vid, 1)))
    qapp.processEvents()
    flat = np.stack(_curves(graph))
    together = float(np.mean(np.std(flat - flat.mean(axis=1, keepdims=True), axis=0)))

    assert together < apart / 4, "the curves must converge on one shape"


def test_detrending_changes_the_picture_and_not_the_data(graphed, qapp):
    from fastfuncstuff.viewer.vocab import SetViewDetrend

    win, graph, original = graphed
    key = next(iter(win.session.state.layers)).key
    win.refresh(win.session.do(SetViewDetrend(graph.vid, 3)))
    qapp.processEvents()
    assert np.allclose(win.session.store.ensure_ram(key), original, atol=1e-4)


def test_the_menu_follows_the_viewport_without_dispatching(graphed, qapp):
    """A widget that writes back on every repaint recurses into the recording."""
    from fastfuncstuff.viewer.vocab import SetViewDetrend

    win, graph, _ = graphed
    win.refresh(win.session.do(SetViewDetrend(graph.vid, 2)))
    qapp.processEvents()
    assert graph.detrend_box.currentData() == 2
    assert graph.detrend_box.currentText() == "quadratic"

    before = len(win.session.to_script().splitlines())
    for _ in range(5):
        graph.apply(win.session.state.viewports.find(graph.vid))
    assert len(win.session.to_script().splitlines()) == before


def test_ctrl_c_in_the_terminal_closes_the_viewer(qapp, datadir):
    """Qt's C++ loop starved Python's SIGINT handler, so Ctrl+C did nothing."""
    import signal

    from fastfuncstuff.viewer.ui.window import quit_on_interrupt

    previous = signal.getsignal(signal.SIGINT)
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.show()
    qapp.processEvents()
    try:
        timer = quit_on_interrupt(qapp, w)
        assert timer.isActive(), "without a wake-up the handler never runs"
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        qapp.processEvents()
        assert not w.isVisible()
    finally:
        signal.signal(signal.SIGINT, previous)
        w.close()


# ---------------------------------------------------------------------------
# InstaGLM, as a mode
#
# The mode is one file and declares everything it needs, so what is worth
# testing through the real window is precisely the part that is NOT declared
# once: the column picker's choices come from the model that was just fitted,
# which means they have to survive the panel being rebuilt under them.
# ---------------------------------------------------------------------------


@pytest.fixture
def win_glm(win, qapp, tmp_path):
    """A run with a bright box, so the automask finds a brain to fit."""
    rng = np.random.default_rng(17)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    data = rng.normal(5.0, 0.5, (10, 12, 8, 40)).astype(np.float32)
    data[2:8, 3:9, 2:6, :] += 1000.0 + np.linspace(0, 30, 40, dtype=np.float32)
    img = nib.Nifti1Image(data, aff)
    img.header["pixdim"][4] = 2.0
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "glmbold.nii.gz"))

    events = tmp_path / "sub-01_events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\n"
        + "".join(f"{o}\t2\tfaces\n" for o in (4, 24, 44))
        + "".join(f"{o}\t2\thouses\n" for o in (14, 34, 54))
    )

    from fastfuncstuff.viewer.vocab import SetOverlay

    win.refresh(win.session.do(SetOverlay(str(tmp_path / "glmbold.nii.gz"))))
    win.session.store.ensure_ram(win.session.state.layers.overlay.key)
    qapp.processEvents()
    win.events_file = str(events)
    return win


def _instaglm(win_glm, qapp, **params):
    win_glm._switch_mode("instaglm")
    assert win_glm.runner.wait(20_000)
    qapp.processEvents()
    for name, value in params.items():
        win_glm._mode_param_changed(name, str(value))
        assert win_glm.runner.wait(20_000)
        qapp.processEvents()
    return win_glm.session.mode


def test_instaglm_declares_its_whole_panel(win_glm, qapp):
    _instaglm(win_glm, qapp)
    assert {
        "events",
        "basis",
        "hrf_index",
        "peak",
        "width",
        "undershoot",
        "polort",
        "ortvec",
        "ort_deriv",
        "pcs",
        "show",
        "column",
        "psc",
        "action:fit",
        "action:keep",
    } <= set(win_glm.mode_panel._widgets)


def test_the_fit_runs_off_the_gui_thread(win_glm, qapp):
    """A refit is tens of milliseconds but the gather is seconds, and a gather
    on the paint thread is the freeze the prepare/compute split exists to
    prevent."""
    win_glm._switch_mode("instaglm")
    assert win_glm.session.mode.defer_preparation, "the window must own preparation"
    assert win_glm.runner.wait(20_000)
    qapp.processEvents()
    assert win_glm.session.mode._fit is not None


def test_the_column_picker_offers_the_model_that_was_fitted(win_glm, qapp):
    """A dynamic ChoiceControl: the choices are discovered by the fit, and the
    panel is rebuilt afterwards, so the combo has to come back with them."""
    _instaglm(win_glm, qapp)
    combo = win_glm.mode_panel._widgets["column"]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Pol#0", "Pol#1", "Pol#2"]

    _instaglm(win_glm, qapp, events=win_glm.events_file)
    combo = win_glm.mode_panel._widgets["column"]
    offered = [combo.itemText(i) for i in range(combo.count())]
    assert offered[:2] == ["faces", "houses"]
    assert "Pol#2" in offered


def test_entering_instaglm_opens_an_hrf_window(win_glm, qapp):
    from fastfuncstuff.viewer.ui.tracewindow import TraceWindow

    _instaglm(win_glm, qapp, events=win_glm.events_file)
    traces = [w for w in win_glm.manager.windows.values() if isinstance(w, TraceWindow)]
    panels = [win_glm.session.state.viewports.get(w.vid).panel for w in traces]
    assert panels == ["hrf"]
    assert traces[0].view._traces[0].label == "HRF"


def test_the_graph_legend_gains_the_models_lines(win_glm, qapp):
    """The five lines the mode contributes have to reach a graph window as tick
    boxes, or the whole "watch the fit improve" half of the mode is invisible."""
    _instaglm(win_glm, qapp, events=win_glm.events_file)
    graph = open_graph(win_glm, qapp)
    idents = {e.ident for e in graph.entries(win_glm.session.state.viewports.get(graph.vid))}
    assert {"mode:data", "mode:signal", "mode:fit", "mode:resid", "mode:column"} <= idents


def test_switching_the_shown_map_redraws_without_a_refit(win_glm, qapp):
    _instaglm(win_glm, qapp, events=win_glm.events_file)
    before = win_glm.session.mode._fit
    win_glm._mode_param_changed("show", "t")
    qapp.processEvents()
    assert win_glm.session.mode._fit is before
    layer = win_glm.session.state.layers.find_by_source("mode:instaglm")
    assert layer is not None and layer.name.endswith(" t")


# ---------------------------------------------------------------------------
# drawing mode and the debug report
# ---------------------------------------------------------------------------


def test_the_draw_box_names_which_way_auto_went(win, qapp):
    """ "auto" alone is a question; the useful thing is knowing the answer."""
    win.layer_list.setCurrentRow(0)
    qapp.processEvents()
    assert win.resample_box.itemText(0).startswith("auto (")
    assert win.resample_box.currentText().startswith("auto")


def test_e_cycles_how_the_layer_is_drawn(win, qapp):
    from fastfuncstuff.viewer.ui.shortcuts import Binding  # noqa: F401

    key = win.current_key()
    assert win.session.state.layers.get(key).resample == "auto"
    for expected in ("nearest", "linear", "auto"):
        win._cycle_resample()
        qapp.processEvents()
        assert win.session.state.layers.get(key).resample == expected


def test_the_debug_button_writes_a_report_and_says_where(win, qapp, tmp_path):
    win._write_debug_report()
    qapp.processEvents()
    message = win.statusBar().currentMessage()
    assert message.startswith("wrote ") and "path copied" in message
    written = Path(message[len("wrote ") :].split(" (")[0])
    assert written.exists()
    text = written.read_text()
    assert "# ffs viewer session report" in text
    assert "DEFINES DISPLAY GRID" in text
    assert "## script" in text


def test_the_debug_report_covers_every_controller(win, qapp):
    win.new_controller()
    qapp.processEvents()
    text = win.debug_report()
    assert text.count("# ffs viewer session report") == len(win.controllers)
