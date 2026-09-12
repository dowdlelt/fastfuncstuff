"""Window behaviour that has no way back if it breaks.

Runs Qt offscreen. These are deliberately not pixel tests -- the compositing is
covered headlessly in test_compose.py. What is tested here is the class of bug
that leaves the user stuck: a panel that cannot be reopened, a pane that cannot
be restored, a control that desynchronises from the state it displays.
"""

from __future__ import annotations

import os

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
    win._tile()
    qapp.processEvents()
    rects = [v.geometry for v in win.session.state.viewports]
    assert all(r is not None for r in rects)
    # Recorded per window: collapsing on command type alone would leave one.
    lines = [ln for ln in win.session.to_script().splitlines() if ln.startswith("SET_VIEW_GEOM")]
    assert len(lines) == len(rects)


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
# the pickers must agree with what is on screen
# ---------------------------------------------------------------------------


def test_a_fresh_read_selects_nothing(qapp, datadir):
    """A picker naming a file while the panes are empty reads as a failed load."""
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    try:
        w.read_directory(datadir)
        qapp.processEvents()
        assert w.underlay_box.currentText() == "(none)"
        assert w.overlay_box.currentText() == "(none)"
        assert len(session.state.layers) == 0
    finally:
        w.close()


def test_the_pickers_follow_what_is_loaded(win):
    assert win.underlay_box.currentText().startswith("anat.nii.gz")
    assert win.overlay_box.currentText().startswith("stats.nii.gz")


def test_picker_rows_are_two_columns(win):
    """Dimensions and volume count must be scannable, not truncated."""
    row = win.underlay_box.itemText(1)
    assert "anat.nii.gz" in row
    assert "10x12x8" in row


def test_pickers_are_wide_enough_for_their_widest_row(win):
    """Qt sizes a combo to its current item, which truncates the rest on macOS."""
    from PySide6 import QtGui

    metrics = QtGui.QFontMetrics(win.underlay_box.font())
    widest = max(
        metrics.horizontalAdvance(win.underlay_box.itemText(i))
        for i in range(win.underlay_box.count())
    )
    assert win.underlay_box.minimumWidth() >= widest


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
    winstats.thrbrick_box.setCurrentIndex(3)  # row 0 is "same as OLAY"
    winstats.thrbrick_box.activated.emit(3)
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
    assert "source" in labels, f"the correlated time course vanished: {labels}"
    assert "prepared" in labels


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

    assert win.session.state.theme == "dark"
    win._toggle_theme()
    qapp.processEvents()
    assert win.session.state.theme == "light"
    assert theme.palette().name == "light"

    # Every window, not just the one the button is on.
    light = theme.LIGHT.bg
    assert light in win.styleSheet()
    assert all(light in w.styleSheet() for w in win.manager.windows.values())

    win._toggle_theme()
    qapp.processEvents()
    assert theme.palette().name == "dark"


def test_the_theme_button_names_where_it_takes_you(win, qapp):
    """A button labelled with the current state reads as a status light."""
    assert win.theme_button.text() == "LIGHT [d]"
    win._toggle_theme()
    qapp.processEvents()
    # The key is bracketed inside the word when the word contains it.
    assert win.theme_button.text() == "[D]ARK"
    win._toggle_theme()
    qapp.processEvents()


def test_a_window_opened_after_the_switch_is_born_light(win, qapp):
    from fastfuncstuff.viewer.ui import theme

    win._toggle_theme()
    qapp.processEvents()
    win._new_graph()
    qapp.processEvents()
    graph = win.manager.windows[win.session.state.viewports.graphs[0].id]
    assert theme.LIGHT.bg in graph.styleSheet()
    win._toggle_theme()
    qapp.processEvents()


def test_the_palette_is_recorded_so_a_replay_looks_the_same(win, qapp):
    win.session.bus.clear_log()
    win._toggle_theme()
    qapp.processEvents()
    assert "SET_THEME light" in win.session.to_script()
    win._toggle_theme()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# deriving a layer
# ---------------------------------------------------------------------------


def test_the_derive_controls_are_off_for_a_three_d_layer(win, qapp):
    """An anatomy has no nuisance to project out of it."""
    win.layer_list.setCurrentRow(1)  # the underlay: 3-D
    qapp.processEvents()
    assert not win.denoise_button.isEnabled()


def test_denoise_runs_off_the_gui_thread_and_installs_on_it(win4d, qapp):
    src = win4d.session.state.layers.overlay.key
    win4d.session.store.ensure_ram(src)
    win4d.layer_list.setCurrentRow(0)
    win4d.polort_spin.setValue(2)
    qapp.processEvents()

    win4d._denoise()
    assert win4d.runner.wait(20_000)
    qapp.processEvents()

    derived = win4d.session.state.layers.find_by_source(f"derived:denoise:{src}")
    assert derived is not None
    assert "DENOISE" in win4d.session.to_script()


def test_a_second_click_re_derives_instead_of_chaining(win4d, qapp):
    """Deriving selects the result, so a naive second click would chain."""
    src = win4d.session.state.layers.overlay.key
    win4d.session.store.ensure_ram(src)
    win4d.layer_list.setCurrentRow(0)
    win4d.polort_spin.setValue(1)
    qapp.processEvents()

    win4d._denoise()
    win4d.runner.wait(20_000)
    qapp.processEvents()
    after_one = len(win4d.session.state.layers)

    win4d.polort_spin.setValue(3)
    win4d._denoise()
    win4d.runner.wait(20_000)
    qapp.processEvents()
    assert len(win4d.session.state.layers) == after_one
    # And the recording names the real source, so a replay does not chain.
    lines = [ln for ln in win4d.session.to_script().splitlines() if ln.startswith("DENOISE")]
    assert all(ln.split()[1] == src for ln in lines)


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
