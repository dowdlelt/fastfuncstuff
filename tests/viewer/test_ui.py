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

from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.state import Plane  # noqa: E402
from fastfuncstuff.viewer.ui.window import ViewerWindow  # noqa: E402
from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay  # noqa: E402

CPU = torch.device("cpu")


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
# the panel must always be recoverable
# ---------------------------------------------------------------------------


def test_the_panel_starts_visible(win):
    assert win.dock.isVisible()
    assert win.panel_button.isChecked()


def test_closing_the_panel_by_its_x_unchecks_the_button(win, qapp):
    """Otherwise the button lies about the state and cannot bring it back."""
    win.dock.close()
    qapp.processEvents()
    assert not win.dock.isVisible()
    assert not win.panel_button.isChecked()


def test_the_button_reopens_a_panel_closed_by_its_x(win, qapp):
    """The bug this file exists for: a panel with no way back."""
    win.dock.close()
    qapp.processEvents()
    win.panel_button.click()
    qapp.processEvents()
    assert win.dock.isVisible()


def test_the_button_round_trips(win, qapp):
    for _ in range(3):
        win.panel_button.click()
        qapp.processEvents()
        assert not win.dock.isVisible()
        win.panel_button.click()
        qapp.processEvents()
        assert win.dock.isVisible()


# ---------------------------------------------------------------------------
# panes
# ---------------------------------------------------------------------------


def test_panes_start_visible(win):
    assert all(b.isChecked() for b in win._pane_buttons.values())


def test_a_pane_can_be_hidden_and_restored(win, qapp):
    button = win._pane_buttons[Plane.SAGITTAL]
    button.setChecked(False)
    qapp.processEvents()
    assert not win._panes[Plane.SAGITTAL].isVisible()
    button.setChecked(True)
    qapp.processEvents()
    assert win._panes[Plane.SAGITTAL].isVisible()


def test_the_last_pane_cannot_be_closed(win, qapp):
    """An image viewer showing no images has no obvious way out."""
    for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
        win._pane_buttons[plane].setChecked(False)
        qapp.processEvents()
    assert sum(b.isChecked() for b in win._pane_buttons.values()) == 1


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def test_no_graph_window_by_default(win):
    """Goal zero is images; a graph must not take space until asked for."""
    assert win._graphs == {}


def test_a_graph_opens_as_a_floating_window(win, qapp):
    win._graph_buttons[Plane.AXIAL].setChecked(True)
    qapp.processEvents()
    graph = win._graphs["axial"]
    assert graph.isWindow(), "the graph must not be docked into the main window"
    assert graph.isVisible()


def test_closing_a_graph_unchecks_its_button(win, qapp):
    win._graph_buttons[Plane.AXIAL].setChecked(True)
    qapp.processEvents()
    win._graphs["axial"].close()
    qapp.processEvents()
    assert not win._graph_buttons[Plane.AXIAL].isChecked()


def test_graph_cell_count_follows_the_size_choice(win, qapp):
    win._graph_buttons[Plane.AXIAL].setChecked(True)
    qapp.processEvents()
    graph = win._graphs["axial"]
    for index, expected in enumerate((1, 4, 9)):
        graph.size_box.setCurrentIndex(index)
        graph.refresh()
        assert len(graph.graph._cells) == expected


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

    assert win.thr_head.text() == "THRESH"
    win.refresh(win.session.do(SetMode("ica")))
    qapp.processEvents()
    # No decomposition here, so no computed layer -- the label stays generic
    # rather than claiming units it is not showing.
    assert win.thr_head.text() == "THRESH"


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
