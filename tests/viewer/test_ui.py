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
