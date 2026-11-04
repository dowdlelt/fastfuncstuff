"""Dropping files on a viewer window: they load, and their folders are listed unread."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from fastfuncstuff.viewer import catalog as cat  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.ui.window import ViewerWindow  # noqa: E402

CPU = torch.device("cpu")
Copy = QtCore.Qt.DropAction.CopyAction
Move = QtCore.Qt.DropAction.MoveAction


def _vol(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((6, 6, 4), np.float32)
    data[0, 0, 0] = 0.5  # not a label volume
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def _mesh(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    info = {
        "head": np.array([2, 0, 20], np.int32),
        "valid": "1  # volume info valid",
        "filename": "vol.nii",
        "volume": np.array([64, 64, 64]),
        "voxelsize": np.array([1.0, 1.0, 1.0]),
        "xras": np.array([-1.0, 0.0, 0.0]),
        "yras": np.array([0.0, 0.0, -1.0]),
        "zras": np.array([0.0, 1.0, 0.0]),
        "cras": np.array([0.0, 0.0, 0.0]),
    }
    nib.freesurfer.write_geometry(path, vertices, faces, volume_info=info)
    return path


def _mime(*paths):
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _drop(target, *paths, actions=Copy, mods=QtCore.Qt.KeyboardModifier.NoModifier):
    """Enter then drop, as Qt delivers them; returns (accepted, action it settled on).

    Values, not the event: the event points at ``mime``, which dies with this frame,
    and pytest printing a stale event on a failed assert segfaults in Qt's debug stream.
    """
    app = QtWidgets.QApplication.instance()
    mime = _mime(*paths)
    point = QtCore.QPoint(2, 2)
    buttons = QtCore.Qt.MouseButton.LeftButton
    app.sendEvent(target, QtGui.QDragEnterEvent(point, actions, mime, buttons, mods))
    event = QtGui.QDropEvent(QtCore.QPointF(point), actions, mime, buttons, mods)
    app.sendEvent(target, event)
    answer = event.isAccepted(), event.dropAction()
    app.processEvents()
    return answer


@pytest.fixture
def window():
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    yield win
    win.manager.close_all()
    win.close()
    session.close()


def _dir_rows(win):
    return [win.dir_box.itemData(r) for r in range(win.dir_box.count())]


def test_a_mesh_is_told_by_its_bytes_and_a_brik_stands_for_its_head(tmp_path):
    vol = _vol(tmp_path / "anat.nii.gz")
    named = _mesh(tmp_path / "lh.white")
    unnamed = _mesh(tmp_path / "thing")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    head, brik = tmp_path / "a+orig.HEAD", tmp_path / "a+orig.BRIK.gz"
    datasets, meshes, skipped = cat.sort_dropped(
        [
            vol,
            named,
            unnamed,
            notes,
            tmp_path,
            tmp_path / "x.surf.gii",
            tmp_path / "y.func.gii",
            head,
            brik,
            tmp_path / "gone",
        ]
    )
    assert datasets == [vol, tmp_path / "y.func.gii", head]
    assert meshes == [named, unnamed, tmp_path / "x.surf.gii"]
    assert skipped == [notes, tmp_path, tmp_path / "gone"]


def test_a_dropped_file_loads_and_lists_its_folder_without_reading_it(
    tmp_path, window, monkeypatch
):
    vol = _vol(tmp_path / "one" / "anat.nii.gz")
    _vol(tmp_path / "one" / "sibling.nii.gz")

    def refuse(*args, **kwargs):
        raise AssertionError(f"listed a directory: {args}")

    monkeypatch.setattr(cat, "discover", refuse)
    _drop(window, vol)

    session = window.session
    assert [lay.path for lay in session.state.layers][-1:] == [str(vol)]
    assert session.catalog_dir is None and not session.catalog
    assert str(vol.parent.resolve()) in [r[0] for r in _dir_rows(window) if r]


def test_a_dropped_mesh_loads_and_lists_its_folder(tmp_path, window):
    mesh = _mesh(tmp_path / "surfs" / "lh.white")
    _drop(window, mesh)
    assert [row.path for row in window.session.surfaces.meshes] == [mesh]
    assert str(mesh.parent.resolve()) in [r[0] for r in _dir_rows(window) if r]
    assert window.mesh_window is not None and window.mesh_window.isVisible()


def test_a_file_manager_proposing_a_move_gets_a_copy(tmp_path, window):
    # Shift is how a file manager's drag asks for a move.
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    vol = _vol(tmp_path / "anat.nii.gz")
    assert _drop(window, vol, actions=Copy | Move, mods=shift) == (True, Copy)


def test_a_drag_that_can_only_move_is_refused(tmp_path, window):
    accepted, _ = _drop(window, _vol(tmp_path / "anat.nii.gz"), actions=Move)
    assert not accepted
    assert not len(window.session.state.layers)


def test_a_field_does_not_swallow_the_drop_as_text(tmp_path, window):
    vol = _vol(tmp_path / "anat.nii.gz")
    field = QtWidgets.QLineEdit(window)
    _drop(field, vol)
    assert field.text() == ""
    assert [lay.path for lay in window.session.state.layers] == [str(vol)]


def test_an_unclaimed_window_keeps_its_own_drags(tmp_path, window):
    # A file dialog's sidebar is one: it takes URL drags for itself.
    _drop(QtWidgets.QLineEdit(), _vol(tmp_path / "anat.nii.gz"))
    assert not len(window.session.state.layers)


def test_a_bad_file_is_reported_and_the_rest_still_load(tmp_path, window):
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    good = _vol(tmp_path / "good.nii.gz")
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    _drop(window, bad, good, notes)
    assert [lay.path for lay in window.session.state.layers] == [str(good)]
    message = window.statusBar().currentMessage()
    assert "loaded 1 of 2" in message and "bad.nii.gz" in message and "notes.txt" in message


def test_a_drop_on_a_companion_window_goes_to_its_controller(tmp_path, window):
    first = window.active
    assert first.manager.windows  # the default layout opened some
    second = window.new_controller()
    window.activate(second)
    vol = _vol(tmp_path / "anat.nii.gz")
    _drop(next(iter(first.manager.windows.values())), vol)
    assert window.active is first
    assert [lay.path for lay in first.session.state.layers] == [str(vol)]
    assert not len(second.session.state.layers)
    window.close_controller(second)
