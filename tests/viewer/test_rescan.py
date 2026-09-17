"""Files written while the viewer is open: listed, marked, and reloaded when overwritten.

The failures worth pinning are the quiet ones. A new file that never shows up
reads as a pipeline that did not run; an overwritten dataset still drawn from
the old voxels reads as a fix that did not work; and a half-written file
reloaded mid-write replaces good data with a truncated read.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from fastfuncstuff.viewer import catalog as cat  # noqa: E402

CPU = torch.device("cpu")
AFF = np.diag([3.0, 3.0, 3.0, 1.0])


def _write(path, data, labels=None):
    from fastfuncstuff.io.afni import save_nifti

    save_nifti(np.asarray(data, np.float32), path, affine=AFF, brick_labels=labels)
    return path


def _bump_mtime(path):
    """Filesystems with coarse mtimes can give a rewrite the same stamp."""
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    _write(tmp_path / "stage01.anat.nii.gz", rng.random((6, 7, 5)) * 100)
    _write(tmp_path / "stage02.run.nii.gz", rng.random((6, 7, 5, 12)) * 100)
    return tmp_path


# ---------------------------------------------------------------------------
# the catalog diff
# ---------------------------------------------------------------------------


def test_an_unchanged_file_is_not_read_again(folder):
    before = cat.scan(folder)
    result = cat.rescan(folder, before)
    assert not result.any
    # The same objects: no header was read to produce them.
    assert all(a is b for a, b in zip(before, result.entries, strict=True))


def test_new_rewritten_and_deleted_files_are_told_apart(folder):
    before = cat.scan(folder)
    _write(folder / "stage03.stats.nii.gz", np.ones((6, 7, 5)))
    _write(folder / "stage01.anat.nii.gz", np.zeros((6, 7, 5)))
    _bump_mtime(folder / "stage01.anat.nii.gz")
    (folder / "stage02.run.nii.gz").unlink()
    result = cat.rescan(folder, before)
    assert result.added == {folder / "stage03.stats.nii.gz"}
    assert result.changed == {folder / "stage01.anat.nii.gz"}
    assert result.removed == {folder / "stage02.run.nii.gz"}
    assert [e.name for e in result.entries] == ["stage01.anat.nii.gz", "stage03.stats.nii.gz"]


def test_a_rewrite_that_changes_the_header_is_read_afresh(folder):
    before = cat.scan(folder)
    _write(folder / "stage02.run.nii.gz", np.zeros((6, 7, 5, 30)))
    _bump_mtime(folder / "stage02.run.nii.gz")
    entry = next(e for e in cat.rescan(folder, before).entries if e.name == "stage02.run.nii.gz")
    assert entry.n_volumes == 30


def test_a_half_written_file_keeps_its_old_entry_and_is_not_reported(folder):
    """Reporting it would reload a truncated dataset over good data."""
    before = cat.scan(folder)
    target = folder / "stage02.run.nii.gz"
    target.write_bytes(b"\x1f\x8b not yet a whole gzip")
    result = cat.rescan(folder, before)
    assert target not in result.changed
    assert any(e.path == target and e.n_volumes == 12 for e in result.entries)


def test_a_half_written_new_file_waits_to_be_listed(folder):
    before = cat.scan(folder)
    (folder / "stage09.partial.nii.gz").write_bytes(b"\x1f\x8b")
    result = cat.rescan(folder, before)
    assert not result.added
    _write(folder / "stage09.partial.nii.gz", np.ones((6, 7, 5)))
    _bump_mtime(folder / "stage09.partial.nii.gz")
    assert cat.rescan(folder, result.entries).added == {folder / "stage09.partial.nii.gz"}


# ---------------------------------------------------------------------------
# reloading a loaded layer
# ---------------------------------------------------------------------------


@pytest.fixture
def session(folder):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay

    s = ViewerSession(device=CPU)
    s.read_directory(folder)
    _write(
        folder / "stage03.stats.nii.gz",
        np.full((6, 7, 5, 2), 2.0),
        labels=["A#0_Coef", "A#0_Tstat"],
    )
    s.do(SetUnderlay(str(folder / "stage01.anat.nii.gz")))
    s.do(SetOverlay(str(folder / "stage03.stats.nii.gz")))
    yield s
    s.close()


def test_reloading_reads_new_voxels_and_keeps_how_the_layer_is_drawn(session, folder):
    from fastfuncstuff.viewer.vocab import ReloadLayer, SetColormap, SetThreshold, SetVolume

    key = session.state.layers.overlay.key
    session.do(SetVolume(key, 1))
    session.do(SetColormap(key, "viridis"))
    session.do(SetThreshold(key, 1.5))
    before = float(session.volume(key, 1)[2, 2, 2])

    _write(
        folder / "stage03.stats.nii.gz",
        np.full((6, 7, 5, 2), 7.0),
        labels=["B#0_Coef", "B#0_Tstat"],
    )
    session.do(ReloadLayer(key))

    layer = session.state.layers.overlay
    assert layer.key == key
    assert (layer.colormap, layer.threshold, layer.volume_index) == ("viridis", 1.5, 1)
    assert layer.labels == ("B#0_Coef", "B#0_Tstat")
    assert before == 2.0 and float(session.volume(key, 1)[2, 2, 2]) == 7.0


def test_a_reload_with_fewer_sub_bricks_keeps_a_valid_index(session, folder):
    from fastfuncstuff.viewer.vocab import ReloadLayer, SetThresholdIndex, SetVolume

    key = session.state.layers.overlay.key
    session.do(SetVolume(key, 1))
    session.do(SetThresholdIndex(key, 1))
    _write(folder / "stage03.stats.nii.gz", np.ones((6, 7, 5)))
    session.do(ReloadLayer(key))
    layer = session.state.layers.overlay
    assert layer.n_volumes == 1 and layer.volume_index == 0 and layer.threshold_brick == 0


def test_applying_a_rescan_marks_files_and_reloads_only_the_loaded_ones(session, folder):
    before = list(session.catalog)
    _write(folder / "stage01.anat.nii.gz", np.full((6, 7, 5), 42.0))
    _bump_mtime(folder / "stage01.anat.nii.gz")
    _write(folder / "stage04.new.nii.gz", np.ones((6, 7, 5)))
    result = cat.rescan(folder, before)
    base = session.state.layers.base.key

    reloaded = session.apply_rescan(result)

    assert reloaded == [base]
    assert float(session.volume(base)[1, 1, 1]) == 42.0
    fresh = session.catalog_fresh
    assert fresh[folder / "stage04.new.nii.gz"] == "new"
    assert fresh[folder / "stage01.anat.nii.gz"] == "updated"


# ---------------------------------------------------------------------------
# the window, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(qapp, folder):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SetUnderlay

    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w._rescan_timer.setInterval(100)
    w.read_directory(folder)
    w.refresh(session.do(SetUnderlay(str(folder / "stage01.anat.nii.gz"))))
    w._sync_layer_list()
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _wait(qapp, condition, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def _rows(box):
    return [box.itemText(i) for i in range(box.count())]


def test_a_file_written_while_open_appears_marked_new(win, qapp, folder):
    _write(folder / "stage05.written_later.nii.gz", np.ones((6, 7, 5)))
    assert _wait(qapp, lambda: any("stage05.written_later" in t for t in _rows(win.overlay_box))), (
        "the directory watcher never produced a rescan"
    )
    row = next(i for i, t in enumerate(_rows(win.overlay_box)) if "stage05" in t)
    assert win.overlay_box.itemText(row).endswith("new")
    assert win.overlay_box.itemData(row, win_background()) is not None


def win_background():
    from fastfuncstuff.viewer.ui.window import BACKGROUND

    return BACKGROUND


def test_opening_a_new_file_clears_its_mark(win, qapp, folder):
    _write(folder / "stage05.written_later.nii.gz", np.ones((6, 7, 5)))
    win._start_rescan(manual=True)
    assert _wait(qapp, lambda: any("stage05" in t for t in _rows(win.overlay_box)))
    row = next(i for i, t in enumerate(_rows(win.overlay_box)) if "stage05" in t)
    win.overlay_box.setCurrentIndex(row)
    win.overlay_box.activated.emit(row)
    qapp.processEvents()
    row = next(i for i, t in enumerate(_rows(win.overlay_box)) if "stage05" in t)
    assert not win.overlay_box.itemText(row).endswith("new")


def test_overwriting_the_loaded_underlay_reloads_it(win, qapp, folder):
    key = win.session.state.layers.base.key
    assert float(win.session.volume(key)[1, 1, 1]) != 42.0
    _write(folder / "stage01.anat.nii.gz", np.full((6, 7, 5), 42.0))
    _bump_mtime(folder / "stage01.anat.nii.gz")
    assert _wait(qapp, lambda: float(win.session.volume(key)[1, 1, 1]) == 42.0), (
        "the overwritten underlay was never reloaded"
    )
    assert "reloaded stage01.anat.nii.gz" in win.statusBar().currentMessage()


def test_the_rescan_button_finds_what_the_watcher_did_not(win, qapp, folder):
    """A write from another machine raises no event; the button is the way in."""
    win._watcher.removePaths(win._watcher.directories() + win._watcher.files())
    _write(folder / "stage06.remote.nii.gz", np.ones((6, 7, 5)))
    win.rescan_button.click()
    assert _wait(qapp, lambda: any("stage06.remote" in t for t in _rows(win.underlay_box)))
