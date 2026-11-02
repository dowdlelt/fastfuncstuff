"""Named files: they load, the catalog is the list, and no directory gets opened.

The failure worth pinning is the quiet one. On a synced folder, listing or
header-reading a directory can fetch every file in it, and a file-list session
that scans anyway looks exactly like one that doesn't.
"""

from __future__ import annotations

import os
import shlex

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from fastfuncstuff.cli.viewer import build_parser, main  # noqa: E402
from fastfuncstuff.viewer import catalog as cat  # noqa: E402
from fastfuncstuff.viewer.commands import parse_script  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.vocab import Read, ReadFiles  # noqa: E402

CPU = torch.device("cpu")


def _vol(path, value=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((6, 6, 4), value, np.float32)
    data[0, 0, 0] = 0.5  # not a label volume
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


@pytest.fixture
def no_listing(monkeypatch):
    """Listing any directory is a failure, whichever route gets there."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"listed a directory: {args}")

    monkeypatch.setattr(cat, "discover", refuse)
    monkeypatch.setattr(os, "scandir", refuse)
    monkeypatch.setattr(os, "listdir", refuse)


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


def test_listed_files_are_the_catalog_in_the_order_given(tmp_path, session, no_listing):
    b = _vol(tmp_path / "b.nii.gz")
    a = _vol(tmp_path / "a.nii.gz")
    _vol(tmp_path / "sibling.nii.gz")
    session.do(ReadFiles((str(b), str(a), str(b))))
    assert [e.path for e in session.catalog] == [b, a]
    assert session.catalog_dir is None and session.reading_files


def test_loading_a_listed_file_does_not_list_its_directory(tmp_path, session, no_listing):
    a = _vol(tmp_path / "a.nii.gz")
    session.read_files([a])
    session.load(a)
    assert [ly.name for ly in session.state.layers] == ["a.nii.gz"]
    assert session.reading_files  # the file's directory joined the list, unread
    assert list(session.directories) == [tmp_path.resolve()]


def test_a_listed_file_opens_from_a_directory_that_refuses_listing(tmp_path, session):
    """The same promise with no mocks: execute-only, so a listing would raise."""
    a = _vol(tmp_path / "locked" / "a.nii.gz")
    (tmp_path / "locked").chmod(0o311)
    try:
        session.read_files([a])
        session.load(a)
        assert cat.rescan_files((a,), session.catalog).entries == session.catalog
        assert [e.path for e in session.catalog] == [a]
    finally:
        (tmp_path / "locked").chmod(0o755)


def test_rescan_stays_inside_the_list(tmp_path, no_listing):
    a, b = _vol(tmp_path / "a.nii.gz"), _vol(tmp_path / "b.nii.gz")
    previous = list(cat.describe([a, b]))
    _vol(tmp_path / "new_sibling.nii.gz")
    _vol(a, 2.0)
    os.utime(a, ns=(10**18, 10**18))
    b.unlink()
    result = cat.rescan_files((a, b), previous)
    assert result.changed == {a} and result.removed == {b} and not result.added
    assert [e.path for e in result.entries] == [a]


def test_a_directory_and_the_list_can_be_gone_between(tmp_path, session):
    a = _vol(tmp_path / "x" / "a.nii.gz")
    _vol(tmp_path / "y" / "c.nii.gz")
    session.read_files([a])
    session.do(Read(str(tmp_path / "y")))
    assert not session.reading_files and [e.name for e in session.catalog] == ["c.nii.gz"]
    session.read_files(session.file_list)
    assert session.reading_files and [e.name for e in session.catalog] == ["a.nii.gz"]


def test_read_files_round_trips_through_a_script():
    cmd = ReadFiles(("/data/run 1.nii.gz", "/data/b.nii.gz"))
    assert parse_script(cmd.to_line()) == [cmd]


def test_a_mistyped_path_stops_the_viewer(tmp_path, capsys):
    with pytest.raises(SystemExit, match="nope.nii.gz"):
        main(["-no_window", "-device", "cpu", "-read", str(tmp_path / "nope.nii.gz")])
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    with pytest.raises(SystemExit, match="notes.txt"):
        main(["-no_window", "-device", "cpu", "-read", str(notes)])


def _replay(monkeypatch, argv):
    """Run the headless path, returning the session main() built."""
    built = []
    real_close = ViewerSession.close

    def close(self):
        built.append(
            (self.catalog_dir, self.file_list, list(self.directories), list(self.state.layers))
        )
        real_close(self)

    monkeypatch.setattr(ViewerSession, "close", close)
    assert main(["-no_window", "-device", "cpu", *argv]) == 0
    return built[0]


def test_cli_read_files_load_with_nothing_to_click(tmp_path, monkeypatch, no_listing):
    a, b = _vol(tmp_path / "a.nii.gz"), _vol(tmp_path / "b.nii.gz")
    catalog_dir, file_list, _, layers = _replay(monkeypatch, ["-read", str(a), str(b)])
    assert catalog_dir is None and file_list == (a, b)
    assert [ly.name for ly in layers] == ["a.nii.gz", "b.nii.gz"]


def test_cli_a_file_named_both_ways_loads_once(tmp_path, monkeypatch, no_listing):
    a, b = _vol(tmp_path / "a.nii.gz"), _vol(tmp_path / "b.nii.gz")
    _, file_list, _, layers = _replay(monkeypatch, [str(a), "-read", str(a), str(b)])
    assert file_list == (a, b) and [ly.name for ly in layers] == ["a.nii.gz", "b.nii.gz"]


def test_cli_several_directories_read_only_the_first(tmp_path, monkeypatch):
    one, two = tmp_path / "one", tmp_path / "two"
    _vol(one / "a.nii.gz")
    _vol(two / "b.nii.gz")
    scanned = []
    real = cat.discover
    monkeypatch.setattr(cat, "discover", lambda d, **kw: scanned.append(str(d)) or real(d, **kw))
    catalog_dir, _, directories, _ = _replay(monkeypatch, ["-read", str(one), str(two)])
    assert catalog_dir == one and scanned == [str(one)]
    assert directories == [one.resolve(), two.resolve()]


def test_cli_files_beside_a_directory_still_load(tmp_path, monkeypatch):
    """``-read DIR a b`` is what ctrl+P printed for a long time."""
    a = _vol(tmp_path / "a.nii.gz")
    catalog_dir, file_list, _, layers = _replay(monkeypatch, ["-read", str(tmp_path), str(a)])
    assert catalog_dir == tmp_path and file_list == ()
    assert [ly.name for ly in layers] == ["a.nii.gz"]


def test_cli_bare_files_load_and_list_nothing_else(tmp_path, monkeypatch, no_listing):
    a, b = _vol(tmp_path / "a.nii.gz"), _vol(tmp_path / "b.nii.gz")
    _vol(tmp_path / "sibling.nii.gz")
    catalog_dir, file_list, _, layers = _replay(monkeypatch, [str(a), str(b)])
    assert catalog_dir is None and file_list == (a, b)
    assert [ly.name for ly in layers] == ["a.nii.gz", "b.nii.gz"]


def test_cli_a_bare_directory_is_read(tmp_path, monkeypatch):
    a = _vol(tmp_path / "a.nii.gz")
    catalog_dir, _, _, layers = _replay(monkeypatch, [str(tmp_path)])
    assert catalog_dir == tmp_path and layers == []
    catalog_dir, _, _, layers = _replay(monkeypatch, [str(tmp_path), str(a)])
    assert catalog_dir == tmp_path and [ly.name for ly in layers] == ["a.nii.gz"]


def test_the_window_opens_bare_files_without_reading_their_directory(
    tmp_path, monkeypatch, no_listing
):
    """launch() is where the parent directory used to get read."""
    import signal

    QtWidgets = pytest.importorskip("PySide6.QtWidgets")  # noqa: N806
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    a = _vol(tmp_path / "a.nii.gz")
    _vol(tmp_path / "sibling.nii.gz")
    from fastfuncstuff.viewer.ui import window

    built = []
    real_init = window.ViewerWindow.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    seen = {}

    def run(_self=None):
        w = built[0]
        seen.update(
            catalog=[e.path for e in w.session.catalog],
            layers=[ly.name for ly in w.session.state.layers],
            box=w.dir_box.currentText(),
            line=w.session.launch_command(),
        )
        w.close()
        w.session.close()
        return 0

    monkeypatch.setattr(window.ViewerWindow, "__init__", init)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", run)
    handler = signal.getsignal(signal.SIGINT)
    try:
        assert main(["-device", "cpu", str(a)]) == 0
    finally:
        signal.signal(signal.SIGINT, handler)
    assert seen["catalog"] == [a] and seen["layers"] == ["a.nii.gz"]
    assert seen["box"] == "list  1 file"
    assert "-read" not in seen["line"]  # reopening it needs only the file


def test_work_dir_names_a_listed_files_folder_without_listing_it(tmp_path, session, no_listing):
    assert session.work_dir is None
    session.read_files([_vol(tmp_path / "x" / "a.nii.gz")])
    assert session.work_dir == tmp_path / "x"


def test_launch_command_reopens_what_a_file_list_loaded(tmp_path, session):
    listed = _vol(tmp_path / "listed.nii.gz")
    layer = _vol(tmp_path / "layer.nii.gz")
    session.read_files([listed, layer])
    args = build_parser().parse_args(shlex.split(session.launch_command())[1:])
    assert args.read is None
    assert args.datasets == [str(listed.resolve()), str(layer.resolve())]  # nothing loaded yet
    session.load(layer)
    args = build_parser().parse_args(shlex.split(session.launch_command())[1:])
    assert args.read is None and args.datasets == [str(layer.resolve())]


def test_launch_command_keeps_the_other_directories(tmp_path, session):
    one, two = tmp_path / "one", tmp_path / "two"
    _vol(one / "a.nii.gz")
    two.mkdir()
    session.read_directory(one)
    session.offer_directories([two])
    args = build_parser().parse_args(shlex.split(session.launch_command())[1:])
    assert args.read == [str(one.resolve()), str(two.resolve())]


def test_window_offers_the_list_and_goes_back_to_it(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")  # noqa: N806
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    a = _vol(tmp_path / "x" / "a.nii.gz")
    _vol(tmp_path / "y" / "c.nii.gz")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    try:
        w.read_files([a])
        assert w.dir_box.currentText() == "list  1 file"
        assert not w.fav_button.isEnabled()
        assert w._watcher.directories() == []
        assert w._watcher.files() == [str(a)]
        w.read_directory(tmp_path / "y")
        assert [e.name for e in session.catalog] == ["c.nii.gz"]
        w._pick_directory(0)
        assert [e.name for e in session.catalog] == ["a.nii.gz"]
        assert w.dir_box.currentText() == "list  1 file"
    finally:
        w.close()
        session.close()
        app.processEvents()
