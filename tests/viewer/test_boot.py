"""The stand-in window: it only earns its place if it appears without torch."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)


def test_boot_window_appears_before_torch_is_imported():
    code = (
        "import sys\n"
        "from fastfuncstuff.viewer.ui.boot import BootWindow, application\n"
        "from fastfuncstuff.cli.viewer import build_parser\n"
        "from fastfuncstuff.viewer.catalog import split_read_targets\n"
        "app = application(); w = BootWindow(); w.show(); app.processEvents()\n"
        "build_parser()\n"
        "print('torch' in sys.modules)\n"
    )
    done = _run(code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_import_runs_off_the_gui_thread_and_returns_the_module(tmp_path, monkeypatch):
    (tmp_path / "boot_probe_ok.py").write_text(
        "import threading\nimported_on = threading.current_thread().name\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    from fastfuncstuff.viewer.ui.boot import import_behind_window

    handler = signal.getsignal(signal.SIGINT)
    try:
        boot, module = import_behind_window("boot_probe_ok")
    finally:
        signal.signal(signal.SIGINT, handler)
    assert module.imported_on == "viewer-import"
    assert boot.isVisible() and not boot.importing
    boot.close()  # past the import, closing is an ordinary close: this test survives it
    assert not boot.isVisible()


def test_an_import_failure_reaches_the_caller_and_closes_the_window(tmp_path, monkeypatch):
    (tmp_path / "boot_probe_bad.py").write_text("raise ImportError('no such extra')\n")
    monkeypatch.syspath_prepend(tmp_path)
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.boot import BootWindow, import_behind_window

    handler = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(ImportError, match="no such extra"):
            import_behind_window("boot_probe_bad")
    finally:
        signal.signal(signal.SIGINT, handler)
    left = [w for w in QtWidgets.QApplication.topLevelWidgets() if isinstance(w, BootWindow)]
    assert not any(w.isVisible() for w in left)


def test_closing_during_the_import_ends_the_process_promptly(tmp_path):
    """Exit is immediate, not after the import: the import here never finishes."""
    (tmp_path / "boot_probe_slow.py").write_text("import time\ntime.sleep(60)\n")
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n"
        "from PySide6 import QtCore, QtWidgets\n"
        "from fastfuncstuff.viewer.ui.boot import application, import_behind_window\n"
        "app = application()\n"
        "QtCore.QTimer.singleShot(100, lambda: QtWidgets.QApplication.topLevelWidgets()[0].close())\n"
        "import_behind_window('boot_probe_slow')\n"
        "sys.exit(7)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr


def test_importing_the_viewer_package_alone_stays_torch_free_but_names_still_resolve():
    done = _run(
        "import sys, fastfuncstuff.viewer as v\n"
        "assert 'torch' not in sys.modules\n"
        "assert callable(v.parse_script) and v.ViewerSession is not None\n"
        "assert 'torch' in sys.modules\n"
    )
    assert done.returncode == 0, done.stderr


def test_ctrl_c_during_the_import_ends_the_process_promptly(tmp_path):
    """Python only runs a signal handler when it has the interpreter, which inside
    Qt's C++ loop it does not unless something wakes it."""
    (tmp_path / "boot_probe_slow.py").write_text("import time\ntime.sleep(60)\n")
    code = (
        f"import os, signal, sys; sys.path.insert(0, {str(tmp_path)!r})\n"
        "from PySide6 import QtCore\n"
        "from fastfuncstuff.viewer.ui.boot import application, import_behind_window\n"
        "app = application()\n"
        "QtCore.QTimer.singleShot(100, lambda: os.kill(os.getpid(), signal.SIGINT))\n"
        "import_behind_window('boot_probe_slow')\n"
        "sys.exit(7)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 130, done.stderr


def test_the_window_shows_the_steps_the_terminal_bar_shows(tmp_path, capfd):
    """Same walk, both places: the file being waited on is named on screen too."""
    from fastfuncstuff.viewer import catalog

    heard = []
    paths = [str(tmp_path / "a.nii.gz"), str(tmp_path / "b.nii.gz")]
    with catalog.progress_to(lambda *step: heard.append(step)):
        assert list(catalog.opening(paths)) == paths
    assert heard == [("opening", 0, 2, "a.nii.gz"), ("opening", 1, 2, "b.nii.gz")]
    assert "opening" in capfd.readouterr().err

    list(catalog.opening(paths))  # the hook ends with its block: a later rescan is not told
    assert len(heard) == 2


def test_progress_fills_the_bar_and_a_long_name_cannot_widen_the_window():
    from fastfuncstuff.viewer.ui.boot import BootWindow, application

    application()
    boot = BootWindow()
    width = boot.width()
    long_name = "sub-001_task-" + "x" * 400 + ".nii.zst"
    boot.progress("opening", 0, 3, long_name)
    assert boot._bar.maximum() == 3 and boot._bar.value() == 0
    assert boot._status.text().startswith("opening 1/3")
    assert "…" in boot._status.text()
    boot.show()
    assert boot.width() == width
    boot.status("building the window")  # a phase with no count goes back to busy
    assert boot._bar.maximum() == 0
    boot.close()
