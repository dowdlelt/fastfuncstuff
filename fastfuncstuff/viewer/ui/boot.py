"""The window on screen while the imports behind the real one run.

Torch is about half of ffs_viewer's startup, and the controller cannot be built
without it, so the only way to answer quickly is to show something else first.
This module has to stay importable without torch -- that is its whole purpose,
and ``tests/viewer/test_boot.py`` is the tripwire.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from importlib import import_module
from types import ModuleType

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.ui.theme import stylesheet

#: How often the wait wakes Python: to see the import finish, and to let a
#: Ctrl+C run its handler, which cannot fire while idle in Qt's C++ loop.
_POLL_MS = 50

#: Each redraw is time taken from the file walk it reports on.
_MIN_DRAW_S = 0.05

_INTERRUPTED_EXIT = 130


def application() -> QtWidgets.QApplication:
    """The process's QApplication, made on first call."""
    # The GTK accessibility bridge prints a dbind warning at every start on a
    # desktop whose at-spi bus it cannot reach, and nothing here uses it.
    # setdefault, so someone who needs a screen reader can still turn it on.
    os.environ.setdefault("NO_AT_BRIDGE", "1")
    app = QtWidgets.QApplication.instance()
    return app if isinstance(app, QtWidgets.QApplication) else QtWidgets.QApplication([])


class BootWindow(QtWidgets.QWidget):
    """Same title, palette and place as the controller, which replaces it."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("nexus")
        self.setStyleSheet(stylesheet())
        self.importing = False
        self._status = QtWidgets.QLabel("starting")
        # Elided rather than wrapped: a long path has no spaces to wrap on, and it
        # would otherwise set the window's minimum width.
        self._status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)  # busy, until something can say how far along it is
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._drawn_at = 0.0
        self._layout = layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._bar)
        layout.addStretch(1)

        screen = QtGui.QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 500, 820)
        self.resize(500, min(820, area.height()))
        self.move(area.topLeft())

    def status(self, text: str) -> None:
        self._bar.setRange(0, 0)
        self._show(text)

    def progress(self, what: str, index: int, total: int, name: str) -> None:
        """A file walk's step, as ``catalog.progress_to`` reports it."""
        now = time.perf_counter()
        # A directory of cached headers walks far faster than anyone can read, and
        # the first step always draws: one slow file may be all there is to see.
        if index and now - self._drawn_at < _MIN_DRAW_S:
            return
        self._drawn_at = now
        self._bar.setRange(0, total)
        self._bar.setValue(index)
        self._show(f"{what} {index + 1}/{total}  {name}")

    def _show(self, text: str) -> None:
        metrics = self._status.fontMetrics()
        room = self.width() - 2 * self._layout.contentsMargins().left()
        self._status.setText(metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideMiddle, room))
        # repaint, not processEvents: once the real window is being built, pumping
        # the loop would run its timers against a half-constructed window.
        self._status.repaint()
        self._bar.repaint()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.importing:
            # Not a normal exit: finalizing the interpreter under a thread still
            # inside torch's import can crash, and there is nothing to clean up.
            os._exit(0)
        super().closeEvent(event)


def import_behind_window(module: str) -> tuple[BootWindow, ModuleType]:
    """Show a :class:`BootWindow`, import ``module`` off the GUI thread, return both.

    The caller closes the window once the real one is up. Whatever the import
    raises is raised here, on the calling thread.
    """
    application()
    boot = BootWindow()
    boot.status("importing torch and the viewer")
    boot.importing = True
    boot.show()
    # Stays installed until launch() replaces it, which covers the window build.
    signal.signal(signal.SIGINT, lambda *_: os._exit(_INTERRUPTED_EXIT))

    loaded: list[ModuleType] = []
    failed: list[BaseException] = []

    def work() -> None:
        try:
            loaded.append(import_module(module))
        except BaseException as exc:  # noqa: BLE001 -- handed to the calling thread
            failed.append(exc)

    worker = threading.Thread(target=work, name="viewer-import", daemon=True)
    worker.start()

    wait = QtCore.QEventLoop()
    poll = QtCore.QTimer()
    poll.timeout.connect(lambda: None if worker.is_alive() else wait.quit())
    poll.start(_POLL_MS)
    wait.exec()
    poll.stop()
    boot.importing = False

    if failed:
        boot.close()
        raise failed[0]
    return boot, loaded[0]
