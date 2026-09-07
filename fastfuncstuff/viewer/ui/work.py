"""Running a mode's slow preparation off the GUI thread.

Mode preparation is seconds on a real dataset -- InstaCorr has to detrend,
filter and normalize the whole 4-D array. Doing that in the click handler
freezes the window, which is the exact failure the whole design exists to
avoid.

The split that makes this safe is in :class:`Mode`: ``prepare()`` is slow and
must not touch session state, ``compute()`` is fast and runs back on the GUI
thread. So the worker only ever mutates the mode's own cached arrays, and every
session mutation stays on the thread that paints.

While a preparation is running the mode's controls are disabled rather than
queued. Queueing would let a drag stack up several multi-second preparations
whose results arrive in an order nobody asked for.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtWidgets


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(bool, str)


class _PrepareTask(QtCore.QRunnable):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        self.signals = _Signals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            ok = self.mode.prepare(self.signals.progress.emit)
        except Exception as exc:  # surfaced in the status bar, never swallowed
            self.signals.finished.emit(False, f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(bool(ok), "")


class PreparationRunner(QtCore.QObject):
    """Runs ``mode.prepare`` on a worker and reports back on the GUI thread."""

    #: (fraction, message) while working.
    progress = QtCore.Signal(float, str)
    #: (succeeded, error message) when done.
    finished = QtCore.Signal(bool, str)
    #: True while a preparation is in flight.
    busy_changed = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        # One at a time: preparations are not independent -- a later one
        # supersedes an earlier one, and running both wastes a core to produce
        # a result that is immediately discarded.
        self._pool.setMaxThreadCount(1)
        self._busy = False
        self._mode = None

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, mode) -> bool:
        """Begin preparing. Returns False if one is already running."""
        if self._busy:
            return False
        self._busy = True
        self.busy_changed.emit(True)
        mode._preparing = True
        self._mode = mode
        task = _PrepareTask(mode)
        task.signals.progress.connect(self.progress, QtCore.Qt.ConnectionType.QueuedConnection)
        task.signals.finished.connect(self._on_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        self._pool.start(task)
        return True

    @QtCore.Slot(bool, str)
    def _on_finished(self, ok: bool, error: str) -> None:
        if getattr(self, "_mode", None) is not None:
            self._mode._preparing = False
        self._busy = False
        self.busy_changed.emit(False)
        self.finished.emit(ok, error)

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until idle. For tests and for shutdown, not for the UI."""
        return self._pool.waitForDone(timeout_ms)


def run_when_ready(
    runner: PreparationRunner, mode, on_ready: Callable[[], None], on_error: Callable[[str], None]
) -> bool:
    """Prepare if needed, then call ``on_ready`` on the GUI thread.

    When nothing slow is pending this calls straight through, so a cheap
    parameter change does not pay a thread hop.
    """
    if not mode.needs_prepare:
        on_ready()
        return True

    def done(ok: bool, error: str) -> None:
        runner.finished.disconnect(done)
        if ok:
            on_ready()
        elif error:
            on_error(error)

    runner.finished.connect(done)
    if not runner.start(mode):
        runner.finished.disconnect(done)
        return False
    return True
