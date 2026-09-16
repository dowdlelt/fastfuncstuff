"""Running slow work off the GUI thread.

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

Deriving a layer -- projecting a design's nuisance out of a run -- has the same
shape and goes through the same runner: seconds of arithmetic over a whole 4-D
array, a progress fraction, and a result installed back on the GUI thread.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable

from PySide6 import QtCore, QtWidgets


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(bool, str)
    logged = QtCore.Signal(str)


class _LineWriter(io.TextIOBase):
    """A stdout stand-in that hands finished lines to a callback.

    Line-buffered rather than per-write, because ``print`` makes two writes --
    the text, then the newline -- and emitting those separately would put every
    other log line in the pane empty.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit
        self._held = ""

    def write(self, text: str) -> int:
        self._held += text
        while "\n" in self._held:
            line, self._held = self._held.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._held:
            self._emit(self._held)
            self._held = ""


class _Task(QtCore.QRunnable):
    """Any slow job that takes a progress callback and returns truthiness.

    Mode preparation was the first; deriving a layer is the second, and it has
    the same shape -- seconds of arithmetic over a whole 4-D array that must
    not run on the thread that paints.
    """

    def __init__(self, job: Callable[[Callable[[float, str], None]], object]) -> None:
        super().__init__()
        self.job = job
        self.signals = _Signals()

    @QtCore.Slot()
    def run(self) -> None:
        # What the job prints is the job explaining itself -- ffs_moco names its
        # device, its cost function and its per-volume timing -- and in a GUI
        # session all of that went to a terminal nobody was reading. Captured
        # here rather than by each caller so any slow job gets a log for free.
        #
        # redirect_stdout swaps sys.stdout process-wide, which is only safe
        # because the runner runs exactly one job at a time.
        writer = _LineWriter(self.signals.logged.emit)
        try:
            with contextlib.redirect_stdout(writer):
                try:
                    ok = self.job(self.signals.progress.emit)
                finally:
                    writer.flush()
        except Exception as exc:  # surfaced in the status bar, never swallowed
            self.signals.finished.emit(False, f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(bool(ok), "")


class PreparationRunner(QtCore.QObject):
    """Runs ``mode.prepare`` on a worker and reports back on the GUI thread."""

    #: (fraction, message) while working.
    progress = QtCore.Signal(float, str)
    #: One line the running job printed.
    logged = QtCore.Signal(str)
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
        mode._preparing = True
        self._mode = mode
        return self.run(mode.prepare)

    def run(self, job: Callable[[Callable[[float, str], None]], object]) -> bool:
        """Run any slow job on the worker. False if one is already running."""
        if self._busy:
            return False
        self._busy = True
        self.busy_changed.emit(True)
        task = _Task(job)
        task.signals.progress.connect(self.progress, QtCore.Qt.ConnectionType.QueuedConnection)
        task.signals.logged.connect(self.logged, QtCore.Qt.ConnectionType.QueuedConnection)
        task.signals.finished.connect(self._on_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        self._pool.start(task)
        return True

    @QtCore.Slot(bool, str)
    def _on_finished(self, ok: bool, error: str) -> None:
        if getattr(self, "_mode", None) is not None:
            self._mode._preparing = False
            self._mode = None
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
