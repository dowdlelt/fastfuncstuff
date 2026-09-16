"""The form a mode's button opens: pick an input, set a few things, run it.

Every widget here is generic. The dialog is handed a
:class:`~fastfuncstuff.viewer.modes.base.DialogSpec` and knows nothing about
motion correction or about any other tool -- which is the point, because a tool
that needed window code would not be one file.

Two things make this a dialog rather than another section of the controller
panel. A mode's declared action runs synchronously on the GUI thread, and a
motion correction is minutes, so it needs the worker and a progress bar of its
own. And the parameters are per-run rather than mode state: closing the window
is how you say you are done, instead of leaving a half-set form in the panel.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import DialogSpec
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.controls import ControlPanel
from fastfuncstuff.viewer.ui.work import PreparationRunner


class ToolDialog(QtWidgets.QDialog):
    """One tool's parameters, its run button, and what it says while working."""

    def __init__(
        self,
        spec: DialogSpec,
        runner: PreparationRunner,
        on_installed: Callable[[Aspect], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._on_installed = on_installed
        self._spec = spec
        self._params: dict[str, object] = {}
        self._running = False

        self.setWindowTitle(spec.title)
        self.setStyleSheet(theme.stylesheet())
        # Modeless: the whole point is to run something and then look at the
        # result, which means reaching the crosshair without closing this.
        self.setModal(False)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        self._blurb = QtWidgets.QLabel()
        self._blurb.setWordWrap(True)
        self._blurb.setObjectName("group")
        self._blurb.setMinimumWidth(340)
        outer.addWidget(self._blurb)

        self.panel = ControlPanel()
        self.panel.changed.connect(self._on_changed)
        outer.addWidget(self.panel)

        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        self._status.setObjectName("group")
        outer.addWidget(self._status)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setMaximumHeight(6)
        self._bar.hide()
        outer.addWidget(self._bar)

        # Folded away by default. The tool says what it is doing on one line;
        # this is where it says how. Worth having open while learning what a
        # step costs, and worth having shut the rest of the time.
        self._details = QtWidgets.QToolButton()
        self._details.setText("details")
        self._details.setCheckable(True)
        self._details.setAutoRaise(True)
        self._details.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self._details.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._details.toggled.connect(self._show_details)
        self._details.hide()
        outer.addWidget(self._details, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QtGui.QFont(theme.MONO, theme.FONT_SMALL))
        self._log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._log.setMinimumHeight(140)
        # A long run is not a reason to hold a megabyte of scrollback.
        self._log.setMaximumBlockCount(2000)
        self._log.hide()
        outer.addWidget(self._log)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self._close = QtWidgets.QPushButton("CLOSE")
        self._close.clicked.connect(self.close)
        row.addWidget(self._close)
        self._run = QtWidgets.QPushButton("RUN")
        self._run.setDefault(True)
        self._run.clicked.connect(self._start)
        row.addWidget(self._run)
        outer.addLayout(row)

        self.reseed(spec)

    # -- state ---------------------------------------------------------
    @property
    def tool_name(self) -> str:
        return self._spec.name

    def reseed(self, spec: DialogSpec) -> None:
        """Rebuild the form from a fresh spec -- new layers, new input choices.

        Never while a run is in flight: replacing the form under a running job
        would change the parameters out from under the result on its way back.
        """
        if self._running:
            return
        self._spec = spec
        self._params = dict(spec.params)
        self._blurb.setText(spec.blurb)
        self._blurb.setVisible(bool(spec.blurb))
        self.panel.rebuild(spec.controls, spec.params)
        self._run.setText(spec.run_label.upper())
        self._set_blocked(spec.blocked)

    def _set_blocked(self, why: str) -> None:
        self._run.setEnabled(not why)
        self.panel.setEnabled(not why)
        self._status.setText(why)
        self._status.setVisible(bool(why))

    def _on_changed(self, name: str, value: str) -> None:
        self._params[name] = value

    # -- running -------------------------------------------------------
    def _start(self) -> None:
        spec = self._spec
        if self._running or spec.run is None or spec.install is None:
            return
        # Bound to locals so the closures below capture the narrowed callables
        # rather than the optional fields.
        run, install = spec.run, spec.install
        # Pending edits first: pressing RUN straight after picking from a combo
        # must run with what is on screen, not with what was there before.
        self.panel.flush_now()
        params = dict(self._params)

        held: dict[str, object] = {}

        def job(progress) -> bool:
            held["result"] = run(params, progress)
            return True

        def done(ok: bool, error: str) -> None:
            self._runner.finished.disconnect(done)
            self._runner.progress.disconnect(self._on_progress)
            self._runner.logged.disconnect(self._on_logged)
            self._set_running(False)
            if not ok:
                self._say(error or f"{spec.title} failed")
                return
            try:
                dirty = install(held.get("result"))
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                self._say(f"{type(exc).__name__}: {exc}")
                return
            self._say(f"{spec.title} done — {self._made()}")
            self._on_installed(dirty)

        self._set_running(True)
        self._say("starting…")
        # Cleared per run: the log describes this attempt, not the history of
        # every interpolation you have tried.
        self._log.clear()
        self._details.hide()
        self._runner.progress.connect(self._on_progress)
        self._runner.logged.connect(self._on_logged)
        self._runner.finished.connect(done)
        if not self._runner.run(job):
            self._runner.finished.disconnect(done)
            self._runner.progress.disconnect(self._on_progress)
            self._runner.logged.disconnect(self._on_logged)
            self._set_running(False)
            self._say("busy with something else; try again in a moment")

    def _made(self) -> str:
        chosen = str(self._params.get("input") or "")
        return f"new layer above {chosen}" if chosen else "new layer added"

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._run.setEnabled(not running and not self._spec.blocked)
        self.panel.setEnabled(not running and not self._spec.blocked)
        self._bar.setVisible(running)
        if running:
            # Indeterminate until something reports a real fraction. A bar
            # pinned at 0% for two minutes reads as hung.
            self._bar.setRange(0, 0)

    @QtCore.Slot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        if fraction > 0.0:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(max(0.0, min(fraction, 1.0)) * 100))
        if message:
            self._say(message)

    def _say(self, text: str) -> None:
        self._status.setText(text)
        self._status.setVisible(bool(text))

    @QtCore.Slot(str)
    def _on_logged(self, line: str) -> None:
        self._log.appendPlainText(line)
        # Only offered once there is something to read, so a tool that prints
        # nothing does not grow a disclosure arrow onto an empty box.
        self._details.show()

    def _show_details(self, open_: bool) -> None:
        self._log.setVisible(open_)
        self._details.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if open_ else QtCore.Qt.ArrowType.RightArrow
        )
        # Shrink back to the form when it folds, rather than leaving a tall
        # empty dialog behind.
        if not open_:
            self.adjustSize()

    # -- closing -------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt)
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        # A run already on the worker is left to finish -- its result still
        # installs, because the work is done and throwing it away would be
        # worse than a layer appearing after the window went.
        super().closeEvent(event)


__all__ = ["ToolDialog"]
