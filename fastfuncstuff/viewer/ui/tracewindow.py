"""A trace window: one named line a mode offers, in a window of its own.

A grid graph is voxel time courses around the crosshair. What a mode has to
show is often not a voxel at all -- an ICA component's time course and its
spectrum describe the whole brain -- and drawn into every cell of a grid it is
the same line sixteen times, too small to read. So a mode names its panels
(``timecourse``, ``spectrum``) and each one gets a window, sized by hand,
beside the images.

The window is also where the mode is driven from while you look at it: the
keys that step and label ICA components live here, so reviewing a
decomposition is left/right and a letter, without reaching for the controller.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.modes.base import Trace
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport

MARGIN_LEFT = 64
MARGIN_BOTTOM = 20
MARGIN_TOP = 20


class PlotView(QtWidgets.QWidget):
    """One line on labelled axes, with an optional time cursor."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._trace: Trace | None = None
        self._cursor: int | None = None
        self._empty = "nothing to plot"
        self.setMinimumSize(120, 80)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_trace(self, trace: Trace | None, cursor: int | None = None, empty: str = "") -> None:
        self._trace = trace
        self._cursor = cursor
        self._empty = empty or "nothing to plot"
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        trace = self._trace
        values = None if trace is None else np.asarray(trace.values, dtype=np.float64)
        if trace is None or values is None or values.size < 2:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._empty)
            p.end()
            return

        x = np.arange(values.size, dtype=np.float64) if trace.x is None else np.asarray(trace.x)
        plot = QtCore.QRectF(
            MARGIN_LEFT,
            MARGIN_TOP,
            max(self.width() - MARGIN_LEFT - 8, 1),
            max(self.height() - MARGIN_TOP - MARGIN_BOTTOM, 1),
        )
        finite = values[np.isfinite(values)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        x0, x1 = float(x[0]), float(x[-1]) if float(x[-1]) != float(x[0]) else float(x[0]) + 1.0

        def to_px(xv: float, yv: float) -> QtCore.QPointF:
            return QtCore.QPointF(
                plot.left() + (xv - x0) / (x1 - x0) * plot.width(),
                plot.bottom() - (yv - lo) / (hi - lo) * plot.height(),
            )

        p.setPen(QtGui.QPen(QtGui.QColor(c.edge)))
        p.drawRect(plot)
        if lo < 0 < hi:
            p.drawLine(to_px(x0, 0.0), to_px(x1, 0.0))

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        path = QtGui.QPainterPath()
        step = max(1, int(values.size / max(plot.width() * 2, 1)))
        for n, i in enumerate(range(0, values.size, step)):
            pt = to_px(float(x[i]), float(values[i]) if np.isfinite(values[i]) else lo)
            path.lineTo(pt) if n else path.moveTo(pt)
        pen = QtGui.QPen(QtGui.QColor(c.accent))
        pen.setWidthF(1.4)
        p.setPen(pen)
        p.drawPath(path)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        if self._cursor is not None and 0 <= self._cursor < values.size:
            p.setPen(QtGui.QPen(QtGui.QColor(c.warn)))
            cx = to_px(float(x[self._cursor]), lo).x()
            p.drawLine(QtCore.QPointF(cx, plot.top()), QtCore.QPointF(cx, plot.bottom()))

        p.setPen(QtGui.QColor(c.faint))
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        p.drawText(QtCore.QRectF(0, plot.top() - 7, MARGIN_LEFT - 4, 14), right, f"{hi:.3g}")
        p.drawText(QtCore.QRectF(0, plot.bottom() - 7, MARGIN_LEFT - 4, 14), right, f"{lo:.3g}")
        below = plot.bottom() + 3
        p.drawText(QtCore.QPointF(plot.left(), below + 11), f"{x0:.3g}")
        tail = f"{x1:.3g} {trace.x_label}".strip()
        width = QtGui.QFontMetrics(p.font()).horizontalAdvance(tail)
        p.drawText(QtCore.QPointF(plot.right() - width, below + 11), tail)
        p.setPen(QtGui.QColor(c.text))
        p.drawText(QtCore.QPointF(plot.left(), MARGIN_TOP - 6), trace.label)
        p.end()


class TraceWindow(QtWidgets.QWidget):
    """A floating window showing one of the active mode's panels."""

    closed = QtCore.Signal(str)
    #: A key asked the mode to do something: the action's name.
    action_requested = QtCore.Signal(str)

    def __init__(
        self,
        vid: str,
        session,
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.vid = vid
        self.session = session
        self._dispatch = dispatch
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.resize(520, 240)
        self.setMinimumSize(140, 90)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.view = PlotView()
        v.addWidget(self.view, 1)

        def ask(name: str) -> Callable[[], None]:
            return lambda: self.action_requested.emit(name)

        self.help = ShortcutHelp(self, f"trace · {vid}")
        self.help.apply(
            [
                Binding("Right", "next component", ask("next"), group="review", aliases=(".",)),
                Binding("Left", "previous component", ask("prev"), group="review", aliases=(",",)),
                Binding("s", "label signal, then next", ask("signal"), group="review"),
                Binding("n", "label noise, then next", ask("noise"), group="review"),
                Binding("u", "clear the label", ask("unlabel"), group="review"),
                Binding("k", "keep a copy of the map", ask("keep"), group="review"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        self.refresh()

    def refresh(self) -> None:
        viewport = self._viewport()
        if viewport is None:
            return
        mode = self.session.mode
        trace = mode.panels().get(viewport.panel)
        cursor = (
            self.session.state.time_index if trace is not None and trace.x_label == "TR" else None
        )
        self.view.set_trace(
            trace,
            cursor,
            empty=f"{mode.label} has no {viewport.panel}"
            if mode.panel_names()
            else "switch to a mode with panels",
        )

    def mark_stale(self) -> None:
        """Drawn from the mode on every refresh; never stale."""

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.view.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["PlotView", "TraceWindow"]
