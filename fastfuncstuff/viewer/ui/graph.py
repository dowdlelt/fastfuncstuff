"""The graph pane: time courses at the crosshair.

Draws every time-linked layer on one set of axes, which is what makes the
multi-echo view a matter of loading several layers rather than a special mode.

Series arrive already extracted; the pane never touches the residency store. A
layer that is not resident yet contributes an empty array and is simply skipped,
so moving the crosshair during a background load neither blocks nor blanks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

#: Distinguishable at a glance on a dark ground, and colour-blind safe.
SERIES_RGB = (
    (0.49, 0.89, 0.76),
    (0.98, 0.65, 0.35),
    (0.45, 0.70, 0.95),
    (0.85, 0.75, 0.35),
    (0.80, 0.55, 0.85),
)


@dataclass(frozen=True)
class Series:
    label: str
    values: np.ndarray


class GraphPane(QtWidgets.QWidget):
    """Time courses, with a marker at the current time index."""

    scrubbed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[Series] = []
        self._index = 0
        self.setMinimumHeight(120)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def set_series(self, series: list[Series], index: int) -> None:
        self._series = [s for s in series if s.values.size > 0]
        self._index = int(index)
        self.update()

    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(38, 8, max(1, self.width() - 48), max(1, self.height() - 30))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(7, 9, 11))
        rect = self._plot_rect()

        if not self._series:
            p.setPen(QtGui.QColor(65, 82, 90))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "no time series at this voxel",
            )
            p.end()
            return

        n = max(s.values.size for s in self._series)
        lo = min(float(np.nanmin(s.values)) for s in self._series)
        hi = max(float(np.nanmax(s.values)) for s in self._series)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = lo - 1.0, lo + 1.0

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtGui.QPen(QtGui.QColor(30, 39, 44)))
        p.drawRect(rect)

        def to_point(i: int, v: float) -> QtCore.QPointF:
            x = rect.left() + (i / max(n - 1, 1)) * rect.width()
            y = rect.bottom() - (v - lo) / (hi - lo) * rect.height()
            return QtCore.QPointF(x, y)

        for si, s in enumerate(self._series):
            path = QtGui.QPainterPath()
            for i, v in enumerate(s.values):
                pt = to_point(i, float(v))
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*SERIES_RGB[si % len(SERIES_RGB)]))
            pen.setWidthF(1.3)
            p.setPen(pen)
            p.drawPath(path)

        if 0 <= self._index < n:
            x = rect.left() + (self._index / max(n - 1, 1)) * rect.width()
            p.setPen(QtGui.QPen(QtGui.QColor(217, 164, 65, 190)))
            p.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))

        p.setPen(QtGui.QColor(85, 101, 112))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(QtCore.QPointF(4, rect.top() + 8), f"{hi:.4g}")
        p.drawText(QtCore.QPointF(4, rect.bottom()), f"{lo:.4g}")
        p.drawText(QtCore.QPointF(rect.left(), rect.bottom() + 15), "0")
        p.drawText(QtCore.QPointF(rect.right() - 20, rect.bottom() + 15), str(n - 1))

        legend = "   ".join(s.label for s in self._series)
        p.setPen(QtGui.QColor(107, 125, 132))
        p.drawText(QtCore.QPointF(rect.left() + 26, rect.bottom() + 15), legend)
        p.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        self._scrub_to(event.position().x())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._scrub_to(event.position().x())

    def _scrub_to(self, x: float) -> None:
        if not self._series:
            return
        n = max(s.values.size for s in self._series)
        rect = self._plot_rect()
        frac = (x - rect.left()) / max(rect.width(), 1.0)
        self.scrubbed.emit(int(round(max(0.0, min(1.0, frac)) * (n - 1))))
