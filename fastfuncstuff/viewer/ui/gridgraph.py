"""Floating grid-graph windows: 1, 4 or 9 voxels around the cursor.

Separate top-level windows rather than a docked pane, because looking at time
courses is something you do sometimes -- the default view is images, and a graph
that is always present is a graph that is always stealing space from them.

One widget paints the whole N x N grid rather than nesting N**2 child widgets.
At 9 cells the difference is not performance so much as control: a single
paintEvent can share one y-scale across every cell, which is the only way the
grid is comparable rather than nine separate autoscaled pictures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.slicing import plane_axes
from fastfuncstuff.viewer.state import Plane

SERIES_RGB = (
    (0.49, 0.89, 0.76),
    (0.98, 0.65, 0.35),
    (0.45, 0.70, 0.95),
    (0.85, 0.75, 0.35),
    (0.80, 0.55, 0.85),
)
GRID_SIZES = (1, 2, 3)  # 1, 4, 9 voxels


@dataclass
class Cell:
    ijk: tuple[int, int, int]
    traces: list[tuple[str, np.ndarray]]
    is_centre: bool


class GridGraph(QtWidgets.QWidget):
    """Paints an N x N block of time courses on one shared scale."""

    picked = QtCore.Signal(int, int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells: list[Cell] = []
        self._n = 1
        self._index = 0
        self._shared_scale = True
        self.setMinimumSize(220, 180)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_cells(self, cells: list[Cell], n: int, index: int) -> None:
        self._cells, self._n, self._index = cells, n, index
        self.update()

    def set_shared_scale(self, on: bool) -> None:
        self._shared_scale = bool(on)
        self.update()

    def _bounds(self, cells: list[Cell]) -> list[tuple[float, float]]:
        """One y-range per trace index, shared across every cell.

        Per *trace* rather than one range for everything, because the traces on
        a cell are routinely in different units -- raw BOLD counts next to a
        filtered, mean-zero signal. A single shared range flattens the smaller
        one onto the axis and makes it useless. Sharing across cells is what
        keeps neighbouring voxels comparable, which is the point of the grid.
        """
        n_traces = max((len(c.traces) for c in cells), default=0)
        out: list[tuple[float, float]] = []
        for i in range(n_traces):
            vals = [c.traces[i][1] for c in cells if len(c.traces) > i and c.traces[i][1].size]
            if not vals:
                out.append((0.0, 1.0))
                continue
            lo = float(min(float(np.nanmin(v)) for v in vals))
            hi = float(max(float(np.nanmax(v)) for v in vals))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = lo - 1.0, lo + 1.0
            out.append((lo, hi))
        return out

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(7, 9, 11))
        if not self._cells:
            p.setPen(QtGui.QColor(65, 82, 90))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "no time series here\n(load a 4-D dataset)",
            )
            p.end()
            return

        n = self._n
        pad = 3
        cw = (self.width() - pad * (n + 1)) / n
        ch = (self.height() - pad * (n + 1)) / n
        shared = self._bounds(self._cells) if self._shared_scale else None
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        for idx, cell in enumerate(self._cells):
            r, c = divmod(idx, n)
            rect = QtCore.QRectF(pad + c * (cw + pad), pad + r * (ch + pad), cw, ch)
            self._paint_cell(p, rect, cell, shared)
        p.end()

    def _paint_cell(
        self,
        p: QtGui.QPainter,
        rect: QtCore.QRectF,
        cell: Cell,
        shared: list[tuple[float, float]] | None,
    ) -> None:
        border = QtGui.QColor(45, 90, 100) if cell.is_centre else QtGui.QColor(30, 39, 44)
        p.setPen(QtGui.QPen(border))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(rect)

        indexed = [(i, t) for i, t in enumerate(cell.traces) if t[1].size]
        if not indexed:
            return
        bounds = shared if shared is not None else self._bounds([cell])
        inner = rect.adjusted(2, 2, -2, -2)
        # Each trace spans the full width using its OWN length. Traces here are
        # not always the same axis -- a 60-bin spectrum beside a 120-point time
        # course is Hz beside TR -- so scaling both to the longest would squash
        # the shorter one into a fraction of the cell and imply a shared x that
        # does not exist.
        cursor_len = 0

        for si, (_, values) in indexed:
            lo, hi = bounds[si] if si < len(bounds) else (0.0, 1.0)
            span = hi - lo or 1.0
            nt = values.size
            cursor_len = max(cursor_len, nt)
            path = QtGui.QPainterPath()
            for i, v in enumerate(values):
                x = inner.left() + (i / max(nt - 1, 1)) * inner.width()
                y = inner.bottom() - (float(v) - lo) / span * inner.height()
                pt = QtCore.QPointF(x, y)
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*SERIES_RGB[si % len(SERIES_RGB)]))
            pen.setWidthF(1.6 if cell.is_centre else 1.0)
            p.setPen(pen)
            p.drawPath(path)

        # The time cursor belongs to the time-domain trace, which is the first
        # one; a spectrum has no "current time point".
        first_len = indexed[0][1][1].size
        if 0 <= self._index < first_len:
            x = inner.left() + (self._index / max(first_len - 1, 1)) * inner.width()
            p.setPen(QtGui.QPen(QtGui.QColor(217, 164, 65, 160)))
            p.drawLine(QtCore.QPointF(x, inner.top()), QtCore.QPointF(x, inner.bottom()))

        if self._n <= 3:
            p.setPen(QtGui.QColor(85, 101, 112))
            f = p.font()
            f.setPointSize(7)
            p.setFont(f)
            i, j, k = cell.ijk
            p.drawText(QtCore.QPointF(rect.left() + 4, rect.top() + 11), f"{i} {j} {k}")


class GridGraphWindow(QtWidgets.QWidget):
    """A floating graph window bound to one plane."""

    closed = QtCore.Signal(str)

    def __init__(self, plane: Plane, session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.session = session
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(f"graph · {plane.value}")
        self.resize(460, 380)
        self.setStyleSheet(
            "QWidget { background: #07090B; color: #C9D6DA; }"
            "QLabel { color: #6B7D84; font-size: 10px; letter-spacing: 1px; }"
            "QComboBox, QCheckBox { color: #C9D6DA; font-size: 11px; }"
            "QComboBox { background: #0E1216; border: 1px solid #1E272C; padding: 2px 5px; }"
        )

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(5)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("VOXELS"))
        self.size_box = QtWidgets.QComboBox()
        self.size_box.addItems(["1", "4", "9"])
        self.size_box.setCurrentIndex(1)
        self.size_box.currentIndexChanged.connect(lambda _: self.refresh())
        bar.addWidget(self.size_box)
        self.shared_check = QtWidgets.QCheckBox("shared scale")
        self.shared_check.setChecked(True)
        self.shared_check.setToolTip(
            "One y-scale across all cells, so neighbouring voxels are comparable."
        )
        self.shared_check.toggled.connect(self._on_shared)
        bar.addWidget(self.shared_check)
        bar.addStretch(1)
        self.info = QtWidgets.QLabel("")
        bar.addWidget(self.info)
        v.addLayout(bar)

        self.graph = GridGraph()
        v.addWidget(self.graph, 1)
        self.refresh()

    def _on_shared(self, on: bool) -> None:
        self.graph.set_shared_scale(on)

    @property
    def grid_n(self) -> int:
        return GRID_SIZES[self.size_box.currentIndex()]

    def refresh(self) -> None:
        """Rebuild the cells around the current crosshair."""
        st = self.session.state
        n = self.grid_n
        _, r_ax, c_ax = plane_axes(self.plane)
        half = n // 2

        cells: list[Cell] = []
        for dr in range(-half, -half + n):
            for dc in range(-half, -half + n):
                ijk = list(st.crosshair)
                ijk[r_ax] += dr
                ijk[c_ax] += dc
                if st.grid is not None:
                    ijk = list(st.grid.clamp(tuple(ijk)))  # type: ignore[arg-type]
                cells.append(
                    Cell(
                        ijk=(ijk[0], ijk[1], ijk[2]),
                        traces=self._traces(tuple(ijk)),  # type: ignore[arg-type]
                        is_centre=(dr == 0 and dc == 0),
                    )
                )
        self.graph.set_cells(cells, n, st.time_index)
        self.info.setText(self.session.mode.status() or f"{n * n} voxels")

    def _traces(self, ijk: tuple[int, int, int]) -> list[tuple[str, np.ndarray]]:
        """Layer time courses plus whatever the active mode contributes."""
        out: list[tuple[str, np.ndarray]] = []
        for layer in self.session.state.layers:
            if not layer.time_linked:
                continue
            values = self.session.timeseries(layer.key, ijk)
            if values.size:
                out.append((layer.name, values))
        for trace in self.session.mode_series(ijk):
            if trace.values.size:
                out.append((trace.label, trace.values))
        return out

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.plane.value)
        super().closeEvent(event)
