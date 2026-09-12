"""Floating graph windows: an N x N block of voxels around the cursor.

Separate top-level windows rather than a docked pane, because looking at time
courses is something you do sometimes -- the default view is images, and a graph
that is always present is a graph that is always stealing space from them.

Two things a graph window owns, both of which the single fixed pane could not:

* **Which layers it plots.** A stack of func, anat and stats has exactly one
  thing worth drawing a line for, and once it also holds a denoised copy of the
  func, which lines you want is a choice. Selection is per window, so one graph
  can stay on the raw series while another follows the cleaned one.
* **How big the block is.** Stepped with + and -, not chosen from 1/4/9: it is
  a square that grows, and how far out you want to look depends on voxel size
  and on what you are chasing.

One widget paints the whole grid rather than nesting N**2 children. At 16 cells
the difference is not performance so much as control: a single paintEvent can
share one y-scale across every cell, which is the only way the grid is
comparable rather than sixteen separate autoscaled pictures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.slicing import plane_layout
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import SetViewGrid, SetViewSharedScale, SetViewTraces

#: Below these the header and the trace toggles go away. Nothing is lost but
#: the reminder -- + - and s still work, and a small graph beside a small image
#: is a reasonable thing to want.
BARE_WIDTH = 190
TRACES_WIDTH = 240


def _decimate(values: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample indices and values, thinned to about two points per pixel.

    Returned as an (index, value) pair rather than a shorter array so the x
    positions stay on the original time axis -- thinning the values alone would
    stretch the trace across the cell and put the time cursor in the wrong
    place.
    """
    n = int(values.size)
    budget = max(2, int(width * 2))
    if n <= budget:
        return np.arange(n), values
    idx = np.linspace(0, n - 1, budget).astype(np.intp)
    return idx, values[idx]


@dataclass
class Cell:
    ijk: tuple[int, int, int]
    traces: list[tuple[str, np.ndarray]]
    is_centre: bool


class GridGraph(QtWidgets.QWidget):
    """Paints an N x N block of time courses on one shared scale."""

    picked = QtCore.Signal(int, int, int)
    #: A time index, from clicking somewhere along a trace.
    scrubbed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells: list[Cell] = []
        self._n = 1
        self._index = 0
        self._shared_scale = True
        self.setMinimumSize(60, 48)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_cells(self, cells: list[Cell], n: int, index: int) -> None:
        self._cells, self._n, self._index = cells, n, index
        self.update()

    def set_shared_scale(self, on: bool) -> None:
        self._shared_scale = bool(on)
        self.update()

    def _time_length(self) -> int:
        """Length of the time-domain trace -- the first one, by convention."""
        for cell in self._cells:
            if cell.traces and cell.traces[0][1].size:
                return int(cell.traces[0][1].size)
        return 0

    def _cell_rects(self) -> list[QtCore.QRectF]:
        n, pad = self._n, 3
        cw = (self.width() - pad * (n + 1)) / n
        ch = (self.height() - pad * (n + 1)) / n
        return [
            QtCore.QRectF(pad + c * (cw + pad), pad + r * (ch + pad), cw, ch)
            for r in range(n)
            for c in range(n)
        ]

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        """Click anywhere on a trace to jump the whole viewer to that volume."""
        nt = self._time_length()
        if nt <= 1:
            return
        for rect in self._cell_rects():
            if not rect.contains(event.position()):
                continue
            inner = rect.adjusted(2, 2, -2, -2)
            frac = (event.position().x() - inner.left()) / max(inner.width(), 1.0)
            self.scrubbed.emit(int(round(max(0.0, min(1.0, frac)) * (nt - 1))))
            return

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
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if not self._cells:
            p.setPen(QtGui.QColor(c.faint))
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
        c = theme.palette()
        border = QtGui.QColor(c.edge_lit) if cell.is_centre else QtGui.QColor(c.edge)
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
            # Decimate to the width of the cell. At a 16x16 grid a cell is
            # about thirty pixels wide, and drawing four hundred points into
            # thirty pixels costs the whole paint and shows nothing -- the
            # y-range still comes from every sample, so the envelope is honest.
            xs, ys = _decimate(values, inner.width())
            path = QtGui.QPainterPath()
            for i, (t, v) in enumerate(zip(xs, ys, strict=True)):
                x = inner.left() + (t / max(nt - 1, 1)) * inner.width()
                y = inner.bottom() - (float(v) - lo) / span * inner.height()
                pt = QtCore.QPointF(x, y)
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*c.series[si % len(c.series)]))
            pen.setWidthF(1.6 if cell.is_centre else 1.0)
            p.setPen(pen)
            p.drawPath(path)

        # The time cursor belongs to the time-domain trace, which is the first
        # one; a spectrum has no "current time point".
        first_len = indexed[0][1][1].size
        if 0 <= self._index < first_len:
            x = inner.left() + (self._index / max(first_len - 1, 1)) * inner.width()
            cursor = QtGui.QColor(c.warn)
            cursor.setAlpha(170)
            p.setPen(QtGui.QPen(cursor))
            p.drawLine(QtCore.QPointF(x, inner.top()), QtCore.QPointF(x, inner.bottom()))

        if self._n <= 3:
            p.setPen(QtGui.QColor(c.dim))
            f = p.font()
            f.setPointSize(9)
            p.setFont(f)
            i, j, k = cell.ijk
            p.drawText(QtCore.QPointF(rect.left() + 4, rect.top() + 11), f"{i} {j} {k}")


class GraphWindow(QtWidgets.QWidget):
    """A floating graph viewport."""

    closed = QtCore.Signal(str)
    #: A time index chosen by clicking in the plot.
    scrubbed = QtCore.Signal(int)

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
        self.resize(520, 420)
        self.setMinimumSize(90, 80)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(7, 6, 7, 7)
        v.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)
        minus = QtWidgets.QPushButton("[-]")
        minus.setToolTip("Fewer voxels (-)")
        minus.clicked.connect(lambda: self.step_grid(-1))
        plus = QtWidgets.QPushButton("[+]")
        plus.setToolTip("More voxels (+)")
        plus.clicked.connect(lambda: self.step_grid(1))
        for b in (minus, plus):
            b.setMaximumWidth(42)
        bar.addWidget(minus)
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setObjectName("value")
        bar.addWidget(self.count_label)
        bar.addWidget(plus)
        bar.addSpacing(10)

        self.shared_check = QtWidgets.QCheckBox(theme.key_label("shared scale", "s"))
        self.shared_check.setToolTip(
            "One y-scale across all cells, so neighbouring voxels are comparable."
        )
        self.shared_check.clicked.connect(
            lambda on: self._dispatch(SetViewSharedScale(self.vid, bool(on)))
        )
        bar.addWidget(self.shared_check)
        bar.addStretch(1)
        self.info = QtWidgets.QLabel("")
        bar.addWidget(self.info)
        self.header = QtWidgets.QWidget()
        self.header.setLayout(bar)
        v.addWidget(self.header)

        # One toggle per plottable layer, in the trace's own colour, so a line
        # in the plot and the control that turns it off are the same object as
        # far as the eye is concerned.
        self.trace_host = QtWidgets.QWidget()
        self.trace_bar = QtWidgets.QHBoxLayout(self.trace_host)
        self.trace_bar.setContentsMargins(0, 0, 0, 0)
        self.trace_bar.setSpacing(4)
        self._trace_buttons: list[QtWidgets.QPushButton] = []
        v.addWidget(self.trace_host)

        self.graph = GridGraph()
        self.graph.scrubbed.connect(self.scrubbed)
        v.addWidget(self.graph, 1)

        # Its own table: a graph window's keys are not an image window's, and
        # `h` should show the keys of whatever has focus.
        self.help = ShortcutHelp(self, f"graph · {vid}")
        self.help.apply(
            [
                Binding(
                    "+", "more voxels", lambda: self.step_grid(1), group="grid", aliases=("=",)
                ),
                Binding("-", "fewer voxels", lambda: self.step_grid(-1), group="grid"),
                Binding("s", "shared scale", self.shared_check.click, group="grid"),
                Binding("click", "jump to that volume", None, group="grid"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        """Shed the controls as the window narrows; the keys still work."""
        size = event.size()
        self.header.setVisible(size.width() >= BARE_WIDTH)
        self.trace_host.setVisible(size.width() >= TRACES_WIDTH and size.height() >= 170)
        super().resizeEvent(event)

    # -- input ---------------------------------------------------------
    def step_grid(self, delta: int) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetViewGrid(self.vid, vp.grid_n + delta))

    def _toggle_trace(self, key: str) -> None:
        """Turn one layer's line on or off in this window.

        Stored as the explicit set of layers to keep rather than as the set to
        drop, so a layer loaded later starts plotted -- which is what someone
        who just loaded it is looking for.
        """
        vp = self._viewport()
        if vp is None:
            return
        current = [ly.key for ly in self.session.traces_for(vp)]
        if key in current:
            current.remove(key)
        else:
            current = [ly.key for ly in self.session.graph_layers() if ly.key in {*current, key}]
        self._dispatch(SetViewTraces(self.vid, ",".join(current)))

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    # -- output --------------------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        self.count_label.setText(f"{viewport.cells:>3d}")
        self.shared_check.setChecked(viewport.shared_scale)
        self.graph.set_shared_scale(viewport.shared_scale)
        self._rebuild_trace_buttons(viewport)

    def _rebuild_trace_buttons(self, viewport: Viewport) -> None:
        while self.trace_bar.count():
            item = self.trace_bar.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._trace_buttons.clear()

        plottable = self.session.graph_layers()
        shown = {ly.key for ly in self.session.traces_for(viewport)}
        for i, layer in enumerate(plottable):
            series = theme.palette().series
            colour = QtGui.QColor.fromRgbF(*series[i % len(series)])
            b = QtWidgets.QPushButton(layer.name)
            b.setCheckable(True)
            b.setChecked(layer.key in shown)
            b.setStyleSheet(f"QPushButton:checked {{ color: {colour.name()}; }}")
            b.clicked.connect(lambda _=False, k=layer.key: self._toggle_trace(k))
            self.trace_bar.addWidget(b)
            self._trace_buttons.append(b)
        if not plottable:
            self.trace_bar.addWidget(QtWidgets.QLabel("no time series loaded"))
        self.trace_bar.addStretch(1)

    def refresh(self) -> None:
        """Rebuild the cells around the current crosshair."""
        st = self.session.state
        vp = self._viewport()
        if vp is None:
            return
        n = vp.grid_n
        if st.grid is None:
            self.graph.set_cells([], n, st.time_index)
            return
        layout = plane_layout(st.grid.affine, vp.plane)
        # Walk the grid in image order, so the cells sit where the voxels
        # appear on screen rather than in array order.
        centre_row, centre_col = layout.to_image(st.crosshair, st.grid.shape)
        half = n // 2

        traced = self.session.traces_for(vp)
        cells: list[Cell] = []
        for dr in range(-half, -half + n):
            for dc in range(-half, -half + n):
                ijk = st.grid.clamp(
                    layout.to_ijk(centre_row + dr, centre_col + dc, st.crosshair, st.grid.shape)
                )
                cells.append(
                    Cell(ijk=ijk, traces=self._traces(traced, ijk), is_centre=(dr == 0 and dc == 0))
                )
        self.graph.set_cells(cells, n, st.time_index)
        self.info.setText(self.session.mode.status())

    def _traces(self, layers, ijk: tuple[int, int, int]) -> list[tuple[str, np.ndarray]]:
        """The selected layers' time courses, plus whatever the mode adds."""
        out: list[tuple[str, np.ndarray]] = []
        for layer in layers:
            values = self.session.timeseries(layer.key, ijk)
            if values.size:
                out.append((layer.name, values))
        for trace in self.session.mode_series(ijk):
            if trace.values.size:
                out.append((trace.label, trace.values))
        return out

    def restyle(self) -> None:
        """Re-read the palette after a theme switch."""
        self.setStyleSheet(theme.stylesheet())
        viewport = self._viewport()
        if viewport is not None:
            self._rebuild_trace_buttons(viewport)
        self.graph.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["Cell", "GraphWindow", "GridGraph"]
