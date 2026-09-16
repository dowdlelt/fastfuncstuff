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
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    SetIndex,
    SetViewGrid,
    SetViewHidden,
    SetViewSharedScale,
    SetViewTraces,
)

#: Longest name a legend tick box shows; the full one is its tooltip.
LEGEND_CHARS = 22

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
    #: ``(identity, values)``. The identity -- a layer key or ``mode:<key>`` --
    #: is what picks the colour and the shared range, so hiding one line leaves
    #: every other line its own colour instead of shifting them all along.
    traces: list[tuple[str, np.ndarray]]
    is_centre: bool


@dataclass(frozen=True)
class Entry:
    """One tickable line in a graph window's legend."""

    ident: str
    name: str
    full: str
    #: Whether the line is a time course -- what the time cursor and a
    #: click-to-scrub are measured along. A spectrum is not.
    is_time: bool = True


class FlowLayout(QtWidgets.QLayout):
    """Left to right, wrapping onto new rows. Qt ships none.

    A stack with a run, its denoised copy and a mode's two lines is four tick
    boxes, and one row of those is wider than a graph window beside an image.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802 (Qt)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 (Qt)
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 (Qt)
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802 (Qt)
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt)
        return self._arrange(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802 (Qt)
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt)
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802 (Qt)
        """The largest single item. Wrapped height comes from heightForWidth."""
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(self._effective(item))
        return size

    @staticmethod
    def _effective(item: QtWidgets.QLayoutItem) -> QtCore.QSize:
        """How much room the item will really take, not what it asks for.

        A styled QCheckBox here hints 187x8 while its own minimumSizeHint is
        198x15 -- the indicator has a floor the hint does not know about, and
        Qt honours the floor when it draws. Laying out by the hint packs rows
        8px apart for widgets that come out 15px tall, so the bottom row
        overruns the host and slides under the plot below it: visible, and
        slightly covered.

        QLayoutItem.minimumSize does not rescue this (it reports the same 8),
        so the widget has to be asked directly.
        """
        size = item.sizeHint().expandedTo(item.minimumSize())
        widget = item.widget()
        return size if widget is None else size.expandedTo(widget.minimumSizeHint())

    def _arrange(self, rect: QtCore.QRect, *, move: bool) -> int:
        x, y, row_h = rect.x(), rect.y(), 0
        gap = self.spacing()
        for item in self._items:
            hint = self._effective(item)
            if x + hint.width() > rect.right() and row_h > 0:
                x, y, row_h = rect.x(), y + row_h + gap, 0
            if move:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + gap
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y()


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
        self._colors: dict[str, QtGui.QColor] = {}
        self._time: set[str] = set()
        self.setMinimumSize(60, 48)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_cells(
        self,
        cells: list[Cell],
        n: int,
        index: int,
        colors: dict[str, QtGui.QColor] | None = None,
        time_keys: set[str] | None = None,
    ) -> None:
        self._cells, self._n, self._index = cells, n, index
        if colors is not None:
            self._colors = colors
        if time_keys is not None:
            self._time = time_keys
        self.update()

    def _time_trace(self, cell: Cell) -> np.ndarray | None:
        """The first time-domain line in a cell; a spectrum has no 'now'."""
        for ident, values in cell.traces:
            if values.size and (not self._time or ident in self._time):
                return values
        return None

    def set_shared_scale(self, on: bool) -> None:
        self._shared_scale = bool(on)
        self.update()

    def _time_length(self) -> int:
        """Length of the first time-domain trace in any cell."""
        for cell in self._cells:
            values = self._time_trace(cell)
            if values is not None:
                return int(values.size)
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

    def _bounds(self, cells: list[Cell]) -> dict[str, tuple[float, float]]:
        """One y-range per line, shared across every cell.

        Per *trace* rather than one range for everything, because the traces on
        a cell are routinely in different units -- raw BOLD counts next to a
        filtered, mean-zero signal. A single shared range flattens the smaller
        one onto the axis and makes it useless. Sharing across cells is what
        keeps neighbouring voxels comparable, which is the point of the grid.
        """
        grouped: dict[str, list[np.ndarray]] = {}
        for cell in cells:
            for ident, values in cell.traces:
                if values.size:
                    grouped.setdefault(ident, []).append(values)
        out: dict[str, tuple[float, float]] = {}
        for ident, vals in grouped.items():
            lo = float(min(float(np.nanmin(v)) for v in vals))
            hi = float(max(float(np.nanmax(v)) for v in vals))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = lo - 1.0, lo + 1.0
            out[ident] = (lo, hi)
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
        shared: dict[str, tuple[float, float]] | None,
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

        for si, (ident, values) in indexed:
            lo, hi = bounds.get(ident, (0.0, 1.0))
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
            colour = self._colors.get(ident) or QtGui.QColor.fromRgbF(*c.series[si % len(c.series)])
            pen = QtGui.QPen(colour)
            pen.setWidthF(1.6 if cell.is_centre else 1.0)
            p.setPen(pen)
            p.drawPath(path)

        # The time cursor belongs to a time-domain trace; a spectrum has no
        # "current time point", and with every time course ticked off there is
        # no cursor to draw.
        timeline = self._time_trace(cell)
        first_len = 0 if timeline is None else timeline.size
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

        # One tick box per line -- every plottable layer and every line the mode
        # adds -- in the line's own colour, so a line in the plot and the
        # control that turns it off are the same object as far as the eye is
        # concerned. Wraps rather than widening the window.
        self.trace_host = QtWidgets.QWidget()
        self.trace_bar = FlowLayout(self.trace_host)
        self._trace_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._legend: tuple = ()
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
                Binding(
                    "Right",
                    "next volume",
                    lambda: self.step_time(1),
                    group="time",
                    aliases=(".",),
                ),
                Binding(
                    "Left",
                    "previous volume",
                    lambda: self.step_time(-1),
                    group="time",
                    aliases=(",",),
                ),
                Binding("click", "jump to that volume", None, group="grid"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

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

    def step_time(self, delta: int) -> None:
        """Step the shared volume index, wrapping, as , and . do everywhere."""
        st = self.session.state
        hi = st.max_time_index()
        if hi > 0:
            self._dispatch(SetIndex((st.time_index + delta) % (hi + 1)))

    def _toggle(self, ident: str, on: bool) -> None:
        """Tick one line on or off in this window.

        Off is recorded in ``hidden``. On also re-admits a layer a script left
        out of ``traces``, so a tick box never shows checked for a line that
        is not drawn.
        """
        vp = self._viewport()
        if vp is None:
            return
        hidden = [k for k in vp.hidden if k != ident]
        if not on:
            hidden.append(ident)
        elif not ident.startswith("mode:") and vp.traces and ident not in vp.traces:
            keep = {*vp.traces, ident}
            order = [ly.key for ly in self.session.graph_layers() if ly.key in keep]
            self._dispatch(SetViewTraces(self.vid, ",".join(order)))
        self._dispatch(SetViewHidden(self.vid, ",".join(hidden)))

    def entries(self, viewport: Viewport) -> list[Entry]:
        """Every line this window could draw, in a fixed order: layers, then the mode's."""
        out = []
        for layer in self.session.graph_layers():
            stem = layer.name.removesuffix(".gz").removesuffix(".nii")
            out.append(Entry(layer.key, _clip(stem), layer.name))
        st = self.session.state
        if st.grid is not None:
            for trace in self.session.mode_series(st.crosshair):
                out.append(
                    Entry(
                        f"mode:{trace.ident}",
                        _clip(trace.legend),
                        trace.label,
                        is_time=trace.x_label in ("", "TR"),
                    )
                )
        return out

    def _drawn(self, viewport: Viewport) -> set[str]:
        layers = {ly.key for ly in self.session.traces_for(viewport)}
        return {
            e.ident
            for e in self.entries(viewport)
            if e.ident not in viewport.hidden and (e.ident.startswith("mode:") or e.ident in layers)
        }

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    # -- output --------------------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        self.count_label.setText(f"{viewport.cells:>3d}")
        self.shared_check.setChecked(viewport.shared_scale)
        self.graph.set_shared_scale(viewport.shared_scale)
        self._sync_legend(viewport)

    def _colors(self, entries: list[Entry]) -> dict[str, QtGui.QColor]:
        """A colour per line by its place in the full list, not the drawn one."""
        series = theme.palette().series
        return {
            e.ident: QtGui.QColor.fromRgbF(*series[i % len(series)]) for i, e in enumerate(entries)
        }

    def _sync_legend(self, viewport: Viewport, *, force: bool = False) -> None:
        """Rebuild the tick boxes only when the set of lines or their state moved.

        Checked on every refresh, because a mode's lines appear without any
        layer changing -- ICA loading a folder is one -- but rebuilt only on a
        real difference, since a crosshair drag refreshes at display rate.
        """
        entries = self.entries(viewport)
        drawn = self._drawn(viewport)
        signature = (tuple((e.ident, e.name) for e in entries), tuple(sorted(drawn)))
        if signature == self._legend and not force:
            return
        self._legend = signature
        while self.trace_bar.count():
            item = self.trace_bar.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._trace_checks.clear()

        faint = theme.palette().faint
        for entry, colour in zip(entries, self._colors(entries).values(), strict=True):
            box = QtWidgets.QCheckBox(entry.name)
            box.setToolTip(entry.full)
            on = entry.ident in drawn
            box.setChecked(on)
            box.setStyleSheet(
                f"QCheckBox {{ color: {colour.name() if on else faint}; }}"
                f"QCheckBox::indicator:checked {{ background: {colour.name()};"
                f" border: 1px solid {colour.name()}; }}"
            )
            box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            box.toggled.connect(lambda checked, k=entry.ident: self._toggle(k, bool(checked)))
            self.trace_bar.addWidget(box)
            self._trace_checks[entry.ident] = box
        if not entries:
            self.trace_bar.addWidget(QtWidgets.QLabel("no time series loaded"))
        self.trace_host.updateGeometry()

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

        self._sync_legend(vp)
        entries = self.entries(vp)
        drawn = self._drawn(vp)
        traced = [ly for ly in self.session.traces_for(vp) if ly.key in drawn]
        cells: list[Cell] = []
        for dr in range(-half, -half + n):
            for dc in range(-half, -half + n):
                ijk = st.grid.clamp(
                    layout.to_ijk(centre_row + dr, centre_col + dc, st.crosshair, st.grid.shape)
                )
                cells.append(
                    Cell(
                        ijk=ijk,
                        traces=self._traces(traced, ijk, drawn),
                        is_centre=(dr == 0 and dc == 0),
                    )
                )
        self.graph.set_cells(
            cells,
            n,
            st.time_index,
            colors=self._colors(entries),
            time_keys={e.ident for e in entries if e.is_time},
        )
        self.info.setText(self.session.mode.status())

    def _traces(
        self, layers, ijk: tuple[int, int, int], drawn: set[str]
    ) -> list[tuple[str, np.ndarray]]:
        """The ticked layers' time courses, plus the mode's ticked lines."""
        out: list[tuple[str, np.ndarray]] = []
        for layer in layers:
            values = self.session.timeseries(layer.key, ijk)
            if values.size:
                out.append((layer.key, values))
        for trace in self.session.mode_series(ijk):
            ident = f"mode:{trace.ident}"
            if trace.values.size and ident in drawn:
                out.append((ident, trace.values))
        return out

    def restyle(self) -> None:
        """Re-read the palette after a theme switch."""
        self.setStyleSheet(theme.stylesheet())
        viewport = self._viewport()
        if viewport is not None:
            self._sync_legend(viewport, force=True)
        self.graph.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


def _clip(text: str) -> str:
    return text if len(text) <= LEGEND_CHARS else text[: LEGEND_CHARS - 1] + "…"


__all__ = ["Cell", "Entry", "FlowLayout", "GraphWindow", "GridGraph"]
