"""The matrix window: every node against every other node.

Painted from the ``(K, K)`` array into a QImage the same way a carpet is, so a
400-node matrix is one blit. Around it sit the two things that make a square of
colour readable: a strip of each node's own colour along both edges, which is
what ties a row to the region drawn in the image windows, and separators on the
diagonal where a hierarchical ordering found a module boundary.

A cell is a question with an answer, so clicking one is wired: it names both
nodes, moves the crosshair to the row's region, and draws that region's time
course underneath. Without that a connectivity matrix is a texture.

Built on the worker, because averaging and correlating a whole run is seconds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.matrix import ORDER_LABELS, ORDERINGS, CorrMatrix
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    SetMatrixOrder,
    SetViewDetrend,
    SetViewRois,
    SetViewScaling,
    SetViewTraces,
)

#: Width of the identity strips along the top and left edges, in pixels.
STRIP = 10
#: Height of the selected node's time course under the matrix.
TRACE_HEIGHT = 46
BARE_WIDTH = 300


class MatrixView(QtWidgets.QWidget):
    """Paints one correlation matrix, its identity strips and its modules."""

    #: (row, column) of a clicked cell, in display order.
    picked = QtCore.Signal(int, int)
    #: Free text for the status line, from hovering.
    hovered = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix: CorrMatrix | None = None
        self._image: QtGui.QImage | None = None
        self._strip: QtGui.QImage | None = None
        self._row = -1
        self._col = -1
        self.setMouseTracking(True)
        self.setMinimumSize(90, 90)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_matrix(self, matrix: CorrMatrix | None) -> None:
        self._matrix = matrix
        self._image = None if matrix is None else self._colored(matrix)
        self._strip = None if matrix is None else self._identity_strip(matrix)
        if matrix is None or self._row >= matrix.n_nodes:
            self._row = self._col = -1
        self.update()

    @property
    def selected(self) -> int:
        return self._row

    @staticmethod
    def _colored(matrix: CorrMatrix) -> QtGui.QImage:
        """The matrix on a diverging scale, because correlation has a sign.

        The opposite choice from the carpet, and for the opposite reason: there
        a band's sign is incidental, here anticorrelation is the finding.
        """
        from fastfuncstuff.viewer.colormap import build_lut

        lut = (build_lut("redblue", 256).cpu().numpy() * 255).astype(np.uint8)
        unit = np.clip((matrix.matrix + 1.0) * 0.5, 0.0, 1.0)
        idx = np.rint(unit * 255).astype(np.int32)
        rgb = np.ascontiguousarray(lut[idx])
        h, w, _ = rgb.shape
        return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _identity_strip(matrix: CorrMatrix) -> QtGui.QImage:
        """One pixel per node in that node's own colour."""
        rgb = np.ascontiguousarray(np.asarray(matrix.colors, dtype=np.uint8).reshape(-1, 1, 3))
        return QtGui.QImage(rgb.data, 1, rgb.shape[0], 3, QtGui.QImage.Format.Format_RGB888).copy()

    # -- geometry ------------------------------------------------------
    def _plot_rect(self) -> QtCore.QRect:
        """The square the matrix occupies. Square on purpose: a stretched
        correlation matrix makes a symmetric structure look directional."""
        trace = TRACE_HEIGHT if self._matrix is not None and self._row >= 0 else 0
        avail_w = max(self.width() - STRIP - 2, 1)
        avail_h = max(self.height() - STRIP - 2 - trace, 1)
        side = max(min(avail_w, avail_h), 1)
        return QtCore.QRect(STRIP + 2, STRIP + 2, side, side)

    def _cell_at(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        if self._matrix is None:
            return None
        rect = self._plot_rect()
        k = self._matrix.n_nodes
        col = int((pos.x() - rect.left()) / max(rect.width(), 1) * k)
        row = int((pos.y() - rect.top()) / max(rect.height(), 1) * k)
        if not (0 <= row < k and 0 <= col < k):
            return None
        return row, col

    # -- painting ------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self._image is None or self._matrix is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "no matrix yet\n(load a 4-D dataset)",
            )
            p.end()
            return

        rect = self._plot_rect()
        p.drawImage(rect, self._image)
        if self._strip is not None:
            p.drawImage(QtCore.QRect(0, rect.top(), STRIP, rect.height()), self._strip)
            p.drawImage(
                QtCore.QRect(rect.left(), 0, rect.width(), STRIP),
                self._strip.transformed(QtGui.QTransform().rotate(-90)),
            )

        self._draw_modules(p, rect)
        self._draw_selection(p, rect)
        self._draw_trace(p, rect)
        p.end()

    def _draw_modules(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        """Separators where a hierarchical cut changed module.

        Drawn as full-width lines rather than boxes on the diagonal: what you
        want to see is whether a block's rows behave differently *outside* the
        block, and a box hides exactly that comparison.
        """
        assert self._matrix is not None
        if not self._matrix.blocks:
            return
        k = self._matrix.n_nodes
        pen = QtGui.QPen(QtGui.QColor(theme.palette().faint))
        pen.setWidth(1)
        p.setPen(pen)
        for edge in self._matrix.blocks:
            x = rect.left() + edge / k * rect.width()
            y = rect.top() + edge / k * rect.height()
            p.drawLine(
                QtCore.QPointF(x, float(rect.top())), QtCore.QPointF(x, float(rect.bottom()))
            )
            p.drawLine(
                QtCore.QPointF(float(rect.left()), y), QtCore.QPointF(float(rect.right()), y)
            )

    def _draw_selection(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        assert self._matrix is not None
        if self._row < 0:
            return
        k = self._matrix.n_nodes
        cell_w = rect.width() / k
        cell_h = rect.height() / k
        pen = QtGui.QPen(QtGui.QColor(theme.palette().key))
        pen.setWidth(1)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(
            QtCore.QRectF(
                rect.left() + self._col * cell_w,
                rect.top() + self._row * cell_h,
                max(cell_w, 2.0),
                max(cell_h, 2.0),
            )
        )

    def _draw_trace(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        """The selected node's own time course, under the matrix."""
        assert self._matrix is not None
        if self._row < 0 or self._row >= self._matrix.series.shape[0]:
            return
        top = rect.bottom() + 4
        height = self.height() - top - 2
        if height < 12:
            return
        series = self._matrix.series[self._row]
        span = float(np.abs(series).max()) or 1.0
        path = QtGui.QPainterPath()
        for i, value in enumerate(series):
            x = rect.left() + i / max(len(series) - 1, 1) * rect.width()
            y = top + height * (0.5 - 0.45 * float(value) / span)
            path.lineTo(x, y) if i else path.moveTo(x, y)
        pen = QtGui.QPen(QtGui.QColor(*self._matrix.colors[self._row]))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawPath(path)

    # -- input ---------------------------------------------------------
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        cell = self._cell_at(event.position())
        if cell is None or self._matrix is None:
            return
        row, col = cell
        m = self._matrix
        self.hovered.emit(f"{m.name_of(row)}  x  {m.name_of(col)}   r = {m.matrix[row, col]:+.3f}")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        cell = self._cell_at(event.position())
        if cell is None:
            return
        self._row, self._col = cell
        self.update()
        self.picked.emit(self._row, self._col)


class MatrixWindow(QtWidgets.QWidget):
    """A floating correlation-matrix viewport."""

    closed = QtCore.Signal(str)
    #: (layer key, label value) of a clicked row's ROI; -1 for a voxel bin.
    node_picked = QtCore.Signal(str, int)
    rebuild_requested = QtCore.Signal(str)

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
        self._stale = False
        self._matrix: CorrMatrix | None = None
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.resize(620, 680)
        self.setMinimumSize(140, 140)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(7, 6, 7, 7)
        v.setSpacing(5)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(5)
        self.layer_box = QtWidgets.QComboBox()
        self.layer_box.setToolTip(
            "Which run to correlate. A DERIVE'd layer gives a cleaned matrix."
        )
        self.layer_box.activated.connect(self._pick_layer)
        bar.addWidget(self.layer_box, 1)

        self.roi_box = QtWidgets.QComboBox()
        self.roi_box.setToolTip(
            "Which ROI layer supplies the nodes. Without one the nodes are "
            "bins of voxels that already move together -- structure, but "
            "unnamed structure."
        )
        self.roi_box.activated.connect(
            lambda _: self._dispatch(SetViewRois(self.vid, self.roi_box.currentData() or ""))
        )
        bar.addWidget(self.roi_box, 1)
        v.addLayout(bar)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(5)
        self.order_box = QtWidgets.QComboBox()
        for name in ORDERINGS:
            self.order_box.addItem(ORDER_LABELS[name], userData=name)
        self.order_box.setToolTip(
            "Node order (o cycles). Atlas order is a picture of the atlas's "
            "numbering; clustered puts the networks on the diagonal."
        )
        self.order_box.activated.connect(
            lambda _: self._dispatch(SetMatrixOrder(self.vid, self.order_box.currentData()))
        )
        row2.addWidget(self.order_box)

        row2.addWidget(QtWidgets.QLabel("POLORT"))
        self.polort_spin = QtWidgets.QSpinBox()
        self.polort_spin.setRange(-1, 9)
        self.polort_spin.setMaximumWidth(64)
        self.polort_spin.setToolTip(
            "Drift projected out first. Two undetrended runs correlate through "
            "their drifts before they correlate through anything else."
        )
        self.polort_spin.valueChanged.connect(
            lambda n: self._dispatch(SetViewDetrend(self.vid, int(n)))
        )
        row2.addWidget(self.polort_spin)

        self.scale_box = QtWidgets.QComboBox()
        self.scale_box.addItem("z", userData="z")
        self.scale_box.addItem("% change", userData="psc")
        self.scale_box.activated.connect(
            lambda _: self._dispatch(SetViewScaling(self.vid, self.scale_box.currentData()))
        )
        row2.addWidget(self.scale_box)
        row2.addStretch(1)
        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(row2)
        v.addWidget(self.controls)

        self.view = MatrixView()
        self.view.picked.connect(self._on_picked)
        self.view.hovered.connect(self._on_hovered)
        v.addWidget(self.view, 1)

        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("value")
        v.addWidget(self.info)

        self.help = ShortcutHelp(self, f"matrix · {vid}")
        self.help.apply(
            [
                Binding("o", "next node order", self._cycle_order, group="matrix"),
                Binding(
                    "r", "rebuild", lambda: self.rebuild_requested.emit(self.vid), group="matrix"
                ),
                Binding("click", "name a pair, and go to the row's region", None, group="matrix"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- input ---------------------------------------------------------
    def _pick_layer(self, _index: int) -> None:
        key = self.layer_box.currentData()
        if key:
            self._dispatch(SetViewTraces(self.vid, str(key)))

    def _cycle_order(self) -> None:
        i = (self.order_box.currentIndex() + 1) % max(self.order_box.count(), 1)
        self.order_box.setCurrentIndex(i)
        self._dispatch(SetMatrixOrder(self.vid, self.order_box.currentData()))

    def _on_hovered(self, text: str) -> None:
        self.info.setText(text)

    def _on_picked(self, row: int, _col: int) -> None:
        if self._matrix is None or not self._matrix.from_rois:
            return
        viewport = self._viewport()
        source = self._roi_key(viewport)
        if source:
            self.node_picked.emit(source, int(self._matrix.indices[row]))

    def _roi_key(self, viewport: Viewport | None) -> str:
        if viewport is not None and viewport.rois:
            return viewport.rois
        layers = self.session.roi_layers()
        return layers[-1].key if layers else ""

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    # -- output --------------------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        source = self.session.series_source(viewport)
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        for layer in self.session.graph_layers():
            self.layer_box.addItem(layer.name, userData=layer.key)
        if source is not None:
            self.layer_box.setCurrentIndex(max(self.layer_box.findData(source.key), 0))
        self.layer_box.blockSignals(False)

        self.roi_box.blockSignals(True)
        self.roi_box.clear()
        self.roi_box.addItem("voxel bins", userData="")
        for layer in self.session.roi_layers():
            self.roi_box.addItem(layer.name, userData=layer.key)
        self.roi_box.setCurrentIndex(max(self.roi_box.findData(viewport.rois), 0))
        self.roi_box.blockSignals(False)

        for box, value in (
            (self.order_box, viewport.matrix_order),
            (self.scale_box, viewport.scaling),
        ):
            box.blockSignals(True)
            box.setCurrentIndex(max(box.findData(value), 0))
            box.blockSignals(False)
        self.polort_spin.blockSignals(True)
        self.polort_spin.setValue(int(viewport.detrend))
        self.polort_spin.blockSignals(False)

    def show_matrix(self, matrix: CorrMatrix | None, message: str = "") -> None:
        self._stale = False
        self._matrix = matrix
        self.view.set_matrix(matrix)
        self.info.setText(message or (matrix.status() if matrix is not None else ""))

    def mark_stale(self) -> None:
        if self._stale or self._matrix is None:
            return
        self._stale = True
        self.info.setText(f"{self.info.text()}   · stale, r rebuilds")

    def set_busy(self, busy: bool) -> None:
        self.controls.setEnabled(not busy)
        if busy:
            self.info.setText("building…")

    def refresh(self) -> None:
        """Nothing here follows the crosshair; the picture is rebuilt on demand."""

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.view.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        self.controls.setVisible(event.size().width() >= BARE_WIDTH)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["MatrixView", "MatrixWindow"]
