"""The carpet window: every voxel's time course, as one picture.

Painted from a ``(rows, T)`` array straight into a QImage, the same way a pane
is -- so a carpet of 900k voxels costs one blit rather than a path per row.

Built on the worker, because masking, detrending and sorting a whole run is
seconds. The window shows the carpet it has while the next one is computed,
which is what makes changing the ordering feel like a choice rather than a
wait.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.carpet import ORDER_LABELS, ORDERINGS, Carpet
from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    SetCarpetOrder,
    SetViewDetrend,
    SetViewScaling,
    SetViewTraces,
)

#: Width of the overlay band drawn beside the carpet, in pixels.
SIDEBAR_WIDTH = 14
BARE_WIDTH = 300
#: Vertical travel, in pixels, before a press becomes a row selection rather
#: than a click. Small enough that a short band is selectable, large enough
#: that a hand tremor on a click does not select three rows.
DRAG_START = 4


class CarpetView(QtWidgets.QWidget):
    """Paints one carpet, plus the overlay band and the time cursor."""

    #: A time index, from clicking somewhere along the carpet.
    scrubbed = QtCore.Signal(int)
    #: A row index, from the same click. A carpet has two axes and both of them
    #: mean something in the rest of the viewer -- across is the volume, down
    #: is a voxel -- so a click that moved only the time cursor was throwing
    #: half of itself away.
    rowed = QtCore.Signal(int)
    #: (first row, last row) of a click-and-drag, inclusive, in either order.
    #: Rows only: a band in time is a volume range, and nothing downstream has
    #: a use for one yet.
    selected = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._carpet: Carpet | None = None
        self._image: QtGui.QImage | None = None
        self._band: QtGui.QImage | None = None
        self._index = 0
        #: Row range currently highlighted, kept after the drag so the picture
        #: still says which rows the selection layer came from.
        self._selection: tuple[int, int] | None = None
        self._press: QtCore.QPointF | None = None
        self._dragging = False
        self.setMinimumSize(80, 60)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_carpet(self, carpet: Carpet | None) -> None:
        # A rebuilt carpet has a different ordering, so the old highlight would
        # sit over rows that now mean other voxels.
        if carpet is not self._carpet:
            self._selection = None
        self._carpet = carpet
        self._image = None if carpet is None else self._grey(carpet)
        self._band = None if carpet is None else self._sidebar(carpet)
        self.update()

    def set_index(self, index: int) -> None:
        if index != self._index:
            self._index = int(index)
            self.update()

    @staticmethod
    def _grey(carpet: Carpet) -> QtGui.QImage:
        """The carpet itself, on a symmetric grey scale.

        Grey and symmetric on purpose: the eye reads a diverging colour map as
        two categories, and a carpet's whole job is to show that one band
        behaves unlike its neighbours, not which sign it took.
        """
        limit = max(float(carpet.limit), 1e-6)
        scaled = np.clip(carpet.image / limit, -1.0, 1.0)
        grey = ((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)
        grey = np.ascontiguousarray(grey)
        h, w = grey.shape
        return QtGui.QImage(grey.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()

    @staticmethod
    def _sidebar(carpet: Carpet) -> QtGui.QImage | None:
        """The overlay's value per row, as a one-pixel-wide column.

        Amber on the dark palette's ground, because it is the only thing on
        screen that is not the data and should not be mistaken for it.
        """
        if carpet.sidebar is None:
            return None
        values = np.nan_to_num(np.abs(carpet.sidebar.astype(np.float32)))
        top = float(values.max())
        if top <= 0:
            return None
        hot = np.clip(values / top, 0.0, 1.0)
        accent = QtGui.QColor(theme.palette().key)
        ground = QtGui.QColor(theme.palette().bg)
        rgb = np.empty((hot.size, 1, 3), dtype=np.uint8)
        for ch, (a, b) in enumerate(
            (
                (ground.red(), accent.red()),
                (ground.green(), accent.green()),
                (ground.blue(), accent.blue()),
            )
        ):
            rgb[..., ch] = (a + (b - a) * hot).astype(np.uint8)[:, None]
        rgb = np.ascontiguousarray(rgb)
        return QtGui.QImage(rgb.data, 1, hot.size, 3, QtGui.QImage.Format.Format_RGB888).copy()

    def _carpet_rect(self) -> QtCore.QRect:
        left = SIDEBAR_WIDTH + 2 if self._band is not None else 0
        return QtCore.QRect(left, 0, max(self.width() - left, 1), self.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self._image is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "no carpet yet\n(load a 4-D dataset)",
            )
            p.end()
            return

        # Smooth across rows only: a carpet is always squashed vertically far
        # past one row per pixel, and nearest-neighbour there drops whole
        # voxels from the picture rather than averaging them into it.
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self._carpet_rect()
        p.drawImage(rect, self._image)
        if self._band is not None:
            p.drawImage(QtCore.QRect(0, 0, SIDEBAR_WIDTH, self.height()), self._band)

        assert self._carpet is not None
        if self._selection is not None:
            rows = max(self._carpet.shape[0], 1)
            lo, hi = sorted(self._selection)
            top = rect.top() + lo / rows * rect.height()
            bottom = rect.top() + (hi + 1) / rows * rect.height()
            accent = QtGui.QColor(c.key)
            accent.setAlpha(70)
            band = QtCore.QRectF(0.0, top, float(self.width()), max(bottom - top, 1.0))
            p.fillRect(band, accent)
            accent.setAlpha(220)
            p.setPen(QtGui.QPen(accent))
            p.drawLine(QtCore.QPointF(0.0, top), QtCore.QPointF(float(self.width()), top))
            p.drawLine(QtCore.QPointF(0.0, bottom), QtCore.QPointF(float(self.width()), bottom))
        nt = self._carpet.shape[1]
        if 0 <= self._index < nt and nt > 1:
            x = rect.left() + (self._index + 0.5) / nt * rect.width()
            pen = QtGui.QPen(QtGui.QColor(c.warn))
            pen.setWidth(1)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x, 0.0), QtCore.QPointF(x, float(self.height())))
        p.end()

    def _row_at(self, y: float) -> int:
        assert self._carpet is not None
        rect = self._carpet_rect()
        rows = self._carpet.shape[0]
        down = (y - rect.top()) / max(rect.height(), 1)
        return max(0, min(int(down * rows), rows - 1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._carpet is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._press = event.position()
        self._dragging = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._carpet is None or self._press is None:
            return
        if not self._dragging and abs(event.position().y() - self._press.y()) < DRAG_START:
            return
        self._dragging = True
        self._selection = (self._row_at(self._press.y()), self._row_at(event.position().y()))
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        """A drag selects rows; anything shorter is a click, as it always was.

        Decided on release rather than press, so starting a drag does not first
        jump the crosshair and the time cursor to wherever it began.
        """
        if self._carpet is None or self._press is None:
            return
        press, self._press = self._press, None
        if self._dragging:
            self._dragging = False
            first, last = self._row_at(press.y()), self._row_at(event.position().y())
            self._selection = (first, last)
            self.update()
            self.selected.emit(first, last)
            return
        rect = self._carpet_rect()
        nt = self._carpet.shape[1]
        frac = (press.x() - rect.left()) / max(rect.width(), 1)
        self.scrubbed.emit(int(round(max(0.0, min(1.0, frac)) * (nt - 1))))
        self.rowed.emit(self._row_at(press.y()))


class CarpetWindow(QtWidgets.QWidget):
    """A floating carpet viewport."""

    closed = QtCore.Signal(str)
    scrubbed = QtCore.Signal(int)
    #: (i, j, k) of the voxel a clicked row stands for.
    located = QtCore.Signal(int, int, int)
    #: Asks the controller to rebuild on the worker; it owns the runner.
    rebuild_requested = QtCore.Signal(str)
    #: (view id, first row, last row) of a dragged selection.
    rows_selected = QtCore.Signal(str, int, int)

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
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.resize(760, 520)
        self.setMinimumSize(120, 90)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(7, 6, 7, 7)
        v.setSpacing(5)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(5)
        self.layer_box = QtWidgets.QComboBox()
        self.layer_box.setToolTip(
            "Which run to draw. Point it at a DERIVE'd layer to see a cleaned "
            "carpet -- that is where nuisance projection lives."
        )
        self.layer_box.activated.connect(self._pick_layer)
        bar.addWidget(self.layer_box, 1)

        self.order_box = QtWidgets.QComboBox()
        for name in ORDERINGS:
            self.order_box.addItem(ORDER_LABELS[name], userData=name)
        self.order_box.setToolTip(
            "Row order (o cycles). Acquisition order is spatial; the rest put "
            "voxels that move together next to each other."
        )
        self.order_box.activated.connect(
            lambda _: self._dispatch(SetCarpetOrder(self.vid, self.order_box.currentData()))
        )
        bar.addWidget(self.order_box)
        v.addLayout(bar)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(5)
        row2.addWidget(QtWidgets.QLabel("POLORT"))
        self.polort_spin = QtWidgets.QSpinBox()
        self.polort_spin.setRange(-1, 9)
        self.polort_spin.setMaximumWidth(64)
        self.polort_spin.setToolTip(
            "Drift projected out first; a carpet is unreadable through a ramp"
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
        self.info = QtWidgets.QLabel("")
        row2.addWidget(self.info)
        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(row2)
        v.addWidget(self.controls)

        self.view = CarpetView()
        self.view.scrubbed.connect(self.scrubbed)
        self.view.rowed.connect(self._locate)
        self.view.selected.connect(lambda a, b: self.rows_selected.emit(self.vid, a, b))
        v.addWidget(self.view, 1)

        self.help = ShortcutHelp(self, f"carpet · {vid}")
        self.help.apply(
            [
                Binding("o", "next row order", self._cycle_order, group="carpet"),
                Binding(
                    "r", "rebuild", lambda: self.rebuild_requested.emit(self.vid), group="carpet"
                ),
                Binding("click", "jump to that volume and that voxel", None, group="carpet"),
                Binding(
                    "drag",
                    "select those rows' voxels as a red overlay layer",
                    None,
                    group="carpet",
                ),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- input ---------------------------------------------------------
    def _locate(self, row: int) -> None:
        """Turn a clicked row back into a place in the brain."""
        carpet = self.view._carpet
        where = carpet.voxel_of(row) if carpet is not None else None
        if where is not None:
            self.located.emit(*where)

    def _pick_layer(self, _index: int) -> None:
        key = self.layer_box.currentData()
        if key:
            self._dispatch(SetViewTraces(self.vid, str(key)))

    def _cycle_order(self) -> None:
        i = (self.order_box.currentIndex() + 1) % max(self.order_box.count(), 1)
        self.order_box.setCurrentIndex(i)
        self._dispatch(SetCarpetOrder(self.vid, self.order_box.currentData()))

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
            index = self.layer_box.findData(source.key)
            self.layer_box.setCurrentIndex(max(index, 0))
        self.layer_box.blockSignals(False)

        for box, value in ((self.order_box, viewport.order), (self.scale_box, viewport.scaling)):
            box.blockSignals(True)
            box.setCurrentIndex(max(box.findData(value), 0))
            box.blockSignals(False)
        self.polort_spin.blockSignals(True)
        self.polort_spin.setValue(int(viewport.detrend))
        self.polort_spin.blockSignals(False)

    def show_carpet(self, carpet: Carpet | None, message: str = "") -> None:
        self._stale = False
        self.view.set_carpet(carpet)
        self.info.setText(message or (carpet.status() if carpet is not None else ""))

    def mark_stale(self) -> None:
        """Say the picture predates the stack it describes.

        A carpet is seconds of work, so it is not rebuilt on every change to
        the layers -- but a sidebar drawn from an overlay that has since been
        swapped is a stale widget of the worst kind, because a carpet has no
        numbers on it to contradict. Saying so costs nothing.
        """
        if self._stale or self.view._carpet is None:
            return
        self._stale = True
        self.info.setText(f"{self.info.text()}   · stale, r rebuilds")

    def set_busy(self, busy: bool) -> None:
        self.controls.setEnabled(not busy)
        if busy:
            self.info.setText("building…")

    def refresh(self) -> None:
        """Follow the time cursor. The picture itself is rebuilt on demand."""
        self.view.set_index(self.session.state.time_index)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.view.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        self.controls.setVisible(event.size().width() >= BARE_WIDTH)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["CarpetView", "CarpetWindow"]
