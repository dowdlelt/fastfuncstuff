"""The image pane: blit a rendered plane, draw the crosshair, take input.

The pane holds no viewer state. It is handed a :class:`PaneImage`, it reports
where the user clicked, and that is all -- every consequence goes back through
the command bus. That is what keeps a recorded session honest: there is no way
for a widget to change the view behind the recorder's back.

Painting stays cheap by converting to ``QImage`` only when the pixels change,
not on every expose event.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.compose import PaneImage
from fastfuncstuff.viewer.slicing import plane_axes
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui import theme


class ImagePane(QtWidgets.QWidget):
    """One display plane."""

    #: (row, col) in display-grid indices of the plane's two spanned axes.
    picked = QtCore.Signal(int, int)
    #: Wheel or key step through slices, in signed slice units.
    stepped = QtCore.Signal(int)
    #: Seed request (ctrl/cmd-click), same coordinates as ``picked``.
    seeded = QtCore.Signal(int, int)

    def __init__(self, plane: Plane, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self._image: QtGui.QImage | None = None
        self._pane: PaneImage | None = None
        self._cross: tuple[int, int] | None = None
        #: Voxel footprints of the open graphs, as (row, col, n_rows, n_cols)
        #: in image indices. The crosshair opens up around them.
        self._coverage: list[tuple[int, int, int, int]] = []
        self._labels: tuple[str, str, str, str] | None = None
        # Deliberately tiny. A pane's minimum is a floor under the whole
        # window, and a wall of small images is a real way to look at data.
        self.setMinimumSize(48, 48)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

    # -- content -------------------------------------------------------
    def set_pane(self, pane: PaneImage | None) -> None:
        """Install new pixels. Converts to QImage here, not in paintEvent."""
        self._pane = pane
        if pane is None:
            self._image = None
        else:
            arr = np.ascontiguousarray(pane.rgba.cpu().numpy())
            h, w = arr.shape[0], arr.shape[1]
            # Qt does not take ownership of the buffer, and a QImage over a
            # freed array paints garbage or crashes -- copy() detaches it.
            self._image = QtGui.QImage(
                arr.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888
            ).copy()
        self.update()

    @property
    def position(self) -> int | None:
        """Which slice is currently drawn, so a redraw can be skipped."""
        return None if self._pane is None else self._pane.position

    def set_layout(self, layout) -> None:
        """Take the plane's anatomical edge labels (top, right, bottom, left)."""
        self._labels = layout.labels
        self.update()

    def set_crosshair(self, row: int, col: int) -> None:
        self._cross = (int(row), int(col))
        self.update()

    def set_coverage(self, boxes: list[tuple[int, int, int, int]]) -> None:
        """Say which voxels the open graphs are reading, in image indices.

        Drawn as the crosshair's own gap rather than as a separate annotation:
        the gap already exists to keep the voxel under inspection visible, and
        a graph makes that "the voxels under inspection". Sizing it to the
        actual footprint is the difference between knowing the grid is 5x5 and
        seeing which 25 voxels that is.
        """
        boxes = [tuple(int(v) for v in b) for b in boxes]  # type: ignore[misc]
        if boxes != self._coverage:
            self._coverage = boxes  # type: ignore[assignment]
            self.update()

    # -- geometry ------------------------------------------------------
    def _target_rect(self) -> QtCore.QRect:
        """Where the image lands, letterboxed to preserve voxel aspect."""
        if self._image is None:
            return QtCore.QRect()
        iw, ih = self._image.width(), self._image.height()
        if iw == 0 or ih == 0:
            return QtCore.QRect()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = max(1, int(iw * scale)), max(1, int(ih * scale))
        return QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _to_indices(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        """Widget point to (row, col) display indices, or None if outside."""
        rect = self._target_rect()
        if self._image is None or not rect.contains(pos.toPoint()):
            return None
        fx = (pos.x() - rect.x()) / rect.width()
        fy = (pos.y() - rect.y()) / rect.height()
        # The image is (H=rows, W=cols); rows run down the widget.
        col = int(fx * self._image.width())
        row = int(fy * self._image.height())
        col = max(0, min(col, self._image.width() - 1))
        row = max(0, min(row, self._image.height() - 1))
        return (row, col)

    # -- painting ------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self._image is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self.plane.value.upper()}\nno data",
            )
            p.end()
            return

        rect = self._target_rect()
        # Nearest-neighbour: a viewer must not invent voxels that are not there.
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(rect, self._image)

        if self._cross is not None:
            self._paint_crosshair(p, rect)

        p.setPen(QtGui.QColor.fromRgbF(*c.label))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        pos = self._pane.position if self._pane is not None else 0
        p.drawText(6, 15, f"{self.plane.value.upper()}  {pos}")

        # Anatomical edge labels. An upside-down or mirrored brain still looks
        # like a brain, so the only thing that says which way round it is, is
        # writing it on the edges.
        if self._labels is not None:
            p.setPen(QtGui.QColor(c.edge_label))
            top, right, bottom, left = self._labels
            r = self.rect()
            flags = QtCore.Qt.AlignmentFlag
            p.drawText(r.adjusted(0, 2, 0, 0), flags.AlignTop | flags.AlignHCenter, top)
            p.drawText(r.adjusted(0, 0, -4, 0), flags.AlignRight | flags.AlignVCenter, right)
            p.drawText(r.adjusted(0, 0, 0, -2), flags.AlignBottom | flags.AlignHCenter, bottom)
            p.drawText(r.adjusted(4, 0, 0, 0), flags.AlignLeft | flags.AlignVCenter, left)
        p.end()

    def _paint_crosshair(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        assert self._cross is not None and self._image is not None
        row, col = self._cross
        iw, ih = self._image.width(), self._image.height()
        sx, sy = rect.width() / iw, rect.height() / ih
        x = rect.x() + (col + 0.5) * sx
        y = rect.y() + (row + 0.5) * sy

        cross = theme.palette().crosshair
        colour = QtGui.QColor.fromRgbF(*cross, 0.85)
        pen = QtGui.QPen(colour)
        pen.setWidth(1)
        p.setPen(pen)

        # Each footprint drawn, and the largest sets the gap. Two graphs at
        # different sizes read as nested squares, which is what they are.
        gap_x = gap_y = 5.0
        faint = QtGui.QColor.fromRgbF(*cross, 0.55)
        for brow, bcol, nrows, ncols in self._coverage:
            bx, by = rect.x() + bcol * sx, rect.y() + brow * sy
            box = QtCore.QRectF(bx, by, ncols * sx, nrows * sy)
            p.setPen(QtGui.QPen(faint))
            p.drawRect(box)
            gap_x = max(gap_x, max(x - box.left(), box.right() - x))
            gap_y = max(gap_y, max(y - box.top(), box.bottom() - y))
        p.setPen(pen)

        # A gap at the centre so the voxel -- or the block of voxels a graph is
        # reading -- stays visible. AFNI's xhair gap, and the reason it exists.
        p.drawLine(QtCore.QLineF(rect.left(), y, x - gap_x, y))
        p.drawLine(QtCore.QLineF(x + gap_x, y, rect.right(), y))
        p.drawLine(QtCore.QLineF(x, rect.top(), x, y - gap_y))
        p.drawLine(QtCore.QLineF(x, y + gap_y, x, rect.bottom()))

    # -- input ---------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        idx = self._to_indices(event.position())
        if idx is None:
            return
        mods = event.modifiers()
        if mods & (
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.seeded.emit(*idx)
        else:
            self.picked.emit(*idx)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if not (event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return
        idx = self._to_indices(event.position())
        if idx is not None:
            self.picked.emit(*idx)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta().y()
        if delta:
            self.stepped.emit(1 if delta > 0 else -1)

    def axes(self) -> tuple[int, int, int]:
        return plane_axes(self.plane)
