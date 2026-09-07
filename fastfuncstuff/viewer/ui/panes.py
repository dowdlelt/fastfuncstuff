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

CROSSHAIR_RGB = (0.35, 0.95, 0.85)
LABEL_RGB = (0.55, 0.65, 0.70)


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
        self.setMinimumSize(160, 160)
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

    def set_crosshair(self, row: int, col: int) -> None:
        self._cross = (int(row), int(col))
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
        p.fillRect(self.rect(), QtGui.QColor(7, 9, 11))
        if self._image is None:
            p.setPen(QtGui.QColor(65, 82, 90))
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
            row, col = self._cross
            x = rect.x() + (col + 0.5) / self._image.width() * rect.width()
            y = rect.y() + (row + 0.5) / self._image.height() * rect.height()
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*CROSSHAIR_RGB, 0.85))
            pen.setWidth(1)
            p.setPen(pen)
            # A gap at the centre so the voxel under inspection stays visible --
            # AFNI's xhair gap, and the reason it exists.
            gap = 5
            p.drawLine(int(rect.left()), int(y), int(x - gap), int(y))
            p.drawLine(int(x + gap), int(y), int(rect.right()), int(y))
            p.drawLine(int(x), int(rect.top()), int(x), int(y - gap))
            p.drawLine(int(x), int(y + gap), int(x), int(rect.bottom()))

        p.setPen(QtGui.QColor.fromRgbF(*LABEL_RGB))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        pos = self._pane.position if self._pane is not None else 0
        p.drawText(6, 15, f"{self.plane.value.upper()}  {pos}")
        p.end()

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
