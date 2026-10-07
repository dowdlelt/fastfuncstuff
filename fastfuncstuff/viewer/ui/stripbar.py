"""The row of neighbour-slice cells under an image window's pane.

Only widgets: which slices, what crop and what each cell says come from
:mod:`viewer.strip`, and every gesture goes back to the window as a signal.
The left half and the right half are split by a marker standing in for the
main view, so the row reads as "below | here | above".
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.panes import ImagePane


class _Centre(QtWidgets.QWidget):
    """A thin bar where the main view sits in the sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(4)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        colour = QtGui.QColor.fromRgbF(*theme.palette().crosshair, 0.7)
        p.fillRect(self.rect().adjusted(1, 6, -1, -6), colour)
        p.end()


class StripBar(QtWidgets.QWidget):
    """``count`` small panes, half either side of a centre marker."""

    #: (cell index, row, col) in that cell's image pixels.
    picked = QtCore.Signal(int, int, int)
    #: Right-drag on any cell: multiply the strip's zoom.
    zoomed = QtCore.Signal(float)
    #: Wheel on any cell: step the main view, as the wheel does everywhere.
    stepped = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._row = QtWidgets.QHBoxLayout(self)
        self._row.setContentsMargins(0, 2, 0, 0)
        self._row.setSpacing(2)
        self.cells: list[ImagePane] = []
        self._centre = _Centre()
        self.setMinimumHeight(40)

    def set_count(self, count: int, plane: Plane) -> None:
        """Rebuild the row when the number of cells changes; keep it otherwise."""
        if count == len(self.cells):
            for cell in self.cells:
                cell.plane = plane
            return
        for cell in self.cells:
            self._row.removeWidget(cell)
            cell.deleteLater()
        self._row.removeWidget(self._centre)
        self.cells = []
        for i in range(count):
            if i == count // 2:
                self._row.addWidget(self._centre)
            cell = ImagePane(plane)
            cell.caption = ""
            cell.drag_picks = False
            # The window keeps the keyboard: a cell is something to look at
            # and click, never somewhere typing should go.
            cell.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            cell.setMinimumSize(32, 32)
            cell.picked.connect(lambda r, c, i=i: self.picked.emit(i, r, c))
            cell.zoomed.connect(self.zoomed.emit)
            cell.stepped.connect(self.stepped.emit)
            self._row.addWidget(cell, 1)
            self.cells.append(cell)

    def preferred_height(self, width: int) -> int:
        """Square cells across ``width``, within reason."""
        n = max(1, len(self.cells))
        return int(max(48, min(220, (width - 4 - 2 * n) / n)))


__all__ = ["StripBar"]
