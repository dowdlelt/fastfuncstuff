"""Small widgets that more than one part of the controller needs.

Only one so far, but it is one that two lists got wrong in two different ways,
which is the argument for it living in one place.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class RowSizedList(QtWidgets.QListWidget):
    """A list as tall as what is in it, capped, scrolling past the cap.

    The default is the opposite: a list expands to whatever space it is given,
    so a stack of two layers claimed the same 190 pixels as a stack of nine and
    spent most of it on blank rows. In a panel that is a column of sections,
    those pixels come out of whatever is below -- which is how changing a mode
    parameter pushed the colour bar off the bottom of the controller.

    The height is a size *hint* rather than a fixed height set when the items
    change, because a row's height is not known until the widget has been
    styled, and the theme's stylesheet arrives after the panel is built. Set
    eagerly, every list came out two thirds of a row short.
    """

    def __init__(self, max_rows: int, min_rows: int = 1) -> None:
        super().__init__()
        self._max_rows = max_rows
        self._min_rows = min_rows
        # Fixed, or the enclosing layout hands it spare height anyway and the
        # hint this class exists to compute is never consulted.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        rows = max(self._min_rows, min(self.count(), self._max_rows))
        step = self.sizeHintForRow(0) if self.count() else 0
        height = rows * max(step, 18) + 2 * self.frameWidth() + 2
        return QtCore.QSize(super().sizeHint().width(), height)

    def rows_changed(self) -> None:
        """Call after adding or removing items, so the layout re-measures."""
        self.updateGeometry()


__all__ = ["RowSizedList"]
