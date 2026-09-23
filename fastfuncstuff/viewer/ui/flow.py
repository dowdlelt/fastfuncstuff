"""A layout that wraps its children onto as many rows as they need.

The controller is 430 pixels wide and the window bar holds a mode picker, eight
buttons, a theme toggle and a time spinner -- about 1200 pixels of them. A
``QToolBar`` does not wrap: everything past the edge goes into an overflow menu
behind a chevron, which is indistinguishable from not being there. +MATRIX and
+CLUSTERS were features nobody could find.

Qt ships no flow layout, so this is the one from its own layout examples,
translated and trimmed to what is used here.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Left to right, top to bottom, wrapping at the available width."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        margin: int = 0,
        spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items: list[QtWidgets.QLayoutItem] = []

    # -- QLayout plumbing ----------------------------------------------
    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802
        return QtCore.Qt.Orientation(0)

    # -- the wrapping itself -------------------------------------------
    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._lay_out(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._lay_out(rect, apply=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        """The widest single item, not the sum of them all.

        A flow layout can always make itself narrower by using another row, so
        the only hard floor is the one item that cannot be broken. Reporting
        the sum instead is what would force the panel wide again.
        """
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _lay_out(self, rect: QtCore.QRect, *, apply: bool) -> int:
        """Place the items in ``rect``; return the height they needed."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, row_height = area.x(), area.y(), 0
        space = self.spacing()

        for item in self._items:
            hint = item.sizeHint()
            if row_height and x + hint.width() > area.right() + 1:
                x, y = area.x(), y + row_height + space
                row_height = 0
            if apply:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + space
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y() + margins.bottom()


class FlowBar(QtWidgets.QWidget):
    """A widget holding a :class:`FlowLayout`, kept exactly as tall as it needs.

    ``QToolBar`` lays its children out by size hint and never asks one for its
    height at a given width, so a flow layout inside it keeps whatever height
    it was first given -- three rows of buttons' worth of empty toolbar after
    the window is widened enough to need one. Measuring on each resize is the
    only hook that fires at the right moment.
    """

    def __init__(self, spacing: int = 4) -> None:
        super().__init__()
        self.flow = FlowLayout(self, spacing=spacing)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def addWidget(self, widget: QtWidgets.QWidget) -> None:  # noqa: N802
        self.flow.addWidget(widget)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        wanted = self.flow.heightForWidth(max(self.width(), 1))
        # Guarded, because setFixedHeight inside a resize is a resize.
        if wanted > 0 and wanted != self.height():
            self.setFixedHeight(wanted)


__all__ = ["FlowBar", "FlowLayout"]
