"""Small widgets that more than one part of the controller needs.

Each is here because more than one place needed it, or -- the combo picker --
because it applies to every dropdown in the viewer at once.
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


class ComboPicker(QtWidgets.QDialog):
    """A dropdown's items as a filterable list in a small window of its own.

    A dropdown of seventeen colour scales is a long scroll through a narrow
    column; here the whole list is visible and typing narrows it. Choosing
    goes back through the combo -- ``setCurrentIndex`` and then ``activated``,
    as a click in its own popup would -- so whatever the combo is wired to
    cannot tell the difference, and nothing has to be connected twice.
    """

    def __init__(self, combo: QtWidgets.QComboBox) -> None:
        super().__init__(combo.window())
        self._combo = combo
        self.setWindowTitle(_combo_title(combo))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._narrow)
        self.filter.returnPressed.connect(self._choose_current)
        # Arrow keys in the filter box move the selection, so the hand can
        # stay on the keyboard between typing and choosing.
        self.filter.installEventFilter(self)
        layout.addWidget(self.filter)

        self.list = QtWidgets.QListWidget()
        icon_size = combo.iconSize()
        self.list.setIconSize(QtCore.QSize(icon_size.width() * 2, icon_size.height()))
        for i in range(combo.count()):
            item = QtWidgets.QListWidgetItem(combo.itemIcon(i), combo.itemText(i))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            if not combo.model().flags(combo.model().index(i, 0)) & (
                QtCore.Qt.ItemFlag.ItemIsEnabled
            ):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        self.list.setCurrentRow(max(combo.currentIndex(), 0))
        self.list.itemActivated.connect(self._choose)
        self.list.itemClicked.connect(self._choose)
        layout.addWidget(self.list)

        rows = min(combo.count(), 24)
        step = max(self.list.sizeHintForRow(0), 18) if combo.count() else 18
        width = max(self.list.sizeHintForColumn(0) + 40, combo.width(), 200)
        self.resize(width, rows * step + self.filter.sizeHint().height() + 30)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self.filter and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
                self._step(-1 if key == QtCore.Qt.Key.Key_Up else 1)
                return True
        return super().eventFilter(obj, event)

    def _visible_rows(self) -> list[int]:
        return [r for r in range(self.list.count()) if not self.list.item(r).isHidden()]

    def _step(self, delta: int) -> None:
        rows = self._visible_rows()
        if not rows:
            return
        at = self.list.currentRow()
        pos = rows.index(at) + delta if at in rows else 0
        self.list.setCurrentRow(rows[max(0, min(pos, len(rows) - 1))])

    def _narrow(self, text: str) -> None:
        needle = text.strip().lower()
        for r in range(self.list.count()):
            item = self.list.item(r)
            item.setHidden(bool(needle) and needle not in item.text().lower())
        rows = self._visible_rows()
        if rows and self.list.currentRow() not in rows:
            self.list.setCurrentRow(rows[0])

    def _choose_current(self) -> None:
        item = self.list.currentItem()
        if item is not None and not item.isHidden():
            self._choose(item)

    def _choose(self, item: QtWidgets.QListWidgetItem) -> None:
        if not item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            return
        index = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        combo = self._combo
        self.accept()
        combo.setCurrentIndex(index)
        combo.activated.emit(index)
        combo.textActivated.emit(combo.itemText(index))


def _combo_title(combo: QtWidgets.QComboBox) -> str:
    """The name the combo goes by on screen: its form label, else its tooltip."""
    parent = combo.parentWidget()
    layout = parent.layout() if parent is not None else None
    if isinstance(layout, QtWidgets.QFormLayout):
        label = layout.labelForField(combo)
        if isinstance(label, QtWidgets.QLabel) and label.text():
            # Strip the [k]ey brackets theme.key_label puts in a heading.
            return label.text().replace("[", "").replace("]", "").strip()
    tip = combo.toolTip().strip()
    return tip.splitlines()[0] if tip else "Choose"


class _ComboPickerFilter(QtCore.QObject):
    """Right-click on any dropdown in the application opens its picker."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if (
            event.type() == QtCore.QEvent.Type.ContextMenu
            and isinstance(obj, QtWidgets.QComboBox)
            and obj.isEnabled()
            and obj.count() > 0
        ):
            open_combo_picker(obj)
            return True
        return False


def open_combo_picker(combo: QtWidgets.QComboBox) -> ComboPicker:
    """Show ``combo``'s picker beside it, and return the dialog."""
    picker = ComboPicker(combo)
    below = combo.mapToGlobal(QtCore.QPoint(0, combo.height()))
    screen = combo.screen().availableGeometry()
    x = min(below.x(), screen.right() - picker.width())
    y = min(below.y(), screen.bottom() - picker.height())
    picker.move(max(x, screen.left()), max(y, screen.top()))
    # Modal to its window: the combo cannot be repopulated under an open picker.
    picker.open()
    picker.filter.setFocus()
    return picker


def install_combo_pickers(app: QtCore.QCoreApplication) -> None:
    """Give every dropdown in ``app`` a right-click picker. Safe to call twice.

    One application-wide filter rather than a combo subclass, so it reaches
    the dropdowns in every window -- carpet, matrix, graph -- and every one
    added later, without anyone having to remember to use it.
    """
    if app.findChild(_ComboPickerFilter) is not None:
        return
    app.installEventFilter(_ComboPickerFilter(app))


__all__ = ["ComboPicker", "RowSizedList", "install_combo_pickers", "open_combo_picker"]
