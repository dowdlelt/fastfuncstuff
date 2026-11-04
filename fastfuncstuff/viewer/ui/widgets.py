"""Small widgets that more than one part of the controller needs.

Each is here because more than one place needed it, or -- the combo picker --
because it applies to every dropdown in the viewer at once.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.ui import theme


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
        self._cap = max_rows
        self._min_rows = min_rows
        #: Pixels the list is never shorter than, asked at measure time -- for
        #: a list sharing a row with something taller, so it fills that
        #: height with rows rather than leaving a gap under itself.
        self.floor: Callable[[], int] | None = None
        # Fixed, or the enclosing layout hands it spare height anyway and the
        # hint this class exists to compute is never consulted.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        rows = max(self._min_rows, min(self.count(), self._max_rows))
        step = self.sizeHintForRow(0) if self.count() else 0
        height = rows * max(step, 18) + 2 * self.frameWidth() + 2
        if self.floor is not None:
            height = max(height, self.floor())
        return QtCore.QSize(super().sizeHint().width(), height)

    def rows_changed(self) -> None:
        """Call after adding or removing items, so the layout re-measures."""
        self.updateGeometry()

    @property
    def expanded(self) -> bool:
        return self._max_rows != self._cap

    def set_expanded(self, on: bool) -> None:
        """Every row on show (no cap), or back to the cap."""
        self._max_rows = 10**6 if on else self._cap
        self.updateGeometry()


#: Item data role a :class:`ValueColumnDelegate` draws at the row's right edge.
VALUE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class ValueColumnDelegate(QtWidgets.QStyledItemDelegate):
    """A list row with a second, right-aligned text: the row's value.

    The value is item data rather than part of the row's text so that the
    name stays the name -- to the tests, to the tick box, to anything that
    reads ``item.text()`` -- and a value changing repaints that one row
    without re-laying-out the list. The name gives way first, elided in the
    middle, because the number is what changes and what was asked for.
    """

    def paint(self, painter, option, index) -> None:  # noqa: D102
        value = index.data(VALUE_ROLE)
        if not value:
            super().paint(painter, option, index)
            return
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        # opt.font, not opt.fontMetrics: under the theme stylesheet the two
        # differ, and measuring with one while drawing with the other clips.
        metrics = QtGui.QFontMetrics(opt.font)
        gap = metrics.horizontalAdvance("  ")
        # Never more than two thirds of the row: a long region name must not
        # push the layer's own name off entirely.
        # +2: the advance is rounded and elidedText measures unrounded, so an
        # exact fit came out as "36...".
        room = max(0, min(metrics.horizontalAdvance(value) + 2, text_rect.width() * 2 // 3))
        opt.text = metrics.elidedText(
            opt.text, QtCore.Qt.TextElideMode.ElideMiddle, max(0, text_rect.width() - room - gap)
        )
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        selected = bool(opt.state & QtWidgets.QStyle.StateFlag.State_Selected)
        painter.save()
        # Selected: the theme's text, not the QPalette's HighlightedText -- the
        # stylesheet never sets that role, so it stays the system's white and
        # vanishes on the light theme's pale selection fill.
        painter.setPen(
            QtGui.QColor(theme.palette().text)
            if selected
            else opt.palette.color(QtGui.QPalette.ColorRole.Text)
        )
        painter.setFont(opt.font)
        shown = metrics.elidedText(value, QtCore.Qt.TextElideMode.ElideRight, room)
        painter.drawText(
            text_rect,
            int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter),
            shown,
        )
        painter.restore()


class PreviewCombo(QtWidgets.QComboBox):
    """A dropdown whose picker shows a row before choosing it.

    ``preview_row(i)`` is called as the picker's highlight moves, and
    ``preview_row(None)`` when it closes without a choice. For pickers whose
    choice is expensive and whose rows are hard to tell apart by name -- a
    directory of 600 datasets -- so looking costs a glance, not a commitment.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview_row: Callable[[int | None], None] | None = None


#: How long the highlight must rest before a preview is drawn. Arrowing down a
#: list emits a row per keypress; drawing each would queue work behind a key
#: that has already moved on.
PREVIEW_DEBOUNCE_MS = 60


class ComboPicker(QtWidgets.QDialog):
    """A dropdown's items as a filterable list in a small window of its own.

    A dropdown of seventeen colour scales is a long scroll through a narrow
    column; here the whole list is visible and typing narrows it. Choosing
    goes back through the combo -- ``setCurrentIndex`` and then ``activated``,
    as a click in its own popup would -- so whatever the combo is wired to
    cannot tell the difference, and nothing has to be connected twice.

    On a :class:`PreviewCombo` a click only previews; a double-click or Enter
    chooses, and closing without choosing withdraws the preview.
    """

    def __init__(self, combo: QtWidgets.QComboBox) -> None:
        super().__init__(combo.window())
        self._combo = combo
        self._preview = combo.preview_row if isinstance(combo, PreviewCombo) else None
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
        if self._preview is None:
            self.list.itemActivated.connect(self._choose)
            self.list.itemClicked.connect(self._choose)
        else:
            # Not itemActivated: under a single-click-activates style (KDE's
            # default) that fires on the click meant only to look.
            self.list.itemDoubleClicked.connect(self._choose)
            self.list.installEventFilter(self)
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
            self._preview_timer.timeout.connect(self._show_preview)
            # Connected after the initial row is set: opening the picker on
            # what is already chosen should not preview it a second time.
            self.list.currentRowChanged.connect(lambda _: self._preview_timer.start())
            self.rejected.connect(self._withdraw_preview)
        layout.addWidget(self.list)
        if self._preview is not None:
            hint = QtWidgets.QLabel("click to look  ·  double-click or Enter to load")
            hint.setObjectName("hint")
            layout.addWidget(hint)

        rows = min(combo.count(), 24)
        step = max(self.list.sizeHintForRow(0), 18) if combo.count() else 18
        width = max(self.list.sizeHintForColumn(0) + 40, combo.width(), 200)
        self.resize(width, rows * step + self.filter.sizeHint().height() + 30)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if obj is self.filter and key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
                self._step(-1 if key == QtCore.Qt.Key.Key_Up else 1)
                return True
            if obj is self.list and key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                self._choose_current()
                return True
        return super().eventFilter(obj, event)

    def _show_preview(self) -> None:
        item = self.list.currentItem()
        if self._preview is None or item is None or item.isHidden():
            return
        if not item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            return
        self._preview(int(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _withdraw_preview(self) -> None:
        self._preview_timer.stop()
        if self._preview is not None:
            self._preview(None)

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
        if self._preview is not None:
            self._preview_timer.stop()
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


class FileDropFilter(QtCore.QObject):
    """Takes local files dragged onto a claimed window, wherever on it they land.

    On the application, not the windows, because a line edit or spin box accepts
    a file drag for itself and would type its path into the field. Other windows
    -- a file dialog's own sidebar drags URLs -- are left alone.

    Always a copy: a file manager may propose a move, and accepting that is
    consent for it to delete the source. Qt keeps a move if the source offers
    nothing else, so a move-only drag is refused. The handler runs after the drop returns,
    since a load, or a dialog asking about a mesh, would otherwise hold the file
    manager waiting on the end of its drag.
    """

    def __init__(
        self,
        claims: Callable[[QtWidgets.QWidget], bool],
        dropped: Callable[[QtWidgets.QWidget, list[str]], None],
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._claims = claims
        self._dropped = dropped

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        # Enter and move events are drop events to Qt, and leave events are not.
        if not isinstance(event, QtGui.QDropEvent) or not isinstance(obj, QtWidgets.QWidget):
            return False
        window = obj.window()
        if not self._claims(window):
            return False
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return False
        if not event.possibleActions() & QtCore.Qt.DropAction.CopyAction:
            event.ignore()
            return True
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        if event.type() == QtCore.QEvent.Type.Drop:
            QtCore.QTimer.singleShot(0, self, lambda: self._dropped(window, paths))
        return True


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


__all__ = [
    "ComboPicker",
    "FileDropFilter",
    "PreviewCombo",
    "RowSizedList",
    "install_combo_pickers",
    "open_combo_picker",
]
