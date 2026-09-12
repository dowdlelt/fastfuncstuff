"""Keyboard bindings, declared once and used twice.

The table here is both what gets installed and what the help lists, so a key
cannot exist without being documented or be documented without existing. That
is the whole reason it is data rather than a run of ``addAction`` calls
alongside a hand-written help string that drifts from them.

Each window declares its own table, and ``h`` shows the table for the window
that has focus -- a graph window's keys are not the main window's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.ui import theme


@dataclass(frozen=True)
class Binding:
    keys: str
    description: str
    action: Callable[[], None] | None = None
    group: str = ""
    #: Extra spellings that do the same thing but are not worth listing twice.
    aliases: tuple[str, ...] = field(default=())


def install(widget: QtWidgets.QWidget, bindings: Sequence[Binding]) -> None:
    """Register every binding on ``widget`` as a window-level shortcut."""
    for binding in bindings:
        if binding.action is None:
            continue
        for spelling in (binding.keys, *binding.aliases):
            action = QtGui.QAction(widget)
            action.setShortcut(QtGui.QKeySequence(spelling))
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(binding.action)  # type: ignore[arg-type]
            widget.addAction(action)


class ShortcutsDialog(QtWidgets.QDialog):
    """The `h` panel: what this window's keys do."""

    def __init__(
        self, title: str, bindings: Sequence[Binding], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"keys · {title}")
        self.setStyleSheet(theme.stylesheet())
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(4)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(3)
        row = 0
        last_group = None
        for binding in bindings:
            if binding.group != last_group:
                if last_group is not None:
                    row += 1
                head = QtWidgets.QLabel(binding.group.upper())
                head.setObjectName("group")
                grid.addWidget(head, row, 0, 1, 2)
                row += 1
                last_group = binding.group
            key = QtWidgets.QLabel(binding.keys)
            key.setObjectName("key")
            key.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            grid.addWidget(key, row, 0)
            grid.addWidget(QtWidgets.QLabel(binding.description), row, 1)
            row += 1
        outer.addLayout(grid)

        hint = QtWidgets.QLabel("esc or h to close")
        hint.setObjectName("group")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        outer.addSpacing(6)
        outer.addWidget(hint)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt)
        # h toggles, so the key that opened it also closes it.
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_H):
            self.close()
            return
        super().keyPressEvent(event)


class ShortcutHelp:
    """Mixin-ish helper: give a window `h` and a dialog built from its table."""

    def __init__(self, widget: QtWidgets.QWidget, title: str) -> None:
        self._widget = widget
        self._title = title
        self._bindings: list[Binding] = []
        self._dialog: ShortcutsDialog | None = None

    def apply(self, bindings: Sequence[Binding]) -> None:
        self._bindings = list(bindings)
        install(self._widget, self._bindings)

    def toggle(self) -> None:
        if self._dialog is not None and self._dialog.isVisible():
            self._dialog.close()
            return
        self._dialog = ShortcutsDialog(self._title, self._bindings, self._widget)
        self._dialog.show()
