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
    """Register every binding on ``widget`` as a window-level shortcut.

    Refuses a table with two bindings on one key. Qt does **not** distinguish
    case in a key sequence -- ``QKeySequence("d") == QKeySequence("D")`` -- so a
    table offering `d` for one thing and `D` for another registers two actions
    on the same key, and Qt then fires *neither*, reporting an ambiguous
    overload to stderr where nobody reads it. Five pairs were written that way
    before anyone tried them. Uppercase means ``shift+d``, spelled out.
    """
    seen: dict[str, str] = {}
    for binding in bindings:
        for spelling in (binding.keys, *binding.aliases):
            if binding.action is None:
                continue
            key = QtGui.QKeySequence(spelling).toString()
            if key in seen:
                raise ValueError(
                    f"two shortcuts on {key!r}: {seen[key]!r} and {binding.description!r}. "
                    "Qt ignores case, so 'd' and 'D' are one key; write 'shift+d'."
                )
            seen[key] = binding.description

    for binding in bindings:
        if binding.action is None:
            continue
        for spelling in (binding.keys, *binding.aliases):
            action = QtGui.QAction(widget)
            action.setShortcut(QtGui.QKeySequence(spelling))
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            # Swallow QAction.triggered's `checked` argument. PySide6 passes it
            # to any slot that will accept one, so a binding written as
            # `lambda p=plane: ...` -- the natural way to capture a loop
            # variable -- silently receives False as `p` instead of the plane.
            # Every binding here is a nullary gesture, so none of them want it.
            action.triggered.connect(lambda *_, fn=binding.action: fn())
            widget.addAction(action)


#: Widget types that eat plain keystrokes when focused. Lists and combos both
#: implement type-to-search, so a focused one consumes every letter; a button
#: takes Space. None of them need the keyboard here -- each has a shortcut of
#: its own -- and a control that silently disables the shortcut table the
#: moment you click it is worse than one you cannot tab to.
_KEY_EATERS = (QtWidgets.QAbstractItemView, QtWidgets.QComboBox, QtWidgets.QAbstractButton)


def keep_keys_for_shortcuts(root: QtWidgets.QWidget) -> None:
    """Stop ``root``'s controls from swallowing single-key shortcuts.

    Clicking a layer in the list is the most common gesture in the viewer, and
    it moved focus into a QListWidget -- after which `d`, `c`, `o` and the rest
    went to its type-to-search instead of to the action table. Reported as "the
    keyboard shortcuts don't work", which is exactly what it looked like.

    Text entry is deliberately untouched: while you are typing a path into a
    line edit, the letters belong to the line edit.
    """
    for widget in root.findChildren(QtWidgets.QWidget):
        if isinstance(widget, _KEY_EATERS):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)


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
