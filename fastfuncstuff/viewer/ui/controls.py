"""Rendering a mode's declared controls, generically.

A mode returns a list of :class:`Control` specs and this builds the widgets. No
mode contributes UI code, which is the property that makes calc, GLM and ICA
cheap to add: a new mode is one file describing what it needs, not a file plus a
panel plus a wiring change.

Three things the layout does that a plain form does not, all of them because
InstaGLM has fourteen parameters and a form of fourteen full-width rows is a
scroll bar rather than a panel:

* **Spans.** A control asks for a fraction of a row and they pack left to
  right. Three pickers on one line beat three lines of one picker each, and a
  drop-down given the whole width of a panel is mostly whitespace.
* **Conditional visibility.** ``visible_when`` hides a control whose owning
  choice is not selected, *keeping its space*, so a library index does not sit
  there inviting a click while the HRF is set to custom -- and revealing it
  does not shove the rest of the panel down a row.
* **Stable widgets.** A rebuild with the same declarations reuses the widgets
  instead of deleting them. The panel is rebuilt on every refit, and a list
  someone is part-way through editing must not vanish underneath them.

Changes are debounced. An InstaCorr parameter that alters preparation costs
~600 ms to apply, and a slider drag emits a value per pixel, so applying every
one would queue a minute of work to answer a gesture that took a second.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6 import QtCore, QtWidgets

from fastfuncstuff.viewer.modes.base import (
    ROW_UNITS,
    ActionControl,
    BoolControl,
    ChoiceControl,
    Control,
    FloatControl,
    IntControl,
    OptionalFloatControl,
    PathControl,
    PathListControl,
)
from fastfuncstuff.viewer.ortvec import describe_entry, with_op
from fastfuncstuff.viewer.ui.widgets import RowSizedList

#: How long to wait after the last change before applying it. Long enough to
#: swallow a drag, short enough that a deliberate single change feels immediate.
DEBOUNCE_MS = 250

#: Sliders are integers, so a float control is quantised into this many steps.
FLOAT_TICKS = 1000

#: Rows a path list shows before it scrolls. Three is enough to see that there
#: is more than one and short enough that the list is not the panel.
PATH_LIST_ROWS = 3


def _decimals(step: float) -> int:
    """Enough decimal places to type the declared step, and no more.

    A spin box with two decimals cannot express a step of 0.005, and one with
    six turns 0.25 into ``0.250000``. Both make the number harder to read than
    the slider it is standing next to.
    """
    for places in range(7):
        if abs(round(step, places) - step) < 1e-12 and round(step, places) != 0:
            return places
    return 3


class _Cell(QtWidgets.QWidget):
    """One control's label and widget, sized so hiding it keeps its place."""

    def __init__(self, label: str, body: QtWidgets.QWidget) -> None:
        super().__init__()
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.caption = QtWidgets.QLabel(label.upper())
        h.addWidget(self.caption)
        h.addWidget(body, 1)
        policy = self.sizePolicy()
        # The whole reason a hidden control still costs a row: without this the
        # panel reflows every time an HRF choice changes and every control
        # below it moves under the pointer that is about to click one.
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)


class ControlPanel(QtWidgets.QWidget):
    """A form built from a mode's control declarations."""

    #: (param name, value as text) -- text so it matches SET_MODE_PARAM exactly.
    changed = QtCore.Signal(str, str)
    #: A declared button was pressed: the action's name.
    action_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows = QtWidgets.QVBoxLayout(self)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(6)
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._cells: dict[str, _Cell] = {}
        self._setters: dict[str, Callable[[object], None]] = {}
        self._specs: tuple[Control, ...] = ()
        self._actions: tuple[ActionControl, ...] = ()
        self._values: dict[str, object] = {}
        self._pending: dict[str, str] = {}
        self._syncing = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._flush)

    # -- building ------------------------------------------------------
    def rebuild(
        self,
        controls: Sequence[Control],
        values: dict[str, object],
        actions: Sequence[ActionControl] = (),
    ) -> None:
        """Build widgets for ``controls``, or refresh them if they are already up.

        The declarations are compared first. A mode's controls are static apart
        from the choice lists that follow a refit, so the common rebuild is a
        no-op that must not be paid for by destroying widgets: a path list
        mid-edit, a spin box with the caret in it and a slider under the thumb
        all die with the panel.
        """
        controls = tuple(controls)
        actions = tuple(actions)
        if controls == self._specs and actions == self._actions and self._widgets:
            self.sync_values(values)
            return

        while self._rows.count():
            item = self._rows.takeAt(0)
            if item is None:
                break
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._widgets.clear()
        self._cells.clear()
        self._setters.clear()
        self._pending.clear()
        self._specs, self._actions = controls, actions
        self._values = dict(values)

        row, used = self._new_row(), 0
        for spec in controls:
            body = self._build(spec, values.get(spec.name))
            if body is None:
                continue
            span = max(1, min(int(spec.span), ROW_UNITS))
            if used and (spec.newline or used + span > ROW_UNITS):
                self._close_row(row, used)
                row, used = self._new_row(), 0
            cell = _Cell(spec.label, body)
            if spec.help:
                cell.setToolTip(spec.help)
                body.setToolTip(spec.help)
            self._widgets[spec.name] = body
            self._cells[spec.name] = cell
            row.layout().addWidget(cell, span)
            used += span
        self._close_row(row, used)

        if actions:
            # A grid of three, not one row: ICA declares seven buttons, and a
            # row that long made the whole panel wider than its window.
            holder = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(holder)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(4)
            for position, spec in enumerate(actions):
                button = QtWidgets.QPushButton(spec.label.upper())
                button.setToolTip(spec.help)
                # Pending edits first: pressing APPLY straight after typing a
                # path must apply the path that was typed.
                button.clicked.connect(
                    lambda _=False, n=spec.name: (self.flush_now(), self.action_requested.emit(n))
                )
                button.setMinimumWidth(10)
                g.addWidget(button, *divmod(position, 3))
                self._widgets[f"action:{spec.name}"] = button
            holder.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
            )
            self._rows.addWidget(holder)
        # Spare height goes here rather than into the last row. A panel given
        # more space than it needs otherwise hands it to whatever is at the
        # bottom, which is how FIT and KEEP ended up four hundred pixels tall.
        self._rows.addStretch(1)

        self.apply_visibility()

    def _new_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        # Rows are as tall as what is in them. Left to grow, a row holding
        # anything vertically expandable -- a path list -- takes the whole
        # panel and pushes everything under it off the bottom.
        row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._rows.addWidget(row)
        return row

    def _close_row(self, row: QtWidgets.QWidget, used: int) -> None:
        """Pad a part-filled row so its controls keep the width they asked for.

        Without this a lone half-width control takes the whole row, and the
        span that was meant to say "this is a small thing" says nothing.
        """
        if 0 < used < ROW_UNITS:
            row.layout().addStretch(ROW_UNITS - used)

    # -- visibility ----------------------------------------------------
    def apply_visibility(self) -> None:
        """Show or hide each ``visible_when`` control against current values."""
        for spec in self._specs:
            cell = self._cells.get(spec.name)
            if cell is None or spec.visible_when is None:
                continue
            other, wanted = spec.visible_when
            cell.setVisible(str(self._values.get(other, "")) in wanted)

    @staticmethod
    def _being_edited(widget: QtWidgets.QWidget | None) -> bool:
        """Whether writing to this control now would fight whoever is using it.

        Not simply "has focus". A combo box holds focus from the moment it is
        clicked and commits on activation, so refusing to update a focused one
        means the panel stops following state as soon as anyone touches it.
        What must be left alone is a half-typed number or a slider mid-drag.
        """
        if widget is None:
            return False
        for slider in widget.findChildren(QtWidgets.QSlider):
            if slider.isSliderDown():
                return True
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None or not (focus is widget or widget.isAncestorOf(focus)):
            return False
        return isinstance(focus, QtWidgets.QLineEdit | QtWidgets.QAbstractSpinBox)

    def sync_values(self, values: dict[str, object]) -> None:
        """Write state back into widgets nobody is part-way through changing."""
        self._syncing = True
        try:
            for name, setter in self._setters.items():
                if name not in values or name in self._pending:
                    continue
                if self._being_edited(self._widgets.get(name)):
                    continue
                setter(values[name])
            self._values.update(values)
        finally:
            self._syncing = False
        self.apply_visibility()

    # -- widgets -------------------------------------------------------
    def _build(self, spec: Control, value: object) -> QtWidgets.QWidget | None:
        if isinstance(spec, BoolControl):
            return self._build_bool(spec, value)
        if isinstance(spec, PathListControl):
            return self._build_path_list(spec, value)
        if isinstance(spec, PathControl):
            return self._build_path(spec, value)
        if isinstance(spec, ChoiceControl):
            if spec.style == "radio":
                return self._build_radio(spec, value)
            return self._build_choice(spec, value)
        if isinstance(spec, IntControl):
            return self._build_int(spec, value)
        if isinstance(spec, OptionalFloatControl):
            return self._build_optional_float(spec, value)
        if isinstance(spec, FloatControl):
            return self._build_float(spec, value)
        return None

    def _build_bool(self, spec: BoolControl, value: object) -> QtWidgets.QWidget:
        box = QtWidgets.QCheckBox()
        box.setChecked(bool(value if value is not None else spec.default))
        box.toggled.connect(lambda on, n=spec.name: self._queue(n, "1" if on else "0", now=True))
        self._setters[spec.name] = lambda v, b=box: b.setChecked(bool(v))
        return box

    def _build_path(self, spec: PathControl, value: object) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        edit = QtWidgets.QLineEdit(str(value if value is not None else spec.default))
        edit.setPlaceholderText("none")
        edit.setMinimumWidth(40)
        # Show the end of a long path -- the folder name -- not its root.
        edit.setCursorPosition(len(edit.text()))
        # On enter or focus-out, not per keystroke: a half-typed path is not
        # a parameter.
        edit.editingFinished.connect(
            lambda n=spec.name, e=edit: self._queue(n, e.text().strip(), now=True)
        )
        browse = QtWidgets.QPushButton("…")
        browse.setMaximumWidth(34)

        def pick(_=False, n=spec.name, e=edit, flt=spec.filter, folder=spec.directory) -> None:
            if folder:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, spec.label, e.text())
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, spec.label, e.text(), flt)
            if path:
                e.setText(path)
                self._queue(n, path, now=True)

        browse.clicked.connect(pick)
        h.addWidget(edit, 1)
        h.addWidget(browse)
        self._setters[spec.name] = lambda v, e=edit: e.setText(str(v or ""))
        return row

    def _build_path_list(self, spec: PathListControl, value: object) -> QtWidgets.QWidget:
        """A ticked list of files, with add and remove beside it.

        Unticking rather than deleting is the feature. "What does this file buy
        me" is a question you ask of each of them in turn and then of the pair,
        and the answer is only cheap if putting one back is one click.
        """
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        listing = RowSizedList(max_rows=PATH_LIST_ROWS)
        listing.setAlternatingRowColors(True)
        listing.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        listing.setUniformItemSizes(True)
        listing.setMinimumWidth(80)

        def fill(entries) -> None:
            listing.blockSignals(True)
            listing.clear()
            for path, on in entries:
                text = describe_entry(path) if spec.transforms else path.rsplit("/", 1)[-1]
                item = QtWidgets.QListWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
                )
                listing.addItem(item)
            listing.blockSignals(False)
            listing.rows_changed()

        def entries() -> list[tuple[str, bool]]:
            out = []
            for i in range(listing.count()):
                item = listing.item(i)
                out.append(
                    (
                        str(item.data(QtCore.Qt.ItemDataRole.UserRole)),
                        item.checkState() is QtCore.Qt.CheckState.Checked,
                    )
                )
            return out

        def commit() -> None:
            if self._syncing:
                return
            self._queue(spec.name, PathListControl.encode(entries()), now=True)

        def add(_=False) -> None:
            paths, _flt = QtWidgets.QFileDialog.getOpenFileNames(self, spec.label, "", spec.filter)
            if not paths:
                return
            have = {p for p, _ in entries()}
            fill(entries() + [(p, True) for p in paths if p not in have])
            commit()

        def drop(_=False) -> None:
            chosen = {listing.row(i) for i in listing.selectedItems()}
            if not chosen:
                return
            fill([e for i, e in enumerate(entries()) if i not in chosen])
            commit()

        def derive(_=False) -> None:
            # Each derivative lands under its source and ticked, so pressing
            # the button is the whole gesture; pressing it on a derivative
            # gives the second one.
            chosen = {listing.row(i) for i in listing.selectedItems()}
            if not chosen:
                return
            have = {p for p, _ in entries()}
            out = []
            for i, (path, on) in enumerate(entries()):
                out.append((path, on))
                derived = with_op(path, "deriv")
                if i in chosen and derived not in have:
                    out.append((derived, True))
            fill(out)
            commit()

        def split(_=False) -> None:
            from fastfuncstuff.viewer.ui.bandsplit import split_interactively

            rows = sorted(listing.row(i) for i in listing.selectedItems())
            if not rows:
                return
            current = entries()
            source = current[rows[0]][0]
            added = split_interactively(source, spec.sample_interval, self)
            if not added:
                return
            # The source goes off, not away: its bands sum back to it, so
            # leaving it on makes the design collinear, but it stays one tick
            # from the before-and-after comparison the split was made for.
            have = {p for p, _ in current}
            out = []
            for i, (path, on) in enumerate(current):
                out.append((path, False if i == rows[0] else on))
                if i == rows[0]:
                    out += [(e, True) for e in added if e not in have]
            fill(out)
            commit()

        listing.itemChanged.connect(lambda _item: commit())

        buttons = QtWidgets.QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        tools = [
            ("+", add, "Add one or more files."),
            ("−", drop, "Remove the selected files. Untick instead to keep them handy."),
        ]
        if spec.transforms:
            tools += [
                (
                    "∂",
                    derive,
                    "Add the derivative of each selected entry, as an entry of its own. "
                    "Press it on a derivative for the second derivative.",
                ),
                (
                    "≋",
                    split,
                    "Split the selected entry by frequency: pick cuts on each column's "
                    "spectrum, and get one entry per band.",
                ),
            ]
        for position, (text, slot, tip) in enumerate(tools):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("tool")
            # Square and small: two stacked buttons otherwise set the row's
            # height, and an empty list next to them reads as a large blank
            # box rather than as a list with nothing in it yet.
            button.setFixedSize(22, 22)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            # Two columns: four stacked buttons would set the list's height.
            buttons.addWidget(button, *divmod(position, 2))

        fill(PathListControl.parse(value if value is not None else spec.default))
        h.addWidget(listing, 1)
        h.addLayout(buttons)
        h.setAlignment(buttons, QtCore.Qt.AlignmentFlag.AlignTop)
        self._setters[spec.name] = lambda v: fill(PathListControl.parse(v))
        return row

    def _build_choice(self, spec: ChoiceControl, value: object) -> QtWidgets.QWidget:
        combo = QtWidgets.QComboBox()
        combo.addItems(list(spec.choices))
        combo.setCurrentText(str(value if value is not None else spec.default))
        # Shrinkable. Sized to its longest choice, a picker offering "unique
        # R2" sets a floor the whole panel has to clear, and three of them on
        # one row made that floor wider than the controller. The popup is where
        # a long choice is read, and that is not elided.
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        combo.setMinimumContentsLength(5)
        view = combo.view()
        if view is not None:
            view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        combo.activated.connect(
            lambda _, n=spec.name, c=combo: self._queue(n, c.currentText(), now=True)
        )
        self._setters[spec.name] = lambda v, c=combo: c.setCurrentText(str(v))
        return combo

    def _build_radio(self, spec: ChoiceControl, value: object) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        group = QtWidgets.QButtonGroup(row)
        current = str(value if value is not None else spec.default)
        for choice in spec.choices:
            button = QtWidgets.QRadioButton(choice)
            button.setChecked(choice == current)
            group.addButton(button)
            button.toggled.connect(
                lambda on, n=spec.name, c=choice: on and self._queue(n, c, now=True)
            )
            h.addWidget(button)
        h.addStretch(1)

        def put(v: object) -> None:
            for button in group.buttons():
                button.setChecked(button.text() == str(v))

        self._setters[spec.name] = put
        return row

    def _build_int(self, spec: IntControl, value: object) -> QtWidgets.QWidget:
        box = QtWidgets.QSpinBox()
        box.setRange(spec.lo, spec.hi)
        box.setValue(int(value if value is not None else spec.default))
        box.setKeyboardTracking(False)
        box.valueChanged.connect(lambda v, n=spec.name: self._queue(n, str(int(v))))
        self._setters[spec.name] = lambda v, b=box: b.setValue(int(v))
        return box

    def _build_float(self, spec: FloatControl, value: object) -> QtWidgets.QWidget:
        """A slider for the gesture, a spin box for the number.

        Both, because they answer different questions. Dragging is how you find
        out what the parameter *does*; typing is how you say what it should be,
        and a read-only label can only ever do the first.
        """
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, FLOAT_TICKS)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setObjectName("value")
        spin.setRange(spec.lo, spec.hi)
        spin.setSingleStep(spec.step or 0.01)
        spin.setDecimals(_decimals(spec.step or 0.01))
        spin.setSuffix(spec.unit)
        # Otherwise every digit typed on the way to 10 is a parameter: 1, then
        # 10, and on a refitting mode the 1 costs a fit nobody asked for.
        spin.setKeyboardTracking(False)
        spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        current = float(value if value is not None else spec.default)
        span = (spec.hi - spec.lo) or 1.0
        guard = {"on": False}

        def to_tick(v: float) -> int:
            return int(round((v - spec.lo) / span * FLOAT_TICKS))

        def to_value(t: int) -> float:
            raw = spec.lo + (t / FLOAT_TICKS) * span
            # Snap to the declared step so the readout shows 0.05, not
            # 0.04999999999999999.
            return round(raw / spec.step) * spec.step if spec.step else raw

        slider.setValue(to_tick(current))
        spin.setValue(current)

        def on_move(t: int, n=spec.name) -> None:
            if guard["on"]:
                return
            v = to_value(t)
            guard["on"] = True
            spin.setValue(v)
            guard["on"] = False
            self._queue(n, repr(float(v)))

        def on_type(v: float, n=spec.name) -> None:
            if guard["on"]:
                return
            guard["on"] = True
            slider.setValue(to_tick(v))
            guard["on"] = False
            self._queue(n, repr(float(v)), now=True)

        slider.valueChanged.connect(on_move)
        spin.valueChanged.connect(on_type)
        h.addWidget(slider, 1)
        h.addWidget(spin)

        def put(v: object) -> None:
            guard["on"] = True
            spin.setValue(float(v))
            slider.setValue(to_tick(float(v)))
            guard["on"] = False

        self._setters[spec.name] = put
        return row

    def _build_optional_float(self, spec: OptionalFloatControl, value: object) -> QtWidgets.QWidget:
        """A checkbox beside a slider, faded when off.

        The distinction matters because ``off_value`` is usually zero and zero
        is also a legal setting: a disabled high-pass and a high-pass set to
        0 Hz look identical on a bare slider, and only one of them is a filter
        someone chose.
        """
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        check = QtWidgets.QCheckBox()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, FLOAT_TICKS)
        readout = QtWidgets.QLabel()
        readout.setObjectName("value")
        readout.setMinimumWidth(56)

        span = (spec.hi - spec.lo) or 1.0
        current = float(value if value is not None else spec.off_value)
        on = current != spec.off_value
        resting = current if on else (spec.on_value or (spec.lo + span / 2.0))

        def to_value(tick: int) -> float:
            raw = spec.lo + (tick / FLOAT_TICKS) * span
            return round(raw / spec.step) * spec.step if spec.step else raw

        def to_tick(v: float) -> int:
            return int(round((v - spec.lo) / span * FLOAT_TICKS))

        def paint(enabled: bool, v: float) -> None:
            slider.setEnabled(enabled)
            readout.setEnabled(enabled)
            readout.setText(f"{v:g}{spec.unit}" if enabled else "off")

        slider.setValue(to_tick(resting))
        check.setChecked(on)
        paint(on, current)

        def on_toggle(checked: bool) -> None:
            v = to_value(slider.value())
            paint(checked, v)
            self._queue(spec.name, repr(float(v if checked else spec.off_value)), now=True)

        def on_move(tick: int) -> None:
            v = to_value(tick)
            if not check.isChecked():
                readout.setText("off")
                return
            readout.setText(f"{v:g}{spec.unit}")
            self._queue(spec.name, repr(float(v)))

        check.toggled.connect(on_toggle)
        slider.valueChanged.connect(on_move)
        h.addWidget(check)
        h.addWidget(slider, 1)
        h.addWidget(readout)

        def put(v: object) -> None:
            val = float(v)
            enabled = val != spec.off_value
            check.blockSignals(True)
            check.setChecked(enabled)
            check.blockSignals(False)
            if enabled:
                slider.blockSignals(True)
                slider.setValue(to_tick(val))
                slider.blockSignals(False)
            paint(enabled, val)

        self._setters[spec.name] = put
        return row

    # -- debounce ------------------------------------------------------
    def _queue(self, name: str, value: str, *, now: bool = False) -> None:
        if self._syncing:
            return
        self._pending[name] = value
        # Before the debounce, not after it: a control that reveals another one
        # must reveal it on the click, not a quarter second later once the
        # refit it also triggered has come back.
        self._values[name] = value
        self.apply_visibility()
        if now:
            self._flush()
        else:
            self._timer.start()

    def _flush(self) -> None:
        pending, self._pending = self._pending, {}
        for name, value in pending.items():
            self.changed.emit(name, value)

    def flush_now(self) -> None:
        """Apply anything still pending -- on focus loss or before a save."""
        self._timer.stop()
        self._flush()


def build_mode_panel(mode, on_change: Callable[[str, str], None]) -> ControlPanel:
    """Convenience: a panel wired to one mode."""
    panel = ControlPanel()
    panel.rebuild(mode.controls(), mode.params, mode.actions())
    panel.changed.connect(on_change)
    return panel
