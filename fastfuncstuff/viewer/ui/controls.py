"""Rendering a mode's declared controls, generically.

A mode returns a list of :class:`Control` specs and this builds the widgets. No
mode contributes UI code, which is the property that makes calc, GLM and ICA
cheap to add: a new mode is one file describing what it needs, not a file plus a
panel plus a wiring change.

Changes are debounced. An InstaCorr parameter that alters preparation costs
~600 ms to apply, and a slider drag emits a value per pixel, so applying every
one would queue a minute of work to answer a gesture that took a second.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6 import QtCore, QtWidgets

from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    BoolControl,
    ChoiceControl,
    Control,
    FloatControl,
    IntControl,
    OptionalFloatControl,
    PathControl,
)

#: How long to wait after the last change before applying it. Long enough to
#: swallow a drag, short enough that a deliberate single change feels immediate.
DEBOUNCE_MS = 250

#: Sliders are integers, so a float control is quantised into this many steps.
FLOAT_TICKS = 1000


class ControlPanel(QtWidgets.QWidget):
    """A form built from a mode's control declarations."""

    #: (param name, value as text) -- text so it matches SET_MODE_PARAM exactly.
    changed = QtCore.Signal(str, str)
    #: A declared button was pressed: the action's name.
    action_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._form = QtWidgets.QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(6)
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._pending: dict[str, str] = {}
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
        """Replace the panel with widgets for ``controls``."""
        while self._form.count():
            item = self._form.takeAt(0)
            if item is None:
                break
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._widgets.clear()
        self._pending.clear()

        for spec in controls:
            widget, label = self._build(spec, values.get(spec.name))
            if widget is None:
                continue
            if spec.help:
                widget.setToolTip(spec.help)
            self._widgets[spec.name] = widget
            self._form.addRow(label, widget)

        if actions:
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(5)
            for spec in actions:
                button = QtWidgets.QPushButton(spec.label.upper())
                button.setToolTip(spec.help)
                # Pending edits first: pressing APPLY straight after typing a
                # path must apply the path that was typed.
                button.clicked.connect(
                    lambda _=False, n=spec.name: (self.flush_now(), self.action_requested.emit(n))
                )
                h.addWidget(button)
                self._widgets[f"action:{spec.name}"] = button
            h.addStretch(1)
            self._form.addRow(QtWidgets.QLabel(""), row)

    def _build(
        self, spec: Control, value: object
    ) -> tuple[QtWidgets.QWidget | None, QtWidgets.QWidget]:
        label = QtWidgets.QLabel(spec.label.upper())

        if isinstance(spec, BoolControl):
            box = QtWidgets.QCheckBox()
            box.setChecked(bool(value if value is not None else spec.default))
            box.toggled.connect(lambda on, n=spec.name: self._queue(n, "1" if on else "0"))
            return box, label

        if isinstance(spec, PathControl):
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            edit = QtWidgets.QLineEdit(str(value if value is not None else spec.default))
            edit.setPlaceholderText("none")
            # On enter or focus-out, not per keystroke: a half-typed path is not
            # a parameter.
            edit.editingFinished.connect(
                lambda n=spec.name, e=edit: self._queue(n, e.text().strip(), now=True)
            )
            browse = QtWidgets.QPushButton("…")
            browse.setMaximumWidth(34)

            def pick(_=False, n=spec.name, e=edit, flt=spec.filter) -> None:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, spec.label, e.text(), flt)
                if path:
                    e.setText(path)
                    self._queue(n, path, now=True)

            browse.clicked.connect(pick)
            h.addWidget(edit, 1)
            h.addWidget(browse)
            return row, label

        if isinstance(spec, ChoiceControl):
            combo = QtWidgets.QComboBox()
            combo.addItems(list(spec.choices))
            combo.setCurrentText(str(value if value is not None else spec.default))
            combo.activated.connect(
                lambda _, n=spec.name, c=combo: self._queue(n, c.currentText(), now=True)
            )
            return combo, label

        if isinstance(spec, IntControl):
            box = QtWidgets.QSpinBox()
            box.setRange(spec.lo, spec.hi)
            box.setValue(int(value if value is not None else spec.default))
            box.valueChanged.connect(lambda v, n=spec.name: self._queue(n, str(int(v))))
            return box, label

        if isinstance(spec, OptionalFloatControl):
            return self._build_optional_float(spec, value), label

        if isinstance(spec, FloatControl):
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, FLOAT_TICKS)
            readout = QtWidgets.QLabel()
            readout.setObjectName("value")
            readout.setMinimumWidth(52)

            current = float(value if value is not None else spec.default)
            span = (spec.hi - spec.lo) or 1.0

            def to_tick(v: float) -> int:
                return int(round((v - spec.lo) / span * FLOAT_TICKS))

            def to_value(t: int) -> float:
                raw = spec.lo + (t / FLOAT_TICKS) * span
                # Snap to the declared step so the readout shows 0.05, not
                # 0.04999999999999999.
                return round(raw / spec.step) * spec.step if spec.step else raw

            slider.setValue(to_tick(current))
            readout.setText(f"{current:g}{spec.unit}")

            def on_move(t: int, n=spec.name, r=readout, u=spec.unit) -> None:
                v = to_value(t)
                r.setText(f"{v:g}{u}")
                self._queue(n, repr(float(v)))

            slider.valueChanged.connect(on_move)
            h.addWidget(slider, 1)
            h.addWidget(readout)
            return row, label

        return None, label

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
        return row

    # -- debounce ------------------------------------------------------
    def _queue(self, name: str, value: str, *, now: bool = False) -> None:
        self._pending[name] = value
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
