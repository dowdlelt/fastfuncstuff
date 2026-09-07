"""The colour bar, and the controls that belong on it.

Min, threshold and max are edited on the bar rather than in a form elsewhere.
Splitting them was a mistake worth naming: a control that sets a number and a
picture that shows it are two views of one thing, and keeping them apart both
doubles the wiring and invites exactly the bug it produced -- the bar going
stale because a colour change dirtied "colormap" and the bar only listened for
"layers".

The bar is drawn from the same LUT, sign mode and pane count the slices use, so
it is a legend rather than a decoration. Sub-threshold values are dimmed rather
than blanked, matching what the alpha ramp does to the image: those voxels are
still drawn, just faintly.
"""

from __future__ import annotations

import torch
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.colormap import apply_colormap, build_lut
from fastfuncstuff.viewer.layers import AlphaMode, SignMode

BAR_HEIGHT = 26
TICKS = 1000


class ColorBar(QtWidgets.QWidget):
    """The gradient itself, with threshold markers."""

    clicked = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._lut_name = "gray"
        self._lo, self._hi = 0.0, 1.0
        self._threshold = 0.0
        self._sign = SignMode.BOTH
        self._panes = 0
        self._alpha = AlphaMode.OFF
        self.setMinimumHeight(BAR_HEIGHT)
        self.setMaximumHeight(BAR_HEIGHT)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def configure(self, layer) -> None:
        self._lut_name = layer.colormap
        self._lo = float(layer.range_lo if layer.range_lo is not None else 0.0)
        self._hi = float(layer.range_hi if layer.range_hi is not None else 1.0)
        self._threshold = float(layer.threshold)
        self._sign = layer.sign_mode
        self._panes = int(layer.n_panes)
        self._alpha = layer.alpha_mode
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(7, 9, 11))
        w = max(self.width() - 2, 1)
        bar = QtCore.QRect(1, 1, w, BAR_HEIGHT - 2)
        try:
            lut = build_lut(self._lut_name, 256, device=torch.device("cpu"))
        except KeyError:
            p.end()
            return

        values = torch.linspace(self._lo, self._hi, w)
        rgb = apply_colormap(
            values,
            lut=lut,
            lo=self._lo,
            hi=self._hi,
            sign_mode=self._sign,
            n_panes=self._panes,
        )
        for x in range(w):
            r, g, b = (float(c) for c in rgb[x])
            if self._threshold > 0 and abs(float(values[x])) < self._threshold:
                fade = 0.28 if self._alpha is AlphaMode.OFF else 0.55
                r, g, b = r * fade, g * fade, b * fade
            p.fillRect(
                QtCore.QRect(bar.x() + x, bar.y(), 1, bar.height()),
                QtGui.QColor.fromRgbF(r, g, b),
            )

        p.setPen(QtGui.QPen(QtGui.QColor(30, 39, 44)))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(bar)

        span = (self._hi - self._lo) or 1.0
        if self._threshold > 0:
            p.setPen(QtGui.QPen(QtGui.QColor(217, 164, 65)))
            for edge in (self._threshold, -self._threshold):
                if not (self._lo <= edge <= self._hi):
                    continue
                if self._sign is SignMode.POS and edge < 0:
                    continue
                if self._sign is SignMode.NEG and edge > 0:
                    continue
                x = bar.x() + int((edge - self._lo) / span * w)
                p.drawLine(x, bar.top(), x, bar.bottom())
        p.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        """Click the bar to set the threshold to the value under the cursor."""
        w = max(self.width() - 2, 1)
        frac = max(0.0, min(1.0, (event.position().x() - 1) / w))
        self.clicked.emit(self._lo + frac * (self._hi - self._lo))


class RangeBar(QtWidgets.QWidget):
    """Colour bar plus the three numbers that define it, in one place.

    Layout, top to bottom: the gradient, a threshold slider spanning exactly
    the bar's width so the handle lines up with the colour it selects, and the
    min / threshold / max editors sitting under the ends and the middle they
    control.
    """

    range_changed = QtCore.Signal(float, float)
    threshold_changed = QtCore.Signal(float)
    autorange_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_hi = 1.0
        self._syncing = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)

        self.bar = ColorBar()
        self.bar.clicked.connect(self._threshold_from_bar)
        v.addWidget(self.bar)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, TICKS)
        # Flush with the bar so the handle points at the colour it thresholds.
        self.slider.setContentsMargins(0, 0, 0, 0)
        self.slider.valueChanged.connect(self._threshold_from_slider)
        v.addWidget(self.slider)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.min_spin = self._spin("lowest value shown")
        self.thr_spin = self._spin("threshold: values nearer zero than this are cut")
        self.max_spin = self._spin("highest value shown")
        self.thr_spin.setStyleSheet("color: #D9A441;")
        self.auto_button = QtWidgets.QPushButton("auto")
        self.auto_button.setToolTip("Re-derive min and max from the data")
        self.auto_button.setMaximumWidth(46)
        self.auto_button.clicked.connect(self.autorange_requested)

        row.addWidget(self.min_spin, 1)
        row.addWidget(self.thr_spin, 1)
        row.addWidget(self.max_spin, 1)
        row.addWidget(self.auto_button)
        v.addLayout(row)

        self.min_spin.valueChanged.connect(self._emit_range)
        self.max_spin.valueChanged.connect(self._emit_range)
        self.thr_spin.valueChanged.connect(self._threshold_from_spin)

    @staticmethod
    def _spin(tip: str) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1e9, 1e9)
        spin.setKeyboardTracking(False)  # commit on enter or focus-out
        spin.setToolTip(tip)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return spin

    # -- incoming ------------------------------------------------------
    def configure(self, layer) -> None:
        """Show one layer. The single entry point, so nothing can go stale."""
        self._syncing = True
        try:
            self.bar.configure(layer)
            lo = float(layer.range_lo if layer.range_lo is not None else 0.0)
            hi = float(layer.range_hi if layer.range_hi is not None else 1.0)
            step = max(abs(hi - lo) / 100.0, 1e-4)
            for spin, value in (
                (self.min_spin, lo),
                (self.max_spin, hi),
                (self.thr_spin, float(layer.threshold)),
            ):
                spin.setSingleStep(step)
                spin.setValue(value)
            # The slider spans the larger half of the range, so a one-sided map
            # does not waste half its travel on values it never shows.
            self._layer_hi = max(abs(hi), abs(lo)) or 1.0
            self.slider.setValue(int(round(min(layer.threshold / self._layer_hi, 1.0) * TICKS)))
        finally:
            self._syncing = False

    # -- outgoing ------------------------------------------------------
    def _emit_range(self) -> None:
        if self._syncing:
            return
        lo, hi = self.min_spin.value(), self.max_spin.value()
        if lo == hi:
            return  # a collapsed range shows nothing; wait for the other box
        self.range_changed.emit(lo, hi)

    def _apply_threshold(self, value: float) -> None:
        if self._syncing:
            return
        self.threshold_changed.emit(max(0.0, float(value)))

    def _threshold_from_slider(self, tick: int) -> None:
        self._apply_threshold(tick / TICKS * self._layer_hi)

    def _threshold_from_spin(self, value: float) -> None:
        self._apply_threshold(value)

    def _threshold_from_bar(self, value: float) -> None:
        self._apply_threshold(abs(value))
