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

import re

import torch
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.colormap import apply_colormap, build_lut
from fastfuncstuff.viewer.layers import AlphaMode, SignMode
from fastfuncstuff.viewer.ui import theme

#: The bar stands on end beside the layer form, so its width is fixed and its
#: height is whatever the form is -- the same arrangement AFNI's controller
#: has, and it gives the panel back the full row the horizontal bar took.
BAR_WIDTH = 22
BAR_MIN_HEIGHT = 150
TICKS = 1000


def colormap_icon(name: str, width: int = 64, height: int = 12) -> QtGui.QIcon:
    """A left-to-right swatch of one scale, for a picker to show beside its name."""
    lut = build_lut(name, width, device=torch.device("cpu"))
    rgb = (lut * 255).round().to(torch.uint8).unsqueeze(0).expand(height, -1, -1).contiguous()
    image = QtGui.QImage(
        rgb.numpy().data, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888
    ).copy()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


def thresholds_itself(layer) -> bool:
    """Whether the threshold reads the sub-brick being coloured."""
    return layer.threshold_brick == layer.volume_index


class ColorBar(QtWidgets.QWidget):
    """The gradient itself, standing on end, with threshold markers.

    The maximum is at the top. A colour scale read bottom-to-top is how every
    figure draws one, and it puts the threshold slider's travel beside the
    colours it cuts.
    """

    #: Shift+click: the value under the cursor, to become the threshold.
    clicked = QtCore.Signal(float)
    #: A plain click: run the scale the other way.
    reverse_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._lut_name = "gray"
        self._reverse = False
        self._lo, self._hi = 0.0, 1.0
        self._threshold = 0.0
        self._sign = SignMode.BOTH
        self._panes = 0
        self._alpha = AlphaMode.OFF
        self.setToolTip(
            "Click to reverse the colour scale.\nShift+click to set the threshold there."
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(BAR_WIDTH)
        self.setMinimumHeight(BAR_MIN_HEIGHT)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def configure(self, layer) -> None:
        self._lut_name = layer.colormap
        self._reverse = bool(layer.colormap_reversed)
        self._lo = float(layer.range_lo if layer.range_lo is not None else 0.0)
        self._hi = float(layer.range_hi if layer.range_hi is not None else 1.0)
        # Marked on the bar only when it is in the bar's units. A t of 3 drawn
        # on a beta scale that tops out at 0.4 is a line in the wrong place.
        self._threshold = float(layer.threshold) if thresholds_itself(layer) else 0.0
        self._sign = layer.sign_mode
        self._panes = int(layer.n_panes)
        self._alpha = layer.alpha_mode
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        h = max(self.height() - 2, 1)
        bar = QtCore.QRect(1, 1, BAR_WIDTH - 2, h)
        try:
            lut = build_lut(self._lut_name, 256, device=torch.device("cpu"), reverse=self._reverse)
        except KeyError:
            p.end()
            return

        values = torch.linspace(self._hi, self._lo, h)
        rgb = apply_colormap(
            values,
            lut=lut,
            lo=self._lo,
            hi=self._hi,
            sign_mode=self._sign,
            n_panes=self._panes,
        )
        for y in range(h):
            r, g, b = (float(c) for c in rgb[y])
            if self._threshold > 0 and abs(float(values[y])) < self._threshold:
                fade = 0.28 if self._alpha is AlphaMode.OFF else 0.55
                r, g, b = r * fade, g * fade, b * fade
            p.fillRect(
                QtCore.QRect(bar.x(), bar.y() + y, bar.width(), 1),
                QtGui.QColor.fromRgbF(r, g, b),
            )

        p.setPen(QtGui.QPen(QtGui.QColor(c.edge)))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(bar)

        span = (self._hi - self._lo) or 1.0
        if self._threshold > 0:
            p.setPen(QtGui.QPen(QtGui.QColor(c.warn)))
            for edge in (self._threshold, -self._threshold):
                if not (self._lo <= edge <= self._hi):
                    continue
                if self._sign is SignMode.POS and edge < 0:
                    continue
                if self._sign is SignMode.NEG and edge > 0:
                    continue
                y = bar.y() + int((self._hi - edge) / span * h)
                p.drawLine(bar.left(), y, bar.right(), y)
        p.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        """Click to reverse the scale; shift+click to threshold at the cursor.

        Reversing is the plain click because it is what the bar is for -- a
        picture of the scale -- and it replaces a ``_r`` twin of every map in
        the picker.
        """
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.reverse_requested.emit()
            return
        h = max(self.height() - 2, 1)
        frac = max(0.0, min(1.0, (event.position().y() - 1) / h))
        self.clicked.emit(self._hi - frac * (self._hi - self._lo))


class PValueSpin(QtWidgets.QDoubleSpinBox):
    """A p box that shows 3e-12 as 3e-12.

    Fixed decimals either cap the smallest p that can be typed or show a
    strong effect as 0.000000. Formatting with ``g`` keeps 0.001 as 0.001 and
    switches to scientific notation below it.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Qt rounds the stored value to `decimals`, so this has to reach the
        # smallest double for a 1e-300 to survive at all.
        self.setDecimals(323)
        self.setRange(1e-300, 0.999999)

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt)
        return f"{value:.3g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802 (Qt)
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int) -> object:
        # Anything a float could become while being typed: "1e", "1e-", "0.".
        if re.fullmatch(r"[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?", text.strip()):
            return (
                (QtGui.QValidator.State.Intermediate, text, pos)
                if not _parses(text)
                else (
                    QtGui.QValidator.State.Acceptable,
                    text,
                    pos,
                )
            )
        return (QtGui.QValidator.State.Invalid, text, pos)


def _parses(text: str) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return 0.0 < value < 1.0


class RangeBar(QtWidgets.QWidget):
    """Colour bar with its range beside it and the threshold below it.

    Max sits at the top of the bar and min at the bottom, with auto between
    them: those three are the colour scale. The threshold, its slider and its
    p go underneath, because the threshold is not a point on this scale -- it
    usually reads another sub-brick (colour by the beta, cut on its t), and
    placing it between max and min implied units it does not have.
    """

    range_changed = QtCore.Signal(float, float)
    mirror_changed = QtCore.Signal(bool)
    threshold_changed = QtCore.Signal(float)
    autorange_requested = QtCore.Signal()
    reverse_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_hi = 1.0
        self._thresholds_itself = True
        self._syncing = False
        #: ``(stat_code, dof)`` of the sub-brick the threshold reads, or None.
        self._stat: tuple[str, object] | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        outer.addLayout(h, 1)

        self.bar = ColorBar()
        self.bar.clicked.connect(self._threshold_from_bar)
        self.bar.reverse_requested.connect(self.reverse_requested)
        h.addWidget(self.bar)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, TICKS)
        self.slider.setContentsMargins(0, 0, 0, 0)
        self.slider.valueChanged.connect(self._threshold_from_slider)

        numbers = QtWidgets.QVBoxLayout()
        numbers.setContentsMargins(2, 0, 0, 0)
        numbers.setSpacing(3)
        self.min_spin = self._spin("lowest value shown")
        self.thr_spin = self._spin("threshold: values nearer zero than this are cut")
        self.max_spin = self._spin("highest value shown")
        self.thr_spin.setStyleSheet(f"color: {theme.palette().warn};")
        self.auto_button = QtWidgets.QPushButton("auto")
        self.auto_button.setToolTip(
            "Re-derive min and max from the sub-brick shown: symmetric about\n"
            "zero when it has both signs, from zero when it has one."
        )
        # Sized by its text, not capped: a fixed width under the stylesheet's
        # padding is what clipped the word to "au".
        self.auto_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.auto_button.clicked.connect(self.autorange_requested)

        numbers.addWidget(self._caption("max"))
        numbers.addWidget(self.max_spin)
        numbers.addStretch(1)
        numbers.addWidget(self.auto_button)
        self.mirror_check = QtWidgets.QCheckBox("mirror")
        self.mirror_check.setToolTip("Hold min at -max, so zero stays in the middle of the bar")
        self.mirror_check.clicked.connect(lambda on: self.mirror_changed.emit(bool(on)))
        numbers.addWidget(self.mirror_check)
        numbers.addStretch(1)
        numbers.addWidget(self._caption("min"))
        numbers.addWidget(self.min_spin)
        h.addLayout(numbers, 1)

        below = QtWidgets.QVBoxLayout()
        below.setContentsMargins(0, 0, 0, 0)
        below.setSpacing(3)
        thr_row = QtWidgets.QHBoxLayout()
        thr_row.setSpacing(4)
        #: Replaced by the window with the mode's name for its threshold.
        self.thr_caption = self._caption("thresh")
        thr_row.addWidget(self.thr_caption)
        thr_row.addWidget(self.thr_spin, 1)
        below.addLayout(thr_row)
        below.addWidget(self.slider)

        # A statistic's threshold is really a p; the stat value is the units it
        # happens to be stored in. The bucket states its own test and DoF in
        # BRICK_STATAUX, so nothing here has to be typed or remembered -- and
        # the p hides entirely on a map that is not a statistic, because a
        # p-value quoted for a beta is a number that means nothing.
        self.stat_label = QtWidgets.QLabel("")
        self.stat_label.setToolTip("The test this sub-brick carries, from BRICK_STATAUX")
        self.stat_label.setWordWrap(True)
        self.stat_label.setStyleSheet(f"font-size: {theme.FONT_SMALL}px;")
        self.p_spin = PValueSpin()
        self.p_spin.setValue(0.001)
        self.p_spin.setKeyboardTracking(False)
        self.p_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.p_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._stat_host = QtWidgets.QWidget()
        stat_col = QtWidgets.QVBoxLayout(self._stat_host)
        stat_col.setContentsMargins(0, 0, 0, 0)
        stat_col.setSpacing(2)
        stat_col.addWidget(self.stat_label)
        stat_col.addWidget(self.p_spin)
        below.addWidget(self._stat_host)
        outer.addLayout(below)

        self.min_spin.valueChanged.connect(self._emit_range)
        self.max_spin.valueChanged.connect(self._emit_range)
        self.thr_spin.valueChanged.connect(self._threshold_from_spin)
        self.p_spin.valueChanged.connect(self._threshold_from_p)

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("head")
        return label

    @staticmethod
    def _spin(tip: str) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1e9, 1e9)
        spin.setKeyboardTracking(False)  # commit on enter or focus-out
        spin.setToolTip(tip)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # A spin box sizes itself to the widest number its range allows, and
        # this range allows -1000000000.0000 -- sixteen characters, 157 pixels,
        # for a box that shows "8.24". Three of them made the range column
        # wider than the whole controller, so the panel scrolled sideways and
        # clipped the numbers off the other end. Ignored lets the layout size
        # it; the minimum keeps four significant figures legible.
        spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(64)
        return spin

    # -- incoming ------------------------------------------------------
    def restyle(self) -> None:
        """Re-read the palette after a theme switch."""
        self.thr_spin.setStyleSheet(f"color: {theme.palette().warn};")
        self.bar.update()

    def configure(self, layer, threshold_scale: float | None = None) -> None:
        """Show one layer. The single entry point, so nothing can go stale.

        ``threshold_scale`` is the largest magnitude in the sub-brick the
        threshold reads. Without it the slider spans the colour range, which
        is only right when the threshold reads the coloured sub-brick.
        """
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
            self.mirror_check.setChecked(bool(layer.range_mirror))
            # Faded rather than hidden: the number is still true, just not yours to set.
            self.min_spin.setEnabled(not layer.range_mirror)
            self._thresholds_itself = thresholds_itself(layer)
            self.thr_spin.setSingleStep(
                step if threshold_scale is None else max(threshold_scale / 100.0, 1e-4)
            )
            # The slider spans the sub-brick the threshold cuts on. On the
            # coloured one that is the larger half of the range, so a one-sided
            # map does not waste half its travel on values it never shows.
            if threshold_scale is not None:
                self._layer_hi = float(threshold_scale) or 1.0
            else:
                self._layer_hi = max(abs(hi), abs(lo)) or 1.0
            self.slider.setValue(int(round(min(layer.threshold / self._layer_hi, 1.0) * TICKS)))
            self._configure_stat(layer)
        finally:
            self._syncing = False

    def _configure_stat(self, layer) -> None:
        """Show the p row only where a p means something, and keep it honest."""
        from fastfuncstuff.stats.fdr import is_two_sided, stat_value_to_pvalue

        self._stat = layer.stat_spec()
        self._stat_host.setVisible(self._stat is not None)
        if self._stat is None:
            return
        code, dof = self._stat
        sided = "2-sided" if is_two_sided(code) else "1-sided"
        # Naming the test and its sidedness on the control, because "p < 0.001"
        # is two different thresholds depending on which one is meant.
        self.stat_label.setText(f"{layer.sub_brick(layer.threshold_brick)}  p {sided}")
        self.p_spin.setToolTip(
            f"Threshold as a {sided} p-value of the {code} this sub-brick carries.\n"
            "Sets the threshold; the threshold sets it back."
        )
        try:
            p = stat_value_to_pvalue(float(layer.threshold), code, dof)
        except (ValueError, OverflowError):
            return
        self.p_spin.setValue(min(max(p, self.p_spin.minimum()), self.p_spin.maximum()))

    # -- outgoing ------------------------------------------------------
    def _emit_range(self) -> None:
        if self._syncing:
            return
        lo, hi = self.min_spin.value(), self.max_spin.value()
        if self.mirror_check.isChecked():
            lo = -abs(hi)
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
        # A point on the colour scale is a threshold only in the scale's units.
        if self._thresholds_itself:
            self._apply_threshold(abs(value))

    def _threshold_from_p(self, p: float) -> None:
        """Typing a p sets the threshold, which is the only stored value.

        One source of truth on purpose: p is a *view* of the threshold, the way
        the colour bar is a view of the range. Storing both would let them
        disagree, and a threshold that disagrees with its own p-value is the
        worst kind of wrong -- it looks precise.
        """
        if self._syncing or self._stat is None:
            return
        from fastfuncstuff.stats.fdr import pvalue_to_stat

        code, dof = self._stat
        try:
            self._apply_threshold(pvalue_to_stat(float(p), code, dof))
        except (ValueError, OverflowError):
            return
