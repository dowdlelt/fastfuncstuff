"""The colour bar: what the numbers under the cursor actually look like.

Drawn from the same LUT, sign mode and pane count the slices use, so it is a
legend rather than a decoration -- a discrete pane count bands the bar exactly
as it bands the map, and a one-sided sign mode shows only the half being drawn.

The sub-threshold region is dimmed rather than hidden, matching what the alpha
ramp does to the image: the bar should show that those values are still being
drawn faintly, not imply they are gone.
"""

from __future__ import annotations

import torch
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.colormap import apply_colormap, build_lut
from fastfuncstuff.viewer.layers import AlphaMode, SignMode

BAR_HEIGHT = 22
TICK_ROOM = 14


class ColorBar(QtWidgets.QWidget):
    """A live legend for one layer's colour mapping."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._lut_name = "gray"
        self._lo = 0.0
        self._hi = 1.0
        self._threshold = 0.0
        self._sign = SignMode.BOTH
        self._panes = 0
        self._alpha = AlphaMode.OFF
        self.setMinimumHeight(BAR_HEIGHT + TICK_ROOM)
        self.setMaximumHeight(BAR_HEIGHT + TICK_ROOM)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def set_layer(self, layer) -> None:
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
        bar = QtCore.QRect(1, 1, w, BAR_HEIGHT)

        try:
            lut = build_lut(self._lut_name, 256, device=torch.device("cpu"))
        except KeyError:
            p.end()
            return

        # Sample the same mapping the slices use, one column per pixel.
        values = torch.linspace(self._lo, self._hi, w)
        rgb = apply_colormap(
            values,
            lut=lut,
            lo=self._lo,
            hi=self._hi,
            sign_mode=self._sign,
            n_panes=self._panes,
        )
        span = (self._hi - self._lo) or 1.0
        for x in range(w):
            value = float(values[x])
            r, g, b = (float(c) for c in rgb[x])
            colour = QtGui.QColor.fromRgbF(r, g, b)
            if self._threshold > 0 and abs(value) < self._threshold:
                # Dimmed, not blank: under an alpha ramp these values are still
                # drawn, just faintly, and the legend should say so.
                fade = 0.28 if self._alpha is AlphaMode.OFF else 0.55
                colour = QtGui.QColor.fromRgbF(r * fade, g * fade, b * fade)
            p.fillRect(QtCore.QRect(bar.x() + x, bar.y(), 1, bar.height()), colour)

        p.setPen(QtGui.QPen(QtGui.QColor(30, 39, 44)))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(bar)

        # Threshold markers, one per side that is actually shown.
        if self._threshold > 0:
            pen = QtGui.QPen(QtGui.QColor(217, 164, 65))
            pen.setWidth(1)
            p.setPen(pen)
            for edge in (self._threshold, -self._threshold):
                if not (self._lo <= edge <= self._hi):
                    continue
                if self._sign is SignMode.POS and edge < 0:
                    continue
                if self._sign is SignMode.NEG and edge > 0:
                    continue
                x = bar.x() + int((edge - self._lo) / span * w)
                p.drawLine(x, bar.top(), x, bar.bottom())

        p.setPen(QtGui.QColor(107, 125, 132))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        y = bar.bottom() + 11
        p.drawText(QtCore.QPoint(1, y), f"{self._lo:.4g}")
        mid = f"{self._threshold:.3g}" if self._threshold > 0 else ""
        if mid:
            p.setPen(QtGui.QColor(217, 164, 65))
            p.drawText(
                QtCore.QRect(1, bar.bottom(), w, TICK_ROOM),
                QtCore.Qt.AlignmentFlag.AlignHCenter,
                mid,
            )
            p.setPen(QtGui.QColor(107, 125, 132))
        p.drawText(
            QtCore.QRect(1, bar.bottom(), w, TICK_ROOM),
            QtCore.Qt.AlignmentFlag.AlignRight,
            f"{self._hi:.4g}",
        )
        p.end()
