"""Splitting a regressor file by frequency, by looking at it.

A band split is only worth making where a column's spectrum says something --
a respiratory peak in a motion trace, a slow drift that wants to be told apart
from the fast jitter. So the cuts are placed *on* the spectrum: click it to cut,
drag a line to move the cut, right-click a line to take it away. Cycling
through the columns is the list's up and down keys, and the column in time,
pulled apart into the bands it would become, sits underneath -- a cut is
judged by what it does to the trace, not by where it lands on an axis.

What comes out is entries, one per band, in :mod:`fastfuncstuff.viewer.ortvec`'s
grammar. The dialog never touches the model; it only writes the list.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer import ortvec
from fastfuncstuff.viewer.modes.base import Trace
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.tracewindow import PlotView

MARGIN_LEFT = 48
MARGIN_RIGHT = 10
MARGIN_TOP = 18
MARGIN_BOTTOM = 22
#: How close, in pixels, a press has to be to a cut to grab it rather than add one.
GRAB_PX = 6
#: Power below this fraction of the peak is drawn at the floor. A DCT bin can be
#: numerically zero, and log10 of it would stretch the axis to nothing useful.
FLOOR = 1e-5


#: Bins a displayed spectrum is averaged over, as a fraction of how many it has.
SMOOTH_FRACTION = 1 / 80


def _smooth(power: np.ndarray) -> np.ndarray:
    """A running mean along frequency, for drawing only.

    A single run's periodogram is a chi-squared draw per bin, and drawn raw
    it is a hedge in which a respiratory peak is one spike among hundreds. The
    shares the dialog quotes are computed from the unsmoothed spectrum.
    """
    width = max(1, round(power.shape[0] * SMOOTH_FRACTION))
    if width == 1 or power.ndim != 2:
        return power
    kernel = np.ones(width)
    counts = np.convolve(np.ones(power.shape[0]), kernel, mode="same")
    return np.stack(
        [np.convolve(power[:, k], kernel, mode="same") / counts for k in range(power.shape[1])],
        axis=1,
    )


class SpectrumView(QtWidgets.QWidget):
    """Per-column power against frequency, with cuts you place by hand."""

    cutoffs_changed = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._freqs = np.zeros(0)
        self._power = np.zeros((0, 0))
        self._shown: list[int] = []
        self._current = 0
        self._cutoffs: list[float] = []
        self._log_x = False
        self._drag: int | None = None
        self._hover: float | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(420, 260)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    # -- state ---------------------------------------------------------
    def set_spectra(self, freqs: np.ndarray, power: np.ndarray) -> None:
        self._freqs, self._power = np.asarray(freqs), _smooth(np.asarray(power))
        self.update()

    def set_shown(self, columns: Sequence[int], current: int) -> None:
        self._shown, self._current = list(columns), int(current)
        self.update()

    def set_cutoffs(self, cutoffs: Sequence[float]) -> None:
        self._cutoffs = sorted(float(c) for c in cutoffs)
        self.update()

    def cutoffs(self) -> list[float]:
        return list(self._cutoffs)

    def set_log_x(self, on: bool) -> None:
        self._log_x = bool(on)
        self.update()

    # -- geometry ------------------------------------------------------
    def _plot(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            MARGIN_LEFT,
            MARGIN_TOP,
            max(self.width() - MARGIN_LEFT - MARGIN_RIGHT, 1),
            max(self.height() - MARGIN_TOP - MARGIN_BOTTOM, 1),
        )

    def _x_span(self) -> tuple[float, float]:
        if self._freqs.size < 2:
            return 0.0, 1.0
        lo, hi = float(self._freqs[0]), float(self._freqs[-1])
        return (np.log10(lo), np.log10(hi)) if self._log_x else (0.0, hi)

    def _to_x(self, f: float) -> float:
        plot, (a, b) = self._plot(), self._x_span()
        v = np.log10(max(f, 1e-12)) if self._log_x else f
        return plot.left() + (v - a) / (b - a) * plot.width()

    def _to_f(self, x: float) -> float:
        plot, (a, b) = self._plot(), self._x_span()
        v = a + (x - plot.left()) / plot.width() * (b - a)
        f = 10.0**v if self._log_x else v
        top = float(self._freqs[-1]) if self._freqs.size else 1.0
        return float(np.clip(f, float(self._freqs[0]) if self._freqs.size else 0.0, top))

    def _near(self, x: float) -> int | None:
        best = None
        for i, f in enumerate(self._cutoffs):
            d = abs(self._to_x(f) - x)
            if d <= GRAB_PX and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else best[1]

    # -- mouse ---------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._freqs.size < 2:
            return
        x = event.position().x()
        hit = self._near(x)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if hit is not None:
                del self._cutoffs[hit]
                self._emit()
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if hit is None:
            self._cutoffs.append(self._to_f(x))
            self._cutoffs.sort()
            hit = self._near(x)
            self._emit()
        self._drag = hit

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._freqs.size < 2:
            return
        x = event.position().x()
        self._hover = self._to_f(x) if self._plot().contains(event.position()) else None
        if self._drag is not None:
            self._cutoffs[self._drag] = self._to_f(x)
            moved = self._cutoffs[self._drag]
            self._cutoffs.sort()
            self._drag = self._cutoffs.index(moved)
            self._emit()
        shape = (
            QtCore.Qt.CursorShape.SizeHorCursor
            if self._drag is not None or self._near(x) is not None
            else QtCore.Qt.CursorShape.CrossCursor
        )
        self.setCursor(shape)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        self._drag = None

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt)
        self._hover = None
        self.update()

    def _emit(self) -> None:
        self.update()
        self.cutoffs_changed.emit(self.cutoffs())

    # -- painting ------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        plot = self._plot()
        if self._freqs.size < 2 or not self._shown:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "tick a column to see it")
            p.end()
            return

        # Bands alternate a faint fill, so the eye reads regions rather than
        # having to pair up lines.
        edges = [float(self._freqs[0]), *self._cutoffs, float(self._freqs[-1])]
        shade = QtGui.QColor(c.select)
        for i in range(0, len(edges) - 1, 2):
            left, right = self._to_x(edges[i]), self._to_x(edges[i + 1])
            p.fillRect(QtCore.QRectF(left, plot.top(), right - left, plot.height()), shade)
        p.setPen(QtGui.QPen(QtGui.QColor(c.edge)))
        p.drawRect(plot)

        shown = self._power[:, self._shown]
        peak = float(shown.max()) if shown.size else 1.0
        floor = np.log10(max(peak * FLOOR, 1e-30))
        top = np.log10(max(peak, 1e-30))
        if top <= floor:
            top = floor + 1.0

        def to_y(v: float) -> float:
            lv = np.log10(max(v, 10.0**floor))
            return plot.bottom() - (lv - floor) / (top - floor) * plot.height()

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        # The current column last, so it is drawn over the others.
        order = [k for k in self._shown if k != self._current]
        order += [self._current] if self._current in self._shown else []
        xs = [self._to_x(float(f)) for f in self._freqs]
        for k in order:
            path = QtGui.QPainterPath()
            for n, (x, v) in enumerate(zip(xs, self._power[:, k], strict=True)):
                pt = QtCore.QPointF(x, to_y(float(v)))
                path.lineTo(pt) if n else path.moveTo(pt)
            colour = QtGui.QColor.fromRgbF(*c.series[k % len(c.series)])
            current = k == self._current
            if not current:
                colour.setAlphaF(0.35)
            pen = QtGui.QPen(colour)
            pen.setWidthF(2.0 if current else 1.0)
            p.setPen(pen)
            p.drawPath(path)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        cut_pen = QtGui.QPen(QtGui.QColor(c.warn))
        cut_pen.setWidthF(1.5)
        for f in self._cutoffs:
            x = self._to_x(f)
            p.setPen(cut_pen)
            p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
            p.drawText(QtCore.QPointF(x + 3, plot.top() + 12), f"{f:.3g}")

        p.setPen(QtGui.QColor(c.faint))
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        p.drawText(QtCore.QRectF(0, plot.top() - 7, MARGIN_LEFT - 4, 14), right, "power")
        p.drawText(QtCore.QRectF(0, plot.bottom() - 7, MARGIN_LEFT - 4, 14), right, "log")
        below = plot.bottom() + 14
        lo_f = float(self._freqs[0]) if self._log_x else 0.0
        p.drawText(QtCore.QPointF(plot.left(), below), f"{lo_f:.3g}")
        tail = f"{float(self._freqs[-1]):.3g} Hz"
        width = QtGui.QFontMetrics(p.font()).horizontalAdvance(tail)
        p.drawText(QtCore.QPointF(plot.right() - width, below), tail)

        if self._hover is not None and self._hover > 0:
            # The period, not just the frequency: "a cycle every 20 s" is the
            # form breathing and slow drift are actually recognised in.
            p.setPen(QtGui.QColor(c.text))
            text = f"{self._hover:.4g} Hz  ·  {1.0 / self._hover:.3g} s"
            width = QtGui.QFontMetrics(p.font()).horizontalAdvance(text)
            p.drawText(QtCore.QPointF(plot.center().x() - width / 2, MARGIN_TOP - 5), text)
        p.end()


class BandSplitDialog(QtWidgets.QDialog):
    """Pick the columns and the cuts; get back the entries that replace one."""

    def __init__(
        self,
        entry: str,
        columns: np.ndarray,
        labels: Sequence[str],
        tr: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Split by frequency — {ortvec.describe_entry(entry)}")
        self._entry = entry
        self._columns = np.asarray(columns, dtype=np.float64)
        self._labels = list(labels)
        self._tr = float(tr)
        freqs, power = ortvec.column_spectra(self._columns, self._tr)

        v = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Click the spectrum to cut it, drag a line to move a cut, right-click one "
            "to remove it. Ticked columns are split, one new entry per band; the "
            "source entry is unticked, since its bands add back up to it."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        body = QtWidgets.QHBoxLayout()
        self.column_list = QtWidgets.QListWidget()
        self.column_list.setMaximumWidth(190)
        for k, label in enumerate(self._labels):
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            series = theme.palette().series
            swatch = QtGui.QColor.fromRgbF(*series[k % len(series)])
            item.setForeground(QtGui.QBrush(swatch))
            self.column_list.addItem(item)
        body.addWidget(self.column_list)

        plots = QtWidgets.QVBoxLayout()
        self.spectrum = SpectrumView()
        self.spectrum.set_spectra(freqs, power)
        plots.addWidget(self.spectrum, 3)
        self.preview = PlotView(legend_outside=True)
        plots.addWidget(self.preview, 2)
        body.addLayout(plots, 1)
        v.addLayout(body, 1)

        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("cuts (Hz)"))
        self.cut_edit = QtWidgets.QLineEdit()
        self.cut_edit.setPlaceholderText("e.g. 0.01, 0.1 -- or click the spectrum")
        form.addWidget(self.cut_edit, 1)
        self.log_x = QtWidgets.QCheckBox("log f")
        form.addWidget(self.log_x)
        v.addLayout(form)

        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        v.addWidget(self.readout)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

        self.column_list.currentRowChanged.connect(lambda _row: self._refresh())
        self.column_list.itemChanged.connect(lambda _item: self._refresh())
        self.spectrum.cutoffs_changed.connect(self._cuts_from_plot)
        self.cut_edit.editingFinished.connect(self._cuts_from_text)
        self.log_x.toggled.connect(self.spectrum.set_log_x)
        self.column_list.setCurrentRow(0)
        self.resize(1180, 760)
        self._refresh()

    # -- state ---------------------------------------------------------
    def checked(self) -> list[int]:
        return [
            k
            for k in range(self.column_list.count())
            if self.column_list.item(k).checkState() is QtCore.Qt.CheckState.Checked
        ]

    def cutoffs(self) -> list[float]:
        return self.spectrum.cutoffs()

    def set_cutoffs(self, cutoffs: Sequence[float]) -> None:
        nyquist = 0.5 / self._tr
        self.spectrum.set_cutoffs([c for c in cutoffs if 0 < c < nyquist])
        self._cuts_from_plot(self.spectrum.cutoffs())

    def entries(self) -> list[str]:
        """What the list should gain. Empty until there is a cut and a column."""
        if not self.cutoffs() or not self.checked():
            return []
        return ortvec.split_entries(
            self._entry,
            self.cutoffs(),
            n_columns=self._columns.shape[1],
            columns=self.checked(),
        )

    # -- reactions -----------------------------------------------------
    def _cuts_from_plot(self, cutoffs: list[float]) -> None:
        self.cut_edit.setText(", ".join(f"{c:.4g}" for c in cutoffs))
        self._refresh()

    def _cuts_from_text(self) -> None:
        values = []
        for word in self.cut_edit.text().replace(",", " ").split():
            try:
                values.append(float(word))
            except ValueError:
                continue
        self.set_cutoffs(values)

    def _refresh(self) -> None:
        current = max(self.column_list.currentRow(), 0)
        self.spectrum.set_shown(self.checked(), current)
        cuts = self.cutoffs()
        bands = [
            ortvec.Band(lo, cuts[i] if i < len(cuts) else None) for i, lo in enumerate([0.0, *cuts])
        ]
        column = self._columns[:, current : current + 1]
        label = self._labels[current] if self._labels else "column"
        traces = [Trace(label=label, key="col", values=column[:, 0], x_label="TR")]
        if cuts:
            traces += [
                Trace(
                    label=band.tag(),
                    key=f"band{i}",
                    values=ortvec.band_columns(column, self._tr, band)[:, 0],
                    x_label="TR",
                )
                for i, band in enumerate(bands)
            ]
        self.preview.set_traces(traces)

        freqs, power = ortvec.column_spectra(column, self._tr)
        if cuts:
            shares = []
            for band in bands:
                keep = freqs >= band.lo
                if band.hi is not None:
                    keep &= freqs < band.hi
                shares.append(f"{band.tag()} {100 * float(power[keep, 0].sum()):.0f}%")
            self.readout.setText(f"{label}, share of variance:  " + "  ·  ".join(shares))
        else:
            self.readout.setText(f"{label}: no cuts yet")
        n_new = len(self.entries())
        ok = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(n_new > 0)
        ok.setText(f"add {n_new} entries" if n_new else "OK")


def split_interactively(
    entry: str, tr: float, parent: QtWidgets.QWidget | None = None
) -> list[str] | None:
    """Open the splitter on one entry; the new entries, or ``None`` if cancelled."""
    if tr <= 0:
        QtWidgets.QMessageBox.information(
            parent, "Split by frequency", "The run has no TR, so there is no Hz axis to cut."
        )
        return None
    path, ops = ortvec.parse_entry(entry)
    try:
        columns, labels = ortvec.read_columns(path)
        columns, labels = ortvec.apply_ops(columns, labels, ops, tr)
    except (OSError, ValueError, IndexError) as exc:
        QtWidgets.QMessageBox.warning(parent, "Split by frequency", str(exc))
        return None
    if columns.shape[0] < 4 or columns.shape[1] == 0:
        QtWidgets.QMessageBox.warning(parent, "Split by frequency", f"{path}: nothing to split")
        return None
    dialog = BandSplitDialog(entry, columns, labels, tr, parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.entries()
