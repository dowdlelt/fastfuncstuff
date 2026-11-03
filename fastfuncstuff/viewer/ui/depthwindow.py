"""Depth profiles of an overlay, over a region of cortex.

Click a spot (the crosshair), choose how the region grows from it -- a
geodesic disc of some radius, the parcel it is in, or the ROI-layer region it
is in -- and the window shows the overlay against cortical depth across that
region: the mean, its spread, and every vertex faintly behind them. Depth is
equivolume by default and runs a little past white and pial, so the edges of
the ribbon are visible in the profile rather than assumed.

A time series gives depth x time as well: the region's mean at every depth
and volume, with the time cursor -- the depth timecourses a laminar analysis
starts from -- and EXPORT writes them as a table, ``(n_time, K)``.

PIN keeps the profiled dataset on the plot as a coloured trace that follows
the region and the time cursor, so selecting another dataset lays the two side
by side wherever you click -- what a processing step changed, spot by spot.
The axes stretch to hold every trace.

The region is published to the surface store, so a surface window shows
which cortex the profile came from.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.surface.profiles import sample_depths
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.derive import drift_basis
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.gridgraph import DETREND_LABELS
from fastfuncstuff.viewer.ui.shortcuts import Binding as Key
from fastfuncstuff.viewer.ui.shortcuts import ShortcutHelp, jump_back_key, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import SetDepthView, SetIndex, SetSurfaceEquivolume, SetViewDetrend

SOURCES = ("disc", "annot", "layer")
#: How far past white (below 0) and pial (above 1) the profile runs.
MARGIN = 0.35


@dataclass
class DepthResult:
    fractions: np.ndarray  # (K,)
    profiles: np.ndarray  # (V, K) at the current volume
    timecourses: np.ndarray | None  # (T, K) region mean per depth and volume, or None
    n_vertices: int
    region: str
    layer_name: str
    time_index: int
    #: Every vertex, depth and volume, kept so stepping through time slices it
    #: instead of re-sampling the whole series (dropped above ~512 MB).
    series: np.ndarray | None = None


@dataclass
class PinnedLayer:
    """A dataset kept on the plot, re-profiled over whatever region is live."""

    key: str
    #: A bucket's sub-brick as pinned; ``None`` follows the time cursor.
    brick: int | None
    result: DepthResult | None = None
    built: tuple | None = None


@dataclass(frozen=True)
class PinnedProfile:
    """One pinned dataset's profile over the current region, as drawn."""

    label: str
    fractions: np.ndarray  # (K,)
    mean: np.ndarray  # (K,)
    sd: np.ndarray  # (K,)
    n_vertices: int

    @classmethod
    def of(cls, result: DepthResult, label: str) -> PinnedProfile:
        with np.errstate(all="ignore"):
            mean = np.nanmean(result.profiles, axis=0)
            sd = np.nanstd(result.profiles, axis=0)
        return cls(label, result.fractions.copy(), mean, sd, result.n_vertices)


def _volume_of(session, layer, brick: int | None = None) -> int:
    if layer.time_linked:
        return int(session.state.time_index)
    return int(layer.volume_index if brick is None else brick)


def depth_profiles(
    session,
    vertices: dict[str, np.ndarray],
    layer,
    fractions: np.ndarray,
    equivolume: bool,
    brick: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(V, K)`` profiles at the current volume, ``(T, K)`` region-mean
    timecourses, and the full ``(V, K, T)`` samples behind them.

    Timecourses only for a time-linked layer whose full series is resident;
    otherwise ``None`` (the series is still loading, or the layer is a stat
    map whose sub-bricks are not time). ``brick`` overrides a bucket's shown
    sub-brick, for a pin that keeps the one it was pinned on.
    """
    idx = _volume_of(session, layer, brick)
    res = session.store.get(layer.key)
    four_d = layer.time_linked and res.array is not None and res.array.ndim == 4
    volume = res.array if four_d else session.volume(layer.key, idx)
    parts = []
    for hemi, ids in vertices.items():
        h = session.surfaces.hemis[hemi]
        w, p = h.states["white"][ids], h.states["pial"][ids]
        areas = None
        if equivolume:
            areas = session.surfaces.vertex_areas_at(hemi, ids)
        parts.append(
            sample_depths(
                w,
                p,
                volume,
                layer.affine,
                fractions,
                white_area=None if areas is None else areas[0],
                pial_area=None if areas is None else areas[1],
                device=session.display_device,
            )
        )
    samples = np.concatenate(parts, axis=0)
    if four_d:
        return samples[:, :, idx], np.nanmean(samples, axis=0).T, samples
    return samples, None, None


class DepthPlot(QtWidgets.QWidget):
    """Value against depth: every vertex faintly, the mean and +-1 SD over them."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.result: DepthResult | None = None
        self.pins: list[PinnedProfile] = []
        self.message = "click cortex to profile it"
        self.setMinimumSize(200, 160)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        r = self.result
        live = r is not None and r.n_vertices > 0
        if not live and not self.pins:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        plot = QtCore.QRectF(48, 22, max(self.width() - 60, 10), max(self.height() - 48, 10))
        # Scale to the mean +- 2 SD rather than to every vertex's extremes, so
        # one wild vertex does not flatten the profile everyone came to see --
        # and over every pinned trace too, so none of them runs off the plot.
        spans = [(q.fractions, q.mean - 2 * q.sd, q.mean + 2 * q.sd) for q in self.pins]
        if live:
            assert r is not None
            prof = r.profiles
            # An all-NaN depth (every vertex in the disc unfit, or off the data)
            # is ordinary, and the empty-slice result is handled by the finite
            # check below. np.errstate does not cover these: nanmean and nanstd
            # raise RuntimeWarning through the warnings module, not the floating
            # point error state, so they printed to the terminal on every repaint.
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(prof, axis=0)
                sd = np.nanstd(prof, axis=0)
            spans.append((r.fractions, mean - 2 * sd, mean + 2 * sd))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = float(np.nanmin([np.nanmin(s[1]) for s in spans]))
            hi = float(np.nanmax([np.nanmax(s[2]) for s in spans]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = (lo - 1, lo + 1) if np.isfinite(lo) else (0.0, 1.0)
        x0 = min(float(s[0][0]) for s in spans)
        x1 = max(float(s[0][-1]) for s in spans)

        def pt(xv: float, yv: float) -> QtCore.QPointF:
            return QtCore.QPointF(
                plot.left() + (xv - x0) / (x1 - x0) * plot.width(),
                plot.bottom() - (yv - lo) / (hi - lo) * plot.height(),
            )

        p.setPen(QtGui.QPen(QtGui.QColor(c.edge)))
        p.drawRect(plot)
        # The ribbon: white at 0, pial at 1.
        guide = QtGui.QPen(QtGui.QColor.fromRgbF(*c.label))
        guide.setStyle(QtCore.Qt.PenStyle.DashLine)
        p.setPen(guide)
        for edge, name in ((0.0, "white"), (1.0, "pial")):
            a, b = pt(edge, lo), pt(edge, hi)
            p.drawLine(a, b)
            p.drawText(QtCore.QPointF(a.x() + 3, plot.top() + 12), name)
        if lo < 0 < hi:
            p.drawLine(pt(x0, 0.0), pt(x1, 0.0))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for n, pin in enumerate(self.pins):
            self._draw_trace(p, pt, pin.fractions, pin.mean, pin.sd, _pin_color(n), 0.12)
        if live:
            assert r is not None
            self._draw_live(p, pt, r.fractions, prof, mean, sd)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        p.setPen(QtGui.QColor(c.faint))
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        p.drawText(QtCore.QRectF(0, plot.top() - 7, 44, 14), right, f"{hi:.3g}")
        p.drawText(QtCore.QRectF(0, plot.bottom() - 7, 44, 14), right, f"{lo:.3g}")
        p.drawText(
            QtCore.QPointF(plot.left(), plot.bottom() + 16),
            "WM  <-  depth (white 0, pial 1)  ->  CSF",
        )
        p.setPen(QtGui.QColor(c.text))
        title = (
            f"{r.layer_name} · {r.region} · {r.n_vertices} vertices · mean +- SD"
            if live and r is not None
            else "pinned profiles"
        )
        p.drawText(QtCore.QPointF(plot.left(), 15), title)
        metrics = p.fontMetrics()
        for n, pin in enumerate(self.pins):
            p.setPen(_pin_color(n))
            row = QtCore.QRectF(
                plot.left() + 6,
                plot.top() + 16 + n * metrics.height(),
                plot.width() - 12,
                metrics.height(),
            )
            text = metrics.elidedText(
                f"{n + 1}  {pin.label}", QtCore.Qt.TextElideMode.ElideMiddle, int(row.width())
            )
            p.drawText(row, QtCore.Qt.AlignmentFlag.AlignRight, text)
        p.end()

    @staticmethod
    def _draw_trace(p, pt, x, mean, sd, colour: QtGui.QColor, band_alpha: float) -> None:
        band = QtGui.QPolygonF(
            [pt(x[k], mean[k] + sd[k]) for k in range(x.size)]
            + [pt(x[k], mean[k] - sd[k]) for k in reversed(range(x.size))]
        )
        fill = QtGui.QColor(colour)
        fill.setAlphaF(band_alpha)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPolygon(band)
        pen = QtGui.QPen(colour)
        pen.setWidthF(2.2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        path = QtGui.QPainterPath(pt(x[0], mean[0]))
        for k in range(1, x.size):
            path.lineTo(pt(x[k], mean[k]))
        p.drawPath(path)

    def _draw_live(self, p, pt, x, prof, mean, sd) -> None:
        accent = QtGui.QColor(theme.palette().accent)
        faint = QtGui.QColor(accent)
        faint.setAlphaF(0.08 if prof.shape[0] > 50 else 0.2)
        p.setPen(QtGui.QPen(faint))
        rng = np.random.default_rng(0)
        shown = (
            prof if prof.shape[0] <= 300 else prof[rng.choice(prof.shape[0], 300, replace=False)]
        )
        for row in shown:
            path = QtGui.QPainterPath(pt(x[0], row[0]))
            for k in range(1, x.size):
                path.lineTo(pt(x[k], row[k]))
            p.drawPath(path)
        self._draw_trace(p, pt, x, mean, sd, accent, 0.22)


def _pin_color(n: int) -> QtGui.QColor:
    # The first series colour is the accent, which the live trace already wears.
    series = theme.palette().series[1:]
    return QtGui.QColor.fromRgbF(*series[n % len(series)][:3])


def depth_time_view(data: np.ndarray, order: int) -> tuple[np.ndarray, bool]:
    """``(T, K)`` depth timecourses as percent signal change, Legendre drift removed.

    Each depth is scaled by its own mean, so a superficial depth's larger
    baseline does not make its response look larger than a deep one's, and
    the colour scale compares like with like. ``order`` -1 and 0 coincide:
    PSC removes the mean either way. Returns ``(view, is_psc)``: data with a
    depth whose mean is not positive (a residual, phase, a stat series) has
    no percent to speak of, so it comes back detrended in its own units.
    """
    d = np.asarray(data, dtype=np.float64)
    mean = np.nanmean(d, axis=0)
    if order >= 0 and d.shape[0] > 1:
        filled = np.where(np.isfinite(d), d, mean)
        basis = drift_basis(d.shape[0], int(order))
        d = d - basis @ (basis.T @ filled)
    else:
        d = d - mean
    if np.all(np.isfinite(mean) & (mean > 0)):
        return 100.0 * d / mean, True
    return d, False


class DepthTimePlot(QtWidgets.QWidget):
    """Region mean at every depth (rows, pial on top) and volume (columns)."""

    clicked = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: np.ndarray | None = None  # (T, K)
        self.fractions: np.ndarray | None = None
        self.cursor = 0
        #: Legendre order projected out before drawing; display only.
        self.detrend = Viewport.detrend
        self.setMinimumHeight(90)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        import torch

        from fastfuncstuff.viewer.colormap import build_lut

        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self.data is None or self.data.size == 0:
            p.end()
            return
        view, is_psc = depth_time_view(self.data, self.detrend)
        d = view.T[::-1]  # (K, T), pial row first
        top = float(np.nanpercentile(np.abs(d), 98)) or 1.0
        unit = np.clip(0.5 + 0.5 * d / top, 0, 1)
        lut = (
            np.clip(build_lut("RdBu", 256, device=torch.device("cpu")).numpy(), 0, 1) * 255
        ).astype(np.uint8)
        rgb = np.ascontiguousarray(
            lut[np.round((1 - np.nan_to_num(unit)) * 255).astype(int)][..., :3]
        )
        img = QtGui.QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            3 * rgb.shape[1],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        area = QtCore.QRect(48, 4, max(self.width() - 60, 10), max(self.height() - 22, 10))
        p.drawImage(area, img)
        p.setPen(QtGui.QColor(c.faint))
        p.drawText(QtCore.QPointF(2, area.top() + 10), "pial")
        p.drawText(QtCore.QPointF(2, area.bottom()), "white")
        trend = dict(DETREND_LABELS).get(self.detrend, f"poly {self.detrend}")
        units = (
            f"% signal change, ±{top:.2g}" if is_psc else f"each depth minus its mean, ±{top:.3g}"
        )
        removed = "" if self.detrend <= 0 else f", {trend} detrend"
        p.drawText(QtCore.QPointF(area.left(), area.bottom() + 14), f"time  ({units}{removed})")
        T = self.data.shape[0]
        x = area.left() + (self.cursor + 0.5) / max(T, 1) * area.width()
        p.setPen(QtGui.QPen(QtGui.QColor(c.warn)))
        p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
        p.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self.data is None:
            return
        left, width = 48, max(self.width() - 60, 10)
        t = int((event.position().x() - left) / width * self.data.shape[0])
        if 0 <= t < self.data.shape[0]:
            self.clicked.emit(t)


class DepthWindow(QtWidgets.QWidget):
    """A depth-profile viewport as a top-level window."""

    closed = QtCore.Signal(str)
    #: The region changed; surface windows show it.
    roi_changed = QtCore.Signal()

    def __init__(
        self,
        vid: str,
        session,
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.vid = vid
        self.session = session
        self._dispatch = dispatch
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setStyleSheet(theme.stylesheet())
        self.follow = True
        self.result: DepthResult | None = None
        self._built: tuple | None = None
        self._vertices: dict[str, np.ndarray] = {}
        self._pinned: list[PinnedLayer] = []
        #: The live trace's layer and volume, which PIN keeps.
        self._live: tuple[str, int | None] | None = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        self.source_box = QtWidgets.QComboBox()
        self.source_box.addItems(["disc", "parcel", "ROI layer"])
        self.source_box.setToolTip("How the region grows from the clicked spot")
        self.source_box.activated.connect(self._changed)
        bar.addWidget(self.source_box)
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 50.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setSuffix(" mm")
        self.radius_spin.setToolTip("Disc radius, along the surface")
        self.radius_spin.editingFinished.connect(self._changed)
        bar.addWidget(self.radius_spin)
        self.layer_box = QtWidgets.QComboBox()
        self.layer_box.setToolTip("Overlay to profile (auto: the selected, else the top one)")
        self.layer_box.activated.connect(self._changed)
        bar.addWidget(self.layer_box)
        self.roi_box = QtWidgets.QComboBox()
        self.roi_box.setToolTip("ROI layer the region comes from (source: ROI layer)")
        self.roi_box.activated.connect(self._changed)
        bar.addWidget(self.roi_box)
        self.detrend_box = QtWidgets.QComboBox()
        for order, label in DETREND_LABELS:
            self.detrend_box.addItem(label, userData=order)
        self.detrend_box.setToolTip(
            "Legendre drift projected out of the depth x time panel (d steps).\n"
            "Display only: EXPORT writes the region's raw timecourses."
        )
        self.detrend_box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.detrend_box.activated.connect(
            lambda _: self._set_detrend(int(self.detrend_box.currentData()))
        )
        bar.addWidget(self.detrend_box)
        bar.addStretch(1)
        pin = QtWidgets.QPushButton("PIN")
        pin.setToolTip(
            "Keep this dataset on the plot, following the region, to compare"
            " another against (p; shift+p clears)"
        )
        pin.clicked.connect(self.pin)
        bar.addWidget(pin)
        self.unpin_button = QtWidgets.QPushButton("UNPIN")
        self.unpin_button.setToolTip("Clear every pinned dataset (shift+p)")
        self.unpin_button.clicked.connect(self.clear_pins)
        self.unpin_button.setEnabled(False)
        bar.addWidget(self.unpin_button)
        export = QtWidgets.QPushButton("EXPORT")
        export.setToolTip("Write the profile (and depth timecourses) as a table")
        export.clicked.connect(self._export_dialog)
        bar.addWidget(export)
        v.addLayout(bar)
        self.plot = DepthPlot(self)
        v.addWidget(self.plot, 3)
        self.time_plot = DepthTimePlot(self)
        self.time_plot.clicked.connect(lambda t: self._dispatch(SetIndex(int(t))))
        v.addWidget(self.time_plot, 1)
        self.resize(560, 520)

        self.help = ShortcutHelp(self, f"depth · {vid}")
        self.help.apply(
            [
                jump_back_key(self._dispatch),
                Key("e", "equivolume / equidistant depth", self._toggle_equivolume, group="depth"),
                Key("(", "smaller disc", lambda: self._radius_by(1 / 1.25), group="depth"),
                Key(")", "larger disc", lambda: self._radius_by(1.25), group="depth"),
                Key("f", "follow the crosshair / pin the region", self._toggle_follow, group="depth"),
                Key("p", "pin this dataset to compare against", self.pin, group="depth"),
                Key("shift+p", "clear the pinned datasets", self.clear_pins, group="depth"),
                Key("d", "next detrend order (depth x time)", self._cycle_detrend, group="depth"),
                Key("click (heatmap)", "go to that volume", None, group="depth"),
                Key("h", "this list", self.help.toggle, group="window"),
                Key("ctrl+w", "close this window", self.close, group="window"),
            ]
        )  # fmt: skip
        keep_keys_for_shortcuts(self)

    # -- plumbing ------------------------------------------------------------
    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        self.refresh(Aspect.ALL)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.plot.update()
        self.time_plot.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.session.surfaces.publish_depth_roi({})
        self.roi_changed.emit()
        self.closed.emit(self.vid)
        super().closeEvent(event)

    def _changed(self, *_args) -> None:
        # The radius box's editingFinished fires on losing focus -- which closing
        # the window does, after its viewport has gone.
        if self._viewport() is None:
            return
        source = SOURCES[self.source_box.currentIndex()]
        layer = self.layer_box.currentData() or ""
        roi = self.roi_box.currentData() or ""
        self._dispatch(SetDepthView(self.vid, source, float(self.radius_spin.value()), layer, roi))

    def _radius_by(self, factor: float) -> None:
        self.radius_spin.setValue(float(np.clip(self.radius_spin.value() * factor, 0.5, 50.0)))
        self._changed()

    def _toggle_equivolume(self) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetSurfaceEquivolume(self.vid, not vp.equivolume))

    def _set_detrend(self, order: int) -> None:
        if self._viewport() is not None:
            self._dispatch(SetViewDetrend(self.vid, int(order)))

    def _cycle_detrend(self) -> None:
        vp = self._viewport()
        if vp is None:
            return
        orders = [k for k, _ in DETREND_LABELS]
        at = orders.index(vp.detrend) if vp.detrend in orders else 0
        self._set_detrend(orders[(at + 1) % len(orders)])

    def pin(self) -> None:
        if self._live is None or self.result is None or self.result.n_vertices == 0:
            return
        key, brick = self._live
        if any((q.key, q.brick) == (key, brick) for q in self._pinned):
            return
        # The live result is this dataset over this region: the pin starts from it.
        self._pinned.append(PinnedLayer(key, brick, self.result, self._built))
        self.unpin_button.setEnabled(True)
        self.refresh(Aspect.ALL)

    def clear_pins(self) -> None:
        self._pinned.clear()
        self.plot.pins = []
        self.unpin_button.setEnabled(False)
        self.plot.update()

    def _toggle_follow(self) -> None:
        self.follow = not self.follow
        if self.follow:
            self.refresh(Aspect.CROSSHAIR)

    def _sync_controls(self, vp: Viewport) -> None:
        st = self.session.state
        layers = list(st.layers)[1:]
        for box, items, current in (
            (
                self.layer_box,
                [("auto", "")] + [(ly.name, ly.key) for ly in layers if not ly.roi],
                vp.depth_layer,
            ),
            (self.roi_box, [(ly.name, ly.key) for ly in layers if ly.roi], vp.depth_roi_layer),
        ):
            box.blockSignals(True)
            box.clear()
            for name, key in items:
                box.addItem(name, key)
            k = box.findData(current)
            box.setCurrentIndex(max(k, 0))
            box.blockSignals(False)
        self.source_box.blockSignals(True)
        self.source_box.setCurrentIndex(SOURCES.index(vp.depth_source))
        self.source_box.blockSignals(False)
        self.radius_spin.blockSignals(True)
        self.radius_spin.setValue(vp.radius)
        self.radius_spin.blockSignals(False)
        self.radius_spin.setEnabled(vp.depth_source == "disc")
        self.detrend_box.blockSignals(True)
        self.detrend_box.setCurrentIndex(max(self.detrend_box.findData(int(vp.detrend)), 0))
        self.detrend_box.blockSignals(False)
        self.roi_box.setEnabled(vp.depth_source == "layer")

    def _is_surface_layer(self, layer) -> bool:
        """A per-vertex result, which has no depth axis left to profile.

        Surface data is already a ribbon average (or one depth of it), so asking
        for its profile samples the same number K times. Worse, it is not free:
        a surface layer's volumes are painted on demand, so every refresh
        allocates the display grid and scatters the whole ribbon into it just to
        read a disc of vertices back out.
        """
        return layer is not None and layer.key in self.session.surface_layers

    def _profiled_layer(self, vp: Viewport):
        st = self.session.state
        if vp.depth_layer:
            # Named explicitly: returned even if it cannot be profiled, so
            # refresh() can say why instead of quietly profiling something else.
            return st.layers.find(vp.depth_layer)
        sel = st.layers.find(st.selected) if st.selected else None
        base = st.layers.base
        if (
            sel is not None
            and not sel.roi
            and not self._is_surface_layer(sel)
            and (base is None or sel.key != base.key)
        ):
            return sel
        for layer in reversed(list(st.layers)[1:]):
            if layer.visible and not layer.roi and not self._is_surface_layer(layer):
                return layer
        return None if self._is_surface_layer(base) else base

    # -- building --------------------------------------------------------------
    def refresh(self, dirty: Aspect) -> None:
        vp = self._viewport()
        surfaces = self.session.surfaces
        st = self.session.state
        if vp is None:
            return
        self._sync_controls(vp)
        layer = self._profiled_layer(vp)
        if not surfaces.hemis or layer is None:
            self.plot.message = "load surfaces and a volume overlay"
            self.plot.result = None
            self.plot.update()
            return
        if self._is_surface_layer(layer):
            self.plot.message = (
                f"{layer.name} is a per-vertex result: no depth axis to profile.\n"
                "Profile the volume it was sampled from, or write the depths\n"
                "separately (ffs_nwarp -surf_depth_combine none)."
            )
            self.plot.result = None
            self.result = None
            self.time_plot.data = None
            self.time_plot.setVisible(False)
            self.detrend_box.setVisible(False)
            self.plot.update()
            return
        mm = st.crosshair_mm
        region_key = (
            None if mm is None else tuple(np.round(mm, 3)),
            vp.depth_source,
            vp.radius,
            vp.depth_roi_layer,
            self._roi_generation(vp.depth_roi_layer),
            st.surface_annot,
            tuple(sorted(surfaces.version.items())),
        )
        if self.follow and mm is not None and region_key != getattr(self, "_region_key", None):
            labels = None
            if vp.depth_source == "layer" and vp.depth_roi_layer:
                roi = st.layers.find(vp.depth_roi_layer)
                if roi is not None:
                    labels = (
                        np.asarray(self.session.volume(roi.key, roi.volume_index)).astype(np.int64),
                        roi.affine,
                    )
            self._vertices = surfaces.depth_roi(
                mm, vp.depth_source, radius=vp.radius, annot=st.surface_annot, labels=labels
            )
            self._region_key = region_key
            surfaces.publish_depth_roi(self._vertices)
            self.roi_changed.emit()
        idx = _volume_of(self.session, layer)
        self._live = (layer.key, None if layer.time_linked else idx)
        n = sum(ids.size for ids in self._vertices.values())
        pins: list[PinnedProfile] = []
        if n == 0:
            self.result = None
            self.plot.message = "no cortex here: click within 3 mm of the ribbon"
            self.plot.result = None
            self.time_plot.data = None
        else:
            self.result, self._built = self._profile(vp, layer, None, self.result, self._built)
            for q in self._pinned:
                pinned = st.layers.find(q.key)
                if pinned is None:  # unloaded since: nothing left to profile
                    continue
                q.result, q.built = self._profile(vp, pinned, q.brick, q.result, q.built)
                assert q.result is not None
                pins.append(PinnedProfile.of(q.result, q.result.layer_name))
        self.plot.pins = pins
        self.plot.result = self.result
        self.time_plot.data = None if self.result is None else self.result.timecourses
        self.time_plot.cursor = idx
        self.time_plot.detrend = int(vp.detrend)
        self.time_plot.setVisible(self.time_plot.data is not None)
        self.detrend_box.setVisible(self.time_plot.data is not None)
        self.plot.update()
        self.time_plot.update()

    def _profile(
        self,
        vp: Viewport,
        layer,
        brick: int | None,
        result: DepthResult | None,
        built: tuple | None,
    ) -> tuple[DepthResult | None, tuple | None]:
        """One layer's profile over the current region, and the key it was built for.

        Re-samples only when the region, layer or depth spacing moved; a series
        whose cursor alone moved is re-sliced instead.
        """
        idx = _volume_of(self.session, layer, brick)
        key = (
            tuple((h, ids.tobytes()) for h, ids in self._vertices.items()),
            layer.key,
            idx if not layer.time_linked else "series",
            vp.equivolume,
            tuple(sorted(self.session.surfaces.version.items())),
        )
        if key != built:
            fractions = np.linspace(-MARGIN, 1 + MARGIN, int(vp.depth_bins))
            profiles, courses, samples = depth_profiles(
                self.session, self._vertices, layer, fractions, vp.equivolume, brick
            )
            keep = samples if samples is not None and samples.nbytes < 512 * 2**20 else None
            n = sum(ids.size for ids in self._vertices.values())
            return (
                DepthResult(
                    fractions,
                    profiles,
                    courses,
                    n,
                    self._region_name(vp),
                    _shown(layer, idx),
                    idx,
                    keep,
                ),
                key,
            )
        if result is not None and result.timecourses is not None:
            # Same region and series: only the current volume moved.
            result.time_index = idx
            if result.series is not None:
                result.profiles = result.series[:, :, idx]
            else:
                result.profiles = depth_profiles(
                    self.session, self._vertices, layer, result.fractions, vp.equivolume
                )[0]
        return result, built

    def _roi_generation(self, key: str) -> int | None:
        """The ROI layer's data version: a painted layer changes in place."""
        try:
            return self.session.store.get(key).generation if key else None
        except KeyError:
            return None

    def _region_name(self, vp: Viewport) -> str:
        st = self.session.state
        if vp.depth_source == "disc":
            return f"{vp.radius:g} mm disc"
        lines = self.session.surfaces.region_lines(st.crosshair_mm, st.surface_annot, "")
        if vp.depth_source == "annot" and lines:
            return lines[0].split("  (")[0]
        return "ROI region"

    def _export_dialog(self) -> None:
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export depth profile", "depth_profile.tsv", "Tables (*.tsv *.txt)"
        )
        if path:
            export_table(self.result, Path(path), self.plot.pins)


def _shown(layer, idx: int) -> str:
    """The layer, and for a bucket the sub-brick -- a coef and its t share a name."""
    if layer.time_linked or layer.n_volumes <= 1:
        return layer.name
    brick = layer.labels[idx] if 0 <= idx < len(layer.labels) and layer.labels[idx] else f"#{idx}"
    return f"{layer.name} [{brick}]"


def export_table(result: DepthResult, path: Path, pins: Sequence[PinnedProfile] = ()) -> None:
    """Write the region's depth profile; with a series, its ``(T, K)`` depth timecourses.

    Columns are depths (fractions, white 0 to pial 1, the margins outside).
    Without a series: rows ``mean``, ``sd``, ``n``. With one: a row per volume
    -- the shape laminar models read -- and the profile in a ``.profile.tsv``
    beside it. Each pinned profile on the same depths adds ``pinN mean`` /
    ``pinN sd`` / ``pinN n`` rows to the profile, and its label to a comment
    line at the top.
    """
    header = "\t".join(f"{f:.4f}" for f in result.fractions)
    mean = np.nanmean(result.profiles, axis=0)
    sd = np.nanstd(result.profiles, axis=0)
    profile = "\n".join(
        [
            "row\t" + header,
            "mean\t" + "\t".join(f"{x:.6g}" for x in mean),
            "sd\t" + "\t".join(f"{x:.6g}" for x in sd),
            "n\t" + "\t".join(str(result.n_vertices) for _ in mean),
        ]
    )
    same = [
        (n, q)
        for n, q in enumerate(pins, start=1)
        if q.fractions.shape == result.fractions.shape
        and np.allclose(q.fractions, result.fractions)
    ]
    if same:
        notes = "".join(f"# pin{n}: {q.label}\n" for n, q in same)
        rows = [profile]
        for n, q in same:
            rows += [
                f"pin{n} mean\t" + "\t".join(f"{x:.6g}" for x in q.mean),
                f"pin{n} sd\t" + "\t".join(f"{x:.6g}" for x in q.sd),
                f"pin{n} n\t" + "\t".join(str(q.n_vertices) for _ in q.mean),
            ]
        profile = notes + "\n".join(rows)
    if result.timecourses is None:
        path.write_text(profile + "\n")
        return
    rows = ["volume\t" + header]
    rows += [
        f"{t}\t" + "\t".join(f"{x:.6g}" for x in row) for t, row in enumerate(result.timecourses)
    ]
    path.write_text("\n".join(rows) + "\n")
    path.with_name(path.stem + ".profile.tsv").write_text(profile + "\n")


__all__ = [
    "DepthPlot",
    "DepthResult",
    "DepthWindow",
    "PinnedLayer",
    "PinnedProfile",
    "depth_profiles",
    "export_table",
]
