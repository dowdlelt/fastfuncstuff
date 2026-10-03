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

The region is published to the surface store, so a surface window shows
which cortex the profile came from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.surface.profiles import sample_depths
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding as Key
from fastfuncstuff.viewer.ui.shortcuts import ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import SetDepthView, SetIndex, SetSurfaceEquivolume

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


def depth_profiles(
    session, vertices: dict[str, np.ndarray], layer, fractions: np.ndarray, equivolume: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(V, K)`` profiles at the current volume, ``(T, K)`` region-mean
    timecourses, and the full ``(V, K, T)`` samples behind them.

    Timecourses only for a time-linked layer whose full series is resident;
    otherwise ``None`` (the series is still loading, or the layer is a stat
    map whose sub-bricks are not time).
    """
    st = session.state
    idx = int(st.time_index if layer.time_linked else layer.volume_index)
    res = session.store.get(layer.key)
    four_d = layer.time_linked and res.array is not None and res.array.ndim == 4
    volume = res.array if four_d else session.volume(layer.key, idx)
    parts = []
    for hemi, ids in vertices.items():
        h = session.surfaces.hemis[hemi]
        w, p = h.states["white"][ids], h.states["pial"][ids]
        areas = None
        if equivolume:
            aw, ap = session.surfaces.vertex_areas(hemi)
            areas = (aw[ids], ap[ids])
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
        self.message = "click cortex to profile it"
        self.setMinimumSize(200, 160)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        r = self.result
        if r is None or r.n_vertices == 0:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        plot = QtCore.QRectF(48, 22, max(self.width() - 60, 10), max(self.height() - 48, 10))
        x = r.fractions
        prof = r.profiles
        mean = np.nanmean(prof, axis=0)
        sd = np.nanstd(prof, axis=0)
        # Scale to the mean +- 2 SD rather than to every vertex's extremes, so
        # one wild vertex does not flatten the profile everyone came to see.
        lo = float(np.nanmin(mean - 2 * sd))
        hi = float(np.nanmax(mean + 2 * sd))
        if not np.isfinite(lo) or hi <= lo:
            lo, hi = (lo - 1, lo + 1) if np.isfinite(lo) else (0.0, 1.0)

        def pt(xv: float, yv: float) -> QtCore.QPointF:
            return QtCore.QPointF(
                plot.left() + (xv - x[0]) / (x[-1] - x[0]) * plot.width(),
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
            p.drawLine(pt(x[0], 0.0), pt(x[-1], 0.0))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        accent = QtGui.QColor(c.accent)
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
        band = QtGui.QPolygonF(
            [pt(x[k], mean[k] + sd[k]) for k in range(x.size)]
            + [pt(x[k], mean[k] - sd[k]) for k in reversed(range(x.size))]
        )
        fill = QtGui.QColor(accent)
        fill.setAlphaF(0.22)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPolygon(band)
        pen = QtGui.QPen(accent)
        pen.setWidthF(2.2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        path = QtGui.QPainterPath(pt(x[0], mean[0]))
        for k in range(1, x.size):
            path.lineTo(pt(x[k], mean[k]))
        p.drawPath(path)
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
        p.drawText(
            QtCore.QPointF(plot.left(), 15),
            f"{r.layer_name} · {r.region} · {r.n_vertices} vertices · mean +- SD",
        )
        p.end()


class DepthTimePlot(QtWidgets.QWidget):
    """Region mean at every depth (rows, pial on top) and volume (columns)."""

    clicked = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: np.ndarray | None = None  # (T, K)
        self.fractions: np.ndarray | None = None
        self.cursor = 0
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
        d = self.data.T[::-1]  # (K, T), pial row first
        # Each depth relative to its own mean, so the time structure shows at
        # every depth rather than only where the baseline is largest.
        d = d - np.nanmean(d, axis=1, keepdims=True)
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
        p.drawText(
            QtCore.QPointF(area.left(), area.bottom() + 14), "time  (each depth minus its mean)"
        )
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
        bar.addStretch(1)
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
                Key("e", "equivolume / equidistant depth", self._toggle_equivolume, group="depth"),
                Key("(", "smaller disc", lambda: self._radius_by(1 / 1.25), group="depth"),
                Key(")", "larger disc", lambda: self._radius_by(1.25), group="depth"),
                Key("f", "follow the crosshair / pin the region", self._toggle_follow, group="depth"),
                Key("click (heatmap)", "go to that volume", None, group="depth"),
                Key("h", "this list", self.help.toggle, group="window"),
                Key("w", "close this window", self.close, group="window"),
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
        self.roi_box.setEnabled(vp.depth_source == "layer")

    def _profiled_layer(self, vp: Viewport):
        st = self.session.state
        if vp.depth_layer:
            return st.layers.find(vp.depth_layer)
        sel = st.layers.find(st.selected) if st.selected else None
        base = st.layers.base
        if sel is not None and not sel.roi and (base is None or sel.key != base.key):
            return sel
        for layer in reversed(list(st.layers)[1:]):
            if layer.visible and not layer.roi:
                return layer
        return base

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
            self.plot.message = "load surfaces and an overlay"
            self.plot.result = None
            self.plot.update()
            return
        mm = st.crosshair_mm
        region_key = (
            None if mm is None else tuple(np.round(mm, 3)),
            vp.depth_source,
            vp.radius,
            vp.depth_roi_layer,
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
        idx = int(st.time_index if layer.time_linked else layer.volume_index)
        key = (
            tuple((h, ids.tobytes()) for h, ids in self._vertices.items()),
            layer.key,
            idx if not layer.time_linked else "series",
            vp.equivolume,
            tuple(sorted(surfaces.version.items())),
        )
        n = sum(ids.size for ids in self._vertices.values())
        if n == 0:
            self.result = None
            self.plot.message = "no cortex here: click within 3 mm of the ribbon"
            self.plot.result = None
            self.time_plot.data = None
        elif key != self._built:
            fractions = np.linspace(-MARGIN, 1 + MARGIN, int(vp.depth_bins))
            profiles, courses, samples = depth_profiles(
                self.session, self._vertices, layer, fractions, vp.equivolume
            )
            keep = samples if samples is not None and samples.nbytes < 512 * 2**20 else None
            self.result = DepthResult(
                fractions, profiles, courses, n, self._region_name(vp), layer.name, idx, keep
            )
            self._built = key
        elif self.result is not None and self.result.timecourses is not None:
            # Same region and series: only the current volume moved.
            self.result.time_index = idx
            if self.result.series is not None:
                self.result.profiles = self.result.series[:, :, idx]
            else:
                self.result.profiles = depth_profiles(
                    self.session, self._vertices, layer, self.result.fractions, vp.equivolume
                )[0]
        self.plot.result = self.result
        self.time_plot.data = None if self.result is None else self.result.timecourses
        self.time_plot.cursor = idx
        self.time_plot.setVisible(self.time_plot.data is not None)
        self.plot.update()
        self.time_plot.update()

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
            export_table(self.result, Path(path))


def export_table(result: DepthResult, path: Path) -> None:
    """Write the region's depth profile; with a series, its ``(T, K)`` depth timecourses.

    Columns are depths (fractions, white 0 to pial 1, the margins outside).
    Without a series: rows ``mean``, ``sd``, ``n``. With one: a row per volume
    -- the shape laminar models read -- and the profile in a ``.profile.tsv``
    beside it.
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
    if result.timecourses is None:
        path.write_text(profile + "\n")
        return
    rows = ["volume\t" + header]
    rows += [
        f"{t}\t" + "\t".join(f"{x:.6g}" for x in row) for t, row in enumerate(result.timecourses)
    ]
    path.write_text("\n".join(rows) + "\n")
    path.with_name(path.stem + ".profile.tsv").write_text(profile + "\n")


__all__ = ["DepthPlot", "DepthResult", "DepthWindow", "depth_profiles", "export_table"]
