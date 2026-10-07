"""CHEDI: a piece of cortex laid flat, the anatomy sampled onto it at one depth.

The patch is centred on the vertex nearest the crosshair, so anything that
moves the crosshair -- a click in a slice, on the inflated surface, in a
profile -- moves it. With no cortex near the crosshair it stays where it was.
Depth runs from white (0) to pial (1) and past both, so grey matter still
showing beyond pial, or dura bright inside it, is where the mesh is wrong.

The sampling is :mod:`viewer.chedi`; this window only draws it and turns keys
into commands. Selection and push/pull are the next step (see the wiki's
CHEDI note).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.chedi import Patch, PatchSampler, build_patch
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.panes import _segment_path
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import SetPatchSize, SetSurfaceDepth, SetViewSampling, SetXYZ

#: How far from the crosshair (mm) a vertex may be and still be followed.
FOLLOW_MM = 10.0
#: Pixels across the sampled patch. The canvas scales it; more is slower to
#: build and sample and shows nothing an anatomy at ~1 mm has to give.
PATCH_PIXELS = 256
#: The ends of the depth range, as the depth command allows.
DEPTH_RANGE = (-0.5, 1.5)


class PatchCanvas(QtWidgets.QWidget):
    """The flat image, the mesh over it, and the centre mark."""

    #: Double-click: fractional (row, col) in patch pixels.
    located = QtCore.Signal(float, float)
    #: Wheel: signed notches.
    wheeled = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.image: QtGui.QImage | None = None
        self.size_px = PATCH_PIXELS
        self.mesh: QtGui.QPainterPath | None = None
        self.show_mesh = True
        self.message = ""
        self.caption = ""
        self.setMinimumSize(160, 160)

    def target(self) -> QtCore.QRectF:
        """The square the patch is drawn into, centred."""
        side = min(self.width(), self.height())
        return QtCore.QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self.image is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        rect = self.target()
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(rect, self.image)
        scale = rect.width() / self.size_px
        if self.show_mesh and self.mesh is not None:
            p.save()
            p.setClipRect(rect)
            p.translate(rect.x(), rect.y())
            p.scale(scale, scale)
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*c.crosshair, 0.22))
            pen.setCosmetic(True)
            pen.setWidthF(0.6)
            p.setPen(pen)
            p.drawPath(self.mesh)
            p.restore()
        # The centre vertex: where the crosshair is.
        mid = rect.center()
        p.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(*c.crosshair, 0.9), 1.0))
        p.drawLine(QtCore.QLineF(mid.x() - 8, mid.y(), mid.x() - 3, mid.y()))
        p.drawLine(QtCore.QLineF(mid.x() + 3, mid.y(), mid.x() + 8, mid.y()))
        p.drawLine(QtCore.QLineF(mid.x(), mid.y() - 8, mid.x(), mid.y() - 3))
        p.drawLine(QtCore.QLineF(mid.x(), mid.y() + 3, mid.x(), mid.y() + 8))
        # On a plate: the caption sits over anatomy, which can be any brightness.
        metrics = QtGui.QFontMetrics(p.font())
        plate = QtCore.QRect(
            2, 2, metrics.horizontalAdvance(self.caption) + 10, metrics.height() + 4
        )
        ground = QtGui.QColor(c.bg)
        ground.setAlphaF(0.72)
        p.fillRect(plate, ground)
        p.setPen(QtGui.QColor(c.text))
        p.drawText(plate.adjusted(5, 0, 0, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, self.caption)
        p.end()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        rect = self.target()
        if self.image is None or not rect.contains(event.position()):
            return
        scale = rect.width() / self.size_px
        self.located.emit(
            (event.position().y() - rect.y()) / scale, (event.position().x() - rect.x()) / scale
        )

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta().y()
        if delta:
            self.wheeled.emit(1 if delta > 0 else -1)


class ChediWindow(QtWidgets.QWidget):
    """A CHEDI viewport as a top-level window."""

    closed = QtCore.Signal(str)

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
        #: The patch on screen and what it was built from:
        #: (hemi, centre, half_mm, topology version).
        self.patch: Patch | None = None
        self._sampler: PatchSampler | None = None
        self._built: tuple | None = None
        #: (hemi, vertex) the patch follows; kept when the crosshair leaves cortex.
        self._centre: tuple[str, int] | None = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.canvas = PatchCanvas()
        self.canvas.located.connect(self._locate)
        self.canvas.wheeled.connect(lambda n: self._depth_by(0.05 * n))
        v.addWidget(self.canvas, 1)
        self.resize(420, 440)

        self.help = ShortcutHelp(self, f"chedi · {vid}")
        self.help.apply(
            [
                Binding(
                    "[", "shallower (toward white)", lambda: self._depth_by(-0.1), group="depth"
                ),
                Binding(
                    "]",
                    "deeper (toward pial and past it)",
                    lambda: self._depth_by(0.1),
                    group="depth",
                ),
                Binding("scroll", "depth in steps of 0.05", None, group="depth"),
                Binding("0", "mid-depth (0.5)", lambda: self._set_depth(0.5), group="depth"),
                Binding("1", "pial (1.0)", lambda: self._set_depth(1.0), group="depth"),
                Binding(
                    "+",
                    "smaller patch (closer)",
                    lambda: self._size_by(1 / 1.25),
                    group="view",
                    aliases=("=",),
                ),
                Binding("-", "larger patch", lambda: self._size_by(1.25), group="view"),
                Binding("m", "mesh over the image on / off", self._toggle_mesh, group="view"),
                Binding(
                    "n", "voxels: nearest / linear / cubic", self._cycle_sampling, group="view"
                ),
                Binding(
                    "double-click", "move the crosshair there (and the patch)", None, group="view"
                ),
                Binding("h", "this list", self.help.toggle, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- state -------------------------------------------------------------
    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def current_depth(self) -> float:
        vp = self._viewport()
        return 0.5 if vp is None else float(vp.depth[0])

    def _set_depth(self, d: float) -> None:
        d = float(np.clip(round(d, 3), *DEPTH_RANGE))
        self._dispatch(SetSurfaceDepth(self.vid, d, d))

    def _depth_by(self, delta: float) -> None:
        self._set_depth(self.current_depth() + delta)

    def _size_by(self, factor: float) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetPatchSize(self.vid, vp.patch_mm * factor))

    def _cycle_sampling(self) -> None:
        from fastfuncstuff.surface.sampling import MODES

        vp = self._viewport()
        if vp is not None:
            nxt = MODES[(MODES.index(vp.sampling) + 1) % len(MODES)]
            self._dispatch(SetViewSampling(self.vid, nxt))

    def layer(self):
        """What the wall shows: the selected layer, else the top visible overlay, else the base.

        The overlay rather than the base because the base is usually what the
        slices are anchored to, and the image worth judging the mesh against
        -- the anatomy loaded again on top, a T2, an edge map -- is above it.
        Selecting a layer is how to choose.
        """
        layers = self.session.state.layers
        base = layers.base
        selected = self.session.state.selected_layer()
        if selected is not None and (base is None or selected.key != base.key):
            return selected
        above = [layer for layer in list(layers)[1:] if layer.visible]
        return above[-1] if above else base

    def _toggle_mesh(self) -> None:
        self.canvas.show_mesh = not self.canvas.show_mesh
        self.canvas.update()

    def _locate(self, row: float, col: float) -> None:
        """Crosshair to the cortex under a double-click, at the depth shown."""
        if self.patch is None or self._sampler is None:
            return
        r, c = int(row), int(col)
        if (
            not (0 <= r < self.patch.size and 0 <= c < self.patch.size)
            or not self.patch.inside[r, c]
        ):
            return
        h = self.session.surfaces.hemis[self.patch.hemi]
        points = self._sampler.points(h, self.current_depth(), self._version())
        # points are the inside pixels in row-major order.
        k = int(np.count_nonzero(self.patch.inside.ravel()[: r * self.patch.size + c]))
        x, y, z = (float(v) for v in points[k])
        self._dispatch(SetXYZ(x, y, z))

    def _version(self) -> tuple:
        surfaces = self.session.surfaces
        hemi = self.patch.hemi if self.patch is not None else ""
        return (surfaces.version.get(hemi, 0), surfaces.topology_version)

    # -- the viewport protocol --------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.canvas.update()

    def refresh(self, dirty: Aspect = Aspect.ALL) -> None:
        vp = self._viewport()
        surfaces = self.session.surfaces
        if vp is None:
            return
        if not surfaces.hemis:
            self._clear("load a subject's surfaces first (SURF)")
            return
        self._follow()
        if self._centre is None:
            self._clear("put the crosshair on cortex")
            return
        hemi, centre = self._centre
        key = (hemi, centre, float(vp.patch_mm), surfaces.topology_version)
        if key != self._built:
            try:
                self.patch = build_patch(surfaces.hemis[hemi], centre, vp.patch_mm, PATCH_PIXELS)
            except ValueError as exc:
                self._clear(str(exc))
                return
            self._sampler = PatchSampler(self.patch)
            self._built = key
            self._build_mesh()
        self._draw()

    def _follow(self) -> None:
        """Re-centre on the vertex nearest the crosshair, if there is one near."""
        mm = self.session.state.crosshair_mm
        if mm is None:
            return
        hit = self.session.surfaces.nearest_vertex(mm, max_mm=FOLLOW_MM)
        if hit is not None:
            self._centre = (hit[0], hit[1])
        elif self._centre is not None and self._centre[0] not in self.session.surfaces.hemis:
            self._centre = None
        if self._centre is not None:
            h = self.session.surfaces.hemis[self._centre[0]]
            if self._centre[1] >= h.n_vertices:  # a topology edit shrank the mesh
                self._centre = None

    def _clear(self, message: str) -> None:
        self.patch = None
        self._sampler = None
        self._built = None
        self.canvas.image = None
        self.canvas.mesh = None
        self.canvas.message = message
        self.canvas.update()

    def _build_mesh(self) -> None:
        """The patch's triangle edges as one path, in patch pixels."""
        p = self.patch
        assert p is not None
        h = self.session.surfaces.hemis[p.hemi]
        uv = p.uv_of(h.n_vertices)
        f = p.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        rc = p.to_pixels(uv[edges])  # (E, 2, 2) as (row, col)
        self.canvas.mesh = _segment_path(rc[..., ::-1])
        self.canvas.size_px = p.size

    def _draw(self) -> None:
        p, sampler = self.patch, self._sampler
        assert p is not None and sampler is not None
        session = self.session
        layer = self.layer()
        vp = self._viewport()
        mode = vp.sampling if vp is not None else "nearest"
        try:
            volume = session.surface_sampler(layer.key if layer is not None else None, mode)
        except ValueError as exc:
            self._clear(str(exc))
            return
        h = session.surfaces.hemis[p.hemi]
        depth = self.current_depth()
        values = sampler.sample(h, depth, volume, self._version())
        lo, hi = self._window(values, layer)
        grey = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = np.where(np.isfinite(grey), grey * 255.0, 0.0).astype(np.uint8)
        rgba = np.empty((*rgb.shape, 4), np.uint8)
        rgba[..., :3] = rgb[..., None]
        rgba[..., 3] = np.where(p.inside, 255, 0)
        self.canvas.image = QtGui.QImage(
            rgba.data, p.size, p.size, 4 * p.size, QtGui.QImage.Format.Format_RGBA8888
        ).copy()
        self.canvas.caption = (
            f"{p.hemi} #{p.centre}   depth {depth:.2f}   ±{p.half_mm:.0f} mm   "
            f"{layer.name if layer is not None else ''} · {mode}"
            + ("" if p.source == "sphere" else "   (no sphere: inflated)")
        )
        self.canvas.update()

    def _window(self, values: np.ndarray, layer) -> tuple[float, float]:
        """The layer's own display range, so depths compare; else this patch's spread.

        Not re-fitted per depth: a pial in dura is a patch that is *brighter*
        than it should be, and an auto-range would scale that away.
        """
        if layer is not None and layer.range_lo is not None and layer.range_hi is not None:
            if layer.range_hi > layer.range_lo:
                return float(layer.range_lo), float(layer.range_hi)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return 0.0, 1.0
        lo, hi = np.percentile(finite, [2, 98])
        return float(lo), float(hi)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["ChediWindow", "PatchCanvas"]
