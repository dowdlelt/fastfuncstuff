"""CHEDI: a piece of cortex laid flat, the anatomy sampled onto it at one depth.

The patch is centred on the vertex nearest the crosshair, so anything that
moves the crosshair -- a click in a slice, on the inflated surface, in a
profile -- moves it. With no cortex near the crosshair it stays where it was.
Depth runs from white (0) to pial (1) and past both, so grey matter still
showing beyond pial, or dura bright inside it, is where the mesh is wrong.

Selecting is the surface highlight every window shares (the 2-D MARK dots,
the 3-D paint), so a selection made here shows everywhere and a recorded
script replays it. Pushing moves the selected *and visible* vertices of one
surface along their normals as a unit: what scrolled off the patch stays put.
The sampling and the selection arithmetic are :mod:`viewer.chedi`; this
window draws them and turns gestures into commands.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.chedi import (
    MAX_CLUSTERS,
    PROFILE_OFFSETS,
    Patch,
    PatchSampler,
    adjacency,
    build_patch,
    cluster_features,
    dilate,
    drop_isolated,
    erode,
    kmeans,
    order_clusters,
    visible_vertices,
    window_select,
)
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.panes import _segment_path
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    HighlightSurface,
    MoveSurfaceHighlight,
    SetPatchLayer,
    SetPatchSize,
    SetSurfaceDepth,
    SetSurfaceStep,
    SetViewSampling,
    SetXYZ,
    encode_ids,
)

#: How far from the crosshair (mm) a vertex may be and still be followed.
FOLLOW_MM = 10.0
#: Pixels across the sampled patch. The canvas scales it; more is slower to
#: build and sample and shows nothing an anatomy at ~1 mm has to give.
PATCH_PIXELS = 256
#: The ends of the depth range, as the depth command allows.
DEPTH_RANGE = (-0.5, 1.5)
#: How far past the selected vertices a push fades out, mm. Small: the
#: shoulder is the one part of a push that can reach past the patch's edge.
SHOULDER_MM = 2.0
#: Sweeps of evening out the spacing one u takes (see surface.mesh.even_spacing).
EVEN_SWEEPS = 30
#: The spacing map's window: log2(pial / white vertex area) from a quarter
#: (black, crowded) to four times (white, stretched); 1 is mid grey.
SPACING_LOG2 = (-2.0, 2.0)
#: What o steps through: the anatomy, gyri/sulci, how crowded pial is.
DISPLAYS = ("data", "folding", "spacing")
#: Widget pixels of ctrl+drag that move the window by its layer's full range.
WINDOW_DRAG_PX = 300.0
#: Selection dot colour.
SELECT_RGB = (1.0, 0.55, 0.1)
#: Mesh wireframe: a bright blue that stays visible over dark CSF and bright
#: white matter alike, and is nothing like the orange selection.
MESH_RGB = (0.3, 0.7, 1.0)
#: Cluster colours, dark to bright like the clusters themselves (cluster 1 is
#: the darkest): viridis at ten steps, so the order reads off the colour.
CLUSTER_RGB = np.array(
    [
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.993, 0.906, 0.144),
    ]
)
#: Position weights ``x`` steps through (see chedi.cluster_features).
SPATIAL_STEPS = (0.0, 0.5, 1.0)


class PatchCanvas(QtWidgets.QWidget):
    """The flat image, the mesh over it, and the centre mark."""

    #: Double-click or shift+click: fractional (row, col) in patch pixels.
    located = QtCore.Signal(float, float)
    #: Wheel: signed notches.
    wheeled = QtCore.Signal(int)
    #: A selecting press: "add" (left), "remove" (right), "window" (ctrl+left)
    #: or "window+" (ctrl+shift+left), at (row, col) patch pixels.
    pressed = QtCore.Signal(str, float, float)
    #: The drag since the press: (row, col) now, and (dx, dy) widget pixels
    #: from where it began.
    dragged = QtCore.Signal(float, float, float, float)
    released = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.image: QtGui.QImage | None = None
        self.size_px = PATCH_PIXELS
        self.mesh: QtGui.QPainterPath | None = None
        self.show_mesh = True
        self.message = ""
        self.caption = ""
        #: Selected vertices on screen, ``(N, 2)`` (row, col) patch pixels.
        self.dots = np.zeros((0, 2))
        #: Every on-screen vertex coloured by its cluster, ``(N, 2)`` and
        #: ``(N, 3)``, and the legend's (colour, text) lines; empty when off.
        self.cluster_dots = np.zeros((0, 2))
        self.cluster_rgb = np.zeros((0, 3))
        self.legend: list[tuple[tuple[float, float, float], str]] = []
        #: Paint-brush radius in patch pixels, drawn at the cursor.
        self.brush_px = 4.0
        self._hover: QtCore.QPointF | None = None
        self._press_at: QtCore.QPointF | None = None
        self.setMinimumSize(160, 160)
        self.setMouseTracking(True)

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
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*MESH_RGB, 0.55))
            pen.setCosmetic(True)
            pen.setWidthF(0.7)
            p.setPen(pen)
            p.drawPath(self.mesh)
            p.restore()
        radius = max(1.5, min(3.0, 0.35 * scale))
        if len(self.cluster_dots):
            p.save()
            p.setClipRect(rect)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            for (r, cc), rgb in zip(self.cluster_dots, self.cluster_rgb, strict=True):
                p.setBrush(QtGui.QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.9))
                p.drawEllipse(
                    QtCore.QPointF(rect.x() + cc * scale, rect.y() + r * scale), radius, radius
                )
            p.restore()
        if len(self.dots):
            p.save()
            p.setClipRect(rect)
            if len(self.cluster_dots):
                # Over cluster colours the selection is a ring, not a fill.
                p.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.95), 1.2))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                ring = radius + 1.5
            else:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor.fromRgbF(*SELECT_RGB, 0.85))
                ring = radius
            for r, cc in self.dots:
                p.drawEllipse(
                    QtCore.QPointF(rect.x() + cc * scale, rect.y() + r * scale), ring, ring
                )
            p.restore()
        if self.legend:
            self._paint_legend(p, rect)
        if self._hover is not None and rect.contains(self._hover):
            ring = QtGui.QPen(QtGui.QColor.fromRgbF(*SELECT_RGB, 0.7), 1.0)
            ring.setStyle(QtCore.Qt.PenStyle.DashLine)
            p.setPen(ring)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._hover, self.brush_px * scale, self.brush_px * scale)
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

    def _paint_legend(self, p: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        """Key, colour, size and mean value of each cluster, bottom right."""
        c = theme.palette()
        font = QtGui.QFont(p.font())
        font.setFamily(theme.MONO)
        font.setPointSize(8)
        p.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        line = metrics.height()
        width = max(metrics.horizontalAdvance(t) for _, t in self.legend) + 26
        height = line * len(self.legend) + 8
        plate = QtCore.QRectF(rect.right() - width - 4, rect.bottom() - height - 4, width, height)
        ground = QtGui.QColor(c.bg)
        ground.setAlphaF(0.78)
        p.fillRect(plate, ground)
        for n, (rgb, text) in enumerate(self.legend):
            y = plate.y() + 4 + n * line
            p.fillRect(
                QtCore.QRectF(plate.x() + 5, y + 2, 12, line - 4), QtGui.QColor.fromRgbF(*rgb)
            )
            p.setPen(QtGui.QColor(c.text))
            p.drawText(QtCore.QPointF(plate.x() + 22, y + metrics.ascent()), text)

    def _patch_point(self, pos: QtCore.QPointF) -> tuple[float, float]:
        rect = self.target()
        scale = rect.width() / self.size_px
        return (pos.y() - rect.y()) / scale, (pos.x() - rect.x()) / scale

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self.image is None:
            return
        mods = event.modifiers()
        ctrl = bool(
            mods
            & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier)
        )
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and shift and not ctrl:
            # Shift+click is "look here", not a selection.
            self.located.emit(*self._patch_point(event.position()))
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            kind = ("window+" if shift else "window") if ctrl else "add"
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            kind = "remove"
        else:
            return
        self._press_at = event.position()
        self.pressed.emit(kind, *self._patch_point(event.position()))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        self._hover = event.position()
        if self._press_at is not None:
            d = event.position() - self._press_at
            self.dragged.emit(*self._patch_point(event.position()), d.x(), d.y())
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._press_at is not None:
            self._press_at = None
            self.released.emit()

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt)
        self._hover = None
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta().y()
        if delta:
            self.wheeled.emit(1 if delta > 0 else -1)


class ChediWindow(QtWidgets.QWidget):
    """A CHEDI viewport as a top-level window."""

    closed = QtCore.Signal(str)
    #: The crosshair was put somewhere from here: the other windows should
    #: bring it into view, not only follow it.
    centre_requested = QtCore.Signal()

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
        #: The one surface a push moves. Pial: what is usually wrong.
        self.surface = "pial"
        #: What is drawn (o): the data, the gyri/sulci map, or the spacing map.
        self.display = "data"
        #: How a push is placed: by hand (the step), or snapped to the shown
        #: image's edge (gated, or "edge": strongest, ungated) -- m, as in a slice.
        self.snap_mode = "hand"
        #: Fraction of each point's spike height a push also takes out (shift+R).
        self.flatten = 0.3
        #: Clusters (c): shown or not, how many, how much position counts,
        #: and the last labels with what they were computed from.
        self.show_clusters = False
        self.k = 6
        self.spatial = 0.5
        self._labels: np.ndarray | None = None
        self._labels_key: tuple | None = None
        self._label_values = np.zeros(0)
        #: Paint-select brush radius, flat mm.
        self.brush_mm = 1.5
        #: Visible vertex ids, their (row, col) pixels, and as a hemisphere mask.
        self._vis = np.zeros(0, np.int64)
        self._vis_px = np.zeros((0, 2))
        self._vis_mask = np.zeros(0, bool)
        self._adj = None
        self._adj_key: tuple | None = None
        #: A gesture in progress: ("add" | "remove", painted ids) or
        #: ("window", base mask, level, width, values) -- previewed here and
        #: recorded once, on release.
        self._gesture: tuple | None = None
        self._preview: np.ndarray | None = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.canvas = PatchCanvas()
        self.canvas.located.connect(self._locate)
        self.canvas.wheeled.connect(lambda n: self._depth_by(0.05 * n))
        self.canvas.pressed.connect(self._press)
        self.canvas.dragged.connect(self._drag)
        self.canvas.released.connect(self._release)
        v.addWidget(self.canvas, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("value")
        self.status.setContentsMargins(6, 2, 6, 2)
        v.addWidget(self.status)
        self.resize(420, 460)

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
                Binding("ctrl+0", "mid-depth (0.5)", lambda: self._set_depth(0.5), group="depth"),
                Binding("ctrl+1", "pial (1.0)", lambda: self._set_depth(1.0), group="depth"),
                Binding(
                    "c",
                    "clusters on / off (k-means of what is shown)",
                    self._toggle_clusters,
                    group="clusters",
                ),
                *[
                    Binding(
                        str(n % 10),
                        f"select / unselect cluster {n}"
                        + (" (darkest)" if n == 1 else " (if k = 10)" if n == MAX_CLUSTERS else ""),
                        lambda n=n: self._pick_cluster(n),
                        group="clusters",
                    )
                    for n in range(1, MAX_CLUSTERS + 1)
                ],
                Binding("k", "more clusters", lambda: self._k_by(1), group="clusters"),
                Binding("shift+k", "fewer clusters", lambda: self._k_by(-1), group="clusters"),
                Binding(
                    "x",
                    "where a point is counts: off / some / strongly",
                    self._cycle_spatial,
                    group="clusters",
                ),
                Binding(
                    "+",
                    "smaller patch (closer)",
                    lambda: self._size_by(1 / 1.25),
                    group="view",
                    aliases=("=",),
                ),
                Binding("-", "larger patch", lambda: self._size_by(1.25), group="view"),
                Binding("shift+m", "mesh over the image on / off", self._toggle_mesh, group="view"),
                Binding(
                    "n", "voxels: nearest / linear / cubic", self._cycle_sampling, group="view"
                ),
                Binding(
                    "l",
                    "sample: follow the selection / each layer in turn",
                    self._cycle_layer,
                    group="view",
                ),
                Binding(
                    "shift+click",
                    "crosshair there; the other windows centre on it",
                    None,
                    group="view",
                ),
                Binding("double-click", "the same", None, group="view"),
                Binding("drag", "select under the brush", None, group="select"),
                Binding("right-drag", "unselect under the brush", None, group="select"),
                Binding(
                    "ctrl+drag",
                    "select by value (or spacing, when shown): up/down the level, "
                    "left/right the window (shift: add)",
                    None,
                    group="select",
                ),
                Binding("a", "erode the selection", lambda: self._morph("erode"), group="select"),
                Binding("d", "dilate the selection", lambda: self._morph("dilate"), group="select"),
                Binding(
                    "e", "drop isolated points", lambda: self._morph("isolated"), group="select"
                ),
                Binding(
                    "v",
                    "invert the selection (on screen)",
                    lambda: self._morph("invert"),
                    group="select",
                ),
                Binding("Escape", "clear the selection", self._clear_selection, group="select"),
                Binding("(", "smaller brush", lambda: self._brush_by(1 / 1.25), group="select"),
                Binding(")", "larger brush", lambda: self._brush_by(1.25), group="select"),
                Binding(
                    "o",
                    "show: data -> gyri / sulci -> spacing (dark = crowded)",
                    self._cycle_display,
                    group="view",
                ),
                Binding(
                    "f", "unselect sulci (on screen)", lambda: self._drop_fold(1), group="select"
                ),
                Binding(
                    "g", "unselect gyri (on screen)", lambda: self._drop_fold(-1), group="select"
                ),
                Binding("w", "push the selection in", lambda: self._push(-1.0), group="move"),
                Binding("s", "pull the selection out", lambda: self._push(1.0), group="move"),
                Binding("t", "move pial / white", self._toggle_surface, group="move"),
                Binding(
                    "m",
                    "pushes: by hand -> snap to edge -> strongest edge",
                    self._cycle_snap,
                    group="move",
                ),
                Binding("r", "relax: smooth the spikes out, no push", self._relax, group="move"),
                Binding(
                    "u",
                    "uncrowd: even out the spacing, within the surface",
                    self._even,
                    group="move",
                ),
                Binding(
                    "shift+r",
                    "flatten with each push: 0 / 0.3 / 0.6",
                    self._cycle_flatten,
                    group="move",
                ),
                Binding("{", "smaller step", lambda: self._step_by(1 / 1.5), group="move"),
                Binding("}", "larger step", lambda: self._step_by(1.5), group="move"),
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

    def _cycle_layer(self) -> None:
        """l: follow the selection -> each loaded layer, bottom to top -> follow again."""
        vp = self._viewport()
        if vp is None:
            return
        keys = ["", *(layer.key for layer in self.session.state.layers)]
        now = vp.patch_layer if vp.patch_layer in keys else ""
        nxt = keys[(keys.index(now) + 1) % len(keys)]
        self._dispatch(SetPatchLayer(self.vid, nxt))
        layer = self.layer()
        name = layer.name if layer is not None else "nothing"
        self.status.setText(f"samples {name}" + ("" if nxt else " (following the selection)"))

    def layer(self):
        """What the wall shows: the layer ``l`` picked; else the selected layer, else
        the top visible overlay, else the base.

        The overlay rather than the base because the base is usually what the
        slices are anchored to, and the image worth judging the mesh against
        -- the anatomy loaded again on top, a T2, an edge map -- is above it.
        Selecting a layer is how to choose.
        """
        layers = self.session.state.layers
        vp = self._viewport()
        if vp is not None and vp.patch_layer:
            picked = layers.find(vp.patch_layer)
            if picked is not None:
                return picked
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
        self.centre_requested.emit()

    # -- selection ---------------------------------------------------------
    def _selected(self) -> np.ndarray:
        """The hemisphere's selection (the shared highlight), as a vertex mask."""
        assert self.patch is not None
        h = self.session.surfaces.hemis[self.patch.hemi]
        mask = self.session.surfaces.highlight.get(self.patch.hemi)
        return np.zeros(h.n_vertices, bool) if mask is None else mask.copy()

    def _set_selected(self, mask: np.ndarray) -> None:
        assert self.patch is not None
        self._dispatch(HighlightSurface(self.patch.hemi, encode_ids(np.flatnonzero(mask)), "set"))

    def _under_brush(self, row: float, col: float) -> np.ndarray:
        """Visible vertex ids within the brush of (row, col) patch pixels."""
        if self.patch is None or not self._vis.size:
            return np.zeros(0, np.int64)
        r = self.brush_mm / self.patch.mm_per_pixel
        d = np.hypot(self._vis_px[:, 0] - row, self._vis_px[:, 1] - col)
        hit = self._vis[d <= r]
        if not hit.size and d.min() <= 2.0 * r:
            hit = self._vis[[int(np.argmin(d))]]  # a click always takes the nearest point
        return hit

    def _vertex_values(self, depth: float | None = None) -> np.ndarray:
        """The image at every visible vertex, at ``depth`` (the one shown) and as sampled."""
        assert self.patch is not None
        h = self.session.surfaces.hemis[self.patch.hemi]
        layer = self.layer()
        vp = self._viewport()
        sampler = self.session.surface_sampler(
            layer.key if layer is not None else None, vp.sampling if vp is not None else "nearest"
        )
        w = h.states["white"][self._vis].astype(np.float64)
        p = h.states["pial"][self._vis].astype(np.float64)
        d = self.current_depth() if depth is None else depth
        return sampler(w + d * (p - w))

    def _press(self, kind: str, row: float, col: float) -> None:
        if self.patch is None or not self._vis.size:
            return
        if kind in ("add", "remove"):
            self._gesture = (kind, set(self._under_brush(row, col).tolist()))
        else:
            # By what is shown: on the spacing map, ctrl+drag selects the crowded.
            if self.display == "spacing":
                values = self._spacing()[self._vis]
                lo, hi = SPACING_LOG2
            else:
                values = self._vertex_values()
                lo, hi = self._window(values, self.layer())
            nearest = int(np.argmin(np.hypot(self._vis_px[:, 0] - row, self._vis_px[:, 1] - col)))
            span = max(hi - lo, 1e-6)
            base = self._selected() if kind == "window+" else self._selected() & ~self._vis_mask
            self._gesture = ("window", base, float(values[nearest]), 0.1 * span, values, span)
        self._update_preview(0.0, 0.0)

    def _drag(self, row: float, col: float, dx: float, dy: float) -> None:
        if self._gesture is None:
            return
        if self._gesture[0] in ("add", "remove"):
            self._gesture[1].update(self._under_brush(row, col).tolist())
        self._update_preview(dx, dy)

    def _update_preview(self, dx: float, dy: float) -> None:
        g = self._gesture
        assert g is not None
        mask = self._selected()
        if g[0] in ("add", "remove"):
            ids = np.fromiter(g[1], np.int64, len(g[1]))
            mask[ids] = g[0] == "add"
            self.status.setText(f"{'select' if g[0] == 'add' else 'unselect'}  {len(ids)} points")
        else:
            _, base, level0, width0, values, span = g
            # Up raises the level, right widens the window, both in units of
            # the layer's range per WINDOW_DRAG_PX of drag.
            level = level0 - dy / WINDOW_DRAG_PX * span
            width = max(0.0, width0 + dx / WINDOW_DRAG_PX * span)
            mask = base.copy()
            mask[self._vis[window_select(values, level, width)]] = True
            self.status.setText(
                f"value {level - width / 2:.4g} .. {level + width / 2:.4g}   "
                f"{int(mask[self._vis].sum())} points on screen"
            )
        self._preview = mask
        self._show_dots()

    def _release(self) -> None:
        g, preview = self._gesture, self._preview
        self._gesture = None
        self._preview = None
        if g is None or preview is None or self.patch is None:
            return
        if g[0] in ("add", "remove"):
            if g[1]:
                ids = np.fromiter(g[1], np.int64, len(g[1]))
                self._dispatch(HighlightSurface(self.patch.hemi, encode_ids(ids), g[0]))
        else:
            self._set_selected(preview)
        self._show_dots()

    def _morph(self, how: str) -> None:
        if self.patch is None or not self._vis.size:
            return
        adj = self._adjacency()
        mask = self._selected()
        before = int(mask[self._vis].sum())
        if how == "erode":
            new = erode(mask, adj, self._vis_mask)
        elif how == "dilate":
            new = dilate(mask, adj, self._vis_mask)
        elif how == "isolated":
            new = drop_isolated(mask, adj)
            # Only on screen: a lone point off it is not one this window shows.
            new = np.where(self._vis_mask, new, mask)
        else:
            new = mask.copy()
            new[self._vis] = ~mask[self._vis]
        self._set_selected(new)
        self.status.setText(f"{how}: {before} -> {int(new[self._vis].sum())} points on screen")

    def _folding(self) -> np.ndarray | None:
        """Per vertex: +1 sulcus, -1 gyrus (0 flat), as the 3-D window's binary shade."""
        from fastfuncstuff.viewer.surface3d import folding_values

        assert self.patch is not None
        h = self.session.surfaces.hemis[self.patch.hemi]
        if "curv" not in h.morph:
            return None
        return folding_values(h, "binary")

    def _spacing(self) -> np.ndarray:
        """Per vertex: log2 of its pial area over its white area -- below 0, pial is crowded.

        Against the vertex's own white area, not a fixed density: vertex
        spacing varies across a hemisphere anyway, and white is where each
        vertex's share of cortex is set. Sulcal fundi sit a little below 0 by
        nature; a protrusion pushed back in sits far below.
        """
        assert self.patch is not None
        white, pial = self.session.surfaces.vertex_areas(self.patch.hemi)
        return np.log2(np.maximum(pial, 1e-6) / np.maximum(white, 1e-6))

    def _cycle_display(self) -> None:
        nxt = DISPLAYS[(DISPLAYS.index(self.display) + 1) % len(DISPLAYS)]
        skipped = ""
        if nxt == "folding" and self.patch is not None and self._folding() is None:
            skipped = "no ?h.curv, no gyri/sulci map -- "
            nxt = "spacing"
        self.display = nxt
        self.status.setText(
            skipped
            + {
                "data": "data",
                "folding": "gyri (light) / sulci (dark)",
                "spacing": "pial / white vertex area: dark = crowded, mid grey = even",
            }[nxt]
        )
        if self.patch is not None:
            self._draw()

    def _drop_fold(self, which: int) -> None:
        """Unselect the on-screen points on sulci (``which`` 1) or gyri (-1)."""
        if self.patch is None or not self._vis.size:
            return
        fold = self._folding()
        if fold is None:
            self.status.setText("no ?h.curv: cannot tell gyri from sulci")
            return
        mask = self._selected()
        before = int(mask[self._vis].sum())
        drop = self._vis[fold[self._vis] * which > 0]
        mask[drop] = False
        self._set_selected(mask)
        name = "sulci" if which > 0 else "gyri"
        self.status.setText(
            f"off {name}: {before} -> {int(mask[self._vis].sum())} points on screen"
        )

    # -- clusters ------------------------------------------------------------
    def _cluster_labels(self) -> np.ndarray | None:
        """Cluster of every visible vertex (0 darkest .. k-1 brightest), cached."""
        if self.patch is None or not self._vis.size:
            return None
        layer = self.layer()
        vp = self._viewport()
        depth = self.current_depth()
        key = (
            self._built,
            self._version(),
            round(depth, 4),
            None if layer is None else layer.key,
            None if vp is None else vp.sampling,
            self.k,
            self.spatial,
        )
        if key == self._labels_key and self._labels is not None:
            return self._labels
        profile = np.stack([self._vertex_values(depth + o) for o in PROFILE_OFFSETS], axis=1)
        fold = self._folding()
        if fold is None:
            fold = np.zeros(self.session.surfaces.hemis[self.patch.hemi].n_vertices)
        h = self.session.surfaces.hemis[self.patch.hemi]
        uv = self.patch.uv_of(h.n_vertices)[self._vis]
        x = cluster_features(profile, fold[self._vis], uv, self.patch.half_mm, self.spatial)
        labels = kmeans(x, self.k)
        middle = PROFILE_OFFSETS.index(0.0)
        self._labels = order_clusters(labels, profile[:, middle], self.k)
        self._labels_key = key
        self._label_values = profile[:, middle]
        return self._labels

    def _toggle_clusters(self) -> None:
        self.show_clusters = not self.show_clusters
        self.status.setText(
            f"{self.k} clusters, 1 darkest .. {self.k} brightest: press a number to (un)select one"
            if self.show_clusters
            else "clusters off"
        )
        self._show_dots()

    def _k_by(self, delta: int) -> None:
        self.k = int(np.clip(self.k + delta, 2, MAX_CLUSTERS))
        self.show_clusters = True
        self.status.setText(f"{self.k} clusters")
        self._show_dots()

    def _cycle_spatial(self) -> None:
        i = SPATIAL_STEPS.index(self.spatial) if self.spatial in SPATIAL_STEPS else 0
        self.spatial = SPATIAL_STEPS[(i + 1) % len(SPATIAL_STEPS)]
        self.show_clusters = True
        words = {0.0: "off", 0.5: "some", 1.0: "strongly"}[self.spatial]
        self.status.setText(f"position in the clustering: {words}")
        self._show_dots()

    def _pick_cluster(self, n: int) -> None:
        """Toggle cluster ``n`` (1-based) in the selection: all in -> out, else in."""
        if self.patch is None:
            return
        if n > self.k:
            self.status.setText(f"only {self.k} clusters (k for more)")
            return
        labels = self._cluster_labels()
        if labels is None:
            return
        self.show_clusters = True
        ids = self._vis[labels == n - 1]
        if not ids.size:
            self.status.setText(f"cluster {n} is empty")
            return
        everything = bool(self._selected()[ids].all())
        mode = "remove" if everything else "add"
        self._dispatch(HighlightSurface(self.patch.hemi, encode_ids(ids), mode))
        self.status.setText(f"cluster {n}: {ids.size} points {'out' if everything else 'in'}")
        self._show_dots()

    def _clear_selection(self) -> None:
        self._gesture = None
        self._preview = None
        self._dispatch(HighlightSurface(mode="clear"))
        self.status.setText("selection cleared")

    def _brush_by(self, factor: float) -> None:
        self.brush_mm = float(np.clip(self.brush_mm * factor, 0.3, 15.0))
        if self.patch is not None:
            self.canvas.brush_px = self.brush_mm / self.patch.mm_per_pixel
        self.canvas.update()
        self.status.setText(f"brush {self.brush_mm:.2g} mm")

    def _toggle_surface(self) -> None:
        self.surface = "white" if self.surface == "pial" else "pial"
        self.status.setText(f"W/S move the {self.surface}")
        if self.patch is not None:
            self._draw()

    def _step_by(self, factor: float) -> None:
        step = float(np.clip(self.session.state.surface_step * factor, 0.05, 5.0))
        self._dispatch(SetSurfaceStep(round(step, 3)))
        self.status.setText(f"step {self.session.state.surface_step:g} mm")

    def _cycle_snap(self) -> None:
        modes = ("hand", "snap", "edge")
        self.snap_mode = modes[(modes.index(self.snap_mode) + 1) % len(modes)]
        self.status.setText(
            {
                "hand": "pushes by hand: exactly the step",
                "snap": "pushes snap to the shown image's edge (gated by tissue level)",
                "edge": "pushes snap to the strongest edge (ungated)",
            }[self.snap_mode]
        )
        if self.patch is not None:
            self._draw()

    def _cycle_flatten(self) -> None:
        steps = (0.0, 0.3, 0.6)
        i = steps.index(self.flatten) if self.flatten in steps else 0
        self.flatten = steps[(i + 1) % len(steps)]
        self.status.setText(f"each push also flattens spikes by {self.flatten:g}")
        if self.patch is not None:
            self._draw()

    def _even(self) -> None:
        """Slide the selection's vertices within the surface until their spacing evens out."""
        self._push(0.0, flatten=0.0, even=EVEN_SWEEPS)

    def _relax(self) -> None:
        """Take spikes out of the selection without pushing it in or out."""
        self._push(0.0, flatten=self.flatten or 0.5)

    def _push(self, sign: float, flatten: float | None = None, even: int = 0) -> None:
        """Move the selected, visible vertices of one surface along their normals, as a unit."""
        if self.patch is None:
            return
        hemi = self.patch.hemi
        seeds = self._vis[self._selected()[self._vis]]
        n = int(seeds.size)
        if not n:
            self.status.setText("nothing selected on screen")
            return
        step = self.session.state.surface_step
        layer = self.layer()
        try:
            self._dispatch(
                MoveSurfaceHighlight(
                    hemi,
                    self.surface,
                    sign * step,
                    SHOULDER_MM,
                    # The selection on screen: only these move, whatever else
                    # is selected off it.
                    encode_ids(seeds),
                    flatten=self.flatten if flatten is None else flatten,
                    even=even,
                    snap=0.0 if self.snap_mode == "hand" else 1.0,
                    gate=self.snap_mode == "snap",
                    snap_key="" if layer is None else layer.key,
                )
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        what = (
            "evened out"
            if even
            else "relaxed"
            if sign == 0
            else f"{'in' if sign < 0 else 'out'} {step:g} mm"
        )
        self.status.setText(f"{self.surface} {what}: {n} points ({self.snap_mode})")

    def _adjacency(self):
        assert self.patch is not None
        h = self.session.surfaces.hemis[self.patch.hemi]
        key = (self.patch.hemi, self.session.surfaces.topology_version)
        if key != self._adj_key:
            self._adj = adjacency(h.faces, h.n_vertices)
            self._adj_key = key
        return self._adj

    def _show_dots(self) -> None:
        canvas = self.canvas
        canvas.cluster_dots = np.zeros((0, 2))
        canvas.cluster_rgb = np.zeros((0, 3))
        canvas.legend = []
        if self.patch is None or not self._vis.size:
            canvas.dots = np.zeros((0, 2))
            canvas.update()
            return
        mask = self._preview if self._preview is not None else self._selected()
        canvas.dots = self._vis_px[mask[self._vis]]
        labels = self._cluster_labels() if self.show_clusters else None
        if labels is not None:
            colours = CLUSTER_RGB[np.round(np.linspace(0, MAX_CLUSTERS - 1, self.k)).astype(int)]
            canvas.cluster_dots = self._vis_px
            canvas.cluster_rgb = colours[labels]
            for j in range(self.k):
                members = labels == j
                mean = float(np.nanmean(self._label_values[members])) if members.any() else np.nan
                text = f"{(j + 1) % 10}  {int(members.sum()):>5}  {mean:>7.4g}"
                canvas.legend.append((tuple(colours[j]), text))
        canvas.update()

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
            h = surfaces.hemis[hemi]
            self._vis = visible_vertices(self.patch)
            self._vis_px = self.patch.to_pixels(self.patch.uv_of(h.n_vertices)[self._vis])
            self._vis_mask = np.zeros(h.n_vertices, bool)
            self._vis_mask[self._vis] = True
            self.canvas.brush_px = self.brush_mm / self.patch.mm_per_pixel
        self._draw()
        self._show_dots()

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
        fold = self._folding() if self.display == "folding" else None
        spacing = self._spacing() if self.display == "spacing" else None
        if fold is not None:
            # Interpolated through the same pixel weights, then two-toned:
            # gyri light, sulci dark, as FreeSurfer draws them.
            values = self._per_pixel(fold)
            values = np.where(np.isfinite(values), np.where(values > 0, 0.3, 0.75), np.nan)
            lo, hi = 0.0, 1.0
        elif spacing is not None:
            values = self._per_pixel(spacing)
            lo, hi = SPACING_LOG2
        else:
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
            + (
                "gyri / sulci"
                if fold is not None
                else "spacing (dark = crowded)"
                if spacing is not None
                else f"{layer.name if layer is not None else ''} · {mode}"
            )
            + f"   moves {self.surface} · {self.snap_mode} · flatten {self.flatten:g}"
            + ("" if p.source == "sphere" else "   (no sphere: inflated)")
        )
        self.canvas.update()

    def _per_pixel(self, per_vertex: np.ndarray) -> np.ndarray:
        """A per-vertex map through the patch's pixel weights; NaN outside the patch."""
        p = self.patch
        assert p is not None
        out = np.full(p.inside.shape, np.nan, np.float32)
        out[p.inside] = (per_vertex[p.corners[p.inside]] * p.weights[p.inside]).sum(axis=1)
        return out

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
