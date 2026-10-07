"""The image pane: blit a rendered plane, draw the crosshair, take input.

The pane holds no viewer state. It is handed a :class:`PaneImage`, it reports
where the user clicked, and that is all -- every consequence goes back through
the command bus. That is what keeps a recorded session honest: there is no way
for a widget to change the view behind the recorder's back.

Painting stays cheap by converting to ``QImage`` only when the pixels change,
not on every expose event.
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.compose import PaneImage
from fastfuncstuff.viewer.slicing import plane_axes
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui import theme


def _segment_path(segments: np.ndarray) -> QtGui.QPainterPath:
    """``(N, 2, 2)`` (x, y) segments as one path of move-to/line-to pairs.

    Built by deserialising bytes rather than one Python call per segment. A
    slice crosses ~10k triangles: a ``QLineF`` per segment made object
    construction most of a surface-drag redraw, and a ``QPolygonF`` of point
    pairs was cheap to fill but ~4x slower to *draw* (the argument is
    converted back through Python). A path is cheapest on both counts.
    The layout is ``QDataStream``'s for ``QPainterPath``: element count,
    then (type, x, y) per element with 0 = move-to and 1 = line-to, then
    the fill rule.
    """
    pts = segments.reshape(-1, 2)
    n = pts.shape[0]
    rec = np.empty(n, dtype=[("type", ">i4"), ("x", ">f8"), ("y", ">f8")])
    rec["type"] = np.tile(np.array([0, 1], ">i4"), n // 2)
    rec["x"], rec["y"] = pts[:, 0], pts[:, 1]
    raw = np.array([n], ">i4").tobytes() + rec.tobytes() + np.array([0], ">i4").tobytes()
    path = QtGui.QPainterPath()
    stream = QtCore.QDataStream(QtCore.QByteArray(raw))
    stream >> path  # type: ignore[operator]
    return path


class ImagePane(QtWidgets.QWidget):
    """One display plane."""

    #: (row, col) in display-grid indices of the plane's two spanned axes.
    picked = QtCore.Signal(int, int)
    #: Wheel or key step through slices, in signed slice units.
    stepped = QtCore.Signal(int)
    #: Seed request (ctrl/cmd-click), same coordinates as ``picked``.
    seeded = QtCore.Signal(int, int)
    #: Middle-button (or shift+left) drag, in image pixels. Left stays the
    #: crosshair, because moving where you are looking is the gesture you
    #: make most.
    panned = QtCore.Signal(float, float)
    #: Right-button drag: multiply the zoom by this factor (up zooms in), the
    #: same gesture as in the surface window.
    zoomed = QtCore.Signal(float)
    #: Align-mode drags. ``slid`` is in image pixels (row, col); ``turned`` in
    #: degrees, positive clockwise on screen. ``released`` ends a drag.
    slid = QtCore.Signal(float, float)
    turned = QtCore.Signal(float)
    released = QtCore.Signal()
    #: Surface-editing gestures, in *fractional* image pixels: an outline is
    #: geometry, and snapping its grab to a voxel centre would put the drag
    #: up to half a voxel from where the hand is.
    edit_pressed = QtCore.Signal(float, float)
    edit_dragged = QtCore.Signal(float, float)
    edit_released = QtCore.Signal()

    #: The rotation ring's radius as a fraction of the drawn image's short side,
    #: and how close to it (widget pixels) a press counts as grabbing it.
    RING = 0.32
    GRAB = 9.0

    def __init__(self, plane: Plane, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self._image: QtGui.QImage | None = None
        self._pane: PaneImage | None = None
        self._cross: tuple[int, int] | None = None
        self._drag_from: QtCore.QPointF | None = None
        #: What the drag from ``_drag_from`` does: "pan" or "zoom".
        self._drag_kind = ""
        #: Voxel footprints of the open graphs, as (row, col, n_rows, n_cols)
        #: in image indices. The crosshair opens up around them.
        self._coverage: list[tuple[int, int, int, int]] = []
        self._labels: tuple[str, str, str, str] | None = None
        self._readout: list[str] = []
        self._zoomed = False
        #: Where the align handle sits, as fractional (row, col) image indices,
        #: or ``None`` outside align mode. The ring is drawn around it.
        self._handle: tuple[float, float] | None = None
        self._grab: str | None = None
        self._grab_at: QtCore.QPointF | None = None
        self._grab_angle = 0.0
        #: Surface outlines as (colour, lines) in image-pixel coordinates,
        #: built once per slice and scaled at paint time.
        #: Keyed by (hemi, surface) so a drag that moves one hemisphere
        #: rebuilds only its lines -- building QLineFs is most of the cost.
        self._outlines: dict[tuple[str, str], tuple[QtGui.QColor, QtGui.QPainterPath]] = {}
        #: Brush radius in image pixels while editing surfaces, else ``None``;
        #: drawn as a circle at the cursor so its reach is visible before a
        #: press commits to it.
        self._brush: float | None = None
        self._hover: QtCore.QPointF | None = None
        self._editing_drag = False
        self._brush_label = ""
        self._outline_width = 1.25
        self._toast = ""
        self._toast_alpha = 0.0
        self._toast_anim = QtCore.QVariantAnimation(self)
        self._toast_anim.setStartValue(1.0)
        self._toast_anim.setKeyValueAt(self.TOAST_HOLD / (self.TOAST_HOLD + self.TOAST_FADE), 1.0)
        self._toast_anim.setEndValue(0.0)
        self._toast_anim.setDuration(self.TOAST_HOLD + self.TOAST_FADE)
        self._toast_anim.valueChanged.connect(self._on_toast)
        #: A stroke being drawn, as fractional (row, col) image pixels.
        self._stroke_pts: list[tuple[float, float]] = []
        #: The selected vertex on this slice, per surface: (row, col, surface).
        self._marks: list[tuple[float, float, str]] = []
        self._highlight = np.zeros((0, 2))
        #: Replaces the plane/slice caption and drops the edge labels: for a
        #: small cell (a neighbour-strip slice) where those would cover the image.
        self.caption: str | None = None
        #: Left-drag keeps picking. Off for a cell that re-centres on the
        #: crosshair, where each pick would move the ground under the drag.
        self.drag_picks = True
        # Deliberately tiny. A pane's minimum is a floor under the whole
        # window, and a wall of small images is a real way to look at data.
        self.setMinimumSize(48, 48)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

    # -- content -------------------------------------------------------
    def set_pane(self, pane: PaneImage | None) -> None:
        """Install new pixels. Converts to QImage here, not in paintEvent."""
        self._pane = pane
        if pane is None:
            self._image = None
        else:
            arr = np.ascontiguousarray(pane.rgba.cpu().numpy())
            h, w = arr.shape[0], arr.shape[1]
            # Qt does not take ownership of the buffer, and a QImage over a
            # freed array paints garbage or crashes -- copy() detaches it.
            self._image = QtGui.QImage(
                arr.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888
            ).copy()
        self.update()

    @property
    def position(self) -> int | None:
        """Which slice is currently drawn, so a redraw can be skipped."""
        return None if self._pane is None else self._pane.position

    def set_layout(self, layout) -> None:
        """Take the plane's anatomical edge labels (top, right, bottom, left)."""
        self._labels = layout.labels
        self.update()

    def set_tilt(self, degrees: float) -> None:
        """How far an oblique window's slice is tilted, for the caption; 0 = not."""
        if degrees != getattr(self, "_tilt", 0.0):
            self._tilt = float(degrees)
            self.update()

    def set_zoomed(self, on: bool) -> None:
        """Whether the pane is showing a crop, for the corner readout."""
        if on != self._zoomed:
            self._zoomed = bool(on)
            self.update()

    def set_readout(self, lines: list[str]) -> None:
        """The overlay value(s) under the crosshair, drawn in the upper right."""
        if lines != self._readout:
            self._readout = list(lines)
            self.update()

    def set_crosshair(self, row: int, col: int) -> None:
        self._cross = (int(row), int(col))
        self.update()

    def set_outline_width(self, width: float) -> None:
        if width != self._outline_width:
            self._outline_width = float(width)
            self.update()

    def set_outlines(self, outlines, only: set[tuple[str, str]] | None = None) -> None:
        """Surface/slice crossings, as :class:`viewer.surfaces.Outline` records.

        With ``only``, just those (hemi, surface) keys are replaced and every
        other outline is kept as built.
        """
        built: dict[tuple[str, str], tuple[QtGui.QColor, QtGui.QPainterPath]] = {}
        for o in outlines:
            # (row, col) -> (x, y) = (col, row), pixel centres at +0.5.
            built[(o.hemi, o.surface)] = (
                QtGui.QColor.fromRgbF(*o.rgb),
                _segment_path(o.segments[..., ::-1] + 0.5),
            )
        if only is None:
            changed = bool(built or self._outlines)
            self._outlines = built
        else:
            for key in only:
                self._outlines.pop(key, None)
            self._outlines.update(built)
            changed = True
        if changed:
            self.update()

    def set_highlight(self, points: np.ndarray) -> None:
        """Highlighted vertices near this slice, ``(N, 2)`` fractional (row, col) pixels."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if points.shape != self._highlight.shape or not np.array_equal(points, self._highlight):
            self._highlight = points
            self.update()

    def set_marks(self, marks: list[tuple[float, float, str]]) -> None:
        if marks != self._marks:
            self._marks = list(marks)
            self.update()

    def set_stroke(self, points: list[tuple[float, float]]) -> None:
        """The line being drawn, in fractional image pixels; empty clears it."""
        self._stroke_pts = list(points)
        self.update()

    def set_brush(self, radius_px: float | None, label: str = "") -> None:
        """Enter (radius in image pixels) or leave (``None``) surface editing."""
        if radius_px != self._brush or label != self._brush_label:
            self._brush = radius_px
            self._brush_label = label
            if radius_px is None:
                self._editing_drag = False
            self.update()

    def _to_fraction(self, pos: QtCore.QPointF) -> tuple[float, float] | None:
        """Widget point to fractional (row, col), pixel ``r`` centred at ``r + 0.5``."""
        rect = self._target_rect()
        if self._image is None or rect.width() == 0 or rect.height() == 0:
            return None
        col = (pos.x() - rect.x()) / rect.width() * self._image.width() - 0.5
        row = (pos.y() - rect.y()) / rect.height() * self._image.height() - 0.5
        return (row, col)

    def set_handle(self, where: tuple[float, float] | None) -> None:
        """Show the align ring around an image point, or hide it (``None``)."""
        where = None if where is None else (float(where[0]), float(where[1]))
        if where != self._handle:
            self._handle = where
            self.update()

    def set_coverage(self, boxes: list[tuple[int, int, int, int]]) -> None:
        """Say which voxels the open graphs are reading, in image indices.

        Drawn as the crosshair's own gap rather than as a separate annotation:
        the gap already exists to keep the voxel under inspection visible, and
        a graph makes that "the voxels under inspection". Sizing it to the
        actual footprint is the difference between knowing the grid is 5x5 and
        seeing which 25 voxels that is.
        """
        boxes = [tuple(int(v) for v in b) for b in boxes]  # type: ignore[misc]
        if boxes != self._coverage:
            self._coverage = boxes  # type: ignore[assignment]
            self.update()

    # -- geometry ------------------------------------------------------
    def _target_rect(self) -> QtCore.QRect:
        """Where the image lands, letterboxed to preserve voxel aspect."""
        if self._image is None:
            return QtCore.QRect()
        iw, ih = self._image.width(), self._image.height()
        if iw == 0 or ih == 0:
            return QtCore.QRect()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = max(1, int(iw * scale)), max(1, int(ih * scale))
        return QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _image_scale(self) -> float:
        """Widget pixels per image pixel, for turning a drag into voxels."""
        rect = self._target_rect()
        if self._image is None or self._image.width() == 0 or rect.width() == 0:
            return 1.0
        return max(rect.width() / self._image.width(), 1e-6)

    def _to_indices(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        """Widget point to (row, col) display indices, or None if outside."""
        rect = self._target_rect()
        if self._image is None or not rect.contains(pos.toPoint()):
            return None
        fx = (pos.x() - rect.x()) / rect.width()
        fy = (pos.y() - rect.y()) / rect.height()
        # The image is (H=rows, W=cols); rows run down the widget.
        col = int(fx * self._image.width())
        row = int(fy * self._image.height())
        col = max(0, min(col, self._image.width() - 1))
        row = max(0, min(row, self._image.height() - 1))
        return (row, col)

    # -- painting ------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self._image is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self.plane.value.upper()}\nno data" if self.caption is None else self.caption,
            )
            if self._toast and self._toast_alpha > 0:
                self._paint_toast(p)
            p.end()
            return

        rect = self._target_rect()
        # Nearest-neighbour: a viewer must not invent voxels that are not there.
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(rect, self._image)
        if self._outlines:
            self._paint_outlines(p, rect)

        if self._cross is not None:
            self._paint_crosshair(p, rect)
        if self._brush is not None and self._hover is not None:
            self._paint_brush(p)
        if len(self._stroke_pts) > 1:
            self._paint_stroke(p, rect)
        if self._highlight.size:
            self._paint_highlight(p, rect)
        if self._marks:
            self._paint_marks(p, rect)
        if self._handle is not None:
            self._paint_handle(p, rect)

        p.setPen(QtGui.QColor.fromRgbF(*c.label))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        pos = self._pane.position if self._pane is not None else 0
        # Saying so on the image, because a cropped brain still looks like a
        # brain -- the same reason the edge labels are written on.
        zoom = "  zoom" if self._zoomed else ""
        tilt = getattr(self, "_tilt", 0.0)
        # Said on the image: a tilted slice still looks like a slice.
        tilted = f"  tilt {tilt:.0f}°" if tilt >= 0.5 else ""
        if self.caption is not None:
            font.setPointSize(8)
            p.setFont(font)
            p.drawText(4, 12, self.caption)
        else:
            p.drawText(6, 15, f"{self.plane.value.upper()}  {pos}{zoom}{tilted}")
        if self._brush is not None and self._brush_label:
            p.drawText(6, 30, self._brush_label)

        # Anatomical edge labels. An upside-down or mirrored brain still looks
        # like a brain, so the only thing that says which way round it is, is
        # writing it on the edges.
        if self._labels is not None and self.caption is None:
            p.setPen(QtGui.QColor(c.edge_label))
            top, right, bottom, left = self._labels
            r = self.rect()
            flags = QtCore.Qt.AlignmentFlag
            p.drawText(r.adjusted(0, 2, 0, 0), flags.AlignTop | flags.AlignHCenter, top)
            p.drawText(r.adjusted(0, 0, -4, 0), flags.AlignRight | flags.AlignVCenter, right)
            p.drawText(r.adjusted(0, 0, 0, -2), flags.AlignBottom | flags.AlignHCenter, bottom)
            p.drawText(r.adjusted(4, 0, 0, 0), flags.AlignLeft | flags.AlignVCenter, left)
        if self._readout:
            self._paint_readout(p)
        if self._toast and self._toast_alpha > 0:
            self._paint_toast(p)
        p.end()

    #: How long a toast stays fully visible, then how long it fades, ms.
    TOAST_HOLD = 3000
    TOAST_FADE = 900

    def show_toast(self, text: str) -> None:
        """A message in a solid box over the image, fading after a few seconds.

        For what an edit refused and why. Written into the caption line it was
        small, unboxed text over a brain and went unread.
        """
        self._toast = text
        self._toast_alpha = 1.0
        self._toast_anim.stop()
        self._toast_anim.start()
        self.update()

    def _on_toast(self, value) -> None:
        self._toast_alpha = float(value)
        if self._toast_alpha <= 0:
            self._toast = ""
        self.update()

    def _paint_toast(self, p: QtGui.QPainter) -> None:
        c = theme.palette()
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setOpacity(self._toast_alpha)
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        pad = 8
        width = max(self.width() - 4 * pad, 40)
        flags = QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap
        text = p.fontMetrics().boundingRect(QtCore.QRect(0, 0, width, 1000), flags, self._toast)
        box = QtCore.QRectF(
            (self.width() - text.width()) / 2 - pad,
            self.height() - text.height() - 3 * pad,
            text.width() + 2 * pad,
            text.height() + 2 * pad,
        )
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(c.warn))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QtGui.QColor(c.bg))
        p.drawText(box, flags, self._toast)
        p.restore()

    def _paint_outlines(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        assert self._image is not None
        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.translate(rect.x(), rect.y())
        p.scale(rect.width() / self._image.width(), rect.height() / self._image.height())
        for colour, path in self._outlines.values():
            pen = QtGui.QPen(colour)
            # Cosmetic: a screen-pixel width however far the slice is
            # magnified, so zooming in to judge a boundary makes the line
            # relatively thinner rather than hiding the edge under it.
            pen.setCosmetic(True)
            pen.setWidthF(self._outline_width)
            p.setPen(pen)
            p.drawPath(path)
        p.restore()

    def _paint_highlight(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        """Highlighted vertices: small dots in the warn colour, as on the 3-D surface."""
        assert self._image is not None
        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()
        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(theme.palette().warn))
        for row, col in self._highlight:
            p.drawEllipse(
                QtCore.QPointF(rect.x() + (col + 0.5) * sx, rect.y() + (row + 0.5) * sy), 2.2, 2.2
            )
        p.restore()

    def _paint_marks(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        """The selected vertex: a ring in its surface's outline colour."""
        from fastfuncstuff.viewer.surfaces import OUTLINE_RGB

        assert self._image is not None
        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for row, col, surface in self._marks:
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*OUTLINE_RGB.get(surface, (1, 1, 1))))
            pen.setWidthF(2.0)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            centre = QtCore.QPointF(rect.x() + (col + 0.5) * sx, rect.y() + (row + 0.5) * sy)
            p.drawEllipse(centre, 6.0, 6.0)
        p.restore()

    def _paint_stroke(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        assert self._image is not None
        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()
        poly = QtGui.QPolygonF(
            [
                QtCore.QPointF(rect.x() + (c + 0.5) * sx, rect.y() + (r + 0.5) * sy)
                for r, c in self._stroke_pts
            ]
        )
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(theme.palette().accent))
        pen.setWidthF(2.0)
        p.setPen(pen)
        p.drawPolyline(poly)
        p.restore()

    def _paint_brush(self, p: QtGui.QPainter) -> None:
        assert self._brush is not None and self._hover is not None
        r = self._brush * self._image_scale()
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor.fromRgbF(*theme.palette().crosshair, 0.9))
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawEllipse(self._hover, r, r)
        p.restore()

    def _paint_readout(self, p: QtGui.QPainter) -> None:
        """Values in the corner, on a translucent plate so they read over the brain."""
        c = theme.palette()
        font = QtGui.QFont(p.font())
        font.setFamily(theme.MONO)
        font.setPointSize(9)
        p.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        pad, line_h = 4, metrics.height()
        width = max(metrics.horizontalAdvance(line) for line in self._readout) + 2 * pad
        # Never wider than the pane: a narrow tile keeps the start of each line.
        width = min(width, self.width() - 8)
        height = line_h * len(self._readout) + 2 * pad
        plate = QtCore.QRect(self.width() - width - 4, 4, width, height)
        ground = QtGui.QColor(c.bg)
        ground.setAlphaF(0.72)
        p.fillRect(plate, ground)
        p.setPen(QtGui.QColor(c.text))
        for n, line in enumerate(self._readout):
            row = QtCore.QRect(
                plate.x() + pad, plate.y() + pad + n * line_h, width - 2 * pad, line_h
            )
            text = metrics.elidedText(line, QtCore.Qt.TextElideMode.ElideRight, row.width())
            p.drawText(
                row, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, text
            )

    def _handle_geometry(self) -> tuple[QtCore.QPointF, float] | None:
        """The handle's centre and the ring's radius, in widget pixels."""
        rect = self._target_rect()
        if self._handle is None or self._image is None or rect.isEmpty():
            return None
        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()
        row, col = self._handle
        centre = QtCore.QPointF(rect.x() + (col + 0.5) * sx, rect.y() + (row + 0.5) * sy)
        return centre, self.RING * min(rect.width(), rect.height())

    def _paint_handle(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        geometry = self._handle_geometry()
        if geometry is None:
            return
        centre, radius = geometry
        colour = QtGui.QColor.fromRgbF(0.35, 0.85, 1.0, 0.9)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        ring = QtGui.QPen(colour)
        ring.setWidthF(3.0 if self._grab == "turn" else 1.5)
        p.setPen(ring)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(centre, radius, radius)
        # Ticks on the ring, so a turn is visible as the ring's own motion.
        for k in range(4):
            angle = k * np.pi / 2 + np.pi / 4
            inner = centre + QtCore.QPointF(np.cos(angle), np.sin(angle)) * (radius - 5)
            outer = centre + QtCore.QPointF(np.cos(angle), np.sin(angle)) * (radius + 5)
            p.drawLine(QtCore.QLineF(inner, outer))
        dot = QtGui.QPen(colour)
        dot.setWidthF(3.0 if self._grab == "slide" else 1.5)
        p.setPen(dot)
        p.drawEllipse(centre, 6.0, 6.0)
        p.drawLine(QtCore.QLineF(centre.x() - 3, centre.y(), centre.x() + 3, centre.y()))
        p.drawLine(QtCore.QLineF(centre.x(), centre.y() - 3, centre.x(), centre.y() + 3))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

    def _grab_kind(self, pos: QtCore.QPointF, shift: bool) -> str | None:
        """What a left press here takes hold of: the ring, the handle, or nothing."""
        geometry = self._handle_geometry()
        if geometry is None:
            return None
        centre, radius = geometry
        distance = float(np.hypot(pos.x() - centre.x(), pos.y() - centre.y()))
        if abs(distance - radius) <= self.GRAB:
            return "turn"
        if distance <= self.GRAB + 4 or shift:
            return "slide"
        return None

    def _angle(self, pos: QtCore.QPointF) -> float:
        geometry = self._handle_geometry()
        if geometry is None:
            return 0.0
        centre, _ = geometry
        return float(np.degrees(np.arctan2(pos.y() - centre.y(), pos.x() - centre.x())))

    def _paint_crosshair(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        assert self._cross is not None and self._image is not None
        row, col = self._cross
        iw, ih = self._image.width(), self._image.height()
        sx, sy = rect.width() / iw, rect.height() / ih
        x = rect.x() + (col + 0.5) * sx
        y = rect.y() + (row + 0.5) * sy

        cross = theme.palette().crosshair
        colour = QtGui.QColor.fromRgbF(*cross, 0.85)
        pen = QtGui.QPen(colour)
        pen.setWidth(1)
        p.setPen(pen)

        # Each footprint drawn, and the largest sets the gap. Two graphs at
        # different sizes read as nested squares, which is what they are.
        gap_x = gap_y = 5.0
        faint = QtGui.QColor.fromRgbF(*cross, 0.55)
        for brow, bcol, nrows, ncols in self._coverage:
            bx, by = rect.x() + bcol * sx, rect.y() + brow * sy
            box = QtCore.QRectF(bx, by, ncols * sx, nrows * sy)
            p.setPen(QtGui.QPen(faint))
            p.drawRect(box)
            gap_x = max(gap_x, max(x - box.left(), box.right() - x))
            gap_y = max(gap_y, max(y - box.top(), box.bottom() - y))
        p.setPen(pen)

        # A gap at the centre so the voxel -- or the block of voxels a graph is
        # reading -- stays visible. AFNI's xhair gap, and the reason it exists.
        p.drawLine(QtCore.QLineF(rect.left(), y, x - gap_x, y))
        p.drawLine(QtCore.QLineF(x + gap_x, y, rect.right(), y))
        p.drawLine(QtCore.QLineF(x, rect.top(), x, y - gap_y))
        p.drawLine(QtCore.QLineF(x, y + gap_y, x, rect.bottom()))

    # -- input ---------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            # macOS turns a physical ctrl+click into a right-button press (and
            # reports ctrl as Meta), so the gesture the status line asks for
            # arrived here as the start of a pan and never set a seed.
            if (
                sys.platform == "darwin"
                and event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
            ):
                idx = self._to_indices(event.position())
                if idx is not None:
                    self.seeded.emit(*idx)
                return
            self._drag_from = event.position()
            self._drag_kind = "zoom"
            return
        mods = event.modifiers()
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        if event.button() == QtCore.Qt.MouseButton.MiddleButton or (
            # Shift+drag pans, except on the align handle, where shift slides.
            left and shift and self._grab_kind(event.position(), True) is None
        ):
            self._drag_from = event.position()
            self._drag_kind = "pan"
            return
        if self._brush is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            frac = self._to_fraction(event.position())
            if frac is not None:
                self._editing_drag = True
                self.edit_pressed.emit(*frac)
            return
        if left:
            kind = self._grab_kind(event.position(), shift)
            if kind is not None:
                self._grab = kind
                self._grab_at = event.position()
                self._grab_angle = self._angle(event.position())
                self.update()
                return
        idx = self._to_indices(event.position())
        if idx is None:
            return
        if mods & (
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.seeded.emit(*idx)
        else:
            self.picked.emit(*idx)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._drag_from is not None:
            delta = event.position() - self._drag_from
            self._drag_from = event.position()
            if self._drag_kind == "zoom":
                self.zoomed.emit(float(np.exp(-delta.y() / 150.0)))
            else:
                scale = self._image_scale()
                # Negated: dragging the image right should bring what is on the
                # left into view, the way dragging a map works.
                self.panned.emit(-delta.y() / scale, -delta.x() / scale)
            return
        if self._brush is not None:
            self._hover = event.position()
            self.update()
        if not (event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return
        if self._editing_drag:
            frac = self._to_fraction(event.position())
            if frac is not None:
                self.edit_dragged.emit(*frac)
            return
        if self._grab is not None and self._grab_at is not None:
            if self._grab == "slide":
                scale = self._image_scale()
                delta = event.position() - self._grab_at
                self._grab_at = event.position()
                self.slid.emit(delta.y() / scale, delta.x() / scale)
            else:
                angle = self._angle(event.position())
                step = (angle - self._grab_angle + 180.0) % 360.0 - 180.0
                self._grab_angle = angle
                if step:
                    self.turned.emit(step)
            return
        idx = self._to_indices(event.position())
        if idx is not None and self.drag_picks:
            self.picked.emit(*idx)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._drag_from is not None:
            self._drag_from = None
            self._drag_kind = ""
            return
        if self._editing_drag and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._editing_drag = False
            self.edit_released.emit()
            return
        if self._grab is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._grab = None
            self._grab_at = None
            self.update()
            self.released.emit()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt)
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta().y()
        if delta:
            self.stepped.emit(1 if delta > 0 else -1)

    def axes(self) -> tuple[int, int, int]:
        return plane_axes(self.plane)
