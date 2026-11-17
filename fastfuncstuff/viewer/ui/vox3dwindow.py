"""The 3-D voxel window: the overlay and the anatomy ray-marched, turnable.

What the slices cannot show is a structure that winds through them -- a vessel
tree, a network, a cluster's real shape. This window draws the volume itself:
a maximum intensity projection of the overlay over an X-ray of the anatomy by
default, or a MinIP, a mean, the first surface over threshold, or a composite.
Everything is read from the layer stack exactly as the slices read it -- the
topmost visible layer above the base is the overlay, with its colour map,
range, sign and threshold; the base is the anatomy -- so a threshold dragged in
the controller redraws the projection, and stepping time -- or the overlay's
own sub-bricks, InstaSpectra's frequency bins -- steps the 3-D view with it.

Rendering is Qt's RHI (``QRhiWidget``, Metal on a Mac), with the shaders in
``viewer/shaders/vox3d.*``; the arithmetic, and the CPU twin that picking uses,
is in :mod:`viewer.vox3d`. The box is drawn with every face and a depth test
that keeps the farthest fragment, so each pixel shades once at the box's back
face whatever the winding or the backend's Y convention, and the view still
works from inside the box.

The camera, spin, render mode and clip plane are view state like the surface
window's camera: not recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import (
    QRhiBuffer,
    QRhiCommandBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiSampler,
    QRhiShaderStage,
    QRhiTexture,
    QRhiTextureSubresourceUploadDescription,
    QRhiTextureUploadDescription,
    QRhiTextureUploadEntry,
    QRhiVertexInputAttribute,
    QRhiVertexInputBinding,
    QRhiVertexInputLayout,
    QRhiViewport,
)
from PySide6.QtGui import QRhiShaderResourceBinding as Binding

from fastfuncstuff.viewer import surface3d as s3
from fastfuncstuff.viewer import vox3d
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding as Key
from fastfuncstuff.viewer.ui.shortcuts import ShortcutHelp, jump_back_key, keep_keys_for_shortcuts
from fastfuncstuff.viewer.ui.surfacewindow import _qmatrix, _shader
from fastfuncstuff.viewer.viewports import Viewport

if TYPE_CHECKING:
    from fastfuncstuff.viewer.layers import Layer

_STAGE = Binding.StageFlag

#: Named views the number keys jump to. Rows are the camera's right, up and
#: toward-camera axes in scanner RAS (surface3d.VIEWS' convention).
VIEWS = {
    "axial": s3.VIEWS["top"],
    "coronal": s3.VIEWS["front"],
    "sagittal": s3.VIEWS["left lateral"],
}
CLIPS = ("off", "sagittal", "coronal", "axial")
#: Radians per spin tick, and the tick: a full turn in about eight seconds.
SPIN_STEP = 2.0 * np.pi / 240.0
SPIN_MS = 33


class _Marks(QtWidgets.QWidget):
    """The box's edges and the crosshair, painted over the render."""

    def __init__(self, canvas: Vox3DCanvas) -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        c = self.canvas
        dom = c.domain
        if dom is None:
            return
        w, h = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = theme.palette()
        if c.show_box:
            corners = vox3d.box_corners(dom.shape, dom.affine)
            px, front = c.camera.to_widget(corners, w, h)
            edge = QtGui.QColor(pal.faint)
            edge.setAlphaF(0.55)
            p.setPen(QtGui.QPen(edge, 1.0))
            for a, b in vox3d.BOX_EDGES:
                if front[a] and front[b]:
                    p.drawLine(QtCore.QPointF(*px[a]), QtCore.QPointF(*px[b]))
        if c.cross_mm is not None and c.show_cross:
            r, g, b = (int(255 * v) for v in pal.crosshair[:3])
            col = QtGui.QColor(r, g, b, 200)
            p.setPen(QtGui.QPen(col, 1.2))
            for a3, b3 in vox3d.crosshair_segments(c.cross_mm, dom.shape, dom.affine):
                ends, front = c.camera.to_widget(np.stack([a3, b3]), w, h)
                if front.all():
                    p.drawLine(QtCore.QPointF(*ends[0]), QtCore.QPointF(*ends[1]))
            dot, front = c.camera.to_widget(np.asarray(c.cross_mm)[None], w, h)
            if front[0]:
                p.setBrush(col)
                p.drawEllipse(QtCore.QPointF(*dot[0]), 3.0, 3.0)
        p.setPen(QtGui.QColor(pal.faint))
        p.drawText(QtCore.QPointF(8, h - 8), c.caption)
        p.end()


class Vox3DCanvas(QtWidgets.QRhiWidget):
    """The GPU side: holds CPU copies of the volumes and uploads them on render."""

    #: A click, in scanner mm: where the pixel's ray found what it shows.
    located = QtCore.Signal(float, float, float)

    ORBIT = 2.0 * np.pi

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSampleCount(1)
        self.camera = s3.Camera()
        self.camera.rotation = VIEWS["sagittal"].copy()
        self.mode = "mip"
        self.anatomy_opacity = 0.5
        self.samples_per_voxel = 1.5
        self.clip = "off"
        self.clip_flip = False
        self.show_box = True
        self.show_cross = True
        self.linear = True
        self.overlay: vox3d.VolumeSlot | None = None
        self.anatomy: vox3d.VolumeSlot | None = None
        self.cross_mm: np.ndarray | None = None
        self.caption = ""
        self._framed_key: tuple | None = None
        self._uploaded: dict[str, tuple | None] = {"ov": None, "an": None, "stat": None}
        self._tex: dict[str, QRhiTexture | None] = {"ov": None, "an": None, "stat": None}
        self._lut_bytes: bytes | None = None
        self._box_key: tuple | None = None
        self._rhi_ready = False
        self._pipeline = None
        self._srb = None
        self._rebind = True
        self._press: QtCore.QPointF | None = None
        self._last: QtCore.QPointF | None = None
        self._moved = False
        self._marks = _Marks(self)
        self._spin = QtCore.QTimer(self)
        self._spin.setInterval(SPIN_MS)
        self._spin.timeout.connect(self._spin_tick)

    # -- content ---------------------------------------------------------------
    @property
    def domain(self) -> vox3d.VolumeSlot | None:
        """The grid the box is drawn around: the overlay's, else the anatomy's."""
        return self.overlay if self.overlay is not None else self.anatomy

    def set_volumes(
        self, overlay: vox3d.VolumeSlot | None, anatomy: vox3d.VolumeSlot | None
    ) -> None:
        self.overlay, self.anatomy = overlay, anatomy
        dom = self.domain
        if dom is not None:
            # Frame once per grid, not on every refresh: a new frequency or
            # threshold must not throw away the view someone turned to.
            key = (dom.shape, np.asarray(dom.affine).tobytes())
            if key != self._framed_key:
                self.camera.frame(vox3d.box_corners(dom.shape, dom.affine))
                self._framed_key = key
        self.update()

    def reset_camera(self, view: str | None = None) -> None:
        if view is not None:
            self.camera.rotation = VIEWS[view].copy()
        dom = self.domain
        if dom is not None:
            self.camera.frame(vox3d.box_corners(dom.shape, dom.affine))
        self.update()

    def set_spinning(self, on: bool) -> None:
        if on:
            self._spin.start()
        else:
            self._spin.stop()

    @property
    def spinning(self) -> bool:
        return self._spin.isActive()

    def _spin_tick(self) -> None:
        # About the head's own superior axis, not the screen's: a brain seen
        # from any tilt turns like a brain on a turntable.
        self.camera.rotation = self.camera.rotation @ s3._rotation([0, 0, 1], -SPIN_STEP)
        self.update()

    def clip_plane(self) -> tuple[float, float, float, float]:
        """The world plane cutting the volume at the crosshair, or none."""
        if self.clip == "off" or self.cross_mm is None:
            return (0.0, 0.0, 0.0, 0.0)
        axis = CLIPS.index(self.clip) - 1
        n = np.zeros(3)
        n[axis] = -1.0 if self.clip_flip else 1.0
        return (float(n[0]), float(n[1]), float(n[2]), float(n @ self.cross_mm))

    # -- RHI ---------------------------------------------------------------------
    def initialize(self, cb: QRhiCommandBuffer) -> None:  # noqa: ARG002 (Qt)
        rhi = self.rhi()
        self._rhi_ptr = shiboken6.getCppPointer(rhi)[0]
        clamp = QRhiSampler.AddressMode.ClampToEdge
        self._nearest = rhi.newSampler(
            QRhiSampler.Filter.Nearest, QRhiSampler.Filter.Nearest,
            QRhiSampler.Filter.None_, clamp, clamp, clamp,
        )  # fmt: skip
        self._nearest.create()
        self._linear_s = rhi.newSampler(
            QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear,
            QRhiSampler.Filter.None_, clamp, clamp, clamp,
        )  # fmt: skip
        self._linear_s.create()
        self._empty = self._new_3d((1, 1, 1))
        self._empty_pending = True
        self._lut_tex = rhi.newTexture(QRhiTexture.Format.RGBA8, QtCore.QSize(256, 1))
        self._lut_tex.create()
        self._lut_bytes = None
        self._ubuf = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, vox3d.UNIFORM_BYTES
        )
        self._ubuf.create()
        self._vbuf = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.VertexBuffer, 36 * 12
        )
        self._vbuf.create()
        self._box_key = None
        self._uploaded = {"ov": None, "an": None, "stat": None}
        self._tex = {"ov": None, "an": None, "stat": None}
        self._srb = None
        self._pipeline = None
        self._rebind = True
        self._rhi_ready = True

    def _new_3d(self, shape: tuple[int, int, int]) -> QRhiTexture:
        nz, ny, nx = shape
        tex = self.rhi().newTexture(
            QRhiTexture.Format.R32F, nx, ny, nz, 1, QRhiTexture.Flag.ThreeDimensional
        )
        tex.create()
        return tex

    @staticmethod
    def _upload_3d(batch, tex: QRhiTexture, data: np.ndarray) -> None:
        entries = [
            QRhiTextureUploadEntry(
                z, 0, QRhiTextureSubresourceUploadDescription(QtCore.QByteArray(data[z].tobytes()))
            )
            for z in range(data.shape[0])
        ]
        desc = QRhiTextureUploadDescription()
        desc.setEntries(entries)
        batch.uploadTexture(tex, desc)

    def _sync_texture(self, batch, name: str, key: tuple | None, data: np.ndarray | None) -> None:
        if key == self._uploaded[name]:
            return
        old = self._tex[name]
        if old is not None:
            old.destroy()
        if data is None:
            self._tex[name] = None
        else:
            tex = self._new_3d(data.shape)
            self._upload_3d(batch, tex, data)
            self._tex[name] = tex
        self._uploaded[name] = key
        self._rebind = True

    def _apply_pending(self, batch) -> None:
        if self._empty_pending:
            self._upload_3d(batch, self._empty, np.zeros((1, 1, 1), np.float32))
            self._empty_pending = False
        ov, an = self.overlay, self.anatomy
        self._sync_texture(
            batch, "ov", None if ov is None else ov.key, None if ov is None else ov.data
        )
        self._sync_texture(
            batch, "an", None if an is None else an.key, None if an is None else an.data
        )
        stat = None if ov is None else ov.stat
        self._sync_texture(batch, "stat", None if stat is None or ov is None else ov.key, stat)
        lut = ov.lut if ov is not None and ov.lut is not None else np.zeros((256, 4), np.uint8)
        raw = np.ascontiguousarray(lut, np.uint8).tobytes()
        if raw != self._lut_bytes:
            img = QtGui.QImage(raw, 256, 1, 4 * 256, QtGui.QImage.Format.Format_RGBA8888)
            batch.uploadTexture(self._lut_tex, img.copy())
            self._lut_bytes = raw
        dom = self.domain
        if dom is not None:
            key = (dom.shape, np.asarray(dom.affine).tobytes())
            if key != self._box_key:
                batch.updateDynamicBuffer(
                    self._vbuf, 0, vox3d.box_vertices(dom.shape, dom.affine).tobytes()
                )
                self._box_key = key
        if self._rebind or self._srb is None:
            self._bindings()

    def _bindings(self) -> None:
        rhi = self.rhi()
        srb = self._srb or rhi.newShaderResourceBindings()
        frag = _STAGE.FragmentStage
        sampler = self._linear_s if self.linear else self._nearest
        srb.setBindings(
            [
                Binding.uniformBuffer(0, _STAGE.VertexStage | frag, self._ubuf),
                Binding.sampledTexture(1, frag, self._tex["ov"] or self._empty, sampler),
                Binding.sampledTexture(2, frag, self._tex["an"] or self._empty, sampler),
                Binding.sampledTexture(3, frag, self._lut_tex, self._nearest),
                Binding.sampledTexture(4, frag, self._tex["stat"] or self._empty, self._nearest),
            ]
        )
        srb.create()
        self._srb = srb
        self._rebind = False

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        ps = self.rhi().newGraphicsPipeline()
        ps.setShaderStages(
            [
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader("vox3d.vert")),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader("vox3d.frag")),
            ]
        )
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(12)])
        layout.setAttributes(
            [QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float3, 0)]
        )
        ps.setVertexInputLayout(layout)
        # Keep the farthest fragment: the back face, where the ray leaves.
        ps.setDepthTest(True)
        ps.setDepthWrite(True)
        ps.setDepthOp(QRhiGraphicsPipeline.CompareOp.Greater)
        ps.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        ps.setSampleCount(self.sampleCount())
        ps.setShaderResourceBindings(self._srb)
        ps.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        ps.create()
        self._pipeline = ps

    def render(self, cb: QRhiCommandBuffer) -> None:
        rhi = self.rhi()
        if shiboken6.getCppPointer(rhi)[0] != getattr(self, "_rhi_ptr", None):
            self.initialize(cb)
        batch = rhi.nextResourceUpdateBatch()
        self._apply_pending(batch)
        size = self.renderTarget().pixelSize()
        bg = QtGui.QColor(theme.palette().bg)
        dom = self.domain
        if dom is not None:
            aspect = size.width() / max(size.height(), 1)
            corr = _qmatrix(rhi.clipSpaceCorrMatrix())
            mvp = corr @ self.camera.projection(aspect) @ self.camera.view()
            eye = self._eye()
            batch.updateDynamicBuffer(
                self._ubuf,
                0,
                vox3d.pack_uniforms(
                    mvp,
                    eye,
                    dom,
                    overlay=self.overlay,
                    anatomy=self.anatomy,
                    mode=self.mode,
                    samples_per_voxel=self.samples_per_voxel,
                    anatomy_opacity=self.anatomy_opacity,
                    clip=self.clip_plane(),
                    background=(bg.redF(), bg.greenF(), bg.blueF()),
                ),
            )
        cb.beginPass(self.renderTarget(), bg, QRhiDepthStencilClearValue(0.0, 0), batch)
        if dom is not None:
            self._ensure_pipeline()
            cb.setGraphicsPipeline(self._pipeline)
            cb.setViewport(QRhiViewport(0, 0, size.width(), size.height()))
            cb.setShaderResources(self._srb)
            cb.setVertexInput(0, [(self._vbuf, 0)])
            cb.draw(36)
        cb.endPass()
        self._marks.update()

    def _eye(self) -> np.ndarray:
        """The camera's position in scanner mm."""
        v = self.camera.view()
        return -v[:3, :3].T @ v[:3, 3]

    def releaseResources(self) -> None:  # noqa: N802 (Qt)
        self._rhi_ready = False
        self._pipeline = None
        self._srb = None

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        self._marks.resize(event.size())
        super().resizeEvent(event)

    # -- input -----------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        self._press = self._last = event.position()
        self._moved = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._last is None:
            return
        d = event.position() - self._last
        self._last = event.position()
        if self._press is not None and (event.position() - self._press).manhattanLength() > 3:
            self._moved = True
        h = max(self.height(), 1)
        buttons = event.buttons()
        shift = event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        if buttons & QtCore.Qt.MouseButton.RightButton:
            self.camera.zoom(float(np.exp(d.y() / 150.0)))
        elif buttons & QtCore.Qt.MouseButton.MiddleButton or (
            buttons & QtCore.Qt.MouseButton.LeftButton and shift
        ):
            self.camera.pan(d.x() / h, d.y() / h)
        elif buttons & QtCore.Qt.MouseButton.LeftButton:
            self.camera.orbit(d.x() / h * self.ORBIT, d.y() / h * self.ORBIT)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and not self._moved
            and self._press is not None
        ):
            hit = self.pick(event.position())
            if hit is not None:
                self.located.emit(float(hit[0]), float(hit[1]), float(hit[2]))
        self._press = self._last = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta()
        steps = (delta.y() or delta.x()) / 120.0
        self.camera.zoom(0.9**steps)
        self.update()

    def ray_at(self, pos: QtCore.QPointF) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        x = 2.0 * pos.x() / w - 1.0
        y = 1.0 - 2.0 * pos.y() / h
        return self.camera.ray(x, y, w / h)

    def pick(self, pos: QtCore.QPointF) -> np.ndarray | None:
        """What the pixel shows, in mm: the CPU twin of the shader's ray."""
        origin, direction = self.ray_at(pos)
        ov, an = self.overlay, self.anatomy
        if ov is not None:
            volume = ov.data.transpose(2, 1, 0)
            stat = None if ov.stat is None else ov.stat.transpose(2, 1, 0)
            return vox3d.pick(
                origin,
                direction,
                volume=volume,
                affine=ov.affine,
                mode=self.mode,
                threshold=ov.threshold,
                sign_mode=ov.sign_mode,
                stat=stat,
                clip=self.clip_plane(),
            )
        if an is not None:
            # No overlay: the anatomy's surface, a quarter of the way up its range.
            return vox3d.pick(
                origin,
                direction,
                volume=an.data.transpose(2, 1, 0),
                affine=an.affine,
                mode="first hit",
                threshold=an.lo + 0.25 * (an.hi - an.lo),
                sign_mode=vox3d.SIGN_POS,
                clip=self.clip_plane(),
            )
        return None


class Vox3DWindow(QtWidgets.QWidget):
    """The window around the canvas: a control row, keys, and the stack's volumes."""

    closed = QtCore.Signal(str)
    #: A click in the volume, in scanner mm.
    located = QtCore.Signal(float, float, float)

    def __init__(
        self,
        vid: str,
        session,
        dispatch,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.vid = vid
        self.session = session
        self._dispatch = dispatch
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.resize(640, 600)
        self.setStyleSheet(theme.stylesheet())
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(list(vox3d.RENDER_MODES))
        self.mode_box.setToolTip(
            "mip: brightest along each ray (vessels, a band's map). minip: darkest. mean: "
            "X-ray. first hit: the surface of the thresholded overlay. composite: anatomy "
            "fog with the overlay as coloured glass.  (m cycles)"
        )
        self.mode_box.currentTextChanged.connect(self._set_mode)
        row.addWidget(self.mode_box)
        row.addWidget(QtWidgets.QLabel("anat"))
        self.anat = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.anat.setRange(0, 100)
        self.anat.setValue(50)
        self.anat.setMaximumWidth(110)
        self.anat.setToolTip("How strongly the anatomy is drawn behind the overlay  ([ / ])")
        self.anat.valueChanged.connect(self._set_anat)
        row.addWidget(self.anat)
        self.clip_box = QtWidgets.QComboBox()
        self.clip_box.addItems([f"clip {c}" for c in CLIPS])
        self.clip_box.setToolTip(
            "Cut the volume away on one side of the crosshair's plane  (c cycles, f flips)"
        )
        self.clip_box.currentIndexChanged.connect(self._set_clip)
        row.addWidget(self.clip_box)
        self.spin = QtWidgets.QPushButton("spin")
        self.spin.setCheckable(True)
        self.spin.setToolTip("Turn the head about its superior axis  (r)")
        self.spin.toggled.connect(self._set_spin)
        row.addWidget(self.spin)
        for text, view, key in (("A", "axial", "1"), ("C", "coronal", "2"), ("S", "sagittal", "3")):
            b = QtWidgets.QPushButton(text)
            b.setMaximumWidth(30)
            b.setToolTip(f"{view} view  ({key})")
            b.clicked.connect(lambda _=False, vw=view: self.canvas.reset_camera(vw))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.canvas = Vox3DCanvas(self)
        self.canvas.located.connect(self.located)
        v.addWidget(self.canvas, 1)

        self.help = ShortcutHelp(self, f"vox3d · {vid}")
        self.help.apply(
            [
                jump_back_key(self._dispatch),
                Key("m", "next render mode", lambda: self._step_mode(1), group="view"),
                Key("shift+m", "previous render mode", lambda: self._step_mode(-1), group="view"),
                Key("r", "spin on / off", self.spin.toggle, group="view"),
                Key("1", "axial view", lambda: self.canvas.reset_camera("axial"), group="view"),
                Key("2", "coronal view", lambda: self.canvas.reset_camera("coronal"), group="view"),
                Key(
                    "3", "sagittal view", lambda: self.canvas.reset_camera("sagittal"), group="view"
                ),
                Key("0", "re-frame the volume", lambda: self.canvas.reset_camera(), group="view"),
                Key("c", "cycle the clip plane", self._cycle_clip, group="view"),
                Key("f", "flip the clip side", self._flip_clip, group="view"),
                Key("[", "anatomy fainter", lambda: self._nudge_anat(-10), group="view"),
                Key("]", "anatomy stronger", lambda: self._nudge_anat(10), group="view"),
                Key("b", "box outline on / off", self._toggle_box, group="view"),
                Key("x", "crosshair on / off", self._toggle_cross, group="view"),
                Key("i", "interpolate / nearest voxels", self._toggle_linear, group="view"),
                Key("q", "quality: samples per voxel", self._cycle_quality, group="view"),
                Key("h", "this list", self.help.toggle, group="window"),
                Key("ctrl+w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- controls ------------------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        self.canvas.mode = mode
        self._caption()
        self.canvas.update()

    def _step_mode(self, step: int) -> None:
        n = len(vox3d.RENDER_MODES)
        self.mode_box.setCurrentIndex((self.mode_box.currentIndex() + step) % n)

    def _set_anat(self, value: int) -> None:
        self.canvas.anatomy_opacity = value / 100.0
        self.canvas.update()

    def _nudge_anat(self, delta: int) -> None:
        self.anat.setValue(max(0, min(100, self.anat.value() + delta)))

    def _set_clip(self, index: int) -> None:
        self.canvas.clip = CLIPS[index]
        self.canvas.update()

    def _cycle_clip(self) -> None:
        self.clip_box.setCurrentIndex((self.clip_box.currentIndex() + 1) % len(CLIPS))

    def _flip_clip(self) -> None:
        self.canvas.clip_flip = not self.canvas.clip_flip
        self.canvas.update()

    def _set_spin(self, on: bool) -> None:
        self.canvas.set_spinning(on)

    def _toggle_box(self) -> None:
        self.canvas.show_box = not self.canvas.show_box
        self.canvas.update()

    def _toggle_cross(self) -> None:
        self.canvas.show_cross = not self.canvas.show_cross
        self.canvas.update()

    def _toggle_linear(self) -> None:
        self.canvas.linear = not self.canvas.linear
        self.canvas._rebind = True
        self._caption()
        self.canvas.update()

    def _cycle_quality(self) -> None:
        steps = (0.75, 1.5, 3.0)
        now = self.canvas.samples_per_voxel
        self.canvas.samples_per_voxel = (
            steps[(steps.index(now) + 1) % len(steps)] if now in steps else 1.5
        )
        self._caption()
        self.canvas.update()

    # -- the session's side ------------------------------------------------------
    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        self.refresh(Aspect.ALL)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.canvas.update()

    def mark_stale(self) -> None:
        """Refreshed by the manager on every aspect it reads; never stale."""

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.canvas.set_spinning(False)
        self.closed.emit(self.vid)
        super().closeEvent(event)

    def chosen_layers(self) -> tuple[Layer | None, Layer | None]:
        """``(overlay, anatomy)``: the topmost drawable visible layer above the base, and the base."""
        layers = list(self.session.state.layers)
        if not layers:
            return None, None
        base = layers[0]
        anatomy = base if base.visible and not base.colour and not base.roi else None
        overlay = next(
            (
                ly
                for ly in reversed(layers[1:])
                if ly.visible
                and not ly.colour
                and not ly.roi
                and not ly.source.startswith("surface:")
            ),
            None,
        )
        return overlay, anatomy

    def refresh(self, dirty: Aspect) -> None:
        if dirty & (
            Aspect.LAYERS | Aspect.SLICES | Aspect.TIME | Aspect.COLORMAP | Aspect.THRESHOLD
        ):
            self._refresh_volumes()
        if dirty & (Aspect.CROSSHAIR | Aspect.LAYERS | Aspect.SLICES):
            self.canvas.cross_mm = (
                None
                if self.session.state.crosshair_mm is None
                else np.asarray(self.session.state.crosshair_mm, np.float64)
            )
        self._caption()
        self.canvas.update()

    def _slot(self, layer, cache: dict, *, overlay: bool) -> vox3d.VolumeSlot | None:
        import torch

        from fastfuncstuff.viewer.compose import cached_lut
        from fastfuncstuff.viewer.layers import SignMode

        st = self.session.state
        idx = int(st.time_index if layer.time_linked else layer.volume_index)
        idx = max(0, min(idx, layer.n_volumes - 1))
        thr = layer.threshold_index if overlay else None
        try:
            generation = self.session.store.get(layer.key).generation
        except KeyError:
            return None
        key = (
            layer.key,
            generation,
            idx,
            thr if thr != idx else None,
            np.asarray(layer.affine).tobytes(),
            layer.time_detrend,
            layer.time_psc,
        )
        data = self._cache.get(key)
        if data is None:
            try:
                value = s3.texture_data(self.session.shown_volume(layer.key, idx))
                stat = (
                    None
                    if thr is None or thr == idx
                    else s3.texture_data(self.session.volume(layer.key, thr))
                )
            except (KeyError, FileNotFoundError, ValueError):
                return None
            data = (value, stat)
        cache[key] = data
        value, stat = data
        lo = layer.range_lo
        hi = layer.range_hi
        if lo is None or hi is None:
            finite = value[value != 0]
            lo, hi = (
                (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                if finite.size
                else (0.0, 1.0)
            )
        if hi <= lo:
            hi = lo + 1.0
        signs = {
            SignMode.BOTH: vox3d.SIGN_BOTH,
            SignMode.POS: vox3d.SIGN_POS,
            SignMode.NEG: vox3d.SIGN_NEG,
        }
        lut = None
        if overlay:
            table = cached_lut(layer.colormap, torch.device("cpu"), reverse=layer.colormap_reversed)
            rgb = np.clip(np.asarray(table.cpu().numpy(), np.float64), 0, 1)
            if rgb.shape[0] != 256:
                grid = np.linspace(0, rgb.shape[0] - 1, 256)
                rgb = np.stack(
                    [np.interp(grid, np.arange(rgb.shape[0]), rgb[:, c]) for c in range(3)], 1
                )
            lut = np.concatenate([np.round(rgb * 255), np.full((256, 1), 255.0)], 1).astype(
                np.uint8
            )
        return vox3d.VolumeSlot(
            key=key,
            data=value,
            frame=vox3d.slot_frame(layer.shape, layer.affine),
            shape=tuple(int(s) for s in layer.shape),
            affine=np.asarray(layer.affine, np.float64),
            lo=float(lo),
            hi=float(hi),
            threshold=float(layer.threshold) if overlay else 0.0,
            opacity=float(layer.opacity),
            sign_mode=signs.get(layer.sign_mode, vox3d.SIGN_BOTH),
            lut=lut,
            stat=stat,
        )

    def _refresh_volumes(self) -> None:
        overlay, anatomy = self.chosen_layers()
        cache: dict = {}
        ov = None if overlay is None else self._slot(overlay, cache, overlay=True)
        an = None if anatomy is None else self._slot(anatomy, cache, overlay=False)
        # Only what is drawn stays cached: a dropped layer's voxels go with it.
        self._cache = cache
        # A sub-brick stack (a bucket, InstaSpectra's bins) says which brick is
        # drawn: the projection of 0.81 Hz and of 0.83 Hz look alike otherwise.
        self._brick = ""
        if overlay is not None and overlay.n_volumes > 1 and not overlay.time_linked:
            i = overlay.volume_index
            self._brick = f" [{overlay.labels[i] if i < len(overlay.labels) else f'#{i}'}]"
        self._names = (
            "" if overlay is None else overlay.name,
            "" if anatomy is None else anatomy.name,
        )
        self.canvas.set_volumes(ov, an)

    def _caption(self) -> None:
        ov, an = getattr(self, "_names", ("", ""))
        parts = [self.canvas.mode]
        if ov:
            o = self.canvas.overlay
            parts.append(
                f"{ov}{getattr(self, '_brick', '')}"
                + (f"  thr {o.threshold:.3g}" if o is not None else "")
            )
        if an:
            parts.append(f"over {an}")
        if self.canvas.clip != "off":
            parts.append(f"clip {self.canvas.clip}{' (flipped)' if self.canvas.clip_flip else ''}")
        parts.append(f"{self.canvas.samples_per_voxel:g} samples/voxel")
        if not self.canvas.linear:
            parts.append("nearest")
        self.canvas.caption = "  ·  ".join(parts)


__all__ = ["Vox3DCanvas", "Vox3DWindow"]
