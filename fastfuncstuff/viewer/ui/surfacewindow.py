"""The 3-D surface window: hemispheres as meshes, data sampled between white and pial.

Rendering is Qt's own RHI through ``QRhiWidget`` -- Vulkan or OpenGL here,
Metal on a Mac, D3D on Windows -- with shaders precompiled by ``pyside6-qsb``
into ``viewer/shaders/*.qsb``. So no extra GPU library: the seam the viewer
architecture reserved for wgpu turned out to be served by what PySide6 ships.

Two PySide6 traps, both silent:

* ``QRhiResourceUpdateBatch.uploadStaticBuffer(buf, QByteArray)`` uploads
  nothing; the ``(buf, offset, bytes)`` overload works.
* ``QRhi.beginOffscreenFrame`` returns its command buffer through a C++
  out-parameter that never reaches Python, so a standalone offscreen QRhi is
  unusable; ``QRhiWidget.grabFramebuffer`` renders without the widget shown.

And one Qt one: a hidden ``QRhiWidget`` makes a fresh ``QRhi`` for every
``grabFramebuffer`` but calls ``initialize`` only the first time, so the second
grab rendered with textures owned by a destroyed QRhi. :meth:`render` checks
the QRhi it is handed and re-initialises when it changed.

Everything not GPU-specific -- shapes, layout, camera, picking, the uniform
block, the colouring arithmetic -- is in :mod:`viewer.surface3d`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources

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
    QShader,
)
from PySide6.QtGui import QRhiShaderResourceBinding as Binding

from fastfuncstuff.io.freesurfer import available_annotations, available_volume_atlases
from fastfuncstuff.surface.mesh import MeshTopology, vertex_areas, vertex_normals
from fastfuncstuff.viewer import surface3d as s3
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding as Key
from fastfuncstuff.viewer.ui.shortcuts import ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    SetAtlas,
    SetSurfaceDepth,
    SetSurfaceEquivolume,
    SetSurfaceFolding,
    SetSurfaceHemis,
    SetSurfaceMap,
    SetSurfaceShape,
)

_STAGE = Binding.StageFlag
_ATTRS = ("posA", "posB", "nrmA", "nrmB", "white", "pial", "curv", "vcolor", "areas")
#: Bytes per vertex and vertex format of each attribute; Float3 unless listed.
_STRIDE = {"curv": 4, "vcolor": 4, "areas": 8}


def _shader(name: str) -> QShader:
    data = resources.files("fastfuncstuff.viewer.shaders").joinpath(f"{name}.qsb").read_bytes()
    return QShader.fromSerialized(QtCore.QByteArray(data))


def _qmatrix(m: QtGui.QMatrix4x4) -> np.ndarray:
    # data() is column-major.
    return np.array(m.data(), np.float64).reshape(4, 4).T


@dataclass
class OverlaySlot:
    """One overlay layer as the canvas needs it.

    ``key`` identifies the voxels: the textures re-upload only when it changes,
    so a threshold drag or a colormap change touches uniforms and the LUT, not
    a 100 MB volume.
    """

    key: tuple
    value: np.ndarray  # (nz, ny, nx) float32, see surface3d.texture_data
    value_frame: np.ndarray  # mm -> texture
    stat: np.ndarray | None
    stat_frame: np.ndarray | None
    shade: s3.ShadeParams
    lut: np.ndarray | None = None  # (256, 4) uint8
    palette: np.ndarray | None = None  # (rows, PALETTE_W, 4) uint8


class _HemiGPU:
    """One hemisphere's buffers and bindings."""

    def __init__(self) -> None:
        self.buffers: dict[str, QRhiBuffer] = {}
        self.index_full: QRhiBuffer | None = None
        self.index_flat: QRhiBuffer | None = None
        self.n_full = 0
        self.n_flat = 0
        self.ubuf: QRhiBuffer | None = None
        self.srb = None


class SurfaceCanvas(QtWidgets.QRhiWidget):
    """The GPU side. Holds CPU copies of what it uploads, and uploads on render."""

    #: A click on the surface, in scanner mm.
    located = QtCore.Signal(float, float, float)
    #: Shift+wheel: move the sampled depth by this fraction.
    depth_scrolled = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSampleCount(4)
        self.camera = s3.Camera()
        self.morph = 1.0
        self.depth: tuple[float, float] = (0.5, 0.5)
        self.samples = 1
        #: Folding shade strength under the overlays.
        self.fold_contrast = 0.16
        self.equivolume = True
        self.map_opacity = 0.85
        #: Overlay layers, bottom to top (at most MAX_LAYERS are drawn).
        self.overlays: list[OverlaySlot] = []
        self.cross: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.linear = False
        #: Per hemisphere: arrays waiting to upload, and what is drawn.
        self._pending: dict[str, dict[str, np.ndarray]] = {}
        self._faces: dict[str, dict[str, np.ndarray]] = {}
        self._flat: dict[str, bool] = {}
        self._visible: set[str] = set()
        self._model: dict[str, np.ndarray] = {}
        #: Each hemisphere's own turn about its centre (alt+drag), on top of
        #: the layout. View state like the camera, so not recorded.
        self.hemi_rotation: dict[str, np.ndarray] = {}
        self._grabbed: str | None = None
        #: CPU copies for picking: drawn positions A/B, white, pial, faces.
        self._cpu: dict[str, dict[str, np.ndarray]] = {}
        #: Per drawn slot: the key its textures hold, and those textures.
        self._uploaded: list[tuple | None] = [None] * s3.MAX_LAYERS
        self._slot_tex: list[tuple[QRhiTexture, QRhiTexture] | None] = [None] * s3.MAX_LAYERS
        self._lut_bytes: bytes | None = None
        self._palette_bytes: bytes | None = None
        self._palette_rows: list[int] = [0] * s3.MAX_LAYERS
        self._rebind = True
        self._gpu: dict[str, _HemiGPU] = {}
        self._rhi_ready = False
        self._press: QtCore.QPointF | None = None
        self._last: QtCore.QPointF | None = None
        self._moved = False
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(450)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_morph)

    # -- content -----------------------------------------------------------
    def set_hemisphere(
        self,
        hemi: str,
        *,
        pos_a: np.ndarray | None,
        pos_b: np.ndarray | None,
        nrm_a: np.ndarray | None,
        nrm_b: np.ndarray | None,
        white: np.ndarray | None = None,
        pial: np.ndarray | None = None,
        curv: np.ndarray | None = None,
        vcolor: np.ndarray | None = None,
        areas: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        flat_faces: np.ndarray | None = None,
    ) -> None:
        """Queue a hemisphere's arrays; only those given are re-uploaded."""
        given = {
            "posA": pos_a,
            "posB": pos_b,
            "nrmA": nrm_a,
            "nrmB": nrm_b,
            "white": white,
            "pial": pial,
            "curv": curv,
            "vcolor": vcolor,
            "areas": areas,
        }
        pend = self._pending.setdefault(hemi, {})
        cpu = self._cpu.setdefault(hemi, {})
        for k, v in given.items():
            if v is not None:
                kind = np.uint8 if k == "vcolor" else np.float32
                arr = np.ascontiguousarray(v, dtype=kind)
                pend[k] = arr
                cpu[k] = arr
        if faces is not None:
            self._faces.setdefault(hemi, {})["full"] = np.ascontiguousarray(faces, np.uint32)
            pend["_faces"] = self._faces[hemi]["full"]
        if flat_faces is not None:
            self._faces.setdefault(hemi, {})["flat"] = np.ascontiguousarray(flat_faces, np.uint32)
            pend["_flat"] = self._faces[hemi]["flat"]
        self.update()

    def drop_hemispheres(self) -> None:
        self._pending.clear()
        self._cpu.clear()
        self._faces.clear()
        for gpu in self._gpu.values():
            for b in gpu.buffers.values():
                b.destroy()
        self._gpu.clear()
        self.update()

    def set_layout(
        self, visible: set[str], models: dict[str, np.ndarray], flat: dict[str, bool]
    ) -> None:
        self._visible = set(visible)
        self._model = {h: np.asarray(m, np.float64) for h, m in models.items()}
        self._flat = dict(flat)
        self.update()

    def model(self, hemi: str) -> np.ndarray:
        """Layout (split) composed with the hemisphere's own turn about its centre."""
        base = self._model.get(hemi, np.eye(4))
        r = self.hemi_rotation.get(hemi)
        now = self.current(hemi)
        if r is None or now is None:
            return base
        c = now[0].mean(axis=0).astype(np.float64)
        turn = np.eye(4)
        turn[:3, :3] = r
        turn[:3, 3] = c - r @ c
        return base @ turn

    def set_overlays(self, overlays: list[OverlaySlot]) -> None:
        """The overlay stack, bottom to top. Textures follow on the next frame."""
        self.overlays = list(overlays)
        self.update()

    def _drawn(self) -> list[OverlaySlot]:
        return self.overlays[-s3.MAX_LAYERS :]

    def animate_to(self) -> None:
        """Run the morph from shape A (0) to shape B (1)."""
        self._anim.stop()
        self.morph = 0.0
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _on_morph(self, value) -> None:
        self.morph = float(value)
        self.update()

    def current(self, hemi: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Positions and normals as drawn right now (mid-morph included), before layout."""
        cpu = self._cpu.get(hemi)
        if not cpu or not all(k in cpu for k in ("posA", "posB", "nrmA", "nrmB")):
            return None
        m = self.morph
        return (1 - m) * cpu["posA"] + m * cpu["posB"], (1 - m) * cpu["nrmA"] + m * cpu["nrmB"]

    def drawn_positions(self, hemi: str) -> np.ndarray | None:
        cpu = self._cpu.get(hemi)
        if not cpu or "posA" not in cpu or "posB" not in cpu:
            return None
        p = (1.0 - self.morph) * cpu["posA"] + self.morph * cpu["posB"]
        m = self.model(hemi)
        return p @ m[:3, :3].T.astype(np.float32) + m[:3, 3].astype(np.float32)

    # -- RHI -----------------------------------------------------------------
    def initialize(self, cb: QRhiCommandBuffer) -> None:  # noqa: ARG002 (Qt)
        rhi = self.rhi()
        self._rhi_ptr = shiboken6.getCppPointer(rhi)[0]
        for gpu in self._gpu.values():
            for b in gpu.buffers.values():
                b.destroy()
        self._gpu.clear()
        # Everything uploaded so far must go again: a new RHI has nothing.
        for hemi, cpu in self._cpu.items():
            pend = self._pending.setdefault(hemi, {})
            pend.update(cpu)
            if hemi in self._faces:
                if "full" in self._faces[hemi]:
                    pend["_faces"] = self._faces[hemi]["full"]
                if "flat" in self._faces[hemi]:
                    pend["_flat"] = self._faces[hemi]["flat"]
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
        self._empty = self._new_3d(np.zeros((1, 1, 1), np.float32))
        self._empty_pending = True
        self._uploaded = [None] * s3.MAX_LAYERS
        self._slot_tex = [None] * s3.MAX_LAYERS
        self._lut_tex = rhi.newTexture(QRhiTexture.Format.RGBA8, QtCore.QSize(256, s3.MAX_LAYERS))
        self._lut_tex.create()
        self._lut_bytes = None
        self._palette_tex = rhi.newTexture(QRhiTexture.Format.RGBA8, QtCore.QSize(s3.PALETTE_W, 1))
        self._palette_tex.create()
        self._palette_bytes = None
        self._rebind = True
        self._pipeline = None
        self._rhi_ready = True

    def _new_3d(self, data: np.ndarray) -> QRhiTexture:
        nz, ny, nx = data.shape
        tex = self.rhi().newTexture(
            QRhiTexture.Format.R32F, nx, ny, nz, 1, QRhiTexture.Flag.ThreeDimensional
        )
        tex.create()
        return tex

    def _upload_3d(self, batch, tex: QRhiTexture, data: np.ndarray) -> None:
        entries = [
            QRhiTextureUploadEntry(
                z, 0, QRhiTextureSubresourceUploadDescription(QtCore.QByteArray(data[z].tobytes()))
            )
            for z in range(data.shape[0])
        ]
        desc = QRhiTextureUploadDescription()
        desc.setEntries(entries)
        batch.uploadTexture(tex, desc)

    def _ensure_pipeline(self, srb) -> None:
        if self._pipeline is not None:
            return
        ps = self.rhi().newGraphicsPipeline()
        ps.setShaderStages(
            [
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader("surface.vert")),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader("surface.frag")),
            ]
        )
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(_STRIDE.get(a, 12)) for a in _ATTRS])
        fmt = QRhiVertexInputAttribute.Format
        layout.setAttributes(
            [
                QRhiVertexInputAttribute(
                    i,
                    i,
                    {"curv": fmt.Float, "vcolor": fmt.UNormByte4, "areas": fmt.Float2}.get(
                        a, fmt.Float3
                    ),
                    0,
                )
                for i, a in enumerate(_ATTRS)
            ]
        )
        ps.setVertexInputLayout(layout)
        ps.setDepthTest(True)
        ps.setDepthWrite(True)
        ps.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        ps.setSampleCount(self.sampleCount())
        ps.setShaderResourceBindings(srb)
        ps.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        ps.create()
        self._pipeline = ps

    def _bindings(self, gpu: _HemiGPU) -> None:
        rhi = self.rhi()
        if gpu.ubuf is None:
            gpu.ubuf = rhi.newBuffer(
                QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, s3.UNIFORM_BYTES
            )
            gpu.ubuf.create()
        srb = gpu.srb or rhi.newShaderResourceBindings()
        stage = _STAGE.VertexStage | _STAGE.FragmentStage
        frag = _STAGE.FragmentStage
        bindings = [Binding.uniformBuffer(0, stage, gpu.ubuf)]
        drawn = self._drawn()
        for k in range(s3.MAX_LAYERS):
            tex = self._slot_tex[k] or (self._empty, self._empty)
            labels = k < len(drawn) and drawn[k].shade.labels
            # Labels are never interpolated: a blend of 12 and 40 is not 26.
            sampler = self._linear_s if self.linear and not labels else self._nearest
            bindings.append(Binding.sampledTexture(1 + k, frag, tex[0], sampler))
            bindings.append(Binding.sampledTexture(5 + k, frag, tex[1], sampler))
        bindings.append(Binding.sampledTexture(9, frag, self._lut_tex, self._nearest))
        bindings.append(Binding.sampledTexture(10, frag, self._palette_tex, self._nearest))
        srb.setBindings(bindings)
        srb.create()
        gpu.srb = srb

    def _apply_pending(self, batch) -> None:
        rhi = self.rhi()
        for hemi, pend in list(self._pending.items()):
            gpu = self._gpu.setdefault(hemi, _HemiGPU())
            for key, arr in pend.items():
                if key in ("_faces", "_flat"):
                    buf = rhi.newBuffer(
                        QRhiBuffer.Type.Immutable, QRhiBuffer.UsageFlag.IndexBuffer, arr.nbytes
                    )
                    buf.create()
                    batch.uploadStaticBuffer(buf, 0, arr.tobytes())
                    if key == "_faces":
                        gpu.index_full, gpu.n_full = buf, arr.size
                    else:
                        gpu.index_flat, gpu.n_flat = buf, arr.size
                    continue
                buf = gpu.buffers.get(key)
                if buf is None or buf.size() != arr.nbytes:
                    if buf is not None:
                        buf.destroy()
                    buf = rhi.newBuffer(
                        QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.VertexBuffer, arr.nbytes
                    )
                    buf.create()
                    gpu.buffers[key] = buf
                batch.updateDynamicBuffer(buf, 0, arr.tobytes())
        self._pending.clear()
        if self._empty_pending:
            self._upload_3d(batch, self._empty, np.zeros((1, 1, 1), np.float32))
            self._empty_pending = False
        drawn = self._drawn()
        for k in range(s3.MAX_LAYERS):
            slot = drawn[k] if k < len(drawn) else None
            key = None if slot is None else slot.key
            if key == self._uploaded[k]:
                continue
            if slot is None:
                self._slot_tex[k] = None
            else:
                value = self._new_3d(slot.value)
                self._upload_3d(batch, value, slot.value)
                stat = value
                if slot.stat is not None:
                    stat = self._new_3d(slot.stat)
                    self._upload_3d(batch, stat, slot.stat)
                self._slot_tex[k] = (value, stat)
            self._uploaded[k] = key
            self._rebind = True
        # One LUT row per slot; label palettes stacked, each slot told its row.
        lut = np.zeros((s3.MAX_LAYERS, 256, 4), np.uint8)
        palettes: list[np.ndarray] = []
        rows = 0
        for k, slot in enumerate(drawn):
            if slot.lut is not None:
                lut[k] = slot.lut
            self._palette_rows[k] = rows
            if slot.palette is not None:
                palettes.append(slot.palette)
                rows += slot.palette.shape[0]
        raw = lut.tobytes()
        if raw != self._lut_bytes:
            img = QtGui.QImage(
                raw, 256, s3.MAX_LAYERS, 4 * 256, QtGui.QImage.Format.Format_RGBA8888
            )
            batch.uploadTexture(self._lut_tex, img.copy())
            self._lut_bytes = raw
        pal = np.concatenate(palettes) if palettes else np.zeros((1, s3.PALETTE_W, 4), np.uint8)
        raw = pal.tobytes()
        if raw != self._palette_bytes:
            size = QtCore.QSize(s3.PALETTE_W, pal.shape[0])
            if self._palette_tex.pixelSize() != size:
                self._palette_tex = rhi.newTexture(QRhiTexture.Format.RGBA8, size)
                self._palette_tex.create()
                self._rebind = True
            img = QtGui.QImage(raw, s3.PALETTE_W, pal.shape[0], 4 * s3.PALETTE_W,
                               QtGui.QImage.Format.Format_RGBA8888)  # fmt: skip
            batch.uploadTexture(self._palette_tex, img.copy())
            self._palette_bytes = raw
        # Which slots are labels decides their samplers.
        labels = tuple(slot.shade.labels for slot in drawn)
        if labels != getattr(self, "_bound_labels", None):
            self._bound_labels = labels
            self._rebind = True
        if self._rebind:
            for gpu in self._gpu.values():
                self._bindings(gpu)
            self._rebind = False

    def set_linear(self, on: bool) -> None:
        self.linear = bool(on)
        self._rebind = True
        self.update()

    def render(self, cb: QRhiCommandBuffer) -> None:
        rhi = self.rhi()
        if shiboken6.getCppPointer(rhi)[0] != getattr(self, "_rhi_ptr", None):
            self.initialize(cb)
        batch = rhi.nextResourceUpdateBatch()
        self._apply_pending(batch)
        size = self.renderTarget().pixelSize()
        aspect = size.width() / max(size.height(), 1)
        corr = _qmatrix(rhi.clipSpaceCorrMatrix())
        proj = corr @ self.camera.projection(aspect)
        view = self.camera.view()
        draws = []
        for hemi, gpu in self._gpu.items():
            if hemi not in self._visible or not all(a in gpu.buffers for a in _ATTRS):
                continue
            flat = self._flat.get(hemi, False)
            index, count = (gpu.index_flat, gpu.n_flat) if flat else (gpu.index_full, gpu.n_full)
            if index is None or count == 0:
                continue
            if gpu.srb is None:
                self._bindings(gpu)
            model = self.model(hemi)
            assert gpu.ubuf is not None
            batch.updateDynamicBuffer(
                gpu.ubuf,
                0,
                s3.pack_uniforms(
                    proj @ view @ model,
                    view @ model,
                    morph=self.morph,
                    depth=self.depth,
                    samples=self.samples,
                    fold_contrast=self.fold_contrast,
                    layers=[
                        s3.LayerUniforms(
                            slot.value_frame,
                            slot.stat_frame if slot.stat_frame is not None else slot.value_frame,
                            slot.shade,
                            lut_row=k,
                            palette_row=self._palette_rows[k],
                        )
                        for k, slot in enumerate(self._drawn())
                    ],
                    cross=self.cross,
                    cross_rgb=tuple(theme.palette().crosshair[:3]),
                    equivolume=self.equivolume,
                    map_opacity=self.map_opacity,
                ),
            )
            draws.append((gpu, index, count))
        bg = QtGui.QColor(theme.palette().bg)
        cb.beginPass(self.renderTarget(), bg, QRhiDepthStencilClearValue(1.0, 0), batch)
        if draws:
            self._ensure_pipeline(draws[0][0].srb)
            cb.setGraphicsPipeline(self._pipeline)
            cb.setViewport(QRhiViewport(0, 0, size.width(), size.height()))
            for gpu, index, count in draws:
                cb.setShaderResources(gpu.srb)
                cb.setVertexInput(
                    0,
                    [(gpu.buffers[a], 0) for a in _ATTRS],
                    index,
                    0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,
                )
                cb.drawIndexed(count)
        cb.endPass()

    def releaseResources(self) -> None:  # noqa: N802 (Qt)
        self._rhi_ready = False
        self._pipeline = None
        self._gpu.clear()

    # -- input -----------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        self._press = self._last = event.position()
        self._moved = False
        self._grabbed = None
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            hit = self._hit(event.position())
            self._grabbed = None if hit is None else hit[0]

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self._last is None:
            return
        d = event.position() - self._last
        self._last = event.position()
        if self._press is not None and (event.position() - self._press).manhattanLength() > 3:
            self._moved = True
        h = max(self.height(), 1)
        if event.buttons() & QtCore.Qt.MouseButton.RightButton:
            self.camera.pan(d.x() / h, d.y() / h)
        elif event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            if self._grabbed is not None:
                # Turn the grabbed hemisphere about the screen's axes, in world
                # terms, so the drag means the same thing from any view.
                right, up = self.camera.rotation[0], self.camera.rotation[1]
                turn = s3._rotation(up, d.x() / h * np.pi) @ s3._rotation(right, d.y() / h * np.pi)
                r = self.hemi_rotation.get(self._grabbed, np.eye(3))
                self.hemi_rotation[self._grabbed] = turn @ r
            else:
                self.camera.orbit(d.x() / h * np.pi, d.y() / h * np.pi)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and not self._moved
            and self._press is not None
        ):
            hit = self.pick_mm(event.position())
            if hit is not None:
                self.located.emit(*hit)
        self._press = self._last = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        delta = event.angleDelta()
        steps = (delta.y() or delta.x()) / 120.0
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Scroll through cortical depth -- the laminar view.
            self.depth_scrolled.emit(0.05 * steps)
            return
        self.camera.zoom(0.9**steps)
        self.update()

    def pick_mm(self, pos: QtCore.QPointF) -> tuple[float, float, float] | None:
        """Scanner mm under a widget point, at the sampled mid-depth."""
        hit = self._hit(pos)
        return None if hit is None else hit[1]

    def _hit(self, pos: QtCore.QPointF) -> tuple[str, tuple[float, float, float]] | None:
        """Which hemisphere is under a widget point, and the scanner mm there."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        x = 2.0 * pos.x() / w - 1.0
        y = 1.0 - 2.0 * pos.y() / h
        origin, direction = self.camera.ray(x, y, w / h)
        best: tuple[float, str, int, np.ndarray] | None = None
        for hemi in self._visible:
            drawn = self.drawn_positions(hemi)
            faces = self._faces.get(hemi, {}).get("flat" if self._flat.get(hemi) else "full")
            if drawn is None or faces is None:
                continue
            found = s3.pick(origin, direction, drawn, faces.reshape(-1, 3))
            if found is not None and (best is None or found[2] < best[0]):
                best = (found[2], hemi, found[0], found[1])
        if best is None:
            return None
        _, hemi, face, bary = best
        corners = self._faces[hemi]["flat" if self._flat.get(hemi) else "full"].reshape(-1, 3)[face]
        cpu = self._cpu[hemi]
        d = np.full(3, 0.5 * (self.depth[0] + self.depth[1]))
        if self.equivolume and "areas" in cpu:
            # Where the shader samples: the same equivolume depth per corner.
            aw, ap = cpu["areas"][corners, 0], cpu["areas"][corners, 1]
            d = s3.equivolume_fraction(d, aw, ap)
        d = d[:, None]
        point = (1 - d) * cpu["white"][corners] + d * cpu["pial"][corners]
        mm = bary @ point
        return hemi, (float(mm[0]), float(mm[1]), float(mm[2]))


class SurfaceWindow(QtWidgets.QWidget):
    """A surface viewport as a top-level window."""

    closed = QtCore.Signal(str)
    #: A click on the surface, scanner mm -- the manager turns it into SET_XYZ.
    located = QtCore.Signal(float, float, float)

    _SHAPE_KEYS = {
        "mid": "1",
        "white": "2",
        "pial": "3",
        "inflated": "4",
        "sphere": "5",
        "flat": "6",
    }

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
        #: What was last built, so refresh() rebuilds only what changed.
        self._built_shape: str | None = None
        self._built_versions: dict[str, int] = {}
        #: Texture-ready voxels per overlay key, so a redraw that changes only
        #: a threshold does not re-read or re-transpose a volume.
        self._slot_cache: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.setSpacing(4)
        self._shape_buttons: dict[str, QtWidgets.QPushButton] = {}
        for shape in s3.SHAPES:
            b = QtWidgets.QPushButton(f"{shape.upper()[:4]} {self._SHAPE_KEYS[shape]}")
            b.setCheckable(True)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, s=shape: self._set_shape(s))
            bar.addWidget(b)
            self._shape_buttons[shape] = b
        bar.addSpacing(8)
        self._hemi_buttons: dict[str, QtWidgets.QPushButton] = {}
        for hemi, key in (("lh", "L"), ("rh", "R")):
            b = QtWidgets.QPushButton(f"{hemi.upper()} {key}")
            b.setCheckable(True)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, h=hemi: self._toggle_hemi(h))
            bar.addWidget(b)
            self._hemi_buttons[hemi] = b
        bar.addSpacing(8)
        self.map_box = self._combo("Per-vertex map painted under the overlay (c cycles)")
        self.map_box.activated.connect(self._pick_map)
        bar.addWidget(self.map_box)
        self.annot_box = self._combo(
            "Surface parcellation: the map's colours and the region readout"
        )
        self.annot_box.activated.connect(self._pick_atlas)
        bar.addWidget(self.annot_box)
        self.atlas_box = self._combo("Label volume the readout names regions from")
        self.atlas_box.activated.connect(self._pick_atlas)
        bar.addWidget(self.atlas_box)
        bar.addStretch(1)
        self.depth_label = QtWidgets.QLabel("")
        self.depth_label.setObjectName("value")
        bar.addWidget(self.depth_label)
        v.addLayout(bar)
        self.region_label = QtWidgets.QLabel("")
        self.region_label.setObjectName("value")
        self.region_label.setContentsMargins(8, 0, 8, 2)
        v.addWidget(self.region_label)

        self.canvas = SurfaceCanvas(self)
        self.canvas.located.connect(self.located)
        self.canvas.depth_scrolled.connect(self._scroll_depth)
        self._built_map: tuple | None = None
        self._built_fold: tuple | None = None
        self._fold_hemis: set[str] = set()
        self._map_hemis: set[str] = set()
        v.addWidget(self.canvas, 1)
        self.resize(640, 520)

        def depth_step(delta: float) -> Callable[[], None]:
            return lambda: self._shift_depth(delta)

        self.help = ShortcutHelp(self, f"surface · {vid}")
        self.help.apply(
            [
                *[
                    Key(k, f"{s} shape", lambda s=s: self._set_shape(s), group="shape")
                    for s, k in self._SHAPE_KEYS.items()
                ],
                Key("shift+l", "show / hide the left hemisphere", lambda: self._toggle_hemi("lh"), group="hemispheres"),
                Key("shift+r", "show / hide the right hemisphere", lambda: self._toggle_hemi("rh"), group="hemispheres"),
                Key(">", "split hemispheres apart", lambda: self._split_by(10.0), group="hemispheres"),
                Key("<", "bring them back together", lambda: self._split_by(-10.0), group="hemispheres"),
                Key("[", "sample shallower (toward white)", depth_step(-0.1), group="depth"),
                Key("]", "sample deeper (toward pial)", depth_step(0.1), group="depth"),
                Key("{", "fewer depth samples", lambda: self._samples_by(-1), group="depth"),
                Key("}", "more depth samples (average white..pial)", lambda: self._samples_by(1), group="depth"),
                Key("shift+scroll", "scroll through cortical depth", None, group="depth"),
                Key("e", "equivolume / equidistant depth", self._toggle_equivolume, group="depth"),
                Key("c", "next per-vertex map (thickness, sulc, curv, parcellation)", self._cycle_map, group="view"),
                Key("k", "folding shade: curv / sulc / binary / off", self._cycle_folding, group="view"),
                Key("n", "nearest / linear voxel sampling", self._toggle_linear, group="view"),
                Key("v", "next view (top, lateral, medial, front...)", lambda: self._cycle_view(1), group="view"),
                Key("shift+v", "previous view", lambda: self._cycle_view(-1), group="view"),
                Key("0", "reset the camera", self._reset_camera, group="view"),
                Key("drag", "rotate", None, group="view"),
                Key("alt+drag", "turn one hemisphere about its centre", None, group="hemispheres"),
                Key("right-drag", "pan", None, group="view"),
                Key("scroll", "zoom", None, group="view"),
                Key("click", "move the crosshair there", None, group="view"),
                Key("h", "this list", self.help.toggle, group="window"),
                Key("w", "close this window", self.close, group="window"),
            ]
        )  # fmt: skip
        keep_keys_for_shortcuts(self)

    def _combo(self, tip: str) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        box.setToolTip(tip)
        box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        return box

    # -- viewport plumbing --------------------------------------------------
    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)
        for shape, b in self._shape_buttons.items():
            b.setChecked(shape == viewport.shape)
        shown = {h for h in viewport.hemis.split(",") if h}
        for hemi, b in self._hemi_buttons.items():
            b.setChecked(hemi in shown)
        self.refresh(Aspect.ALL)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.canvas.update()

    def _set_shape(self, shape: str) -> None:
        self._dispatch(SetSurfaceShape(self.vid, shape))

    def _toggle_hemi(self, hemi: str) -> None:
        vp = self._viewport()
        if vp is None:
            return
        shown = [h for h in vp.hemis.split(",") if h]
        shown = [h for h in shown if h != hemi] if hemi in shown else sorted([*shown, hemi])
        self._dispatch(SetSurfaceHemis(self.vid, ",".join(shown), vp.split))

    def _split_by(self, delta: float) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetSurfaceHemis(self.vid, vp.hemis, max(0.0, vp.split + delta)))

    def _shift_depth(self, delta: float) -> None:
        vp = self._viewport()
        if vp is None:
            return
        lo, hi = vp.depth
        if vp.samples <= 1:
            d = float(np.clip(lo + delta, 0.0, 1.0))
            self._dispatch(SetSurfaceDepth(self.vid, round(d, 3), round(d, 3), 1))
        else:
            # Several samples: [ and ] slide the whole window.
            width = hi - lo
            lo = float(np.clip(lo + delta, 0.0, 1.0 - width))
            self._dispatch(
                SetSurfaceDepth(self.vid, round(lo, 3), round(lo + width, 3), vp.samples)
            )

    def _samples_by(self, delta: int) -> None:
        vp = self._viewport()
        if vp is None:
            return
        n = int(np.clip(vp.samples + delta, 1, 16))
        lo, hi = (0.0, 1.0) if vp.samples == 1 and n > 1 else vp.depth
        if n == 1:
            lo = hi = 0.5 * (lo + hi)
        self._dispatch(SetSurfaceDepth(self.vid, lo, hi, n))

    def _toggle_linear(self) -> None:
        self.canvas.set_linear(not self.canvas.linear)

    def _cycle_view(self, step: int) -> None:
        names = list(s3.VIEWS)
        self._view_index = (getattr(self, "_view_index", 0) + step) % len(names)
        name = names[self._view_index]
        cam = self.canvas.camera
        cam.rotation = s3.VIEWS[name].copy()
        # Medial views are of one hemisphere; hide the other or it is in the way.
        vp = self._viewport()
        if vp is not None and "medial" in name:
            keep = "lh" if name.startswith("left") else "rh"
            if vp.hemis != keep:
                self._dispatch(SetSurfaceHemis(self.vid, keep, vp.split))
        self._frame()
        self.canvas.update()
        self.depth_label.setText(name)

    def _reset_camera(self) -> None:
        self.canvas.camera = s3.Camera()
        self.canvas.hemi_rotation.clear()
        self._frame()
        self.canvas.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)

    # -- building ---------------------------------------------------------------
    def _topology(self, hemi: str) -> MeshTopology:
        return self.session.surfaces.topology(hemi)

    def _shape_arrays(self, shape: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Laid-out positions and normals of ``shape`` per hemisphere."""
        hemis = self.session.surfaces.hemis
        raw: dict[str, np.ndarray] = {}
        faces: dict[str, np.ndarray] = {}
        for h, hemi in hemis.items():
            pos = s3.shape_positions(hemi, shape)
            if pos is None:
                pos = s3.shape_positions(hemi, "mid")
            assert pos is not None
            raw[h] = pos
            faces[h] = s3.shape_faces(hemi, shape)
        offsets = s3.layout_offsets(raw, shape, faces)
        out = {}
        for h, pos in raw.items():
            laid = pos + offsets[h]
            normals = vertex_normals(laid.astype(np.float64), self._topology(h)).astype(np.float32)
            if shape == "flat":
                normals = np.tile(np.float32([0, 0, 1]), (laid.shape[0], 1))
            out[h] = (laid.astype(np.float32), normals)
        return out

    def refresh(self, dirty: Aspect) -> None:
        vp = self._viewport()
        surfaces = self.session.surfaces
        if vp is None:
            return
        if not surfaces.hemis:
            self.canvas.drop_hemispheres()
            self._built_shape = None
            self._built_versions = {}
            self._map_hemis = set()
            self._fold_hemis = set()
            return
        c = self.canvas
        shape_changed = vp.shape != self._built_shape
        moved = {
            h for h in surfaces.hemis if surfaces.version.get(h) != self._built_versions.get(h)
        }
        if (
            moved
            and not shape_changed
            and self._built_shape is not None
            and (vp.shape not in s3.ANATOMICAL_SHAPES)
        ):
            # An edit moved white/pial under a laid-out shape (inflated, flat):
            # only where data is sampled changed, not where anything is drawn.
            for h in moved:
                hemi = surfaces.hemis[h]
                c.set_hemisphere(
                    h,
                    pos_a=None,
                    pos_b=None,
                    nrm_a=None,
                    nrm_b=None,
                    white=hemi.states["white"],
                    pial=hemi.states["pial"],
                    areas=self._areas(h),
                )
            self._built_versions = {h: surfaces.version.get(h, 0) for h in surfaces.hemis}
            moved = set()
        if shape_changed or moved:
            new = self._shape_arrays(vp.shape)
            first = self._built_shape is None or bool(moved - set(self._built_versions))
            for h, (pos, nrm) in new.items():
                hemi = surfaces.hemis[h]
                now = c.current(h)
                if first or not shape_changed or now is None:
                    # Nothing to morph from -- or an edit, which should not animate.
                    pos_a, nrm_a = pos, nrm
                else:
                    # From wherever it is drawn now, so a shape change mid-morph
                    # turns smoothly rather than jumping back to the old shape.
                    pos_a, nrm_a = now
                patch = s3.flat_patch(hemi)
                flat_faces = (patch.faces(hemi.faces) if patch is not None else hemi.faces).astype(
                    np.uint32
                )
                c.set_hemisphere(
                    h,
                    pos_a=pos_a,
                    pos_b=pos,
                    nrm_a=nrm_a,
                    nrm_b=nrm,
                    white=hemi.states["white"],
                    pial=hemi.states["pial"],
                    areas=self._areas(h),
                    faces=hemi.faces.astype(np.uint32) if h not in self._built_versions else None,
                    flat_faces=flat_faces if h not in self._built_versions else None,
                )
            if shape_changed and self._built_shape is not None:
                c.animate_to()
            else:
                c.morph = 1.0
            if self._built_shape is None:
                self._built_shape = vp.shape
                self._frame()
            self._built_shape = vp.shape
            self._built_versions = {h: surfaces.version.get(h, 0) for h in surfaces.hemis}
        self._refresh_map(vp)
        self._refresh_folding(vp)
        self._apply_layout(vp)
        c.depth = vp.depth
        c.samples = vp.samples
        c.equivolume = vp.equivolume
        self._refresh_data()
        self._refresh_cross()
        self._sync_header(vp)
        c.update()

    def _areas(self, hemi: str) -> np.ndarray:
        """``(V, 2)`` white and pial vertex areas, for equivolume depth.

        From the current meshes rather than ``?h.area``/``?h.area.pial``, so an
        edit's change of area is in the next frame.
        """
        h = self.session.surfaces.hemis[hemi]
        faces = h.faces.astype(np.int64)
        return np.stack(
            [
                vertex_areas(h.states["white"], faces, h.n_vertices),
                vertex_areas(h.states["pial"], faces, h.n_vertices),
            ],
            axis=1,
        ).astype(np.float32)

    def _refresh_folding(self, vp: Viewport) -> None:
        surfaces = self.session.surfaces
        key = (vp.folding, surfaces.subject)
        fresh = {h for h in surfaces.hemis if h not in self._fold_hemis}
        if key == self._built_fold and not fresh:
            return
        for h, hemi in surfaces.hemis.items():
            self.canvas.set_hemisphere(
                h,
                pos_a=None,
                pos_b=None,
                nrm_a=None,
                nrm_b=None,
                curv=s3.folding_values(hemi, vp.folding),
            )
        self._built_fold = key
        self._fold_hemis = set(surfaces.hemis)

    def _cycle_folding(self) -> None:
        vp = self._viewport()
        if vp is not None:
            modes = list(s3.FOLDING)
            nxt = modes[(modes.index(vp.folding) + 1) % len(modes)]
            self._dispatch(SetSurfaceFolding(self.vid, nxt))

    def _refresh_map(self, vp: Viewport) -> None:
        surfaces = self.session.surfaces
        annot = self.session.state.surface_annot
        key = (
            vp.vertex_map,
            annot if vp.vertex_map == "annot" else "",
            surfaces.flags_version if vp.vertex_map == "flags" else 0,
            surfaces.subject,
        )
        fresh = {h for h in surfaces.hemis if h not in self._map_hemis}
        if key == self._built_map and not fresh:
            return
        for h, hemi in surfaces.hemis.items():
            ann = surfaces.annotation(h, annot) if vp.vertex_map == "annot" else None
            colours = s3.vertex_colors(hemi, vp.vertex_map, ann, surfaces.flags.get(h))
            if colours is None:
                colours = np.zeros((hemi.n_vertices, 4), np.uint8)
            self.canvas.set_hemisphere(
                h, pos_a=None, pos_b=None, nrm_a=None, nrm_b=None, vcolor=colours
            )
        self._built_map = key
        self._map_hemis = set(surfaces.hemis)

    def _sync_header(self, vp: Viewport) -> None:
        lo, hi = vp.depth
        how = "equivol" if vp.equivolume else "linear"
        self.depth_label.setText(
            f"{how} {lo:.2f}" if vp.samples <= 1 else f"{how} {lo:.2f}-{hi:.2f} x{vp.samples}"
        )
        st = self.session.state
        for box, items, current in (
            (self.map_box, list(s3.VERTEX_MAPS), vp.vertex_map),
            (self.annot_box, ["", *self._annots()], st.surface_annot),
            (self.atlas_box, ["", *self._atlases()], st.volume_atlas),
        ):
            box.blockSignals(True)
            if [box.itemText(i) for i in range(box.count())] != items:
                box.clear()
                box.addItems(items)
            box.setCurrentText(current)
            box.blockSignals(False)
        lines = self.session.surfaces.region_lines(
            st.crosshair_mm, st.surface_annot, st.volume_atlas
        )
        self.region_label.setText("   ".join(lines) if lines else "")

    def _annots(self) -> list[str]:
        subject = self.session.surfaces.subject
        return available_annotations(subject) if subject is not None else []

    def _atlases(self) -> list[str]:
        subject = self.session.surfaces.subject
        return available_volume_atlases(subject) if subject is not None else []

    def _pick_map(self, _index: int) -> None:
        self._dispatch(SetSurfaceMap(self.vid, self.map_box.currentText()))

    def _pick_atlas(self, _index: int) -> None:
        self._dispatch(SetAtlas(self.annot_box.currentText(), self.atlas_box.currentText()))

    def _cycle_map(self) -> None:
        vp = self._viewport()
        if vp is not None:
            maps = list(s3.VERTEX_MAPS)
            self._dispatch(
                SetSurfaceMap(self.vid, maps[(maps.index(vp.vertex_map) + 1) % len(maps)])
            )

    def _toggle_equivolume(self) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetSurfaceEquivolume(self.vid, not vp.equivolume))

    def _scroll_depth(self, delta: float) -> None:
        """Shift+wheel: slide the sampled depth (or depth window) by ``delta``."""
        self._shift_depth(delta)

    def _apply_layout(self, vp: Viewport) -> None:
        shown = {h for h in vp.hemis.split(",") if h}
        models: dict[str, np.ndarray] = {}
        for h in self.session.surfaces.hemis:
            m = np.eye(4)
            side = -1.0 if h == "lh" else 1.0
            m[0, 3] = side * vp.split / 2.0
            models[h] = m
        flat = {h: vp.shape == "flat" for h in self.session.surfaces.hemis}
        self.canvas.set_layout(shown, models, flat)

    def _frame(self) -> None:
        pts = [
            p
            for h in self.session.surfaces.hemis
            if (p := self.canvas.drawn_positions(h)) is not None
        ]
        if pts:
            self.canvas.camera.frame(np.concatenate(pts))

    def _overlay_layers(self) -> list:
        """Every visible layer above the base, bottom to top -- the slices' stack.

        The base (the anatomy) is not drawn: under the overlays is the folding,
        which is what says where the sulci and gyri are on an inflated surface.
        """
        st = self.session.state
        layers = list(st.layers)[1:]
        return [layer for layer in layers if layer.visible]

    def _refresh_data(self) -> None:
        import torch

        from fastfuncstuff.viewer.compose import cached_lut
        from fastfuncstuff.viewer.layers import AlphaMode, SignMode

        st = self.session.state
        signs = {SignMode.BOTH: 0, SignMode.POS: 1, SignMode.NEG: 2}
        alphas = {AlphaMode.OFF: 0, AlphaMode.LINEAR: 1, AlphaMode.QUADRATIC: 2}
        slots: list[OverlaySlot] = []
        cache: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}
        for layer in self._overlay_layers()[-s3.MAX_LAYERS :]:
            idx = int(st.time_index if layer.time_linked else layer.volume_index)
            thr = layer.threshold_index
            key = (layer.key, idx, thr if thr != idx else None, np.asarray(layer.affine).tobytes())
            data = self._slot_cache.get(key)
            if data is None:
                try:
                    value = s3.texture_data(self.session.volume(layer.key, idx))
                    stat = (
                        None
                        if thr is None or thr == idx
                        else s3.texture_data(self.session.volume(layer.key, thr))
                    )
                except (KeyError, FileNotFoundError, ValueError):
                    continue
                data = (value, stat)
            cache[key] = data
            frame = s3.texture_from_mm(layer.shape, layer.affine)
            outline = (
                s3.OUTLINE_ONLY
                if layer.edges
                else (s3.OUTLINE_BOXED if layer.boxed else s3.OUTLINE_NONE)
            )
            opacity = float(layer.opacity)
            if layer.roi:
                palette = self.session.roi_palette(layer.key, torch.device("cpu"))
                if palette is None:
                    continue
                slots.append(
                    OverlaySlot(
                        key=key,
                        value=data[0],
                        value_frame=frame,
                        stat=None,
                        stat_frame=None,
                        shade=s3.ShadeParams(
                            opacity=opacity, has_data=True, labels=True, outline=outline
                        ),
                        palette=s3.palette_texture(palette.numpy()),
                    )
                )
                continue
            lut = cached_lut(layer.colormap, torch.device("cpu"), reverse=layer.colormap_reversed)
            rgb = np.clip(np.asarray(lut.cpu().numpy(), np.float64), 0, 1)
            rgba = np.concatenate([np.round(rgb * 255), np.full((rgb.shape[0], 1), 255.0)], 1)
            slots.append(
                OverlaySlot(
                    key=key,
                    value=data[0],
                    value_frame=frame,
                    stat=data[1],
                    stat_frame=None if data[1] is None else frame,
                    shade=s3.ShadeParams(
                        lo=float(layer.range_lo if layer.range_lo is not None else 0.0),
                        hi=float(layer.range_hi if layer.range_hi is not None else 1.0),
                        threshold=float(layer.threshold),
                        opacity=opacity,
                        sign_mode=signs[layer.sign_mode],
                        alpha_mode=alphas[layer.alpha_mode],
                        n_panes=int(layer.n_panes),
                        has_data=True,
                        outline=outline,
                    ),
                    lut=rgba.astype(np.uint8),
                )
            )
        # Keep only what is drawn: a dropped layer's 100 MB goes with it.
        self._slot_cache = cache
        self.canvas.set_overlays(slots)

    def _refresh_cross(self) -> None:
        mm = self.session.state.crosshair_mm
        if mm is None:
            self.canvas.cross = (0.0, 0.0, 0.0, 0.0)
            return
        self.canvas.cross = (float(mm[0]), float(mm[1]), float(mm[2]), 1.5)


__all__ = ["SurfaceCanvas", "SurfaceWindow"]
