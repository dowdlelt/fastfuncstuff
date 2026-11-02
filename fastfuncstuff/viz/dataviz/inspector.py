"""The bridge from picture to data: one probed voxel, its slices and its series.

Click anything in a scene and the inspector (lower right) follows that voxel:

* three orthogonal slices through it at the current moment -- the raw value
  (z or percent signal change) over the mean image, with a crosshair. Clicking
  in a slice moves the probe there, so you can walk the data by hand.
* the voxel's time series across the whole run, in the same unit, with what
  the scene actually draws for it (the pooled value) on top, its family's time
  course faint behind, and the playhead.

z and PSC disagree in useful ways -- a vein or an edge voxel with a large mean
change can be big in PSC and ordinary in z, a quiet low-variance voxel the
reverse -- so the unit is a toggle, not a choice made for you.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

from .drives import interp_cols


class VoxelIndex:
    """Voxel indices re-expressed on a RAS-ordered grid, plus slice lookups."""

    def __init__(self, ijk: np.ndarray, shape: tuple[int, int, int], affine: np.ndarray):
        ornt = nib.orientations.io_orientation(affine)  # per voxel axis: (world axis, flip)
        ras = np.empty_like(ijk)
        ras_shape = [0, 0, 0]
        for ax, (world, flip) in enumerate(ornt):
            w = int(world)
            ras_shape[w] = shape[ax]
            ras[:, w] = shape[ax] - 1 - ijk[:, ax] if flip < 0 else ijk[:, ax]
        self.ras = ras
        self.shape = tuple(ras_shape)
        self.lookup = np.full(self.shape, -1, dtype=np.int64)
        self.lookup[tuple(ras.T)] = np.arange(len(ras))

    def slice_members(self, axis: int, at: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(voxel indices, rows, cols) of one slice in display orientation.

        Axial: anterior up, right on the right. Coronal: superior up, right on
        the right. Sagittal: superior up, anterior on the right.
        """
        sel = np.nonzero(self.ras[:, axis] == at)[0]
        x, y, z = self.ras[sel].T
        nx, ny, nz = self.shape
        if axis == 2:
            return sel, ny - 1 - y, x
        if axis == 1:
            return sel, nz - 1 - z, x
        return sel, nz - 1 - z, y

    def slice_shape(self, axis: int) -> tuple[int, int]:
        nx, ny, nz = self.shape
        return {2: (ny, nx), 1: (nz, nx), 0: (nz, ny)}[axis]


class Inspector(QtWidgets.QWidget):
    """Lower-right overlay on the canvas. Owns the probe."""

    probe_changed = QtCore.Signal(int)

    def __init__(self, viz, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.viz = viz
        f = viz.feats
        self.f = f
        self.vi = VoxelIndex(f.ijk, f.extras["shape"], f.extras["affine"])
        self.psc = False
        self.overlay_shown = False  # slices: raw value, or what the scene draws (pooled)
        self.probe = int(torch.argmax(f.salience))
        self._slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._slice_rects: list[QtCore.QRectF] = []
        self._series_key = None
        self.mean = f.mean.float().cpu().numpy()
        self.scale_psc = (
            (f.extras["sd"] / f.extras["raw_mean"].abs().clamp_min(1e-6) * 100)
            .float()
            .cpu()
            .numpy()
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(False)
        self.set_probe(self.probe)

    # -------------------------------------------------------------- probe

    def set_probe(self, v: int) -> None:
        self.probe = int(v)
        r = self.vi.ras[self.probe]
        self._slices = [self.vi.slice_members(ax, int(r[ax])) for ax in (2, 1, 0)]
        # One device index over all three slices, so a frame reads only the
        # voxels it draws.
        sels = [sel for sel, _, _ in self._slices]
        self._slice_idx = torch.as_tensor(np.concatenate(sels), device=self.f.z.device)
        self._slice_split = np.cumsum([len(sel) for sel in sels])[:-1]
        self._series_key = None
        self.probe_changed.emit(self.probe)
        self.update()

    def unit(self) -> str:
        return "% signal" if self.psc else "z"

    def _scale(self, idx: np.ndarray | int) -> np.ndarray | float:
        return self.scale_psc[idx] if self.psc else 1.0

    def _series(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dr = self.viz.vis.drives
        key = (self.probe, self.psc, dr.p["clarity"], dr.p["pool"], int(dr.p["reach"]))
        if key != self._series_key:
            v = self.probe
            s = self._scale(v)
            raw = self.f.z[:, v].float().cpu().numpy() * s
            shown = dr.shown_series_voxel(v).float().cpu().numpy() * s
            fam = self.f.family_tcs[:, int(self.f.family[v])].float().cpu().numpy()
            # The family course is unit variance; draw it at the raw series' scale.
            fam = fam * float(np.std(raw))
            self._cached = (raw, shown, fam)
            self._series_key = key
        return self._cached

    # -------------------------------------------------------------- paint

    def paintEvent(self, _e) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(6, 7, 12, 215))
        p.setPen(QtGui.QPen(QtGui.QColor(40, 46, 66), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        f = self.f
        v = self.probe
        font = p.font()
        font.setPointSizeF(8.0)
        p.setFont(font)
        p.setPen(QtGui.QColor(170, 180, 210))
        ijk = f.ijk[v]
        fam = int(f.family[v])
        flags = f.extras.get("family_flags", [])
        flag = f" [{', '.join(flags[fam])}]" if fam < len(flags) and flags[fam] else ""
        p.drawText(
            8,
            14,
            f"voxel {ijk[0]},{ijk[1]},{ijk[2]}   family {fam}{flag} (w {float(f.family_weight[v]):.2f})   "
            f"salience {float(f.salience[v]):.2f}   [{self.unit()}, {'pooled' if self.overlay_shown else 'raw'}]"
            "   p z/PSC · o raw/pooled",
        )
        w, h = self.width(), self.height()
        top = 22
        slice_h = int(h * 0.48)
        self._draw_slices(p, QtCore.QRectF(6, top, w - 12, slice_h))
        self._draw_trace(p, QtCore.QRectF(6, top + slice_h + 6, w - 12, h - top - slice_h - 12))
        p.end()

    def _draw_slices(self, p: QtGui.QPainter, area: QtCore.QRectF) -> None:
        t = self.viz.vis.t
        dr = self.viz.vis.drives
        if self.overlay_shown:
            vals = dr.live[self._slice_idx]  # exactly what the scene is drawing
        else:
            vals = interp_cols(self.f.z, t, self._slice_idx)
        per_slice = np.split(vals.float().cpu().numpy(), self._slice_split)
        self._slice_rects = []
        gap = 6
        each_w = (area.width() - 2 * gap) / 3
        for k, ((sel, rows, cols), axis) in enumerate(zip(self._slices, (2, 1, 0), strict=True)):
            sh = self.vi.slice_shape(axis)
            img = np.zeros((*sh, 3), dtype=np.float32)
            g = self.mean[sel] * 0.75
            val = per_slice[k] * self._scale(sel)
            thr = 1.0 if not self.psc else 0.5
            mag = np.clip((np.abs(val) - thr) / (2.5 * thr), 0, 1)[:, None]
            hot = np.array([1.0, 0.55, 0.15], dtype=np.float32)
            cold = np.array([0.2, 0.55, 1.0], dtype=np.float32)
            col = np.where((val >= 0)[:, None], hot, cold)
            img[rows, cols] = g[:, None] * (1 - mag) + col * mag
            u8 = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            qimg = QtGui.QImage(u8.data, sh[1], sh[0], 3 * sh[1], QtGui.QImage.Format.Format_RGB888)
            s = min(each_w / sh[1], area.height() / sh[0])
            rw, rh = sh[1] * s, sh[0] * s
            rect = QtCore.QRectF(
                area.left() + k * (each_w + gap) + (each_w - rw) / 2, area.top(), rw, rh
            )
            p.drawImage(rect, qimg)
            self._slice_rects.append(rect)
            # Crosshair at the probe.
            prow, pcol = self._probe_rc(axis, self.vi.ras[self.probe])
            cx = rect.left() + (pcol + 0.5) * s
            cy = rect.top() + (prow + 0.5) * s
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1))
            p.drawLine(QtCore.QPointF(cx - 6, cy), QtCore.QPointF(cx - 2, cy))
            p.drawLine(QtCore.QPointF(cx + 2, cy), QtCore.QPointF(cx + 6, cy))
            p.drawLine(QtCore.QPointF(cx, cy - 6), QtCore.QPointF(cx, cy - 2))
            p.drawLine(QtCore.QPointF(cx, cy + 2), QtCore.QPointF(cx, cy + 6))

    def _probe_rc(self, axis: int, r: np.ndarray) -> tuple[int, int]:
        nx, ny, nz = self.vi.shape
        x, y, z = (int(c) for c in r)
        if axis == 2:
            return ny - 1 - y, x
        if axis == 1:
            return nz - 1 - z, x
        return nz - 1 - z, y

    def _draw_trace(self, p: QtGui.QPainter, area: QtCore.QRectF) -> None:
        """The series, drawn once into a pixmap; a frame adds only the playhead.

        Rebuilding three 262-point paths in Python every frame was ~6 ms on the
        CPU, enough to drop a CPU-only Mac from smooth to stuttering.
        """
        raw, _, _ = self._series()
        n = raw.size
        key = (self._series_key, int(area.width()), int(area.height()))
        if getattr(self, "_trace_key", None) != key:
            self._trace_pix = self._render_trace(area)
            self._trace_key = key
        p.drawPixmap(area.topLeft(), self._trace_pix)
        px = area.left() + self.viz.vis.t / max(n - 1, 1) * area.width()
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1))
        p.drawLine(QtCore.QPointF(px, area.top()), QtCore.QPointF(px, area.bottom()))

    def _render_trace(self, area: QtCore.QRectF) -> QtGui.QPixmap:
        raw, shown, fam = self._series()
        w, h = max(1, int(area.width())), max(1, int(area.height()))
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        n = raw.size
        lim = max(1e-6, float(np.percentile(np.abs(raw), 99)) * 1.1)
        xs = np.arange(n) / max(n - 1, 1) * w

        def poly(y: np.ndarray) -> QtGui.QPolygonF:
            ys = h / 2 - np.clip(y / lim, -1, 1) * h / 2
            return QtGui.QPolygonF(
                [QtCore.QPointF(float(a), float(b)) for a, b in zip(xs, ys, strict=True)]
            )

        p.setPen(QtGui.QPen(QtGui.QColor(60, 66, 90), 1))
        p.drawLine(QtCore.QPointF(0, h / 2), QtCore.QPointF(w, h / 2))
        for start in self.f.run_starts[1:]:
            p.drawLine(QtCore.QPointF(xs[start], 0), QtCore.QPointF(xs[start], h))
        for series, colour, width in (
            (fam, (150, 120, 220, 90), 1.2),
            (raw, (150, 160, 190, 150), 1.0),
            (shown, (255, 190, 90, 230), 1.6),
        ):
            p.setPen(QtGui.QPen(QtGui.QColor(*colour), width))
            p.drawPolyline(poly(series))
        p.setPen(QtGui.QColor(130, 140, 170))
        p.drawText(
            QtCore.QPointF(2, 10),
            f"±{lim:.2g} {self.unit()}   grey raw · gold shown (pooled) · violet family",
        )
        p.end()
        return pix

    # -------------------------------------------------------------- input

    def mousePressEvent(self, e) -> None:
        pos = e.position()
        for rect, axis in zip(self._slice_rects, (2, 1, 0), strict=True):
            if rect.contains(pos):
                sh = self.vi.slice_shape(axis)
                row = int((pos.y() - rect.top()) / rect.height() * sh[0])
                col = int((pos.x() - rect.left()) / rect.width() * sh[1])
                r = self.vi.ras[self.probe].copy()
                nx, ny, nz = self.vi.shape
                if axis == 2:
                    r[0], r[1] = col, ny - 1 - row
                elif axis == 1:
                    r[0], r[2] = col, nz - 1 - row
                else:
                    r[1], r[2] = col, nz - 1 - row
                r = np.clip(r, 0, np.array(self.vi.shape) - 1)
                v = int(self.vi.lookup[tuple(r)])
                if v >= 0:
                    self.set_probe(v)
                return
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self.viz is not None:
            # Clicking the trace seeks.
            frac = (pos.x() - 6) / max(1, self.width() - 12)
            if pos.y() > self.height() * 0.5:
                self.viz.seek(float(np.clip(frac, 0, 1)) * (self.f.n_time - 1))
