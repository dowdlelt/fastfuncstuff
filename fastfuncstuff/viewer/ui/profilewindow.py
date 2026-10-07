"""The profile column window: the whole cortex's ribbon profiles, back to front.

Three strips, left to right: an overview of the whole brain (the selected
flag, max-pooled into every pixel row, with the visible span boxed), the
profiles themselves (CSF black to WM white, one row per vertex, white and
pial marked in fraction mode), and the flag of each shown row. Click a row
and the crosshair goes there; move the crosshair and the column follows.

Building the column is seconds (sampling every vertex), so it is built when
the window opens and when its settings change, and an edit re-samples only
the vertices it moved.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.surface.profiles import SCORES, ProfileSpec
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.profilecolumn import (
    ProfileColumn,
    build_column,
    flags_by_vertex,
    update_column,
)
from fastfuncstuff.viewer.surface3d import flag_ramp
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding as Key
from fastfuncstuff.viewer.ui.shortcuts import ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import SetProfileView

OVERVIEW_W = 26
SCORE_W = 14
GAP = 4


class ProfileView(QtWidgets.QWidget):
    """Paints a :class:`ProfileColumn`; reports clicks as rows."""

    #: A row was clicked (in the profile or score strip).
    rowed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.column: ProfileColumn | None = None
        self.score = "worst"
        #: First visible row, and rows per screen pixel.
        self.top = 0.0
        self.rows_per_px = 1.0
        self.marked: int | None = None
        self._ramp = flag_ramp()
        self.setMinimumSize(160, 200)
        self.setMouseTracking(True)

    # -- geometry ----------------------------------------------------------
    def _profile_rect(self) -> QtCore.QRect:
        x0 = OVERVIEW_W + GAP
        w = max(10, self.width() - x0 - GAP - SCORE_W)
        return QtCore.QRect(x0, 0, w, self.height())

    def visible_span(self) -> tuple[int, int]:
        start = int(self.top)
        return start, int(start + self.height() * self.rows_per_px)

    def clamp(self) -> None:
        if self.column is None:
            return
        n = self.column.n_rows
        self.rows_per_px = float(
            np.clip(self.rows_per_px, 0.05, max(n / max(self.height(), 1), 0.05))
        )
        span = self.height() * self.rows_per_px
        self.top = float(np.clip(self.top, 0, max(n - span, 0)))

    def centre_on(self, row: int) -> None:
        self.top = row - 0.5 * self.height() * self.rows_per_px
        self.clamp()
        self.update()

    def zoom(self, factor: float, about_y: float | None = None) -> None:
        y = self.height() / 2 if about_y is None else about_y
        anchor = self.top + y * self.rows_per_px
        self.rows_per_px *= factor
        self.clamp()
        self.top = anchor - y * self.rows_per_px
        self.clamp()
        self.update()

    def fit_all(self) -> None:
        if self.column is not None:
            self.rows_per_px = self.column.n_rows / max(self.height(), 1)
            self.top = 0
            self.clamp()
            self.update()

    # -- painting ------------------------------------------------------------
    def _rows_on_screen(self) -> np.ndarray:
        assert self.column is not None
        start, stop = self.visible_span()
        n_px = self.height() if self.rows_per_px >= 1 else max(1, int(np.ceil(stop - start)))
        return self.column.pick_rows(start, stop, n_px, self.score)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        col = self.column
        if col is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "building profiles...")
            p.end()
            return
        h = self.height()
        rows = self._rows_on_screen()
        pr = self._profile_rect()
        if rows.size:
            grey = col.grey(rows)  # (R, S)
            img = np.ascontiguousarray(np.repeat(grey[..., None], 3, axis=2))
            qimg = QtGui.QImage(
                img.data,
                img.shape[1],
                img.shape[0],
                3 * img.shape[1],
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(pr, qimg)
            if self.rows_per_px > 1:
                start, stop = self.visible_span()
                s = col.flag_density(start, stop, rows.size, self.score)
            else:
                s = col.scores[self.score][rows]
            bar = np.ascontiguousarray(self._ramp[(np.clip(s, 0, 1) * 255).astype(int)][:, None, :])
            qbar = QtGui.QImage(
                bar.data, 1, bar.shape[0], 3, QtGui.QImage.Format.Format_RGB888
            ).copy()
            p.drawImage(QtCore.QRect(pr.right() + GAP, 0, SCORE_W, h), qbar)
        guides = col.boundary_columns()
        if guides is not None:
            s_cols = col.profiles.values.shape[1]
            pen = QtGui.QPen(QtGui.QColor.fromRgbF(*c.label))
            pen.setStyle(QtCore.Qt.PenStyle.DotLine)
            p.setPen(pen)
            for k in guides:
                x = pr.x() + (k + 0.5) / s_cols * pr.width()
                p.drawLine(QtCore.QLineF(x, 0, x, h))
        self._paint_overview(p)
        if self.marked is not None:
            start, _ = self.visible_span()
            y = (self.marked - start) / self.rows_per_px
            if 0 <= y < h:
                pen = QtGui.QPen(QtGui.QColor(c.accent))
                pen.setWidth(1)
                p.setPen(pen)
                p.drawLine(QtCore.QLineF(pr.x() - GAP, y, pr.right() + GAP + SCORE_W, y))
        p.end()

    def _paint_overview(self, p: QtGui.QPainter) -> None:
        col = self.column
        assert col is not None
        h = max(self.height(), 1)
        # Fraction of each bin flagged, not its max: max-pooling ~300 rows into
        # a pixel catches something nearly everywhere, and the overview's job
        # is to say where the trouble is concentrated.
        pooled = col.flag_density(0, col.n_rows, h, self.score)
        rgb = np.ascontiguousarray(
            self._ramp[(np.clip(pooled, 0, 1) * 255).astype(int)][:, None, :]
        )
        q = QtGui.QImage(rgb.data, 1, rgb.shape[0], 3, QtGui.QImage.Format.Format_RGB888).copy()
        p.drawImage(QtCore.QRect(0, 0, OVERVIEW_W, h), q)
        start, stop = self.visible_span()
        y0, y1 = start / col.n_rows * h, min(stop, col.n_rows) / col.n_rows * h
        pen = QtGui.QPen(QtGui.QColor(theme.palette().accent))
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(QtCore.QRectF(0.5, y0, OVERVIEW_W - 1.0, max(y1 - y0, 2.0)))

    # -- input -----------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        if self.column is None:
            return
        y = event.position().y()
        if event.position().x() < OVERVIEW_W:
            # The overview is the whole brain: jump there.
            row = int(y / max(self.height(), 1) * self.column.n_rows)
            self.centre_on(row)
            return
        rows = self._rows_on_screen()
        if rows.size:
            k = int(np.clip(y / self.height() * rows.size, 0, rows.size - 1))
            self.rowed.emit(int(rows[k]))

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt)
        steps = event.angleDelta().y() / 120.0
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.zoom(0.8**steps, event.position().y())
            return
        self.top -= steps * 0.1 * self.height() * self.rows_per_px
        self.clamp()
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        self.clamp()
        super().resizeEvent(event)


class ProfileWindow(QtWidgets.QWidget):
    """A profile viewport as a top-level window."""

    closed = QtCore.Signal(str)
    located = QtCore.Signal(float, float, float)
    #: New flags were published to the surface store.
    flags_changed = QtCore.Signal()

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
        self._built_key: tuple | None = None
        self._built_versions: dict[str, int] = {}
        self.follow = True

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        self.score_box = QtWidgets.QComboBox()
        self.score_box.addItems(list(SCORES))
        self.score_box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.score_box.activated.connect(self._pick_score)
        bar.addWidget(self.score_box)
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("value")
        bar.addWidget(self.info, 1)
        v.addLayout(bar)
        self.view = ProfileView(self)
        self.view.rowed.connect(self._locate)
        v.addWidget(self.view, 1)
        self.resize(300, 760)

        self.help = ShortcutHelp(self, f"profiles · {vid}")
        self.help.apply(
            [
                Key("m", "depth axis: stretched (fraction) / mm", self._toggle_mode, group="profiles"),
                Key("(", "thinner tube", lambda: self._tube_by(-0.25), group="profiles"),
                Key(")", "wider tube", lambda: self._tube_by(0.25), group="profiles"),
                Key("s", "next flag", lambda: self._cycle_score(1), group="profiles"),
                Key("n", "next flagged row below", lambda: self._next_flag(1), group="profiles"),
                Key("p", "previous flagged row above", lambda: self._next_flag(-1), group="profiles"),
                Key("f", "follow the crosshair on / off", self._toggle_follow, group="view"),
                Key("+", "zoom in", lambda: self.view.zoom(0.5), group="view", aliases=("=",)),
                Key("-", "zoom out", lambda: self.view.zoom(2.0), group="view"),
                Key("0", "whole brain", self.view.fit_all, group="view"),
                Key("scroll", "move along (ctrl: zoom)", None, group="view"),
                Key("click", "move the crosshair to that vertex", None, group="view"),
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
        self.score_box.setCurrentText(viewport.profile_score)
        self.view.score = viewport.profile_score
        self.refresh(Aspect.ALL)

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.view.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)

    def _set(self, **changes) -> None:
        vp = self._viewport()
        if vp is None:
            return
        mode = changes.get("mode", vp.profile_mode)
        score = changes.get("score", vp.profile_score)
        tube = changes.get("tube", vp.tube)
        self._dispatch(SetProfileView(self.vid, mode, score, round(float(tube), 3)))

    def _pick_score(self, _index: int) -> None:
        self._set(score=self.score_box.currentText())

    def _cycle_score(self, step: int) -> None:
        vp = self._viewport()
        if vp is not None:
            names = list(SCORES)
            self._set(score=names[(names.index(vp.profile_score) + step) % len(names)])

    def _toggle_mode(self) -> None:
        vp = self._viewport()
        if vp is not None:
            self._set(mode="mm" if vp.profile_mode == "fraction" else "fraction")

    def _tube_by(self, delta: float) -> None:
        vp = self._viewport()
        if vp is not None:
            self._set(tube=float(np.clip(vp.tube + delta, 0.0, 3.0)))

    def _toggle_follow(self) -> None:
        self.follow = not self.follow
        self._sync_info()

    def _locate(self, row: int) -> None:
        col = self.view.column
        if col is None:
            return
        self.view.marked = row
        self.located.emit(*col.row_mm(row))

    def _next_flag(self, step: int) -> None:
        """Jump to the next row past the marked one whose flag is high."""
        col = self.view.column
        if col is None:
            return
        s = col.scores.get(self.view.score)
        if s is None:
            return
        here = self.view.marked if self.view.marked is not None else int(self.view.top)
        flagged = np.flatnonzero(s >= 0.8)
        if flagged.size == 0:
            return
        k = np.searchsorted(flagged, here, side="right" if step > 0 else "left")
        k = k if step > 0 else k - 1
        row = int(flagged[k % flagged.size])
        self.view.centre_on(row)
        self._locate(row)

    # -- building --------------------------------------------------------------
    def _anatomy(self):
        st = self.session.state
        layer = st.layers.find(st.surface_snap_key) if st.surface_snap_key else st.layers.base
        if layer is None:
            return None
        idx = st.time_index if layer.time_linked else layer.volume_index
        return layer, int(idx)

    def refresh(self, dirty: Aspect) -> None:
        vp = self._viewport()
        surfaces = self.session.surfaces
        found = self._anatomy()
        if vp is None or not surfaces.hemis or found is None:
            self.view.column = None
            self.view.update()
            return
        layer, idx = found
        spec = ProfileSpec(mode=vp.profile_mode, tube_radius=float(vp.tube))
        key = (
            spec,
            layer.key,
            idx,
            np.asarray(layer.affine).tobytes(),
            surfaces.subject,
            # A topology edit renumbers vertices: re-sampling only "what moved"
            # would index the wrong rows.
            surfaces.topology_version,
        )
        device = getattr(self.session, "display_device", None)
        if key != self._built_key:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                volume = self.session.volume(layer.key, idx)
                self.view.column = build_column(
                    surfaces.hemis, volume, layer.affine, spec, device=device
                )
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._built_key = key
            self._built_versions = dict(surfaces.version)
            self.view.fit_all()
        elif dict(surfaces.version) != self._built_versions and self.view.column is not None:
            # An edit landed: re-sample only what moved.
            volume = self.session.volume(layer.key, idx)
            update_column(self.view.column, surfaces.hemis, volume, layer.affine, device=device)
            self._built_versions = dict(surfaces.version)
        col = self.view.column
        if col is not None and vp.profile_score not in col.scores:
            self.view.score = "worst"
        else:
            self.view.score = vp.profile_score
        if col is not None and self.follow and dirty & Aspect.CROSSHAIR:
            mm = self.session.state.crosshair_mm
            if mm is not None:
                row = col.nearest_row(mm)
                if self.view.marked != row:
                    self.view.marked = row
                    start, stop = self.view.visible_span()
                    if not start <= row < stop:
                        self.view.centre_on(row)
        self._publish()
        self._sync_info()
        self.view.update()

    def _publish(self) -> None:
        """Hand the current flag to the surface windows (map: flags)."""
        col = self.view.column
        if col is None:
            return
        key = (id(col), self.view.score, tuple(sorted(self.session.surfaces.version.items())))
        if key == getattr(self, "_published", None):
            return
        sizes = {h: hemi.n_vertices for h, hemi in self.session.surfaces.hemis.items()}
        self.session.surfaces.publish_flags(flags_by_vertex(col, self.view.score, sizes))
        self._published = key
        self.flags_changed.emit()

    def _sync_info(self) -> None:
        vp = self._viewport()
        col = self.view.column
        if vp is None or col is None:
            self.info.setText("")
            return
        flagged = int((col.scores[self.view.score] >= 0.8).sum())
        self.info.setText(
            f"{col.n_rows:,} vertices · {vp.profile_mode} · tube {vp.tube:g} mm · "
            f"{flagged:,} flagged{'' if self.follow else ' · parked'}"
        )


__all__ = ["ProfileView", "ProfileWindow"]
