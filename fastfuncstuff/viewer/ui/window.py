"""The main window.

Layout is single-window with docked panes, per the architecture poll. The window
does three things and delegates everything else: it turns input into commands,
it repaints what a command reports dirty, and it marshals background-load
completions back onto the GUI thread.

The repaint discipline is the point. A command returns the aspects it actually
changed, and only those panes redraw -- a crosshair move never re-slices, and a
colourmap change never re-uploads a volume. That, plus loading on a worker, is
what keeps the window responsive while a dataset inflates behind it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.colormap import available_colormaps
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.compose import render_plane
from fastfuncstuff.viewer.layers import AlphaMode, SignMode
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import plane_axes, voxel_value
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui.graph import GraphPane, Series
from fastfuncstuff.viewer.ui.panes import ImagePane
from fastfuncstuff.viewer.vocab import (
    SetAlpha,
    SetBoxed,
    SetColormap,
    SetIJK,
    SetIndex,
    SetLayerVisible,
    SetSeed,
    SetSign,
    SetThreshold,
    SetTimeLinked,
)

STYLESHEET = """
QMainWindow, QWidget { background: #07090B; color: #C9D6DA; }
QDockWidget { titlebar-close-icon: none; }
QDockWidget::title {
    background: #0E1216; padding: 6px 8px;
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
}
QListWidget {
    background: #0E1216; border: 1px solid #1E272C; outline: none;
    font-family: monospace; font-size: 11px;
}
QListWidget::item { padding: 5px 7px; }
QListWidget::item:selected { background: #16323A; color: #C9D6DA; }
QLabel { color: #6B7D84; font-size: 10px; letter-spacing: 1px; }
QLabel#value { color: #C9D6DA; font-family: monospace; font-size: 11px; }
QSlider::groove:horizontal { height: 2px; background: #1E272C; }
QSlider::handle:horizontal {
    background: #7DE3C3; width: 8px; margin: -5px 0; border-radius: 0;
}
QComboBox {
    background: #0E1216; border: 1px solid #1E272C; padding: 3px 6px;
    font-family: monospace; font-size: 11px; color: #C9D6DA;
}
QComboBox QAbstractItemView {
    background: #0E1216; color: #C9D6DA; selection-background-color: #16323A;
}
QCheckBox { color: #6B7D84; font-size: 10px; letter-spacing: 1px; }
QStatusBar {
    background: #0E1216; color: #6B7D84;
    font-family: monospace; font-size: 11px;
}
QSplitter::handle { background: #1E272C; }
"""


class _Bridge(QtCore.QObject):
    """Marshals worker-thread completions onto the GUI thread.

    Background loads finish on a pool thread; touching widgets from there is
    undefined behaviour in Qt. A queued signal is the supported crossing.
    """

    loaded = QtCore.Signal(str)


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, session: ViewerSession) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle("nexus")
        self.setStyleSheet(STYLESHEET)
        self.resize(1240, 860)

        self._panes: dict[Plane, ImagePane] = {}
        self._bridge = _Bridge()
        self._bridge.loaded.connect(self._on_layer_loaded, QtCore.Qt.ConnectionType.QueuedConnection)
        session.on_loaded(self._bridge.loaded.emit)

        self._build_panes()
        self._build_layer_dock()
        self._build_statusbar()
        self._install_shortcuts()

        self._play = QtCore.QTimer(self)
        self._play.setInterval(60)
        self._play.timeout.connect(lambda: self._step_time(1))

        self.refresh(Aspect.ALL)

    # -- construction --------------------------------------------------
    def _build_panes(self) -> None:
        grid = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(grid)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        order = [(Plane.AXIAL, 0, 0), (Plane.SAGITTAL, 0, 1), (Plane.CORONAL, 1, 0)]
        for plane, r, c in order:
            pane = ImagePane(plane)
            pane.picked.connect(lambda a, b, p=plane: self._pick(p, a, b))
            pane.seeded.connect(lambda a, b, p=plane: self._pick(p, a, b, seed=True))
            pane.stepped.connect(lambda d, p=plane: self._step_slice(p, d))
            self._panes[plane] = pane
            layout.addWidget(pane, r, c)

        self.graph = GraphPane()
        self.graph.scrubbed.connect(self._set_time)
        layout.addWidget(self.graph, 1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.setCentralWidget(grid)

    def _build_layer_dock(self) -> None:
        dock = QtWidgets.QDockWidget("layers", self)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(7)

        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.currentRowChanged.connect(lambda _: self._sync_controls())
        v.addWidget(self.layer_list, 1)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(available_colormaps())
        self.cmap_box.activated.connect(
            lambda _: self._apply(SetColormap, colormap=self.cmap_box.currentText())
        )
        form.addRow(QtWidgets.QLabel("COLOR"), self.cmap_box)

        self.sign_box = QtWidgets.QComboBox()
        self.sign_box.addItems([m.value for m in SignMode])
        self.sign_box.activated.connect(
            lambda _: self._apply(SetSign, mode=self.sign_box.currentText())
        )
        form.addRow(QtWidgets.QLabel("SIGN"), self.sign_box)

        self.alpha_box = QtWidgets.QComboBox()
        self.alpha_box.addItems([m.value for m in AlphaMode])
        self.alpha_box.activated.connect(
            lambda _: self._apply(SetAlpha, mode=self.alpha_box.currentText())
        )
        form.addRow(QtWidgets.QLabel("ALPHA"), self.alpha_box)

        self.thr_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.thr_slider.setRange(0, 1000)
        self.thr_slider.valueChanged.connect(self._threshold_moved)
        form.addRow(QtWidgets.QLabel("THRESH"), self.thr_slider)

        self.thr_label = QtWidgets.QLabel("0")
        self.thr_label.setObjectName("value")
        form.addRow(QtWidgets.QLabel(""), self.thr_label)

        self.boxed_check = QtWidgets.QCheckBox("boxed")
        self.boxed_check.toggled.connect(lambda on: self._apply(SetBoxed, on=bool(on)))
        form.addRow(QtWidgets.QLabel(""), self.boxed_check)

        self.timelink_check = QtWidgets.QCheckBox("follows time")
        self.timelink_check.setToolTip(
            "4-D NIfTI cannot say whether sub-bricks are time points or "
            "contrasts. Uncheck for a stats dataset."
        )
        self.timelink_check.toggled.connect(lambda on: self._apply(SetTimeLinked, on=bool(on)))
        form.addRow(QtWidgets.QLabel(""), self.timelink_check)

        v.addLayout(form)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setMinimumWidth(240)

    def _build_statusbar(self) -> None:
        self.coord_label = QtWidgets.QLabel("")
        self.value_label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.coord_label)
        self.statusBar().addPermanentWidget(self.value_label)

    def _install_shortcuts(self) -> None:
        # Keyboard first, per the brief: every action here is reachable without
        # the mouse, and each one goes through the same commands the UI does.
        binds: list[tuple[str, object]] = [
            ("Left", lambda: self._nudge(0, -1)),
            ("Right", lambda: self._nudge(0, 1)),
            ("Down", lambda: self._nudge(1, -1)),
            ("Up", lambda: self._nudge(1, 1)),
            ("PgDown", lambda: self._nudge(2, -1)),
            ("PgUp", lambda: self._nudge(2, 1)),
            (",", lambda: self._step_time(-1)),
            (".", lambda: self._step_time(1)),
            ("v", self._toggle_play),
            ("space", self._toggle_visible),
            ("[", lambda: self._cycle_layer(-1)),
            ("]", lambda: self._cycle_layer(1)),
            ("t", lambda: self._nudge_threshold(-0.05)),
            ("Shift+T", lambda: self._nudge_threshold(0.05)),
            ("a", self._cycle_alpha),
            ("b", self._toggle_boxed),
            ("s", self._cycle_sign),
            ("c", self._cycle_colormap),
            ("Ctrl+O", self._open_dialog),
            ("Ctrl+S", self._save_script_dialog),
        ]
        for seq, fn in binds:
            act = QtGui.QAction(self)
            act.setShortcut(QtGui.QKeySequence(seq))
            act.triggered.connect(fn)  # type: ignore[arg-type]
            self.addAction(act)

    # -- helpers -------------------------------------------------------
    def current_key(self) -> str | None:
        row = self.layer_list.currentRow()
        keys = self.session.state.layers.keys
        if 0 <= row < len(keys):
            # The list shows the stack top-first, which is how people read it.
            return list(reversed(keys))[row]
        return None

    def _apply(self, cls, **kwargs) -> None:
        key = self.current_key()
        if key is None:
            return
        self.refresh(self.session.do(cls(key=key, **kwargs)))

    # -- input ---------------------------------------------------------
    def _pick(self, plane: Plane, row: int, col: int, *, seed: bool = False) -> None:
        _, r_ax, c_ax = plane_axes(plane)
        ijk = list(self.session.state.crosshair)
        ijk[r_ax], ijk[c_ax] = row, col
        cmd = SetSeed(*ijk) if seed else SetIJK(*ijk)
        self.refresh(self.session.do(cmd))

    def _nudge(self, axis: int, delta: int) -> None:
        ijk = list(self.session.state.crosshair)
        ijk[axis] += delta
        self.refresh(self.session.do(SetIJK(*ijk)))

    def _step_slice(self, plane: Plane, delta: int) -> None:
        self._nudge(plane_axes(plane)[0], delta)

    def _step_time(self, delta: int) -> None:
        self._set_time(self.session.state.time_index + delta)

    def _set_time(self, index: int) -> None:
        hi = self.session.state.max_time_index()
        self.refresh(self.session.do(SetIndex(int(index) % (hi + 1) if hi > 0 else 0)))

    def _toggle_play(self) -> None:
        self._play.stop() if self._play.isActive() else self._play.start()

    def _toggle_visible(self) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        self.refresh(self.session.do(SetLayerVisible(key, not layer.visible)))
        self._sync_layer_list()

    def _cycle_layer(self, delta: int) -> None:
        n = self.layer_list.count()
        if n:
            self.layer_list.setCurrentRow((self.layer_list.currentRow() + delta) % n)

    def _threshold_moved(self, tick: int) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        hi = abs(layer.range_hi or 1.0)
        self.refresh(self.session.do(SetThreshold(key, tick / 1000.0 * hi)))
        self.thr_label.setText(f"{tick / 1000.0 * hi:.4g}")

    def _nudge_threshold(self, frac: float) -> None:
        self.thr_slider.setValue(max(0, min(1000, self.thr_slider.value() + int(frac * 1000))))

    def _cycle_alpha(self) -> None:
        modes = [m.value for m in AlphaMode]
        i = (self.alpha_box.currentIndex() + 1) % len(modes)
        self.alpha_box.setCurrentIndex(i)
        self._apply(SetAlpha, mode=modes[i])

    def _cycle_sign(self) -> None:
        modes = [m.value for m in SignMode]
        i = (self.sign_box.currentIndex() + 1) % len(modes)
        self.sign_box.setCurrentIndex(i)
        self._apply(SetSign, mode=modes[i])

    def _cycle_colormap(self) -> None:
        maps = available_colormaps()
        i = (self.cmap_box.currentIndex() + 1) % len(maps)
        self.cmap_box.setCurrentIndex(i)
        self._apply(SetColormap, colormap=maps[i])

    def _toggle_boxed(self) -> None:
        self.boxed_check.toggle()

    # -- dialogs -------------------------------------------------------
    def _open_dialog(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open dataset", "", "Images (*.nii *.nii.gz *.nii.zst *.HEAD);;All (*)"
        )
        for p in paths:
            self.open_path(p)

    def open_path(self, path: str | Path) -> None:
        self.session.load(path)
        self._sync_layer_list()
        self.layer_list.setCurrentRow(0)
        self.refresh(Aspect.ALL)

    def _save_script_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session script", "session.ffs", "Scripts (*.ffs);;All (*)"
        )
        if path:
            self.session.save_script(path, header="nexus session")
            self.statusBar().showMessage(f"wrote {path}", 4000)

    # -- refresh -------------------------------------------------------
    def _on_layer_loaded(self, key: str) -> None:
        """A background inflate finished: the layer is now scrubbable."""
        self.session.invalidate(key)
        self._sync_layer_list()
        self.refresh(Aspect.SLICES | Aspect.GRAPH)

    def refresh(self, dirty: Aspect) -> None:
        if dirty is Aspect.NOTHING:
            return
        if dirty & (Aspect.LAYERS | Aspect.GRID):
            self._sync_layer_list()
        if dirty & (Aspect.SLICES | Aspect.COLORMAP | Aspect.THRESHOLD | Aspect.TIME | Aspect.GRID):
            self._redraw_panes()
        if dirty & (Aspect.CROSSHAIR | Aspect.GRID):
            self._redraw_crosshairs()
        if dirty & (Aspect.CROSSHAIR | Aspect.GRAPH | Aspect.TIME | Aspect.LAYERS):
            self._redraw_graph()
        if dirty & (Aspect.CROSSHAIR | Aspect.SLICES | Aspect.TIME | Aspect.LAYERS):
            self._sync_readout()

    def _redraw_panes(self) -> None:
        for plane, pane in self._panes.items():
            pane.set_pane(render_plane(self.session, plane))
        self._redraw_crosshairs()

    def _redraw_crosshairs(self) -> None:
        ijk = self.session.state.crosshair
        for plane, pane in self._panes.items():
            _, r_ax, c_ax = plane_axes(plane)
            pane.set_crosshair(ijk[r_ax], ijk[c_ax])

    def _redraw_graph(self) -> None:
        series: list[Series] = []
        for layer in self.session.state.layers:
            if not layer.time_linked:
                continue
            values = self.session.timeseries(layer.key)
            if values.size:
                series.append(Series(label=layer.name, values=values))
        self.graph.set_series(series, self.session.state.time_index)

    def _sync_layer_list(self) -> None:
        row = self.layer_list.currentRow()
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in reversed(list(self.session.state.layers)):
            res = self.session.store.get(layer.key)
            mark = "▣" if layer.visible else "▢"
            tier = "  · loading" if res.pending else ""
            self.layer_list.addItem(f"{mark} {layer.name}{tier}")
        self.layer_list.blockSignals(False)
        if self.layer_list.count():
            self.layer_list.setCurrentRow(max(0, min(row, self.layer_list.count() - 1)))
        self._sync_controls()

    def _sync_controls(self) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        for box, value in (
            (self.cmap_box, layer.colormap),
            (self.sign_box, layer.sign_mode.value),
            (self.alpha_box, layer.alpha_mode.value),
        ):
            box.blockSignals(True)
            box.setCurrentText(value)
            box.blockSignals(False)
        for check, value, enabled in (
            (self.boxed_check, layer.boxed, True),
            (self.timelink_check, layer.time_linked, layer.n_volumes > 1),
        ):
            check.blockSignals(True)
            check.setChecked(value)
            check.setEnabled(enabled)
            check.blockSignals(False)
        hi = abs(layer.range_hi or 1.0)
        self.thr_slider.blockSignals(True)
        self.thr_slider.setValue(int(layer.threshold / hi * 1000) if hi else 0)
        self.thr_slider.blockSignals(False)
        self.thr_label.setText(f"{layer.threshold:.4g}")

    def _sync_readout(self) -> None:
        st = self.session.state
        if st.grid is None:
            return
        i, j, k = st.crosshair
        mm = st.crosshair_mm or (0.0, 0.0, 0.0)
        self.coord_label.setText(
            f"ijk {i:>3d} {j:>3d} {k:>3d}   "
            f"xyz {mm[0]:>7.1f} {mm[1]:>7.1f} {mm[2]:>7.1f}   t {st.time_index}"
        )
        parts: list[str] = []
        for layer in reversed(list(st.layers)):
            vol = self.session.display_volume(layer.key)
            if vol is None:
                continue
            val = voxel_value(vol, st.grid, layer.affine, st.crosshair)
            parts.append(f"{layer.name}={'--' if val is None else f'{val:.4g}'}")
        self.value_label.setText("   ".join(parts[:3]))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self._play.stop()
        self.session.close()
        super().closeEvent(event)


def launch(paths: list[str], *, device: str | None = None, script: str | None = None) -> int:
    """Open a window on the given datasets and run the Qt loop."""
    from fastfuncstuff.cli_utils import setup_device

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # setup_device, not torch.device: the canonical parser is what accepts the
    # documented cuda,N / cpu,N forms and applies the thread override.
    session = ViewerSession(device=setup_device(device))
    win = ViewerWindow(session)
    for p in paths:
        win.open_path(p)
    if script:
        win.refresh(session.run_script(Path(script).read_text()))
        win._sync_layer_list()
    win.show()
    return app.exec()


__all__ = ["ViewerWindow", "launch"]
