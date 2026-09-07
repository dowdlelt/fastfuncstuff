"""The main window: a data selector, with everything else modular on top.

The core is one row -- Read, Underlay, Overlay, +1, Mode. That is the whole
viewer; images, graphs and mode panels are things you turn on above it.

Two consequences of taking that seriously:

* **No graph in the default layout.** Goal zero is looking at an underlay and
  an overlay together, and a graph that is always present is always taking
  space from the images. Graphs are floating windows opened per plane, the way
  AFNI's image and graph buttons pair up.
* **The window contains no mode-specific code.** It asks the active mode what
  controls to show and renders whatever it declares, so adding calc, GLM or ICA
  never touches this file.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.catalog import CatalogEntry
from fastfuncstuff.viewer.colormap import available_colormaps
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.compose import plane_position, render_plane
from fastfuncstuff.viewer.layers import AlphaMode, SignMode
from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import OverlayKind
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import plane_layout, voxel_value
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui.colorbar import RangeBar
from fastfuncstuff.viewer.ui.controls import ControlPanel
from fastfuncstuff.viewer.ui.gridgraph import GridGraphWindow
from fastfuncstuff.viewer.ui.panes import ImagePane
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp
from fastfuncstuff.viewer.ui.work import PreparationRunner, run_when_ready
from fastfuncstuff.viewer.vocab import (
    AddOverlay,
    Read,
    SetAlpha,
    SetBoxed,
    SetColormap,
    SetIJK,
    SetIndex,
    SetLayerOpacity,
    SetLayerVisible,
    SetMode,
    SetModeParam,
    SetOverlay,
    SetRange,
    SetSeed,
    SetSign,
    SetThreshold,
    SetTimeLinked,
    SetUnderlay,
)

#: Shown when no dataset is chosen. A picker that names a file while nothing is
#: displayed reads as a load that failed.
NONE_LABEL = "(none)"
PICKER_FONT = "Menlo" if __import__("sys").platform == "darwin" else "monospace"

STYLESHEET = """
QMainWindow, QWidget { background: #07090B; color: #C9D6DA; }
QDockWidget::title { background: #0E1216; padding: 6px 8px;
    font-size: 10px; letter-spacing: 2px; }
QListWidget { background: #0E1216; border: 1px solid #1E272C; outline: none;
    font-family: monospace; font-size: 11px; }
QListWidget::item { padding: 4px 7px; }
QListWidget::item:selected { background: #16323A; }
QLabel { color: #6B7D84; font-size: 10px; letter-spacing: 1px; }
QLabel#value { color: #C9D6DA; font-family: monospace; font-size: 11px; }
QLabel#head { color: #5C8EA0; font-size: 10px; letter-spacing: 2px; }
QSlider::groove:horizontal { height: 2px; background: #1E272C; }
QSlider::handle:horizontal { background: #7DE3C3; width: 8px; margin: -5px 0; }
QComboBox, QSpinBox { background: #0E1216; border: 1px solid #1E272C;
    padding: 3px 6px; font-family: monospace; font-size: 11px; color: #C9D6DA; }
QComboBox QAbstractItemView { background: #0E1216; color: #C9D6DA;
    selection-background-color: #16323A; }
QPushButton { background: #0E1216; border: 1px solid #1E272C; padding: 4px 10px;
    font-family: monospace; font-size: 11px; letter-spacing: 1px; color: #C9D6DA; }
QPushButton:hover { background: #16323A; }
QPushButton:checked { background: #16323A; border-color: #5C8EA0; color: #7DE3C3; }
QPushButton:disabled { color: #41525A; border-color: #161D22; }
QCheckBox { color: #6B7D84; font-size: 10px; letter-spacing: 1px; }
QCheckBox::indicator { width: 12px; height: 12px;
    border: 1px solid #3A4A52; background: #0E1216; }
QCheckBox::indicator:checked { background: #7DE3C3; border-color: #7DE3C3; }
QCheckBox::indicator:disabled { border-color: #202B31; }
QStatusBar { background: #0E1216; color: #6B7D84;
    font-family: monospace; font-size: 11px; }
QProgressBar { background: #0E1216; border: 1px solid #1E272C; height: 12px;
    text-align: center; font-size: 9px; color: #6B7D84; }
QProgressBar::chunk { background: #5C8EA0; }
QDoubleSpinBox { background: #0E1216; border: 1px solid #1E272C; padding: 2px 4px;
    font-family: monospace; font-size: 11px; color: #C9D6DA; }
QToolBar { background: #0E1216; border: 0; spacing: 5px; padding: 5px 7px; }
"""


class _Bridge(QtCore.QObject):
    """Marshals worker-thread load completions onto the GUI thread."""

    loaded = QtCore.Signal(str)


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, session: ViewerSession) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle("nexus")
        self.setStyleSheet(STYLESHEET)
        self.resize(1280, 880)

        self._panes: dict[Plane, ImagePane] = {}
        self._graphs: dict[str, GridGraphWindow] = {}
        self._bridge = _Bridge()
        self._bridge.loaded.connect(
            self._on_layer_loaded, QtCore.Qt.ConnectionType.QueuedConnection
        )
        session.on_loaded(self._bridge.loaded.emit)

        self._build_selector()
        self._build_panes()
        self._build_dock()
        self._build_statusbar()
        self._install_shortcuts()

        # Mode preparation runs on a worker; the mode is told to defer so a
        # seed click never runs seconds of filtering inside the click handler.
        self.runner = PreparationRunner(self)
        self.runner.progress.connect(self._on_prepare_progress)
        self.runner.busy_changed.connect(self._on_prepare_busy)
        session.defer_mode_preparation = True
        session.mode.defer_preparation = True

        self._play = QtCore.QTimer(self)
        self._play.setInterval(60)
        self._play.timeout.connect(lambda: self._step_time(1))

        self.refresh(Aspect.ALL)

    # ------------------------------------------------------------------
    # the core: read / underlay / overlay / +1 / mode
    # ------------------------------------------------------------------
    def _build_selector(self) -> None:
        bar = QtWidgets.QToolBar("selector")
        bar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        self.read_button = QtWidgets.QPushButton("READ")
        self.read_button.setToolTip("Read a directory into the pickers (ctrl+O)")
        self.read_button.clicked.connect(self._read_dialog)
        bar.addWidget(self.read_button)

        self.dir_label = QtWidgets.QLabel("no directory")
        bar.addWidget(self.dir_label)
        bar.addSeparator()

        bar.addWidget(self._head("UNDERLAY"))
        self.underlay_box = QtWidgets.QComboBox()
        self.underlay_box.setFont(QtGui.QFont(PICKER_FONT))
        self.underlay_box.addItem(NONE_LABEL, userData=None)
        self.underlay_box.activated.connect(lambda _: self._pick(self.underlay_box, SetUnderlay))
        bar.addWidget(self.underlay_box)

        bar.addWidget(self._head("OVERLAY"))
        self.overlay_box = QtWidgets.QComboBox()
        self.overlay_box.setFont(QtGui.QFont(PICKER_FONT))
        self.overlay_box.addItem(NONE_LABEL, userData=None)
        self.overlay_box.activated.connect(lambda _: self._pick(self.overlay_box, SetOverlay))
        bar.addWidget(self.overlay_box)

        self.plus_button = QtWidgets.QPushButton("+1")
        self.plus_button.setToolTip("Add the selected dataset on top, keeping the current overlay")
        self.plus_button.clicked.connect(lambda: self._pick(self.overlay_box, AddOverlay))
        bar.addWidget(self.plus_button)
        bar.addSeparator()

        bar.addWidget(self._head("MODE"))
        self.mode_box = QtWidgets.QComboBox()
        labels = registry.labels()
        for name in registry.names():
            self.mode_box.addItem(labels[name], userData=name)
        self.mode_box.setCurrentIndex(self.mode_box.findData(self.session.mode.name))
        self.mode_box.activated.connect(lambda _: self._switch_mode(self.mode_box.currentData()))
        bar.addWidget(self.mode_box)

        # Pane and graph toggles, paired the way AFNI's image/graph buttons are.
        view_bar = QtWidgets.QToolBar("views")
        self._view_bar = view_bar
        view_bar.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, view_bar)
        view_bar.addWidget(self._head("VIEW"))

        self._pane_buttons: dict[Plane, QtWidgets.QPushButton] = {}
        self._graph_buttons: dict[Plane, QtWidgets.QPushButton] = {}
        for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
            b = QtWidgets.QPushButton(plane.value[:3].capitalize())
            b.setCheckable(True)
            b.setChecked(True)
            b.toggled.connect(lambda on, p=plane: self._toggle_pane(p, on))
            view_bar.addWidget(b)
            self._pane_buttons[plane] = b

            g = QtWidgets.QPushButton("Gr")
            g.setCheckable(True)
            g.setToolTip(f"Floating {plane.value} graph: 1, 4 or 9 voxels at the cursor")
            g.toggled.connect(lambda on, p=plane: self._toggle_graph(p, on))
            view_bar.addWidget(g)
            self._graph_buttons[plane] = g
            view_bar.addSeparator()

        view_bar.addWidget(self._head("T"))
        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.setToolTip("Jump to a volume")
        self.time_spin.setKeyboardTracking(False)
        self.time_spin.setMaximumWidth(78)
        self.time_spin.valueChanged.connect(self._time_spin_changed)
        view_bar.addWidget(self.time_spin)
        self.time_label = QtWidgets.QLabel("")
        view_bar.addWidget(self.time_label)

        # The panel toggle is appended in _build_dock, once there is a dock to
        # toggle. It lives here with the other view switches rather than in a
        # menu, because a panel you cannot get back is the same bug as a pane
        # you cannot get back.

    @staticmethod
    def _fit_picker(box: QtWidgets.QComboBox) -> None:
        """Size a picker to its widest entry.

        Qt sizes a combo to its *current* item, which on macOS truncates every
        other row -- the volume count falls off the end, which is exactly the
        column you scan a results directory for. The popup view needs its width
        set separately from the closed box.
        """
        metrics = QtGui.QFontMetrics(box.font())
        widest = max(
            (metrics.horizontalAdvance(box.itemText(i)) for i in range(box.count())),
            default=120,
        )
        box.setMinimumWidth(min(widest + 44, 520))
        view = box.view()
        if view is not None:
            view.setMinimumWidth(min(widest + 28, 620))

    @staticmethod
    def _head(text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setObjectName("head")
        return lab

    def _pick(self, box: QtWidgets.QComboBox, cls) -> None:
        entry: CatalogEntry | None = box.currentData()
        if entry is None:
            return
        self.refresh(self.session.do(cls(str(entry.path))))
        self._sync_layer_list()

    def _read_dialog(self) -> None:
        start = str(self.session.catalog_dir or Path.cwd())
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Read directory", start)
        if directory:
            self.read_directory(directory)

    def read_directory(self, directory: str | Path) -> None:
        self.session.do(Read(str(directory)))
        self._sync_catalog()
        self.statusBar().showMessage(
            f"read {len(self.session.catalog)} datasets from {directory}", 5000
        )

    def _sync_catalog(self) -> None:
        """Fill the pickers, defaulting the underlay to the likeliest base image."""
        entries = self.session.catalog
        self.dir_label.setText(
            self.session.catalog_dir.name if self.session.catalog_dir else "no directory"
        )
        # Two columns, monospaced and padded, so a directory can be scanned by
        # dimensions and volume count rather than read name by name.
        width = max((len(e.name) for e in entries), default=0)
        for box in (self.underlay_box, self.overlay_box):
            box.blockSignals(True)
            box.clear()
            # Nothing is loaded until something is picked; an entry showing in
            # the box while the panes are empty reads as a failed load.
            box.addItem(NONE_LABEL, userData=None)
            for e in entries:
                box.addItem(f"{e.name:<{width}}   {e.summary}", userData=e)
            box.setCurrentIndex(0)
            box.blockSignals(False)
            self._fit_picker(box)
        self._sync_pickers()

    # ------------------------------------------------------------------
    # panes
    # ------------------------------------------------------------------
    def _build_panes(self) -> None:
        self._pane_host = QtWidgets.QWidget()
        self._pane_layout = QtWidgets.QGridLayout(self._pane_host)
        self._pane_layout.setContentsMargins(0, 0, 0, 0)
        self._pane_layout.setSpacing(1)
        for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
            pane = ImagePane(plane)
            pane.picked.connect(lambda a, b, p=plane: self._on_pick(p, a, b))
            pane.seeded.connect(lambda a, b, p=plane: self._on_pick(p, a, b, seed=True))
            pane.stepped.connect(lambda d, p=plane: self._step_slice(p, d))
            self._panes[plane] = pane
        self.setCentralWidget(self._pane_host)
        self._relayout_panes()

    def _relayout_panes(self) -> None:
        """Re-flow whichever panes are enabled into as square a grid as fits."""
        for plane, pane in self._panes.items():
            self._pane_layout.removeWidget(pane)
            pane.setVisible(self._pane_buttons[plane].isChecked())
        shown = [
            p
            for p in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL)
            if self._pane_buttons[p].isChecked()
        ]
        cols = 1 if len(shown) <= 1 else 2
        for i, plane in enumerate(shown):
            self._pane_layout.addWidget(self._panes[plane], i // cols, i % cols)
        for c in range(2):
            self._pane_layout.setColumnStretch(c, 1 if c < cols else 0)
        for r in range(2):
            self._pane_layout.setRowStretch(r, 1)

    def _toggle_pane(self, plane: Plane, on: bool) -> None:
        # Refuse to close the last pane: an image viewer showing no images is a
        # state whose only way out is the control the user just used.
        if not on and not any(b.isChecked() for b in self._pane_buttons.values()):
            self._pane_buttons[plane].setChecked(True)
            return
        self._relayout_panes()
        self._redraw_panes()

    def _toggle_graph(self, plane: Plane, on: bool) -> None:
        key = plane.value
        if on:
            win = self._graphs.get(key)
            if win is None:
                win = GridGraphWindow(plane, self.session, self)
                win.closed.connect(self._on_graph_closed)
                win.scrubbed.connect(self._time_spin_changed)
                self._graphs[key] = win
            win.show()
            win.raise_()
            win.refresh()
        elif key in self._graphs:
            self._graphs[key].hide()

    def _on_graph_closed(self, key: str) -> None:
        for plane, button in self._graph_buttons.items():
            if plane.value == key:
                button.setChecked(False)

    # ------------------------------------------------------------------
    # dock: layers, layer controls, mode controls
    # ------------------------------------------------------------------
    def _build_dock(self) -> None:
        dock = QtWidgets.QDockWidget("layers", self)
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setMaximumHeight(150)
        self.layer_list.currentRowChanged.connect(lambda _: self._sync_layer_controls())
        v.addWidget(self.layer_list)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(available_colormaps())
        self.cmap_box.activated.connect(
            lambda _: self._apply(SetColormap, colormap=self.cmap_box.currentText())
        )
        form.addRow(self._head("COLOR"), self.cmap_box)

        self.sign_box = QtWidgets.QComboBox()
        self.sign_box.addItems([m.value for m in SignMode])
        self.sign_box.activated.connect(
            lambda _: self._apply(SetSign, mode=self.sign_box.currentText())
        )
        form.addRow(self._head("SIGN"), self.sign_box)

        self.alpha_box = QtWidgets.QComboBox()
        self.alpha_box.addItems([m.value for m in AlphaMode])
        self.alpha_box.activated.connect(
            lambda _: self._apply(SetAlpha, mode=self.alpha_box.currentText())
        )
        form.addRow(self._head("ALPHA"), self.alpha_box)

        # Min, threshold and max are edited on the bar itself. Splitting the
        # number from the picture of the number is what let the bar go stale.
        self.thr_head = self._head("THRESH")
        self.rangebar = RangeBar()
        self.rangebar.range_changed.connect(self._range_changed)
        self.rangebar.threshold_changed.connect(self._threshold_changed)
        self.rangebar.autorange_requested.connect(self._autorange)
        form.addRow(self.thr_head, self.rangebar)

        # Kept as attributes so the rest of the window (and the tests) address
        # them by the name of the thing they control, not through the composite.
        self.min_spin = self.rangebar.min_spin
        self.max_spin = self.rangebar.max_spin
        self.thr_spin = self.rangebar.thr_spin
        self.thr_slider = self.rangebar.slider
        self.colorbar = self.rangebar.bar
        self.autorange_button = self.rangebar.auto_button

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        form.addRow(self._head("OPACITY"), self.opacity_slider)

        self.opacity_label = QtWidgets.QLabel("100%")
        self.opacity_label.setObjectName("value")
        form.addRow(QtWidgets.QLabel(""), self.opacity_label)

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

        self.mode_head = self._head("MODE PARAMETERS")
        v.addWidget(self.mode_head)
        self.mode_panel = ControlPanel()
        self.mode_panel.changed.connect(self._mode_param_changed)
        v.addWidget(self.mode_panel)
        v.addStretch(1)

        dock.setWidget(panel)
        dock.setMinimumWidth(268)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.dock = dock

        # toggleViewAction rather than a hand-rolled show/hide: Qt keeps its
        # checked state in sync with the dock however it was closed, including
        # the X on the dock's own title bar.
        self._view_bar.addSeparator()
        self.panel_button = QtWidgets.QPushButton("Panel")
        self.panel_button.setCheckable(True)
        self.panel_button.setChecked(True)
        self.panel_button.setToolTip("Show or hide the layers panel (p)")
        action = dock.toggleViewAction()
        self.panel_button.toggled.connect(
            lambda on: action.trigger() if on != dock.isVisible() else None
        )
        action.toggled.connect(self.panel_button.setChecked)
        self._view_bar.addWidget(self.panel_button)

    def _switch_mode(self, name: str) -> None:
        self.refresh(self.session.do(SetMode(name)))
        self._prepare_then_refresh()

    def _mode_param_changed(self, name: str, value: str) -> None:
        self.refresh(self.session.do(SetModeParam(name, value)))
        self._prepare_then_refresh()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _install_shortcuts(self) -> None:
        """Declare every key once. The help panel reads this same table."""
        self.help = ShortcutHelp(self, "viewer")
        self.help.apply(
            [
                Binding("Left", "crosshair -x", lambda: self._nudge(0, -1), group="navigate"),
                Binding("Right", "crosshair +x", lambda: self._nudge(0, 1), group="navigate"),
                Binding("Down", "crosshair -y", lambda: self._nudge(1, -1), group="navigate"),
                Binding("Up", "crosshair +y", lambda: self._nudge(1, 1), group="navigate"),
                Binding("PgDn", "crosshair -z", lambda: self._nudge(2, -1), group="navigate"),
                Binding("PgUp", "crosshair +z", lambda: self._nudge(2, 1), group="navigate"),
                Binding("click", "move the crosshair", None, group="navigate"),
                Binding("ctrl+click", "set the InstaCorr seed", None, group="navigate"),
                Binding("scroll", "step through slices", None, group="navigate"),
                Binding("click graph", "jump to that volume", None, group="time"),
                Binding(",", "previous volume", lambda: self._step_time(-1), group="time"),
                Binding(".", "next volume", lambda: self._step_time(1), group="time"),
                Binding("v", "play / pause", self._toggle_play, group="time"),
                Binding("1", "toggle axial", self._pane_buttons[Plane.AXIAL].toggle, group="view"),
                Binding(
                    "2", "toggle sagittal", self._pane_buttons[Plane.SAGITTAL].toggle, group="view"
                ),
                Binding(
                    "3", "toggle coronal", self._pane_buttons[Plane.CORONAL].toggle, group="view"
                ),
                Binding(
                    "g", "axial graph window", self._graph_buttons[Plane.AXIAL].toggle, group="view"
                ),
                Binding("p", "toggle the panel", self.panel_button.toggle, group="view"),
                Binding("[", "previous layer", lambda: self._cycle_layer(-1), group="layer"),
                Binding("]", "next layer", lambda: self._cycle_layer(1), group="layer"),
                Binding("space", "show / hide layer", self._toggle_visible, group="layer"),
                Binding("t", "threshold down", lambda: self._nudge_threshold(-0.05), group="layer"),
                Binding("T", "threshold up", lambda: self._nudge_threshold(0.05), group="layer"),
                Binding("c", "next colormap", self._cycle_colormap, group="layer"),
                Binding("s", "next sign mode", self._cycle_sign, group="layer"),
                Binding("a", "next alpha mode", self._cycle_alpha, group="layer"),
                Binding("b", "toggle boxed", self.boxed_check.toggle, group="layer"),
                Binding("ctrl+o", "read a directory", self._read_dialog, group="session"),
                Binding("ctrl+s", "save session script", self._save_script_dialog, group="session"),
                Binding("h", "this list", self.help.toggle, group="session"),
            ]
        )

    def current_key(self) -> str | None:
        row = self.layer_list.currentRow()
        keys = list(reversed(self.session.state.layers.keys))
        return keys[row] if 0 <= row < len(keys) else None

    def _apply(self, cls, **kwargs) -> None:
        key = self.current_key()
        if key is None:
            return
        self.refresh(self.session.do(cls(key=key, **kwargs)))

    def _on_pick(self, plane: Plane, row: int, col: int, *, seed: bool = False) -> None:
        grid = self.session.state.grid
        if grid is None:
            return
        layout = plane_layout(grid.affine, plane)
        ijk = layout.to_ijk(row, col, self.session.state.crosshair, grid.shape)
        # A seed click moves the crosshair as well: you clicked a voxel, and
        # leaving the crosshair behind means the graph and the readout describe
        # somewhere else. Two commands rather than one so SET_SEED stays a
        # primitive that a script can use without moving the view.
        dirty = self.session.do(SetIJK(*ijk))
        if seed:
            dirty |= self.session.do(SetSeed(*ijk))
        self.refresh(dirty)
        if seed and self.session.mode.needs_prepare:
            # First seed after a mode switch: the mode deferred, so preparation
            # happens here, on a worker, with the progress bar up.
            self._prepare_then_refresh()

    def _nudge(self, axis: int, delta: int) -> None:
        ijk = list(self.session.state.crosshair)
        ijk[axis] += delta
        self.refresh(self.session.do(SetIJK(*ijk)))

    def _step_slice(self, plane: Plane, delta: int) -> None:
        grid = self.session.state.grid
        if grid is not None:
            self._nudge(plane_layout(grid.affine, plane).fixed, delta)

    def _step_time(self, delta: int) -> None:
        hi = self.session.state.max_time_index()
        if hi <= 0:
            return
        nxt = (self.session.state.time_index + delta) % (hi + 1)
        self.refresh(self.session.do(SetIndex(nxt)))

    def _time_spin_changed(self, value: int) -> None:
        self.refresh(self.session.do(SetIndex(int(value))))

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

    def _range_changed(self, lo: float, hi: float) -> None:
        key = self.current_key()
        if key is not None:
            self.refresh(self.session.do(SetRange(key, lo, hi)))

    def _threshold_changed(self, value: float) -> None:
        key = self.current_key()
        if key is not None:
            self.refresh(self.session.do(SetThreshold(key, value)))

    def _autorange(self) -> None:
        key = self.current_key()
        if key is None:
            return
        from fastfuncstuff.viewer.session import derive_range

        volume = self.session.volume(key)
        lo, hi = derive_range(volume)
        self.refresh(self.session.do(SetRange(key, float(lo), float(hi))))

    def _opacity_changed(self, value: int) -> None:
        key = self.current_key()
        if key is None:
            return
        self.opacity_label.setText(f"{value}%")
        self.refresh(self.session.do(SetLayerOpacity(key, value / 100.0)))

    def _nudge_threshold(self, frac: float) -> None:
        slider = self.rangebar.slider
        slider.setValue(max(0, min(1000, slider.value() + int(frac * 1000))))

    def _cycle_combo(self, box: QtWidgets.QComboBox, cls, kwarg: str) -> None:
        i = (box.currentIndex() + 1) % max(box.count(), 1)
        box.setCurrentIndex(i)
        self._apply(cls, **{kwarg: box.currentText()})

    def _cycle_alpha(self) -> None:
        self._cycle_combo(self.alpha_box, SetAlpha, "mode")

    def _cycle_sign(self) -> None:
        self._cycle_combo(self.sign_box, SetSign, "mode")

    def _cycle_colormap(self) -> None:
        self._cycle_combo(self.cmap_box, SetColormap, "colormap")

    def _save_script_dialog(self) -> None:
        self.mode_panel.flush_now()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session script", "session.ffs", "Scripts (*.ffs);;All (*)"
        )
        if path:
            self.session.save_script(path, header="nexus session")
            self.statusBar().showMessage(f"wrote {path}", 4000)

    def open_path(self, path: str | Path) -> None:
        """Open one dataset: as the underlay if there is none, else on top."""
        cmd = SetUnderlay if not len(self.session.state.layers) else AddOverlay
        self.refresh(self.session.do(cmd(str(path))))
        self._sync_layer_list()

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        self.coord_label = QtWidgets.QLabel("")
        self.mode_label = QtWidgets.QLabel("")
        self.value_label = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(190)
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.statusBar().addWidget(self.coord_label)
        self.statusBar().addWidget(self.mode_label, 1)
        self.statusBar().addWidget(self.progress)
        self.statusBar().addPermanentWidget(self.value_label)

    # -- mode preparation ----------------------------------------------
    def _on_prepare_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(fraction * 100))
        self.progress.setFormat(f"{message}  %p%")

    def _on_prepare_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        # Disabled rather than queued: a drag would otherwise stack up several
        # multi-second preparations whose results land out of order.
        self.mode_panel.setEnabled(not busy)
        self.mode_box.setEnabled(not busy)
        if not busy:
            self.progress.reset()

    def _prepare_then_refresh(self) -> None:
        """Run the mode's slow half on a worker, then install its overlay."""
        run_when_ready(
            self.runner,
            self.session.mode,
            on_ready=lambda: self.refresh(
                self.session.refresh_mode() | Aspect.LAYERS | Aspect.SLICES
            ),
            on_error=lambda msg: self.statusBar().showMessage(f"mode failed: {msg}", 8000),
        )

    def _on_layer_loaded(self, key: str) -> None:
        self.session.invalidate(key)
        self._sync_layer_list()
        self.refresh(Aspect.SLICES | Aspect.GRAPH)

    def refresh(self, dirty: Aspect) -> None:
        if dirty is Aspect.NOTHING:
            return
        if dirty & (Aspect.LAYERS | Aspect.GRID):
            self._sync_layer_list()
            self._sync_pickers()
            self._sync_mode_panel()
        elif dirty & (Aspect.COLORMAP | Aspect.THRESHOLD | Aspect.SLICES):
            # The bar is a view of colormap, range and threshold, so it has to
            # follow those aspects and not only LAYERS -- listening for the
            # wrong one is what left it showing the previous colour scale.
            self._sync_layer_controls()
        if dirty & (
            Aspect.SLICES
            | Aspect.COLORMAP
            | Aspect.THRESHOLD
            | Aspect.TIME
            | Aspect.GRID
            | Aspect.CROSSHAIR
        ):
            # CROSSHAIR belongs here: the crosshair position *is* which slice
            # each pane shows. Leaving it out only moved the drawn lines, so a
            # click in one pane left the other two on their previous slices --
            # and stepping time, which did force a redraw, made them all
            # "jump" as they caught up.
            self._redraw_panes(force=bool(dirty & ~(Aspect.CROSSHAIR | Aspect.GRAPH)))
        if dirty & (Aspect.CROSSHAIR | Aspect.GRID):
            self._redraw_crosshairs()
        if dirty & (Aspect.CROSSHAIR | Aspect.GRAPH | Aspect.TIME | Aspect.LAYERS):
            self._refresh_graphs()
        self._sync_readout()

    def _redraw_panes(self, *, force: bool = True) -> None:
        """Re-slice the visible panes.

        With ``force`` false only panes whose slice actually moved are
        re-rendered, so dragging the crosshair across the axial view redraws
        the two panes that changed rather than all three.
        """
        for plane, pane in self._panes.items():
            if not self._pane_buttons[plane].isChecked():
                continue
            if not force and pane.position == plane_position(self.session.state, plane):
                continue
            pane.set_pane(render_plane(self.session, plane))
        self._redraw_crosshairs()

    def _redraw_crosshairs(self) -> None:
        grid = self.session.state.grid
        if grid is None:
            return
        ijk = self.session.state.crosshair
        for plane, pane in self._panes.items():
            layout = plane_layout(grid.affine, plane)
            pane.set_layout(layout)
            pane.set_crosshair(*layout.to_image(ijk, grid.shape))

    def _refresh_graphs(self) -> None:
        for win in self._graphs.values():
            if win.isVisible():
                win.refresh()

    def _sync_pickers(self) -> None:
        """Point each picker at the layer it currently governs.

        A picker reading "(none)" while that layer is on screen is the same
        confusion as one naming a file while nothing is displayed -- in both
        cases the control disagrees with the view.
        """
        base = self.session.state.layers.base
        overlay = self.session.state.layers.overlay
        for box, layer in ((self.underlay_box, base), (self.overlay_box, overlay)):
            box.blockSignals(True)
            index = 0  # (none)
            if layer is not None and not layer.is_computed:
                for i in range(1, box.count()):
                    entry = box.itemData(i)
                    if entry is not None and str(entry.path) == layer.path:
                        index = i
                        break
                else:
                    # Loaded from outside the catalog: name it rather than lie.
                    box.addItem(layer.name, userData=None)
                    index = box.count() - 1
            box.setCurrentIndex(index)
            box.blockSignals(False)

    def _sync_layer_list(self) -> None:
        row = self.layer_list.currentRow()
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in reversed(list(self.session.state.layers)):
            try:
                pending = self.session.store.get(layer.key).pending
            except KeyError:
                pending = False
            mark = "▣" if layer.visible else "▢"
            tag = " ·computed" if layer.is_computed else (" ·loading" if pending else "")
            self.layer_list.addItem(f"{mark} {layer.name}{tag}")
        self.layer_list.blockSignals(False)
        if self.layer_list.count():
            self.layer_list.setCurrentRow(max(0, min(row, self.layer_list.count() - 1)))
        self._sync_layer_controls()

    def _sync_mode_panel(self) -> None:
        mode = self.session.mode
        self.mode_panel.rebuild(mode.controls(), mode.params)
        has = bool(mode.controls())
        self.mode_head.setVisible(has)
        self.mode_panel.setVisible(has)
        idx = self.mode_box.findData(mode.name)
        if idx >= 0 and idx != self.mode_box.currentIndex():
            self.mode_box.blockSignals(True)
            self.mode_box.setCurrentIndex(idx)
            self.mode_box.blockSignals(False)

    def _sync_layer_controls(self) -> None:
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
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(round(layer.opacity * 100)))
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{int(round(layer.opacity * 100))}%")
        self.rangebar.configure(layer)
        # One slider in every mode; only what its numbers mean moves.
        kind = self.session.mode.overlay_kind if layer.is_computed else OverlayKind.VALUE
        self.thr_head.setText(
            {
                OverlayKind.STATISTIC: "THRESH stat",
                OverlayKind.CORRELATION: "THRESH r",
                OverlayKind.COMPONENT: "THRESH z",
            }.get(kind, "THRESH")
        )

    def _sync_readout(self) -> None:
        st = self.session.state
        if st.grid is None:
            self.coord_label.setText("no data — press READ")
            return
        i, j, k = st.crosshair
        mm = st.crosshair_mm or (0.0, 0.0, 0.0)
        self.coord_label.setText(
            f"ijk {i:>3d} {j:>3d} {k:>3d}   xyz {mm[0]:>7.1f} {mm[1]:>7.1f} {mm[2]:>7.1f}"
        )
        hi = st.max_time_index()
        self.time_spin.blockSignals(True)
        self.time_spin.setRange(0, max(hi, 0))
        self.time_spin.setValue(st.time_index)
        self.time_spin.setEnabled(hi > 0)
        self.time_spin.blockSignals(False)
        self.time_label.setText(f"/ {hi}" if hi else "")
        self.mode_label.setText(self.session.mode.status())
        parts: list[str] = []
        for layer in reversed(list(st.layers)):
            if not layer.visible:
                continue
            vol = self.session.display_volume(layer.key)
            if vol is None:
                continue
            val = voxel_value(vol, st.grid, layer.affine, st.crosshair)
            parts.append(f"{layer.name}={'--' if val is None else f'{val:.4g}'}")
        self.value_label.setText("   ".join(parts[:3]))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self._play.stop()
        self.runner.wait(2000)
        for win in self._graphs.values():
            win.close()
        self.session.close()
        super().closeEvent(event)


def launch(
    paths: list[str],
    *,
    device: str | None = None,
    script: str | None = None,
    directory: str | None = None,
) -> int:
    """Open a window and run the Qt loop."""
    from fastfuncstuff.cli_utils import setup_device

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=setup_device(device))
    win = ViewerWindow(session)

    # A directory, or the one the first dataset lives in, so the pickers are
    # populated before anyone reaches for Read.
    start = directory or (str(Path(paths[0]).parent) if paths else None)
    if start:
        win.read_directory(start)
    for p in paths:
        win.open_path(p)
    if script:
        win.refresh(session.run_script(Path(script).read_text()))
    win.show()
    return app.exec()


__all__ = ["ViewerWindow", "launch"]
