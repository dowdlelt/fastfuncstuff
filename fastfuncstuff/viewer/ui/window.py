"""The controller: a data selector, driving N companion windows.

The core is one row -- Read, Underlay, Overlay, +1, Mode. That is the whole
viewer; images and graphs are things you turn on beside it.

This window holds no brain. Every image and every graph is a top-level window
described by a :class:`~viewer.viewports.Viewport` and reconciled by
:class:`~viewer.ui.manager.WindowManager`, which is what makes two views of the
same plane, a per-window zoom, a parked reference slice and a layout you can
replay all the same mechanism rather than four. The controller's job is to hold
the things there is exactly one of: what is loaded, what the stack looks like,
how the selected layer is coloured, and what the mode is doing.

The other rule kept from the single-window layout: **the window contains no
mode-specific code.** It asks the active mode what controls to show and renders
whatever it declares.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.catalog import CatalogEntry
from fastfuncstuff.viewer.colormap import available_colormaps
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.layers import AlphaMode, SignMode
from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import OverlayKind
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import voxel_value
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.colorbar import RangeBar
from fastfuncstuff.viewer.ui.controls import ControlPanel
from fastfuncstuff.viewer.ui.manager import WindowManager
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp
from fastfuncstuff.viewer.ui.theme import MONO, key_label, stylesheet
from fastfuncstuff.viewer.ui.work import PreparationRunner, run_when_ready
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import (
    AddOverlay,
    Read,
    SelectLayer,
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
    SetTheme,
    SetThreshold,
    SetThresholdIndex,
    SetTimeLinked,
    SetUnderlay,
    SetVolume,
)

#: Shown when no dataset is chosen. A picker that names a file while nothing is
#: displayed reads as a load that failed.
NONE_LABEL = "(none)"


class _Bridge(QtCore.QObject):
    """Marshals worker-thread load completions onto the GUI thread."""

    loaded = QtCore.Signal(str)


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, session: ViewerSession) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle("nexus")
        self.setStyleSheet(stylesheet())
        self.resize(430, 820)

        self._bridge = _Bridge()
        self._bridge.loaded.connect(
            self._on_layer_loaded, QtCore.Qt.ConnectionType.QueuedConnection
        )
        session.on_loaded(self._bridge.loaded.emit)

        self.manager = WindowManager(session, self._dispatch, self)

        self._build_selector()
        self._build_panel()
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

        session.default_layout()
        self._apply_theme()
        self.refresh(Aspect.ALL)

    def _dispatch(self, cmd: Command) -> None:
        """The single entry point every widget and window mutates state through.

        Mode preparation is kicked off here rather than at each call site: the
        seed can be set from any image window, and the first seed after a mode
        switch is exactly the one whose filtering would freeze the GUI if it
        ran inline.
        """
        self.refresh(self.session.do(cmd))
        if isinstance(cmd, SetSeed) and self.session.mode.needs_prepare:
            self._prepare_then_refresh()

    # ------------------------------------------------------------------
    # the core: read / underlay / overlay / +1 / mode
    # ------------------------------------------------------------------
    def _build_selector(self) -> None:
        bar = QtWidgets.QToolBar("selector")
        bar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        self.read_button = QtWidgets.QPushButton(key_label("READ", "^O"))
        self.read_button.setToolTip("Read a directory into the pickers (ctrl+O)")
        self.read_button.clicked.connect(self._read_dialog)
        bar.addWidget(self.read_button)

        self.dir_label = QtWidgets.QLabel("no directory")
        bar.addWidget(self.dir_label)

        picks = QtWidgets.QToolBar("data")
        picks.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, picks)

        grid_host = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)

        grid.addWidget(self._head("UNDERLAY"), 0, 0)
        self.underlay_box = QtWidgets.QComboBox()
        self.underlay_box.setFont(QtGui.QFont(MONO))
        self.underlay_box.addItem(NONE_LABEL, userData=None)
        self.underlay_box.activated.connect(lambda _: self._pick(self.underlay_box, SetUnderlay))
        grid.addWidget(self.underlay_box, 0, 1)

        grid.addWidget(self._head("OVERLAY"), 1, 0)
        self.overlay_box = QtWidgets.QComboBox()
        self.overlay_box.setFont(QtGui.QFont(MONO))
        self.overlay_box.addItem(NONE_LABEL, userData=None)
        self.overlay_box.activated.connect(lambda _: self._pick(self.overlay_box, SetOverlay))
        grid.addWidget(self.overlay_box, 1, 1)

        self.plus_button = QtWidgets.QPushButton("[+]1")
        self.plus_button.setToolTip("Add the selected dataset on top, keeping the current overlay")
        self.plus_button.clicked.connect(lambda: self._pick(self.overlay_box, AddOverlay))
        grid.addWidget(self.plus_button, 1, 2)

        grid.addWidget(self._head("MODE"), 2, 0)
        self.mode_box = QtWidgets.QComboBox()
        labels = registry.labels()
        for name in registry.names():
            self.mode_box.addItem(labels[name], userData=name)
        self.mode_box.setCurrentIndex(self.mode_box.findData(self.session.mode.name))
        self.mode_box.activated.connect(lambda _: self._switch_mode(self.mode_box.currentData()))
        grid.addWidget(self.mode_box, 2, 1)
        grid.setColumnStretch(1, 1)
        picks.addWidget(grid_host)

        self._build_window_bar()

    def _build_window_bar(self) -> None:
        """Open and arrange the companion windows."""
        bar = QtWidgets.QToolBar("windows")
        bar.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)
        bar.addWidget(self._head("WINDOWS"))

        for text, key, tip, slot in (
            ("+IMAGE", "n", "Open another image window", self._new_image),
            ("+GRAPH", "N", "Open a graph window", self._new_graph),
            ("TILE", "f", "Lay every window out on a grid", self._tile),
            ("STACK", "F", "Stagger the windows so each title bar is reachable", self._cascade),
            ("RAISE", "r", "Bring every companion window to the front", self._raise_all),
        ):
            b = QtWidgets.QPushButton(key_label(text, key))
            b.setToolTip(f"{tip} ({key})")
            b.clicked.connect(slot)
            bar.addWidget(b)

        bar.addSeparator()
        # Names the palette you would switch *to*, not the one you are in: a
        # button labelled with the current state reads as a status light and
        # gets pressed by people who wanted it to stay that way.
        self.theme_button = QtWidgets.QPushButton("")
        self.theme_button.setToolTip("Switch between the dark and light palette (d)")
        self.theme_button.clicked.connect(self._toggle_theme)
        bar.addWidget(self.theme_button)

        bar.addSeparator()
        bar.addWidget(self._head("T"))
        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.setToolTip("Jump to a volume ( , and . step, v plays )")
        self.time_spin.setKeyboardTracking(False)
        self.time_spin.setMaximumWidth(84)
        self.time_spin.valueChanged.connect(self._time_spin_changed)
        bar.addWidget(self.time_spin)
        self.time_label = QtWidgets.QLabel("")
        bar.addWidget(self.time_label)

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
        self._dispatch(cls(str(entry.path)))
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
    # companion windows
    # ------------------------------------------------------------------
    def _new_image(self) -> None:
        self.manager.open(ViewKind.IMAGE, self._next_plane())
        self.refresh(Aspect.VIEWPORTS | Aspect.SLICES)

    def _new_graph(self) -> None:
        self.manager.open(ViewKind.GRAPH, self._next_plane())
        self.refresh(Aspect.VIEWPORTS | Aspect.GRAPH)

    def _next_plane(self) -> Plane:
        """Offer the plane that is not already on screen, then wrap.

        Opening a second image window onto the plane you are already looking at
        is almost never what was meant the first few times, and is one keypress
        away when it is.
        """
        shown = [v.plane for v in self.session.state.viewports.images]
        for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
            if plane not in shown:
                return plane
        return Plane.AXIAL

    def _tile(self) -> None:
        self.manager.tile(self)

    def _cascade(self) -> None:
        self.manager.cascade(self)

    def _raise_all(self) -> None:
        self.manager.raise_all()

    def _toggle_theme(self) -> None:
        self._dispatch(SetTheme("light" if self.session.state.theme == "dark" else "dark"))

    def _apply_theme(self) -> None:
        """Push the palette into every window, including this one."""
        name = self.session.state.theme
        theme.set_theme(name)
        self.setStyleSheet(stylesheet())
        self.rangebar.restyle()
        self.theme_button.setText(key_label("LIGHT" if name == "dark" else "DARK", "d"))
        self.manager.restyle()

    # ------------------------------------------------------------------
    # the panel: layers, layer controls, mode controls
    # ------------------------------------------------------------------
    def _build_panel(self) -> None:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(9, 9, 9, 9)
        v.setSpacing(8)

        v.addWidget(self._head("LAYERS  [ / ]"))
        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setMaximumHeight(190)
        self.layer_list.setToolTip(
            "[ and ] step through the stack; space hides a layer.\n"
            "A soloed image window draws whichever one is selected here."
        )
        self.layer_list.currentRowChanged.connect(self._row_selected)
        v.addWidget(self.layer_list)

        form = QtWidgets.QFormLayout()
        form.setSpacing(7)

        # A stats bucket is a stack of named contrasts, and the names are in
        # the header -- ffs and 3dDeconvolve both write them. Showing "volume
        # 3" instead of "Faces#0_Coef" makes you go and run 3dinfo to find out
        # what you are looking at, which is the one thing a viewer is for.
        self.brick_box = QtWidgets.QComboBox()
        self.brick_box.setFont(QtGui.QFont(MONO))
        self.brick_box.setToolTip("Which sub-brick this layer colours by")
        self.brick_box.activated.connect(
            lambda i: self._apply(SetVolume, index=int(self.brick_box.itemData(i)))
        )
        self.brick_head = self._head("OLAY")
        form.addRow(self.brick_head, self.brick_box)

        self.thrbrick_box = QtWidgets.QComboBox()
        self.thrbrick_box.setFont(QtGui.QFont(MONO))
        self.thrbrick_box.setToolTip(
            "Which sub-brick the threshold reads.\n"
            "The stats case is colouring by a coefficient and thresholding on its t."
        )
        self.thrbrick_box.activated.connect(
            lambda i: self._apply(SetThresholdIndex, index=self.thrbrick_box.itemData(i))
        )
        self.thrbrick_head = self._head("THR ON")
        form.addRow(self.thrbrick_head, self.thrbrick_box)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(available_colormaps())
        self.cmap_box.activated.connect(
            lambda _: self._apply(SetColormap, colormap=self.cmap_box.currentText())
        )
        form.addRow(self._head(key_label("COLOR", "c")), self.cmap_box)

        self.sign_box = QtWidgets.QComboBox()
        self.sign_box.addItems([m.value for m in SignMode])
        self.sign_box.activated.connect(
            lambda _: self._apply(SetSign, mode=self.sign_box.currentText())
        )
        form.addRow(self._head(key_label("SIGN", "s")), self.sign_box)

        self.alpha_box = QtWidgets.QComboBox()
        self.alpha_box.addItems([m.value for m in AlphaMode])
        self.alpha_box.activated.connect(
            lambda _: self._apply(SetAlpha, mode=self.alpha_box.currentText())
        )
        form.addRow(self._head(key_label("ALPHA", "a")), self.alpha_box)

        # Min, threshold and max are edited on the bar itself. Splitting the
        # number from the picture of the number is what let the bar go stale.
        self.thr_head = self._head(key_label("THRESH", "t"))
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

        self.boxed_check = QtWidgets.QCheckBox(key_label("boxed", "b"))
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

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        self.setCentralWidget(scroll)

    def _switch_mode(self, name: str) -> None:
        self._dispatch(SetMode(name))
        self._prepare_then_refresh()

    def _mode_param_changed(self, name: str, value: str) -> None:
        self._dispatch(SetModeParam(name, value))
        self._prepare_then_refresh()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _install_shortcuts(self) -> None:
        """Declare every key once. The help panel reads this same table."""
        self.help = ShortcutHelp(self, "nexus")
        self.help.apply(
            [
                Binding("Left", "crosshair -x", lambda: self._nudge(0, -1), group="navigate"),
                Binding("Right", "crosshair +x", lambda: self._nudge(0, 1), group="navigate"),
                Binding("Down", "crosshair -y", lambda: self._nudge(1, -1), group="navigate"),
                Binding("Up", "crosshair +y", lambda: self._nudge(1, 1), group="navigate"),
                Binding("PgDn", "crosshair -z", lambda: self._nudge(2, -1), group="navigate"),
                Binding("PgUp", "crosshair +z", lambda: self._nudge(2, 1), group="navigate"),
                Binding("n", "open an image window", self._new_image, group="windows"),
                Binding("N", "open a graph window", self._new_graph, group="windows"),
                Binding("f", "tile every window", self._tile, group="windows"),
                Binding("F", "stagger every window", self._cascade, group="windows"),
                Binding("r", "raise every window", self._raise_all, group="windows"),
                Binding("d", "dark / light palette", self._toggle_theme, group="windows"),
                Binding(",", "previous volume", lambda: self._step_time(-1), group="time"),
                Binding(".", "next volume", lambda: self._step_time(1), group="time"),
                Binding("v", "play / pause", self._toggle_play, group="time"),
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
        layer = self.session.state.selected_layer()
        return None if layer is None else layer.key

    def _apply(self, cls, **kwargs) -> None:
        key = self.current_key()
        if key is None:
            return
        self._dispatch(cls(key=key, **kwargs))

    def _row_selected(self, row: int) -> None:
        keys = list(reversed(self.session.state.layers.keys))
        if 0 <= row < len(keys):
            self._dispatch(SelectLayer(keys[row]))

    def _nudge(self, axis: int, delta: int) -> None:
        ijk = list(self.session.state.crosshair)
        ijk[axis] += delta
        self._dispatch(SetIJK(*ijk))

    def _step_time(self, delta: int) -> None:
        hi = self.session.state.max_time_index()
        if hi <= 0:
            return
        nxt = (self.session.state.time_index + delta) % (hi + 1)
        self._dispatch(SetIndex(nxt))

    def _time_spin_changed(self, value: int) -> None:
        self._dispatch(SetIndex(int(value)))

    def _toggle_play(self) -> None:
        self._play.stop() if self._play.isActive() else self._play.start()

    def _toggle_visible(self) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        self._dispatch(SetLayerVisible(key, not layer.visible))
        self._sync_layer_list()

    def _cycle_layer(self, delta: int) -> None:
        n = self.layer_list.count()
        if n:
            self.layer_list.setCurrentRow((self.layer_list.currentRow() + delta) % n)

    def _range_changed(self, lo: float, hi: float) -> None:
        key = self.current_key()
        if key is not None:
            self._dispatch(SetRange(key, lo, hi))

    def _threshold_changed(self, value: float) -> None:
        key = self.current_key()
        if key is not None:
            self._dispatch(SetThreshold(key, value))

    def _autorange(self) -> None:
        key = self.current_key()
        if key is None:
            return
        from fastfuncstuff.viewer.session import derive_range

        volume = self.session.volume(key)
        lo, hi = derive_range(volume)
        self._dispatch(SetRange(key, float(lo), float(hi)))

    def _opacity_changed(self, value: int) -> None:
        key = self.current_key()
        if key is None:
            return
        self.opacity_label.setText(f"{value}%")
        self._dispatch(SetLayerOpacity(key, value / 100.0))

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
        self._dispatch(cmd(str(path)))
        self._sync_layer_list()

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        self.coord_label = QtWidgets.QLabel("")
        self.coord_label.setObjectName("value")
        self.mode_label = QtWidgets.QLabel("")
        self.value_label = QtWidgets.QLabel("")
        self.value_label.setObjectName("value")
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
        if dirty & Aspect.THEME:
            self._apply_theme()
        if dirty & (Aspect.LAYERS | Aspect.GRID):
            self._sync_layer_list()
            self._sync_pickers()
            self._sync_mode_panel()
        elif dirty & (Aspect.COLORMAP | Aspect.THRESHOLD | Aspect.SLICES):
            # The bar is a view of colormap, range and threshold, so it has to
            # follow those aspects and not only LAYERS -- listening for the
            # wrong one is what left it showing the previous colour scale.
            self._sync_layer_controls()
        # Windows first: a viewport that has just appeared has to exist before
        # anything tries to draw into it.
        if dirty & (Aspect.VIEWPORTS | Aspect.LAYERS | Aspect.GRID):
            self.manager.sync()
        self.manager.redraw(dirty)
        self._sync_readout()

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
        selected = self.current_key()
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        keys = list(reversed(self.session.state.layers.keys))
        for layer in reversed(list(self.session.state.layers)):
            try:
                pending = self.session.store.get(layer.key).pending
            except KeyError:
                pending = False
            mark = "▣" if layer.visible else "▢"
            tag = " ·computed" if layer.is_computed else (" ·loading" if pending else "")
            self.layer_list.addItem(f"{mark} {layer.name}{tag}")
        if keys:
            row = keys.index(selected) if selected in keys else 0
            self.layer_list.setCurrentRow(row)
        # Unblocked only after the row is set. A sync that writes back into
        # state is a refresh that dispatches, which puts a SELECT_LAYER into
        # the recording for every repaint and can recurse.
        self.layer_list.blockSignals(False)
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
        self._sync_brick_pickers(layer)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(round(layer.opacity * 100)))
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{int(round(layer.opacity * 100))}%")
        self.rangebar.configure(layer)
        # One slider in every mode; only what its numbers mean moves.
        kind = self.session.mode.overlay_kind if layer.is_computed else OverlayKind.VALUE
        self.thr_head.setText(
            {
                OverlayKind.STATISTIC: key_label("THRESH", "t") + " stat",
                OverlayKind.CORRELATION: key_label("THRESH", "t") + " r",
                OverlayKind.COMPONENT: key_label("THRESH", "t") + " z",
            }.get(kind, key_label("THRESH", "t"))
        )

    def _sync_brick_pickers(self, layer) -> None:
        """Show the sub-brick pickers only where sub-bricks are a choice.

        A time series is scrubbed by the T control and its "sub-bricks" are
        time points, so a picker listing four hundred of them is noise. What
        wants naming is the other 4-D case: a bucket of unrelated contrasts.
        """
        show = layer.n_volumes > 1 and not layer.time_linked
        for head, box in (
            (self.brick_head, self.brick_box),
            (self.thrbrick_head, self.thrbrick_box),
        ):
            head.setVisible(show)
            box.setVisible(show)
        if not show:
            return
        names = [layer.sub_brick(i) for i in range(layer.n_volumes)]
        self.brick_box.blockSignals(True)
        self.brick_box.clear()
        for i, name in enumerate(names):
            self.brick_box.addItem(name, userData=i)
        self.brick_box.setCurrentIndex(min(layer.volume_index, layer.n_volumes - 1))
        self.brick_box.blockSignals(False)

        self.thrbrick_box.blockSignals(True)
        self.thrbrick_box.clear()
        # "same" rather than a blank row: thresholding on the displayed
        # sub-brick is a real choice, and the one a plain map wants.
        self.thrbrick_box.addItem("same as OLAY", userData=None)
        for i, name in enumerate(names):
            self.thrbrick_box.addItem(name, userData=i)
        self.thrbrick_box.setCurrentIndex(
            0 if layer.threshold_index is None else layer.threshold_index + 1
        )
        self.thrbrick_box.blockSignals(False)

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
            # Only where the header actually named something: appending "#0" to
            # every 3-D anatomy would be noise dressed as information.
            tag = f" {layer.sub_brick()}" if layer.labels else ""
            parts.append(f"{layer.name}{tag}={'--' if val is None else f'{val:.4g}'}")
        self.value_label.setText("   ".join(parts[:3]))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        # The controller is the session; closing it closes the companions too,
        # or they linger with nothing driving them.
        self._play.stop()
        self.runner.wait(2000)
        self.manager.close_all()
        self.session.close()
        super().closeEvent(event)


def launch(
    paths: list[str],
    *,
    device: str | None = None,
    script: str | None = None,
    directory: str | None = None,
) -> int:
    """Open the controller and run the Qt loop."""
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
    win.manager.tile(win)
    return app.exec()


__all__ = ["ViewerWindow", "launch"]
