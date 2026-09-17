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

**Controllers.** AFNI's A, B, C: one controller per thing being compared -- two
subjects, a subject and a template, raw and denoised -- each with its own
directory, stack, mode and windows, and all of them following one crosshair.
Here a controller is a tab. The panel widgets exist once and always show the
active tab, so a tab is only a :class:`ViewerSession` plus the windows that
draw it; nothing about the panel had to be taught that there is more than one.
The crosshair is linked in millimetres, not voxels, because two subjects are
two grids, and time is linked by index.
"""

from __future__ import annotations

import os
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer import catalog as catalog_mod
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
from fastfuncstuff.viewer.ui.colorbar import RangeBar, thresholds_itself
from fastfuncstuff.viewer.ui.controls import ControlPanel
from fastfuncstuff.viewer.ui.manager import WindowManager
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.ui.theme import MONO, key_label, stylesheet
from fastfuncstuff.viewer.ui.tooldialog import ToolDialog
from fastfuncstuff.viewer.ui.work import PreparationRunner, run_when_ready
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import (
    AddOverlay,
    ModeAction,
    MoveLayer,
    Read,
    RemoveLayer,
    SelectLayer,
    SetAlpha,
    SetBoxed,
    SetCarpetOrder,
    SetColormap,
    SetIJK,
    SetIndex,
    SetLayerOpacity,
    SetLayerRoi,
    SetLayerVisible,
    SetMatrixOrder,
    SetMode,
    SetModeParam,
    SetOverlay,
    SetRange,
    SetRangeMirror,
    SetSeed,
    SetSign,
    SetTheme,
    SetThreshold,
    SetThresholdFollow,
    SetThresholdIndex,
    SetTimeLinked,
    SetUnderlay,
    SetViewDetrend,
    SetViewRois,
    SetViewScaling,
    SetViewTraces,
    SetVolume,
    SetXYZ,
)

#: Commands that change what a built window draws, as opposed to what is around
#: it. A carpet and a matrix are both seconds of arithmetic, so these rebuild
#: and everything else only marks the picture stale.
BUILT_SETTINGS = (
    SetViewDetrend,
    SetCarpetOrder,
    SetMatrixOrder,
    SetViewRois,
    SetMatrixOrder,
    SetViewRois,
    SetViewScaling,
    SetViewTraces,
)

#: Shown when no dataset is chosen. A picker that names a file while nothing is
#: displayed reads as a load that failed.
NONE_LABEL = "(none)"


class _Bridge(QtCore.QObject):
    """Marshals worker-thread load completions onto the GUI thread."""

    loaded = QtCore.Signal(str)


#: Controller names, in the order they are handed out. Ten is past the point
#: where tabs stop being a comparison and start being a filing system.
LETTERS = "ABCDEFGHIJ"


@dataclass(eq=False)
class Controller:
    """One tab: a session, and the windows that draw it."""

    letter: str
    session: ViewerSession
    manager: WindowManager
    bridge: _Bridge


#: How long a directory has to stay quiet before a rescan runs.
RESCAN_QUIET_MS = 1200
BACKGROUND = QtCore.Qt.ItemDataRole.BackgroundRole
FOREGROUND = QtCore.Qt.ItemDataRole.ForegroundRole


class ViewerWindow(QtWidgets.QMainWindow):
    #: ``(session, future, manual)`` from the rescan worker, delivered on the GUI thread.
    _rescanned = QtCore.Signal(object)

    def __init__(self, session: ViewerSession) -> None:
        super().__init__()
        self.setWindowTitle("nexus")
        self.setStyleSheet(stylesheet())
        self.resize(430, 820)

        # Watching the directory. A pipeline writes a file in bursts, so a
        # change starts a quiet period and the rescan runs once it ends,
        # rather than once per write.
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(lambda _: self._schedule_rescan())
        self._watcher.fileChanged.connect(lambda _: self._schedule_rescan())
        self._rescan_timer = QtCore.QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(RESCAN_QUIET_MS)
        self._rescan_timer.timeout.connect(self._start_rescan)
        self._rescan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rescan")
        self._rescan_busy = False
        self._rescan_again = False
        self._rescanned.connect(self._on_rescanned, QtCore.Qt.ConnectionType.QueuedConnection)

        self.controllers: list[Controller] = []
        #: Whether controllers follow each other's crosshair and time index.
        #: On by default, as in AFNI: comparing two subjects at the same place
        #: is the reason to open a second controller at all.
        self.linked = True
        self._active = self._add_controller(session)

        self._build_selector()
        self._build_panel()
        self._build_statusbar()
        self._install_shortcuts()
        keep_keys_for_shortcuts(self)

        # Mode preparation runs on a worker; the mode is told to defer so a
        # seed click never runs seconds of filtering inside the click handler.
        self.runner = PreparationRunner(self)
        self.runner.progress.connect(self._on_prepare_progress)
        self.runner.busy_changed.connect(self._on_prepare_busy)
        self._rebuild_queue: list[tuple[Controller, str]] = []
        #: Open tool dialogs, by tool name. One per tool, reused on re-press.
        self._tool_dialogs: dict[str, ToolDialog] = {}
        # Deferred a turn so the job that just finished has handed its picture
        # over before the next one takes the worker.
        self.runner.finished.connect(lambda *_: QtCore.QTimer.singleShot(0, self._drain_rebuilds))

        self._play = QtCore.QTimer(self)
        self._play.setInterval(60)
        self._play.timeout.connect(lambda: self._step_time(1))

        self._sync_tabs()
        self._apply_theme()
        self.refresh(Aspect.ALL)

    # ------------------------------------------------------------------
    # controllers
    # ------------------------------------------------------------------
    @property
    def session(self) -> ViewerSession:
        """The active controller's session; what every panel widget acts on."""
        return self._active.session

    @property
    def manager(self) -> WindowManager:
        """The active controller's windows."""
        return self._active.manager

    @property
    def active(self) -> Controller:
        return self._active

    def _add_controller(self, session: ViewerSession) -> Controller:
        used = {c.letter for c in self.controllers}
        letter = next((ch for ch in LETTERS if ch not in used), f"{len(self.controllers) + 1}")
        bridge = _Bridge()
        manager = WindowManager(session, lambda cmd: None, self)
        ctl = Controller(letter=letter, session=session, manager=manager, bridge=bridge)
        # Each controller's windows dispatch into their own session. Bound per
        # controller rather than through ``self.session``, which is whichever
        # tab is active -- a click in B's axial window while A's tab is showing
        # would otherwise move A.
        manager._dispatch = lambda cmd, c=ctl: self._dispatch_from(c, cmd)
        manager.label = letter
        session.label = letter
        manager.rebuild_requested.connect(lambda vid, c=ctl: self._on(c, self.rebuild_view, vid))
        manager.rois_requested.connect(lambda vid, c=ctl: self._on(c, self._clusters_to_rois, vid))
        manager.mode_action_requested.connect(
            lambda name, c=ctl: self._on(c, self._mode_action, name)
        )
        manager.rows_selected.connect(
            lambda vid, a, b, c=ctl: self._on(c, self._carpet_rows_to_layer, vid, a, b)
        )
        bridge.loaded.connect(
            lambda key, c=ctl: self._on_layer_loaded(c, key),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        session.on_loaded(bridge.loaded.emit)
        # Mode preparation runs on a worker; the mode is told to defer so a
        # seed click never runs seconds of filtering inside the click handler.
        session.defer_mode_preparation = True
        session.mode.defer_preparation = True
        session.default_layout()
        self.controllers.append(ctl)
        return ctl

    def _on(self, ctl: Controller, slot, *args) -> None:
        """Run a controller-level slot with that controller active."""
        self.activate(ctl)
        slot(*args)

    def new_controller(self) -> Controller:
        """Open another controller, linked, on the directory the active one reads.

        Same directory because the usual next step is picking a different file
        from it -- another subject, the template -- and READ is one click away
        when it is not.
        """
        source = self._active.session
        session = ViewerSession(device=source.store.device)
        if source.state.theme != session.state.theme:
            session.do(SetTheme(source.state.theme))
        if source.catalog_dir is not None:
            session.do(Read(str(source.catalog_dir)))
        ctl = self._add_controller(session)
        self.tabs.blockSignals(True)
        self.tabs.addTab(ctl.letter)
        self.tabs.blockSignals(False)
        self._sync_tabs()
        ctl.manager.sync()
        ctl.manager.restyle()
        self.activate(ctl)
        # Carry the crosshair across on the way in, or the new windows open at
        # their own grid's centre while every other controller is elsewhere.
        if self.linked and source.state.crosshair_mm is not None:
            self._follow(ctl, source.state.crosshair_mm, source.state.time_index)
        self.refresh(Aspect.ALL)
        self._tile()
        return ctl

    def close_controller(self, ctl: Controller) -> bool:
        """Close one controller and its windows. The last one stays."""
        if len(self.controllers) <= 1:
            self.statusBar().showMessage("the last controller stays", 4000)
            return False
        if self.runner.busy:
            self.statusBar().showMessage("wait for the running job before closing a tab", 5000)
            return False
        index = self.controllers.index(ctl)
        ctl.manager.close_all()
        ctl.session.close()
        self.controllers.remove(ctl)
        self.tabs.blockSignals(True)
        self.tabs.removeTab(index)
        self.tabs.blockSignals(False)
        if ctl is self._active:
            self._active = self.controllers[max(0, index - 1)]
            self._show_active()
        self._sync_tabs()
        return True

    def activate(self, ctl: Controller) -> None:
        """Point the panel at one controller."""
        if ctl is self._active:
            return
        self._close_tool_dialogs()
        self._active = ctl
        self._show_active()

    def _show_active(self) -> None:
        ctl = self._active
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(self.controllers.index(ctl))
        self.tabs.blockSignals(False)
        self.setWindowTitle(f"nexus · {ctl.letter}")
        self._sync_catalog()
        self._sync_layer_list()
        self._sync_mode_panel()
        self._sync_readout()

    def _sync_tabs(self) -> None:
        many = len(self.controllers) > 1
        self.tabs.setTabsClosable(many)
        self.link_check.setVisible(many)

    def _dispatch_from(self, ctl: Controller, cmd: Command) -> None:
        """A companion window acted: that window's controller becomes active.

        Touching a window is choosing what the controls are about, exactly as
        clicking a tab is, so the panel follows the hand rather than staying on
        a tab whose windows nobody is looking at.
        """
        self.activate(ctl)
        self._dispatch(cmd)

    def _follow(self, ctl: Controller, mm, time_index: int | None) -> None:
        """Move one controller to a place and a volume, and redraw its windows."""
        dirty = Aspect.NOTHING
        if mm is not None:
            dirty |= ctl.session.do(SetXYZ(*mm))
        if time_index is not None:
            dirty |= ctl.session.do(SetIndex(int(time_index)))
        self._refresh_windows(ctl, dirty)

    def _propagate(self, before_mm, before_time: int) -> None:
        """Carry a crosshair or time change from the active controller to the rest."""
        if not self.linked or len(self.controllers) < 2:
            return
        state = self.session.state
        mm = state.crosshair_mm
        moved = mm is not None and mm != before_mm
        stepped = state.time_index != before_time
        if not (moved or stepped):
            return
        for other in self.controllers:
            if other is not self._active:
                self._follow(other, mm if moved else None, state.time_index if stepped else None)

    def _set_linked(self, on: bool) -> None:
        self.linked = bool(on)
        if self.linked:
            state = self.session.state
            for other in self.controllers:
                if other is not self._active:
                    self._follow(other, state.crosshair_mm, state.time_index)

    def _dispatch(self, cmd: Command) -> None:
        """The single entry point every widget and window mutates state through.

        Mode preparation is kicked off here rather than at each call site: the
        seed can be set from any image window, and the first seed after a mode
        switch is exactly the one whose filtering would freeze the GUI if it
        ran inline.
        """
        before_mm, before_time = self.session.state.crosshair_mm, self.session.state.time_index
        self.refresh(self.session.do(cmd))
        self._propagate(before_mm, before_time)
        if isinstance(cmd, SetSeed) and self.session.mode.needs_prepare:
            self._prepare_then_refresh()
        # A built window's own settings decide its picture, so changing one
        # rebuilds it. Everything else that could invalidate it -- a new
        # overlay, a new layer -- only marks it stale, because this is seconds
        # of work and rebuilding on a threshold drag would be unusable.
        if isinstance(cmd, BUILT_SETTINGS):
            viewport = self.session.state.viewports.find(cmd.view)
            if viewport is not None and (viewport.is_carpet or viewport.is_matrix):
                self.rebuild_view(cmd.view)

    # ------------------------------------------------------------------
    # the core: read / underlay / overlay / +1 / mode
    # ------------------------------------------------------------------
    def _build_selector(self) -> None:
        tabs_bar = QtWidgets.QToolBar("controllers")
        tabs_bar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tabs_bar)
        self.tabs = QtWidgets.QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setToolTip(
            "Controllers. Each has its own directory, stack, mode and windows;\n"
            "their windows are titled [A], [B]... and share one crosshair while linked."
        )
        for ctl in self.controllers:
            self.tabs.addTab(ctl.letter)
        self.tabs.currentChanged.connect(
            lambda i: self.activate(self.controllers[i]) if 0 <= i < len(self.controllers) else None
        )
        self.tabs.tabCloseRequested.connect(
            lambda i: (
                self.close_controller(self.controllers[i])
                if 0 <= i < len(self.controllers)
                else None
            )
        )
        tabs_bar.addWidget(self.tabs)
        new_tab = QtWidgets.QPushButton(key_label("+", "^T"))
        new_tab.setToolTip("Open another controller, linked to this one (ctrl+T)")
        new_tab.clicked.connect(self.new_controller)
        tabs_bar.addWidget(new_tab)
        self.link_check = QtWidgets.QCheckBox("linked")
        self.link_check.setChecked(True)
        self.link_check.setToolTip("Controllers follow one crosshair (in mm) and one time index")
        self.link_check.toggled.connect(self._set_linked)
        tabs_bar.addWidget(self.link_check)

        bar = QtWidgets.QToolBar("selector")
        bar.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        self.read_button = QtWidgets.QPushButton(key_label("READ", "^O"))
        self.read_button.setToolTip("Read a directory into the pickers (ctrl+O)")
        self.read_button.clicked.connect(self._read_dialog)
        bar.addWidget(self.read_button)

        self.rescan_button = QtWidgets.QPushButton("RESCAN")
        self.rescan_button.setToolTip(
            "Look for new and rewritten files now.\n"
            "This happens on its own when the directory changes; the button is\n"
            "for writes the system does not report, such as from another machine."
        )
        self.rescan_button.clicked.connect(lambda: self._start_rescan(manual=True))
        bar.addWidget(self.rescan_button)

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
            ("+GRAPH", "\u21e7N", "Open a graph window", self._new_graph),
            ("+CARPET", "\u21e7C", "Open a carpet plot of the selected run", self._new_carpet),
            (
                "+MATRIX",
                "\u21e7M",
                "Open a correlation matrix of the ROIs, or of voxel bins",
                self._new_matrix,
            ),
            (
                "+CLUSTERS",
                "\u21e7K",
                "Clusterize the selected layer at its current threshold",
                self._new_clusters,
            ),
            ("TILE", "f", "Lay every window out on a grid", self._tile),
            (
                "STACK",
                "\u21e7F",
                "Stagger the windows so each title bar is reachable",
                self._cascade,
            ),
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
    def _shrinkable(box: QtWidgets.QComboBox) -> None:
        """Let a combo narrow below its longest entry; the popup still shows it.

        A sub-brick label can be forty characters, and a form column sized to
        it pushes the colour bar off the side of the controller.
        """
        box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        box.setMinimumContentsLength(8)
        view = box.view()
        if view is not None:
            view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)

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
        if self.session.catalog_fresh.pop(entry.path, None) is not None:
            self._sync_catalog()
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

    # -- rescanning ------------------------------------------------------
    def _watch(self) -> None:
        """Watch the catalog's directory and every loaded file.

        Files as well as the directory: an overwrite in place changes a file
        without necessarily touching its directory. Re-armed after every
        rescan, because a file replaced by rename is a new inode and its old
        watch is gone.
        """
        wanted: set[str] = set()
        if self.session.catalog_dir is not None:
            wanted.add(str(self.session.catalog_dir))
            wanted.update(ly.path for ly in self.session.state.layers if ly.source == "file")
        wanted = {p for p in wanted if Path(p).exists()}
        current = set(self._watcher.directories()) | set(self._watcher.files())
        if current - wanted:
            self._watcher.removePaths(sorted(current - wanted))
        if wanted - current:
            self._watcher.addPaths(sorted(wanted - current))

    def _schedule_rescan(self) -> None:
        if self.session.catalog_dir is not None:
            self._rescan_timer.start()

    def _start_rescan(self, manual: bool = False) -> None:
        """Stat the directory on a worker and header-read only what changed."""
        session = self.session
        if session.catalog_dir is None:
            return
        if self._rescan_busy:
            self._rescan_again = True
            return
        self._rescan_busy = True
        future = self._rescan_pool.submit(
            catalog_mod.rescan,
            session.catalog_dir,
            list(session.catalog),
            recursive=session.catalog_recursive,
        )
        future.add_done_callback(lambda f, s=session, m=manual: self._rescanned.emit((s, f, m)))

    def _on_rescanned(self, payload) -> None:
        session, future, manual = payload
        self._rescan_busy = False
        try:
            result = future.result()
        except Exception as exc:  # a vanished directory, a permissions change
            self.statusBar().showMessage(f"rescan failed: {exc}", 6000)
            result = None
        live = any(ctl.session is session for ctl in self.controllers)
        if result is not None and live and (result.any or manual):
            reloaded = session.apply_rescan(result)
            if session is self.session:
                self._sync_catalog()
                self.refresh(Aspect.ALL if reloaded else Aspect.NOTHING)
            names = [session.state.layers.get(k).name for k in reloaded]
            parts = [
                f"{len(result.added)} new",
                f"{len(result.changed)} updated",
                f"{len(result.removed)} gone",
            ]
            tail = f"; reloaded {', '.join(names)}" if names else ""
            self.statusBar().showMessage(f"rescan: {', '.join(parts)}{tail}", 8000)
        elif result is not None:
            self._watch()
        if self._rescan_again:
            self._rescan_again = False
            self._start_rescan()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt)
        # Coming back to the window is when someone expects to see what they
        # just wrote -- and a write from another machine raises no event.
        if event.type() == QtCore.QEvent.Type.ActivationChange and self.isActiveWindow():
            self._schedule_rescan()
        super().changeEvent(event)

    def _sync_catalog(self) -> None:
        """Fill the pickers, defaulting the underlay to the likeliest base image."""
        entries = self.session.catalog
        fresh = self.session.catalog_fresh
        self.dir_label.setText(
            self.session.catalog_dir.name if self.session.catalog_dir else "no directory"
        )
        self._watch()
        c = theme.palette()
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
                mark = fresh.get(e.path)
                box.addItem(
                    f"{e.name:<{width}}   {e.summary}" + (f"   {mark}" if mark else ""),
                    userData=e,
                )
                if mark:
                    # Picked out until it is opened, so what the pipeline just
                    # wrote can be found without reading 277 names.
                    row = box.count() - 1
                    box.setItemData(row, QtGui.QBrush(QtGui.QColor(c.select)), BACKGROUND)
                    box.setItemData(row, QtGui.QBrush(QtGui.QColor(c.accent)), FOREGROUND)
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

    def _new_carpet(self) -> None:
        self._open_built(ViewKind.CARPET)

    def _new_matrix(self) -> None:
        self._open_built(ViewKind.MATRIX)

    def _new_clusters(self) -> None:
        vid = self.manager.open(ViewKind.CLUSTERS, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)
        self.refresh_clusters(vid)

    def refresh_clusters(self, vid: str | None = None, ctl: Controller | None = None) -> None:
        """Recompute one cluster table, or every one, from the current threshold.

        On the GUI thread on purpose: connected components on a single volume
        is milliseconds, so a table that lags the slider would be a worse lie
        than the wait is a cost. The heavy windows are the ones on the worker.
        """
        from fastfuncstuff.viewer.ui.clusterwindow import ClusterWindow

        ctl = ctl or self._active
        targets = (
            ctl.manager.cluster_windows()
            if vid is None
            else [w for w in [ctl.manager.windows.get(vid)] if isinstance(w, ClusterWindow)]
        )
        for window in targets:
            try:
                source, table = ctl.session.clusterize(
                    None, nn=window.nn, min_voxels=window.min_voxels
                )
            except (ValueError, KeyError, FileNotFoundError) as exc:
                # On the window, not the status bar: the thing that could not
                # be clustered is the thing you are looking at.
                window.show_table("", None, str(exc))
                continue
            window.show_table(source.key, table, f"{source.name}   {table.summary()}")

    def _clusters_to_rois(self, vid: str) -> None:
        """Adopt one cluster table as an ROI layer.

        The clusters become an atlas: matrix nodes, a named readout, a seed.
        Dispatched through the session so the new layer is a layer like any
        other -- the only thing it does not have is a file behind it.
        """
        from fastfuncstuff.viewer.clusters import rois_from_clusters
        from fastfuncstuff.viewer.ui.clusterwindow import ClusterWindow

        window = self.manager.windows.get(vid)
        if not isinstance(window, ClusterWindow) or window._table is None:
            return
        source = window._source_key
        name = f"clusters of {self.session.state.layers.get(source).name}" if source else "clusters"
        rois = rois_from_clusters(window._table, name=name, source=f"clusters:{source}")
        self.session.install_rois(rois, name=name, source=f"clusters:{source}")
        self.refresh(Aspect.LAYERS | Aspect.SLICES)
        self.statusBar().showMessage(f"{len(rois)} clusters are now an ROI layer", 4000)

    def _carpet_rows_to_layer(self, vid: str, first: int, last: int) -> None:
        """Turn a dragged band of carpet rows into a mask layer in the stack.

        One layer per carpet window, updated in place by every new drag, so
        refining a selection is refining one overlay. The mask is on the run's
        own grid -- a carpet row is that run's voxel -- and the image windows
        resample it like any other layer.
        """
        from fastfuncstuff.viewer.ui.carpetwindow import CarpetWindow

        window = self.manager.windows.get(vid)
        viewport = self.session.state.viewports.find(vid)
        if not isinstance(window, CarpetWindow) or viewport is None:
            return
        carpet = window.view._carpet
        run = self.session.series_source(viewport)
        if carpet is None or run is None or carpet.volume_shape != run.shape:
            self.statusBar().showMessage("rebuild the carpet (r) before selecting from it", 5000)
            return
        mask = carpet.mask_of_rows(first, last)
        lo, hi = sorted((first, last))
        name = f"{vid} rows {lo}-{hi} ·{int(mask.sum()):,} vox"
        _key, dirty = self.session.install_selection(f"selection:{vid}", mask, like=run, name=name)
        self.refresh(dirty)
        self.statusBar().showMessage(f"{int(mask.sum()):,} voxels selected from {run.name}", 4000)

    def _save_layer_dialog(self) -> None:
        """Write the selected layer to disk, whatever made it."""
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in layer.name)
        stem = stem.removesuffix(".nii.gz").removesuffix(".nii").strip("_") or key
        start = str((self.session.catalog_dir or Path.cwd()) / f"{stem}.nii.gz")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Save {layer.name}", start, "NIfTI (*.nii.gz *.nii);;All (*)"
        )
        if not path:
            return
        try:
            self.session.save_layer(key, path)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"could not save: {exc}", 8000)
            return
        self.statusBar().showMessage(f"wrote {path}", 5000)

    def _open_built(self, kind: ViewKind) -> None:
        vid = self.manager.open(kind, Plane.AXIAL)
        # Start it on the layer the controls are aimed at, which is what
        # "carpet this" or "correlate this" means when a run is selected.
        layer = self.session.state.selected_layer()
        if layer is not None and layer.time_linked and layer.n_volumes > 1:
            self.session.do(SetViewTraces(vid, layer.key))
        self.refresh(Aspect.VIEWPORTS | Aspect.GRAPH)
        self.rebuild_view(vid)

    def rebuild_view(self, vid: str) -> None:
        """Build one carpet or matrix on the worker and hand the picture back.

        Same split as a mode and as DERIVE: the arithmetic touches only arrays,
        the install happens here. Either picture is several seconds on a real
        run, so this is never allowed near the click handler.
        """
        from fastfuncstuff.viewer.ui.carpetwindow import CarpetWindow
        from fastfuncstuff.viewer.ui.clusterwindow import ClusterWindow
        from fastfuncstuff.viewer.ui.matrixwindow import MatrixWindow

        window = self.manager.windows.get(vid)
        # A cluster window asks through the same signal, but its answer is
        # milliseconds and belongs on this thread -- and before this branch
        # its NN and MIN controls asked a question nobody answered.
        if isinstance(window, ClusterWindow):
            self.refresh_clusters(vid)
            return
        viewport = self.session.state.viewports.find(vid)
        if viewport is None or self.runner.busy:
            return
        if isinstance(window, CarpetWindow):
            build, show, what = self.session.build_carpet, window.show_carpet, "a carpet"
        elif isinstance(window, MatrixWindow):
            build, show, what = self.session.build_matrix, window.show_matrix, "a matrix"
        else:
            return
        built: dict[str, object] = {}

        def job(progress) -> bool:
            _, picture = build(viewport, progress=progress)
            built["picture"] = picture
            return True

        def done(ok: bool, error: str) -> None:
            self.runner.finished.disconnect(done)
            window.set_busy(False)
            if ok:
                show(built.get("picture"))
            else:
                # On the window rather than the status bar: the thing that
                # failed is the thing you are looking at.
                show(None, error or f"could not build {what}")

        window.set_busy(True)
        self.runner.finished.connect(done)
        if not self.runner.run(job):
            self.runner.finished.disconnect(done)
            window.set_busy(False)

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

    def _peers(self) -> list[WindowManager]:
        return [c.manager for c in self.controllers if c is not self._active]

    def _tile(self) -> None:
        """Tile every controller's windows together, so A and B sit side by side."""
        self.manager.tile(self, peers=self._peers())

    def _cascade(self) -> None:
        self.manager.cascade(self, peers=self._peers())

    def _raise_all(self) -> None:
        for ctl in self.controllers:
            ctl.manager.raise_all()

    def _toggle_theme(self) -> None:
        """One palette for the whole screen, recorded in every controller."""
        name = "light" if self.session.state.theme == "dark" else "dark"
        for other in self.controllers:
            if other is not self._active:
                other.session.do(SetTheme(name))
        self._dispatch(SetTheme(name))

    def _apply_theme(self) -> None:
        """Push the palette into every window, including this one."""
        name = self.session.state.theme
        theme.set_theme(name)
        self.setStyleSheet(stylesheet())
        self.rangebar.restyle()
        self.theme_button.setText(key_label("LIGHT" if name == "dark" else "DARK", "d"))
        for ctl in self.controllers:
            ctl.manager.restyle()

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
            "[ and ] step through the stack; space or the tick box hides a layer.\n"
            "A soloed image window draws whichever one is selected here."
        )
        self.layer_list.currentRowChanged.connect(self._row_selected)
        # Visibility used to be a glyph in the row's text, which looked like a
        # tick box and was not one -- clicking it only selected the row. A real
        # check state is the same information and answers the click.
        self.layer_list.itemChanged.connect(self._item_checked)
        # Drag to reorder. The list is drawn top-first while the stack is
        # stored bottom-first, so the drop row is translated in one place --
        # _row_selected and this are the only two that know about the flip.
        self.layer_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.layer_list.model().rowsMoved.connect(self._rows_moved)
        v.addWidget(self.layer_list)

        # Two rows of three: five buttons in one row is wider than the panel,
        # and a panel wider than its window scrolls sideways.
        stack_row = QtWidgets.QGridLayout()
        stack_row.setSpacing(4)
        for position, (text, key, tip, slot) in enumerate(
            (
                ("LOWER", "{", "Move the selected layer down the stack", lambda: self._reorder(-1)),
                ("RAISE", "}", "Move the selected layer up the stack", lambda: self._reorder(1)),
                ("UNDERLAY", "u", "Make the selected layer the underlay", self._make_underlay),
                ("DROP", "del", "Remove the selected layer", self._drop_layer),
                (
                    "SAVE",
                    "\u21e7S",
                    "Write the selected layer to a NIfTI file",
                    self._save_layer_dialog,
                ),
            )
        ):
            b = QtWidgets.QPushButton(key_label(text, key))
            b.setToolTip(f"{tip} ({key})")
            b.setStyleSheet(f"QPushButton {{ font-size: {theme.FONT_SMALL}px; padding: 3px 6px; }}")
            b.clicked.connect(slot)
            stack_row.addWidget(b, *divmod(position, 3))
        v.addLayout(stack_row)

        # The layer form and the colour bar side by side: the bar stands on end
        # to the right of the pickers, so the numbers that define it sit in the
        # column of space the form's labels leave free instead of claiming a
        # full-width band of their own below it.
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setSpacing(7)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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
        self._shrinkable(self.brick_box)
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
        self._shrinkable(self.thrbrick_box)
        self.thrbrick_head = self._head("THR ON")
        # Two exclusive boxes rather than more rows in the menu: they say how
        # the threshold sub-brick moves when OLAY does, not which one it is.
        self.thr_same_check = QtWidgets.QCheckBox("same")
        self.thr_same_check.setToolTip(
            "Threshold on the OLAY sub-brick itself, following it as it changes.\n"
            "The case for a t or an F shown directly."
        )
        self.thr_next_check = QtWidgets.QCheckBox("+1")
        self.thr_next_check.setToolTip(
            "Threshold on the sub-brick after OLAY, following it as it changes:\n"
            "stepping through the _Coef sub-bricks keeps cutting on each one's _Tstat."
        )
        for check, mode in ((self.thr_same_check, "same"), (self.thr_next_check, "next")):
            check.clicked.connect(
                lambda on, m=mode: self._apply(SetThresholdFollow, mode=m if on else "fixed")
            )
        self.thrbrick_row = QtWidgets.QWidget()
        thr_row = QtWidgets.QHBoxLayout(self.thrbrick_row)
        thr_row.setContentsMargins(0, 0, 0, 0)
        thr_row.setSpacing(4)
        thr_row.addWidget(self.thrbrick_box, 1)
        thr_row.addWidget(self.thr_same_check)
        thr_row.addWidget(self.thr_next_check)
        form.addRow(self.thrbrick_head, self.thrbrick_row)

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

        # Off is both boxes clear: a fade is a choice you make, not a state
        # you have to find in a menu to leave.
        self.alpha_linear_check = QtWidgets.QCheckBox("linear")
        self.alpha_quad_check = QtWidgets.QCheckBox("quadratic")
        for check, mode in (
            (self.alpha_linear_check, AlphaMode.LINEAR),
            (self.alpha_quad_check, AlphaMode.QUADRATIC),
        ):
            check.setToolTip("Fade sub-threshold voxels instead of hiding them.")
            check.clicked.connect(
                lambda on, m=mode: self._apply(SetAlpha, mode=(m if on else AlphaMode.OFF).value)
            )
        self.alpha_row = QtWidgets.QWidget()
        alpha_row = QtWidgets.QHBoxLayout(self.alpha_row)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.setSpacing(6)
        alpha_row.addWidget(self.alpha_linear_check)
        alpha_row.addWidget(self.alpha_quad_check)
        alpha_row.addStretch(1)
        form.addRow(self._head(key_label("ALPHA", "a")), self.alpha_row)

        # Min, threshold and max are edited on the bar itself. Splitting the
        # number from the picture of the number is what let the bar go stale.
        self.rangebar = RangeBar()
        self.thr_head = self.rangebar.thr_caption
        self.thr_head.setText(key_label("THRESH", "t"))
        self.rangebar.range_changed.connect(self._range_changed)
        self.rangebar.threshold_changed.connect(self._threshold_changed)
        self.rangebar.autorange_requested.connect(self._autorange)
        self.rangebar.mirror_changed.connect(lambda on: self._apply(SetRangeMirror, on=on))

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
        self.opacity_label = QtWidgets.QLabel("100%")
        self.opacity_label.setObjectName("value")
        self.opacity_label.setMinimumWidth(
            QtGui.QFontMetrics(self.font()).horizontalAdvance("100%") + 4
        )
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.setSpacing(4)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        form.addRow(self._head("OPACITY"), opacity_row)

        self.boxed_check = QtWidgets.QCheckBox(key_label("boxed", "b"))
        self.boxed_check.toggled.connect(lambda on: self._apply(SetBoxed, on=bool(on)))
        form.addRow(QtWidgets.QLabel(""), self.boxed_check)

        self.roi_check = QtWidgets.QCheckBox("is ROIs")
        self.roi_check.setToolTip(
            "The values are region identities, not magnitudes: colour by "
            "identity, name the region in the readout, and let a correlation "
            "matrix or a seed be built from it. Guessed for a 3-D volume of "
            "small integers; turn it on for a 4-D stack of masks."
        )
        self.roi_check.toggled.connect(lambda on: self._apply(SetLayerRoi, on=bool(on)))
        form.addRow(QtWidgets.QLabel(""), self.roi_check)

        self.timelink_check = QtWidgets.QCheckBox("follows time")
        self.timelink_check.setToolTip(
            "4-D NIfTI cannot say whether sub-bricks are time points or "
            "contrasts. Uncheck for a stats dataset."
        )
        self.timelink_check.toggled.connect(lambda on: self._apply(SetTimeLinked, on=bool(on)))
        form.addRow(QtWidgets.QLabel(""), self.timelink_check)
        controls.addLayout(form, 1)

        bar_column = QtWidgets.QVBoxLayout()
        bar_column.setSpacing(4)
        bar_column.addWidget(self._head("RANGE"))
        bar_column.addWidget(self.rangebar, 1)
        controls.addLayout(bar_column)
        v.addLayout(controls)

        self.mode_head = self._head("MODE PARAMETERS")
        v.addWidget(self.mode_head)
        self.mode_panel = ControlPanel()
        self.mode_panel.changed.connect(self._mode_param_changed)
        self.mode_panel.action_requested.connect(self._mode_action)
        v.addWidget(self.mode_panel)
        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        self.setCentralWidget(scroll)

    def _switch_mode(self, name: str) -> None:
        self._dispatch(SetMode(name))
        self.refresh(self.session.open_mode_panels())
        self._prepare_then_refresh()

    def _mode_param_changed(self, name: str, value: str) -> None:
        self._dispatch(SetModeParam(name, value))
        self._prepare_then_refresh()

    def _mode_action(self, name: str) -> None:
        """Press a mode's button, then finish whatever slow work it started.

        A button may open windows -- Denoise's CARPETS opens two -- and a built
        window opened that way has no picture until someone rebuilds it, so any
        that appeared are queued for the worker.
        """
        before = set(self.session.state.viewports.ids)
        if not any(a.name == name for a in self.session.mode.actions()):
            return  # a trace window's key for an action this mode does not have
        # Asked before the action is dispatched, so a mode declares a button
        # once and decides for itself whether pressing it acts or asks first.
        spec = self.session.mode.dialog_for(name)
        if spec is not None:
            self._open_tool_dialog(spec)
            return
        try:
            self._dispatch(ModeAction(name))
        except (KeyError, ValueError, OSError) as exc:
            self.statusBar().showMessage(f"{name} failed: {exc}", 8000)
            return
        if self.session.mode.needs_prepare and self.runner.busy:
            self.statusBar().showMessage(f"busy; press {name.upper()} again when it finishes", 6000)
        self._prepare_then_refresh()
        for vid in self.session.state.viewports.ids:
            if vid not in before:
                viewport = self.session.state.viewports.find(vid)
                if viewport is not None and (viewport.is_carpet or viewport.is_matrix):
                    self._queue_rebuild(vid)
        self._sync_mode_panel()

    def _open_tool_dialog(self, spec) -> None:
        """Show one tool's form, reusing the window if it is already up."""
        dialog = self._tool_dialogs.get(spec.name)
        if dialog is None:
            dialog = ToolDialog(spec, self.runner, self._on_tool_installed, self)
            self._tool_dialogs[spec.name] = dialog
        else:
            # Re-seeded rather than rebuilt: the input dropdown has to pick up
            # layers loaded since it was last opened.
            dialog.reseed(spec)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_tool_installed(self, dirty: Aspect) -> None:
        """A tool's result is in; open any plots it brought with it.

        Through the same open_mode_panels the mode selector uses, so a tool's
        motion plot is a trace window like ICA's spectrum -- in the recording,
        closable, and reopened by entering the mode again.
        """
        self.refresh(dirty | self.session.open_mode_panels())

    def _close_tool_dialogs(self) -> None:
        """Drop every tool form.

        Called when the active controller changes. A dialog's spec closes over
        the session that made it, so leaving one open across a tab switch would
        quietly run the tool against the stack you are no longer looking at.
        """
        for dialog in self._tool_dialogs.values():
            dialog.close()
            dialog.deleteLater()
        self._tool_dialogs.clear()

    def _queue_rebuild(self, vid: str) -> None:
        """Rebuild a window now, or as soon as the worker is free.

        The runner does one job at a time and refuses a second, so two carpets
        opened by one click would otherwise leave the second one empty.
        """
        self._rebuild_queue.append((self._active, vid))
        self._drain_rebuilds()

    def _drain_rebuilds(self) -> None:
        while self._rebuild_queue and not self.runner.busy:
            ctl, vid = self._rebuild_queue.pop(0)
            if ctl in self.controllers and ctl.session.state.viewports.find(vid) is not None:
                self._on(ctl, self.rebuild_view, vid)

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
                Binding("shift+n", "open a graph window", self._new_graph, group="windows"),
                Binding("shift+c", "open a carpet plot", self._new_carpet, group="windows"),
                Binding("shift+m", "open a correlation matrix", self._new_matrix, group="windows"),
                Binding("f", "tile every window", self._tile, group="windows"),
                Binding("shift+f", "stagger every window", self._cascade, group="windows"),
                Binding("r", "raise every window", self._raise_all, group="windows"),
                Binding("d", "dark / light palette", self._toggle_theme, group="windows"),
                Binding(",", "previous volume", lambda: self._step_time(-1), group="time"),
                Binding(".", "next volume", lambda: self._step_time(1), group="time"),
                Binding(
                    "<",
                    "previous sub-brick of this layer",
                    lambda: self._step_brick(-1),
                    group="time",
                ),
                Binding(
                    ">",
                    "next sub-brick of this layer",
                    lambda: self._step_brick(1),
                    group="time",
                ),
                Binding("v", "play / pause", self._toggle_play, group="time"),
                Binding("[", "previous layer", lambda: self._cycle_layer(-1), group="layer"),
                Binding("]", "next layer", lambda: self._cycle_layer(1), group="layer"),
                Binding("space", "show / hide layer", self._toggle_visible, group="layer"),
                Binding("t", "threshold down", lambda: self._nudge_threshold(-0.05), group="layer"),
                Binding(
                    "shift+t", "threshold up", lambda: self._nudge_threshold(0.05), group="layer"
                ),
                Binding("c", "next colormap", self._cycle_colormap, group="layer"),
                Binding("s", "next sign mode", self._cycle_sign, group="layer"),
                Binding("a", "next alpha mode", self._cycle_alpha, group="layer"),
                Binding("b", "toggle boxed", self.boxed_check.toggle, group="layer"),
                Binding("{", "move layer down the stack", lambda: self._reorder(-1), group="layer"),
                Binding("}", "move layer up the stack", lambda: self._reorder(1), group="layer"),
                Binding("u", "make it the underlay", self._make_underlay, group="layer"),
                Binding(
                    "shift+s", "save the layer to a file", self._save_layer_dialog, group="layer"
                ),
                Binding(
                    "Del",
                    "remove the layer",
                    self._drop_layer,
                    group="layer",
                    aliases=("Backspace",),
                ),
                Binding(
                    "ctrl+t", "open another controller", self.new_controller, group="controllers"
                ),
                *[
                    Binding(
                        f"ctrl+{n}",
                        f"controller {letter}",
                        lambda letter=letter: self._activate_letter(letter),
                        group="controllers",
                    )
                    for n, letter in enumerate(LETTERS[:5], start=1)
                ],
                Binding("ctrl+o", "read a directory", self._read_dialog, group="session"),
                Binding("ctrl+s", "save session script", self._save_script_dialog, group="session"),
                Binding("h", "this list", self.help.toggle, group="session"),
            ]
        )

    def _activate_letter(self, letter: str) -> None:
        found = next((c for c in self.controllers if c.letter == letter), None)
        if found is not None:
            self.activate(found)

    def current_key(self) -> str | None:
        layer = self.session.state.selected_layer()
        return None if layer is None else layer.key

    def _apply(self, cls, **kwargs) -> None:
        key = self.current_key()
        if key is None:
            return
        self._dispatch(cls(key=key, **kwargs))

    # -- reordering the stack -------------------------------------------
    def _stack_index(self, row: int) -> int:
        """List row (top-first) to stack index (bottom-first)."""
        return self.layer_list.count() - 1 - row

    def _rows_moved(self, _parent, start: int, _end, _dest, row: int) -> None:
        """A drag landed. Translate it and let the command bus do the move."""
        key = self._key_at_row(start)
        if key is None:
            return
        # Qt reports the destination as the row the item was inserted *before*,
        # which is one past itself when dragging downward in the widget.
        target = row - 1 if row > start else row
        self._dispatch(MoveLayer(key, self._stack_index(target)))
        self._sync_layer_list()

    def _key_at_row(self, row: int) -> str | None:
        keys = list(reversed(self.session.state.layers.keys))
        return keys[row] if 0 <= row < len(keys) else None

    def _reorder(self, delta: int) -> None:
        """Move the selected layer one place through the stack."""
        key = self.current_key()
        if key is None:
            return
        stack = self.session.state.layers
        target = stack.index_of(key) + delta
        if 0 <= target < len(stack):
            self._dispatch(MoveLayer(key, target))
            self._sync_layer_list()

    def _make_underlay(self) -> None:
        """Promote the selected layer to the bottom, grid and all.

        The gesture a derived layer wanted: denoise a run, then put the result
        underneath everything and look at the stats on top of it.
        """
        key = self.current_key()
        if key is not None and self.session.state.layers.index_of(key) != 0:
            self._dispatch(MoveLayer(key, 0))
            self._sync_layer_list()

    def _drop_layer(self) -> None:
        """Remove the selected layer, unless it is the only one left.

        Same rule as the last image window and the panel that could not be
        reopened: a state whose only way out is the control you just used is
        not a state to allow.
        """
        key = self.current_key()
        if key is None:
            return
        if len(self.session.state.layers) <= 1:
            self.statusBar().showMessage(
                "the last layer stays; there would be nothing to show", 5000
            )
            return
        name = self.session.state.layers.get(key).name
        self._dispatch(RemoveLayer(key))
        self._sync_layer_list()
        self.statusBar().showMessage(f"removed {name}", 4000)

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
        self._hint_if_unmoved()

    def _hint_if_unmoved(self) -> None:
        """Say so when the time keys cannot move the layer being looked at.

        A QC stack of first and last is 4-D and selected and does not follow the
        time slider, by design -- its sub-bricks are named states, not time
        points. Pressing `.` on it therefore scrubs everything *else*, which
        reads as the key being broken rather than as aimed somewhere else.
        """
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.find(key)
        if layer is None or layer.time_linked or layer.n_volumes <= 1:
            return
        self.statusBar().showMessage(
            f"{layer.name} does not follow the time slider — use < and > to step its sub-bricks",
            6000,
        )

    def _step_brick(self, delta: int) -> None:
        """Step the selected layer's own sub-brick, whatever time is doing."""
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.find(key)
        if layer is None or layer.n_volumes <= 1:
            return
        nxt = (layer.volume_index + delta) % layer.n_volumes
        self._dispatch(SetVolume(key=key, index=nxt))
        name = layer.labels[nxt] if nxt < len(layer.labels) else f"#{nxt}"
        self.statusBar().showMessage(f"{layer.name}: {name}", 4000)
        self._sync_layer_controls()

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

    def _item_checked(self, item: QtWidgets.QListWidgetItem) -> None:
        """A tick box was clicked: show or hide that layer.

        Addressed by the key stored on the item rather than by its row, because
        the list is drawn top-first over a bottom-first stack and a row number
        is the one thing here that means two different things.
        """
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if key is None:
            return
        layer = self.session.state.layers.find(key)
        wanted = item.checkState() is QtCore.Qt.CheckState.Checked
        if layer is None or layer.visible == wanted:
            return
        self._dispatch(SetLayerVisible(key, wanted))

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

        layer = self.session.state.layers.get(key)
        if layer.time_linked or layer.n_volumes <= 1 or layer.is_computed:
            lo, hi = derive_range(self.session.volume(key))
            self._dispatch(SetRange(key, float(lo), float(hi)))
            return
        # The sub-brick on screen, not the time index -- which on a bucket is
        # always 0, so auto kept re-deriving the F's range whatever was shown.
        look = self.session.overlay_look(key, layer.volume_index, colormap=layer.colormap)
        if "range_lo" in look:
            self._dispatch(SetRange(key, float(look["range_lo"]), float(look["range_hi"])))
        if look.get("colormap", layer.colormap) != layer.colormap:
            self._dispatch(SetColormap(key, str(look["colormap"])))

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
        layer = self.session.state.layers.find(self.current_key() or "")
        if layer is None:
            return
        modes = list(AlphaMode)
        nxt = modes[(modes.index(layer.alpha_mode) + 1) % len(modes)]
        self._apply(SetAlpha, mode=nxt.value)

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
        ctl = self._active

        def ready() -> None:
            dirty = ctl.session.refresh_mode() | Aspect.LAYERS | Aspect.SLICES
            if ctl is self._active:
                self.refresh(dirty)
            else:
                self._refresh_windows(ctl, dirty)

        run_when_ready(
            self.runner,
            ctl.session.mode,
            on_ready=ready,
            on_error=lambda msg: self.statusBar().showMessage(f"mode failed: {msg}", 8000),
        )

    def _on_layer_loaded(self, ctl: Controller, key: str) -> None:
        ctl.session.invalidate(key)
        if ctl is self._active:
            self._sync_layer_list()
            self.refresh(Aspect.SLICES | Aspect.GRAPH)
        else:
            self._refresh_windows(ctl, Aspect.SLICES | Aspect.GRAPH)

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
        self._refresh_windows(self._active, dirty)
        self._sync_readout()

    def _refresh_windows(self, ctl: Controller, dirty: Aspect) -> None:
        """Bring one controller's windows up to date, active or not."""
        if dirty is Aspect.NOTHING:
            return
        # Windows first: a viewport that has just appeared has to exist before
        # anything tries to draw into it.
        if dirty & (Aspect.VIEWPORTS | Aspect.LAYERS | Aspect.GRID):
            ctl.manager.sync()
        if dirty & (Aspect.LAYERS | Aspect.GRID):
            ctl.manager.mark_built_stale()
        # The cluster table describes the picture, so it follows the threshold
        # rather than waiting to be asked. Anything cheaper than the redraw it
        # sits beside can afford to.
        if dirty & (Aspect.THRESHOLD | Aspect.LAYERS | Aspect.GRID):
            self.refresh_clusters(ctl=ctl)
        ctl.manager.redraw(dirty)

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
                    # Reused rather than appended, or every sync grows the
                    # picker by one more copy of the same name.
                    index = next(
                        (
                            i
                            for i in range(1, box.count())
                            if box.itemData(i) is None and box.itemText(i) == layer.name
                        ),
                        -1,
                    )
                    if index < 0:
                        box.addItem(layer.name, userData=None)
                        index = box.count() - 1
            box.setCurrentIndex(index)
            box.blockSignals(False)

    def _sync_layer_list(self) -> None:
        self._watch()
        selected = self.current_key()
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        keys = list(reversed(self.session.state.layers.keys))
        for layer in reversed(list(self.session.state.layers)):
            try:
                pending = self.session.store.get(layer.key).pending
            except KeyError:
                pending = False
            tag = " ·computed" if layer.is_computed else (" ·loading" if pending else "")
            item = QtWidgets.QListWidgetItem(f"{layer.name}{tag}")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if layer.visible else QtCore.Qt.CheckState.Unchecked
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, layer.key)
            self.layer_list.addItem(item)
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
        self.mode_panel.rebuild(mode.controls(), mode.params, mode.actions())
        has = bool(mode.controls()) or bool(mode.actions())
        self.mode_head.setVisible(has)
        self.mode_panel.setVisible(has)
        idx = self.mode_box.findData(mode.name)
        if idx >= 0 and idx != self.mode_box.currentIndex():
            self.mode_box.blockSignals(True)
            self.mode_box.setCurrentIndex(idx)
            self.mode_box.blockSignals(False)

    def _sync_layer_controls(self) -> None:
        key = self.current_key()
        # Off rather than left showing the last layer: an empty controller's
        # panel otherwise reads as holding the stack of the tab before it.
        for widget in (
            self.brick_box,
            self.thrbrick_box,
            self.cmap_box,
            self.sign_box,
            self.alpha_row,
            self.rangebar,
            self.opacity_slider,
            self.boxed_check,
        ):
            widget.setEnabled(key is not None)
        if key is None:
            for check in (self.roi_check, self.timelink_check):
                check.setEnabled(False)
            return
        layer = self.session.state.layers.get(key)
        for box, value in (
            (self.cmap_box, layer.colormap),
            (self.sign_box, layer.sign_mode.value),
        ):
            box.blockSignals(True)
            box.setCurrentText(value)
            box.blockSignals(False)
        for check, value, enabled in (
            (self.boxed_check, layer.boxed, True),
            (self.alpha_linear_check, layer.alpha_mode is AlphaMode.LINEAR, True),
            (self.alpha_quad_check, layer.alpha_mode is AlphaMode.QUADRATIC, True),
            (self.roi_check, layer.roi, not layer.is_computed),
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
        scale = None if thresholds_itself(layer) else self.session.threshold_scale(key)
        self.rangebar.configure(layer, threshold_scale=scale)
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
            (self.thrbrick_head, self.thrbrick_row),
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
        for i, name in enumerate(names):
            self.thrbrick_box.addItem(name, userData=i)
        # Always the sub-brick actually cut on, whichever rule chose it.
        self.thrbrick_box.setCurrentIndex(min(layer.threshold_brick, layer.n_volumes - 1))
        self.thrbrick_box.blockSignals(False)
        for check, mode in ((self.thr_same_check, "same"), (self.thr_next_check, "next")):
            check.blockSignals(True)
            check.setChecked(layer.threshold_follow == mode)
            check.blockSignals(False)

    def _sync_readout(self) -> None:
        st = self.session.state
        if st.grid is None:
            self.coord_label.setText("no data — press READ")
            self.value_label.setText("")
            self.mode_label.setText("")
            return
        i, j, k = st.crosshair
        mm = st.crosshair_mm or (0.0, 0.0, 0.0)
        which = f"{self._active.letter}  " if len(self.controllers) > 1 else ""
        self.coord_label.setText(
            f"{which}ijk {i:>3d} {j:>3d} {k:>3d}   xyz {mm[0]:>7.1f} {mm[1]:>7.1f} {mm[2]:>7.1f}"
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
            if layer.roi:
                # "37" is not what you want to know when you are pointing at
                # the thalamus, and it is the one readout that has to survive
                # being outside every region -- hence "--", not "0".
                rois = self.session.roi_set(layer.key)
                found = rois.at(st.crosshair) if rois is not None else None
                parts.append(f"{layer.name}={found.name if found else '--'}")
                continue
            # Only where the header actually named something: appending "#0" to
            # every 3-D anatomy would be noise dressed as information.
            tag = f" {layer.sub_brick()}" if layer.labels else ""
            parts.append(f"{layer.name}{tag}={'--' if val is None else f'{val:.4g}'}")
        self.value_label.setText("   ".join(parts[:3]))

    def dock_left(self) -> None:
        """Park the controller at the left edge of its screen, full height.

        Left rather than wherever the window system drops it, so tiling always
        has one contiguous region to its right to fill.
        """
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.resize(self.width(), area.height() - (self.frameGeometry().height() - self.height()))
        self.move(area.topLeft())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        # The controller is the session; closing it closes the companions too,
        # or they linger with nothing driving them.
        self._play.stop()
        self._rescan_timer.stop()
        self._rescan_pool.shutdown(wait=False, cancel_futures=True)
        self.runner.wait(2000)
        for ctl in self.controllers:
            ctl.manager.close_all()
            ctl.session.close()
        super().closeEvent(event)


def quit_on_interrupt(app: QtWidgets.QApplication, win: QtWidgets.QWidget) -> QtCore.QTimer:
    """Let Ctrl+C in the terminal close the viewer.

    Python only runs its signal handlers between bytecodes, and inside
    ``app.exec()`` the interpreter is idle in Qt's C++ loop -- so SIGINT sat
    pending until something else woke Python, which on an idle window is
    never. A timer that does nothing hands control back often enough for the
    handler to run. Closing the window rather than exiting means the same
    cleanup as the close button: workers stopped, sessions released.
    """

    def interrupted(_signum, _frame) -> None:
        win.close()
        app.quit()

    signal.signal(signal.SIGINT, interrupted)
    timer = QtCore.QTimer(win)
    timer.timeout.connect(lambda: None)
    timer.start(200)
    return timer


def launch(
    paths: list[str],
    *,
    device: str | None = None,
    script: str | None = None,
    directory: str | None = None,
) -> int:
    """Open the controller and run the Qt loop."""
    from fastfuncstuff.cli_utils import setup_device

    # The GTK accessibility bridge prints a dbind warning at every start on a
    # desktop whose at-spi bus it cannot reach, and nothing here uses it.
    # setdefault, so someone who needs a screen reader can still turn it on.
    os.environ.setdefault("NO_AT_BRIDGE", "1")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=setup_device(device))
    win = ViewerWindow(session)
    quit_on_interrupt(app, win)

    # A directory, or the one the first dataset lives in, so the pickers are
    # populated before anyone reaches for Read.
    start = directory or (str(Path(paths[0]).parent) if paths else None)
    if start:
        win.read_directory(start)
    for p in paths:
        win.open_path(p)
    if script:
        win.refresh(session.run_script(Path(script).read_text()))
    win.dock_left()
    win.show()
    win._tile()
    return app.exec()


__all__ = ["ViewerWindow", "launch"]
