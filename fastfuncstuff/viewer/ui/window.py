"""The controller: a data selector, driving N companion windows.

The core is one row -- Read, Data, Load, Mode -- over a stack list. That is the
whole viewer; images and graphs are things you turn on beside it.

It used to be Read, Underlay, Overlay, +1, Mode, which made you say what kind
of thing a file was before you had opened it. The answer changes with the
question: an anatomical is an overlay when you are checking alignment, a run is
an underlay while you set its mask, and a GLM result wants the run loaded but
not drawn. So there is one load, and where a layer sits, whether it is drawn
and whether a mode reads it are three separate things decided afterwards -- the
first in the stack list, the second by its tick box, the third by INPUT.

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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer import catalog as catalog_mod
from fastfuncstuff.viewer import favorites
from fastfuncstuff.viewer.catalog import CatalogEntry
from fastfuncstuff.viewer.colormap import available_colormaps
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.layers import AlphaMode, SignMode
from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import ActionControl, OverlayKind
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.state import Plane, grid_res_choices
from fastfuncstuff.viewer.surfaces import next_outlines
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.boot import application
from fastfuncstuff.viewer.ui.colorbar import RangeBar, colormap_icon, thresholds_itself
from fastfuncstuff.viewer.ui.controls import ControlPanel
from fastfuncstuff.viewer.ui.flow import FlowBar
from fastfuncstuff.viewer.ui.manager import WindowManager
from fastfuncstuff.viewer.ui.shortcuts import (
    Binding,
    ShortcutHelp,
    jump_back_key,
    keep_keys_for_shortcuts,
)
from fastfuncstuff.viewer.ui.theme import MONO, key_label, stylesheet
from fastfuncstuff.viewer.ui.tooldialog import ToolDialog
from fastfuncstuff.viewer.ui.widgets import (
    VALUE_ROLE,
    FileDropFilter,
    PreviewCombo,
    RowSizedList,
    ValueColumnDelegate,
    install_combo_pickers,
)
from fastfuncstuff.viewer.ui.work import PreparationRunner, run_when_ready
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import (
    RESAMPLE_MODES,
    Load,
    LoadSurfaces,
    ModeAction,
    MoveLayer,
    Read,
    ReadFiles,
    RemoveLayer,
    SaveSurfaces,
    SelectLayer,
    SetAlpha,
    SetBoxed,
    SetCarpetOrder,
    SetColormap,
    SetColormapReversed,
    SetEdges,
    SetGridRes,
    SetIJK,
    SetIndex,
    SetInput,
    SetLayerOpacity,
    SetLayerRoi,
    SetLayerVisible,
    SetMatrixOrder,
    SetMode,
    SetModeParam,
    SetRange,
    SetRangeFixed,
    SetRangeMirror,
    SetResample,
    SetSeed,
    SetSign,
    SetTheme,
    SetThreshold,
    SetThresholdFollow,
    SetThresholdIndex,
    SetTimeLinked,
    SetViewDetrend,
    SetViewRois,
    SetViewScaling,
    SetViewTraces,
    SetVolume,
    SetXYZ,
    ShowSurfaces,
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
#: The picker's first row. It names what LOAD will open next rather than what
#: is on screen -- the stack list below says that -- so it prompts rather than
#: reporting "(none)" over a window full of data.
NONE_LABEL = "(choose a dataset)"


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
#: How often the window checks the card and hands memory back if it is short.
VRAM_POLL_MS = 2000
#: Layers shown before the list scrolls. Six covers an anatomy, a run, a mode's
#: output and a couple of kept copies -- past that the list is being used as a
#: workspace and scrolling it is better than spending the panel on it.
LAYER_ROWS = 6
#: Fastest the layer list's value column follows a moving crosshair.
VALUES_EVERY_MS = 40
BACKGROUND = QtCore.Qt.ItemDataRole.BackgroundRole
FOREGROUND = QtCore.Qt.ItemDataRole.ForegroundRole


class ViewerWindow(QtWidgets.QMainWindow):
    #: ``(session, future, manual)`` from the rescan worker, delivered on the GUI thread.
    _rescanned = QtCore.Signal(object)

    def __init__(self, session: ViewerSession) -> None:
        super().__init__()
        install_combo_pickers(QtWidgets.QApplication.instance())
        self.setAcceptDrops(True)
        QtWidgets.QApplication.instance().installEventFilter(
            FileDropFilter(self._takes_drops, self._on_files_dropped, self)
        )
        self.setWindowTitle("nexus")
        self.setStyleSheet(stylesheet())
        # Wide enough that the panel never scrolls sideways. The layer form
        # beside the colour bar needs ~480 once a stats layer gives the
        # sub-brick pickers something to show, and a controller that clips its
        # own range numbers to save twenty pixels is saving the wrong thing.
        self.resize(500, 820)

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

        # Hands device copies back when something else -- a fit -- is filling
        # the card, and keeps the readout honest. Cheap: one driver query.
        self._vram_timer = QtCore.QTimer(self)
        self._vram_timer.setInterval(VRAM_POLL_MS)
        self._vram_timer.timeout.connect(self._tick_vram)
        self._vram_timer.start()
        keep_keys_for_shortcuts(self)

        # Mode preparation runs on a worker; the mode is told to defer so a
        # seed click never runs seconds of filtering inside the click handler.
        self.runner = PreparationRunner(self)
        self.runner.progress.connect(self._on_prepare_progress)
        self.runner.busy_changed.connect(self._on_prepare_busy)
        self._rebuild_queue: list[tuple[Controller, str]] = []
        #: Controllers whose mode is waiting for the worker. One worker, so a
        #: seed carried to three tabs prepares them one after another rather
        #: than dropping the two that found it busy.
        self._prepare_queue: list[Controller] = []
        #: Open tool dialogs, by tool name. One per tool, reused on re-press.
        self._tool_dialogs: dict[str, ToolDialog] = {}
        # Deferred a turn so the job that just finished has handed its picture
        # over before the next one takes the worker.
        self.runner.finished.connect(lambda *_: QtCore.QTimer.singleShot(0, self._drain_work))

        from fastfuncstuff.viewer.ui.player import Player

        self._play = Player(
            lambda: self._step_time(1), lambda: self.manager.surface_canvases(), self
        )
        self._play.toggled.connect(self._play_toggled)
        self._play.rate.connect(
            lambda fps: self.statusBar().showMessage(f"playing · {fps:.1f} frames/s", 1500)
        )

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
        manager.painted.connect(lambda d, c=ctl: self._on(c, self.refresh, Aspect(d)))
        manager.play_requested.connect(lambda c=ctl: self._on(c, self._toggle_play))
        manager.graph_requested.connect(
            lambda plane, c=ctl: self._on(c, self._new_graph, Plane(plane))
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
        elif source.reading_files:
            session.do(ReadFiles(tuple(str(p) for p in source.file_list)))
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
        self._row()
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
        # The player steps the active controller's time: it stops with it.
        self._play.stop()
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

    def _propagate_seed(self) -> None:
        """Carry the active controller's seed to the rest, by millimetres.

        Each tab correlates its own run from the same place, so two subjects,
        or raw against denoised, are compared like with like from one click.
        Off with the link, as the crosshair is.
        """
        if not self.linked or len(self.controllers) < 2:
            return
        state = self.session.state
        if state.seed is None or state.grid is None:
            return
        mm = state.grid.ijk_to_mm(state.seed)
        for other in self.controllers:
            grid = other.session.state.grid
            if other is self._active or grid is None:
                continue
            i, j, k = (round(v) for v in grid.mm_to_ijk(mm))
            self._refresh_windows(other, other.session.do(SetSeed(i, j, k)))
            if other.session.mode.needs_prepare:
                self._queue_prepare(other)

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
        if isinstance(cmd, SetSeed):
            if self.session.mode.needs_prepare:
                self._queue_prepare(self._active)
            self._propagate_seed()
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

        #: The mesh list, made on first use (ctrl+M / MESH).
        self.mesh_window = None
        self.surf_button = QtWidgets.QPushButton("SURF")
        self.surf_button.setToolTip(
            "Load a FreeSurfer subject's surfaces and outline white/pial on the slices.\n"
            "shift+O cycles which outlines are drawn."
        )
        self.surf_button.clicked.connect(self._surfaces_dialog)
        bar.addWidget(self.surf_button)
        self.ica_button = QtWidgets.QPushButton("ICA")
        self.ica_button.setToolTip(
            "Review an ICA made elsewhere -- a MELODIC or ffs_ica folder: step through\n"
            "components, label signal/noise, save the labels. To decompose the run\n"
            "you are looking at, use the InstaICA mode."
        )
        self.ica_button.clicked.connect(self._ica_folder_dialog)
        bar.addWidget(self.ica_button)
        # The surface windows, as buttons as well as keys: a window nobody can
        # find is a window that does not exist.
        for text, tip, slot in (
            (
                "3D",
                "3-D surface window: overlays on folded/inflated/flat (shift+V)",
                self._new_surface,
            ),
            (
                "DEPTH",
                "Depth profiles of the overlay around the crosshair (shift+L)",
                self._new_depth,
            ),
            (
                "QC",
                "Ribbon profile column: surface QC, back to front (shift+P)",
                self._new_profiles,
            ),
            (
                "CHEDI",
                "Cortex laid flat around the crosshair, sampled at a depth (shift+E)",
                self._new_chedi,
            ),
            (
                "MESH",
                "Mesh list: load, compare, use and back up white/pial meshes -- no subject needed (ctrl+M)",
                self._open_meshes,
            ),
        ):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)

        # Pushed to the right end of the row: it is not part of the load
        # gesture and should not sit in the middle of it.
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
        )
        bar.addWidget(spacer)

        self.debug_button = QtWidgets.QPushButton("DEBUG")
        self.debug_button.setToolTip(
            "Write everything about this session to a text file and copy the\n"
            "path: every layer's grid, affine and voxel size, which one is\n"
            "defining the display, what the mode is reading, and the full\n"
            "recorded script -- which replays the session against the same\n"
            "files. For handing to someone who cannot see your screen."
        )
        self.debug_button.clicked.connect(self._write_debug_report)
        bar.addWidget(self.debug_button)

        # Its own row, under the buttons, so the directory in use and the way
        # back to the others are always on screen. Every directory read or
        # loaded from this session, numbered as the layer list tags its
        # files, then the favourites: picking one reads it.
        dirs = QtWidgets.QToolBar("directories")
        dirs.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, dirs)
        dir_host = QtWidgets.QWidget()
        dir_row = QtWidgets.QHBoxLayout(dir_host)
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(5)
        dir_row.addWidget(self._head("DIR"))
        self._favorites: list[Path] | None = None
        self.dir_box = QtWidgets.QComboBox()
        self.dir_box.setFont(QtGui.QFont(MONO))
        self.dir_box.setToolTip(
            "Directories this session has read or loaded from, numbered as the\n"
            "layer list tags their files, then the favourites (★). Pick one to read it."
        )
        # Never sized by its longest entry: a deep path would set the window's
        # minimum width. The full path is the item's tooltip.
        self.dir_box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.dir_box.setMinimumContentsLength(12)
        self.dir_box.activated.connect(self._pick_directory)
        dir_row.addWidget(self.dir_box, 1)
        self.fav_button = QtWidgets.QPushButton("☆")
        self.fav_button.setCheckable(True)
        self.fav_button.setToolTip(
            f"Keep this directory in the favourites, for every viewer\n({favorites.config_path()})"
        )
        self.fav_button.clicked.connect(self._toggle_favorite)
        dir_row.addWidget(self.fav_button)
        dirs.addWidget(dir_host)

        picks = QtWidgets.QToolBar("data")
        picks.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, picks)

        # One row, not the UNDERLAY / OVERLAY / [+]1 grid this replaces. Those
        # three asked which of two kinds of thing a file was before it had been
        # opened, and the answer was mostly wrong: an anatomical is an overlay
        # when you are checking alignment, a run is an underlay while you set
        # its mask. Loading is loading; where it sits and whether it is drawn
        # are answered afterwards, in the stack, by the layer it became.
        grid_host = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(grid_host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)

        row.addWidget(self._head("DATA"))
        self.data_box = PreviewCombo()
        self.data_box.preview_row = self._preview_row
        self.data_box.setFont(QtGui.QFont(MONO))
        self.data_box.addItem(NONE_LABEL, userData=None)
        # Picking is loading: the extra click on LOAD would be a confirmation
        # of something already unambiguous, and the button stays for the second
        # copy -- the same file over itself, at two thresholds.
        self.data_box.activated.connect(lambda _: self._pick(self.data_box, Load))
        row.addWidget(self.data_box, 1)

        self.load_button = QtWidgets.QPushButton(key_label("LOAD", "^L"))
        self.load_button.setToolTip(
            "Put the chosen dataset on top of the stack (ctrl+L).\n"
            "Press it again for a second copy of the same file.\n"
            "Or drop datasets, meshes, folders or a FreeSurfer subject on any viewer window."
        )
        self.load_button.clicked.connect(lambda: self._pick(self.data_box, Load))
        row.addWidget(self.load_button)

        picks.addWidget(grid_host)

        self._build_window_bar()

    def _build_window_bar(self) -> None:
        """Open and arrange the companion windows."""
        bar = QtWidgets.QToolBar("windows")
        bar.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        # The mode picker lives here rather than as a third row of the dataset
        # grid. That grid's value column is stretched to fit a forty-character
        # filename, and the mode names are the six shortest strings in the
        # interface -- so MODE was the widest control in the window and said
        # the least. Here it is sized to its own text and the row it used to
        # occupy goes back to the window buttons, several of which were being
        # pushed into the toolbar's overflow menu and so were not findable at
        # all.
        # Expanding, or the toolbar hands the host only its size hint -- which
        # a flow layout reports as its widest single item, because it can
        # always wrap. The buttons then flow one per row down a 104-pixel
        # column, which is worse than the overflow menu this replaces.
        flow = FlowBar(spacing=4)
        bar.addWidget(flow)

        flow.addWidget(self._head("MODE"))
        self.mode_box = QtWidgets.QComboBox()
        labels = registry.labels()
        for name in registry.names():
            if registry.get(name).in_selector:
                self.mode_box.addItem(labels[name], userData=name)
        self.mode_box.setCurrentIndex(self.mode_box.findData(self.session.mode.name))
        self.mode_box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mode_box.activated.connect(lambda _: self._switch_mode(self.mode_box.currentData()))
        flow.addWidget(self.mode_box)

        flow.addWidget(self._head("GRID"))
        self.grid_box = QtWidgets.QComboBox()
        self.grid_box.setToolTip(
            "Display voxel size: the grid every layer is painted into.\n"
            "'underlay' draws on the bottom layer's own voxels (the default).\n"
            "'match <layer>' or a size keeps the underlay's field of view and\n"
            "orientation with finer or coarser voxels -- finer shows a\n"
            "high-resolution overlay on its own voxels; coarser shows the\n"
            "anatomy the way a run sees it. Each layer's DRAW still decides\n"
            "how it is painted (cubic sharpens an upsampled underlay).\n"
            "Display only: no layer's data changes."
        )
        self.grid_box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.grid_box.activated.connect(
            lambda i: self._dispatch(SetGridRes(float(self.grid_box.itemData(i))))
        )
        flow.addWidget(self.grid_box)

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
            (
                "CMD",
                "ctrl+P",
                "Print the ffs_viewer command that reopens what is loaded (files, "
                "directory, surfaces) to the terminal, and copy it",
                self._print_launch_command,
            ),
        ):
            b = QtWidgets.QPushButton(key_label(text, key))
            b.setObjectName("tool")
            b.setToolTip(f"{tip} ({key})")
            b.clicked.connect(slot)
            flow.addWidget(b)

        # Names the palette you would switch *to*, not the one you are in: a
        # button labelled with the current state reads as a status light and
        # gets pressed by people who wanted it to stay that way.
        self.theme_button = QtWidgets.QPushButton("")
        self.theme_button.setToolTip("Switch between the dark and light palette (d)")
        self.theme_button.setObjectName("tool")
        self.theme_button.clicked.connect(self._toggle_theme)
        flow.addWidget(self.theme_button)

        flow.addWidget(self._head("T"))
        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.setToolTip("Jump to a volume ( , and . step, v plays )")
        self.time_spin.setKeyboardTracking(False)
        self.time_spin.setMaximumWidth(84)
        self.time_spin.valueChanged.connect(self._time_spin_changed)
        flow.addWidget(self.time_spin)
        self.time_label = QtWidgets.QLabel("")
        flow.addWidget(self.time_label)

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

        Five characters rather than eight: eight still left the THR ON row --
        a picker plus two tick boxes -- as the widest thing in the panel, so
        the panel scrolled sideways and clipped the range numbers off the far
        edge. The popup is where a long label is read, and that is not elided.
        """
        box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        box.setMinimumContentsLength(5)
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
        path = str(entry.path) if entry is not None else None
        if path is not None and cls is Load and self.session.preview_path == path:
            # Already on screen from the picker: keep it rather than open it twice.
            self.session.commit_preview()
        else:
            self._preview_path(None)
            if entry is None:
                return
            self._dispatch(cls(path))
        assert entry is not None
        if self.session.catalog_fresh.pop(entry.path, None) is not None:
            self._sync_catalog()
        self._sync_layer_list()

    def _preview_row(self, row: int | None) -> None:
        """The DATA picker's highlight: show that dataset's first volume, uncommitted."""
        entry = self.data_box.itemData(row) if row is not None else None
        self._preview_path(str(entry.path) if entry is not None else None)

    def _preview_path(self, path: str | None) -> None:
        if path is None and self.session.preview_path is None:
            return
        if path is not None and path == self.session.preview_path:
            return
        try:
            dirty = self.session.preview(path)
        except (OSError, ValueError) as exc:  # a half-written file, a bad header
            self.statusBar().showMessage(f"cannot preview {Path(path or '').name}: {exc}", 5000)
            # The previous preview is already gone; the stack changed either way.
            dirty = Aspect.LAYERS | Aspect.SLICES
        self.refresh(dirty)
        if path is not None:
            self.statusBar().showMessage(f"previewing {Path(path).name}  (volume 0)", 3000)

    def _sync_directories(self, *, reread: bool = False) -> None:
        session = self.session
        current = session.catalog_dir.expanduser().resolve() if session.catalog_dir else None
        if reread or self._favorites is None:
            # Re-read on READ, not per layer-list sync: another viewer may have
            # changed the list, and a READ is when this one is looked at.
            try:
                self._favorites = favorites.load()
            except OSError:
                self._favorites = []
        favs = self._favorites or []
        box = self.dir_box
        box.blockSignals(True)
        box.clear()
        if session.file_list:
            n_files = len(session.file_list)
            box.addItem(f"list  {n_files} file{'s' * (n_files != 1)}", userData=(None, False))
            tip = "\n".join(str(p) for p in session.file_list)
            box.setItemData(box.count() - 1, tip, QtCore.Qt.ItemDataRole.ToolTipRole)
        elif current is None:
            box.addItem("no directory", userData=None)
        for n, (d, recursive) in enumerate(session.directories.items(), start=1):
            star = "  ★" if d in favs else ""
            box.addItem(f"{n}  {d.name or d}{star}", userData=(str(d), recursive))
            box.setItemData(box.count() - 1, str(d), QtCore.Qt.ItemDataRole.ToolTipRole)
        rest = [f for f in favs if f not in session.directories]
        if rest:
            box.insertSeparator(box.count())
            for f in rest:
                box.addItem(f"★  {f.name or f}", userData=(str(f), False))
                box.setItemData(box.count() - 1, str(f), QtCore.Qt.ItemDataRole.ToolTipRole)
        rows = [box.itemData(r) for r in range(box.count())]
        wanted = None if current is None else str(current)
        here = [r for r, data in enumerate(rows) if data and data[0] == wanted]
        box.setCurrentIndex(here[0] if here else 0)
        box.blockSignals(False)
        self.fav_button.setEnabled(current is not None)
        self.fav_button.setChecked(current in favs)
        self.fav_button.setText("★" if current in favs else "☆")

    def _pick_directory(self, row: int) -> None:
        data = self.dir_box.itemData(row)
        if not data:
            return
        directory, recursive = data
        if directory is None:
            if not self.session.reading_files:
                self.read_files(self.session.file_list)
            return
        current = self.session.catalog_dir
        if current is not None and current.expanduser().resolve() == Path(directory):
            return
        try:
            self.read_directory(directory, recursive=recursive)
        except OSError as exc:
            self.statusBar().showMessage(f"read {directory}: {exc}", 8000)
            self._sync_directories()

    def _toggle_favorite(self) -> None:
        if self.session.catalog_dir is None:
            return
        try:
            on = favorites.toggle(self.session.catalog_dir)
        except OSError as exc:
            self.statusBar().showMessage(f"favourites: {exc}", 8000)
        else:
            what = "added to" if on else "removed from"
            self.statusBar().showMessage(f"{self.session.catalog_dir} {what} the favourites", 4000)
        self._sync_directories(reread=True)

    def _read_dialog(self) -> None:
        start = str(self.session.work_dir or Path.cwd())
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Read directory", start)
        if directory:
            self.read_directory(directory)

    def _ica_folder_dialog(self) -> None:
        previous = self.session.mode_named("ica")
        start = str(previous.params.get("folder") or "") if previous is not None else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "ICA folder (melodic_IC + melodic_mix)", start or str(Path.cwd())
        )
        if folder:
            self.open_ica_folder(folder)

    def open_ica_folder(self, folder: str | Path) -> None:
        """The ICA folder reviewer, on ``folder``."""
        from fastfuncstuff.viewer.vocab import SetModeParam

        self._switch_mode("ica")
        self._dispatch(SetModeParam("folder", str(folder)))
        self._prepare_then_refresh()

    def _surfaces_dialog(self) -> None:
        start = self.session.state.surface_subject or os.environ.get(
            "SUBJECTS_DIR", str(Path.cwd())
        )
        subject = QtWidgets.QFileDialog.getExistingDirectory(
            self, "FreeSurfer subject directory", start
        )
        if subject:
            self.load_surfaces(subject)

    def load_surfaces(self, subject: str | Path, space: str = "") -> None:
        try:
            self._dispatch(LoadSurfaces(str(subject), space=space))
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"surfaces: {exc}", 8000)
            return
        hemis = self.session.surfaces.hemis
        self.statusBar().showMessage(
            f"surfaces: {', '.join(hemis)} from {subject} "
            f"({sum(h.n_vertices for h in hemis.values()):,} vertices)",
            5000,
        )

    def _save_surfaces_dialog(self) -> None:
        surfaces = self.session.surfaces
        if not surfaces.edited:
            self.statusBar().showMessage("surfaces: nothing edited to save", 4000)
            return
        names = ", ".join(f"{h}.{s}" for h, s in sorted(surfaces.edited))
        suffix, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save edited surfaces",
            f"Write {names} as copies beside the originals, with suffix:",
            text="ffsedit",
        )
        if not ok or not suffix.strip():
            return
        try:
            self._dispatch(SaveSurfaces(suffix.strip()))
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"surfaces: {exc}", 8000)
            return
        self.statusBar().showMessage(f"surfaces: wrote {names} .{suffix.strip()}", 6000)

    def _layers_dialog(self) -> None:
        from fastfuncstuff.viewer.layerexport import export_dialog

        self._open_tool_dialog(export_dialog(self.session))

    def _install_surfaces_dialog(self) -> None:
        """Replace the subject's surface files with the edits, after asking.

        Not a command, on purpose: a recorded session that replays must never
        overwrite a subject's surfaces. Saving copies (ctrl+shift+S) is the
        recordable way to keep edits.
        """
        surfaces = self.session.surfaces
        if not surfaces.edited:
            self.statusBar().showMessage("surfaces: nothing edited to install", 4000)
            return
        plan = surfaces.install_plan()
        listing = "\n".join(f"  {o.name}  ->  backup {b.name}" for o, b in plan.files)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Install edited surfaces",
            f"Replace these files in {plan.files[0][0].parent} with the edits?\n\n{listing}\n\n"
            "Each original is kept as the backup shown. FreeSurfer stages that "
            "regenerate these surfaces will overwrite them again.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            surfaces.install(plan)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"surfaces: install failed: {exc}", 10000)
            return
        self.statusBar().showMessage(
            f"surfaces: installed {len(plan.files)} files (backups *.pre-ffsedit-*)", 8000
        )

    def _cycle_outlines(self) -> None:
        nxt = next_outlines(self.session.state.surfaces_shown)
        self._dispatch(ShowSurfaces(nxt))
        self.statusBar().showMessage(f"outlines: {nxt or 'off'}", 2000)

    def read_directory(self, directory: str | Path, *, recursive: bool = False) -> None:
        self.session.do(Read(str(directory), recursive=recursive))
        self._sync_catalog()
        self._sync_layer_list()  # a second directory turns the layer numbers on
        self.statusBar().showMessage(
            f"read {len(self.session.catalog)} datasets from {directory}", 5000
        )

    def read_files(self, paths) -> None:
        self.session.do(ReadFiles(tuple(str(p) for p in paths)))
        self._sync_catalog()
        self._sync_layer_list()
        listed = {e.path for e in self.session.catalog}
        bad = [p.name for p in self.session.file_list if p not in listed]
        tail = f"; could not read {', '.join(bad)}" if bad else ""
        self.statusBar().showMessage(f"listed {len(listed)} datasets{tail}", 8000)

    # -- rescanning ------------------------------------------------------
    def _watch(self) -> None:
        """Watch the catalog's directory, or its listed files, and every loaded file.

        Files as well as the directory: an overwrite in place changes a file
        without necessarily touching its directory. Re-armed after every
        rescan, because a file replaced by rename is a new inode and its old
        watch is gone.
        """
        wanted: set[str] = set()
        if self.session.catalog_dir is not None:
            wanted.add(str(self.session.catalog_dir))
        elif self.session.reading_files:
            wanted.update(str(p) for p in self.session.file_list)
        if wanted:
            wanted.update(ly.path for ly in self.session.state.layers if ly.source == "file")
        wanted = {p for p in wanted if Path(p).exists()}
        current = set(self._watcher.directories()) | set(self._watcher.files())
        if current - wanted:
            self._watcher.removePaths(sorted(current - wanted))
        if wanted - current:
            self._watcher.addPaths(sorted(wanted - current))

    def _schedule_rescan(self) -> None:
        if self.session.catalog_dir is not None or self.session.reading_files:
            self._rescan_timer.start()

    def _start_rescan(self, manual: bool = False) -> None:
        """Stat the directory on a worker and header-read only what changed."""
        session = self.session
        if session.catalog_dir is None and not session.reading_files:
            return
        if self._rescan_busy:
            self._rescan_again = True
            return
        self._rescan_busy = True
        if session.catalog_dir is None:
            future = self._rescan_pool.submit(
                catalog_mod.rescan_files, session.file_list, list(session.catalog)
            )
        else:
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
        self._sync_directories(reread=True)
        self._watch()
        c = theme.palette()
        # Two columns, monospaced and padded, so a directory can be scanned by
        # dimensions and volume count rather than read name by name.
        width = max((len(e.name) for e in entries), default=0)
        for box in (self.data_box,):
            # A rescan lands whenever the pipeline writes a file, often while
            # the popup is open on the file about to be picked. Rows are
            # matched back by path, so the pick, the highlight and the scroll
            # survive the refill instead of snapping to the top.
            def path_at(row: int, box=box) -> Path | None:
                entry = box.itemData(row) if row >= 0 else None
                return entry.path if entry is not None else None

            view = box.view()
            popup = view is not None and view.isVisible()
            picked = path_at(box.currentIndex())
            lit = top = None
            if popup and view is not None:
                lit = path_at(view.currentIndex().row())
                top = path_at(view.indexAt(QtCore.QPoint(0, 0)).row())
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
            rows = {path_at(r): r for r in range(1, box.count())}
            box.setCurrentIndex(rows.get(picked, 0))
            if popup and view is not None:
                model = box.model()
                if lit in rows:
                    view.setCurrentIndex(model.index(rows[lit], 0))
                if top in rows:
                    view.scrollTo(
                        model.index(rows[top], 0),
                        QtWidgets.QAbstractItemView.ScrollHint.PositionAtTop,
                    )
            box.blockSignals(False)
            self._fit_picker(box)

    # ------------------------------------------------------------------
    # companion windows
    # ------------------------------------------------------------------
    def _new_image(self) -> None:
        self.manager.open(ViewKind.IMAGE, self._next_plane())
        self.refresh(Aspect.VIEWPORTS | Aspect.SLICES)

    def _new_graph(self, plane: Plane | None = None) -> None:
        self.manager.open(ViewKind.GRAPH, plane or self._next_plane())
        self.refresh(Aspect.VIEWPORTS | Aspect.GRAPH)

    def _new_surface(self) -> None:
        if not self.session.surfaces.hemis:
            self.statusBar().showMessage(
                "surface window: load a subject (SURF) or a mesh (MESH) first", 5000
            )
            return
        self.manager.open(ViewKind.SURFACE, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)

    def _new_chedi(self) -> None:
        if not self.session.surfaces.hemis:
            self.statusBar().showMessage(
                "chedi: load a subject (SURF) or a mesh (MESH) first", 5000
            )
            return
        self.manager.open(ViewKind.CHEDI, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)

    def _open_meshes(self) -> None:
        from fastfuncstuff.viewer.ui.meshwindow import MeshWindow

        if self.mesh_window is None:
            self.mesh_window = MeshWindow(lambda: self.session, self._dispatch, self)
            self.mesh_window.setAcceptDrops(True)
        self.mesh_window.refresh()
        self.mesh_window.show()
        self.mesh_window.raise_()

    def _new_depth(self) -> None:
        if not self.session.surfaces.hemis:
            self.statusBar().showMessage(
                "depth profiles: load a subject (SURF) or a mesh (MESH) first", 5000
            )
            return
        self.manager.open(ViewKind.DEPTH, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)

    def _new_profiles(self) -> None:
        if not self.session.surfaces.hemis:
            self.statusBar().showMessage(
                "profile column: load a subject (SURF) or a mesh (MESH) first", 5000
            )
            return
        self.statusBar().showMessage("profile column: sampling every vertex...", 3000)
        self.manager.open(ViewKind.PROFILE, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)

    def _new_carpet(self) -> None:
        self._open_built(ViewKind.CARPET)

    def _new_matrix(self) -> None:
        self._open_built(ViewKind.MATRIX)

    def _new_clusters(self) -> None:
        vid = self.manager.open(ViewKind.CLUSTERS, Plane.AXIAL)
        self.refresh(Aspect.VIEWPORTS)
        self.refresh_clusters(vid)
        self.manager.redraw(Aspect.SLICES)  # HIDE SMALL starts on

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
        session = ctl.session
        for window in targets:
            layer = session.state.selected_layer()
            window.sync_layers(list(session.state.layers), layer.key if layer else None)
            within = None
            try:
                if layer is not None and window.within is not None:
                    within = session.mask_from_layer(window.within, layer)
                source, table = session.clusterize(
                    None,
                    nn=window.nn,
                    min_voxels=window.min_voxels,
                    mask=within,
                    alpha=window.alpha,
                )
            except (ValueError, KeyError, FileNotFoundError) as exc:
                # On the window, not the status bar: the thing that could not
                # be clustered is the thing you are looking at. The WITHIN mask
                # still applies -- "just show me the brain" needs no threshold.
                session.set_display_mask(window.vid, layer.key if layer else None, within)
                window.show_table("", None, str(exc))
                continue
            keep = within
            if window.hide_small and table.dropped is not None:
                keep = ~table.dropped if keep is None else keep & ~table.dropped
            session.set_display_mask(window.vid, source.key, keep)
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
        start = str((self.session.work_dir or Path.cwd()) / f"{stem}.nii.gz")
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
            # Its mask changes the picture, and nothing else is going to ask
            # the images to redraw after a click on HIDE SMALL or WITHIN.
            self.manager.redraw(Aspect.SLICES)
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

    def _row(self) -> None:
        """Every controller's windows at their own size, in a row at the top right."""
        self.manager.row(self, peers=self._peers())

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
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(5)

        v.addWidget(self._head("LAYERS  [ / ]"))
        # Sized to the stack rather than to the space available. Four layers is
        # the common case and nine is a lot; a fixed 190-pixel box spent most of
        # itself on blank rows, and those pixels came out of the sections below
        # it -- which is how changing a mode parameter pushed the colour bar off
        # the bottom of the controller. Past the cap it scrolls, which is the
        # right thing for the list to give up rather than the whole panel.
        self.layer_list = RowSizedList(max_rows=LAYER_ROWS, min_rows=2)
        self.layer_list.setToolTip(
            "[ and ] step through the stack; space or the tick box hides a layer.\n"
            "A soloed image window draws whichever one is selected here.\n"
            "Right of each shown layer: its value at the crosshair\n"
            "(value / threshold when it is cut on another sub-brick)."
        )
        # The values at the crosshair live here, beside the names they belong
        # to, rather than in a box over the image that hid the very voxels
        # being read. Coalesced: a drag moves the crosshair faster than
        # anyone reads numbers, and each value is a device-to-host copy.
        self.layer_list.setItemDelegate(ValueColumnDelegate(self.layer_list))
        self._values_timer = QtCore.QTimer(self)
        self._values_timer.setSingleShot(True)
        self._values_timer.timeout.connect(self._update_values)
        self._values_clock = QtCore.QElapsedTimer()
        # Rows scrolled into view were skipped while out of it.
        self.layer_list.verticalScrollBar().valueChanged.connect(self._schedule_values)
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

        # The stack's buttons in a narrow column beside the list rather than
        # full-width rows under it: layer names are mostly short, so the list
        # gives up width it was not using and the panel gets the height back.
        stack_column = QtWidgets.QWidget()
        stack_buttons = QtWidgets.QVBoxLayout(stack_column)
        stack_buttons.setContentsMargins(0, 0, 0, 0)
        stack_buttons.setSpacing(2)
        for text, key, tip, slot in (
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
        ):
            b = QtWidgets.QPushButton(key_label(text, key))
            b.setToolTip(f"{tip} ({key})")
            b.setStyleSheet(f"QPushButton {{ font-size: {theme.FONT_SMALL}px; padding: 1px 5px; }}")
            b.clicked.connect(slot)
            stack_buttons.addWidget(b)
        stack_buttons.addStretch(1)
        # Bottom right of the stack: past the cap the list scrolls, and with a
        # long stack that is a lot of scrolling to find one layer.
        self.layers_expand = QtWidgets.QToolButton()
        self.layers_expand.setCheckable(True)
        self.layers_expand.setText("\u25be")
        self.layers_expand.setAutoRaise(True)
        self.layers_expand.setToolTip(
            "Grow the list to show every layer (the window grows with it);\n"
            "press again to cap it at its usual height."
        )
        self.layers_expand.setStyleSheet(
            f"QToolButton {{ font-size: {theme.FONT_SMALL}px; padding: 0px 3px; }}"
        )
        self.layers_expand.toggled.connect(self._expand_layers)
        stack_buttons.addWidget(self.layers_expand, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        stack_column.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred
        )
        # Less the expand button, which hangs below the list's bottom edge
        # rather than making a short stack's list taller than its rows.
        self.layer_list.floor = lambda: (
            stack_column.sizeHint().height()
            - self.layers_expand.sizeHint().height()
            - stack_buttons.spacing()
        )
        layer_row = QtWidgets.QHBoxLayout()
        layer_row.setSpacing(4)
        layer_row.addWidget(self.layer_list, 1)
        # Full height, so the stretch puts the expand button at the bottom right.
        layer_row.addWidget(stack_column, 0)
        v.addLayout(layer_row)

        # The layer form and the colour bar side by side: the bar stands on end
        # to the right of the pickers, so the numbers that define it sit in the
        # column of space the form's labels leave free instead of claiming a
        # full-width band of their own below it.
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        form = QtWidgets.QFormLayout()
        self._layer_form = form
        form.setSpacing(4)
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
            "stepping through the _Coef sub-bricks keeps cutting on each one's _Tstat.\n"
            ", and . then step one pair at a time."
        )
        self.thr_next2_check = QtWidgets.QCheckBox("+2")
        self.thr_next2_check.setToolTip(
            "Threshold on the sub-brick two after OLAY, following it as it changes:\n"
            "a _Coef / _Tstat / _Zstat bucket cut on each _Zstat.\n"
            ", and . then step one triplet at a time."
        )
        for check, mode in self._follow_checks():
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
        thr_row.addWidget(self.thr_next2_check)
        form.addRow(self.thrbrick_head, self.thrbrick_row)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.setIconSize(QtCore.QSize(40, 10))
        for name in available_colormaps():
            self.cmap_box.addItem(colormap_icon(name), name)
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

        # Display only, and the label says so, because the obvious worry on
        # seeing an interpolation control next to a statistic map is that it is
        # touching the numbers. It is not: the voxels, the graphs, the cluster
        # table and the mode's input all read the layer's own array on the
        # layer's own grid whatever this says.
        self.resample_box = QtWidgets.QComboBox()
        self.resample_box.setToolTip(
            "How this layer is PAINTED into the display grid. Nothing else --\n"
            "the data, the graphs, the cluster table and what a mode fits are\n"
            "untouched by it.\n\n"
            "auto: nearest when the layer is coarser than the grid, linear when\n"
            "finer. So a 3 mm run drawn on a 1 mm anatomy keeps its own voxels\n"
            "instead of being smoothed into a resolution it does not have.\n"
            "cubic: sharper than linear, for an anatomy upsampled onto a finer\n"
            "display grid.\n"
            "Press 'e' to cycle."
        )
        for value in RESAMPLE_MODES:
            self.resample_box.addItem(value, userData=value)
        self.resample_box.activated.connect(
            lambda i: self._apply(SetResample, how=str(self.resample_box.itemData(i)))
        )
        self._shrinkable(self.resample_box)
        self.resample_head = self._head(key_label("DRAW", "e"))
        form.addRow(self.resample_head, self.resample_box)

        # How a time series is SHOWN as time is stepped: detrended and/or as
        # percent signal change, so a frame shows activity instead of each
        # voxel's mean. Display only, like DRAW; DERIVE keeps a copy.
        from fastfuncstuff.viewer.ui.gridgraph import DETREND_LABELS

        self.time_row = QtWidgets.QWidget()
        time_row = QtWidgets.QHBoxLayout(self.time_row)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(6)
        self.time_detrend_box = QtWidgets.QComboBox()
        self.time_detrend_box.setToolTip(
            "Show each frame with this much Legendre drift taken out of every\n"
            "voxel's series -- mean, linear, ... -- so stepping through time\n"
            "shows the activity, not the anatomy and the scanner's drift.\n"
            "Display only: nothing is written, and graphs keep their own detrend."
        )
        for order, name in DETREND_LABELS:
            self.time_detrend_box.addItem(name, userData=order)
        self.time_detrend_box.activated.connect(lambda _: self._set_time_view())
        self._shrinkable(self.time_detrend_box)
        time_row.addWidget(self.time_detrend_box, 1)
        self.time_psc_check = QtWidgets.QCheckBox("PSC")
        self.time_psc_check.setToolTip(
            "Percent signal change from each voxel's mean (after any detrend).\n"
            "Zero outside the head, where the mean is ~0."
        )
        self.time_psc_check.toggled.connect(lambda _: self._set_time_view())
        time_row.addWidget(self.time_psc_check)
        self.time_head = self._head("TIME")
        form.addRow(self.time_head, self.time_row)

        # Min, threshold and max are edited on the bar itself. Splitting the
        # number from the picture of the number is what let the bar go stale.
        self.rangebar = RangeBar()
        self.thr_head = self.rangebar.thr_caption
        self.thr_head.setText(key_label("THRESH", "t"))
        self.rangebar.range_changed.connect(self._range_changed)
        self.rangebar.threshold_changed.connect(self._threshold_changed)
        self.rangebar.autorange_requested.connect(self._autorange)
        self.rangebar.negative_tail_requested.connect(self._min_to_negative_tail)
        self.rangebar.reverse_requested.connect(self._reverse_colormap)
        self.rangebar.mirror_changed.connect(lambda on: self._apply(SetRangeMirror, on=on))
        self.rangebar.fixed_changed.connect(lambda on: self._apply(SetRangeFixed, on=on))

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
        self.edges_check = QtWidgets.QCheckBox("edges")
        self.edges_check.setToolTip(
            "Draw the layer as its thin edges, over whatever is below it -- the "
            "alignment check. (e in an image window)"
        )
        self.edges_check.toggled.connect(lambda on: self._apply(SetEdges, on=bool(on)))
        look_row = QtWidgets.QHBoxLayout()
        look_row.setSpacing(12)
        look_row.addWidget(self.boxed_check)
        look_row.addWidget(self.edges_check)
        look_row.addStretch(1)
        form.addRow(QtWidgets.QLabel(""), look_row)

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
        self.range_head = self._head("RANGE")
        bar_column.addWidget(self.range_head)
        bar_column.addWidget(self.rangebar, 1)
        controls.addLayout(bar_column)
        v.addLayout(controls)

        # Above the MODE PARAMETERS header, not under it: the input is not a
        # parameter, it is the thing the parameters are all about, and it is
        # the first question the mode asks. Its own row rather than a declared
        # control, since every mode that reads anything asks it and none of
        # them should have to say so.
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(6)
        self.input_head = self._head("INPUT")
        input_row.addWidget(self.input_head)
        self.input_box = QtWidgets.QComboBox()
        self.input_box.setFont(QtGui.QFont(MONO))
        self.input_box.setToolTip(
            "Which loaded dataset this mode computes from.\n"
            "Independent of what is drawn: a run can be unticked in the stack,\n"
            "or sitting under the anatomy, and still be what the fit reads."
        )
        self.input_box.activated.connect(
            lambda i: self._dispatch_input(self.input_box.itemData(i) or "")
        )
        self._shrinkable(self.input_box)
        input_row.addWidget(self.input_box, 1)
        self.input_row_host = QtWidgets.QWidget()
        self.input_row_host.setLayout(input_row)
        v.addWidget(self.input_row_host)

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
        # The dispatch already recomputed and repainted unless slow work is
        # now pending; refreshing again unconditionally redrew every window a
        # second time on every slider tick.
        if self.session.mode.needs_prepare:
            self._prepare_then_refresh()

    def _mode_action(self, name: str) -> None:
        """Press a mode's button, then finish whatever slow work it started.

        A button may open windows -- Denoise's CARPETS opens two -- and a built
        window opened that way has no picture until someone rebuilds it, so any
        that appeared are queued for the worker.
        """
        before = set(self.session.state.viewports.ids)
        mode = self.session.mode
        inline = (c for c in mode.controls() if isinstance(c, ActionControl))
        if not any(a.name == name for a in (*mode.actions(), *inline)):
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

    def _drain_work(self) -> None:
        """The worker is free: queued preparations first, then window rebuilds.

        Preparations first because they are what a click is waiting on.
        """
        while self._prepare_queue and not self.runner.busy:
            ctl = self._prepare_queue.pop(0)
            if ctl in self.controllers and ctl.session.mode.needs_prepare:
                self._prepare_then_refresh(ctl)
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
                jump_back_key(self._dispatch),
                Binding("Left", "crosshair -x", lambda: self._nudge(0, -1), group="navigate"),
                Binding("Right", "crosshair +x", lambda: self._nudge(0, 1), group="navigate"),
                Binding("Down", "crosshair -y", lambda: self._nudge(1, -1), group="navigate"),
                Binding("Up", "crosshair +y", lambda: self._nudge(1, 1), group="navigate"),
                Binding("PgDown", "crosshair -z", lambda: self._nudge(2, -1), group="navigate"),
                Binding("PgUp", "crosshair +z", lambda: self._nudge(2, 1), group="navigate"),
                Binding("n", "open an image window", self._new_image, group="windows"),
                Binding("shift+n", "open a graph window", self._new_graph, group="windows"),
                Binding("shift+c", "open a carpet plot", self._new_carpet, group="windows"),
                Binding("shift+m", "open a correlation matrix", self._new_matrix, group="windows"),
                Binding("f", "tile every window", self._tile, group="windows"),
                Binding(
                    "\\",
                    "tile the images inside every image window (zoom shared per window)",
                    lambda: self.manager.set_tiles(not self.manager.any_tiled()),
                    group="windows",
                ),
                Binding("shift+f", "stagger every window", self._cascade, group="windows"),
                Binding("r", "raise every window", self._raise_all, group="windows"),
                Binding("d", "dark / light palette", self._toggle_theme, group="windows"),
                Binding(
                    ",",
                    "previous volume (a selected layer that does not follow time: its sub-brick)",
                    lambda: self._step_time(-1),
                    group="time",
                ),
                Binding(
                    ".",
                    "next volume (a selected layer that does not follow time: its sub-brick; "
                    "with THR +1 / +2, one Coef group)",
                    lambda: self._step_time(1),
                    group="time",
                ),
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
                Binding(
                    "o",
                    "show / hide the overlay (the selected layer)",
                    self._toggle_visible,
                    group="layer",
                    aliases=("space",),
                ),
                Binding(
                    "shift+o",
                    "flip the overlay against the layer under it (press again to swap back)",
                    self._flip_with_below,
                    group="layer",
                ),
                Binding("t", "threshold down", lambda: self._nudge_threshold(-0.05), group="layer"),
                Binding(
                    "shift+t", "threshold up", lambda: self._nudge_threshold(0.05), group="layer"
                ),
                Binding("c", "next colormap", self._cycle_colormap, group="layer"),
                Binding("s", "next sign mode", self._cycle_sign, group="layer"),
                Binding("a", "next alpha mode", self._cycle_alpha, group="layer"),
                Binding("b", "toggle boxed", self.boxed_check.toggle, group="layer"),
                Binding(
                    "e",
                    "cycle how it is drawn (auto / nearest / linear)",
                    self._cycle_resample,
                    group="layer",
                ),
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
                Binding("alt+o", "cycle surface outlines", self._cycle_outlines, group="layer"),
                Binding("shift+v", "open a 3-D surface window", self._new_surface, group="windows"),
                Binding(
                    "ctrl+m",
                    "mesh list (compare / use / back up)",
                    self._open_meshes,
                    group="windows",
                ),
                Binding(
                    "shift+e",
                    "CHEDI: cortex laid flat at a depth",
                    self._new_chedi,
                    group="windows",
                ),
                Binding(
                    "shift+l",
                    "open depth profiles (laminar) of the overlay around the crosshair",
                    self._new_depth,
                    group="windows",
                ),
                Binding(
                    "shift+p",
                    "open the ribbon profile column (surface QC)",
                    self._new_profiles,
                    group="windows",
                ),
                Binding(
                    "ctrl+shift+i",
                    "install edited surfaces (replace originals, keep backups)",
                    self._install_surfaces_dialog,
                    group="session",
                ),
                Binding(
                    "ctrl+shift+s",
                    "save edited surfaces (as copies)",
                    self._save_surfaces_dialog,
                    group="session",
                ),
                Binding(
                    "ctrl+shift+l",
                    "LayNii layers from the surfaces as shown (edits included)",
                    self._layers_dialog,
                    group="session",
                ),
                Binding(
                    "ctrl+l",
                    "load the chosen dataset",
                    lambda: self._pick(self.data_box, Load),
                    group="session",
                ),
                Binding("ctrl+s", "save session script", self._save_script_dialog, group="session"),
                Binding(
                    "ctrl+p",
                    "print the command that reopens what is loaded",
                    self._print_launch_command,
                    group="session",
                ),
                Binding("h", "this list", self.help.toggle, group="session"),
            ]
        )

    def _print_launch_command(self) -> None:
        """To the terminal it was started from, and the clipboard: close, paste, back."""
        line = self.session.launch_command()
        print(line, flush=True)
        QtGui.QGuiApplication.clipboard().setText(line.split("  # not reloaded:")[0])
        self.statusBar().showMessage("reopen command printed to the terminal and copied", 6000)

    def _activate_letter(self, letter: str) -> None:
        found = next((c for c in self.controllers if c.letter == letter), None)
        if found is not None:
            self.activate(found)

    def current_key(self) -> str | None:
        layer = self.session.state.selected_layer()
        return None if layer is None else layer.key

    def _set_time_view(self) -> None:
        """Dispatch the TIME row, then re-range: a detrended frame sits around
        zero and a PSC one in percent, so the raw run's range shows nothing."""
        from fastfuncstuff.viewer.vocab import SetLayerTimeView

        key = self.current_key()
        if key is None:
            return
        order = int(self.time_detrend_box.currentData())
        psc = self.time_psc_check.isChecked()
        self._dispatch(SetLayerTimeView(key, order, psc))
        self._autorange()

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

    def _expand_layers(self, on: bool) -> None:
        self.layer_list.set_expanded(on)
        self.layers_expand.setText("\u25b4" if on else "\u25be")

        # Shrink on the way back too: the extra height was this button's.
        def settle() -> None:
            self.fit_to_content(shrink=not on)
            self._schedule_values()  # rows that were scrolled out have none

        QtCore.QTimer.singleShot(0, settle)

    def _row_selected(self, row: int) -> None:
        keys = list(reversed(self.session.state.layers.keys))
        if 0 <= row < len(keys):
            self._dispatch(SelectLayer(keys[row]))

    def _nudge(self, axis: int, delta: int) -> None:
        ijk = list(self.session.state.crosshair)
        ijk[axis] += delta
        self._dispatch(SetIJK(*ijk))

    def _step_time(self, delta: int) -> None:
        """`,` / `.`: time, or the selected layer's sub-bricks (see ``frame_step``)."""
        cmd = self.session.frame_step(delta)
        if cmd is None:
            return
        self._dispatch(cmd)
        if isinstance(cmd, SetVolume):
            self.statusBar().showMessage(self.session.brick_caption(cmd.key), 4000)
            self._sync_layer_controls()

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
        self.statusBar().showMessage(self.session.brick_caption(key), 4000)
        self._sync_layer_controls()

    def _time_spin_changed(self, value: int) -> None:
        self._dispatch(SetIndex(int(value)))

    def _toggle_play(self) -> None:
        if not self._play.isActive() and self.session.state.max_time_index() <= 0:
            self.statusBar().showMessage("nothing to play: no 4-D series is loaded", 4000)
            return
        self._play.toggle()

    def _play_toggled(self, on: bool) -> None:
        self.manager.set_playing(on)
        if not on:
            self.statusBar().showMessage("paused", 1500)

    def _toggle_visible(self) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        self._dispatch(SetLayerVisible(key, not layer.visible))
        self._sync_layer_list()

    def _flip_with_below(self) -> None:
        """A/B the selected layer against the one under it in the stack."""
        key = self.current_key()
        if key is None:
            return
        changes = self.session.state.layers.flip_with_below(key)
        if not changes:
            self.statusBar().showMessage("nothing under this layer to flip against", 2000)
            return
        for layer_key, on in changes:
            self._dispatch(SetLayerVisible(layer_key, on))
        shown = next(
            (self.session.state.layers.get(k).name for k, on in changes if on),
            "nothing",
        )
        self.statusBar().showMessage(f"showing {shown}", 2000)
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
        # Ticking a layer on is choosing to look at it, so the controls follow
        # it there -- through the row, so the selection takes the one path a
        # click on the name does.
        if wanted and self.layer_list.currentItem() is not item:
            self.layer_list.setCurrentItem(item)

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

    def _reverse_colormap(self) -> None:
        layer = self.session.state.layers.find(self.current_key() or "")
        if layer is not None:
            self._dispatch(SetColormapReversed(layer.key, not layer.colormap_reversed))

    def _autorange(self) -> None:
        key = self.current_key()
        if key is None:
            return
        from fastfuncstuff.viewer.session import derive_range

        layer = self.session.state.layers.get(key)
        if layer.time_linked or layer.n_volumes <= 1 or layer.is_computed:
            lo, hi = derive_range(self.session.shown_volume(key))
            self._dispatch(SetRange(key, float(lo), float(hi)))
            return
        # The sub-brick on screen, not the time index -- which on a bucket is
        # always 0, so auto kept re-deriving the F's range whatever was shown.
        look = self.session.overlay_look(key, layer.volume_index, colormap=layer.colormap)
        if "range_lo" in look:
            self._dispatch(SetRange(key, float(look["range_lo"]), float(look["range_hi"])))
        if look.get("colormap", layer.colormap) != layer.colormap:
            self._dispatch(SetColormap(key, str(look["colormap"])))

    def _min_to_negative_tail(self) -> None:
        key = self.current_key()
        if key is None:
            return
        layer = self.session.state.layers.get(key)
        index = 0 if layer.time_linked or layer.is_computed else layer.volume_index
        lo = self.session.negative_tail(key, index)
        if lo is not None and layer.range_hi is not None:
            self._dispatch(SetRange(key, lo, float(layer.range_hi)))

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

    def debug_report(self) -> str:
        """This controller's state as text, with every other controller after it.

        All of them, because a question about one tab is usually a question
        about how it differs from the tab beside it.
        """
        from fastfuncstuff.viewer import report

        parts = [report.describe(self.session)]
        for ctl in self.controllers:
            if ctl.session is not self.session:
                parts.append(report.describe(ctl.session))
        return ("\n\n" + "=" * 78 + "\n\n").join(parts)

    def _write_debug_report(self) -> None:
        """Write the report beside the data, or to a temp file, and say where.

        Beside the data because that is the directory already open in the
        terminal the question is going to be asked from. The path goes on the
        clipboard as well as into the status bar: it exists to be pasted.
        """
        import tempfile
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"ffs_viewer_state_{stamp}.txt"
        directory = self.session.work_dir
        out = Path(directory) / name if directory else Path(tempfile.gettempdir()) / name
        try:
            out.write_text(self.debug_report())
        except OSError:
            # A read-only results directory is normal; the report is worth more
            # than the placement.
            out = Path(tempfile.gettempdir()) / name
            out.write_text(self.debug_report())
        clip = QtWidgets.QApplication.clipboard()
        if clip is not None:
            clip.setText(str(out))
        self.statusBar().showMessage(f"wrote {out} (path copied)", 8000)

    def open_path(self, path: str | Path) -> None:
        """Open one dataset, on top of whatever is already loaded."""
        self._dispatch(Load(str(path)))
        self._sync_layer_list()

    # -- dropped files ---------------------------------------------------
    def _takes_drops(self, window: QtWidgets.QWidget) -> bool:
        return window is self or window is self.mesh_window or self._owner_of(window) is not None

    def _owner_of(self, window: QtWidgets.QWidget) -> Controller | None:
        for ctl in self.controllers:
            if any(window is win for win in ctl.manager.windows.values()):
                return ctl
        return None

    def _on_files_dropped(self, window: QtWidgets.QWidget, paths: list[str]) -> None:
        # A file dropped on B's axial is meant for B, whichever tab is showing.
        owner = self._owner_of(window)
        if owner is not None:
            self.activate(owner)
        self.open_dropped(paths)

    def open_dropped(self, paths: list[str]) -> None:
        """Open a drop: a subject as SURF does, a folder as the command line does.

        Files go on top of the stack, with their own folders listed unread.
        """
        dropped = catalog_mod.sort_dropped(paths)
        datasets, meshes = dropped.datasets, dropped.meshes
        session = self.session
        self._preview_path(None)
        parts = []
        if dropped.subjects:
            # SURF asks first in a dialog; a drop can be a slip, and loading a
            # subject throws away unsaved edits with no undo.
            if session.surfaces.edited:
                parts.append(f"surfaces have unsaved edits: not loading {dropped.subjects[0].name}")
            else:
                self.load_surfaces(dropped.subjects[0])
                parts.append(self.statusBar().currentMessage())
            if dropped.subjects[1:]:
                names = ", ".join(s.name for s in dropped.subjects[1:])
                parts.append(f"one subject at a time, ignored {names}")
        if dropped.directories:
            first, *rest = dropped.directories
            session.offer_directories(rest)
            try:
                self.read_directory(first)
                parts.append(self.statusBar().currentMessage())
            except OSError as exc:
                self._sync_directories()
                parts.append(f"read {first}: {exc}")
        # A LOAD lists its own folder; a mesh's has to be asked for. Same test as
        # a LOAD's: a folder already read recursively covers what is inside it.
        session.offer_directories(m.parent for m in meshes if session.directory_number(m) is None)
        meshes_before = len(session.surfaces.meshes)
        # Meshes first: a surface result needs the mesh it was made on to be there.
        if meshes:
            self._open_meshes()
            assert self.mesh_window is not None
            for mesh in meshes:
                self.mesh_window.load_path(str(mesh))
        failed = []
        for path in datasets:
            try:
                self.open_path(path)
            # Not just OSError/ValueError: nibabel's ImageFileError is a bare
            # Exception, and one bad file must not lose the rest of the drop.
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
        self._sync_layer_list()
        if datasets or meshes:
            # A mesh that failed says why in the mesh window, which is open by now.
            took = len(session.surfaces.meshes) - meshes_before + len(datasets) - len(failed)
            parts.append(f"loaded {took} of {len(datasets) + len(meshes)} dropped")
        if failed:
            parts.append("could not load " + "; ".join(failed))
        if dropped.skipped:
            names = ", ".join(p.name for p in dropped.skipped)
            parts.append(f"not a dataset or mesh: {names}")
        self.statusBar().showMessage("; ".join(parts), 8000)

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
        self.vram_label = QtWidgets.QLabel("")
        self.vram_label.setToolTip(
            "GPU memory the viewer holds, of its budget (-vram_budget). Everything "
            "counted here also has a RAM copy, so the viewer hands it back when "
            "another process needs the card."
        )
        self.statusBar().addPermanentWidget(self.vram_label)

    def _tick_vram(self) -> None:
        store = self.session.store
        if store.device.type == "cpu":
            self.vram_label.hide()
            return
        released = store.relieve()
        if released:
            self.statusBar().showMessage(
                f"the GPU is short: handed {released / 2**30:.1f} GB back (kept in RAM)", 5000
            )
        used, budget = store.device_used_all(), store.device_budget
        self.vram_label.setText(f"GPU {used / 2**30:.1f}/{budget / 2**30:.1f} GB")
        self.vram_label.show()

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

    def _queue_prepare(self, ctl: Controller) -> None:
        """Prepare a controller's mode now, or once the worker is free."""
        if not self.runner.busy:
            self._prepare_then_refresh(ctl)
        elif ctl not in self._prepare_queue:
            self._prepare_queue.append(ctl)

    def _prepare_then_refresh(self, ctl: Controller | None = None) -> None:
        """Run the mode's slow half on a worker, then install its overlay."""
        ctl = self._active if ctl is None else ctl

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
            self._sync_mode_panel()
        elif dirty & (Aspect.COLORMAP | Aspect.THRESHOLD | Aspect.SLICES):
            # The bar is a view of colormap, range and threshold, so it has to
            # follow those aspects and not only LAYERS -- listening for the
            # wrong one is what left it showing the previous colour scale.
            self._sync_layer_controls()
            self._sync_layer_checks()
            # A mode's parameters can be a reading of the picture rather than
            # an input to it -- Align's sliders follow a drag in an image
            # window -- so they are re-read whenever the picture moves.
            self.mode_panel.sync_values(self.session.mode.params)
        self._refresh_windows(self._active, dirty)
        if self.mesh_window is not None and dirty & (Aspect.SLICES | Aspect.THEME):
            if dirty & Aspect.THEME:
                self.mesh_window.restyle()
            self.mesh_window.refresh()
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

    def _sync_grid_box(self) -> None:
        st = self.session.state
        items = grid_res_choices(st)
        self.grid_box.blockSignals(True)
        self.grid_box.clear()
        for label, mm in items:
            self.grid_box.addItem(label, userData=mm)
        current = next(i for i, (_, mm) in enumerate(items) if abs(mm - st.grid_mm) <= 1e-6)
        self.grid_box.setCurrentIndex(current)
        self.grid_box.setEnabled(st.layers.base is not None)
        self.grid_box.blockSignals(False)

    def _sync_layer_list(self) -> None:
        """Bring the stack's rows up to date, in place whenever the stack's order allows.

        Rebuilt only when layers came, went or moved. Every selection is a
        LAYERS refresh, and clearing the list for it threw the scroll position
        away, so with a long stack the row just clicked could land off screen.
        Even a rebuild keeps the scroll where it was.
        """
        self._watch()
        self._sync_grid_box()
        selected = self.current_key()
        listed = self.layer_list
        listed.blockSignals(True)
        self._sync_directories()  # a load can add a directory
        keys = list(reversed(self.session.state.layers.keys))
        # Which directory a file came from, by its number in the directory
        # box -- only once there is more than one to tell apart.
        numbered = len(self.session.directories) > 1
        rows: list[tuple[str, str, str, QtCore.Qt.CheckState]] = []
        for layer in reversed(list(self.session.state.layers)):
            try:
                pending = self.session.store.get(layer.key).pending
            except KeyError:
                pending = False
            tag = " ·computed" if layer.is_computed else (" ·loading" if pending else "")
            n = self.session.directory_number(layer.path) if layer.source == "file" else None
            where = f"{n}  " if numbered and n is not None else ""
            check = (
                QtCore.Qt.CheckState.Checked if layer.visible else QtCore.Qt.CheckState.Unchecked
            )
            tip = layer.path if layer.source == "file" else ""
            rows.append((layer.key, f"{where}{layer.name}{tag}", tip, check))
        role = QtCore.Qt.ItemDataRole.UserRole
        present = [listed.item(r).data(role) for r in range(listed.count())]
        if present == keys:
            # Setters only where something differs: each one is a dataChanged
            # and a repaint of its row.
            for r, (_key, text, tip, check) in enumerate(rows):
                item = listed.item(r)
                if item.text() != text:
                    item.setText(text)
                if item.toolTip() != tip:
                    item.setToolTip(tip)
                if item.checkState() != check:
                    item.setCheckState(check)
        else:
            scroll = listed.verticalScrollBar().value()
            listed.clear()
            for key, text, tip, check in rows:
                item = QtWidgets.QListWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(check)
                item.setData(role, key)
                listed.addItem(item)
            # Laid out now, or the restored value is clamped to an empty range.
            listed.doItemsLayout()
            listed.verticalScrollBar().setValue(scroll)
        if keys:
            row = keys.index(selected) if selected in keys else 0
            if listed.currentRow() != row:
                listed.setCurrentRow(row)
        # Unblocked only after the row is set. A sync that writes back into
        # state is a refresh that dispatches, which puts a SELECT_LAYER into
        # the recording for every repaint and can recurse.
        listed.blockSignals(False)
        listed.rows_changed()
        self._sync_layer_controls()
        self._schedule_values()

    def _sync_layer_checks(self) -> None:
        """The tick boxes alone, for a visibility change made in another window.

        The rows follow LAYERS; `o` and `O` in an image window are SLICES --
        which a threshold drag is too, so the full row sync is too heavy to
        hang on it -- and the list sat showing a hidden layer as drawn.
        """
        listed = self.layer_list
        role = QtCore.Qt.ItemDataRole.UserRole
        listed.blockSignals(True)
        for r in range(listed.count()):
            item = listed.item(r)
            layer = self.session.state.layers.find(item.data(role))
            if layer is None:
                continue
            check = (
                QtCore.Qt.CheckState.Checked if layer.visible else QtCore.Qt.CheckState.Unchecked
            )
            if item.checkState() != check:
                item.setCheckState(check)
        listed.blockSignals(False)

    def _dispatch_input(self, key: str) -> None:
        self._dispatch(SetInput(key))
        self._prepare_then_refresh()

    def _sync_input_box(self) -> None:
        """List what the mode could read, and point at what it does read.

        Hidden entirely for a mode that reads nothing from the stack, and shown
        disabled with a reason when there is nothing it could read -- an empty
        drop-down says "pick one" about a list with nothing in it.
        """
        mode = self.session.mode
        if mode.input_kind == "none":
            self.input_row_host.setVisible(False)
            return
        self.input_row_host.setVisible(True)
        candidates = self.session.input_candidates()
        current = self.session.input_layer()
        self.input_box.blockSignals(True)
        self.input_box.clear()
        if not candidates:
            wanted = "a 4-D time series" if mode.input_kind == "4d" else "a dataset"
            self.input_box.addItem(f"(load {wanted})", userData="")
            self.input_box.setEnabled(False)
        else:
            self.input_box.setEnabled(True)
            # "(auto)" names the default rather than hiding it, so a stack with
            # two runs in it does not silently fit one of them.
            head = candidates[0].name if candidates else ""
            self.input_box.addItem(f"(auto: {head})", userData="")
            for layer in candidates:
                self.input_box.addItem(f"{layer.name} [{layer.key}]", userData=layer.key)
            named = self.session.state.input_key
            index = self.input_box.findData(named) if named else 0
            self.input_box.setCurrentIndex(index if index >= 0 else 0)
        self.input_box.blockSignals(False)
        self.input_box.setToolTip(
            "Which loaded dataset this mode computes from"
            + (f" -- now {current.name}.\n" if current is not None else ".\n")
            + "Independent of what is drawn: a run can be unticked in the stack,\n"
            "or sitting under the anatomy, and still be what the fit reads."
        )

    def _sync_mode_panel(self) -> None:
        mode = self.session.mode
        self.mode_panel.rebuild(mode.panel_controls(), mode.params, mode.panel_actions())
        has = bool(mode.panel_controls()) or bool(mode.panel_actions())
        self.mode_head.setVisible(has)
        self.mode_panel.setVisible(has)
        self._sync_input_box()
        # After the layout has taken the rebuild in.
        changed = mode.name != getattr(self, "_fitted_mode", None)
        self._fitted_mode = mode.name
        if self.isVisible():
            QtCore.QTimer.singleShot(0, lambda: self.fit_to_content(shrink=changed))
        # A mode reached by a button is shown while it is active, so the box
        # never claims a mode that is not running, and dropped when it is not.
        self.mode_box.blockSignals(True)
        for i in reversed(range(self.mode_box.count())):
            name = self.mode_box.itemData(i)
            if not registry.get(name).in_selector and name != mode.name:
                self.mode_box.removeItem(i)
        if self.mode_box.findData(mode.name) < 0:
            self.mode_box.addItem(mode.label or mode.name, userData=mode.name)
        self.mode_box.blockSignals(False)
        idx = self.mode_box.findData(mode.name)
        if idx >= 0 and idx != self.mode_box.currentIndex():
            self.mode_box.blockSignals(True)
            self.mode_box.setCurrentIndex(idx)
            self.mode_box.blockSignals(False)

    def _cycle_resample(self) -> None:
        """auto -> nearest -> linear -> cubic -> auto on the selected layer.

        A cycle rather than a toggle because the useful gesture is comparing
        them against the automatic choice, and every state is one key.
        """
        layer = self.session.state.selected_layer()
        if layer is None:
            return
        if self.session.is_label_layer(layer):
            self.statusBar().showMessage(f"{layer.name} holds labels: always drawn nearest", 4000)
            return
        order = RESAMPLE_MODES
        nxt = (
            order[(order.index(layer.resample) + 1) % len(order)]
            if layer.resample in order
            else "auto"
        )
        self._apply(SetResample, how=nxt)

    def _sync_colour_rows(self) -> None:
        """A colour layer has no colour map, sign, alpha or range: hide those rows.

        What it has instead -- how its channels become a colour -- is the RGB
        mode's panel.
        """
        layer = self.session.state.selected_layer()
        is_colour = layer is not None and layer.colour
        for widget in (self.cmap_box, self.sign_box, self.alpha_row):
            self._layer_form.setRowVisible(widget, not is_colour)
        self.rangebar.setVisible(not is_colour)
        self.range_head.setVisible(not is_colour)

    def _sync_layer_controls(self) -> None:
        self._sync_colour_rows()
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
            self.edges_check,
            self.resample_box,
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
            (self.edges_check, layer.edges, True),
            (self.alpha_linear_check, layer.alpha_mode is AlphaMode.LINEAR, True),
            (self.alpha_quad_check, layer.alpha_mode is AlphaMode.QUADRATIC, True),
            (self.roi_check, layer.roi, not layer.is_computed),
            (self.timelink_check, layer.time_linked, layer.n_volumes > 1),
        ):
            check.blockSignals(True)
            check.setChecked(value)
            check.setEnabled(enabled)
            check.blockSignals(False)
        # "auto (nearest)" rather than "auto": the whole value of the automatic
        # setting is knowing which way it went, and a layer that looks blocky
        # while the box says "auto" is a question rather than an answer.
        resolved = self.session.resample_mode(layer)
        labels = self.session.is_label_layer(layer)
        self.resample_box.blockSignals(True)
        self.resample_box.setItemText(0, "labels (nearest)" if labels else f"auto ({resolved})")
        index = 0 if labels else self.resample_box.findData(layer.resample)
        self.resample_box.setCurrentIndex(max(index, 0))
        # Labels have no in-between to interpolate: offering linear would be
        # offering a picture of regions that do not exist.
        self.resample_box.setEnabled(not labels)
        self.resample_box.blockSignals(False)
        timed = layer.time_linked and layer.n_volumes > 1
        self.time_head.setVisible(timed)
        self.time_row.setVisible(timed)
        for widget in (self.time_detrend_box, self.time_psc_check):
            widget.blockSignals(True)
        self.time_detrend_box.setCurrentIndex(
            max(self.time_detrend_box.findData(layer.time_detrend), 0)
        )
        self.time_psc_check.setChecked(layer.time_psc)
        for widget in (self.time_detrend_box, self.time_psc_check):
            widget.blockSignals(False)
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
        for check, mode in self._follow_checks():
            check.blockSignals(True)
            check.setChecked(layer.threshold_follow == mode)
            check.blockSignals(False)

    def _follow_checks(self) -> tuple[tuple[QtWidgets.QCheckBox, str], ...]:
        return (
            (self.thr_same_check, "same"),
            (self.thr_next_check, "next"),
            (self.thr_next2_check, "next2"),
        )

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
        self._schedule_values()

    def _schedule_values(self, *_args) -> None:
        """Bring the value column up to date -- now, or once the throttle allows."""
        if self._values_timer.isActive():
            return
        wait = VALUES_EVERY_MS - (
            self._values_clock.elapsed() if self._values_clock.isValid() else VALUES_EVERY_MS
        )
        if wait <= 0:
            self._update_values()
        else:
            self._values_timer.start(int(wait))

    def _update_values(self) -> None:
        """Each shown layer's value at the crosshair, beside its row in the list.

        Only rows on screen are read: a stack scrolled past its cap would
        otherwise pay a device-to-host copy per off-screen layer per move. A
        row's data is written only when its text changed, so the repaint is
        the rows whose numbers moved and the list is never rebuilt. The status
        bar names what the selected layer's numbers are, plus the subject's
        own atlas regions -- the readout lines that have no row.
        """
        self._values_clock.restart()
        st = self.session.state
        if st.grid is None:
            self.value_label.setText("")
            return
        listed = self.layer_list
        view = listed.viewport().rect()
        selected = self.current_key()
        selected_text = ""
        listed.blockSignals(True)
        try:
            for row in range(listed.count()):
                item = listed.item(row)
                key = item.data(QtCore.Qt.ItemDataRole.UserRole)
                layer = st.layers.find(key) if key is not None else None
                text = ""
                # The selected layer is read wherever it is scrolled to: the
                # status bar speaks for it.
                if (
                    layer is not None
                    and layer.visible
                    and (key == selected or listed.visualItemRect(item).intersects(view))
                ):
                    reading = self.session.layer_reading(layer)
                    if reading is not None:
                        text = reading.short
                        if key == selected:
                            selected_text = f"{layer.name}  {reading.long}"
                if (item.data(VALUE_ROLE) or "") != text:
                    item.setData(VALUE_ROLE, text)
        finally:
            listed.blockSignals(False)
        regions = self.session.surfaces.region_lines(
            st.crosshair_mm, st.surface_annot, st.volume_atlas
        )
        self.value_label.setText("   ".join([t for t in (selected_text, *regions) if t]))

    def dock_left(self) -> None:
        """Park the controller at the top-left of its screen, as tall as it needs.

        Left rather than wherever the window system drops it, so tiling always
        has one contiguous region to its right to fill.
        """
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.topLeft())
        # Sized once shown: before that the toolbars have no real geometry.
        QtCore.QTimer.singleShot(0, lambda: self.fit_to_content(shrink=True))

    def fit_to_content(self, shrink: bool = False) -> None:
        """As tall as the panel holds, up to the screen.

        Full height left View mode -- no parameters -- as a column of empty
        stretch between the stack and the status bar. Grows whenever there is
        more to show; shrinks only when asked (a mode change), so a window made
        taller by hand is not pulled back every time a layer loads.
        """
        scroll = self.centralWidget()
        if not isinstance(scroll, QtWidgets.QScrollArea):
            return
        panel = scroll.widget()
        layout = panel.layout() if panel is not None else None
        if panel is None or layout is None:
            return
        layout.activate()
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        frame = self.frameGeometry().height() - self.height()
        cap = screen.availableGeometry().height() - frame if screen is not None else 10**6
        chrome = self.height() - scroll.viewport().height()
        want = min(panel.sizeHint().height() + chrome, cap)
        if want > self.height() or (shrink and want < self.height()):
            self.resize(self.width(), want)

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
    directories: list[str] | None = None,
    files: list[str] | None = None,
    surfaces: str | None = None,
    meshes: list[str] | None = None,
    surf_space: str | None = None,
    surf_data: list | None = None,
    vram_budget: str | None = None,
    say: Callable[[str], None] = lambda _text: None,
    started: float | None = None,
    on_shown: Callable[[], None] | None = None,
) -> int:
    """Open the controller and run the Qt loop, saying what startup is doing.

    ``on_shown`` runs once the loop is turning, so whatever stood in for the
    controller stays up until the controller has had its chance to paint.
    """
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.viewer.residency import apply_vram_budget

    app = application()
    session = ViewerSession(device=setup_device(device))
    apply_vram_budget(vram_budget, session.store.device)
    say(f"building the window, device {session.store.device}")
    win = ViewerWindow(session)
    quit_on_interrupt(app, win)

    # Named paths only. Reading a dataset's directory uninvited can fetch every
    # file in it on a synced folder, and the dropdown is one click from it.
    if directories:
        say(f"reading {directories[0]}")
        win.read_directory(directories[0])
        session.offer_directories(directories[1:])
        win._sync_layer_list()
        say(f"found {len(session.catalog)} datasets")
    for p in catalog_mod.opening(paths):
        win.open_path(p)
    # After the opens: the headers it reads are in hand by now, so all the waiting
    # on a slow file happens under the bar that names it.
    if files:
        win.read_files(files)
    if surfaces:
        say(f"loading surfaces from {surfaces}")
        win.load_surfaces(surfaces, space=surf_space or "")
    if meshes:
        win._open_meshes()
        assert win.mesh_window is not None
        for m in meshes:
            # The mesh window's loader asks for whatever the name doesn't say.
            win.mesh_window.load_path(m)
    for cmd in surf_data or []:
        win.refresh(session.do(cmd))
    if script:
        say(f"replaying {script}")
        win.refresh(session.run_script(Path(script).read_text()))
    win.dock_left()
    win.show()
    if on_shown is not None:
        QtCore.QTimer.singleShot(0, on_shown)
    win._row()
    took = "" if started is None else f" in {time.perf_counter() - started:.1f} s"
    say(f"ready{took}")
    return app.exec()


__all__ = ["ViewerWindow", "launch"]
