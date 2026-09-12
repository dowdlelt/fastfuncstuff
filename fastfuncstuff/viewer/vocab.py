"""The command vocabulary and its handlers.

This is the table AFNI grew as ``afni_driver.c``. Names echo AFNI's where the
concept survived (``SET_IJK``, ``SET_THRESHOLD``) and diverge where the data
model did: there is no ``SET_UNDERLAY`` because there is no underlay role, only
a layer at the bottom of the stack.

Handlers return the aspects they actually dirtied, which is usually narrower
than what the command declares -- a crosshair move onto the same voxel dirties
nothing, and the UI is then free to skip the repaint entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastfuncstuff.viewer.commands import Aspect, Command, CommandBus, command
from fastfuncstuff.viewer.layers import AlphaMode, Layer, SignMode
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.viewports import ViewKind, clamp_grid

# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class SetIJK(Command):
    """Move the crosshair to a display-grid voxel."""

    name = "SET_IJK"
    aspects = Aspect.CROSSHAIR | Aspect.GRAPH
    i: int
    j: int
    k: int


@command
@dataclass(frozen=True)
class SetXYZ(Command):
    """Move the crosshair to a scanner-millimetre location."""

    name = "SET_XYZ"
    aspects = Aspect.CROSSHAIR | Aspect.GRAPH
    x: float
    y: float
    z: float


@command
@dataclass(frozen=True)
class SetIndex(Command):
    """Set the time index (volume) shown across the stack."""

    name = "SET_INDEX"
    aspects = Aspect.TIME | Aspect.SLICES | Aspect.GRAPH
    index: int


# ---------------------------------------------------------------------------
# viewports: the open windows
#
# Every one of these names a window. That is the whole point of the group --
# zoom, pan and "what am I locked to" were state on the viewer until there was
# more than one window, at which point they stopped describing anything.
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class SetTheme(Command):
    """Switch the interface palette. ``dark`` or ``light``."""

    name = "SET_THEME"
    aspects = Aspect.THEME
    theme: str


@command
@dataclass(frozen=True)
class OpenView(Command):
    """Open an image or graph window.

    The id is given rather than returned because a command has to be fully
    determined to replay: a script that says ``OPEN_VIEW V2 image axial``
    rebuilds the same window, where one that minted an id at replay time would
    drift from every later line that addresses it.
    """

    name = "OPEN_VIEW"
    aspects = Aspect.VIEWPORTS
    major = True
    view: str
    kind: str = "image"
    plane: str = "axial"


@command
@dataclass(frozen=True)
class CloseView(Command):
    name = "CLOSE_VIEW"
    aspects = Aspect.VIEWPORTS
    major = True
    view: str


@command
@dataclass(frozen=True)
class SetViewPlane(Command):
    """Point a window at a different plane, without opening another."""

    name = "SET_VIEW_PLANE"
    aspects = Aspect.VIEWPORTS | Aspect.SLICES
    view: str
    plane: str


@command
@dataclass(frozen=True)
class SetViewSolo(Command):
    """Draw only the selected layer in this window, instead of the stack."""

    name = "SET_VIEW_SOLO"
    aspects = Aspect.VIEWPORTS | Aspect.SLICES
    view: str
    on: bool


@command
@dataclass(frozen=True)
class SetViewLocked(Command):
    """Whether this window follows the shared crosshair and time index."""

    name = "SET_VIEW_LOCKED"
    aspects = Aspect.VIEWPORTS
    view: str
    on: bool


@command
@dataclass(frozen=True)
class SetViewPosition(Command):
    """Park an unlocked window on one slice. ``-`` puts it back on follow."""

    name = "SET_VIEW_POSITION"
    aspects = Aspect.VIEWPORTS | Aspect.SLICES
    view: str
    position: int | None = None


@command
@dataclass(frozen=True)
class SetViewGrid(Command):
    """Cells per side in a graph window: 1 -> 1 voxel, 3 -> 9, 4 -> 16."""

    name = "SET_VIEW_GRID"
    aspects = Aspect.VIEWPORTS | Aspect.GRAPH
    view: str
    n: int


@command
@dataclass(frozen=True)
class SetViewTraces(Command):
    """Which layers a graph window plots, as comma-separated layer keys.

    Empty (or ``-``) means every time-linked layer, which is both the sensible
    default and the only selection that keeps meaning something as the stack
    grows. Naming keys is what lets a graph stay on the functional while the
    images show a stat map on an anatomy.
    """

    name = "SET_VIEW_TRACES"
    aspects = Aspect.VIEWPORTS | Aspect.GRAPH
    view: str
    keys: str = ""


@command
@dataclass(frozen=True)
class SetViewSharedScale(Command):
    name = "SET_VIEW_SHARED_SCALE"
    aspects = Aspect.VIEWPORTS | Aspect.GRAPH
    view: str
    on: bool


@command
@dataclass(frozen=True)
class SetViewGeometry(Command):
    """Where a window sits on screen, so a saved session comes back tiled."""

    name = "SET_VIEW_GEOMETRY"
    aspects = Aspect.VIEWPORTS
    view: str
    x: int
    y: int
    w: int
    h: int


@command
@dataclass(frozen=True)
class SetZoom(Command):
    """Zoom one window. Zoom is per window; there is no viewer-wide zoom."""

    name = "SET_ZOOM"
    aspects = Aspect.VIEWPORTS | Aspect.SLICES
    view: str
    zoom: float


@command
@dataclass(frozen=True)
class SetPan(Command):
    name = "SET_PAN"
    aspects = Aspect.VIEWPORTS | Aspect.SLICES
    view: str
    x: float
    y: float


# ---------------------------------------------------------------------------
# the data selector
#
# These are the core of the viewer. Everything else -- panes, graphs, modes --
# is modular on top of picking a directory, a base image and what goes over it.
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class Read(Command):
    """Read a directory into the catalog the pickers draw from."""

    name = "READ"
    aspects = Aspect.NOTHING
    major = True
    directory: str
    recursive: bool = False


@command
@dataclass(frozen=True)
class SetUnderlay(Command):
    """Replace the base image, keeping whatever is stacked over it."""

    name = "SET_UNDERLAY"
    aspects = Aspect.LAYERS | Aspect.SLICES | Aspect.GRID
    major = True
    path: str
    key: str = ""


@command
@dataclass(frozen=True)
class SetOverlay(Command):
    """Replace the primary overlay. Extra overlays are left alone."""

    name = "SET_OVERLAY"
    aspects = Aspect.LAYERS | Aspect.SLICES
    major = True
    path: str
    key: str = ""


@command
@dataclass(frozen=True)
class AddOverlay(Command):
    """Push another overlay on top of the stack -- the +1 button."""

    name = "ADD_OVERLAY"
    aspects = Aspect.LAYERS | Aspect.SLICES
    major = True
    path: str
    key: str = ""


@command
@dataclass(frozen=True)
class SetMode(Command):
    """Switch where the overlay comes from."""

    name = "SET_MODE"
    aspects = Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH
    major = True
    mode: str


@command
@dataclass(frozen=True)
class SetModeParam(Command):
    """Set one parameter of the active mode.

    The value is carried as text so the vocabulary stays closed over scalars and
    a recorded script keeps round-tripping; the mode coerces it to its control's
    declared type.
    """

    name = "SET_MODE_PARAM"
    aspects = Aspect.LAYERS | Aspect.SLICES
    param: str
    value: str


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class AddLayer(Command):
    """Load a dataset and push it onto the stack.

    Major: this is one of the points a replay should be able to stop at.
    """

    name = "ADD_LAYER"
    aspects = Aspect.LAYERS | Aspect.SLICES | Aspect.GRID
    major = True
    path: str
    key: str = ""


@command
@dataclass(frozen=True)
class RemoveLayer(Command):
    name = "REMOVE_LAYER"
    aspects = Aspect.LAYERS | Aspect.SLICES
    key: str


@command
@dataclass(frozen=True)
class MoveLayer(Command):
    """Reorder a layer; 0 is the bottom of the stack."""

    name = "MOVE_LAYER"
    aspects = Aspect.LAYERS | Aspect.SLICES
    key: str
    to: int


@command
@dataclass(frozen=True)
class SelectLayer(Command):
    """Aim the controls at one layer -- and say what a soloed window draws."""

    name = "SELECT_LAYER"
    aspects = Aspect.LAYERS
    key: str


@command
@dataclass(frozen=True)
class SetLayerVisible(Command):
    name = "SET_LAYER_VISIBLE"
    aspects = Aspect.SLICES
    key: str
    on: bool


@command
@dataclass(frozen=True)
class SetLayerOpacity(Command):
    name = "SET_LAYER_OPACITY"
    aspects = Aspect.SLICES
    key: str
    opacity: float


@command
@dataclass(frozen=True)
class SetVolume(Command):
    """Choose which sub-brick a layer displays."""

    name = "SET_VOLUME"
    aspects = Aspect.SLICES | Aspect.GRAPH
    key: str
    index: int


@command
@dataclass(frozen=True)
class SetThresholdIndex(Command):
    """Which sub-brick supplies the threshold statistic.

    ``-`` thresholds on the sub-brick being displayed, which is what a plain
    intensity map wants. Naming a different one is the stats case: colour by
    the coefficient, threshold on its t.
    """

    name = "SET_THRESHOLD_INDEX"
    aspects = Aspect.THRESHOLD | Aspect.SLICES
    key: str
    index: int | None = None


@command
@dataclass(frozen=True)
class SetTimeLinked(Command):
    """Whether the global time index drives this layer.

    The override for when a 4-D file's own header cannot say whether its
    sub-bricks are time points or unrelated contrasts.
    """

    name = "SET_TIME_LINKED"
    aspects = Aspect.SLICES | Aspect.GRAPH | Aspect.TIME
    key: str
    on: bool


# ---------------------------------------------------------------------------
# colour and threshold
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class SetColormap(Command):
    name = "SET_COLORMAP"
    aspects = Aspect.COLORMAP
    key: str
    colormap: str


@command
@dataclass(frozen=True)
class SetPanes(Command):
    """Number of discrete colour panes; 0 for a continuous scale."""

    name = "SET_PANES"
    aspects = Aspect.COLORMAP
    key: str
    panes: int


@command
@dataclass(frozen=True)
class SetSign(Command):
    """Show both signs, positive only, or negative only."""

    name = "SET_SIGN"
    aspects = Aspect.COLORMAP
    key: str
    mode: str


@command
@dataclass(frozen=True)
class SetRange(Command):
    name = "SET_RANGE"
    aspects = Aspect.COLORMAP
    key: str
    lo: float
    hi: float


@command
@dataclass(frozen=True)
class SetThreshold(Command):
    name = "SET_THRESHOLD"
    aspects = Aspect.THRESHOLD
    key: str
    value: float


@command
@dataclass(frozen=True)
class SetAlpha(Command):
    """Set the sub-threshold fade mode (off / linear / quadratic)."""

    name = "SET_ALPHA"
    aspects = Aspect.THRESHOLD
    key: str
    mode: str


@command
@dataclass(frozen=True)
class SetBoxed(Command):
    """Outline suprathreshold voxels."""

    name = "SET_BOXED"
    aspects = Aspect.THRESHOLD
    key: str
    on: bool


# ---------------------------------------------------------------------------
# instacorr
# ---------------------------------------------------------------------------


@command
@dataclass(frozen=True)
class SetSeed(Command):
    """Set the InstaCorr seed voxel."""

    name = "SET_SEED"
    aspects = Aspect.LAYERS | Aspect.SLICES
    major = True
    i: int
    j: int
    k: int


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------

OpenLayer = Callable[[str, str], Layer]


def install(
    bus: CommandBus,
    *,
    open_layer: OpenLayer | None = None,
    session: Any = None,
) -> CommandBus:
    """Register every handler on ``bus``.

    ``open_layer(path, key) -> Layer`` performs the actual load. It is injected
    rather than imported so the core stays testable without touching a disk, and
    so the UI can route loading through a worker thread without the vocabulary
    knowing that happened.

    ``session`` is optional and only the catalog and mode commands need it; the
    layer and navigation vocabulary works against bare state, which is what
    keeps most of the test suite free of a session.
    """

    def _load(path: str, key: str) -> Layer:
        if open_layer is None:
            raise RuntimeError("no loader installed: cannot open a dataset")
        return open_layer(path, key)

    def _adopt_grid_preserving_position(st: ViewerState, layer: Layer) -> Aspect:
        """Move the display grid to a new underlay, staying at the same place.

        Swapping the base image must not teleport the crosshair. The anatomical
        location is what the user is looking at, so it is carried across in
        millimetres and re-expressed in the new grid.
        """
        mm = st.crosshair_mm
        st.adopt_grid(layer.shape, layer.affine)
        if mm is not None and st.grid is not None:
            ijk = st.grid.mm_to_ijk(mm)
            st.crosshair = st.grid.clamp((round(ijk[0]), round(ijk[1]), round(ijk[2])))
        return Aspect.GRID | Aspect.CROSSHAIR

    @bus.handle(Read.name)
    def _read(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, Read)
        if session is None:
            raise RuntimeError("READ needs a session")
        session.read_directory(cmd.directory, recursive=bool(cmd.recursive))
        return Aspect.NOTHING

    @bus.handle(SetUnderlay.name)
    def _set_underlay(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetUnderlay)
        old = st.layers.base
        layer = _load(cmd.path, cmd.key or st.layers.mint_key("U"))
        st.layers.set_underlay(layer)
        if old is not None and session is not None:
            session.forget(old.key)
        # The underlay defines the display grid: it is the base image everything
        # else is resampled onto, so a new one re-establishes the space.
        return Aspect.LAYERS | Aspect.SLICES | _adopt_grid_preserving_position(st, layer)

    @bus.handle(SetOverlay.name)
    def _set_overlay(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetOverlay)
        old = st.layers.overlay
        layer = _load(cmd.path, cmd.key or st.layers.mint_key("O"))
        st.layers.set_overlay(layer)
        if old is not None and old.key != layer.key and session is not None:
            session.forget(old.key)
        if session is not None:
            session.apply_overlay_defaults(layer.key)
        dirty = SetOverlay.aspects
        if st.grid is None:
            dirty |= _adopt_grid_preserving_position(st, layer)
        return dirty

    @bus.handle(AddOverlay.name)
    def _add_overlay(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, AddOverlay)
        layer = _load(cmd.path, cmd.key or st.layers.mint_key("O"))
        st.layers.add_overlay(layer)
        if session is not None:
            session.apply_overlay_defaults(layer.key)
        dirty = AddOverlay.aspects
        if st.grid is None:
            dirty |= _adopt_grid_preserving_position(st, layer)
        return dirty

    @bus.handle(SetMode.name)
    def _set_mode(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetMode)
        if session is None:
            raise RuntimeError("SET_MODE needs a session")
        return session.set_mode(cmd.mode)

    @bus.handle(SetModeParam.name)
    def _set_mode_param(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetModeParam)
        if session is None:
            raise RuntimeError("SET_MODE_PARAM needs a session")
        return session.set_mode_param(cmd.param, cmd.value)

    @bus.handle(SetIJK.name)
    def _set_ijk(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetIJK)
        target = (cmd.i, cmd.j, cmd.k)
        if st.grid is not None:
            target = st.grid.clamp(target)
        if target == st.crosshair:
            return Aspect.NOTHING
        st.crosshair = target
        return SetIJK.aspects

    @bus.handle(SetXYZ.name)
    def _set_xyz(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetXYZ)
        if st.grid is None:
            return Aspect.NOTHING
        ijk = st.grid.mm_to_ijk((cmd.x, cmd.y, cmd.z))
        target = st.grid.clamp((round(ijk[0]), round(ijk[1]), round(ijk[2])))
        if target == st.crosshair:
            return Aspect.NOTHING
        st.crosshair = target
        return SetXYZ.aspects

    @bus.handle(SetIndex.name)
    def _set_index(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetIndex)
        hi = st.max_time_index()
        target = max(0, min(cmd.index, hi))
        if target == st.time_index:
            return Aspect.NOTHING
        st.time_index = target
        return SetIndex.aspects

    def _set_view(st: ViewerState, vid: str, aspects: Aspect, **changes: object) -> Aspect:
        """Apply changes to one viewport, reporting nothing when unchanged.

        Every viewport command routes through here so that "did this actually
        change" is answered once. A drag that re-sends the value it already has
        must not repaint every window.
        """
        current = st.viewports.get(vid)
        if all(getattr(current, k) == v for k, v in changes.items()):
            return Aspect.NOTHING
        st.viewports.update(vid, **changes)
        return aspects

    @bus.handle(SetTheme.name)
    def _set_theme(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetTheme)
        if cmd.theme not in ("dark", "light"):
            raise KeyError(f"unknown theme {cmd.theme!r}")
        if st.theme == cmd.theme:
            return Aspect.NOTHING
        st.theme = cmd.theme
        return SetTheme.aspects

    @bus.handle(OpenView.name)
    def _open_view(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, OpenView)
        if st.viewports.find(cmd.view) is not None:
            return Aspect.NOTHING  # replaying a script that already opened it
        st.viewports.open(ViewKind(cmd.kind), Plane(cmd.plane), vid=cmd.view)
        return OpenView.aspects

    @bus.handle(CloseView.name)
    def _close_view(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, CloseView)
        if st.viewports.find(cmd.view) is None:
            return Aspect.NOTHING
        st.viewports.close(cmd.view)
        return CloseView.aspects

    @bus.handle(SetViewPlane.name)
    def _set_view_plane(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewPlane)
        return _set_view(st, cmd.view, SetViewPlane.aspects, plane=Plane(cmd.plane))

    @bus.handle(SetViewSolo.name)
    def _set_view_solo(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewSolo)
        return _set_view(st, cmd.view, SetViewSolo.aspects, solo=bool(cmd.on))

    @bus.handle(SetViewLocked.name)
    def _set_view_locked(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewLocked)
        return _set_view(st, cmd.view, SetViewLocked.aspects, locked=bool(cmd.on))

    @bus.handle(SetViewPosition.name)
    def _set_view_position(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewPosition)
        pos = None if cmd.position is None else int(cmd.position)
        return _set_view(st, cmd.view, SetViewPosition.aspects, position=pos)

    @bus.handle(SetViewGrid.name)
    def _set_view_grid(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewGrid)
        return _set_view(st, cmd.view, SetViewGrid.aspects, grid_n=clamp_grid(cmd.n))

    @bus.handle(SetViewTraces.name)
    def _set_view_traces(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewTraces)
        keys = tuple(k for k in (cmd.keys or "").split(",") if k and k != "-")
        return _set_view(st, cmd.view, SetViewTraces.aspects, traces=keys)

    @bus.handle(SetViewSharedScale.name)
    def _set_view_shared(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewSharedScale)
        return _set_view(st, cmd.view, SetViewSharedScale.aspects, shared_scale=bool(cmd.on))

    @bus.handle(SetViewGeometry.name)
    def _set_view_geometry(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetViewGeometry)
        rect = (int(cmd.x), int(cmd.y), int(cmd.w), int(cmd.h))
        return _set_view(st, cmd.view, SetViewGeometry.aspects, geometry=rect)

    @bus.handle(SetZoom.name)
    def _set_zoom(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetZoom)
        return _set_view(st, cmd.view, SetZoom.aspects, zoom=max(0.05, float(cmd.zoom)))

    @bus.handle(SetPan.name)
    def _set_pan(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetPan)
        return _set_view(st, cmd.view, SetPan.aspects, pan=(float(cmd.x), float(cmd.y)))

    @bus.handle(SelectLayer.name)
    def _select_layer(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SelectLayer)
        if st.layers.find(cmd.key) is None:
            raise KeyError(f"no layer {cmd.key!r}")
        if st.selected == cmd.key:
            return Aspect.NOTHING
        st.selected = cmd.key
        # SLICES as well as LAYERS: a soloed window draws the selected layer,
        # so changing the selection changes what is on screen.
        return SelectLayer.aspects | Aspect.SLICES

    @bus.handle(AddLayer.name)
    def _add_layer(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, AddLayer)
        key = cmd.key or st.layers.mint_key()
        layer = _load(cmd.path, key)
        st.layers.add(layer)
        dirty = Aspect.LAYERS | Aspect.SLICES
        if st.grid is None:
            st.adopt_grid(layer.shape, layer.affine)
            dirty |= Aspect.GRID | Aspect.CROSSHAIR
        return dirty

    @bus.handle(RemoveLayer.name)
    def _remove_layer(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, RemoveLayer)
        st.layers.remove(cmd.key)
        return RemoveLayer.aspects

    @bus.handle(MoveLayer.name)
    def _move_layer(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, MoveLayer)
        before = st.layers.index_of(cmd.key)
        after = st.layers.move(cmd.key, cmd.to)
        return Aspect.NOTHING if before == after else MoveLayer.aspects

    @bus.handle(SetColormap.name)
    def _set_colormap(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetColormap)
        if st.layers.get(cmd.key).colormap == cmd.colormap:
            return Aspect.NOTHING
        st.layers.update(cmd.key, colormap=cmd.colormap)
        return SetColormap.aspects

    @bus.handle(SetThreshold.name)
    def _set_threshold(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetThreshold)
        value = float(cmd.value)
        if st.layers.get(cmd.key).threshold == value:
            return Aspect.NOTHING
        st.layers.update(cmd.key, threshold=value)
        return SetThreshold.aspects

    @bus.handle(SetLayerVisible.name)
    def _set_visible(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetLayerVisible)
        if st.layers.get(cmd.key).visible == bool(cmd.on):
            return Aspect.NOTHING
        st.layers.update(cmd.key, visible=bool(cmd.on))
        return SetLayerVisible.aspects

    @bus.handle(SetLayerOpacity.name)
    def _set_opacity(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetLayerOpacity)
        value = max(0.0, min(1.0, float(cmd.opacity)))
        if st.layers.get(cmd.key).opacity == value:
            return Aspect.NOTHING
        st.layers.update(cmd.key, opacity=value)
        return SetLayerOpacity.aspects

    @bus.handle(SetVolume.name)
    def _set_volume(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetVolume)
        layer = st.layers.get(cmd.key)
        value = max(0, min(int(cmd.index), layer.n_volumes - 1))
        if layer.volume_index == value:
            return Aspect.NOTHING
        st.layers.update(cmd.key, volume_index=value)
        return SetVolume.aspects

    @bus.handle(SetThresholdIndex.name)
    def _set_threshold_index(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetThresholdIndex)
        layer = st.layers.get(cmd.key)
        value = None if cmd.index is None else max(0, min(int(cmd.index), layer.n_volumes - 1))
        if layer.threshold_index == value:
            return Aspect.NOTHING
        st.layers.update(cmd.key, threshold_index=value)
        return SetThresholdIndex.aspects

    @bus.handle(SetPanes.name)
    def _set_panes(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetPanes)
        value = max(0, int(cmd.panes))
        if st.layers.get(cmd.key).n_panes == value:
            return Aspect.NOTHING
        st.layers.update(cmd.key, n_panes=value)
        return SetPanes.aspects

    @bus.handle(SetSign.name)
    def _set_sign(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetSign)
        mode = SignMode(cmd.mode)
        if st.layers.get(cmd.key).sign_mode == mode:
            return Aspect.NOTHING
        st.layers.update(cmd.key, sign_mode=mode)
        return SetSign.aspects

    @bus.handle(SetRange.name)
    def _set_range(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetRange)
        lo, hi = float(cmd.lo), float(cmd.hi)
        if hi < lo:
            lo, hi = hi, lo
        layer = st.layers.get(cmd.key)
        if (layer.range_lo, layer.range_hi) == (lo, hi):
            return Aspect.NOTHING
        st.layers.update(cmd.key, range_lo=lo, range_hi=hi)
        return SetRange.aspects

    @bus.handle(SetAlpha.name)
    def _set_alpha(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetAlpha)
        mode = AlphaMode(cmd.mode)
        if st.layers.get(cmd.key).alpha_mode == mode:
            return Aspect.NOTHING
        st.layers.update(cmd.key, alpha_mode=mode)
        return SetAlpha.aspects

    @bus.handle(SetBoxed.name)
    def _set_boxed(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetBoxed)
        if st.layers.get(cmd.key).boxed == bool(cmd.on):
            return Aspect.NOTHING
        st.layers.update(cmd.key, boxed=bool(cmd.on))
        return SetBoxed.aspects

    @bus.handle(SetTimeLinked.name)
    def _set_time_linked(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetTimeLinked)
        if st.layers.get(cmd.key).time_linked == bool(cmd.on):
            return Aspect.NOTHING
        st.layers.update(cmd.key, time_linked=bool(cmd.on))
        return SetTimeLinked.aspects

    @bus.handle(SetSeed.name)
    def _set_seed(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetSeed)
        target = (cmd.i, cmd.j, cmd.k)
        if st.grid is not None:
            target = st.grid.clamp(target)
        if st.seed == target:
            return Aspect.NOTHING
        st.seed = target
        return SetSeed.aspects

    return bus
