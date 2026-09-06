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

from fastfuncstuff.viewer.commands import Aspect, Command, CommandBus, command
from fastfuncstuff.viewer.layers import AlphaMode, Layer, SignMode
from fastfuncstuff.viewer.state import ViewerState

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


@command
@dataclass(frozen=True)
class SetZoom(Command):
    name = "SET_ZOOM"
    aspects = Aspect.SLICES
    zoom: float


@command
@dataclass(frozen=True)
class SetPan(Command):
    name = "SET_PAN"
    aspects = Aspect.SLICES
    x: float
    y: float


@command
@dataclass(frozen=True)
class SetLock(Command):
    """Toggle one pane-synchronisation lock."""

    name = "SET_LOCK"
    aspects = Aspect.NOTHING
    which: str
    on: bool


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


def install(bus: CommandBus, *, open_layer: OpenLayer | None = None) -> CommandBus:
    """Register every handler on ``bus``.

    ``open_layer(path, key) -> Layer`` performs the actual load. It is injected
    rather than imported so the core stays testable without touching a disk, and
    so the UI can route loading through a worker thread without the vocabulary
    knowing that happened.
    """

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

    @bus.handle(SetZoom.name)
    def _set_zoom(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetZoom)
        z = max(0.05, float(cmd.zoom))
        if z == st.zoom:
            return Aspect.NOTHING
        st.zoom = z
        return SetZoom.aspects

    @bus.handle(SetPan.name)
    def _set_pan(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetPan)
        target = (float(cmd.x), float(cmd.y))
        if target == st.pan:
            return Aspect.NOTHING
        st.pan = target
        return SetPan.aspects

    @bus.handle(SetLock.name)
    def _set_lock(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, SetLock)
        if not hasattr(st.locks, cmd.which):
            raise KeyError(f"unknown lock {cmd.which!r}")
        setattr(st.locks, cmd.which, bool(cmd.on))
        return Aspect.NOTHING

    @bus.handle(AddLayer.name)
    def _add_layer(cmd: Command, st: ViewerState) -> Aspect:
        assert isinstance(cmd, AddLayer)
        if open_layer is None:
            raise RuntimeError("no loader installed: ADD_LAYER cannot run")
        key = cmd.key or st.layers.mint_key()
        layer = open_layer(cmd.path, key)
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
