"""A viewer session: state, residency, and the command bus wired together.

This is what a UI, a CLI or a test drives. Nothing above this layer touches
:class:`ViewerState` directly -- everything goes through
:meth:`ViewerSession.do`, which is what keeps the recording honest.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.dsetinfo import DatasetInfo
from fastfuncstuff.viewer import catalog as catalog_mod
from fastfuncstuff.viewer.catalog import CatalogEntry
from fastfuncstuff.viewer.commands import Aspect, Command, CommandBus
from fastfuncstuff.viewer.layers import AlphaMode, Layer
from fastfuncstuff.viewer.modes import Mode, registry
from fastfuncstuff.viewer.modes.base import ComputedOverlay, Trace
from fastfuncstuff.viewer.residency import Resident, VolumeStore
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import AddLayer, CloseView, OpenView, SetVolume, install

#: Percentiles used to auto-range a layer. AFNI's autorange takes the maximum,
#: which one bright voxel is enough to ruin; percentiles are what make a map
#: readable without anyone reaching for a slider.
AUTORANGE_PERCENTILES = (2.0, 98.0)

#: Where a freshly-picked overlay starts its threshold. High enough that the
#: underlay reads through the noise -- a quarter of the brain tinted is not a
#: view of anything -- and low enough that real structure is already on screen
#: before anyone touches the slider.
OVERLAY_START_PERCENTILE = 90.0


def derive_range(
    values: np.ndarray, percentiles: tuple[float, float] = AUTORANGE_PERCENTILES
) -> tuple[float, float]:
    """Display range from a volume, ignoring non-finite voxels.

    Falls back to the finite min/max when the percentile span collapses, which
    happens on masks and other near-constant volumes.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, percentiles))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def infer_time_linked(info: DatasetInfo) -> bool:
    """Whether the global time index should drive this dataset.

    A time series and a stats dataset are both 4-D on disk, and NIfTI cannot
    reliably tell them apart: ``pixdim[4]`` defaults to 1.0, so almost every
    file claims a TR. Sub-brick labels are the usable signal — 3dDeconvolve and
    ffs write them on stats output, raw time series carry none.

    So: 4-D defaults to time-linked because that is the common case, and labels
    turn it off. Both are guesses about a file that does not say, which is why
    SET_TIME_LINKED exists to correct it.
    """
    if info.n_volumes <= 1:
        return False
    return not info.labels


def layer_from_info(info: DatasetInfo, key: str, path: Path) -> Layer:
    """Build a display layer from a header-only read."""
    nx, ny, nz, nv = info.shape
    return Layer(
        key=key,
        name=path.name,
        path=str(path),
        shape=(int(nx), int(ny), int(nz)),
        n_volumes=max(int(nv), 1),
        affine=np.asarray(info.affine, dtype=float),
        labels=tuple(info.labels),
        stataux=dict(info.stataux),
        time_linked=infer_time_linked(info),
    )


class ViewerSession:
    """Owns the state, the residency store and the bus."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        record: bool = True,
        store: VolumeStore | None = None,
    ) -> None:
        self.state = ViewerState()
        self.store = store or VolumeStore(device=device)
        self.bus = install(
            CommandBus(self.state, record=record), open_layer=self._open, session=self
        )
        self.catalog: list[CatalogEntry] = []
        self.catalog_dir: Path | None = None
        self.mode: Mode = registry.get("plain")()
        self.mode.attach(self)
        self._volume_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._mode_dirty: Aspect = Aspect.NOTHING
        self._displaced_overlay: Layer | None = None
        #: Set once by a UI that runs mode preparation on a worker. Applied to
        #: every mode as it is attached -- setting it on the mode afterwards
        #: would be too late, since set_mode refreshes on the way in and that
        #: first refresh is exactly the one that would freeze the window.
        self.defer_mode_preparation = False
        # A time-index or sub-brick change invalidates the cached device volume;
        # doing it here rather than in each handler means a command added later
        # cannot forget to.
        self.bus.subscribe(self._on_command)

    # -- loading -------------------------------------------------------
    def _open(self, path: str, key: str) -> Layer:
        """Header peek plus first volume; the full inflate starts in background.

        Returning after only the preview is deliberate. Volume 0 costs about
        2.5 ms while a full dataset takes seconds, so the layer becomes visible
        immediately and becomes scrubbable when the worker finishes.
        """
        res = self.store.open(path, key=key)
        layer = layer_from_info(res.info, key, res.path)
        preview = self.store.preview(key)
        lo, hi = derive_range(preview)
        layer = layer.with_(range_lo=lo, range_hi=hi)
        if layer.n_volumes > 1:
            self.store.load_async(key, on_done=self._on_loaded)
        return layer

    def _on_loaded(self, key: str) -> None:
        self._notify_loaded(key)

    #: Replaced by the UI to marshal completion back onto its own thread. The
    #: worker calls this from a load thread, so the default must stay trivial.
    _notify_loaded: Callable[[str], None] = staticmethod(lambda key: None)

    def on_loaded(self, callback: Callable[[str], None]) -> None:
        """Install the completion hook for background loads."""
        self._notify_loaded = callback  # type: ignore[assignment]

    def load(self, path: str | Path, *, key: str | None = None) -> str:
        """Add a dataset as a new top layer; returns its key."""
        chosen = key or self.state.layers.mint_key()
        self.do(AddLayer(str(path), chosen))
        return chosen

    # -- dispatch ------------------------------------------------------
    def _on_command(self, cmd: Command, dirty: Aspect) -> None:
        if dirty & (Aspect.TIME | Aspect.LAYERS):
            self.invalidate()
        elif isinstance(cmd, SetVolume):
            self.invalidate(cmd.key)
        # The mode reacts after the state has settled, so an InstaCorr seed sees
        # the seed already moved. Its own dirty aspects are folded into what the
        # dispatch reports, which is how a recomputed overlay reaches the panes.
        self._mode_dirty = self.mode.on_command(cmd, dirty)

    def do(self, cmd: Command) -> Aspect:
        self._mode_dirty = Aspect.NOTHING
        dirty = self.bus.dispatch(cmd)
        return dirty | self._mode_dirty

    # -- catalog -------------------------------------------------------
    def read_directory(self, directory: str | Path, *, recursive: bool = False) -> list:
        """Populate the catalog the pickers draw from."""
        self.catalog = catalog_mod.scan(directory, recursive=recursive)
        self.catalog_dir = Path(directory)
        return self.catalog

    def apply_overlay_defaults(self, key: str) -> None:
        """Give a freshly-picked overlay a state you can actually see through.

        A layer loaded with threshold 0 and no alpha is opaque everywhere, so
        dropping one on an anatomical hides it completely -- which defeats the
        first thing anyone does, checking that the two line up. Starting at a
        high percentile means the overlay reads as structure over anatomy from
        the moment it lands, and the slider takes it from there.
        """
        layer = self.state.layers.find(key)
        if layer is None or layer.is_computed:
            return
        try:
            values = self.volume(key, 0)
        except (KeyError, FileNotFoundError, ValueError):
            return
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        signed = bool((finite < 0).any() and (finite > 0).any())
        threshold = float(np.percentile(np.abs(finite), OVERLAY_START_PERCENTILE))
        self.state.layers.update(
            key,
            colormap="redblue" if signed else "hot",
            threshold=threshold,
            alpha_mode=AlphaMode.LINEAR,
        )

    def suggested_underlay(self) -> CatalogEntry | None:
        return catalog_mod.suggest_underlay(self.catalog)

    # -- modes ---------------------------------------------------------
    # -- viewports -----------------------------------------------------
    def open_view(self, kind: ViewKind, plane: Plane) -> str:
        """Open a window and return its id.

        The id is minted here and passed into the command rather than being
        returned by it, so the recorded line names the window it opened and
        every later line that addresses that window still resolves on replay.
        """
        vid = self.state.viewports.mint_id(kind)
        self.do(OpenView(vid, str(kind), str(plane)))
        return vid

    def default_layout(self) -> Aspect:
        """Open what a viewer with nothing configured should show.

        Three images, no graph. Goal zero is an underlay and an overlay
        together; a graph is something you ask for.

        Dispatched rather than built directly, so the recording starts with the
        windows it started with -- a script that replays into a viewer with no
        windows in it is a script that does not restore the session.
        """
        if len(self.state.viewports):
            return Aspect.NOTHING
        dirty = Aspect.NOTHING
        for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
            dirty |= self.do(OpenView(self.state.viewports.mint_id(ViewKind.IMAGE), "image", plane))
        return dirty

    def close_view(self, vid: str) -> Aspect:
        return self.do(CloseView(vid))

    def graph_layers(self) -> list[Layer]:
        """Layers a graph can plot: the time-linked ones, bottom-up.

        A 3-D anatomy is never offered. It has no time course, and listing it
        with an empty checkbox invites the reading that the trace is hidden
        rather than that it does not exist.
        """
        return [ly for ly in self.state.layers if ly.time_linked and ly.n_volumes > 1]

    def traces_for(self, viewport) -> list[Layer]:
        """The layers one graph viewport should plot.

        An empty selection means all of them: a new layer starts plotted, which
        is what someone who just loaded it expects. Keys that no longer name a
        layer are dropped rather than erroring, because a viewport outlives the
        layers it was pointed at.
        """
        available = self.graph_layers()
        if not viewport.traces:
            return available
        wanted = set(viewport.traces)
        return [ly for ly in available if ly.key in wanted]

    def carpet_source(self, viewport) -> Layer | None:
        """The one layer a carpet window draws.

        Reuses a graph's trace selection rather than inventing a second way to
        say "this layer": a carpet is a graph of every voxel, and the picker
        that chooses what a graph plots is the same question.
        """
        available = self.graph_layers()
        if not available:
            return None
        for key in viewport.traces:
            found = next((ly for ly in available if ly.key == key), None)
            if found is not None:
                return found
        return available[-1]

    def carpet_overlay(self, source: Layer) -> Layer | None:
        """Which layer labels a carpet's rows.

        The topmost visible layer that is not the run being drawn -- not
        "overlay-prime". In a stack of anat, run and stats the thing worth
        drawing beside the rows is the stats map on top, and overlay-prime is
        the run itself.
        """
        for layer in reversed(list(self.state.layers)):
            if layer.key != source.key and layer.visible:
                return layer
        return None

    def build_carpet(self, viewport, *, progress=None):
        """Render one carpet window's picture. Slow; runs on the worker.

        Reads arrays and returns one, like a mode's prepare() -- no session
        state is touched, because the thread that paints owns all of that.
        """
        from fastfuncstuff.viewer import carpet as carpet_mod

        layer = self.carpet_source(viewport)
        if layer is None:
            raise ValueError("no time series loaded to draw a carpet of")
        data = self.store.ensure_ram(layer.key)

        overlay = self.carpet_overlay(layer)
        order_volume = None
        if viewport.order in ("overlay", "roi"):
            if overlay is None or overlay.key == layer.key:
                raise ValueError(f"ordering by {viewport.order!r} needs an overlay above the run")
            order_volume = self._aligned_volume(overlay, layer)

        seed = None
        if viewport.order == "seed":
            ijk = self.state.seed or self.state.crosshair
            seed = self.timeseries(layer.key, ijk)

        sidebar = None
        if overlay is not None and overlay.key != layer.key:
            sidebar = self._aligned_volume(overlay, layer)

        return layer, carpet_mod.build_carpet(
            data,
            order=viewport.order,
            seed_series=seed,
            order_volume=order_volume,
            sidebar_volume=sidebar,
            polort=int(viewport.detrend),
            normalize=viewport.scaling,
            device=self.store.device,
            progress=progress,
        )

    def _aligned_volume(self, layer: Layer, like: Layer) -> np.ndarray | None:
        """One layer's values on another's voxel grid, or None if they differ.

        A carpet's rows are the *series*' voxels, so an overlay can only label
        them if it sits on the same grid. Resampling it here would work, but a
        silent resample is how a stat map ends up labelling the wrong rows --
        better to say the overlay does not apply.
        """
        if layer.shape != like.shape or not np.allclose(layer.affine, like.affine, atol=1e-4):
            return None
        volume = self.volume(layer.key)
        return np.asarray(volume, dtype=np.float32)

    def set_mode(self, name: str) -> Aspect:
        """Switch modes, tearing down the old one's overlay."""
        if self.mode.name == name:
            return Aspect.NOTHING
        self.mode.detach()
        self.mode = registry.get(name)()
        self.mode.defer_preparation = self.defer_mode_preparation
        self.mode.attach(self)
        return (Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH) | self.mode.refresh()

    def refresh_mode(self) -> Aspect:
        """Recompute and install the active mode's overlay."""
        return self.mode.refresh()

    def set_mode_param(self, param: str, value: str) -> Aspect:
        """Coerce a text parameter to its control's type and apply it."""
        spec = next((c for c in self.mode.controls() if c.name == param), None)
        if spec is None:
            raise KeyError(f"mode {self.mode.name!r} has no parameter {param!r}")
        coerced: object = value
        default = getattr(spec, "default", None)
        if isinstance(default, bool):
            coerced = value not in ("0", "false", "False", "")
        elif isinstance(default, int):
            coerced = int(float(value))
        elif isinstance(default, float):
            coerced = float(value)
        return self.mode.set_param(param, coerced)

    def mode_series(self, ijk: tuple[int, int, int] | None = None) -> list[Trace]:
        return self.mode.series(ijk or self.state.crosshair)

    # -- derived layers -------------------------------------------------
    #
    # Same split as a mode's prepare()/compute(): the arithmetic is seconds
    # over a whole 4-D array and runs on a worker, so it must not touch session
    # state; installing the result is instant and happens on the GUI thread.

    def compute_denoise(
        self,
        key: str,
        *,
        matrix: str = "",
        polort: int = -1,
        keep_mean: bool = True,
        progress=None,
    ) -> tuple[np.ndarray, str]:
        """The slow half: residualise a run, and say what was projected out.

        Touches nothing but the arrays it reads and the one it builds, so it is
        safe on a worker thread.
        """
        from fastfuncstuff.viewer import derive

        layer = self.state.layers.get(key)
        if layer.n_volumes <= 1:
            raise ValueError(f"{layer.name} is not a time series; nothing to denoise")
        nuisance = derive.read_nuisance(matrix or None, n_time=layer.n_volumes, polort=polort)
        data = self.store.ensure_ram(key)
        values = derive.denoise(
            data,
            nuisance,
            device=self.store.device,
            keep_mean=keep_mean,
            progress=progress,
        )
        return values, nuisance.description

    def denoise(
        self,
        key: str,
        *,
        matrix: str = "",
        polort: int = -1,
        keep_mean: bool = True,
        progress=None,
    ) -> Aspect:
        """Both halves, in order. What the DENOISE command runs on replay."""
        values, detail = self.compute_denoise(
            key, matrix=matrix, polort=polort, keep_mean=keep_mean, progress=progress
        )
        return self.install_derived(key, values, op="denoise", detail=detail)

    def install_derived(
        self, source_key: str, values: np.ndarray, *, op: str, detail: str = ""
    ) -> Aspect:
        """Put a computed dataset into the stack, right above what made it.

        Two decisions, both about making the comparison the easy one:

        * It lands **immediately above its source** and inherits how the source
          is drawn. Neighbours in the stack are what `[`, `]` and a soloed
          window flip between, so raw against denoised is one keypress -- the
          same gesture that checks an EPI against an anat.
        * Re-deriving from the same source with the same operation **replaces**
          the layer rather than pushing another. Clicking twice must not grow
          the stack without bound, and the parameters that produced it are in
          the recorded command either way.
        """
        source = self.state.layers.get(source_key)
        tag = f"derived:{op}:{source_key}"
        existing = self.state.layers.find_by_source(tag)
        key = existing.key if existing is not None else self.state.layers.mint_key("D")
        name = f"{source.name} ·{op}d"

        self.store.adopt(key, values, name=name)
        self.invalidate(key)
        if existing is not None:
            self.state.layers.update(key, name=name, path=f"<{op}: {detail}>")
            return Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH

        self.state.layers.add(
            Layer(
                key=key,
                name=name,
                # No file backs it, so the field that would hold one carries
                # the provenance instead: what was projected out, in words.
                path=f"<{op}: {detail}>",
                shape=source.shape,
                n_volumes=int(values.shape[3]) if values.ndim == 4 else 1,
                affine=source.affine,
                labels=source.labels,
                visible=source.visible,
                opacity=source.opacity,
                colormap=source.colormap,
                # The same display range as its source, so the two are
                # comparable at a glance rather than each auto-scaled to
                # itself -- which would hide exactly the difference you made
                # the layer to see.
                range_lo=source.range_lo,
                range_hi=source.range_hi,
                time_linked=source.time_linked,
                source=tag,
            ),
            at=self.state.layers.index_of(source_key) + 1,
        )
        self.state.selected = key
        return Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH

    # -- computed overlays ---------------------------------------------
    def install_computed_overlay(self, source: str, overlay: ComputedOverlay) -> str:
        """Install (or update in place) the layer a mode owns.

        Updating in place matters: a mode recomputes on every seed click, and
        pushing a new layer each time would grow the stack without bound and
        reset the threshold the user just set.
        """
        existing = self.state.layers.find_by_source(source)
        key = existing.key if existing is not None else self.state.layers.mint_key("M")
        self.store.adopt(key, overlay.values, name=overlay.name)
        self.invalidate(key)

        lo, hi = overlay.display_range or derive_range(overlay.values)
        if existing is not None:
            # The name is identity -- showing "IC 0" while displaying IC 4 is a
            # lie. Range and threshold deliberately do NOT follow: stepping
            # through components at a threshold you set is how they get
            # reviewed, and resetting it on every step would undo the gesture.
            if existing.name != overlay.name:
                self.state.layers.update(key, name=overlay.name)
        else:
            layer = Layer(
                key=key,
                name=overlay.name,
                path=f"<{overlay.name}>",
                shape=tuple(int(v) for v in overlay.values.shape[:3]),
                n_volumes=1,
                affine=np.asarray(overlay.affine, dtype=float),
                colormap=overlay.colormap,
                range_lo=lo,
                range_hi=hi,
                threshold=overlay.threshold or 0.0,
                source=source,
            )
            # The computed map takes the primary overlay slot. That is the one
            # thing it may displace: the underlay is the base image everything
            # is drawn on and must survive a mode switch. The displaced layer
            # is remembered, not destroyed, and comes back when the mode is
            # left -- and the mode itself holds its source data by reference,
            # so being displaced here cannot strand it.
            self._displaced_overlay = self.state.layers.overlay
            self.state.layers.set_overlay(layer)
            if self.state.grid is None:
                self.state.adopt_grid(layer.shape, layer.affine)
        return key

    def remove_computed_overlay(self, source: str) -> None:
        """Drop a mode's overlay and put back whatever it displaced."""
        existing = self.state.layers.find_by_source(source)
        if existing is None:
            return
        self.state.layers.remove(existing.key)
        self.forget(existing.key)
        displaced, self._displaced_overlay = self._displaced_overlay, None
        if displaced is not None and self.state.layers.find(displaced.key) is None:
            self.state.layers.set_overlay(displaced)

    def forget(self, key: str) -> None:
        """Drop a layer's cached and resident data.

        Skips anything the current mode is using or has displaced: dropping a
        mode's source mid-session is what turns a re-prepare into a silent
        stale map.
        """
        held = {self.mode.input_layer_key()}
        if self._displaced_overlay is not None:
            held.add(self._displaced_overlay.key)
        if key in held:
            return
        self.invalidate(key)
        self.store.close(key)

    def run_script(self, text: str) -> Aspect:
        return self.bus.run_script(text)

    def to_script(self, *, header: str | None = None) -> str:
        return self.bus.to_script(header=header)

    def save_script(self, path: str | Path, *, header: str | None = None) -> Path:
        out = Path(path)
        out.write_text(self.to_script(header=header))
        return out

    # -- data access ---------------------------------------------------
    def resident(self, key: str) -> Resident:
        return self.store.get(key)

    def volume(self, key: str, index: int | None = None) -> np.ndarray:
        """One 3-D volume for display, from RAM when resident, disk when not."""
        res = self.store.get(key)
        idx = self.state.time_index if index is None else index
        idx = max(0, min(idx, res.info.n_volumes - 1))
        if res.array is not None:
            return res.array[..., idx]
        return self.store.preview(key, idx)

    def display_volume(self, key: str, index: int | None = None) -> torch.Tensor | None:
        """The currently displayed sub-brick as a device tensor, cached.

        Cached per ``(layer, sub-brick)`` because a redraw asks for the same
        volume three times -- once per plane -- and the host-to-device copy
        measures 10 GB/s even on unified memory. Returns ``None`` when the
        layer's data has gone, so a repaint mid-eviction draws nothing rather
        than raising into the paint handler.
        """
        try:
            res = self.store.get(key)
        except KeyError:
            return None
        layer = self.state.layers.find(key)
        if layer is None:
            return None

        if index is not None:
            idx = index
        elif layer.time_linked:
            idx = self.state.time_index
        else:
            idx = layer.volume_index
        idx = max(0, min(int(idx), res.info.n_volumes - 1))

        cache_key = (key, idx)
        hit = self._volume_cache.get(cache_key)
        if hit is not None:
            return hit

        try:
            arr = self.volume(key, idx)
        except (KeyError, FileNotFoundError, ValueError):
            return None
        tensor = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32).to(
            self.store.device
        )
        # One sub-brick per layer is enough: panes share it, and holding more
        # would quietly duplicate what the residency store already owns.
        for stale in [k for k in self._volume_cache if k[0] == key]:
            del self._volume_cache[stale]
        self._volume_cache[cache_key] = tensor
        return tensor

    def invalidate(self, key: str | None = None) -> None:
        """Drop cached device volumes for one layer, or all of them."""
        if key is None:
            self._volume_cache.clear()
            return
        for stale in [k for k in self._volume_cache if k[0] == key]:
            del self._volume_cache[stale]

    def timeseries(self, key: str, ijk: tuple[int, int, int] | None = None) -> np.ndarray:
        """The time course at a voxel, or an empty array if not yet resident.

        Returns empty rather than blocking: the graph pane asks on every
        crosshair move, and waiting seconds for an inflate would be exactly the
        lock-up this design exists to avoid.
        """
        res = self.store.get(key)
        if res.array is None:
            return np.empty(0, dtype=np.float32)
        i, j, k = ijk if ijk is not None else self.state.crosshair
        nx, ny, nz = res.array.shape[:3]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return np.empty(0, dtype=np.float32)
        return np.asarray(res.array[i, j, k, :], dtype=np.float32)

    def close(self) -> None:
        self.store.shutdown()
