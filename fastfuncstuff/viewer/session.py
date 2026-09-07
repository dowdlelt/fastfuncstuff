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
from fastfuncstuff.viewer.state import ViewerState
from fastfuncstuff.viewer.vocab import AddLayer, SetVolume, install

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
        self._hidden_input: str | None = None
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
        if self._mode_dirty:
            self._hide_mode_input()

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
    def set_mode(self, name: str) -> Aspect:
        """Switch modes, tearing down the old one's overlay."""
        if self.mode.name == name:
            return Aspect.NOTHING
        self._restore_mode_input()
        self.mode.detach()
        self.mode = registry.get(name)()
        self.mode.attach(self)
        dirty = (Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH) | self.mode.refresh()
        self._hide_mode_input()
        return dirty

    def _hide_mode_input(self) -> None:
        """Hide the layer the active mode consumes, remembering its state."""
        key = self.mode.input_layer_key()
        if key is None:
            return
        layer = self.state.layers.find(key)
        if layer is None or not layer.visible:
            return
        self._hidden_input = key
        self.state.layers.update(key, visible=False)

    def _restore_mode_input(self) -> None:
        key = getattr(self, "_hidden_input", None)
        if key is None:
            return
        if self.state.layers.find(key) is not None:
            self.state.layers.update(key, visible=True)
        self._hidden_input = None

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
        if existing is None:
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
            # Pushed on top, never into the overlay slot: replacing there would
            # consume the mode's own input dataset, which is where the values
            # came from in the first place.
            self.state.layers.add_overlay(layer)
            if self.state.grid is None:
                self.state.adopt_grid(layer.shape, layer.affine)
        return key

    def remove_computed_overlay(self, source: str) -> None:
        existing = self.state.layers.find_by_source(source)
        if existing is None:
            return
        self.state.layers.remove(existing.key)
        self.forget(existing.key)

    def forget(self, key: str) -> None:
        """Drop a layer's cached and resident data."""
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
