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
from fastfuncstuff.viewer.commands import Aspect, Command, CommandBus
from fastfuncstuff.viewer.layers import Layer
from fastfuncstuff.viewer.residency import Resident, VolumeStore
from fastfuncstuff.viewer.state import ViewerState
from fastfuncstuff.viewer.vocab import AddLayer, SetVolume, install

#: Percentiles used to auto-range a layer. AFNI's autorange takes the maximum,
#: which one bright voxel is enough to ruin; percentiles are what make a map
#: readable without anyone reaching for a slider.
AUTORANGE_PERCENTILES = (2.0, 98.0)


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
        self.bus = install(CommandBus(self.state, record=record), open_layer=self._open)
        self._volume_cache: dict[tuple[str, int], torch.Tensor] = {}
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

    def do(self, cmd: Command) -> Aspect:
        return self.bus.dispatch(cmd)

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
