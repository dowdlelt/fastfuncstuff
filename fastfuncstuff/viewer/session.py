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
from fastfuncstuff.viewer.vocab import AddLayer, install

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
