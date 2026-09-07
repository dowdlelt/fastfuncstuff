"""Tiered residency: disk to CPU RAM to compute device.

CPU RAM is the residency tier and device memory is the working set. Several
datasets live in RAM at once; only whatever InstaCorr or InstaGLM is currently
touching gets promoted. That split is what lets a 24 GB card hold a working set
while system RAM holds the session.

The measurements this is built around, on an M4 Max:

* header-only peek -- microseconds, so a directory can be indexed eagerly
* first volume out of a 66 MB ``.nii.gz`` -- 2.45 ms, because volume 0 sits at
  the head of the stream
* full inflate -- 335 MB/s single-threaded, so a 2 GB dataset takes ~6 s
* RAM to device -- 10 GB/s

That last figure is the one that surprises people on Apple silicon: unified
memory removes the *capacity* split between host and device, not the copy. A
promotion still costs real bandwidth, so eviction policy matters on MPS exactly
as it does on CUDA.

The ~6 s inflate is also why loading is split in two. Volume 0 goes on screen
almost immediately and the rest arrives on a worker thread; ``zlib`` releases
the GIL while inflating, so that thread genuinely does not block the UI.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.dsetinfo import DatasetInfo, read_info, read_volume
from fastfuncstuff.memory import get_available_memory


class Tier(IntEnum):
    """How far along the disk-to-device path a dataset currently is.

    Ordered so that ``>=`` is a meaningful "at least this resident" test.
    """

    ABSENT = 0
    HEADER = 1  # DatasetInfo only
    PREVIEW = 2  # first volume in RAM
    RAM = 3  # whole 4-D array in RAM
    DEVICE = 4  # promoted to the compute device


@dataclass
class Resident:
    """One dataset's residency record."""

    key: str
    path: Path
    info: DatasetInfo
    preview: np.ndarray | None = None
    array: np.ndarray | None = None
    tensor: torch.Tensor | None = None
    #: Monotonic counter, not a clock: promotion order is all the LRU needs and
    #: a counter cannot go backwards when the system clock does.
    used_at: int = 0
    error: BaseException | None = None
    _future: Future[np.ndarray] | None = field(default=None, repr=False)

    @property
    def tier(self) -> Tier:
        if self.tensor is not None:
            return Tier.DEVICE
        if self.array is not None:
            return Tier.RAM
        if self.preview is not None:
            return Tier.PREVIEW
        return Tier.HEADER

    @property
    def pending(self) -> bool:
        """Whether a background inflate is actually in flight.

        Not the same as "no array yet": a 3-D dataset is complete the moment its
        single volume is previewed and never schedules a load, so testing for a
        missing array would leave it reading as loading forever.
        """
        return self._future is not None

    @property
    def ram_bytes(self) -> int:
        return 0 if self.array is None else int(self.array.nbytes)

    @property
    def device_bytes(self) -> int:
        if self.tensor is None:
            return 0
        return int(self.tensor.numel() * self.tensor.element_size())

    @property
    def full_bytes(self) -> int:
        """What a full float32 residency would cost, from the header alone."""
        nx, ny, nz, nv = self.info.shape
        return int(nx) * int(ny) * int(nz) * max(int(nv), 1) * 4


class VolumeStore:
    """Holds datasets across the residency tiers, evicting by least-recent use.

    Thread-safe: the UI thread reads previews and dispatches promotions while a
    worker inflates. Only the bookkeeping is locked -- the inflate itself runs
    outside the lock so a slow load never blocks a crosshair move.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        ram_budget: int | None = None,
        device_budget: int | None = None,
        max_workers: int = 2,
        zstd_threads: int | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self._ram_budget = ram_budget
        self._device_budget = device_budget
        self._zstd_threads = zstd_threads
        self._items: dict[str, Resident] = {}
        self._lock = threading.RLock()
        self._clock = 0
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ffs-viewer-load"
        )

    # -- budgets -------------------------------------------------------
    #
    # Both go through memory.py rather than being hardcoded, so the 0.5 GPU
    # safety factor that compensates for the caching allocator applies here too.

    @property
    def ram_budget(self) -> int:
        if self._ram_budget is not None:
            return self._ram_budget
        return get_available_memory(torch.device("cpu"))

    @property
    def device_budget(self) -> int:
        if self._device_budget is not None:
            return self._device_budget
        if self.device.type == "cpu":
            return self.ram_budget
        return get_available_memory(self.device, empty_cache=False)

    def ram_used(self) -> int:
        with self._lock:
            return sum(r.ram_bytes for r in self._items.values())

    def device_used(self) -> int:
        with self._lock:
            return sum(r.device_bytes for r in self._items.values())

    # -- opening -------------------------------------------------------
    def open(self, path: str | Path, key: str | None = None) -> Resident:
        """Read the header and register the dataset. Cheap enough to do eagerly."""
        p = Path(path)
        k = key or str(p)
        with self._lock:
            existing = self._items.get(k)
            if existing is not None:
                return existing
        info = read_info(p)
        if not info.exists:
            raise FileNotFoundError(f"no such dataset: {p}")
        res = Resident(key=k, path=p, info=info)
        with self._lock:
            self._items[k] = res
        return res

    def adopt(self, key: str, array: np.ndarray, *, name: str = "") -> Resident:
        """Register an in-memory volume as if it had been loaded.

        Modes compute overlays rather than reading them, but everything
        downstream -- slicing, colour mapping, the value readout -- addresses
        data through the store. Adopting keeps that one path instead of
        teaching each consumer about a second kind of layer.
        """
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., None]
        nx, ny, nz, nv = arr.shape
        info = DatasetInfo(
            path=Path(name or key),
            iname=name or key,
            exists=True,
            storage="MEMORY",
            shape=(int(nx), int(ny), int(nz), int(nv)),
        )
        res = Resident(key=key, path=Path(name or key), info=info, array=arr)
        with self._lock:
            self._items[key] = res
            self._touch(res)
        return res

    def get(self, key: str) -> Resident:
        with self._lock:
            try:
                return self._items[key]
            except KeyError:
                raise KeyError(f"dataset {key!r} is not open") from None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def close(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    # -- preview -------------------------------------------------------
    def preview(self, key: str, index: int = 0) -> np.ndarray:
        """One volume, read only as far into the file as it sits.

        This is the "something on screen now" path. It deliberately does not
        promote the dataset or count against a budget: it is one volume, and
        making it evict a resident working set would be a bad trade.
        """
        res = self.get(key)
        if index == 0 and res.preview is not None:
            return res.preview
        vol, _ = read_volume(res.path, index)
        if index == 0:
            with self._lock:
                res.preview = vol
        return vol

    # -- full residency ------------------------------------------------
    def load_async(
        self, key: str, *, on_done: Callable[[str], None] | None = None
    ) -> Future[np.ndarray]:
        """Start (or join) a background inflate to RAM.

        Repeated calls share one future, so a UI that asks on every repaint does
        not start a second inflate of the same file.
        """
        res = self.get(key)
        with self._lock:
            if res.array is not None:
                fut: Future[np.ndarray] = Future()
                fut.set_result(res.array)
                return fut
            if res._future is not None:
                return res._future
            fut = self._pool.submit(self._inflate, key)
            res._future = fut

        if on_done is not None:
            fut.add_done_callback(lambda _f, k=key: on_done(k))
        return fut

    def _inflate(self, key: str) -> np.ndarray:
        res = self.get(key)
        try:
            from fastfuncstuff.io.afni import load_nifti

            img = load_nifti(res.path, zstd_threads=self._zstd_threads)
            arr = np.asanyarray(img.dataobj, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., None]
            # NIfTI arrives Fortran-ordered. Normalizing to C order once here
            # costs one copy inside a load the user is already waiting on, and
            # makes every later (voxels, time) reshape a free view. Left as-is,
            # that copy was being paid again on every mode preparation --
            # measured at 1.9 GB of copying per InstaCorr parameter change on a
            # 1.4 GB dataset, for data that was already resident.
            arr = np.ascontiguousarray(arr)
        except BaseException as exc:  # surfaced via Resident.error, not swallowed
            with self._lock:
                res.error = exc
                res._future = None
            raise
        with self._lock:
            res.array = arr
            res._future = None
            self._touch(res)
        self._enforce_ram_budget(protect=key)
        return arr

    def ensure_ram(self, key: str) -> np.ndarray:
        """Block until the dataset is fully resident in RAM."""
        res = self.get(key)
        if res.array is not None:
            with self._lock:
                self._touch(res)
            return res.array
        return self.load_async(key).result()

    def promote(self, key: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Move a dataset onto the compute device, evicting others if needed.

        Returns a ``(nx, ny, nz, nv)`` tensor. Promotion is a real copy even on
        unified memory, so callers should hold the result rather than re-promote
        per frame.
        """
        res = self.get(key)
        if res.tensor is not None and res.tensor.dtype == dtype:
            with self._lock:
                self._touch(res)
            return res.tensor

        arr = self.ensure_ram(key)
        need = arr.size * torch.empty((), dtype=dtype).element_size()
        self._make_device_room(need, protect=key)
        tensor = torch.from_numpy(arr).to(device=self.device, dtype=dtype)
        with self._lock:
            res.tensor = tensor
            self._touch(res)
        return tensor

    def demote(self, key: str) -> None:
        """Drop the device copy, keeping RAM residency."""
        with self._lock:
            res = self._items.get(key)
            if res is None:
                return
            res.tensor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def release(self, key: str) -> None:
        """Drop both device and RAM copies, keeping the header and preview."""
        self.demote(key)
        with self._lock:
            res = self._items.get(key)
            if res is not None:
                res.array = None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- eviction ------------------------------------------------------
    def _touch(self, res: Resident) -> None:
        self._clock += 1
        res.used_at = self._clock

    def _lru_order(self, predicate: Callable[[Resident], bool]) -> list[Resident]:
        with self._lock:
            return sorted(
                (r for r in self._items.values() if predicate(r)),
                key=lambda r: r.used_at,
            )

    def _make_device_room(self, need: int, *, protect: str) -> None:
        budget = self.device_budget
        for res in self._lru_order(lambda r: r.tensor is not None):
            if self.device_used() + need <= budget:
                return
            if res.key == protect:
                continue
            self.demote(res.key)

    def _enforce_ram_budget(self, *, protect: str) -> None:
        budget = self.ram_budget
        for res in self._lru_order(lambda r: r.array is not None):
            if self.ram_used() <= budget:
                return
            if res.key == protect or res.tensor is not None:
                continue  # never evict what the device is currently working on
            self.release(res.key)
