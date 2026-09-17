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

**Eviction needs something to evict back to.** A dataset read off disk can be
dropped for free, because the file is still there. One the viewer *made* -- a
mode's output, a motion-corrected run, a selection -- has no file behind it, so
dropping it destroys it. Those are parked in a spill directory first, as raw
``.npy``: no compression, because a student's laptop is the machine that both
runs out of RAM and has the slow disk, and paying gzip on the way out would
turn a stall into a freeze. Reads come back memory-mapped, so scrubbing a
spilled run pages in one volume at a time instead of inflating the whole thing.
"""

from __future__ import annotations

import shutil
import tempfile
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
    #: Where this dataset's array was parked when RAM ran short. Only ever set
    #: for a memory-backed dataset; a file-backed one already has its file.
    spill: Path | None = None
    _future: Future[np.ndarray] | None = field(default=None, repr=False)

    @property
    def memory_backed(self) -> bool:
        """Whether this dataset exists only because the viewer made it.

        ``path`` is a label for these, not a location, so it is the one thing
        that must never be handed to a reader.
        """
        return self.info.storage == "MEMORY"

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
        spill_dir: str | Path | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self._ram_budget = ram_budget
        self._device_budget = device_budget
        self._zstd_threads = zstd_threads
        self._items: dict[str, Resident] = {}
        self._lock = threading.RLock()
        self._clock = 0
        # Made on first spill, not on construction: a session that never runs
        # short of RAM should leave nothing behind in the temp directory. A
        # caller-supplied directory is used as-is and never removed.
        self._spill_dir: Path | None = Path(spill_dir).expanduser() if spill_dir else None
        self._owns_spill_dir = spill_dir is None
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

    # -- spill ---------------------------------------------------------
    def spill_dir(self) -> Path:
        """The directory parked arrays go to, made on demand."""
        with self._lock:
            if self._spill_dir is None:
                self._spill_dir = Path(tempfile.mkdtemp(prefix="ffs-viewer-"))
            self._spill_dir.mkdir(parents=True, exist_ok=True)
            return self._spill_dir

    def _write_spill(self, key: str, array: np.ndarray) -> Path:
        """Park one array, atomically enough that a crash cannot half-write it."""
        # Keys are minted (``D1``, ``A_ICORR``) so they are already filename-safe,
        # but a mode is free to name its output, so do not trust that.
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        final = self.spill_dir() / f"{stem}.npy"
        scratch = final.with_suffix(".npy.part")
        # Through a file object, not a path: np.save silently appends ".npy" to
        # a path that does not already end in it, so the rename below would
        # look for a file that was never written.
        with open(scratch, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        scratch.replace(final)
        return final

    def _drop_spill(self, res: Resident) -> None:
        if res.spill is None:
            return
        try:
            res.spill.unlink(missing_ok=True)
        except OSError:
            pass  # a temp file we could not remove is not worth failing a close over
        res.spill = None

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
            res = self._items.pop(key, None)
        if res is not None:
            self._drop_spill(res)

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
        if res.spill is not None:
            # Memory-mapped, so stepping through a spilled run touches one
            # volume's worth of pages rather than reading the whole array back
            # in to throw all but one slab away.
            mapped = np.load(res.spill, mmap_mode="r", allow_pickle=False)
            vol = np.ascontiguousarray(mapped[..., index], dtype=np.float32)
        else:
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
            if res.spill is not None:
                arr = np.ascontiguousarray(np.load(res.spill, allow_pickle=False), dtype=np.float32)
                with self._lock:
                    res.array = arr
                    res._future = None
                    self._touch(res)
                self._enforce_ram_budget(protect=key)
                return arr

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
        """Drop both device and RAM copies, keeping the header and preview.

        Free for a dataset read off disk -- the file is still there to read
        again. A dataset the viewer *made* has no such file, so it is written
        to the spill directory first. Without that step the LRU silently
        destroys exactly the results a session exists to produce, and the next
        access fails on a path that never existed.
        """
        self.demote(key)
        with self._lock:
            res = self._items.get(key)
            if res is None or res.array is None:
                return
            array = res.array if (res.memory_backed and res.spill is None) else None
        if array is not None:
            try:
                spill = self._write_spill(key, array)
            except (OSError, ValueError):
                # Overshooting the RAM budget is recoverable; losing the only
                # copy of a result is not. Keep it resident and let the
                # eviction pass move on to a dataset that has a file.
                return
            with self._lock:
                res.spill = spill
        with self._lock:
            res.array = None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            spill_dir = self._spill_dir if self._owns_spill_dir else None
            self._spill_dir = None
            for res in self._items.values():
                res.spill = None
        if spill_dir is not None:
            shutil.rmtree(spill_dir, ignore_errors=True)

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
