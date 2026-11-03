"""Painted ROI layers in a session: making them, painting into them, saving them.

A painted layer is an ordinary ROI layer -- integer labels, drawn by identity,
named in the readout, usable wherever an ROI is (the RGB mode's stats, the
matrix, a mask) -- with three things of its own:

* **colour by value.** A loaded atlas colours its regions by rank, so a label
  painted between two others would recolour both. Here label k is always
  ``roi_color(k - 1)``, and the palette comes from a small table instead of a
  scan of the volume -- which is what lets a stroke show at once.
* **painting in place.** A stroke writes the host array and the on-screen copy
  of the voxels it touched; the volume is never re-uploaded.
* **undo by stroke**, kept here rather than in a window so a replayed script
  undoes what the person undid.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastfuncstuff.io.labels import LabelEntry
from fastfuncstuff.unique import sorted_unique
from fastfuncstuff.viewer.paint import brush_voxels
from fastfuncstuff.viewer.rois import roi_color
from fastfuncstuff.viewer.state import Plane

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession

#: Strokes kept for undo, per layer.
UNDO_DEPTH = 200
#: New store generations for painted layers, far from the store's own count so
#: the two can never coincide.
_GENERATIONS = itertools.count(1 << 40)


@dataclass
class _Stroke:
    """What one stroke changed: flat voxel indices and the values they had."""

    index: np.ndarray
    before: np.ndarray


def label_entry(value: int) -> LabelEntry:
    return LabelEntry(int(value), f"label {int(value)}", roi_color(int(value) - 1))


class Painter:
    """The session's painted layers. One per session."""

    def __init__(self, session: ViewerSession) -> None:
        self.session = session
        self.tables: dict[str, dict[int, LabelEntry]] = {}
        self._history: dict[str, list[_Stroke]] = {}
        self._open: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        #: Voxels per label, kept up to date by every change rather than
        #: counted: the status line asks on every repaint, and a scan of a
        #: whole-head anatomy per brush movement is not affordable.
        self._counts: dict[str, dict[int, int]] = {}

    def owns(self, key: str) -> bool:
        return key in self.tables

    # -- making ------------------------------------------------------------
    def new_layer(self, key: str, name: str = "") -> str:
        """An empty label layer on the display grid, on top of the stack."""
        from fastfuncstuff.viewer.layers import Layer

        st = self.session.state
        if st.grid is None:
            raise ValueError("load an image first: a painted layer takes its grid")
        shape = tuple(int(v) for v in st.grid.shape)
        affine = np.asarray(st.grid.affine, dtype=float)
        if st.layers.find(key) is not None:
            raise ValueError(f"layer {key} already exists")
        name = name or f"paint {key}"
        self.session.store.adopt(key, np.zeros(shape, dtype=np.float32), name=name)
        self.tables[key] = {}
        self._history[key] = []
        self._counts[key] = {}
        st.layers.add(
            Layer(
                key=key,
                name=name,
                path=f"<{name}>",
                shape=shape,  # type: ignore[arg-type]
                n_volumes=1,
                affine=affine,
                roi=True,
                opacity=0.6,
                range_lo=0.0,
                range_hi=1.0,
                source="paint:",
            )
        )
        st.selected = key
        return key

    # -- painting ----------------------------------------------------------
    def apply(
        self,
        key: str,
        value: int,
        centres_mm: np.ndarray,
        *,
        radius_mm: float,
        brush: str,
        plane: Plane,
        overwrite: bool = True,
    ) -> int:
        """Paint ``value`` (0 erases) at each centre; return how many voxels changed.

        Changes are held as the open stroke until :meth:`close_stroke`, which
        makes them one undo step.
        """
        layer = self.session.state.layers.get(key)
        res = self.session.store.get(key)
        if res.array is None:
            raise ValueError(f"{layer.name} is not resident")
        vol = res.array[..., 0]
        idx_sets = [
            brush_voxels(
                c,
                radius_mm,
                affine=layer.affine,
                shape=layer.shape,
                plane=plane,
                brush=brush,
            )
            for c in np.atleast_2d(centres_mm)
        ]
        if not idx_sets:
            return 0
        pts = sorted_unique(np.concatenate(idx_sets), axis=0)
        flat = np.ravel_multi_index(pts.T, layer.shape)
        current = vol.reshape(-1)[flat]
        change = current != value
        if value != 0 and not overwrite:
            # Paint only empty voxels: a region drawn next to another stops
            # at its edge instead of eating into it.
            change &= current == 0
        flat, before = flat[change], current[change]
        if flat.size == 0:
            return 0
        vol.reshape(-1)[flat] = value
        self._count(key, before, value)
        self._open.setdefault(key, []).append((flat, before))
        if value and value not in self.tables[key]:
            self.tables[key][value] = label_entry(value)
            self._palette_changed(key)
        self._touched(key, flat, value)
        return int(flat.size)

    def close_stroke(self, key: str) -> int:
        """Make the open stroke one undo step; return the voxels it changed."""
        parts = self._open.pop(key, [])
        if not parts:
            return 0
        # First change of a voxel within the stroke is its true "before".
        flat = np.concatenate([p[0] for p in parts])
        before = np.concatenate([p[1] for p in parts])
        flat, first = sorted_unique(flat, return_index=True)
        history = self._history.setdefault(key, [])
        history.append(_Stroke(flat, before[first]))
        del history[:-UNDO_DEPTH]
        self._changed(key)
        return int(flat.size)

    def undo(self, key: str) -> bool:
        self.close_stroke(key)
        history = self._history.get(key) or []
        if not history:
            return False
        stroke = history.pop()
        vol = self.session.store.get(key).array[..., 0]  # type: ignore[index]
        now = vol.reshape(-1)[stroke.index].copy()
        vol.reshape(-1)[stroke.index] = stroke.before
        self._changed(key)
        counts = self._counts.setdefault(key, {})
        for values, sign in ((now, -1), (stroke.before, 1)):
            for v, n in zip(
                *sorted_unique(values.astype(np.int64), return_counts=True), strict=True
            ):
                if v > 0:
                    counts[int(v)] = counts.get(int(v), 0) + sign * int(n)
        self.session.invalidate(key)
        self._roi_changed(key)
        return True

    def can_undo(self, key: str) -> bool:
        return bool(self._history.get(key)) or bool(self._open.get(key))

    def _touched(self, key: str, flat: np.ndarray, value: int) -> None:
        """Write the change into the on-screen copy too, so nothing re-uploads."""
        for cache_key, (tensor, _has_nan) in list(self.session._volume_cache.items()):
            if cache_key[0] != key:
                continue
            idx = torch.as_tensor(flat, device=tensor.device)
            tensor.view(-1)[idx] = float(value)
        self._roi_changed(key)

    def _changed(self, key: str) -> None:
        """A finished stroke or an undo: tell everything that caches the layer.

        In-place painting keeps the 2-D views current dab by dab. What caches
        by the store's generation -- the 3-D window's textures, a sampler, a
        depth window's region -- or by key -- a mask made from this layer --
        learns of a stroke here, once, at its end rather than every dab.
        """
        self.session.store.get(key).generation = next(_GENERATIONS)
        for k in [k for k in self.session._source_masks if k[0] == key]:
            del self.session._source_masks[k]
        self._roi_changed(key)

    def _roi_changed(self, key: str) -> None:
        # The region description is rebuilt only when something asks for it.
        self.session._roi_sets.pop(key, None)

    def _palette_changed(self, key: str) -> None:
        for k in [k for k in self.session._roi_palettes if k[0] == key]:
            del self.session._roi_palettes[k]

    # -- reading -----------------------------------------------------------
    def palette(self, key: str, device: torch.device) -> torch.Tensor:
        """``(max label + 1, 3)`` in [0, 1], by value; row 0 unused."""
        table = self.tables.get(key, {})
        top = max(table, default=1)
        out = np.zeros((top + 1, 3), dtype=np.float32)
        for value, entry in table.items():
            out[value] = np.asarray(entry.color or roi_color(value - 1), np.float32) / 255.0
        return torch.as_tensor(out, device=device)

    def name_at(self, key: str, value: float) -> str:
        v = int(round(value))
        if v <= 0:
            return "--"
        entry = self.tables.get(key, {}).get(v)
        return entry.name if entry is not None else f"label {v}"

    def counts(self, key: str) -> dict[int, int]:
        return {v: n for v, n in self._counts.get(key, {}).items() if n > 0}

    def _count(self, key: str, before: np.ndarray, value: int) -> None:
        counts = self._counts.setdefault(key, {})
        for v, n in zip(*sorted_unique(before.astype(np.int64), return_counts=True), strict=True):
            if v > 0:
                counts[int(v)] = counts.get(int(v), 0) - int(n)
        if value > 0:
            counts[value] = counts.get(value, 0) + int(before.size)

    # -- saving ------------------------------------------------------------
    def save(self, key: str, path: str | Path, like: str | None = None) -> Path:
        """Integer labels as NIfTI, on the layer's own grid or nearest-resampled onto ``like``'s."""
        import nibabel as nib

        layer = self.session.state.layers.get(key)
        labels = np.rint(self.session.store.get(key).array[..., 0]).astype(np.int16)  # type: ignore[index]
        affine = np.asarray(layer.affine, dtype=float)
        if like:
            target = self.session.state.layers.get(like)
            labels = resample_labels_nearest(labels, affine, target.shape, target.affine)
            affine = np.asarray(target.affine, dtype=float)
        img = nib.Nifti1Image(labels, affine)
        img.header.set_xyzt_units("mm", "sec")
        out = Path(path)
        nib.save(img, str(out))
        return out


def resample_labels_nearest(
    labels: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    target_affine: np.ndarray,
) -> np.ndarray:
    """Labels onto another grid, nearest neighbour, 0 outside -- one z-slab at a time.

    Nearest because a label interpolated is a label that does not exist.
    """
    to_source = np.linalg.inv(np.asarray(affine, float)) @ np.asarray(target_affine, float)
    rot, shift = to_source[:3, :3], to_source[:3, 3]
    nx, ny, nz = (int(n) for n in shape)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    plane = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size)]).astype(float)
    out = np.zeros((nx, ny, nz), dtype=labels.dtype)
    bounds = np.asarray(labels.shape)[:, None]
    for k in range(nz):
        plane[2] = k
        src = np.rint(rot @ plane + shift[:, None]).astype(np.int64)
        inside = np.all((src >= 0) & (src < bounds), axis=0)
        slab = np.zeros(plane.shape[1], dtype=labels.dtype)
        slab[inside] = labels[src[0, inside], src[1, inside], src[2, inside]]
        out[:, :, k] = slab.reshape(nx, ny)
    return out
