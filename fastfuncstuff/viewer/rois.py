"""ROI sets: a labelled set of voxel groups, and the one thing that makes them.

An atlas, a segmentation, a hand-drawn mask and the output of a clusterize are
the same object wearing four names -- some voxels, grouped, each group with an
identity. Everything this viewer wants to do with any of them is the same list:
name the group under the crosshair, average its time course, seed a correlation
from it, make it a row of a matrix, colour it distinctly.

So there is one type. Two shapes on disk collapse into it:

* **3-D labels** -- one volume whose integer values name the groups. This is
  what ``-atlas`` means in every tool that has the flag.
* **4-D frames** -- one sub-brick per group, each a mask. This is what falls out
  of a tool that had no way to say "these overlap", and the reason it stays
  supported is that the sub-brick labels then *are* the ROI names, which is the
  best-named case of the lot.

Overlap is the one thing the 3-D form cannot express, so converting 4-D to it
loses information: a voxel in two frames lands in the first. That is reported
(:attr:`RoiSet.overlaps`) rather than hidden, because a parcellation with 12%
overlap silently resolved is a correlation matrix that is quietly wrong.

No Qt, no session. The colours are here because a group's identity is visual.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.io.labels import LabelEntry

#: Above this many distinct values a 3-D integer volume is far more likely to
#: be an anatomy stored as int16 than a parcellation. Schaefer's largest is
#: 1000, so the line sits just above the biggest atlas anyone actually uses.
MAX_AUTO_LABELS = 1000

#: Fraction of neighbouring non-zero voxel pairs that must share a value for a
#: volume to read as labels. Measured on one subject: aparc+aseg 0.90, a binary
#: mask 1.0, and the anatomies 0.19-0.21 -- including FreeSurfer's 8-bit brain
#: in float32, 147 whole-number values that passed every other test here and
#: loaded as an atlas painted in 147 colours.
MIN_LABEL_FLATNESS = 0.5

#: Golden angle on the hue circle. Successive indices land as far apart as any
#: sequence can, which matters because adjacent labels in an atlas are usually
#: adjacent in space too -- a linear hue ramp would make neighbouring parcels
#: neighbouring colours, and the boundary between them would vanish.
_GOLDEN = 0.61803398875


@dataclass(frozen=True)
class Roi:
    """One group of voxels."""

    index: int
    name: str
    n_voxels: int
    #: Centre of mass in voxel indices. Fractional on purpose -- rounding it
    #: here would hide that the centre of a C-shaped region is outside it.
    center: tuple[float, float, float]
    color: tuple[int, int, int]

    @property
    def center_ijk(self) -> tuple[int, int, int]:
        """The centre as somewhere you can actually put the crosshair."""
        i, j, k = (int(round(c)) for c in self.center)
        return (i, j, k)


@dataclass(frozen=True)
class RoiSet:
    """Labelled groups over one voxel grid.

    ``labels`` is the canonical form and the only one anything downstream
    reads; ``rois`` is the description of it, in ascending label order.
    """

    name: str
    labels: np.ndarray  # (nx, ny, nz) int32, 0 = outside every group
    rois: tuple[Roi, ...] = ()
    #: Where this came from: ``"file:<layer key>"`` or ``"clusters:<key>#<n>"``.
    source: str = ""
    #: Voxels that were in more than one frame of a 4-D set and had to be
    #: assigned to one of them. Zero for a 3-D set, which cannot overlap.
    overlaps: int = 0
    _by_index: dict[int, Roi] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_index", {r.index: r for r in self.rois})

    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self) -> Iterator[Roi]:
        return iter(self.rois)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.labels.shape[0]), int(self.labels.shape[1]), int(self.labels.shape[2]))

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(r.index for r in self.rois)

    def find(self, index: int) -> Roi | None:
        return self._by_index.get(int(index))

    def at(self, ijk: tuple[int, int, int]) -> Roi | None:
        """The ROI under one voxel, or ``None`` outside every group."""
        i, j, k = (int(v) for v in ijk)
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return None
        return self.find(int(self.labels[i, j, k]))

    def mask(self, index: int) -> np.ndarray:
        return self.labels == int(index)

    def palette(self) -> np.ndarray:
        """``(max_index + 1, 3)`` uint8, row 0 black.

        Indexed by label value rather than by position, so the renderer needs
        no lookup and a set whose values are 2, 7 and 900 costs three rows of
        colour plus the gaps -- which is nothing, and is what keeps the drawn
        colour identical to the swatch in every list that names the same ROI.
        """
        top = max(self.indices, default=0)
        out = np.zeros((top + 1, 3), dtype=np.uint8)
        for roi in self.rois:
            out[roi.index] = roi.color
        return out

    def summary(self) -> str:
        total = int(sum(r.n_voxels for r in self.rois))
        text = f"{len(self.rois)} ROIs, {total:,} voxels"
        return f"{text}, {self.overlaps:,} overlapping" if self.overlaps else text


def roi_color(position: int) -> tuple[int, int, int]:
    """A distinct colour for the n-th ROI, stable and unlimited in count.

    Lightness alternates as well as hue, because on a grey underlay two hues at
    the same lightness can read as one region when the boundary is a single
    voxel wide.
    """
    import colorsys

    hue = (position * _GOLDEN) % 1.0
    value = 0.95 if position % 2 == 0 else 0.72
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62 if position % 3 else 0.85, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def looks_like_labels(volume: np.ndarray) -> bool:
    """Whether a 3-D volume is a label image rather than a picture.

    Integral, non-negative, and few enough distinct values to be names. The
    test deliberately errs towards "no": mistaking an anatomy for an atlas
    paints it in 200 colours, while missing an atlas costs one keypress to
    correct.
    """
    arr = np.asarray(volume)
    if arr.size == 0:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or finite.min() < 0:
        return False
    if not np.array_equal(finite, np.round(finite)):
        return False
    distinct = np.unique(finite)
    positive = distinct[distinct > 0]
    if not 0 < positive.size <= MAX_AUTO_LABELS:
        return False
    return arr.ndim != 3 or flatness(arr) >= MIN_LABEL_FLATNESS


def flatness(volume: np.ndarray) -> float:
    """How often a non-zero voxel's neighbour holds the same value.

    What separates a parcellation from an integer-valued picture. Counting
    distinct values cannot: an 8-bit anatomy has fewer than a Schaefer atlas.
    But labels come in patches, and intensity changes from voxel to voxel.
    """
    inside = volume > 0
    equal = pairs = 0
    for axis in range(3):
        a = np.moveaxis(volume, axis, 0)
        m = np.moveaxis(inside, axis, 0)
        both = m[1:] & m[:-1]
        # Masks multiplied, not used to index: fancy indexing copies both
        # halves of a 7M-voxel volume three times over.
        pairs += int(np.count_nonzero(both))
        equal += int(np.count_nonzero(both & (a[1:] == a[:-1])))
    return equal / pairs if pairs else 1.0


def _describe(
    labels: np.ndarray,
    names: dict[int, LabelEntry] | None = None,
) -> tuple[Roi, ...]:
    """Measure every group once: size, centre of mass, name, colour."""
    flat = labels.reshape(-1)
    top = int(flat.max()) if flat.size else 0
    if top <= 0:
        return ()
    counts = np.bincount(flat, minlength=top + 1)
    present = np.nonzero(counts)[0]
    present = present[present > 0]
    if present.size == 0:
        return ()
    # One pass per axis over the whole volume rather than one pass per ROI:
    # 400 boolean masks of a 900k-voxel grid is 360M comparisons and half a
    # second, and this is rebuilt whenever a clusterize threshold moves.
    coords = np.indices(labels.shape, dtype=np.float64).reshape(3, -1)
    sums = np.stack([np.bincount(flat, weights=c, minlength=top + 1) for c in coords])
    out = []
    for position, value in enumerate(present):
        n = int(counts[value])
        center = tuple(float(sums[a, value] / n) for a in range(3))
        entry = (names or {}).get(int(value))
        color = (entry.color if entry and entry.color else None) or roi_color(position)
        out.append(
            Roi(
                index=int(value),
                name=entry.name if entry else f"#{int(value)}",
                n_voxels=n,
                center=center,  # type: ignore[arg-type]
                color=color,
            )
        )
    return tuple(out)


def rois_from_labels(
    volume: np.ndarray,
    *,
    name: str = "rois",
    names: dict[int, LabelEntry] | None = None,
    source: str = "",
) -> RoiSet:
    """Build a set from a 3-D integer label volume."""
    arr = np.asarray(volume)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"a label volume must be 3-D, got shape {arr.shape}")
    labels = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.clip(np.round(labels), 0, None).astype(np.int32)
    return RoiSet(name=name, labels=labels, rois=_describe(labels, names), source=source)


def rois_from_frames(
    data: np.ndarray,
    *,
    name: str = "rois",
    frame_names: tuple[str, ...] = (),
    source: str = "",
    threshold: float = 0.0,
) -> RoiSet:
    """Build a set from a 4-D stack where each sub-brick is one ROI's mask.

    Frames are numbered from 1 so that 0 keeps meaning "outside", and an
    earlier frame wins a contested voxel -- an arbitrary rule, which is why the
    count of contested voxels is carried on the result rather than discarded.
    """
    arr = np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"ROI frames must be 4-D, got shape {arr.shape}")
    labels = np.zeros(arr.shape[:3], dtype=np.int32)
    overlaps = 0
    for frame in range(arr.shape[3]):
        hit = np.abs(np.nan_to_num(arr[..., frame])) > threshold
        overlaps += int(np.count_nonzero(hit & (labels > 0)))
        labels[hit & (labels == 0)] = frame + 1
    names = {
        i + 1: LabelEntry(i + 1, frame_names[i])
        for i in range(arr.shape[3])
        if i < len(frame_names) and frame_names[i]
    }
    return RoiSet(
        name=name,
        labels=labels,
        rois=_describe(labels, names),
        source=source,
        overlaps=overlaps,
    )


def roi_means(
    rows: torch.Tensor,
    labels_v: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    """Mean time course of each ROI: ``(V, T)`` rows to ``(K, T)``.

    A scatter-add rather than a mask per ROI. Both are correct; this one is a
    single pass over the data instead of K of them, which is the difference
    between a matrix that redraws as you change a setting and one that does not.
    """
    if not indices:
        raise ValueError("no ROIs to average")
    lookup = torch.full((max(indices) + 1,), -1, dtype=torch.long, device=rows.device)
    for position, value in enumerate(indices):
        lookup[value] = position
    slot = lookup[labels_v.clamp(min=0, max=len(lookup) - 1).long()]
    keep = slot >= 0
    slot, kept = slot[keep], rows[keep]
    out = torch.zeros((len(indices), rows.shape[1]), dtype=rows.dtype, device=rows.device)
    out.index_add_(0, slot, kept)
    counts = torch.zeros(len(indices), dtype=rows.dtype, device=rows.device)
    counts.index_add_(0, slot, torch.ones_like(slot, dtype=rows.dtype))
    return out / counts.clamp(min=1).unsqueeze(1)


__all__ = [
    "MAX_AUTO_LABELS",
    "Roi",
    "RoiSet",
    "looks_like_labels",
    "roi_color",
    "roi_means",
    "rois_from_frames",
    "rois_from_labels",
]
