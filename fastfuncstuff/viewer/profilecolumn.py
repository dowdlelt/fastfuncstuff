"""The profile column's model: every vertex's ribbon profile, ordered and scored.

Qt-free so the ordering, the row<->vertex<->mm bookkeeping and the zoomed-out
pixel rule are testable on their own. :mod:`viewer.ui.profilewindow` draws
what this builds.

The one display rule worth stating: zoomed out, a screen row stands for many
vertices, and it shows the **most suspicious** of them rather than their
mean. Averaging would dissolve exactly the one bad profile the column exists
to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

from fastfuncstuff.surface.profiles import (
    Profiles,
    ProfileSpec,
    profile_scores,
    sample_profiles,
    slab_contour_order,
    tissue_levels,
)


@dataclass
class ProfileColumn:
    """Profiles for every vertex of every hemisphere, in row order."""

    hemis: tuple[str, ...]
    #: Per row: which hemisphere (index into ``hemis``) and which vertex.
    row_hemi: np.ndarray
    row_vertex: np.ndarray
    profiles: Profiles  # rows in row order
    scores: dict[str, np.ndarray]  # rows in row order
    #: Mid-thickness point of each row, scanner mm.
    mid: np.ndarray
    lo: float  # intensity drawn black (CSF level)
    hi: float  # intensity drawn white (WM level)
    spec: ProfileSpec
    #: White/pial the profiles were sampled from, per hemisphere, so an edit
    #: can be found and only its vertices recomputed.
    sampled: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    _tree: cKDTree | None = None
    _row_of: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.row_vertex.shape[0])

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.mid)
        return self._tree

    def row_of(self, hemi: str, vertices: np.ndarray) -> np.ndarray:
        """Rows holding the given vertices of one hemisphere."""
        lookup = self._row_of.get(hemi)
        if lookup is None:
            h = self.hemis.index(hemi)
            rows = np.flatnonzero(self.row_hemi == h)
            lookup = np.full(int(self.row_vertex[rows].max()) + 1, -1, np.int64)
            lookup[self.row_vertex[rows]] = rows
            self._row_of[hemi] = lookup
        v = np.asarray(vertices, np.int64)
        out = np.full(v.shape, -1, np.int64)
        ok = v < lookup.size
        out[ok] = lookup[v[ok]]
        return out

    def nearest_row(self, mm: tuple[float, float, float]) -> int:
        return int(self.tree().query(np.asarray(mm, np.float64))[1])

    def row_mm(self, row: int) -> tuple[float, float, float]:
        p = self.mid[int(row)]
        return (float(p[0]), float(p[1]), float(p[2]))

    def pick_rows(self, start: int, stop: int, n_pixels: int, score: str) -> np.ndarray:
        """Which row each of ``n_pixels`` screen rows shows, for rows [start, stop).

        One row per pixel when zoomed in; when several rows share a pixel, the
        one with the highest ``score`` -- the anomaly is what must survive.
        """
        start, stop = max(0, start), min(self.n_rows, stop)
        if stop <= start or n_pixels <= 0:
            return np.zeros(0, np.int64)
        edges = np.linspace(start, stop, n_pixels + 1)
        lo = np.floor(edges[:-1]).astype(np.int64)
        hi = np.maximum(np.floor(edges[1:]).astype(np.int64), lo + 1)
        s = self.scores[score]
        if (hi - lo).max() <= 1:
            return lo
        out = np.empty(n_pixels, np.int64)
        for k in range(n_pixels):
            seg = s[lo[k] : hi[k]]
            out[k] = lo[k] + int(np.argmax(seg))
        return out

    def flag_density(self, start: int, stop: int, n_pixels: int, score: str) -> np.ndarray:
        """Per screen row, the fraction of its rows flagged (score >= 0.8), x4, clipped.

        What a zoomed-out score strip shows. The per-row score of the row a
        pixel *draws* is always high there -- the worst row is the one drawn --
        so the strip says how much of the bin is in trouble instead. A quarter
        of a bin flagged already reads as full scale.
        """
        start, stop = max(0, start), min(self.n_rows, stop)
        if stop <= start or n_pixels <= 0:
            return np.zeros(0, np.float32)
        flagged = (self.scores[score][start:stop] >= 0.8).astype(np.float64)
        edges = np.linspace(0, stop - start, n_pixels + 1).astype(np.int64)
        starts = np.minimum(edges[:-1], stop - start - 1)
        counts = np.maximum(np.diff(np.append(starts, stop - start)), 1)
        return np.clip(np.add.reduceat(flagged, starts) / counts * 4.0, 0, 1).astype(np.float32)

    def grey(self, rows: np.ndarray) -> np.ndarray:
        """``(len(rows), S)`` uint8 profile intensities, CSF black to WM white."""
        v = self.profiles.values[rows]
        span = max(self.hi - self.lo, 1e-6)
        return (np.clip((v - self.lo) / span, 0.0, 1.0) * 255).astype(np.uint8)

    def boundary_columns(self) -> tuple[int, int] | None:
        """Columns of white and pial in fraction mode (where the guides go)."""
        ribbon = np.flatnonzero(self.profiles.kind == 0)
        if self.profiles.mode != "fraction" or ribbon.size == 0:
            return None
        return int(ribbon[0]), int(ribbon[-1])


def flags_by_vertex(col: ProfileColumn, score: str, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """One flag per vertex of each hemisphere (NaN where there is no row: medial wall)."""
    out: dict[str, np.ndarray] = {}
    values = col.scores.get(score)
    for i, h in enumerate(col.hemis):
        arr = np.full(sizes[h], np.nan, np.float32)
        if values is not None:
            rows = col.row_hemi == i
            arr[col.row_vertex[rows]] = values[rows]
        out[h] = arr
    return out


def build_column(
    hemis: dict,
    volume: np.ndarray,
    affine: np.ndarray,
    spec: ProfileSpec = ProfileSpec(),
    *,
    device: torch.device | None = None,
) -> ProfileColumn:
    """Sample, score and order every vertex of ``hemis`` (name -> Hemisphere)."""
    names = tuple(hemis)
    white = np.concatenate([hemis[h].states["white"] for h in names])
    pial = np.concatenate([hemis[h].states["pial"] for h in names])
    ring = np.concatenate(
        [
            hemis[h].states.get("sphere", hemis[h].states.get("inflated", hemis[h].states["white"]))
            for h in names
        ]
    )
    group = np.concatenate([np.full(hemis[h].n_vertices, i) for i, h in enumerate(names)])
    vertex = np.concatenate([np.arange(hemis[h].n_vertices) for h in names])
    # Cortex only: the medial wall's white and pial coincide, its "profiles"
    # are flat noise, and they topped the flag list on the first real subject.
    keep = np.concatenate(
        [
            hemis[h].cortex if hemis[h].cortex is not None else np.ones(hemis[h].n_vertices, bool)
            for h in names
        ]
    )
    white, pial, ring, group, vertex = (
        white[keep],
        pial[keep],
        ring[keep],
        group[keep],
        vertex[keep],
    )
    mid = 0.5 * (white + pial)
    order = slab_contour_order(mid, ring, group)
    prof = sample_profiles(white[order], pial[order], volume, affine, spec, device=device)
    return _finish(
        names,
        group[order],
        vertex[order],
        prof,
        mid[order],
        spec,
        {h: (hemis[h].states["white"].copy(), hemis[h].states["pial"].copy()) for h in names},
    )


def _finish(names, row_hemi, row_vertex, prof, mid, spec, sampled) -> ProfileColumn:
    if prof.mode == "fraction":
        scores = profile_scores(prof)
    else:
        # mm-mode columns do not line up with pial; there is nothing to score
        # by column, so rows are left unflagged.
        scores = {"worst": np.zeros(prof.values.shape[0], np.float32)}
    lv = tissue_levels(prof)
    return ProfileColumn(
        hemis=names,
        row_hemi=row_hemi,
        row_vertex=row_vertex,
        profiles=prof,
        scores=scores,
        mid=mid.astype(np.float64),
        lo=lv.csf,
        hi=lv.wm,
        spec=spec,
        sampled=sampled,
    )


def update_column(
    col: ProfileColumn,
    hemis: dict,
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Re-sample only cortex vertices whose white or pial moved; returns their rows.

    Scores are recomputed for every row afterwards, because they are judged
    against the whole brain -- cheap next to sampling.
    """
    changed_rows: list[np.ndarray] = []
    for h in col.hemis:
        old_w, old_p = col.sampled[h]
        w, p = hemis[h].states["white"], hemis[h].states["pial"]
        moved = np.flatnonzero(np.any(old_w != w, axis=1) | np.any(old_p != p, axis=1))
        if moved.size == 0:
            continue
        rows = col.row_of(h, moved)
        # Moved medial-wall vertices have no row.
        moved, rows = moved[rows >= 0], rows[rows >= 0]
        if moved.size == 0:
            col.sampled[h] = (w.copy(), p.copy())
            continue
        fresh = sample_profiles(w[moved], p[moved], volume, affine, col.spec, device=device)
        col.profiles.values[rows] = fresh.values
        col.profiles.thickness[rows] = fresh.thickness
        col.mid[rows] = 0.5 * (w[moved] + p[moved])
        col.sampled[h] = (w.copy(), p.copy())
        changed_rows.append(rows)
    if not changed_rows:
        return np.zeros(0, np.int64)
    col._tree = None
    if col.profiles.mode == "fraction":
        col.scores = profile_scores(col.profiles)
    return np.concatenate(changed_rows)


__all__ = ["ProfileColumn", "build_column", "flags_by_vertex", "update_column"]
