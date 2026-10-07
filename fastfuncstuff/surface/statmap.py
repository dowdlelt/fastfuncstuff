"""A per-vertex statistical result, thresholded the way a volume stat is.

A ``.func.gii`` bucket from ``ffs_reml`` on a surface carries, per sub-brick, a label
and (for statistics) the AFNI stat code and parameters, and, in its file metadata,
SurfClustSim tables of cluster AREA (mm^2) by (pthr, alpha) for each sidedness. This
module turns that into what a viewer shows: a p-threshold on the statistic, and only
the clusters at least as large as the table demands. No Qt here, so it is tested and
scripted like any other primitive.

This is the second way a surface carries data, and deliberately separate from the
first: the 3-D window *samples a volume* between white and pial per fragment, but a
surface result is *bound to the mesh by vertex index* -- the mesh fingerprint is the
contract -- and is never resampled.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SurfaceData",
    "cluster_area_threshold",
    "data_colors",
    "load_surface_data",
    "stat_threshold",
    "surviving_clusters",
]

#: AFNI stat codes this module can turn a p into a threshold for.
_FITT, _FIFT, _FIZT = 3, 4, 5


@dataclass
class SurfaceData:
    """One ``.func.gii``/``.shape.gii``: values per vertex and sub-brick, and its stats."""

    path: str
    values: np.ndarray  # (V, K)
    labels: list[str]
    stat: dict[int, tuple[int, tuple[float, ...]]] = field(default_factory=dict)
    tables: dict[str, dict] = field(default_factory=dict)  # sided -> ClustSim JSON
    fingerprint: str = ""
    #: Every array a time point (NIFTI_INTENT_TIME_SERIES): a series to scrub, not a
    #: bucket of contrasts. Decided by the arrays' intents, not by a TR in the
    #: metadata -- a stats bucket inherits its input's TR.
    is_series: bool = False
    tr: float | None = None

    @property
    def n_vertices(self) -> int:
        return int(self.values.shape[0])

    def default_sub_brick(self) -> int:
        """The first t statistic, else the first statistic, else 0 -- what a viewer opens on."""
        for code in (_FITT, _FIZT, _FIFT):
            for k, (c, _) in sorted(self.stat.items()):
                if c == code:
                    return k
        return 0


def load_surface_data(path: str | os.PathLike) -> SurfaceData:
    import nibabel as nib

    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    cols, labels, stat = [], [], {}
    series = bool(img.darrays) and all(a.intent == 2001 for a in img.darrays)
    for k, arr in enumerate(img.darrays):
        if arr.intent in (1008, 1009):  # POINTSET / TRIANGLE: geometry, not data
            raise ValueError(f"{path} is a surface geometry file, not per-vertex data")
        cols.append(np.asarray(arr.data, np.float32).reshape(-1))
        meta = dict(arr.meta)
        labels.append(meta.get("Name", f"#{k}"))
        if "StatCode" in meta:
            params = tuple(float(x) for x in meta.get("StatParams", "").split())
            stat[k] = (int(meta["StatCode"]), params)
    file_meta = dict(img.meta)
    tables = {
        key.removeprefix("ClustSim_"): json.loads(val)
        for key, val in file_meta.items()
        if key.startswith("ClustSim_")
    }
    tr = file_meta.get("TR_seconds")
    return SurfaceData(
        os.fspath(path),
        np.stack(cols, axis=1),
        labels,
        stat,
        tables,
        file_meta.get("mesh_fingerprint", ""),
        is_series=series and len(cols) > 1,
        tr=float(tr) if tr else None,
    )


def stat_threshold(code: int | None, params: tuple[float, ...], p: float) -> float | None:
    """The |statistic| a p-value maps to, two-sided for t and z (as AFNI's viewer by
    default), one-sided for F. None when the sub-brick is not a statistic we know."""
    from scipy import stats

    if code == _FITT and params:
        return float(stats.t.isf(p / 2.0, params[0]))
    if code == _FIZT:
        return float(stats.norm.isf(p / 2.0))
    if code == _FIFT and len(params) >= 2:
        return float(stats.f.isf(p, params[0], params[1]))
    return None


def cluster_area_threshold(
    data: SurfaceData, p: float, alpha: float, sided: str = "bi-sided"
) -> float | None:
    """Minimum cluster area (mm^2) at voxel-wise ``p`` and family-wise ``alpha``.

    Read off the attached SurfClustSim table, interpolating log(area) against log(p)
    between the bracketing rows and against log(alpha) between columns (a table holds
    9 pthr x 4 alpha; asking between them is normal). None without a table, or outside
    its range (no extrapolation: a threshold must come from the simulation).
    """
    table = data.tables.get(sided)
    if not table:
        return None
    pthr = np.asarray(table["pthr"], np.float64)
    athr = np.asarray(table["athr"], np.float64)
    area = np.asarray(table["area_mm2"], np.float64)
    if not (pthr.min() <= p <= pthr.max() and athr.min() <= alpha <= athr.max()):
        return None
    la = np.log(np.maximum(area, 1e-6))
    # Rows: interpolate in log p (pthr descend in a table, so sort ascending first).
    order = np.argsort(pthr)
    rows = np.array(
        [np.interp(np.log(p), np.log(pthr[order]), la[order, j]) for j in range(la.shape[1])]
    )
    aord = np.argsort(athr)
    return float(np.exp(np.interp(np.log(alpha), np.log(athr[aord]), rows[aord])))


def surviving_clusters(
    values: np.ndarray,
    threshold: float,
    faces: np.ndarray,
    vertex_area: np.ndarray,
    min_area: float | None,
    sided: str = "bi-sided",
    one_sided_positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``(keep, labels)``: vertices in suprathreshold clusters of at least ``min_area``.

    ``bi-sided`` clusters positive and negative vertices apart (AFNI's default for t);
    ``2-sided`` lets them join; ``one_sided_positive`` keeps only values above
    ``threshold`` (an F). ``labels`` numbers the surviving clusters 1..n by area,
    largest first, 0 elsewhere.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    from .smooth import mesh_edges

    v = np.asarray(values, np.float64)
    n = v.shape[0]
    if one_sided_positive:
        groups = [v > threshold]
    elif sided == "2-sided":
        groups = [np.abs(v) > threshold]
    else:
        groups = [v > threshold, v < -threshold]
    e = mesh_edges(faces)
    labels = np.zeros(n, np.int64)
    found: list[tuple[float, np.ndarray]] = []
    for active in groups:
        ok = active[e[:, 0]] & active[e[:, 1]]
        g = sparse.csr_matrix((np.ones(ok.sum()), (e[ok, 0], e[ok, 1])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        areas = np.bincount(comp, weights=np.where(active, vertex_area, 0.0))
        for c in np.flatnonzero(areas > 0):
            if min_area is None or areas[c] >= min_area:
                found.append((float(areas[c]), np.flatnonzero((comp == c) & active)))
    for i, (_, ids) in enumerate(sorted(found, key=lambda x: -x[0]), start=1):
        labels[ids] = i
    return labels > 0, labels


def data_colors(
    values: np.ndarray,
    keep: np.ndarray,
    scale: float,
    lut: np.ndarray,
) -> np.ndarray:
    """``(V, 4)`` uint8: kept vertices on a symmetric colour scale, the rest clear."""
    unit = np.clip(0.5 + 0.5 * np.asarray(values, np.float64) / max(scale, 1e-12), 0.0, 1.0)
    out = np.zeros((len(unit), 4), np.uint8)
    out[:, :3] = lut[np.round(unit * 255).astype(int)][:, :3]
    out[:, 3] = np.where(keep, 255, 0)
    return out
