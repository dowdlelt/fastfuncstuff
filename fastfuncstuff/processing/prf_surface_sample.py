"""A volume pRF bucket onto a subject's surfaces, for maps drawn on the sphere.

Sampling fitted *parameters* is not sampling a time series. A series wants the
ribbon mean, interpolated; a parameter map does not:

* interpolating blends neighbours, so a read between a voxel at angle 350 and one
  at 10 would be 180. Every read here takes the value of the voxel it lands in
  (nearest neighbour): the statistic sees real fits, never blends of them;
* polar angle is cyclic, so it is a **circular** mean (the weighted resultant of
  cos and sin): 350 and 10 make 0;
* every other parameter (x, y, eccentricity, size, ...) is a **weighted median**
  by default: one fit in white matter or CSF that landed at the screen's edge moves
  a mean, not a median. ``stat="mean"`` gives the weighted mean instead;
* the weight of a read is its share of the vertex's footprint and depth, times its
  R2, and zero below ``r2_min`` (``weight="r2"``); ``weight="none"`` drops the R2
  factor and the floor;
* R2 itself is the plain ribbon mean (footprint and depth weights only), so a
  threshold on it asks how much of the column the model explains.

The read points are the same equivolume footprints ``project_to_surface`` uses
(:func:`surface_projection.surface_targets`), so the two cannot disagree about
where a vertex's cortex is. The chain is the identity, as for ``ffs_pyrf
-fs_subj``: the bucket must already be in register with the subject's anatomy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .surface_projection import DEFAULT_DEPTHS, resolve_subject, surface_targets

__all__ = ["SURFACE_STATS", "SURFACE_WEIGHTS", "SurfacePrf", "sample_prf_bucket"]

#: Statistic for the non-cyclic maps.
SURFACE_STATS = ("median", "mean")
#: ``r2``: weighted by R2, reads under ``r2_min`` excluded; ``none``: footprint and
#: depth weights only.
SURFACE_WEIGHTS = ("r2", "none")

_CYCLIC = ("angle",)


@dataclass
class SurfacePrf:
    """One hemisphere's sampled bucket."""

    hemi: str
    maps: dict[str, np.ndarray]  # label -> (V,), NaN where not covered
    covered: np.ndarray  # (V,) bool: cortex, every read inside the volume, some weight
    meta: dict[str, str]  # subject, mesh, mesh_fingerprint, ...


def _is_cyclic(label: str) -> bool:
    return label.split("_t")[0] in _CYCLIC


def _per_vertex(rows: np.ndarray, weights: np.ndarray, values: np.ndarray, n: int, stat: str):
    """Weighted ``stat`` of ``values`` grouped by ``rows``; NaN for empty rows."""
    keep = weights > 0
    rows, weights, values = rows[keep], weights[keep], values[keep]
    out = np.full(n, np.nan)
    total = np.bincount(rows, weights, minlength=n)
    has = total > 0
    if stat == "mean":
        out[has] = np.bincount(rows, weights * values, minlength=n)[has] / total[has]
        return out
    if rows.size == 0:
        return out
    order = np.lexsort((values, rows))
    rows, weights, values = rows[order], weights[order], values[order]
    cum = np.cumsum(weights)
    first = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
    start = np.repeat((cum - weights)[first], np.diff(np.r_[first, rows.size]))
    # The lower weighted median: the first read whose running weight reaches half.
    past = np.flatnonzero(cum - start >= 0.5 * total[rows] * (1 - 1e-9))
    hit_rows, at = np.unique(rows[past], return_index=True)
    out[hit_rows] = values[past[at]]
    return out


def _circular(rows: np.ndarray, weights: np.ndarray, degrees: np.ndarray, n: int):
    theta = np.deg2rad(degrees)
    c = np.bincount(rows, weights * np.cos(theta), minlength=n)
    s = np.bincount(rows, weights * np.sin(theta), minlength=n)
    total = np.bincount(rows, weights, minlength=n)
    return np.where(total > 0, np.degrees(np.arctan2(s, c)) % 360.0, np.nan)


def sample_prf_bucket(
    bucket: dict[str, np.ndarray],
    affine: np.ndarray,
    zooms: tuple[float, ...],
    subject_dir: str | os.PathLike,
    labels: list[str],
    r2_key: str = "r2",
    r2_min: float = 0.1,
    weight: str = "r2",
    stat: str = "median",
    mesh: str = "native",
    hemis=("lh", "rh"),
    fractions=DEFAULT_DEPTHS,
    sample: str = "footprint",
) -> list[SurfacePrf]:
    """Sample ``labels`` of a volume bucket (label -> 3-D array) onto each hemisphere.

    ``r2_key`` is the column the weights come from, in the bucket's own units
    (ffs_pyrf: a fraction), and ``r2_min`` the floor below which a read gives no
    weight, in the same units. ``sample="footprint"`` reads each vertex's patch of
    cortex (one voxel face); ``"point"`` one point per vertex and depth.
    """
    if weight not in SURFACE_WEIGHTS:
        raise ValueError(f"weight must be one of {SURFACE_WEIGHTS}, got {weight!r}")
    if stat not in SURFACE_STATS:
        raise ValueError(f"stat must be one of {SURFACE_STATS}, got {stat!r}")
    if sample not in ("footprint", "point"):
        raise ValueError(f"sample must be 'footprint' or 'point', got {sample!r}")
    if r2_key not in bucket:
        raise ValueError(f"the bucket has no {r2_key!r} to weight by")
    labels = list(dict.fromkeys(labels))
    shape = np.asarray(next(iter(bucket.values())).shape[:3])
    face = float(np.prod(np.asarray(zooms[:3], np.float64)) ** (2.0 / 3.0))
    to_voxel = np.linalg.inv(affine)
    subject = resolve_subject(subject_dir)
    columns = {k: np.asarray(bucket[k], np.float64).reshape(-1) for k in {*labels, r2_key}}

    out = []
    targets = surface_targets(
        subject, mesh, tuple(hemis), tuple(fractions), face if sample == "footprint" else None
    )
    for target in targets:
        smp = target.sampling
        n = smp.n_vertices
        ijk = np.rint(smp.points @ to_voxel[:3, :3].T + to_voxel[:3, 3]).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < shape)).all(axis=1)
        flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, shape - 1).T), tuple(shape))

        coo = smp.operator.tocoo()
        rows = (coo.row % n).astype(np.int64)  # (depth, vertex) rows -> vertex
        reads = coo.col.astype(np.int64)
        share = coo.data.astype(np.float64) / smp.n_depths
        voxel = flat[reads]
        read_inside = inside[reads]

        values = {k: columns[k][voxel] for k in labels}
        r2 = np.nan_to_num(columns[r2_key][voxel], nan=0.0)
        valid = read_inside & np.all([np.isfinite(v) for v in values.values()], axis=0)
        if weight == "r2":
            w = np.where(valid & (r2 >= r2_min), share * np.clip(r2, 0.0, None), 0.0)
        else:
            w = np.where(valid, share, 0.0)

        outside = np.bincount(rows, ~read_inside, minlength=n) > 0
        has_reads = np.bincount(rows, minlength=n) > 0
        covered = has_reads & ~outside & (np.bincount(rows, w, minlength=n) > 0)
        if target.cortex is not None:
            covered &= target.cortex

        maps: dict[str, np.ndarray] = {}
        for label in labels:
            if label == r2_key:
                plain = np.where(read_inside, share, 0.0)
                result = _per_vertex(rows, plain, r2, n, "mean")
            elif _is_cyclic(label):
                result = _circular(rows, w, np.where(valid, values[label], 0.0), n)
            else:
                result = _per_vertex(rows, w, np.where(valid, values[label], 0.0), n, stat)
            maps[label] = np.where(covered, result, np.nan)

        meta = {k: target.meta[k] for k in ("subject", "mesh", "mesh_fingerprint")}
        meta.update(
            depths=" ".join(f"{float(f):g}" for f in fractions),
            sampling=sample,
            surface_weight=weight,
            surface_stat=stat,
        )
        out.append(SurfacePrf(hemi=target.hemi, maps=maps, covered=covered, meta=meta))
    return out
