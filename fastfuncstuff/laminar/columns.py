"""Cortical columns made on the fly: overlapping geodesic disks of the ribbon,
each read through depth, with its own measured depth PSF.

The laminar model wants one depth-resolved response per region. Instead of a
region drawn ahead of time, cortex is covered by many overlapping disks centred
on responsive cortex, so every responsive vertex is fitted by several regions
that differ only in where their edges fall.

Everything is linear, so nothing is read per column:

1. the ribbon under all columns is tiled once into cells (:mod:`surface.slab`),
   each owned by a vertex and split over depth bins;
2. one sparse matrix reads the data grid into **vertex x depth** sums (cell
   volume x interpolated value), with their volumes and depth-PSF numerators
   (:func:`processing.roi_depth.mix_rows`);
3. a column is a set of vertices, so its curves, coverage and PSF are ratios of
   sums over its members: one sparse product for every column at once.

Step 3 is exactly the column read ``processing.roi_depth`` would do with the
column as an ROI: same cells, same weights, summed in a different order.
Fitting voxel curves first and sampling them after is exact too, because the
curves come from a linear estimator (a fixed-lambda smooth FIR,
``ffs_deconvolve -tent-smooth global``), so the column curve of the fits equals
the fit of the column mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from fastfuncstuff.processing.roi_depth import (
    DepthOptions,
    _compose,
    _Roi,
    default_step,
    mix_rows,
    plan_depth_multi,
    psf_halo_mm,
)

_KEY = 1 << 40


def responsive_vertices(
    path: str,
    surfaces,
    *,
    threshold: float | None = None,
    absolute: bool = True,
    fracs: tuple[float, ...] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85),
) -> list[np.ndarray]:
    """Per surface, the vertices whose column the volume ``path`` marks.

    Read nearest-voxel along each column at ``fracs`` (white 0 .. pial 1): a
    column is responsive if any read passes -- non-zero for a mask, ``|value| >=
    threshold`` (or ``value >= threshold`` with ``absolute=False``) for a
    statistic. Any depth, because a responsive patch need not span the ribbon
    (a thresholded map that peaks by pial). The medial wall never is.
    """
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(path)
    arr = np.asanyarray(img.dataobj).astype(np.float64)
    if arr.ndim == 4:
        arr = arr[..., 0]
    inv = np.linalg.inv(np.asarray(img.affine, np.float64))
    out = []
    for s in surfaces:
        w = np.asarray(s.white, np.float64)
        p = np.asarray(s.pial, np.float64)
        hit = np.zeros(len(w), bool)
        for frac in fracs:
            ijk = np.rint((w + frac * (p - w)) @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < np.asarray(arr.shape[:3])), axis=1)
            v = np.zeros(len(w))
            v[inside] = arr[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
            if threshold is None:
                hit |= v != 0
            else:
                hit |= (np.abs(v) if absolute else v) >= threshold
        if s.cortex is not None:
            hit &= np.asarray(s.cortex, bool)
        out.append(hit)
    return out


@dataclass
class ColumnSet:
    """Columns as vertex sets. ``members`` is ``(n_columns, n_vertices)`` over
    :attr:`vertices` (a ``(surface, vertex)`` list shared with the reads)."""

    centre_surface: np.ndarray  # (C,)
    centre_vertex: np.ndarray  # (C,)
    members: sp.csr_matrix  # (C, n_v) 0/1
    vertex_surface: np.ndarray  # (n_v,)
    vertex_id: np.ndarray  # (n_v,)
    radius_mm: float
    spacing_mm: float

    @property
    def n_columns(self) -> int:
        return int(self.centre_vertex.size)

    def coverage_count(self) -> np.ndarray:
        """How many columns contain each vertex of :attr:`vertex_id`."""
        return np.asarray(self.members.sum(axis=0)).ravel()


def _midthickness(s) -> np.ndarray:
    return 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))


def make_columns(
    surfaces,
    responsive: list[np.ndarray],
    *,
    radius_mm: float,
    spacing_mm: float | None = None,
    seed: int = 0,
) -> ColumnSet:
    """Disks of geodesic ``radius_mm`` (on the midthickness) centred so that every
    responsive vertex is within ``spacing_mm`` of a centre.

    Centres are a greedy cover of the responsive vertices in a random order (a
    new centre wherever the cover has not reached), so their number is set by the
    responsive area and the spacing -- the data decide how many columns there
    are. ``spacing_mm`` defaults to half the radius: every vertex well inside
    responsive cortex then falls in several columns, and a column edge never
    decides a vertex alone. A column takes its whole disk, responsive or not:
    the responsive mask places columns, it does not trim them.
    """
    from fastfuncstuff.surface.mesh import MeshTopology, geodesic_ball

    spacing = float(spacing_mm) if spacing_mm else 0.5 * float(radius_mm)
    rng = np.random.default_rng(seed)
    cs, cv, member_lists = [], [], []
    for h, (s, resp) in enumerate(zip(surfaces, responsive, strict=True)):
        cand = np.flatnonzero(resp)
        if cand.size == 0:
            continue
        mid = _midthickness(s)
        topo = MeshTopology.from_faces(np.asarray(s.faces), mid.shape[0])
        cortex = None if s.cortex is None else np.asarray(s.cortex, bool)
        covered = np.zeros(mid.shape[0], bool)
        for v in rng.permutation(cand):
            if covered[v]:
                continue
            near, _ = geodesic_ball(mid, topo, int(v), spacing)
            covered[near] = True
            disk, _ = geodesic_ball(mid, topo, int(v), float(radius_mm))
            if cortex is not None:
                disk = disk[cortex[disk]]
            cs.append(h)
            cv.append(int(v))
            member_lists.append(h * _KEY + disk.astype(np.int64))
    if not cs:
        raise ValueError("no responsive vertex to centre a column on")
    keys = np.unique(np.concatenate(member_lists))
    rows = np.repeat(np.arange(len(member_lists)), [m.size for m in member_lists])
    cols = np.searchsorted(keys, np.concatenate(member_lists))
    members = sp.csr_matrix(
        (np.ones(cols.size), (rows, cols)), shape=(len(member_lists), keys.size)
    )
    return ColumnSet(
        np.asarray(cs), np.asarray(cv), members, (keys // _KEY).astype(np.int64),
        (keys % _KEY).astype(np.int64), float(radius_mm), spacing,
    )  # fmt: skip


@dataclass
class VertexDepthReads:
    """Vertex x depth reads of a data grid, per depth count ``K``.

    Rows are vertex-major, depth 1 (deepest) first, over the vertices of the
    :class:`ColumnSet` they were built for. ``weights[K] @ data[voxels]`` gives
    volume-weighted sums; ``volume[K]`` holds what they divide by.
    """

    ks: list[int]
    voxels: np.ndarray  # flat indices into the data grid that any read touches
    weights: dict[int, sp.csr_matrix]  # (n_v * K, voxels.size)
    volume: dict[int, np.ndarray]  # (n_v, K) read weight kept (mask / FOV drop reads)
    volume_all: dict[int, np.ndarray]  # (n_v, K) cell volume before any drop
    mixed: dict[int, np.ndarray] = field(default_factory=dict)  # (n_v, K, K) PSF numerators
    total: dict[int, np.ndarray] = field(default_factory=dict)  # (n_v, K)
    notes: list[str] = field(default_factory=list)


def vertex_depth_reads(
    surfaces,
    columns: ColumnSet,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    ks: list[int],
    opts: DepthOptions = DepthOptions(),
    *,
    mask: np.ndarray | None = None,
    psf: bool = True,
    device=None,
) -> VertexDepthReads:
    """Tile the ribbon under every column once and read it onto the data grid.

    ``mask`` (data grid) limits which voxels a read may use, as for ROI reads;
    the depth PSF is geometry and reads unmasked, as ``depth_sampling_multi``
    does.
    """
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    n_vox = int(np.prod(shape3))
    mask3 = None if mask is None else np.asarray(mask, bool).reshape(shape3)
    members = {}
    for h in np.unique(columns.vertex_surface):
        m = np.zeros(len(surfaces[int(h)].white), bool)
        m[columns.vertex_id[columns.vertex_surface == h]] = True
        members[int(h)] = m
    step = opts.dxyz or default_step(aff)
    ks = sorted({int(k) for k in ks})
    plans = plan_depth_multi(
        [_Roi(1, "columns", members)], surfaces, step, opts, ks,
        halo_mm=psf_halo_mm(opts.interp, default_step(aff)) if psf else 3.0,
        keep_all=psf, device=device,
    )  # fmt: skip
    cells = plans[0].cells
    key = cells.surface.astype(np.int64) * _KEY + cells.vertex
    vkeys = columns.vertex_surface * _KEY + columns.vertex_id
    slot = np.searchsorted(vkeys, key)
    slot_ok = (slot < vkeys.size) & (vkeys[np.minimum(slot, vkeys.size - 1)] == key)
    n_v = vkeys.size
    ops = {}
    for k in ks:
        cell, k_bin, share = cells.bin_shares(k)
        ok = slot_ok[cell]
        ops[k] = sp.csr_matrix(
            (cells.volume[cell[ok]] * share[ok], (slot[cell[ok]] * k + k_bin[ok], cell[ok])),
            shape=(n_v * k, len(cells)),
        )
    stacked = sp.vstack([ops[k] for k in ks]).tocsr()
    w_all, dropped = _compose(stacked, cells.centre, aff, shape3, opts.interp, mask3, n_vox)
    voxels = np.flatnonzero(np.asarray((w_all != 0).sum(axis=0)).ravel() > 0)
    w_all = w_all[:, voxels].tocsr()
    out = VertexDepthReads(ks, voxels, {}, {}, {}, notes=[ln for ln in plans[0].notes])
    row0 = 0
    for k in ks:
        r1 = row0 + n_v * k
        out.weights[k] = w_all[row0:r1].tocsr()
        out.volume[k] = np.asarray(out.weights[k].sum(axis=1)).reshape(n_v, k)
        out.volume_all[k] = np.asarray(ops[k].sum(axis=1)).reshape(n_v, k)
        row0 = r1
    if psf:
        w_psf, _ = _compose(stacked, cells.centre, aff, shape3, opts.interp, None, n_vox)
        every = plans[0].all_cells
        assert every is not None
        inv = np.linalg.inv(aff)
        every_ijk = every.centre @ inv[:3, :3].T + inv[:3, 3]
        row0 = 0
        for k in ks:
            r1 = row0 + n_v * k
            mixed, total = mix_rows(
                every, k, w_psf[row0:r1].tocsr(), every_ijk, shape3, abs(np.linalg.det(aff[:3, :3]))
            )
            out.mixed[k] = mixed.reshape(n_v, k, k)
            out.total[k] = total.reshape(n_v, k)
            row0 = r1
    return out


@dataclass
class ColumnCurves:
    """Every column's depth-resolved curves at one depth count."""

    K: int
    values: np.ndarray  # (C, K, *trailing) depth 1 deepest first
    coverage: np.ndarray  # (C, K) share of each bin's ribbon that was read
    psf: np.ndarray | None  # (C, K, K) rows normalised, depth 1 deepest first


def column_curves(
    reads: VertexDepthReads, columns: ColumnSet, data: np.ndarray, K: int
) -> ColumnCurves:
    """Column x depth means of ``data`` ``(reads.voxels.size, ...)`` -- any trailing
    shape (time x condition) -- with each column's coverage and PSF."""
    n_v = columns.vertex_id.size
    trailing = data.shape[1:]
    flat = np.asarray(data, np.float64).reshape(data.shape[0], -1)
    sums = (reads.weights[K] @ flat).reshape(n_v, K * flat.shape[1])
    M = columns.members
    col_sum = np.asarray(M @ sums).reshape(-1, K, flat.shape[1])
    vol = np.asarray(M @ reads.volume[K])
    vol_all = np.asarray(M @ reads.volume_all[K])
    values = np.divide(col_sum, vol[..., None], out=np.full_like(col_sum, np.nan),
                       where=vol[..., None] > 0)  # fmt: skip
    coverage = np.divide(vol, vol_all, out=np.zeros_like(vol), where=vol_all > 0)
    psf = None
    if K in reads.mixed:
        mixed = np.asarray(M @ reads.mixed[K].reshape(n_v, K * K)).reshape(-1, K, K)
        total = np.asarray(M @ reads.total[K])
        psf = np.divide(mixed, total[..., None], out=np.full_like(mixed, np.nan),
                        where=total[..., None] > 0)  # fmt: skip
        sm = psf.sum(axis=2, keepdims=True)
        psf = np.divide(psf, sm, out=np.full_like(psf, np.nan), where=sm > 0)
    return ColumnCurves(K, values.reshape(-1, K, *trailing), coverage, psf)


__all__ = [
    "ColumnCurves",
    "ColumnSet",
    "VertexDepthReads",
    "column_curves",
    "make_columns",
    "responsive_vertices",
    "vertex_depth_reads",
]
