"""AFNI-faithful local Pearson cost (LPC / LPA) via blok tiling.

Ports 3dAllineate's local-Pearson machinery:
  - blok geometry from ``create_GA_BLOK_set`` (mri_genalign_util.c)
  - per-blok correlation aggregation from ``GA_pearson_local`` (mri_genalign.c)

Space is tiled into space-filling polyhedra (RHDD / TOHD / CUBE — the
Voronoi cells of the FCC / BCC / cubic lattices).  For each blok we compute a
weighted Pearson ``r``, stretch it with a Fisher transform
``p = log((1+r)/(1-r)) = 2*atanh(r)`` (clamped at |r| <= CMAX), and aggregate

    psum = sum_blok( ws * p * |p|^ppow )      wss = sum_blok( ws )
    value = 0.25 * psum / wss

where ``ws`` is the summed weight inside the blok.  This matches AFNI exactly:
  - LPC  = value                (signed; more negative == better cross-modal)
  - LPA  = 1 - |value|          (minimised; rewards strong |correlation|)

AFNI minimises its cost; ffs maximises, so the convention helpers return the
ffs (higher == better) values:
  - lpc -> -value
  - lpa -> |value|

The blok lattice and the per-voxel blok assignment depend only on the base
grid, so :func:`assign_bloks` is computed once and reused for every cost
evaluation.  The eval (:func:`local_pearson`) is a handful of ``index_add``
scatter-reductions and is differentiable in the warped source.

One deliberate deviation from AFNI: where a point falls inside more than one
blok (rare, only near boundaries when the shrink factor != 1) AFNI counts it in
each; we assign it to the single nearest blok centre (a clean Voronoi
partition).  With the default space-filling shapes this affects a negligible
fraction of voxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Max correlation allowed before the Fisher stretch blows up (mri_genalign.c).
_CMAX = 0.9999

# volume(blok) / siz**3 for each shape — used by the auto-radius formula
# (GA_BLOK_VOLFAC + the inside-test comments in mri_genalign_util.c).
_VOLFAC = {"ball": 4.18879, "cube": 8.0, "rhdd": 2.0, "tohd": 4.0}

# Minimum points in a blok for it to contribute (MINCOR in GA_pearson_local).
_MIN_BLOK_PTS = 9


def auto_blok_radius(voxdims: tuple[float, float, float], bloktype: str = "rhdd") -> float:
    """Blok radius giving ~555 base voxels per blok (3dAllineate.c).

    ``voxdims`` is (dx, dy, dz) in mm.  Mirrors
    ``blokrad = cbrt(555 * voxvol / VOLFAC)``.
    """
    dx, dy, dz = (abs(float(v)) or 1.0 for v in voxdims)
    voxvol = dx * dy * dz
    return float((555.0 * voxvol / _VOLFAC[bloktype]) ** (1.0 / 3.0))


def _lattice(bloktype: str, blokrad: float) -> tuple[list[list[float]], float]:
    """Return (lattice basis columns, siz) for a blok type (shfac == 1).

    The 3x3 matrix's *columns* are the lattice vectors that translate a blok
    centre by integer (p, q, r); ``siz`` parametrises the inside test.
    """
    a = float(blokrad)
    if bloktype == "cube":
        lat = [[2 * a, 0.0, 0.0], [0.0, 2 * a, 0.0], [0.0, 0.0, 2 * a]]
        siz = a
    elif bloktype == "rhdd":
        lat = [[a, 0.0, a], [a, a, 0.0], [0.0, a, a]]
        siz = a
    elif bloktype == "tohd":
        lat = [[-a, a, a], [a, -a, a], [a, a, -a]]
        siz = a
    else:
        raise ValueError(f"Unsupported blok type for tiling: {bloktype!r}")
    return lat, siz


def _inside(off: Tensor, bloktype: str, siz: float) -> Tensor:
    """Boolean test: is offset (u, v, w) inside the blok polyhedron?

    ``off`` is (..., 3); returns bool of shape (...).  Ports the
    ``GA_BLOK_inside_*`` macros (FAS(a, s) == |a| <= s).
    """
    u, v, w = off[..., 0], off[..., 1], off[..., 2]
    au, av, aw = u.abs(), v.abs(), w.abs()
    if bloktype == "cube":
        return (au <= siz) & (av <= siz) & (aw <= siz)
    if bloktype == "rhdd":
        return (
            ((u + v).abs() <= siz)
            & ((u - v).abs() <= siz)
            & ((u + w).abs() <= siz)
            & ((u - w).abs() <= siz)
            & ((v + w).abs() <= siz)
            & ((v - w).abs() <= siz)
        )
    # tohd
    s15 = 1.5 * siz
    return (
        (au <= siz)
        & (av <= siz)
        & (aw <= siz)
        & ((u + v + w).abs() <= s15)
        & ((u - v + w).abs() <= s15)
        & ((u + v - w).abs() <= s15)
        & ((u - v - w).abs() <= s15)
    )


@dataclass
class BlokSet:
    """Per-voxel blok assignment for one base grid (computed once)."""

    index: Tensor  # (N,) long; -1 for excluded voxels (masked / sparse blok)
    nblok: int
    bloktype: str
    blokrad: float
    n_populated: int = 0  # distinct bloks that survived pruning (for overlap)


def assign_bloks(
    shape: tuple[int, int, int],
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    bloktype: str = "rhdd",
    blokrad: float | None = None,
    mask: Tensor | None = None,
    device: torch.device | None = None,
) -> BlokSet:
    """Assign every voxel of a (nz, ny, nx) grid to a space-filling blok.

    Args:
        shape: (nz, ny, nx) base grid.
        voxdims: (dx, dy, dz) voxel sizes in mm (x is the fastest axis).
        bloktype: "rhdd" (default), "tohd", or "cube".
        blokrad: blok radius in mm; ``None`` auto-sizes to ~555 voxels.
        mask: optional (nz, ny, nx) tensor; voxels where ``mask <= 0`` are
            excluded and do not count toward the sparse-blok threshold.
        device: torch device (defaults to mask's / cpu).

    Returns:
        :class:`BlokSet` with a flat (N,) blok index per voxel (row-major,
        matching ``vol.reshape(-1)``); excluded voxels carry index -1.
    """
    nz, ny, nx = shape
    dx, dy, dz = (float(v) for v in voxdims)
    if device is None:
        device = mask.device if mask is not None else torch.device("cpu")
    if blokrad is None:
        blokrad = auto_blok_radius((dx, dy, dz), bloktype)

    lat_list, siz = _lattice(bloktype, blokrad)
    lat = torch.tensor(lat_list, dtype=torch.float64, device=device)  # (3,3)
    invlat = torch.linalg.inv(lat)

    # Physical coordinates of every voxel, in reshape(-1) (z, y, x) order.
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64, device=device),
        torch.arange(ny, dtype=torch.float64, device=device),
        torch.arange(nx, dtype=torch.float64, device=device),
        indexing="ij",
    )
    xyz = torch.stack(
        [(ii * dx).reshape(-1), (jj * dy).reshape(-1), (kk * dz).reshape(-1)],
        dim=1,
    )  # (N, 3)

    # Nearest integer lattice index (AFNI: floor(pqr + 0.499)).
    pqr = xyz @ invlat.T  # (N, 3)
    base_lat = torch.floor(pqr + 0.499)  # (N, 3) float

    # Search the 3x3x3 neighbourhood; keep the nearest centre whose polyhedron
    # actually contains the voxel (a clean Voronoi partition).
    best_d2 = torch.full((xyz.shape[0],), float("inf"), dtype=torch.float64, device=device)
    best_lat = base_lat.clone()
    found = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
    for dp in (-1, 0, 1):
        for dq in (-1, 0, 1):
            for dr in (-1, 0, 1):
                cand = base_lat + torch.tensor([dp, dq, dr], dtype=torch.float64, device=device)
                centre = cand @ lat.T  # (N, 3)
                off = xyz - centre
                ins = _inside(off, bloktype, siz)
                d2 = (off * off).sum(dim=1)
                take = ins & (d2 < best_d2)
                best_d2 = torch.where(take, d2, best_d2)
                best_lat = torch.where(take[:, None], cand, best_lat)
                found = found | ins

    # Map chosen lattice triples to contiguous blok ids.
    chosen = best_lat.round().to(torch.int64)  # (N, 3)
    _, inverse = torch.unique(chosen, dim=0, return_inverse=True)
    index = inverse.clone()
    index[~found] = -1
    if mask is not None:
        index[mask.reshape(-1) <= 0] = -1

    nblok = int(inverse.max().item()) + 1 if inverse.numel() else 0

    # Count matching points per blok and drop sparse bloks (AFNI minel rule:
    # minel = 0.456 * max_count + 1 when blokmin < 9).
    valid = index >= 0
    if valid.any():
        counts = torch.zeros(nblok, dtype=torch.int64, device=device)
        counts.index_add_(
            0, index[valid], torch.ones(int(valid.sum()), dtype=torch.int64, device=device)
        )
        minel = int(0.456 * int(counts.max().item())) + 1
        minel = max(minel, _MIN_BLOK_PTS)
        sparse = counts < minel  # (nblok,)
        drop = sparse[index.clamp(min=0)] & valid
        index[drop] = -1

    n_populated = int(torch.unique(index[index >= 0]).numel()) if (index >= 0).any() else 0
    return BlokSet(
        index=index, nblok=nblok, bloktype=bloktype, blokrad=float(blokrad), n_populated=n_populated
    )


def local_pearson_value(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None,
    blokset: BlokSet,
    ppow: float = 1.0,
) -> Tensor:
    """AFNI's signed local-Pearson aggregate ``0.25 * sum(ws*p*|p|^ppow)/sum(ws)``.

    This is the raw value matching ``GA_pearson_local``: LPC uses it directly,
    LPA uses ``1 - |value|``.  Differentiable in ``warped``.

    Args:
        base, warped, weight: (nz, ny, nx) or flat (N,) tensors on one device.
            ``weight`` may be None (each blok then weighted equally, ws == 1).
        blokset: precomputed :class:`assign_bloks` result for the base grid.
        ppow: emphasis exponent on |p| (AFNI default 1.0).

    Returns:
        Scalar tensor (AFNI-convention signed value).
    """
    device = base.device
    idx = blokset.index.to(device)
    b = base.reshape(-1)
    y = warped.reshape(-1)
    valid = idx >= 0
    bi = idx[valid]
    bv = b[valid]
    yv = y[valid]
    nblok = blokset.nblok

    if weight is None:
        wv = torch.ones_like(bv)
    else:
        wv = weight.reshape(-1)[valid]

    def _seg_sum(vals: Tensor) -> Tensor:
        out = torch.zeros(nblok, dtype=vals.dtype, device=device)
        return out.index_add(0, bi, vals)

    sw = _seg_sum(wv)
    swx = _seg_sum(wv * bv)
    swy = _seg_sum(wv * yv)
    swxx = _seg_sum(wv * bv * bv)
    swyy = _seg_sum(wv * yv * yv)
    swxy = _seg_sum(wv * bv * yv)
    cnt = _seg_sum(torch.ones_like(wv))

    sw_safe = sw.clamp(min=1e-12)
    xv_ = swxx - swx * swx / sw_safe
    yv_ = swyy - swy * swy / sw_safe
    xy_ = swxy - swx * swy / sw_safe

    ok = (cnt >= _MIN_BLOK_PTS) & (xv_ > 0) & (yv_ > 0) & (sw > 0)

    denom = (xv_ * yv_).clamp(min=1e-24).sqrt()
    pcor = (xy_ / denom).clamp(-_CMAX, _CMAX)
    # Fisher stretch: log((1+r)/(1-r)) == 2*atanh(r).
    p = torch.log((1.0 + pcor) / (1.0 - pcor))
    pabs = p.abs() if ppow == 1.0 else p.abs().pow(ppow)

    # blok weight ws: summed weights (weighted) or 1 per blok (unweighted)
    ws = torch.ones_like(sw) if weight is None else sw
    contrib = torch.where(ok, ws * p * pabs, torch.zeros_like(p))
    wsum = torch.where(ok, ws, torch.zeros_like(ws)).sum()

    if wsum <= 0:
        return torch.zeros((), device=device, dtype=base.dtype)
    agg = 0.25 * contrib.sum() / wsum
    return agg * _overlap_factor(ok, blokset)


def _overlap_factor(ok: Tensor, blokset: BlokSet) -> Tensor:
    """Squared fraction of populated bloks that contributed (detached gate).

    Scaling the aggregate by overlap**2 strongly penalises low-overlap
    configurations (a brain shifted partly out of the FOV), so the coarse
    search can't drift to a far position that manufactures a spuriously strong
    correlation from the few bloks that still overlap. Squared (rather than
    linear) because a partial overlap with high spurious local correlation can
    otherwise still out-score the true, fully-overlapping alignment.
    """
    if blokset.n_populated <= 0:
        return torch.ones((), device=ok.device)
    frac = (ok.sum(dim=-1).float() / blokset.n_populated).clamp(max=1.0)
    return (frac * frac).detach()


def local_pearson_value_batched(
    base: Tensor,
    warped_batch: Tensor,
    weight: Tensor | None,
    blokset: BlokSet,
    ppow: float = 1.0,
) -> Tensor:
    """Batched :func:`local_pearson_value` over B warped volumes -> (B,).

    The base-only per-blok sums are computed once; only the warped-dependent
    sums are batched, so this is far cheaper than a Python loop for the broad
    coarse search.
    """
    device = base.device
    idx = blokset.index.to(device)
    valid = idx >= 0
    bi = idx[valid]
    b = base.reshape(-1)[valid]
    B = warped_batch.shape[0]
    yb = warped_batch.reshape(B, -1)[:, valid]  # (B, Nv)
    w = torch.ones_like(b) if weight is None else weight.reshape(-1)[valid]
    nblok = blokset.nblok

    def _seg(vals):
        out = torch.zeros(nblok, dtype=vals.dtype, device=device)
        return out.index_add(0, bi, vals)

    sw = _seg(w)
    swx = _seg(w * b)
    swxx = _seg(w * b * b)
    cnt = _seg(torch.ones_like(w))

    wy = w[None, :] * yb  # (B, Nv)

    def _segB(vals):
        out = torch.zeros(B, nblok, dtype=vals.dtype, device=device)
        return out.index_add(1, bi, vals)

    swy = _segB(wy)
    swyy = _segB(wy * yb)
    swxy = _segB(wy * b[None, :])

    sw_safe = sw.clamp(min=1e-12)
    xv = swxx - swx * swx / sw_safe  # (nblok,)
    yv = swyy - swy * swy / sw_safe[None, :]  # (B, nblok)
    xy = swxy - swy * swx[None, :] / sw_safe[None, :]

    ok = (cnt[None, :] >= _MIN_BLOK_PTS) & (xv[None, :] > 0) & (yv > 0) & (sw[None, :] > 0)
    denom = (xv[None, :] * yv).clamp(min=1e-24).sqrt()
    pcor = (xy / denom).clamp(-_CMAX, _CMAX)
    p = torch.log((1.0 + pcor) / (1.0 - pcor))
    pabs = p.abs() if ppow == 1.0 else p.abs().pow(ppow)

    ws = (torch.ones_like(sw) if weight is None else sw)[None, :]
    contrib = torch.where(ok, ws * p * pabs, torch.zeros_like(p))
    wsum = torch.where(ok, ws.expand(B, -1), torch.zeros_like(p)).sum(dim=1)
    agg = 0.25 * contrib.sum(dim=1) / wsum.clamp(min=1e-12)
    agg = agg * _overlap_factor(ok, blokset)
    agg = torch.where(wsum > 0, agg, torch.zeros_like(agg))
    return agg


@dataclass
class BlokPairs:
    """Compact (patch, blok) binning for batched per-patch local Pearson.

    Depends only on the per-patch blok assignment (constant while a patch set is
    optimized), so it is built once per checkerboard phase and reused across the
    optimizer's iterations. ``torch.unique`` collapses the global blok ids each
    patch touches (a handful) into a dense bin space, so memory is O(occupied
    (patch, blok) pairs) rather than O(B * nblok_global).
    """

    inv: Tensor      # (Nvalid,) valid-voxel -> compact bin id
    bin_row: Tensor  # (P,) compact bin -> patch index
    valid: Tensor    # (B*V,) bool, voxels assigned to a real blok
    n_bins: int
    batch: int


def prepare_blok_pairs(blok_idx: Tensor, nblok: int) -> BlokPairs:
    """Precompute the compact binning for :func:`local_pearson_value_pairs`.

    Args:
        blok_idx: (B, V) long; global blok id per patch voxel, <0 to exclude.
        nblok: total number of bloks in the global lattice.
    """
    B, V = blok_idx.shape
    device = blok_idx.device
    valid = (blok_idx >= 0).reshape(-1)
    row = torch.arange(B, device=device)[:, None].expand(B, V).reshape(-1)
    gid = (row * nblok + blok_idx.reshape(-1))[valid]  # unique per (patch, blok)
    uniq, inv = torch.unique(gid, return_inverse=True)
    bin_row = torch.div(uniq, nblok, rounding_mode="floor")
    return BlokPairs(inv=inv, bin_row=bin_row, valid=valid, n_bins=int(uniq.numel()), batch=B)


def local_pearson_value_pairs(
    base: Tensor, warped: Tensor, weight: Tensor, prep: BlokPairs, ppow: float = 1.0,
) -> Tensor:
    """Local-Pearson value for B *independent* (base, warped) pairs -> (B,).

    Unlike :func:`local_pearson_value_batched` (one shared base, B warped
    candidates -- 3dAllineate's coarse search), every batch element here is its
    own base+warped pair sharing the per-voxel blok assignment in ``prep``. This
    is what 3dQwarp needs: B overlapping patches, each scored over the global
    blok-lattice voxels that fall inside it.

    The per-blok aggregation matches AFNI's GA_pearson_local (Fisher-z log
    transform, ppow weighting, MINCOR points/blok). The FOV-overlap guard from
    the allineate path is intentionally omitted -- 3dQwarp does not use it.
    Differentiable through ``warped``.
    """
    device = warped.device
    v = prep.valid
    x = base.reshape(-1)[v]
    y = warped.reshape(-1)[v]
    w = weight.reshape(-1)[v]
    inv, P = prep.inv, prep.n_bins

    def _seg(vals: Tensor) -> Tensor:
        return torch.zeros(P, dtype=vals.dtype, device=device).index_add(0, inv, vals)

    sw = _seg(w)
    swx = _seg(w * x)
    swxx = _seg(w * x * x)
    swy = _seg(w * y)
    swyy = _seg(w * y * y)
    swxy = _seg(w * x * y)
    cnt = _seg(torch.ones_like(w))

    sw_safe = sw.clamp(min=1e-12)
    xv = swxx - swx * swx / sw_safe
    yv = swyy - swy * swy / sw_safe
    xy = swxy - swx * swy / sw_safe

    ok = (cnt >= _MIN_BLOK_PTS) & (xv > 0) & (yv > 0) & (sw > 0)
    denom = (xv * yv).clamp(min=1e-24).sqrt()
    pcor = (xy / denom).clamp(-_CMAX, _CMAX)
    p = torch.log((1.0 + pcor) / (1.0 - pcor))
    pabs = p.abs() if ppow == 1.0 else p.abs().pow(ppow)

    # Per-blok contributions, then reduce each blok into its patch.
    contrib = torch.where(ok, sw * p * pabs, torch.zeros_like(p))
    wcon = torch.where(ok, sw, torch.zeros_like(sw))
    num = torch.zeros(prep.batch, dtype=contrib.dtype, device=device).index_add(0, prep.bin_row, contrib)
    den = torch.zeros(prep.batch, dtype=wcon.dtype, device=device).index_add(0, prep.bin_row, wcon)
    agg = 0.25 * num / den.clamp(min=1e-12)
    return torch.where(den > 0, agg, torch.zeros_like(agg))


def lpc_value_pairs(base, warped, weight, prep, ppow=1.0) -> Tensor:
    """Per-patch LPC (higher == better): ``-value``."""
    return -local_pearson_value_pairs(base, warped, weight, prep, ppow)


def lpa_value_pairs(base, warped, weight, prep, ppow=1.0) -> Tensor:
    """Per-patch LPA (higher == better): ``|value|``."""
    return local_pearson_value_pairs(base, warped, weight, prep, ppow).abs()


def lpc_cost_batched(base, warped_batch, weight, blokset, ppow=1.0) -> Tensor:
    """Batched LPC (higher == better): ``-value`` per warped volume."""
    return -local_pearson_value_batched(base, warped_batch, weight, blokset, ppow)


def lpa_cost_batched(base, warped_batch, weight, blokset, ppow=1.0) -> Tensor:
    """Batched LPA (higher == better): ``|value|`` per warped volume."""
    return local_pearson_value_batched(base, warped_batch, weight, blokset, ppow).abs()


def lpc_cost(base, warped, weight, blokset, ppow: float = 1.0) -> Tensor:
    """LPC in ffs (higher == better) convention: ``-value``."""
    return -local_pearson_value(base, warped, weight, blokset, ppow)


def lpa_cost(base, warped, weight, blokset, ppow: float = 1.0) -> Tensor:
    """LPA in ffs (higher == better) convention: ``|value|``."""
    return local_pearson_value(base, warped, weight, blokset, ppow).abs()
