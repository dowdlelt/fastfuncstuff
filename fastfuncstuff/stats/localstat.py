"""Local spatial statistics on a per-voxel neighborhood (GPU-accelerated).

This module is the GPU re-implementation of AFNI's ``3dLocalstat`` family.
Implemented statistics:

* ``local_acf`` -- the local spatial **AutoCorrelation Function** model, i.e.
  AFNI's ``3dLocalACF`` (the heavy one; see the GPU-parallel notes below).
* ``local_fwhm`` -- local image **smoothness** (FWHM) via AFNI's Forman
  finite-difference estimator, i.e. ``3dLocalstat -stat fwhm``.

Both share the neighborhood machinery (``build_neighborhood``,
``_neighbor_shift``) and the bounds-aware accumulation pattern, so adding a new
``3dLocalstat`` statistic mostly means writing a new per-neighborhood reduction.

The ACF is the interesting one. AFNI's ``3dLocalACF`` walks every voxel,
extracts the neighborhood time series, correlates the center with each neighbor,
bins the correlations by radius, then runs a per-voxel NEWUOA fit of the model

    ACF(r) = a * exp(-r^2 / (2 b^2)) + (1 - a) * exp(-r / c)

That is "very slow" (its own help text says so).  The key observation that makes
it embarrassingly GPU-parallel is that the neighborhood *offsets* -- and hence
the radius of every neighbor and the 1%-radius bins AFNI collapses into -- are
**identical for every voxel**.  So instead of a per-voxel extract/sort/collapse:

1. For each fixed offset we compute a whole-volume cosine-similarity map
   (time-dot / norms, bounds-aware, never wrapping).
2. We average those maps into the fixed radius bins, giving every voxel a
   ``(radius, corr)`` curve at once.
3. We fit ``a, b, c`` to the ACF model for all voxels simultaneously with a
   batched, bounded Levenberg-Marquardt solver (float64 normal equations).
4. FWHM / FWQM come from batched bisection of the fitted model.
5. AFNI's small 19-voxel median post-filter is applied to each output map.

See ``../fmri_wiki`` for the ACF model rationale.  Faithfulness notes vs AFNI
are inline where the two differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.memory import get_available_memory

# ACF model is fit with at least this many distinct radius bins present; below
# this a 3-parameter fit is meaningless.  AFNI requires >= 10 surviving
# neighbors (``if(pp<9) return``); we keep the same spirit on the binned curve.
_MIN_BINS = 6


# ---------------------------------------------------------------------------
# Neighborhood construction (faithful to AFNI edt_buildmask.c)
# ---------------------------------------------------------------------------


@dataclass
class Neighborhood:
    """Fixed set of neighbor offsets shared by every voxel.

    Attributes
    ----------
    offsets : (M, 3) int tensor
        Offsets in tensor-axis order ``(dz, dy, dx)`` (our volumes are
        ``(nz, ny, nx)``).  Includes the central ``(0, 0, 0)`` offset.
    radii : (M,) float tensor
        ACF radius of each offset in mm, always computed with the true voxel
        dimensions (AFNI computes the *inclusion* test in voxel units when the
        radius is negative, but the ACF radius always uses real voxel sizes).
    bin_id : (M,) long tensor
        Index of the radius bin each offset belongs to (AFNI's 1%-radius
        collapse, precomputed once on the global sorted radius list).
    bin_radius : (K,) float tensor
        Representative radius of each bin (the smallest radius in the bin, as
        AFNI keeps ``rrar[pp]`` of the first element after averaging).
    """

    offsets: Tensor
    radii: Tensor
    bin_id: Tensor
    bin_radius: Tensor


def _parse_nbhd(nbhd: str) -> tuple[str, list[float]]:
    """Parse an AFNI ``-nbhd`` string like ``SPHERE(25)`` or ``RECT(2,2,0)``."""
    m = re.match(r"\s*([A-Za-z]+)\s*\(([^)]*)\)\s*$", nbhd)
    if not m:
        raise ValueError(
            f"Cannot parse -nbhd '{nbhd}'. Expected e.g. SPHERE(25), RECT(a,b,c), RHDD(r), TOHD(r)."
        )
    kind = m.group(1).upper()
    nums = [float(x) for x in re.split(r"[,\s]+", m.group(2).strip()) if x != ""]
    return kind, nums


def _build_offsets_xyz(
    kind: str, nums: list[float], dx: float, dy: float, dz: float
) -> list[tuple[int, int, int]]:
    """Replicate AFNI MCW_*mask: return (i, j, k) offsets in (x, y, z) order.

    A negative size means "in voxel index units" (inclusion uses unit spacing),
    exactly as AFNI's ``if (na < 0) dx = dy = dz = 1``.
    """
    adx, ady, adz = abs(dx), abs(dy), abs(dz)

    def inc_dims(neg: bool) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0) if neg else (adx, ady, adz)

    pts: list[tuple[int, int, int]] = [(0, 0, 0)]  # center always first

    if kind == "SPHERE":
        if not nums:
            raise ValueError("SPHERE needs a radius")
        r = nums[0]
        neg = r < 0
        r = abs(r)
        ex, ey, ez = inc_dims(neg)
        # MCW_build_mask: idx = max_dist/dx (truncated), include if 0 < d^2 <= r^2
        idx, jdy, kdz = int(r / ex), int(r / ey), int(r / ez)
        rq = r * r
        for k in range(-kdz, kdz + 1):
            zq = (k * ez) ** 2
            for j in range(-jdy, jdy + 1):
                yq = zq + (j * ey) ** 2
                for i in range(-idx, idx + 1):
                    xq = yq + (i * ex) ** 2
                    if 0.0 < xq <= rq:
                        pts.append((i, j, k))

    elif kind == "RECT":
        if len(nums) < 3:
            raise ValueError("RECT needs three half-widths a,b,c")
        a, b, c = nums[0], nums[1], nums[2]
        ex = 1.0 if a < 0 else adx
        ey = 1.0 if b < 0 else ady
        ez = 1.0 if c < 0 else adz
        idx, jdy, kdz = int(abs(a) / ex), int(abs(b) / ey), int(abs(c) / ez)
        for k in range(-kdz, kdz + 1):
            for j in range(-jdy, jdy + 1):
                for i in range(-idx, idx + 1):
                    if i or j or k:
                        pts.append((i, j, k))

    elif kind == "RHDD":
        if not nums:
            raise ValueError("RHDD needs a radius")
        r = nums[0]
        neg = r <= 0
        r = 1.01 if r <= 0 else r
        ex, ey, ez = inc_dims(neg)
        idx, jdy, kdz = int(r / ex), int(r / ey), int(r / ez)
        for k in range(-kdz, kdz + 1):
            cc = k * ez
            for j in range(-jdy, jdy + 1):
                bb = j * ey
                for i in range(-idx, idx + 1):
                    if not (i or j or k):
                        continue
                    aa = i * ex
                    if (
                        abs(aa + bb) <= r
                        and abs(aa - bb) <= r
                        and abs(aa + cc) <= r
                        and abs(aa - cc) <= r
                        and abs(bb + cc) <= r
                        and abs(bb - cc) <= r
                    ):
                        pts.append((i, j, k))

    elif kind == "TOHD":
        if not nums:
            raise ValueError("TOHD needs a radius")
        r = nums[0]
        neg = r <= 0
        r = 1.01 if r <= 0 else r
        ex, ey, ez = inc_dims(neg)
        idx, jdy, kdz = int(r / ex), int(r / ey), int(r / ez)

        def fas(v: float, s: float) -> bool:
            return abs(v) <= s

        for k in range(-kdz, kdz + 1):
            cc = k * ez
            for j in range(-jdy, jdy + 1):
                bb = j * ey
                for i in range(-idx, idx + 1):
                    if not (i or j or k):
                        continue
                    aa = i * ex
                    if (
                        fas(aa, r)
                        and fas(bb, r)
                        and fas(cc, r)
                        and fas(aa + bb + cc, 1.5 * r)
                        and fas(aa - bb + cc, 1.5 * r)
                        and fas(aa + bb - cc, 1.5 * r)
                        and fas(aa - bb - cc, 1.5 * r)
                    ):
                        pts.append((i, j, k))
    else:
        raise ValueError(f"Unknown -nbhd shape '{kind}'")

    return pts


def build_neighborhood(nbhd: str, voxdims: tuple[float, float, float]) -> Neighborhood:
    """Build the shared neighborhood + radius bins from an AFNI ``-nbhd`` string.

    ``voxdims`` is ``(dx, dy, dz)`` in mm (x, y, z order, as in NIfTI zooms).
    Offsets are returned in tensor-axis order ``(dz, dy, dx)``.
    """
    dx, dy, dz = voxdims
    adx, ady, adz = abs(dx), abs(dy), abs(dz)
    kind, nums = _parse_nbhd(nbhd)
    pts_xyz = _build_offsets_xyz(kind, nums, dx, dy, dz)

    # Map (i, j, k) = (x, y, z) -> tensor (dz, dy, dx); ACF radius uses real dims.
    offs = []
    rads = []
    for i, j, k in pts_xyz:
        offs.append((k, j, i))
        rads.append(np.sqrt((adx * i) ** 2 + (ady * j) ** 2 + (adz * k) ** 2))
    offs_arr = np.asarray(offs, dtype=np.int64)
    rads_arr = np.asarray(rads, dtype=np.float64)

    # Sort by ascending radius (center, r=0, ends up first).
    order = np.argsort(rads_arr, kind="stable")
    offs_arr = offs_arr[order]
    rads_arr = rads_arr[order]

    # Greedy 1%-radius collapse, faithful to ACF_nbhd_vec_to_modelE: starting at
    # the first non-center entry, all subsequent radii <= 1.01 * (group start)
    # share a bin; the bin's radius is the group-start (smallest) radius.
    bin_id = np.zeros(len(rads_arr), dtype=np.int64)
    bin_radius: list[float] = [0.0]  # bin 0 is the center (r = 0)
    cur_bin = 0
    m = 1
    while m < len(rads_arr):
        start_r = rads_arr[m]
        cur_bin += 1
        bin_radius.append(float(start_r))
        thresh = start_r * 1.01
        while m < len(rads_arr) and rads_arr[m] <= thresh:
            bin_id[m] = cur_bin
            m += 1

    return Neighborhood(
        offsets=torch.from_numpy(offs_arr),
        radii=torch.from_numpy(rads_arr),
        bin_id=torch.from_numpy(bin_id),
        bin_radius=torch.tensor(bin_radius, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Bin accumulation: per-offset cosine-similarity maps -> radius-binned curves
# ---------------------------------------------------------------------------


def _accumulate_bins(
    x_sub: Tensor,
    nrm_sub: Tensor,
    mask_sub: Tensor,
    nbhd: Neighborhood,
    z_sub_lo: int,
    out_z_lo: int,
    out_z_hi: int,
    n_bins: int,
    progress: tqdm | None,
) -> tuple[Tensor, Tensor]:
    """Accumulate radius-binned correlation curves for an output z-slab.

    ``x_sub`` is the device sub-volume ``(nt, nz_sub, ny, nx)`` covering global z
    in ``[z_sub_lo, z_sub_lo + nz_sub)``; it must include the halo needed by the
    neighborhood.  Output planes are global z in ``[out_z_lo, out_z_hi)``.

    Returns ``(bin_sum, bin_cnt)`` of shape ``(K, slab, ny, nx)`` -- the summed
    correlations and counts per bin for the slab's output planes.
    """
    nz_sub, ny, nx = x_sub.shape[1:]
    slab = out_z_hi - out_z_lo
    device = x_sub.device

    bin_sum = torch.zeros((n_bins, slab, ny, nx), dtype=torch.float32, device=device)
    bin_cnt = torch.zeros((n_bins, slab, ny, nx), dtype=torch.float32, device=device)

    offsets = nbhd.offsets.tolist()
    bin_ids = nbhd.bin_id.tolist()

    # The sub-volume's global upper z bound; its lower bound is z_sub_lo. Callers
    # pass a sub-volume covering [out_z_lo, out_z_hi) plus halo, so a center z in
    # the output range with an in-bounds shift always maps into x_sub.
    nz_global = z_sub_lo + nz_sub

    for (oz, oy, ox), b in zip(offsets, bin_ids, strict=True):
        # Center z-range (global) that (a) is an output plane, (b) keeps the
        # shifted index in-bounds of the *full* volume.
        cz0 = max(out_z_lo, max(0, -oz))
        cz1 = min(out_z_hi, nz_global if oz <= 0 else nz_global - oz)
        # y / x are full-extent on the sub-volume.
        cy0, cy1 = max(0, -oy), min(ny, ny - oy)
        cx0, cx1 = max(0, -ox), min(nx, nx - ox)
        if cz0 >= cz1 or cy0 >= cy1 or cx0 >= cx1:
            if progress is not None:
                progress.update(1)
            continue

        # Local indices into x_sub (z offset by z_sub_lo) for center & shifted.
        czc0, czc1 = cz0 - z_sub_lo, cz1 - z_sub_lo
        czs0, czs1 = cz0 + oz - z_sub_lo, cz1 + oz - z_sub_lo

        center = x_sub[:, czc0:czc1, cy0:cy1, cx0:cx1]
        shifted = x_sub[:, czs0:czs1, cy0 + oy : cy1 + oy, cx0 + ox : cx1 + ox]
        dot = (center * shifted).sum(0)  # (region z, y, x)

        nrm_c = nrm_sub[czc0:czc1, cy0:cy1, cx0:cx1]
        nrm_s = nrm_sub[czs0:czs1, cy0 + oy : cy1 + oy, cx0 + ox : cx1 + ox]
        msk_c = mask_sub[czc0:czc1, cy0:cy1, cx0:cx1]
        msk_s = mask_sub[czs0:czs1, cy0 + oy : cy1 + oy, cx0 + ox : cx1 + ox]

        denom = nrm_c * nrm_s
        valid = (denom > 0) & msk_c & msk_s
        corr = torch.where(valid, dot / denom.clamp_min(1e-20), torch.zeros_like(dot))
        validf = valid.to(torch.float32)

        # Place into slab-local output coordinates (z relative to out_z_lo).
        tz0, tz1 = cz0 - out_z_lo, cz1 - out_z_lo
        bin_sum[b, tz0:tz1, cy0:cy1, cx0:cx1] += corr * validf
        bin_cnt[b, tz0:tz1, cy0:cy1, cx0:cx1] += validf

        if progress is not None:
            progress.update(1)

    return bin_sum, bin_cnt


# ---------------------------------------------------------------------------
# Batched ACF model fit (Levenberg-Marquardt) + FWHM/FWQM
# ---------------------------------------------------------------------------


def _acf_model(r: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """ACF(r) = a exp(-r^2/2b^2) + (1-a) exp(-r/c).  Broadcasts (V,1) x (1,K)."""
    return a * torch.exp(-0.5 * r * r / (b * b)) + (1.0 - a) * torch.exp(-r / c)


# Multiplicative perturbations of the AFNI seed used as extra LM start points.
# The ACF cost has a near-flat (a, b)/(b, c) ridge in low-smoothness data, so a
# single start lands in a worse-than-global local minimum ~30% of the time. A
# few cheap restarts (kept per-voxel by lowest cost) reach the global minimum
# far more reliably; FWHM is what we actually consume and it tightens with it.
_LM_STARTS = (
    (0.5, 1.0, 1.0),  # AFNI's own start
    (0.5, 2.0, 1.0),
    (0.5, 0.5, 1.0),
    (0.3, 1.0, 2.0),
    (0.7, 1.0, 0.5),
)


def fit_acf_batched(
    radii: Tensor,
    y: Tensor,
    w: Tensor,
    n_iter: int = 50,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Bounded, multi-start batched Levenberg-Marquardt fit of the ACF model.

    Parameters
    ----------
    radii : (K,) float64 tensor
        Bin radii (shared by all voxels).
    y : (V, K) float64 tensor
        Per-voxel binned correlation curve.
    w : (V, K) float64 tensor
        0/1 weight: 1 where the bin is present (count > 0) for that voxel.

    Returns
    -------
    a, b, c : (V,) float64 tensors
        Fitted parameters (lowest-cost over the restart starts).
    ok : (V,) bool tensor
        Whether the per-voxel curve had a usable first crossing below 0.5
        (AFNI returns an error/leaves zeros otherwise); the caller masks these.
    """
    device = y.device
    v = y.shape[0]
    r = radii.view(1, -1)  # (1, K)

    # --- AFNI seed: first present bin with corr <= 0.5 sets the length scale ---
    below = (y <= 0.5) & (w > 0)
    k_idx = torch.arange(y.shape[1], device=device).view(1, -1)
    big = y.shape[1] + 1
    first_below = torch.where(below, k_idx, torch.full_like(k_idx, big)).min(dim=1).values
    has_cross = first_below < big
    fb = first_below.clamp(max=y.shape[1] - 1)
    yk = y.gather(1, fb.view(-1, 1)).squeeze(1)
    rk = radii[fb]

    # apar0 = 1/log(yk) where 0<yk<1; otherwise fall back to the crossing radius
    # as a sensible length scale (AFNI uses a fixed voxel-diagonal here).
    safe = (yk > 0.0) & (yk < 1.0)
    apar0 = torch.where(safe, 1.0 / torch.log(yk.clamp(1e-6, 1 - 1e-6)), torch.full_like(yk, -1.0))
    b0 = torch.where(safe, torch.sqrt((-0.5 * apar0).clamp_min(1e-12)) * rk, rk).clamp_min(1e-3)
    c0 = torch.where(safe, (-apar0 * rk), rk).clamp_min(1e-3)

    # --- bounds (AFNI: a in [.006,.994], b,c in [.05,5.55] * seed) ---
    a_lo, a_hi = 0.006, 0.994
    b_lo, b_hi = (0.05 * b0).view(-1, 1), (5.55 * b0).view(-1, 1)
    c_lo, c_hi = (0.05 * c0).view(-1, 1), (5.55 * c0).view(-1, 1)
    b0c = b0.view(-1, 1)
    c0c = c0.view(-1, 1)

    def clamp(a_, b_, c_):
        return (
            a_.clamp(a_lo, a_hi),
            torch.max(torch.min(b_, b_hi), b_lo),
            torch.max(torch.min(c_, c_hi), c_lo),
        )

    def cost(a_, b_, c_):
        res = (_acf_model(r, a_, b_, c_) - y) * w
        return (res * res).sum(dim=1)

    def lm_run(a, b, c):
        """One bounded LM descent from (a, b, c); returns final params + cost."""
        a, b, c = clamp(a, b, c)
        lam = torch.full((v,), 1e-3, dtype=torch.float64, device=device)
        cur = cost(a, b, c)
        idx = torch.arange(3, device=device)
        for _ in range(n_iter):
            ex = torch.exp(-0.5 * r * r / (b * b))
            fx = torch.exp(-r / c)
            res = (a * ex + (1.0 - a) * fx - y) * w  # (V,K)

            ja = (ex - fx) * w
            jb = (a * ex * (r * r) / (b**3)) * w
            jc = ((1.0 - a) * fx * r / (c * c)) * w

            jtj = torch.empty((v, 3, 3), dtype=torch.float64, device=device)
            jtj[:, 0, 0] = (ja * ja).sum(1)
            jtj[:, 1, 1] = (jb * jb).sum(1)
            jtj[:, 2, 2] = (jc * jc).sum(1)
            jtj[:, 0, 1] = jtj[:, 1, 0] = (ja * jb).sum(1)
            jtj[:, 0, 2] = jtj[:, 2, 0] = (ja * jc).sum(1)
            jtj[:, 1, 2] = jtj[:, 2, 1] = (jb * jc).sum(1)
            g = torch.stack([(ja * res).sum(1), (jb * res).sum(1), (jc * res).sum(1)], dim=1)

            diag = torch.diagonal(jtj, dim1=1, dim2=2)
            aug = jtj.clone()
            aug[:, idx, idx] += lam.view(-1, 1) * diag.clamp_min(1e-12)

            try:
                delta = torch.linalg.solve(aug, -g.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                delta = torch.linalg.lstsq(aug, -g.unsqueeze(-1)).solution.squeeze(-1)

            na = (a.squeeze(1) + delta[:, 0]).view(-1, 1)
            nb = (b.squeeze(1) + delta[:, 1]).view(-1, 1)
            nc = (c.squeeze(1) + delta[:, 2]).view(-1, 1)
            na, nb, nc = clamp(na, nb, nc)
            new = cost(na, nb, nc)

            improved = new < cur
            imp = improved.view(-1, 1)
            a = torch.where(imp, na, a)
            b = torch.where(imp, nb, b)
            c = torch.where(imp, nc, c)
            cur = torch.where(improved, new, cur)
            lam = torch.where(improved, (lam * 0.4).clamp_min(1e-9), (lam * 2.5).clamp_max(1e9))
        return a, b, c, cur

    def start_params(sa, sb, sc):
        return torch.full((v, 1), sa, dtype=torch.float64, device=device), b0c * sb, c0c * sc

    # Multi-start: keep the lowest-cost descent per voxel. (A tolerance that
    # prefers the seed solution was tried to tame the c lower-tail drift -- it
    # regressed a/b/c/FWHM because each LM run itself slides c to the floor in
    # the flat region, so the seed solution is no better there. Strict argmin is
    # the right call; the low-c scatter is unobservable-tail noise that does not
    # affect FWHM/FWQM.)
    sa, sb, sc = _LM_STARTS[0]
    best_a, best_b, best_c, best_cost = lm_run(*start_params(sa, sb, sc))
    for sa, sb, sc in _LM_STARTS[1:]:
        a_f, b_f, c_f, cst = lm_run(*start_params(sa, sb, sc))
        take = (cst < best_cost).view(-1, 1)
        best_a = torch.where(take, a_f, best_a)
        best_b = torch.where(take, b_f, best_b)
        best_c = torch.where(take, c_f, best_c)
        best_cost = torch.minimum(best_cost, cst)

    return best_a.squeeze(1), best_b.squeeze(1), best_c.squeeze(1), has_cross


def acf_fwhm_batched(a: Tensor, b: Tensor, c: Tensor, level: float) -> Tensor:
    """Width at which ACF(r) crosses ``level`` (returns 2*r, i.e. full width).

    The ACF model is monotonically decreasing from ACF(0)=1, so a bisection on
    ``r`` is exact.  AFNI brackets at ``[0.0333,1]*(2b+c)`` and minimizes
    ``|fit-level|``; bisection to the root is equivalent and robust.
    """
    a = a.view(-1)
    b = b.view(-1)
    c = c.view(-1)

    def m(r):
        return a * torch.exp(-0.5 * r * r / (b * b)) + (1.0 - a) * torch.exp(-r / c)

    lo = torch.zeros_like(a)
    hi = (2.0 * b + c).clamp_min(1e-3)
    # Expand hi until model(hi) < level (handles wide ACFs).
    for _ in range(40):
        need = m(hi) >= level
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0, hi)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        high = m(mid) > level
        lo = torch.where(high, mid, lo)
        hi = torch.where(high, hi, mid)
    return 2.0 * (0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Shared neighbor-shift primitive (used by FWHM accumulation + median filter)
# ---------------------------------------------------------------------------


def _neighbor_shift(field: Tensor, oz: int, oy: int, ox: int) -> tuple[Tensor, Tensor]:
    """Gather ``field`` at a fixed offset, aligned back to the output grid.

    Returns ``(g, inb)`` where ``g[..., v] = field[..., v + (oz, oy, ox)]`` (0
    where the source is out of bounds) and ``inb`` is the ``(nz, ny, nx)`` bool
    mask of voxels whose shifted source is in bounds. Works for any leading
    dimensions (a time/stat axis is fine); only the trailing 3 axes shift.
    """
    nz, ny, nx = field.shape[-3:]
    g = torch.zeros_like(field)
    inb = torch.zeros((nz, ny, nx), dtype=torch.bool, device=field.device)
    zlo, zhi = max(0, -oz), min(nz, nz - oz)
    ylo, yhi = max(0, -oy), min(ny, ny - oy)
    xlo, xhi = max(0, -ox), min(nx, nx - ox)
    if zlo < zhi and ylo < yhi and xlo < xhi:
        g[..., zlo:zhi, ylo:yhi, xlo:xhi] = field[
            ..., zlo + oz : zhi + oz, ylo + oy : yhi + oy, xlo + ox : xhi + ox
        ]
        inb[zlo:zhi, ylo:yhi, xlo:xhi] = True
    return g, inb


# ---------------------------------------------------------------------------
# 19-voxel masked median post-filter (AFNI mri_medianfilter radius 1.444)
# ---------------------------------------------------------------------------

# Offsets with index-distance <= 1.444: center + 6 faces + 12 edges = 19.
_MEDIAN_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz * dz + dy * dy + dx * dx) <= 2  # 0,1,sqrt2 -> <=2; excludes corners (3)
]


def _median_filter19(vol: Tensor, mask: Tensor) -> Tensor:
    """Masked 19-voxel median of a single 3-D map (faithful to AFNI post-step)."""
    stacks = []
    valids = []
    maskf = mask.to(vol.dtype)
    for dz, dy, dx in _MEDIAN_OFFSETS:
        shifted, inb = _neighbor_shift(vol, dz, dy, dx)
        vmask, _ = _neighbor_shift(maskf, dz, dy, dx)
        stacks.append(shifted)
        valids.append((vmask > 0.5) & inb)
    s = torch.stack(stacks, 0)  # (19, nz,ny,nx)
    vv = torch.stack(valids, 0)
    # Median over valid neighbors: push invalid to +inf, take lower-median by
    # sorting and selecting the middle of the valid count.
    big = s.max().item() + 1.0 if s.numel() else 1.0
    s_masked = torch.where(vv, s, torch.full_like(s, big))
    s_sorted, _ = torch.sort(s_masked, dim=0)
    cnt = vv.sum(0).clamp_min(1)  # (nz,ny,nx)
    mid_idx = ((cnt - 1) // 2).long()
    out = torch.gather(s_sorted, 0, mid_idx.unsqueeze(0)).squeeze(0)
    return torch.where(mask, out, vol)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

ACF_LABELS = ("ACF:a", "ACF:b", "ACF:c", "ACF:FWHM", "ACF:FWQM")


def local_acf(
    data: Tensor,
    voxdims: tuple[float, float, float],
    nbhd: str = "SPHERE(-9.666)",
    mask: Tensor | None = None,
    device: torch.device | None = None,
    do_median: bool = True,
    lm_iters: int = 50,
    verbose: int = 1,
) -> Tensor:
    """Estimate the local spatial ACF model at every voxel (GPU, AFNI-faithful).

    Parameters
    ----------
    data : (nt, nz, ny, nx) tensor
        Time-series volume.  Should already be detrended/despiked (AFNI expects
        an ``errts``-like input).
    voxdims : (dx, dy, dz)
        Voxel sizes in mm (x, y, z order).
    nbhd : str
        AFNI ``-nbhd`` string.  Default matches ``3dLocalACF``'s SPHERE(-9.666).
    mask : (nz, ny, nx) bool tensor, optional
        Brain mask.  Strongly recommended.
    do_median : bool
        Apply AFNI's 19-voxel median post-filter to each output map.

    Returns
    -------
    out : (5, nz, ny, nx) float32 tensor
        Stacked maps ``a, b, c, FWHM, FWQM`` (see ``ACF_LABELS``).
    """
    if device is None:
        device = data.device
    nt, nz, ny, nx = data.shape

    if mask is None:
        mask = torch.ones((nz, ny, nx), dtype=torch.bool)
    mask = mask.to(torch.bool)

    nb = build_neighborhood(nbhd, voxdims)
    n_offsets = nb.offsets.shape[0]
    n_bins = int(nb.bin_radius.shape[0])
    halo_z = int(nb.offsets[:, 0].abs().max().item())

    if verbose:
        print(
            f"local_acf: {nbhd} -> {n_offsets} offsets, {n_bins} radius bins, "
            f"halo_z={halo_z}, device={device}"
        )
        if nt < 39:
            print(
                f"  WARNING: only {nt} time points (AFNI recommends >= 39); "
                "ACF estimates will be noisy."
            )

    bin_radius_d = nb.bin_radius.to(device)

    # --- choose z-slab thickness from the memory model ---
    # Per output z-plane device footprint: input sub-volume planes (with halo),
    # the per-offset product temporary (nt), and the (K x 2) bin accumulators.
    plane = ny * nx
    bytes_per_out_plane = (
        nt * plane * 4  # x_sub plane
        + nt * plane * 4  # per-offset product temporary
        + 2 * n_bins * plane * 4  # bin_sum + bin_cnt
        + 8 * n_bins * plane  # fit-stage float64 curves (rough)
    )
    avail = get_available_memory(device)
    # Halo adds 2*halo_z input planes regardless of slab; budget for it.
    halo_bytes = 2 * halo_z * nt * plane * 4
    slab = max(1, int((avail - halo_bytes) // max(1, bytes_per_out_plane)))
    slab = min(slab, nz)

    out = torch.zeros((5, nz, ny, nx), dtype=torch.float32, device=device)

    # Precompute norms on-device for the whole volume (small: nz*ny*nx).
    # Computed slab-wise to avoid holding a second full 4-D copy when streaming.
    n_slabs = (nz + slab - 1) // slab
    resident = data.device == device

    progress = None
    if verbose:
        progress = tqdm(
            total=n_offsets * n_slabs,
            desc="local_acf offsets",
            leave=False,
            disable=n_offsets * n_slabs < 200,
        )

    for s in range(n_slabs):
        out_z_lo = s * slab
        out_z_hi = min(nz, out_z_lo + slab)
        z_sub_lo = max(0, out_z_lo - halo_z)
        z_sub_hi = min(nz, out_z_hi + halo_z)

        x_sub = data[:, z_sub_lo:z_sub_hi]
        if not resident:
            x_sub = x_sub.to(device)
        else:
            x_sub = x_sub.contiguous()
        nrm_sub = torch.sqrt((x_sub * x_sub).sum(0))
        mask_sub = mask[z_sub_lo:z_sub_hi].to(device)

        bin_sum, bin_cnt = _accumulate_bins(
            x_sub, nrm_sub, mask_sub, nb, z_sub_lo, out_z_lo, out_z_hi, n_bins, progress
        )
        del x_sub, nrm_sub

        # Binned curve for this slab's voxels. Keep the full-slab curves in
        # float32 (the fit promotes to float64 per voxel-chunk only) -- a full
        # (V, K) float64 fit on a large volume is what blows up VRAM.
        present = bin_cnt > 0
        ccbar = torch.where(present, bin_sum / bin_cnt.clamp_min(1), torch.zeros_like(bin_sum))
        del bin_sum, bin_cnt
        # (K, slab, ny, nx) -> (V, K); reshape after permute forces a contiguous
        # copy, so ccbar/present can be freed afterwards.
        slab_nz = out_z_hi - out_z_lo
        v = slab_nz * ny * nx
        y32 = ccbar.permute(1, 2, 3, 0).reshape(v, n_bins)
        wbool = present.permute(1, 2, 3, 0).reshape(v, n_bins)
        del ccbar, present

        # Only fit voxels that are in-mask and have enough present bins.
        slab_mask = mask_sub[out_z_lo - z_sub_lo : out_z_hi - z_sub_lo].reshape(v)
        enough = (wbool.sum(1) >= _MIN_BINS) & slab_mask
        idx = torch.nonzero(enough, as_tuple=False).squeeze(1)
        del mask_sub

        if idx.numel() > 0:
            flat = out.reshape(5, nz * ny * nx)
            # The batched LM fit holds ~14 live (chunk, K) float64 temporaries;
            # chunk voxels so that footprint stays within the memory budget.
            bytes_per_voxel = n_bins * 8 * 16
            fit_chunk = int(get_available_memory(device) // max(1, bytes_per_voxel))
            fit_chunk = max(4096, min(fit_chunk, idx.numel()))
            for c0 in range(0, idx.numel(), fit_chunk):
                cidx = idx[c0 : c0 + fit_chunk]
                yc = y32[cidx].to(torch.float64)
                wc = wbool[cidx].to(torch.float64)
                a, b, c, ok = fit_acf_batched(bin_radius_d, yc, wc, n_iter=lm_iters)
                fwhm = acf_fwhm_batched(a, b, c, 0.5)
                fwqm = acf_fwhm_batched(a, b, c, 0.25)
                params = torch.stack([a, b, c, fwhm, fwqm], dim=0)  # (5, n)
                params = torch.where(ok.view(1, -1), params, torch.zeros_like(params))
                flat[:, cidx + out_z_lo * ny * nx] = params.to(torch.float32)
                del yc, wc, a, b, c, fwhm, fwqm, params

        del y32, wbool

    if progress is not None:
        progress.close()

    if do_median:
        if verbose:
            print("local_acf: 19-voxel median post-filter")
        for k in range(5):
            out[k] = _median_filter19(out[k], mask.to(device))

    return out


# ---------------------------------------------------------------------------
# Local FWHM (Forman finite-difference smoothness estimator; 3dLocalstat -stat fwhm)
# ---------------------------------------------------------------------------

# AFNI labels: FWHMx/FWHMy/FWHMz per axis, FWHMavg for the mean (mri_nstats.c).
FWHM_LABELS = ("FWHMx", "FWHMy", "FWHMz", "FWHMavg")

# sigma -> FWHM (= sqrt(8 ln 2)); AFNI hardcodes 2.35482.
_SIGMA_TO_FWHM = 2.35482

# AFNI mri_nstat_fwhmxyz requires this many neighbors for a variance estimate.
_FWHM_MIN_COUNT = 6


def local_fwhm(
    data: Tensor,
    voxdims: tuple[float, float, float],
    nbhd: str = "SPHERE(-2.0)",
    mask: Tensor | None = None,
    device: torch.device | None = None,
    do_median: bool = False,
    verbose: int = 1,
) -> Tensor:
    """Local image smoothness (FWHM) per voxel, AFNI ``3dLocalstat -stat fwhm``.

    Note: unlike ``3dLocalACF``, ``3dLocalstat`` does *not* median-filter its
    output, so ``do_median`` defaults to ``False`` (with it off the maps match
    AFNI to machine precision).

    Implements AFNI's ``mri_nstat_fwhmxyz``: in each neighborhood, the FWHM along
    an axis comes from the ratio of the first-difference variance to the data
    variance (Forman 1995) -- ``FWHM = 2.35482 * sqrt(-1/(4 ln(1 - varxx/2var))) * dx``.
    The estimator is spatial (per 3-D volume); for a 4-D time series we estimate
    it per volume and average over time (positive estimates only), matching how
    ``3dFWHMx`` averages over sub-bricks.

    Parameters
    ----------
    data : (nt, nz, ny, nx) or (nz, ny, nx) tensor
        Input volume(s). 4-D is averaged over time.
    voxdims : (dx, dy, dz)
        Voxel sizes in mm (x, y, z order).
    nbhd : str
        AFNI ``-nbhd`` string. Needs >= 19 voxels (AFNI's floor for FWHM).

    Returns
    -------
    out : (4, nz, ny, nx) float32 tensor
        Stacked maps ``FWHMx, FWHMy, FWHMz, FWHMavg`` (see ``FWHM_LABELS``).
    """
    if device is None:
        device = data.device
    if data.ndim == 3:
        data = data.unsqueeze(0)
    nt, nz, ny, nx = data.shape
    dx, dy, dz = voxdims

    if mask is None:
        mask = torch.ones((nz, ny, nx), dtype=torch.bool)
    mask = mask.to(torch.bool).to(device)
    maskf = mask.to(torch.float32)

    nb = build_neighborhood(nbhd, voxdims)
    offsets = nb.offsets.tolist()  # (dz, dy, dx)
    n_off = len(offsets)
    if verbose:
        print(f"local_fwhm: {nbhd} -> {n_off} offsets, device={device}")
        if n_off < 19:
            print(
                f"  WARNING: neighborhood has {n_off} voxels; AFNI requires >= 19 "
                "for a stable FWHM estimate."
            )

    axis_vox = torch.tensor([dx, dy, dz], dtype=torch.float32, device=device).view(3, 1, 1, 1)

    # Time-averaged accumulators (positive estimates only): x, y, z, avg.
    fw_sum = torch.zeros((4, nz, ny, nx), dtype=torch.float32, device=device)
    fw_cnt = torch.zeros((4, nz, ny, nx), dtype=torch.float32, device=device)

    # Volume chunking: per volume we hold a (4, tc, nz, ny, nx) field stack plus
    # its running sum/sumsq -- size it to the memory budget.
    bytes_per_vol = 4 * 5 * nz * ny * nx * 4  # ~4 fields x (field+sum+sq+temps)
    tchunk = int(get_available_memory(device) // max(1, bytes_per_vol))
    tchunk = max(1, min(tchunk, nt, 32))

    resident = data.device == device
    n_chunks = (nt + tchunk - 1) // tchunk
    progress = (
        tqdm(total=n_off * n_chunks, desc="local_fwhm", leave=False, disable=n_off * n_chunks < 200)
        if verbose
        else None
    )

    for t0 in range(0, nt, tchunk):
        vt = data[t0 : t0 + tchunk]
        vt = vt.to(device) if not resident else vt.contiguous()

        # Forward first-difference fields (full size; last slab along each axis is
        # zero/invalid) and their validity = both endpoints in mask.
        gx = torch.zeros_like(vt)
        gy = torch.zeros_like(vt)
        gz = torch.zeros_like(vt)
        gx[..., :-1] = vt[..., 1:] - vt[..., :-1]
        gy[..., :-1, :] = vt[..., 1:, :] - vt[..., :-1, :]
        gz[..., :-1, :, :] = vt[..., 1:, :, :] - vt[..., :-1, :, :]
        vg = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device=device)
        vg[0, ..., :-1] = maskf[..., 1:] * maskf[..., :-1]
        vg[1, ..., :-1, :] = maskf[..., 1:, :] * maskf[..., :-1, :]
        vg[2, ..., :-1, :, :] = maskf[..., 1:, :, :] * maskf[..., :-1, :, :]

        # Stack [value, dx, dy, dz] so one shift serves all four reductions.
        fields = torch.stack([vt, gx, gy, gz], dim=0)  # (4, tc, nz, ny, nx)
        valids = torch.stack([maskf, vg[0], vg[1], vg[2]], dim=0)  # (4, nz, ny, nx)
        del gx, gy, gz, vg

        s1 = torch.zeros_like(fields)  # sum over neighborhood
        s2 = torch.zeros_like(fields)  # sum of squares
        cnt = torch.zeros((4, nz, ny, nx), dtype=torch.float32, device=device)

        for oz, oy, ox in offsets:
            gf, inb = _neighbor_shift(fields, oz, oy, ox)  # (4, tc, ...)
            gv, _ = _neighbor_shift(valids, oz, oy, ox)  # (4, nz, ny, nx)
            vm = ((gv > 0.5) & inb).to(torch.float32)  # (4, nz, ny, nx)
            vmt = vm.unsqueeze(1)  # broadcast over time
            s1 += gf * vmt
            s2 += gf * gf * vmt
            cnt += vm
            if progress is not None:
                progress.update(1)

        # Per-(field, volume) neighborhood variance (count-1 denominator).
        cnt_b = cnt.unsqueeze(1)
        denom = (cnt_b - 1.0).clamp_min(1.0)
        var = (s2 - s1 * s1 / cnt_b.clamp_min(1.0)) / denom  # (4, tc, nz, ny, nx)
        enough = cnt >= _FWHM_MIN_COUNT  # (4, nz, ny, nx)

        var_data = var[0]  # (tc, nz, ny, nx)
        data_ok = (var_data > 0) & enough[0]
        # arg = 1 - 0.5 * varaxis / var_data, per axis (fields 1,2,3).
        for ax in range(3):
            varax = var[ax + 1]
            arg = 1.0 - 0.5 * varax / var_data.clamp_min(1e-20)
            valid = data_ok & enough[ax + 1] & (arg > 0.0) & (arg < 1.0)
            fwhm = (
                _SIGMA_TO_FWHM
                * torch.sqrt((-1.0 / (4.0 * torch.log(arg.clamp(1e-12, 1 - 1e-12)))).clamp_min(0.0))
                * axis_vox[ax]
            )
            fwhm = torch.where(valid, fwhm, torch.zeros_like(fwhm))  # (tc, nz, ny, nx)
            pos = fwhm > 0
            fw_sum[ax] += torch.where(pos, fwhm, torch.zeros_like(fwhm)).sum(0)
            fw_cnt[ax] += pos.to(torch.float32).sum(0)

        del fields, valids, s1, s2, cnt, var

    if progress is not None:
        progress.close()

    # Per-axis time-average, then the avg brick = mean of the positive axes.
    fw = fw_sum / fw_cnt.clamp_min(1.0)  # (4, nz, ny, nx); avg slot still zero
    axis_pos = (fw_cnt[:3] > 0).to(torch.float32)
    npos = axis_pos.sum(0).clamp_min(1.0)
    fw[3] = (fw[:3] * axis_pos).sum(0) / npos
    fw = fw * maskf  # zero outside mask

    if do_median:
        if verbose:
            print("local_fwhm: 19-voxel median post-filter")
        for k in range(4):
            fw[k] = _median_filter19(fw[k], mask)

    return fw
