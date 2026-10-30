"""Temporal deveining by response timing: the primitives behind ``ffs_hrftdv``.

Large draining veins deliver the BOLD response later and wider than the
capillary bed they drain (Kay et al. 2020, TDM; wiki [[Kay 2020]]).  TDM finds
one early and one late timecourse for a whole dataset and fits both, as two
competing columns per condition, so the late column soaks up the venous part.
Two of its assumptions do not hold in general: one timecourse pair for the whole
brain, and FIR estimates of short events as the input.  Here the pair belongs
to a cortical *column*, anchored on that column's own HRF, and every candidate
goes through the GLM convolved with the real events, so any design works.

The steps, each a function below:

1. **Units.**  Ribbon voxels are grouped into geodesic patches of the
   midthickness mesh (through each voxel's owning vertex), so a patch is a
   column: every depth under a few mm of cortex.  Voxels just outside the
   ribbon join the nearest ribbon voxel's patch; everything else is tiled.
2. **Unit shape.**  Each unit's mean response picks the best library curve.
3. **Per-voxel lag and width.**  Every voxel scores the unit curve shifted by
   ``tau`` and stretched by ``w`` about its peak (so ``tau`` is a pure peak
   shift).  Exact shifting, not a derivative: the linearisation is only good
   to about +-0.5 s (see ``design/shifted_hrf.py``).
4. **Early/late pair.**  R²-weighted low and high percentiles of the
   responsive voxels' lags, each with the width of the voxels near it.
5. **Library.**  Pairs are snapped to a grid and become a ``(E, 2, n_t)``
   library plus an index map, which ``ffs_reml -hrf_opt`` fits as two columns
   per condition.

Measured on 1 mm floc data (TR 2, 10 s blocks, 2 runs): patch-relative lag
replicates run to run (r 0.46 at R² > 0.2), rises 0.26 s from white to pial, is
later in dark-EPI and vein-mask voxels, and is unrelated to slice time.  Width
is weakly identified with block designs.  The early/late fit flattens the depth
profile of the early betas, at a real cost in sensitivity (VIF ~10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.utils import parabolic_peak_offset

DEFAULT_WIDTHS = (0.6, 0.75, 0.875, 1.0, 1.15, 1.3, 1.5)
"""Stretch factors about the peak.  Wide on purpose: at low R² the fit piles up
on the grid's bounds, and a bound that is too tight hides that."""


@dataclass(frozen=True)
class LagGrid:
    """The candidate (width, lag) grid; candidate ``i`` is ``(i // n_lags, i % n_lags)``."""

    lags: np.ndarray
    widths: np.ndarray

    @classmethod
    def make(cls, max_lag: float, step: float = 0.1, widths: Sequence[float] = DEFAULT_WIDTHS):
        n = int(round(max_lag / step))
        return cls(np.round(np.arange(-n, n + 1) * step, 6), np.asarray(widths, dtype=np.float64))

    @property
    def step(self) -> float:
        return float(self.lags[1] - self.lags[0]) if self.lags.size > 1 else 0.0

    @property
    def n(self) -> int:
        return self.lags.size * self.widths.size


def stretch_shift(h: np.ndarray, dt: float, lag: float, width: float) -> np.ndarray:
    """``h`` stretched by ``width`` about its peak, then moved ``lag`` seconds; peak 1.

    Stretching about the onset would also move the peak by ``(w - 1) * t_peak``,
    so lag and width would trade against each other.  Nothing before t = 0.
    """
    t = np.arange(h.size) * dt
    tpk = t[int(np.argmax(h))]
    arg = tpk + (t - tpk - lag) / width
    v = np.interp(arg, t, h, left=0.0, right=0.0)
    v[arg < 0] = 0.0
    pk = float(np.abs(v).max())
    return v / pk if pk > 0 else v


def curve_peak_fwhm(h: np.ndarray, dt: float) -> tuple[float, float]:
    """Peak time and full width at half maximum of a curve sampled every ``dt`` s."""
    i = int(np.argmax(h))
    above = np.flatnonzero(h >= 0.5 * h[i])
    return i * dt, float(above[-1] - above[0]) * dt


def lag_width_bank(h: np.ndarray, dt: float, grid: LagGrid) -> np.ndarray:
    """``(grid.n, n_t)`` candidates in grid order (width-major, then lag)."""
    return np.stack([stretch_shift(h, dt, lag, w) for w in grid.widths for lag in grid.lags])


def span_basis(curves: np.ndarray, k: int = 3) -> np.ndarray:
    """``(k, n_t)`` leading components of ``curves`` ``(n, n_t)``, the first one positive.

    An F-test on these asks "does the voxel respond with any shape in this set?" in
    one go.  Fitting every curve and keeping the best asks the same thing but gives a
    noise voxel ``n`` chances, so its p-value is not what it says.  Being fixed before
    the data are seen, the span keeps the test honest; for the 20-curve library three
    components hold 99.5% of the shapes' variance.
    """
    _, _, vt = np.linalg.svd(np.asarray(curves, np.float64), full_matrices=False)
    b = vt[:k].copy()
    if b[0, np.argmax(np.abs(b[0]))] < 0:
        b[0] = -b[0]
    return b / np.abs(b).max(1, keepdims=True).clip(1e-12, None)


def screen_basis(h: np.ndarray, dt: float, grid: LagGrid, k: int = 3) -> np.ndarray:
    """:func:`span_basis` of ``h``'s own (width, lag) bank.

    A screen on ``h`` alone keeps the voxels shaped like ``h``, and those are the
    ones whose lag says least.  Canonical sits near the *late* (venous) response,
    so it is the early, narrow microvascular voxels it fits worst; any 2 s shift
    costs ~30% of the R².
    """
    return span_basis(lag_width_bank(h, dt, grid), k)


def as_bank(design: Tensor, n_curves: int) -> Tensor:
    """``(T, P * C)`` condition-major, curve-minor design -> ``(C, T, P)``."""
    T = design.shape[0]
    return design.reshape(T, -1, n_curves).permute(2, 0, 1)


def orthonormal_nuisance(nuisance_per_run: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    """Per-run orthonormal bases of the nuisance column space (rank-safe)."""
    out = []
    for n in nuisance_per_run:
        n = n.to(device=device, dtype=torch.float64)
        if n.shape[1] == 0:
            out.append(n)
            continue
        u, s, _ = torch.linalg.svd(n, full_matrices=False)
        out.append(u[:, s > 1e-8 * s[0]])
    return out


def project_runs(x: Tensor, q_runs: Sequence[Tensor], run_starts: Sequence[int]) -> Tensor:
    """Remove each run's nuisance from the rows of ``x`` (``(..., T, k)``), run by run.

    Per run rather than block-diagonal: the same projection as
    ``glm/xval.py:project_out_nuisance_per_run``, applied to a stack of designs.
    """
    out = x.clone()
    bounds = list(run_starts) + [x.shape[-2]]
    for q, a, b in zip(q_runs, bounds[:-1], bounds[1:], strict=True):
        if q.shape[1] == 0:
            continue
        seg = out[..., a:b, :]
        qq = q.to(seg.dtype)
        out[..., a:b, :] = seg - qq @ (qq.transpose(0, 1) @ seg)
    return out


def bank_q(designs: Tensor) -> Tensor:
    """Orthonormal bases of each projected candidate design, ``(C, T, P)``."""
    q, _ = torch.linalg.qr(designs.double())
    return q.float()


def screen_responsive(
    y: Tensor,
    design: Tensor,
    n_nuisance: int,
    p_thresh: float,
    device: torch.device,
    chunk_size: int | None = None,
    sign_cols: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxels whose projected series an F-test on ``design`` passes, with a positive mean beta.

    ``y`` is ``(T, V)`` and ``design`` ``(T, P)``, both nuisance-projected. Positive only:
    a negative response's lag says nothing about veins draining it.  The sign is the
    mean beta over ``sign_cols`` (default all): with a :func:`screen_basis` design only
    the first component has a meaningful sign.  Returns (responsive, R², positive).
    """
    from scipy.stats import f as fdist

    from fastfuncstuff.memory import estimate_chunk_size

    T, V = y.shape
    P = design.shape[1]
    x = design.to(device=device, dtype=torch.float64)
    pinv = torch.linalg.pinv(x)
    if sign_cols is not None:
        pinv = pinv[torch.as_tensor(sign_cols, device=device)]
    q, _ = torch.linalg.qr(x)
    chunk = chunk_size or estimate_chunk_size(V, T, P, device, operation="glm")
    r2 = np.empty(V)
    mean_beta = np.empty(V)
    for a in range(0, V, chunk):
        yc = y[:, a : a + chunk].to(device=device, dtype=torch.float64)
        ss = ((q.T @ yc) ** 2).sum(0)
        r2[a : a + chunk] = (ss / (yc**2).sum(0).clamp_min(1e-12)).cpu().numpy()
        mean_beta[a : a + chunk] = (pinv @ yc).mean(0).cpu().numpy()
    dof = T - n_nuisance - P
    F = (r2 / P) / np.clip((1 - r2) / dof, 1e-12, None)
    p = fdist.sf(F, P, dof)
    positive = mean_beta > 0
    return (p < p_thresh) & positive, r2, positive


def screen_library(
    y: Tensor,
    designs: Tensor,
    n_nuisance: int,
    p_thresh: float,
    device: torch.device,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Responsive voxels by their best library curve: (responsive, R², positive, best).

    ``designs`` ``(S, T, P)`` are the nuisance-projected designs of the S library
    curves.  Each voxel is fit with every curve and keeps the best, so a voxel only
    counts as responsive in a shape the library allows -- free components would also
    admit dips and double peaks.  The p-value is the best curve's own F, which is
    optimistic for a best-of-S choice; with library curves this correlated the
    effective number of chances is small, and this is a screen, not inference.
    Positive means the best curve's mean beta is.
    """
    from scipy.stats import f as fdist

    from fastfuncstuff.memory import estimate_chunk_size

    T, V = y.shape
    S, _, P = designs.shape
    x = designs.to(device=device, dtype=torch.float64)
    q = torch.linalg.qr(x)[0].float()
    pinv = torch.linalg.pinv(x).float()  # (S, P, T)
    chunk = chunk_size or estimate_chunk_size(
        V, T, P, device, operation="lag_bank", n_designs=S, empty_cache=False
    )
    r2 = np.empty(V)
    best = np.empty(V, np.int64)
    mean_beta = np.empty(V)
    for a in range(0, V, chunk):
        yc = y[:, a : a + chunk].to(device=device, dtype=torch.float32)
        ss = (torch.einsum("stp,tv->spv", q, yc) ** 2).sum(1)
        r2_all = ss / (yc**2).sum(0).clamp_min(1e-12)
        b = r2_all.argmax(0)
        cols = torch.arange(b.numel(), device=device)
        beta = torch.einsum("spt,tv->spv", pinv, yc)[b, :, cols]  # (v, P)
        r2[a : a + chunk] = r2_all[b, cols].cpu().numpy()
        best[a : a + chunk] = b.cpu().numpy()
        mean_beta[a : a + chunk] = beta.mean(1).cpu().numpy()
    dof = T - n_nuisance - P
    F = (r2 / P) / np.clip((1 - r2) / dof, 1e-12, None)
    positive = mean_beta > 0
    return (fdist.sf(F, P, dof) < p_thresh) & positive, r2, positive, best


def unit_shapes(means: Tensor, q_lib: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Best library curve for each unit's mean series: index ``(U,)`` and R² ``(S, U)``.

    In-sample R² is a fair referee here: every candidate has the same number
    of columns, so none can win on complexity.
    """
    m = means.to(device=q_lib.device, dtype=torch.float32)
    ss = (torch.einsum("stp,tu->spu", q_lib, m) ** 2).sum(1)
    r2 = ss / (m**2).sum(0).clamp_min(1e-12)
    return r2.argmax(0).cpu().numpy(), r2.cpu().numpy()


@dataclass
class LagFit:
    lag: np.ndarray  # (V,) seconds, relative to the unit curve's peak
    width: np.ndarray  # (V,) stretch about the peak
    r2: np.ndarray  # (V,) at the best candidate
    at_bound: np.ndarray  # (V,) bool: lag on the edge of the grid


def fit_lag_width(
    y: Tensor,
    q: Tensor,
    grid: LagGrid,
    device: torch.device,
    chunk_size: int | None = None,
) -> LagFit:
    """Each voxel's best (width, lag) candidate, with sub-step parabolic lag refinement.

    ``y`` is ``(T, V)`` nuisance-projected data (any device), ``q`` the
    candidates' Q factors ``(grid.n, T, P)`` on ``device``.  R² of a candidate
    is ``||Q^T y||² / ||y||²``: the projection is the whole fit.
    """
    from fastfuncstuff.memory import estimate_chunk_size

    T, V = y.shape
    C, _, P = q.shape
    n_w, n_l = grid.widths.size, grid.lags.size
    chunk = chunk_size or estimate_chunk_size(
        V, T, P, device, operation="lag_bank", n_designs=C, empty_cache=False
    )
    lags_t = torch.as_tensor(grid.lags, device=device, dtype=torch.float32)
    out = LagFit(np.empty(V), np.empty(V), np.empty(V), np.zeros(V, bool))
    for a in range(0, V, chunk):
        yc = y[:, a : a + chunk].to(device=device, dtype=torch.float32)
        ss = (torch.einsum("ctp,tv->cpv", q, yc) ** 2).sum(1)
        r2 = (ss / (yc**2).sum(0).clamp_min(1e-12)).view(n_w, n_l, -1)
        best = r2.reshape(C, -1).argmax(0)
        wi, li = best // n_l, best % n_l
        cols = torch.arange(best.numel(), device=device)
        curve = r2[wi, :, cols].T  # (n_l, v): the lag profile at each voxel's width
        off = parabolic_peak_offset(curve, li, dim=0) if n_l >= 3 else torch.zeros_like(lags_t[li])
        edge = (li == 0) | (li == n_l - 1)
        sl = slice(a, a + yc.shape[1])
        out.lag[sl] = (lags_t[li] + torch.where(edge, 0.0, off) * grid.step).cpu().numpy()
        out.width[sl] = grid.widths[wi.cpu().numpy()]
        out.r2[sl] = r2.reshape(C, -1).max(0).values.cpu().numpy()
        out.at_bound[sl] = edge.cpu().numpy()
    return out


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """``q``-th percentile of ``x`` with weights ``w`` (step interpolation)."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(int(np.searchsorted(c, q / 100.0 * c[-1])), x.size - 1)])


@dataclass(frozen=True)
class ColumnPair:
    early_lag: float
    late_lag: float
    early_width: float
    late_width: float
    n_voxels: int
    fallback: bool = False


def column_pair(
    lag: np.ndarray,
    width: np.ndarray,
    r2: np.ndarray,
    percentiles: tuple[float, float] = (10.0, 90.0),
    tail: float = 20.0,
    min_voxels: int = 8,
) -> ColumnPair | None:
    """Early and late lags of one unit from its usable voxels (R²-weighted).

    Percentiles rather than the extremes: the extremes of noisy estimates are
    biased outward, and with them the pair would be set by the two worst
    voxels.  Each end takes the weighted median width of the voxels in its
    ``tail`` percent, so the narrower early curve and the wider late one come
    from the data.
    """
    if lag.size < min_voxels:
        return None
    w = np.clip(r2, 1e-6, None)
    lo, hi = (weighted_percentile(lag, w, p) for p in percentiles)
    lo_cut, hi_cut = weighted_percentile(lag, w, tail), weighted_percentile(lag, w, 100 - tail)
    sel_lo, sel_hi = lag <= lo_cut, lag >= hi_cut
    return ColumnPair(
        lo,
        hi,
        weighted_percentile(width[sel_lo], w[sel_lo], 50),
        weighted_percentile(width[sel_hi], w[sel_hi], 50),
        int(lag.size),
    )


def surface_patches(
    mid: np.ndarray,
    faces: np.ndarray,
    seeds: np.ndarray,
    radius_mm: float,
    seed: int = 0,
) -> np.ndarray:
    """Geodesic patches covering ``seeds``: a label per mesh vertex (-1 = not reached).

    Greedy cover (a random seed order, each new centre claiming its geodesic
    ball), then every vertex to its nearest centre along the mesh -- along
    the mesh so a patch never jumps a sulcus to the opposite bank.
    """
    from scipy.sparse.csgraph import dijkstra

    from fastfuncstuff.surface.mesh import MeshTopology, geodesic_ball

    topo = MeshTopology.from_faces(faces, mid.shape[0])
    covered = np.zeros(mid.shape[0], bool)
    centres = []
    for v in np.random.default_rng(seed).permutation(np.unique(seeds)):
        if covered[v]:
            continue
        centres.append(int(v))
        ball, _ = geodesic_ball(mid, topo, int(v), radius_mm)
        covered[ball] = True
    labels = np.full(mid.shape[0], -1, np.int64)
    if not centres:
        return labels
    _, _, src = dijkstra(
        topo.edge_graph(mid),
        indices=centres,
        min_only=True,
        return_predecessors=True,
        limit=2.0 * radius_mm,
    )
    reached = src >= 0
    lookup = np.full(mid.shape[0], -1, np.int64)
    lookup[centres] = np.arange(len(centres))
    labels[reached] = lookup[src[reached]]
    return labels


def volume_tiles(ijk: np.ndarray, voxel_mm: Sequence[float], tile_mm: float) -> np.ndarray:
    """Cubic tiles of side ``tile_mm`` over voxel coordinates ``(N, 3)``: a label per voxel."""
    cell = np.floor(ijk * np.asarray(voxel_mm) / tile_mm).astype(np.int64)
    _, labels = np.unique(cell, axis=0, return_inverse=True)
    return labels.reshape(-1)


@dataclass
class RibbonPatches:
    """Ribbon voxels grouped into cortical columns, flat C-order over ``(X, Y, Z)``."""

    label: np.ndarray  # patch per voxel, -1 = not a covered ribbon voxel
    n_patches: int
    depth: np.ndarray  # equivolume, 0 white .. 1 pial, NaN outside the ribbon
    owner_hemi: np.ndarray | None = None  # surface index per voxel, -1 = not covered ribbon
    owner_vertex: np.ndarray | None = None  # owning vertex per voxel, -1 likewise


def ribbon_patches(
    surfaces: Sequence,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    coverage: np.ndarray,
    patch_mm: float,
    device: torch.device | None = None,
) -> RibbonPatches:
    """Covered ribbon voxels -> geodesic midthickness patches via their owning vertex.

    ``coverage`` is ``(X, Y, Z)``; patch numbers run on across hemispheres. Depth
    is kept for every ribbon voxel, covered or not.
    """
    from fastfuncstuff.surface.volume_depth import cortical_depth_volumes

    dv = cortical_depth_volumes(
        list(surfaces), affine, shape, n_layers=3, owners=True, device=device
    )
    assert dv.owner_hemi is not None and dv.owner_vertex is not None
    in_cov = coverage.reshape(-1)[dv.gm_idx]
    gm_idx, o_hemi, o_vert = dv.gm_idx[in_cov], dv.owner_hemi[in_cov], dv.owner_vertex[in_cov]
    depth = np.full(int(np.prod(shape)), np.nan, np.float32)
    depth[dv.gm_idx] = dv.equivol
    label = np.full(int(np.prod(shape)), -1, np.int64)
    owner_h = np.full(label.size, -1, np.int64)
    owner_v = np.full(label.size, -1, np.int64)
    owner_h[gm_idx], owner_v[gm_idx] = o_hemi, o_vert
    n_units = 0
    for h, s in enumerate(surfaces):
        sel = o_hemi == h
        if not sel.any():
            continue
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        vlab = surface_patches(mid, np.asarray(s.faces), o_vert[sel], patch_mm)
        lab = vlab[o_vert[sel]]
        ok = lab >= 0
        label[gm_idx[sel][ok]] = lab[ok] + n_units
        n_units += int(vlab.max()) + 1
    return RibbonPatches(label, n_units, depth, owner_h, owner_v)


@dataclass
class AtlasUnits:
    """Ribbon voxels relabelled by ROI: units ``0..n_roi_units-1`` belong to ROIs."""

    label: np.ndarray  # unit per voxel, -1 = none; ROI units first, then other patches
    n_units: int  # every cortical unit (ROI units + remaining patches)
    n_roi_units: int
    unit_roi: np.ndarray  # (n_units,) region index of each unit, -1 = not in an ROI
    n_overlap: int  # covered ribbon voxels whose column more than one ROI claims


def atlas_labels(
    patches: RibbonPatches,
    members: Sequence[dict[int, np.ndarray]],
    unit: str = "roi",
    keep_rest: bool = True,
) -> AtlasUnits:
    """Units from ROIs' columns: one per ROI, or one per (ROI, patch) with ``unit="patch"``.

    ``members[r]`` maps a surface index to the vertex mask of region ``r``.  A voxel
    joins the region owning its column, the first one listed where regions overlap
    (a unit has to be a partition).  ``keep_rest`` keeps the ordinary patch of every
    covered ribbon voxel outside the atlas, so the rest of cortex still gets pairs.
    """
    if unit not in ("roi", "patch"):
        raise ValueError(f"atlas unit must be 'roi' or 'patch', not {unit!r}")
    assert patches.owner_hemi is not None and patches.owner_vertex is not None
    oh, ov = patches.owner_hemi, patches.owner_vertex
    covered = patches.label >= 0
    region = np.full(oh.size, -1, np.int64)
    claims = np.zeros(oh.size, np.int64)
    for r, m in enumerate(members):
        hit = np.zeros(oh.size, bool)
        for h, mask in m.items():
            mask = np.asarray(mask, bool)
            sel = covered & (oh == h)
            hit[sel] = mask[ov[sel]]
        claims += hit
        region[hit & (region < 0)] = r
    in_roi = region >= 0
    key = region if unit == "roi" else region * patches.n_patches + patches.label
    roi_keys, roi_lab = np.unique(key[in_roi], return_inverse=True)
    n_roi = roi_keys.size
    label = np.full(oh.size, -1, np.int64)
    label[in_roi] = roi_lab
    unit_roi = roi_keys if unit == "roi" else roi_keys // patches.n_patches
    n_units = n_roi
    if keep_rest:
        rest = covered & ~in_roi
        rest_keys, rest_lab = np.unique(patches.label[rest], return_inverse=True)
        label[rest] = rest_lab + n_roi
        n_units += rest_keys.size
        unit_roi = np.r_[unit_roi, np.full(rest_keys.size, -1, np.int64)]
    return AtlasUnits(label, n_units, n_roi, unit_roi.astype(np.int64), int((claims > 1).sum()))


def extend_labels(labels: np.ndarray, voxel_mm: Sequence[float], extend_mm: float) -> np.ndarray:
    """Volume of labels (-1 empty) grown by up to ``extend_mm`` into empty voxels.

    Each empty voxel takes the label of the *nearest* labelled voxel, so where
    two gyri touch, the gap splits between them instead of either claiming it.
    """
    if extend_mm <= 0 or not (labels >= 0).any():
        return labels
    src = nearest_labelled(labels, voxel_mm, extend_mm)
    grown = labels.reshape(-1)[np.maximum(src, 0)].reshape(labels.shape)
    return np.where(src.reshape(labels.shape) >= 0, grown, labels)


def nearest_labelled(labels: np.ndarray, voxel_mm: Sequence[float], extend_mm: float) -> np.ndarray:
    """Flat index of each empty voxel's nearest labelled voxel within ``extend_mm``.

    -1 for labelled voxels and those farther away. Nearest, so where two gyri
    touch the gap splits between them.
    """
    from scipy.ndimage import distance_transform_edt

    out = np.full(labels.size, -1, np.int64)
    if extend_mm <= 0 or not (labels >= 0).any():
        return out
    dist, idx = distance_transform_edt(labels < 0, sampling=voxel_mm, return_indices=True)
    near = np.ravel_multi_index(tuple(idx), labels.shape).reshape(-1)
    take = ((labels < 0) & (dist <= extend_mm)).reshape(-1)
    out[take] = near[take]
    return out


@dataclass
class PairLibrary:
    curves: np.ndarray  # (E, 2, n_t): early, late
    entries: np.ndarray  # (E, 5): shape index, early lag, late lag, early width, late width
    index: np.ndarray  # (U,) entry of each unit

    def table(self, shape_labels: Sequence[str]) -> list[dict]:
        return [
            {
                "entry": i + 1,
                "shape": shape_labels[int(e[0])],
                "early_lag": float(e[1]),
                "late_lag": float(e[2]),
                "early_width": float(e[3]),
                "late_width": float(e[4]),
            }
            for i, e in enumerate(self.entries)
        ]


def build_pair_library(
    shapes: np.ndarray,
    unit_shape: np.ndarray,
    pairs: Sequence[ColumnPair],
    dt: float,
    lag_step: float,
    widths: Sequence[float],
) -> PairLibrary:
    """Snap each unit's pair to the lag step and width grid; one curve pair per distinct key.

    Snapping keeps the library small: ``ffs_reml`` builds a design per entry,
    and units with the same timing within ``lag_step`` share one.
    """
    wgrid = np.asarray(widths, dtype=np.float64)

    def snap_w(w: float) -> float:
        return float(wgrid[np.argmin(np.abs(wgrid - w))])

    keys = np.array(
        [
            (
                int(s),
                round(p.early_lag / lag_step) * lag_step,
                round(p.late_lag / lag_step) * lag_step,
                snap_w(p.early_width),
                snap_w(p.late_width),
            )
            for s, p in zip(unit_shape, pairs, strict=True)
        ]
    ).round(6)
    entries, index = np.unique(keys, axis=0, return_inverse=True)
    curves = np.stack(
        [
            np.stack(
                [
                    stretch_shift(shapes[int(e[0])], dt, e[1], e[3]),
                    stretch_shift(shapes[int(e[0])], dt, e[2], e[4]),
                ]
            )
            for e in entries
        ]
    ).astype(np.float32)
    return PairLibrary(curves, entries, index.reshape(-1))


def pooled_late_patterns(
    y: Tensor,
    unit: np.ndarray,
    responsive: np.ndarray,
    library: PairLibrary,
    design_for: Callable[[np.ndarray], Tensor],
    project: Callable[[Tensor], Tensor],
    device: torch.device,
) -> np.ndarray:
    """Each unit's late *condition pattern*, ``(U, P)``, unit L1 norm.

    The free early/late model gives every voxel a late beta per condition, and
    two collinear columns per condition fight: one goes strongly negative to
    pay for the other.  Venous signal is drained from upstream tissue, so its
    mix of conditions should be the neighbourhood's rather than the voxel's
    own.  That pattern is fit once per unit, early and late together on the
    mean of its responsive voxels, and each voxel then needs a single late
    column (its size), which ``ffs_reml -dsort`` fits.  Units with no
    responsive voxel take the median pattern of the rest.

    The price is an assumption: whatever selectivity the whole unit shares is
    called late, so a genuinely smooth pattern loses some of its early part.
    """
    U = int(unit.max()) + 1
    T = y.shape[0]
    sums = torch.zeros(T, U, dtype=torch.float64)
    counts = np.bincount(unit[responsive], minlength=U)
    idx = np.flatnonzero(responsive)
    for a in range(0, idx.size, 50000):
        sel = idx[a : a + 50000]
        sums.index_add_(1, torch.as_tensor(unit[sel]), y[:, sel].double().cpu())
    means = (sums / torch.as_tensor(np.clip(counts, 1, None))).to(device)
    pattern = np.full((U, 0), np.nan)
    for e in range(library.curves.shape[0]):
        units = np.flatnonzero((library.index == e) & (counts > 0))
        if units.size == 0:
            continue
        x = project(design_for(library.curves[e]))  # (T, 2P): early, late per condition
        beta = torch.linalg.lstsq(x, means[:, torch.as_tensor(units, device=device)]).solution
        late = beta[1::2].T.cpu().numpy()  # (n, P)
        if pattern.shape[1] == 0:
            pattern = np.full((U, late.shape[1]), np.nan)
        pattern[units] = late / np.clip(np.abs(late).sum(1, keepdims=True), 1e-12, None)
    if pattern.shape[1] == 0:
        raise ValueError("no unit has a responsive voxel: no late pattern to pool")
    fill = np.nanmedian(pattern, axis=0)
    fill /= max(float(np.abs(fill).sum()), 1e-12)
    pattern[~np.isfinite(pattern).all(1)] = fill
    return pattern


def late_regressors(
    library: PairLibrary, pattern: np.ndarray, design_for: Callable[[np.ndarray], Tensor]
) -> np.ndarray:
    """``(T, U)``: each unit's pooled late column, its pair's late curve x its pattern."""
    U = pattern.shape[0]
    out = None
    for e in range(library.curves.shape[0]):
        units = np.flatnonzero(library.index == e)
        if units.size == 0:
            continue
        x = design_for(library.curves[e][1]).double().cpu().numpy()  # (T, P)
        if out is None:
            out = np.zeros((x.shape[0], U), np.float32)
        out[:, units] = x @ pattern[units].T
    assert out is not None
    return out


@dataclass
class TdvResult:
    unit: np.ndarray  # (V,) unit label
    unit_kind: np.ndarray  # (U,) 0 surface patch, 1 volume tile
    shape: np.ndarray  # (V,) library curve of the voxel's unit
    fit: LagFit
    rel_lag: np.ndarray  # (V,) lag minus the unit's R²-weighted median (responsive units)
    responsive: np.ndarray  # (V,) bool
    screen_r2: np.ndarray  # (V,)
    pairs: list[ColumnPair]  # (U,)
    library: PairLibrary
    voxel_entry: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # (V,) 0-based

    @property
    def n_fallback(self) -> int:
        return sum(p.fallback for p in self.pairs)


def fit_hrftdv(
    y: Tensor,
    unit: np.ndarray,
    unit_kind: np.ndarray,
    shapes: np.ndarray,
    design_for: Callable[[np.ndarray], Tensor],
    project: Callable[[Tensor], Tensor],
    responsive: np.ndarray,
    screen_r2: np.ndarray,
    grid: LagGrid,
    dt: float,
    *,
    percentiles: tuple[float, float] = (10.0, 90.0),
    min_voxels: int = 8,
    lag_step: float = 0.2,
    unit_weight: np.ndarray | None = None,
    device: torch.device,
    verbose: bool = True,
) -> TdvResult:
    """Steps 2-5 of the module docstring for every voxel of ``y`` (``(T, V)``, projected).

    ``design_for(curves)`` builds the ``(T, P * C)`` task design for ``C``
    impulse curves; ``project`` removes the nuisance from a ``(..., T, k)``
    stack.  ``unit`` labels each voxel ``0..U-1``.  ``unit_weight`` ``(V,)``
    replaces the responsive-only unit mean of step 2 with a weighted mean
    over every voxel, so a unit the screen barely reached still gets its own
    shape; the pair (step 4) still comes from the responsive voxels.
    """
    V = y.shape[1]
    U = int(unit.max()) + 1
    S = shapes.shape[0]

    # -- step 2: each unit's curve, from the (weighted) mean of its voxels
    q_lib = bank_q(project(as_bank(design_for(shapes), S).to(device)))
    w = (
        responsive.astype(np.float64)
        if unit_weight is None
        else np.asarray(unit_weight, np.float64)
    )
    sums = torch.zeros(y.shape[0], U, dtype=torch.float64)
    counts = np.bincount(unit, weights=w, minlength=U)
    resp_idx = np.flatnonzero(w > 0)
    for a in range(0, resp_idx.size, 50000):
        sel = resp_idx[a : a + 50000]
        wt = torch.as_tensor(w[sel])
        sums.index_add_(1, torch.as_tensor(unit[sel]), y[:, sel].double().cpu() * wt)
    means = (sums / torch.as_tensor(np.where(counts > 0, counts, 1.0))).float()
    best, _ = unit_shapes(means, q_lib)
    del q_lib
    has = counts > 0
    fallback_shape = (
        int(np.bincount(best[has], weights=counts[has], minlength=S).argmax()) if has.any() else 0
    )
    best = np.where(has, best, fallback_shape)
    shape = best[unit]

    # -- step 3: lag x width per voxel, one bank per curve in use
    fit = LagFit(np.full(V, np.nan), np.full(V, np.nan), np.full(V, np.nan), np.zeros(V, bool))
    used = np.unique(shape)
    for s in tqdm(used, desc="lag x width", leave=True, disable=not verbose or used.size < 2):
        vox = np.flatnonzero(shape == s)
        q = bank_q(
            project(as_bank(design_for(lag_width_bank(shapes[s], dt, grid)), grid.n).to(device))
        )
        part = fit_lag_width(y[:, vox], q, grid, device)
        del q
        fit.lag[vox], fit.width[vox], fit.r2[vox], fit.at_bound[vox] = (
            part.lag,
            part.width,
            part.r2,
            part.at_bound,
        )

    # -- step 4: a pair per unit from its usable voxels
    usable = responsive & ~fit.at_bound
    pairs: list[ColumnPair | None] = []
    rel = np.full(V, np.nan)
    order = np.argsort(unit, kind="stable")
    bounds = np.searchsorted(unit[order], np.arange(U + 1))
    for u in range(U):
        members = order[bounds[u] : bounds[u + 1]]
        m = members[usable[members]]
        pair = column_pair(fit.lag[m], fit.width[m], fit.r2[m], percentiles, min_voxels=min_voxels)
        pairs.append(pair)
        if m.size:
            rel[members] = fit.lag[members] - weighted_percentile(
                fit.lag[m], np.clip(fit.r2[m], 1e-6, None), 50
            )
    valid = [p for p in pairs if p is not None]
    if valid:
        glob = ColumnPair(
            float(np.median([p.early_lag for p in valid])),
            float(np.median([p.late_lag for p in valid])),
            float(np.median([p.early_width for p in valid])),
            float(np.median([p.late_width for p in valid])),
            0,
            fallback=True,
        )
    else:
        # Nothing responsive anywhere: a pair one lag step either side of the curve.
        glob = ColumnPair(-lag_step, lag_step, 1.0, 1.0, 0, fallback=True)
    filled = [p if p is not None else glob for p in pairs]

    # -- step 5: the library reml reads
    library = build_pair_library(shapes, best, filled, dt, lag_step, grid.widths)
    return TdvResult(
        unit=unit,
        unit_kind=unit_kind,
        shape=shape,
        fit=fit,
        rel_lag=rel,
        responsive=responsive,
        screen_r2=screen_r2,
        pairs=filled,
        library=library,
        voxel_entry=library.index[unit],
    )


@dataclass
class RowLags:
    lag: np.ndarray  # (R,) s, relative to the row's unit curve peak
    width: np.ndarray  # (R,)
    r2: np.ndarray  # (R,)
    at_bound: np.ndarray  # (R,) bool
    run_lag: np.ndarray  # (R, n_runs): the same fit on each run alone


def fit_row_lags(
    y_rows: Tensor,
    row_shape: np.ndarray,
    shapes: np.ndarray,
    design_for: Callable[[np.ndarray], Tensor],
    project: Callable[[Tensor], Tensor],
    grid: LagGrid,
    dt: float,
    run_starts: Sequence[int],
    device: torch.device,
) -> RowLags:
    """Step 3 on ROI x depth means: each row's (width, lag) against its unit's curve.

    ``y_rows`` is ``(T, R)``, nuisance-projected like the voxels.  An ROI mean has the
    SNR a single voxel never does, so this is where lag is best measured; fitting
    each run alone as well gives the spread a depth profile should be read against.
    """
    R = y_rows.shape[1]
    T = y_rows.shape[0]
    bounds = [*run_starts, T]
    out = RowLags(np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan),
                  np.zeros(R, bool), np.full((R, len(run_starts)), np.nan))  # fmt: skip
    for s in np.unique(row_shape):
        sel = np.flatnonzero(row_shape == s)
        designs = project(
            as_bank(design_for(lag_width_bank(shapes[int(s)], dt, grid)), grid.n).to(device)
        )
        part = fit_lag_width(y_rows[:, sel], bank_q(designs), grid, device)
        out.lag[sel], out.width[sel], out.r2[sel], out.at_bound[sel] = (
            part.lag, part.width, part.r2, part.at_bound,
        )  # fmt: skip
        for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            one = fit_lag_width(y_rows[a:b, sel], bank_q(designs[:, a:b, :]), grid, device)
            out.run_lag[sel, r] = one.lag
    return out


def override_pairs(
    res: TdvResult,
    overrides: dict[int, ColumnPair],
    shapes: np.ndarray,
    dt: float,
    lag_step: float,
    widths: Sequence[float],
) -> TdvResult:
    """``res`` with the pairs of the units in ``overrides`` replaced and the library rebuilt."""
    pairs = [overrides.get(u, p) for u, p in enumerate(res.pairs)]
    unit_shape = np.zeros(len(pairs), np.int64)
    unit_shape[res.unit] = res.shape
    library = build_pair_library(shapes, unit_shape, pairs, dt, lag_step, widths)
    return replace(res, pairs=pairs, library=library, voxel_entry=library.index[res.unit])
