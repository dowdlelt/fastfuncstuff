"""Column sampling, continuous vascular inversion, and voxel residual preservation."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from fastfuncstuff import policy_linalg as plinalg
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.xval import project_out_nuisance_per_run
from fastfuncstuff.laminar.params import P0, ModelSpec
from fastfuncstuff.laminar.timeseries import (
    DynamicResult,
    dynamic_inverse,
    linearized_vasculature,
    loro_prior_ridge,
    split_half_scores,
    vascular_transfer,
)
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import accum_dtype, invalid_voxel_mask


def depth_patches(
    mask: np.ndarray, depth: np.ndarray, spacing: np.ndarray, radius_mm: float = 5.0
) -> np.ndarray:
    """Disjoint local patches reached through GM, avoiding jumps across sulci.

    Labels are flat C-order within mask. With segmentation-only depths, genuine
    touching-bank bridges cannot be resolved. Prefer surface column labels when
    available; this fallback does not claim to trace anatomical vessels.
    """
    if radius_mm <= 0 or np.any(np.asarray(spacing) <= 0):
        raise ValueError("radius and voxel spacing must be positive")
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return np.empty(0, np.int64)
    index = np.full(mask.shape, -1, np.int64)
    index.reshape(-1)[ids] = np.arange(ids.size)
    rows, cols, weights = [], [], []
    for ax in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        a, b = index[tuple(lo)], index[tuple(hi)]
        good = (a >= 0) & (b >= 0)
        rows.extend([a[good], b[good]])
        cols.extend([b[good], a[good]])
        weights.extend([np.full(good.sum(), spacing[ax])] * 2)
    graph = sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ids.size, ids.size),
    )
    # Mid-ribbon seeds give patches a chance to span depth before boundary seeds.
    order = np.argsort(np.abs(depth.reshape(-1)[ids] - 0.5), kind="stable")
    labels = np.full(ids.size, -1, np.int64)
    next_label = 0
    for seed in order:
        if labels[seed] >= 0:
            continue
        distance = dijkstra(graph, indices=int(seed), limit=radius_mm, directed=False)
        take = (labels < 0) & np.isfinite(distance)
        labels[take] = next_label
        next_label += 1
    return labels


def depth_interpolation(depth: np.ndarray, K: int) -> np.ndarray:
    """Point-sample weights (V,K), 0 pial, 1 white; rows sum to one.

    This estimates bin-sharing blur exactly for these weights. It cannot infer
    subvoxel cortical geometry from a segmentation's single mean-depth number.
    """
    if K < 2 or not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
        raise ValueError("K>=2 and finite depths between 0 and 1 required")
    pos = np.clip(np.asarray(depth) * K - 0.5, 0, K - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, K - 1)
    weights = np.zeros((len(depth), K), np.float32)
    weights[np.arange(len(depth)), lo] = 1 - (pos - lo)
    weights[np.arange(len(depth)), hi] += pos - lo
    return weights


@dataclass
class ColumnProfiles:
    """Each valid patch's detrended PSC depth profile and how its bins were read."""

    members: list[np.ndarray]  # voxel indices per valid patch
    # R x U x T x K; for runs of different lengths, one 1 x U x T_r x K per run
    profiles: torch.Tensor | list[torch.Tensor]
    # U x K x K; a list (one per run) when runs observe different voxels
    sampling: torch.Tensor | list[torch.Tensor]  # physiological depths -> measured bins
    noise_cov: torch.Tensor | list[torch.Tensor]  # bin covariance from shared voxels
    read: np.ndarray  # V x K point-sample weights
    nuisance: list[torch.Tensor]  # per run, T x n_poly
    unit_labels: np.ndarray  # U, the label of each valid patch
    present: list[np.ndarray] | None = None  # per run, V bool; None = every voxel, every run


def column_profiles(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    K: int,
    polort: int,
    device: torch.device,
    use: np.ndarray | None = None,
    robust: bool = False,
    present: list[np.ndarray] | None = None,
    voxel_weight: np.ndarray | None = None,
) -> ColumnProfiles:
    """Percent change, per-run Legendre drift removed, averaged into K bins per patch.

    ``present`` (per run, a V bool) marks the voxels a run actually observed, for
    slabs that move between runs. Absent entries are never read. Each run then
    builds its patches from its own voxels (see :func:`_partial_column_profiles`).

    ``voxel_weight`` (V) scales each voxel's share of its bins: a voxel's ribbon
    fraction, so one the ribbon only clips fills the mask without counting as a
    whole voxel of cortex.

    Only voxels flagged in ``use`` (default all) build the profiles -- the ribbon,
    not the shell around it. Patches that do not observe every model depth are
    left out (they pass through).

    ``robust`` weights each voxel by min(1, (median SD of its patch's voxels at
    the same depth bin / its own SD)^2). A vein three times as variable as its
    neighbours then counts 1/9: unweighted, it inflates the bin and the inverse
    reads the excess as LOCAL signal, which every other voxel then inherits.
    """
    if not runs or any(r.ndim != 2 or r.shape[0] != runs[0].shape[0] for r in runs):
        raise ValueError("runs must be (voxels, time) over the same voxels")
    V = runs[0].shape[0]
    lengths = [int(r.shape[1]) for r in runs]
    if len(labels) != V or len(depth) != V or min(lengths) < 4:
        raise ValueError("depth/labels must match voxels and runs need >=4 samples")
    if present is not None and not all(bool(np.all(p)) for p in present):
        return _partial_column_profiles(
            runs, depth, labels, K, polort, device, use, robust, present, voxel_weight
        )
    # Chunked finiteness: a whole-series isfinite allocates a full float copy of
    # the run for the abs it is built on. The all-zero half of the mask is
    # implied by the positive-mean test that follows it.
    if not all(
        not bool(invalid_voxel_mask(r).any()) and bool((r.mean(-1) > 0).all()) for r in runs
    ):
        raise ValueError("runs need finite data and positive voxel means")
    R = len(runs)
    read = depth_interpolation(depth, K)
    unique, inverse = np.unique(labels, return_inverse=True)
    members, covs, sampling, kept = [], [], [], []
    samples: list[list[np.ndarray]] = [[] for _ in runs]  # per run, (T_r, K) per patch
    use = np.ones(V, bool) if use is None else np.asarray(use, bool)
    psc = torch.cat([100 * (r / r.mean(-1, keepdim=True) - 1) for r in runs], -1)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    nuisance = [construct_polynomial_matrix(t, polort, device, torch.float32) for t in lengths]
    psc, _ = project_out_nuisance_per_run(
        psc, torch.empty(sum(lengths), 0, device=device), nuisance, starts, device=device,
        verbose=False,
    )  # fmt: skip
    # Spatial means use CPU storage; only the depth inverse is sent to compute.
    psc_np = np.split(psc.cpu().numpy(), starts[1:], axis=1)  # per run, V x T_r
    sd = psc.std(-1).cpu().numpy() if robust else None
    del psc
    for unit in range(len(unique)):
        ix = np.flatnonzero((inverse == unit) & use)
        W = read[ix].T if voxel_weight is None else read[ix].T * voxel_weight[ix][None]
        if sd is not None and ix.size:
            weight = np.ones(ix.size, np.float32)
            home = read[ix].argmax(1)
            for k in np.unique(home):
                at = home == k
                med = np.median(sd[ix[at]])
                weight[at] = np.minimum(1.0, (med / np.maximum(sd[ix[at]], 1e-12)) ** 2)
            W = W * weight[None]
        mass = W.sum(-1)
        # Every model depth must be observed; interpolating missing depths would
        # manufacture the independent information the inverse needs.
        if ix.size < 2 * K or mass.min() < 1.0:
            continue
        S = W / mass[:, None]
        kept.append(unique[unit])
        members.append(ix)
        sampling.append(S @ read[ix])
        covs.append(S @ S.T)
        for r in range(R):
            samples[r].append((S @ psc_np[r][ix]).T)
    if not kept:
        raise ValueError("no patch spans every model depth; enlarge patches or reduce -depths")
    if len(set(lengths)) == 1:
        profiles = torch.as_tensor(np.stack([np.stack(s) for s in samples]))
    else:
        profiles = [torch.as_tensor(np.stack(s))[None] for s in samples]
    return ColumnProfiles(
        members,
        profiles,
        torch.as_tensor(np.stack(sampling)),
        torch.as_tensor(np.stack(covs)),
        read,
        nuisance,
        np.asarray(kept),
    )


def _partial_column_profiles(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    K: int,
    polort: int,
    device: torch.device,
    use: np.ndarray | None,
    robust: bool,
    present: list[np.ndarray],
    voxel_weight: np.ndarray | None = None,
) -> ColumnProfiles:
    """Column profiles when runs observe different voxels (a slab that moved).

    Requiring every voxel in every run keeps only the runs' intersection; on a
    thin slab that drifts a few mm between runs that is a third of the ribbon.
    Here each run averages each patch over the voxels it observed. A patch is
    kept if it spans every model depth in at least one run; in a run where it
    does not, its profile is zero and its sampling the identity, so nothing is
    inverted or removed there. Sampling and noise covariance are per run because
    they depend on which voxels were averaged. Absent entries are never read.
    """
    R, V = len(runs), runs[0].shape[0]
    present = [np.asarray(p, bool) for p in present]
    if len(present) != R or any(p.shape != (V,) for p in present):
        raise ValueError("present needs one (voxels,) mask per run")
    read = depth_interpolation(depth, K)
    unique, inverse = np.unique(labels, return_inverse=True)
    use = np.ones(V, bool) if use is None else np.asarray(use, bool)
    psc_np, nuisance = [], []
    s1, ss, n = np.zeros(V), np.zeros(V), np.zeros(V)
    for run, pr in zip(runs, present, strict=True):
        T = int(run.shape[1])
        rows = run[torch.from_numpy(pr)]
        if bool(invalid_voxel_mask(rows).any()) or not bool((rows.mean(-1) > 0).all()):
            raise ValueError("present voxels need finite data and positive means")
        nuis = construct_polynomial_matrix(T, polort, device, torch.float32)
        psc, _ = project_out_nuisance_per_run(
            100 * (rows / rows.mean(-1, keepdim=True) - 1),
            torch.empty(T, 0, device=device), [nuis], [0], device=device, verbose=False,
        )  # fmt: skip
        full = np.zeros((V, T), np.float32)
        full[pr] = psc.cpu().numpy()
        del rows, psc
        psc_np.append(full)
        nuisance.append(nuis)
        if robust:
            s1[pr] += full[pr].sum(-1)
            ss[pr] += np.square(full[pr], dtype=np.float64).sum(-1)
            n[pr] += T
    # Pooled over the runs that saw the voxel; the unbiased SD the full path takes.
    sd = (
        np.sqrt(np.maximum(ss - s1**2 / np.maximum(n, 1), 0) / np.maximum(n - 1, 1))
        if robust
        else None
    )
    eye = np.eye(K)
    members, kept = [], []
    samples: list[list[np.ndarray]] = [[] for _ in runs]
    sampling: list[list[np.ndarray]] = [[] for _ in runs]
    covs: list[list[np.ndarray]] = [[] for _ in runs]
    for unit in range(len(unique)):
        ix_all = np.flatnonzero((inverse == unit) & use)
        fits = []
        for r in range(R):
            ix = ix_all[present[r][ix_all]]
            W = read[ix].T if voxel_weight is None else read[ix].T * voxel_weight[ix][None]
            if sd is not None and ix.size:
                weight = np.ones(ix.size, np.float32)
                home = read[ix].argmax(1)
                for k in np.unique(home):
                    at = home == k
                    med = np.median(sd[ix[at]])
                    weight[at] = np.minimum(1.0, (med / np.maximum(sd[ix[at]], 1e-12)) ** 2)
                W = W * weight[None]
            mass = W.sum(-1)
            # Every model depth must be observed in this run (see column_profiles).
            fits.append(None if ix.size < 2 * K or mass.min() < 1.0 else (ix, W / mass[:, None]))
        if all(f is None for f in fits):
            continue
        kept.append(unique[unit])
        members.append(ix_all)
        for r, fit in enumerate(fits):
            if fit is None:
                samples[r].append(np.zeros((runs[r].shape[1], K), np.float32))
                sampling[r].append(eye)
                covs[r].append(eye)
            else:
                ix, S = fit
                samples[r].append((S @ psc_np[r][ix]).T)
                sampling[r].append(S @ read[ix])
                covs[r].append(S @ S.T)
    if not kept:
        raise ValueError("no patch spans every model depth; enlarge patches or reduce -depths")
    return ColumnProfiles(
        members,
        [torch.as_tensor(np.stack(x))[None] for x in samples],
        [torch.as_tensor(np.stack(x)) for x in sampling],
        [torch.as_tensor(np.stack(x)) for x in covs],
        read,
        nuisance,
        np.asarray(kept),
        present,
    )


@dataclass
class PhysiologyCV:
    """Leave-one-run-out design prediction for each candidate physiology."""

    candidates: list[P0]
    sse: np.ndarray  # P x U held-out whitened SSE
    total: np.ndarray  # U

    @property
    def pooled_r2(self) -> np.ndarray:
        return 1 - self.sse.sum(1) / self.total.sum()

    @property
    def patch_r2(self) -> np.ndarray:
        return 1 - self.sse / np.maximum(self.total, 1e-30)

    @property
    def best(self) -> int:
        return int(np.argmax(self.pooled_r2))


def physiology_crossval(
    columns: ColumnProfiles,
    candidates: list[P0],
    spec: ModelSpec,
    design_for: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    params: dict[str, torch.Tensor] | None = None,
    microtime_dt: float = 0.1,
    tail_seconds: float = 60.0,
    verbose: bool = True,
) -> PhysiologyCV:
    """Score candidate fixed physiologies by how well each run's events predict it.

    ``design_for(cbf_curve)`` returns per-run ``(T, C)`` regressors for a CBF
    response sampled every ``microtime_dt`` s. Every candidate is scored on the
    same profiles, so the comparison is paired. See ``design_crossval`` for what
    the free amplitudes can and cannot tell apart.
    """
    from fastfuncstuff.laminar.timeseries import cbf_kernel, design_crossval

    if isinstance(columns.profiles, list):
        raise ValueError(
            "physiology CV scores runs against one schedule: runs must be the same length"
        )
    R, U, T, K = columns.profiles.shape
    n_fft = T + int(np.ceil(tail_seconds / spec.TR))
    freq = torch.fft.rfftfreq(n_fft, spec.TR)
    sse = np.empty((len(candidates), U))
    total = np.empty(U)
    for i, p0 in enumerate(
        tqdm(
            candidates, desc="Physiology CV", leave=True, disable=not verbose or len(candidates) < 2
        )
    ):
        cand = replace(spec, N=K, p0=p0)
        full, _ = vascular_transfer(linearized_vasculature(cand, params), freq)
        designs = design_for(cbf_kernel(cand, params, microtime_dt))
        err, tot = design_crossval(
            columns.profiles,
            designs,
            full,
            n_fft,
            nuisance=columns.nuisance,
            sampling=columns.sampling,
            noise_cov=columns.noise_cov,
            device=spec.device,
        )
        sse[i], total[:] = err.numpy(), tot.numpy()
    return PhysiologyCV(list(candidates), sse, total)


VOXEL_GAIN_TERMS = ("local", "transport", "shift")
"""The three per-voxel regressors: the column's local BOLD at the voxel's depth
(free), its transported BOLD (gain, prior 1 = the column model) and that
transport's time derivative (a small timing shift, prior 0)."""
DEFAULT_GAIN_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 1e4)


@dataclass
class VoxelGains:
    removed: list[torch.Tensor]  # per run, V x T_r PSC, detrended
    fit: list[torch.Tensor]  # per run, V x T, the voxel's full model fit
    beta: np.ndarray  # V x 3, VOXEL_GAIN_TERMS; NaN where not fitted
    lam: np.ndarray  # V, chosen lambda
    cv_gain: np.ndarray  # V, held-out R² gained over the column model (largest lambda)


def voxel_gain_removal(
    runs: list[torch.Tensor],
    unit_of: np.ndarray,
    read: np.ndarray,
    grid_transport: torch.Tensor,
    grid_local: torch.Tensor,
    nuisance: list[torch.Tensor],
    *,
    present: list[np.ndarray] | None = None,
    lambdas: tuple[float, ...] = DEFAULT_GAIN_LAMBDAS,
    device: torch.device,
    verbose: bool = True,
) -> VoxelGains:
    """Remove each voxel's own share of its column's transported signal.

    The column fit says WHAT drains through the voxel's depth and when (its
    local and transported BOLD); the voxel says how MUCH: a vein voxel carries
    far more than the column mean. Each voxel regresses its detrended PSC on
    local + transport + d(transport)/dt, the last two ridged toward the column
    model (gain 1, shift 0) with a strength chosen per voxel by leave-one-run-out.
    The local term is fit freely beside them, so a gain only takes variance the
    voxel's local response cannot explain. Removed = gain x transport + shift x
    derivative; the local part and the residual stay.

    ``unit_of`` is each voxel's index into the valid patches (-1 = untouched).
    The voxel's own data is part of its column's profile, mostly through the
    local term: transport at a depth comes from the sources below it.
    """
    R, V = len(runs), runs[0].shape[0]
    K = read.shape[1]
    removed = [torch.zeros_like(r) for r in runs]
    fit = [torch.zeros_like(r) for r in runs]
    beta_out = np.full((V, 3), np.nan, np.float32)
    lam_out = np.full(V, np.nan, np.float32)
    gain_out = np.full(V, np.nan, np.float32)
    todo = np.flatnonzero(unit_of >= 0)
    if todo.size == 0:
        return VoxelGains(removed, fit, beta_out, lam_out, gain_out)
    dtype = accum_dtype(device)
    q = [plinalg.qr(n.to(device=device, dtype=torch.float32))[0] for n in nuisance]

    def detrend(x, r):  # (b, T_r, ...) of run r
        f = x.reshape(x.shape[0], x.shape[1], -1)
        return (f - q[r] @ (q[r].T @ f)).reshape(x.shape)

    prior = torch.tensor([0.0, 1.0, 0.0])
    penalized = torch.tensor([0.0, 1.0, 1.0])
    chunk = estimate_chunk_size(
        todo.size,
        sum(int(r.shape[1]) for r in runs),
        K,
        device,
        operation="laminar_voxel_gain",
        n_designs=len(lambdas),
    )
    for start in tqdm(
        range(0, todo.size, chunk),
        desc="Voxel gains",
        leave=True,
        disable=not verbose or todo.size <= chunk,
    ):
        j = todo[start : start + chunk]
        u = torch.as_tensor(unit_of[j])
        w = torch.as_tensor(read[j], device=device, dtype=torch.float32)
        # Runs are handled one at a time so they may differ in length; only their
        # (b, 3, 3) normal equations meet, in the leave-one-run-out ridge.
        Xs, xtx, xty, yy = [], [], [], []
        seen = (
            None
            if present is None
            else torch.stack([torch.as_tensor(p[j], device=device) for p in present])
        )  # (R, b): which runs observed each voxel; absent rows add nothing below
        for r in range(R):
            raw = runs[r][j].to(device=device, dtype=torch.float32)
            mean = raw.mean(-1, keepdim=True)
            if seen is not None:
                mean = torch.where(seen[r, :, None], mean, torch.ones_like(mean))
            y = detrend(100 * (raw / mean - 1), r)
            del raw
            trans = torch.einsum("bk,btk->bt", w, grid_transport[r][u].to(device))
            local = torch.einsum("bk,btk->bt", w, grid_local[r][u].to(device))
            shift = torch.gradient(trans, dim=-1)[0]
            X = detrend(torch.stack([local, trans, shift], -1), r)
            del local, trans, shift
            if seen is not None:
                y = y * seen[r, :, None]
                X = X * seen[r, :, None, None]
            Xd, yd = X.to(dtype), y.to(dtype)
            xtx.append(torch.einsum("btp,btq->bpq", Xd, Xd))
            xty.append(torch.einsum("btp,bt->bp", Xd, yd))
            yy.append(yd.square().sum(-1))
            Xs.append(X)
        beta, best, sse = loro_prior_ridge(
            torch.stack(xtx), torch.stack(xty), torch.stack(yy), prior, penalized, lambdas,
            n_runs=None if seen is None else seen.sum(0),
            # The pull toward the column model is measured against the column's
            # local response at the voxel's depth. Against the transport's own
            # size it shrank a gain by the same fraction however little drained
            # there, and near white (nothing drains into the deepest depth) a
            # transport gain cancelled the task-shaped local term to soak up the
            # voxel's own late response (x55-177 on ds004539, 2% of the signal
            # removed). Task-locked data cannot tell a vein's delayed drainage
            # from a voxel's own latency, so the model's drainage at that depth
            # decides how far a gain may go: free near pial, held near 1 deep.
            scale=torch.stack(xtx).sum(0)[:, 0, 0],
        )  # fmt: skip
        yy = torch.stack(yy)
        b32 = beta.to(torch.float32)
        for r, X in enumerate(Xs):
            rem = b32[:, 1, None] * X[..., 1] + b32[:, 2, None] * X[..., 2]
            full = rem + b32[:, 0, None] * X[..., 0]
            removed[r][j] = rem.to(removed[r].device).to(removed[r].dtype)
            fit[r][j] = full.to(fit[r].device).to(fit[r].dtype)
        del Xs
        beta_out[j] = beta.float().cpu().numpy()
        lam_out[j] = np.asarray(lambdas, np.float32)[best.cpu().numpy()]
        gain_out[j] = ((sse[-1] - sse[best, torch.arange(len(j), device=sse.device)])
                       / yy.sum(0).clamp_min(1e-30)).float().cpu().numpy()  # fmt: skip
    return VoxelGains(removed, fit, beta_out, lam_out, gain_out)


@dataclass
class SplitSelection:
    scores: torch.Tensor  # (S, U) held-out R² across voxels, NaN where a unit had no split
    leak: np.ndarray  # (U,) A-B cross-spectrum above the model band, NaN likewise
    n_scored: np.ndarray  # (U,) splits that scored each unit


def split_half_selection(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    columns: ColumnProfiles,
    blocks: np.ndarray,
    full: torch.Tensor | None,
    *,
    K: int,
    polort: int,
    strengths: tuple[float, ...],
    tr: float,
    pad: int,
    cutoff_hz: float,
    robust: bool,
    in_columns: np.ndarray | None,
    repeats: int = 2,
    seed: int = 0,
    device: torch.device,
    verbose: bool = True,
    present: list[np.ndarray] | None = None,
    transfer: Callable[[int], tuple[int, torch.Tensor, torch.Tensor]] | None = None,
) -> SplitSelection:
    """Per-unit ridge scores from voxel halves (:func:`split_half_scores`), no repeated runs.

    ``blocks`` labels each voxel's small spatial block; whole blocks go to one half,
    so neighbours that share resampled noise stay together. Each repeat draws a new
    assignment; a unit whose halves do not both span every depth goes unscored.

    Runs of different length or coverage (``present``) are scored one at a time on
    their own frequency grid (``transfer(T) -> (pad, full, target)``) and pooled by
    squared error, so a run where a unit is empty adds nothing rather than a
    perfect score.
    """
    U = len(columns.unit_labels)
    pos = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
    use = np.ones(len(labels), bool) if in_columns is None else np.asarray(in_columns, bool)
    total = torch.zeros(len(strengths), U)
    leak = np.zeros(U)
    count = np.zeros(U, np.int64)
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    for _ in range(repeats):
        side = np.zeros(int(ids.max()) + 1, bool)
        side[ids] = rng.random(ids.size) < 0.5
        halves = []
        for want in (True, False):
            try:
                halves.append(column_profiles(
                    runs, depth, labels, K, polort, device, use=use & (side[blocks] == want),
                    robust=robust, present=present,
                ))  # fmt: skip
            except ValueError:  # no unit spans every depth in this half
                break
        if len(halves) < 2:
            continue
        a, b = halves
        both = np.intersect1d(a.unit_labels, b.unit_labels)
        if both.size == 0:
            continue
        ia = np.searchsorted(a.unit_labels, both)
        ib = np.searchsorted(b.unit_labels, both)
        if isinstance(a.profiles, list):
            assert transfer is not None and isinstance(b.profiles, list)
            err = torch.zeros(len(strengths), both.size)
            tot = torch.zeros(both.size)
            lk = torch.zeros(both.size)
            for r, (pa, pb) in enumerate(zip(a.profiles, b.profiles, strict=True)):
                ya, yb = pa[:, ia], pb[:, ib]
                t_r = ya.double().square().sum((0, 2, 3)) + yb.double().square().sum((0, 2, 3))
                if not bool((t_r > 0).any()):
                    continue
                pad_r, full_r, _ = transfer(int(ya.shape[2]))
                sc_r, lk_r = split_half_scores(
                    (ya, _per_run(a.sampling, r)[ia], _per_run(a.noise_cov, r)[ia]),
                    (yb, _per_run(b.sampling, r)[ib], _per_run(b.noise_cov, r)[ib]),
                    full_r, strengths=strengths, tr=tr, pad=pad_r, cutoff_hz=cutoff_hz,
                    device=device, verbose=verbose,
                )  # fmt: skip
                live = t_r > 0  # a unit empty in this run's halves scores nothing here
                err[:, live] += ((1 - sc_r.double()) * t_r)[:, live].float()
                tot[live] += t_r[live].float()
                lk[live] += (lk_r.double() * t_r)[live].float()
            seen = tot > 0
            sc = torch.full((len(strengths), both.size), float("nan"))
            sc[:, seen] = 1 - err[:, seen] / tot[seen]
            lk = torch.where(seen, lk / tot.clamp_min(1e-30), torch.full_like(lk, float("nan")))
            keep = seen.numpy()
            both, sc, lk = both[keep], sc[:, keep], lk[keep]
        else:
            assert full is not None and isinstance(a.profiles, torch.Tensor)
            assert isinstance(b.profiles, torch.Tensor)
            sc, lk = split_half_scores(
                (a.profiles[:, ia], a.sampling[ia], a.noise_cov[ia]),
                (b.profiles[:, ib], b.sampling[ib], b.noise_cov[ib]),
                full, strengths=strengths, tr=tr, pad=pad, cutoff_hz=cutoff_hz, device=device,
                verbose=verbose,
            )  # fmt: skip
        dst = np.array([pos[int(x)] for x in both])
        total[:, dst] += sc
        leak[dst] += lk.numpy()
        count[dst] += 1
    scores = total / torch.as_tensor(np.maximum(count, 1), dtype=total.dtype)
    scores[:, count == 0] = float("nan")
    out_leak = np.where(count > 0, leak / np.maximum(count, 1), np.nan)
    return SplitSelection(scores, out_leak, count)


def column_snr(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    columns: ColumnProfiles,
    blocks: np.ndarray,
    *,
    K: int,
    polort: int,
    tr: float,
    band: tuple[float, float],
    robust: bool,
    in_columns: np.ndarray | None,
    present: list[np.ndarray] | None = None,
    repeats: int = 2,
    seed: int = 0,
    device: torch.device,
) -> np.ndarray:
    """Each valid patch's signal-to-noise ratio from two halves of its voxels.

    Signal is the in-``band`` cross-power of the halves' depth profiles (what
    both share), noise the power of their half-difference (what each has on its
    own), both summed over depths, in-band frequencies and runs. Whole
    ``blocks`` go to one half so neighbours that share resampled noise do not
    read as signal. No task: every fluctuation the halves share counts. A patch
    no split could cover gets NaN.
    """
    U = len(columns.unit_labels)
    pos = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
    use = np.ones(len(labels), bool) if in_columns is None else np.asarray(in_columns, bool)
    sig, noi = np.zeros(U), np.zeros(U)
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    for _ in range(repeats):
        side = np.zeros(int(ids.max()) + 1, bool)
        side[ids] = rng.random(ids.size) < 0.5
        try:
            a, b = (
                column_profiles(
                    runs,
                    depth,
                    labels,
                    K,
                    polort,
                    device,
                    use=use & (side[blocks] == want),
                    robust=robust,
                    present=present,
                )  # fmt: skip
                for want in (True, False)
            )
        except ValueError:  # no unit spans every depth in one half
            continue
        both = np.intersect1d(a.unit_labels, b.unit_labels)
        ia, ib = np.searchsorted(a.unit_labels, both), np.searchsorted(b.unit_labels, both)
        dst = np.array([pos[int(x)] for x in both if int(x) in pos], np.int64)
        keep = np.array([int(x) in pos for x in both])
        pa = (
            a.profiles
            if isinstance(a.profiles, list)
            else [a.profiles[r : r + 1] for r in range(len(runs))]
        )
        pb = (
            b.profiles
            if isinstance(b.profiles, list)
            else [b.profiles[r : r + 1] for r in range(len(runs))]
        )
        for ya, yb in zip(pa, pb, strict=True):
            T = ya.shape[2]
            f = np.fft.rfftfreq(T, tr)
            inb = torch.as_tensor((f >= band[0]) & (f <= band[1]))
            fa = torch.fft.rfft(ya[0, ia][keep].double(), dim=1)[:, inb]
            fb = torch.fft.rfft(yb[0, ib][keep].double(), dim=1)[:, inb]
            sig[dst] += (fa * fb.conj()).real.sum((1, 2)).numpy()
            noi[dst] += ((fa - fb).abs().square() / 4).sum((1, 2)).numpy()
    out = np.full(U, np.nan)
    ok = noi > 0
    out[ok] = sig[ok] / noi[ok]
    return out


@dataclass
class VolumeTdvResult:
    corrected: list[torch.Tensor]  # input units, V x T per run, original means retained
    removed_psc: list[torch.Tensor]  # V x T
    smooth_psc: list[torch.Tensor]  # full-model reconstruction, V x T, smoothing control
    regularization: np.ndarray  # V; NaN for patches not spanning enough depths
    cv_r2: np.ndarray
    noise_gain: np.ndarray
    valid: np.ndarray
    profiles: torch.Tensor | list[torch.Tensor]  # R x U x T x K (per run if ragged), GE only
    gains: VoxelGains | None = None  # removal="voxel" only; its removed/fit are above
    split_leak: np.ndarray | None = None  # V; ridge_cv="split" only
    snr: np.ndarray | None = None  # V; ridge_snr only: the patch SNR that set its ridge
    snr_level: float | None = None  # ridge_snr only: the C used (chosen, with "auto")
    snr_level_at_edge: bool = False  # "auto" chose an end of SNR_RIDGE_GRID
    lag_gate: np.ndarray | None = None  # per LAG_GATE_EDGES band, the removal share applied


def devein_volume_runs(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    spec: ModelSpec,
    *,
    params: dict[str, torch.Tensor] | None = None,
    target: str = "local",
    transfer_mode: str = "dynamic",
    strengths: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    pad_seconds: float = 60.0,
    cutoff_hz: float = 0.1,
    polort: int = 2,
    columns: ColumnProfiles | None = None,
    removal: str = "column",
    in_columns: np.ndarray | None = None,
    gain_lambdas: tuple[float, ...] = DEFAULT_GAIN_LAMBDAS,
    ridge_cv: str = "loro",
    split_blocks: np.ndarray | None = None,
    split_repeats: int = 2,
    present: list[np.ndarray] | None = None,
    voxel_weight: np.ndarray | None = None,
    ridge_snr: float | str | None = None,
    snr_band: tuple[float, float] = (0.01, 0.05),
    lag_band: tuple[float, float] = (0.01, 0.03),
    lag_gate: bool = False,
    verbose: bool = True,
) -> VolumeTdvResult:
    """Estimate patch transport jointly, subtract from individual voxel data.

    Multi-strength selection assumes matched task timing across equal-length
    runs. SE data must never enter this function or select its hyperparameters.
    Insufficient-depth patches pass through unchanged and are marked invalid.
    ``columns`` reuses profiles already built (with the same ``polort``).

    ``removal="column"`` gives every voxel its column's transported BOLD at its
    depth; ``"voxel"`` scales it per voxel (:func:`voxel_gain_removal`). Only
    ``in_columns`` voxels (default all) build the column profiles; with
    ``"voxel"`` the others (a shell around the ribbon) are corrected too.

    With ``"voxel"`` the column profiles are precision-weighted (see
    :func:`column_profiles`), so the column describes its typical voxel.

    ``ridge_cv="split"`` chooses each patch's strength across voxel halves
    (:func:`split_half_selection`, by ``split_blocks``) instead of across runs, so
    the runs need not share a schedule; a patch no split could score takes the
    strength most patches chose.

    ``lag_gate`` scales the removal at each frequency by the share of the model's
    transport lag the data actually show there (:func:`transport_lag_share`).
    """
    if pad_seconds < 0:
        raise ValueError("padding must be nonnegative")
    if removal not in ("column", "voxel"):
        raise ValueError("removal must be column or voxel")
    if ridge_cv not in ("loro", "split"):
        raise ValueError("ridge_cv must be loro or split")
    if ridge_cv == "split" and len(strengths) > 1 and split_blocks is None:
        raise ValueError("ridge_cv split needs split_blocks")
    if removal == "voxel" and len(runs) < 2:
        raise ValueError("voxel gains are chosen by leave-one-run-out: need >=2 runs")
    device = spec.device
    K = spec.K
    if columns is None:
        columns = column_profiles(
            runs, depth, labels, K, polort, device, use=in_columns,
            robust=removal == "voxel", present=present, voxel_weight=voxel_weight,
        )  # fmt: skip
    partial = columns.present is not None
    if present is not None and not all(bool(np.all(p)) for p in present) and not partial:
        raise ValueError("columns were built without the runs' present masks")
    V = runs[0].shape[0]
    lengths = [int(r.shape[1]) for r in runs]
    ragged = len(set(lengths)) > 1
    if (ragged or partial) and len(strengths) > 1 and ridge_cv != "split" and ridge_snr is None:
        raise ValueError(
            "runs of different lengths or coverage choose between several ridge strengths "
            "by voxel halves (ridge_cv='split'); leave-one-run-out compares runs sample by sample"
        )
    means = [r.mean(-1, keepdim=True) for r in runs]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    removed = [torch.zeros_like(r) for r in runs]
    smooth = [torch.zeros_like(r) for r in runs]
    diagnostics = [np.full(V, np.nan, np.float32) for _ in range(3)]
    valid = np.zeros(V, bool)
    profiles = columns.profiles
    system = linearized_vasculature(replace(spec, N=K), params)
    if transfer_mode not in ("dynamic", "static"):
        raise ValueError("transfer_mode must be dynamic or static")

    def transfer(T):
        """Each run length has its own padded frequency grid."""
        pad = min(int(round(pad_seconds / spec.TR)), T - 1)
        frequencies = torch.fft.rfftfreq(T + 2 * pad, spec.TR)
        if transfer_mode == "static":
            frequencies = torch.zeros_like(frequencies)
        return pad, *vascular_transfer(system, frequencies, target)

    scores, leak_u, unscored_np, snr_v, snr_level = None, None, None, None, None
    snr_edge = False
    gate_rho, gate_for = None, None

    def make_gate(weight):
        nonlocal gate_rho, gate_for
        gate_rho, gate_for = transport_lag_share(profiles, transfer, weight, tr=spec.TR)
        if verbose:
            print("  Lag gate (share of the model's transport lag present, by band): "
                  + "  ".join(f"{lo:g}-{hi:g} Hz {r:.2f}" for (lo, hi), r in
                              zip(itertools.pairwise(LAG_GATE_EDGES), gate_rho, strict=True)))  # fmt: skip

    if lag_gate and ridge_snr is None:
        make_gate(np.ones(len(columns.unit_labels)))
    if ridge_snr is not None:
        # Wiener-like shrinkage: each patch's ridge is C x its noise/signal ratio,
        # so a quiet patch (little shared signal, mostly noise) is shrunk hard and
        # corrected little, a driven one corrected fully. The spatial pattern comes
        # from the data's own SNR; C sets the overall level.
        assert split_blocks is not None, "ridge_snr needs split_blocks"
        snr = column_snr(
            runs, depth, labels, columns, split_blocks, K=K, polort=polort, tr=spec.TR,
            band=snr_band, robust=removal == "voxel", in_columns=in_columns,
            present=columns.present, repeats=split_repeats, device=device,
        )  # fmt: skip
        fallback = float(np.nanmedian(snr)) if np.isfinite(snr).any() else 1.0
        snr_f = np.maximum(np.where(np.isfinite(snr), snr, fallback), 1e-6)
        if lag_gate:
            make_gate(np.clip(np.where(np.isfinite(snr), snr, 0.0), 0.0, 20.0))
        strengths = tuple(float(x) for x in np.geomspace(1e-2, 1e3, 51))

        def snr_scores(c: float) -> torch.Tensor:
            lam = np.clip(c / snr_f, 1e-2, 1e3)
            pick = np.abs(np.log(lam)[None] - np.log(strengths)[:, None]).argmin(0)
            out = torch.zeros(len(strengths), len(lam))
            out[pick, np.arange(len(lam))] = 1.0
            return out

        if ridge_snr == "auto":
            # The level that differs between datasets (C fitted on one session
            # over-corrected the next) is chosen per dataset: the C whose corrected
            # column profiles come closest to the depth lag the model predicts
            # survives, pooled over patches with SNR x power weights. Neural timing
            # is instantaneous at this scale, so once ascending transport is gone a
            # column's depths differ only by the local veins' own delay, which the
            # model predicts (_local_lag). No task, no reference; pooled over
            # hundreds of patches the lag is stable where one patch's is noise.
            weight = np.clip(np.where(np.isfinite(snr), snr, 0.0), 0.0, 20.0)
            lag_target = _local_lag(transfer, lengths[0], spec.TR, lag_band)
            grads = {}
            for c in SNR_RIDGE_GRID:
                grads[c] = _pooled_depth_lag(
                    profiles, columns, transfer, snr_scores(c), weight, strengths=strengths,
                    tr=spec.TR, cutoff_hz=cutoff_hz, band=lag_band, device=device,
                    gate_for=gate_for,
                )  # fmt: skip
            c_best = min(grads, key=lambda c: abs(grads[c] - lag_target))
            snr_edge = c_best in (SNR_RIDGE_GRID[0], SNR_RIDGE_GRID[-1])
            if verbose:
                print(f"  SNR ridge, auto level: target pial-deep lag {lag_target:+.2f} s (the local "
                      "veins' own, kept by the target); corrected lag by C: "
                      + "  ".join(f"{c:g}:{g:+.2f}" for c, g in grads.items())
                      + f"  -> C = {c_best:g}")  # fmt: skip
            if snr_edge:
                side = "strongest" if c_best == SNR_RIDGE_GRID[-1] else "weakest"
                print(f"  WARNING: the chosen level is the grid's {side} shrinkage: no level "
                      "brings the corrected depth lag to the model's, so the model does not "
                      "describe these data well and the correction "
                      + ("is held to a light touch" if side == "strongest" else "may be too weak"))  # fmt: skip
            ridge_snr = c_best
        scores = snr_scores(float(ridge_snr))
        snr_level = float(ridge_snr)
        pos_u = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
        unit_v = np.array([pos_u.get(int(lab), -1) for lab in labels], np.int64)
        snr_v = np.where(unit_v >= 0, snr[np.maximum(unit_v, 0)], np.nan).astype(np.float32)
        if verbose:
            q = np.nanpercentile(snr, [25, 50, 75])
            print(f"  SNR ridge: patch SNR quartiles {q[0]:.2f} {q[1]:.2f} {q[2]:.2f} -> ridge "
                  + " ".join(f"{snr_level / x:.2g}" for x in q[::-1]))  # fmt: skip
    elif ridge_cv == "split" and len(strengths) > 1:
        assert split_blocks is not None
        pad0, full0 = (0, None) if (ragged or partial) else transfer(lengths[0])[:2]
        sel = split_half_selection(
            runs, depth, labels, columns, split_blocks, full0, K=K, polort=polort,
            strengths=strengths, tr=spec.TR, pad=pad0, cutoff_hz=cutoff_hz,
            robust=removal == "voxel", in_columns=in_columns, repeats=split_repeats,
            device=device, verbose=verbose, present=columns.present, transfer=transfer,
        )  # fmt: skip
        scores, leak_u = sel.scores, sel.leak
        unscored = ~torch.isfinite(scores).all(0)
        unscored_np = unscored.numpy()
        if unscored.all():
            raise ValueError("no patch could be split into two halves spanning every depth")
        if unscored.any():
            mode = int(torch.bincount(scores[:, ~unscored].argmax(0)).argmax())
            scores[:, unscored] = 0.0
            scores[mode, unscored] = 1.0
        if verbose:
            print(
                f"  Split-half ridge: {int((~unscored).sum()):,}/{len(unscored):,} patches scored; "
                f"median leak above the model band {np.nanmedian(leak_u):.3f}"
            )

    if ragged or partial:
        # Each run's inverse is independent once the strength is fixed (one value,
        # or each patch's split-half choice), so every run is solved on its own
        # grid and its own voxels; nothing is padded or shared.
        parts = []
        for r, prof in enumerate(profiles):
            pad, full_r, desired_r = transfer(prof.shape[2])
            parts.append(
                dynamic_inverse(
                    prof,
                    full_r,
                    desired_r,
                    sampling=_per_run(columns.sampling, r),
                    noise_cov=_per_run(columns.noise_cov, r),
                    strengths=strengths,
                    tr=spec.TR,
                    pad=pad,
                    cutoff_hz=cutoff_hz,
                    device=device,
                    scores=scores,
                    gate=None if gate_for is None else gate_for(prof.shape[2]),
                    verbose=verbose,
                )  # fmt: skip
            )
        weight = torch.tensor(lengths, dtype=parts[0].noise_gain.dtype)
        result = DynamicResult(
            [p.correction[0] for p in parts],
            [p.predicted[0] for p in parts],
            [p.grid_correction[0] for p in parts],
            [p.grid_predicted[0] for p in parts],
            parts[0].regularization,
            # With split-half scores every run shares them; across runs, a held-out
            # prediction needs runs that share a schedule, so it is undefined here.
            parts[0].cv_r2 if scores is not None else torch.full_like(parts[0].cv_r2, float("nan")),
            (torch.stack([p.noise_gain for p in parts]) * weight[:, None]).sum(0) / weight.sum(),
        )
    else:
        pad, full, desired = transfer(lengths[0])
        result = dynamic_inverse(
            profiles, full, desired, sampling=columns.sampling, noise_cov=columns.noise_cov,
            strengths=strengths, tr=spec.TR, pad=pad, cutoff_hz=cutoff_hz, device=device,
            scores=scores, gate=None if gate_for is None else gate_for(lengths[0]),
            verbose=verbose,
        )  # fmt: skip
    leak_v = None
    if leak_u is not None and unscored_np is not None:
        result.cv_r2[torch.as_tensor(unscored_np)] = float("nan")  # chosen by the majority
        pos_u = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
        unit_v = np.array([pos_u.get(int(lab), -1) for lab in labels], np.int64)
        leak_v = np.where(unit_v >= 0, leak_u[np.maximum(unit_v, 0)], np.nan).astype(np.float32)
    out = _finish(
        result, runs, columns, labels, removal, gain_lambdas, device, verbose,
        means, starts, removed, smooth, diagnostics, valid, leak_v,
    )  # fmt: skip
    out.snr = snr_v
    out.snr_level = snr_level
    out.snr_level_at_edge = snr_edge
    out.lag_gate = gate_rho
    return out


SNR_RIDGE_GRID = (0.5, 0.75, 1, 1.5, 2, 3, 4.5, 6, 9, 13, 20, 30, 45, 70, 100, 150, 220, 300)
"""Levels C that ``ridge_snr="auto"`` scores (ridge = C x noise/signal)."""


def _spectral_lags(F: np.ndarray, f: np.ndarray, band: tuple[float, float]):
    """Spectra ``(U, nf, K)`` on frequencies ``f`` -> each depth's lag behind the
    column mean (s) ``(U, K)`` from the cross-spectral phase slope in ``band``, and
    each column's in-band power ``(U,)``."""
    inb = (f >= band[0]) & (f <= band[1])
    ref = F.mean(2, keepdims=True)
    S = (F * ref.conj())[:, inb]
    w, phi, fb = np.abs(S), np.angle(S), f[inb][None, :, None]
    lag = -(w * phi * fb).sum(1) / (2 * np.pi * (w * fb**2).sum(1) + 1e-30)
    return lag, (np.abs(ref[:, inb, 0]) ** 2).sum(1)


def _depth_lags(y: np.ndarray, tr: float, band: tuple[float, float]):
    """``(U, T, K)`` column profiles -> depth lags behind the column mean (s) and
    in-band power (0 for an empty profile); see :func:`_spectral_lags`."""
    return _spectral_lags(np.fft.rfft(y, axis=1), np.fft.rfftfreq(y.shape[1], tr), band)


def _local_lag(transfer, T: int, tr: float, band: tuple[float, float]) -> float:
    """Pial-minus-deep lag (s) the model predicts survives the correction: each
    depth's own (local) response to one drive common to every depth. With target
    "local" the local venous part, larger and slower toward pial, is kept by
    design, so a perfect correction leaves this lag, not zero."""
    pad, _, desired = transfer(T)
    f = np.fft.rfftfreq(T + 2 * pad, tr)
    r = desired.sum(-1).cpu().numpy()[None]  # (1, nf, K): common unit drive at every depth
    lag, _ = _spectral_lags(r, f, band)
    return float(lag[0, 0] - lag[0, -1])


LAG_GATE_EDGES = (0.004, 0.02, 0.035, 0.05, 0.065, 0.08, 0.1, 0.13, 0.25)
"""Bands (Hz) in which :func:`transport_lag_share` measures the data's depth lag."""


def transport_lag_share(profiles, transfer, weight, *, tr: float, edges=LAG_GATE_EDGES):
    """How much of the model's ascending-transport lag the data show, by frequency.

    Blood carried up the column delays pial behind deep by the same time at every
    frequency (the model: ~0.7 s). Where the measured pial-minus-deep lag of the
    raw column profiles, pooled over patches (``weight`` x in-band power) and runs,
    is only a share of that, only that share of the fluctuations behaves like
    transported blood; the rest (noise, motion, anything common to all depths at
    once) has none, and removing transport from it pushes pial ahead of deep. The
    removal is scaled by that share, clipped to [0, 1]. Nothing is fitted to a
    target lag: the data's own lag sets the share, so the level that differed
    between datasets (block runs carry their power where the share is high,
    event runs where it is low) follows the data.

    Returns the share per band and ``gate_for(T)``, the share interpolated over
    band centres onto run length T's padded frequency grid.
    """
    run_profiles = (
        profiles
        if isinstance(profiles, list)
        else [profiles[r : r + 1] for r in range(len(profiles))]
    )
    bands = list(itertools.pairwise(edges))
    num, den = np.zeros(len(bands)), np.zeros(len(bands))
    for prof in run_profiles:
        y = prof[0].cpu().double().numpy()
        F, f = np.fft.rfft(y, axis=1), np.fft.rfftfreq(y.shape[1], tr)
        seen = np.abs(y).sum((1, 2)) > 0  # a patch this run did not observe is all zero
        for b, band in enumerate(bands):
            lag, power = _spectral_lags(F, f, band)
            w = weight * power * seen
            num[b] += (lag[:, 0] - lag[:, -1]) @ w
            den[b] += w.sum()
    pad, full, _ = transfer(run_profiles[0].shape[2])
    fm = np.fft.rfftfreq(run_profiles[0].shape[2] + 2 * pad, tr)
    model = np.array(
        [(lambda x: x[0, 0] - x[0, -1])(_spectral_lags(full.sum(-1).cpu().numpy()[None], fm, band)[0])
         for band in bands]
    )  # fmt: skip
    share = np.clip(num / np.maximum(den, 1e-30) / np.maximum(model, 1e-6), 0.0, 1.0)
    centres = np.array([(lo + hi) / 2 for lo, hi in bands])

    def gate_for(T: int) -> torch.Tensor:
        pad_t = transfer(T)[0]
        return torch.as_tensor(np.interp(np.fft.rfftfreq(T + 2 * pad_t, tr), centres, share))

    return share, gate_for


def _pooled_depth_lag(
    profiles,
    columns,
    transfer,
    scores,
    weight,
    *,
    strengths,
    tr,
    cutoff_hz,
    band,
    device,
    gate_for=None,
) -> float:
    """Pial-minus-deep lag (s) of the corrected column profiles, pooled over runs and
    patches (``weight`` x in-band power). Profiles index depth 0 = pial."""
    run_profiles = (
        profiles
        if isinstance(profiles, list)
        else [profiles[r : r + 1] for r in range(len(profiles))]
    )
    num, den = 0.0, 0.0
    for r, prof in enumerate(run_profiles):
        pad, full_r, desired_r = transfer(prof.shape[2])
        res = dynamic_inverse(
            prof, full_r, desired_r, sampling=_per_run(columns.sampling, r),
            noise_cov=_per_run(columns.noise_cov, r), strengths=strengths, tr=tr, pad=pad,
            cutoff_hz=cutoff_hz, device=device, scores=scores, verbose=False,
            gate=None if gate_for is None else gate_for(prof.shape[2]),
        )  # fmt: skip
        y = (prof[0] - res.correction[0]).double().numpy()
        lag, power = _depth_lags(y, tr, band)
        w = weight * power
        num = num + (lag[:, 0] - lag[:, -1]) @ w
        den += float(w.sum())
    return float(num / max(den, 1e-30))


def _per_run(x, r):
    """A per-run list's entry r, or the one tensor every run shares."""
    return x[r] if isinstance(x, list) else x


def _finish(
    result, runs, columns, labels, removal, gain_lambdas, device, verbose,
    means, starts, removed, smooth, diagnostics, valid, leak_v,
) -> VolumeTdvResult:  # fmt: skip
    """Turn the column inverse into per-voxel removal, for either run layout.

    ``result``'s grids index by run first (one tensor, or one per run when the
    runs differ in length), so ``grid[r][u]`` is run r, patch u.
    """
    R, V, K = len(runs), runs[0].shape[0], columns.read.shape[1]
    lengths = [int(r.shape[1]) for r in runs]
    members, read, nuisance, profiles = (
        columns.members, columns.read, columns.nuisance, columns.profiles,
    )  # fmt: skip
    if removal == "voxel":
        pos = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
        unit_of = np.array([pos.get(int(lab), -1) for lab in labels], np.int64)
        gains = voxel_gain_removal(
            runs,
            unit_of,
            read,
            result.grid_correction,
            [p - c for p, c in zip(result.grid_predicted, result.grid_correction, strict=True)],
            nuisance,
            present=columns.present,
            lambdas=gain_lambdas,
            device=device,
            verbose=verbose,
        )
        valid = unit_of >= 0
        for out, val in zip(
            diagnostics, (result.regularization, result.cv_r2, result.noise_gain), strict=True
        ):
            out[valid] = val.cpu().numpy()[unit_of[valid]]
        corrected = [
            r - rem * m / 100 for r, rem, m in zip(runs, gains.removed, means, strict=True)
        ]
        return VolumeTdvResult(
            corrected, gains.removed, gains.fit, *diagnostics, valid, profiles, gains, leak_v
        )
    chunk = estimate_chunk_size(V, max(lengths), K, device, operation="paired_corr", n_trials=4)
    for u in tqdm(
        range(len(members)),
        desc="Map correction to voxels",
        leave=True,
        disable=not verbose or len(members) < 10,
    ):
        ix = members[u]
        rem = [g[u].to(device) for g in result.grid_correction]
        fit = [g[u].to(device) for g in result.grid_predicted]
        for start in range(0, len(ix), chunk):
            j = ix[start : start + chunk]
            w = torch.as_tensor(read[j], device=device)
            for r in range(R):
                removed[r][j] = (w @ rem[r].T).to(removed[r].device).to(removed[r].dtype)
                smooth[r][j] = (w @ fit[r].T).to(smooth[r].device).to(smooth[r].dtype)
        valid[ix] = True
        for out, val in zip(
            diagnostics,
            (result.regularization[u], result.cv_r2[u], result.noise_gain[u]),
            strict=True,
        ):
            out[ix] = float(val)
    # The correction acts on deviations. Keep the acquired baseline and the
    # user-specified polynomial subspace exactly, including reflected-end effects.
    rem_concat, _ = project_out_nuisance_per_run(
        torch.cat(removed, -1),
        torch.empty(sum(lengths), 0, device=device),
        nuisance,
        starts,
        device=device,
        verbose=False,
    )
    removed = list(rem_concat.split(lengths, dim=-1))
    if columns.present is not None:  # a run that did not observe a voxel takes nothing from it
        for r, pr in enumerate(columns.present):
            gone = torch.from_numpy(~pr)
            removed[r][gone] = 0
            smooth[r][gone] = 0
    corrected = [r - rem * m / 100 for r, rem, m in zip(runs, removed, means, strict=True)]
    return VolumeTdvResult(
        corrected, removed, smooth, *diagnostics, valid, profiles, split_leak=leak_v
    )
