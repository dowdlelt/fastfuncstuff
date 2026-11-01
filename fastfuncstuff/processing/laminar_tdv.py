"""Column sampling, continuous vascular inversion, and voxel residual preservation."""

from __future__ import annotations

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
    dynamic_inverse,
    linearized_vasculature,
    loro_prior_ridge,
    split_half_scores,
    vascular_transfer,
)
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import accum_dtype


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
    profiles: torch.Tensor  # R x U x T x K
    sampling: torch.Tensor  # U x K x K, physiological depths -> measured bins
    noise_cov: torch.Tensor  # U x K x K, bin covariance from shared voxels
    read: np.ndarray  # V x K point-sample weights
    nuisance: list[torch.Tensor]  # per run, T x n_poly
    unit_labels: np.ndarray  # U, the label of each valid patch


def column_profiles(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    K: int,
    polort: int,
    device: torch.device,
    use: np.ndarray | None = None,
    robust: bool = False,
) -> ColumnProfiles:
    """Percent change, per-run Legendre drift removed, averaged into K bins per patch.

    Only voxels flagged in ``use`` (default all) build the profiles -- the ribbon,
    not the shell around it. Patches that do not observe every model depth are
    left out (they pass through).

    ``robust`` weights each voxel by min(1, (median SD of its patch's voxels at
    the same depth bin / its own SD)^2). A vein three times as variable as its
    neighbours then counts 1/9: unweighted, it inflates the bin and the inverse
    reads the excess as LOCAL signal, which every other voxel then inherits.
    """
    if not runs or any(r.ndim != 2 or r.shape != runs[0].shape for r in runs):
        raise ValueError("runs must have the same (voxels,time) shape")
    V, T = runs[0].shape
    if len(labels) != V or len(depth) != V or T < 4:
        raise ValueError("depth/labels must match voxels and runs need >=4 samples")
    if not all(torch.isfinite(r).all() and (r.mean(-1) > 0).all() for r in runs):
        raise ValueError("runs need finite data and positive voxel means")
    R = len(runs)
    read = depth_interpolation(depth, K)
    unique, inverse = np.unique(labels, return_inverse=True)
    members, samples, covs, sampling, kept = [], [], [], [], []
    use = np.ones(V, bool) if use is None else np.asarray(use, bool)
    psc = torch.cat([100 * (r / r.mean(-1, keepdim=True) - 1) for r in runs], -1)
    starts = [i * T for i in range(R)]
    nuisance = [construct_polynomial_matrix(T, polort, device, torch.float32) for _ in runs]
    psc, _ = project_out_nuisance_per_run(
        psc, torch.empty(R * T, 0, device=device), nuisance, starts, device=device, verbose=False
    )
    # Spatial means use CPU storage; only the depth inverse is sent to compute.
    psc_np = psc.cpu().numpy().reshape(V, R, T).transpose(1, 0, 2)
    sd = psc.std(-1).cpu().numpy() if robust else None
    del psc
    for unit in range(len(unique)):
        ix = np.flatnonzero((inverse == unit) & use)
        W = read[ix].T
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
        samples.append(np.stack([S @ psc_np[r, ix] for r in range(R)]).transpose(0, 2, 1))
    if not samples:
        raise ValueError("no patch spans every model depth; enlarge patches or reduce -depths")
    return ColumnProfiles(
        members,
        torch.as_tensor(np.stack(samples, axis=1)),
        torch.as_tensor(np.stack(sampling)),
        torch.as_tensor(np.stack(covs)),
        read,
        nuisance,
        np.asarray(kept),
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
    removed: list[torch.Tensor]  # per run, V x T PSC, detrended
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
    R, V, T = len(runs), runs[0].shape[0], runs[0].shape[1]
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

    def detrend(x):  # (R, b, T, ...) per run
        flat = x.reshape(R, x.shape[1], T, -1)
        out = torch.stack([f - qr @ (qr.T @ f) for f, qr in zip(flat, q, strict=True)])
        return out.reshape(x.shape)

    prior = torch.tensor([0.0, 1.0, 0.0])
    penalized = torch.tensor([0.0, 1.0, 1.0])
    chunk = estimate_chunk_size(
        todo.size, R * T, K, device, operation="laminar_voxel_gain", n_designs=len(lambdas)
    )
    for start in tqdm(
        range(0, todo.size, chunk),
        desc="Voxel gains",
        leave=True,
        disable=not verbose or todo.size <= chunk,
    ):
        j = todo[start : start + chunk]
        raw = torch.stack([r[j] for r in runs]).to(device=device, dtype=torch.float32)
        y = detrend(100 * (raw / raw.mean(-1, keepdim=True) - 1))
        del raw
        u = torch.as_tensor(unit_of[j])
        w = torch.as_tensor(read[j], device=device, dtype=torch.float32)
        trans = torch.einsum("bk,rbtk->rbt", w, grid_transport[:, u].to(device))
        local = torch.einsum("bk,rbtk->rbt", w, grid_local[:, u].to(device))
        shift = torch.gradient(trans, dim=-1)[0]
        X = detrend(torch.stack([local, trans, shift], -1))
        del local, trans, shift
        Xd, yd = X.to(dtype), y.to(dtype)
        xtx = torch.einsum("rbtp,rbtq->rbpq", Xd, Xd)
        xty = torch.einsum("rbtp,rbt->rbp", Xd, yd)
        yy = yd.square().sum(-1)
        beta, best, sse = loro_prior_ridge(xtx, xty, yy, prior, penalized, lambdas)
        b32 = beta.to(torch.float32)
        rem = b32[None, :, 1, None] * X[..., 1] + b32[None, :, 2, None] * X[..., 2]
        full = rem + b32[None, :, 0, None] * X[..., 0]
        for r in range(R):
            removed[r][j] = rem[r].to(removed[r].device).to(removed[r].dtype)
            fit[r][j] = full[r].to(fit[r].device).to(fit[r].dtype)
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
    full: torch.Tensor,
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
) -> SplitSelection:
    """Per-unit ridge scores from voxel halves (:func:`split_half_scores`), no repeated runs.

    ``blocks`` labels each voxel's small spatial block; whole blocks go to one half,
    so neighbours that share resampled noise stay together. Each repeat draws a new
    assignment; a unit whose halves do not both span every depth goes unscored.
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
                    robust=robust,
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


@dataclass
class VolumeTdvResult:
    corrected: list[torch.Tensor]  # input units, V x T per run, original means retained
    removed_psc: list[torch.Tensor]  # V x T
    smooth_psc: list[torch.Tensor]  # full-model reconstruction, V x T, smoothing control
    regularization: np.ndarray  # V; NaN for patches not spanning enough depths
    cv_r2: np.ndarray
    noise_gain: np.ndarray
    valid: np.ndarray
    profiles: torch.Tensor  # R x U x T x K, GE only
    gains: VoxelGains | None = None  # removal="voxel" only; its removed/fit are above
    split_leak: np.ndarray | None = None  # V; ridge_cv="split" only


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
    R, K = len(runs), spec.K
    if columns is None:
        columns = column_profiles(
            runs, depth, labels, K, polort, device, use=in_columns, robust=removal == "voxel"
        )
    V, T = runs[0].shape
    members, read, nuisance = columns.members, columns.read, columns.nuisance
    means = [r.mean(-1, keepdim=True) for r in runs]
    starts = [i * T for i in range(R)]
    removed = [torch.zeros_like(r) for r in runs]
    smooth = [torch.zeros_like(r) for r in runs]
    diagnostics = [np.full(V, np.nan, np.float32) for _ in range(3)]
    valid = np.zeros(V, bool)
    profiles = columns.profiles
    pad = min(int(round(pad_seconds / spec.TR)), T - 1)
    system = linearized_vasculature(replace(spec, N=K), params)
    frequencies = torch.fft.rfftfreq(T + 2 * pad, spec.TR)
    if transfer_mode not in ("dynamic", "static"):
        raise ValueError("transfer_mode must be dynamic or static")
    if transfer_mode == "static":
        frequencies = torch.zeros_like(frequencies)
    full, desired = vascular_transfer(system, frequencies, target)
    scores, leak_u = None, None
    if ridge_cv == "split" and len(strengths) > 1:
        assert split_blocks is not None
        sel = split_half_selection(
            runs, depth, labels, columns, split_blocks, full, K=K, polort=polort,
            strengths=strengths, tr=spec.TR, pad=pad, cutoff_hz=cutoff_hz,
            robust=removal == "voxel", in_columns=in_columns, repeats=split_repeats,
            device=device, verbose=verbose,
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

    result = dynamic_inverse(
        profiles,
        full,
        desired,
        sampling=columns.sampling,
        noise_cov=columns.noise_cov,
        strengths=strengths,
        tr=spec.TR,
        pad=pad,
        cutoff_hz=cutoff_hz,
        device=device,
        scores=scores,
        verbose=verbose,
    )
    leak_v = None
    if leak_u is not None:
        result.cv_r2[torch.as_tensor(unscored_np)] = float("nan")  # chosen by the majority
        pos_u = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
        unit_v = np.array([pos_u.get(int(lab), -1) for lab in labels], np.int64)
        leak_v = np.where(unit_v >= 0, leak_u[np.maximum(unit_v, 0)], np.nan).astype(np.float32)
    if removal == "voxel":
        pos = {int(lab): i for i, lab in enumerate(columns.unit_labels)}
        unit_of = np.array([pos.get(int(lab), -1) for lab in labels], np.int64)
        gains = voxel_gain_removal(
            runs,
            unit_of,
            read,
            result.grid_correction,
            result.grid_predicted - result.grid_correction,
            nuisance,
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
    chunk = estimate_chunk_size(V, T, K, device, operation="paired_corr", n_trials=4)
    for u in tqdm(
        range(len(members)),
        desc="Map correction to voxels",
        leave=True,
        disable=not verbose or len(members) < 10,
    ):
        ix = members[u]
        rem = result.grid_correction[:, u].to(device)
        fit = result.grid_predicted[:, u].to(device)
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
        torch.empty(R * T, 0, device=device),
        nuisance,
        starts,
        device=device,
        verbose=False,
    )
    removed = list(rem_concat.split(T, dim=-1))
    corrected = [r - rem * m / 100 for r, rem, m in zip(runs, removed, means, strict=True)]
    return VolumeTdvResult(
        corrected, removed, smooth, *diagnostics, valid, profiles, split_leak=leak_v
    )
