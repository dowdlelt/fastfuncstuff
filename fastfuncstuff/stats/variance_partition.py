"""Per-voxel variance partitioning for fully crossed factorial designs.

Method and rationale: ``../fmri_wiki/concepts/Variance partitioning.md``.

Given single-trial betas from an exhaustively crossed design (the motivating case:
21 tasks x 20 stimuli, 3 repeats per cell), split the reliable response into what only
factor A explains, what only factor B explains, what they share, and what lives in their
interaction -- all cross-validated over held-out repeats.

Three facts about this design shape the whole implementation:

1. **Shared variance is zero by construction.** Exhaustive crossing makes the factor
   column spaces orthogonal, so a nonzero ``C`` means the design lost its balance
   downstream (censoring, dropped trials). It is reported as a *diagnostic*, not a result.

2. **The saturated-vs-additive comparison is an estimation-noise comparison, not a
   degrees-of-freedom one.** At 3 repeats under leave-one-repeat-out, the additive model
   averages ~40 trials per estimated level while the saturated model averages 2. That is
   ~20x the prediction noise, and it drives the measured interaction to zero even when it
   is real and large. Unregularized cell means are unusable at realistic repeat counts, so
   every band gets its own shrinkage.

3. **That shrinkage is free, because of (1).** Balance makes ``X^T X`` block diagonal, so
   banded ridge decouples exactly -- no joint hyperparameter grid. With orthonormal
   contrast codes it collapses further to ``X^T X = n*I``, and each band's ridge is a
   single scalar ``gamma_b = n/(n + lambda_b)`` multiplying that band's OLS coefficients.
   That scalar is exactly the fractional-ridge ``frac`` of :mod:`fastfuncstuff.glm.ridge`.

Because predictions are *linear* in the per-band gammas, the optimal gammas come from a
tiny per-voxel least-squares solve rather than a grid search. Selection therefore happens
on the reporting folds and the R2 is mildly optimistic; the partition is a set of
differences under an identical procedure so the optimism is largely common-mode, and the
permutation null (which re-runs selection inside each permutation) calibrates the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import get_device

# Off-diagonal Gram mass above this fraction of the diagonal means the bands are no longer
# orthogonal and the closed-form decoupling in this module does not hold.
BALANCE_TOL = 1e-6


@dataclass
class FactorDesign:
    """Orthonormal contrast coding of a crossed factorial design.

    ``bands`` maps a band name to its trial-level columns. Band names are the factor name
    for a main effect and ``"a:b"`` for an interaction; the intercept is separate because
    it is never shrunk (it is estimated from every trial and is effectively noiseless).
    """

    factor_names: list[str]
    levels: list[list]
    codes: Tensor  # (n_trials, n_factors) int64 level indices
    contrasts: list[Tensor]  # per factor, (n_levels, n_levels - 1) orthonormal
    bands: dict[str, Tensor]  # band name -> (n_trials, n_cols)
    band_order: list[str]
    balanced: bool
    max_offdiag: float
    cell_counts: Tensor  # (n_levels_0, n_levels_1, ...) trials per cell

    @property
    def n_trials(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)


@dataclass
class VarPartResult:
    """Per-voxel partition maps plus the diagnostics needed to trust them."""

    r2: dict[str, Tensor] = field(default_factory=dict)  # model name -> (n_voxels,)
    unique: dict[str, Tensor] = field(default_factory=dict)  # factor name -> (n_voxels,)
    shared: Tensor | None = None
    interaction: Tensor | None = None
    gammas: dict[str, Tensor] = field(default_factory=dict)  # band name -> (n_voxels,)
    preference: Tensor | None = None
    rank_e: Tensor | None = None  # (n_voxels,) CV rank; -1 = below detection floor
    rank_e_raw: Tensor | None = None  # (n_voxels,) unmasked argmax, for diagnostics
    rank_r2: Tensor | None = None  # (n_voxels, max_rank + 1)
    ncsnr: Tensor | None = None
    noise_ceiling: Tensor | None = None
    diagnostics: dict = field(default_factory=dict)


def _orthonormal_contrasts(n_levels: int, dtype: torch.dtype) -> Tensor:
    """Orthonormal contrast basis for one factor: columns span the sum-to-zero subspace.

    Dummy/treatment coding would give neither orthonormal columns nor orthogonality to the
    intercept, and the closed-form band decoupling this module relies on collapses without
    both. QR of ``[1, e_1, ..., e_{L-1}]`` puts the constant in the first Q column, so
    dropping it leaves an orthonormal basis orthogonal to the intercept.
    """
    if n_levels < 2:
        raise ValueError(f"factor needs >= 2 levels, got {n_levels}")
    m = torch.eye(n_levels, dtype=dtype)
    m[:, 0] = 1.0
    q, _ = torch.linalg.qr(m)
    return q[:, 1:].contiguous()


def build_factor_design(
    factor_codes: dict[str, np.ndarray],
    dtype: torch.dtype = torch.float64,
) -> FactorDesign:
    """Build orthonormal contrast bands for a crossed factorial design.

    Parameters
    ----------
    factor_codes
        Maps factor name -> per-trial label array (any dtype; labels are factorized).
        Insertion order sets band order, so it also sets output map naming.
    dtype
        Contrast construction runs in float64 by default -- the Gram is compared against a
        tight tolerance to decide the fast path, and float32 QR noise alone can exceed it.
    """
    if len(factor_codes) < 2:
        raise ValueError("variance partitioning needs at least 2 factors")

    names = list(factor_codes.keys())
    levels: list[list] = []
    code_cols: list[Tensor] = []
    for name in names:
        labels = np.asarray(factor_codes[name])
        uniq, idx = np.unique(labels, return_inverse=True)
        levels.append(list(uniq))
        code_cols.append(torch.as_tensor(idx, dtype=torch.int64))
    codes = torch.stack(code_cols, dim=1)

    contrasts = [_orthonormal_contrasts(len(lv), dtype) for lv in levels]

    # Main-effect bands: each trial takes its level's contrast row.
    bands: dict[str, Tensor] = {}
    band_order: list[str] = []
    for f, name in enumerate(names):
        bands[name] = contrasts[f][codes[:, f], :]
        band_order.append(name)

    # Interaction bands: elementwise products across every column pair. Because each
    # factor's contrast columns sum to zero over levels, these are automatically
    # orthogonal to both parent main effects under balance.
    if len(names) == 2:
        a, b = bands[names[0]], bands[names[1]]
        inter = (a.unsqueeze(2) * b.unsqueeze(1)).reshape(a.shape[0], -1)
        inter_name = f"{names[0]}:{names[1]}"
        bands[inter_name] = inter
        band_order.append(inter_name)
    else:
        raise NotImplementedError(
            f"partition algebra is defined here for 2 factors, got {len(names)}. "
            "Band construction generalizes; the unique/shared/interaction bookkeeping "
            "does not, and guessing at it would be worse than refusing."
        )

    shape = tuple(len(lv) for lv in levels)
    flat = codes[:, 0] * shape[1] + codes[:, 1]
    cell_counts = torch.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)

    # Balance check. Under an exhaustively crossed balanced design every band Gram is
    # n*I and cross-band Grams vanish; departures mean censoring or dropped trials broke
    # the orthogonality everything downstream assumes.
    full = torch.cat(
        [torch.ones(codes.shape[0], 1, dtype=dtype)] + [bands[n] for n in band_order], dim=1
    )
    gram = full.T @ full
    diag = torch.diagonal(gram)
    offdiag = gram - torch.diag(diag)
    max_offdiag = float(offdiag.abs().max() / diag.abs().max())
    balanced = bool(max_offdiag < BALANCE_TOL) and bool(cell_counts.min() > 0)

    return FactorDesign(
        factor_names=names,
        levels=levels,
        codes=codes,
        contrasts=contrasts,
        bands=bands,
        band_order=band_order,
        balanced=balanced,
        max_offdiag=max_offdiag,
        cell_counts=cell_counts,
    )


def derive_repeat_index(design: FactorDesign) -> np.ndarray:
    """Number each trial by how many times its cell has already occurred.

    Used when the sidecar has no explicit ``repeat`` column. Row order is trial order, so
    this reproduces "1st/2nd/3rd presentation of this cell".
    """
    shape = tuple(len(lv) for lv in design.levels)
    flat = (design.codes[:, 0] * shape[1] + design.codes[:, 1]).numpy()
    seen: dict[int, int] = {}
    out = np.empty(len(flat), dtype=np.int64)
    for i, cell in enumerate(flat):
        out[i] = seen.get(int(cell), 0)
        seen[int(cell)] = out[i] + 1
    return out


def build_repeat_folds(
    repeat: np.ndarray,
    run: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Leave-one-repeat-out folds, refusing any fold that leaks a run across the split.

    Every cell is trained and tested exactly once per fold, so the folds stay balanced.
    The constraint that matters is run locality: if two repeats of a cell share a run, then
    run-level nuisance (drift, motion, alertness, shared noise PCs) is present in both the
    training and test side of that cell, which inflates *every* model equally and reads as
    signal. Same trap as fold-local nuisance projection in LORO cross-validation.
    """
    repeat = np.asarray(repeat)
    uniq = np.unique(repeat)
    folds = []
    for r in uniq:
        test = np.flatnonzero(repeat == r)
        train = np.flatnonzero(repeat != r)
        folds.append((train, test))

    diag: dict = {"n_folds": len(folds), "repeat_levels": [int(u) for u in uniq]}
    if run is not None:
        run = np.asarray(run)
        leaks = []
        for fi, (train, test) in enumerate(folds):
            shared = np.intersect1d(np.unique(run[train]), np.unique(run[test]))
            if shared.size:
                leaks.append({"fold": fi, "runs": [str(s) for s in shared.tolist()]})
        diag["run_leaks"] = leaks
        diag["run_locality_ok"] = not leaks
    else:
        diag["run_locality_ok"] = None
    return folds, diag


def _solve_gammas(
    partials: Tensor,  # (n_vox, n_test, n_bands)
    target: Tensor,  # (n_vox, n_test)
    active: list[int],
) -> Tensor:
    """Least-squares per-band shrinkage, clamped to [0, 1].

    Held-out prediction is linear in the gammas, so the optimum is a (n_active x n_active)
    normal-equation solve per voxel -- no grid search. n_active is at most 3 here.
    Clamping is exact whenever the unconstrained optimum is interior, which it is wherever
    the band carries signal; the boundary cases are voxels where a band is pure noise
    (gamma -> 0) or already unshrunk (gamma -> 1).
    """
    p = partials[:, :, active]
    gram = torch.einsum("vtb,vtc->vbc", p, p)
    rhs = torch.einsum("vtb,vt->vb", p, target)
    # Jitter keeps the solve defined for voxels whose bands are identically zero.
    eye = torch.eye(len(active), dtype=p.dtype, device=p.device)
    scale = torch.diagonal(gram, dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)
    gram = gram + 1e-8 * scale[:, None, None] * eye
    gam = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    return gam.clamp(0.0, 1.0)


def _r2_from_pred(y: Tensor, pred: Tensor) -> Tensor:
    """Coefficient of determination against the voxel's own mean over held-out trials."""
    ss_res = ((y - pred) ** 2).sum(dim=1)
    ss_tot = ((y - y.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _compute_ncsnr(betas: Tensor, cell_flat: Tensor, n_cells: int) -> tuple[Tensor, Tensor]:
    """NSD-style noise ceiling from repeat-to-repeat variability.

    Split-half with Spearman-Brown is unavailable at 3 repeats (there is no even split),
    so the variance-components estimator is the one that works at realistic repeat counts.
    """
    n_vox = betas.shape[0]
    dev, dt = betas.device, betas.dtype
    counts = torch.bincount(cell_flat, minlength=n_cells).to(dt)
    sums = torch.zeros(n_vox, n_cells, dtype=dt, device=dev)
    sums.index_add_(1, cell_flat, betas)
    means = sums / counts.clamp_min(1)
    resid = betas - means[:, cell_flat]

    # Within-cell variance, pooled with the right dof (sum of n_c - 1 over cells).
    dof = float((counts - 1).clamp_min(0).sum())
    noise_var = (resid**2).sum(dim=1) / max(dof, 1.0)
    total_var = betas.var(dim=1, unbiased=True)
    signal_var = (total_var - noise_var).clamp_min(0.0)
    ncsnr = torch.sqrt(signal_var) / torch.sqrt(noise_var.clamp_min(1e-12))

    n_rep = float(counts[counts > 0].float().mean())
    nc = ncsnr**2 / (ncsnr**2 + 1.0 / n_rep)
    return ncsnr, nc


def partition_variance(
    betas: Tensor | np.ndarray,
    factor_codes: dict[str, np.ndarray],
    repeat: np.ndarray | None = None,
    run: np.ndarray | None = None,
    max_rank: int | None = None,
    min_ncsnr_for_rank: float = 0.75,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> VarPartResult:
    """Partition per-voxel variance across two crossed factors.

    Parameters
    ----------
    betas
        (n_voxels, n_trials) single-trial estimates. Column order must match the trial
        table row order. These are *consumed*, not estimated -- see ffs_ridge / GLMsingle.
    factor_codes
        Factor name -> per-trial labels. Exactly two factors.
    repeat
        Per-trial repeat index. Derived from cell-occurrence order when omitted.
    run
        Per-trial run label. Used only to verify fold locality; omitted means unchecked.
    max_rank
        Highest interaction rank to cross-validate. Defaults to the full rank of the
        interaction matrix.
    min_ncsnr_for_rank
        Noise-ceiling SNR below which the interaction rank is reported as ``-1``
        (undetermined) rather than ``0``. Rank selection misses real structure long
        before it invents any, so an unmasked map turns low-SNR tissue into a false
        "task-invariant" region. The 0.75 default is where a rank-1 interaction starts
        being missed >50% of the time on synthetic 20x21x3 data; raise it for a stricter
        map, set it to 0 to disable masking.

    Returns
    -------
    VarPartResult
        Partition maps, per-band shrinkage maps, interaction rank, noise ceiling, and the
        diagnostics that decide whether any of it is trustworthy.
    """
    if device is None:
        device = get_device()

    design = build_factor_design(factor_codes)
    fa, fb = design.factor_names
    inter_name = f"{fa}:{fb}"
    n_trials = design.n_trials

    betas_t = torch.as_tensor(np.asarray(betas)) if not isinstance(betas, Tensor) else betas
    if betas_t.shape[1] != n_trials:
        raise ValueError(
            f"betas has {betas_t.shape[1]} trials but the factor table has {n_trials}; "
            "one row per volume is required, with excluded trials dropped from both."
        )
    betas_t = betas_t.to(torch.float32)
    n_vox = betas_t.shape[0]

    if repeat is None:
        repeat = derive_repeat_index(design)
    folds, fold_diag = build_repeat_folds(repeat, run)
    if fold_diag.get("run_leaks"):
        raise ValueError(
            f"fold construction leaks runs across the train/test split: {fold_diag['run_leaks']}. "
            "Some cell has repeats sharing a run, so run-level nuisance would inflate every "
            "model equally and read as signal. Fix the repeat assignment or drop the trials."
        )

    n_a, n_b = len(design.levels[0]), len(design.levels[1])
    ia, ib = n_a - 1, n_b - 1
    if max_rank is None:
        max_rank = min(ia, ib)
    max_rank = int(min(max_rank, min(ia, ib)))

    band_order = design.band_order
    n_bands = len(band_order)
    band_mats = {k: v.to(device=device, dtype=torch.float32) for k, v in design.bands.items()}
    band_slices = {}
    off = 0
    for name in band_order:
        w = band_mats[name].shape[1]
        band_slices[name] = slice(off, off + w)
        off += w

    models = {
        "M0": [],
        f"M_{fa}": [fa],
        f"M_{fb}": [fb],
        "M_add": [fa, fb],
        "M_full": [fa, fb, inter_name],
    }
    band_index = {name: i for i, name in enumerate(band_order)}

    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=n_trials,
            n_regressors=off + 1,
            device=device,
            operation="xval",
        )

    r2_out = {m: torch.zeros(n_vox) for m in models}
    gam_out = {b: torch.zeros(n_vox) for b in band_order}
    rank_r2_out = torch.zeros(n_vox, max_rank + 1)

    cell_flat = (design.codes[:, 0] * n_b + design.codes[:, 1]).to(device)

    fold_t = [
        (
            torch.as_tensor(tr, dtype=torch.long, device=device),
            torch.as_tensor(te, dtype=torch.long, device=device),
        )
        for tr, te in folds
    ]

    # Per-fold band pseudoinverses. Under leave-one-repeat-out on a balanced design the
    # training set keeps n-1 repeats of every cell, so it stays balanced and each band's
    # solve is a scaled transpose; pinv is used anyway so imbalance degrades gracefully
    # instead of silently returning the wrong coefficients.
    fold_solvers = []
    for tr, _ in fold_t:
        design_tr = torch.cat(
            [torch.ones(len(tr), 1, device=device)] + [band_mats[n][tr] for n in band_order], dim=1
        )
        fold_solvers.append(torch.linalg.pinv(design_tr.double()).float())

    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for c0 in tqdm(
        range(0, n_vox, chunk_size),
        total=n_chunks,
        desc="varpart",
        leave=True,
        disable=not verbose or n_chunks < 2,
    ):
        c1 = min(c0 + chunk_size, n_vox)
        y = betas_t[c0:c1].to(device)
        nvc = c1 - c0

        partials = torch.zeros(nvc, n_trials, n_bands, device=device)
        target = torch.zeros(nvc, n_trials, device=device)

        # Rank sweep accumulates *sufficient statistics* rather than predictions. Holding
        # (n_vox_chunk, n_trials, max_rank + 1) was the peak-memory term by an order of
        # magnitude -- 6 GB of the 9.6 GB peak at a 60k chunk, for values only ever
        # consumed by one R2 reduction. Residual SS per rank plus the target's sum and
        # sum-of-squares give the same R2 in (n_vox_chunk, max_rank + 1).
        rank_ss_res = torch.zeros(nvc, max_rank + 1, device=device)

        for (tr, te), solver in zip(fold_t, fold_solvers, strict=True):
            coef = y[:, tr] @ solver.T  # (nvc, 1 + sum band widths)
            intercept = coef[:, 0:1]
            tgt_fold = y[:, te] - intercept
            target[:, te] = tgt_fold

            for name in band_order:
                sl = band_slices[name]
                cb = coef[:, 1 + sl.start : 1 + sl.stop]
                partials[:, te, band_index[name]] = cb @ band_mats[name][te].T

            # Interaction rank sweep. rank(E) == rank(B_I) because the contrast bases are
            # full column rank, so the SVD runs on the (ia, ib) coefficient matrix rather
            # than the (n_a, n_b) cell-mean matrix -- same answer, smaller decomposition.
            sl = band_slices[inter_name]
            b_i = coef[:, 1 + sl.start : 1 + sl.stop].reshape(nvc, ia, ib)
            u, s, vh = torch.linalg.svd(b_i, full_matrices=False)
            add_pred = partials[:, te, band_index[fa]] + partials[:, te, band_index[fb]]
            inter_te = band_mats[inter_name][te]
            for r in range(max_rank + 1):
                if r == 0:
                    pred_r = add_pred
                else:
                    trunc = (u[:, :, :r] * s[:, None, :r]) @ vh[:, :r, :]
                    pred_r = add_pred + trunc.reshape(nvc, -1) @ inter_te.T
                rank_ss_res[:, r] += ((tgt_fold - pred_r) ** 2).sum(dim=1)

        for mname, active_names in models.items():
            if not active_names:
                pred = torch.zeros_like(target)
            else:
                active = [band_index[n] for n in active_names]
                gam = _solve_gammas(partials, target, active)
                pred = torch.einsum("vtb,vb->vt", partials[:, :, active], gam)
                if mname == "M_full":
                    for k, n in enumerate(active_names):
                        gam_out[n][c0:c1] = gam[:, k].cpu()
            r2_out[mname][c0:c1] = _r2_from_pred(target, pred).cpu()

        ss_tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        rank_r2_out[c0:c1] = (1.0 - rank_ss_res / (ss_tot[:, None] + 1e-12)).cpu()

        del partials, target, rank_ss_res, y

    ncsnr, noise_ceiling = _compute_ncsnr(betas_t.to(device), cell_flat, n_a * n_b)
    ncsnr = ncsnr.cpu()

    # Rank selection is one-sided in its errors. Measured on synthetic 20x21x3 data:
    # a purely additive truth never yields rank >= 1 (0/200 voxels at every noise level
    # tested), and a true rank-1 interaction never inflates to rank >= 2. Every error is
    # a *miss* -- below ncsnr ~0.75 a real rank-1 interaction collapses to rank 0
    # (P(rank=0) 0.005 -> 0.455 -> 0.995 as ncsnr falls 1.49 -> 0.74 -> 0.37).
    #
    # So the rank map needs no permutation null, but it does need a sensitivity floor:
    # unmasked, it prints "task-invariant" over exactly the low-SNR territory (white
    # matter, dropout, edges), which is a spatial artifact that reads as a finding.
    # Voxels below the floor are marked -1 ("undetermined"), never 0.
    rank_e_raw = rank_r2_out.argmax(dim=1)
    rank_e_masked = torch.where(
        ncsnr >= min_ncsnr_for_rank, rank_e_raw, torch.full_like(rank_e_raw, -1)
    )

    u_a = r2_out["M_add"] - r2_out[f"M_{fb}"]
    u_b = r2_out["M_add"] - r2_out[f"M_{fa}"]
    shared = r2_out[f"M_{fa}"] + r2_out[f"M_{fb}"] - r2_out["M_add"]
    interaction = r2_out["M_full"] - r2_out["M_add"]
    denom = u_a + u_b
    preference = torch.where(denom.abs() > 1e-8, (u_b - u_a) / denom, torch.zeros_like(denom))

    diagnostics = {
        "balanced": design.balanced,
        "max_offdiag_gram": design.max_offdiag,
        "fast_path": design.balanced,
        "cells_total": int(design.cell_counts.numel()),
        "cells_empty": int((design.cell_counts == 0).sum()),
        "repeats_min": int(design.cell_counts.min()),
        "repeats_max": int(design.cell_counts.max()),
        "n_levels": {fa: n_a, fb: n_b},
        "shared_abs_median": float(shared.abs().median()),
        "min_ncsnr_for_rank": min_ncsnr_for_rank,
        "rank_undetermined_frac": float((rank_e_masked < 0).float().mean()),
        **fold_diag,
    }
    if not design.balanced:
        diagnostics["warning"] = (
            "design is not balanced: band orthogonality is broken, so the closed-form "
            "decoupling and the C~0 expectation no longer hold. Treat the partition as "
            "approximate and inspect shared variance."
        )

    return VarPartResult(
        r2=r2_out,
        unique={fa: u_a, fb: u_b},
        shared=shared,
        interaction=interaction,
        gammas=gam_out,
        preference=preference,
        rank_e=rank_e_masked,
        rank_e_raw=rank_e_raw,
        rank_r2=rank_r2_out,
        ncsnr=ncsnr,
        noise_ceiling=noise_ceiling.cpu(),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Permutation inference
#
# The statistic is a difference of cross-validated R2 with per-voxel hyperparameter
# selection. It has no tractable null distribution, so inference is by permutation.
#
# What to permute is the subtle part. Shuffling task labels across trials would destroy
# the very balance the partition depends on -- the permuted design would have unequal cell
# counts, band orthogonality would break, and the null statistic would be computed under a
# different estimator variance than the observed one. Permuting *level names* is a no-op
# (the model is invariant to level naming). So the labels stay fixed and the data moves:
# Freedman-Lane permutes the residuals of the reduced model and adds the reduced fit back,
# which nulls the effect under test while preserving everything else and leaving the design
# untouched.
#
#     statistic       reduced model       what the null destroys
#     unique(A)       intercept + B       any effect of A
#     unique(B)       intercept + A       any effect of B
#     interaction     intercept + A + B   the interaction, main effects preserved
#
# Exchangeability blocks are runs: single-trial beta noise carries run-level structure
# (shared drift, motion, noise PCs), so free permutation across runs is anticonservative.
#
# Crucially the per-band gamma selection re-runs inside every permutation, so the null
# absorbs the selection optimism that makes the raw R2 values mildly optimistic.
# ---------------------------------------------------------------------------


@dataclass
class _CellOps:
    """Cell-space operators for the balanced fast path.

    Every model here depends on the data only through per-cell means, so the whole
    computation compresses from n_trials to n_cells columns. The interaction projection is
    never formed directly: the four projections are complementary, so

        P_interaction = I - P_intercept - P_A - P_B

    which replaces a 380-column band with a subtraction. That is what makes a permutation
    null affordable -- the direct route costs roughly ten times more per permutation.
    """

    n_cells: int
    n_a: int
    n_b: int
    d_a: Tensor  # (n_cells, n_a - 1) orthonormal in cell space
    d_b: Tensor  # (n_cells, n_b - 1)
    cell_of_trial: Tensor  # (n_trials,)
    folds: list[tuple[Tensor, Tensor, Tensor, float]]
    n_trials: int


def _build_cell_ops(
    design: FactorDesign,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> _CellOps:
    n_a, n_b = len(design.levels[0]), len(design.levels[1])
    n_cells = n_a * n_b
    cell_of_trial = (design.codes[:, 0] * n_b + design.codes[:, 1]).to(device)

    # Cell-level orthonormal bases. A main-effect column is constant across the other
    # factor, so its cell-space norm picks up a sqrt of that factor's level count.
    cell_a = torch.arange(n_cells, device=device) // n_b
    cell_b = torch.arange(n_cells, device=device) % n_b
    c_a = design.contrasts[0].to(device=device, dtype=torch.float32)
    c_b = design.contrasts[1].to(device=device, dtype=torch.float32)
    d_a = c_a[cell_a, :] / float(n_b) ** 0.5
    d_b = c_b[cell_b, :] / float(n_a) ** 0.5

    fold_ops = []
    for train, test in folds:
        tr = torch.as_tensor(train, dtype=torch.long, device=device)
        te = torch.as_tensor(test, dtype=torch.long, device=device)
        tr_cells = cell_of_trial[tr]
        te_cells = cell_of_trial[te]
        if te_cells.numel() != n_cells or torch.unique(te_cells).numel() != n_cells:
            raise ValueError(
                "the cell-space fast path needs exactly one held-out trial per cell per "
                "fold; this design does not have equal repeats across all cells"
            )
        # Order the held-out trials by cell so column j always means cell j.
        order = torch.argsort(te_cells)
        counts = torch.bincount(tr_cells, minlength=n_cells)
        if int(counts.min()) != int(counts.max()):
            raise ValueError("unequal training repeats per cell; fast path unavailable")
        fold_ops.append((tr, tr_cells, te[order], float(counts[0])))

    return _CellOps(
        n_cells=n_cells,
        n_a=n_a,
        n_b=n_b,
        d_a=d_a,
        d_b=d_b,
        cell_of_trial=cell_of_trial,
        folds=fold_ops,
        n_trials=design.n_trials,
    )


def _cellspace_partials(y: Tensor, ops: _CellOps) -> tuple[Tensor, Tensor]:
    """Per-band held-out predictions and targets, concatenated over folds.

    Returns ``(partials, target)`` of shapes ``(n_vox, n_folds * n_cells, 3)`` and
    ``(n_vox, n_folds * n_cells)``. Band order is (factor A, factor B, interaction).
    """
    n_vox = y.shape[0]
    nc = ops.n_cells
    n_folds = len(ops.folds)
    partials = torch.empty(n_vox, n_folds * nc, 3, device=y.device, dtype=y.dtype)
    target = torch.empty(n_vox, n_folds * nc, device=y.device, dtype=y.dtype)

    for fi, (tr, tr_cells, te_by_cell, n_rep_train) in enumerate(ops.folds):
        m = torch.zeros(n_vox, nc, device=y.device, dtype=y.dtype)
        m.index_add_(1, tr_cells, y[:, tr])
        m = m / n_rep_train
        mu = m.mean(dim=1, keepdim=True)
        p_a = (m @ ops.d_a) @ ops.d_a.T
        p_b = (m @ ops.d_b) @ ops.d_b.T
        sl = slice(fi * nc, (fi + 1) * nc)
        partials[:, sl, 0] = p_a
        partials[:, sl, 1] = p_b
        partials[:, sl, 2] = m - mu - p_a - p_b
        target[:, sl] = y[:, te_by_cell] - mu

    return partials, target


_CELL_MODELS: dict[str, list[int]] = {
    "M0": [],
    "M_a": [0],
    "M_b": [1],
    "M_add": [0, 1],
    "M_full": [0, 1, 2],
}


def _partition_stats(partials: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Cross-validated R2 per nested model plus the three partition statistics."""
    out: dict[str, Tensor] = {}
    gammas: Tensor | None = None
    for name, active in _CELL_MODELS.items():
        if not active:
            pred = torch.zeros_like(target)
        else:
            gam = _solve_gammas(partials, target, active)
            pred = torch.einsum("vtb,vb->vt", partials[:, :, active], gam)
            if name == "M_full":
                gammas = gam
        out[name] = _r2_from_pred(target, pred)

    out["unique_a"] = out["M_add"] - out["M_b"]
    out["unique_b"] = out["M_add"] - out["M_a"]
    out["interaction"] = out["M_full"] - out["M_add"]
    out["shared"] = out["M_a"] + out["M_b"] - out["M_add"]
    if gammas is not None:
        for i, key in enumerate(("gamma_a", "gamma_b", "gamma_interaction")):
            out[key] = gammas[:, i]
    return out


def _reduced_fit_cells(y: Tensor, ops: _CellOps, bands: tuple[int, ...]) -> Tensor:
    """Fitted values of a reduced model, in cell space, using every repeat.

    Freedman-Lane needs the reduced fit on the *whole* dataset (not fold-local): the fold
    structure belongs to the statistic, not to the null.
    """
    n_vox = y.shape[0]
    m = torch.zeros(n_vox, ops.n_cells, device=y.device, dtype=y.dtype)
    m.index_add_(1, ops.cell_of_trial, y)
    counts = torch.bincount(ops.cell_of_trial, minlength=ops.n_cells).to(y.dtype)
    m = m / counts.clamp_min(1)
    fit = m.mean(dim=1, keepdim=True).expand(-1, ops.n_cells).clone()
    if 0 in bands:
        fit = fit + (m @ ops.d_a) @ ops.d_a.T
    if 1 in bands:
        fit = fit + (m @ ops.d_b) @ ops.d_b.T
    return fit


def _within_block_permutation(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(blocks.shape[0])
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        perm[idx] = rng.permutation(idx)
    return perm


@dataclass
class PermutationResult:
    """Observed statistics with uncorrected and FWE-corrected permutation p-values."""

    observed: dict[str, Tensor] = field(default_factory=dict)
    p_uncorrected: dict[str, Tensor] = field(default_factory=dict)
    p_fwe: dict[str, Tensor] = field(default_factory=dict)
    null_max: dict[str, Tensor] = field(default_factory=dict)
    n_perms: int = 0
    diagnostics: dict = field(default_factory=dict)


# Reduced model (as band indices) whose residuals get permuted to null each statistic.
_REDUCED_FOR: dict[str, tuple[int, ...]] = {
    "unique_a": (1,),
    "unique_b": (0,),
    "interaction": (0, 1),
}


def permutation_test(
    betas: Tensor | np.ndarray,
    factor_codes: dict[str, np.ndarray],
    repeat: np.ndarray | None = None,
    run: np.ndarray | None = None,
    statistics: tuple[str, ...] = ("unique_a", "unique_b", "interaction"),
    n_perms: int = 1000,
    seed: int = 0,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> PermutationResult:
    """Freedman-Lane permutation null for the partition statistics.

    Parameters
    ----------
    statistics
        Any of ``"unique_a"`` (unique variance of the first factor), ``"unique_b"``, and
        ``"interaction"``. Each gets its own null, because each has a different reduced
        model.
    n_perms
        Permutations per statistic. FWE p-values are floored at ``1/(n_perms + 1)``.
    run
        Exchangeability blocks. Omitting it permutes freely across all trials, which is
        anticonservative whenever run-level noise structure exists -- so it warns.

    Returns
    -------
    PermutationResult
        Observed statistic, uncorrected p, max-statistic FWE p, and the null max
        distribution, per requested statistic.
    """
    if device is None:
        device = get_device()

    bad = set(statistics) - set(_REDUCED_FOR)
    if bad:
        raise ValueError(f"unknown statistics {sorted(bad)}; expected {sorted(_REDUCED_FOR)}")

    design = build_factor_design(factor_codes)
    if not design.balanced:
        raise ValueError(
            "permutation inference requires a balanced crossed design: the null must be "
            "computed under the same estimator variance as the observed statistic, and "
            "imbalance already invalidates the orthogonality the partition relies on. "
            f"max off-diagonal Gram mass = {design.max_offdiag:.3e}"
        )

    betas_t = torch.as_tensor(np.asarray(betas)) if not isinstance(betas, Tensor) else betas
    if betas_t.shape[1] != design.n_trials:
        raise ValueError(
            f"betas has {betas_t.shape[1]} trials but the factor table has {design.n_trials}"
        )
    betas_t = betas_t.to(torch.float32)
    n_vox = betas_t.shape[0]

    if repeat is None:
        repeat = derive_repeat_index(design)
    folds, fold_diag = build_repeat_folds(repeat, run)
    if fold_diag.get("run_leaks"):
        raise ValueError(f"fold construction leaks runs: {fold_diag['run_leaks']}")

    ops = _build_cell_ops(design, folds, device)

    if run is None:
        blocks = np.zeros(design.n_trials, dtype=np.int64)
        import warnings

        warnings.warn(
            "no run labels: permuting freely across all trials. Single-trial beta noise "
            "carries run-level structure, so this is anticonservative. Pass run= to block.",
            stacklevel=2,
        )
    else:
        blocks = np.asarray(run)

    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=design.n_trials,
            n_regressors=ops.n_cells,
            device=device,
            operation="xval",
        )

    # Observed statistics, computed by the same engine the permutations use so that the
    # comparison is exact rather than merely close.
    observed: dict[str, Tensor] = {k: torch.zeros(n_vox) for k in statistics}
    for c0 in range(0, n_vox, chunk_size):
        c1 = min(c0 + chunk_size, n_vox)
        chunk = betas_t[c0:c1].to(device)
        stats = _partition_stats(*_cellspace_partials(chunk, ops))
        for key in statistics:
            observed[key][c0:c1] = stats[key].cpu()

    rng = np.random.default_rng(seed)
    result = PermutationResult(
        observed=observed,
        n_perms=n_perms,
        diagnostics={
            "n_blocks": int(np.unique(blocks).size),
            "blocked_by_run": run is not None,
            **fold_diag,
        },
    )

    # All permutations are drawn up front and shared across chunks and statistics. Sharing
    # across chunks is required, not an optimisation: the max-statistic null is a maximum
    # over voxels *within* a permutation, so every chunk has to see the same shuffle.
    perms = torch.stack(
        [
            torch.as_tensor(_within_block_permutation(blocks, rng), dtype=torch.long)
            for _ in range(n_perms)
        ]
    ).to(device)

    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for key in statistics:
        reduced = _REDUCED_FOR[key]
        # Running max over voxels, kept on device so the inner loop never synchronises.
        null_max = torch.full((n_perms,), float("-inf"), device=device)
        count_ge = torch.zeros(n_vox, dtype=torch.int32, device=device)
        obs_dev = observed[key].to(device)

        # Chunks outer, permutations inner. The reduced fit and residuals do not depend on
        # the permutation, so this computes them once per chunk instead of once per
        # (chunk, permutation) -- and keeps the chunk resident on the device across all
        # permutations rather than re-uploading it n_perms times.
        with tqdm(
            total=n_chunks * n_perms,
            desc=f"perm[{key}]",
            leave=True,
            disable=not verbose,
        ) as bar:
            for c0 in range(0, n_vox, chunk_size):
                c1 = min(c0 + chunk_size, n_vox)
                chunk = betas_t[c0:c1].to(device)
                fit_cells = _reduced_fit_cells(chunk, ops, reduced)
                fit_trials = fit_cells[:, ops.cell_of_trial]
                resid = chunk - fit_trials
                obs_chunk = obs_dev[c0:c1]
                for p in range(n_perms):
                    y_star = fit_trials + resid[:, perms[p]]
                    stats = _partition_stats(*_cellspace_partials(y_star, ops))
                    s = stats[key]
                    null_max[p] = torch.maximum(null_max[p], s.max())
                    count_ge[c0:c1] += (s >= obs_chunk).to(torch.int32)
                    bar.update(1)
                del chunk, fit_cells, fit_trials, resid

        null_max_cpu = null_max.cpu()
        obs = observed[key]
        result.p_uncorrected[key] = (count_ge.cpu().float() + 1.0) / (n_perms + 1)
        result.p_fwe[key] = ((null_max_cpu[None, :] >= obs[:, None]).sum(dim=1).float() + 1.0) / (
            n_perms + 1
        )
        result.null_max[key] = null_max_cpu

    return result


# ---------------------------------------------------------------------------
# ROI collapsing
# ---------------------------------------------------------------------------


def build_roi_weights(
    atlas: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Turn a parcellation into per-ROI voxel weights.

    Accepts either form an atlas usually ships as:

    * **3-D integer label map** — one ROI per distinct non-zero value. ROIs are
      disjoint, so collapsing is a grouped mean.
    * **4-D stack** — one volume per ROI, values used as weights. Binary masks and
      probabilistic/partial-volume maps both work, and ROIs may overlap.

    Returns ``(roi_index_or_weights, roi_ids, roi_sizes)``. For the 3-D case the first
    element is a ``(n_vox,)`` int array giving each voxel's ROI slot (``-1`` = unassigned),
    which lets the collapse run as an ``index_add`` instead of materialising an
    ``(n_rois, n_vox)`` matrix. For the 4-D case it is that ``(n_rois, n_vox)`` weight
    matrix, which is affordable because overlapping atlases carry far fewer ROIs.
    """
    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    if atlas.ndim == 3:
        flat = atlas.reshape(-1)
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        ids = [int(v) for v in np.unique(flat) if v != 0]
        if not ids:
            raise ValueError("atlas contains no non-zero labels")
        lookup = {v: i for i, v in enumerate(ids)}
        slot = np.full(flat.shape[0], -1, dtype=np.int64)
        for v, i in lookup.items():
            slot[flat == v] = i
        sizes = np.bincount(slot[slot >= 0], minlength=len(ids)).astype(np.float64)
        return slot, ids, sizes

    if atlas.ndim == 4:
        n_rois = atlas.shape[3]
        w = atlas.reshape(-1, n_rois).T.astype(np.float64)  # (n_rois, n_vox_full)
        if mask is not None:
            w = w[:, mask.reshape(-1)]
        w = np.clip(w, 0.0, None)
        sizes = w.sum(axis=1)
        if not np.any(sizes > 0):
            raise ValueError("4-D atlas has no ROI with positive weight inside the mask")
        ids = list(range(1, n_rois + 1))
        return w, ids, sizes

    raise ValueError(f"atlas must be 3-D (labels) or 4-D (per-ROI masks), got {atlas.ndim}-D")


def collapse_to_rois(
    betas: Tensor,
    roi_spec: np.ndarray,
    roi_sizes: np.ndarray,
    device: torch.device | None = None,
    chunk_size: int = 20000,
) -> Tensor:
    """Average single-trial betas within each ROI.

    Returns ``(n_rois, n_trials)``. Empty ROIs come back as zeros rather than NaN so a
    partial atlas cannot poison downstream reductions.

    Collapsing before partitioning is not only a speed lever. Averaging voxels raises the
    noise ceiling roughly as sqrt(n_voxels) for signal shared across the ROI, which
    directly relaxes the ncsnr floor that rank selection needs -- an interaction rank that
    is undetectable per voxel is often resolvable per parcel.
    """
    if device is None:
        device = betas.device
    n_trials = betas.shape[1]
    sizes = torch.as_tensor(roi_sizes, dtype=torch.float32, device=device).clamp_min(1e-12)

    if roi_spec.ndim == 1:
        # Disjoint labels: grouped sum via index_add, no (n_rois, n_vox) matrix.
        slot = torch.as_tensor(roi_spec, dtype=torch.long, device=device)
        n_rois = len(roi_sizes)
        out = torch.zeros(n_rois, n_trials, device=device, dtype=torch.float32)
        keep = slot >= 0
        out.index_add_(0, slot[keep], betas.to(device)[keep].float())
        return out / sizes[:, None]

    # Weighted (possibly overlapping) ROIs: chunk the voxel axis so the weight matrix
    # never has to be resident in full alongside the data.
    w_full = torch.as_tensor(roi_spec, dtype=torch.float32)
    n_rois, n_vox = w_full.shape
    out = torch.zeros(n_rois, n_trials, device=device, dtype=torch.float32)
    for c0 in range(0, n_vox, chunk_size):
        c1 = min(c0 + chunk_size, n_vox)
        out += w_full[:, c0:c1].to(device) @ betas[c0:c1].to(device).float()
    return out / sizes[:, None]


def paint_rois_to_voxels(
    values: Tensor | np.ndarray,
    roi_spec: np.ndarray,
    n_voxels: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Broadcast per-ROI values back onto voxels, for display.

    The result carries no more spatial information than the ROI table it came from --
    every voxel in a parcel gets the same number. It exists so parcel results can be
    rendered on a brain; do not read within-parcel structure into it, because there
    isn't any.

    Overlapping 4-D atlases are resolved by weighted average of the ROIs covering each
    voxel, which reduces to the exact ROI value wherever coverage is disjoint and binary.
    Voxels no ROI covers get *fill*.
    """
    vals = np.asarray(values.cpu() if isinstance(values, Tensor) else values, dtype=np.float64)

    if roi_spec.ndim == 1:
        out = np.full(n_voxels, fill, dtype=np.float32)
        assigned = roi_spec >= 0
        out[assigned] = vals[roi_spec[assigned]].astype(np.float32)
        return out

    w = np.asarray(roi_spec, dtype=np.float64)  # (n_rois, n_vox)
    denom = w.sum(axis=0)
    out = np.full(n_voxels, fill, dtype=np.float32)
    covered = denom > 0
    out[covered] = ((vals @ w)[covered] / denom[covered]).astype(np.float32)
    return out
