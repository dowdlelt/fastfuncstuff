"""Noise ceilings for GLM tools, in the units their cross-validation reports.

A held-out R2 of 0.08 is uninterpretable on its own. It is an excellent fit in
data whose reproducible signal tops out at 0.09, and a poor one where the
ceiling is 0.4. The ceiling is what turns the number into a judgement, and the
*explainable* R2 -- ``xval_r2 / ceiling`` -- is the judgement itself: the
fraction of the achievable variance the model actually got.

The hard constraint is that a ceiling is only meaningful against an R2 measured
in the same units, on the same held-out data, with the same denominator. These
tools cross-validate in two different spaces, so there are two ceilings:

**Timeseries space** (``-cv_design condition``): train-run betas predict a
held-out run's timecourse, and R2 is scored against that run's total variance.
:func:`loro_two_half_ceiling` estimates the ceiling in exactly those units by
splitting each fold's *training* runs in two, producing two independent
predictions of the held-out run. Their beta-estimation noise is independent, so
their covariance estimates the reproducible signal variance while the noise
cancels. Sharing the folds is what makes it comparable -- same timepoints, same
denominator, same accumulation -- rather than something we have to argue for.

**Beta space** (``-cv_design single``): held-out single-trial betas are
predicted from same-condition training betas. :func:`ncsnr_noise_ceiling` is the
NSD/GLMsingle estimator: repeats of one condition should give identical betas,
so their spread is noise and what is left is signal. It needs repeated
*conditions*, not repeated runs, which ordinary event-related designs have.

:func:`df_corrected_ceiling` is the fallback for timeseries space when there are
too few runs to split the training set. It needs no repeats at all, but it
bounds "the true-beta version of this design" rather than any model at all --
it cannot tell you the design is wrong.

The third estimator, for runs whose stimulus is bit-identical, lives in
:mod:`fastfuncstuff.stats.reliability` alongside the split-half machinery it
came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.glm.xval import _compute_projection_matrix
from fastfuncstuff.memory import estimate_chunk_size


@dataclass
class CeilingResult:
    """A per-voxel ceiling plus what the caller needs to judge whether to trust it."""

    ceiling: torch.Tensor
    """(n_voxels,) ceiling in the R2 units of the matching cross-validation."""

    method: str
    """Which estimator produced it: 'loro_two_half', 'ncsnr', or 'df_corrected'."""

    n_usable: int = 0
    """Folds (timeseries) or conditions (beta space) that contributed."""

    notes: list[str] = field(default_factory=list)
    """Human-readable caveats worth printing: thin splits, dropped conditions."""

    def explainable_r2(self, xval_r2: torch.Tensor, min_ceiling: float = 0.01) -> torch.Tensor:
        """``xval_r2 / ceiling`` -- the fraction of achievable variance captured.

        Values slightly above 1 are noise in the ceiling estimate, not a model
        that beat the ceiling, and are deliberately not clamped: a map that runs
        well above 1 is evidence the ceiling is underestimated, and hiding that
        would cost the diagnostic.

        Voxels whose ceiling falls below ``min_ceiling`` come back NaN rather
        than as a ratio. Where nothing reproduces there is no achievable
        variance to take a fraction *of*, so the quantity is undefined -- and
        dividing by a near-zero ceiling anyway would fill most of a brain with
        values in the hundreds, wrecking the map's scaling and any mean taken
        over it. NaN keeps "undefined here" visible instead of laundering it
        into a number.
        """
        ceiling = self.ceiling.to(xval_r2.device)
        defined = ceiling >= min_ceiling
        return torch.where(defined, xval_r2 / ceiling.clamp_min(min_ceiling), torch.nan)

    def summarize(self, explainable: torch.Tensor | None = None) -> str:
        """One-line summary over the voxels where the ceiling is actually defined.

        Medians over the whole brain are dominated by voxels with no signal, and
        so report a near-zero ceiling and a nonsense explainable fraction no
        matter how well the model did where it mattered.
        """
        finite = self.ceiling[torch.isfinite(self.ceiling)]
        usable = finite[finite >= 0.01]
        if usable.numel() == 0:
            return "no voxel had an estimable ceiling above 0.01"
        parts = [f"{usable.numel():,} voxels with ceiling >= 0.01, median {usable.median():.4f}"]
        if explainable is not None:
            valid = explainable[torch.isfinite(explainable)]
            if valid.numel():
                parts.append(f"median explainable R2 {valid.median():.4f}")
        return "; ".join(parts)


def loro_ceiling_by_voxel_group(
    *,
    data: torch.Tensor,
    nuisance_per_run: list[torch.Tensor],
    run_starts: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    design_matrix: torch.Tensor | None = None,
    designs_by_hrf: dict | None = None,
    hrf_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    zero_event_strategy: str = "zero",
    progress_desc: str = "  Ceiling by HRF",
    show_progress: bool = False,
) -> CeilingResult:
    """A timeseries ceiling for one design, or for a per-voxel-HRF set of designs.

    Wraps the three steps every caller needs in the same order: project the
    nuisance (the *selected noise PCs included*, so the ceiling's denominator is
    the denoised variance the R2 was actually scored against), estimate the
    ceiling on the shared folds, and -- in per-HRF mode -- do that per group and
    merge. Getting the PC block into ``nuisance_per_run`` is the caller's job and
    the easiest thing to get wrong: a ceiling built on undenoised data bounds a
    different quantity than the R2 it is meant to qualify.

    Pass either ``design_matrix`` or ``designs_by_hrf`` + ``hrf_indices``, and
    the same ``zero_event_strategy`` the R2 was scored with -- see
    :func:`loro_two_half_ceiling` for why a mismatch there is not a bound.
    """
    from fastfuncstuff.glm.xval import project_out_nuisance_per_run

    def _one(subset_data: torch.Tensor, subset_design: torch.Tensor) -> CeilingResult:
        projected_data, projected_design = project_out_nuisance_per_run(
            data=subset_data,
            design=subset_design,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=subset_data.device,
        )
        result = loro_two_half_ceiling(
            data=projected_data,
            design_matrix=projected_design,
            run_starts=run_starts,
            stim_indices=list(range(projected_design.shape[1])),
            nuisance_indices=[],  # already projected, per run
            cv_splits=cv_splits,
            device=device,
            zero_event_strategy=zero_event_strategy,
            verbose=False,
        )
        del projected_data, projected_design
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    if designs_by_hrf is None:
        if design_matrix is None:
            raise ValueError("pass either design_matrix or designs_by_hrf")
        return _one(data, design_matrix)

    if hrf_indices is None:
        raise ValueError("designs_by_hrf requires hrf_indices")

    # Each voxel is scored against its own HRF's design, so its ceiling has to
    # come from that design too. The groups partition the voxels and var(y) is
    # design-free, so the maps merge onto one scale; only the design-side
    # factorisations repeat, not the passes over the brain.
    from tqdm import tqdm

    groups: list[tuple[torch.Tensor, CeilingResult]] = []
    group_iter = tqdm(
        torch.unique(hrf_indices).tolist(),
        desc=progress_desc,
        unit="group",
        leave=True,
        disable=not show_progress or len(designs_by_hrf) < 2,
    )
    for hrf_idx in group_iter:
        voxel_mask = hrf_indices == hrf_idx
        groups.append((voxel_mask, _one(data[voxel_mask, :], designs_by_hrf[hrf_idx])))
    return merge_voxel_group_ceilings(groups, n_voxels=data.shape[0])


def merge_voxel_group_ceilings(
    groups: list[tuple[torch.Tensor, CeilingResult]],
    n_voxels: int,
) -> CeilingResult:
    """Combine ceilings estimated separately per voxel group into one map.

    Per-voxel HRF modes fit each group of voxels against its own design, so the
    ceiling has to be estimated per group as well -- a voxel with its own HRF
    has its own design and therefore its own ceiling, which is what a per-voxel
    ceiling *is*. The groups partition the voxels and ``var(y)`` is design-free,
    so the maps stay on one common scale and simply scatter into place.

    The bookkeeping does not scatter. ``n_usable`` counts folds that could be
    split into two halves, which is a property of the design: a group whose
    conditions vanish from a training half loses folds the other groups kept.
    Taking any single group's count would hide that, so the merged count is the
    **minimum** over groups -- the guarantee that actually holds brain-wide --
    and a note records the spread when they disagree.

    ``groups`` is a list of ``(voxel_mask, result)``; masks must be disjoint and
    boolean over ``n_voxels``.
    """
    if not groups:
        raise ValueError("merge_voxel_group_ceilings needs at least one group")

    template = groups[0][1].ceiling
    ceiling = torch.full((n_voxels,), torch.nan, dtype=template.dtype, device=template.device)
    for mask, result in groups:
        ceiling[mask.to(ceiling.device)] = result.ceiling.to(ceiling.device, ceiling.dtype)

    usable = [result.n_usable for _, result in groups]
    notes: list[str] = []
    for _, result in groups:
        for note in result.notes:
            if note not in notes:
                notes.append(note)
    if min(usable) != max(usable):
        notes.append(
            f"folds usable for the ceiling varied across voxel groups "
            f"({min(usable)}-{max(usable)}); the reported count is the minimum"
        )

    methods = {result.method for _, result in groups}
    return CeilingResult(
        ceiling=ceiling,
        method=methods.pop() if len(methods) == 1 else "mixed",
        n_usable=min(usable),
        notes=notes,
    )


def _centered_cov_and_ss(
    first: torch.Tensor, second: torch.Tensor, actual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-voxel sums needed to pool cov(A,B) and var(actual) across folds.

    Returns raw (unnormalised) sums rather than moments so the caller can pool
    folds of unequal length by simple addition -- the same reason
    :func:`compute_xval_r2` accumulates sums instead of per-fold R2 values.
    """
    a = first.to(torch.float64)
    b = second.to(torch.float64)
    y = actual.to(torch.float64)
    return (
        (a * b).sum(dim=1),
        a.sum(dim=1),
        b.sum(dim=1),
        y,
    )


def loro_two_half_ceiling(
    data: torch.Tensor,
    design_matrix: np.ndarray | torch.Tensor,
    run_starts: list[int],
    stim_indices: list[int],
    nuisance_indices: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    device: torch.device | None = None,
    batch_size: int | None = None,
    min_runs_per_half: int = 1,
    zero_event_strategy: str = "zero",
    verbose: bool = True,
) -> CeilingResult:
    """Timeseries R2 ceiling from two independent predictions of held-out data.

    For each fold, the training runs are split into two disjoint halves and
    fitted separately. Both halves' betas predict the *same* held-out run, so
    both predictions carry the same true signal ``s`` and independent beta
    noise::

        y_hat_A = s + e_A,  y_hat_B = s + e_B,  E[e_A e_B] = 0
        cov(y_hat_A, y_hat_B) -> var(s)
        ceiling = var(s) / var(y)

    ``var(y)`` is the held-out data's variance -- the same denominator
    :func:`compute_xval_r2` divides by -- so the result is directly comparable
    to the R2 it is meant to bound. **Pass the same ``cv_splits``, the same
    (equally preprocessed) ``data``, and the same ``zero_event_strategy``**; a
    ceiling built on different folds, differently projected data, or a different
    missing-event policy is not a bound on anything. Under ``'nuisance'`` the R2
    is scored only on the subspace its fold could predict, so the ceiling must
    drop the same subspace or the ratio divides two incommensurate numbers --
    which shows up as an explainable-R2 map running far above 1.

    Runs alternate between halves rather than splitting contiguously, so slow
    session drift that survived nuisance projection is shared by both halves
    instead of loading onto one, which would depress the covariance.

    Folds whose training set cannot supply ``min_runs_per_half`` runs to each
    half are skipped. With three runs total each half is a single run and the
    estimate is noisy but real; below that there is nothing to split and the
    result is all-NaN, so callers should fall back to
    :func:`df_corrected_ceiling`.
    """
    from fastfuncstuff.utils import get_device, to_tensor

    if device is None:
        device = get_device()

    design_matrix = to_tensor(design_matrix, device=device, dtype=torch.float32)
    n_voxels, n_timepoints = data.shape
    run_lengths = np.diff(run_starts + [n_timepoints])
    notes: list[str] = []

    if batch_size is None:
        batch_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_timepoints,
            n_regressors=len(stim_indices) + len(nuisance_indices),
            device=device,
            operation="xval",
        )

    accumulator_device = device if device.type == "cuda" else torch.device("cpu")
    sum_ab = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_a = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_b = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_y = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_y_sq = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    n_points = 0
    n_usable = 0

    for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
        # Alternating assignment, not a contiguous cut: see docstring.
        half_a = list(train_runs[0::2])
        half_b = list(train_runs[1::2])
        if len(half_a) < min_runs_per_half or len(half_b) < min_runs_per_half:
            continue

        test_tps: list[int] = []
        for run_idx in test_runs:
            start = run_starts[run_idx]
            test_tps.extend(range(start, start + run_lengths[run_idx]))

        test_design = design_matrix[test_tps, :]
        q_test = _compute_projection_matrix(test_design, nuisance_indices)
        test_design_clean = (
            test_design - q_test @ (q_test.T @ test_design) if q_test is not None else test_design
        )
        test_stim = test_design_clean[:, stim_indices]

        # The conditions this fold's TRAINING RUNS cannot supply a beta for.
        # compute_xval_r2 under 'nuisance' removes their span from the held-out
        # data and from the design predicting it; the ceiling has to do the same
        # or var(y) here is not the SS_tot the R2 was divided by.
        unpred_basis = None
        if zero_event_strategy == "nuisance":
            from fastfuncstuff.glm.xval import _unpredictable_basis

            train_tps: list[int] = []
            for run_idx in train_runs:
                start = run_starts[run_idx]
                train_tps.extend(range(start, start + run_lengths[run_idx]))
            train_stim = design_matrix[train_tps, :][:, stim_indices]
            test_only = (train_stim.abs().sum(dim=0) <= 1e-10) & (
                test_stim.abs().sum(dim=0) > 1e-10
            )
            unpred_basis = _unpredictable_basis(test_stim, test_only)

        half_fits = []
        usable_half = True
        for half in (half_a, half_b):
            half_tps: list[int] = []
            for run_idx in half:
                start = run_starts[run_idx]
                half_tps.extend(range(start, start + run_lengths[run_idx]))

            half_design = design_matrix[half_tps, :]
            q_half = _compute_projection_matrix(half_design, nuisance_indices)
            half_clean = (
                half_design - q_half @ (q_half.T @ half_design)
                if q_half is not None
                else half_design
            )
            half_stim = half_clean[:, stim_indices]
            half_fits.append((half_tps, half_stim, q_half))
            if not (half_stim.abs().sum(dim=0) > 1e-10).any():
                usable_half = False

        if not usable_half:
            continue

        # A condition absent from either half contributes no signal to the
        # covariance while still sitting in the held-out variance, which biases
        # the ceiling DOWN. Restricting both halves to the conditions all three
        # sets share keeps the estimate honest; the count is reported so a
        # caller can see how much of the design was usable.
        present = (test_stim.abs().sum(dim=0) > 1e-10).clone()
        for _, half_stim, _ in half_fits:
            present &= half_stim.abs().sum(dim=0) > 1e-10
        n_shared = int(present.sum().item())
        if n_shared == 0:
            continue
        if n_shared < len(stim_indices) and split_idx == 0:
            notes.append(
                f"{len(stim_indices) - n_shared} of {len(stim_indices)} conditions were "
                "missing from a training half and were excluded from the ceiling"
            )

        test_stim_shared = test_stim[:, present]
        if unpred_basis is not None:
            from fastfuncstuff.glm.xval import _project_out_basis

            test_stim_shared = _project_out_basis(test_stim_shared, unpred_basis, time_dim=0)
        pseudo_inverses = []
        for _, half_stim, _ in half_fits:
            shared = half_stim[:, present]
            gram = shared.T @ shared
            gram_inv = torch.linalg.inv(gram + 1e-6 * torch.eye(gram.shape[0], device=device))
            pseudo_inverses.append(gram_inv @ shared.T)

        for batch_start in range(0, n_voxels, batch_size):
            batch_end = min(batch_start + batch_size, n_voxels)
            batch = slice(batch_start, batch_end)

            predictions = []
            for (half_tps, _, q_half), pseudo_inverse in zip(
                half_fits, pseudo_inverses, strict=True
            ):
                half_data = data[batch][:, half_tps].to(device)
                if q_half is not None:
                    half_data = half_data - (half_data @ q_half) @ q_half.T
                betas = half_data @ pseudo_inverse.T
                predictions.append(betas @ test_stim_shared.T)
                del half_data, betas

            test_data = data[batch][:, test_tps].to(device)
            if q_test is not None:
                test_data = test_data - (test_data @ q_test) @ q_test.T
            if unpred_basis is not None:
                test_data = _project_out_basis(test_data, unpred_basis, time_dim=1)

            pred_a, pred_b = predictions
            ab, a_sum, b_sum, actual = _centered_cov_and_ss(pred_a, pred_b, test_data)
            sum_ab[batch] += ab.to(accumulator_device)
            sum_a[batch] += a_sum.to(accumulator_device)
            sum_b[batch] += b_sum.to(accumulator_device)
            sum_y[batch] += actual.sum(dim=1).to(accumulator_device)
            sum_y_sq[batch] += actual.square().sum(dim=1).to(accumulator_device)
            del predictions, pred_a, pred_b, test_data

        n_points += len(test_tps)
        n_usable += 1

        if verbose:
            print(f"  Ceiling split {split_idx + 1}/{len(cv_splits)}: halves {half_a} | {half_b}")

    if n_usable == 0 or n_points < 2:
        return CeilingResult(
            ceiling=torch.full((n_voxels,), torch.nan, device=data.device, dtype=data.dtype),
            method="loro_two_half",
            n_usable=0,
            notes=[
                "no fold had enough training runs to split in two; "
                "fall back to the df-corrected ceiling"
            ],
        )

    covariance = sum_ab - sum_a * sum_b / n_points
    total = sum_y_sq - sum_y.square() / n_points
    # Negative covariance means the two halves agreed on nothing; clamping keeps
    # the ceiling usable as a divisor without inventing signal, matching
    # split_half_noise_ceiling.
    ceiling = torch.where(
        total > 0, (covariance / total.clamp_min(1e-30)).clamp_min(0.0), torch.nan
    )

    if n_usable < len(cv_splits):
        notes.append(f"{len(cv_splits) - n_usable} of {len(cv_splits)} folds could not be split")

    return CeilingResult(
        ceiling=ceiling.to(data.dtype),
        method="loro_two_half",
        n_usable=n_usable,
        notes=notes,
    )


def ncsnr_noise_ceiling(
    betas: torch.Tensor,
    condition_ids: torch.Tensor,
    n_train_repeats: float | None = None,
    min_repeats: int = 2,
) -> CeilingResult:
    """Beta-space R2 ceiling from repeats of the same condition (NSD-style).

    Trials of one condition should have identical betas, so their spread is
    measurement noise and whatever variance is left across conditions is
    signal::

        var_noise  = mean over conditions of within-condition variance
        var_signal = var(all betas) - var_noise
        ncsnr      = sqrt(var_signal / var_noise)

    Only *conditions* need to repeat -- not runs -- which is why this fires on
    ordinary event-related designs where the timeseries ceiling cannot.

    ``n_train_repeats`` selects which ceiling comes back, and the distinction
    matters because it is easy to divide by the wrong one:

    ``None`` gives the published NSD quantity ``ncsnr^2 / (ncsnr^2 + 1)``, the
    ceiling on predicting a single trial from a *noiseless* predictor. It is
    always in [0, 1] and is what to report next to NSD maps.

    A number gives the ceiling on what these tools actually score -- a held-out
    trial predicted by the average of ``m`` training trials, whose own noise
    does not vanish::

        (var_signal - var_noise / m) / (var_signal + var_noise)

    This is the honest divisor for the beta-space cross-validated R2, and it can
    go negative where a single trial is not predictable at all. That is a real
    statement about the data, not a defect.

    Conditions with fewer than ``min_repeats`` trials carry no information about
    the noise and are dropped from the noise term while still contributing to
    the signal term, which is what NSD's estimator does.
    """
    if betas.ndim != 2:
        raise ValueError("betas must have shape (n_voxels, n_trials)")
    if condition_ids.numel() != betas.shape[1]:
        raise ValueError("condition_ids must have one entry per trial")

    work = betas.to(torch.float64)
    condition_ids = condition_ids.to(betas.device)
    notes: list[str] = []

    unique, counts = torch.unique(condition_ids, return_counts=True)
    repeated = unique[counts >= min_repeats]
    n_dropped = int((counts < min_repeats).sum().item())
    if n_dropped:
        notes.append(
            f"{n_dropped} of {len(unique)} conditions had fewer than {min_repeats} trials "
            "and could not contribute to the noise estimate"
        )

    if repeated.numel() == 0:
        return CeilingResult(
            ceiling=torch.full(
                (betas.shape[0],), torch.nan, device=betas.device, dtype=betas.dtype
            ),
            method="ncsnr",
            n_usable=0,
            notes=["no condition repeated; a beta-space noise ceiling is not estimable"],
        )

    # Unbiased within-condition variance, pooled by weighting each condition by
    # its degrees of freedom rather than counting every condition equally: a
    # condition seen twice says much less about the noise than one seen twenty
    # times.
    numerator = torch.zeros(betas.shape[0], dtype=torch.float64, device=betas.device)
    dof = 0.0
    for condition in repeated.tolist():
        trials = work[:, condition_ids == condition]
        centered = trials - trials.mean(dim=1, keepdim=True)
        numerator += centered.square().sum(dim=1)
        dof += trials.shape[1] - 1
    var_noise = numerator / max(dof, 1.0)

    var_total = work.var(dim=1, unbiased=True)
    var_signal = (var_total - var_noise).clamp_min(0.0)

    if n_train_repeats is None:
        ncsnr_sq = var_signal / var_noise.clamp_min(1e-30)
        ceiling = ncsnr_sq / (ncsnr_sq + 1.0)
    else:
        m = max(float(n_train_repeats), 1.0)
        denominator = (var_signal + var_noise).clamp_min(1e-30)
        ceiling = (var_signal - var_noise / m) / denominator

    ceiling = torch.where(var_noise > 0, ceiling, torch.nan)

    return CeilingResult(
        ceiling=ceiling.to(betas.dtype),
        method="ncsnr",
        n_usable=int(repeated.numel()),
        notes=notes,
    )


def zscore_betas_by_run(betas: torch.Tensor, run_ids: torch.Tensor) -> torch.Tensor:
    """Per-run z-scoring of single-trial betas, matching the CV's own normalisation.

    ``single_trial_cv_helper(zscore_by_run=True)`` -- the GLMsingle default that
    ``ffs_ridge`` inherits -- scores z-scored betas, which strips the per-run mean
    and scale before the R2 is computed. A ceiling estimated from the raw betas is
    then a bound on a *different* quantity: the run-level variance the CV removed
    is still in the ceiling's denominator, so the ceiling reads too low and the
    explainable fraction sails past 1.

    Bug of record: ffs_ridge reported a median explainable R2 of 1.26 on synthetic
    data where the fitted model was the generating model, purely from this
    mismatch.
    """
    if betas.shape[1] != run_ids.numel():
        raise ValueError("run_ids must have one entry per trial")
    out = betas.clone()
    run_ids = run_ids.to(betas.device)
    for run_id in torch.unique(run_ids).tolist():
        mask = run_ids == run_id
        block = out[:, mask]
        mean = block.mean(dim=1, keepdim=True)
        std = block.std(dim=1, keepdim=True).clamp_min(1e-10)
        out[:, mask] = (block - mean) / std
    return out


def mean_train_repeats(
    condition_ids: torch.Tensor,
    run_ids: torch.Tensor,
    cv_splits: list[tuple[list[int], list[int]]],
) -> float:
    """Average training trials per predicted condition, across the actual folds.

    This is the ``m`` :func:`ncsnr_noise_ceiling` needs to match the ceiling to
    what the cross-validation really scores. It is measured from the folds rather
    than assumed to be ``n_trials_per_condition * (n_runs - 1) / n_runs``, because
    designs where a condition appears in only some runs -- which is common enough
    to be the default assumption in this codebase -- make that formula wrong in
    exactly the voxels the ceiling matters for.

    Conditions with no training trials in a fold are skipped: they contribute a
    test trial nothing could predict, which is the cross-validation's problem to
    handle, not the ceiling's.
    """
    condition_ids = condition_ids.cpu()
    run_ids = run_ids.cpu()
    counts: list[int] = []
    for train_runs, test_runs in cv_splits:
        in_train = torch.isin(run_ids, torch.tensor(list(train_runs)))
        in_test = torch.isin(run_ids, torch.tensor(list(test_runs)))
        for condition in torch.unique(condition_ids[in_test]).tolist():
            n_train = int(((condition_ids == condition) & in_train).sum())
            if n_train > 0:
                counts.append(n_train)
    if not counts:
        return 1.0
    return float(sum(counts) / len(counts))


def ncsnr(betas: torch.Tensor, condition_ids: torch.Tensor, min_repeats: int = 2) -> torch.Tensor:
    """The NSD noise-ceiling SNR itself, saved as a standalone diagnostic map.

    Reported separately from the ceiling because it is unbounded and so keeps
    resolving differences between voxels that the ceiling has already
    compressed against 1.
    """
    result = ncsnr_noise_ceiling(
        betas, condition_ids, n_train_repeats=None, min_repeats=min_repeats
    )
    ratio = result.ceiling.to(torch.float64)
    return torch.sqrt((ratio / (1.0 - ratio).clamp_min(1e-30)).clamp_min(0.0)).to(betas.dtype)


@dataclass
class BetaSpaceCeiling:
    """Everything a CLI needs to write a beta-space ceiling, computed once."""

    result: CeilingResult
    ncsnr_map: torch.Tensor
    explainable: torch.Tensor | None
    n_train_repeats: float
    explainable_withheld_because: str | None = None


def beta_space_ceiling(
    betas: torch.Tensor,
    condition_ids: torch.Tensor,
    run_ids: torch.Tensor,
    cv_splits: list[tuple[list[int], list[int]]],
    xval_r2: torch.Tensor | None = None,
    zscore_by_run: bool = False,
    metric: str = "cod",
    min_repeats: int = 2,
) -> BetaSpaceCeiling:
    """The whole beta-space ceiling recipe, so four CLIs cannot each get it wrong.

    ``ffs_ridge``, ``ffs_denoise``, ``ffs_hrfopt`` and ``ffs_reml -beta_cv`` all
    score held-out trial betas against same-condition training betas, and all
    four need the same three corrections. Duplicating them was how the first
    version shipped an explainable R2 of 1.26:

    1. **Normalise as the CV did.** ``zscore_by_run`` must match the flag the
       cross-validation ran under, or run-level variance sits in the ceiling's
       denominator but not the R2's.
    2. **Match the divisor to the fold.** The predictor is an average of ``m``
       training trials with noise of its own, and ``m`` is measured from the
       folds rather than assumed.
    3. **Refuse the ratio off-scale.** Only a coefficient of determination lives
       on the ceiling's variance-fraction scale; ``sse``, ``corr`` and ``corr2``
       do not, and dividing them would produce a confident-looking nonsense.
    """
    if zscore_by_run:
        betas = zscore_betas_by_run(betas, run_ids)

    repeats = mean_train_repeats(condition_ids, run_ids, cv_splits)
    result = ncsnr_noise_ceiling(
        betas, condition_ids, n_train_repeats=repeats, min_repeats=min_repeats
    )
    ncsnr_map = ncsnr(betas, condition_ids, min_repeats=min_repeats)

    explainable: torch.Tensor | None = None
    withheld: str | None = None
    if xval_r2 is None:
        withheld = "no cross-validated R2 was computed"
    elif metric != "cod":
        withheld = (
            f"-metric {metric} is not on the variance-fraction scale the ceiling "
            "uses, so the ratio would not mean anything; use -metric cod"
        )
    elif result.n_usable == 0:
        withheld = "the ceiling was not estimable"
    else:
        explainable = result.explainable_r2(xval_r2)

    return BetaSpaceCeiling(
        result=result,
        ncsnr_map=ncsnr_map,
        explainable=explainable,
        n_train_repeats=repeats,
        explainable_withheld_because=withheld,
    )


def df_corrected_ceiling(
    ss_model: torch.Tensor,
    ss_total: torch.Tensor,
    sigma2_residual: torch.Tensor,
    n_timepoints: int,
    n_task_columns: int,
) -> CeilingResult:
    """Timeseries R2 ceiling from degrees of freedom, requiring no repeats.

    In-sample ``ss_model`` overstates the task-locked signal by roughly
    ``p * sigma2`` -- fitting p free columns to noise explains p noise
    dimensions -- so subtracting that gives an unbiased signal variance and

        ceiling = (ss_model - p * sigma2) / ss_total

    The fallback when there are too few runs for :func:`loro_two_half_ceiling`,
    and useful as a cross-check when there are: two estimates of the same
    quantity in the same units disagreeing is itself a finding.

    **It bounds this design, not every design.** The quantity is "how close
    could held-out prediction get if the betas were exact", which is the right
    question for a denoising or HRF choice and the wrong one for asking whether
    the model is specified correctly -- a design missing a real condition gets a
    confidently low ceiling and a flattering explainable R2.
    """
    signal = (
        ss_model.to(torch.float64) - n_task_columns * sigma2_residual.to(torch.float64)
    ).clamp_min(0.0)
    total = ss_total.to(torch.float64)
    ceiling = torch.where(total > 0, signal / total.clamp_min(1e-30), torch.nan)
    notes = []
    if n_task_columns >= n_timepoints // 2:
        notes.append(
            f"{n_task_columns} task columns against {n_timepoints} timepoints: the "
            "df correction is large and the ceiling is correspondingly uncertain"
        )
    return CeilingResult(
        ceiling=ceiling.to(ss_model.dtype),
        method="df_corrected",
        n_usable=1,
        notes=notes,
    )
