"""Condition-level cross-validated R2 for ffs_reml, with the ARMA fit reused.

``ffs_reml`` could report how well its model predicts data it never saw, and
until now could not: its only cross-validation was in single-trial beta space.
This module supplies the timeseries half -- train-run betas predict a held-out
run's timecourse -- in the same units the other tools report, so a REML R2, a
denoise R2 and a ceiling are all commensurable.

**The ARMA(1,1) parameters are not re-estimated per fold.** They come from the
whole-data REML fit and are held fixed while only the betas are refitted. That
is leakage, and it is the right trade: re-running the grid search inside every
fold multiplies the most expensive stage in the tool by the fold count, while
the parameters it would re-estimate move very little between folds that share
most of their runs. What leaks is a slightly optimistic *noise model*, which
shows up in the standard errors rather than in the betas -- and it is the betas
that generate the prediction being scored. The R2 is therefore a fair measure
of prediction quality and a mildly optimistic measure of the noise model, so
the number to compare across tools is this one, not a stricter refit.

**R2 is computed in unwhitened space.** Whitening is a property of the
estimator, not of the thing being predicted: scoring in whitened space would
compare each voxel against a differently rescaled target and make the values
incomparable between voxels, between tools, and with AFNI. So the betas are
GLS, and the prediction and its R2 are in the data's own units.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance
from fastfuncstuff.glm.xval import _compute_projection_matrix


def _run_local_segments(
    run_indices: list[int], run_lengths: np.ndarray
) -> list[tuple[int, int, int]]:
    """``(run_index, offset_into_concatenated_block, length)`` for chosen runs."""
    segments = []
    offset = 0
    for run_idx in run_indices:
        length = int(run_lengths[run_idx])
        segments.append((run_idx, offset, length))
        offset += length
    return segments


def _whiten_per_run(
    design: torch.Tensor,
    data: torch.Tensor,
    segments: list[tuple[int, int, int]],
    a: float,
    b: float,
    cholesky_cache: dict[tuple[float, float, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the ARMA(1,1) whitening transform run by run.

    The covariance is block-diagonal across runs -- timepoints in different runs
    are uncorrelated because the scanner stopped -- so whitening each run against
    its own ``(run_length x run_length)`` block gives exactly the same answer as
    one big block-diagonal solve at a fraction of the cost. At a hundred runs the
    concatenated covariance would be tens of gigabytes; the per-run blocks are
    megabytes.

    Cholesky factors are cached on ``(a, b, run_length)`` because runs of equal
    length share a factor, and the same factors recur across folds.
    """
    design_white = torch.empty_like(design)
    data_white = torch.empty_like(data)

    for _, offset, length in segments:
        key = (a, b, length)
        factor = cholesky_cache.get(key)
        if factor is None:
            covariance = build_arma11_covariance(
                a, b, length, torch.device("cpu"), dtype=torch.float64
            )
            if covariance is None:
                raise ValueError(f"invalid ARMA(1,1) parameters a={a}, b={b}")
            factor = torch.linalg.cholesky(covariance).to(design.device, dtype=design.dtype)
            cholesky_cache[key] = factor

        block = slice(offset, offset + length)
        design_white[block] = torch.linalg.solve_triangular(factor, design[block], upper=False)
        data_white[block] = torch.linalg.solve_triangular(factor, data[block], upper=False)

    return design_white, data_white


def compute_xval_r2_arma(
    data: torch.Tensor,
    design_matrix: torch.Tensor | np.ndarray,
    run_starts: list[int],
    stim_indices: list[int],
    nuisance_indices: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    arma_params: torch.Tensor,
    device: torch.device | None = None,
    metric: str = "cod",
    verbose: bool = True,
) -> dict[str, torch.Tensor | int]:
    """Held-out R2 on the condition-level design, using GLS betas.

    Mirrors :func:`compute_xval_r2` fold for fold -- same splits, same per-fold
    nuisance projection, same concatenate-then-score accumulation -- and differs
    only in fitting the betas by GLS under each voxel's ARMA(1,1) covariance
    instead of by OLS. Keeping the structure identical is deliberate: the two
    R2 values are meant to be compared, and the ceiling in
    :func:`loro_two_half_ceiling` bounds both.

    Voxels are grouped by their ``(a, b)`` pair, of which the default grid
    admits about 117, so the expensive Cholesky work happens once per distinct
    pair per fold rather than once per voxel.

    Parameters
    ----------
    arma_params : (n_voxels, 2) tensor
        Per-voxel ``(a, b)`` from the whole-data REML fit; held fixed here.
    """
    from fastfuncstuff.utils import get_device, to_tensor

    if device is None:
        device = get_device()

    if metric != "cod":
        # The streaming accumulators below compute a CoD directly and never keep
        # the full prediction other metrics would need. Fail up front rather than
        # after the work, and never return a CoD mislabelled as something else.
        raise ValueError(
            f"metric='{metric}' is not supported by the ARMA CV path (only 'cod'); "
            "use compute_xval_r2 for correlation-based metrics"
        )

    design_matrix = to_tensor(design_matrix, device=device, dtype=torch.float32)
    n_voxels, n_timepoints = data.shape
    run_lengths = np.diff(list(run_starts) + [n_timepoints])

    if arma_params.shape[0] != n_voxels:
        raise ValueError(f"arma_params has {arma_params.shape[0]} voxels, data has {n_voxels}")

    # Quantise to the grid step so floating-point noise in the stored parameters
    # cannot explode the group count into thousands of near-identical pairs.
    quantised = torch.round(arma_params.detach().cpu() * 1000.0) / 1000.0
    pairs, inverse = torch.unique(quantised, dim=0, return_inverse=True)
    if verbose:
        print(f"  ARMA-aware CV: {len(pairs)} distinct (a, b) group(s) over {n_voxels:,} voxels")

    accumulator_device = device if device.type == "cuda" else torch.device("cpu")
    ss_residual = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_actual = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    sum_sq_actual = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
    n_scored = 0

    cholesky_cache: dict[tuple[float, float, int], torch.Tensor] = {}

    for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
        # Only the training block is whitened: the test prediction and its R2
        # live in unwhitened space by design (see the module docstring).
        train_segments = _run_local_segments(train_runs, run_lengths)

        train_tps: list[int] = []
        for run_idx in train_runs:
            start = run_starts[run_idx]
            train_tps.extend(range(start, start + int(run_lengths[run_idx])))
        test_tps: list[int] = []
        for run_idx in test_runs:
            start = run_starts[run_idx]
            test_tps.extend(range(start, start + int(run_lengths[run_idx])))

        train_design = design_matrix[train_tps, :]
        test_design = design_matrix[test_tps, :]

        # Nuisance leaves the test fold by projection, exactly as in
        # compute_xval_r2: the prediction is task-only, so the target it is
        # scored against must be task-only too.
        q_test = _compute_projection_matrix(test_design, nuisance_indices)
        test_design_clean = (
            test_design - q_test @ (q_test.T @ test_design) if q_test is not None else test_design
        )
        test_stim = test_design_clean[:, stim_indices]

        # A condition absent from the training runs cannot be predicted; zeroing
        # its beta keeps the prediction valid rather than dropping the fold.
        train_present = train_design[:, stim_indices].abs().sum(dim=0) > 1e-10
        if not train_present.any():
            continue

        train_data_all = data[:, train_tps].to(device)
        test_data_all = data[:, test_tps].to(device)
        if q_test is not None:
            test_data_all = test_data_all - (test_data_all @ q_test) @ q_test.T

        for group_idx in range(len(pairs)):
            voxel_mask = inverse == group_idx
            if not bool(voxel_mask.any()):
                continue
            a = float(pairs[group_idx, 0])
            b = float(pairs[group_idx, 1])

            group_data = train_data_all[voxel_mask.to(train_data_all.device)]
            design_white, data_white = _whiten_per_run(
                train_design, group_data.T.contiguous(), train_segments, a, b, cholesky_cache
            )

            # GLS betas = OLS on the whitened pair. The full design is fitted so
            # the nuisance columns absorb what they should; only the task betas
            # go on to predict.
            keep = torch.ones(design_white.shape[1], dtype=torch.bool, device=device)
            for column, present in zip(stim_indices, train_present.tolist(), strict=True):
                keep[column] = present
            fit_design = design_white[:, keep]
            gram = fit_design.T @ fit_design
            gram_inv = torch.linalg.inv(gram + 1e-6 * torch.eye(gram.shape[0], device=device))
            betas_kept = (gram_inv @ fit_design.T) @ data_white  # (n_kept, n_group_voxels)

            betas_full = torch.zeros(
                design_white.shape[1], betas_kept.shape[1], device=device, dtype=betas_kept.dtype
            )
            betas_full[keep] = betas_kept
            task_betas = betas_full[stim_indices]  # (n_stim, n_group_voxels)

            predicted = (test_stim @ task_betas).T  # (n_group_voxels, n_test_tps)
            actual = test_data_all[voxel_mask.to(test_data_all.device)]

            residual = (actual - predicted).to(torch.float64)
            index = voxel_mask.nonzero(as_tuple=True)[0].to(accumulator_device)
            ss_residual[index] += residual.square().sum(dim=1).to(accumulator_device)
            sum_actual[index] += actual.to(torch.float64).sum(dim=1).to(accumulator_device)
            sum_sq_actual[index] += (
                actual.to(torch.float64).square().sum(dim=1).to(accumulator_device)
            )

            del group_data, design_white, data_white, predicted, actual, residual

        n_scored += len(test_tps)
        del train_data_all, test_data_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if verbose:
            print(
                f"  Split {split_idx + 1}/{len(cv_splits)}: train {train_runs} | test {test_runs}"
            )

    if n_scored == 0:
        return {"r2": torch.full((n_voxels,), torch.nan, device=data.device), "n_splits": 0}

    ss_total = sum_sq_actual - sum_actual.square() / n_scored
    r2 = torch.where(ss_total > 0, 1.0 - ss_residual / ss_total.clamp_min(1e-30), torch.nan)

    return {
        "r2": r2.to(torch.float32),
        "n_splits": len(cv_splits),
        "n_groups": len(pairs),
    }


__all__ = ["compute_xval_r2_arma"]
