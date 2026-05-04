"""MIGP — MELODIC's incremental group PCA (Smith 2014).

Stacks per-subject (or per-run) data along time, then periodically reduces the
running stack to its top `migp_n` PC time-courses. This bounds peak memory at
~`migp_factor * migp_n * n_voxels` regardless of the number of subjects, while
preserving the top-`migp_n` subspace of the full concatenation.

Reference
---------
Smith S et al., NeuroImage 2014 — "Group-PCA for very large fMRI datasets"
MELODIC source: `meldata.cc::setup_migp` (FSL 6+).
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.inference_mode()
def migp_reduce(
    runs: Iterable[torch.Tensor],
    migp_n: int | None = None,
    migp_factor: float = 2.0,
    scale_by_n: bool = True,
    device: torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Run MELODIC-style MIGP on a sequence of per-run (T_i, V) tensors.

    Parameters
    ----------
    runs : iterable of (T_i, V) Tensors
        Each element is one run/subject's time-by-voxels data. Iteration order
        is preserved (caller is responsible for any shuffling).
    migp_n : int, optional
        Target dimensionality of the reduced subspace. If None, defaults to
        `2*T_first - 1` matching MELODIC's auto-pick.
    migp_factor : float, default 2.0
        Reduction trigger threshold. Reduce whenever the accumulated stack
        exceeds ``migp_factor * migp_n`` rows. Default 2.0 matches MELODIC
        (``meldata.cc:554``, option ``--migp_factor`` defaults to 2). Larger
        values batch more files between reductions (fewer SVD calls, more
        memory).
    scale_by_n : bool, default True
        If True, divides each input by the total run count before stacking —
        matches MELODIC's `tmpData / numfiles` normalization so all inputs
        contribute equally regardless of length.
    device : torch.device, optional
        Device for the running stack. Defaults to the device of the first run.
    verbose : bool
        Print per-step shape diagnostics.

    Returns
    -------
    reduced : (migp_n, V) Tensor on `device`
        Top-`migp_n` PC time-courses approximating the full concatenation.
    """
    runs_list: list[torch.Tensor] = list(runs)
    if not runs_list:
        raise ValueError("migp_reduce: no runs provided")
    n_runs_total = len(runs_list)
    scale = 1.0 / float(n_runs_total) if scale_by_n else 1.0
    if device is None:
        device = runs_list[0].device

    # Default migp_n: match MELODIC's `2*T_per_file - 1` rule from setup_migp.
    if migp_n is None:
        migp_n = max(1, 2 * int(runs_list[0].shape[0]) - 1)

    stack: torch.Tensor | None = None
    for i, x in enumerate(runs_list):
        if x.dim() != 2:
            raise ValueError(f"runs[{i}] must be 2D (T_i, V); got {tuple(x.shape)}")
        x = x.to(device, non_blocking=True)
        if scale_by_n:
            x = x * scale
        stack = x if stack is None else torch.cat([stack, x], dim=0)
        is_last = (i == n_runs_total - 1)
        # Reduce when stack outgrows the trigger, or after the last run.
        if stack.shape[0] > migp_factor * migp_n or is_last:
            stack = _reduce_to_topk(stack, k=migp_n)
        if verbose:
            print(f"  MIGP run {i + 1}/{n_runs_total}: stack={tuple(stack.shape)}")

    assert stack is not None
    return stack


def _reduce_to_topk(data: torch.Tensor, k: int) -> torch.Tensor:
    """Reduce an (R, V) matrix to its top-k PC time-courses.

    Matches MELODIC's ``std_pca`` → ``EigenValues`` → ``pcaE.t() * Data``
    in ``setup_migp``: row-centers (spatial mean per timepoint) before
    computing the temporal covariance, eigendecomposes it (matching newmat's
    ``EigenValues`` — ascending order), keeps the top-k eigenvectors, then
    projects the **original** (non-centered) data.

    Sign convention: after ``eigh``, each eigenvector is flipped so its
    largest-absolute-value element is positive. This eliminates the arbitrary
    sign ambiguity inherent to LAPACK's ``dsyevd`` and matches the convention
    used by Armadillo's ``eig_sym`` (MELODIC's backend).

    Precision: the entire reduction runs in float64 for numerical accuracy.
    Errors from incremental reductions compound through varnorm's thresholding
    step — the ~2x speed cost is negligible (covariance is only R×R and the
    projection dominates runtime). The result is cast back to the input dtype.
    """
    R, V = data.shape
    k_eff = min(k, R, V)
    if R <= k_eff:
        return data
    orig_dtype = data.dtype
    d64 = data.to(torch.float64)
    row_mean = d64.mean(dim=1, keepdim=True)
    cov = (d64 @ d64.T - V * (row_mean @ row_mean.T)) / float(V)
    evals, evecs = torch.linalg.eigh(cov)
    evecs = evecs.flip(1)[:, :k_eff]
    max_idx = evecs.abs().argmax(dim=0)
    signs = torch.sign(evecs[max_idx, torch.arange(k_eff, device=evecs.device)])
    signs[signs == 0] = 1.0
    evecs = evecs * signs.unsqueeze(0)
    return (evecs.T @ d64).to(orig_dtype)
