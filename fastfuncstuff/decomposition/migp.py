"""MIGP — incremental group PCA (Smith et al. 2014).

Stacks per-subject (or per-run) data along time, then periodically reduces the
running stack to its top `migp_n` PC time-courses. This bounds peak memory at
~`migp_factor * migp_n * n_voxels` regardless of the number of subjects, while
preserving the top-`migp_n` subspace of the full concatenation.

Reference
---------
Smith, S.M., Hyvärinen, A., Varoquaux, G., Miller, K.L. & Beckmann, C.F. (2014).
"Group-PCA for very large fMRI datasets", NeuroImage 101:738-749.

Written from that paper; FSL's implementation is not a reference for it. See
``../fmri_wiki/notes/FSL clean-room policy.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from fastfuncstuff.utils import to_linalg_f64


@torch.inference_mode()
def migp_reduce(
    runs: Iterable[torch.Tensor],
    migp_n: int | None = None,
    migp_factor: float = 2.0,
    scale_by_n: bool = True,
    device: torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Incremental group PCA over a sequence of per-run (T_i, V) tensors.

    Parameters
    ----------
    runs : iterable of (T_i, V) Tensors
        Each element is one run/subject's time-by-voxels data. Iteration order
        is preserved (caller is responsible for any shuffling).
    migp_n : int, optional
        Target dimensionality of the reduced subspace. If None, defaults to
        ``2*T_first - 1``: comfortably more than one run contributes, so the
        truncation cannot discard a subspace that a single run could have
        supported on its own. The overshoot above the eventual component count is
        what absorbs the incremental truncation error.
    migp_factor : float, default 2.0
        Reduction trigger threshold. Reduce whenever the accumulated stack
        exceeds ``migp_factor * migp_n`` rows. This trades SVD calls against peak
        memory and nothing else -- the answer is the same either way, so 2.0 is
        simply the point where the stack is large enough that a reduction is
        worth its cost. Larger values batch more runs between reductions.
    scale_by_n : bool, default True
        If True, divides each input by the total run count before stacking, so
        every run contributes equally to the group subspace rather than in
        proportion to its length.
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

    # Default: twice one run's timepoints, less one. See the parameter docstring.
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
        is_last = i == n_runs_total - 1
        # Reduce when stack outgrows the trigger, or after the last run.
        if stack.shape[0] > migp_factor * migp_n or is_last:
            stack = _reduce_to_topk(stack, k=migp_n)
        if verbose:
            print(f"  MIGP run {i + 1}/{n_runs_total}: stack={tuple(stack.shape)}")

    assert stack is not None
    return stack


def _reduce_to_topk(data: torch.Tensor, k: int) -> torch.Tensor:
    """Reduce an (R, V) matrix to its top-k PC time-courses.

    Row-centres (removes the spatial mean per timepoint) before forming the
    temporal covariance, eigendecomposes it, keeps the top-k eigenvectors, then
    projects the **original**, non-centred data. Centring belongs to the
    covariance estimate, not to the data being reduced: the running stack has to
    stay in the same frame across iterations or successive reductions are not
    composing the same quantity.

    Sign convention: after ``eigh``, each eigenvector is flipped so its
    largest-absolute-value element is positive. Eigenvector sign is arbitrary and
    LAPACK makes no guarantee about it, so without fixing a convention the same
    input can reduce to sign-flipped time-courses run to run.

    Precision: the entire reduction runs in float64 for numerical accuracy.
    Errors from incremental reductions compound through the varnorm step
    — the ~2x speed cost is negligible (covariance is only R×R and the
    projection dominates runtime). The result is cast back to the input dtype.
    """
    R, V = data.shape
    k_eff = min(k, R, V)
    if R <= k_eff:
        return data
    orig_dtype = data.dtype
    # MIGP's incremental SVD needs float64; MPS has none, so run on CPU and
    # return the reduced result to the original device.
    d64 = to_linalg_f64(data)
    row_mean = d64.mean(dim=1, keepdim=True)
    cov = (d64 @ d64.T - V * (row_mean @ row_mean.T)) / float(V)
    evals, evecs = torch.linalg.eigh(cov)
    evecs = evecs.flip(1)[:, :k_eff]
    max_idx = evecs.abs().argmax(dim=0)
    signs = torch.sign(evecs[max_idx, torch.arange(k_eff, device=evecs.device)])
    signs[signs == 0] = 1.0
    evecs = evecs * signs.unsqueeze(0)
    return (evecs.T @ d64).to(device=data.device, dtype=orig_dtype)
