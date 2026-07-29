"""
Per-run OLS sufficient statistics.

When nuisance is projected out *per run* — the [[Block-diagonal nuisance]]
arrangement every GLM in this codebase uses — both normal-equation terms are
plain sums over runs::

    X'X = Σ_r Xr'Xr        X'y = Σ_r Xr'yr

so any fit over a *subset* of runs is a sum of precomputed blocks plus one
small ``(C, C)`` solve. Nothing voxel-sized has to be re-projected, re-sliced,
or copied across the host↔device boundary once the blocks exist.

That turns two otherwise-expensive diagnostics into cheap ones:

* the held-out learning curve, which refits on every k-run subset
  (:mod:`fastfuncstuff.denoise.heldout`);
* the aligned PC-selection criterion, which swaps a single run's block between
  its denoised and undenoised form and refits
  (:mod:`fastfuncstuff.denoise.combinatorial`).

Both cost O(runs) projections instead of O(candidates × runs).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = ["RunMoments", "compute_run_moments", "solve_from_moments", "run_bounds"]


def run_bounds(run_starts: list[int], n_timepoints: int, run_idx: int) -> tuple[int, int]:
    """Half-open ``[start, end)`` timepoint bounds of one run."""
    end = run_starts[run_idx + 1] if run_idx < len(run_starts) - 1 else n_timepoints
    return run_starts[run_idx], end


class RunMoments(NamedTuple):
    """Per-run ``X'X`` and ``X'y`` blocks for one design and one voxel set."""

    xtx: list[torch.Tensor]  # per run, (C, C) float64 on the compute device
    xty: list[torch.Tensor]  # per run, (C, V) on the storage device


def compute_run_moments(
    data: torch.Tensor,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    design: torch.Tensor,
    device: torch.device,
    store_device: torch.device | None = None,
    voxel_mask: torch.Tensor | None = None,
    runs: list[int] | None = None,
    chunk_size: int = 20000,
) -> RunMoments:
    """Project each run's nuisance out and accumulate its normal-equation blocks.

    Parameters
    ----------
    data : torch.Tensor
        ``(n_voxels, n_timepoints)``, runs concatenated.
    run_starts : list of int
        Start index of each run in ``data``.
    nuisance_per_run : list of torch.Tensor
        ``(run_length, n_nuisance)`` per run — polynomials, user regressors and
        any noise PCs already appended.
    design : torch.Tensor
        ``(n_timepoints, C)`` task design, no nuisance columns.
    device : torch.device
        Compute device.
    store_device : torch.device, optional
        Where the ``(C, V)`` blocks live. Defaults to ``device``; pass the data's
        own device when the full stack would not fit in VRAM.
    voxel_mask : torch.Tensor, optional
        Restrict to these voxels (e.g. a per-HRF group, or criteria voxels).
    runs : list of int, optional
        Only build blocks for these runs. Entries for other runs are omitted, so
        the returned lists are indexed by position in ``runs``, not run index.
    chunk_size : int, default=20000
        Voxels per ``X'y`` chunk.

    Returns
    -------
    RunMoments
    """
    from fastfuncstuff.glm.xval import project_out_nuisance_per_run

    if store_device is None:
        store_device = device

    n_timepoints = data.shape[1]
    n_cols = design.shape[1]
    wanted = list(range(len(run_starts))) if runs is None else list(runs)

    xtx_blocks: list[torch.Tensor] = []
    xty_blocks: list[torch.Tensor] = []

    for run_idx in wanted:
        start, end = run_bounds(run_starts, n_timepoints, run_idx)
        run_data = data[:, start:end]
        if voxel_mask is not None:
            run_data = run_data[voxel_mask, :]

        # Single-run call: reuses the canonical projector, including its
        # zero-column handling for runs whose nuisance is partly padding.
        data_proj, design_proj = project_out_nuisance_per_run(
            data=run_data,
            design=design[start:end, :],
            nuisance_per_run=[nuisance_per_run[run_idx]],
            run_starts=[0],
            device=device,
        )
        X = design_proj.to(device)
        xtx_blocks.append(X.T.double() @ X.double())

        n_vox = data_proj.shape[0]
        xty = torch.zeros(n_cols, n_vox, device=store_device, dtype=X.dtype)
        for cs in range(0, n_vox, chunk_size):
            ce = min(cs + chunk_size, n_vox)
            xty[:, cs:ce] = (X.T @ data_proj[cs:ce, :].to(device).T).to(store_device)
        xty_blocks.append(xty)
        del data_proj, design_proj, X

    return RunMoments(xtx_blocks, xty_blocks)


def solve_from_moments(
    xtx_blocks: list[torch.Tensor],
    xty_blocks: list[torch.Tensor],
    device: torch.device,
    ridge: float = 1e-6,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sum the given blocks and solve the normal equations.

    Returns ``(C, n_voxels)`` betas. The solve is float64 regardless of the
    stored dtype — it is a ``(C, C)`` system, so the promotion is free, and the
    summed ``X'X`` can be near-singular when few runs contribute.
    """
    xtx = torch.stack(xtx_blocks, dim=0).sum(dim=0)
    xty = xty_blocks[0].clone()
    for block in xty_blocks[1:]:
        xty += block

    # The same small ridge the combinatorial fit uses: a subset design can be
    # genuinely rank-deficient for a condition no contributing run sampled.
    n_cols = xtx.shape[0]
    xtx = xtx + ridge * torch.eye(n_cols, device=device, dtype=xtx.dtype)
    return torch.linalg.solve(xtx, xty.to(device).double()).to(out_dtype)
