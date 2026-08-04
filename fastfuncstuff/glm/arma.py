"""
GPU-Accelerated ARMA(1,1) Prewhitening for GLM Analysis

Implementation of AFNI 3dREMLfit-style ARMA(1,1) temporal autocorrelation modeling
with massive GPU speedup (5-30x faster than AFNI).

This module provides the ANALYSIS side (not simulation) - use this after data
collection for final GLM fits with proper temporal autocorrelation correction.

Key Features:
- REML (Restricted Maximum Likelihood) estimation of ARMA(1,1) parameters
- Grid search optimization over (a, b) parameter space
- Batched GPU computation for parallel voxel processing
- Generalized Least Squares (GLS) with Cholesky prewhitening
- Produces accurate t-statistics correcting for temporal autocorrelation

References:
- AFNI 3dREMLfit: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/remlfit.html
- Math notes: https://afni.nimh.nih.gov/pub/dist/doc/misc/3dREMLfit/3dREMLfit_mathnotes.pdf
- Woolrich et al. (2001): Temporal autocorrelation in univariate linear modeling
- Worsley & Friston (1995): Analysis of fMRI time-series revisited
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.memory import bytes_per_voxel_arma, make_vram_debugger
from fastfuncstuff.utils import get_device, to_tensor, warn_mps_float32_precision

from .xval import compute_r2_metric


class _RemlMode:
    """Run-wide REML mode, mirroring AFNI's ``static double corcut`` global.

    By default FFS builds the *exact* dense ARMA(1,1) covariance and runs a
    slightly more thorough hierarchical search than 3dREMLfit — more accurate,
    but not byte-for-byte identical to AFNI. Enabling ``afni_faithful`` switches
    the small divergences to AFNI's choices for parity validation:

    - **Banded R**: truncate the covariance to AFNI's ``bmax`` off-diagonals,
      where the last kept correlation is ~``corcut`` (remla.c:692).
    - **Grid filter**: drop grid points with ``0 < lam < corcut`` — AFNI folds
      these near-white cases into the ``a=b=0`` point (remla.c:1355).
    - **Hierarchical ``ltop``**: use AFNI's coarser top level
      (``min(log2(na), log2(nb)) - 2`` on *interval* counts; remla.c:1444).

    This is a module-level singleton on purpose: corcut/mode are constants for a
    whole run, exactly as in the C reference. Set via :func:`set_afni_mode`.
    """

    afni_faithful: bool = False
    corcut: float = 1e-4  # CORCUT_default, remla.c:238


_REML_MODE = _RemlMode()


def set_afni_mode(enabled: bool, corcut: float = 1e-4) -> None:
    """Toggle AFNI-faithful REML behaviour for the rest of the run.

    See :class:`_RemlMode`. The CLI wires this from ``-afni_mode``; the default
    (``enabled=False``) keeps FFS's more-accurate dense covariance and search.
    """
    _REML_MODE.afni_faithful = bool(enabled)
    _REML_MODE.corcut = float(corcut)


def _afni_bmax(a: float, lam: float, corcut: float) -> int:
    """AFNI's banded bandwidth: last kept off-diagonal correlation ~= corcut.

    Mirrors rcmat_arma11 (remla.c:687-699). ``rho`` (=a) is clamped to ±0.9 as
    in the C source. Returns 0 for the identity (near-white) case.
    """
    rho = min(0.9, max(-0.9, a))
    alam = abs(lam)
    if alam < corcut:
        return 0
    if rho == 0.0:
        return 1  # pure MA(1)
    import math

    return 1 + int(math.ceil(math.log(corcut / alam) / math.log(abs(rho))))


def _debug_memory_snapshot(
    label: str, device: torch.device, tensors: dict[str, torch.Tensor] | None = None
) -> None:
    """Print detailed memory snapshot for debugging.

    Parameters
    ----------
    label : str
        Description of current step
    device : torch.device
        Device to check memory for
    tensors : dict, optional
        Dictionary of tensor_name -> tensor to report sizes for
    """
    print(f"\n{'=' * 70}")
    print(f"DEBUG MEMORY: {label}")
    print(f"{'=' * 70}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved

        print(f"GPU Memory ({device}):")
        print(f"  Allocated:     {allocated:>8.3f} GiB")
        print(f"  Reserved:      {reserved:>8.3f} GiB")
        print(f"  Max Allocated: {max_allocated:>8.3f} GiB")
        print(f"  Total:         {total:>8.3f} GiB")
        print(f"  Free:          {free:>8.3f} GiB")
        print()

    if tensors:
        print("Tensor Sizes:")
        total_bytes = 0
        for name, tensor in tensors.items():
            if tensor is None:
                print(f"  {name:30s}: None")
                continue

            size_mb = tensor.element_size() * tensor.nelement() / 1024**2
            total_bytes += tensor.element_size() * tensor.nelement()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            device_str = str(tensor.device)

            print(
                f"  {name:30s}: {str(tuple(tensor.shape)):20s} {dtype_str:10s} {device_str:10s} {size_mb:>8.1f} MiB"
            )

        print(f"  {'TOTAL':30s}: {'':<20s} {'':<10s} {'':<10s} {total_bytes / 1024**2:>8.1f} MiB")
        print()

    print(f"{'=' * 70}\n")


# AFNI 3dREMLfit default grid parameters (Grid 3 - medium resolution)
# These are well-validated values from AFNI documentation
# Every per-voxel array on ARMA11Results, in one place, so anything that fits a
# voxel subset and scatters it back (per-HRF/slice grouping, missing-data
# families) agrees on what has to be carried across.
VOXEL_SCATTER_ATTRS = [
    "betas",
    "tstats",
    "r2",
    "r2_partial",
    "r2_partial_nuisance",
    "r2_semipartial",
    "r2_semipartial_nuisance",
    "arma_params",
    "arma_lambda",
    "reml_likelihood",
    "sigma2",
    "fstats",
    "residuals",
    "residuals_whitened",
    "predicted",
    "contrast_betas",
    "contrast_tstats",
    "contrast_fstats",
    "contrast_r2_partial",
    "contrast_r2_semipartial",
    "dsort_betas",
    "dsort_tstats",
]

# The GLMResults equivalent of VOXEL_SCATTER_ATTRS (OLS has stderr/meanvol/r2_run
# and no ARMA parameters).
OLS_VOXEL_SCATTER_ATTRS = [
    "betas",
    "r2",
    "r2_partial",
    "r2_partial_nuisance",
    "r2_semipartial",
    "r2_semipartial_nuisance",
    "r2_run",
    "residuals",
    "predicted",
    "meanvol",
    "tstats",
    "stderr",
    "sigma2",
    "fstats",
    "contrast_betas",
    "contrast_tstats",
    "contrast_fstats",
]

# Arrays whose second axis is TIME, not regressors. A censored fit produces
# fewer timepoints than the full timeline, so these need scattering back onto
# the original time axis rather than a straight row copy.
TIME_AXIS_ATTRS = frozenset({"residuals", "residuals_whitened", "predicted"})

# AFNI 3dREMLfit defaults: -MAXa 0.8, -MAXb 0.8, -Grid 3 (step=0.1).
# a is non-negative (POScor); b is symmetric.
# 9 a-values × 17 b-values = 153 candidates (gamma0>0 filter trims to ~117).
# Users can widen with the CLI -MAXa/-MAXb flags; the absolute upper bound
# is 0.9 (any closer to 1 and the ARMA(1,1) model is degenerate).
DEFAULT_ARMA_A_GRID = (
    0.0,
    0.8,
    9,
)  # (start, end, num_points) -> [0.0, 0.1, ..., 0.8]
DEFAULT_ARMA_B_GRID = (
    -0.8,
    0.8,
    17,
)  # (start, end, num_points) -> [-0.8, -0.7, ..., 0.7, 0.8]


def check_cuda_memory_before_batch(
    batch_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    device: torch.device,
    verbose: bool = False,
    use_qr: bool = False,
    has_glts: bool = True,
    persistent_tensors: dict | None = None,  # NEW: Account for data already on GPU
    use_double: bool = False,
) -> int:
    """
    Check available CUDA memory and adjust batch size if needed.

    This is the forward-looking logic to prevent OOM before it happens!

    Parameters
    ----------
    batch_voxels : int
        Desired batch size
    n_timepoints : int
        Number of timepoints
    n_regressors : int
        Number of regressors
    device : torch.device
        Device to check
    verbose : bool
        Print warnings if adjustment needed

    Returns
    -------
    int
        Adjusted batch size that will fit in available memory
    """
    if device.type != "cuda":
        return batch_voxels

    # Clear cache first to get accurate reading
    torch.cuda.empty_cache()

    # Calculate required memory for GLS solve
    # MAJOR OPTIMIZATION: X'X is precomputed at GROUP level (once per (a,b) pair)
    # Per-batch allocations are now TINY - just voxel-specific data!
    bpe = 8 if use_double else 4  # bytes per element
    # Per-batch (batch, n_time) tensors that coexist on GPU during the GLS step:
    #   Y_batch_dev (raw), y_w_batch (prewhitened), pred_w_batch, resid_w_batch,
    #   pred_orig_batch, resid_orig_batch  → 6 copies
    # Plus the transient fp64 accumulator inside `(x**2).sum(..., dtype=torch.float64)`
    # peaks at +1 fp32-equivalent copy worth of workspace.
    bnt_copies = 6
    fp64_accum_peak = batch_voxels * n_timepoints * 4  # cuSOLVER/sum workspace ≈ one fp32 copy
    common_time = batch_voxels * n_timepoints * bpe * bnt_copies + fp64_accum_peak
    common_reg = batch_voxels * n_regressors * bpe * 2  # X'y + betas

    if use_qr:
        # QR path: Q and R precomputed at group level
        required_bytes = common_time + common_reg
    else:
        # X'X path: (X'X)^{-1} precomputed at group level
        required_bytes = common_time + common_reg

    # F-stat path with no GLTs no longer materializes (batch, n_reg, n_reg) —
    # constant A_task^{-1} at group level handles it (see fit_glm_arma11).
    # With GLTs we still build var_beta_batch + the per-batch contrast tensors.
    if has_glts:
        required_bytes += batch_voxels * n_regressors * n_regressors * bpe  # var_beta_batch

    # Get available memory
    total_mem = torch.cuda.mem_get_info(device)[1]
    free_mem = torch.cuda.mem_get_info(device)[0]
    reserved_mem = torch.cuda.memory_reserved(device)
    allocated_mem = torch.cuda.memory_allocated(device)

    # Available = free + any reserved-but-unallocated memory
    # But if allocated > reserved (shouldn't happen, but be safe), just use free
    unallocated_reserved = max(0, reserved_mem - allocated_mem)
    available_mem = free_mem + unallocated_reserved

    # CRITICAL FIX: Persistent tensors (L_inv, X_w, design) are ALREADY on GPU
    # They're already counted in allocated_mem, so free_mem already excludes them.
    # We should NOT subtract them from available (that's double-counting!)
    # Instead, we need to exclude them from required_bytes calculation below.
    #
    # Track persistent tensors for later use (don't subtract from available!)
    persistent_mem_info = {}
    if persistent_tensors is not None and verbose:
        for name, tensor in persistent_tensors.items():
            if (
                tensor is not None
                and isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cuda"
            ):
                tensor_bytes = tensor.numel() * tensor.element_size()
                persistent_mem_info[name] = tensor_bytes

    # Use 50% of available — headroom for PyTorch fragmentation, internal buffers,
    # and the fp64-accumulator workspace inside reductions. Was 60%, but OOMs at
    # later groups (~11/82) showed fragmentation eats more than that on 16 GB cards
    # once the 7.5 GB grid is resident.
    usable_mem = available_mem * 0.50

    if required_bytes > usable_mem:
        # Need to reduce batch size
        reduction_factor = usable_mem / required_bytes
        new_batch = int(batch_voxels * reduction_factor * 0.85)  # Extra 15% safety for peak memory
        new_batch = max(new_batch, 100)  # Minimum batch (reduced from 500)

        if verbose:
            print(
                f"\n⚠️  Memory check: batch needs {required_bytes / 1e9:.2f} GB, "
                f"but {usable_mem / 1e9:.2f} GB usable (60% of {available_mem / 1e9:.2f} GB free)"
            )
            print(
                f"  GPU: {allocated_mem / 1e9:.2f} GB allocated, {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total"
            )
            print(f"  📉 Reducing batch: {batch_voxels:,} → {new_batch:,} voxels")
            print("  ℹ️  Conservative: Accounts for peak memory during Cholesky/solve operations\n")

        return new_batch
    return batch_voxels


def estimate_valid_grid_pairs(
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
) -> int:
    """
    Estimate number of valid (a,b) pairs that will pass filtering.

    This applies the same filtering logic as build_arma11_covariance_batch
    but without building the expensive covariance matrices.

    Parameters
    ----------
    a_grid : torch.Tensor
        AR parameter grid
    b_grid : torch.Tensor
        MA parameter grid

    Returns
    -------
    n_valid : int
        Estimated number of valid (a,b) pairs
    """
    # Create all (a,b) combinations
    a_mesh, b_mesh = torch.meshgrid(a_grid, b_grid, indexing="ij")
    a_flat = a_mesh.reshape(-1)
    b_flat = b_mesh.reshape(-1)

    # Apply same filtering as build_arma11_covariance_batch
    valid_mask = (torch.abs(a_flat) < 1.0) & (torch.abs(b_flat) < 1.0)

    denom = 1 - a_flat**2
    valid_mask &= torch.abs(denom) >= 1e-6

    gamma0 = (1 + b_flat**2 + 2 * a_flat * b_flat) / (denom + 1e-10)
    valid_mask &= gamma0 > 0

    a_valid = a_flat[valid_mask]
    b_valid = b_flat[valid_mask]

    # Apply lambda filtering (only allow lambda >= 0)
    rho1_valid = _compute_arma11_lambda(a_valid, b_valid)
    if isinstance(rho1_valid, torch.Tensor):
        lambda_mask = rho1_valid >= 0
        n_valid = int(lambda_mask.sum().item())
    else:
        n_valid = 0

    return n_valid


def calculate_grid_memory_footprint(
    n_valid_ab_pairs: int,
    n_timepoints: int,
    n_regressors: int,
    use_double: bool = False,
) -> int:
    """
    Calculate exact memory footprint of REML grid precomputation.

    The grid precomputation stores these tensors for each (a,b) pair:
    - L: lower Cholesky factor (n_timepoints, n_timepoints)
    - X_w: prewhitened design (n_timepoints, n_regressors)
    - R_qr: upper triangular R from QR(X_w) (n_regressors, n_regressors) - AFNI's "D"
    - logdet_Rcorr: log det of correlation matrix
    - logdet_XwTXw: log det of X'R^-1 X (computed via QR)

    Parameters
    ----------
    n_valid_ab_pairs : int
        Number of valid (a,b) parameter pairs in grid
    n_timepoints : int
        Number of timepoints (autocorrelation matrices are n×n)
    n_regressors : int
        Number of regressors in design matrix
    use_double : bool, default=False
        If True, use float64 precision (8 bytes per element)

    Returns
    -------
    memory_bytes : int
        Total memory required in bytes
    """
    bytes_per_element = 8 if use_double else 4

    # Memory per (a,b) pair
    L_size = n_timepoints * n_timepoints * bytes_per_element  # Cholesky factor
    X_w_size = n_timepoints * n_regressors * bytes_per_element
    XwTXw_size = n_regressors * n_regressors * bytes_per_element
    scalars_size = 2 * bytes_per_element  # logdet_R + logdet_XwTXw

    per_pair_bytes = L_size + X_w_size + XwTXw_size + scalars_size

    # Total for all pairs
    total_bytes = n_valid_ab_pairs * per_pair_bytes

    return int(total_bytes)


def get_adaptive_batch_size(
    device: torch.device,
    n_timepoints: int,
    n_regressors: int,
    use_double: bool = False,
    use_qr: bool = False,
) -> int:
    """
    Recommend a per-call voxel batch size for ARMA(1,1) REML fits.

    Thin wrapper around ``fastfuncstuff.memory.estimate_chunk_size`` with
    ``operation="arma"`` so the whole codebase agrees on memory budgeting.
    GPU vs CPU defaults (safety factor, max chunk size) live in
    ``MemoryConfig``; tune them there if you want different policy.

    Parameters
    ----------
    device : torch.device
        Computing device
    n_timepoints, n_regressors : int
        Design dimensions
    use_double : bool, default=False
        Float64 (passed through; doubles per-voxel memory estimate).
    use_qr : bool, default=False
        Retained for backwards compatibility; the memory module does not
        currently differentiate the QR vs X'X paths, so the result is the
        same. The QR path uses a Q matrix the same shape as X_w, which is
        already bounded by the conservative per-voxel estimate.

    Returns
    -------
    batch_size : int
        Recommended number of voxels to process in parallel.
    """
    del use_qr  # currently unused, see docstring
    # n_voxels=None at this stage; estimate_chunk_size needs a number, so
    # pass the configured max so the result is the unconstrained best-fit
    # batch size. Callers that know n_voxels (e.g. fit_glm_arma11) will
    # clamp to it themselves.
    from fastfuncstuff.memory import estimate_chunk_size, get_memory_config

    config = get_memory_config()
    n_voxels_placeholder = (
        config.max_chunk_size_cpu if device.type != "cuda" else config.max_chunk_size_gpu
    )
    return estimate_chunk_size(
        n_voxels=n_voxels_placeholder,
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        device=device,
        operation="arma",
        use_double=use_double,
    )


def compute_arma_lambda(a: float, b: float) -> float:
    """
    Compute lambda (lag-1 correlation) for ARMA(1,1) parameters

    This is a public convenience wrapper around the internal function.

    Lambda = (b+a)(1+a*b)/(1+2*a*b+b²)

    Parameters
    ----------
    a : float
        AR parameter
    b : float
        MA parameter

    Returns
    -------
    lambda : float
        Lag-1 autocorrelation

    Notes
    -----
    AFNI constraint: lambda must be >= 0 (unless -NEGcor is used)

    Examples
    --------
    >>> compute_arma_lambda(0.5, 0.2)
    0.538...
    >>> compute_arma_lambda(0.5, -0.8)  # May give negative lambda!
    -0.xxx
    """
    return float(_compute_arma11_lambda(a, b))


def get_default_arma_grids(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get default ARMA(1,1) parameter grids — matches AFNI 3dREMLfit defaults.

    Returns
    -------
    a_grid : torch.Tensor
        AR parameter grid [0.0, 0.1, ..., 0.8] (9 points, step=0.1)
    b_grid : torch.Tensor
        MA parameter grid [-0.8, -0.7, ..., 0.7, 0.8] (17 points, step=0.1)

    Notes
    -----
    Total: 9 × 17 = 153 (a, b) combinations; the gamma0>0 stability filter
    inside build_arma11_covariance_batch trims this to ~117 valid points
    (matching AFNI's effective grid). Equivalent to 3dREMLfit's defaults
    `-MAXa 0.8 -MAXb 0.8 -Grid 3`.

    To widen the search (rare; only useful for high-correlation signals),
    pass custom grids via `a_grid` / `b_grid`, e.g.::

        a_grid = torch.arange(0.0, 0.91, 0.1, device=device)   # MAXa=0.9
        b_grid = torch.arange(-0.9, 0.91, 0.1, device=device)  # MAXb=0.9

    The absolute upper bound for either parameter is 0.9 — beyond that
    the ARMA(1,1) covariance becomes ill-conditioned.
    """
    a_grid = torch.linspace(*DEFAULT_ARMA_A_GRID, device=device)
    b_grid = torch.linspace(*DEFAULT_ARMA_B_GRID, device=device)
    return a_grid, b_grid


def ensure_zero_in_grid(
    a_grid: torch.Tensor, b_grid: torch.Tensor, tolerance: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure that both grids contain exactly 0.0, snapping or adding as needed.

    This guarantees that the special case (a=0, b=0) is always tested,
    even if the user provides a grid that starts at 0.1 or has spacing
    that would skip zero.

    Parameters
    ----------
    a_grid : torch.Tensor
        AR parameter grid
    b_grid : torch.Tensor
        MA parameter grid
    tolerance : float, default=1e-6
        Tolerance for detecting near-zero values.  Must be larger than float32
        linspace rounding error (~1.2e-7 for values near 1.0) to avoid false
        negatives, but small enough not to snap a genuine 0.001 grid point.

    Returns
    -------
    a_grid : torch.Tensor
        AR parameter grid with 0.0 included
    b_grid : torch.Tensor
        MA parameter grid with 0.0 included

    Notes
    -----
    The (a=0, b=0) case represents white noise (no autocorrelation),
    which is an important baseline for comparison. This function ensures
    it's always tested regardless of user grid specification.

    Two cases handled:
    - Near-zero value already exists (e.g. 2.98e-8 from float32 linspace):
      snap it to exactly 0.0 to avoid two near-identical grid points.
    - No near-zero value: insert 0.0 and re-sort.

    Examples
    --------
    >>> a_grid = torch.tensor([0.1, 0.2, 0.3])
    >>> b_grid = torch.tensor([0.1, 0.2, 0.3])
    >>> a_new, b_new = ensure_zero_in_grid(a_grid, b_grid)
    >>> print(a_new)  # [0.0, 0.1, 0.2, 0.3]
    >>> print(b_new)  # [0.0, 0.1, 0.2, 0.3]
    """
    device = a_grid.device
    dtype = a_grid.dtype

    def _ensure_zero(grid: torch.Tensor) -> torch.Tensor:
        near_zero = torch.abs(grid) < tolerance
        if torch.any(near_zero):
            # Snap the near-zero value(s) to exactly 0.0
            # (avoids duplicates from float32 linspace rounding, e.g. 2.98e-8)
            grid = grid.clone()
            grid[near_zero] = 0.0
        else:
            # Zero is genuinely absent — insert it
            zero_tensor = torch.tensor([0.0], device=device, dtype=dtype)
            grid = torch.sort(torch.cat([zero_tensor, grid]))[0]
        return grid

    a_grid = _ensure_zero(a_grid)
    b_grid = _ensure_zero(b_grid)

    return a_grid, b_grid


class ARMA11Results:
    """Container for ARMA(1,1) GLM results"""

    def __init__(self):
        self.betas: torch.Tensor | None = None  # (n_voxels, n_regressors) GLS parameter estimates
        self.tstats: torch.Tensor | None = None  # (n_voxels, n_regressors) t-statistics (corrected)
        self.r2: torch.Tensor | None = None  # (n_voxels,) R² values (total model)
        self.r2_partial: torch.Tensor | None = (
            None  # (n_voxels, n_task_regressors) partial R² per TASK condition
        )
        self.r2_partial_nuisance: torch.Tensor | None = (
            None  # (n_voxels, n_nuisance_regressors) partial R² per NUISANCE regressor
        )
        self.r2_semipartial: torch.Tensor | None = (
            None  # (n_voxels, n_task_regressors) semi-partial R² per TASK condition
        )
        self.r2_semipartial_nuisance: torch.Tensor | None = (
            None  # (n_voxels, n_nuisance_regressors) semi-partial R² per NUISANCE regressor
        )
        self.arma_params: torch.Tensor | None = None  # (n_voxels, 2) - (a, b) per voxel
        self.arma_lambda: torch.Tensor | None = None  # (n_voxels,) - lag-1 correlation
        self.reml_likelihood: torch.Tensor | None = (
            None  # (n_voxels,) - optimized REML log-likelihood
        )
        self.residuals: torch.Tensor | None = (
            None  # (n_voxels, n_timepoints) - residuals in original space
        )
        self.predicted: torch.Tensor | None = (
            None  # (n_voxels, n_timepoints) - predictions (original space)
        )
        self.residuals_whitened: torch.Tensor | None = (
            None  # (n_voxels, n_timepoints) - residuals after whitening
        )
        self.ljung_box: torch.Tensor | None = (
            None  # (n_voxels,) - whiteness stat of the whitened residuals (AFNI Rvar[5])
        )
        self.ljung_box_dof: int | None = None  # chi-squared DOF for ljung_box (h - 2)
        self.sigma2: torch.Tensor | None = None  # (n_voxels,) - noise variance estimates
        self.var_betas: torch.Tensor | None = (
            None  # (n_voxels, n_regressors, n_regressors) - covariance
        )
        self.original_shape: tuple[int, int, int] | None = None  # Original spatial dimensions
        self.fstats: torch.Tensor | None = (
            None  # (n_voxels,) - omnibus F-statistic across regressors
        )
        self.dof: int | None = None  # Degrees of freedom (n_timepoints - n_regressors)
        self.tr: float | None = None  # Repetition time
        self.voxel_mask: torch.Tensor | None = None  # Optional boolean mask for sparse analyses
        self.full_shape: tuple[int, int, int] | None = None  # Original spatial shape before masking

        # Design filtering metadata (tracks what was actually fitted)
        self.fitted_column_indices: list[int] | None = (
            None  # Indices of columns that were fitted (None = all columns)
        )
        self.n_regressors_full: int | None = (
            None  # Total columns in original design before filtering
        )

        # GLT contrast results (computed in-loop, not post-hoc)
        self.contrast_labels: list[str] | None = None  # List of contrast names
        self.contrast_betas: torch.Tensor | None = None  # (n_voxels, n_contrasts) - c'β estimates
        self.contrast_tstats: torch.Tensor | None = None  # (n_voxels, n_contrasts) - t-statistics
        self.contrast_fstats: torch.Tensor | None = (
            None  # (n_voxels, n_contrasts) - F-statistics (for multi-row GLTs)
        )
        self.contrast_r2_partial: torch.Tensor | None = (
            None  # (n_voxels, n_contrasts) - partial R² for each contrast
        )
        self.contrast_r2_semipartial: torch.Tensor | None = (
            None  # (n_voxels, n_contrasts) - semi-partial R² for each contrast
        )
        self.affine: np.ndarray | None = None  # Spatial affine if available
        self.ols_results: Any | None = None  # Optional GLMResults for OLS comparison

        # Voxel-wise (-dsort / ANATICOR) regressor results. The base task/nuisance
        # betas above are recomputed with the extended per-voxel design; the dsort
        # coefficients are kept separate (AFNI appends them last in -Rbeta).
        self.dsort_betas: torch.Tensor | None = (
            None  # (n_voxels, n_dsort) - per-voxel regressor coefficients
        )
        self.dsort_tstats: torch.Tensor | None = (
            None  # (n_voxels, n_dsort) - t-statistics for dsort coefficients
        )
        self.dsort_labels: list[str] | None = None  # one label per dsort regressor
        # When -dsort_nods is set, the no-dsort fit (base design only) is kept here.
        self.nods_results: Any | None = None  # ARMA11Results without dsort regressors

        # Full REML likelihood surface over (a,b) grid (optional, see save_lklhd_surface)
        self.reml_lklhd_surface: torch.Tensor | None = (
            None  # (n_voxels, n_valid_pairs) — L(a_k, b_k) per voxel
        )
        self.reml_surface_params: list | None = (
            None  # [(a_0,b_0), (a_1,b_1), ...] — grid points in column order
        )

        # Per-HRF merge: OLS write callback deferred until spatial metadata
        # (shape/mask/affine/header) is attached to the merged result.
        self._deferred_ols_write_callback: Callable | None = None


def _compute_arma11_lambda(
    a: float | torch.Tensor, b: float | torch.Tensor
) -> float | torch.Tensor:
    """
    Compute ARMA(1,1) lag-1 correlation (lambda) - SINGLE SOURCE OF TRUTH

    lambda = gamma1 / gamma0
           = [(a+b)(1+ab)/(1-a²)] / [(1+b²+2ab)/(1-a²)]
           = (a+b)(1+ab) / (1+b²+2ab)

    This is the ONLY place where this calculation should be implemented.
    Both scalar and batch versions use this function.

    Parameters
    ----------
    a : float or torch.Tensor
        AR parameter(s)
    b : float or torch.Tensor
        MA parameter(s)

    Returns
    -------
    lambda : float or torch.Tensor
        Lag-1 autocorrelation(s)

    Notes
    -----
    AFNI 3dREMLfit formula (see 3dREMLfit.c and the AFNI tech note).
    Valid for |a| < 1, |b| < 1 (checked by caller)
    """
    # This formula is mathematically equivalent to gamma1/gamma0
    # but more numerically stable (no (1-a²) division)
    numerator = (a + b) * (1 + a * b)
    denominator = 1 + b**2 + 2 * a * b
    return numerator / denominator


def _run_block_mask(
    run_starts: list[int] | torch.Tensor, n: int, device: torch.device
) -> torch.Tensor:
    """Boolean mask of shape (n, n) — True iff i and j are in the same run.

    Used to enforce AFNI's block-diagonal correlation structure: ARMA(1,1)
    correlations stop at run boundaries (concatenated runs are independent
    realisations). The mask is computed from a per-timepoint run id, so it
    is a pure tensor op — broadcasts cleanly over a batch of (n, n) Rs.
    """
    if isinstance(run_starts, torch.Tensor):
        starts = run_starts.to(device=device, dtype=torch.long).flatten()
    else:
        starts = torch.tensor(list(run_starts), device=device, dtype=torch.long)
    # run_id[i] = number of run_starts <= i, minus 1
    idx = torch.arange(n, device=device)
    run_id = torch.bucketize(idx, starts, right=True) - 1
    return run_id.unsqueeze(0) == run_id.unsqueeze(1)


def build_censor_run_info(
    run_starts: list[int] | torch.Tensor,
    n_total: int,
    good_list: list[int] | torch.Tensor | None = None,
) -> tuple[list[int], torch.Tensor | None]:
    """Unify run boundaries and within-run censoring into (run_starts, tau).

    This is the single place where the two concepts meet:

    - **Run boundary** — a hard cut. Concatenated runs are independent
      acquisitions; correlation must not reach across. Enforced downstream by the
      block mask (`_run_block_mask`), which needs run starts in *retained*
      (post-censor) index space.
    - **Censoring** — a "bad" timepoint that should not contribute (it is dropped
      from y and X), but whose neighbours are fine. Correlation reaches *across*
      the hole, weakened by the true time gap. Encoded by `tau`: each surviving
      point keeps its original within-run time index, so two survivors flanking a
      censored TR sit at lag-2 rather than lag-1.

    Parameters
    ----------
    run_starts : list[int] or torch.Tensor
        Run start indices in the *original* (pre-censor) concatenated timeline,
        e.g. [0, 200, 400] for three 200-TR runs.
    n_total : int
        Number of timepoints in the original (pre-censor) timeline.
    good_list : list[int] or torch.Tensor, optional
        Sorted indices (into the original timeline) of the timepoints that were
        kept. When None, no censoring — returns the inputs unchanged and tau=None.

    Returns
    -------
    run_starts_retained : list[int]
        Run starts re-expressed in retained-index space, for the block mask.
    tau : torch.Tensor or None
        Within-run true time index of each retained point (None if no censoring).
    """
    starts = run_starts.tolist() if isinstance(run_starts, torch.Tensor) else list(run_starts)
    if good_list is None:
        return [int(s) for s in starts], None

    retained = (
        (
            good_list
            if isinstance(good_list, torch.Tensor)
            else torch.tensor(list(good_list), dtype=torch.long)
        )
        .to(dtype=torch.long)
        .flatten()
    )

    starts_t = torch.tensor(starts, dtype=torch.long)
    # Original run id of each retained point, and that run's original start.
    run_id = torch.bucketize(retained, starts_t, right=True) - 1
    run_start_of = starts_t[run_id]
    # tau = original within-run time index (gaps from censored TRs preserved).
    tau = retained - run_start_of

    # Run starts in retained space: first retained point belonging to each run.
    n_runs = len(starts)
    run_starts_retained: list[int] = []
    for r in range(n_runs):
        pos = (run_id == r).nonzero(as_tuple=False)
        if pos.numel() > 0:
            run_starts_retained.append(int(pos[0].item()))
    if not run_starts_retained:
        run_starts_retained = [0]

    return run_starts_retained, tau


def build_arma11_covariance(
    a: float,
    b: float,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Build ARMA(1,1) covariance matrix (Toeplitz structure)

    R[i,j] = λ * a^|tau[i]-tau[j]|  where λ = lag-1 correlation

    This is the core correlation structure for ARMA(1,1) noise model.

    Parameters
    ----------
    a : float
        AR parameter (typically 0.2-0.9 for fMRI)
    b : float
        MA parameter (typically -0.3 to +0.3)
    n : int
        Number of timepoints
    device : torch.device
        Computing device
    dtype : torch.dtype, default=torch.float32
        Data type for the covariance matrix (float32 or float64)
    run_starts : list[int] or torch.Tensor, optional
        Starting timepoint of each run, in *retained* (post-censor) index space
        (e.g. [0, 150, 300] for 3 equal runs). When provided, R is block-diagonal
        across run boundaries (AFNI 3dREMLfit behaviour) — correlations do not
        bleed between concatenated runs. For single-run data leave as None.
    tau : torch.Tensor, optional
        True within-run time index of each retained timepoint, length n. The lag
        between points i and j is |tau[i]-tau[j]| rather than |i-j|, so a censored
        ("bad") timepoint leaves a hole the correlation steps across, weakened by
        the true time gap (AFNI's tau[] mechanism). When None, defaults to
        arange(n) — i.e. evenly spaced, no censoring (backward compatible).

    Returns
    -------
    R : torch.Tensor, shape (n, n)
        ARMA(1,1) covariance matrix, or None if parameters invalid

    Notes
    -----
    - λ >= 0 is enforced (AFNI default, invalid combinations return None)
    - Matrix is symmetric Toeplitz within each run-block; block-diagonal across runs
    - Fast construction: O(n²) but highly vectorized on GPU (~1ms for n=300)
    - Uses _compute_arma11_lambda() for single source of truth

    Special cases:
    - b=0: AR(1) with R[i,j] = a^|i-j|
    - a=0: MA(1) with R[i,j] = b for |i-j|=1, else 0
    """
    # Check parameter bounds
    if abs(a) >= 1 or abs(b) >= 1:
        return None

    # Compute lambda using the single source of truth function
    rho1 = _compute_arma11_lambda(a, b)

    # AFNI constraint: Only allow lambda >= 0 (unless -NEGcor is used)
    # Lambda=0 is valid (uncorrelated noise case)
    if rho1 < 0:
        return None

    # AFNI-faithful grid filter: near-white points (0 < lam < corcut) are folded
    # into the a=b=0 case, not evaluated separately (remla.c:1355).
    if _REML_MODE.afni_faithful and 0 < rho1 < _REML_MODE.corcut:
        return None

    # Lag = |tau[i] - tau[j]|. With tau=None this is the plain |i-j| Toeplitz;
    # with censoring tau carries the true time index so survivors flanking a
    # censored TR sit at their real distance (lag-2, not lag-1).
    if tau is None:
        tau_t = torch.arange(n, device=device)
    else:
        tau_t = tau.to(device=device, dtype=torch.long).flatten()
    distance = torch.abs(tau_t.unsqueeze(0) - tau_t.unsqueeze(1))
    max_lag = int(distance.max().item()) if n > 0 else 0

    # Correlation vector out to the largest lag present: [1, λ, λ*a, λ*a², ...]
    corr = torch.zeros(max_lag + 1, device=device, dtype=dtype)
    corr[0] = 1.0
    if max_lag >= 1:
        corr[1] = rho1
        if max_lag >= 2:
            powers = torch.full((max_lag - 1,), a, device=device, dtype=dtype)
            powers = torch.cumprod(powers, dim=0)
            corr[2:] = rho1 * powers

    R = corr[distance]

    # AFNI-faithful banded R: keep only the nearest bmax off-diagonals, so the
    # truncated correlation is ~corcut (remla.c:692). FFS default keeps the exact
    # dense Toeplitz, which is the true ARMA(1,1) covariance.
    if _REML_MODE.afni_faithful and rho1 > 0:
        bmax = _afni_bmax(a, rho1, _REML_MODE.corcut)
        if bmax == 0:
            R = torch.eye(n, device=device, dtype=dtype)
        else:
            R = R * (distance <= bmax).to(dtype=dtype)

    # Block-diagonal across run boundaries (AFNI behaviour for concatenated runs)
    if run_starts is not None:
        mask = _run_block_mask(run_starts, n, device)
        R = R * mask.to(dtype=dtype)

    # Ensure contiguous for MPS compatibility (PyTorch MPS Cholesky bug)
    R = R.contiguous()

    return R


def build_arma11_covariance_batch(
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """
    Build ALL ARMA(1,1) covariance matrices at once (VECTORIZED!)

    This is a MASSIVE speedup over loop-based construction:
    - 10-30x faster than calling build_arma11_covariance() in a loop
    - Single GPU kernel launch for all (a,b) combinations
    - Memory efficient with early filtering of invalid parameters
    - Uses _compute_arma11_lambda() for single source of truth (same math as scalar version)

    Parameters
    ----------
    a_grid : torch.Tensor, shape (n_a,)
        AR parameter grid
    b_grid : torch.Tensor, shape (n_b,)
        MA parameter grid
    n : int
        Number of timepoints
    device : torch.device
        Computing device
    dtype : torch.dtype, default=torch.float32
        Data type for the covariance matrices (float32 or float64)

    Returns
    -------
    R_batch : torch.Tensor, shape (n_valid, n, n)
        Covariance matrices for all valid (a,b) combinations
    params_tensor : torch.Tensor, shape (n_valid, 2)
        Valid (a, b) parameter pairs
    param_list : list of tuples
        List of (a, b) tuples for dictionary keys

    Notes
    -----
    Speedup mechanism:
    - Vectorized computation across entire (a,b) grid
    - Single call to torch.pow() for all power calculations
    - Efficient broadcasting for Toeplitz structure
    - Early filtering of invalid parameters (|a|>=1, |b|>=1, gamma0<=0)

    Example
    -------
    >>> a_grid = torch.linspace(0.1, 0.9, 9)
    >>> b_grid = torch.linspace(-0.3, 0.3, 7)
    >>> R_batch, params, keys = build_arma11_covariance_batch(a_grid, b_grid, 300, device)
    >>> print(f"Built {len(params)} covariance matrices in one shot!")
    """
    # Create all (a,b) combinations using meshgrid
    a_mesh, b_mesh = torch.meshgrid(a_grid, b_grid, indexing="ij")
    a_flat = a_mesh.reshape(-1)  # (n_grid,)
    b_flat = b_mesh.reshape(-1)  # (n_grid,)

    # Filter valid parameters BEFORE expensive computation
    valid_mask = (torch.abs(a_flat) < 1.0) & (torch.abs(b_flat) < 1.0)

    # Additional validity checks
    denom = 1 - a_flat**2
    valid_mask &= torch.abs(denom) >= 1e-6

    gamma0 = (1 + b_flat**2 + 2 * a_flat * b_flat) / (denom + 1e-10)
    valid_mask &= gamma0 > 0

    # Extract only valid parameters
    a_valid = a_flat[valid_mask]  # (n_valid,)
    b_valid = b_flat[valid_mask]  # (n_valid,)
    n_valid = len(a_valid)

    if n_valid == 0:
        # No valid parameters - return empty
        return (
            torch.empty(0, n, n, device=device),
            torch.empty(0, 2, device=device),
            [],
        )

    # Compute lambda using SINGLE SOURCE OF TRUTH function
    # This ensures batch and scalar versions use identical math
    rho1_valid_raw = _compute_arma11_lambda(a_valid, b_valid)  # (n_valid,)
    # Type narrowing: ensure it's a Tensor (it will be since inputs are Tensors)
    if not isinstance(rho1_valid_raw, torch.Tensor):
        raise TypeError("Expected rho1_valid to be a Tensor")
    rho1_valid = cast(torch.Tensor, rho1_valid_raw)  # Explicit cast for pyright

    # AFNI constraint: Only allow lambda >= 0 (unless -NEGcor is used)
    # Lambda=0 is valid (uncorrelated noise case)
    lambda_mask = rho1_valid >= 0
    # AFNI-faithful grid filter: drop near-white points (0 < lam < corcut), which
    # AFNI folds into the a=b=0 case (remla.c:1355). lam==0 is kept.
    if _REML_MODE.afni_faithful:
        lambda_mask &= (rho1_valid == 0) | (rho1_valid >= _REML_MODE.corcut)
    a_valid = a_valid[lambda_mask]
    b_valid = b_valid[lambda_mask]
    rho1_valid = rho1_valid[lambda_mask]
    n_valid = len(a_valid)

    if n_valid == 0:
        # No valid parameters after lambda filtering - return empty
        return (
            torch.empty(0, n, n, device=device),
            torch.empty(0, 2, device=device),
            [],
        )

    # Lag = |tau[i] - tau[j]| (see scalar build_arma11_covariance). tau=None gives
    # the plain |i-j| Toeplitz; censoring lengthens the gap across bad timepoints.
    if tau is None:
        tau_t = torch.arange(n, device=device)
    else:
        tau_t = tau.to(device=device, dtype=torch.long).flatten()
    distance = torch.abs(tau_t.unsqueeze(0) - tau_t.unsqueeze(1))  # (n, n)
    max_lag = int(distance.max().item()) if n > 0 else 0

    # Correlation vectors for ALL parameters: [1, λ, λ*a, λ*a², ...] out to max_lag.
    # This must match the scalar version exactly!
    corr = torch.zeros(n_valid, max_lag + 1, device=device, dtype=dtype)
    corr[:, 0] = 1.0  # lag-0 correlation is always 1
    if max_lag >= 1:
        corr[:, 1] = rho1_valid  # lag-1 correlation is lambda
        if max_lag >= 2:
            # lags k≥2: corr[k] = λ * a^(k-1)
            exponents = torch.arange(1, max_lag, device=device, dtype=dtype)
            powers = a_valid.unsqueeze(1) ** exponents.unsqueeze(0)  # (n_valid, max_lag-1)
            corr[:, 2:] = rho1_valid.unsqueeze(1) * powers

    # Broadcast: corr is (n_valid, max_lag+1), distance is (n, n)
    # Result: (n_valid, n, n) - all covariance matrices at once!
    R_batch = corr[:, distance]

    # AFNI-faithful banded R: per-(a,b) bmax truncation (remla.c:692). Each matrix
    # keeps only its own nearest bmax off-diagonals; the default keeps dense R.
    if _REML_MODE.afni_faithful:
        bmax = torch.tensor(
            [
                _afni_bmax(a.item(), lam.item(), _REML_MODE.corcut)
                for a, lam in zip(a_valid, rho1_valid, strict=True)
            ],
            device=device,
            dtype=torch.long,
        )  # (n_valid,)
        # The diagonal (distance 0) is always kept by `<=`, so bmax==0 correctly
        # yields the identity (near-white) matrix.
        band = distance.unsqueeze(0) <= bmax.view(-1, 1, 1)  # (n_valid, n, n)
        R_batch = R_batch * band.to(dtype=dtype)

    # Block-diagonal across run boundaries (AFNI behaviour for concatenated runs)
    if run_starts is not None:
        mask = _run_block_mask(run_starts, n, device).to(dtype=dtype)
        R_batch = R_batch * mask  # broadcasts over the n_valid axis

    # Ensure contiguous for MPS compatibility
    R_batch = R_batch.contiguous()

    # Create parameter tensor and list for output
    params_tensor = torch.stack([a_valid, b_valid], dim=1)  # (n_valid, 2)
    param_list = [(a.item(), b.item()) for a, b in params_tensor]

    return R_batch, params_tensor, param_list


def compute_reml_likelihood(X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor) -> float:
    """
    Compute REML (Restricted Maximum Likelihood) log-likelihood

    L(a,b) = log(det(R)) + log(det(X'R^(-1)X)) + (n-m)log(Y'P Y)

    where P = R^(-1) - R^(-1)X(X'R^(-1)X)^(-1)X'R^(-1) (projection matrix)

    This is the objective function minimized in AFNI 3dREMLfit to find optimal (a,b).

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    Y : torch.Tensor, shape (n_timepoints,)
        Data vector for single voxel
    R : torch.Tensor, shape (n_timepoints, n_timepoints)
        ARMA(1,1) covariance matrix

    Returns
    -------
    likelihood : float
        REML log-likelihood (smaller = better fit)

    Notes
    -----
    Implementation uses Cholesky decomposition for numerical stability:
    - R = L L' → log(det(R)) = 2 * sum(log(diag(L)))
    - Prewhitening: L^(-1) applied to X and Y
    - Fast on GPU: ~10ms for n=300 timepoints

    References
    ----------
    - AFNI 3dREMLfit_mathnotes.pdf (equation on page 4)
    - Worsley & Friston (1995) for REML in fMRI context
    """
    n_timepoints, n_regressors = X.shape
    device = X.device

    # solve_triangular requires 2D input
    Y_2d = Y.unsqueeze(1) if Y.ndim == 1 else Y

    try:
        # Cholesky decomposition: R = L L'
        # Conservative CPU fallback on MPS (the result is float32 here, since
        # use_double is forced off on MPS) — factor on CPU, move back.
        if device.type == "mps":
            R_cpu = R.cpu()
            L = torch.linalg.cholesky(R_cpu).to(device)
        else:
            L = torch.linalg.cholesky(R)

        # Term 1: log(det(R)) = 2 * sum(log(diag(L)))
        term1 = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10))

        # Prewhiten design and data via triangular solve
        X_w = torch.linalg.solve_triangular(L, X, upper=False)
        Y_w = torch.linalg.solve_triangular(L, Y_2d, upper=False)

        # Term 2: log(det(X' R^(-1) X)) = log(det(X_w' X_w))
        XwTXw = X_w.T @ X_w
        XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device, dtype=XwTXw.dtype)

        # Use logdet for numerical stability
        sign, logdet_val = torch.linalg.slogdet(XwTXw_reg)
        term2 = logdet_val if sign > 0 else 1e10  # Large penalty if singular

        # Term 3: (n-m) log(Y' P Y)
        # P = R^(-1) - R^(-1)X(X'R^(-1)X)^(-1)X'R^(-1)
        # Simplified via prewhitening: Y'PY = ||Y_w - X_w β_w||²
        beta_w = torch.linalg.solve(XwTXw_reg, X_w.T @ Y_w)
        residuals_w = Y_w - X_w @ beta_w
        rss = torch.sum(residuals_w**2)

        term3 = (n_timepoints - n_regressors) * torch.log(rss + 1e-10)

        # Total likelihood (smaller = better)
        likelihood = term1 + term2 + term3

        return likelihood.item()

    except RuntimeError as e:
        # Cholesky failed - return large penalty
        warnings.warn(f"Cholesky decomposition failed: {e}", stacklevel=2)
        return 1e10


def reml_grid_search(
    X: torch.Tensor,
    Y: torch.Tensor,
    a_grid: torch.Tensor | None = None,
    b_grid: torch.Tensor | None = None,
    device: torch.device | None = None,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """
    Find optimal ARMA(1,1) parameters via REML grid search

    This is the core parameter estimation method used by AFNI 3dREMLfit.
    Searches over discrete grid of (a, b) values to minimize REML likelihood.

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    Y : torch.Tensor, shape (n_timepoints,)
        Data vector (single voxel timeseries)
    a_grid : torch.Tensor, optional
        Grid of AR parameters to search (default: 0.1 to 0.9 in steps of 0.1)
    b_grid : torch.Tensor, optional
        Grid of MA parameters to search (default: -0.3 to 0.3 in steps of 0.1)
    device : torch.device, optional
        Computing device

    Returns
    -------
    a_opt : float
        Optimal AR parameter
    b_opt : float
        Optimal MA parameter
    likelihood_opt : float
        Minimum REML likelihood achieved

    Notes
    -----
    Default grid (AFNI -Grid 3 equivalent - standardized across all functions):
    - a: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] (9 values)
    - b: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] (7 values)
    - Total: 63 (a,b) combinations

    This grid is used consistently across:
    - reml_grid_search() (single voxel)
    - batch_reml_grid_search() (vectorized)
    - fit_glm_arma11() (full analysis)

    For higher resolution (AFNI -Grid 5):
    - a: 0.1 to 0.9 in steps of 0.05 (17 values)
    - b: -0.3 to 0.3 in steps of 0.05 (13 values)
    - Total: 221 combinations

    GPU Speed: ~1s pre-computation + ~0.5s per 10k voxels (with MPS)

    References
    ----------
    - AFNI 3dREMLfit uses discrete grid search (not continuous optimization)
    - Default grid balances accuracy vs speed
    - Finer grids don't substantially change results in practice
    """
    if device is None:
        device = get_device()

    X = X.to(device)
    Y = Y.to(device)

    # Default grids
    if a_grid is None and b_grid is None:
        a_grid, b_grid = get_default_arma_grids(device)
    else:
        if a_grid is None:
            a_grid, _ = get_default_arma_grids(device)
        else:
            a_grid = to_tensor(a_grid, device=device)

        if b_grid is None:
            _, b_grid = get_default_arma_grids(device)
        else:
            b_grid = to_tensor(b_grid, device=device)

    n_timepoints = X.shape[0]

    # Grid search
    best_likelihood = float("inf")
    best_a = 0.5
    best_b = 0.0

    # Type assertion: grids are guaranteed to be set by now
    assert a_grid is not None and b_grid is not None

    for a in a_grid:
        for b in b_grid:
            a_val = a.item()
            b_val = b.item()

            # Build ARMA(1,1) covariance matrix
            R = build_arma11_covariance(
                a_val, b_val, n_timepoints, device, run_starts=run_starts, tau=tau
            )

            if R is None:
                # Invalid (a,b) combination
                continue

            # Compute REML likelihood
            likelihood = compute_reml_likelihood(X, Y, R)

            if likelihood < best_likelihood:
                best_likelihood = likelihood
                best_a = a_val
                best_b = b_val

    return best_a, best_b, best_likelihood


def precompute_autocorr_grid(
    n_timepoints: int,
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    device: torch.device,
    *,
    verbose: bool = False,
    cholesky_on_cpu: bool = True,
    dtype: torch.dtype = torch.float32,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> dict:
    """Design-independent half of the REML grid: per-(a,b) L⁻¹ and logdet(R).

    These quantities depend ONLY on the autocorrelation structure (a, b,
    run_starts, n_timepoints) — not on the design matrix X. Computing them once
    and reusing them across multiple designs (e.g., per-voxel HRFs) saves a full
    Cholesky pass per design.

    Returns
    -------
    dict
        ``{"param_list": [(a,b), ...], "L_inv_batch": (n_valid, n, n) tensor,
        "logdet_Rcorr_batch": (n_valid,) tensor, "dtype": torch.dtype}``.
        Tensors are on CPU when ``cholesky_on_cpu=True``.
    """
    import time as _time

    build_device = torch.device("cpu") if (cholesky_on_cpu or device.type == "mps") else device
    a_grid = a_grid.to(build_device)
    b_grid = b_grid.to(build_device)
    a_grid, b_grid = ensure_zero_in_grid(a_grid, b_grid)

    if verbose:
        n_total_grid = len(a_grid) * len(b_grid)
        print("  Building ALL covariance matrices (vectorized)...")
        print(f"    Initial grid: {len(a_grid)} a × {len(b_grid)} b = {n_total_grid} combinations")

    R_batch, _params_tensor, param_list = build_arma11_covariance_batch(
        a_grid,
        b_grid,
        n_timepoints,
        build_device,
        dtype,
        run_starts=run_starts,
        tau=tau,
    )
    n_valid = len(param_list)
    if n_valid == 0:
        if verbose:
            print("  Warning: No valid (a,b) parameters in grid!")
        return {
            "param_list": [],
            "L_inv_batch": None,
            "logdet_Rcorr_batch": None,
            "dtype": dtype,
        }

    # GPU streaming Cholesky: when the all-at-once GPU path doesn't fit but
    # device is CUDA, process the (a,b) grid in chunks on the GPU and stream
    # L⁻¹ back to CPU. Much faster than CPU Cholesky on the same workload.
    use_chunked_gpu = (
        cholesky_on_cpu  # all-at-once was rejected
        and device.type == "cuda"
        and build_device.type == "cpu"
    )

    chol_start = _time.time()
    if use_chunked_gpu:
        # Pick chunk size by GPU free memory. Per-chunk peak ≈ 3·chunk·n² bytes/elem
        # (R_chunk, L_chunk, L_inv_chunk all coexist). Add 1.5× safety for PyTorch
        # workspace and the persistent eye/run_block_mask tensors.
        bpe = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        per_pair_bytes = n_timepoints * n_timepoints * bpe
        # Cap chunk: cuSOLVER workspace grows with batch — empirically batches of
        # ~32 give the best amortization without ballooning workspace.
        chunk_size = max(
            1,
            min(32, int(free_bytes * 0.5 / (per_pair_bytes * 3))),
        )

        if verbose:
            n_filtered = len(a_grid) * len(b_grid) - n_valid
            print(f"  ✓ Built {n_valid} covariance matrices (filtered {n_filtered} with λ ≤ 0)")
            print(
                f"  🚀 Streaming Cholesky on GPU in chunks of {chunk_size} "
                f"(per-chunk peak ≈ {per_pair_bytes * chunk_size * 3 / 1e9:.2f} GB)..."
            )

        L_inv_batch = torch.empty(
            (n_valid, n_timepoints, n_timepoints), dtype=dtype, device=build_device
        )
        logdet_Rcorr_batch = torch.empty(n_valid, dtype=dtype, device=build_device)
        eye_gpu = torch.eye(n_timepoints, dtype=dtype, device=device)
        try:
            for start in range(0, n_valid, chunk_size):
                end = min(start + chunk_size, n_valid)
                R_chunk_gpu = R_batch[start:end].to(device, non_blocking=True)
                L_chunk = torch.linalg.cholesky(R_chunk_gpu)
                diag = torch.diagonal(L_chunk, dim1=1, dim2=2)
                logdet_Rcorr_batch[start:end] = (
                    2 * torch.sum(torch.log(diag + 1e-10), dim=1)
                ).cpu()
                L_inv_chunk = torch.linalg.solve_triangular(
                    L_chunk,
                    eye_gpu.unsqueeze(0).expand(end - start, -1, -1),
                    upper=False,
                )
                L_inv_batch[start:end] = L_inv_chunk.cpu()
                del R_chunk_gpu, L_chunk, L_inv_chunk
            del eye_gpu
            torch.cuda.empty_cache()
        except (torch.linalg.LinAlgError, RuntimeError) as _e:
            # GPU chunk failed — fall back to all-at-once CPU
            if verbose:
                print(f"  ⚠️  GPU chunked Cholesky failed ({_e}); falling back to CPU.")
            del L_inv_batch
            torch.cuda.empty_cache()
            L_batch_cpu = torch.linalg.cholesky(R_batch)
            logdet_Rcorr_batch = 2 * torch.sum(
                torch.log(torch.diagonal(L_batch_cpu, dim1=1, dim2=2) + 1e-10), dim=1
            )
            eye_cpu = torch.eye(n_timepoints, dtype=dtype, device=R_batch.device)
            L_inv_batch = torch.linalg.solve_triangular(
                L_batch_cpu, eye_cpu.unsqueeze(0).expand(n_valid, -1, -1), upper=False
            )
            del L_batch_cpu, eye_cpu
    else:
        if verbose:
            n_filtered = len(a_grid) * len(b_grid) - n_valid
            chol_location = "CPU" if cholesky_on_cpu else "GPU (all-at-once)"
            print(f"  ✓ Built {n_valid} covariance matrices (filtered {n_filtered} with λ ≤ 0)")
            print(f"  Computing ALL Cholesky factorizations (batched on {chol_location})...")
        L_batch = torch.linalg.cholesky(R_batch)
        logdet_Rcorr_batch = 2 * torch.sum(
            torch.log(torch.diagonal(L_batch, dim1=1, dim2=2) + 1e-10), dim=1
        )
        eye_n = torch.eye(n_timepoints, dtype=dtype, device=L_batch.device)
        L_inv_batch = torch.linalg.solve_triangular(
            L_batch, eye_n.unsqueeze(0).expand(n_valid, -1, -1), upper=False
        )
        del L_batch, eye_n

    if verbose:
        print(f"  ✓ Computed {n_valid} Cholesky factorizations! ({_time.time() - chol_start:.1f}s)")
    del R_batch

    return {
        "param_list": param_list,
        "L_inv_batch": L_inv_batch,
        "logdet_Rcorr_batch": logdet_Rcorr_batch,
        "dtype": dtype,
    }


def precompute_reml_grid(
    X: torch.Tensor,
    n_timepoints: int,
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
    cholesky_on_cpu: bool = True,
    dtype: torch.dtype = torch.float32,
    debug_memory: bool = False,
    use_qr: bool = False,
    run_starts: list[int] | torch.Tensor | None = None,
    autocorr_cache: dict | None = None,
    tau: torch.Tensor | None = None,
) -> dict:
    """
    Pre-compute Cholesky factorizations for entire REML grid (BATCHED!)

    This is the CRITICAL optimization from AFNI 3dREMLfit, now with GPU batching:
    - Build ALL covariance matrices at once (10-30x faster!)
    - Compute ALL Cholesky factorizations in one batch (5-20x faster!)
    - Store all factorizations in memory
    - Reuse for every voxel = MASSIVE speedup!

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    n_timepoints : int
        Number of timepoints
    a_grid : torch.Tensor
        AR parameter grid
    b_grid : torch.Tensor
        MA parameter grid
    device : torch.device
        Computing device
    verbose : bool, default=False
        Show progress bar
    cholesky_on_cpu : bool, default=True
        Compute Cholesky decompositions on CPU to save GPU memory.
        Recommended for large grids (>200 combinations). Still very fast!
    dtype : torch.dtype, default=torch.float32
        Data type for covariance matrices (float32 or float64)
    use_qr : bool, default=False
        If True, use QR factorization (AFNI approach, more stable, uses 2.25x more memory).
        If False, use X'X (faster, less memory, good for well-conditioned matrices with float64).

    Returns
    -------
    precomputed : dict
        Contains pre-computed matrices for each valid (a,b):
        - 'L_inv': lower Cholesky inverse — use L_inv @ Y for GEMM-based whitening
        - 'X_w': prewhitened design matrix — kept for GLS fitting path
        - 'Q': orthonormal columns of X_w — use Q'Y_w for Pythagorean RSS
        - 'logdet_Rcorr': log(det(R))
        - 'logdet_XwTXw': log(det(X'R^-1 X))
    """
    # Ensure X has the correct dtype for all operations
    X = X.to(dtype=dtype)
    n_regressors = X.shape[1]
    precomputed = {}

    # Phases 1-3 (build R, Cholesky, L_inv, logdet_Rcorr) depend ONLY on (a,b)
    # and run_starts. They can be shared across designs via autocorr_cache.
    if autocorr_cache is None:
        autocorr_cache = precompute_autocorr_grid(
            n_timepoints,
            a_grid,
            b_grid,
            device,
            verbose=verbose,
            cholesky_on_cpu=cholesky_on_cpu,
            dtype=dtype,
            run_starts=run_starts,
            tau=tau,
        )
    elif verbose:
        print(
            f"  ♻️  Reusing cached autocorr grid "
            f"({len(autocorr_cache.get('param_list', []))} (a,b) pairs)"
        )

    param_list = autocorr_cache["param_list"]
    L_inv_batch = autocorr_cache["L_inv_batch"]
    logdet_Rcorr_batch = autocorr_cache["logdet_Rcorr_batch"]
    n_valid = len(param_list)
    if n_valid == 0:
        return precomputed

    if verbose:
        print("  Prewhitening design matrix for all parameters...")

    try:
        if debug_memory:
            _debug_memory_snapshot(
                "Grid: Entering design-dependent phase",
                device,
                {"L_inv_batch": L_inv_batch},
            )

        # PHASE 4-5: prewhiten design + compute Q.
        # When L_inv lives on CPU but a CUDA device is available, stream chunks
        # to GPU: phase 4 is a (n_valid, n_t, n_reg) bmm that's wildly slow on
        # CPU for big n_t (the 16 s CPU usage you noticed). On GPU it's
        # milliseconds. Storage remains CPU to keep the policy unchanged.
        n_regressors_x = X.shape[1]
        gpu_stream_design = L_inv_batch.device.type == "cpu" and device.type == "cuda"

        if gpu_stream_design:
            bpe = dtype.itemsize if hasattr(dtype, "itemsize") else 4
            free_bytes, _ = torch.cuda.mem_get_info(device)
            # Per-chunk peak ≈ L_inv (n_t²) + X_w (n_t · n_reg) + Q (n_t · n_reg)
            #                + XtX (n_reg²) + 2× workspace ≈ ~2× L_inv chunk.
            per_pair_bytes = n_timepoints * n_timepoints * bpe * 2
            chunk_size = max(1, min(32, int(free_bytes * 0.5 / per_pair_bytes)))
            if verbose:
                print(f"  🚀 Streaming design prewhitening on GPU in chunks of {chunk_size}...")

            X_gpu = X.to(device=device, dtype=dtype)
            X_w_batch = torch.empty(
                (n_valid, n_timepoints, n_regressors_x), dtype=dtype, device="cpu"
            )
            Q_batch = torch.empty_like(X_w_batch)
            logdet_XwTXw_batch = torch.empty(n_valid, dtype=dtype, device="cpu")

            cholesky_q_failed = False
            for start in range(0, n_valid, chunk_size):
                end = min(start + chunk_size, n_valid)
                L_inv_chunk = L_inv_batch[start:end].to(device, non_blocking=True)
                X_w_chunk = torch.bmm(L_inv_chunk, X_gpu.unsqueeze(0).expand(end - start, -1, -1))
                # Cholesky-of-XtX path for Q. If it fails, mark and fall back below.
                try:
                    XtX_chunk = torch.bmm(X_w_chunk.transpose(1, 2), X_w_chunk)
                    L_xtx_chunk = torch.linalg.cholesky(XtX_chunk)
                    Q_chunk = torch.linalg.solve_triangular(
                        L_xtx_chunk, X_w_chunk.transpose(1, 2), upper=False
                    ).transpose(1, 2)
                    logdet_chunk = 2 * torch.sum(
                        torch.log(torch.abs(torch.diagonal(L_xtx_chunk, dim1=1, dim2=2)) + 1e-10),
                        dim=1,
                    )
                    del XtX_chunk, L_xtx_chunk
                except (torch.linalg.LinAlgError, RuntimeError):
                    Q_chunk, R_qr_chunk = torch.linalg.qr(X_w_chunk)
                    logdet_chunk = 2 * torch.sum(
                        torch.log(torch.abs(torch.diagonal(R_qr_chunk, dim1=1, dim2=2)) + 1e-10),
                        dim=1,
                    )
                    del R_qr_chunk
                    cholesky_q_failed = True
                X_w_batch[start:end] = X_w_chunk.cpu()
                Q_batch[start:end] = Q_chunk.cpu()
                logdet_XwTXw_batch[start:end] = logdet_chunk.cpu()
                del L_inv_chunk, X_w_chunk, Q_chunk, logdet_chunk
            del X_gpu
            torch.cuda.empty_cache()
            if cholesky_q_failed and verbose:
                print(
                    "  ⚠️  Cholesky-of-XtX failed on some chunks; "
                    "fell back to QR for those (still on GPU)."
                )

        else:
            # Original path: build_device == device (both CPU or both GPU all-at-once).
            X_cpu = X.to(L_inv_batch.device)
            X_w_batch = torch.bmm(
                L_inv_batch, X_cpu.unsqueeze(0).expand(n_valid, -1, -1)
            )  # (n_valid, n_timepoints, n_regressors)
            del X_cpu

        # PHASE 5 (non-streamed path): the streaming path computed Q + logdet
        # inline above. Only execute when we didn't stream.
        if not gpu_stream_design:
            try:
                XtX_batch = torch.bmm(X_w_batch.transpose(1, 2), X_w_batch)
                L_xtx = torch.linalg.cholesky(XtX_batch)
                Q_batch = torch.linalg.solve_triangular(
                    L_xtx,
                    X_w_batch.transpose(1, 2),
                    upper=False,
                ).transpose(1, 2)
                logdet_XwTXw_batch = 2 * torch.sum(
                    torch.log(torch.abs(torch.diagonal(L_xtx, dim1=1, dim2=2)) + 1e-10),
                    dim=1,
                )
                del XtX_batch, L_xtx
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except (torch.linalg.LinAlgError, RuntimeError) as e:
                is_oom = isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
                is_linalg = isinstance(e, torch.linalg.LinAlgError)
                if not (is_oom or is_linalg):
                    raise
                if verbose:
                    reason = "OOM" if is_oom else "Cholesky failed (rank-deficient X)"
                    print(
                        f"  Fast Q path failed ({reason}); falling back to "
                        f"torch.linalg.qr — slower but more numerically robust."
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                Q_batch, R_qr_batch = torch.linalg.qr(X_w_batch)
                logdet_XwTXw_batch = 2 * torch.sum(
                    torch.log(torch.abs(torch.diagonal(R_qr_batch, dim1=1, dim2=2)) + 1e-10),
                    dim=1,
                )
                del R_qr_batch

        if verbose:
            print("  ✓ Precomputed all matrices!")
            print(f"  Storing {n_valid} parameter sets...")

        # PHASE 6: Store per grid point on CPU
        # L_inv: GEMM-based whitening in search (no triangular solve in hot path)
        # X_w: kept for GLS fitting (design is constant across all voxels in a group)
        # Q: orthonormal cols for Pythagorean RSS = ||Y_w||² - ||Q'Y_w||²
        for i, (a_val, b_val) in enumerate(param_list):
            precomputed[(a_val, b_val)] = {
                "L_inv": L_inv_batch[i],  # (n, n) - lower Cholesky inverse
                "X_w": X_w_batch[i],  # (n, n_reg) - prewhitened design
                "Q": Q_batch[i],  # (n, n_reg) - orthonormal cols for RSS
                "logdet_Rcorr": logdet_Rcorr_batch[i],
                "logdet_XwTXw": logdet_XwTXw_batch[i],
                "a": a_val,
                "b": b_val,
            }

    except RuntimeError as e:
        # Fallback to sequential if batch fails
        if verbose:
            warnings.warn(
                f"Batch Cholesky failed ({e}), falling back to sequential...", stacklevel=2
            )

        # CRITICAL: Clean up GPU memory from failed batch attempt!
        _local_names = locals()
        if "L_inv_batch" in _local_names:
            del L_inv_batch
        if "X_w_batch" in _local_names:
            del X_w_batch
        if "Q_batch" in _local_names:
            del Q_batch

        # Force GPU memory release
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if verbose:
            print("  ✓ Cleared GPU memory from failed batch attempt")

        # Fall back to old sequential method
        grid_pairs = (
            tqdm(param_list, desc="Precomputing REML grid (sequential)", unit="pair")
            if verbose
            else param_list
        )

        for a_val, b_val in grid_pairs:
            R = build_arma11_covariance(
                a_val,
                b_val,
                n_timepoints,
                device,
                dtype,
                run_starts=run_starts,
                tau=tau,
            )
            if R is None:
                continue

            try:
                # Compute Cholesky on CPU
                if cholesky_on_cpu or device.type == "mps":
                    R_cpu = R.cpu()
                    L = torch.linalg.cholesky(R_cpu)
                    del R_cpu
                else:
                    L = torch.linalg.cholesky(R).cpu()

                logdet_Rcorr = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10))

                # Compute L_inv — GEMM-based whitening avoids per-voxel triangular solve
                n_t = L.shape[0]
                L_inv = torch.linalg.solve_triangular(
                    L, torch.eye(n_t, dtype=dtype, device=L.device), upper=False
                )
                del L  # L no longer needed; L_inv replaces it

                X_cpu = X.cpu() if X.device.type != "cpu" else X
                X_w = L_inv @ X_cpu  # GEMM: (n_time, n_reg)

                # QR always — eliminates slogdet/DLASWP, enables Pythagorean RSS
                Q, R_qr = torch.linalg.qr(X_w)
                logdet_XwTXw = 2 * torch.sum(torch.log(torch.abs(torch.diag(R_qr)) + 1e-10))
                del R_qr  # Not stored; GLS path recomputes fresh QR from X_w

                precomputed[(a_val, b_val)] = {
                    "L_inv": L_inv,
                    "X_w": X_w,
                    "Q": Q,
                    "logdet_Rcorr": logdet_Rcorr,
                    "logdet_XwTXw": logdet_XwTXw,
                    "a": a_val,
                    "b": b_val,
                }
            except RuntimeError as e2:
                if verbose:
                    warnings.warn(
                        f"Cholesky failed for (a={a_val:.3f}, b={b_val:.3f}): {e2}", stacklevel=2
                    )
                continue

    # CRITICAL: Delete batch tensors to free memory before GPU transfer
    if debug_memory:
        _debug_memory_snapshot(
            "Grid: Before deleting batch tensors",
            device,
            {
                "L_inv_batch": locals().get("L_inv_batch"),
                "X_w_batch": locals().get("X_w_batch"),
                "Q_batch": locals().get("Q_batch"),
            },
        )

        if "L_inv_batch" in locals():
            n_pairs = len(param_list)
            n_regs = X.shape[1]
            bytes_per_elem = L_inv_batch.element_size()

            expected_L_inv = n_pairs * n_timepoints * n_timepoints * bytes_per_elem
            expected_X_w = n_pairs * n_timepoints * n_regs * bytes_per_elem
            expected_Q = n_pairs * n_timepoints * n_regs * bytes_per_elem
            expected_scalars = n_pairs * 2 * bytes_per_elem
            expected_total = expected_L_inv + expected_X_w + expected_Q + expected_scalars

            actual_L_inv = L_inv_batch.element_size() * L_inv_batch.nelement()
            actual_X_w = X_w_batch.element_size() * X_w_batch.nelement()
            actual_Q = Q_batch.element_size() * Q_batch.nelement()
            actual_total = actual_L_inv + actual_X_w + actual_Q

            print("\nGRID SIZE VERIFICATION:")
            print(
                f"  n_pairs={n_pairs}, n_timepoints={n_timepoints}, n_regressors={n_regs}, dtype={dtype}"
            )
            print(f"  Expected grid size: {expected_total / 1024**3:.3f} GiB")
            print(f"    - L_inv: {expected_L_inv / 1024**3:.3f} GiB")
            print(f"    - X_w:   {expected_X_w / 1024**3:.3f} GiB")
            print(f"    - Q:     {expected_Q / 1024**3:.3f} GiB")
            print(f"  Actual batch tensors: {actual_total / 1024**3:.3f} GiB")
            print(
                f"  Match: {'✅ YES' if abs(actual_total - expected_total) < 1024**2 else '❌ NO'}"
            )

    # `del` only frees if the name is bound; ``in locals()`` guards the names
    # that the fallback path may have left unset.
    _local_names = locals()
    if "L_inv_batch" in _local_names:
        del L_inv_batch
    if "X_w_batch" in _local_names:
        del X_w_batch
    if "Q_batch" in _local_names:
        del Q_batch
    if "logdet_Rcorr_batch" in _local_names:
        del logdet_Rcorr_batch
    if "logdet_XwTXw_batch" in _local_names:
        del logdet_XwTXw_batch

    if debug_memory:
        _debug_memory_snapshot("Grid: After deleting batch tensors (grid in dict only)", device, {})

    return precomputed


@torch.inference_mode()
def batch_reml_grid_search(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    a_grid: torch.Tensor | None = None,
    b_grid: torch.Tensor | None = None,
    device: torch.device | None = None,
    precomputed: dict | None = None,
    dtype: torch.dtype = torch.float32,
    enable_early_stopping: bool = False,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exhaustive batch-parallel REML grid search.

    Evaluates all valid (a, b) grid points across the entire voxel batch using
    batched GEMMs over precomputed Cholesky inverses. Memory-adaptive grid
    chunking keeps peak usage bounded; smart grid ordering (most common (a, b)
    pairs first) combined with optional chunk-level early stopping cuts
    evaluations on data where most voxels share a few common ARMA parameters.

    The same strategy is used on CPU and GPU: BLAS-3 throughput dominates the
    cost of evaluating any single grid point on either device.

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix (shared across all voxels)
    Y_batch : torch.Tensor, shape (n_timepoints, n_voxels_batch)
        Data matrix for batch of voxels
    a_grid : torch.Tensor, optional
        AR parameter grid
    b_grid : torch.Tensor, optional
        MA parameter grid
    device : torch.device, optional
        Computing device
    precomputed : dict, optional
        Pre-computed REML grid (from precompute_reml_grid)
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    enable_early_stopping : bool, default=False
        **GPU only**: Enable batch convergence early stopping.
        If True, stops evaluating grid once >80% of voxels converged.
        This uses smart grid ordering (most common (a,b) pairs first) based on
        real fMRI data analysis. Can provide 2-3x speedup on GPU.
        **WARNING**: May miss true optima for some voxels if they converge to
        suboptimal values. Use False for publication-quality results.
        Use True for exploratory analysis or when speed is critical.

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels_batch, 2)
        Optimal (a, b) for each voxel
    best_likelihoods : torch.Tensor, shape (n_voxels_batch,)
        Minimum likelihood for each voxel

    Notes
    -----
    Memory-Adaptive Chunking:
    - Automatically chunks grid if (n_grid × n_timepoints × n_voxels) > 1.5GB
    - Processes grid in chunks, tracking best params across all chunks
    - Prevents OOM errors with large grids (100+ params)

    Performance:
    - Small grids (63 params): 1 kernel launch, ~0.5s per 10k voxels
    - Large grids (100 params): 3-4 kernel launches, ~1.5s per 10k voxels
    - Still 10-50x faster than 3dREMLfit sequential search!
    """
    if device is None:
        device = get_device()

    X = X.to(device)
    Y_batch = Y_batch.to(device)

    n_timepoints, n_voxels_batch = Y_batch.shape
    n_regressors = X.shape[1]

    # Default grids
    if a_grid is None and b_grid is None:
        a_grid, b_grid = get_default_arma_grids(device)
    else:
        if a_grid is None:
            a_grid, _ = get_default_arma_grids(device)
        else:
            a_grid = to_tensor(a_grid, device=device)

        if b_grid is None:
            _, b_grid = get_default_arma_grids(device)
        else:
            b_grid = to_tensor(b_grid, device=device)

    # Ensure (a=0, b=0) is always in the grid (white noise baseline)
    a_grid, b_grid = ensure_zero_in_grid(a_grid, b_grid)

    # Pre-compute grid if not provided
    if precomputed is None:
        precomputed = precompute_reml_grid(
            X,
            n_timepoints,
            a_grid,
            b_grid,
            device,
            dtype=dtype,
            run_starts=run_starts,
            tau=tau,
        )

    n_grid = len(precomputed)
    if n_grid == 0:
        # No valid grid points - return defaults
        best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=dtype)
        best_params[:, 0] = 0.5
        best_params[:, 1] = 0.0
        best_likelihoods = torch.full((n_voxels_batch,), float("inf"), device=device, dtype=dtype)
        return best_params, best_likelihoods

    # EXHAUSTIVE GRID SEARCH - batch-parallel across all voxels.
    # Same strategy on CPU and GPU: one BLAS-3 GEMM per grid chunk over the
    # whole voxel batch, vs AFNI's sequential per-voxel search. With smart
    # grid ordering and chunk-level early stopping this is ~10-50x faster
    # than 3dREMLfit on either device.

    param_list = list(precomputed.keys())

    # MEMORY-ADAPTIVE CHUNKING:
    # For large grids or many voxels, process grid in chunks
    bytes_per_element = 8 if dtype == torch.float64 else 4
    target_chunk_gb = 2.5  # Target memory per grid chunk

    # Memory per grid point includes:
    # - L_inv: (n_timepoints, n_timepoints) - DOMINATES for large n_timepoints!
    # - X_w: (n_timepoints, n_regressors)
    # - Y_w, XwTYw, beta, pred, resid: all depend on n_voxels_batch
    #
    # For 3609 timepoints:
    # - L_inv alone: 3609^2 * 8 = 104 MB per grid point
    # - With 117 grid points: 12.2 GB just for L_inv_stack!
    #
    # Strategy: Include voxel-dependent terms to get accurate chunk size
    mem_per_grid_point = bytes_per_element * (
        n_timepoints * n_timepoints  # L_inv (dominant!)
        + n_timepoints * n_regressors  # X_w
        + n_timepoints * n_voxels_batch * 5  # Y_w, XwTYw, beta, pred, resid (scaled)
    )
    max_grid_chunk = int((target_chunk_gb * 1e9) / mem_per_grid_point)
    max_grid_chunk = max(5, max_grid_chunk)  # At least 5

    # Clamp to full grid if small enough (avoids unnecessary chunking overhead)
    n_grid = len(param_list)
    if max_grid_chunk >= n_grid:
        max_grid_chunk = n_grid  # Use full grid in one chunk

    # FAST PATH: Detect background/zero voxels
    # These voxels have no signal, so grid search is wasted
    # Just use (0,0) for them
    data_variance = Y_batch.var(dim=0)  # (n_voxels_batch,)
    is_background = data_variance < 1e-6  # Near-zero variance
    n_background = is_background.sum().item()

    # Track best params across all chunks
    best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=dtype)
    best_likelihoods = torch.full((n_voxels_batch,), float("inf"), device=device, dtype=dtype)

    # Set background voxels to (0,0) immediately and mark as "done"
    if n_background > 0:
        best_params[is_background, 0] = 0.0
        best_params[is_background, 1] = 0.0
        best_likelihoods[is_background] = -1.0  # Sentinel value (skip in search)

    # Only search non-background voxels
    _n_searchable = n_voxels_batch - n_background

    # SMART GRID ORDERING: Evaluate most likely optima first
    # Based on analysis of real fMRI data (synthetic_data/true_afni_remlVar.nii.gz):
    # - Top 4 pairs cover 75% of all voxels!
    # - (0.0, 0.3): 30.5%, (0.0, 0.2): 24.9%, (0.0, 0.4): 10.3%, (0.0, 0.1): 5.4%
    # - Next most common: (0.8, -0.6), (0.1, 0.3), (0.8, -0.5)
    # This ordering enables early stopping after just 10-15 evaluations for most voxels!
    priority_params = [
        (0.0, 0.3),  # 30.5% of voxels
        (0.0, 0.2),  # 24.9% of voxels
        (0.0, 0.4),  # 10.3% of voxels
        (0.0, 0.1),  # 5.4% of voxels
        (0.0, 0.5),  # 3.1% of voxels
        (0.8, -0.6),  # 2.1% of voxels
        (0.1, 0.3),  # 1.6% of voxels
        (0.8, -0.5),  # 1.6% of voxels
        (0.8, -0.4),  # 1.4% of voxels
        (0.0, 0.0),  # White noise baseline
    ]
    priority_indices = []
    remaining_indices = []

    for i, params in enumerate(param_list):
        if params in priority_params:
            priority_indices.append(i)
        else:
            remaining_indices.append(i)

    # Reorder: priority first, then rest
    ordered_indices = priority_indices + remaining_indices

    # Chunk the grid if needed for memory
    n_grid = len(param_list)
    if n_grid <= max_grid_chunk:
        grid_chunks = [ordered_indices]
    else:
        grid_chunks = [
            ordered_indices[i : i + max_grid_chunk] for i in range(0, n_grid, max_grid_chunk)
        ]

    n_chunks_evaluated = 0
    # Multi-chunk runs on CPU can take many minutes per chunk on large
    # voxel batches; surface progress so the user can see something is
    # happening. Single-chunk runs (typical small-voxel-batch case) skip
    # the bar to avoid clutter.
    from tqdm.auto import tqdm as _tqdm

    chunk_iter = (
        _tqdm(
            list(enumerate(grid_chunks)),
            desc="REML grid search (chunks)",
            unit="chunk",
        )
        if len(grid_chunks) > 1
        else enumerate(grid_chunks)
    )
    for chunk_idx, chunk_indices in chunk_iter:
        # PHASE 1: Stack precomputed matrices for THIS CHUNK only
        chunk_keys = [param_list[i] for i in chunk_indices]

        L_inv_stack = torch.stack([precomputed[k]["L_inv"] for k in chunk_keys]).to(
            device
        )  # (n_chunk, n_time, n_time) - lower Cholesky inverses
        Q_stack = torch.stack([precomputed[k]["Q"] for k in chunk_keys]).to(
            device
        )  # (n_chunk, n_time, n_regressors) - orthonormal cols for RSS
        logdet_Rcorr_stack = torch.stack([precomputed[k]["logdet_Rcorr"] for k in chunk_keys]).to(
            device
        )  # (n_chunk,)
        logdet_XwTXw_stack = torch.stack([precomputed[k]["logdet_XwTXw"] for k in chunk_keys]).to(
            device
        )  # (n_chunk,)

        # PHASE 2: Prewhiten data via batched GEMM: Y_w = L_inv @ Y  (no triangular solve!)
        # Y_batch: (n_time, n_voxels)  →  expand to (n_chunk, n_time, n_voxels)
        Y_batch_expanded = Y_batch.unsqueeze(0).expand(
            len(chunk_indices), -1, -1
        )  # VIEW - no VRAM copy
        Y_w_all = torch.bmm(L_inv_stack, Y_batch_expanded)  # (n_chunk, n_time, n_voxels)
        del Y_batch_expanded, L_inv_stack

        # PHASE 3: Pythagorean RSS = ||Y_w||² - ||Q'Y_w||²  (no betas needed!)
        # Q_stack: (n_chunk, n_time, n_reg), Y_w_all: (n_chunk, n_time, n_vox)
        Qt_Yw_all = torch.bmm(Q_stack.transpose(1, 2), Y_w_all)  # (n_chunk, n_reg, n_vox)
        rss_all = Y_w_all.pow(2).sum(dim=1) - Qt_Yw_all.pow(2).sum(dim=1)  # (n_chunk, n_vox)
        del Y_w_all, Qt_Yw_all, Q_stack

        # PHASE 4: Compute likelihoods for this chunk
        term1 = logdet_Rcorr_stack.unsqueeze(1)  # (n_chunk, 1)
        term2 = logdet_XwTXw_stack.unsqueeze(1)  # (n_chunk, 1)
        term3 = (n_timepoints - n_regressors) * torch.log(rss_all + 1e-10)
        del rss_all

        chunk_likelihoods = term1 + term2 + term3  # (n_chunk, n_voxels)
        del term1, term2, term3
        del logdet_Rcorr_stack, logdet_XwTXw_stack

        # PHASE 6: Update best parameters if this chunk has better likelihoods
        chunk_best_idx = torch.argmin(chunk_likelihoods, dim=0)  # (n_voxels,)
        chunk_best_likelihoods = chunk_likelihoods[
            chunk_best_idx, torch.arange(n_voxels_batch, device=device)
        ]
        del chunk_likelihoods

        # Track improvement before updating
        improvement = best_likelihoods - chunk_best_likelihoods

        # Update global best where this chunk is better
        improve_mask = chunk_best_likelihoods < best_likelihoods
        best_likelihoods[improve_mask] = chunk_best_likelihoods[improve_mask]

        # Vectorized param update: build params tensor for this chunk, index by best
        if improve_mask.any():
            chunk_params = torch.tensor(
                [param_list[chunk_indices[i]] for i in range(len(chunk_indices))],
                dtype=dtype,
                device=device,
            )  # (n_chunk, 2)
            best_params[improve_mask] = chunk_params[chunk_best_idx[improve_mask]]

        n_chunks_evaluated += 1

        # EARLY STOPPING: Check if batch has converged (OPTIONAL - controlled by flag)
        if enable_early_stopping:
            # Skip background voxels (marked with -1.0 likelihood)
            searchable_mask = best_likelihoods > 0  # Not background
            if searchable_mask.any():
                # For searchable voxels, check if improvement is negligible
                # Use relative improvement to handle different likelihood scales
                rel_improvement = improvement[searchable_mask] / (
                    torch.abs(best_likelihoods[searchable_mask]) + 1e-10
                )
                converged = (rel_improvement < 0.001) | (improvement[searchable_mask] == 0)
                convergence_rate = converged.float().mean().item()

                # If >80% of searchable voxels converged, stop early
                # With smart ordering, most voxels find optimal (a,b) in first 10-15 evals
                if convergence_rate > 0.80 and chunk_idx > 0:  # At least eval 2 chunks (20 params)
                    n_remaining = len(grid_chunks) - n_chunks_evaluated
                    # Note: Only stop if we have more chunks to skip
                    if n_remaining > 0:
                        break

    return best_params, best_likelihoods


def search_voxels_precomputed_grid(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    precomputed_grid: dict,
    device: torch.device,
    verbose: bool = False,
    enable_timing: bool = False,
    return_profile: bool = False,
) -> tuple:
    """
    Search for optimal ARMA parameters using a precomputed grid.

    This function evaluates each voxel's likelihood against all precomputed
    (a, b) parameter combinations and selects the best one.

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    Y_batch : torch.Tensor, shape (n_voxels_batch, n_timepoints)
        Voxel timeseries data
    precomputed_grid : dict
        Dictionary mapping (a, b) tuples to precomputed matrices:
        - 'L_inv': lower Cholesky inverse for GEMM-based prewhitening
        - 'Q': orthonormal columns of prewhitened design for Pythagorean RSS
        - 'logdet_Rcorr': log determinant of correlation matrix
        - 'logdet_XwTXw': log determinant of X'R^-1 X
    device : torch.device
        Device for computation
    verbose : bool
        Print progress information
    enable_timing : bool
        Enable detailed timing profiling
    return_profile : bool, default=False
        If True, also return the full likelihood surface over the (a,b) grid:
        - surface: (n_voxels, n_valid_pairs) — L(a_k, b_k) per voxel per grid point
        - param_list: list of (a, b) tuples in column order
        Sub-brik k of the surface corresponds to param_list[k].
        Use argmin across sub-briks to recover the selected (a,b) per voxel.

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels_batch, 2)
        Optimal (a, b) for each voxel
    best_likelihoods : torch.Tensor, shape (n_voxels_batch,)
        Minimum REML likelihood for each voxel
    surface, param_list : only returned when return_profile=True
    """
    from fastfuncstuff.timing_utils import profile_section

    n_voxels_batch = Y_batch.shape[0]
    n_timepoints, n_regressors = X.shape
    _n_grid = len(precomputed_grid)

    # Initialize results — match dtype of input data (important when use_double=True)
    _dtype = Y_batch.dtype
    # float64 RSS accumulation guards the Pythagorean cancellation; MPS has no
    # float64 so it keeps the input dtype there.
    _acc_dtype = _dtype if device.type == "mps" else torch.float64
    best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=_dtype)
    best_likelihoods = torch.full((n_voxels_batch,), float("inf"), device=device, dtype=_dtype)

    # Extract grid keys and precomputed values
    param_list = list(precomputed_grid.keys())

    # Move Y_batch to device if needed
    if Y_batch.device != device:
        Y_batch = Y_batch.to(device)

    # Allocate full surface array if requested: (n_voxels, n_valid_pairs)
    if return_profile:
        surface = torch.empty(n_voxels_batch, len(param_list), device=device, dtype=_dtype)

    # Evaluate each grid point using L_inv GEMM + Pythagorean RSS (no betas needed)
    # On CPU at T~1500 / V~50k each grid point is a multi-second BLAS-3 GEMM, so
    # surface progress when verbose; quiet otherwise.
    if verbose:
        from tqdm.auto import tqdm as _tqdm

        grid_iter = _tqdm(
            list(enumerate(param_list)),
            desc="REML grid search",
            unit="pair",
        )
    else:
        grid_iter = enumerate(param_list)
    for _grid_idx, (a, b) in grid_iter:
        with profile_section("1_get_grid_data", enabled=enable_timing):
            grid_data = precomputed_grid[(a, b)]
            L_inv = grid_data["L_inv"].to(device)  # (n_time, n_time)
            Q = grid_data["Q"].to(device)  # (n_time, n_reg) orthonormal
            logdet_Rcorr = grid_data["logdet_Rcorr"].item()
            logdet_XwTXw = grid_data["logdet_XwTXw"].item()

        # Prewhiten data via GEMM: Y_w = L_inv @ Y  (no triangular solve!)
        # Y_batch: (n_voxels, n_time) → transpose for matmul → transpose back
        with profile_section("2_prewhiten_data", enabled=enable_timing):
            Y_w_batch = (L_inv @ Y_batch.T).T  # (n_voxels, n_timepoints)

        # Pythagorean RSS = ||Y_w||² - ||Q'Y_w||²  (no betas, no residuals!)
        # Accumulate in float64: for high-R² task voxels the two terms nearly
        # cancel, and float32 cancellation corrupts the likelihood for exactly
        # the voxels that drive the omnibus F.
        with profile_section("3_pythagorean_rss", enabled=enable_timing):
            Qt_Yw = Q.T @ Y_w_batch.T  # (n_reg, n_voxels)
            rss_batch = Y_w_batch.pow(2).sum(dim=1, dtype=_acc_dtype) - Qt_Yw.pow(2).sum(
                dim=0, dtype=_acc_dtype
            )  # (n_voxels,)

        # Compute REML likelihood for this (a, b). rss is float64 (cancellation
        # already handled); cast back to storage dtype — grid points differ by
        # far more than float32 epsilon.
        with profile_section("4_compute_likelihood", enabled=enable_timing):
            term3 = (n_timepoints - n_regressors) * torch.log(rss_batch + 1e-10)
            likelihoods = (logdet_Rcorr + logdet_XwTXw + term3).to(_dtype)  # (n_voxels,)

        # Update best parameters where this (a, b) is better
        with profile_section("5_update_best", enabled=enable_timing):
            improve_mask = likelihoods < best_likelihoods
            best_likelihoods[improve_mask] = likelihoods[improve_mask]
            best_params[improve_mask, 0] = a
            best_params[improve_mask, 1] = b

        # Store likelihood for this grid point in the surface
        if return_profile:
            surface[:, _grid_idx] = likelihoods

    if return_profile:
        return best_params, best_likelihoods, surface, param_list

    return best_params, best_likelihoods


def search_voxels_precomputed_grid_hierarchical(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    precomputed_grid: dict,
    device: torch.device,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched per-voxel hierarchical REML grid search (AFNI parity).

    Implements the same power-of-2 descent as AFNI's REML_find_best_case
    (remla.c:1398): each voxel starts at (0, 0), then at each level
    ``dab = 2^lev`` the grid points within ``best ± dab`` are evaluated;
    after each level the window narrows to ``±dab`` around the voxel's
    new best. The result is the per-voxel local optimum on the same
    surface AFNI searches.

    The vectorisation: at each level we evaluate the **union** of all
    voxels' windows in batched GEMMs over the entire voxel batch, but
    only update voxels whose own window includes the grid point. This
    keeps BLAS-3 throughput while preserving per-voxel windowing.

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix (used only for shape — actual prewhitening uses
        precomputed L_inv).
    Y_batch : torch.Tensor, shape (n_voxels_batch, n_timepoints)
        Voxel timeseries data, voxels-major (matches
        search_voxels_precomputed_grid's convention).
    precomputed_grid : dict
        Same dict shape as precompute_reml_grid produces.
    device : torch.device
        Computing device.
    verbose : bool, default=False
        Show per-level progress bar.

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels_batch, 2)
        Optimal (a, b) for each voxel.
    best_likelihoods : torch.Tensor, shape (n_voxels_batch,)
        Minimum REML neg-log-likelihood per voxel.
    """
    n_voxels_batch = Y_batch.shape[0]
    n_timepoints, n_regressors = X.shape
    _dtype = Y_batch.dtype
    _acc_dtype = _dtype if device.type == "mps" else torch.float64
    T_minus_K = n_timepoints - n_regressors

    if Y_batch.device != device:
        Y_batch = Y_batch.to(device)

    # Build sorted (a, b) axes from the precomputed dict.
    param_list = list(precomputed_grid.keys())
    a_vals_sorted = sorted({a for (a, _) in param_list})
    b_vals_sorted = sorted({b for (_, b) in param_list})
    n_a = len(a_vals_sorted)
    n_b = len(b_vals_sorted)

    # Build a 2D table (ai, bi) -> precomputed dict key (or None). Cached so
    # we don't linear-scan param_list per evaluation.
    key_table: list[list[tuple[float, float] | None]] = [[None] * n_b for _ in range(n_a)]
    for k in param_list:
        a_val, b_val = k
        for ai, av in enumerate(a_vals_sorted):
            if abs(av - a_val) < 1e-6:
                for bi, bv in enumerate(b_vals_sorted):
                    if abs(bv - b_val) < 1e-6:
                        key_table[ai][bi] = k
                        break
                break

    # Index of (0, 0) — AFNI's OLS init (always in grid via ensure_zero_in_grid).
    zero_a_idx = next(i for i, v in enumerate(a_vals_sorted) if abs(v) < 1e-6)
    zero_b_idx = next(i for i, v in enumerate(b_vals_sorted) if abs(v) < 1e-6)

    # Per-voxel state.
    best_a_idx = torch.full((n_voxels_batch,), zero_a_idx, dtype=torch.long, device=device)
    best_b_idx = torch.full((n_voxels_batch,), zero_b_idx, dtype=torch.long, device=device)
    best_likelihoods = torch.full((n_voxels_batch,), float("inf"), device=device, dtype=_dtype)

    # Power-of-2 descent setup. AFNI computes ltop from *interval* counts:
    # ltop = min(log2(na), log2(nb)) - 2 where na,nb are intervals = points-1
    # (remla.c:1444). FFS defaults to one coarser top level (an extra dab pass) —
    # strictly more thorough, never misses an AFNI minimum, but not byte-for-byte.
    # -afni_mode restores AFNI's exact ltop for parity.
    if _REML_MODE.afni_faithful:
        pna = (n_a - 1).bit_length() - 1  # floor(log2(n_a - 1))
        pnb = (n_b - 1).bit_length() - 1
        ltop = max(0, min(pna, pnb) - 2)
    else:
        ltop = max(0, min(n_a, n_b).bit_length() - 2)
    levels = list(range(ltop, -1, -1))

    if verbose:
        from tqdm.auto import tqdm as _tqdm

        _pair_pbar = _tqdm(
            total=len(param_list),
            desc="REML grid search (hierarchical)",
            unit="pair",
        )
    else:
        _pair_pbar = None

    def _eval_subset(ai: int, bi: int, active_idx: torch.Tensor) -> None:
        """Evaluate REML likelihood at grid point (ai, bi) for the given
        voxel subset and update their (best_a_idx, best_b_idx, best_lik)
        where this evaluation improves on the current best.

        This is the per-voxel work-elimination: BLAS-3 GEMM runs over only
        the voxels that need this grid point (those with it in their own
        ± step window), not the full V batch.
        """
        key = key_table[ai][bi]
        if key is None:
            return
        gd = precomputed_grid[key]
        L_inv = gd["L_inv"].to(device)
        Q = gd["Q"].to(device)
        logdet_Rcorr = gd["logdet_Rcorr"].item()
        logdet_XwTXw = gd["logdet_XwTXw"].item()

        Y_sub = Y_batch.index_select(0, active_idx)  # (n_active, T)
        Y_w = Y_sub @ L_inv.T  # (n_active, T)
        Qt_Yw = Y_w @ Q  # (n_active, K)
        # float64 accumulation — see search_voxels_precomputed_grid: the
        # Pythagorean cancellation in float32 corrupts high-R² voxels.
        rss = Y_w.pow(2).sum(dim=1, dtype=_acc_dtype) - Qt_Yw.pow(2).sum(
            dim=1, dtype=_acc_dtype
        )  # (n_active,)
        lik = (logdet_Rcorr + logdet_XwTXw + T_minus_K * torch.log(rss + 1e-10)).to(_dtype)

        cur_best = best_likelihoods.index_select(0, active_idx)
        improve = lik < cur_best
        if bool(improve.any()):
            improved_globals = active_idx.masked_select(improve)
            best_likelihoods[improved_globals] = lik.masked_select(improve)
            best_a_idx[improved_globals] = ai
            best_b_idx[improved_globals] = bi
        if _pair_pbar is not None:
            _pair_pbar.update(1)

    # Step 1: OLS init at (0, 0) for every voxel.
    _eval_subset(
        zero_a_idx,
        zero_b_idx,
        torch.arange(n_voxels_batch, dtype=torch.long, device=device),
    )

    # Step 2: power-of-2 descent. At each level a voxel evaluates only the
    # grid points within ±step of *its own* current best. We iterate the
    # union of all per-voxel windows, but the BLAS-3 GEMM at each point
    # runs only over the voxels whose own window contains it. That is
    # AFNI's per-voxel work elimination, batched over voxel cohorts.
    for lev in levels:
        step = 1 << lev
        if _pair_pbar is not None:
            _pair_pbar.set_postfix_str(f"level {lev}/{ltop} (step={step})")

        # Window bounds per voxel for this level.
        if lev == ltop:
            # Coarsest pass: window is the entire grid, every voxel
            # participates in every grid point at this step.
            a_lo = torch.zeros(n_voxels_batch, dtype=torch.long, device=device)
            a_hi = torch.full((n_voxels_batch,), n_a - 1, dtype=torch.long, device=device)
            b_lo = torch.zeros(n_voxels_batch, dtype=torch.long, device=device)
            b_hi = torch.full((n_voxels_batch,), n_b - 1, dtype=torch.long, device=device)
        else:
            a_lo = (best_a_idx - step).clamp_(0, n_a - 1)
            a_hi = (best_a_idx + step).clamp_(0, n_a - 1)
            b_lo = (best_b_idx - step).clamp_(0, n_b - 1)
            b_hi = (best_b_idx + step).clamp_(0, n_b - 1)

        for bi in range(0, n_b, step):
            in_b = (bi >= b_lo) & (bi <= b_hi)
            if not bool(in_b.any()):
                continue
            for ai in range(0, n_a, step):
                if key_table[ai][bi] is None:
                    continue
                in_a = (ai >= a_lo) & (ai <= a_hi)
                in_window = in_a & in_b
                if not bool(in_window.any()):
                    continue
                # Only the voxels whose own window contains (ai, bi)
                # contribute to this GEMM. At level 0 each voxel's window
                # is ±1 around its best, so a typical (ai, bi) is active
                # for ~1/9 of the voxel batch — a real subset GEMM.
                active_idx = in_window.nonzero(as_tuple=True)[0]
                _eval_subset(ai, bi, active_idx)

    if _pair_pbar is not None:
        _pair_pbar.close()

    a_vals_t = torch.tensor(a_vals_sorted, dtype=_dtype, device=device)
    b_vals_t = torch.tensor(b_vals_sorted, dtype=_dtype, device=device)
    best_params = torch.stack([a_vals_t[best_a_idx], b_vals_t[best_b_idx]], dim=1)
    return best_params, best_likelihoods


def select_arma_params_hierarchical_from_surface(
    surface: torch.Tensor,
    param_list: list[tuple[float, float]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AFNI's hierarchical (a,b) descent expressed as indexing on a precomputed
    likelihood surface — no per-grid-point GEMMs.

    ``search_voxels_precomputed_grid_hierarchical`` reproduces AFNI's
    ``REML_find_best_case`` descent but pays for it with a sync-heavy storm of
    small subset GEMMs (``.any()``/``.nonzero()`` per grid point), which is fine
    on CPU but throttles CUDA. On GPU we already pay for the *full* surface
    cheaply via the exhaustive batched GEMMs; this runs the identical power-of-2
    descent as pure vectorised gather/argmin over that surface, so GPU keeps
    exhaustive-grid speed while landing on AFNI's local optimum (not the global
    grid argmin, which differs from AFNI along the degenerate a≈b ridge).

    Parameters
    ----------
    surface : torch.Tensor, shape (n_voxels, n_pairs)
        REML neg-log-likelihood per voxel per grid point; column k ↔ param_list[k].
    param_list : list of (a, b)
        Grid points in surface-column order (as returned by
        ``search_voxels_precomputed_grid(return_profile=True)``).
    device : torch.device

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels, 2)
    best_likelihoods : torch.Tensor, shape (n_voxels,)
    """
    _dtype = surface.dtype
    n_voxels = surface.shape[0]

    a_vals_sorted = sorted({a for (a, _) in param_list})
    b_vals_sorted = sorted({b for (_, b) in param_list})
    n_a, n_b = len(a_vals_sorted), len(b_vals_sorted)
    a_to_i = {v: i for i, v in enumerate(a_vals_sorted)}
    b_to_i = {v: i for i, v in enumerate(b_vals_sorted)}

    # Scatter the (V, n_pairs) surface into a (V, n_a, n_b) lattice; grid points
    # filtered out upstream (λ ≤ 0) stay +inf so argmin never selects them.
    col_ai = torch.tensor([a_to_i[a] for (a, _) in param_list], device=device, dtype=torch.long)
    col_bi = torch.tensor([b_to_i[b] for (_, b) in param_list], device=device, dtype=torch.long)
    grid = torch.full((n_voxels, n_a, n_b), float("inf"), device=device, dtype=_dtype)
    grid[:, col_ai, col_bi] = surface.to(device)

    # ltop: same convention as search_voxels_precomputed_grid_hierarchical.
    if _REML_MODE.afni_faithful:
        pna = (n_a - 1).bit_length() - 1
        pnb = (n_b - 1).bit_length() - 1
        ltop = max(0, min(pna, pnb) - 2)
    else:
        ltop = max(0, min(n_a, n_b).bit_length() - 2)

    zero_a_idx = next(i for i, v in enumerate(a_vals_sorted) if abs(v) < 1e-6)
    zero_b_idx = next(i for i, v in enumerate(b_vals_sorted) if abs(v) < 1e-6)

    vrange = torch.arange(n_voxels, device=device)
    best_a = torch.full((n_voxels,), zero_a_idx, dtype=torch.long, device=device)
    best_b = torch.full((n_voxels,), zero_b_idx, dtype=torch.long, device=device)
    best_lik = grid[vrange, best_a, best_b].clone()  # OLS (0,0) init

    # Tie-break must match search_voxels_precomputed_grid_hierarchical so the CPU
    # and GPU paths land on the *same* point along the flat a≈b ridge: that
    # function iterates b-outer/a-inner and keeps the first strict improvement,
    # so we flatten candidates b-outer/a-inner and let argmin take the first min.
    for lev in range(ltop, -1, -1):
        step = 1 << lev
        if lev == ltop:
            # Coarsest pass: the fixed coarse lattice, identical for every voxel.
            a_cand = torch.arange(0, n_a, step, device=device)
            b_cand = torch.arange(0, n_b, step, device=device)
            na_c = a_cand.numel()
            sub = grid[:, a_cand][:, :, b_cand].transpose(1, 2)  # (V, Nb, Na)
            mn, am = sub.reshape(n_voxels, -1).min(dim=1)
            ai_sel = a_cand[am % na_c]
            bi_sel = b_cand[am // na_c]
        else:
            # ±step window (3×3) around each voxel's current best.
            offs = torch.tensor([-step, 0, step], device=device, dtype=torch.long)
            a_cand = (best_a[:, None] + offs).clamp_(0, n_a - 1)  # (V, 3)
            b_cand = (best_b[:, None] + offs).clamp_(0, n_b - 1)  # (V, 3)
            vb = b_cand[:, :, None].expand(n_voxels, 3, 3)  # b outer
            va = a_cand[:, None, :].expand(n_voxels, 3, 3)  # a inner
            vi = vrange[:, None, None].expand(n_voxels, 3, 3)
            sub = grid[vi, va, vb]  # (V, 3_b, 3_a)
            mn, am = sub.reshape(n_voxels, -1).min(dim=1)
            ai_sel = a_cand[vrange, am % 3]
            bi_sel = b_cand[vrange, am // 3]
        improve = mn < best_lik
        best_lik = torch.where(improve, mn, best_lik)
        best_a = torch.where(improve, ai_sel, best_a)
        best_b = torch.where(improve, bi_sel, best_b)

    a_vals_t = torch.tensor(a_vals_sorted, dtype=_dtype, device=device)
    b_vals_t = torch.tensor(b_vals_sorted, dtype=_dtype, device=device)
    best_params = torch.stack([a_vals_t[best_a], b_vals_t[best_b]], dim=1)
    return best_params, best_lik


@torch.inference_mode()
def prewhiten_with_arma11(
    X: torch.Tensor,
    Y: torch.Tensor,
    a: float,
    b: float,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prewhiten design and data using ARMA(1,1) covariance

    Returns X* and Y* where * indicates prewhitening via Cholesky decomposition.
    This is the core GLS transformation that accounts for temporal autocorrelation.

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    Y : torch.Tensor, shape (n_timepoints,) or (n_timepoints, n_voxels)
        Data vector(s)
    a : float
        AR parameter
    b : float
        MA parameter

    Returns
    -------
    X_white : torch.Tensor
        Prewhitened design matrix
    Y_white : torch.Tensor
        Prewhitened data
    L : torch.Tensor
        Lower Cholesky factor of ARMA covariance.
        Callers needing L^{-1} @ v should use
        ``torch.linalg.solve_triangular(L, v, upper=False)``.

    Notes
    -----
    Prewhitening transformation:
    - R = L L' (Cholesky)
    - X* = L^(-1) X  (computed via triangular solve, not explicit inverse)
    - Y* = L^(-1) Y

    After prewhitening, OLS on (X*, Y*) = GLS on (X, Y).

    GPU Speed: ~10ms for n=300 timepoints (Cholesky is highly optimized in PyTorch)

    Memory optimization: For large n_timepoints (e.g., 17140), the covariance matrix
    is huge (17140^2 = 2+ GB). Building on CPU avoids GPU fragmentation and OOM errors.
    Cholesky on CPU is fast, and only L is moved to GPU.
    """
    device = X.device
    n_timepoints = X.shape[0]

    # Build ARMA(1,1) covariance on CPU to save GPU memory and avoid fragmentation
    # Use X.dtype so precision matches the inputs (important when use_double=True)
    R = build_arma11_covariance(
        a,
        b,
        n_timepoints,
        torch.device("cpu"),
        dtype=X.dtype,
        run_starts=run_starts,
        tau=tau,
    )

    if R is None:
        raise ValueError(f"Invalid ARMA(1,1) parameters: a={a}, b={b}")

    # Try GPU Cholesky (fast), fall back to CPU if OOM (safe)
    try:
        # Compact GPU memory before attempting allocation (minimal cost, big benefit!)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Move R to GPU temporarily for fast Cholesky
        R_gpu = R.to(device)
        L = torch.linalg.cholesky(R_gpu)
        del R_gpu

    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # GPU OOM - fall back to CPU (slower but safe)
        try:
            del L  # noqa: F821 — L may be unbound if the try failed before assignment
        except NameError:
            pass

        if device.type == "cuda":
            torch.cuda.empty_cache()

        L = torch.linalg.cholesky(R).to(device)

    # R is bound on both branches here; free the CPU covariance before the solves.
    del R

    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Prewhiten via triangular solve: X* = L^{-1} X, Y* = L^{-1} Y
    # This is mathematically identical to forming L_inv and multiplying,
    # but avoids the O(n^3) explicit inverse and exploits triangular structure.
    X_white = torch.linalg.solve_triangular(L, X, upper=False)
    Y_white = torch.linalg.solve_triangular(L, Y, upper=False)

    if Y_white.shape[1] == 1:
        Y_white = Y_white.squeeze(1)

    return X_white, Y_white, L


def determine_adaptive_batching_strategy(
    n_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    n_grid_points: int,
    device: torch.device,
    use_double: bool,
    verbose: bool = False,
) -> dict:
    """
    Intelligently determine optimal batching strategy for REML grid search.

    Considers the three-way tradeoff between:
    1. Grid size (how many (a,b) pairs)
    2. Data size (how many voxels)
    3. Available GPU memory

    Returns strategy with optimal grid_chunk_size and voxel_batch_size.

    Parameters
    ----------
    n_voxels : int
        Number of voxels
    n_timepoints : int
        Number of timepoints
    n_regressors : int
        Number of regressors
    n_grid_points : int
        Number of (a,b) pairs in grid
    device : torch.device
        Computing device
    use_double : bool
        Using float64 (True) or float32 (False)
    verbose : bool
        Print decision reasoning

    Returns
    -------
    dict with keys:
        - strategy: "full_grid" | "grid_batching_full_data" | "hybrid"
        - grid_chunk_size: int (how many grid points to load at once)
        - voxel_batch_size: int (how many voxels to process at once)
        - reason: str (explanation)
    """
    bytes_per_elem = 8 if use_double else 4

    # Calculate memory requirements
    grid_mem_per_point = (
        n_timepoints * n_timepoints * bytes_per_elem  # L_inv (dominant!)
        + n_timepoints * n_regressors * bytes_per_elem  # X_w
        + n_regressors * n_regressors * bytes_per_elem  # R_qr
        + 2 * bytes_per_elem  # scalars (logdet)
    )
    grid_mem_total = n_grid_points * grid_mem_per_point
    data_mem = n_voxels * n_timepoints * bytes_per_elem

    # Get available memory
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem, _ = torch.cuda.mem_get_info(device)
        # Use conservative estimate: 60% of total OR 80% of free
        available = min(total_mem * 0.6, free_mem * 0.8)
    else:
        # For CPU, assume we have plenty (32GB)
        available = 32 * 1024**3

    # Estimate per-voxel batch overhead (whitening, fitting, etc.)
    # Factor of 2-3x for intermediate computations
    batch_overhead_factor = 2.5
    mem_per_voxel_batch = lambda n_v: n_v * n_timepoints * bytes_per_elem * batch_overhead_factor

    if verbose:
        print(f"\n{'=' * 70}")
        print("ADAPTIVE BATCHING STRATEGY")
        print(f"{'=' * 70}")
        print(
            f"  Grid: {n_grid_points} points × {grid_mem_per_point / 1024**2:.1f} MB = {grid_mem_total / 1024**3:.2f} GB"
        )
        print(f"  Data: {n_voxels} voxels × {n_timepoints} TPs = {data_mem / 1024**3:.2f} GB")
        print(f"  Available memory: {available / 1024**3:.2f} GB")
        print()

    # Decision tree
    if grid_mem_total + mem_per_voxel_batch(n_voxels * 0.1) < available:
        # Case 1: Full grid + voxel batches (AFNI approach - FASTEST!)
        # Leave 30% for voxel batches
        mem_for_voxels = available - grid_mem_total
        voxel_batch_size = int(
            mem_for_voxels / (n_timepoints * bytes_per_elem * batch_overhead_factor)
        )
        voxel_batch_size = max(500, min(voxel_batch_size, n_voxels))

        strategy = {
            "strategy": "full_grid",
            "grid_chunk_size": n_grid_points,
            "voxel_batch_size": voxel_batch_size,
            "reason": f"Grid fits ({grid_mem_total / 1024**3:.1f}GB) - using AFNI approach (fastest!)",
        }

    elif data_mem < available * 0.45:
        # Case 2: Grid batching + full data
        # Data fits comfortably, so load it once and batch the grid
        # Reserve 40% for grid chunks, 50% for data, 10% for overhead
        mem_for_grid = available * 0.4
        grid_chunk_size = int(mem_for_grid / grid_mem_per_point)

        # Sweet spot: 5-20 grid points gives good amortization
        grid_chunk_size = max(1, min(grid_chunk_size, 20, n_grid_points))

        strategy = {
            "strategy": "grid_batching_full_data",
            "grid_chunk_size": grid_chunk_size,
            "voxel_batch_size": n_voxels,  # All voxels
            "reason": f"Data fits ({data_mem / 1024**3:.1f}GB), batching grid in chunks of {grid_chunk_size}",
        }

    else:
        # Case 3: Hybrid - batch BOTH grid and voxels
        # Split available memory: 40% grid, 40% data, 20% overhead
        mem_for_grid = available * 0.4
        mem_for_data = available * 0.4

        grid_chunk_size = int(mem_for_grid / grid_mem_per_point)
        grid_chunk_size = max(1, min(grid_chunk_size, 10, n_grid_points))

        voxel_batch_size = int(
            mem_for_data / (n_timepoints * bytes_per_elem * batch_overhead_factor)
        )
        voxel_batch_size = max(100, min(voxel_batch_size, n_voxels))

        strategy = {
            "strategy": "hybrid",
            "grid_chunk_size": grid_chunk_size,
            "voxel_batch_size": voxel_batch_size,
            "reason": f"Both large - grid chunks of {grid_chunk_size}, voxel batches of {voxel_batch_size:,}",
        }

    if verbose:
        print(f"  Strategy: {strategy['strategy']}")
        print(f"    Grid chunk size: {strategy['grid_chunk_size']}")
        print(f"    Voxel batch size: {strategy['voxel_batch_size']:,}")
        print(f"    Reason: {strategy['reason']}")
        print(f"{'=' * 70}\n")

    return strategy


def reml_grid_search_batched(
    data: torch.Tensor,
    design: torch.Tensor,
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
    dtype: torch.dtype = torch.float32,
    grid_chunk_size: int = 1,
    voxel_batch_size: int | None = None,
    y_scale: torch.Tensor | None = None,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REML grid search with adaptive batching (memory-efficient).

    Supports three strategies:
    1. Process grid in chunks, all voxels at once (grid_chunk_size > 1, voxel_batch_size = n_voxels)
    2. Process one grid point, all voxels (grid_chunk_size = 1, voxel_batch_size = n_voxels)
    3. Process grid chunks AND voxel batches (both < full size) - for huge datasets

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        fMRI data (on CPU)
    design : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    a_grid : torch.Tensor
        AR parameter grid
    b_grid : torch.Tensor
        MA parameter grid
    device : torch.device
        Device for computation
    verbose : bool
        Print progress
    dtype : torch.dtype
        Data type (float32 or float64)
    grid_chunk_size : int, default=1
        Number of grid points to process at once (1-20 recommended)
    voxel_batch_size : int or None
        Number of voxels to process at once (None = all voxels)

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels, 2)
        Best (a, b) for each voxel
    best_likelihood : torch.Tensor, shape (n_voxels,)
        Best REML likelihood for each voxel
    """
    n_voxels, n_timepoints = data.shape
    n_regressors = design.shape[1]

    # Ensure zero is in grid (AFNI requirement)
    a_grid, b_grid = ensure_zero_in_grid(a_grid, b_grid)

    # Create all (a,b) combinations
    a_mesh, b_mesh = torch.meshgrid(a_grid, b_grid, indexing="ij")
    a_flat = a_mesh.reshape(-1)
    b_flat = b_mesh.reshape(-1)

    # Filter invalid (a,b) pairs using same logic as build_arma11_covariance_batch
    # but WITHOUT building the huge covariance matrices
    valid_mask = (torch.abs(a_flat) < 1.0) & (torch.abs(b_flat) < 1.0)

    # Check stationarity: gamma0 > 0
    denom = 1 - a_flat**2
    valid_mask &= torch.abs(denom) >= 1e-6
    gamma0 = (1 + b_flat**2 + 2 * a_flat * b_flat) / (denom + 1e-10)
    valid_mask &= gamma0 > 0

    # Filter lambda >= 0
    a_valid_temp = a_flat[valid_mask]
    b_valid_temp = b_flat[valid_mask]
    rho1_valid = _compute_arma11_lambda(a_valid_temp, b_valid_temp)
    lambda_mask = rho1_valid >= 0

    # Get final valid pairs
    valid_indices = torch.where(valid_mask)[0]
    final_valid_indices = valid_indices[lambda_mask]
    a_valid = a_flat[final_valid_indices]
    b_valid = b_flat[final_valid_indices]
    n_valid = len(a_valid)

    if verbose:
        print(f"  Valid grid points: {n_valid} / {len(a_flat)}")
        if voxel_batch_size and voxel_batch_size < n_voxels:
            print(f"  Processing {voxel_batch_size:,} voxels at a time (hybrid mode)")

    # Determine if we load all data or batch it
    load_all_data = (voxel_batch_size is None) or (voxel_batch_size >= n_voxels)

    if load_all_data:
        # Move all data to device ONCE (faster for most cases)
        data_dev = data.to(device, dtype=dtype)
        if y_scale is not None:
            data_dev = data_dev / y_scale.to(device=device, dtype=dtype).unsqueeze(1)
        design_dev = design.to(device, dtype=dtype)
    else:
        # Keep data on CPU, will move batches as needed
        design_dev = design.to(device, dtype=dtype)

    # Allocate results.
    # Convention matches compute_reml_likelihood and batch_reml_grid_search:
    # REML neg-log-likelihood, smaller = better.
    best_params = torch.zeros(n_voxels, 2, dtype=dtype, device=torch.device("cpu"))
    best_likelihood = torch.full((n_voxels,), float("inf"), dtype=dtype, device=torch.device("cpu"))

    # Process grid in chunks for efficiency
    n_chunks = (n_valid + grid_chunk_size - 1) // grid_chunk_size

    if verbose:
        from tqdm import tqdm

        print(f"  Processing {n_chunks} grid chunks (chunk size: {grid_chunk_size})")
        chunk_iterator = tqdm(range(n_chunks), desc="Grid search", unit="chunk")
    else:
        chunk_iterator = range(n_chunks)

    # Loop through grid chunks
    for chunk_idx in chunk_iterator:
        # Get grid points for this chunk
        start_idx = chunk_idx * grid_chunk_size
        end_idx = min(start_idx + grid_chunk_size, n_valid)
        chunk_grid_points = range(start_idx, end_idx)

        # Process each grid point in this chunk
        for grid_idx in chunk_grid_points:
            a = a_valid[grid_idx].item()
            b = b_valid[grid_idx].item()

            try:
                # Build covariance matrix for this (a,b) - do on CPU to save GPU memory
                R = build_arma11_covariance(
                    torch.tensor(a, dtype=dtype),
                    torch.tensor(b, dtype=dtype),
                    n_timepoints,
                    device=torch.device("cpu"),
                    dtype=dtype,
                    run_starts=run_starts,
                    tau=tau,
                )

                # Compute Cholesky
                L = torch.linalg.cholesky(R)  # On CPU

                # Move L to GPU for triangular solve
                L_dev = L.to(device)

                # Prewhiten design via triangular solve (no explicit inverse!)
                X_w = torch.linalg.solve_triangular(L_dev, design_dev, upper=False)  # (n_tp, n_reg)
                XwTXw = X_w.T @ X_w
                XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device, dtype=dtype)

                # Log determinants for REML
                logdet_R = 2.0 * torch.sum(torch.log(torch.diag(L)))
                logdet_XwTXw = torch.logdet(XwTXw_reg)

                # Process voxels (either all at once or in batches)
                if load_all_data:
                    # All data already on device - process at once
                    # Prewhiten ALL voxels via triangular solve
                    y_w = torch.linalg.solve_triangular(
                        L_dev, data_dev.T, upper=False
                    ).T  # (n_voxels, n_tp)

                    # Solve for ALL voxels: beta = (X'X)^-1 X'y
                    XwTy_w = X_w.T @ y_w.T  # (n_reg, n_voxels)
                    betas = torch.linalg.solve(XwTXw_reg, XwTy_w).T  # (n_voxels, n_reg)

                    # Compute residuals and SSE
                    resid_w = y_w - (betas @ X_w.T)  # (n_voxels, n_tp)
                    sse = torch.sum(resid_w**2, dim=1)  # (n_voxels,)

                    # REML neg-log-likelihood (matches AFNI 3dREMLfit and the
                    # CPU/chunked-GPU paths in this file):
                    #   logdet_R + logdet(X'R^-1 X) + (n - m) * log(SSE)
                    # Earlier versions used the ML form n*log(SSE/n), which has
                    # a different multiplier on log(SSE) and so can shift the
                    # argmin between adjacent grid points.
                    likelihoods = (
                        logdet_R
                        + logdet_XwTXw
                        + (n_timepoints - n_regressors) * torch.log(sse + 1e-10)
                    )

                    # Update best parameters (vectorized comparison; smaller = better)
                    likelihoods_cpu = likelihoods.cpu()
                    mask = likelihoods_cpu < best_likelihood
                    best_params[mask, 0] = a
                    best_params[mask, 1] = b
                    best_likelihood[mask] = likelihoods_cpu[mask]

                    # Free GPU memory before next iteration
                    del y_w, betas, resid_w, sse, likelihoods

                else:
                    # Process voxels in batches
                    n_voxel_batches = (n_voxels + voxel_batch_size - 1) // voxel_batch_size
                    for voxel_batch_idx in range(n_voxel_batches):
                        voxel_start = voxel_batch_idx * voxel_batch_size
                        voxel_end = min(voxel_start + voxel_batch_size, n_voxels)

                        # Move this batch to device
                        data_batch = data[voxel_start:voxel_end].to(device, dtype=dtype)
                        if y_scale is not None:
                            data_batch = data_batch / y_scale[voxel_start:voxel_end].to(
                                device=device, dtype=dtype
                            ).unsqueeze(1)

                        # Prewhiten this batch via triangular solve
                        y_w = torch.linalg.solve_triangular(
                            L_dev, data_batch.T, upper=False
                        ).T  # (batch_voxels, n_tp)

                        # Solve for this batch: beta = (X'X)^-1 X'y
                        XwTy_w = X_w.T @ y_w.T  # (n_reg, batch_voxels)
                        betas = torch.linalg.solve(XwTXw_reg, XwTy_w).T  # (batch_voxels, n_reg)

                        # Compute residuals and SSE
                        resid_w = y_w - (betas @ X_w.T)  # (batch_voxels, n_tp)
                        sse = torch.sum(resid_w**2, dim=1)  # (batch_voxels,)

                        # REML neg-log-likelihood; see comment in the
                        # load_all_data branch above for the formula rationale.
                        likelihoods = (
                            logdet_R
                            + logdet_XwTXw
                            + (n_timepoints - n_regressors) * torch.log(sse + 1e-10)
                        )

                        # Update best parameters for this batch (smaller = better)
                        likelihoods_cpu = likelihoods.cpu()
                        batch_best_likelihood = best_likelihood[voxel_start:voxel_end]
                        mask = likelihoods_cpu < batch_best_likelihood

                        # Update using proper indexing (avoid chained indexing which creates copies)
                        if mask.any():
                            batch_params = best_params[voxel_start:voxel_end]
                            batch_params[mask, 0] = a
                            batch_params[mask, 1] = b
                            best_params[voxel_start:voxel_end] = batch_params

                            batch_best_likelihood[mask] = likelihoods_cpu[mask]
                            best_likelihood[voxel_start:voxel_end] = batch_best_likelihood

                        # Free GPU memory for this batch
                        del data_batch, y_w, betas, resid_w, sse, likelihoods

                # Free common GPU memory before next iteration
                del L_dev, X_w, XwTXw, XwTXw_reg

            except (torch.linalg.LinAlgError, RuntimeError) as e:
                # Skip this (a,b) if Cholesky fails
                if verbose:
                    print(f"  Skipping (a={a:.2f}, b={b:.2f}): {e}")
                continue

    # Clean up all GPU memory before returning
    if load_all_data:
        del data_dev
    del design_dev
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return best_params, best_likelihood


def fit_glm_arma11_grouped(
    data: torch.Tensor | np.ndarray,
    designs_by_group: dict[int, torch.Tensor],
    group_indices: torch.Tensor | np.ndarray,
    tr: float,
    group_label: str = "group",
    **fit_kwargs,
) -> ARMA11Results:
    """REML with a per-voxel-group design: loop fit_glm_arma11 per group, merge.

    A "group" is any partition of voxels that share a design matrix — per-voxel HRF
    (group = HRF index), slicewise -slibase (group = z-slice), or a composite of the
    two. The design differs between groups but the noise-covariance grid does not (it
    depends only on (a,b) and n_timepoints), so a single ``autocorr_cache`` is built
    once and reused across all groups. This is NOT about convolution — each group is
    just handed whatever design matrix the caller built for it.

    Parameters
    ----------
    data : (n_voxels, n_timepoints) tensor or ndarray
        fMRI data, already masked to the analysis voxel set.
    designs_by_group : dict[int, (n_timepoints, n_regressors) tensor]
        Full design matrix (task+nuisance) per group id. All designs MUST share
        the same n_regressors and column ordering.
    group_indices : (n_voxels,) long tensor
        Group id per voxel, aligned to ``data``'s voxel axis.
    tr : float
        Repetition time.
    group_label : str
        Human-readable noun for the grouping ("HRF", "slice", ...), used in prints.
    **fit_kwargs
        Forwarded to each fit_glm_arma11 call. ``task_indices`` should reference
        the shared task columns. ``run_starts`` / ``dsort`` propagated as-is
        (``dsort`` is sliced to each group's voxel subset).

    Returns
    -------
    ARMA11Results
        Voxel arrays sized to ``n_voxels`` of ``data``, scattered from each
        per-group subset.
    """
    import torch as _torch

    if not isinstance(group_indices, _torch.Tensor):
        group_indices = _torch.as_tensor(group_indices)
    group_indices = group_indices.long().cpu()

    if isinstance(data, np.ndarray):
        data_t = _torch.from_numpy(data)
    else:
        data_t = data
    n_voxels = data_t.shape[0]
    if group_indices.shape[0] != n_voxels:
        raise ValueError(
            f"group_indices length {group_indices.shape[0]} != data n_voxels {n_voxels}"
        )

    unique_hrfs = sorted(int(h) for h in group_indices.unique().tolist())
    missing = [h for h in unique_hrfs if h not in designs_by_group]
    if missing:
        raise KeyError(
            f"{group_label} ids {missing} present in voxels but missing from "
            f"designs_by_group (have {sorted(designs_by_group.keys())})"
        )

    # All designs must share shape — we allocate per-voxel arrays sized to the
    # canonical n_regressors. Verify upfront.
    first_design = designs_by_group[unique_hrfs[0]]
    n_timepoints, n_regressors = first_design.shape
    for h in unique_hrfs[1:]:
        if designs_by_group[h].shape != (n_timepoints, n_regressors):
            raise ValueError(
                f"designs_by_group[{h}] shape {tuple(designs_by_group[h].shape)} "
                f"!= {(n_timepoints, n_regressors)} — per-HRF designs must share dims"
            )

    verbose = fit_kwargs.get("verbose", True)
    if verbose:
        print(
            f"\n🎚️  Grouped REML (per {group_label}): {len(unique_hrfs)} "
            f"{group_label} group(s) across {n_voxels:,} voxels"
        )

    # Drop spatial_metadata from the inner calls — voxel_mask there refers to
    # the global voxel axis and would confuse the subsetted results. We'll set
    # them on the merged results from the caller.
    inner_kwargs = dict(fit_kwargs)
    inner_kwargs.pop("spatial_metadata", None)

    # OLS write callback would fire once per HRF and clobber outputs. Strip it
    # and re-fire later on the merged OLS result if want_ols is set.
    ols_write_callback = inner_kwargs.pop("ols_write_callback", None)
    if ols_write_callback is not None and verbose:
        print(f"  (OLS write callback deferred until after per-{group_label} merge)")

    # Subset voxel-wise (-dsort) regressors per HRF, like the data and the
    # precomputed (a,b). Passed whole, it would mismatch each HRF subset's voxels.
    dsort_full = inner_kwargs.pop("dsort", None)
    if dsort_full is not None:
        if isinstance(dsort_full, np.ndarray):
            dsort_full = _torch.from_numpy(dsort_full)
        if dsort_full.ndim == 2:
            dsort_full = dsort_full.unsqueeze(1)
        if dsort_full.shape[0] != n_voxels:
            raise ValueError(f"dsort has {dsort_full.shape[0]} voxels but data has {n_voxels}")

    # Subset precomputed (a,b) per HRF — must align with masked voxel order.
    precomputed_arma_full = inner_kwargs.pop("precomputed_arma_params", None)
    if precomputed_arma_full is not None:
        if isinstance(precomputed_arma_full, np.ndarray):
            precomputed_arma_full = _torch.from_numpy(precomputed_arma_full)
        precomputed_arma_full = precomputed_arma_full.reshape(-1, 2)
        if precomputed_arma_full.shape[0] != n_voxels:
            raise ValueError(
                f"precomputed_arma_params has {precomputed_arma_full.shape[0]} rows "
                f"but data has {n_voxels} voxels — mask alignment failed."
            )

    # Build the (a,b) Cholesky grid ONCE — it depends only on autocorrelation
    # structure, not the design. Saves ~10–20 s × (n_hrfs − 1).
    fit_device = inner_kwargs.get("device")
    if fit_device is None:
        fit_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        inner_kwargs["device"] = fit_device
    a_grid_shared = inner_kwargs.get("a_grid")
    b_grid_shared = inner_kwargs.get("b_grid")
    if a_grid_shared is None or b_grid_shared is None:
        _a_def, _b_def = get_default_arma_grids(fit_device)
        if a_grid_shared is None:
            a_grid_shared = _a_def
        if b_grid_shared is None:
            b_grid_shared = _b_def
    inner_kwargs["a_grid"] = a_grid_shared
    inner_kwargs["b_grid"] = b_grid_shared

    if precomputed_arma_full is None and inner_kwargs.get("estimate_per_voxel", True):
        if verbose:
            print(f"\n🔁 Building autocorr grid (shared across all {group_label} groups)...")
        use_double = inner_kwargs.get("use_double", False)
        cholesky_on_cpu = inner_kwargs.get("cholesky_on_cpu", True)
        autocorr_cache = precompute_autocorr_grid(
            first_design.shape[0],
            a_grid_shared,
            b_grid_shared,
            fit_device,
            verbose=verbose,
            cholesky_on_cpu=cholesky_on_cpu,
            dtype=_torch.float64 if use_double else _torch.float32,
            run_starts=inner_kwargs.get("run_starts"),
            tau=inner_kwargs.get("tau"),
        )
        inner_kwargs["autocorr_cache"] = autocorr_cache
    else:
        autocorr_cache = None

    per_hrf_results: dict[int, ARMA11Results] = {}
    per_hrf_local_mask: dict[int, _torch.Tensor] = {}  # voxel positions in `data` axis

    for hrf_idx in unique_hrfs:
        sub_mask = group_indices == hrf_idx
        n_sub = int(sub_mask.sum().item())
        if verbose:
            print(
                f"\n  ─── {group_label} {hrf_idx}: {n_sub:,} voxels "
                f"({100.0 * n_sub / n_voxels:.1f}%) ───"
            )
        sub_data = data_t[sub_mask]
        sub_design = designs_by_group[hrf_idx]

        # Pre-load the per-HRF voxel subset onto the GPU so every per-sub-batch
        # `data[indices]` inside fit_glm_arma11 becomes a GPU view instead of a
        # CPU allocation + host→device copy. Subset is small (~n_sub × n_t × 4
        # bytes); if it doesn't fit, fall back to the CPU tensor.
        if fit_device.type == "cuda":
            try:
                sub_data = sub_data.to(device=fit_device, non_blocking=True)
            except _torch.cuda.OutOfMemoryError:
                if verbose:
                    print("  ⚠️  HRF subset too large for GPU; falling back to CPU data")
                _torch.cuda.empty_cache()

        sub_kwargs = dict(inner_kwargs)
        if precomputed_arma_full is not None:
            sub_kwargs["precomputed_arma_params"] = precomputed_arma_full[sub_mask]
        if dsort_full is not None:
            sub_kwargs["dsort"] = dsort_full[sub_mask]

        results_h = fit_glm_arma11(
            sub_data,
            sub_design,
            tr=tr,
            **sub_kwargs,
        )

        # Free the GPU subset before the next HRF allocates its own.
        if sub_data.device.type == "cuda":
            del sub_data
            _torch.cuda.empty_cache()
        per_hrf_results[hrf_idx] = results_h
        per_hrf_local_mask[hrf_idx] = sub_mask

    # Merge: allocate full-size arrays on CPU, scatter per-HRF results in.
    merged = ARMA11Results()
    template = per_hrf_results[unique_hrfs[0]]
    merged.dof = template.dof
    merged.tr = template.tr
    merged.fitted_column_indices = template.fitted_column_indices
    merged.n_regressors_full = template.n_regressors_full

    _scatter_attrs = VOXEL_SCATTER_ATTRS

    for attr in _scatter_attrs:
        # Find first sub-result that has this attr populated to learn shape/dtype
        ref = None
        for h in unique_hrfs:
            v = getattr(per_hrf_results[h], attr, None)
            if v is not None:
                ref = v
                break
        if ref is None:
            continue
        full_shape = (n_voxels,) + tuple(ref.shape[1:])
        full = _torch.zeros(full_shape, dtype=ref.dtype)
        for h in unique_hrfs:
            v_h = getattr(per_hrf_results[h], attr, None)
            if v_h is None:
                continue
            full[per_hrf_local_mask[h]] = v_h.cpu()
        setattr(merged, attr, full)

    # Contrast labels (scalar metadata, same across HRFs)
    merged.contrast_labels = template.contrast_labels

    # -dsort: scatter the no-dsort snapshot (if produced) and carry labels.
    merged.dsort_labels = template.dsort_labels
    template_nods = getattr(template, "nods_results", None)
    if template_nods is not None:
        nods_merged = ARMA11Results()
        nods_merged.dof = template_nods.dof
        nods_merged.tr = template_nods.tr
        nods_merged.fitted_column_indices = template_nods.fitted_column_indices
        nods_merged.n_regressors_full = template_nods.n_regressors_full
        nods_merged.contrast_labels = template_nods.contrast_labels
        for attr in _scatter_attrs:
            ref = None
            for h in unique_hrfs:
                src = getattr(per_hrf_results[h], "nods_results", None)
                v = getattr(src, attr, None) if src is not None else None
                if v is not None:
                    ref = v
                    break
            if ref is None:
                continue
            full = _torch.zeros((n_voxels,) + tuple(ref.shape[1:]), dtype=ref.dtype)
            for h in unique_hrfs:
                src = getattr(per_hrf_results[h], "nods_results", None)
                v_h = getattr(src, attr, None) if src is not None else None
                if v_h is None:
                    continue
                full[per_hrf_local_mask[h]] = v_h.cpu()
            setattr(nods_merged, attr, full)
        merged.nods_results = nods_merged

    # REML likelihood surface (-Rlklhd): each per-HRF call produced a
    # (n_sub, n_valid_pairs) surface. Scatter into a full (n_voxels, n_valid_pairs)
    # surface. All per-HRF calls share the same a/b grid → same column ordering.
    surf_ref = None
    for h in unique_hrfs:
        s = getattr(per_hrf_results[h], "reml_lklhd_surface", None)
        if s is not None:
            surf_ref = s
            merged.reml_surface_params = per_hrf_results[h].reml_surface_params
            break
    if surf_ref is not None:
        full_surf = _torch.zeros((n_voxels, surf_ref.shape[1]), dtype=surf_ref.dtype)
        for h in unique_hrfs:
            s_h = getattr(per_hrf_results[h], "reml_lklhd_surface", None)
            if s_h is None:
                continue
            full_surf[per_hrf_local_mask[h]] = s_h.cpu()
        merged.reml_lklhd_surface = full_surf

    # Merge OLS results if any inner call produced them.
    ols_template = None
    for h in unique_hrfs:
        if getattr(per_hrf_results[h], "ols_results", None) is not None:
            ols_template = per_hrf_results[h].ols_results
            break
    if ols_template is not None:
        from fastfuncstuff.glm.core import GLMResults

        ols_merged = GLMResults()
        ols_merged.dof = ols_template.dof
        ols_merged.tr = ols_template.tr
        ols_merged.xtx_inv = getattr(ols_template, "xtx_inv", None)
        ols_merged.contrast_labels = getattr(ols_template, "contrast_labels", None)

        _ols_scatter_attrs = OLS_VOXEL_SCATTER_ATTRS
        for attr in _ols_scatter_attrs:
            ref = None
            for h in unique_hrfs:
                ols_h = getattr(per_hrf_results[h], "ols_results", None)
                if ols_h is None:
                    continue
                v = getattr(ols_h, attr, None)
                if v is not None:
                    ref = v
                    break
            if ref is None:
                continue
            full_shape = (n_voxels,) + tuple(ref.shape[1:])
            full = _torch.zeros(full_shape, dtype=ref.dtype)
            for h in unique_hrfs:
                ols_h = getattr(per_hrf_results[h], "ols_results", None)
                if ols_h is None:
                    continue
                v_h = getattr(ols_h, attr, None)
                if v_h is None:
                    continue
                full[per_hrf_local_mask[h]] = v_h.cpu()
            setattr(ols_merged, attr, full)
        merged.ols_results = ols_merged

        # Fire the deferred OLS write callback once on the merged result.
        # The caller passes original_shape/affine alongside through merged
        # results (analyze_from_design_matrix sets them after this returns),
        # so just attach what we have here and let the caller finish.
        merged._deferred_ols_write_callback = ols_write_callback

    return merged


def fit_glm_arma11_perhrf(
    data: torch.Tensor | np.ndarray,
    designs_by_hrf: dict[int, torch.Tensor],
    hrf_indices: torch.Tensor | np.ndarray,
    tr: float,
    **fit_kwargs,
) -> ARMA11Results:
    """Per-voxel HRF REML — thin wrapper over :func:`fit_glm_arma11_grouped`.

    The group is the HRF index; each HRF's design differs only in its task columns.
    """
    return fit_glm_arma11_grouped(
        data,
        designs_by_hrf,
        hrf_indices,
        tr,
        group_label="HRF",
        **fit_kwargs,
    )


def _dsort_constant_guard(dsort: torch.Tensor) -> torch.Tensor:
    """Replace ~constant voxel timeseries in each dsort dataset with its mean.

    Matches 3dREMLfit: a -dsort voxel that is constant through time is degenerate
    (it collapses onto the baseline), so AFNI substitutes the dataset's mean
    timeseries for that voxel. ``dsort`` is (n_voxels, n_dsort, n_timepoints).

    Block-safe: a block-diagonal (per-run) column is legitimately zero outside its
    run, which reads as low variance but is NOT a collapse-onto-baseline. Only a
    genuinely CONSTANT-NONZERO column is substituted; all-zero (zero-padded)
    columns are left untouched.

    Always warns when it fires (this is a data condition, not progress chatter).
    """
    # std over time, per (voxel, dsort dataset)
    stds = dsort.std(dim=2)  # (n_voxels, n_dsort)
    means = dsort.abs().mean(dim=2)  # (n_voxels, n_dsort)
    bad = (stds < 1e-8) & (means > 1e-8)  # constant-nonzero only
    n_bad = int(bad.sum().item())
    if n_bad == 0:
        return dsort
    dsort = dsort.clone()
    # Per-dataset mean timeseries computed over the non-degenerate voxels.
    for d in range(dsort.shape[1]):
        bad_d = bad[:, d]
        if not bool(bad_d.any()):
            continue
        good_d = ~bad_d
        if bool(good_d.any()):
            mean_ts = dsort[good_d, d, :].mean(dim=0)  # (n_timepoints,)
        else:
            mean_ts = dsort[:, d, :].mean(dim=0)
        dsort[bad_d, d, :] = mean_ts
    warnings.warn(
        f"-dsort: {n_bad} voxel timeseries were ~constant through time and "
        "were replaced by their dataset mean (matches 3dREMLfit).",
        stacklevel=2,
    )
    return dsort


def _fit_dsort_gls_pass(
    data: torch.Tensor,
    design: torch.Tensor,
    dsort: torch.Tensor,
    results: ARMA11Results,
    *,
    fitted_column_indices: list[int] | None,
    glt_contrasts_tensor: torch.Tensor | None,
    want_r2_partial: bool,
    r2_partial_mode: str,
    want_r2_semipartial: bool,
    r2_semipartial_mode: str,
    want_residuals: bool,
    want_predicted: bool,
    run_starts: list[int] | torch.Tensor | None,
    tau: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    y_norm_scale: torch.Tensor | None,
    batch_size: int,
    verbose: bool,
    run_bounds: list[int] | None = None,
    want_ljung_box: bool = False,
    lj_max_lag: int | None = None,
    lj_tau: torch.Tensor | None = None,
) -> None:
    """Redo the final GLS per voxel with an extended ``[design | dsort_v]`` design.

    ``(a, b)`` is taken from ``results.arma_params`` (estimated WITHOUT dsort, per
    AFNI). Whitening ``L`` is constant within an ``(a, b)`` group, so we whiten the
    base design once per group and whiten each voxel's dsort columns, then solve a
    batched per-voxel GLS. Results overwrite ``results.betas/tstats/sigma2/r2/fstats``
    and populate ``results.dsort_betas/dsort_tstats``. DoF drops by ``n_dsort``.

    ``run_bounds`` (``[t0, t1, ..., n_time]``) turns on lazy per-run block-diagonal
    expansion: ``dsort`` is then the COMPACT ``(n_vox, n_sets, n_time)`` form (one
    run-concatenated column per -dsort set) and each set is expanded into one
    zero-padded column per run *inside* each voxel sub-batch. This keeps the full
    dense ``(n_vox, n_sets*n_runs, n_time)`` block tensor from ever being
    materialized — at whole-brain voxel counts that ×n_runs blow-up OOMs.

    All betas are produced in the same per-voxel-normalized units as the main loop
    (Y divided by its std when ``y_norm_scale`` is set); the caller's unscale block
    restores physical units for both base and dsort betas.
    """
    n_voxels, n_cols, n_timepoints = dsort.shape
    if run_bounds is not None:
        n_runs = len(run_bounds) - 1
        n_sets = n_cols
        n_dsort = n_sets * n_runs  # expanded per-run column count
    else:
        n_runs = 1
        n_sets = n_cols
        n_dsort = n_cols
    p = design.shape[1]
    p_ext = p + n_dsort
    dof = max(1, n_timepoints - p_ext)
    results.dof = dof

    results.dsort_betas = torch.zeros(n_voxels, n_dsort, device="cpu", dtype=dtype)
    results.dsort_tstats = torch.zeros(n_voxels, n_dsort, device="cpu", dtype=dtype)

    fci = (
        torch.as_tensor(fitted_column_indices, device=device, dtype=torch.long)
        if fitted_column_indices is not None
        else None
    )

    # Group voxels by optimal (a, b) so L (and whitened base design) is computed
    # once per group, mirroring the main GLS loop's grouping. arma_params was
    # populated by the main (no-dsort) GLS fit that always precedes this pass.
    assert results.arma_params is not None
    ab_cpu = results.arma_params.cpu().contiguous()
    unique_pairs, inverse_indices = torch.unique(ab_cpu, dim=0, return_inverse=True)
    sort_keys, sort_perm = torch.sort(inverse_indices, stable=True)
    group_starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            (sort_keys[1:] != sort_keys[:-1]).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([len(sort_keys)], dtype=torch.long),
        ]
    )

    # The per-voxel extended solve carries a (B, p_ext, p_ext) X'X per sub-batch,
    # so it needs a smaller chunk than the shared-design path. Size it through the
    # memory module (operation="arma_dsort"; n_trials carries the dsort count).
    from fastfuncstuff.memory import estimate_chunk_size

    sub_batch = estimate_chunk_size(
        n_voxels=n_voxels,
        n_timepoints=n_timepoints,
        n_regressors=p,
        device=device,
        operation="arma_dsort",
        use_double=(dtype == torch.float64),
        n_trials=n_dsort,
    )
    sub_batch = max(1, min(sub_batch, batch_size, n_voxels))

    if verbose:
        print(
            f"\n📦 -dsort: per-voxel GLS with {n_dsort} voxel-wise regressor(s) "
            f"(DoF {n_timepoints} - {p} - {n_dsort} = {dof}); "
            f"sub-batch {sub_batch:,} voxels"
        )
    pbar = (
        tqdm(total=len(unique_pairs), desc="dsort GLS", unit="group")
        if verbose and len(unique_pairs) > 1
        else None
    )

    for g_idx in range(len(unique_pairs)):
        a_opt = float(unique_pairs[g_idx, 0])
        b_opt = float(unique_pairs[g_idx, 1])
        voxel_indices = sort_perm[group_starts[g_idx] : group_starts[g_idx + 1]]

        # Whiten the base design once for this (a, b); L is its Cholesky factor.
        X_w, _, L_chol = prewhiten_with_arma11(
            design, design[:, 0], a_opt, b_opt, run_starts=run_starts, tau=tau
        )

        n_group = voxel_indices.numel()
        sub = max(1, min(sub_batch, n_group))
        for s0 in range(0, n_group, sub):
            idx = voxel_indices[s0 : s0 + sub]
            B = idx.numel()

            # Data (n_time, B), normalized to match the main loop's conditioning.
            Y = data[idx].T.to(device)
            if y_norm_scale is not None:
                Y = Y / y_norm_scale[idx].to(device).unsqueeze(0)
            Yw = torch.linalg.solve_triangular(L_chol, Y, upper=False)  # (n_time, B)

            # This batch's dsort columns (B, n_dsort, n_time). With run_bounds the
            # stored form is compact (B, n_sets, n_time); expand each set into one
            # zero-padded column per run HERE, so the dense ×n_runs tensor only ever
            # exists for B voxels at a time (never the whole brain).
            if run_bounds is not None:
                Dc = dsort[idx].to(device)  # (B, n_sets, n_time)
                D = torch.zeros(B, n_dsort, n_timepoints, device=device, dtype=Dc.dtype)
                for si in range(n_sets):
                    for r in range(n_runs):
                        t0, t1 = run_bounds[r], run_bounds[r + 1]
                        D[:, si * n_runs + r, t0:t1] = Dc[:, si, t0:t1]
            else:
                D = dsort[idx].to(device)  # (B, n_dsort, n_time)
            # Whiten: (B, n_dsort, n_time) -> (n_time, B*n_dsort).
            Dmat = D.permute(2, 0, 1).reshape(n_timepoints, B * n_dsort)
            Dw = torch.linalg.solve_triangular(L_chol, Dmat, upper=False)
            Dw = Dw.reshape(n_timepoints, B, n_dsort).permute(1, 0, 2)  # (B,n_time,q)

            # Extended whitened design per voxel: (B, n_time, p+q).
            X_ext = torch.cat([X_w.unsqueeze(0).expand(B, -1, -1), Dw], dim=2)
            Yw_b = Yw.T.unsqueeze(2)  # (B, n_time, 1)

            XtX = X_ext.transpose(1, 2) @ X_ext  # (B, p_ext, p_ext)
            Xty = X_ext.transpose(1, 2) @ Yw_b  # (B, p_ext, 1)
            eye = torch.eye(p_ext, device=device, dtype=dtype)
            try:
                Lc = torch.linalg.cholesky(XtX + 1e-8 * eye)
                betas = torch.cholesky_solve(Xty, Lc)  # (B, p_ext, 1)
                XtX_inv = torch.cholesky_inverse(Lc)  # (B, p_ext, p_ext)
            except torch.linalg.LinAlgError:
                XtX_inv = torch.linalg.pinv(XtX)
                betas = XtX_inv @ Xty
            betas = betas.squeeze(2)  # (B, p_ext)

            # Whitened residuals and noise variance (fp64 accumulation off MPS).
            resid_w = (Yw_b - X_ext @ betas.unsqueeze(2)).squeeze(2)  # (B, n_time)
            sigma2 = (resid_w.pow(2).sum(dim=1, dtype=accum_dtype) / dof).to(dtype)

            diag_inv = torch.diagonal(XtX_inv, dim1=1, dim2=2)  # (B, p_ext)
            se = torch.sqrt(sigma2.unsqueeze(1) * diag_inv)
            tstats = betas / (se + 1e-10)

            betas_base = betas[:, :p]
            tstats_base = tstats[:, :p]

            # Prediction / R² in original (un-whitened) space, normalized units.
            pred = (design @ betas_base.T).T  # (B, n_time)
            pred = pred + torch.einsum("bq,bqt->bt", betas[:, p:], D)
            r2 = compute_r2_metric(Y.T, pred, metric="cod")

            # Omnibus F over task columns (or all base columns), matching the
            # main loop: F = β' A^{-1} β / (σ² p) with A = XtX_inv task sub-block.
            if fci is not None:
                A = XtX_inv.index_select(1, fci).index_select(2, fci)
                beta_t = betas_base.index_select(1, fci)
                p_f = fci.numel()
            else:
                A = XtX_inv[:, :p, :p]
                beta_t = betas_base
                p_f = p
            A_inv = torch.linalg.inv(A + 1e-8 * torch.eye(p_f, device=device, dtype=dtype))
            quad = torch.einsum("bi,bij,bj->b", beta_t, A_inv, beta_t)
            fstats = quad / (sigma2 * p_f + 1e-30)

            # ---- store base + dsort results ----
            if fci is not None:
                results.betas[idx] = betas_base.index_select(1, fci).cpu()
                results.tstats[idx] = tstats_base.index_select(1, fci).cpu()
            else:
                results.betas[idx] = betas_base.cpu()
                results.tstats[idx] = tstats_base.cpu()
            results.dsort_betas[idx] = betas[:, p:].cpu()
            results.dsort_tstats[idx] = tstats[:, p:].cpu()
            results.sigma2[idx] = sigma2.cpu()
            results.fstats[idx] = fstats.cpu()
            results.r2[idx] = r2.cpu()

            # Partial / semi-partial R² per base regressor (from t and R²).
            if want_r2_partial or want_r2_semipartial:
                _store_partial_semipartial(
                    results,
                    idx,
                    tstats_base=tstats_base,
                    r2=r2,
                    dof=dof,
                    p=p,
                    fitted_column_indices=fitted_column_indices,
                    want_r2_partial=want_r2_partial,
                    r2_partial_mode=r2_partial_mode,
                    want_r2_semipartial=want_r2_semipartial,
                    r2_semipartial_mode=r2_semipartial_mode,
                    dtype=dtype,
                )

            # GLT contrasts use the base-design covariance block (dsort betas
            # cannot enter a GLT, per AFNI).
            if glt_contrasts_tensor is not None:
                var_base = sigma2.unsqueeze(1).unsqueeze(2) * XtX_inv[:, :p, :p]
                c_betas = betas_base @ glt_contrasts_tensor.T
                c_vars = torch.einsum(
                    "cr,brs,cs->bc",
                    glt_contrasts_tensor,
                    var_base,
                    glt_contrasts_tensor,
                )
                c_se = torch.sqrt(torch.clamp(c_vars, min=0.0))
                c_t = c_betas / (c_se + 1e-10)
                results.contrast_betas[idx] = c_betas.cpu()
                results.contrast_tstats[idx] = c_t.cpu()
                if want_r2_partial and results.contrast_r2_partial is not None:
                    ct2 = c_t**2
                    results.contrast_r2_partial[idx] = (ct2 / (ct2 + dof)).cpu()
                if want_r2_semipartial and results.contrast_r2_semipartial is not None:
                    ct2 = c_t**2
                    var_rem = torch.clamp(1.0 - r2.to(dtype).unsqueeze(1), min=0.0)
                    results.contrast_r2_semipartial[idx] = ((ct2 / (ct2 + dof)) * var_rem).cpu()

            # LB reflects the final (dsort-extended) fit, so it is recomputed here
            # over the residuals that actually ship, not the no-dsort pass's.
            if want_ljung_box and results.ljung_box is not None:
                results.ljung_box[idx] = (
                    _ljung_box_batched(resid_w, lj_max_lag, lj_tau).to(torch.float32).cpu()
                )
            if want_residuals:
                results.residuals[idx] = (Y.T - pred).cpu()
                results.residuals_whitened[idx] = resid_w.cpu()
            if want_predicted:
                results.predicted[idx] = pred.cpu()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()


def _store_partial_semipartial(
    results: ARMA11Results,
    idx: torch.Tensor,
    *,
    tstats_base: torch.Tensor,
    r2: torch.Tensor,
    dof: int,
    p: int,
    fitted_column_indices: list[int] | None,
    want_r2_partial: bool,
    r2_partial_mode: str,
    want_r2_semipartial: bool,
    r2_semipartial_mode: str,
    dtype: torch.dtype,
) -> None:
    """Partial/semi-partial R² for a sub-batch, split into task vs nuisance.

    Mirrors the main GLS loop's formulas (r²_partial = t²/(t²+df);
    r²_semi = r²_partial · (1 - R²_full)) but operates on the dsort-extended fit's
    base t-stats. Used only by the -dsort pass.
    """
    t2 = tstats_base**2
    r2_partial_full = t2 / (t2 + dof)  # (B, p)

    if fitted_column_indices is not None:
        nuisance_indices = sorted(set(range(p)) - set(fitted_column_indices))
        r2_partial_task = r2_partial_full[:, fitted_column_indices]
        r2_partial_nuis = r2_partial_full[:, nuisance_indices] if nuisance_indices else None
    else:
        nuisance_indices = []
        r2_partial_task = r2_partial_full
        r2_partial_nuis = None

    if want_r2_partial:
        if r2_partial_mode == "task" and r2_partial_nuis is not None:
            denom = torch.clamp(1.0 - r2_partial_nuis.sum(dim=1, keepdim=True), min=0.01)
            results.r2_partial[idx] = (r2_partial_task / denom).cpu()
        else:
            results.r2_partial[idx] = r2_partial_task.cpu()
        if r2_partial_nuis is not None and results.r2_partial_nuisance is not None:
            results.r2_partial_nuisance[idx] = r2_partial_nuis.cpu()

    if want_r2_semipartial:
        var_rem = torch.clamp(1.0 - r2.to(dtype).unsqueeze(1), min=0.0)
        r2_semi_task = r2_partial_task * var_rem
        r2_semi_nuis = r2_partial_nuis * var_rem if r2_partial_nuis is not None else None
        if r2_semipartial_mode == "task" and r2_semi_nuis is not None:
            denom = torch.clamp(1.0 - r2_semi_nuis.sum(dim=1, keepdim=True), min=0.01)
            results.r2_semipartial[idx] = (r2_semi_task / denom).cpu()
        else:
            results.r2_semipartial[idx] = r2_semi_task.cpu()
        if r2_semi_nuis is not None and results.r2_semipartial_nuisance is not None:
            results.r2_semipartial_nuisance[idx] = r2_semi_nuis.cpu()


def fit_glm_arma11(
    data: torch.Tensor | np.ndarray,
    design: torch.Tensor | np.ndarray,
    tr: float,
    a_grid: torch.Tensor | np.ndarray | None = None,
    b_grid: torch.Tensor | np.ndarray | None = None,
    estimate_per_voxel: bool = True,
    batch_size: int | None = None,
    want_residuals: bool = False,
    want_predicted: bool = False,
    want_ljung_box: bool = False,
    want_r2_partial: bool = False,
    r2_partial_mode: str = "full",  # "full" or "task" - how to compute partial R²
    want_r2_semipartial: bool = False,
    r2_semipartial_mode: str = "full",  # "full" or "task" - how to compute semi-partial R²
    want_ols: bool = False,
    want_ols_residuals: bool = False,
    want_ols_predicted: bool = False,
    ols_write_callback: Callable | None = None,
    precomputed_arma_params: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    cholesky_on_cpu: bool = True,
    use_double: bool = False,
    use_qr: bool = False,
    debug_memory: bool = False,
    enable_quick_estimate: bool = False,
    force_exhaustive_search: bool = False,
    glt_labels: list[str] | None = None,
    glt_matrices: list[np.ndarray] | None = None,
    task_indices: list[int] | None = None,
    use_grid_batching: bool | None = None,
    spatial_metadata: dict | None = None,
    legacy_contrasts: bool = False,
    save_profile_likelihoods: bool = False,
    run_starts: list[int] | torch.Tensor | None = None,
    autocorr_cache: dict | None = None,
    dsort: torch.Tensor | np.ndarray | None = None,
    dsort_labels: list[str] | None = None,
    dsort_run_bounds: list[int] | None = None,
    want_dsort_nods: bool = False,
    tau: torch.Tensor | None = None,
) -> ARMA11Results:
    """
    Fit GLM with ARMA(1,1) prewhitening (AFNI 3dREMLfit style)

    This is the main entry point for ARMA(1,1) GLM analysis. Provides:
    - Accurate parameter estimates via GLS
    - Corrected t-statistics accounting for temporal autocorrelation
    - Per-voxel or global ARMA parameter estimation
    - Massive GPU acceleration (5-30x faster than AFNI)

    Parameters
    ----------
    data : array-like, shape (n_voxels, n_timepoints) or (n_x, n_y, n_z, n_timepoints)
        fMRI data
    design : array-like, shape (n_timepoints, n_regressors)
        Design matrix (already convolved with HRF)
    tr : float
        Repetition time in seconds (for info only)
    a_grid : array-like, optional
        Grid of AR parameters (default: 0.1 to 0.9 in 0.1 steps)
    b_grid : array-like, optional
        Grid of MA parameters (default: -0.3 to 0.3 in 0.1 steps)
    estimate_per_voxel : bool, default=True
        If True: Estimate (a,b) separately for each voxel (most accurate)
        If False: Estimate (a,b) once from mean timeseries (faster)
    batch_size : int or None, default=None
        Number of voxels to process in parallel (for per-voxel estimation)
        If None (default): Auto-detect based on GPU memory
          - MPS (Mac): 20k-100k voxels (36GB unified memory)
          - CUDA: 10k-60k voxels (based on VRAM)
          - CPU: 5k voxels (conservative)
        Manual override: specify exact number (e.g., 50000)
        Larger = faster but more GPU memory required
    want_residuals : bool, default=False
        Return prewhitened residuals
    want_predicted : bool, default=False
        Return predicted timecourses
    want_ljung_box : bool, default=False
        Compute the Ljung-Box whiteness statistic (AFNI ``-Rvar`` sub-brick 5)
        into ``results.ljung_box``. Accumulated per voxel-batch from the whitened
        residuals that already exist in the GLS loop, so it does **not** require
        ``want_residuals`` and never materialises a whole-brain residual array.
    want_ols : bool, default=False
        Also compute OLS fit for comparison (stored in results.ols_results)
    ols_write_callback : callable, optional
        Function to call immediately after OLS completes to write results to disk.
        Signature: callback(ols_results: GLMResults, original_shape: tuple, affine: ndarray) -> None
        Receives OLS results plus spatial metadata for writing.
        This frees OLS memory before starting ARMA loop (critical for large datasets).
        If provided, results.ols_results will be set to None after callback.
    precomputed_arma_params : array-like, shape (n_voxels, 2), optional
        Precomputed ARMA(1,1) parameters [a, b] for each voxel.
        If provided, skips REML estimation (saves ~80% of compute time).
        Useful for:
        - Re-running analysis with different contrasts on same data
        - Using AFNI-estimated parameters for validation/comparison
        - Iterative analysis workflows
    device : torch.device, optional
        Computing device
    verbose : bool, default=True
        Print progress
    cholesky_on_cpu : bool, default=True
        Compute Cholesky decompositions on CPU (recommended for large grids).
        - True (default): Uses system RAM, avoids GPU OOM, still very fast
        - False: Uses GPU VRAM, faster but may OOM with large grids (>200 params)
        Set to False only if you have abundant GPU memory and small grids.
    use_double : bool, default=False
        If True, use float64 precision (matches AFNI exactly, ~2x memory, ~1.5x slower).
        If False, use float32 precision (faster, tiny differences from AFNI).
    use_qr : bool, default=False
        If True, use QR factorization (AFNI approach, more stable, uses 2.25x more memory).
        If False, use X'X (faster, less memory, good for well-conditioned matrices with float64).
        Recommendation: False for float64 + well-conditioned data, True for float32 or ill-conditioned data.
    use_grid_batching : bool or None, default=None
        Strategy for REML grid search (per-voxel estimation only):
        - None (default): Auto-detect based on grid memory
          - If grid > 8 GB: Use grid batching (low memory, slightly slower)
          - If grid ≤ 8 GB: Precompute full grid (fast, more memory)
        - True: Force grid batching (loop through (a,b) pairs, process all voxels per pair)
          - Memory: ~3 GB regardless of grid size
          - Speed: Slightly slower (no precomputation)
          - Best for: Long timeseries, double precision, limited GPU memory
        - False: Force full grid precomputation (AFNI's approach)
          - Memory: Can be 10+ GB with long timeseries
          - Speed: Fastest (all Cholesky factorizations precomputed)
          - Best for: Short timeseries, float32, abundant GPU memory
    n_glt : int, default=0
        Number of GLT contrasts (parsed from design matrix .xmat.1D file).
        If > 0: Allocates var_betas (n_voxels × n_regressors × n_regressors) for contrast computation.
        If 0: Skips var_betas allocation to save memory (critical with many regressors).
        Example: 322 regressors × 918k voxels = 762 GB saved when n_glt=0!

    Returns
    -------
    results : ARMA11Results
        Object containing:
        - betas : (n_voxels, n_regressors) GLS parameter estimates
        - tstats : (n_voxels, n_regressors) t-statistics (corrected for autocorrelation)
        - arma_params : (n_voxels, 2) estimated (a, b) per voxel
        - arma_lambda : (n_voxels,) lag-1 correlation
        - r2 : (n_voxels,) R² values
        - sigma2 : (n_voxels,) noise variance estimates
        - [optional] residuals, predicted

    Notes
    -----
    Algorithm (per AFNI 3dREMLfit):
    1. OLS fit to get initial residuals
    2. REML grid search to find optimal (a, b) per voxel
    3. Prewhiten design and data with ARMA(1,1) covariance
    4. GLS fit on prewhitened data
    5. Compute corrected t-statistics

    GPU Performance (with vectorized batch processing):
    - 100,000 voxels, per-voxel estimation: ~5-10 minutes (batch_size=10000)
    - 300,000 voxels (typical masked brain): ~15-30 minutes
    - vs AFNI 3dREMLfit: hours to days
    - Speedup: 10-50x depending on dataset size and GPU

    When to use:
    - Final publication-quality analysis
    - When temporal autocorrelation is significant (most fMRI data)
    - For accurate t-statistics and p-values
    - When using 3dMEMA or other meta-analysis methods

    When NOT to use:
    - Quick exploratory analysis (use OLS or AR(1))
    - Design optimization (use metrics_empirical.py AR(1) methods)

    Examples
    --------
    >>> # Simple example
    >>> results = fit_glm_arma11(data, design, tr=2.0)
    >>> print(f"Mean (a,b): ({results.arma_params[:, 0].mean():.3f}, "
    ...       f"{results.arma_params[:, 1].mean():.3f})")
    >>> print(f"Mean R²: {results.r2.mean():.3f}")
    >>>
    >>> # Compare to OLS
    >>> from glm_core import fit_glm
    >>> ols_results = fit_glm(data, design, tr=2.0)
    >>> print(f"OLS R²: {ols_results.r2.mean():.3f}")
    >>> print(f"ARMA R²: {results.r2.mean():.3f}")  # Usually slightly higher

    References
    ----------
    - AFNI 3dREMLfit: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/remlfit.html
    - Woolrich et al. (2001): Temporal autocorrelation in univariate linear modeling
    - Worsley & Friston (1995): Analysis of fMRI time-series revisited—again
    """
    # Setup precision
    if device is None:
        device = get_device()

    # MPS has no float64. Honour an explicit full-precision request by running on
    # CPU rather than silently downgrading — and CPU is faster for the REML path
    # anyway (MPS's batched Cholesky/triangular-solve is ~5x slower than LAPACK).
    if use_double and device.type == "mps":
        warnings.warn(
            "use_double requested but MPS has no float64; running ARMA on CPU to "
            "honour the full-precision request. Pass -device cpu to silence this, "
            "or drop use_double to run float32 on MPS.",
            stacklevel=2,
        )
        device = torch.device("cpu")
        # Move inputs off MPS before the float64 cast below (MPS can't cast f64).
        if isinstance(data, torch.Tensor) and data.device.type == "mps":
            data = data.cpu()
        if isinstance(design, torch.Tensor) and design.device.type == "mps":
            design = design.cpu()
    dtype = torch.float64 if use_double else torch.float32

    # On CUDA/CPU the fp64 accumulator eliminates rounding in sum-of-squares;
    # MPS cannot hold float64 so it accumulates in float32.
    _accum_dtype = dtype if device.type == "mps" else torch.float64
    if device.type == "mps":
        warn_mps_float32_precision("ARMA RSS accumulation")
    storage_device = torch.device("cpu")

    # Enable cuDNN benchmarking for optimal kernel selection (GPU only)
    # This auto-tunes algorithms for better performance (~5-15% speedup)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Disable TF32 for float32: Blackwell/Ampere use TF32 by default (10-bit mantissa).
    # Full float32 mantissa (23-bit) is needed for accurate REML/GLS. No-op for float64.
    _saved_tf32: bool | None = None
    if device.type == "cuda" and not use_double:
        _saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    # Convert to tensors
    data = to_tensor(data, device=None, dtype=dtype)
    if data.device.type != "cpu":
        data = data.to(storage_device)
    design = to_tensor(design, device=device).to(dtype)

    # Track original design dimensions
    n_regressors_full = design.shape[1]

    # NOTE: We fit the FULL design (including nuisance regressors like polynomials and motion)
    # but only SAVE the task regressors. Filtering happens during output, not during fitting!
    # This ensures proper baseline modeling - without nuisance regressors, task betas absorb drift.
    fitted_column_indices = task_indices  # Track which columns to extract for output
    n_regressors = n_regressors_full  # Fit all regressors

    if debug_memory:
        _debug_memory_snapshot(
            "START (after data/design load)",
            device,
            {
                "data": data,
                "design": design,
            },
        )

    # Handle spatial dimensions
    original_shape = None
    if data.ndim == 4:
        # (n_x, n_y, n_z, n_timepoints)
        original_shape = data.shape[:3]
        data = data.reshape(-1, data.shape[-1])  # (n_voxels, n_timepoints)
    elif data.ndim == 2:
        # Already (n_voxels, n_timepoints)
        pass
    else:
        raise ValueError(f"Data must be 2D or 4D, got {data.ndim}D")

    n_voxels, n_timepoints = data.shape

    # Voxel-wise regressors (-dsort / ANATICOR). Already masked to the analysis
    # voxels upstream. Keep on the CPU storage device and stream per sub-batch in
    # the dsort GLS pass; (a, b) is still estimated on the base design only.
    if dsort is not None:
        dsort = to_tensor(dsort, device=None, dtype=dtype)
        if dsort.device.type != "cpu":
            dsort = dsort.to(storage_device)
        if dsort.ndim == 2:
            dsort = dsort.unsqueeze(1)  # (n_voxels, n_time) -> (n_voxels, 1, n_time)
        if dsort.shape[0] != n_voxels or dsort.shape[2] != n_timepoints:
            raise ValueError(
                f"dsort must be (n_voxels={n_voxels}, n_dsort, n_timepoints="
                f"{n_timepoints}); got {tuple(dsort.shape)}"
            )
        dsort = _dsort_constant_guard(dsort)

    # Per-voxel normalization for float32 numerical conditioning.
    # Dividing Y by its std before REML and GLS eliminates catastrophic cancellation
    # (residuals are O(1) instead of O(signal/noise_ratio)).
    # REML argmin, R², t-stats, and F-stats are all scale-invariant.
    # Betas, sigma2, residuals, and predicted are unscaled in post-processing.
    _y_norm_scale: torch.Tensor | None = None
    if not use_double:
        _y_norm_scale = data.std(dim=1).clamp(min=1e-8)  # (n_voxels,) on CPU

    # Validate design matrix
    if design.shape[0] != n_timepoints:
        raise ValueError(
            f"Timepoints mismatch: Data has {n_timepoints}, Design has {design.shape[0]}"
        )

    n_regressors = design.shape[1]

    # Adaptive batch sizing based on GPU memory
    if batch_size is None:
        base_batch_size = get_adaptive_batch_size(
            device, n_timepoints, n_regressors, use_double, use_qr
        )

        # Account for grid memory (AFNI strategy)
        # Grid is loaded ONCE and reused for all batches - much faster than streaming!
        if (device.type == "cuda" or device.type == "cpu") and estimate_per_voxel:
            # Estimate grid memory footprint
            if a_grid is None and b_grid is None:
                temp_a_grid, temp_b_grid = get_default_arma_grids(device)
            else:
                temp_a_grid = a_grid if a_grid is not None else get_default_arma_grids(device)[0]
                temp_b_grid = b_grid if b_grid is not None else get_default_arma_grids(device)[1]

            if not isinstance(temp_a_grid, torch.Tensor):
                temp_a_grid = to_tensor(temp_a_grid, device=device)
            if not isinstance(temp_b_grid, torch.Tensor):
                temp_b_grid = to_tensor(temp_b_grid, device=device)

            # Estimate valid pairs
            temp_a_grid, temp_b_grid = ensure_zero_in_grid(temp_a_grid, temp_b_grid)
            n_valid_pairs = estimate_valid_grid_pairs(temp_a_grid, temp_b_grid)

            # Calculate grid memory
            grid_memory_bytes = calculate_grid_memory_footprint(
                n_valid_pairs, n_timepoints, n_regressors, use_double
            )

            # Auto-detect whether to use grid batching
            GRID_BATCHING_THRESHOLD_GB = 8.0
            if use_grid_batching is None:
                # Auto-detect: use grid batching if grid > 8 GB
                use_grid_batching = grid_memory_bytes > (GRID_BATCHING_THRESHOLD_GB * 1024**3)
                if use_grid_batching and verbose:
                    print(
                        f"\n💡 Auto-enabling grid batching (grid: {grid_memory_bytes / 1024**3:.1f} GB > {GRID_BATCHING_THRESHOLD_GB} GB threshold)"
                    )
                    print("   This saves memory by processing one (a,b) pair at a time.")
                    print("   Use -no_grid_batching to force full grid precomputation.\n")

            if debug_memory:
                print(f"\n{'=' * 70}")
                print("EXPECTED GRID SIZE CALCULATION")
                print(f"{'=' * 70}")
                print(f"  n_valid_pairs: {n_valid_pairs}")
                print(f"  n_timepoints:  {n_timepoints}")
                print(f"  n_regressors:  {n_regressors}")
                print(f"  dtype:         {'float64' if use_double else 'float32'}")
                print(f"  bytes_per_element: {8 if use_double else 4}")
                bytes_elem = 8 if use_double else 4
                L_inv_bytes = n_valid_pairs * n_timepoints * n_timepoints * bytes_elem
                X_w_bytes = n_valid_pairs * n_timepoints * n_regressors * bytes_elem
                XwTXw_bytes = n_valid_pairs * n_regressors * n_regressors * bytes_elem
                scalars_bytes = n_valid_pairs * 2 * bytes_elem
                print("  Expected grid memory:")
                print(
                    f"    L_inv:   {L_inv_bytes / 1024**3:.3f} GiB  ({n_valid_pairs} × {n_timepoints} × {n_timepoints})"
                )
                print(
                    f"    X_w:     {X_w_bytes / 1024**3:.3f} GiB  ({n_valid_pairs} × {n_timepoints} × {n_regressors})"
                )
                print(
                    f"    XwTXw:   {XwTXw_bytes / 1024**3:.3f} GiB  ({n_valid_pairs} × {n_regressors} × {n_regressors})"
                )
                print(f"    scalars: {scalars_bytes / 1024**2:.1f} MiB  ({n_valid_pairs} × 2)")
                print(f"  TOTAL EXPECTED: {grid_memory_bytes / 1024**3:.3f} GiB")
                print(f"{'=' * 70}\n")

            # Route batch sizing through the memory module so GPU/CPU policy
            # lives in one place. Subtract grid memory from the budget first
            # so we don't blow up when the precomputed grid is big.
            from fastfuncstuff.memory import (
                estimate_chunk_size,
                get_available_memory,
                get_memory_config,
            )

            _mem_cfg = get_memory_config()
            available_mem = get_available_memory(device)
            memory_for_batches = available_mem - grid_memory_bytes

            if memory_for_batches > 0:
                bytes_per_voxel = bytes_per_voxel_arma(n_timepoints, n_regressors)
                if use_double:
                    bytes_per_voxel = int(bytes_per_voxel * _mem_cfg.double_precision_multiplier)
                # estimate_chunk_size clamps to n_voxels, so the result is
                # already the "fits everything in one batch" answer when
                # memory allows.
                batch_size = estimate_chunk_size(
                    n_voxels=n_voxels,
                    n_timepoints=n_timepoints,
                    n_regressors=n_regressors,
                    device=device,
                    operation="arma",
                    use_double=use_double,
                )
                # Honour the per-call cap from base_batch_size as well.
                batch_size = max(100, min(batch_size, base_batch_size, n_voxels))
            else:
                # Grid too large! This shouldn't happen if auto-detection worked
                # Only possible if user forced use_grid_batching=False
                if use_grid_batching:
                    # Grid batching should handle this - continue
                    batch_size = base_batch_size
                else:
                    raise MemoryError(
                        f"\n{'=' * 70}\n"
                        f"ERROR: Grid too large for available GPU memory!\n"
                        f"{'=' * 70}\n"
                        f"  Grid needs: {grid_memory_bytes / 1024**3:.2f} GB\n"
                        f"  Available:  {available_mem / 1024**3:.2f} GB\n"
                        f"  Deficit:    {-memory_for_batches / 1024**3:.2f} GB\n\n"
                        f"  Timeseries length: {n_timepoints} TRs\n"
                        f"  Regressors: {n_regressors}\n"
                        f"  Precision: {'float64' if use_double else 'float32'}\n\n"
                        f"Solutions:\n"
                        f"  1. Enable grid batching (should auto-enable, or add -grid_batching)\n"
                        f"  2. Use float32 (remove -double flag) → 2x less memory\n"
                        f"  3. Use global ARMA (add -GOFORIT) → no grid needed\n"
                        f"  4. Reduce grid size (use -Grid_...) → smaller grid\n"
                        f"  5. Use a GPU with more memory\n"
                        f"{'=' * 70}\n"
                    )

            if verbose:
                grid_mb = grid_memory_bytes / (1024**2)
                if use_grid_batching:
                    print(f"Grid size: {n_valid_pairs} (a,b) pairs")
                    print("Strategy: Grid batching (low memory, process all voxels per grid point)")
                    print(
                        f"  Memory per grid point: ~{grid_memory_bytes / n_valid_pairs / 1024**2:.1f} MiB"
                    )
                    print(f"  Batch size: {batch_size:,} voxels")
                else:
                    print(f"Grid memory: {grid_mb:.1f} MiB ({n_valid_pairs} pairs)")
                    print("Strategy: Full grid precomputation (AFNI approach)")
                    print(f"Adjusted batch size: {batch_size:,} voxels (grid + batches fit in GPU)")
        else:
            batch_size = base_batch_size
            if verbose:
                print(f"Auto-detected batch size: {batch_size:,} voxels (device={device.type})")

    if verbose:
        print("ARMA(1,1) GLM Fit")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Regressors: {n_regressors}")
        print(f"  Per-voxel estimation: {estimate_per_voxel}")
        if estimate_per_voxel:
            print(f"  Batch size: {batch_size:,}")

    # Initialize results
    results = ARMA11Results()
    results.original_shape = original_shape

    # Track which columns were fitted (for proper labeling of outputs)
    results.fitted_column_indices = fitted_column_indices  # None if all fitted, list if filtered
    results.n_regressors_full = n_regressors_full  # Original design matrix width before filtering

    raw_dof = n_timepoints - n_regressors
    if raw_dof <= 0:
        warnings.warn(
            "Non-positive degrees of freedom detected in ARMA(1,1) fit; statistics may be unreliable",
            stacklevel=2,
        )

    # Allocate storage for results
    # If task_indices specified, only store those columns (save memory)
    # Otherwise store all columns
    n_output_regressors = len(fitted_column_indices) if fitted_column_indices else n_regressors
    results.betas = torch.zeros(n_voxels, n_output_regressors, device=storage_device, dtype=dtype)
    results.tstats = torch.zeros(n_voxels, n_output_regressors, device=storage_device, dtype=dtype)
    results.r2 = torch.zeros(
        n_voxels, device=storage_device
    )  # always float32 (compute_r2_metric returns float32)

    # Allocate partial R² storage if requested
    if want_r2_partial:
        results.r2_partial = torch.zeros(
            n_voxels, n_output_regressors, device=storage_device, dtype=dtype
        )
        # Also allocate for nuisance if we have filtered columns
        if fitted_column_indices is not None:
            n_nuisance = n_regressors - len(fitted_column_indices)
            if n_nuisance > 0:
                results.r2_partial_nuisance = torch.zeros(
                    n_voxels, n_nuisance, device=storage_device, dtype=dtype
                )

    # Allocate semi-partial R² storage if requested
    if want_r2_semipartial:
        results.r2_semipartial = torch.zeros(
            n_voxels, n_output_regressors, device=storage_device, dtype=dtype
        )
        # Also allocate for nuisance if we have filtered columns
        if fitted_column_indices is not None:
            n_nuisance = n_regressors - len(fitted_column_indices)
            if n_nuisance > 0:
                results.r2_semipartial_nuisance = torch.zeros(
                    n_voxels, n_nuisance, device=storage_device, dtype=dtype
                )

    results.arma_params = torch.zeros(n_voxels, 2, device=storage_device, dtype=dtype)
    results.arma_lambda = torch.zeros(n_voxels, device=storage_device, dtype=dtype)
    results.reml_likelihood = torch.zeros(n_voxels, device=storage_device, dtype=dtype)
    results.sigma2 = torch.zeros(n_voxels, device=storage_device, dtype=dtype)
    results.fstats = torch.zeros(n_voxels, device=storage_device, dtype=dtype)

    # MEMORY OPTIMIZATION: Compute contrasts in-loop, never store full var_betas!
    # OLD approach: Store (n_voxels, n_reg, n_reg) covariance = 381 GB for 322 regressors
    # NEW approach: Store (n_voxels, n_contrasts) results = ~14 MB for 2 contrasts
    # Setup GLT contrasts (if present in design matrix)
    glt_contrasts_tensor = None
    n_contrasts = 0
    if glt_labels and glt_matrices:
        n_contrasts = len(glt_labels)
        if verbose:
            old_mem_gb = (n_voxels * n_regressors * n_regressors * 8) / (1024**3) / 2
            new_mem_mb = (n_voxels * n_contrasts * 8) / (1024**2)
            print(f"📊 {n_contrasts} GLT contrasts will be computed in-loop")
            print(
                f"   Memory: {new_mem_mb:.1f} MB (vs {old_mem_gb:.1f} GB if storing full var_betas)"
            )

        results.contrast_labels = glt_labels
        results.contrast_betas = torch.zeros(
            n_voxels, n_contrasts, device=storage_device, dtype=dtype
        )
        results.contrast_tstats = torch.zeros(
            n_voxels, n_contrasts, device=storage_device, dtype=dtype
        )
        # Allocate partial R² for contrasts if requested
        if want_r2_partial:
            results.contrast_r2_partial = torch.zeros(
                n_voxels, n_contrasts, device=storage_device, dtype=dtype
            )

        # Allocate semi-partial R² for contrasts if requested
        if want_r2_semipartial:
            results.contrast_r2_semipartial = torch.zeros(
                n_voxels, n_contrasts, device=storage_device, dtype=dtype
            )

        # Convert GLT matrices to tensors on device for fast computation
        glt_contrasts_list = []
        for glt_mat in glt_matrices:
            glt_tensor = torch.as_tensor(glt_mat, dtype=dtype, device=device)
            # Check if this is a single-row contrast (t-test) or multi-row (F-test)
            if glt_tensor.ndim == 1:
                # Already 1D - single-row contrast (t-test)
                glt_contrasts_list.append(glt_tensor)
            elif glt_tensor.ndim == 2 and glt_tensor.shape[0] == 1:
                # Shape (1, n_regressors) - squeeze to 1D for single-row contrast
                glt_contrasts_list.append(glt_tensor.squeeze(0))
            else:
                # Multi-row contrast (F-test) - not yet supported
                raise NotImplementedError(
                    f"Multi-row GLT contrasts (F-tests) not yet supported. "
                    f"Got shape {glt_tensor.shape}, expected (n_regressors,) or (1, n_regressors)"
                )
        glt_contrasts_tensor = torch.stack(glt_contrasts_list)  # (n_contrasts, n_regressors_full)

        # NOTE: GLT contrasts use the FULL design (all 131 columns)
        # We fit the full design, so contrasts can involve any regressor
        # No filtering needed here!

    results.dof = max(1, raw_dof)
    results.tr = tr

    if want_residuals:
        results.residuals = torch.zeros(n_voxels, n_timepoints, device=storage_device, dtype=dtype)
        results.residuals_whitened = torch.zeros(
            n_voxels, n_timepoints, device=storage_device, dtype=dtype
        )
    if want_predicted:
        results.predicted = torch.zeros(n_voxels, n_timepoints, device=storage_device, dtype=dtype)

    # Ljung-Box: one scalar per voxel, reduced inside the GLS loop from residuals
    # that are already resident, so -Rvar never has to pay for whole-brain
    # whitened residuals (see [[Memory module]]).
    lj_max_lag: int | None = None
    lj_tau: torch.Tensor | None = None
    if want_ljung_box:
        _lj_starts = sorted(
            {int(s) for s in (run_starts.tolist() if torch.is_tensor(run_starts) else run_starts)}
            if run_starts is not None
            else {0}
        )
        _min_run = (
            min(b - a for a, b in zip(_lj_starts, _lj_starts[1:] + [n_timepoints], strict=True))
            if len(_lj_starts) > 1
            else n_timepoints
        )
        lj_max_lag = ljung_box_max_lag(n_timepoints, n_regressors, _min_run)
        lj_tau = build_ljung_box_tau(n_timepoints, run_starts, tau, device=device)
        results.ljung_box = torch.zeros(n_voxels, device="cpu", dtype=torch.float32)
        results.ljung_box_dof = max(1, _resolve_ljung_box_lag(n_timepoints, lj_max_lag) - 2)

    # OLS baseline fit (if requested)
    if want_ols:
        if verbose:
            print("\nComputing OLS baseline for comparison...")
        from .core import fit_glm

        # OLS chunk size: let fit_glm auto-estimate from available GPU memory.
        # OLS is much lighter than ARMA (no prewhitening matrices), so it should
        # get a large chunk size independently of the ARMA batch size.
        ols_chunk_size = None  # auto-estimated inside fit_glm

        # Set preload_data_to_device=False to stream chunks from CPU
        # CRITICAL: Pass max_poly_degree=-1 to prevent adding ANY polynomials (including constant)
        # The design matrix from AFNI already includes all polynomials and intercept!
        # NOTE: task_indices filtering has already been applied to design matrix above (line 2369-2370)
        results.ols_results = fit_glm(
            data,
            design,
            tr=tr,
            device=device,
            chunk_size=ols_chunk_size,
            verbose=verbose,  # Show progress bar for OLS chunks
            preload_data_to_device=False,
            use_double=use_double,
            want_residuals=want_ols_residuals,  # OLS errts (-Oerrts)
            want_predicted=want_ols_predicted,  # OLS fitts (-Ofitts)
            max_poly_degree=-1,  # Design matrix is complete - don't add ANYTHING!
            want_r2_partial=want_r2_partial,  # Compute partial R² for OLS if requested
            r2_partial_mode=r2_partial_mode,  # "full" or "task" mode
            want_r2_semipartial=want_r2_semipartial,  # Compute semi-partial R² for OLS if requested
            r2_semipartial_mode=r2_semipartial_mode,  # "full" or "task" mode
            glt_labels=glt_labels,
            glt_matrices=glt_matrices,
            task_indices=fitted_column_indices,  # Extract these columns for output
            debug_memory=debug_memory,
        )
        if verbose:
            print("✓ OLS fit complete")

        # Set spatial metadata on OLS results BEFORE callback (critical for masking!)
        if spatial_metadata is not None:
            if "volume_shape" in spatial_metadata:
                results.ols_results.original_shape = spatial_metadata["volume_shape"]
                results.ols_results.full_shape = spatial_metadata["volume_shape"]
            if "voxel_mask" in spatial_metadata:
                results.ols_results.voxel_mask = spatial_metadata["voxel_mask"]
            if "affine" in spatial_metadata:
                results.ols_results.affine = spatial_metadata["affine"]

        # Write OLS results immediately if callback provided (frees memory!)
        if ols_write_callback is not None:
            if verbose:
                print("  Writing OLS results to disk...")
            # Pass OLS results plus spatial metadata to callback
            ols_write_callback(
                results.ols_results,
                original_shape=original_shape,
                affine=getattr(results, "affine", None),
            )
            # Clear OLS results from memory to free RAM/GPU
            results.ols_results = None

            if verbose:
                print("  ✓ OLS results written and cleared from memory")

        # CRITICAL: Move OLS results to CPU storage before grid precomputation
        # Grid precomputation needs GPU memory, and OLS results can be huge!
        elif results.ols_results is not None:
            if verbose:
                print("  Moving OLS results to CPU to free GPU memory...")
            # Move all OLS result tensors from GPU to CPU
            ols_cpu = type(results.ols_results)()  # Create new results object of same type
            for attr in dir(results.ols_results):
                if not attr.startswith("_"):
                    val = getattr(results.ols_results, attr)
                    if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                        setattr(ols_cpu, attr, val.cpu())
                    else:
                        setattr(ols_cpu, attr, val)
            results.ols_results = ols_cpu

            if verbose:
                print("  ✓ OLS results moved to CPU")

        # Explicitly free GPU memory (critical before ARMA grid precomputation!)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if verbose:
            print()

    if estimate_per_voxel:
        # Per-voxel ARMA estimation (most accurate)
        n_batches = (n_voxels + batch_size - 1) // batch_size

        if verbose:
            print(f"Processing {n_batches} batches of {batch_size:,} voxels...")

        # Check if using precomputed ARMA parameters
        use_precomputed_arma = precomputed_arma_params is not None

        if use_precomputed_arma:
            # Validate precomputed parameters
            precomputed_arma_params = to_tensor(precomputed_arma_params, device=storage_device)
            if precomputed_arma_params.shape != (n_voxels, 2):
                raise ValueError(
                    f"precomputed_arma_params must have shape ({n_voxels}, 2), "
                    f"got {precomputed_arma_params.shape}"
                )

            if verbose:
                print("✓ Using precomputed ARMA parameters (skipping REML estimation)")
                print(f"  Mean a: {precomputed_arma_params[:, 0].mean():.3f}")
                print(f"  Mean b: {precomputed_arma_params[:, 1].mean():.3f}")
                print("  This will save ~80% of compute time!\n")

            # Store precomputed parameters
            results.arma_params = precomputed_arma_params

            # Compute λ for each voxel
            a_vals = precomputed_arma_params[:, 0]
            b_vals = precomputed_arma_params[:, 1]
            lambda_vals = (
                (b_vals + a_vals)
                * (1 + a_vals * b_vals)
                / (1 + 2 * a_vals * b_vals + b_vals**2 + 1e-10)
            )
            results.arma_lambda = lambda_vals

            # Build precomputed grid from unique (a,b) pairs for efficient GLS
            if verbose:
                print("Building Cholesky cache from precomputed parameters...")

            unique_params = torch.unique(precomputed_arma_params, dim=0)
            precomputed_grid = {}

            # Free GPU memory before building cache (critical for large datasets!)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            for params in unique_params:
                a_opt, b_opt = params[0].item(), params[1].item()
                X_w, _, L = prewhiten_with_arma11(
                    design, design[:, 0], a_opt, b_opt, run_starts=run_starts, tau=tau
                )
                # CRITICAL: Store L on CPU to save GPU memory (1.1 GB per matrix for 17k timepoints!)
                # Move to GPU only when needed for batch computation
                precomputed_grid[(a_opt, b_opt)] = {
                    "X_w": X_w.cpu(),  # Store on CPU
                    "L": L.cpu(),  # Store on CPU - lower Cholesky factor
                    "a": a_opt,
                    "b": b_opt,
                }

            if verbose:
                print(
                    f"✓ Built cache for {len(precomputed_grid)} unique (a,b) pairs (stored on CPU)\n"
                )

        else:
            # Use provided grids or defaults
            if a_grid is None and b_grid is None:
                a_grid, b_grid = get_default_arma_grids(device)
            else:
                if a_grid is None:
                    a_grid, _ = get_default_arma_grids(device)
                else:
                    a_grid = to_tensor(a_grid, device=device)

                if b_grid is None:
                    _, b_grid = get_default_arma_grids(device)
                else:
                    b_grid = to_tensor(b_grid, device=device)

            if use_grid_batching:
                # ADAPTIVE BATCHING MODE: Intelligently choose strategy

                # Estimate grid size
                n_grid_estimate = len(a_grid) * len(b_grid)

                # Determine optimal strategy
                strategy = determine_adaptive_batching_strategy(
                    n_voxels=n_voxels,
                    n_timepoints=n_timepoints,
                    n_regressors=n_regressors,
                    n_grid_points=n_grid_estimate,
                    device=device,
                    use_double=use_double,
                    verbose=verbose,
                )

                # Run grid search with adaptive batching
                best_params, best_likelihood = reml_grid_search_batched(
                    data,
                    design,
                    a_grid,
                    b_grid,
                    device,
                    verbose=verbose,
                    dtype=dtype,
                    grid_chunk_size=strategy["grid_chunk_size"],
                    voxel_batch_size=strategy["voxel_batch_size"],
                    y_scale=_y_norm_scale,
                    run_starts=run_starts,
                    tau=tau,
                )

                # Store results
                results.arma_params[:] = best_params
                results.reml_likelihood[:] = best_likelihood

                # Compute lambda from (a, b)
                a_vals = best_params[:, 0]
                b_vals = best_params[:, 1]
                lam = (
                    (b_vals + a_vals)
                    * (1 + a_vals * b_vals)
                    / (1 + 2 * a_vals * b_vals + b_vals**2 + 1e-10)
                )
                results.arma_lambda[:] = lam

                if verbose:
                    print("\n✓ Grid search complete!")
                    print(f"  Mean (a, b): ({a_vals.mean():.3f}, {b_vals.mean():.3f})")
                    print(f"  Mean λ: {lam.mean():.3f}\n")

                # No precomputed grid for batching mode
                precomputed_grid = None

            else:
                # PRE-COMPUTE REML GRID (KEY OPTIMIZATION!)
                if verbose:
                    print("Pre-computing REML grid (Cholesky factorizations)...")

                # Clear GPU cache before grid precomputation (critical for memory!)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # SMART CHOLESKY SELECTION: Use GPU when memory allows
                # GPU is 10-20× faster but needs more memory
                use_gpu_cholesky = False
                if device.type == "cuda" and cholesky_on_cpu:
                    # Estimate PEAK GPU memory during batch Cholesky
                    # (R_batch + L_batch + L_inv_batch + X_w_batch + R_qr_batch + workspace ≈ 6×)
                    n_grid_estimate = len(a_grid) * len(b_grid)  # upper bound before λ filter
                    bytes_per_matrix = n_timepoints * n_timepoints * 4  # float32
                    cholesky_peak_gb = (bytes_per_matrix * n_grid_estimate * 6) / 1e9

                    # Get available GPU memory
                    gpu_props = torch.cuda.get_device_properties(device)
                    total_gb = gpu_props.total_memory / 1e9
                    # Reserve 2GB for other operations
                    available_gb = total_gb - 2.0

                    # Use GPU Cholesky only if peak workspace fits
                    if cholesky_peak_gb < (available_gb * 0.7):
                        use_gpu_cholesky = True
                        if verbose:
                            print(
                                f"  🚀 Using GPU Cholesky (peak workspace: {cholesky_peak_gb:.1f}GB < {available_gb:.1f}GB available)"
                            )
                            print("     Expected 10-20× speedup over CPU!")
                    else:
                        if verbose:
                            print(
                                f"  Using CPU Cholesky (Cholesky peak: {cholesky_peak_gb:.1f}GB > {available_gb:.1f}GB available)"
                            )
                            print(
                                f"  (Stored grid ~{cholesky_peak_gb / 6:.1f}GB will be loaded to GPU after computation)"
                            )

                # Override cholesky_on_cpu if GPU is better
                actual_cholesky_on_cpu = cholesky_on_cpu and not use_gpu_cholesky

                precomputed_grid = precompute_reml_grid(
                    design,
                    n_timepoints,
                    a_grid,
                    b_grid,
                    device,
                    verbose=verbose,
                    cholesky_on_cpu=actual_cholesky_on_cpu,
                    dtype=dtype,
                    debug_memory=debug_memory,
                    use_qr=use_qr,
                    run_starts=run_starts,
                    autocorr_cache=autocorr_cache,
                    tau=tau,
                )

                # Type assertion: grids are guaranteed to be set by now
                assert a_grid is not None and b_grid is not None

                if verbose:
                    print(f"Precomputed {len(precomputed_grid)} valid (a,b) pairs")
                    print(f"  Grid: {len(a_grid)} a values × {len(b_grid)} b values")

                if debug_memory:
                    sample_key = list(precomputed_grid.keys())[0]
                    _debug_memory_snapshot(
                        "AFTER grid precomputation (on CPU)",
                        device,
                        {
                            "sample_L_inv": precomputed_grid[sample_key]["L_inv"],
                            "sample_X_w": precomputed_grid[sample_key]["X_w"],
                            "sample_Q": precomputed_grid[sample_key]["Q"],
                        },
                    )

                # CRITICAL: Move entire grid to GPU ONCE (not per-batch!)
                if device.type == "cuda":
                    if verbose:
                        print("  Loading grid to GPU (one-time cost)...")
                    for key in precomputed_grid:
                        precomputed_grid[key]["L_inv"] = precomputed_grid[key]["L_inv"].to(device)
                        precomputed_grid[key]["X_w"] = precomputed_grid[key]["X_w"].to(device)
                        precomputed_grid[key]["Q"] = precomputed_grid[key]["Q"].to(device)
                        precomputed_grid[key]["logdet_Rcorr"] = precomputed_grid[key][
                            "logdet_Rcorr"
                        ].to(device)
                        precomputed_grid[key]["logdet_XwTXw"] = precomputed_grid[key][
                            "logdet_XwTXw"
                        ].to(device)

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    if verbose:
                        print(
                            f"  ✓ Grid loaded to GPU (will be reused for all {n_batches} batches)\n"
                        )

                    if debug_memory:
                        sample_key = list(precomputed_grid.keys())[0]
                        _debug_memory_snapshot(
                            "AFTER grid loaded to GPU",
                            device,
                            {
                                "sample_L_inv": precomputed_grid[sample_key]["L_inv"],
                                "sample_X_w": precomputed_grid[sample_key]["X_w"],
                                "sample_Q": precomputed_grid[sample_key]["Q"],
                            },
                        )

                # CRITICAL FIX: Search each voxel against precomputed grid
                # This was missing, causing all voxels to stay at initialization (0,0)
                # Process voxels in batches to find optimal parameters
                batch_iter = range(n_batches)
                if verbose:
                    batch_iter = tqdm(batch_iter, desc="🔍 ARMA grid search", unit="batch")

                # One-time surface accumulator setup: (n_voxels, n_valid_pairs) on CPU
                _surface_accum: torch.Tensor | None = None
                _surface_params: list | None = None

                _arma_grid_dbg = make_vram_debugger(
                    device,
                    batch_size
                    * bytes_per_voxel_arma(n_timepoints, n_regressors)
                    * (dtype.itemsize // 4),
                    operation="arma_grid_search",
                    chunk_size=batch_size,
                    enabled=debug_memory,
                )
                _arma_grid_dbg.__enter__()
                for batch_idx in batch_iter:
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_voxels)

                    # Get voxel data for this batch
                    Y_batch = data[batch_start:batch_end]  # (batch_voxels, n_timepoints)
                    if _y_norm_scale is not None:
                        Y_batch = Y_batch / _y_norm_scale[batch_start:batch_end].unsqueeze(1)

                    # Search against precomputed grid. AFNI's hierarchical
                    # descent (3dREMLfit's REML_find_best_case) is the default
                    # on BOTH devices — the exhaustive global argmin lands on a
                    # different point along the degenerate ARMA(1,1) a≈b ridge
                    # than AFNI, hurting AFNI parity (worst on the omnibus F).
                    # CPU runs the work-eliminating subset-GEMM descent; GPU
                    # builds the full surface (cheap, same cost as exhaustive)
                    # then descends via vectorised indexing, dodging the
                    # per-grid-point sync storm that throttled the subset path
                    # on CUDA. -exhaustive forces the global argmin on either.
                    _use_hierarchical = not force_exhaustive_search
                    if _use_hierarchical and device.type == "cpu" and not save_profile_likelihoods:
                        best_params_batch, best_lik_batch = (
                            search_voxels_precomputed_grid_hierarchical(
                                design,
                                Y_batch,
                                precomputed_grid,
                                device,
                                verbose=verbose and (batch_idx == 0),
                            )
                        )
                        search_result = (best_params_batch, best_lik_batch)
                    else:
                        # Surface needed for: -Rlklhd save, or GPU hierarchical.
                        _need_profile = save_profile_likelihoods or (
                            _use_hierarchical and device.type != "cpu"
                        )
                        _res = search_voxels_precomputed_grid(
                            design,
                            Y_batch,
                            precomputed_grid,
                            device,
                            verbose=verbose and (batch_idx == 0),
                            return_profile=_need_profile,
                        )
                        if _need_profile:
                            _bp_exh, _bl_exh, surface_batch, surf_params = _res
                            if _use_hierarchical:
                                best_params_batch, best_lik_batch = (
                                    select_arma_params_hierarchical_from_surface(
                                        surface_batch, surf_params, device
                                    )
                                )
                            else:
                                best_params_batch, best_lik_batch = _bp_exh, _bl_exh
                            if save_profile_likelihoods:
                                search_result = (
                                    best_params_batch,
                                    best_lik_batch,
                                    surface_batch,
                                    surf_params,
                                )
                            else:
                                search_result = (best_params_batch, best_lik_batch)
                        else:
                            search_result = _res

                    if save_profile_likelihoods:
                        best_params_batch, best_lik_batch, surface_batch, surf_params = (
                            search_result
                        )
                        # Allocate full surface array on first batch
                        if _surface_accum is None:
                            _surface_params = surf_params
                            _surface_accum = torch.empty(
                                n_voxels, len(surf_params), dtype=surface_batch.dtype
                            )
                        _surface_accum[batch_start:batch_end] = surface_batch.cpu()
                    else:
                        best_params_batch, best_lik_batch = search_result

                    # Store results
                    results.arma_params[batch_start:batch_end] = best_params_batch.cpu()
                    results.reml_likelihood[batch_start:batch_end] = best_lik_batch.cpu()
                _arma_grid_dbg.__exit__(None, None, None)

                # Store likelihood surface in results if computed
                if save_profile_likelihoods and _surface_accum is not None:
                    results.reml_lklhd_surface = _surface_accum
                    results.reml_surface_params = _surface_params

                # Compute lambda from (a, b)
                a_vals = results.arma_params[:, 0]
                b_vals = results.arma_params[:, 1]
                lam = (
                    (b_vals + a_vals)
                    * (1 + a_vals * b_vals)
                    / (1 + 2 * a_vals * b_vals + b_vals**2 + 1e-10)
                )
                results.arma_lambda = lam

                if verbose:
                    print("\n✓ Grid search complete!")
                    print(f"  Mean (a, b): ({a_vals.mean():.3f}, {b_vals.mean():.3f})")
                    print(f"  Mean λ: {lam.mean():.3f}\n")

        # NEW OPTIMIZED APPROACH: Group voxels by (a,b), process each group together
        # Eliminates CPU<->GPU transfers of L (saves ~87 minutes on large datasets!)
        # Compute each L once, keep on GPU, process all voxels with that (a,b), then delete
        if verbose:
            print("\n📦 Grouping voxels by optimal ARMA parameters...")

        # Vectorized grouping by optimal (a,b). The Python loop with `.item()`
        # per voxel is ~5 µs/voxel — adds up to seconds on a few-hundred-k voxel
        # set, and we pay it once per HRF in per-voxel HRF mode. torch.unique
        # does it in one tensor op.
        ab_cpu = results.arma_params.cpu().contiguous()
        unique_pairs, inverse_indices = torch.unique(ab_cpu, dim=0, return_inverse=True)
        # Sort voxel indices by group to find per-group slice boundaries fast.
        sort_keys, sort_perm = torch.sort(inverse_indices, stable=True)
        # Group boundaries: where the group id changes.
        group_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                (sort_keys[1:] != sort_keys[:-1]).nonzero(as_tuple=True)[0] + 1,
                torch.tensor([len(sort_keys)], dtype=torch.long),
            ]
        )
        voxel_groups: dict[tuple[float, float], list[int]] = {}
        for g_idx in range(len(unique_pairs)):
            a_v = float(unique_pairs[g_idx, 0])
            b_v = float(unique_pairs[g_idx, 1])
            members = sort_perm[group_starts[g_idx] : group_starts[g_idx + 1]].tolist()
            voxel_groups[(a_v, b_v)] = members

        n_unique_pairs = len(voxel_groups)
        total_voxels_in_groups = sum(len(indices) for indices in voxel_groups.values())

        if verbose:
            print(f"  Found {n_unique_pairs} unique (a,b) pairs")
            print(f"  Voxels to process: {total_voxels_in_groups:,}")
            largest_group = max(len(indices) for indices in voxel_groups.values())
            print(f"  Largest group: {largest_group:,} voxels\n")

        # Progress bar tracks unique (a,b) groups
        if verbose and n_unique_pairs > 1:
            pbar = tqdm(total=n_unique_pairs, desc="ARMA(1,1) groups", unit="group")
        else:
            pbar = None

        # Process each (a,b) group
        for group_idx, ((a_opt, b_opt), voxel_indices) in enumerate(voxel_groups.items()):
            n_group_voxels = len(voxel_indices)

            # Get L and X_w for this (a,b) pair
            # If precomputed_grid exists (from precomputed ARMA params or non-batched grid search),
            # reuse cached values. Otherwise compute on-demand.
            if use_precomputed_arma and (a_opt, b_opt) in precomputed_grid:
                # Reuse the cached Cholesky factor L and X_w (stored on CPU to save
                # GPU memory); move to GPU only when needed for this group. The cache
                # holds L (not its inverse), so whiten via triangular solve.
                L_chol = precomputed_grid[(a_opt, b_opt)]["L"].to(device)
                X_w = precomputed_grid[(a_opt, b_opt)]["X_w"].to(device)
                _using_l_inv = False  # L_chol is the lower-triangular Cholesky factor
            else:
                # Compute L ONCE for this (a,b) group, keep on GPU
                y_dummy = data[voxel_indices[0]].to(device)
                X_w, _, L_chol = prewhiten_with_arma11(
                    design, y_dummy, a_opt, b_opt, run_starts=run_starts, tau=tau
                )
                _using_l_inv = False  # L_chol is actual Cholesky factor here

            # OPTIMIZATION: Precompute group-level matrices ONCE per (a,b)
            # X_w is the same for all voxels in this group, so derivatives are constant
            import time

            t_qr_start = time.time()
            if use_qr:
                # QR path: Precompute Q and R
                Q_group, R_qr_group = torch.linalg.qr(X_w)  # Q: (n_time, n_reg), R: (n_reg, n_reg)
                # Precompute R^{-1} for variance computation (same for all voxels)
                eye = torch.eye(n_regressors, device=device, dtype=dtype)
                R_inv_group = torch.linalg.solve_triangular(R_qr_group, eye, upper=True)
                XwTXw_inv_group = R_inv_group @ R_inv_group.T  # (R'R)^{-1}
                del eye
            else:
                # X'X path: Precompute X'X and its inverse ONCE per group (HUGE SAVINGS!)
                # This is the SAME for all voxels in the group - no need to recompute per batch!
                XwTXw_group = X_w.T @ X_w  # (n_reg, n_reg) - only 12.5 MB for 1771 regressors!

                try:
                    L_group = torch.linalg.cholesky(XwTXw_group)
                    XwTXw_inv_group = torch.cholesky_inverse(L_group)
                    cholesky_success_group = True
                    del L_group
                except torch.linalg.LinAlgError:
                    XwTXw_inv_group = torch.linalg.inv(XwTXw_group)
                    cholesky_success_group = False

                # For diagonal-only path (no GLTs), extract diagonal once
                if glt_contrasts_tensor is None:
                    XwTXw_inv_diag_group = torch.diagonal(XwTXw_inv_group)

            # Group-level F-stat precomputes — eliminate the per-batch
            # (batch, n_reg, n_reg) `var_beta_batch` allocation + `linalg.inv`.
            # Since var_beta = σ² · A with A = XwTXw_inv constant per group,
            # F = β_task' (σ² A_task)^{-1} β_task / p
            #   = (β_task' A_task^{-1} β_task) / (σ² · p)
            # A_task^{-1} is constant per group → compute ONCE.
            if use_qr:
                # QR path didn't have a diag precompute; add it for se_beta path.
                XwTXw_inv_diag_group = torch.diagonal(XwTXw_inv_group)
            if fitted_column_indices is not None:
                _fci_tensor = torch.as_tensor(
                    fitted_column_indices, device=device, dtype=torch.long
                )
                _A_task = XwTXw_inv_group.index_select(0, _fci_tensor).index_select(1, _fci_tensor)
                _eye_task = torch.eye(len(fitted_column_indices), device=device, dtype=dtype)
                A_task_inv_group = torch.linalg.inv(_A_task + 1e-8 * _eye_task)
                del _A_task, _eye_task
                A_full_inv_group = None
            else:
                # Full-model F: inverse of XwTXw_inv == X_w'X_w (already at hand)
                if use_qr:
                    A_full_inv_group = R_qr_group.T @ R_qr_group
                else:
                    A_full_inv_group = XwTXw_group  # already computed above

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_qr = time.time() - t_qr_start

            # Timing accumulators for this group
            t_prewhiten_total = 0.0
            t_qr_solve_total = 0.0
            t_stats_total = 0.0
            t_glt_total = 0.0

            # Process voxels in this group using sub-batches (memory-aware)
            # Determine sub-batch size based on available GPU memory
            # CRITICAL: Pass persistent tensors that stay on GPU (L_chol, X_w, design, etc)
            persistent_gpu_tensors = {
                "L_chol": L_chol,
                "X_w": X_w,
                "design": design if design.device.type == "cuda" else None,
            }
            if use_qr:
                persistent_gpu_tensors["Q_group"] = Q_group
                persistent_gpu_tensors["R_qr_group"] = R_qr_group
                persistent_gpu_tensors["XwTXw_inv_group"] = XwTXw_inv_group

            sub_batch_size = check_cuda_memory_before_batch(
                batch_size,  # Start with original batch size
                n_timepoints,
                n_regressors,
                device,
                verbose=(verbose and group_idx == 0),  # Only warn for first group
                use_qr=use_qr,
                has_glts=(glt_contrasts_tensor is not None),
                persistent_tensors=persistent_gpu_tensors,
                use_double=use_double,
            )

            n_sub_batches = (n_group_voxels + sub_batch_size - 1) // sub_batch_size

            # OPTIMIZATION: Compute λ ONCE per group (same for all sub-batches)
            a_val = torch.tensor(a_opt, device=device, dtype=dtype)
            b_val = torch.tensor(b_opt, device=device, dtype=dtype)
            lambda_val = (
                (b_val + a_val) * (1 + a_val * b_val) / (1 + 2 * a_val * b_val + b_val**2 + 1e-10)
            )
            # Convert to CPU scalar once for reuse
            lambda_val_cpu = lambda_val.cpu().item()

            # OPTIMIZATION: Create identity matrix ONCE per group (reused for diagonal inverse)
            # Only needed if no GLTs (diagonal-only path uses this)
            if glt_contrasts_tensor is None and not use_qr:
                eye_group = torch.eye(n_regressors, device=device, dtype=dtype)

            # Process each sub-batch within this (a,b) group
            _gls_dbg = make_vram_debugger(
                device,
                sub_batch_size
                * bytes_per_voxel_arma(n_timepoints, n_regressors)
                * (dtype.itemsize // 4),
                operation="gls_fitting",
                chunk_size=sub_batch_size,
                enabled=debug_memory and group_idx == 0,  # only report for first group
            )
            _gls_dbg.__enter__()
            for sub_batch_idx in range(n_sub_batches):
                sub_start = sub_batch_idx * sub_batch_size
                sub_end = min(sub_start + sub_batch_size, n_group_voxels)
                sub_voxel_indices = voxel_indices[sub_start:sub_end]
                batch_voxels = len(sub_voxel_indices)

                # Load data for these specific voxels (non-contiguous indexing)
                Y_batch = data[sub_voxel_indices].T  # (n_timepoints, batch_voxels)
                Y_batch_dev = Y_batch.to(device)
                if _y_norm_scale is not None:
                    Y_batch_dev = Y_batch_dev / _y_norm_scale[sub_voxel_indices].to(
                        device
                    ).unsqueeze(0)

                # Store λ for these voxels (pre-computed above, reuse for all sub-batches)
                results.arma_lambda[sub_voxel_indices] = lambda_val_cpu

                # Prewhiten data (L_chol on GPU from group computation)
                # All voxels in this sub-batch have the same (a,b), so same whitening
                t_pw_start = time.time()
                if _using_l_inv:
                    # Precomputed path: L_chol is L_inv — use GEMM (faster on GPU)
                    y_w_batch = (L_chol @ Y_batch_dev).T  # (batch_voxels, n_timepoints)
                else:
                    # On-demand path: L_chol is lower triangular — triangular solve
                    y_w_batch = torch.linalg.solve_triangular(
                        L_chol, Y_batch_dev, upper=False
                    ).T  # (batch_voxels, n_timepoints)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_prewhiten_total += time.time() - t_pw_start

                # X_w is constant for all voxels (same design, same (a,b))
                # Expand to batch dimension (creates view, no memory copy!)
                X_w_batch = X_w.unsqueeze(0).expand(batch_voxels, -1, -1)

                # Batch GLS solve
                # X_w_batch: (batch_voxels, n_timepoints, n_regressors)
                # y_w_batch: (batch_voxels, n_timepoints)

                if use_qr:
                    # OPTIMIZED: Use precomputed Q and R from group level
                    # Q_group: (n_time, n_reg), R_qr_group: (n_reg, n_reg)
                    # These are the same for all voxels in this (a,b) group

                    t_solve_start = time.time()
                    # Compute Q'y for each voxel: (n_reg, n_time) @ (n_time, batch) = (n_reg, batch)
                    QTy_batch = (Q_group.T @ y_w_batch.T).T  # (batch_voxels, n_regressors)

                    # Solve R β = Q'y using triangular solve
                    # Expand R_qr_group to batch for solve_triangular
                    R_qr_batch = R_qr_group.unsqueeze(0).expand(batch_voxels, -1, -1)
                    betas_batch = torch.linalg.solve_triangular(
                        R_qr_batch, QTy_batch.unsqueeze(2), upper=True
                    ).squeeze(2)  # (batch_voxels, n_regressors)
                    del QTy_batch, R_qr_batch
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_qr_solve_total += time.time() - t_solve_start

                    # Statistics computation using precomputed matrices
                    t_stats_start = time.time()
                    # pred_w = X_w @ beta for each voxel
                    pred_w_batch = (X_w @ betas_batch.T).T  # (batch_voxels, n_timepoints)
                    resid_w_batch = y_w_batch - pred_w_batch
                    del X_w_batch

                    pred_orig_batch = torch.mm(design, betas_batch.T).T
                    resid_orig_batch = Y_batch_dev.T - pred_orig_batch

                    df = results.dof
                    # Double-precision accumulation for RSS: eliminates rounding in sum-of-squares
                    # fp64 accumulator avoids RSS rounding without materializing a fp64 copy
                    sigma2_batch = (resid_w_batch.pow(2).sum(dim=1, dtype=_accum_dtype) / df).to(
                        dtype
                    )

                    if want_ljung_box:
                        results.ljung_box[sub_voxel_indices] = (
                            _ljung_box_batched(resid_w_batch, lj_max_lag, lj_tau)
                            .to(torch.float32)
                            .cpu()
                        )

                    if not want_residuals:
                        del resid_w_batch

                    # se_beta from diagonal — broadcast σ² · diag(A), no (batch,n_reg,n_reg) tensor
                    se_beta_batch = torch.sqrt(
                        sigma2_batch.unsqueeze(1) * XwTXw_inv_diag_group.unsqueeze(0)
                    )
                    # Only materialize var_beta_batch when GLTs need it (legacy contrast path)
                    if glt_contrasts_tensor is not None:
                        XwTXw_inv_batch = XwTXw_inv_group.unsqueeze(0).expand(batch_voxels, -1, -1)
                        var_beta_batch = sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
                    else:
                        var_beta_batch = None
                    tstats_batch = betas_batch / (se_beta_batch + 1e-10)
                    del se_beta_batch
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_stats_total += time.time() - t_stats_start

                    # Compute R² now (before partial/semi-partial, which need it)
                    # Y_batch_dev is (n_timepoints, batch_voxels), pred_orig_batch is (batch_voxels, n_timepoints)
                    r2_batch = compute_r2_metric(Y_batch_dev.T, pred_orig_batch, metric="cod")

                    # Partial R² per regressor: r²_partial_i = t²_i / (t²_i + df)
                    if want_r2_partial:
                        df = results.dof
                        t_squared_batch = tstats_batch**2
                        r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
                        del t_squared_batch

                        # Split into task vs nuisance, rescale if mode='task'
                        if fitted_column_indices is not None:
                            task_indices_set = set(fitted_column_indices)
                            all_indices = set(range(n_regressors))
                            nuisance_indices = sorted(list(all_indices - task_indices_set))

                            r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                            # Always extract nuisance for storage (used for -bout output)
                            r2_partial_nuisance_batch = (
                                r2_partial_full_batch[:, nuisance_indices]
                                if len(nuisance_indices) > 0
                                else None
                            )

                            if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                                # nuisance_indices non-empty here => set to non-None above
                                assert r2_partial_nuisance_batch is not None
                                # Rescale task partial R² by variance remaining after nuisance
                                r2_nuisance_total = r2_partial_nuisance_batch.sum(
                                    dim=1, keepdim=True
                                )
                                denominator = torch.clamp(1.0 - r2_nuisance_total, min=0.01)
                                r2_partial_batch = r2_partial_task_batch / denominator
                            else:
                                r2_partial_batch = r2_partial_task_batch
                        else:
                            # No filtering - use all
                            r2_partial_batch = r2_partial_full_batch
                            r2_partial_nuisance_batch = None

                    # Semi-partial R² per regressor: r²_semi_i = partial_r²_i * (1 - R²_full)
                    if want_r2_semipartial:
                        # r2_batch is float32 (from compute_r2_metric); cast to dtype for float64 safety
                        variance_remaining = torch.clamp(
                            1.0 - r2_batch.to(dtype).unsqueeze(1), min=0.0
                        )

                        # If we didn't compute partial R² above, compute it now for semi-partial
                        if not want_r2_partial:
                            df = results.dof
                            t_squared_batch = tstats_batch**2
                            r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)

                            # Split into task vs nuisance
                            if fitted_column_indices is not None:
                                task_indices_set = set(fitted_column_indices)
                                all_indices = set(range(n_regressors))
                                nuisance_indices = sorted(list(all_indices - task_indices_set))

                                r2_partial_task_batch = r2_partial_full_batch[
                                    :, fitted_column_indices
                                ]
                                r2_partial_nuisance_batch = (
                                    r2_partial_full_batch[:, nuisance_indices]
                                    if len(nuisance_indices) > 0
                                    else None
                                )
                            else:
                                r2_partial_task_batch = r2_partial_full_batch
                                r2_partial_nuisance_batch = None

                        # Compute semi-partial R² from partial R²
                        r2_semipartial_task_batch = r2_partial_task_batch * variance_remaining

                        # Nuisance semi-partial R²
                        r2_semipartial_nuisance_batch = (
                            r2_partial_nuisance_batch * variance_remaining
                            if r2_partial_nuisance_batch is not None
                            else None
                        )

                        # Apply rescaling mode for task regressors
                        if (
                            r2_semipartial_mode == "task"
                            and r2_semipartial_nuisance_batch is not None
                        ):
                            # Rescale by variance remaining after nuisance
                            r2_semi_nuisance_total = r2_semipartial_nuisance_batch.sum(
                                dim=1, keepdim=True
                            )
                            denominator = torch.clamp(1.0 - r2_semi_nuisance_total, min=0.01)
                            r2_semipartial_batch = r2_semipartial_task_batch / denominator
                        else:
                            # Full mode: use raw semi-partial R² values
                            r2_semipartial_batch = r2_semipartial_task_batch

                    # F-stat: F = (β' M β) / (σ² · p)  with M = A_task^{-1} (or A_full^{-1})
                    # constant per group → no (batch, n_reg, n_reg) tensor, no per-batch inv.
                    if fitted_column_indices is not None:
                        betas_task_batch = betas_batch[:, fitted_column_indices]
                        n_task_params = len(fitted_column_indices)
                        quad_batch = torch.einsum(
                            "bi,ij,bj->b",
                            betas_task_batch,
                            A_task_inv_group,
                            betas_task_batch,
                        )
                        fstats_batch = quad_batch / (sigma2_batch * n_task_params + 1e-10)
                        del quad_batch, betas_task_batch
                    else:
                        quad_batch = torch.einsum(
                            "bi,ij,bj->b", betas_batch, A_full_inv_group, betas_batch
                        )
                        fstats_batch = quad_batch / (sigma2_batch * n_regressors + 1e-10)
                        del quad_batch

                    # GLT CONTRASTS (QR path): Compute in-loop
                    if glt_contrasts_tensor is not None:
                        t_glt_start = time.time()
                        contrast_betas_batch_qr = torch.mm(betas_batch, glt_contrasts_tensor.T)

                        if legacy_contrasts:
                            # LEGACY: Loop-based computation (slow, for validation only)
                            contrast_vars_batch_qr = torch.zeros(
                                batch_voxels, n_contrasts, device=device, dtype=dtype
                            )
                            for c_idx in range(n_contrasts):
                                c = glt_contrasts_tensor[c_idx]
                                c_var = torch.bmm(
                                    c.unsqueeze(0).unsqueeze(1).expand(batch_voxels, 1, -1),
                                    var_beta_batch,
                                )
                                contrast_vars_batch_qr[:, c_idx] = torch.bmm(
                                    c_var,
                                    c.unsqueeze(0).unsqueeze(2).expand(batch_voxels, -1, 1),
                                ).squeeze()
                        else:
                            # OPTIMIZED: Vectorized einsum (10-50x faster!)
                            # Compute Var(c'β) = c' Var(β) c for all contrasts at once
                            # glt_contrasts_tensor: (n_contrasts, n_regressors)
                            # var_beta_batch: (batch_voxels, n_regressors, n_regressors)
                            # Result: (batch_voxels, n_contrasts)
                            contrast_vars_batch_qr = torch.einsum(
                                "cr,brs,cs->bc",
                                glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                                var_beta_batch,  # (batch_voxels, n_regressors, n_regressors)
                                glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                            )

                        contrast_se_batch_qr = torch.sqrt(
                            torch.clamp(contrast_vars_batch_qr, min=0.0)
                        )
                        contrast_tstats_batch_qr = contrast_betas_batch_qr / (
                            contrast_se_batch_qr + 1e-10
                        )

                        # Compute partial R² for contrasts if requested
                        if want_r2_partial:
                            df = results.dof
                            contrast_t_squared = contrast_tstats_batch_qr**2
                            contrast_r2_partial_batch = contrast_t_squared / (
                                contrast_t_squared + df
                            )

                        # Store immediately for QR path
                        results.contrast_betas[sub_voxel_indices] = contrast_betas_batch_qr.cpu()
                        results.contrast_tstats[sub_voxel_indices] = contrast_tstats_batch_qr.cpu()
                        if want_r2_partial:
                            results.contrast_r2_partial[sub_voxel_indices] = (
                                contrast_r2_partial_batch.cpu()
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t_glt_total += time.time() - t_glt_start

                else:
                    # OPTIMIZED X'X PATH: Use precomputed group-level X'X inverse
                    # X'X is constant for all voxels in group → only compute X'y per batch!
                    t_solve_start = time.time()

                    # Compute X'y for each voxel (only batch-specific part)
                    # X_w: (n_time, n_reg), y_w_batch: (batch, n_time) → X'y: (batch, n_reg)
                    XwTy_batch = torch.mm(
                        y_w_batch, X_w
                    )  # Efficient: no expand/bmm materialization!

                    # Solve: β = (X'X)^{-1} X'y using precomputed (X'X)^{-1}
                    # XwTXw_inv_group: (n_reg, n_reg), XwTy_batch: (batch, n_reg)
                    betas_batch = torch.mm(XwTy_batch, XwTXw_inv_group.T)  # (batch, n_reg)

                    del XwTy_batch
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_qr_solve_total += time.time() - t_solve_start

                    # Statistics computation
                    t_stats_start = time.time()
                    # pred = X @ beta - use efficient mm instead of expand+bmm
                    # X_w: (n_time, n_reg), betas_batch: (batch, n_reg) → pred: (batch, n_time)
                    pred_w_batch = torch.mm(betas_batch, X_w.T)  # Efficient!
                    resid_w_batch = y_w_batch - pred_w_batch

                    pred_orig_batch = torch.mm(betas_batch, design.T)  # Also efficient!
                    resid_orig_batch = Y_batch_dev.T - pred_orig_batch

                    df = results.dof
                    # Double-precision accumulation for RSS: eliminates rounding in sum-of-squares
                    # fp64 accumulator avoids RSS rounding without materializing a fp64 copy
                    sigma2_batch = (resid_w_batch.pow(2).sum(dim=1, dtype=_accum_dtype) / df).to(
                        dtype
                    )

                    if want_ljung_box:
                        results.ljung_box[sub_voxel_indices] = (
                            _ljung_box_batched(resid_w_batch, lj_max_lag, lj_tau)
                            .to(torch.float32)
                            .cpu()
                        )

                    if not want_residuals:
                        del resid_w_batch

                    # Variance: (X'X)^{-1} σ² - Use precomputed (X'X)^{-1} from group level!
                    # No need to invert per batch - just broadcast σ² across voxels
                    if glt_contrasts_tensor is not None:
                        # Need full var_beta_batch for GLT contrast computation
                        # Expand precomputed inverse to batch dimension
                        XwTXw_inv_batch = XwTXw_inv_group.unsqueeze(0).expand(batch_voxels, -1, -1)

                        var_beta_batch = sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
                        se_beta_batch = torch.sqrt(torch.diagonal(var_beta_batch, dim1=1, dim2=2))
                    else:
                        # No GLTs - only need diagonal for standard errors (much faster!)
                        # Use precomputed diagonal from group level
                        # XwTXw_inv_diag_group: (n_reg,) - same for all voxels!
                        se_beta_batch = torch.sqrt(
                            sigma2_batch.unsqueeze(1) * XwTXw_inv_diag_group.unsqueeze(0)
                        )
                        var_beta_batch = None  # Not computed when no GLTs

                    tstats_batch = betas_batch / (se_beta_batch + 1e-10)
                    del se_beta_batch
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_stats_total += time.time() - t_stats_start

                    # Partial R² per regressor: r²_partial_i = t²_i / (t²_i + df)
                    if want_r2_partial:
                        df = results.dof
                        t_squared_batch = tstats_batch**2
                        r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
                        del t_squared_batch

                        # Split into task vs nuisance, rescale if mode='task'
                        if fitted_column_indices is not None:
                            task_indices_set = set(fitted_column_indices)
                            all_indices = set(range(n_regressors))
                            nuisance_indices = sorted(list(all_indices - task_indices_set))

                            r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                            # Always extract nuisance for storage (used for -bout output)
                            r2_partial_nuisance_batch = (
                                r2_partial_full_batch[:, nuisance_indices]
                                if len(nuisance_indices) > 0
                                else None
                            )

                            if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                                # nuisance_indices non-empty here => set to non-None above
                                assert r2_partial_nuisance_batch is not None
                                # Rescale task partial R² by variance remaining after nuisance
                                r2_nuisance_total = r2_partial_nuisance_batch.sum(
                                    dim=1, keepdim=True
                                )
                                denominator = torch.clamp(1.0 - r2_nuisance_total, min=0.01)
                                r2_partial_batch = r2_partial_task_batch / denominator
                            else:
                                r2_partial_batch = r2_partial_task_batch
                        else:
                            # No filtering - use all
                            r2_partial_batch = r2_partial_full_batch
                            r2_partial_nuisance_batch = None

                    # Semi-partial R² per regressor: r²_semi_i = partial_r²_i * (1 - R²_full)
                    if want_r2_semipartial:
                        # Need to get R² for each voxel in the batch (computed later, so use a placeholder)
                        # We'll compute it after R² calculation below
                        pass  # Mark for later computation

                    # F-stat: F = (β' M β) / (σ² · p)  with M = A_task^{-1} (or A_full^{-1})
                    # constant per group → no (batch, n_reg, n_reg) tensor, no per-batch inv.
                    if fitted_column_indices is not None:
                        betas_task_batch = betas_batch[:, fitted_column_indices]
                        n_task_params = len(fitted_column_indices)
                        quad_batch = torch.einsum(
                            "bi,ij,bj->b",
                            betas_task_batch,
                            A_task_inv_group,
                            betas_task_batch,
                        )
                        fstats_batch = quad_batch / (sigma2_batch * n_task_params + 1e-10)
                        del quad_batch, betas_task_batch
                    else:
                        quad_batch = torch.einsum(
                            "bi,ij,bj->b", betas_batch, A_full_inv_group, betas_batch
                        )
                        fstats_batch = quad_batch / (sigma2_batch * n_regressors + 1e-10)
                        del quad_batch

                    # R²
                    # Compute using unified function (Y_batch_dev is transposed)
                    r2_batch = compute_r2_metric(Y_batch_dev.T, pred_orig_batch, metric="cod")

                    # Now compute semi-partial R² (non-QR path)
                    if want_r2_semipartial:
                        # r2_batch is float32 (from compute_r2_metric); cast to dtype for mixed-precision safety
                        variance_remaining = torch.clamp(
                            1.0 - r2_batch.to(dtype).unsqueeze(1), min=0.0
                        )

                        # If we didn't compute partial R² above, compute it now for semi-partial
                        if not want_r2_partial:
                            df = results.dof
                            t_squared_batch = tstats_batch**2
                            r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)

                            # Split into task vs nuisance
                            if fitted_column_indices is not None:
                                task_indices_set = set(fitted_column_indices)
                                all_indices = set(range(n_regressors))
                                nuisance_indices = sorted(list(all_indices - task_indices_set))

                                r2_partial_task_batch = r2_partial_full_batch[
                                    :, fitted_column_indices
                                ]
                                r2_partial_nuisance_batch = (
                                    r2_partial_full_batch[:, nuisance_indices]
                                    if len(nuisance_indices) > 0
                                    else None
                                )
                            else:
                                r2_partial_task_batch = r2_partial_full_batch
                                r2_partial_nuisance_batch = None

                        # Compute semi-partial R² from partial R²
                        r2_semipartial_task_batch = r2_partial_task_batch * variance_remaining

                        # Nuisance semi-partial R²
                        r2_semipartial_nuisance_batch = (
                            r2_partial_nuisance_batch * variance_remaining
                            if r2_partial_nuisance_batch is not None
                            else None
                        )

                        # Apply rescaling mode for task regressors
                        if (
                            r2_semipartial_mode == "task"
                            and r2_semipartial_nuisance_batch is not None
                        ):
                            # Rescale by variance remaining after nuisance
                            r2_semi_nuisance_total = r2_semipartial_nuisance_batch.sum(
                                dim=1, keepdim=True
                            )
                            denominator = torch.clamp(1.0 - r2_semi_nuisance_total, min=0.01)
                            r2_semipartial_batch = r2_semipartial_task_batch / denominator
                        else:
                            # Full mode: use raw semi-partial R² values
                            r2_semipartial_batch = r2_semipartial_task_batch

                # GLT CONTRASTS: Compute in-loop (never store full var_betas!)
                # For each contrast c: compute c'β and Var(c'β) = c' Var(β) c
                if glt_contrasts_tensor is not None:
                    t_glt_start = time.time()
                    # Compute c'β for all contrasts at once
                    # glt_contrasts_tensor: (n_contrasts, n_regressors)
                    # betas_batch: (batch_size, n_regressors)
                    # Result: (batch_size, n_contrasts)
                    contrast_betas_batch = torch.mm(betas_batch, glt_contrasts_tensor.T)

                    # Compute Var(c'β) = c' Var(β) c for each contrast
                    # var_beta_batch: (batch_size, n_reg, n_reg)
                    # c: (n_reg,) for each contrast
                    if legacy_contrasts:
                        # LEGACY: Loop-based computation (slow, for validation only)
                        contrast_vars_batch = torch.zeros(
                            batch_voxels, n_contrasts, device=device, dtype=dtype
                        )
                        for c_idx in range(n_contrasts):
                            c = glt_contrasts_tensor[c_idx]  # (n_reg,)
                            # c' Var(β) c = quadratic form
                            c_var = torch.bmm(
                                c.unsqueeze(0)
                                .unsqueeze(1)
                                .expand(batch_voxels, 1, -1),  # (batch, 1, n_reg)
                                var_beta_batch,  # (batch, n_reg, n_reg)
                            )  # (batch, 1, n_reg)
                            contrast_vars_batch[:, c_idx] = torch.bmm(
                                c_var,  # (batch, 1, n_reg)
                                c.unsqueeze(0)
                                .unsqueeze(2)
                                .expand(batch_voxels, -1, 1),  # (batch, n_reg, 1)
                            ).squeeze()  # (batch,)
                    else:
                        # OPTIMIZED: Vectorized einsum (10-50x faster!)
                        # Compute Var(c'β) = c' Var(β) c for all contrasts at once
                        # glt_contrasts_tensor: (n_contrasts, n_regressors)
                        # var_beta_batch: (batch_voxels, n_regressors, n_regressors)
                        # Result: (batch_voxels, n_contrasts)
                        contrast_vars_batch = torch.einsum(
                            "cr,brs,cs->bc",
                            glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                            var_beta_batch,  # (batch_voxels, n_regressors, n_regressors)
                            glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                        )

                    # Compute t-statistics for contrasts
                    contrast_se_batch = torch.sqrt(torch.clamp(contrast_vars_batch, min=0.0))
                    contrast_tstats_batch = contrast_betas_batch / (contrast_se_batch + 1e-10)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_glt_total += time.time() - t_glt_start

                    # Compute partial R² for contrasts if requested
                    if want_r2_partial:
                        df = results.dof
                        contrast_t_squared = contrast_tstats_batch**2
                        contrast_r2_partial_batch = contrast_t_squared / (contrast_t_squared + df)

                    # Compute semi-partial R² for contrasts if requested
                    if want_r2_semipartial:
                        df = results.dof
                        contrast_t_squared = contrast_tstats_batch**2
                        contrast_r2_partial_batch_temp = contrast_t_squared / (
                            contrast_t_squared + df
                        )
                        # Use the r2_batch that was fetched earlier in semi-partial computation
                        # If we didn't compute semi-partial above (e.g., no task regressors), get it now
                        if "r2_batch" not in locals():
                            # Cast to dtype: results.r2 is float32 but computation may be float64
                            r2_batch = results.r2[sub_voxel_indices].to(device=device, dtype=dtype)
                        variance_remaining_contrasts = torch.clamp(
                            1.0 - r2_batch.to(dtype).unsqueeze(1), min=0.0
                        )
                        contrast_r2_semipartial_batch = (
                            contrast_r2_partial_batch_temp * variance_remaining_contrasts
                        )

                    # Store contrast results
                    results.contrast_betas[sub_voxel_indices] = contrast_betas_batch.cpu()
                    results.contrast_tstats[sub_voxel_indices] = contrast_tstats_batch.cpu()
                    if want_r2_partial:
                        results.contrast_r2_partial[sub_voxel_indices] = (
                            contrast_r2_partial_batch.cpu()
                        )
                    if want_r2_semipartial:
                        results.contrast_r2_semipartial[sub_voxel_indices] = (
                            contrast_r2_semipartial_batch.cpu()
                        )

                # Move to CPU and store
                # Extract only task columns if filtering is requested
                if fitted_column_indices is not None:
                    results.betas[sub_voxel_indices] = betas_batch[:, fitted_column_indices].cpu()
                    results.tstats[sub_voxel_indices] = tstats_batch[:, fitted_column_indices].cpu()
                else:
                    results.betas[sub_voxel_indices] = betas_batch.cpu()
                    results.tstats[sub_voxel_indices] = tstats_batch.cpu()
                results.sigma2[sub_voxel_indices] = sigma2_batch.cpu()
                results.fstats[sub_voxel_indices] = fstats_batch.cpu()
                results.r2[sub_voxel_indices] = r2_batch.cpu()

                # Store partial R² if requested
                # NOTE: r2_partial_batch is already filtered to task columns in the computation above
                if want_r2_partial:
                    results.r2_partial[sub_voxel_indices] = r2_partial_batch.cpu()
                    # Store nuisance partial R² if we extracted it
                    if (
                        r2_partial_nuisance_batch is not None
                        and results.r2_partial_nuisance is not None
                    ):
                        results.r2_partial_nuisance[sub_voxel_indices] = (
                            r2_partial_nuisance_batch.cpu()
                        )

                # Store semi-partial R² if requested
                # NOTE: r2_semipartial_batch is already filtered to task columns in the computation above
                if want_r2_semipartial:
                    results.r2_semipartial[sub_voxel_indices] = r2_semipartial_batch.cpu()
                    # Store nuisance semi-partial R² if we extracted it
                    if (
                        r2_semipartial_nuisance_batch is not None
                        and results.r2_semipartial_nuisance is not None
                    ):
                        results.r2_semipartial_nuisance[sub_voxel_indices] = (
                            r2_semipartial_nuisance_batch.cpu()
                        )

                # Optional outputs
                if want_residuals:
                    results.residuals[sub_voxel_indices] = resid_orig_batch.cpu()
                    results.residuals_whitened[sub_voxel_indices] = resid_w_batch.cpu()
                if want_predicted:
                    results.predicted[sub_voxel_indices] = pred_orig_batch.cpu()

                # Explicitly free GPU tensors after copying to CPU
                del (
                    betas_batch,
                    sigma2_batch,
                    tstats_batch,
                    fstats_batch,
                    r2_batch,
                    var_beta_batch,
                )
                del Y_batch_dev
                if want_residuals or want_predicted:
                    del resid_orig_batch
                if want_residuals:
                    del resid_w_batch
                if want_predicted:
                    del pred_orig_batch

                # Aggressive GPU cache clearing after each sub-batch (helps on smaller GPUs)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _gls_dbg.__exit__(None, None, None)

            # End of sub-batch loop - all voxels for this (a,b) group processed

            # Delete group-level precomputed matrices to free GPU memory for next group
            del L_chol, X_w
            if use_qr:
                del Q_group, R_qr_group, R_inv_group, XwTXw_inv_group
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Update progress bar (one update per group)
            if pbar is not None:
                pbar.update(1)

        # End of group loop - close progress bar
        if pbar is not None:
            pbar.close()

        if verbose:
            print(f"\n✓ Processed all {n_unique_pairs} unique (a,b) groups")
            print(f"  Total voxels processed: {total_voxels_in_groups:,}")

    else:
        # Global ARMA estimation (faster, less accurate)
        if verbose:
            print("Estimating global ARMA parameters from mean timeseries...")

        # Use mean timeseries for parameter estimation (move to device)
        y_mean_cpu = data.mean(dim=0)
        y_mean = y_mean_cpu.to(device)

        # REML grid search
        a_opt, b_opt, likelihood_opt = reml_grid_search(
            design, y_mean, a_grid, b_grid, device, run_starts=run_starts, tau=tau
        )

        if verbose:
            print(f"  Optimal (a, b) = ({a_opt:.3f}, {b_opt:.3f})")

        # Fill parameters for all voxels
        results.arma_params[:, 0] = a_opt
        results.arma_params[:, 1] = b_opt
        results.reml_likelihood[:] = likelihood_opt

        # Compute λ
        lam = (b_opt + a_opt) * (1 + a_opt * b_opt) / (1 + 2 * a_opt * b_opt + b_opt**2 + 1e-10)
        results.arma_lambda[:] = lam

        # Move design to device for prewhitening
        design_dev = design.to(device) if design.device != device else design

        # Prewhiten design (shared across all voxels)
        X_w, _, L_global = prewhiten_with_arma11(
            design_dev, y_mean, a_opt, b_opt, run_starts=run_starts, tau=tau
        )
        XwTXw = X_w.T @ X_w
        XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device, dtype=dtype)

        # GLS fit for all voxels
        if verbose:
            print("Fitting all voxels with global parameters...")

        voxel_iterator = range(n_voxels)
        if verbose and n_voxels > 1000:
            voxel_iterator = tqdm(voxel_iterator, desc="Global ARMA fitting", unit="voxel")

        for v in voxel_iterator:
            y_v_cpu = data[v]
            y_v_dev = y_v_cpu.to(device)
            if _y_norm_scale is not None:
                _sv = _y_norm_scale[v]
                y_v_dev = y_v_dev / _sv
                y_v_cpu = y_v_cpu / _sv  # keep CPU copy consistent for residuals/r2

            # Prewhiten data via triangular solve (reuse L from global estimation)
            y_v_col = y_v_dev.unsqueeze(1) if y_v_dev.ndim == 1 else y_v_dev
            y_w = torch.linalg.solve_triangular(L_global, y_v_col, upper=False)
            if y_w.shape[-1] == 1:
                y_w = y_w.squeeze(-1)

            # GLS fit
            beta = torch.linalg.solve(XwTXw_reg, X_w.T @ y_w)
            beta_cpu = beta.cpu()
            results.betas[v] = beta_cpu

            # Predictions and residuals
            pred_w = X_w @ beta
            resid_w = y_w - pred_w
            pred_orig = design_dev @ beta
            pred_orig_cpu = pred_orig.cpu()
            resid_orig_cpu = y_v_cpu - pred_orig_cpu

            # Variance
            df = results.dof
            sigma2 = (resid_w.to(_accum_dtype).pow(2).sum() / df).to(dtype)
            sigma2_cpu = sigma2.cpu()
            results.sigma2[v] = sigma2_cpu

            # t-statistics
            var_beta = sigma2 * torch.linalg.inv(XwTXw_reg)
            se_beta = torch.sqrt(torch.diag(var_beta))
            tstats = beta / (se_beta + 1e-10)
            results.tstats[v] = tstats.cpu()

            # Partial R² per regressor: r²_partial_i = t²_i / (t²_i + df)
            if want_r2_partial:
                df = results.dof
                t_squared = tstats**2
                r2_partial_full = t_squared / (t_squared + df)

                # Split into task vs nuisance, rescale if mode='task'
                if fitted_column_indices is not None:
                    task_indices_set = set(fitted_column_indices)
                    all_indices = set(range(len(beta)))
                    nuisance_indices = sorted(list(all_indices - task_indices_set))

                    r2_partial_task = r2_partial_full[fitted_column_indices]
                    # Always extract nuisance for storage (used for -bout output)
                    r2_partial_nuisance = (
                        r2_partial_full[nuisance_indices] if len(nuisance_indices) > 0 else None
                    )

                    if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                        # nuisance_indices non-empty here => set to non-None above
                        assert r2_partial_nuisance is not None
                        # Rescale task partial R² by variance remaining after nuisance
                        r2_nuisance_total = r2_partial_nuisance.sum()
                        denominator = torch.clamp(1.0 - r2_nuisance_total, min=0.01)
                        r2_partial = r2_partial_task / denominator
                    else:
                        r2_partial = r2_partial_task

                    results.r2_partial[v] = r2_partial.cpu()
                    # Store nuisance partial R² if allocated
                    if r2_partial_nuisance is not None and results.r2_partial_nuisance is not None:
                        results.r2_partial_nuisance[v] = r2_partial_nuisance.cpu()
                else:
                    # No filtering - use all
                    results.r2_partial[v] = r2_partial_full.cpu()

            # F-stat: Test only TASK regressors (not nuisance)
            # Use CORRECT formula: F = β_task' Var(β_task)^{-1} β_task / p_task
            if fitted_column_indices is not None:
                # Extract task columns only
                beta_task = beta[fitted_column_indices]
                n_task_params = len(fitted_column_indices)

                # Extract task block from variance matrix: Var_task = σ² (X'X)^{-1}_task
                var_task = var_beta[fitted_column_indices, :][:, fitted_column_indices]

                # Invert variance block: Var_task^{-1}
                var_task_inv = torch.linalg.inv(
                    var_task + 1e-8 * torch.eye(n_task_params, device=device, dtype=dtype)
                )

                # Compute quadratic form: β_task' Var_task^{-1} β_task
                quad = torch.dot(beta_task, var_task_inv @ beta_task)
                fstat = quad / n_task_params
            else:
                # No filtering - test all regressors
                var_inv = torch.linalg.inv(
                    var_beta + 1e-8 * torch.eye(n_regressors, device=device, dtype=dtype)
                )
                quad = torch.dot(beta, var_inv @ beta)
                fstat = quad / n_regressors

            results.fstats[v] = fstat.cpu()

            # R² (per-voxel case - reshape to use unified function)
            # y_v_cpu and pred_orig_cpu are 1D tensors (n_timepoints,)
            results.r2[v] = compute_r2_metric(
                y_v_cpu.unsqueeze(0), pred_orig_cpu.unsqueeze(0), metric="cod"
            ).squeeze()

            # Optional outputs
            if want_ljung_box:
                results.ljung_box[v] = _ljung_box_batched(
                    resid_w.reshape(1, -1), lj_max_lag, lj_tau
                ).item()
            if want_residuals:
                results.residuals[v] = resid_orig_cpu
                results.residuals_whitened[v] = resid_w.cpu()
            if want_predicted:
                results.predicted[v] = pred_orig_cpu

    # Voxel-wise (-dsort / ANATICOR) regressors: the base GLS above is the
    # no-dsort ("_nods") fit. Snapshot it if requested, then redo the final GLS
    # per voxel with the extended design (a, b unchanged, per AFNI).
    nods_results: ARMA11Results | None = None
    if dsort is not None:
        if want_dsort_nods:
            nods_results = ARMA11Results()
            for _attr, _val in vars(results).items():
                if isinstance(_val, torch.Tensor):
                    setattr(nods_results, _attr, _val.clone())
                else:
                    setattr(nods_results, _attr, _val)

        _fit_dsort_gls_pass(
            data,
            design,
            cast(torch.Tensor, dsort),
            results,
            fitted_column_indices=fitted_column_indices,
            glt_contrasts_tensor=glt_contrasts_tensor,
            want_r2_partial=want_r2_partial,
            r2_partial_mode=r2_partial_mode,
            want_r2_semipartial=want_r2_semipartial,
            r2_semipartial_mode=r2_semipartial_mode,
            want_residuals=want_residuals,
            want_predicted=want_predicted,
            run_starts=run_starts,
            tau=tau,
            device=device,
            dtype=dtype,
            accum_dtype=_accum_dtype,
            y_norm_scale=_y_norm_scale,
            batch_size=batch_size,
            verbose=verbose,
            run_bounds=dsort_run_bounds,
            want_ljung_box=want_ljung_box,
            lj_max_lag=lj_max_lag,
            lj_tau=lj_tau,
        )
        _n_dsort_cols = int(dsort.shape[1]) * (
            (len(dsort_run_bounds) - 1) if dsort_run_bounds is not None else 1
        )
        results.dsort_labels = dsort_labels or [f"dsort#{k}" for k in range(_n_dsort_cols)]
        results.nods_results = nods_results

    # Unscale results from per-voxel float32 conditioning.
    # t-stats, R², F-stats, partial R² are scale-invariant — no unscaling needed.
    # Betas, sigma2, residuals, and predicted must be scaled back to original units.
    if _y_norm_scale is not None:
        scale_col = _y_norm_scale.unsqueeze(1)  # (n_voxels, 1)
        results.betas.mul_(scale_col)
        results.sigma2.mul_(_y_norm_scale**2)
        if results.contrast_betas is not None:
            results.contrast_betas.mul_(scale_col)
        if results.residuals is not None:
            results.residuals.mul_(scale_col)
        if results.residuals_whitened is not None:
            results.residuals_whitened.mul_(scale_col)
        if results.predicted is not None:
            results.predicted.mul_(scale_col)
        # dsort coefficients scale like any beta (Y was divided by its std).
        if results.dsort_betas is not None:
            results.dsort_betas.mul_(scale_col)
        # The _nods snapshot was taken in normalized units — unscale it too.
        _nods = results.nods_results
        if isinstance(_nods, ARMA11Results):
            assert _nods.betas is not None and _nods.sigma2 is not None
            _nods.betas.mul_(scale_col)
            _nods.sigma2.mul_(_y_norm_scale**2)
            if _nods.contrast_betas is not None:
                _nods.contrast_betas.mul_(scale_col)
            if _nods.residuals is not None:
                _nods.residuals.mul_(scale_col)
            if _nods.residuals_whitened is not None:
                _nods.residuals_whitened.mul_(scale_col)
            if _nods.predicted is not None:
                _nods.predicted.mul_(scale_col)

    # Restore TF32 setting
    if _saved_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = _saved_tf32

    if verbose:
        print("\nComplete!")
        print(
            f"  Mean (a, b): ({results.arma_params[:, 0].mean():.3f}, "
            f"{results.arma_params[:, 1].mean():.3f})"
        )
        print(f"  Mean λ: {results.arma_lambda.mean():.3f}")
        print(f"  Mean R²: {results.r2.mean():.3f}")
        print(f"  Mean |t|: {results.tstats.abs().mean():.3f}")

    return results


def compare_ols_vs_arma11(
    data: torch.Tensor | np.ndarray,
    design: torch.Tensor | np.ndarray,
    tr: float,
    device: torch.device | None = None,
) -> dict:
    """
    Compare OLS vs ARMA(1,1) GLM results

    Useful for demonstrating the impact of temporal autocorrelation correction.

    Parameters
    ----------
    data : array-like
        fMRI data
    design : array-like
        Design matrix
    tr : float
        Repetition time
    device : torch.device, optional
        Computing device

    Returns
    -------
    comparison : dict
        'ols': OLS results
        'arma11': ARMA(1,1) results
        'tstat_ratio': ratio of |t-stats| (ARMA/OLS)
        'r2_improvement': R² difference (ARMA - OLS)
        'summary': summary text
    """
    from .core import fit_glm

    if device is None:
        device = get_device()

    print("Running OLS GLM...")
    ols_results = fit_glm(data, design, tr, device=device, verbose=False)

    print("\nRunning ARMA(1,1) GLM...")
    arma_results = fit_glm_arma11(data, design, tr, device=device, verbose=True)

    # A completed fit_glm/fit_glm_arma11 call always populates these fields.
    assert ols_results.tstats is not None and ols_results.r2 is not None
    assert arma_results.tstats is not None and arma_results.r2 is not None
    assert arma_results.arma_params is not None and arma_results.arma_lambda is not None

    # Compare t-statistics
    tstat_ratio = arma_results.tstats.abs().mean() / (ols_results.tstats.abs().mean() + 1e-10)

    # R² improvement
    r2_improvement = arma_results.r2.mean() - ols_results.r2.mean()

    summary = f"""
    OLS vs ARMA(1,1) Comparison:
    ============================
    OLS Mean R²:      {ols_results.r2.mean():.4f}
    ARMA Mean R²:     {arma_results.r2.mean():.4f}
    R² Improvement:   {r2_improvement:.4f}

    OLS Mean |t|:     {ols_results.tstats.abs().mean():.3f}
    ARMA Mean |t|:    {arma_results.tstats.abs().mean():.3f}
    |t| Ratio:        {tstat_ratio:.3f}

    ARMA Parameters:
    Mean (a, b):      ({arma_results.arma_params[:, 0].mean():.3f}, {arma_results.arma_params[:, 1].mean():.3f})
    Mean λ (lag-1):   {arma_results.arma_lambda.mean():.3f}

    Interpretation:
    - t-ratio < 1: ARMA reduces inflated t-stats (typical for positive autocorrelation)
    - t-ratio > 1: ARMA increases t-stats (rare, negative autocorrelation)
    - R² improvement: ARMA captures more variance due to better model
    """

    print(summary)

    return {
        "ols": ols_results,
        "arma11": arma_results,
        "tstat_ratio": tstat_ratio.item(),
        "r2_improvement": r2_improvement.item(),
        "summary": summary,
    }


# AFNI separates runs in the pseudo-time vector by a huge constant so no lag bin
# can ever straddle a run boundary (3dREMLfit.c:2439, "the 66666 means 'very far
# apart'"). Reused here for exactly that purpose.
_LJ_RUN_SEPARATION = 66666


def ljung_box_max_lag(n_time: int, n_regressors: int = 0, min_run: int | None = None) -> int:
    """AFNI's semi-arbitrary Ljung-Box max lag ``h`` (``3dREMLfit.c:3412``).

    ``h = nrega + 2 + min(min_run/8, round(3·ln min_run))``, clamped to
    ``min_run/2``. The reported chi-squared DOF is ``h - 2``.

    Parameters
    ----------
    n_time : int
        Retained timepoint count; used as ``min_run`` for single-run data.
    n_regressors : int, default=0
        Design matrix column count (AFNI's ``nrega``).
    min_run : int, optional
        Shortest run length. Defaults to *n_time* (AFNI's single-run branch).

    Notes
    -----
    AFNI derives ``min_run`` from the *pre-censor* timeline; we take it from the
    retained one, because run lengths that deep in the fit are already
    censor-collapsed. Identical without censoring, marginally smaller ``h`` with.
    """
    if min_run is None:
        min_run = n_time
    if min_run < 2:
        return 0
    h1 = min_run // 8
    h2 = int(round(3.0 * math.log(min_run)))
    hh = n_regressors + 2 + min(h1, h2)
    return min(hh, min_run // 2)


def _resolve_ljung_box_lag(n_time: int, max_lag: int | None) -> int:
    """``ljung_box_uneven``'s own sanity clamp on ``hh`` (``thd_ljungbox.c:21``).

    An out-of-range request (including the ``h = 0`` "you pick" sentinel) is
    replaced by ``2 + min(n/8, round(3·ln n))``, capped at ``n/2``. Note this
    fallback carries no ``nrega`` term — that is only in the caller's formula.
    """
    hh = int(max_lag) if max_lag else 0
    if hh < 2 or hh > n_time // 2:
        hh = 2 + min(n_time // 8, int(round(3.0 * math.log(n_time))))
        hh = min(hh, n_time // 2)
    return hh


def build_ljung_box_tau(
    n_time: int,
    run_starts: list[int] | torch.Tensor | None = None,
    tau: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Pseudo-time index for Ljung-Box lag binning, AFNI ``tau[]`` semantics.

    Combines the two things that make a lag-``k`` *index* pair not a lag-``k``
    *time* pair: censored TRs (survivors flanking a hole are further apart than
    their indices suggest) and run boundaries (never correlated at all). The
    former comes from *tau* — :func:`build_censor_run_info`'s within-run time
    index — and the latter is imposed by offsetting each run by
    :data:`_LJ_RUN_SEPARATION`, so cross-run pairs land outside every bin.

    Returns ``None`` for uncensored single-run data, where lag == index and the
    cheaper no-tau path in :func:`compute_ljung_box_statistic` is exact.
    """
    starts = (
        run_starts.tolist()
        if isinstance(run_starts, torch.Tensor)
        else (list(run_starts) if run_starts is not None else [])
    )
    starts = sorted({int(s) for s in starts if 0 <= int(s) < n_time}) or [0]
    if tau is None and len(starts) == 1 and starts[0] == 0:
        return None

    starts_t = torch.tensor(starts, dtype=torch.long)
    idx = torch.arange(n_time, dtype=torch.long)
    run_id = torch.bucketize(idx, starts_t, right=True) - 1
    within = tau.to(dtype=torch.long).flatten().cpu() if tau is not None else idx - starts_t[run_id]
    out = within + _LJ_RUN_SEPARATION * run_id
    return out.to(device) if device is not None else out


def _ljung_box_batched(
    resid: torch.Tensor,
    max_lag: int | None = None,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched Ljung-Box over voxels — port of AFNI ``ljung_box_uneven``.

    ``LB = n(n+2) · Σ_k c_k² / n_k`` with ``c_k = Σ_j r_j r_{j+k} / Σ_j r_j²``
    and ``n_k`` the number of pairs landing in time-lag bin ``k``. Note the
    numerator is a raw lagged product over the *un-centred, un-scaled* residual,
    not a Pearson autocorrelation — LB is scale-invariant (``c_k`` is a ratio) so
    per-voxel normalisation upstream cannot shift it, but re-centring would.

    Returns ``(n_voxels,)`` float64 on *resid*'s device. Zero marks "could not be
    computed" exactly as AFNI does: fewer than 10 points, an all-zero residual,
    or a bin with too few pairs.
    """
    n_vox, n_time = resid.shape
    zeros = torch.zeros(n_vox, dtype=torch.float64, device=resid.device)
    if n_time < 10:
        return zeros

    hh = _resolve_ljung_box_lag(n_time, max_lag)
    if hh < 1:
        return zeros

    # float64 throughout: c_k is a ratio of sums over the whole series, and the
    # lagged products are signed, so float32 cancellation is a real risk.
    r = resid.to(torch.float64)
    sum0 = r.pow(2).sum(dim=1)
    ok = sum0 >= 1e-10
    sum0 = torch.where(ok, sum0, torch.ones_like(sum0))

    sumk = torch.zeros(n_vox, hh + 1, dtype=torch.float64, device=r.device)
    nj = torch.zeros(hh + 1, dtype=torch.long, device=r.device)

    if tau is not None:
        tau = tau.to(device=r.device, dtype=torch.long).flatten()

    for kk in range(1, hh + 1):
        prod = r[:, : n_time - kk] * r[:, kk:]
        if tau is None:
            sumk[:, kk] = prod.sum(dim=1)
            nj[kk] = n_time - kk
        else:
            # Bin by *time* lag dj, not index lag kk: one index lag can scatter
            # across several bins once censoring stretches the gaps, and pairs
            # further apart than hh (including every cross-run pair) drop out.
            dj = tau[kk:] - tau[: n_time - kk]
            keep = (dj > 0) & (dj <= hh)
            if not bool(keep.any()):
                continue
            dj = dj[keep]
            sumk.index_add_(1, dj, prod[:, keep])
            nj.index_add_(0, dj, torch.ones_like(dj))

    # AFNI requires nj > 1 (not > 0) before a bin contributes.
    good = nj > 1
    good[0] = False
    if not bool(good.any()):
        return zeros

    ck = sumk[:, good] / sum0.unsqueeze(1)
    gsum = (ck.pow(2) / nj[good].to(torch.float64)).sum(dim=1) * (n_time * (n_time + 2.0))
    return torch.where(ok, gsum.clamp(max=1.0e10), torch.zeros_like(gsum))


def compute_ljung_box_statistic(
    residuals: torch.Tensor | np.ndarray,
    max_lag: int | None = None,
    tau: torch.Tensor | None = None,
) -> np.ndarray:
    """Ljung-Box whiteness statistic of prewhitened residuals (AFNI ``Rvar[5]``).

    "Did the ARMA(1,1) actually remove the autocorrelation?" — small is good;
    large means correlation survived the prewhitening and the model was
    inadequate for that voxel.

    Parameters
    ----------
    residuals : array-like, shape (n_voxels, n_timepoints)
        Prewhitened residuals.
    max_lag : int, optional
        Max lag ``h``. Default (None) uses AFNI's own choice for the series
        length; see :func:`ljung_box_max_lag` for the ``nrega``-aware form the
        ``-Rvar`` writer passes.
    tau : torch.Tensor, optional
        Pseudo-time index per retained point, from :func:`build_ljung_box_tau`.
        Required for censored or multi-run data, where index lag ≠ time lag.

    Returns
    -------
    lb_stats : np.ndarray, shape (n_voxels,)
        Chi-squared statistics with ``h - 2`` DOF. Zero = not computable.
    """
    resid = residuals if isinstance(residuals, torch.Tensor) else torch.from_numpy(residuals)
    return (
        _ljung_box_batched(resid.detach(), max_lag=max_lag, tau=tau)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def save_arma_rvar(
    results: ARMA11Results,
    output_path: str | Path,
    volume_shape: tuple[int, int, int] | None = None,
    voxel_mask: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    max_lag: int | None = None,
    tau: torch.Tensor | None = None,
) -> Path:
    """
    Save ARMA(1,1) variance parameters in AFNI 3dREMLfit -Rvar format

    Creates a 4D NIfTI file with 6 volumes (AFNI-compatible):
    - Volume 0: 'a' = AR parameter (decay rate of correlations)
    - Volume 1: 'b' = MA parameter
    - Volume 2: 'lam' = lambda = lag-1 correlation = (b+a)(1+a*b)/(1+2*a*b+b²)
    - Volume 3: 'StDev' = standard deviation of prewhitened residuals
    - Volume 4: '-LogLik' = negative REML log-likelihood
    - Volume 5: 'LjungBox' = Ljung-Box statistic (chi² with h-2 df)

    Each sub-brick is labeled in AFNI format for easy visualization.
    Use load_arma_params() to extract volumes 0 and 1 (a, b) for reuse.

    Parameters
    ----------
    results : ARMA11Results
        Results from fit_glm_arma11()
    output_path : str or Path
        Output file path (e.g., 'arma_rvar.nii.gz')
    volume_shape : tuple of int, optional
        Spatial dimensions (nx, ny, nz) from results.full_shape
    voxel_mask : np.ndarray, optional
        Boolean mask from results.voxel_mask
    affine : np.ndarray, optional
        4x4 affine from results.affine
    max_lag : int, optional
        Ljung-Box max lag, used only for the fallback recompute below. Ignored
        when ``results.ljung_box`` is already populated.
    tau : torch.Tensor, optional
        Pseudo-time index for the fallback recompute; see
        :func:`build_ljung_box_tau`. Ignored when ``results.ljung_box`` is set.

    Returns
    -------
    output_path : Path
        Path to saved file

    Examples
    --------
    >>> # Save AFNI-compatible output
    >>> results = ffs.fit_glm_arma11(data, design, tr=2.0, want_residuals=True)
    >>> ffs.save_arma_rvar(
    ...     results,
    ...     'arma_rvar.nii.gz',
    ...     volume_shape=results.full_shape,
    ...     voxel_mask=results.voxel_mask,
    ...     affine=results.affine
    ... )
    >>>
    >>> # Reuse parameters later (extracts volumes 0 and 1)
    >>> arma_params = ffs.load_arma_params('arma_rvar.nii.gz', mask)
    >>> results2 = ffs.fit_glm_arma11(
    ...     data2, design2, tr=2.0,
    ...     precomputed_arma_params=arma_params  # 80% faster!
    ... )

    References
    ----------
    AFNI 3dREMLfit: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dREMLfit.html
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    arma_params = (
        results.arma_params.detach().cpu().numpy()
        if isinstance(results.arma_params, torch.Tensor)
        else results.arma_params
    )
    arma_lambda = (
        results.arma_lambda.detach().cpu().numpy()
        if isinstance(results.arma_lambda, torch.Tensor)
        else results.arma_lambda
    )
    sigma = np.sqrt(
        results.sigma2.detach().cpu().numpy()
        if isinstance(results.sigma2, torch.Tensor)
        else results.sigma2
    )
    neg_loglik = (
        results.reml_likelihood.detach().cpu().numpy()
        if isinstance(results.reml_likelihood, torch.Tensor)
        else results.reml_likelihood
    )
    assert arma_lambda is not None, "arma_lambda should not be None"
    assert neg_loglik is not None, "reml_likelihood should not be None"

    # Ljung-Box. Normally computed inside the GLS loop (fit_glm_arma11
    # want_ljung_box=True) so -Rvar doesn't have to retain whole-brain whitened
    # residuals; fall back to those residuals only if some caller kept them.
    # AFNI writes this brick for every -Rvar, so a zero fill here is a real gap
    # in the output, not a default — warn rather than silently ship zeros.
    if results.ljung_box is not None:
        lj = results.ljung_box
        ljung_box = (lj.detach().cpu().numpy() if isinstance(lj, torch.Tensor) else lj).astype(
            np.float32
        )
    elif results.residuals_whitened is not None:
        ljung_box = compute_ljung_box_statistic(
            results.residuals_whitened, max_lag=max_lag, tau=tau
        )
    else:
        assert arma_params is not None, "arma_params should not be None"
        warnings.warn(
            "Rvar sub-brick 5 (LjungBox) written as zeros: no whiteness statistic was "
            "computed. Pass want_ljung_box=True to fit_glm_arma11() to populate it.",
            stacklevel=2,
        )
        ljung_box = np.zeros(len(arma_params), dtype=np.float32)

    # Reshape to volume if shape provided
    if volume_shape is not None:
        if voxel_mask is not None:
            # Masked data: expand back to full volume
            # Convert voxel_mask to numpy if it's a tensor
            if isinstance(voxel_mask, torch.Tensor):
                mask_flat = voxel_mask.detach().cpu().numpy().reshape(-1)
            else:
                mask_flat = voxel_mask.reshape(-1)
            full_size = np.prod(volume_shape)

            # Create full volumes with zeros
            a_vol = np.zeros(full_size, dtype=np.float32)
            b_vol = np.zeros(full_size, dtype=np.float32)
            lam_vol = np.zeros(full_size, dtype=np.float32)
            stdev_vol = np.zeros(full_size, dtype=np.float32)
            loglik_vol = np.zeros(full_size, dtype=np.float32)
            lb_vol = np.zeros(full_size, dtype=np.float32)

            # Fill in masked voxels
            a_vol[mask_flat] = arma_params[:, 0]
            b_vol[mask_flat] = arma_params[:, 1]
            lam_vol[mask_flat] = arma_lambda
            stdev_vol[mask_flat] = sigma
            loglik_vol[mask_flat] = neg_loglik
            lb_vol[mask_flat] = ljung_box

            # Reshape to 3D
            a_vol = a_vol.reshape(volume_shape)
            b_vol = b_vol.reshape(volume_shape)
            lam_vol = lam_vol.reshape(volume_shape)
            stdev_vol = stdev_vol.reshape(volume_shape)
            loglik_vol = loglik_vol.reshape(volume_shape)
            lb_vol = lb_vol.reshape(volume_shape)
        else:
            # No mask: direct reshape
            a_vol = arma_params[:, 0].reshape(volume_shape)
            b_vol = arma_params[:, 1].reshape(volume_shape)
            lam_vol = arma_lambda.reshape(volume_shape)
            stdev_vol = sigma.reshape(volume_shape)
            loglik_vol = neg_loglik.reshape(volume_shape)
            lb_vol = ljung_box.reshape(volume_shape)
    else:
        # No volume shape: save as flat
        a_vol = arma_params[:, 0].reshape(-1, 1, 1)
        b_vol = arma_params[:, 1].reshape(-1, 1, 1)
        lam_vol = arma_lambda.reshape(-1, 1, 1)
        stdev_vol = sigma.reshape(-1, 1, 1)
        loglik_vol = neg_loglik.reshape(-1, 1, 1)
        lb_vol = ljung_box.reshape(-1, 1, 1)

    # Stack into 4D: (nx, ny, nz, 6) - AFNI order
    rvar_4d = np.stack([a_vol, b_vol, lam_vol, stdev_vol, loglik_vol, lb_vol], axis=-1)

    # Get affine from results if not provided
    if affine is None and hasattr(results, "affine") and results.affine is not None:
        affine = results.affine
    # Create default affine if still None
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    from fastfuncstuff.io.afni import save_nifti, stat_type_to_stataux

    # Sub-brick 5 is a chi-squared statistic (AFNI EDIT_BRICK_TO_FICT,
    # 3dREMLfit.c:3417) — without the stataux tag viewers treat it as a plain
    # float and can neither threshold it nor compute its FDR curve.
    stataux = None
    if results.ljung_box_dof:
        stataux = {5: stat_type_to_stataux("fict", (float(results.ljung_box_dof),))}

    save_nifti(
        rvar_4d.astype(np.float32),
        output_path=output_path,
        affine=affine,
        brick_labels=["a", "b", "lam", "StDev", "-LogLik", "LjungBox"],
        brick_stataux=stataux,
    )

    return output_path


def load_arma_params(
    filepath: str | Path,
    voxel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Load precomputed ARMA(1,1) parameters from -Rvar NIfTI file

    Extracts volumes 0 and 1 (a and b) from the 6-volume -Rvar format:
    - Volume 0: 'a' (AR parameter)
    - Volume 1: 'b' (MA parameter)
    - [Volumes 2-5 are ignored: lambda, StDev, -LogLik, LjungBox]

    Parameters
    ----------
    filepath : str or Path
        Path to ARMA -Rvar file (from save_arma_rvar)
    voxel_mask : np.ndarray, optional
        Boolean mask indicating which voxels to extract.
        If provided, returns only masked voxels (n_masked_voxels, 2).
        If None, returns all voxels flattened (n_voxels, 2).

    Returns
    -------
    arma_params : np.ndarray, shape (n_voxels, 2)
        ARMA parameters [a, b] ready for use with fit_glm_arma11()

    Examples
    --------
    >>> # Load precomputed parameters from -Rvar file
    >>> arma_params = ffs.load_arma_params('arma_rvar.nii.gz', mask=mask)
    >>>
    >>> # Use for fast refit with different contrasts
    >>> results = ffs.fit_glm_arma11(
    ...     data, design, tr=2.0,
    ...     precomputed_arma_params=arma_params  # Skip REML estimation!
    ... )
    """
    from fastfuncstuff.io.afni import load_nifti

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ARMA params file not found: {filepath}")

    # Load NIfTI
    img = load_nifti(filepath)
    arma_4d = img.get_fdata(dtype=np.float32)

    # Extract a and b volumes (volumes 0 and 1)
    if arma_4d.ndim == 4 and arma_4d.shape[-1] >= 2:
        a_vol = arma_4d[..., 0]
        b_vol = arma_4d[..., 1]
    else:
        raise ValueError(f"Expected 4D NIfTI with at least 2 volumes, got shape {arma_4d.shape}")

    # Apply mask if provided
    if voxel_mask is not None:
        # Convert voxel_mask to numpy if it's a tensor
        if isinstance(voxel_mask, torch.Tensor):
            mask_flat = voxel_mask.detach().cpu().numpy().reshape(-1)
        else:
            mask_flat = voxel_mask.reshape(-1)
        a_flat = a_vol.reshape(-1)
        b_flat = b_vol.reshape(-1)

        arma_params = np.stack([a_flat[mask_flat], b_flat[mask_flat]], axis=1)
    else:
        # Return all voxels flattened
        a_flat = a_vol.reshape(-1)
        b_flat = b_vol.reshape(-1)
        arma_params = np.stack([a_flat, b_flat], axis=1)

    return arma_params
