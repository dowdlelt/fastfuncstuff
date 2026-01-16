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

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from tqdm.auto import tqdm


def _debug_memory_snapshot(
    label: str, device: torch.device, tensors: Optional[Dict[str, torch.Tensor]] = None
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

        print(
            f"  {'TOTAL':30s}: {'':<20s} {'':<10s} {'':<10s} {total_bytes / 1024**2:>8.1f} MiB"
        )
        print()

    print(f"{'=' * 70}\n")


from .utils import get_device, to_tensor

# AFNI 3dREMLfit default grid parameters (Grid 3 - medium resolution)
# These are well-validated values from AFNI documentation
DEFAULT_ARMA_A_GRID = (0.0, 0.9, 10)  # (start, end, num_points) -> [0.0, 0.1, 0.2, ..., 0.9]
DEFAULT_ARMA_B_GRID = (-0.9, 0.9, 19)  # (start, end, num_points) -> [-0.9, -0.8, ..., 0.8, 0.9]


def check_cuda_memory_before_batch(
    batch_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    device: torch.device,
    verbose: bool = False,
    use_qr: bool = False,
    has_glts: bool = True,
    persistent_tensors: Optional[dict] = None,  # NEW: Account for data already on GPU
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
    if use_qr:
        # QR path: Q and R are precomputed at group level, not allocated per batch
        required_bytes = (
            batch_voxels * n_timepoints * 4  # y_w_batch
            + batch_voxels * n_regressors * 4  # QTy_batch
            + batch_voxels * n_regressors * 4  # betas
            + batch_voxels * n_timepoints * 4  # pred_w_batch
            + batch_voxels * n_timepoints * 4  # resid_w_batch
            + batch_voxels * n_timepoints * 4  # pred_orig_batch
            + batch_voxels * n_timepoints * 4  # resid_orig_batch
        )
        if has_glts:
            # XwTXw_inv expanded from group (VIEW - no new allocation!)
            pass
    else:
        # X'X approach: NOW EXTREMELY MEMORY EFFICIENT!
        # X'X and (X'X)^{-1} precomputed at group level - only ~25 MB total!
        # Per-batch: just compute X'y and results - NO matrix inversions!
        required_bytes = (
            batch_voxels * n_timepoints * 4  # y_w_batch (Y_batch_dev)
            + batch_voxels * n_regressors * 4  # XwTy_batch
            + batch_voxels * n_regressors * 4  # betas
            + batch_voxels * n_timepoints * 4  # pred_w_batch
            + batch_voxels * n_timepoints * 4  # resid_w_batch
            + batch_voxels * n_timepoints * 4  # pred_orig_batch
            + batch_voxels * n_timepoints * 4  # resid_orig_batch
            # NO XwTXw_batch! NO L_batch! NO L_inv_batch! All precomputed!
        )

        # Variance computation uses precomputed group-level (X'X)^{-1}
        # Just expanded as a view - no per-batch allocation!
        # With diagonal-only optimization, we only need diagonal (already extracted at group level)
        if has_glts:
            # var_beta uses expanded view of XwTXw_inv_group - no allocation
            pass
        else:
            # Uses XwTXw_inv_diag_group - just broadcast, no allocation
            pass

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
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                tensor_bytes = tensor.numel() * tensor.element_size()
                persistent_mem_info[name] = tensor_bytes
                if verbose:
                    print(f"  Persistent GPU tensor '{name}': {tensor_bytes / 1e9:.2f} GB (already allocated)")

    # Use 60% of available (leave headroom for PyTorch overhead and temporary allocations)
    # Conservative factor accounts for:
    # - PyTorch internal temporary buffers (especially during Cholesky operations)
    # - Memory fragmentation
    # - Peak memory usage during operations (not just total allocated)
    usable_mem = available_mem * 0.60

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
            print(f"  GPU: {allocated_mem / 1e9:.2f} GB allocated, {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
            print(f"  📉 Reducing batch: {batch_voxels:,} → {new_batch:,} voxels")
            print("  ℹ️  Conservative: Accounts for peak memory during Cholesky/solve operations\n")

        return new_batch
    elif verbose:
        # Success - batch fits!
        print(f"  ✓ Batch fits: {required_bytes / 1e9:.2f} GB needed < {usable_mem / 1e9:.2f} GB available")

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
    - L_inv: inverse Cholesky factor (n_timepoints, n_timepoints)
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
    L_inv_size = n_timepoints * n_timepoints * bytes_per_element
    X_w_size = n_timepoints * n_regressors * bytes_per_element
    XwTXw_size = n_regressors * n_regressors * bytes_per_element
    scalars_size = 2 * bytes_per_element  # logdet_R + logdet_XwTXw

    per_pair_bytes = L_inv_size + X_w_size + XwTXw_size + scalars_size

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
    Intelligently determine optimal batch size based on GPU memory

    Uses device-specific heuristics to maximize GPU utilization while
    avoiding OOM (Out of Memory) errors.

    Parameters
    ----------
    device : torch.device
        Computing device
    n_timepoints : int
        Number of timepoints in data
    n_regressors : int
        Number of regressors in design
    use_double : bool, default=False
        If True, use float64 precision (8 bytes per element).
        If False, use float32 precision (4 bytes per element).

    Returns
    -------
    batch_size : int
        Recommended number of voxels to process in parallel

    Notes
    -----
    Memory scaling (with QR factorization):
    - QR creates temporary Q matrix same size as X_w (~2x memory during factorization!)
    - Each voxel needs ~2 × (n_timepoints × n_regressors × bytes_per_element) peak memory
    - Additional workspace for R, inverse matrices, residuals, betas, etc.
    - Float64 uses 2x memory vs float32, so batch size is automatically halved
    - QR uses ~2.25x more memory than old X'X approach, but gives better numerical stability

    Device-specific tuning:
    - MPS (Mac M-series): 36GB unified memory → aggressive batching (50k+ voxels float32, 25k float64)
    - CUDA (NVIDIA): Depends on VRAM (8GB=10k, 16GB=25k, 24GB=40k, 40GB=60k float32)
    - CPU: Conservative (5k voxels float32, 2.5k float64) due to slower computation

    Example
    -------
    >>> device = torch.device('mps')
    >>> batch_size = get_adaptive_batch_size(device, n_timepoints=300, n_regressors=48)
    >>> print(f"Optimal batch: {batch_size:,} voxels")
    Optimal batch: 50,000 voxels
    """
    # ACCURATE memory per voxel calculation for GLS solve
    bytes_per_element = 8 if use_double else 4

    if use_qr:
        # QR factorization: needs temporary Q matrix (same size as X_w!)
        mem_per_voxel = (
            n_timepoints * n_regressors * bytes_per_element  # X_w_batch
            + n_timepoints
            * n_regressors
            * bytes_per_element  # Q_batch (temporary during QR!)
            + n_timepoints * bytes_per_element  # y_w_batch
            + n_regressors * n_regressors * bytes_per_element  # R_qr_batch
            + n_regressors
            * n_regressors
            * bytes_per_element  # eye_batch (for computing inv)
            + n_regressors * n_regressors * bytes_per_element  # R_inv_batch
            + n_regressors * n_regressors * bytes_per_element  # XwTXw_inv_batch
            + n_regressors * bytes_per_element  # betas
            + n_regressors * bytes_per_element  # tstats
            + 3 * bytes_per_element  # params (a, b, lambda)
        )
    else:
        # X'X approach: less memory, no temporary Q matrix
        mem_per_voxel = (
            n_timepoints * n_regressors * bytes_per_element  # X_w_batch
            + n_timepoints * bytes_per_element  # y_w_batch
            + n_regressors * n_regressors * bytes_per_element  # XtX
            + n_regressors * n_regressors * bytes_per_element  # XtX_inv
            + n_regressors * bytes_per_element  # betas
            + n_regressors * bytes_per_element  # tstats
            + 3 * bytes_per_element  # params (a, b, lambda)
        )

    if device.type == "mps":
        # Mac M-series with unified memory (typically 16-64GB)
        available_gb = 8.0
        batch_size = int((available_gb * 1e9) / mem_per_voxel)
        batch_size = max(5000, min(batch_size, 50000))

    elif device.type == "cuda":
        # NVIDIA GPU - query actual VRAM
        try:
            total_mem = torch.cuda.get_device_properties(device).total_memory
            # Use 50% of VRAM for batch sizing (assumes dedicated GPU)
            # Leave 13% for PyTorch overhead, CUDA context, fragmentation, and grid memory
            # (fit_glm_arma11 will further refine this to account for actual grid size)
            available_mem = total_mem * 0.75
            batch_size = int(available_mem / mem_per_voxel)

            # Safety clamps based on GPU size
            # NOTE: With group-level X'X precomputation (99.6% memory reduction),
            # these caps are MUCH higher than before while still being conservative.
            # The actual batch size will be further adjusted in fit_glm_arma11().
            if total_mem < 10e9:  # < 10GB
                batch_size = min(batch_size, 50000)
            elif total_mem < 20e9:  # 10-20GB (e.g., RTX 4070, 15-16GB GPUs)
                batch_size = min(batch_size, 200000)
            else:  # > 20GB (e.g., RTX 4090, A100)
                batch_size = min(batch_size, 500000)
        except Exception:
            batch_size = 3000
    else:
        # CPU or unknown device
        # For CPU, we can use much larger batches since we have more RAM
        # and can parallelize across cores
        try:
            import psutil

            # Use 50% of available RAM (conservative, leaves room for OS and other processes)
            available_ram = psutil.virtual_memory().available * 0.5
            batch_size = int(available_ram / mem_per_voxel)
            # Reasonable limits for CPU (parallelizes well with larger batches)
            batch_size = max(5000, min(batch_size, 50000))
        except ImportError:
            # psutil not available, use conservative default
            batch_size = 5000

    # Ensure reasonable minimum
    batch_size = max(batch_size, 1000)

    return batch_size


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
    return _compute_arma11_lambda(a, b)


def get_default_arma_grids(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get default ARMA(1,1) parameter grids (AFNI -Grid 3 equivalent)

    Returns
    -------
    a_grid : torch.Tensor
        AR parameter grid [0.0, 0.1, 0.2, ..., 0.9] (10 points, step=0.1)
    b_grid : torch.Tensor
        MA parameter grid [-0.9, -0.8, ..., 0.8, 0.9] (19 points, step=0.1)

    Notes
    -----
    Total: 190 (a, b) combinations (10 × 19)
    Matches AFNI 3dREMLfit -Grid 3 (medium resolution, good balance)
    Invalid combinations filtered by stability criteria during fitting
    """
    a_grid = torch.linspace(*DEFAULT_ARMA_A_GRID, device=device)
    b_grid = torch.linspace(*DEFAULT_ARMA_B_GRID, device=device)
    return a_grid, b_grid


def ensure_zero_in_grid(
    a_grid: torch.Tensor, b_grid: torch.Tensor, tolerance: float = 1e-9
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure that both grids contain 0.0, adding it if necessary.

    This guarantees that the special case (a=0, b=0) is always tested,
    even if the user provides a grid that starts at 0.1 or has spacing
    that would skip zero.

    Parameters
    ----------
    a_grid : torch.Tensor
        AR parameter grid
    b_grid : torch.Tensor
        MA parameter grid
    tolerance : float, default=1e-9
        Tolerance for checking if zero is already in the grid

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

    # Check if 0.0 is already in a_grid (within tolerance)
    if not torch.any(torch.abs(a_grid) < tolerance):
        # Add 0.0 to a_grid and sort
        zero_tensor = torch.tensor([0.0], device=device, dtype=dtype)
        a_grid = torch.cat([zero_tensor, a_grid])
        a_grid = torch.sort(a_grid)[0]

    # Check if 0.0 is already in b_grid (within tolerance)
    if not torch.any(torch.abs(b_grid) < tolerance):
        # Add 0.0 to b_grid and sort
        zero_tensor = torch.tensor([0.0], device=device, dtype=dtype)
        b_grid = torch.cat([zero_tensor, b_grid])
        b_grid = torch.sort(b_grid)[0]

    return a_grid, b_grid


class ARMA11Results:
    """Container for ARMA(1,1) GLM results"""

    def __init__(self):
        self.betas: Optional[torch.Tensor] = (
            None  # (n_voxels, n_regressors) GLS parameter estimates
        )
        self.tstats: Optional[torch.Tensor] = (
            None  # (n_voxels, n_regressors) t-statistics (corrected)
        )
        self.r2: Optional[torch.Tensor] = None  # (n_voxels,) R² values (total model)
        self.r2_partial: Optional[torch.Tensor] = None  # (n_voxels, n_task_regressors) partial R² per TASK condition
        self.r2_partial_nuisance: Optional[torch.Tensor] = None  # (n_voxels, n_nuisance_regressors) partial R² per NUISANCE regressor
        self.r2_semipartial: Optional[torch.Tensor] = None  # (n_voxels, n_task_regressors) semi-partial R² per TASK condition
        self.r2_semipartial_nuisance: Optional[torch.Tensor] = None  # (n_voxels, n_nuisance_regressors) semi-partial R² per NUISANCE regressor
        self.arma_params: Optional[torch.Tensor] = (
            None  # (n_voxels, 2) - (a, b) per voxel
        )
        self.arma_lambda: Optional[torch.Tensor] = (
            None  # (n_voxels,) - lag-1 correlation
        )
        self.reml_likelihood: Optional[torch.Tensor] = (
            None  # (n_voxels,) - optimized REML log-likelihood
        )
        self.residuals: Optional[torch.Tensor] = (
            None  # (n_voxels, n_timepoints) - residuals in original space
        )
        self.predicted: Optional[torch.Tensor] = (
            None  # (n_voxels, n_timepoints) - predictions (original space)
        )
        self.residuals_whitened: Optional[torch.Tensor] = (
            None  # (n_voxels, n_timepoints) - residuals after whitening
        )
        self.sigma2: Optional[torch.Tensor] = (
            None  # (n_voxels,) - noise variance estimates
        )
        self.var_betas: Optional[torch.Tensor] = (
            None  # (n_voxels, n_regressors, n_regressors) - covariance
        )
        self.original_shape: Optional[Tuple[int, int, int]] = (
            None  # Original spatial dimensions
        )
        self.fstats: Optional[torch.Tensor] = (
            None  # (n_voxels,) - omnibus F-statistic across regressors
        )
        self.dof: Optional[int] = (
            None  # Degrees of freedom (n_timepoints - n_regressors)
        )
        self.tr: Optional[float] = None  # Repetition time
        self.voxel_mask: Optional[torch.Tensor] = (
            None  # Optional boolean mask for sparse analyses
        )
        self.full_shape: Optional[Tuple[int, int, int]] = (
            None  # Original spatial shape before masking
        )

        # Design filtering metadata (tracks what was actually fitted)
        self.fitted_column_indices: Optional[List[int]] = (
            None  # Indices of columns that were fitted (None = all columns)
        )
        self.n_regressors_full: Optional[int] = (
            None  # Total columns in original design before filtering
        )

        # GLT contrast results (computed in-loop, not post-hoc)
        self.contrast_labels: Optional[List[str]] = None  # List of contrast names
        self.contrast_betas: Optional[torch.Tensor] = (
            None  # (n_voxels, n_contrasts) - c'β estimates
        )
        self.contrast_tstats: Optional[torch.Tensor] = (
            None  # (n_voxels, n_contrasts) - t-statistics
        )
        self.contrast_fstats: Optional[torch.Tensor] = (
            None  # (n_voxels, n_contrasts) - F-statistics (for multi-row GLTs)
        )
        self.contrast_r2_partial: Optional[torch.Tensor] = (
            None  # (n_voxels, n_contrasts) - partial R² for each contrast
        )
        self.contrast_r2_semipartial: Optional[torch.Tensor] = (
            None  # (n_voxels, n_contrasts) - semi-partial R² for each contrast
        )
        self.affine: Optional[np.ndarray] = None  # Spatial affine if available
        self.ols_results: Optional[Any] = None  # Optional GLMResults for OLS comparison


def _compute_arma11_lambda(
    a: Union[float, torch.Tensor], b: Union[float, torch.Tensor]
) -> Union[float, torch.Tensor]:
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
    AFNI 3dREMLfit formula from Cox & Reynolds (2006)
    Valid for |a| < 1, |b| < 1 (checked by caller)
    """
    # This formula is mathematically equivalent to gamma1/gamma0
    # but more numerically stable (no (1-a²) division)
    numerator = (a + b) * (1 + a * b)
    denominator = 1 + b**2 + 2 * a * b
    return numerator / denominator


def build_arma11_covariance(
    a: float, b: float, n: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """
    Build ARMA(1,1) covariance matrix (Toeplitz structure)

    R[i,j] = λ * a^|i-j|  where λ = lag-1 correlation

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

    Returns
    -------
    R : torch.Tensor, shape (n, n)
        ARMA(1,1) covariance matrix, or None if parameters invalid

    Notes
    -----
    - λ >= 0 is enforced (AFNI default, invalid combinations return None)
    - Matrix is symmetric Toeplitz (constant along diagonals)
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

    # Build correlation vector: [1, λ, λ*a, λ*a², λ*a³, ...]
    corr = torch.zeros(n, device=device, dtype=dtype)
    corr[0] = 1.0

    if n > 1:
        corr[1] = rho1

        if n > 2:
            powers = torch.full((n - 2,), a, device=device, dtype=dtype)
            powers = torch.cumprod(powers, dim=0)
            corr[2:] = rho1 * powers

    # Build Toeplitz matrix from correlation vector
    idx = torch.arange(n, device=device)
    distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    R = corr[distance]

    # Ensure contiguous for MPS compatibility (PyTorch MPS Cholesky bug)
    R = R.contiguous()

    return R


def build_arma11_covariance_batch(
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
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

    # Build correlation vectors for ALL parameters: [1, λ, λ*a, λ*a², λ*a³, ...]
    # This must match the scalar version exactly!
    corr = torch.zeros(n_valid, n, device=device, dtype=dtype)
    corr[:, 0] = 1.0  # lag-0 correlation is always 1

    if n > 1:
        corr[:, 1] = rho1_valid  # lag-1 correlation is lambda

        if n > 2:
            # For lags k≥2: corr[k] = λ * a^(k-1)
            # k=2 → a^1, k=3 → a^2, ..., k=n-1 → a^(n-2)
            # So we need powers [a^1, a^2, a^3, ..., a^(n-2)]
            exponents = torch.arange(
                1, n - 1, device=device, dtype=dtype
            )  # [1, 2, 3, ..., n-2]
            powers = a_valid.unsqueeze(1) ** exponents.unsqueeze(0)  # (n_valid, n-2)
            corr[:, 2:] = rho1_valid.unsqueeze(1) * powers  # (n_valid, n-2)

    # Build Toeplitz matrices using fancy indexing
    # For each valid param, create (n, n) matrix from correlation vector
    idx = torch.arange(n, device=device)
    distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # (n, n)

    # Broadcast: corr is (n_valid, n), distance is (n, n)
    # Result: (n_valid, n, n) - all covariance matrices at once!
    R_batch = corr[:, distance]

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

    try:
        # Cholesky decomposition: R = L L'
        # MPS has a bug with Cholesky - use CPU as workaround
        if device.type == "mps":
            R_cpu = R.cpu()
            L_cpu = torch.linalg.cholesky(R_cpu)
            L_inv_cpu = torch.linalg.inv(L_cpu)
            L = L_cpu.to(device)
            L_inv = L_inv_cpu.to(device)
        else:
            L = torch.linalg.cholesky(R)
            L_inv = torch.linalg.inv(L)

        # Term 1: log(det(R)) = 2 * sum(log(diag(L)))
        term1 = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10))

        # Prewhiten design and data
        X_w = L_inv @ X
        Y_w = L_inv @ Y

        # Term 2: log(det(X' R^(-1) X)) = log(det(X_w' X_w))
        XwTXw = X_w.T @ X_w
        XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device)

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
        warnings.warn(f"Cholesky decomposition failed: {e}")
        return 1e10


def reml_grid_search(
    X: torch.Tensor,
    Y: torch.Tensor,
    a_grid: Optional[torch.Tensor] = None,
    b_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float]:
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
            R = build_arma11_covariance(a_val, b_val, n_timepoints, device)

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
        - 'L_inv': inverse Cholesky factor
        - 'X_w': prewhitened design
        - 'XwTXw_reg': regularized X'X
        - 'logdet_R': log(det(R))
        - 'logdet_XwTXw': log(det(X'R^-1 X))
    """
    # Ensure X has the correct dtype for all operations
    X = X.to(dtype=dtype)
    n_regressors = X.shape[1]
    precomputed = {}

    # PHASE 1: Build ALL covariance matrices at once (VECTORIZED!)
    # Build on CPU if we're going to do Cholesky on CPU anyway (saves GPU memory!)
    build_device = (
        torch.device("cpu") if (cholesky_on_cpu or device.type == "mps") else device
    )

    # Ensure optimal CPU threading for CPU-intensive operations
    if build_device.type == "cpu" and verbose:
        num_threads = torch.get_num_threads()
        if num_threads < 4:
            print(f"  WARNING: PyTorch using only {num_threads} CPU threads!")
            print(
                "  Set OMP_NUM_THREADS or torch.set_num_threads() for better performance"
            )

    # Move grids to build_device BEFORE ensure_zero_in_grid
    a_grid = a_grid.to(build_device)
    b_grid = b_grid.to(build_device)

    # Ensure (a=0, b=0) is always in the grid (white noise baseline)
    a_grid, b_grid = ensure_zero_in_grid(a_grid, b_grid)

    if verbose:
        print("  Building ALL covariance matrices (vectorized)...")
        n_total_grid = len(a_grid) * len(b_grid)
        print(
            f"    Initial grid: {len(a_grid)} a × {len(b_grid)} b = {n_total_grid} combinations"
        )

    R_batch, params_tensor, param_list = build_arma11_covariance_batch(
        a_grid, b_grid, n_timepoints, build_device, dtype
    )
    n_valid = len(param_list)

    if n_valid == 0:
        if verbose:
            print("  Warning: No valid (a,b) parameters in grid!")
        return precomputed

    if verbose:
        n_total_grid = len(a_grid) * len(b_grid)
        n_filtered = n_total_grid - n_valid
        print(
            f"  ✓ Built {n_valid} covariance matrices (filtered {n_filtered} with λ ≤ 0)"
        )
        chol_location = "CPU (recommended)" if cholesky_on_cpu else "GPU"
        print(
            f"  Computing ALL Cholesky factorizations (batched on {chol_location})..."
        )
        print(
            f"    [DEBUG] R_batch shape: {R_batch.shape}, device: {R_batch.device}, dtype: {R_batch.dtype}"
        )
        print(
            f"    [DEBUG] Target device: {device.type}, cholesky_on_cpu: {cholesky_on_cpu}"
        )
        print(f"    [DEBUG] PyTorch threads: {torch.get_num_threads()}")

    # PHASE 2: Batch Cholesky decomposition
    try:
        # Compute on CPU (default) or GPU
        # CPU is recommended for large grids to avoid OOM, and is still very fast
        import time

        chol_start = time.time()

        if verbose:
            print("    [DEBUG] Entering Cholesky computation...")

        if cholesky_on_cpu or device.type == "mps":
            # CPU Cholesky: For pure CPU mode with large matrices, chunk the batch
            # to avoid BLAS backend performance issues
            # When CUDA is available (even if computing on CPU), PyTorch uses optimized MKL
            # In pure CPU mode, it may use slower backends, so we chunk

            if verbose:
                print("    [DEBUG] In cholesky_on_cpu branch")
                print(
                    f"    [DEBUG] Checking chunking: device={device.type}, n_tp={n_timepoints}, n_valid={n_valid}"
                )
                will_chunk = (
                    device.type == "cpu" and n_timepoints > 1000 and n_valid > 50
                )
                print(f"    [DEBUG] Will use chunking: {will_chunk}")

            # Standard batch Cholesky for GPU-available mode or small grids
            if verbose:
                print("    [DEBUG] Using STANDARD batch Cholesky (non-chunked)")
                print(
                    f"    [DEBUG] About to call torch.linalg.cholesky on shape {R_batch.shape}..."
                )

            L_batch = torch.linalg.cholesky(R_batch)  # (n_valid, n, n)

            if verbose:
                print("    [DEBUG] Cholesky done, computing batch inverse...")
                print(
                    f"    [DEBUG] L_batch device: {L_batch.device}, dtype: {L_batch.dtype}"
                )

            # WORKAROUND: In pure CPU mode, batch inverse hangs!
            # Do sequential inverse instead when device=cpu
            if device.type == "cpu":
                if verbose:
                    print(
                        "    [DEBUG] Using SEQUENTIAL inverse for pure CPU mode (batch inv hangs)"
                    )
                L_inv_list = []
                from tqdm.auto import tqdm

                for i in tqdm(
                    range(n_valid),
                    desc="  Inverting matrices",
                    unit="matrix",
                    disable=not verbose,
                ):
                    L_inv_list.append(torch.linalg.inv(L_batch[i]))
                L_inv_batch = torch.stack(L_inv_list, dim=0)
            else:
                L_inv_batch = torch.linalg.inv(L_batch)  # (n_valid, n, n)

            if verbose:
                print("    [DEBUG] Batch inverse done")

            # KEEP ON CPU - will be moved to GPU ONCE after precomputation
            # This avoids OOM during grid construction (6+ GiB for large grids)
        else:
            # Direct GPU Cholesky (faster but uses more VRAM)
            # R_batch already on GPU from build_device logic above
            L_batch = torch.linalg.cholesky(R_batch)  # (n_valid, n, n)
            L_inv_batch = torch.linalg.inv(L_batch)  # (n_valid, n, n)
            # Move to CPU for storage (will be loaded to GPU ONCE after grid complete)
            L_batch = L_batch.cpu()
            L_inv_batch = L_inv_batch.cpu()

        chol_time = time.time() - chol_start

        if verbose:
            print(f"  ✓ Computed {n_valid} Cholesky factorizations! ({chol_time:.1f}s)")
            print("  Prewhitening design matrix for all parameters...")

        if debug_memory:
            _debug_memory_snapshot(
                "Grid: After Cholesky (before R_batch delete)",
                device,
                {
                    "L_batch": L_batch,
                    "L_inv_batch": L_inv_batch,
                },
            )

        # Free R_batch immediately - don't need it anymore
        del R_batch

        if debug_memory:
            _debug_memory_snapshot(
                "Grid: After R_batch deleted",
                device,
                {
                    "L_batch": L_batch,
                    "L_inv_batch": L_inv_batch,
                },
            )

        # PHASE 3: Batch prewhitening of design matrix
        # X_w[i] = L_inv[i] @ X for all i
        # Use batch matrix multiplication: (n_valid, n, n) @ (n, m) -> (n_valid, n, m)
        # Move X to same device as L_inv_batch (CPU if cholesky_on_cpu=True)
        X_build = X.to(L_inv_batch.device)
        # expand() creates a VIEW (no memory copy!)
        X_expanded = X_build.unsqueeze(0).expand(n_valid, -1, -1)  # VIEW!
        X_w_batch = torch.bmm(
            L_inv_batch, X_expanded
        )  # (n_valid, n_timepoints, n_regressors)
        del X_expanded, X_build  # Free view and temp tensor

        # PHASE 4: Compute factorization (QR or X'X depending on use_qr flag)
        if use_qr:
            # QR factorization (AFNI approach - more stable, uses more memory)
            # AFNI uses matrix_qrr(*W, D) to get R factor from QR decomposition
            # This is numerically superior to forming X'X (avoids squaring condition number)
            Q_batch, R_batch = torch.linalg.qr(
                X_w_batch
            )  # (n_valid, n_timepoints, n_regressors), (n_valid, n_regressors, n_regressors)
            del Q_batch  # Don't need Q, only R (called "D" in AFNI terminology)
            # R_batch is the upper triangular factor (AFNI's "D" matrix)
        else:
            # X'X approach (faster, less memory, good for well-conditioned matrices)
            X_w_batch_T = X_w_batch.transpose(1, 2)  # Cache transpose
            XwTXw_batch = torch.bmm(
                X_w_batch_T, X_w_batch
            )  # (n_valid, n_regressors, n_regressors)
            del X_w_batch_T
            R_batch = XwTXw_batch  # Store as R_batch for consistent naming

        # PHASE 5: Batch likelihood terms (that don't depend on voxel data)
        # logdet_Rcorr = 2 * sum(log(diag(L))) - correlation matrix determinant
        logdet_Rcorr_batch = 2 * torch.sum(
            torch.log(torch.diagonal(L_batch, dim1=1, dim2=2) + 1e-10), dim=1
        )  # (n_valid,)
        del L_batch  # Don't need L anymore, only L_inv

        # logdet(X'R^-1 X)
        if use_qr:
            # QR: logdet(X'X) = 2 * sum(log(diag(R))) where R is from QR of X_w
            # This is MUCH more stable than computing det(X'X)!
            diag_R_batch = torch.diagonal(
                R_batch, dim1=1, dim2=2
            )  # (n_valid, n_regressors)
            logdet_XwTXw_batch = 2 * torch.sum(
                torch.log(torch.abs(diag_R_batch) + 1e-10), dim=1
            )  # (n_valid,)
        else:
            # X'X: use slogdet
            sign_batch, logdet_XwTXw_batch = torch.linalg.slogdet(R_batch)
            logdet_XwTXw_batch = torch.where(
                sign_batch > 0,
                logdet_XwTXw_batch,
                torch.tensor(1e10, device=logdet_XwTXw_batch.device, dtype=dtype),
            )
            del sign_batch

        if verbose:
            print("  ✓ Precomputed all matrices!")
            print(f"  Storing {n_valid} parameter sets...")

        # PHASE 6: Store everything in dictionary ON CPU
        # Tensors are already on CPU from Phase 2 - store as-is
        # They will be moved to GPU on-demand during REML search
        for i, (a_val, b_val) in enumerate(param_list):
            precomputed[(a_val, b_val)] = {
                "L_inv": L_inv_batch[
                    i
                ],  # (n, n) - CPU tensor - inverse Cholesky of correlation matrix
                "X_w": X_w_batch[
                    i
                ],  # (n, n_regressors) - CPU tensor - prewhitened design
                "R_qr": R_batch[
                    i
                ],  # (n_regressors, n_regressors) - CPU tensor - QR's R or X'X (depending on use_qr)
                "logdet_Rcorr": logdet_Rcorr_batch[
                    i
                ],  # scalar tensor - log det of correlation matrix
                "logdet_XwTXw": logdet_XwTXw_batch[
                    i
                ],  # scalar tensor - log det of X'R^-1 X
                "use_qr": use_qr,  # bool - which approach was used
                "a": a_val,
                "b": b_val,
            }

    except RuntimeError as e:
        # Fallback to sequential if batch fails
        if verbose:
            warnings.warn(f"Batch Cholesky failed ({e}), falling back to sequential...")

        # CRITICAL: Clean up GPU memory from failed batch attempt!
        # Delete any partially-allocated batch tensors
        if "R_batch" in locals():
            del R_batch
        if "L_batch" in locals():
            del L_batch  # noqa: F821
        if "L_inv_batch" in locals():
            del L_inv_batch  # noqa: F821
        if "X_w_batch" in locals():
            del X_w_batch
        if "Q_batch" in locals():
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
            R = build_arma11_covariance(a_val, b_val, n_timepoints, device, dtype)
            if R is None:
                continue

            try:
                # Compute on CPU and store on CPU (moved to GPU on-demand later)
                if cholesky_on_cpu or device.type == "mps":
                    R_cpu = R.cpu()
                    L = torch.linalg.cholesky(R_cpu)
                    L_inv = torch.linalg.inv(L)
                    del R_cpu
                else:
                    # Even if cholesky_on_cpu=False, move to CPU for storage
                    L = torch.linalg.cholesky(R).cpu()
                    L_inv = torch.linalg.inv(L)

                # Prewhiten design on CPU
                X_cpu = X.cpu() if X.device.type != "cpu" else X
                X_w = L_inv @ X_cpu

                logdet_Rcorr = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10))

                if use_qr:
                    # QR factorization (AFNI's matrix_qrr)
                    Q, R_qr = torch.linalg.qr(X_w)
                    logdet_XwTXw = 2 * torch.sum(
                        torch.log(torch.abs(torch.diag(R_qr)) + 1e-10)
                    )
                else:
                    # X'X approach
                    XwTXw = X_w.T @ X_w
                    R_qr = XwTXw  # Store as R_qr for consistent naming
                    sign, logdet_XwTXw = torch.linalg.slogdet(XwTXw)
                    logdet_XwTXw = (
                        logdet_XwTXw
                        if sign > 0
                        else torch.tensor(1e10, device=torch.device("cpu"), dtype=dtype)
                    )

                # Store on CPU - will be moved to GPU on-demand during REML search
                precomputed[(a_val, b_val)] = {
                    "L_inv": L_inv,  # CPU tensor
                    "X_w": X_w,  # CPU tensor
                    "R_qr": R_qr,  # CPU tensor - QR's R or X'X
                    "logdet_Rcorr": logdet_Rcorr,  # CPU tensor
                    "logdet_XwTXw": logdet_XwTXw,  # CPU tensor
                    "use_qr": use_qr,  # bool
                    "a": a_val,
                    "b": b_val,
                }
            except RuntimeError as e2:
                if verbose:
                    warnings.warn(
                        f"Cholesky failed for (a={a_val:.3f}, b={b_val:.3f}): {e2}"
                    )
                continue

    # CRITICAL: Delete batch tensors to free memory before GPU transfer
    # Each precomputed[(a,b)] entry holds a VIEW into these batches
    # If we don't delete them, moving views to GPU keeps batch tensors alive
    if debug_memory:
        _debug_memory_snapshot(
            "Grid: Before deleting batch tensors",
            device,
            {
                "L_inv_batch": locals().get("L_inv_batch"),
                "X_w_batch": locals().get("X_w_batch"),
                "R_batch": locals().get("R_batch"),
            },
        )

        # Calculate expected grid size
        if "L_inv_batch" in locals():
            n_pairs = len(param_list)
            n_regs = X.shape[1]
            bytes_per_elem = L_inv_batch.element_size()

            expected_L_inv = n_pairs * n_timepoints * n_timepoints * bytes_per_elem
            expected_X_w = n_pairs * n_timepoints * n_regs * bytes_per_elem
            expected_R_qr = n_pairs * n_regs * n_regs * bytes_per_elem
            expected_scalars = n_pairs * 2 * bytes_per_elem
            expected_total = (
                expected_L_inv + expected_X_w + expected_R_qr + expected_scalars
            )

            actual_L_inv = L_inv_batch.element_size() * L_inv_batch.nelement()
            actual_X_w = X_w_batch.element_size() * X_w_batch.nelement()
            actual_R_qr = R_batch.element_size() * R_batch.nelement()
            actual_total = actual_L_inv + actual_X_w + actual_R_qr

            print("\nGRID SIZE VERIFICATION:")
            print(
                f"  n_pairs={n_pairs}, n_timepoints={n_timepoints}, n_regressors={n_regs}, dtype={dtype}"
            )
            print(f"  Expected grid size: {expected_total / 1024**3:.3f} GiB")
            print(f"    - L_inv:  {expected_L_inv / 1024**3:.3f} GiB")
            print(f"    - X_w:    {expected_X_w / 1024**3:.3f} GiB")
            print(f"    - R_qr:   {expected_R_qr / 1024**3:.3f} GiB")
            print(f"  Actual batch tensors: {actual_total / 1024**3:.3f} GiB")
            print(
                f"  Match: {'✅ YES' if abs(actual_total - expected_total) < 1024**2 else '❌ NO'}"
            )

    if "L_inv_batch" in locals():
        del L_inv_batch
    if "X_w_batch" in locals():
        del X_w_batch
    if "R_batch" in locals():
        del R_batch
    if "logdet_Rcorr_batch" in locals():
        del logdet_Rcorr_batch
    if "logdet_XwTXw_batch" in locals():
        del logdet_XwTXw_batch
    if "diag_R_batch" in locals():
        del diag_R_batch

    if debug_memory:
        _debug_memory_snapshot(
            "Grid: After deleting batch tensors (grid in dict only)", device, {}
        )

    return precomputed


def _cpu_hierarchical_reml_search(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    a_grid: torch.Tensor,
    b_grid: torch.Tensor,
    precomputed: dict,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-OPTIMIZED hierarchical REML grid search with per-voxel early stopping.

    This implements AFNI's power-of-2 descent strategy (step=8→4→2→1) which
    works efficiently on CPU because each voxel can narrow its search window
    independently. This approach does NOT work on GPU batch processing where
    all voxels must evaluate the same grid points.

    **AFNI-style algorithm:**
    1. Start at step=8: evaluate grid at (0,0), (0,8), (8,0), (8,8), etc.
    2. Find best for THIS voxel
    3. Narrow window around best ± step
    4. Reduce step to 4, then 2, then 1
    5. Early stop if no improvement

    **Why CPU-only:**
    - CPUs process voxels sequentially (or in small thread batches)
    - Each voxel can have independent search trajectory
    - Window narrowing works per-voxel
    - Evaluates ~40-50 grid points instead of 117 (2-3x speedup)

    **Why NOT GPU:**
    - GPUs process large batches (1000s of voxels) in parallel
    - All voxels must evaluate same grid points
    - Can't narrow per-voxel without losing parallelism
    - Would evaluate all 117 points anyway with extra overhead

    Parameters
    ----------
    X : torch.Tensor, shape (n_timepoints, n_regressors)
        Design matrix
    Y_batch : torch.Tensor, shape (n_timepoints, n_voxels_batch)
        Data for batch of voxels
    a_grid : torch.Tensor
        AR parameter grid (sorted)
    b_grid : torch.Tensor
        MA parameter grid (sorted)
    precomputed : dict
        Precomputed REML matrices for all (a,b) pairs
    dtype : torch.dtype
        Data type for computation

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels_batch, 2)
        Optimal (a, b) for each voxel
    best_likelihoods : torch.Tensor, shape (n_voxels_batch,)
        Minimum likelihood for each voxel
    """
    device = X.device
    n_timepoints, n_voxels_batch = Y_batch.shape
    n_regressors = X.shape[1]

    # Sort grids and create lookup
    a_vals_sorted = torch.sort(a_grid)[0]
    b_vals_sorted = torch.sort(b_grid)[0]

    # Create index mapping for (a_idx, b_idx) -> precomputed key
    param_list = list(precomputed.keys())
    param_to_idx = {params: idx for idx, params in enumerate(param_list)}

    def get_param_key(a_idx: int, b_idx: int):
        """Get parameter key if it exists in grid"""
        if a_idx < 0 or a_idx >= len(a_vals_sorted):
            return None
        if b_idx < 0 or b_idx >= len(b_vals_sorted):
            return None
        a_val = a_vals_sorted[a_idx].item()
        b_val = b_vals_sorted[b_idx].item()
        # Find closest match in precomputed (handle floating point)
        for key in param_list:
            if abs(key[0] - a_val) < 1e-6 and abs(key[1] - b_val) < 1e-6:
                return key
        return None

    # Initialize results
    best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=dtype)
    best_likelihoods = torch.full(
        (n_voxels_batch,), float("inf"), device=device, dtype=dtype
    )

    # Process each voxel independently (the CPU way!)
    # Show progress bar for CPU since it processes voxels sequentially
    from tqdm.auto import tqdm

    voxel_iterator = range(n_voxels_batch)
    if n_voxels_batch > 100:  # Only show for reasonable batch sizes
        voxel_iterator = tqdm(voxel_iterator, desc="CPU REML search", unit="voxel")

    for voxel_idx in voxel_iterator:
        Y_voxel = Y_batch[:, voxel_idx : voxel_idx + 1]  # (n_timepoints, 1)

        # Initialize search at (0, 0) if available
        zero_key = get_param_key(0, 0)
        if zero_key is not None:
            lik = _evaluate_single_param(
                X, Y_voxel, zero_key, precomputed, n_timepoints, n_regressors
            )
            best_likelihoods[voxel_idx] = lik
            best_params[voxel_idx, 0] = zero_key[0]
            best_params[voxel_idx, 1] = zero_key[1]

        # Current best indices
        best_a_idx = 0
        best_b_idx = 0

        # Power-of-2 descent: 8 → 4 → 2 → 1
        max_level = max(0, min(len(a_vals_sorted), len(b_vals_sorted)).bit_length() - 2)

        for level in range(max_level, -1, -1):
            step = 1 << level  # 2^level

            # Determine search window around current best
            # For first iteration (step=8), search entire grid
            # For later iterations, narrow around best ± 2*step
            if level == max_level:
                a_min, a_max = 0, len(a_vals_sorted) - 1
                b_min, b_max = 0, len(b_vals_sorted) - 1
            else:
                window_size = step * 3  # Search ± 3*step around best
                a_min = max(0, best_a_idx - window_size)
                a_max = min(len(a_vals_sorted) - 1, best_a_idx + window_size)
                b_min = max(0, best_b_idx - window_size)
                b_max = min(len(b_vals_sorted) - 1, best_b_idx + window_size)

            # Evaluate grid points at this step size within window
            improved = False
            for b_idx in range(b_min, b_max + 1, step):
                for a_idx in range(a_min, a_max + 1, step):
                    # Skip (0,0) if we already evaluated it
                    if level == max_level and a_idx == 0 and b_idx == 0:
                        continue

                    param_key = get_param_key(a_idx, b_idx)
                    if param_key is None:
                        continue

                    lik = _evaluate_single_param(
                        X, Y_voxel, param_key, precomputed, n_timepoints, n_regressors
                    )

                    if lik < best_likelihoods[voxel_idx]:
                        best_likelihoods[voxel_idx] = lik
                        best_params[voxel_idx, 0] = param_key[0]
                        best_params[voxel_idx, 1] = param_key[1]
                        best_a_idx = a_idx
                        best_b_idx = b_idx
                        improved = True

            # Early stopping: if no improvement at this level, done
            if not improved and level < max_level:
                break

    return best_params, best_likelihoods


def _evaluate_single_param(
    X: torch.Tensor,
    Y_voxel: torch.Tensor,
    param_key: Tuple[float, float],
    precomputed: dict,
    n_timepoints: int,
    n_regressors: int,
) -> float:
    """
    Evaluate REML likelihood for a single voxel at single (a,b) parameter.

    Helper function for CPU hierarchical search.
    """
    # Get precomputed matrices
    L_inv = precomputed[param_key]["L_inv"]  # (n_time, n_time)
    X_w = precomputed[param_key]["X_w"]  # (n_time, n_reg)
    R_qr = precomputed[param_key]["R_qr"]  # (n_reg, n_reg)
    logdet_Rcorr = precomputed[param_key]["logdet_Rcorr"]  # scalar
    logdet_XwTXw = precomputed[param_key]["logdet_XwTXw"]  # scalar
    use_qr = precomputed[param_key]["use_qr"]

    # Prewhiten data
    Y_w = L_inv @ Y_voxel  # (n_time, 1)

    # Compute beta
    XwTYw = X_w.T @ Y_w  # (n_reg, 1)
    if use_qr:
        beta_w = torch.linalg.solve_triangular(R_qr, XwTYw, upper=True)
    else:
        beta_w = torch.linalg.solve(R_qr, XwTYw)

    # Compute residuals and RSS
    pred_w = X_w @ beta_w
    resid_w = Y_w - pred_w
    rss = torch.sum(resid_w**2).item()

    # Compute REML likelihood
    likelihood = (
        logdet_Rcorr.item()
        + logdet_XwTXw.item()
        + (n_timepoints - n_regressors) * float(torch.log(torch.tensor(rss + 1e-10)))
    )

    return likelihood


def batch_reml_grid_search(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    a_grid: Optional[torch.Tensor] = None,
    b_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    precomputed: Optional[dict] = None,
    dtype: torch.dtype = torch.float32,
    enable_early_stopping: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Device-adaptive REML grid search with optimal strategy per device.

    **DEVICE-SPECIFIC OPTIMIZATION:**

    **CPU Mode** (device.type == 'cpu'):
    - Uses hierarchical search with per-voxel early stopping (AFNI-style)
    - Power-of-2 descent: step=8→4→2→1
    - Each voxel narrows search window independently
    - Evaluates ~40-50 grid points instead of 117 (2-3x speedup)
    - Why: CPUs process voxels sequentially, can optimize per-voxel

    **GPU Mode** (device.type in ['cuda', 'mps']):
    - Uses exhaustive batch-parallel search
    - Evaluates ALL grid points for ALL voxels in massive parallel operation
    - Automatically chunks grid if memory usage would be too high
    - Why: GPUs process 1000s of voxels in parallel, all must evaluate same grid
    - Hierarchical doesn't work here (can't narrow per-voxel without losing parallelism)

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
            X, n_timepoints, a_grid, b_grid, device, dtype=dtype
        )

    n_grid = len(precomputed)
    if n_grid == 0:
        # No valid grid points - return defaults
        best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=dtype)
        best_params[:, 0] = 0.5
        best_params[:, 1] = 0.0
        best_likelihoods = torch.full(
            (n_voxels_batch,), float("inf"), device=device, dtype=dtype
        )
        return best_params, best_likelihoods

    # DEVICE-SPECIFIC SEARCH STRATEGY
    # CPU: Use hierarchical search with per-voxel early stopping (AFNI-style)
    # GPU: Use exhaustive batch parallel search (GPU-optimized)
    if device.type == "cpu":
        # CPU-OPTIMIZED: Hierarchical search with early stopping
        # Each voxel can narrow its search window independently
        # Evaluates ~40-50 grid points instead of 117 (2-3x speedup)
        return _cpu_hierarchical_reml_search(
            X, Y_batch, a_grid, b_grid, precomputed, dtype=dtype
        )

    # EXHAUSTIVE GRID SEARCH - GPU Batch Parallel Approach
    # Unlike AFNI (sequential per-voxel), we process ALL voxels in parallel
    # This means we must evaluate the SAME grid points for ALL voxels
    # Hierarchical search doesn't help because we can't narrow per-voxel
    # Solution: Evaluate all ~117 grid points in massive parallel operation
    # This is what GPUs are designed for!

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
        n_timepoints * n_timepoints +  # L_inv (dominant!)
        n_timepoints * n_regressors +  # X_w
        n_timepoints * n_voxels_batch * 5  # Y_w, XwTYw, beta, pred, resid (scaled)
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
    best_likelihoods = torch.full(
        (n_voxels_batch,), float("inf"), device=device, dtype=dtype
    )

    # Set background voxels to (0,0) immediately and mark as "done"
    if n_background > 0:
        best_params[is_background, 0] = 0.0
        best_params[is_background, 1] = 0.0
        best_likelihoods[is_background] = -1.0  # Sentinel value (skip in search)

    # Only search non-background voxels
    n_searchable = n_voxels_batch - n_background

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
            ordered_indices[i : i + max_grid_chunk]
            for i in range(0, n_grid, max_grid_chunk)
        ]

    n_chunks_evaluated = 0
    for chunk_idx, chunk_indices in enumerate(grid_chunks):
        # PHASE 1: Stack precomputed matrices for THIS CHUNK only
        # Note: precomputed tensors may be on CPU (when cholesky_on_cpu=True)
        # We move them to the target device after stacking
        chunk_keys = [param_list[i] for i in chunk_indices]

        L_inv_stack = torch.stack(
            [precomputed[k]["L_inv"] for k in chunk_keys]
        ).to(device)  # (n_chunk, n_time, n_time)
        X_w_stack = torch.stack(
            [precomputed[k]["X_w"] for k in chunk_keys]
        ).to(device)  # (n_chunk, n_time, n_regressors)
        R_qr_stack = torch.stack(
            [precomputed[k]["R_qr"] for k in chunk_keys]
        ).to(device)  # (n_chunk, n_reg, n_reg) - upper triangular from QR
        logdet_Rcorr_stack = torch.stack(
            [precomputed[k]["logdet_Rcorr"] for k in chunk_keys]
        ).to(device)  # (n_chunk,)
        logdet_XwTXw_stack = torch.stack(
            [precomputed[k]["logdet_XwTXw"] for k in chunk_keys]
        ).to(device)  # (n_chunk,)

        # PHASE 2: Prewhiten data for THIS CHUNK
        # Broadcast: (n_chunk, n_time, n_time) @ (n_time, n_voxels) -> (n_chunk, n_time, n_voxels)
        Y_batch_expanded = Y_batch.unsqueeze(0).expand(
            len(chunk_indices), -1, -1
        )  # VIEW - no VRAM copy!
        Y_w_all = torch.bmm(
            L_inv_stack, Y_batch_expanded
        )  # (n_chunk, n_time, n_voxels)
        del Y_batch_expanded

        # PHASE 3: Compute betas for this chunk
        XwTYw_all = torch.bmm(
            X_w_stack.transpose(1, 2), Y_w_all
        )  # (n_chunk, n_reg, n_voxels)

        # Check if using QR or X'X (all entries in chunk should have same use_qr)
        use_qr = precomputed[chunk_keys[0]]["use_qr"]

        if use_qr:
            # QR approach: triangular solve (upper triangular R)
            beta_w_all = torch.linalg.solve_triangular(
                R_qr_stack,  # (n_chunk, n_reg, n_reg) - upper triangular
                XwTYw_all,  # (n_chunk, n_reg, n_voxels)
                upper=True,
            )
        else:
            # X'X approach: general solve (symmetric positive definite)
            beta_w_all = torch.linalg.solve(
                R_qr_stack,  # (n_chunk, n_reg, n_reg) - X'X matrix
                XwTYw_all,  # (n_chunk, n_reg, n_voxels)
            )
        del XwTYw_all

        # PHASE 4: Compute residuals and RSS
        pred_w_all = torch.bmm(X_w_stack, beta_w_all)
        residuals_w_all = Y_w_all - pred_w_all
        del Y_w_all, pred_w_all

        rss_all = torch.sum(residuals_w_all**2, dim=1)  # (n_chunk, n_voxels)
        del residuals_w_all

        # PHASE 5: Compute likelihoods for this chunk
        term1 = logdet_Rcorr_stack.unsqueeze(1)  # (n_chunk, 1)
        term2 = logdet_XwTXw_stack.unsqueeze(1)  # (n_chunk, 1)
        term3 = (n_timepoints - n_regressors) * torch.log(rss_all + 1e-10)
        del rss_all

        chunk_likelihoods = term1 + term2 + term3  # (n_chunk, n_voxels)
        del term1, term2, term3, L_inv_stack, X_w_stack, R_qr_stack
        del logdet_Rcorr_stack, logdet_XwTXw_stack, beta_w_all

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

        # Map chunk indices to global param list indices
        for voxel_idx in torch.where(improve_mask)[0]:
            chunk_param_idx = chunk_best_idx[voxel_idx].item()
            global_param_idx = chunk_indices[chunk_param_idx]
            a_val, b_val = param_list[global_param_idx]
            best_params[voxel_idx, 0] = a_val
            best_params[voxel_idx, 1] = b_val

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
                converged = (rel_improvement < 0.001) | (
                    improvement[searchable_mask] == 0
                )
                convergence_rate = converged.float().mean().item()

                # If >80% of searchable voxels converged, stop early
                # With smart ordering, most voxels find optimal (a,b) in first 10-15 evals
                if (
                    convergence_rate > 0.80 and chunk_idx > 0
                ):  # At least eval 2 chunks (20 params)
                    n_remaining = len(grid_chunks) - n_chunks_evaluated
                    # Note: Only stop if we have more chunks to skip
                    if n_remaining > 0:
                        break

    return best_params, best_likelihoods


def search_voxels_precomputed_grid(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    precomputed_grid: Dict,
    device: torch.device,
    verbose: bool = False,
    enable_timing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        - 'L_inv': Cholesky factor inverse
        - 'X_w': Prewhitened design matrix
        - 'R_qr': QR decomposition of X_w
        - 'logdet_Rcorr': log determinant of correlation matrix
        - 'logdet_XwTXw': log determinant of X'X
    device : torch.device
        Device for computation
    verbose : bool
        Print progress information
    enable_timing : bool
        Enable detailed timing profiling

    Returns
    -------
    best_params : torch.Tensor, shape (n_voxels_batch, 2)
        Optimal (a, b) for each voxel
    best_likelihoods : torch.Tensor, shape (n_voxels_batch,)
        Minimum REML likelihood for each voxel
    """
    from .timing_utils import profile_section

    n_voxels_batch = Y_batch.shape[0]
    n_timepoints, n_regressors = X.shape
    n_grid = len(precomputed_grid)

    # Initialize results
    best_params = torch.zeros(n_voxels_batch, 2, device=device)
    best_likelihoods = torch.full((n_voxels_batch,), float('inf'), device=device)

    # Extract grid keys and precomputed values
    param_list = list(precomputed_grid.keys())

    # Move Y_batch to device if needed
    if Y_batch.device != device:
        Y_batch = Y_batch.to(device)

    # Check which approach was used (all grid entries should have same use_qr)
    use_qr = precomputed_grid[param_list[0]]["use_qr"]

    # Evaluate each grid point
    for grid_idx, (a, b) in enumerate(param_list):
        with profile_section("1_get_grid_data", enabled=enable_timing):
            grid_data = precomputed_grid[(a, b)]

            # Get precomputed matrices (already on GPU)
            L_inv = grid_data['L_inv']
            X_w = grid_data['X_w']
            R_qr = grid_data['R_qr']
            logdet_Rcorr = grid_data['logdet_Rcorr']
            logdet_XwTXw = grid_data['logdet_XwTXw']

        # Prewhiten data: Y_w = L_inv @ Y for all voxels
        with profile_section("2_prewhiten_data", enabled=enable_timing):
            # Y_batch: (n_voxels, n_timepoints)
            # L_inv: (n_timepoints, n_timepoints)
            # Result: (n_voxels, n_timepoints)
            Y_w_batch = (Y_batch @ L_inv.T)  # More efficient than L_inv @ Y_batch.T

        # Solve X_w @ beta = Y_w for each voxel
        with profile_section("3_compute_XwTYw", enabled=enable_timing):
            # Compute X_w' Y_w
            XwT_Yw = X_w.T @ Y_w_batch.T  # (n_regressors, n_voxels)

        # Solve for beta using appropriate method
        with profile_section("4_solve_beta", enabled=enable_timing):
            if use_qr:
                # QR approach: R' R beta = X_w' Y_w (triangular solves)
                # R_qr is upper triangular R factor from QR decomposition
                beta_temp = torch.linalg.solve_triangular(
                    R_qr.T, XwT_Yw, upper=False
                )
                beta_batch = torch.linalg.solve_triangular(
                    R_qr, beta_temp, upper=True
                )
            else:
                # X'X approach: (X_w' X_w) beta = X_w' Y_w (general solve)
                # R_qr is actually X_w' X_w (symmetric positive definite)
                beta_batch = torch.linalg.solve(
                    R_qr, XwT_Yw
                )
            beta_batch = beta_batch.T  # (n_voxels, n_regressors)

        # Compute prewhitened residuals
        with profile_section("5_compute_residuals", enabled=enable_timing):
            pred_w = (X_w @ beta_batch.T).T  # (n_voxels, n_timepoints)
            resid_w = Y_w_batch - pred_w
            rss_batch = torch.sum(resid_w ** 2, dim=1)  # (n_voxels,)

        # Compute REML likelihood for this (a, b)
        with profile_section("6_compute_likelihood", enabled=enable_timing):
            # L = log(det(R)) + log(det(X'R^-1 X)) + (n-m) log(RSS)
            term1 = logdet_Rcorr  # Scalar, same for all voxels
            term2 = logdet_XwTXw  # Scalar, same for all voxels
            term3 = (n_timepoints - n_regressors) * torch.log(rss_batch + 1e-10)  # (n_voxels,)

            likelihoods = term1 + term2 + term3  # (n_voxels,)

        # Update best parameters where this (a, b) is better
        with profile_section("7_update_best", enabled=enable_timing):
            improve_mask = likelihoods < best_likelihoods
            best_likelihoods[improve_mask] = likelihoods[improve_mask]
            best_params[improve_mask, 0] = a
            best_params[improve_mask, 1] = b

    return best_params, best_likelihoods


def prewhiten_with_arma11(
    X: torch.Tensor, Y: torch.Tensor, a: float, b: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    L_inv : torch.Tensor
        Prewhitening matrix (Cholesky factor inverse)

    Notes
    -----
    Prewhitening transformation:
    - R = L L' (Cholesky)
    - X* = L^(-1) X
    - Y* = L^(-1) Y

    After prewhitening, OLS on (X*, Y*) = GLS on (X, Y).

    GPU Speed: ~10ms for n=300 timepoints (Cholesky is highly optimized in PyTorch)

    Memory optimization: For large n_timepoints (e.g., 17140), the covariance matrix
    is huge (17140^2 = 2+ GB). Building on CPU avoids GPU fragmentation and OOM errors.
    Cholesky on CPU is fast, and only L_inv is moved to GPU (much smaller after inversion).
    """
    device = X.device
    n_timepoints = X.shape[0]

    # Build ARMA(1,1) covariance on CPU to save GPU memory and avoid fragmentation
    # For large n_timepoints, building on CPU avoids initial 2+ GB allocation spike
    R = build_arma11_covariance(a, b, n_timepoints, torch.device("cpu"))

    if R is None:
        raise ValueError(f"Invalid ARMA(1,1) parameters: a={a}, b={b}")

    # Try GPU Cholesky (fast), fall back to CPU if OOM (safe)
    try:
        # Compact GPU memory before attempting allocation (minimal cost, big benefit!)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Ensure all ops complete
            torch.cuda.empty_cache()  # Release fragmented memory

        # Move R to GPU temporarily for fast Cholesky
        R_gpu = R.to(device)

        # Cholesky on GPU (10-100x faster than CPU!)
        L = torch.linalg.cholesky(R_gpu)
        del R_gpu  # Free GPU R (but keep CPU R for potential fallback)

        # Compute L_inv on GPU
        identity = torch.eye(n_timepoints, dtype=L.dtype, device=device)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
        del L  # Free GPU L
        del R  # Success - can now delete CPU R

    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # GPU OOM - fall back to CPU (slower but safe)
        # R still exists here since we didn't delete it in try block

        # CRITICAL: Clean up GPU memory from failed try block
        # If exception occurred, L and identity may still be on GPU
        try:
            del L
        except NameError:
            pass
        try:
            del identity
        except NameError:
            pass

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Cholesky on CPU
        # R exists here from line 2382 (only deleted if try succeeds at line 2405)
        L_cpu = torch.linalg.cholesky(R)  # noqa: F821

        # Compute L_inv on CPU
        identity_cpu = torch.eye(n_timepoints, dtype=L_cpu.dtype, device=torch.device("cpu"))
        L_inv_cpu = torch.linalg.solve_triangular(L_cpu, identity_cpu, upper=False)
        del L_cpu  # Free CPU memory

        # Move only L_inv to GPU (smaller than R)
        L_inv = L_inv_cpu.to(device)
        del L_inv_cpu, R  # Cleanup both  # noqa: F821

    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Prewhiten using L_inv (on target device, matching X and Y device)
    X_white = L_inv @ X
    Y_white = L_inv @ Y

    if Y_white.shape[1] == 1:
        Y_white = Y_white.squeeze(1)

    return X_white, Y_white, L_inv


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
        print(f"\n{'='*70}")
        print("ADAPTIVE BATCHING STRATEGY")
        print(f"{'='*70}")
        print(f"  Grid: {n_grid_points} points × {grid_mem_per_point/1024**2:.1f} MB = {grid_mem_total/1024**3:.2f} GB")
        print(f"  Data: {n_voxels} voxels × {n_timepoints} TPs = {data_mem/1024**3:.2f} GB")
        print(f"  Available memory: {available/1024**3:.2f} GB")
        print()

    # Decision tree
    if grid_mem_total + mem_per_voxel_batch(n_voxels * 0.1) < available:
        # Case 1: Full grid + voxel batches (AFNI approach - FASTEST!)
        # Leave 30% for voxel batches
        mem_for_voxels = available - grid_mem_total
        voxel_batch_size = int(mem_for_voxels / (n_timepoints * bytes_per_elem * batch_overhead_factor))
        voxel_batch_size = max(500, min(voxel_batch_size, n_voxels))

        strategy = {
            "strategy": "full_grid",
            "grid_chunk_size": n_grid_points,
            "voxel_batch_size": voxel_batch_size,
            "reason": f"Grid fits ({grid_mem_total/1024**3:.1f}GB) - using AFNI approach (fastest!)"
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
            "reason": f"Data fits ({data_mem/1024**3:.1f}GB), batching grid in chunks of {grid_chunk_size}"
        }

    else:
        # Case 3: Hybrid - batch BOTH grid and voxels
        # Split available memory: 40% grid, 40% data, 20% overhead
        mem_for_grid = available * 0.4
        mem_for_data = available * 0.4

        grid_chunk_size = int(mem_for_grid / grid_mem_per_point)
        grid_chunk_size = max(1, min(grid_chunk_size, 10, n_grid_points))

        voxel_batch_size = int(mem_for_data / (n_timepoints * bytes_per_elem * batch_overhead_factor))
        voxel_batch_size = max(100, min(voxel_batch_size, n_voxels))

        strategy = {
            "strategy": "hybrid",
            "grid_chunk_size": grid_chunk_size,
            "voxel_batch_size": voxel_batch_size,
            "reason": f"Both large - grid chunks of {grid_chunk_size}, voxel batches of {voxel_batch_size:,}"
        }

    if verbose:
        print(f"  Strategy: {strategy['strategy']}")
        print(f"    Grid chunk size: {strategy['grid_chunk_size']}")
        print(f"    Voxel batch size: {strategy['voxel_batch_size']:,}")
        print(f"    Reason: {strategy['reason']}")
        print(f"{'='*70}\n")

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
    voxel_batch_size: Optional[int] = None,
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
        design_dev = design.to(device, dtype=dtype)
    else:
        # Keep data on CPU, will move batches as needed
        design_dev = design.to(device, dtype=dtype)

    # Allocate results
    best_params = torch.zeros(n_voxels, 2, dtype=dtype, device=torch.device("cpu"))
    best_likelihood = torch.full((n_voxels,), float("-inf"), dtype=dtype, device=torch.device("cpu"))

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
                    dtype=dtype
                )

                # Compute Cholesky
                L = torch.linalg.cholesky(R)  # On CPU

                # Compute L_inv efficiently
                identity = torch.eye(n_timepoints, dtype=dtype)
                L_inv_cpu = torch.linalg.solve_triangular(L, identity, upper=False)

                # Move to GPU
                L_inv = L_inv_cpu.to(device)

                # Prewhiten design (same for all voxels)
                X_w = L_inv @ design_dev  # (n_tp, n_reg)
                XwTXw = X_w.T @ X_w
                XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device, dtype=dtype)

                # Log determinants for REML
                logdet_R = 2.0 * torch.sum(torch.log(torch.diag(L)))
                logdet_XwTXw = torch.logdet(XwTXw_reg)

                # Process voxels (either all at once or in batches)
                if load_all_data:
                    # All data already on device - process at once
                    # Prewhiten ALL voxels at once (matrix multiply!)
                    y_w = data_dev @ L_inv.T  # (n_voxels, n_tp)

                    # Solve for ALL voxels: beta = (X'X)^-1 X'y
                    XwTy_w = X_w.T @ y_w.T  # (n_reg, n_voxels)
                    betas = torch.linalg.solve(XwTXw_reg, XwTy_w).T  # (n_voxels, n_reg)

                    # Compute residuals and SSE
                    resid_w = y_w - (betas @ X_w.T)  # (n_voxels, n_tp)
                    sse = torch.sum(resid_w ** 2, dim=1)  # (n_voxels,)

                    # REML likelihood (same formula as AFNI)
                    n = n_timepoints
                    p = n_regressors
                    likelihoods = -0.5 * (
                        n * torch.log(sse / n)
                        + logdet_R
                        + logdet_XwTXw
                        + n * (1 + torch.log(torch.tensor(2 * torch.pi, dtype=dtype, device=device)))
                    )

                    # Update best parameters (vectorized comparison!)
                    likelihoods_cpu = likelihoods.cpu()
                    mask = likelihoods_cpu > best_likelihood
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

                        # Prewhiten this batch
                        y_w = data_batch @ L_inv.T  # (batch_voxels, n_tp)

                        # Solve for this batch: beta = (X'X)^-1 X'y
                        XwTy_w = X_w.T @ y_w.T  # (n_reg, batch_voxels)
                        betas = torch.linalg.solve(XwTXw_reg, XwTy_w).T  # (batch_voxels, n_reg)

                        # Compute residuals and SSE
                        resid_w = y_w - (betas @ X_w.T)  # (batch_voxels, n_tp)
                        sse = torch.sum(resid_w ** 2, dim=1)  # (batch_voxels,)

                        # REML likelihood (same formula as AFNI)
                        n = n_timepoints
                        p = n_regressors
                        likelihoods = -0.5 * (
                            n * torch.log(sse / n)
                            + logdet_R
                            + logdet_XwTXw
                            + n * (1 + torch.log(torch.tensor(2 * torch.pi, dtype=dtype, device=device)))
                        )

                        # Update best parameters for this batch (vectorized comparison!)
                        likelihoods_cpu = likelihoods.cpu()
                        batch_best_likelihood = best_likelihood[voxel_start:voxel_end]
                        mask = likelihoods_cpu > batch_best_likelihood

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
                del L_inv, X_w, XwTXw, XwTXw_reg

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


def fit_glm_arma11(
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    tr: float,
    a_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    b_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    estimate_per_voxel: bool = True,
    batch_size: Optional[int] = None,
    want_residuals: bool = False,
    want_predicted: bool = False,
    want_r2_partial: bool = False,
    r2_partial_mode: str = "full",  # "full" or "task" - how to compute partial R²
    want_r2_semipartial: bool = False,
    r2_semipartial_mode: str = "full",  # "full" or "task" - how to compute semi-partial R²
    want_ols: bool = False,
    ols_write_callback: Optional[Callable] = None,
    precomputed_arma_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    cholesky_on_cpu: bool = True,
    use_double: bool = False,
    use_qr: bool = False,
    debug_memory: bool = False,
    enable_quick_estimate: bool = False,
    glt_labels: Optional[List[str]] = None,
    glt_matrices: Optional[List[np.ndarray]] = None,
    task_indices: Optional[List[int]] = None,
    use_grid_batching: Optional[bool] = None,
    spatial_metadata: Optional[dict] = None,
    legacy_contrasts: bool = False,
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
    dtype = torch.float64 if use_double else torch.float32

    if device is None:
        device = get_device()
    storage_device = torch.device("cpu")

    # Enable cuDNN benchmarking for optimal kernel selection (GPU only)
    # This auto-tunes algorithms for better performance (~5-15% speedup)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
    
    # Validate design matrix
    if design.shape[0] != n_timepoints:
        raise ValueError(
            f"Timepoints mismatch: Data has {n_timepoints}, "
            f"Design has {design.shape[0]}"
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
                temp_a_grid = (
                    a_grid if a_grid is not None else get_default_arma_grids(device)[0]
                )
                temp_b_grid = (
                    b_grid if b_grid is not None else get_default_arma_grids(device)[1]
                )

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
                    print(f"\n💡 Auto-enabling grid batching (grid: {grid_memory_bytes/1024**3:.1f} GB > {GRID_BATCHING_THRESHOLD_GB} GB threshold)")
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
                print(
                    f"    scalars: {scalars_bytes / 1024**2:.1f} MiB  ({n_valid_pairs} × 2)"
                )
                print(f"  TOTAL EXPECTED: {grid_memory_bytes / 1024**3:.3f} GiB")
                print(f"{'=' * 70}\n")

            # Reduce batch size to leave room for grid
            if device.type == "cuda":
                total_mem = torch.cuda.get_device_properties(device).total_memory
                # CONSERVATIVE: Use only 60% of GPU memory to leave room for grid + batches
                # With large design matrices (many regressors), grid can be 10+ GB!
                available_mem = total_mem * 0.6
            else:  # CPU
                try:
                    import psutil

                    # Use 50% of available RAM for CPU
                    available_mem = psutil.virtual_memory().available * 0.5
                except ImportError:
                    # Fallback if psutil not available
                    available_mem = 16 * 1024**3  # Assume 16GB available

            memory_for_batches = available_mem - grid_memory_bytes

            if memory_for_batches > 0:
                bytes_per_element = 8 if use_double else 4
                mem_per_voxel = (
                    n_timepoints * n_regressors * bytes_per_element  # X_w_batch
                    + n_timepoints
                    * n_regressors
                    * bytes_per_element  # Q_batch (temporary during QR!)
                    + n_timepoints * bytes_per_element  # y_w_batch
                    + n_regressors * n_regressors * bytes_per_element  # R_qr_batch
                    + n_regressors * n_regressors * bytes_per_element  # eye_batch
                    + n_regressors * n_regressors * bytes_per_element  # R_inv_batch
                    + n_regressors * n_regressors * bytes_per_element  # XwTXw_inv_batch
                    + n_regressors * bytes_per_element  # betas
                    + n_regressors * bytes_per_element  # tstats
                    + 3 * bytes_per_element  # params (a, b, lambda)
                )
                batch_size = int(memory_for_batches / mem_per_voxel)
                batch_size = max(100, min(batch_size, base_batch_size))
            else:
                # Grid too large! This shouldn't happen if auto-detection worked
                # Only possible if user forced use_grid_batching=False
                if use_grid_batching:
                    # Grid batching should handle this - continue
                    batch_size = base_batch_size
                else:
                    raise MemoryError(
                        f"\n{'='*70}\n"
                        f"ERROR: Grid too large for available GPU memory!\n"
                        f"{'='*70}\n"
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
                        f"{'='*70}\n"
                    )

            if verbose:
                grid_mb = grid_memory_bytes / (1024**2)
                if use_grid_batching:
                    print(f"Grid size: {n_valid_pairs} (a,b) pairs")
                    print("Strategy: Grid batching (low memory, process all voxels per grid point)")
                    print(f"  Memory per grid point: ~{grid_memory_bytes/n_valid_pairs/1024**2:.1f} MiB")
                    print(f"  Batch size: {batch_size:,} voxels")
                else:
                    print(f"Grid memory: {grid_mb:.1f} MiB ({n_valid_pairs} pairs)")
                    print("Strategy: Full grid precomputation (AFNI approach)")
                    print(
                        f"Adjusted batch size: {batch_size:,} voxels (grid + batches fit in GPU)"
                    )
        else:
            batch_size = base_batch_size
            if verbose:
                print(
                    f"Auto-detected batch size: {batch_size:,} voxels (device={device.type})"
                )

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
            "Non-positive degrees of freedom detected in ARMA(1,1) fit; statistics may be unreliable"
        )

    # Allocate storage for results
    # If task_indices specified, only store those columns (save memory)
    # Otherwise store all columns
    n_output_regressors = len(fitted_column_indices) if fitted_column_indices else n_regressors
    results.betas = torch.zeros(n_voxels, n_output_regressors, device=storage_device)
    results.tstats = torch.zeros(n_voxels, n_output_regressors, device=storage_device)
    results.r2 = torch.zeros(n_voxels, device=storage_device)

    # Allocate partial R² storage if requested
    if want_r2_partial:
        results.r2_partial = torch.zeros(n_voxels, n_output_regressors, device=storage_device)
        # Also allocate for nuisance if we have filtered columns
        if fitted_column_indices is not None:
            n_nuisance = n_regressors - len(fitted_column_indices)
            if n_nuisance > 0:
                results.r2_partial_nuisance = torch.zeros(n_voxels, n_nuisance, device=storage_device)

    # Allocate semi-partial R² storage if requested
    if want_r2_semipartial:
        results.r2_semipartial = torch.zeros(n_voxels, n_output_regressors, device=storage_device)
        # Also allocate for nuisance if we have filtered columns
        if fitted_column_indices is not None:
            n_nuisance = n_regressors - len(fitted_column_indices)
            if n_nuisance > 0:
                results.r2_semipartial_nuisance = torch.zeros(n_voxels, n_nuisance, device=storage_device)

    results.arma_params = torch.zeros(n_voxels, 2, device=storage_device)
    results.arma_lambda = torch.zeros(n_voxels, device=storage_device)
    results.reml_likelihood = torch.zeros(n_voxels, device=storage_device)
    results.sigma2 = torch.zeros(n_voxels, device=storage_device)
    results.fstats = torch.zeros(n_voxels, device=storage_device)

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
            n_voxels, n_contrasts, device=storage_device
        )
        results.contrast_tstats = torch.zeros(
            n_voxels, n_contrasts, device=storage_device
        )
        # Allocate partial R² for contrasts if requested
        if want_r2_partial:
            results.contrast_r2_partial = torch.zeros(
                n_voxels, n_contrasts, device=storage_device
            )

        # Allocate semi-partial R² for contrasts if requested
        if want_r2_semipartial:
            results.contrast_r2_semipartial = torch.zeros(
                n_voxels, n_contrasts, device=storage_device
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
        glt_contrasts_tensor = torch.stack(
            glt_contrasts_list
        )  # (n_contrasts, n_regressors_full)

        # NOTE: GLT contrasts use the FULL design (all 131 columns)
        # We fit the full design, so contrasts can involve any regressor
        # No filtering needed here!

    results.dof = max(1, raw_dof)
    results.tr = tr

    if want_residuals:
        results.residuals = torch.zeros(n_voxels, n_timepoints, device=storage_device)
        results.residuals_whitened = torch.zeros(
            n_voxels, n_timepoints, device=storage_device
        )
    if want_predicted:
        results.predicted = torch.zeros(n_voxels, n_timepoints, device=storage_device)

    # OLS baseline fit (if requested)
    if want_ols:
        if verbose:
            print("\nComputing OLS baseline for comparison...")
        from .glm_core import fit_glm

        # OLS needs MUCH less memory than ARMA (no X_w prewhitened matrices)
        # ARMA bottleneck: X_w_batch = (batch_size, n_timepoints, n_regressors)
        # OLS bottleneck: data_chunk = (chunk_size, n_timepoints)
        # So OLS can use ~n_regressors times larger batches!
        # Use 16x the ARMA batch size as a safe multiplier
        ols_chunk_size = min(batch_size * 16, n_voxels)

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
            max_poly_degree=-1,  # Design matrix is complete - don't add ANYTHING!
            want_r2_partial=want_r2_partial,  # Compute partial R² for OLS if requested
            r2_partial_mode=r2_partial_mode,  # "full" or "task" mode
            want_r2_semipartial=want_r2_semipartial,  # Compute semi-partial R² for OLS if requested
            r2_semipartial_mode=r2_semipartial_mode,  # "full" or "task" mode
            glt_labels=glt_labels,
            glt_matrices=glt_matrices,
            task_indices=fitted_column_indices,  # Extract these columns for output
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
            ols_cpu = type(
                results.ols_results
            )()  # Create new results object of same type
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
            precomputed_arma_params = to_tensor(
                precomputed_arma_params, device=storage_device
            )
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
                X_w, _, L_inv = prewhiten_with_arma11(
                    design, design[:, 0], a_opt, b_opt
                )
                # CRITICAL: Store L_inv on CPU to save GPU memory (1.1 GB per matrix for 17k timepoints!)
                # Move to GPU only when needed for batch computation
                precomputed_grid[(a_opt, b_opt)] = {
                    "X_w": X_w.cpu(),  # Store on CPU
                    "L_inv": L_inv.cpu(),  # Store on CPU
                    "a": a_opt,
                    "b": b_opt,
                }

            if verbose:
                print(f"✓ Built cache for {len(precomputed_grid)} unique (a,b) pairs (stored on CPU)\n")

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
                    # Estimate memory needed for grid (L, L_inv, X_w, R_qr, workspace)
                    n_grid_estimate = len(a_grid) * len(b_grid)  # Upper bound
                    bytes_per_matrix = n_timepoints * n_timepoints * 4  # float32
                    # Need: R_batch + L_batch + L_inv_batch + X_w_batch + R_qr_batch + workspace
                    estimated_gb = (bytes_per_matrix * n_grid_estimate * 6) / 1e9

                    # Get available GPU memory
                    gpu_props = torch.cuda.get_device_properties(device)
                    total_gb = gpu_props.total_memory / 1e9
                    # Reserve 2GB for other operations
                    available_gb = total_gb - 2.0

                    # Use GPU if grid fits in 70% of available memory
                    if estimated_gb < (available_gb * 0.7):
                        use_gpu_cholesky = True
                        if verbose:
                            print(f"  🚀 Using GPU Cholesky (grid: {estimated_gb:.1f}GB < {available_gb:.1f}GB available)")
                            print("     Expected 10-20× speedup over CPU!")
                    else:
                        if verbose:
                            print(f"  Using CPU Cholesky (grid: {estimated_gb:.1f}GB > {available_gb:.1f}GB available)")

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
                )

                # Type assertion: grids are guaranteed to be set by now
                assert a_grid is not None and b_grid is not None

                if verbose:
                    print(f"✓ Precomputed {len(precomputed_grid)} valid (a,b) pairs")
                    print(f"  Grid: {len(a_grid)} a values × {len(b_grid)} b values")
                    print(
                        "  These matrices will be reused for ALL voxels (massive speedup!)"
                    )

                if debug_memory:
                    # Sample one grid entry to check size
                    sample_key = list(precomputed_grid.keys())[0]
                    _debug_memory_snapshot(
                        "AFTER grid precomputation (on CPU)",
                        device,
                        {
                            "sample_L_inv": precomputed_grid[sample_key]["L_inv"],
                            "sample_X_w": precomputed_grid[sample_key]["X_w"],
                            "sample_R_qr": precomputed_grid[sample_key]["R_qr"],
                        },
                    )

                # CRITICAL: Move entire grid to GPU ONCE (not per-batch!)
                # This is the AFNI strategy: precompute once, reuse for all voxels
                if device.type == "cuda":
                    if verbose:
                        print("  Loading grid to GPU (one-time cost)...")
                    for key in precomputed_grid:
                        precomputed_grid[key]["L_inv"] = precomputed_grid[key]["L_inv"].to(
                            device
                        )
                        precomputed_grid[key]["X_w"] = precomputed_grid[key]["X_w"].to(
                            device
                        )
                        precomputed_grid[key]["R_qr"] = precomputed_grid[key]["R_qr"].to(
                            device
                        )
                        precomputed_grid[key]["logdet_Rcorr"] = precomputed_grid[key][
                            "logdet_Rcorr"
                        ].to(device)
                        precomputed_grid[key]["logdet_XwTXw"] = precomputed_grid[key][
                            "logdet_XwTXw"
                        ].to(device)

                    # Clear any fragmentation from grid loading
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
                                "sample_R_qr": precomputed_grid[sample_key]["R_qr"],
                            },
                        )

                # CRITICAL FIX: Search each voxel against precomputed grid
                # This was missing, causing all voxels to stay at initialization (0,0)
                if verbose:
                    print("\n🔍 Searching for optimal (a,b) for each voxel...")

                # Process voxels in batches to find optimal parameters
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_voxels)
                    batch_voxels = batch_end - batch_start

                    # Get voxel data for this batch
                    Y_batch = data[batch_start:batch_end]  # (batch_voxels, n_timepoints)

                    # Search against precomputed grid
                    best_params_batch, best_lik_batch = search_voxels_precomputed_grid(
                        design,
                        Y_batch,
                        precomputed_grid,
                        device,
                        verbose=False,
                    )

                    # Store results
                    results.arma_params[batch_start:batch_end] = best_params_batch.cpu()
                    results.reml_likelihood[batch_start:batch_end] = best_lik_batch.cpu()

                    if verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                        progress_pct = 100 * (batch_idx + 1) / n_batches
                        print(f"  Progress: {progress_pct:.0f}% ({batch_idx + 1}/{n_batches} batches)")

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
        # Eliminates CPU<->GPU transfers of L_inv (saves ~87 minutes on large datasets!)
        # Compute each L_inv once, keep on GPU, process all voxels with that (a,b), then delete
        if verbose:
            print("\n📦 Grouping voxels by optimal ARMA parameters...")

        # Group voxels by their optimal (a,b) parameters
        from collections import defaultdict
        voxel_groups = defaultdict(list)

        for voxel_idx in range(n_voxels):
            a = float(results.arma_params[voxel_idx, 0].item())
            b = float(results.arma_params[voxel_idx, 1].item())
            key = (a, b)
            voxel_groups[key].append(voxel_idx)

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

            # Get L_inv and X_w for this (a,b) pair
            # If precomputed_grid exists (from precomputed ARMA params or non-batched grid search),
            # reuse cached values. Otherwise compute on-demand.
            if use_precomputed_arma and (a_opt, b_opt) in precomputed_grid:
                # Reuse cached L_inv and X_w (stored on CPU to save GPU memory)
                # Move to GPU only when needed for this group
                L_inv = precomputed_grid[(a_opt, b_opt)]["L_inv"].to(device)
                X_w = precomputed_grid[(a_opt, b_opt)]["X_w"].to(device)
            else:
                # Compute L_inv ONCE for this (a,b) group, keep on GPU
                # Use first voxel as dummy for prewhiten_with_arma11
                y_dummy = data[voxel_indices[0]].to(device)
                X_w, _, L_inv = prewhiten_with_arma11(design, y_dummy, a_opt, b_opt)

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
            # CRITICAL: Pass persistent tensors that stay on GPU (L_inv, X_w, design, etc)
            persistent_gpu_tensors = {
                "L_inv": L_inv,
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
            )

            n_sub_batches = (n_group_voxels + sub_batch_size - 1) // sub_batch_size

            # OPTIMIZATION: Compute λ ONCE per group (same for all sub-batches)
            a_val = torch.tensor(a_opt, device=device, dtype=dtype)
            b_val = torch.tensor(b_opt, device=device, dtype=dtype)
            lambda_val = (
                (b_val + a_val)
                * (1 + a_val * b_val)
                / (1 + 2 * a_val * b_val + b_val**2 + 1e-10)
            )
            # Convert to CPU scalar once for reuse
            lambda_val_cpu = lambda_val.cpu().item()

            # OPTIMIZATION: Create identity matrix ONCE per group (reused for diagonal inverse)
            # Only needed if no GLTs (diagonal-only path uses this)
            if glt_contrasts_tensor is None and not use_qr:
                eye_group = torch.eye(n_regressors, device=device, dtype=dtype)

            # Process each sub-batch within this (a,b) group
            for sub_batch_idx in range(n_sub_batches):
                sub_start = sub_batch_idx * sub_batch_size
                sub_end = min(sub_start + sub_batch_size, n_group_voxels)
                sub_voxel_indices = voxel_indices[sub_start:sub_end]
                batch_voxels = len(sub_voxel_indices)

                # Load data for these specific voxels (non-contiguous indexing)
                Y_batch = data[sub_voxel_indices].T  # (n_timepoints, batch_voxels)
                Y_batch_dev = Y_batch.to(device)

                # Store λ for these voxels (pre-computed above, reuse for all sub-batches)
                results.arma_lambda[sub_voxel_indices] = lambda_val_cpu

                # Prewhiten data using L_inv (already on GPU from group computation!)
                # All voxels in this sub-batch have the same (a,b), so same L_inv
                # Vectorized prewhitening: L_inv @ Y for all voxels at once
                t_pw_start = time.time()
                y_w_batch = (L_inv @ Y_batch_dev).T  # (batch_voxels, n_timepoints)
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
                    sigma2_batch = torch.sum(resid_w_batch**2, dim=1) / df

                    if not want_residuals:
                        del resid_w_batch

                    # Use precomputed (R'R)^{-1} - same for all voxels in group
                    XwTXw_inv_batch = XwTXw_inv_group.unsqueeze(0).expand(batch_voxels, -1, -1)

                    var_beta_batch = (
                        sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
                    )
                    se_beta_batch = torch.sqrt(
                        torch.diagonal(var_beta_batch, dim1=1, dim2=2)
                    )
                    tstats_batch = betas_batch / (se_beta_batch + 1e-10)
                    del se_beta_batch
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_stats_total += time.time() - t_stats_start
    
                    # Partial R² per regressor: r²_partial_i = t²_i / (t²_i + df)
                    if want_r2_partial:
                        df = results.dof
                        t_squared_batch = tstats_batch ** 2
                        r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
                        del t_squared_batch
    
                        # Split into task vs nuisance, rescale if mode='task'
                        if fitted_column_indices is not None:
                            task_indices_set = set(fitted_column_indices)
                            all_indices = set(range(n_regressors))
                            nuisance_indices = sorted(list(all_indices - task_indices_set))
    
                            r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                            # Always extract nuisance for storage (used for -bout output)
                            r2_partial_nuisance_batch = r2_partial_full_batch[:, nuisance_indices] if len(nuisance_indices) > 0 else None
    
                            if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                                # Rescale task partial R² by variance remaining after nuisance
                                r2_nuisance_total = r2_partial_nuisance_batch.sum(dim=1, keepdim=True)
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
                        # Need to get R² for each voxel in the batch
                        r2_batch = results.r2[sub_voxel_indices].to(device)
                        variance_remaining = torch.clamp(1.0 - r2_batch.unsqueeze(1), min=0.0)
    
                        # If we didn't compute partial R² above, compute it now for semi-partial
                        if not want_r2_partial:
                            df = results.dof
                            t_squared_batch = tstats_batch ** 2
                            r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
    
                            # Split into task vs nuisance
                            if fitted_column_indices is not None:
                                task_indices_set = set(fitted_column_indices)
                                all_indices = set(range(n_regressors))
                                nuisance_indices = sorted(list(all_indices - task_indices_set))
    
                                r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                                r2_partial_nuisance_batch = r2_partial_full_batch[:, nuisance_indices] if len(nuisance_indices) > 0 else None
                            else:
                                r2_partial_task_batch = r2_partial_full_batch
                                r2_partial_nuisance_batch = None
    
                        # Compute semi-partial R² from partial R²
                        r2_semipartial_task_batch = r2_partial_task_batch * variance_remaining
    
                        # Nuisance semi-partial R²
                        r2_semipartial_nuisance_batch = r2_partial_nuisance_batch * variance_remaining if r2_partial_nuisance_batch is not None else None
    
                        # Apply rescaling mode for task regressors
                        if r2_semipartial_mode == "task" and r2_semipartial_nuisance_batch is not None:
                            # Rescale by variance remaining after nuisance
                            r2_semi_nuisance_total = r2_semipartial_nuisance_batch.sum(dim=1, keepdim=True)
                            denominator = torch.clamp(1.0 - r2_semi_nuisance_total, min=0.01)
                            r2_semipartial_batch = r2_semipartial_task_batch / denominator
                        else:
                            # Full mode: use raw semi-partial R² values
                            r2_semipartial_batch = r2_semipartial_task_batch
    
                    # F-stat: Test only TASK regressors (not nuisance)
                    # Use CORRECT formula: F = β_task' Var(β_task)^{-1} β_task / p_task
                    # where Var(β_task) is task block of σ² (X'X)^{-1} from FULL model
                    if fitted_column_indices is not None:
                        # Extract task columns only
                        betas_task_batch = betas_batch[:, fitted_column_indices]
                        n_task_params = len(fitted_column_indices)

                        # Extract task block from variance matrix: Var_task = σ² (X'X)^{-1}_task
                        var_task_batch = var_beta_batch[:, fitted_column_indices, :][:, :, fitted_column_indices]

                        # Invert variance block: Var_task^{-1}
                        var_task_inv_batch = torch.linalg.inv(var_task_batch + 1e-8 * torch.eye(
                            n_task_params, device=device
                        ).unsqueeze(0))

                        # Compute quadratic form: β_task' Var_task^{-1} β_task
                        quad_batch = torch.bmm(
                            betas_task_batch.unsqueeze(1),
                            torch.bmm(var_task_inv_batch, betas_task_batch.unsqueeze(2))
                        ).squeeze()

                        fstats_batch = quad_batch / n_task_params
                        del quad_batch, betas_task_batch, var_task_batch, var_task_inv_batch
                    else:
                        # No filtering - test all regressors
                        betas_batch_col = betas_batch.unsqueeze(1)  # (batch, 1, n_reg)
                        var_inv_batch = torch.linalg.inv(var_beta_batch + 1e-8 * torch.eye(
                            n_regressors, device=device
                        ).unsqueeze(0))

                        quad_batch = torch.bmm(
                            betas_batch_col,
                            torch.bmm(var_inv_batch, betas_batch.unsqueeze(2))
                        ).squeeze()

                        fstats_batch = quad_batch / n_regressors
                        del quad_batch, betas_batch_col, var_inv_batch

                    y_mean_batch = Y_batch_dev.mean(dim=0)
                    ss_total_batch = torch.sum(
                        (Y_batch_dev - y_mean_batch.unsqueeze(0)) ** 2, dim=0
                    )
                    del y_mean_batch
                    ss_residual_batch = torch.sum(resid_orig_batch**2, dim=1)
                    r2_batch = 1 - ss_residual_batch / (ss_total_batch + 1e-10)
                    r2_batch = torch.clamp(r2_batch, 0, 1)  # Clamp to valid range
                    del ss_total_batch, ss_residual_batch
    
                    # GLT CONTRASTS (QR path): Compute in-loop
                    if glt_contrasts_tensor is not None:
                        t_glt_start = time.time()
                        contrast_betas_batch_qr = torch.mm(
                            betas_batch, glt_contrasts_tensor.T
                        )

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
                                'cr,brs,cs->bc',
                                glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                                var_beta_batch,        # (batch_voxels, n_regressors, n_regressors)
                                glt_contrasts_tensor   # (n_contrasts, n_regressors)
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
                            contrast_t_squared = contrast_tstats_batch_qr ** 2
                            contrast_r2_partial_batch = contrast_t_squared / (contrast_t_squared + df)
    
                        # Store immediately for QR path
                        results.contrast_betas[sub_voxel_indices] = (
                            contrast_betas_batch_qr.cpu()
                        )
                        results.contrast_tstats[sub_voxel_indices] = (
                            contrast_tstats_batch_qr.cpu()
                        )
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
                    XwTy_batch = torch.mm(y_w_batch, X_w)  # Efficient: no expand/bmm materialization!

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
                    sigma2_batch = torch.sum(resid_w_batch**2, dim=1) / df

                    if not want_residuals:
                        del resid_w_batch

                    # Variance: (X'X)^{-1} σ² - Use precomputed (X'X)^{-1} from group level!
                    # No need to invert per batch - just broadcast σ² across voxels
                    if glt_contrasts_tensor is not None:
                        # Need full var_beta_batch for GLT contrast computation
                        # Expand precomputed inverse to batch dimension
                        XwTXw_inv_batch = XwTXw_inv_group.unsqueeze(0).expand(batch_voxels, -1, -1)

                        var_beta_batch = (
                            sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
                        )
                        se_beta_batch = torch.sqrt(
                            torch.diagonal(var_beta_batch, dim1=1, dim2=2)
                        )
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
                        t_squared_batch = tstats_batch ** 2
                        r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
                        del t_squared_batch
    
                        # Split into task vs nuisance, rescale if mode='task'
                        if fitted_column_indices is not None:
                            task_indices_set = set(fitted_column_indices)
                            all_indices = set(range(n_regressors))
                            nuisance_indices = sorted(list(all_indices - task_indices_set))
    
                            r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                            # Always extract nuisance for storage (used for -bout output)
                            r2_partial_nuisance_batch = r2_partial_full_batch[:, nuisance_indices] if len(nuisance_indices) > 0 else None
    
                            if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                                # Rescale task partial R² by variance remaining after nuisance
                                r2_nuisance_total = r2_partial_nuisance_batch.sum(dim=1, keepdim=True)
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

                    # F-stat: Test only TASK regressors (not nuisance)
                    # Use CORRECT formula: F = β_task' Var(β_task)^{-1} β_task / p_task
                    # where Var(β_task) is task block of σ² (X'X)^{-1} from FULL model
                    # If var_beta_batch was not computed (no GLTs optimization), compute it now
                    if var_beta_batch is None:
                        XwTXw_inv_batch = XwTXw_inv_group.unsqueeze(0).expand(batch_voxels, -1, -1)
                        var_beta_batch = (
                            sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
                        )
                    if fitted_column_indices is not None:
                        # Extract task columns only
                        betas_task_batch = betas_batch[:, fitted_column_indices]
                        n_task_params = len(fitted_column_indices)

                        # Extract task block from variance matrix: Var_task = σ² (X'X)^{-1}_task
                        # var_beta_batch is (batch, n_reg, n_reg)
                        var_task_batch = var_beta_batch[:, fitted_column_indices, :][:, :, fitted_column_indices]

                        # Invert variance block: Var_task^{-1}
                        var_task_inv_batch = torch.linalg.inv(var_task_batch + 1e-8 * torch.eye(
                            n_task_params, device=device
                        ).unsqueeze(0))

                        # Compute quadratic form: β_task' Var_task^{-1} β_task
                        quad_batch = torch.bmm(
                            betas_task_batch.unsqueeze(1),
                            torch.bmm(var_task_inv_batch, betas_task_batch.unsqueeze(2))
                        ).squeeze()

                        fstats_batch = quad_batch / n_task_params
                        del quad_batch, betas_task_batch, var_task_batch, var_task_inv_batch
                    else:
                        # No filtering - test all regressors
                        betas_batch_col = betas_batch.unsqueeze(1)  # (batch, 1, n_reg)
                        var_inv_batch = torch.linalg.inv(var_beta_batch + 1e-8 * torch.eye(
                            n_regressors, device=device
                        ).unsqueeze(0))

                        quad_batch = torch.bmm(
                            betas_batch_col,
                            torch.bmm(var_inv_batch, betas_batch.unsqueeze(2))
                        ).squeeze()

                        fstats_batch = quad_batch / n_regressors
                        del quad_batch, betas_batch_col, var_inv_batch
    
                    # R²
                    y_mean_batch = Y_batch_dev.mean(dim=0)
                    ss_total_batch = torch.sum(
                        (Y_batch_dev - y_mean_batch.unsqueeze(0)) ** 2, dim=0
                    )
                    del y_mean_batch
                    ss_residual_batch = torch.sum(resid_orig_batch**2, dim=1)
                    r2_batch = 1 - ss_residual_batch / (ss_total_batch + 1e-10)
                    r2_batch = torch.clamp(r2_batch, 0, 1)  # Clamp to valid range
                    del ss_total_batch, ss_residual_batch
    
                    # Now compute semi-partial R² (non-QR path)
                    if want_r2_semipartial:
                        variance_remaining = torch.clamp(1.0 - r2_batch.unsqueeze(1), min=0.0)
    
                        # If we didn't compute partial R² above, compute it now for semi-partial
                        if not want_r2_partial:
                            df = results.dof
                            t_squared_batch = tstats_batch ** 2
                            r2_partial_full_batch = t_squared_batch / (t_squared_batch + df)
    
                            # Split into task vs nuisance
                            if fitted_column_indices is not None:
                                task_indices_set = set(fitted_column_indices)
                                all_indices = set(range(n_regressors))
                                nuisance_indices = sorted(list(all_indices - task_indices_set))
    
                                r2_partial_task_batch = r2_partial_full_batch[:, fitted_column_indices]
                                r2_partial_nuisance_batch = r2_partial_full_batch[:, nuisance_indices] if len(nuisance_indices) > 0 else None
                            else:
                                r2_partial_task_batch = r2_partial_full_batch
                                r2_partial_nuisance_batch = None
    
                        # Compute semi-partial R² from partial R²
                        r2_semipartial_task_batch = r2_partial_task_batch * variance_remaining
    
                        # Nuisance semi-partial R²
                        r2_semipartial_nuisance_batch = r2_partial_nuisance_batch * variance_remaining if r2_partial_nuisance_batch is not None else None
    
                        # Apply rescaling mode for task regressors
                        if r2_semipartial_mode == "task" and r2_semipartial_nuisance_batch is not None:
                            # Rescale by variance remaining after nuisance
                            r2_semi_nuisance_total = r2_semipartial_nuisance_batch.sum(dim=1, keepdim=True)
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
                            'cr,brs,cs->bc',
                            glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                            var_beta_batch,        # (batch_voxels, n_regressors, n_regressors)
                            glt_contrasts_tensor   # (n_contrasts, n_regressors)
                        )

                    # Compute t-statistics for contrasts
                    contrast_se_batch = torch.sqrt(
                        torch.clamp(contrast_vars_batch, min=0.0)
                    )
                    contrast_tstats_batch = contrast_betas_batch / (
                        contrast_se_batch + 1e-10
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_glt_total += time.time() - t_glt_start
    
                    # Compute partial R² for contrasts if requested
                    if want_r2_partial:
                        df = results.dof
                        contrast_t_squared = contrast_tstats_batch ** 2
                        contrast_r2_partial_batch = contrast_t_squared / (contrast_t_squared + df)
    
                    # Compute semi-partial R² for contrasts if requested
                    if want_r2_semipartial:
                        df = results.dof
                        contrast_t_squared = contrast_tstats_batch ** 2
                        contrast_r2_partial_batch_temp = contrast_t_squared / (contrast_t_squared + df)
                        # Use the r2_batch that was fetched earlier in semi-partial computation
                        # If we didn't compute semi-partial above (e.g., no task regressors), get it now
                        if 'r2_batch' not in locals():
                            r2_batch = results.r2[sub_voxel_indices].to(device)
                        variance_remaining_contrasts = torch.clamp(1.0 - r2_batch.unsqueeze(1), min=0.0)
                        contrast_r2_semipartial_batch = contrast_r2_partial_batch_temp * variance_remaining_contrasts
    
                    # Store contrast results
                    results.contrast_betas[sub_voxel_indices] = contrast_betas_batch.cpu()
                    results.contrast_tstats[sub_voxel_indices] = contrast_tstats_batch.cpu()
                    if want_r2_partial:
                        results.contrast_r2_partial[sub_voxel_indices] = contrast_r2_partial_batch.cpu()
                    if want_r2_semipartial:
                        results.contrast_r2_semipartial[sub_voxel_indices] = contrast_r2_semipartial_batch.cpu()
    
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
                    if r2_partial_nuisance_batch is not None and results.r2_partial_nuisance is not None:
                        results.r2_partial_nuisance[sub_voxel_indices] = r2_partial_nuisance_batch.cpu()
    
                # Store semi-partial R² if requested
                # NOTE: r2_semipartial_batch is already filtered to task columns in the computation above
                if want_r2_semipartial:
                    results.r2_semipartial[sub_voxel_indices] = r2_semipartial_batch.cpu()
                    # Store nuisance semi-partial R² if we extracted it
                    if r2_semipartial_nuisance_batch is not None and results.r2_semipartial_nuisance is not None:
                        results.r2_semipartial_nuisance[sub_voxel_indices] = r2_semipartial_nuisance_batch.cpu()
    
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

            # End of sub-batch loop - all voxels for this (a,b) group processed

            # Print timing summary for this group (first 3 groups only to avoid spam)
            if verbose and group_idx < 3:
                print(f"\n  Group {group_idx+1} timing ({n_group_voxels:,} voxels, {n_sub_batches} batches):")
                print(f"    QR setup:     {t_qr:6.2f}s (once per group)")
                print(f"    Prewhiten:    {t_prewhiten_total:6.2f}s ({t_prewhiten_total/n_sub_batches:.3f}s/batch)")
                print(f"    QR solve:     {t_qr_solve_total:6.2f}s ({t_qr_solve_total/n_sub_batches:.3f}s/batch)")
                print(f"    Statistics:   {t_stats_total:6.2f}s ({t_stats_total/n_sub_batches:.3f}s/batch)")
                if glt_contrasts_tensor is not None and n_contrasts > 0:
                    print(f"    GLT ({n_contrasts}):     {t_glt_total:6.2f}s ({t_glt_total/n_sub_batches:.3f}s/batch)")
                total_group_time = t_qr + t_prewhiten_total + t_qr_solve_total + t_stats_total + t_glt_total
                print(f"    Total:        {total_group_time:6.2f}s\n")

            # Delete group-level precomputed matrices to free GPU memory for next group
            del L_inv, X_w
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
            design, y_mean, a_grid, b_grid, device
        )

        if verbose:
            print(f"  Optimal (a, b) = ({a_opt:.3f}, {b_opt:.3f})")

        # Fill parameters for all voxels
        results.arma_params[:, 0] = a_opt
        results.arma_params[:, 1] = b_opt
        results.reml_likelihood[:] = likelihood_opt

        # Compute λ
        lam = (
            (b_opt + a_opt)
            * (1 + a_opt * b_opt)
            / (1 + 2 * a_opt * b_opt + b_opt**2 + 1e-10)
        )
        results.arma_lambda[:] = lam

        # Move design to device for prewhitening
        design_dev = design.to(device) if design.device != device else design

        # Prewhiten design (shared across all voxels)
        X_w, _, L_inv = prewhiten_with_arma11(design_dev, y_mean, a_opt, b_opt)
        XwTXw = X_w.T @ X_w
        XwTXw_reg = XwTXw + 1e-6 * torch.eye(n_regressors, device=device)

        # GLS fit for all voxels
        if verbose:
            print("Fitting all voxels with global parameters...")

        voxel_iterator = range(n_voxels)
        if verbose and n_voxels > 1000:
            voxel_iterator = tqdm(
                voxel_iterator, desc="Global ARMA fitting", unit="voxel"
            )

        for v in voxel_iterator:
            y_v_cpu = data[v]
            y_v_dev = y_v_cpu.to(device)

            # Prewhiten data
            _, y_w, _ = prewhiten_with_arma11(design_dev, y_v_dev, a_opt, b_opt)

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
            sigma2 = torch.sum(resid_w**2) / df
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
                t_squared = tstats ** 2
                r2_partial_full = t_squared / (t_squared + df)

                # Split into task vs nuisance, rescale if mode='task'
                if fitted_column_indices is not None:
                    task_indices_set = set(fitted_column_indices)
                    all_indices = set(range(len(beta)))
                    nuisance_indices = sorted(list(all_indices - task_indices_set))

                    r2_partial_task = r2_partial_full[fitted_column_indices]
                    # Always extract nuisance for storage (used for -bout output)
                    r2_partial_nuisance = r2_partial_full[nuisance_indices] if len(nuisance_indices) > 0 else None

                    if r2_partial_mode == "task" and len(nuisance_indices) > 0:
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
                var_task_inv = torch.linalg.inv(var_task + 1e-8 * torch.eye(
                    n_task_params, device=device
                ))

                # Compute quadratic form: β_task' Var_task^{-1} β_task
                quad = torch.dot(beta_task, var_task_inv @ beta_task)
                fstat = quad / n_task_params
            else:
                # No filtering - test all regressors
                var_inv = torch.linalg.inv(var_beta + 1e-8 * torch.eye(
                    n_regressors, device=device
                ))
                quad = torch.dot(beta, var_inv @ beta)
                fstat = quad / n_regressors

            results.fstats[v] = fstat.cpu()

            # R²
            y_mean_val = y_v_cpu.mean()
            ss_total = torch.sum((y_v_cpu - y_mean_val) ** 2)
            ss_residual = torch.sum(resid_orig_cpu**2)
            r2_val = (1 - ss_residual / (ss_total + 1e-10)).cpu()
            results.r2[v] = torch.clamp(r2_val, 0, 1)  # Clamp to valid range

            # Optional outputs
            if want_residuals:
                results.residuals[v] = resid_orig_cpu
                results.residuals_whitened[v] = resid_w.cpu()
            if want_predicted:
                results.predicted[v] = pred_orig_cpu

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
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    tr: float,
    device: Optional[torch.device] = None,
) -> Dict:
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
    from .glm_core import fit_glm

    if device is None:
        device = get_device()

    print("Running OLS GLM...")
    ols_results = fit_glm(data, design, tr, device=device, verbose=False)

    print("\nRunning ARMA(1,1) GLM...")
    arma_results = fit_glm_arma11(data, design, tr, device=device, verbose=True)

    # Compare t-statistics
    tstat_ratio = arma_results.tstats.abs().mean() / (
        ols_results.tstats.abs().mean() + 1e-10
    )

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


def compute_ljung_box_statistic(
    residuals: Union[torch.Tensor, np.ndarray], max_lag: int = 30
) -> np.ndarray:
    """
    Compute Ljung-Box statistic for residual autocorrelation

    The LB statistic tests for remaining autocorrelation in prewhitened residuals.
    Small values indicate successful prewhitening; large values indicate the
    ARMA(1,1) model was inadequate.

    Parameters
    ----------
    residuals : array-like, shape (n_voxels, n_timepoints)
        Prewhitened residuals from ARMA(1,1) fit
    max_lag : int, default=30
        Maximum lag for autocorrelation (AFNI uses h=30)

    Returns
    -------
    lb_stats : np.ndarray, shape (n_voxels,)
        Ljung-Box chi-squared statistics (df = max_lag - 2)
        Zero values indicate computation failed (e.g., zero residuals)

    Notes
    -----
    LB = n(n+2) * sum_{k=1}^h [ rho_k^2 / (n-k) ]
    where rho_k is the autocorrelation at lag k, n is sample size

    Follows chi-squared distribution with (h-2) degrees of freedom
    """
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()

    n_voxels, n_timepoints = residuals.shape
    lb_stats = np.zeros(n_voxels, dtype=np.float32)

    for v in range(n_voxels):
        resid = residuals[v]

        # Skip if residuals are all zero
        if np.all(resid == 0) or np.std(resid) < 1e-10:
            lb_stats[v] = 0.0
            continue

        # Standardize residuals
        resid = (resid - np.mean(resid)) / (np.std(resid) + 1e-10)
        n = len(resid)

        # Compute autocorrelations up to max_lag
        lb = 0.0
        for k in range(1, min(max_lag + 1, n)):
            # Autocorrelation at lag k
            rho_k = np.corrcoef(resid[:-k], resid[k:])[0, 1]
            if np.isnan(rho_k):
                continue
            lb += (rho_k**2) / (n - k)

        lb_stats[v] = n * (n + 2) * lb

    return lb_stats


def save_arma_rvar(
    results: ARMA11Results,
    output_path: Union[str, Path],
    volume_shape: Optional[Tuple[int, int, int]] = None,
    voxel_mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    max_lag: int = 30,
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
    max_lag : int, default=30
        Maximum lag for Ljung-Box statistic (AFNI default)

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
    import nibabel as nib

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

    # Compute Ljung-Box statistic
    if results.residuals_whitened is not None:
        ljung_box = compute_ljung_box_statistic(
            results.residuals_whitened, max_lag=max_lag
        )
    else:
        # No residuals saved - set to zero
        assert arma_params is not None, "arma_params should not be None"
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

    # Save as NIfTI
    img = nib.Nifti1Image(rvar_4d.astype(np.float32), affine)
    img.header.set_xyzt_units(xyz="mm")

    # Add description (AFNI-compatible)
    img.header["descrip"] = (
        b"ARMA(1,1) -Rvar: [0]=a [1]=b [2]=lam [3]=StDev [4]=-LogLik [5]=LjungBox"
    )

    # Add AFNI-style sub-brick labels for each volume
    # These will show up in AFNI's overlay GUI
    nifti_extension = nib.nifti1.Nifti1Extension(
        code=6,  # AFNI extension code
        content=(
            b"BRICK_LABS~"
            b"a~"  # Volume 0: AR parameter
            b"b~"  # Volume 1: MA parameter
            b"lam~"  # Volume 2: lambda (lag-1 correlation)
            b"StDev~"  # Volume 3: standard deviation
            b"-LogLik~"  # Volume 4: negative log-likelihood
            b"LjungBox"  # Volume 5: Ljung-Box statistic
        ),
    )
    img.header.extensions.append(nifti_extension)

    nib.save(img, output_path)

    return output_path


def load_arma_params(
    filepath: Union[str, Path],
    voxel_mask: Optional[np.ndarray] = None,
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
    import nibabel as nib

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ARMA params file not found: {filepath}")

    # Load NIfTI
    img = nib.load(str(filepath))
    arma_4d = img.get_fdata(dtype=np.float32)

    # Extract a and b volumes (volumes 0 and 1)
    if arma_4d.ndim == 4 and arma_4d.shape[-1] >= 2:
        a_vol = arma_4d[..., 0]
        b_vol = arma_4d[..., 1]
    else:
        raise ValueError(
            f"Expected 4D NIfTI with at least 2 volumes, got shape {arma_4d.shape}"
        )

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
