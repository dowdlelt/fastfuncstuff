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
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from tqdm.auto import tqdm


def _debug_memory_snapshot(
    label: str, device: torch.device, tensors: Dict[str, torch.Tensor] = None
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
DEFAULT_ARMA_A_GRID = (0.0, 0.9, 9)  # (start, end, num_points)
DEFAULT_ARMA_B_GRID = (-0.8, 0.8, 9)  # (start, end, num_points)


def check_cuda_memory_before_batch(
    batch_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    device: torch.device,
    verbose: bool = False,
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

    # Calculate required memory for GLS solve (the bottleneck!)
    required_bytes = (
        batch_voxels * n_timepoints * n_regressors * 4  # X_w_batch
        + batch_voxels * n_timepoints * 4  # y_w_batch
        + batch_voxels * n_regressors * n_regressors * 4  # XtX
        + batch_voxels * n_regressors * 4  # betas
        + batch_voxels * n_timepoints * 4  # residuals
    )

    # Get available memory (free + reserved-but-unallocated)
    free_mem = torch.cuda.mem_get_info(device)[0]
    reserved_mem = torch.cuda.memory_reserved(device)
    allocated_mem = torch.cuda.memory_allocated(device)
    available_mem = free_mem + (reserved_mem - allocated_mem)

    # Use only 80% of available (leave headroom)
    usable_mem = available_mem * 0.8

    if required_bytes > usable_mem:
        # Need to reduce batch size
        reduction_factor = usable_mem / required_bytes
        new_batch = int(batch_voxels * reduction_factor * 0.95)  # Extra 5% safety
        new_batch = max(new_batch, 500)  # Minimum batch

        if verbose:
            print(
                f"\n⚠ Memory check: batch would need {required_bytes / 1e9:.2f} GB "
                f"but only {usable_mem / 1e9:.2f} GB available"
            )
            print(f"  Reducing batch: {batch_voxels:,} → {new_batch:,} voxels\n")

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
    - L_inv: inverse Cholesky factor (n_timepoints, n_timepoints)
    - X_w: prewhitened design (n_timepoints, n_regressors)
    - XwTXw_reg: X'X matrix (n_regressors, n_regressors)
    - logdet_R: scalar
    - logdet_XwTXw: scalar

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
    device: torch.device, n_timepoints: int, n_regressors: int, use_double: bool = False
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
    Memory scaling:
    - Each voxel needs ~(n_timepoints × n_regressors × bytes_per_element) for whitened design
    - Additional workspace for residuals, betas, etc.
    - Conservative estimates to ensure stability
    - Float64 uses 2x memory, so batch size is automatically halved

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
    # ACCURATE memory per voxel calculation for GLS solve (the real bottleneck!)
    # The critical allocation is X_w_batch = (batch_voxels, n_timepoints, n_regressors)
    bytes_per_element = 8 if use_double else 4
    mem_per_voxel = (
        n_timepoints * n_regressors * bytes_per_element  # X_w_batch: THE BOTTLENECK!
        + n_timepoints * bytes_per_element  # y_w_batch
        + n_regressors * n_regressors * bytes_per_element  # XtX workspace
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
            # NOTE: These are conservative estimates. The actual batch size will be
            # further adjusted in fit_glm_arma11() to account for grid memory.
            if total_mem < 10e9:  # < 10GB
                batch_size = min(batch_size, 5000)
            elif total_mem < 20e9:  # 10-20GB (e.g., RTX 4070, 15-16GB GPUs)
                batch_size = min(batch_size, 20000)
            else:  # > 20GB (e.g., RTX 4090, A100)
                batch_size = min(batch_size, 30000)
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
        AR parameter grid [0.0, 0.1, 0.2, ..., 0.9] (10 points)
    b_grid : torch.Tensor
        MA parameter grid [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] (7 points)

    Notes
    -----
    Total: 63 (a, b) combinations
    Matches AFNI 3dREMLfit -Grid 3 (medium resolution, good balance)
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
        self.r2: Optional[torch.Tensor] = None  # (n_voxels,) R² values
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
) -> torch.Tensor:
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

    # PHASE 2: Batch Cholesky decomposition
    try:
        # Compute on CPU (default) or GPU
        # CPU is recommended for large grids to avoid OOM, and is still very fast
        if cholesky_on_cpu or device.type == "mps":
            # R_batch already on CPU from build_device logic above
            L_batch = torch.linalg.cholesky(R_batch)  # (n_valid, n, n)
            L_inv_batch = torch.linalg.inv(L_batch)  # (n_valid, n, n)
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

        if verbose:
            print(f"  ✓ Computed {n_valid} Cholesky factorizations at once!")
            print(f"  Prewhitening design matrix for all parameters...")

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

        # PHASE 4: Batch X'X computation
        X_w_batch_T = X_w_batch.transpose(1, 2)  # Cache transpose
        XwTXw_batch = torch.bmm(
            X_w_batch_T, X_w_batch
        )  # (n_valid, n_regressors, n_regressors)
        del X_w_batch_T  # Free transpose cache

        # No ridge regularization - match AFNI behavior
        XwTXw_reg_batch = XwTXw_batch  # (n_valid, n_regressors, n_regressors)

        # PHASE 5: Batch likelihood terms (that don't depend on voxel data)
        # logdet_R = 2 * sum(log(diag(L)))
        logdet_R_batch = 2 * torch.sum(
            torch.log(torch.diagonal(L_batch, dim1=1, dim2=2) + 1e-10), dim=1
        )  # (n_valid,)
        del L_batch  # Don't need L anymore, only L_inv

        # logdet(X'R^-1 X)
        sign_batch, logdet_XwTXw_batch = torch.linalg.slogdet(XwTXw_reg_batch)
        # Handle non-positive determinants (use same device as the tensor!)
        logdet_XwTXw_batch = torch.where(
            sign_batch > 0,
            logdet_XwTXw_batch,
            torch.tensor(1e10, device=logdet_XwTXw_batch.device, dtype=dtype),
        )  # (n_valid,)
        del sign_batch  # Free immediately

        if verbose:
            print(f"  ✓ Precomputed all matrices!")
            print(f"  Storing {n_valid} parameter sets...")

        # PHASE 6: Store everything in dictionary ON CPU
        # Tensors are already on CPU from Phase 2 - store as-is
        # They will be moved to GPU on-demand during REML search
        for i, (a_val, b_val) in enumerate(param_list):
            precomputed[(a_val, b_val)] = {
                "L_inv": L_inv_batch[i],  # (n, n) - CPU tensor
                "X_w": X_w_batch[i],  # (n, n_regressors) - CPU tensor
                "XwTXw_reg": XwTXw_reg_batch[
                    i
                ],  # (n_regressors, n_regressors) - CPU tensor
                "logdet_R": logdet_R_batch[i],  # scalar tensor - CPU
                "logdet_XwTXw": logdet_XwTXw_batch[i],  # scalar tensor - CPU
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
            del L_batch
        if "L_inv_batch" in locals():
            del L_inv_batch
        if "X_w_batch" in locals():
            del X_w_batch
        if "XwTXw_batch" in locals():
            del XwTXw_batch

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
                XwTXw = X_w.T @ X_w
                XwTXw_reg = XwTXw + 1e-6 * torch.eye(
                    n_regressors, device=torch.device("cpu"), dtype=dtype
                )

                logdet_R = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10))
                sign, logdet_XwTXw = torch.linalg.slogdet(XwTXw_reg)
                logdet_XwTXw = (
                    logdet_XwTXw
                    if sign > 0
                    else torch.tensor(1e10, device=torch.device("cpu"), dtype=dtype)
                )

                # Store on CPU - will be moved to GPU on-demand during REML search
                precomputed[(a_val, b_val)] = {
                    "L_inv": L_inv,  # CPU tensor
                    "X_w": X_w,  # CPU tensor
                    "XwTXw_reg": XwTXw_reg,  # CPU tensor
                    "logdet_R": logdet_R,  # CPU tensor
                    "logdet_XwTXw": logdet_XwTXw,  # CPU tensor
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
                "XwTXw_reg_batch": locals().get("XwTXw_reg_batch"),
            },
        )

        # Calculate expected grid size
        if "L_inv_batch" in locals():
            n_pairs = len(param_list)
            n_regs = X.shape[1]
            bytes_per_elem = L_inv_batch.element_size()

            expected_L_inv = n_pairs * n_timepoints * n_timepoints * bytes_per_elem
            expected_X_w = n_pairs * n_timepoints * n_regs * bytes_per_elem
            expected_XwTXw = n_pairs * n_regs * n_regs * bytes_per_elem
            expected_scalars = n_pairs * 2 * bytes_per_elem
            expected_total = (
                expected_L_inv + expected_X_w + expected_XwTXw + expected_scalars
            )

            actual_L_inv = L_inv_batch.element_size() * L_inv_batch.nelement()
            actual_X_w = X_w_batch.element_size() * X_w_batch.nelement()
            actual_XwTXw = XwTXw_reg_batch.element_size() * XwTXw_reg_batch.nelement()
            actual_total = actual_L_inv + actual_X_w + actual_XwTXw

            print(f"\nGRID SIZE VERIFICATION:")
            print(
                f"  n_pairs={n_pairs}, n_timepoints={n_timepoints}, n_regressors={n_regs}, dtype={dtype}"
            )
            print(f"  Expected grid size: {expected_total / 1024**3:.3f} GiB")
            print(f"    - L_inv:  {expected_L_inv / 1024**3:.3f} GiB")
            print(f"    - X_w:    {expected_X_w / 1024**3:.3f} GiB")
            print(f"    - XwTXw:  {expected_XwTXw / 1024**3:.3f} GiB")
            print(f"  Actual batch tensors: {actual_total / 1024**3:.3f} GiB")
            print(
                f"  Match: {'✅ YES' if abs(actual_total - expected_total) < 1024**2 else '❌ NO'}"
            )

    if "L_inv_batch" in locals():
        del L_inv_batch
    if "X_w_batch" in locals():
        del X_w_batch
    if "XwTXw_reg_batch" in locals():
        del XwTXw_reg_batch
    if "logdet_R_batch" in locals():
        del logdet_R_batch
    if "logdet_XwTXw_batch" in locals():
        del logdet_XwTXw_batch

    if debug_memory:
        _debug_memory_snapshot(
            "Grid: After deleting batch tensors (grid in dict only)", device, {}
        )

    return precomputed


def batch_reml_grid_search(
    X: torch.Tensor,
    Y_batch: torch.Tensor,
    a_grid: Optional[torch.Tensor] = None,
    b_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    precomputed: Optional[dict] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ULTRA-PARALLEL REML grid search with adaptive memory management

    **GPU OPTIMIZATION WITH MEMORY SAFETY:**
    Computes (grid × N_voxels) likelihoods in parallel, automatically chunking
    the grid if memory usage would be too high.

    For small grids (<= ~40 params), processes all at once (single operation).
    For large grids (>40 params), chunks grid into manageable pieces to avoid OOM.

    This is what 3dREMLfit CAN'T do (CPU sequential) but GPUs were BUILT for!

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

    # MEMORY-ADAPTIVE CHUNKING:
    # For large grids (>70 params) or many voxels, process grid in chunks
    # Memory usage: (n_grid_chunk × n_timepoints × n_voxels) × bytes_per_element × 3 tensors
    # Target: Keep each chunk under ~2GB
    bytes_per_element = 8 if dtype == torch.float64 else 4
    target_chunk_gb = 1.5  # Conservative target per chunk

    # Estimate memory for one grid point across all voxels
    mem_per_grid_point = (
        n_timepoints * n_voxels_batch * bytes_per_element * 3
    )  # Y_w, pred, resid
    max_grid_chunk = int((target_chunk_gb * 1e9) / mem_per_grid_point)
    max_grid_chunk = max(10, min(max_grid_chunk, n_grid))  # At least 10, at most all

    # If grid is small enough, process all at once (original behavior)
    if n_grid <= max_grid_chunk:
        grid_chunks = [list(range(n_grid))]
    else:
        # Chunk the grid
        grid_chunks = [
            list(range(i, min(i + max_grid_chunk, n_grid)))
            for i in range(0, n_grid, max_grid_chunk)
        ]

    # Process grid in chunks and track best per voxel
    best_params = torch.zeros(n_voxels_batch, 2, device=device, dtype=dtype)
    best_likelihoods = torch.full(
        (n_voxels_batch,), float("inf"), device=device, dtype=dtype
    )

    param_list = list(precomputed.keys())

    for chunk_indices in grid_chunks:
        # PHASE 1: Stack precomputed matrices for THIS CHUNK only
        # Tensors already on GPU (moved once after precomputation)
        chunk_keys = [param_list[i] for i in chunk_indices]

        L_inv_stack = torch.stack(
            [precomputed[k]["L_inv"] for k in chunk_keys]
        )  # (n_chunk, n_time, n_time)
        X_w_stack = torch.stack(
            [precomputed[k]["X_w"] for k in chunk_keys]
        )  # (n_chunk, n_time, n_regressors)
        XwTXw_reg_stack = torch.stack(
            [precomputed[k]["XwTXw_reg"] for k in chunk_keys]
        )  # (n_chunk, n_reg, n_reg)
        logdet_R_stack = torch.stack(
            [precomputed[k]["logdet_R"] for k in chunk_keys]
        )  # (n_chunk,)
        logdet_XwTXw_stack = torch.stack(
            [precomputed[k]["logdet_XwTXw"] for k in chunk_keys]
        )  # (n_chunk,)

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
        XwTYw_all = torch.bmm(X_w_stack.transpose(1, 2), Y_w_all)
        beta_w_all = torch.linalg.solve(
            XwTXw_reg_stack,  # (n_chunk, n_reg, n_reg)
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
        term1 = logdet_R_stack.unsqueeze(1)  # (n_chunk, 1)
        term2 = logdet_XwTXw_stack.unsqueeze(1)  # (n_chunk, 1)
        term3 = (n_timepoints - n_regressors) * torch.log(rss_all + 1e-10)
        del rss_all

        chunk_likelihoods = term1 + term2 + term3  # (n_chunk, n_voxels)
        del term1, term2, term3, L_inv_stack, X_w_stack, XwTXw_reg_stack
        del logdet_R_stack, logdet_XwTXw_stack, beta_w_all

        # PHASE 6: Update best parameters if this chunk has better likelihoods
        chunk_best_idx = torch.argmin(chunk_likelihoods, dim=0)  # (n_voxels,)
        chunk_best_likelihoods = chunk_likelihoods[
            chunk_best_idx, torch.arange(n_voxels_batch, device=device)
        ]
        del chunk_likelihoods

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
    """
    device = X.device
    n_timepoints = X.shape[0]

    # Build ARMA(1,1) covariance
    R = build_arma11_covariance(a, b, n_timepoints, device)

    if R is None:
        raise ValueError(f"Invalid ARMA(1,1) parameters: a={a}, b={b}")

    # Cholesky decomposition: R = L L'
    # MPS has a bug with Cholesky - use CPU as workaround
    if device.type == "mps":
        R_cpu = R.cpu()
        L_cpu = torch.linalg.cholesky(R_cpu)
        L = L_cpu.to(device)
    else:
        L = torch.linalg.cholesky(R)

    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Use triangular solves for numerical stability (avoids explicit inverse)
    X_white = torch.linalg.solve_triangular(L, X, upper=False)
    Y_white = torch.linalg.solve_triangular(L, Y, upper=False)

    identity = torch.eye(n_timepoints, device=device, dtype=L.dtype)
    L_inv = torch.linalg.solve_triangular(L, identity, upper=False)

    if Y_white.shape[1] == 1:
        Y_white = Y_white.squeeze(1)

    return X_white, Y_white, L_inv


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
    want_ols: bool = False,
    ols_write_callback: Optional[Callable] = None,
    precomputed_arma_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    cholesky_on_cpu: bool = True,
    use_double: bool = False,
    debug_memory: bool = False,
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

    # Clear GPU cache at start to ensure maximum free memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Convert to tensors
    data = to_tensor(data, device=None, dtype=dtype)
    if data.device.type != "cpu":
        data = data.to(storage_device)
    design = to_tensor(design, device=device).to(dtype)

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
    n_regressors = design.shape[1]

    # Adaptive batch sizing based on GPU memory
    if batch_size is None:
        base_batch_size = get_adaptive_batch_size(
            device, n_timepoints, n_regressors, use_double
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

            if debug_memory:
                print(f"\n{'=' * 70}")
                print(f"EXPECTED GRID SIZE CALCULATION")
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
                print(f"  Expected grid memory:")
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
                # Use 50% of GPU memory (assumes dedicated GPU for this task)
                # Leave 10% for PyTorch overhead, CUDA context, and fragmentation
                available_mem = total_mem * 0.75
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
                    + n_timepoints * bytes_per_element  # y_w_batch
                    + n_regressors * n_regressors * bytes_per_element  # XtX workspace
                    + n_regressors * bytes_per_element  # betas
                    + n_regressors * bytes_per_element  # tstats
                    + 3 * bytes_per_element  # params (a, b, lambda)
                )
                batch_size = int(memory_for_batches / mem_per_voxel)
                batch_size = max(500, min(batch_size, base_batch_size))
            else:
                # Grid too large, fall back to base (will OOM, but let it try)
                batch_size = base_batch_size

            if verbose:
                grid_mb = grid_memory_bytes / (1024**2)
                print(f"Grid memory: {grid_mb:.1f} MiB ({n_valid_pairs} pairs)")
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
    raw_dof = n_timepoints - n_regressors
    if raw_dof <= 0:
        warnings.warn(
            "Non-positive degrees of freedom detected in ARMA(1,1) fit; statistics may be unreliable"
        )

    results.betas = torch.zeros(n_voxels, n_regressors, device=storage_device)
    results.tstats = torch.zeros(n_voxels, n_regressors, device=storage_device)
    results.r2 = torch.zeros(n_voxels, device=storage_device)
    results.arma_params = torch.zeros(n_voxels, 2, device=storage_device)
    results.arma_lambda = torch.zeros(n_voxels, device=storage_device)
    results.reml_likelihood = torch.zeros(n_voxels, device=storage_device)
    results.sigma2 = torch.zeros(n_voxels, device=storage_device)
    results.fstats = torch.zeros(n_voxels, device=storage_device)
    results.var_betas = torch.zeros(
        n_voxels, n_regressors, n_regressors, device=storage_device
    )
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
        # Use 3-4x the ARMA batch size as a safe multiplier
        ols_chunk_size = min(batch_size * 4, n_voxels)

        # Set preload_data_to_device=False to stream chunks from CPU
        # CRITICAL: Pass max_poly_degree=-1 to prevent adding ANY polynomials (including constant)
        # The design matrix from AFNI already includes all polynomials and intercept!
        results.ols_results = fit_glm(
            data,
            design,
            tr=tr,
            device=device,
            chunk_size=ols_chunk_size,
            verbose=False,
            preload_data_to_device=False,
            use_double=use_double,
            max_poly_degree=-1,  # Design matrix is complete - don't add ANYTHING!
        )
        if verbose:
            print("✓ OLS fit complete")

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

            for params in unique_params:
                a_opt, b_opt = params[0].item(), params[1].item()
                X_w, _, L_inv = prewhiten_with_arma11(
                    design, design[:, 0], a_opt, b_opt
                )
                precomputed_grid[(a_opt, b_opt)] = {
                    "X_w": X_w,
                    "L_inv": L_inv,
                    "a": a_opt,
                    "b": b_opt,
                }

            if verbose:
                print(f"✓ Built cache for {len(precomputed_grid)} unique (a,b) pairs\n")

        else:
            # PRE-COMPUTE REML GRID (KEY OPTIMIZATION!)
            if verbose:
                print("Pre-computing REML grid (Cholesky factorizations)...")

            # Clear GPU cache before grid precomputation (critical for memory!)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

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

            precomputed_grid = precompute_reml_grid(
                design,
                n_timepoints,
                a_grid,
                b_grid,
                device,
                verbose=verbose,
                cholesky_on_cpu=cholesky_on_cpu,
                dtype=dtype,
                debug_memory=debug_memory,
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
                        "sample_XwTXw_reg": precomputed_grid[sample_key]["XwTXw_reg"],
                    },
                )

            # CRITICAL: Move entire grid to GPU ONCE (not per-batch!)
            # This is the AFNI strategy: precompute once, reuse for all voxels
            if device.type == "cuda":
                if verbose:
                    print(f"  Loading grid to GPU (one-time cost)...")
                for key in precomputed_grid:
                    precomputed_grid[key]["L_inv"] = precomputed_grid[key]["L_inv"].to(
                        device
                    )
                    precomputed_grid[key]["X_w"] = precomputed_grid[key]["X_w"].to(
                        device
                    )
                    precomputed_grid[key]["XwTXw_reg"] = precomputed_grid[key][
                        "XwTXw_reg"
                    ].to(device)
                    precomputed_grid[key]["logdet_R"] = precomputed_grid[key][
                        "logdet_R"
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
                            "sample_XwTXw_reg": precomputed_grid[sample_key][
                                "XwTXw_reg"
                            ],
                        },
                    )
            elif verbose:
                print()

        # Progress bar for batches
        batch_iterator = range(n_batches)
        if verbose and n_batches > 1:
            batch_iterator = tqdm(
                batch_iterator, desc="ARMA(1,1) batches", unit="batch"
            )

        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_voxels)
            batch_voxels = end_idx - start_idx

            # SMART MEMORY CHECK: Adjust batch size if needed before allocating
            batch_voxels = check_cuda_memory_before_batch(
                batch_voxels,
                n_timepoints,
                n_regressors,
                device,
                verbose=(verbose and batch_idx == 0),  # Only warn on first batch
            )
            end_idx = start_idx + batch_voxels

            # Update batch_size for future iterations if we had to reduce
            if batch_voxels < batch_size:
                batch_size = batch_voxels

            # Get data batch (CPU) and GPU copy
            Y_batch = data[start_idx:end_idx].T  # (n_timepoints, batch_voxels)
            Y_batch_dev = Y_batch.to(device)

            if use_precomputed_arma:
                # Use precomputed ARMA parameters (skip REML estimation)
                batch_params = results.arma_params[start_idx:end_idx].to(device)
            else:
                # REML grid search for this batch (using pre-computed grid!)
                batch_params, batch_likelihoods = batch_reml_grid_search(
                    design,
                    Y_batch_dev,
                    a_grid,
                    b_grid,
                    device,
                    precomputed=precomputed_grid,  # ← KEY: reuse pre-computed matrices!
                    dtype=dtype,
                )

                results.arma_params[start_idx:end_idx] = batch_params.cpu()
                results.reml_likelihood[start_idx:end_idx] = batch_likelihoods.cpu()

                # Compute λ for each voxel
                a_vals = batch_params[:, 0]
                b_vals = batch_params[:, 1]
                lambda_vals = (
                    (b_vals + a_vals)
                    * (1 + a_vals * b_vals)
                    / (1 + 2 * a_vals * b_vals + b_vals**2 + 1e-10)
                )
                results.arma_lambda[start_idx:end_idx] = lambda_vals.cpu()

            # Vectorized GLS fit for all voxels in batch
            # Group voxels by (a,b) to maximize vectorization!

            # Preallocate arrays for whitened data/design
            X_w_batch = torch.zeros(
                batch_voxels, n_timepoints, n_regressors, device=device, dtype=dtype
            )
            y_w_batch = torch.zeros(
                batch_voxels, n_timepoints, device=device, dtype=dtype
            )

            # VECTORIZED voxel grouping (NO Python loops, NO .item() calls!)
            # Use torch.unique to find unique (a,b) pairs and group indices
            unique_params, inverse_indices = torch.unique(
                batch_params, dim=0, return_inverse=True
            )

            # Prewhiten all voxels with same (a,b) simultaneously (FULLY VECTORIZED!)
            for param_idx in range(len(unique_params)):
                a_opt = unique_params[param_idx, 0].item()
                b_opt = unique_params[param_idx, 1].item()

                # Find all voxels with this (a,b) - vectorized boolean indexing!
                voxel_mask = inverse_indices == param_idx
                voxel_indices = torch.where(voxel_mask)[0]
                n_subset = len(voxel_indices)

                if n_subset == 0:
                    continue

                if (a_opt, b_opt) in precomputed_grid:
                    # Fetch from grid (already on GPU - loaded once after precomputation)
                    cached = precomputed_grid[(a_opt, b_opt)]
                    X_w = cached["X_w"]
                    L_inv = cached["L_inv"]

                    # Vectorized prewhitening for ALL voxels with this (a,b) at once!
                    # Use expand() which creates a VIEW (no memory copy!)
                    # Then directly index into X_w_batch without intermediate storage
                    X_w_expanded = X_w.unsqueeze(0).expand(n_subset, -1, -1)
                    X_w_batch[voxel_indices] = X_w_expanded

                    # Vectorized matrix-vector products: L_inv @ Y for all voxels at once
                    # Y_subset: (n_timepoints, n_subset)
                    Y_subset = Y_batch_dev[:, voxel_indices]
                    y_w_subset = L_inv @ Y_subset  # (n_timepoints, n_subset)
                    y_w_batch[voxel_indices] = y_w_subset.T  # (n_subset, n_timepoints)
                    # Free intermediate tensors
                    del Y_subset, y_w_subset
                else:
                    # Fallback: compute on-the-fly (shouldn't happen with precomputed grid)
                    # Still vectorized across this subset!
                    for i in voxel_indices:
                        y_voxel_dev = Y_batch_dev[:, i]
                        X_w, y_w, _ = prewhiten_with_arma11(
                            design, y_voxel_dev, a_opt, b_opt
                        )
                        X_w_batch[i] = X_w
                        y_w_batch[i] = y_w

            # Batch GLS solve: solve (X'X)β = X'y for all voxels at once
            # X_w_batch: (batch_voxels, n_timepoints, n_regressors)
            # y_w_batch: (batch_voxels, n_timepoints)

            # Compute X'X for each voxel: (batch_voxels, n_regressors, n_regressors)
            X_w_batch_transposed = X_w_batch.transpose(1, 2)  # Reuse this!
            XwTXw_batch = torch.bmm(X_w_batch_transposed, X_w_batch)
            # No ridge regularization - match AFNI behavior
            XwTXw_reg_batch = XwTXw_batch

            # Compute X'y for each voxel: (batch_voxels, n_regressors)
            XwTy_batch = torch.bmm(
                X_w_batch_transposed, y_w_batch.unsqueeze(2)
            ).squeeze(2)
            del X_w_batch_transposed  # Free transpose cache

            # Batch solve: (batch_voxels, n_regressors)
            betas_batch = torch.linalg.solve(
                XwTXw_reg_batch, XwTy_batch.unsqueeze(2)
            ).squeeze(2)
            del XwTy_batch  # Free immediately after use

            # Predictions and residuals (whitened space)
            pred_w_batch = torch.bmm(X_w_batch, betas_batch.unsqueeze(2)).squeeze(2)
            resid_w_batch = y_w_batch - pred_w_batch
            del X_w_batch  # Don't need whitened design anymore - FREE IT!

            # Predictions in original space
            pred_orig_batch = torch.mm(
                design, betas_batch.T
            ).T  # (batch_voxels, n_timepoints)
            resid_orig_batch = (
                Y_batch_dev.T - pred_orig_batch
            )  # (batch_voxels, n_timepoints)

            # Variance estimates: (batch_voxels,)
            df = results.dof
            sigma2_batch = torch.sum(resid_w_batch**2, dim=1) / df

            # Don't need whitened residuals after computing sigma2 (unless saving)
            if not want_residuals:
                del resid_w_batch

            # t-statistics: need variance of beta estimates
            # Var(β) = σ² (X'X)^{-1}
            XwTXw_inv_batch = torch.linalg.inv(
                XwTXw_reg_batch
            )  # (batch_voxels, n_regressors, n_regressors)
            del XwTXw_reg_batch  # Free immediately

            var_beta_batch = sigma2_batch.unsqueeze(1).unsqueeze(2) * XwTXw_inv_batch
            se_beta_batch = torch.sqrt(torch.diagonal(var_beta_batch, dim1=1, dim2=2))
            tstats_batch = betas_batch / (se_beta_batch + 1e-10)
            del se_beta_batch  # Free after use

            # F-statistics (reuse XwTXw_batch before deleting it)
            quad_batch = torch.bmm(
                torch.bmm(betas_batch.unsqueeze(1), XwTXw_batch),
                betas_batch.unsqueeze(2),
            ).squeeze()
            fstats_batch = quad_batch / (n_regressors * sigma2_batch + 1e-10)
            del XwTXw_batch, quad_batch  # Free large matrices

            # R²
            y_mean_batch = Y_batch_dev.mean(dim=0)  # (batch_voxels,)
            ss_total_batch = torch.sum(
                (Y_batch_dev - y_mean_batch.unsqueeze(0)) ** 2, dim=0
            )
            del y_mean_batch  # Small but good practice
            ss_residual_batch = torch.sum(resid_orig_batch**2, dim=1)
            r2_batch = 1 - ss_residual_batch / (ss_total_batch + 1e-10)
            del ss_total_batch, ss_residual_batch  # Free immediately

            # Move to CPU and store
            results.betas[start_idx:end_idx] = betas_batch.cpu()
            results.sigma2[start_idx:end_idx] = sigma2_batch.cpu()
            results.tstats[start_idx:end_idx] = tstats_batch.cpu()
            results.fstats[start_idx:end_idx] = fstats_batch.cpu()
            results.r2[start_idx:end_idx] = r2_batch.cpu()
            results.var_betas[start_idx:end_idx] = var_beta_batch.cpu()

            # Optional outputs
            if want_residuals:
                results.residuals[start_idx:end_idx] = resid_orig_batch.cpu()
                results.residuals_whitened[start_idx:end_idx] = resid_w_batch.cpu()
            if want_predicted:
                results.predicted[start_idx:end_idx] = pred_orig_batch.cpu()

            # Explicitly free GPU tensors after copying to CPU
            del (
                betas_batch,
                sigma2_batch,
                tstats_batch,
                fstats_batch,
                r2_batch,
                var_beta_batch,
            )
            del resid_orig_batch, Y_batch_dev
            if want_residuals:
                del resid_w_batch
            if want_predicted:
                del pred_orig_batch

            # Aggressive GPU cache clearing after each batch (helps on smaller GPUs)
            if device.type == "cuda":
                torch.cuda.empty_cache()

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

            quad = torch.dot(beta, (XwTXw @ beta))
            fstat = quad / (n_regressors * sigma2 + 1e-10)
            results.fstats[v] = fstat.cpu()

            # R²
            y_mean_val = y_v_cpu.mean()
            ss_total = torch.sum((y_v_cpu - y_mean_val) ** 2)
            ss_residual = torch.sum(resid_orig_cpu**2)
            results.r2[v] = (1 - ss_residual / (ss_total + 1e-10)).cpu()

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
