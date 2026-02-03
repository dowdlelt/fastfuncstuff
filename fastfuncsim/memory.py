"""
Unified memory management and chunking strategy for fastfuncsim.

This module provides centralized functions for estimating memory requirements
and determining optimal chunk sizes for GPU-accelerated fMRI processing.

Key principles:
- Conservative memory estimation to avoid OOM errors
- Device-aware optimization (GPU vs CPU vs MPS)
- Per-operation memory models based on actual algorithm requirements
- Unified chunk size estimation across all modules
"""
from __future__ import annotations

from typing import Optional

import torch

# Default configuration
DEFAULT_MIN_CHUNK_SIZE = 1000
DEFAULT_MAX_CHUNK_SIZE_GPU = 50000
DEFAULT_MAX_CHUNK_SIZE_CPU = 100000
DEFAULT_GPU_MEMORY_SAFETY_FACTOR = 0.3  # Use 30% of available GPU memory
DEFAULT_CPU_MEMORY_THRESHOLD_GB = 4.0   # Use GPU-like chunks if data < 4GB


def get_available_memory(
    device: torch.device,
    safety_factor: float = DEFAULT_GPU_MEMORY_SAFETY_FACTOR,
) -> int:
    """
    Get available memory on the specified device in bytes.

    Parameters
    ----------
    device : torch.device
        Target compute device
    safety_factor : float, default=0.3
        Fraction of available memory to use (0 < safety_factor <= 1)

    Returns
    -------
    int
        Available memory in bytes

    Notes
    -----
    - For GPU: Returns free GPU memory * safety_factor
    - For CPU: Returns a fraction of system RAM (conservative estimate)
    - For MPS: Uses conservative 4GB estimate
    """
    if device.type == "cuda":
        try:
            # Use reserved memory to account for fragmentation
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            free = total - reserved
            return int(free * safety_factor)
        except Exception:
            # Fallback to conservative 2GB estimate
            return 2 * 1024**3
    elif device.type == "mps":
        # MPS doesn't have good memory querying - conservative estimate
        return int(DEFAULT_CPU_MEMORY_THRESHOLD_GB * 1024**3 * safety_factor)
    else:
        # CPU - estimate available RAM (conservative 8GB estimate)
        # In practice, system RAM is usually sufficient for CPU processing
        return 8 * 1024**3


def bytes_per_voxel_glm(
    n_timepoints: int,
    n_regressors: int,
) -> int:
    """
    Estimate memory per voxel for GLM fitting operations.

    Accounts for: data, design matrix, betas, residuals, predictions,
    and intermediate arrays.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints in design
    n_regressors : int
        Number of regressors in design matrix

    Returns
    -------
    int
        Bytes per voxel

    Notes
    -----
    Memory model: (5 * n_timepoints + n_regressors) * 4 bytes
    - Data: n_timepoints * 4 bytes
    - Betas: n_regressors * 4 bytes
    - Residuals: n_timepoints * 4 bytes
    - Predictions: n_timepoints * 4 bytes
    - Intermediates: n_timepoints * 4 bytes
    """
    return (5 * n_timepoints + n_regressors) * 4


def bytes_per_voxel_xval(
    n_timepoints: int,
    n_regressors: int,
) -> int:
    """
    Estimate memory per voxel for cross-validation operations.

    Accounts for: GLM operations plus projection overhead and
    cross-validation intermediate arrays.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints in design
    n_regressors : int
        Number of regressors in design matrix

    Returns
    -------
    int
        Bytes per voxel

    Notes
    -----
    Memory model: (6 * n_timepoints + max(n_regressors, 10)) * 4 bytes
    - Includes GLM requirements (5 * n_timepoints + n_regressors)
    - Plus projection overhead: n_timepoints * 4 bytes
    - Plus cross-validation intermediates
    """
    return (6 * n_timepoints + max(n_regressors, 10)) * 4


def bytes_per_voxel_ridge(
    n_timepoints: int,
    n_regressors: int,
    n_fractions: int = 100,
) -> int:
    """
    Estimate memory per voxel for ridge regression operations.

    Accounts for: GLM operations plus multiple ridge fractions
    fitted simultaneously.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints in design
    n_regressors : int
        Number of regressors in design matrix
    n_fractions : int, default=100
        Number of ridge fractions to fit simultaneously

    Returns
    -------
    int
        Bytes per voxel

    Notes
    -----
    Memory model: bytes_per_voxel_glm() * n_fractions
    - Ridge regression fits all fractions simultaneously using SVD
    - Each fraction requires storing betas and predictions
    """
    base_memory = bytes_per_voxel_glm(n_timepoints, n_regressors)
    # Ridge fits all fractions simultaneously, but shares some intermediates
    # Estimate: base memory + fraction overhead
    return base_memory + (n_fractions * n_regressors * 4)


def bytes_per_voxel_denoise(
    n_timepoints: int,
    n_noise_pcs: int,
) -> int:
    """
    Estimate memory per voxel for denoising operations.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints
    n_noise_pcs : int
        Number of noise PCs being considered

    Returns
    -------
    int
        Bytes per voxel
    """
    # Data + PCs + projections + GLM intermediates
    return (6 * n_timepoints + n_noise_pcs) * 4


def bytes_per_voxel_arma(
    n_timepoints: int,
    n_regressors: int,
    max_lag: int = 10,
) -> int:
    """
    Estimate memory per voxel for ARMA-REML operations.

    ARMA has complex memory requirements due to:
    - Grid-based HRF fitting
    - ARMA prewhitening matrices (L_inv per grid point)
    - Per-voxel REML estimation

    This is a rough estimate; ARMA modules use their own specialized
    chunking logic that accounts for grid points, voxel batches, etc.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints
    n_regressors : int
        Number of regressors in design
    max_lag : int, default=10
        Maximum AR lag

    Returns
    -------
    int
        Bytes per voxel (rough estimate)

    Notes
    -----
    ARMA actually uses more sophisticated chunking via:
    - arma_glm.get_adaptive_batch_size()
    - arma_glm.compute_arma_grid_strategy()
    This function is provided for API completeness only.
    """
    # Prewhitened design: n_timepoints * n_regressors * 4
    # L_inv matrix: n_timepoints * n_timepoints * 4 (shared across voxels)
    # Per-voxel: betas, residuals, likelihood
    # This is just a rough estimate - actual ARMA uses grid-specific logic
    return (8 * n_timepoints + 2 * n_regressors + max_lag) * 4


def estimate_chunk_size(
    n_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    device: torch.device,
    operation: str = "glm",
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    safety_factor: float = DEFAULT_GPU_MEMORY_SAFETY_FACTOR,
    verbose: bool = False,
) -> int:
    """
    Estimate optimal chunk size for memory-efficient processing.

    This is the unified chunk size estimator for all fastfuncsim operations.
    It replaces the previous separate estimators in utils.py and xval.py.

    Parameters
    ----------
    n_voxels : int
        Total number of voxels to process
    n_timepoints : int
        Number of timepoints in design
    n_regressors : int
        Number of regressors in design matrix
    device : torch.device
        Target compute device
    operation : str, default="glm"
        Type of operation: "glm", "xval", "ridge", "denoise", "arma"
        Note: ARMA uses its own specialized chunking logic; this is provided
        for API completeness but ARMA modules typically use their own estimators.
    min_chunk_size : int, optional
        Minimum chunk size (default: 1000 or n_voxels if smaller)
    max_chunk_size : int, optional
        Maximum chunk size (default: 50000 for GPU, 100000 for CPU)
    safety_factor : float, default=0.3
        Fraction of available memory to use (0 < safety_factor <= 1)
    verbose : bool, default=False
        Print chunk size estimation details

    Returns
    -------
    int
        Optimal chunk size (number of voxels)

    Notes
    -----
    The chunk size is determined by:
    1. Per-voxel memory requirement for the operation
    2. Available device memory
    3. Min/max bounds to ensure reasonable chunks

    Examples
    --------
    >>> chunk_size = estimate_chunk_size(
    ...     n_voxels=50000,
    ...     n_timepoints=200,
    ...     n_regressors=10,
    ...     device=torch.device("cuda"),
    ...     operation="xval",
    ... )
    """
    # Set defaults
    if min_chunk_size is None:
        min_chunk_size = min(DEFAULT_MIN_CHUNK_SIZE, n_voxels)

    if max_chunk_size is None:
        if device.type == "cuda":
            max_chunk_size = min(DEFAULT_MAX_CHUNK_SIZE_GPU, n_voxels)
        else:
            max_chunk_size = min(DEFAULT_MAX_CHUNK_SIZE_CPU, n_voxels)

    # Get per-voxel memory requirement
    operation = operation.lower()
    if operation == "glm":
        bytes_per_voxel = bytes_per_voxel_glm(n_timepoints, n_regressors)
    elif operation == "xval":
        bytes_per_voxel = bytes_per_voxel_xval(n_timepoints, n_regressors)
    elif operation == "ridge":
        bytes_per_voxel = bytes_per_voxel_ridge(n_timepoints, n_regressors)
    elif operation == "denoise":
        bytes_per_voxel = bytes_per_voxel_denoise(n_timepoints, n_regressors)
    elif operation == "arma":
        bytes_per_voxel = bytes_per_voxel_arma(n_timepoints, n_regressors)
    else:
        print(f"WARNING: Unknown operation '{operation}', using GLM memory model")
        bytes_per_voxel = bytes_per_voxel_glm(n_timepoints, n_regressors)

    # Get available memory
    available_bytes = get_available_memory(device, safety_factor)

    # Estimate chunk size based on memory
    estimated_chunk = available_bytes // bytes_per_voxel

    # Apply bounds
    chunk_size = max(min_chunk_size, min(estimated_chunk, max_chunk_size))

    # Sanity check: don't exceed total voxels
    chunk_size = min(chunk_size, n_voxels)

    if verbose:
        memory_per_chunk_mb = (chunk_size * bytes_per_voxel) / (1024**2)
        available_mb = available_bytes / (1024**2)
        print("  Chunk size estimation:")
        print(f"    Operation: {operation}")
        print(f"    Per-voxel memory: {bytes_per_voxel} bytes")
        print(f"    Available memory: {available_mb:.1f} MB")
        print(f"    Chunk size: {chunk_size} voxels")
        print(f"    Memory per chunk: {memory_per_chunk_mb:.1f} MB")

    return chunk_size


def estimate_keep_on_cpu(
    n_voxels: int,
    n_timepoints_total: int,
    device: torch.device,
    force_cpu: bool = False,
    data_threshold_gb: float = DEFAULT_CPU_MEMORY_THRESHOLD_GB,
) -> bool:
    """
    Estimate whether to keep data on CPU based on dataset size.

    This is a convenience wrapper that was previously duplicated in CLI files.

    Parameters
    ----------
    n_voxels : int
        Number of voxels (after masking, if applicable)
    n_timepoints_total : int
        Total number of timepoints across all runs
    device : torch.device
        Target compute device
    force_cpu : bool, default=False
        Force CPU storage regardless of data size
    data_threshold_gb : float, default=4.0
        Size threshold in GB for GPU memory

    Returns
    -------
    keep_on_cpu : bool
        True if data should be kept on CPU, False if direct GPU loading is safe
    """
    data_size_gb = (n_voxels * n_timepoints_total * 4) / (1024**3)

    if force_cpu:
        return True
    elif device.type == "cuda":
        # Check actual GPU free memory if available
        try:
            free_mem = (torch.cuda.get_device_properties(device).total_memory
                       - torch.cuda.memory_reserved(device)) / (1024**3)
            return data_size_gb > free_mem * 0.6  # 60% of free memory
        except Exception:
            return data_size_gb > data_threshold_gb
    else:
        # CPU or MPS - use threshold
        return data_size_gb > data_threshold_gb
