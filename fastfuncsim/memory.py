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
DEFAULT_MAX_CHUNK_SIZE_GPU = 90000
DEFAULT_MAX_CHUNK_SIZE_CPU = 350000
DEFAULT_GPU_MEMORY_SAFETY_FACTOR = 0.5  # Use 50% of available GPU memory
DEFAULT_CPU_MEMORY_THRESHOLD_GB = 4.0  # Use GPU-like chunks if data < 4GB


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
    safety_factor : float, default=0.5
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
        return 128 * 1024**3


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
    Memory model: (8 * n_timepoints + max(n_regressors, 10)) * 4 bytes
    - Includes GLM requirements (5 * n_timepoints + n_regressors)
    - Plus projection overhead: 3 * n_timepoints * 4 bytes
      * train_data_batch.T: n_timepoints * batch_voxels
      * Q_train.T @ train_data_batch.T: n_nuisance * batch_voxels (smaller)
      * Q_train @ (...): n_timepoints * batch_voxels (largest intermediate)
    - Plus cross-validation intermediates
    """
    # FIXED: Account for transpose intermediate matrix in projection
    # The operation (Q_train @ (Q_train.T @ train_data_batch.T)).T
    # creates an intermediate of size (n_timepoints, batch_voxels)
    # This requires 2 additional n_timepoints worth of memory per voxel
    return (8 * n_timepoints + max(n_regressors, 10)) * 4


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


def dyn_chunk_estimator(
    n_voxels: int,
    n_timepoints: int,
    n_task_regressors: int,
    n_nuisance_regressors: int = 0,
    device: Optional[torch.device] = None,
    operation: str = "glm",
    cv_strategy: Optional[int | float | str] = None,
    n_runs: int = 1,
    max_components: int = 0,
    data_location: str = "auto",
    streaming_stats: Optional[bool] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    safety_factor: Optional[float] = None,
    verbose: bool = False,
) -> int:
    """
    Dynamic chunk size estimator with operation-specific memory modeling.

    This is the next-generation chunk estimator that accounts for actual memory
    patterns including CV strategy, data location, and operation-specific needs.
    Currently used in 3dDenoisefast.py; will gradually replace estimate_chunk_size()
    as we validate across different operations.

    Parameters
    ----------
    n_voxels : int
        Total number of voxels to process
    n_timepoints : int
        Number of timepoints (temporal dimension)
    n_task_regressors : int
        Number of task-related regressors (trials, blocks, events, etc.)
        For single-trial: this will be large (one per trial)
        For block design: this will be small (one per condition)
    n_nuisance_regressors : int, default=0
        Number of nuisance regressors (polynomials, PCs, ICs, motion, etc.)
        Total design columns = n_task_regressors + n_nuisance_regressors
    device : torch.device, optional
        Target compute device (auto-detected if None)
    operation : str, default="glm"
        Operation type: "glm", "ridge", "denoise", "xval"
    cv_strategy : int, float, or str, optional
        Cross-validation strategy:
        - 1 or "loro": Leave-one-run-out (streaming stats)
        - int > 1: Leave-k-runs-out
        - float in (0, 1): Train fraction (e.g., 0.5 for split-half)
        - None: No cross-validation
    n_runs : int, default=1
        Number of runs (affects CV memory calculation)
    max_components : int, default=0
        For denoising: maximum number of PCs/ICs to test
        Affects memory when storing predictions for all component counts
    data_location : str, default="auto"
        Where full data lives: "cpu", "gpu", or "auto" (detect from device)
        "cpu" = data on CPU, chunks stream to GPU (lower GPU memory)
        "gpu" = data on GPU (higher GPU memory usage)
    streaming_stats : bool, optional
        Whether to use streaming statistics (LORO) vs full accumulators
        If None, auto-detected from cv_strategy (LORO → True, else → False)
    min_chunk_size : int, optional
        Minimum chunk size (default: 1000 or n_voxels if smaller)
    max_chunk_size : int, optional
        Maximum chunk size (default: device-dependent)
    safety_factor : float, optional
        Fraction of available memory to use (0 < safety_factor <= 1)
        Default: 0.3 for GPU, 0.5 for CPU
    verbose : bool, default=False
        Print detailed chunk size calculation

    Returns
    -------
    int
        Optimal chunk size (number of voxels)

    Notes
    -----
    Memory calculation accounts for:
    1. Data dimensions (n_voxels × n_timepoints)
    2. Design dimensions (n_timepoints × n_regressors)
    3. CV strategy (streaming stats vs full accumulators)
    4. Operation type (GLM vs ridge vs denoising)
    5. Device and data location

    Examples
    --------
    LORO denoising with data on CPU:
    >>> chunk_size = dyn_chunk_estimator(
    ...     n_voxels=300000,
    ...     n_timepoints=11520,
    ...     n_task_regressors=1,
    ...     n_nuisance_regressors=20,  # 20 PCs
    ...     device=torch.device("cuda"),
    ...     operation="denoise",
    ...     cv_strategy=1,  # LORO
    ...     n_runs=40,
    ...     max_components=20,
    ...     data_location="cpu",
    ... )

    Split-half ridge regression:
    >>> chunk_size = dyn_chunk_estimator(
    ...     n_voxels=50000,
    ...     n_timepoints=200,
    ...     n_task_regressors=10,
    ...     device=torch.device("cuda"),
    ...     operation="ridge",
    ...     cv_strategy=0.5,  # Split-half
    ... )
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set defaults
    if min_chunk_size is None:
        min_chunk_size = min(DEFAULT_MIN_CHUNK_SIZE, n_voxels)

    if max_chunk_size is None:
        if device.type == "cuda":
            max_chunk_size = min(DEFAULT_MAX_CHUNK_SIZE_GPU, n_voxels)
        else:
            max_chunk_size = min(DEFAULT_MAX_CHUNK_SIZE_CPU, n_voxels)

    if safety_factor is None:
        safety_factor = DEFAULT_GPU_MEMORY_SAFETY_FACTOR if device.type == "cuda" else 0.5

    # Auto-detect data location
    if data_location == "auto":
        # Assume CPU for large datasets on GPU to avoid OOM
        data_size_gb = (n_voxels * n_timepoints * 4) / (1024**3)
        if device.type == "cuda" and data_size_gb > DEFAULT_CPU_MEMORY_THRESHOLD_GB:
            data_location = "cpu"
        else:
            data_location = device.type

    # Auto-detect streaming stats from cv_strategy
    if streaming_stats is None:
        if cv_strategy is None:
            streaming_stats = False
        elif cv_strategy == 1 or cv_strategy == "loro":
            streaming_stats = True
        else:
            streaming_stats = False

    # Normalize cv_strategy to determine train/test split
    is_loro = cv_strategy == 1 or cv_strategy == "loro"
    if cv_strategy is None or cv_strategy == 1 or cv_strategy == "loro":
        # LORO: train = (n_runs - 1) / n_runs, test = 1 / n_runs
        train_fraction = (n_runs - 1) / n_runs if n_runs > 1 else 0.8
        test_fraction = 1 / n_runs if n_runs > 1 else 0.2
    elif isinstance(cv_strategy, float) and 0 < cv_strategy < 1:
        # Train fraction specified (e.g., 0.5 for split-half)
        train_fraction = cv_strategy
        test_fraction = 1 - cv_strategy
    elif isinstance(cv_strategy, int) and cv_strategy > 1:
        # Leave-k-out: train = (n_runs - k) / n_runs
        train_fraction = (n_runs - cv_strategy) / n_runs if n_runs > cv_strategy else 0.8
        test_fraction = cv_strategy / n_runs if n_runs > cv_strategy else 0.2
    else:
        # No CV or unknown: assume full dataset
        train_fraction = 1.0
        test_fraction = 0.0

    # Calculate memory per voxel based on operation and CV strategy
    n_total_regressors = n_task_regressors + n_nuisance_regressors
    n_train_tps = int(n_timepoints * train_fraction)
    n_test_tps = int(n_timepoints * test_fraction)

    operation = operation.lower()

    if operation == "denoise" and streaming_stats:
        # LORO denoising with streaming statistics
        # GPU memory per voxel:
        # - Train data chunk: n_train_tps × 4 bytes
        # - Test data chunk: n_test_tps × 4 bytes
        # - Streaming stats accumulators: (max_components + 1) × 3 × 8 bytes (negligible)
        # - Design matrices and pinv: shared across voxels (overhead)
        bytes_per_voxel = (n_train_tps + n_test_tps) * 4

        # Add small overhead for intermediate arrays
        bytes_per_voxel = int(bytes_per_voxel * 1.15)

    elif operation == "denoise" and not streaming_stats:
        # Split-half or other CV with full prediction accumulators
        # GPU memory per voxel:
        # - Data: n_timepoints × 4 bytes
        # - Predictions for each PC count: (max_components + 1) × n_timepoints × 4 bytes
        # - Actuals: n_timepoints × 4 bytes
        bytes_per_voxel = n_timepoints * (max_components + 2) * 4

    elif operation == "ridge":
        # Ridge regression with fraction grid
        # Memory per voxel: data + betas for all fractions + R² maps
        # Typical: ~10x n_timepoints for full ridge path
        bytes_per_voxel = n_timepoints * 10 * 4

    elif operation == "xval":
        # Cross-validated GLM
        if streaming_stats:
            # LORO with streaming stats
            bytes_per_voxel = (n_train_tps + n_test_tps) * 4
        else:
            # Full prediction storage
            bytes_per_voxel = n_timepoints * 6 * 4

    elif operation == "glm":
        # Standard GLM: data + design + betas + residuals
        bytes_per_voxel = (
            n_timepoints * 4  # data
            + n_total_regressors * 4  # betas
            + n_timepoints * 4  # residuals
            + n_timepoints * 4
        )  # intermediate arrays

    else:
        # Unknown operation: use conservative estimate
        bytes_per_voxel = n_timepoints * 6 * 4

    # Adjust for data location
    if data_location == "cpu" and device.type == "cuda":
        # Data on CPU, only chunks on GPU: already accounted for in per-voxel calculation
        # No adjustment needed
        pass
    elif data_location == "gpu" and device.type == "gpu":
        # Full data on GPU: need to account for it in available memory
        # But chunk calculation is per-voxel, so this is already handled
        pass

    # Get available memory
    available_bytes = get_available_memory(device, safety_factor)

    # Estimate chunk size
    if bytes_per_voxel > 0:
        estimated_chunk = available_bytes // bytes_per_voxel
    else:
        estimated_chunk = max_chunk_size

    # Apply bounds
    chunk_size = max(min_chunk_size, min(estimated_chunk, max_chunk_size))

    # Sanity check: don't exceed total voxels
    chunk_size = min(chunk_size, n_voxels)

    if verbose:
        memory_per_chunk_mb = (chunk_size * bytes_per_voxel) / (1024**2)
        available_mb = available_bytes / (1024**2)
        print(f"\n{'=' * 70}")
        print("Dynamic Chunk Size Estimation")
        print(f"{'=' * 70}")
        print(f"  Operation: {operation}")
        print(f"  Data: {n_voxels:,} voxels × {n_timepoints:,} timepoints")
        print(
            f"  Regressors: {n_task_regressors} task + {n_nuisance_regressors} nuisance = {n_total_regressors} total"
        )
        print(
            f"  CV strategy: {cv_strategy} ({'streaming stats' if streaming_stats else 'full accumulators'})"
        )
        if cv_strategy:
            print(
                f"  Train/test split: {train_fraction:.1%} / {test_fraction:.1%} ({n_train_tps}/{n_test_tps} TPs)"
            )
        print(f"  Device: {device} (data on {data_location})")
        print(f"  Max components: {max_components}" if max_components > 0 else "")
        print(f"\n  Memory calculation:")
        print(f"    Bytes per voxel: {bytes_per_voxel:,} bytes")
        print(f"    Available memory: {available_mb:.1f} MB (safety_factor={safety_factor})")
        print(f"    Estimated chunk: {estimated_chunk:,} voxels")
        print(f"    Bounds: [{min_chunk_size:,}, {max_chunk_size:,}]")
        print(f"\n  Final chunk size: {chunk_size:,} voxels")
        print(f"  Memory per chunk: {memory_per_chunk_mb:.1f} MB")
        print(f"  Number of chunks: {(n_voxels + chunk_size - 1) // chunk_size}")
        print(f"{'=' * 70}\n")

    return chunk_size


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
    gpu_safety_fraction: float = 0.6,
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
    gpu_safety_fraction : float, default=0.6
        Fraction of free GPU memory that the data may occupy. Use lower
        values when the downstream computation needs significant working
        memory (e.g., 0.25 for LORO CV which creates train-split copies).

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
            free_mem = (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_reserved(device)
            ) / (1024**3)
            return data_size_gb > free_mem * gpu_safety_fraction
        except Exception:
            return data_size_gb > data_threshold_gb
    else:
        # CPU or MPS - use threshold
        return data_size_gb > data_threshold_gb
