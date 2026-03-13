"""
Unified memory management and chunking strategy for fastfuncstuff.

This module provides centralized functions for estimating memory requirements
and determining optimal chunk sizes for GPU-accelerated fMRI processing.

Key principles:
- Conservative memory estimation to avoid OOM errors
- Device-aware optimization (GPU vs CPU vs MPS)
- Per-operation memory models based on actual algorithm requirements
- Unified chunk size estimation across all modules
- User-configurable safety factors via MemoryConfig

Configuration
-------------
Memory behavior can be customized globally via MemoryConfig:

    from fastfuncstuff.memory import MemoryConfig, set_memory_config

    # Use 70% of available GPU memory (more aggressive)
    config = MemoryConfig(gpu_safety_factor=0.7)
    set_memory_config(config)

    # Or use 30% of available GPU memory (more conservative)
    config = MemoryConfig(gpu_safety_factor=0.3)
    set_memory_config(config)

The configuration applies to all chunk size calculations automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class MemoryConfig:
    """
    Global memory configuration for fastfuncstuff.

    This class centralizes all memory-related parameters that affect
    chunk size estimation and memory usage across the package.

    Attributes
    ----------
    gpu_safety_factor : float, default=0.5
        Fraction of available GPU memory to use (0 < factor <= 1).
        Lower values = more conservative (fewer OOM errors but smaller chunks).
        Higher values = more aggressive (larger chunks but OOM risk).
        Recommended: 0.3-0.5 for limited VRAM, 0.6-0.8 for abundant VRAM.

    cpu_safety_factor : float, default=0.75
        Fraction of system RAM to use for CPU processing.
        Aggressive by default (75%) since RAM is typically abundant (64-256GB+).
        Uses psutil to query actual available memory.

    min_chunk_size : int, default=1000
        Minimum chunk size (voxels). Prevents excessive overhead from tiny chunks.

    max_chunk_size_gpu : int, default=90000
        Maximum chunk size on GPU. Limits peak memory even if available.

    max_chunk_size_cpu : int, default=1000000
        Maximum chunk size on CPU (1M voxels). High since RAM is typically 64-256GB+.

    data_threshold_gb : float, default=4.0
        Dataset size threshold (GB) for deciding GPU vs CPU data storage.
        Datasets larger than this will use CPU storage with GPU streaming.

    arma_safety_factor : float, default=0.6
        Safety factor specifically for ARMA operations, which have higher
        peak memory due to Cholesky decomposition and matrix operations.

    double_precision_multiplier : float, default=2.0
        Memory multiplier when using float64 instead of float32.

    Examples
    --------
    Default configuration (balanced):
        >>> config = MemoryConfig()  # 50% GPU, 75% CPU memory usage

    Aggressive configuration (for high-VRAM GPUs):
        >>> config = MemoryConfig(gpu_safety_factor=0.7, max_chunk_size_gpu=150000)

    Conservative configuration (for limited VRAM):
        >>> config = MemoryConfig(gpu_safety_factor=0.3, min_chunk_size=500)
    """

    gpu_safety_factor: float = 0.5
    cpu_safety_factor: float = 0.75
    min_chunk_size: int = 1000
    max_chunk_size_gpu: int = 90000
    max_chunk_size_cpu: int = 1000000
    data_threshold_gb: float = 4.0
    arma_safety_factor: float = 0.6
    double_precision_multiplier: float = 2.0

    def __post_init__(self):
        if not 0 < self.gpu_safety_factor <= 1:
            raise ValueError(f"gpu_safety_factor must be in (0, 1], got {self.gpu_safety_factor}")
        if not 0 < self.cpu_safety_factor <= 1:
            raise ValueError(f"cpu_safety_factor must be in (0, 1], got {self.cpu_safety_factor}")
        if self.min_chunk_size < 100:
            raise ValueError(f"min_chunk_size must be >= 100, got {self.min_chunk_size}")


# Global configuration instance (singleton pattern)
_global_config: MemoryConfig | None = None


def get_memory_config() -> MemoryConfig:
    """
    Get the global memory configuration.

    Returns
    -------
    MemoryConfig
        The current global memory configuration. If not set, returns
        a default configuration, optionally initialized from environment
        variables.

    Notes
    -----
    Environment variables can override defaults:
    - FFS_GPU_SAFETY_FACTOR: Override gpu_safety_factor
    - FFS_MIN_CHUNK_SIZE: Override min_chunk_size
    - FFS_MAX_CHUNK_SIZE_GPU: Override max_chunk_size_gpu
    - FFS_DATA_THRESHOLD_GB: Override data_threshold_gb
    """
    global _global_config

    if _global_config is None:
        # Check environment variables
        env_overrides = {}
        if "FFS_GPU_SAFETY_FACTOR" in os.environ:
            env_overrides["gpu_safety_factor"] = float(os.environ["FFS_GPU_SAFETY_FACTOR"])
        if "FFS_CPU_SAFETY_FACTOR" in os.environ:
            env_overrides["cpu_safety_factor"] = float(os.environ["FFS_CPU_SAFETY_FACTOR"])
        if "FFS_MIN_CHUNK_SIZE" in os.environ:
            env_overrides["min_chunk_size"] = int(os.environ["FFS_MIN_CHUNK_SIZE"])
        if "FFS_MAX_CHUNK_SIZE_GPU" in os.environ:
            env_overrides["max_chunk_size_gpu"] = int(os.environ["FFS_MAX_CHUNK_SIZE_GPU"])
        if "FFS_MAX_CHUNK_SIZE_CPU" in os.environ:
            env_overrides["max_chunk_size_cpu"] = int(os.environ["FFS_MAX_CHUNK_SIZE_CPU"])
        if "FFS_DATA_THRESHOLD_GB" in os.environ:
            env_overrides["data_threshold_gb"] = float(os.environ["FFS_DATA_THRESHOLD_GB"])
        if "FFS_ARMA_SAFETY_FACTOR" in os.environ:
            env_overrides["arma_safety_factor"] = float(os.environ["FFS_ARMA_SAFETY_FACTOR"])

        _global_config = MemoryConfig(**env_overrides)

    return _global_config


def set_memory_config(config: MemoryConfig) -> None:
    """
    Set the global memory configuration.

    Parameters
    ----------
    config : MemoryConfig
        The configuration to use globally.

    Examples
    --------
    >>> config = MemoryConfig(gpu_safety_factor=0.7)
    >>> set_memory_config(config)
    >>> # All subsequent operations use 70% of available GPU memory
    """
    global _global_config
    _global_config = config


def reset_memory_config() -> None:
    """
    Reset the global memory configuration to defaults.

    This clears any custom configuration and environment variable overrides.
    """
    global _global_config
    _global_config = None


# Legacy constants (for backwards compatibility)
DEFAULT_MIN_CHUNK_SIZE = 1000
DEFAULT_MAX_CHUNK_SIZE_GPU = 90000
DEFAULT_MAX_CHUNK_SIZE_CPU = 350000
DEFAULT_GPU_MEMORY_SAFETY_FACTOR = 0.5
DEFAULT_CPU_MEMORY_THRESHOLD_GB = 4.0


def get_available_memory(
    device: torch.device,
    safety_factor: float | None = None,
) -> int:
    """
    Get available memory on the specified device in bytes.

    Parameters
    ----------
    device : torch.device
        Target compute device
    safety_factor : float, optional
        Fraction of available memory to use (0 < safety_factor <= 1).
        If None, uses config defaults (0.5 for GPU, 0.75 for CPU).

    Returns
    -------
    int
        Available memory in bytes

    Notes
    -----
    - For GPU: Returns free GPU memory * gpu_safety_factor (default 50%)
    - For CPU: Queries actual system RAM via psutil, uses cpu_safety_factor (default 75%)
    - For MPS: Uses unified memory estimate based on system RAM
    """
    config = get_memory_config()

    if device.type == "cuda":
        if safety_factor is None:
            safety_factor = config.gpu_safety_factor
        try:
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            free = total - reserved
            return int(free * safety_factor)
        except Exception:
            return 2 * 1024**3
    elif device.type == "mps":
        if safety_factor is None:
            safety_factor = config.cpu_safety_factor
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            return int(available_gb * 1024**3 * safety_factor)
        except ImportError:
            return int(8 * 1024**3 * safety_factor)
    else:
        if safety_factor is None:
            safety_factor = config.cpu_safety_factor
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_bytes = mem.available
            return int(available_bytes * safety_factor)
        except ImportError:
            return int(128 * 1024**3 * safety_factor)


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
    device: torch.device | None = None,
    operation: str = "glm",
    cv_strategy: int | float | str | None = None,
    n_runs: int = 1,
    max_components: int = 0,
    data_location: str = "auto",
    streaming_stats: bool | None = None,
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
    safety_factor: float | None = None,
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
    config = get_memory_config()

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set defaults from config
    if min_chunk_size is None:
        min_chunk_size = min(config.min_chunk_size, n_voxels)

    if max_chunk_size is None:
        if device.type == "cuda":
            max_chunk_size = min(config.max_chunk_size_gpu, n_voxels)
        else:
            max_chunk_size = min(config.max_chunk_size_cpu, n_voxels)

    if safety_factor is None:
        safety_factor = (
            config.gpu_safety_factor if device.type == "cuda" else config.cpu_safety_factor
        )

    # Auto-detect data location
    if data_location == "auto":
        # Assume CPU for large datasets on GPU to avoid OOM
        data_size_gb = (n_voxels * n_timepoints * 4) / (1024**3)
        if device.type == "cuda" and data_size_gb > config.data_threshold_gb:
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
        print("\n  Memory calculation:")
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
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
    safety_factor: float | None = None,
    use_double: bool = False,
    verbose: bool = False,
) -> int:
    """
    Estimate optimal chunk size for memory-efficient processing.

    This is the unified chunk size estimator for all fastfuncstuff operations.
    Automatically uses appropriate defaults for CPU (aggressive) vs GPU (conservative).

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
    min_chunk_size : int, optional
        Minimum chunk size. Default: from config (1000)
    max_chunk_size : int, optional
        Maximum chunk size. Default: 90000 for GPU, 1000000 for CPU
    safety_factor : float, optional
        Fraction of available memory to use. Default: 0.5 GPU, 0.75 CPU
    use_double : bool, default=False
        If True, account for float64 (2x memory)
    verbose : bool, default=False
        Print chunk size estimation details

    Returns
    -------
    int
        Optimal chunk size (number of voxels)

    Notes
    -----
    CPU vs GPU behavior:
    - CPU: Aggressive! Uses 75% of available RAM by default. Data is already
      in RAM, so chunking is for computation efficiency, not memory limits.
    - GPU: Conservative! Uses 50% of VRAM by default. We stream chunks from
      CPU to GPU, so careful chunking prevents OOM errors.

    The chunk size is determined by:
    1. Per-voxel memory requirement for the operation
    2. Available device memory (from config)
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

    Customize for aggressive GPU usage:
    >>> from fastfuncstuff.memory import MemoryConfig, set_memory_config
    >>> config = MemoryConfig(gpu_safety_factor=0.7)
    >>> set_memory_config(config)
    """
    config = get_memory_config()

    # Set defaults from config
    if min_chunk_size is None:
        min_chunk_size = min(config.min_chunk_size, n_voxels)

    if max_chunk_size is None:
        if device.type == "cuda":
            max_chunk_size = min(config.max_chunk_size_gpu, n_voxels)
        else:
            max_chunk_size = min(config.max_chunk_size_cpu, n_voxels)

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
        bytes_per_voxel = bytes_per_voxel_glm(n_timepoints, n_regressors)

    # Adjust for double precision
    if use_double:
        bytes_per_voxel = int(bytes_per_voxel * config.double_precision_multiplier)

    # Get available memory (uses config defaults if safety_factor is None)
    available_bytes = get_available_memory(device, safety_factor)

    # Estimate chunk size based on memory
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
        effective_safety = (
            safety_factor
            if safety_factor
            else (config.gpu_safety_factor if device.type == "cuda" else config.cpu_safety_factor)
        )
        print("\n  Chunk size estimation:")
        print(f"    Operation: {operation}")
        print(f"    Device: {device}")
        print(f"    Per-voxel memory: {bytes_per_voxel:,} bytes")
        print(f"    Available memory: {available_mb:.1f} MB (safety={effective_safety})")
        print(f"    Chunk size: {chunk_size:,} voxels")
        print(f"    Memory per chunk: {memory_per_chunk_mb:.1f} MB")
        print(f"    Number of chunks: {(n_voxels + chunk_size - 1) // chunk_size}")

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
