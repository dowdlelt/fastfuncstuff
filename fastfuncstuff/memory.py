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

import contextlib
import os
import threading
import time
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
    empty_cache: bool = True,
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
    empty_cache : bool, default True
        Release the CUDA caching allocator's free blocks before reading
        ``memory_reserved`` so ``total - reserved`` reflects truly-free memory.
        This forces a device sync and churns the allocator, so hot paths that
        size a chunk on *every* call (e.g. the separable resampler) should pass
        ``False``: skipping it leaves cached-but-free blocks counted in
        ``reserved``, which only ever *underestimates* free memory (smaller,
        safe chunks) since a new allocation can still reuse the reserved pool.

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
            if empty_cache:
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


def bytes_per_voxel_hrf_xval(
    n_timepoints: int,
    n_regressors: int,
    n_designs: int,
) -> int:
    """Conservative working memory for HRF-library cross-validation.

    Split-half CV keeps a prediction accumulator for every candidate, while
    LORO evaluates candidates sequentially. Planning for the larger split-half
    case makes both paths safe. The extra three time-series terms cover input
    data, residual/reduction temporaries, and output statistics.
    """
    return (n_designs * (n_timepoints + n_regressors) + 3 * n_timepoints) * 4


def bytes_per_voxel_prf(
    n_timepoints: int,
    n_pixels: int,
) -> int:
    """
    Estimate memory per voxel for CSS pRF Gauss-Newton refinement.

    Parameters
    ----------
    n_timepoints : int
        Total timepoints across all runs
    n_pixels : int
        Flattened stimulus aperture size (rows * columns)

    Returns
    -------
    int
        Bytes per voxel

    Notes
    -----
    Memory model: (6 * n_pixels + 30 * n_timepoints) * 4 bytes.

    Unlike the GLM operations, the aperture term dominates: every voxel carries
    its own Gaussian receptive field over ``n_pixels`` (typically 100x100), and
    the spatial derivative is formed one field at a time (~3 live fields, x2 for
    the line search trial evaluated alongside the current parameters). The
    temporal term covers the prediction, the four parameter derivatives, and the
    variable-projection Jacobian, again doubled for the line search.
    """
    return (6 * n_pixels + 30 * n_timepoints) * 4


def compute_reml_batched_search_strategy(
    n_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    n_grid: int,
    device: torch.device,
    bytes_per_element: int,
) -> tuple[int, int]:
    """
    Jointly optimise grid_chunk_size and voxel_batch_size for batched REML search.

    The batched search stacks G grid points, reads each voxel batch once, and
    computes all G likelihoods in a single batched GEMM pass.  Memory layout:

    Persistent while the inner voxel loop runs (one allocation per grid chunk):
        L_inv_stack : G × n_time × n_time  (dominant for large n_time)
        Q_stack     : G × n_time × n_reg

    Transient per voxel mini-batch (created and freed each pass):
        Y_w_all     : G × n_time × V       (dominant)
        QTYw        : G × n_reg  × V
        scalars     : G × V × 3

    The GPU cuBLAS batched-GEMM throughput plateaus at G ≈ 8–16 for typical
    matrix sizes; beyond that, adding more grid points reduces V without
    improving GPU utilisation.  We cap G at ``G_SAT`` and push remaining memory
    budget into maximising V.

    Parameters
    ----------
    n_voxels : int
    n_timepoints : int
    n_regressors : int
    n_grid : int
        Total number of (a,b) pairs in the precomputed grid.
    device : torch.device
    bytes_per_element : int
        4 for float32, 8 for float64.

    Returns
    -------
    grid_chunk : int
        Number of (a,b) pairs to stack in a single batched GEMM pass.
    voxel_batch : int
        Number of voxels to process per inner loop iteration.
    """
    config = get_memory_config()
    available = get_available_memory(device)

    bpe = bytes_per_element

    # --- Persistent cost per grid point (L_inv + Q) ---
    per_grid_persistent = (n_timepoints * n_timepoints + n_timepoints * n_regressors) * bpe

    # --- Transient cost per (grid_point × voxel) ---
    # Y_w column + QTYw column + 3 scalars
    per_grid_per_voxel = (n_timepoints + n_regressors + 3) * bpe

    # cuBLAS batched-GEMM efficiency plateau — beyond this G adds no GPU benefit
    G_SAT = 16

    # Budget: 40 % for the persistent L_inv + Q stack, 50 % for transient Y_w.
    # The remaining 10 % is implicit headroom for PyTorch allocator overhead.
    persistent_budget = int(available * 0.40)
    transient_budget = int(available * 0.50)

    # Max grid chunk constrained by persistent memory (and the saturation point)
    max_grid_mem = max(1, persistent_budget // per_grid_persistent)
    grid_chunk = min(n_grid, max_grid_mem, G_SAT)

    # Max voxel batch given the chosen grid chunk
    if grid_chunk > 0 and per_grid_per_voxel > 0:
        max_voxel_mem = transient_budget // (grid_chunk * per_grid_per_voxel)
    else:
        max_voxel_mem = config.min_chunk_size

    voxel_batch = int(max_voxel_mem)
    voxel_batch = min(voxel_batch, n_voxels)

    # Clamp to configured limits
    if device.type == "cuda":
        voxel_batch = min(voxel_batch, config.max_chunk_size_gpu)
    else:
        voxel_batch = min(voxel_batch, config.max_chunk_size_cpu)
    voxel_batch = max(voxel_batch, config.min_chunk_size)

    return grid_chunk, voxel_batch


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


def bytes_per_voxel_ica_varnorm(
    n_timepoints: int,
) -> int:
    """
    Estimate memory per voxel for ICA variance normalization chunked ops.

    The MELODIC varnorm step computes residual = x_t - dewhite @ ws per voxel,
    then takes std.  When chunked, the peak per-voxel cost is:
      - reconstruction column (T floats)
      - residual column       (T floats)

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints

    Returns
    -------
    int
        Bytes per voxel (float32)
    """
    # reconstruction + residual, both (T,) per voxel
    return n_timepoints * 4 * 2


def bytes_per_voxel_arma_search(
    n_timepoints: int,
    n_regressors: int,
) -> int:
    """Memory per voxel for the REML grid-search inner loop.

    Used by ``estimate_arma11_per_voxel`` (and ``ffs_reml``'s search
    path) when scanning all (a, b) grid points against batched voxel
    timeseries.  Per voxel, per grid point we materialise a whitened
    Y row (n_timepoints float32), then a small Q'Y_w (n_regressors
    float32) for the Pythagorean RSS, plus per-voxel result/best-
    likelihood scratch.  The L_inv and Q matrices themselves are
    shared across voxels (per grid point) — they live in the
    precomputed grid, not in the per-voxel chunk.

    Memory model: ``(3 * n_timepoints + n_regressors + 4) * 4 bytes``
    - Y row on device                : n_timepoints * 4
    - Y_w transient (L_inv @ Y)      : n_timepoints * 4
    - resid-scratch (Q' Y_w etc.)    : n_timepoints * 4 (upper bound)
    - Q'Y_w + likelihood scratch     : n_regressors * 4
    - best_params / best_likelihoods : 4 floats
    """
    return (3 * n_timepoints + n_regressors + 4) * 4


def bytes_per_voxel_lss(
    n_timepoints: int,
    n_regressors_total: int,
    n_trials: int,
) -> int:
    """Memory per voxel for the batched Least-Squares-Separate solve.

    ``n_regressors_total`` is K_total per trial (trial K + rest-of-cond
    K + per-other-cond × K).  ``n_trials`` is the per-condition trial
    count being batched in the current solve.

    Per voxel during a chunk:
    - y_chunk slice (float32 on device)        : n_timepoints * 4
    - batched X.T @ y RHS (float64)            : n_trials * K_total * 8
    - batched β result (float64)               : n_trials * K_total * 8
    - transient (resid scratch, etc.)          : n_trials * K_total * 8
    """
    return n_timepoints * 4 + 3 * n_trials * n_regressors_total * 8


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


def bytes_per_voxel_arma_dsort(
    n_timepoints: int,
    n_regressors: int,
    n_dsort: int,
) -> int:
    """Memory per voxel for the ``-dsort`` (ANATICOR) GLS pass.

    Unlike the shared-design GLS path, ``-dsort`` gives every voxel its own
    ``q`` extra columns, so the design (and its X'X / inverse) cannot be shared
    across the batch — each voxel carries its own extended-design workspace.
    This is the "near duplication" the per-voxel solve costs.

    Per voxel during a sub-batch (``p = n_regressors``, ``q = n_dsort``,
    ``p_ext = p + q``):
    - extended whitened design X_ext : n_timepoints * p_ext
    - X'X, Cholesky factor, inverse  : 3 * p_ext^2
    - whitened Y row + dsort cols    : (1 + q) * n_timepoints
    - residual / prediction scratch  : 2 * n_timepoints

    Returned in float32 bytes; ``estimate_chunk_size`` scales for float64.
    """
    p_ext = n_regressors + n_dsort
    return (n_timepoints * p_ext + 3 * p_ext * p_ext + (3 + n_dsort) * n_timepoints) * 4


def compute_moco_resample_batch_size(
    nz: int,
    ny: int,
    nx: int,
    nt: int,
    device: torch.device,
    interp: str = "wsinc5",
) -> int:
    """Compute batch size for Pass 2 motion-correction resampling.

    In the batched Pass 2 loop we preload B source volumes to the GPU,
    compute all B coordinate transforms in a single batched matmul, then
    loop through the B volumes calling the separable resampling kernel.

    Per-batch GPU memory:
      - B source volumes:   B * nz*ny*nx * 4 bytes
      - B output volumes:   B * nz*ny*nx * 4 bytes
      - B coord transforms (src_x, src_y, src_z):  B * 3 * nz*ny*nx * 4 bytes
      → per-volume cost: 5 * vol_bytes

    One-time kernel overhead (constant, independent of B):
      - z_results, y_results, x_vals, weights, index arrays: ~6*ntaps * vol_bytes

    Parameters
    ----------
    nz, ny, nx : int
        Spatial dimensions of one volume.
    nt : int
        Total number of timepoints (caps the batch size).
    device : torch.device
        Target compute device.
    interp : str
        Interpolation method; determines ntaps (wsinc5→11, heptic→8,
        quintic→6, cubic→4, linear→1).

    Returns
    -------
    int
        Batch size (number of volumes per GPU batch), at least 1.
    """
    _ntaps = {"wsinc5": 11, "heptic": 8, "quintic": 6, "cubic": 4, "linear": 1}
    ntaps = _ntaps.get(interp, 11)

    vol_bytes = nz * ny * nx * 4  # float32
    per_vol_bytes = 5 * vol_bytes
    kernel_overhead_bytes = 6 * ntaps * vol_bytes

    available = get_available_memory(device)
    usable = available - kernel_overhead_bytes
    if usable <= 0:
        return 1

    batch_size = max(1, min(nt, usable // per_vol_bytes))
    return batch_size


def estimate_nordic_llr_memory(
    shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int, int],
    svd_batch_size: int,
    dtype_bytes: int,
    return_recon: bool = True,
    n_echoes: int = 1,
) -> dict[str, int]:
    """Estimate GPU memory breakdown for a single ``_llr_denoise`` call.

    Parameters
    ----------
    shape : (nx, ny, nz, nt)
        Volume shape (one echo).
    kernel_size : (wx, wy, wz)
        Patch kernel size (clamped to volume dims internally).
    svd_batch_size : int
        Batch size for decomposition.
    dtype_bytes : int
        Bytes per element (8 for complex64, 4 for float32).
    return_recon : bool
        Whether reconstruction is enabled (allocates recon_acc).
    n_echoes : int
        Number of echoes processed jointly (the multi-echo rescue path holds
        E echoes' data + E recon accumulators, and keeps all E echoes' temporal
        singular vectors ``vh`` live per batch for the cross-echo test). For
        ``n_echoes == 1`` this is exactly the single-echo footprint.

    Returns
    -------
    dict with keys:
        data : input volume bytes (already on GPU), summed over echoes
        recon_acc : reconstruction accumulator bytes, summed over echoes
        diag_maps : diagnostic map bytes
        batch_working : peak per-batch working set bytes
        total : sum of above
    """
    nx, ny, nz, nt = shape
    wx = min(kernel_size[0], nx)
    wy = min(kernel_size[1], ny)
    wz = min(kernel_size[2], nz)
    M = wx * wy * wz
    N = nt
    K = min(M, N)
    B = svd_batch_size
    n_spatial = nx * ny * nz
    E = max(1, n_echoes)

    data_bytes = E * n_spatial * nt * dtype_bytes
    recon_bytes = (E * n_spatial * nt * dtype_bytes) if return_recon else 0
    # weight + 4 maps (single-echo) or weight + 2 maps/echo (multi-echo).
    diag_bytes = (5 if E == 1 else 1 + 2 * E) * n_spatial * 4  # float32

    # Peak batch working set.
    #  single-echo (eigh path, worst case): 3*(B,M,N) + (B,N,N)
    #  multi-echo: all E temporal vectors vh(B,K,N) live at once for the
    #    cross-echo test, plus one echo's transient mats/proj/recon
    #    (2*(B,M,N) + (B,N,N)) during pass-3 reconstruction.
    if E == 1:
        batch_bytes = (3 * B * M * N + B * N * N) * dtype_bytes
    else:
        batch_bytes = (E * B * K * N + 2 * B * M * N + B * N * N) * dtype_bytes

    return {
        "data": data_bytes,
        "recon_acc": recon_bytes,
        "diag_maps": diag_bytes,
        "batch_working": batch_bytes,
        "total": data_bytes + recon_bytes + diag_bytes + batch_bytes,
    }


def plan_nordic_llr_memory(
    shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int, int],
    svd_batch_size: int,
    dtype_bytes: int,
    avail_bytes: int,
    return_recon: bool = True,
    n_echoes: int = 1,
    min_batch: int = 1,
) -> dict[str, object]:
    """Fit a NORDIC LLR pass into ``avail_bytes`` of GPU memory.

    The recon accumulator and diagnostic maps must stay resident on the GPU for
    the whole pass (they receive scattered per-batch reconstructions), so the
    only two knobs are: keep vs. offload the input volume, and how many patches
    to decompose per batch. The per-batch working set scales linearly with the
    batch size, so once the input is offloaded we can always shrink the batch to
    fit whatever budget remains after the fixed accumulators.

    Offloading the input is *not* sufficient on its own: when the working set —
    not the input — is what overflows (large patch M×N, big batch), the batch
    size must come down too. Callers that only offload the input will OOM in
    that regime.

    Returns
    -------
    dict with keys:
        offload_data : bool  — move the input volume to CPU and stream patches.
        svd_batch_size : int — batch size that fits the budget (>= ``min_batch``).
        fits : bool          — False if even ``min_batch`` overruns the budget
                               (the resident accumulators alone exceed ``avail``);
                               caller should proceed as best-effort and expect
                               memory pressure.
        est : dict           — the estimate at the *original* batch size.
    """
    est = estimate_nordic_llr_memory(
        shape, kernel_size, svd_batch_size, dtype_bytes, return_recon, n_echoes
    )
    # Everything fits at the requested batch size — no changes needed.
    if est["total"] <= avail_bytes:
        return {
            "offload_data": False,
            "svd_batch_size": svd_batch_size,
            "fits": True,
            "est": est,
        }

    # Doesn't fit: offload the input and size the batch to the remaining budget.
    per_batch = est["batch_working"] / max(1, svd_batch_size)  # bytes / batch-unit
    fixed_resident = est["recon_acc"] + est["diag_maps"]  # must stay on GPU
    budget = avail_bytes - fixed_resident
    if budget <= 0 or per_batch <= 0:
        # Even the accumulators don't fit; run at min_batch and warn upstream.
        return {
            "offload_data": True,
            "svd_batch_size": max(1, min_batch),
            "fits": False,
            "est": est,
        }

    fitted = int(budget // per_batch)
    fitted = max(min_batch, min(svd_batch_size, fitted))
    return {
        "offload_data": True,
        "svd_batch_size": fitted,
        "fits": fitted >= min_batch and budget > 0,
        "est": est,
    }


# NORDIC LLR budgets an *itemized* footprint (estimate_nordic_llr_memory counts
# every large tensor explicitly), so it does not need the blanket multiplicative
# model-error cushion the per-voxel chunk estimators rely on. It needs only slack
# for the caching allocator's reservation/fragmentation, plus courtesy on a
# shared card. A fixed reserve protects other tenants when the card is nearly
# full; a fraction cap avoids hogging a big idle card.
NORDIC_GPU_RESERVE_BYTES = 2 * 1024**3  # allocator + fragmentation headroom
NORDIC_GPU_MAX_FRACTION = 0.8  # never grab more than this share of free memory


def _nordic_budget_from_free(free_bytes: int) -> int:
    """Policy: bytes usable for a NORDIC LLR pass given ``free_bytes`` free.

    ``free - reserve`` floored against ``fraction * free`` — the reserve binds on
    a nearly-full card, the fraction binds on a large idle one.
    """
    return max(
        0,
        min(free_bytes - NORDIC_GPU_RESERVE_BYTES, int(free_bytes * NORDIC_GPU_MAX_FRACTION)),
    )


def nordic_llr_gpu_budget(device: torch.device) -> int:
    """GPU memory budget (bytes) for a NORDIC LLR pass.

    Unlike :func:`get_available_memory`'s flat multiplicative safety factor, this
    uses a fixed reserve floored against a fraction of free memory, which suits
    the itemized LLR estimate (see :func:`estimate_nordic_llr_memory`). An
    explicit ``FFS_GPU_SAFETY_FACTOR`` override still wins, and non-CUDA devices
    or query failures fall back to :func:`get_available_memory`.
    """
    if device.type != "cuda" or "FFS_GPU_SAFETY_FACTOR" in os.environ:
        return get_available_memory(device)
    try:
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        free = total - reserved
    except Exception:
        return get_available_memory(device)
    return _nordic_budget_from_free(free)


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
    n_fractions: int = 100,
    n_trials: int = 1,
    n_designs: int = 1,
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
        Type of operation: "glm", "xval", "ridge", "denoise", "arma", "prf"
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
    elif operation == "hrf_xval":
        bytes_per_voxel = bytes_per_voxel_hrf_xval(n_timepoints, n_regressors, n_designs)
    elif operation == "ridge":
        bytes_per_voxel = bytes_per_voxel_ridge(
            n_timepoints,
            n_regressors,
            n_fractions=n_fractions,
        )
    elif operation == "denoise":
        bytes_per_voxel = bytes_per_voxel_denoise(n_timepoints, n_regressors)
    elif operation == "ica_varnorm":
        bytes_per_voxel = bytes_per_voxel_ica_varnorm(n_timepoints)
    elif operation == "prf":
        # n_regressors carries the aperture pixel count, which is what the pRF
        # refinement actually scales with (see bytes_per_voxel_prf).
        bytes_per_voxel = bytes_per_voxel_prf(n_timepoints, n_regressors)
    elif operation == "arma":
        bytes_per_voxel = bytes_per_voxel_arma(n_timepoints, n_regressors)
    elif operation == "arma_search":
        bytes_per_voxel = bytes_per_voxel_arma_search(n_timepoints, n_regressors)
    elif operation == "arma_dsort":
        # n_trials carries the dsort regressor count q (per the lss precedent of
        # overloading n_trials for an operation-specific multiplicity).
        bytes_per_voxel = bytes_per_voxel_arma_dsort(n_timepoints, n_regressors, n_dsort=n_trials)
    elif operation == "lss":
        bytes_per_voxel = bytes_per_voxel_lss(
            n_timepoints,
            n_regressors,
            n_trials=n_trials,
        )
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


class VRAMDebugger:
    """
    Context manager that samples peak GPU allocation during chunk processing
    and compares it to the memory module's prediction.

    Sampling uses ``torch.cuda.memory_allocated()`` from a background daemon
    thread.  That call reads PyTorch's internal allocator counter — an atomic
    read with no CUDA driver interaction and no GPU synchronisation.  Overhead
    is in the microsecond range per sample; the thread sleeps between samples.

    Parameters
    ----------
    device : torch.device
        CUDA device to monitor.
    predicted_peak_bytes : int
        Expected peak allocation for one chunk (``chunk_size × bytes_per_voxel``).
    operation : str
        Label for the report header (e.g. ``"glm"``, ``"xval"``).
    chunk_size : int
        Chunk size used, for reporting.
    sample_ms : float, default=25
        Sampling interval in milliseconds.  25ms catches peaks in operations
        that take > ~50ms; lower it if chunks are very fast.

    Examples
    --------
    As a context manager around the chunk loop::

        predicted = chunk_size * bytes_per_voxel_glm(n_tp, n_reg)
        with VRAMDebugger(device, predicted, operation="glm", chunk_size=chunk_size):
            for start in range(0, n_voxels, chunk_size):
                chunk = data[start : start + chunk_size].to(device)
                results[start : start + chunk_size] = fit_glm_chunk(chunk, design).cpu()

    Or use ``make_vram_debugger()`` so the same code path is a no-op when
    debug mode is off.
    """

    def __init__(
        self,
        device: torch.device,
        predicted_peak_bytes: int,
        operation: str = "unknown",
        chunk_size: int = 0,
        sample_ms: float = 25.0,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("VRAMDebugger only supports CUDA devices")
        self.device = device
        self.predicted_peak_bytes = predicted_peak_bytes
        self.operation = operation
        self.chunk_size = chunk_size
        self._sample_interval = sample_ms / 1000.0
        self._baseline: int = 0
        self._peak: int = 0
        self._n_samples: int = 0
        self._running: bool = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Flush pending GPU work, snapshot baseline, start sampling thread."""
        torch.cuda.synchronize(self.device)  # flush queued ops so baseline is clean
        self._baseline = torch.cuda.memory_allocated(self.device)
        self._peak = self._baseline
        self._n_samples = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="ffs-vram-debug"
        )
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Stop the sampling thread and print the comparison report."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.report()

    def _sample_loop(self) -> None:
        # Hot loop: atomic read + compare, then sleep.
        # torch.cuda.memory_allocated() never touches the CUDA driver.
        while self._running:
            current = torch.cuda.memory_allocated(self.device)
            if current > self._peak:
                self._peak = current
            self._n_samples += 1
            time.sleep(self._sample_interval)

    def report(self) -> dict[str, float]:
        """Print comparison and return a stats dict."""
        _gb = 1024**3
        net_peak = self._peak - self._baseline
        predicted = self.predicted_peak_bytes
        ratio = net_peak / predicted if predicted > 0 else float("nan")

        if ratio < 0.5:
            verdict = "VERY CONSERVATIVE — model over-predicted >2x; chunk size could be larger"
        elif ratio < 0.8:
            verdict = "conservative — safely under-utilized"
        elif ratio <= 1.05:
            verdict = "accurate — model matched reality"
        elif ratio <= 1.25:
            verdict = "TIGHT — model under-predicted; within 25%, watch for OOM on other hardware"
        else:
            verdict = "UNDER-PREDICTED — actual exceeded model; OOM risk was real"

        print(f"\n[VRAM DEBUG] operation={self.operation}  chunk_size={self.chunk_size:,}")
        print(f"  Baseline (pre-loop):    {self._baseline / _gb:.3f} GB")
        print(f"  Peak (during loop):     {self._peak / _gb:.3f} GB")
        print(f"  Net delta:              {net_peak / _gb:.3f} GB")
        print(f"  Model predicted:        {predicted / _gb:.3f} GB")
        print(f"  Ratio actual/predicted: {ratio:.2f}x  →  {verdict}")
        print(f"  Samples: {self._n_samples} @ {self._sample_interval * 1000:.0f} ms intervals")

        return {
            "baseline_gb": self._baseline / _gb,
            "peak_gb": self._peak / _gb,
            "net_peak_gb": net_peak / _gb,
            "predicted_gb": predicted / _gb,
            "ratio": ratio,
            "n_samples": self._n_samples,
        }

    def __enter__(self) -> VRAMDebugger:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def make_vram_debugger(
    device: torch.device,
    predicted_peak_bytes: int,
    operation: str = "unknown",
    chunk_size: int = 0,
    enabled: bool | None = None,
    sample_ms: float = 25.0,
) -> VRAMDebugger | contextlib.AbstractContextManager[None]:
    """
    Return a :class:`VRAMDebugger` when debug mode is active, otherwise a
    no-op context manager.  Use this so a single code path handles both modes.

    Parameters
    ----------
    device : torch.device
    predicted_peak_bytes : int
        Predicted peak bytes for one chunk (``chunk_size × bytes_per_voxel``).
    operation : str
        Label for the report.
    chunk_size : int
        Chunk size, for the report.
    enabled : bool, optional
        If ``None``, checks the ``FFS_DEBUG_VRAM`` environment variable
        (any non-empty value other than ``"0"``, ``"false"``, or ``"no"``
        activates it).  Pass ``True`` / ``False`` to override.
    sample_ms : float, default=25
        Sampling interval passed to :class:`VRAMDebugger`.

    Returns
    -------
    context manager
        Wrap your chunk loop with ``with make_vram_debugger(...): ...``.

    Examples
    --------
    In a CLI that accepts ``-debug``::

        dbg = make_vram_debugger(
            device, chunk_size * bytes_per_voxel_glm(n_tp, n_reg),
            operation="glm", chunk_size=chunk_size,
            enabled=args.debug,
        )
        with dbg:
            for start in range(0, n_voxels, chunk_size):
                ...

    Via environment variable (no code change needed)::

        FFS_DEBUG_VRAM=1 ffs_ridge -input data.nii.gz ...
    """
    if enabled is None:
        raw = os.environ.get("FFS_DEBUG_VRAM", "0").strip().lower()
        enabled = raw not in ("", "0", "false", "no")

    if enabled and device.type == "cuda":
        return VRAMDebugger(device, predicted_peak_bytes, operation, chunk_size, sample_ms)
    return contextlib.nullcontext()
