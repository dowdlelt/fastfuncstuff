"""
Utility functions for fastfuncsim
Device management and helper functions
"""

import platform
import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    pass


def get_device(prefer_device: Optional[str] = None) -> torch.device:
    """
    Select the execution device with mandatory MPS enforcement on macOS.

    When running on macOS, FastFuncSim requires the Apple Metal Performance
    Shaders (MPS) backend. CUDA is supported on other platforms, and CPU
    execution is only used when no GPU backend is available off macOS.

    Parameters
    ----------
    prefer_device : str, optional
        Preferred device ('mps', 'cuda', 'cpu'). The specified backend must be
        available; otherwise a RuntimeError is raised.

    Returns
    -------
    device : torch.device
        The selected device.

    Raises
    ------
    RuntimeError
        If the required backend (especially MPS on macOS) is unavailable.
    """
    is_mac = platform.system() == "Darwin"

    if prefer_device is not None:
        prefer_device = prefer_device.lower()
        if prefer_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but not available. Enable Apple Metal Performance "
                    "Shaders (macOS 13+ with Apple Silicon) before running FastFuncSim."
                )
            return torch.device("mps")
        if prefer_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but not available. Ensure NVIDIA drivers and CUDA are installed."
                )
            return torch.device("cuda")
        if prefer_device == "cpu":
            if is_mac:
                raise RuntimeError(
                    "CPU execution is disabled on macOS builds; Apple MPS backend is required."
                )
            warnings.warn(
                "CPU execution requested. Performance may be significantly reduced without GPU acceleration."
            )
            return torch.device("cpu")
        raise ValueError(
            f"Unknown prefer_device='{prefer_device}'. Expected 'mps', 'cuda', or 'cpu'."
        )

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if is_mac:
        raise RuntimeError(
            "FastFuncSim requires the Apple Metal Performance Shaders (MPS) backend on macOS, but it was not detected. "
            "Please update to macOS 13+ with Apple Silicon and install a recent PyTorch build with MPS support."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")

    warnings.warn(
        "No GPU backend detected; falling back to CPU execution. Performance will be limited."
    )
    return torch.device("cpu")


def print_device_info(device: torch.device):
    """Print information about the device being used"""
    if device.type == "cuda":
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
        )
    elif device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        print("Using CPU")


def to_tensor(
    x: Union[torch.Tensor, np.ndarray, list, tuple],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert input to torch tensor with specified dtype and device

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input data (numpy array, list, tuple, or torch.Tensor)
    dtype : torch.dtype
        Target dtype
    device : torch.device, optional
        Target device. If None, keep on current device

    Returns
    -------
    tensor : torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    else:
        x = x.to(dtype=dtype)

    if device is not None:
        x = x.to(device=device)

    return x


def calc_memory_usage(shape: tuple, dtype: torch.dtype = torch.float32) -> float:
    """
    Calculate memory usage in GB for a tensor of given shape

    Parameters
    ----------
    shape : tuple
        Tensor shape
    dtype : torch.dtype
        Data type

    Returns
    -------
    memory_gb : float
        Memory usage in gigabytes
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    return (num_elements * bytes_per_element) / 1e9


def optimal_chunk_size(
    n_voxels: int,
    n_timepoints: int,
    n_regressors: int,
    device: torch.device,
    safety_factor: float = 0.5,
) -> int:
    """
    Calculate optimal chunk size for processing voxels given memory constraints

    Parameters
    ----------
    n_voxels : int
        Total number of voxels
    n_timepoints : int
        Number of timepoints
    n_regressors : int
        Number of regressors in design matrix
    device : torch.device
        Computing device
    safety_factor : float
        Fraction of available memory to use (default: 0.5)

    Returns
    -------
    chunk_size : int
        Optimal chunk size
    """
    # Estimate available memory
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        used_mem = torch.cuda.memory_allocated(device) / 1e9
        available_mem = (total_mem - used_mem) * safety_factor
    elif device.type == "mps":
        # MPS doesn't expose memory info, use conservative estimate
        available_mem = 4.0 * safety_factor  # Assume 4GB available
    else:
        available_mem = 8.0 * safety_factor  # Conservative CPU estimate

    # Memory per voxel in GLM fitting (float32 = 4 bytes):
    # - data: n_timepoints
    # - betas: n_regressors
    # - residuals: n_timepoints
    # - predicted/temps: n_timepoints (design @ betas.T creates big intermediate)
    # - ss_total temp: n_timepoints (data - data_mean)
    # Conservative: 5x timepoints + regressors for all intermediates
    bytes_per_voxel = (5 * n_timepoints + n_regressors) * 4  # 4 bytes per float32
    mem_per_voxel_gb = bytes_per_voxel / 1e9

    # Calculate chunk size
    chunk_size = int(available_mem / mem_per_voxel_gb)

    # Set sensible bounds: at least 1000, at most all voxels
    # For large datasets, we want to process tens of thousands at once
    min_chunk = min(1000, n_voxels)
    chunk_size = max(min_chunk, min(chunk_size, n_voxels))

    return chunk_size


def scale_to_percent_signal(
    data: torch.Tensor,
    run_starts: list[int],
    max_scale: float = 200.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Scale voxel timeseries to mean=100 per run (percent signal change units).

    This is equivalent to AFNI's scaling: min(max_scale, a/b*100) where
    a is the timeseries and b is the mean of that timeseries.

    The max_scale (default 200) prevents extreme values - a voxel should
    never more than double its mean signal in physiologically plausible data.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data (n_voxels, n_timepoints) - will be modified in-place
    run_starts : list of int
        Starting timepoint for each run
    max_scale : float, default=200.0
        Maximum allowed scaled value (clips to this)
    verbose : bool, default=True
        Print scaling statistics

    Returns
    -------
    data_scaled : torch.Tensor
        Scaled data (n_voxels, n_timepoints) with mean~100 per run
    violations_mask : torch.Tensor
        Boolean mask (n_voxels, n_timepoints) where values hit max_scale ceiling
    scale_info : dict
        Statistics about the scaling:
        - 'n_violations': total number of timepoints that hit ceiling
        - 'n_voxels_with_violations': number of voxels with any violations
        - 'violation_voxel_indices': 1D tensor of voxel indices with violations
        - 'mean_per_run': (n_voxels, n_runs) mean before scaling
        - 'scale_factors': (n_voxels, n_runs) scale factors used (100/mean)

    Notes
    -----
    The scaling is: scaled = min(max_scale, raw / mean * 100)

    This converts raw signal to percent signal change units where:
    - Mean = 100 (by construction)
    - A value of 101 = 1% signal increase
    - A value of 99 = 1% signal decrease

    Violations (hitting max_scale) indicate potentially problematic voxels
    that may have:
    - Very low mean signal (near noise floor)
    - Motion spikes or other artifacts
    - Edge voxels with partial volume effects
    """
    n_voxels, n_timepoints_total = data.shape
    n_runs = len(run_starts)
    device = data.device

    # Compute run boundaries
    run_ends = run_starts[1:] + [n_timepoints_total]
    run_lengths = [end - start for start, end in zip(run_starts, run_ends)]

    # Storage for per-run statistics
    mean_per_run = torch.zeros(n_voxels, n_runs, device=device)
    scale_factors = torch.zeros(n_voxels, n_runs, device=device)

    # Track violations
    violations_mask = torch.zeros(
        n_voxels, n_timepoints_total, dtype=torch.bool, device=device
    )

    if verbose:
        print("Scaling to percent signal change (mean=100 per run)...")

    for run_idx in range(n_runs):
        start = run_starts[run_idx]
        end = run_ends[run_idx]

        # Get this run's data
        run_data = data[:, start:end]  # (n_voxels, run_length)

        # Compute mean per voxel for this run
        run_mean = run_data.mean(dim=1, keepdim=True)  # (n_voxels, 1)
        mean_per_run[:, run_idx] = run_mean.squeeze()

        # Avoid division by zero (set scale factor to 0 for zero-mean voxels)
        # These voxels will become all zeros after scaling
        safe_mean = run_mean.clone()
        zero_mask = run_mean.abs() < 1e-10
        safe_mean[zero_mask] = 1.0  # Prevent div by zero

        # Compute scale factor: 100 / mean
        scale_factor = 100.0 / safe_mean  # (n_voxels, 1)
        scale_factors[:, run_idx] = scale_factor.squeeze()

        # Scale: a / b * 100 = a * scale_factor
        scaled_run = run_data * scale_factor  # (n_voxels, run_length)

        # Apply ceiling and track violations
        # Values above max_scale (e.g., 200) indicate >100% signal increase
        run_violations = scaled_run > max_scale
        violations_mask[:, start:end] = run_violations

        # Clip to max_scale (only upper bound - lower values are fine)
        # AFNI uses min(max_scale, scaled_value) - we preserve negative values
        # since fMRI can have signal decreases
        scaled_run = torch.clamp(scaled_run, max=max_scale)

        # Handle zero-mean voxels (set to 100 to avoid weird values)
        # Actually, set them to 0 since they're essentially dead voxels
        zero_voxels = zero_mask.squeeze()
        if zero_voxels.any():
            scaled_run[zero_voxels, :] = 0.0

        # Store back
        data[:, start:end] = scaled_run

    # Compute violation statistics
    n_violations = violations_mask.sum().item()
    voxels_with_violations = violations_mask.any(dim=1)
    n_voxels_with_violations = voxels_with_violations.sum().item()
    violation_voxel_indices = torch.where(voxels_with_violations)[0]

    scale_info = {
        "n_violations": int(n_violations),
        "n_voxels_with_violations": int(n_voxels_with_violations),
        "violation_voxel_indices": violation_voxel_indices,
        "mean_per_run": mean_per_run,
        "scale_factors": scale_factors,
    }

    if verbose:
        print(f"  Scaled {n_voxels:,} voxels × {n_runs} runs")
        if n_violations > 0:
            pct_violations = 100 * n_violations / (n_voxels * n_timepoints_total)
            print(
                f"  ⚠️  Ceiling violations (>{max_scale}): {n_violations:,} timepoints ({pct_violations:.4f}%)"
            )
            print(
                f"      Affecting {n_voxels_with_violations:,} voxels ({100 * n_voxels_with_violations / n_voxels:.2f}%)"
            )
        else:
            print(f"  ✓ No ceiling violations (all values ≤ {max_scale})")

    return data, violations_mask, scale_info


def gaussian_blur_3d(
    data: np.ndarray,
    fwhm_mm: float,
    voxel_sizes: tuple[float, float, float],
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply 3D Gaussian spatial smoothing to 4D fMRI data.

    Uses separable 1D convolutions along each spatial axis for efficiency.
    Can process on GPU for speed, chunking by timepoint if needed.

    Parameters
    ----------
    data : np.ndarray
        4D fMRI data (x, y, z, t) - will NOT be modified in place
    fwhm_mm : float
        Full-width at half-maximum of Gaussian kernel in millimeters
    voxel_sizes : tuple of float
        Voxel dimensions in mm (voxel_x, voxel_y, voxel_z)
    device : torch.device, optional
        Device for computation. If None, auto-detect GPU/CPU.
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    data_blurred : np.ndarray
        Blurred 4D data (x, y, z, t), same shape as input

    Notes
    -----
    FWHM to sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355

    The kernel is computed in voxel units using the voxel sizes.
    For anisotropic voxels, the sigma differs in each dimension.

    Memory: For large datasets, processes one timepoint at a time to limit
    GPU memory usage. A single 3D volume is typically manageable.
    """
    import torch.nn.functional as F

    if device is None:
        device = get_device()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data (x, y, z, t), got shape {data.shape}")

    nx, ny, nz, nt = data.shape

    # Convert FWHM to sigma (FWHM = 2.355 * sigma)
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247
    sigma_mm = fwhm_mm * fwhm_to_sigma

    # Convert sigma from mm to voxels for each dimension
    sigma_vox = [sigma_mm / vs for vs in voxel_sizes]

    if verbose:
        print(f"Gaussian blur: FWHM = {fwhm_mm:.1f} mm")
        print(f"  Voxel sizes: {voxel_sizes[0]:.2f} × {voxel_sizes[1]:.2f} × {voxel_sizes[2]:.2f} mm")
        print(f"  Sigma (voxels): {sigma_vox[0]:.2f} × {sigma_vox[1]:.2f} × {sigma_vox[2]:.2f}")

    # Create 1D Gaussian kernels for each dimension
    # Kernel size should be large enough to capture the Gaussian (typically 3-4 sigma each side)
    kernels = []
    for dim, sigma in enumerate(sigma_vox):
        if sigma < 0.1:
            # Very small sigma - skip this dimension (identity)
            kernels.append(None)
            continue

        # Kernel radius: 4 sigma, but at least 1 voxel
        radius = max(1, int(np.ceil(4 * sigma)))
        kernel_size = 2 * radius + 1

        # Create 1D Gaussian kernel
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize

        kernels.append(kernel)

    if verbose:
        kernel_sizes = [len(k) if k is not None else 1 for k in kernels]
        print(f"  Kernel sizes: {kernel_sizes[0]} × {kernel_sizes[1]} × {kernel_sizes[2]} voxels")

    # Estimate memory for one volume
    vol_size_gb = (nx * ny * nz * 4) / (1024**3)  # float32

    # Decide whether to use GPU based on volume size
    # Most GPUs can handle a few GB per volume easily
    use_gpu = device.type in ("cuda", "mps") and vol_size_gb < 2.0

    if verbose and not use_gpu and device.type != "cpu":
        print(f"  Note: Processing on CPU (volume size {vol_size_gb:.2f} GB)")

    compute_device = device if use_gpu else torch.device("cpu")

    # Allocate output
    data_blurred = np.zeros_like(data)

    # Process each timepoint
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(nt), desc="  Blurring", unit="vol")
    else:
        iterator = range(nt)

    for t in iterator:
        # Get single volume and convert to tensor
        vol = torch.from_numpy(data[:, :, :, t]).float().to(compute_device)

        # Apply separable 1D convolutions
        # For F.conv1d, we need: (batch, channels, length)
        # We'll process each dimension by reshaping appropriately

        # X dimension: reshape to (ny*nz, 1, nx), convolve, reshape back
        if kernels[0] is not None:
            k = kernels[0].to(compute_device)
            pad = len(k) // 2
            # Reshape: (nx, ny, nz) -> (ny*nz, 1, nx)
            vol_x = vol.permute(1, 2, 0).reshape(-1, 1, nx)
            vol_x = F.conv1d(vol_x, k.view(1, 1, -1), padding=pad)
            vol = vol_x.reshape(ny, nz, nx).permute(2, 0, 1)

        # Y dimension: reshape to (nx*nz, 1, ny), convolve, reshape back
        if kernels[1] is not None:
            k = kernels[1].to(compute_device)
            pad = len(k) // 2
            # Reshape: (nx, ny, nz) -> (nx*nz, 1, ny)
            vol_y = vol.permute(0, 2, 1).reshape(-1, 1, ny)
            vol_y = F.conv1d(vol_y, k.view(1, 1, -1), padding=pad)
            vol = vol_y.reshape(nx, nz, ny).permute(0, 2, 1)

        # Z dimension: reshape to (nx*ny, 1, nz), convolve, reshape back
        if kernels[2] is not None:
            k = kernels[2].to(compute_device)
            pad = len(k) // 2
            # Reshape: (nx, ny, nz) -> (nx*ny, 1, nz)
            vol_z = vol.reshape(nx * ny, 1, nz)
            vol_z = F.conv1d(vol_z, k.view(1, 1, -1), padding=pad)
            vol = vol_z.reshape(nx, ny, nz)

        # Store result
        data_blurred[:, :, :, t] = vol.cpu().numpy()

    if verbose:
        print(f"  ✓ Blurred {nt} volumes")

    return data_blurred
