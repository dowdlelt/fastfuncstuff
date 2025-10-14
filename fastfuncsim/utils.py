"""
Utility functions for fastfuncsim
Device management and helper functions
"""

import platform
import warnings
from typing import Optional, Union

import torch


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
    x: Union[torch.Tensor, list, tuple],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert input to torch tensor with specified dtype and device

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input data
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

    # Memory per voxel (data + betas + residuals + working space)
    mem_per_voxel = calc_memory_usage((n_timepoints + n_regressors * 2,))

    # Calculate chunk size
    chunk_size = int(available_mem / mem_per_voxel)

    # Set sensible bounds: at least 1000, at most all voxels
    # For large datasets, we want to process tens of thousands at once
    min_chunk = min(1000, n_voxels)
    chunk_size = max(min_chunk, min(chunk_size, n_voxels))

    return chunk_size
