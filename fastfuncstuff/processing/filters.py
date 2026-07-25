"""GPU-accelerated Savitzky-Golay filtering for 1-D time series.

Implements the Savitzky-Golay (SGF) filter as batched 1-D convolution on
PyTorch tensors, enabling voxel-parallel filtering on GPU.  Used by
phase regression (ffs_phasereg) to smooth noisy phase time series
(Barry & Gore, Hum Brain Mapp 2014) and generally applicable to any
per-voxel 1-D signal processing.

The filter fits a polynomial of order *p* to a sliding window of *N*
samples and replaces the centre sample with the polynomial value.
This is equivalent to convolution with precomputed coefficients derived
from the pseudoinverse of a Vandermonde matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _sgf_coefficients(
    window_length: int,
    poly_order: int,
    deriv: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute Savitzky-Golay convolution kernel.

    Parameters
    ----------
    window_length : int
        Must be odd and > poly_order.
    poly_order : int
        Polynomial order for local fitting.
    deriv : int
        Derivative order (0 = smoothing).
    device, dtype : torch device and dtype for the kernel.

    Returns
    -------
    kernel : Tensor, shape (window_length,)
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if poly_order >= window_length:
        raise ValueError("poly_order must be < window_length")

    half = window_length // 2
    order_range = torch.arange(-half, half + 1, device=device, dtype=dtype)
    n_terms = poly_order + 1

    vander = order_range.unsqueeze(1).pow(
        torch.arange(n_terms, device=device, dtype=dtype).unsqueeze(0)
    )

    try:
        coeffs = torch.linalg.pinv(vander, atol=1e-7)[deriv]
    except torch._C._LinAlgError:
        coeffs = torch.zeros(window_length, device=device, dtype=dtype)
        coeffs[half] = 1.0
    return coeffs


def savgol_filter_1d(
    data: Tensor,
    window_length: int,
    poly_order: int,
) -> Tensor:
    """Apply Savitzky-Golay filter to batched 1-D time series on GPU.

    Parameters
    ----------
    data : Tensor, shape (..., n_timepoints)
        Input time series.  Filtering is applied along the last dimension.
        Any leading dimensions are treated as batch (e.g. n_voxels).
    window_length : int
        Odd integer, width of the filtering window.
    poly_order : int
        Polynomial order for the local fit (< window_length).

    Returns
    -------
    filtered : Tensor, same shape as *data*
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if poly_order >= window_length:
        raise ValueError("poly_order must be < window_length")
    if data.shape[-1] < window_length:
        return data.clone()

    kernel = _sgf_coefficients(
        window_length,
        poly_order,
        device=data.device,
        dtype=data.dtype,
    )

    original_shape = data.shape
    if data.dim() == 1:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        data = data.unsqueeze(1)
    else:
        data = data.reshape(-1, 1, data.shape[-1])

    pad = window_length // 2
    padded = torch.nn.functional.pad(data, (pad, pad), mode="reflect")

    kernel_2d = kernel.view(1, 1, -1)
    filtered = torch.nn.functional.conv1d(padded, kernel_2d)

    return filtered.reshape(original_shape)


def savgol_filter_explore(
    data: Tensor,
    n_timepoints: int,
    device: torch.device,
    min_window: int = 5,
    max_window: int | None = None,
    min_order: int = 2,
    max_order: int | None = None,
    step: int = 4,
    metric_fn=None,
    return_params: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Data-driven SGF parameter search per voxel (Barry & Gore 2014).

    For each voxel, tries multiple (window, order) combinations and
    selects the one that optimises *metric_fn*.  If metric_fn is not
    provided, returns the unfiltered data (pass-through).

    The **unfiltered** series competes as a candidate alongside every (N, p)
    pair.  This is step 3 of Barry & Gore's algorithm: "if R²_no_filtering >
    R²_SGF then the resultant time series after standard PR is retained".
    A few voxels have phase SNR high enough that filtering only costs them
    signal, and the search must be able to say so.  (Earlier versions seeded
    the running best at -inf, so any filter — however harmful — always won.)

    Parameters
    ----------
    data : Tensor, shape (n_voxels, n_timepoints)
        Input time series.
    n_timepoints : int
        Number of timepoints (must match data.shape[-1]).
    device : torch.device
    min_window : int
        Minimum SGF window (odd).
    max_window : int or None
        Maximum SGF window.  None = n_timepoints // 2 (Barry & Gore).
    min_order : int
        Minimum polynomial order.
    max_order : int or None
        Maximum polynomial order.  None = max_window // 4 (Barry & Gore).
    step : int
        Step between window sizes (kept odd).
    metric_fn : callable or None
        Function(filtered_data) -> Tensor (n_voxels,) to maximise.
        If None, returns unfiltered data.
    return_params : bool
        If True, also return the per-voxel chosen (window, order) as two
        int tensors of shape (n_voxels,).  Voxels never filtered (empty grid)
        report window=0, order=0.

    Returns
    -------
    best_filtered : Tensor, shape (n_voxels, n_timepoints)
        Filtered data using per-voxel optimal parameters.
    best_window, best_order : Tensor, shape (n_voxels,)
        Only returned when return_params=True.  The window length and
        polynomial order selected per voxel.
    """
    if metric_fn is None:
        if return_params:
            zeros = torch.zeros(data.shape[0], dtype=torch.int64, device=device)
            return data, zeros, zeros.clone()
        return data

    # Barry & Gore search N up to roughly half the run length (N <= 49 for 96
    # timepoints, N <= 97 for 192) and p up to N/4-ish (12 and 24 for those two
    # cases). The old defaults — a hard 97-TR cap and max_order=5 — were not
    # from the paper: they let the window run to the full run length on short
    # runs while capping order so low that only the heavily-smoothing corner of
    # the grid was reachable.
    if max_window is None:
        max_window = max(min_window, n_timepoints // 2)
    if max_order is None:
        max_order = max(min_order, max_window // 4)

    max_window = min(max_window, n_timepoints)
    if max_window % 2 == 0:
        max_window -= 1

    windows = list(range(min_window, max_window + 1, step))
    windows = [w | 1 for w in windows]
    windows = sorted(set(w for w in windows if w < n_timepoints))

    # Seed with the unfiltered series scored on its own terms (window=order=0
    # marks "no filtering won" in the returned parameter maps).
    best_score = metric_fn(data)
    best_filtered = data.clone()
    best_window = torch.zeros(data.shape[0], dtype=torch.int64, device=device)
    best_order = torch.zeros(data.shape[0], dtype=torch.int64, device=device)

    for w in windows:
        for p in range(min_order, min(max_order, w - 1) + 1):
            try:
                filt = savgol_filter_1d(data, w, p)
            except torch._C._LinAlgError:
                continue
            score = metric_fn(filt)
            improved = score > best_score
            best_score = torch.where(improved, score, best_score)
            best_filtered = torch.where(improved.unsqueeze(-1), filt, best_filtered)
            best_window = torch.where(improved, w, best_window)
            best_order = torch.where(improved, p, best_order)

    if return_params:
        return best_filtered, best_window, best_order
    return best_filtered
