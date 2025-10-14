#!/usr/bin/env python
"""
Comprehensive ISI Simulation Study

Systematic parameter sweep over:
- ISI means: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5] seconds
- ISI distribution: Poisson (with bisection method to hit exact means)
- Design: Alternating A/B conditions
- HRF libraries: CNVLab/NSD (20 HRFs) and existing canonical
- Activation patterns: Multiple magnitude combinations
- Noise levels: Range of SNR values

Uses OLS fitting (not ARMA) for computational speed.
Generates comprehensive figures showing design efficiency and HRF recovery.

Inspired by simulate_movietasks.m
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.stats import pearsonr

import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise

# =============================================================================
# HRF Library Functions
# =============================================================================


def load_cnvlab_hrf_library(
    duration: float = 5.0, tr: float = 1.0, library_path: Optional[Path] = None
) -> np.ndarray:
    """
    Load and prepare CNVLab/NSD canonical HRF library.

    Based on GLMsingle's getcanonicalhrflibrary() function.

    Parameters
    ----------
    duration : float
        Stimulus duration in seconds (for convolution with boxcar)
    tr : float
        Repetition time in seconds
    library_path : Path, optional
        Path to getcanonicalhrflibrary.tsv file

    Returns
    -------
    hrfs : ndarray
        Array of HRFs, shape (n_hrfs, n_timepoints)
        Each HRF is normalized to max of 1.0
    """
    if library_path is None:
        # Default: look in fastfuncsim package directory
        import fastfuncsim

        pkg_dir = Path(fastfuncsim.__file__).parent
        library_path = pkg_dir / "getcanonicalhrflibrary.tsv"

    if not library_path.exists():
        raise FileNotFoundError(
            f"CNVLab HRF library not found at {library_path}\n"
            "Please download from: "
            "https://raw.githubusercontent.com/cvnlab/GLMsingle/refs/heads/main/"
            "glmsingle/hrf/getcanonicalhrflibrary.tsv"
        )

    # Load HRF library (501 timepoints × 20 HRFs at 0.1s resolution)
    hrfs = np.genfromtxt(library_path).T  # (20, 501)

    # High-resolution sampling rate
    tr_old = 0.1  # 100ms

    if duration == 0:
        duration = 0.1

    # Convolve each HRF with boxcar of specified duration
    boxcar_length = max(1, round(duration / tr_old))
    boxcar = np.ones(boxcar_length)

    hrfs_convolved = []
    for hrf in hrfs:
        hrf_conv = np.convolve(hrf, boxcar, mode="full")
        hrfs_convolved.append(hrf_conv)

    # Downsample to target TR using pchip interpolation
    n_samples_highres = hrfs_convolved[0].shape[0]
    time_highres = np.arange(n_samples_highres) * tr_old

    # New sampling times
    time_new = np.arange(0, time_highres[-1], tr)

    hrfs_downsampled = []
    for hrf_conv in hrfs_convolved:
        # Interpolate
        interpolator = PchipInterpolator(time_highres, hrf_conv)
        hrf_ds = interpolator(time_new)

        # Normalize to max of 1.0
        hrf_ds = hrf_ds / np.max(hrf_ds)

        hrfs_downsampled.append(hrf_ds)

    return np.array(hrfs_downsampled, dtype=np.float32)


def get_canonical_hrf_library(
    duration: float = 5.0, tr: float = 1.0, n_hrfs: int = 20
) -> np.ndarray:
    """
    Get canonical HRF library from fastfuncsim.

    Parameters
    ----------
    duration : float
        Stimulus duration in seconds
    tr : float
        Repetition time in seconds
    n_hrfs : int
        Number of HRFs to generate

    Returns
    -------
    hrfs : ndarray
        Array of HRFs, shape (n_hrfs, n_timepoints)
    """
    # Use fastfuncsim's existing function if available
    # Otherwise create simple SPM-style HRF variations

    from scipy.stats import gamma

    # SPM canonical HRF parameters
    # Peak at ~5-6s, undershoot at ~15-16s
    peak_delays = np.linspace(4, 7, n_hrfs)  # Peak delay variations

    # Time vector at high resolution
    dt = 0.1
    t_highres = np.arange(0, 30, dt)

    hrfs_highres = []
    for peak_delay in peak_delays:
        # Double gamma HRF
        # Positive gamma (peak)
        peak_shape = 6.0
        peak_scale = peak_delay / peak_shape
        peak = gamma.pdf(t_highres, peak_shape, scale=peak_scale)

        # Negative gamma (undershoot)
        undershoot_shape = 12.0
        undershoot_delay = peak_delay + 10.0
        undershoot_scale = undershoot_delay / undershoot_shape
        undershoot = gamma.pdf(t_highres, undershoot_shape, scale=undershoot_scale)

        # Combine (peak - 0.35*undershoot is typical)
        hrf = peak - 0.35 * undershoot

        # Convolve with boxcar for stimulus duration
        boxcar_length = max(1, round(duration / dt))
        boxcar = np.ones(boxcar_length)
        hrf_conv = np.convolve(hrf, boxcar, mode="full")

        hrfs_highres.append(hrf_conv)

    # Downsample to target TR
    hrfs_downsampled = []
    for hrf_highres in hrfs_highres:
        t_hr = np.arange(len(hrf_highres)) * dt
        t_new = np.arange(0, t_hr[-1], tr)

        interpolator = PchipInterpolator(t_hr, hrf_highres)
        hrf_ds = interpolator(t_new)

        # Normalize
        hrf_ds = hrf_ds / np.max(hrf_ds)

        hrfs_downsampled.append(hrf_ds)

    return np.array(hrfs_downsampled, dtype=np.float32)


# =============================================================================
# Poisson ISI Generation (from simulate_movietasks.m)
# =============================================================================


def generate_poisson_isis(
    target_mean: float,
    n_isis: int,
    lower_limit: float = 2.0,
    upper_limit: float = 8.0,
    max_iter: int = 1000,
    tolerance: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate ISIs from truncated Poisson distribution with exact target mean.

    Uses bisection method to find optimal lambda parameter that produces
    the desired mean after truncation.

    Parameters
    ----------
    target_mean : float
        Desired mean ISI in seconds
    n_isis : int
        Number of ISIs to generate
    lower_limit : float
        Minimum ISI (seconds)
    upper_limit : float
        Maximum ISI (seconds)
    max_iter : int
        Maximum iterations for bisection search
    tolerance : float
        Convergence tolerance for mean
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    isis : ndarray
        Array of ISIs in seconds, shape (n_isis,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Bisection search for optimal lambda
    lambda_low = 1.0
    lambda_high = 10.0
    lambda_opt = (lambda_low + lambda_high) / 2.0

    converged = False

    for iter_num in range(max_iter):
        # Generate ISIs from Poisson with truncation
        isis = []
        while len(isis) < n_isis:
            # Generate batch
            batch = np.random.poisson(lambda_opt, n_isis * 5)

            # Apply truncation
            truncated = batch[(batch >= lower_limit) & (batch <= upper_limit)]

            isis.extend(truncated)

        # Trim to exact size
        isis = np.array(isis[:n_isis], dtype=np.float32)

        # Check mean
        current_mean = isis.mean()

        # Check convergence
        if abs(current_mean - target_mean) < tolerance:
            converged = True
            print(
                f"  Converged! Optimal lambda: {lambda_opt:.3f}, Mean: {current_mean:.3f}s"
            )
            break

        # Adjust lambda
        if current_mean < target_mean:
            lambda_low = lambda_opt
        else:
            lambda_high = lambda_opt

        lambda_opt = (lambda_low + lambda_high) / 2.0

    if not converged:
        print(f"  Warning: Max iterations reached. Final mean: {isis.mean():.3f}s")

    return isis


# =============================================================================
# Design Construction
# =============================================================================


def build_alternating_design(
    isis_tr: np.ndarray,
    stim_dur_tr: int,
    n_conds: int,
    total_tr: int,
    padding_tr: int = 10,
) -> np.ndarray:
    """
    Build alternating A/B design matrix from ISIs.

    Parameters
    ----------
    isis_tr : ndarray
        Inter-stimulus intervals in TRs
    stim_dur_tr : int
        Stimulus duration in TRs
    n_conds : int
        Number of conditions (typically 2 for A/B)
    total_tr : int
        Total scan duration in TRs
    padding_tr : int
        Padding at start and end in TRs

    Returns
    -------
    design : ndarray
        Design matrix, shape (total_tr, n_conds)
    """
    # Shuffle ISIs
    shuffled_isis = isis_tr[np.random.permutation(len(isis_tr))]

    # Calculate onset times (cumulative sum with stimulus duration)
    onsets_from_stim = np.concatenate(
        [[shuffled_isis[0]], shuffled_isis[1:] + stim_dur_tr]
    )
    actual_onsets = np.cumsum(onsets_from_stim) + padding_tr
    actual_onsets = actual_onsets.astype(int)

    # Build design matrix
    design = np.zeros((total_tr, n_conds), dtype=np.float32)

    # Alternating assignment: 0, 1, 0, 1, ...
    for cond_idx in range(n_conds):
        onset_indices = actual_onsets[cond_idx::n_conds]
        # Only use onsets that fit in the scan
        onset_indices = onset_indices[onset_indices < total_tr - padding_tr]
        design[onset_indices, cond_idx] = 1.0

    return design


def convolve_design_with_hrf(design: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """
    Convolve design matrix with HRF.

    Parameters
    ----------
    design : ndarray
        Design matrix, shape (n_timepoints, n_conds)
    hrf : ndarray
        HRF, shape (hrf_length,)

    Returns
    -------
    design_convolved : ndarray
        Convolved design matrix, shape (n_timepoints, n_conds)
    """
    n_timepoints, n_conds = design.shape
    design_convolved = np.zeros_like(design)

    for cond_idx in range(n_conds):
        conv_full = np.convolve(design[:, cond_idx], hrf, mode="full")
        design_convolved[:, cond_idx] = conv_full[:n_timepoints]

    return design_convolved


# =============================================================================
# Design Efficiency Metrics
# =============================================================================


def compute_design_efficiency(X: np.ndarray) -> Dict[str, float]:
    """
    Compute design efficiency metrics.

    Parameters
    ----------
    X : ndarray
        Design matrix, shape (n_timepoints, n_regressors)

    Returns
    -------
    metrics : dict
        Dictionary with efficiency metrics:
        - condition_number: Condition number of X'X
        - efficiency: Average diagonal of (X'X)^-1
        - vif_mean: Mean variance inflation factor
        - correlation_max: Maximum absolute correlation between columns
    """
    XTX = X.T @ X

    # Condition number
    cond_num = np.linalg.cond(XTX)

    # Efficiency (average diagonal of (X'X)^-1)
    try:
        XTX_inv = np.linalg.inv(XTX)
        efficiency = 1.0 / np.mean(np.diag(XTX_inv))
        vif_mean = np.mean(np.diag(XTX_inv))
    except np.linalg.LinAlgError:
        efficiency = 0.0
        vif_mean = np.inf

    # Maximum correlation between columns
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        # Get upper triangle without diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_max = np.max(np.abs(corr_matrix[mask]))
    else:
        corr_max = 0.0

    return {
        "condition_number": cond_num,
        "efficiency": efficiency,
        "vif_mean": vif_mean,
        "correlation_max": corr_max,
    }


# =============================================================================
# Polynomial Nuisance Regressors
# =============================================================================


def create_polynomial_regressors(n_timepoints: int, max_order: int = 3) -> np.ndarray:
    """
    Create orthogonalized polynomial regressors for drift modeling.

    Uses Gram-Schmidt orthogonalization to create orthogonal polynomials
    (similar to Legendre polynomials but simpler).

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    max_order : int
        Maximum polynomial order (default: 3)
        Order 0 = constant (intercept)
        Order 1 = linear trend
        Order 2 = quadratic drift
        Order 3 = cubic drift

    Returns
    -------
    poly_regressors : ndarray
        Polynomial regressors, shape (n_timepoints, max_order+1)
        Columns are orthonormalized
    """
    # Normalized time vector [-1, 1]
    t = np.linspace(-1, 1, n_timepoints)

    # Initialize polynomial matrix
    poly = np.zeros((n_timepoints, max_order + 1), dtype=np.float32)

    # Generate raw polynomials
    for order in range(max_order + 1):
        poly[:, order] = t**order

    # Gram-Schmidt orthogonalization
    poly_orth = np.zeros_like(poly)

    for i in range(max_order + 1):
        # Start with current polynomial
        v = poly[:, i].copy()

        # Subtract projections onto previous orthogonal polynomials
        for j in range(i):
            projection = np.dot(v, poly_orth[:, j]) * poly_orth[:, j]
            v = v - projection

        # Normalize
        v = v / (np.linalg.norm(v) + 1e-10)

        poly_orth[:, i] = v

    return poly_orth


# =============================================================================
# OLS Fitting
# =============================================================================


def fit_ols_torch(
    Y: torch.Tensor, X: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit OLS model: Y = X * beta + epsilon

    Parameters
    ----------
    Y : torch.Tensor
        Data, shape (n_timepoints, n_voxels)
    X : torch.Tensor
        Design matrix, shape (n_timepoints, n_regressors)
    device : torch.device
        Device to use

    Returns
    -------
    beta : torch.Tensor
        Parameter estimates, shape (n_regressors, n_voxels)
    predicted : torch.Tensor
        Predicted values, shape (n_timepoints, n_voxels)
    r2 : torch.Tensor
        R-squared values, shape (n_voxels,)
    """
    Y = Y.to(device)
    X = X.to(device)

    # Solve: beta = (X'X)^-1 X'Y
    XTX = X.T @ X
    XTY = X.T @ Y

    # Add small ridge for numerical stability
    ridge = 1e-6
    XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)

    beta = torch.linalg.solve(XTX_reg, XTY)

    # Predicted values
    predicted = X @ beta

    # R-squared
    residuals = Y - predicted
    ss_res = torch.sum(residuals**2, dim=0)
    ss_tot = torch.sum((Y - Y.mean(dim=0, keepdim=True)) ** 2, dim=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return beta, predicted, r2


# =============================================================================
# Spatial Grid Simulation (like simulate_movietasks.m)
# =============================================================================


def create_spatial_grid_simulation(
    hrfs: np.ndarray,
    design: np.ndarray,
    activation_patterns: np.ndarray,
    noise_levels: np.ndarray,
    tr: float,
    grid_size: Tuple[int, int] = (100, 100),
    voxels_per_hrf: int = 5,
    voxels_per_activation: int = 5,
    baseline: float = 100.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Create spatial grid simulation like simulate_movietasks.m

    Spatial organization:
    - X-axis: Different HRFs (voxels_per_hrf voxels per HRF)
    - Y-axis: Different activation patterns (voxels_per_activation per pattern)
    - Z-axis: Different noise levels (one slice per noise level)

    Parameters
    ----------
    hrfs : ndarray
        HRF library, shape (n_hrfs, hrf_length)
    design : ndarray
        Design matrix (unconvolved), shape (n_timepoints, n_conds)
    activation_patterns : ndarray
        Activation magnitudes for each condition, shape (n_patterns, n_conds)
    noise_levels : ndarray
        Noise standard deviations, shape (n_noise_levels,)
    tr : float
        Repetition time
    grid_size : tuple
        (n_x, n_y) spatial dimensions
    voxels_per_hrf : int
        Number of voxels per HRF along X-axis
    voxels_per_activation : int
        Number of voxels per activation pattern along Y-axis
    baseline : float
        Baseline signal level
    seed : int, optional
        Random seed

    Returns
    -------
    data : ndarray
        Simulated data, shape (n_x, n_y, n_z, n_timepoints)
    ground_truth : dict
        Dictionary with ground truth information:
        - hrf_map: (n_x, n_y) - which HRF index for each voxel
        - activation_map: (n_x, n_y) - which activation pattern for each voxel
        - noise_map: (n_x, n_y, n_z) - noise level for each voxel
    """
    if seed is not None:
        np.random.seed(seed)

    n_x, n_y = grid_size
    n_timepoints = design.shape[0]
    n_conds = design.shape[1]
    n_hrfs = hrfs.shape[0]
    n_patterns = activation_patterns.shape[0]
    n_noise_levels = len(noise_levels)

    # Calculate how many HRFs and patterns fit in grid
    n_hrf_groups = n_x // voxels_per_hrf
    n_pattern_groups = n_y // voxels_per_activation

    # Ensure we have enough HRFs and patterns
    if n_hrf_groups > n_hrfs:
        print(f"Warning: Not enough HRFs ({n_hrfs}) for grid. Will cycle.")
    if n_pattern_groups > n_patterns:
        print(f"Warning: Not enough patterns ({n_patterns}) for grid. Will cycle.")

    # Create data volume
    data = np.zeros((n_x, n_y, n_noise_levels, n_timepoints), dtype=np.float32)

    # Ground truth maps
    hrf_map = np.zeros((n_x, n_y), dtype=int)
    activation_map = np.zeros((n_x, n_y), dtype=int)
    noise_map = np.zeros((n_x, n_y, n_noise_levels), dtype=np.float32)

    print("Creating spatial grid simulation...")

    # Iterate over slices (noise levels)
    for slice_idx, noise_std in enumerate(noise_levels):
        print(f"  Slice {slice_idx + 1}/{n_noise_levels} (noise σ={noise_std:.2f})...")

        # Iterate over X dimension (HRFs)
        for x_idx in range(n_x):
            # Determine which HRF to use (cycle if needed)
            hrf_group = x_idx // voxels_per_hrf
            hrf_idx = hrf_group % n_hrfs
            hrf = hrfs[hrf_idx]

            # Iterate over Y dimension (activation patterns)
            for y_idx in range(n_y):
                # Determine which activation pattern to use
                pattern_group = y_idx // voxels_per_activation
                pattern_idx = pattern_group % n_patterns
                activations = activation_patterns[pattern_idx]

                # Store ground truth (only once!)
                hrf_map[x_idx, y_idx] = hrf_idx
                activation_map[x_idx, y_idx] = pattern_idx
                noise_map[x_idx, y_idx, slice_idx] = noise_std

                # Create onset vector with activations
                onset_vector = np.zeros(n_timepoints)
                for cond_idx in range(n_conds):
                    onset_times = np.where(design[:, cond_idx] > 0)[0]
                    onset_vector[onset_times] = activations[cond_idx]

                # Convolve with HRF
                signal = np.convolve(onset_vector, hrf, mode="full")[:n_timepoints]

                # Add baseline + signal (noise added per slice below)
                data[x_idx, y_idx, slice_idx, :] = baseline + signal

        # Generate realistic fMRI noise for entire slice
        # This is more efficient and creates realistic spatial/temporal structure
        print(f"    Generating realistic fMRI noise (cardiac, respiratory, 1/f)...")
        noise_slice = generate_fmri_noise(
            tr=tr,
            duration_s=n_timepoints * tr,
            matrix_size=(n_x, n_y),
            normalize=True,
            device=torch.device("cpu"),  # Generate on CPU then convert
        )

        # Convert to numpy and scale by noise_std
        noise_np = noise_slice.numpy() * noise_std  # (n_timepoints, n_x, n_y)

        # Add noise to data (transpose to match data shape)
        data[:, :, slice_idx, :] += noise_np.transpose(
            1, 2, 0
        )  # (n_x, n_y, n_timepoints)

    ground_truth = {
        "hrf_map": hrf_map,
        "activation_map": activation_map,
        "noise_map": noise_map,
        "hrfs": hrfs,
        "activation_patterns": activation_patterns,
        "noise_levels": noise_levels,
    }

    return data, ground_truth


# =============================================================================
# Main Simulation Function
# =============================================================================


def run_isi_simulation(
    isi_mean: float,
    hrf_library_name: str,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    n_runs: int = 4,
    n_conds: int = 2,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run single ISI simulation with specified parameters.

    Parameters
    ----------
    isi_mean : float
        Target mean ISI in seconds
    hrf_library_name : str
        'cnvlab' or 'canonical'
    tr : float
        Repetition time in seconds
    total_duration : float
        Total scan duration in seconds
    stim_duration : float
        Stimulus duration in seconds
    n_runs : int
        Number of runs
    n_conds : int
        Number of conditions
    device : torch.device, optional
        Device for computation
    seed : int, optional
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Simulation results including:
        - design_efficiency: Design efficiency metrics
        - ols_results: OLS fitting results
        - ground_truth: Ground truth information
        - hrf_recovery: HRF recovery metrics
    """
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"ISI Simulation: mean={isi_mean}s, library={hrf_library_name}")
        print(f"{'=' * 70}\n")

    # Calculate durations in TRs
    total_tr = int(total_duration / tr)
    stim_tr = int(stim_duration / tr)
    padding_tr = int(10 / tr)

    # Load HRF library
    if hrf_library_name == "cnvlab":
        hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
    elif hrf_library_name == "canonical":
        hrfs = get_canonical_hrf_library(duration=stim_duration, tr=tr, n_hrfs=20)
    else:
        raise ValueError(f"Unknown HRF library: {hrf_library_name}")

    if verbose:
        print(f"Loaded {hrfs.shape[0]} HRFs, length={hrfs.shape[1]} TRs")

    # Define activation patterns (magnitude combinations for A/B conditions)
    # From simulate_movietasks.m: ab_ixs array
    activation_patterns = np.array(
        [
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
            [4, 5],
            [3, 5],
            [2, 5],
            [1, 5],
            [1, 3],
            [3, 3],
            [3, 1],
            [1, 1],
            [0, 2],
            [2, 0],
            [-1, -1],
            [-3, -3],
            [-3, 4],
            [4, -3],
            [0, 0],
        ],
        dtype=np.float32,
    )

    # Noise levels (progressively higher)
    noise_levels = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # Generate Poisson ISIs
    if verbose:
        print(f"\nGenerating Poisson ISIs (target mean={isi_mean}s)...")

    # Calculate number of trials
    block_average = isi_mean + stim_duration
    n_trials = int((total_duration - 2 * padding_tr * tr) / block_average)
    # Make sure divisible by n_conds
    n_trials = n_trials - (n_trials % n_conds)

    isis_sec = generate_poisson_isis(
        target_mean=isi_mean,
        n_isis=n_trials,
        lower_limit=2.0,
        upper_limit=8.0,
        seed=seed,
    )
    isis_tr = np.round(isis_sec / tr).astype(int)

    if verbose:
        print(f"  Generated {len(isis_sec)} ISIs")
        print(f"  Actual mean: {isis_sec.mean():.3f}s (SD={isis_sec.std():.3f}s)")

    # Build design matrix (unconvolved)
    if verbose:
        print("\nBuilding design matrix...")

    design_unconvolved = build_alternating_design(
        isis_tr=isis_tr,
        stim_dur_tr=stim_tr,
        n_conds=n_conds,
        total_tr=total_tr,
        padding_tr=padding_tr,
    )

    n_trials_a = np.sum(design_unconvolved[:, 0])
    n_trials_b = np.sum(design_unconvolved[:, 1])

    if verbose:
        print(f"  Condition A: {n_trials_a} trials")
        print(f"  Condition B: {n_trials_b} trials")

    # Create spatial grid simulation
    if verbose:
        print("\nCreating spatial grid simulation...")

    data, ground_truth = create_spatial_grid_simulation(
        hrfs=hrfs,
        design=design_unconvolved,
        activation_patterns=activation_patterns,
        noise_levels=noise_levels,
        tr=tr,
        grid_size=(100, 100),
        voxels_per_hrf=5,
        voxels_per_activation=5,
        seed=seed,
    )

    if verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  {data.shape[0] * data.shape[1] * data.shape[2]:,} total voxels")

    # Fit OLS for each HRF in library
    if verbose:
        print("\nFitting OLS models for all HRFs in library...")

    # Reshape data to 2D (voxels × time)
    n_x, n_y, n_z, n_t = data.shape
    n_voxels = n_x * n_y * n_z
    data_2d = data.reshape(n_voxels, n_t).T  # (n_t, n_voxels)

    # For each HRF, convolve design and fit
    ols_results = []
    for hrf_idx, hrf in enumerate(hrfs):
        if verbose and hrf_idx % 5 == 0:
            print(f"  HRF {hrf_idx + 1}/{len(hrfs)}...")

        # Convolve design with this HRF
        X = convolve_design_with_hrf(design_unconvolved, hrf)

        # Add polynomial nuisance regressors (orders 0-3: intercept, linear, quadratic, cubic)
        # This models baseline + drift like real fMRI analysis
        poly_regressors = create_polynomial_regressors(X.shape[0], max_order=3)

        # Full design: [task_regressors, poly_regressors]
        X_full = np.hstack([X, poly_regressors]).astype(np.float32)

        # Compute design efficiency (on task regressors only, not nuisance)
        efficiency = compute_design_efficiency(X)

        # Fit OLS with full design (task + nuisance regressors)
        Y_torch = torch.from_numpy(data_2d).float()
        X_torch = torch.from_numpy(X_full).float()

        beta, predicted, r2 = fit_ols_torch(Y_torch, X_torch, device)

        ols_results.append(
            {
                "hrf_idx": hrf_idx,
                "beta": beta.cpu().numpy(),
                "r2": r2.cpu().numpy(),
                "design_efficiency": efficiency,
            }
        )

    # Find best-fitting HRF for each voxel
    if verbose:
        print("\nFinding best-fitting HRF for each voxel...")

    r2_matrix = np.array([res["r2"] for res in ols_results])  # (n_hrfs, n_voxels)
    best_hrf_idx = np.argmax(r2_matrix, axis=0)  # (n_voxels,)
    best_r2 = np.max(r2_matrix, axis=0)

    # Reshape back to 3D
    best_hrf_map = best_hrf_idx.reshape(n_x, n_y, n_z)
    best_r2_map = best_r2.reshape(n_x, n_y, n_z)

    # Compute HRF recovery accuracy
    hrf_ground_truth_3d = np.repeat(
        ground_truth["hrf_map"][:, :, np.newaxis], n_z, axis=2
    )

    hrf_recovery_accuracy = np.mean(best_hrf_map == hrf_ground_truth_3d)

    if verbose:
        print(f"  HRF recovery accuracy: {hrf_recovery_accuracy * 100:.1f}%")
        print(f"  Mean R²: {best_r2.mean():.3f}")

    # Package results
    results = {
        "isi_mean": isi_mean,
        "hrf_library_name": hrf_library_name,
        "isis": isis_sec,
        "design_unconvolved": design_unconvolved,
        "data": data,
        "ground_truth": ground_truth,
        "ols_results": ols_results,
        "best_hrf_map": best_hrf_map,
        "best_r2_map": best_r2_map,
        "hrf_recovery_accuracy": hrf_recovery_accuracy,
        "r2_matrix": r2_matrix,
    }

    return results


# =============================================================================
# NIfTI Output Functions
# =============================================================================


def save_simulation_nifti(
    data: np.ndarray,
    design_unconvolved: np.ndarray,
    ground_truth: Dict,
    isi_mean: float,
    hrf_library_name: str,
    output_dir: Path,
    tr: float = 1.0,
):
    """
    Save simulation data as NIfTI file with design matrix.

    Parameters
    ----------
    data : ndarray
        Simulated fMRI data, shape (n_x, n_y, n_z, n_timepoints)
    design_unconvolved : ndarray
        Design matrix, shape (n_timepoints, n_conds)
    ground_truth : dict
        Ground truth information
    isi_mean : float
        Mean ISI for this simulation
    hrf_library_name : str
        HRF library name
    output_dir : Path
        Output directory
    tr : float
        Repetition time in seconds
    """
    # Create simulation-specific directory
    sim_name = f"isi{isi_mean:.1f}_{hrf_library_name}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(exist_ok=True)

    # Create simple affine matrix (2mm isotropic voxels)
    affine = np.eye(4)
    affine[0, 0] = 2.0  # x voxel size
    affine[1, 1] = 2.0  # y voxel size
    affine[2, 2] = 2.0  # z voxel size

    # Save fMRI data
    nifti_path = sim_dir / "func_data.nii.gz"
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_xyzt_units(xyz="mm", t="sec")
    img.header["pixdim"][4] = tr
    nib.save(img, nifti_path)

    # Save design matrix
    design_path = sim_dir / "design_matrix.txt"
    np.savetxt(
        design_path,
        design_unconvolved,
        fmt="%.6f",
        header=f"Unconvolved design matrix (n_timepoints={design_unconvolved.shape[0]}, n_conditions={design_unconvolved.shape[1]})",
    )

    # Save ground truth HRF map
    hrf_map_3d = np.repeat(
        ground_truth["hrf_map"][:, :, np.newaxis], data.shape[2], axis=2
    )
    hrf_map_path = sim_dir / "ground_truth_hrf_map.nii.gz"
    hrf_img = nib.Nifti1Image(hrf_map_3d.astype(np.int16), affine)
    nib.save(hrf_img, hrf_map_path)

    # Save ground truth activation map
    activation_map_3d = np.repeat(
        ground_truth["activation_map"][:, :, np.newaxis], data.shape[2], axis=2
    )
    activation_map_path = sim_dir / "ground_truth_activation_map.nii.gz"
    activation_img = nib.Nifti1Image(activation_map_3d.astype(np.int16), affine)
    nib.save(activation_img, activation_map_path)

    # Save metadata
    metadata_path = sim_dir / "simulation_info.txt"
    with open(metadata_path, "w") as f:
        f.write(f"ISI Simulation Metadata\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"ISI Mean: {isi_mean:.1f}s\n")
        f.write(f"HRF Library: {hrf_library_name}\n")
        f.write(f"TR: {tr}s\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Number of HRFs in library: {len(ground_truth['hrfs'])}\n")
        f.write(
            f"Number of activation patterns: {len(ground_truth['activation_patterns'])}\n"
        )
        f.write(f"Noise levels (σ): {ground_truth['noise_levels']}\n")
        f.write(f"\nFiles:\n")
        f.write(f"  - func_data.nii.gz: fMRI timeseries data\n")
        f.write(f"  - design_matrix.txt: Unconvolved design matrix\n")
        f.write(f"  - ground_truth_hrf_map.nii.gz: Which HRF used for each voxel\n")
        f.write(
            f"  - ground_truth_activation_map.nii.gz: Which activation pattern for each voxel\n"
        )

    print(f"  Saved NIfTI data to: {sim_dir}/")

    return sim_dir


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_isi_distribution(isis: np.ndarray, target_mean: float, ax=None):
    """Plot ISI distribution histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(isis, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(
        target_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target: {target_mean}s",
    )
    ax.axvline(
        isis.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Actual: {isis.mean():.2f}s",
    )
    ax.set_xlabel("ISI (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("ISI Distribution (Truncated Poisson)")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_design_matrix(design: np.ndarray, title: str = "Design Matrix", ax=None):
    """Plot design matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for cond_idx in range(design.shape[1]):
        ax.plot(design[:, cond_idx], label=f"Condition {cond_idx + 1}", alpha=0.7)

    ax.set_xlabel("Time (TR)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_hrf_recovery(
    ground_truth_map: np.ndarray, estimated_map: np.ndarray, slice_idx: int = 2, ax=None
):
    """Plot HRF recovery: ground truth vs estimated."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = ax

    # Ground truth
    im1 = axes[0].imshow(
        ground_truth_map[:, :, slice_idx], cmap="tab20", interpolation="nearest"
    )
    axes[0].set_title("Ground Truth HRF Map")
    axes[0].set_xlabel("Y (activation pattern)")
    axes[0].set_ylabel("X (HRF)")
    plt.colorbar(im1, ax=axes[0], label="HRF Index")

    # Estimated
    im2 = axes[1].imshow(
        estimated_map[:, :, slice_idx], cmap="tab20", interpolation="nearest"
    )
    axes[1].set_title("Estimated HRF Map")
    axes[1].set_xlabel("Y (activation pattern)")
    axes[1].set_ylabel("X (HRF)")
    plt.colorbar(im2, ax=axes[1], label="HRF Index")


def plot_r2_map(r2_map: np.ndarray, slice_idx: int = 2, ax=None):
    """Plot R² map."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        r2_map[:, :, slice_idx], cmap="hot", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title(f"R² Map (Slice {slice_idx})")
    ax.set_xlabel("Y (activation pattern)")
    ax.set_ylabel("X (HRF)")
    plt.colorbar(im, ax=ax, label="R²")


def plot_design_efficiency_comparison(
    results_list: List[Dict], metric: str = "efficiency", ax=None
):
    """Plot design efficiency metric across ISI means."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    isi_means = []
    efficiency_values = []

    for result in results_list:
        isi_means.append(result["isi_mean"])
        # Average efficiency across all HRFs
        efficiencies = [
            res["design_efficiency"][metric] for res in result["ols_results"]
        ]
        efficiency_values.append(np.mean(efficiencies))

    ax.plot(isi_means, efficiency_values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Mean ISI (seconds)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Design {metric.replace('_', ' ').title()} vs ISI")
    ax.grid(alpha=0.3)


def create_comprehensive_figure(result: Dict, output_path: Path):
    """Create comprehensive figure for single simulation."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # ISI distribution
    ax1 = fig.add_subplot(gs[0, 0])
    plot_isi_distribution(result["isis"], result["isi_mean"], ax=ax1)

    # Design matrix (unconvolved)
    ax2 = fig.add_subplot(gs[0, 1:3])
    plot_design_matrix(
        result["design_unconvolved"], "Unconvolved Design Matrix", ax=ax2
    )

    # Design matrix (convolved with first HRF)
    ax3 = fig.add_subplot(gs[0, 3])
    hrf = result["ground_truth"]["hrfs"][0]
    X_conv = convolve_design_with_hrf(result["design_unconvolved"], hrf)
    ax3.plot(X_conv[:, 0], label="Cond A", alpha=0.7)
    ax3.plot(X_conv[:, 1], label="Cond B", alpha=0.7)
    ax3.set_title("Convolved Design (HRF 0)")
    ax3.set_xlabel("Time (TR)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # HRF library
    ax4 = fig.add_subplot(gs[1, 0])
    hrfs = result["ground_truth"]["hrfs"]
    for i, hrf in enumerate(hrfs[::4]):  # Plot every 4th HRF
        ax4.plot(hrf, alpha=0.6, label=f"HRF {i * 4}")
    ax4.set_title(f"HRF Library ({result['hrf_library_name']})")
    ax4.set_xlabel("Time (TR)")
    ax4.set_ylabel("Amplitude")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # Ground truth HRF map
    ax5 = fig.add_subplot(gs[1, 1])
    gt_map_3d = np.repeat(
        result["ground_truth"]["hrf_map"][:, :, np.newaxis],
        result["best_hrf_map"].shape[2],
        axis=2,
    )
    im5 = ax5.imshow(gt_map_3d[:, :, 2], cmap="tab20", interpolation="nearest")
    ax5.set_title("Ground Truth HRF Map")
    ax5.set_xlabel("Y (activation)")
    ax5.set_ylabel("X (HRF)")
    plt.colorbar(im5, ax=ax5, label="HRF Index", fraction=0.046)

    # Estimated HRF map
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(
        result["best_hrf_map"][:, :, 2], cmap="tab20", interpolation="nearest"
    )
    ax6.set_title(
        f"Estimated HRF Map (Acc={result['hrf_recovery_accuracy'] * 100:.1f}%)"
    )
    ax6.set_xlabel("Y (activation)")
    ax6.set_ylabel("X (HRF)")
    plt.colorbar(im6, ax=ax6, label="HRF Index", fraction=0.046)

    # R² map
    ax7 = fig.add_subplot(gs[1, 3])
    im7 = ax7.imshow(
        result["best_r2_map"][:, :, 2],
        cmap="hot",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax7.set_title(f"R² Map (Mean={result['best_r2_map'].mean():.3f})")
    ax7.set_xlabel("Y (activation)")
    ax7.set_ylabel("X (HRF)")
    plt.colorbar(im7, ax=ax7, label="R²", fraction=0.046)

    # HRF recovery by noise level
    ax8 = fig.add_subplot(gs[2, 0])
    gt_map_flat = gt_map_3d.reshape(-1, gt_map_3d.shape[2])
    est_map_flat = result["best_hrf_map"].reshape(-1, result["best_hrf_map"].shape[2])

    accuracies_by_slice = []
    for slice_idx in range(gt_map_3d.shape[2]):
        acc = np.mean(gt_map_flat[:, slice_idx] == est_map_flat[:, slice_idx])
        accuracies_by_slice.append(acc)

    noise_levels = result["ground_truth"]["noise_levels"]
    ax8.plot(
        noise_levels,
        np.array(accuracies_by_slice) * 100,
        "o-",
        linewidth=2,
        markersize=8,
    )
    ax8.set_xlabel("Noise Level (σ)")
    ax8.set_ylabel("HRF Recovery Accuracy (%)")
    ax8.set_title("Recovery vs Noise")
    ax8.grid(alpha=0.3)

    # Mean R² by noise level
    ax9 = fig.add_subplot(gs[2, 1])
    r2_by_slice = []
    for slice_idx in range(result["best_r2_map"].shape[2]):
        r2_by_slice.append(result["best_r2_map"][:, :, slice_idx].mean())

    ax9.plot(noise_levels, r2_by_slice, "o-", linewidth=2, markersize=8, color="red")
    ax9.set_xlabel("Noise Level (σ)")
    ax9.set_ylabel("Mean R²")
    ax9.set_title("R² vs Noise")
    ax9.grid(alpha=0.3)

    # Design efficiency metrics
    ax10 = fig.add_subplot(gs[2, 2])
    efficiency_metrics = [
        "efficiency",
        "condition_number",
        "vif_mean",
        "correlation_max",
    ]
    metric_values = []
    for metric in efficiency_metrics:
        values = [res["design_efficiency"][metric] for res in result["ols_results"]]
        metric_values.append(np.mean(values))

    ax10.bar(range(len(efficiency_metrics)), metric_values, alpha=0.7)
    ax10.set_xticks(range(len(efficiency_metrics)))
    ax10.set_xticklabels([m.replace("_", "\n") for m in efficiency_metrics], fontsize=8)
    ax10.set_ylabel("Value")
    ax10.set_title("Design Efficiency Metrics")
    ax10.grid(alpha=0.3, axis="y")

    # Summary text
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis("off")
    summary_text = f"""
SIMULATION SUMMARY

ISI Mean: {result["isi_mean"]:.1f}s
HRF Library: {result["hrf_library_name"]}
Number of HRFs: {len(result["ground_truth"]["hrfs"])}

Design:
  - Conditions: {result["design_unconvolved"].shape[1]}
  - Timepoints: {result["design_unconvolved"].shape[0]}
  - Total trials: {np.sum(result["design_unconvolved"])}

Data:
  - Voxels: {result["data"].size // result["data"].shape[-1]:,}
  - Noise levels: {len(noise_levels)}

Results:
  - HRF Recovery: {result["hrf_recovery_accuracy"] * 100:.1f}%
  - Mean R²: {result["best_r2_map"].mean():.3f}
  - Median R²: {np.median(result["best_r2_map"]):.3f}
    """
    ax11.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.suptitle(
        f"ISI Simulation: Mean={result['isi_mean']}s, Library={result['hrf_library_name']}",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {output_path}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Setup
    device = ffs.get_device()
    print(f"Using device: {device}\n")

    # Output directory
    output_dir = Path("simulation_results_isi_sweep")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Simulation parameters
    isi_means = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    hrf_libraries = ["cnvlab", "canonical"]

    # Run simulations
    all_results = []

    total_sims = len(isi_means) * len(hrf_libraries)
    sim_num = 0

    start_time = time.time()

    for hrf_library in hrf_libraries:
        for isi_mean in isi_means:
            sim_num += 1
            print(f"\n{'=' * 70}")
            print(f"SIMULATION {sim_num}/{total_sims}")
            print(f"{'=' * 70}")

            # Run simulation
            result = run_isi_simulation(
                isi_mean=isi_mean,
                hrf_library_name=hrf_library,
                tr=1.0,
                total_duration=290.0,
                stim_duration=5.0,
                n_runs=4,
                n_conds=2,
                device=device,
                seed=42 + sim_num,  # Different seed for each simulation
                verbose=True,
            )

            all_results.append(result)

            # Save NIfTI data
            print(f"\nSaving NIfTI data...")
            save_simulation_nifti(
                data=result["data"],
                design_unconvolved=result["design_unconvolved"],
                ground_truth=result["ground_truth"],
                isi_mean=isi_mean,
                hrf_library_name=hrf_library,
                output_dir=output_dir,
                tr=1.0,
            )

            # Create comprehensive figure
            output_filename = f"simulation_isi{isi_mean:.1f}_{hrf_library}.png"
            output_path = output_dir / output_filename

            print(f"\nCreating figure...")
            create_comprehensive_figure(result, output_path)

    elapsed = time.time() - start_time

    # Create summary comparison figures
    print(f"\n{'=' * 70}")
    print("Creating summary comparison figures...")
    print(f"{'=' * 70}\n")

    # Group by library
    for hrf_library in hrf_libraries:
        library_results = [
            r for r in all_results if r["hrf_library_name"] == hrf_library
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # HRF recovery accuracy vs ISI
        ax = axes[0, 0]
        isi_vals = [r["isi_mean"] for r in library_results]
        accuracy_vals = [r["hrf_recovery_accuracy"] * 100 for r in library_results]
        ax.plot(isi_vals, accuracy_vals, "o-", linewidth=2, markersize=10)
        ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
        ax.set_ylabel("HRF Recovery Accuracy (%)", fontsize=12)
        ax.set_title(f"HRF Recovery vs ISI ({hrf_library})", fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

        # Mean R² vs ISI
        ax = axes[0, 1]
        r2_vals = [r["best_r2_map"].mean() for r in library_results]
        ax.plot(isi_vals, r2_vals, "o-", linewidth=2, markersize=10, color="red")
        ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
        ax.set_ylabel("Mean R²", fontsize=12)
        ax.set_title(f"R² vs ISI ({hrf_library})", fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # Design efficiency vs ISI
        ax = axes[1, 0]
        efficiency_vals = []
        for r in library_results:
            effs = [res["design_efficiency"]["efficiency"] for res in r["ols_results"]]
            efficiency_vals.append(np.mean(effs))
        ax.plot(
            isi_vals, efficiency_vals, "o-", linewidth=2, markersize=10, color="green"
        )
        ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
        ax.set_ylabel("Mean Design Efficiency", fontsize=12)
        ax.set_title(f"Design Efficiency vs ISI ({hrf_library})", fontsize=14)
        ax.grid(alpha=0.3)

        # Condition number vs ISI
        ax = axes[1, 1]
        condnum_vals = []
        for r in library_results:
            condnums = [
                res["design_efficiency"]["condition_number"] for res in r["ols_results"]
            ]
            condnum_vals.append(np.mean(condnums))
        ax.plot(
            isi_vals, condnum_vals, "o-", linewidth=2, markersize=10, color="purple"
        )
        ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
        ax.set_ylabel("Mean Condition Number", fontsize=12)
        ax.set_title(f"Condition Number vs ISI ({hrf_library})", fontsize=14)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        summary_path = output_dir / f"summary_{hrf_library}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"Saved summary: {summary_path}")
        plt.close()

    # Compare libraries
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # HRF recovery comparison
    ax = axes[0]
    for hrf_library in hrf_libraries:
        library_results = [
            r for r in all_results if r["hrf_library_name"] == hrf_library
        ]
        isi_vals = [r["isi_mean"] for r in library_results]
        accuracy_vals = [r["hrf_recovery_accuracy"] * 100 for r in library_results]
        ax.plot(
            isi_vals, accuracy_vals, "o-", linewidth=2, markersize=10, label=hrf_library
        )
    ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
    ax.set_ylabel("HRF Recovery Accuracy (%)", fontsize=12)
    ax.set_title("HRF Recovery: Library Comparison", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    # R² comparison
    ax = axes[1]
    for hrf_library in hrf_libraries:
        library_results = [
            r for r in all_results if r["hrf_library_name"] == hrf_library
        ]
        isi_vals = [r["isi_mean"] for r in library_results]
        r2_vals = [r["best_r2_map"].mean() for r in library_results]
        ax.plot(isi_vals, r2_vals, "o-", linewidth=2, markersize=10, label=hrf_library)
    ax.set_xlabel("Mean ISI (seconds)", fontsize=12)
    ax.set_ylabel("Mean R²", fontsize=12)
    ax.set_title("R²: Library Comparison", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    comparison_path = output_dir / "library_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison: {comparison_path}")
    plt.close()

    # Final summary
    print(f"\n{'=' * 70}")
    print("SIMULATION SWEEP COMPLETE!")
    print(f"{'=' * 70}\n")
    print(f"Total simulations: {total_sims}")
    print(
        f"Total time: {elapsed / 60:.1f} minutes ({elapsed / total_sims:.1f}s per simulation)"
    )
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - {total_sims} individual simulation figures (.png)")
    print(f"  - {len(hrf_libraries)} library summary figures (.png)")
    print(f"  - 1 library comparison figure (.png)")
    print(f"  - {total_sims} NIfTI datasets (func_data.nii.gz + ground truth maps)")
    print(f"  - {total_sims} design matrices (.txt)")
    print(
        f"\nTotal: {total_sims + len(hrf_libraries) + 1} figures + {total_sims} NIfTI datasets"
    )
