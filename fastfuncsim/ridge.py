"""
Ridge regression for fMRI with single-trial estimation

This module implements GPU-accelerated ridge regression using fracridge,
with support for:
- Single-trial beta estimation (one beta per event)
- Per-voxel HRF selection (from HRFoptfast output)
- Noise regressor integration (from Denoisefast output)
- Cross-validated ridge fraction selection
- Non-TR-locked onsets and variable durations

Design philosophy:
- Reproduces GLMsingle functionality but GPU-accelerated
- Compatible with existing fastfuncsim HRF and denoising pipelines
- Supports flexible timing (non-TR-locked, variable durations)
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from fracridge import FracRidgeRegressor
except ImportError:
    raise ImportError(
        "fracridge is required for ridge regression. "
        "Install with: pip install fracridge"
    )


@dataclass
class RidgeResults:
    """Results from ridge regression GLM fitting

    Attributes
    ----------
    betas_single_trial : torch.Tensor, shape (n_voxels, n_trials)
        Single-trial beta estimates at optimal ridge fraction
    r2 : torch.Tensor, shape (n_voxels,)
        R² for each voxel at optimal fraction
    xval_r2 : torch.Tensor, shape (n_voxels,)
        Cross-validated R² at optimal fraction
    optimal_fracs : torch.Tensor, shape (n_voxels,)
        Optimal ridge fraction per voxel (selected via CV)
    r2_by_frac : torch.Tensor, shape (n_voxels, n_fracs)
        Cross-validated R² for each ridge fraction
    trial_labels : List[str]
        Labels for each trial (condition + trial number)
    metadata : Dict
        Processing metadata
    """

    betas_single_trial: torch.Tensor
    r2: torch.Tensor
    xval_r2: torch.Tensor
    optimal_fracs: torch.Tensor
    r2_by_frac: torch.Tensor
    trial_labels: List[str]
    metadata: Dict


def create_single_trial_design(
    onsets_by_condition: List[List[np.ndarray]],
    durations: List[float],
    run_starts: List[int],
    tr: float,
    n_timepoints: int,
    hrf_library: Optional[List[torch.Tensor]] = None,
    hrf_index_per_voxel: Optional[torch.Tensor] = None,
    microtime_dt: float = 0.1,
    condition_labels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Create single-trial design matrix with optional per-voxel HRFs

    Each trial (event) gets its own regressor, allowing estimation of
    trial-specific beta weights. This is the core of GLMsingle-style analysis.

    Parameters
    ----------
    onsets_by_condition : list of list of np.ndarray
        Onsets organized as [condition][run] -> np.ndarray of onset times (seconds)
    durations : list of float
        Duration in seconds for each condition
    run_starts : list of int
        Starting timepoint index for each run
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of timepoints
    hrf_library : list of torch.Tensor, optional
        Library of HRF shapes. Each tensor has shape (hrf_length,).
        If None, uses canonical HRF.
    hrf_index_per_voxel : torch.Tensor, optional
        HRF index for each voxel (0-indexed). Shape (n_voxels,).
        If provided, creates per-voxel design matrices.
    microtime_dt : float, default=0.1
        Microtime resolution in seconds for non-TR-locked onsets
    condition_labels : list of str, optional
        Labels for each condition (for trial naming)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design_matrix : torch.Tensor
        If hrf_index_per_voxel is None: (n_timepoints, n_trials)
        If hrf_index_per_voxel provided: (n_voxels, n_timepoints, n_trials)
    trial_labels : list of str
        Label for each trial (e.g., "face_001", "house_023")
    """
    from .design import convolve_hrf_microtime
    from .hrf import get_canonical_hrf

    device = device or torch.device("cpu")
    n_conditions = len(onsets_by_condition)

    if condition_labels is None:
        condition_labels = [f"cond{i+1:02d}" for i in range(n_conditions)]

    # Count total trials across all conditions and runs
    total_trials = 0
    trial_info = []  # (condition_idx, run_idx, trial_idx_in_run, onset_time)

    for cond_idx, runs_onsets in enumerate(onsets_by_condition):
        for run_idx, run_onsets in enumerate(runs_onsets):
            for trial_idx, onset_time in enumerate(run_onsets):
                trial_info.append((cond_idx, run_idx, trial_idx, onset_time))
                total_trials += 1

    # Generate trial labels
    trial_labels = []
    for cond_idx, run_idx, trial_idx, onset_time in trial_info:
        label = f"{condition_labels[cond_idx]}_{trial_idx+1:03d}"
        trial_labels.append(label)

    # Create onset matrix at microtime resolution
    bins_per_tr = int(round(tr / microtime_dt))
    n_microtime = n_timepoints * bins_per_tr

    # Build single-trial onset matrix (microtime)
    onset_matrix_micro = torch.zeros(
        n_microtime, total_trials, dtype=torch.float32, device=device
    )

    for trial_idx, (cond_idx, run_idx, trial_in_run, onset_time) in enumerate(trial_info):
        # Convert onset time to microtime bin
        run_start_time = run_starts[run_idx] * tr
        onset_relative = onset_time + run_start_time
        onset_bin = int(round(onset_relative / microtime_dt))

        # Duration in microtime bins
        duration_bins = int(round(durations[cond_idx] / microtime_dt))

        # Set boxcar (handle edge cases)
        start_bin = max(0, onset_bin)
        end_bin = min(n_microtime, onset_bin + duration_bins)
        if end_bin > start_bin:
            onset_matrix_micro[start_bin:end_bin, trial_idx] = 1.0

    # Apply HRF convolution
    if hrf_index_per_voxel is None:
        # Single design matrix for all voxels
        if hrf_library is None or len(hrf_library) == 0:
            # Use canonical HRF
            hrf = get_canonical_hrf(
                stim_duration=0.0, tr=tr, duration=32.0, device=device
            )
        else:
            # Use first HRF from library (or could default to canonical)
            hrf = hrf_library[0].to(device)

        # Convolve - returns (n_timepoints, n_trials)
        design_matrix = convolve_hrf_microtime(
            onset_matrix_micro,
            hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
            return_single_trials=False,
        )

        return design_matrix, trial_labels

    else:
        # Per-voxel design matrices
        n_voxels = hrf_index_per_voxel.shape[0]
        n_hrfs = len(hrf_library) if hrf_library is not None else 0

        if n_hrfs == 0:
            raise ValueError("hrf_library must be provided when using per-voxel HRFs")

        # Pre-compute convolved designs for each HRF
        designs_by_hrf = []
        for hrf_idx in range(n_hrfs):
            hrf = hrf_library[hrf_idx].to(device)
            design_tr = convolve_hrf_microtime(
                onset_matrix_micro,
                hrf,
                n_timepoints=n_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
                return_single_trials=False,
            )
            designs_by_hrf.append(design_tr)

        # Stack: (n_hrfs, n_timepoints, n_trials)
        designs_stacked = torch.stack(designs_by_hrf, dim=0)

        # Select per-voxel designs
        # Result: (n_voxels, n_timepoints, n_trials)
        hrf_indices = hrf_index_per_voxel.long().to(device)
        design_per_voxel = designs_stacked[hrf_indices, :, :]

        return design_per_voxel, trial_labels


def fit_ridge_single_trial(
    data: torch.Tensor,
    design_matrix: Union[torch.Tensor, List[torch.Tensor]],
    run_starts: List[int],
    fracs: Optional[np.ndarray] = None,
    nuisance: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    cv_splits: Optional[List[Tuple[List[int], List[int]]]] = None,
    trial_labels: Optional[List[str]] = None,
    chunk_size: int = 10000,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> RidgeResults:
    """
    Fit ridge regression with single-trial design using fracridge

    Uses cross-validation to select optimal ridge fraction per voxel.
    Supports per-voxel design matrices for HRF-specific regressors.

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        fMRI data
    design_matrix : torch.Tensor or list of torch.Tensor
        Single design: (n_timepoints, n_trials)
        Per-voxel designs: list of (n_timepoints, n_trials) per voxel chunk
        OR (n_voxels, n_timepoints, n_trials) for full per-voxel
    run_starts : list of int
        Starting timepoint for each run
    fracs : np.ndarray, optional
        Ridge fractions to test. Default: np.arange(0.05, 1.05, 0.05)
    nuisance : torch.Tensor or list of torch.Tensor, optional
        Nuisance regressors (e.g., drift, motion, noise PCs).
        Per-run format: list of (n_timepoints_run, n_nuisance)
    cv_splits : list of tuples, optional
        Cross-validation splits as (train_runs, test_runs).
        Default: leave-one-run-out
    trial_labels : list of str, optional
        Labels for each trial
    chunk_size : int, default=10000
        Number of voxels to process at once
    device : torch.device, optional
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    results : RidgeResults
        Ridge regression results with per-trial betas

    Notes
    -----
    This function implements the GLMsingle ridge regression approach:
    1. Fit ridge regression for multiple fractions
    2. Select optimal fraction per voxel via cross-validation
    3. Refit with optimal fraction for final beta estimates

    The per-voxel design matrix support allows using different HRFs
    per voxel (from HRFoptfast output).
    """
    device = device or torch.device("cpu")
    n_voxels, n_timepoints = data.shape

    # Default ridge fractions (similar to GLMsingle)
    if fracs is None:
        fracs = np.arange(0.05, 1.05, 0.05)

    n_fracs = len(fracs)

    # Generate CV splits if not provided (LORO by default)
    if cv_splits is None:
        n_runs = len(run_starts)
        from .xval import generate_cv_splits
        cv_splits = generate_cv_splits(n_runs, strategy=1, n_perms=n_runs)

    # Determine if we have per-voxel designs
    per_voxel_design = isinstance(design_matrix, torch.Tensor) and design_matrix.ndim == 3

    if per_voxel_design:
        if verbose:
            print("Using per-voxel HRF-specific design matrices")
        n_trials = design_matrix.shape[2]
    else:
        n_trials = design_matrix.shape[1] if isinstance(design_matrix, torch.Tensor) else design_matrix[0].shape[1]

    if verbose:
        print(f"\nRidge regression single-trial estimation")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Trials: {n_trials}")
        print(f"  Ridge fractions: {n_fracs} ({fracs[0]:.2f} to {fracs[-1]:.2f})")
        print(f"  CV splits: {len(cv_splits)}")
        print()

    # Placeholder for actual implementation
    # This will be expanded with:
    # 1. Project out nuisance per run
    # 2. Cross-validate ridge fractions
    # 3. Select optimal fraction per voxel
    # 4. Refit with optimal fractions
    # 5. Return results

    raise NotImplementedError(
        "Ridge regression fitting will be implemented in next iteration. "
        "Core infrastructure is in place."
    )


def load_hrf_indices(hrf_index_file: str, mask: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Load HRF indices from HRFoptfast output

    Parameters
    ----------
    hrf_index_file : str
        Path to {prefix}_hrf_index.nii.gz from HRFoptfast
    mask : np.ndarray, optional
        Boolean mask to apply (if data was masked)

    Returns
    -------
    hrf_indices : torch.Tensor, shape (n_voxels,)
        HRF index per voxel (0-indexed, converted from 1-indexed NIFTI)
    """
    import nibabel as nib

    img = nib.load(hrf_index_file)
    hrf_data = img.get_fdata()

    # Convert from 1-indexed (NIFTI) to 0-indexed (Python)
    hrf_data = hrf_data - 1.0

    if mask is not None:
        hrf_data = hrf_data[mask]
    else:
        hrf_data = hrf_data.flatten()

    return torch.from_numpy(hrf_data).long()


def load_noise_pcs(noise_pc_file: str, run_starts: List[int], n_timepoints: int) -> List[torch.Tensor]:
    """
    Load noise PCs from Denoisefast output

    Parameters
    ----------
    noise_pc_file : str
        Path to {prefix}_noise_pcs.xmat.1D from Denoisefast
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints

    Returns
    -------
    noise_pcs_per_run : list of torch.Tensor
        List of per-run noise PC matrices, each (n_timepoints_run, n_pcs)
    """
    # Load PC timecourses
    pcs = np.loadtxt(noise_pc_file)  # (n_timepoints, n_pcs)
    pcs_tensor = torch.from_numpy(pcs).float()

    # Split by run
    n_runs = len(run_starts)
    pcs_per_run = []

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        pcs_per_run.append(pcs_tensor[start_tp:end_tp, :])

    return pcs_per_run
