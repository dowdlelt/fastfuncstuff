"""
Cross-validated HRF selection per voxel

This module provides the core engine for selecting the optimal HRF per voxel
using cross-validation across runs. Unlike in-sample selection which can overfit,
CV-based selection provides a more reliable estimate of which HRF shape best
captures the true hemodynamic response for each voxel.

Key function: fit_glm_hrf_library_with_xval()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .design import convolve_hrf, convolve_hrf_microtime
from .glm_core import GLMResults, fit_glm
from .hrf import get_hrf_library
from .utils import get_device, to_tensor
from .xval import generate_cv_splits, slice_by_runs


@dataclass
class HRFSelectionResults:
    """Container for voxel-wise HRF selection results.

    Attributes
    ----------
    hrf_index : torch.Tensor
        (n_voxels,) Index of best HRF for each voxel (0 to n_hrfs-1)
    xval_r2_best : torch.Tensor
        (n_voxels,) Median cross-validated R² for the selected HRF
    xval_r2_std : torch.Tensor
        (n_voxels,) Std of cross-validated R² across CV splits for selected HRF
    xval_r2_all_hrfs : torch.Tensor
        (n_voxels, n_hrfs) Median CV R² for each HRF (for diagnostics)
    final_results : GLMResults
        Results from final full-data fit with voxel-wise optimal HRFs
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_timepoints) The HRF library used (stored for ARMA reuse)
    hrf_metadata : dict
        Metadata about HRF selection: mode, tr, stim_durations, cv_strategy, etc.
    """

    hrf_index: torch.Tensor = None
    xval_r2_best: torch.Tensor = None
    xval_r2_std: torch.Tensor = None
    xval_r2_all_hrfs: torch.Tensor = None
    final_results: GLMResults = None
    hrf_library: torch.Tensor = None
    hrf_metadata: Dict = field(default_factory=dict)

    # For ARMA integration: store the convolved design per HRF group
    # This allows reloading without reconvolving
    hrf_group_indices: Dict[int, torch.Tensor] = field(default_factory=dict)


def fit_glm_hrf_library_with_xval(
    data: torch.Tensor,
    onsets: torch.Tensor,
    hrf_library: torch.Tensor,
    tr: float,
    run_starts: List[int],
    stim_durations: Optional[List[float]] = None,
    cv_strategy: Union[float, int] = 1,
    n_perms: int = 100,
    metric: str = "cod",
    microtime_resolution: int = 16,
    microtime_onset: Optional[int] = None,
    polort: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    chunk_size: Optional[int] = None,
) -> HRFSelectionResults:
    """
    Select best HRF per voxel using cross-validated R².

    This function:
    1. Loops through each HRF in the library
    2. For each HRF, computes cross-validated R² across CV splits
    3. Selects the HRF with highest median CV R² per voxel
    4. Refits the full dataset using voxel-wise optimal HRFs

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    onsets : torch.Tensor
        (n_timepoints, n_conditions) or (n_microtime, n_conditions) binary onset matrix
        If microtime_resolution > 1, should be at microtime resolution
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_timepoints) Library of HRF candidates at TR resolution
    tr : float
        Repetition time in seconds
    run_starts : list of int
        Starting timepoint index for each run (required for CV splits)
    stim_durations : list of float, optional
        Duration in seconds for each condition. If None, assumes impulse (0s).
        Single value applies to all conditions.
    cv_strategy : float or int, default=1
        Cross-validation strategy:
        - int: Leave-N-out (1 = LORO, 2 = leave-2-out)
        - float: Split fraction (0.5 = split halves)
    n_perms : int, default=100
        Number of permutations for random split strategies
    metric : str, default='cod'
        R² metric: 'cod' (coefficient of determination), 'corr', or 'corr2'
    microtime_resolution : int, default=16
        Sub-TR resolution for onset timing
    microtime_onset : int, optional
        Which microtime bin to sample (default: middle of TR)
    polort : int, optional
        Polynomial order for detrending (None = auto)
    device : torch.device, optional
        Compute device (auto-detected if None)
    verbose : bool, default=True
        Print progress information
    chunk_size : int, optional
        Number of voxels to process at once (auto if None)

    Returns
    -------
    HRFSelectionResults
        Container with:
        - hrf_index: (n_voxels,) best HRF per voxel
        - xval_r2_best: (n_voxels,) CV R² for selected HRF
        - xval_r2_std: (n_voxels,) std across CV splits
        - xval_r2_all_hrfs: (n_voxels, n_hrfs) CV R² for all HRFs
        - final_results: GLMResults from full refit
        - hrf_library: The HRF library used (for ARMA reuse)
        - hrf_metadata: Selection parameters

    Notes
    -----
    The final fit groups voxels by their selected HRF to maintain GPU efficiency.
    This is similar to how 3dREMLfast handles voxel-wise ARMA parameters.
    """
    if device is None:
        device = get_device()

    # Set microtime_onset default
    if microtime_onset is None:
        microtime_onset = microtime_resolution // 2 + 1

    # Move data to device
    data = to_tensor(data, device=device)
    onsets = to_tensor(onsets, device=device)
    hrf_library = to_tensor(hrf_library, device=device)

    n_voxels, n_timepoints_data = data.shape
    n_hrfs = hrf_library.shape[0]
    n_runs = len(run_starts)

    # Determine n_timepoints at TR resolution
    if microtime_resolution > 1:
        n_timepoints = onsets.shape[0] // microtime_resolution
    else:
        n_timepoints = onsets.shape[0]

    # Validate data/design alignment
    if n_timepoints_data != n_timepoints:
        raise ValueError(
            f"Data has {n_timepoints_data} timepoints but design implies {n_timepoints}. "
            f"Check microtime_resolution setting."
        )

    if verbose:
        print("=" * 70)
        print("CROSS-VALIDATED HRF SELECTION")
        print("=" * 70)
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Runs: {n_runs}")
        print(f"  HRF candidates: {n_hrfs}")
        print(
            f"  CV strategy: {cv_strategy} ({'LORO' if cv_strategy == 1 else 'split-halves' if cv_strategy == 0.5 else cv_strategy})"
        )
        print(f"  Microtime: {microtime_resolution}x (onset bin {microtime_onset})")
        print()

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    if verbose:
        print(f"  CV splits: {n_splits}")
        print()

    # Storage for CV results: (n_voxels, n_hrfs, n_splits)
    xval_r2_all = torch.zeros(n_voxels, n_hrfs, n_splits, device=device)

    # Main loop: evaluate each HRF via cross-validation
    hrf_iterator = tqdm(range(n_hrfs), desc="HRF candidates", disable=not verbose)

    for hrf_idx in hrf_iterator:
        hrf = hrf_library[hrf_idx]

        # Convolve onsets with this HRF
        if microtime_resolution > 1:
            design_convolved = convolve_hrf_microtime(
                onsets,
                hrf,
                n_timepoints,
                microtime_resolution,
                microtime_onset=microtime_onset,
                device=device,
            )
        else:
            design_convolved = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Cross-validation loop
        for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
            # Slice data and design for train/test
            # slice_by_runs expects (n_voxels, n_timepoints) and returns same shape
            data_train, design_train, _ = slice_by_runs(
                data, design_convolved, run_starts, train_runs
            )
            data_test, design_test, _ = slice_by_runs(
                data, design_convolved, run_starts, test_runs
            )

            # data_train: (n_voxels, n_timepoints_train)
            # design_train: (n_timepoints_train, n_regressors)

            # Fit OLS on training data
            # Simple least squares: beta = (X'X)^-1 X'Y
            XtX = design_train.T @ design_train  # (n_regressors, n_regressors)
            XtY = design_train.T @ data_train.T  # (n_regressors, n_voxels)

            # Solve for betas
            try:
                betas = torch.linalg.solve(XtX, XtY)  # (n_regressors, n_voxels)
            except torch.linalg.LinAlgError:
                # Singular matrix - use pseudoinverse
                betas = torch.linalg.lstsq(XtX, XtY).solution

            # Predict test data
            predictions = design_test @ betas  # (n_test_timepoints, n_voxels)

            # Compute R² metric
            r2 = _compute_r2_metric(
                data_test.T, predictions, metric=metric, device=device
            )

            xval_r2_all[:, hrf_idx, split_idx] = r2

    # Aggregate across CV splits (median)
    xval_r2_median = xval_r2_all.median(dim=2)[0]  # (n_voxels, n_hrfs)
    xval_r2_std_all = xval_r2_all.std(dim=2)  # (n_voxels, n_hrfs)

    # Select best HRF per voxel
    hrf_index = xval_r2_median.argmax(dim=1)  # (n_voxels,)

    # Extract R² for selected HRF
    xval_r2_best = xval_r2_median[torch.arange(n_voxels, device=device), hrf_index]
    xval_r2_std = xval_r2_std_all[torch.arange(n_voxels, device=device), hrf_index]

    if verbose:
        print()
        print("HRF Selection Summary:")
        hrf_counts = torch.bincount(hrf_index, minlength=n_hrfs)
        print(f"  HRF usage distribution: {hrf_counts.cpu().tolist()}")
        print(f"  Mean xval R²: {xval_r2_best.mean().item():.4f}")
        print(f"  Median xval R²: {xval_r2_best.median().item():.4f}")
        print()

    # Final fit: refit entire dataset with voxel-wise optimal HRFs
    if verbose:
        print("Refitting full dataset with voxel-wise optimal HRFs...")

    final_results = _fit_voxelwise_hrf(
        data=data,
        onsets=onsets,
        hrf_library=hrf_library,
        hrf_index=hrf_index,
        tr=tr,
        microtime_resolution=microtime_resolution,
        microtime_onset=microtime_onset,
        polort=polort,
        device=device,
        verbose=verbose,
        chunk_size=chunk_size,
    )

    # Build metadata for ARMA reuse
    hrf_metadata = {
        "hrf_mode": "library",  # Will be set by caller if known
        "n_hrfs": n_hrfs,
        "tr": tr,
        "stim_durations": stim_durations,
        "cv_strategy": cv_strategy,
        "n_splits": n_splits,
        "metric": metric,
        "microtime_resolution": microtime_resolution,
        "microtime_onset": microtime_onset,
        "polort": polort,
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "n_runs": n_runs,
        "hrf_usage_counts": torch.bincount(hrf_index, minlength=n_hrfs).cpu().tolist(),
    }

    # Build HRF group indices for efficient ARMA reuse
    hrf_group_indices = {}
    for h in range(n_hrfs):
        mask = hrf_index == h
        if mask.any():
            hrf_group_indices[h] = torch.where(mask)[0]

    # Create results container
    results = HRFSelectionResults(
        hrf_index=hrf_index.cpu(),
        xval_r2_best=xval_r2_best.cpu(),
        xval_r2_std=xval_r2_std.cpu(),
        xval_r2_all_hrfs=xval_r2_median.cpu(),
        final_results=final_results,
        hrf_library=hrf_library.cpu(),
        hrf_metadata=hrf_metadata,
        hrf_group_indices={k: v.cpu() for k, v in hrf_group_indices.items()},
    )

    if verbose:
        print()
        print("=" * 70)
        print("HRF SELECTION COMPLETE")
        print("=" * 70)
        print(f"  Best HRF per voxel stored in hrf_index")
        print(f"  Final betas shape: {final_results.betas.shape}")
        print(f"  Final R² mean: {final_results.r2.mean().item():.4f}")
        print()

    return results


def _fit_voxelwise_hrf(
    data: torch.Tensor,
    onsets: torch.Tensor,
    hrf_library: torch.Tensor,
    hrf_index: torch.Tensor,
    tr: float,
    microtime_resolution: int,
    microtime_onset: int,
    polort: Optional[int],
    device: torch.device,
    verbose: bool,
    chunk_size: Optional[int],
) -> GLMResults:
    """
    Fit GLM with voxel-wise HRFs by grouping voxels with same HRF.

    This is the key efficiency trick: instead of fitting each voxel separately,
    we group voxels by their selected HRF and fit each group together.
    Similar to how 3dREMLfast handles voxel-wise ARMA parameters.
    """
    n_voxels = data.shape[0]
    n_hrfs = hrf_library.shape[0]

    if microtime_resolution > 1:
        n_timepoints = onsets.shape[0] // microtime_resolution
    else:
        n_timepoints = onsets.shape[0]

    n_conditions = onsets.shape[1]

    # Determine number of betas (conditions + polynomials if any)
    # For now, just conditions - polynomials handled by fit_glm

    # Initialize output tensors
    all_betas = torch.zeros(n_voxels, n_conditions, device=device)
    all_r2 = torch.zeros(n_voxels, device=device)
    all_tstats = torch.zeros(n_voxels, n_conditions, device=device)
    all_sigma2 = torch.zeros(n_voxels, device=device)

    # Group voxels by HRF
    unique_hrfs = torch.unique(hrf_index)

    if verbose:
        print(f"  Fitting {len(unique_hrfs)} HRF groups...")

    hrf_iterator = tqdm(unique_hrfs, desc="HRF groups", disable=not verbose)

    for hrf_idx in hrf_iterator:
        hrf_idx_int = hrf_idx.item()

        # Get voxels using this HRF
        voxel_mask = hrf_index == hrf_idx
        voxel_indices = torch.where(voxel_mask)[0]
        n_group_voxels = len(voxel_indices)

        if n_group_voxels == 0:
            continue

        # Get data for this group
        group_data = data[voxel_indices, :]  # (n_group_voxels, n_timepoints)

        # Convolve with this HRF
        hrf = hrf_library[hrf_idx_int]

        if microtime_resolution > 1:
            design_convolved = convolve_hrf_microtime(
                onsets,
                hrf,
                n_timepoints,
                microtime_resolution,
                microtime_onset=microtime_onset,
                device=device,
            )
        else:
            design_convolved = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Fit GLM for this group
        group_results = fit_glm(
            group_data,
            design_convolved,
            tr=tr,
            max_poly_degree=polort,
            device=device,
            verbose=False,
        )

        # Store results (with None checks for type safety)
        if group_results.betas is not None:
            all_betas[voxel_indices, :] = group_results.betas
        if group_results.r2 is not None:
            all_r2[voxel_indices] = group_results.r2

        if group_results.tstats is not None:
            all_tstats[voxel_indices, :] = group_results.tstats
        if group_results.sigma2 is not None:
            all_sigma2[voxel_indices] = group_results.sigma2

    # Build GLMResults
    results = GLMResults()
    results.betas = all_betas.cpu()
    results.r2 = all_r2.cpu()
    results.tstats = all_tstats.cpu()
    results.sigma2 = all_sigma2.cpu()
    results.meanvol = data.mean(dim=1).cpu()

    return results


def _compute_r2_metric(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute R² metric for cross-validation.

    Parameters
    ----------
    y_true : torch.Tensor
        (n_timepoints, n_voxels) true values
    y_pred : torch.Tensor
        (n_timepoints, n_voxels) predicted values
    metric : str
        'cod' (coefficient of determination), 'corr', or 'corr2'

    Returns
    -------
    r2 : torch.Tensor
        (n_voxels,) R² values
    """
    if metric == "cod":
        # Coefficient of determination: 1 - SS_res / SS_tot
        ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
        ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)

        # Avoid division by zero
        ss_tot = torch.clamp(ss_tot, min=1e-10)
        r2 = 1 - ss_res / ss_tot

        # Clamp to reasonable range (can be negative for bad predictions)
        r2 = torch.clamp(r2, min=-1.0, max=1.0)

    elif metric == "corr":
        # Pearson correlation
        y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
        y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)

        numerator = (y_true_centered * y_pred_centered).sum(dim=0)
        denominator = torch.sqrt(
            (y_true_centered**2).sum(dim=0) * (y_pred_centered**2).sum(dim=0)
        )
        denominator = torch.clamp(denominator, min=1e-10)

        r2 = numerator / denominator

    elif metric == "corr2":
        # Squared Pearson correlation
        y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
        y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)

        numerator = (y_true_centered * y_pred_centered).sum(dim=0)
        denominator = torch.sqrt(
            (y_true_centered**2).sum(dim=0) * (y_pred_centered**2).sum(dim=0)
        )
        denominator = torch.clamp(denominator, min=1e-10)

        r2 = (numerator / denominator) ** 2

    else:
        raise ValueError(f"Unknown metric: {metric}. Choose 'cod', 'corr', or 'corr2'")

    return r2


def save_hrf_selection_results(
    results: HRFSelectionResults,
    output_prefix: str,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_mask: Optional[torch.Tensor] = None,
    condition_labels: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Save HRF selection results to disk.

    Parameters
    ----------
    results : HRFSelectionResults
        Results from fit_glm_hrf_library_with_xval
    output_prefix : str
        Output file prefix (e.g., 'output/subject01')
    volume_shape : tuple, optional
        (x, y, z) shape for NIfTI output
    affine : np.ndarray, optional
        4x4 affine transformation for NIfTI
    voxel_mask : torch.Tensor, optional
        Boolean mask for voxels (if data was masked)
    condition_labels : list of str, optional
        Labels for each condition

    Returns
    -------
    output_files : dict
        Mapping of output type to file path
    """
    import json
    from pathlib import Path

    from .glm_outputs import write_glm_bucket_as_nifti

    output_prefix = Path(output_prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # 1. Save HRF index map
    hrf_index_file = f"{output_prefix}_hrf_index.nii.gz"
    _save_volume(
        results.hrf_index.float(), hrf_index_file, volume_shape, affine, voxel_mask
    )
    output_files["hrf_index"] = hrf_index_file

    # 2. Save CV R² for best HRF
    xval_r2_file = f"{output_prefix}_xval_r2.nii.gz"
    _save_volume(results.xval_r2_best, xval_r2_file, volume_shape, affine, voxel_mask)
    output_files["xval_r2"] = xval_r2_file

    # 3. Save CV R² std
    xval_std_file = f"{output_prefix}_xval_r2_std.nii.gz"
    _save_volume(results.xval_r2_std, xval_std_file, volume_shape, affine, voxel_mask)
    output_files["xval_r2_std"] = xval_std_file

    # 4. Save final betas
    if results.final_results is not None:
        results.final_results.original_shape = volume_shape
        results.final_results.affine = affine
        if voxel_mask is not None:
            results.final_results.voxel_mask = voxel_mask

        betas_file = f"{output_prefix}_stats.nii.gz"
        write_glm_bucket_as_nifti(
            results.final_results,
            betas_file,
            condition_names=condition_labels,
            volume_shape=volume_shape,
            affine=affine,
        )
        output_files["stats"] = betas_file

    # 5. Save HRF library for ARMA reuse
    hrf_lib_file = f"{output_prefix}_hrf_library.pt"
    torch.save(
        {
            "hrf_library": results.hrf_library,
            "hrf_index": results.hrf_index,
            "hrf_group_indices": results.hrf_group_indices,
            "metadata": results.hrf_metadata,
        },
        hrf_lib_file,
    )
    output_files["hrf_library"] = hrf_lib_file

    # 6. Save metadata JSON
    metadata_file = f"{output_prefix}_metadata.json"
    metadata = results.hrf_metadata.copy()
    metadata["output_files"] = {k: str(v) for k, v in output_files.items()}
    metadata["hrf_library_shape"] = list(results.hrf_library.shape)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_file

    return output_files


def load_hrf_selection_for_arma(hrf_library_file: str) -> Dict:
    """
    Load HRF selection results for ARMA analysis.

    This allows running ARMA/REML with the previously selected HRFs,
    without re-running the CV selection process.

    Parameters
    ----------
    hrf_library_file : str
        Path to {prefix}_hrf_library.pt file

    Returns
    -------
    hrf_data : dict
        Contains:
        - hrf_library: (n_hrfs, n_timepoints) HRF shapes
        - hrf_index: (n_voxels,) selected HRF per voxel
        - hrf_group_indices: dict mapping HRF index to voxel indices
        - metadata: selection parameters
    """
    return torch.load(hrf_library_file, weights_only=False)


def _save_volume(
    data: torch.Tensor,
    filepath: str,
    volume_shape: Optional[Tuple[int, int, int]],
    affine: Optional[np.ndarray],
    voxel_mask: Optional[torch.Tensor],
):
    """Helper to save a 1D tensor as a 3D NIfTI volume."""
    import nibabel as nib

    data_np = data.cpu().numpy()

    if volume_shape is not None:
        if voxel_mask is not None:
            # Unmask data
            full_volume = np.zeros(np.prod(volume_shape), dtype=np.float32)
            full_volume[voxel_mask.cpu().numpy()] = data_np
            volume_data = full_volume.reshape(volume_shape)
        else:
            volume_data = data_np.reshape(volume_shape)
    else:
        # Save as 1D
        volume_data = data_np

    if affine is None:
        affine = np.eye(4)

    img = nib.Nifti1Image(volume_data.astype(np.float32), affine)
    nib.save(img, filepath)
