"""
Cross-validated denoising via adaptive nuisance regressor selection

This module implements a sophisticated approach to data-driven denoising:
1. Identify noise pool voxels (low task R²) vs criteria voxels (high task R²)
2. Extract principal components from noise pool as candidate nuisance regressors
3. Cross-validate to select optimal number of PCs that maximizes prediction
4. Always train on denoised data but test on raw data to prevent overfitting

The key innovation: we denoise training data but predict non-denoised test data,
ensuring we're improving signal recovery rather than just fitting the denoising.

Memory Strategy (for 16GB GPU)
------------------------------
The implementation is carefully designed to handle large datasets efficiently:

1. **PCA Extraction** (extract_noise_pcs_per_run):
   - Requires full noise pool voxels in memory (can't chunk PCA across voxels)
   - But noise pool is typically 10-50% of brain voxels (subset of full data)
   - Extracts lightweight PC timecourses (n_timepoints x n_components)
   - PCs are cached and reused throughout cross-validation
   
2. **GLM Fitting** (fit_glm):
   - Supports automatic voxel chunking via chunk_size parameter
   - Passes chunk_size through entire stack: fit_denoising_model → cross_validate_noise_pcs → fit_glm_with_noise_pcs → fit_glm
   - For 16GB GPU: chunk_size=None (auto-detect) works for most datasets
   - For larger datasets: explicit chunk_size or keep_on_cpu=True
   
3. **Cross-Validation**:
   - Splits data into train/test per fold (temporal split, not voxel)
   - Each GLM fit respects chunking strategy
   - PCs are pre-cached, so no redundant extraction
   
4. **Recommended Usage**:
   - 16GB GPU: Default settings (chunk_size=None) handle most datasets
   - Larger data: Use --keep-on-cpu flag in 3dDenoisefast.py
   - Memory bottleneck is typically in GLM fitting, not PCA extraction

Key Features
------------
- Voxel-based noise pool selection via R² threshold
- Run-specific PC extraction from noise pool
- Leave-one-run-out cross-validation
- Flexible nuisance regressor framework (PCs, ICs, custom)
- Polynomial drift and other nuisance regressors maintained
- Median R² across CV folds for robust selection
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal, Callable

import numpy as np
import torch

from .glm_core import fit_glm, orthogonalize_design
from .pca import PCA
from .utils import get_device, to_tensor


@dataclass
class DenoiseResults:
    """Results from cross-validated denoising

    Attributes
    ----------
    optimal_n_components : int
        Optimal number of nuisance PCs selected by cross-validation
    xval_r2_by_n_components : np.ndarray, shape (max_components + 1,)
        Cross-validated R² for each number of components (0 to max_components)
        Index 0 = no denoising, index k = k components
    xval_r2_median_by_n_components : np.ndarray, shape (max_components + 1,)
        Median R² across CV folds for each number of components
    xval_r2_per_fold : np.ndarray, shape (n_folds, max_components + 1)
        R² for each CV fold and number of components
    noise_pool_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask: True for voxels in noise pool
    criteria_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask: True for criteria voxels (evaluation target)
    noise_pool_r2 : torch.Tensor, shape (n_voxels,)
        Initial R² values used for noise pool selection
    noise_pcs_per_run : List[torch.Tensor]
        PCA components for each run, shape per run: (n_timepoints_run, max_components)
    baseline_r2 : float
        Mean R² in criteria voxels without denoising
    optimal_r2 : float
        Mean R² in criteria voxels with optimal denoising
    improvement : float
        R² improvement (optimal_r2 - baseline_r2)
    metadata : dict
        Additional metadata (thresholds, CV strategy, etc.)
    """

    optimal_n_components: int
    xval_r2_by_n_components: np.ndarray
    xval_r2_median_by_n_components: np.ndarray
    xval_r2_per_fold: np.ndarray
    noise_pool_mask: torch.Tensor
    criteria_mask: torch.Tensor
    noise_pool_r2: torch.Tensor
    noise_pcs_per_run: List[torch.Tensor]
    baseline_r2: float
    optimal_r2: float
    improvement: float
    metadata: dict


def select_noise_pool_voxels(
    r2: torch.Tensor,
    threshold: float = 0.1,
    min_noise_voxels: int = 100,
    max_noise_fraction: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select noise pool and criteria voxels based on R² threshold

    Parameters
    ----------
    r2 : torch.Tensor, shape (n_voxels,)
        R² values from initial task model fit
    threshold : float, default=0.1
        R² threshold: voxels below this are noise pool, above are criteria
    min_noise_voxels : int, default=100
        Minimum number of voxels required in noise pool
    max_noise_fraction : float, default=0.5
        Maximum fraction of voxels allowed in noise pool

    Returns
    -------
    noise_pool_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask for noise pool voxels
    criteria_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask for criteria voxels

    Raises
    ------
    ValueError
        If noise pool has too few voxels or criteria pool is empty
    """
    n_voxels = r2.shape[0]

    # Initial threshold-based selection
    noise_pool_mask = r2 < threshold
    criteria_mask = r2 >= threshold

    n_noise = noise_pool_mask.sum().item()
    n_criteria = criteria_mask.sum().item()

    # Validate noise pool size
    if n_noise < min_noise_voxels:
        raise ValueError(
            f"Noise pool has only {n_noise} voxels (threshold={threshold}). "
            f"Need at least {min_noise_voxels}. "
            f"Try lowering threshold or using fewer voxels."
        )

    # Validate noise pool isn't too large
    max_noise_voxels = int(n_voxels * max_noise_fraction)
    if n_noise > max_noise_voxels:
        # Adjust threshold to limit noise pool size
        # Find threshold that gives max_noise_voxels
        r2_sorted, _ = torch.sort(r2)
        new_threshold = r2_sorted[max_noise_voxels].item()

        noise_pool_mask = r2 < new_threshold
        criteria_mask = r2 >= new_threshold

        n_noise = noise_pool_mask.sum().item()
        n_criteria = criteria_mask.sum().item()

        print(f"  ⚠️  Adjusted threshold from {threshold:.3f} to {new_threshold:.3f}")
        print(f"      to limit noise pool to {max_noise_fraction * 100:.0f}% of voxels")

    # Validate criteria pool exists
    if n_criteria == 0:
        raise ValueError(
            f"No criteria voxels (R² >= {threshold}). Model fit may be too poor. Check your design."
        )

    return noise_pool_mask, criteria_mask


def extract_noise_pcs_per_run(
    data: torch.Tensor,
    run_starts: List[int],
    noise_pool_mask: torch.Tensor,
    max_components: int = 20,
    variance_threshold: float = 0.95,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> List[torch.Tensor]:
    """
    Extract principal components from noise pool for each run independently

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        fMRI data
    run_starts : list of int
        Starting timepoint index for each run
    noise_pool_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask for noise pool voxels
    max_components : int, default=20
        Maximum number of PCs to extract
    variance_threshold : float, default=0.95
        Extract PCs up to this cumulative variance (within max_components)
    device : torch.device, optional
        Device for computation
    verbose : bool, default=False
        Print progress information

    Returns
    -------
    noise_pcs_per_run : list of torch.Tensor
        PCA timecourses for each run
        Each tensor has shape (n_timepoints_run, n_components_run)
        n_components_run may vary by run based on variance_threshold
    """
    device = device or get_device()
    n_runs = len(run_starts)

    # Extract noise pool data
    noise_data = data[noise_pool_mask, :]  # (n_noise_voxels, n_timepoints)

    noise_pcs_per_run = []

    for run_idx in range(n_runs):
        # Get run timepoints
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else data.shape[1]

        # Extract run data (n_noise_voxels, n_timepoints_run)
        run_data = noise_data[:, start_tp:end_tp]

        # Transpose for PCA: (n_timepoints_run, n_noise_voxels)
        # PCA will reduce voxel dimension → output is (n_timepoints_run, n_components)
        run_data_T = run_data.T

        # Fit PCA
        pca = PCA(n_components=max_components, device=device)
        pca.fit(run_data_T)

        # Determine number of components based on variance threshold
        cumvar = pca.get_explained_variance_cumsum()
        n_comp = int(torch.searchsorted(cumvar, variance_threshold).item() + 1)
        n_comp = min(n_comp, max_components)

        # Transform to get PC timecourses
        pc_timecourses = pca.transform(run_data_T)[:, :n_comp]  # (n_timepoints_run, n_comp)

        noise_pcs_per_run.append(pc_timecourses)

        if verbose:
            var_explained = cumvar[n_comp - 1].item()
            print(f"  Run {run_idx + 1}: {n_comp} PCs ({var_explained * 100:.1f}% variance)")

    return noise_pcs_per_run


def fit_glm_with_noise_pcs(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    noise_pcs: List[torch.Tensor],
    run_starts: List[int],
    n_pcs_to_use: int,
    tr: float,
    nuisance: Optional[torch.Tensor] = None,
    eval_mask: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    preload_data_to_device: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit GLM with noise PCs projected out, compute R² in eval voxels

    Memory Strategy:
    ----------------
    - PCs are lightweight timecourses (already extracted)
    - GLM fitting can chunk voxels via chunk_size parameter
    - For 16GB GPU: chunk_size=None (auto) works for most datasets
    - For larger datasets: fit_glm will auto-chunk or use explicit chunk_size

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        fMRI data (can be raw or already partially denoised)
        Can be on CPU or GPU - fit_glm handles device transfer
    design_matrix : torch.Tensor, shape (n_timepoints, n_predictors)
        Task design matrix
    noise_pcs : list of torch.Tensor
        PC timecourses per run (from extract_noise_pcs_per_run)
        Lightweight - already cached in memory
    run_starts : list of int
        Starting timepoint for each run
    n_pcs_to_use : int
        Number of PCs to use from each run (0 = no denoising)
    tr : float
        Repetition time in seconds
    nuisance : torch.Tensor, optional, shape (n_timepoints, n_nuisance)
        Other nuisance regressors (e.g., polynomial drift, motion)
    eval_mask : torch.Tensor, optional, shape (n_voxels,)
        Boolean mask for voxels to evaluate R² (default: all voxels)
    chunk_size : int, optional
        Number of voxels to process per GPU batch
        If None, fit_glm auto-detects based on available memory
    device : torch.device, optional
        Device for computation

    Returns
    -------
    betas : torch.Tensor, shape (n_voxels, n_predictors)
        GLM betas for task predictors
    r2 : torch.Tensor, shape (n_voxels,)
        R² in evaluation voxels (NaN for non-eval voxels if mask provided)
    """
    device = device or get_device()
    n_voxels, n_timepoints = data.shape
    n_runs = len(noise_pcs)

    if eval_mask is None:
        eval_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

    # Build combined nuisance matrix
    # Start with other nuisance (polort, motion, etc.)
    if nuisance is not None:
        combined_nuisance = nuisance.clone()
    else:
        combined_nuisance = torch.zeros((n_timepoints, 0), device=device)

    # Add noise PCs if requested
    if n_pcs_to_use > 0:
        # Build run-specific PC matrix
        pc_nuisance = torch.zeros((n_timepoints, 0), device=device)

        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = end_tp - start_tp

            # Get PCs for this run
            run_pcs = noise_pcs[run_idx][:, :n_pcs_to_use]  # (run_length, n_pcs_to_use)

            # Create run-specific columns (zero outside this run)
            run_pc_cols = torch.zeros((n_timepoints, n_pcs_to_use), device=device)
            run_pc_cols[start_tp:end_tp, :] = run_pcs

            # Append to nuisance matrix
            pc_nuisance = torch.cat([pc_nuisance, run_pc_cols], dim=1)

        # Combine with other nuisance
        combined_nuisance = torch.cat([combined_nuisance, pc_nuisance], dim=1)

    # Fit GLM (with memory-aware chunking)
    results = fit_glm(
        data=data,
        design=design_matrix,
        tr=tr,
        extra_regressors=combined_nuisance if combined_nuisance.shape[1] > 0 else None,
        want_residuals=False,
        chunk_size=chunk_size,
        preload_data_to_device=preload_data_to_device,
        device=device,
    )

    # Compute R² only in eval voxels
    r2_full = torch.full((n_voxels,), float("nan"), device=device)
    r2_full[eval_mask] = results.r2[eval_mask]

    return results.betas, r2_full


def cross_validate_noise_pcs(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    noise_pcs: List[torch.Tensor],
    run_starts: List[int],
    criteria_mask: torch.Tensor,
    tr: float,
    max_components: int = 20,
    nuisance: Optional[torch.Tensor] = None,
    metric: Literal["mean", "median"] = "median",
    chunk_size: Optional[int] = None,
    preload_data_to_device: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-validate noise PC denoising: train on denoised, test on raw

    The key anti-overfitting strategy:
    - Training: Fit GLM on N-1 runs WITH k PCs projected out (denoised)
    - Testing: Predict held-out run WITHOUT denoising (raw data)
    - This ensures we're improving signal recovery, not just fitting denoising

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        Raw fMRI data (never denoised for testing)
    design_matrix : torch.Tensor, shape (n_timepoints, n_predictors)
        Task design matrix
    noise_pcs : list of torch.Tensor
        PC timecourses per run from noise pool
    run_starts : list of int
        Starting timepoint for each run
    criteria_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask for criteria voxels (where we evaluate)
    max_components : int, default=20
        Maximum number of PCs to test
    nuisance : torch.Tensor, optional, shape (n_timepoints, n_nuisance)
        Other nuisance regressors (polort, motion, etc.)
    metric : {'mean', 'median'}, default='median'
        Aggregation metric across CV folds
    device : torch.device, optional
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    r2_by_n_components : np.ndarray, shape (max_components + 1,)
        Aggregated R² for each number of components (0 to max_components)
    r2_median_by_n_components : np.ndarray, shape (max_components + 1,)
        Median R² across folds for each number of components
    r2_per_fold : np.ndarray, shape (n_folds, max_components + 1)
        R² for each fold and number of components
    """
    device = device or get_device()
    n_runs = len(run_starts)
    n_voxels = data.shape[0]

    # Storage for CV results
    # Rows = folds (runs), Cols = n_components (0 to max_components)
    r2_per_fold = np.zeros((n_runs, max_components + 1))

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Cross-validating noise PC denoising (0 to {max_components} PCs)")
        print(f"{'=' * 70}")

    # Leave-one-run-out cross-validation
    for held_out_run in range(n_runs):
        if verbose:
            print(f"\nFold {held_out_run + 1}/{n_runs}: Held-out run {held_out_run + 1}")

        # Split data into train and test
        train_runs = [i for i in range(n_runs) if i != held_out_run]

        # Build train/test indices
        train_tps = []
        for run_idx in train_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else data.shape[1]
            train_tps.extend(range(start_tp, end_tp))

        test_start = run_starts[held_out_run]
        test_end = run_starts[held_out_run + 1] if held_out_run < n_runs - 1 else data.shape[1]
        test_tps = list(range(test_start, test_end))

        train_tps = torch.tensor(train_tps, device=device)
        test_tps = torch.tensor(test_tps, device=device)

        # Extract train/test data
        data_train = data[:, train_tps]
        data_test = data[:, test_tps]  # RAW (never denoised)

        design_train = design_matrix[train_tps, :]
        design_test = design_matrix[test_tps, :]

        nuisance_train = nuisance[train_tps, :] if nuisance is not None else None
        nuisance_test = nuisance[test_tps, :] if nuisance is not None else None

        # Train noise PCs (only from training runs)
        train_noise_pcs = [noise_pcs[i] for i in train_runs]
        train_run_starts = [0]  # Reindex for train data
        current_tp = 0
        for run_idx in train_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else data.shape[1]
            run_length = end_tp - start_tp
            current_tp += run_length
            if run_idx != train_runs[-1]:
                train_run_starts.append(current_tp)

        # Test each number of PCs
        for n_pcs in range(max_components + 1):
            # Train on denoised data
            if n_pcs == 0:
                # No denoising - just fit GLM
                train_results = fit_glm(
                    data=data_train,
                    design=design_train,
                    tr=tr,
                    extra_regressors=nuisance_train,
                    want_residuals=False,
                    chunk_size=chunk_size,
                    preload_data_to_device=preload_data_to_device,
                    device=device,
                )
                betas_train = train_results.betas
            else:
                # Denoise training data with n_pcs
                betas_train, _ = fit_glm_with_noise_pcs(
                    data=data_train,
                    design_matrix=design_train,
                    noise_pcs=train_noise_pcs,
                    tr=tr,
                    run_starts=train_run_starts,
                    n_pcs_to_use=n_pcs,
                    nuisance=nuisance_train,
                    chunk_size=chunk_size,
                    preload_data_to_device=preload_data_to_device,
                    device=device,
                )

            # Predict test data (RAW - no denoising)
            # Model: Y_test = X_test @ betas + nuisance_test @ nuisance_betas + error
            # But we only have task betas, so predict task signal only

            # Project out nuisance from test data (if any)
            if nuisance_test is not None and nuisance_test.shape[1] > 0:
                # Project out nuisance: Y_clean = Y - Z * (Z'Z)^-1 * Z'Y
                Q, _ = torch.linalg.qr(nuisance_test)
                data_test_clean = data_test - (Q @ (Q.T @ data_test.T)).T
            else:
                data_test_clean = data_test

            # Predict task signal
            y_pred = design_test @ betas_train.T  # (n_test_tps, n_voxels)
            y_pred = y_pred.T  # (n_voxels, n_test_tps)

            # Compute R² in criteria voxels
            y_true = data_test_clean[criteria_mask, :]
            y_pred = y_pred[criteria_mask, :]

            # R² = 1 - SS_res / SS_tot
            ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
            ss_tot = ((y_true - y_true.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            # Mean R² across criteria voxels
            r2_mean = r2.mean().item()
            r2_per_fold[held_out_run, n_pcs] = r2_mean

            if verbose and (n_pcs % 5 == 0 or n_pcs == max_components):
                print(f"  {n_pcs:2d} PCs: R² = {r2_mean:.4f}")

    # Aggregate across folds
    if metric == "median":
        r2_by_n_components = np.median(r2_per_fold, axis=0)
    else:  # mean
        r2_by_n_components = np.mean(r2_per_fold, axis=0)

    # Always compute median for reference
    r2_median_by_n_components = np.median(r2_per_fold, axis=0)

    if verbose:
        print(f"\n{'=' * 70}")
        print("Cross-validation complete")
        print(f"{'=' * 70}")
        print(f"Baseline (0 PCs): R² = {r2_by_n_components[0]:.4f}")
        best_idx = np.argmax(r2_by_n_components)
        print(f"Best ({best_idx} PCs): R² = {r2_by_n_components[best_idx]:.4f}")
        print(f"Improvement: {r2_by_n_components[best_idx] - r2_by_n_components[0]:+.4f}")

    return r2_by_n_components, r2_median_by_n_components, r2_per_fold


def fit_denoising_model(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    run_starts: List[int],
    tr: float,
    initial_r2: Optional[torch.Tensor] = None,
    r2_threshold: float = 0.1,
    max_components: int = 20,
    variance_threshold: float = 0.95,
    nuisance: Optional[torch.Tensor] = None,
    metric: Literal["mean", "median"] = "median",
    min_noise_voxels: int = 100,
    max_noise_fraction: float = 0.5,
    chunk_size: Optional[int] = None,
    preload_data_to_device: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> DenoiseResults:
    """
    Fit cross-validated denoising model

    Complete pipeline:
    1. Compute initial R² (if not provided) to select noise/criteria voxels
    2. Extract PCs from noise pool voxels per run
    3. Cross-validate to select optimal number of PCs
    4. Return results with optimal denoising parameters

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        Raw fMRI data
    design_matrix : torch.Tensor, shape (n_timepoints, n_predictors)
        Task design matrix
    run_starts : list of int
        Starting timepoint for each run
    initial_r2 : torch.Tensor, optional, shape (n_voxels,)
        Pre-computed R² for noise pool selection. If None, computed from data.
    r2_threshold : float, default=0.1
        R² threshold for noise pool selection
    max_components : int, default=20
        Maximum number of PCs to test
    variance_threshold : float, default=0.95
        Cumulative variance threshold for PC extraction
    nuisance : torch.Tensor, optional, shape (n_timepoints, n_nuisance)
        Other nuisance regressors (polort, motion, etc.)
    metric : {'mean', 'median'}, default='median'
        Aggregation metric for CV folds
    min_noise_voxels : int, default=100
        Minimum voxels required in noise pool
    max_noise_fraction : float, default=0.5
        Maximum fraction of voxels in noise pool
    device : torch.device, optional
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    results : DenoiseResults
        Complete denoising results with optimal parameters

    Examples
    --------
    >>> # Basic usage
    >>> results = fit_denoising_model(
    ...     data=fmri_data,
    ...     design_matrix=X_task,
    ...     run_starts=[0, 200, 400],
    ...     verbose=True
    ... )
    >>> print(f"Optimal: {results.optimal_n_components} PCs")
    >>> print(f"Improvement: {results.improvement:.4f}")

    >>> # Use pre-computed R² from HRF optimization
    >>> hrf_results = fit_glm_hrf_library_with_xval(...)
    >>> denoise_results = fit_denoising_model(
    ...     data=fmri_data,
    ...     design_matrix=X_task,
    ...     run_starts=[0, 200, 400],
    ...     initial_r2=hrf_results.xval_r2_best,
    ...     verbose=True
    ... )
    """
    device = device or get_device()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Cross-Validated Denoising Pipeline")
        print(f"{'=' * 70}")

    # Step 1: Compute initial R² if not provided
    if initial_r2 is None:
        if verbose:
            print("\nStep 1: Computing initial R² for noise pool selection...")

        results_init = fit_glm(
            data=data,
            design=design_matrix,
            tr=tr,
            extra_regressors=nuisance,
            want_residuals=False,
            chunk_size=chunk_size,
            preload_data_to_device=preload_data_to_device,
            device=device,
        )
        initial_r2 = results_init.r2

        if verbose:
            print(f"  Mean R²: {initial_r2.mean().item():.4f}")
    else:
        if verbose:
            print("\nStep 1: Using pre-computed R² for noise pool selection")
            print(f"  Mean R²: {initial_r2.mean().item():.4f}")

    # Step 2: Select noise pool and criteria voxels
    if verbose:
        print(f"\nStep 2: Selecting noise pool (R² < {r2_threshold}) and criteria voxels...")

    # At this point initial_r2 is guaranteed to be set (either provided or computed)
    assert initial_r2 is not None, "initial_r2 should be set by this point"
    
    noise_pool_mask, criteria_mask = select_noise_pool_voxels(
        r2=initial_r2,
        threshold=r2_threshold,
        min_noise_voxels=min_noise_voxels,
        max_noise_fraction=max_noise_fraction,
    )

    n_noise = noise_pool_mask.sum().item()
    n_criteria = criteria_mask.sum().item()

    if verbose:
        print(f"  Noise pool: {n_noise:,} voxels ({n_noise / data.shape[0] * 100:.1f}%)")
        print(f"  Criteria: {n_criteria:,} voxels ({n_criteria / data.shape[0] * 100:.1f}%)")

    # Step 3: Extract noise PCs per run
    if verbose:
        print(f"\nStep 3: Extracting PCs from noise pool (max={max_components})...")

    noise_pcs = extract_noise_pcs_per_run(
        data=data,
        run_starts=run_starts,
        noise_pool_mask=noise_pool_mask,
        max_components=max_components,
        variance_threshold=variance_threshold,
        device=device,
        verbose=verbose,
    )

    # Step 4: Cross-validate PC selection
    if verbose:
        print(f"\nStep 4: Cross-validating PC selection...")

    r2_by_n_components, r2_median_by_n_components, r2_per_fold = cross_validate_noise_pcs(
        data=data,
        design_matrix=design_matrix,
        noise_pcs=noise_pcs,
        run_starts=run_starts,
        criteria_mask=criteria_mask,
        max_components=max_components,
        tr=tr,
        nuisance=nuisance,
        metric=metric,
        chunk_size=chunk_size,
        preload_data_to_device=preload_data_to_device,
        device=device,
        verbose=verbose,
    )

    # Select optimal number of components
    optimal_n_components = int(np.argmax(r2_by_n_components))
    baseline_r2 = float(r2_by_n_components[0])
    optimal_r2 = float(r2_by_n_components[optimal_n_components])
    improvement = optimal_r2 - baseline_r2

    # Build metadata
    metadata = {
        "r2_threshold": r2_threshold,
        "max_components": max_components,
        "variance_threshold": variance_threshold,
        "metric": metric,
        "min_noise_voxels": min_noise_voxels,
        "max_noise_fraction": max_noise_fraction,
        "n_noise_voxels": n_noise,
        "n_criteria_voxels": n_criteria,
        "n_runs": len(run_starts),
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ Denoising Complete")
        print(f"{'=' * 70}")
        print(f"  Optimal: {optimal_n_components} PCs")
        print(f"  Baseline R²: {baseline_r2:.4f}")
        print(f"  Optimal R²: {optimal_r2:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        print(f"{'=' * 70}\n")

    return DenoiseResults(
        optimal_n_components=optimal_n_components,
        xval_r2_by_n_components=r2_by_n_components,
        xval_r2_median_by_n_components=r2_median_by_n_components,
        xval_r2_per_fold=r2_per_fold,
        noise_pool_mask=noise_pool_mask,
        criteria_mask=criteria_mask,
        noise_pool_r2=initial_r2,
        noise_pcs_per_run=noise_pcs,
        baseline_r2=baseline_r2,
        optimal_r2=optimal_r2,
        improvement=improvement,
        metadata=metadata,
    )
