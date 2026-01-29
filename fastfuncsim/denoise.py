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
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .glm_core import fit_glm
from .pca import PCA
from .utils import get_device
from .xval import (
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance_per_run,
)


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
    xval_r2_per_fold : np.ndarray, shape (1, max_components + 1)
        R² for each PC count from concatenated predictions.
        Note: With GLMdenoise-style concatenated predictions, this is a single-row
        array since R² is computed from all folds concatenated together.
    xval_r2_per_voxel : np.ndarray, optional, shape (n_voxels, max_components + 1)
        Per-voxel cross-validated R² for each number of components
        Stored when pcR2cutoff is used. Median is taken only over voxels
        with R² > pcR2cutoff in at least one PC count.
    noise_pool_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask: True for voxels in noise pool
    criteria_mask : torch.Tensor, shape (n_voxels,)
        Boolean mask: True for criteria voxels (evaluation target)
    pcselection_mask : torch.Tensor, optional, shape (n_voxels,)
        Boolean mask: True for voxels used in PC selection (R² > pcR2cutoff in any PC)
        None if pcR2cutoff is not used.
    noise_pool_r2 : torch.Tensor, shape (n_voxels,)
        Initial R² values used for noise pool selection
    noise_pcs_per_run : List[torch.Tensor]
        PCA components for each run, shape per run: (n_timepoints_run, max_components)
    pc_loadings_per_run : List[torch.Tensor], optional
        PC loadings (spatial weights) for each run on noise pool voxels
        Shape per run: (n_noise_voxels, n_components_run)
        None if return_loadings=False during fit
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
    xval_r2_per_voxel: Optional[np.ndarray]  # Per-voxel R² when pcR2cutoff used
    xval_r2_optimal: Optional[torch.Tensor]  # Per-voxel xval R² at optimal PC count
    xval_r2_optimal_full: Optional[torch.Tensor]  # Per-voxel xval R² at optimal PCs (all voxels)
    xval_r2_optimal_per_fold: Optional[np.ndarray]  # Per-fold xval R² at optimal PCs (all voxels)
    xval_r2_optimal_full: Optional[torch.Tensor]  # Per-voxel xval R² at optimal PCs (all voxels)
    xval_r2_optimal_per_fold: Optional[np.ndarray]  # Per-fold xval R² at optimal PCs (all voxels)
    noise_pool_mask: torch.Tensor
    criteria_mask: torch.Tensor
    pcselection_mask: Optional[torch.Tensor]  # Voxels used for PC selection (if pcR2cutoff)
    valid_voxel_mask: torch.Tensor  # Voxels that passed extreme R² filtering
    noise_pool_r2: torch.Tensor
    noise_pcs_per_run: List[torch.Tensor]
    pc_loadings_per_run: Optional[List[torch.Tensor]]
    baseline_r2: float
    optimal_r2: float
    improvement: float
    metadata: dict


def select_noise_pool_voxels(
    r2: torch.Tensor,
    threshold: float = 0.1,
    min_noise_voxels: int = 100,
    max_noise_fraction: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    max_noise_fraction : float, default=0.95
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
        # criteria_mask = r2 >= new_threshold # Leave criteria mask unchanged. We still want to evaluate on original criteria.

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
    return_loadings: bool = False,
    nuisance_per_run: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """
    Extract principal components from noise pool for each run independently

    CRITICAL: Projects out nuisance regressors from noise pool data BEFORE PCA.
    This is essential to prevent drift/nuisance from dominating the extracted components.
    Uses the EXACT SAME nuisance regressors that will be used in the GLM model.

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
    return_loadings : bool, default=False
        If True, also return PC loadings (spatial weights for noise pool voxels)
    nuisance_per_run : list of torch.Tensor, optional
        Nuisance regressors per run (e.g., polynomials, motion parameters).
        Each tensor has shape (n_timepoints_run, n_nuisance_cols).
        These will be projected out before PCA.
        CRITICAL: Must be the SAME nuisance used in the GLM model.
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
    pc_loadings_per_run : list of torch.Tensor (only if return_loadings=True)
        PC loadings (spatial weights) for each run
        Each tensor has shape (n_noise_voxels, n_components_run)
    """
    device = device or get_device()
    pca_device = device
    n_runs = len(run_starts)

    # Ensure mask on same device as data
    noise_pool_mask = noise_pool_mask.to(data.device)

    noise_pcs_per_run = []
    pc_loadings_per_run = []

    for run_idx in range(n_runs):
        # Get run timepoints
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else data.shape[1]
        run_length = end_tp - start_tp

        # Extract run data first, then apply noise pool mask (limits peak memory)
        run_data_full = data[:, start_tp:end_tp]
        run_data = run_data_full[noise_pool_mask, :]

        # CRITICAL: Project out nuisance regressors BEFORE PCA
        # This prevents drift/nuisance from dominating the extracted components
        # Uses the EXACT SAME nuisance matrices that will be used in the GLM
        if nuisance_per_run is not None and nuisance_per_run[run_idx].shape[1] > 0:
            nuisance = nuisance_per_run[run_idx].to(run_data.device)
            # Project out: Y_clean = Y - nuisance @ (nuisance'nuisance)^-1 @ nuisance' @ Y
            # Using QR for numerical stability
            q_nuisance, _ = torch.linalg.qr(nuisance)
            # run_data is (n_voxels, n_timepoints), q_nuisance is (n_timepoints, n_nuisance)
            # Project out: run_data_clean = run_data - run_data @ q_nuisance @ q_nuisance'
            run_data = run_data - (run_data @ q_nuisance) @ q_nuisance.T

        # CRITICAL: Unit-length normalize each voxel time-series BEFORE PCA
        # GLMdenoise does this to ensure PCA identifies noise structure patterns,
        # not just which voxels have higher signal magnitudes
        # This makes all voxels contribute equally to the noise pattern extraction
        voxel_norms = torch.norm(run_data, dim=1, keepdim=True)  # (n_voxels, 1)
        voxel_norms = torch.clamp(voxel_norms, min=1e-10)  # Avoid division by zero
        run_data = run_data / voxel_norms  # Each voxel time-series has unit norm

        # Transpose for PCA: (n_timepoints_run, n_noise_voxels)
        # PCA will reduce voxel dimension → output is (n_timepoints_run, n_components)
        run_data_t = None
        run_data_t = run_data.T.to(pca_device)

        # Fit PCA
        pca = PCA(n_components=max_components, device=pca_device)
        pca.fit(run_data_t)

        # Determine number of components that explain variance_threshold for reporting
        cumvar = pca.get_explained_variance_cumsum()
        n_comp_var = int(torch.searchsorted(cumvar, variance_threshold).item() + 1)
        n_comp_var = min(n_comp_var, max_components)

        # IMPORTANT: Always extract max_components PCs so all runs have same shape
        # This is required for proper stacking during cross-validation
        # The CV loop will select subsets (1, 2, 3, ... n_pcs) for testing
        pc_timecourses = pca.transform(run_data_t)[:, :max_components]
        # Move outputs back to data device to avoid GPU memory growth
        pc_timecourses = pc_timecourses.to(data.device)

        # CRITICAL: Normalize PCs to unit variance for stable GLM fitting
        # Raw PCA scores have variance = eigenvalue, which can be huge for fMRI data
        # This causes severe numerical issues when combined with normalized design matrices
        pc_std = pc_timecourses.std(dim=0, keepdim=True)
        pc_std = torch.clamp(pc_std, min=1e-10)  # Avoid division by zero
        pc_timecourses = pc_timecourses / pc_std

        noise_pcs_per_run.append(pc_timecourses)

        # Get loadings (spatial weights) if requested
        if return_loadings:
            # Loadings are the principal components (right singular vectors scaled by singular values)
            # components_ has shape (n_components, n_features) = (max_components, n_noise_voxels)
            loadings = pca.components_[:max_components, :].T  # (n_noise_voxels, max_components)
            pc_loadings_per_run.append(loadings.to(data.device))

        # Free GPU memory between runs (large noise pools)
        if pca_device.type == "cuda":
            del run_data_t
            del pca
            torch.cuda.empty_cache()

        if data.device.type == "cuda":
            del run_data_full
            del run_data
            torch.cuda.empty_cache()

        if verbose:
            var_explained = (
                cumvar[max_components - 1].item()
                if max_components <= len(cumvar)
                else cumvar[-1].item()
            )
            n_nuisance = (
                nuisance_per_run[run_idx].shape[1]
                if nuisance_per_run is not None
                else 0
            )
            nuisance_msg = f", {n_nuisance} nuisance projected" if n_nuisance > 0 else ""
            print(
                f"  Run {run_idx + 1}: {max_components} PCs ({var_explained * 100:.1f}% var{nuisance_msg})"
            )

    if return_loadings:
        return noise_pcs_per_run, pc_loadings_per_run
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
        eval_mask = torch.ones(n_voxels, dtype=torch.bool, device=data.device)
    else:
        # Ensure eval_mask is on same device as data
        eval_mask = eval_mask.to(data.device)

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
    # NOTE: verbose=False to suppress misleading "1 runs" output during CV
    results = fit_glm(
        data=data,
        design=design_matrix,
        tr=tr,
        extra_regressors=combined_nuisance if combined_nuisance.shape[1] > 0 else None,
        want_residuals=False,
        chunk_size=chunk_size,
        preload_data_to_device=preload_data_to_device,
        device=device,
        verbose=False,
    )

    # Compute R² only in eval voxels
    r2_full = torch.full((n_voxels,), float("nan"), device=data.device)
    r2_full[eval_mask] = results.r2[eval_mask]

    return results.betas, r2_full


def cross_validate_noise_pcs(
    data: torch.Tensor,
    design_matrix: Optional[torch.Tensor] = None,
    noise_pcs: List[torch.Tensor] = None,
    run_starts: List[int] = None,
    tr: float = None,
    max_components: int = 20,
    nuisance: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    metric: Literal["mean", "median"] = "median",
    cv_strategy: Union[int, float] = 1,
    n_perms: int = 100,
    chunk_size: Optional[int] = None,
    preload_data_to_device: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    designs_by_hrf: Optional[dict] = None,
    hrf_indices: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-validate noise PC denoising for ALL voxels.

    GLMdenoise-style approach:
    - Process ALL voxels (criteria determined after from R² maps)
    - Train on N-1 runs WITH k PCs in model
    - Predict held-out run using task betas only
    - Concatenate predictions across folds, compute single R² per voxel

    Supports two modes:
    - Standard: Single design_matrix for all voxels
    - Per-HRF: designs_by_hrf + hrf_indices for voxel-specific HRFs

    Memory optimization:
    - LORO (leave-one-run-out): Uses streaming stats (ss_res, sum, sum_sq)
      instead of full prediction accumulators. Saves ~1000x memory.
    - Other CV strategies: Uses full prediction accumulators per chunk

    PCs are assumed to already have nuisance projected out during extraction.

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        Raw fMRI data
    design_matrix : torch.Tensor, optional, shape (n_timepoints, n_predictors)
        Task design matrix (shared across runs). Used in standard mode.
        Mutually exclusive with designs_by_hrf.
    designs_by_hrf : dict, optional
        Dictionary mapping HRF indices to design matrices. Used in per-HRF mode.
        Mutually exclusive with design_matrix.
    hrf_indices : torch.Tensor, optional, shape (n_voxels,)
        Integer tensor mapping voxels to HRF indices. Required with designs_by_hrf.
    noise_pcs : list of torch.Tensor
        PC timecourses per run from noise pool (already nuisance-projected)
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    max_components : int, default=20
        Maximum number of PCs to test
    nuisance : torch.Tensor or list of torch.Tensor, optional
        Nuisance regressors (polort, motion, etc.)
    metric : {'mean', 'median'}, default='median'
        Aggregation metric for summary statistics
    cv_strategy : int or float, default=1
        CV strategy: 1=leave-one-out, >1=leave-k-out, <1=train fraction
    chunk_size : int, optional
        Voxel chunk size for memory management
    device : torch.device, optional
        Device for computation

    Returns
    -------
    r2_maps : np.ndarray, shape (n_voxels, max_components + 1)
        Per-voxel cross-validated R² for each PC count (0 to max_components)
    r2_summary : np.ndarray, shape (max_components + 1,)
        Summary R² (median across voxels) for each PC count
    """
    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)
    device = device or get_device()

    # Validate inputs
    if design_matrix is None and designs_by_hrf is None:
        raise ValueError("Either design_matrix or designs_by_hrf must be provided")
    if design_matrix is not None and designs_by_hrf is not None:
        raise ValueError("Cannot provide both design_matrix and designs_by_hrf")
    if designs_by_hrf is not None and hrf_indices is None:
        raise ValueError("hrf_indices required when designs_by_hrf is provided")

    # Determine mode
    per_hrf_mode = designs_by_hrf is not None

    # Detect if we can use streaming stats (LORO CV)
    # Need to check this early to determine projection device
    cv_splits_temp = generate_cv_splits(
        n_runs=len(run_starts), strategy=cv_strategy, n_perms=n_perms
    )
    is_loro_temp = cv_strategy == 1 and all(len(test) == 1 for _, test in cv_splits_temp)

    # Projection device: use GPU when available, fall back to CPU if OOM risk
    # For split-half: check if GPU has enough free memory for projections
    use_gpu_for_projections = False
    if device.type == "cuda":
        # Check available GPU memory
        gpu_free_gb = torch.cuda.mem_get_info(device)[0] / (1024**3)

        if is_loro_temp:
            # LORO: streaming stats are tiny, always use GPU
            use_gpu_for_projections = True
        else:
            # Split-half: use GPU if we have >6GB free (conservative threshold)
            # This leaves room for projection matrices + accumulators
            use_gpu_for_projections = gpu_free_gb > 6.0
            if verbose:
                if use_gpu_for_projections:
                    print(f"  GPU has {gpu_free_gb:.1f} GB free → using GPU for projections")
                else:
                    print(f"  GPU has {gpu_free_gb:.1f} GB free → using CPU for projections (need >6GB)")

    proj_device = device if use_gpu_for_projections else torch.device("cpu")

    n_runs = len(run_starts)
    n_voxels = data.shape[0]
    n_timepoints = data.shape[1]

    # Convert nuisance to list format if provided as concatenated tensor
    nuisance_per_run: Optional[list[torch.Tensor]] = None
    if nuisance is not None:
        if isinstance(nuisance, list):
            nuisance_per_run = nuisance
        else:
            # Split concatenated nuisance by run
            nuisance_per_run = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                nuisance_per_run.append(nuisance[start_tp:end_tp, :])

    # Generate CV splits based on strategy
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    # Describe CV strategy
    if cv_strategy == 1:
        cv_desc = f"Leave-one-run-out ({n_splits} folds)"
    elif isinstance(cv_strategy, int) and cv_strategy > 1:
        cv_desc = f"Leave-{cv_strategy}-runs-out ({n_splits} folds)"
    else:
        cv_desc = f"{cv_strategy:.0%} train split ({n_splits} folds)"

    print(f"\n{'=' * 70}")
    print(f"Cross-validating noise PC denoising (0 to {max_components} PCs)")
    print(f"Strategy: {cv_desc}")
    print(f"Total voxels: {n_voxels:,} (processing ALL voxels)")
    print(f"Method: GLMdenoise-style concatenated predictions")
    print(f"{'=' * 70}")

    # =========================================================================
    # GLMdenoise-style: Accumulate predictions across folds, compute R² once
    # =========================================================================
    # For each PC count (0-20):
    # 1. Train on N-1 runs WITH k PCs in model
    # 2. Predict held-out run using task betas
    # 3. Accumulate predictions across all folds
    # 4. Compute single R² from concatenated predictions vs actual data

    # Detect if we can use streaming stats (LORO CV)
    is_loro = cv_strategy == 1 and all(len(test) == 1 for _, test in cv_splits)

    # Determine chunk size for voxel processing
    # Account for both n_voxels AND n_timepoints to avoid OOM
    # Main memory consumers: chunk_data (voxels × timepoints), projections, accumulators
    if chunk_size is not None:
        voxel_chunk_size = chunk_size
    elif is_loro:
        # Streaming stats: smaller accumulators + GPU projections = can use larger chunks
        # Memory: chunk_data + projected train/test data (held during projection before concat)
        # During projection we hold multiple copies, so be conservative
        # Target: 0.6GB for chunk_data to leave ~1-2GB headroom for projection operations
        target_chunk_memory_gb = 0.3
        bytes_per_voxel = n_timepoints * 4  # float32
        max_voxels_from_memory = int((target_chunk_memory_gb * 1024**3) / bytes_per_voxel)
        voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 10000), 42000)
    else:
        # Full accumulator: much more memory for storing full predictions per PC
        # Memory: chunk_data + (voxels × timepoints × 4 bytes × n_PCs)
        # CRITICAL: Full accumulators for split-half are HUGE (voxels × timepoints × n_PCs)
        # Must keep on CPU and use conservative chunk sizes
        if use_gpu_for_projections:
            # GPU projections but CPU accumulators: moderate chunks
            # Accumulators: voxels × timepoints × (max_components+1) × 4 bytes
            # For 1.9M voxels × 1065 TPs × 21 PCs = 169 GB (must chunk!)
            # Target: 50k voxels = ~4.5GB accumulators (fits in RAM)
            target_chunk_memory_gb = 0.8
            bytes_per_voxel = n_timepoints * 4
            max_voxels_from_memory = int((target_chunk_memory_gb * 1024**3) / bytes_per_voxel)
            # Cap at 50k for split-half to keep accumulator size reasonable
            voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 10000), 50000)
        else:
            # CPU projections: smaller chunks to avoid memory issues
            target_chunk_memory_gb = 0.5
            bytes_per_voxel = n_timepoints * 4
            max_voxels_from_memory = int((target_chunk_memory_gb * 1024**3) / bytes_per_voxel)
            voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 5000), 20000)

    # Move data to CPU for memory efficiency
    data_cpu = data.to(proj_device)
    design_matrix_cpu = design_matrix.to(proj_device)

    # Output: R² maps for all voxels and all PC counts
    r2_maps = np.zeros((n_voxels, max_components + 1), dtype=np.float32)

    if is_loro:
        print(f"\nProcessing {n_voxels:,} voxels in chunks of {voxel_chunk_size:,}")
        print(f"Memory strategy: LORO streaming stats (minimal memory)")
        print(f"Projection device: {proj_device} (GPU acceleration enabled)")
    else:
        print(f"\nProcessing {n_voxels:,} voxels in chunks of {voxel_chunk_size:,}")
        print(f"Memory strategy: accumulate predictions per chunk, compute R², discard")
        print(f"Projection device: {proj_device} (CPU to save GPU memory)")

    n_chunks = (n_voxels + voxel_chunk_size - 1) // voxel_chunk_size

    # Progress bar for chunks
    try:
        from tqdm import tqdm

        chunk_iter = tqdm(range(n_chunks), desc="Denoising CV", unit="chunk")
    except ImportError:
        chunk_iter = range(n_chunks)

    # Process voxels in chunks to manage memory
    for chunk_idx in chunk_iter:
        chunk_start = chunk_idx * voxel_chunk_size
        chunk_end = min(chunk_start + voxel_chunk_size, n_voxels)
        chunk_size_actual = chunk_end - chunk_start

        # Get actual data for this chunk
        chunk_data_cpu = data_cpu[chunk_start:chunk_end, :]

        if is_loro:
            # Streaming stats: accumulate ss_res, sum, sum_sq for each PC count
            # Keep on GPU - they're tiny (~24 bytes/voxel) and avoid 1000s of transfers
            ss_res_by_pc = [
                torch.zeros(chunk_size_actual, dtype=torch.float64, device=device)
                for _ in range(max_components + 1)
            ]
            sum_actual_by_pc = [
                torch.zeros(chunk_size_actual, dtype=torch.float64, device=device)
                for _ in range(max_components + 1)
            ]
            sum_sq_actual_by_pc = [
                torch.zeros(chunk_size_actual, dtype=torch.float64, device=device)
                for _ in range(max_components + 1)
            ]
        else:
            # Full accumulator mode for non-LORO CV
            # CRITICAL: Full accumulators are HUGE (chunk_voxels × timepoints × n_PCs)
            # For 50k voxels × 1065 TPs × 21 PCs = 4.5GB
            # Must use CPU to avoid GPU OOM
            accumulator_device = torch.device("cpu")
            pred_by_pc_chunk = [
                torch.zeros(chunk_size_actual, n_timepoints, dtype=torch.float32, device=accumulator_device)
                for _ in range(max_components + 1)
            ]
            actual_projected_chunk = torch.zeros(
                chunk_size_actual, n_timepoints, dtype=torch.float32, device=accumulator_device
            )

        # Cross-validation loop: accumulate predictions or stats for this chunk
        for _, (train_runs, test_runs) in enumerate(cv_splits):
            # Build train/test timepoint indices
            train_tps = []
            for run_idx in train_runs:
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                train_tps.extend(range(start_tp, end_tp))

            test_tps = []
            for run_idx in test_runs:
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                test_tps.extend(range(start_tp, end_tp))

            train_tps_t = torch.tensor(train_tps, device=proj_device)
            test_tps_t = torch.tensor(test_tps, device=proj_device)

            # Extract train/test data for THIS CHUNK
            chunk_data_train = chunk_data_cpu[:, train_tps_t]
            chunk_data_test = chunk_data_cpu[:, test_tps_t]

            design_train = design_matrix_cpu[train_tps_t, :]
            design_test = design_matrix_cpu[test_tps_t, :]

            # ================================================================
            # PROJECT-FIRST APPROACH (GLMdenoise-style)
            # Project nuisance from data and design per run
            # ================================================================
            if nuisance_per_run is not None:
                # Project out nuisance from TRAINING data and design per run
                train_data_projected_runs = []
                train_design_projected_runs = []

                for run_idx in train_runs:
                    # Get this run's slice from the concatenated train data
                    run_start_in_train = 0
                    for prev_run in train_runs:
                        if prev_run == run_idx:
                            break
                        prev_start = run_starts[prev_run]
                        prev_end = (
                            run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                        )
                        run_start_in_train += prev_end - prev_start

                    run_start_global = run_starts[run_idx]
                    run_end_global = (
                        run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                    )
                    run_length = run_end_global - run_start_global
                    run_end_in_train = run_start_in_train + run_length

                    # Get this run's data and design from the train split
                    run_data = chunk_data_train[:, run_start_in_train:run_end_in_train]
                    run_design = design_train[run_start_in_train:run_end_in_train, :]
                    run_nuisance = nuisance_per_run[run_idx].to(proj_device)

                    if run_nuisance.shape[1] > 0:
                        # Compute projection matrix: P_perp = I - X(X'X)^-1 X'
                        XtX = run_nuisance.T @ run_nuisance
                        XtX_inv = torch.linalg.inv(
                            XtX + 1e-6 * torch.eye(XtX.shape[0], device=proj_device)
                        )
                        P_nuisance = run_nuisance @ XtX_inv @ run_nuisance.T
                        projection = torch.eye(run_length, device=proj_device) - P_nuisance

                        # Project data and design
                        run_data_proj = (projection @ run_data.T).T
                        run_design_proj = projection @ run_design
                    else:
                        run_data_proj = run_data
                        run_design_proj = run_design

                    train_data_projected_runs.append(run_data_proj)
                    train_design_projected_runs.append(run_design_proj)

                # Concatenate projected training data/design
                data_train_projected = torch.cat(train_data_projected_runs, dim=1)
                design_train_projected = torch.cat(train_design_projected_runs, dim=0)

                # Free intermediate projection results to reduce fragmentation
                del train_data_projected_runs, train_design_projected_runs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # Project out nuisance from TEST data and design
                test_data_projected_runs = []
                test_design_projected_runs = []

                for run_idx in test_runs:
                    run_start_global = run_starts[run_idx]
                    run_end_global = (
                        run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                    )
                    run_length = run_end_global - run_start_global

                    run_start_in_test = 0
                    for prev_run in test_runs:
                        if prev_run == run_idx:
                            break
                        prev_start = run_starts[prev_run]
                        prev_end = (
                            run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                        )
                        run_start_in_test += prev_end - prev_start

                    run_end_in_test = run_start_in_test + run_length
                    run_test_data = chunk_data_test[:, run_start_in_test:run_end_in_test]
                    run_test_design = design_test[run_start_in_test:run_end_in_test, :]

                    nuisance_test_run = nuisance_per_run[run_idx].to(proj_device)

                    if nuisance_test_run.shape[1] > 0:
                        XtX_test = nuisance_test_run.T @ nuisance_test_run
                        XtX_test_inv = torch.linalg.inv(
                            XtX_test + 1e-6 * torch.eye(XtX_test.shape[0], device=proj_device)
                        )
                        P_nuisance_test = nuisance_test_run @ XtX_test_inv @ nuisance_test_run.T
                        projection_test = (
                            torch.eye(run_length, device=proj_device) - P_nuisance_test
                        )

                        run_test_data_proj = (projection_test @ run_test_data.T).T
                        run_test_design_proj = projection_test @ run_test_design
                    else:
                        run_test_data_proj = run_test_data
                        run_test_design_proj = run_test_design

                    test_data_projected_runs.append(run_test_data_proj)
                    test_design_projected_runs.append(run_test_design_proj)

                # Concatenate projected test data/design
                data_test_projected = torch.cat(test_data_projected_runs, dim=1)
                design_test_projected = torch.cat(test_design_projected_runs, dim=0)

                # Free intermediate projection results
                del test_data_projected_runs, test_design_projected_runs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                # No nuisance - no projection needed
                data_train_projected = chunk_data_train
                design_train_projected = design_train
                data_test_projected = chunk_data_test
                design_test_projected = design_test

            # Collect training run PCs (already nuisance-projected during extraction)
            train_noise_pcs_list = [noise_pcs[run_idx].to(device) for run_idx in train_runs]

            # ================================================================
            # Pre-compute pseudo-inverses for ALL PC counts
            # ================================================================
            n_task_regs = design_train_projected.shape[1]
            pinv_task_list = []

            for n_pcs in range(max_components + 1):
                components = [design_train_projected.to(device)]

                if n_pcs > 0:
                    # Build zero-padded PC matrix
                    n_train_runs = len(train_noise_pcs_list)
                    pc_padded_blocks = []

                    for block_idx, pcs_run in enumerate(train_noise_pcs_list):
                        run_length = pcs_run.shape[0]
                        n_available = pcs_run.shape[1]
                        n_use = min(n_pcs, n_available)

                        padded = torch.zeros((run_length, n_train_runs * n_pcs), device=device)
                        start_col = block_idx * n_pcs
                        end_col = start_col + n_use
                        padded[:, start_col:end_col] = pcs_run[:, :n_use]

                        pc_padded_blocks.append(padded)

                    pc_combined = torch.cat(pc_padded_blocks, dim=0)
                    components.append(pc_combined)

                x_full = torch.cat(components, dim=1)

                # Compute pseudo-inverse for task betas only
                xtx = x_full.T @ x_full
                xtx_inv = torch.linalg.pinv(xtx)
                pinv_full = xtx_inv @ x_full.T
                pinv_task = pinv_full[:n_task_regs, :]
                pinv_task_list.append(pinv_task)

            design_test_gpu = design_test_projected.to(device)

            # ================================================================
            # Fit and predict for ALL PC counts
            # ================================================================
            # Get projected test data for R² computation (predictions are in projected space)
            if nuisance_per_run is not None:
                test_actual_projected = data_test_projected
            else:
                test_actual_projected = chunk_data_test

            for n_pcs in range(max_components + 1):
                pinv_task = pinv_task_list[n_pcs]

                # Fit: betas = pinv @ Y
                data_train_gpu = data_train_projected.to(device)
                betas = (pinv_task @ data_train_gpu.T).T  # (chunk, n_task_regs)

                # Predict: y_pred = X @ betas
                y_pred = (design_test_gpu @ betas.T).T  # (chunk, n_test_tps)

                if is_loro:
                    # Streaming stats: accumulate ss_res, sum_actual, sum_sq_actual
                    # Keep on GPU - accumulators are on GPU, avoid thousands of transfers
                    test_actual_f64 = test_actual_projected.double()
                    y_pred_f64 = y_pred.double()

                    residuals = test_actual_f64 - y_pred_f64
                    ss_res_fold = (residuals**2).sum(dim=1)
                    sum_fold = test_actual_f64.sum(dim=1)
                    sum_sq_fold = (test_actual_f64**2).sum(dim=1)

                    ss_res_by_pc[n_pcs] += ss_res_fold
                    sum_actual_by_pc[n_pcs] += sum_fold
                    sum_sq_actual_by_pc[n_pcs] += sum_sq_fold
                else:
                    # Full accumulator mode: store predictions
                    # Move to accumulator device (GPU if doing GPU projections, otherwise CPU)
                    pred_by_pc_chunk[n_pcs][:, test_tps] = y_pred.to(pred_by_pc_chunk[n_pcs].device)

            # For non-LORO mode, store projected actuals
            if not is_loro:
                if nuisance_per_run is not None:
                    actual_projected_chunk[:, test_tps] = data_test_projected.to(actual_projected_chunk.device)
                else:
                    actual_projected_chunk[:, test_tps] = chunk_data_test.to(actual_projected_chunk.device)

            # Free GPU memory for this fold
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ================================================================
        # Compute R² for this chunk
        # ================================================================
        if is_loro:
            # Streaming stats: compute R² from accumulated stats (on GPU)
            # In LORO, each voxel sees all timepoints exactly once
            for n_pcs in range(max_components + 1):
                ss_res = ss_res_by_pc[n_pcs]
                sum_act = sum_actual_by_pc[n_pcs]
                sum_sq_act = sum_sq_actual_by_pc[n_pcs]

                # Compute ss_tot from streaming stats: ss_tot = sum_sq - sum^2 / n
                mean_actual = sum_act / n_timepoints
                ss_tot = sum_sq_act - n_timepoints * (mean_actual**2)

                # R² = 1 - ss_res / ss_tot (all on GPU, transfer only final result)
                r2 = (1.0 - ss_res / (ss_tot + 1e-10)).float()
                r2_maps[chunk_start:chunk_end, n_pcs] = r2.cpu().numpy()

            # Free chunk memory
            del ss_res_by_pc, sum_actual_by_pc, sum_sq_actual_by_pc, chunk_data_cpu
        else:
            # Full accumulator mode: compute R² from full predictions
            for n_pcs in range(max_components + 1):
                pred = pred_by_pc_chunk[n_pcs]  # (chunk_size, n_timepoints)
                actual = actual_projected_chunk  # (chunk_size, n_timepoints) - PROJECTED data

                # Compute R² per voxel (works on both CPU and GPU)
                ss_res = ((actual - pred) ** 2).sum(dim=1)
                ss_tot = ((actual - actual.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))

                # Move to CPU if on GPU before converting to numpy
                r2_maps[chunk_start:chunk_end, n_pcs] = r2.cpu().numpy() if r2.is_cuda else r2.numpy()

            # Free chunk memory
            del pred_by_pc_chunk, chunk_data_cpu, actual_projected_chunk

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ================================================================
    # Compute summary statistics
    # ================================================================
    print("\n" + "=" * 70)
    print("R² computation complete")
    print("=" * 70)

    # Summary: median R² across ALL voxels for each PC count
    r2_summary = np.median(r2_maps, axis=0)

    print(f"  Baseline (0 PCs): median R² = {r2_summary[0]:.4f}")
    best_idx = int(np.argmax(r2_summary))
    print(f"  Best ({best_idx} PCs): median R² = {r2_summary[best_idx]:.4f}")
    print(f"  Improvement: {r2_summary[best_idx] - r2_summary[0]:+.4f}")

    # Count voxels with positive R² in any PC count (for criteria selection info)
    n_positive_any = np.sum(np.any(r2_maps > 0, axis=1))
    print(f"  Voxels with R² > 0 in any PC count: {n_positive_any:,} / {n_voxels:,}")

    return r2_maps, r2_summary


def select_optimal_pcs(
    r2_maps: np.ndarray,
    threshold: float = 0.0,
    metric: str = "median",
) -> Tuple[int, np.ndarray]:
    """
    Select optimal PC count using GLMdenoise criteria.

    GLMdenoise Step 7: Select number of noise regressors
    1. Identify criteria voxels: any voxel where R² > threshold in ANY PC count
    2. Compute median R² of criteria voxels for each PC count
    3. Select PC count with maximum median R²

    Parameters
    ----------
    r2_maps : np.ndarray, shape (n_voxels, n_pc_counts)
        Per-voxel cross-validated R² for each PC count
    threshold : float, default=0.0
        R² threshold for criteria voxel selection
    metric : str, default='median'
        Aggregation metric ('median' or 'mean')

    Returns
    -------
    optimal_n_pcs : int
        Best number of PCs
    criteria_mask : np.ndarray, shape (n_voxels,)
        Boolean mask of criteria voxels (R² > threshold in any PC count)
    """
    # Criteria: voxels above threshold in ANY PC count
    criteria_mask = np.any(r2_maps > threshold, axis=1)
    n_criteria = np.sum(criteria_mask)

    print(f"\nSelecting optimal PC count:")
    print(f"  Criteria voxels (R² > {threshold} in any PC): {n_criteria:,} / {r2_maps.shape[0]:,}")

    if n_criteria == 0:
        print("  WARNING: No voxels meet criteria! Using all voxels.")
        criteria_mask = np.ones(r2_maps.shape[0], dtype=bool)
        n_criteria = r2_maps.shape[0]

    # Aggregate R² of criteria voxels per PC count
    r2_criteria = r2_maps[criteria_mask, :]
    if metric == "median":
        r2_agg = np.median(r2_criteria, axis=0)
    else:
        r2_agg = np.mean(r2_criteria, axis=0)

    optimal_n_pcs = int(np.argmax(r2_agg))

    print(f"  Baseline (0 PCs): {metric} R² = {r2_agg[0]:.4f}")
    print(f"  Best ({optimal_n_pcs} PCs): {metric} R² = {r2_agg[optimal_n_pcs]:.4f}")
    print(f"  Improvement: {r2_agg[optimal_n_pcs] - r2_agg[0]:+.4f}")

    return optimal_n_pcs, criteria_mask


def compute_xval_r2_optimal_full(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    noise_pcs: List[torch.Tensor],
    run_starts: List[int],
    optimal_n_components: int,
    nuisance: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    cv_strategy: Union[int, float] = 1,
    n_perms: int = 100,
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Compute per-voxel cross-validated R² at the optimal PC count for ALL voxels.

    Uses GLMdenoise-style concatenated predictions:
    - Accumulates predictions across all folds
    - Computes single R² from concatenated predictions vs actual timeseries

    Returns
    -------
    r2_all : torch.Tensor, shape (n_voxels,)
        Cross-validated R² from concatenated predictions for all voxels
    r2_per_fold_all : np.ndarray, shape (n_folds, n_voxels)
        Legacy per-fold R² output (for compatibility; not meaningful with concatenated approach)
    """
    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)
    device = device or get_device()
    # CRITICAL: Use CPU for all data operations to avoid GPU OOM on large datasets
    # Only move small chunks to GPU for final GLM computation
    proj_device = torch.device("cpu")

    # Move data to CPU if it's on GPU (avoids OOM during indexing/concatenation)
    data_cpu = data.to(proj_device)
    design_matrix_cpu = design_matrix.to(proj_device)

    n_runs = len(run_starts)
    n_timepoints = data_cpu.shape[1]
    n_task_regs = design_matrix_cpu.shape[1]
    n_voxels = data_cpu.shape[0]

    # Determine chunk size for voxel-wise projection (to avoid OOM even on CPU)
    voxel_chunk_size = chunk_size or 50000

    # Convert nuisance to list format if needed
    nuisance_per_run: Optional[list[torch.Tensor]] = None
    if nuisance is not None:
        if isinstance(nuisance, list):
            nuisance_per_run = nuisance
        else:
            nuisance_per_run = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                nuisance_per_run.append(nuisance[start_tp:end_tp, :])

    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    # Initialize prediction accumulator (on CPU to save memory)
    predictions = torch.zeros((n_voxels, n_timepoints), dtype=torch.float32, device=proj_device)

    if verbose:
        print(
            f"  Processing {n_voxels:,} voxels in chunks of {voxel_chunk_size:,} (CPU→GPU streaming)"
        )

    for fold_idx, (train_runs, test_runs) in enumerate(cv_splits):
        # Build train/test indices
        train_tps = []
        for run_idx in train_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            train_tps.extend(range(start_tp, end_tp))

        test_tps = []
        for run_idx in test_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            test_tps.extend(range(start_tp, end_tp))

        train_tps_t = torch.tensor(train_tps, device=proj_device)
        test_tps_t = torch.tensor(test_tps, device=proj_device)

        data_train = data_cpu[:, train_tps_t]
        data_test = data_cpu[:, test_tps_t]
        design_train = design_matrix_cpu[train_tps_t, :]
        design_test = design_matrix_cpu[test_tps_t, :]

        # Project out nuisance per run (project-first)
        if nuisance_per_run is not None:
            train_data_projected_runs = []
            train_design_projected_runs = []

            for run_idx in train_runs:
                run_start_in_train = 0
                for prev_run in train_runs:
                    if prev_run == run_idx:
                        break
                    prev_start = run_starts[prev_run]
                    prev_end = run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                    run_start_in_train += prev_end - prev_start

                run_start_global = run_starts[run_idx]
                run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = run_end_global - run_start_global
                run_end_in_train = run_start_in_train + run_length

                run_design = design_train[run_start_in_train:run_end_in_train, :].to(proj_device)
                run_nuisance = nuisance_per_run[run_idx].to(proj_device)

                if run_nuisance.shape[1] > 0:
                    # Compute projection matrix on CPU
                    xtx = run_nuisance.T @ run_nuisance
                    xtx_inv = torch.linalg.inv(
                        xtx + 1e-6 * torch.eye(xtx.shape[0], device=proj_device)
                    )
                    p_nuisance = run_nuisance @ xtx_inv @ run_nuisance.T
                    projection = torch.eye(run_length, device=proj_device) - p_nuisance

                    # Project design matrix (small)
                    run_design_proj = projection @ run_design

                    # Project data in chunks to avoid OOM (projection @ data.T creates huge temporaries)
                    run_data_proj_chunks = []
                    for vox_start in range(0, n_voxels, voxel_chunk_size):
                        vox_end = min(vox_start + voxel_chunk_size, n_voxels)
                        chunk = data_train[
                            vox_start:vox_end, run_start_in_train:run_end_in_train
                        ].to(proj_device)
                        chunk_proj = (projection @ chunk.T).T  # (n_voxels_chunk, run_length)
                        run_data_proj_chunks.append(chunk_proj)
                    run_data_proj = torch.cat(run_data_proj_chunks, dim=0)
                else:
                    run_data_proj = data_train[:, run_start_in_train:run_end_in_train].to(
                        proj_device
                    )
                    run_design_proj = run_design

                train_data_projected_runs.append(run_data_proj)
                train_design_projected_runs.append(run_design_proj)

            data_train_projected = torch.cat(train_data_projected_runs, dim=1)
            design_train_projected = torch.cat(train_design_projected_runs, dim=0)

            test_data_projected_runs = []
            test_design_projected_runs = []

            for run_idx in test_runs:
                run_start_global = run_starts[run_idx]
                run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = run_end_global - run_start_global

                run_start_in_test = 0
                for prev_run in test_runs:
                    if prev_run == run_idx:
                        break
                    prev_start = run_starts[prev_run]
                    prev_end = run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                    run_start_in_test += prev_end - prev_start

                run_end_in_test = run_start_in_test + run_length
                run_test_design = design_test[run_start_in_test:run_end_in_test, :].to(proj_device)
                run_nuisance = nuisance_per_run[run_idx].to(proj_device)

                if run_nuisance.shape[1] > 0:
                    # Compute projection matrix on CPU
                    xtx_test = run_nuisance.T @ run_nuisance
                    xtx_test_inv = torch.linalg.inv(
                        xtx_test + 1e-6 * torch.eye(xtx_test.shape[0], device=proj_device)
                    )
                    p_nuisance_test = run_nuisance @ xtx_test_inv @ run_nuisance.T
                    projection_test = torch.eye(run_length, device=proj_device) - p_nuisance_test

                    # Project design matrix (small)
                    run_test_design_proj = projection_test @ run_test_design

                    # Project data in chunks to avoid OOM
                    run_test_data_proj_chunks = []
                    for vox_start in range(0, n_voxels, voxel_chunk_size):
                        vox_end = min(vox_start + voxel_chunk_size, n_voxels)
                        chunk = data_test[vox_start:vox_end, run_start_in_test:run_end_in_test].to(
                            proj_device
                        )
                        chunk_proj = (projection_test @ chunk.T).T
                        run_test_data_proj_chunks.append(chunk_proj)
                    run_test_data_proj = torch.cat(run_test_data_proj_chunks, dim=0)
                else:
                    run_test_data_proj = data_test[:, run_start_in_test:run_end_in_test].to(
                        proj_device
                    )
                    run_test_design_proj = run_test_design

                test_data_projected_runs.append(run_test_data_proj)
                test_design_projected_runs.append(run_test_design_proj)

            data_test_projected = torch.cat(test_data_projected_runs, dim=1)
            design_test_projected = torch.cat(test_design_projected_runs, dim=0)
        else:
            # No nuisance - move data to CPU if needed
            data_train_projected = data_train.to(proj_device)
            design_train_projected = design_train.to(proj_device)
            data_test_projected = data_test.to(proj_device)
            design_test_projected = design_test.to(proj_device)

        # Project nuisance from PCs for training runs
        train_noise_pcs_projected = []
        for run_idx in train_runs:
            run_start_global = run_starts[run_idx]
            run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = run_end_global - run_start_global

            pcs_this_run = noise_pcs[run_idx]
            if nuisance_per_run is not None:
                run_nuisance = nuisance_per_run[run_idx].to(device)
                if run_nuisance.shape[1] > 0:
                    xtx = run_nuisance.T @ run_nuisance
                    xtx_inv = torch.linalg.inv(xtx + 1e-6 * torch.eye(xtx.shape[0], device=device))
                    p_nuisance = run_nuisance @ xtx_inv @ run_nuisance.T
                    projection = torch.eye(run_length, device=device) - p_nuisance
                    pcs_projected = projection @ pcs_this_run.to(device)
                else:
                    pcs_projected = pcs_this_run.to(device)
            else:
                pcs_projected = pcs_this_run.to(device)

            train_noise_pcs_projected.append(pcs_projected)

        # Build design matrix with optimal PCs
        components = [design_train_projected.to(device)]
        if optimal_n_components > 0:
            n_train_runs = len(train_noise_pcs_projected)
            pc_padded_blocks = []
            for block_idx, pcs_run in enumerate(train_noise_pcs_projected):
                run_length = pcs_run.shape[0]
                n_available = pcs_run.shape[1]
                n_use = min(optimal_n_components, n_available)
                padded = torch.zeros(
                    (run_length, n_train_runs * optimal_n_components), device=device
                )
                start_col = block_idx * optimal_n_components
                end_col = start_col + n_use
                padded[:, start_col:end_col] = pcs_run[:, :n_use]
                pc_padded_blocks.append(padded)
            pc_combined = torch.cat(pc_padded_blocks, dim=0)
            components.append(pc_combined)

        x_full = torch.cat(components, dim=1)
        xtx = x_full.T @ x_full
        xtx_inv = torch.linalg.pinv(xtx)
        pinv_full = xtx_inv @ x_full.T
        pinv_task = pinv_full[:n_task_regs, :]
        design_test_gpu = design_test_projected.to(device)

        # Process voxels in chunks (data stays on CPU, transfer to GPU per chunk)
        # Accumulate predictions for test timepoints
        test_tps_list = test_tps  # Already a list from above

        for chunk_start in range(0, n_voxels, voxel_chunk_size):
            chunk_end = min(chunk_start + voxel_chunk_size, n_voxels)
            chunk_train = data_train_projected[chunk_start:chunk_end, :].to(device)

            # Fit model and predict test data
            betas = (pinv_task @ chunk_train.T).T
            y_pred = (design_test_gpu @ betas.T).T  # (chunk, n_test_tps)

            # Accumulate predictions for this fold's test timepoints
            predictions[chunk_start:chunk_end, test_tps_list] = y_pred.cpu()

            # Free GPU memory immediately after each chunk
            del chunk_train, betas, y_pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_splits} complete")

        # Clean up fold-level data to prevent accumulation
        del data_train_projected, data_test_projected, design_train_projected, design_test_projected
        del x_full, xtx, xtx_inv, pinv_full, pinv_task, design_test_gpu
        del train_noise_pcs_projected
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute R² from accumulated predictions (concatenated across all folds)
    if verbose:
        print("  Computing R² from concatenated predictions...")

    ss_res = ((data_cpu - predictions) ** 2).sum(dim=1)
    ss_tot = ((data_cpu - data_cpu.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    r2_all = 1 - (ss_res / (ss_tot + 1e-10))

    # Legacy per-fold output (for compatibility; duplicate the single R² value)
    r2_per_fold_all = np.tile(r2_all.numpy(), (n_splits, 1))

    return r2_all, r2_per_fold_all


def fit_denoising_model(
    data: torch.Tensor,
    design_matrix: Optional[torch.Tensor] = None,
    run_starts: List[int] = None,
    tr: float = None,
    initial_r2: Optional[torch.Tensor] = None,
    r2_threshold: float = 0.05,
    intensity_mask: Optional[torch.Tensor] = None,
    max_components: int = 20,
    variance_threshold: float = 0.95,
    nuisance: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    metric: Literal["mean", "median"] = "median",
    min_noise_voxels: int = 100,
    max_noise_fraction: float = 0.95,
    chunk_size: Optional[int] = None,
    preload_data_to_device: bool = True,
    return_loadings: bool = False,
    polort: Optional[int] = 2,
    pcstop: float = 1.05,
    pcR2cutoff: Optional[float] = None,
    cv_strategy: Union[int, float] = 1,
    n_perms: int = 100,
    r2_method: str = "auto",
    device: Optional[torch.device] = None,
    verbose: bool = False,
    designs_by_hrf: Optional[dict] = None,
    hrf_indices: Optional[torch.Tensor] = None,
) -> DenoiseResults:
    """
    Fit cross-validated denoising model

    Complete pipeline:
    1. Compute initial R² (if not provided) to select noise/criteria voxels
       - Standard mode: Use single design_matrix for all voxels
       - Per-HRF mode: Use designs_by_hrf + hrf_indices for per-voxel HRFs
    2. Extract PCs from unified noise pool (after projecting out polynomial drift)
    3. Cross-validate to select optimal number of PCs
       - Standard mode: Single design matrix
       - Per-HRF mode: Each voxel uses its HRF-specific design + shared PCs
    4. Return results with optimal denoising parameters

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        Raw fMRI data
    design_matrix : torch.Tensor, optional, shape (n_timepoints, n_predictors)
        Task design matrix (for standard single-HRF mode).
        Mutually exclusive with designs_by_hrf.
    designs_by_hrf : dict, optional
        Dictionary mapping HRF indices to design matrices.
        Format: {hrf_idx: design_matrix}, where each design_matrix has
        shape (n_timepoints, n_predictors). Used for per-voxel HRF mode.
        Mutually exclusive with design_matrix.
    hrf_indices : torch.Tensor, optional, shape (n_voxels,)
        Integer tensor mapping each voxel to its HRF index.
        Required when designs_by_hrf is provided.
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    initial_r2 : torch.Tensor, optional, shape (n_voxels,)
        Pre-computed R² for noise pool selection. If None, computed from data.
    r2_threshold : float, default=0.05
        R² threshold for noise pool selection
    intensity_mask : torch.Tensor, optional, shape (n_voxels,)
        Boolean mask for signal intensity threshold (brainthresh). If provided,
        noise pool selection is restricted to voxels where intensity_mask is True.
        This excludes low-intensity background voxels from the noise pool.
    max_components : int, default=20
        Maximum number of PCs to test
    variance_threshold : float, default=0.95
        Cumulative variance threshold for PC extraction
    nuisance : torch.Tensor, optional, shape (n_timepoints, n_nuisance)
        Other nuisance regressors (polort, motion, etc.)
    metric : {'mean', 'median'}, default='median'
        Aggregation metric for CV folds
    pcstop : float, default=1.05
        Early stopping threshold for PC selection (GLMdenoise-style).
        - If >= 1.0: Stop when performance is within (pcstop-1)*100% of max.
          E.g., 1.05 means stop when within 5% of max (default, more robust).
        - If < 0: Use exactly abs(pcstop) PCs (user override).
        - If == 1.0: Use pure argmax (pick maximum).
    pcR2cutoff : float, optional
        R² cutoff for PC count selection. If provided, only voxels that achieve
        R² > pcR2cutoff in at least one PC count are used for computing the
        selection curve. This is more robust than using all criteria voxels
        because it focuses on voxels that actually respond to denoising.
        GLMdenoise default is 0.05.
    min_noise_voxels : int, default=100
        Minimum voxels required in noise pool
    max_noise_fraction : float, default=0.95
        Maximum fraction of voxels in noise pool
    polort : int, default=2
        Polynomial order to project out from noise pool before PCA.
        This prevents slow drift from dominating extracted components.
        Set to -1 to disable polynomial projection.
    cv_strategy : int or float, default=1
        Cross-validation strategy:
        - 1 (or 'loro'): Leave-one-run-out (default)
        - int > 1: Leave-N-runs-out
        - float in (0, 1): Train fraction (e.g., 0.5 for split-halves)
    n_perms : int, default=100
        Maximum number of CV permutations for random splits.
        For LORO, this should be >= n_runs to get all splits.
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

    # Validate inputs
    if design_matrix is None and designs_by_hrf is None:
        raise ValueError("Either design_matrix or designs_by_hrf must be provided")
    if design_matrix is not None and designs_by_hrf is not None:
        raise ValueError("Cannot provide both design_matrix and designs_by_hrf")
    if designs_by_hrf is not None and hrf_indices is None:
        raise ValueError("hrf_indices required when designs_by_hrf is provided")

    # Determine mode
    per_hrf_mode = designs_by_hrf is not None

    n_runs = len(run_starts)
    n_timepoints = data.shape[1]
    n_voxels = data.shape[0]

    # Auto-determine polort based on run length if not specified
    # Uses AFNI formula: 1 + floor(run_duration / 150)
    if polort is None:
        # Calculate median run length (in case runs have different lengths)
        run_lengths = []
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_lengths.append(end_tp - start_tp)

        median_run_length = int(np.median(run_lengths))
        run_duration = median_run_length * tr
        polort = int(np.floor(1 + run_duration / 150.0))

        if verbose:
            print(f"\nAuto-determined polort={polort} (median run: {median_run_length} TRs = {run_duration:.1f}s)")

    # Convert nuisance to list format for consistent handling
    # This ensures extract_noise_pcs_per_run gets the exact same nuisance
    # matrices that will be used in the GLM model
    nuisance_per_run: Optional[List[torch.Tensor]] = None
    if nuisance is not None:
        if isinstance(nuisance, list):
            nuisance_per_run = nuisance
        else:
            # Single matrix - need to split by runs
            nuisance_per_run = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                nuisance_per_run.append(nuisance[start_tp:end_tp, :])

    if verbose:
        print(f"\n{'=' * 70}")
        if per_hrf_mode:
            print("Cross-Validated Denoising Pipeline (Per-Voxel HRF Mode)")
        else:
            print("Cross-Validated Denoising Pipeline")
        print(f"{'=' * 70}")

    # Per-HRF mode: Compute initial R² for each HRF group to build unified noise pool
    if per_hrf_mode and initial_r2 is None:
        if verbose:
            print("\nStep 1a: Computing baseline R² per HRF group for unified noise pool...")
            unique_hrf_indices = torch.unique(hrf_indices).tolist()
            print(f"  Processing {len(unique_hrf_indices)} HRF groups")

        # Allocate unified R² array
        initial_r2 = torch.zeros(n_voxels, device=device)

        # Create empty nuisance per run if needed
        nuisance_per_run_local = nuisance_per_run
        if nuisance_per_run_local is None:
            nuisance_per_run_local = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = end_tp - start_tp
                nuisance_per_run_local.append(torch.zeros((run_length, 0), device=device))

        # Generate CV splits once (same for all HRF groups)
        cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)

        # Process each HRF group to compute baseline R²
        for hrf_idx in unique_hrf_indices:
            voxel_mask = hrf_indices == hrf_idx
            n_voxels_group = voxel_mask.sum().item()

            if verbose:
                print(f"    HRF {hrf_idx}: {n_voxels_group:,} voxels")

            # Extract data and design for this group
            group_data = data[voxel_mask, :]
            group_design = designs_by_hrf[hrf_idx]
            n_task_cols = group_design.shape[1]

            # Project out nuisance from data/design
            projected_data, projected_design = project_out_nuisance_per_run(
                data=group_data,
                design=group_design,
                nuisance_per_run=nuisance_per_run_local,
                run_starts=run_starts,
                device=group_data.device,
            )

            # Compute cross-validated R² for this group
            xval_results = compute_xval_r2(
                data=projected_data,
                design_matrix=projected_design,
                run_starts=run_starts,
                stim_indices=list(range(n_task_cols)),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric="cod",
                zero_event_strategy="zero",
                device=device,
                batch_size=chunk_size,
                r2_method=r2_method,
                verbose=False,
            )

            # Store R² values for this group
            initial_r2[voxel_mask] = xval_results['r2']  # Task-only R²

        if verbose:
            print(f"\n  Unified noise pool created from {n_voxels:,} voxels across all HRFs")
            print(f"  Median baseline R²: {initial_r2.median().item():.6f}")

    # Step 1: Compute initial R² if not provided (standard single-HRF mode)
    # CRITICAL: Use cross-validated TASK-ONLY R² (not full-model R²)
    # Full-model R² includes variance explained by polynomials (drift) which
    # inflates R² by 20-50%! This would incorrectly classify many noise voxels
    # as criteria voxels.
    #
    # GLMdenoise/GLMsingle approach (PROJECT FIRST):
    # 1. Project out nuisance (polys, motion) from BOTH data AND design per run
    # 2. Run CV on projected data with projected task-only design
    # 3. Compute R² on projected test data vs predictions
    #
    # This avoids numerical issues with block-diagonal nuisance during CV
    # (zero-padded columns would cause singular projection matrices)
    if initial_r2 is None and not per_hrf_mode:
        if verbose:
            print("\nStep 1: Computing cross-validated task-only R² for noise pool selection...")
            print("  (project-first nuisance removal, per run)")

        n_task_cols = design_matrix.shape[1]

        # Use the already-converted nuisance_per_run (or create empty if None)
        nuisance_per_run_local = nuisance_per_run
        if nuisance_per_run_local is None:
            # No nuisance - create empty tensors per run
            nuisance_per_run_local = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = end_tp - start_tp
                nuisance_per_run_local.append(torch.zeros((run_length, 0), device=device))

        # Generate CV splits based on strategy
        cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
        n_splits = len(cv_splits)

        # Compute cross-validated R² using the shared xval utilities
        # 1) Project out nuisance from data/design per run (project-first)
        # 2) Run LORO CV on projected data/design
        projected_data, projected_design = project_out_nuisance_per_run(
            data=data,
            design=design_matrix,
            nuisance_per_run=nuisance_per_run_local,
            run_starts=run_starts,
            device=data.device,
        )

        xval_results = compute_xval_r2(
            data=projected_data,
            design_matrix=projected_design,
            run_starts=run_starts,
            stim_indices=list(range(n_task_cols)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy="zero",
            device=device,
            batch_size=chunk_size,
            r2_method=r2_method,
            verbose=False,
        )

        initial_r2 = xval_results["r2"].to(device)

        if verbose:
            print(f"  Completed {n_splits} CV folds (GLMdenoise-style concatenation)")
            print(f"  Median xval R²: {initial_r2.median().item():.4f}")
            print(f"  R² range: [{initial_r2.min().item():.4f}, {initial_r2.max().item():.4f}]")
    else:
        if verbose:
            print("\nStep 1: Using pre-computed R² for noise pool selection")
            print(f"  Median R²: {initial_r2.median().item():.4f}")

    # Filter out voxels with impossible R² values (|R²| > 1)
    # R² must be in [-1, 1] by definition, values outside indicate numerical issues
    # Common at brain edges, CSF boundaries, and regions with extreme artifacts
    # Set them to zero and exclude from all analysis
    valid_r2_mask = (initial_r2 >= -1.0) & (initial_r2 <= 1.0)
    n_extreme = (~valid_r2_mask).sum().item()

    # Track the valid voxel mask for output (all voxels are initially valid)
    # Use computation device (not data device) to match initial_r2
    valid_voxel_mask = torch.ones(data.shape[0], dtype=torch.bool, device=initial_r2.device)

    if n_extreme > 0:
        if verbose:
            print(f"\n⚠️  Found {n_extreme:,} voxels with impossible R² (|R²| > 1.0)")
            print(
                f"    R² range before filtering: [{initial_r2.min().item():.4f}, {initial_r2.max().item():.4f}]"
            )
            n_too_low = (initial_r2 < -1.0).sum().item()
            n_too_high = (initial_r2 > 1.0).sum().item()
            print(f"    R² < -1.0: {n_too_low:,} voxels")
            print(f"    R² > 1.0: {n_too_high:,} voxels")

        # Set impossible R² values to zero (excluded from analysis)
        # This ensures they won't be selected for noise pool or criteria
        initial_r2 = torch.where(
            valid_r2_mask, initial_r2, torch.tensor(0.0, device=initial_r2.device)
        )

        # Mark these voxels as invalid in the output mask
        valid_voxel_mask = valid_r2_mask

        if verbose:
            print(
                f"    R² range after filtering: [{initial_r2.min().item():.4f}, {initial_r2.max().item():.4f}]"
            )
            print(
                "    These voxels will be excluded from noise pool, criteria, and set to zero in outputs"
            )

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

    # Exclude voxels with extreme R² from both noise pool and criteria
    if n_extreme > 0:
        noise_pool_mask = noise_pool_mask & valid_r2_mask
        criteria_mask = criteria_mask & valid_r2_mask
        if verbose:
            print(f"  Excluded {n_extreme:,} extreme R² voxels from noise pool and criteria")

    # Apply intensity mask (brainthresh) to noise pool
    # This excludes low-intensity voxels from the noise pool
    if intensity_mask is not None:
        # Ensure intensity_mask is on the same device as noise_pool_mask
        # (noise_pool_mask is on computation device, intensity_mask may be on data device)
        intensity_mask_device = intensity_mask.to(noise_pool_mask.device)
        n_noise_before = noise_pool_mask.sum().item()
        noise_pool_mask = noise_pool_mask & intensity_mask_device
        n_noise_after = noise_pool_mask.sum().item()
        if verbose:
            print(
                f"  Applied intensity threshold: {n_noise_before:,} → {n_noise_after:,} noise pool voxels"
            )

    n_noise = noise_pool_mask.sum().item()
    n_criteria = criteria_mask.sum().item()

    if verbose:
        print(f"  Noise pool: {n_noise:,} voxels ({n_noise / data.shape[0] * 100:.1f}%)")
        print(f"  Criteria: {n_criteria:,} voxels ({n_criteria / data.shape[0] * 100:.1f}%)")
        print(f"  Note: Criteria mask will be refined after computing per-PC R² maps")

    # Step 3: Extract noise PCs per run (and optionally loadings)
    if verbose:
        n_nuisance = (
            sum(n.shape[1] for n in nuisance_per_run) // n_runs
            if nuisance_per_run is not None
            else 0
        )
        nuisance_msg = f", projecting {n_nuisance} nuisance" if n_nuisance > 0 else ""
        print(f"\nStep 3: Extracting PCs from noise pool (max={max_components}{nuisance_msg})...")

    pc_loadings = None
    if return_loadings:
        noise_pcs, pc_loadings = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            max_components=max_components,
            variance_threshold=variance_threshold,
            return_loadings=True,
            nuisance_per_run=nuisance_per_run,
            device=device,
            verbose=verbose,
        )
    else:
        noise_pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            max_components=max_components,
            variance_threshold=variance_threshold,
            return_loadings=False,
            nuisance_per_run=nuisance_per_run,
            device=device,
            verbose=verbose,
        )

    # Aggressive GPU cleanup between PCA extraction and cross-validation
    # CRITICAL: Move data to CPU to avoid OOM during cross-validation indexing
    # The cross-validation will stream chunks to GPU as needed
    if device.type == "cuda":
        if verbose:
            allocated_before = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU memory after PCA: {allocated_before:.2f} GB allocated")
            print(f"  Moving data to CPU for cross-validation (will stream chunks to GPU)")

        data = data.to("cpu")
        design_matrix = design_matrix.to("cpu")

        # Also move nuisance regressors to CPU
        if nuisance is not None:
            if isinstance(nuisance, list):
                nuisance = [n.to("cpu") for n in nuisance]
            else:
                nuisance = nuisance.to("cpu")

        torch.cuda.empty_cache()

        if verbose:
            allocated_after = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU memory after moving to CPU: {allocated_after:.2f} GB allocated")

    # =========================================================================
    # Step 4: Cross-validate PC selection (GLMdenoise-style)
    # =========================================================================
    # Uses concatenated predictions across folds:
    # - For each PC count (0-20), accumulates predictions across all folds
    # - Computes single R² from concatenated predictions vs full timeseries
    # - Returns per-PC R² maps (n_voxels × n_pc_counts) for ALL voxels
    # - Criteria mask determined AFTER from voxels with R² > threshold in ANY PC count
    if verbose:
        print("\nStep 4: Cross-validating PC selection (ALL voxels)...")

    if per_hrf_mode:
        # Per-HRF mode: Process each HRF group separately, then aggregate
        unique_hrf_indices = torch.unique(hrf_indices).tolist()
        r2_maps = np.zeros((n_voxels, max_components + 1), dtype=np.float32)

        if verbose:
            print(f"  Processing {len(unique_hrf_indices)} HRF groups with shared PCs")

        for hrf_idx in unique_hrf_indices:
            voxel_mask = hrf_indices == hrf_idx
            n_voxels_group = voxel_mask.sum().item()

            if verbose:
                print(f"    HRF {hrf_idx}: {n_voxels_group:,} voxels")

            # Extract data and design for this group
            group_data = data[voxel_mask, :]
            group_design = designs_by_hrf[hrf_idx]

            # Cross-validate with this HRF's design + shared PCs
            group_r2_maps, _ = cross_validate_noise_pcs(
                data=group_data,
                design_matrix=group_design,
                noise_pcs=noise_pcs,
                run_starts=run_starts,
                max_components=max_components,
                tr=tr,
                nuisance=nuisance,
                metric=metric,
                cv_strategy=cv_strategy,
                n_perms=n_perms,
                chunk_size=chunk_size,
                preload_data_to_device=preload_data_to_device,
                device=device,
                verbose=False,  # Suppress per-group verbosity
            )

            # Scatter results back to full array
            r2_maps[voxel_mask.cpu().numpy(), :] = group_r2_maps

        # Compute summary statistics across all voxels
        r2_summary = np.median(r2_maps, axis=0) if metric == "median" else np.mean(r2_maps, axis=0)

    else:
        # Standard single-HRF mode
        r2_maps, r2_summary = cross_validate_noise_pcs(
            data=data,
            design_matrix=design_matrix,
            noise_pcs=noise_pcs,
            run_starts=run_starts,
            max_components=max_components,
            tr=tr,
            nuisance=nuisance,
            metric=metric,
            cv_strategy=cv_strategy,
            n_perms=n_perms,
            chunk_size=chunk_size,
            preload_data_to_device=preload_data_to_device,
            device=device,
            verbose=verbose,
        )

    # Determine criteria voxels: R² > threshold in ANY PC count (GLMdenoise Step 7)
    threshold = pcR2cutoff if pcR2cutoff is not None else 0.0
    optimal_n_components, criteria_mask_final = select_optimal_pcs(
        r2_maps=r2_maps,
        threshold=threshold,
        metric=metric,
    )

    # Convert criteria_mask_final to torch tensor
    criteria_mask = torch.from_numpy(criteria_mask_final)

    # Transpose r2_maps for compatibility with downstream code
    r2_per_voxel = r2_maps  # (n_voxels, n_pc_counts)
    r2_by_n_components = r2_summary  # (n_pc_counts,)
    r2_median_by_n_components = r2_summary  # Same in new approach

    # With concatenated predictions, per-fold R² is not meaningful
    # Provide a single-row array for backward compatibility with visualization
    r2_per_fold = r2_summary.reshape(1, -1)  # (1, max_components+1)

    # Apply pcR2cutoff: already handled in select_optimal_pcs above
    # But we can recompute the selection curve using criteria voxels if needed
    pcselection_mask: Optional[np.ndarray] = None
    if pcR2cutoff is not None and pcR2cutoff > 0:
        # r2_per_voxel shape: (n_voxels, max_components+1)
        # Recompute selection curve using only criteria voxels
        r2_criteria = r2_per_voxel[criteria_mask_final, :]
        n_selected = criteria_mask_final.sum()

        if verbose:
            print(
                f"\n  pcR2cutoff={pcR2cutoff}: {n_selected:,} of {r2_per_voxel.shape[0]:,} voxels selected"
            )

        if n_selected > 0:
            r2_by_n_components = np.median(r2_criteria, axis=0)
            r2_median_by_n_components = r2_by_n_components

            # Re-select optimal using refined curve
            optimal_n_components = int(np.argmax(r2_by_n_components))

            if verbose:
                print(
                    f"  Recomputed curve: Best R² = {r2_by_n_components[optimal_n_components]:.4f} at {optimal_n_components} PCs"
                )

    # Select optimal number of components using GLMdenoise-style early stopping
    # This is more robust to noise than pure argmax
    if pcstop < 0:
        # User override: use exactly this many PCs
        optimal_n_components = int(abs(pcstop))
        optimal_n_components = min(optimal_n_components, max_components)
        if verbose:
            print(f"  Using user-specified {optimal_n_components} PCs (pcstop={pcstop})")
    elif pcstop == 1.0:
        # Pure argmax (original behavior)
        optimal_n_components = int(np.argmax(r2_by_n_components))
    else:
        # GLMdenoise-style early stopping: find first PC count within threshold of max
        # Performance curve relative to 0 PCs
        curve = r2_by_n_components - r2_by_n_components[0]
        max_improvement = curve.max()

        # Find first PC count that achieves threshold * max_improvement
        # Start from 0 PCs and stop when we're within threshold of max
        threshold = max_improvement / pcstop  # e.g., max/1.05 = within 5% of max

        optimal_n_components = 0
        best_so_far = -np.inf
        for n_pcs in range(len(curve)):
            if curve[n_pcs] > best_so_far:
                optimal_n_components = n_pcs
                best_so_far = curve[n_pcs]
                # If we're within threshold of max, stop here
                if best_so_far >= threshold:
                    break

        if verbose and pcstop != 1.0:
            argmax_n = int(np.argmax(r2_by_n_components))
            if optimal_n_components != argmax_n:
                print(
                    f"  Early stopping: {optimal_n_components} PCs (within {(pcstop - 1) * 100:.0f}% of max at {argmax_n} PCs)"
                )

    baseline_r2 = float(r2_by_n_components[0])
    optimal_r2 = float(r2_by_n_components[optimal_n_components])
    improvement = optimal_r2 - baseline_r2

    # Build per-voxel xval R² map at optimal PC count
    # Now we have R² for ALL voxels, not just criteria
    xval_r2_optimal: Optional[torch.Tensor] = None
    if r2_per_voxel is not None:
        optimal_r2_per_voxel = r2_per_voxel[:, optimal_n_components]
        xval_r2_optimal = torch.from_numpy(optimal_r2_per_voxel).to(torch.float32)

    # We already have full-brain R² maps from cross_validate_noise_pcs
    # No need for separate compute_xval_r2_optimal_full call
    xval_r2_optimal_full = xval_r2_optimal
    xval_r2_optimal_per_fold: Optional[np.ndarray] = (
        None  # Not meaningful with concatenated approach
    )

    if verbose:
        print(f"\nFull-brain cross-validated R² at optimal PC count ({optimal_n_components} PCs):")
        valid_r2 = optimal_r2_per_voxel[~np.isnan(optimal_r2_per_voxel)]
        print(f"  Valid voxels: {len(valid_r2):,}")
        print(f"  Median R²: {np.median(valid_r2):.4f}")
        print(f"  Mean R²: {np.mean(valid_r2):.4f}")

    # Filter out impossible R² values from full-brain xval results
    if xval_r2_optimal_full is not None:
        xval_valid_mask = (xval_r2_optimal_full >= -1.0) & (xval_r2_optimal_full <= 1.0)
        n_xval_extreme = (~xval_valid_mask).sum().item()
        if n_xval_extreme > 0:
            if verbose:
                print(f"  Found {n_xval_extreme:,} voxels with impossible xval R² (|R²| > 1.0)")
                print("  Setting these to zero in xval R² output")
            xval_r2_optimal_full = torch.where(
                xval_valid_mask,
                xval_r2_optimal_full,
                torch.tensor(0.0, device=xval_r2_optimal_full.device),
            )

        # Also apply the original valid_voxel_mask (from initial R² filtering)
        xval_r2_optimal_full = torch.where(
            valid_voxel_mask.to(xval_r2_optimal_full.device),
            xval_r2_optimal_full,
            torch.tensor(0.0, device=xval_r2_optimal_full.device),
        )

    # Build metadata
    metadata = {
        "r2_threshold": r2_threshold,
        "max_components": max_components,
        "variance_threshold": variance_threshold,
        "metric": metric,
        "min_noise_voxels": min_noise_voxels,
        "max_noise_fraction": max_noise_fraction,
        "n_noise_voxels": n_noise,
        "n_criteria_voxels": int(criteria_mask.sum().item()),
        "n_runs": len(run_starts),
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print("✅ Denoising Complete")
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
        xval_r2_per_voxel=r2_per_voxel,
        xval_r2_optimal=xval_r2_optimal,
        xval_r2_optimal_full=xval_r2_optimal_full,
        xval_r2_optimal_per_fold=xval_r2_optimal_per_fold,
        pcselection_mask=pcselection_mask,
        noise_pool_mask=noise_pool_mask,
        criteria_mask=criteria_mask,
        valid_voxel_mask=valid_voxel_mask,
        noise_pool_r2=initial_r2,
        noise_pcs_per_run=noise_pcs,
        pc_loadings_per_run=pc_loadings,
        baseline_r2=baseline_r2,
        optimal_r2=optimal_r2,
        improvement=improvement,
        metadata=metadata,
    )
