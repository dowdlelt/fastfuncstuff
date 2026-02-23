"""
High-level analysis workflows for fMRI data
Complete pipelines from AFNI files to GLM results
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch

from .afni_io import (
    extract_stimulus_columns,
    get_run_lengths,
    load_afni_mask,
    load_and_concatenate_runs,
    onsets_to_binary_matrix,
    read_afni_design_matrix,
    read_afni_onset_files,
)
from .arma_glm import ARMA11Results, fit_glm_arma11, get_default_arma_grids
from .design import build_glm_design, convolve_hrf_microtime
from .glm_core import GLMResults, fit_glm, fit_glm_hrf_library
from .hrf import get_hrf_library
from .utils import get_device, to_tensor


def analyze_from_onsets(
    fmri_data: Union[str, Path, List[Union[str, Path]], np.ndarray, torch.Tensor],
    onset_files: List[Union[str, Path]],
    tr: float,
    hrf_mode: str = "canonical",
    stim_duration: float = 0.0,
    n_hrfs: int = 20,
    method: str = "ols",
    arma_a_grid: Optional[torch.Tensor] = None,
    arma_b_grid: Optional[torch.Tensor] = None,
    run_starts: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    test_n_voxels: Optional[int] = None,
    enable_quick_estimate: bool = False,
    stim_labels: Optional[List[str]] = None,
    polort: Optional[int] = None,
    verbose: bool = True,
    microtime_dt: float = 0.1,
    microtime_onset: int = 0,
    **hrf_kwargs,
) -> Union[GLMResults, ARMA11Results]:
    """
    Complete analysis pipeline: AFNI onset files → HRF → GLM results

    This function provides a complete analysis workflow starting from AFNI onset
    timing files, building the design matrix with HRF convolution, and fitting
    the GLM using either OLS or ARMA(1,1) prewhitened GLS.

    Supports both single concatenated files and multiple run files.

    Parameters
    ----------
    fmri_data : str, Path, list of str/Path, np.ndarray, or torch.Tensor
        fMRI data as:
        - Path to single concatenated NIfTI file (all runs in one file)
        - List of paths to NIfTI files (one per run) - will be concatenated
        - np.ndarray with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
        - torch.Tensor with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
    onset_files : list of str or Path
        Paths to AFNI onset timing files, one per condition
    tr : float
        Repetition time in seconds
    hrf_mode : str
        HRF mode: 'canonical', 'pighs', 'library', or 'fir'
        - 'canonical': Single SPM canonical HRF (assumed HRF)
        - 'library': Fit library of HRF variants, pick best per voxel
        - 'pighs': Use PIGHS (Parametric Individually Generated HRFs) library
        - 'fir': Finite impulse response (no HRF assumption)
    stim_duration : float
        Stimulus duration in seconds (for HRF convolution)
    n_hrfs : int
        Number of HRFs in library (for 'library' or 'pighs' modes)
    method : str
        GLM method: 'ols' or 'arma11'
        - 'ols': Ordinary least squares
        - 'arma11': ARMA(1,1) prewhitened generalized least squares
    arma_a_grid : torch.Tensor, optional
        Grid of 'a' parameter values for ARMA(1,1) (only used if method='arma11')
        If None, uses AFNI -Grid 3 defaults: linspace(0.1, 0.9, 9) = 9 points, 63 total combos
    arma_b_grid : torch.Tensor, optional
        Grid of 'b' parameter values for ARMA(1,1) (only used if method='arma11')
        If None, uses AFNI -Grid 3 defaults: linspace(-0.3, 0.3, 7) = 7 points, 63 total combos
    run_starts : list of int, optional
        Starting timepoint indices for each run (AFNI RunStart parameter)
        If None, assumes equal-length runs
        Required if fmri_data is a list of files or if runs have unequal length
        Example: [0, 300, 600, 900] for 4 runs starting at TRs 0, 300, 600, 900
    device : torch.device, optional
        Device for computation
    test_n_voxels : int, optional
        If provided, only analyze a subset of voxels for testing (extracts cube from center)
    enable_quick_estimate : bool
        Enable quick ARMA parameter estimation (only used with method='arma11')
    stim_labels : list of str, optional
        Labels for each stimulus condition (for metadata/output organization)
    polort : int, optional
        Polynomial order for detrending. If None, auto-computed based on run duration.
        Equivalent to AFNI's -polort option.
    verbose : bool
        Print progress information (default: True)
    microtime_dt : float, default=0.1
        Microtime resolution in seconds. Default 0.1s is the standard throughout
        the pipeline. Onsets and HRFs are at this resolution.
    microtime_onset : int, default=0
        Which microtime bin within each TR to sample (0-indexed).
        0 = start of TR, bins_per_tr/2 = middle of TR.
    **hrf_kwargs : dict
        Additional arguments passed to HRF generation functions

    Returns
    -------
    results : GLMResults or ARMA11Results
        GLM results with betas, t-stats, F-stats, etc.

    Examples
    --------
    >>> # Single concatenated file with canonical HRF and OLS
    >>> results = analyze_from_onsets(
    ...     'func_all_runs.nii.gz',
    ...     ['onsets_cond1.txt', 'onsets_cond2.txt'],
    ...     tr=2.0,
    ...     hrf_mode='canonical',
    ...     method='ols'
    ... )
    >>> print(f"R² = {results.r2.mean():.3f}")

    >>> # Multiple run files (will be concatenated automatically)
    >>> results = analyze_from_onsets(
    ...     ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz'],
    ...     ['onsets_cond1.txt', 'onsets_cond2.txt'],
    ...     tr=2.0,
    ...     hrf_mode='canonical',
    ...     method='ols'
    ... )

    >>> # Unequal-length runs with explicit run_starts
    >>> results = analyze_from_onsets(
    ...     'func_all_runs.nii.gz',
    ...     ['onsets_cond1.txt', 'onsets_cond2.txt'],
    ...     tr=1.5,
    ...     run_starts=[0, 300, 600, 900],  # 4 runs of 300 TRs each
    ...     hrf_mode='library',
    ...     n_hrfs=20,
    ...     method='arma11'
    ... )
    """
    if device is None:
        device = get_device()

    # 1. Load fMRI data (handle both single file and multiple run files)
    original_shape: Optional[Tuple[int, int, int]] = None

    if isinstance(fmri_data, list):
        # Multiple run files - concatenate them
        data, inferred_run_starts = load_and_concatenate_runs(fmri_data, device)
        # Use inferred run_starts if not provided
        if run_starts is None:
            run_starts = inferred_run_starts
    else:
        # Single file or tensor
        data = _load_fmri_data(fmri_data, device)

    # data should be (n_voxels, n_timepoints)
    if data.ndim == 4:
        # Reshape (x, y, z, t) -> (n_voxels, t)
        original_shape = data.shape[:3]
        data = data.reshape(-1, data.shape[3])
    elif data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (n_voxels, n_timepoints) or 4D (x, y, z, t), got shape {data.shape}"
        )

    n_voxels, n_timepoints = data.shape

    # 1.5 subset data if test_n_voxels provided
    if test_n_voxels is not None:
        if original_shape is None:
            # If not 4D, just take the first N voxels
            data = data[:test_n_voxels, :]
        else:
            # If 4D, extract from center (matches analyze_from_design_matrix)
            cube_side = int(np.ceil(test_n_voxels ** (1 / 3)))
            center = np.array(original_shape) // 2
            half_cube = cube_side // 2

            test_mask_3d = np.zeros(original_shape, dtype=bool)
            x_start = max(0, center[0] - half_cube)
            x_end = min(original_shape[0], center[0] + half_cube)
            y_start = max(0, center[1] - half_cube)
            y_end = min(original_shape[1], center[1] + half_cube)
            z_start = max(0, center[2] - half_cube)
            z_end = min(original_shape[2], center[2] + half_cube)

            test_mask_3d[x_start:x_end, y_start:y_end, z_start:z_end] = True
            test_mask_flat = test_mask_3d.reshape(-1)
            mask_tensor = torch.from_numpy(test_mask_flat.astype(np.bool_))
            data = data[mask_tensor, :]

        n_voxels = data.shape[0]
        print(f"🧪 Test mode: using {n_voxels:,} voxels")

    # 2. Set microtime defaults
    if microtime_onset is None:
        # Default to first bin (no shift) - events at actual onset times
        microtime_onset = 1

    # 3. Read onset files and convert to binary matrix
    onset_data = read_afni_onset_files(onset_files)
    onsets = onsets_to_binary_matrix(
        onset_data,
        n_timepoints,
        tr,
        run_starts=run_starts,
        device=device,
        microtime_dt=microtime_dt,
    )

    bins_per_tr = int(round(tr / microtime_dt))
    if verbose:
        print(
            f"📐 Microtime: dt={microtime_dt}s ({bins_per_tr} bins/TR), "
            f"sampling at bin {microtime_onset}"
        )

    # 4. Build design matrix based on HRF mode
    if hrf_mode == "fir":
        # FIR: no HRF assumption (microtime not applicable - FIR estimates full response)
        # For FIR, downsample onsets back to TR resolution
        # FIR doesn't benefit from microtime since it estimates the full response
        onsets_tr = onsets[microtime_onset::bins_per_tr, :]
        design = build_glm_design(
            onsets_tr,
            hrf=None,
            n_timepoints=n_timepoints,
            mode="fir",
            device=device,
            **hrf_kwargs,
        )

    elif hrf_mode == "canonical":
        # Single canonical HRF with microtime convolution
        # Get HRF at microtime resolution
        hrf = get_hrf_library(
            mode="single",
            stim_duration=stim_duration,
            microtime_dt=microtime_dt,
            device=device,
        )
        # Use microtime convolution and downsample
        design = convolve_hrf_microtime(
            onsets,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            microtime_onset=microtime_onset,
            device=device,
        )

    elif hrf_mode in ["library", "pighs", "flobs"]:
        # HRF library - will fit with library below
        # Note: 'flobs' is kept for backwards compatibility but 'pighs' is preferred
        hrf_library = get_hrf_library(
            mode="library" if hrf_mode == "library" else "pighs",
            stim_duration=stim_duration,
            microtime_dt=microtime_dt,
            n_hrfs=n_hrfs,
            device=device,
            **hrf_kwargs,
        )
        # For library mode, we'll use fit_glm_hrf_library below
        # For now, just flag it
        design = None

    else:
        raise ValueError(
            f"Unknown hrf_mode: {hrf_mode}. Choose 'canonical', 'library', 'pighs', or 'fir'"
        )

    # 5. Fit GLM
    if method == "ols":
        if hrf_mode in ["library", "pighs", "flobs"]:
            # Use HRF library fitting
            # Filter kwargs to only pass valid fit_glm parameters
            fit_glm_params = set(inspect.signature(fit_glm).parameters.keys())
            glm_kwargs = {k: v for k, v in hrf_kwargs.items() if k in fit_glm_params}
            # Add explicit parameters
            glm_kwargs["max_poly_degree"] = polort
            glm_kwargs["verbose"] = verbose

            results, hrf_idx, r2_all = fit_glm_hrf_library(
                data,
                onsets,
                hrf_library,
                tr=tr,
                device=device,
                microtime_dt=microtime_dt,
                microtime_onset=microtime_onset,
                n_timepoints=n_timepoints,
                **glm_kwargs,
            )
            # Store HRF indices in results
            results.hrf_idx = hrf_idx
            results.r2_per_hrf = r2_all

        else:
            # Standard OLS
            assert design is not None, "design should not be None for non-library modes"
            results = fit_glm(
                data,
                design,
                tr=tr,
                device=device,
                max_poly_degree=polort,
                verbose=verbose,
            )

    elif method == "arma11":
        if hrf_mode in ["library", "pighs", "flobs"]:
            raise NotImplementedError(
                "ARMA(1,1) with HRF library not yet implemented. "
                "Use method='ols' or hrf_mode='canonical'/'fir'"
            )

        # Create default grids if not provided (standardized AFNI defaults)
        if arma_a_grid is None and arma_b_grid is None:
            arma_a_grid, arma_b_grid = get_default_arma_grids(device)
        else:
            if arma_a_grid is None:
                arma_a_grid, _ = get_default_arma_grids(device)
            if arma_b_grid is None:
                _, arma_b_grid = get_default_arma_grids(device)

        # ARMA(1,1) prewhitened GLS
        assert design is not None, "design should not be None for ARMA (library mode blocked above)"
        results = fit_glm_arma11(
            data,
            design,
            tr=tr,
            a_grid=arma_a_grid,
            b_grid=arma_b_grid,
            device=device,
            enable_quick_estimate=enable_quick_estimate,
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ols' or 'arma11'")

    if original_shape is not None:
        results.original_shape = original_shape
        results.full_shape = original_shape

    return results


def analyze_from_design_matrix(
    fmri_data: Union[str, Path, List[Union[str, Path]], np.ndarray, torch.Tensor],
    design_matrix_file: Union[str, Path],
    method: str = "ols",
    use_stimulus_only: bool = False,
    arma_a_grid: Optional[torch.Tensor] = None,
    arma_b_grid: Optional[torch.Tensor] = None,
    precomputed_arma_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
    want_ols: bool = False,
    ols_output_path: Optional[Union[str, Path]] = None,
    ols_output_format: str = "nii.gz",
    device: Optional[torch.device] = None,
    mask_file: Optional[Union[str, Path]] = None,
    mask_threshold: float = 0.0,
    cache_file: Optional[Union[str, Path]] = None,
    cached_metadata: Optional[Dict] = None,
    test_n_voxels: Optional[int] = None,
    voxel_chunk_size: Optional[int] = None,
    use_double: bool = False,
    debug_memory: bool = False,
    enable_quick_estimate: bool = False,
    use_grid_batching: Optional[bool] = None,
    want_r2_partial: bool = False,
    r2_partial_mode: str = "full",  # "full" or "task" - how to compute partial R²
    want_r2_semipartial: bool = False,
    r2_semipartial_mode: str = "full",  # "full" or "task" - how to compute semi-partial R²
    legacy_contrasts: bool = False,
) -> Tuple[Union[GLMResults, ARMA11Results], Dict]:
    """
    Complete analysis pipeline: AFNI design matrix → GLM results

    This function reads a pre-built AFNI design matrix (X.xmat.1D format)
    and fits the GLM using either OLS or ARMA(1,1) prewhitened GLS.

    The design matrix is assumed to already include HRF convolution, polynomials,
    motion parameters, and any other nuisance regressors.

    Supports both single concatenated files and multiple run files. The RunStart
    parameter from the design matrix is used to validate alignment.

    Parameters
    ----------
    fmri_data : str, Path, list of str/Path, np.ndarray, or torch.Tensor
        fMRI data as:
        - Path to single concatenated NIfTI file (all runs in one file)
        - List of paths to NIfTI files (one per run) - will be concatenated
        - np.ndarray with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
        - torch.Tensor with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
    design_matrix_file : str or Path
        Path to AFNI design matrix file (X.xmat.1D)
    method : str
        GLM method: 'ols' or 'arma11'
    use_stimulus_only : bool
        If True, extract only stimulus columns (ignore polynomials and nuisance)
        If False, use complete design matrix
    arma_a_grid : torch.Tensor, optional
        Grid of 'a' parameter values for ARMA(1,1) (only used if method='arma11')
        If None, uses AFNI -Grid 3 defaults: linspace(0.1, 0.9, 9) = 9 points, 63 total combos
    arma_b_grid : torch.Tensor, optional
        Grid of 'b' parameter values for ARMA(1,1) (only used if method='arma11')
        If None, uses AFNI -Grid 3 defaults: linspace(-0.3, 0.3, 7) = 7 points, 63 total combos
    precomputed_arma_params : array-like, shape (n_voxels, 2) or (x, y, z, 2), optional
        Precomputed ARMA(1,1) parameters [a, b] for each voxel (only used if method='arma11')
        Can be 2D (n_voxels, 2) or 4D (x, y, z, 2) - will be reshaped/masked consistently with data
        If provided, skips REML estimation (saves ~80% of compute time)
        Useful for re-running analysis with different contrasts or validating against AFNI
    want_ols : bool, default=False
        Also compute OLS baseline for comparison (only used if method='arma11')
        When True, adds `ols_results` attribute to returned ARMA11Results object
        Useful for validating ARMA improvement over OLS
    device : torch.device, optional
        Device for computation
    mask_file : str or Path, optional
        Optional NIfTI mask file used to restrict analysis to a subset of voxels
    mask_threshold : float
        Threshold applied to mask values; voxels greater than this are included
    voxel_chunk_size : int, optional
        Number of voxels to process per chunk. Overrides automatic heuristic
    use_double : bool, default=False
        If True, use float64 precision (matches AFNI exactly, ~2x memory, ~1.5x slower).
        If False, use float32 precision (faster, tiny differences from AFNI).

    Returns
    -------
    results : GLMResults or ARMA11Results
        GLM results with betas, t-stats, F-stats, etc.
    design_info : dict
        Parsed design matrix information from AFNI file

    Examples
    --------
    >>> # Single concatenated file with OLS
    >>> results, design_info = analyze_from_design_matrix(
    ...     'func_all_runs.nii.gz',
    ...     'X.xmat.1D',
    ...     method='ols'
    ... )
    >>> print(f"Conditions: {design_info['stim_labels']}")
    >>> print(f"Run starts: {design_info['run_starts']}")
    >>> print(f"R² = {results.r2.mean():.3f}")

    >>> # Multiple run files (will be concatenated and validated)
    >>> results, design_info = analyze_from_design_matrix(
    ...     ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    ...     'X.xmat.1D',
    ...     method='arma11',
    ...     use_stimulus_only=True
    ... )
    >>> # RunStart from design matrix will be used to validate alignment
    """
    if device is None:
        device = get_device()

    # 1. Read design matrix first to get run_starts
    design_info = read_afni_design_matrix(design_matrix_file)
    expected_run_starts = design_info.get("run_starts", None)

    # 2. Load fMRI data (handle both single file and multiple run files)
    storage_device = torch.device("cpu")
    mask_tensor: Optional[torch.Tensor] = None
    volume_shape: Optional[Tuple[int, int, int]] = None
    affine = None
    nifti_header = None  # Store full NIfTI header for cache

    # CRITICAL: If using cached data, extract header/affine from cache metadata
    if cached_metadata is not None:
        if "nifti_header" in cached_metadata:
            nifti_header = cached_metadata["nifti_header"]
        if "affine" in cached_metadata:
            affine = cached_metadata["affine"]
        if "volume_shape" in cached_metadata:
            volume_shape = cached_metadata["volume_shape"]

    if isinstance(fmri_data, list):
        if len(fmri_data) == 0:
            raise ValueError("fmri_data list is empty; provide at least one run file")

        first_path = Path(fmri_data[0])
        if not first_path.exists():
            raise FileNotFoundError(f"Run file not found: {first_path}")

        first_img = nib.load(str(first_path))
        if len(first_img.shape) < 4:
            raise ValueError(
                f"Expected 4D fMRI runs, but '{first_path.name}' has shape {first_img.shape}"
            )

        volume_shape = tuple(int(dim) for dim in first_img.shape[:3])
        affine = first_img.affine
        nifti_header = first_img.header  # Capture full header for cache

        data, actual_run_starts = load_and_concatenate_runs(fmri_data, storage_device)

        # Validate run_starts match if both are available
        if expected_run_starts is not None:
            if len(actual_run_starts) != len(expected_run_starts):
                raise ValueError(
                    f"Number of run files ({len(actual_run_starts)}) doesn't match "
                    f"number of runs in design matrix ({len(expected_run_starts)})"
                )
            # Check that run lengths match
            expected_lengths = get_run_lengths(expected_run_starts, design_info["n_timepoints"])
            actual_lengths = get_run_lengths(actual_run_starts, data.shape[1])
            if expected_lengths != actual_lengths:
                raise ValueError(
                    f"Run lengths from files {actual_lengths} don't match "
                    f"run lengths from design matrix {expected_lengths}"
                )
    else:
        # Single file or tensor
        data = _load_fmri_data(fmri_data, storage_device)

        if isinstance(fmri_data, (str, Path)):
            fmri_path = Path(fmri_data)
            if not fmri_path.exists():
                raise FileNotFoundError(f"fMRI data file not found: {fmri_path}")
            img = nib.load(str(fmri_path))
            if len(img.shape) >= 3:
                volume_shape = tuple(int(dim) for dim in img.shape[:3])
            affine = img.affine
            nifti_header = img.header  # Capture full header for cache

    # data should be (n_voxels, n_timepoints)
    if data.ndim == 4:
        # Reshape (x, y, z, t) -> (n_voxels, t)
        volume_shape = tuple(int(dim) for dim in data.shape[:3])
        data = data.reshape(-1, data.shape[3])
    elif data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (n_voxels, n_timepoints) or 4D (x, y, z, t), got shape {data.shape}"
        )

    n_voxels, n_timepoints = data.shape

    # PARALLEL TRANSFORM: Reshape precomputed_arma_params same as data (4D → 2D)
    # This ensures consistent masking/test mode transformations
    if precomputed_arma_params is not None:
        # Convert to tensor if needed
        if isinstance(precomputed_arma_params, np.ndarray):
            precomputed_arma_params = torch.from_numpy(precomputed_arma_params)

        if precomputed_arma_params.ndim == 4:
            # Reshape (x, y, z, 2) -> (n_voxels, 2)
            arma_volume_shape = tuple(int(dim) for dim in precomputed_arma_params.shape[:3])

            # Validate spatial dimensions match data
            if volume_shape is not None and arma_volume_shape != volume_shape:
                raise ValueError(
                    f"Precomputed ARMA params volume shape {arma_volume_shape} "
                    f"does not match data volume shape {volume_shape}"
                )

            precomputed_arma_params = precomputed_arma_params.reshape(
                -1, precomputed_arma_params.shape[3]
            )
            print(
                f"📊 Reshaped precomputed ARMA params: {arma_volume_shape} → ({precomputed_arma_params.shape[0]:,}, {precomputed_arma_params.shape[1]})"
            )

        elif precomputed_arma_params.ndim == 2:
            # Already 2D - validate voxel count matches
            if precomputed_arma_params.shape[0] != n_voxels:
                raise ValueError(
                    f"Precomputed ARMA params has {precomputed_arma_params.shape[0]:,} voxels, "
                    f"but data has {n_voxels:,} voxels"
                )
        else:
            raise ValueError(
                f"Precomputed ARMA params must be 2D (n_voxels, 2) or 4D (x, y, z, 2), "
                f"got shape {precomputed_arma_params.shape}"
            )

        # Validate parameter dimension
        if precomputed_arma_params.shape[1] != 2:
            raise ValueError(
                f"Precomputed ARMA params must have 2 parameters (a, b), "
                f"got {precomputed_arma_params.shape[1]}"
            )

        # Initial validation: voxel counts match
        assert precomputed_arma_params.shape[0] == n_voxels, (
            f"After reshape: precomputed ARMA params has {precomputed_arma_params.shape[0]:,} voxels, "
            f"but data has {n_voxels:,} voxels"
        )

    # Scaling is the caller's responsibility (e.g. via -do_scale in the CLI,
    # which uses scale_to_percent_signal for correct per-run scaling).
    # Record for cache metadata only.
    was_scaled = False
    original_mean = float(data.mean())

    # Save to HDF5 cache if requested (BEFORE masking/test mode)
    # This allows fast loading on subsequent runs
    if cache_file is not None and isinstance(fmri_data, list):
        from .data_cache import save_cache

        # Get run starts - need to extract from design_info or actual_run_starts
        run_starts_for_cache = None
        if "actual_run_starts" in locals():
            run_starts_for_cache = actual_run_starts

        # Convert data to numpy for saving
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        save_cache(
            cache_file=cache_file,
            data=data_np,
            input_files=fmri_data,
            run_starts=run_starts_for_cache,
            affine=affine,
            volume_shape=volume_shape,
            was_scaled=was_scaled,
            original_mean=original_mean,
            nifti_header=nifti_header,
        )

    # TEST MODE: Create test mask to extract ~N voxels from center
    if test_n_voxels is not None:
        if volume_shape is None:
            raise ValueError("Test mode requires 4D input data to determine volume shape")

        # Calculate cube size to get approximately test_n_voxels
        cube_side = int(np.ceil(test_n_voxels ** (1 / 3)))

        # Find center of volume
        center = np.array(volume_shape) // 2
        half_cube = cube_side // 2

        # Create 3D mask cube around center
        test_mask_3d = np.zeros(volume_shape, dtype=bool)
        x_start = max(0, center[0] - half_cube)
        x_end = min(volume_shape[0], center[0] + half_cube)
        y_start = max(0, center[1] - half_cube)
        y_end = min(volume_shape[1], center[1] + half_cube)
        z_start = max(0, center[2] - half_cube)
        z_end = min(volume_shape[2], center[2] + half_cube)

        test_mask_3d[x_start:x_end, y_start:y_end, z_start:z_end] = True

        # Flatten and apply
        test_mask_flat = test_mask_3d.reshape(-1)
        mask_tensor = torch.from_numpy(test_mask_flat.astype(np.bool_))
        kept_voxels = int(mask_tensor.sum().item())

        print(f"🧪 Test mode: extracting {kept_voxels:,} voxels from center")
        print(f"   Cube: {cube_side}³ at center {tuple(center)}")
        print(f"   Bounds: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], z=[{z_start}:{z_end}]")

        data = data[mask_tensor, :]
        n_voxels = kept_voxels

        # PARALLEL TRANSFORM: Apply same test mask to precomputed_arma_params
        if precomputed_arma_params is not None:
            precomputed_arma_params = precomputed_arma_params[mask_tensor, :]
            print(
                f"   • Also masked precomputed ARMA params: {precomputed_arma_params.shape[0]:,} voxels"
            )

            # Validate dimensions still match after masking
            assert precomputed_arma_params.shape[0] == n_voxels, (
                f"After test mask: precomputed ARMA params has {precomputed_arma_params.shape[0]:,} voxels, "
                f"but data has {n_voxels:,} voxels"
            )

    elif mask_file is not None:
        mask_array = load_afni_mask(mask_file, threshold=mask_threshold)
        if volume_shape is None:
            volume_shape = mask_array.shape
        elif mask_array.shape != volume_shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} does not match data volume shape {volume_shape}"
            )

        mask_flat = mask_array.reshape(-1)
        if mask_flat.size != n_voxels:
            raise ValueError(
                "Mask voxel count does not match flattened data. Ensure the mask and data are aligned."
            )

        mask_tensor = torch.from_numpy(mask_flat.astype(np.bool_))
        kept_voxels = int(mask_tensor.sum().item())
        excluded_voxels = n_voxels - kept_voxels
        if kept_voxels == 0:
            raise ValueError(
                f"Mask '{mask_file}' excluded all voxels (threshold={mask_threshold})."
            )

        print("📊 Mask applied:")
        print(f"  Total voxels: {n_voxels:,}")
        print(f"  Kept (in mask): {kept_voxels:,} ({100 * kept_voxels / n_voxels:.1f}%)")
        print(f"  Excluded: {excluded_voxels:,} ({100 * excluded_voxels / n_voxels:.1f}%)")

        data = data[mask_tensor, :]
        n_voxels = kept_voxels

        # PARALLEL TRANSFORM: Apply same mask to precomputed_arma_params
        if precomputed_arma_params is not None:
            precomputed_arma_params = precomputed_arma_params[mask_tensor, :]
            print(
                f"  • Also masked precomputed ARMA params: {precomputed_arma_params.shape[0]:,} voxels"
            )

            # Validate dimensions still match after masking
            assert precomputed_arma_params.shape[0] == n_voxels, (
                f"After mask: precomputed ARMA params has {precomputed_arma_params.shape[0]:,} voxels, "
                f"but data has {n_voxels:,} voxels"
            )

    # 3. Extract design matrix (already read above)
    if use_stimulus_only:
        # Extract only stimulus columns
        design = extract_stimulus_columns(design_info, device=device)
    else:
        # Use complete design matrix
        design = to_tensor(design_info["matrix"], device=device, dtype=torch.float32)

    # Transpose to (n_timepoints, n_regressors) if needed
    if design.shape[0] != data.shape[1]:
        design = design.T

    # 4. Get TR from design info
    tr = design_info["tr"]

    # Extract stimulus metadata (needed for both OLS and ARMA)
    from .afni_io import extract_design_metadata

    full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)

    # 5. Fit GLM
    glm_poly = None if use_stimulus_only else -1

    if method == "ols":
        results = fit_glm(
            data,
            design,
            tr=tr,
            device=device,
            chunk_size=voxel_chunk_size,
            max_poly_degree=glm_poly,
            preload_data_to_device=False,
            use_double=use_double,
            glt_labels=design_info.get("glt_labels", None),
            glt_matrices=design_info.get("glt_matrices", None),
            task_indices=stim_indices if stim_indices else None,
            want_r2_partial=want_r2_partial,
            r2_partial_mode=r2_partial_mode,
            want_r2_semipartial=want_r2_semipartial,
            r2_semipartial_mode=r2_semipartial_mode,
        )

    elif method == "arma11":
        # Create default grids if not provided (standardized AFNI defaults)
        if arma_a_grid is None and arma_b_grid is None:
            arma_a_grid, arma_b_grid = get_default_arma_grids(device)
        else:
            if arma_a_grid is None:
                arma_a_grid, _ = get_default_arma_grids(device)
            if arma_b_grid is None:
                _, arma_b_grid = get_default_arma_grids(device)

        # Create simple OLS write callback if output path provided
        ols_write_callback = None
        if ols_output_path is not None:
            from .glm_outputs import write_glm_bucket_as_nifti

            # Capture volume_shape and affine from outer scope for the callback
            callback_volume_shape = volume_shape
            callback_affine = affine
            callback_stim_labels = stim_labels  # Capture labels
            callback_stim_indices = stim_indices if stim_indices else None  # Capture indices
            callback_design_matrix = design_info[
                "matrix"
            ]  # Capture design matrix for single trials

            def write_ols(ols_results, original_shape, affine):
                """Write OLS callback outputs during ARMA workflow.

                Parameters
                ----------
                ols_results : GLMResults
                    OLS fit results produced inside ARMA comparison workflow.
                original_shape : tuple[int, int, int] or None
                    Spatial shape inferred by caller; may be ``None`` when data was
                    provided in flattened form.
                affine : np.ndarray or None
                    Spatial affine matrix supplied by caller.

                Returns
                -------
                None
                    Writes OLS bucket outputs and optional single-trial files.
                """
                # IMPORTANT: When task_indices is passed to fit_glm(), the results
                # already contain ONLY the task regressors. No slicing needed!
                # The betas/tstats/fstats already correspond to the stimulus columns.

                # Use captured volume_shape from closure - original_shape from arma may be None
                # if data was already 2D when passed to fit_glm_arma11
                shape_to_use = (
                    original_shape if original_shape is not None else callback_volume_shape
                )
                affine_to_use = affine if affine is not None else callback_affine

                ols_results.original_shape = shape_to_use
                ols_results.affine = affine_to_use

                # Extract contrast information if available (GLT contrasts)
                contrast_names = getattr(ols_results, "contrast_labels", None)
                ols_contrast_results = None
                if (
                    hasattr(ols_results, "contrast_betas")
                    and ols_results.contrast_betas is not None
                ):
                    ols_contrast_results = {
                        "contrast_betas": ols_results.contrast_betas,
                        "contrast_tstats": ols_results.contrast_tstats,
                    }
                    # Add partial R² if available
                    if (
                        hasattr(ols_results, "contrast_r2_partial")
                        and ols_results.contrast_r2_partial is not None
                    ):
                        ols_contrast_results["contrast_r2_partial"] = (
                            ols_results.contrast_r2_partial
                        )
                    # Add semi-partial R² if available
                    if (
                        hasattr(ols_results, "contrast_r2_semipartial")
                        and ols_results.contrast_r2_semipartial is not None
                    ):
                        ols_contrast_results["contrast_r2_semipartial"] = (
                            ols_results.contrast_r2_semipartial
                        )

                write_glm_bucket_as_nifti(
                    ols_results,
                    ols_output_path,
                    condition_names=callback_stim_labels,  # Use stimulus labels
                    contrast_names=contrast_names,
                    contrast_results=ols_contrast_results,
                    volume_shape=shape_to_use,
                    affine=affine_to_use,
                    output_format=ols_output_format,
                )

                # Write single-trials output if requested (check for env var set by 3dREMLfast.py)
                import os

                single_trials_label = os.environ.get("FASTFUNCSIM_SINGLE_TRIALS")
                if single_trials_label and callback_stim_indices:
                    output_filename = f"ols_{single_trials_label}_single.nii.gz"
                    print(f"  • Writing OLS single-trial betas (onset order): {output_filename}")
                    from .glm_outputs import write_single_trials_output

                    write_single_trials_output(
                        ols_results,
                        output_filename,
                        callback_design_matrix,
                        callback_stim_indices,
                        callback_stim_labels,
                    )

                # Write partial R² if requested (check for env var set by 3dREMLfast.py)
                r2_partial_mode_env = os.environ.get("FASTFUNCSIM_R2_PARTIAL_MODE")
                if (
                    r2_partial_mode_env
                    and hasattr(ols_results, "r2_partial")
                    and ols_results.r2_partial is not None
                ):
                    # Generate output path
                    if ols_output_path:
                        if str(ols_output_path).endswith(".nii.gz"):
                            partial_r2_path = str(ols_output_path).replace(
                                ".nii.gz", "_partialR2.nii.gz"
                            )
                        elif str(ols_output_path).endswith(".nii"):
                            partial_r2_path = str(ols_output_path).replace(
                                ".nii", "_partialR2.nii.gz"
                            )
                        else:
                            partial_r2_path = str(ols_output_path) + "_partialR2.nii.gz"
                    else:
                        partial_r2_path = "OLS_partialR2.nii.gz"

                    print(f"  • Writing OLS partial R² per condition: {partial_r2_path}")

                    from .glm_outputs import (
                        _get_voxel_mask,
                        _resolve_shape,
                        write_partial_r2_with_labels,
                    )

                    # Get design info from outer scope
                    n_timepoints_ols = design_info.get("n_timepoints")
                    n_regressors_ols = design_info.get("n_regressors")

                    write_partial_r2_with_labels(
                        ols_results.r2_partial,
                        partial_r2_path,
                        condition_labels=callback_stim_labels,
                        volume_shape=_resolve_shape(ols_results, shape_to_use),
                        voxel_mask=_get_voxel_mask(ols_results),
                        affine=affine_to_use,
                        n_timepoints=n_timepoints_ols,
                        n_regressors=n_regressors_ols,
                        apply_afni_metadata=True,
                        mode=r2_partial_mode_env,  # "full" or "task"
                    )

                    suffix = "_partialR2_task" if r2_partial_mode_env == "task" else "_partialR2"
                    print("     Sub-bricks (partial R² with AFNI stat params):")
                    for idx, label in enumerate(callback_stim_labels):
                        print(f"       [{idx}] {label}{suffix}")

                # Write semi-partial R² if requested (check for env var set by 3dREMLfast.py)
                r2_semipartial_mode_env = os.environ.get("FASTFUNCSIM_R2_SEMIPARTIAL_MODE")
                if (
                    r2_semipartial_mode_env
                    and hasattr(ols_results, "r2_semipartial")
                    and ols_results.r2_semipartial is not None
                ):
                    # Generate output path
                    if ols_output_path:
                        if str(ols_output_path).endswith(".nii.gz"):
                            semipartial_r2_path = str(ols_output_path).replace(
                                ".nii.gz", "_semipartialR2.nii.gz"
                            )
                        elif str(ols_output_path).endswith(".nii"):
                            semipartial_r2_path = str(ols_output_path).replace(
                                ".nii", "_semipartialR2.nii.gz"
                            )
                        else:
                            semipartial_r2_path = str(ols_output_path) + "_semipartialR2.nii.gz"
                    else:
                        semipartial_r2_path = "OLS_semipartialR2.nii.gz"

                    print(f"  • Writing OLS semi-partial R² per condition: {semipartial_r2_path}")

                    from .glm_outputs import (
                        _get_voxel_mask,
                        _resolve_shape,
                        write_partial_r2_with_labels,
                    )

                    # Get design info from outer scope
                    n_timepoints_ols = design_info.get("n_timepoints")
                    n_regressors_ols = design_info.get("n_regressors")

                    write_partial_r2_with_labels(
                        ols_results.r2_semipartial,
                        semipartial_r2_path,
                        condition_labels=callback_stim_labels,
                        volume_shape=_resolve_shape(ols_results, shape_to_use),
                        voxel_mask=_get_voxel_mask(ols_results),
                        affine=affine_to_use,
                        n_timepoints=n_timepoints_ols,
                        n_regressors=n_regressors_ols,
                        apply_afni_metadata=True,
                        mode=r2_semipartial_mode_env,  # "full" or "task"
                    )

                    suffix = (
                        "_semipartialR2_task"
                        if r2_semipartial_mode_env == "task"
                        else "_semipartialR2"
                    )
                    print("     Sub-bricks (semi-partial R² with AFNI stat params):")
                    for idx, label in enumerate(callback_stim_labels):
                        print(f"       [{idx}] {label}{suffix}")

            ols_write_callback = write_ols

        # Prepare spatial metadata dict for fit_glm_arma11
        # This ensures OLS results have mask/shape/affine BEFORE the write callback
        spatial_metadata = {}
        if volume_shape is not None:
            spatial_metadata["volume_shape"] = volume_shape
        if mask_tensor is not None:
            spatial_metadata["voxel_mask"] = mask_tensor
        if affine is not None:
            spatial_metadata["affine"] = affine

        results = fit_glm_arma11(
            data,
            design,
            tr=tr,
            a_grid=arma_a_grid,
            b_grid=arma_b_grid,
            precomputed_arma_params=precomputed_arma_params,
            want_ols=want_ols,
            want_r2_partial=want_r2_partial,
            r2_partial_mode=r2_partial_mode,  # "full" or "task"
            want_r2_semipartial=want_r2_semipartial,
            r2_semipartial_mode=r2_semipartial_mode,  # "full" or "task"
            ols_write_callback=ols_write_callback,
            batch_size=voxel_chunk_size,  # Pass through manual override
            device=device,
            use_double=use_double,
            debug_memory=debug_memory,
            enable_quick_estimate=enable_quick_estimate,
            use_grid_batching=use_grid_batching,
            glt_labels=design_info.get("glt_labels", None),
            glt_matrices=design_info.get("glt_matrices", None),
            task_indices=stim_indices if stim_indices else None,
            spatial_metadata=spatial_metadata if spatial_metadata else None,
            legacy_contrasts=legacy_contrasts,
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ols' or 'arma11'")

    if volume_shape is not None:
        results.original_shape = volume_shape
        results.full_shape = volume_shape
        # Also set on OLS results if present
        if hasattr(results, "ols_results") and results.ols_results is not None:
            results.ols_results.original_shape = volume_shape
            results.ols_results.full_shape = volume_shape

    if mask_tensor is not None:
        results.voxel_mask = mask_tensor
        design_info["mask_file"] = str(mask_file)
        design_info["mask_threshold"] = mask_threshold
        design_info["mask_voxels"] = int(mask_tensor.sum().item())
        # Also set on OLS results if present
        if hasattr(results, "ols_results") and results.ols_results is not None:
            results.ols_results.voxel_mask = mask_tensor

    if affine is not None:
        results.affine = affine
        # Also set on OLS results if present
        if hasattr(results, "ols_results") and results.ols_results is not None:
            results.ols_results.affine = affine

    # CRITICAL: Attach NIfTI header for perfect output reconstruction
    if nifti_header is not None:
        results.nifti_header = nifti_header
        # Also set on OLS results if present
        if hasattr(results, "ols_results") and results.ols_results is not None:
            results.ols_results.nifti_header = nifti_header

    # Add scaling info to design_info for cache saving
    design_info["was_scaled"] = was_scaled
    design_info["original_mean"] = original_mean

    return results, design_info


def analyze_with_cross_validation(
    fmri_data: Union[str, Path, List[Union[str, Path]], np.ndarray, torch.Tensor],
    design_matrix_file: Union[str, Path],
    cv_strategy: Union[float, int] = 0.5,
    n_perms: int = 100,
    metric: str = "cod",
    zero_event_strategy: str = "zero",
    use_stimulus_only: bool = False,
    device: Optional[torch.device] = None,
    mask_file: Optional[Union[str, Path]] = None,
    mask_threshold: float = 0.0,
    batch_size: Optional[int] = None,
    test_n_voxels: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Union[torch.Tensor, int]], Dict]:
    """
    Cross-validated analysis pipeline: compute out-of-sample R²

    This function provides a high-level interface for cross-validation,
    following the same patterns as analyze_from_design_matrix(). It computes
    cross-validated R² which provides a more reliable measure of model
    generalization than in-sample R².

    Main use cases:
    - Testing denoising methods
    - Model selection (e.g., comparing HRF choices)
    - Evaluating preprocessing pipelines
    - Detecting overfitting

    Parameters
    ----------
    fmri_data : str, Path, list of str/Path, np.ndarray, or torch.Tensor
        fMRI data as:
        - Path to single concatenated NIfTI file (all runs in one file)
        - List of paths to NIfTI files (one per run) - will be concatenated
        - np.ndarray with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
        - torch.Tensor with shape (n_voxels, n_timepoints) or (x, y, z, n_timepoints)
    design_matrix_file : str or Path
        Path to AFNI design matrix file (X.xmat.1D)
    cv_strategy : float or int, default=0.5
        Cross-validation strategy:
        - float (0.0-1.0): Fraction for training (e.g., 0.5 = split halves)
        - int: Number of runs to leave out (e.g., 1 = leave-one-run-out)
    n_perms : int, default=100
        Number of permutations for random split strategies (ignored for leave-N-out)
    metric : str, default='cod'
        R² metric to use: 'cod', 'corr', or 'corr2'
    zero_event_strategy : str, default='zero'
        How to handle STIMULUS events that are missing across train/test runs:
        - 'zero': Use zero betas for missing events. Conservative approach.
        - 'nuisance': Treat unpredictable events as nuisance in test set.
        NOTE: This ONLY applies to stimulus events. Nuisance regressors are
        ALWAYS projected out regardless of this setting.
    use_stimulus_only : bool, default=False
        If True, only include stimulus columns (exclude nuisance regressors)
        in the design matrix for fitting. Nuisance is still projected out.
    device : torch.device, optional
        Compute device (auto-detected if None)
    mask_file : str or Path, optional
        Path to mask file (NIfTI or AFNI)
    mask_threshold : float, default=0.0
        Threshold for mask (voxels > threshold are included)
    batch_size : int, optional
        Voxels per batch for projection (auto-detected if None)
    test_n_voxels : int, optional
        If provided, only process first N voxels (for testing)
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    results : dict
        Cross-validation results:
        - 'r2_median': (n_voxels,) median R² across CV splits
        - 'r2_std': (n_voxels,) standard deviation across splits
        - 'r2_min': (n_voxels,) minimum R² across splits
        - 'r2_max': (n_voxels,) maximum R² across splits
        - 'r2_splits': (n_splits, n_voxels) R² for each split
        - 'n_splits': int, number of CV splits performed
    design_info : dict
        Design matrix metadata (from extract_design_metadata)

    Examples
    --------
    >>> # Single file with 4 runs
    >>> results, design_info = analyze_with_cross_validation(
    ...     fmri_data='all_runs.nii.gz',
    ...     design_matrix_file='X.xmat.1D',
    ...     cv_strategy=0.5,  # Split halves
    ...     n_perms=10
    ... )
    >>> print(f"Mean xval R²: {results['r2_median'].mean():.3f}")

    >>> # Multiple run files with leave-one-run-out
    >>> results, design_info = analyze_with_cross_validation(
    ...     fmri_data=['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    ...     design_matrix_file='X.xmat.1D',
    ...     cv_strategy=1,  # Leave-one-run-out
    ... )
    >>> print(f"LORO R²: {results['r2_median'].mean():.3f}")

    Notes
    -----
    - Requires at least 2 runs (checked automatically)
    - Uses run-based splits (respects temporal structure)
    - Nuisance regressors are projected out before computing R²
    - Results are CPU tensors for easy saving/analysis
    """
    from .afni_io import extract_design_metadata, read_afni_design_matrix
    from .xval import compute_xval_r2, generate_cv_splits

    # Auto-detect device
    if device is None:
        device = get_device()

    if verbose:
        print("=" * 80)
        print("CROSS-VALIDATED GLM ANALYSIS")
        print("=" * 80)
        print()

    # 1. Load design matrix
    if verbose:
        print(f"📋 Loading design matrix: {design_matrix_file}")
    design_matrix_path = Path(design_matrix_file)
    if not design_matrix_path.exists():
        raise FileNotFoundError(f"Design matrix not found: {design_matrix_path}")

    design_info = read_afni_design_matrix(design_matrix_path)
    design_matrix = design_info["matrix"]

    # Extract metadata
    run_starts = design_info["run_starts"]
    n_runs = len(run_starts)

    if n_runs < 2:
        raise ValueError(
            f"Cross-validation requires at least 2 runs, found {n_runs}. "
            "Check the RunStart parameter in your design matrix."
        )

    if verbose:
        print(f"  • Runs: {n_runs}")
        print(f"  • Timepoints: {design_matrix.shape[0]}")
        print(f"  • Regressors: {design_matrix.shape[1]}")
        print()

    # Get regressor indices
    # For cross-validation, we ALWAYS separate stimulus from nuisance
    # (nuisance = run-specific regressors like polynomials that are zero in other runs)
    full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)
    nuisance_indices = [i for i in range(len(full_labels)) if i not in stim_indices]

    if not stim_indices:
        raise ValueError(
            "No stimulus indices found in design matrix. Cannot perform cross-validation without stimulus regressors."
        )

    # use_stimulus_only now controls which columns are used for fitting/prediction
    # (but nuisance is ALWAYS projected out first)
    if not use_stimulus_only:
        # Include all regressors in the fit (both stimulus and nuisance columns)
        # Nuisance will still be projected out first, then all columns used for prediction
        stim_indices = list(range(design_matrix.shape[1]))

    # 2. Generate CV splits
    if verbose:
        print(f"🔀 Generating CV splits (strategy={cv_strategy})...")
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    if verbose:
        print(f"  • Generated {len(cv_splits)} splits")
        print()

    # 3. Load fMRI data
    if verbose:
        print("📂 Loading fMRI data...")

    # Handle different input types
    if isinstance(fmri_data, list):
        # Multiple run files
        data, actual_run_starts = load_and_concatenate_runs(fmri_data, device=torch.device("cpu"))
        if verbose:
            print(f"  • Loaded {len(fmri_data)} run files")
    elif isinstance(fmri_data, (str, Path)):
        # Single file
        img = nib.load(str(fmri_data))
        data_np = img.get_fdata()  # type: ignore[attr-defined]
        data_np = data_np.reshape(-1, data_np.shape[-1])
        data = torch.from_numpy(data_np).float()
        if verbose:
            print(f"  • Loaded single file: {fmri_data}")
    elif isinstance(fmri_data, np.ndarray):
        # NumPy array
        if fmri_data.ndim == 4:
            fmri_data = fmri_data.reshape(-1, fmri_data.shape[-1])
        data = torch.from_numpy(fmri_data).float()
    elif isinstance(fmri_data, torch.Tensor):
        # Already a tensor
        if fmri_data.ndim == 4:
            fmri_data = fmri_data.reshape(-1, fmri_data.shape[-1])
        data = fmri_data
    else:
        raise TypeError(f"Unsupported fmri_data type: {type(fmri_data)}")

    if verbose:
        print(f"  • Shape: {data.shape} (voxels × timepoints)")
        print()

    # Validate timepoints match design matrix
    if data.shape[1] != design_matrix.shape[0]:
        raise ValueError(
            f"Timepoint mismatch: fMRI data has {data.shape[1]} timepoints, "
            f"design matrix has {design_matrix.shape[0]}"
        )

    # 4. Apply mask if provided
    mask_indices = None
    if mask_file is not None:
        if verbose:
            print(f"🎭 Applying mask: {mask_file}")
        mask_img = nib.load(str(mask_file))
        mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
        mask_bool = mask_data.flatten() > mask_threshold
        mask_indices = np.where(mask_bool)[0]
        data = data[mask_indices]
        if verbose:
            print(f"  • Masked voxels: {data.shape[0]:,} / {mask_bool.size:,}")
            print()

    # 5. Test mode: limit voxels
    if test_n_voxels is not None:
        data = data[:test_n_voxels]
        if verbose:
            print(f"⚠️  TEST MODE: Using first {test_n_voxels:,} voxels only")
            print()

    # 6. Compute cross-validated R²
    if verbose:
        print("🔬 Computing cross-validated R²...")
        print()

    xval_results = compute_xval_r2(
        data=data,
        design_matrix=design_matrix,
        run_starts=run_starts,
        stim_indices=stim_indices,
        nuisance_indices=nuisance_indices,
        cv_splits=cv_splits,
        metric=metric,
        zero_event_strategy=zero_event_strategy,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )

    # 7. Add metadata to results
    xval_results["mask_indices"] = mask_indices
    xval_results["cv_strategy"] = cv_strategy
    xval_results["metric"] = metric

    # Add to design_info for consistency with analyze_from_design_matrix
    design_info["cv_strategy"] = cv_strategy
    design_info["n_splits"] = xval_results["n_splits"]

    if verbose:
        print()
        print("=" * 80)
        print("✅ CROSS-VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Mean xval R² ({metric}): {xval_results['r2_median'].mean():.4f}")
        print(f"Std xval R² ({metric}): {xval_results['r2_std'].mean():.4f}")
        print()

    return xval_results, design_info


def _load_fmri_data(
    fmri_data: Union[str, Path, np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """
    Load fMRI data from various formats

    Parameters
    ----------
    fmri_data : str, Path, np.ndarray, or torch.Tensor
        fMRI data
    device : torch.device
        Device for tensor

    Returns
    -------
    data : torch.Tensor
        fMRI data as torch tensor
    """
    if isinstance(fmri_data, (str, Path)):
        # Load from NIfTI file
        filepath = Path(fmri_data)

        if not filepath.exists():
            raise FileNotFoundError(f"fMRI data file not found: {filepath}")

        img = nib.load(str(filepath))
        data_np = img.get_fdata(dtype=np.float32)

        # Convert to tensor
        data = to_tensor(data_np, device=device, dtype=torch.float32)

    elif isinstance(fmri_data, np.ndarray):
        # Convert numpy to tensor
        data = to_tensor(fmri_data, device=device, dtype=torch.float32)

    elif isinstance(fmri_data, torch.Tensor):
        # Already a tensor, just move to device
        data = fmri_data.to(device)

    else:
        raise TypeError(f"Unsupported fmri_data type: {type(fmri_data)}")

    return data


def compute_contrasts(
    results: Union[GLMResults, ARMA11Results],
    contrasts: Union[np.ndarray, torch.Tensor, List[float]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute contrast t-statistics and p-values

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        GLM results from fit_glm() or fit_glm_arma11()
    contrasts : np.ndarray, torch.Tensor, or list
        Contrast vector(s) with shape (n_regressors,) or (n_contrasts, n_regressors)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    contrast_results : dict
        Dictionary containing:
        - 'contrast_betas': Contrast effect sizes (n_voxels,) or (n_voxels, n_contrasts)
        - 'contrast_tstats': Contrast t-statistics
        - 'contrast_stderr': Contrast standard errors

    Examples
    --------
    >>> # Single contrast: condition 1 > condition 2
    >>> contrast = [1, -1, 0, 0, 0]  # First two regressors are conditions
    >>> contrast_results = compute_contrasts(results, contrast)
    >>> print(f"Mean t-stat: {contrast_results['contrast_tstats'].mean():.3f}")
    """
    if device is None:
        device = get_device()

    # Convert contrast to tensor
    if isinstance(contrasts, list):
        contrasts = np.array(contrasts)

    contrasts = to_tensor(contrasts, device=device, dtype=torch.float32)

    # Ensure contrast is 2D
    if contrasts.ndim == 1:
        contrasts = contrasts.unsqueeze(0)  # (1, n_regressors)

    n_contrasts, n_regressors = contrasts.shape

    # Get betas and variance
    betas = to_tensor(results.betas, device=device, dtype=torch.float32)
    sigma2 = to_tensor(results.sigma2, device=device, dtype=torch.float32)

    # Compute contrast betas
    # (n_voxels, n_regressors) @ (n_regressors, n_contrasts) = (n_voxels, n_contrasts)
    contrast_betas = betas @ contrasts.T

    # Compute contrast variance
    if hasattr(results, "xtx_inv"):
        # OLS results: Var(c'β) = c' (X'X)^-1 c * σ²
        xtx_inv = to_tensor(
            results.xtx_inv, device=device, dtype=torch.float32
        )  # (n_regressors, n_regressors)

        # c' (X'X)^-1 c for each contrast
        # (n_contrasts, n_regressors) @ (n_regressors, n_regressors) @ (n_regressors, n_contrasts)
        contrast_var_factor = torch.sum(contrasts @ xtx_inv * contrasts, dim=1)  # (n_contrasts,)

        # Broadcast sigma2 and contrast_var_factor
        # (n_voxels, 1) * (1, n_contrasts) = (n_voxels, n_contrasts)
        contrast_var = sigma2.unsqueeze(1) * contrast_var_factor.unsqueeze(0)

    elif hasattr(results, "var_betas") and results.var_betas is not None:
        # ARMA(1,1) results: Var(c'β) = c' Cov(β) c
        # var_betas is (n_voxels, n_regressors, n_regressors)
        var_betas = to_tensor(results.var_betas, device=device, dtype=torch.float32)

        # For each contrast and each voxel: c' Cov(β) c
        # contrasts: (n_contrasts, n_regressors)
        # var_betas: (n_voxels, n_regressors, n_regressors)

        n_voxels = var_betas.shape[0]
        contrast_var = torch.zeros((n_voxels, n_contrasts), device=device)

        for c_idx in range(n_contrasts):
            c = contrasts[c_idx]  # (n_regressors,)
            # For each voxel: c' @ var_betas[v] @ c
            # c: (1, n_regressors)
            # var_betas[v]: (n_regressors, n_regressors)
            # Result: scalar for each voxel
            c_expanded = c.unsqueeze(0).unsqueeze(0)  # (1, 1, n_regressors)
            c_var_betas = torch.bmm(
                c_expanded.expand(n_voxels, -1, -1), var_betas
            )  # (n_voxels, 1, n_regressors)
            contrast_var[:, c_idx] = torch.bmm(
                c_var_betas, c.unsqueeze(0).unsqueeze(2).expand(n_voxels, -1, -1)
            ).squeeze()

    else:
        raise ValueError(
            "Results must have either 'xtx_inv' (OLS) or 'var_betas' (ARMA) for contrast computation. "
            "For ARMA results, var_betas was not computed (disabled by default to save memory). "
            "To enable GLT contrasts with many regressors, var_betas computation needs to be implemented on-demand."
        )

    contrast_stderr = torch.sqrt(torch.clamp(contrast_var, min=0.0))

    # Compute t-statistics
    contrast_tstats = contrast_betas / contrast_stderr

    # Squeeze if single contrast
    if n_contrasts == 1:
        contrast_betas = contrast_betas.squeeze(1)
        contrast_tstats = contrast_tstats.squeeze(1)
        contrast_stderr = contrast_stderr.squeeze(1)

    return {
        "contrast_betas": contrast_betas.cpu(),
        "contrast_tstats": contrast_tstats.cpu(),
        "contrast_stderr": contrast_stderr.cpu(),
    }


def compute_contrasts_from_design(
    results: Union[GLMResults, ARMA11Results],
    design_info: dict,
    device: Optional[torch.device] = None,
    auto_cpu_fallback: bool = True,
    memory_threshold_timepoints: int = 1000,
) -> Optional[dict]:
    """
    Compute contrasts from AFNI design matrix metadata with automatic CPU fallback.

    Extracts GLT (General Linear Test) contrast definitions from design_info
    and computes contrast statistics. For large datasets (many timepoints),
    automatically falls back to CPU computation to avoid GPU OOM errors.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        GLM results from fit_glm() or fit_glm_arma11()
    design_info : dict
        Output from read_afni_design_matrix() containing:
        - 'glt_labels': list of str (contrast names)
        - 'glt_matrices': list of arrays (contrast weight vectors)
        - 'n_regressors': int (total number of regressors)
        - 'n_timepoints': int (optional, for auto fallback detection)
    device : torch.device, optional
        Device for computation. Overridden if auto_cpu_fallback triggers.
    auto_cpu_fallback : bool, default=True
        Automatically use CPU for large datasets to avoid OOM
    memory_threshold_timepoints : int, default=1000
        Threshold for auto CPU fallback (timepoints > threshold → CPU)

    Returns
    -------
    contrast_results : dict or None
        Dictionary with keys 'contrast_betas', 'contrast_tstats', 'contrast_stderr',
        or None if no contrasts are defined in design_info

    Examples
    --------
    >>> # Load design and compute contrasts automatically
    >>> design_info = ffs.read_afni_design_matrix('X.xmat.1D')
    >>> results = ffs.fit_glm_arma11(data, design_info['matrix'], tr=2.0)
    >>> contrast_results = ffs.compute_contrasts_from_design(results, design_info)
    >>> if contrast_results:
    ...     print(f"Computed {len(design_info['glt_labels'])} contrasts")
    ...     for name, tstat in zip(design_info['glt_labels'],
    ...                            contrast_results['contrast_tstats'].T):
    ...         print(f"  {name}: mean t = {tstat.mean():.3f}")

    Notes
    -----
    - Returns None if design_info contains no GLT definitions
    - For ARMA results with >1000 timepoints, automatically uses CPU to avoid
      GPU OOM (var_betas matrix can be 20+ GB)
    - Contrasts are extracted directly from X.xmat.1D metadata, no manual
      specification needed
    - Works with both OLS (xtx_inv) and ARMA (var_betas) results
    """
    if device is None:
        device = get_device()

    # Check if any contrasts are defined
    if not design_info.get("glt_labels") or not design_info.get("glt_matrices"):
        return None

    # Auto CPU fallback for large datasets
    if auto_cpu_fallback:
        n_timepoints = design_info.get("n_timepoints", 0)
        if n_timepoints > memory_threshold_timepoints:
            print(
                f"  ⚠ Large dataset ({n_timepoints} timepoints): "
                "computing contrasts on CPU (var_betas too large for GPU)"
            )
            device = torch.device("cpu")

    # Stack all GLT matrices into a single tensor
    n_regressors = design_info["n_regressors"]
    n_contrasts = len(design_info["glt_matrices"])

    contrasts = torch.zeros((n_contrasts, n_regressors), device=device, dtype=torch.float32)

    for i, glt_matrix in enumerate(design_info["glt_matrices"]):
        # Convert numpy array to torch tensor if needed
        glt_tensor = torch.as_tensor(glt_matrix, device=device, dtype=torch.float32)
        # GLT matrices are typically 1D (one row per contrast)
        if glt_tensor.ndim == 1:
            contrasts[i, :] = glt_tensor
        else:
            # If 2D, take the first row (single contrast per GLT)
            contrasts[i, :] = glt_tensor[0, :]

    # Compute contrasts
    return compute_contrasts(results, contrasts, device=device)
