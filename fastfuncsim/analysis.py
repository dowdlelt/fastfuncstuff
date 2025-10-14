"""
High-level analysis workflows for fMRI data
Complete pipelines from AFNI files to GLM results
"""

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
from .design import build_glm_design
from .glm_core import GLMResults, fit_glm, fit_glm_hrf_library
from .hrf import get_canonical_hrf, get_hrf_library
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
        HRF mode: 'canonical', 'flobs', 'library', or 'fir'
        - 'canonical': Single SPM canonical HRF (assumed HRF)
        - 'library': Fit library of HRF variants, pick best per voxel
        - 'flobs': Use FLOBS HRF library
        - 'fir': Finite impulse response (no HRF assumption)
    stim_duration : float
        Stimulus duration in seconds (for HRF convolution)
    n_hrfs : int
        Number of HRFs in library (for 'library' or 'flobs' modes)
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
            f"Data must be 2D (n_voxels, n_timepoints) or 4D (x, y, z, t), "
            f"got shape {data.shape}"
        )

    n_voxels, n_timepoints = data.shape

    # 2. Read onset files and convert to binary matrix
    onset_data = read_afni_onset_files(onset_files)
    onsets = onsets_to_binary_matrix(
        onset_data, n_timepoints, tr, run_starts=run_starts, device=device
    )

    # 3. Build design matrix based on HRF mode
    if hrf_mode == "fir":
        # FIR: no HRF assumption
        design = build_glm_design(
            onsets,
            hrf=None,
            n_timepoints=n_timepoints,
            mode="fir",
            device=device,
            **hrf_kwargs,
        )

    elif hrf_mode == "canonical":
        # Single canonical HRF
        hrf = get_canonical_hrf(stim_duration=stim_duration, tr=tr, device=device)
        design = build_glm_design(
            onsets, hrf=hrf, n_timepoints=n_timepoints, mode="assumed", device=device
        )

    elif hrf_mode in ["library", "flobs"]:
        # HRF library - will fit with library below
        hrf_library = get_hrf_library(
            mode="canonical" if hrf_mode == "library" else "flobs",
            stim_duration=stim_duration,
            tr=tr,
            n_hrfs=n_hrfs,
            device=device,
            **hrf_kwargs,
        )
        # For library mode, we'll use fit_glm_hrf_library below
        # For now, just flag it
        design = None

    else:
        raise ValueError(
            f"Unknown hrf_mode: {hrf_mode}. "
            f"Choose 'canonical', 'library', 'flobs', or 'fir'"
        )

    # 4. Fit GLM
    if method == "ols":
        if hrf_mode in ["library", "flobs"]:
            # Use HRF library fitting
            results, hrf_idx, r2_all = fit_glm_hrf_library(
                data, onsets, hrf_library, tr=tr, device=device
            )
            # Store HRF indices in results
            results.hrf_idx = hrf_idx
            results.r2_per_hrf = r2_all

        else:
            # Standard OLS
            results = fit_glm(data, design, tr=tr, device=device)

    elif method == "arma11":
        if hrf_mode in ["library", "flobs"]:
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
        results = fit_glm_arma11(
            data, design, tr=tr, a_grid=arma_a_grid, b_grid=arma_b_grid, device=device
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ols' or 'arma11'")

    if original_shape is not None:
        results.original_shape = original_shape
        results.full_shape = original_shape  # type: ignore[attr-defined]

    return results


def analyze_from_design_matrix(
    fmri_data: Union[str, Path, List[Union[str, Path]], np.ndarray, torch.Tensor],
    design_matrix_file: Union[str, Path],
    method: str = "ols",
    use_stimulus_only: bool = False,
    arma_a_grid: Optional[torch.Tensor] = None,
    arma_b_grid: Optional[torch.Tensor] = None,
    precomputed_arma_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Optional[torch.device] = None,
    mask_file: Optional[Union[str, Path]] = None,
    mask_threshold: float = 0.0,
    voxel_chunk_size: Optional[int] = None,
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
    precomputed_arma_params : array-like, shape (n_voxels, 2), optional
        Precomputed ARMA(1,1) parameters [a, b] for each voxel (only used if method='arma11')
        If provided, skips REML estimation (saves ~80% of compute time)
        Useful for re-running analysis with different contrasts or validating against AFNI
    device : torch.device, optional
        Device for computation
    mask_file : str or Path, optional
        Optional NIfTI mask file used to restrict analysis to a subset of voxels
    mask_threshold : float
        Threshold applied to mask values; voxels greater than this are included
    voxel_chunk_size : int, optional
        Number of voxels to process per chunk. Overrides automatic heuristic

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

        data, actual_run_starts = load_and_concatenate_runs(fmri_data, storage_device)

        # Validate run_starts match if both are available
        if expected_run_starts is not None:
            if len(actual_run_starts) != len(expected_run_starts):
                raise ValueError(
                    f"Number of run files ({len(actual_run_starts)}) doesn't match "
                    f"number of runs in design matrix ({len(expected_run_starts)})"
                )
            # Check that run lengths match
            expected_lengths = get_run_lengths(
                expected_run_starts, design_info["n_timepoints"]
            )
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

    if mask_file is not None:
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
        if kept_voxels == 0:
            raise ValueError(
                f"Mask '{mask_file}' excluded all voxels (threshold={mask_threshold})."
            )

        data = data[mask_tensor, :]
        n_voxels = kept_voxels

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

        results = fit_glm_arma11(
            data,
            design,
            tr=tr,
            a_grid=arma_a_grid,
            b_grid=arma_b_grid,
            precomputed_arma_params=precomputed_arma_params,
            device=device,
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ols' or 'arma11'")

    if volume_shape is not None:
        results.original_shape = volume_shape
        results.full_shape = volume_shape  # type: ignore[attr-defined]

    if mask_tensor is not None:
        results.voxel_mask = mask_tensor  # type: ignore[attr-defined]
        design_info["mask_file"] = str(mask_file)
        design_info["mask_threshold"] = mask_threshold
        design_info["mask_voxels"] = int(mask_tensor.sum().item())

    if affine is not None:
        results.affine = affine  # type: ignore[attr-defined]

    return results, design_info


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
        xtx_inv = results.xtx_inv  # (n_regressors, n_regressors)

        # c' (X'X)^-1 c for each contrast
        # (n_contrasts, n_regressors) @ (n_regressors, n_regressors) @ (n_regressors, n_contrasts)
        contrast_var_factor = torch.sum(
            contrasts @ xtx_inv * contrasts, dim=1
        )  # (n_contrasts,)

        # Broadcast sigma2 and contrast_var_factor
        # (n_voxels, 1) * (1, n_contrasts) = (n_voxels, n_contrasts)
        contrast_var = sigma2.unsqueeze(1) * contrast_var_factor.unsqueeze(0)

    elif hasattr(results, "var_betas"):
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
            "Results must have either 'xtx_inv' (OLS) or 'var_betas' (ARMA) for contrast computation"
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
