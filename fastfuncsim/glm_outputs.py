"""Utilities for exporting GLM analysis results to neuroimaging formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

try:
    import nibabel as nib
except (
    ImportError
) as exc:  # pragma: no cover - nibabel should be installed alongside package
    raise ImportError(
        "nibabel is required to write GLM results as NIfTI files. Install it with `pip install nibabel`."
    ) from exc

from .arma_glm import ARMA11Results
from .glm_core import GLMResults

ResultsLike = Union[GLMResults, ARMA11Results]


def extract_onset_times_from_design(design_matrix: np.ndarray, column_indices: List[int]) -> List[int]:
    """
    Extract onset times for stimulus columns from design matrix.

    For each stimulus column, finds the first timepoint where the column becomes non-zero.
    This represents the onset time of that stimulus.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix (n_timepoints, n_regressors)
    column_indices : list of int
        Column indices to extract onset times for

    Returns
    -------
    onset_times : list of int
        Onset timepoint for each column (same length as column_indices)
    """
    onset_times = []

    for col_idx in column_indices:
        column = design_matrix[:, col_idx]

        # Find first non-zero timepoint
        nonzero_indices = np.nonzero(column)[0]

        if len(nonzero_indices) > 0:
            onset_time = int(nonzero_indices[0])
        else:
            # Column is all zeros - use large value to sort to end
            onset_time = len(column) + col_idx  # Add col_idx to maintain stable sort

        onset_times.append(onset_time)

    return onset_times


def write_single_trials_output(
    results: ResultsLike,
    output_path: Union[str, Path],
    design_matrix: np.ndarray,
    stim_indices: List[int],
    stim_labels: Optional[List[str]],
) -> Path:
    """
    Write single-trial betas reordered by presentation time.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        Results object with betas attribute
    output_path : str or Path
        Output file path (e.g., "ols_single.nii.gz")
    design_matrix : np.ndarray
        Full design matrix (n_timepoints, n_regressors)
    stim_indices : list of int
        Column indices for stimulus regressors
    stim_labels : list of str, optional
        Labels for stimulus regressors

    Returns
    -------
    output_path : Path
        Path to written file
    """
    # Extract onset times for each stimulus column
    onset_times = extract_onset_times_from_design(design_matrix, stim_indices)

    # Create sort order (sorts by onset time, maintaining stable order for ties)
    sort_indices = sorted(range(len(onset_times)), key=lambda i: (onset_times[i], i))

    # Reorder betas by onset time
    betas_np = _ensure_numpy(results.betas)
    betas_reordered = betas_np[:, sort_indices]  # (n_voxels, n_stimuli)

    # Reorder labels
    labels_reordered = [stim_labels[i] for i in sort_indices] if stim_labels else None

    # Reshape to volume
    affine = getattr(results, "affine", np.eye(4))
    volume_shape = _resolve_shape(results, None)
    voxel_mask = _get_voxel_mask(results)
    betas_vol = _reshape_parameter_map(betas_reordered, volume_shape, voxel_mask)

    # Write NIfTI file with complete header preservation
    tr = getattr(results, "tr", None)
    img = _create_nifti_with_header(betas_vol, affine, results, tr)
    output_path = Path(output_path)
    # Ensure .nii.gz extension
    if not str(output_path).endswith('.nii.gz'):
        if str(output_path).endswith('.nii'):
            output_path = Path(str(output_path) + '.gz')
        else:
            output_path = Path(str(output_path) + '.nii.gz')
    nib.save(img, str(output_path))

    # Write labels as JSON sidecar
    if labels_reordered:
        # Strip image extensions (.nii.gz or .nii) to get basename, then add .json
        path_str = str(output_path)
        if path_str.endswith('.nii.gz'):
            json_path = Path(path_str[:-7] + '.json')  # Remove .nii.gz
        elif path_str.endswith('.nii'):
            json_path = Path(path_str[:-4] + '.json')  # Remove .nii
        else:
            json_path = output_path.with_suffix('.json')  # Fallback
        with json_path.open('w') as f:
            json.dump({
                "Description": "Single-trial betas reordered by presentation time (onset order)",
                "Labels": labels_reordered,
                "OnsetTimes": [onset_times[i] for i in sort_indices],
                "OriginalColumnIndices": [stim_indices[i] for i in sort_indices],
            }, f, indent=2)

    return output_path


def _ensure_numpy(array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensors to numpy arrays (float32)."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.dtype != np.float32:
        return array.astype(np.float32, copy=False)
    return array


def _resolve_shape(
    results: ResultsLike, volume_shape: Optional[Sequence[int]]
) -> Sequence[int]:
    if results.original_shape is not None:
        return results.original_shape
    full_shape = getattr(results, "full_shape", None)
    if full_shape is not None:
        return tuple(int(dim) for dim in full_shape)
    if volume_shape is None:
        raise ValueError(
            "GLM results do not contain spatial shape information. Provide `volume_shape=(nx, ny, nz)`"
        )
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must be a 3-tuple (nx, ny, nz)")
    return tuple(int(dim) for dim in volume_shape)


def slice_glm_results(
    results: ResultsLike,
    indices: Union[List[int], np.ndarray, torch.Tensor],
) -> ResultsLike:
    """
    Create a new results object with only selected regressors.

    Works with both GLMResults (OLS) and ARMA11Results objects. Preserves
    spatial metadata and handles both torch tensors and numpy arrays.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        Original GLM results to slice
    indices : list of int, or array
        Regressor indices to keep (0-indexed)

    Returns
    -------
    sliced : GLMResults or ARMA11Results
        New results object with only selected regressors

    Examples
    --------
    >>> # Extract only stimulus regressors (columns 0-3)
    >>> stim_results = slice_glm_results(results, [0, 1, 2, 3])
    >>>
    >>> # Use with AFNI design info
    >>> design_info = ffs.read_afni_design_matrix('X.xmat.1D')
    >>> stim_indices = design_info['stim_bots']
    >>> stim_results = slice_glm_results(results, stim_indices)

    Notes
    -----
    - Scalar attributes (R², sigma², ARMA params) are copied unchanged
    - Regressor-specific attributes (betas, tstats) are sliced
    - Covariance matrices (var_betas, xtx_inv) are sliced in both dimensions
    - Time-series attributes (residuals, predicted) are copied unchanged
    - Uses .clone() for torch tensors and .copy() for numpy arrays to avoid aliasing
    """
    import torch

    # Convert indices to list for consistent indexing
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy().tolist()
    elif isinstance(indices, np.ndarray):
        indices = indices.tolist()

    # Create new instance of same type
    sliced = type(results)()

    # Copy scalar attributes (common to both GLMResults and ARMA11Results)
    sliced.r2 = results.r2
    sliced.sigma2 = results.sigma2
    sliced.dof = results.dof
    sliced.tr = results.tr

    # Copy spatial metadata
    if hasattr(results, "voxel_mask"):
        sliced.voxel_mask = results.voxel_mask
    if hasattr(results, "full_shape"):
        sliced.full_shape = results.full_shape
    if hasattr(results, "affine"):
        sliced.affine = results.affine
    if hasattr(results, "original_shape"):
        sliced.original_shape = results.original_shape

    # Copy ARMA-specific attributes (only if present)
    if hasattr(results, "arma_params"):
        sliced.arma_params = results.arma_params
    if hasattr(results, "arma_lambda"):
        sliced.arma_lambda = results.arma_lambda
    if hasattr(results, "reml_likelihood"):
        sliced.reml_likelihood = results.reml_likelihood

    # Slice regressor-specific attributes
    # IMPORTANT: Use .clone() for tensors and .copy() for arrays to create independent copies
    if results.betas is not None:
        if torch.is_tensor(results.betas):
            sliced.betas = results.betas[:, indices].clone()
        else:
            sliced.betas = results.betas[:, indices].copy()

    if results.tstats is not None:
        if torch.is_tensor(results.tstats):
            sliced.tstats = results.tstats[:, indices].clone()
        else:
            sliced.tstats = results.tstats[:, indices].copy()

    if results.fstats is not None:
        # F-stat is for ALL regressors - keep original for now
        # (Recomputing F-stat for subset would require re-fitting)
        sliced.fstats = results.fstats

    # Slice covariance matrices (regressor-specific in both dimensions)
    if hasattr(results, "var_betas") and results.var_betas is not None:
        # ARMA results: var_betas is (n_voxels, n_regressors, n_regressors)
        if torch.is_tensor(results.var_betas):
            sliced.var_betas = results.var_betas[:, indices, :][:, :, indices].clone()
        else:
            sliced.var_betas = results.var_betas[:, indices, :][:, :, indices].copy()

    if hasattr(results, "xtx_inv") and results.xtx_inv is not None:
        # OLS results: xtx_inv is (n_regressors, n_regressors)
        if torch.is_tensor(results.xtx_inv):
            sliced.xtx_inv = results.xtx_inv[indices, :][:, indices].clone()
        else:
            sliced.xtx_inv = results.xtx_inv[indices, :][:, indices].copy()

    # Copy time-series attributes (not regressor-specific)
    if hasattr(results, "residuals"):
        sliced.residuals = results.residuals
    if hasattr(results, "predicted"):
        sliced.predicted = results.predicted
    if hasattr(results, "residuals_whitened"):
        sliced.residuals_whitened = results.residuals_whitened

    # Handle OLS results embedded in ARMA results
    if hasattr(results, "ols_results") and results.ols_results is not None:
        sliced.ols_results = slice_glm_results(results.ols_results, indices)

    return sliced


def _build_affine(
    affine: Optional[np.ndarray], voxel_size: Sequence[float]
) -> np.ndarray:
    if affine is not None:
        return affine
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = voxel_size[0]
    mat[1, 1] = voxel_size[1]
    mat[2, 2] = voxel_size[2]
    return mat


def _get_voxel_mask(results: ResultsLike) -> Optional[np.ndarray]:
    mask = getattr(results, "voxel_mask", None)
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(mask, dtype=bool)


def _reshape_parameter_map(
    data: np.ndarray,
    volume_shape: Sequence[int],
    voxel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if voxel_mask is None:
        if data.ndim == 2:
            return data.reshape(*volume_shape, data.shape[1])
        if data.ndim == 1:
            return data.reshape(*volume_shape)
        return data

    mask_flat = voxel_mask.reshape(-1)
    flat_size = int(np.prod(volume_shape))

    if data.ndim == 2:
        n_maps = data.shape[1]
        vol = np.zeros((flat_size, n_maps), dtype=data.dtype)
        vol[mask_flat] = data
        return vol.reshape(*volume_shape, n_maps)

    if data.ndim == 1:
        vol = np.zeros(flat_size, dtype=data.dtype)
        vol[mask_flat] = data
        return vol.reshape(*volume_shape)

    raise ValueError("Data must be 1D or 2D when using a voxel mask")


def _strip_imaging_extension(filepath: str) -> str:
    """
    Strip AFNI/NIfTI extension from filepath, preserving periods in base name.

    Examples
    --------
    >>> _strip_imaging_extension("stats.blur.2mm+tlrc.HEAD")
    'stats.blur.2mm'
    >>> _strip_imaging_extension("errts.sub-01.nii.gz")
    'errts.sub-01'
    """
    EXTENSIONS = [
        '+orig.BRIK.gz', '+tlrc.BRIK.gz',
        '+orig.BRIK', '+tlrc.BRIK',
        '+orig.HEAD', '+tlrc.HEAD',
        '.nii.gz',
        '.nii',
    ]

    for ext in EXTENSIONS:
        if filepath.endswith(ext):
            return filepath[:-len(ext)]

    return filepath


def _normalize_output_path(output_path: Union[str, Path]) -> tuple[Path, str]:
    """
    Normalize output path and detect format from extension.

    IMPORTANT: Preserves periods in filenames (e.g., "stats.blur.2mm")
    Uses _strip_imaging_extension() instead of Path.with_suffix()

    Returns
    -------
    path : Path
        Normalized path object
    format : str
        File format: 'nifti', 'nifti_gz', or 'afni'
    """
    output_path = Path(output_path)
    path_str = str(output_path)

    # Detect format from extension
    if path_str.endswith(('.HEAD', '.head')):
        base = _strip_imaging_extension(path_str)
        return Path(base), "afni"
    elif path_str.endswith(('.BRIK.gz', '.brik.gz', '.BRIK', '.brik')):
        base = _strip_imaging_extension(path_str)
        return Path(base), "afni"
    elif path_str.endswith(('.nii.gz', '.NII.GZ')):
        return output_path, "nifti_gz"
    elif path_str.endswith(('.nii', '.NII')):
        return output_path, "nifti"
    else:
        # Default to NIfTI compressed - ADD extension, don't replace
        return Path(path_str + ".nii.gz"), "nifti_gz"


def _create_nifti_with_header(
    data: np.ndarray,
    affine: np.ndarray,
    results: Optional[ResultsLike] = None,
    tr: Optional[float] = None,
) -> nib.Nifti1Image:
    """
    Create NIfTI image preserving complete header from original data.

    Parameters
    ----------
    data : ndarray
        Image data (3D or 4D)
    affine : ndarray
        4x4 affine matrix
    results : GLMResults or ARMA11Results, optional
        If provided and has nifti_header attribute, uses complete header
    tr : float, optional
        TR to set in header if creating new header

    Returns
    -------
    img : Nifti1Image
        NIfTI image with complete header metadata preserved
    """
    nifti_header = getattr(results, "nifti_header", None) if results is not None else None

    if nifti_header is not None:
        # Copy the complete header from original data
        import copy
        new_header = copy.deepcopy(nifti_header)
        # Update shape to match new data
        new_header.set_data_shape(data.shape)
        img = nib.Nifti1Image(data, affine, header=new_header)
    else:
        # Fallback: Create basic header
        img = nib.Nifti1Image(data, affine)
        img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            img.header["pixdim"][4] = float(tr)

    return img


def _save_nifti_with_format(
    img: nib.Nifti1Image,
    output_path: Path,
    format: str,
    compress_brik: bool = True,
) -> Path:
    """
    Save NIfTI image in specified format (NIfTI or AFNI BRIK).

    Parameters
    ----------
    img : nibabel image
        Image to save
    output_path : Path
        Base output path (without final extension)
    format : str
        Output format: 'nifti', 'nifti_gz', or 'afni'
    compress_brik : bool
        If True and format='afni', save as .BRIK.gz (compressed)
        If False and format='afni', save as .BRIK (uncompressed)

    Returns
    -------
    final_path : Path
        Actual path to saved file (with proper extension)
    """
    if format == "afni":
        # Redirect AFNI requests to NIfTI with warning
        import warnings
        warnings.warn(
            "Direct writing of AFNI HEAD/BRIK format is not supported (cannot preserve metadata). "
            "Writing as compressed NIfTI (.nii.gz) instead. AFNI programs read NIfTI files natively.",
            UserWarning,
            stacklevel=2
        )
        # Use nifti_gz logic
        format = "nifti_gz"
        
    if format == "nifti_gz":
        # Save as compressed NIfTI
        nifti_path = (
            output_path.with_suffix(".nii.gz")
            if not str(output_path).endswith(".nii.gz")
            else output_path
        )
        nib.save(img, str(nifti_path))
        return nifti_path

    else:  # format == 'nifti'
        # Save as uncompressed NIfTI
        nifti_path = (
            output_path.with_suffix(".nii")
            if not str(output_path).endswith(".nii")
            else output_path
        )
        nib.save(img, str(nifti_path))
        return nifti_path


def write_glm_results_nifti(
    results: ResultsLike,
    output_dir: Union[str, Path],
    prefix: str = "glm",
    condition_names: Optional[Sequence[str]] = None,
    include_beta: bool = True,
    include_tstat: bool = True,
    include_fstat: bool = True,
    include_r2: bool = True,
    include_mean: bool = True,
    include_sigma: bool = False,
    write_residuals: bool = False,
    write_predictions: bool = False,
    volume_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Sequence[float] = (2.0, 2.0, 2.0),
    dtype: Union[np.dtype, str] = np.float32,
) -> dict:
    """
    Write GLM analysis outputs to NIfTI files with AFNI-style stacking of betas and t-stats.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        Output from `fit_glm` or `fit_glm_arma11`.
    output_dir : str or Path
        Destination directory; created if needed.
    prefix : str
        Base filename for generated files (default "glm").
    condition_names : sequence of str, optional
        Labels for regressors/conditions. If None, generic names are generated.
    include_beta / include_tstat / include_fstat / include_r2 / include_mean / include_sigma : bool
        Toggle which maps are written. F-statistics are appended after beta/t-stat stacks.
    write_residuals / write_predictions : bool
        If True, residual or predicted timeseries are written as 4D volumes. Requires these
        arrays to be present in `results`.
    volume_shape : tuple, optional
        Override the spatial shape when results were computed on flattened data.
    affine : np.ndarray, optional
        4x4 affine matrix for spatial orientation. Defaults to diagonal with specified voxel size.
    voxel_size : tuple of float
        Used only when `affine` is None.
    dtype : numpy dtype
        Output datatype (default float32).

    Returns
    -------
    outputs : dict
        Mapping of logical artifact names to written file paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_shape = _resolve_shape(results, volume_shape)
    affine_mat = _build_affine(affine, voxel_size)
    dtype = np.dtype(dtype)
    tr = getattr(results, "tr", None)
    voxel_mask = _get_voxel_mask(results)

    betas_np = _ensure_numpy(results.betas)
    betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)
    n_regressors = betas_vol.shape[-1] if betas_vol.ndim == 4 else 1

    if condition_names is not None and len(condition_names) != n_regressors:
        raise ValueError(
            f"Provided condition_names has length {len(condition_names)} but results contain {n_regressors} regressors"
        )
    if condition_names is None:
        condition_names = [f"cond_{i + 1:02d}" for i in range(n_regressors)]

    stats_stack: List[np.ndarray] = []
    stats_labels: List[dict] = []

    if include_beta:
        for idx, name in enumerate(condition_names):
            stats_stack.append(np.asarray(betas_vol[..., idx], dtype=dtype))
            stats_labels.append({"condition": name, "metric": "beta"})

    if include_tstat and getattr(results, "tstats", None) is not None:
        tstats_np = _ensure_numpy(results.tstats)
        tstats_vol = _reshape_parameter_map(tstats_np, volume_shape, voxel_mask)
        for idx, name in enumerate(condition_names):
            stats_stack.append(np.asarray(tstats_vol[..., idx], dtype=dtype))
            stats_labels.append({"condition": name, "metric": "tstat"})
    elif include_tstat:
        raise ValueError(
            "T-statistics requested but not present in results. Re-run GLM with t-stats enabled."
        )

    outputs = {}

    if stats_stack:
        stats_data = np.stack(stats_stack, axis=-1)
        stats_img = _create_nifti_with_header(stats_data.astype(dtype, copy=False), affine_mat, results, tr)
        stats_path = output_dir / f"{prefix}_stats.nii.gz"
        nib.save(stats_img, stats_path)
        outputs["stats"] = stats_path

        label_path = output_dir / f"{prefix}_stats.json"
        with label_path.open("w", encoding="utf-8") as f:
            json.dump({"volumes": stats_labels}, f, indent=2)
        outputs["stats_meta"] = label_path

    if include_fstat and getattr(results, "fstats", None) is not None:
        fstat_np = _ensure_numpy(results.fstats)
        fstat_vol = _reshape_parameter_map(fstat_np, volume_shape, voxel_mask)
        fstat_img = _create_nifti_with_header(fstat_vol.astype(dtype, copy=False), affine_mat, results, tr)
        fstat_path = output_dir / f"{prefix}_fstat.nii.gz"
        nib.save(fstat_img, fstat_path)
        outputs["fstat"] = fstat_path

    if include_r2 and getattr(results, "r2", None) is not None:
        r2_np = _ensure_numpy(results.r2)
        r2_vol = _reshape_parameter_map(r2_np, volume_shape, voxel_mask)
        r2_img = _create_nifti_with_header(r2_vol.astype(dtype, copy=False), affine_mat, results, tr)
        r2_path = output_dir / f"{prefix}_r2.nii.gz"
        nib.save(r2_img, r2_path)
        outputs["r2"] = r2_path

    if include_mean and getattr(results, "meanvol", None) is not None:
        mean_np = _ensure_numpy(results.meanvol)
        mean_vol = _reshape_parameter_map(mean_np, volume_shape, voxel_mask)
        mean_img = _create_nifti_with_header(mean_vol.astype(dtype, copy=False), affine_mat, results, tr)
        mean_path = output_dir / f"{prefix}_mean.nii.gz"
        nib.save(mean_img, mean_path)
        outputs["mean"] = mean_path

    if include_sigma and getattr(results, "sigma2", None) is not None:
        sigma_np = np.sqrt(np.maximum(_ensure_numpy(results.sigma2), 0.0))
        sigma_vol = _reshape_parameter_map(sigma_np, volume_shape, voxel_mask)
        sigma_img = _create_nifti_with_header(sigma_vol.astype(dtype, copy=False), affine_mat, results, tr)
        sigma_path = output_dir / f"{prefix}_sigma.nii.gz"
        nib.save(sigma_img, sigma_path)
        outputs["sigma"] = sigma_path

    if write_residuals:
        if getattr(results, "residuals", None) is None:
            raise ValueError(
                "Residuals were requested for export but are not stored in results. Set `want_residuals=True` when fitting."
            )
        resid_np = _ensure_numpy(results.residuals)
        resid_vol = _reshape_parameter_map(resid_np, volume_shape, voxel_mask)
        resid_img = _create_nifti_with_header(resid_vol.astype(dtype, copy=False), affine_mat, results, tr)
        resid_path = output_dir / f"{prefix}_residuals.nii.gz"
        nib.save(resid_img, resid_path)
        outputs["residuals"] = resid_path

    if write_predictions:
        if getattr(results, "predicted", None) is None:
            raise ValueError(
                "Predicted timecourses requested but not stored. Set `want_predicted=True` when fitting."
            )
        pred_np = _ensure_numpy(results.predicted)
        pred_vol = _reshape_parameter_map(pred_np, volume_shape, voxel_mask)
        pred_img = _create_nifti_with_header(pred_vol.astype(dtype, copy=False), affine_mat, results, tr)
        pred_path = output_dir / f"{prefix}_predicted.nii.gz"
        nib.save(pred_img, pred_path)
        outputs["predicted"] = pred_path

    return outputs


def write_afni_bucket(
    results: ResultsLike,
    output_path: Union[str, Path],
    condition_names: Optional[Sequence[str]] = None,
    contrast_names: Optional[Sequence[str]] = None,
    contrast_results: Optional[dict] = None,
    volume_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Sequence[float] = (2.0, 2.0, 2.0),
    dtype: Union[np.dtype, str] = np.float32,
    apply_afni_metadata: bool = True,
    compress_output: bool = True,
    output_format: Optional[str] = None,
) -> Path:
    """
    write_afni_bucket(results, output_path, ...)

    .. deprecated:: 1.0
        Use `write_glm_bucket_as_nifti` instead.

    See `write_glm_bucket_as_nifti` for documentation.
    """
    import warnings
    warnings.warn(
        "`write_afni_bucket` is deprecated and renamed to `write_glm_bucket_as_nifti`. "
        "It writes NIfTI files (.nii.gz) which AFNI supports. "
        "Please update your code.",
        DeprecationWarning,
        stacklevel=2
    )
    return write_glm_bucket_as_nifti(
        results, output_path, condition_names, contrast_names, contrast_results,
        volume_shape, affine, voxel_size, dtype, apply_afni_metadata,
        compress_output, output_format
    )


def write_glm_bucket_as_nifti(
    results: ResultsLike,
    output_path: Union[str, Path],
    condition_names: Optional[Sequence[str]] = None,
    contrast_names: Optional[Sequence[str]] = None,
    contrast_results: Optional[dict] = None,
    volume_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Sequence[float] = (2.0, 2.0, 2.0),
    dtype: Union[np.dtype, str] = np.float32,
    apply_afni_metadata: bool = True,
    compress_output: bool = True,
    output_format: Optional[str] = None,
) -> Path:
    """
    Write GLM results as a 4D output file (NIfTI) with AFNI-style sub-bricks.

    Creates a single 4D NIfTI file with sub-bricks in AFNI order:
    1. Overall F-statistic
    2. For each condition: Beta coefficient, then T-statistic
    3. For each contrast: Beta coefficient, then T-statistic

    This matches AFNI's 3dDeconvolve output format. The output is always NIfTI
    (.nii or .nii.gz), as AFNI reads NIfTI files natively and this preserves
    metadata better than writing .BRIK files from Python.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        GLM fitting results
    output_path : str or Path
        Output file path. If an AFNI extension (.HEAD, .BRIK) is provided,
        it will be replaced with .nii.gz and a warning will be issued.
    condition_names : sequence of str, optional
        Names for each condition/regressor
    contrast_names : sequence of str, optional
        Names for each contrast
    contrast_results : dict, optional
        Output from compute_contrasts() containing:
        - 'contrast_betas': (n_voxels, n_contrasts)
        - 'contrast_tstats': (n_voxels, n_contrasts)
    volume_shape : tuple, optional
        Spatial dimensions (nx, ny, nz)
    affine : np.ndarray, optional
        4x4 affine transformation matrix
    voxel_size : tuple of float
        Voxel dimensions in mm
    dtype : numpy dtype
        Output data type
    apply_afni_metadata : bool, default=True
        If True, automatically run 3drefit to apply AFNI metadata:
        - Sub-brick labels (-relabel_all_str)
        - Statistical parameters (-substatpar) for F-stats and t-stats
        Requires AFNI to be installed. Gracefully skips if not available.
    compress_output : bool, default=True
        If True, compress final output (.nii.gz).
    output_format : str, optional
        Ignored (always NIfTI). Retained for compatibility.

    Returns
    -------
    output_path : Path
        Path to written file (usually .nii.gz)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    volume_shape = _resolve_shape(results, volume_shape)
    # Try to get affine from results if not provided
    if affine is None and hasattr(results, "affine") and results.affine is not None:
        affine = results.affine
    affine_mat = _build_affine(affine, voxel_size)
    dtype = np.dtype(dtype)
    tr = getattr(results, "tr", None)
    voxel_mask = _get_voxel_mask(results)

    # Get betas and t-stats
    betas_np = _ensure_numpy(results.betas)
    betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)
    n_regressors = betas_vol.shape[-1] if betas_vol.ndim == 4 else 1

    # T-stats are optional - if not present, we only output betas
    has_tstats = getattr(results, "tstats", None) is not None
    tstats_vol = None
    if has_tstats:
        tstats_np = _ensure_numpy(results.tstats)
        tstats_vol = _reshape_parameter_map(tstats_np, volume_shape, voxel_mask)

    # Generate condition names if not provided
    if condition_names is None:
        condition_names = [f"cond{i + 1:02d}" for i in range(n_regressors)]
    elif len(condition_names) != n_regressors:
        raise ValueError(
            f"condition_names has length {len(condition_names)} but results have {n_regressors} regressors"
        )

    # Build sub-brick stack
    subbricks = []
    labels = []

    # 1. Overall F-statistic (first sub-brick, AFNI style) - optional
    if getattr(results, "fstats", None) is not None:
        fstat_np = _ensure_numpy(results.fstats)
        fstat_vol = _reshape_parameter_map(fstat_np, volume_shape, voxel_mask)
        subbricks.append(fstat_vol.astype(dtype, copy=False))
        labels.append("Full_Fstat")

    # 2. Beta and T-stat for each condition
    for idx, name in enumerate(condition_names):
        # Beta coefficient
        subbricks.append(betas_vol[..., idx].astype(dtype, copy=False))
        labels.append(f"{name}#0_Coef")

        # T-statistic (if available)
        if has_tstats and tstats_vol is not None:
            subbricks.append(tstats_vol[..., idx].astype(dtype, copy=False))
            labels.append(f"{name}#0_Tstat")

    # 3. Beta and T-stat for each contrast
    # Check if contrasts are in results object first (from in-loop GLT computation)
    if (
        contrast_results is None
        and hasattr(results, "contrast_betas")
        and results.contrast_betas is not None
    ):
        # Build contrast_results dict from results attributes
        contrast_results = {
            "contrast_betas": results.contrast_betas,
            "contrast_tstats": results.contrast_tstats,
        }
        # Use contrast_labels from results if available and contrast_names not provided
        if contrast_names is None and hasattr(results, "contrast_labels"):
            contrast_names = results.contrast_labels

    if contrast_results is not None:
        contrast_betas = _ensure_numpy(contrast_results["contrast_betas"])
        contrast_tstats = _ensure_numpy(contrast_results["contrast_tstats"])
        contrast_r2_partial = contrast_results.get("contrast_r2_partial", None)
        if contrast_r2_partial is not None:
            contrast_r2_partial = _ensure_numpy(contrast_r2_partial)
        contrast_r2_semipartial = contrast_results.get("contrast_r2_semipartial", None)
        if contrast_r2_semipartial is not None:
            contrast_r2_semipartial = _ensure_numpy(contrast_r2_semipartial)

        # Handle single contrast case
        if contrast_betas.ndim == 1:
            contrast_betas = contrast_betas[:, np.newaxis]
            contrast_tstats = contrast_tstats[:, np.newaxis]
            if contrast_r2_partial is not None and contrast_r2_partial.ndim == 1:
                contrast_r2_partial = contrast_r2_partial[:, np.newaxis]
            if contrast_r2_semipartial is not None and contrast_r2_semipartial.ndim == 1:
                contrast_r2_semipartial = contrast_r2_semipartial[:, np.newaxis]

        n_contrasts = contrast_betas.shape[1]

        # Generate contrast names if not provided
        if contrast_names is None:
            contrast_names = [f"contrast{i + 1:02d}" for i in range(n_contrasts)]
        elif len(contrast_names) != n_contrasts:
            raise ValueError(
                f"contrast_names has length {len(contrast_names)} but have {n_contrasts} contrasts"
            )

        for idx, name in enumerate(contrast_names):
            # Reshape contrast results
            cb_vol = _reshape_parameter_map(
                contrast_betas[:, idx], volume_shape, voxel_mask
            )
            ct_vol = _reshape_parameter_map(
                contrast_tstats[:, idx], volume_shape, voxel_mask
            )

            # Beta coefficient
            subbricks.append(cb_vol.astype(dtype, copy=False))
            labels.append(f"{name}#0_Coef")

            # T-statistic
            subbricks.append(ct_vol.astype(dtype, copy=False))
            labels.append(f"{name}#0_Tstat")

            # Partial R² (if available)
            if contrast_r2_partial is not None:
                cr2_vol = _reshape_parameter_map(
                    contrast_r2_partial[:, idx], volume_shape, voxel_mask
                )
                subbricks.append(cr2_vol.astype(dtype, copy=False))
                labels.append(f"{name}#0_R2")

            # Semi-partial R² (if available)
            if contrast_r2_semipartial is not None:
                cr2semi_vol = _reshape_parameter_map(
                    contrast_r2_semipartial[:, idx], volume_shape, voxel_mask
                )
                subbricks.append(cr2semi_vol.astype(dtype, copy=False))
                labels.append(f"{name}#0_R2semi")

    # Stack all sub-bricks
    bucket_data = np.stack(subbricks, axis=-1)

    # Create NIfTI image with complete header preservation
    bucket_img = _create_nifti_with_header(bucket_data, affine_mat, results, tr)

    # Normalize output path and detect format
    base_path, detected_format = _normalize_output_path(output_path)
    if output_format is not None:
        detected_format = output_format

    # IMPORTANT: Bucket files must be saved as NIfTI format
    # We create a Nifti1Image, and nibabel cannot convert NIfTI headers to AFNI headers
    # 3drefit works fine with NIfTI files, so always use NIfTI for buckets
    if detected_format == "afni":
        detected_format = "nifti_gz"  # Default to compressed NIfTI
        
        # Warn user
        import warnings
        warnings.warn(
            f"Requested AFNI format output '{output_path}' but direct writing of .HEAD/.BRIK is not supported. "
            f"Writing as compressed NIfTI (.nii.gz) instead, which AFNI reads natively.",
            UserWarning,
            stacklevel=2
        )
        
        # Strip AFNI extension and add .nii.gz
        base_name = _strip_imaging_extension(str(base_path))
        base_path = Path(base_name + ".nii.gz")

    # For 3drefit, we need uncompressed file first
    # Write uncompressed NIfTI first (for 3drefit)
    # Handle .nii.gz properly: strip all extensions, then add .nii
    if str(base_path).endswith(".nii.gz"):
        temp_path = base_path.parent / (base_path.name[:-7] + ".nii")
    elif str(base_path).endswith(".nii"):
        temp_path = base_path
    else:
        temp_path = base_path.with_suffix(".nii")
    nib.save(bucket_img, str(temp_path))

    # Write labels as JSON sidecar
    # Strip all suffixes (.nii, .nii.gz, .BRIK, etc.) then add .json
    json_base = base_path.parent / base_path.name.split(".")[0]
    label_path = json_base.with_suffix(".json")
    with label_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "Description": "AFNI-style bucket file from FastFuncSim GLM",
                "SubBricks": labels,
                "Order": "F-stat, then (Beta, T-stat) pairs for each condition, then (Beta, T-stat) pairs for each contrast",
                "Format": detected_format,
            },
            f,
            indent=2,
        )

    # Apply AFNI metadata with 3drefit (if requested and available)
    if apply_afni_metadata:
        import shutil
        import subprocess

        if shutil.which("3drefit"):
            try:
                # Get DoF from results
                dof = getattr(results, "dof", None)
                if dof is None:
                    print(
                        "  ⚠ Warning: No DoF found in results, skipping stat parameters"
                    )
                else:
                    # Split into two 3drefit commands to avoid buffer overflow
                    # Command 1: Set labels (write to file to avoid buffer overflow)
                    labels_file = temp_path.parent / f"{temp_path.stem}_labels.txt"
                    with labels_file.open('w') as f:
                        # Write space-separated labels (AFNI format for -relabel_all)
                        f.write(" ".join(labels))

                    cmd_relabel = ["3drefit", "-relabel_all", str(labels_file), str(temp_path)]

                    # Write relabel command to file for debugging
                    cmd_file_relabel = temp_path.parent / f"{temp_path.stem}_relabel_3drefit_cmd.txt"
                    with cmd_file_relabel.open('w') as f:
                        f.write("# 3drefit command for setting sub-brick labels\n")
                        f.write("# This file is created automatically and can be deleted\n\n")
                        f.write(f"Labels written to: {labels_file}\n\n")
                        f.write("Command as list:\n")
                        f.write(f"{cmd_relabel}\n\n")
                        f.write("Command as shell string:\n")
                        import shlex
                        shell_cmd = " ".join(shlex.quote(arg) for arg in cmd_relabel)
                        f.write(f"{shell_cmd}\n")

                    # Run relabel command
                    subprocess.run(cmd_relabel, check=True, capture_output=True, text=True)

                    # Command 2: Set statistical parameters
                    cmd_statpar = ["3drefit"]
                    brick_idx = 0

                    # F-statistic (sub-brick 0)
                    cmd_statpar.extend(
                        [
                            "-substatpar",
                            str(brick_idx),
                            "fift",
                            str(n_regressors),
                            str(dof),
                        ]
                    )
                    brick_idx += 1

                    # Regressor t-statistics
                    for _ in condition_names:
                        # Skip beta (brick_idx), add t-stat (brick_idx + 1)
                        cmd_statpar.extend(
                            ["-substatpar", str(brick_idx + 1), "fitt", str(dof)]
                        )
                        brick_idx += 2

                    # Contrast t-statistics (if any)
                    if contrast_names:
                        for _ in contrast_names:
                            cmd_statpar.extend(
                                ["-substatpar", str(brick_idx + 1), "fitt", str(dof)]
                            )
                            brick_idx += 2

                    # Add file path
                    cmd_statpar.append(str(temp_path))

                    # Write statpar command to file for debugging
                    cmd_file_statpar = temp_path.parent / f"{temp_path.stem}_statpar_3drefit_cmd.txt"
                    with cmd_file_statpar.open('w') as f:
                        f.write("# 3drefit command for setting statistical parameters\n")
                        f.write("# This file is created automatically and can be deleted\n\n")
                        f.write("Command as list:\n")
                        f.write(f"{cmd_statpar}\n\n")
                        f.write("Command as shell string:\n")
                        shell_cmd = " ".join(shlex.quote(arg) for arg in cmd_statpar)
                        f.write(f"{shell_cmd}\n")

                    # Run statpar command
                    subprocess.run(cmd_statpar, check=True, capture_output=True, text=True)

            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Warning: 3drefit failed: {e.stderr}")
            except Exception as e:
                print(f"  ⚠ Warning: 3drefit error: {e}")
        else:
            # AFNI not installed, skip silently (user opted in with apply_afni_metadata=True)
            pass

    # Compress output if requested
    final_path = temp_path
    if compress_output:
        import shutil
        import subprocess

        # Note: detected_format is always 'nifti' or 'nifti_gz' for bucket files
        # (we force it above), so we always compress as NIfTI
        # Compress NIfTI: .nii → .nii.gz
        # IMPORTANT: Strip .nii extension properly, preserving periods in filename
        base_name = _strip_imaging_extension(str(temp_path))
        compressed_path = Path(base_name + ".nii.gz")

        # OPTIMIZATION: Use pigz (parallel gzip) if available - much faster!
        # For large files (e.g., 870k voxels), pigz can be 4-8× faster than gzip
        if shutil.which("pigz"):
            try:
                # pigz: parallel gzip, uses all CPU cores
                subprocess.run(
                    ["pigz", "-f", str(temp_path)],  # -f: force overwrite
                    check=True,
                    capture_output=True
                )
                # pigz creates .nii.gz automatically
                # Rename if needed (pigz adds .gz to existing name)
                pigz_output = Path(str(temp_path) + ".gz")
                if pigz_output != compressed_path:
                    pigz_output.rename(compressed_path)
                final_path = compressed_path
            except subprocess.CalledProcessError:
                # pigz failed, fall back to gzip
                import gzip
                with open(temp_path, "rb") as f_in:
                    with gzip.open(compressed_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                temp_path.unlink()
                final_path = compressed_path
        else:
            # pigz not available, use standard gzip (slower but universal)
            import gzip
            with open(temp_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            temp_path.unlink()
            final_path = compressed_path
    else:
        # No compression requested
        # Note: detected_format is always 'nifti' or 'nifti_gz' for bucket files
        # (we force it above), so this is always a NIfTI file
        pass  # final_path = temp_path (already set)

    return final_path


def write_ols_arma_comparison(
    arma_results: ARMA11Results,
    output_prefix: Union[str, Path],
    **kwargs,
) -> dict:
    """
    Write side-by-side comparison of OLS and ARMA(1,1) results.

    This is a convenience wrapper that calls write_afni_bucket() twice with
    modified filenames (_OLS and _ARMA suffixes) and generates a JSON summary.

    Creates three files:
    - {output_prefix}_OLS.nii.gz (or .BRIK.gz) - OLS baseline results
    - {output_prefix}_ARMA.nii.gz (or .BRIK.gz) - ARMA corrected results
    - {output_prefix}_comparison_summary.json - Quantitative comparison

    Parameters
    ----------
    arma_results : ARMA11Results
        ARMA(1,1) fitting results with ols_results populated (want_ols=True)
    output_prefix : str or Path
        Output file prefix (e.g., 'results/analysis')
    **kwargs
        All other arguments passed to write_afni_bucket() for both files:
        condition_names, contrast_names, contrast_results, volume_shape,
        affine, voxel_size, dtype, apply_afni_metadata, compress_output,
        output_format, etc.

    Returns
    -------
    dict
        {'ols': Path, 'arma': Path, 'comparison_summary': Path}

    Raises
    ------
    ValueError
        If arma_results.ols_results is None (must use want_ols=True)

    Examples
    --------
    >>> # Fit with OLS comparison
    >>> results = ffs.fit_glm_arma11(data, design, tr=2.0, want_ols=True)
    >>>
    >>> # Write both outputs
    >>> outputs = ffs.write_ols_arma_comparison(
    ...     results,
    ...     'outputs/analysis',
    ...     condition_names=['Task', 'Rest'],
    ... )

    Notes
    -----
    This is equivalent to manually calling:
    >>> ols_file = ffs.write_afni_bucket(results.ols_results, 'analysis_OLS.nii.gz', ...)
    >>> arma_file = ffs.write_afni_bucket(results, 'analysis_ARMA.nii.gz', ...)

    The wrapper is just for convenience and adds the JSON comparison summary.
    """
    # Validate
    if arma_results.ols_results is None:
        raise ValueError(
            "ARMA results missing ols_results. "
            "Use want_ols=True when calling fit_glm_arma11()."
        )

    ols_results = arma_results.ols_results
    output_prefix = Path(output_prefix)

    # Extract contrast_results for OLS and ARMA if provided
    # User might pass contrast_results_ols and contrast_results_arma
    contrast_results_ols = kwargs.pop(
        "contrast_results_ols", kwargs.get("contrast_results")
    )
    contrast_results_arma = kwargs.pop(
        "contrast_results_arma", kwargs.get("contrast_results")
    )

    # Remove the generic one if it exists
    kwargs.pop("contrast_results", None)

    # Determine output format and construct paths
    output_format = kwargs.get("output_format")
    if output_format == "afni" or (
        output_format is None and "+tlrc" in str(output_prefix)
    ):
        # AFNI format
        ols_path = str(output_prefix.parent / f"{output_prefix.stem}_OLS+tlrc")
        arma_path = str(output_prefix.parent / f"{output_prefix.stem}_ARMA+tlrc")
    else:
        # NIfTI format (default)
        ols_path = str(output_prefix.parent / f"{output_prefix.stem}_OLS.nii.gz")
        arma_path = str(output_prefix.parent / f"{output_prefix.stem}_ARMA.nii.gz")

    # Write OLS bucket
    print(f"\n{'=' * 70}")
    print(f"Writing OLS results to: {Path(ols_path).name}")
    print(f"{'=' * 70}")
    ols_file = write_afni_bucket(
        ols_results,
        ols_path,
        contrast_results=contrast_results_ols,
        **kwargs,
    )

    # Write ARMA bucket
    print(f"\n{'=' * 70}")
    print(f"Writing ARMA results to: {Path(arma_path).name}")
    print(f"{'=' * 70}")
    arma_file = write_afni_bucket(
        arma_results,
        arma_path,
        contrast_results=contrast_results_arma,
        **kwargs,
    )

    # Create comparison summary
    summary = {
        "ols": {
            "mean_r2": float(ols_results.r2.mean()),
            "mean_abs_tstat": float(ols_results.tstats.abs().mean()),
            "max_abs_tstat": float(ols_results.tstats.abs().max()),
        },
        "arma": {
            "mean_r2": float(arma_results.r2.mean()),
            "mean_abs_tstat": float(arma_results.tstats.abs().mean()),
            "max_abs_tstat": float(arma_results.tstats.abs().max()),
            "mean_a": float(arma_results.arma_params[:, 0].mean()),
            "mean_b": float(arma_results.arma_params[:, 1].mean()),
            "std_a": float(arma_results.arma_params[:, 0].std()),
            "std_b": float(arma_results.arma_params[:, 1].std()),
        },
        "comparison": {
            "r2_improvement": float(arma_results.r2.mean() - ols_results.r2.mean()),
            "tstat_ratio": float(
                arma_results.tstats.abs().mean()
                / (ols_results.tstats.abs().mean() + 1e-10)
            ),
            "beta_correlation": float(
                np.corrcoef(
                    arma_results.betas.flatten().cpu().numpy(),
                    ols_results.betas.flatten().cpu().numpy(),
                )[0, 1]
            ),
        },
        "interpretation": {},
    }

    # Add interpretation
    tstat_ratio = summary["comparison"]["tstat_ratio"]
    r2_improvement = summary["comparison"]["r2_improvement"]

    if r2_improvement > 0.001:
        summary["interpretation"]["fit_quality"] = (
            f"ARMA has better fit (R² improvement: +{100 * r2_improvement:.2f}%)"
        )
    else:
        summary["interpretation"]["fit_quality"] = (
            "ARMA and OLS similar (low autocorrelation)"
        )

    if tstat_ratio < 0.95:
        summary["interpretation"]["tstat_correction"] = (
            f"ARMA corrects inflated t-stats (reduction: {100 * (1 - tstat_ratio):.1f}%)"
        )
    elif tstat_ratio > 1.05:
        summary["interpretation"]["tstat_correction"] = (
            "ARMA increases t-stats (suggests negative autocorrelation)"
        )
    else:
        summary["interpretation"]["tstat_correction"] = (
            "ARMA and OLS similar (minimal autocorrelation)"
        )

    # Write summary JSON
    summary_path = (
        output_prefix.parent / f"{output_prefix.stem}_comparison_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("OLS vs ARMA Comparison Summary")
    print(f"{'=' * 70}")
    print("\nOLS:")
    print(f"  Mean R²: {summary['ols']['mean_r2']:.4f}")
    print(f"  Mean |t|: {summary['ols']['mean_abs_tstat']:.3f}")

    print("\nARMA:")
    print(f"  Mean R²: {summary['arma']['mean_r2']:.4f}")
    print(f"  Mean |t|: {summary['arma']['mean_abs_tstat']:.3f}")
    print(
        f"  Mean (a,b): ({summary['arma']['mean_a']:.3f}, {summary['arma']['mean_b']:.3f})"
    )

    print("\nComparison:")
    print(f"  R² improvement: {summary['comparison']['r2_improvement']:.4f}")
    print(f"  t-stat ratio: {tstat_ratio:.3f}")
    print(f"  β correlation: {summary['comparison']['beta_correlation']:.4f}")

    print("\nInterpretation:")
    print(f"  • {summary['interpretation']['fit_quality']}")
    print(f"  • {summary['interpretation']['tstat_correction']}")

    print("\n✓ Comparison files written:")
    print(f"  OLS:     {ols_file}")
    print(f"  ARMA:    {arma_file}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}\n")

    return {
        "ols": ols_file,
        "arma": arma_file,
        "comparison_summary": summary_path,
    }


def write_partial_r2_with_labels(
    r2_partial_data: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    condition_labels: Sequence[str],
    volume_shape: Sequence[int],
    voxel_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    affine: Optional[np.ndarray] = None,
    n_timepoints: Optional[int] = None,
    n_regressors: Optional[int] = None,
    apply_afni_metadata: bool = True,
    mode: str = "full",  # "full" or "task" - affects label suffix
) -> Path:
    """
    Write partial R² per condition with proper AFNI labels and stat parameters.

    Partial R² represents the variance uniquely explained by each regressor.
    In AFNI terms, this is a correlation coefficient (fico type).

    Parameters
    ----------
    r2_partial_data : tensor or array, shape (n_voxels, n_conditions)
        Partial R² values for each condition
    output_path : str or Path
        Output file path (will be .nii.gz)
    condition_labels : sequence of str
        Labels for each condition
    volume_shape : tuple of int
        Spatial dimensions (nx, ny, nz)
    voxel_mask : tensor or array, optional
        Boolean mask for sparse data
    affine : np.ndarray, optional
        4x4 affine transformation matrix
    n_timepoints : int, optional
        Number of timepoints (for AFNI stat params)
    n_regressors : int, optional
        Total number of regressors (for AFNI stat params)
    apply_afni_metadata : bool, default=True
        Apply AFNI labels and stat parameters using 3drefit

    Returns
    -------
    Path
        Path to written file
    """
    import shutil
    import subprocess

    # Convert to numpy
    r2_partial_np = _ensure_numpy(r2_partial_data)
    n_conditions = r2_partial_np.shape[1]

    if len(condition_labels) != n_conditions:
        raise ValueError(f"Expected {n_conditions} labels, got {len(condition_labels)}")

    # Reshape to 4D volume
    if voxel_mask is not None:
        voxel_mask_np = _ensure_numpy(voxel_mask)
        voxel_mask_flat = voxel_mask_np.reshape(-1)
        r2_partial_vol = np.zeros((*volume_shape, n_conditions), dtype=np.float32)
        for cond_idx in range(n_conditions):
            vol_flat = np.zeros(int(np.prod(volume_shape)), dtype=np.float32)
            vol_flat[voxel_mask_flat] = r2_partial_np[:, cond_idx]
            r2_partial_vol[..., cond_idx] = vol_flat.reshape(volume_shape)
    else:
        r2_partial_vol = r2_partial_np.reshape(*volume_shape, n_conditions)

    # Create NIfTI
    if affine is None:
        affine = np.eye(4)
    r2_img = nib.Nifti1Image(r2_partial_vol.astype(np.float32), affine)

    # Ensure output path is .nii (uncompressed for 3drefit)
    output_path = Path(output_path)
    if str(output_path).endswith(".nii.gz"):
        temp_path = output_path.parent / (output_path.name[:-7] + ".nii")
    else:
        temp_path = output_path.parent / (output_path.stem + ".nii")

    # Save uncompressed first
    nib.save(r2_img, temp_path)

    # Apply AFNI metadata if requested
    if apply_afni_metadata and shutil.which("3drefit"):
        try:
            # Build labels: include mode suffix
            # "full" mode: "cond1_partialR2" (proportion of total variance)
            # "task" mode: "cond1_partialR2_task" (proportion of variance after nuisance)
            suffix = "_partialR2_task" if mode == "task" else "_partialR2"
            labels = [f"{label}{suffix}" for label in condition_labels]

            # Split into two 3drefit commands to avoid buffer overflow
            # Command 1: Set labels (write to file to avoid buffer overflow)
            labels_file = temp_path.parent / f"{temp_path.stem}_labels.txt"
            with labels_file.open('w') as f:
                # Write space-separated labels (AFNI format for -relabel_all)
                f.write(" ".join(labels))

            cmd_relabel = ["3drefit", "-relabel_all", str(labels_file), str(temp_path)]

            # Write relabel command to file for debugging
            cmd_file_relabel = temp_path.parent / f"{temp_path.stem}_relabel_3drefit_cmd.txt"
            with cmd_file_relabel.open('w') as f:
                f.write("# 3drefit command for setting sub-brick labels (partial R²)\n")
                f.write("# This file is created automatically and can be deleted\n\n")
                f.write(f"Labels written to: {labels_file}\n\n")
                f.write("Command as list:\n")
                f.write(f"{cmd_relabel}\n\n")
                f.write("Command as shell string:\n")
                import shlex
                shell_cmd = " ".join(shlex.quote(arg) for arg in cmd_relabel)
                f.write(f"{shell_cmd}\n")

            # Run relabel command
            subprocess.run(cmd_relabel, check=True, capture_output=True, text=True)

            # Command 2: Set stat parameters (if available)
            # Type: fico (Correlation)
            # Params: SAMPLES FIT-PARAMETERS ORT-PARAMETERS
            if n_timepoints and n_regressors:
                cmd_statpar = ["3drefit"]
                for brick_idx in range(n_conditions):
                    # Partial R² tests one regressor against all others
                    # SAMPLES = n_timepoints
                    # FIT-PARAMETERS = 1 (testing 1 regressor)
                    # ORT-PARAMETERS = n_regressors - 1 (orthogonalized against others)
                    cmd_statpar.extend([
                        "-substatpar", str(brick_idx), "fico",
                        str(n_timepoints), "1", str(n_regressors - 1)
                    ])

                cmd_statpar.append(str(temp_path))

                # Write statpar command to file for debugging
                cmd_file_statpar = temp_path.parent / f"{temp_path.stem}_statpar_3drefit_cmd.txt"
                with cmd_file_statpar.open('w') as f:
                    f.write("# 3drefit command for setting statistical parameters (partial R²)\n")
                    f.write("# This file is created automatically and can be deleted\n\n")
                    f.write("Command as list:\n")
                    f.write(f"{cmd_statpar}\n\n")
                    f.write("Command as shell string:\n")
                    shell_cmd = " ".join(shlex.quote(arg) for arg in cmd_statpar)
                    f.write(f"{shell_cmd}\n")

                # Run statpar command
                subprocess.run(cmd_statpar, check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Warning: 3drefit failed: {e.stderr}")
        except Exception as e:
            print(f"  ⚠ Warning: 3drefit error: {e}")

    # Compress to .nii.gz if requested
    if str(output_path).endswith(".nii.gz"):
        _save_nifti_with_format(nib.load(temp_path), output_path, "nifti_gz")
        temp_path.unlink()  # Remove uncompressed
        final_path = output_path
    else:
        final_path = temp_path

    return final_path
