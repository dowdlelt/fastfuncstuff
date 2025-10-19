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


def _normalize_output_path(output_path: Union[str, Path]) -> tuple[Path, str]:
    """
    Normalize output path and detect format from extension.

    Returns
    -------
    path : Path
        Normalized path object
    format : str
        File format: 'nifti', 'nifti_gz', or 'afni'
    """
    output_path = Path(output_path)

    # Detect format from extension
    suffixes = "".join(output_path.suffixes).lower()

    if suffixes.endswith(".head"):
        # AFNI HEAD file specified
        return output_path.with_suffix(""), "afni"
    elif suffixes.endswith(".brik") or suffixes.endswith(".brik.gz"):
        # AFNI BRIK file specified - remove extension, keep prefix
        if suffixes.endswith(".brik.gz"):
            return output_path.with_suffix("").with_suffix(""), "afni"
        else:
            return output_path.with_suffix(""), "afni"
    elif suffixes.endswith(".nii.gz"):
        return output_path, "nifti_gz"
    elif suffixes.endswith(".nii"):
        return output_path, "nifti"
    else:
        # Default to NIfTI compressed
        return output_path.with_suffix(".nii.gz"), "nifti_gz"


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
        # Save as AFNI BRIK/HEAD
        # nibabel automatically creates .HEAD and .BRIK files
        brik_path = output_path.with_suffix(
            ".BRIK" if not compress_brik else ".BRIK.gz"
        )
        nib.save(img, str(brik_path))
        # Return the HEAD file path (convention)
        return output_path.with_suffix(".HEAD")

    elif format == "nifti_gz":
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
        stats_img = nib.Nifti1Image(stats_data.astype(dtype, copy=False), affine_mat)
        stats_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            stats_img.header["pixdim"][4] = float(tr)
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
        fstat_img = nib.Nifti1Image(fstat_vol.astype(dtype, copy=False), affine_mat)
        fstat_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            fstat_img.header["pixdim"][4] = float(tr)
        fstat_path = output_dir / f"{prefix}_fstat.nii.gz"
        nib.save(fstat_img, fstat_path)
        outputs["fstat"] = fstat_path

    if include_r2 and getattr(results, "r2", None) is not None:
        r2_np = _ensure_numpy(results.r2)
        r2_vol = _reshape_parameter_map(r2_np, volume_shape, voxel_mask)
        r2_img = nib.Nifti1Image(r2_vol.astype(dtype, copy=False), affine_mat)
        r2_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            r2_img.header["pixdim"][4] = float(tr)
        r2_path = output_dir / f"{prefix}_r2.nii.gz"
        nib.save(r2_img, r2_path)
        outputs["r2"] = r2_path

    if include_mean and getattr(results, "meanvol", None) is not None:
        mean_np = _ensure_numpy(results.meanvol)
        mean_vol = _reshape_parameter_map(mean_np, volume_shape, voxel_mask)
        mean_img = nib.Nifti1Image(mean_vol.astype(dtype, copy=False), affine_mat)
        mean_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            mean_img.header["pixdim"][4] = float(tr)
        mean_path = output_dir / f"{prefix}_mean.nii.gz"
        nib.save(mean_img, mean_path)
        outputs["mean"] = mean_path

    if include_sigma and getattr(results, "sigma2", None) is not None:
        sigma_np = np.sqrt(np.maximum(_ensure_numpy(results.sigma2), 0.0))
        sigma_vol = _reshape_parameter_map(sigma_np, volume_shape, voxel_mask)
        sigma_img = nib.Nifti1Image(sigma_vol.astype(dtype, copy=False), affine_mat)
        sigma_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            sigma_img.header["pixdim"][4] = float(tr)
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
        resid_img = nib.Nifti1Image(resid_vol.astype(dtype, copy=False), affine_mat)
        resid_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            resid_img.header["pixdim"][4] = float(tr)
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
        pred_img = nib.Nifti1Image(pred_vol.astype(dtype, copy=False), affine_mat)
        pred_img.header.set_xyzt_units(xyz="mm", t="sec")
        if tr is not None:
            pred_img.header["pixdim"][4] = float(tr)
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
    Write GLM results as an AFNI-style bucket file

    Creates a single 4D NIfTI file with sub-bricks in AFNI order:
    1. Overall F-statistic
    2. For each condition: Beta coefficient, then T-statistic
    3. For each contrast: Beta coefficient, then T-statistic

    This matches AFNI's 3dDeconvolve output format.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        GLM fitting results
    output_path : str or Path
        Output file path (.nii or .nii.gz)
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
        If True, compress final output.
        - For NIfTI: creates .nii.gz
        - For AFNI: creates .BRIK.gz (compressed)
        If False:
        - For NIfTI: creates .nii
        - For AFNI: creates .BRIK (uncompressed)
    output_format : str, optional
        Output format: 'nifti', 'nifti_gz', or 'afni'
        If None (default), auto-detect from output_path extension:
        - .nii → NIfTI uncompressed
        - .nii.gz → NIfTI compressed
        - .HEAD / .BRIK / .BRIK.gz → AFNI format
        - Default: NIfTI compressed (.nii.gz)

    Returns
    -------
    output_path : Path
        Path to written bucket file (with .nii.gz if compressed)

    Examples
    --------
    >>> # Fit GLM
    >>> results = ffs.fit_glm_arma11(data, design, tr=2.0)
    >>>
    >>> # Compute contrasts
    >>> contrasts = [[1, -1, 0, 0]]  # Condition 1 vs 2
    >>> contrast_results = ffs.compute_contrasts(results, contrasts)
    >>>
    >>> # Write as NIfTI (compressed) - default
    >>> ffs.write_afni_bucket(
    ...     results,
    ...     'glm_bucket.nii.gz',
    ...     condition_names=['faces', 'places', 'poly0', 'poly1'],
    ...     contrast_names=['faces_vs_places'],
    ...     contrast_results=contrast_results,
    ...     apply_afni_metadata=True  # Automatic 3drefit!
    ... )
    >>>
    >>> # Write as AFNI BRIK/HEAD format (auto-detected from extension)
    >>> ffs.write_afni_bucket(
    ...     results,
    ...     'glm_bucket+tlrc.HEAD',  # AFNI format auto-detected
    ...     condition_names=['faces', 'places', 'poly0', 'poly1'],
    ...     contrast_names=['faces_vs_places'],
    ...     contrast_results=contrast_results,
    ... )
    >>> # Creates: glm_bucket+tlrc.HEAD and glm_bucket+tlrc.BRIK.gz

    Notes
    -----
    Sub-brick order matches AFNI 3dDeconvolve:
    - [0]: Full_Fstat (overall model F-statistic)
    - [1]: cond1#0_Coef (beta for condition 1)
    - [2]: cond1#0_Tstat (t-stat for condition 1)
    - [3]: cond2#0_Coef (beta for condition 2)
    - [4]: cond2#0_Tstat (t-stat for condition 2)
    - ...
    - [N]: contrast1#0_Coef (beta for contrast 1)
    - [N+1]: contrast1#0_Tstat (t-stat for contrast 1)
    - ...
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

    if getattr(results, "tstats", None) is None:
        raise ValueError(
            "T-statistics required for AFNI bucket. Ensure GLM was fit with t-stats enabled."
        )

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

    # 1. Overall F-statistic (first sub-brick, AFNI style)
    if getattr(results, "fstats", None) is not None:
        fstat_np = _ensure_numpy(results.fstats)
        fstat_vol = _reshape_parameter_map(fstat_np, volume_shape, voxel_mask)
        subbricks.append(fstat_vol.astype(dtype, copy=False))
        labels.append("Full_Fstat")
    else:
        raise ValueError(
            "F-statistics required for AFNI bucket. Ensure GLM was fit with f-stats enabled."
        )

    # 2. Beta and T-stat for each condition
    for idx, name in enumerate(condition_names):
        # Beta coefficient
        subbricks.append(betas_vol[..., idx].astype(dtype, copy=False))
        labels.append(f"{name}#0_Coef")

        # T-statistic
        subbricks.append(tstats_vol[..., idx].astype(dtype, copy=False))
        labels.append(f"{name}#0_Tstat")

    # 3. Beta and T-stat for each contrast
    if contrast_results is not None:
        contrast_betas = _ensure_numpy(contrast_results["contrast_betas"])
        contrast_tstats = _ensure_numpy(contrast_results["contrast_tstats"])

        # Handle single contrast case
        if contrast_betas.ndim == 1:
            contrast_betas = contrast_betas[:, np.newaxis]
            contrast_tstats = contrast_tstats[:, np.newaxis]

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

    # Stack all sub-bricks
    bucket_data = np.stack(subbricks, axis=-1)

    # Create NIfTI image
    bucket_img = nib.Nifti1Image(bucket_data, affine_mat)
    bucket_img.header.set_xyzt_units(xyz="mm", t="sec")
    if tr is not None:
        bucket_img.header["pixdim"][4] = float(tr)

    # Normalize output path and detect format
    base_path, detected_format = _normalize_output_path(output_path)
    if output_format is not None:
        detected_format = output_format

    # IMPORTANT: Bucket files must be saved as NIfTI format
    # We create a Nifti1Image, and nibabel cannot convert NIfTI headers to AFNI headers
    # 3drefit works fine with NIfTI files, so always use NIfTI for buckets
    if detected_format == "afni":
        detected_format = "nifti_gz"  # Default to compressed NIfTI
        # Update base_path to have .nii.gz extension
        base_path = base_path.with_suffix(".nii.gz")

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
                    # Build combined 3drefit command for labels AND stats
                    label_str = " ".join(labels)
                    cmd = ["3drefit", "-relabel_all_str", label_str]

                    # Add statistical parameters
                    brick_idx = 0

                    # F-statistic (sub-brick 0)
                    cmd.extend(
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
                        cmd.extend(
                            ["-substatpar", str(brick_idx + 1), "fitt", str(dof)]
                        )
                        brick_idx += 2

                    # Contrast t-statistics (if any)
                    if contrast_names:
                        for _ in contrast_names:
                            cmd.extend(
                                ["-substatpar", str(brick_idx + 1), "fitt", str(dof)]
                            )
                            brick_idx += 2

                    # Add file path
                    cmd.append(str(temp_path))

                    # Run 3drefit
                    subprocess.run(cmd, check=True, capture_output=True, text=True)

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
        import gzip
        import shutil

        # Note: detected_format is always 'nifti' or 'nifti_gz' for bucket files
        # (we force it above), so we always compress as NIfTI
        # Compress NIfTI: .nii → .nii.gz
        # Use parent/stem + .nii.gz to avoid double .nii.nii.gz
        compressed_path = temp_path.parent / (temp_path.stem + ".nii.gz")
        with open(temp_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove uncompressed file
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
