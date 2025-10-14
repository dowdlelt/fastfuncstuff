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
        Output file path (.nii.gz recommended)
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

    Returns
    -------
    output_path : Path
        Path to written bucket file

    Examples
    --------
    >>> # Fit GLM
    >>> results = ffs.fit_glm_arma11(data, design, tr=2.0)
    >>>
    >>> # Compute contrasts
    >>> contrasts = [[1, -1, 0, 0]]  # Condition 1 vs 2
    >>> contrast_results = ffs.compute_contrasts(results, contrasts)
    >>>
    >>> # Write AFNI bucket
    >>> ffs.write_afni_bucket(
    ...     results,
    ...     'glm_bucket.nii.gz',
    ...     condition_names=['faces', 'places', 'poly0', 'poly1'],
    ...     contrast_names=['faces_vs_places'],
    ...     contrast_results=contrast_results
    ... )

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

    # Save
    nib.save(bucket_img, output_path)

    # Write labels as JSON sidecar
    label_path = output_path.with_suffix(".json")
    with label_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "Description": "AFNI-style bucket file from FastFuncSim GLM",
                "SubBricks": labels,
                "Order": "F-stat, then (Beta, T-stat) pairs for each condition, then (Beta, T-stat) pairs for each contrast",
            },
            f,
            indent=2,
        )

    return output_path
