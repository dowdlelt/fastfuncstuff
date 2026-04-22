"""Comparison utilities for benchmark validation.

Thin wrappers around existing spatial correlation infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from ..stats.spatial import (
    consistency_report,
    optimal_matching,
    spatial_correlation,
    spatial_correlation_matrix,
)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays (numpy)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-15:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _matlab_flat_to_c_order(flat: np.ndarray, vol_size: tuple[int, ...]) -> np.ndarray:
    """Reshape a MATLAB Fortran-order flattened vector to C-order flat.

    MATLAB's ``(:)`` operator flattens in column-major (Fortran) order.
    NIfTI loaded by nibabel uses C-order. This function reshapes a 1-D
    MATLAB vector back to 3-D in Fortran order, then re-flattens in C
    order so voxel indices match nibabel's layout.
    """
    return flat.reshape(vol_size, order="F").flatten(order="C")


def _load_vol(path: str | Path) -> tuple[Tensor, np.ndarray]:
    """Load a NIfTI file as a torch tensor + affine.

    Squeezes any singleton dimensions (AFNI bucket files sometimes have
    shape (x,y,z,1,n) instead of (x,y,z,n)).
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    return torch.from_numpy(data), img.affine


def _automask(vol: Tensor) -> Tensor:
    """Simple threshold mask: voxels > 10% of robust max."""
    flat = vol.reshape(-1)
    robust_max = flat.quantile(0.98)
    return vol > (robust_max * 0.1)


def compare_volumes(
    a_path: str | Path,
    b_path: str | Path,
    mask_path: str | Path | None = None,
) -> dict:
    """Spatial correlation between two 3D volumes.

    Returns dict with 'r' (Pearson correlation).
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    # NIfTI 4D: (x, y, z, t) — average over time (last dim)
    if a.dim() == 4:
        a = a.mean(-1)
    if b.dim() == 4:
        b = b.mean(-1)

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        mask = _automask(a) & _automask(b)

    r = spatial_correlation(a, b, mask=mask, method="pearson")
    return {"r": float(r), "n_voxels": int(mask.sum())}


def compare_masks(
    a_path: str | Path,
    b_path: str | Path,
) -> dict:
    """Dice coefficient and voxel counts between two binary masks.

    Returns dict with 'dice', 'n_a', 'n_b', 'n_overlap'.
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    mask_a = a > 0.5
    mask_b = b > 0.5

    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    n_overlap = int((mask_a & mask_b).sum())

    dice = 2.0 * n_overlap / max(n_a + n_b, 1)

    return {"dice": float(dice), "n_a": n_a, "n_b": n_b, "n_overlap": n_overlap}


def compare_timeseries_4d(
    a_path: str | Path,
    b_path: str | Path,
    mask_path: str | Path | None = None,
    sample_frac: float = 0.1,
) -> dict:
    """Voxelwise temporal correlation between two 4D volumes.

    Computes Pearson r of each voxel's timeseries, returns median and
    fraction above various thresholds.

    Args:
        a_path, b_path: Paths to 4D NIfTI files.
        mask_path: Optional mask. If None, auto-masked from mean.
        sample_frac: Fraction of voxels to sample (for speed).
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    # NIfTI 4D: (x, y, z, t) -> (t, x, y, z) for temporal analysis
    if a.dim() == 4:
        a = a.permute(3, 0, 1, 2)
    if b.dim() == 4:
        b = b.permute(3, 0, 1, 2)

    if a.shape != b.shape:
        return {"error": f"Shape mismatch: {a.shape} vs {b.shape}"}

    # Compute mean for masking — mean over time (dim 0)
    a_mean = a.mean(0)
    b_mean = b.mean(0)

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        mask = _automask(a_mean) & _automask(b_mean)

    # Flatten spatial dims: (T, V)
    nt = a.shape[0]
    a_flat = a.reshape(nt, -1)[:, mask.reshape(-1)]
    b_flat = b.reshape(nt, -1)[:, mask.reshape(-1)]

    n_vox = a_flat.shape[1]

    # Sample if too many voxels
    if sample_frac < 1.0 and n_vox > 1000:
        n_sample = max(1000, int(n_vox * sample_frac))
        idx = torch.randperm(n_vox)[:n_sample]
        a_flat = a_flat[:, idx]
        b_flat = b_flat[:, idx]
        n_vox = n_sample

    # Pearson r per voxel (population std, correction=0, so denominator nt is consistent)
    a_z = a_flat - a_flat.mean(0, keepdim=True)
    b_z = b_flat - b_flat.mean(0, keepdim=True)
    a_std = a_z.std(0, correction=0)
    b_std = b_z.std(0, correction=0)
    valid = (a_std > 1e-8) & (b_std > 1e-8)

    r_vals = torch.zeros(n_vox)
    if valid.any():
        num = (a_z[:, valid] * b_z[:, valid]).sum(0)
        denom = a_std[valid] * b_std[valid] * nt
        r_vals[valid] = num / denom

    r_np = r_vals[valid].numpy()
    return {
        "median_r": float(np.median(r_np)) if len(r_np) > 0 else 0.0,
        "mean_r": float(np.mean(r_np)) if len(r_np) > 0 else 0.0,
        "frac_above_0.95": float((r_np > 0.95).mean()) if len(r_np) > 0 else 0.0,
        "frac_above_0.99": float((r_np > 0.99).mean()) if len(r_np) > 0 else 0.0,
        "n_voxels": int(valid.sum()),
        "n_total_mask": int(mask.sum()),
    }


def compare_1d_params(
    a_path: str | Path,
    b_path: str | Path,
) -> dict:
    """Compare two AFNI .1D parameter files column-by-column.

    Returns per-column Pearson correlations.
    """
    a = np.loadtxt(str(a_path))
    b = np.loadtxt(str(b_path))

    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]

    n_rows = min(a.shape[0], b.shape[0])
    n_cols = min(a.shape[1], b.shape[1])
    a = a[:n_rows, :n_cols]
    b = b[:n_rows, :n_cols]

    correlations = []
    for c in range(n_cols):
        if np.std(a[:, c]) < 1e-10 or np.std(b[:, c]) < 1e-10:
            correlations.append(float("nan"))
        else:
            r = np.corrcoef(a[:, c], b[:, c])[0, 1]
            correlations.append(float(r))

    return {
        "per_column_r": correlations,
        "mean_r": float(np.nanmean(correlations)),
        "min_r": float(np.nanmin(correlations)),
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def compare_moco_ssd(
    a_path: str | Path,
    b_path: str | Path,
    mask_path: str | Path | None = None,
) -> dict:
    """Mean of squared differences between two motion-corrected 4D volumes.

    Reports per-volume MSD and summary stats. MSD normalises over voxel count
    so it's comparable across different mask sizes.
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    # NIfTI 4D: (x, y, z, t) -> (t, x, y, z)
    if a.dim() == 4:
        a = a.permute(3, 0, 1, 2)
    if b.dim() == 4:
        b = b.permute(3, 0, 1, 2)

    if a.shape != b.shape:
        return {"error": f"Shape mismatch: {a.shape} vs {b.shape}"}

    a_mean = a.mean(0)
    b_mean = b.mean(0)

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        mask = _automask(a_mean) & _automask(b_mean)

    n_vox = int(mask.sum())
    nt = a.shape[0]

    # Per-volume MSD within mask
    mask_flat = mask.reshape(-1)
    per_vol_msd = []
    for t in range(nt):
        diff = a[t].reshape(-1)[mask_flat] - b[t].reshape(-1)[mask_flat]
        msd = float((diff ** 2).mean())
        per_vol_msd.append(msd)

    per_vol_msd_arr = np.array(per_vol_msd)

    # Normalise by signal intensity for interpretability
    signal_mean = float(a.mean(0).reshape(-1)[mask_flat].mean())

    return {
        "mean_msd": float(per_vol_msd_arr.mean()),
        "max_msd": float(per_vol_msd_arr.max()),
        "median_msd": float(np.median(per_vol_msd_arr)),
        "signal_mean": signal_mean,
        "nrmsd": float(np.sqrt(per_vol_msd_arr.mean()) / signal_mean) if signal_mean > 0 else 0.0,
        "n_voxels": n_vox,
        "n_volumes": nt,
    }


def compare_ica_components(
    a_path: str | Path,
    b_path: str | Path,
    mask_path: str | Path | None = None,
) -> dict:
    """Compare ICA spatial maps using optimal matching.

    Uses absolute correlation since ICA components are sign-ambiguous.

    Args:
        a_path, b_path: Paths to 4D NIfTI component maps.
        mask_path: Optional mask.
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    # NIfTI 4D: (x, y, z, n_comp) -> need (n_comp, x, y, z)
    if a.dim() == 4:
        a = a.permute(3, 0, 1, 2)
    elif a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 4:
        b = b.permute(3, 0, 1, 2)
    elif b.dim() == 3:
        b = b.unsqueeze(0)

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        # Use union of nonzero voxels across all components
        a_any = a.abs().sum(0) > 1e-8
        b_any = b.abs().sum(0) > 1e-8
        mask = a_any | b_any

    corr_matrix = spatial_correlation_matrix(a, b, mask=mask)
    # Use absolute values for sign-ambiguous ICA components
    abs_corr = np.abs(corr_matrix)
    row_idx, col_idx, matched_corrs = optimal_matching(abs_corr)
    report = consistency_report(abs_corr)

    matches = [(int(r), int(c), float(v)) for r, c, v in zip(row_idx, col_idx, matched_corrs)]
    return {
        "mean_matched_r": float(matched_corrs.mean()),
        "median_matched_r": float(np.median(matched_corrs)),
        "n_components_a": a.shape[0],
        "n_components_b": b.shape[0],
        "n_matched": len(matches),
        "coverage_0.5": report.coverage_at_thresholds.get(0.5, 0.0),
        "coverage_0.7": report.coverage_at_thresholds.get(0.7, 0.0),
        "matches": matches,
    }


def compare_aff12(
    a_path: str | Path,
    b_path: str | Path,
) -> dict:
    """Compare two AFNI .aff12.1D affine matrices.

    Decomposes each 3×4 matrix into rotation angles (degrees) and
    translation (mm), then reports per-parameter differences.

    Returns:
        dict with rotation_diff_deg (3,), translation_diff_mm (3,),
        max_angle_diff, max_trans_diff, frobenius_norm.
    """
    def _load_aff12(path: str | Path) -> np.ndarray:
        """Load an AFNI .aff12.1D file as a 3×4 matrix."""
        raw = np.loadtxt(str(path), comments="#")
        if raw.ndim == 1 and raw.size == 12:
            return raw.reshape(3, 4)
        if raw.ndim == 2:
            # Multi-row (per-volume matrices) — take first row
            if raw.shape[1] == 12:
                return raw[0].reshape(3, 4)
            if raw.shape == (3, 4):
                return raw
        raise ValueError(f"Cannot parse aff12 from {path}: shape {raw.shape}")

    def _decompose_aff(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract rotation angles (deg) and translation (mm) from 3×4 affine.

        Uses the convention: mat = [R | t] where R is 3×3 rotation.
        Rotation angles extracted via Euler angle decomposition (xyz convention).
        """
        R = mat[:3, :3]
        t = mat[:3, 3]

        # Euler angles (xyz convention, same as AFNI)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0

        angles_deg = np.degrees([rx, ry, rz])
        return angles_deg, t

    mat_a = _load_aff12(a_path)
    mat_b = _load_aff12(b_path)

    angles_a, trans_a = _decompose_aff(mat_a)
    angles_b, trans_b = _decompose_aff(mat_b)

    angle_diff = np.abs(angles_a - angles_b)
    trans_diff = np.abs(trans_a - trans_b)
    frob = float(np.linalg.norm(mat_a - mat_b, "fro"))

    return {
        "rotation_diff_deg": angle_diff.tolist(),
        "translation_diff_mm": trans_diff.tolist(),
        "max_angle_diff": float(angle_diff.max()),
        "max_trans_diff": float(trans_diff.max()),
        "frobenius_norm": frob,
        "angles_a": angles_a.tolist(),
        "angles_b": angles_b.tolist(),
        "trans_a": trans_a.tolist(),
        "trans_b": trans_b.tolist(),
    }


def compare_aff12_series(
    a_path: str | Path,
    b_path: str | Path,
) -> dict:
    """Compare two multi-row AFNI ``.aff12.1D`` files row-by-row.

    Each row is a 3×4 matrix in DICOM coords. Reports per-entry max/mean
    differences separated into rotation (columns 0,1,2,4,5,6,8,9,10) and
    translation (columns 3,7,11) parts, plus Frobenius distance per volume.

    Returns:
        dict with keys: max_rot_diff, mean_rot_diff, max_trans_diff (mm),
        mean_trans_diff (mm), max_frobenius, mean_frobenius, n_volumes.
    """
    a = np.loadtxt(str(a_path), comments="#")
    b = np.loadtxt(str(b_path), comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if a.shape != b.shape or a.shape[1] != 12:
        return {
            "error": f"shape mismatch or wrong width: a={a.shape} b={b.shape}"
        }

    diff = np.abs(a - b)
    trans_cols = [3, 7, 11]
    rot_cols = [c for c in range(12) if c not in trans_cols]
    per_row_frob = np.linalg.norm(a - b, axis=1)

    return {
        "n_volumes": int(a.shape[0]),
        "max_rot_diff": float(diff[:, rot_cols].max()),
        "mean_rot_diff": float(diff[:, rot_cols].mean()),
        "max_trans_diff": float(diff[:, trans_cols].max()),
        "mean_trans_diff": float(diff[:, trans_cols].mean()),
        "max_frobenius": float(per_row_frob.max()),
        "mean_frobenius": float(per_row_frob.mean()),
    }


def compare_bucket_volumes(
    a_path: str | Path,
    b_path: str | Path,
    sub_brick_indices: list[int] | None = None,
    mask_path: str | Path | None = None,
) -> dict:
    """Compare sub-bricks of two GLM bucket files.

    Computes spatial correlation for each sub-brick pair.

    Args:
        a_path, b_path: Paths to bucket NIfTI files.
        sub_brick_indices: Which sub-bricks to compare. None = all.
        mask_path: Optional mask.
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    # NIfTI 4D: (x, y, z, n_bricks) -> (n_bricks, x, y, z)
    if a.dim() == 4:
        a = a.permute(3, 0, 1, 2)
    elif a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 4:
        b = b.permute(3, 0, 1, 2)
    elif b.dim() == 3:
        b = b.unsqueeze(0)

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        # Mask from first volume of a
        mask = _automask(a[0])

    if sub_brick_indices is None:
        n = min(a.shape[0], b.shape[0])
        sub_brick_indices = list(range(n))

    results = []
    for idx in sub_brick_indices:
        if idx < a.shape[0] and idx < b.shape[0]:
            r = spatial_correlation(a[idx], b[idx], mask=mask, method="pearson")
            results.append({"index": idx, "r": float(r)})

    r_vals = [x["r"] for x in results]
    return {
        "per_brick": results,
        "mean_r": float(np.mean(r_vals)) if r_vals else 0.0,
        "min_r": float(np.min(r_vals)) if r_vals else 0.0,
        "n_bricks": len(results),
    }


def compare_im_bucket(
    a_path: str | Path,
    b_path: str | Path,
    mask_path: str | Path | None = None,
) -> dict:
    """Compare IM-model GLM bucket files (F-stat + beta/t-stat pairs).

    Expected bucket layout (both AFNI and ffs_reml -tout):
        vol 0:    Full-model F-stat
        vol 1:    beta event-1
        vol 2:    t-stat event-1
        vol 3:    beta event-2
        vol 4:    t-stat event-2
        ...

    Validation:
        - F-stat: spatial correlation between AFNI and FFS volumes.
        - Betas:  per-sub-brick spatial r AND voxelwise "temporal" r
          (treating the ordered beta volumes as a pseudo-timeseries per voxel).
        - T-stats: same as betas.

    Returns a dict with separate sub-dicts for fstat, betas, and tstats,
    each containing mean_r, min_r, and (for betas/tstats) temporal_median_r.
    """
    a, _ = _load_vol(a_path)
    b, _ = _load_vol(b_path)

    if a.dim() == 4:
        a = a.permute(3, 0, 1, 2)   # (n_bricks, x, y, z)
    elif a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 4:
        b = b.permute(3, 0, 1, 2)
    elif b.dim() == 3:
        b = b.unsqueeze(0)

    n = min(a.shape[0], b.shape[0])

    if mask_path is not None:
        mask_vol, _ = _load_vol(mask_path)
        mask = mask_vol > 0.5
    else:
        mask = _automask(a[0])

    def _spatial_r(vol_a: Tensor, vol_b: Tensor) -> float:
        from ..stats.spatial import spatial_correlation
        return float(spatial_correlation(vol_a, vol_b, mask=mask, method="pearson"))

    def _temporal_r(vols_a: Tensor, vols_b: Tensor) -> float:
        """Voxelwise correlation across the volume axis (pseudo-temporal)."""
        # vols: (n_vols, x, y, z) → flatten to (n_vols, n_vox_masked)
        mask_flat = mask.reshape(-1)
        a_flat = vols_a.reshape(vols_a.shape[0], -1)[:, mask_flat].float()
        b_flat = vols_b.reshape(vols_b.shape[0], -1)[:, mask_flat].float()
        n_t = a_flat.shape[0]
        a_z = a_flat - a_flat.mean(0, keepdim=True)
        b_z = b_flat - b_flat.mean(0, keepdim=True)
        std_a = a_z.std(0, correction=0)
        std_b = b_z.std(0, correction=0)
        valid = (std_a > 1e-8) & (std_b > 1e-8)
        if not valid.any():
            return 0.0
        r_vox = (a_z[:, valid] * b_z[:, valid]).sum(0) / (std_a[valid] * std_b[valid] * n_t)
        return float(np.median(r_vox.numpy()))

    # F-stat (vol 0)
    fstat = {"r": _spatial_r(a[0], b[0])}

    # Separate even/odd sub-bricks (starting from vol 1)
    beta_idx  = list(range(1, n, 2))   # 1, 3, 5, ...
    tstat_idx = list(range(2, n, 2))   # 2, 4, 6, ...

    def _brick_stats(indices: list[int]) -> dict:
        if not indices:
            return {"mean_r": 0.0, "min_r": 0.0, "temporal_median_r": 0.0, "n": 0}
        r_vals = [_spatial_r(a[i], b[i]) for i in indices]
        vols_a = torch.stack([a[i] for i in indices])
        vols_b = torch.stack([b[i] for i in indices])
        return {
            "mean_r": float(np.mean(r_vals)),
            "min_r":  float(np.min(r_vals)),
            "temporal_median_r": _temporal_r(vols_a, vols_b),
            "n": len(indices),
        }

    beta_stats  = _brick_stats(beta_idx)
    tstat_stats = _brick_stats(tstat_idx)

    return {
        "fstat": fstat,
        "betas": beta_stats,
        "tstats": tstat_stats,
        "n_bricks_compared": n,
    }
