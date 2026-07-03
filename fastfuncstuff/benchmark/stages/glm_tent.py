"""TENT deconvolution benchmark: 3dDeconvolve TENT vs ffs_deconvolve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..runner import BenchmarkContext, run_timed
from ..validation import _automask, _load_vol, _pearson_r

name = "glm_tent"
description = "TENT deconvolution (3dDeconvolve TENT(0,15,11) vs ffs_deconvolve)"

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]

THRESHOLDS = {
    # Voxelwise temporal correlation across the TENT basis (per condition)
    "temporal_min_median_r": 0.95,
    # Spatial correlation of middle timepoint
    "spatial_middle_min_r": 0.95,
}


# ── Config helpers ────────────────────────────────────────────────────────────


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _stim_labels(ctx: BenchmarkContext) -> list[str]:
    return _glm_params(ctx).get("stim_labels", _DEFAULT_STIM_LABELS)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _afni_dir(ctx: BenchmarkContext) -> Path:
    return ctx.data_dir / "afni_glm_TENT"


def _ffs_dir(ctx: BenchmarkContext) -> Path:
    return ctx.data_dir / f"ffs_glm_TENT{ctx.ffs_tag}"


def _resampled_input(ctx: BenchmarkContext, run: int) -> Path:
    """Unscaled MNI-resampled input: prefer ffs_mni_resampled_*, fall back to afni_mni_resampled_*."""
    task = _primary_task(ctx)
    ffs = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{run}.nii.gz"
    if ffs.exists():
        return ffs
    return ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{run}.nii.gz"


def _automask_path(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "MNI_automask.nii.gz"


def _ensure_mni_automask(ctx: BenchmarkContext) -> None:
    """Create MNI_automask.nii.gz from run-1 if it doesn't exist."""
    mask_path = _automask_path(ctx)
    if mask_path.exists():
        return
    src = _resampled_input(ctx, 1)
    run_timed(
        f"3dAutomask -overwrite -prefix {mask_path} {src}",
        label="3dAutomask MNI",
        cwd=ctx.processing_dir,
    )


def _afni_iresp(ctx: BenchmarkContext, label: str) -> Path:
    """AFNI iresp output path (HEAD file). Data is MNI-warped so AFNI uses +tlrc."""
    return _afni_dir(ctx) / f"iresp_{label}+tlrc.HEAD"


def _ffs_iresp(ctx: BenchmarkContext, label: str) -> Path:
    return _ffs_dir(ctx) / f"ffs_tent_iresp_{label}.nii.gz"


# ── Stage interface ───────────────────────────────────────────────────────────


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for label in _stim_labels(ctx):
            afni = _afni_iresp(ctx, label)
            ffs = _ffs_iresp(ctx, label)
            if not afni.exists():
                missing.append(str(afni))
            if not ffs.exists():
                missing.append(str(ffs))
    else:
        for run in _runs(ctx):
            src = _resampled_input(ctx, run)
            if not src.exists():
                missing.append(str(src))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run 3dDeconvolve with TENT(0,15,11) basis functions."""
    afni = _afni_dir(ctx)
    afni.mkdir(parents=True, exist_ok=True)
    _ensure_mni_automask(ctx)

    task = _primary_task(ctx)
    stim_labels = _stim_labels(ctx)

    # Skip if all iresps already exist
    all_exist = all(_afni_iresp(ctx, label).exists() for label in stim_labels)
    if all_exist and not ctx.force_ref:
        return 0.0

    inputs = " ".join(str(_resampled_input(ctx, r)) for r in _runs(ctx))
    mask = _automask_path(ctx)

    stim_args = " ".join(
        f"-stim_times {i} {ctx.timing_dir}/onsets.{task}.times.{label}.txt "
        f"'TENT(0,15,11)' -stim_label {i} {label}"
        for i, label in enumerate(stim_labels, 1)
    )
    iresp_args = " ".join(
        f"-iresp {i} {afni}/iresp_{label}" for i, label in enumerate(stim_labels, 1)
    )

    elapsed, _ = run_timed(
        f"3dDeconvolve -overwrite "
        f"-input {inputs} "
        f"-polort A -num_stimts {len(stim_labels)} -float "
        f"{stim_args} "
        f"-jobs 10 -noFDR "
        f"{iresp_args} "
        f"-mask {mask} "
        f"-x1D {afni}/TENT_X.xmat.1D "
        f"-bucket {afni}/TENT_afni_stats_{task}.nii.gz",
        label="3dDeconvolve TENT",
        cwd=afni,
    )
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_deconvolve with TENT model."""
    ffs = _ffs_dir(ctx)
    ffs.mkdir(parents=True, exist_ok=True)
    _ensure_mni_automask(ctx)

    task = _primary_task(ctx)
    stim_labels = _stim_labels(ctx)

    all_exist = all(_ffs_iresp(ctx, label).exists() for label in stim_labels)
    if all_exist and not ctx.force_ffs:
        return 0.0

    inputs = " ".join(str(_resampled_input(ctx, r)) for r in _runs(ctx))
    timing_args = " ".join(
        str(ctx.timing_dir / f"onsets.{task}.times.{label}.txt") for label in stim_labels
    )
    mask = _automask_path(ctx)

    elapsed, _ = run_timed(
        f"ffs_deconvolve "
        f"-input {inputs} "
        f"-onsets {timing_args} "
        f"-prefix ffs_tent "
        f"-model TENT "
        f"-window 0 15 "
        f"-polort 4 "
        f"-mask {mask}"
        f"{ctx.ffs_device_flag()}",
        label="ffs_deconvolve TENT",
        cwd=ffs,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare AFNI and FFS iresp files per condition."""
    mask_path = _automask_path(ctx)
    mask_available = mask_path.exists()

    per_cond: dict[str, dict] = {}
    temporal_medians: list[float] = []
    spatial_middles: list[float] = []

    for label in _stim_labels(ctx):
        afni_path = _afni_iresp(ctx, label)
        ffs_path = _ffs_iresp(ctx, label)

        if not afni_path.exists() or not ffs_path.exists():
            per_cond[label] = {"error": "missing file"}
            continue

        a_vol, _ = _load_vol(afni_path)  # (x, y, z, n_basis) or (x, y, z, 1, n_basis)
        b_vol, _ = _load_vol(ffs_path)

        # Squeeze any singleton dims from AFNI format
        a = a_vol.squeeze()
        b = b_vol.squeeze()

        if a.dim() == 3:
            a = a.unsqueeze(-1)
        if b.dim() == 3:
            b = b.unsqueeze(-1)

        # Shape: (x, y, z, n_basis)
        *_, na = a.shape
        nb = b.shape[-1]
        n_basis = min(na, nb)

        # Build mask
        if mask_available:
            mask_vol, _ = _load_vol(mask_path)
            mask = mask_vol > 0.5
        else:
            mask = _automask(a.mean(-1)) & _automask(b.mean(-1))

        mask_flat = mask.reshape(-1)
        # (n_basis, n_vox)
        a_flat = a[..., :n_basis].reshape(-1, n_basis).T[:, mask_flat]
        b_flat = b[..., :n_basis].reshape(-1, n_basis).T[:, mask_flat]

        n_vox = a_flat.shape[1]

        # ── Voxelwise temporal correlation (each voxel's n_basis values) ──
        a_z = a_flat - a_flat.mean(0, keepdim=True)
        b_z = b_flat - b_flat.mean(0, keepdim=True)
        # Use population std (correction=0) so denominator n_basis is consistent.
        # PyTorch's default std() uses Bessel correction (n-1), which would give
        # r = true_r * (n-1)/n — for n=11 that's 0.909 even for identical outputs.
        a_std = a_z.std(0, correction=0)
        b_std = b_z.std(0, correction=0)
        valid = (a_std > 1e-8) & (b_std > 1e-8)

        r_vox = torch.zeros(n_vox)
        if valid.any():
            num = (a_z[:, valid] * b_z[:, valid]).sum(0)
            denom = a_std[valid] * b_std[valid] * n_basis
            r_vox[valid] = num / denom

        r_np = r_vox[valid].numpy()
        temporal_med = float(np.median(r_np)) if len(r_np) > 0 else 0.0
        temporal_medians.append(temporal_med)

        # ── Spatial correlation of middle timepoint ──
        mid_idx = n_basis // 2
        r_mid = _pearson_r(
            a[..., mid_idx][mask].numpy(),
            b[..., mid_idx][mask].numpy(),
        )
        spatial_middles.append(r_mid)

        per_cond[label] = {
            "temporal_median_r": temporal_med,
            "spatial_middle_r": r_mid,
            "n_basis_a": na,
            "n_basis_b": nb,
            "n_voxels": int(mask.sum()),
        }

    # Pass: all conditions must meet both thresholds
    any_error = any("error" in v for v in per_cond.values())
    if any_error or not temporal_medians:
        return {"passed": False, "summary": "missing outputs", "per_condition": per_cond}

    all_temporal_ok = all(r >= THRESHOLDS["temporal_min_median_r"] for r in temporal_medians)
    all_spatial_ok = all(r >= THRESHOLDS["spatial_middle_min_r"] for r in spatial_middles)

    passed = all_temporal_ok and all_spatial_ok
    summary = (
        f"temporal median_r min={min(temporal_medians):.4f} "
        f"(thr={THRESHOLDS['temporal_min_median_r']}), "
        f"spatial_mid min_r={min(spatial_middles):.4f} "
        f"(thr={THRESHOLDS['spatial_middle_min_r']})"
    )

    return {
        "passed": passed,
        "summary": summary,
        "per_condition": per_cond,
        "temporal_medians": temporal_medians,
        "spatial_middles": spatial_middles,
    }
