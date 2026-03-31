"""Cross-run EPI alignment benchmark: 3dAllineate vs ffs_allineate.

Aligns each run's mean image to the localizer run-1 reference.
Compares the estimated affine matrices (rotation/translation decomposition)
and the aligned output images between AFNI and FFS.

This was previously hidden inside the warp stage's prerequisite setup.
Breaking it out gives us:
  - Timing comparison for EPI-to-EPI alignment
  - Validation of ffs_allineate on functional data (not just T1→MNI)
  - 9 alignment pairs (loc 2-5 + rest 1-5) → good statistical power
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_aff12, compare_volumes

name = "crossalign"
description = "Cross-run EPI alignment (3dAllineate vs ffs_allineate)"

# All runs that need alignment to localizer run-1
ALIGN_PAIRS = [
    ("localizer", 2),
    ("localizer", 3),
    ("localizer", 4),
    ("localizer", 5),
    ("rest", 1),
    ("rest", 2),
    ("rest", 3),
    ("rest", 4),
    ("rest", 5),
]

THRESHOLDS = {
    "aligned_image_min_r": 0.98,  # aligned images should be nearly identical
    "max_angle_diff_deg": 0.5,    # rotation must agree within 0.5 degrees
    "max_trans_diff_mm": 0.5,     # translation must agree within 0.5 mm
}


def _ref_mean(ctx: BenchmarkContext) -> Path:
    """Reference mean image (localizer run-1, from AFNI moco)."""
    return ctx.processing_dir / "afni_mean_sub-01_ses-01_task-localizer_run-1_bold.nii"


def _src_mean(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii"


def _afni_mat(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mean_{task}_run-{run}_to_localizer_run-1_mat.aff12.1D"


def _afni_aligned(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mean_sub-01_ses-01_task-{task}_run-{run}_bold_al.nii"


def _ffs_mat(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"ffs_mean_{task}_run-{run}_to_localizer_run-1_mat.aff12.1D"


def _ffs_aligned(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"ffs_mean_sub-01_ses-01_task-{task}_run-{run}_bold_al.nii"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for task, run in ALIGN_PAIRS:
            for p in [_afni_mat(ctx, task, run), _ffs_mat(ctx, task, run)]:
                if not p.exists():
                    missing.append(str(p))
    else:
        # Need moco mean images (from moco stage)
        ref = _ref_mean(ctx)
        if not ref.exists():
            missing.append(str(ref))
        for task, run in ALIGN_PAIRS:
            src = _src_mean(ctx, task, run)
            if not src.exists():
                missing.append(str(src))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run AFNI 3dAllineate for all cross-run pairs."""
    total = 0.0
    ref = _ref_mean(ctx)
    for task, run in ALIGN_PAIRS:
        mat = _afni_mat(ctx, task, run)
        if mat.exists() and not ctx.force_ref:
            continue
        src = _src_mean(ctx, task, run)
        al_out = _afni_aligned(ctx, task, run)
        elapsed, _ = run_timed(
            f"3dAllineate -overwrite -cost lpa -onepass "
            f"-source_automask -autoweight -warp shr "
            f"-base {ref} -source {src} "
            f"-prefix {al_out} "
            f"-1Dmatrix_save {mat}",
            label=f"3dAllineate mean {task} run-{run} → loc run-1",
            cwd=ctx.processing_dir,
        )
        total += elapsed
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_allineate for all cross-run pairs."""
    total = 0.0
    ref = _ref_mean(ctx)
    for task, run in ALIGN_PAIRS:
        mat = _ffs_mat(ctx, task, run)
        if mat.exists() and not ctx.force_ffs:
            continue
        src = _src_mean(ctx, task, run)
        al_out = _ffs_aligned(ctx, task, run)
        elapsed, _ = run_timed(
            f"ffs_allineate -cost lpa -lpa_kernel box -lpa_sigma 0 -onepass "
            f"-interp cubic -final cubic "
            f"-source_automask -autoweight -rigid "
            f"-base {ref} -source {src} "
            f"-prefix {al_out} "
            f"-1Dmatrix_save {mat}",
            label=f"ffs_allineate mean {task} run-{run} → loc run-1",
            cwd=ctx.processing_dir,
        )
        total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare affine matrices and aligned images between AFNI and FFS."""
    mat_results = []
    img_results = []

    for task, run in ALIGN_PAIRS:
        afni_m = _afni_mat(ctx, task, run)
        ffs_m = _ffs_mat(ctx, task, run)

        label = f"{task}_run-{run}"

        # Matrix comparison
        if afni_m.exists() and ffs_m.exists():
            aff = compare_aff12(afni_m, ffs_m)
            aff["label"] = label
            mat_results.append(aff)

        # Aligned image comparison
        afni_img = _afni_aligned(ctx, task, run)
        ffs_img = _ffs_aligned(ctx, task, run)
        if afni_img.exists() and ffs_img.exists():
            vol = compare_volumes(afni_img, ffs_img)
            vol["label"] = label
            img_results.append(vol)

    if not mat_results:
        return {
            "passed": False,
            "summary": "No matrix comparisons available",
        }

    # Aggregate matrix differences
    max_angle = max(m["max_angle_diff"] for m in mat_results)
    max_trans = max(m["max_trans_diff"] for m in mat_results)
    mean_angle = sum(m["max_angle_diff"] for m in mat_results) / len(mat_results)
    mean_trans = sum(m["max_trans_diff"] for m in mat_results) / len(mat_results)

    # Aggregate image correlations
    img_min_r = min(v["r"] for v in img_results) if img_results else 0.0
    img_mean_r = sum(v["r"] for v in img_results) / len(img_results) if img_results else 0.0

    passed = (
        max_angle <= THRESHOLDS["max_angle_diff_deg"]
        and max_trans <= THRESHOLDS["max_trans_diff_mm"]
        and img_min_r >= THRESHOLDS["aligned_image_min_r"]
    )

    return {
        "passed": passed,
        "summary": (
            f"max Δangle={max_angle:.3f}°, max Δtrans={max_trans:.3f}mm, "
            f"aligned img min_r={img_min_r:.4f}"
        ),
        "matrices": {
            "max_angle_diff": max_angle,
            "mean_angle_diff": mean_angle,
            "max_trans_diff": max_trans,
            "mean_trans_diff": mean_trans,
            "per_pair": mat_results,
        },
        "images": {
            "min_r": img_min_r,
            "mean_r": img_mean_r,
            "per_pair": img_results,
        },
    }
