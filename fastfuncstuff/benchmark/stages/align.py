"""Alignment benchmark: ffs_allineate anat-to-MNI.

Compares the FFS-aligned + qwarped anatomical (anatFFS) against the
AFNI sswarper output (anatQQ). Both are warped to MNI space, so
spatial correlation measures how well the two pipelines agree.

The mean-to-mean alignment (3dAllineate) and epi-to-anat alignment
(align_epi_anat.py) are AFNI-only preprocessing steps that feed into
the warping pipeline. The FFS comparison point is at the final warped
output (handled by the warp stage).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_volumes

name = "align"
description = "Affine alignment + qwarp (sswarper vs ffs_allineate + ffs_qwarp)"

THRESHOLDS = {
    "warped_anat_r": 0.80,  # different warping algorithms, expect some disagreement
}


def _subid(ctx: BenchmarkContext) -> str:
    return f"sub-{ctx.subject}"


def _afni_anat(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "sswarper_output" / f"anatQQ.{_subid(ctx)}.nii"


def _ffs_anat(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "ffs_warper" / f"anatFFS.{_subid(ctx)}.nii.gz"


def _anat_input(ctx: BenchmarkContext) -> Path:
    return ctx.anat_dir / f"{_subid(ctx)}_ses-{ctx.session}_T1w.nii"


def _afni_template() -> str:
    """MNI template — prefers test_data/ copy, falls back to AFNI binary directory."""
    # Bundled copy in test_data/ works on any machine (including Colab)
    bundled = Path(__file__).resolve().parents[3] / "test_data" / "MNI152_2009_template.nii.gz"
    if bundled.exists():
        return str(bundled)
    afni_bin = shutil.which("afni")
    if afni_bin:
        return str(Path(afni_bin).parent / "MNI152_2009_template.nii.gz")
    return "MNI152_2009_template.nii.gz"


def _afni_ssw_template() -> str:
    """AFNI SSW template."""
    afni_bin = shutil.which("afni")
    if afni_bin:
        return str(Path(afni_bin).parent / "MNI152_2009_template_SSW.nii.gz")
    return "MNI152_2009_template_SSW.nii.gz"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        if not _afni_anat(ctx).exists():
            missing.append(str(_afni_anat(ctx)))
        if not _ffs_anat(ctx).exists():
            missing.append(str(_ffs_anat(ctx)))
    else:
        if not _anat_input(ctx).exists():
            missing.append(str(_anat_input(ctx)))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run sswarper2 (skull strip + nonlinear warp to MNI)."""
    ssw_dir = ctx.processing_dir / "sswarper_output"
    if _afni_anat(ctx).exists() and not ctx.force_ref:
        return 0.0

    elapsed, _ = run_timed(
        f"sswarper2 "
        f"-input {_anat_input(ctx)} "
        f"-base {_afni_ssw_template()} "
        f"-subid {_subid(ctx)} "
        f"-odir {ssw_dir}",
        label="sswarper2",
        cwd=ctx.processing_dir,
    )
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_allineate + ffs_qwarp to MNI."""
    ffs_dir = ctx.processing_dir / f"ffs_warper{ctx.ffs_tag}"
    ffs_dir.mkdir(exist_ok=True)

    total = 0.0

    # Need AFNI skull-stripped anat as input
    anatSS = ctx.processing_dir / "sswarper_output" / f"anatSS.{_subid(ctx)}.nii"
    if not anatSS.exists():
        raise RuntimeError(f"Need AFNI skull-stripped anat: {anatSS}")

    al_out = ffs_dir / f"al_ffs_anatSS.{_subid(ctx)}.nii"
    if not al_out.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_allineate "
            f"-source {anatSS} "
            f"-base {_afni_template()} "
            f"-prefix {al_out} "
            f"-source_automask -autoweight "
            f"-cost lpa "
            f"{ctx.ffs_device_flag()}",
            label="ffs_allineate anat-to-MNI",
            cwd=ffs_dir,
        )
        total += elapsed

    qwarp_out = _ffs_anat(ctx)
    if not qwarp_out.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_qwarp "
            f"-source {al_out} "
            f"-base {_afni_template()} "
            f"-minpatch 11 -lpa "
            f"-prefix {qwarp_out}"
            f"{ctx.ffs_device_flag()}",
            label="ffs_qwarp anat-to-MNI",
            cwd=ffs_dir,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare AFNI sswarper output (anatQQ) vs FFS allineate+qwarp (anatFFS)."""
    result = compare_volumes(_afni_anat(ctx), _ffs_anat(ctx))
    passed = result["r"] >= THRESHOLDS["warped_anat_r"]

    return {
        "passed": passed,
        "summary": f"anatQQ vs anatFFS r={result['r']:.4f}",
        **result,
    }
