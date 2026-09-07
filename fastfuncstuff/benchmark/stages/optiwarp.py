"""Nonlinear anat-to-MNI benchmark: 3dQwarp vs ffs_optiwarp.

The *align* stage compares whole pipelines (sswarper2 against
ffs_allineate + ffs_qwarp), so its timing mixes skull-stripping and the
affine step in with the nonlinear warp. This stage isolates the nonlinear
step: both tools start from the SAME affine-aligned, skull-stripped
anatomical that align already produced, and both warp it to the same MNI
template. What differs is only the deformation engine -- AFNI's patch
optimizer versus the optical-flow solver -- so the times are comparable
and the correlation between the two outputs is a like-for-like agreement
measure.

That also makes this the stage to run with ``-device cpu``: optical flow
is a handful of convolutions and gathers per iteration, so unlike qwarp's
patch search it has a real chance of staying respectable without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_volumes
from .align import _afni_anat, _afni_template, _ffs_anat, _subid

name = "optiwarp"
description = "Nonlinear anat-to-MNI (3dQwarp vs ffs_optiwarp)"
# align produces the affine-aligned anatSS both engines consume, plus the
# sswarper anatQQ that validate() uses as the third reference point.
requires = ["align"]

THRESHOLDS = {
    # Same input, same target, different deformation engines. Held to a
    # tighter bar than align's 0.80 because the affine step -- the largest
    # source of disagreement there -- is shared here.
    "optiwarp_vs_qwarp_r": 0.85,
    "optiwarp_vs_sswarper_r": 0.80,
}

DEFAULTS = {
    "type": "MNI_T1",  # tuned preset (tunespec.PRESETS)
    "metric": "lpa",  # the tool default (cc) folds on same-modality data
    "extra_args": "",
    "ref_minpatch": 11,
    "ref_cost": "lpa",
    "ref_extra_args": "",
}


def _params(ctx: BenchmarkContext) -> dict:
    return {**DEFAULTS, **ctx.get_stage_params(name)}


def _affine_dir(ctx: BenchmarkContext) -> Path:
    """The align stage's output dir holding the affine-aligned anatSS.

    Prefers this run's device-tagged dir, but falls back to the untagged one:
    a CPU run of this stage should be able to reuse a GPU align's affine
    result rather than demanding align be re-run per device.
    """
    tagged = ctx.processing_dir / f"ffs_warper{ctx.ffs_tag}"
    if (tagged / f"al_ffs_anatSS.{_subid(ctx)}.nii").exists():
        return tagged
    untagged = ctx.processing_dir / "ffs_warper"
    if (untagged / f"al_ffs_anatSS.{_subid(ctx)}.nii").exists():
        return untagged
    return tagged


def _affine_tag(ctx: BenchmarkContext) -> str:
    """Tag of the dir the affine input actually came from, for output naming."""
    return _affine_dir(ctx).name[len("ffs_warper") :]


def _affine_source(ctx: BenchmarkContext) -> Path:
    return _affine_dir(ctx) / f"al_ffs_anatSS.{_subid(ctx)}.nii"


def _qwarp_ref(ctx: BenchmarkContext) -> Path:
    # Named for the affine input it consumed, not for this run's device: the
    # reference is CPU AFNI either way, and re-running it per device would
    # burn minutes to reproduce the same file.
    return ctx.processing_dir / f"afni_qwarp_anat{_affine_tag(ctx)}.{_subid(ctx)}.nii.gz"


def _optiwarp_out(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / f"ffs_optiwarp{ctx.ffs_tag}" / f"anatOPTI.{_subid(ctx)}.nii.gz"


def validation_inputs(ctx: BenchmarkContext) -> list[Path]:
    """Files validate() reads. anatFFS is optional (informational only)."""
    return [_afni_anat(ctx), _qwarp_ref(ctx), _optiwarp_out(ctx)]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        missing.extend(str(p) for p in validation_inputs(ctx) if not p.exists())
    else:
        if not _affine_source(ctx).exists():
            missing.append(str(_affine_source(ctx)))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """3dQwarp from the affine-aligned anat to the MNI template."""
    out = _qwarp_ref(ctx)
    if out.exists() and not ctx.force_ref:
        ctx.note_items("ref", 0, 1)
        return 0.0

    p = _params(ctx)
    elapsed, _ = run_timed(
        f"3dQwarp -overwrite "
        f"-base {_afni_template()} "
        f"-source {_affine_source(ctx)} "
        f"-prefix {out} "
        f"-minpatch {p['ref_minpatch']} -{p['ref_cost']} "
        f"{p['ref_extra_args']}",
        label="3dQwarp anat-to-MNI",
        cwd=ctx.processing_dir,
    )
    ctx.note_items("ref", 1, 1)
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """ffs_optiwarp from the same affine-aligned anat to the same template."""
    out = _optiwarp_out(ctx)
    out.parent.mkdir(exist_ok=True)
    if out.exists() and not ctx.force_ffs:
        ctx.note_items("ffs", 0, 1)
        return 0.0

    p = _params(ctx)
    elapsed, _ = run_timed(
        f"ffs_optiwarp "
        f"-base {_afni_template()} "
        f"-source {_affine_source(ctx)} "
        f"-prefix {out} "
        f"-type {p['type']} -metric {p['metric']} -save_warp "
        f"{p['extra_args']}"
        f"{ctx.ffs_device_flag()}",
        label="ffs_optiwarp anat-to-MNI",
        cwd=out.parent,
    )
    ctx.note_items("ffs", 1, 1)
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Correlate the optical-flow warp against both AFNI reference points."""
    vs_qwarp = compare_volumes(_qwarp_ref(ctx), _optiwarp_out(ctx))
    vs_ssw = compare_volumes(_afni_anat(ctx), _optiwarp_out(ctx))

    result = {
        "r": vs_qwarp["r"],  # headline: the like-for-like comparison
        "optiwarp_vs_qwarp_r": vs_qwarp["r"],
        "optiwarp_vs_sswarper_r": vs_ssw["r"],
        "n_voxels": vs_qwarp["n_voxels"],
    }
    summary = f"optiwarp vs 3dQwarp r={vs_qwarp['r']:.4f}, vs anatQQ r={vs_ssw['r']:.4f}"

    # The ffs_qwarp anat is align's output, not ours -- report the three-way
    # agreement when it happens to exist, but never gate on it.
    ffs_qwarp = _ffs_anat(ctx)
    if ffs_qwarp.exists():
        vs_ffs = compare_volumes(ffs_qwarp, _optiwarp_out(ctx))
        result["optiwarp_vs_ffs_qwarp_r"] = vs_ffs["r"]
        summary += f", vs anatFFS r={vs_ffs['r']:.4f}"

    result["passed"] = (
        vs_qwarp["r"] >= THRESHOLDS["optiwarp_vs_qwarp_r"]
        and vs_ssw["r"] >= THRESHOLDS["optiwarp_vs_sswarper_r"]
    )
    result["summary"] = summary
    return result
