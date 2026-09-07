"""Nonlinear anat-to-MNI: ffs_optiwarp against the align stage's references.

The *align* stage warps the affine-aligned, skull-stripped anatomical to
MNI with ffs_qwarp and scores it against AFNI's sswarper anatQQ. This
stage puts the optical-flow engine on exactly that footing: same affine
input, same template, same reference. So no reference tool runs here --
anatQQ already exists, and re-running 3dQwarp on our own affine input
would cost tens of minutes to produce a second copy of a comparison we
already have.

That leaves two numbers worth reading: agreement with anatQQ, directly
comparable to what align reports for qwarp, and agreement with anatFFS,
which is the two FFS engines head to head on identical input.

It is also the stage worth running with ``-device cpu``. Optical flow is
convolutions and gathers per iteration rather than a patch search, so it
has a real chance of staying respectable without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_volumes
from .align import _afni_anat, _afni_template, _subid

name = "optiwarp"
description = "Nonlinear anat-to-MNI (ffs_optiwarp vs sswarper/ffs_qwarp)"
# align produces the affine-aligned anatSS this consumes, the sswarper anatQQ
# it is scored against, and the ffs_qwarp anatFFS it is compared to.
requires = ["align"]

THRESHOLDS = {
    # Same bar align holds ffs_qwarp to, for the same comparison.
    "optiwarp_vs_sswarper_r": 0.80,
}

DEFAULTS = {
    "type": "MNI_T1",  # tuned preset (tunespec.PRESETS)
    "metric": "lpa",  # the tool default (cc) folds on same-modality data
    "extra_args": "",
}


def _params(ctx: BenchmarkContext) -> dict:
    return {**DEFAULTS, **ctx.get_stage_params(name)}


def _affine_dir(ctx: BenchmarkContext) -> Path:
    """The align stage's output dir: affine-aligned anatSS plus anatFFS.

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


def _affine_source(ctx: BenchmarkContext) -> Path:
    return _affine_dir(ctx) / f"al_ffs_anatSS.{_subid(ctx)}.nii"


def _qwarp_anat(ctx: BenchmarkContext) -> Path:
    """align's ffs_qwarp result, from whichever dir the affine input came from."""
    return _affine_dir(ctx) / f"anatFFS.{_subid(ctx)}.nii.gz"


def _optiwarp_out(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / f"ffs_optiwarp{ctx.ffs_tag}" / f"anatOPTI.{_subid(ctx)}.nii.gz"


def validation_inputs(ctx: BenchmarkContext) -> list[Path]:
    """Files validate() reads. anatFFS is reported when present, never required."""
    return [_afni_anat(ctx), _optiwarp_out(ctx)]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        missing.extend(str(p) for p in validation_inputs(ctx) if not p.exists())
    else:
        if not _affine_source(ctx).exists():
            missing.append(str(_affine_source(ctx)))
    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """ffs_optiwarp from align's affine-aligned anat to the MNI template."""
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
    """Score the optical-flow warp against anatQQ, and against anatFFS."""
    vs_ssw = compare_volumes(_afni_anat(ctx), _optiwarp_out(ctx))

    result = {
        "r": vs_ssw["r"],  # headline: the same comparison align reports
        "optiwarp_vs_sswarper_r": vs_ssw["r"],
        "n_voxels": vs_ssw["n_voxels"],
    }
    summary = f"anatQQ vs anatOPTI r={vs_ssw['r']:.4f}"

    # anatFFS is align's output, not ours -- report the head-to-head when it
    # exists, but never gate on it.
    qwarp_anat = _qwarp_anat(ctx)
    if qwarp_anat.exists():
        vs_qwarp = compare_volumes(qwarp_anat, _optiwarp_out(ctx))
        result["optiwarp_vs_ffs_qwarp_r"] = vs_qwarp["r"]
        summary += f", vs anatFFS r={vs_qwarp['r']:.4f}"

    result["passed"] = vs_ssw["r"] >= THRESHOLDS["optiwarp_vs_sswarper_r"]
    result["summary"] = summary
    return result
