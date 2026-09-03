"""Warping benchmark: 3dNwarpApply vs ffs_nwarp."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_timeseries_4d, compare_volumes

name = "warp"
description = "Warp apply (3dNwarpApply vs ffs_nwarp)"
# Upstream stages whose outputs this stage consumes: moco (per-run motion
# matrices + reference mean), crossalign (inter-run alignment matrices), align
# (sswarper anatQQ/anatSS + the ffs_qwarp anatFFS that validate() compares).
requires = ["moco", "crossalign", "align"]

THRESHOLDS = {
    "warped_anat_r": 0.80,  # different warping algorithms, expect some disagreement
    "warped_func_min_r": 0.95,
}


def _ref_task_run(ctx: BenchmarkContext) -> tuple[str, int]:
    params = ctx.get_stage_params("crossalign")
    return params.get("reference_task", "localizer"), params.get("reference_run", 1)


def _afni_anat(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "sswarper_output" / f"anatQQ.sub-{ctx.subject}.nii"


def _ffs_anat(ctx: BenchmarkContext) -> Path:
    # Tagged to match what the align stage writes (CPU vs GPU separation).
    return ctx.processing_dir / f"ffs_warper{ctx.ffs_tag}" / f"anatFFS.sub-{ctx.subject}.nii.gz"


def _afni_mni(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mni_task-{task}_run-{run}.nii.gz"


def _ffs_mni(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"ffs_mni_task-{task}_run-{run}.nii.gz"


def _input_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.func_dir / f"{ctx.bids_prefix(task, run)}_bold.nii"


def _nwarp_chain(ctx: BenchmarkContext, task: str, run: int) -> str:
    """Build the -nwarp chain string for 3dNwarpApply / ffs_nwarp."""
    p = ctx.processing_dir
    ssw = p / "sswarper_output"
    subid = f"sub-{ctx.subject}"
    ref_task, ref_run = _ref_task_run(ctx)

    parts = [
        str(ssw / f"anatQQ.{subid}_WARP.nii"),
        str(ssw / f"anatQQ.{subid}.aff12.1D"),
        str(p / "anat_al_keep_e2a_only_mat.aff12.1D"),
    ]

    # Reference run uses the moco matrix directly;
    # other runs need the inter-run alignment matrix first
    if not (task == ref_task and run == ref_run):
        align_mat = p / f"afni_mean_{task}_run-{run}_to_{ref_task}_run-{ref_run}_mat.aff12.1D"
        parts.append(str(align_mat))

    moco_mat = p / f"afni_moco_{ctx.bids_prefix(task, run)}_bold_mat.aff12.1D"
    parts.append(str(moco_mat))

    return " ".join(parts)


def validation_inputs(ctx: BenchmarkContext) -> list[Path]:
    """Files validate() reads: warped anatomicals + all warped functionals.

    ``anatFFS`` is produced by the *align* stage (ffs_qwarp), not by warp, so a
    warp-only run with no prior align run will list it here -- surfaced as
    INCOMPLETE rather than a raw I/O error mid-validation.
    """
    paths = [_afni_anat(ctx), _ffs_anat(ctx)]
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            paths.append(_afni_mni(ctx, task, run))
            paths.append(_ffs_mni(ctx, task, run))
    return paths


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    subid = f"sub-{ctx.subject}"
    ref_task, ref_run = _ref_task_run(ctx)
    if ctx.validate_only:
        missing.extend(str(p) for p in validation_inputs(ctx) if not p.exists())
    else:
        # Need sswarper output (from align stage)
        ssw = ctx.processing_dir / "sswarper_output" / f"anatQQ.{subid}.nii"
        if not ssw.exists():
            missing.append(str(ssw))
        # Need moco reference mean
        ref_mean = ctx.processing_dir / f"afni_mean_{ctx.bids_prefix(ref_task, ref_run)}_bold.nii"
        if not ref_mean.exists():
            missing.append(str(ref_mean))
        # Need inter-run alignment matrices (from crossalign stage)
        for task, runs in ctx.all_task_run_pairs():
            for run in runs:
                if task == ref_task and run == ref_run:
                    continue
                mat = (
                    ctx.processing_dir
                    / f"afni_mean_{task}_run-{run}_to_{ref_task}_run-{ref_run}_mat.aff12.1D"
                )
                if not mat.exists():
                    missing.append(str(mat))
    return missing


def _prepare_warp_prerequisites(ctx: BenchmarkContext) -> None:
    """Create intermediate files needed for warping (autobox, e2a matrix).

    Inter-run alignment matrices are created by the crossalign stage.
    """
    p = ctx.processing_dir
    ssw = p / "sswarper_output"
    subid = f"sub-{ctx.subject}"
    ref_task, ref_run = _ref_task_run(ctx)

    # 1. Autobox the MNI-warped anat for a smaller master grid
    autobox = p / f"autobox_anatQQ.{subid}.nii"
    if not autobox.exists():
        run_timed(
            f"3dAutobox -overwrite -npad 3 "
            f"-prefix {ssw / f'autobox_anatQQ.{subid}.nii'} "
            f"{ssw / f'anatQQ.{subid}.nii'}",
            label="3dAutobox anatQQ",
            cwd=p,
        )
        import shutil as sh

        sh.copy2(ssw / f"autobox_anatQQ.{subid}.nii", autobox)

    # 2. EPI-to-anat alignment matrix (align_epi_anat.py + cat_matvec)
    e2a = p / "anat_al_keep_e2a_only_mat.aff12.1D"
    if not e2a.exists():
        ref_mean = f"afni_mean_{ctx.bids_prefix(ref_task, ref_run)}_bold.nii"
        run_timed(
            f"align_epi_anat.py -overwrite "
            f"-rigid_body -anat_has_skull no -anat2epi "
            f"-anat {ssw / f'anatSS.{subid}.nii'} "
            f"-epi {p / ref_mean} "
            f"-epi_base 0 -suffix _al",
            label="align_epi_anat.py",
            cwd=p,
        )
        run_timed(
            f"cat_matvec {p / f'anatSS.{subid}_al_mat.aff12.1D'} -I -ONELINE > {e2a}",
            label="cat_matvec e2a",
            cwd=p,
        )


def run_ref(ctx: BenchmarkContext) -> float:
    """Run 3dNwarpApply for all runs."""
    _prepare_warp_prerequisites(ctx)
    subid = f"sub-{ctx.subject}"
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"
    total = 0.0
    n_total = n_ran = 0

    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            n_total += 1
            out = _afni_mni(ctx, task, run)
            if out.exists() and not ctx.force_ref:
                continue
            nwarp = _nwarp_chain(ctx, task, run)
            elapsed, _ = run_timed(
                f"3dNwarpApply -overwrite "
                f"-master {master} -dxyz 3.0 -wsinc5 "
                f'-nwarp "{nwarp}" '
                f"-source {_input_path(ctx, task, run)} "
                f"-prefix {out}",
                label=f"3dNwarpApply {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed
            n_ran += 1
    ctx.note_items("ref", n_ran, n_total)
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_nwarp for all runs, in one batched process.

    Ten runs invoked separately pay the interpreter/torch/CUDA startup and the
    resampler's torch.compile warmup ten times over; -batch pays both once and
    leaves the per-run warp untouched. -master/-dxyz/-interp are the same for
    every run, so they stay on the outer command and each manifest line carries
    only its own warp chain, source and prefix.
    """
    subid = f"sub-{ctx.subject}"
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"

    jobs = []
    n_total = 0
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            n_total += 1
            out = _ffs_mni(ctx, task, run)
            if out.exists() and not ctx.force_ffs:
                continue
            jobs.append(
                f'-nwarp "{_nwarp_chain(ctx, task, run)}" '
                f"-source {_input_path(ctx, task, run)} "
                f"-prefix {out}"
            )
    ctx.note_items("ffs", len(jobs), n_total)

    if not jobs:
        return 0.0

    manifest = ctx.processing_dir / "ffs_warp_batch.txt"
    manifest.write_text("\n".join(jobs) + "\n")

    elapsed, _ = run_timed(
        f"ffs_nwarp -master {master} -dxyz 3.0 -interp wsinc5 -no_autopad "
        f"-batch {manifest}"
        f"{ctx.ffs_device_flag()}",
        label=f"ffs_nwarp batch ({len(jobs)} runs)",
        cwd=ctx.processing_dir,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare warped anatomicals and functionals between AFNI and FFS."""
    results = {}

    # 1. Compare warped anatomicals
    anat_result = compare_volumes(_afni_anat(ctx), _ffs_anat(ctx))
    results["anat_r"] = anat_result["r"]

    # 2. Compare warped functionals (mean correlation across runs)
    func_results = []
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            ts = compare_timeseries_4d(
                _afni_mni(ctx, task, run),
                _ffs_mni(ctx, task, run),
                sample_frac=0.05,
            )
            ts["task"] = task
            ts["run"] = run
            func_results.append(ts)

    func_median_rs = [r["median_r"] for r in func_results if "median_r" in r]
    func_min_r = min(func_median_rs) if func_median_rs else 0.0
    func_mean_r = sum(func_median_rs) / len(func_median_rs) if func_median_rs else 0.0

    passed = (
        anat_result["r"] >= THRESHOLDS["warped_anat_r"]
        and func_min_r >= THRESHOLDS["warped_func_min_r"]
    )

    return {
        "passed": passed,
        "summary": (f"anat r={anat_result['r']:.4f}, func min median_r={func_min_r:.4f}"),
        "anat": anat_result,
        "func_min_r": func_min_r,
        "func_mean_r": func_mean_r,
        "func_per_run": func_results,
    }
