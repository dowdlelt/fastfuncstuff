"""GLM benchmark: 3dDeconvolve + 3dREMLfit vs ffs_reml."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_bucket_volumes

name = "glm"
description = "GLM/REML (3dDeconvolve + 3dREMLfit vs ffs_reml)"

RUNS = [1, 2, 3, 4, 5]

THRESHOLDS = {
    "ols_beta_min_r": 0.99,
    "reml_beta_min_r": 0.95,
}

# Timing file info for 3dDeconvolve
STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]
GLTS = [
    ("faces_vs_objects", "+1*faces -1*objects"),
    ("faces_vs_scenes", "+1*faces -1*scenes"),
    ("faces_vs_scrambled", "+1*faces -1*scrambled"),
]


def _scaled_input(ctx: BenchmarkContext, run: int) -> Path:
    return ctx.processing_dir / f"scaled_afni_mni_task-localizer_run-{run}.nii.gz"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    p = ctx.processing_dir

    if ctx.validate_only:
        # OLS
        if not (p / "afni_stats_localizer.nii.gz").exists():
            missing.append(str(p / "afni_stats_localizer.nii.gz"))
        if not (p / "ffs_stats_localizer.nii.gz").exists():
            missing.append(str(p / "ffs_stats_localizer.nii.gz"))

        # REML - AFNI produces BRIK/HEAD
        afni_reml = p / "afni_stats_localizer_REML+tlrc.HEAD"
        if not afni_reml.exists():
            missing.append(str(afni_reml))
        if not (p / "ffs_stats_localizer_REML.nii.gz").exists():
            missing.append(str(p / "ffs_stats_localizer_REML.nii.gz"))

        # REML var
        afni_var = p / "afni_stats_localizer_REMLvar+tlrc.HEAD"
        if not afni_var.exists():
            missing.append(str(afni_var))
        if not (p / "ffs_stats_localizer_REML_var.nii.gz").exists():
            missing.append(str(p / "ffs_stats_localizer_REML_var.nii.gz"))
    else:
        # Need scaled inputs and timing files
        for run in RUNS:
            si = _scaled_input(ctx, run)
            if not si.exists():
                missing.append(str(si))
        timing_dir = p / "timing_files"
        if not timing_dir.exists():
            missing.append(str(timing_dir))

    return missing


def _prepare_scaled_inputs(ctx: BenchmarkContext) -> None:
    """Scale warped data (percent signal change) if not already done."""
    p = ctx.processing_dir
    for run in RUNS:
        scaled = _scaled_input(ctx, run)
        if scaled.exists():
            continue
        src = p / f"afni_mni_task-localizer_run-{run}.nii.gz"
        run_timed(
            f"3dTstat -overwrite -prefix mean_this_localizer.nii.gz {src}",
            label=f"3dTstat scale run-{run}",
            cwd=p,
        )
        run_timed(
            f"3dcalc -overwrite -prefix {scaled} "
            f"-a {src} -b mean_this_localizer.nii.gz "
            f"-expr 'a/b*100'",
            label=f"3dcalc scale run-{run}",
            cwd=p,
        )


def _prepare_timing_files(ctx: BenchmarkContext) -> None:
    """Extract timing files from events TSVs if not already done."""
    p = ctx.processing_dir
    timing_dir = p / "timing_files"
    if timing_dir.exists() and any(timing_dir.iterdir()):
        return
    timing_dir.mkdir(exist_ok=True)
    events = " ".join(
        str(ctx.func_dir / f"sub-01_ses-01_task-localizer_run-{r}_events.tsv")
        for r in RUNS
    )
    run_timed(
        f"timing_tool.py -write_multi_timing {timing_dir}/onsets.localizer. "
        f"-multi_timing_ncol_tsv {events}",
        label="timing_tool.py",
        cwd=p,
    )


def run_afni(ctx: BenchmarkContext) -> float:
    """Run 3dDeconvolve + 3dREMLfit."""
    p = ctx.processing_dir

    _prepare_scaled_inputs(ctx)
    _prepare_timing_files(ctx)

    total = 0.0

    # 3dDeconvolve (OLS)
    afni_ols = p / "afni_stats_localizer.nii.gz"
    if not afni_ols.exists() or ctx.force_afni:
        inputs = " ".join(str(_scaled_input(ctx, r)) for r in RUNS)
        timing_dir = p / "timing_files"

        stim_args = ""
        for i, label in enumerate(STIM_LABELS, 1):
            stim_args += (
                f"-stim_times {i} {timing_dir}/onsets.localizer.times.{label}.txt "
                f"'SPMG1(3)' -stim_label {i} {label} "
            )

        glt_args = ""
        for i, (label, sym) in enumerate(GLTS, 1):
            glt_args += f"-gltsym 'SYM: {sym}' -glt_label {i} {label} "

        elapsed, _ = run_timed(
            f"3dDeconvolve -overwrite "
            f"-input {inputs} "
            f"-polort A -num_stimts {len(STIM_LABELS)} -float "
            f"{stim_args} "
            f"-jobs 10 -noFDR "
            f"{glt_args} "
            f"-tout -x1D X.xmat.1D -bucket {afni_ols}",
            label="3dDeconvolve",
            cwd=p,
        )
        total += elapsed

    # 3dREMLfit
    afni_reml = p / "afni_stats_localizer_REML+tlrc.HEAD"
    if not afni_reml.exists() or ctx.force_afni:
        reml_cmd = p / "afni_stats_localizer.REML_cmd"
        if reml_cmd.exists():
            elapsed, _ = run_timed(
                f"tcsh {reml_cmd} -overwrite -nofdr",
                label="3dREMLfit",
                cwd=p,
            )
            total += elapsed

    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_reml (OLS + REML)."""
    p = ctx.processing_dir
    inputs = " ".join(str(_scaled_input(ctx, r)) for r in RUNS)
    total = 0.0

    # OLS
    ffs_ols = p / "ffs_stats_localizer.nii.gz"
    if not ffs_ols.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml "
            f"-input {inputs} "
            f"-matrix X.xmat.1D "
            f"-use_double "
            f"-Obuck {ffs_ols} "
            f"-tout",
            label="ffs_reml OLS",
            cwd=p,
        )
        total += elapsed

    # REML
    ffs_reml = p / "ffs_stats_localizer_REML.nii.gz"
    ffs_var = p / "ffs_stats_localizer_REML_var.nii.gz"
    if not ffs_reml.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml "
            f"-input {inputs} "
            f"-matrix X.xmat.1D "
            f"-use_double "
            f"-Rbuck {ffs_reml} "
            f"-Rvar {ffs_var} "
            f"-tout",
            label="ffs_reml REML",
            cwd=p,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare OLS and REML results between AFNI and FFS."""
    p = ctx.processing_dir

    # OLS comparison
    ols_result = compare_bucket_volumes(
        p / "afni_stats_localizer.nii.gz",
        p / "ffs_stats_localizer.nii.gz",
    )

    # REML comparison - AFNI uses BRIK/HEAD format
    reml_result = compare_bucket_volumes(
        p / "afni_stats_localizer_REML+tlrc.HEAD",
        p / "ffs_stats_localizer_REML.nii.gz",
    )

    # REML variance parameters
    var_result = compare_bucket_volumes(
        p / "afni_stats_localizer_REMLvar+tlrc.HEAD",
        p / "ffs_stats_localizer_REML_var.nii.gz",
    )

    passed = (
        ols_result["min_r"] >= THRESHOLDS["ols_beta_min_r"]
        and reml_result["min_r"] >= THRESHOLDS["reml_beta_min_r"]
    )

    return {
        "passed": passed,
        "summary": (
            f"OLS min_r={ols_result['min_r']:.4f}, "
            f"REML min_r={reml_result['min_r']:.4f}, "
            f"var min_r={var_result['min_r']:.4f}"
        ),
        "ols": ols_result,
        "reml": reml_result,
        "reml_var": var_result,
    }
