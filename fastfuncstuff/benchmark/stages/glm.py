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

    if ctx.validate_only:
        afni = ctx.afni_glm_dir
        ffs = ctx.ffs_glm_dir
        # OLS
        if not (afni / "stats_localizer.nii.gz").exists():
            missing.append(str(afni / "stats_localizer.nii.gz"))
        if not (ffs / "stats_localizer.nii.gz").exists():
            missing.append(str(ffs / "stats_localizer.nii.gz"))
        # REML
        if not (afni / "stats_localizer_REML+tlrc.HEAD").exists():
            missing.append(str(afni / "stats_localizer_REML+tlrc.HEAD"))
        if not (ffs / "stats_localizer_REML.nii.gz").exists():
            missing.append(str(ffs / "stats_localizer_REML.nii.gz"))
        # REML var
        if not (afni / "stats_localizer_REMLvar+tlrc.HEAD").exists():
            missing.append(str(afni / "stats_localizer_REMLvar+tlrc.HEAD"))
        if not (ffs / "stats_localizer_REML_var.nii.gz").exists():
            missing.append(str(ffs / "stats_localizer_REML_var.nii.gz"))
    else:
        # Need warped MNI data (scaling is done as part of run_ref)
        for run in RUNS:
            src = ctx.processing_dir / f"afni_mni_task-localizer_run-{run}.nii.gz"
            if not src.exists():
                missing.append(str(src))

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
    timing_dir = ctx.timing_dir
    if timing_dir.exists() and any(timing_dir.iterdir()):
        return
    timing_dir.mkdir(exist_ok=True)
    events = " ".join(
        str(ctx.func_dir / f"sub-01_ses-01_task-localizer_run-{r}_events.tsv") for r in RUNS
    )
    run_timed(
        f"timing_tool.py -write_multi_timing {timing_dir}/onsets.localizer. "
        f"-multi_timing_ncol_tsv {events}",
        label="timing_tool.py",
        cwd=ctx.processing_dir,
    )


def run_ref(ctx: BenchmarkContext) -> float:
    """Run 3dDeconvolve + 3dREMLfit."""
    p = ctx.processing_dir
    afni = ctx.afni_glm_dir
    afni.mkdir(parents=True, exist_ok=True)

    _prepare_scaled_inputs(ctx)
    _prepare_timing_files(ctx)

    total = 0.0

    # 3dDeconvolve (OLS)
    afni_ols = afni / "stats_localizer.nii.gz"
    if not afni_ols.exists() or ctx.force_ref:
        inputs = " ".join(str(_scaled_input(ctx, r)) for r in RUNS)

        stim_args = ""
        for i, label in enumerate(STIM_LABELS, 1):
            stim_args += (
                f"-stim_times {i} {ctx.timing_dir}/onsets.localizer.times.{label}.txt "
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
            cwd=afni,
        )
        total += elapsed

    # 3dREMLfit
    afni_reml = afni / "stats_localizer_REML+tlrc.HEAD"
    if not afni_reml.exists() or ctx.force_ref:
        reml_cmd = afni / "stats_localizer.REML_cmd"
        if reml_cmd.exists():
            elapsed, _ = run_timed(
                f"tcsh {reml_cmd} -overwrite -nofdr",
                label="3dREMLfit",
                cwd=afni,
            )
            total += elapsed

    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_reml (OLS + REML)."""
    afni = ctx.afni_glm_dir
    ffs = ctx.ffs_glm_dir
    ffs.mkdir(parents=True, exist_ok=True)

    inputs = " ".join(str(_scaled_input(ctx, r)) for r in RUNS)
    # X.xmat.1D produced by 3dDeconvolve in afni_glm_dir
    xmat = afni / "X.xmat.1D"
    total = 0.0

    # OLS
    ffs_ols = ffs / "stats_localizer.nii.gz"
    if not ffs_ols.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml -input {inputs} -matrix {xmat} -Obuck {ffs_ols} -tout"
            f"{ctx.ffs_device_flag()}",
            label="ffs_reml OLS",
            cwd=ffs,
        )
        total += elapsed

    # REML
    ffs_reml = ffs / "stats_localizer_REML.nii.gz"
    ffs_var = ffs / "stats_localizer_REML_var.nii.gz"
    if not ffs_reml.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml -input {inputs} -matrix {xmat} -Rbuck {ffs_reml} -Rvar {ffs_var} -tout"
            f"{ctx.ffs_device_flag()}",
            label="ffs_reml REML",
            cwd=ffs,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare OLS and REML results between AFNI and FFS."""
    afni = ctx.afni_glm_dir
    ffs = ctx.ffs_glm_dir

    # OLS comparison
    ols_result = compare_bucket_volumes(
        afni / "stats_localizer.nii.gz",
        ffs / "stats_localizer.nii.gz",
    )

    # REML comparison - AFNI uses BRIK/HEAD format
    reml_result = compare_bucket_volumes(
        afni / "stats_localizer_REML+tlrc.HEAD",
        ffs / "stats_localizer_REML.nii.gz",
    )

    # REML variance parameters
    var_result = compare_bucket_volumes(
        afni / "stats_localizer_REMLvar+tlrc.HEAD",
        ffs / "stats_localizer_REML_var.nii.gz",
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
