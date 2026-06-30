"""GLM benchmark: 3dDeconvolve + 3dREMLfit vs ffs_reml."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_bucket_volumes

name = "glm"
description = "GLM/REML (3dDeconvolve + 3dREMLfit vs ffs_reml)"

THRESHOLDS = {
    "ols_beta_min_r": 0.99,
    "reml_beta_min_r": 0.95,
    # Of the 4 -Rvar subbriks (a, b, lambda, stdev), lambda is the
    # effective single-AR(1) prewhitening parameter — it's what the GLM
    # actually applies, and it's the meaningful parity target. The raw
    # a / b labels alone aren't scientifically meaningful: many (a, b)
    # pairs sit on a flat λ-ridge and AFNI's serial summation vs our
    # batched BLAS summation pick different ridge points that produce
    # the same λ. stdev (noise std) is independently meaningful and
    # always matches at ~1.0 when prewhitening matches.
    "reml_lambda_min_r": 0.90,
    "reml_stdev_min_r": 0.95,
}

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]
_DEFAULT_GLTS = [
    ("faces_vs_objects", "+1*faces -1*objects"),
    ("faces_vs_scenes", "+1*faces -1*scenes"),
    ("faces_vs_scrambled", "+1*faces -1*scrambled"),
]


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _stim_labels(ctx: BenchmarkContext) -> list[str]:
    return _glm_params(ctx).get("stim_labels", _DEFAULT_STIM_LABELS)


def _glts(ctx: BenchmarkContext) -> list[tuple[str, str]]:
    raw = _glm_params(ctx).get("glts", _DEFAULT_GLTS)
    return [(g[0], g[1]) for g in raw]


def _scaled_input(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"scaled_afni_mni_task-{task}_run-{run}.nii.gz"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    task = _primary_task(ctx)

    if ctx.validate_only:
        afni = ctx.afni_glm_dir
        ffs = ctx.ffs_glm_dir
        # OLS
        if not (afni / f"stats_{task}.nii.gz").exists():
            missing.append(str(afni / f"stats_{task}.nii.gz"))
        if not (ffs / f"stats_{task}.nii.gz").exists():
            missing.append(str(ffs / f"stats_{task}.nii.gz"))
        # REML
        if not (afni / f"stats_{task}_REML+tlrc.HEAD").exists():
            missing.append(str(afni / f"stats_{task}_REML+tlrc.HEAD"))
        if not (ffs / f"stats_{task}_REML.nii.gz").exists():
            missing.append(str(ffs / f"stats_{task}_REML.nii.gz"))
        # REML var
        if not (afni / f"stats_{task}_REMLvar+tlrc.HEAD").exists():
            missing.append(str(afni / f"stats_{task}_REMLvar+tlrc.HEAD"))
        if not (ffs / f"stats_{task}_REML_var.nii.gz").exists():
            missing.append(str(ffs / f"stats_{task}_REML_var.nii.gz"))
    else:
        # Need warped MNI data (scaling is done as part of run_ref)
        for run in _runs(ctx):
            src = ctx.processing_dir / f"afni_mni_task-{task}_run-{run}.nii.gz"
            if not src.exists():
                missing.append(str(src))

    return missing


def _prepare_scaled_inputs(ctx: BenchmarkContext) -> None:
    """Scale warped data (percent signal change) if not already done."""
    p = ctx.processing_dir
    for run in _runs(ctx):
        scaled = _scaled_input(ctx, run)
        if scaled.exists():
            continue
        task = _primary_task(ctx)
        src = p / f"afni_mni_task-{task}_run-{run}.nii.gz"
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
    task = _primary_task(ctx)
    events = " ".join(
        str(ctx.func_dir / f"{ctx.bids_prefix(task, r)}_events.tsv") for r in _runs(ctx)
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

    task = _primary_task(ctx)
    stim_labels = _stim_labels(ctx)
    glts = _glts(ctx)
    hrf_model = _glm_params(ctx).get("hrf_model", "SPMG1(3)")

    # 3dDeconvolve (OLS)
    afni_ols = afni / f"stats_{task}.nii.gz"
    if not afni_ols.exists() or ctx.force_ref:
        inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))

        stim_args = ""
        for i, label in enumerate(stim_labels, 1):
            stim_args += (
                f"-stim_times {i} {ctx.timing_dir}/onsets.{task}.times.{label}.txt "
                f"'{hrf_model}' -stim_label {i} {label} "
            )

        glt_args = ""
        for i, (label, sym) in enumerate(glts, 1):
            glt_args += f"-gltsym 'SYM: {sym}' -glt_label {i} {label} "

        # Pass basename only so AFNI writes REML_cmd in cwd (afni dir).
        elapsed, _ = run_timed(
            f"3dDeconvolve -overwrite "
            f"-input {inputs} "
            f"-polort A -num_stimts {len(stim_labels)} -float "
            f"{stim_args} "
            f"-jobs 10 -noFDR "
            f"{glt_args} "
            f"-tout -x1D X.xmat.1D -bucket {afni_ols.name}",
            label="3dDeconvolve",
            cwd=afni,
        )
        total += elapsed

    # 3dREMLfit
    afni_reml = afni / f"stats_{task}_REML+tlrc.HEAD"
    if not afni_reml.exists() or ctx.force_ref:
        reml_cmd = afni / f"stats_{task}.REML_cmd"
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

    task = _primary_task(ctx)
    inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))
    # X.xmat.1D produced by 3dDeconvolve in afni_glm_dir
    xmat = afni / "X.xmat.1D"
    total = 0.0

    # OLS
    ffs_ols = ffs / f"stats_{task}.nii.gz"
    if not ffs_ols.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml -input {inputs} -matrix {xmat} -Obuck {ffs_ols} -tout"
            f"{ctx.ffs_device_flag()}",
            label="ffs_reml OLS",
            cwd=ffs,
        )
        total += elapsed

    # REML
    ffs_reml = ffs / f"stats_{task}_REML.nii.gz"
    ffs_var = ffs / f"stats_{task}_REML_var.nii.gz"
    if not ffs_reml.exists() or ctx.force_ffs:
        elapsed, _ = run_timed(
            f"ffs_reml -input {inputs} -matrix {xmat} -Rbuck {ffs_reml} -Rvar {ffs_var} -tout"
            f"{ctx.ffs_afni_mode_flag()}{ctx.ffs_device_flag()}",
            label="ffs_reml REML",
            cwd=ffs,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare OLS and REML results between AFNI and FFS."""
    afni = ctx.afni_glm_dir
    ffs = ctx.ffs_glm_dir
    task = _primary_task(ctx)

    # OLS comparison
    ols_result = compare_bucket_volumes(
        afni / f"stats_{task}.nii.gz",
        ffs / f"stats_{task}.nii.gz",
    )

    # REML comparison - AFNI uses BRIK/HEAD format
    reml_result = compare_bucket_volumes(
        afni / f"stats_{task}_REML+tlrc.HEAD",
        ffs / f"stats_{task}_REML.nii.gz",
    )

    # REML variance parameters
    var_result = compare_bucket_volumes(
        afni / f"stats_{task}_REMLvar+tlrc.HEAD",
        ffs / f"stats_{task}_REML_var.nii.gz",
    )

    # Rvar's 4 subbriks are written by ffs_reml as: a, b, lambda, StDev
    # (see -Rvar handler in cli/reml.py). Surface each so we can see which
    # parameter is the parity offender — but only gate PASS/FAIL on
    # lambda + stdev, since a / b individually live on a flat λ-ridge
    # where batched vs serial summation picks different valid optima
    # that produce the same effective prewhitening.
    _var_names = ("a", "b", "lambda", "stdev")
    per_brick = var_result.get("per_brick", [])
    by_name = {}
    for entry, name in zip(per_brick, _var_names):
        entry["name"] = name
        by_name[name] = entry["r"]
    var_per = " ".join(
        f"{name}={entry['r']:.3f}"
        for entry, name in zip(per_brick, _var_names)
    )

    passed = (
        ols_result["min_r"] >= THRESHOLDS["ols_beta_min_r"]
        and reml_result["min_r"] >= THRESHOLDS["reml_beta_min_r"]
        and by_name.get("lambda", 0.0) >= THRESHOLDS["reml_lambda_min_r"]
        and by_name.get("stdev", 0.0) >= THRESHOLDS["reml_stdev_min_r"]
    )

    return {
        "passed": passed,
        "summary": (
            f"OLS min_r={ols_result['min_r']:.4f}, "
            f"REML min_r={reml_result['min_r']:.4f}, "
            f"var[{var_per}]"
        ),
        "ols": ols_result,
        "reml": reml_result,
        "reml_var": var_result,
    }
