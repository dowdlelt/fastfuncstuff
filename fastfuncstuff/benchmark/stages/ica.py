"""ICA benchmark: melodic vs ffs_ica."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_ica_components

name = "ica"
description = "ICA (melodic vs ffs_ica -temp_concat)"

THRESHOLDS = {
    "mean_matched_r": 0.70,
    "coverage_0.5": 0.80,
}


def _ica_tasks(ctx: BenchmarkContext) -> list[str]:
    """Tasks to run ICA on (from config or default to all tasks)."""
    params = ctx.get_stage_params("ica")
    return params.get("tasks", ctx.task_names())


def _melodic_dir(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.melodic_ica_dir / f"all_{dataset}_melodic.ica"


def _melodic_ic(ctx: BenchmarkContext, dataset: str) -> Path:
    return _melodic_dir(ctx, dataset) / "melodic_IC.nii.gz"


def _melodic_mask(ctx: BenchmarkContext, dataset: str) -> Path:
    return _melodic_dir(ctx, dataset) / "mask.nii.gz"


def _ffs_ic(ctx: BenchmarkContext, dataset: str) -> Path:
    """FFS GMM-z'd component maps (apples-to-apples vs MELODIC's `melodic_IC`).

    MELODIC's `melodic_IC.nii.gz` stores the GGM-mixture-modeled z-stat maps,
    not the raw ICA components. We therefore compare against ffs_ica's
    `*_concat_ica_zmaps.nii.gz` (output of `batch_mixture_zscores` ⇒ same
    space as MELODIC's IC). Lives inside the `.ica/ffs_outputs/` subfolder
    after the output-layout refactor.
    """
    base = f"all_{dataset}"
    return (
        ctx.ffs_ica_dir
        / f"{base}_concat.ica"
        / "ffs_outputs"
        / f"{base}_concat_ica_zmaps.nii.gz"
    )


def _mni_inputs(ctx: BenchmarkContext, dataset: str) -> list[Path]:
    """Warped functional inputs for ICA."""
    return [
        ctx.processing_dir / f"afni_mni_task-{dataset}_run-{r}.nii.gz"
        for r in ctx.runs_for_task(dataset)
    ]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for dataset in _ica_tasks(ctx):
            for path in [
                _melodic_ic(ctx, dataset),
                _ffs_ic(ctx, dataset),
                _melodic_mask(ctx, dataset),
            ]:
                if not path.exists():
                    missing.append(str(path))
    else:
        # Need warped functionals as input
        for dataset in _ica_tasks(ctx):
            for inp in _mni_inputs(ctx, dataset):
                if not inp.exists():
                    missing.append(str(inp))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run FSL melodic for both datasets."""
    ctx.melodic_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        out_dir = _melodic_dir(ctx, dataset)
        if _melodic_ic(ctx, dataset).exists() and not ctx.force_ref:
            continue
        inputs = ",".join(str(p) for p in _mni_inputs(ctx, dataset))
        elapsed, _ = run_timed(
            f"melodic -i {inputs} "
            f"-o {out_dir} --bgthreshold=3 --tr=1.7500 "
            f"--report --guireport={out_dir}/report.html "
            f"-d 0 --mmthresh=0.5 --Oall --Ostats -v",
            label=f"melodic {dataset}",
            cwd=ctx.melodic_ica_dir,
        )
        total += elapsed
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_ica -temp_concat for both datasets."""
    ctx.ffs_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        if _ffs_ic(ctx, dataset).exists() and not ctx.force_ffs:
            continue
        inputs = " ".join(str(p) for p in _mni_inputs(ctx, dataset))
        mask = _melodic_mask(ctx, dataset)
        mask_arg = f"-mask {mask}" if mask.exists() else ""
        elapsed, _ = run_timed(
            f"ffs_ica -input {inputs} "
            f"{mask_arg} "
            f"-temp_concat -ordering stdev "
            f"-migp "
            f"-prefix {ctx.ffs_ica_dir / f'all_{dataset}'} -verbose",
            label=f"ffs_ica {dataset}",
            cwd=ctx.ffs_ica_dir,
        )
        total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare ICA components between melodic and ffs_ica."""
    results = {}

    for dataset in _ica_tasks(ctx):
        melodic_path = _melodic_ic(ctx, dataset)
        ffs_path = _ffs_ic(ctx, dataset)
        mask_path = _melodic_mask(ctx, dataset)

        ica_result = compare_ica_components(melodic_path, ffs_path, mask_path)
        results[dataset] = ica_result

    # Aggregate
    mean_rs = [r["mean_matched_r"] for r in results.values()]
    overall_mean = sum(mean_rs) / len(mean_rs)

    cov_05 = [r["coverage_0.5"] for r in results.values()]
    overall_cov = sum(cov_05) / len(cov_05)

    passed = (
        overall_mean >= THRESHOLDS["mean_matched_r"] and overall_cov >= THRESHOLDS["coverage_0.5"]
    )

    comp_counts = {ds: (r["n_components_a"], r["n_components_b"]) for ds, r in results.items()}

    return {
        "passed": passed,
        "summary": (
            f"mean |r|={overall_mean:.4f}, coverage@0.5={overall_cov:.2f}, "
            + ", ".join(f"{ds}: melodic={na} ffs={nb}" for ds, (na, nb) in comp_counts.items())
        ),
        "per_dataset": results,
    }
