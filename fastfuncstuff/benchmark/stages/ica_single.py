"""ICA single-run benchmark: melodic vs ffs_ica on individual runs.

Runs melodic and ffs_ica independently on each run of rest and localizer
(5 runs x 2 tasks = 10 comparisons, plus 2 tasks x component count check).
This complements the temporal-concatenation ICA stage by testing per-run
decomposition and giving more data points for component count agreement.
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_ica_components

name = "ica_single"
description = "ICA single-run (melodic vs ffs_ica per run)"

THRESHOLDS = {
    "mean_matched_r": 0.60,  # per-run ICA is noisier than concat
    "coverage_0.5": 0.60,
}


def _ica_tasks(ctx: BenchmarkContext) -> list[str]:
    """Tasks to run ICA on (from config or default to all tasks)."""
    params = ctx.get_stage_params("ica")
    return params.get("tasks", ctx.task_names())


def _melodic_dir(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return ctx.melodic_ica_dir / f"{dataset}_run{run:02d}_melodic.ica"


def _melodic_ic(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return _melodic_dir(ctx, dataset, run) / "melodic_IC.nii.gz"


def _melodic_mask(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return _melodic_dir(ctx, dataset, run) / "mask.nii.gz"


def _ffs_prefix(ctx: BenchmarkContext, dataset: str) -> str:
    return str(ctx.ffs_ica_dir / f"{dataset}_single")


def _ffs_ic(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    """FFS GMM-z'd component maps (apples-to-apples vs `melodic_IC`).

    Lives inside the per-run `.ica/ffs_outputs/` subfolder after the
    output-layout refactor.
    """
    base = f"{dataset}_single"
    return (
        ctx.ffs_ica_dir
        / f"{base}_run{run:02d}.ica"
        / "ffs_outputs"
        / f"{base}_run{run:02d}_ica_zmaps.nii.gz"
    )


def _mni_input(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mni_task-{dataset}_run-{run}.nii.gz"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for dataset in _ica_tasks(ctx):
            for run in ctx.runs_for_task(dataset):
                for path in [_melodic_ic(ctx, dataset, run), _ffs_ic(ctx, dataset, run)]:
                    if not path.exists():
                        missing.append(str(path))
    else:
        for dataset in _ica_tasks(ctx):
            for run in ctx.runs_for_task(dataset):
                inp = _mni_input(ctx, dataset, run)
                if not inp.exists():
                    missing.append(str(inp))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run FSL melodic on each run independently."""
    ctx.melodic_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        for run in ctx.runs_for_task(dataset):
            out_dir = _melodic_dir(ctx, dataset, run)
            if _melodic_ic(ctx, dataset, run).exists() and not ctx.force_ref:
                continue
            inp = _mni_input(ctx, dataset, run)
            elapsed, _ = run_timed(
                f"melodic -i {inp} "
                f"-o {out_dir} --bgthreshold=3 --tr=1.7500 "
                f"--report --guireport={out_dir}/report.html "
                f"-d 0 --mmthresh=0.5 --Oall --Ostats -v",
                label=f"melodic {dataset} run-{run}",
                cwd=ctx.melodic_ica_dir,
            )
            total += elapsed
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_ica on each run independently (run-wise default mode)."""
    ctx.ffs_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        # Check if all runs already exist
        runs = ctx.runs_for_task(dataset)
        all_exist = all(_ffs_ic(ctx, dataset, run).exists() for run in runs)
        if all_exist and not ctx.force_ffs:
            continue

        # ffs_ica in default run-wise mode processes all inputs independently
        # but outputs with run01..runNN tags
        inputs = " ".join(str(_mni_input(ctx, dataset, r)) for r in runs)
        prefix = _ffs_prefix(ctx, dataset)
        elapsed, _ = run_timed(
            f"ffs_ica -input {inputs} -ordering stdev "
            f"-prefix {prefix} -verbose"
            f"{ctx.ffs_device_flag()}",
            label=f"ffs_ica single {dataset}",
            cwd=ctx.ffs_ica_dir,
        )
        total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare ICA components per run between melodic and ffs_ica."""
    per_run_results = []
    comp_counts_melodic = []
    comp_counts_ffs = []

    for dataset in _ica_tasks(ctx):
        for run in ctx.runs_for_task(dataset):
            melodic_path = _melodic_ic(ctx, dataset, run)
            ffs_path = _ffs_ic(ctx, dataset, run)
            mask_path = _melodic_mask(ctx, dataset, run)

            if not melodic_path.exists() or not ffs_path.exists():
                per_run_results.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "error": "missing output",
                    }
                )
                continue

            mask_arg = mask_path if mask_path.exists() else None
            ica_result = compare_ica_components(melodic_path, ffs_path, mask_arg)
            ica_result["dataset"] = dataset
            ica_result["run"] = run
            per_run_results.append(ica_result)
            comp_counts_melodic.append(ica_result["n_components_a"])
            comp_counts_ffs.append(ica_result["n_components_b"])

    # Aggregate across all valid runs
    valid = [r for r in per_run_results if "error" not in r]
    if not valid:
        return {
            "passed": False,
            "summary": "No valid single-run ICA comparisons",
            "per_run": per_run_results,
        }

    mean_rs = [r["mean_matched_r"] for r in valid]
    overall_mean_r = sum(mean_rs) / len(mean_rs)

    cov_05 = [r["coverage_0.5"] for r in valid]
    overall_cov = sum(cov_05) / len(cov_05)

    passed = (
        overall_mean_r >= THRESHOLDS["mean_matched_r"] and overall_cov >= THRESHOLDS["coverage_0.5"]
    )

    # Component count summary
    n_comps_str = (
        (
            f"melodic={min(comp_counts_melodic)}-{max(comp_counts_melodic)}, "
            f"ffs={min(comp_counts_ffs)}-{max(comp_counts_ffs)}"
        )
        if comp_counts_melodic
        else "n/a"
    )

    return {
        "passed": passed,
        "summary": (
            f"mean |r|={overall_mean_r:.4f}, coverage@0.5={overall_cov:.2f}, "
            f"n_runs={len(valid)}, comps: {n_comps_str}"
        ),
        "overall_mean_r": overall_mean_r,
        "overall_coverage_0.5": overall_cov,
        "n_valid_runs": len(valid),
        "comp_counts_melodic": comp_counts_melodic,
        "comp_counts_ffs": comp_counts_ffs,
        "per_run": per_run_results,
    }
