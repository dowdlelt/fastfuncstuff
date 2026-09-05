"""Residual non-rigid motion correction: ffs_locomoco, with no reference tool.

Nothing in AFNI/FSL measures what locomoco measures -- the per-frame shift along
the phase-encode axis that rigid moco leaves behind -- so this stage cannot be
validated against a reference the way every other stage is. It is here to get
the tool onto the timing/profiling path with a real dataset behind it, and to
keep its outputs on disk so a reference (or a synthetic ground truth) can be
scored against them later.

``validate()`` therefore checks only that the run produced usable output: the
corrected series is on the input grid and finite, the estimated flow is non-zero
and bounded, and the in-mask tSNR change is reported as a diagnostic. None of it
is an accuracy claim, and the summary says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed
from ..validation import _automask, _load_vol

name = "locomoco"
description = "Residual EPI distortion (ffs_locomoco; no reference yet)"
requires = ["moco"]

# One run only. locomoco is the most expensive per-run tool in the suite and the
# stage buys timing, not statistics -- a second run would double the cost and
# tell us nothing a single one doesn't.
DEFAULTS = {
    "task": "localizer",
    "run": 1,
    "backend": "flow",
    "refine": 1,
}

# The flow is measured in voxels along the PE axis. Residual distortion after
# rigid moco is a fraction of a voxel to a couple of voxels; anything past this
# is the estimator having diverged, not a head that moved.
MAX_PLAUSIBLE_FLOW_VOX = 10.0

_PE_AXIS = {"i": "x", "j": "y", "k": "z"}


def _params(ctx: BenchmarkContext) -> dict:
    params = dict(DEFAULTS)
    params.update(ctx.get_stage_params(name))
    return params


def _task_run(ctx: BenchmarkContext) -> tuple[str, int]:
    """The (task, run) this stage corrects, clamped to what the config has."""
    params = _params(ctx)
    task = str(params["task"])
    if task not in ctx.tasks:
        task = ctx.task_names()[0]
    runs = ctx.runs_for_task(task)
    run = int(params["run"])
    if run not in runs:
        run = runs[0]
    return task, run


def _pe_axis(ctx: BenchmarkContext, task: str, run: int) -> str:
    """PE axis letter from the BIDS sidecar's ``PhaseEncodingDirection``.

    The sign carries no information here -- locomoco estimates a signed shift
    along the axis either way -- so only the axis letter is passed through.
    """
    params = _params(ctx)
    if params.get("pe_dir"):
        return str(params["pe_dir"])
    sidecar = ctx.func_dir / f"{ctx.bids_prefix(task, run)}_bold.json"
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return "y"
    return _PE_AXIS.get(str(meta.get("PhaseEncodingDirection", "j"))[:1], "y")


def _out_dir(ctx: BenchmarkContext) -> Path:
    """Own directory: a default run writes a warp file bigger than the input."""
    return ctx.data_dir / f"ffs_locomoco{ctx.ffs_tag}"


def _input_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    """FFS's own moco output -- locomoco corrects what rigid moco left behind."""
    return ctx.processing_dir / f"{ctx.ffs_prefix}_moco_{ctx.bids_prefix(task, run)}_bold.nii"


def _stem(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return _out_dir(ctx) / f"{ctx.ffs_prefix}_{ctx.bids_prefix(task, run)}_bold"


def _corrected_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_stem(ctx, task, run)}_locomoco.nii.gz")


def _flow_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_stem(ctx, task, run)}_flow.nii.gz")


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    task, run = _task_run(ctx)
    missing = []
    if ctx.validate_only:
        for path in (_corrected_path(ctx, task, run), _flow_path(ctx, task, run)):
            if not path.exists():
                missing.append(str(path))
    else:
        src = _input_path(ctx, task, run)
        if not src.exists():
            missing.append(str(src))
    return missing


def validation_inputs(ctx: BenchmarkContext) -> list[Path]:
    task, run = _task_run(ctx)
    return [
        _input_path(ctx, task, run),
        _corrected_path(ctx, task, run),
        _flow_path(ctx, task, run),
    ]


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_locomoco once on the configured run."""
    params = _params(ctx)
    task, run = _task_run(ctx)
    out = _corrected_path(ctx, task, run)
    ctx.note_items("ffs", 0 if (out.exists() and not ctx.force_ffs) else 1, 1)
    if out.exists() and not ctx.force_ffs:
        print("  Skipping ffs_locomoco (output exists, use -force-ffs to re-run)")
        return 0.0

    out_dir = _out_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed, _ = run_timed(
        f"ffs_locomoco "
        f"-input {_input_path(ctx, task, run)} "
        f"-prefix {_stem(ctx, task, run)} "
        f"-pe_dir {_pe_axis(ctx, task, run)} "
        f"-backend {params['backend']} "
        f"-refine {int(params['refine'])}"
        f"{ctx.ffs_device_flag()}",
        label=f"ffs_locomoco {task} run-{run}",
        cwd=out_dir,
        verbose=ctx.verbose,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Sanity-check the outputs. There is no reference to compare against."""
    task, run = _task_run(ctx)
    src_path = _input_path(ctx, task, run)
    corr_path = _corrected_path(ctx, task, run)
    flow_path = _flow_path(ctx, task, run)

    for path in (src_path, corr_path, flow_path):
        if not path.exists():
            return {"passed": False, "summary": f"Missing: {path.name}"}

    src, _ = _load_vol(src_path)
    corr, _ = _load_vol(corr_path)
    flow, _ = _load_vol(flow_path)

    if tuple(corr.shape) != tuple(src.shape):
        return {
            "passed": False,
            "summary": f"Grid mismatch: input={tuple(src.shape)} corrected={tuple(corr.shape)}",
        }
    if tuple(flow.shape) != tuple(src.shape):
        return {
            "passed": False,
            "summary": f"Grid mismatch: input={tuple(src.shape)} flow={tuple(flow.shape)}",
        }

    finite = bool(np.isfinite(corr.numpy()).all() and np.isfinite(flow.numpy()).all())
    mask = _automask(src.mean(dim=-1))
    flow_in_mask = flow.numpy()[mask.numpy()]
    max_flow = float(np.abs(flow_in_mask).max())
    rms_flow = float(np.sqrt((flow_in_mask.astype(np.float64) ** 2).mean()))

    def _tsnr(series) -> float:
        arr = series.numpy()
        std = arr.std(axis=-1)
        mean = arr.mean(axis=-1)
        keep = mask.numpy() & (std > 0)
        return float(np.median(mean[keep] / std[keep])) if keep.any() else float("nan")

    tsnr_before, tsnr_after = _tsnr(src), _tsnr(corr)

    # A zero field means the estimator never moved anything -- the one outcome
    # that is definitely wrong, and the only accuracy-free failure available.
    moved = rms_flow > 0.0
    bounded = max_flow <= MAX_PLAUSIBLE_FLOW_VOX
    passed = finite and moved and bounded

    problems = []
    if not finite:
        problems.append("non-finite values in output")
    if not moved:
        problems.append("flow is identically zero")
    if not bounded:
        problems.append(f"max |flow| {max_flow:.2f} vox exceeds {MAX_PLAUSIBLE_FLOW_VOX}")

    detail = f" [{'; '.join(problems)}]" if problems else ""
    return {
        "passed": passed,
        "summary": (
            f"no reference — diagnostics only: RMS flow={rms_flow:.3f} vox, "
            f"max={max_flow:.2f} vox, median tSNR {tsnr_before:.2f}→{tsnr_after:.2f}"
            f"{detail}"
        ),
        "task": task,
        "run": run,
        "rms_flow_vox": rms_flow,
        "max_flow_vox": max_flow,
        "tsnr_before": tsnr_before,
        "tsnr_after": tsnr_after,
        "finite": finite,
        "reference": None,
    }
