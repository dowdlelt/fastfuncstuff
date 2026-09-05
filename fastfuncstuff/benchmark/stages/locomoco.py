"""Residual non-rigid motion correction: ffs_locomoco, with no reference tool.

Nothing in AFNI/FSL measures what locomoco measures -- the per-frame shift along
the phase-encode axis that rigid moco leaves behind -- so this stage cannot be
validated against a reference the way every other stage is. It is here to get
the tool onto the timing/profiling path with a real dataset behind it, and to
keep its outputs on disk so a reference (or a synthetic ground truth) can be
scored against them later.

So the stage supplies its own truth instead. It runs twice:

* on the real run, for timing and for the usability checks -- the corrected
  series on the input grid and finite, a flow neither identically zero nor
  divergent, and the in-mask tSNR change as a diagnostic. No accuracy claim.
* on a SYNTHETIC case, where a known respiration-like PE field is imposed on a
  cropped piece of that same real run. There the recovery IS scored, and those
  are the thresholds the stage passes or fails on.

The synthetic forward model uses a Catmull-Rom cubic while locomoco resamples
with a Lanczos-3 windowed sinc, deliberately: distorting and undistorting with
the same kernel is an inverse crime that hides the interpolator's own error.
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

# Synthetic-recovery thresholds, calibrated on ds005165 localizer run-1, where
# the measured values are r=-0.907, RMS err=0.099 vox and residual=0.517. Set
# with ~1.5-2x headroom so they catch a regression without tracking run-to-run
# noise. Per [[Benchmark validation]]: if this stage fails, investigate before
# loosening these.
THRESHOLDS = {
    "min_flow_r": 0.80,  # |corr| recovered vs imposed field, in mask (meas. 0.907)
    "max_rms_err_vox": 0.20,  # RMS recovery error (meas. 0.099)
    "max_residual_ratio": 0.70,  # imposed distortion surviving correction (meas. 0.517)
}

# The cropped synthetic case: enough frames for the estimator to work with,
# small enough that validating costs a fraction of the real run.
SYNTH_FRAMES = 96
SYNTH_SLICES = 24
SYNTH_AMPLITUDE_VOX = 0.8

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


def _synth_stem(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return _out_dir(ctx) / f"{ctx.ffs_prefix}_{ctx.bids_prefix(task, run)}_synth"


def _synth_input_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_synth_stem(ctx, task, run)}_distorted.nii.gz")


def _synth_truth_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_synth_stem(ctx, task, run)}_truth.nii.gz")


def _synth_clean_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    """The undistorted crop the distorted series was made from."""
    return Path(f"{_synth_stem(ctx, task, run)}_clean.nii.gz")


def _synth_corrected_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_synth_stem(ctx, task, run)}_locomoco.nii.gz")


def _synth_flow_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return Path(f"{_synth_stem(ctx, task, run)}_flow.nii.gz")


def _build_synthetic_case(ctx: BenchmarkContext, task: str, run: int) -> None:
    """Crop the real run, impose a known PE field, and write clean/distorted/truth.

    Cropped rather than synthesised from scratch so the estimator still faces
    real anatomy, real noise and a real tSNR -- only the displacement is ours.
    """
    import nibabel as nib
    import torch

    from fastfuncstuff.simulation.distortion import apply_pe_shift, synthetic_pe_field

    src = _input_path(ctx, task, run)
    img = nib.load(str(src))
    # nib.load()'s stub returns the loose FileBasedImage base; the moco output is
    # a real NIfTI (same narrowing as validation.py:_load_vol).
    assert isinstance(img, (nib.Nifti1Image, nib.Nifti2Image))
    n_slices, n_frames = img.shape[2], img.shape[3]
    z0 = max(0, (n_slices - SYNTH_SLICES) // 2)  # central slices: most brain
    z1 = min(n_slices, z0 + SYNTH_SLICES)
    t1 = min(n_frames, SYNTH_FRAMES)

    clean = torch.from_numpy(np.asarray(img.dataobj[:, :, z0:z1, :t1], dtype=np.float32))
    pe_axis = "xyz".index(_pe_axis(ctx, task, run))
    tr = float(img.header.get_zooms()[3]) or 1.5
    field = synthetic_pe_field(
        tuple(clean.shape[:3]), clean.shape[3], amplitude=SYNTH_AMPLITUDE_VOX, tr=tr
    )
    distorted = apply_pe_shift(clean, field, pe_axis)

    affine = img.affine
    out_dir = _out_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, data in (
        (_synth_clean_path(ctx, task, run), clean),
        (_synth_input_path(ctx, task, run), distorted),
        (_synth_truth_path(ctx, task, run), field),
    ):
        out = nib.Nifti1Image(data.numpy(), affine)
        out.header.set_zooms(img.header.get_zooms()[:4])
        out.to_filename(str(path))


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
        _synth_clean_path(ctx, task, run),
        _synth_flow_path(ctx, task, run),
    ]


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_locomoco once on the configured run."""
    params = _params(ctx)
    task, run = _task_run(ctx)
    out = _corrected_path(ctx, task, run)
    done = out.exists() and _synth_flow_path(ctx, task, run).exists()
    ctx.note_items("ffs", 0 if (done and not ctx.force_ffs) else 1, 1)
    if done and not ctx.force_ffs:
        print("  Skipping ffs_locomoco (output exists, use -force-ffs to re-run)")
        return 0.0

    out_dir = _out_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    pe = _pe_axis(ctx, task, run)
    flags = (
        f"-pe_dir {pe} -backend {params['backend']} -refine {int(params['refine'])}"
        f"{ctx.ffs_device_flag()}"
    )
    elapsed, _ = run_timed(
        f"ffs_locomoco "
        f"-input {_input_path(ctx, task, run)} "
        f"-prefix {_stem(ctx, task, run)} " + flags,
        label=f"ffs_locomoco {task} run-{run}",
        cwd=out_dir,
        verbose=ctx.verbose,
    )

    # The scored half: a cropped piece of the same run with a field we chose.
    # Its time is deliberately NOT added to the stage's reported timing -- the
    # benchmark's speed number should mean "correcting one real run", not
    # "correcting one real run plus a validation fixture".
    _build_synthetic_case(ctx, task, run)
    run_timed(
        f"ffs_locomoco "
        f"-input {_synth_input_path(ctx, task, run)} "
        f"-prefix {_synth_stem(ctx, task, run)} "
        f"-no_movie -no_warp " + flags,
        label=f"ffs_locomoco synthetic ({SYNTH_FRAMES}f x {SYNTH_SLICES}sl)",
        cwd=out_dir,
        verbose=ctx.verbose,
    )
    return elapsed


def _score_synthetic(ctx: BenchmarkContext, task: str, run: int) -> dict:
    """Score the recovered field against the one we imposed.

    Three numbers, because each fails differently:

    * ``flow_r`` -- does the recovered field have the right SHAPE in space and
      time? Correlation is blind to a scale error, which is why it is not alone.
    * ``rms_err_vox`` -- is it the right SIZE? Catches a systematically
      under- or over-estimated amplitude that still correlates perfectly.
    * ``residual_ratio`` -- did correcting actually help? RMS(corrected - clean)
      over RMS(distorted - clean), so <1 means the imposed distortion shrank.
      This is the end-to-end question and is sign-convention-free, which the
      first two are not.
    """
    paths = {
        "clean": _synth_clean_path(ctx, task, run),
        "distorted": _synth_input_path(ctx, task, run),
        "truth": _synth_truth_path(ctx, task, run),
        "corrected": _synth_corrected_path(ctx, task, run),
        "flow": _synth_flow_path(ctx, task, run),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"error": f"missing synthetic output(s): {', '.join(missing)}"}

    clean, _ = _load_vol(paths["clean"])
    distorted, _ = _load_vol(paths["distorted"])
    truth, _ = _load_vol(paths["truth"])
    corrected, _ = _load_vol(paths["corrected"])
    flow, _ = _load_vol(paths["flow"])

    if tuple(flow.shape) != tuple(truth.shape):
        return {"error": f"flow {tuple(flow.shape)} != truth {tuple(truth.shape)}"}

    mask = _automask(clean.mean(dim=-1))
    sel = mask.numpy()
    truth_v = truth.numpy()[sel].ravel()
    flow_v = flow.numpy()[sel].ravel()

    # locomoco reports the PULL that corrects the distortion, i.e. the negative
    # of the pull that created it, so the correlation is expected NEGATIVE. The
    # sign is reported rather than assumed: a flip is a real regression, and
    # scoring |r| alone would hide it.
    r = float(np.corrcoef(truth_v, flow_v)[0, 1])
    rms_err = float(np.sqrt(np.mean((flow_v + truth_v) ** 2)))

    def _rms_gap(a, b) -> float:
        diff = (a - b).numpy()[sel]
        return float(np.sqrt(np.mean(diff**2)))

    before = _rms_gap(distorted, clean)
    after = _rms_gap(corrected, clean)
    ratio = after / before if before > 0 else float("nan")

    return {
        "flow_r": r,
        "rms_err_vox": rms_err,
        "residual_ratio": ratio,
        "rms_before": before,
        "rms_after": after,
        "n_mask_voxels": int(sel.sum()),
    }


def validate(ctx: BenchmarkContext) -> dict:
    """Score the synthetic recovery; sanity-check the real run's output."""
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

    # The scored half: recovery of a field we imposed ourselves.
    synth = _score_synthetic(ctx, task, run)
    if "error" in synth:
        problems.append(str(synth["error"]))
        synth_summary = "synthetic: unavailable"
    else:
        # Expected strongly NEGATIVE: locomoco reports the pull that CORRECTS
        # the distortion, the negative of the pull that created it. Scoring |r|
        # would hide a sign flip, which is a real regression.
        if synth["flow_r"] > -THRESHOLDS["min_flow_r"]:
            problems.append(
                f"flow r={synth['flow_r']:+.3f} (want <= -{THRESHOLDS['min_flow_r']:.2f})"
            )
        if synth["rms_err_vox"] > THRESHOLDS["max_rms_err_vox"]:
            problems.append(
                f"RMS err {synth['rms_err_vox']:.3f} vox "
                f"(want <= {THRESHOLDS['max_rms_err_vox']:.2f})"
            )
        if not synth["residual_ratio"] <= THRESHOLDS["max_residual_ratio"]:
            problems.append(
                f"residual ratio {synth['residual_ratio']:.3f} "
                f"(want <= {THRESHOLDS['max_residual_ratio']:.2f})"
            )
        synth_summary = (
            f"synthetic r={synth['flow_r']:+.3f}, RMS err={synth['rms_err_vox']:.3f} vox, "
            f"residual={synth['residual_ratio']:.3f}"
        )

    passed = passed and not problems
    detail = f" [{'; '.join(problems)}]" if problems else ""
    return {
        "passed": passed,
        "summary": (
            f"{synth_summary} | real run: RMS flow={rms_flow:.3f} vox, "
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
        "reference": "synthetic ground truth",
        **{f"synth_{k}": v for k, v in synth.items()},
    }
