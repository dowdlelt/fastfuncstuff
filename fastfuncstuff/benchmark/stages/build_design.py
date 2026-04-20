"""Build design matrix benchmark: 3dDeconvolve X.xmat.1D vs ffs_build_design.

Validates that ffs_build_design produces stimulus columns that match AFNI's
3dDeconvolve output to high precision.  The AFNI X.xmat.1D is produced by the
glm stage — this stage only runs ffs_build_design and compares.

No ref timing is recorded (the AFNI matrix already exists on disk).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "build_design"
description = "Design matrix (3dDeconvolve X.xmat.1D vs ffs_build_design)"

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]
_DEFAULT_HRF_MODEL = "SPMG1(3)"
_DEFAULT_GLTS = [
    ("faces_vs_objects", "SYM: +1*faces -1*objects"),
    ("faces_vs_scenes",  "SYM: +1*faces -1*scenes"),
    ("faces_vs_scrambled", "SYM: +1*faces -1*scrambled"),
]

THRESHOLD_MIN_R = 0.99  # applied per-run, per-label (within-run only)


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _stim_labels(ctx: BenchmarkContext) -> list[str]:
    return _glm_params(ctx).get("stim_labels", _DEFAULT_STIM_LABELS)


def _hrf_model(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("hrf_model", _DEFAULT_HRF_MODEL)


def _glts(ctx: BenchmarkContext) -> list[tuple[str, str]]:
    raw = _glm_params(ctx).get("glts", _DEFAULT_GLTS)
    return [(g[0], g[1]) for g in raw]


def _afni_xmat(ctx: BenchmarkContext) -> Path:
    return ctx.afni_glm_dir / "X.xmat.1D"


def _ffs_xmat(ctx: BenchmarkContext) -> Path:
    return ctx.ffs_glm_dir / "ffs_X.xmat.1D"


def _scaled_input(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"scaled_afni_mni_task-{task}_run-{run}.nii.gz"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    task = _primary_task(ctx)

    afni_xmat = _afni_xmat(ctx)
    if not afni_xmat.exists():
        missing.append(
            f"{afni_xmat}  (run the 'glm' stage first to produce X.xmat.1D)"
        )

    timing_dir = ctx.timing_dir
    for label in _stim_labels(ctx):
        t = timing_dir / f"onsets.{task}.times.{label}.txt"
        if not t.exists():
            missing.append(str(t))

    for run in _runs(ctx):
        s = _scaled_input(ctx, run)
        if not s.exists():
            missing.append(str(s))

    return missing


def _detect_polort(afni_xmat_path: Path) -> int:
    """Infer polort from the AFNI xmat's baseline column count."""
    from fastfuncstuff.io.afni import read_afni_design_matrix

    info = read_afni_design_matrix(afni_xmat_path)
    col_groups = info.get("column_groups") or []
    # AFNI convention: group <= 0 → baseline/polynomial columns
    n_baseline = sum(1 for g in col_groups if g <= 0)
    n_runs = len(info.get("run_starts") or [0])
    if n_runs == 0:
        return 3  # safe default
    # polort + 1 baseline columns per run
    return max(0, (n_baseline // n_runs) - 1)


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_build_design with the same stims/GLTs as the reference 3dDeconvolve."""
    ctx.ffs_glm_dir.mkdir(parents=True, exist_ok=True)

    out_xmat = _ffs_xmat(ctx)
    if out_xmat.exists() and not ctx.force_ffs:
        print("  Skipping ffs_build_design (output exists, use -force-ffs to re-run)")
        return 0.0

    polort = _detect_polort(_afni_xmat(ctx))

    task = _primary_task(ctx)
    inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))

    stim_args = " ".join(
        f"-stim {ctx.timing_dir}/onsets.{task}.times.{label}.txt '{_hrf_model(ctx)}' {label}"
        for label in _stim_labels(ctx)
    )

    glt_args = " ".join(
        f"-gltsym '{sym}' {label}"
        for label, sym in _glts(ctx)
    )

    elapsed, _ = run_timed(
        f"ffs_build_design "
        f"-input {inputs} "
        f"-polort {polort} "
        f"{stim_args} "
        f"{glt_args} "
        f"-xmat {out_xmat}",
        label="ffs_build_design",
        cwd=ctx.ffs_glm_dir,
        verbose=ctx.verbose,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare stimulus columns between AFNI and ffs_build_design xmats."""
    from fastfuncstuff.io.afni import read_afni_design_matrix

    afni_xmat = _afni_xmat(ctx)
    ffs_xmat = _ffs_xmat(ctx)

    if not ffs_xmat.exists():
        return {"passed": False, "summary": "ffs_X.xmat.1D not found"}

    afni = read_afni_design_matrix(afni_xmat)
    ffs  = read_afni_design_matrix(ffs_xmat)

    afni_mat = afni["matrix"]  # (n_tp, n_cols)
    ffs_mat  = ffs["matrix"]

    # Sanity: same shape
    if afni_mat.shape[0] != ffs_mat.shape[0]:
        return {
            "passed": False,
            "summary": (
                f"Timepoint mismatch: AFNI={afni_mat.shape[0]} FFS={ffs_mat.shape[0]}"
            ),
        }

    # Extract stimulus columns by label from each xmat
    def _stim_cols(info: dict, mat: np.ndarray) -> dict[str, np.ndarray]:
        cols = {}
        labels  = info.get("stim_labels") or []
        bots    = info.get("stim_bots")   or []
        tops    = info.get("stim_tops")   or []
        for label, bot, top in zip(labels, bots, tops, strict=False):
            cols[label] = mat[:, bot : top + 1]
        return cols

    afni_stims = _stim_cols(afni, afni_mat)
    ffs_stims  = _stim_cols(ffs,  ffs_mat)

    common = sorted(set(afni_stims) & set(ffs_stims))
    if not common:
        return {"passed": False, "summary": "No matching stimulus labels found"}

    # Build run slice boundaries from AFNI xmat run_starts
    run_starts: list[int] = afni.get("run_starts") or [0]
    n_tp = afni_mat.shape[0]
    run_ends = list(run_starts[1:]) + [n_tp]

    # Correlate within each run so block-diagonal zeros don't inflate r
    rs: list[float] = []
    per_label: dict[str, dict] = {}
    for label in common:
        a_col = afni_stims[label]  # (n_tp, n_basis)
        f_col = ffs_stims[label]
        run_rs = []
        for run_idx, (rstart, rend) in enumerate(zip(run_starts, run_ends, strict=False)):
            a_seg = a_col[rstart:rend].ravel()
            f_seg = f_col[rstart:rend].ravel()
            # Skip runs where both sides are essentially flat (no events in this run)
            if a_seg.std() < 1e-10 and f_seg.std() < 1e-10:
                continue
            if a_seg.std() < 1e-10 or f_seg.std() < 1e-10:
                r = 0.0
            else:
                r = float(np.corrcoef(a_seg, f_seg)[0, 1])
            run_rs.append(r)
            rs.append(r)
        per_label[label] = {
            f"run{run_idx + 1}": round(r, 6)
            for run_idx, r in enumerate(run_rs)
        }

    if not rs:
        return {"passed": False, "summary": "No within-run variance found in any stim column"}

    min_r  = float(np.min(rs))
    mean_r = float(np.mean(rs))
    passed = min_r >= THRESHOLD_MIN_R

    n_comparisons = len(rs)
    return {
        "passed": passed,
        "summary": (
            f"within-run stim min_r={min_r:.4f} mean_r={mean_r:.4f} "
            f"({len(common)} labels × {n_comparisons // len(common)} runs = {n_comparisons} comparisons)"
        ),
        "per_label": per_label,
        "min_r": min_r,
        "mean_r": mean_r,
    }
