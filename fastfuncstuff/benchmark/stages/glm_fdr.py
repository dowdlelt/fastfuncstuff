"""GLM FDR benchmark: 3dREMLfit (with FDR) vs ffs_reml -add_fdr.

Uses ``3dDeconvolve -x1D_stop`` to generate a separate ``.REML_cmd`` script
that outputs to a dedicated prefix, then runs ``3dREMLfit`` *without*
``-nofdr`` so AFNI computes FDRCURVE attributes.  The FFS path mirrors this
with ``ffs_reml -add_fdr``.

Validation compares AFNI FDRCURVE z(q) curves for every stat sub-brick.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_bucket_volumes

name = "glm_fdr"
description = "GLM FDR (3dREMLfit with FDR vs ffs_reml -add_fdr)"

THRESHOLDS = {
    "fdr_curve_max_mae": 0.05,
    "reml_beta_min_r": 0.95,
}

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]
_DEFAULT_GLTS = [
    ("faces_vs_objects", "+1*faces -1*objects"),
    ("faces_vs_scenes", "+1*faces -1*scenes"),
    ("faces_vs_scrambled", "+1*faces -1*scrambled"),
]


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm_fdr")


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


def _hrf_model(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("hrf_model", "SPMG1(3)")


def _fdr_prefix(ctx: BenchmarkContext) -> str:
    task = _primary_task(ctx)
    return f"stats_{task}_REML_FDR"


def _afni_reml_fdr_head(ctx: BenchmarkContext) -> Path:
    return ctx.afni_glm_dir / f"{_fdr_prefix(ctx)}+tlrc.HEAD"


def _afni_reml_cmd(ctx: BenchmarkContext) -> Path:
    return ctx.afni_glm_dir / f"{_fdr_prefix(ctx)}.REML_cmd"


def _ffs_reml_fdr(ctx: BenchmarkContext) -> Path:
    return ctx.ffs_glm_dir / f"{_fdr_prefix(ctx)}.nii.gz"


# ---------------------------------------------------------------------------
# AFNI HEAD FDRCURVE parser
# ---------------------------------------------------------------------------


def _read_afni_head_fdrcurves(head_path: Path) -> dict[int, dict]:
    """Parse FDRCURVE_NNNNNN float-attrs from an AFNI .HEAD file.

    AFNI float-attributes are stored as unquoted whitespace-separated floats
    spanning multiple lines, terminated by a blank line — *not* the `'...'~`
    form used for string-attributes.
    """
    text = head_path.read_text(errors="replace")
    curves: dict[int, dict] = {}

    pattern = re.compile(
        r"type\s*=\s*float-attribute\s*\n"
        r"name\s*=\s*FDRCURVE_(\d+)\s*\n"
        r"count\s*=\s*(\d+)\s*\n"
        r"((?:[^\n]*\n)+?)"  # one-or-more value lines (non-greedy)
        r"(?:\n|\Z)",  # terminated by blank line or EOF
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        brick_idx = int(m.group(1))
        count = int(m.group(2))
        body = m.group(3)
        try:
            values = np.fromstring(body, sep=" ", dtype=np.float64)
        except ValueError:
            continue
        if values.size < count:
            # Body got truncated by the blank-line terminator pattern; retry
            # by scanning until we have `count` floats.
            continue
        values = values[:count]
        if values.size < 3:
            continue
        curves[brick_idx] = {
            "x0": float(values[0]),
            "dx": float(values[1]),
            "z": values[2:].astype(np.float32),
        }

    return curves


def _read_nifti_fdrcurves(nifti_path: Path) -> dict[int, dict]:
    """Parse FDRCURVE_NNNNNN attrs from a NIfTI file's AFNI extension."""
    import nibabel as nib

    img = nib.load(str(nifti_path))
    extensions = getattr(img.header, "extensions", [])
    afni_ext = None
    for ext in extensions:
        if ext.get_code() == 4:
            afni_ext = ext.content.decode("utf-8", errors="replace")
            break
    if afni_ext is None:
        return {}

    curves: dict[int, dict] = {}
    for m in re.finditer(
        r'atr_name="FDRCURVE_(\d+)"[^>]*>\s*(.*?)\s*</AFNI_atr>',
        afni_ext,
        re.DOTALL,
    ):
        brick_idx = int(m.group(1))
        values = np.array([float(v) for v in m.group(2).split() if v])
        if values.size < 3:
            continue
        curves[brick_idx] = {
            "x0": float(values[0]),
            "dx": float(values[1]),
            "z": values[2:].astype(np.float32),
        }

    return curves


# ---------------------------------------------------------------------------
# FDR curve comparison
# ---------------------------------------------------------------------------


def _compare_fdr_curves(
    ref_curves: dict[int, dict],
    ffs_curves: dict[int, dict],
) -> dict:
    """Compare FDRCURVE z(q) arrays between AFNI and FFS."""
    common = sorted(set(ref_curves) & set(ffs_curves))
    if not common:
        return {"error": "no common FDRCURVE bricks", "n_compared": 0}

    per_brick = []
    for idx in common:
        ref = ref_curves[idx]
        ffs = ffs_curves[idx]
        # Curves are stored as (x0, dx, z[0..N-1]) — z evaluated at
        # x = x0 + i*dx in |stat| space. AFNI and FFS pick different x0/dx
        # (because they clip the top at ZTOP differently and the in-mask
        # voxel sets aren't identical), so element-wise z[i]-z[i] is
        # meaningless. Interpolate both onto a common |stat| grid that
        # spans the overlap of their ranges.
        z_ref = np.asarray(ref["z"], dtype=np.float64)
        z_ffs = np.asarray(ffs["z"], dtype=np.float64)
        x_ref = ref["x0"] + ref["dx"] * np.arange(len(z_ref))
        x_ffs = ffs["x0"] + ffs["dx"] * np.arange(len(z_ffs))
        lo = max(x_ref[0], x_ffs[0])
        hi = min(x_ref[-1], x_ffs[-1])
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 101)
        z_ref_i = np.interp(grid, x_ref, z_ref)
        z_ffs_i = np.interp(grid, x_ffs, z_ffs)
        diff = z_ref_i - z_ffs_i
        mae = float(np.mean(np.abs(diff)))
        max_ae = float(np.max(np.abs(diff)))
        corr = float(
            np.corrcoef(z_ref_i, z_ffs_i)[0, 1]
            if np.std(z_ref_i) > 0 and np.std(z_ffs_i) > 0
            else 0.0
        )
        per_brick.append(
            {
                "brick_idx": idx,
                "mae": mae,
                "max_ae": max_ae,
                "r": corr,
                "n_points": int(grid.size),
                "stat_lo": float(lo),
                "stat_hi": float(hi),
            }
        )

    if not per_brick:
        return {"error": "no valid curve comparisons", "n_compared": 0}

    maes = [b["mae"] for b in per_brick]
    rs = [b["r"] for b in per_brick]
    return {
        "n_compared": len(per_brick),
        "mean_mae": float(np.mean(maes)),
        "max_mae": float(np.max(maes)),
        "mean_r": float(np.mean(rs)),
        "min_r": float(np.min(rs)),
        "per_brick": per_brick,
    }


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []

    if ctx.validate_only:
        ref_head = _afni_reml_fdr_head(ctx)
        if not ref_head.exists():
            missing.append(str(ref_head))
        ffs_path = _ffs_reml_fdr(ctx)
        if not ffs_path.exists():
            missing.append(str(ffs_path))
    else:
        for run in _runs(ctx):
            src = _scaled_input(ctx, run)
            if not src.exists():
                missing.append(str(src))

    return missing


def _prepare_timing_files(ctx: BenchmarkContext) -> None:
    timing_dir = ctx.timing_dir
    if timing_dir.exists() and any(timing_dir.iterdir()):
        return
    timing_dir.mkdir(exist_ok=True)
    task = _primary_task(ctx)
    events = " ".join(
        str(ctx.func_dir / f"{ctx.bids_prefix(task, r)}_events.tsv") for r in _runs(ctx)
    )
    run_timed(
        f"timing_tool.py -write_multi_timing {timing_dir}/onsets.{task}. "
        f"-multi_timing_ncol_tsv {events}",
        label="timing_tool.py",
        cwd=ctx.processing_dir,
    )


def run_ref(ctx: BenchmarkContext) -> float:
    """3dDeconvolve -x1D_stop (generates REML_cmd) + 3dREMLfit without -nofdr."""
    afni = ctx.afni_glm_dir
    afni.mkdir(parents=True, exist_ok=True)

    _prepare_timing_files(ctx)

    task = _primary_task(ctx)
    stim_labels = _stim_labels(ctx)
    glts = _glts(ctx)
    hrf = _hrf_model(ctx)
    prefix = _fdr_prefix(ctx)
    reml_cmd = _afni_reml_cmd(ctx)
    total = 0.0

    # Step 1: 3dDeconvolve -x1D_stop to generate the REML_cmd script
    if not reml_cmd.exists() or ctx.force_ref:
        inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))
        stim_args = ""
        for i, label in enumerate(stim_labels, 1):
            stim_args += (
                f"-stim_times {i} {ctx.timing_dir}/onsets.{task}.times.{label}.txt "
                f"'{hrf}' -stim_label {i} {label} "
            )
        glt_args = ""
        for i, (label, sym) in enumerate(glts, 1):
            glt_args += f"-gltsym 'SYM: {sym}' -glt_label {i} {label} "

        elapsed, _ = run_timed(
            f"3dDeconvolve -overwrite "
            f"-input {inputs} "
            f"-polort A -num_stimts {len(stim_labels)} -float "
            f"{stim_args} "
            f"-jobs 10 -noFDR "
            f"{glt_args} "
            f"-tout -x1D_stop "
            f"-x1D {afni / prefix}.xmat.1D "
            f"-bucket {afni / prefix}",
            label="3dDeconvolve -x1D_stop",
            cwd=afni,
        )
        total += elapsed

    # Step 2: Run the REML_cmd script without -nofdr (FDR enabled)
    reml_head = _afni_reml_fdr_head(ctx)
    if not reml_head.exists() or ctx.force_ref:
        elapsed, _ = run_timed(
            f"tcsh {reml_cmd} -overwrite",
            label="3dREMLfit (with FDR)",
            cwd=afni,
        )
        total += elapsed

    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_reml REML with -add_fdr."""
    ffs = ctx.ffs_glm_dir
    ffs.mkdir(parents=True, exist_ok=True)

    inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))
    prefix = _fdr_prefix(ctx)
    xmat = ctx.afni_glm_dir / f"{prefix}.xmat.1D"

    out_path = _ffs_reml_fdr(ctx)
    out_var = ffs / f"{prefix}_var.nii.gz"

    if out_path.exists() and not ctx.force_ffs:
        return 0.0

    elapsed, _ = run_timed(
        f"ffs_reml -input {inputs} -matrix {xmat} "
        f"-Rbuck {out_path} -Rvar {out_var} -tout -add_fdr"
        f"{ctx.ffs_afni_mode_flag()}{ctx.ffs_device_flag()}",
        label="ffs_reml REML+FDR",
        cwd=ffs,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare FDR curves between AFNI and FFS."""
    ref_head = _afni_reml_fdr_head(ctx)
    ffs_path = _ffs_reml_fdr(ctx)

    ref_curves = _read_afni_head_fdrcurves(ref_head)
    ffs_curves = _read_nifti_fdrcurves(ffs_path)

    fdr_result = _compare_fdr_curves(ref_curves, ffs_curves)

    reml_result = compare_bucket_volumes(ref_head, ffs_path)

    fdr_passed = (
        "error" not in fdr_result
        and fdr_result.get("max_mae", 999) < THRESHOLDS["fdr_curve_max_mae"]
    )
    reml_passed = reml_result["min_r"] >= THRESHOLDS["reml_beta_min_r"]
    passed = fdr_passed and reml_passed

    if "error" in fdr_result:
        fdr_summary = fdr_result["error"]
    else:
        fdr_summary = (
            f"n_bricks={fdr_result['n_compared']}, "
            f"mean_mae={fdr_result['mean_mae']:.4f}, "
            f"max_mae={fdr_result['max_mae']:.4f}, "
            f"mean_r={fdr_result['mean_r']:.4f}"
        )

    return {
        "passed": passed,
        "summary": f"FDR: {fdr_summary}, REML min_r={reml_result['min_r']:.4f}",
        "fdr": fdr_result,
        "reml": reml_result,
    }
