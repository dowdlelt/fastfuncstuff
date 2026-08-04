"""Phase regression benchmark: phaseprep (scipy.odr) vs ffs_phasereg.

Reference: run_phaseprep_reference.py — a faithful port of phaseprep's
PhaseFitOdr math (linear detrend → FFT noise → scipy.odr per voxel) with
multiprocessing parallelism.

FFS: ffs_phasereg with -polort 1 (Legendre degree-1 = same subspace as
phaseprep's polyfit(1)), -task_removal none, -phase_filter none, default
Deming regression.

Data: ds003427 sub-03 checkerboard GE (one run). The prep step (download,
moco, ROMEO unwrap, ffs_nwarp, 3dAutomask) runs automatically and is NOT
timed — only the phase regression itself (ref + FFS) is timed.

Prep tools required: aws-cli, 3dvolreg, 3dAutomask, ROMEO, ffs_nwarp.
Set ROMEO_PATH env var if ROMEO is not on PATH.

Validation:
  - R² map: Pearson r over automask (primary quality metric)
  - Corrected magnitude: per-voxel timeseries correlation (micro signal)
  - Macro component: per-voxel timeseries correlation (macro signal)
  - Slope map: Pearson r and median |Δ| over automask (informational only —
    scipy.odr vs closed-form pick different quadratic roots in ~80k
    ill-conditioned voxels, but ODR shrinkage makes this irrelevant for
    the corrected output)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "phasereg"
description = "Phase regression (phaseprep scipy.odr vs ffs_phasereg)"

DATASET_DIR = "ds003427-download"
BOLD_FILE = "sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz"
PHASE_FILE = "sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_phase.nii.gz"
REF_SCRIPT = "run_phaseprep_reference.py"
REF_PREFIX = "phaseprep_ref"
FFS_PREFIX = "benchmark_ffs_pr"

MAG_FILE = "method_direct_aligned.nii"
PHA_FILE = "method_direct_aligned_phase.nii"
MASK_FILE = "automask.nii.gz"

THRESHOLDS = {
    "r2_r": 0.90,
    "corrected_ts_median_r": 0.99,
    "macro_ts_median_r": 0.90,
}


def _data_dir(ctx: BenchmarkContext) -> Path:
    from ..._paths import get_benchmark_data_dir

    return get_benchmark_data_dir() / DATASET_DIR


def _proc_dir(ctx: BenchmarkContext) -> Path:
    return _data_dir(ctx) / "processing"


def _ref_script_source() -> Path:
    return Path(__file__).resolve().parents[3] / "test_data" / "phasereg_refs" / REF_SCRIPT


def _find_romeo() -> str:
    which = shutil.which("romeo")
    if which:
        return which
    env = os.environ.get("ROMEO_PATH")
    if env and Path(env).exists():
        return env
    raise FileNotFoundError(
        "ROMEO not found. Add it to PATH or set ROMEO_PATH=/path/to/romeo.\n"
        "Download from https://github.com/korbinian90/ROMEO/releases"
    )


def _ensure_ref_script(pd: Path) -> None:
    dest = pd / REF_SCRIPT
    if not dest.exists():
        src = _ref_script_source()
        if src.exists():
            shutil.copy2(src, dest)


def _prep_needed(pd: Path) -> bool:
    return not all((pd / f).exists() for f in (MAG_FILE, PHA_FILE, MASK_FILE))


def _run_prep(ctx: BenchmarkContext) -> None:
    dd = _data_dir(ctx)
    pd = _proc_dir(ctx)
    pd.mkdir(parents=True, exist_ok=True)

    bold = dd / BOLD_FILE
    phase = dd / PHASE_FILE

    if not bold.exists() or not phase.exists():
        raise FileNotFoundError(
            f"Raw data not found in {dd}\n"
            f"  Run:  ffs_benchmark -download  to fetch all benchmark datasets."
        )

    moco_params = pd / "moco_params.aff12.1D"
    if not moco_params.exists():
        print("  Running 3dvolreg (motion correction)...")
        subprocess.run(
            [
                "3dvolreg",
                "-overwrite",
                "-heptic",
                "-prefix",
                str(pd / "ignore_moco.nii.gz"),
                "-1Dmatrix_save",
                str(moco_params),
                str(bold),
            ],
            check=True,
        )

    unwrapped = pd / "unwrapped_phase.nii.gz"
    if not unwrapped.exists():
        romeo = _find_romeo()
        print(f"  Running ROMEO (phase unwrapping) via {romeo}...")
        subprocess.run(
            [
                romeo,
                "-p",
                str(phase),
                "-m",
                str(bold),
                "-t",
                "epi",
                "-o",
                str(unwrapped),
                "-v",
            ],
            check=True,
        )

    if not (pd / MAG_FILE).exists():
        print("  Running ffs_nwarp (apply motion correction to mag+phase)...")
        subprocess.run(
            [
                "ffs_nwarp",
                "-source",
                str(bold),
                "-phase",
                str(unwrapped),
                "-nwarp",
                str(moco_params),
                "-phase_warp",
                "direct",
                "-prefix",
                str(pd / "method_direct_aligned"),
                "-phase_units",
                "rad",
            ],
            check=True,
        )

    if not (pd / MASK_FILE).exists():
        print("  Running 3dAutomask...")
        subprocess.run(
            [
                "3dAutomask",
                "-overwrite",
                "-prefix",
                str(pd / MASK_FILE),
                str(bold),
            ],
            check=True,
        )

    print("  Prep complete.")


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    pd = _proc_dir(ctx)
    missing = []

    if ctx.validate_only:
        for f in (
            f"{REF_PREFIX}_slope.nii.gz",
            f"{REF_PREFIX}_r2.nii.gz",
            f"{REF_PREFIX}_corrected.nii.gz",
            f"{REF_PREFIX}_macro.nii.gz",
            f"{FFS_PREFIX}_slope.nii.gz",
            f"{FFS_PREFIX}_r2.nii.gz",
            f"{FFS_PREFIX}_corrected.nii.gz",
            f"{FFS_PREFIX}_macro.nii.gz",
            MASK_FILE,
        ):
            if not (pd / f).exists():
                missing.append(str(pd / f))
    else:
        if not _ref_script_source().exists() and not (pd / REF_SCRIPT).exists():
            missing.append(f"{_ref_script_source()}  (ref script not found)")
        try:
            import scipy  # noqa: F401
        except ImportError:
            missing.append("scipy (needed for phaseprep reference)")
        for tool in ("3dvolreg", "3dAutomask", "ffs_nwarp"):
            if not shutil.which(tool):
                missing.append(f"{tool} not found on PATH")
        try:
            _find_romeo()
        except FileNotFoundError as e:
            missing.append(str(e))

    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    pd = _proc_dir(ctx)

    if _prep_needed(pd):
        print("  Prep outputs missing — running prep (NOT timed)...")
        t0 = time.monotonic()
        _run_prep(ctx)
        print(f"  Prep took {time.monotonic() - t0:.1f}s")

    _ensure_ref_script(pd)

    ref_slope = pd / f"{REF_PREFIX}_slope.nii.gz"
    if ref_slope.exists() and not ctx.force_ref:
        print("  Skipping phaseprep ref (output exists, use -force-ref to re-run)")
        return 0.0

    n_workers = ctx.get_stage_params("phasereg").get("ref_n_workers", 8)

    elapsed, _ = run_timed(
        f"FFS_BENCHMARK_PHASEPREP_WORKERS={n_workers} python {REF_SCRIPT}",
        label=f"phaseprep reference ({n_workers} workers, scipy.odr)",
        cwd=pd,
    )
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    pd = _proc_dir(ctx)

    if _prep_needed(pd):
        print("  Prep outputs missing — running prep (NOT timed)...")
        t0 = time.monotonic()
        _run_prep(ctx)
        print(f"  Prep took {time.monotonic() - t0:.1f}s")

    out_corr = pd / f"{FFS_PREFIX}_corrected.nii.gz"
    if out_corr.exists() and not ctx.force_ffs:
        print("  Skipping ffs_phasereg (output exists, use -force-ffs to re-run)")
        return 0.0

    cmd = (
        f"ffs_phasereg "
        f"-magnitude {MAG_FILE} "
        f"-phase {PHA_FILE} "
        f"-prefix {FFS_PREFIX} "
        f"-polort 1 "
        f"-task_removal none "
        f"-mask {MASK_FILE} "
        f"-verbose"
        f"{ctx.ffs_device_flag()}"
    )
    elapsed, _ = run_timed(
        cmd,
        label="ffs_phasereg",
        cwd=pd,
    )
    return elapsed


def _load_vol(path: Path) -> np.ndarray:
    import nibabel as nib

    # nibabel ships without type stubs; ty infers the loose base FileBasedImage
    # type from its untyped source, but nib.load() always returns a concrete
    # image with a real .dataobj.
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)  # ty: ignore[unresolved-attribute]


def _load_4d(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)  # ty: ignore[unresolved-attribute]


def _ts_corr_median(a4d: np.ndarray, b4d: np.ndarray, mask3d: np.ndarray) -> float:
    a = a4d.reshape(-1, a4d.shape[-1])
    b = b4d.reshape(-1, b4d.shape[-1])
    m = mask3d.ravel().astype(bool)
    a, b = a[m], b[m]
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
    ok = denom > 0
    if not ok.any():
        return 0.0
    rs = (am[ok] * bm[ok]).sum(axis=1) / denom[ok]
    return float(np.median(rs))


def validate(ctx: BenchmarkContext) -> dict:
    pd = _proc_dir(ctx)

    ref_slope = _load_vol(pd / f"{REF_PREFIX}_slope.nii.gz")
    ref_r2 = _load_vol(pd / f"{REF_PREFIX}_r2.nii.gz")
    ref_corr = _load_4d(pd / f"{REF_PREFIX}_corrected.nii.gz")
    ref_macro = _load_4d(pd / f"{REF_PREFIX}_macro.nii.gz")

    ffs_slope = _load_vol(pd / f"{FFS_PREFIX}_slope.nii.gz")
    ffs_r2 = _load_vol(pd / f"{FFS_PREFIX}_r2.nii.gz")
    ffs_corr = _load_4d(pd / f"{FFS_PREFIX}_corrected.nii.gz")
    ffs_macro = _load_4d(pd / f"{FFS_PREFIX}_macro.nii.gz")

    mask = _load_vol(pd / MASK_FILE).astype(bool)
    n_mask = int(mask.sum())

    ref_s = ref_slope[mask]
    ffs_s = ffs_slope[mask]
    both_nz = (ref_s != 0) & (ffs_s != 0)
    if both_nz.sum() > 10:
        slope_r = float(np.corrcoef(ref_s[both_nz], ffs_s[both_nz])[0, 1])
    else:
        slope_r = 0.0
    slope_med_abs_diff = float(np.median(np.abs(ref_s - ffs_s)))

    ref_r = ref_r2[mask]
    ffs_r = ffs_r2[mask]
    r2_r = float(np.corrcoef(ref_r, ffs_r)[0, 1]) if ref_r.std() > 0 else 0.0

    corr_med_r = _ts_corr_median(ref_corr, ffs_corr, mask)
    macro_med_r = _ts_corr_median(ref_macro, ffs_macro, mask)

    passed = (
        r2_r >= THRESHOLDS["r2_r"]
        and corr_med_r >= THRESHOLDS["corrected_ts_median_r"]
        and macro_med_r >= THRESHOLDS["macro_ts_median_r"]
    )

    return {
        "passed": passed,
        "summary": (
            f"r2_r={r2_r:.4f}, "
            f"corrected_ts_med_r={corr_med_r:.4f}, macro_ts_med_r={macro_med_r:.4f}"
        ),
        "slope": {
            "r": slope_r,
            "median_abs_diff": slope_med_abs_diff,
            "n_both_nonzero": int(both_nz.sum()),
            "n_mask": n_mask,
        },
        "r2": {
            "r": r2_r,
        },
        "corrected_ts": {
            "median_r": corr_med_r,
        },
        "macro_ts": {
            "median_r": macro_med_r,
        },
    }
