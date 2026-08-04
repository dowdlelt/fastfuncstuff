"""NORDIC benchmark: MATLAB NIFTI_NORDIC vs ffs_nordic.

Reference: MATLAB NIFTI_NORDIC with default settings + phase + noise-volume-last=3.
FFS:       ffs_nordic with matching settings.

Validation:
  - Median per-voxel timeseries correlation (noise volumes excluded)
  - Spatial correlation of gfactor maps
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "nordic"
description = "NORDIC denoising (MATLAB NIFTI_NORDIC vs ffs_nordic)"

MAGN = "sub-3003_ses-fine_task-expres_run-1_part-mag_bold.nii.gz"
PHASE = "sub-3003_ses-fine_task-expres_run-1_part-phase_bold.nii.gz"
REF_OUT = "NORDIC_sub-3003_ses-fine_task-expres_run-1_part-mag_bold.nii"
REF_GFACTOR = "gfactor_NORDIC_sub-3003_ses-fine_task-expres_run-1_part-mag_bold.nii"
FFS_PREFIX = "ffs_nordic_defaults"
NOISE_VOLS = 3


def _get_fdata(path: Path, dtype: type | None = None) -> np.ndarray:
    """Load a NIfTI file's data as ndarray.

    nib.load()'s stub return type is the loose FileBasedImage base; a real
    .nii/.nii.gz is always a Nifti1Image/Nifti2Image at runtime.
    """
    import nibabel as nib

    img = nib.load(str(path))
    assert isinstance(img, (nib.Nifti1Image, nib.Nifti2Image))
    return img.get_fdata(dtype=dtype) if dtype is not None else img.get_fdata()


THRESHOLDS = {
    "ts_median_r": 0.95,
    "gfactor_r": 0.99,
}


def _nordic_dir(ctx: BenchmarkContext) -> Path:
    from ..._paths import get_benchmark_data_dir

    return get_benchmark_data_dir() / "nordic_test_data"


def _nordic_toolbox() -> str | None:
    """Find NORDIC MATLAB toolbox path via env var or the reference .m script."""
    env = os.environ.get("NORDIC_MATLAB_PATH")
    if env and Path(env).exists():
        return env
    script = Path(__file__).resolve().parents[3] / "test_data" / "temp_run_nordic_defaults.m"
    if script.exists():
        for line in script.read_text().splitlines():
            m = re.search(r"addpath\(genpath\('([^']+)'\)\)", line)
            if m:
                p = Path(m.group(1).replace("~", str(Path.home())))
                if p.exists():
                    return str(p)
    return None


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    nd = _nordic_dir(ctx)
    missing = []

    for f in (MAGN, PHASE):
        if not (nd / f).exists():
            missing.append(str(nd / f))

    if ctx.validate_only:
        for f in (REF_OUT, REF_GFACTOR, f"{FFS_PREFIX}.nii.gz", f"{FFS_PREFIX}_gfactor.nii.gz"):
            if not (nd / f).exists():
                missing.append(str(nd / f))
    else:
        # Need MATLAB + toolbox to regenerate ref, OR pre-computed ref output
        ref_exists = (nd / REF_OUT).exists()
        if not ref_exists:
            if not shutil.which("matlab"):
                missing.append("matlab not found on PATH (needed to generate NORDIC reference)")
            elif _nordic_toolbox() is None:
                missing.append(
                    "NORDIC MATLAB toolbox not found — "
                    "set NORDIC_MATLAB_PATH env var or ensure test_data/temp_run_nordic_defaults.m "
                    "has a valid addpath line"
                )

    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run MATLAB NIFTI_NORDIC to produce reference denoised output."""
    nd = _nordic_dir(ctx)
    ref = nd / REF_OUT

    if ref.exists() and not ctx.force_ref:
        print("  Skipping MATLAB NORDIC (output exists, use -force-ref to re-run)")
        return 0.0

    toolbox = _nordic_toolbox()
    if toolbox is None or not shutil.which("matlab"):
        raise RuntimeError("MATLAB or NORDIC toolbox not available for run_ref")

    fn_out = REF_OUT[:-4]  # strip .nii — NIFTI_NORDIC appends it
    matlab_cmd = (
        f'matlab -batch "'
        f"cd('{nd}'); "
        f"addpath(genpath('{toolbox}')); "
        f"ARGA.temporal_phase = 1; "
        f"ARGA.phase_filter_width = 10; "
        f"ARGA.noise_volume_last = {NOISE_VOLS}; "
        f"ARGA.NORDIC = 1; "
        f"ARGA.save_gfactor_map = 1; "
        f"NIFTI_NORDIC('{MAGN}', '{PHASE}', '{fn_out}', ARGA);\""
    )

    elapsed, _ = run_timed(matlab_cmd, label="MATLAB NIFTI_NORDIC", cwd=nd)
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_nordic."""
    nd = _nordic_dir(ctx)
    out = nd / f"{FFS_PREFIX}.nii.gz"

    if out.exists() and not ctx.force_ffs:
        print("  Skipping ffs_nordic (output exists, use -force-ffs to re-run)")
        return 0.0

    elapsed, _ = run_timed(
        f"ffs_nordic "
        f"-input-magn {MAGN} "
        f"-input-phase {PHASE} "
        f"-prefix {FFS_PREFIX} "
        f"-noise-volume-last {NOISE_VOLS} "
        f"-decomp-method eigh "
        f"-save-gfactor-map"
        f"{ctx.ffs_device_flag()}",
        label="ffs_nordic",
        cwd=nd,
        verbose=ctx.verbose,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare ffs_nordic output against MATLAB NORDIC reference."""
    nd = _nordic_dir(ctx)

    ref_path = nd / REF_OUT
    ffs_path = nd / f"{FFS_PREFIX}.nii.gz"
    ref_gf_path = nd / REF_GFACTOR
    ffs_gf_path = nd / f"{FFS_PREFIX}_gfactor.nii.gz"

    for p in (ref_path, ffs_path):
        if not p.exists():
            return {"passed": False, "summary": f"Missing: {p.name}"}

    ref_img = _get_fdata(ref_path, dtype=np.float32)
    ffs_img = _get_fdata(ffs_path, dtype=np.float32)

    keep = ref_img.shape[3] - NOISE_VOLS
    mask = ref_img[..., :keep].std(axis=3) > 0

    ts_med = _ts_corr_median(ref_img, ffs_img, mask, keep)

    gf_r = None
    if ref_gf_path.exists() and ffs_gf_path.exists():
        gf_ref = _get_fdata(ref_gf_path).squeeze()
        gf_ffs = _get_fdata(ffs_gf_path).squeeze()
        gf_mask = gf_ref.ravel() > 0
        if gf_mask.sum() > 10:
            gf_r = float(np.corrcoef(gf_ref.ravel()[gf_mask], gf_ffs.ravel()[gf_mask])[0, 1])

    passed = ts_med >= THRESHOLDS["ts_median_r"] and (
        gf_r is None or gf_r >= THRESHOLDS["gfactor_r"]
    )

    gf_str = f" gfactor_r={gf_r:.4f}" if gf_r is not None else ""
    return {
        "passed": passed,
        "summary": f"ts_median_r={ts_med:.4f}{gf_str}",
        "ts_median_r": ts_med,
        "gfactor_r": gf_r,
    }


def _ts_corr_median(a4d: np.ndarray, b4d: np.ndarray, mask: np.ndarray, keep: int) -> float:
    """Median per-voxel timeseries correlation, computed slice-by-slice."""
    rs: list[float] = []
    for z in range(a4d.shape[2]):
        av = a4d[:, :, z, :keep].reshape(-1, keep).astype(np.float64)
        bv = b4d[:, :, z, :keep].reshape(-1, keep).astype(np.float64)
        mv = mask[:, :, z].ravel()
        av, bv = av[mv], bv[mv]
        if len(av) == 0:
            continue
        am = av - av.mean(axis=1, keepdims=True)
        bm = bv - bv.mean(axis=1, keepdims=True)
        num = (am * bm).sum(axis=1)
        denom = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
        valid = denom > 0
        rs.extend((num[valid] / denom[valid]).tolist())
    return float(np.median(rs)) if rs else 0.0
