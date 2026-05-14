"""SAUNA benchmark: ffs_sauna vs MATLAB NIFTI_NORDIC reference.

There is no existing SAUNA implementation to compare against, so we use the
MATLAB NORDIC output as a reference (both do similar patch-based denoising).
The NORDIC reference is produced by the 'nordic' stage — run that first.

Validation:
  - Median per-voxel timeseries correlation (noise volumes excluded)
  - Spatial correlation of gfactor maps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed
from .nordic import MAGN, NOISE_VOLS, PHASE, REF_GFACTOR, REF_OUT, _nordic_dir, _ts_corr_median

name = "sauna"
description = "SAUNA denoising (ffs_sauna vs MATLAB NORDIC reference)"

FFS_PREFIX = "ffs_sauna"

THRESHOLDS = {
    "ts_median_r": 0.90,
    "gfactor_r":   0.97,
}


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    nd = _nordic_dir(ctx)
    missing = []

    for f in (MAGN, PHASE):
        if not (nd / f).exists():
            missing.append(str(nd / f))

    # NORDIC reference must exist (produced by the nordic stage)
    if not (nd / REF_OUT).exists():
        missing.append(
            f"{nd / REF_OUT}  (run the 'nordic' stage first to produce NORDIC reference)"
        )

    if ctx.validate_only:
        for f in (f"{FFS_PREFIX}.nii.gz", f"{FFS_PREFIX}_gfactor.nii.gz"):
            if not (nd / f).exists():
                missing.append(str(nd / f))

    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_sauna."""
    nd = _nordic_dir(ctx)
    out = nd / f"{FFS_PREFIX}.nii.gz"

    if out.exists() and not ctx.force_ffs:
        print(f"  Skipping ffs_sauna (output exists, use -force-ffs to re-run)")
        return 0.0

    elapsed, _ = run_timed(
        f"ffs_sauna "
        f"-input-magn {MAGN} "
        f"-input-phase {PHASE} "
        f"-prefix {FFS_PREFIX} "
        f"-noise-volume-last {NOISE_VOLS} "
        f"-decomp-method eigh "
        f"-save-gfactor-map "
        f"-gfactor-method polynomial "
        f"-gfactor-degree-range 16 18 20 23 25 27"
        f"{ctx.ffs_device_flag()}",
        label="ffs_sauna",
        cwd=nd,
        verbose=ctx.verbose,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare ffs_sauna output against MATLAB NORDIC reference."""
    import nibabel as nib

    nd = _nordic_dir(ctx)

    ref_path = nd / REF_OUT
    ffs_path = nd / f"{FFS_PREFIX}.nii.gz"
    ref_gf_path = nd / REF_GFACTOR
    ffs_gf_path = nd / f"{FFS_PREFIX}_gfactor.nii.gz"

    for p in (ref_path, ffs_path):
        if not p.exists():
            return {"passed": False, "summary": f"Missing: {p.name}"}

    ref_img = nib.load(str(ref_path)).get_fdata(dtype=np.float32)
    ffs_img = nib.load(str(ffs_path)).get_fdata(dtype=np.float32)

    keep = ref_img.shape[3] - NOISE_VOLS
    mask = ref_img[..., :keep].std(axis=3) > 0

    ts_med = _ts_corr_median(ref_img, ffs_img, mask, keep)

    gf_r = None
    if ref_gf_path.exists() and ffs_gf_path.exists():
        gf_ref = nib.load(str(ref_gf_path)).get_fdata().squeeze()
        gf_ffs = nib.load(str(ffs_gf_path)).get_fdata().squeeze()
        gf_mask = gf_ref.ravel() > 0
        if gf_mask.sum() > 10:
            gf_r = float(np.corrcoef(gf_ref.ravel()[gf_mask], gf_ffs.ravel()[gf_mask])[0, 1])

    passed = ts_med >= THRESHOLDS["ts_median_r"] and (
        gf_r is None or gf_r >= THRESHOLDS["gfactor_r"]
    )

    gf_str = f" gfactor_r={gf_r:.4f}" if gf_r is not None else ""
    return {
        "passed": passed,
        "summary": f"ts_median_r={ts_med:.4f}{gf_str} (vs NORDIC reference)",
        "ts_median_r": ts_med,
        "gfactor_r": gf_r,
    }
