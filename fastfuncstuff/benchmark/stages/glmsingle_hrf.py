"""GLMsingle Type B benchmark: HRF selection (GLMsingle vs ffs_hrfopt -single_trials).

Compares per-voxel HRF index selection between GLMsingle's FITHRF step and
FFS's ``ffs_hrfopt -single_trials``. Both use the same 20-HRF library from
getcanonicalhrflibrary.tsv and select the best HRF per voxel based on
in-sample R² of a single-trial OLS fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_hrf"
description = "HRF selection (GLMsingle Type B vs ffs_hrfopt -single_trials)"

# Thresholds: HRF selection can differ at boundaries between similar HRFs,
# so we use agreement percentage rather than correlation
THRESHOLDS = {
    "hrf_index_agreement": 0.70,  # fraction of voxels with same HRF index
    "r2_spatial_corr": 0.90,      # spatial correlation of R² maps
}


def _input_files(ctx: BenchmarkContext) -> list[Path]:
    """MNI-space resampled localizer runs."""
    return [
        ctx.processing_dir / f"ffs_mni_resampled_task-localizer_run-{r}.nii.gz"
        for r in range(1, 6)
    ]


def _onset_files(ctx: BenchmarkContext) -> list[Path]:
    """AFNI-style onset timing files (one per condition)."""
    conds = ["faces", "bodies", "objects", "scenes", "scrambled"]
    return [ctx.timing_dir / f"onsets.localizer.times.{c}.txt" for c in conds]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    gs = ctx.glmsingle_dir
    for f in ["glmsingle_hrf_index.nii.gz", "glmsingle_r2_B.nii.gz",
              "glmsingle_mask.nii.gz"]:
        p = gs / f
        if not p.exists():
            missing.append(f"GLMsingle: {p}")

    if ctx.validate_only:
        ffs = ctx.ffs_hrfopt_dir
        if not (ffs / "hrfopt_hrf_index.nii.gz").exists():
            missing.append(f"FFS: {ffs / 'hrfopt_hrf_index.nii.gz'}")
    else:
        for f in _input_files(ctx):
            if not f.exists():
                missing.append(str(f))
        for f in _onset_files(ctx):
            if not f.exists():
                missing.append(str(f))
    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_hrfopt -single_trials on localizer data."""
    out_dir = ctx.ffs_hrfopt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "hrfopt")

    # Skip if outputs exist and not forcing
    if not ctx.force_ffs and (out_dir / "hrfopt_hrf_index.nii.gz").exists():
        print("  Skipping ffs_hrfopt (outputs exist, use --force-ffs to re-run)")
        return 0.0

    inputs = " ".join(str(f) for f in _input_files(ctx))
    onsets = " ".join(str(f) for f in _onset_files(ctx))

    cmd = (
        f"ffs_hrfopt -input {inputs} "
        f"-onsets {onsets} "
        f"-durations 3.0 3.0 3.0 3.0 3.0 "
        f"-prefix {out_prefix} "
        f"-single_trials "
        f"-save_single_trial_betas "
        f"-hrf_mode library "
        f"-metric cod "
        f"-device cuda"
    )

    elapsed, _ = run_timed(cmd, label="ffs_hrfopt -single_trials", cwd=ctx.processing_dir)
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare HRF indices and R² maps between GLMsingle and FFS."""
    import nibabel as nib

    gs = ctx.glmsingle_dir
    ffs = ctx.ffs_hrfopt_dir

    # Load GLMsingle results
    matlab_hrf_index = np.array(
        nib.load(str(gs / "glmsingle_hrf_index.nii.gz")).dataobj
    ).flatten()
    matlab_r2 = np.array(
        nib.load(str(gs / "glmsingle_r2_B.nii.gz")).dataobj
    ).flatten()
    matlab_mask = np.array(
        nib.load(str(gs / "glmsingle_mask.nii.gz")).dataobj
    ).flatten().astype(bool)

    # Load FFS results
    ffs_hrf_index = np.array(nib.load(str(ffs / "hrfopt_hrf_index.nii.gz")).dataobj).flatten()
    ffs_r2 = np.array(nib.load(str(ffs / "hrfopt_xval_r2.nii.gz")).dataobj).flatten()

    # Compare within mask
    mask = matlab_mask & (ffs_r2 > 0)
    n_mask = mask.sum()

    # HRF index agreement (MATLAB is 1-indexed, FFS is 0-indexed)
    matlab_idx_masked = matlab_hrf_index[mask]
    ffs_idx_masked = ffs_hrf_index[mask]

    # Check if FFS is 0-indexed by looking at range
    if ffs_idx_masked.min() == 0 and matlab_idx_masked.min() == 1:
        ffs_idx_masked = ffs_idx_masked + 1  # Convert to 1-indexed for comparison

    agreement = (matlab_idx_masked == ffs_idx_masked).mean()

    # R² spatial correlation
    from ..validation import _pearson_r
    r2_corr = _pearson_r(matlab_r2[mask], ffs_r2[mask])

    passed = (
        agreement >= THRESHOLDS["hrf_index_agreement"]
        and r2_corr >= THRESHOLDS["r2_spatial_corr"]
    )

    summary = f"HRF agree={agreement:.1%}, R² r={r2_corr:.4f}"
    return {
        "passed": passed,
        "summary": summary,
        "hrf_index_agreement": float(agreement),
        "r2_spatial_corr": float(r2_corr),
        "n_voxels": int(n_mask),
    }
