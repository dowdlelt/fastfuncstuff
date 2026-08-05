#!/usr/bin/env python3
"""
3dDenoisefast - Fast cross-validated data-driven denoising using GPU acceleration

This tool implements adaptive denoising via noise pool PCA:
1. Identify noise pool voxels (low task R²) and criteria voxels (high task R²)
2. Extract PCs from noise pool as candidate nuisance regressors
3. Cross-validate to select optimal number of PCs that maximizes prediction
4. Train on denoised data but test on raw data to prevent overfitting

The key anti-overfitting strategy: we denoise training data but predict non-denoised
test data, ensuring we're improving signal recovery rather than just fitting noise removal.

Basic usage:
    3dDenoisefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                  -onsets cond1.txt cond2.txt \\
                  -durations 2.0 5.0 \\
                  -tr 2.0 \\
                  -prefix subject01_denoised

For help:
    3dDenoisefast -help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import nibabel as nib  # noqa: F401
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.cli_utils import (
        LoadResult,
        add_load_threads_arg,
        add_ortvec_arguments,
        add_verbose_arg,
        auto_polort,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
        parse_cv_strategy,
        parse_input_files,
        parse_prefix,
        preflight_check,
        spinner,
    )
    from fastfuncstuff.denoise.sequential import (
        DenoiseResults,
        compute_full_brain_pc_loadings,
        compute_noise_pool_pca_scree_per_run,
        estimate_noise_component_caps_per_run,
        fit_denoising_model,
    )
    from fastfuncstuff.design.builder import (
        create_onset_matrix_microtime,
    )
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.design.hrf_selection import load_nuisance_file  # noqa: F401
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.glm.ridge import load_hrf_indices
    from fastfuncstuff.io.afni import (
        load_afni_mask,  # noqa: F401 — availability check
        load_nifti,  # noqa: F401 — availability check
        save_nifti,
    )
    from fastfuncstuff.utils import (
        configure_torch_backends,
        get_device,
        scale_to_percent_signal,
        to_tensor,  # noqa: F401
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dDenoisefast - Fast GPU-accelerated cross-validated denoising",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic denoising with automatic R² threshold
  3dDenoisefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                -onsets cond1.txt cond2.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -prefix subject01_denoised

  # With full diagnostic plots and model fit outputs
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -plots full \\
                -save_model_fit \\
                -prefix sub01_full_diagnostics

    # With mask and spatial component weight maps
  3dDenoisefast -input run*.nii.gz \\
                -onsets stim.txt \\
                -durations 1.0 \\
                -tr 2.0 \\
                -mask brain_mask.nii.gz \\
                                -noise ica \\
                                -max_comps 30 \\
                -save_pcs both \\
                -prefix masked_denoised

  # With motion nuisance regressors
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -ortvec motion_all.1D motion \\
                -prefix sub01_motion_denoised

Outputs:
    Core outputs (always saved):
        {prefix}_noise_pool_mask.nii.gz       - Noise pool voxels (low task R²)
        {prefix}_criteria_mask.nii.gz         - Criteria voxels (high task R²)
        {prefix}_initial_r2.nii.gz            - Initial xval R² (task-only)
        {prefix}_xval_r2_optimal.nii.gz       - Xval R² at optimal PC count (criteria voxels)
        {prefix}_xval_r2_optimal_full.nii.gz  - Xval R² at optimal PC count (all voxels)
        {prefix}_xval_r2_optimal_per_fold.nii.gz - Per-fold xval R² at optimal PCs (4D)
        {prefix}_xval_r2_by_npcs.npy          - CV R² for each number of PCs
        {prefix}_metadata.json                - Full metadata for reproducibility

  With -save_pcs timecourse/both:
    {prefix}_noise_pcs.pt              - PC timecourses (.pt PyTorch file)

  With -save_pcs spatial/both:
    {prefix}_run01_pc_weights.nii.gz   - Spatial PC weights per run (4D NIfTI)

  With -plots yes/full:
    {prefix}_denoising_summary.png     - CV performance summary
    {prefix}_component_diagnostics_PC01.png - Per-component diagnostic plots (full mode)

    With -save_model_fit:
        {prefix}_initial_betas.nii.gz      - Initial model betas (4D)
        {prefix}_denoised_betas.nii.gz     - Denoised model betas (4D)

Workflow:
  1. Fit initial GLM to compute R² for each voxel
  2. Select noise pool (R² < threshold) and criteria voxels (R² >= threshold)
  3. Extract PCs from noise pool voxels per run
  4. Cross-validate: train on denoised data, test on raw data
  5. Select optimal number of PCs that maximizes CV R²
  6. Save results and optimal denoising parameters

Notes:
  - At least 2 runs required for cross-validation
  - Noise pool must have sufficient voxels (default: min 100)
  - Training on denoised but testing on raw prevents overfitting
  - Polynomial drift is always included (auto-determined or via -polort)
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Multiple files = multiple runs.",
    )
    required.add_argument(
        "-onsets",
        nargs="+",
        required=False,
        default=None,
        help="Onset timing files (AFNI format). One file per condition. "
        "Mutually exclusive with -events.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=False,
        default=None,
        help="Stimulus durations in seconds. Either single value or one per condition. "
        "Required when using -onsets; derived automatically from -events.",
    )
    required.add_argument(
        "-tr",
        type=float,
        required=False,
        default=None,
        help="Repetition time (TR) in seconds. If not specified, read from NIfTI header.",
    )
    required.add_argument(
        "-prefix",
        required=True,
        help="Output file prefix (e.g., 'output/subject01')",
    )

    # BIDS events options
    bids_opts = parser.add_argument_group("BIDS Events (alternative to -onsets/-durations)")
    bids_opts.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files, one per run. Sorted by run number automatically. "
        "Mutually exclusive with -onsets.",
    )
    bids_opts.add_argument(
        "-event_ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude. Only valid with -events.",
    )
    bids_opts.add_argument(
        "-event_cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom column names for onset, duration, trial_type. "
        "Default: onset duration trial_type. Only valid with -events.",
    )
    bids_opts.add_argument(
        "-round_onsets",
        nargs="?",
        const=0.7,
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Round onsets to nearest TR. Fraction-through-TR >= THRESHOLD → ceil, else floor. "
        "Default threshold if flag given without value: 0.7.",
    )
    bids_opts.add_argument(
        "-round_durations",
        type=int,
        default=None,
        metavar="PLACES",
        help="Round stimulus durations to PLACES decimal places.",
    )

    # Denoising options
    denoise_opts = parser.add_argument_group("Denoising Options")
    denoise_opts.add_argument(
        "-r2_threshold",
        type=float,
        default=0.05,
        help="Cross-validated R² threshold for noise pool selection (default: 0.05). "
        "Voxels with CV R² < threshold are noise pool, >= threshold are criteria. "
        "R² is computed using condition-level design (not single-trial) with COD metric "
        "on a 0-1 scale. For unmasked data, use -brainthresh to exclude background "
        "voxels first, otherwise most voxels will have R²≈0 and flood the noise pool. "
        "GLMsingle auto-determines this via GMM (findtailthreshold); typical values "
        "range 0.01-0.10 depending on data quality.",
    )
    denoise_opts.add_argument(
        "-noise",
        type=str,
        choices=["pca", "ica"],
        default="pca",
        help="Noise component method: 'pca' (default) or 'ica'.",
    )
    denoise_opts.add_argument(
        "-max_comps",
        "-max_pcs",
        dest="max_comps",
        type=int,
        default=20,
        help="Maximum number of noise components to test (default: 20). Alias: -max_pcs",
    )
    denoise_opts.add_argument(
        "-pcstop",
        type=float,
        default=1.05,
        help="PC selection stopping threshold (default: 1.05, GLMdenoise-style). "
        ">=1: Stop when R² is within (pcstop-1)*100%% of max (e.g., 1.05 = 5%%). "
        "<0: Use exactly abs(pcstop) PCs (user override). "
        "=1: Pure argmax (pick maximum R²).",
    )
    denoise_opts.add_argument(
        "-pcR2cutoff",
        type=float,
        default=None,
        help="R² cutoff for PC selection (GLMdenoise default: 0.05). "
        "If set, only voxels with max R² > cutoff across any PC count are used "
        "to compute the selection curve. More robust to noisy voxels.",
    )
    denoise_opts.add_argument(
        "-brainthresh",
        nargs=2,
        type=float,
        metavar=("PERCENTILE", "FRACTION"),
        default=None,
        help="Signal intensity threshold to exclude non-brain voxels from noise pool. "
        "Computed from raw (unscaled) mean volume: thresh = percentile(mean, P) * F. "
        "Voxels with mean intensity BELOW this threshold are excluded. "
        "GLMsingle default: '99 0.1' (99th percentile × 0.1). "
        "Defaults to '99 0.5' (the GLMdenoise default) when none of -mask, "
        "-automask or -brainthresh is given, since otherwise background voxels "
        "(R²≈0) dominate the noise pool. Pass '-brainthresh 0 0' to disable. "
        "Example: -brainthresh 99 0.5 (more conservative, excludes dim voxels).",
    )
    denoise_opts.add_argument(
        "-min_noise_voxels",
        type=int,
        default=100,
        help="Minimum voxels required in noise pool (default: 100)",
    )
    denoise_opts.add_argument(
        "-max_noise_fraction",
        type=float,
        default=0.95,
        help="Maximum fraction of voxels in noise pool (default: 0.5)",
    )
    denoise_opts.add_argument(
        "-variance_threshold",
        type=float,
        default=0.95,
        help="Cumulative variance threshold for PC extraction (default: 0.95)",
    )
    denoise_opts.add_argument(
        "-auto_component_caps",
        action="store_true",
        help=(
            "Estimate per-run component caps independently of denoising CV "
            "(noise-pool spectrum only), then extract only top components per run."
        ),
    )
    denoise_opts.add_argument(
        "-auto_component_min",
        type=int,
        default=5,
        help="Minimum per-run component cap when -auto_component_caps is enabled (default: 5)",
    )
    denoise_opts.add_argument(
        "-auto_component_var_threshold",
        type=float,
        default=0.90,
        help=(
            "Variance target used by independent cap estimator (default: 0.90). "
            "Lower values are more conservative."
        ),
    )
    denoise_opts.add_argument(
        "-auto_component_estimate_max",
        type=int,
        default=None,
        help=(
            "Upper bound used only for independent component-count estimation. "
            "If not set, defaults to 2x -max_comps. Denoising sweep still uses -max_comps."
        ),
    )
    denoise_opts.add_argument(
        "-auto_component_no_mp",
        action="store_true",
        help=(
            "Disable soft Marchenko-Pastur prior in independent cap estimation. "
            "Useful if MP assumptions are not trusted for your dataset."
        ),
    )
    denoise_opts.add_argument(
        "-ica_restarts",
        type=int,
        default=5,
        help=(
            "Number of ICA random restarts per run; best non-Gaussian solution is kept "
            "(default: 5). Increase for more robust ICA at higher compute cost."
        ),
    )
    denoise_opts.add_argument(
        "-ica_max_iter",
        type=int,
        default=1000,
        help="Maximum FastICA iterations per restart (default: 1000)",
    )
    denoise_opts.add_argument(
        "-ica_tol",
        type=float,
        default=1e-6,
        help="FastICA convergence tolerance per restart (default: 1e-6)",
    )
    denoise_opts.add_argument(
        "-cv_strategy",
        default="loro",
        help=(
            "Cross-validation strategy. Options: "
            "'loro' or '1' for leave-one-run-out (default), "
            "'0.5' for split-halves, "
            "any float (0-1) for that train fraction, "
            "any int > 1 for leave-N-out"
        ),
    )
    denoise_opts.add_argument(
        "-n_perms",
        type=int,
        default=100,
        help="Max number of CV permutations for random splits (default: 100)",
    )
    denoise_opts.add_argument(
        "-cv_metric",
        choices=["cod", "corr", "corr2", "sse"],
        default="cod",
        help="CV metric: cod (coefficient of determination), "
        "corr (Pearson correlation), corr2 (correlation squared), "
        "sse (sum of squared errors, GLMsingle-compatible). Default: cod.",
    )
    denoise_opts.add_argument(
        "-single_trials",
        action="store_true",
        help="Use beta-space cross-validation (GLMsingle-style). "
        "Fits single-trial model once on all data, evaluates R² on "
        "condition-averaged vs individual trial betas across folds.",
    )
    denoise_opts.add_argument(
        "-zscore_by_run",
        action="store_true",
        default=False,
        help="Z-score betas per run before CV using OLS normalization stats "
        "(GLMsingle default). Only applies with -single_trials.",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-mask",
        help="Mask file to restrict analysis to brain voxels",
    )
    proc_opts.add_argument(
        "-automask",
        action="store_true",
        help="Compute brain mask automatically from mean EPI (AFNI-style automask "
        "with dilate=4). Applied before brainthresh. Mutually exclusive with -mask.",
    )
    proc_opts.add_argument(
        "-keep_constant_voxels",
        "-keep-constant-voxels",
        dest="keep_constant_voxels",
        action="store_true",
        help="Keep voxels that are flat (zero variance) in one or more runs. Off by "
        "default: such voxels are out-of-FoV background and score a meaningless R2=1.",
    )
    proc_opts.add_argument(
        "-hrf_opt",
        type=str,
        default=None,
        help="HRFoptfast output prefix. Loads {prefix}_hrf_index.nii.gz for per-voxel HRFs.",
    )
    proc_opts.add_argument(
        "-hrf-library",
        dest="hrf_library",
        type=str,
        default=None,
        metavar="TSV",
        help=(
            "Custom HRF library TSV (e.g. from ffs_librarian); used when "
            "the per-voxel HRF library is loaded for denoising."
        ),
    )
    proc_opts.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order for drift modeling (default: auto based on run length)",
    )
    add_ortvec_arguments(proc_opts)
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution in seconds (default: 0.1)",
    )
    proc_opts.add_argument(
        "-hrf_model",
        type=str,
        default="spmg1",
        help="HRF model: 'spmg1' (default), 'spmg2', 'spmg3', 'glmsingle', 'FIR', 'TENT', or 'TENT(bot,top,n)'. "
        "SPMG2 = canonical + temporal derivative (2 basis). "
        "SPMG3 = canonical + time + dispersion derivatives (3 basis). "
        "FIR/TENT windows are estimated from the stimulus durations (stimulus + "
        "HRF tail); override with -fir_duration. "
        "Example: -hrf_model TENT(0,15,6) creates 6 tent bases from 0-15s.",
    )
    proc_opts.add_argument(
        "-fir_duration",
        "-fir-duration",
        "-tent_duration",
        "-tent-duration",
        dest="fir_duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="FIR/TENT window length in seconds (0 to SECONDS), overriding the window "
        "estimated from the stimulus durations. -tent_duration is a synonym. "
        "Ignored for canonical HRF models.",
    )
    proc_opts.add_argument(
        "-canonical",
        type=str,
        default=None,
        help="DEPRECATED: Use -hrf_model instead. Kept for backwards compatibility.",
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change)",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process in GPU chunks (for large datasets)",
    )
    add_load_threads_arg(proc_opts)
    add_verbose_arg(proc_opts, default=0)
    proc_opts.add_argument(
        "-dry_run",
        action="store_true",
        help="Fast testing mode: load only first run, generate synthetic data for rest. "
        "Results are nonsensical but pipeline runs quickly for testing.",
    )
    proc_opts.add_argument(
        "-R2method",
        type=str,
        choices=["auto", "fast", "slow"],
        default="auto",
        help="R² computation method. 'fast' uses streaming stats (~3MB vs ~8GB memory), "
        "requires LORO CV. 'slow' stores full timeseries (for non-LORO CV). "
        "'auto' selects based on CV strategy (default: auto).",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-plots",
        type=str,
        choices=["no", "yes", "full"],
        default="no",
        help="Save diagnostic plots: 'no' (none), 'yes' (summary), 'full' (summary + per-PC plots)",
    )
    out_opts.add_argument(
        "-no_scree_plot",
        dest="scree_plot",
        action="store_false",
        help=(
            "Disable default noise-pool PCA scree plot output. "
            "By default, scree is saved to {prefix}_figures/noise_pool_pca_scree.png"
        ),
    )
    out_opts.add_argument(
        "-scree_max_comps",
        type=int,
        default=None,
        help=(
            "Max PCA components per run for noise-pool scree computation. "
            "Default: auto (uses auto estimate max when available, else max(2x -max_comps, 60))."
        ),
    )
    out_opts.add_argument(
        "-plot_ax",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Slice axis for PC spatial maps: 'x' (sagittal), 'y' (coronal), 'z' (axial)",
    )
    out_opts.add_argument(
        "-save_pcs",
        type=str,
        choices=["no", "timecourse", "spatial", "both"],
        default="timecourse",
        help="Save noise PCs: 'no', 'timecourse' (default: .pt file), 'spatial' (NIfTI weight maps), 'both'",
    )
    out_opts.add_argument(
        "-component_map_space",
        type=str,
        choices=["full", "noise_pool"],
        default="full",
        help=(
            "Spatial map strategy for component diagnostics: "
            "'full' (default, refit component weights to all brain voxels) or "
            "'noise_pool' (show extraction-space weights only)."
        ),
    )
    out_opts.add_argument(
        "-save_model_fit",
        action="store_true",
        help="Save initial and final (denoised) model fit outputs (betas, tstats) as NIfTI",
    )
    out_opts.add_argument(
        "-snr",
        action="store_true",
        help="Compute and save SNR (signal-to-noise ratio) outputs. "
        "Creates SNR volumes before/after denoising and scatter plot comparison. "
        "Computes both residual-based SNR and bootstrap-based SNR (if -numboots > 0).",
    )
    out_opts.add_argument(
        "-numboots",
        type=int,
        default=0,
        help="Number of bootstrap iterations for SE estimation (default: 0 = no bootstrapping). "
        "Recommended: 100-1000 for robust SE. Enables bootstrap-based SNR if -snr is also set.",
    )

    return parser


def print_header(args):
    """Print program header"""
    print("=" * 70)
    print("3dDenoisefast - GPU-Accelerated Cross-Validated Denoising")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def _select_design_for_visualization(
    task_design: torch.Tensor | None,
    designs_by_hrf: dict | None,
    hrf_library: torch.Tensor | None,
) -> tuple[torch.Tensor, int | None]:
    """Select task design to visualize.

    For per-voxel HRF mode, picks the middle HRF index (floor(n_hrfs/2)) when
    n_hrfs > 2. If that HRF is missing from designs_by_hrf, falls back to the
    nearest available HRF index.
    """
    if task_design is not None:
        return task_design, None

    if designs_by_hrf is None or len(designs_by_hrf) == 0:
        raise ValueError("No design matrix available for visualization")

    available_indices = sorted(int(k) for k in designs_by_hrf.keys())
    if hrf_library is not None and len(hrf_library) > 2:
        target_hrf_idx = int(len(hrf_library) // 2)
    else:
        target_hrf_idx = available_indices[len(available_indices) // 2]

    if target_hrf_idx in designs_by_hrf:
        selected_idx = target_hrf_idx
    else:
        selected_idx = min(available_indices, key=lambda x: abs(x - target_hrf_idx))

    return designs_by_hrf[selected_idx], selected_idx


def save_final_design_matrix_plot(
    output_prefix: str,
    task_design_to_plot: torch.Tensor,
    nuisance_per_run: list[torch.Tensor],
    run_starts: list[int],
    selected_hrf_idx: int | None = None,
) -> str:
    """Save high-resolution, non-blurry final design matrix image.

    Final design = [task design, block-diagonal nuisance], in TR-bin space.
    Time is shown vertically (rows), regressors horizontally (columns).
    """
    import matplotlib.pyplot as plt

    fig_dir = Path(f"{output_prefix}_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(fig_dir / "final_design_matrix.png")

    task_cpu = task_design_to_plot.detach().cpu().float()
    nuisance_cpu = [n.detach().cpu().float() for n in nuisance_per_run]

    n_task_cols = int(task_cpu.shape[1])
    total_nuisance_cols = sum(int(n.shape[1]) for n in nuisance_cpu)

    if total_nuisance_cols > 0:
        nuisance_block = torch.block_diag(*nuisance_cpu)
        final_design = torch.cat([task_cpu, nuisance_block], dim=1)
    else:
        final_design = task_cpu

    design_np = final_design.numpy()
    abs_max = float(np.percentile(np.abs(design_np), 99.5))
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0

    n_tps, n_cols = design_np.shape
    fig_h = min(22, max(8, n_tps / 220))
    fig_w = min(24, max(10, n_cols / 12))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(
        design_np,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="nearest",
        origin="upper",
    )

    if n_task_cols < n_cols:
        ax.axvline(n_task_cols - 0.5, color="black", linewidth=1.0)
    for rs in run_starts[1:]:
        ax.axhline(rs - 0.5, color="black", linewidth=0.7, alpha=0.6)

    title = "Final design matrix (TR space): task + block-diagonal nuisance"
    if selected_hrf_idx is not None:
        title += f" | HRF index used: {selected_hrf_idx}"
    ax.set_title(title)
    ax.set_xlabel("Regressors")
    ax.set_ylabel("Time (TR)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


def save_denoising_results(
    results: DenoiseResults,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    run_starts: list[int],
    tr: float,
    data_for_component_maps: torch.Tensor | None = None,
    voxel_mask: torch.Tensor | None = None,
    plots_mode: str = "no",
    slice_axis: str = "x",
    component_map_space: str = "full",
    noise_method: str = "pca",
    save_pcs_mode: str = "timecourse",
    condition_labels: list[str] | None = None,
    save_scree_plot: bool = True,
    nii_ext: str = ".nii.gz",
    nifti_header: object | None = None,
):
    """
    Save denoising results to disk

    Parameters
    ----------
    results : DenoiseResults
        Denoising results
    output_prefix : str
        Output file prefix
    volume_shape : tuple
        Shape of 3D volume
    affine : np.ndarray
        Affine matrix for NIfTI files
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    data_for_component_maps : torch.Tensor, optional
        Full data tensor (n_voxels, n_timepoints). Required for
        component_map_space='full' to refit spatial weights to all voxels.
    voxel_mask : torch.Tensor, optional
        Voxel mask (if brain mask was used)
    plots_mode : str
        'no', 'yes' (summary only), or 'full' (summary + per-PC)
    component_map_space : str
        Spatial map source for plots: 'full' (refit to all voxels) or
        'noise_pool' (use extraction-space loadings)
    noise_method : str
        Noise component method label ('pca' or 'ica') for logging/metadata.
    save_pcs_mode : str
        'no', 'timecourse', 'spatial', or 'both'
    condition_labels : list of str, optional
        Labels for task conditions

    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    # Note: Results already have extreme R² voxels excluded via valid_voxel_mask
    # All results tensors are in the same space as the input data (no reallocation was done)
    # So we can use voxel_mask directly without modification

    # Helper to reshape flat data to volume
    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # 1. Noise pool mask
    noise_pool_vol = to_volume(results.noise_pool_mask.cpu().numpy().astype(np.float32))
    noise_pool_path = f"{output_prefix}_noise_pool_mask{nii_ext}"
    save_nifti(noise_pool_vol, output_path=noise_pool_path, affine=affine, header=nifti_header)
    output_files["noise_pool_mask"] = noise_pool_path

    # 2. Criteria mask
    criteria_vol = to_volume(results.criteria_mask.cpu().numpy().astype(np.float32))
    criteria_path = f"{output_prefix}_criteria_mask{nii_ext}"
    save_nifti(criteria_vol, output_path=criteria_path, affine=affine, header=nifti_header)
    output_files["criteria_mask"] = criteria_path

    # 3. Initial R²
    initial_r2_vol = to_volume(results.noise_pool_r2.cpu().numpy().astype(np.float32))
    initial_r2_path = f"{output_prefix}_initial_r2{nii_ext}"
    save_nifti(initial_r2_vol, output_path=initial_r2_path, affine=affine, header=nifti_header)
    output_files["initial_r2"] = initial_r2_path

    # 3b. Xval R² at optimal PC count (criteria voxels only)
    if results.xval_r2_optimal is not None:
        xval_opt_vol = to_volume(results.xval_r2_optimal.cpu().numpy().astype(np.float32))
        xval_opt_path = f"{output_prefix}_xval_r2_optimal{nii_ext}"
        save_nifti(xval_opt_vol, output_path=xval_opt_path, affine=affine, header=nifti_header)
        output_files["xval_r2_optimal"] = xval_opt_path

    # 3c. Xval R² at optimal PC count (all voxels)
    if results.xval_r2_optimal_full is not None:
        xval_opt_full_vol = to_volume(results.xval_r2_optimal_full.cpu().numpy().astype(np.float32))
        xval_opt_full_path = f"{output_prefix}_xval_r2_optimal_full{nii_ext}"
        save_nifti(
            xval_opt_full_vol, output_path=xval_opt_full_path, affine=affine, header=nifti_header
        )
        output_files["xval_r2_optimal_full"] = xval_opt_full_path

    # 3d. Per-fold xval R² at optimal PCs (4D)
    if results.xval_r2_optimal_per_fold is not None:
        fold_vols = []
        for fold_idx in range(results.xval_r2_optimal_per_fold.shape[0]):
            fold_vol = to_volume(results.xval_r2_optimal_per_fold[fold_idx].astype(np.float32))
            fold_vols.append(fold_vol)

        fold_4d = np.stack(fold_vols, axis=-1)
        fold_path = f"{output_prefix}_xval_r2_optimal_per_fold{nii_ext}"
        save_nifti(fold_4d, output_path=fold_path, affine=affine, header=nifti_header)
        output_files["xval_r2_optimal_per_fold"] = fold_path

    # 4. CV R² arrays
    xval_r2_path = f"{output_prefix}_xval_r2_by_npcs.npy"
    np.save(xval_r2_path, results.xval_r2_by_n_components)
    output_files["xval_r2_by_npcs"] = xval_r2_path

    xval_r2_folds_path = f"{output_prefix}_xval_r2_per_fold.npy"
    np.save(xval_r2_folds_path, results.xval_r2_per_fold)
    output_files["xval_r2_per_fold"] = xval_r2_folds_path

    # 5. Metadata
    metadata = {
        **results.metadata,
        "optimal_n_components": results.optimal_n_components,
        "baseline_r2": results.baseline_r2,
        "optimal_r2": results.optimal_r2,
        "improvement": results.improvement,
        "volume_shape": list(volume_shape),
        "tr": tr,
        "run_starts": run_starts,
        "component_map_space": component_map_space,
        "noise_method": noise_method,
    }
    if condition_labels:
        metadata["condition_labels"] = condition_labels

    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_path

    # 6. Noise PCs (based on save_pcs_mode)
    if save_pcs_mode in ["timecourse", "both"]:
        pcs_path = f"{output_prefix}_noise_pcs.pt"
        torch.save(
            {
                "noise_pcs_per_run": results.noise_pcs_per_run,
                "optimal_n_components": results.optimal_n_components,
                "run_starts": run_starts,
            },
            pcs_path,
        )
        output_files["noise_pcs_timecourse"] = pcs_path

    if save_pcs_mode in ["spatial", "both"]:
        # Save PC spatial weights as NIfTI files (per run, per PC)
        if results.pc_loadings_per_run is not None:
            # Helper to reshape noise pool loadings to full volume
            noise_pool_np = results.noise_pool_mask.cpu().numpy()

            def loadings_to_volume(loadings_flat):
                """Map noise pool loadings back to full volume (zeros outside noise pool)"""
                if voxel_mask_np is not None:
                    # Two-level mask: voxel_mask (full volume) and noise_pool (within masked voxels)
                    # Map noise_pool indices into full-volume indices via voxel_mask
                    brain_indices = np.where(voxel_mask_np)[0]
                    noise_pool_indices = brain_indices[noise_pool_np]
                    vol = np.zeros(np.prod(volume_shape), dtype=loadings_flat.dtype)
                    vol[noise_pool_indices] = loadings_flat
                else:
                    # No brain mask, noise_pool is directly in volume space
                    vol = np.zeros(np.prod(volume_shape), dtype=loadings_flat.dtype)
                    vol[noise_pool_np] = loadings_flat
                return vol.reshape(volume_shape)

            n_runs = len(results.pc_loadings_per_run)
            for run_idx, loadings in enumerate(results.pc_loadings_per_run):
                loadings_np = loadings.cpu().numpy() if torch.is_tensor(loadings) else loadings
                n_pcs = loadings_np.shape[1]

                # Save each PC as a separate volume (or combine into 4D)
                pc_vols = []
                for pc_idx in range(
                    min(n_pcs, results.optimal_n_components + 3)
                ):  # Save optimal + a few more
                    pc_vol = loadings_to_volume(loadings_np[:, pc_idx])
                    pc_vols.append(pc_vol)

                # Stack into 4D and save
                pc_4d = np.stack(pc_vols, axis=-1)
                pc_path = f"{output_prefix}_run{run_idx + 1:02d}_pc_weights{nii_ext}"
                save_nifti(pc_4d, output_path=pc_path, affine=affine, header=nifti_header)
                output_files[f"run{run_idx + 1}_pc_weights"] = pc_path

            print(f"  Saved PC spatial weights for {n_runs} runs")
        else:
            print("  Warning: PC loadings not available (run with return_loadings=True)")

    # 6b. Save selected PCs as text files (one per run)
    # These are the PC timecourses for the optimal number of components
    n_runs = len(results.noise_pcs_per_run)
    for run_idx, pcs in enumerate(results.noise_pcs_per_run):
        pcs_np = pcs.cpu().numpy() if torch.is_tensor(pcs) else pcs
        # Take only the selected (optimal) number of PCs
        n_selected = min(results.optimal_n_components, pcs_np.shape[1])
        selected_pcs = pcs_np[:, :n_selected]

        pc_txt_path = f"{output_prefix}_run{run_idx + 1:02d}_selected_PCs.txt"
        # Save with header
        with open(pc_txt_path, "w") as f:
            f.write(f"# Selected noise PCs for run {run_idx + 1}\n")
            f.write(f"# n_components: {n_selected}\n")
            f.write(f"# Shape: {selected_pcs.shape[0]} timepoints x {selected_pcs.shape[1]} PCs\n")
            f.write(f"# Columns: PC1, PC2, ..., PC{n_selected}\n")
            np.savetxt(f, selected_pcs, fmt="%.6f", delimiter="\t")
        output_files[f"run{run_idx + 1}_selected_pcs_txt"] = pc_txt_path

    print(
        f"  Saved selected PCs ({results.optimal_n_components} PCs) as text files for {n_runs} runs"
    )

    # 6c. Noise-pool PCA scree plot (default on)
    if save_scree_plot:
        scree_ratio = results.metadata.get("noise_pool_pca_scree_ratio_per_run")
        if scree_ratio is not None and len(scree_ratio) > 0:
            try:
                import matplotlib.pyplot as plt

                from fastfuncstuff.visualization import plot_noise_pool_pca_scree

                fig_prefix = f"{output_prefix}_figures"
                Path(fig_prefix).mkdir(parents=True, exist_ok=True)
                scree_path = f"{fig_prefix}/noise_pool_pca_scree.png"
                scree_fig = plot_noise_pool_pca_scree(
                    scree_ratio_per_run=scree_ratio,
                    variance_threshold=results.metadata.get("auto_component_var_threshold", None),
                    output_path=scree_path,
                )
                plt.close(scree_fig)
                output_files["noise_pool_pca_scree_plot"] = scree_path
                print(f"  Saved: {scree_path}")
            except Exception as e:
                print(f"  Warning: Could not save noise-pool PCA scree plot: {e}")

    # 7. Plots (based on plots_mode)
    if plots_mode in ["yes", "full"]:
        try:
            from fastfuncstuff.visualization import plot_denoising_pcs, plot_denoising_summary

            # create figure prefix
            fig_prefix = f"{output_prefix}_figures"
            Path(fig_prefix).mkdir(parents=True, exist_ok=True)

            # Summary plot
            r2_cpu = results.noise_pool_r2.cpu().numpy()
            summary_fig = plot_denoising_summary(
                xval_r2_by_n_components=results.xval_r2_by_n_components,
                xval_r2_per_fold=results.xval_r2_per_fold,
                optimal_n_components=results.optimal_n_components,
                initial_r2_distribution=r2_cpu,
                r2_threshold=results.metadata["r2_threshold"],
                n_noise_voxels=results.metadata["n_noise_voxels"],
                n_criteria_voxels=results.metadata["n_criteria_voxels"],
                output_path=f"{fig_prefix}/denoising_summary.png",
            )
            output_files["denoising_summary_plot"] = f"{fig_prefix}/denoising_summary.png"
            import matplotlib.pyplot as plt

            plt.close(summary_fig)

            # Per-PC plots (only for "full" mode)
            if plots_mode == "full":
                print(
                    f"  Component diagnostics: method={noise_method.upper()}, map_space={component_map_space}"
                )
                # Convert PC tensors to CPU for plotting
                pcs_cpu = (
                    [pc.cpu() for pc in results.noise_pcs_per_run]
                    if results.noise_pcs_per_run
                    else None
                )

                # Create combined mask: voxel_mask (brain) AND noise_pool_mask (low R²)
                # This maps noise pool indices to full volume space
                noise_pool_mask_np = results.noise_pool_mask.cpu().numpy()

                if component_map_space == "full" and data_for_component_maps is not None:
                    full_loadings = compute_full_brain_pc_loadings(
                        data=data_for_component_maps,
                        noise_pcs_per_run=results.noise_pcs_per_run,
                        run_starts=run_starts,
                        brain_mask=None,
                        chunk_size=5000,
                        device=None,
                        verbose=False,
                    )
                    loadings_cpu = [ld.numpy() for ld in full_loadings]
                    noise_pool_mask_for_plot = None
                else:
                    loadings_cpu = (
                        [ld.cpu().numpy() for ld in results.pc_loadings_per_run]
                        if results.pc_loadings_per_run
                        else None
                    )
                    noise_pool_mask_for_plot = noise_pool_mask_np

                plot_denoising_pcs(
                    noise_pcs_per_run=pcs_cpu,
                    run_starts=run_starts,
                    component_variance_ratio_per_run=results.metadata.get(
                        "ic_variance_ratio_per_run"
                    ),
                    pc_weights_per_run=loadings_cpu,
                    volume_shape=volume_shape,
                    voxel_mask=voxel_mask_np,
                    noise_pool_mask=noise_pool_mask_for_plot,
                    n_pcs_to_show=results.metadata.get(
                        "extraction_max_components", results.metadata.get("max_components", 0)
                    ),
                    n_slices=5,
                    slice_axis=slice_axis,
                    tr=tr,
                    optimal_n_pcs=results.optimal_n_components,
                    output_prefix=f"{fig_prefix}/component_diagnostics",
                    return_figs=False,
                )
                output_files["component_diagnostic_plots"] = (
                    f"{fig_prefix}/component_diagnostics_PC*.png"
                )
                output_files["pc_diagnostic_plots"] = output_files["component_diagnostic_plots"]

        except ImportError as e:
            print(f"  Warning: Could not import visualization module: {e}")
        except Exception as e:
            print(f"  Warning: Error creating plots: {e}")

    return output_files


def compute_bootstrap_se(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    n_boots: int = 100,
    chunk_size: int = 5000,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute bootstrap standard errors for beta coefficients.

    Uses residual bootstrap: resample residuals, add to fitted values, refit.

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    design : torch.Tensor
        (n_timepoints, n_regressors) full design matrix
    n_task : int
        Number of task regressors (first n columns)
    n_boots : int
        Number of bootstrap iterations
    chunk_size : int
        Voxels per batch
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    bootstrap_se : np.ndarray
        (n_voxels, n_task) standard error for each task beta
    """
    if device is None:
        device = get_device()

    n_voxels, n_timepoints = data.shape
    n_regressors = design.shape[1]

    # Storage for bootstrap betas
    boot_betas = np.zeros((n_boots, n_voxels, n_task), dtype=np.float32)

    # Compute original fit once
    design_gpu = design.to(device)
    _XtX_inv = torch.linalg.inv(
        design_gpu.T @ design_gpu + 1e-6 * torch.eye(n_regressors, device=device)
    )

    if verbose:
        from tqdm import tqdm

        boot_iter = tqdm(range(n_boots), desc="  Bootstrap iterations")
    else:
        boot_iter = range(n_boots)

    for boot_idx in boot_iter:
        # Resample timepoints with replacement
        resample_idx = torch.randint(0, n_timepoints, (n_timepoints,), device=device)

        design_boot = design_gpu[resample_idx, :]
        XtX_inv_boot = torch.linalg.inv(
            design_boot.T @ design_boot + 1e-6 * torch.eye(n_regressors, device=device)
        )

        # Process in chunks
        for chunk_start in range(0, n_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_voxels)

            data_chunk = data[chunk_start:chunk_end, :].to(device)
            data_boot = data_chunk[:, resample_idx]

            # OLS fit
            betas_boot = (XtX_inv_boot @ design_boot.T @ data_boot.T).T  # (chunk, n_regressors)
            boot_betas[boot_idx, chunk_start:chunk_end, :] = betas_boot[:, :n_task].cpu().numpy()

    # Compute SE as std across bootstrap samples
    bootstrap_se = np.std(boot_betas, axis=0)  # (n_voxels, n_task)

    return bootstrap_se


def compute_snr(
    betas: np.ndarray,
    residual_std: np.ndarray | None = None,
    bootstrap_se: np.ndarray | None = None,
) -> dict:
    """
    Compute SNR metrics from betas and noise estimates.

    Parameters
    ----------
    betas : np.ndarray
        (n_voxels, n_task) beta coefficients
    residual_std : np.ndarray, optional
        (n_voxels,) residual standard deviation
    bootstrap_se : np.ndarray, optional
        (n_voxels, n_task) bootstrap standard errors

    Returns
    -------
    snr_dict : dict
        'snr_residual': (n_voxels,) max|beta| / residual_std
        'snr_bootstrap': (n_voxels,) max(|beta| / bootstrap_se)
    """
    result = {}

    # Max absolute beta across conditions (the "signal")
    max_abs_beta = np.max(np.abs(betas), axis=1)  # (n_voxels,)

    # Residual-based SNR: signal / noise_floor
    if residual_std is not None:
        snr_residual = max_abs_beta / (residual_std + 1e-10)
        result["snr_residual"] = snr_residual

    # Bootstrap-based SNR: max of per-condition SNR
    if bootstrap_se is not None:
        # Per-condition SNR
        per_cond_snr = np.abs(betas) / (bootstrap_se + 1e-10)  # (n_voxels, n_task)
        snr_bootstrap = np.max(per_cond_snr, axis=1)  # (n_voxels,)
        result["snr_bootstrap"] = snr_bootstrap

    return result


def save_snr_outputs(
    snr_initial: dict,
    snr_denoised: dict,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    voxel_mask: torch.Tensor | None = None,
    create_plots: bool = True,
    nii_ext: str = ".nii.gz",
    nifti_header: object | None = None,
) -> dict:
    """
    Save SNR volumes and create before/after comparison plots.
    """
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # Save residual-based SNR volumes
    if "snr_residual" in snr_initial:
        snr_vol = to_volume(snr_initial["snr_residual"].astype(np.float32))
        snr_path = f"{output_prefix}_snr_residual_initial{nii_ext}"
        save_nifti(snr_vol, output_path=snr_path, affine=affine, header=nifti_header)
        output_files["snr_residual_initial"] = snr_path

    if "snr_residual" in snr_denoised:
        snr_vol = to_volume(snr_denoised["snr_residual"].astype(np.float32))
        snr_path = f"{output_prefix}_snr_residual_denoised{nii_ext}"
        save_nifti(snr_vol, output_path=snr_path, affine=affine, header=nifti_header)
        output_files["snr_residual_denoised"] = snr_path

    # Save bootstrap-based SNR volumes
    if "snr_bootstrap" in snr_initial:
        snr_vol = to_volume(snr_initial["snr_bootstrap"].astype(np.float32))
        snr_path = f"{output_prefix}_snr_bootstrap_initial{nii_ext}"
        save_nifti(snr_vol, output_path=snr_path, affine=affine, header=nifti_header)
        output_files["snr_bootstrap_initial"] = snr_path

    if "snr_bootstrap" in snr_denoised:
        snr_vol = to_volume(snr_denoised["snr_bootstrap"].astype(np.float32))
        snr_path = f"{output_prefix}_snr_bootstrap_denoised{nii_ext}"
        save_nifti(snr_vol, output_path=snr_path, affine=affine, header=nifti_header)
        output_files["snr_bootstrap_denoised"] = snr_path

    # Create scatter plots
    if create_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Residual-based SNR scatter
            if "snr_residual" in snr_initial and "snr_residual" in snr_denoised:
                ax = axes[0]
                x = snr_initial["snr_residual"]
                y = snr_denoised["snr_residual"]

                # Subsample for plotting if too many points
                if len(x) > 10000:
                    idx = np.random.choice(len(x), 10000, replace=False)
                    x_plot, y_plot = x[idx], y[idx]
                else:
                    x_plot, y_plot = x, y

                ax.scatter(x_plot, y_plot, alpha=0.3, s=1, c="steelblue")
                max_val = max(np.percentile(x, 99), np.percentile(y, 99))
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="unity")
                ax.set_xlabel("SNR (initial)")
                ax.set_ylabel("SNR (denoised)")
                ax.set_title(f"Residual-based SNR\nMean: {x.mean():.2f} → {y.mean():.2f}")
                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.legend()

            # Bootstrap-based SNR scatter
            if "snr_bootstrap" in snr_initial and "snr_bootstrap" in snr_denoised:
                ax = axes[1]
                x = snr_initial["snr_bootstrap"]
                y = snr_denoised["snr_bootstrap"]

                if len(x) > 10000:
                    idx = np.random.choice(len(x), 10000, replace=False)
                    x_plot, y_plot = x[idx], y[idx]
                else:
                    x_plot, y_plot = x, y

                ax.scatter(x_plot, y_plot, alpha=0.3, s=1, c="darkorange")
                max_val = max(np.percentile(x, 99), np.percentile(y, 99))
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="unity")
                ax.set_xlabel("SNR (initial)")
                ax.set_ylabel("SNR (denoised)")
                ax.set_title(f"Bootstrap-based SNR\nMean: {x.mean():.2f} → {y.mean():.2f}")
                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.legend()
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "Bootstrap SNR\nnot computed\n(use -numboots)",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].set_title("Bootstrap-based SNR")

            plt.tight_layout()
            plot_path = f"{output_prefix}_snr_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            output_files["snr_plot"] = plot_path

        except Exception as e:
            print(f"  Warning: Could not create SNR plots: {e}")

    return output_files


def save_model_fit_outputs(
    results,  # GLMResults
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    model_type: str,  # "initial" or "denoised"
    condition_labels: list[str] | None = None,
    voxel_mask: torch.Tensor | None = None,
    n_timepoints: int | None = None,
    n_regressors: int | None = None,
    bootstrap_se: np.ndarray | None = None,
    nii_ext: str = ".nii.gz",
    nifti_header: object | None = None,
):
    """
    Save GLM model fit outputs (betas, tstats) as NIfTI files with AFNI labeling

    AFNI-style output: betas and tstats are interleaved in a single 4D file
    with sub-bricks ordered as: [beta1, tstat1, beta2, tstat2, ...]

    Sub-brick labels and t-stat DOF are written into the AFNI extension
    in-script by save_nifti (no 3drefit dependency).

    Parameters
    ----------
    results : GLMResults
        GLM results from fit_glm
    output_prefix : str
        Output file prefix
    volume_shape : tuple
        Shape of 3D volume
    affine : np.ndarray
        Affine matrix for NIfTI files
    model_type : str
        "initial" or "denoised"
    condition_labels : list of str, optional
        Labels for task conditions
    voxel_mask : torch.Tensor, optional
        Voxel mask (if brain mask was used)
    n_timepoints : int, optional
        Number of timepoints (for DOF calculation)
    n_regressors : int, optional
        Total number of regressors in model (for DOF calculation)
    bootstrap_se : np.ndarray, optional
        Bootstrap standard errors (n_voxels, n_task) if available

    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # Get number of task regressors (first n columns, rest are nuisance)
    betas = results.betas.cpu().numpy() if torch.is_tensor(results.betas) else results.betas
    n_total_regs = betas.shape[1]
    n_task = len(condition_labels) if condition_labels else n_total_regs

    # Get tstats if available
    has_tstats = results.tstats is not None
    if has_tstats:
        tstats = results.tstats.cpu().numpy() if torch.is_tensor(results.tstats) else results.tstats

    # Calculate DOF for t-statistics
    dof = None
    if n_timepoints is not None and n_regressors is not None:
        dof = n_timepoints - n_regressors

    # Build AFNI-style bucket file: interleaved betas and tstats
    # Sub-brick order: [beta1, tstat1, beta2, tstat2, ...]
    bucket_vols = []
    sub_brick_labels = []
    sub_brick_types = []  # 'coef' or 'tstat' for 3drefit

    for reg_idx in range(n_task):
        label = condition_labels[reg_idx] if condition_labels else f"reg{reg_idx}"

        # Add beta
        beta_vol = to_volume(betas[:, reg_idx].astype(np.float32))
        bucket_vols.append(beta_vol)
        sub_brick_labels.append(f"{label}#0_Coef")
        sub_brick_types.append("coef")

        # Add tstat (if available)
        if has_tstats:
            tstat_vol = to_volume(tstats[:, reg_idx].astype(np.float32))
            bucket_vols.append(tstat_vol)
            sub_brick_labels.append(f"{label}#0_Tstat")
            sub_brick_types.append("tstat")

    # Add bootstrap SE sub-bricks if available
    if bootstrap_se is not None:
        for reg_idx in range(n_task):
            label = condition_labels[reg_idx] if condition_labels else f"reg{reg_idx}"
            se_vol = to_volume(bootstrap_se[:, reg_idx].astype(np.float32))
            bucket_vols.append(se_vol)
            sub_brick_labels.append(f"{label}#0_SE")
            sub_brick_types.append("se")

    # Stack into 4D
    bucket_4d = np.stack(bucket_vols, axis=-1)
    bucket_path = f"{output_prefix}_{model_type}_bucket{nii_ext}"

    # AFNI sub-brick labels + t-stat DOF written in-script by save_nifti (no
    # 3drefit round-trip): tag each 'tstat' sub-brick as fitt(dof).
    from fastfuncstuff.io.afni import stat_type_to_stataux

    brick_stataux: dict[int, tuple[int, tuple[float, ...]]] | None = None
    if dof is not None:
        brick_stataux = {
            i: stat_type_to_stataux("fitt", (dof,))
            for i, sbtype in enumerate(sub_brick_types)
            if sbtype == "tstat"
        }

    save_nifti(
        bucket_4d,
        output_path=bucket_path,
        affine=affine,
        header=nifti_header,
        brick_labels=sub_brick_labels,
        brick_stataux=brick_stataux or None,
    )
    output_files[f"{model_type}_bucket"] = bucket_path

    return output_files


def main():
    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    print_header(args)

    # Parse and validate inputs
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    # Validate onset/events mutual exclusivity
    _has_onsets = bool(args.onsets)
    _has_events = bool(args.events)
    if _has_onsets and _has_events:
        print("ERROR: Specify only one of -onsets/-durations or -events")
        sys.exit(1)
    if not _has_onsets and not _has_events:
        print("ERROR: Must specify one of -onsets/-durations or -events")
        sys.exit(1)
    if args.event_ignore and not _has_events:
        print("ERROR: -event_ignore requires -events")
        sys.exit(1)
    if args.event_cols and not _has_events:
        print("ERROR: -event_cols requires -events")
        sys.exit(1)
    if _has_onsets and args.durations is None:
        print("ERROR: -durations is required when using -onsets")
        sys.exit(1)

    # Parse onset metadata (condition labels and durations)
    from fastfuncstuff.cli_utils import parse_timing_spec

    try:
        timing = parse_timing_spec(
            events=args.events,
            onsets=args.onsets,
            durations_arg=args.durations,
            n_runs=n_runs,
            event_ignore=args.event_ignore,
            event_cols=tuple(args.event_cols) if args.event_cols else None,
            round_durations=args.round_durations,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_onsets = timing.all_onsets
    durations = timing.durations
    condition_labels = timing.condition_labels
    n_conditions = timing.n_conditions
    onset_files = timing.onset_files

    # Parse CV strategy
    cv_strategy = parse_cv_strategy(args.cv_strategy)
    if args.verb >= 1:
        print(f"  CV strategy: {cv_strategy}")

    # Pre-flight checks (before slow data loading)
    preflight_check(
        input_files=input_files,
        onset_files=onset_files,
        ortvec_files=[(f, label) for f, label in args.ortvec] if args.ortvec else None,
        hrf_opt_prefix=args.hrf_opt or None,
    )

    # Setup device
    if args.device:
        device = torch.device(args.device if args.device.lower() != "cpu" else "cpu")
    else:
        device = get_device()
    configure_torch_backends(device)
    print(f"  Device: {device}")

    # ==========================================================================
    # Load data
    # ==========================================================================

    print()

    # Load and preprocess data using unified utility
    load_result: LoadResult = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=False,  # Scaling will be applied later if requested
        device=device,
        force_cpu=args.keep_on_cpu,
        dry_run=args.dry_run,
        verbose=True,
        load_threads=args.load_threads,
    )

    # Modify prefix for dry run mode
    if args.dry_run:
        args.prefix = f"dry_run_{args.prefix}"

    # Extract loaded data
    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
    nifti_header = load_result.nifti_header
    volume_shape = load_result.volume_shape
    voxel_sizes = load_result.voxel_sizes
    mask = load_result.mask
    mask_flat = load_result.mask_flat
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    n_runs = load_result.n_runs
    keep_on_cpu = load_result.keep_on_cpu

    # Update args.tr with loaded value (for later use)
    if args.tr is None:
        args.tr = load_result.tr

    # HRF model parsing needs the TR (FIR/TENT derive their basis count from it),
    # so it has to wait until the header has been read.
    from fastfuncstuff.cli_utils import parse_hrf_model_args, validate_hrf_compatibility

    hrf_info = parse_hrf_model_args(
        hrf_model_arg=args.hrf_model,
        canonical_arg=args.canonical,
        durations=durations,
        condition_labels=condition_labels,
        tr=args.tr,
        fir_window_s=args.fir_duration,
    )

    hrf_model_name = hrf_info["hrf_model_name"]
    _hrf_params = hrf_info["hrf_params"]
    is_fir_model = hrf_info["is_fir_model"]
    fir_bot = hrf_info["fir_bot"]
    fir_top = hrf_info["fir_top"]
    n_basis = hrf_info["n_basis"]
    _condition_labels_full = hrf_info["condition_labels_full"]

    validate_hrf_compatibility(
        is_fir_model=is_fir_model,
        single_trial=args.single_trials,
        hrf_opt=args.hrf_opt,
    )

    # Compute automask if requested (before scaling, before brainthresh)
    if args.automask:
        if args.mask:
            print("ERROR: -automask and -mask are mutually exclusive")
            sys.exit(1)

        from fastfuncstuff.cli_utils import apply_automask

        print()
        print("Computing automask from mean EPI...")

        data, mask, mask_flat, n_voxels = apply_automask(
            data=data,
            run_starts=run_starts,
            volume_shape=volume_shape,
            mask_flat=mask_flat,
            verbose=True,
        )
        # Saved after the constant-voxel filter below, so the file on disk is
        # the voxel set actually analysed rather than the pre-filter automask.

    print()

    # Compute brainthresh intensity mask BEFORE scaling
    # This excludes low-intensity voxels from the noise pool.
    # Unrestricted, the noise pool is "every voxel with low task R²", which is
    # mostly air; those PCs are noise/ghosting, not shared nuisance structure.
    # GLMdenoise's 99th-percentile x 0.5 is the reference default, so apply it
    # unless the user has already restricted the voxel set some other way.
    if args.brainthresh is None and args.mask is None and not args.automask:
        args.brainthresh = [99.0, 0.5]
        print()
        print("No -mask / -automask / -brainthresh given: defaulting to")
        print("  -brainthresh 99 0.5 so background voxels stay out of the noise pool.")
        print("  Pass '-brainthresh 0 0' to disable the intensity cut entirely.")

    brainthresh_mask = None
    if args.brainthresh is not None:
        percentile, fraction = args.brainthresh
        print()
        print(f"Computing intensity threshold (brainthresh={percentile}, {fraction})...")

        # Compute mean intensity per voxel (across time)
        mean_intensity = data.mean(dim=1)  # Shape: (n_voxels,)

        # Get the percentile value
        percentile_value = torch.quantile(mean_intensity, percentile / 100.0)
        threshold = percentile_value * fraction

        # Create mask: True for voxels ABOVE threshold (valid voxels)
        brainthresh_mask = mean_intensity > threshold
        n_above = brainthresh_mask.sum().item()

        print(f"  {percentile:.0f}th percentile intensity: {percentile_value:.2f}")
        print(f"  Threshold ({fraction:.2f} × {percentile_value:.2f}): {threshold:.2f}")
        print(
            f"  Voxels above threshold: {n_above:,} of {n_voxels:,} ({n_above / n_voxels * 100:.1f}%)"
        )

    # Drop voxels with zero/low variance in ANY run (out-of-FoV background, the
    # empty corners of an oblique-rotated volume, dead slices). Excluding them
    # from the noise pool alone is not enough: they still reach the R² maps,
    # where SS_res = SS_tot = 0 scores them a spurious 1.0.
    from fastfuncstuff.cli_utils import find_constant_voxels, restrict_voxels

    print()
    print("Filtering voxels with invalid data in any run...")
    valid_per_run_mask = find_constant_voxels(data, run_starts)
    n_valid = int(valid_per_run_mask.sum().item())
    n_invalid = n_voxels - n_valid

    if n_invalid > 0:
        if args.keep_constant_voxels:
            print(
                f"  {n_invalid:,} voxels are zero/constant in at least one run "
                "(kept: -keep_constant_voxels); excluded from the noise pool"
            )
            brainthresh_mask = (
                valid_per_run_mask
                if brainthresh_mask is None
                else brainthresh_mask & valid_per_run_mask
            )
        else:
            print(f"  Removed {n_invalid:,} voxels with zero/constant values in any run")
            print(f"  Valid voxels: {n_valid:,} ({n_valid / n_voxels * 100:.1f}%)")
            data, mask, mask_flat, n_voxels = restrict_voxels(
                data, valid_per_run_mask, volume_shape, mask_flat
            )
            if brainthresh_mask is not None:
                brainthresh_mask = brainthresh_mask[valid_per_run_mask.to(brainthresh_mask.device)]

    if args.automask and mask is not None:
        save_nifti(
            mask.astype(np.float32),
            output_path=f"{args.prefix}_automask{_nii_ext}",
            affine=affine,
            header=nifti_header,
        )
        print(f"  Saved: {args.prefix}_automask{_nii_ext}")

    # Optional scaling
    if args.do_scale:
        print()
        data, _, scale_info = scale_to_percent_signal(
            data=data,
            run_starts=run_starts,
            max_scale=200.0,
            verbose=True,
        )

    print(f"  Data shape: {data.shape} ({n_voxels:,} voxels × {n_timepoints} timepoints)")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ==========================================================================
    # Load per-voxel HRFs if provided
    # ==========================================================================
    hrf_library = None
    hrf_indices = None

    if args.hrf_opt:
        print()
        print(f"Loading HRF optimization results from {args.hrf_opt}...")
        hrf_index_file = f"{args.hrf_opt}_hrf_index.nii.gz"

        # Load HRF indices (applies mask if data was masked)
        hrf_indices = load_hrf_indices(hrf_index_file, mask=mask)
        print(f"  Loaded HRF indices: {hrf_indices.shape}")

        # Load HRF library (reconstruct from metadata or use default)
        # For now, use default library matching 3dHRFoptfast
        hrf_library = get_hrf_library(
            mode="library",
            stim_duration=0.0,
            microtime_dt=args.microtime_dt,
            n_hrfs=20,
            library_path=args.hrf_library,
        )
        if args.hrf_library:
            print(f"  Loaded custom HRF library from {args.hrf_library}")
        print(f"  Using HRF library with {len(hrf_library)} HRFs")

        # Show HRF distribution
        unique_hrfs, counts = torch.unique(hrf_indices, return_counts=True)
        print(f"  HRF distribution across {len(unique_hrfs)} unique HRFs:")
        for hrf_idx, count in zip(unique_hrfs[:5].tolist(), counts[:5].tolist(), strict=False):
            print(f"    HRF {hrf_idx}: {count:,} voxels ({count / n_voxels * 100:.1f}%)")
        if len(unique_hrfs) > 5:
            print(f"    ... and {len(unique_hrfs) - 5} more HRFs")

    # ==========================================================================
    # Build design matrix
    # ==========================================================================

    # Design matrix structure:
    # - TASK regressors: Shared across all runs (e.g., ring_01, ring_02, ...)
    #   Shape: (n_total_timepoints, n_task_predictors)
    #   No padding needed - same columns used by all runs
    #
    # - NUISANCE regressors: Run-specific (polynomial drift per run)
    #   Stored as list: nuisance_per_run[i] = (n_timepoints_run_i, n_nuisance_cols)
    #   Column padding needed: All runs padded to max # of nuisance columns
    #
    # Total model: Y = X_task @ beta_task + X_nuisance @ beta_nuisance + error
    # Columns: n_task (shared) + n_nuisance_padded (run-specific, column-padded)

    print()
    print("Building design matrix...")

    # Apply onset rounding (after TR is known from data load)
    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, args.tr, threshold=args.round_onsets)

    # Build onset matrix at microtime resolution (shared function ensures
    # grid-consistent bin placement matching convolve_hrf_microtime)
    onset_matrix_micro = create_onset_matrix_microtime(
        all_onsets=all_onsets,
        run_starts=run_starts,
        tr=args.tr,
        n_timepoints=n_timepoints,
        microtime_dt=args.microtime_dt,
        stim_durations=durations,
        device=device,
    )

    # Build design matrix based on HRF model
    from fastfuncstuff.cli_utils import build_task_design_from_args

    task_design, designs_by_hrf = build_task_design_from_args(
        hrf_model_name=hrf_model_name,
        is_fir_model=is_fir_model,
        fir_bot=fir_bot,
        fir_top=fir_top,
        n_basis=n_basis,
        all_onsets=all_onsets,
        onset_matrix_micro=onset_matrix_micro,
        n_conditions=n_conditions,
        n_timepoints=n_timepoints,
        run_starts=run_starts,
        tr=args.tr,
        microtime_dt=args.microtime_dt,
        device=device,
        hrf_opt=args.hrf_opt,
        hrf_library=hrf_library,
        hrf_indices=hrf_indices,
        n_voxels=n_voxels,
    )

    # Build polynomial nuisance regressors PER RUN
    # -------------------------------------------
    # CRITICAL: Nuisance regressors are RUN-SPECIFIC (each run has its own drift)
    # Different runs can have different # of columns (different polort based on duration)
    # We'll pad all to max # columns so they can be concatenated during CV

    nuisance_per_run = []
    max_nuisance_cols = 0  # Track max columns across all runs

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp

        # Auto-determine polort if not specified (per-run basis)
        if args.polort is None:
            run_duration = run_length * args.tr
            polort = auto_polort(run_duration, formula="afni")
        else:
            polort = args.polort

        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
        else:
            poly = torch.zeros((run_length, 0), device=device)

        nuisance_per_run.append(poly)
        max_nuisance_cols = max(max_nuisance_cols, poly.shape[1])

    # Add user-supplied nuisance blocks (any of -ortvec / -ortvec_run / -ortvec_glob).
    nuisance_blocks_user = collect_nuisance_blocks(
        args,
        run_starts,
        n_timepoints,
        verbose=(args.verb >= 1),
    )
    if nuisance_blocks_user:
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = end_tp - start_tp
            for block in nuisance_blocks_user:
                if block.n_columns == 0:
                    continue
                m = block.get_run(run_idx, run_length).copy()
                col_mean = m.mean(axis=0, keepdims=True)
                if np.max(np.abs(col_mean)) > 1e-4:
                    m = m - col_mean
                m_t = torch.from_numpy(m).to(
                    device=device,
                    dtype=nuisance_per_run[run_idx].dtype,
                )
                nuisance_per_run[run_idx] = torch.cat(
                    [nuisance_per_run[run_idx], m_t],
                    dim=1,
                )
            max_nuisance_cols = max(max_nuisance_cols, nuisance_per_run[run_idx].shape[1])

    # Pad all runs to have same number of columns (for CV concatenation compatibility)
    # -------------------------------------------
    # PADDING STRUCTURE:
    # - Task columns: NO padding (shared across runs)
    # - Nuisance columns: YES padding (run-specific, must match for concatenation)
    #
    # Example: Run 0 has 3 poly cols, Run 1 has 4 poly cols → pad Run 0 to 4
    # This allows clean concatenation during CV: concat([run1_nuisance, run2_nuisance])

    for run_idx in range(n_runs):
        n_cols = nuisance_per_run[run_idx].shape[1]
        if n_cols < max_nuisance_cols:
            # Pad with zeros on the right (extra polynomial terms this run doesn't need)
            padding = torch.zeros(
                (nuisance_per_run[run_idx].shape[0], max_nuisance_cols - n_cols), device=device
            )
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], padding], dim=1)

    # Summary
    if task_design is not None:
        if is_fir_model:
            # FIR: Show structure, not all labels (too many)
            print(
                f"  Task predictors: {task_design.shape[1]} ({n_conditions} conditions × {n_basis} lags)"
            )
        else:
            # Canonical: Show condition names
            print(f"  Task predictors: {task_design.shape[1]} ({', '.join(condition_labels)})")
        print(f"  Nuisance predictors per run: {nuisance_per_run[0].shape[1]} (polynomial drift)")
        print(
            f"  Total columns per run: {task_design.shape[1]} task + {nuisance_per_run[0].shape[1]} nuisance = {task_design.shape[1] + nuisance_per_run[0].shape[1]}"
        )
        designs_by_hrf = None
    else:
        # Per-HRF mode: get shape from first design matrix.
        # build_task_design_from_args returns exactly one of (task_design, designs_by_hrf).
        assert designs_by_hrf is not None
        first_hrf_idx = list(designs_by_hrf.keys())[0]
        n_task_cols = designs_by_hrf[first_hrf_idx].shape[1]
        print(f"  Task predictors: {n_task_cols} ({', '.join(condition_labels)})")
        print(f"  Nuisance predictors per run: {nuisance_per_run[0].shape[1]} (polynomial drift)")
        print(
            f"  Total columns per run: {n_task_cols} task + {nuisance_per_run[0].shape[1]} nuisance = {n_task_cols + nuisance_per_run[0].shape[1]}"
        )

    design_plot_path: str | None = None
    if args.plots == "full":
        try:
            design_for_plot, selected_hrf_idx = _select_design_for_visualization(
                task_design=task_design,
                designs_by_hrf=designs_by_hrf,
                hrf_library=hrf_library,
            )
            design_plot_path = save_final_design_matrix_plot(
                output_prefix=args.prefix,
                task_design_to_plot=design_for_plot,
                nuisance_per_run=nuisance_per_run,
                run_starts=run_starts,
                selected_hrf_idx=selected_hrf_idx,
            )
            print(f"  Saved final design matrix image: {design_plot_path}")
        except Exception as exc:
            print(f"  Warning: could not save final design matrix image: {exc}")

    # ==========================================================================
    # Fit denoising model
    # ==========================================================================

    print()
    print("=" * 70)
    print("Fitting cross-validated denoising model...")
    print("=" * 70)

    # Memory strategy:
    # - PCA needs noise pool voxels loaded (subset of data - can't chunk)
    # - GLM fitting chunks voxels automatically via chunk_size
    # - PCs are cached timecourses (tiny memory footprint)
    # - For 16GB GPU: chunk_size=None (auto) works for most datasets
    # - When keep_on_cpu=True: set preload_data_to_device=False to avoid GPU OOM

    # Determine chunk_size based on device:
    # - CPU: process all voxels at once (no GPU memory limit)
    # - GPU: auto-detect based on available memory
    if device.type == "cpu":
        chunk_size = n_voxels  # Process all voxels at once on CPU
        if args.verb >= 1:
            print(f"  CPU mode: chunk_size = {n_voxels:,} (all voxels)")
    else:
        chunk_size = None  # Auto-detect for GPU

    # Fit denoising model
    if args.single_trials:
        # ========== SINGLE-TRIAL BETA-SPACE CV PATH ==========
        from fastfuncstuff.denoise.sequential import (
            extract_noise_ics_per_run,
            extract_noise_pcs_per_run,
        )
        from fastfuncstuff.glm.core import fit_glm
        from fastfuncstuff.glm.outputs import save_single_trial_results
        from fastfuncstuff.glm.ridge import create_single_trial_design
        from fastfuncstuff.glm.xval import (
            compute_xval_r2,
            compute_xval_r2_single_trials,
            generate_cv_splits,
            metric_higher_is_better,
            project_out_nuisance_per_run,
            single_trial_cv_helper,
        )

        print()
        print("=" * 70)
        print("Using beta-space cross-validation for PC selection")
        print("=" * 70)
        print()

        # 1. Build single-trial design (with optional per-voxel HRFs and derivatives)
        print("Building single-trial design...")
        st_design, trial_labels, trial_cond_ids, trial_run_ids, cond_design = (
            create_single_trial_design(
                onsets_by_condition=all_onsets,
                durations=durations,
                run_starts=run_starts,
                tr=args.tr,
                n_timepoints=n_timepoints,
                hrf_library=hrf_library,
                hrf_index_per_voxel=hrf_indices,
                microtime_dt=args.microtime_dt,
                condition_labels=condition_labels,
                device=device,
                hrf_model_name=hrf_model_name,
                n_basis=n_basis if not is_fir_model else 1,  # FIR incompatible with single-trial
            )
        )
        # Companion table describing every single-trial volume, in design order.
        from fastfuncstuff.design.trial_table import write_single_trial_event_table

        write_single_trial_event_table(
            args.prefix,
            args.events,
            run_starts,
            args.tr,
            event_ignore=args.event_ignore,
            event_cols=tuple(args.event_cols) if args.event_cols else None,
            n_runs=n_runs,
            n_basis=n_basis if not is_fir_model else 1,
        )

        per_voxel_st = st_design.ndim == 3  # (n_unique_hrfs, n_timepoints, n_trials * n_basis)
        n_columns = (
            trial_labels.__len__() if hasattr(trial_labels, "__len__") else len(trial_labels)
        )
        n_basis_actual = n_basis if not is_fir_model else 1
        n_trials_actual = n_columns // n_basis_actual

        print(f"  Single-trial design: {st_design.shape}")
        print(f"  Total columns: {n_columns} ({n_trials_actual} trials × {n_basis_actual} basis)")
        print(f"  Condition design: {cond_design.shape}")

        # Pre-compute HRF group info for per-voxel path (used in steps 2, 5, 7)
        if per_voxel_st:
            # per_voxel_st (3D st_design) only occurs when hrf_index_per_voxel was
            # passed to create_single_trial_design, i.e. hrf_indices is not None.
            assert hrf_indices is not None
            hrf_indices_dev = hrf_indices.to(data.device)
            unique_hrfs = torch.unique(hrf_indices_dev).tolist()
            n_trials = st_design.shape[-1]
            n_conditions_st = int(trial_cond_ids.max().item()) + 1
            print(f"  Per-voxel HRF mode: {len(unique_hrfs)} unique HRFs")

        # 2. Compute initial cross-validated R² for noise pool selection
        # Project out nuisance (polynomials) from both data and design, then
        # cross-validate with condition-level design (same approach as non-single-trial path)
        print()
        print(
            "Computing cross-validated R² with condition-level design for noise pool selection..."
        )
        print("  (project-first nuisance removal, per run)")
        cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=args.n_perms)

        if not per_voxel_st:
            # Standard path: single condition design for all voxels
            projected_data, projected_cond_design = project_out_nuisance_per_run(
                data=data,
                design=cond_design,
                nuisance_per_run=nuisance_per_run,
                run_starts=run_starts,
                device=data.device,
            )

            n_cond_cols = cond_design.shape[1]
            xval_init = compute_xval_r2(
                data=projected_data,
                design_matrix=projected_cond_design,
                run_starts=run_starts,
                stim_indices=list(range(n_cond_cols)),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric="cod",
                zero_event_strategy="zero",
                device=device,
                verbose=False,
            )

            r2_init = xval_init["r2"]
            assert isinstance(
                r2_init, torch.Tensor
            )  # "r2" key is always a tensor, unlike n_splits etc.
            initial_r2 = r2_init.to(device)
            del projected_data, projected_cond_design
        else:
            # Per-voxel HRF path: compute R² per HRF group with correct design
            # (matches denoise.py fit_denoising_model per-HRF initial R² logic)
            print(f"  Computing per-HRF-group R² ({len(unique_hrfs)} groups)...")
            initial_r2 = torch.zeros(data.shape[0], device=device)

            for hrf_idx in unique_hrfs:
                voxel_mask = hrf_indices_dev == hrf_idx
                n_group = voxel_mask.sum().item()

                # Build condition-level design from this HRF's single-trial design
                hrf_st_design = st_design[hrf_idx]  # (n_tp, n_trials)
                cond_design_hrf = torch.zeros(n_timepoints, n_conditions_st, device=device)
                for c in range(n_conditions_st):
                    cond_mask = trial_cond_ids == c
                    if cond_mask.sum() > 0:
                        cond_design_hrf[:, c] = hrf_st_design[:, cond_mask].sum(dim=1)

                # Project nuisance and compute xval R² for this group
                group_data = data[voxel_mask]
                proj_data, proj_design = project_out_nuisance_per_run(
                    data=group_data,
                    design=cond_design_hrf,
                    nuisance_per_run=nuisance_per_run,
                    run_starts=run_starts,
                    device=data.device,
                )

                xval_group = compute_xval_r2(
                    data=proj_data,
                    design_matrix=proj_design,
                    run_starts=run_starts,
                    stim_indices=list(range(n_conditions_st)),
                    nuisance_indices=[],
                    cv_splits=cv_splits,
                    metric="cod",
                    zero_event_strategy="zero",
                    device=device,
                    verbose=False,
                )

                r2_group = xval_group["r2"]
                assert isinstance(r2_group, torch.Tensor)  # "r2" key is always a tensor
                initial_r2[voxel_mask] = r2_group.to(device)
                print(
                    f"    HRF {hrf_idx}: {n_group:,} voxels, "
                    f"median R²={r2_group.median().item():.4f}"
                )

                del proj_data, proj_design

        print(f"  Initial xval R²: mean={initial_r2.mean():.4f}, median={initial_r2.median():.4f}")
        print(f"  R² range: [{initial_r2.min().item():.4f}, {initial_r2.max().item():.4f}]")

        # 3. Select noise pool (R² < threshold)
        print()
        print(f"Selecting noise pool (R² < {args.r2_threshold})...")
        noise_pool_mask = initial_r2 < args.r2_threshold

        # Apply intensity threshold if provided
        if brainthresh_mask is not None:
            noise_pool_mask = noise_pool_mask & brainthresh_mask.to(device)

        n_noise = noise_pool_mask.sum().item()
        n_total = noise_pool_mask.numel()
        noise_fraction = n_noise / n_total

        print(f"  Noise pool: {n_noise:,} voxels ({100 * noise_fraction:.1f}%)")

        if n_noise < args.min_noise_voxels:
            print(f"ERROR: Insufficient noise voxels ({n_noise} < {args.min_noise_voxels})")
            sys.exit(1)

        if noise_fraction > args.max_noise_fraction:
            print(
                f"WARNING: Noise fraction ({noise_fraction:.2f}) exceeds max ({args.max_noise_fraction})"
            )
            print("  Consider increasing -r2_threshold or using -mask")

        # 4. Extract noise PCs from noise pool
        print()
        component_caps_per_run = None
        extraction_max_comps = args.max_comps
        noise_pool_scree_ratio_per_run = None
        if args.auto_component_caps:
            print("Estimating independent per-run component caps...")
            estimate_max_comps = (
                args.auto_component_estimate_max
                if args.auto_component_estimate_max is not None
                else max(args.max_comps, 2 * args.max_comps)
            )
            cap_info = estimate_noise_component_caps_per_run(
                data=data,
                run_starts=run_starts,
                noise_pool_mask=noise_pool_mask,
                max_components=estimate_max_comps,
                nuisance_per_run=nuisance_per_run,
                min_components=args.auto_component_min,
                variance_threshold=args.auto_component_var_threshold,
                use_mp_prior=not args.auto_component_no_mp,
                device=device,
                verbose=True,
            )
            component_caps_per_run = cap_info.per_run_caps
            component_caps_per_run = [
                int(max(1, min(args.max_comps, c))) for c in component_caps_per_run
            ]
            extraction_max_comps = args.max_comps
            print(f"  Per-run caps: {component_caps_per_run}")
            print(
                f"  Decomposition estimate ceiling: {estimate_max_comps}; denoising sweep cap: {args.max_comps}"
            )
            print("  Per-run search diagnostics:")
            for run_i, cap in enumerate(component_caps_per_run):
                mp_cap = cap_info.mp_caps[run_i]
                mp_txt = f"n/a[{cap_info.mp_reasons[run_i]}]" if mp_cap is None else str(mp_cap)
                print(
                    f"    Run {run_i + 1}: cap={cap}, "
                    f"search {cap_info.search_final_max_per_run[run_i]}/{cap_info.search_ceiling_per_run[run_i]} "
                    f"in {cap_info.search_iterations[run_i]} iter "
                    f"(var={cap_info.variance_caps[run_i]}, "
                    f"erank={cap_info.entropy_rank_caps[run_i]}, mp={mp_txt})"
                )

        if args.scree_plot:
            scree_eval_max = (
                args.scree_max_comps
                if args.scree_max_comps is not None
                else (
                    args.auto_component_estimate_max
                    if args.auto_component_estimate_max is not None
                    else max(args.max_comps * 2, 60)
                )
            )
            scree_eval_max = max(5, int(scree_eval_max))
            try:
                noise_pool_scree_ratio_per_run = compute_noise_pool_pca_scree_per_run(
                    data=data,
                    run_starts=run_starts,
                    noise_pool_mask=noise_pool_mask,
                    max_components=scree_eval_max,
                    nuisance_per_run=nuisance_per_run,
                    device=device,
                )
                print(
                    f"  Computed noise-pool PCA scree spectra (up to {scree_eval_max} components per run)"
                )
            except Exception as e:
                print(f"  Warning: Failed to compute noise-pool PCA scree spectra: {e}")

        print(
            f"Extracting noise components via {args.noise.upper()} "
            f"(up to {extraction_max_comps} for decomposition)"
        )
        want_noise_pool_maps = args.plots == "full" and args.component_map_space == "noise_pool"
        component_loadings_per_run = None
        ic_variance_ratio_per_run = None
        if args.noise == "pca":
            if want_noise_pool_maps:
                noise_pcs_per_run, component_loadings_per_run = extract_noise_pcs_per_run(
                    data=data,
                    run_starts=run_starts,
                    noise_pool_mask=noise_pool_mask,
                    max_components=extraction_max_comps,
                    variance_threshold=args.variance_threshold,
                    return_loadings=True,
                    nuisance_per_run=nuisance_per_run,
                    component_caps_per_run=component_caps_per_run,
                    device=device,
                    verbose=args.verb >= 1,
                )
            else:
                noise_pcs_per_run = extract_noise_pcs_per_run(
                    data=data,
                    run_starts=run_starts,
                    noise_pool_mask=noise_pool_mask,
                    max_components=extraction_max_comps,
                    variance_threshold=args.variance_threshold,
                    nuisance_per_run=nuisance_per_run,
                    component_caps_per_run=component_caps_per_run,
                    device=device,
                    verbose=args.verb >= 1,
                )
        else:
            if want_noise_pool_maps:
                noise_pcs_per_run, component_loadings_per_run, ic_variance_ratio_per_run = (
                    extract_noise_ics_per_run(
                        data=data,
                        run_starts=run_starts,
                        noise_pool_mask=noise_pool_mask,
                        max_components=extraction_max_comps,
                        return_loadings=True,
                        return_variance_ratio=True,
                        nuisance_per_run=nuisance_per_run,
                        component_caps_per_run=component_caps_per_run,
                        ica_restarts=args.ica_restarts,
                        ica_max_iter=args.ica_max_iter,
                        ica_tol=args.ica_tol,
                        device=device,
                        verbose=args.verb >= 1,
                    )
                )
            else:
                noise_pcs_per_run, ic_variance_ratio_per_run = extract_noise_ics_per_run(
                    data=data,
                    run_starts=run_starts,
                    noise_pool_mask=noise_pool_mask,
                    max_components=extraction_max_comps,
                    return_variance_ratio=True,
                    nuisance_per_run=nuisance_per_run,
                    component_caps_per_run=component_caps_per_run,
                    ica_restarts=args.ica_restarts,
                    ica_max_iter=args.ica_max_iter,
                    ica_tol=args.ica_tol,
                    device=device,
                    verbose=args.verb >= 1,
                )

        # 5. For each PC count: fit single-trial model with wide design, collect betas
        print()
        print(f"Optimizing component count (0 to {args.max_comps})...")
        n_voxels_st = data.shape[0]
        n_pc_counts = args.max_comps + 1

        # Determine n_trials from the design
        if not per_voxel_st:
            n_trials_st = st_design.shape[1]
        else:
            n_trials_st = st_design.shape[2]  # (n_unique_hrfs, n_tp, n_trials)

        # Collect all betas: (n_pc_counts, n_voxels, n_trials) on CPU
        all_st_betas = torch.zeros(n_pc_counts, n_voxels_st, n_trials_st)

        pc_range = range(n_pc_counts)
        for n_pcs in tqdm(pc_range, desc="  Fitting PC counts", disable=not args.verb >= 1):
            # Build nuisance: polynomials + first n_pcs per run
            nuisance_blocks = []
            for run_idx in range(n_runs):
                run_nuisance = nuisance_per_run[run_idx]
                if n_pcs > 0:
                    pcs = noise_pcs_per_run[run_idx][:, :n_pcs]
                    run_nuisance = torch.cat([run_nuisance, pcs], dim=1)
                nuisance_blocks.append(run_nuisance)
            nuisance_design = torch.block_diag(*nuisance_blocks)

            # Fit GLM and extract single-trial betas
            if not per_voxel_st:
                # Standard 2D path
                full_design = torch.cat([st_design, nuisance_design], dim=1)
                task_indices = list(range(st_design.shape[1]))
                glm_results = fit_glm(
                    data,
                    full_design,
                    tr=args.tr,
                    max_poly_degree=-1,  # block-diagonal nuisance already has per-run constants
                    device=device,
                    verbose=False,
                    task_indices=task_indices,
                )
                assert glm_results.betas is not None  # set by fit_glm above
                all_st_betas[n_pcs] = glm_results.betas.cpu()
            else:
                # Per-voxel HRF path: group by HRF index
                task_indices = list(range(n_trials_st))
                for hrf_idx in unique_hrfs:
                    voxel_mask = hrf_indices_dev == hrf_idx
                    group_design_2d = st_design[hrf_idx]  # (n_tp, n_trials)
                    full_design = torch.cat([group_design_2d, nuisance_design], dim=1)
                    glm_results = fit_glm(
                        data[voxel_mask],
                        full_design,
                        tr=args.tr,
                        max_poly_degree=-1,  # block-diagonal nuisance already has per-run constants
                        device=device,
                        verbose=False,
                        task_indices=task_indices,
                    )
                    assert glm_results.betas is not None  # set by fit_glm above
                    all_st_betas[n_pcs, voxel_mask] = glm_results.betas.cpu()

        # Batch beta-space CV across all PC counts at once
        print("  Computing beta-space CV R² for all PC counts (batch)...")
        xval = single_trial_cv_helper(
            all_st_betas,
            trial_cond_ids,
            trial_run_ids,
            cv_splits,
            metric=args.cv_metric,
            zscore_by_run=args.zscore_by_run,
            reference_variant_idx=0,  # 0 PCs = unregularized baseline for z-score stats
            device=device,
            verbose=False,
        )
        r2_maps_st = xval["r2"].T  # (n_voxels, n_pc_counts)
        _hib = metric_higher_is_better(args.cv_metric)
        _metric_label = args.cv_metric.upper()

        # Determine criteria mask from initial R² (COD-based, computed earlier).
        # This is the GLMsingle/GLMdenoise pattern: criteria voxels are those with
        # meaningful task signal, regardless of what metric is used for PC optimization.
        # initial_r2 is always COD (computed at step 2 above).
        criteria_mask = (initial_r2 > args.r2_threshold).cpu()
        n_criteria = criteria_mask.sum().item()
        print(
            f"  Criteria voxels (initial R² > {args.r2_threshold}): "
            f"{n_criteria:,} / {n_voxels_st:,}"
        )

        if n_criteria == 0:
            print("  WARNING: No voxels meet criteria! Using all voxels.")
            criteria_mask = torch.ones(n_voxels_st, dtype=torch.bool)

        r2_criteria = r2_maps_st[criteria_mask, :]  # (n_criteria, n_pc_counts)
        r2_by_pc = r2_criteria.median(dim=0).values  # (n_pc_counts,)

        # For SSE, negate to get "xvaltrend" (higher = better), matching GLMsingle's
        # convention: xvaltrend = -median(glmbadness).  This lets the same pcstop
        # logic work for both directions.
        if not _hib:
            xvaltrend = -r2_by_pc  # negate SSE so higher = better
        else:
            xvaltrend = r2_by_pc

        print(f"  Baseline (0 PCs): median {_metric_label} = {r2_by_pc[0].item():.4f}")
        if _hib:
            best_idx = int(xvaltrend.argmax().item())
        else:
            best_idx = int(xvaltrend.argmax().item())
        print(f"  Best ({best_idx} PCs): median {_metric_label} = {r2_by_pc[best_idx].item():.4f}")

        # 6. Select optimal PC count using pcstop criterion
        # xvaltrend is always in "higher = better" convention here
        print()
        if args.pcstop < 0:
            # User override: use exactly abs(pcstop) PCs
            optimal_pcs = int(abs(args.pcstop))
            print(f"User-specified PC count: {optimal_pcs}")
        elif args.pcstop == 1.0:
            optimal_pcs = int(xvaltrend.argmax().item())
            print(f"Optimal PC count (argmax): {optimal_pcs}")
        else:
            # GLMdenoise-style: walk forward, track best, stop when within
            # pcstop fraction of the eventual max.  Operates on xvaltrend
            # (always higher = better regardless of original metric).
            curve = xvaltrend - xvaltrend[0]  # starts at 0 for 0 PCs
            mx = curve.max().item()
            best = float("-inf")
            optimal_pcs = 0
            for p in range(len(curve)):
                if curve[p].item() > best:
                    optimal_pcs = p
                    best = curve[p].item()
                    if best * args.pcstop >= mx:
                        break
            print(f"Optimal PC count (pcstop={args.pcstop}): {optimal_pcs}")

        # 7. Refit with optimal PC count and save
        print()
        print("Refitting with optimal PC count...")
        nuisance_blocks = []
        for run_idx in range(n_runs):
            run_nuisance = nuisance_per_run[run_idx]
            if optimal_pcs > 0:
                pcs = noise_pcs_per_run[run_idx][:, :optimal_pcs]
                run_nuisance = torch.cat([run_nuisance, pcs], dim=1)
            nuisance_blocks.append(run_nuisance)
        nuisance_design_final = torch.block_diag(*nuisance_blocks)

        if not per_voxel_st:
            # Standard 2D path
            full_design_final = torch.cat([st_design, nuisance_design_final], dim=1)
            task_indices_final = list(range(st_design.shape[1]))
            glm_results_final = fit_glm(
                data,
                full_design_final,
                tr=args.tr,
                max_poly_degree=-1,  # block-diagonal nuisance already has per-run constants
                device=device,
                verbose=args.verb >= 1,
                task_indices=task_indices_final,
            )
            final_betas = glm_results_final.betas
        else:
            # Per-voxel HRF path: group by HRF index
            print(f"  Fitting {len(unique_hrfs)} HRF groups...")
            final_betas = torch.zeros(n_voxels_st, n_trials, device=data.device)
            task_indices_final = list(range(n_trials))
            for i_hrf, hrf_idx in enumerate(
                tqdm(unique_hrfs, desc="  HRF groups", disable=not args.verb >= 1)
            ):
                voxel_mask = hrf_indices_dev == hrf_idx
                group_design_2d = st_design[hrf_idx]  # (n_tp, n_trials)
                full_design_final = torch.cat([group_design_2d, nuisance_design_final], dim=1)

                # Diagnostic for first HRF group
                if i_hrf == 0:
                    cond_num = torch.linalg.cond(full_design_final).item()
                    print(f"  Design: {full_design_final.shape}, cond#={cond_num:.1f}")
                    # Per-run nuisance shape (verify polynomials present)
                    print(
                        f"  Nuisance per-run shape: {nuisance_per_run[0].shape} "
                        f"(block-diag total: {nuisance_design_final.shape})"
                    )
                    # Check per-run design energy
                    for run_idx in range(n_runs):
                        start_tp = run_starts[run_idx]
                        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                        run_trial_mask = trial_run_ids == run_idx
                        run_design = group_design_2d[start_tp:end_tp, run_trial_mask]
                        design_energy = run_design.abs().sum().item()
                        n_nonzero = (run_design.abs() > 1e-6).sum().item()
                        data_run = data[voxel_mask][:, start_tp:end_tp]
                        print(
                            f"    Run {run_idx + 1}: trials={run_trial_mask.sum()}, "
                            f"design energy={design_energy:.1f}, nonzero={n_nonzero}, "
                            f"data std={data_run.std().item():.2f}"
                        )

                glm_results_final = fit_glm(
                    data[voxel_mask],
                    full_design_final,
                    tr=args.tr,
                    max_poly_degree=-1,  # block-diagonal nuisance already has per-run constants
                    device=device,
                    verbose=False,
                    task_indices=task_indices_final,
                )
                assert glm_results_final.betas is not None  # set by fit_glm above
                final_betas[voxel_mask] = glm_results_final.betas
            assert glm_results_final.r2 is not None  # set by fit_glm above
            print(f"  Complete. Mean R² = {glm_results_final.r2.mean().item():.3f}")

        # Per-run beta diagnostics (detect runs with degenerate betas)
        print()
        print("  Per-run beta diagnostics:")
        for run_idx in range(n_runs):
            run_trial_mask = trial_run_ids == run_idx
            n_run_trials = run_trial_mask.sum().item()
            run_betas = final_betas[:, run_trial_mask]
            run_mean = run_betas.mean().item()
            run_std = run_betas.std().item()
            run_absmax = run_betas.abs().max().item()
            print(
                f"    Run {run_idx + 1}: {n_run_trials} trials, "
                f"mean={run_mean:.4f}, std={run_std:.4f}, |max|={run_absmax:.2f}"
            )

        # Compute final beta-space cross-validation
        # Always compute COD R² for interpretability. If optimization metric
        # differs (e.g. SSE), also compute and save that separately.
        print()
        print("Evaluating with cross-validation...")
        final_xval_cod = compute_xval_r2_single_trials(
            final_betas,
            trial_cond_ids,
            trial_run_ids,
            cv_splits,
            metric="cod",
            device=device,
            verbose=False,
        )
        final_r2_cod = final_xval_cod["r2"]
        assert isinstance(final_r2_cod, torch.Tensor)  # "r2" key is always a tensor

        final_xval_metric = None
        final_r2_metric: torch.Tensor | None = None
        if args.cv_metric != "cod":
            final_xval_metric = compute_xval_r2_single_trials(
                final_betas,
                trial_cond_ids,
                trial_run_ids,
                cv_splits,
                metric=args.cv_metric,
                device=device,
                verbose=False,
            )
            final_r2_metric = final_xval_metric["r2"]
            assert isinstance(final_r2_metric, torch.Tensor)

        # Print CV summary
        n_folds = len(cv_splits)
        n_test_trials = final_r2_cod.numel()
        print(
            f"  Beta-space CV R² (COD): mean={final_r2_cod.mean():.4f}, "
            f"median={final_r2_cod.median():.4f} "
            f"({n_test_trials} trials across {n_folds} folds)"
        )
        if final_r2_metric is not None:
            _ml = args.cv_metric.upper()
            print(
                f"  Beta-space CV {_ml}: mean={final_r2_metric.mean():.4f}, "
                f"median={final_r2_metric.median():.4f}"
            )

        # 8. Save outputs
        print()
        print("Saving single-trial outputs...")

        # Prepare voxel_mask for saving
        voxel_mask = None
        if mask is not None:
            voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

        output_files = save_single_trial_results(
            betas=final_betas,
            xval_r2=final_r2_cod,
            trial_labels=trial_labels,
            trial_condition_ids=trial_cond_ids,
            trial_run_ids=trial_run_ids,
            condition_labels=condition_labels,
            output_prefix=args.prefix,
            volume_shape=volume_shape,
            affine=affine,
            voxel_mask=voxel_mask,
            nifti_header=nifti_header,
        )

        # Also save PC selection curve
        np.save(f"{args.prefix}_pc_selection_curve.npy", r2_by_pc.cpu().numpy())
        output_files["pc_selection_curve"] = f"{args.prefix}_pc_selection_curve.npy"

        # Save diagnostic masks and initial R²
        # These help understand which voxels were used for noise pool vs criteria
        print()
        print("Saving diagnostic masks...")

        # Need to map data voxels back to 3D volume for saving
        if mask_flat is not None:
            mask_flat_np = mask_flat.cpu().numpy() if torch.is_tensor(mask_flat) else mask_flat

            # Helper to map data voxels to full volume
            def map_to_volume(data_1d, mask_1d):
                vol = np.zeros(mask_1d.size, dtype=data_1d.dtype)
                vol[mask_1d] = data_1d
                return vol

            def _to_vol(data_1d):
                return map_to_volume(data_1d, mask_flat_np).reshape(volume_shape)
        else:
            # No mask — data already has all voxels, just reshape
            def _to_vol(data_1d):
                return data_1d.reshape(volume_shape)

        # Save noise pool mask (low R² voxels used for PC extraction)
        noise_pool_vol = _to_vol(noise_pool_mask.cpu().numpy().astype(np.float32))
        save_nifti(
            noise_pool_vol,
            output_path=f"{args.prefix}_noise_pool_mask{_nii_ext}",
            affine=affine,
            header=nifti_header,
        )
        output_files["noise_pool_mask"] = f"{args.prefix}_noise_pool_mask{_nii_ext}"
        print(f"  Saved: noise pool mask ({noise_pool_mask.sum().item():,} voxels)")

        # Save initial criteria mask (voxels above R² threshold in any PC count)
        initial_criteria_vol = _to_vol(criteria_mask.cpu().numpy().astype(np.float32))
        save_nifti(
            initial_criteria_vol,
            output_path=f"{args.prefix}_initial_criteria_mask{_nii_ext}",
            affine=affine,
            header=nifti_header,
        )
        output_files["initial_criteria_mask"] = f"{args.prefix}_initial_criteria_mask{_nii_ext}"
        print(f"  Saved: initial criteria mask ({criteria_mask.sum().item():,} voxels)")

        # Save initial cross-validated R² (before denoising)
        initial_r2_vol = _to_vol(initial_r2.cpu().numpy())
        save_nifti(
            initial_r2_vol,
            output_path=f"{args.prefix}_initial_xval_r2{_nii_ext}",
            affine=affine,
            header=nifti_header,
        )
        output_files["initial_xval_r2"] = f"{args.prefix}_initial_xval_r2{_nii_ext}"
        print("  Saved: initial cross-validated R²")

        # Save optimization metric map if different from COD
        if final_r2_metric is not None:
            _ml = args.cv_metric.lower()
            _metric_vol = _to_vol(final_r2_metric.cpu().numpy())
            _metric_path = f"{args.prefix}_xval_{_ml}{_nii_ext}"
            save_nifti(_metric_vol, output_path=_metric_path, affine=affine, header=nifti_header)
            output_files[f"xval_{_ml}"] = _metric_path
            print(f"  Saved: xval {_ml} map")

        # Save per-run selected PCs as text files (for ridge -denoise consumption)
        for run_idx in range(n_runs):
            pcs = noise_pcs_per_run[run_idx]
            # extract_noise_{pcs,ics}_per_run's return type is a bool-flag-dependent
            # union (list vs. tuple) that ty can't resolve through the branches above.
            pcs_np = pcs.cpu().numpy() if torch.is_tensor(pcs) else pcs
            n_selected = min(optimal_pcs, pcs_np.shape[1])  # ty: ignore[unresolved-attribute]
            selected_pcs = pcs_np[:, :n_selected]
            pc_txt_path = f"{args.prefix}_run{run_idx + 1:02d}_selected_PCs.txt"
            with open(pc_txt_path, "w") as f:
                f.write(f"# Selected noise PCs for run {run_idx + 1}\n")
                f.write(f"# n_components: {n_selected}\n")
                f.write(
                    f"# Shape: {selected_pcs.shape[0]} timepoints x {selected_pcs.shape[1]} PCs\n"
                )
                np.savetxt(f, selected_pcs, fmt="%.6f", delimiter="\t")
            output_files[f"run{run_idx + 1}_selected_pcs_txt"] = pc_txt_path
        print(f"  Saved: per-run selected PCs ({optimal_pcs} PCs) for {n_runs} runs")

        # Save metadata
        metadata = {
            "optimal_pcs": int(optimal_pcs),
            "r2_threshold": args.r2_threshold,
            "n_noise_voxels": int(n_noise),
            "noise_fraction": float(noise_fraction),
            "pc_selection_curve": r2_by_pc.cpu().tolist(),
            "final_median_r2": float(final_r2_cod.median()),
            "cv_metric": args.cv_metric,
            "noise_method": args.noise,
            "auto_component_caps": bool(args.auto_component_caps),
            "auto_component_estimate_max": int(estimate_max_comps)
            if args.auto_component_caps
            else None,
            "extraction_max_components": int(extraction_max_comps),
            "auto_component_caps_per_run": component_caps_per_run,
            "auto_component_variance_caps": cap_info.variance_caps
            if args.auto_component_caps
            else None,
            "auto_component_effective_rank_caps": cap_info.entropy_rank_caps
            if args.auto_component_caps
            else None,
            "auto_component_mp_caps": cap_info.mp_caps if args.auto_component_caps else None,
            "auto_component_mp_reasons": cap_info.mp_reasons if args.auto_component_caps else None,
            "auto_component_search_iterations": cap_info.search_iterations
            if args.auto_component_caps
            else None,
            "auto_component_search_final_max_per_run": cap_info.search_final_max_per_run
            if args.auto_component_caps
            else None,
            "auto_component_search_ceiling_per_run": cap_info.search_ceiling_per_run
            if args.auto_component_caps
            else None,
            "auto_component_min": int(args.auto_component_min),
            "auto_component_var_threshold": float(args.auto_component_var_threshold),
            "auto_component_use_mp": bool(not args.auto_component_no_mp),
            "component_map_space": args.component_map_space,
            "ic_variance_ratio_per_run": [
                v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
                for v in ic_variance_ratio_per_run
            ]
            if ic_variance_ratio_per_run is not None
            else None,
            "noise_pool_pca_scree_ratio_per_run": [
                v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
                for v in noise_pool_scree_ratio_per_run
            ]
            if noise_pool_scree_ratio_per_run is not None
            else None,
        }
        with open(f"{args.prefix}_denoise_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        output_files["denoise_metadata"] = f"{args.prefix}_denoise_metadata.json"

        if args.scree_plot and noise_pool_scree_ratio_per_run is not None:
            try:
                import matplotlib

                from fastfuncstuff.visualization import plot_noise_pool_pca_scree

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                figs_dir = f"{args.prefix}_figures"
                Path(figs_dir).mkdir(parents=True, exist_ok=True)
                scree_fig = plot_noise_pool_pca_scree(
                    scree_ratio_per_run=noise_pool_scree_ratio_per_run,
                    variance_threshold=args.auto_component_var_threshold,
                    output_path=f"{figs_dir}/noise_pool_pca_scree.png",
                )
                output_files["noise_pool_pca_scree_plot"] = f"{figs_dir}/noise_pool_pca_scree.png"
                plt.close(scree_fig)
                print(f"  Saved: {output_files['noise_pool_pca_scree_plot']}")
            except Exception as e:
                print(f"  Warning: Failed to save noise-pool PCA scree plot: {e}")

        # 9. Generate diagnostic plots (if requested)
        if args.plots in ["yes", "full"]:
            try:
                import matplotlib

                from fastfuncstuff.visualization import (
                    plot_denoising_pcs,
                    plot_denoising_summary,
                )

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                print()
                print("Generating diagnostic plots...")

                # Create figures directory
                figs_dir = f"{args.prefix}_figures"
                Path(figs_dir).mkdir(parents=True, exist_ok=True)

                # Prepare masks and data for plotting
                # The data has already been masked (e.g., 311,522 voxels from original 870k)
                # For spatial plots, we need to map these voxels back to 3D volume positions.
                mask_mismatch = False
                if mask_flat is not None:
                    mask_flat_np = (
                        mask_flat.cpu().numpy() if torch.is_tensor(mask_flat) else mask_flat
                    )
                    if mask_flat_np.sum() == data.shape[0]:
                        # mask_flat.sum() matches data size - good, can map back to volume
                        voxel_mask_np = mask_flat_np
                    else:
                        # Can't safely map data voxels back to 3D volume
                        print(
                            f"  Warning: Cannot map {data.shape[0]} data voxels to volume positions"
                        )
                        print(
                            f"    mask_flat.sum() = {mask_flat_np.sum()}, will generate timecourse plots only"
                        )
                        voxel_mask_np = None
                        mask_mismatch = True
                else:
                    voxel_mask_np = None

                noise_pool_mask_np = noise_pool_mask.cpu().numpy()

                # Create summary plot (CV R² curve)
                # Note: r2_by_pc is already computed and contains median R² across PC counts
                # For compatibility with plot_denoising_summary, we need to format it correctly
                xval_r2_by_n = r2_by_pc.cpu().numpy()
                # Create a dummy per-fold array (single-trial mode uses beta-space CV)
                xval_r2_per_fold = xval_r2_by_n.reshape(1, -1)

                # Get initial R² distribution (computed during noise pool selection)
                initial_r2_np = initial_r2.cpu().numpy() if initial_r2 is not None else None

                summary_fig = plot_denoising_summary(
                    xval_r2_by_n_components=xval_r2_by_n,
                    xval_r2_per_fold=xval_r2_per_fold,
                    optimal_n_components=optimal_pcs,
                    initial_r2_distribution=initial_r2_np,
                    r2_threshold=args.r2_threshold,
                    n_noise_voxels=int(n_noise),
                    n_criteria_voxels=int(n_criteria),
                    output_path=f"{figs_dir}/denoising_summary.png",
                )
                output_files["denoising_summary_plot"] = f"{figs_dir}/denoising_summary.png"
                plt.close(summary_fig)
                print(f"  Saved: {output_files['denoising_summary_plot']}")

                # Per-PC plots (only for "full" mode)
                if args.plots == "full":
                    # Convert PC tensors to CPU for plotting.
                    # extract_noise_{pcs,ics}_per_run's return type is a bool-flag-dependent
                    # union (list vs. tuple) that ty can't resolve through the branches above.
                    pcs_cpu = [pc.cpu() for pc in noise_pcs_per_run]  # ty: ignore[unresolved-attribute]
                    print(
                        f"  Component diagnostics: method={args.noise.upper()}, map_space={args.component_map_space}"
                    )

                    # Only compute spatial loadings if mask matches data
                    # Otherwise, just plot timecourses without spatial maps
                    if mask_mismatch:
                        # Skip spatial weight computation - just plot timecourses
                        loadings_cpu = None
                        noise_pool_mask_for_plot = None
                    else:
                        if args.component_map_space == "full":
                            # Refit component weights on all data voxels
                            pc_loadings_brain = compute_full_brain_pc_loadings(
                                data=data,
                                noise_pcs_per_run=noise_pcs_per_run,
                                run_starts=run_starts,
                                brain_mask=None,
                                chunk_size=5000,
                                device=None,
                                verbose=args.verb >= 1,
                            )
                            loadings_cpu = [ld.numpy() for ld in pc_loadings_brain]
                            noise_pool_mask_for_plot = None
                        else:
                            loadings_cpu = (
                                [ld.cpu().numpy() for ld in component_loadings_per_run]
                                if component_loadings_per_run is not None
                                else None
                            )
                            noise_pool_mask_for_plot = noise_pool_mask_np

                    # Show all extracted components up to configured cap
                    n_pcs_to_show = args.max_comps

                    plot_denoising_pcs(
                        noise_pcs_per_run=pcs_cpu,
                        run_starts=run_starts,
                        component_variance_ratio_per_run=ic_variance_ratio_per_run,
                        pc_weights_per_run=loadings_cpu,  # None = no spatial maps
                        volume_shape=volume_shape,
                        voxel_mask=voxel_mask_np,
                        noise_pool_mask=noise_pool_mask_for_plot,
                        n_pcs_to_show=n_pcs_to_show,
                        n_slices=5,
                        slice_axis=args.plot_ax,
                        tr=args.tr,
                        optimal_n_pcs=optimal_pcs,
                        output_prefix=f"{args.prefix}_figures/component_diagnostics",
                        voxel_sizes=voxel_sizes,  # Preserve physical voxel shape
                        return_figs=False,
                    )
                    output_files["component_diagnostic_plots"] = (
                        f"{args.prefix}_figures/component_diagnostics_PC*.png"
                    )
                    output_files["pc_diagnostic_plots"] = output_files["component_diagnostic_plots"]
                    print(f"  Saved: {output_files['component_diagnostic_plots']}")

            except ImportError as e:
                print(f"  Warning: Could not import visualization module: {e}")
            except Exception as e:
                print(f"  Warning: Error creating plots: {e}")

        print()
        print("=" * 70)
        print("✅ 3dDenoisefast (single-trial mode) Complete!")
        print("=" * 70)
        print(f"  Optimal PCs: {optimal_pcs}")
        print(f"  Final median beta-space R²: {final_r2_cod.median():.4f}")
        print()
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        print("=" * 70)

        return  # Exit early - don't run standard pipeline

    elif designs_by_hrf is not None:
        # Per-voxel HRF mode: pass designs_by_hrf + hrf_indices
        # fit_denoising_model handles the per-HRF logic internally
        print()
        print(f"Fitting denoising model with per-voxel HRFs ({len(designs_by_hrf)} unique HRFs)...")

        results = fit_denoising_model(
            data=data,
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
            run_starts=run_starts,
            tr=args.tr,
            r2_threshold=args.r2_threshold,
            intensity_mask=brainthresh_mask,
            max_components=args.max_comps,
            variance_threshold=args.variance_threshold,
            nuisance=nuisance_per_run,
            polort=args.polort,
            min_noise_voxels=args.min_noise_voxels,
            max_noise_fraction=args.max_noise_fraction,
            pcstop=args.pcstop,
            pcR2cutoff=args.pcR2cutoff,
            noise_method=args.noise,
            auto_component_caps=args.auto_component_caps,
            auto_component_estimate_max=args.auto_component_estimate_max,
            auto_component_min=args.auto_component_min,
            auto_component_var_threshold=args.auto_component_var_threshold,
            auto_component_use_mp=(not args.auto_component_no_mp),
            ica_restarts=args.ica_restarts,
            ica_max_iter=args.ica_max_iter,
            ica_tol=args.ica_tol,
            compute_noise_pool_pca_scree=args.scree_plot,
            scree_max_components=args.scree_max_comps,
            cv_strategy=cv_strategy,
            n_perms=args.n_perms,
            r2_method=args.R2method,
            chunk_size=chunk_size,
            preload_data_to_device=not keep_on_cpu,
            return_loadings=(args.save_pcs in ["spatial", "both"] or args.plots == "full"),
            device=device,
            verbose=args.verb >= 1,
        )

    else:
        # Single HRF for all voxels (standard pipeline)
        results = fit_denoising_model(
            data=data,
            design_matrix=task_design,
            run_starts=run_starts,
            tr=args.tr,
            r2_threshold=args.r2_threshold,
            intensity_mask=brainthresh_mask,
            max_components=args.max_comps,
            variance_threshold=args.variance_threshold,
            nuisance=nuisance_per_run,
            polort=args.polort,
            min_noise_voxels=args.min_noise_voxels,
            max_noise_fraction=args.max_noise_fraction,
            pcstop=args.pcstop,
            pcR2cutoff=args.pcR2cutoff,
            noise_method=args.noise,
            auto_component_caps=args.auto_component_caps,
            auto_component_estimate_max=args.auto_component_estimate_max,
            auto_component_min=args.auto_component_min,
            auto_component_var_threshold=args.auto_component_var_threshold,
            auto_component_use_mp=(not args.auto_component_no_mp),
            ica_restarts=args.ica_restarts,
            ica_max_iter=args.ica_max_iter,
            ica_tol=args.ica_tol,
            compute_noise_pool_pca_scree=args.scree_plot,
            scree_max_components=args.scree_max_comps,
            cv_strategy=cv_strategy,
            n_perms=args.n_perms,
            r2_method=args.R2method,
            chunk_size=chunk_size,
            preload_data_to_device=not keep_on_cpu,
            return_loadings=(args.save_pcs in ["spatial", "both"] or args.plots == "full"),
            device=device,
            verbose=args.verb >= 1,
        )

    # ==========================================================================
    # Save outputs
    # ==========================================================================

    print()
    print("Saving outputs...")

    output_dir = Path(args.prefix).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    voxel_mask = None
    if mask is not None:
        voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

    with spinner("Writing denoising results"):
        output_files = save_denoising_results(
            results=results,
            output_prefix=args.prefix,
            volume_shape=volume_shape,
            affine=affine,
            run_starts=run_starts,
            tr=args.tr,
            data_for_component_maps=data,
            voxel_mask=voxel_mask,
            plots_mode=args.plots,
            slice_axis=args.plot_ax,
            component_map_space=args.component_map_space,
            noise_method=args.noise,
            save_pcs_mode=args.save_pcs,
            condition_labels=condition_labels,
            save_scree_plot=args.scree_plot,
            nii_ext=_nii_ext,
            nifti_header=nifti_header,
        )

    if design_plot_path is not None:
        output_files["final_design_matrix_plot"] = design_plot_path

    # ==========================================================================
    # Save initial and final model fits (if requested or needed for SNR)
    # ==========================================================================

    # We need model fits if either save_model_fit or snr is requested
    need_model_fits = args.save_model_fit or args.snr

    # Clear GPU memory after saving denoising results
    # The results object and plotting may have left tensors on GPU
    if device.type == "cuda":
        # Move results tensors to CPU to free GPU memory
        if hasattr(results, "noise_pcs_per_run") and results.noise_pcs_per_run is not None:
            results.noise_pcs_per_run = [
                pc.cpu() if torch.is_tensor(pc) else pc for pc in results.noise_pcs_per_run
            ]
        if hasattr(results, "pc_loadings_per_run") and results.pc_loadings_per_run is not None:
            results.pc_loadings_per_run = [
                ld.cpu() if torch.is_tensor(ld) else ld for ld in results.pc_loadings_per_run
            ]

        # CRITICAL: Move main data tensor to CPU if it's on GPU
        # This frees up the largest allocation before model fitting
        if torch.is_tensor(data) and data.device.type == "cuda":
            data = data.cpu()
            if args.verb >= 1:
                print("  Moved data tensor to CPU to free GPU memory")

        torch.cuda.empty_cache()
        if args.verb >= 1:
            print("  Cleared GPU cache before model fitting")

    initial_results = None
    final_results = None
    initial_bootstrap_se = None
    final_bootstrap_se = None

    # Per-HRF mode: Skip model fitting for now (requires per-HRF GLM fitting)
    if designs_by_hrf is not None:
        print()
        print("⚠️  Skipping model fitting in per-HRF mode (not yet implemented)")
        print("   Denoising outputs (PCs, masks, R² maps) have been saved.")
        need_model_fits = False

    if need_model_fits:
        print()
        print("Fitting initial model (no denoising)...")

        from fastfuncstuff.glm.core import fit_glm

        # need_model_fits is forced False above when designs_by_hrf is not None,
        # so task_design (the mutually-exclusive alternative) is set here.
        assert task_design is not None

        # Build zero-padded nuisance for concatenated fit
        n_total_timepoints = data.shape[1]
        nuisance_padded_list = []
        current_tp = 0
        for run_nuisance in nuisance_per_run:
            run_length = run_nuisance.shape[0]
            n_cols = run_nuisance.shape[1]
            padded = torch.zeros((n_total_timepoints, n_cols), device=device)
            padded[current_tp : current_tp + run_length, :] = run_nuisance
            nuisance_padded_list.append(padded)
            current_tp += run_length
        nuisance_concat = torch.cat(nuisance_padded_list, dim=1)

        # Full design for initial fit
        full_design_initial = torch.cat([task_design, nuisance_concat], dim=1)
        n_task_cols = task_design.shape[1]
        n_total_regs_initial = full_design_initial.shape[1]

        initial_results = fit_glm(
            data=data,
            design=task_design,
            tr=args.tr,
            extra_regressors=nuisance_concat,
            want_residuals=True,  # Need residuals for SNR
            chunk_size=chunk_size,  # CPU: all voxels, GPU: auto-detect
            preload_data_to_device=False,  # ALWAYS stream from CPU for model fits (safer)
            device=device,
            verbose=False,
        )

        # Bootstrap SE for initial model
        if args.numboots > 0:
            print(f"  Computing bootstrap SE ({args.numboots} iterations)...")
            initial_bootstrap_se = compute_bootstrap_se(
                data=data,
                design=full_design_initial,
                n_task=n_task_cols,
                n_boots=args.numboots,
                device=device,
                verbose=args.verb >= 1,
            )

        if args.save_model_fit:
            with spinner("Writing initial model-fit outputs"):
                initial_files = save_model_fit_outputs(
                    results=initial_results,
                    output_prefix=args.prefix,
                    volume_shape=volume_shape,
                    affine=affine,
                    model_type="initial",
                    condition_labels=condition_labels,
                    voxel_mask=voxel_mask,
                    n_timepoints=n_timepoints,
                    n_regressors=n_total_regs_initial,
                    bootstrap_se=initial_bootstrap_se,
                    nii_ext=_nii_ext,
                    nifti_header=nifti_header,
                )
            output_files.update(initial_files)

        # Final fit (with optimal denoising)
        print(f"Fitting final model (with {results.optimal_n_components} noise PCs)...")

        # Build combined nuisance with noise PCs for final fit
        n_pcs_optimal = results.optimal_n_components
        if n_pcs_optimal > 0:
            pc_padded_blocks = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = end_tp - start_tp
                pcs_run = results.noise_pcs_per_run[run_idx][:, :n_pcs_optimal]

                padded = torch.zeros((n_timepoints, n_runs * n_pcs_optimal), device=device)
                start_col = run_idx * n_pcs_optimal
                end_col = start_col + n_pcs_optimal
                padded[start_tp:end_tp, start_col:end_col] = pcs_run

                pc_padded_blocks.append(padded)

            pc_concat = sum(pc_padded_blocks)

            if nuisance_concat is not None:
                nuisance_with_pcs = torch.cat([nuisance_concat, pc_concat], dim=1)
            else:
                nuisance_with_pcs = pc_concat
        else:
            nuisance_with_pcs = nuisance_concat

        full_design_final = torch.cat([task_design, nuisance_with_pcs], dim=1)
        n_total_regs_final = full_design_final.shape[1]

        final_results = fit_glm(
            data=data,
            design=task_design,
            tr=args.tr,
            extra_regressors=nuisance_with_pcs,
            want_residuals=True,  # Need residuals for SNR
            chunk_size=chunk_size,  # CPU: all voxels, GPU: auto-detect
            preload_data_to_device=False,  # ALWAYS stream from CPU for model fits (safer)
            device=device,
            verbose=False,
        )

        # Bootstrap SE for final model
        if args.numboots > 0:
            print(f"  Computing bootstrap SE for denoised model ({args.numboots} iterations)...")
            final_bootstrap_se = compute_bootstrap_se(
                data=data,
                design=full_design_final,
                n_task=n_task_cols,
                n_boots=args.numboots,
                device=device,
                verbose=args.verb >= 1,
            )

        if args.save_model_fit:
            with spinner("Writing final model-fit outputs"):
                final_files = save_model_fit_outputs(
                    results=final_results,
                    output_prefix=args.prefix,
                    volume_shape=volume_shape,
                    affine=affine,
                    model_type="denoised",
                    condition_labels=condition_labels,
                    voxel_mask=voxel_mask,
                    n_timepoints=n_timepoints,
                    n_regressors=n_total_regs_final,
                    bootstrap_se=final_bootstrap_se,
                    nifti_header=nifti_header,
                    nii_ext=_nii_ext,
                )
            output_files.update(final_files)

    # ==========================================================================
    # Compute and save SNR (if requested)
    # ==========================================================================

    if args.snr and initial_results is not None and final_results is not None:
        print()
        print("Computing SNR metrics...")

        # Get betas and residuals
        initial_betas = initial_results.betas[:, :n_conditions].cpu().numpy()
        final_betas = final_results.betas[:, :n_conditions].cpu().numpy()

        # Residual std from MSE
        initial_residual_std = (
            torch.sqrt(initial_results.mse).cpu().numpy()
            if hasattr(initial_results, "mse") and initial_results.mse is not None
            else None
        )
        final_residual_std = (
            torch.sqrt(final_results.mse).cpu().numpy()
            if hasattr(final_results, "mse") and final_results.mse is not None
            else None
        )

        # If MSE not available, compute from residuals
        if initial_residual_std is None and initial_results.residuals is not None:
            initial_residual_std = initial_results.residuals.std(dim=1).cpu().numpy()
        if final_residual_std is None and final_results.residuals is not None:
            final_residual_std = final_results.residuals.std(dim=1).cpu().numpy()

        # Compute SNR
        snr_initial = compute_snr(
            betas=initial_betas,
            residual_std=initial_residual_std,
            bootstrap_se=initial_bootstrap_se,
        )
        snr_denoised = compute_snr(
            betas=final_betas,
            residual_std=final_residual_std,
            bootstrap_se=final_bootstrap_se,
        )

        # Report improvement
        if "snr_residual" in snr_initial and "snr_residual" in snr_denoised:
            mean_initial = snr_initial["snr_residual"].mean()
            mean_denoised = snr_denoised["snr_residual"].mean()
            print(
                f"  Residual-based SNR: {mean_initial:.2f} → {mean_denoised:.2f} ({(mean_denoised / mean_initial - 1) * 100:+.1f}%)"
            )

        if "snr_bootstrap" in snr_initial and "snr_bootstrap" in snr_denoised:
            mean_initial = snr_initial["snr_bootstrap"].mean()
            mean_denoised = snr_denoised["snr_bootstrap"].mean()
            print(
                f"  Bootstrap-based SNR: {mean_initial:.2f} → {mean_denoised:.2f} ({(mean_denoised / mean_initial - 1) * 100:+.1f}%)"
            )

        # Save SNR outputs
        with spinner("Writing SNR outputs"):
            snr_files = save_snr_outputs(
                snr_initial=snr_initial,
                snr_denoised=snr_denoised,
                output_prefix=args.prefix,
                volume_shape=volume_shape,
                affine=affine,
                voxel_mask=voxel_mask,
                create_plots=True,
                nii_ext=_nii_ext,
                nifti_header=nifti_header,
            )
        output_files.update(snr_files)

    print()
    print("=" * 70)
    print("📁 Output Files")
    print("=" * 70)
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")
    print("=" * 70)

    # Print summary
    print()
    print("=" * 70)
    print("✅ 3dDenoisefast Complete!")
    print("=" * 70)
    print(f"  Noise pool: {results.metadata['n_noise_voxels']:,} voxels")
    print(f"  Criteria: {results.metadata['n_criteria_voxels']:,} voxels")
    print(f"  Baseline R²: {results.baseline_r2:.4f}")
    print(f"  Optimal R²: {results.optimal_r2:.4f}")
    print(f"  Improvement: {results.improvement:+.4f}")
    print(f"  Optimal PCs: {results.optimal_n_components}")
    print()
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
