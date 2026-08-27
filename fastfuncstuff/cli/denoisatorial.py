#!/usr/bin/env python3
"""
ffs_denoisatorial - Combinatorial PC denoising via exhaustive subset evaluation

Instead of testing prefix subsets {0}, {0,1}, {0,1,2}, ... like 3dDenoisefast,
this tool tests ALL 2^k subsets of noise PCs to find the optimal per-run
combination. This discovers non-contiguous optimal subsets (e.g., PCs 0, 3, 5).

Algorithm:
  For each held-out run (outer LORO):
    1. Fit OLS betas on N-1 training runs
    2. Inner LORO CV on training runs -> criteria voxel pool
    3. Extract k PCs from held-out run's noise pool
    4. Evaluate all 2^k combinations on held-out run
    5. Select optimal combination (argmax median CoD)

Basic usage:
    ffs_denoisatorial -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                      -onsets cond1.txt cond2.txt \\
                      -durations 2.0 5.0 \\
                      -tr 2.0 \\
                      -prefix subject01_combinatorial

    ffs_denoisatorial -input run*.nii.gz \\
                      -events sub-01_task-loc_run-*_events.tsv \\
                      -tr 2.0 \\
                      -prefix subject01_combinatorial

For help:
    ffs_denoisatorial -help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib  # noqa: F401
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    from fastfuncstuff.cli_utils import (
        LoadResult,
        ScannableHelpFormatter,
        add_device_arg,
        add_load_threads_arg,
        add_noise_ceiling_args,
        add_ortvec_arguments,
        add_trim_args,
        add_verbose_arg,
        append_nuisance_blocks,
        apply_trim_to_timing,
        auto_polort,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
        parse_input_files,
        parse_prefix,
        print_cli_header,
        resolve_microtime_dt,
        run_lengths_from_starts,
        setup_device,
        trim_spec_from_args,
    )
    from fastfuncstuff.denoise.combinatorial import (
        CombinatorialDenoiseResults,
        compute_initial_xval_r2,
        compute_optimized_xval_r2_3dDenoise_style,
        fit_combinatorial_denoising,
        plot_combinatorial_results,
        plot_inclusion_heatmap,
        plot_plateau_curves,
        plot_singleton_contributions,
    )
    from fastfuncstuff.denoise.sequential import select_noise_pool_voxels
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.design.hrf_selection import load_nuisance_file  # noqa: F401
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.glm.ridge import load_hrf_indices
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.utils import (
        scale_to_percent_signal,
        to_tensor,  # noqa: F401
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


# ============================================================================
# Argument parser
# ============================================================================


class _HelpFormatter(ScannableHelpFormatter):
    """The shared scannable formatter; kept as a local name for this module's parser."""


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="ffs_denoisatorial - Combinatorial PC Denoising",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic combinatorial denoising (7 PCs, 128 combinations)
  ffs_denoisatorial -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                    -onsets cond1.txt cond2.txt \\
                    -durations 2.0 \\
                    -tr 2.0 \\
                    -prefix subject01_combinatorial

  # Fewer PCs for faster evaluation
  ffs_denoisatorial -input run*.nii.gz \\
                    -onsets face.txt house.txt \\
                    -durations 2.0 \\
                    -tr 2.0 \\
                    -max_pcs 5 \\
                    -prefix sub01_combo5

  # BIDS events TSVs (one per run) instead of AFNI timing files
  ffs_denoisatorial -input run*.nii.gz \\
                    -events sub-01_task-loc_run-*_events.tsv \\
                    -event_ignore fixation \\
                    -tr 2.0 \\
                    -prefix sub01_bids

  # One shared events TSV, broadcast to every run (identical timing per run)
  ffs_denoisatorial -input run*.nii.gz \\
                    -events task-loc_events.tsv \\
                    -tr 2.0 \\
                    -prefix sub01_shared_timing

  # With diagnostic plots
  ffs_denoisatorial -input run*.nii.gz \\
                    -onsets stim.txt \\
                    -durations 1.0 \\
                    -tr 2.0 \\
                    -plots full \\
                    -prefix sub01_full_diagnostics

  # Recommended for most datasets: score each PC on its own (fast, k+1 candidates
  # instead of 2^k) and make it beat phase-randomised surrogates of itself before
  # it is kept. Without -null_surrogates the rule is a bare "delta > 0", which has
  # no magnitude floor and keeps about half of any set of useless PCs.
  ffs_denoisatorial -input run*.nii.gz \\
                    -events events.tsv \\
                    -tr 2.0 \\
                    -singleton_only \\
                    -null_surrogates 20 \\
                    -prefix sub01_singleton_null

Outputs:
    Core outputs:
        {prefix}_initial_r2.nii.gz               - Initial xval R2 (task-only, no PCs).
                                                   With -noise_ceiling, a 3-volume stack:
                                                   initial_R2, noise_ceiling, explainable_R2.
        {prefix}_optimized_xval_r2.nii.gz         - Xval R2 with optimal per-run PCs. With
                                                   -noise_ceiling, the same 3-volume stack,
                                                   built at the SELECTED PC combination.
                                                   explainable_R2 (R2/ceiling) is the number
                                                   to compare across tools and designs.
        {prefix}_noise_pool_mask.nii.gz           - Noise pool voxels
        {prefix}_run{NN}_optimal_pcs.json         - Optimal PC indices per run
        {prefix}_run{NN}_selected_PCs.txt         - Selected PC timecourses per run
        {prefix}_combinatorial_results.pt         - Full results (PyTorch)
        {prefix}_metadata.json                    - Reproducibility metadata

    With -plots yes/full:
        {prefix}_figures/combinatorial_scatter.png  - Per-run CoD vs variance plots
        {prefix}_figures/combinatorial_heatmap.png  - PC selection heatmap

Workflow:
  1. Compute initial task-only cross-validated R2
  2. Select noise pool (low R2) and criteria voxels (high R2)
  3. For each held-out run:
     a. Fit betas on training runs
     b. Inner CV to refine criteria pool
     c. Extract PCs from held-out run
     d. Evaluate all 2^k PC combinations
     e. Select optimal combination
  4. Compute final cross-validated R2 with optimal PCs
  5. Save results and plots

Notes:
  - At least 3 runs required (outer LORO + inner LORO needs >=2 training runs)
  - Default max_pcs=7 gives 128 combinations (GPU-friendly)
  - max_pcs=10 gives 1024 combinations (still fast on GPU)
  - max_pcs>12 not recommended (4096+ combinations, memory-heavy)
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
        help="Onset timing files (AFNI format). One file per condition. "
        "Mutually exclusive with -events.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        help="Stimulus durations in seconds. Either single value or one per condition. "
        "Required with -onsets.",
    )
    required.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files, one per run, or a single shared TSV broadcast "
        "across all runs. Files are sorted by run number (run-1 and run-01 both work). "
        "Conditions and durations are derived from the TSV. "
        "Mutually exclusive with -onsets/-durations.",
    )
    # Both hyphen and underscore forms, matching the other ffs_* GLM tools.
    required.add_argument(
        "-event-ignore",
        "-event_ignore",
        dest="event_ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to drop from BIDS events (e.g. fixation null rest).",
    )
    required.add_argument(
        "-event-cols",
        "-event_cols",
        dest="event_cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom column names for -events TSVs, replacing the BIDS defaults "
        "(onset, duration, trial_type).",
    )
    required.add_argument(
        "-round-durations",
        "-round_durations",
        dest="round_durations",
        type=int,
        default=None,
        metavar="PLACES",
        help="Round stimulus durations to PLACES decimal places before uniquing "
        "(0=integer, 1=tenth). Keeps 3.03 and 3.0 from becoming distinct durations.",
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

    # Combinatorial options
    combo_opts = parser.add_argument_group("Combinatorial Options")
    combo_opts.add_argument(
        "-max_pcs",
        type=int,
        default=7,
        help="Number of PCs to extract per run. 2^max_pcs combinations evaluated. "
        "(default: 7, giving 128 combinations)",
    )
    combo_opts.add_argument(
        "-r2_threshold",
        type=float,
        default=0.05,
        help="R2 threshold for noise pool selection (default: 0.05). "
        "Voxels with R2 < threshold are noise pool.",
    )
    combo_opts.add_argument(
        "-criteria_r2_threshold",
        "-criteria-r2-threshold",
        dest="criteria_r2_threshold",
        type=str,
        default="0.05",
        metavar="SPEC",
        help="Which voxels the CoD is medianed over — the responsive ones. "
        "A float is an absolute inner-CV R2 threshold (0.05); 'N%%' takes the top "
        "N percent by inner-CV R2 ('5%%'); '(N)' takes the top N voxels ('(1000)'). "
        "An absolute threshold that yields under 100 voxels falls back to "
        "-criteria_fallback_pct. Too permissive is the dangerous direction: at 0.0 "
        "the median lands on an unresponsive voxel and PC deltas collapse to noise.",
    )
    combo_opts.add_argument(
        "-criteria_fallback_pct",
        "-criteria-fallback-pct",
        dest="criteria_fallback_pct",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Top-percentile fallback when an absolute -criteria_r2_threshold "
        "selects too few voxels (default: 5.0).",
    )
    combo_opts.add_argument(
        "-selection_strategy",
        type=str,
        choices=["argmax", "parsimonious"],
        default="argmax",
        help="Strategy for selecting optimal combination: "
        "'argmax' (highest CoD, default), "
        "'parsimonious' (fewest PCs within 1%% of max). "
        "Ignored if -singleton_only is set.",
    )
    combo_opts.add_argument(
        "-singleton_only",
        action="store_true",
        help="Singleton-only mode: evaluate only individual PCs, not all combinations. "
        "Selects all PCs with positive delta vs baseline. "
        "Much faster (k+1 combos instead of 2^k).",
    )
    combo_opts.add_argument(
        "-criterion",
        choices=["within_run", "cross_run"],
        default=None,  # resolved to cross_run below; None only marks "not given"
        help="How a candidate PC set is scored.\n"
        "  cross_run   (default) Remove the run's PCs while that run"
        " contributes to the betas, then score those betas on the OTHER runs, which are never"
        " cleaned. This is how the denoising is actually used. Because the scored target never"
        " changes with the candidate, SS_tot is fixed and a candidate can only win by producing"
        " better betas. It also puts N-1 runs of evidence behind each decision instead of one.\n"
        "  within_run  Hold the training betas fixed and ask how well they explain what is LEFT"
        " of the held-out run after removal. Re-derives SS_tot from the cleaned data, so CoD"
        " rises mechanically whenever residual variance is removed — and a noise PC is by"
        " construction a direction the design does not explain, so it takes disproportionately"
        " from the residual. Measured at +0.0078 vs +0.0001 for the same PCs on the same data."
        " Kept for comparison; not a default.",
    )
    combo_opts.add_argument(
        "-whole_brain_noise_pool",
        "-whole-brain-noise-pool",
        dest="whole_brain_noise_pool",
        action="store_true",
        help="Extract noise PCs from every in-brain voxel instead of the low-R2 "
        "noise pool. Task-dominated components are not a safety problem — removing "
        "one strips the variance the betas exist to predict, so the criterion "
        "rejects it — but they do crowd the top of the variance ordering, so raise "
        "-max_pcs alongside this. The criteria pool is unaffected, and a -compare "
        "GLMdenoise baseline keeps its own noise-pool PCs so the comparison stays "
        "fair.",
    )
    combo_opts.add_argument(
        "-null_surrogates",
        "-null-surrogates",
        dest="null_surrogates",
        type=int,
        default=0,
        metavar="N",
        help="Calibrate singleton selection against N phase-randomised surrogates per PC."
        " 0 = off, 20 is the recommended setting and worth turning on.\n"
        "Without it, selection is a bare 'delta > 0' sign test with no magnitude floor, so"
        " under a true null each PC clears about half the time and you keep roughly half of"
        " them for nothing, paying their degrees of freedom on unseen data. Each surrogate"
        " keeps its PC's variance and spectrum but carries no real structure, so a PC has to"
        " beat its own surrogates to survive.\n"
        "It is off by default only because it changes which PCs are selected; it is not"
        " expensive — the scorer batches every candidate, so 20 surrogates cost far less than"
        " the name suggests. Requires -singleton_only.",
    )
    combo_opts.add_argument(
        "-null_percentile",
        "-null-percentile",
        dest="null_percentile",
        type=float,
        default=95.0,
        metavar="P",
        help="Percentile of its own surrogate deltas a PC must beat (default: 95).",
    )
    combo_opts.add_argument(
        "-null_seed",
        "-null-seed",
        dest="null_seed",
        type=int,
        default=0,
        help="Seed for surrogate generation, so a rerun reproduces the selection (default: 0).",
    )
    combo_opts.add_argument(
        "-compare",
        action="store_true",
        help="Also run a standard GLMdenoise-style baseline (incremental PCs in "
        "variance order, stop when adding the next PC gains less than 5%% in "
        "median R² — i.e. pcstop=1.05). Reports R² boost difference between the "
        "combinatorial/singleton selection and the GLMdenoise selection, and "
        "writes {prefix}_r2_glmdenoise and {prefix}_r2_delta NIfTIs alongside "
        "the usual outputs. Useful for quantifying whether the combinatorial "
        "approach is actually buying you anything on your data.",
    )
    heldout_opts = parser.add_argument_group("Held-out validation (optional)")
    heldout_opts.add_argument(
        "-test_input",
        "-test-input",
        dest="test_input",
        nargs="+",
        default=None,
        metavar="DSET",
        help="Completely held-out run(s), never used to pick PCs, the noise pool or "
        "anything else. After denoising finishes, task betas are fit on ALL input "
        "runs with the winning denoising in place (per-run polynomials + that "
        "run's selected PCs), and those betas times the held-out design predict "
        "these runs, whose own polynomials (plus any -test_ortvec) are removed so "
        "the prediction is valid. "
        "No PCs are removed from the held-out data — what is being tested is "
        "whether denoising produced better betas. Writes {prefix}_heldout_r2 and "
        "{prefix}_heldout_initial_r2 (same fit with no PCs, as the reference).",
    )
    heldout_opts.add_argument(
        "-test_events",
        "-test-events",
        dest="test_events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS events TSV(s) for -test_input: one per held-out run, or a single "
        "shared TSV broadcast across them. Conditions must match the input runs' "
        "conditions exactly (same labels, same order). Required with -test_input.",
    )
    heldout_opts.add_argument(
        "-test_curve",
        "-test-curve",
        dest="test_curve",
        action="store_true",
        help="Learning curve on the held-out runs: refit the betas using k of the "
        "input runs for k=1..N and score each arm's prediction, so the curves can "
        "be compared as a function of how much training data the betas saw. A "
        "denoising that only reduces beta variance is nearly invisible at k=N (the "
        "variance term shrinks like 1/N against a fixed held-out noise floor) but "
        "separates clearly at small k. Writes {prefix}_heldout_curve.png and "
        "{prefix}_heldout_curve.tsv. Requires -test_input. Costs "
        "k x subsets x arms extra GLM fits.",
    )
    heldout_opts.add_argument(
        "-test_curve_subsets",
        "-test-curve-subsets",
        dest="test_curve_subsets",
        type=int,
        default=8,
        metavar="N",
        help="Run subsets sampled per k for -test_curve (default: 8). Exhaustive "
        "when there are fewer than N distinct subsets of that size.",
    )
    # The held-out runs need their own nuisance: the -ortvec files describe the
    # input runs and have the wrong length. Unmodelled held-out motion is common
    # to both scored arms, so it only dilutes the denoising difference — but
    # dilution is exactly what makes the held-out effect hard to read.
    add_ortvec_arguments(heldout_opts, prefix="test_")
    combo_opts.add_argument(
        "-brainthresh",
        nargs=2,
        type=float,
        metavar=("PERCENTILE", "FRACTION"),
        default=None,
        help="Signal intensity threshold for noise pool selection: voxels whose mean "
        "intensity is below percentile(mean, P) * F are excluded from the noise pool. "
        "Defaults to '99 0.5' (the GLMdenoise default) when none of -mask, -automask "
        "or -brainthresh is given, so background voxels can't flood the noise pool. "
        "Pass '-brainthresh 0 0' to disable the cut. Example: -brainthresh 99 0.5",
    )
    combo_opts.add_argument(
        "-min_noise_voxels",
        type=int,
        default=100,
        help="Minimum voxels required in noise pool (default: 100)",
    )
    combo_opts.add_argument(
        "-max_noise_fraction",
        type=float,
        default=0.95,
        help="Maximum fraction of voxels in noise pool (default: 0.95)",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-hrf_opt",
        type=str,
        default=None,
        help="3dHRFoptfast output prefix. Loads {prefix}_hrf_index.nii.gz for "
        "per-voxel HRF optimization. Each voxel uses its assigned HRF for "
        "design construction. Mutually exclusive with -canonical.",
    )
    proc_opts.add_argument(
        "-mask",
        help="Mask file to restrict analysis to brain voxels",
    )
    proc_opts.add_argument(
        "-automask",
        action="store_true",
        help="Compute a brain mask automatically from the mean EPI (AFNI-style "
        "automask with dilate=4) and restrict the analysis to it. Applied before "
        "brainthresh. Mutually exclusive with -mask.",
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
        "SPMG2 = canonical + temporal derivative. SPMG3 = canonical + time + dispersion derivatives. "
        "FIR/TENT windows are estimated from the stimulus durations (stimulus + HRF "
        "tail); override with -fir_duration. Mutually exclusive with -hrf_opt.",
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
        help="DEPRECATED: Use -hrf_model instead.",
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
    add_device_arg(proc_opts)
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process in GPU chunks (for large datasets)",
    )
    add_load_threads_arg(proc_opts)
    add_trim_args(proc_opts)
    add_noise_ceiling_args(
        proc_opts,
        stage_note="Built at the per-run PC sets this tool selected, so the ceiling "
        "and the optimized R2 share a denominator.",
    )
    add_verbose_arg(proc_opts, default=0)
    proc_opts.add_argument(
        "-dry_run",
        action="store_true",
        help="Fast testing mode: load only first run, generate synthetic data for rest. "
        "Results are nonsensical but pipeline runs quickly for testing.",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-plots",
        type=str,
        choices=["no", "yes", "full"],
        default="no",
        help="Save diagnostic plots: 'no' (none), 'yes' (scatter only), 'full' (scatter + heatmap)",
    )
    out_opts.add_argument(
        "-save_pcs",
        type=str,
        choices=["no", "timecourse", "both"],
        default="timecourse",
        help="Save noise PCs: 'no', 'timecourse' (default: selected PCs as txt), 'both' (.pt + txt)",
    )

    return parser


# ============================================================================
# Output saving
# ============================================================================


def _report_learning_curve(
    curve: dict,
    active_mask: torch.Tensor,
    prefix: str,
    plot,
) -> dict[str, str]:
    """Print the learning-curve table, write the TSV and the plot.

    Medians are taken over the active voxels only; the deltas are per-voxel
    against the ``initial`` arm before medianing, which keeps the comparison
    paired (same subsets, same held-out data, same voxels).
    """
    subset_sizes = curve["subset_sizes"]
    curves = curve["curves"]
    mask = active_mask.cpu()
    names = list(curves.keys())

    medians = {n: curves[n][:, mask].median(dim=1).values for n in names}
    deltas = {
        n: (curves[n] - curves["initial"])[:, mask].median(dim=1).values
        for n in names
        if n != "initial"
    }

    header = f"    {'k':>3}  {'subsets':>7}" + "".join(f"  {n[:18]:>18}" for n in names)
    print()
    print("  Median held-out R² over active voxels:")
    print(header)
    for i, k in enumerate(subset_sizes):
        row = f"    {k:>3}  {curve['n_subsets'][i]:>7}"
        row += "".join(f"  {medians[n][i].item():>18.4f}" for n in names)
        print(row)

    if deltas:
        print("  Δ vs no denoising (per-voxel paired, then medianed):")
        print(f"    {'k':>3}" + "".join(f"  {n[:18]:>18}" for n in deltas))
        for i, k in enumerate(subset_sizes):
            print(f"    {k:>3}" + "".join(f"  {d[i].item():>+18.4f}" for d in deltas.values()))

    files: dict[str, str] = {}
    tsv_path = f"{prefix}_heldout_curve.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("k\tn_subsets\t" + "\t".join(names) + "\n")
        for i, k in enumerate(subset_sizes):
            vals = "\t".join(f"{medians[n][i].item():.6f}" for n in names)
            fh.write(f"{k}\t{curve['n_subsets'][i]}\t{vals}\n")
    files["heldout_curve_tsv"] = tsv_path
    print(f"  Saved: {tsv_path}")

    plot_path = f"{prefix}_heldout_curve.png"
    plot(curve=curve, active_mask=active_mask, output_path=plot_path)
    files["heldout_curve_plot"] = plot_path
    print(f"  Saved: {plot_path}")
    return files


def save_combinatorial_results(
    results: CombinatorialDenoiseResults,
    initial_r2_full: torch.Tensor,
    optimized_r2_full: torch.Tensor,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    run_starts: list[int],
    tr: float,
    condition_labels: list[str],
    mask_flat: np.ndarray | None = None,
    plots_mode: str = "no",
    save_pcs_mode: str = "timecourse",
    nii_ext: str = ".nii.gz",
    initial_ceiling_layers: list | None = None,
    optimized_ceiling_layers: list | None = None,
) -> dict:
    """Save combinatorial denoising results to disk."""
    output_files = {}

    # Helper to reshape flat data to volume
    def to_volume(flat_data):
        if torch.is_tensor(flat_data):
            flat_np = flat_data.cpu().numpy()
        else:
            flat_np = flat_data
        flat_np = flat_np.astype(np.float32)

        if mask_flat is not None:
            vol = np.zeros(mask_flat.size, dtype=np.float32)
            vol[mask_flat] = flat_np
        else:
            vol = flat_np
        return vol.reshape(volume_shape)

    # Ensure output directory exists
    prefix_dir = Path(output_prefix).parent
    if prefix_dir != Path("."):
        prefix_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. The two R2 stacks. Each carries the ceiling built at ITS OWN PC set --
    # none for the initial R2, the selected per-run combination for the optimized
    # one -- because a ceiling only bounds an R2 with the same denominator.
    from fastfuncstuff.cli_utils import save_r2_ceiling_stack, spinner

    for key, r2_map, layers in (
        ("initial_r2", initial_r2_full, initial_ceiling_layers or []),
        ("optimized_xval_r2", optimized_r2_full, optimized_ceiling_layers or []),
    ):
        label = "initial_R2" if key == "initial_r2" else "optimized_xval_R2"
        path = f"{output_prefix}_{key}{nii_ext}"
        with spinner(f"Writing {Path(path).name}"):
            save_r2_ceiling_stack([(r2_map, label), *layers], path, volume_shape, affine, mask_flat)
        output_files[key] = path
        print(f"  Saved: {path}")

    # 3. Noise pool mask
    noise_pool_vol = to_volume(results.noise_pool_mask)
    noise_pool_path = f"{output_prefix}_noise_pool_mask{nii_ext}"
    with spinner(f"Writing {Path(noise_pool_path).name}"):
        save_nifti(noise_pool_vol, output_path=noise_pool_path, affine=affine)
    output_files["noise_pool_mask"] = noise_pool_path
    print(f"  Saved: {noise_pool_path}")

    # 4. Per-run optimal PC indices (JSON)
    for run_res in results.per_run_results:
        run_idx = run_res.run_idx
        pc_info = {
            "run_idx": run_idx,
            "optimal_combination": list(run_res.optimal_combination),
            "optimal_cod": float(run_res.optimal_cod),
            "baseline_cod": float(run_res.baseline_cod),
            "n_criteria_voxels": run_res.n_criteria_voxels,
            "explained_variance_ratios": run_res.explained_variance_ratios.tolist(),
        }
        if run_res.pc_status is not None:
            pc_info["pc_status"] = list(run_res.pc_status)
        if run_res.null_thresholds is not None:
            pc_info["null_thresholds"] = run_res.null_thresholds.tolist()
        json_path = f"{output_prefix}_run{run_idx:02d}_optimal_pcs.json"
        with open(json_path, "w") as f:
            json.dump(pc_info, f, indent=2)
        output_files[f"run{run_idx:02d}_optimal_pcs"] = json_path

    print(f"  Saved optimal PC indices for {len(results.per_run_results)} runs")

    # 5. Selected PC timecourses as text files
    if save_pcs_mode in ["timecourse", "both"]:
        for run_res in results.per_run_results:
            run_idx = run_res.run_idx
            pcs = results.noise_pcs_per_run[run_idx]
            if torch.is_tensor(pcs):
                pcs_np = pcs.cpu().numpy()
            else:
                pcs_np = pcs

            selected_idx = list(run_res.optimal_combination)
            if len(selected_idx) > 0:
                selected_pcs = pcs_np[:, selected_idx]
            else:
                selected_pcs = np.zeros((pcs_np.shape[0], 0))

            pc_txt_path = f"{output_prefix}_run{run_idx:02d}_selected_PCs.txt"
            with open(pc_txt_path, "w") as f:
                f.write(f"# Selected noise PCs for run {run_idx}\n")
                f.write(f"# Selected PC indices: {selected_idx}\n")
                f.write(
                    f"# Shape: {selected_pcs.shape[0]} timepoints x {selected_pcs.shape[1]} PCs\n"
                )
                if selected_pcs.shape[1] > 0:
                    np.savetxt(f, selected_pcs, fmt="%.6f", delimiter="\t")
            output_files[f"run{run_idx:02d}_selected_pcs_txt"] = pc_txt_path

        print(f"  Saved selected PC timecourses for {len(results.per_run_results)} runs")

    # 6. Full results as PyTorch file
    if save_pcs_mode == "both":
        results_path = f"{output_prefix}_combinatorial_results.pt"
        with spinner(f"Writing {Path(results_path).name}"):
            torch.save(
                {
                    "noise_pcs_per_run": results.noise_pcs_per_run,
                    "per_run_optimal_combinations": [
                        r.optimal_combination for r in results.per_run_results
                    ],
                    "per_run_all_cod": [r.all_cod for r in results.per_run_results],
                    "per_run_all_var_explained": [
                        r.all_var_explained for r in results.per_run_results
                    ],
                    "per_run_variance_ratios": [
                        r.explained_variance_ratios for r in results.per_run_results
                    ],
                    "metadata": results.metadata,
                },
                results_path,
            )
        output_files["combinatorial_results"] = results_path
        print(f"  Saved: {results_path}")

    # 7. Metadata JSON
    metadata = {
        **results.metadata,
        "per_run_optimal_combinations": {
            f"run{r.run_idx:02d}": list(r.optimal_combination) for r in results.per_run_results
        },
        "per_run_optimal_cod": {
            f"run{r.run_idx:02d}": float(r.optimal_cod) for r in results.per_run_results
        },
        "per_run_baseline_cod": {
            f"run{r.run_idx:02d}": float(r.baseline_cod) for r in results.per_run_results
        },
        "condition_labels": condition_labels,
        "volume_shape": list(volume_shape),
        "tr": tr,
        "run_starts": run_starts,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_path
    print(f"  Saved: {metadata_path}")

    # 8. Plots
    if plots_mode in ["yes", "full"]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_prefix = f"{output_prefix}_figures"
            Path(fig_prefix).mkdir(parents=True, exist_ok=True)

            figs = plot_combinatorial_results(results, f"{fig_prefix}/")

            # Singleton PC contributions: individual effect of each PC
            singleton_figs = plot_singleton_contributions(results, f"{fig_prefix}/")
            figs.extend(singleton_figs)

            # Plateau curves: best achievable CoD with N PCs
            plateau_figs = plot_plateau_curves(results, f"{fig_prefix}/")
            figs.extend(plateau_figs)

            # Inclusion heatmap: delta R² coloring with X marks
            heatmap_figs = plot_inclusion_heatmap(results, f"{fig_prefix}/")
            figs.extend(heatmap_figs)

            for fig in figs:
                plt.close(fig)

            print(f"  Saved plots to {fig_prefix}/")
            output_files["plots_dir"] = fig_prefix
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")

    return output_files


# ============================================================================
# Design / nuisance construction (shared by the input runs and -test_input)
# ============================================================================


def build_task_designs(
    all_onsets: list,
    durations: list[float],
    n_conditions: int,
    run_starts: list[int],
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    hrf_model_name: str,
    is_fir_model: bool,
    fir_bot: float | None,
    fir_top: float | None,
    n_basis: int,
    device: torch.device,
    hrf_opt: str | None = None,
    hrf_library: torch.Tensor | None = None,
    hrf_indices: torch.Tensor | None = None,
    n_voxels: int | None = None,
) -> tuple[torch.Tensor | None, dict | None]:
    """Build the task design (or per-HRF designs) for one set of runs.

    Held-out evaluation has to build its design exactly the way the input runs
    did — same HRF, same basis count, same microtime grid — so this lives in
    one function rather than inline in ``main``.
    """
    from fastfuncstuff.cli_utils import build_task_design_from_args

    bins_per_tr = int(np.round(tr / microtime_dt))
    n_microtime = n_timepoints * bins_per_tr
    onset_matrix_micro = torch.zeros((n_microtime, n_conditions), device=device)

    for cond_idx in range(n_conditions):
        duration_bins = max(1, int(np.round(durations[cond_idx] / microtime_dt)))
        for run_idx in range(len(run_starts)):
            run_start_micro = run_starts[run_idx] * bins_per_tr
            for onset_time in all_onsets[cond_idx][run_idx]:
                onset_bin = run_start_micro + int(np.round(onset_time / microtime_dt))
                if onset_bin < n_microtime:
                    onset_matrix_micro[
                        onset_bin : min(onset_bin + duration_bins, n_microtime),
                        cond_idx,
                    ] = 1.0

    return build_task_design_from_args(
        hrf_model_name=hrf_model_name,
        is_fir_model=is_fir_model,
        fir_bot=fir_bot,
        fir_top=fir_top,
        n_basis=n_basis,
        all_onsets=all_onsets,
        stim_durations=durations,
        onset_matrix_micro=onset_matrix_micro,
        n_conditions=n_conditions,
        n_timepoints=n_timepoints,
        run_starts=run_starts,
        tr=tr,
        microtime_dt=microtime_dt,
        device=device,
        hrf_opt=hrf_opt,
        hrf_library=hrf_library,
        hrf_indices=hrf_indices,
        n_voxels=n_voxels,
    )


def build_polort_nuisance(
    run_starts: list[int],
    n_timepoints: int,
    tr: float,
    polort_arg: int | None,
    device: torch.device,
) -> list[torch.Tensor]:
    """Per-run Legendre polynomial nuisance, one block per run."""
    nuisance_per_run = []
    n_runs = len(run_starts)
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp

        if polort_arg is None:
            polort = auto_polort(run_length * tr, formula="afni")
        else:
            polort = polort_arg

        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
        else:
            poly = torch.zeros((run_length, 0), device=device)
        nuisance_per_run.append(poly)
    return nuisance_per_run


# ============================================================================
# Main
# ============================================================================


def main():
    parser = create_parser()

    # Show help and exit when called with no args (argparse's -h/--help is fine)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Timing must come from exactly one source.
    if bool(args.onsets) == bool(args.events):
        print("ERROR: Specify exactly one of -onsets/-durations or -events")
        sys.exit(1)
    if args.onsets and not args.durations:
        print("ERROR: -durations is required with -onsets")
        sys.exit(1)
    if args.durations and args.events:
        print("ERROR: -durations is not used with -events (durations come from the TSV)")
        sys.exit(1)
    if args.event_ignore and not args.events:
        print("ERROR: -event_ignore requires -events")
        sys.exit(1)
    if args.event_cols and not args.events:
        print("ERROR: -event_cols requires -events")
        sys.exit(1)
    if bool(args.test_input) != bool(args.test_events):
        print("ERROR: -test_input and -test_events must be given together")
        sys.exit(1)
    # Silently ignoring these would look like the held-out nuisance was applied.
    if not args.test_input and any(
        getattr(args, f"test_{name}", None)
        for name in ("ortvec", "ortvec_run", "ortvec_glob", "ortvec_concat")
    ):
        print("ERROR: -test_ortvec* requires -test_input")
        sys.exit(1)
    if args.test_curve and not args.test_input:
        print("ERROR: -test_curve requires -test_input")
        sys.exit(1)
    if args.null_surrogates < 0:
        print("ERROR: -null_surrogates must be >= 0")
        sys.exit(1)
    # cross_run's >= 3 run requirement is the same one combinatorial denoising
    # already imposes (outer LORO plus an inner LORO), so it can simply be the
    # default -- there is no run count that reaches here and cannot use it.
    criterion = args.criterion or "cross_run"
    if args.null_surrogates > 0 and not args.singleton_only:
        print("ERROR: -null_surrogates requires -singleton_only")
        sys.exit(1)
    if not 0 < args.null_percentile <= 100:
        print("ERROR: -null_percentile must be in (0, 100]")
        sys.exit(1)
    if args.test_curve_subsets < 1:
        print("ERROR: -test_curve_subsets must be >= 1")
        sys.exit(1)

    # Fail on a malformed criteria spec now, not after a long data load.
    from fastfuncstuff.denoise.combinatorial import parse_criteria_spec

    try:
        parse_criteria_spec(args.criteria_r2_threshold)
    except ValueError as exc:
        print(f"ERROR: -criteria_r2_threshold {args.criteria_r2_threshold!r}: {exc}")
        print("  Expected a float (0.05), a percentile ('5%'), or a top-N ('(1000)').")
        sys.exit(1)

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    print_cli_header("ffs_denoisatorial", "Combinatorial PC Denoising")

    # ======================================================================
    # Parse and validate inputs
    # ======================================================================
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 3:
        print("ERROR: At least 3 runs required for combinatorial denoising")
        print("  (outer LORO needs >=2 training runs for inner LORO)")
        sys.exit(1)

    # Parse the timing spec: BIDS events TSVs or AFNI timing files.
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
            input_files=input_files,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_onsets = timing.all_onsets
    durations = timing.durations
    condition_labels = timing.condition_labels
    n_conditions = timing.n_conditions

    # Warn about large max_pcs
    if args.max_pcs > 12:
        print(f"  WARNING: max_pcs={args.max_pcs} gives {2**args.max_pcs} combinations.")
        print("  This may be slow and memory-intensive. Consider max_pcs <= 10.")

    n_combos = 2**args.max_pcs
    print(f"  Max PCs: {args.max_pcs} -> {n_combos} combinations per run")

    # Setup device
    device = setup_device(args.device)
    print(f"  Device: {device}")

    # ======================================================================
    # Load data
    # ======================================================================
    print()
    load_result: LoadResult = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=False,
        device=device,
        force_cpu=args.keep_on_cpu,
        dry_run=args.dry_run,
        verbose=True,
        load_threads=args.load_threads,
        drop_first=args.drop_first,
        drop_last=args.drop_last,
    )

    # Modify prefix for dry run mode
    if args.dry_run:
        args.prefix = f"dry_run_{args.prefix}"

    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
    volume_shape = load_result.volume_shape
    mask = load_result.mask
    mask_flat = load_result.mask_flat
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    n_runs = load_result.n_runs

    if args.tr is None:
        args.tr = load_result.tr

    args.microtime_dt = resolve_microtime_dt(args.tr, args.microtime_dt)
    # Timing was parsed before the load, so the -drop_first shift lands here.
    trim = trim_spec_from_args(args, tr=args.tr)
    apply_trim_to_timing(
        timing,
        trim,
        run_lengths_tr=run_lengths_from_starts(run_starts, n_timepoints),
        n_runs=n_runs,
    )
    all_onsets = timing.all_onsets

    nifti_header = load_result.nifti_header

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
        single_trial=False,  # ffs_denoisatorial doesn't have -single_trial
        hrf_opt=args.hrf_opt,
    )

    # Automask BEFORE brainthresh / scaling, so every later step sees brain only
    if args.automask:
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

    # Compute brainthresh intensity mask BEFORE scaling.
    # Without any brain restriction the noise pool is "every voxel with low
    # task R²", which is mostly air — background PCs are pure noise/ghosting and
    # swamp the real nuisance structure. GLMdenoise's default (99th percentile
    # x 0.5) is the reference behaviour, so apply it unless the user has already
    # restricted the voxel set some other way.
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
        mean_intensity = data.mean(dim=1)
        percentile_value = torch.quantile(mean_intensity, percentile / 100.0)
        threshold = percentile_value * fraction
        brainthresh_mask = mean_intensity > threshold
        n_above = brainthresh_mask.sum().item()
        print(f"  {percentile:.0f}th percentile intensity: {percentile_value:.2f}")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Voxels above: {n_above:,} of {n_voxels:,} ({n_above / n_voxels * 100:.1f}%)")

    # Drop zero-variance voxels outright. Keeping them only in the noise-pool
    # exclusion (the old behaviour) still let them reach the R² maps, where
    # 1 - 0/0 scores them a perfect 1.0 and painted the out-of-FoV background.
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
    else:
        print("  All voxels have usable variance in every run")

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
        data, _, _ = scale_to_percent_signal(
            data=data,
            run_starts=run_starts,
            max_scale=200.0,
            verbose=True,
        )

    print(f"\n  Data shape: {data.shape} ({n_voxels:,} voxels x {n_timepoints} timepoints)")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ======================================================================
    # Build design matrix
    # ======================================================================
    print()
    print("Building design matrix...")

    # Convolve with HRF(s)
    hrf_indices = None
    hrf_library = None

    if args.hrf_opt:
        # Per-voxel HRF mode: load HRF indices and library from 3dHRFoptfast output
        print(f"  Loading HRF optimization results from {args.hrf_opt}...")
        hrf_index_file = f"{args.hrf_opt}_hrf_index.nii.gz"
        if not Path(hrf_index_file).exists():
            print(f"ERROR: HRF index file not found: {hrf_index_file}")
            print("  Expected output from 3dHRFoptfast with prefix:", args.hrf_opt)
            sys.exit(1)

        mask_for_hrf = mask if mask is not None else None
        hrf_indices = load_hrf_indices(hrf_index_file, mask=mask_for_hrf)
        hrf_indices = hrf_indices.to(data.device)
        print(f"  Loaded HRF indices: {hrf_indices.shape}")

        # Load or reconstruct HRF library
        hrf_lib_file = f"{args.hrf_opt}_hrf_library.pt"
        if Path(hrf_lib_file).exists():
            from fastfuncstuff.cli_utils import spinner

            with spinner(f"Loading {Path(hrf_lib_file).name}"):
                hrf_lib_data = torch.load(hrf_lib_file, weights_only=False)
            hrf_library = hrf_lib_data["hrf_library"]
            print(f"  Loaded HRF library from {hrf_lib_file}: {hrf_library.shape}")
        else:
            # Determine n_hrfs from the unique indices
            n_hrfs = int(hrf_indices.max().item()) + 1
            hrf_library = get_hrf_library(
                mode="library",
                tr=args.tr,
                n_hrfs=n_hrfs,
                microtime_dt=args.microtime_dt,
                device=device,
            )
            print(f"  Using default HRF library with {hrf_library.shape[0]} HRFs")

        # Show HRF distribution
        unique_hrfs, counts = torch.unique(hrf_indices, return_counts=True)
        print(f"  HRF distribution across {len(unique_hrfs)} unique HRFs:")
        for hrf_idx_show, count in zip(unique_hrfs[:5].tolist(), counts[:5].tolist(), strict=False):
            print(f"    HRF {hrf_idx_show}: {count:,} voxels ({count / n_voxels * 100:.1f}%)")
        if len(unique_hrfs) > 5:
            print(f"    ... and {len(unique_hrfs) - 5} more HRFs")

    task_design, designs_by_hrf = build_task_designs(
        all_onsets=all_onsets,
        durations=durations,
        n_conditions=n_conditions,
        run_starts=run_starts,
        n_timepoints=n_timepoints,
        tr=args.tr,
        microtime_dt=args.microtime_dt,
        hrf_model_name=hrf_model_name,
        is_fir_model=is_fir_model,
        fir_bot=fir_bot,
        fir_top=fir_top,
        n_basis=n_basis,
        device=device,
        hrf_opt=args.hrf_opt,
        hrf_library=hrf_library,
        hrf_indices=hrf_indices,
        n_voxels=n_voxels if args.hrf_opt else None,
    )

    # Build nuisance per run (polynomials + ortvec)
    nuisance_per_run = build_polort_nuisance(
        run_starts=run_starts,
        n_timepoints=n_timepoints,
        tr=args.tr,
        polort_arg=args.polort,
        device=device,
    )
    # Add user nuisance blocks (-ortvec / -ortvec_run / -ortvec_glob).
    user_blocks = collect_nuisance_blocks(
        args,
        run_starts,
        n_timepoints,
        verbose=(args.verb >= 1),
        trim=trim,
    )
    nuisance_per_run = append_nuisance_blocks(
        nuisance_per_run, user_blocks, run_starts, n_timepoints
    )

    print(
        f"  Nuisance per run: {nuisance_per_run[0].shape[1]} cols "
        f"(polort{'+ortvec' if user_blocks else ''})"
    )

    # ======================================================================
    # Step 1: Compute initial cross-validated R2 (task-only, for noise pool)
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 1: Computing initial cross-validated R2 (task-only)...")
    print("=" * 70)

    initial_r2 = compute_initial_xval_r2(
        data=data,
        design=task_design,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=args.verb >= 1,
    )

    print(
        f"  Initial R2: median={initial_r2.median().item():.4f}, "
        f"mean={initial_r2.mean().item():.4f}"
    )

    # ======================================================================
    # Step 2: Select noise pool
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 2: Selecting noise pool...")
    print("=" * 70)

    noise_pool_mask, criteria_mask = select_noise_pool_voxels(
        r2=initial_r2,
        threshold=args.r2_threshold,
        min_noise_voxels=args.min_noise_voxels,
        max_noise_fraction=args.max_noise_fraction,
    )

    # Apply brainthresh mask to noise pool
    if brainthresh_mask is not None:
        noise_pool_mask = noise_pool_mask & brainthresh_mask.to(noise_pool_mask.device)

    n_noise = noise_pool_mask.sum().item()
    n_criteria = criteria_mask.sum().item()
    print(f"  Noise pool: {n_noise:,} voxels")
    print(f"  Criteria: {n_criteria:,} voxels")

    # -whole_brain_noise_pool widens where PCs are *extracted* from without
    # touching the criteria pool they are scored on. Task-dominated components
    # are not a safety problem here — removing one strips the variance the
    # betas exist to predict, so the criterion rejects it — they just crowd the
    # top of the variance ordering and push real artifacts past max_pcs.
    pc_source_mask = noise_pool_mask
    if args.whole_brain_noise_pool:
        pc_source_mask = (
            brainthresh_mask.to(noise_pool_mask.device)
            if brainthresh_mask is not None
            else torch.ones_like(noise_pool_mask)
        )
        print(
            f"  PC source: whole brain ({int(pc_source_mask.sum()):,} voxels) — not the noise pool"
        )

    # ======================================================================
    # Step 3: Run combinatorial denoising
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 3: Combinatorial PC denoising...")
    print("=" * 70)

    results = fit_combinatorial_denoising(
        data=data,
        design=task_design,
        run_starts=run_starts,
        tr=args.tr,
        nuisance_per_run=nuisance_per_run,
        noise_pool_mask=pc_source_mask,
        initial_r2=initial_r2,
        max_pcs=args.max_pcs,
        criteria_r2_threshold=args.criteria_r2_threshold,
        criteria_fallback_percentile=args.criteria_fallback_pct,
        selection_strategy=args.selection_strategy,
        singleton_only=args.singleton_only,
        criterion=criterion,
        n_null_surrogates=args.null_surrogates,
        null_percentile=args.null_percentile,
        null_seed=args.null_seed,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=True,
    )

    # ======================================================================
    # Step 4: Compute final cross-validated R2 with optimal PCs
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 4: Computing optimized cross-validated R2...")
    print("=" * 70)

    optimized_r2 = compute_optimized_xval_r2_3dDenoise_style(
        data=data,
        design=task_design,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        noise_pcs_per_run=results.noise_pcs_per_run,
        per_run_results=results.per_run_results,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=True,
    )

    print(
        f"  Optimized R2: median={optimized_r2.median().item():.4f}, "
        f"mean={optimized_r2.mean().item():.4f}"
    )

    improvement = optimized_r2.median().item() - initial_r2.median().item()
    print(f"  Improvement: {improvement:+.4f} (median)")

    # ======================================================================
    # Step 4a (optional): the ceilings that make those R2s comparable
    # ======================================================================
    initial_ceiling_layers: list = []
    optimized_ceiling_layers: list = []
    if args.noise_ceiling in ("auto", "loro"):
        from fastfuncstuff.glm.xval import generate_cv_splits
        from fastfuncstuff.stats.noise_ceiling import loro_ceiling_by_voxel_group

        print()
        print("Noise ceiling:")
        ceiling_splits = generate_cv_splits(len(run_starts), strategy=1)

        def _ceiling(nuis, label):
            result = loro_ceiling_by_voxel_group(
                data=data,
                nuisance_per_run=nuis,
                run_starts=run_starts,
                cv_splits=ceiling_splits,
                design_matrix=None if designs_by_hrf else task_design,
                designs_by_hrf=designs_by_hrf,
                hrf_indices=hrf_indices,
                device=device,
                progress_desc=f"  {label} ceiling by HRF",
                show_progress=True,
            )
            for note in result.notes:
                print(f"  NOTE: {note}")
            return result

        # The initial R2 sees no PCs; the optimized R2 sees each run's SELECTED
        # combination -- an arbitrary subset here, not a leading-k prefix, which
        # is exactly why the ceiling has to be rebuilt rather than reused.
        initial_result = _ceiling(nuisance_per_run, "Initial")
        if initial_result.n_usable:
            initial_explainable = initial_result.explainable_r2(initial_r2.detach().cpu())
            print(f"  Initial (0 PCs): {initial_result.summarize(initial_explainable)}")
            initial_ceiling_layers = [
                (initial_result.ceiling, "noise_ceiling"),
                (initial_explainable, "explainable_R2"),
            ]

        nuisance_with_pcs = []
        for run_idx, base in enumerate(nuisance_per_run):
            selected = list(results.per_run_results[run_idx].optimal_combination)
            if not selected:
                nuisance_with_pcs.append(base)
                continue
            pcs = results.noise_pcs_per_run[run_idx]
            pcs = pcs if torch.is_tensor(pcs) else torch.as_tensor(pcs)
            block = pcs[:, selected].to(device=base.device, dtype=base.dtype)
            nuisance_with_pcs.append(torch.cat([base, block], dim=1))

        optimized_result = _ceiling(nuisance_with_pcs, "Optimized")
        if optimized_result.n_usable:
            optimized_explainable = optimized_result.explainable_r2(optimized_r2.detach().cpu())
            print(
                f"  Optimized (selected PCs): {optimized_result.summarize(optimized_explainable)}"
            )
            optimized_ceiling_layers = [
                (optimized_result.ceiling, "noise_ceiling"),
                (optimized_explainable, "explainable_R2"),
            ]

    # ======================================================================
    # Step 4b (optional): GLMdenoise-style baseline comparison
    # ======================================================================
    # When -compare is set, run the standard incremental noise-PC sweep
    # (variance-ordered PCs, take 0..k, pick k via the GLMdenoise pcstop=1.05
    # rule on median R²) on the *same* noise pool and noise PCs. Reports how
    # much R² the combinatorial / singleton choice buys you over the
    # GLMdenoise default. No-op when -compare isn't set.
    #
    # FAIRNESS NOTE: both paths use matching LORO CV with identical inputs
    # (same noise pool, same noise PCs, same nuisance, same task design), so
    # the comparison is conceptually apples-to-apples. BUT the two paths
    # evaluate R² through different code:
    #   - combinatorial: compute_optimized_xval_r2_3dDenoise_style →
    #     compute_xval_r2 (full prediction accumulators, combinatorial.py:980)
    #   - baseline below: cross_validate_noise_pcs (streaming sufficient-
    #     stats accumulators, sequential.py:1111)
    # Math is equivalent but float reduction order differs → delta map
    # carries a ~1e-5 implementation-noise floor. Matters only when the
    # combinatorial advantage is itself near that scale. Tighter fix is to
    # route the baseline through the same evaluator (fabricate a per-run
    # result with optimal_combination=(0..k-1) and call
    # compute_optimized_xval_r2_3dDenoise_style for both). Deferred until
    # we've seen real-data deltas and know whether the floor matters.
    baseline_r2_t: torch.Tensor | None = None
    delta_r2_t: torch.Tensor | None = None
    baseline_k: int | None = None
    baseline_pcs_per_run = results.noise_pcs_per_run
    if args.compare:
        print()
        print("=" * 70)
        print("Step 4b: GLMdenoise-style incremental baseline (pcstop=1.05)...")
        print("=" * 70)

        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

        # GLMdenoise is defined on a noise pool. Handing it whole-brain PCs
        # because -whole_brain_noise_pool happened to be set would compare our
        # change against a straw man, so the baseline always gets its own
        # classic-pool PCs.
        if args.whole_brain_noise_pool:
            from fastfuncstuff.denoise.combinatorial import (
                extract_pcs_single_run_with_variance,
            )

            print("  Re-extracting noise-pool PCs for the baseline (not whole-brain)")
            baseline_pcs_per_run = []
            for run_idx in range(n_runs):
                r_start = run_starts[run_idx]
                r_end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_pcs, _ = extract_pcs_single_run_with_variance(
                    run_data=data[:, r_start:r_end],
                    noise_pool_mask=noise_pool_mask,
                    nuisance=nuisance_per_run[run_idx],
                    max_components=args.max_pcs,
                    device=device,
                )
                baseline_pcs_per_run.append(run_pcs.cpu())

        r2_maps_inc, r2_summary_inc = cross_validate_noise_pcs(
            data=data,
            design_matrix=task_design,
            noise_pcs=baseline_pcs_per_run,
            run_starts=run_starts,
            tr=args.tr,
            max_components=args.max_pcs,
            nuisance=nuisance_per_run,
            cv_strategy=1,  # LORO — matches the combinatorial path
            device=device,
            verbose=args.verb >= 1,
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
        )

        # pcstop=1.05 rule: walk up while next/current >= 1.05 on median R².
        # If median is non-positive, fall back to argmax to stay defensive.
        PCSTOP = 1.05
        baseline_k = 0
        for k in range(len(r2_summary_inc) - 1):
            cur, nxt = r2_summary_inc[k], r2_summary_inc[k + 1]
            if cur > 1e-6 and nxt / cur >= PCSTOP:
                baseline_k = k + 1
            else:
                break
        if r2_summary_inc[baseline_k] <= 0:
            baseline_k = int(np.argmax(r2_summary_inc))

        baseline_r2_np = r2_maps_inc[:, baseline_k]
        baseline_r2_t = torch.from_numpy(baseline_r2_np).to(optimized_r2.device)
        delta_r2_t = optimized_r2 - baseline_r2_t

        print(f"  Baseline picked k={baseline_k} PCs (pcstop=1.05)")
        print(
            f"  Baseline R²: median={float(baseline_r2_t.median()):.4f}, "
            f"mean={float(baseline_r2_t.mean()):.4f}"
        )
        print("  Δ R² (combinatorial − baseline):")
        print(
            f"    Mean:    {float(delta_r2_t.mean()):+.4f}    "
            f"Median: {float(delta_r2_t.median()):+.4f}"
        )
        q25, q75 = (
            float(torch.quantile(delta_r2_t, 0.25)),
            float(torch.quantile(delta_r2_t, 0.75)),
        )
        print(f"    IQR:     [{q25:+.4f}, {q75:+.4f}]")
        combo_wins = int((delta_r2_t > 0.01).sum())
        base_wins = int((delta_r2_t < -0.01).sum())
        n_total = delta_r2_t.numel()
        print(
            f"    Voxels combinatorial wins (Δ > 0.01): "
            f"{combo_wins:,}/{n_total:,} ({100 * combo_wins / max(n_total, 1):.1f}%)"
        )
        print(
            f"    Voxels baseline wins (Δ < -0.01):     "
            f"{base_wins:,}/{n_total:,} ({100 * base_wins / max(n_total, 1):.1f}%)"
        )

    # ======================================================================
    # Step 4c (optional): fully held-out prediction on -test_input
    # ======================================================================
    # Everything above is cross-validated *within* the input runs, so the
    # selection saw every voxel it is scored on. These runs were never loaded
    # until now: the winning per-run PCs give betas on all input runs, the
    # consensus PC indices are re-extracted from each held-out run's own noise
    # pool, and the betas predict runs nothing in the fit has seen.
    heldout_maps: dict[str, torch.Tensor] = {}
    curve_files: dict[str, str] = {}
    if args.test_input:
        from fastfuncstuff.cli_utils import parse_timing_spec
        from fastfuncstuff.denoise.heldout import heldout_prediction_r2

        print()
        print("=" * 70)
        print("Step 4c: Fully held-out prediction (-test_input)...")
        print("=" * 70)

        test_files = parse_input_files(args.test_input)
        n_test_runs = len(test_files)

        try:
            test_timing = parse_timing_spec(
                events=args.test_events,
                onsets=None,
                durations_arg=None,
                n_runs=n_test_runs,
                event_ignore=args.event_ignore,
                event_cols=tuple(args.event_cols) if args.event_cols else None,
                round_durations=args.round_durations,
                input_files=test_files,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: -test_events: {exc}", file=sys.stderr)
            sys.exit(1)

        # Design columns are positional, so the condition lists must match
        # exactly — same labels in the same order — or the betas fit on the
        # input runs would be applied to the wrong regressors.
        if test_timing.condition_labels != condition_labels:
            print("ERROR: -test_events conditions differ from the input conditions.")
            print(f"  input: {condition_labels}")
            print(f"  test:  {test_timing.condition_labels}")
            sys.exit(1)

        # Held-out data is scored by streaming chunks to the GPU, so it never
        # needs to live there — and there can be far more held-out runs than
        # input runs (whole extra sessions), which is exactly the case where
        # the loader's concatenate would OOM the card.
        test_load: LoadResult = load_and_preprocess_runs(
            input_files=test_files,
            tr=args.tr,
            mask_file=None,  # the analysed voxel set comes from the input runs
            blur_fwhm=args.do_blur,
            do_scale=False,
            device=device,
            force_cpu=True,
            dry_run=False,
            verbose=True,
            load_threads=args.load_threads,
            drop_first=args.drop_first,
            drop_last=args.drop_last,
        )

        if tuple(test_load.volume_shape) != tuple(volume_shape):
            print(
                f"ERROR: -test_input grid {tuple(test_load.volume_shape)} does not match "
                f"the input grid {tuple(volume_shape)}."
            )
            sys.exit(1)

        test_data = test_load.data
        test_run_starts = test_load.run_starts
        test_n_timepoints = test_load.n_timepoints

        # The held-out runs are trimmed the same way, so their timing needs the
        # same shift -- against their own run lengths, which may differ.
        apply_trim_to_timing(
            test_timing,
            trim,
            run_lengths_tr=run_lengths_from_starts(test_run_starts, test_n_timepoints),
            n_runs=n_test_runs,
        )

        # Restrict the held-out runs to exactly the voxels that were analysed
        # (mask / automask / constant-voxel filter all folded into mask_flat).
        if mask_flat is not None:
            keep_rows = torch.from_numpy(np.asarray(mask_flat, dtype=bool))
            if test_data.shape[0] != keep_rows.numel():
                print(
                    f"ERROR: -test_input has {test_data.shape[0]:,} voxels, expected "
                    f"{keep_rows.numel():,} to match the input volume."
                )
                sys.exit(1)
            test_data = test_data[keep_rows.to(test_data.device)]
            # The full-volume load is a second copy of a dataset that can be
            # bigger than the input runs; drop it now, not at scope exit.
            test_load.data = None  # type: ignore[assignment]
        if test_data.shape[0] != n_voxels:
            print(
                f"ERROR: held-out voxel count {test_data.shape[0]:,} != analysed "
                f"{n_voxels:,}; cannot align the two datasets."
            )
            sys.exit(1)

        if args.do_scale:
            test_data, _, _ = scale_to_percent_signal(
                data=test_data,
                run_starts=test_run_starts,
                max_scale=200.0,
                verbose=args.verb >= 1,
            )

        # Voxels that are flat in a held-out run cannot be scored there
        # (SS_tot = 0 → a spurious R² of 1). Track them and zero them out.
        test_valid = find_constant_voxels(test_data, test_run_starts)
        n_test_invalid = int((~test_valid).sum().item())
        if n_test_invalid > 0:
            print(
                f"  {n_test_invalid:,} voxels are constant in a held-out run; "
                "their held-out R² is set to 0"
            )

        test_designs = build_task_designs(
            all_onsets=test_timing.all_onsets,
            durations=test_timing.durations,
            n_conditions=n_conditions,
            run_starts=test_run_starts,
            n_timepoints=test_n_timepoints,
            tr=args.tr,
            microtime_dt=args.microtime_dt,
            hrf_model_name=hrf_model_name,
            is_fir_model=is_fir_model,
            fir_bot=fir_bot,
            fir_top=fir_top,
            n_basis=n_basis,
            device=device,
            hrf_opt=args.hrf_opt,
            hrf_library=hrf_library,
            hrf_indices=hrf_indices,
            n_voxels=n_voxels if args.hrf_opt else None,
        )
        test_task_design, test_designs_by_hrf = test_designs

        # Polynomials, plus whatever -test_ortvec supplies. The input runs'
        # -ortvec blocks are never reused here: they are the wrong length and
        # describe different runs.
        test_nuisance_per_run = build_polort_nuisance(
            run_starts=test_run_starts,
            n_timepoints=test_n_timepoints,
            tr=args.tr,
            polort_arg=args.polort,
            device=device,
        )
        test_blocks = collect_nuisance_blocks(
            args,
            test_run_starts,
            test_n_timepoints,
            verbose=(args.verb >= 1),
            prefix="test_",
            trim=trim,
        )
        test_nuisance_per_run = append_nuisance_blocks(
            test_nuisance_per_run, test_blocks, test_run_starts, test_n_timepoints
        )
        print(
            f"  Held-out nuisance per run: {test_nuisance_per_run[0].shape[1]} cols "
            f"(polort{'+test_ortvec' if test_blocks else ''})"
        )
        if user_blocks and not test_blocks:
            print(
                "  Note: -ortvec describes the input runs only; pass -test_ortvec "
                "to model the held-out runs' nuisance too"
            )

        train_selections = [r.optimal_combination for r in results.per_run_results]
        print(f"  Per-run PCs in the fit: {[list(s) for s in train_selections]}")

        def _run_heldout(train_sel, train_pcs=None):
            r2 = heldout_prediction_r2(
                train_data=data,
                train_run_starts=run_starts,
                train_nuisance_per_run=nuisance_per_run,
                train_pcs_per_run=(results.noise_pcs_per_run if train_pcs is None else train_pcs),
                train_selections=train_sel,
                test_data=test_data,
                test_run_starts=test_run_starts,
                test_nuisance_per_run=test_nuisance_per_run,
                train_design=task_design,
                test_design=test_task_design,
                train_designs_by_hrf=designs_by_hrf,
                test_designs_by_hrf=test_designs_by_hrf,
                hrf_indices=hrf_indices,
                device=device,
                verbose=args.verb >= 1,
            )
            # compute_xval_r2 returns an inference-mode tensor; clone before
            # masking the unscoreable voxels.
            r2 = r2.to(test_valid.device).clone()
            r2[~test_valid] = 0.0
            return r2

        heldout_initial = _run_heldout([() for _ in train_selections])
        heldout_r2 = _run_heldout(train_selections)

        heldout_maps["heldout_initial_r2"] = heldout_initial
        heldout_maps["heldout_r2"] = heldout_r2

        print(
            f"  Held-out R² (no denoising):  median={heldout_initial.median().item():+.4f}, "
            f"mean={heldout_initial.mean().item():+.4f}"
        )
        print(
            f"  Held-out R² (denoised fit):  median={heldout_r2.median().item():+.4f}, "
            f"mean={heldout_r2.mean().item():+.4f}"
        )
        print(
            f"  Held-out improvement:        "
            f"{heldout_r2.median().item() - heldout_initial.median().item():+.4f} (median)"
        )
        # The internal R² scores against data that also had its PCs removed;
        # this one scores against the raw held-out timeseries. Same prediction,
        # bigger denominator — compare held-out to held-out, not to Step 4.
        print("  (held-out R² scores raw data; not on the same scale as the internal xval R²)")

        if args.compare and baseline_k is not None:
            # Same procedure for the GLMdenoise choice: refit with its first
            # baseline_k variance-ordered PCs in every run, predict the same
            # held-out data.
            gd_sel = tuple(range(baseline_k))
            heldout_gd = _run_heldout(
                [gd_sel for _ in train_selections], train_pcs=baseline_pcs_per_run
            )
            heldout_delta = heldout_r2 - heldout_gd
            heldout_maps["heldout_r2_glmdenoise"] = heldout_gd
            heldout_maps["heldout_r2_delta"] = heldout_delta

            print(
                f"  Held-out R² (GLMdenoise k={baseline_k}): "
                f"median={heldout_gd.median().item():+.4f}, mean={heldout_gd.mean().item():+.4f}"
            )
            print("  Δ held-out R² (combinatorial − GLMdenoise):")
            print(
                f"    Mean:    {heldout_delta.mean().item():+.4f}    "
                f"Median: {heldout_delta.median().item():+.4f}"
            )
            hd_wins = int((heldout_delta > 0.01).sum())
            hd_loss = int((heldout_delta < -0.01).sum())
            hd_total = heldout_delta.numel()
            print(
                f"    Voxels combinatorial wins (Δ > 0.01): "
                f"{hd_wins:,}/{hd_total:,} ({100 * hd_wins / max(hd_total, 1):.1f}%)"
            )
            print(
                f"    Voxels GLMdenoise wins (Δ < -0.01):   "
                f"{hd_loss:,}/{hd_total:,} ({100 * hd_loss / max(hd_total, 1):.1f}%)"
            )

        if args.test_curve:
            from fastfuncstuff.denoise.heldout import (
                heldout_learning_curve,
                plot_heldout_learning_curve,
            )

            print()
            print("  Learning curve: held-out R² vs number of training runs")

            # Union of the pre- and post-denoising active pools. Scoring only
            # on voxels the denoised fit calls active would hide the failure
            # that matters most: a voxel that improved in training and fell
            # apart on held-out data never enters the average.
            thr = args.r2_threshold
            active_initial = initial_r2 >= thr
            active_denoised = optimized_r2.to(active_initial.device) >= thr
            active_mask = (active_initial | active_denoised) & test_valid.to(active_initial.device)
            n_active = int(active_mask.sum().item())
            if n_active == 0:
                print(
                    f"  ⚠️  No voxels reach R² >= {thr:g} before or after denoising; "
                    "skipping the learning curve"
                )
            else:
                print(
                    f"  Active voxels (R² >= {thr:g} before OR after denoising): "
                    f"{n_active:,} of {n_voxels:,} "
                    f"({int(active_initial.sum()):,} initial, "
                    f"{int(active_denoised.sum()):,} denoised)"
                )

                curve_arms: dict[str, list[tuple[int, ...]]] = {
                    "initial": [() for _ in train_selections],
                    "denoised": list(train_selections),
                }
                curve_arm_pcs: dict[str, list[torch.Tensor]] = {}
                if args.compare and baseline_k is not None:
                    gd_name = f"glmdenoise (k={baseline_k})"
                    curve_arms[gd_name] = [tuple(range(baseline_k)) for _ in train_selections]
                    # Keeps its noise-pool PCs under -whole_brain_noise_pool.
                    curve_arm_pcs[gd_name] = baseline_pcs_per_run

                curve = heldout_learning_curve(
                    train_data=data,
                    train_run_starts=run_starts,
                    train_nuisance_per_run=nuisance_per_run,
                    train_pcs_per_run=results.noise_pcs_per_run,
                    arms=curve_arms,
                    arm_pcs_per_run=curve_arm_pcs,
                    test_data=test_data,
                    test_run_starts=test_run_starts,
                    test_nuisance_per_run=test_nuisance_per_run,
                    train_design=task_design,
                    test_design=test_task_design,
                    train_designs_by_hrf=designs_by_hrf,
                    test_designs_by_hrf=test_designs_by_hrf,
                    hrf_indices=hrf_indices,
                    max_subsets=args.test_curve_subsets,
                    device=device,
                    verbose=args.verb >= 1,
                )

                curve_files.update(
                    _report_learning_curve(
                        curve=curve,
                        active_mask=active_mask,
                        prefix=args.prefix,
                        plot=plot_heldout_learning_curve,
                    )
                )

        del test_data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ======================================================================
    # Step 5: Save results
    # ======================================================================
    print()
    print("=" * 70)
    print("Saving results...")
    print("=" * 70)

    output_files = save_combinatorial_results(
        results=results,
        initial_r2_full=initial_r2,
        optimized_r2_full=optimized_r2,
        output_prefix=args.prefix,
        volume_shape=volume_shape,
        affine=affine,
        run_starts=run_starts,
        tr=args.tr,
        condition_labels=condition_labels,
        mask_flat=mask_flat,
        plots_mode=args.plots,
        save_pcs_mode=args.save_pcs,
        nii_ext=_nii_ext,
        initial_ceiling_layers=initial_ceiling_layers,
        optimized_ceiling_layers=optimized_ceiling_layers,
    )

    def _flat_to_vol(flat_t: torch.Tensor) -> np.ndarray:
        flat_np = flat_t.detach().cpu().numpy().astype(np.float32)
        if mask_flat is not None:
            vol = np.zeros(mask_flat.size, dtype=np.float32)
            vol[mask_flat] = flat_np
        else:
            vol = flat_np
        return vol.reshape(volume_shape)

    # Held-out prediction maps (-test_input).
    if heldout_maps:
        from fastfuncstuff.cli_utils import spinner

        for name, r2_map in heldout_maps.items():
            path = f"{args.prefix}_{name}{_nii_ext}"
            with spinner(f"Writing {Path(path).name}"):
                save_nifti(_flat_to_vol(r2_map), output_path=path, affine=affine)
            output_files[name] = path
            print(f"  Saved: {path}")

    # -test_curve wrote its own files during Step 4c; register them so the
    # metadata manifest lists everything the run produced.
    output_files.update(curve_files)

    # Save -compare outputs alongside the standard ones.
    if args.compare and baseline_r2_t is not None and delta_r2_t is not None:
        from fastfuncstuff.cli_utils import spinner

        baseline_path = f"{args.prefix}_r2_glmdenoise{_nii_ext}"
        delta_path = f"{args.prefix}_r2_delta{_nii_ext}"
        with spinner("Writing comparison results"):
            save_nifti(_flat_to_vol(baseline_r2_t), output_path=baseline_path, affine=affine)
            save_nifti(_flat_to_vol(delta_r2_t), output_path=delta_path, affine=affine)
        output_files["r2_glmdenoise"] = baseline_path
        output_files["r2_delta"] = delta_path
        print(f"  Saved: {baseline_path}")
        print(f"  Saved: {delta_path}")

    # ======================================================================
    # Summary
    # ======================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Runs: {n_runs}")
    print(f"  Max PCs: {args.max_pcs} ({n_combos} combinations per run)")
    print(f"  Noise pool: {n_noise:,} voxels")
    print(f"  Initial median R2: {initial_r2.median().item():.4f}")
    print(f"  Optimized median R2: {optimized_r2.median().item():.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print()
    print("  Per-run selections:")
    for run_res in results.per_run_results:
        print(
            f"    Run {run_res.run_idx}: PCs {run_res.optimal_combination} "
            f"(CoD={run_res.optimal_cod:.4f}, "
            f"{run_res.optimal_cod - run_res.baseline_cod:+.4f} vs baseline)"
        )
    print()
    print(f"  Outputs: {len(output_files)} files saved with prefix '{args.prefix}'")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
