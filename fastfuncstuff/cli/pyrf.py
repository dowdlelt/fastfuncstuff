#!/usr/bin/env python3
"""ffs_pyrf - GPU population receptive field (pRF) mapping.

A pyre for receptive fields: the compressive spatial summation (CSS) pRF model,
fit on the GPU.

This is a reimplementation of the *method* published by Kendrick Kay and
colleagues, and the model, the super-grid seeding strategy, the staged fit, and
the parameter conventions are all theirs:

    Kay KN, Winawer J, Mezer A, Wandell BA (2013).
    Compressive spatial summation in human visual cortex.
    Journal of Neurophysiology 110(2), 481-494.
    https://doi.org/10.1152/jn.00105.2013

    analyzePRF (MATLAB), by Kendrick Kay - http://kendrickkay.net/analyzePRF/
    Copyright (c) 2014 Kendrick Kay. Licensed CC BY 3.0 Unported.

**If you publish results from this tool, cite Kay et al. 2013.** The reference
toolbox's licence asks for that citation, and it is the right thing regardless.

What is ours is the implementation, not the idea: batched analytic Gauss-Newton
with variable projection over the gain, per-voxel HRF selection, fold-local
leave-one-run-out cross-validation, and the chunking that keeps it on the GPU.
Numerical agreement with the reference is checked in ``tests/test_prf.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_load_threads_arg,
    add_ortvec_arguments,
    add_verbose_arg,
    auto_polort,
    build_nuisance_per_run,
    collect_nuisance_blocks,
    compute_run_lengths,
    get_average_run_duration,
    load_and_preprocess_runs,
    parse_device_arg,
    parse_input_files,
    parse_prefix,
)
from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run
from fastfuncstuff.design.hrf import (
    get_hrf_library,
    load_canonical_hrf_basic,
    load_canonical_hrf_library,
)
from fastfuncstuff.design.prf import (
    PRFRefinedFit,
    PRFRefinementConfig,
    balanced_group_halves,
    downsample_aperture,
    fit_prf_folds,
    fit_prf_supergrid,
    grid_seeds_as_fit,
    loro_folds,
    make_analyzeprf_grid,
    prf_parameter_maps,
    refine_prf_all_hrfs,
    refine_prf_hrf_window,
    refine_prf_supergrid,
    screen_voxels_ridge,
    select_noise_pc_count,
    summarize_hrf_selection,
)
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.stats.reliability import (
    circular_correlation,
    identical_design_groups,
    identical_design_labels,
    spearman_brown,
    spearman_correlation,
    split_half_noise_ceiling,
)
from fastfuncstuff.utils import configure_torch_backends

_EPILOG = """
OUTPUT
  Everything lands in one labeled 4D bucket at {prefix}. Spatial quantities are
  in degrees of visual angle when -screen_extent is given, and in aperture
  pixels otherwise -- pixels are not comparable across studies, or even across
  -stim_downsample settings, so pass -screen_extent for anything you will share.

    x, y            pRF center relative to the aperture center. y is positive
                    upward in the visual field.
    angle           polar angle in degrees; 0 = right, increasing toward y.
                    NaN where eccentricity is exactly 0.
    eccentricity    distance from the aperture center, hypot(x, y).
    sigma           Gaussian standard deviation of the pRF, before the CSS
                    exponent is applied.
    exponent        CSS compressive exponent n. Fixed at 1 for -model_mode 3.
    rfsize          sigma / sqrt(n) -- THE size measure to report. sigma and n
                    are individually unidentifiable (the objective has a flat
                    valley where growing sigma trades against shrinking n), so
                    validating against raw sigma will look broken when nothing
                    is.
    gain            amplitude of the fitted response, in the units of the input.
                    Divide by meanvol for percent signal change.
    meanvol         each voxel's mean over time, the scale gain refers to. A
                    property of the input, so it is kept for every voxel even
                    where the fit is NaN (screened out, or all-zero).
    correlation     correlation between prediction and data after nuisance
                    projection; r2 is the coefficient of determination.
    xval_r2         held-out R2 (multi-run input only). See -xval_hrf: by
                    default the folds pick their own HRF at grid resolution, so
                    this scores a slightly different model than the other
                    sub-bricks report.
    noise_ceiling   largest R2 any model could reach on this data. Present only
                    when two or more runs share a bit-identical aperture movie.
                    See NOISE CEILING below for how it is derived; NaN (not 0)
                    where no repeats exist.
    xval_r2_normalized
                    xval_r2 / noise_ceiling -- the fraction of the EXPLAINABLE
                    variance the model got, so 1.0 means it captured everything
                    that reproduces at all. Values slightly above 1 are noise in
                    the ceiling estimate, not a model that beat it.
    hrf_index       selected HRF, ONE-BASED, into the library actually used.
    hrf_index_continuous, hrf_evidence
                    (-hrf_select refine only) parabolic sub-step interpolation
                    of the R2-vs-HRF curve, and how peaked that curve is.
                    hrf_evidence near zero means the library is indistinguishable
                    for that voxel -- threshold on it before believing hrf_index.
    grid_index      winning super-grid candidate, one-based.
    residual_ss, gn_iterations, gn_converged
                    fit diagnostics. A voxel that never converged has
                    gn_iterations equal to -maxiter; that is common and not by
                    itself a failure.

OPTIONAL FILES  (each needs the flag in brackets)
    {prefix}_canonical.nii.gz      [-save_canonical]
        The same bucket refit with the canonical HRF forced, so the
        HRF-selected and fixed-HRF fits can be compared voxelwise on identical
        data rather than across two runs of the tool.
    {prefix}_hrf_r2.nii.gz         [-save_hrf_r2]
        Raw per-voxel x per-HRF R2 matrix, the input to any HRF-selection
        criterion. One sub-brick per library entry.
    {prefix}_screen.nii.gz         [-save_screen]
        Cross-validated R2 of the fast linear screening model. Usable on its
        own as a functionally derived mask.
    {prefix}_noisepool.nii.gz, _noisepcs.1D, _denoise.png, _noisepc*.png
                                   [-save_denoise]
        Which voxels fed the noise pool, the component timecourses actually
        projected out, the held-out-R2-vs-count sweep, and per-component
        timecourse/spatial-weight figures.
    {prefix}_reliability.tsv, .png, .nii.gz    [-xval halves]
        Split-half parameter reliability. See RELIABILITY below.

NOISE CEILING
  Runs with a bit-identical aperture movie have the same expected response, so
  everything they disagree about is noise. Writing a voxel as y = s + e with e
  independent across repeats, corr(y_i, y_j) = var(s) / var(y) -- which is
  exactly the largest R2 any model can reach on a single run. So the ceiling IS
  the mean pairwise correlation across repeats: already in R2 units, no further
  correction, no assumption about the noise structure.

  Three things worth knowing:
    - It is computed AFTER per-run nuisance projection. Shared drift reproduces
      perfectly across repeats and would otherwise be counted as signal, pushing
      the ceiling toward 1 for every voxel with a slow trend in it.
    - Repeats are DETECTED by comparing aperture movies, not declared. That is
      stricter than -stim_groups on purpose: clockwise and counter-clockwise
      wedges are one stimulus group but have different expected timecourses, so
      they are not repeats. A -stim-nii-multi source is recognised for free.
    - Odd/even TR splits within a single run are NOT a substitute. Temporal
      autocorrelation makes the halves agree for reasons unrelated to
      reproducible signal, and the ceiling comes out inflated. Hence NaN rather
      than a fabricated number when no repeats exist.

RELIABILITY  (-xval halves)
  Held-out R2 measures whether the model predicts unseen data. It does NOT
  measure whether a parameter estimate is stable, and for pRF size the two
  answers differ: Lage-Castellanos et al. (2020) compared five pRF tools whose
  prediction accuracy was identical to two decimals while their split-half
  reliability of size ranged from 0.39 to 0.81. If you care about pRF size,
  reliability is the number to look at, and -xval halves is how to get it.

  It works by splitting the runs into two disjoint halves that each cover every
  stimulus group, fitting both, and using them twice over: the two fits predict
  each other's data (held-out R2) and are compared to each other (reliability).
  One pair of fits, both numbers. The halves must be disjoint for the comparison
  to mean anything, which is what forces two folds rather than more -- and two
  folds train on half the data, so the R2 it reports is conservative relative to
  the all-runs fit in the bucket.

  How the number is computed:
    - Reliability is an ACROSS-VOXEL correlation of one parameter between the
      two halves. It therefore needs voxels whose true pRFs differ; over an ROI
      with no retinotopic spread it measures nothing.
    - Voxels are selected by the FULL fit's r2 (-reliability_threshold), never
      by either half's. Selecting on a half keeps the voxels that half happened
      to fit well and biases the agreement upward.
    - Spearman rank correlation for every parameter except angle, because
      eccentricity and size are strongly skewed across an ROI. Ranks are
      tie-averaged: pRF parameters pile up on their bounds, so ties are routine.
    - Polar angle uses circular correlation (Jammalamadaka-Sarma), since a
      linear coefficient calls two estimates either side of the 0/360 seam
      completely inconsistent.
    - Averaged over -xval_draws independent half-splits, then Spearman-Brown
      corrected (r_full = 2r / (1 + r)) because each half saw half the data
      while the reported fit saw all of it. Both forms are written out.

  The three files:
    .tsv   one row per R2 threshold: threshold, n_voxels, the raw correlation
           per parameter, then the same Spearman-Brown corrected (*_sb columns).
           The sweep stops once too few voxels survive for a correlation to
           mean anything, so the last row tells you where the data ran out.
    .png   those curves, Spearman-Brown corrected, with -reliability_threshold
           marked and the surviving voxel count on a log right-hand axis.
    .nii.gz  per-voxel abs_delta_{x,y,eccentricity,angle,sigma,rfsize}: the mean
           absolute half-to-half disagreement, averaged over draws. Same units
           as the main bucket, EXCEPT abs_delta_angle which is always degrees of
           polar angle (wrapped, so it never exceeds 180).

  Read the curve, not just the printed number. Reliability of rfsize typically
  climbs steeply with data quality while position is already flat near 1.0; the
  figure shows whether the threshold you picked sits on a plateau or a cliff,
  with the surviving voxel count on the right axis so you can see where the
  climb is just a shrinking sample.

CHOOSING SETTINGS  (measured on 3 runs x 300 TRs, one subject -- confirm on yours)
  -hrf_select     'grid+1' is the sweet spot: it agrees with the full 20-HRF
                  search on 96% of well-fit voxels at equal R2, for well under
                  half the time. Plain 'grid' is cheaper again and still agrees
                  ~84% of the time; the grid choice is within one library step
                  of the refined one essentially always, which is exactly why
                  grid+1 works and grid+2 adds nothing.
  -grid_angles    32 costs no measurable time (the super-grid stage is
                  launch-bound, not candidate-bound) and clearly beats
                  analyzePRF's 16. Subdividing sigma or eccentricity instead
                  sharpens the seeds but does NOT change the refined fit.
  -stim_downsample
                  refinement cost and memory scale with aperture AREA. A native
                  1080x1080 aperture is ~136x the work of the ~100x100 the
                  reference resizes to, and is not usable as-is.
  -model_mode     3 (linear Gaussian, exponent fixed at 1) is the classic
                  Dumoulin-Wandell model and the natural comparison baseline,
                  but it is NOT meaningfully faster: cost is dominated by the
                  per-voxel Gaussian over the aperture, which every mode pays.
  -xval           'halves' is both cheaper and more informative than 'loro' when
                  runs have repeat structure: LORO fits one model per run (6
                  fits on 5 runs for 6 runs), halves fits two per draw on half
                  the data, and the saving grows with the number of stimulus
                  sets (4 sets x 3 repeats: 12 LORO fits vs 2 per draw).
                  -stim_groups declares which runs belong together and defaults
                  to grouping runs that share an aperture movie, so a
                  -stim-nii-multi source is recognised without saying anything.
  -mask           cost is linear in voxels, so masking is a large lever.
  -denoise        data-derived noise components, GLMdenoise style, with the
                  noise pool taken from the screening pass rather than an
                  anatomical guess. How many components to keep is chosen by
                  cross-validation ON THE SUPER-GRID FIT -- seconds per
                  candidate count, where refining the full model at every count
                  and fold would be tens of fits -- and zero is a legitimate
                  answer when denoising does not earn its degrees of freedom.
                  Needs 2+ runs. -save_denoise writes the sweep figure.
                  -noise_pool_mask decouples the pool from -mask: fit an ROI,
                  but take the components from the whole brain.
  -screen_top     larger still, and functional rather than anatomical: a linear
                  ridge pRF is fitted to every voxel in seconds, and only the
                  best-screening ones get the CSS refinement. Whole brain,
                  496k voxels, canonical HRF: 5:48 -> 20 s at -screen_top 0.1,
                  retaining 98% of the voxels the full fit gives R2 > 0.4 and
                  99.8% of those above 0.5. -save_screen writes the map, which
                  is usable on its own as a functionally derived mask.

Method and conventions are from Kay, Winawer, Mezer & Wandell (2013),
'Compressive spatial summation in human visual cortex', J Neurophysiol
110(2):481-494, doi:10.1152/jn.00105.2013 -- and the analyzePRF MATLAB toolbox
by Kendrick Kay (http://kendrickkay.net/analyzePRF/, Copyright (c) 2014
Kendrick Kay, CC BY 3.0).
PLEASE CITE Kay et al. 2013 if you publish results from this tool.
"""


# Sub-bricks that describe the input data rather than the fit, and so stay valid
# for voxels that were never fitted.
_DATA_COLUMNS = frozenset({"meanvol"})


class _PyrfHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show each flag's default, but leave the epilog's own layout alone."""


def create_parser() -> argparse.ArgumentParser:
    """Create the ffs_pyrf command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "ffs_pyrf - GPU compressive-spatial-summation (CSS) population receptive "
            "field mapping. A pyre for receptive fields."
        ),
        epilog=_EPILOG,
        formatter_class=_PyrfHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument("-input", nargs="+", required=True, help="fMRI run(s), one file per run")
    required.add_argument(
        "-stimulus",
        nargs="+",
        default=None,
        help=(
            "Aperture source(s), one per input run: a row x column x time NIfTI/.npy movie "
            "or a directory of TR-aligned PNG frames"
        ),
    )
    required.add_argument(
        "-stim-pngs",
        "-stim_pngs",
        dest="stim_sources",
        action="append",
        default=[],
        metavar="DIR",
        help="PNG frame directory for one input run; repeat in -input order",
    )
    required.add_argument(
        "-stim-pngs-multi",
        "-stim_pngs_multi",
        dest="stim_sources",
        action="append",
        nargs=2,
        metavar=("DIR", "N_RUNS"),
        help=(
            "Naturally ordered PNG frames from DIR covering the next N_RUNS input runs, "
            "either concatenated and split between them or one run's worth reused for "
            "all of them. The frame count decides which"
        ),
    )
    required.add_argument(
        "-stim-nii",
        "-stim_nii",
        dest="stim_sources",
        action="append",
        metavar="FILE",
        help=(
            "Aperture NIfTI (row x column [x 1] x time) for one input run; repeat in "
            "-input order. Interchangeable with -stim-pngs and evaluated in the order given"
        ),
    )
    required.add_argument(
        "-stim-nii-multi",
        "-stim_nii_multi",
        dest="stim_sources",
        action="append",
        nargs=2,
        metavar=("FILE", "N_RUNS"),
        help=(
            "Aperture NIfTI covering the next N_RUNS input runs. Holding those runs' "
            "frames concatenated, it is split between them; holding exactly one run's "
            "worth, the same frames are reused for every one of them (a repeated "
            "identical sweep). The frame count decides which"
        ),
    )
    required.add_argument(
        "-stim-downsample",
        "-stim_downsample",
        type=int,
        default=100,
        metavar="PIXELS",
        help=(
            "Block-average each aperture axis down to about this many pixels, using the "
            "nearest factor that divides evenly (1080 -> 108). Refinement cost scales "
            "with aperture area, so full-resolution apertures are ruinously slow. "
            "Use 0 to keep the native resolution"
        ),
    )
    required.add_argument("-prefix", required=True, help="Output prefix")

    model = parser.add_argument_group("pRF model")
    model.add_argument(
        "-tr", type=float, default=None, help="TR in seconds; defaults to the input header"
    )
    model.add_argument(
        "-screen-extent",
        "-screen_extent",
        "-screen-deg",
        "-screen_deg",
        dest="screen_extent",
        type=float,
        default=None,
        metavar="DEGREES",
        help=(
            "Degrees of visual angle spanned by the FULL WIDTH of the stimulus "
            "aperture (so half of it is the largest eccentricity reachable along "
            "the horizontal). Reports x, y, sigma, eccentricity, and rfsize in "
            "degrees instead of aperture pixels"
        ),
    )
    model.add_argument(
        "-hrf",
        "-hrf-mode",
        "-hrf_mode",
        dest="hrf_mode",
        choices=["canonical", "library", "pighs"],
        default="library",
        help=(
            "HRF source. 'canonical' is a single fixed double-gamma, which skips "
            "per-HRF selection entirely and is by far the fastest. 'library' is the "
            "20-HRF double-gamma family (or -hrf_library); pair it with "
            "-hrf_select grid+1. 'pighs' generates a half-cosine family of "
            "-num_hrfs shapes"
        ),
    )
    model.add_argument(
        "-num-hrfs",
        "-num_hrfs",
        "-n_hrfs",
        dest="num_hrfs",
        type=int,
        default=None,
        help=(
            "How many HRFs to fit. For -hrf library this evenly subsamples the "
            "library (fewer, more widely spaced shapes); for -hrf pighs it is the "
            "number generated. Defaults to the whole library / 20 PIGHS shapes"
        ),
    )
    model.add_argument(
        "-hrf-library",
        "-hrf_library",
        dest="hrf_library",
        default=None,
        help="Custom column-wise HRF TSV, e.g. from ffs_librarian (-hrf library only)",
    )
    model.add_argument(
        "-save-canonical",
        "-save_canonical",
        action="store_true",
        help=(
            "Also fit the canonical HRF and write it to {prefix}_canonical. The "
            "canonical is appended to the library, so this costs one extra "
            "refinement pass and lets the HRF-selected and fixed-HRF fits be "
            "compared voxelwise on identical data"
        ),
    )
    model.add_argument(
        "-hrf-duration", "-hrf_duration", type=float, default=32.0, help="HRF duration in seconds"
    )
    model.add_argument(
        "-grid-angles",
        "-grid_angles",
        dest="grid_angles",
        type=int,
        default=32,
        help=(
            "Polar angles in the super-grid. The highest-value density knob: "
            "doubling it from analyzePRF's 16 costs no measurable time (the grid "
            "stage is launch-bound, not candidate-bound) and measurably improves "
            "both the HRF choice and the seed position"
        ),
    )
    model.add_argument(
        "-grid-angle-mode",
        "-grid_angle_mode",
        dest="grid_angle_mode",
        choices=["uniform", "arc"],
        default="uniform",
        help=(
            "How -grid_angles is spread over the eccentricity rings. 'uniform' is "
            "analyzePRF's: the same count on every ring. 'arc' scales the count with "
            "ring radius for constant pixel spacing, so -grid_angles applies to the "
            "OUTERMOST ring. 'arc' sounds better but measured WORSE (82.0%% vs 84.2%% "
            "HRF agreement at matched cost): pRF size grows with eccentricity, so "
            "uniform angles are already evenly spaced in units of pRF width, and arc "
            "leaves mid-eccentricity rings with only 2-5 angles"
        ),
    )
    model.add_argument(
        "-grid-sigma-mode",
        "-grid_sigma_mode",
        dest="grid_sigma_mode",
        choices=["absolute", "slope"],
        default="absolute",
        help=(
            "How super-grid pRF sizes are sampled. 'absolute' is analyzePRF's fixed "
            "sigma ladder at every eccentricity, which spends candidates on sizes "
            "anatomy rules out (a 64-px pRF at the fovea). 'slope' samples the slope "
            "of the linear size-vs-eccentricity relationship instead, so each ring "
            "gets the sizes plausible there (-grid_sigma_steps does not apply). "
            "Measured: HRF agreement 84.2%% -> 85.1%% for 22%% more candidates"
        ),
    )
    model.add_argument(
        "-grid-sigma-steps",
        "-grid_sigma_steps",
        dest="grid_sigma_steps",
        type=int,
        default=1,
        help="Super-grid sigma samples per octave (1 = analyzePRF's powers of two)",
    )
    model.add_argument(
        "-grid-ecc-steps",
        "-grid_ecc_steps",
        dest="grid_ecc_steps",
        type=int,
        default=1,
        help="Super-grid rings between each pair of analyzePRF's reference eccentricities",
    )
    model.add_argument(
        "-candidate-chunk",
        "-candidate_chunk",
        type=int,
        default=256,
        help="Spatial/CSS candidates evaluated per GPU prediction batch",
    )
    model.add_argument(
        "-batch-size",
        "-batch_size",
        type=int,
        default=None,
        help="Voxel chunk size; defaults to the shared device-aware memory estimate",
    )
    model.add_argument(
        "-polort",
        type=int,
        default=None,
        help=(
            "Per-run Legendre polynomial degree; defaults to AFNI-style automatic "
            "selection. Use -1 for no drift terms at all (analyzePRF's NaN case)"
        ),
    )
    model.add_argument("-maxiter", type=int, default=50, help="Maximum CSS Gauss-Newton iterations")
    model.add_argument(
        "-expt-lower-bound",
        "-expt_lower_bound",
        type=float,
        default=1e-3,
        help="Lower bound on the CSS exponent (analyzePRF exptlowerbound)",
    )
    model.add_argument(
        "-hrf-select",
        "-hrf_select",
        choices=["refine", "grid", "grid+1", "grid+2"],
        default="refine",
        help=(
            "How the per-voxel HRF is chosen. 'refine' refits the pRF under every "
            "HRF and keeps the best. 'grid' keeps the super-grid's choice, which "
            "matches the refined one ~84%% of the time on well-fit voxels. 'grid+N' "
            "refits only the HRFs within N steps of the grid choice: grid+1 reaches "
            "~96%% agreement at equal R2 for well under half the time, and is the "
            "recommended setting. grid+2 measured identical to grid+1, because the "
            "grid choice is within one library step essentially always. The window "
            "slides at the library edges, so a voxel that picked HRF 1 is still "
            "scored on three HRFs"
        ),
    )
    model.add_argument(
        "-save-hrf-r2",
        "-save_hrf_r2",
        action="store_true",
        help=(
            "Write the full per-voxel x per-HRF R2 map to {prefix}_hrf_r2, the raw "
            "input to any HRF-selection criterion (-hrf_select refine only)"
        ),
    )
    model.add_argument(
        "-quick",
        action="store_true",
        help=(
            "Skip Gauss-Newton refinement and return the super-grid seeds "
            "(analyzePRF seedmode -2); disables cross-validation"
        ),
    )
    model.add_argument(
        "-gn-damping",
        "-gn_damping",
        type=float,
        default=1e-3,
        help="Initial relative Levenberg damping",
    )
    model.add_argument(
        "-gn-step-tol",
        "-gn_step_tol",
        type=float,
        default=1e-4,
        help="CSS parameter convergence tolerance",
    )
    model.add_argument(
        "-xval",
        choices=["auto", "none", "loro", "halves"],
        default="auto",
        help=(
            "Cross-validation scheme. 'loro' holds out one run at a time. 'halves' "
            "splits the runs into two disjoint halves that each cover every "
            "stimulus group (see -stim_groups) and fits both, which yields the "
            "held-out R2 AND split-half parameter reliability from the same fits. "
            "'auto' is LORO for multiple runs"
        ),
    )
    model.add_argument(
        "-stim-groups",
        "-stim_groups",
        dest="stim_groups",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "One label per -input run, in the same order, naming which stimulus "
            "set each run belongs to (e.g. '1 1 1 2 2 2' for three bar runs then "
            "three wedge runs). Runs in a group are complementary probes of the "
            "same receptive field, so -xval halves keeps every group represented "
            "in both halves rather than training on a deficient subset. Defaults "
            "to grouping runs whose aperture movies are identical"
        ),
    )
    model.add_argument(
        "-xval-draws",
        "-xval_draws",
        dest="xval_draws",
        type=int,
        default=2,
        help=(
            "How many independent half-splits -xval halves averages over. Groups "
            "with an odd number of runs leave one out per draw, and which one "
            "rotates with the draw, so more draws stop the estimate depending on "
            "whichever runs happened to sit out. Draws that repeat are skipped"
        ),
    )
    model.add_argument(
        "-reliability-threshold",
        "-reliability_threshold",
        dest="reliability_threshold",
        type=float,
        default=0.2,
        help=(
            "Full-fit R2 a voxel must reach to enter the -xval halves reliability "
            "summary. Reliability over unresponsive voxels measures nothing but "
            "noise agreement, so this threshold is part of the number and is "
            "reported alongside it"
        ),
    )
    model.add_argument(
        "-seed",
        type=int,
        default=0,
        help="Random seed for the screening tile basis and the -xval halves draws",
    )
    model.add_argument(
        "-xval-hrf",
        "-xval_hrf",
        dest="xval_hrf",
        choices=["grid", "fixed", "refine"],
        default="grid",
        help=(
            "How each cross-validation fold picks its HRF. 'grid' lets the fold's "
            "own super-grid choose: leak-free, but grid-level HRF choice is near "
            "chance, so xval_r2 scores a worse model than the other sub-bricks "
            "report. 'fixed' reuses the full fit's per-voxel HRF, which scores the "
            "reported model at no extra cost but leaks one discrete choice from the "
            "held-out run. 'refine' refits every HRF inside each fold: exact and "
            "leak-free, at roughly n_hrfs times the cross-validation cost"
        ),
    )
    model.add_argument(
        "-model-mode",
        "-model_mode",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "1=staged CSS (exponent frozen for the first half of the iterations), "
            "2=direct CSS, 3=fixed-exponent linear Gaussian pRF. Mode 3 is the "
            "classic Dumoulin-Wandell model and the natural comparison baseline; it "
            "is not meaningfully faster, since the per-voxel Gaussian over the "
            "aperture dominates every mode"
        ),
    )
    add_ortvec_arguments(model)

    screening = parser.add_argument_group("screening (fast linear pre-pass)")
    screening.add_argument(
        "-screen",
        type=float,
        default=None,
        metavar="R2",
        help=(
            "Fit a fast linear pRF (ridge onto a random hashed-Gaussian basis) to "
            "every voxel first, and refine only those whose cross-validated R2 "
            "exceeds this. Seconds for a whole brain, and CSS refinement cost is "
            "linear in voxels, so this is the big lever on unmasked data. Excluded "
            "voxels are written as NaN. Try 0.0 to keep anything the linear model "
            "predicts at all"
        ),
    )
    screening.add_argument(
        "-screen-top",
        "-screen_top",
        dest="screen_top",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Instead of a threshold, keep this fraction of the best-screening voxels",
    )
    screening.add_argument(
        "-save-screen",
        "-save_screen",
        action="store_true",
        help="Write the screening R2 map to {prefix}_screen -- a functionally derived mask",
    )
    screening.add_argument(
        "-denoise",
        action="store_true",
        help=(
            "Project data-derived noise components out of the fit (GLMdenoise "
            "style). The noise pool is the voxels the screening pass says carry no "
            "stimulus response at all; components are taken per run, and how many "
            "to keep is chosen by cross-validation -- including the option of zero, "
            "when denoising does not earn its degrees of freedom. Needs 2+ runs"
        ),
    )
    screening.add_argument(
        "-max-pcs",
        "-max_pcs",
        dest="max_pcs",
        type=int,
        default=10,
        help="Largest number of noise components to consider",
    )
    screening.add_argument(
        "-noise-pool-r2",
        "-noise_pool_r2",
        dest="noise_pool_r2",
        type=float,
        default=0.0,
        help=(
            "Screening R2 below which a voxel joins the noise pool. The default 0 "
            "takes only voxels the linear pRF model fits WORSE than their own mean, "
            "which is a strong statement that there is no stimulus response to "
            "leak into the components"
        ),
    )
    screening.add_argument(
        "-noise-pool-mask",
        "-noise_pool_mask",
        dest="noise_pool_mask",
        default=None,
        metavar="DSET|all",
        help=(
            "Draw the noise pool from this region instead of from -mask. Use it when "
            "the fit is deliberately local -- occipital only, say -- but the noise you "
            "want to remove is not: pass a whole-brain mask (or 'all' for the whole "
            "volume) and the components come from everywhere the screening pass sees "
            "no stimulus response, while the fit still happens only inside -mask. "
            "Voxels outside -mask are loaded but never fit"
        ),
    )
    screening.add_argument(
        "-denoise-tolerance",
        "-denoise_tolerance",
        dest="denoise_tolerance",
        type=float,
        default=0.05,
        help=(
            "Keep the fewest components within this fraction of the best "
            "cross-validated improvement, rather than the noisy argmax"
        ),
    )
    screening.add_argument(
        "-save-denoise",
        "-save_denoise",
        action="store_true",
        help=(
            "Write the noise-pool mask to {prefix}_noisepool, the components to "
            "{prefix}_noisepcs.1D, the component-count sweep to {prefix}_denoise.png, "
            "and per-component timecourse/spatial-weight figures to "
            "{prefix}_noisepc_PC*.png -- check those: a component landing on visual "
            "cortex means the pool is contaminated"
        ),
    )
    screening.add_argument(
        "-screen-tiles",
        "-screen_tiles",
        dest="screen_tiles",
        type=int,
        default=250,
        help="Random Gaussian tiles in the screening basis",
    )

    processing = parser.add_argument_group("processing")
    processing.add_argument("-mask", default=None, help="Optional brain mask")
    processing.add_argument(
        "-do-scale", "-do_scale", action="store_true", help="Scale to percent signal"
    )
    processing.add_argument("-device", default=None, help="Compute device, e.g. cuda or cpu")
    processing.add_argument(
        "-keep-on-cpu",
        "-keep_on_cpu",
        action="store_true",
        help="Stream CPU-held data to the compute device",
    )
    add_load_threads_arg(processing)
    add_verbose_arg(processing, default=1)
    return parser


def _natural_path_key(path: Path) -> list[int | str]:
    """Sort frame names numerically, so ``frame_2.png`` precedes ``frame_10.png``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_png_directory(path: Path) -> np.ndarray:
    """Load a naturally ordered PNG frame directory as a row x column x time movie."""
    frame_paths = sorted(
        (frame for frame in path.iterdir() if frame.is_file() and frame.suffix.lower() == ".png"),
        key=_natural_path_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in stimulus directory {path}")

    frames: list[np.ndarray] = []
    frame_shape: tuple[int, int] | None = None
    for frame_path in frame_paths:
        frame = mpimg.imread(frame_path)
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            frame = np.tensordot(
                frame[..., :3], np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
            )
        if frame.ndim != 2:
            raise ValueError(
                f"PNG stimulus frame {frame_path} must be grayscale or RGB/RGBA, got {frame.shape}"
            )
        if frame_shape is not None and frame.shape != frame_shape:
            raise ValueError(
                f"PNG stimulus frame {frame_path} has shape {frame.shape}; expected {frame_shape}"
            )
        frame_shape = frame.shape
        frames.append(np.asarray(frame, dtype=np.float32))

    return np.stack(frames, axis=-1)


def _load_stimulus_run(path: str, downsample: int = 0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load a TR-aligned aperture movie or PNG directory as time-by-pixel samples."""
    stimulus_path = Path(path)
    if stimulus_path.is_dir():
        movie = _load_png_directory(stimulus_path)
    elif stimulus_path.suffix.lower() == ".npy":
        movie = np.load(stimulus_path)
    else:
        movie = load_nifti(stimulus_path).get_fdata(dtype=np.float32)
    if movie.ndim == 4:
        # An aperture written as a NIfTI volume is row x column x 1 x time: the
        # slice axis only exists because NIfTI has no 2D-plus-time layout.
        singleton_axes = tuple(axis for axis in range(3) if movie.shape[axis] == 1)
        if len(singleton_axes) == 1:
            movie = movie.reshape(
                tuple(size for axis, size in enumerate(movie.shape) if axis != singleton_axes[0])
            )
    if movie.ndim != 3:
        raise ValueError(
            f"Stimulus {stimulus_path} must be a row x column x time movie, got {movie.shape}"
        )
    rows, columns, n_timepoints = movie.shape
    if min(rows, columns, n_timepoints) < 1:
        raise ValueError(f"Stimulus {stimulus_path} has an empty dimension: {movie.shape}")
    aperture = torch.from_numpy(np.ascontiguousarray(movie, dtype=np.float32))
    if downsample > 0:
        aperture, (rows, columns) = downsample_aperture(aperture, downsample)
    frames = aperture.permute(2, 0, 1).reshape(n_timepoints, rows * columns)
    return frames.contiguous(), (rows, columns)


def _load_stimulus_sources(
    sources: list[str | list[str]],
    run_lengths: list[int],
    downsample: int = 0,
    verbose: bool = False,
) -> tuple[list[torch.Tensor], tuple[int, int]]:
    """Load ordered PNG/NIfTI sources and map multi-run ones onto their input runs.

    A source claiming ``N_RUNS`` runs is either the runs' frames concatenated, or
    one run's frames to be reused by all of them -- the two cases retinotopy
    actually produces, and the frame count tells them apart unambiguously. A
    repeated identical sweep (the common design) is the reuse case.
    """
    runs: list[torch.Tensor] = []
    stimulus_shape: tuple[int, int] | None = None
    for source in sources:
        if isinstance(source, str):
            location, n_runs = source, 1
        else:
            location, n_runs_text = source
            try:
                n_runs = int(n_runs_text)
            except ValueError as error:
                raise ValueError(
                    f"Stimulus run count for {location} must be an integer, got {n_runs_text!r}"
                ) from error
        if n_runs < 1:
            raise ValueError(f"Stimulus run count for {location} must be positive, got {n_runs}")
        if len(runs) + n_runs > len(run_lengths):
            raise ValueError(
                f"Stimulus source {location} maps beyond the {len(run_lengths)} input runs"
            )

        frames, shape = _load_stimulus_run(location, downsample)
        if stimulus_shape is not None and shape != stimulus_shape:
            raise ValueError(
                f"Stimulus source {location} has shape {shape}; expected {stimulus_shape}"
            )
        stimulus_shape = shape
        mapped_lengths = run_lengths[len(runs) : len(runs) + n_runs]
        expected_frames = sum(mapped_lengths)
        n_frames = frames.shape[0]
        if n_frames == expected_frames:
            offset = 0
            for run_length in mapped_lengths:
                runs.append(frames[offset : offset + run_length].contiguous())
                offset += run_length
            if verbose and n_runs > 1:
                print(f"  {location}: splitting {n_frames} frames across {n_runs} runs")
        elif n_runs > 1 and all(length == n_frames for length in mapped_lengths):
            runs.extend(frames for _ in mapped_lengths)
            if verbose:
                print(f"  {location}: reusing the same {n_frames} frames for all {n_runs} runs")
        else:
            raise ValueError(
                f"Stimulus source {location} has {n_frames} frames, which is neither "
                f"{expected_frames} (the concatenated length of its {n_runs} mapped runs) "
                f"nor one run's length (runs are {mapped_lengths})"
            )

    if len(runs) != len(run_lengths):
        raise ValueError(
            f"Stimulus sources provide {len(runs)} runs but -input contains {len(run_lengths)} runs"
        )
    assert stimulus_shape is not None
    return runs, stimulus_shape


def _build_hrf_library(
    args: argparse.Namespace, tr: float, device: torch.device
) -> tuple[torch.Tensor, int | None]:
    """Build the HRF set to fit, and the index of the canonical HRF within it.

    The canonical is appended last rather than searched for in the library: the
    library's own entries are a parametric family that does not contain the
    canonical shape exactly, so "which library entry is the canonical" has no
    honest answer.
    """
    canonical = load_canonical_hrf_basic(
        microtime_dt=tr, hrf_duration=args.hrf_duration, device=device
    ).unsqueeze(0)
    if args.hrf_mode == "canonical":
        return canonical, 0

    if args.hrf_mode == "pighs":
        library = get_hrf_library(
            mode="pighs",
            microtime_dt=tr,
            hrf_duration=args.hrf_duration,
            n_hrfs=args.num_hrfs or 20,
            device=device,
        )
    else:
        library = load_canonical_hrf_library(
            microtime_dt=tr,
            hrf_duration=args.hrf_duration,
            device=device,
            library_path=args.hrf_library,
        )
        if args.num_hrfs is not None and args.num_hrfs < library.shape[0]:
            # The library is ordered by peak time, so an even stride keeps the
            # full timing range instead of truncating one end of it.
            keep = torch.linspace(0, library.shape[0] - 1, args.num_hrfs).round().long()
            library = library[keep.unique()]

    if not args.save_canonical:
        return library, None
    return torch.cat([library, canonical], dim=0), library.shape[0]


def _expand_to_all_voxels(values: torch.Tensor, keep: torch.Tensor, n_voxels: int) -> torch.Tensor:
    """Scatter a screened subset's results back into full voxel order."""
    full = torch.zeros((n_voxels, *values.shape[1:]), device=values.device, dtype=values.dtype)
    full[keep] = values
    return full


def _expand_fit(fit: PRFRefinedFit, keep: torch.Tensor, n_voxels: int) -> PRFRefinedFit:
    """Same, for every field of a refined fit."""
    return PRFRefinedFit(
        **{
            name: _expand_to_all_voxels(getattr(fit, name), keep, n_voxels)
            for name in (
                "candidate_index",
                "hrf_index",
                "parameters",
                "gain",
                "correlation",
                "r2",
                "residual_ss",
                "n_iters",
                "converged",
            )
        }
    )


def _add_noise_components(
    args: argparse.Namespace,
    data: torch.Tensor,
    screen_scores: torch.Tensor,
    invalid_voxels: torch.Tensor,
    stimulus_runs: list[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid,
    hrf_library: torch.Tensor,
    loaded,
    nuisance_per_run: list[torch.Tensor],
    device: torch.device,
    pool_candidates: torch.Tensor | None = None,
    fit_voxels: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """Pick a noise-component count by cross-validation and fold it into the nuisance.

    The screening pass has already said which voxels carry a stimulus response,
    so the pool is simply the ones that carry none -- by default the ones the
    linear pRF fits worse than their own mean. Components are extracted per run
    (``extract_noise_pcs_per_run``), which is what keeps this honest under
    cross-validation: a held-out run's regressors come from that run's own
    noise-pool voxels, never from the training runs and never from the voxel
    being scored.

    *pool_candidates* and *fit_voxels* are the two regions when they differ
    (``-noise_pool_mask``): the pool is drawn from the first, the sweep is scored
    on the second. Nothing about a noise component requires it to come from a
    voxel you intend to fit, and a fit mask tight enough to be worth drawing --
    occipital cortex, a single ROI -- is usually too tight to contain a
    representative pool.

    How many to keep is decided on the SUPER-GRID fit rather than the refined
    one. Refining the full model at every candidate count and every fold is tens
    of fits; the grid is about a second, and the question here is only which
    nuisance model predicts held-out data better, which does not need the
    refinement. The count is then used for the one real fit.
    """
    verbose = args.verb > 0
    pool = (screen_scores < args.noise_pool_r2) & torch.isfinite(screen_scores) & ~invalid_voxels
    if pool_candidates is not None:
        pool &= pool_candidates
    if int(pool.sum()) < 100:
        raise ValueError(
            f"noise pool has only {int(pool.sum())} voxels at -noise_pool_r2 "
            f"{args.noise_pool_r2:g}; raise the threshold"
        )
    # Score the sweep on the strongest responders, EXCLUDING the noise pool. A
    # voxel that helped build the components cannot also judge them: with a tight
    # mask the two sets otherwise overlap, and denoising is then scored partly on
    # its own input. A bounded set also keeps the sweep cheap.
    excluded = invalid_voxels | pool
    if fit_voxels is not None:
        excluded |= ~fit_voxels
    rankable = torch.where(excluded, torch.full_like(screen_scores, -torch.inf), screen_scores)
    n_criteria = min(int(torch.isfinite(rankable).sum()), 3000)
    if n_criteria < 10:
        raise ValueError(
            "fewer than 10 fit voxels sit outside the noise pool; -noise_pool_r2 is too high"
        )
    criteria = torch.topk(rankable, n_criteria).indices
    if verbose:
        print(
            f"Denoising: {int(pool.sum()):,} noise-pool voxels (screen R2 < "
            f"{args.noise_pool_r2:g}), scored on {n_criteria:,} responders"
        )

    components, loadings = extract_noise_pcs_per_run(
        data,
        list(loaded.run_starts),
        pool,
        max_components=args.max_pcs,
        nuisance_per_run=nuisance_per_run,
        return_loadings=True,
        device=device,
    )
    available = min(args.max_pcs, min(block.shape[1] for block in components))
    criteria_data = data[criteria]

    r2_by_count: list[torch.Tensor] = []
    counts = range(available + 1)
    if verbose:
        from tqdm.auto import tqdm

        counts = tqdm(counts, desc="pRF noise-PC sweep", leave=True)
    for n_components in counts:
        trial_nuisance = _append_components(nuisance_per_run, components, n_components)
        fit, _ = fit_prf_folds(
            criteria_data,
            stimulus_runs,
            stimulus_shape,
            grid,
            hrf_library,
            loaded.run_starts,
            loro_folds(len(stimulus_runs)),
            nuisance_per_run=trial_nuisance,
            candidate_chunk_size=args.candidate_chunk,
            voxel_chunk_size=n_criteria,
            refine=False,
            device=device,
        )
        r2_by_count.append(fit.r2.detach().cpu())

    median_r2 = [float(values.median()) for values in r2_by_count]
    chosen = select_noise_pc_count(median_r2, tolerance=args.denoise_tolerance)
    if verbose:
        curve = "  ".join(f"{n}:{value:.4f}" for n, value in enumerate(median_r2))
        print(f"  held-out median R2 by component count: {curve}")
        if chosen == 0:
            print("  keeping 0 components: denoising did not improve held-out R2")
        else:
            print(
                f"  keeping {chosen} components "
                f"(+{median_r2[chosen] - median_r2[0]:.4f} held-out median R2)"
            )
    if args.save_denoise:
        stem = parse_prefix(args.prefix).stem
        _plot_noise_sweep(r2_by_count, chosen, f"{stem}_denoise.png", n_criteria)
        _plot_noise_components(components, loadings, pool, chosen, available, loaded, stem)
    kept = [block[:, :chosen] for block in components]
    return _append_components(nuisance_per_run, components, chosen), pool, kept


def _plot_noise_components(
    components: list[torch.Tensor],
    loadings: list[torch.Tensor],
    pool: torch.Tensor,
    chosen: int,
    available: int,
    loaded,
    stem: str,
) -> None:
    """Per-component timecourse and spatial-weight figures, one file per component.

    Reuses ffs_denoise's ``plot_denoising_pcs``: the point of looking at these is
    to recognise a component as physiological, motion, or scanner drift -- and
    that judgement is spatial. A component whose weights sit in white matter and
    ventricles is what you want; one that lands on visual cortex means the pool
    is contaminated and the fit is having real response regressed out of it.
    """
    from fastfuncstuff.visualization import plot_denoising_pcs

    voxel_mask = loaded.mask_flat
    if voxel_mask is None:
        voxel_mask = np.ones(int(np.prod(loaded.volume_shape)), dtype=bool)
    figure_prefix = f"{stem}_noisepc"
    plot_denoising_pcs(
        noise_pcs_per_run=[block.cpu() for block in components],
        run_starts=list(loaded.run_starts),
        pc_weights_per_run=[weights.cpu().numpy() for weights in loadings],
        volume_shape=tuple(loaded.volume_shape),
        voxel_mask=voxel_mask,
        noise_pool_mask=pool.cpu().numpy(),
        n_pcs_to_show=min(available, max(chosen, 5)),
        tr=loaded.tr,
        optimal_n_pcs=chosen,
        output_prefix=figure_prefix,
        voxel_sizes=tuple(loaded.voxel_sizes),
        return_figs=False,
    )


def _plot_noise_sweep(
    r2_by_count: list[torch.Tensor], chosen: int, output_path: str, n_criteria: int
) -> None:
    """Plot held-out R2 against noise-component count, and the gain over none.

    Two panels for the same reason ffs_denoise uses two: the absolute curve says
    whether the fit is any good, and the paired difference from the undenoised
    model -- same voxels, same folds -- says whether denoising is what did it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = list(range(len(r2_by_count)))
    stacked = torch.stack(r2_by_count)
    median = stacked.median(dim=1).values.numpy()
    low = stacked.quantile(0.25, dim=1).numpy()
    high = stacked.quantile(0.75, dim=1).numpy()
    delta = stacked - stacked[0]

    figure, (axis, axis_delta) = plt.subplots(1, 2, figsize=(11, 4.2))
    line = axis.plot(counts, median, marker="o")[0]
    axis.fill_between(counts, low, high, alpha=0.15, color=line.get_color())
    axis.axvline(chosen, color="crimson", lw=1.2, ls="--", label=f"kept {chosen}")
    axis.set_xlabel("Noise components projected out")
    axis.set_ylabel(f"Median held-out R2 ({n_criteria:,} responders)")
    axis.set_title("Cross-validated fit vs component count")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis_delta.plot(counts, delta.median(dim=1).values.numpy(), marker="o", color="tab:green")
    axis_delta.fill_between(
        counts,
        delta.quantile(0.25, dim=1).numpy(),
        delta.quantile(0.75, dim=1).numpy(),
        alpha=0.15,
        color="tab:green",
    )
    axis_delta.axhline(0, color="0.5", lw=1.0)
    axis_delta.axvline(chosen, color="crimson", lw=1.2, ls="--")
    axis_delta.set_xlabel("Noise components projected out")
    axis_delta.set_ylabel("Change in held-out R2")
    axis_delta.set_title("Improvement over no denoising")
    axis_delta.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _append_components(
    nuisance_per_run: list[torch.Tensor], components: list[torch.Tensor], n_components: int
) -> list[torch.Tensor]:
    """Concatenate the first ``n_components`` noise PCs onto each run's nuisance block."""
    if n_components < 1:
        return nuisance_per_run
    return [
        torch.cat([block, run_components[:, :n_components].to(block.device, block.dtype)], dim=1)
        for block, run_components in zip(nuisance_per_run, components, strict=True)
    ]


def _invalid_voxels(data: torch.Tensor, run_starts: list[int]) -> torch.Tensor:
    """Match analyzePRF: reject non-finite voxels and voxels zero in any run."""
    invalid = ~torch.isfinite(data).all(dim=1)
    run_ends = [*run_starts[1:], data.shape[1]]
    for start, end in zip(run_starts, run_ends, strict=True):
        invalid |= data[:, start:end].eq(0).all(dim=1)
    return invalid


def _save_voxel_matrix(
    values: torch.Tensor,
    output_path: str,
    loaded,
    labels: list[str],
    invalid_voxels: torch.Tensor | None,
) -> None:
    """Save an ``(n_voxels, n_columns)`` matrix as a labeled 4D bucket."""
    array = values.cpu().numpy()
    if invalid_voxels is not None:
        array[invalid_voxels.cpu().numpy()] = np.nan
    if loaded.mask_flat is not None:
        full = np.zeros((loaded.mask_flat.size, array.shape[1]), dtype=array.dtype)
        full[loaded.mask_flat] = array
        array = full
    save_nifti(
        array.reshape((*loaded.volume_shape, len(labels))),
        output_path=output_path,
        affine=loaded.affine,
        header=loaded.nifti_header,
        brick_labels=labels,
    )


def _save_results(
    results,
    output_path: str,
    loaded,
    stimulus_shape: tuple[int, int],
    xval_r2: torch.Tensor | None = None,
    invalid_voxels: torch.Tensor | None = None,
    mean_volume: torch.Tensor | None = None,
    hrf_r2_map: torch.Tensor | None = None,
    screen_extent: float | None = None,
    noise_ceiling: torch.Tensor | None = None,
) -> None:
    """Save all primary pRF parameters in one labeled 4D NIfTI bucket."""
    # Positions are reported as x/y offsets from the aperture center (x right,
    # y up) rather than as the raw one-based row/column the optimizer works in.
    # With -screen_extent every spatial quantity is scaled to degrees of visual
    # angle by the same factor; pixel units are an artifact of the aperture
    # resolution and change under -stim_downsample, so they are not comparable
    # across studies.
    scale = 1.0 if screen_extent is None else screen_extent / float(max(stimulus_shape))
    maps = prf_parameter_maps(results.parameters, stimulus_shape, scale)
    labels = [
        "x",
        "y",
        "sigma",
        "exponent",
        "gain",
        "angle",
        "eccentricity",
        "rfsize",
        "correlation",
        "r2",
        "hrf_index",
        "grid_index",
        "residual_ss",
        "gn_iterations",
        "gn_converged",
    ]
    columns = [
        maps["x"],
        maps["y"],
        maps["sigma"],
        maps["exponent"],
        results.gain,
        maps["angle"],
        maps["eccentricity"],
        maps["rfsize"],
        results.correlation,
        results.r2,
        results.hrf_index + 1,
        results.candidate_index + 1,
        results.residual_ss,
        results.n_iters,
        results.converged,
    ]
    if xval_r2 is not None:
        labels.append("xval_r2")
        columns.append(xval_r2)
    if noise_ceiling is not None:
        # The ceiling is already in R2 units, so the normalised score is a plain
        # ratio: 1.0 means the model captured everything that reproduces at all.
        labels.append("noise_ceiling")
        columns.append(noise_ceiling)
        if xval_r2 is not None:
            labels.append("xval_r2_normalized")
            columns.append(xval_r2 / noise_ceiling.clamp_min(1e-6))
    if mean_volume is not None:
        # analyzePRF's results.meanvol - the scale the gain is expressed against,
        # and the reference for turning gain into a percent-signal-change.
        labels.append("meanvol")
        columns.append(mean_volume)
    if hrf_r2_map is not None:
        _, continuous_hrf, hrf_evidence = summarize_hrf_selection(hrf_r2_map)
        # hrf_evidence near zero means the library is indistinguishable for this
        # voxel: threshold on it before believing hrf_index.
        labels += ["hrf_index_continuous", "hrf_evidence"]
        columns += [continuous_hrf + 1, hrf_evidence]
    values = torch.column_stack(columns).cpu().numpy()
    if invalid_voxels is not None:
        # Blank the fit, but not the data. meanvol is a property of the input,
        # true everywhere, and it is the reference the gain is expressed against
        # -- screening out a voxel says nothing about its mean signal, so
        # blanking it would hand back a mean image with holes in it.
        excluded = invalid_voxels.cpu().numpy()
        fit_columns = [index for index, name in enumerate(labels) if name not in _DATA_COLUMNS]
        values[np.ix_(excluded, fit_columns)] = np.nan
    if loaded.mask_flat is not None:
        full = np.zeros((loaded.mask_flat.size, len(labels)), dtype=values.dtype)
        full[loaded.mask_flat] = values
        data = full.reshape((*loaded.volume_shape, len(labels)))
    else:
        data = values.reshape((*loaded.volume_shape, len(labels)))
    save_nifti(
        data,
        output_path=output_path,
        affine=loaded.affine,
        header=loaded.nifti_header,
        brick_labels=labels,
    )


# Parameters whose split-half agreement is worth reporting. Position is known to
# be reliable and is here as the control: a method that wrecks it is broken, and
# rfsize is the one the literature actually argues about.
_RELIABILITY_PARAMETERS = ("x", "y", "eccentricity", "angle", "sigma", "rfsize")


def _resolve_stim_groups(
    explicit: list[str] | None,
    stimulus_runs: list[torch.Tensor],
    parser: argparse.ArgumentParser,
) -> list[int]:
    """Map each run to a stimulus-group index, declared or inferred.

    Inferring from identical aperture movies covers the common case for free: a
    ``-stim-nii-multi bars.nii 3`` source hands the same frames to three runs, so
    those three are recognised as one group without anything being declared.
    Labels are positional against ``-input``, which is the same footgun that
    silently mispaired ``-events`` across sessions, so the resolved grouping is
    echoed rather than assumed.
    """
    if explicit is None:
        return identical_design_labels(stimulus_runs)
    if len(explicit) != len(stimulus_runs):
        parser.error(
            f"-stim_groups has {len(explicit)} labels but there are "
            f"{len(stimulus_runs)} input runs; labels pair with -input by position"
        )
    seen: dict[str, int] = {}
    return [seen.setdefault(label, len(seen)) for label in explicit]


def _noise_ceiling(
    data: torch.Tensor,
    stimulus_runs: list[torch.Tensor],
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor] | None,
    chunk_size: int,
) -> tuple[torch.Tensor | None, list[list[int]]]:
    """Per-voxel R2 ceiling from runs whose aperture movies are identical."""
    repeat_groups = identical_design_groups(stimulus_runs)
    if not repeat_groups:
        return None, []
    from fastfuncstuff.design.prf import project_nuisance_per_run

    ceilings = []
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start : start + chunk_size]
        ceilings.append(
            split_half_noise_ceiling(
                project_nuisance_per_run(chunk, nuisance_per_run, run_starts),
                repeat_groups,
                run_starts,
                data.shape[1],
            )
        )
    return torch.cat(ceilings), repeat_groups


def _angular_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Smallest absolute difference between two angles in degrees."""
    return ((first - second + 180.0).remainder(360.0) - 180.0).abs()


def _half_parameter_pairs(
    fold_fits: list[PRFRefinedFit],
    stimulus_shape: tuple[int, int],
    scale: float,
) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    """Derived parameters for the two disjoint halves of each draw.

    ``fold_fits`` arrives two per draw, the first trained on half A and the
    second on half B, so consecutive pairs are exactly the disjoint fits that
    reliability needs. Deriving the maps once here is what makes sweeping a
    range of thresholds free: every threshold reuses these and only recomputes
    correlations over a different voxel subset.
    """
    return [
        (
            prf_parameter_maps(first.parameters, stimulus_shape, scale),
            prf_parameter_maps(second.parameters, stimulus_shape, scale),
        )
        for first, second in zip(fold_fits[0::2], fold_fits[1::2], strict=True)
    ]


def _reliability_at(
    pairs: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    responsive: torch.Tensor,
) -> dict[str, float]:
    """Mean split-half correlation per parameter over one voxel subset."""
    summary: dict[str, float] = {}
    for name in _RELIABILITY_PARAMETERS:
        values = []
        for first, second in pairs:
            left, right = first[name], second[name]
            usable = responsive & torch.isfinite(left) & torch.isfinite(right)
            if int(usable.sum()) < 2:
                continue
            if name == "angle":
                # Polar angle wraps, so a linear correlation would call two
                # estimates either side of 0/360 completely inconsistent.
                value = circular_correlation(
                    torch.deg2rad(left[usable]), torch.deg2rad(right[usable])
                )
            else:
                value = spearman_correlation(left[usable], right[usable])
            if value == value:
                values.append(value)
        summary[name] = sum(values) / len(values) if values else float("nan")
    return summary


def _reliability_curve(
    pairs: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    r2: torch.Tensor,
    *,
    step: float = 0.05,
    min_voxels: int = 50,
) -> list[dict[str, float]]:
    """Sweep the responsiveness threshold and report reliability at each level.

    A single threshold hides the shape of the answer. Reliability of pRF size
    typically climbs steeply with data quality while position is already flat,
    and one number cannot show that -- nor whether the number you quoted sits on
    a plateau or on a cliff. The fits are already done, so the whole curve costs
    only a few rank correlations per threshold.

    The sweep stops once fewer than ``min_voxels`` voxels survive, because a
    correlation over a handful of voxels is noise with a decimal point on it.
    """
    thresholds = []
    value = step
    while value < 1.0:
        thresholds.append(round(value, 4))
        value += step

    curve = []
    for threshold in thresholds:
        responsive = torch.isfinite(r2) & (r2 >= threshold)
        n_voxels = int(responsive.sum())
        if n_voxels < min_voxels:
            break
        row: dict[str, float] = {"threshold": threshold, "n_voxels": n_voxels}
        row.update(_reliability_at(pairs, responsive))
        curve.append(row)
    return curve


def _reliability_deltas(
    pairs: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    """Mean absolute between-half disagreement per voxel, averaged over draws."""
    deltas: dict[str, torch.Tensor] = {}
    for name in _RELIABILITY_PARAMETERS:
        stacked = []
        for first, second in pairs:
            if name == "angle":
                stacked.append(_angular_difference(first[name], second[name]))
            else:
                stacked.append((first[name] - second[name]).abs())
        deltas[name] = torch.stack(stacked).mean(dim=0)
    return deltas


def _plot_reliability_curve(
    curve: list[dict[str, float]],
    chosen: float,
    output_path: str,
    unit: str,
    n_draws: int,
) -> None:
    """Plot split-half reliability against the responsiveness threshold.

    The voxel count is drawn on a second axis because the two have to be read
    together: reliability that keeps climbing as the threshold rises is partly
    just a shrinking, better-behaved sample, and without the count there is no
    way to see where that takes over.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in curve]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for name in _RELIABILITY_PARAMETERS:
        values = [spearman_brown(row[name]) for row in curve]
        axis.plot(thresholds, values, marker="o", ms=3.5, lw=1.4, label=name)

    nearest = min(curve, key=lambda row: abs(row["threshold"] - chosen), default=None)
    if nearest is not None:
        axis.axvline(chosen, color="crimson", lw=1.2, ls="--")
        annotation = "  ".join(
            f"{name}={spearman_brown(nearest[name]):.2f}" for name in ("rfsize", "eccentricity")
        )
        axis.annotate(
            f"R2>={chosen:g}  n={int(nearest['n_voxels']):,}\n{annotation}",
            xy=(chosen, 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color="crimson",
            ha="left" if chosen < (thresholds[-1] + thresholds[0]) / 2 else "right",
        )

    axis.set_xlabel("Full-fit R2 threshold")
    axis.set_ylabel("Split-half reliability (Spearman-Brown corrected)")
    axis.set_title(f"pRF parameter reliability vs data quality ({n_draws} half-split draws)")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower right", ncol=2)

    count_axis = axis.twinx()
    count_axis.plot(
        thresholds,
        [row["n_voxels"] for row in curve],
        color="0.55",
        lw=1.0,
        ls=":",
    )
    count_axis.set_ylabel(f"Voxels surviving ({unit} maps)", color="0.45", fontsize=9)
    count_axis.set_yscale("log")
    count_axis.tick_params(axis="y", labelcolor="0.45", labelsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_reliability(
    curve: list[dict[str, float]],
    deltas: dict[str, torch.Tensor],
    tsv_path: str,
    map_path: str,
    loaded,
    invalid_voxels: torch.Tensor | None,
) -> None:
    """Write the full reliability curve and the per-voxel disagreement maps.

    The table holds every threshold, not just the reported one, so the choice of
    threshold stays auditable after the fact.
    """
    columns = ["threshold", "n_voxels", *_RELIABILITY_PARAMETERS]
    with open(tsv_path, "w") as handle:
        handle.write("\t".join([*columns, *(f"{name}_sb" for name in _RELIABILITY_PARAMETERS)]))
        handle.write("\n")
        for row in curve:
            values = [
                f"{row['threshold']:g}",
                f"{int(row['n_voxels'])}",
                *(f"{row[name]:.6f}" for name in _RELIABILITY_PARAMETERS),
                *(f"{spearman_brown(row[name]):.6f}" for name in _RELIABILITY_PARAMETERS),
            ]
            handle.write("\t".join(values) + "\n")
    _save_voxel_matrix(
        torch.column_stack([deltas[name] for name in _RELIABILITY_PARAMETERS]),
        map_path,
        loaded,
        [f"abs_delta_{name}" for name in _RELIABILITY_PARAMETERS],
        invalid_voxels,
    )


def _resolve_masks(
    args: argparse.Namespace, reference_file: str, parser: argparse.ArgumentParser
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Work out the fit region, the noise-pool region, and what to load.

    Without ``-noise_pool_mask`` the three are one thing and this returns Nones,
    leaving the loader to apply ``-mask`` itself. With it, the data has to cover
    the union so the pool can reach outside the fit mask, and the caller needs
    both flat masks to tell the two regions apart afterwards.
    """
    if not args.noise_pool_mask:
        return None, None, None

    from fastfuncstuff.io.afni import load_afni_mask, nifti_shape

    volume_shape = tuple(nifti_shape(reference_file)[:3])
    n_volume_voxels = int(np.prod(volume_shape))

    def _load(path: str) -> np.ndarray:
        mask = load_afni_mask(path)
        if tuple(mask.shape[:3]) != volume_shape:
            parser.error(f"Mask {path} has shape {tuple(mask.shape[:3])}; data is {volume_shape}")
        return mask.reshape(-1).astype(bool)

    fit_flat = _load(args.mask) if args.mask else np.ones(n_volume_voxels, dtype=bool)
    if args.noise_pool_mask.lower() in {"all", "full", "volume"}:
        pool_flat = np.ones(n_volume_voxels, dtype=bool)
    else:
        pool_flat = _load(args.noise_pool_mask)
    if not fit_flat.any():
        parser.error(f"-mask {args.mask} is empty")
    if not pool_flat.any():
        parser.error(f"-noise_pool_mask {args.noise_pool_mask} is empty")
    return fit_flat, pool_flat, fit_flat | pool_flat


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    prefix_info = parse_prefix(args.prefix)
    input_files = parse_input_files(args.input)
    if bool(args.stimulus) == bool(args.stim_sources):
        parser.error(
            "Specify exactly one of -stimulus or -stim-pngs/-stim-pngs-multi/"
            "-stim-nii/-stim-nii-multi"
        )
    if args.stimulus is not None and len(args.stimulus) != len(input_files):
        parser.error("-stimulus must provide exactly one aperture source per -input run")
    if args.denoise and len(input_files) < 2:
        parser.error("-denoise needs at least two runs to cross-validate the component count")
    if args.max_pcs < 1:
        parser.error("-max_pcs must be positive")
    if args.noise_pool_mask and not args.denoise:
        parser.error("-noise_pool_mask applies only to -denoise")
    if (
        args.noise_pool_mask
        and args.noise_pool_mask.lower() not in {"all", "full", "volume"}
        and not Path(args.noise_pool_mask).exists()
    ):
        parser.error(f"-noise_pool_mask does not exist: {args.noise_pool_mask}")
    if not 0 <= args.denoise_tolerance < 1:
        parser.error("-denoise_tolerance must be in [0, 1)")
    if args.screen is not None and args.screen_top is not None:
        parser.error("Specify at most one of -screen and -screen_top")
    if args.screen_top is not None and not 0 < args.screen_top <= 1:
        parser.error("-screen_top must be a fraction in (0, 1]")
    if args.screen_tiles < 1:
        parser.error("-screen_tiles must be positive")
    if args.grid_angles < 1 or args.grid_sigma_steps < 1 or args.grid_ecc_steps < 1:
        parser.error("-grid_angles, -grid_sigma_steps, and -grid_ecc_steps must be positive")
    if args.screen_extent is not None and args.screen_extent <= 0:
        parser.error("-screen_extent must be positive")
    if args.num_hrfs is not None and args.num_hrfs < 1:
        parser.error("-num_hrfs must be positive")
    if args.hrf_library and args.hrf_mode != "library":
        parser.error("-hrf_library applies only to -hrf library")
    if args.stim_downsample < 0:
        parser.error("-stim-downsample must be non-negative (0 keeps the native resolution)")
    if args.candidate_chunk < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("-candidate-chunk and -batch-size must be positive")
    if args.maxiter < 1 or args.gn_damping <= 0 or args.gn_step_tol <= 0:
        parser.error("-maxiter, -gn-damping, and -gn-step-tol must be positive")
    if args.xval in ("loro", "halves") and len(input_files) < 2:
        parser.error(f"-xval {args.xval} requires at least two input runs")
    if args.xval_draws < 1:
        parser.error("-xval_draws must be positive")
    if args.stim_groups is not None and len(args.stim_groups) != len(input_files):
        parser.error(
            f"-stim_groups has {len(args.stim_groups)} labels but -input has "
            f"{len(input_files)} runs; labels pair with -input by position"
        )
    # Check the aperture paths before the fMRI load, which can take minutes.
    for source in args.stimulus or []:
        if not Path(source).exists():
            parser.error(f"-stimulus source does not exist: {source}")
    for source in args.stim_sources:
        location = source if isinstance(source, str) else source[0]
        if not Path(location).exists():
            parser.error(f"Stimulus source does not exist: {location}")

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    # The loaded region is the union of where we fit and where the noise pool may
    # come from; which of the two a loaded voxel belongs to is tracked below.
    fit_mask_flat, pool_mask_flat, load_mask_flat = _resolve_masks(args, input_files[0], parser)
    loaded = load_and_preprocess_runs(
        input_files,
        tr=args.tr,
        mask_file=None if load_mask_flat is not None else args.mask,
        mask_array=load_mask_flat,
        do_scale=args.do_scale,
        device=device,
        force_cpu=args.keep_on_cpu,
        verbose=args.verb > 0,
        load_threads=args.load_threads,
    )
    run_lengths = compute_run_lengths(loaded.run_starts, loaded.n_timepoints)
    fit_voxels = pool_candidates = None
    if load_mask_flat is not None:
        assert fit_mask_flat is not None and pool_mask_flat is not None
        # Same device as the data, which is where the screening scores land and so
        # where every mask these are combined with lives.
        selected = load_mask_flat
        voxel_device = loaded.data.device
        fit_voxels = torch.from_numpy(fit_mask_flat[selected]).to(voxel_device)
        pool_candidates = torch.from_numpy(pool_mask_flat[selected]).to(voxel_device)
        if args.verb:
            print(
                f"Loaded {int(selected.sum()):,} voxels: {int(fit_voxels.sum()):,} to fit, "
                f"{int(pool_candidates.sum()):,} eligible for the noise pool"
            )
    invalid_voxels = _invalid_voxels(loaded.data, loaded.run_starts)
    analysis_data = torch.nan_to_num(loaded.data, nan=0.0, posinf=0.0, neginf=0.0)
    if args.verb and invalid_voxels.any():
        print(f"Excluding {invalid_voxels.sum().item():,} non-finite or all-zero voxels")
    if args.stim_sources:
        stimulus_runs, stimulus_shape = _load_stimulus_sources(
            args.stim_sources, run_lengths, args.stim_downsample, verbose=args.verb > 0
        )
    else:
        stimulus_runs = []
        stimulus_shape: tuple[int, int] | None = None
        for input_file, stimulus_file, run_length in zip(
            input_files, args.stimulus, run_lengths, strict=True
        ):
            stimulus, shape = _load_stimulus_run(stimulus_file, args.stim_downsample)
            if stimulus.shape[0] != run_length:
                raise ValueError(
                    f"Stimulus {stimulus_file} has {stimulus.shape[0]} frames but {input_file} has {run_length} volumes"
                )
            if stimulus_shape is not None and shape != stimulus_shape:
                raise ValueError(
                    f"Stimulus {stimulus_file} has shape {shape}; expected {stimulus_shape}"
                )
            stimulus_runs.append(stimulus)
            stimulus_shape = shape
        assert stimulus_shape is not None
    stim_groups = _resolve_stim_groups(args.stim_groups, stimulus_runs, parser)
    if args.verb:
        print(f"Aperture resolution: {stimulus_shape[0]} x {stimulus_shape[1]} pixels")
        if args.screen_extent:
            degrees_per_pixel = args.screen_extent / max(stimulus_shape)
            print(
                f"Visual angle: {args.screen_extent:g} deg full width "
                f"({degrees_per_pixel:.4f} deg/pixel, "
                f"{args.screen_extent / 2:g} deg maximum eccentricity)"
            )
    polort = args.polort
    if polort is None:
        polort = auto_polort(get_average_run_duration(run_lengths, loaded.tr))
    nuisance_blocks = collect_nuisance_blocks(
        args, loaded.run_starts, loaded.n_timepoints, verbose=args.verb > 1
    )
    nuisance_per_run = build_nuisance_per_run(
        loaded.run_starts,
        loaded.n_timepoints,
        polort,
        device,
        blocks=nuisance_blocks,
        verbose=args.verb > 1,
    )
    hrf_library, canonical_index = _build_hrf_library(args, loaded.tr, device)

    screen_scores = None
    keep_index = None
    noise_pool = None
    noise_components: list[torch.Tensor] = []
    if args.screen is not None or args.screen_top is not None or args.denoise:
        screen_scores = screen_voxels_ridge(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            load_canonical_hrf_basic(
                microtime_dt=loaded.tr, hrf_duration=args.hrf_duration, device=device
            ),
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            n_tiles=args.screen_tiles,
            voxel_chunk_size=estimate_chunk_size(
                n_voxels=loaded.n_voxels,
                n_timepoints=loaded.n_timepoints,
                n_regressors=args.screen_tiles,
                device=device,
                operation="xval",
            ),
            seed=args.seed,
            device=device,
            verbose=args.verb > 0,
        )
    if args.denoise:
        grid_for_sweep = make_analyzeprf_grid(
            stimulus_shape,
            exponents=(1.0,) if args.model_mode == 3 else (0.5, 0.25, 0.125),
            n_angles=args.grid_angles,
            angle_mode=args.grid_angle_mode,
            sigma_mode=args.grid_sigma_mode,
            sigma_steps_per_octave=args.grid_sigma_steps,
            eccentricity_steps=args.grid_ecc_steps,
            device=device,
        )
        nuisance_per_run, noise_pool, noise_components = _add_noise_components(
            args,
            analysis_data,
            screen_scores,
            invalid_voxels,
            stimulus_runs,
            stimulus_shape,
            grid_for_sweep,
            hrf_library,
            loaded,
            nuisance_per_run,
            device,
            pool_candidates=pool_candidates,
            fit_voxels=fit_voxels,
        )

    if args.screen is not None or args.screen_top is not None:
        assert screen_scores is not None
        # Rank only over voxels that could be fit at all, so the fraction is a
        # fraction of real data rather than of background. Voxels loaded solely to
        # feed the noise pool are not candidates either.
        unfittable = invalid_voxels if fit_voxels is None else invalid_voxels | ~fit_voxels
        rankable = torch.where(
            unfittable, torch.full_like(screen_scores, -torch.inf), screen_scores
        )
        if args.screen_top is not None:
            n_valid = int((~unfittable).sum())
            n_keep = max(1, min(n_valid, int(round(args.screen_top * n_valid))))
            threshold = torch.topk(rankable, n_keep).values.min().item()
        else:
            threshold = args.screen
        keep = (rankable >= threshold) & torch.isfinite(rankable)
        if not bool(keep.any()):
            parser.error(f"screening kept no voxels at threshold {threshold:g}")
        keep_index = torch.nonzero(keep, as_tuple=False).squeeze(1)
        analysis_data = analysis_data[keep_index]
        if args.verb:
            print(
                f"Screening kept {keep_index.numel():,} of {loaded.n_voxels:,} voxels "
                f"({100 * keep_index.numel() / loaded.n_voxels:.1f}%) at R2 >= {threshold:g}"
            )
    elif fit_voxels is not None and not bool(fit_voxels.all()):
        # No screening threshold, but part of what was loaded exists only to supply
        # noise components. Reuse the screening subset machinery so those voxels are
        # never fit and come back NaN rather than as a fit of the wrong region.
        keep_index = torch.nonzero(fit_voxels, as_tuple=False).squeeze(1)
        analysis_data = analysis_data[keep_index]

    n_fit_voxels = analysis_data.shape[0]
    grid = make_analyzeprf_grid(
        stimulus_shape,
        exponents=(1.0,) if args.model_mode == 3 else (0.5, 0.25, 0.125),
        n_angles=args.grid_angles,
        angle_mode=args.grid_angle_mode,
        sigma_mode=args.grid_sigma_mode,
        sigma_steps_per_octave=args.grid_sigma_steps,
        eccentricity_steps=args.grid_ecc_steps,
        device=device,
    )
    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    voxel_chunk_size = args.batch_size or estimate_chunk_size(
        n_voxels=n_fit_voxels,
        n_timepoints=loaded.n_timepoints,
        n_regressors=args.candidate_chunk,
        device=device,
        operation="xval",
    )
    # Refinement is dominated by the per-voxel Gaussian over the aperture, not by
    # the design, so it needs its own (much smaller) chunk than the grid search.
    refine_chunk_size = args.batch_size or estimate_chunk_size(
        n_voxels=n_fit_voxels,
        n_timepoints=loaded.n_timepoints,
        n_regressors=n_pixels,
        device=device,
        operation="prf",
        min_chunk_size=1,
    )
    if args.verb:
        print(
            f"Fitting {n_fit_voxels:,} voxels against {grid.n_candidates:,} CSS candidates "
            f"and {hrf_library.shape[0]} {args.hrf_mode} "
            f"HRF{'s' if hrf_library.shape[0] > 1 else ''} on {device}."
            + (" (last is the canonical)" if canonical_index and args.save_canonical else "")
        )
        print(
            f"Chunk sizes: {voxel_chunk_size:,} voxels (grid), "
            f"{refine_chunk_size:,} voxels (refinement, {n_pixels:,}-pixel aperture)"
        )
    grid_results = fit_prf_supergrid(
        analysis_data,
        stimulus_runs,
        stimulus_shape,
        grid,
        hrf_library,
        loaded.run_starts,
        nuisance_per_run=nuisance_per_run,
        candidate_chunk_size=args.candidate_chunk,
        voxel_chunk_size=voxel_chunk_size,
        device=device,
        verbose=args.verb > 0,
    )
    refinement_config = PRFRefinementConfig(
        max_iter=args.maxiter,
        damping=args.gn_damping,
        step_tolerance=args.gn_step_tol,
        min_exponent=args.expt_lower_bound,
        fix_exponent=args.model_mode == 3,
        stagewise_exponent=args.model_mode == 1,
    )
    hrf_r2_map = None
    canonical_results = None
    if args.quick:
        results = grid_seeds_as_fit(grid_results)
    elif args.hrf_select == "refine" and hrf_library.shape[0] > 1:
        results, hrf_r2_map, canonical_results = refine_prf_all_hrfs(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            keep_hrf_index=canonical_index,
            verbose=args.verb > 0,
        )
    elif args.hrf_select.startswith("grid+") and hrf_library.shape[0] > 1:
        results = refine_prf_hrf_window(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            window=int(args.hrf_select.removeprefix("grid+")),
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            verbose=args.verb > 0,
        )
    else:
        results = refine_prf_supergrid(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            verbose=args.verb > 0,
        )
    scheme = args.xval
    if scheme == "auto":
        scheme = "loro" if len(input_files) > 1 else "none"
    if args.quick:
        scheme = "none"
    xval_r2 = None
    reliability_curve: list[dict[str, float]] | None = None
    reliability_deltas: dict[str, torch.Tensor] | None = None
    n_reliability_draws = 0
    if scheme != "none":
        if scheme == "halves":
            draws = balanced_group_halves(stim_groups, n_draws=args.xval_draws, seed=args.seed)
            if not draws:
                parser.error(
                    "-xval halves needs at least one stimulus group with two or more "
                    "runs; every group here has a single run, so no two disjoint "
                    "halves can both cover it. Use -xval loro, or declare coarser "
                    "groups with -stim_groups"
                )
            singletons = {label for label in set(stim_groups) if stim_groups.count(label) < 2}
            if singletons and args.verb:
                print(
                    f"Note: {len(singletons)} stimulus group(s) have a single run and are "
                    "excluded from the split-half analysis entirely"
                )
            # Each draw contributes both directions, so every timepoint is held
            # out exactly once per draw and the pooled R2 stays a whole-dataset
            # number rather than an average over partial coverage.
            folds = [fold for first, second in draws for fold in ((first, second), (second, first))]
            if args.verb:
                print(
                    f"Cross-validating over {len(draws)} balanced half-split draw(s) "
                    f"({len(folds)} fold fits); groups: {stim_groups}"
                )
        else:
            folds = loro_folds(len(stimulus_runs))
        xval_results, fold_fits = fit_prf_folds(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid,
            hrf_library,
            loaded.run_starts,
            folds,
            nuisance_per_run=nuisance_per_run,
            candidate_chunk_size=args.candidate_chunk,
            voxel_chunk_size=voxel_chunk_size,
            refine_chunk_size=refine_chunk_size,
            device=device,
            refinement_config=refinement_config,
            hrf_mode=args.xval_hrf,
            fixed_hrf_index=results.hrf_index if args.xval_hrf == "fixed" else None,
            return_fits=scheme == "halves",
            verbose=args.verb > 0,
        )
        xval_r2 = xval_results.r2
        if scheme == "halves":
            scale = (
                1.0
                if args.screen_extent is None
                else args.screen_extent / float(max(stimulus_shape))
            )
            pairs = _half_parameter_pairs(fold_fits, stimulus_shape, scale)
            n_reliability_draws = len(pairs)
            reliability_deltas = _reliability_deltas(pairs)
            # Thresholded on the FULL fit, which both halves share. Selecting on
            # either half's own R2 would keep the voxels that half happened to
            # fit well and bias the agreement upward.
            reliability_curve = _reliability_curve(pairs, results.r2)
            if not reliability_curve:
                if args.verb:
                    print("No reliability curve: too few voxels survive any R2 threshold")
            elif args.verb:
                responsive = torch.isfinite(results.r2) & (results.r2 >= args.reliability_threshold)
                chosen = _reliability_at(pairs, responsive)
                print(
                    f"Split-half reliability over {int(responsive.sum()):,} voxels "
                    f"at R2 >= {args.reliability_threshold:g}, from "
                    f"{n_reliability_draws} draw(s); half-length (Spearman-Brown):"
                )
                for name in _RELIABILITY_PARAMETERS:
                    print(f"  {name:>12}: {chosen[name]:.3f}  ({spearman_brown(chosen[name]):.3f})")

    ceiling_chunk = refine_chunk_size if refine_chunk_size > 0 else 20_000
    noise_ceiling, repeat_groups = _noise_ceiling(
        analysis_data,
        stimulus_runs,
        list(loaded.run_starts),
        nuisance_per_run,
        max(1, ceiling_chunk),
    )
    if args.verb:
        if noise_ceiling is None:
            print(
                "No noise ceiling: no two runs share a bit-identical aperture movie, "
                "so there are no repeats to estimate reproducible variance from"
            )
        else:
            print(
                f"Noise ceiling from {len(repeat_groups)} repeated design(s): "
                f"median {float(noise_ceiling.nanmedian()):.3f}"
            )
    if keep_index is not None:
        results = _expand_fit(results, keep_index, loaded.n_voxels)
        if hrf_r2_map is not None:
            hrf_r2_map = _expand_to_all_voxels(hrf_r2_map, keep_index, loaded.n_voxels)
        if xval_r2 is not None:
            xval_r2 = _expand_to_all_voxels(xval_r2, keep_index, loaded.n_voxels)
        if noise_ceiling is not None:
            noise_ceiling = _expand_to_all_voxels(
                noise_ceiling.unsqueeze(1), keep_index, loaded.n_voxels
            ).squeeze(1)
        if reliability_deltas is not None:
            reliability_deltas = {
                name: _expand_to_all_voxels(
                    values.unsqueeze(1), keep_index, loaded.n_voxels
                ).squeeze(1)
                for name, values in reliability_deltas.items()
            }
        if canonical_results is not None:
            canonical_results = _expand_fit(canonical_results, keep_index, loaded.n_voxels)
        # Screened-out voxels were never fit, so they are reported as NaN rather
        # than as zeros that look like a fit that failed.
        screened_out = torch.ones_like(invalid_voxels)
        screened_out[keep_index] = False
        invalid_voxels = invalid_voxels | screened_out
    _save_results(
        results,
        f"{prefix_info.stem}{prefix_info.nifti_ext}",
        loaded,
        stimulus_shape,
        xval_r2=xval_r2,
        invalid_voxels=invalid_voxels,
        mean_volume=loaded.data.mean(dim=1),
        hrf_r2_map=hrf_r2_map,
        screen_extent=args.screen_extent,
        noise_ceiling=noise_ceiling,
    )
    if args.verb:
        print(f"Wrote pRF results: {prefix_info.stem}{prefix_info.nifti_ext}")
    if reliability_curve and reliability_deltas is not None:
        tsv_path = f"{prefix_info.stem}_reliability.tsv"
        map_path = f"{prefix_info.stem}_reliability{prefix_info.nifti_ext}"
        _save_reliability(
            reliability_curve,
            reliability_deltas,
            tsv_path,
            map_path,
            loaded,
            invalid_voxels,
        )
        figure_path = f"{prefix_info.stem}_reliability.png"
        _plot_reliability_curve(
            reliability_curve,
            args.reliability_threshold,
            figure_path,
            "deg" if args.screen_extent else "px",
            n_reliability_draws,
        )
        if args.verb:
            print(f"Wrote reliability curve: {tsv_path}, {figure_path}")
            print(f"Wrote per-voxel half-to-half disagreement: {map_path}")
    if canonical_results is not None:
        canonical_path = f"{prefix_info.stem}_canonical{prefix_info.nifti_ext}"
        _save_results(
            canonical_results,
            canonical_path,
            loaded,
            stimulus_shape,
            invalid_voxels=invalid_voxels,
            mean_volume=loaded.data.mean(dim=1),
            screen_extent=args.screen_extent,
        )
        if args.verb:
            print(f"Wrote fixed-canonical-HRF results: {canonical_path}")
    elif args.save_canonical and args.verb and args.hrf_mode != "canonical":
        print("-save_canonical ignored: needs -hrf_select refine")
    if args.save_denoise and noise_pool is not None:
        pool_path = f"{prefix_info.stem}_noisepool{prefix_info.nifti_ext}"
        _save_voxel_matrix(
            noise_pool.to(torch.float32).unsqueeze(1), pool_path, loaded, ["noise_pool"], None
        )
        if args.verb:
            print(f"Wrote noise pool: {pool_path}")
        if noise_components and noise_components[0].shape[1] > 0:
            components_path = f"{prefix_info.stem}_noisepcs.1D"
            np.savetxt(
                components_path,
                torch.cat(noise_components, dim=0).cpu().numpy(),
                fmt="%.8g",
            )
            if args.verb:
                print(f"Wrote noise components: {components_path}")
    elif args.save_denoise and args.verb:
        print("-save_denoise ignored: needs -denoise")
    if args.save_screen and screen_scores is not None:
        _save_voxel_matrix(
            screen_scores.unsqueeze(1),
            f"{prefix_info.stem}_screen{prefix_info.nifti_ext}",
            loaded,
            ["screen_r2"],
            None,
        )
        if args.verb:
            print(f"Wrote screening map: {prefix_info.stem}_screen{prefix_info.nifti_ext}")
    elif args.save_screen and args.verb:
        print("-save_screen ignored: needs -screen or -screen_top")
    if args.save_hrf_r2 and hrf_r2_map is not None:
        _save_voxel_matrix(
            hrf_r2_map,
            f"{prefix_info.stem}_hrf_r2{prefix_info.nifti_ext}",
            loaded,
            [f"hrf{index + 1:02d}" for index in range(hrf_r2_map.shape[1])],
            invalid_voxels,
        )
        if args.verb:
            print(f"Wrote per-HRF R2 map: {prefix_info.stem}_hrf_r2{prefix_info.nifti_ext}")
    elif args.save_hrf_r2 and args.verb:
        print("-save_hrf_r2 ignored: needs -hrf_select refine with more than one HRF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
