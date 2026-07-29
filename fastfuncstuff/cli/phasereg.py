#!/usr/bin/env python3

"""
ffs_phasereg - Phase regression for macrovascular BOLD suppression

Regresses magnitude fMRI signal on phase to remove the contribution of
oriented macrovasculature (pial veins, cerebral veins).  The corrected
output retains microvascular BOLD.

Uses Deming regression (errors-in-variables) by default — equivalent to
scipy.odr (Menon 2002, Curtis 2014, Stanley 2021); see Liem 2023 /
phaseprep for the canonical reference implementation.

INPUTS REQUIRED UPSTREAM
  - Coil-combination preserving phase (SVD / VRC / COMPOSER).
  - Motion correction in real/imag space (so phase isn't scrambled).
  - Temporal phase unwrap.  (First-volume subtraction is optional —
    polynomial detrending here absorbs the constant shift.)
  - Phase passed in must be radians.

OUTPUT MODES
  Default (analysis endpoint): drift dropped, output re-centered on per-
  voxel mean. Matches phaseprep / Stanley 2021 §2.2.3 exactly.

  -keep_drift (GLM input): drift preserved, slope shrunk to A_eff =
  A/(1 + A²/φ). Use when the corrected mag feeds a downstream GLM.

RECOMMENDED PIPELINES

  Phaseprep / Stanley parity (default for analysis):
    ffs_phasereg -magnitude run_mag.nii.gz -phase run_phase.nii.gz \\
                 -prefix out_pr -polort 1

  Task data, output feeds a GLM (preserves drift + task variance):
    ffs_phasereg -magnitude run*.nii.gz -phase run*_phase.nii.gz \\
                 -prefix epi_pr \\
                 -task_removal tent -onsets stim_A.1D stim_B.1D \\
                 -tent_window 20 -keep_drift

  Resting-state, 7T (noisy phase, regularise via Savitzky-Golay):
    ffs_phasereg -magnitude rest_mag.nii.gz -phase rest_phase.nii.gz \\
                 -prefix rest_pr -polort 1 -phase_filter sgf

  Resting-state, 3T (NORDIC upstream is assumed; see Knudsen 2023):
    ffs_phasereg -magnitude rest_mag.nii.gz -phase rest_phase.nii.gz \\
                 -prefix rest_pr -polort 1

  OLS variant (Chang & Giovanello 2026 approach for 3T):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -regression ols

  Knudsen 2023 residual-φ approach (requires task model):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -task_removal tent -onsets stim.1D \\
                 -phi_method residual

  Data-driven phase smoothing (Barry & Gore 2014, expensive):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -phase_filter explore

  7T laminar: veins SURVIVE standard PR (Vu & Gallant 2015 sPR).
  A vein about the size of a voxel has almost no phase of its own, so
  standard PR either leaves it untouched or blows up an ill-conditioned
  slope and flattens the voxel. -spr borrows phase from the vein-adjacent
  neighbour that does carry it; -shrink none stops half the fitted macro
  component being discarded before subtraction:
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -spr -shrink none -phase_filter explore

  Layer extraction: EXCLUDE vessel voxels instead of correcting them.
  Writes prefix_vein_keep as the inverse mask to feed a layer profile.
  Regression preserves laminar structure (all layers go in, outer ones
  suppressed); the mask is the complementary "drop it" option:
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -vein_mask -vein_fdr 0.01

OUTPUTS
  prefix_corrected.nii.gz   M_micro time series (4D)
  prefix_macro.nii.gz       what was subtracted (4D)
  prefix_macro_std.nii.gz   temporal std of the macro (3D, QC overlay)
  prefix_slope.nii.gz       per-voxel A (3D, raw Deming slope)
  prefix_phi.nii.gz         per-voxel variance ratio used (3D)
  prefix_r2.nii.gz          ODR-style R² matching phaseprep (3D) [-r2_mode odr|both]
  prefix_r2_naive.nii.gz    naive R² (voxels most affected) (3D) [-r2_mode naive|both]
  prefix_mask.nii.gz        analysis mask (3D)
  prefix_sgf_window.nii.gz  per-voxel SGF window chosen (3D) [-phase_filter explore]
  prefix_sgf_order.nii.gz   per-voxel SGF order chosen (3D) [-phase_filter explore]
  prefix_spr_donor_corr.nii.gz    corr(mag, donor phase) at chosen donor (3D) [-spr]
  prefix_spr_donor_offset.nii.gz  distance in voxels to the donor, 0=self (3D) [-spr]
  prefix_vein_mask.nii.gz   voxels to EXCLUDE as vessel-dominated (3D) [-vein_mask]
  prefix_vein_keep.nii.gz   its complement within the analysis mask (3D) [-vein_mask]
  prefix_vein_p.nii.gz      p-value of magnitude-phase coupling (3D) [-vein_mask]
  prefix_coupling_r.nii.gz  corr(magnitude, phase) the test is built on (3D) [-vein_mask]

  Intermediates (written with -save_intermediates):
  prefix_mag_dt.nii.gz      magnitude after poly detrending (4D)
  prefix_pha_dt.nii.gz      phase after poly detrending (4D)
  prefix_pha_dt_filt.nii.gz phase after poly detrending + SGF filter (4D)
  prefix_mag_res.nii.gz     magnitude after poly + task removal (4D)
  prefix_pha_res_filt.nii.gz phase after poly + task removal + SGF (4D)

References
----------
Menon RS (2002). MRM 47:1-9.
Barry RL & Gore JC (2014). Hum Brain Mapp 35:3832-3840.
Curtis AT et al (2014). NeuroImage 100:51-59.
Stanley OW et al (2021). NeuroImage 117631.
Knudsen L et al (2023). NeuroImage 271:120011.
Chang WT & Giovanello KS (2026). eLife 12:RP92805.
Liem BT (2023). MSc thesis, Western University. (phaseprep)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_ortvec_arguments, add_verbose_arg, spinner


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


# Printed at the end of -h. Kept self-contained (no wiki required): it tells a
# first-time user what each file is FOR and how to read it, then how the flags
# reshape those files. Any literal percent sign must be doubled (%%) — argparse
# runs this text through %-formatting.
_EPILOG = """\
READING THE OUTPUTS
-------------------
Every run writes these (3D = one value per voxel, 4D = a time series):

  prefix_corrected.nii.gz  (4D)  THE DELIVERABLE. Magnitude with the phase-
      correlated (macrovascular) fluctuations removed; what you feed to your
      GLM or connectivity analysis. In the default mode it is re-centered on
      each voxel's mean and polynomial drift is dropped; with -keep_drift the
      drift is preserved (see below). Voxels the fit could not constrain fall
      back to their mean (default) or to no correction (-keep_drift) rather
      than being corrupted — so a "flat" voxel here means "nothing removed",
      not "broken".

  prefix_macro.nii.gz      (4D)  What was subtracted, exactly
      (mag_orig - corrected). This is your estimate of the macrovascular
      signal. QC LOOK: the temporal std / variance of this map should light
      up pial veins, the sagittal/transverse sinuses and other large vessels
      — vessel-shaped, not a diffuse whole-brain haze. Diffuse structure means
      the fit is capturing noise (check phase preprocessing / phi / polort).

  prefix_macro_std.nii.gz  (3D)  Temporal std of prefix_macro, i.e. the QC
      overlay described above precomputed for you. THIS is the map to overlay
      on anatomy to see WHERE suppression acted — bright on veins/sinuses when
      the fit is good. Prefer it over prefix_r2, which saturates (see below).

  prefix_r2.nii.gz         (3D)  ODR-shrinkage R^2 — the phaseprep / Stanley
      2021 parity metric: fraction of magnitude variance explained once the
      1/(1 + A^2/phi) shrinkage is folded in. Report it for parity, but DO NOT
      read it as a "where did the correction act" map. The shrinkage factor
      also multiplies the residual, which puts a floor under R^2 (~ 1 -
      shrink^2). When magnitude is in raw scanner units phi is large and
      A^2/phi sits near 1, so shrink ~ 0.5 and R^2 saturates high (~0.5-0.8)
      across the WHOLE brain — it does NOT fall to ~0 in weakly-coupled voxels,
      so a bright R^2 does not mean that voxel was actually changed. To see
      where suppression genuinely happened, use prefix_macro_std or
      prefix_r2_naive instead.

  prefix_slope.nii.gz      (3D)  Per-voxel regression slope A (magnitude units
      per radian). Larger |A| = stronger magnitude<->phase coupling. Read it
      WITH phi and R^2, never alone: a large A on low-quality data is an
      ill-conditioned noise fit, not strong signal — which is exactly why the
      correction is shrunk by 1/(1 + A^2/phi) before it is applied.

  prefix_phi.nii.gz        (3D)  Variance ratio phi = Var(mag noise)/Var(phase
      noise) used by the Deming fit. Mostly a diagnostic: it sets how much the
      errors-in-variables fit trusts phase vs magnitude, and it drives the
      shrinkage above. Constant across the brain only if you passed -phi.

  prefix_spr_donor_offset.nii.gz  (3D)  [-spr] Distance in voxels to the
      neighbour whose phase was borrowed; 0 = the voxel kept its own phase
      (i.e. standard PR). QC LOOK: non-zero values should trace vessels, not
      scatter uniformly. A vessel-shaped rim of 1s is sPR doing its job. Salt-
      and-pepper 1s everywhere means the donor search is fitting noise — raise
      phase SNR (NORDIC, -phase_filter explore) before trusting the result.

  prefix_spr_donor_corr.nii.gz    (3D)  [-spr] Signed corr(magnitude, donor
      phase) at the chosen donor — Vu & Gallant's z-scored sPR slope. Low
      |values| brain-wide mean no neighbour had usable phase either.

  IF VEINS SURVIVE PHASE REGRESSION (the common 7T laminar complaint), the
  slope map is the first thing to read. Two distinct failure modes look
  similar in an activation map but differ completely in prefix_slope:
    * Slope ~ 0: the voxel's phase carries no task signal, so nothing is
      subtracted and the vein passes straight through. This is the vein-about-
      the-size-of-a-voxel / magic-angle case. Fix: -spr.
    * Slope enormous (1e3-1e5): an ill-conditioned fit against phase noise.
      The 1/(1+A^2/phi) shrinkage then collapses the voxel to its mean, which
      LOOKS like perfect suppression but has destroyed the voxel's thermal
      noise along with everything else. Check prefix_corrected's temporal std:
      if it is far below neighbouring grey matter, the voxel was flattened,
      not corrected. Fix: -spr gives the voxel a real regressor and the slope
      drops to a sane value.
  A third, quieter cause is the shrinkage itself: in raw scanner units
  A^2/phi ~ 1, so roughly half the fitted macro component is discarded before
  subtraction even in well-behaved voxels. See -shrink none.

  prefix_mask.nii.gz       (3D)  Which voxels were actually fit (1) vs skipped
      as air/skull/low-SNR (0). Everything outside it is zero in the 3D maps.

  prefix_r2_naive.nii.gz   (3D)  [only with -r2_mode naive|both] Raw R^2 with
      NO shrinkage. A QC/where-did-it-act map, NOT a metric to report: it is
      bright wherever phase regression had the largest raw effect, including
      ill-conditioned noise voxels. Use it (or macro variance) to see which
      voxels changed most; prefix_r2 is the phaseprep-parity number to report.

  prefix_sgf_window.nii.gz / prefix_sgf_order.nii.gz  (3D)  [only with
      -phase_filter explore] The window length and polynomial order the
      per-voxel search picked. Read them together: large window + low order =
      the search wanted heavy smoothing there (noisy phase); small window =
      it left the phase nearly untouched (already clean). A useful sanity
      check that explore is adapting sensibly and not smoothing signal away.

QC in one pass: overlay prefix_macro_std (or prefix_r2_naive) on anatomy and
confirm the bright voxels sit on veins/sinuses. Prefer macro_std over prefix_r2
here — the ODR R^2 saturates in raw-unit runs (see above) and makes a poor QC
overlay. If suppression looks diffuse or noisy, suspect phase preprocessing
(unwrap / coil combine), then phi, then polort.

NUISANCE REGRESSORS & EVENTS
----------------------------
  -ortvec (and -ortvec_run / -ortvec_glob / -ortvec_concat) are all REPEATABLE
      and STACK: pass one file per source and every column is concatenated into
      that run's nuisance design, then projected out before the slope fit. A file
      can have any number of columns. Give each a distinct LABEL:
        -ortvec motion.1D motion -ortvec physio.1D physio -ortvec extra.1D extra
      -ortvec expects FULL-LENGTH files (rows = total TRs across all runs);
      -ortvec_run FILE LABEL RUN is the per-run form (zero-padded elsewhere);
      -ortvec_glob / -ortvec_concat are convenience globs over per-run file sets.
      Mix freely in one call.

  -event_cols ONSET DUR TYPE selects columns BY NAME, mapping your TSV's headers
      to the three roles (onset, duration, trial_type); default names are
      onset / duration / trial_type. ONLY these three columns are read — every
      other column in the TSV is ignored. The trial_type column defines the
      conditions (one TENT/canonical set per unique value); use -event_ignore to
      drop specific values. Durations come from the TSV, so -durations is not
      needed with -events. (No parametric-modulation column is supported — only
      the three roles.)

HOW THE FLAGS CHANGE WHAT YOU GET
---------------------------------
  -keep_drift        Rewrites prefix_corrected AND prefix_macro. Default:
                     drift dropped, output = mean + shrunk residual. On:
                     drift kept, output = mag_orig - A_eff*(phase-mean) with
                     A_eff = A/(1+A^2/phi). slope/phi/r2 are unchanged; only
                     the two 4D files differ. Use it when a GLM downstream
                     will model the drift; use default for an analysis
                     endpoint.

  -task_removal / -onsets / -events
                     Estimates slope and phi on TASK-RESIDUAL data, so A
                     reflects vessel coupling rather than a shared task
                     response. Changes slope, phi and r2 (and therefore the
                     correction). The correction is still applied to the full
                     detrended series. Pair with -phi_method residual on task
                     data; 'fft' is safe either way.

  -regression ols    Fills prefix_slope with an ordinary least-squares slope
                     (no errors-in-variables term) instead of Deming. phi is
                     still estimated and still governs the shrinkage/R^2.

  -phase_filter sgf  Fits and corrects on a Savitzky-Golay-smoothed phase,
                     suppressing high-frequency phase noise before the slope
                     fit; recommended at 7T. Changes slope/macro (and r2). Was
                     a silent no-op prior to this build — verify your output
                     differs from -phase_filter none. Tune with -sgf_window /
                     -sgf_order.

  -phase_filter explore
                     Searches SGF parameters PER VOXEL, keeping the window/
                     order that maximises |corr(smoothed phase, mag)| (Barry &
                     Gore 2014). Adaptive but expensive. -sgf_window/-sgf_order
                     are IGNORED here; tune the search grid with -sgf_window_max
                     / -sgf_order_max / -sgf_step. Writes prefix_sgf_window and
                     prefix_sgf_order so you can see what it chose per voxel.

  -polort N          Higher order removes more drift from both series before
                     the fit. Removes more nuisance low-frequency content but
                     can also absorb shared low-frequency mag/phase you may
                     want to keep. -polort 1 = exact phaseprep parity.

  -phi VALUE         Pins prefix_phi to a constant (no per-voxel estimation).

  -signal_thresh 0   Disables signal gating. prefix_corrected stays clean
                     (shrinkage caps every voxel), but prefix_r2 will show
                     bright values in air/noise — the ODR metric reads
                     unfittable voxels as "perfect within the noise budget".
                     A visual artifact, not a bug.

  -r2_mode           Selects which R^2 map(s) are written (odr / naive / both).

  -save_intermediates
                     Adds prefix_mag_dt, _pha_dt, _pha_dt_filt, _mag_res,
                     _pha_res_filt (4D) for step-by-step inspection of the
                     detrend -> filter -> task-removal chain.
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase regression for macrovascular BOLD suppression",
        epilog=_EPILOG,
        formatter_class=_HelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────────
    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-magnitude",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Magnitude fMRI data files (one per run). Supports wildcards.",
    )
    req.add_argument(
        "-phase",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Phase fMRI data files (radians, unwrapped, one per run). "
        "Must match magnitude files in order and dimensions.",
    )
    req.add_argument(
        "-prefix",
        required=True,
        metavar="OUTPUT",
        help="Output file prefix (e.g. results/epi_pr).",
    )

    # ── Task removal ─────────────────────────────────────────────────────
    task = parser.add_argument_group("Task Removal Options")
    task.add_argument(
        "-task_removal",
        choices=["none", "tent", "canonical"],
        default="none",
        help="Task removal before slope estimation: "
        "'none' (direct regression, OK after NORDIC or for RS), "
        "'tent' (TENT/FIR basis - model-free, recommended), "
        "'canonical' (SPM HRF convolution).",
    )
    task.add_argument(
        "-onsets",
        nargs="+",
        metavar="FILE",
        help="AFNI-format onset timing files (one per condition, "
        "each with one row per run). Mutually exclusive with -events.",
    )
    task.add_argument(
        "-events",
        nargs="+",
        metavar="TSV",
        help="BIDS events TSV files (one per run). Mutually exclusive with -onsets.",
    )
    task.add_argument(
        "-event_ignore",
        nargs="+",
        metavar="CONDITION",
        help="trial_type values to skip when using -events.",
    )
    task.add_argument(
        "-event_cols",
        nargs=3,
        metavar=("ONSET", "DUR", "TYPE"),
        help="Custom column names for -events, mapping to (onset, duration, "
        "trial_type) by name (default: onset, duration, trial_type). Only these "
        "three columns are read; any others in the TSV are ignored.",
    )
    task.add_argument(
        "-tent_window",
        type=float,
        default=20.0,
        metavar="SECONDS",
        help="TENT window duration in seconds (for -task_removal tent).",
    )
    task.add_argument(
        "-durations",
        nargs="+",
        metavar="SECONDS",
        help="Per-condition stimulus durations for auto window estimation "
        "(with -onsets). With -events, durations come from the TSV.",
    )

    # ── Regression options ───────────────────────────────────────────────
    reg = parser.add_argument_group("Regression Options")
    reg.add_argument(
        "-regression",
        choices=["deming", "ols"],
        default="deming",
        help="Regression method. 'deming' is the closed-form Deming/ODR "
        "(equivalent to scipy.odr at fMRI φ scales — see Menon 2002, "
        "Curtis 2014, Stanley 2021, phaseprep). 'ols' regresses mag on "
        "phase without errors-in-variables correction (Chang & "
        "Giovanello 2026 use OLS at 3T after NORDIC).",
    )
    reg.add_argument(
        "-phi",
        type=float,
        default=None,
        metavar="VALUE",
        help="Fixed variance ratio for Deming regression. "
        "If not set, estimated automatically per voxel via -phi_method.",
    )
    reg.add_argument(
        "-phi_method",
        choices=["fft", "residual"],
        default="fft",
        help="How to estimate phi (Var(eps_mag)/Var(eps_phase)): "
        "'fft' uses out-of-band spectral power above -freq_range "
        "(Curtis 2014, phaseprep default; works for task and rest). "
        "'residual' uses temporal variance of (poly+task+nuisance)-"
        "projected residuals (Knudsen 2023). "
        "WARNING: 'residual' without -task_removal on task data will "
        "include task variance in the noise estimate and inflate phi. "
        "Pair 'residual' with -task_removal {tent,canonical} for "
        "task data; 'fft' is safe for both.",
    )
    reg.add_argument(
        "-freq_range",
        nargs="+",
        type=float,
        default=[0.15],
        metavar="HZ",
        help="Frequency range for FFT noise estimation. "
        "One value = lower bound (upper = Nyquist). "
        "Two values = lower and upper bounds. "
        "Default 0.15 Hz matches Stanley 2021 §2.2.3 and phaseprep "
        "(noise_lb=0.15). Curtis 2014 used 0.1 Hz; either is reasonable.",
    )

    # ── Phase filtering ──────────────────────────────────────────────────
    pfilt = parser.add_argument_group("Phase Filtering Options (Barry & Gore 2014)")
    pfilt.add_argument(
        "-phase_filter",
        choices=["none", "sgf", "explore"],
        default="none",
        help="Pre-filter phase time series before regression. "
        "'none': no filtering (Stanley/phaseprep default). "
        "'sgf': Savitzky-Golay with -sgf_window/-sgf_order — same fit "
        "behaviour as 'none' but on a smoother phase, which acts as a "
        "phase-noise regulariser: it removes high-frequency phase content "
        "before the slope fit, changing slope/macro (and r2). "
        "'explore': per-voxel data-driven parameter search. Barry & Gore "
        "define the optimal (N, p) as the pair MINIMISING the temporal "
        "variance of the phase-regressed magnitude, i.e. maximising "
        "R2 = 1 - sigma_PR/sigma_orig. We maximise |Pearson r(filtered_phase, "
        "mag_dt)| instead, which for an OLS fit is the identical argmax "
        "(residual variance = var_mag * (1 - r^2)) but costs one correlation "
        "rather than a full refit per grid point. Under Deming the two "
        "criteria can differ slightly, since the ODR shrinkage 1/(1+A^2/phi) "
        "also moves with the fit. The unfiltered series competes as a "
        "candidate, so voxels with good phase SNR can decline filtering "
        "(Barry & Gore step 3); those report window=order=0. Expensive. "
        "Recommended at 7T where phase SNR is low (Hagberg et al 2008 "
        "found k_phi >> k_mag).",
    )
    pfilt.add_argument(
        "-sgf_window",
        type=int,
        default=None,
        metavar="N",
        help="SGF window length in TRs (must be odd; auto-incremented if "
        "even). Larger = more smoothing. Default: auto = round(20s / TR), "
        "floored at 5. Treat that default as a placeholder, not a "
        "recommendation: it is NOT from Barry & Gore, who explored N over "
        "5..n_TR/2 per voxel precisely because the right window is not "
        "knowable a priori. Note 20 s is also close to the duration of the "
        "HRF itself, so this window can smooth away the task-locked phase "
        "modulation you are trying to regress with, especially in block "
        "designs. Prefer -phase_filter explore, or set this deliberately from "
        "your design's timescale. Only used with -phase_filter sgf.",
    )
    pfilt.add_argument(
        "-sgf_order",
        type=int,
        default=3,
        metavar="P",
        help="SGF polynomial order (must be < window length). Lower = "
        "more smoothing. Only used with -phase_filter sgf.",
    )
    pfilt.add_argument(
        "-sgf_window_max",
        type=int,
        default=None,
        metavar="N",
        help="EXPLORE grid: largest window (odd) searched per voxel. "
        "Default: n_TRs // 2, matching Barry & Gore (N <= 49 for their 96-TR "
        "runs, N <= 97 for their 192-TR PRESTO runs). Only used with "
        "-phase_filter explore.",
    )
    pfilt.add_argument(
        "-sgf_order_max",
        type=int,
        default=None,
        metavar="P",
        help="EXPLORE grid: largest polynomial order searched per voxel "
        "(min is 2). Default: sgf_window_max // 4, i.e. ~12 for a 96-TR run "
        "and ~24 for 192 TRs, matching Barry & Gore. The previous default of "
        "5 was not from the paper and made only the heavily-smoothing corner "
        "of the grid reachable. Only used with -phase_filter explore.",
    )
    pfilt.add_argument(
        "-sgf_step",
        type=int,
        default=4,
        metavar="N",
        help="EXPLORE grid: window step (odd-ified) between candidate "
        "windows. Smaller = finer, slower search. Default: 4. Only used "
        "with -phase_filter explore.",
    )

    parser.add_argument(
        "-shrink",
        choices=["odr", "none"],
        default="odr",
        help="Whether the ODR shrinkage factor 1/(1+A^2/phi) is applied to the "
        "correction. 'odr' (default) is phaseprep/Stanley parity and makes "
        "ill-conditioned voxels decay to their mean rather than speckle. It is "
        "not free: with magnitude in raw scanner units A^2/phi sits near 1 over "
        "much of the brain, so roughly HALF the fitted macrovascular component "
        "is thrown away before it is subtracted. If veins are surviving phase "
        "regression, try 'none' — it applies the textbook M - A*phi at full "
        "strength. Verbose output reports how much parity mode is discarding.",
    )

    vg = parser.add_argument_group("Vein Exclusion Mask")
    vg.add_argument(
        "-vein_mask",
        action="store_true",
        help="Also write a vein EXCLUSION mask: voxels whose magnitude covaries "
        "with phase more than chance. Randomly-oriented microvasculature "
        "produces magnitude change with no coherent phase change, so a "
        "significant magnitude-phase correlation is direct evidence of an "
        "oriented vessel. Intended for laminar / layer-extraction workflows "
        "that would rather DROP contaminated voxels than trust a subtraction: "
        "no slope is applied, so nothing can be over- or under-corrected. "
        "Complements the corrected output rather than replacing it — the "
        "corrected series is unchanged by this flag. Same logic as the "
        "H_c-vs-H_a contrast of Rowe 2005, without fitting the full "
        "complex-valued GLM. "
        "INTERACTION WITH -spr, a real trade-off: without -spr the test uses "
        "the voxel's OWN phase, which is specific (grey matter phase is not "
        "task-locked) but MISSES the phase-blind veins -spr exists for. With "
        "-spr the test uses the donor's phase, which catches them, but also "
        "flags healthy grey matter that merely borrowed a strong nearby phase "
        "source — in practice dilating the mask by about one voxel around each "
        "source. For a conservative exclusion mask that dilation may be what "
        "you want (vein-adjacent voxels are partly contaminated anyway); for a "
        "tight one, omit -spr. Inspect prefix_coupling_r before committing.",
    )
    vg.add_argument(
        "-vein_fdr",
        type=float,
        default=0.05,
        metavar="Q",
        help="Benjamini-Hochberg q for -vein_mask. Lower = fewer voxels "
        "excluded. The correlation is Sidak-corrected for the sPR donor argmax "
        "before FDR, so -spr does not silently inflate the mask. Default: 0.05.",
    )

    # ── Source-localized phase regression ────────────────────────────────
    sprg = parser.add_argument_group("Source-Localized Phase Regression (sPR; Vu & Gallant 2015)")
    sprg.add_argument(
        "-spr",
        action="store_true",
        help="Regress each voxel's magnitude on the phase of whichever "
        "neighbour best tracks it, instead of on its own phase. Targets the "
        "dominant PR failure mode at 7T: a vein roughly the size of a voxel "
        "straddles the whole off-resonance dipole, so its sampled field "
        "offsets cancel and its own phase fSNR is near zero — huge magnitude "
        "change, no phase to regress it away with, vein survives. Adjacent "
        "voxels sample one polarity of the dipole and DO have high phase "
        "fSNR. Also recovers veins near the magic angle (~54.7 deg), where "
        "intravascular phase accrual vanishes. "
        "NOTE sPR has two separable halves and they pull in OPPOSITE "
        "directions. (1) Donor borrowing (this flag) finds phase where the "
        "voxel had none, so it suppresses MORE. (2) Vu also swaps Menon's "
        "chi-squared loss for plain OLS, which suppresses LESS — that half "
        "exists to fix over-correction (Nencka & Rowe 2007), a problem you "
        "only have if PR is eating your grey-matter signal. If your complaint "
        "is that veins SURVIVE, take the donor and keep -regression deming. "
        "Use -regression ols only for literal Vu & Gallant parity.",
    )
    sprg.add_argument(
        "-spr_neighborhood",
        "-spr_connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        dest="spr_neighborhood",
        metavar="K",
        help="Donor search neighbourhood: 6 = face-adjacent (Vu & Gallant's "
        "7-voxel set: self + 6 faces), 18 adds edges, 26 the full 3x3x3. "
        "Vu chose 6 as the smallest sensible increment over standard PR and "
        "left larger neighbourhoods explicitly open. Bigger K searches harder "
        "but the argmax overfits more (see -spr_select_run). Default: 6.",
    )
    sprg.add_argument(
        "-spr_select_run",
        type=int,
        default=None,
        metavar="R",
        help="0-based run used ONLY to pick donors, which are then applied to "
        "all runs. Vu & Gallant set aside their first run for exactly this, so "
        "the argmax over neighbours is not chosen and evaluated on the same "
        "data. Omit (default) to select on all runs concatenated — required "
        "for single-run data, but then donor selection is in-sample and biases "
        "mildly toward over-suppression.",
    )

    # ── Processing options ───────────────────────────────────────────────
    proc = parser.add_argument_group("Processing Options")
    proc.add_argument(
        "-polort",
        type=str,
        default="A",
        metavar="N",
        help="Polynomial drift order (Legendre, per run). 'A' = auto "
        "(AFNI formula based on run duration). Integer for fixed "
        "order. -1 for none. Use -polort 1 for exact phaseprep parity "
        "(linear-only detrend, what PreprocessPhase + DetrendMag do). "
        "Higher orders remove more drift before the fit but can absorb "
        "shared low-frequency mag-phase content that you may want to "
        "keep — there's a tradeoff.",
    )
    proc.add_argument(
        "-tr",
        type=float,
        metavar="SECONDS",
        help="Repetition time. Default: read from NIfTI pixdim/zooms. "
        "Override here if your header is wrong.",
    )
    proc.add_argument(
        "-mask",
        metavar="FILE",
        help="Brain mask (restricts analysis to mask voxels). "
        "If not provided, voxels are filtered by -signal_thresh.",
    )
    proc.add_argument(
        "-signal_thresh",
        type=float,
        default=0.03,
        metavar="FRAC",
        help="Minimum mean-signal fraction to include a voxel in "
        "regression (default 0.03 = 3%% of max, matching phaseprep). "
        "Voxels below this have slope=0 and R²=0. "
        "Set to 0 to disable: corrected output stays clean (ODR "
        "shrinkage caps every voxel), but R² map shows bright noise "
        "voxels because ODR-style R² reads unfittable data as "
        "'perfect within noise budget' — visual artifact, not a bug.",
    )
    proc.add_argument(
        "-no_auto_mask",
        action="store_true",
        help="Disable automatic brain masking when -mask is not provided. "
        "By default, ffs_phasereg computes a brain mask from the mean "
        "magnitude (using AFNI-compatible automask) to exclude air, "
        "skull, and non-brain tissue before regression.",
    )
    proc.add_argument(
        "-keep_drift",
        action="store_true",
        help="GLM-friendly output mode. Default (off) is phaseprep parity: "
        "corrected = mean + shrunk_residual, drift dropped, ill-"
        "conditioned voxels collapse to per-voxel mean. With this flag "
        "ON, corrected = mag_orig - A_eff·(phase_dt - mean), where "
        "A_eff = A/(1 + A²/φ) is the shrunken slope. Polynomial drift "
        "is preserved (your downstream GLM models it), and A_eff is "
        "naturally bounded by √φ/2 so high-Deming-slope voxels still "
        "produce a clean correction instead of speckle. "
        "Recommended when the corrected mag is the input to a GLM "
        "rather than the analysis endpoint.",
    )
    proc.add_argument(
        "-motion",
        nargs="+",
        metavar="FILE",
        help="(DEPRECATED — use -ortvec_run instead.) "
        "Motion parameter files, one per run (AFNI dfile format). "
        "Equivalent to passing each as -ortvec_run FILE motion RUN. "
        "Added as nuisance regressors.",
    )
    add_ortvec_arguments(proc)

    # ── Hardware ─────────────────────────────────────────────────────────
    hw = parser.add_argument_group("Hardware Options")
    hw.add_argument(
        "-device",
        type=str,
        default=None,
        help="PyTorch device: cpu, cuda, mps (default: auto-detect).",
    )

    # ── Output ───────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output Options")
    out.add_argument(
        "-r2_mode",
        choices=["odr", "naive", "both"],
        default="odr",
        help="Which R² map(s) to write. 'odr' (default): ODR-shrinkage R² "
        "matching phaseprep / Stanley 2021 — the canonical metric. "
        "'naive': 1 - SS_res(observed) / SS_tot without shrinkage — "
        "highlights voxels with the largest raw phase-regression effect, "
        "useful for spotting which voxels were most changed even when "
        "ODR inflation makes the canonical R² appear small. "
        "'both': write prefix_r2.nii.gz (ODR) and prefix_r2_naive.nii.gz.",
    )
    out.add_argument(
        "-save_intermediates",
        action="store_true",
        help="Write intermediate time-series volumes for step-by-step inspection: "
        "prefix_mag_dt (poly-detrended magnitude), "
        "prefix_pha_dt (poly-detrended phase), "
        "prefix_pha_dt_filt (detrended + SGF-filtered phase, differs from "
        "prefix_pha_dt only when -phase_filter sgf|explore is active), "
        "prefix_mag_res (magnitude after poly + task removal, used for "
        "slope estimation), and "
        "prefix_pha_res_filt (phase after poly + task + SGF, used for phi "
        "estimation and slope estimation).",
    )
    add_verbose_arg(out, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for ffs_phasereg."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # ── Imports ──────────────────────────────────────────────────────────
    try:
        from fastfuncstuff.cli_utils import (
            auto_polort,
            parse_device_arg,
            parse_prefix,
        )
        from fastfuncstuff.io.afni import (
            get_tr_from_file,
            load_afni_mask,
            load_nifti,
            save_nifti,
        )
        from fastfuncstuff.phasereg.core import phase_regress
        from fastfuncstuff.utils import configure_torch_backends, get_device  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Could not import fastfuncstuff: {e}", file=sys.stderr)
        return 1

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)

    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("Phase Regression (ffs_phasereg)")
        print(f"{'=' * 70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")

    # ── Validate inputs ──────────────────────────────────────────────────
    n_runs = len(args.magnitude)
    if len(args.phase) != n_runs:
        print(
            f"ERROR: Number of magnitude ({n_runs}) and phase "
            f"({len(args.phase)}) files must match.",
            file=sys.stderr,
        )
        return 1

    if args.task_removal != "none" and not args.onsets and not args.events:
        print(
            f"ERROR: -task_removal {args.task_removal} requires -onsets or -events.",
            file=sys.stderr,
        )
        return 1

    if args.onsets and args.events:
        print("ERROR: -onsets and -events are mutually exclusive.", file=sys.stderr)
        return 1

    # ── Parse frequency range ────────────────────────────────────────────
    if len(args.freq_range) == 1:
        freq_range = (args.freq_range[0], None)
    elif len(args.freq_range) == 2:
        freq_range = (args.freq_range[0], args.freq_range[1])
    else:
        print("ERROR: -freq_range takes 1 or 2 values.", file=sys.stderr)
        return 1

    # ── Load data ────────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nLoading data...")

    mag_list = []
    pha_list = []
    n_tp_per_run = []
    tr_values = []
    nx = ny = nz = None
    ref_img_path = args.magnitude[0]

    for i in range(n_runs):
        mag_path = args.magnitude[i]
        pha_path = args.phase[i]

        for p in (mag_path, pha_path):
            if not Path(p).exists():
                print(f"ERROR: File not found: {p}", file=sys.stderr)
                return 1

        mag_img = load_nifti(mag_path)
        pha_img = load_nifti(pha_path)

        mag_data = mag_img.get_fdata(dtype=np.float32)
        pha_data = pha_img.get_fdata(dtype=np.float32)

        if mag_data.ndim != 4:
            print(f"ERROR: Expected 4D data, got {mag_data.ndim}D: {mag_path}", file=sys.stderr)
            return 1
        if pha_data.shape != mag_data.shape:
            print(
                f"ERROR: Phase shape {pha_data.shape} != magnitude shape {mag_data.shape}",
                file=sys.stderr,
            )
            return 1

        if nx is None:
            nx, ny, nz = mag_data.shape[:3]

        n_tp = mag_data.shape[3]
        n_tp_per_run.append(n_tp)

        if args.tr is None:
            tr_values.append(get_tr_from_file(mag_path))

        mag_list.append(torch.tensor(mag_data.reshape(-1, n_tp), dtype=torch.float32))
        pha_list.append(torch.tensor(pha_data.reshape(-1, n_tp), dtype=torch.float32))

        if args.verb >= 1:
            print(f"  Run {i + 1}: {mag_path} ({n_tp} TRs)")

    if args.tr is None:
        if len(set(tr_values)) > 1:
            print(f"ERROR: Inconsistent TRs: {tr_values}", file=sys.stderr)
            return 1
        tr = tr_values[0]
    else:
        tr = args.tr

    if args.verb >= 1:
        print(f"  TR: {tr}s")
        print(f"  Volume: {nx} x {ny} x {nz}")

    # ── SGF window: TR-adaptive default (~20s, HRF-duration based) ──────
    if args.sgf_window is None:
        sgf_window = round(20.0 / tr)
        if sgf_window % 2 == 0:
            sgf_window += 1
        sgf_window = max(sgf_window, 5)
        if args.verb >= 1:
            print(f"  SGF window: {sgf_window} (~20s / {tr:.2f}s TR)")
    else:
        sgf_window = args.sgf_window
        if sgf_window % 2 == 0:
            print("ERROR: -sgf_window must be odd.", file=sys.stderr)
            return 1

    # ── Mask ─────────────────────────────────────────────────────────────
    mask = None
    n_all_voxels = nx * ny * nz
    if args.mask:
        mask = load_afni_mask(args.mask)
        if mask.shape != (nx, ny, nz):
            print(f"ERROR: Mask shape {mask.shape} != data shape {(nx, ny, nz)}", file=sys.stderr)
            return 1
        mask_flat = mask.flatten()
        n_voxels = int(mask_flat.sum())
        mag_list = [m[mask_flat.astype(bool)] for m in mag_list]
        pha_list = [p[mask_flat.astype(bool)] for p in pha_list]
        if args.verb >= 1:
            print(f"  Mask (provided): {n_voxels:,} / {n_all_voxels:,} voxels")
    elif not args.no_auto_mask:
        from fastfuncstuff.processing.mask import automask

        mean_mag = torch.stack([m.mean(dim=1) for m in mag_list]).mean(dim=0)
        mean3d = mean_mag.reshape(nx, ny, nz)
        mask3d = automask(mean3d, dilate_extra=2, device=device, verbose=args.verb >= 1)
        mask_flat = mask3d.flatten().cpu()
        n_voxels = int(mask_flat.sum().item())
        mask_bool = mask_flat.bool()
        mag_list = [m[mask_bool] for m in mag_list]
        pha_list = [p[mask_bool] for p in pha_list]
        if args.verb >= 1:
            print(f"  Mask (auto): {n_voxels:,} / {n_all_voxels:,} voxels")
    else:
        mask_flat = None
        n_voxels = n_all_voxels
        if args.verb >= 1:
            print(f"  Mask: none (using all {n_all_voxels:,} voxels)")

    # ── Polort ───────────────────────────────────────────────────────────
    polort_str = str(args.polort).strip().upper()
    if polort_str == "A":
        run_dur = min(n_tp_per_run) * tr
        polort = auto_polort(run_dur, formula="afni")
        if args.verb >= 1:
            print(f"  Polort: A -> {polort} (run duration {run_dur:.1f}s)")
    else:
        polort = int(polort_str)

    # ── Parse onsets ─────────────────────────────────────────────────────
    onsets_per_condition = None
    if args.onsets:
        from fastfuncstuff.design.builder import parse_afni_timing_file

        onsets_per_condition = []
        for onset_file in args.onsets:
            onsets_by_run = parse_afni_timing_file(onset_file)
            if len(onsets_by_run) != n_runs:
                print(
                    f"ERROR: {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}",
                    file=sys.stderr,
                )
                return 1
            onsets_per_condition.append(onsets_by_run)
        if args.verb >= 1:
            print(f"  Onsets: {len(onsets_per_condition)} conditions")

    elif args.events:
        from fastfuncstuff.design.bids_events import parse_bids_events

        if len(args.events) not in (1, n_runs):
            print(
                f"ERROR: -events requires one TSV per run or a single shared TSV: "
                f"got {len(args.events)} events files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        if len(args.events) == 1 and n_runs > 1:
            print(f"  Broadcasting 1 events file across {n_runs} runs")
        event_cols = tuple(args.event_cols) if args.event_cols else None
        try:
            bids_onsets, bids_durations, bids_labels = parse_bids_events(
                event_files=args.events,
                event_ignore=args.event_ignore,
                event_cols=event_cols,
                n_runs=n_runs,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        onsets_per_condition = bids_onsets
        if args.verb >= 1:
            print(f"  BIDS events: {len(bids_labels)} conditions ({', '.join(bids_labels)})")

    # ── Auto tent window from durations ──────────────────────────────────
    tent_window = args.tent_window
    if args.task_removal == "tent" and args.durations and onsets_per_condition is not None:
        from fastfuncstuff.design.builder import parse_durations
        from fastfuncstuff.design.hrf import compute_windows_from_durations

        n_conds = len(onsets_per_condition)
        cond_labels = [f"cond{i}" for i in range(n_conds)]
        condition_durations = parse_durations(args.durations, n_conds, cond_labels)
        windows = compute_windows_from_durations(condition_durations, tr)
        tent_window = max(top for _, top in windows)
        if args.verb >= 1:
            print(f"  Auto TENT window: {tent_window:.1f}s (from stimulus durations)")

    # ── Build per-run nuisance from -motion / -ortvec* flags ─────────────
    # -motion remains as a deprecated alias for N invocations of
    # -ortvec_run FILE motion RUN. Both styles funnel through the unified
    # NuisanceBlock pathway so per-run / glob / full-length inputs all work.
    if args.motion:
        if len(args.motion) != n_runs:
            print(
                f"ERROR: {len(args.motion)} motion files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        if args.verb >= 1:
            print(
                "WARNING: -motion is deprecated; use "
                "-ortvec_run FILE motion RUN (repeatable) instead."
            )
        args.ortvec_run = list(args.ortvec_run or []) + [
            [str(f), "motion", str(i + 1)] for i, f in enumerate(args.motion)
        ]

    nuisance_per_run = None
    if args.ortvec or args.ortvec_run or args.ortvec_glob or args.ortvec_concat:
        from fastfuncstuff.cli_utils import collect_nuisance_blocks

        run_lengths = [int(m.shape[1]) for m in mag_list]
        run_starts = [0]
        for rl in run_lengths[:-1]:
            run_starts.append(run_starts[-1] + rl)
        n_timepoints = sum(run_lengths)

        blocks = collect_nuisance_blocks(
            args,
            run_starts,
            n_timepoints,
            verbose=(args.verb >= 1),
        )
        nuisance_per_run = []
        for run_idx, run_length in enumerate(run_lengths):
            cols: list[np.ndarray] = []
            for block in blocks:
                if block.n_columns == 0:
                    continue
                m = block.get_run(run_idx, run_length).copy()
                col_mean = m.mean(axis=0, keepdims=True)
                if np.max(np.abs(col_mean)) > 1e-4:
                    m = m - col_mean
                cols.append(m)
            if cols:
                arr = np.concatenate(cols, axis=1).astype(np.float32)
                nuisance_per_run.append(torch.tensor(arr, dtype=torch.float32, device=device))
            else:
                nuisance_per_run.append(
                    torch.zeros((run_length, 0), dtype=torch.float32, device=device)
                )
        if args.verb >= 1:
            print(
                f"  Nuisance: {nuisance_per_run[0].shape[1]} regressor(s) per run "
                f"({len(blocks)} block(s))"
            )

    # ── Run phase regression ─────────────────────────────────────────────
    if args.verb >= 1:
        print()

    result = phase_regress(
        magnitude=mag_list,
        phase=pha_list,
        tr=tr,
        task_removal=args.task_removal,
        onsets_per_condition=onsets_per_condition,
        nuisance_per_run=nuisance_per_run,
        max_poly_degree=polort,
        phi=args.phi,
        phi_method=args.phi_method,
        phi_freq_range=freq_range,
        regression=args.regression,
        tent_window=tent_window,
        phase_filter=args.phase_filter,
        sgf_window=sgf_window,
        sgf_order=args.sgf_order,
        sgf_window_max=args.sgf_window_max,
        sgf_order_max=args.sgf_order_max,
        sgf_step=args.sgf_step,
        signal_thresh=args.signal_thresh,
        keep_drift=args.keep_drift,
        shrink_mode=args.shrink,
        vein_mask=args.vein_mask,
        vein_fdr_q=args.vein_fdr,
        spr=args.spr,
        spr_connectivity=args.spr_neighborhood,
        spr_select_run=args.spr_select_run,
        volume_shape=(nx, ny, nz),
        mask_flat=mask_flat,
        device=str(device),
        verbose=args.verb >= 1,
        r2_mode=args.r2_mode,
        save_intermediates=args.save_intermediates,
    )

    # ── Save outputs ─────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nSaving outputs...")

    def _to_volume(data_flat, is_4d=False):
        if is_4d:
            n_tp = data_flat.shape[1]
            if mask_flat is not None:
                vol = np.zeros((n_all_voxels, n_tp), dtype=np.float32)
                vol[np.asarray(mask_flat, dtype=bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz, n_tp)
        else:
            if mask_flat is not None:
                vol = np.zeros(n_all_voxels, dtype=np.float32)
                vol[np.asarray(mask_flat, dtype=bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz)

    outputs = {}

    with spinner("Writing outputs"):
        corrected_np = result.magnitude_corrected.numpy()
        fname = f"{args.prefix}_corrected{nii_ext}"
        save_nifti(_to_volume(corrected_np, is_4d=True), fname, reference_img=ref_img_path)
        outputs["corrected"] = fname

        macro_np = result.macrovascular_component.numpy()
        fname = f"{args.prefix}_macro{nii_ext}"
        save_nifti(_to_volume(macro_np, is_4d=True), fname, reference_img=ref_img_path)
        outputs["macro"] = fname

        # Temporal std of what was subtracted: the recommended QC overlay for
        # localising suppression (vessel-shaped), since the ODR R² saturates.
        macro_std_np = result.macrovascular_component.std(dim=1).numpy()
        fname = f"{args.prefix}_macro_std{nii_ext}"
        save_nifti(_to_volume(macro_std_np), fname, reference_img=ref_img_path)
        outputs["macro_std"] = fname

        fname = f"{args.prefix}_slope{nii_ext}"
        save_nifti(_to_volume(result.slope.numpy()), fname, reference_img=ref_img_path)
        outputs["slope"] = fname

        if args.r2_mode in ("odr", "both"):
            fname = f"{args.prefix}_r2{nii_ext}"
            save_nifti(_to_volume(result.r2_phase.numpy()), fname, reference_img=ref_img_path)
            outputs["r2"] = fname

        if args.r2_mode in ("naive", "both") and result.r2_naive is not None:
            fname = f"{args.prefix}_r2_naive{nii_ext}"
            save_nifti(_to_volume(result.r2_naive.numpy()), fname, reference_img=ref_img_path)
            outputs["r2_naive"] = fname

        fname = f"{args.prefix}_phi{nii_ext}"
        save_nifti(_to_volume(result.phi.numpy()), fname, reference_img=ref_img_path)
        outputs["phi"] = fname

        fname = f"{args.prefix}_mask{nii_ext}"
        save_nifti(
            _to_volume(result.voxel_mask.numpy().astype(np.float32)),
            fname,
            reference_img=ref_img_path,
        )
        outputs["mask"] = fname

        # Vein exclusion mask: the voxels to DROP from a layer profile.
        if result.vein_exclude is not None and result.vein_p is not None:
            fname = f"{args.prefix}_vein_mask{nii_ext}"
            save_nifti(
                _to_volume(result.vein_exclude.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["vein_mask"] = fname

            fname = f"{args.prefix}_vein_keep{nii_ext}"
            keep = result.voxel_mask & ~result.vein_exclude
            save_nifti(
                _to_volume(keep.numpy().astype(np.float32)), fname, reference_img=ref_img_path
            )
            outputs["vein_keep"] = fname

            fname = f"{args.prefix}_vein_p{nii_ext}"
            save_nifti(_to_volume(result.vein_p.numpy()), fname, reference_img=ref_img_path)
            outputs["vein_p"] = fname

            if result.coupling_r is not None:
                fname = f"{args.prefix}_coupling_r{nii_ext}"
                save_nifti(_to_volume(result.coupling_r.numpy()), fname, reference_img=ref_img_path)
                outputs["coupling_r"] = fname

        # sPR diagnostics: where phase was borrowed from, and how well it fit.
        if result.spr_donor_corr is not None and result.spr_donor_offset is not None:
            fname = f"{args.prefix}_spr_donor_corr{nii_ext}"
            save_nifti(_to_volume(result.spr_donor_corr.numpy()), fname, reference_img=ref_img_path)
            outputs["spr_donor_corr"] = fname

            fname = f"{args.prefix}_spr_donor_offset{nii_ext}"
            save_nifti(
                _to_volume(result.spr_donor_offset.numpy()),
                fname,
                reference_img=ref_img_path,
            )
            outputs["spr_donor_offset"] = fname

        # Explore-mode diagnostic: the per-voxel window/order the search chose.
        if result.sgf_window_map is not None and result.sgf_order_map is not None:
            fname = f"{args.prefix}_sgf_window{nii_ext}"
            save_nifti(
                _to_volume(result.sgf_window_map.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["sgf_window"] = fname

            fname = f"{args.prefix}_sgf_order{nii_ext}"
            save_nifti(
                _to_volume(result.sgf_order_map.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["sgf_order"] = fname

        if args.save_intermediates:
            interm_specs = [
                ("mag_dt", result.mag_detrended, True, "poly-detrended magnitude"),
                ("pha_dt", result.pha_detrended, True, "poly-detrended phase"),
                ("pha_dt_filt", result.pha_detrended_filt, True, "detrended + filtered phase"),
                ("mag_res", result.mag_residual, True, "magnitude task-residual"),
                ("pha_res_filt", result.pha_residual_filt, True, "phase task-residual + filtered"),
            ]
            for key, data, is_4d, label in interm_specs:
                if data is None:
                    continue
                fname = f"{args.prefix}_{key}{nii_ext}"
                save_nifti(_to_volume(data.numpy(), is_4d=is_4d), fname, reference_img=ref_img_path)
                outputs[key] = fname
                if args.verb >= 1:
                    print(f"  {key} ({label}): {fname}")

    if args.verb >= 1:
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        print(f"\n{'=' * 70}")
        print("Phase regression complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")
    else:
        print(f"Phase regression complete. Main output: {outputs['corrected']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
