#!/usr/bin/env python3
"""
ffs_fitbasis — per-condition response estimation: amplitude, latency, width.

This is the parametric / basis-set counterpart to [[ffs_deconvolve]].
Where ``ffs_deconvolve`` does non-parametric FIR/TENT/CSPLIN
deconvolution (one regressor per lag, no shape assumption),
``ffs_fitbasis`` fits the HRF as a small linear combination of basis
functions and optionally applies a Gaussian shape prior so the
combination can't produce nonsense HRFs.

Three things this tool can do that ``ffs_deconvolve`` cannot:

1. **A response shape per voxel** (``-hrf library``/``pighs``, or an
   imported ``-hrf-index``), with ``-derivatives`` letting each condition
   depart from it — and ``-save-shape`` reporting that departure as
   latency and width in seconds.
2. **FLOBS** (``-flobs``) — K=3 eigenHRFs derived from half-cosine HRF
   samples (Woolrich, Behrens, Smith 2004 TR04MW2), with an empirical
   MVN(m, C) shape prior.
3. **Single-trial fits** (``-single-trials``) — one block of basis
   regressors per trial, with the prior applied per-trial.  This is
   where unconstrained derivative fits famously go off the rails at short
   ISIs; the constraint pulls each trial's coefficients back toward
   sensible HRF shapes.

The constraint is **generalised ridge** under the hood
(:func:`fastfuncstuff.design.flobs.fit_basis_constrained_ridge`):

.. math::

    \\hat{\\beta} = (X'X + \\lambda P)^{-1} (X' y + \\lambda P \\bar{m})

with ``P = block-diag(C^{-1})``.  Three choices of (m, C):

- ``-reg none``  → no constraint, plain OLS.
- ``-reg ridge`` → :func:`spmg_prior` for SPMG models (canonical free,
  derivative coefficients tightly shrunk to zero), or :func:`ridge_prior`
  isotropic for FLOBS.  Hand-picked weights; transparent.
- ``-reg mvn``   → :func:`flobs_prior` for FLOBS (empirical (m, C) from
  half-cosine samples), :func:`spmg_prior` defaults for SPMG.  This is
  the closest thing to filmbabe (TR04MW2 §3) implemented as a
  closed-form generalised ridge instead of full Variational Bayes —
  matches in the shape-constraint piece, skips AR(P)/MRF.

Outputs (per condition for the default fit, per trial with
``-single-trials``):

- ``{prefix}_fitbasis_basis.tsv``                — the K basis functions
- ``{prefix}_fitbasis_amplitude.nii.gz``        — peak amplitude, one
  sub-brick per condition, + xvalR2 / taskR2
- ``{prefix}_fitbasis_shape.nii.gz``            — latency (s), validity and
  friends, interleaved per condition, + xvalR2 / taskR2
- ``{prefix}_fitbasis_basisweights.nii.gz``     — raw coefficients
  (``-no-basis`` to skip)
- ``{prefix}_fitbasis_xvalr2.nii.gz``           — held-out R² (``-xval-r2``)
- ``{prefix}_fitbasis_hrf_index.nii.gz``        — selected curve per voxel
- ``{prefix}_fitbasis_iresp_<cond>.nii.gz``     — reconstructed HRF (4-D)
- ``{prefix}_fitbasis_metadata.json``            — full provenance

With ``-single-trials``: filenames carry ``_trial<NNN>`` suffix
instead of ``_<cond>``, and amplitudes are stacked into a single 4-D
volume per condition (time = trial number) — the GLMsingle-style
output used for 2nd-level analyses across trials.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from fastfuncstuff.cli_utils import (
        add_device_arg,
        add_load_threads_arg,
        add_ortvec_arguments,
        add_trim_args,
        add_verbose_arg,
        append_nuisance_blocks,
        apply_trim_to_timing,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
        parse_input_files,
        parse_prefix,
        preflight_check,
        run_lengths_from_starts,
        setup_device,
        trim_spec_from_args,
    )
    from fastfuncstuff.design.builder import (
        pack_for_shared_task_glm,
    )
    from fastfuncstuff.design.flobs import (
        ARMAWhitenCell,
        FLOBSBasis,
        FLOBSFitResult,
        bin_and_whiten_arma11,
        compute_per_voxel_residuals,
        compute_vb_block_trace,
        compute_xval_r2_per_voxel,
        cv_basis_constrained_ridge,
        decouple_amplitude_prior,
        estimate_and_apply_arma11_prewhitening,
        estimate_arma11_per_voxel,
        fit_basis_cone_prior,
        fit_basis_constrained_ridge,
        fit_basis_fracridge,
        fit_basis_lss,
        flobs_prior,
        generate_flobs_basis,
        generate_spmg_basis,
        ridge_prior,
        spmg_prior,
        vb_update_beta_size,
    )
    from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run
    from fastfuncstuff.design.matrices import save_iresp
    from fastfuncstuff.io.afni import save_nifti
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Repo help style, minus two sources of pure noise.

    ``(default: None)`` / ``(default: False)`` are dropped: for every flag
    here those mean "off" or "auto", which the help text already says, and
    they accounted for a third of the ``(default: ...)`` lines.

    Hyphen/underscore spelling variants collapse to one entry.  Every
    ``-foo-bar`` also accepts ``-foo_bar`` (documented once in the epilog);
    printing both doubled the width of ~20 entries to say nothing.  An alias
    that is *not* a mere spelling variant — ``-parametrisation``, ``-quiet``
    — still prints, because it carries information.
    """

    def _get_help_string(self, action):
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        canon = action.option_strings[0]

        def norm(s: str) -> str:
            return s.replace("_", "-")

        shown = [canon] + [o for o in action.option_strings[1:] if norm(o) != norm(canon)]
        if action.nargs == 0:
            return ", ".join(shown)
        args = self._format_args(action, self._get_default_metavar_for_optional(action))
        return ", ".join(shown) + " " + args


_USAGE = (
    "ffs_fitbasis -input RUN [RUN ...] -prefix OUT\n"
    "                    (-events TSV [TSV ...] | -onsets FILE [FILE ...] -durations D [D ...])\n"
    "                    [-parametrization {linear,shift}] [-single-trials] [-xval-r2]\n"
    "                    [-hrf SRC] [-derivatives D] [-reg R] [-save-shape] [...]\n"
    "\n"
    "                    Run with -h for the grouped flag list."
)

_DESCRIPTION = """\
[BETA] Per-condition or per-trial response estimation: amplitude alone, or
amplitude AND latency together.

  Two models, chosen by -parametrization.  This is the first decision and it
  decides which of the groups below apply to you:

    linear   The -hrf curve plus its -derivatives, per block, optionally
             constrained by -reg.  Estimates AMPLITUDE — and, PER CONDITION
             with -save-shape, latency in seconds (plus width, with
             -derivatives time+width) by inverting the derivative ratios.
             That readout does NOT survive per trial: one trial's base
             coefficient is too noisy and the ratio runs to +-5 s
             (measured r=0.03).

    shift    ONE column per block: the HRF shifted exactly, with a free
             amplitude and a box-bounded delay reported in seconds.  This is
             the path that recovers per-trial LATENCY (measured r=0.69).
             Works with any -hrf shape and needs no derivative at all, so
             -derivatives and -reg are ignored here.

  A "block" is one condition, or one trial with -single-trials.

  Rationale, measurements and open questions live in the wiki note
  "ffs_fitbasis latency rework"; this help states behaviour only.
"""

_EPILOG = """\
EXAMPLES

  # Simplest useful run: one amplitude per condition, no prior.
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -derivatives none -prefix out/sub01_amp

  # Per-trial amplitudes, GLMsingle-style (the ffs_ridge alternative).
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -single-trials -reg cone \\
               -mask brain_mask.nii.gz -prefix out/sub01

  # Per-trial amplitude AND latency, with the held-out validator that says
  # whether the latencies are real.
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -parametrization shift -single-trials -xval-r2 \\
               -mask brain_mask.nii.gz -prefix out/sub01_shift

  # Same, but each voxel keeps its own HRF from an ffs_hrfopt index map.
  # Absolute delay then splits between shape TTP and the delay map — read
  # _shift_delay_dev_<cond> for the clean per-trial quantity.
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -parametrization shift -single-trials -xval-r2 \\
               -shift-shape-index out/sub01_hrf_index.nii.gz \\
               -mask brain_mask.nii.gz -prefix out/sub01_shift

  # Per-voxel HRF chosen by held-out R2, then per-condition latency
  # relative to each voxel's own curve.  Read _latency_dev for contrasts.
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -hrf library -derivatives time+width -save-shape \\
               -mask brain_mask.nii.gz -prefix out/sub01_hrf

  # Same, reusing an ffs_hrfopt selection instead of re-running it.
  ffs_fitbasis -input run*.nii.gz -events sub01_run*_events.tsv \\
               -hrf library -hrf-index out/sub01_hrf_index.nii.gz \\
               -derivatives time+width -save-shape -prefix out/sub01_hrf

  # Per-condition fit with AFNI-style onsets and denoising components.
  ffs_fitbasis -input run*.nii.gz -onsets faces.1D houses.1D -durations 2.0 \\
               -reg cone -ortvec_glob 'out/sub01_run*_pcs.1D' pcs \\
               -polort 4 -prefix out/sub01_cond

READING THE OUTPUT

  Both parametrisations write the SAME labelled 4-D buckets, so a linear
  fit and a shift fit of the same data compare sub-brick for sub-brick.
  Every bucket you might threshold ends with the same two QC sub-bricks:

      xvalR2   held-out R² (-xval-r2).  The honest referee.
      taskR2   in-sample task R², task variance over NON-drift variance.
               Shows WHERE the task explains signal; free parameters
               inflate it, so never threshold on it.

  _amplitude       one sub-brick per condition, data units
  _shape           interleaved per condition, value beside the map that
                   gates it: <cond>_latency (s), <cond>_latency_dev,
                   <cond>_valid, plus <cond>_shape_r2 / _fwhm /
                   _dispersion on the linear side
  _diagnostics     taskR2 and its variants; shift adds taskR2_tau0,
                   taskR2_incl_drift, fstat, amp_lambda
  _hrf_index       curve selected per voxel; shift adds shape_ttp and
                   mean_timing (TTP + mean delay), which is how the
                   shape/delay timing confound stays decomposable
  _hrf_shapes.tsv  the candidate curves themselves
  _xvalr2          held-out R²; shift adds xvalR2_tau0 and
                   xvalR2_delay_gain — the latter answers "are the
                   delays real": >0 yes, <=0 no

  READ _latency_dev, not _latency, whenever -hrf named a SET.  Absolute
  latency is then measured against the curve that voxel selected, so a
  curve peaking late shifts every condition in it alike; the deviation is
  what survives.  Under shift the same confound is explicit — the chosen
  curve's TTP absorbs part of the timing, which is why _hrf_index carries
  shape_ttp and mean_timing.

  linear-only extras:
           _basisweights    raw coefficients, <cond>_base / _dLatency /
                            _dWidth.  -no-basis to skip.
           _iresp_<cond>    reconstructed response, 4-D over time
           _shape sub-bricks, in full:
                            <cond>_latency      seconds
                            <cond>_latency_dev  latency minus this voxel's
                                                own mean across conditions.
                                                READ THIS when -hrf named a
                                                set: absolute latency is
                                                measured against the curve
                                                that voxel selected.
                            <cond>_fwhm         seconds (-derivatives
                            <cond>_dispersion   time+width only)
                            <cond>_shape_r2     calibration quality
                            <cond>_valid        0 = clamped to the envelope;
                                                threshold on this
           _basisweights    raw coefficients, <cond>_base / _dLatency /
                            _dWidth.  -no-basis to skip.
           _hrf_index       curve selected per voxel (1-based) + its
                            selection R².  Feeds -hrf-index.
           _iresp_<cond>    reconstructed response, 4-D over time
           _xvalr2          the held-out map on its own, for convenience

  In-sample R² is NOT evidence of a task response, and r2 minus r2_tau0 is
  NOT evidence of latency: n_blocks free parameters buy in-sample fit even
  on data simulated with none.  Threshold the held-out maps.

NOTES

  - Every -foo-bar flag also accepts -foo_bar.  Only one spelling is listed.
  - -flobs-dt and -flobs-window set the HRF sampling grid for BOTH
    parametrisations despite the name.
  - Never combine -polort 0 with per-run nuisance blocks (rank-deficient).
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_fitbasis",
        usage=_USAGE,
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=_HelpFormatter,
    )

    # ── STEP 1: the two decisions that scope everything else ───────────
    # -parametrization used to sit in group 5 of 11, AFTER the -model /
    # -reg flags it renders inert, so a top-down reader spent their
    # attention on flags that do not apply to them.  It leads now.
    step1 = parser.add_argument_group(
        "STEP 1 — Parametrisation and unit of analysis",
        description=(
            "The two flags that decide which groups below apply to you.  Read these first.\n"
            "The rest of the help is in the order you decide things: what shape,\n"
            "how much freedom around it, how it is constrained, what comes out."
        ),
    )
    step1.add_argument(
        "-parametrization",
        "-parametrisation",
        dest="parametrization",
        choices=["linear", "shift"],
        default="linear",
        help=(
            "linear — K basis columns per block, free betas, constrained by "
            "-reg; estimates amplitude.  shift — one exactly-shifted column "
            "per block, giving a free amplitude AND a bounded delay in "
            "seconds; the only path here that recovers latency.  "
            "-model / -reg are ignored under shift."
        ),
    )
    step1.add_argument(
        "-single-trials",
        "-single_trials",
        dest="single_trials",
        action="store_true",
        help=(
            "Fit one block per TRIAL instead of per condition (both "
            "parametrisations).  Output is stacked 4-D per condition, time = "
            "trial number, for 2nd-level analyses across trials."
        ),
    )

    req = parser.add_argument_group("Required arguments")
    req.add_argument("-input", nargs="+", required=True, help="Input fMRI run files.")
    req.add_argument("-prefix", required=True, help="Output prefix (e.g. out/sub01_fb).")

    onset_grp = parser.add_argument_group("Event timing (choose one)")
    onset_grp.add_argument(
        "-onsets", nargs="+", default=None, help="AFNI-format onset files, one per condition."
    )
    onset_grp.add_argument(
        "-durations",
        nargs="+",
        default=None,
        help=(
            "Stimulus durations (s); one per condition (or single value).  "
            "Required with -onsets.  Convolved into the basis, so a block "
            "design is modelled as a block — omitting it (0) models every "
            "event as an impulse, which mis-times the predicted peak by "
            "roughly D/2 and mis-scales the amplitude."
        ),
    )
    onset_grp.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files, one per run.  Paired with -input BY POSITION, never sorted.",
    )
    # Each event-related flag accepts both hyphen and underscore forms
    # (``-event-cols`` and ``-event_cols``) so muscle memory from AFNI /
    # older ffs_* tools works.  argparse dispatches both to the same
    # ``args.event_cols`` attribute via the canonical dest.
    onset_grp.add_argument(
        "-event-ignore",
        "-event_ignore",
        dest="event_ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to drop from BIDS events.",
    )
    onset_grp.add_argument(
        "-event-cols",
        "-event_cols",
        dest="event_cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Override BIDS column names (default: onset duration trial_type).",
    )
    onset_grp.add_argument(
        "-round-onsets",
        "-round_onsets",
        dest="round_onsets",
        nargs="?",
        const=0.7,
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Snap onsets to nearest TR (default threshold 0.7).",
    )
    onset_grp.add_argument(
        "-round-durations",
        "-round_durations",
        dest="round_durations",
        type=int,
        default=None,
        metavar="PLACES",
        help="Round event durations to N decimals before grouping.",
    )

    # ══ STEP 2 — the response shape ════════════════════════════════════
    hrf_grp = parser.add_argument_group(
        "STEP 2 — Response shape (the HRF).  BOTH parametrisations",
        description=(
            "Give each voxel its own response shape, then model on top of it.\n"
            "Applies to BOTH parametrisations, and is the ffs_hrfopt approach\n"
            "folded in: score every candidate curve against the whole dataset\n"
            "and keep the best one per voxel.\n"
            "\n"
            "  linear  the derivative columns are built around each voxel's own\n"
            "          curve, so -save-shape reports each CONDITION's departure\n"
            "          from a shape that already fits that voxel.\n"
            "  shift   the exactly-shifted column uses each voxel's own curve.\n"
            "\n"
            "-shift-shapes / -shift-shape-index / -shift-n-shapes remain as\n"
            "aliases for the flags below."
        ),
    )
    hrf_grp.add_argument(
        "-hrf",
        "-basis-hrf",
        "-basis_hrf",
        "-hrf-shapes",
        "-hrf_shapes",
        "-shift-shapes",
        "-shift_shapes",
        "-shift-hrf",
        "-shift_hrf",
        dest="hrf",
        default="canonical",
        metavar="SOURCE",
        help=(
            "The response shape everything else is built on.\n"
            "  ONE curve for every voxel:\n"
            "    canonical   the SPM curve (DEFAULT)\n"
            "    glmsingle   the GLMsingle single curve\n"
            "    FILE        one column of numbers sampled at -flobs-dt, "
            "e.g. a subject- or ROI-level curve from ffs_hrfopt\n"
            "  A SET, selected PER VOXEL (see -hrf-select):\n"
            "    library     the 20 double-gammas ffs_hrfopt uses\n"
            "    pighs       half-cosine curves stratified over peak time\n"
            "    flobs       curves drawn from the FLOBS prior\n"
            "Naming a set is what turns per-voxel selection on.  Getting the "
            "shape right first is the point: the -derivatives columns then "
            "encode each CONDITION's departure from a curve that already fits "
            "that voxel, instead of spending themselves correcting a wrong "
            "average shape."
        ),
    )
    hrf_grp.add_argument(
        "-hrf-index",
        "-hrf_index",
        "-shift-shape-index",
        "-shift_shape_index",
        dest="hrf_index",
        default=None,
        metavar="MAP",
        help=(
            "Skip selection and IMPORT the per-voxel curve assignment — "
            "sub-brick 0 of {prefix}_hrf_index.nii.gz from ffs_hrfopt or from "
            "an earlier ffs_fitbasis run (1-based).  -hrf must then name the "
            "SAME set the indices were fit against (default 'library'); "
            "nothing can detect a mismatch beyond the index range.  Must be on "
            "the -input grid, from the same -mask."
        ),
    )
    hrf_grp.add_argument(
        "-hrf-select",
        "-hrf_select",
        dest="hrf_select",
        choices=["full", "xval", "none"],
        default="full",
        help=(
            "How to pick each voxel's curve when -hrf names a SET.  "
            "full (DEFAULT) scores every candidate in-sample across the whole "
            "dataset, which is GLMsingle's FITHRF and what ffs_hrfopt does by "
            "default.  xval scores leave-one-run-out instead — slower, and the "
            "stricter referee if you suspect the selection is fitting noise.  "
            "none takes curve 0 for every voxel.  Ignored with -hrf-index."
        ),
    )
    hrf_grp.add_argument(
        "-hrf-n-shapes",
        "-hrf_n_shapes",
        "-shift-n-shapes",
        "-shift_n_shapes",
        dest="hrf_n_shapes",
        type=int,
        default=20,
        metavar="N",
        help="Candidate count for -hrf pighs / flobs (library is fixed at 20).",
    )

    # ══ LINEAR parametrisation ═════════════════════════════════════════
    model_grp = parser.add_argument_group(
        "STEP 3 — Freedom around that shape.  [linear]",
        description=(
            "How much each CONDITION may depart from the STEP 2 curve.\n"
            "Ignored with -parametrization shift."
        ),
    )
    model_grp.add_argument(
        "-derivatives",
        "-deriv",
        dest="derivatives",
        choices=["none", "time", "time+width"],
        default="time",
        help=(
            "How much freedom each condition gets AROUND the -hrf curve.\n"
            "  none        1 column per condition — amplitude only, a plain "
            "GLM against that shape.\n"
            "  time        DEFAULT.  + the latency derivative, so a condition "
            "can respond earlier or later than the curve.\n"
            "  time+width  + the width derivative, so it can also be narrower "
            "or broader.\n"
            "-save-shape turns these coefficients back into seconds.  With "
            "-hrf canonical these are exactly SPMG1 / SPMG2 / SPMG3; with any "
            "other curve they are the same construction around that curve."
        ),
    )
    model_grp.add_argument(
        "-flobs",
        dest="use_flobs",
        action="store_true",
        help=(
            "Use K FLOBS eigen-HRFs with an empirical MVN(m, C) prior instead "
            "of a curve plus derivatives.  A different basis family, so -hrf "
            "and -derivatives do not apply and -save-shape cannot read "
            "latency off it."
        ),
    )
    # ══ STEP 4 — regularisation ════════════════════════════════════════
    reg_grp = parser.add_argument_group(
        "STEP 4 — Regularisation.  [linear]",
        description="Ignored with -parametrization shift.",
    )
    reg_grp.add_argument(
        "-reg",
        choices=["none", "cone", "mvn-shape", "fracridge", "ridge", "mvn"],
        default="none",
        help=(
            "Shape prior on the basis coefficients.\n"
            "  none      DEFAULT.  Plain OLS — no prior at all.\n"
            "  cone      Scale-invariant: constrains the shape "
            "DIRECTION of beta only, leaving amplitude entirely free, and "
            "keeps the sign so negative BOLD survives.  Use this.\n"
            "  mvn-shape constrains only the shape direction orthogonal to "
            "the prior mean; amplitude unconstrained.\n"
            "  fracridge no HRF prior; CV picks each voxel's fraction of "
            "||beta_OLS|| to keep (see -fracs).\n"
            "  ridge, mvn  legacy — see the Legacy group at the end."
        ),
    )

    reg_opts = parser.add_argument_group(
        "STEP 4b — Prior strength.  [linear]",
        description="How hard the -reg prior pulls.  Ignored with -reg none.",
    )
    reg_opts.add_argument(
        "-prior-weight",
        "-prior_weight",
        dest="prior_weight",
        default="auto",
        metavar="VALUE",
        help=(
            "'auto' uses the Bayesian-optimal weight sigma^2 from an OLS "
            "pre-pass.  A float multiplies it (2.0 = twice as strong)."
        ),
    )
    reg_opts.add_argument(
        "-lambda-mode",
        "-lambda_mode",
        dest="lambda_mode",
        choices=["global", "voxelwise", "auto"],
        default="auto",
        help=(
            "Per-voxel lambda.  auto → voxelwise with -single-trials, global "
            "otherwise.  voxelwise gives high-SNR voxels less shrinkage; "
            "costs little (voxels are binned by sigma^2 quantile)."
        ),
    )
    reg_opts.add_argument(
        "-lambda-n-bins",
        "-lambda_n_bins",
        dest="lambda_n_bins",
        type=int,
        default=20,
        metavar="N",
        help="sigma^2 quantile bins for -lambda-mode voxelwise.",
    )
    reg_opts.add_argument(
        "-fracs",
        dest="fracs",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        metavar="LIST",
        help=(
            "-reg fracridge ONLY.  Comma-separated grid of the fraction of "
            "||beta_OLS|| to retain (1.0=OLS, →0=max shrinkage); CV picks "
            "the best per voxel."
        ),
    )

    noise_opts = parser.add_argument_group(
        "[linear] Temporal noise model",
        description="Ignored with -parametrization shift.",
    )
    noise_opts.add_argument(
        "-prewhiten",
        dest="prewhiten",
        choices=["none", "arma11", "arma11-voxel"],
        default="none",
        help=(
            "none — i.i.d. white noise.  arma11 — one global ARMA(1,1) via "
            "REML.  arma11-voxel — per-voxel (a, b) via REML grid search "
            "(3dREMLfit / filmbabe style), which suppresses the "
            "trial-to-trial amplitude oscillation autocorrelated noise "
            "induces.  Not supported with -reg fracridge."
        ),
    )
    noise_opts.add_argument(
        "-vb-iters",
        "-vb_iters",
        dest="vb_iters",
        type=int,
        default=0,
        metavar="N",
        help=(
            "-prewhiten arma11-voxel ONLY.  Variational-Bayes iterations "
            "re-estimating (a, b) from the constrained-fit residuals.  0 = "
            "single pass; 2-3 is the FSL norm and more rarely helps."
        ),
    )
    noise_opts.add_argument(
        "-vb-tol",
        "-vb_tol",
        dest="vb_tol",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Stop the VB loop when median |da|+|db| falls below this.",
    )
    noise_opts.add_argument(
        "-vb-update-prior",
        dest="vb_update_prior",
        action="store_true",
        help=(
            "Also update the per-voxel prior precision each VB iteration "
            "(the full filmbabe scheme).  Only meaningful with -reg in "
            "{ridge, mvn, mvn-shape}."
        ),
    )

    st_opts = parser.add_argument_group(
        "[linear] Single-trial estimator",
        description="Requires -single-trials.  Ignored with -parametrization shift.",
    )
    st_opts.add_argument(
        "-lss",
        dest="lss",
        action="store_true",
        help=(
            "Least-Squares-Separate (Mumford 2012 / 3dLSS / GLMsingle 'L') "
            "instead of all-at-once LSA: one small design per trial, which "
            "reduces trial-to-trial collinearity at tight ISIs.  Supports "
            "-reg in {none, ridge, mvn, mvn-shape}; not -reg fracridge, "
            "-prewhiten arma11-voxel, -vb-iters, or -prior-from."
        ),
    )
    st_opts.add_argument(
        "-lss-exclude",
        "-lss_exclude",
        dest="lss_exclude",
        nargs="+",
        default=[],
        metavar="COND",
        help=(
            "Conditions that contribute their summed regressor to every LSS "
            "design but are not fit per-trial; their betas come from the "
            "parallel LSA pre-fit."
        ),
    )
    st_opts.add_argument(
        "-prior-from",
        "-prior_from",
        dest="prior_from",
        choices=["none", "per-condition"],
        default="none",
        help=(
            "per-condition — run a per-condition fit first and use its betas "
            "as the prior mean for every trial of that condition at that "
            "voxel (the GLMsingle shrinkage philosophy).  Rejects -reg "
            "mvn-shape and -reg fracridge."
        ),
    )

    flobs_opts = parser.add_argument_group(
        "STEP 3b — FLOBS basis (-flobs)",
        description=(
            "-flobs-dt and -flobs-window also set the HRF sampling grid\n"
            "for -parametrization shift, despite the name."
        ),
    )
    flobs_opts.add_argument(
        "-flobs-n-basis",
        "-flobs_n_basis",
        dest="flobs_n_basis",
        type=int,
        default=3,
        metavar="K",
        help="Number of FLOBS eigenHRFs (TR04MW2 used 3).",
    )
    flobs_opts.add_argument(
        "-flobs-n-samples",
        "-flobs_n_samples",
        dest="flobs_n_samples",
        type=int,
        default=1000,
        metavar="N",
        help="Number of half-cosine HRF samples for the basis SVD.",
    )
    flobs_opts.add_argument(
        "-flobs-window",
        "-flobs_window",
        dest="flobs_window",
        type=float,
        default=32.0,
        metavar="SECONDS",
        help="HRF duration (s).  Used by both parametrisations.",
    )
    flobs_opts.add_argument(
        "-flobs-dt",
        "-flobs_dt",
        dest="flobs_dt",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="HRF sample spacing (s).  Used by both parametrisations.",
    )
    flobs_opts.add_argument(
        "-flobs-seed",
        "-flobs_seed",
        dest="flobs_seed",
        type=int,
        default=42,
        help="Seed for the half-cosine sampler.",
    )

    delay_grp = parser.add_argument_group(
        "[shift] Delay search",
        description="Requires -parametrization shift.",
    )
    delay_grp.add_argument(
        "-tau-max",
        "-tau_max",
        dest="tau_max",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help=(
            "Hard bound on the per-block delay.  Set it wide enough to "
            "contain the real spread: truth OUTSIDE the bound does NOT clamp "
            "gracefully — the amplitude solve compensates through "
            "overlapping trials and emits alternating signed amplitudes."
        ),
    )
    delay_grp.add_argument(
        "-tau-step",
        "-tau_step",
        dest="tau_step",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Delay grid spacing.  Costs linearly in the gram table.",
    )
    delay_grp.add_argument(
        "-delay-prior-sd",
        "-delay_prior_sd",
        dest="delay_prior_sd",
        type=float,
        default=0.75,
        metavar="SECONDS",
        help=(
            "Std of a Gaussian prior shrinking each block's delay toward the "
            "voxel's OWN mean delay (not toward zero, which would bias every "
            "genuine delay).  Keep it on: delay is chosen by maximising fit, "
            "so at low SNR the winner partly fits noise and inflates "
            "amplitude with it.  Pass 0 to disable."
        ),
    )
    delay_grp.add_argument(
        "-amp-ridge",
        "-amp_ridge",
        dest="amp_ridge",
        default="auto",
        metavar="auto|FRAC",
        help=(
            "Ridge on the per-trial amplitude solve.  Keep it on: the delay "
            "search is free to slide two overlapping trials into "
            "near-coincidence, which improves in-sample fit via a "
            "(+huge, -huge) amplitude pair.  Measured on a 192-trial "
            "2.05s-ISI design, cond(XtX) is 61.7 at delay=0 but 1.1e5 at the "
            "fitted delays.\n"
            "  auto (DEFAULT) — per-voxel empirical Bayes, λ_v = σ²_v/τ²_v, "
            "with τ² by method of moments off the delay=0 fit.  σ² spans "
            "orders of magnitude across a brain, so no single fixed value "
            "suits both high- and low-SNR voxels; measured better than every "
            "fixed value at BOTH extremes.  Costs nothing — σ² is already "
            "computed for the delay prior.  Emits a λ diagnostic map.\n"
            "  FRAC — a fixed factor relative to mean(diag(XtX)).  Amplitude "
            "recovery vs known truth: 0.58 at 1e-8 (off), 0.74 at 1e-3, 0.68 "
            "at 3e-2, against 0.69 for not fitting latency at all — so with "
            "this off the delays cost more than they buy.  Above ~1e-2 "
            "amplitudes shrink and delay recovery collapses.\n"
            "  0 — off (numerical floor only), the pre-fix behaviour."
        ),
    )
    delay_grp.add_argument(
        "-shift-sweeps",
        "-shift_sweeps",
        dest="shift_sweeps",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Coordinate-descent sweeps over blocks.  Affects the delay "
            "SCALE, not just convergence — the prior's centre is "
            "re-estimated each sweep, so too few leaves delays compressed "
            "toward zero.  Raise it if absolute magnitudes matter."
        ),
    )

    # ══ Both parametrisations ══════════════════════════════════════════
    cv_opts = parser.add_argument_group(
        "Validation",
        description=(
            "In-sample R² does not identify task-responsive voxels, and the\n"
            "in-sample latency gain does not show latency is real — free\n"
            "parameters buy in-sample fit regardless.  These are the maps that\n"
            "answer those questions."
        ),
    )
    cv_opts.add_argument(
        "-xval-r2",
        "-xval_r2",
        dest="xval_r2",
        action="store_true",
        help=(
            "Emit held-out (LORO) R².  Both parametrisations.  Under shift "
            "it also writes _xvalr2_tau0 and _xvalr2_delay_gain — the latter "
            "is the map that says whether the delays are real (>0 = yes).  "
            "Single-trial mode scores condition-level generalisation, since "
            "per-trial parameters cannot predict a held-out run."
        ),
    )
    cv_opts.add_argument(
        "-cv-leave-n-out",
        "-cv_leave_n_out",
        dest="cv_leave_n_out",
        type=int,
        default=1,
        metavar="N",
        help="Runs left out per fold (1 = LORO).  Both parametrisations.",
    )
    cv_opts.add_argument(
        "-cv-runs",
        "-cv_runs",
        dest="cv_runs",
        action="store_true",
        help=(
            "[linear] Sweep -prior-weight over a grid with LORO and emit "
            "per-voxel held-out R² per weight, plus a per-voxel argmax map.  "
            "Use it to check the constraint helps rather than over-shrinks."
        ),
    )
    cv_opts.add_argument(
        "-cv-grid",
        "-cv_grid",
        dest="cv_grid",
        default="0.1,0.3,1.0,3.0,10.0",
        metavar="W1,W2,...",
        help="[linear] Prior-weight multipliers for -cv-runs.",
    )

    proc = parser.add_argument_group("Processing")
    proc.add_argument(
        "-tr", type=float, default=None, help="TR in seconds; read from header if omitted."
    )
    proc.add_argument("-mask", default=None, help="Brain mask NIfTI.")
    proc.add_argument(
        "-atlas",
        default=None,
        metavar="LABELS",
        help=(
            "Integer label volume, aligned to -input.  Averages each parcel's "
            "voxel timeseries and fits ONE timeseries per parcel instead of "
            "one per voxel.  Averaging BEFORE the fit (not after) is the "
            "point: the delay search itself sees the sqrt(N)-cleaner signal, "
            "rather than smoothing noisy per-voxel estimates afterwards.  "
            "Results are painted back into every voxel of their parcel, so "
            "all the usual maps still come out, plus a per-parcel TSV.  "
            "Turns a 292k-voxel fit into a few hundred rows, so sweeps over "
            "-tau-max / -delay-prior-sd become interactive.  Currently "
            "-parametrization shift only."
        ),
    )
    proc.add_argument(
        "-do_blur",
        "-do-blur",
        dest="do_blur",
        type=float,
        default=None,
        metavar="FWHM",
        help=(
            "3-D Gaussian spatial smoothing, FWHM in mm, applied BEFORE masking "
            "so edges do not bleed.  Typical 4-8 mm.  Raises SNR everywhere at "
            "the cost of spatial specificity — for a targeted question prefer "
            "-atlas, which buys the same averaging inside regions you chose."
        ),
    )
    proc.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Per-run polynomial drift order.  None → auto from run duration.",
    )
    add_device_arg(proc, default="auto")
    proc.add_argument(
        "-debug-design",
        "-debug_design",
        dest="debug_design",
        action="store_true",
        help="Print design rank/conditioning before the fit.",
    )
    add_load_threads_arg(proc)
    add_trim_args(proc)
    add_verbose_arg(proc, default=1)

    nuis_grp = parser.add_argument_group(
        "External nuisance regressors",
        description=(
            "Motion, physio, or denoising components (ffs_denoise /\n"
            "ffs_denoisatorial PC timeseries).  Columns join the per-run\n"
            "polynomial block-diagonal, so they stay run-specific, and are\n"
            "projected out before the amplitudes are read."
        ),
    )
    add_ortvec_arguments(nuis_grp)

    out = parser.add_argument_group(
        "STEP 5 — Outputs",
        description="iresp applies to -parametrization linear only.",
    )
    # iresp save defaults differ by mode:
    #   per-condition:  ON  (small tensor, useful for inspection)
    #   single-trial:   OFF (typically tens of GB; explicit opt-in)
    # The action below records whether the user actually passed the
    # flag (`save_iresp_explicit`) so the single-trial branch can
    # honour user intent vs the per-condition default.
    out.add_argument(
        "-save-iresp",
        "-save_iresp",
        dest="save_iresp_explicit",
        action="store_true",
        default=False,
        help=(
            "Force-save the reconstructed per-block HRF as a 4-D iresp "
            "NIfTI.  Default ON per-condition (small), OFF per-trial "
            "(typically tens of GB)."
        ),
    )
    out.add_argument(
        "-no-iresp",
        "-no_iresp",
        dest="save_iresp_off",
        action="store_true",
        default=False,
        help="Force iresp save off (only PC weights + amplitude maps emitted).",
    )
    out.add_argument(
        "-no-basis",
        "-no_basis",
        "-no-basisweights",
        dest="no_basisweights",
        action="store_true",
        default=False,
        help=(
            "Skip the raw basis-coefficient bucket.  Those are the fitted "
            "coefficients themselves — one sub-brick per (condition, basis "
            "column) — which are what -save-shape converts into seconds and "
            "what the amplitude map is reconstructed from.  Useful for "
            "debugging a fit; rarely what you threshold."
        ),
    )
    out.add_argument(
        "-iresp-dt",
        "-iresp_dt",
        dest="iresp_dt",
        default=None,
        metavar="SECONDS",
        help=(
            "Time resolution (s) of SAVED iresp NIfTIs; default TR.  The "
            "internal basis stays at -flobs-dt for amplitude accuracy, so "
            "this only shrinks the files (typically 10-20x)."
        ),
    )

    out.add_argument(
        "-save-shape",
        "-save_shape",
        dest="save_shape",
        action="store_true",
        default=False,
        help=(
            "Convert the derivative coefficients into a per-condition "
            "LATENCY map in seconds (and, with -derivatives time+width, a "
            "WIDTH map as FWHM in seconds plus the width multiplier).  Needs "
            "-parametrization linear with -derivatives time or time+width, "
            "per condition "
            "(not -single-trials: the ratio needs a well-determined beta_c, "
            "which one trial does not give — use -parametrization shift "
            "there).  Read off the UNPENALISED betas, since -reg cone "
            "constrains the very shape direction being measured.  Latency is "
            "measured against the condition's own modelled response, "
            "-durations included, so zero means 'on time for this stimulus' "
            "and the value is comparable across conditions of differing "
            "length."
        ),
    )
    out.add_argument(
        "-shape-tau-max",
        "-shape_tau_max",
        dest="shape_tau_max",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help=(
            "Half-width of the calibrated latency range for -save-shape "
            "(default 1.5).  Past ~2 s the basis stops representing a "
            "shifted HRF at all (shape R2 ~0.94), so widening this trades "
            "range for meaning."
        ),
    )
    out.add_argument(
        "-shape-r2-floor",
        "-shape_r2_floor",
        dest="shape_r2_floor",
        type=float,
        default=0.95,
        metavar="R2",
        help=(
            "Drop calibration grid points the basis reproduces worse than "
            "this (default 0.95).  Sets the edge of the validity mask."
        ),
    )

    # ══ Legacy ═════════════════════════════════════════════════════════
    legacy = parser.add_argument_group(
        "Legacy — kept for reproducing older runs",
        description=(
            "These still work and are unchanged; they are collected here\n"
            "because something measured better has superseded each.\n"
            "\n"
            "  -reg mvn    An amplitude prior in data units, not a shape\n"
            "              prior: its mean comes from peak-normalised HRFs,\n"
            "              so it drags every voxel toward peak ~0.8\n"
            "              (measured: true 3.0 → 1.37, true 0.2 → 0.72).\n"
            "              This affected per-condition fits too.  Superseded\n"
            "              by -reg cone, which constrains shape only.\n"
            "  -reg ridge  Diagonal generalised ridge with hand-picked\n"
            "              weights.  Transparent, but -reg cone is the\n"
            "              principled form."
        ),
    )
    legacy.add_argument(
        "-model",
        choices=["SPMG1", "SPMG2", "SPMG3", "FLOBS"],
        default=None,
        help=(
            "Superseded by -hrf + -derivatives, which separate 'which curve' "
            "from 'how much freedom around it' — the old name said SPMG even "
            "when the curve came from a library.  SPMG1/2/3 map to "
            "-derivatives none/time/time+width, FLOBS to -flobs."
        ),
    )
    legacy.add_argument(
        "-canonical-std",
        "-canonical_std",
        dest="canonical_std",
        type=float,
        default=5.0,
        help="Prior std on the canonical coefficient.  -reg ridge/mvn with -model SPMG* only.",
    )
    legacy.add_argument(
        "-derivative-std",
        "-derivative_std",
        dest="derivative_std",
        type=float,
        default=0.3,
        help=(
            "Prior std on the temporal-derivative coefficient.  -reg "
            "ridge/mvn with -model SPMG* only.  Tuning this to extract "
            "LATENCY is futile — there is no valid operating point (left free "
            "the per-trial derivative ratio runs to ±5 s; tightened it "
            "collapses to ±0.15 s and carries nothing).  Per condition, read "
            "latency with -save-shape off an unpenalised fit; per trial, use "
            "-parametrization shift."
        ),
    )
    legacy.add_argument(
        "-dispersion-std",
        "-dispersion_std",
        dest="dispersion_std",
        type=float,
        default=0.2,
        help="Prior std on the dispersion coefficient.  -reg ridge/mvn with -model SPMG3 only.",
    )

    return parser


def _build_prior(
    *,
    model: str,
    reg: str,
    basis: FLOBSBasis,
    canonical_std: float,
    derivative_std: float,
    dispersion_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve (-model, -reg) into a concrete (m, C) prior.

    Naming:
      ``-reg none``  → identity covariance with trivial precision (the
                       caller skips applying it via prior_weight=0).
      ``-reg ridge`` → hand-picked diagonal:
                         SPMG → spmg_prior(canonical_std, derivative_std[, dispersion_std])
                         FLOBS → ridge_prior with std picked from FLOBS sample variance
      ``-reg mvn``   → empirical full-covariance:
                         FLOBS → flobs_prior(basis)
                         SPMG  → same as -reg ridge for SPMG (no empirical
                                 (m, C) to derive — would need a population
                                 of canonical/derivative-coefficient samples,
                                 which isn't a thing for SPMG).
    """
    n_basis = basis.basis_functions.shape[0]
    if reg == "cone" and model in ("SPMG1",):
        raise ValueError(
            "-reg cone needs at least 2 basis functions: with K=1 there is "
            "no shape direction to constrain, only amplitude — which the "
            "cone prior leaves free by design.  Use -reg none with SPMG1."
        )
    if reg in ("none", "fracridge"):
        # Returned but ignored downstream (fracridge uses its own
        # CV-tuned shrinkage; no MVN prior involved).
        return np.zeros(n_basis), np.eye(n_basis)

    # First build the base (m, C); then optionally decouple amplitude.
    if model == "FLOBS":
        if reg in ("mvn", "mvn-shape", "cone"):
            base_m, base_C = flobs_prior(basis)
        else:  # ridge
            std = float(np.sqrt(np.median(np.diag(basis.C))))
            base_m, base_C = ridge_prior(n_basis, coefficient_std=max(std, 1e-3))
    elif model == "SPMG1":
        base_m, base_C = ridge_prior(1, coefficient_std=canonical_std)
    elif model == "SPMG2":
        base_m, base_C = spmg_prior(canonical_std=canonical_std, derivative_std=derivative_std)
    elif model == "SPMG3":
        base_m, base_C = spmg_prior(
            canonical_std=canonical_std,
            derivative_std=derivative_std,
            dispersion_std=dispersion_std,
        )
    else:
        raise ValueError(f"Unknown model {model}")

    if reg == "cone":
        # The cone prior needs a non-zero mean: it IS the prior ray.
        # spmg_prior centres at the origin (amplitude sign-free), which
        # leaves no direction to constrain, so point m along the
        # canonical-amplitude axis.  Its scale is not a free knob —
        # with spmg_prior's C the angular strength mᵀC⁻¹m comes out to
        # exactly 1 regardless of canonical_std, and the shape
        # tightness is set by the (canonical_std / derivative_std)
        # anisotropy alone.
        if np.linalg.norm(base_m) < 1e-12:
            base_m = np.zeros(n_basis, dtype=np.float64)
            base_m[0] = float(canonical_std)
        return base_m, base_C

    if reg == "mvn-shape":
        # Need a non-zero mean to define the amplitude direction.
        if np.linalg.norm(base_m) < 1e-12:
            # SPMG default has zero mean; pick the canonical-amplitude
            # axis (first basis function) as the amplitude direction.
            base_m = np.zeros(n_basis, dtype=np.float64)
            base_m[0] = float(canonical_std)  # arbitrary positive direction
        return decouple_amplitude_prior(base_m, base_C)

    return base_m, base_C


_HRF_SETS = {"library", "pighs", "flobs"}


def _hrf_set_name(spec: str | None) -> str | None:
    """The set name in ``-hrf``, or None when it names a single curve.

    Naming a set is what turns per-voxel selection on, so this one
    predicate decides between "one curve everywhere" and "a curve per
    voxel" for both parametrisations.
    """
    if spec is None:
        return None
    key = str(spec).strip().lower()
    return key if key in _HRF_SETS else None


_BASIS_COL_NAMES = ("base", "dLatency", "dWidth")


def _save_bucket(
    arrays: list[np.ndarray],
    labels: list[str],
    path: str,
    *,
    to_volume,
    reference_img: str,
    qc: list[tuple[str, np.ndarray]] | None = None,
) -> None:
    """Write one labelled 4-D bucket, with the QC maps appended.

    Anything a user might threshold ships with the map they would
    threshold it BY in the same dataset — held-out R2, and the
    task-explained R2.  Writing them as separate files meant loading two
    datasets and hoping the grids matched; as sub-bricks an AFNI viewer
    can set the threshold from the same file.
    """
    stack = list(arrays)
    names = list(labels)
    for name, arr in qc or []:
        stack.append(arr)
        names.append(name)
    vol = to_volume(np.stack([np.asarray(a, dtype=np.float32) for a in stack], axis=1))
    save_nifti(vol, output_path=path, reference_img=reference_img, brick_labels=names)
    print(
        f"  Wrote {path}  ({len(names)} sub-bricks: {', '.join(names[:4])}"
        + (", …" if len(names) > 4 else "")
        + ")"
    )


def _print_r2(fit, reg: str) -> None:
    """R2 line that does not claim a constraint the run did not apply.

    Under ``-reg none`` both numbers come from the same OLS solve, so
    printing "OLS / constrained" side by side invited the reader to look
    for a difference that cannot exist.
    """
    if reg == "none":
        print(f"  R² mean: {fit.r2.mean():.3f}  (OLS — no prior applied)")
    else:
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  -reg {reg}: {fit.r2.mean():.3f}")


def _shape_summary(args) -> str:
    """One line naming the actual model, for the log.

    Printing "SPMG2" while the curve came from a library was the thing
    that made runs unreadable — the name has to track what was asked for.
    """
    if args.use_flobs:
        return f"FLOBS ({args.flobs_n_basis} eigen-HRFs)"
    set_name = _hrf_set_name(args.hrf)
    if args.hrf_index:
        shape = f"per-voxel from {args.hrf_index}"
    elif set_name:
        shape = f"per-voxel from {set_name} (-hrf-select {args.hrf_select})"
    else:
        shape = str(args.hrf)
    extra = {"none": "amplitude only", "time": "+ latency", "time+width": "+ latency + width"}[
        args.derivatives
    ]
    return f"{shape}, {extra}"


def _build_basis(args) -> FLOBSBasis:
    """Construct the basis FLOBSBasis container from the chosen model."""
    if args.use_flobs:
        return generate_flobs_basis(
            n_basis=args.flobs_n_basis,
            n_samples=args.flobs_n_samples,
            duration=args.flobs_window,
            dt=args.flobs_dt,
            seed=args.flobs_seed,
        )
    n_basis = 1 + {"none": 0, "time": 1, "time+width": 2}[args.derivatives]
    if str(args.hrf).strip().lower() == "canonical":
        # Keep the hand-written SPM path for the default so existing
        # runs stay bit-identical; make_derivative_basis reproduces it
        # to r > 0.999 but "reproduces" is not "is".
        return generate_spmg_basis(
            n_basis=n_basis,
            duration=args.flobs_window,
            dt=args.flobs_dt,
        )

    from fastfuncstuff.design.flobs import FLOBSBasis
    from fastfuncstuff.design.hrf import make_derivative_basis

    base = _resolve_shift_hrf(args.hrf, args.flobs_dt, args.flobs_window)
    G = make_derivative_basis(base, args.flobs_dt, n_basis)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    G = G / np.where(norms > 1e-12, norms, 1.0)
    return FLOBSBasis(
        basis_functions=G,
        eigenvalues=np.ones(n_basis, dtype=np.float64),
        m=np.zeros(n_basis, dtype=np.float64),
        C=np.eye(n_basis, dtype=np.float64),
        dt=float(args.flobs_dt),
        duration=float(args.flobs_window),
        n_samples=0,
        parametrization={"derivatives": args.derivatives, "hrf": str(args.hrf)},
    )


def _build_block_bases(
    basis: FLOBSBasis,
    durations_per_block: list[float],
) -> np.ndarray:
    """Per-block basis curves, each convolved with that block's stimulus.

    A GLM regressor for a D-second event is the HRF convolved with a
    D-second boxcar.  Blocks can differ in D (one condition is a 1 s
    cue, another an 8 s block), so the basis is per block rather than
    global.

    Rows are re-L2-normalised afterwards, matching what
    ``generate_spmg_basis`` hands out — the ``-reg`` priors are tuned
    against that scaling, and the latency calibration is measured in
    design space so it absorbs the change either way.
    """
    from fastfuncstuff.design.hrf import convolve_curves_with_duration

    out = np.empty((len(durations_per_block),) + basis.basis_functions.shape, dtype=np.float64)
    for b, dur in enumerate(durations_per_block):
        G = convolve_curves_with_duration(
            basis.basis_functions, basis.dt, float(dur), normalize=False
        )
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        out[b] = G / np.where(norms > 1e-12, norms, 1.0)
    return out


def _shape_readout(
    *,
    n_deriv: int,
    betas_ols: np.ndarray,
    per_run_designs: list[np.ndarray],
    block_onsets_per_run: list[list[np.ndarray]],
    basis_lag_times: np.ndarray,
    basis_dt: float,
    basis_mode: str,
    tr: float,
    n_tp_per_run: list[int],
    n_basis: int,
    tau_max: float,
    r2_floor: float,
    base_hrf: np.ndarray | None,
    block_durations: list[float],
) -> tuple[list[dict[str, np.ndarray]], list]:
    """Per-condition latency (and width) from the SPMG derivative ratios.

    One :class:`ShapeCalibration` per block, because the ratio a given
    latency produces depends on that block's own onset pattern.  The
    calibration targets are built with the *same* convolution routine
    that produced the design, so basis normalisation, duration handling
    and TR sampling all match by construction rather than by assumption.
    """
    from fastfuncstuff.design.basis_shape import (
        ShapeCalibration,
        build_shape_hrf_bank,
        calibrate_shape_ratios,
        invert_shape_ratios,
    )
    from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run

    taus = np.arange(-tau_max, tau_max + 1e-9, min(0.1, tau_max / 12.0))
    dispersions = np.arange(0.7, 1.451, 0.05) if n_deriv == 2 else np.array([1.0])
    duration = float(basis_lag_times[-1] + basis_dt)

    design_concat = np.concatenate([np.asarray(d, dtype=np.float64) for d in per_run_designs], 0)
    n_runs = len(per_run_designs)
    n_blocks = len(block_onsets_per_run)

    # One bank per distinct stimulus duration: the calibration targets
    # have to be regressors for the real stimulus, not for an impulse,
    # or the map is being inverted against the wrong family.
    banks: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    out: list[dict[str, np.ndarray]] = []
    calibs: list[ShapeCalibration] = []
    for b_idx in range(n_blocks):
        key = round(float(block_durations[b_idx]), 4)
        if key not in banks:
            banks[key] = build_shape_hrf_bank(
                taus,
                dispersions,
                dt=basis_dt,
                duration=duration,
                base_hrf=base_hrf,
                stim_duration=key,
            )
        bank, bank_lags, fwhm = banks[key]

        targets = np.concatenate(
            build_pc_basis_design_per_run(
                onsets_per_run=[block_onsets_per_run[b_idx][r] for r in range(n_runs)],
                pcs=bank,
                lag_times=bank_lags,
                tr=tr,
                n_timepoints_per_run=n_tp_per_run,
                basis=basis_mode,
            ),
            axis=0,
        ).astype(np.float64)

        block_design = design_concat[:, b_idx * n_basis : (b_idx + 1) * n_basis]
        calib = calibrate_shape_ratios(block_design, targets, taus, dispersions, fwhm)

        beta_c = betas_ols[:, b_idx, 0]
        # A near-zero canonical beta is what destroys the ratio; at the
        # condition level it means "no response here", not "latency is
        # huge".  Send those through as NaN so they land outside the
        # hull and come back flagged invalid.
        safe = np.where(np.abs(beta_c) > 1e-12, beta_c, np.nan)
        r_t = betas_ols[:, b_idx, 1] / safe
        r_d = betas_ols[:, b_idx, 2] / safe if n_basis == 3 else None
        out.append(invert_shape_ratios(calib, r_t, r_d, shape_r2_floor=r2_floor))
        calibs.append(calib)

    return out, calibs


def _fit_shape_groups(
    *,
    hrf_index: np.ndarray,
    shapes: np.ndarray,
    n_basis: int,
    block_durations: list[float],
    block_onsets_per_run: list[list[np.ndarray]],
    per_run_data: list[torch.Tensor],
    basis_dt: float,
    basis_mode: str,
    tr: float,
    n_tp_per_run: list[int],
    fit_one,
) -> tuple[object, np.ndarray, list[np.ndarray], dict[int, list[np.ndarray]]]:
    """Fit voxels in groups, one per distinct per-voxel HRF.

    A per-voxel response shape means a per-voxel design, which sounds
    ruinous but is not: the shapes come from a library of ~20 curves, so
    there are at most 20 distinct designs no matter how many voxels.
    Group by curve, build one design per group, fit that group's voxels.

    ``fit_one(voxel_idx, per_run_designs) -> FLOBSFitResult`` is the
    caller's existing single-design fit path, so every ``-reg`` branch
    behaves here exactly as it does without shape selection.

    Returns the stitched fit, the per-curve basis stack (small — indexed
    by curve, never expanded to per-voxel, which would run to tens of GB
    on a real volume), the group voxel-index arrays, and each occupied
    group's design so the latency calibration can reuse it.
    """
    from fastfuncstuff.design.hrf import convolve_curves_with_duration, make_derivative_basis
    from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run

    n_blocks = len(block_onsets_per_run)
    n_runs = len(per_run_data)
    n_vox = int(hrf_index.size)
    groups = [np.where(hrf_index == k)[0] for k in range(shapes.shape[0])]
    active = [(k, idx) for k, idx in enumerate(groups) if idx.size]
    print(f"  Shape groups to fit: {len(active)} of {shapes.shape[0]} curves occupied")

    betas = r2 = betas_ols = r2_ols = None
    group_bases = np.empty((shapes.shape[0], n_blocks, n_basis, shapes.shape[1]), dtype=np.float64)
    group_designs: dict[int, list[np.ndarray]] = {}
    sigma2_sum = 0.0
    pw_sum = 0.0

    for k, idx in tqdm(active, desc="  Shape groups", leave=True, disable=len(active) <= 1):
        G = make_derivative_basis(shapes[k], basis_dt, n_basis)
        for b, dur in enumerate(block_durations):
            Gb = convolve_curves_with_duration(G, basis_dt, float(dur), normalize=False)
            norms = np.linalg.norm(Gb, axis=1, keepdims=True)
            group_bases[k, b] = Gb / np.where(norms > 1e-12, norms, 1.0)

        designs: list[torch.Tensor] = []
        for r in range(n_runs):
            cols = [
                build_pc_basis_design_per_run(
                    onsets_per_run=[block_onsets_per_run[b][r]],
                    pcs=group_bases[k, b],
                    lag_times=np.arange(shapes.shape[1]) * basis_dt,
                    tr=tr,
                    n_timepoints_per_run=[n_tp_per_run[r]],
                    basis=basis_mode,
                )[0]
                for b in range(n_blocks)
            ]
            designs.append(torch.from_numpy(np.concatenate(cols, axis=1).astype(np.float32)))

        group_designs[k] = [d.numpy() for d in designs]
        gfit = fit_one(idx, designs)
        if betas is None:
            betas = np.zeros((n_vox, gfit.betas.shape[1]), dtype=np.float32)
            betas_ols = np.zeros_like(betas)
            r2 = np.zeros(n_vox, dtype=np.float32)
            r2_ols = np.zeros(n_vox, dtype=np.float32)
        betas[idx] = gfit.betas[:, : betas.shape[1]]
        betas_ols[idx] = gfit.betas_ols[:, : betas.shape[1]]
        r2[idx] = np.asarray(gfit.r2, dtype=np.float32)
        r2_ols[idx] = np.asarray(gfit.r2_ols, dtype=np.float32)
        sigma2_sum += float(gfit.sigma2_mean) * idx.size
        pw_sum += float(gfit.effective_prior_weight) * idx.size

    from fastfuncstuff.design.flobs import FLOBSFitResult

    stitched = FLOBSFitResult(
        betas=betas,
        hrfs=None,  # type: ignore[arg-type]
        r2=r2,
        betas_ols=betas_ols,
        hrfs_ols=None,  # type: ignore[arg-type]
        r2_ols=r2_ols,
        sigma2_mean=sigma2_sum / max(1, n_vox),
        effective_prior_weight=pw_sum / max(1, n_vox),
        n_iter=1,
    )
    return stitched, group_bases, groups, group_designs


def _select_hrf_per_voxel(
    *,
    shapes: np.ndarray,
    per_run_data: list[torch.Tensor],
    all_onsets: list[list[np.ndarray]],
    durations: list[float],
    run_starts: list[int],
    n_timepoints: int,
    tr: float,
    dt: float,
    polort: int,
    extra_regs_per_run: list[torch.Tensor] | None,
    select_mode: str,
    device: torch.device,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each voxel's response curve by scoring the whole dataset.

    This is ``ffs_hrfopt``'s procedure, reused rather than reimplemented
    (:func:`~fastfuncstuff.design.hrf_selection.fit_glm_hrf_library_with_xval`):
    fit every candidate curve against all the data and keep the one that
    scores best per voxel.  ``select_mode='xval'`` scores leave-one-run-out,
    which is the referee that does not reward overfitting; a single run has
    no held-out data, so it falls back to in-sample.

    Returns ``(index, score)``, both ``(n_voxels,)``, index 0-based.
    """
    from fastfuncstuff.design.builder import create_onset_matrix_microtime
    from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval

    onset_matrix = create_onset_matrix_microtime(
        all_onsets,
        list(run_starts),
        tr,
        n_timepoints,
        dt,
        stim_durations=list(durations),
        device=device,
    )
    data = torch.cat([d for d in per_run_data], dim=1)

    extra = None
    if extra_regs_per_run is not None:
        extra = torch.cat([e for e in extra_regs_per_run], dim=0)

    res = fit_glm_hrf_library_with_xval(
        data=data,
        onsets=onset_matrix,
        hrf_library=torch.from_numpy(np.ascontiguousarray(shapes)).to(device).float(),
        tr=tr,
        run_starts=list(run_starts),
        stim_durations=list(durations),
        polort=polort,
        extra_regressors=extra,
        microtime_dt=dt,
        select_mode=select_mode,
        device=device,
        verbose=verbose,
        # Only the index is wanted: ffs_fitbasis immediately refits with its
        # own derivative basis, so the engine's own full refit is discarded.
        skip_final_fit=True,
    )
    idx = res.hrf_index.detach().cpu().numpy().astype(np.int64)
    score = res.xval_r2_best.detach().cpu().numpy().astype(np.float32)
    return idx, score


def _resolve_prior_weight_arg(arg: str | float, reg: str) -> float | str:
    """Translate the CLI string into a fit_basis_constrained_ridge value."""
    if reg in ("none", "fracridge"):
        return 0.0
    val = str(arg).strip().lower()
    if val == "auto":
        return "auto"
    return float(arg)


def _resolve_shift_hrf(spec: str, dt: float, duration: float) -> np.ndarray:
    """Resolve -shift-hrf into a single response curve sampled at ``dt``.

    Any shape works — exact shifting needs no derivative — so this accepts
    the SPM canonical, the first curve of the standard library, or a
    user-supplied one-column text file (e.g. a per-voxel-cluster HRF
    exported from ``ffs_hrfopt`` / ``ffs_librarian``).
    """
    from fastfuncstuff.design.hrf import get_hrf_library, get_spm_hrf_with_derivatives

    cpu = torch.device("cpu")
    key = spec.strip().lower()
    if key == "canonical":
        h = (
            get_spm_hrf_with_derivatives(
                microtime_dt=dt, hrf_duration=duration, n_basis=1, device=cpu
            )
            .cpu()
            .numpy()[0]
        )
    elif key in ("library", "glmsingle"):
        lib = get_hrf_library(
            mode="single" if key == "glmsingle" else "library",
            microtime_dt=dt,
            hrf_duration=duration,
            device=cpu,
        )
        arr = lib.cpu().numpy()
        h = arr[0] if arr.ndim == 2 else arr
    else:
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(
                f"-shift-hrf {spec!r} is neither a keyword "
                f"(canonical, library, glmsingle) nor an existing file."
            )
        h = np.loadtxt(path, dtype=np.float64)
        if h.ndim > 1:
            h = h[:, 0] if h.shape[0] >= h.shape[1] else h[0]
    peak = float(np.max(np.abs(h)))
    if peak <= 0:
        raise ValueError(f"-shift-hrf {spec!r} produced an all-zero curve.")
    # Peak-normalise so the reported amplitude is in data units.
    return np.asarray(h, dtype=np.float64) / peak


def _load_shape_index_map(
    path: str,
    *,
    n_shapes: int,
    volume_shape,
    mask,
    n_voxels: int,
) -> np.ndarray:
    """Read an ``ffs_hrfopt`` HRF-index map into 0-based per-voxel indices.

    ``{prefix}_hrf_index.nii.gz`` is a 2-sub-brick bucket: [0] is the
    1-based HRF index, [1] the R² at that HRF.  Only sub-brick 0 is read,
    and the 1→0 base conversion happens here so exactly one place in the
    codebase knows about the off-by-one.

    Voxels outside the mask never reach the fit; voxels the selection left
    at 0 (no valid index — e.g. a voxel hrfopt's own mask excluded) fall
    back to shape 0 rather than aborting, since a mask mismatch at the
    edges is normal and shape 0 is a real curve.
    """
    from fastfuncstuff.io.afni import load_nifti

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"-hrf-index {path!r} does not exist.")
    vol = np.asanyarray(load_nifti(str(p)).dataobj)
    if vol.ndim == 4:
        vol = vol[..., 0]
    elif vol.ndim != 3:
        raise ValueError(f"-hrf-index {path!r}: expected a 3-D or 4-D volume, got {vol.shape}.")
    if tuple(vol.shape) != tuple(volume_shape):
        raise ValueError(
            f"-hrf-index {path!r} has grid {vol.shape} but the input "
            f"data has {tuple(volume_shape)}.  The index map must be on the "
            "same grid as -input (same ffs_hrfopt run, or resampled first)."
        )
    idx = np.rint(np.asarray(vol, dtype=np.float64)).astype(np.int64)
    idx = idx[mask] if mask is not None else idx.reshape(-1)
    if idx.size != n_voxels:
        raise ValueError(
            f"-hrf-index {path!r} yielded {idx.size} voxels but the "
            f"data has {n_voxels}.  Use the same -mask as the ffs_hrfopt run."
        )
    # 1-based (hrfopt convention) → 0-based; unset voxels (0) become shape 0.
    out = np.clip(idx - 1, 0, None)
    too_big = out >= n_shapes
    if too_big.any():
        raise ValueError(
            f"-hrf-index {path!r} contains index "
            f"{int(idx[too_big].max())} (1-based) but the shape library has "
            f"only {n_shapes} curves.  The map was fit against a different "
            "library — pass it via -shift-shapes."
        )
    return out


def _run_shift_mode(
    *,
    args,
    data: torch.Tensor,
    run_starts_ext: list[int],
    n_tp_per_run: list[int],
    all_onsets,
    condition_labels: list[str],
    condition_durations: list[float],
    tr: float,
    polort: int,
    volume_shape,
    mask,
    device: torch.device,
    nii_ext: str,
    extra_regs_per_run: list[torch.Tensor] | None = None,
    parcel_of_voxel: np.ndarray | None = None,
    parcel_labels: list[int] | None = None,
) -> int:
    """-parametrization shift: per-block amplitude + bounded latency.

    Fits on the concatenated timeseries with block-diagonal per-run drift,
    then optionally runs the LORO held-out validator.  Emits amplitude and
    delay maps rather than basis-coefficient maps — there are no basis
    coefficients in this model.
    """
    from fastfuncstuff.design.shifted_hrf import (
        append_blockdiag_extras,
        build_blockdiag_polys,
        build_shape_library,
        fit_shifted_hrf,
        fit_shifted_hrf_per_voxel_shape,
        shape_time_to_peak,
        xval_shifted_hrf,
    )

    n_runs = len(n_tp_per_run)

    # Loudly disown the flags this parametrisation does not read.  -model and
    # -reg select a basis and a prior on basis coefficients; the shift model
    # has neither, so passing them silently did nothing — a command reading
    # "-model FLOBS -parametrization shift" would quietly fit the SPM
    # canonical.  Shape comes from -shift-hrf and nowhere else.
    ignored = [f for f in ("-model", "-reg") if any(a.startswith(f) for a in sys.argv[1:])]
    if ignored:
        print(
            f"\n  WARNING: {', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} "
            f"IGNORED with -parametrization shift.\n"
            f"           This model has no basis coefficients, so there is no "
            f"basis to choose\n"
            f"           and no prior on coefficients to apply.  The response "
            f"shape comes\n"
            f"           from -hrf (currently {args.hrf!r}); the delay "
            f"prior from\n"
            f"           -delay-prior-sd.  Use -parametrization linear if you "
            f"wanted -derivatives {args.derivatives} / -reg {args.reg}."
        )

    print("\n  Parametrisation: shift (amplitude + bounded latency per block)")
    shapes = None
    shape_labels: list[str] = []
    imported_index: np.ndarray | None = None
    # An imported index map only says WHICH curve each voxel took; the curves
    # themselves still have to be rebuilt here, and the default library is
    # what ffs_hrfopt uses unless the user pointed it elsewhere.
    shape_source = _hrf_set_name(args.hrf) or ("library" if args.hrf_index else None)
    if shape_source:
        shapes, shape_labels = build_shape_library(
            shape_source,
            args.flobs_dt,
            args.flobs_window,
            n_hrfs=args.hrf_n_shapes,
            drop_empty=args.hrf_index is None,
        )
        hrf = shapes[0]
        if args.hrf_index:
            try:
                imported_index = _load_shape_index_map(
                    args.hrf_index,
                    n_shapes=shapes.shape[0],
                    volume_shape=volume_shape,
                    mask=mask,
                    n_voxels=data.shape[0],
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            n_used = int(np.unique(imported_index).size)
            print(
                f"  Shape source: {shape_source} — {shapes.shape[0]} curves, "
                f"assignment IMPORTED from {args.hrf_index} "
                f"({n_used} distinct shape(s) in the mask)"
            )
            print(
                "  NOTE: the imported indices must come from THIS library.  A\n"
                "        different library (or a different -hrf_mode) makes each\n"
                "        index point at an unrelated curve, and nothing here can\n"
                "        detect that — only the range is checked."
            )
        else:
            print(
                f"  Shape source: {shape_source} — {shapes.shape[0]} candidates, "
                f"selected PER VOXEL at zero delay (overrides -shift-hrf)"
            )
            print(
                "  NOTE: a shape library that varies PEAK TIME competes with the\n"
                "        delay parameter — both move the response in time.  Measured\n"
                "        on synthetic data with a true ±1.2 s delay, shape selection\n"
                "        absorbed all of it (corr(shape, true delay)=0.99) and the\n"
                "        held-out delay gain fell to zero.  With shapes on, read the\n"
                "        delay map as RESIDUAL timing and shape_index as the main\n"
                "        carrier of voxel-level timing.  For one clean absolute delay\n"
                "        map, use a single -shift-hrf instead."
            )
    else:
        hrf = _resolve_shift_hrf(args.hrf, args.flobs_dt, args.flobs_window)
        print(
            f"  HRF source: {args.hrf}  ({hrf.size} samples @ "
            f"{args.flobs_dt}s, peak-normalised)  [one shape for all voxels]"
        )
    print(
        f"  Delay search: ±{args.tau_max}s step {args.tau_step}s"
        f"{f', prior sd {args.delay_prior_sd}s' if args.delay_prior_sd else ', no prior'}"
        f", amp ridge {args.amp_ridge if isinstance(args.amp_ridge, str) else f'{args.amp_ridge:g}'}"
    )

    per_run_data = [
        data[:, run_starts_ext[r] : run_starts_ext[r + 1]].clone().detach() for r in range(n_runs)
    ]
    # onsets in CONCATENATED time; blocks are trials or conditions
    offsets = np.cumsum([0] + n_tp_per_run[:-1]) * tr
    block_onsets: list[np.ndarray] = []
    block_labels: list[str] = []
    block_cond: list[int] = []
    for c, label in enumerate(condition_labels):
        if args.single_trials:
            k = 0
            for r in range(n_runs):
                for t in np.atleast_1d(all_onsets[c][r]):
                    block_onsets.append(np.array([float(t) + offsets[r]]))
                    block_labels.append(f"{label}_trial{k:03d}_run{r + 1}")
                    block_cond.append(c)
                    k += 1
        else:
            merged = [np.atleast_1d(all_onsets[c][r]) + offsets[r] for r in range(n_runs)]
            block_onsets.append(np.concatenate(merged) if merged else np.array([]))
            block_labels.append(label)
            block_cond.append(c)
    # Condition-level blocks, used ONLY for shape selection: pooling a
    # condition's trials into one regressor spends 1 amplitude DOF instead
    # of n_trials, so the shape comparison is far better determined.
    cond_block_onsets: list[np.ndarray] = []
    for c in range(len(condition_labels)):
        merged = [np.atleast_1d(all_onsets[c][r]) + offsets[r] for r in range(n_runs)]
        cond_block_onsets.append(np.concatenate(merged) if merged else np.array([]))
    print(f"  Blocks: {len(block_onsets)}")
    # Shift used to ignore -durations entirely: it modelled every event as an
    # impulse and let the delay search absorb the mismatch, so a 4 s block
    # read as ~0.8 s of spurious latency.  Same bug the linear path had; the
    # design-bank builder already took durations, the CLI never passed them.
    shift_block_durations = [condition_durations[c] for c in block_cond]

    # Sample-index run boundaries: keeps each block's response inside its own
    # run (an event in the last ~32 s of a run would otherwise spill its tail
    # into the next run's samples) and makes cross-run gram pairs exactly zero.
    _b = np.cumsum([0] + n_tp_per_run)
    run_bounds = [(int(_b[r]), int(_b[r + 1])) for r in range(n_runs)]

    Z = build_blockdiag_polys(n_tp_per_run, polort, device)
    if extra_regs_per_run is not None:
        Z = append_blockdiag_extras(Z, extra_regs_per_run, n_tp_per_run, device)
    shape_index = None
    if shapes is not None:
        fit, shape_index = fit_shifted_hrf_per_voxel_shape(
            durations=shift_block_durations,
            data=data,
            block_onsets=block_onsets,
            selection_block_onsets=cond_block_onsets,
            shapes=shapes,
            shape_index=imported_index,
            hrf_dt=args.flobs_dt,
            tr=tr,
            nuisance=Z,
            tau_max=args.tau_max,
            tau_step=args.tau_step,
            run_bounds=run_bounds,
            n_sweeps=args.shift_sweeps,
            delay_prior_sd=args.delay_prior_sd,
            amp_ridge=args.amp_ridge,
            device=device,
            verbose=True,
        )
    else:
        fit = fit_shifted_hrf(
            durations=shift_block_durations,
            data=data,
            block_onsets=block_onsets,
            hrf=hrf,
            hrf_dt=args.flobs_dt,
            tr=tr,
            nuisance=Z,
            tau_max=args.tau_max,
            tau_step=args.tau_step,
            run_bounds=run_bounds,
            n_sweeps=args.shift_sweeps,
            delay_prior_sd=args.delay_prior_sd,
            amp_ridge=args.amp_ridge,
            device=device,
            verbose=True,
        )
    del Z
    if device.type == "cuda":
        torch.cuda.empty_cache()

    nx, ny, nz = volume_shape

    def _to_volume(masked: np.ndarray) -> np.ndarray:
        # With -atlas the rows are parcels, not voxels: paint each parcel's
        # value into every voxel that belongs to it.  Voxels the atlas did not
        # label stay zero.
        if parcel_of_voxel is not None:
            ok = parcel_of_voxel >= 0
            painted = np.zeros((parcel_of_voxel.size,) + tuple(masked.shape[1:]), dtype=np.float32)
            painted[ok] = masked[parcel_of_voxel[ok]]
            masked = painted
        out_shape = (nx, ny, nz) + tuple(masked.shape[1:])
        out = np.zeros(out_shape, dtype=np.float32)
        if mask is not None:
            out[mask, ...] = masked
        else:
            out = masked.reshape(out_shape)
        return out

    if parcel_of_voxel is not None and parcel_labels:
        # The parcel-level numbers are the actual estimates; the painted maps
        # are a convenience view of them.  Ship them as a table too.
        tsv = f"{args.prefix}_fitbasis_shift_parcels.tsv"
        hdr = ["label", "n_voxels", "r2", "r2_tau0", "fstat", "amp_lambda", "mean_delay"]
        rows = []
        for i, lab in enumerate(parcel_labels):
            rows.append(
                [
                    lab,
                    int((parcel_of_voxel == i).sum()),
                    float(fit.r2[i]),
                    float(fit.r2_fixed[i]),
                    float(fit.fstat[i]),
                    float(fit.amp_lambda[i]),
                    float(fit.delays[i].mean()),
                ]
            )
        with open(tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write(
                    "\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n"
                )
        print(f"  Wrote {tsv}  ({len(rows)} parcels)")

    # ── Held-out validation first, so it can ride inside the buckets ──
    xval_maps: list[tuple[str, np.ndarray]] = []
    if args.xval_r2:
        if n_runs < 2:
            print("  -xval-r2 needs ≥2 runs; skipping.")
        else:
            print("\n  Held-out validation (LORO, condition-level generalisation)…")
            if imported_index is not None:
                print(
                    "  NOTE: the imported shape assignment is held FIXED across\n"
                    "        folds (it is an input to this model, not something\n"
                    "        fitted here).  If it was selected on these same runs,\n"
                    "        xvalR2 is optimistic by that much; xvalR2_delay_gain\n"
                    "        is not, since both scored models carry the same shape\n"
                    "        and only the delays are refit per fold."
                )
            r2_shift, r2_tau0 = xval_shifted_hrf(
                durations=list(condition_durations),
                per_run_data=per_run_data,
                per_run_condition_onsets=[
                    [np.atleast_1d(all_onsets[c][r]) for c in range(len(condition_labels))]
                    for r in range(n_runs)
                ],
                hrf=hrf,
                hrf_dt=args.flobs_dt,
                tr=tr,
                polort=polort,
                single_trials=args.single_trials,
                shapes=shapes,
                shape_index=imported_index,
                extra_regs_per_run=extra_regs_per_run,
                tau_max=args.tau_max,
                tau_step=args.tau_step,
                delay_prior_sd=args.delay_prior_sd,
                amp_ridge=args.amp_ridge,
                n_sweeps=args.shift_sweeps,
                leave_n_out=args.cv_leave_n_out,
                device=device,
                verbose=True,
            )
            xval_maps = [
                ("xvalR2", r2_shift),
                ("xvalR2_tau0", r2_tau0),
                ("xvalR2_delay_gain", r2_shift - r2_tau0),
            ]

    # Same QC contract as the linear path: the honest referee first, the
    # in-sample task R2 next, in every bucket the user might threshold.
    qc_bricks: list[tuple[str, np.ndarray]] = []
    if xval_maps:
        qc_bricks.append(("xvalR2", np.asarray(xval_maps[0][1], dtype=np.float32)))
    qc_bricks.append(("taskR2", np.asarray(fit.r2, dtype=np.float32)))

    if xval_maps:
        _save_bucket(
            [np.asarray(a, dtype=np.float32) for _, a in xval_maps],
            [n for n, _ in xval_maps],
            f"{args.prefix}_fitbasis_xvalr2{nii_ext}",
            to_volume=_to_volume,
            reference_img=args.input[0],
        )
        print(
            "  xvalR2_delay_gain is the map that answers 'is the delay real':\n"
            "  positive = the estimated delays predicted held-out runs better\n"
            "  than pinning them to zero; ≤0 = they did not."
        )

    # Everything that is a diagnostic rather than a result, in one place.
    _save_bucket(
        [fit.r2, fit.r2_fixed, fit.r2_total, fit.fstat, fit.amp_lambda],
        ["taskR2", "taskR2_tau0", "taskR2_incl_drift", "fstat", "amp_lambda"],
        f"{args.prefix}_fitbasis_diagnostics{nii_ext}",
        to_volume=_to_volume,
        reference_img=args.input[0],
    )
    print(
        "  NOTE: taskR2 is TASK variance / NON-DRIFT variance — the nuisance is\n"
        "        removed from both terms, so drift is not credited to the model.\n"
        "        taskR2_incl_drift uses raw total variance (what most tools print)\n"
        "        and is inflated by the polynomial model.  Neither is evidence of\n"
        "        task response on its own — n_blocks free amplitudes buy in-sample\n"
        "        fit too.  Threshold on xvalR2; for latency, on xvalR2_delay_gain."
    )

    if shape_index is not None and shapes is not None:
        # The selected shape absorbs part of the voxel's mean timing, so the
        # delay map alone understates it.  Emit the pieces that make the
        # confound decomposable rather than destructive: time-to-peak of the
        # chosen curve, and TTP + mean delay, which recovers the true mean
        # timing (verified to sum exactly at good SNR).
        ttp = shape_time_to_peak(shapes, args.flobs_dt)[shape_index]
        _save_bucket(
            [shape_index.astype(np.float32) + 1.0, ttp, ttp + fit.delays.mean(axis=1)],
            ["hrf_index", "shape_ttp", "mean_timing"],
            f"{args.prefix}_fitbasis_hrf_index{nii_ext}",
            to_volume=_to_volume,
            reference_img=args.input[0],
        )
        curves_path = f"{args.prefix}_fitbasis_hrf_shapes.tsv"
        np.savetxt(
            curves_path,
            shapes.T,
            fmt="%.10g",
            delimiter="\t",
            header="\t".join(shape_labels),
            comments="",
        )
        print(f"  Wrote {curves_path}  (columns = candidates @ {args.flobs_dt}s)")

    cond_idx: dict[str, list[int]] = {}
    for b, c in enumerate(block_cond):
        cond_idx.setdefault(condition_labels[c], []).append(b)
    # Per-block deviation from each voxel's OWN mean delay.  This is the
    # quantity that survives the shape/delay confound: the shape is fixed
    # within a voxel, so it cannot absorb block-to-block variation.
    delay_dev = fit.delays - fit.delays.mean(axis=1, keepdims=True)
    # A delay sitting on the search rail was not estimated, it was clamped —
    # the same thing `valid` means on the linear side.
    delay_valid = (np.abs(fit.delays) < args.tau_max - 1e-9).astype(np.float32)

    if all(len(idxs) == 1 for idxs in cond_idx.values()):
        # Per-condition: one bucket each, named and ordered to match the
        # linear parametrisation so the two are directly comparable.
        _save_bucket(
            [fit.amplitudes[:, idxs[0]] for idxs in cond_idx.values()],
            list(cond_idx.keys()),
            f"{args.prefix}_fitbasis_amplitude{nii_ext}",
            to_volume=_to_volume,
            reference_img=args.input[0],
            qc=qc_bricks,
        )
        arrays, labels = [], []
        for cond, idxs in cond_idx.items():
            b = idxs[0]
            for arr, name in (
                (fit.delays[:, b], "latency"),
                (delay_dev[:, b], "latency_dev"),
                (delay_valid[:, b], "valid"),
            ):
                arrays.append(np.asarray(arr, dtype=np.float32))
                labels.append(f"{cond}_{name}")
        _save_bucket(
            arrays,
            labels,
            f"{args.prefix}_fitbasis_shape{nii_ext}",
            to_volume=_to_volume,
            reference_img=args.input[0],
            qc=qc_bricks,
        )
    else:
        # Single-trial: the sub-brick axis is TRIALS, so conditions cannot
        # share a bucket without colliding with it.
        for cond, idxs in cond_idx.items():
            for arr, name in (
                (fit.amplitudes, "amplitude"),
                (fit.delays, "latency"),
                (delay_dev, "latency_dev"),
                (delay_valid, "valid"),
            ):
                path = f"{args.prefix}_fitbasis_{name}_{cond}{nii_ext}"
                save_nifti(
                    _to_volume(arr[:, idxs]),
                    output_path=path,
                    reference_img=args.input[0],
                    brick_labels=[f"{cond}_{name}_{t:03d}" for t in range(len(idxs))],
                )
                print(f"  Wrote {path}  (n_trials={len(idxs)})")

    meta = {
        "tool": "ffs_fitbasis",
        "parametrization": "shift",
        "started": datetime.now().isoformat(timespec="seconds"),
        "tr": float(tr),
        "hrf_source": (args.hrf if shapes is None else f"per-voxel:{shape_source}"),
        "n_shape_candidates": (0 if shapes is None else int(shapes.shape[0])),
        "shape_index_source": (
            "imported" if imported_index is not None else ("fitted" if shapes is not None else None)
        ),
        "shape_index_map": args.hrf_index,
        "ortvec_columns_per_run": (
            0 if extra_regs_per_run is None else int(extra_regs_per_run[0].shape[1])
        ),
        "tau_max": float(args.tau_max),
        "tau_step": float(args.tau_step),
        "delay_prior_sd": (None if args.delay_prior_sd is None else float(args.delay_prior_sd)),
        "amp_ridge": (args.amp_ridge if isinstance(args.amp_ridge, str) else float(args.amp_ridge)),
        "fstat_df": list(fit.fstat_df),
        "n_sweeps": int(fit.n_sweeps),
        "single_trials": bool(args.single_trials),
        "n_blocks": len(block_onsets),
        "condition_labels": list(condition_labels),
        "polort": int(polort),
        "blur_fwhm": args.do_blur,
        "r2_median_task_relative": float(np.median(fit.r2)),
        "r2_tau0_median": float(np.median(fit.r2_fixed)),
        "r2_median_incl_drift": float(np.median(fit.r2_total)),
    }
    meta_path = f"{args.prefix}_fitbasis_metadata.json"
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    print(f"  Wrote {meta_path}")
    print(f"\n{'=' * 72}")
    print(" ✓ ffs_fitbasis complete (shift parametrisation)")
    print(f"{'=' * 72}")
    return 0


def main() -> int:
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()

    # Legacy -model -> (-flobs, -derivatives).  Explicit new flags win, and
    # saying both contradictory things is an error rather than a coin toss.
    if args.model is not None:
        mapped = {"SPMG1": "none", "SPMG2": "time", "SPMG3": "time+width"}
        given = {a.split("=")[0] for a in sys.argv[1:]}
        if args.model == "FLOBS":
            args.use_flobs = True
        elif given & {"-derivatives", "-deriv"} and args.derivatives != mapped[args.model]:
            parser.error(
                f"-model {args.model} means -derivatives {mapped[args.model]}, "
                f"but -derivatives {args.derivatives} was also given."
            )
        else:
            args.derivatives = mapped[args.model]
        print(
            f"  NOTE: -model {args.model} is superseded by "
            + ("-flobs" if args.model == "FLOBS" else f"-derivatives {args.derivatives}")
        )

    n_deriv = {"none": 0, "time": 1, "time+width": 2}[args.derivatives]

    if getattr(args, "delay_prior_sd", None) is not None and args.delay_prior_sd <= 0:
        args.delay_prior_sd = None

    if args.save_shape:
        # Each of these makes the readout meaningless rather than merely
        # worse, so they are errors, not warnings.
        if args.parametrization != "linear":
            parser.error(
                "-save-shape needs -parametrization linear; the shift "
                "parametrisation already reports delay in seconds directly."
            )
        if args.use_flobs or n_deriv < 1:
            parser.error(
                "-save-shape needs -derivatives time or time+width"
                + (" (-flobs has no derivative columns)" if args.use_flobs else "")
                + "; there is no derivative coefficient to read latency from."
            )
        if args.single_trials:
            parser.error(
                "-save-shape is a per-condition readout.  Per trial, beta_c is "
                "not well enough determined for the ratio (measured: it runs to "
                "±5 s in ~45% of trials) — use -parametrization shift instead."
            )

    if isinstance(getattr(args, "amp_ridge", None), str):
        if args.amp_ridge.strip().lower() == "auto":
            args.amp_ridge = "auto"
        else:
            try:
                args.amp_ridge = float(args.amp_ridge)
            except ValueError:
                print(
                    f"ERROR: -amp-ridge must be 'auto' or a number; got {args.amp_ridge!r}",
                    file=sys.stderr,
                )
                return 1

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    print("=" * 72)
    print(" ffs_fitbasis — per-condition response estimation")
    print("=" * 72)
    print(f"  Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Prefix:  {args.prefix}")

    # ── Validate event input ────────────────────────────────────────
    has_onsets = bool(args.onsets)
    has_events = bool(args.events)
    if has_onsets == has_events:
        print("ERROR: Specify exactly one of -onsets/-durations or -events.")
        return 1
    if has_onsets and args.durations is None:
        print("ERROR: -durations is required with -onsets.")
        return 1
    if args.event_cols and not has_events:
        print("ERROR: -event-cols requires -events.")
        return 1
    if args.event_ignore and not has_events:
        print("ERROR: -event-ignore requires -events.")
        return 1

    # ── Parse inputs / events ───────────────────────────────────────
    input_files = parse_input_files(args.input)
    # parse_input_files expands glob patterns (?, *, [..]) — overwrite
    # args.input with the expanded list so downstream references
    # (reference_img=args.input[0] for save_nifti) see real file paths,
    # not the unexpanded pattern.  When the user's shell already
    # expanded the glob, this is a no-op.
    args.input = input_files
    n_runs = len(input_files)
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
        return 1

    all_onsets = timing.all_onsets
    condition_labels = timing.condition_labels
    n_conditions = timing.n_conditions

    preflight_check(
        input_files=input_files, onset_files=args.onsets if has_onsets else None, ortvec_files=None
    )

    # ── Load data ──────────────────────────────────────────────────
    device = setup_device(args.device)
    print(f"  Compute device: {device}")

    load_result = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=True,
        device=device,
        force_cpu=True,
        dry_run=False,
        verbose=True,
        load_threads=args.load_threads,
        drop_first=args.drop_first,
        drop_last=args.drop_last,
    )
    data = load_result.data
    run_starts = load_result.run_starts
    volume_shape = load_result.volume_shape
    tr = load_result.tr
    mask = load_result.mask
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    if args.tr is None:
        args.tr = tr

    # Shift timing onto the retained window before rounding touches the onsets.
    trim = trim_spec_from_args(args, tr=tr)
    apply_trim_to_timing(
        timing,
        trim,
        run_lengths_tr=run_lengths_from_starts(list(run_starts), n_timepoints),
        n_runs=n_runs,
    )
    all_onsets = timing.all_onsets

    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, tr, threshold=args.round_onsets)
        print(f"  Rounded onsets to nearest TR (threshold={args.round_onsets:.2f}).")

    print(f"  Data: {n_voxels:,} voxels × {n_timepoints} TR ({n_runs} runs, TR={tr}s)")

    # ── Optional parcellation ──────────────────────────────────────
    # Collapse to one timeseries per atlas label BEFORE fitting.  Everything
    # downstream then works on parcels; ``parcel_of_voxel`` paints the results
    # back so the output maps keep their usual voxel geometry.
    parcel_of_voxel: np.ndarray | None = None
    parcel_labels: list[int] = []
    if args.atlas:
        if args.parametrization != "shift":
            print(
                "ERROR: -atlas is currently implemented for -parametrization shift only.",
                file=sys.stderr,
            )
            return 1
        from fastfuncstuff.io.afni import load_nifti

        atlas_img = load_nifti(args.atlas)
        atlas = np.asarray(atlas_img.dataobj)
        if atlas.shape[:3] != tuple(volume_shape):
            print(
                f"ERROR: -atlas shape {atlas.shape[:3]} does not match the "
                f"input grid {tuple(volume_shape)}.  Resample it first "
                f"(ffs_util_resample).",
                file=sys.stderr,
            )
            return 1
        lab_v = (atlas[mask] if mask is not None else atlas.reshape(-1)).astype(np.int64)
        parcel_labels = [int(x) for x in np.unique(lab_v) if x > 0]
        if not parcel_labels:
            print("ERROR: -atlas contains no positive labels inside the mask.", file=sys.stderr)
            return 1
        lut = {lab: i for i, lab in enumerate(parcel_labels)}
        parcel_of_voxel = np.full(lab_v.shape, -1, dtype=np.int64)
        for lab, i in lut.items():
            parcel_of_voxel[lab_v == lab] = i
        pdata = torch.zeros((len(parcel_labels), data.shape[1]), dtype=data.dtype)
        counts = []
        for i in range(len(parcel_labels)):
            sel = parcel_of_voxel == i
            counts.append(int(sel.sum()))
            pdata[i] = data[torch.from_numpy(np.flatnonzero(sel))].mean(dim=0)
        data = pdata
        print(
            f"  Atlas: {len(parcel_labels)} parcel(s) from {args.atlas} — "
            f"{int((parcel_of_voxel >= 0).sum()):,}/{n_voxels:,} masked voxels assigned, "
            f"median parcel {int(np.median(counts))} voxels"
        )
        print(
            "  Fitting one timeseries PER PARCEL; maps are painted back into "
            "every voxel of their parcel."
        )
        n_voxels = data.shape[0]

    # ── Build basis + prior ────────────────────────────────────────
    # Resolve -lambda-mode auto: voxelwise for single-trial fits
    # (low per-trial DOF + high σ² variation across the brain at
    # 9.4T-style SNR means a global λ over-shrinks high-SNR voxels);
    # global otherwise (the per-condition fit has enough DOF that a
    # single shared λ is fine and faster).
    # -reg none disables the prior entirely; the lambda-mode setting
    # is meaningless then (no λ to set per voxel).  Resolve only when
    # the prior is actually in play.
    if args.lambda_mode == "auto":
        if args.reg == "none":
            args.lambda_mode = "global"  # placeholder; unused downstream
        else:
            args.lambda_mode = "voxelwise" if args.single_trials else "global"
            print(
                f"  Lambda mode auto → {args.lambda_mode} "
                f"({'single-trials' if args.single_trials else 'per-condition'})"
            )

    _shift_mode = args.parametrization == "shift"
    if not _shift_mode:
        print(
            f"\n  Response shape: {_shape_summary(args)}"
            f"\n  Regularisation: {args.reg}    Single-trials: {args.single_trials}"
        )
    basis = _build_basis(args)
    n_basis = basis.basis_functions.shape[0]
    prior_m, prior_C = _build_prior(
        model="FLOBS" if args.use_flobs else f"SPMG{n_basis}",
        reg=args.reg,
        basis=basis,
        canonical_std=args.canonical_std,
        derivative_std=args.derivative_std,
        dispersion_std=args.dispersion_std,
    )
    pw = _resolve_prior_weight_arg(args.prior_weight, args.reg)
    if not _shift_mode:
        print(
            f"  Columns per condition: {n_basis} "
            f"(curves at dt={basis.dt}s over {basis.duration:.1f}s)"
        )
    # Only print prior info when the prior is actually applied.  With
    # -reg none, m/C/λ are all zeros / unused — printing them is noise
    # that confuses users about whether the prior is active.
    if _shift_mode:
        # -reg / -model / the basis play no part in the shift model; printing
        # them here previously implied a prior was active when none was.
        pass
    elif args.reg != "none":
        print(f"  Prior: m = {prior_m},  σ_diag = {np.sqrt(np.diag(prior_C))}")
        print(f"  Prior weight: {pw!r}")
    else:
        print("  Prior: disabled (-reg none)")

    # ── Polort resolution ─────────────────────────────────────────
    if args.polort is None:
        run_dur_min = (n_timepoints / n_runs) * tr / 60.0
        polort_resolved = max(0, round(run_dur_min / 2))
        print(f"  Polort auto: {polort_resolved} (run ≈ {run_dur_min:.1f} min)")
    else:
        polort_resolved = int(args.polort)

    # ── Build per-run task design (one block per CONDITION OR per TRIAL) ─
    # build_pc_basis_design_per_run handles convolving onsets with each
    # basis function for FIR-locked or TENT-style onsets.  For
    # single-trial mode we treat each trial as its own "condition" so
    # the same machinery produces one K-column block per trial.
    run_starts_ext = list(run_starts) + [n_timepoints]
    n_tp_per_run = [run_starts_ext[r + 1] - run_starts_ext[r] for r in range(n_runs)]
    basis_lag_times = np.arange(basis.basis_functions.shape[1]) * basis.dt

    # ── External nuisance (-ortvec family) ─────────────────────────
    # Kept per-run on the block diagonal, exactly like the polynomials:
    # denoising components estimated per run do not describe the other
    # runs, and sharing them would let one run's noise soak up another's
    # signal ([[Block-diagonal nuisance]]).
    try:
        nuisance_blocks = collect_nuisance_blocks(
            args,
            run_starts=list(run_starts),
            n_timepoints=n_timepoints,
            verbose=True,
            trim=trim,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    extra_regs_per_run: list[torch.Tensor] | None = None
    if nuisance_blocks:
        extra_regs_per_run = append_nuisance_blocks(
            [torch.zeros((n, 0), dtype=torch.float32) for n in n_tp_per_run],
            nuisance_blocks,
            list(run_starts),
            n_timepoints,
        )
        n_extra = extra_regs_per_run[0].shape[1]
        print(
            f"  External nuisance: {n_extra} column(s) per run "
            f"({n_extra * n_runs} total, block-diagonal)"
        )

    # Auto-detect TR-lock vs sub-TR for the basis convolution path.
    all_onset_times = [
        float(t)
        for cond_runs in all_onsets
        for run_onsets in cond_runs
        for t in (run_onsets.tolist() if run_onsets.size else [])
    ]
    from fastfuncstuff.design.matrices import is_tr_locked

    basis_mode = (
        "FIR" if (all_onset_times and is_tr_locked(all_onset_times, tr, threshold=0.1)) else "TENT"
    )
    print(
        f"  Onset basis-convolution mode: {basis_mode} "
        f"({'TR-locked' if basis_mode == 'FIR' else 'sub-TR onsets'})"
    )

    # Build "block list" — each block is one condition (default) or one
    # trial (single-trials mode).  For single-trial mode, expand each
    # condition's onset list to a separate one-event-per-block.
    block_labels: list[str] = []
    block_onsets_per_run: list[list[np.ndarray]] = []
    block_cond_idx: list[int] = []
    if args.single_trials:
        for cond_idx, cond_label in enumerate(condition_labels):
            cond_runs = all_onsets[cond_idx]
            # number trials in deterministic order: per cond, per run
            trial_num_global = 0
            for r, run_arr in enumerate(cond_runs):
                for t_in_run in run_arr:
                    # build one-event onset list per run for this trial
                    per_run = [
                        np.array([t_in_run]) if rr == r else np.array([]) for rr in range(n_runs)
                    ]
                    block_labels.append(f"{cond_label}_trial{trial_num_global:03d}_run{r + 1}")
                    block_onsets_per_run.append(per_run)
                    block_cond_idx.append(cond_idx)
                    trial_num_global += 1
    else:
        for cond_idx, cond_label in enumerate(condition_labels):
            block_labels.append(cond_label)
            block_onsets_per_run.append(all_onsets[cond_idx])
            block_cond_idx.append(cond_idx)

    n_blocks = len(block_labels)

    # ── Stimulus durations into the basis ──────────────────────────
    # Previously the linear path built impulse regressors and ignored
    # -durations entirely, which delays the modelled peak by ~D/2 and
    # mis-scales the amplitude.  Durations are per condition (the
    # repo's timing model); a block inherits its condition's.
    block_durations = [
        float(timing.durations[c]) if timing.durations_given else 0.0 for c in block_cond_idx
    ]
    block_bases = _build_block_bases(basis, block_durations)

    uniq_dur = sorted({round(d, 4) for d in block_durations})
    if any(d > basis.dt for d in uniq_dur):
        shown = ", ".join(f"{d:g}s" for d in uniq_dur)
        print(f"  Stimulus duration convolved into the basis: {shown}")
    else:
        print("  Stimulus duration: impulse (0s) — regressors are the bare HRF.")
    print(
        f"  Blocks to fit: {n_blocks}  ({'one per trial' if args.single_trials else 'one per condition'})"
    )

    # ── Dispatch: shifted-HRF parametrisation ──────────────────────
    # This model has no basis-coefficient vector at all (one column per
    # block, amplitude + bounded delay), so it bypasses the whole
    # basis / prior / packed-design pipeline below rather than
    # threading a special case through it.
    if args.parametrization == "shift":
        return _run_shift_mode(
            args=args,
            data=data,
            run_starts_ext=run_starts_ext,
            n_tp_per_run=n_tp_per_run,
            all_onsets=all_onsets,
            condition_labels=list(condition_labels),
            condition_durations=[
                float(timing.durations[c]) if timing.durations_given else 0.0
                for c in range(len(condition_labels))
            ],
            tr=tr,
            polort=polort_resolved,
            volume_shape=volume_shape,
            mask=mask,
            device=device,
            nii_ext=nii_ext,
            extra_regs_per_run=extra_regs_per_run,
            parcel_of_voxel=parcel_of_voxel,
            parcel_labels=parcel_labels,
        )

    # ── Per-voxel HRF shape (shared by both parametrisations) ──────
    hrf_shapes: np.ndarray | None = None
    hrf_shape_labels: list[str] = []
    hrf_index: np.ndarray | None = None
    hrf_index_score: np.ndarray | None = None
    shape_source = _hrf_set_name(args.hrf) or ("library" if args.hrf_index else None)
    if shape_source:
        from fastfuncstuff.design.shifted_hrf import build_shape_library

        hrf_shapes, hrf_shape_labels = build_shape_library(
            shape_source,
            args.flobs_dt,
            args.flobs_window,
            n_hrfs=args.hrf_n_shapes,
            # An imported index numbers rows of the SOURCE library, so
            # dropping a degenerate curve here would renumber everything
            # after it.
            drop_empty=args.hrf_index is None,
        )
        print(f"  HRF shape candidates: {hrf_shapes.shape[0]} ({shape_source})")

    # Build per-run design with K basis cols per block
    per_run_designs: list[torch.Tensor] = []
    for r in range(n_runs):
        block_designs = []
        for b_idx in range(n_blocks):
            # Use the helper's onsets-per-run path: one-condition view.
            bd = build_pc_basis_design_per_run(
                onsets_per_run=[block_onsets_per_run[b_idx][r]],
                pcs=block_bases[b_idx],
                lag_times=basis_lag_times,
                tr=tr,
                n_timepoints_per_run=[n_tp_per_run[r]],
                basis=basis_mode,
            )
            block_designs.append(bd[0])
        concat = np.concatenate(block_designs, axis=1).astype(np.float32)
        per_run_designs.append(torch.from_numpy(concat))

    # ── Per-run data list ──────────────────────────────────────────
    # ``load_and_preprocess_runs`` already applies the brain mask when
    # one is supplied — its returned ``data`` tensor has shape
    # ``(n_voxels_in_mask, total_tp)``.  Just split by run boundary.
    per_run_data = [
        data[:, run_starts_ext[r] : run_starts_ext[r + 1]].clone().detach().float()
        for r in range(n_runs)
    ]

    # Keep an un-modified copy of per_run_data for xval R² (the
    # -prior-from shift and -prewhiten arma11 both mutate
    # per_run_data in place; xval needs the original).  Cheap:
    # one extra copy of (n_vox × total_tp) float32.
    per_run_data_orig: list[torch.Tensor] | None = (
        [d.clone() for d in per_run_data] if args.xval_r2 else None
    )

    # ── Empirical-Bayes per-voxel prior from per-condition pre-fit ─
    # When -prior-from per-condition and -single-trials: run a plain
    # per-condition constrained fit first, extract (n_vox, n_cond, K)
    # betas, and use them as the prior mean for *every trial of that
    # condition at that voxel* in the single-trial fit below.  Then
    # shift the data by X·m per voxel so the rest of the pipeline
    # solves for β_centered = β_trial − β_cond.  We restore the full
    # β_trial = β_centered + β_cond before output.
    #
    # Algebra check: y = X β + ε.  With y' = y − X m_v, β' = β − m_v:
    # y' = X β' + ε.  Residuals match; only the mean of β shifts.
    # The shape prior is applied to β_centered with prior_mean=0,
    # so the quadratic ‖β − m_v‖²_{C⁻¹} that the prior penalises
    # equals ‖β_centered‖²_{C⁻¹} — exactly the right thing.
    empirical_prior_mean_full: np.ndarray | None = None
    if args.prior_from == "per-condition":
        if not args.single_trials:
            print(
                "  WARNING: -prior-from per-condition has no effect "
                "without -single-trials; ignoring."
            )
        elif args.reg in {"mvn-shape", "fracridge"}:
            print(
                f"ERROR: -prior-from per-condition is not yet compatible "
                f"with -reg {args.reg} (rotation / CV-loop issues).  "
                f"Use -reg in {{none, ridge, mvn}}."
            )
            return 1
        else:
            print(
                f"\n  Empirical Bayes: per-condition pre-fit "
                f"({len(condition_labels)} cond × K={n_basis} on "
                f"-derivatives {args.derivatives} / -reg {args.reg})…"
            )
            cond_bases = _build_block_bases(
                basis,
                [
                    float(timing.durations[c]) if timing.durations_given else 0.0
                    for c in range(len(condition_labels))
                ],
            )
            pc_designs: list[torch.Tensor] = []
            for r in range(n_runs):
                cond_blocks = []
                for c in range(len(condition_labels)):
                    bd = build_pc_basis_design_per_run(
                        onsets_per_run=[all_onsets[c][r]],
                        pcs=cond_bases[c],
                        lag_times=basis_lag_times,
                        tr=tr,
                        n_timepoints_per_run=[n_tp_per_run[r]],
                        basis=basis_mode,
                    )
                    cond_blocks.append(bd[0])
                pc_concat = np.concatenate(cond_blocks, axis=1).astype(np.float32)
                pc_designs.append(torch.from_numpy(pc_concat))

            n_cond = len(condition_labels)
            n_pc_task_cols = n_cond * n_basis
            packed_pc = pack_for_shared_task_glm(
                per_run_data=per_run_data,
                per_run_task_designs=pc_designs,
                polort=polort_resolved,
                task_column_labels=[
                    f"{lbl}#PC{b}" for lbl in condition_labels for b in range(n_basis)
                ],
                extra_regressors_per_run=extra_regs_per_run,
                device=device,
            )
            pc_task_design = packed_pc.design_concat[:, :n_pc_task_cols]
            pc_nuisance = (
                packed_pc.design_concat[:, n_pc_task_cols:]
                if packed_pc.design_concat.shape[1] > n_pc_task_cols
                else None
            )
            # Inherit -lambda-mode and -prior-weight from the user.
            # The previous hard-coded ``global / auto`` combination
            # took σ²_mean across the brain mask, which at 9.4T PSC
            # scaling is dominated by CSF/edge voxels with huge
            # variance — every voxel saw the same enormous λ and the
            # per-condition β collapsed toward `m`, defeating the
            # whole point of -prior-from per-condition (the prior
            # ended up being "basically global m" for every voxel).
            pc_fit = fit_basis_constrained_ridge(
                data=packed_pc.data_concat,
                design_task=pc_task_design,
                basis_functions=basis.basis_functions,
                prior_mean=prior_m,
                prior_cov=prior_C,
                n_blocks=n_cond,
                nuisance=pc_nuisance,
                prior_weight=pw,
                device=device,
                reconstruct_hrfs=False,
                lambda_mode=args.lambda_mode,
                lambda_n_bins=args.lambda_n_bins,
            )
            cond_betas = pc_fit.betas[:, :n_pc_task_cols].reshape(
                -1,
                n_cond,
                n_basis,
            )
            print(
                f"  ✓ Per-condition pre-fit complete.  "
                f"R² mean: OLS={pc_fit.r2_ols.mean():.3f}, "
                f"constrained={pc_fit.r2.mean():.3f}"
            )

            # Map per-condition → per-trial prior mean using block_labels.
            n_vox_masked_pc = cond_betas.shape[0]
            empirical_prior_mean_full = np.zeros(
                (n_vox_masked_pc, n_blocks * n_basis),
                dtype=np.float32,
            )
            cond_to_idx = {c: i for i, c in enumerate(condition_labels)}
            for b_idx, label in enumerate(block_labels):
                cond_label = label.split("_trial")[0]
                ci = cond_to_idx[cond_label]
                empirical_prior_mean_full[:, b_idx * n_basis : (b_idx + 1) * n_basis] = cond_betas[
                    :, ci, :
                ]

            # Shift per_run_data: y → y − X · m_v in place per run.
            m_t = torch.from_numpy(empirical_prior_mean_full).to(
                device=device,
                dtype=torch.float32,
            )  # (n_vox, n_blocks*n_basis)
            for r in range(n_runs):
                X_r = per_run_designs[r].to(device).float()  # (n_tp_r, n_task)
                shift_r = m_t @ X_r.T  # (n_vox, n_tp_r)
                per_run_data[r] = per_run_data[r].to(device).float() - shift_r
            print(
                "  ✓ Data shifted; single-trial fit now solves "
                "β_centered = β_trial − β_cond per voxel."
            )
            del m_t

    # ── Optional ARMA(1,1) prewhitening (VB-loop foundation) ───────
    # When enabled, replace the i.i.d. noise assumption with an
    # ARMA(1,1) covariance, applied per-run via Cholesky.
    #   arma11        — single global (a, b) from mean OLS residual.
    #                   Whitens data + design in place; the regular
    #                   pack-and-fit path consumes the whitened
    #                   versions unchanged.
    #   arma11-voxel  — per-voxel (a, b) via REML grid search (same
    #                   primitives as ffs_reml).  Voxels are binned
    #                   by grid cell, each cell whitens + fits the
    #                   constrained basis-set solver independently;
    #                   per-cell betas are gathered into a synthetic
    #                   FLOBSFitResult that the downstream output
    #                   code consumes unchanged.  Suppresses
    #                   trial-to-trial amplitude oscillations from
    #                   autocorrelated noise.
    arma_ab: tuple[float, float] | None = None
    arma_ab_per_voxel: np.ndarray | None = None
    arma_cells: list[ARMAWhitenCell] | None = None
    if args.prewhiten != "none" and extra_regs_per_run is not None:
        # The ARMA path rebuilds its own nuisance from polort when it whitens,
        # so external columns would be dropped from the whitening AND from the
        # fit — silently leaving the noise they describe in the data.
        print(
            "ERROR: -ortvec* is not yet supported with -prewhiten "
            f"{args.prewhiten} (the ARMA whitening builds its own nuisance "
            "from -polort).  Run without prewhitening, or regress the "
            "components out beforehand.",
            file=sys.stderr,
        )
        return 1
    if args.prewhiten == "arma11":
        per_run_data, per_run_designs, a_opt, b_opt = estimate_and_apply_arma11_prewhitening(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            polort=polort_resolved,
            device=device,
            verbose=True,
        )
        arma_ab = (a_opt, b_opt)
    elif args.prewhiten == "arma11-voxel":
        if args.reg == "fracridge":
            print(
                "ERROR: -prewhiten arma11-voxel is not yet supported "
                "with -reg fracridge (cell-wise CV needs more work). "
                "Use -prewhiten arma11 (global) with fracridge for now."
            )
            return 1
        arma_ab_per_voxel = estimate_arma11_per_voxel(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            polort=polort_resolved,
            device=device,
            verbose=True,
        )
        arma_cells = bin_and_whiten_arma11(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            arma_per_voxel=arma_ab_per_voxel,
            polort=polort_resolved,
            device=device,
            verbose=True,
        )

    # ── Pack shared-task + block-diag polys ────────────────────────
    # Pack on the compute device so the fit doesn't waste cycles
    # bouncing the data back and forth.  load_and_preprocess_runs
    # produces data on CPU (memory-friendly), so this is the
    # one-time host→device transfer; from here through the solver
    # it stays on the chosen device.
    #
    # arma11-voxel skips this single packed-design path: each ARMA
    # cell has its own whitened design, so packing happens per cell
    # inside the dispatch block below.
    task_column_labels = [f"{lbl}#PC{b}" for lbl in block_labels for b in range(n_basis)]
    if arma_cells is None:
        # ── Per-voxel HRF selection (linear path) ──────────────────
        if hrf_shapes is not None:
            unsupported = [
                name
                for name, bad in (
                    ("-prewhiten " + str(args.prewhiten), args.prewhiten != "none"),
                    ("-lss", args.lss),
                    ("-prior-from " + str(args.prior_from), args.prior_from != "none"),
                    ("-reg fracridge", args.reg == "fracridge"),
                    ("-cv-runs", args.cv_runs),
                )
                if bad
            ]
            if unsupported:
                print(
                    f"ERROR: -hrf-shapes does not yet combine with "
                    f"{', '.join(unsupported)} under -parametrization linear.  "
                    "Each of those has its own per-voxel binning or CV loop "
                    "that would have to be crossed with the shape groups; "
                    "re-run without it, or use -parametrization shift.",
                    file=sys.stderr,
                )
                return 1

            if args.hrf_index:
                try:
                    hrf_index = _load_shape_index_map(
                        args.hrf_index,
                        n_shapes=hrf_shapes.shape[0],
                        volume_shape=volume_shape,
                        mask=mask,
                        n_voxels=per_run_data[0].shape[0],
                    )
                except (FileNotFoundError, ValueError) as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1
                print(f"  HRF assignment IMPORTED from {args.hrf_index}")
            elif args.hrf_select == "none":
                hrf_index = np.zeros(per_run_data[0].shape[0], dtype=np.int64)
            else:
                print(f"\n── Selecting one HRF per voxel ({args.hrf_select}) ──")
                hrf_index, hrf_index_score = _select_hrf_per_voxel(
                    shapes=hrf_shapes,
                    per_run_data=per_run_data,
                    all_onsets=all_onsets,
                    durations=[
                        float(timing.durations[c]) if timing.durations_given else 0.0
                        for c in range(len(condition_labels))
                    ],
                    run_starts=list(run_starts),
                    n_timepoints=n_timepoints,
                    tr=tr,
                    dt=basis.dt,
                    polort=polort_resolved,
                    extra_regs_per_run=extra_regs_per_run,
                    select_mode=args.hrf_select,
                    device=device,
                    verbose=args.verb >= 1,
                )
            occupied = np.bincount(hrf_index, minlength=hrf_shapes.shape[0])
            print(
                f"  Curves in use: {int((occupied > 0).sum())}/{hrf_shapes.shape[0]}  "
                f"(modal curve {int(np.argmax(occupied)) + 1} holds "
                f"{100 * occupied.max() / max(1, hrf_index.size):.0f}% of voxels)"
            )

        packed = pack_for_shared_task_glm(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            polort=polort_resolved,
            task_column_labels=task_column_labels,
            extra_regressors_per_run=extra_regs_per_run,
            device=device,
        )
        n_task_cols = packed.n_task_cols
        print(
            f"  Design: {packed.design_concat.shape}  "
            f"({n_task_cols} task + "
            f"{packed.design_concat.shape[1] - n_task_cols} nuisance)"
        )

        task_design = packed.design_concat[:, :n_task_cols]
        nuisance = (
            packed.design_concat[:, n_task_cols:]
            if packed.design_concat.shape[1] > n_task_cols
            else None
        )
    else:
        # n_task_cols is fixed by model; nuisance lives per cell.
        n_task_cols = n_blocks * n_basis

    # ── Fit ─────────────────────────────────────────────────────────
    # Eager full-HRF reconstruction (n_vox × n_blocks × n_t × 8 bytes)
    # is the main memory hog — for single-trial mode with hundreds of
    # blocks it can hit tens of GB.  Skip it inside the solver; we
    # build amplitude / iresp downstream in voxel chunks via the
    # memory module's chunk-size estimator.
    print("\n  Fitting …")
    if args.lss:
        if not args.single_trials:
            print("ERROR: -lss requires -single-trials.")
            return 1
        if args.reg == "fracridge":
            print(
                "ERROR: -lss is not yet supported with -reg fracridge. "
                "Use -reg in {none, ridge, mvn, mvn-shape}."
            )
            return 1
        if arma_cells is not None:
            print(
                "ERROR: -lss + -prewhiten arma11-voxel is not yet "
                "supported.  Use -prewhiten none or -prewhiten arma11 "
                "(global) with -lss for now."
            )
            return 1
        if args.vb_iters > 0:
            print(
                "WARNING: -vb-iters is ignored with -lss (LSS solves "
                "each trial independently; there's nothing to iterate)."
            )
        if args.prior_from == "per-condition":
            print(
                "ERROR: -lss + -prior-from per-condition is not yet "
                "supported.  LSS already anchors each trial to its "
                "condition-level structure via the rest-of-cond + "
                "other-cond design columns."
            )
            return 1
        # Validate excluded conditions are real condition labels.
        bad = [c for c in args.lss_exclude if c not in condition_labels]
        if bad:
            print(
                f"ERROR: -lss-exclude conditions {bad} not in "
                f"condition_labels {list(condition_labels)}."
            )
            return 1
        fit = fit_basis_lss(
            per_run_data=per_run_data,
            per_run_designs=per_run_designs,
            block_labels=block_labels,
            condition_labels=list(condition_labels),
            basis_functions=basis.basis_functions,
            prior_mean=prior_m,
            prior_cov=prior_C,
            polort=polort_resolved,
            prior_weight=pw,
            lambda_mode=args.lambda_mode,
            lambda_n_bins=args.lambda_n_bins,
            lss_exclude=args.lss_exclude,
            device=device,
            verbose=True,
        )
        print(
            f"  ✓ LSS fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
            f"effective λ_mean={fit.effective_prior_weight:.4g}"
        )
        print(
            f"  R² mean — LSA OLS (baseline): {fit.r2_ols.mean():.3f}  "
            f"LSA constrained: {fit.r2.mean():.3f}  "
            f"(LSS per-trial R² is not a single value; use -xval-r2)"
        )
    elif arma_cells is not None:
        # Per-cell constrained fit — each cell shares (a, b), so it
        # also shares its whitened task design and polynomial nuisance;
        # only the data (and the subset of voxels) differs.
        n_voxels_total = sum(c.voxel_indices.size for c in arma_cells)
        n_poly_per_run = (polort_resolved + 1) if polort_resolved >= 0 else 0
        n_total_cols = n_task_cols + n_runs * n_poly_per_run

        # Per-voxel diagnostics gathered across cells.  These are kept
        # at the top-level scope so the VB loop below can pass them
        # back into the next iteration (β_size update needs σ²_v +
        # λ_v from the current fit).
        sigma2_per_voxel_all = np.zeros(n_voxels_total, dtype=np.float32)
        lambda_per_voxel_all = np.zeros(n_voxels_total, dtype=np.float32)
        # Track per-cell packed designs so the block-trace pass can
        # reuse them without re-packing — pack itself is cheap but
        # this also lets us avoid re-whitening per cell.
        cell_packed_cache: list = []  # filled by _run_cell_fit

        def _run_cell_fit(
            cells: list[ARMAWhitenCell],
            *,
            prior_weight_per_voxel: np.ndarray | None = None,
        ) -> FLOBSFitResult:
            betas_all = np.zeros((n_voxels_total, n_total_cols), dtype=np.float64)
            betas_ols_all = np.zeros_like(betas_all)
            r2_all = np.zeros(n_voxels_total, dtype=np.float32)
            r2_ols_all = np.zeros_like(r2_all)
            sigma2_sum = 0.0
            eff_pw_sum = 0.0
            # Reset caches on each call (cells may have been rebinned).
            cell_packed_cache.clear()

            cell_iter = tqdm(
                cells,
                total=len(cells),
                desc="  Cells × constrained fit",
                unit="cell",
                leave=False,
                disable=len(cells) <= 1,
            )
            for cell in cell_iter:
                packed_cell = pack_for_shared_task_glm(
                    per_run_data=cell.per_run_data,
                    per_run_task_designs=cell.per_run_task_designs,
                    polort=-1,  # polys go via extra
                    task_column_labels=task_column_labels,
                    extra_regressors_per_run=cell.per_run_polys,
                    device=device,
                )
                cell_packed_cache.append(packed_cell)
                task_design_cell = packed_cell.design_concat[:, :n_task_cols]
                nuisance_cell = (
                    packed_cell.design_concat[:, n_task_cols:]
                    if packed_cell.design_concat.shape[1] > n_task_cols
                    else None
                )
                pw_cell = (
                    prior_weight_per_voxel[cell.voxel_indices]
                    if prior_weight_per_voxel is not None
                    else None
                )
                cell_fit = fit_basis_constrained_ridge(
                    data=packed_cell.data_concat,
                    design_task=task_design_cell,
                    basis_functions=basis.basis_functions,
                    prior_mean=prior_m,
                    prior_cov=prior_C,
                    n_blocks=n_blocks,
                    nuisance=nuisance_cell,
                    prior_weight=pw,
                    prior_weight_per_voxel=pw_cell,
                    device=device,
                    reconstruct_hrfs=False,
                    lambda_mode=args.lambda_mode,
                    lambda_n_bins=args.lambda_n_bins,
                    return_vb_diagnostics=True,
                )
                idx = cell.voxel_indices
                betas_all[idx] = cell_fit.betas[:, :n_total_cols]
                betas_ols_all[idx] = cell_fit.betas_ols[:, :n_total_cols]
                r2_all[idx] = cell_fit.r2.astype(np.float32)
                r2_ols_all[idx] = cell_fit.r2_ols.astype(np.float32)
                sigma2_sum += cell_fit.sigma2_mean * idx.size
                eff_pw_sum += cell_fit.effective_prior_weight * idx.size
                if cell_fit.sigma2_per_voxel is not None:
                    sigma2_per_voxel_all[idx] = cell_fit.sigma2_per_voxel
                if cell_fit.lambda_per_voxel is not None:
                    lambda_per_voxel_all[idx] = cell_fit.lambda_per_voxel

            return FLOBSFitResult(
                betas=betas_all,
                hrfs=None,  # type: ignore[arg-type]
                r2=r2_all,
                betas_ols=betas_ols_all,
                hrfs_ols=None,  # type: ignore[arg-type]
                r2_ols=r2_ols_all,
                sigma2_mean=sigma2_sum / max(1, n_voxels_total),
                effective_prior_weight=eff_pw_sum / max(1, n_voxels_total),
                n_iter=1,
                sigma2_per_voxel=sigma2_per_voxel_all.copy(),
                lambda_per_voxel=lambda_per_voxel_all.copy(),
            )

        fit = _run_cell_fit(arma_cells)
        print(
            f"  ✓ Per-voxel ARMA fit complete (iter 0).  "
            f"σ²_mean={fit.sigma2_mean:.4g}, "
            f"effective λ_mean={fit.effective_prior_weight:.4g}"
        )
        _print_r2(fit, args.reg)

        # ── VB iterative loop (filmbabe §3) ────────────────────────
        # Each iteration:
        #   1. Compute residuals in the original space from the
        #      current (a, b)-whitened fit's β.
        #   2. Re-estimate per-voxel (a, b) by REML on the residuals
        #      (intercept-only design — pure noise-covariance fit).
        #   3. Convergence test: median |Δa| + |Δb| < -vb-tol.
        #   4. Re-bin / re-whiten cells and refit.
        # Under REML the (a, b) estimate is independent of β when
        # using the full design Y — but feeding residuals back is
        # what makes the iteration meaningful: the AR fit now sees
        # the *cleaner* noise structure left after the constrained
        # β has absorbed task + nuisance variance.
        if args.vb_iters >= 1 and arma_ab_per_voxel is not None:
            # β_size_v: VB-updated prior precision multiplier per voxel.
            # Starts at the user's -prior-weight (1.0 for auto).  Updated
            # each iteration when -vb-update-prior is on, following
            # filmbabe's gamma posterior (TR04MW2 §3).
            user_mult = 1.0 if isinstance(pw, str) else float(pw)
            beta_size_per_voxel = np.full(
                n_voxels_total,
                float(user_mult),
                dtype=np.float32,
            )
            prior_pw_per_voxel: np.ndarray | None = None
            if args.vb_update_prior and args.reg in {"ridge", "mvn", "mvn-shape"}:
                # Compute initial β_size from the iter-0 fit's posterior
                # moments so that iter-1 fits with the *updated* prior.
                # Block-trace per voxel: σ²_v · Σ_b tr(C⁻¹ · (A⁻¹)_b)
                # via per-cell packed designs cached in _run_cell_fit.
                block_trace_summed = np.zeros(n_voxels_total, dtype=np.float32)
                for cell, packed_cell in zip(arma_cells, cell_packed_cache, strict=False):
                    idx = cell.voxel_indices
                    block_trace_summed[idx] = compute_vb_block_trace(
                        design=packed_cell.design_concat,
                        prior_cov=prior_C,
                        n_blocks=n_blocks,
                        n_basis=n_basis,
                        lambda_per_voxel=lambda_per_voxel_all[idx],
                        sigma2_per_voxel=sigma2_per_voxel_all[idx],
                        device=device,
                        n_bins=args.lambda_n_bins,
                        verbose=False,
                    )
                task_betas_3d = fit.betas[:, :n_task_cols].reshape(
                    n_voxels_total, n_blocks, n_basis
                )
                beta_size_per_voxel = vb_update_beta_size(
                    task_betas=task_betas_3d,
                    prior_mean=prior_m,
                    prior_cov=prior_C,
                    block_trace_summed=block_trace_summed,
                )
                # Translate to per-voxel λ: λ_v = β_size_v · σ²_v.
                prior_pw_per_voxel = beta_size_per_voxel * sigma2_per_voxel_all
                print(
                    f"  VB β_size update (iter 0): median={float(np.median(beta_size_per_voxel)):.3f}, "
                    f"5–95% [{float(np.percentile(beta_size_per_voxel, 5)):.3f}, "
                    f"{float(np.percentile(beta_size_per_voxel, 95)):.3f}]"
                )
            vb_iter = 0
            for vb_iter in range(1, args.vb_iters + 1):
                print(f"\n  VB iter {vb_iter}/{args.vb_iters}…")
                # Free the previous iter's whitened-cell state BEFORE
                # we allocate anything for this iter.  Cells hold ~7 GB
                # at 9.4T scale (data + packed designs per cell).
                # Residuals use the original per_run_data / fit.betas,
                # not the cells, so dropping them now is safe — and
                # essential, since compute_per_voxel_residuals itself
                # OOMs allocating y_pred when 14 GB of cells are
                # still resident.
                arma_cells = None
                cell_packed_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # 1. Residuals in original space.
                nuis_betas_arr = (
                    fit.betas[:, n_task_cols:] if fit.betas.shape[1] > n_task_cols else None
                )
                residuals_per_run = compute_per_voxel_residuals(
                    per_run_data=per_run_data,
                    per_run_task_designs=per_run_designs,
                    polort=polort_resolved,
                    task_betas=fit.betas[:, :n_task_cols],
                    nuisance_betas=nuis_betas_arr,
                    device=device,
                )
                # 2. Re-estimate (a, b) from residuals.  Use polort=-1
                # (residuals are already drift-free).
                new_ab = estimate_arma11_per_voxel(
                    per_run_data=residuals_per_run,
                    per_run_task_designs=per_run_designs,
                    polort=-1,
                    device=device,
                    verbose=True,
                )
                # Free residuals before re-binning + re-fit, which
                # will allocate new cells.
                del residuals_per_run
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # 3. Convergence on (a, b).
                delta = np.abs(new_ab - arma_ab_per_voxel).sum(axis=1)
                median_delta = float(np.median(delta))
                print(
                    f"  VB iter {vb_iter}: median |Δa|+|Δb| = "
                    f"{median_delta:.4f}  (tol={args.vb_tol})"
                )
                arma_ab_per_voxel = new_ab
                if median_delta < args.vb_tol and not args.vb_update_prior:
                    # (a, b) loop converged and no β_size update wanted.
                    print(f"  VB converged at iter {vb_iter}.")
                    break
                # 4. Re-bin + re-fit (with optional VB β_size override).
                arma_cells = bin_and_whiten_arma11(
                    per_run_data=per_run_data,
                    per_run_task_designs=per_run_designs,
                    arma_per_voxel=arma_ab_per_voxel,
                    polort=polort_resolved,
                    device=device,
                    verbose=True,
                )
                fit = _run_cell_fit(
                    arma_cells,
                    prior_weight_per_voxel=prior_pw_per_voxel,
                )
                print(
                    f"  VB iter {vb_iter} fit: σ²_mean={fit.sigma2_mean:.4g}, "
                    f"R² constrained={fit.r2.mean():.3f}"
                )
                # 5. VB β_size update from new posterior.
                if args.vb_update_prior and args.reg in {"ridge", "mvn", "mvn-shape"}:
                    block_trace_summed = np.zeros(n_voxels_total, dtype=np.float32)
                    for cell, packed_cell in zip(arma_cells, cell_packed_cache, strict=False):
                        idx = cell.voxel_indices
                        block_trace_summed[idx] = compute_vb_block_trace(
                            design=packed_cell.design_concat,
                            prior_cov=prior_C,
                            n_blocks=n_blocks,
                            n_basis=n_basis,
                            lambda_per_voxel=lambda_per_voxel_all[idx],
                            sigma2_per_voxel=sigma2_per_voxel_all[idx],
                            device=device,
                            n_bins=args.lambda_n_bins,
                            verbose=False,
                        )
                    task_betas_3d = fit.betas[:, :n_task_cols].reshape(
                        n_voxels_total, n_blocks, n_basis
                    )
                    new_beta_size = vb_update_beta_size(
                        task_betas=task_betas_3d,
                        prior_mean=prior_m,
                        prior_cov=prior_C,
                        block_trace_summed=block_trace_summed,
                    )
                    bs_change = float(np.median(np.abs(new_beta_size - beta_size_per_voxel)))
                    beta_size_per_voxel = new_beta_size
                    prior_pw_per_voxel = beta_size_per_voxel * sigma2_per_voxel_all
                    print(
                        f"  VB β_size iter {vb_iter}: "
                        f"median={float(np.median(beta_size_per_voxel)):.3f}, "
                        f"5–95% [{float(np.percentile(beta_size_per_voxel, 5)):.3f}, "
                        f"{float(np.percentile(beta_size_per_voxel, 95)):.3f}], "
                        f"median |Δβ_size|={bs_change:.4f}"
                    )
            print(f"  ✓ VB loop complete after {vb_iter} iter(s).")

        # Stage-exit cleanup.  ``arma_cells`` holds the per-cell
        # whitened (data, design, polys) tensors — ~7 GB on
        # 9.4T-scale brains.  ``cell_packed_cache`` mirrors them
        # via pack_for_shared_task_glm's output.  Nothing downstream
        # uses either: ``fit.betas`` carries all per-voxel info we
        # need, the original ``per_run_data`` is intact for
        # amplitude reconstruction, and ``per_run_data_orig`` is
        # already cloned for -xval-r2.  Drop them now so the next
        # stage (amplitude/iresp, xval-r2) doesn't hit OOM.
        arma_cells = None
        cell_packed_cache.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elif hrf_index is not None:
        # Per-voxel HRF: one design per distinct curve, not per voxel.
        # ``fit_one`` re-enters the ordinary single-design path on a
        # voxel subset, so each -reg branch behaves exactly as it does
        # without shape selection.
        def _fit_one(voxel_idx: np.ndarray, designs: list[torch.Tensor]):
            packed_g = pack_for_shared_task_glm(
                per_run_data=[d[voxel_idx] for d in per_run_data],
                per_run_task_designs=designs,
                polort=polort_resolved,
                task_column_labels=task_column_labels,
                extra_regressors_per_run=extra_regs_per_run,
                device=device,
            )
            td = packed_g.design_concat[:, : packed_g.n_task_cols]
            nz = (
                packed_g.design_concat[:, packed_g.n_task_cols :]
                if packed_g.design_concat.shape[1] > packed_g.n_task_cols
                else None
            )
            common = dict(
                data=packed_g.data_concat,
                design_task=td,
                basis_functions=basis.basis_functions,
                prior_mean=prior_m,
                prior_cov=prior_C,
                n_blocks=n_blocks,
                nuisance=nz,
                prior_weight=pw,
                device=device,
                lambda_mode=args.lambda_mode,
                reconstruct_hrfs=False,
            )
            if args.reg == "cone":
                return fit_basis_cone_prior(**common, verbose=False)
            return fit_basis_constrained_ridge(**common, lambda_n_bins=args.lambda_n_bins)

        fit, group_bases, shape_groups, group_designs = _fit_shape_groups(
            hrf_index=hrf_index,
            shapes=hrf_shapes,
            n_basis=n_basis,
            block_durations=block_durations,
            block_onsets_per_run=block_onsets_per_run,
            per_run_data=per_run_data,
            basis_dt=basis.dt,
            basis_mode=basis_mode,
            tr=tr,
            n_tp_per_run=n_tp_per_run,
            fit_one=_fit_one,
        )
        print(
            f"  ✓ Fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
            f"effective λ={fit.effective_prior_weight:.4g}"
        )
        _print_r2(fit, args.reg)

    elif args.reg == "fracridge":
        # fracridge has its own per-run nuisance projection and
        # SVD-based multi-frac solver, so it bypasses the packed
        # block-diag design entirely and consumes per_run_data /
        # per_run_designs directly.  The dispatch keeps the
        # downstream output code (amplitude, iresp, etc.) untouched
        # because FracRidgeFitResult mirrors FLOBSFitResult's fields.
        try:
            fracs_grid = np.array(
                [float(x) for x in args.fracs.split(",") if x.strip()],
                dtype=np.float64,
            )
        except ValueError as e:
            print(f"ERROR: -fracs must be comma-separated floats: {e}")
            return 1
        if fracs_grid.size == 0 or (fracs_grid <= 0).any() or (fracs_grid > 1).any():
            print(f"ERROR: -fracs values must be in (0, 1]; got {fracs_grid.tolist()}")
            return 1
        fracs_grid.sort()
        fit = fit_basis_fracridge(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            n_blocks=n_blocks,
            n_basis=n_basis,
            polort=polort_resolved,
            fracs=fracs_grid,
            device=device,
            verbose=True,
        )
        print(
            f"  ✓ Fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
            f"median optimal frac = {float(np.median(fit.optimal_fracs)):.2f}"
        )
        print(
            f"  Held-out R² mean — OLS (frac=1.0): {fit.r2_ols.mean():.3f}  "
            f"fracridge optimal: {fit.r2.mean():.3f}"
        )
    elif args.reg == "cone":
        fit = fit_basis_cone_prior(
            data=packed.data_concat,
            design_task=task_design,
            basis_functions=basis.basis_functions,
            prior_mean=prior_m,
            prior_cov=prior_C,
            n_blocks=n_blocks,
            nuisance=nuisance,
            prior_weight=pw,
            device=device,
            lambda_mode=args.lambda_mode,
            reconstruct_hrfs=False,
            verbose=args.verb >= 1,
        )
        print(
            f"  ✓ Cone fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
            f"λ_mean={fit.effective_prior_weight:.4g}, "
            f"{fit.n_iter} IRLS iter(s)"
        )
        _print_r2(fit, args.reg)
    else:
        fit = fit_basis_constrained_ridge(
            data=packed.data_concat,
            design_task=task_design,
            basis_functions=basis.basis_functions,
            prior_mean=prior_m,
            prior_cov=prior_C,
            n_blocks=n_blocks,
            nuisance=nuisance,
            prior_weight=pw,
            device=device,
            reconstruct_hrfs=False,
            lambda_mode=args.lambda_mode,
            lambda_n_bins=args.lambda_n_bins,
        )
        print(
            f"  ✓ Fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
            f"effective λ={fit.effective_prior_weight:.4g}"
        )
        _print_r2(fit, args.reg)

    # ── Stage cleanup: release fit-stage GPU tensors ───────────────
    # Everything downstream (amplitude / iresp reconstruction,
    # xval-r2, output writes) reads ``fit`` (CPU numpy) or
    # ``per_run_data`` / ``per_run_data_orig`` (already on CPU).
    # The packed task+nuisance design that fit_basis_constrained_ridge
    # consumed (and fracridge's internals) is no longer needed —
    # explicit release prevents it from sitting on GPU through the
    # xval-r2 path which OOMs on big data otherwise.  Each branch
    # above sets these only in the paths that use them; try/except
    # keeps the cleanup branch-agnostic.
    try:
        del packed
    except NameError:
        pass
    try:
        del task_design
    except NameError:
        pass
    try:
        del nuisance
    except NameError:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Reshape + save outputs ─────────────────────────────────────
    n_vox_masked = fit.betas.shape[0]
    nx, ny, nz = volume_shape

    def _to_volume(masked: np.ndarray) -> np.ndarray:
        out_shape = (nx, ny, nz) + tuple(masked.shape[1:])
        out = np.zeros(out_shape, dtype=np.float32)
        if mask is not None:
            out[mask, ...] = masked
        else:
            out = masked.reshape(out_shape)
        return out

    # Restore the empirical-Bayes per-condition prior mean back into
    # the betas (we fit β_centered = β − m_v on shifted data; the user-
    # facing output should be the full β_trial = β_centered + β_cond).
    # Both the constrained and the OLS betas need the same restoration.
    if empirical_prior_mean_full is not None:
        fit.betas[:, :n_task_cols] = fit.betas[:, :n_task_cols] + empirical_prior_mean_full
        fit.betas_ols[:, :n_task_cols] = fit.betas_ols[:, :n_task_cols] + empirical_prior_mean_full
        print(
            "  Restored per-condition empirical prior into single-trial "
            "betas; output reflects β_trial in original signal units."
        )

    task_betas = fit.betas[:, :n_task_cols].reshape(n_vox_masked, n_blocks, n_basis)
    task_betas_ols = fit.betas_ols[:, :n_task_cols].reshape(n_vox_masked, n_blocks, n_basis)

    # ── Latency / width readout from the derivative coefficients ────
    # Deliberately on the OLS betas: -reg cone's prior mean points along
    # the canonical axis and penalises angular deviation from it, which
    # is shrinkage of exactly the ratio being measured here.
    shape_maps: list[dict[str, np.ndarray]] | None = None
    shape_calibs: list = []
    shape_calib_note = ""
    if args.save_shape:
        from fastfuncstuff.design.basis_shape import usable_region

        print("\n── Latency / width readout (SPMG derivative ratios) ──")
        print(f"  Calibrating {n_blocks} condition(s) over ±{args.shape_tau_max:g} s")
        readout_kw = dict(
            n_deriv=n_deriv,
            block_onsets_per_run=block_onsets_per_run,
            basis_lag_times=basis_lag_times,
            basis_dt=basis.dt,
            basis_mode=basis_mode,
            tr=tr,
            n_tp_per_run=list(n_tp_per_run),
            n_basis=n_basis,
            tau_max=args.shape_tau_max,
            r2_floor=args.shape_r2_floor,
            block_durations=block_durations,
        )

        if hrf_index is None:
            shape_maps, shape_calibs = _shape_readout(
                betas_ols=task_betas_ols,
                per_run_designs=[d.cpu().numpy() for d in per_run_designs],
                base_hrf=None
                if str(args.hrf).strip().lower() == "canonical"
                else _resolve_shift_hrf(args.hrf, basis.dt, basis.duration),
                **readout_kw,
            )
        else:
            # The ratio a given latency produces depends on the curve the
            # derivatives were built around, so the calibration is per
            # shape GROUP as well as per condition.  Voxels are scattered
            # back into whole-volume maps afterwards.
            assert hrf_shapes is not None
            n_vox_sel = task_betas_ols.shape[0]
            keys = ["latency", "dispersion", "fwhm", "shape_r2"]
            acc: list[dict[str, np.ndarray]] = [
                {k: np.full(n_vox_sel, np.nan, dtype=np.float64) for k in keys}
                | {"valid": np.zeros(n_vox_sel, dtype=bool)}
                for _ in range(n_blocks)
            ]
            occupied = [(k, idx) for k, idx in enumerate(shape_groups) if idx.size]
            for k, idx in tqdm(
                occupied, desc="  Calibrating groups", leave=True, disable=len(occupied) <= 1
            ):
                g_maps, g_cals = _shape_readout(
                    betas_ols=task_betas_ols[idx],
                    per_run_designs=group_designs[k],
                    base_hrf=hrf_shapes[k],
                    **readout_kw,
                )
                for b in range(n_blocks):
                    for key in keys:
                        acc[b][key][idx] = g_maps[b][key]
                    acc[b]["valid"][idx] = g_maps[b]["valid"]
                if k == int(np.bincount(hrf_index, minlength=hrf_shapes.shape[0]).argmax()):
                    shape_calibs = g_cals  # modal curve, for the calibration TSV
            shape_maps = acc
            shape_calib_note = "  (calibration TSV is the modal curve's; the map is per-group)"
        for lbl, res, calib in zip(block_labels, shape_maps, shape_calibs, strict=False):
            frac = float(np.mean(res["valid"]))
            med = float(np.median(res["latency"][res["valid"]])) if frac > 0 else float("nan")
            keep = usable_region(calib, args.shape_r2_floor)
            rows = np.where(keep.any(axis=1))[0]
            env = (
                f"[{calib.taus[rows[0]]:+.2f},{calib.taus[rows[-1]]:+.2f}]s"
                if rows.size
                else "EMPTY"
            )
            print(
                f"    {lbl}: envelope {env}, {frac * 100:.1f}% in range, "
                f"median latency {med:+.3f} s"
            )

    # Amplitude = peak of reconstructed HRF per (voxel, block).  Computed
    # in voxel chunks via the memory module so we never materialise the
    # full (n_voxels × n_blocks × n_t) HRF tensor — in single-trial mode
    # that's tens of GB.
    #
    # Two time grids in play:
    #   basis.dt          high resolution (default 0.1 s) used for
    #                     accurate amplitude estimation (peak finding).
    #   iresp_dt          user-chosen save resolution (default = TR).
    #                     Iresp NIfTIs are resampled to this grid
    #                     before writing — typical 10-20× size reduction.
    from fastfuncstuff.memory import estimate_chunk_size, get_available_memory

    n_t_basis = basis.basis_functions.shape[1]

    # Resolve -iresp-dt: default to TR.
    iresp_dt = float(args.iresp_dt) if args.iresp_dt is not None else float(tr)
    if iresp_dt < basis.dt:
        print(
            f"  WARNING: -iresp-dt ({iresp_dt}s) is finer than basis "
            f"dt ({basis.dt}s); clamping to basis dt (no upsampling)."
        )
        iresp_dt = basis.dt
    # Build resampled basis ONCE (linear interpolation along time axis).
    # Used for iresp save only; amplitude still computed at basis.dt.
    src_times = np.arange(n_t_basis) * basis.dt
    n_t_iresp = max(1, int(np.floor((basis.duration - basis.dt) / iresp_dt)) + 1)
    dst_times = np.arange(n_t_iresp) * iresp_dt

    # Per BLOCK, because each block's basis carries its own stimulus
    # duration — a 1 s cue and an 8 s block do not share a regressor
    # shape, so they must not share a reconstruction basis either.
    def _resample(curves: np.ndarray) -> np.ndarray:
        """(..., K, n_t_basis) -> (..., K, n_t_iresp)."""
        flat = curves.reshape(-1, curves.shape[-1])
        out = np.stack([np.interp(dst_times, src_times, row) for row in flat], axis=0)
        return out.reshape(curves.shape[:-1] + (dst_times.size,)).astype(np.float64)

    # Per-curve, NOT per-voxel: a per-voxel basis stack would be
    # n_vox x n_blocks x K x n_t, which is tens of GB on a real volume.
    # The chunk loop gathers each voxel's curve out of this instead.
    basis_iresp_groups = (
        _resample(group_bases) if hrf_index is not None else None
    )  # (n_shapes, n_blocks, K, n_t_iresp)
    basis_iresp = _resample(block_bases)  # (n_blocks, K, n_t_iresp)
    _resp_note = "response" if args.save_iresp_off else "iresp / response"
    print(
        f"  Reconstruction grid ({_resp_note}): dt={iresp_dt:.3f}s × "
        f"{n_t_iresp} samples (curves stored at {basis.dt:.3f}s × {n_t_basis})"
    )

    # Chunk size: amplitude peak-finding and (optional) iresp save
    # both share a single HRF tensor reconstructed at the resampled
    # iresp grid (chunk × n_blocks × n_t_iresp × 8, float64 from the
    # matmul).  The previous code reconstructed at basis.dt (0.1 s,
    # ~320 samples per HRF) for amplitude only — for single-trial
    # fits with hundreds of blocks that materialised a multi-GB
    # tensor per chunk and OOM-crashed.  TR resolution costs ~1-3 %
    # peak accuracy (HRF is smooth near its extremum) and is
    # ~15× smaller.
    bytes_per_vox = 2 * n_blocks * n_t_iresp * 8
    chunk_size = estimate_chunk_size(
        n_voxels=n_vox_masked,
        n_timepoints=n_t_basis,
        n_regressors=n_blocks * n_basis,
        device=torch.device("cpu"),
        operation="glm",
    )
    avail_bytes = get_available_memory(torch.device("cpu"))
    chunk_from_hrf = max(1, int(avail_bytes * 0.25 / max(bytes_per_vox, 1)))
    chunk_size = min(chunk_size, chunk_from_hrf)
    print(f"  Reconstruction chunking: {chunk_size:,} voxels per chunk (n_blocks={n_blocks})")

    amplitude = np.zeros((n_vox_masked, n_blocks), dtype=np.float32)
    amplitude_ols = np.zeros_like(amplitude)
    # Pre-allocate full iresp ONLY when -save-iresp and we're in the
    # per-condition path (n_blocks small).  Single-trial mode skips
    # the eager full iresp; see "iresp save" section below for the
    # per-condition save path that streams chunks into save_iresp.
    # Resolve iresp save policy:
    #   - per-condition (n_blocks small):  ON by default.
    #   - single-trial:                    OFF by default; -save-iresp
    #                                      to force on; -no-iresp to
    #                                      keep off (redundant).
    #   - -no-iresp:                       ALWAYS off, in either mode.
    if args.save_iresp_off:
        save_full_iresp = False
        if args.save_iresp_explicit:
            print("  -no-iresp overrides -save-iresp; iresp save: OFF.")
    elif args.single_trials:
        save_full_iresp = bool(args.save_iresp_explicit)
        if not save_full_iresp:
            print(
                "  Single-trial mode: per-trial iresp save is OFF by "
                "default (pass -save-iresp to opt in).  Amplitude "
                "maps still emitted."
            )
    else:
        save_full_iresp = True  # per-condition default
    if save_full_iresp:
        # Memory estimate for the full iresp tensor at the resampled grid:
        full_iresp_bytes = 2 * n_vox_masked * n_blocks * n_t_iresp * 4  # float32
        full_iresp_gb = full_iresp_bytes / (1024**3)
        if full_iresp_gb > 8.0:
            print(
                f"  WARNING: full iresp tensor would be {full_iresp_gb:.1f} GB "
                f"even at iresp_dt={iresp_dt}s.  Skipping iresp save.  "
                "Amplitude maps are still emitted; try a coarser -iresp-dt."
            )
            save_full_iresp = False
        else:
            print(f"  Per-block iresp save: ~{full_iresp_gb:.2f} GB in memory.")
    iresp_buf = (
        np.zeros((n_vox_masked, n_blocks, n_t_iresp), dtype=np.float32) if save_full_iresp else None
    )
    iresp_buf_ols = (
        np.zeros((n_vox_masked, n_blocks, n_t_iresp), dtype=np.float32) if save_full_iresp else None
    )

    def _signed_peak(hrfs: np.ndarray) -> np.ndarray:
        """Peak of the reconstructed HRF, **preserving sign**.

        Voxels with inverted BOLD responses (e.g. CSF-adjacent voxels,
        or anti-correlated networks) have HRFs whose largest deflection
        is *negative*.  Taking ``hrfs.max(axis=-1)`` reports +0.05 for
        such voxels (the tiny positive ripple before the trough),
        which makes second-level maps look uniformly positive.  Instead
        pick the time-point with the largest *absolute* value and
        report the signed value there — positive peaks stay positive,
        negative troughs come out as negative numbers.
        """
        idx = np.argmax(np.abs(hrfs), axis=-1, keepdims=True)
        return np.take_along_axis(hrfs, idx, axis=-1).squeeze(-1)

    n_amp_chunks = (n_vox_masked + chunk_size - 1) // chunk_size
    for start in tqdm(
        range(0, n_vox_masked, chunk_size),
        total=n_amp_chunks,
        desc="  Amplitude/iresp",
        unit="chunk",
        leave=False,
        disable=n_amp_chunks <= 1,
    ):
        end = min(start + chunk_size, n_vox_masked)
        # Amplitude and iresp share a single HRF reconstruction at
        # TR (iresp_dt) resolution.  Peak finding on this grid is
        # within ~1-3 % of the high-res peak because the BOLD HRF is
        # smooth near its extremum; in exchange the per-voxel cost
        # drops ~15× vs the basis.dt reconstruction (critical for
        # single-trial mode with hundreds of blocks).
        if basis_iresp_groups is None:
            hrfs_chunk = np.einsum("vbk,bkt->vbt", task_betas[start:end], basis_iresp)
            hrfs_ols_chunk = np.einsum("vbk,bkt->vbt", task_betas_ols[start:end], basis_iresp)
        else:
            gb = basis_iresp_groups[hrf_index[start:end]]  # (chunk, n_blocks, K, n_t)
            hrfs_chunk = np.einsum("vbk,vbkt->vbt", task_betas[start:end], gb)
            hrfs_ols_chunk = np.einsum("vbk,vbkt->vbt", task_betas_ols[start:end], gb)
        amplitude[start:end] = _signed_peak(hrfs_chunk).astype(np.float32)
        amplitude_ols[start:end] = _signed_peak(hrfs_ols_chunk).astype(np.float32)
        if iresp_buf is not None:
            iresp_buf[start:end] = hrfs_chunk.astype(np.float32)
            iresp_buf_ols[start:end] = hrfs_ols_chunk.astype(np.float32)

    if hrf_index is not None and hrf_shapes is not None:
        # 1-based on disk, matching ffs_hrfopt's convention, so the map
        # round-trips straight back in through -hrf-index.
        bucket = [(hrf_index + 1).astype(np.float32)]
        labels = ["hrf_index"]
        if hrf_index_score is not None:
            bucket.append(hrf_index_score.astype(np.float32))
            labels.append("hrf_select_r2")
        idx_path = f"{args.prefix}_fitbasis_hrf_index{nii_ext}"
        save_nifti(
            _to_volume(np.stack(bucket, axis=1)),
            output_path=idx_path,
            reference_img=args.input[0],
            brick_labels=labels,
        )
        print(f"  Wrote {idx_path}")
        shapes_path = f"{args.prefix}_fitbasis_hrf_shapes.tsv"
        np.savetxt(shapes_path, hrf_shapes.T, fmt="%.10g", delimiter="\t")
        print(f"  Wrote {shapes_path}")

    # Basis TSV (shared)
    basis_path = f"{args.prefix}_fitbasis_basis.tsv"
    np.savetxt(basis_path, basis.basis_functions.T, fmt="%.10g", delimiter="\t")
    print(f"  Wrote {basis_path}")

    from fastfuncstuff.cli_utils import spinner

    # The in-sample R2 is not written on its own any more: as a standalone
    # map it invites thresholding by a number that free parameters inflate.
    # It rides along as a labelled sub-brick of each bucket instead, next to
    # the held-out R2 that is the honest referee.
    xval_r2: np.ndarray | None = None
    write_ols = args.reg != "none"  # with no prior, "constrained" IS the OLS fit

    # ── Cross-validated R² (held-out) ──────────────────────────────
    # Per the user's choice of -reg + -prior-weight, do LORO with
    # train/test runs and report per-voxel held-out R² as a 3-D
    # NIfTI.  Operates on the un-modified per_run_data (we kept a
    # copy upstream when -xval-r2 was on).  Skips prewhitening and
    # VB iteration inside — the xval measures the model's
    # generalization, not the noise-model layer.
    if args.xval_r2:
        if args.reg in {"fracridge", "mvn-shape"}:
            print(f"  -xval-r2 not yet supported with -reg {args.reg}; skipping.")
        elif n_runs < 2:
            print("  -xval-r2 needs ≥2 runs; skipping.")
        elif per_run_data_orig is None:
            print("  -xval-r2 requested but per_run_data_orig was not saved (internal); skipping.")
        else:
            xval_r2 = compute_xval_r2_per_voxel(  # noqa: F841 — consumed by the buckets
                per_run_data=per_run_data_orig,
                all_onsets=all_onsets,
                condition_labels=list(condition_labels),
                basis_functions=basis.basis_functions,
                basis_functions_per_cond=_build_block_bases(
                    basis,
                    [
                        float(timing.durations[c]) if timing.durations_given else 0.0
                        for c in range(len(condition_labels))
                    ],
                ),
                basis_lag_times=basis_lag_times,
                basis_mode=basis_mode,
                tr=tr,
                n_tp_per_run=n_tp_per_run,
                polort=polort_resolved,
                prior_mean=prior_m,
                prior_cov=prior_C,
                prior_weight=pw,
                single_trials=args.single_trials,
                # Single-trial mode: re-use existing betas (no per-fold re-fit).
                single_trial_betas=task_betas if args.single_trials else None,
                block_labels=block_labels if args.single_trials else None,
                cone_prior=args.reg == "cone",
                device=device,
                verbose=args.verb >= 1,
            )
            xval_path = f"{args.prefix}_fitbasis_xvalr2{nii_ext}"
            with spinner(f"Writing {Path(xval_path).name}"):
                save_nifti(
                    _to_volume(xval_r2[:, None]).squeeze(-1),
                    output_path=xval_path,
                    reference_img=args.input[0],
                )
            print(
                f"  Wrote {xval_path}  "
                f"(median={float(np.median(xval_r2)):.3f}, "
                f"mean={float(np.mean(xval_r2)):.3f}, "
                f"max={float(np.max(xval_r2)):.3f})"
            )

    # QC sub-bricks every bucket carries: the honest referee first.
    qc_bricks: list[tuple[str, np.ndarray]] = []
    if xval_r2 is not None:
        qc_bricks.append(("xvalR2", np.asarray(xval_r2, dtype=np.float32)))
    qc_bricks.append(("taskR2", np.asarray(fit.r2, dtype=np.float32)))

    # Diagnostics bucket, named to match the shift parametrisation's so the
    # two are comparable file for file.
    _diag = [(np.asarray(fit.r2, dtype=np.float32), "taskR2")]
    if write_ols:
        _diag.append((np.asarray(fit.r2_ols, dtype=np.float32), "taskR2_unconstrained"))
    _save_bucket(
        [a for a, _ in _diag],
        [n for _, n in _diag],
        f"{args.prefix}_fitbasis_diagnostics{nii_ext}",
        to_volume=_to_volume,
        reference_img=args.input[0],
    )

    # ── Per-block outputs ──────────────────────────────────────────
    # Single-trial mode: amplitude maps stack across trials per cond
    # for downstream 2nd-level convenience.  iresp & pcweights are still
    # per-trial 4-D NIfTIs (saved via save_iresp grouping).
    if args.single_trials:
        # Group block_labels back into per-condition trial lists for the
        # amplitude 4D map (time = trial number).  Also emit the iresp
        # per trial.
        per_cond_trial_idx: dict[str, list[int]] = {}
        for b_idx, lbl in enumerate(block_labels):
            cond = lbl.split("_trial", 1)[0]
            per_cond_trial_idx.setdefault(cond, []).append(b_idx)

        # Per-condition amplitude 4D (last axis = trial number)
        for cond, idxs in tqdm(
            per_cond_trial_idx.items(),
            desc="  Single-trial amplitudes",
            unit="cond",
            leave=False,
            disable=len(per_cond_trial_idx) <= 1,
        ):
            _amp_variants = [(amplitude, "")]
            if write_ols:
                _amp_variants.append((amplitude_ols, "_unconstrained"))
            for amps, sfx in _amp_variants:
                amp_stack = amps[:, idxs]  # (n_vox, n_trials_for_cond)
                amp_vol = _to_volume(amp_stack)  # (nx, ny, nz, n_trials)
                path = f"{args.prefix}_fitbasis_amplitude_{cond}{sfx}{nii_ext}"
                save_nifti(amp_vol, output_path=path, reference_img=args.input[0])
                print(f"  Wrote {path}  (n_trials={len(idxs)})")

        if save_full_iresp and iresp_buf is not None:
            # Per-trial iresp 4-D (time axis = HRF basis-dt).  Only
            # available when the iresp buffer fit in memory above.
            _iresp_variants = [(iresp_buf, "")]
            if write_ols:
                _iresp_variants.append((iresp_buf_ols, "_unconstrained"))
            for hrfs_arr, sfx in _iresp_variants:
                iresp_vol = _to_volume(hrfs_arr)
                save_iresp(
                    iresp=iresp_vol,
                    output_prefix=f"{args.prefix}_fitbasis",
                    condition_labels=[f"{lbl}{sfx}" for lbl in block_labels],
                    tr=iresp_dt,
                    bot=0.0,
                    top=iresp_dt * (n_t_iresp - 1),
                    reference_img=args.input[0],
                    nii_ext=nii_ext,
                )
            print(f"  Wrote {n_blocks} × {len(_iresp_variants)} iresp file(s).")

        # Basis weights per trial (optional but useful for diagnostics).
        # One 4-D NIfTI per basis function: time axis = trial number.
        # This makes each basis-coefficient trajectory directly
        # viewable as a time series (basisweight01 = canonical amp
        # across trials, basisweight02 = derivative, etc.) instead of
        # the previous trial-major interleaving where coefficients
        # for different basis functions were mixed.
        for cond, idxs in tqdm(
            per_cond_trial_idx.items(),
            desc="  Single-trial basis weights",
            unit="cond",
            leave=False,
            disable=len(per_cond_trial_idx) <= 1,
        ):
            stack = task_betas[:, idxs, :]  # (n_vox, n_trials, K)
            stack_ols = task_betas_ols[:, idxs, :]
            _stack_variants = [(stack, "")]
            if write_ols:
                _stack_variants.append((stack_ols, "_unconstrained"))
            for arr, sfx in _stack_variants:
                for b in range(n_basis):
                    # (n_vox, n_trials) → (nx, ny, nz, n_trials)
                    vol_4d = _to_volume(arr[:, :, b])
                    path = f"{args.prefix}_fitbasis_basisweight{b + 1:02d}_{cond}{sfx}{nii_ext}"
                    save_nifti(vol_4d, output_path=path, reference_img=args.input[0])
            print(f"  Wrote basisweight01..{n_basis:02d} for {cond} (n_trials={len(idxs)})")

    else:
        # Per-condition outputs (the simple case)
        if save_full_iresp and iresp_buf is not None:
            _iresp_variants = [(iresp_buf, "")]
            if write_ols:
                _iresp_variants.append((iresp_buf_ols, "_unconstrained"))
            for hrfs_arr, sfx in _iresp_variants:
                iresp_vol = _to_volume(hrfs_arr)
                save_iresp(
                    iresp=iresp_vol,
                    output_prefix=f"{args.prefix}_fitbasis",
                    condition_labels=[f"{lbl}{sfx}" for lbl in block_labels],
                    tr=iresp_dt,
                    bot=0.0,
                    top=iresp_dt * (n_t_iresp - 1),
                    reference_img=args.input[0],
                    nii_ext=nii_ext,
                )
            print(f"  Wrote {n_blocks} × {len(_iresp_variants)} iresp file(s).")

        # One labelled bucket per quantity instead of 2 x n_conditions
        # loose files: 8 conditions used to mean 32 volumes whose only
        # distinguishing feature was the filename.
        variants = [(task_betas, amplitude, "")]
        if write_ols:
            variants.append((task_betas_ols, amplitude_ols, "_unconstrained"))
        for tbetas, amps, sfx in variants:
            _save_bucket(
                [amps[:, b] for b in range(n_blocks)],
                list(block_labels),
                f"{args.prefix}_fitbasis_amplitude{sfx}{nii_ext}",
                to_volume=_to_volume,
                reference_img=args.input[0],
                qc=qc_bricks,
            )
            if not args.no_basisweights:
                _save_bucket(
                    [tbetas[:, b, k] for b in range(n_blocks) for k in range(n_basis)],
                    [
                        f"{lbl}_{name}"
                        for lbl in block_labels
                        for name in _BASIS_COL_NAMES[:n_basis]
                    ],
                    f"{args.prefix}_fitbasis_basisweights{sfx}{nii_ext}",
                    to_volume=_to_volume,
                    reference_img=args.input[0],
                    qc=qc_bricks,
                )

    if shape_maps is not None:
        # SPMG2 holds width at canonical by construction, so writing a
        # constant map would only invite it being interpreted.
        # Latency minus the voxel's own mean across conditions.  When each
        # voxel selected its own curve, absolute latency is measured
        # against THAT curve, so a curve peaking 0.2 s late shifts every
        # condition in that voxel alike — measured, and it is the same
        # trap -shift-shapes documents for the delay parameter.  The
        # deviation is what survives: cross-condition deltas came back
        # +0.628 / +1.493 against a true +0.600 / +1.500.
        if n_blocks > 1:
            lat_stack = np.stack([m["latency"] for m in shape_maps], axis=1)
            with np.errstate(invalid="ignore"):
                lat_mean = np.nanmean(lat_stack, axis=1, keepdims=True)
            for b in range(n_blocks):
                shape_maps[b]["latency_dev"] = lat_stack[:, b] - lat_mean[:, 0]

        # Interleaved per condition, so the value and the map you threshold
        # it BY sit next to each other: latency_A, valid_A, latency_B, ...
        # then the QC bricks.  Separate _latency and _valid files meant
        # loading two datasets to look at one thing.
        per_cond = ["latency"]
        if n_blocks > 1:
            per_cond.append("latency_dev")
        if n_deriv == 2:
            per_cond += ["fwhm", "dispersion"]
        per_cond += ["shape_r2", "valid"]

        arrays, labels = [], []
        for b, lbl in enumerate(block_labels):
            for name in per_cond:
                arrays.append(np.asarray(shape_maps[b][name], dtype=np.float32))
                labels.append(f"{lbl}_{name}")
        _save_bucket(
            arrays,
            labels,
            f"{args.prefix}_fitbasis_shape{nii_ext}",
            to_volume=_to_volume,
            reference_img=args.input[0],
            qc=qc_bricks,
        )

        calib_tsv = f"{args.prefix}_fitbasis_shape_calibration.tsv"
        with open(calib_tsv, "w") as fh:
            fh.write("condition\ttau_s\tdispersion\tfwhm_s\tratio_t\tratio_d\tshape_r2\n")
            for lbl, calib in zip(block_labels, shape_calibs, strict=True):
                for i, tau in enumerate(calib.taus):
                    for j, disp in enumerate(calib.dispersions):
                        rd = "" if calib.ratio_d is None else f"{calib.ratio_d[i, j]:.6f}"
                        fh.write(
                            f"{lbl}\t{tau:.4f}\t{disp:.4f}\t{calib.fwhm[i, j]:.4f}\t"
                            f"{calib.ratio_t[i, j]:.6f}\t{rd}\t{calib.shape_r2[i, j]:.6f}\n"
                        )
        print(f"  Wrote {calib_tsv}{shape_calib_note}")

    # ── Cross-validation (regularization sanity check) ──────────────
    # When -cv-runs is set, run LORO across a grid of prior-weight
    # multipliers + OLS baseline and emit per-voxel held-out R² maps.
    # This is the framework that answers "is the constraint actually
    # helping" empirically per voxel — composes with every later
    # regularization change (voxel-wise λ, amplitude decoupling,
    # fracridge) since each can be A/B-tested against held-out R².
    cv_result = None

    # ── fracridge: write intrinsic per-frac CV maps ────────────────
    # fracridge's CV is baked into the fit; no separate -cv-runs pass
    # needed.  Always emit the optimal-frac map + the per-frac R² TSV
    # (and 4-D NIfTI when -cv-runs is set, to match the MVN branch's
    # outputs).
    if args.reg == "fracridge":
        from fastfuncstuff.design.flobs import FracRidgeFitResult

        assert isinstance(fit, FracRidgeFitResult)  # narrow for type checker
        fr: FracRidgeFitResult = fit
        frac_tsv = f"{args.prefix}_fitbasis_cv_r2.tsv"
        with open(frac_tsv, "w") as fh:
            fh.write("frac\tmedian_r2\tmean_r2\tmax_r2\tn_voxels_best\n")
            for j, frac_val in enumerate(fr.fracs):
                col = fr.r2_by_frac[:, j]
                n_best = int((np.argmax(fr.r2_by_frac, axis=1) == j).sum())
                fh.write(
                    f"{float(frac_val):.3f}\t{float(np.median(col)):.6f}\t"
                    f"{float(np.mean(col)):.6f}\t{float(np.max(col)):.6f}\t"
                    f"{n_best}\n"
                )
        print(f"  Wrote {frac_tsv}")

        # Per-voxel optimal frac (3-D).
        optfrac_3d = np.zeros(volume_shape, dtype=np.float32)
        if mask is not None:
            optfrac_3d[mask] = fr.optimal_fracs
        else:
            optfrac_3d = fr.optimal_fracs.reshape(volume_shape)
        of_path = f"{args.prefix}_fitbasis_optimal_frac{nii_ext}"
        with spinner(f"Writing {Path(of_path).name}"):
            save_nifti(optfrac_3d, output_path=of_path, reference_img=args.input[0])
        print(f"  Wrote {of_path}")

        if args.cv_runs:
            r2_4d_shape = volume_shape + (fr.r2_by_frac.shape[1],)
            r2_4d = np.zeros(r2_4d_shape, dtype=np.float32)
            if mask is not None:
                r2_4d[mask, :] = fr.r2_by_frac
            else:
                r2_4d = fr.r2_by_frac.reshape(r2_4d_shape)
            r2_path = f"{args.prefix}_fitbasis_cv_r2_per_frac{nii_ext}"
            with spinner(f"Writing {Path(r2_path).name}"):
                save_nifti(r2_4d, output_path=r2_path, reference_img=args.input[0])
            print(f"  Wrote {r2_path}  (4-D; volume k = held-out R² at fracs[k])")

    if args.cv_runs and args.reg == "cone":
        # cv_basis_constrained_ridge sweeps the prior weight through the
        # FIXED-mean ridge solver.  Running it under -reg cone would
        # silently cross-validate a different model than the one that
        # was fit, so refuse rather than emit a misleading map.
        # -xval-r2 does honour -reg cone (it takes the solver swap).
        print(
            "  -cv-runs is not supported with -reg cone (the weight sweep "
            "uses the fixed-mean ridge solver); use -xval-r2 instead.  "
            "Skipping CV."
        )
    elif args.cv_runs and args.reg != "fracridge":
        if n_runs < 2:
            print(f"  WARNING: -cv-runs requires ≥2 runs (got {n_runs}); skipping CV.")
        else:
            try:
                grid = [float(x) for x in str(args.cv_grid).split(",") if x.strip()]
            except ValueError:
                print(f"ERROR: -cv-grid must be comma-separated floats; got {args.cv_grid!r}")
                return 1
            print(
                f"\n  Running LORO cross-validation: leave-{args.cv_leave_n_out}-out, "
                f"{len(grid)} weights {grid} + OLS baseline"
            )
            cv_result = cv_basis_constrained_ridge(
                per_run_data=per_run_data,
                per_run_task_designs=per_run_designs,
                basis_functions=basis.basis_functions,
                prior_mean=prior_m,
                prior_cov=prior_C,
                n_blocks=n_blocks,
                polort=polort_resolved,
                weight_grid=grid,
                include_ols=True,
                leave_n_out=int(args.cv_leave_n_out),
                lambda_mode=args.lambda_mode,
                lambda_n_bins=args.lambda_n_bins,
                device=device,
                verbose=args.verb >= 1,
            )

            # TSV summary: median held-out R² per weight.
            cv_tsv = f"{args.prefix}_fitbasis_cv_r2.tsv"
            with open(cv_tsv, "w") as f:
                f.write("weight\tmedian_r2\tmean_r2\tmax_r2\tn_voxels_best\n")
                argmax = cv_result.argmax_weight_idx
                for i, w in enumerate(cv_result.weights):
                    col = cv_result.r2_per_weight[:, i]
                    n_best = int((argmax == i).sum())
                    f.write(
                        f"{w}\t{float(np.median(col)):.6f}\t"
                        f"{float(np.mean(col)):.6f}\t{float(np.max(col)):.6f}\t"
                        f"{n_best}\n"
                    )
            print(f"  Wrote {cv_tsv}")
            # Print summary table with two "best" indicators:
            #   ← best (median): the weight whose median held-out R²
            #     across all voxels is highest;
            #   ← most voxels: the weight that wins per-voxel in the
            #     largest fraction of the brain.
            #   They often disagree — the first describes the
            #   "average" voxel, the second is dominated by noise
            #   voxels where any small bias wins.
            medians_arr = np.array(
                [
                    float(np.median(cv_result.r2_per_weight[:, i]))
                    for i in range(len(cv_result.weights))
                ]
            )
            n_best_arr = np.array(
                [
                    int((cv_result.argmax_weight_idx == i).sum())
                    for i in range(len(cv_result.weights))
                ]
            )
            best_median_i = int(np.argmax(medians_arr))
            best_count_i = int(np.argmax(n_best_arr))
            print("  Held-out R² summary:")
            print(
                f"    {'weight':>8}  {'median':>9}  {'mean':>9}  {'max':>9}  {'n_best_voxels':>14}"
            )
            for i, w in enumerate(cv_result.weights):
                col = cv_result.r2_per_weight[:, i]
                tags = []
                if i == best_median_i:
                    tags.append("best median")
                if i == best_count_i:
                    tags.append("most voxels")
                tag_str = " ← " + " / ".join(tags) if tags else ""
                print(
                    f"    {w!r:>8}  {float(np.median(col)):+.4f}  "
                    f"{float(np.mean(col)):+.4f}  {float(np.max(col)):+.4f}  "
                    f"{n_best_arr[i]:>14,}{tag_str}"
                )

            # Per-voxel argmax NIfTI (3-D, one int per voxel).
            argmax_3d = np.zeros(volume_shape, dtype=np.int32)
            if mask is not None:
                argmax_3d[mask] = cv_result.argmax_weight_idx
            else:
                argmax_3d = cv_result.argmax_weight_idx.reshape(volume_shape)
            argmax_path = f"{args.prefix}_fitbasis_cv_argmax{nii_ext}"
            with spinner(f"Writing {Path(argmax_path).name}"):
                save_nifti(
                    argmax_3d.astype(np.float32),
                    output_path=argmax_path,
                    reference_img=args.input[0],
                )
            print(f"  Wrote {argmax_path}  (int → index into weights list above)")

            # Per-weight R² 4-D NIfTI (one volume per weight in the grid).
            r2_4d_shape = volume_shape + (cv_result.r2_per_weight.shape[1],)
            r2_4d = np.zeros(r2_4d_shape, dtype=np.float32)
            if mask is not None:
                r2_4d[mask, :] = cv_result.r2_per_weight
            else:
                r2_4d = cv_result.r2_per_weight.reshape(r2_4d_shape)
            r2_4d_path = f"{args.prefix}_fitbasis_cv_r2_per_weight{nii_ext}"
            with spinner(f"Writing {Path(r2_4d_path).name}"):
                save_nifti(r2_4d, output_path=r2_4d_path, reference_img=args.input[0])
            print(f"  Wrote {r2_4d_path}  (4-D; volume k = held-out R² at weights[k])")

    # ── Metadata sidecar ────────────────────────────────────────────
    metadata = {
        "tool": "ffs_fitbasis",
        "started": datetime.now().isoformat(timespec="seconds"),
        "tr": float(tr),
        "derivatives": args.derivatives,
        "hrf": args.hrf,
        "regularisation": args.reg,
        "single_trials": bool(args.single_trials),
        "n_basis": int(n_basis),
        "basis_dt_s": float(basis.dt),
        "basis_duration_s": float(basis.duration),
        "n_voxels": int(n_voxels),
        "n_runs": int(n_runs),
        "n_conditions": int(n_conditions),
        "n_blocks_fit": int(n_blocks),
        "condition_labels": list(condition_labels),
        "polort": int(polort_resolved),
        "prior_mean": prior_m.tolist(),
        "prior_cov_diag": np.diag(prior_C).tolist(),
        "sigma2_mean": float(fit.sigma2_mean),
        "effective_prior_weight": float(fit.effective_prior_weight),
        "r2_mean_constrained": float(fit.r2.mean()),
        "r2_mean_unconstrained": float(fit.r2_ols.mean()),
        "prewhiten": args.prewhiten,
    }
    if arma_ab is not None:
        metadata["arma11"] = {"a": float(arma_ab[0]), "b": float(arma_ab[1])}
    if arma_ab_per_voxel is not None:
        # 4-D NIfTI: last axis is (a, b).  Lets the user inspect the
        # ARMA parameter map directly (high-a areas usually = vascular
        # / large-vessel, low-a = parenchyma).
        ab_vol = np.zeros(volume_shape + (2,), dtype=np.float32)
        if mask is not None:
            ab_vol[mask, :] = arma_ab_per_voxel
        else:
            ab_vol = arma_ab_per_voxel.reshape(volume_shape + (2,))
        ab_path = f"{args.prefix}_fitbasis_arma_ab{nii_ext}"
        with spinner(f"Writing {Path(ab_path).name}"):
            save_nifti(ab_vol, output_path=ab_path, reference_img=args.input[0])
        print(f"  Wrote {ab_path}  (4-D: sub-brick 0=a, 1=b)")
        metadata["arma11_per_voxel"] = {
            "median_a": float(np.median(arma_ab_per_voxel[:, 0])),
            "median_b": float(np.median(arma_ab_per_voxel[:, 1])),
            "n_unique_cells": int(arma_cells.__len__() if arma_cells else 0),
        }
    if cv_result is not None:
        metadata["cv"] = {
            "weights": [str(w) for w in cv_result.weights],
            "n_splits": int(cv_result.n_splits),
            "leave_n_out": int(args.cv_leave_n_out),
            "median_r2_per_weight": [
                float(np.median(cv_result.r2_per_weight[:, i]))
                for i in range(len(cv_result.weights))
            ],
            "n_voxels_best_per_weight": [
                int((cv_result.argmax_weight_idx == i).sum()) for i in range(len(cv_result.weights))
            ],
        }
    meta_path = f"{args.prefix}_fitbasis_metadata.json"
    Path(meta_path).write_text(json.dumps(metadata, indent=2))
    print(f"  Wrote {meta_path}")

    print(f"\n{'=' * 72}")
    print(" ✓ ffs_fitbasis complete")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
