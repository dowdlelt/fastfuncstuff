#!/usr/bin/env python3
"""
ffs_fitbasis — constrained basis-set HRF fits (SPMG1/SPMG2/SPMG3/FLOBS).

This is the parametric / basis-set counterpart to [[ffs_deconvolve]].
Where ``ffs_deconvolve`` does non-parametric FIR/TENT/CSPLIN
deconvolution (one regressor per lag, no shape assumption),
``ffs_fitbasis`` fits the HRF as a small linear combination of basis
functions and optionally applies a Gaussian shape prior so the
combination can't produce nonsense HRFs.

Three things this tool can do that ``ffs_deconvolve`` cannot:

1. **SPMG2 / SPMG3** — canonical + temporal-derivative (± dispersion-
   derivative) — recovering both amplitude AND latency per
   condition/trial.
2. **FLOBS** — K=3 eigenHRFs derived from half-cosine HRF samples
   (Woolrich, Behrens, Smith 2004 TR04MW2), with an empirical MVN(m, C)
   shape prior.
3. **Single-trial fits** (``-single-trials``) — one block of basis
   regressors per trial, with the prior applied per-trial.  This is
   where unconstrained SPMG2/3 fits famously go off the rails at short
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
- ``{prefix}_fitbasis_r2.nii.gz``                — fit R²
- ``{prefix}_fitbasis_r2_unconstrained.nii.gz``  — OLS R² for comparison
- ``{prefix}_fitbasis_iresp_<cond>.nii.gz``      — reconstructed HRF (4-D)
- ``{prefix}_fitbasis_iresp_<cond>_unconstrained.nii.gz``
- ``{prefix}_fitbasis_pcweights_<cond>.nii.gz``  — coefficient maps
- ``{prefix}_fitbasis_pcweights_<cond>_unconstrained.nii.gz``
- ``{prefix}_fitbasis_amplitude_<cond>.nii.gz``  — peak amplitude for 2nd-level
- ``{prefix}_fitbasis_amplitude_<cond>_unconstrained.nii.gz``
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
        add_load_threads_arg,
        add_ortvec_arguments,
        add_verbose_arg,
        append_nuisance_blocks,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
        parse_device_arg,
        parse_input_files,
        parse_prefix,
        preflight_check,
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
    from fastfuncstuff.utils import configure_torch_backends
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_fitbasis",
        description=(
            "[BETA] Constrained basis-set HRF fitting "
            "(SPMG1/SPMG2/SPMG3/FLOBS) with optional shape prior, "
            "single-trial mode (LSA or LSS), per-voxel ARMA(1,1) "
            "prewhitening, and a filmbabe-style VB loop.  Outputs "
            "and flag set are evolving — see "
            "wiki/notes/ffs_fitbasis_changes.md for the current "
            "behaviour and open questions, especially around "
            "single-trial validation."
        ),
        formatter_class=_HelpFormatter,
    )

    req = parser.add_argument_group("Required Arguments")
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
        help="Stimulus durations (s); one per condition (or single value).",
    )
    onset_grp.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files, one per run.",
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

    model_grp = parser.add_argument_group("Model + constraint")
    model_grp.add_argument(
        "-model",
        choices=["SPMG1", "SPMG2", "SPMG3", "FLOBS"],
        default="SPMG2",
        help=(
            "Basis-set model.  SPMG1=canonical only (no shape variation), "
            "SPMG2=canonical+temporal derivative, SPMG3=+ dispersion "
            "derivative, FLOBS=K eigenHRFs from half-cosine samples + "
            "empirical MVN(m, C) prior."
        ),
    )
    model_grp.add_argument(
        "-reg",
        choices=["none", "ridge", "mvn", "mvn-shape", "cone", "fracridge"],
        default="cone",
        help=(
            "Regularisation / shape prior:\n"
            "  none      — plain OLS, no shape constraint (see how it fails);\n"
            "  ridge     — diagonal generalised ridge with hand-picked weights;\n"
            "  cone      — DEFAULT.  Scale-invariant shape prior, the "
            "faithful filmbabe form: β must lie near the ray {s·m} with "
            "tolerance ∝ s, where s is a free per-block size parameter "
            "(TR04MW2 §2.4; filmbabe_vb_flobs.cc gam_Beta).  Constrains "
            "SHAPE only — amplitude is entirely free, and negative BOLD "
            "keeps its shape and sign.  Use this;\n"
            "  mvn       — full MVN(m, C) prior with a FIXED mean.  "
            "BIASED: m comes from peak-normalised HRF samples, so this is "
            "an amplitude prior in data units — it drags every voxel "
            "toward peak ≈ 0.8 (true 3.0 → 1.37, true 0.2 → 0.72).  Kept "
            "for reproducing older runs; prefer -reg cone;\n"
            "  mvn-shape — amplitude-decoupled prior: constrain only the "
            "*shape* direction orthogonal to the prior mean; leave "
            "amplitude unconstrained.  Fixes amplitude over-shrinkage on "
            "high-SNR voxels.\n"
            "  fracridge — Rokem & Kay (2020) fractional ridge.  No HRF "
            "prior at all; instead CV picks the per-voxel fraction of "
            "||β_OLS|| to keep (-fracs grid).  Bypasses prior tuning "
            "entirely — let the data decide how much shrinkage each "
            "voxel needs."
        ),
    )
    model_grp.add_argument(
        "-single-trials",
        "-single_trials",
        dest="single_trials",
        action="store_true",
        help=(
            "Fit one block of basis regressors per TRIAL instead of per "
            "condition.  Amplitudes / shapes recovered per trial; the "
            "shape prior is applied independently to each trial's "
            "coefficient block.  GLMsingle-style output suitable for "
            "2nd-level analyses across trials."
        ),
    )

    shift_grp = parser.add_argument_group(
        "Shifted-HRF parametrisation (per-trial latency)",
        description=(
            "An alternative MODEL, not another -reg.  Instead of K basis "
            "columns per block with free betas, each block gets ONE column "
            "— the HRF shifted exactly — plus a free amplitude and a "
            "box-bounded delay reported directly in seconds.  This is the "
            "only path here that recovers per-trial latency: the SPMG2 "
            "derivative ratio measured r≈0.03 against known truth, this "
            "measures r≈0.69.  Most -reg values are meaningless under it "
            "(there are no basis coefficients left to regularise)."
        ),
    )
    shift_grp.add_argument(
        "-parametrization",
        "-parametrisation",
        dest="parametrization",
        choices=["linear", "shift"],
        default="linear",
        help=(
            "linear (default) — K basis columns per block, free betas, "
            "constrained by -reg.  shift — one exactly-shifted column per "
            "block with amplitude + bounded delay."
        ),
    )
    shift_grp.add_argument(
        "-shift-hrf",
        "-shift_hrf",
        dest="shift_hrf",
        default="canonical",
        metavar="SOURCE",
        help=(
            "Response shape for -parametrization shift.  'canonical' (SPM), "
            "'library' / 'glmsingle', or a path to a one-column text file "
            "sampled at -flobs-dt.  Any shape works — exact shifting needs "
            "no temporal derivative, so a curve from ffs_hrfopt / "
            "ffs_librarian drops straight in."
        ),
    )
    shift_grp.add_argument(
        "-shift-shapes",
        "-shift_shapes",
        dest="shift_shapes",
        default=None,
        metavar="SOURCE",
        help=(
            "Select a per-voxel response shape before fitting delays, from "
            "'library' (20 double-gammas), 'pighs' (half-cosines), "
            "'flobs' (curves drawn from the empirical FLOBS coefficient "
            "prior, so every candidate is a shape that prior calls "
            "sensible), or a path to a custom HRF library TSV (ffs_librarian "
            "output).  Overrides -shift-hrf; defaults to 'library' when "
            "-shift-shape-index is given.  Two stages on purpose: "
            "shape is chosen at zero delay, then delays are fit within "
            "each shape group — with both free at once a wrong shape can "
            "masquerade as a delay and neither means what it says.  Cheap, "
            "because the design bank depends on the shape, not the voxel."
        ),
    )
    shift_grp.add_argument(
        "-shift-shape-index",
        "-shift_shape_index",
        dest="shift_shape_index",
        default=None,
        metavar="MAP",
        help=(
            "Skip shape selection and take the per-voxel shape from an "
            "existing HRF index map — sub-brick 0 of "
            "{prefix}_hrf_index.nii.gz from ffs_hrfopt (1-based indices "
            "into the HRF library).  -shift-shapes then only supplies the "
            "curves, and must be the SAME library the indices were fit "
            "against (default 'library'; pass the ffs_librarian TSV if you "
            "ran ffs_hrfopt -hrf-library).  Preferred over letting this "
            "tool pick: ffs_hrfopt selects on cross-validated R² and can "
            "run on denoised data, where the selection here is in-sample "
            "at zero delay."
        ),
    )
    shift_grp.add_argument(
        "-shift-n-shapes",
        "-shift_n_shapes",
        dest="shift_n_shapes",
        type=int,
        default=20,
        metavar="N",
        help="Candidate count for -shift-shapes pighs / flobs (library is fixed at 20).",
    )
    shift_grp.add_argument(
        "-tau-max",
        "-tau_max",
        dest="tau_max",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help=(
            "Hard bound on the per-block delay.  Set it wide enough to "
            "contain the real latency spread: truth OUTSIDE the bound does "
            "not clamp gracefully — the amplitude solve compensates via "
            "overlapping trials and emits alternating signed amplitudes."
        ),
    )
    shift_grp.add_argument(
        "-tau-step",
        "-tau_step",
        dest="tau_step",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Delay search grid spacing.  Finer costs linearly in the gram table.",
    )
    shift_grp.add_argument(
        "-delay-prior-sd",
        "-delay_prior_sd",
        dest="delay_prior_sd",
        type=float,
        default=0.75,
        metavar="SECONDS",
        help=(
            "Std of a Gaussian prior shrinking each block's delay toward "
            "the voxel's own mean delay.  Not optional in spirit: latency "
            "is chosen by maximising fit, so at low SNR the winning delay "
            "partly fits noise and inflates amplitude (measured 1.86 vs a "
            "true 1.0 unshrunk; 1.15 shrunk).  Pass 0 to disable."
        ),
    )
    shift_grp.add_argument(
        "-shift-sweeps",
        "-shift_sweeps",
        dest="shift_sweeps",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Coordinate-descent sweeps over blocks (amplitudes re-solved "
            "after each).  Affects the delay SCALE, not just convergence: "
            "the prior's centre is re-estimated each sweep, so too few "
            "sweeps leaves delays compressed toward zero (measured 1.09 / "
            "1.15 / 1.18 s against a true 1.20 s at 2 / 4 / 8 sweeps).  "
            "Raise it if absolute delay magnitudes matter to you."
        ),
    )

    # FLOBS-specific knobs (each flag accepts hyphen + underscore forms)
    flobs_opts = parser.add_argument_group("FLOBS Options (-model FLOBS)")
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
        help="FLOBS basis duration (s).",
    )
    flobs_opts.add_argument(
        "-flobs-dt",
        "-flobs_dt",
        dest="flobs_dt",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="FLOBS basis sample spacing (s).",
    )
    flobs_opts.add_argument(
        "-flobs-seed",
        "-flobs_seed",
        dest="flobs_seed",
        type=int,
        default=42,
        help="Seed for the half-cosine sampler.",
    )

    # SPMG/ridge knobs (used when -reg ridge or -reg mvn with SPMG models)
    spmg_opts = parser.add_argument_group("SPMG Prior Options (-model SPMG*)")
    spmg_opts.add_argument(
        "-canonical-std",
        "-canonical_std",
        dest="canonical_std",
        type=float,
        default=5.0,
        help="Prior std on the canonical-amplitude coefficient (weak prior).",
    )
    spmg_opts.add_argument(
        "-derivative-std",
        "-derivative_std",
        dest="derivative_std",
        type=float,
        default=0.3,
        help="Prior std on the temporal-derivative coefficient (tight).",
    )
    spmg_opts.add_argument(
        "-dispersion-std",
        "-dispersion_std",
        dest="dispersion_std",
        type=float,
        default=0.2,
        help="Prior std on the dispersion-derivative coefficient (SPMG3).",
    )

    # Cross-validation
    cv_opts = parser.add_argument_group("Cross-validation (regularization sanity check)")
    cv_opts.add_argument(
        "-xval-r2",
        "-xval_r2",
        dest="xval_r2",
        action="store_true",
        help=(
            "Emit a per-voxel **held-out R² volume** (LORO) for the "
            "user's chosen -reg / -prior-weight config.  Writes "
            "<prefix>_fitbasis_xvalr2.nii.gz alongside the in-sample "
            "<prefix>_fitbasis_r2.nii.gz.\n"
            "  Per-condition mode: standard LORO — fit on N−1 runs, "
            "predict held-out run with the per-condition task betas.\n"
            "  Single-trial mode: train fits single-trial on N−1 runs, "
            "averages within-condition trial betas, predicts the "
            "held-out run with the per-condition design.  Catches the "
            "case where single-trial estimates collectively fail to "
            "capture the per-condition response (oscillation around "
            "the mean, runaway shrinkage, etc.).\n"
            "  Skips ARMA prewhitening / VB iteration inside the xval "
            "for speed."
        ),
    )
    cv_opts.add_argument(
        "-cv-runs",
        "-cv_runs",
        dest="cv_runs",
        action="store_true",
        help=(
            "Run LORO cross-validation over a grid of prior-weight "
            "multipliers (+ OLS baseline) and emit per-voxel held-out "
            "R² maps.  Use this to validate that the constraint is "
            "actually helping rather than over-shrinking amplitudes.  "
            "Adds two output files: <prefix>_fitbasis_cv_r2.tsv "
            "(median R² per weight) and <prefix>_fitbasis_cv_r2_per_weight"
            ".nii.gz (4-D, one volume per weight).  Also emits "
            "<prefix>_fitbasis_cv_argmax.nii.gz (3-D, per-voxel best weight)."
        ),
    )
    cv_opts.add_argument(
        "-cv-grid",
        "-cv_grid",
        dest="cv_grid",
        default="0.1,0.3,1.0,3.0,10.0",
        metavar="W1,W2,...",
        help="Comma-separated grid of prior-weight multipliers to evaluate.",
    )
    cv_opts.add_argument(
        "-cv-leave-n-out",
        "-cv_leave_n_out",
        dest="cv_leave_n_out",
        type=int,
        default=1,
        metavar="N",
        help="Number of runs left out per CV fold (1 = LORO).",
    )

    # Constraint strength
    reg_opts = parser.add_argument_group("Constraint strength")
    reg_opts.add_argument(
        "-lambda-mode",
        "-lambda_mode",
        dest="lambda_mode",
        choices=["global", "voxelwise", "auto"],
        default="auto",
        help=(
            "How λ is set per voxel.  ``auto`` (default): voxelwise "
            "when -single-trials is set (per-trial DOF is too low to "
            "tolerate a global λ at 9.4T-style SNR variation), global "
            "otherwise.  ``global``: one scalar λ for every voxel "
            "from σ²_mean across the brain mask.  ``voxelwise``: "
            "per-voxel λ_v = σ²_v from that voxel's own OLS residual "
            "variance — Bayesian-honest at the voxel scale (high-SNR "
            "voxels get less shrinkage, low-SNR more).  Implementation "
            "bins voxels by σ² quantile (-lambda-n-bins, default 20) "
            "and Cholesky-factors one matrix per bin; negligible "
            "extra cost."
        ),
    )
    reg_opts.add_argument(
        "-lambda-n-bins",
        "-lambda_n_bins",
        dest="lambda_n_bins",
        type=int,
        default=20,
        metavar="N",
        help="σ² quantile bins for -lambda-mode voxelwise.",
    )
    reg_opts.add_argument(
        "-prior-weight",
        "-prior_weight",
        dest="prior_weight",
        default="auto",
        metavar="VALUE",
        help=(
            "Strength of the shape prior.  'auto' uses the Bayesian-"
            "optimal weight σ² (estimated from an OLS pre-pass).  A "
            "float overrides as a multiplier on σ² (e.g. 2.0 = twice "
            "as strong).  Ignored when -reg none."
        ),
    )
    reg_opts.add_argument(
        "-prewhiten",
        dest="prewhiten",
        choices=["none", "arma11", "arma11-voxel"],
        default="none",
        help=(
            "Temporal-noise model.  ``none`` (default): i.i.d. white "
            "noise (standard OLS / ridge / fracridge assumption).  "
            "``arma11``: single global ARMA(1,1) (a, b) estimated "
            "from the OLS-residual mean timeseries via REML grid "
            "search; prewhiten data + design per run before fitting. "
            "``arma11-voxel``: per-voxel (a, b) via REML grid search "
            "(AFNI 3dREMLfit / FSL filmbabe-style noise model); bin "
            "voxels by grid cell, apply each cell's Cholesky factor, "
            "and run the constrained fit per cell.  Suppresses the "
            "trial-to-trial amplitude oscillation that autocorrelated "
            "noise induces.  Not yet supported with -reg fracridge."
        ),
    )
    reg_opts.add_argument(
        "-vb-iters",
        "-vb_iters",
        dest="vb_iters",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Variational-Bayes iteration count for -prewhiten arma11-voxel "
            "(FSL filmbabe-style loop, TR04MW2 §3).  ``0`` (default) runs "
            "the single-pass per-voxel ARMA path.  ``N >= 1`` re-estimates "
            "(a, b) from the current constrained-fit residuals and re-fits, "
            "stopping early when the median |Δ(a, b)| < -vb-tol or N is "
            "reached.  2-3 iterations is the FSL norm; more rarely helps."
        ),
    )
    reg_opts.add_argument(
        "-vb-tol",
        "-vb_tol",
        dest="vb_tol",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help=(
            "Convergence threshold on median |Δa| + |Δb| across voxels "
            "between successive VB iterations.  Default 0.05 (one grid "
            "step in a; tighter values rarely change the final β)."
        ),
    )
    reg_opts.add_argument(
        "-lss",
        dest="lss",
        action="store_true",
        help=(
            "Least-Squares-Separate single-trial estimator (Mumford 2012, "
            "also AFNI 3dLSS / GLMsingle 'L'-mode).  For each trial, fit a "
            "small design with cols [trial_t | rest_of_cond_c | "
            "per_other_cond] — the shape prior penalises only the "
            "current trial's K cols; the rest are nuisance.  Reduces "
            "trial-to-trial collinearity that the all-at-once LSA path "
            "suffers from with tightly-packed trials.  Requires "
            "-single-trials.  Currently supports -reg in {none, ridge, "
            "mvn, mvn-shape}.  Composes with -prewhiten arma11 (global) "
            "and -xval-r2; deferred for -prewhiten arma11-voxel, "
            "-vb-iters, -prior-from per-condition, -reg fracridge."
        ),
    )
    reg_opts.add_argument(
        "-lss-exclude",
        "-lss_exclude",
        dest="lss_exclude",
        nargs="+",
        default=[],
        metavar="COND",
        help=(
            "Conditions to exclude from LSS — they contribute their "
            "summed regressor to every LSS design but are not fit "
            "per-trial.  These conditions' main betas come from the "
            "parallel LSA pre-fit.  Useful when only certain conditions "
            "need trial-level estimates."
        ),
    )
    reg_opts.add_argument(
        "-prior-from",
        "-prior_from",
        dest="prior_from",
        choices=["none", "per-condition"],
        default="none",
        help=(
            "Empirical-Bayes source for the per-voxel prior mean.  "
            "``none`` (default): use the model's default prior mean "
            "(zero for SPMG, the empirical FLOBS mean for FLOBS).  "
            "``per-condition``: run a plain per-condition constrained "
            "fit first, take the resulting (n_vox, n_cond, K) betas, "
            "and use them as the prior mean for *every trial of that "
            "condition at that voxel* in the subsequent single-trial "
            "fit.  This is the GLMsingle / LSS shrinkage philosophy: "
            "anchor trial estimates to the condition average, let "
            "deviations only emerge where the data demands them.  "
            "Requires -single-trials; rejects -reg mvn-shape and "
            "fracridge (rotation / CV-loop incompatibilities)."
        ),
    )
    reg_opts.add_argument(
        "-vb-update-prior",
        dest="vb_update_prior",
        action="store_true",
        help=(
            "Inside the VB loop, also update the per-voxel prior "
            "precision β_size each iteration (TR04MW2 §3, gamma "
            "posterior).  Without this flag the loop only updates "
            "(a, b); with it the loop updates (a, b) AND β_size, "
            "matching the full filmbabe VB scheme.  Only meaningful "
            "with -reg in {ridge, mvn, mvn-shape} since fracridge "
            "and none have no prior precision to update."
        ),
    )
    reg_opts.add_argument(
        "-fracs",
        dest="fracs",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        metavar="LIST",
        help=(
            "Comma-separated frac grid for -reg fracridge.  Each frac "
            "is the fraction of ||β_OLS|| to retain (1.0=OLS, →0=max "
            "shrinkage).  CV picks the best per voxel.  Default 0.1…1.0 "
            "by 0.1."
        ),
    )

    # Processing
    proc = parser.add_argument_group("Processing")
    proc.add_argument(
        "-tr", type=float, default=None, help="TR in seconds; read from header if omitted."
    )
    proc.add_argument("-mask", default=None, help="Brain mask NIfTI.")
    proc.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial drift order (per run).  None → auto via run duration.",
    )
    proc.add_argument("-device", default="auto", help="Compute device: auto, cpu, cuda, mps.")

    nuis_grp = parser.add_argument_group(
        "External nuisance regressors",
        description=(
            "Motion, physio, or denoising components (e.g. ffs_denoise / "
            "ffs_denoisatorial PC timeseries).  Columns join the per-run "
            "polynomial block-diagonal, so they stay run-specific, and are "
            "projected out alongside drift before the amplitudes are read."
        ),
    )
    add_ortvec_arguments(nuis_grp)
    proc.add_argument(
        "-debug-design",
        "-debug_design",
        dest="debug_design",
        action="store_true",
        help="Print design rank/conditioning before the fit.",
    )
    add_load_threads_arg(proc)
    add_verbose_arg(proc, default=1)

    # I/O extras
    out = parser.add_argument_group("Output")
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
            "Force-save reconstructed per-block HRF as 4-D iresp NIfTI. "
            "Default: ON for per-condition fits (small), OFF for "
            "single-trial fits (typically tens of GB).  Pass this "
            "flag to force it on in single-trial mode."
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
        "-iresp-dt",
        "-iresp_dt",
        dest="iresp_dt",
        default=None,
        metavar="SECONDS",
        help=(
            "Time-axis resolution (s) for saved iresp NIfTIs.  Default: "
            "TR.  The internal basis stays at -flobs-dt (default 0.1 s) "
            "for accurate amplitude computation; only the SAVED iresp "
            "is resampled.  Coarser values shrink iresp files by the "
            "ratio (typically 10-20×) — there's no reason to ship fMRI "
            "data sampled at 10 ms on a 1.5 s TR.  Pass an explicit "
            "value like 0.5 for sub-TR HRF inspection."
        ),
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


def _build_basis(args) -> FLOBSBasis:
    """Construct the basis FLOBSBasis container from the chosen model."""
    if args.model == "FLOBS":
        return generate_flobs_basis(
            n_basis=args.flobs_n_basis,
            n_samples=args.flobs_n_samples,
            duration=args.flobs_window,
            dt=args.flobs_dt,
            seed=args.flobs_seed,
        )
    n_basis_map = {"SPMG1": 1, "SPMG2": 2, "SPMG3": 3}
    return generate_spmg_basis(
        n_basis=n_basis_map[args.model],
        duration=args.flobs_window,
        dt=args.flobs_dt,
    )


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
        raise FileNotFoundError(f"-shift-shape-index {path!r} does not exist.")
    vol = np.asanyarray(load_nifti(str(p)).dataobj)
    if vol.ndim == 4:
        vol = vol[..., 0]
    elif vol.ndim != 3:
        raise ValueError(
            f"-shift-shape-index {path!r}: expected a 3-D or 4-D volume, got {vol.shape}."
        )
    if tuple(vol.shape) != tuple(volume_shape):
        raise ValueError(
            f"-shift-shape-index {path!r} has grid {vol.shape} but the input "
            f"data has {tuple(volume_shape)}.  The index map must be on the "
            "same grid as -input (same ffs_hrfopt run, or resampled first)."
        )
    idx = np.rint(np.asarray(vol, dtype=np.float64)).astype(np.int64)
    idx = idx[mask] if mask is not None else idx.reshape(-1)
    if idx.size != n_voxels:
        raise ValueError(
            f"-shift-shape-index {path!r} yielded {idx.size} voxels but the "
            f"data has {n_voxels}.  Use the same -mask as the ffs_hrfopt run."
        )
    # 1-based (hrfopt convention) → 0-based; unset voxels (0) become shape 0.
    out = np.clip(idx - 1, 0, None)
    too_big = out >= n_shapes
    if too_big.any():
        raise ValueError(
            f"-shift-shape-index {path!r} contains index "
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
    tr: float,
    polort: int,
    volume_shape,
    mask,
    device: torch.device,
    nii_ext: str,
    extra_regs_per_run: list[torch.Tensor] | None = None,
) -> int:
    """-parametrization shift: per-block amplitude + bounded latency.

    Fits on the concatenated timeseries with block-diagonal per-run drift,
    then optionally runs the LORO held-out validator.  Emits amplitude and
    delay maps rather than basis-coefficient maps — there are no basis
    coefficients in this model.
    """
    from fastfuncstuff.cli_utils import spinner
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
            f"           from -shift-hrf (currently {args.shift_hrf!r}); the delay "
            f"prior from\n"
            f"           -delay-prior-sd.  Use -parametrization linear if you "
            f"wanted {args.model}/{args.reg}."
        )

    print("\n  Parametrisation: shift (amplitude + bounded latency per block)")
    shapes = None
    shape_labels: list[str] = []
    imported_index: np.ndarray | None = None
    # An imported index map only says WHICH curve each voxel took; the curves
    # themselves still have to be rebuilt here, and the default library is
    # what ffs_hrfopt uses unless the user pointed it elsewhere.
    shape_source = args.shift_shapes or ("library" if args.shift_shape_index else None)
    if shape_source:
        shapes, shape_labels = build_shape_library(
            shape_source,
            args.flobs_dt,
            args.flobs_window,
            n_hrfs=args.shift_n_shapes,
            drop_empty=args.shift_shape_index is None,
        )
        hrf = shapes[0]
        if args.shift_shape_index:
            try:
                imported_index = _load_shape_index_map(
                    args.shift_shape_index,
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
                f"assignment IMPORTED from {args.shift_shape_index} "
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
        hrf = _resolve_shift_hrf(args.shift_hrf, args.flobs_dt, args.flobs_window)
        print(
            f"  HRF source: {args.shift_hrf}  ({hrf.size} samples @ "
            f"{args.flobs_dt}s, peak-normalised)  [one shape for all voxels]"
        )
    print(
        f"  Delay search: ±{args.tau_max}s step {args.tau_step}s"
        f"{f', prior sd {args.delay_prior_sd}s' if args.delay_prior_sd else ', no prior'}"
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
            device=device,
            verbose=True,
        )
    else:
        fit = fit_shifted_hrf(
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
            device=device,
            verbose=True,
        )
    del Z
    if device.type == "cuda":
        torch.cuda.empty_cache()

    nx, ny, nz = volume_shape

    def _to_volume(masked: np.ndarray) -> np.ndarray:
        out_shape = (nx, ny, nz) + tuple(masked.shape[1:])
        out = np.zeros(out_shape, dtype=np.float32)
        if mask is not None:
            out[mask, ...] = masked
        else:
            out = masked.reshape(out_shape)
        return out

    for arr, name in (
        (fit.r2, "r2"),
        (fit.r2_fixed, "r2_tau0"),
        (fit.r2_total, "r2_incl_drift"),
    ):
        path = f"{args.prefix}_fitbasis_shift_{name}{nii_ext}"
        with spinner(f"Writing {Path(path).name}"):
            save_nifti(
                _to_volume(arr[:, None]).squeeze(-1),
                output_path=path,
                reference_img=args.input[0],
            )
        print(f"  Wrote {path}")
    print(
        "  NOTE: _r2 is TASK variance / NON-DRIFT variance — the nuisance is\n"
        "        removed from both terms, so drift is not credited to the model.\n"
        "        _r2_incl_drift uses raw total variance (what most tools print)\n"
        "        and is inflated by the polynomial model: with -polort 4 over 10\n"
        "        runs that is 50 nuisance columns absorbing real variance.\n"
        "        Neither is evidence of task response on its own — n_blocks free\n"
        "        amplitudes buy in-sample fit too.  For a task-responsiveness\n"
        "        map use the held-out _xvalr2 from -xval-r2; for latency, use\n"
        "        _xvalr2_delay_gain.  r2 minus r2_tau0 is in-sample and proves\n"
        "        nothing."
    )

    if shape_index is not None and shapes is not None:
        path = f"{args.prefix}_fitbasis_shift_shape_index{nii_ext}"
        with spinner(f"Writing {Path(path).name}"):
            save_nifti(
                _to_volume(shape_index.astype(np.float32)[:, None]).squeeze(-1),
                output_path=path,
                reference_img=args.input[0],
            )
        print(f"  Wrote {path}  (int index into the curves TSV below)")
        curves_path = f"{args.prefix}_fitbasis_shift_shapes.tsv"
        np.savetxt(
            curves_path,
            shapes.T,
            fmt="%.10g",
            delimiter="\t",
            header="\t".join(shape_labels),
            comments="",
        )
        print(f"  Wrote {curves_path}  (columns = candidates @ {args.flobs_dt}s)")

        # The selected shape absorbs part of the voxel's mean timing, so the
        # delay map alone understates it.  Emit the two pieces that make the
        # confound decomposable rather than destructive: time-to-peak of the
        # chosen curve, and TTP + mean delay, which recovers the true mean
        # timing (verified to sum exactly at good SNR).
        ttp = shape_time_to_peak(shapes, args.flobs_dt)[shape_index]
        mean_timing = ttp + fit.delays.mean(axis=1)
        for arr, name, note in (
            (ttp, "shape_ttp", "time-to-peak of each voxel's selected curve (s)"),
            (mean_timing, "mean_timing", "TTP + mean delay = mean response timing (s)"),
        ):
            path = f"{args.prefix}_fitbasis_shift_{name}{nii_ext}"
            save_nifti(
                _to_volume(arr.astype(np.float32)[:, None]).squeeze(-1),
                output_path=path,
                reference_img=args.input[0],
            )
            print(f"  Wrote {path}  ({note})")

    cond_idx: dict[str, list[int]] = {}
    for b, c in enumerate(block_cond):
        cond_idx.setdefault(condition_labels[c], []).append(b)
    # Per-trial deviation from each voxel's OWN mean delay.  This is the
    # quantity that survives the shape/delay confound: the shape is fixed
    # within a voxel across trials, so it cannot absorb trial-to-trial
    # variation.  Verified to track true jitter at r=0.61 (noise sd 0.3)
    # even when the voxel's absolute delay was absorbed into its shape.
    delay_dev = fit.delays - fit.delays.mean(axis=1, keepdims=True)
    for cond, idxs in cond_idx.items():
        for arr, name in (
            (fit.amplitudes, "amplitude"),
            (fit.delays, "delay"),
            (delay_dev, "delay_dev"),
        ):
            vol = _to_volume(arr[:, idxs])
            path = f"{args.prefix}_fitbasis_shift_{name}_{cond}{nii_ext}"
            save_nifti(vol, output_path=path, reference_img=args.input[0])
            print(f"  Wrote {path}  (n_blocks={len(idxs)})")

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
                    "        _xvalr2 is optimistic by that much; _xvalr2_delay_gain\n"
                    "        is not, since both scored models carry the same shape\n"
                    "        and only the delays are refit per fold."
                )
            r2_shift, r2_tau0 = xval_shifted_hrf(
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
                n_sweeps=args.shift_sweeps,
                leave_n_out=args.cv_leave_n_out,
                device=device,
                verbose=True,
            )
            for arr, name in (
                (r2_shift, "xvalr2"),
                (r2_tau0, "xvalr2_tau0"),
                (r2_shift - r2_tau0, "xvalr2_delay_gain"),
            ):
                path = f"{args.prefix}_fitbasis_shift_{name}{nii_ext}"
                save_nifti(
                    _to_volume(arr[:, None]).squeeze(-1),
                    output_path=path,
                    reference_img=args.input[0],
                )
                print(f"  Wrote {path}")
            print(
                "  _xvalr2_delay_gain is the map that answers 'is the delay real':\n"
                "  positive = the estimated delays predicted held-out runs better\n"
                "  than pinning them to zero; ≤0 = they did not."
            )

    meta = {
        "tool": "ffs_fitbasis",
        "parametrization": "shift",
        "started": datetime.now().isoformat(timespec="seconds"),
        "tr": float(tr),
        "hrf_source": (args.shift_hrf if shapes is None else f"per-voxel:{shape_source}"),
        "n_shape_candidates": (0 if shapes is None else int(shapes.shape[0])),
        "shape_index_source": (
            "imported" if imported_index is not None else ("fitted" if shapes is not None else None)
        ),
        "shape_index_map": args.shift_shape_index,
        "ortvec_columns_per_run": (
            0 if extra_regs_per_run is None else int(extra_regs_per_run[0].shape[1])
        ),
        "tau_max": float(args.tau_max),
        "tau_step": float(args.tau_step),
        "delay_prior_sd": (None if args.delay_prior_sd is None else float(args.delay_prior_sd)),
        "n_sweeps": int(fit.n_sweeps),
        "single_trials": bool(args.single_trials),
        "n_blocks": len(block_onsets),
        "condition_labels": list(condition_labels),
        "polort": int(polort),
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

    if getattr(args, "delay_prior_sd", None) is not None and args.delay_prior_sd <= 0:
        args.delay_prior_sd = None

    if args.shift_shape_index and args.parametrization != "shift":
        print(
            "ERROR: -shift-shape-index only applies to -parametrization shift. "
            "The linear parametrisation has no per-voxel HRF — the basis set "
            "IS its shape model, so an index map has nothing to select.",
            file=sys.stderr,
        )
        return 1

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    print("=" * 72)
    print(" ffs_fitbasis — constrained basis-set HRF fits")
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
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    print(f"  Compute device: {device}")

    load_result = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=None,
        do_scale=True,
        device=device,
        force_cpu=True,
        dry_run=False,
        verbose=True,
        load_threads=args.load_threads,
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

    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, tr, threshold=args.round_onsets)
        print(f"  Rounded onsets to nearest TR (threshold={args.round_onsets:.2f}).")

    print(f"  Data: {n_voxels:,} voxels × {n_timepoints} TR ({n_runs} runs, TR={tr}s)")

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
            f"\n  Model: {args.model}    Regularisation: {args.reg}    "
            f"Single-trials: {args.single_trials}"
        )
    basis = _build_basis(args)
    n_basis = basis.basis_functions.shape[0]
    prior_m, prior_C = _build_prior(
        model=args.model,
        reg=args.reg,
        basis=basis,
        canonical_std=args.canonical_std,
        derivative_std=args.derivative_std,
        dispersion_std=args.dispersion_std,
    )
    pw = _resolve_prior_weight_arg(args.prior_weight, args.reg)
    if not _shift_mode:
        print(
            f"  Basis: {n_basis} fns × {basis.basis_functions.shape[1]} samples "
            f"(dt={basis.dt}s, window={basis.duration:.1f}s)"
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
            args, run_starts=list(run_starts), n_timepoints=n_timepoints, verbose=True
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
                    trial_num_global += 1
    else:
        for cond_idx, cond_label in enumerate(condition_labels):
            block_labels.append(cond_label)
            block_onsets_per_run.append(all_onsets[cond_idx])

    n_blocks = len(block_labels)
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
            tr=tr,
            polort=polort_resolved,
            volume_shape=volume_shape,
            mask=mask,
            device=device,
            nii_ext=nii_ext,
            extra_regs_per_run=extra_regs_per_run,
        )

    # Build per-run design with K basis cols per block
    per_run_designs: list[torch.Tensor] = []
    for r in range(n_runs):
        block_designs = []
        for b_idx in range(n_blocks):
            # Use the helper's onsets-per-run path: one-condition view.
            bd = build_pc_basis_design_per_run(
                onsets_per_run=[block_onsets_per_run[b_idx][r]],
                pcs=basis.basis_functions,
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
                f"{args.model} / {args.reg})…"
            )
            pc_designs: list[torch.Tensor] = []
            for r in range(n_runs):
                cond_blocks = []
                for c in range(len(condition_labels)):
                    bd = build_pc_basis_design_per_run(
                        onsets_per_run=[all_onsets[c][r]],
                        pcs=basis.basis_functions,
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
    print(f"\n  Fitting ({args.model} × {args.reg}) …")
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
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  constrained: {fit.r2.mean():.3f}")

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
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  constrained: {fit.r2.mean():.3f}")
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
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  constrained: {fit.r2.mean():.3f}")

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
    basis_iresp = np.stack(
        [np.interp(dst_times, src_times, basis.basis_functions[k]) for k in range(n_basis)],
        axis=0,
    ).astype(np.float64)  # (K, n_t_iresp)
    print(
        f"  iresp / amplitude grid: dt={iresp_dt:.3f}s × {n_t_iresp} "
        f"samples (basis stored at {basis.dt:.3f}s × {n_t_basis})"
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
    print(f"  Amplitude/iresp chunking: {chunk_size:,} voxels per chunk (n_blocks={n_blocks})")

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
        hrfs_chunk = task_betas[start:end] @ basis_iresp
        hrfs_ols_chunk = task_betas_ols[start:end] @ basis_iresp
        amplitude[start:end] = _signed_peak(hrfs_chunk).astype(np.float32)
        amplitude_ols[start:end] = _signed_peak(hrfs_ols_chunk).astype(np.float32)
        if iresp_buf is not None:
            iresp_buf[start:end] = hrfs_chunk.astype(np.float32)
            iresp_buf_ols[start:end] = hrfs_ols_chunk.astype(np.float32)

    # Basis TSV (shared)
    basis_path = f"{args.prefix}_fitbasis_basis.tsv"
    np.savetxt(basis_path, basis.basis_functions.T, fmt="%.10g", delimiter="\t")
    print(f"  Wrote {basis_path}")

    # R² volumes (constrained + unconstrained)
    from fastfuncstuff.cli_utils import spinner

    with spinner("Writing R² maps"):
        for arr, sfx in ((fit.r2, ""), (fit.r2_ols, "_unconstrained")):
            path = f"{args.prefix}_fitbasis_r2{sfx}{nii_ext}"
            save_nifti(
                _to_volume(arr[:, None]).squeeze(-1),
                output_path=path,
                reference_img=args.input[0],
            )
            print(f"  Wrote {path}")

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
            xval_r2 = compute_xval_r2_per_voxel(
                per_run_data=per_run_data_orig,
                all_onsets=all_onsets,
                condition_labels=list(condition_labels),
                basis_functions=basis.basis_functions,
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
            for amps, sfx in ((amplitude, ""), (amplitude_ols, "_unconstrained")):
                amp_stack = amps[:, idxs]  # (n_vox, n_trials_for_cond)
                amp_vol = _to_volume(amp_stack)  # (nx, ny, nz, n_trials)
                path = f"{args.prefix}_fitbasis_amplitude_{cond}{sfx}{nii_ext}"
                save_nifti(amp_vol, output_path=path, reference_img=args.input[0])
                print(f"  Wrote {path}  (n_trials={len(idxs)})")

        if save_full_iresp and iresp_buf is not None:
            # Per-trial iresp 4-D (time axis = HRF basis-dt).  Only
            # available when the iresp buffer fit in memory above.
            for hrfs_arr, sfx in ((iresp_buf, ""), (iresp_buf_ols, "_unconstrained")):
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
            print(f"  Wrote {n_blocks} × 2 iresp files (constrained + unconstrained).")

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
            for arr, sfx in ((stack, ""), (stack_ols, "_unconstrained")):
                for b in range(n_basis):
                    # (n_vox, n_trials) → (nx, ny, nz, n_trials)
                    vol_4d = _to_volume(arr[:, :, b])
                    path = f"{args.prefix}_fitbasis_basisweight{b + 1:02d}_{cond}{sfx}{nii_ext}"
                    save_nifti(vol_4d, output_path=path, reference_img=args.input[0])
            print(f"  Wrote basisweight01..{n_basis:02d} for {cond} (n_trials={len(idxs)})")

    else:
        # Per-condition outputs (the simple case)
        if save_full_iresp and iresp_buf is not None:
            for hrfs_arr, sfx in ((iresp_buf, ""), (iresp_buf_ols, "_unconstrained")):
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
            print(f"  Wrote {n_blocks} × 2 iresp files (constrained + unconstrained).")

        for b_idx, lbl in enumerate(block_labels):
            for tbetas, amps, sfx in (
                (task_betas, amplitude, ""),
                (task_betas_ols, amplitude_ols, "_unconstrained"),
            ):
                w = _to_volume(tbetas[:, b_idx, :])
                a = _to_volume(amps[:, b_idx][:, None]).squeeze(-1)
                w_path = f"{args.prefix}_fitbasis_basisweights_{lbl}{sfx}{nii_ext}"
                a_path = f"{args.prefix}_fitbasis_amplitude_{lbl}{sfx}{nii_ext}"
                save_nifti(w, output_path=w_path, reference_img=args.input[0])
                save_nifti(a, output_path=a_path, reference_img=args.input[0])
                print(f"  Wrote {w_path}")
                print(f"  Wrote {a_path}")

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
        "model": args.model,
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
