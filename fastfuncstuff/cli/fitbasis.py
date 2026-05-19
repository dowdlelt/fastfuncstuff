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
        add_verbose_arg,
        load_and_preprocess_runs,
        parse_device_arg,
        parse_input_files,
        parse_prefix,
        preflight_check,
    )
    from fastfuncstuff.design.builder import (
        pack_for_shared_task_glm,
        parse_afni_timing_file,
        parse_durations,
    )
    from fastfuncstuff.design.flobs import (
        FLOBSBasis,
        ARMAWhitenCell,
        FLOBSFitResult,
        bin_and_whiten_arma11,
        compute_per_voxel_residuals,
        compute_vb_block_trace,
        compute_xval_r2_per_voxel,
        cv_basis_constrained_ridge,
        decouple_amplitude_prior,
        estimate_and_apply_arma11_prewhitening,
        estimate_arma11_per_voxel,
        fit_basis_constrained_ridge,
        fit_basis_fracridge,
        fit_basis_lss,
        vb_update_beta_size,
        flobs_prior,
        generate_flobs_basis,
        generate_spmg_basis,
        ridge_prior,
        spmg_prior,
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
            "Constrained basis-set HRF fitting (SPMG1/SPMG2/SPMG3/FLOBS) "
            "with optional shape prior and single-trial mode."
        ),
        formatter_class=_HelpFormatter,
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument("-input", nargs="+", required=True,
                     help="Input fMRI run files.")
    req.add_argument("-prefix", required=True,
                     help="Output prefix (e.g. out/sub01_fb).")

    onset_grp = parser.add_argument_group("Event timing (choose one)")
    onset_grp.add_argument("-onsets", nargs="+", default=None,
                           help="AFNI-format onset files, one per condition.")
    onset_grp.add_argument("-durations", nargs="+", default=None,
                           help="Stimulus durations (s); one per condition (or single value).")
    onset_grp.add_argument("-events", nargs="+", default=None, metavar="TSV",
                           help="BIDS *_events.tsv files, one per run.")
    # Each event-related flag accepts both hyphen and underscore forms
    # (``-event-cols`` and ``-event_cols``) so muscle memory from AFNI /
    # older ffs_* tools works.  argparse dispatches both to the same
    # ``args.event_cols`` attribute via the canonical dest.
    onset_grp.add_argument("-event-ignore", "-event_ignore",
                           dest="event_ignore",
                           nargs="+", default=None, metavar="LABEL",
                           help="trial_type values to drop from BIDS events.")
    onset_grp.add_argument("-event-cols", "-event_cols",
                           dest="event_cols",
                           nargs=3, default=None,
                           metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
                           help="Override BIDS column names (default: onset duration trial_type).")
    onset_grp.add_argument("-round-onsets", "-round_onsets",
                           dest="round_onsets",
                           nargs="?", const=0.7, type=float,
                           default=None, metavar="THRESHOLD",
                           help="Snap onsets to nearest TR (default threshold 0.7).")
    onset_grp.add_argument("-round-durations", "-round_durations",
                           dest="round_durations",
                           type=int, default=None, metavar="PLACES",
                           help="Round event durations to N decimals before grouping.")

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
        choices=["none", "ridge", "mvn", "mvn-shape", "fracridge"],
        default="mvn",
        help=(
            "Regularisation / shape prior:\n"
            "  none      — plain OLS, no shape constraint (see how it fails);\n"
            "  ridge     — diagonal generalised ridge with hand-picked weights;\n"
            "  mvn       — full MVN(m, C) prior (empirical from half-cosine "
            "samples for FLOBS; spmg_prior defaults for SPMG).  Closest to "
            "the VB shape-prior path without full VB;\n"
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
        "-single-trials", "-single_trials",
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

    # FLOBS-specific knobs (each flag accepts hyphen + underscore forms)
    flobs_opts = parser.add_argument_group("FLOBS Options (-model FLOBS)")
    flobs_opts.add_argument("-flobs-n-basis", "-flobs_n_basis",
                            dest="flobs_n_basis",
                            type=int, default=3, metavar="K",
                            help="Number of FLOBS eigenHRFs (TR04MW2 used 3).")
    flobs_opts.add_argument("-flobs-n-samples", "-flobs_n_samples",
                            dest="flobs_n_samples",
                            type=int, default=1000, metavar="N",
                            help="Number of half-cosine HRF samples for the basis SVD.")
    flobs_opts.add_argument("-flobs-window", "-flobs_window",
                            dest="flobs_window",
                            type=float, default=32.0,
                            metavar="SECONDS", help="FLOBS basis duration (s).")
    flobs_opts.add_argument("-flobs-dt", "-flobs_dt",
                            dest="flobs_dt",
                            type=float, default=0.1,
                            metavar="SECONDS", help="FLOBS basis sample spacing (s).")
    flobs_opts.add_argument("-flobs-seed", "-flobs_seed",
                            dest="flobs_seed",
                            type=int, default=42,
                            help="Seed for the half-cosine sampler.")

    # SPMG/ridge knobs (used when -reg ridge or -reg mvn with SPMG models)
    spmg_opts = parser.add_argument_group("SPMG Prior Options (-model SPMG*)")
    spmg_opts.add_argument("-canonical-std", "-canonical_std",
                           dest="canonical_std",
                           type=float, default=5.0,
                           help="Prior std on the canonical-amplitude coefficient (weak prior).")
    spmg_opts.add_argument("-derivative-std", "-derivative_std",
                           dest="derivative_std",
                           type=float, default=0.3,
                           help="Prior std on the temporal-derivative coefficient (tight).")
    spmg_opts.add_argument("-dispersion-std", "-dispersion_std",
                           dest="dispersion_std",
                           type=float, default=0.2,
                           help="Prior std on the dispersion-derivative coefficient (SPMG3).")

    # Cross-validation
    cv_opts = parser.add_argument_group("Cross-validation (regularization sanity check)")
    cv_opts.add_argument(
        "-xval-r2", "-xval_r2",
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
        "-cv-runs", "-cv_runs",
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
        "-cv-grid", "-cv_grid",
        dest="cv_grid",
        default="0.1,0.3,1.0,3.0,10.0",
        metavar="W1,W2,...",
        help="Comma-separated grid of prior-weight multipliers to evaluate.",
    )
    cv_opts.add_argument(
        "-cv-leave-n-out", "-cv_leave_n_out",
        dest="cv_leave_n_out",
        type=int, default=1, metavar="N",
        help="Number of runs left out per CV fold (1 = LORO).",
    )

    # Constraint strength
    reg_opts = parser.add_argument_group("Constraint strength")
    reg_opts.add_argument(
        "-lambda-mode", "-lambda_mode",
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
        "-lambda-n-bins", "-lambda_n_bins",
        dest="lambda_n_bins",
        type=int, default=20, metavar="N",
        help="σ² quantile bins for -lambda-mode voxelwise.",
    )
    reg_opts.add_argument(
        "-prior-weight", "-prior_weight",
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
        "-vb-iters", "-vb_iters",
        dest="vb_iters",
        type=int, default=0, metavar="N",
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
        "-vb-tol", "-vb_tol",
        dest="vb_tol",
        type=float, default=0.05, metavar="FLOAT",
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
        "-lss-exclude", "-lss_exclude",
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
        "-prior-from", "-prior_from",
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
    proc.add_argument("-tr", type=float, default=None,
                      help="TR in seconds; read from header if omitted.")
    proc.add_argument("-mask", default=None, help="Brain mask NIfTI.")
    proc.add_argument("-polort", type=int, default=None,
                      help="Polynomial drift order (per run).  None → auto via run duration.")
    proc.add_argument("-device", default="auto",
                      help="Compute device: auto, cpu, cuda, mps.")
    proc.add_argument("-debug-design", "-debug_design",
                      dest="debug_design",
                      action="store_true",
                      help="Print design rank/conditioning before the fit.")
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
        "-save-iresp", "-save_iresp",
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
        "-no-iresp", "-no_iresp",
        dest="save_iresp_off",
        action="store_true",
        default=False,
        help="Force iresp save off (only PC weights + amplitude maps emitted).",
    )
    out.add_argument(
        "-iresp-dt", "-iresp_dt",
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
    if reg in ("none", "fracridge"):
        # Returned but ignored downstream (fracridge uses its own
        # CV-tuned shrinkage; no MVN prior involved).
        return np.zeros(n_basis), np.eye(n_basis)

    # First build the base (m, C); then optionally decouple amplitude.
    if model == "FLOBS":
        if reg in ("mvn", "mvn-shape"):
            base_m, base_C = flobs_prior(basis)
        else:                                              # ridge
            std = float(np.sqrt(np.median(np.diag(basis.C))))
            base_m, base_C = ridge_prior(n_basis, coefficient_std=max(std, 1e-3))
    elif model == "SPMG1":
        base_m, base_C = ridge_prior(1, coefficient_std=canonical_std)
    elif model == "SPMG2":
        base_m, base_C = spmg_prior(canonical_std=canonical_std,
                                    derivative_std=derivative_std)
    elif model == "SPMG3":
        base_m, base_C = spmg_prior(canonical_std=canonical_std,
                                    derivative_std=derivative_std,
                                    dispersion_std=dispersion_std)
    else:
        raise ValueError(f"Unknown model {model}")

    if reg == "mvn-shape":
        # Need a non-zero mean to define the amplitude direction.
        if np.linalg.norm(base_m) < 1e-12:
            # SPMG default has zero mean; pick the canonical-amplitude
            # axis (first basis function) as the amplitude direction.
            base_m = np.zeros(n_basis, dtype=np.float64)
            base_m[0] = float(canonical_std)              # arbitrary positive direction
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


def main() -> int:
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()

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
    if has_events:
        from fastfuncstuff.cli_utils import clean_condition_labels  # noqa: F401
        from fastfuncstuff.design.bids_events import parse_bids_events
        if len(args.events) != n_runs:
            print(f"ERROR: -events: {len(args.events)} files but {n_runs} input runs.")
            return 1
        event_cols = tuple(args.event_cols) if args.event_cols else None
        all_onsets, durations, condition_labels = parse_bids_events(
            event_files=args.events,
            event_ignore=args.event_ignore,
            event_cols=event_cols,
            round_durations=args.round_durations,
        )
    else:
        from fastfuncstuff.cli_utils import clean_condition_labels
        onset_files = args.onsets
        n_conditions = len(onset_files)
        condition_labels = clean_condition_labels([Path(f).stem for f in onset_files])
        for f in onset_files:
            if not Path(f).exists():
                print(f"ERROR: Onset file not found: {f}")
                return 1
        durations = parse_durations(args.durations, n_conditions, condition_labels)
        if args.round_durations is not None:
            durations = [round(d, args.round_durations) for d in durations]
        all_onsets = [parse_afni_timing_file(f) for f in onset_files]
        for i, cond_runs in enumerate(all_onsets):
            if len(cond_runs) != n_runs:
                print(
                    f"ERROR: Onset file {onset_files[i]} has "
                    f"{len(cond_runs)} runs, but {n_runs} input runs."
                )
                return 1

    n_conditions = len(condition_labels)
    print(f"  {n_conditions} conditions: {condition_labels}")

    preflight_check(input_files=input_files,
                    onset_files=args.onsets if has_onsets else None,
                    ortvec_files=None)

    # ── Load data ──────────────────────────────────────────────────
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    print(f"  Compute device: {device}")

    load_result = load_and_preprocess_runs(
        input_files=input_files, tr=args.tr, mask_file=args.mask,
        blur_fwhm=None, do_scale=True, device=device, force_cpu=True,
        dry_run=False, verbose=True,
    )
    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
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
            args.lambda_mode = "global"     # placeholder; unused downstream
        else:
            args.lambda_mode = "voxelwise" if args.single_trials else "global"
            print(f"  Lambda mode auto → {args.lambda_mode} "
                  f"({'single-trials' if args.single_trials else 'per-condition'})")

    print(f"\n  Model: {args.model}    Regularisation: {args.reg}    "
          f"Single-trials: {args.single_trials}")
    basis = _build_basis(args)
    n_basis = basis.basis_functions.shape[0]
    prior_m, prior_C = _build_prior(
        model=args.model, reg=args.reg, basis=basis,
        canonical_std=args.canonical_std,
        derivative_std=args.derivative_std,
        dispersion_std=args.dispersion_std,
    )
    pw = _resolve_prior_weight_arg(args.prior_weight, args.reg)
    print(f"  Basis: {n_basis} fns × {basis.basis_functions.shape[1]} samples "
          f"(dt={basis.dt}s, window={basis.duration:.1f}s)")
    # Only print prior info when the prior is actually applied.  With
    # -reg none, m/C/λ are all zeros / unused — printing them is noise
    # that confuses users about whether the prior is active.
    if args.reg != "none":
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

    # Auto-detect TR-lock vs sub-TR for the basis convolution path.
    all_onset_times = [
        float(t) for cond_runs in all_onsets for run_onsets in cond_runs
        for t in (run_onsets.tolist() if run_onsets.size else [])
    ]
    from fastfuncstuff.design.matrices import is_tr_locked
    basis_mode = "FIR" if (all_onset_times and is_tr_locked(all_onset_times, tr, threshold=0.1)) else "TENT"
    print(f"  Onset basis-convolution mode: {basis_mode} "
          f"({'TR-locked' if basis_mode == 'FIR' else 'sub-TR onsets'})")

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
                        np.array([t_in_run]) if rr == r else np.array([])
                        for rr in range(n_runs)
                    ]
                    block_labels.append(
                        f"{cond_label}_trial{trial_num_global:03d}_run{r + 1}"
                    )
                    block_onsets_per_run.append(per_run)
                    trial_num_global += 1
    else:
        for cond_idx, cond_label in enumerate(condition_labels):
            block_labels.append(cond_label)
            block_onsets_per_run.append(all_onsets[cond_idx])

    n_blocks = len(block_labels)
    print(f"  Blocks to fit: {n_blocks}  ({'one per trial' if args.single_trials else 'one per condition'})")

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
        data[:, run_starts_ext[r]:run_starts_ext[r + 1]].clone().detach().float()
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
                    f"{lbl}#PC{b}" for lbl in condition_labels
                    for b in range(n_basis)
                ],
                device=device,
            )
            pc_task_design = packed_pc.design_concat[:, :n_pc_task_cols]
            pc_nuisance = (
                packed_pc.design_concat[:, n_pc_task_cols:]
                if packed_pc.design_concat.shape[1] > n_pc_task_cols else None
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
                -1, n_cond, n_basis,
            )
            print(
                f"  ✓ Per-condition pre-fit complete.  "
                f"R² mean: OLS={pc_fit.r2_ols.mean():.3f}, "
                f"constrained={pc_fit.r2.mean():.3f}"
            )

            # Map per-condition → per-trial prior mean using block_labels.
            n_vox_masked_pc = cond_betas.shape[0]
            empirical_prior_mean_full = np.zeros(
                (n_vox_masked_pc, n_blocks * n_basis), dtype=np.float32,
            )
            cond_to_idx = {c: i for i, c in enumerate(condition_labels)}
            for b_idx, label in enumerate(block_labels):
                cond_label = label.split("_trial")[0]
                ci = cond_to_idx[cond_label]
                empirical_prior_mean_full[
                    :, b_idx * n_basis:(b_idx + 1) * n_basis
                ] = cond_betas[:, ci, :]

            # Shift per_run_data: y → y − X · m_v in place per run.
            m_t = torch.from_numpy(empirical_prior_mean_full).to(
                device=device, dtype=torch.float32,
            )                                          # (n_vox, n_blocks*n_basis)
            for r in range(n_runs):
                X_r = per_run_designs[r].to(device).float()   # (n_tp_r, n_task)
                shift_r = m_t @ X_r.T                          # (n_vox, n_tp_r)
                per_run_data[r] = (
                    per_run_data[r].to(device).float() - shift_r
                )
            print(
                f"  ✓ Data shifted; single-trial fit now solves "
                f"β_centered = β_trial − β_cond per voxel."
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
    if args.prewhiten == "arma11":
        per_run_data, per_run_designs, a_opt, b_opt = (
            estimate_and_apply_arma11_prewhitening(
                per_run_data=per_run_data,
                per_run_task_designs=per_run_designs,
                polort=polort_resolved,
                device=device,
                verbose=True,
            )
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
    task_column_labels = [
        f"{lbl}#PC{b}" for lbl in block_labels for b in range(n_basis)
    ]
    if arma_cells is None:
        packed = pack_for_shared_task_glm(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            polort=polort_resolved,
            task_column_labels=task_column_labels,
            device=device,
        )
        n_task_cols = packed.n_task_cols
        print(f"  Design: {packed.design_concat.shape}  "
              f"({n_task_cols} task + "
              f"{packed.design_concat.shape[1] - n_task_cols} nuisance)")

        task_design = packed.design_concat[:, :n_task_cols]
        nuisance = (packed.design_concat[:, n_task_cols:]
                    if packed.design_concat.shape[1] > n_task_cols else None)
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
            prior_mean=prior_m, prior_cov=prior_C,
            polort=polort_resolved,
            prior_weight=pw,
            lambda_mode=args.lambda_mode,
            lambda_n_bins=args.lambda_n_bins,
            lss_exclude=args.lss_exclude,
            device=device, verbose=True,
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
                cells, total=len(cells),
                desc="  Cells × constrained fit", unit="cell",
                leave=False, disable=len(cells) <= 1,
            )
            for cell in cell_iter:
                packed_cell = pack_for_shared_task_glm(
                    per_run_data=cell.per_run_data,
                    per_run_task_designs=cell.per_run_task_designs,
                    polort=-1,                          # polys go via extra
                    task_column_labels=task_column_labels,
                    extra_regressors_per_run=cell.per_run_polys,
                    device=device,
                )
                cell_packed_cache.append(packed_cell)
                task_design_cell = packed_cell.design_concat[:, :n_task_cols]
                nuisance_cell = (
                    packed_cell.design_concat[:, n_task_cols:]
                    if packed_cell.design_concat.shape[1] > n_task_cols else None
                )
                pw_cell = (
                    prior_weight_per_voxel[cell.voxel_indices]
                    if prior_weight_per_voxel is not None else None
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
                hrfs=None,                              # type: ignore[arg-type]
                r2=r2_all,
                betas_ols=betas_ols_all,
                hrfs_ols=None,                          # type: ignore[arg-type]
                r2_ols=r2_ols_all,
                sigma2_mean=sigma2_sum / max(1, n_voxels_total),
                effective_prior_weight=eff_pw_sum / max(1, n_voxels_total),
                n_iter=1,
                sigma2_per_voxel=sigma2_per_voxel_all.copy(),
                lambda_per_voxel=lambda_per_voxel_all.copy(),
            )

        fit = _run_cell_fit(arma_cells)
        print(f"  ✓ Per-voxel ARMA fit complete (iter 0).  "
              f"σ²_mean={fit.sigma2_mean:.4g}, "
              f"effective λ_mean={fit.effective_prior_weight:.4g}")
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  "
              f"constrained: {fit.r2.mean():.3f}")

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
            user_mult = (
                1.0 if isinstance(pw, str)
                else float(pw)
            )
            beta_size_per_voxel = np.full(
                n_voxels_total, float(user_mult), dtype=np.float32,
            )
            prior_pw_per_voxel: np.ndarray | None = None
            if args.vb_update_prior and args.reg in {"ridge", "mvn", "mvn-shape"}:
                # Compute initial β_size from the iter-0 fit's posterior
                # moments so that iter-1 fits with the *updated* prior.
                # Block-trace per voxel: σ²_v · Σ_b tr(C⁻¹ · (A⁻¹)_b)
                # via per-cell packed designs cached in _run_cell_fit.
                block_trace_summed = np.zeros(n_voxels_total, dtype=np.float32)
                for cell, packed_cell in zip(arma_cells, cell_packed_cache):
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
                task_betas_3d = (
                    fit.betas[:, :n_task_cols]
                    .reshape(n_voxels_total, n_blocks, n_basis)
                )
                beta_size_per_voxel = vb_update_beta_size(
                    task_betas=task_betas_3d,
                    prior_mean=prior_m,
                    prior_cov=prior_C,
                    block_trace_summed=block_trace_summed,
                )
                # Translate to per-voxel λ: λ_v = β_size_v · σ²_v.
                prior_pw_per_voxel = (
                    beta_size_per_voxel * sigma2_per_voxel_all
                )
                print(
                    f"  VB β_size update (iter 0): median={float(np.median(beta_size_per_voxel)):.3f}, "
                    f"5–95% [{float(np.percentile(beta_size_per_voxel, 5)):.3f}, "
                    f"{float(np.percentile(beta_size_per_voxel, 95)):.3f}]"
                )
            vb_iter = 0
            for vb_iter in range(1, args.vb_iters + 1):
                print(f"\n  VB iter {vb_iter}/{args.vb_iters}…")
                # 1. Residuals in original space.
                nuis_betas_arr = (
                    fit.betas[:, n_task_cols:]
                    if fit.betas.shape[1] > n_task_cols else None
                )
                residuals_per_run = compute_per_voxel_residuals(
                    per_run_data=per_run_data,
                    per_run_task_designs=per_run_designs,
                    polort=polort_resolved,
                    task_betas=fit.betas[:, :n_task_cols],
                    nuisance_betas=nuis_betas_arr,
                    device=device,
                )
                # Free the previous iter's whitened-cell state — the
                # 88 cells hold a full copy of the data on GPU + the
                # packed designs (~7 GB at 9.4T scale).  Residuals
                # have been computed already (they use the original
                # per_run_data, not the cells), so the cells are no
                # longer needed.  Without this, the next REML
                # precompute + Y_full allocation OOMs.
                arma_cells = None
                cell_packed_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # 2. Re-estimate (a, b) from residuals.  Use polort=-1
                # (residuals are already drift-free).
                new_ab = estimate_arma11_per_voxel(
                    per_run_data=residuals_per_run,
                    per_run_task_designs=per_run_designs,
                    polort=-1,
                    device=device, verbose=True,
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
                    device=device, verbose=True,
                )
                fit = _run_cell_fit(
                    arma_cells, prior_weight_per_voxel=prior_pw_per_voxel,
                )
                print(
                    f"  VB iter {vb_iter} fit: σ²_mean={fit.sigma2_mean:.4g}, "
                    f"R² constrained={fit.r2.mean():.3f}"
                )
                # 5. VB β_size update from new posterior.
                if args.vb_update_prior and args.reg in {"ridge", "mvn", "mvn-shape"}:
                    block_trace_summed = np.zeros(n_voxels_total, dtype=np.float32)
                    for cell, packed_cell in zip(arma_cells, cell_packed_cache):
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
                    task_betas_3d = (
                        fit.betas[:, :n_task_cols]
                        .reshape(n_voxels_total, n_blocks, n_basis)
                    )
                    new_beta_size = vb_update_beta_size(
                        task_betas=task_betas_3d,
                        prior_mean=prior_m,
                        prior_cov=prior_C,
                        block_trace_summed=block_trace_summed,
                    )
                    bs_change = float(np.median(
                        np.abs(new_beta_size - beta_size_per_voxel)
                    ))
                    beta_size_per_voxel = new_beta_size
                    prior_pw_per_voxel = (
                        beta_size_per_voxel * sigma2_per_voxel_all
                    )
                    print(
                        f"  VB β_size iter {vb_iter}: "
                        f"median={float(np.median(beta_size_per_voxel)):.3f}, "
                        f"5–95% [{float(np.percentile(beta_size_per_voxel, 5)):.3f}, "
                        f"{float(np.percentile(beta_size_per_voxel, 95)):.3f}], "
                        f"median |Δβ_size|={bs_change:.4f}"
                    )
            print(f"  ✓ VB loop complete after {vb_iter} iter(s).")

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
        print(f"  ✓ Fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
              f"median optimal frac = {float(np.median(fit.optimal_fracs)):.2f}")
        print(f"  Held-out R² mean — OLS (frac=1.0): {fit.r2_ols.mean():.3f}  "
              f"fracridge optimal: {fit.r2.mean():.3f}")
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
        print(f"  ✓ Fit complete.  σ²_mean={fit.sigma2_mean:.4g}, "
              f"effective λ={fit.effective_prior_weight:.4g}")
        print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}  "
              f"constrained: {fit.r2.mean():.3f}")

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
        fit.betas[:, :n_task_cols] = (
            fit.betas[:, :n_task_cols] + empirical_prior_mean_full
        )
        fit.betas_ols[:, :n_task_cols] = (
            fit.betas_ols[:, :n_task_cols] + empirical_prior_mean_full
        )
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
        [np.interp(dst_times, src_times, basis.basis_functions[k])
         for k in range(n_basis)],
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
    print(
        f"  Amplitude/iresp chunking: {chunk_size:,} voxels per chunk "
        f"(n_blocks={n_blocks})"
    )

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
        save_full_iresp = True   # per-condition default
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
        np.zeros((n_vox_masked, n_blocks, n_t_iresp), dtype=np.float32)
        if save_full_iresp else None
    )
    iresp_buf_ols = (
        np.zeros((n_vox_masked, n_blocks, n_t_iresp), dtype=np.float32)
        if save_full_iresp else None
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
        desc="  Amplitude/iresp", unit="chunk",
        leave=False, disable=n_amp_chunks <= 1,
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
    for arr, sfx in ((fit.r2, ""), (fit.r2_ols, "_unconstrained")):
        path = f"{args.prefix}_fitbasis_r2{sfx}{nii_ext}"
        save_nifti(_to_volume(arr[:, None]).squeeze(-1),
                   output_path=path, reference_img=args.input[0])
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
            print(
                f"  -xval-r2 not yet supported with -reg {args.reg}; "
                "skipping."
            )
        elif n_runs < 2:
            print("  -xval-r2 needs ≥2 runs; skipping.")
        elif per_run_data_orig is None:
            print(
                "  -xval-r2 requested but per_run_data_orig was not "
                "saved (internal); skipping."
            )
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
                device=device,
                verbose=args.verb >= 1,
            )
            xval_path = f"{args.prefix}_fitbasis_xvalr2{nii_ext}"
            save_nifti(_to_volume(xval_r2[:, None]).squeeze(-1),
                       output_path=xval_path, reference_img=args.input[0])
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
            desc="  Single-trial amplitudes", unit="cond",
            leave=False, disable=len(per_cond_trial_idx) <= 1,
        ):
            for amps, sfx in ((amplitude, ""), (amplitude_ols, "_unconstrained")):
                amp_stack = amps[:, idxs]              # (n_vox, n_trials_for_cond)
                amp_vol = _to_volume(amp_stack)         # (nx, ny, nz, n_trials)
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
                    tr=iresp_dt, bot=0.0, top=iresp_dt * (n_t_iresp - 1),
                    reference_img=args.input[0], nii_ext=nii_ext,
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
            desc="  Single-trial basis weights", unit="cond",
            leave=False, disable=len(per_cond_trial_idx) <= 1,
        ):
            stack = task_betas[:, idxs, :]            # (n_vox, n_trials, K)
            stack_ols = task_betas_ols[:, idxs, :]
            for arr, sfx in ((stack, ""), (stack_ols, "_unconstrained")):
                for b in range(n_basis):
                    # (n_vox, n_trials) → (nx, ny, nz, n_trials)
                    vol_4d = _to_volume(arr[:, :, b])
                    path = (
                        f"{args.prefix}_fitbasis_basisweight{b + 1:02d}_"
                        f"{cond}{sfx}{nii_ext}"
                    )
                    save_nifti(vol_4d, output_path=path, reference_img=args.input[0])
            print(
                f"  Wrote basisweight01..{n_basis:02d} for {cond} "
                f"(n_trials={len(idxs)})"
            )

    else:
        # Per-condition outputs (the simple case)
        if save_full_iresp and iresp_buf is not None:
            for hrfs_arr, sfx in ((iresp_buf, ""), (iresp_buf_ols, "_unconstrained")):
                iresp_vol = _to_volume(hrfs_arr)
                save_iresp(
                    iresp=iresp_vol,
                    output_prefix=f"{args.prefix}_fitbasis",
                    condition_labels=[f"{lbl}{sfx}" for lbl in block_labels],
                    tr=iresp_dt, bot=0.0, top=iresp_dt * (n_t_iresp - 1),
                    reference_img=args.input[0], nii_ext=nii_ext,
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
        assert isinstance(fit, FracRidgeFitResult)            # narrow for type checker
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
            save_nifti(r2_4d, output_path=r2_path, reference_img=args.input[0])
            print(f"  Wrote {r2_path}  (4-D; volume k = held-out R² at fracs[k])")

    if args.cv_runs and args.reg != "fracridge":
        if n_runs < 2:
            print(
                f"  WARNING: -cv-runs requires ≥2 runs (got {n_runs}); skipping CV."
            )
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
                [float(np.median(cv_result.r2_per_weight[:, i]))
                 for i in range(len(cv_result.weights))]
            )
            n_best_arr = np.array(
                [int((cv_result.argmax_weight_idx == i).sum())
                 for i in range(len(cv_result.weights))]
            )
            best_median_i = int(np.argmax(medians_arr))
            best_count_i = int(np.argmax(n_best_arr))
            print("  Held-out R² summary:")
            print(f"    {'weight':>8}  {'median':>9}  {'mean':>9}  {'max':>9}  {'n_best_voxels':>14}")
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
            save_nifti(argmax_3d.astype(np.float32),
                       output_path=argmax_path, reference_img=args.input[0])
            print(f"  Wrote {argmax_path}  (int → index into weights list above)")

            # Per-weight R² 4-D NIfTI (one volume per weight in the grid).
            r2_4d_shape = volume_shape + (cv_result.r2_per_weight.shape[1],)
            r2_4d = np.zeros(r2_4d_shape, dtype=np.float32)
            if mask is not None:
                r2_4d[mask, :] = cv_result.r2_per_weight
            else:
                r2_4d = cv_result.r2_per_weight.reshape(r2_4d_shape)
            r2_4d_path = f"{args.prefix}_fitbasis_cv_r2_per_weight{nii_ext}"
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
                int((cv_result.argmax_weight_idx == i).sum())
                for i in range(len(cv_result.weights))
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
