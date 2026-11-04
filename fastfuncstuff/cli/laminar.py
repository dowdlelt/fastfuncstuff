"""CLI: the laminar BOLD model on ROI x depth response curves -- model-based deveining.

Command: ffs_laminar

Usage:
    # one run at several vascular resolutions, then the model on its outputs
    ffs_deconvolve -input bold.nii.gz -onsets std.1D dev.1D -model TENT -window 0 14 \
        -atlas V1.lh.label -fs_subj $SUBJ -depths 7 9 11 -equivolume -prefix fit/sub01
    ffs_laminar -fit_prefix fit/sub01 -conds std dev -duration 1.6 -prefix lam/sub01
    # or from per-K files
    ffs_laminar -cond1 k7_iresp_std.tsv k9_iresp_std.tsv -cond2 k7_iresp_dev.tsv k9_iresp_dev.tsv \
        -duration 1.6 -psf_matrix k7_depth_psf.tsv k9_depth_psf.tsv -prefix lam/sub01
    ffs_laminar -cond1 a_iresp_c1.tsv -cond2 a_iresp_c3.tsv -duration 1.6 -roi V1 -hemi lh \
        -psf_kernel shipped_kernel.1D -models null superficial -prefix quick
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser

_DESCRIPTION = """\
Invert the Havlicek & Uludag laminar BOLD model on depth-resolved response curves,
the analysis of Faes et al. (2026) (the notebook laminar_dcm_predictive_tones, as a
tool): two conditions of one ROI are concatenated with masked white-noise padding,
every layer hypothesis -- which neuronal depths the second condition modulates -- is
inverted at each vascular resolution K, each model's posteriors are averaged across K,
and the averaged models are compared by free energy.

Inputs are the tidy curve tables ffs_deconvolve (or ffs_reml) writes in ROI x depth
mode, {prefix}_iresp_<condition>.tsv, in PERCENT signal change (deconvolve's default;
not -dont_scale).  Sample several depth counts in one run (-depths 7 9 11: the reference
uses 7, 9, 10 and 11) -- one table then holds them all, with a K column -- or give one
table per K.  -fit_prefix PREFIX -conds A B finds the tables and the measured PSF from a
run's prefix.  The model's scan interval is the curves' knot spacing; -duration is the
event they respond to.

The two conditions are fitted as ONE series: the first response, rows of padding noise,
then the second.  The padding is masked out of the fit; its job is time -- the model is
one continuous ODE, and the quiet rows let its states relax to baseline so the second
response starts from rest, as the data do.  See -pad_rows / -pad_sd / -pad_seed.

The depth point-spread function, one of:
  -psf_matrix FILE [FILE ...]  the measured mixing of this sampling, one
                               *_depth_psf.tsv per K (reml/deconvolve/nwarp write
                               them).  Rows normalised to 1 unless -psf_raw.
  -psf_kernel FILE             a 1-D kernel (numbers in a text file) applied at every
                               K, as the reference driver does -- parity.
  -psf_variance V              the reference's Gaussian form, built for each K.
  none of them                 no PSF.

Writes {prefix}_laminar_models.tsv (F, model and per-K evidence), _depths.tsv
(P(depth modulated) marginalised over the model space -- prefer it to the winner --
and the best model's B with 90% credible intervals), _devein.tsv (the static
deveining operator, neuronal depth x measured depth, depth 1 deepest), _neuronal.tsv
(deveined neuronal time courses) and _laminar.json.

Runs on the CPU in float64, as the model's small ODE wants; ROIs run in parallel
across -workers processes.
"""


_EPILOG = """\
READING THE OUTPUT

Read the tables in this order; each step decides whether the next one means anything.

1. Is the fit sound?  {prefix}_laminar_models.tsv carries per-model F and the
   per-K evidence, and the tool warns on two conditions that make a ranking
   worthless:
     stalled      the run ended on rejected steps.  SPM's rule calls a fit
                  converged when the predicted gain is under 0.1 four times
                  running, and after a rejected step the damping can collapse so
                  far that the gain vanishes hundreds of nats short -- measured
                  once at 480 nats.  -rescue 2 restarts from the best point.
     nesting      a model scoring below a model it CONTAINS.  That is impossible
                  (the smaller model is the larger one with B pinned at 0), so it
                  is a local optimum, not evidence.  >10 nats is warned about.
   A ranking with either flag set is a report about the optimiser.

2. Is there anything to explain?  The curves are percent signal change, so look
   at their peak.  An ROI with no response gives the model nothing to work on and
   its B values are noise however confidently flagged.

3. Is there any modulation at all?  F(best) - F(null).  The conventional reading
   of a log Bayes factor applies: ~3 nats is about 20:1, which is the usual floor
   for calling a difference worth interpreting; below it the honest answer is
   "no evidence that these conditions differ at any depth", and the depth
   probabilities below are then a decomposition of nothing.

   That floor is a CONVENTION, and on real data it is too low.  The model space's
   'null' is a model fitted to the same data, so its Bayes factor answers "does
   something differ" and carries no false-positive rate: measured over 903 real
   columns, permuting the condition labels gave a null median of 4.2 nats -- above
   the 3-nat floor -- and a null maximum of 26.5.  Use -null_fit_prefix to get the
   rate itself.  {prefix}_laminar_null.tsv then reports, per ROI and statistic,
   p_perm = (1 + #{null >= real}) / (1 + n): the share of permutations of YOUR
   labels that reached YOUR result.  That is the number to quote, and with -plots
   the distribution behind it is drawn.  19 permutations is the fewest that can
   express 0.05, so a smaller n is reported as the floor it is.

4. Only then, WHERE.  Prefer P(depth modulated) in {prefix}_depths.tsv, which is
   marginalised over the whole model space, to the winning model's name.  The
   winner is one of 2^N hypotheses and its p=0.4 victory over a near-tie is not
   a laminar claim; the marginal says how much of the posterior mass puts the
   effect at that depth whatever else is going on.  The B credible intervals in
   the same file are the best model's.

5. Sign and size.  B modulates the self-inhibition sigma (A_kk = -sigma + u*B_k),
   so POSITIVE B lowers inhibition.  sigma - B below zero is net self-excitation
   while the modulation is on, which the published example ROI's fit has
   (sigma 0.13 Hz, B 2-5 Hz).  P(depth modulated) ignores the sign, so a
   comparison is non-directional however its two sides are ordered.

THE DEVEINING OUTPUT

_devein.tsv is the (N, K) pseudo-inverse of the fitted neuronal -> BOLD mixing.
Three things about the true operator decide how far it can be trusted, and they
are properties of the vasculature, not of this implementation:

  It is TRIANGULAR, not diagonal.  Blood drains pial-ward, so a depth carries
  contributions from every depth below it and none from above.  Measured on a
  fitted model at K=6, N=3, superficial BOLD was ~49% contamination from below.
  This is why a per-depth scalar is the wrong SHAPE: rescaling a depth against
  itself cannot subtract what drained into it.

  It has a TIME AXIS.  The ascending vein imposes a transit delay and smears the
  response, and a scalar cannot represent a delay.  The amplitude reduction is
  therefore exact for a summary statistic (peak, area) and not for a timecourse.
  _neuronal.tsv is the exact answer instead, but it is tied to this stimulus and
  this fit and is not an operator you can carry elsewhere.

  It is AMPLITUDE-DEPENDENT.  The balloon and dHb terms are nonlinear, so an
  operator derived at one effect size is wrong at another.  Fitting a CONTRAST
  cancels much of this; a single condition (or -models null) does not, so its
  operator is calibrated at that response amplitude and no other.

And deconvolution amplifies noise: removing the deep contribution from a
superficial depth subtracts a PREDICTION carrying posterior uncertainty.  A
scalar multiplier hides that entirely -- it scales signal and noise together, so
tSNR looks untouched and the correction looks free.  It is not.

WHAT NOT TO BELIEVE

  Low s_d is the dangerous direction.  Underestimating the ascending vein's
  baseline-CBV slope gives qualitatively WRONG laminar profiles; overestimating
  it is safe (Uludag & Havlicek 2021).  Look at the fitted s_d before trusting a
  profile, and treat a low one as a reason to doubt the depth assignment.

  Absolute F is not comparable across runs.  Under -hyperprior reference (the
  default, for parity) the noise prior's centre is taken over the PADDED series,
  so -pad_sd and -pad_seed move F by ~1 nat while the model comparison moves
  ~0.2.  Rankings within a run are safe; F between runs with different seeds is
  not.  -hyperprior rows removes the dependence and gives up comparability with
  published values.

  -free_energy reference (the default) takes F from the FIRST K only, as
  spm_dcm_bpa leaves it, while the PARAMETERS are averaged over every K.  The
  ranking therefore rests on one vascular resolution.  'sum' / 'mean' pool it,
  at the cost of parity -- and neither is a proper joint log-evidence, since the
  Ks are different samplings of the same voxels, not independent observations.

  Identifiability at real SNR is unproven, by the model's authors' own account:
  the 2021 paper declined to explore dynamic invertibility and warned it depends
  on SNR, ROI size and within-ROI timecourse variability, and the neuronal model
  is deliberately cross-laminar-uncoupled BECAUSE richer models stop being
  distinguishable under fMRI noise.  Treat a depth assignment as a hypothesis
  with evidence attached, not a measurement.

  Depth is not layer.  Layer IV moves with curvature, and sensitivity varies
  with depth at constant activity (baseline CBV, T2*, angle to B0).  Equivolume
  sampling is the best cheap fix and is still imperfect.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_laminar", description=_DESCRIPTION, epilog=_EPILOG)
    g = p.add_argument_group("Data")
    g.add_argument("-cond1", nargs="+", default=None, metavar="TSV",
                   help="The first (baseline) condition's curve table(s): one per K, or one "
                   "table holding several K (-depths 7 9 11).")  # fmt: skip
    g.add_argument("-cond2", nargs="+", default=None, metavar="TSV",
                   help="The second condition's, the one the modulatory input marks.")  # fmt: skip
    g.add_argument("-fit_prefix", default=None, metavar="PREFIX",
                   help="Instead of -cond1/-cond2/-psf_matrix: the -prefix of an ffs_deconvolve "
                   "or ffs_reml ROI x depth run; with -conds, finds PREFIX_iresp_<cond>.tsv and "
                   "PREFIX_depth_psf.tsv (the measured PSF, used unless another PSF is chosen).")  # fmt: skip
    g.add_argument("-conds", nargs=2, default=None, metavar=("COND1", "COND2"),
                   help="With -fit_prefix: the two conditions' names, baseline first.")  # fmt: skip
    g.add_argument("-duration", type=float, required=True, nargs="+", metavar="SEC",
                   help="Duration of the events the curves respond to (s): one value, or "
                   "two (BASELINE TEST) when the two conditions' events differ in length. "
                   "With -modulation event the modulatory input follows the TEST event.")  # fmt: skip
    g.add_argument("-roi", nargs="+", default=None, metavar="ROI",
                   help="ROIs to fit, by id or name (default: every ROI in every table).")  # fmt: skip
    g.add_argument("-hemi", default=None, choices=("lh", "rh"),
                   help="Hemisphere, when an ROI name is in both.")  # fmt: skip
    pad = p.add_argument_group(
        "Padding between the two conditions",
        "The two responses are fitted as one series, separated by rows of noise the fit "
        "ignores: the padding exists so the model returns to baseline between the "
        "responses instead of bridging them.  Each row is one sample of the curves (the "
        "TENT knot spacing), so the default 20 rows is 20 knots of quiet.",
    )
    pad.add_argument("-pad_rows", "-gap", dest="pad_rows", type=int, default=20, metavar="N",
                     help="Rows of padding between the conditions (default 20, the reference's).")  # fmt: skip
    pad.add_argument("-pad_sd", type=float, default=0.1, metavar="PCT",
                     help="Standard deviation of the padding noise, in percent signal change "
                     "(default 0.1, the reference's).")  # fmt: skip
    pad.add_argument("-pad_seed", "-seed", dest="pad_seed", type=int, default=20260918, metavar="N",
                     help="Seed of the padding noise.")  # fmt: skip
    pad.add_argument("-hyperprior", choices=("reference", "rows"), default="reference",
                     help="Where the noise prior's centre comes from.  'reference': the "
                     "variance of the whole padded series, as the published analysis does -- "
                     "so -pad_sd and -pad_seed move the absolute free energy (~1 nat; the model "
                     "comparison ~0.2).  'rows': the fitted rows only -- the padding then "
                     "changes nothing, but F is no longer comparable with published values.")  # fmt: skip

    psf = p.add_argument_group("Depth point-spread function")
    one = psf.add_mutually_exclusive_group()
    one.add_argument("-psf_matrix", nargs="+", default=None, metavar="TSV",
                     help="*_depth_psf.tsv tables, one per K (matched by their number of depths).")  # fmt: skip
    one.add_argument("-psf_kernel", default=None, metavar="FILE",
                     help="A 1-D kernel applied at every K (the reference's parity form).")  # fmt: skip
    one.add_argument("-psf_variance", type=float, default=None, metavar="V",
                     help="Variance of the reference's Gaussian PSF on the 0..1 depth axis.")  # fmt: skip
    psf.add_argument("-psf_raw", action="store_true",
                     help="Use -psf_matrix rows as written (cortex share < 1 at the edge depths) "
                     "instead of normalised to 1.")  # fmt: skip

    m = p.add_argument_group("Model")
    m.add_argument("-models", nargs="+", default=None, metavar="NAME",
                   help="Subset of the model space by name (null, superficial, middle, deep, "
                   "superficial+middle, ...). Default: all 2^N.")  # fmt: skip
    m.add_argument("-N", type=int, default=3, help="Neuronal depths.")
    m.add_argument("-free_params", nargs="+", default=[], metavar="NAME",
                   choices=("mu", "lam", "tau_v", "tau_d"),
                   help="Estimate these instead of fixing them at the published values: mu and "
                   "lam (the inhibitory feedback, gain and rate) and tau_v and tau_d (venule and "
                   "ascending-vein volume recovery). Faes et al. fix them because a short ITI "
                   "cannot observe the post-stimulus undershoot they shape, and fixed they "
                   "predict almost none; a deconvolved curve that reaches past it does observe "
                   "it. 'mu lam' fits an undershoot by neuronal inhibition, tau_* by slow venous "
                   "recovery, and data this short may not tell the two apart. Default: none, the "
                   "published model. F is comparable only between runs on the same curves.")  # fmt: skip
    m.add_argument("-free_energy", choices=("reference", "sum", "mean"), default="reference",
                   help="Which F an averaged model carries: 'reference' is the first K's, as "
                   "spm_dcm_bpa leaves it; 'sum'/'mean' accumulate over K.")  # fmt: skip
    m.add_argument("-delay", type=float, default=0.0, metavar="SEC",
                   help="Where in each sample the prediction is read. TENT knots sit at exact "
                   "times after the event, so 0.")  # fmt: skip
    m.add_argument("-reference_delay", action="store_true",
                   help="The reference's slice-timing delay instead, TR/2.")  # fmt: skip
    m.add_argument("-modulation", nargs="+", default=["event"], metavar="WHEN",
                   help="When the modulatory input (the second condition's difference) is on: "
                   "'event' (default) for the whole second event -- the conditions are "
                   "different stimuli or tasks; 'reference' for the published 0.1 s pulse "
                   "ending one sample after the event, which in Faes et al. is the fourth tone "
                   "of the quartet (the prediction violation) -- for parity only; or START DUR, "
                   "seconds after the second onset.")  # fmt: skip
    m.add_argument("-TE", type=float, default=0.028, help="Echo time (s).")
    m.add_argument("-B0", type=float, default=7.0, help="Field strength (T).")
    m.add_argument("-max_iter", type=int, default=128, help="Variational Laplace iterations.")
    m.add_argument("-rescue", type=int, default=0, metavar="N",
                   help="Restart a fit that stalled up to N times from its best point with the "
                   "damping reset.  The reference's rule calls a run converged when its "
                   "predicted gain is under 0.1 four times running; after a rejected step the "
                   "damping can collapse so far that the gain vanishes while the fit is hundreds "
                   "of nats short (seen: a model scoring 480 nats below one it contains).  "
                   "Stalls are always flagged and warned about; the default does not change a "
                   "number, so results stay comparable with the published analysis.  2 is "
                   "plenty.")  # fmt: skip

    n = p.add_argument_group("Permutation null")
    n.add_argument("-null_fit_prefix", nargs="+", default=None, metavar="PREFIX",
                   help="One or more PREFIXes of deconvolve runs whose condition labels were "
                   "PERMUTED (scripts/laminar_null_onsets.py writes them), read with -conds "
                   "exactly as -fit_prefix is.  Each is refitted through the identical model "
                   "space and ROI selection, giving the distribution of every statistic when "
                   "the two conditions are known not to differ -- which is the only thing that "
                   "turns a free energy into a false-positive rate.  The model space's own "
                   "'null' is a MODEL fitted to the same data, and on real data its Bayes "
                   "factor is large almost everywhere (a measured null median of 4.2 nats "
                   "against the conventional 3-nat floor), so it cannot play this part.  Cost "
                   "is one extra fit per ROI per permutation, which is why this lives here and "
                   "not in ffs_laminascope: 19 permutations is the fewest that can express "
                   "p = 0.05, and ~100 is a readable distribution.")  # fmt: skip
    n.add_argument("-null_cond1", nargs="+", default=None, metavar="FILE",
                   help="Permuted replicates given as explicit tables instead of prefixes: "
                   "N files for condition 1, matched by position with -null_cond2.  One K "
                   "only; use -null_fit_prefix for multi-K runs.")  # fmt: skip
    n.add_argument("-null_cond2", nargs="+", default=None, metavar="FILE",
                   help="The condition-2 side of -null_cond1.")  # fmt: skip

    o = p.add_argument_group("Output")
    o.add_argument("-prefix", required=True, help="Output prefix.")
    o.add_argument("-plots", action="store_true",
                   help="Also write one figure per ROI, {prefix}_laminar_<roi>_<hemi>.png: the "
                   "model input and what was masked, free energy per model and K, model and "
                   "depth probabilities, the modulation B with credible intervals, data against "
                   "fit, measured against deveined time courses and depth profiles, the deveining "
                   "operator, the vein-attributable shift, transit delay and the depth PSF.  With several "
                   "ROIs also {prefix}_laminar_summary.png: per ROI and neuronal depth, where the "
                   "effect is.")  # fmt: skip
    o.add_argument("-workers", type=int, default=0,
                   help="Processes, one ROI each (0: half the cores).")  # fmt: skip
    o.add_argument("-verb", type=int, default=1, help="0 quiet, 1 progress.")
    return p.parse_args(argv)


def _by_k(paths, what):
    """``{K: table}`` from curve tables: one count per file, or several counts in one
    file (a ``K`` column, from ``-depths 7 9 11``)."""
    from fastfuncstuff.laminar.io import read_iresp_by_k

    out = {}
    for path in paths:
        for k, table in read_iresp_by_k(path).items():
            if k in out:
                raise ValueError(f"{what}: two tables have K = {k} ({path})")
            out[k] = table
    return out


def _kernels(args, ks, name, hemi):
    import torch

    from fastfuncstuff.laminar.io import model_kernel
    from fastfuncstuff.laminar.layers import gaussian_depth_kernel

    if args.psf_kernel:
        kern = np.loadtxt(args.psf_kernel).reshape(-1)
        return {k: torch.as_tensor(kern, dtype=torch.float64) for k in ks}
    if args.psf_variance is not None:
        return {k: torch.as_tensor(gaussian_depth_kernel(args.psf_variance, k)) for k in ks}
    if args.psf_matrix:
        from fastfuncstuff.processing.roi_depth import read_depth_psf_by_k

        mats = {}
        for path in args.psf_matrix:
            for k, per in read_depth_psf_by_k(path).items():
                if (name, hemi) in per:
                    mats[k] = per[(name, hemi)]
        missing = [k for k in ks if k not in mats]
        if missing:
            raise ValueError(f"ROI {name} ({hemi}): no -psf_matrix table with K = {missing}")
        return {k: model_kernel(mats[k], normalise=not args.psf_raw) for k in ks}
    return None


def _resolve_fit_prefix(args) -> None:
    """Fill -cond1 / -cond2 / -psf_matrix from a fit's output prefix."""
    from pathlib import Path

    if args.fit_prefix:
        if not args.conds:
            raise ValueError("-fit_prefix needs -conds COND1 COND2")
        if args.cond1 or args.cond2:
            raise ValueError("give -fit_prefix or -cond1/-cond2, not both")
        found = [f"{args.fit_prefix}_iresp_{c}.tsv" for c in args.conds]
        for path in found:
            if not Path(path).exists():
                raise ValueError(f"{path}: not found (-fit_prefix {args.fit_prefix}, -conds)")
        args.cond1, args.cond2 = [found[0]], [found[1]]
        psf = Path(f"{args.fit_prefix}_depth_psf.tsv")
        if psf.exists() and not (
            args.psf_matrix or args.psf_kernel or args.psf_variance is not None
        ):
            args.psf_matrix = [str(psf)]
    if not args.cond1 or not args.cond2:
        raise ValueError("give -cond1 and -cond2 (or -fit_prefix and -conds)")


def _null_tables(args, ks):
    """``[(cond1_by_k, cond2_by_k), ...]``, one pair per permutation replicate."""
    from pathlib import Path

    pairs = []
    if args.null_fit_prefix:
        if not args.conds:
            raise ValueError("-null_fit_prefix needs -conds COND1 COND2")
        for prefix in args.null_fit_prefix:
            found = [f"{prefix}_iresp_{c}.tsv" for c in args.conds]
            for path in found:
                if not Path(path).exists():
                    raise ValueError(f"{path}: not found (-null_fit_prefix {prefix})")
            pairs.append(([found[0]], [found[1]]))
    if args.null_cond1 or args.null_cond2:
        if not (args.null_cond1 and args.null_cond2):
            raise ValueError("-null_cond1 and -null_cond2 go together")
        if len(args.null_cond1) != len(args.null_cond2):
            raise ValueError(f"-null_cond1 has {len(args.null_cond1)} file(s) and -null_cond2 "
                             f"{len(args.null_cond2)}; they pair by position")  # fmt: skip
        pairs.extend(([a], [b]) for a, b in zip(args.null_cond1, args.null_cond2, strict=True))
    out = []
    for i, (c1, c2) in enumerate(pairs):
        one, two = _by_k(c1, "-null_cond1"), _by_k(c2, "-null_cond2")
        if sorted(one) != sorted(two) or sorted(one) != list(ks):
            raise ValueError(f"permutation {i} has K = {sorted(one)} / {sorted(two)}, "
                             f"but the real data have {list(ks)}")  # fmt: skip
        out.append((one, two))
    return out


def _fit_one(job):
    import torch

    torch.set_num_threads(1)
    from fastfuncstuff.laminar.pipeline import fit_roi

    data_for_k, kernels, targets, names, kw = job
    return fit_roi(data_for_k, kernels, targets=targets, names=names, **kw)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.laminar.experiment import layer_model_names, layer_model_targets
    from fastfuncstuff.laminar.io import pick_roi, two_condition_data
    from fastfuncstuff.laminar.pipeline import reference_p0, write_results

    args = parse_args(argv)
    t0 = time.time()
    try:
        _resolve_fit_prefix(args)
        one, two = _by_k(args.cond1, "-cond1"), _by_k(args.cond2, "-cond2")
        if sorted(one) != sorted(two):
            raise ValueError(f"-cond1 has K = {sorted(one)} but -cond2 has K = {sorted(two)}")
        ks = sorted(one)
        all_names, all_targets = layer_model_names(args.N), layer_model_targets(args.N)
        if args.models:
            bad = [m for m in args.models if m not in all_names]
            if bad:
                raise ValueError(f"unknown model(s) {bad}; the space is {all_names}")
            names = list(args.models)
            targets = [all_targets[all_names.index(m)] for m in names]
        else:
            names, targets = all_names, all_targets
        keys = [k for k in one[ks[0]] if all(k in one[q] and k in two[q] for q in ks)]
        if args.roi:
            wanted = set(args.roi)
            keys = [k for k in keys if (k[0] in wanted or k[1] in wanted)]
        if args.hemi:
            keys = [k for k in keys if k[2] == args.hemi]
        if not keys:
            raise ValueError("no ROI is in every table (and matches -roi / -hemi)")

        delay = None if args.reference_delay else args.delay
        modulation = parse_modulation(args.modulation)

        def build(one_tables, two_tables):
            built = []
            for rid, name, hemi in keys:
                data_for_k = {
                    k: two_condition_data(
                        pick_roi(one_tables[k], rid, hemi),
                        pick_roi(two_tables[k], rid, hemi),
                        duration=args.duration,
                        n_neuronal=args.N,
                        gap=args.pad_rows,
                        seed=args.pad_seed,
                        noise_sd=args.pad_sd,
                        delay_seconds=delay,
                        TE=args.TE,
                        B0=args.B0,
                        p0=reference_p0(),
                        modulation=modulation,
                    )  # fmt: skip
                    for k in ks
                }
                kw = {"free_energy": args.free_energy, "max_iter": args.max_iter,
                      "hyper_from_rows": args.hyperprior == "rows", "rescue": args.rescue,
                      "free": args.free_params, "roi": rid, "name": name, "hemi": hemi}  # fmt: skip
                # The PSF is geometry, so the permutations reuse the real run's: permuting
                # labels cannot change how a depth was sampled.
                built.append((data_for_k, _kernels(args, ks, name, hemi), targets, names, kw))
            return built

        jobs = build(one, two)
        n_real = len(jobs)
        for job in jobs:  # the null replicates draw no curves
            job[4]["detail"] = args.plots
        null_pairs = _null_tables(args, ks)
        for one_n, two_n in null_pairs:
            jobs.extend(build(one_n, two_n))
    except (OSError, ValueError) as exc:
        print(f"ffs_laminar: {exc}", file=sys.stderr)
        return 1

    psf = (
        "matrix" if args.psf_matrix else "kernel" if args.psf_kernel
        else "gaussian" if args.psf_variance is not None else "none"
    )  # fmt: skip
    if args.verb >= 1:
        extra = (f" + {len(null_pairs)} permutation(s) of the same ROIs"
                 if null_pairs else "")  # fmt: skip
        print(f"ffs_laminar: {n_real} ROI(s){extra} x {len(names)} model(s) x K = {ks} "
              f"({len(jobs) * len(names) * len(ks)} inversions), PSF: {psf}")  # fmt: skip
        if null_pairs and len(null_pairs) < 19:
            print(f"  NOTE: {len(null_pairs)} permutation(s) cannot express a p below "
                  f"{1 / (1 + len(null_pairs)):.3g}; 19 is the fewest that reaches 0.05",
                  file=sys.stderr)  # fmt: skip
    workers = args.workers or max(1, __import__("os").cpu_count() // 2)
    workers = min(workers, len(jobs))
    if workers == 1:
        fits = [_fit_one(j) for j in _progress(jobs, args.verb)]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            fits = list(_progress(pool.map(_fit_one, jobs), args.verb, total=len(jobs)))
    null_fits = fits[n_real:]
    fits = fits[:n_real]
    nulls_by_roi: dict[tuple[str, str], list] = {(f.roi, f.hemi): [] for f in fits}
    for f in null_fits:
        nulls_by_roi.setdefault((f.roi, f.hemi), []).append(f)
    written = write_results(
        args.prefix, fits,
        provenance={"cond1": args.cond1, "cond2": args.cond2, "duration": args.duration,
                    "K": ks, "psf": psf, "psf_matrix": args.psf_matrix,
                    "psf_kernel": args.psf_kernel, "psf_variance": args.psf_variance,
                    "psf_raw": args.psf_raw, "models": names,
                    "padding": {"rows": args.pad_rows, "sd": args.pad_sd, "seed": args.pad_seed,
                                "hyperprior": args.hyperprior}, "free_energy": args.free_energy,
                    "delay_seconds": "TR/2" if delay is None else delay, "TE": args.TE,
                    "modulation": modulation, "free_params": args.free_params,
                    "null_fit_prefix": args.null_fit_prefix,
                    "null_replicates": len(null_pairs),
                    "B0": args.B0, "p0": "reference driver (V0t 3, nr 3, al_v 0.35, w_v 0.5)"},
    )  # fmt: skip
    if null_pairs:
        from fastfuncstuff.laminar.pipeline import write_null

        written.extend(write_null(args.prefix, fits, nulls_by_roi))
    if args.plots:
        from fastfuncstuff.laminar.plots import plot_curves, plot_roi, plot_summary

        conds = tuple(args.conds) if args.conds else ("cond 1", "cond 2")
        for f in fits:
            tag = f"{f.name}_{f.hemi}".replace("/", "-").replace(" ", "_")
            written.append(plot_roi(f, f"{args.prefix}_laminar_{tag}.png", conds))
            written.append(plot_curves(f, f"{args.prefix}_laminar_{tag}_curves.png", conds))
            if nulls_by_roi.get((f.roi, f.hemi)):
                from fastfuncstuff.laminar.plots import plot_null

                written.append(plot_null(f, nulls_by_roi[(f.roi, f.hemi)],
                                         f"{args.prefix}_laminar_{tag}_null.png"))  # fmt: skip
        if len(fits) > 1:
            written.append(plot_summary(fits, f"{args.prefix}_laminar_summary.png", conds))
    for f in fits:
        for w in f.warnings:
            print(f"ffs_laminar: WARNING {f.name} ({f.hemi}): {w}", file=sys.stderr)
    if args.verb >= 1:
        for f in fits:
            print(f"  {f.name} ({f.hemi}): best '{f.models[f.best]}' "
                  f"p={f.p_model[f.best]:.3f}; P(modulated) "
                  + ", ".join(f"{d} {p:.2f}" for d, p in zip(_names(args.N), f.p_depth, strict=True)))  # fmt: skip
            if nulls_by_roi.get((f.roi, f.hemi)):
                from fastfuncstuff.laminar.pipeline import null_statistics

                st = null_statistics(f, nulls_by_roi[(f.roi, f.hemi)])
                real, pp, draws = st["logBF_vs_null"]
                print(f"    vs {len(draws)} permutation(s): logBF {real:.1f} against a null "
                      f"median {np.median(draws):.1f} (max {np.max(draws):.1f}), "
                      f"p_perm {pp:.3g}")  # fmt: skip
        for path in written:
            print(f"  wrote {path}")
        print(f"ffs_laminar: done in {time.time() - t0:.0f} s")
    return 0


def parse_modulation(words: list[str]):
    """``-modulation``: 'event', 'reference', or START DUR (seconds)."""
    if len(words) == 1 and words[0] in ("event", "reference"):
        return words[0]
    if len(words) == 2:
        try:
            return (float(words[0]), float(words[1]))
        except ValueError:
            pass
    raise ValueError(f"-modulation takes 'event', 'reference' or START DUR, not {' '.join(words)}")


def _names(n):
    from fastfuncstuff.laminar.pipeline import depth_names

    return depth_names(n)


def _progress(it, verb, total=None):
    if verb < 1:
        return it
    try:
        from tqdm.auto import tqdm

        return tqdm(it, total=total, desc="ROIs", leave=True, disable=(total or 2) < 2)
    except ImportError:  # pragma: no cover
        return it


if __name__ == "__main__":
    sys.exit(main())
