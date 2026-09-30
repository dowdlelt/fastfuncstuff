"""ffs_simulate -- how does this design look? Single-subject power by Monte Carlo.

Describe an experiment (or hand over its timing files), say how noisy the data
will be, and get back what amplitude each contrast needs to be detected, per
tSNR level, plus whether autocorrelation would fool a naive OLS analysis.

TIMING -- either explicit events:
    -events A.txt B.txt -durations 2 2 -nt 240 240
AFNI timing files, one per condition, one row per run, onsets in seconds.

or a described experiment, realized -ndesigns times with fresh jitter/order:
    -trial NAME DUR COUNT         a trial type: COUNT per run, DUR seconds
    -miniblock NAME ITEMS COUNT   items shown in order, each LABEL:DUR[:OFF][xN]
    -block NAME DUR COUNT         a block (one long item, counted as a block)
    -null DUR COUNT               blank trials: time with no event. COUNT per run, or a
                                  share of all units (20% or 0.2); DUR:OFF gives it its
                                  own gap, like a -miniblock item
    -isi SPEC                     default gap between trials/blocks (offset to onset)
    -within_isi SPEC              default gap between items inside a miniblock
    -initial_fix S / -post_fix S  fixation before the first / after the last
SPEC is a fixed number of seconds, uniform:LO,HI, exp:MEAN,MIN,MAX or
poisson:MEAN,MIN,MAX. A block is a -trial with a long duration.

An item's OFF is the gap after it, and takes any SPEC, so every position can
have its own (jittered) gap:
    -miniblock ABC "A:0.5:0, B:2:2, C:3:uniform:2,4" 10
is A 0.5 s, straight into B 2 s, 2 s off, C 3 s, then 2-4 s to the next
miniblock: the LAST item's OFF is the gap to the next unit and overrides -isi.
Items without an OFF use -within_isi (inside) and -isi (after the unit).
"A:1:0.5x10" repeats an item. "null" is time without an event; uniform, exp
and poisson cannot be condition names.

HOW MANY -- each family (events: -trial/-null; blocks: -block/-miniblock) is
fixed by exactly one thing:
    COUNTs alone       as given; run length is whatever they take
    -num_events N      total -trial units per run; COUNTs become weights
                       (-trial A 2 3 -trial B 2 1 -num_events 40 -> 30 A, 10 B),
                       and -null scales along with them
    -num_blocks N      the same for -block/-miniblock units
    -scan_time S       at most S seconds per run. Families without a total
                       are scaled to the whole units that fit, and the run is
                       trimmed to them (noted) when that saves 5% or more --
                       padding with fixation would understate the design per
                       minute. With every family fixed, S sets the run length
                       and leftover time is trailing fixation. Events jitter
                       pushes past the end are dropped and counted.

NOISE -- tSNR levels (-tsnr 20 50 100) with a physiological share and its
correlation time in seconds (-phys_fraction, -tau), or calibrated from a real
dataset's REML outputs (-noise_profile RVAR TSNR): tSNR binned inside the
mask, lowest bin the worst case, with each bin's typical ARMA(1,1).

ANALYSIS -- OLS with per-run Legendre drift. t is corrected for the noise's
known ARMA(1,1) (sandwich variance + Satterthwaite dof), so no per-voxel REML is
needed; naive OLS false positives are reported alongside. 0 is always
simulated, so the false-positive rate is measured.

WHAT IS SWEPT -- -amplitudes (peak % signal change of an isolated event) means
the response amplitude for a condition contrast (A, or any contrast whose
weights do not sum to zero; -pattern sets relative responses), and the
difference itself for a difference contrast (A-B, A+B-2*C): at 1%, A is 1%
above B. -shared X puts every condition at X% underneath the difference
(A = X + d, B = X) -- it cancels under a correct HRF, and with -true_hrf or
-true_delay shows what a large common response costs.

WHAT IT REPORTS -- for each design, at each noise level:
    detection    the effect each contrast needs for -target power, 80% (the classic design
                 efficiency, in % signal), by Monte Carlo and analytically
    estimation   how precisely the response *shape* is recovered (FIR SD per bin;
                 Liu & Frank's estimation efficiency beside it), and the shape
                 resolution: how many steps apart in the ordered 20-HRF library
                 two shapes must be to be told apart (also whether two
                 conditions differ in shape)
    single trials  how well each trial is estimated on its own: LSS, LSA and
                 single-trial ridge, as trial-pattern reliability (-trial_sd)
They trade off: rapid jittered designs estimate well and detect poorly, blocks
the reverse, and single trials want trials spread apart. Design quality (rank,
VIF, how separable each pair of conditions is) is checked first.

SEARCHING DESIGNS -- beyond scoring one design:
    -scan_times S...  how long to scan: effect needed against total minutes
    -compare TSV...   earlier runs side by side, per unit of scan time too
    -explore N        N designs drawn from ranges [a-b] and choices {x,y} written
                      inside the flags; their trade-off, what each range does,
                      and a shortlist. One budget fixed: -scan_time or the counts
    -optimize G       the best realization (order and gaps) of one design, by an
                      evolutionary search averaged over HRF shapes
-objective picks the goal: a contrast or 'detection' (all contrasts), 'shape',
'shape_diff' or 'trials' -- or several, weighted: detection=1,shape=0.5 (each
relative to a typical design). -rank_by takes the same, for _designs.png.
Searches are analytic and assume the fitted HRF; each prints the command that
runs the full Monte Carlo on its result.

Outputs (one design): PREFIX_summary.txt, _power.tsv, _spec.json, _events/, and
figures (-no_plots skips them):
    _power.png     power against effect, with the summary as text
    _tstats.png    t under the null and at the effect, corrected against naive
    _design.png    events, regressors, correlation, the effect each pair needs
    _designs.png   the typical, best and worst realization (-rank_by)
    _spectrum.png  where each contrast's information sits against noise and drift
    _voxels.png    what the data look like, at each noise level
    _tent.png      the response shape one voxel gives (TENT deconvolution)
    _shape.png     shape resolution: power against library steps apart
    _trials.png    single trials by LSS, LSA and ridge, against the truth
    _hrf.png       with -true_hrf: the true shapes and what was recovered
    _robust.png    every library HRF as the truth: effect needed, fraction recovered
    _soa.png       efficiency against SOA: fixed, jittered, with blanks; this design
    _liu.png       Liu's estimation-vs-detection plane, with the theoretical bound
    _tsnr.png      the effect needed against tSNR: what tSNR an effect needs
    _matrix.png    the design matrix, SPM-style
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

EXAMPLES = """\
Examples
--------
Evaluate one design
    # rapid events: two conditions, jittered, a fifth of the slots blank
    ffs_simulate -tr 1 -nruns 2 -scan_time 330 -initial_fix 10 -post_fix 15 \\
        -trial A 0.25 1 -trial B 0.25 1 -null 0.25 20% -isi exp:4,2,12 \\
        -contrast A -contrast A-B -tsnr 30 60 100 -prefix sim/er

    # blocks, shuffled but evenly represented (one of each per group)
    ffs_simulate -tr 1 -nruns 2 -scan_time 330 -initial_fix 10 -post_fix 15 \\
        -block E1 30 1 -block E2 30 1 -isi 10 -order permuted_block \\
        -contrast E1 -contrast E1-E2 -tsnr 30 60 100 -prefix sim/blocks

    # your own timing files, noise calibrated from a real dataset
    ffs_simulate -tr 2 -events A.txt B.txt -durations 2 -nt 240 240 \\
        -noise_profile stats_REMLvar+tlrc.HEAD TSNR+tlrc.HEAD -noise_mask mask+tlrc.HEAD \\
        -contrast A-B -prefix sim/mine

How long to scan, and designs side by side
    ffs_simulate ... -scan_times 150 240 330 480 660 900 -prefix sim/length
    # two runs of the same conditions and contrasts, e.g. two ISI choices
    ffs_simulate -compare sim/er_power.tsv sim/er_slow_power.tsv -prefix sim/cmp

Check a family of designs (quote the ranges; one budget: -scan_time here)
    ffs_simulate -explore 400 -tr 1 -nruns 2 -scan_time 330 -initial_fix 10 -post_fix 15 \\
        -trial E1 0.25 1 -isi "poisson:[1.0-6.0],1,[6-12]" -null 0.25 "[0-50%]" \\
        -order "{random,permuted_block}" -contrast E1 -tsnr 30 60 100 \\
        -objective detection -prefix sim/family
    #   -objective shape     for the response shape (estimation efficiency)
    #   -objective trials    for single trials (LSS / ridge reliability)
    #   a shortlist at the edge of a range is flagged: widen it and run again

Optimize the realization you will actually run
    ffs_simulate -tr 1 -nruns 2 -scan_time 330 -initial_fix 10 -post_fix 15 \\
        -trial E1 0.25 1 -trial E2 0.25 1 -isi exp:4,1,12 -contrast E1-E2 \\
        -tsnr 30 60 100 -objective E1-E2 -optimize 30 -max_repeat 4 -prefix sim/opt
    # ... or -explore 300 ... -optimize 20 to optimize every shortlisted design

    # several goals at once, each relative to a typical design
    ffs_simulate ... -objective "E1-E2=1,shape_diff=1,trials=0.5" -optimize 30 -prefix sim/multi
"""

VERDICTS = ((0.2, "hopeless"), (0.8, "marginal"), (1.01, "good"))
# The scorecard's number per measure, named with its unit for _spec.json / -compare.
SCORE_KEYS = {
    "detection": "detection, % (mean)",
    "per minute": "detection x sqrt(min)",
    "shape precision": "shape SD per bin, %",
    "Liu & Frank": "estimation efficiency",
    "shape resolution": "shape steps apart",
    "single trials": "trial reliability",
    "HRF robustness": "HRFs detectable, frac",
    "false positives": "false positives",
    "collinearity": "largest VIF",
}
SWEEP_COLS = (
    "scan_time",
    "run_s",
    "minutes",
    "counts",
    "design",
    "noise",
    "contrast",
    "needed",
    "per_minute",
)
SWEEP_DESIGNS = 50  # realizations per -scan_times value: the medians settle well before


def _build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_simulate",
        description=__doc__,
        epilog=EXAMPLES,
        formatter_class=FfsHelpFormatter,
    )
    t = p.add_argument_group("Timing")
    t.add_argument("-tr", type=float, help="Repetition time (s); required unless -compare.")
    t.add_argument(
        "-events", nargs="+", metavar="FILE", help="AFNI timing files, one per condition."
    )
    t.add_argument("-labels", nargs="+", help="Condition names for -events (default: file stems).")
    t.add_argument(
        "-durations",
        nargs="+",
        type=float,
        help="Event duration(s) for -events: one, or one per file.",
    )
    t.add_argument("-nt", nargs="+", type=int, help="Timepoints per run for -events.")
    t.add_argument(
        "-trial",
        nargs=3,
        action="append",
        metavar=("NAME", "DUR", "COUNT"),
        help="A trial type (repeatable). A block is a trial with a long DUR.",
    )
    t.add_argument(
        "-miniblock",
        nargs=3,
        action="append",
        metavar=("NAME", "ITEMS", "COUNT"),
        help='Items in order, each LABEL:DUR[:OFF][xN], e.g. "A:0.5:0, B:2:uniform:2,4" '
        "(repeatable). The last item's OFF is the gap to the next unit.",
    )
    t.add_argument(
        "-block",
        nargs=3,
        action="append",
        metavar=("NAME", "DUR", "COUNT"),
        help="A block: one long item, counted with -num_blocks (repeatable).",
    )
    t.add_argument(
        "-null",
        nargs=2,
        action="append",
        metavar=("DUR", "COUNT"),
        help="Blank trials: DUR seconds (DUR:OFF for its own gap, as in a -miniblock item), "
        "COUNT per run, or a share of all units -- 20%% or 0.2 -- as a weight (repeatable). "
        "With a fixed -isi this is 'identical ISIs, then drop trials'.",
    )
    t.add_argument("-isi", default="0", metavar="SPEC", help="Gap between units (default 0).")
    t.add_argument("-within_isi", default="0", metavar="SPEC", help="Gap inside a miniblock.")
    t.add_argument("-initial_fix", type=float, default=0.0, help="Fixation before (s).")
    t.add_argument("-post_fix", type=float, default=16.0, help="Fixation after (s, default 16).")
    t.add_argument(
        "-order",
        default="random",
        choices=["random", "alternating", "blocked", "permuted_block"],
        help="Order of units within a run.",
    )
    t.add_argument("-nruns", type=int, default=1, help="Runs (described experiment).")
    t.add_argument(
        "-tr_lock",
        action="store_true",
        help="Every onset on a TR boundary: each gap is drawn as usual and the next onset "
        "snaps to the nearest boundary (later if it would overlap), so gaps come in TR steps "
        "with their mean kept. Described experiments; -explore/-optimize respect it.",
    )
    t.add_argument(
        "-scan_time",
        type=float,
        metavar="S",
        help="Seconds per run, at most: trimmed to the units that fit (see HOW MANY).",
    )
    t.add_argument("-num_events", type=int, metavar="N", help="-trial units per run in total.")
    t.add_argument("-num_blocks", type=int, metavar="N", help="-block/-miniblock units per run.")
    t.add_argument(
        "-scan_times",
        nargs="+",
        type=float,
        metavar="S",
        help="Also sweep the per-run scan time over these values (analytic, fitted HRF "
        "assumed right): _scantime.png/.tsv and a summary table of the effect needed and "
        "effect x sqrt(minutes) -- how long to scan.",
    )
    t.add_argument(
        "-ndesigns",
        type=int,
        default=20,
        help="Realizations of a described experiment (default 20).",
    )

    n = p.add_argument_group("Noise")
    n.add_argument("-tsnr", nargs="+", type=float, help="tSNR levels (default 20 50 100).")
    n.add_argument(
        "-phys_fraction",
        type=float,
        default=0.5,
        help="Physiological share of the noise variance for -tsnr (default 0.5).",
    )
    n.add_argument(
        "-tau",
        type=float,
        default=6.0,
        help="Correlation time of the physiological noise, seconds (default 6).",
    )
    n.add_argument(
        "-noise_profile",
        nargs=2,
        metavar=("RVAR", "TSNR"),
        help="Calibrate noise from REML outputs: an Rvar and a tSNR map.",
    )
    n.add_argument("-noise_mask", metavar="MASK", help="Mask for -noise_profile.")
    n.add_argument("-noise_bins", type=int, default=5, help="tSNR quantile bins (default 5).")
    n.add_argument(
        "-profile_tr", type=float, help="TR of the -noise_profile data if it differs from -tr."
    )

    e = p.add_argument_group("Effects")
    e.add_argument(
        "-amplitudes",
        nargs="+",
        default=["0.1:3:15"],
        help="Peak %% signal change to sweep: values, or START:STOP:NUM.",
    )
    e.add_argument(
        "-effect", type=float, help="Report a verdict at this amplitude (added to the sweep)."
    )
    e.add_argument(
        "-pattern",
        nargs="+",
        metavar="COND=W",
        help="Response per unit amplitude, e.g. A=1 B=0 (default all 1).",
    )
    e.add_argument(
        "-contrast",
        action="append",
        metavar="EXPR",
        help="A contrast such as A-B or A+B-2*C (repeatable). Default: each "
        "condition and every pairwise difference.",
    )
    e.add_argument(
        "-shared",
        type=float,
        default=0.0,
        metavar="PSC",
        help="Difference contrasts (A-B) sweep the difference itself; -shared puts every "
        "condition at this %% underneath it (A = shared + d, B = shared). Cancels exactly "
        "under a correct HRF; with -true_hrf/-true_delay it shows what a large common "
        "response costs.",
    )
    e.add_argument(
        "-trial_sd",
        type=float,
        default=0.5,
        metavar="PSC",
        help="Trial-to-trial SD of the response (%% signal) for single-trial reliability; the "
        "mean is -effect, else 1%% (default 0.5).",
    )
    e.add_argument(
        "-hrf",
        default="spmg1",
        help="HRF the GLM fits: spmg1 (default) or lib:K, one of the 20-HRF library.",
    )
    e.add_argument(
        "-true_hrf",
        default="same",
        help="HRF that generates the data: same (default), spmg1, lib:K, or lib:all to "
        "sweep the library as the truth (reports how much amplitude is recovered).",
    )
    e.add_argument(
        "-true_delay",
        type=float,
        default=0.0,
        help="Generate the response this many seconds late (fit nominal onsets).",
    )

    a = p.add_argument_group("Analysis")
    a.add_argument("-nreps", type=int, default=500, help="Replicates per cell (default 500).")
    a.add_argument("-alpha", type=float, default=0.001, help="Two-tailed p threshold (0.001).")
    a.add_argument("-polort", type=int, help="Per-run drift degree (default AFNI 1+floor(s/150)).")
    a.add_argument("-seed", type=int, default=0)
    a.add_argument(
        "-target",
        type=float,
        default=0.8,
        help="Power every 'effect needed' is for: the summary, design quality, the searches "
        "and -compare (default 0.8).",
    )
    from fastfuncstuff.cli_utils import add_device_arg

    add_device_arg(a)
    p.add_argument("-prefix", required=True, help="Output prefix (a directory is created).")
    p.add_argument("-no_plots", action="store_true", help="Skip the PNG figures.")
    p.add_argument(
        "-rank_by",
        metavar="GOAL",
        help="What makes a realization best or worst in _designs.png: a contrast, "
        "'detection' (all contrasts), 'shape' or 'trials' -- as -objective (default: the "
        "mean effect every condition and pair needs).",
    )

    c = p.add_argument_group("Compare")
    c.add_argument(
        "-compare",
        nargs="+",
        metavar="TSV",
        help="Compare earlier runs instead of simulating: their PREFIX_power.tsv files, one "
        "design each, summarised over that design's realizations. Writes "
        "PREFIX_compare.txt and one PREFIX_compare_<contrast>.png per shared contrast.",
    )
    c.add_argument(
        "-compare_names",
        nargs="+",
        metavar="NAME",
        help="Names for the -compare designs (default: the file prefixes).",
    )

    x = p.add_argument_group("Explore")
    x.add_argument(
        "-explore",
        type=int,
        metavar="N",
        help="Score N designs drawn from the ranges [a-b] and choices {x,y} written inside "
        "the other flags (quote them), e.g. -isi 'exp:[3-8],[1-3],12' -null 0.25 '[0-40%%]' "
        "-order '{random,permuted_block}'. One budget must be fixed: -scan_time (time) or "
        "the trial counts (-num_events/-num_blocks or plain COUNTs), not both. Analytic "
        "(fitted HRF assumed right). -objective trials targets single-trial estimability. "
        "Writes _explore.tsv/.png/_summary.txt and, for a "
        "shortlist off the detection-vs-estimation Pareto front, the best realization's "
        "timing files with the commands that reproduce everything.",
    )
    x.add_argument(
        "-objective",
        metavar="GOAL",
        help="What to optimize, with -explore/-optimize. Detection: a contrast's name "
        "(default: the first with a true effect) or 'detection' (alias 'efficiency', meaning "
        "detection efficiency -- not Liu & Frank's estimation efficiency, which is 'shape': the mean "
        "over all contrasts). Estimation: 'shape' (response shape, FIR). Single trials: "
        "'trials' (trial-pattern reliability, the better of LSS and ridge; see -trial_sd); "
        "'shape_diff' (how many library steps apart two response shapes must be to be told "
        "apart -- also whether two conditions differ in shape). "
        "The explorer trades detection against shape, or against trials for 'trials'.",
    )
    x.add_argument(
        "-explore_designs", type=int, default=3, help="Realizations scored per design (3)."
    )
    x.add_argument("-explore_keep", type=int, default=5, help="Shortlisted designs (5).")
    x.add_argument(
        "-explore_pick",
        type=int,
        default=50,
        help="Realizations searched for each shortlisted design's best one (50).",
    )
    x.add_argument(
        "-optimize",
        type=int,
        metavar="G",
        help="Search G generations for the best realization of this design (an evolutionary "
        "search over orders and gaps, the recipe kept exact) on -objective, instead of "
        "simulating it; with -explore, for each shortlisted design instead of best-of-N. "
        "Scored over several HRFs (-optimize_hrfs) and checked on held-out ones: optimized "
        "for one HRF, a design lost to a median random draw under another.",
    )
    x.add_argument("-optimize_pop", type=int, default=30, help="Population per generation (30).")
    x.add_argument(
        "-optimize_hrfs",
        nargs="+",
        metavar="HRF",
        help="HRFs the search averages over (default: the fitted one and lib:0 lib:6 lib:13 "
        "lib:19, fast to slow).",
    )
    x.add_argument(
        "-max_repeat",
        type=int,
        metavar="K",
        help="With -optimize: at most K consecutive units of one kind.",
    )
    return p


def _amplitudes(tokens: list[str], effect: float | None) -> list[float]:
    vals: list[float] = []
    for tok in tokens:
        if ":" in tok:
            start, stop, num = tok.split(":")
            vals += list(np.linspace(float(start), float(stop), int(num)))
        else:
            vals.append(float(tok))
    if effect is not None:
        vals.append(effect)
    return sorted({round(v, 6) for v in vals})


def _explicit_realization(args) -> Any:
    from fastfuncstuff.io.afni import read_afni_onset_files
    from fastfuncstuff.simulation.experiment import Realization

    onsets = read_afni_onset_files(args.events)
    labels = args.labels or [Path(f).name.split(".")[0] for f in args.events]
    if len(labels) != len(args.events):
        raise ValueError("-labels needs one name per -events file")
    if not args.nt:
        raise ValueError("-events needs -nt (timepoints per run)")
    n_runs = len(onsets[0])
    nt = args.nt * n_runs if len(args.nt) == 1 else args.nt
    if len(nt) != n_runs or any(len(o) != n_runs for o in onsets):
        raise ValueError(f"timing files and -nt disagree on the number of runs ({n_runs})")
    durs = args.durations or [0.0]
    durs = durs * len(labels) if len(durs) == 1 else durs
    if len(durs) != len(labels):
        raise ValueError("-durations needs one value, or one per -events file")
    return Realization(0, labels, list(durs), onsets, list(nt), [n * args.tr for n in nt])


def _null_fraction(token: str) -> float | None:
    """'20%' or '0.2' -> 0.2; a whole number (a count) -> None."""
    text = str(token).strip()
    if not text.endswith("%") and "." not in text:
        return None
    frac = float(text.rstrip("%")) / (100.0 if text.endswith("%") else 1.0)
    if not 0 <= frac < 1:
        raise ValueError(f"-null {token!r}: a fraction must be between 0 and 1 (or 0-100%)")
    return frac


def _spec_from_args(args):
    from fastfuncstuff.simulation.experiment import NULL, ExperimentSpec, Interval, Unit

    units = [Unit.parse(nm, f"{nm}:{d}", int(c)) for nm, d, c in (args.trial or [])]
    units += [Unit.parse(nm, f"{nm}:{d}", int(c), "block") for nm, d, c in (args.block or [])]
    units += [Unit.parse(nm, items, int(c), "block") for nm, items, c in (args.miniblock or [])]
    # A -null COUNT that is a fraction ("20%", "0.2") is that share of all units:
    # weight p / (1 - p) against the rest, so it survives -num_events/-scan_time
    # scaling. A whole number is a count.
    rest = sum(u.count for u in units)
    for i, (d, c) in enumerate(args.null or []):
        frac = _null_fraction(c)
        if frac == 0:  # an explored share can draw 0%: no blank trials
            continue
        weight = frac / (1 - frac) * rest if frac is not None else int(c)
        units.append(Unit.parse(f"null{i}", f"{NULL}:{d}", weight))
    return ExperimentSpec(
        tr=args.tr,
        units=units,
        n_runs=args.nruns,
        isi=Interval.parse(args.isi),
        within_isi=Interval.parse(args.within_isi),
        initial_fix=args.initial_fix,
        post_fix=args.post_fix,
        order=args.order,
        scan_time=args.scan_time,
        num_events=args.num_events,
        num_blocks=args.num_blocks,
        tr_lock=args.tr_lock,
    )


def _noise_conditions(args) -> tuple[list[dict[str, Any]], str]:
    conds: list[dict[str, Any]] = []
    text = ""
    if args.noise_profile:
        from fastfuncstuff.simulation.calibrate import noise_profile_from_reml

        q = tuple(np.linspace(0, 1, args.noise_bins + 1))
        prof = noise_profile_from_reml(
            args.noise_profile[0],
            args.noise_profile[1],
            tr=args.profile_tr or args.tr,
            mask=args.noise_mask,
            quantiles=q,
        )
        text = prof.summary()
        for b in prof.bins:
            kw = b.simulation_kwargs(tr=args.tr)
            kw["label"] = (
                f"tSNR {b.tsnr:.0f} (p{b.quantile_range[0]:.0%}-{b.quantile_range[1]:.0%})"
            )
            conds.append(kw)
    tsnrs = args.tsnr if args.tsnr else ([] if args.noise_profile else [20.0, 50.0, 100.0])
    for t in tsnrs:
        conds.append(
            {
                "label": f"tSNR {t:g}",
                "tsnr": t,
                "phys_fraction": args.phys_fraction,
                "tau": args.tau,
            }
        )
    return conds, text


def _pattern(tokens: list[str] | None, conditions: list[str]) -> list[float]:
    w = [1.0] * len(conditions)
    for tok in tokens or []:
        name, _, val = tok.partition("=")
        if name not in conditions or not val:
            raise ValueError(f"-pattern {tok!r}: use COND=W with COND in {conditions}")
        w[conditions.index(name)] = float(val)
    return w


def _scorecard(res, reals, conds, contrasts, pattern, args, quality, steps, robust):
    """One value per measure at the reference noise level: [(name, text, number)].

    Median over realizations throughout. The number (for _spec.json and
    -compare) is the headline one; the text says what it is.
    """
    from fastfuncstuff.simulation.power import effect_needed, has_true_effect

    ref = _reference_noise(conds)
    live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
    minutes = float(np.mean([sum(r.run_lengths) for r in reals])) * args.tr / 60
    out = []
    need = effect_needed(res, args.target)
    det = {
        c: float(np.nanmedian(need[(ref, c)])) for c in live if np.isfinite(need[(ref, c)]).any()
    }
    if live and not det:  # say so, rather than leave the measure out
        top = max(abs(x["true_effect"]) for x in res["table"])
        out.append(("detection", f"> {top:g}% for every contrast: {args.target:.0%} power not "
                    "reached within the sweep (raise -amplitudes)", float("inf")))  # fmt: skip
        out.append(("  per minute", "-", float("inf")))
    if det:
        out.append(
            (
                "detection",
                ", ".join(f"{c} {v:.2f}%" for c, v in det.items())
                + f"  (% signal for {args.target:.0%} power)",
                float(np.mean(list(det.values()))),
            )
        )
        out.append(
            (
                "  per minute",
                ", ".join(f"{c} {v * np.sqrt(minutes):.2f}" for c, v in det.items())
                + f"  (effect x sqrt({minutes:.1f} min); lower = more per minute)",
                float(np.mean(list(det.values()))) * float(np.sqrt(minutes)),
            )
        )
    qs = [q for q in quality if "shape_sd" in q]
    if qs:
        sd = float(np.median([np.mean(q["shape_sd"][ref]) for q in qs]))
        xi = float(np.median([q["xi"] for q in qs]))
        lp = float(np.median([q.get("liu_power", np.nan) for q in qs]))
        out.append((
            "shape precision",
            f"{sd:.2f}% SD per FIR bin" if np.isfinite(sd) else "not estimable (the FIR "
            "lags alias: too few events, or a fixed SOA)",
            sd,
        ))  # fmt: skip
        out.append(("Liu & Frank", f"estimation efficiency {xi:.2f}, detection power {lp:.2f} "
                    "(fractions of their bounds)", xi))  # fmt: skip
    if steps is not None:
        st = steps["steps"][ref]
        fin = st[np.isfinite(st)]
        if fin.size:
            m = float(np.mean(fin))
            out.append(("shape resolution", f"{m:.1f} library steps (~{0.16 * m:.1f} s of peak "
                        "latency) to tell two shapes apart", m))  # fmt: skip
        else:
            out.append(("shape resolution", f"> {steps['max_step']} library steps: shapes "
                        "cannot be told apart", float("inf")))  # fmt: skip
    ss = [q["single"]["reliability"][ref] for q in quality if "single" in q]
    if ss:
        best = {k: float(np.median([r[k] for r in ss])) for k in ("lss", "lsa", "ridge")}
        top = max(("lss", "ridge"), key=lambda k: best[k])
        out.append(
            (
                "single trials",
                f"reliability {best[top]:.2f} by {top.upper() if top == 'lss' else 'ridge'} "
                f"(LSS {best['lss']:.2f}, LSA {best['lsa']:.2f}, ridge {best['ridge']:.2f}; trial "
                f"SD {args.trial_sd:g}%)",
                best[top],
            )
        )
    if robust:
        c0 = live[0]
        r = robust["contrasts"][c0]
        n_ok = int(np.sum(np.isfinite(r["needed"])))
        ok = [v for v in r["needed"] if np.isfinite(v)]
        ratio = float(np.median(ok) / r["fitted"]) if ok and r["fitted"] > 0 else float("nan")
        out.append(
            (
                "HRF robustness",
                f"{c0} detectable under {n_ok}/{len(r['needed'])} library HRFs (fitting "
                f"{args.hrf}); median cost x{ratio:.2f} of the right HRF's",
                n_ok / len(r["needed"]),
            )
        )
    nulls = [x for x in res["table"] if x["noise"] == ref and x["amplitude"] == 0.0]
    if nulls:
        fp = float(np.mean([x["power"] for x in nulls]))
        fpn = float(np.mean([x["power_naive"] for x in nulls]))
        out.append(("false positives", f"{fp:.4f} corrected, {fpn:.4f} naive OLS (nominal "
                    f"{args.alpha:g})", fp))  # fmt: skip
    vif = float(np.median([np.max(q["vif"]) for q in quality]))
    out.append(("collinearity", f"largest VIF {vif:.2f} (1 = orthogonal, > 5 hard)", vif))
    return out


def _needed_cell(vals: np.ndarray, has_effect: bool, top: float) -> tuple[str, bool]:
    """'median [min-max]' of the effect needed over realizations, and whether some never got there.

    One formatter for the text summary and the summary panel of _power.png.
    """
    if not has_effect:
        return "no true effect", False
    if np.all(np.isnan(vals)):
        return f"> {top:g}", False
    text = f"{np.nanmedian(vals):.2f} [{np.nanmin(vals):.2f}-{np.nanmax(vals):.2f}]"
    partial = bool(np.isnan(vals).any())
    return text + ("*" if partial else ""), partial


def _figure_summary(res, reals, conds, contrasts, pattern, args, quality, spec, card=None) -> dict:
    """The headline facts and the answer table, for the text panel of _power.png."""
    from fastfuncstuff.simulation.power import effect_needed, has_mismatch, has_true_effect

    rows = res["table"]
    top = max(r["amplitude"] for r in rows)
    names = reals[0].conditions
    n_ev = [
        float(np.mean([sum(len(o) for o in r.onsets[i]) for r in reals])) for i in range(len(names))
    ]
    total_s = float(np.mean([sum(r.run_lengths) for r in reals])) * args.tr
    run_s = float(np.mean(reals[0].run_lengths)) * args.tr
    facts = [
        (
            "events",
            ", ".join(f"{c} {n:g}" for c, n in zip(names, n_ev, strict=True))
            + " (all runs, per realization)",
        ),
        (
            "scan",
            f"TR {args.tr:g} s, {len(reals[0].run_lengths)} run(s) x {run_s:g} s = "
            f"{total_s / 60:.1f} min; {len(reals)} realization(s)"
            + (f", order {args.order}" if spec is not None else ""),
        ),
    ]
    if args.noise_profile:
        facts.append(
            (
                "noise",
                f"calibrated from {Path(args.noise_profile[0]).name} ({args.noise_bins} tSNR bins)",
            )
        )
    else:
        facts.append(
            (
                "noise",
                f"physiological {args.phys_fraction:.0%} of the variance, "
                f"tau {args.tau:g} s; the rest white",
            )
        )
    fit, truths = res.get("hrf", "spmg1"), res.get("true_hrfs", [])
    model = f"{fit} fitted, polort {quality[0]['poly_degree']}, two-tailed p < {args.alpha:g}"
    if truths and truths != [fit]:
        shown = truths if len(truths) <= 3 else [f"{len(truths)} library HRFs"]
        model += f"; data from {', '.join(shown)}"
    facts.append(("model", model + "; t corrected for the noise ARMA"))
    vif = np.median([q["vif"] for q in quality], axis=0)
    facts.append(("VIF", ", ".join(f"{c} {v:.2f}" for c, v in zip(names, vif, strict=True))))
    ref = _reference_noise(conds)
    shape = _shape_text(quality, names, ref)
    if shape is not None:
        xi = float(np.median([q["xi"] for q in quality]))
        facts.append(
            (
                "shape",
                f"{shape} SD per {args.tr:g} s bin at {ref} (FIR, "
                f"{quality[0]['fir_lags']} bins); Liu & Frank estimation efficiency {xi:.2f} "
                "of its bound",
            )
        )
    single = _single_text(quality, names, ref)
    if single is not None:
        facts.append(("trials", f"{single}; SD per trial at {ref}"))
    if len(names) > 2:
        m = np.median([q["needed"][ref] for q in quality], axis=0)
        pairs = sorted((m[i, j], i, j) for i in range(len(names)) for j in range(i))
        facts.append(
            (
                "pairs",
                f"hardest {names[pairs[-1][1]]}-{names[pairs[-1][2]]} "
                f"({pairs[-1][0]:.2f}%), easiest {names[pairs[0][1]]}-"
                f"{names[pairs[0][2]]} ({pairs[0][0]:.2f}%) at {ref}",
            )
        )
    notes = [
        ln.strip().removeprefix("note: ")
        for ln in (spec.describe() if spec else "").splitlines()
        if ln.strip().startswith("note:")
    ]
    need = effect_needed(res, args.target)
    header = ["noise", *contrasts, "false pos. corr / naive"]
    table, partial = [], False
    for cond in conds:
        cells = [cond["label"]]
        for c in contrasts:
            text, part = _needed_cell(
                need[(cond["label"], c)], has_true_effect(contrasts[c], pattern), top
            )
            partial |= part
            cells.append(text)
        nulls = [r for r in rows if r["noise"] == cond["label"] and r["amplitude"] == 0.0]
        cells.append(
            f"{np.mean([r['power'] for r in nulls]):.4f} / "
            f"{np.mean([r['power_naive'] for r in nulls]):.4f}"
        )
        table.append(cells)
    foot = [
        f"effect for {args.target:.0%} power, % signal change -- the contrast's true value "
        "(a condition's response, or the difference for A-B); median [range] over realizations"
        + ("; Monte Carlo, the fitted HRF is wrong" if has_mismatch(rows) else "")
    ]
    if partial:
        foot.append(
            f"* some realizations never reach {args.target:.0%} within the sweep (max {top:g}%)"
        )
    if card:  # the scorecard first: one value per measure (it covers shape and trials)
        facts = [f for f in facts if f[0] not in ("shape", "trials")]
        facts = [(k.strip(), v) for k, v, _ in card] + [("", "")] + facts
    return {"facts": facts, "notes": notes, "header": header, "rows": table, "footer": foot}


def _verdict(power: float) -> str:
    return next(label for cut, label in VERDICTS if power < cut)


def _effect_cell(sel: list[dict[str, Any]], column: str) -> str:
    """'analytic / Monte Carlo  verdict' for one noise x contrast at -effect.

    The verdict follows ``column`` (see :func:`power_column`): under an HRF
    mismatch the analytic power overstated Monte Carlo 13x (0.137 vs 0.010),
    enough to call a hopeless design marginal.
    """
    if abs(sel[0]["true_effect"]) < 1e-12:
        return "no true effect"
    pa = float(np.median([r["power_predicted"] for r in sel]))
    pm = float(np.mean([r["power"] for r in sel]))
    return f"{pa:>6.2f} / {pm:.2f} {_verdict(pm if column == 'power' else pa):>9}"


def _summarise(
    res,
    reals,
    conds,
    contrasts,
    pattern,
    args,
    spec_text,
    profile_text,
    quality,
    steps=None,
    card=None,
) -> str:
    from fastfuncstuff.simulation.power import effect_needed, has_true_effect, is_difference

    rows = res["table"]
    out = ["ffs_simulate", "=" * 72]
    if card:
        out += [f"Scorecard at {_reference_noise(conds)} (one value per measure; details below):"]
        out += [f"  {k:<18} {v}" for k, v, _ in card]
        out += [""]
    if spec_text:
        out += [spec_text, f"{len(reals)} realization(s)"]
    durs = np.array([sum(r.run_durations) for r in reals])
    out.append(
        f"total scan time per realization: {durs.mean():.0f} s (range {durs.min():.0f}-{durs.max():.0f}), "
        f"runs {reals[0].run_lengths} TRs in the first"
    )
    n_ev = {
        c: [sum(len(o) for o in r.onsets[i]) for r in reals]
        for i, c in enumerate(reals[0].conditions)
    }
    dropped = sum(r.n_dropped for r in reals)
    if dropped:
        out.append(
            f"note: {dropped} event(s) over {len(reals)} realization(s) did not end -post_fix "
            "before the end of the scan and were dropped, whole units at a time -- jitter "
            "that is not mean-matched (uniform) makes some runs longer than the average"
        )
    out.append("events per condition: " + ", ".join(f"{c} {np.mean(v):g}" for c, v in n_ev.items()))
    out += ["", *_quality_lines(quality, reals[0].conditions, conds, args.target)]
    if steps is not None:
        ref = _reference_noise(conds)
        st_txt = ", ".join(
            f"{c} {'> ' + str(steps['max_step']) if not np.isfinite(v) else f'{v:.1f}'}"
            for c, v in zip(reals[0].conditions, steps["steps"][ref], strict=True)
        )
        out.append(
            f"  shape resolution at {ref}: library steps apart for {args.target:.0%} power to "
            f"tell two shapes apart -- {st_txt} (about 0.16 s of peak latency per step; also "
            "how different two conditions' shapes must be to be told apart)"
        )
    if len(reals[0].run_lengths) < 2 and any("single" in q for q in quality):
        out.append(
            "  note: one run -- single-trial ridge chooses its fraction by cross-validation "
            "across runs, so with one run the ridge figure is an optimistic oracle"
        )
    cond_c = [c for c in contrasts if not is_difference(contrasts[c])]
    diff_c = [c for c in contrasts if is_difference(contrasts[c])]
    if cond_c:
        out.append(
            f"condition contrasts ({', '.join(cond_c)}): sweep = response amplitude, pattern "
            + ", ".join(f"{c}={w:g}" for c, w in zip(reals[0].conditions, pattern, strict=True))
        )
    if diff_c:
        out.append(
            f"difference contrasts ({', '.join(diff_c)}): sweep = the difference itself, every "
            f"condition at {res.get('shared', 0.0):g}% shared underneath"
        )
    truths = res.get("true_hrfs", [res.get("hrf", "spmg1")])
    fit = res.get("hrf", "spmg1")
    if truths != [fit]:
        shown = truths if len(truths) <= 4 else [f"{len(truths)} library HRFs"]
        out.append(f"fitted HRF {fit}; data generated with {', '.join(shown)}")
    else:
        out.append(f"HRF {fit} (generated and fitted)")
    if args.true_delay:
        out.append(f"true response delayed {args.true_delay:g} s relative to the fitted model")
    if profile_text:
        out += ["", profile_text]
    out += ["", f"threshold: two-tailed p < {args.alpha:g}; {args.nreps} replicates per cell", ""]

    # Amplitude needed, from the analytic curve, across realizations.
    from fastfuncstuff.simulation.power import has_mismatch, power_column

    over = "realizations" + (" x true HRFs" if len(truths) > 1 else "")
    if has_mismatch(rows):
        over += "; Monte Carlo power, since the fitted HRF is wrong"
    out.append(
        "Effect (% signal change: the contrast's true value -- a condition's response, or "
        f"the difference for A-B) for {args.target:.0%} power -- "
        f"median [range] over {over}"
    )
    need = effect_needed(res, args.target)
    unreached = False
    header = f"{'noise':<24}" + "".join(f"{c:>18}" for c in contrasts)
    out.append(header)
    for cond in conds:
        cells = []
        for c in contrasts:
            text, partial = _needed_cell(
                need[(cond["label"], c)],
                has_true_effect(contrasts[c], pattern),
                max(r["amplitude"] for r in rows),
            )
            unreached |= partial
            cells.append(f"{text:>18}")
        out.append(f"{cond['label']:<24}" + "".join(cells))
    if unreached:
        out.append(
            f"  * some realizations/HRFs never reach {args.target:.0%} within the sweep (max "
            f"{max(r['amplitude'] for r in rows):g}%); the median and range leave them out"
        )

    if truths != [fit] or args.true_delay:
        top = max(r["amplitude"] for r in rows)
        out += [
            "",
            f"Recovered fraction of the true amplitude (mean estimate / truth, {fit} fitted):",
        ]
        for c in contrasts:
            if not has_true_effect(contrasts[c], pattern):
                continue
            fr = {
                th: np.mean(
                    [
                        r["expected_est"] / r["true_effect"]
                        for r in rows
                        if r["contrast"] == c and r["true_hrf"] == th and r["amplitude"] == top
                    ]
                )
                for th in truths
            }
            vals = np.array(list(fr.values()))
            detail = (
                "  ".join(f"{k} {v:.2f}" for k, v in fr.items())
                if len(fr) <= 6
                else (
                    f"median {np.median(vals):.2f}, range {vals.min():.2f}-{vals.max():.2f}; worst "
                    + ", ".join(f"{k} {fr[k]:.2f}" for k in sorted(fr, key=fr.get)[:3])
                )
            )
            out.append(f"  {c:<10} {detail}")

    # False positives at amplitude 0.
    # The rate checks the ARMA correction. Under a mismatch with a shared response
    # a difference contrast's zero-difference estimate is biased by the misfit --
    # real false positives, but of another kind; averaged in, they made the
    # correction look broken. They get their own line below.
    split = bool(res.get("shared", 0.0)) and has_mismatch(rows) and bool(cond_c)
    out += [
        "",
        "False-positive rate at amplitude 0 (should be ~alpha)"
        + (": condition contrasts; differences below" if split else ":"),
    ]
    for cond in conds:
        nulls = [r for r in rows if r["noise"] == cond["label"] and r["amplitude"] == 0.0]
        calib = [r for r in nulls if r["contrast"] not in diff_c] if split else nulls
        fp = np.mean([r["power"] for r in calib])
        fpn = np.mean([r["power_naive"] for r in calib])
        flag = (
            "  <- naive OLS is anticonservative here"
            if fpn > 3 * args.alpha and fpn > fp * 2
            else ""
        )
        out.append(f"  {cond['label']:<24} corrected {fp:.4f}   naive OLS {fpn:.4f}{flag}")
        for c in diff_c:
            fp_c = np.mean([r["power"] for r in nulls if r["contrast"] == c])
            if fp_c > 3 * args.alpha and res.get("shared", 0.0):
                out.append(
                    f"    {c}: {fp_c:.4f} at zero difference -- the {res['shared']:g}% shared "
                    "response, fitted with the wrong HRF, reads as a difference"
                )

    if args.effect is not None:
        column = power_column(rows)
        judged = "Monte Carlo" if column == "power" else "analytic"
        out += [
            "",
            f"At {args.effect:g}% signal change -- amplitude, or difference for A-B "
            f"(power, analytic / Monte Carlo; verdict from {judged}):",
        ]
        out.append(f"{'noise':<24}" + "".join(f"{c:>22}" for c in contrasts))
        for cond in conds:
            cells = []
            for c in contrasts:
                sel = [
                    r
                    for r in rows
                    if r["noise"] == cond["label"]
                    and r["contrast"] == c
                    and abs(r["amplitude"] - args.effect) < 1e-6
                ]
                cells.append(f"{_effect_cell(sel, column):>22}")
            out.append(f"{cond['label']:<24}" + "".join(cells))
        out.append("  hopeless < 0.2 <= marginal < 0.8 <= good")
    return "\n".join(out)


def _voxel_amplitudes(
    res, conds, contrasts, pattern, effect, target: float = 0.8
) -> tuple[list[float], str, list[str] | None]:
    """Amplitude for each noise row of the example-voxel figure, what it is, and row notes.

    -effect when given. Otherwise the effect each noise level needs for ``target``
    power (median over realizations), on the first contrast with a true effect
    -- a fixed default (1%) was below detectability at tSNR 50 and invisible
    against tSNR 20's noise, so the picture showed nothing the design could
    find. A level that never reaches ``target`` shows the top of the sweep.
    """
    from fastfuncstuff.simulation.power import effect_needed, has_true_effect

    if effect is not None:
        return [effect] * len(conds), f"{effect:g}% (-effect)", None
    live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
    if not live:
        return [1.0] * len(conds), "1% (no contrast has a true effect)", None
    need = effect_needed(res, target)
    top = max(r["amplitude"] for r in res["table"])
    amps, notes = [], []
    for cond in conds:
        v = need.get((cond["label"], live[0]), np.full(1, np.nan))
        reached = bool(np.isfinite(v).any())
        amps.append(float(np.nanmedian(v)) if reached else top)
        notes.append("" if reached else f"; {target:.0%} not reached, top of the sweep")
    return amps, f"the effect {live[0]} needs for {target:.0%} power at each level", notes


def _rank_message(q, conditions, bad, n_reals) -> str:
    w = q["null_weights"]
    combo = " ".join(
        f"{'+' if v > 0 else '-'}{abs(v):.2g}*{c}"
        for c, v in zip(conditions, w, strict=True)
        if abs(v) > 0.05
    )
    return (
        f"the design is rank-deficient in {len(bad)} of {n_reals} realization(s) (first: "
        f"#{bad[0]}): {q['rank']} of {q['n_columns']} columns are independent. The "
        f"combination {combo or '(drift only)'} is reproduced by the other columns and the "
        "drift -- conditions with identical timing, a condition with no events in some run, "
        "or conditions that tile a run with no baseline."
    )


def _reference_noise(conds) -> str:
    """The middle noise level: where the design-quality matrix is reported."""
    return str(conds[len(conds) // 2]["label"])


def _quality_lines(quality, conditions, conds, target: float = 0.8) -> list[str]:
    """Rank, VIF, drift-removed correlation and the effect-needed matrix, as text."""
    q0 = quality[0]
    n = len(conditions)
    out = [
        f"Design quality (fitted model; median over {len(quality)} realization(s)):",
        f"  rank {q0['rank']} of {q0['n_columns']} columns ({n} task + drift, polort "
        f"{q0['poly_degree']} per run): full rank",
    ]
    vif = np.array([q["vif"] for q in quality])
    worst = float(vif.max())
    out.append(
        "  VIF (1 = orthogonal, > 5 hard, > 10 severe): "
        + ", ".join(f"{c} {v:.2f}" for c, v in zip(conditions, np.median(vif, axis=0), strict=True))
        + (f"   [worst realization {worst:.2f}]" if len(quality) > 1 else "")
        + ("  <- collinear" if worst > 5 else "")
    )
    out += _shape_lines(quality, conditions, conds)
    if n < 2:
        return out
    corr = np.median([q["corr"] for q in quality], axis=0)
    pairs = sorted(((i, j) for i in range(n) for j in range(i)), key=lambda ij: -abs(corr[ij]))
    out.append(
        "  regressor correlation after drift removal (largest): "
        + ", ".join(f"{conditions[i]}~{conditions[j]} {corr[i, j]:+.2f}" for i, j in pairs[:3])
    )
    ref = _reference_noise(conds)
    m = np.median([q["needed"][ref] for q in quality], axis=0)
    out.append(
        f"  effect for {target:.0%} power at {ref} (analytic, % signal change): diagonal = "
        "condition "
        "vs baseline, below it = the difference"
    )
    w = max(8, max(len(c) for c in conditions) + 2)
    out.append("    " + " " * w + "".join(f"{c:>{w}}" for c in conditions))
    for i, c in enumerate(conditions):
        out.append("    " + f"{c:<{w}}" + "".join(f"{m[i, j]:>{w}.2f}" for j in range(i + 1)))
    if len(pairs) > 1:  # with one pair, "hardest" and "easiest" are the same pair
        diffs = sorted(((m[i, j], i, j) for i, j in pairs))
        hard, easy = diffs[-1], diffs[0]
        out.append(
            "  (conditions that co-occur correlate positively and are costly to tell apart; "
            "one following the other correlates negatively and is cheap)"
        )
        out.append(
            f"  hardest to tell apart: {conditions[hard[1]]}-{conditions[hard[2]]} "
            f"({hard[0]:.2f}%); easiest: {conditions[easy[1]]}-{conditions[easy[2]]} "
            f"({easy[0]:.2f}%)."
        )
    return out


def _shape_text(quality, conditions, ref) -> str | None:
    """'E1 0.27%, ... per 1 s bin' of the FIR shape precision at ``ref``, median over realizations."""
    if not all("shape_sd" in q for q in quality):
        return None
    sd = np.median([q["shape_sd"][ref] for q in quality], axis=0)
    return ", ".join(
        f"{c} {'not estimable' if not np.isfinite(v) else f'{v:.2f}%'}"
        for c, v in zip(conditions, sd, strict=True)
    )


def _single_text(quality, conditions, ref) -> str | None:
    """LSS precision and leakage per condition, and LSA precision, medians over realizations."""
    qs = [q["single"] for q in quality if "single" in q]
    if not qs:
        return None
    lss = np.median([q["lss_sd"][ref] for q in qs], axis=0)
    lsa = np.median([q["lsa_sd"][ref] for q in qs], axis=0)
    leak = np.median([q["leakage"] for q in qs], axis=0)
    rel = {
        k: float(np.median([q["reliability"][ref][k] for q in qs]))
        for k in ("lss", "lsa", "ridge", "ridge_frac")
    }
    return (
        f"reliability of the trial pattern: LSS {rel['lss']:.2f}, LSA {rel['lsa']:.2f}, "
        f"ridge {rel['ridge']:.2f} (fraction {rel['ridge_frac']:.2f}); "
    ) + "; ".join(
        f"{c} LSS {a:.2f}% (leakage {k:.2f}), LSA "
        + ("not estimable" if not np.isfinite(b) else f"{b:.2f}%")
        for c, a, k, b in zip(conditions, lss, leak, lsa, strict=True)
    )


def _shape_lines(quality, conditions, conds) -> list[str]:
    ref = _reference_noise(conds)
    text = _shape_text(quality, conditions, ref)
    if text is None:
        return []
    k = quality[0]["fir_lags"]
    xi = float(np.median([q["xi"] for q in quality]))
    return [
        f"  response shape (FIR, {k} bins after onset), SD of one bin's estimate at {ref}: "
        f"{text}; Liu & Frank estimation efficiency {xi:.2f} of its bound",
        "  (detection and shape estimation trade off: rapid jitter recovers the shape and "
        "detects poorly, blocks the reverse; blank trials help both)",
    ] + (
        [
            f"  single trials at {ref}: {single}",
            "  (reliability: expected correlation of the estimated with the true trial-to-"
            "trial pattern, for -trial_sd variation around -effect (else 1%); ridge's "
            "fraction is the best one, which cross-validation across runs estimates -- it "
            "needs two or more runs. Per condition: SD of one trial's estimate, and LSS "
            "leakage, the neighbours' variation mixed into each estimate)",
        ]
        if (single := _single_text(quality, conditions, ref))
        else []
    )


def _sweep_lines(sweep, conds, contrasts, reals, tr, target: float = 0.8) -> list[str]:
    """The -scan_times table at the reference noise level: effect, and effect x sqrt(min)."""
    rows = sweep["rows"]
    ref = _reference_noise(conds)
    names = [
        c for c in contrasts if any(np.isfinite(r["needed"]) for r in rows if r["contrast"] == c)
    ]
    n_real = len({r["design"] for r in rows})
    out = [
        f"How long to scan (analytic, fitted HRF assumed right; median over {n_real} "
        f"realization(s); {ref}):",
        f"  effect for {target:.0%} power (% signal), and in brackets effect x sqrt(total "
        "minutes): "
        "flat = the design scales ideally, lower = more per minute of scanning",
        f"  {'-scan_time':>10} {'run s':>6} {'counts':>8} {'total min':>9}"
        + "".join(f"{c:>18}" for c in names),
    ]
    best: dict[str, tuple[float, float, float]] = {}
    for st in sorted({r["scan_time"] for r in rows}):
        sel = [r for r in rows if r["scan_time"] == st and r["noise"] == ref]
        cells = []
        for c in names:
            v = [r for r in sel if r["contrast"] == c]
            need = float(np.nanmedian([r["needed"] for r in v]))
            pm = float(np.nanmedian([r["per_minute"] for r in v]))
            cells.append(f"{f'{need:.2f} ({pm:.2f})':>18}")
            if c not in best or pm < best[c][0]:
                best[c] = (pm, st, sel[0]["run_s"])
        out.append(
            f"  {st:>10g} {sel[0]['run_s']:>6.0f} {sel[0]['counts']:>8} "
            f"{sel[0]['minutes']:>9.1f}" + "".join(cells)
        )
    for st, why in sweep["skipped"].items():
        out.append(f"  {st:>10g}  skipped: {why}")
    now = float(np.mean([sum(r.run_lengths) for r in reals])) * tr
    out.append(
        "  most per minute: "
        + "; ".join(f"{c} at {b[1]:g} ({b[2]:.0f} s runs, {b[0]:.2f})" for c, b in best.items())
        + f". This design: {now / 60:.1f} min."
    )
    return out


EXPLORE_ONLY = (
    "explore", "objective", "explore_designs", "explore_keep", "explore_pick",
    "optimize", "optimize_pop", "optimize_hrfs", "max_repeat",
)  # fmt: skip
TIMING_FLAGS = (
    "trial", "block", "miniblock", "null", "isi", "within_isi", "initial_fix", "post_fix",
    "order", "nruns", "scan_time", "num_events", "num_blocks", "ndesigns", "scan_times",
    "events", "labels", "durations", "nt",
)  # fmt: skip


def _mean_response(args) -> float:
    """The condition-mean response single-trial reliability assumes: -effect, else 1%."""
    return args.effect if args.effect is not None else 1.0


def _goal_of(sc: dict, goal: str) -> float:
    """One goal's value from an explorer design's aggregated scores."""
    key = {"shape": "shape_sd", "trials": "unreliability", "shape_diff": "shape_steps"}.get(goal)
    return float(sc.get(key, np.nan)) if key else float(sc["needed"].get(goal, np.nan))


def _objective(args, contrasts, pattern) -> str:
    """-objective: a contrast (its detection), 'shape' or 'trials'; default the first live one."""
    return _goal(args.objective, contrasts, pattern, "-objective")


def _goal(value, contrasts, pattern, flag: str) -> str:
    """A goal (-objective, -rank_by): a contrast, 'detection', 'shape' or 'trials'.

    Default: the first contrast with a true effect. 'efficiency' is 'detection'.
    """
    from fastfuncstuff.simulation.power import has_true_effect

    if value and ("," in value or "=" in value):
        # Combined goals, each validated: 'detection=1,shape=0.5,trials=1'.
        from fastfuncstuff.simulation.optimize import parse_goals

        parts = {_goal(g, contrasts, pattern, flag): w for g, w in parse_goals(value).items()}
        return ",".join(f"{g}={w:g}" for g, w in parts.items())
    goal = value or next((c for c, w in contrasts.items() if has_true_effect(w, pattern)), None)
    if goal == "efficiency":
        goal = "detection"
    goals = ("detection", "shape", "shape_diff", "trials")
    if goal is None or (goal not in goals and goal not in contrasts):
        raise ValueError(
            f"{flag} {goal!r}: use 'detection', 'shape', 'shape_diff', 'trials' or one of "
            f"{list(contrasts)}"
        )
    return goal


def _objective_label(objective: str, target: float = 0.8) -> str:
    if "=" in objective:
        from fastfuncstuff.simulation.optimize import parse_goals

        parts = " + ".join(f"{g} x{w:g}" for g, w in parse_goals(objective).items())
        return f"combined: {parts} (each relative to a typical design; 1 = typical)"
    if objective == "shape":
        return "response-shape SD per FIR bin (%)"
    if objective == "detection":
        return f"all contrasts: mean % signal for {target:.0%} power"
    if objective == "trials":
        return "single trials: 1 - reliability (best of LSS, ridge)"
    if objective == "shape_diff":
        return "library steps two shapes must be apart to be told apart"
    return f"{objective}: % signal for {target:.0%} power"


def _optimize_realization(args, spec, contrasts, pattern, conds, objective, progress=True):
    """evolve() on the HRF-averaged fitness, plus the held-out-HRF check. -> (result, lines)."""
    from fastfuncstuff.simulation.optimize import HELD_OUT_HRFS, ROBUST_HRFS, evolve, make_fitness

    ref = _reference_noise(conds)
    hrfs = list(dict.fromkeys(args.optimize_hrfs or [args.hrf, *ROBUST_HRFS]))
    trial_kw = {
        "mean_response": _mean_response(args),
        "trial_sd": args.trial_sd,
        "target": args.target,
    }
    # A combined objective scales each goal by its value in typical random draws.
    from fastfuncstuff.simulation.experiment import realize

    trial_kw["reference"] = [realize(spec, args.seed + 1_000_000 + i) for i in range(20)]
    fit = make_fitness(
        args.tr,
        contrasts,
        conds,
        pattern,
        objective,
        ref,
        hrfs,
        args.alpha,
        args.polort,
        **trial_kw,
    )
    res = evolve(
        spec,
        fit,
        population=args.optimize_pop,
        generations=args.optimize,
        seed=args.seed,
        max_repeat=args.max_repeat,
        progress=progress,
    )
    n = res["evaluations"][-1]
    gain = 100 * (1 - res["best_fitness"] / res["random_best"][-1])
    lines = [
        f"search: {args.optimize} generations x {args.optimize_pop}: {n} realizations scored, "
        f"averaged over HRFs {', '.join(hrfs)}"
        + (f"; {res['n_rejected']} children broke -max_repeat" if res["n_rejected"] else ""),
        f"  {_objective_label(objective, args.target)}, mean over those HRFs: median random draw "
        f"{res['random_median']:.3f}, best of {n} random {res['random_best'][-1]:.3f}, "
        f"evolved {res['best_fitness']:.3f} ({gain:+.1f}% vs best-of-N)",
    ]
    held = [h for h in HELD_OUT_HRFS if h not in hrfs]
    if held:
        check = make_fitness(
            args.tr,
            contrasts,
            conds,
            pattern,
            objective,
            ref,
            held,
            args.alpha,
            args.polort,
            **trial_kw,
        )
        rb, ev = check(res["random_best_realization"]), check(res["best"])
        lines.append(
            f"  held-out HRFs ({', '.join(held)}), never searched on: best-of-N {rb:.3f}, "
            f"evolved {ev:.3f}"
            + ("" if ev <= rb else "  <- the search overfit its HRFs; prefer best-of-N")
        )
    return res, lines


def _run_optimize(args, argv, spec, contrasts, pattern, conds, started) -> int:
    """-optimize: the best realization of one design, its timing files and the evidence."""
    import shlex

    from fastfuncstuff.simulation.core import write_timing_files

    try:
        objective = _objective(args, contrasts, pattern)
        res, lines = _optimize_realization(args, spec, contrasts, pattern, conds, objective)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    best = res["best"]
    out_dir = Path(f"{prefix}_optimized")
    write_timing_files(best.onsets, best.conditions, out_dir)
    raw = list(sys.argv[1:] if argv is None else argv)
    rest = _strip_flags(raw, _build_parser(), (*TIMING_FLAGS, *EXPLORE_ONLY, "tr", "prefix"))
    cmd = ["ffs_simulate", "-tr", f"{args.tr:g}", "-events",
           *[str(out_dir / f"{c}.txt") for c in best.conditions],
           "-durations", *[f"{d:g}" for d in best.durations],
           "-nt", *[str(n) for n in best.run_lengths], *rest, "-prefix", f"{prefix}_opt"]  # fmt: skip
    text = [
        "ffs_simulate -optimize",
        "=" * 72,
        f"objective: {_objective_label(objective, args.target)} at {_reference_noise(conds)} (analytic)",
        *lines,
        f"optimized realization: {out_dir}/ -- full Monte Carlo and figures:",
        "  " + shlex.join(cmd),
    ]
    summary = "\n".join(text)
    print(summary)
    Path(f"{prefix}_optimize_summary.txt").write_text(summary + "\n")
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from fastfuncstuff.simulation.plots import plot_optimize

        plot_optimize(res, _objective_label(objective, args.target), path=f"{prefix}_optimize.png")
    written = sorted(
        str(q.name).removeprefix(prefix.name)
        for q in prefix.parent.glob(f"{prefix.name}_*")
        if q.stat().st_mtime >= started
    )
    print(f"\nwrote {prefix}: " + ", ".join(written))
    return 0


def _strip_flags(argv: list[str], parser, drop: tuple[str, ...]) -> list[str]:
    """argv without the flags named in ``drop`` (either spelling) and their values."""
    known = parser._option_string_actions
    names = {"-" + d for d in drop} | {"-" + d.replace("_", "-") for d in drop}
    out, skipping = [], False
    for tok in argv:
        if tok in known:
            skipping = tok in names
        if not skipping:
            out.append(tok)
    return out


def _run_explore(raw: list[str], started: float) -> int:
    """-explore: score N designs drawn from the placeholders, shortlist the Pareto front."""
    import shlex

    from fastfuncstuff.simulation import explore as ex
    from fastfuncstuff.simulation.experiment import default_contrasts, parse_contrast, realize
    from fastfuncstuff.simulation.power import RealizationScorer, has_true_effect

    parser = _build_parser()
    try:
        axes = ex.find_axes(raw)
        if not axes:
            raise ValueError(
                "-explore needs at least one range [a-b] or choice {x,y} inside the flags"
            )
        mid = ex.render(raw, axes, {a.label: a.value(0.5) for a in axes})
        args = parser.parse_args(mid)
        if args.tr is None or not (args.trial or args.block or args.miniblock):
            raise ValueError("-explore needs -tr and a described experiment")
        budget_axes = [a.label for a in axes if a.label.split(".")[0] in
                       ("scan_time", "num_events", "num_blocks")]  # fmt: skip
        if budget_axes:
            raise ValueError(f"the budget must be fixed, not explored ({budget_axes[0]})")
        if args.scan_time is not None and (args.num_events or args.num_blocks):
            raise ValueError(
                "-explore takes one budget: -scan_time (time) or the trial counts "
                "(-num_events/-num_blocks), not both -- otherwise the search space is unbounded"
            )
        conditions = realize(_spec_from_args(args), args.seed).conditions
        contrasts = (
            {e: parse_contrast(e, conditions) for e in args.contrast}
            if args.contrast
            else default_contrasts(conditions)
        )
        pattern = _pattern(args.pattern, conditions)
        conds, _ = _noise_conditions(args)
        objective = _objective(args, contrasts, pattern)
    except (ValueError, FileNotFoundError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            return int(exc.code or 1)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ref = _reference_noise(conds)
    configs = ex.sample(axes, args.explore, args.seed)
    rendered, specs, refused = [], [], {}
    for values in configs:
        argv_k = ex.render(raw, axes, values)
        rendered.append(argv_k)
        try:
            spec = _spec_from_args(parser.parse_args(argv_k))
            spec.resolve_counts()
            specs.append(spec)
        except (ValueError, SystemExit) as exc:
            specs.append(None)
            why = str(exc).split(" -- ")[0]
            refused[why] = refused.get(why, 0) + 1
    scorer = RealizationScorer(
        args.tr,
        contrasts,
        conds,
        pattern,
        args.hrf,
        args.alpha,
        args.target,
        poly_degree=args.polort,
        mean_response=_mean_response(args),
        trial_sd=args.trial_sd,
    )
    from fastfuncstuff.simulation.optimize import combine, parse_goals

    goals = parse_goals(objective)
    flags = {"single": "trials" in goals, "steps": "shape_diff" in goals}
    scores = ex.score_configs(
        specs,
        scorer,
        args.explore_designs,
        ref,
        args.seed,
        single_all="trials" in goals,
        steps_all="shape_diff" in goals,
    )
    live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
    det_c = next((g for g in goals if g in contrasts or g == "detection"), live[0])
    # The trade-off: detection against response shape -- or, when single trials are
    # the target, against LSS leakage (neighbours mixed into each trial's estimate).
    y_key = {"trials": "unreliability", "shape_diff": "shape_steps"}.get(objective, "shape_sd")
    x = np.array([sc["needed"][det_c] if sc else np.nan for sc in scores])
    y = np.array([sc.get(y_key, np.nan) if sc else np.nan for sc in scores])
    combined = None
    if len(goals) > 1:
        # Each goal relative to its median over the explored designs (1 = typical);
        # the shortlist is the best on the weighted mean, not only the 2-D front.
        vals = [{g: _goal_of(sc, g) for g in goals} if sc else None for sc in scores]
        typical = {g: float(np.nanmedian([v[g] for v in vals if v])) for g in goals}
        combined = np.array([combine(v, goals, typical) if v else np.nan for v in vals])
        y = combined
    front = ex.pareto_front(x, y)
    if combined is not None:
        order = [int(i) for i in np.argsort(np.where(np.isfinite(combined), combined, np.inf))]
        keep = [i for i in order if np.isfinite(combined[i])][: args.explore_keep]
    else:
        keep = ex.shortlist(
            y if objective in ("shape", "trials", "shape_diff") else x, front, args.explore_keep
        )
    edges = ex.at_edges(axes, configs, keep)

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cols = ["design", *[a.label for a in axes], "feasible", "on_front", "counts", "minutes",
            *[f"needed_{c}" for c in [*live, "detection"]],
            *[f"worst_{c}" for c in [*live, "detection"]],
            "shape_sd", "estimation_efficiency", "lss_sd", "leakage", "lsa_sd", "unreliability",
            "ridge_frac", "shape_steps", "combined", "dropped"]  # fmt: skip
    with open(f"{prefix}_explore.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for k, (values, sc) in enumerate(zip(configs, scores, strict=True)):
            w.writerow(
                [k, *[values[a.label] for a in axes], int(bool(sc)), int(front[k])]
                + (
                    [sc["counts"], f"{sc['minutes']:.2f}"]
                    + [f"{sc['needed'][c]:.4f}" for c in [*live, "detection"]]
                    + [f"{sc['worst'][c]:.4f}" for c in [*live, "detection"]]
                    + [f"{sc['shape_sd']:.4f}", f"{sc['xi']:.4f}"]
                    + [
                        f"{sc[k]:.4f}" if k in sc else ""
                        for k in (
                            "lss_sd",
                            "leakage",
                            "lsa_sd",
                            "unreliability",
                            "ridge_frac",
                            "shape_steps",
                        )
                    ]  # fmt: skip
                    + [f"{combined[k]:.4f}" if combined is not None else ""]
                    + [f"{sc['dropped']:.2f}"]
                    if sc
                    else [""] * (len(cols) - len(axes) - 3)
                )
            )

    # The shortlist: the recipe to reproduce, and its best realization's timing files.
    best_dir = Path(f"{prefix}_explore_best")
    lines = []
    for rank, k in enumerate(keep, start=1):
        sc = scores[k]
        # Render on the original argv (placeholder positions index it), then strip.
        filled = _strip_flags(rendered[k], parser, (*EXPLORE_ONLY, "prefix"))
        recipe = filled
        noise_etc = _strip_flags(filled, parser, (*TIMING_FLAGS, "tr"))
        if args.optimize:
            opt, opt_lines = _optimize_realization(
                args, specs[k], contrasts, pattern, conds, objective, progress=False
            )
            picked = (opt["best"], scorer.score(opt["best"], **flags))
            fit_val = opt["best_fitness"]
        elif combined is not None:
            # best_realization ranks one goal; a combined one needs its typical
            # values, so draw and score directly.
            from fastfuncstuff.simulation.experiment import realize
            from fastfuncstuff.simulation.optimize import make_fitness

            opt_lines = []
            draws = [realize(specs[k], args.seed + 7919 * k + i) for i in range(args.explore_pick)]
            fit = make_fitness(
                args.tr, contrasts, conds, pattern, objective, ref, [args.hrf], args.alpha,
                args.polort, _mean_response(args), args.trial_sd, args.target,
                reference=draws[:10],
            )  # fmt: skip
            fits = [fit(r) for r in draws]
            best_i = int(np.nanargmin(fits))
            picked = (draws[best_i], scorer.score(draws[best_i], **flags))
            fit_val = fits[best_i]
        else:
            opt_lines = []
            picked = ex.best_realization(
                specs[k], scorer, args.explore_pick, ref, objective, args.seed + 7919 * k
            )
            fit_val = None
        desc = "  ".join(f"{a.label}={configs[k][a.label]}" for a in axes)
        lines += [
            f"#{rank}  {desc}",
            "    "
            + "  ".join(f"{c} {sc['needed'][c]:.2f}%" for c in live)
            + f"  shape {sc['shape_sd']:.2f}% (estimation efficiency {sc['xi']:.2f} of bound)"
            + (f"  shape resolution {sc['shape_steps']:.1f} steps" if "shape_steps" in sc else "")
            + (f"  combined {combined[k]:.2f}" if combined is not None else "")
            + (
                f"  trials: reliability {1 - sc['unreliability']:.2f} "
                f"(LSS leakage {sc['leakage']:.2f}, ridge fraction {sc['ridge_frac']:.2f})"
                if "unreliability" in sc
                else ""
            )
            + f"  {sc['minutes']:.1f} min  counts {sc['counts']}",
            "    recipe:  " + shlex.join(["ffs_simulate", *recipe, "-prefix", f"{prefix}_d{rank}"]),
        ]
        if picked is not None:
            real, psc = picked
            out_dir = best_dir / str(rank)
            from fastfuncstuff.simulation.core import write_timing_files

            write_timing_files(real.onsets, real.conditions, out_dir)
            files = [str(out_dir / f"{c}.txt") for c in real.conditions]
            val = (
                fit_val
                if fit_val is not None
                else psc["shape_sd"][ref]
                if objective == "shape"
                else psc["unreliability"][ref]
                if objective == "trials"
                else psc["shape_steps"][ref]
                if objective == "shape_diff"
                else psc["needed"][(ref, objective)]
            )
            cmd = ["ffs_simulate", "-tr", f"{args.tr:g}", "-events", *files,
                   "-durations", *[f"{d:g}" for d in real.durations],
                   "-nt", *[str(n) for n in real.run_lengths], *noise_etc,
                   "-prefix", f"{prefix}_d{rank}_best"]  # fmt: skip
            how = (
                f"optimized, {args.optimize} generations"
                if args.optimize
                else f"best of {args.explore_pick} realizations"
            )
            lines += ["    " + ln for ln in opt_lines]
            lines.append(
                f"    {how} ("
                + (
                    f"combined {val:.2f}"
                    if combined is not None
                    else f"{objective} {val:.2f}"
                    + ("" if objective in ("trials", "shape_diff") else "%")
                )
                + "): "
                + shlex.join(cmd)
            )
    n_ok = sum(1 for sc in scores if sc)
    text = [
        "ffs_simulate -explore",
        "=" * 72,
        f"{len(configs)} designs drawn, {n_ok} feasible"
        + (
            " -- refused: " + "; ".join(f"{n} x {why}" for why, n in refused.items())
            if refused
            else ""
        ),
        "budget: "
        + (
            f"-scan_time {args.scan_time:g} s per run (time)"
            if args.scan_time is not None
            else "trial counts (time follows)"
        ),
        "axes: " + ", ".join(f"{a.label} {raw[a.token][a.span[0] : a.span[1]]}" for a in axes),
        f"scored analytically at {ref} (fitted HRF assumed right), median of "
        f"{args.explore_designs} realization(s) each",
        "trade-off: "
        + ("detection, mean over contrasts" if det_c == "detection" else f"{det_c} detection")
        + f" (% signal for {args.target:.0%} power) against "
        + (
            "single-trial 1 - reliability (best of LSS, ridge)"
            if objective == "trials"
            else "library steps two shapes must be apart to be told apart"
            if objective == "shape_diff"
            else _objective_label(objective, args.target)
            if combined is not None
            else "response-shape SD per FIR bin"
        )
        + "; lower is better on both",
        "",
        "what matters (rank correlation with detection / "
        + (
            "combined"
            if combined is not None
            else {"trials": "1 - reliability", "shape_diff": "shape steps"}.get(objective, "shape")
        )
        + " over feasible designs):",
    ]
    for a in axes:
        text.append(
            "  "
            + _axis_effect(
                a,
                configs,
                x,
                y,
                "combined"
                if combined is not None
                else {"trials": "1-reliab.", "shape_diff": "steps"}.get(objective, "shape"),
            )  # fmt: skip
        )
    text += ["", f"Pareto front: {int(front.sum())} design(s); shortlist:", *lines]
    for label, side, span in edges:
        text.append(
            f"note: the shortlist sits at the {side} end of {label} {span} -- the best designs "
            "may lie beyond it; widen that range"
        )
    if best_dir.exists():
        text.append(
            f"\nbest realizations: {best_dir}/<rank>/ -- run a command above for the full "
            "Monte Carlo and figures of that design"
        )
    summary = "\n".join(text)
    print(summary)
    Path(f"{prefix}_explore_summary.txt").write_text(summary + "\n")
    if not args.no_plots and n_ok:
        import matplotlib

        matplotlib.use("Agg")
        from fastfuncstuff.simulation.plots import plot_exploration

        plot_exploration(
            axes,
            configs,
            x,
            y,
            front,
            keep,
            det_c,
            ref,
            path=f"{prefix}_explore.png",
            y_label=(
                "combined score (1 = a typical design; lower is better)"
                if combined is not None
                else {
                    "trials": "single trials: 1 - reliability (best of LSS, ridge)",
                    "shape_diff": "shape resolution: library steps apart",
                }.get(objective)
            ),
        )
        from fastfuncstuff.simulation.plots import plot_liu

        ok = [i for i, sc in enumerate(scores) if sc]
        pts = {"explored designs": [(scores[i]["xi"], scores[i]["liu_power"]) for i in ok]}
        for rank, i in enumerate(keep, start=1):
            pts[f"shortlist #{rank}"] = [(scores[i]["xi"], scores[i]["liu_power"])]
        plot_liu(
            pts, int(round(16.0 / args.tr)), len(conditions), path=f"{prefix}_explore_liu.png",
            scatter=True,
        )  # fmt: skip
    written = sorted(
        str(q.name).removeprefix(prefix.name)
        for q in prefix.parent.glob(f"{prefix.name}_*")
        if q.stat().st_mtime >= started
    )
    print(f"\nwrote {prefix}: " + ", ".join(written))
    return 0


def _axis_effect(axis, configs, x, y, y_name: str = "shape") -> str:
    """One line on how an axis moves detection and shape (Spearman, or medians per choice)."""
    from scipy.stats import spearmanr

    vals = [axis.numeric(c[axis.label]) for c in configs]
    ok = np.isfinite(x) & np.isfinite(y)
    if axis.is_choice:
        parts = []
        for choice in axis.choices:
            sel = ok & np.array([v == choice for v in vals])
            if sel.any():
                parts.append(f"{choice} {np.median(x[sel]):.2f}% / {np.median(y[sel]):.2f}%")
        return f"{axis.label:<14} median detection / {y_name}: " + "; ".join(parts)
    v = np.array(vals, dtype=float)
    if ok.sum() < 3:
        return f"{axis.label:<14} too few feasible designs"

    def rho(a, b):  # a constant measure (none resolved, say) has no rank correlation
        return float("nan") if np.ptp(a) == 0 or np.ptp(b) == 0 else spearmanr(a, b).statistic

    rd, rs = rho(v[ok], x[ok]), rho(v[ok], y[ok])
    return (
        f"{axis.label:<14} detection rho {rd:+.2f}, {y_name} rho {rs:+.2f} "
        "(negative: larger values help)"
    )


def _run_compare(args) -> int:
    from fastfuncstuff.simulation.power import compare_designs, load_power_table

    try:
        loaded = [load_power_table(f) for f in args.compare]
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    names = args.compare_names or [r["name"] for r in loaded]
    if len(names) != len(loaded) or len(set(names)) != len(names):
        print("ERROR: -compare_names needs one distinct name per file", file=sys.stderr)
        return 1
    results = dict(zip(names, loaded, strict=True))
    rows = compare_designs(results, args.target)
    if not rows:
        print("ERROR: the files share no noise level and contrast", file=sys.stderr)
        return 1

    out = ["ffs_simulate -compare", "=" * 72]
    scans = {n: r["scan_s"] for n, r in ((r["design"], r) for r in rows)}
    for n in names:
        s = scans[n]
        out.append(
            f"  {n:<28} {'scan time unknown (no _spec.json)' if s is None else f'{s:.0f} s total'}"
        )
    known = [s for s in scans.values() if s is not None]
    all_known = len(known) == len(scans)
    if known and (max(known) - min(known)) > 0.02 * max(known):
        out.append(
            "note: scan times differ -- the longer design wins the effect table partly by "
            "having more data"
            + (
                "; the x sqrt(minutes) table below each contrast compares them per unit of "
                "scan time"
                if all_known
                else "; some files have no _spec.json, so there is no per-minute view"
            )
        )
    contrasts = list(dict.fromkeys(r["contrast"] for r in rows))
    noises = list(dict.fromkeys(r["noise"] for r in rows))
    quals = {n: (res.get("spec") or {}).get("quality") for n, res in results.items()}
    if any(quals.values()):
        out += [
            "",
            "Response-shape estimation: SD of one FIR bin's estimate, % signal (mean over "
            "conditions, median over realizations), and Liu & Frank estimation efficiency (of its "
            "bound)",
            f"{'design':<28}" + "".join(f"{n:>22}" for n in noises) + f"{'est. effic.':>12}",
        ]
        for n in names:
            q = quals[n]
            if not q:
                out.append(f"{n:<28}  (no estimation in its _spec.json -- rerun ffs_simulate)")
                continue
            cells = []
            for noise in noises:
                v = [np.mean(x) for x in q["shape_sd"].get(noise, [])]
                cells.append(f"{(f'{np.median(v):.2f}' if v else '-'):>22}")
            out.append(f"{n:<28}" + "".join(cells) + f"{np.median(q['xi']):>12.2f}")
    cards = {n: (res.get("spec") or {}).get("scorecard") for n, res in results.items()}
    if any(cards.values()):
        keys = list(dict.fromkeys(k for card in cards.values() if card for k in card))
        out += ["", "Scorecards (one value per measure, at each design's middle noise level):"]
        out.append(f"{'measure':<24}" + "".join(f"{n[:20]:>22}" for n in names))
        for k in keys:
            cells = [f"{cards[n][k]:.3g}" if cards[n] and k in cards[n] else "-" for n in names]
            out.append(f"{k:<24}" + "".join(f"{c:>22}" for c in cells))
    for c in contrasts:
        swept = next(
            (r.get("swept") for r in loaded[0]["table"] if r["contrast"] == c), "amplitude"
        )
        what = "difference" if swept == "difference" else "amplitude"
        out += [
            "",
            f"{c}: {what} (% signal change) for {args.target:.0%} power -- "
            "median [range] over realizations",
        ]
        out.append(f"{'design':<28}" + "".join(f"{n:>22}" for n in noises))
        for n in names:
            cells = []
            for noise in noises:
                r = next(
                    x
                    for x in rows
                    if x["design"] == n and x["contrast"] == c and x["noise"] == noise
                )
                if not r["has_effect"]:
                    text = "no true effect"
                elif np.isnan(r["median"]):
                    text = "not reached"
                else:
                    text = f"{r['median']:.2f} [{r['min']:.2f}-{r['max']:.2f}]"
                    if r["n_unreached"]:
                        text += f" +{r['n_unreached']}"
                cells.append(f"{text:>22}")
            out.append(f"{n:<28}" + "".join(cells))
        if all_known:
            out.append(
                "  per unit of scan time: effect x sqrt(total minutes), lower = more per minute"
            )
            for n in names:
                cells = []
                for noise in noises:
                    r = next(
                        x
                        for x in rows
                        if x["design"] == n and x["contrast"] == c and x["noise"] == noise
                    )
                    pm = r["per_minute"]
                    text = "-" if not r["has_effect"] or np.isnan(pm) else f"{pm:.2f}"
                    cells.append(f"{text:>22}")
                out.append(f"  {n:<26}" + "".join(cells))
    out.append("  +k: k realizations never reach the target within their sweep")
    text = "\n".join(out)
    print(text)
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}_compare.txt").write_text(text + "\n")
    if not args.no_plots:
        import re

        import matplotlib

        matplotlib.use("Agg")
        from fastfuncstuff.simulation.plots import plot_design_comparison

        for c in contrasts:
            if not any(r["has_effect"] for r in rows if r["contrast"] == c):
                continue
            safe = re.sub(r"[^\w.+-]", "_", c)
            plot_design_comparison(
                results, c, noises, args.target, path=f"{prefix}_compare_{safe}.png"
            )
            if any(quals.values()):
                from fastfuncstuff.simulation.plots import plot_detection_estimation

                plot_detection_estimation(
                    results,
                    c,
                    noises[len(noises) // 2],
                    args.target,
                    path=f"{prefix}_compare_tradeoff_{safe}.png",
                )
        liu = {
            n: list(zip(q["xi"], q["liu_power"], strict=True))
            for n, q in quals.items()
            if q and "liu_power" in q
        }
        if liu:
            from fastfuncstuff.simulation.plots import plot_liu

            k = next(int(q["fir_lags"]) for q in quals.values() if q and "fir_lags" in q)
            n_cond = max(len(r.get("spec", {}).get("conditions", [1])) for r in results.values())
            plot_liu(liu, k, n_cond, path=f"{prefix}_compare_liu.png", scatter=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import time

    started = time.time() - 1.0  # files written from here on (mtime resolution)
    raw = list(sys.argv[1:] if argv is None else argv)
    if any(t in ("-explore",) for t in raw):
        # Before argparse: placeholders inside typed flags ("-initial_fix [5-15]") would
        # fail its type check; each design is parsed once its values are in.
        return _run_explore(raw, started)
    args = _build_parser().parse_args(argv)
    if args.compare:
        return _run_compare(args)
    if args.tr is None:
        print("ERROR: -tr is required (unless -compare)", file=sys.stderr)
        return 1
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.simulation.experiment import default_contrasts, parse_contrast, realize
    from fastfuncstuff.simulation.power import has_true_effect, simulate_realizations_power

    described = bool(args.trial or args.miniblock or args.block)
    if described == bool(args.events):
        print(
            "ERROR: give either -events or a described experiment (-trial/-block/-miniblock)",
            file=sys.stderr,
        )
        return 1
    try:
        if described:
            spec = _spec_from_args(args)
            spec_text = spec.describe()
            reals = [realize(spec, args.seed + i) for i in range(args.ndesigns)]
        else:
            spec_text = ""
            reals = [_explicit_realization(args)]
        conditions = reals[0].conditions
        contrasts = (
            {e: parse_contrast(e, conditions) for e in args.contrast}
            if args.contrast
            else default_contrasts(conditions)
        )
        pattern = _pattern(args.pattern, conditions)
        amps = _amplitudes(args.amplitudes, args.effect)
        conds, profile_text = _noise_conditions(args)
        import torch

        from fastfuncstuff.simulation.core import hrfs_from_spec

        for hrf_spec in {args.hrf, args.true_hrf} - {"same"}:
            hrfs_from_spec(hrf_spec, 0.1, torch.device("cpu"))  # fail before simulating
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    from fastfuncstuff.simulation.power import realizations_design_quality

    quality = realizations_design_quality(
        reals,
        args.tr,
        conds,
        hrf=args.hrf,
        alpha=args.alpha,
        poly_degree=args.polort,
        mean_response=_mean_response(args),
        trial_sd=args.trial_sd,
        target=args.target,
    )
    bad = [i for i, q in enumerate(quality) if q["deficient"]]
    if bad:
        print(
            f"ERROR: {_rank_message(quality[bad[0]], conditions, bad, len(reals))}", file=sys.stderr
        )
        return 1

    if args.optimize:
        if not described:
            print("ERROR: -optimize needs a described experiment", file=sys.stderr)
            return 1
        return _run_optimize(args, argv, spec, contrasts, pattern, conds, started)

    sweep = None
    if args.scan_times:
        from dataclasses import replace

        from fastfuncstuff.simulation.power import scan_time_sweep

        if not described or not replace(spec, scan_time=1.0).scan_sized():
            print(
                "ERROR: -scan_times needs a described experiment whose counts -scan_time "
                "sets (a family without -num_events/-num_blocks); otherwise longer runs "
                "only add fixation",
                file=sys.stderr,
            )
            return 1
        sweep = scan_time_sweep(
            spec,
            sorted(args.scan_times),
            min(args.ndesigns, SWEEP_DESIGNS),
            contrasts,
            conds,
            beta_pattern=pattern,
            hrf=args.hrf,
            alpha=args.alpha,
            poly_degree=args.polort,
            seed=args.seed,
            target=args.target,
        )

    device = setup_device(args.device)
    res = simulate_realizations_power(
        reals,
        args.tr,
        contrasts,
        amps,
        conds,
        beta_pattern=pattern,
        n_reps=args.nreps,
        alpha=args.alpha,
        true_delay=args.true_delay,
        hrf=args.hrf,
        true_hrf=args.true_hrf,
        shared=args.shared,
        poly_degree=args.polort,
        device=device,
        seed=args.seed,
    )

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    from fastfuncstuff.simulation.power import shape_steps

    shape_amps = [(args.effect if args.effect is not None else 1.0) * w for w in pattern]
    steps = shape_steps(reals[0], args.tr, conds, shape_amps, args.alpha, args.target, args.polort)
    live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
    ref = _reference_noise(conds)
    robust = None
    if live:
        from fastfuncstuff.simulation.power import hrf_robustness

        robust = hrf_robustness(
            reals[0], args.tr, {c: contrasts[c] for c in live},
            next(c for c in conds if c["label"] == ref), pattern, args.hrf, args.shared,
            args.alpha, args.target, args.polort,
        )  # fmt: skip
    card = _scorecard(res, reals, conds, contrasts, pattern, args, quality, steps, robust)
    summary = _summarise(
        res, reals, conds, contrasts, pattern, args, spec_text, profile_text, quality, steps,
        card,
    )  # fmt: skip
    if sweep is not None:
        summary += "\n\n" + "\n".join(
            _sweep_lines(sweep, conds, contrasts, reals, args.tr, args.target)
        )
    print(summary)
    Path(f"{prefix}_summary.txt").write_text(summary + "\n")
    if sweep is not None:
        with open(f"{prefix}_scantime.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(SWEEP_COLS), delimiter="\t")
            w.writeheader()
            w.writerows(sweep["rows"])

    cols = [
        "design",
        "true_hrf",
        "noise",
        "tsnr",
        "amplitude",
        "contrast",
        "swept",
        "shared",
        "true_effect",
        "mean_est",
        "expected_est",
        "sd_est",
        "sd_predicted",
        "mean_t",
        "power",
        "power_predicted",
        "mean_t_naive",
        "power_naive",
    ]
    with open(f"{prefix}_power.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(res["table"])

    Path(f"{prefix}_spec.json").write_text(
        json.dumps(
            {
                "argv": sys.argv if argv is None else argv,
                "conditions": conditions,
                "durations": reals[0].durations,
                "contrasts": {k: v.tolist() for k, v in contrasts.items()},
                "pattern": pattern,
                "shared": args.shared,
                "amplitudes": amps,
                "noise": conds,
                "seeds": [r.seed for r in reals],
                "run_lengths": [r.run_lengths for r in reals],
                "tr": args.tr,
                "scan_time": args.scan_time,
                "polort": res["designs"][0]["poly_degree"],
                "hrf": res["hrf"],
                "true_hrfs": res["true_hrfs"],
                "alpha": args.alpha,
                "scorecard": {SCORE_KEYS.get(k.strip(), k.strip()): v for k, _, v in card},
                "tr_lock": args.tr_lock,
                # per realization, for -compare's detection-vs-estimation view
                "quality": {
                    "shape_sd": {
                        c["label"]: [q["shape_sd"][c["label"]].tolist() for q in quality]
                        for c in conds
                    },
                    "xi": [q["xi"] for q in quality],
                    "liu_power": [q.get("liu_power", np.nan) for q in quality],
                    "fir_lags": quality[0]["fir_lags"],
                    "single": {
                        "lss_sd": {
                            c["label"]: [
                                q["single"]["lss_sd"][c["label"]].tolist()
                                for q in quality
                                if "single" in q
                            ]
                            for c in conds
                        },
                        "lsa_sd": {
                            c["label"]: [
                                q["single"]["lsa_sd"][c["label"]].tolist()
                                for q in quality
                                if "single" in q
                            ]
                            for c in conds
                        },
                        "leakage": [
                            q["single"]["leakage"].tolist() for q in quality if "single" in q
                        ],
                        "reliability": [
                            q["single"]["reliability"] for q in quality if "single" in q
                        ],
                    },
                },
            },
            indent=2,
            default=str,
        )
    )
    if described:
        from fastfuncstuff.simulation.core import write_timing_files

        write_timing_files(reals[0].onsets, reals[0].conditions, Path(f"{prefix}_events"))
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from fastfuncstuff.simulation.plots import plot_design, plot_power

        labels = [c["label"] for c in conds]
        if any(has_true_effect(w, pattern) for w in contrasts.values()):
            plot_power(
                res,
                labels,
                contrasts,
                pattern,
                args.alpha,
                args.effect,
                path=f"{prefix}_power.png",
                summary=_figure_summary(
                    res,
                    reals,
                    conds,
                    contrasts,
                    pattern,
                    args,
                    quality,
                    spec if described else None,
                    card,
                ),
            )
        if res["true_hrfs"] != [res["hrf"]]:
            from fastfuncstuff.simulation.plots import plot_hrf_recovery

            effective = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
            if effective:
                plot_hrf_recovery(res, args.tr, effective[0], path=f"{prefix}_hrf.png")
        from fastfuncstuff.simulation.plots import plot_example_voxels

        truth = res["true_hrfs"][0] if res["true_hrfs"] != [res["hrf"]] else res["hrf"]
        amp, basis, notes = _voxel_amplitudes(
            res, conds, contrasts, pattern, args.effect, args.target
        )
        plot_example_voxels(
            reals[0],
            args.tr,
            conds,
            amplitude=amp,
            basis=basis,
            notes=notes,
            true_hrf=truth,
            seed=args.seed,
            path=f"{prefix}_voxels.png",
        )
        from fastfuncstuff.simulation.plots import plot_tent
        from fastfuncstuff.simulation.power import tent_estimate

        tent_amps = [(args.effect if args.effect is not None else 1.0) * w for w in pattern]
        plot_tent(
            tent_estimate(reals[0], args.tr, conds, tent_amps, truth, seed=args.seed,
                          poly_degree=args.polort),
            reals[0].conditions,
            [c["label"] for c in conds],
            tent_amps,
            path=f"{prefix}_tent.png",
        )  # fmt: skip
        from fastfuncstuff.simulation.plots import plot_shape_steps

        plot_shape_steps(
            steps, reals[0].conditions, [c["label"] for c in conds], shape_amps, args.target,
            path=f"{prefix}_shape.png",
        )  # fmt: skip
        from fastfuncstuff.simulation.plots import plot_single_trials
        from fastfuncstuff.simulation.power import single_trial_example

        ref = _reference_noise(conds)
        plot_single_trials(
            single_trial_example(
                reals[0], args.tr, next(c for c in conds if c["label"] == ref),
                _mean_response(args), args.trial_sd, args.hrf, args.polort, args.seed,
            ),
            reals[0].conditions,
            args.trial_sd,
            path=f"{prefix}_trials.png",
        )  # fmt: skip
        live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
        if live:
            from fastfuncstuff.simulation.plots import plot_tstats
            from fastfuncstuff.simulation.power import effect_needed, t_example

            c0 = live[0]
            need = effect_needed(res, args.target)
            top = max(abs(r["true_effect"]) for r in res["table"] if r["contrast"] == c0)
            effects = {}
            for cnd in conds:
                v = need.get((cnd["label"], c0), np.full(1, np.nan))
                effects[cnd["label"]] = (
                    args.effect
                    if args.effect is not None
                    else (float(np.nanmedian(v)) if np.isfinite(v).any() else top)
                )
            plot_tstats(
                t_example(
                    reals[0], args.tr, c0, contrasts[c0], conds, effects, pattern, args.hrf,
                    args.alpha, args.polort, seed=args.seed,
                ),
                c0,
                [c["label"] for c in conds],
                args.alpha,
                path=f"{prefix}_tstats.png",
            )  # fmt: skip
        if live:
            from fastfuncstuff.simulation.plots import plot_spectrum
            from fastfuncstuff.simulation.power import design_spectrum

            plot_spectrum(
                design_spectrum(
                    reals[0], args.tr, {c: contrasts[c] for c in live},
                    next(c for c in conds if c["label"] == ref), args.hrf, args.polort,
                ),
                live,
                path=f"{prefix}_spectrum.png",
            )  # fmt: skip
        if live:
            from fastfuncstuff.simulation.plots import plot_liu, plot_soa
            from fastfuncstuff.simulation.power import RealizationScorer, soa_sweep

            ref_noise = [c for c in conds if c["label"] == ref]
            soa_scorer = RealizationScorer(
                args.tr, {c: contrasts[c] for c in live}, ref_noise, pattern, args.hrf,
                args.alpha, args.target, poly_degree=args.polort,
            )  # fmt: skip
            soa = soa_sweep(
                reals[0], args.tr, soa_scorer, ref, args.initial_fix, args.post_fix,
                seed=args.seed,
            )  # fmt: skip
            this = [sc for sc in (soa_scorer.score(r) for r in reals[:10]) if sc]
            plot_soa(soa, {c: contrasts[c] for c in live}, this, args.target,
                     path=f"{prefix}_soa.png")  # fmt: skip
            pts = {
                f"events, {fam}": [
                    (sc["xi"], sc["liu_power"]) for per in soa["families"][fam] for sc in per[:1]
                ]
                for fam in soa["families"]
            }
            pts["blocks, 4-40 s"] = [
                (v[0]["xi"], v[0]["liu_power"]) for v in soa["blocks"].values() if v
            ]
            pts["this design"] = [(sc["xi"], sc["liu_power"]) for sc in this]
            plot_liu(pts, int(round(16.0 / args.tr)), len(reals[0].conditions),
                     path=f"{prefix}_liu.png")  # fmt: skip
        if live:
            from fastfuncstuff.simulation.plots import plot_robustness

            plot_robustness(robust, live, args.hrf, ref, args.target, path=f"{prefix}_robust.png")
        if live:
            from fastfuncstuff.simulation.plots import plot_tsnr
            from fastfuncstuff.simulation.power import RealizationScorer, effect_needed

            # One noise level at tSNR 100 with this run's physiology; the effect
            # needed scales with the noise SD, so it gives every tSNR.
            unit_noise = [{"label": "t100", "tsnr": 100.0, "phys_fraction": args.phys_fraction,
                           "tau": args.tau}]  # fmt: skip
            t_scorer = RealizationScorer(
                args.tr, {c: contrasts[c] for c in live}, unit_noise, pattern, args.hrf,
                args.alpha, args.target, poly_degree=args.polort,
            )  # fmt: skip
            t_scores = [sc for sc in (t_scorer.score(r, shape=False) for r in reals[:10]) if sc]
            grid = np.geomspace(10, 300, 60)
            curves = {
                c: np.median([sc["needed"][("t100", c)] for sc in t_scores]) * 100.0 / grid
                for c in live
            }
            need_mc = effect_needed(res, args.target)
            points = {
                c: [
                    (float(n["tsnr"]), float(np.nanmedian(need_mc[(n["label"], c)])))
                    for n in conds
                    if "arma" not in n and np.isfinite(need_mc[(n["label"], c)]).any()
                ]
                for c in live
            }
            plot_tsnr(curves, grid, points, target=args.target, path=f"{prefix}_tsnr.png")
        from fastfuncstuff.simulation.plots import plot_design_matrix

        d0 = res["designs"][0]
        plot_design_matrix(
            d0["X"].numpy(), reals[0].conditions, d0["run_lengths"], d0["poly_degree"], args.tr,
            path=f"{prefix}_matrix.png",
        )  # fmt: skip
        if sweep is not None and sweep["rows"]:
            from fastfuncstuff.simulation.plots import plot_scan_time

            plot_scan_time(
                sweep,
                [c["label"] for c in conds],
                list(contrasts),
                current_minutes=float(np.mean([sum(r.run_lengths) for r in reals])) * args.tr / 60,
                path=f"{prefix}_scantime.png",
            )
        if len(reals) > 1:
            from fastfuncstuff.simulation.plots import plot_design_spread

            score, score_label = None, None
            if args.rank_by:
                from fastfuncstuff.simulation.optimize import make_fitness

                goal = _goal(args.rank_by, contrasts, pattern, "-rank_by")
                fit = make_fitness(
                    args.tr, contrasts, conds, pattern, goal, ref, [args.hrf], args.alpha,
                    args.polort, _mean_response(args), args.trial_sd, args.target,
                    reference=reals,
                )  # fmt: skip
                score = np.array([fit(r) for r in reals])
                score_label = _objective_label(goal, args.target)
            plot_design_spread(
                quality, reals, ref, path=f"{prefix}_designs.png", score=score,
                score_label=score_label,
            )  # fmt: skip
        plot_design(
            res,
            reals[0],
            args.tr,
            corr=quality[0]["corr"],
            needed=np.median([q["needed"][ref] for q in quality], axis=0),
            needed_label=ref,
            path=f"{prefix}_design.png",
            title="Events, first realization, run 1",
        )
    written = sorted(
        str(q.name).removeprefix(prefix.name)
        for q in prefix.parent.glob(f"{prefix.name}_*")
        if q.stat().st_mtime >= started
    )
    print(f"\nwrote {prefix}: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
