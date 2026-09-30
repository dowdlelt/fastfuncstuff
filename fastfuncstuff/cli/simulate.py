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
    -null DUR COUNT               blank trials (time with no event)
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
    -scan_time S       S seconds per run (fixed volumes). Families without a
                       total are scaled to fill it on average; with every
                       family fixed it only sets the run length (leftover time
                       is trailing fixation). Events jitter pushes past the end
                       are dropped and counted, as on a scanner.

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

Examples
--------
    # 2 conditions, 20 trials each per run, exponential jitter, 2 runs, TR 1.25
    ffs_simulate -tr 1.25 -nruns 2 -trial A 2 20 -trial B 2 20 -null 2 10 \\
        -isi exp:4,2,12 -initial_fix 10 -post_fix 16 -pattern A=1 B=0 \\
        -effect 1 -prefix sim/er

    # miniblocks of A then B, 1 s apart, noise calibrated from a real dataset
    ffs_simulate -tr 2 -miniblock AB "A:2,B:2" 12 -within_isi 1 -isi uniform:8,12 \\
        -noise_profile stats_REMLvar+tlrc.HEAD TSNR+tlrc.HEAD -noise_mask mask+tlrc.HEAD \\
        -contrast A-B -prefix sim/mb

Outputs: PREFIX_summary.txt, PREFIX_power.tsv, PREFIX_power.png,
PREFIX_design.png, PREFIX_spec.json, PREFIX_events/ (timing files of the
first realization, for a described experiment), PREFIX_voxels.png (an active
and a silent voxel at each noise level, every condition at -effect, or else at
the effect that level needs for 80% power),
and PREFIX_hrf.png when the true HRF differs from the fitted one.

COMPARE -- designs simulated separately, side by side:
    ffs_simulate -compare sim/cycle_power.tsv sim/jittered_power.tsv -prefix sim/cmp
Each file is one design, summarised over its realizations (median and range
of the amplitude needed for -target power). Designs should share the noise
levels and contrasts, and a scan time -- a longer scan wins by having more
data, so differing scan times are flagged.
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

VERDICTS = ((0.2, "hopeless"), (0.8, "marginal"), (1.01, "good"))


def _build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_simulate", description=__doc__, formatter_class=FfsHelpFormatter
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
        help="Blank trials: DUR seconds, COUNT per run (repeatable).",
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
    t.add_argument("-scan_time", type=float, metavar="S", help="Seconds per run (fixed volumes).")
    t.add_argument("-num_events", type=int, metavar="N", help="-trial units per run in total.")
    t.add_argument("-num_blocks", type=int, metavar="N", help="-block/-miniblock units per run.")
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
    from fastfuncstuff.cli_utils import add_device_arg

    add_device_arg(a)
    p.add_argument("-prefix", required=True, help="Output prefix (a directory is created).")
    p.add_argument("-no_plots", action="store_true", help="Skip the PNG figures.")

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
    c.add_argument("-target", type=float, default=0.8, help="Power to reach (default 0.8).")
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


def _spec_from_args(args):
    from fastfuncstuff.simulation.experiment import NULL, ExperimentSpec, Interval, Unit

    units = [Unit.parse(nm, f"{nm}:{d}", int(c)) for nm, d, c in (args.trial or [])]
    units += [Unit.parse(nm, f"{nm}:{d}", int(c), "block") for nm, d, c in (args.block or [])]
    units += [Unit.parse(nm, items, int(c), "block") for nm, items, c in (args.miniblock or [])]
    units += [
        Unit.parse(f"null{i}", f"{NULL}:{d}", int(c)) for i, (d, c) in enumerate(args.null or [])
    ]
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
    res, reals, conds, contrasts, pattern, args, spec_text, profile_text, quality
) -> str:
    from fastfuncstuff.simulation.power import effect_needed, has_true_effect, is_difference

    rows = res["table"]
    out = ["ffs_simulate", "=" * 72]
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
            f"note: {dropped} event(s) over {len(reals)} realization(s) fell past -scan_time "
            "and were dropped"
        )
    out.append("events per condition: " + ", ".join(f"{c} {np.mean(v):g}" for c, v in n_ev.items()))
    out += ["", *_quality_lines(quality, reals[0].conditions, conds)]
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
        "Effect (% signal change: amplitude, or the difference for A-B) for 80% power -- "
        f"median [range] over {over}"
    )
    need = effect_needed(res, 0.8)
    unreached = False
    header = f"{'noise':<24}" + "".join(f"{c:>18}" for c in contrasts)
    out.append(header)
    for cond in conds:
        cells = []
        for c in contrasts:
            vals = need[(cond["label"], c)]
            if not has_true_effect(contrasts[c], pattern):
                cells.append(f"{'no true effect':>18}")
            elif np.all(np.isnan(vals)):
                cells.append(f"{'> ' + format(max(r['amplitude'] for r in rows), 'g'):>18}")
            else:
                med = np.nanmedian(vals)
                text = f"{med:.2f} [{np.nanmin(vals):.2f}-{np.nanmax(vals):.2f}]"
                if np.isnan(vals).any():
                    text += "*"
                    unreached = True
                cells.append(f"{text:>18}")
        out.append(f"{cond['label']:<24}" + "".join(cells))
    if unreached:
        out.append(
            f"  * some realizations/HRFs never reach 80% within the sweep (max "
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
    out += ["", "False-positive rate at amplitude 0 (should be ~alpha):"]
    for cond in conds:
        nulls = [r for r in rows if r["noise"] == cond["label"] and r["amplitude"] == 0.0]
        fp = np.mean([r["power"] for r in nulls])
        fpn = np.mean([r["power_naive"] for r in nulls])
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
    res, conds, contrasts, pattern, effect
) -> tuple[list[float], str, list[str] | None]:
    """Amplitude for each noise row of the example-voxel figure, what it is, and row notes.

    -effect when given. Otherwise the effect each noise level needs for 80%
    power (median over realizations), on the first contrast with a true effect
    -- a fixed default (1%) was below detectability at tSNR 50 and invisible
    against tSNR 20's noise, so the picture showed nothing the design could
    find. A level that never reaches 80% shows the top of the sweep.
    """
    from fastfuncstuff.simulation.power import effect_needed, has_true_effect

    if effect is not None:
        return [effect] * len(conds), f"{effect:g}% (-effect)", None
    live = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
    if not live:
        return [1.0] * len(conds), "1% (no contrast has a true effect)", None
    need = effect_needed(res, 0.8)
    top = max(r["amplitude"] for r in res["table"])
    amps, notes = [], []
    for cond in conds:
        v = need.get((cond["label"], live[0]), np.full(1, np.nan))
        reached = bool(np.isfinite(v).any())
        amps.append(float(np.nanmedian(v)) if reached else top)
        notes.append("" if reached else "; 80% not reached, top of the sweep")
    return amps, f"the effect {live[0]} needs for 80% power at each level", notes


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


def _quality_lines(quality, conditions, conds) -> list[str]:
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
        f"  effect for 80% power at {ref} (analytic, % signal change): diagonal = condition "
        "vs baseline, below it = the difference"
    )
    w = max(8, max(len(c) for c in conditions) + 2)
    out.append("    " + " " * w + "".join(f"{c:>{w}}" for c in conditions))
    for i, c in enumerate(conditions):
        out.append("    " + f"{c:<{w}}" + "".join(f"{m[i, j]:>{w}.2f}" for j in range(i + 1)))
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
    if known and (max(known) - min(known)) > 0.02 * max(known):
        out.append(
            "WARNING: scan times differ -- the longer design wins partly by having more data"
        )
    contrasts = list(dict.fromkeys(r["contrast"] for r in rows))
    noises = list(dict.fromkeys(r["noise"] for r in rows))
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
    return 0


def main(argv: list[str] | None = None) -> int:
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
        reals, args.tr, conds, hrf=args.hrf, alpha=args.alpha, poly_degree=args.polort
    )
    bad = [i for i, q in enumerate(quality) if q["deficient"]]
    if bad:
        print(
            f"ERROR: {_rank_message(quality[bad[0]], conditions, bad, len(reals))}", file=sys.stderr
        )
        return 1

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
    summary = _summarise(
        res, reals, conds, contrasts, pattern, args, spec_text, profile_text, quality
    )
    print(summary)
    Path(f"{prefix}_summary.txt").write_text(summary + "\n")

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
                res, labels, contrasts, pattern, args.alpha, args.effect, path=f"{prefix}_power.png"
            )
        if res["true_hrfs"] != [res["hrf"]]:
            from fastfuncstuff.simulation.plots import plot_hrf_recovery

            effective = [c for c, w in contrasts.items() if has_true_effect(w, pattern)]
            if effective:
                plot_hrf_recovery(res, args.tr, effective[0], path=f"{prefix}_hrf.png")
        from fastfuncstuff.simulation.plots import plot_example_voxels

        truth = res["true_hrfs"][0] if res["true_hrfs"] != [res["hrf"]] else res["hrf"]
        amp, basis, notes = _voxel_amplitudes(res, conds, contrasts, pattern, args.effect)
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
        ref = _reference_noise(conds)
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
    print(
        f"\nwrote {prefix}_summary.txt, _power.tsv, _spec.json"
        + ("" if args.no_plots else ", _power.png, _design.png")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
