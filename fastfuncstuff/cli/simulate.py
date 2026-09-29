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
needed; naive OLS false positives are reported alongside. Amplitudes are peak
percent signal change of an isolated event; amplitude 0 is always simulated
so the false-positive rate is measured.

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
PREFIX_design.png, PREFIX_spec.json, and PREFIX_events/ (timing files of the
first realization, for a described experiment).
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
    t.add_argument("-tr", type=float, required=True, help="Repetition time (s).")
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
        help="Peak % signal change to sweep: values, or START:STOP:NUM.",
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


def _summarise(res, reals, conds, contrasts, pattern, args, spec_text, profile_text) -> str:
    from fastfuncstuff.simulation.power import amplitude_for_power

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
    out.append(
        "response pattern: "
        + ", ".join(f"{c}={w:g}" for c, w in zip(reals[0].conditions, pattern, strict=True))
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
    over = "realizations" + (" x true HRFs" if len(truths) > 1 else "")
    out.append(f"Amplitude (% signal change) for 80% power -- median [range] over {over}")
    per_design = {}
    for d in range(len(reals)):
        for th in truths:
            sub = {"table": [r for r in rows if r["design"] == d and r["true_hrf"] == th]}
            per_design[(d, th)] = amplitude_for_power(sub, 0.8)
    unreached = False
    header = f"{'noise':<24}" + "".join(f"{c:>18}" for c in contrasts)
    out.append(header)
    for cond in conds:
        cells = []
        for c in contrasts:
            vals = np.array([per_design[d][(cond["label"], c)] for d in per_design])
            if abs(contrasts[c] @ np.asarray(pattern)) == 0:
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
            if abs(contrasts[c] @ np.asarray(pattern)) == 0:
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

    if args.effect is not None:
        out += ["", f"At {args.effect:g}% signal change (power, analytic / Monte Carlo):"]
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
                pa = float(np.median([r["power_predicted"] for r in sel]))
                pm = float(np.mean([r["power"] for r in sel]))
                if abs(sel[0]["true_effect"]) < 1e-12:
                    cells.append(f"{'no true effect':>22}")
                else:
                    cells.append(f"{pa:>6.2f} / {pm:.2f} {_verdict(pa):>9}")
            out.append(f"{cond['label']:<24}" + "".join(cells))
        out.append("  hopeless < 0.2 <= marginal < 0.8 <= good")
    return "\n".join(out)


# ---------------------------------------------------------------- plotting
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# Sequential blue, 200 -> 700: tSNR bins are ordered, so they are magnitude, not identity.
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _ramp(n: int) -> list[str]:
    if n <= 1:
        return [BLUES[3]]
    if n > len(BLUES):
        import matplotlib.colors as mcolors

        cmap = mcolors.LinearSegmentedColormap.from_list("seq", [BLUES[0], BLUES[-1]])
        return [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, n)]
    idx = np.linspace(0, len(BLUES) - 1, n).round().astype(int)
    return [BLUES[i] for i in idx]


def _plot_power(res, conds, contrasts, pattern, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in contrasts if abs(contrasts[c] @ np.asarray(pattern)) > 0][:6]
    if not names:
        return
    rows = res["table"]
    colors = _ramp(len(conds))
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(axes[0], names, strict=True):
        _style(ax)
        ax.axhline(0.8, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for cond, col in zip(conds, colors, strict=True):
            sel = [r for r in rows if r["noise"] == cond["label"] and r["contrast"] == c]
            amps = sorted({r["amplitude"] for r in sel})
            by = {a: [r for r in sel if r["amplitude"] == a] for a in amps}
            med = [np.median([r["power_predicted"] for r in by[a]]) for a in amps]
            lo = [np.min([r["power_predicted"] for r in by[a]]) for a in amps]
            hi = [np.max([r["power_predicted"] for r in by[a]]) for a in amps]
            mc = [np.mean([r["power"] for r in by[a]]) for a in amps]
            ax.fill_between(amps, lo, hi, color=col, alpha=0.18, linewidth=0)
            ax.plot(amps, med, color=col, linewidth=2, label=cond["label"])
            ax.plot(
                amps, mc, "o", color=col, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1
            )
            if len(conds) <= 4:
                # Label each curve where it crosses 50% power: saturated curves all
                # end at 1.0, so labels at the right edge land on top of each other.
                k = int(np.argmin(np.abs(np.asarray(med) - 0.5)))
                ax.annotate(
                    cond["label"].split(" (")[0],
                    (amps[k], med[k]),
                    xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK2,
                    va="top",
                )
        if args.effect is not None:
            ax.axvline(args.effect, color=INK2, linewidth=1, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(c, color=INK, fontsize=11)
        ax.set_xlabel("amplitude (% signal change x pattern)", color=INK2, fontsize=9)
    axes[0][0].set_ylabel("power", color=INK2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="lower center",
        ncol=min(len(labels), 5),
    )
    fig.suptitle(
        f"Power at two-tailed p < {args.alpha:g} -- line: analytic (band: range over "
        f"realizations), dots: Monte Carlo; dashed: 80%",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _plot_design(res, reals, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    real = reals[0]
    X = res["designs"][0]["X"].numpy()
    n0 = real.run_lengths[0]
    t = np.arange(n0) * args.tr
    fig = plt.figure(figsize=(13, 5.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 1.4])
    ax_ev, ax_x, ax_c = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[:, 1]),
    )
    for ax in (ax_ev, ax_x):
        _style(ax)
    for i, (cond, dur) in enumerate(zip(real.conditions, real.durations, strict=True)):
        col = CATEGORICAL[i % len(CATEGORICAL)]
        for on in real.onsets[i][0]:
            ax_ev.broken_barh([(on, max(dur, args.tr / 4))], (i + 0.15, 0.7), color=col)
        ax_x.plot(t, X[:n0, i], color=col, linewidth=2, label=cond)
    ax_ev.set_yticks(np.arange(len(real.conditions)) + 0.5, real.conditions)
    ax_ev.set_xlim(0, t[-1] + args.tr)
    ax_ev.set_title("Events, first realization, run 1", color=INK, fontsize=10, loc="left")
    ax_x.set_xlim(0, t[-1] + args.tr)
    ax_x.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax_x.set_title("Regressors (unit peak)", color=INK, fontsize=10, loc="left")
    ax_x.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=min(4, len(real.conditions)))

    corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.ones((1, 1))
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    ax_c.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(corr):
        ax_c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax_c.set_xticks(range(len(real.conditions)), real.conditions)
    ax_c.set_yticks(range(len(real.conditions)), real.conditions)
    ax_c.tick_params(colors=INK2, labelsize=9)
    ax_c.set_title("Regressor correlation", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _write_events(real, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, cond in enumerate(real.conditions):
        with open(outdir / f"{cond}.txt", "w") as f:
            for run in real.onsets[i]:
                f.write((" ".join(f"{t:.3f}" for t in run) if len(run) else "*") + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.simulation.experiment import default_contrasts, parse_contrast, realize
    from fastfuncstuff.simulation.power import simulate_realizations_power

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
        poly_degree=args.polort,
        device=device,
        seed=args.seed,
    )

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = _summarise(res, reals, conds, contrasts, pattern, args, spec_text, profile_text)
    print(summary)
    Path(f"{prefix}_summary.txt").write_text(summary + "\n")

    cols = [
        "design",
        "true_hrf",
        "noise",
        "tsnr",
        "amplitude",
        "contrast",
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
                "amplitudes": amps,
                "noise": conds,
                "seeds": [r.seed for r in reals],
                "run_lengths": [r.run_lengths for r in reals],
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
        _write_events(reals[0], Path(f"{prefix}_events"))
    if not args.no_plots:
        _plot_power(res, conds, contrasts, pattern, args, Path(f"{prefix}_power.png"))
        _plot_design(res, reals, args, Path(f"{prefix}_design.png"))
    print(
        f"\nwrote {prefix}_summary.txt, _power.tsv, _spec.json"
        + ("" if args.no_plots else ", _power.png, _design.png")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
