"""ffs_util_eventcheck -- what your event timing lets FIR / TENT resolve.

Reads the event timing (and, optionally, the runs it goes with) and reports,
before any fitting:

1. dangers   -- designs that are singular or amplify noise (the up-down TENT
                artefact), timing that rounding would shift;
2. solutions -- the fix that fits this timing (aligned knots, smoothing,
                rounding and its cost);
3. options   -- how finely the response can be resolved: knots at TR, TR/2,
                TR/3 ... with the noise cost of each.

All numbers come from the design alone (see design/event_timing.py).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_microtime_offset_arg,
    parse_input_files,
    parse_timing_spec,
    resolve_microtime_offset,
)
from fastfuncstuff.design.event_timing import (
    NOISY_AMPLIFICATION,
    UNSTABLE_AMPLIFICATION,
    TimingReport,
    aligned_knot_start,
    check_event_timing,
    finest_usable_grid,
)

EPILOG = f"""\
Examples:
  ffs_util_eventcheck -onsets face.1D house.1D -input run*.nii.gz
  ffs_util_eventcheck -events sub-01_task-x_run-*_events.tsv -TR 1.5 -window 0 20
  ffs_util_eventcheck -onsets stim.1D -TR 2 -run_lengths 200 200 -precision 0.1

Reading the grid table:
  amplification = noise in the worst knot combination, relative to rounding the
  same events onto the samples (plain FIR at TR knots = 1.0).
    ok              <= {NOISY_AMPLIFICATION}x    fine unpenalized
    noisy           <= {UNSTABLE_AMPLIFICATION}x    usable, smoothing recommended
    unstable        >  {UNSTABLE_AMPLIFICATION}x    up-down artefacts expected without smoothing
    unidentifiable              some knot combination is never observed
"""


def create_parser() -> argparse.ArgumentParser:
    parser = FfsArgumentParser(
        prog="ffs_util_eventcheck",
        description="What event timing lets a FIR/TENT response estimate resolve: "
        "dangers, fixes, and the finest usable knot spacing.",
        epilog=EPILOG,
        formatter_class=FfsHelpFormatter,
    )
    timing = parser.add_argument_group("Timing (one of)")
    timing.add_argument(
        "-onsets",
        nargs="+",
        metavar="FILE",
        help="AFNI timing files, one per condition, one row per run.",
    )
    timing.add_argument(
        "-events", nargs="+", metavar="TSV", help="BIDS events.tsv, one per run (or one shared)."
    )
    timing.add_argument(
        "-event_ignore", nargs="+", metavar="COND", help="trial_types to leave out."
    )
    timing.add_argument(
        "-event_cols",
        nargs=3,
        metavar=("ONSET", "DURATION", "TRIAL_TYPE"),
        help="Non-standard events.tsv columns.",
    )
    runs = parser.add_argument_group("Acquisition (from -input, or given)")
    runs.add_argument(
        "-input",
        nargs="+",
        metavar="FILE",
        help="The runs: TR, run lengths and the header's sample time.",
    )
    runs.add_argument("-TR", type=float, metavar="SEC", help="Repetition time (overrides -input).")
    runs.add_argument(
        "-run_lengths",
        nargs="+",
        type=int,
        metavar="N",
        help="Volumes per run. Default without -input: last onset + window.",
    )
    add_microtime_offset_arg(runs)
    model = parser.add_argument_group("Response model")
    model.add_argument(
        "-window",
        nargs=2,
        type=float,
        default=[0.0, 16.0],
        metavar=("BOT", "TOP"),
        help="Response window in seconds after onset.",
    )
    model.add_argument(
        "-precision",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Quantize onsets to this resolution first, so logged digits (10.2467 s) "
        "don't read as timing diversity. 0 = use as given.",
    )
    model.add_argument(
        "-max_subdivision",
        type=int,
        default=4,
        metavar="M",
        help="Check knot spacings TR, TR/2, ... TR/M.",
    )
    model.add_argument(
        "-polort", type=int, default=2, help="Drift polynomial order per run (as in the GLM)."
    )
    return parser


def _phase_histogram(phases: np.ndarray, n_bins: int = 10, width: int = 40) -> list[str]:
    counts, edges = np.histogram(phases, bins=n_bins, range=(0.0, 1.0))
    top = max(int(counts.max()), 1)
    return [
        f"    {edges[i]:.1f}-{edges[i + 1]:.1f} TR  {'#' * int(round(width * c / top)):<{width}} {c}"
        for i, c in enumerate(counts)
    ]


def _tent_spec(bot: float, top: float, dt: float) -> tuple[float, int]:
    n = int(round((top - bot) / dt)) + 1
    return bot + (n - 1) * dt, n


def format_report(report: TimingReport) -> str:
    tr = report.tr
    bot, top = report.window
    out = ["", "ffs_util_eventcheck", "=" * 60]
    out.append(
        f"TR {tr:g} s · sample time {report.microtime_offset:g} s into each TR · "
        f"onsets quantized to {report.precision:g} s · window {bot:g}-{top:g} s"
    )
    for label, n in zip(report.condition_labels, report.n_events, strict=True):
        out.append(f"  {label}: {n} events")
    out += ["", "Onset phase within the TR (0 = on a sample, 0.5 = midway):"]
    out += _phase_histogram(report.phases)
    out.append(
        f"  concentration {report.phase_concentration:.2f} (1 = one shared phase, 0 = spread) · "
        f"alternation visibility {report.alternation_visibility:.2f} (1 = locked, 0 = all mid-TR)"
    )

    base = report.grids[0]
    aligned = aligned_knot_start(report)
    dangers, solutions = [], []
    if base.status == "unidentifiable":
        dangers.append(
            "TENT with knots every TR is SINGULAR for this timing: some knot "
            "combination is never sampled, so the response carries an arbitrary up-down component."
        )
    elif base.status in ("unstable", "noisy"):
        dangers.append(
            f"TENT with knots every TR amplifies noise {base.amplification:.1f}x in its worst "
            f"direction ({base.status}) -- the up-down artefact."
        )
    if report.rounding_shift_max > 0.25 * tr:
        kind = "shifts every event by" if aligned is not None else "moves events by up to"
        dangers.append(
            f"Rounding onto the samples (FIR, -round-onsets) {kind} {report.rounding_shift_max:.2f} s "
            f"(RMS {report.rounding_shift_rms:.2f} s): a "
            + ("timing bias." if aligned is not None else "blur of the response.")
        )
    few = [lab for lab, n in zip(report.condition_labels, report.n_events, strict=True) if n < 8]
    if few:
        dangers.append(
            f"Few events ({', '.join(few)}): every knot estimate is noisy regardless of grid."
        )
    heavy = [
        lab
        for lab, g in zip(report.condition_labels, base.median_gain, strict=True)
        if base.identifiable and g > 3.0
    ]
    if heavy:
        dangers.append(
            f"Heavy overlap ({', '.join(heavy)}): median knot noise gain > 3 even at TR knots."
        )

    if aligned is not None:
        # The sample nearest the onset: just after it, or just before when the
        # phase is late (a knot at a small negative lag still pins lag ~0).
        a_bot = aligned - tr if aligned > tr / 2 else aligned
        a_top, a_n = _tent_spec(a_bot, top, tr)
        solutions.append(
            f"Every event shares one phase: put the knots ON the samples -- exact FIR at the true "
            f"lags, no rounding shift, no amplification:  -model TENT -window {a_bot:.3g} {a_top:.3g} "
            f"-tent-n-basis {a_n}"
        )
    if base.status == "ok":
        solutions.append("TENT with knots every TR is well conditioned for this timing.")
    else:
        solutions.append(
            "ffs_deconvolve -tent-smooth: roughness penalty, strength chosen per voxel (REML); "
            "stable for any timing."
        )
    if report.rounding_shift_max > 1e-6:
        solutions.append(
            f"-round-onsets 0.5 -model FIR: always stable, at the cost of the "
            f"{report.rounding_shift_max:.2f} s shift above."
        )
    else:
        solutions.append("Onsets already sit on the samples: FIR is exact.")

    out += ["", "Dangers:"] + [f"  - {d}" for d in dangers or ["none found"]]
    out += ["", "Solutions:"] + [f"  - {s}" for s in solutions]
    out += ["", "Options -- knot spacing (TENT over the window):"]
    out.append(
        f"    {'spacing':>12} {'knots':>6} {'status':>15} {'amplif.':>8} {'median gain':>12}"
    )
    for m, g in enumerate(report.grids, start=1):
        _, n = _tent_spec(bot, top, g.knot_dt)
        amp = "inf" if not np.isfinite(g.amplification) else f"{g.amplification:.2f}"
        med = f"{max(g.median_gain):.2f}" if g.identifiable else "-"
        label = "TR" if m == 1 else f"TR/{m}"
        out.append(
            f"    {label + f' ({g.knot_dt:.3g}s)':>12} {n:>6} {g.status:>15} {amp:>8} {med:>12}"
        )
    finest = finest_usable_grid(report)
    smooth_ok = finest_usable_grid(report, allow=("ok", "noisy", "unstable"))
    if finest is not None:
        f_top, f_n = _tent_spec(bot, top, finest.knot_dt)
        out.append(
            f"  Finest without smoothing: {finest.knot_dt:.3g} s  (-window {bot:g} {f_top:.3g} "
            f"-tent-n-basis {f_n})"
        )
    if smooth_ok is not None and (finest is None or smooth_ok.knot_dt < finest.knot_dt):
        s_top, s_n = _tent_spec(bot, top, smooth_ok.knot_dt)
        out.append(
            f"  Finest with -tent-smooth: {smooth_ok.knot_dt:.3g} s  (-window {bot:g} {s_top:.3g} "
            f"-tent-n-basis {s_n} -tent-smooth)"
        )
    if any(g.status == "unidentifiable" for g in report.grids):
        out.append(
            "  -tent-smooth also solves the unidentifiable spacings, but there the knots "
            "between observed lags come from the smoothness prior, not from the data."
        )
    if (
        finest is not None
        and finest.knot_dt >= tr - 1e-9
        and (smooth_ok is None or smooth_ok.knot_dt >= tr - 1e-9)
    ):
        out.append("  The timing does not resolve anything finer than the TR: stay with TR knots.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    input_files = parse_input_files(args.input) if args.input else []
    if args.TR is None and not input_files:
        print("ERROR: give -input or -TR", file=sys.stderr)
        return 1

    lengths: list[int] | None = args.run_lengths
    tr = args.TR
    if input_files:
        from fastfuncstuff.io.dsetinfo import read_info

        infos = [read_info(p) for p in input_files]
        tr = tr if tr is not None else infos[0].tr
        lengths = lengths or [int(i.shape[3]) for i in infos]
    n_runs = len(lengths) if lengths else (len(args.events) if args.events else None)
    if n_runs is None:
        with open(args.onsets[0]) as fh:
            n_runs = sum(1 for ln in fh if ln.strip() and not ln.lstrip().startswith("#"))
    try:
        timing = parse_timing_spec(
            events=args.events,
            onsets=args.onsets,
            durations_arg=None,
            n_runs=n_runs,
            event_ignore=args.event_ignore,
            event_cols=tuple(args.event_cols) if args.event_cols else None,
            input_files=input_files or None,
            verbose=False,
            allow_missing_durations=True,
        )
        offset = resolve_microtime_offset(args.microtime_offset, input_files, tr, verbose=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    bot, top = args.window
    if lengths is None:
        lengths = []
        for r in range(n_runs):
            last = max(
                (float(np.max(c[r])) for c in timing.all_onsets if np.size(c[r])), default=0.0
            )
            lengths.append(int(np.ceil((last + top) / tr)) + 1)
    report = check_event_timing(
        timing.all_onsets,
        lengths,
        tr,
        window=(bot, top),
        microtime_offset=offset,
        precision=args.precision,
        max_subdivision=args.max_subdivision,
        polort=args.polort,
        condition_labels=timing.condition_labels,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
