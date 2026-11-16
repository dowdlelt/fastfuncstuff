"""CLI for RF (k-space) spike detection and repair from magnitude EPI.

Command: ffs_util_spikehunt (registered as entry point in pyproject.toml)

Not a despiker in the 3dDespike sense (which clips voxel time series): this
finds slices whose 2D spectrum has a transient, compact bright blob, the
signature of a radio-frequency spike during that slice's readout, and repairs
only those k-points. See ``fastfuncstuff/processing/spikehunt.py``.

Times are reported per excitation: volume * TR + that slice's ``SliceTiming``
offset. Multiband partner slices share an excitation and are one event. A
sequence with a silent gap in the TR (TA < TR) cannot see spikes in the gap;
the report says how much of the TR was being read out.

Usage:
    ffs_util_spikehunt -input bold.nii -prefix sub01_spikes -plots
    ffs_util_spikehunt -input bold.nii -prefix sub01_spikes -clean -plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from fastfuncstuff.processing import spikehunt as S

    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_util_spikehunt",
        description="Detect (and optionally repair) RF spikes: per-slice transient "
        "blobs in the 2D spectrum of magnitude EPI. Reports hit volumes and times "
        "with slice timing and the TR gap taken into account.",
    )
    parser.add_argument("-input", required=True, help="4D magnitude series (one run).")
    parser.add_argument(
        "-prefix",
        required=True,
        help="Output prefix: PREFIX.spikes.tsv (one row per hit slice), "
        "PREFIX.score.tsv (volume x slice scores), PREFIX.spikehunt.png with -plots, "
        "PREFIX.clean.nii.gz with -clean.",
    )
    parser.add_argument(
        "-tpattern",
        default=None,
        metavar="FILE",
        help="Slice timing: a BIDS JSON (SliceTiming, RepetitionTime) or a text file "
        "of one offset per slice. Default: the JSON sidecar beside -input. Without "
        "one, hits are reported by volume only.",
    )
    parser.add_argument(
        "-TR",
        type=float,
        default=None,
        help="Repetition time in seconds (default: JSON RepetitionTime, else header).",
    )
    parser.add_argument(
        "-thresh",
        type=float,
        default=S.DEFAULT_THRESHOLD,
        help="Candidate threshold on the slice-normalised blob score (a robust z "
        "against the same slice's other volumes). Candidates then pass a point check "
        "(-point_frac), whatever their score. On hand-labelled runs nearly every "
        "candidate above 6 was a spike; on a clean run, motion frames reached 14.",
    )
    parser.add_argument(
        "-point_frac",
        type=float,
        default=S.DEFAULT_POINT_FRAC,
        help="Point check: a candidate is a spike when at least this fraction of its "
        "frame's excess k-space energy sits in its 18 strongest k-points (one blob and "
        "its mirror), with that energy away from the k centre. Inflow and motion "
        "spread theirs (median 0.05 against 0.16 for spikes).",
    )
    parser.add_argument(
        "-window",
        type=int,
        default=S.DEFAULT_WINDOW,
        help="Running-median width (volumes) used as each k-point's baseline. Must "
        "be odd and wider than twice the longest run of consecutive hits.",
    )
    parser.add_argument(
        "-exclude_radius",
        type=float,
        default=S.DEFAULT_EXCLUDE_RADIUS,
        help="Radius (fraction of k-max per axis) of the central k-space ellipse left "
        "out of the score: motion, inflow and global signal change live there. "
        "0.25 is ~5%% of k-space; a spike landing inside it is not seen.",
    )
    parser.add_argument(
        "-clean",
        action="store_true",
        help="Repair the hits and write PREFIX.clean.nii.gz (float32). Only hit "
        "frames of hit slices change.",
    )
    parser.add_argument(
        "-repair",
        choices=S.REPAIR_METHODS,
        default="kinterp",
        help="With -clean. kclip: clip each k-point's deviation from the temporal "
        "neighbours to -clip times that k-point's normal deviation (phase kept); "
        "normal k-points pass unchanged; removes ~100%% of the spike power. kinterp: "
        "replace only the outlier k-points (> -kinterp_z) by the temporal "
        "interpolation; ~8%% of k-space, removes ~75%%. slice: replace the whole slice "
        "by temporal interpolation, which also replaces the frame's own state.",
    )
    parser.add_argument(
        "-clip",
        type=float,
        default=S.DEFAULT_CLIP,
        help="With -repair kclip: the limit, in multiples of the median normal "
        "deviation. 1.5 brought hit frames to a normal frame's noise level on the "
        "test runs; 2 left ~25%% of the spike power.",
    )
    parser.add_argument(
        "-kinterp_z",
        type=float,
        default=S.DEFAULT_KINTERP_Z,
        help="With -repair kinterp: k-points deviating from the temporal interpolation "
        "by more than this many times their normal deviation are replaced by it.",
    )
    parser.add_argument(
        "-noise_vols",
        default="auto",
        help="Trailing noise scans (flip angle 0, e.g. for NORDIC) to leave out: a "
        "count, or auto = trailing volumes darker than 20%% of the median volume.",
    )
    parser.add_argument(
        "-plots",
        action="store_true",
        help="Write PREFIX.spikehunt.png: score heatmap, per-volume trace, within-TR "
        "timing histogram, and the worst hits (before/after with -clean).",
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    args = parser.parse_args(argv)
    if args.window < 3 or args.window % 2 == 0:
        parser.error("-window must be an odd integer >= 3")
    if args.noise_vols != "auto" and not args.noise_vols.isdigit():
        parser.error("-noise_vols takes a non-negative integer or 'auto'")
    if args.clip <= 0.0:
        parser.error("-clip must be > 0")
    return args


def main(argv: list[str] | None = None) -> None:
    import numpy as np

    from fastfuncstuff.processing import spikehunt as S
    from fastfuncstuff.processing.io import load_image, save_image

    args = parse_args(argv)
    device = setup_device(args.device)
    verb = args.verb
    t0 = time.time()

    with spinner(f"Loading {Path(args.input).name}", enabled=verb >= 1):
        data, hdr = load_image(args.input)
    if data.ndim != 4:
        raise SystemExit(f"ffs_util_spikehunt: {args.input} is not 4D (shape {tuple(data.shape)})")
    n_noise = S.trailing_noise_volumes(data) if args.noise_vols == "auto" else int(args.noise_vols)
    nt_all, nz = data.shape[0], data.shape[1]
    nt = nt_all - n_noise
    if nt < 5:
        raise SystemExit(
            f"ffs_util_spikehunt: only {nt} volumes left after {n_noise} noise volumes"
        )
    tr, slice_times, tsrc = S.resolve_timing(args.input, args.tpattern, args.TR)
    if slice_times is not None and len(slice_times) != nz:
        print(
            f"  ** {tsrc} has {len(slice_times)} slice times for {nz} slices: ignoring it",
            file=sys.stderr,
        )
        slice_times = None
    acq = S.acquisition_window(tr, slice_times)

    cleaned = data.clone() if args.clean else None
    scan = S.hunt_spikes(
        data[:nt],
        threshold=args.thresh,
        window=args.window,
        exclude_radius=args.exclude_radius,
        clean=args.clean,
        out=None if cleaned is None else cleaned[:nt],
        method=args.repair,
        clip=args.clip,
        point_frac=args.point_frac,
        kinterp_z=args.kinterp_z,
        slice_times=slice_times,
        compute_device=device,
        progress=verb >= 1,
    )
    events = S.group_events(scan, acq)

    if verb >= 1:
        print(
            f"ffs_util_spikehunt: {Path(args.input).name}  ({nz} slices x {nt} volumes, timing: {tsrc})"
        )
        if n_noise:
            print(f"  {n_noise} trailing noise volumes left out (kept unchanged in outputs)")
        S.print_report(scan, events, acq, nt)

    S.write_hits_tsv(f"{args.prefix}.spikes.tsv", scan, acq)
    S.write_score_tsv(f"{args.prefix}.score.tsv", scan)
    if cleaned is not None:
        save_image(cleaned, f"{args.prefix}.clean.nii.gz", hdr)
        n_left = 0 if scan.residual is None else len(scan.residual)
        if verb >= 1:
            fr = list(scan.clipped_frac.values())
            how = (
                f", median {100 * float(np.median(fr)):.1f}% of k-points changed"
                if args.repair != "slice" and fr
                else ""
            )
            print(
                f"  repaired {len(scan.repair)} slice-volumes ({args.repair}{how}); rescored, "
                f"{n_left} still > {args.thresh:g} (max {scan.residual_max:.1f})"
            )
    if args.plots:
        S.plot_spikehunt(
            f"{args.prefix}.spikehunt.png",
            scan,
            acq,
            data[:nt],
            None if cleaned is None else cleaned[:nt],
            title=Path(args.input).name,
            window=args.window,
        )
    if verb >= 1:
        sys.stdout.flush()
        print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
