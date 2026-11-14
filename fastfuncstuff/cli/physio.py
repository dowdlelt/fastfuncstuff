"""ffs_physio: cardiac and respiratory regressors from the fMRI data, no recordings.

Thin wrapper: parse -> validate -> processing/physio.py -> write. See that module
and the wiki note *Physiological noise from the data* for the method and its
validation against belt and pulse-oximeter regressors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_physio",
        description=(
            "Cardiac and respiratory noise regressors estimated from the data itself.\n\n"
            "Each multiband excitation is one physiological sample, so a run read in\n"
            "acquisition order is a trace at n_excitations/TR (17 Hz for MB5, 85 slices,\n"
            "TR 1 s) -- fast enough for the heartbeat that the volume rate aliases.\n"
            "Cardiac is read from magnitude and respiration from phase (when given).\n\n"
            "Run it on RAW slices: before motion correction or slice timing, which move\n"
            "samples away from their acquisition times. Output regressors are per slice,\n"
            "evaluated at each slice's own acquisition time, in the same 13-per-slice\n"
            "slice-major layout as AFNI physio_calc.py (ffs_reml -slibase_sm, or\n"
            "afni_proc -ricor_regs)."
        ),
    )
    parser.add_argument(
        "-input", required=True, help="Raw magnitude 4-D series (.nii/.nii.gz/.nii.zst)."
    )
    parser.add_argument(
        "-phase",
        default=None,
        help="Raw phase 4-D series matching -input (scanner units or radians; no unwrap "
        "needed -- only its change over time is used). Without it respiration is read "
        "from magnitude, which was unreliable on task runs.",
    )
    parser.add_argument(
        "-prefix",
        required=True,
        help="Output prefix (PREFIX.slibase.1D, .traces.tsv, .rates.tsv, .json, .qc.png).",
    )
    parser.add_argument(
        "-tpattern",
        default=None,
        help="Slice timing: BIDS JSON with SliceTiming, or a text file (one time per "
        "line, seconds). Default: the -input sidecar JSON.",
    )
    parser.add_argument(
        "-TR",
        type=float,
        default=None,
        help="Repetition time (s). Default: sidecar JSON, then header.",
    )
    parser.add_argument(
        "-drop_last",
        type=int,
        default=None,
        help="Volumes to drop from the end (e.g. NORDIC noise scans). Default: detect "
        "volumes whose mean signal collapses. The regressors cover the kept volumes.",
    )
    parser.add_argument(
        "-card_from",
        choices=["auto", "magnitude", "phase"],
        default="auto",
        help="Trace to read cardiac from. auto = magnitude (measured best).",
    )
    parser.add_argument(
        "-resp_from",
        choices=["auto", "magnitude", "phase"],
        default="auto",
        help="Trace to read respiration from. auto = phase when given, else magnitude.",
    )
    parser.add_argument("-no_qc", action="store_true", help="Skip the QC spectrogram PNG.")
    add_device_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args(argv)


def _slice_times_and_tr(args) -> tuple[np.ndarray, float]:
    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.processing.slicetime import load_slice_timing

    src = args.tpattern
    if src is None:
        stem = str(args.input)
        for ext in (".nii.gz", ".nii.zst", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        src = stem + ".json"
        if not Path(src).exists():
            raise SystemExit("ffs_physio: no -tpattern and no sidecar JSON beside -input.")
    st = np.asarray(load_slice_timing(src), dtype=float)
    tr = args.TR
    if tr is None and str(src).endswith(".json"):
        tr = json.loads(Path(src).read_text()).get("RepetitionTime")
    if tr is None:
        tr = get_tr_from_file(args.input)
    if not tr or tr <= 0:
        raise SystemExit("ffs_physio: no TR in the sidecar or header; pass -TR.")
    return st, float(tr)


def _write_slibase(path: Path, X: np.ndarray, labels: list[str]) -> None:
    """AFNI-style 1D with its column labels in a comment, as physio_calc.py writes."""
    with open(path, "w") as fh:
        fh.write("# <RetroTSout\n")
        fh.write(f'# ni_type = "{X.shape[1]}*double"\n')
        fh.write(f'# ni_dimen = "{X.shape[0]}"\n')
        fh.write('# ColumnLabels = "' + " ; ".join(labels) + '"\n# >\n')
        np.savetxt(fh, X, fmt="%.6f")
        fh.write("# </RetroTSout>\n")


def _qc_figure(path: Path, traces, sources: dict, diag: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for ax, (name, band, key) in zip(
        axes,
        (
            ("cardiac", (0.5, 2.2), "cardiac_rate_hz"),
            ("respiration", (0.05, 0.8), "respiration_rate_hz"),
        ),
        strict=True,
    ):
        t, x = traces.series(sources[name])
        fs = 1.0 / float(np.median(np.diff(t)))
        f, tt, S = signal.spectrogram(
            x - x.mean(), fs=fs, nperseg=int(20 * fs), noverlap=int(18 * fs)
        )
        keep = (f >= band[0]) & (f <= band[1])
        ax.pcolormesh(
            tt + t[0], f[keep] * 60, np.log10(S[keep] + 1e-20), shading="auto", cmap="magma"
        )
        rt, rf = diag[key]
        ax.plot(rt, rf * 60, color="cyan", lw=1.2)
        ax.set_ylabel("per minute")
        ax.set_title(f"{name} from {sources[name]} (cyan: tracked rate)", fontsize=10)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    device = setup_device(args.device)
    verb = args.verb

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.processing import physio

    st, tr = _slice_times_and_tr(args)
    mag = np.asarray(load_nifti(args.input).dataobj, dtype=np.float32)
    if mag.ndim != 4:
        raise SystemExit("ffs_physio: -input must be 4-D")
    if mag.shape[2] != st.size:
        raise SystemExit(f"ffs_physio: {st.size} slice times but {mag.shape[2]} slices")
    ph = None
    if args.phase is not None:
        ph = np.asarray(load_nifti(args.phase).dataobj, dtype=np.float32)
        if ph.shape != mag.shape:
            raise SystemExit(f"ffs_physio: -phase shape {ph.shape} != -input {mag.shape}")
    ndrop = (
        args.drop_last
        if args.drop_last is not None
        else physio.trailing_noise_volumes(mag.reshape(-1, mag.shape[-1]).mean(0))
    )
    nt = mag.shape[-1] - ndrop
    mag = mag[..., :nt]
    ph = ph[..., :nt] if ph is not None else None
    if verb >= 1:
        _, gt = physio.excitation_groups(st)
        print(
            f"ffs_physio: {nt} volumes (dropped {ndrop} at the end), {st.size} slices, "
            f"{gt.size} excitations per TR = {gt.size / tr:.1f} Hz, phase {'yes' if ph is not None else 'no'}"
        )

    traces = physio.excitation_traces(mag, st, tr, phase=ph, device=device)
    del mag, ph
    card_auto, card_scores = physio.choose_source(traces, physio.CARDIAC_BAND, 20.0)
    resp_auto, resp_scores = physio.choose_source(traces, physio.RESP_BAND, 40.0)
    sources = {
        "cardiac": card_auto if args.card_from == "auto" else args.card_from,
        "respiration": resp_auto if args.resp_from == "auto" else args.resp_from,
    }
    if "phase" in sources.values() and traces.phase is None:
        raise SystemExit("ffs_physio: phase was asked for but -phase was not given")
    X, labels, diag = physio.slice_regressors(traces, sources["cardiac"], sources["respiration"])

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base = str(prefix)
    _write_slibase(Path(base + ".slibase.1D"), X, labels)
    t_m, x_m = traces.series("magnitude")
    cols = {"time_s": t_m, "magnitude": x_m}
    if traces.phase is not None:
        cols["phase_rad"] = traces.series("phase")[1]
    np.savetxt(base + ".traces.tsv", np.column_stack(list(cols.values())), delimiter="\t",
               header="\t".join(cols), comments="", fmt="%.6f")  # fmt: skip
    ct, cf = diag["cardiac_rate_hz"]
    rt, rf = diag["respiration_rate_hz"]
    with open(base + ".rates.tsv", "w") as fh:
        fh.write("band\ttime_s\trate_per_min\n")
        for name, tt, ff in (("cardiac", ct, cf), ("respiration", rt, rf)):
            for a, b in zip(tt, ff, strict=True):
                fh.write(f"{name}\t{a:.2f}\t{b * 60:.2f}\n")
    summary = {
        "n_volumes": int(nt),
        "dropped_last": int(ndrop),
        "tr": tr,
        "excitations_per_tr": int(traces.times.shape[1]),
        "sources": sources,
        "peak_prominence": {"cardiac": card_scores, "respiration": resp_scores},
        "heart_rate_per_min_median": float(np.median(cf) * 60),
        "breathing_rate_per_min_median": float(np.median(rf) * 60),
        "columns_per_slice": 13,
        "layout": "slice-major (ffs_reml -slibase_sm, afni_proc -ricor_regs)",
    }
    Path(base + ".json").write_text(json.dumps(summary, indent=2))
    if not args.no_qc:
        _qc_figure(Path(base + ".qc.png"), traces, sources, diag)
    if verb >= 1:
        print(
            f"  heart rate {summary['heart_rate_per_min_median']:.1f}/min from {sources['cardiac']}, "
            f"breathing {summary['breathing_rate_per_min_median']:.1f}/min from {sources['respiration']}"
        )
        print(f"  wrote {base}.slibase.1D ({X.shape[0]} x {X.shape[1]}) + traces, rates, json"
              f"{'' if args.no_qc else ', qc.png'}  [{time.time() - t0:.1f}s]")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
