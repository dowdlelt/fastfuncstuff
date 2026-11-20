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
            "EXPERIMENTAL. Cardiac and respiratory noise regressors estimated from the data\n"
            "itself. Cardiac detection is judged by slice timing and reports 'not found'\n"
            "rather than guess, but it is not yet trusted on all data: it finds few runs at\n"
            "TR >= 2 s (see docs/physio.md, Limits). Check the QC before relying on it.\n\n"
            "Each multiband excitation is one physiological sample, so a run read in\n"
            "acquisition order is a trace at n_excitations/TR (17 Hz for MB5, 85 slices,\n"
            "TR 1 s) -- fast enough for the heartbeat that the volume rate aliases.\n"
            "Cardiac is read from magnitude and respiration from phase (when given).\n\n"
            "Run it on RAW slices: before motion correction or slice timing, which move\n"
            "samples away from their acquisition times. Output regressors are per slice,\n"
            "evaluated at each slice's own acquisition time, in the same 13-per-slice\n"
            "slice-major layout as AFNI physio_calc.py (ffs_reml -slibase_sm, or\n"
            "afni_proc -ricor_regs).\n\n"
            "Multi-echo: give every echo to -input (and -phase), shortest TE first. Each\n"
            "echo is another reading of the same excitation: the cardiac detector draws its\n"
            "vessel voxels from all echoes (on ds006926 it found every run any single echo\n"
            "did); respiration is read from the middle echo's phase, which a fit across\n"
            "echoes did not beat.\n\n"
            "3D EPI (BETA; sidecar MRAcquisitionType 3D, or -acquisition 3d): needs -phase.\n"
            "Each partition cell (Rz kz lines, shot together) is one sample of the field:\n"
            "the complex image is transformed along the partition axis and each line's\n"
            "phase change read against the run mean (with 2+ echoes, its TE slope). That\n"
            "trace gives breathing and, at short TR, the heartbeat; cardiac counts as found\n"
            "only if it lines up under the true cell times better than under every shuffle\n"
            "of them. Regressors are evaluated at the k-space centre's acquisition time.\n"
            "Not yet checked against recordings (docs/physio.md, 3D EPI)."
        ),
    )
    parser.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Raw magnitude 4-D series (.nii/.nii.gz/.nii.zst); multi-echo: one per echo, "
        "shortest TE first. Slice timing and TR come from the first one's sidecar.",
    )
    parser.add_argument(
        "-phase",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Raw phase 4-D series matching -input, one per echo (scanner units or "
        "radians; no unwrap needed -- only its change over time is used). Without it "
        "respiration is read from magnitude, which was unreliable on task runs.",
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
        help="Repetition time (s). Default: sidecar JSON, then header. 3D: the VOLUME TR "
        "(some 3D sidecars store the per-shot TR; below 0.3 s it is refused).",
    )
    parser.add_argument(
        "-acquisition",
        choices=["auto", "2d", "3d"],
        default="auto",
        help="auto = the -input sidecar's MRAcquisitionType (2d when absent).",
    )
    parser.add_argument(
        "-partition_accel",
        type=int,
        default=None,
        metavar="RZ",
        help="3D: partition lines acquired together (the CAIPI pattern's Rz). Default: the "
        "sidecar's ParallelReductionFactorOutOfPlane, else 1.",
    )
    parser.add_argument(
        "-echo_times",
        nargs="+",
        type=float,
        default=None,
        metavar="MS",
        help="3D with 2+ echoes: echo times (ms), for each partition line's TE slope. "
        "Default: the sidecars' EchoTime.",
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
        choices=["auto", "weighted", "magnitude", "phase"],
        default="auto",
        help="Trace to read cardiac from. weighted (= auto): rebuilt from the voxels "
        "that carry the heartbeat (the vessels, ~1 %% of the brain), starting from the "
        "magnitude trace -- against a pulse oximeter on TASKFORCE, phase locking "
        "0.95-0.99 vs 0.88-0.91 for magnitude; ~5 s per run. It also tests whether a "
        "heartbeat was found (two halves of the brain must agree) and writes "
        "PREFIX.cardiac_weight (where it was found). magnitude: the mean over the brain.",
    )
    parser.add_argument(
        "-card_unfound",
        choices=["drop", "keep"],
        default="drop",
        help="When -card_from weighted finds no heartbeat: drop = leave the cardiac "
        "columns out of PREFIX.nuisance.1D (they would fit noise and cost the GLM "
        "their degrees of freedom); keep = write them from the magnitude trace anyway. "
        "The slibase keeps its 13-per-slice layout either way.",
    )
    parser.add_argument(
        "-resp_from",
        choices=["auto", "magnitude", "phase"],
        default="auto",
        help="Trace to read respiration from. auto = phase when given, else magnitude.",
    )
    parser.add_argument(
        "-tzero",
        type=float,
        default=None,
        help="Also write PREFIX.nuisance.1D: the -nuisance_set evaluated at f*TR + TZERO, "
        "the instant every voxel stands for after a slice-timing-corrected resample. Give "
        "the -tzero the final resample used. Use it as a per-run GLM nuisance (like "
        "motion) on the RESAMPLED data, fitted jointly with the task -- PhysIO's "
        "placement. Cardiac is included: slice timing shifts the aliased heartbeat by a "
        "constant phase per slice, which each voxel's cos/sin pair absorbs.",
    )
    parser.add_argument(
        "-nuisance_set",
        choices=["afni", "tapas"],
        default="afni",
        help="Columns of PREFIX.nuisance.1D. afni (13): RVT at 0-4 s lags, cardiac and "
        "respiratory RETROICOR to 2nd order -- the slibase columns, as physio_calc.py "
        "writes them. tapas (20): PhysIO's default -- cardiac 3rd, respiratory 4th, "
        "cardiac x respiratory 1st order (Harvey 2008), HRV*CRF (Chang 2009), RVT*RRF "
        "(Birn 2008). On TASKFORCE the interaction and HRV terms were significant in "
        "1-2 %% of voxels (p < 0.001), against ~15 %% for cardiac.",
    )
    parser.add_argument(
        "-clean",
        default=None,
        help="Also write the magnitude (multi-echo: each echo, as OUT.echo-N) with each voxel's physiological fit removed "
        "(its own slice's regressors, fitted jointly with drift; mean and drift kept), "
        "as afni_proc's ricor block does. CAUTION: on ImaginationStimulation this, done "
        "before motion correction and slice timing, LOWERED held-out task R2 in all four "
        "subjects (cause not established; breathing there is task-locked, and cleaning "
        "first gives shared variance to the noise). -tzero and a GLM nuisance after the "
        "resample did not. "
        "Writes OUT and OUT's _physio_removed map (fraction of each voxel's variance "
        "taken). Costs 13 degrees of freedom per voxel that a later GLM does not see.",
    )
    parser.add_argument("-no_qc", action="store_true", help="Skip the QC spectrogram PNG.")
    parser.add_argument(
        "-kz_plot",
        action="store_true",
        help="3D: also write PREFIX.kz.png, the evidence that the partition axis carries "
        "timing: the strongest volume-rate rhythm read per kz line and echo (spectrum, "
        "|A(kz)|, angle A(kz) across kz, A(kz) in the complex plane, line phase over "
        "volumes, each echo against the last). A timed field advances in angle across "
        "kz, the same on every echo, with |A| scaling with TE. Its numbers go in the json "
        "under partition.kz.",
    )
    from fastfuncstuff.qc.record import add_qc_args

    add_qc_args(parser)
    add_device_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args(argv)


def _split_axis(args) -> int:
    """In-plane axis to split the brain along for the found-a-heartbeat test: the one
    NOT phase-encoded, since the N/2 ghost copies a vessel half a FOV along PE and
    would hand both halves the same voxel. Read from the sidecar; j (A/P) if unknown."""
    src = args.tpattern
    if src is None or not str(src).endswith(".json"):
        stem = str(args.input[0])
        for ext in (".nii.gz", ".nii.zst", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        src = stem + ".json"
    try:
        pe = str(json.loads(Path(src).read_text()).get("PhaseEncodingDirection", "j"))
    except (OSError, ValueError):
        pe = "j"
    return 1 if pe.startswith("i") else 0


def _echo_times(inputs: list[str]) -> list[float]:
    """Sidecar echo times (ms) of a multi-echo -input, checked shortest first (the phase
    is read from the middle echo by position); [] for one echo or no sidecars."""
    if len(inputs) == 1:
        return []
    from fastfuncstuff.processing.b0fmap import read_echo_times

    tes = read_echo_times(inputs) or []
    if any(b <= a for a, b in zip(tes, tes[1:], strict=False)):
        raise SystemExit(
            f"ffs_physio: -input echo times {tes} ms must increase (shortest TE first)"
        )
    return [float(t) for t in tes]


def _slice_times_and_tr(args) -> tuple[np.ndarray, float]:
    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.processing.slicetime import load_slice_timing

    src = args.tpattern
    if src is None:
        stem = str(args.input[0])
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
        tr = get_tr_from_file(args.input[0])
    if not tr or tr <= 0:
        raise SystemExit("ffs_physio: no TR in the sidecar or header; pass -TR.")
    return st, float(tr)


def _read_series(img, affine_invariant: bool = False) -> np.ndarray:
    """The 4-D data at its stored dtype where that is exact, else float32.

    Converting a whole run to float32 went through float64 inside nibabel and peaked
    at ~5 copies of the series; the library converts one slice at a time instead.
    ``affine_invariant``: the consumer only uses the data through a positive-slope
    linear map of its own range (the phase, mapped onto [-pi, pi] by its min and max),
    so the stored integers can stand in for the scaled values."""
    proxy = img.dataobj
    slope = float(getattr(proxy, "slope", 1.0))
    inter = float(getattr(proxy, "inter", 0.0))
    unscaled = slope == 1.0 and inter == 0.0
    if hasattr(proxy, "get_unscaled") and (unscaled or (affine_invariant and slope > 0)):
        a = np.asarray(proxy.get_unscaled())
        if a.dtype.kind in "iub" or a.dtype == np.float32:
            return a
        return a.astype(np.float32)
    return np.asarray(proxy, dtype=np.float32)


def _torch_volume(vol: np.ndarray):
    """A (nx, ny, nz) array in save_image's (z, y, x) tensor order."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 1, 0)))


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
    """The standalone PREFIX.qc.png: the same rates figure -qc puts in its folder."""
    from fastfuncstuff.qc import checks, render

    series = {k: traces.series(sources[k], uniform=True) for k in ("cardiac", "respiration")}
    rates = {"cardiac": diag["cardiac_rate_hz"], "respiration": diag["respiration_rate_hz"]}
    render.save_png(checks.physio_rates_figure(series, rates, sources), path)


def _write_qc(
    args, mag, X, labels, traces, sources, diag, affine, tr, device, refined=None
) -> None:
    """-qc: tracked rates and traces, and where each group explains variance on the
    raw slices (the maps cost one batched fit per slice; only made under -qc)."""
    from fastfuncstuff.processing import physio
    from fastfuncstuff.qc import checks
    from fastfuncstuff.qc.record import QCRecord

    rec = QCRecord.for_args(args, "physio", args.input[0])
    if rec is None:
        return
    m = X.shape[1] // mag.shape[2]
    one = [lab.split(".", 1)[1] for lab in labels[:m]]
    groups = {
        name: [i for i, lab in enumerate(one) if lab.startswith(prefix)]
        for name, prefix in (("cardiac", "card."), ("respiration", "resp."), ("RVT", "rvt"))
    }
    mean_img = mag.mean(-1)
    mask = mean_img > 0.25 * np.percentile(mean_img[mean_img > 0], 98)
    maps = physio.explained_maps(mag, X, tr, groups, device=device, mask=mask)
    series = {
        "cardiac": traces.series(sources["cardiac"], uniform=True),
        "respiration": traces.series(sources["respiration"], uniform=True),
    }
    rates = {"cardiac": diag["cardiac_rate_hz"], "respiration": diag["respiration_rate_hz"]}
    rec.info["input"] = " ".join(map(str, args.input))
    if len(args.input) > 1:
        rec.info["maps"] = "variance explained on echo 1"
    rec.info["sources"] = f"cardiac {sources['cardiac']}, respiration {sources['respiration']}"
    cardiac = (
        None
        if refined is None
        else {
            "weight": refined.weight,
            "found": refined.found,
            "split_half": refined.split_half,
            "alignment": refined.alignment,
            "alignment_null": refined.alignment_null,
        }
    )
    checks.physio_check(rec, 1, series, rates, sources, maps, mean_img, affine, mask, cardiac)
    rec.write(args.verb)


def _write_rates(base: str, diag: dict) -> tuple[np.ndarray, np.ndarray]:
    """PREFIX.rates.tsv; returns the cardiac and respiratory rates (Hz)."""
    ct, cf = diag["cardiac_rate_hz"]
    rt, rf = diag["respiration_rate_hz"]
    with open(base + ".rates.tsv", "w") as fh:
        fh.write("band\ttime_s\trate_per_min\n")
        for name, tt, ff in (("cardiac", ct, cf), ("respiration", rt, rf)):
            for a, b in zip(tt, ff, strict=True):
                fh.write(f"{name}\t{a:.2f}\t{b * 60:.2f}\n")
    return cf, rf


def _write_nuisance(args, base, summary, traces, sources, diag, nt, tr, tzero,
                    cardiac_in_nuisance) -> None:  # fmt: skip
    """PREFIX.nuisance.1D: the -nuisance_set at f*TR + tzero."""
    from fastfuncstuff.processing import physio

    times = np.arange(nt) * tr + tzero
    if args.nuisance_set == "tapas":
        model = physio.model_from_traces(traces, sources["cardiac"], sources["respiration"])
        no_card = {} if cardiac_in_nuisance else {"card_order": 0, "inter_order": 0, "hrv": False}
        Xv, vlabels = physio.standard_regressors(model, times, **no_card)
    else:
        Xv, vlabels = physio.volume_regressors(diag, times, cardiac=cardiac_in_nuisance)
    with open(base + ".nuisance.1D", "w") as fh:
        fh.write(f"# ffs_physio volume regressors at f*TR + {tzero:g} s\n")
        fh.write('# ColumnLabels = "' + " ; ".join(vlabels) + '"\n')
        np.savetxt(fh, Xv, fmt="%.6f")
    summary["nuisance"] = {"file": base + ".nuisance.1D", "tzero": tzero,
                           "set": args.nuisance_set, "columns": vlabels}  # fmt: skip
    Path(base + ".json").write_text(json.dumps(summary, indent=2))


def _write_clean(args, base, summary, parts, hdr, verb) -> None:
    """-clean outputs: each echo's cleaned series (OUT, or OUT.echo-N) and its
    _physio_removed variance map."""
    import torch

    from fastfuncstuff.cli_utils import entity_path
    from fastfuncstuff.processing.io import save_image

    outs = []
    for e, (c_e, r_e) in enumerate(parts):
        stem = str(args.clean) if len(parts) == 1 else entity_path(args.clean, echo=e + 1)
        for ext in (".nii.gz", ".nii.zst", ".nii"):
            if stem.endswith(ext):
                rm_path = stem[: -len(ext)] + "_physio_removed" + ext
                break
        else:
            rm_path = stem + "_physio_removed.nii.gz"
        r_e = r_e if r_e.ndim == 3 else r_e[..., 0]
        save_image(torch.from_numpy(np.ascontiguousarray(c_e.transpose(3, 2, 1, 0))), stem,
                   header_info=hdr)  # fmt: skip
        save_image(torch.from_numpy(np.ascontiguousarray(r_e.transpose(2, 1, 0))), rm_path,
                   header_info=hdr)  # fmt: skip
        outs.append({"output": stem, "removed_map": rm_path})
        if verb >= 1:
            print(f"  cleaned magnitude -> {stem} (variance-removed map {rm_path})")
    summary["clean"] = outs[0] if len(parts) == 1 else outs
    Path(base + ".json").write_text(json.dumps(summary, indent=2))


def _sidecar(path: str) -> dict:
    """The BIDS sidecar beside a NIfTI path, or {}."""
    stem = str(path)
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    try:
        return json.loads(Path(stem + ".json").read_text())
    except (OSError, ValueError):
        return {}


def _acquisition(args) -> str:
    if args.acquisition != "auto":
        return args.acquisition
    kind = str(_sidecar(args.input[0]).get("MRAcquisitionType", "")).upper()
    return "3d" if kind == "3D" else "2d"


def _main_3d(args, device, t0: float) -> int:
    """3D EPI (BETA): the field per partition cell from phase (processing/physio_3d)."""
    from fastfuncstuff.io.afni import get_tr_from_file, load_nifti
    from fastfuncstuff.processing import physio, physio_3d

    verb = args.verb
    inputs = [str(p) for p in args.input]
    ne = len(inputs)
    if args.phase is None:
        raise SystemExit("ffs_physio: 3D EPI needs -phase (the field is read from it)")
    if len(args.phase) != ne:
        raise SystemExit(f"ffs_physio: {len(args.phase)} -phase files for {ne} -input echoes")
    side = _sidecar(inputs[0])
    tr = args.TR or side.get("RepetitionTime") or get_tr_from_file(inputs[0])
    if not tr or tr < 0.3:
        raise SystemExit(
            f"ffs_physio: TR {tr} s looks like the per-shot TR some 3D sidecars store; "
            "pass the volume TR with -TR"
        )
    tr = float(tr)
    if args.echo_times is not None:
        tes_ms = [float(t) for t in args.echo_times]
        if len(tes_ms) != ne or any(b <= a for a, b in zip(tes_ms, tes_ms[1:], strict=False)):
            raise SystemExit(f"ffs_physio: -echo_times {tes_ms} must give {ne} increasing values")
    elif ne > 1:
        tes_ms = _echo_times(inputs)
    else:
        tes_ms = [float(side["EchoTime"]) * 1e3] if "EchoTime" in side else []
    if ne > 1 and len(tes_ms) != ne:
        raise SystemExit("ffs_physio: no EchoTime in the -input sidecars; pass -echo_times (ms)")
    rz = int(args.partition_accel or side.get("ParallelReductionFactorOutOfPlane") or 1)
    mag_img = load_nifti(inputs[0])
    mags = [_read_series(mag_img)] + [_read_series(load_nifti(p)) for p in inputs[1:]]
    phs = [_read_series(load_nifti(p), affine_invariant=True) for p in args.phase]
    shape = mags[0].shape
    if len(shape) != 4:
        raise SystemExit("ffs_physio: -input must be 4-D")
    for name, arr in [*zip(inputs, mags, strict=True), *zip(args.phase, phs, strict=True)]:
        if arr.shape[:3] != shape[:3] or arr.shape[3] < shape[3]:
            raise SystemExit(f"ffs_physio: {name} shape {arr.shape} does not match {shape}")
    ndrop = (
        args.drop_last
        if args.drop_last is not None
        else physio.trailing_noise_volumes(mags[0].mean(axis=(0, 1, 2)))
    )
    nt = shape[3] - ndrop
    mags = [m[..., :nt] for m in mags]
    phs = [p[..., :nt] for p in phs]
    pt = physio_3d.partition_field_traces(
        mags, phs, tr, rz, [t / 1e3 for t in tes_ms] or None, device=device,
        split_axis=_split_axis(args),
    )  # fmt: skip
    del phs
    if verb >= 1:
        tes = f", TE {', '.join(f'{t:g}' for t in tes_ms)} ms" if tes_ms else ""
        print(
            f"ffs_physio (3D, BETA): {nt} volumes (dropped {ndrop} at the end), {shape[2]} "
            f"partitions in {pt.times.shape[1]} cells of {rz} = {pt.cell_rate:.1f} Hz, {ne} "
            f"echo(es){tes}; fill {pt.order} (boundary/within step {pt.boundary_ratio:.2f}); "
            f"k-space centre at {pt.centre_offset:.2f} s into the TR"
        )
        if not pt.order_agrees:
            print(
                "  WARNING: the fill-order second opinion (spectral line) disagrees -- check "
                "the breathing rate against the volume Nyquist rate"
            )
    traces = physio_3d.excitation_view(pt, shape[2])
    card = physio_3d.cardiac(pt, tr)
    traces.cardiac, traces.cardiac_phase = card.trace, card.phase
    sources = {"cardiac": "weighted", "respiration": "phase"}
    if not card.testable:
        # no open cardiac band: the slibase keeps its layout from the plain tracker on
        # the breathing-removed trace (k/TR avoided only); out of the nuisance by default
        traces.magnitude, traces.cardiac_phase = card.trace, None
        sources["cardiac"] = "magnitude"
    cardiac_in_nuisance = card.found or args.card_unfound == "keep"
    if verb >= 1:
        if not card.testable:
            print(
                "  cardiac UNTESTABLE: the 1/TR lines and their breathing sidebands cover the band"
            )
        else:
            print(
                f"  cardiac {card.rate_per_min:.1f}/min: {'FOUND' if card.found else 'NOT FOUND'} "
                f"(peak prominence {card.prominence:.1f} at the true cell times must beat all "
                f"{card.n_shuffles} shuffles, best {card.shuffled_max:.1f})"
            )
    X, labels, diag = physio.slice_regressors(traces, sources["cardiac"], sources["respiration"])
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base = str(prefix)
    _write_slibase(Path(base + ".slibase.1D"), X, labels)
    t_f, x_f = traces.series("phase")
    _, x_c = traces.series("weighted")
    np.savetxt(base + ".traces.tsv", np.column_stack([t_f, x_f, x_c]), delimiter="\t",
               header="time_s\tfield_hz\tfield_minus_breathing_hz", comments="", fmt="%.6f")  # fmt: skip
    cf, rf = _write_rates(base, diag)
    # Field change across each volume's sweep, as the TE-scaled partition shift it implies
    # (voxels, sign in the file's partition index): what ffs_moco -me_3depi measures from
    # the magnitudes, here from phase alone.
    shift = (
        -np.outer(pt.sweep_change, np.asarray(tes_ms) / 1e3) if tes_ms else pt.sweep_change[:, None]
    )
    np.savetxt(base + ".partition_shift.1D", shift, fmt="%.5f",
               header="partition shift per volume (voxels) per echo, -TE x field change across the sweep")  # fmt: skip
    summary = {
        "acquisition": "3d",
        "beta": True,
        "n_volumes": int(nt),
        "dropped_last": int(ndrop),
        "tr": tr,
        "sources": sources,
        "heart_rate_per_min_median": float(np.median(cf) * 60),
        "breathing_rate_per_min_median": float(np.median(rf) * 60),
        "columns_per_slice": 13,
        "echoes": ne,
        "echo_times_ms": tes_ms or None,
        "layout": "slice-major (ffs_reml -slibase_sm, afni_proc -ricor_regs); every slice "
        "evaluated at the k-space centre's acquisition time",
        "partition": {
            "rows_per_cell": rz,
            "cells_per_volume": int(pt.times.shape[1]),
            "cell_rate_hz": pt.cell_rate,
            "fill_order": pt.order,
            "fill_order_second_opinion_agrees": bool(pt.order_agrees),
            "boundary_step_ratio": pt.boundary_ratio,
            "kspace_centre_offset_s": pt.centre_offset,
            "shift_file": base + ".partition_shift.1D",
        },
        "cardiac_3d": {
            "found": bool(card.found),
            "testable": bool(card.testable),
            "rate_per_min": card.rate_per_min,
            "prominence": card.prominence,
            "shuffled_max": card.shuffled_max,
            "rule": f"found = cardiac-band peak prominence of the breathing-removed cell trace "
            f"at the true cell times beats all {card.n_shuffles} derangements of the cell times "
            f"(search repeated for each)",
            "in_nuisance": bool(cardiac_in_nuisance),
        },
    }
    if args.kz_plot:
        from fastfuncstuff.qc import checks, render

        kz = physio_3d.kz_diagnostic(pt, tr)
        summary["partition"]["kz"] = {
            "file": base + ".kz.png",
            "peak_hz": kz.alias_hz,
            "peak_over_median": kz.peak_over_median,
            "phase_span_rad": kz.phase_span_rad,
            "nonlinearity": kz.nonlinearity,
            "echo_agreement": kz.echo_agreement,
            "amp_ratio_vs_last_echo": kz.amp_ratio,
            "te_ratio_vs_last_echo": [t / tes_ms[-1] for t in tes_ms] if tes_ms else None,
        }
        render.save_png(checks.physio_kz_figure(kz, Path(inputs[0]).name), base + ".kz.png")
    Path(base + ".json").write_text(json.dumps(summary, indent=2))
    if not args.no_qc:
        _qc_figure(Path(base + ".qc.png"), traces, sources, diag)
    _write_qc(args, mags[0], X, labels, traces, sources, diag, mag_img.affine, tr, device)
    _write_nuisance(args, base, summary, traces, sources, diag, nt, tr,
                    pt.centre_offset if args.tzero is None else args.tzero, cardiac_in_nuisance)  # fmt: skip
    if args.clean is not None:
        hdr = {"affine": mag_img.affine.copy(), "header": mag_img.header.copy()}
        parts = [physio.regress_out(m, X, tr, device=device) for m in mags]
        _write_clean(args, base, summary, parts, hdr, verb)
    if verb >= 1:
        print(
            f"  heart rate {summary['heart_rate_per_min_median']:.1f}/min"
            f"{'' if card.found else ' (NOT FOUND: left out of the nuisance)' if not cardiac_in_nuisance else ' (not found; kept)'}, "
            f"breathing {summary['breathing_rate_per_min_median']:.1f}/min from the partition field trace"
        )
        print(f"  wrote {base}.slibase.1D + nuisance, traces, rates, partition_shift, json"
              f"{'' if args.no_qc else ', qc.png'}{', kz.png' if args.kz_plot else ''}"
              f"  [{time.time() - t0:.1f}s]")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    device = setup_device(args.device)
    verb = args.verb

    if _acquisition(args) == "3d":
        return _main_3d(args, device, t0)
    if args.kz_plot:
        print("ffs_physio: -kz_plot applies to 3D EPI only; ignored")

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.processing import physio

    st, tr = _slice_times_and_tr(args)
    inputs = [str(p) for p in args.input]
    ne = len(inputs)
    if args.phase is not None and len(args.phase) != ne:
        raise SystemExit(f"ffs_physio: {len(args.phase)} -phase files for {ne} -input echoes")
    echo_times = _echo_times(inputs)
    split_axis = _split_axis(args)
    mag_img = load_nifti(inputs[0])
    zooms = tuple(float(z) for z in mag_img.header.get_zooms()[:3])
    echo_axis = None
    if ne == 1:
        mag = _read_series(mag_img)
    else:
        # one array, echoes tiled in-plane: the cardiac detector's voxels are every echo's
        mag, echo_axis = physio.stack_echoes(
            ne, lambda e: _read_series(mag_img if e == 0 else load_nifti(inputs[e])), split_axis
        )
    if mag.ndim != 4:
        raise SystemExit("ffs_physio: -input must be 4-D")
    if mag.shape[2] != st.size:
        raise SystemExit(f"ffs_physio: {st.size} slice times but {mag.shape[2]} slices")
    echo_shape = mag_img.shape[:3]
    ph = None
    if args.phase is not None:
        ph = []
        for path in args.phase:
            p_e = _read_series(load_nifti(path), affine_invariant=True)
            if p_e.shape[:3] != echo_shape or p_e.shape[3] < mag.shape[3]:
                raise SystemExit(
                    f"ffs_physio: -phase {path} shape {p_e.shape} does not cover the "
                    f"-input echoes {(*echo_shape, mag.shape[3])}"
                )
            if p_e.shape[3] > mag.shape[3]:
                # A NORDIC magnitude has its noise volumes trimmed; the raw phase still
                # carries them at the end.
                if verb >= 1:
                    print(
                        f"  phase has {p_e.shape[3] - mag.shape[3]} more volumes than -input: trimmed from its end"
                    )
                p_e = p_e[..., : mag.shape[3]]
            ph.append(p_e)
    ndrop = (
        args.drop_last
        if args.drop_last is not None
        # A mean over axes, not over a reshape: reshaping the Fortran-ordered series
        # copies all of it (5 s on a 2 GB run).
        else physio.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    )
    nt = mag.shape[-1] - ndrop
    mag = mag[..., :nt]
    ph = [p_e[..., :nt] for p_e in ph] if ph is not None else None
    mags = mag if echo_axis is None else physio.echo_views(mag, ne, echo_axis)
    if verb >= 1:
        _, gt = physio.excitation_groups(st)
        tes = f" (TE {', '.join(f'{t:g}' for t in echo_times)} ms)" if echo_times else ""
        echoes = "" if ne == 1 else f", {ne} echoes{tes}, phase from echo {ne // 2 + 1}"
        print(
            f"ffs_physio: {nt} volumes (dropped {ndrop} at the end), {st.size} slices, "
            f"{gt.size} excitations per TR = {gt.size / tr:.1f} Hz, phase {'yes' if ph is not None else 'no'}"
            f"{echoes}"
        )

    traces = physio.excitation_traces(
        mags, st, tr, phase=ph, device=device, voxel_size=zooms,
    )  # fmt: skip
    del ph
    _, card_scores = physio.choose_source(traces, physio.CARDIAC_BAND, 20.0)
    resp_auto, resp_scores = physio.choose_source(traces, physio.RESP_BAND, 40.0)
    sources = {
        "cardiac": "weighted" if args.card_from == "auto" else args.card_from,
        "respiration": resp_auto if args.resp_from == "auto" else args.resp_from,
    }
    if "phase" in sources.values() and traces.phase is None:
        raise SystemExit("ffs_physio: phase was asked for but -phase was not given")
    refined = None
    cardiac_in_nuisance = True
    if sources["cardiac"] == "weighted":
        refined = physio.refine_cardiac(
            mag, traces, st, tr, sources["respiration"], split_axis=split_axis,
            device=device, n_echoes=ne, echo_axis=echo_axis,
        )  # fmt: skip
        traces.cardiac = refined.trace
        traces.cardiac_phase = refined.phase
        if verb >= 1:
            for lab, rate, align, null, harm, hnull in refined.tried:
                shuf = (
                    ""
                    if np.isnan(null)
                    else f" (shuffled best {null:.2f}); harmonic {harm:.2f} (non-harmonic best {hnull:.2f})"
                )
                print(f"    start {lab}: {rate:.1f}/min, slice alignment {align:.2f}{shuf}")
            verdict = (
                "FOUND" if refined.found else ("NOT FOUND" if refined.testable else "UNTESTABLE")
            )
            stat = (
                "fundamental + harmonic alignment"
                if refined.method == "constant_modulus"
                else "slice alignment"
            )
            print(
                f"  cardiac from {refined.n_voxels} voxels' best 1 %: {verdict} via {refined.start}, "
                f"{refined.heart_rate_per_min:.1f}/min ({stat} {refined.alignment:.2f} must beat "
                f"every slice-time shuffle, best {refined.alignment_null:.2f}; harmonic "
                f"{refined.harmonic:.2f} must beat {refined.harmonic_null:.2f}; halves {refined.split_half:.2f})"
            )
            if refined.echo_share:
                share = ", ".join(
                    f"echo {e + 1} {100 * x:.0f}%" for e, x in enumerate(refined.echo_share)
                )
                print(f"  cardiac weight by echo: {share}")
        if not refined.found:
            # The slibase keeps its layout, built from the mean trace as before.
            sources["cardiac"] = "magnitude"
            cardiac_in_nuisance = args.card_unfound == "keep"
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
    cf, rf = _write_rates(base, diag)
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
        "echoes": ne,
        "echo_times_ms": echo_times or None,
        "phase_echo": ne // 2 + 1 if args.phase is not None else None,
        "layout": "slice-major (ffs_reml -slibase_sm, afni_proc -ricor_regs)",
    }
    if refined is not None:
        from fastfuncstuff.processing.io import save_image

        wpath = base + ".cardiac_weight.nii.gz"
        hdr = {"affine": mag_img.affine.copy(), "header": mag_img.header.copy()}
        save_image(_torch_volume(refined.weight), wpath, header_info=hdr)
        summary["cardiac_weighted"] = {
            "found": bool(refined.found),
            "split_half_phase_locking": refined.split_half,
            "heart_rate_per_min_median": refined.heart_rate_per_min,
            "start_heart_rate_per_min": refined.start_rate_per_min,
            "n_voxels": refined.n_voxels,
            "start": refined.start,
            "alias_coherence": refined.alias_coherence,
            "alias_ratio": refined.alias_ratio,
            "slice_alignment": refined.alignment,
            "slice_alignment_null": refined.alignment_null,
            "harmonic_alignment": refined.harmonic,
            "harmonic_alignment_null": refined.harmonic_null,
            "testable": bool(refined.testable),
            "slice_resolvability": refined.resolvability,
            "method": refined.method,
            "rule": (
                f"constant-modulus starts: found = slice alignment of the phase plus of its second "
                f"harmonic beats {physio.SLICE_PERMUTATIONS} slice-time derangements per start "
                f"tried (all drawn for the kept start), re-choosing among the same aliases, and "
                f"the harmonic beats its alignment at all {len(physio.HARMONIC_CONTROLS)} "
                f"non-harmonic multiples"
                if refined.method == "constant_modulus"
                else f"found = slice alignment beats all {physio.SLICE_PERMUTATIONS} slice-time "
                f"shuffles of each of up to {physio.PERMUTED_ANSWERS} distinct answers, and its "
                f"second harmonic beats its alignment at all {len(physio.HARMONIC_CONTROLS)} non-harmonic multiples"
            ),
            "tried": [
                dict(
                    zip(
                        (
                            "start",
                            "heart_rate_per_min",
                            "slice_alignment",
                            "shuffled_max",
                            "harmonic_alignment",
                            "harmonic_control_max",
                        ),
                        t,
                        strict=True,
                    )
                )
                for t in refined.tried
            ],
            "weight_map": wpath,
            "echo_share": list(refined.echo_share) or None,
            "in_nuisance": bool(cardiac_in_nuisance),
        }
    Path(base + ".json").write_text(json.dumps(summary, indent=2))
    if not args.no_qc:
        _qc_figure(Path(base + ".qc.png"), traces, sources, diag)
    mag1 = mags if echo_axis is None else mags[0]
    _write_qc(args, mag1, X, labels, traces, sources, diag, mag_img.affine, tr, device, refined)
    if args.tzero is not None:
        _write_nuisance(args, base, summary, traces, sources, diag, nt, tr, args.tzero,
                        cardiac_in_nuisance)  # fmt: skip
    if args.clean is not None:
        # every echo at once: the fit is per voxel, and an echo's voxels are voxels
        cleaned, removed = physio.regress_out(mag, X, tr, device=device)
        hdr = {"affine": mag_img.affine.copy(), "header": mag_img.header.copy()}
        if echo_axis is None:
            parts = [(cleaned, removed)]
        else:
            parts = list(
                zip(
                    physio.echo_views(cleaned, ne, echo_axis),
                    physio.echo_views(removed[..., None], ne, echo_axis),
                    strict=True,
                )
            )
        _write_clean(args, base, summary, parts, hdr, verb)
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
