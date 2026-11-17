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
        help="Also write the magnitude with each voxel's physiological fit removed "
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
        stem = str(args.input)
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

    rec = QCRecord.for_args(args, "physio", args.input)
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
    rec.info["input"] = str(args.input)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    device = setup_device(args.device)
    verb = args.verb

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.processing import physio

    st, tr = _slice_times_and_tr(args)
    mag_img = load_nifti(args.input)
    mag = _read_series(mag_img)
    zooms = tuple(float(z) for z in mag_img.header.get_zooms()[:3])
    if mag.ndim != 4:
        raise SystemExit("ffs_physio: -input must be 4-D")
    if mag.shape[2] != st.size:
        raise SystemExit(f"ffs_physio: {st.size} slice times but {mag.shape[2]} slices")
    ph = None
    if args.phase is not None:
        ph = _read_series(load_nifti(args.phase), affine_invariant=True)
        if ph.shape[:3] != mag.shape[:3] or ph.shape[3] < mag.shape[3]:
            raise SystemExit(
                f"ffs_physio: -phase shape {ph.shape} does not cover -input {mag.shape}"
            )
        if ph.shape[3] > mag.shape[3]:
            # A NORDIC magnitude has its noise volumes trimmed; the raw phase still
            # carries them at the end.
            if verb >= 1:
                print(
                    f"  phase has {ph.shape[3] - mag.shape[3]} more volumes than -input: trimmed from its end"
                )
            ph = ph[..., : mag.shape[3]]
    ndrop = (
        args.drop_last
        if args.drop_last is not None
        # A mean over axes, not over a reshape: reshaping the Fortran-ordered series
        # copies all of it (5 s on a 2 GB run).
        else physio.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
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

    traces = physio.excitation_traces(mag, st, tr, phase=ph, device=device, voxel_size=zooms)
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
            mag, traces, st, tr, sources["respiration"], split_axis=_split_axis(args),
            device=device,
        )  # fmt: skip
        traces.cardiac = refined.trace
        if verb >= 1:
            for lab, rate, align, null in refined.tried:
                shuf = "" if np.isnan(null) else f" (shuffled slice times: best {null:.2f})"
                print(f"    start {lab}: {rate:.1f}/min, slice alignment {align:.2f}{shuf}")
            verdict = (
                "FOUND" if refined.found else ("NOT FOUND" if refined.testable else "UNTESTABLE")
            )
            print(
                f"  cardiac from {refined.n_voxels} voxels' best 1 %: {verdict} via {refined.start}, "
                f"{refined.heart_rate_per_min:.1f}/min (slice alignment {refined.alignment:.2f} must beat "
                f"every slice-time shuffle, best {refined.alignment_null:.2f}; halves {refined.split_half:.2f})"
            )
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
            "testable": bool(refined.testable),
            "rule": (
                f"found = slice alignment beats all {physio.SLICE_PERMUTATIONS} slice-time "
                f"shuffles of each of up to {physio.PERMUTED_ANSWERS} distinct answers"
            ),
            "tried": [
                dict(
                    zip(
                        ("start", "heart_rate_per_min", "slice_alignment", "shuffled_max"),
                        t,
                        strict=True,
                    )
                )
                for t in refined.tried
            ],
            "weight_map": wpath,
            "in_nuisance": bool(cardiac_in_nuisance),
        }
    Path(base + ".json").write_text(json.dumps(summary, indent=2))
    if not args.no_qc:
        _qc_figure(Path(base + ".qc.png"), traces, sources, diag)
    _write_qc(args, mag, X, labels, traces, sources, diag, mag_img.affine, tr, device, refined)
    if args.tzero is not None:
        times = np.arange(nt) * tr + args.tzero
        if args.nuisance_set == "tapas":
            model = physio.model_from_traces(traces, sources["cardiac"], sources["respiration"])
            no_card = (
                {} if cardiac_in_nuisance else {"card_order": 0, "inter_order": 0, "hrv": False}
            )
            Xv, vlabels = physio.standard_regressors(model, times, **no_card)
        else:
            Xv, vlabels = physio.volume_regressors(diag, times, cardiac=cardiac_in_nuisance)
        with open(base + ".nuisance.1D", "w") as fh:
            fh.write(f"# ffs_physio volume regressors at f*TR + {args.tzero:g} s\n")
            fh.write('# ColumnLabels = "' + " ; ".join(vlabels) + '"\n')
            np.savetxt(fh, Xv, fmt="%.6f")
        summary["nuisance"] = {
            "file": base + ".nuisance.1D",
            "tzero": args.tzero,
            "set": args.nuisance_set,
            "columns": vlabels,
        }
        Path(base + ".json").write_text(json.dumps(summary, indent=2))
    if args.clean is not None:
        import torch

        from fastfuncstuff.processing.io import save_image

        cleaned, removed = physio.regress_out(mag, X, tr, device=device)
        img = load_nifti(args.input)  # header only; the data stays unread
        hdr = {"affine": img.affine.copy(), "header": img.header.copy()}
        save_image(torch.from_numpy(cleaned.transpose(3, 2, 1, 0)), args.clean, header_info=hdr)
        stem = str(args.clean)
        for ext in (".nii.gz", ".nii.zst", ".nii"):
            if stem.endswith(ext):
                rm_path = stem[: -len(ext)] + "_physio_removed" + ext
                break
        else:
            rm_path = stem + "_physio_removed.nii.gz"
        save_image(torch.from_numpy(removed.transpose(2, 1, 0)), rm_path, header_info=hdr)
        summary["clean"] = {"output": stem, "removed_map": rm_path}
        Path(base + ".json").write_text(json.dumps(summary, indent=2))
        if verb >= 1:
            print(f"  cleaned magnitude -> {stem} (variance-removed map {rm_path})")
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
