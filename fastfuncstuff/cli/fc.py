"""ffs_fc — functional connectivity between ROIs, seeds and voxels.

3dNetCorr's job, with the cleaning done in the same call. Each ROI's mean time
course (any label map, mask stack or surface atlas -- thousands of parcels is fine)
is cleaned per run on the uncensored TRs, then correlated:

  * ROI x ROI matrices: Pearson r (CC) and Fisher z (FZ) always; partial
    correlation (PC, PCB) with -part_corr; a Ledoit-Wolf shrunk correlation (LW,
    and PC_LW) with -shrink lw, which stays invertible with more ROIs than TRs.
  * per-TR weighted correlation (-weight_corr): one matrix set per weight column.
    A condition's block or HRF regressor as the weight is a task-weighted FC.
  * whole-brain maps (-wb_corr / -wb_z): every voxel against every ROI and seed.
  * edge time series (-edge): the co-fluctuation amplitude RSS(t) per TR, and with
    -edge_ts every z_i(t) z_j(t) product streamed to a .npy.
  * graph metrics per ROI (-graph): strength, clustering, betweenness, efficiency.

Cleaning (per run, on the kept TRs only): Legendre drift (-polort), -ortvec
regressors, and -passband as stopband sines/cosines -- all removed in one
projection, so the band and the motion regressors are removed given each other.
ROI means are averaged before cleaning; the projection is linear, so this equals
cleaning every voxel first, at a fraction of the cost.

Outputs (PREFIX = -prefix without extension):
  PREFIX.netcc       matrices in 3dNetCorr's layout (4 decimals; FATCAT tools read it)
  PREFIX_fc.npz      the same matrices at full precision, plus ids and names
  PREFIX_rois.tsv    one row per ROI: id, name, voxels (and graph metrics)
  PREFIX.json        TRs kept, nuisance dimensions removed, residual dof, settings
  PREFIX_ts.tsv      cleaned ROI series (-ts_out)
  PREFIX_wb_r / _wb_z  whole-brain maps, one sub-brick per ROI/seed
  PREFIX_rss.tsv     RSS per TR (-edge); PREFIX_edge_ts.npy + _edges.tsv (-edge_ts)
  PREFIX_fc.png      the matrix, clustered, and the RSS trace (-plot)

Examples
--------
    ffs_fc -input rest_run-1.nii.gz rest_run-2.nii.gz -atlas Schaefer400.nii.gz \\
        -mask brain.nii.gz -ortvec_glob 'motion_run-*.1D' motion \\
        -censor censor.1D -passband 0.01 0.1 -part_corr -shrink lw -prefix sub01

    ffs_fc -input task.nii.gz -atlas glasser.nii.gz -mask brain.nii.gz \\
        -weight_corr cond_blocks.1D -prefix sub01_taskfc

    ffs_fc -input rest.nii.gz -atlas pcc.nii.gz -mask brain.nii.gz -wb_z -prefix pcc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_ortvec_arguments, parse_prefix


def _build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(prog="ffs_fc", description=__doc__, formatter_class=FfsHelpFormatter)
    io = p.add_argument_group("Input / output")
    io.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="4-D runs on one grid (or surface .gii), concatenated in order.",
    )
    io.add_argument("-prefix", required=True, help="Output prefix (see Outputs above).")
    io.add_argument(
        "-atlas",
        "-rois",
        dest="atlas",
        nargs="+",
        default=None,
        metavar="ROI_FILE",
        help="ROIs whose mean time courses are correlated. A 3-D integer label map\n"
        "(one ROI per value -- any number of parcels), a 3-D binary mask (one ROI,\n"
        "named after the file) or a 4-D weight stack (may overlap); surface data take\n"
        ".label.gii / .annot. Several files stack into one set. Voxels outside -mask\n"
        "and voxels constant in any run are left out of every mean.",
    )
    io.add_argument(
        "-seed_1D",
        nargs="+",
        default=None,
        metavar="FILE",
        help="External seed time courses (one column each, full concatenated length;\n"
        "AFNI [cols]/{rows} selectors work). Cleaned like the data, then treated as\n"
        "extra ROIs (ids -1, -2, ...) in the matrices and maps.",
    )
    io.add_argument(
        "-mask",
        default=None,
        metavar="FILE",
        help="Analysis mask. Without it, every voxel with variance in every run counts.",
    )
    io.add_argument("-tr", type=float, default=None, help="TR in s (default: header).")
    io.add_argument(
        "-no_scale",
        action="store_true",
        help="Average raw intensities into each ROI (3dNetCorr's behaviour). By default\n"
        "each voxel is first scaled to percent signal change per run, so a voxel's\n"
        "weight in its ROI mean does not depend on its brightness -- otherwise voxels\n"
        "near the coil and large veins pull every ROI mean toward their own signal.\n"
        "Voxel-wise maps are unaffected either way (a correlation ignores scale).",
    )

    cl = p.add_argument_group("Cleaning (per run, uncensored TRs only)")
    cl.add_argument(
        "-polort",
        default="auto",
        metavar="N|auto",
        help="Legendre drift degree per run. Slow drift is shared by every ROI and\n"
        "correlates them for no neural reason; per-run means differ, and concatenating\n"
        "runs without removing them puts a step shared by all ROIs at every boundary.\n"
        "auto = AFNI's 1 + floor(run_seconds/150); costs polort+1 dof per run.\n"
        "-1 = no detrend (only the overall mean, as a plain Pearson r).",
    )
    add_ortvec_arguments(cl)
    cl.add_argument(
        "-censor",
        default=None,
        metavar="FILE.1D",
        help="1 = keep, 0 = censor, one row per concatenated TR. Censored TRs are\n"
        "dropped before the nuisance fit, so a spike cannot pull the drift fit.",
    )
    cl.add_argument(
        "-passband",
        nargs=2,
        type=float,
        default=None,
        metavar=("FBOT", "FTOP"),
        help="Keep FBOT..FTOP Hz: every Fourier frequency of each run outside it is\n"
        "projected out as a sine/cosine pair (3dTproject -passband). Correct across\n"
        "censored gaps. Costs two dof per removed frequency -- 0.01-0.1 Hz at TR 2 s\n"
        "keeps ~36%% of the spectrum. FTOP 0 = highpass only.",
    )

    me = p.add_argument_group("Measures")
    me.add_argument(
        "-part_corr",
        action="store_true",
        help="Also write partial correlation (PC) and partial-regression (PCB, row\n"
        "regressed on column) matrices from the inverse correlation matrix.",
    )
    me.add_argument(
        "-shrink",
        choices=["none", "lw"],
        default="none",
        help="lw: also write the Ledoit-Wolf shrunk correlation (LW; with -part_corr,\n"
        "PC_LW from it). Needed for a partial correlation when ROIs approach the\n"
        "residual dof. Sampling noise is counted in residual dof, not TRs (a passband\n"
        "leaves far fewer independent samples than time points); the shrinkage used\n"
        "is in PREFIX.json.",
    )
    me.add_argument(
        "-lw_samples",
        choices=["dof", "tr"],
        default="dof",
        help="What -shrink lw counts as its sample size. dof: the residual degrees of\n"
        "freedom -- after a passband far fewer independent samples than TRs, so the\n"
        "shrinkage is larger (0.058 vs 0.009 on one TASKFORCE run, which predicted the\n"
        "held-out run better; one subject, one split). tr: the textbook formula with\n"
        "n = TRs, as scikit-learn / nilearn compute it, for comparison with them.",
    )
    me.add_argument(
        "-weight_corr",
        default=None,
        metavar="FILE.1D",
        help="Per-TR non-negative weights (full concatenated length, one column per\n"
        "weighting): a weighted Pearson r for each column, written as CC_W1, FZ_W1, ...\n"
        "(3dNetCorr -weight_corr). Censored TRs get weight 0 regardless.",
    )
    me.add_argument("-wb_corr", action="store_true", help="Whole-brain r map per ROI/seed.")
    me.add_argument("-wb_z", action="store_true", help="Whole-brain Fisher-z map per ROI/seed.")
    me.add_argument("-ts_out", action="store_true", help="Write the cleaned ROI time series.")
    me.add_argument(
        "-edge",
        action="store_true",
        help="Edge co-fluctuation: RSS(t) over all ROI pairs per kept TR (PREFIX_rss.tsv).",
    )
    me.add_argument(
        "-edge_ts",
        action="store_true",
        help="Also write every edge time series, N(N-1)/2 x kept TRs float32, to\n"
        "PREFIX_edge_ts.npy (400 ROIs x 600 TRs = 192 MB) with PREFIX_edges.tsv.",
    )
    me.add_argument(
        "-graph",
        action="store_true",
        help="Graph metrics per ROI on the CC matrix (added to PREFIX_rois.tsv).",
    )
    me.add_argument(
        "-graph_signed",
        choices=["absolute", "positive"],
        default="positive",
        help="Edge weights for -graph: positive r only, or |r|.",
    )
    me.add_argument(
        "-graph_density",
        type=float,
        default=None,
        metavar="FRAC",
        help="Keep the strongest FRAC of edges for -graph (default: all).",
    )
    me.add_argument(
        "-plot",
        action="store_true",
        help="PREFIX_fc.png: the CC matrix in clustered order with module lines, on a\n"
        "fixed -1..1 scale (with -edge, the RSS trace under it).",
    )
    me.add_argument(
        "-no_netcc",
        action="store_true",
        help="Skip the text .netcc (the .npz is always written). For very large atlases.",
    )
    add_device_arg(p)
    p.add_argument("-quiet", action="store_true", help="Suppress progress output.")
    return p


def _say(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _read_columns(path: str, n_rows: int, what: str) -> np.ndarray:
    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    arr = np.asarray(load_nuisance_file(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n_rows and arr.shape[1] == n_rows:
        arr = arr.T
    if arr.shape[0] != n_rows:
        raise ValueError(f"{what} {path}: {arr.shape[0]} rows for {n_rows} concatenated TRs")
    return arr


def _resolve_rows(args, shape3, mask3, want_maps, header, path):
    """``(roi_set | None, rows)``: the ROI set and the voxels to load."""
    from fastfuncstuff.processing.roi_collapse import build_roi_set

    if args.atlas:
        rs = build_roi_set(
            args.atlas, shape3, mask=mask3, restrict_to_rois=not want_maps,
            reference_header=header, reference_path=path,
        )  # fmt: skip
        return rs, rs.rows
    rows = mask3 if mask3 is not None else np.ones(shape3, bool)
    return None, rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    verbose = not args.quiet
    try:
        return _run(args, verbose)
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2


def _run(args, verbose: bool) -> int:
    import torch

    from fastfuncstuff.cli_utils import (
        assemble_per_run_nuisance,
        collect_nuisance_blocks,
        find_constant_voxels,
        load_and_preprocess_runs,
        run_lengths_from_starts,
        setup_device,
    )
    from fastfuncstuff.connectivity import edge as edge_mod
    from fastfuncstuff.connectivity.maps import seed_maps
    from fastfuncstuff.connectivity.matrix import (
        correlation_matrix,
        fisher_z,
        ledoit_wolf,
        partial_correlation,
    )
    from fastfuncstuff.connectivity.netcc import write_netcc
    from fastfuncstuff.connectivity.prep import build_cleaner
    from fastfuncstuff.design.polort import resolve_polort
    from fastfuncstuff.io.afni import load_nifti, save_nifti

    if not args.atlas and not args.seed_1D:
        raise ValueError("give -atlas/-rois, -seed_1D, or both")
    want_maps = args.wb_corr or args.wb_z
    if not args.atlas and not want_maps:
        raise ValueError("-seed_1D alone needs -wb_corr or -wb_z (nothing else to correlate)")
    if args.edge_ts:
        args.edge = True
    if args.passband is not None:
        fbot, ftop = args.passband
        if fbot < 0 or ftop < 0 or (ftop > 0 and ftop <= fbot):
            raise ValueError(
                f"-passband needs 0 <= FBOT < FTOP (FTOP 0 = no lowpass), got {fbot} {ftop}"
            )

    # TF32 off: every step here is a projection or a correlation, where its 1e-3
    # relative error is the result (see Cleaner.apply).
    device = setup_device(args.device, tf32=False)
    pre = parse_prefix(args.prefix)
    stem, ext = pre.stem, pre.nifti_ext
    Path(stem).parent.mkdir(parents=True, exist_ok=True)

    img0 = load_nifti(args.input[0])
    shape3 = tuple(img0.shape[:3])
    mask3 = None
    if args.mask:
        mask3 = np.asarray(load_nifti(args.mask).dataobj).reshape(shape3) != 0
    roi_set, rows = _resolve_rows(args, shape3, mask3, want_maps, img0.header, args.input[0])

    loaded = load_and_preprocess_runs(
        args.input, tr=args.tr, mask_array=rows, do_scale=False, device=device,
        verbose=verbose,
    )  # fmt: skip
    data, run_starts, n_t, tr = loaded.data, list(loaded.run_starts), loaded.n_timepoints, loaded.tr

    # Out-of-FOV and dead voxels: averaging zeros into an ROI shrinks its signal, and a
    # constant voxel has no correlation to report. Drop them from every mean and map.
    valid = find_constant_voxels(data, run_starts).cpu().numpy()
    if not valid.all():
        n_drop = int((~valid).sum())
        rows_new = np.zeros_like(rows)
        rows_new[rows] = valid
        data = data[torch.as_tensor(valid, device=data.device)]
        if roi_set is not None:
            roi_set, _ = _resolve_rows(
                args, shape3, rows_new, want_maps, img0.header, args.input[0]
            )
        rows = rows_new
        _say(verbose, f"   {n_drop:,} constant voxel(s) left out")
    if roi_set is not None and roi_set.rows.sum() != data.shape[0]:
        raise ValueError("internal: ROI rows and loaded voxels disagree")

    if not args.no_scale:
        from fastfuncstuff.connectivity.prep import has_baseline
        from fastfuncstuff.utils import scale_to_percent_signal

        if not has_baseline(data, run_starts):
            raise ValueError(
                "the input has no positive baseline (residuals, demeaned or z-scored data?): "
                "percent change would divide by ~0. Use -no_scale."
            )
        data, _, _ = scale_to_percent_signal(
            data, run_starts, verbose=False, track_violations=False
        )

    # -- series -------------------------------------------------------------------
    names: list[str] = []
    ids: list[int] = []
    series = []
    sizes: list[float] = []
    if roi_set is not None:
        for line in roi_set.describe():
            _say(verbose, f"   ROIs: {line}")
        series.append(roi_set.collapse(data, device=device))
        names += list(roi_set.names)
        ids += [int(i) for i in roi_set.ids]
        sizes += [float(s) for s in roi_set.sizes]
    if args.seed_1D:
        for path in args.seed_1D:
            cols = _read_columns(path, n_t, "-seed_1D")
            base = Path(path.split("[")[0]).stem
            for c in range(cols.shape[1]):
                names.append(base if cols.shape[1] == 1 else f"{base}_{c}")
                ids.append(-(len(ids) - (roi_set.n_rois if roi_set else 0) + 1))
                sizes.append(0.0)
            series.append(torch.as_tensor(cols.T, dtype=torch.float32, device=device))
    raw = torch.cat([s.to(device) for s in series], dim=0)

    # -- cleaner ------------------------------------------------------------------
    keep = None
    if args.censor:
        keep = _read_columns(args.censor, n_t, "-censor")[:, 0] != 0
    run_lengths = run_lengths_from_starts(run_starts, n_t)
    polort = resolve_polort(
        None if args.polort == "auto" else int(args.polort), run_lengths, tr, verbose=verbose,
        indent="   ",
    )  # fmt: skip
    blocks = collect_nuisance_blocks(args, run_starts, n_t, verbose=verbose)
    nuis = assemble_per_run_nuisance(blocks, run_starts, n_t, polort, device=torch.device("cpu"))
    cleaner = build_cleaner(
        nuis.per_run, run_starts, n_t, keep=keep, tr=tr,
        passband=tuple(args.passband) if args.passband else None,
    )  # fmt: skip
    _say(
        verbose,
        f"   {cleaner.n_kept}/{n_t} TRs kept, {cleaner.n_removed} nuisance dimensions "
        f"removed, residual dof {cleaner.dof}",
    )
    if args.passband is not None:
        lo, hi = args.passband
        band = f"passband {lo:g}-{hi:g} Hz" if hi > 0 else f"highpass {lo:g} Hz"
        _say(
            verbose,
            f"   {band} alone removes "
            f"{cleaner.passband_cost} of {cleaner.n_kept} dimensions "
            f"({100 * cleaner.passband_cost / max(cleaner.n_kept, 1):.0f}%): each r rests on "
            f"{cleaner.dof} independent samples, not {cleaner.n_kept}"
            + (" -- a highpass only (-passband FBOT 0) keeps them" if args.passband[1] > 0 else ""),
        )
    n_series = raw.shape[0]
    if n_series >= cleaner.dof:
        _say(
            verbose,
            f"   ⚠ {n_series} ROIs/seeds from {cleaner.dof} residual dof: Pearson r is fine "
            "(pairwise), but the matrix is rank-deficient -- no plain partial correlation "
            "or inverse exists; use -shrink lw for those",
        )
    if cleaner.dof < 3:
        raise ValueError(f"residual dof {cleaner.dof}: nothing left to correlate")
    clean = cleaner.apply(raw, device=device)

    weight_sets: list[tuple[str, np.ndarray | None]] = [("", None)]
    if args.weight_corr:
        w = _read_columns(args.weight_corr, n_t, "-weight_corr")[cleaner.keep]
        if (w < 0).any():
            raise ValueError("-weight_corr weights must be non-negative")
        weight_sets = [(f"_W{k + 1}", w[:, k]) for k in range(w.shape[1])]

    # -- matrices -----------------------------------------------------------------
    mats: dict[str, np.ndarray] = {}
    meta: dict = {}
    for tag, wt in weight_sets:
        r = correlation_matrix(clean, weights=wt).cpu().numpy()
        mats[f"CC{tag}"] = r
        mats[f"FZ{tag}"] = fisher_z(r)
        if args.part_corr:
            try:
                pc, pcb = partial_correlation(r, dof=cleaner.dof)
                mats[f"PC{tag}"], mats[f"PCB{tag}"] = pc, pcb
            except ValueError as exc:
                if args.shrink != "lw" or tag:
                    raise
                # The shrunk PC below is the answer the user asked for as well.
                _say(verbose, f"   PC skipped, PC_LW only: {exc}")
                meta["pc_skipped"] = str(exc)
    if args.shrink == "lw":
        if args.weight_corr:
            raise ValueError("-shrink lw is not defined for weighted correlations")
        n_eff = cleaner.dof if args.lw_samples == "dof" else None
        lw, shrinkage = ledoit_wolf(clean, n_effective=n_eff)
        mats["LW"] = lw.cpu().numpy()
        meta["lw_shrinkage"] = shrinkage
        meta["lw_samples"] = args.lw_samples
        _say(verbose, f"   Ledoit-Wolf shrinkage {shrinkage:.4f}")
        if args.part_corr:
            mats["PC_LW"] = partial_correlation(mats["LW"])[0]  # PD by construction

    written = []
    if not args.no_netcc:
        written.append(write_netcc(f"{stem}.netcc", mats, ids, names))
    np.savez_compressed(f"{stem}_fc.npz", ids=np.asarray(ids), names=np.asarray(names), **mats)
    written.append(f"{stem}_fc.npz")

    table = {"roi": ids, "name": names, "n_voxels": sizes}
    if args.graph:
        from fastfuncstuff.dynamics.graph import (
            betweenness_centrality,
            efficiency,
            fc_to_weights,
            node_strength,
            weighted_clustering,
        )

        key = "CC" if "CC" in mats else "CC_W1"
        gw = fc_to_weights(mats[key], signed=args.graph_signed, density=args.graph_density)
        if len(ids) > 400:
            _say(verbose, f"   graph metrics on {len(ids)} nodes: betweenness takes a while")
        glob, nodal = efficiency(gw)
        table.update(
            strength=node_strength(gw),
            clustering=weighted_clustering(gw),
            betweenness=betweenness_centrality(gw),
            nodal_efficiency=nodal,
        )
        meta["global_efficiency"] = glob
    with open(f"{stem}_rois.tsv", "w") as fh:
        cols = list(table)
        fh.write("\t".join(cols) + "\n")
        for i in range(len(ids)):
            vals = [table[c][i] for c in cols]
            fh.write("\t".join(v if isinstance(v, str) else f"{v:.6g}" for v in vals) + "\n")
    written.append(f"{stem}_rois.tsv")

    kept_idx = np.flatnonzero(cleaner.keep)
    if args.ts_out:
        ts = clean.cpu().numpy()
        with open(f"{stem}_ts.tsv", "w") as fh:
            fh.write("tr\ttime_s\t" + "\t".join("_".join(n.split()) for n in names) + "\n")
            for j, t in enumerate(kept_idx):
                fh.write(f"{t}\t{t * tr:.6g}\t" + "\t".join(f"{v:.6g}" for v in ts[:, j]) + "\n")
        written.append(f"{stem}_ts.tsv")

    amp = None
    if args.edge:
        z = edge_mod.zscore_rows(clean)
        amp = edge_mod.rss(z)
        with open(f"{stem}_rss.tsv", "w") as fh:
            fh.write("tr\ttime_s\trss\n")
            for t, a in zip(kept_idx, amp, strict=True):
                fh.write(f"{t}\t{t * tr:.6g}\t{a:.6g}\n")
        written.append(f"{stem}_rss.tsv")
        if args.edge_ts:
            n_e, _ = edge_mod.write_edge_timeseries(z, f"{stem}_edge_ts.npy")
            iu, ju = edge_mod.edge_index(len(ids))
            with open(f"{stem}_edges.tsv", "w") as fh:
                fh.write("edge\troi_i\troi_j\tname_i\tname_j\n")
                for e, (a, b) in enumerate(zip(iu, ju, strict=True)):
                    fh.write(f"{e}\t{ids[a]}\t{ids[b]}\t{names[a]}\t{names[b]}\n")
            written += [f"{stem}_edge_ts.npy", f"{stem}_edges.tsv"]
            _say(verbose, f"   {n_e:,} edge series written")

    if args.plot:
        from fastfuncstuff.connectivity.plots import plot_fc

        key = "CC" if "CC" in mats else "CC_W1"
        written.append(
            plot_fc(f"{stem}_fc.png", mats[key], names, rss=amp, times=kept_idx * tr,
                    title=f"{Path(stem).name}: {key}, {len(ids)} ROIs, dof {cleaner.dof}")
        )  # fmt: skip

    if want_maps:
        vox = cleaner.apply(data, device=device, out_device=data.device, verbose=verbose)
        del data
        for tag, wt in weight_sets:
            r = seed_maps(vox, clean, weights=wt, device=device, verbose=verbose).numpy()
            # AFNI fico: (samples, fitted params, nuisance params) -> dof for thresholds.
            aux = None if wt is not None else (2, (cleaner.n_kept, 1, cleaner.n_removed))
            for kind, flag in (("r", args.wb_corr), ("z", args.wb_z)):
                if not flag:
                    continue
                vals = r if kind == "r" else fisher_z(r)
                vol = np.zeros((*shape3, vals.shape[1]), np.float32)
                vol[rows] = vals
                path = f"{stem}_wb_{kind}{tag}{ext}"
                stataux = {i: aux for i in range(vals.shape[1])} if aux and kind == "r" else None
                save_nifti(
                    vol, path, affine=loaded.affine, brick_labels=names, brick_stataux=stataux
                )
                written.append(path)

    meta.update(
        tool="ffs_fc",
        inputs=list(args.input),
        # Absolute, so a viewer (or a later group step) can find the ROIs behind the rows.
        atlas=[str(Path(a.split("[")[0]).resolve()) for a in (args.atlas or [])],
        mask=str(Path(args.mask).resolve()) if args.mask else None,
        seeds=list(args.seed_1D or []),
        n_rois=len(ids),
        n_timepoints=n_t,
        n_kept=cleaner.n_kept,
        n_removed=cleaner.n_removed,
        passband_dims_removed=cleaner.passband_cost,
        nuisance_rank_per_run=cleaner.ranks,
        dof=cleaner.dof,
        tr=tr,
        polort=polort,
        passband=args.passband,
        censor=args.censor,
        weights=args.weight_corr,
        scaled=not args.no_scale,
        fisher_z_cap=4.0,
        matrices=list(mats),
        notes=roi_set.notes if roi_set is not None else [],
    )
    with open(f"{stem}.json", "w") as fh:
        json.dump(
            meta, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        )
    written.append(f"{stem}.json")
    for path in written:
        _say(verbose, f"   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
