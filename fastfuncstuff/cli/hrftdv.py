"""ffs_hrftdv: per-column early/late HRF pairs for temporal deveining with ffs_reml.

Command: ffs_hrftdv (registered as entry point in pyproject.toml)

Usage:
    ffs_hrftdv -input run1.nii.gz run2.nii.gz -events ev1.tsv ev2.tsv \\
        -fs_subj $SUBJECTS_DIR/sub01 -prefix tdv/sub01
    ffs_reml -input run1.nii.gz run2.nii.gz -events ev1.tsv ev2.tsv \\
        -hrf_opt tdv/sub01 -Rbuck stats_tdv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter, suggest
from fastfuncstuff.cli_utils import (
    add_bids_events_args,
    add_device_arg,
    add_microtime_offset_arg,
    add_ortvec_arguments,
    add_round_onsets_arguments,
    add_verbose_arg,
)
from fastfuncstuff.design.hrf_modes import HRF_LIBRARY_MODES, hrf_mode_type
from fastfuncstuff.surface.ribbon_args import add_depth_args, add_ribbon_args

_EPILOG = """\
what it does

  Veins deliver the BOLD response later and wider than the capillaries they
  drain (Kay et al. 2020, "TDM").  Fitting an early and a late curve together,
  two columns per condition, lets the late column take the venous part, so
  the early betas are less dominated by the pial surface.  TDM uses one pair
  for the whole brain; this tool finds a pair per cortical column:

    1. units    ribbon voxels -> geodesic patches of the midthickness mesh
                (-patch_mm), through each voxel's owning vertex, so a unit is
                every depth under a few mm of cortex.  Voxels within
                -extend_mm of the ribbon join the nearest ribbon voxel's patch
                (touching gyri split the gap, never share it); the rest of the
                covered brain is cut into -tile_mm cubes.
    2. shape    each unit's mean response picks its library curve.
    3. lag      every voxel scores that curve shifted by up to +-max_lag s and
                stretched by -widths about its peak (exact shifting).
    4. pair     R²-weighted -percentiles of the responsive voxels' lags give
                the unit's early and late lags; each end takes the width of
                the voxels near it, so the early curve can be narrower than
                the unit mean (which is a blend of lags, so too wide).
    5. library  pairs snapped to -pair_step become {prefix}_hrf_library.pt;
                ffs_reml -hrf_opt PREFIX fits them as {cond}_early / _late.

  Surfaces must be in the scanner space of -input (an -fs_subj aligned to the
  functional data, or meshes already moved there via -white_mesh/-pial_mesh).
  No -mask: the surfaces define the ribbon, and coverage comes from the data
  (voxels positive in every run).  Tiles stay inside a 3dAutomask of the mean
  (or -tile_mask).

outputs

  {prefix}_hrf_index.nii.gz    [0] pair entry (1..E, 0 = not covered), [1] lag-fit R²
  {prefix}_hrf_library.pt      (E, 2, n_t) early/late curves + entry table
  {prefix}_tdv.nii.gz          bucket: lag, rel_lag, width, r2, at_bound, responsive,
                               unit, unit_early_lag, unit_late_lag, depth
  {prefix}_pairs.tsv           one row per library entry
  {prefix}_roi.tsv             with -atlas: one row per ROI (its pair, curve timing,
                               voxel and responsive counts)
  {prefix}_roi_depth.tsv       with -roi_lag: one row per ROI per depth (lag, width,
                               R², peak, FWHM, per-run lags), and one PNG of the
                               best-fit ROI (more: python -m fastfuncstuff.cli.hrftdv_plot)
  {prefix}_metadata.json

  lag is relative to the unit curve's peak (s); rel_lag subtracts the unit's
  R²-weighted median, so it compares voxels within a column.  depth is
  equivolume, 0 at white, 1 at pial (ribbon voxels only).

what to expect

  Measured on 1 mm data (TR 2, 10 s blocks): rel_lag replicates run to run in
  well-fit voxels and rises toward pial and into dark-EPI voxels.  Width is
  weakly determined with block designs.  Fitting early + late costs
  sensitivity (the two columns are collinear, VIF ~5-20 by design); event
  designs with short events separate them best.
"""


def create_parser() -> FfsArgumentParser:
    parser = FfsArgumentParser(
        prog="ffs_hrftdv",
        description="Per-column early/late HRF pairs for temporal deveining (ffs_reml -hrf_opt)",
        formatter_class=FfsHelpFormatter,
        epilog=_EPILOG,
    )
    req = parser.add_argument_group("Inputs")
    req.add_argument("-input", nargs="+", required=True, help="Runs (one file per run)")
    req.add_argument("-prefix", required=True, help="Output prefix")
    req.add_argument("-onsets", nargs="+", default=None, help="AFNI onset files, one per condition")
    req.add_argument("-durations", nargs="+", default=None, help="Durations for -onsets (s)")
    req.add_argument(
        "-spec",
        default=None,
        metavar="TOML",
        help="Design TOML (ffs_design_spec): its conditions, durations, nuisance blocks "
        "and polort, the same model ffs_reml -spec ... -hrf_opt fits. Its HRFs and "
        "censor are not used here: the pairs come from the library curves, fit uncensored",
    )
    add_bids_events_args(req, events_help="BIDS *_events.tsv files, one per run")
    add_round_onsets_arguments(req, hyphen=False)
    req.add_argument(
        "-round_durations",
        type=int,
        default=None,
        metavar="PLACES",
        help="Round durations to PLACES decimals before condition uniquing",
    )
    req.add_argument("-tr", type=float, default=None, help="TR in seconds (default: header)")

    surf = parser.add_argument_group("Cortical ribbon")
    add_ribbon_args(surf)

    units = parser.add_argument_group("Units")
    units.add_argument(
        "-patch_mm",
        type=float,
        default=5.0,
        help="Geodesic radius of a cortical patch (the column a pair belongs to)",
    )
    units.add_argument(
        "-extend_mm",
        type=float,
        default=2.0,
        help="Ribbon grown this far into WM and past pial, each voxel to its nearest "
        "ribbon voxel's patch. A hedge against a tight segmentation, and a way to see "
        "whether the late component really sits on the pial veins. 0 = ribbon only",
    )
    units.add_argument(
        "-tile_mm",
        type=float,
        default=4.0,
        help="Cube size for the brain voxels outside the (grown) ribbon. 0 = leave "
        "them out (reml then gives them the entry closest to canonical)",
    )
    units.add_argument(
        "-tile_mask",
        default=None,
        metavar="MASK",
        help="Brain mask for the tiles (default: 3dAutomask of the mean). Does not "
        "restrict the ribbon, which the surfaces define",
    )
    units.add_argument(
        "-min_voxels",
        type=int,
        default=8,
        help="Responsive voxels a unit needs for its own pair; fewer -> the median pair",
    )

    atlas = parser.add_argument_group("Atlas units (-atlas)")
    atlas.add_argument(
        "-atlas",
        "-rois",
        dest="atlas",
        nargs="+",
        default=None,
        metavar="ROI_FILE",
        help="ROIs as units instead of geodesic patches. Volume ROIs (3-D labels, a "
        "mask, a 4-D stack) claim the columns they touch; surface ROIs (.label.gii, "
        ".label, .annot) name them. A voxel joins the ROI owning its column (the first "
        "listed where ROIs overlap), so every pair is an ROI's. Writes {prefix}_roi.tsv. "
        "For ROI x depth early/late betas: ffs_reml -atlas (same files) -roi_claim any "
        "-fs_subj ... -depths N -hrf_opt PREFIX",
    )
    atlas.add_argument(
        "-atlas_unit",
        choices=["roi", "patch"],
        default="roi",
        help="roi: one pair per ROI. patch: one per geodesic patch inside each ROI, "
        "so a large ROI keeps local timing (the table then reports medians)",
    )
    atlas.add_argument(
        "-atlas_only",
        action="store_true",
        help="Fit only the atlas: no patches outside it, no tiles (reml gives the rest "
        "the entry closest to canonical)",
    )
    atlas.add_argument(
        "-roi_hemi",
        choices=["auto", "split", "major", "both"],
        default="auto",
        help="An ROI with columns on both hemispheres: split into one per hemisphere, "
        "keep the major side, or keep both as one unit (auto: split when the smaller "
        "side has a third of the columns)",
    )
    add_depth_args(atlas, ribbon=False)
    atlas.add_argument(
        "-roi_lag",
        action="store_true",
        help="Also fit lag and width on each ROI x depth mean (-depths N), against its "
        "ROI's column curve, on all runs and on each run alone: the best-SNR measure of "
        "how timing changes through cortex. Writes {prefix}_roi_depth.tsv and a plot of "
        "the best-fit ROI",
    )
    atlas.add_argument(
        "-pair_from",
        choices=["percentiles", "depth"],
        default="percentiles",
        help="Where an ROI's early/late pair comes from. percentiles: its voxels' lags "
        "(-percentiles). depth: the -roi_lag fit of its deepest and outermost depth rows, "
        "anchoring the split on anatomy. Experimental; an ROI whose outer row is not "
        "later than its deep row keeps its percentile pair",
    )

    model = parser.add_argument_group("HRF and timing")
    suggest(
        model.add_argument(
            "-hrf_mode",
            type=hrf_mode_type(),
            default="library",
            help="Column curves: the 20-curve library, -n_hrfs PIGHS shapes, or "
            "blend[:N], the library plus the N (default 20) PIGHS shapes it matches worst",
        ),
        HRF_LIBRARY_MODES,
    )
    model.add_argument("-n_hrfs", type=int, default=20, help="PIGHS curves for -hrf_mode pighs")
    model.add_argument(
        "-hrf_library",
        default=None,
        metavar="TSV",
        help="Impulse-response library TSV (getcanonicalhrflibrary.tsv format) for "
        "-hrf_mode library",
    )
    model.add_argument(
        "-max_lag", type=float, default=2.0, help="Largest lag searched either way (s)"
    )
    model.add_argument("-lag_step", type=float, default=0.1, help="Lag grid step (s)")
    model.add_argument(
        "-widths",
        type=float,
        nargs="+",
        default=[0.6, 0.75, 0.875, 1.0, 1.15, 1.3, 1.5],
        help="Stretch factors about the peak",
    )
    model.add_argument(
        "-percentiles",
        type=float,
        nargs=2,
        default=[10.0, 90.0],
        metavar=("EARLY", "LATE"),
        help="R²-weighted lag percentiles that define the early and late curves",
    )
    model.add_argument(
        "-pair_step",
        type=float,
        default=0.2,
        help="Lag snapping for the library (s): units within a step share an entry",
    )
    model.add_argument(
        "-pooled_late",
        action="store_true",
        help="Also write the pooled-late model: {prefix}_early_hrf_* (one early curve per "
        "voxel, for -hrf_opt) and {prefix}_late_dsort.run-NN (each voxel's single late "
        "column: its unit's late curve x the unit's late condition pattern, for -dsort). "
        "One late column instead of one per condition stops the early and late betas "
        "fighting (VIF ~16 -> ~4 on an event design), at the cost of calling the "
        "unit's shared selectivity late",
    )
    model.add_argument(
        "-p_thresh",
        type=float,
        default=1e-4,
        help="F-test p a voxel must pass to help set its unit's shape and pair",
    )
    model.add_argument(
        "-screen",
        choices=["canonical", "bank", "library"],
        default="canonical",
        help="Design of the responsiveness F-test. canonical: one column per condition, "
        "which favours voxels shaped like the canonical HRF, the ones whose lag says "
        "least: canonical is close to the late (venous) response, so the early, "
        "narrow voxels fare worst. library: every voxel fit with every curve of the "
        "HRF library in use (-hrf_mode), judged by its best one -- responsive only in "
        "a shape the library allows. bank: an F-test on the leading -screen_k "
        "components of the canonical curve's lag x width bank (free combinations: "
        "admits shapes no library curve has)",
    )
    model.add_argument(
        "-screen_k", type=int, default=3, help="Components per condition for -screen bank"
    )
    model.add_argument(
        "-unit_mean",
        choices=["responsive", "weighted", "all"],
        default="responsive",
        help="Which voxels set a unit's shape. responsive: the mean of those passing "
        "-p_thresh (units with none take the commonest shape). weighted: every "
        "positive voxel, weighted by its screen R². all: the plain mean of every "
        "voxel, no screen at all. Veins dominate the last two; the pair is relative "
        "to the unit curve, so that only moves the anchor",
    )
    model.add_argument("-microtime_dt", type=float, default=0.1, help="Microtime step (s)")
    add_microtime_offset_arg(model)

    nuis = parser.add_argument_group("Nuisance")
    nuis.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Legendre drift order for every run (default: per run, 1 + floor(s/150))",
    )
    add_ortvec_arguments(nuis)
    add_device_arg(nuis)
    add_verbose_arg(nuis)
    parser.set_defaults(roi_claim="any")
    return parser


def _atlas_regions(args, surfaces, verbose: bool) -> list:
    """-atlas read against the ribbon, one region per hemisphere by -roi_hemi."""
    from fastfuncstuff.processing.roi_depth import CLAIM_ANY, CLAIM_MID, read_depth_rois
    from fastfuncstuff.surface.volume_depth import split_regions

    notes: list[str] = []
    try:
        rois = read_depth_rois(
            args.atlas, surfaces, notes, CLAIM_ANY if args.roi_claim == "any" else CLAIM_MID
        )
        regions = split_regions(rois, surfaces, args.roi_hemi, notes)
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: -atlas: {exc}")
    if verbose:
        for note in notes:
            print(f"  {note}")
    return regions


def _write_roi_table(
    path, regions, surfaces, roi_of_unit, unit_of, res, shapes, shape_labels, dt
) -> str:
    """One row per ROI: its pair and curve timing, voxel-weighted medians over its units."""
    import csv

    from fastfuncstuff.processing.hrftdv import curve_peak_fwhm, stretch_shift

    usable = res.responsive & ~res.fit.at_bound
    n_vox = np.bincount(unit_of, minlength=len(res.pairs))
    n_resp = np.bincount(unit_of, weights=res.responsive.astype(float), minlength=len(res.pairs))
    n_use = np.bincount(unit_of, weights=usable.astype(float), minlength=len(res.pairs))
    unit_shape = res.shape[np.unique(unit_of, return_index=True)[1]]
    cols = ["roi", "name", "hemi", "n_columns", "n_units", "n_fallback", "n_voxels",
            "n_responsive", "n_usable", "shape", "column_peak_s", "column_fwhm_s",
            "early_lag", "late_lag", "separation", "early_width", "late_width",
            "early_peak_s", "late_peak_s", "early_fwhm_s", "late_fwhm_s"]  # fmt: skip

    def wmed(x, w):
        if w.sum() <= 0:
            return float("nan")
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(x[o][min(int(np.searchsorted(c, 0.5 * c[-1])), x.size - 1)])

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r, reg in enumerate(regions):
            units = np.flatnonzero(roi_of_unit == r)
            hemi = ",".join(surfaces[h].name for h, m in sorted(reg.members.items()) if np.any(m))
            n_cols = int(sum(np.asarray(m, bool).sum() for m in reg.members.values()))
            if units.size == 0:
                w.writerow([reg.rid, reg.name, hemi, n_cols, 0] + [""] * (len(cols) - 5))
                continue
            wt = n_vox[units].astype(np.float64)
            pr = [res.pairs[u] for u in units]
            shp = unit_shape[units]
            main_shape = int(np.bincount(shp, weights=wt).argmax())
            col = curve_peak_fwhm(shapes[main_shape], dt)
            e_lag = wmed(np.array([p.early_lag for p in pr]), wt)
            l_lag = wmed(np.array([p.late_lag for p in pr]), wt)
            e_w = wmed(np.array([p.early_width for p in pr]), wt)
            l_w = wmed(np.array([p.late_width for p in pr]), wt)
            early = curve_peak_fwhm(stretch_shift(shapes[main_shape], dt, e_lag, e_w), dt)
            late = curve_peak_fwhm(stretch_shift(shapes[main_shape], dt, l_lag, l_w), dt)
            vals = [reg.rid, reg.name, hemi, n_cols, units.size, sum(p.fallback for p in pr),
                    int(wt.sum()), int(n_resp[units].sum()), int(n_use[units].sum()),
                    shape_labels[main_shape], *col, e_lag, l_lag, l_lag - e_lag, e_w, l_w,
                    early[0], late[0], early[1], late[1]]  # fmt: skip
            w.writerow([f"{v:.4g}" if isinstance(v, float) else v for v in vals])
    return str(path)


def _fit_roi_depth(
    args, surfaces, affine, shape, analysed, y, unit_of, res, shapes, design_for, project,
    grid, dt, run_starts, device, verbose,
) -> dict:  # fmt: skip
    """ROI x depth means of the projected data, each fit for lag and width (-roi_lag)."""
    import torch

    from fastfuncstuff.processing import hrftdv as tdv
    from fastfuncstuff.processing.roi_depth import (
        build_depth_roi_sets,
        depth_list,
        options_from_args,
    )

    try:
        (rs,) = build_depth_roi_sets(
            args.atlas, surfaces, affine, shape, analysed.reshape(shape),
            options_from_args(args), depth_list(args), device=device,
        )  # fmt: skip
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: -roi_lag: {exc}")
    vox = np.flatnonzero(analysed)
    pos = np.searchsorted(vox, np.flatnonzero(rs.rows.reshape(-1)))
    y_rows = rs.collapse(y[:, torch.as_tensor(pos)].T.contiguous()).T.float()
    row_unit, share = rs.dominant(unit_of[pos])
    unit_shape = np.zeros(len(res.pairs), np.int64)
    unit_shape[res.unit] = res.shape
    row_shape = unit_shape[row_unit]
    lags = tdv.fit_row_lags(
        y_rows, row_shape, shapes, design_for, project, grid, dt, run_starts, device
    )
    if verbose:
        print(
            f"  ROI x depth lags: {rs.n_rois} rows, median R² {np.nanmedian(lags.r2):.2f}, "
            f"{int(lags.at_bound.sum())} at the lag bound"
        )
    return {"set": rs, "unit": row_unit, "share": share, "shape": row_shape, "lags": lags}


def _roi_keys(rs) -> list[tuple]:
    """(roi id, hemi) of every row of a depth-sampled set."""
    hemi = rs.index.get("hemi", [""] * rs.n_rois)
    return list(zip(rs.ids, hemi, strict=True))


def _pairs_from_depth(rd, res, shapes, dt, lag_step, grid, verbose):
    """-pair_from depth: each ROI's pair from its deepest and outermost depth rows."""
    from fastfuncstuff.processing import hrftdv as tdv

    rs, lags = rd["set"], rd["lags"]
    depth = np.asarray(rs.index["depth"]).astype(int)
    keys = _roi_keys(rs)
    overrides, kept = {}, []
    for key in dict.fromkeys(keys):
        rows = [i for i, k in enumerate(keys) if k == key]
        deep = min(rows, key=lambda i: depth[i])
        outer = max(rows, key=lambda i: depth[i])
        unit = int(rd["unit"][deep])
        ok = (
            rd["unit"][outer] == unit
            and min(rd["share"][deep], rd["share"][outer]) >= 0.9
            and not (lags.at_bound[deep] or lags.at_bound[outer])
            and lags.lag[outer] > lags.lag[deep]
        )
        if not ok:
            kept.append(rs.names[deep])
            continue
        overrides[unit] = tdv.ColumnPair(
            float(lags.lag[deep]), float(lags.lag[outer]),
            float(lags.width[deep]), float(lags.width[outer]),
            res.pairs[unit].n_voxels,
        )  # fmt: skip
    if verbose:
        print(f"  Pairs from depth: {len(overrides)} ROI(s) set by their deepest/outermost rows")
        if kept:
            more = ", ..." if len(kept) > 5 else ""
            print(
                f"  ⚠️  {len(kept)} kept their percentile pair (outer row not later, at the "
                f"lag bound, or not one unit): {', '.join(kept[:5])}{more}"
            )
    return tdv.override_pairs(res, overrides, shapes, dt, lag_step, grid.widths)


def _write_roi_depth(stem, rd, res, shapes, dt) -> list[str]:
    """{stem}.tsv, one row per ROI x depth, and a PNG of the best-fit ROI."""
    from fastfuncstuff.processing.hrftdv import curve_peak_fwhm, stretch_shift

    rs, lags = rd["set"], rd["lags"]
    timing = np.array([
        curve_peak_fwhm(stretch_shift(shapes[s], dt, lg, w), dt)
        if np.isfinite(lg) else (np.nan, np.nan)
        for s, lg, w in zip(rd["shape"], lags.lag, lags.width, strict=True)
    ])  # fmt: skip
    pairs = [res.pairs[u] for u in rd["unit"]]
    columns = {
        "unit_share": rd["share"],
        "lag": lags.lag,
        "width": lags.width,
        "r2": lags.r2,
        "at_bound": lags.at_bound.astype(float),
        "peak_s": timing[:, 0],
        "fwhm_s": timing[:, 1],
        "shape": rd["shape"].astype(float),
        "pair_early_lag": np.array([p.early_lag for p in pairs]),
        "pair_late_lag": np.array([p.late_lag for p in pairs]),
        "pair_early_width": np.array([p.early_width for p in pairs]),
        "pair_late_width": np.array([p.late_width for p in pairs]),
        "lag_run_sd": np.nanstd(lags.run_lag, axis=1),
        "lag_run": lags.run_lag,
    }
    return [rs.write_table(f"{stem}.tsv", columns)]


def _plot_best_roi(prefix: str, rd) -> list[str]:
    """The sanity-check plot: the ROI whose depth rows fit best (mean R²)."""
    keys = _roi_keys(rd["set"])
    r2 = rd["lags"].r2

    def score(k):
        return float(np.nanmean([r2[i] for i, kk in enumerate(keys) if kk == k]))

    best = max(dict.fromkeys(keys), key=score)
    try:
        from fastfuncstuff.cli.hrftdv_plot import plot_from_files

        return plot_from_files(prefix, [best])
    except ImportError:  # no matplotlib: the table is the result
        return []


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args(argv)
    t0 = time.time()

    import torch

    from fastfuncstuff.cli_utils import (
        apply_onset_rounding,
        bids_event_kwargs,
        build_nuisance_per_run,
        check_bids_event_args,
        collect_nuisance_blocks,
        compute_run_lengths,
        load_and_preprocess_runs,
        microtime_offset_bins,
        parse_input_files,
        parse_prefix,
        parse_timing_spec,
        resolve_microtime_dt,
        resolve_microtime_offset,
        resolve_polort,
        setup_device,
    )
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.design.hrf_modes import BLEND_DEFAULT_EXTRA, parse_hrf_mode
    from fastfuncstuff.design.matrices import build_task_design
    from fastfuncstuff.design.shifted_hrf import build_shape_library
    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.processing import hrftdv as tdv
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    verbose = args.verb >= 1
    pfx = parse_prefix(args.prefix)
    ext = ".nii.gz" if pfx.nifti_ext.endswith(".gii") else pfx.nifti_ext
    Path(pfx.stem).parent.mkdir(parents=True, exist_ok=True)

    if sum(map(bool, (args.onsets, args.events, args.spec))) != 1:
        sys.exit("ERROR: give exactly one of -onsets/-durations, -events or -spec")
    if args.spec:
        from fastfuncstuff.cli.design_spec import resolve_spec_timing
        from fastfuncstuff.cli_utils import resolve_event_filters

        try:
            # Row filters stack on the spec's own, as with ffs_reml -spec.
            st = resolve_spec_timing(args.spec, event_filters=resolve_event_filters(args))
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"ERROR: -spec: {exc}")
        args.event_filter_in = args.event_filter_out = None
        args.onsets, args.durations = st.onset_files, [f"{d:g}" for d in st.durations]
        args.ortvec = (args.ortvec or []) + [list(x) for x in st.ortvec]
        args.ortvec_run = (args.ortvec_run or []) + [list(x) for x in st.ortvec_run]
        if args.polort is None:
            args.polort = st.polort
        if verbose:
            print(f"  Spec {args.spec}: {', '.join(st.labels)}")
    check_bids_event_args(args, has_events=bool(args.events))
    if args.onsets and args.durations is None:
        sys.exit("ERROR: -durations is required with -onsets")
    if args.fs_subj is None and not (args.white_mesh and args.pial_mesh):
        sys.exit("ERROR: give -fs_subj, or -white_mesh and -pial_mesh")
    if (args.roi_lag or args.pair_from == "depth") and not args.atlas:
        sys.exit("ERROR: -roi_lag and -pair_from depth fit ROI x depth means: give -atlas")
    if len(args.depths) != 1:
        sys.exit("ERROR: -depths takes one count here")
    if args.pair_from == "depth" and (args.depths[0] < 2 or args.atlas_unit != "roi"):
        sys.exit("ERROR: -pair_from depth needs -depths 2 or more and -atlas_unit roi")

    input_files = parse_input_files(args.input)
    try:
        timing = parse_timing_spec(
            events=args.events,
            onsets=args.onsets,
            durations_arg=args.durations,
            n_runs=len(input_files),
            **bids_event_kwargs(args),
            round_durations=args.round_durations,
            input_files=input_files,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    device = setup_device(args.device)

    # -- coverage: positive in every run, one run in memory at a time
    coverage = None
    affine = None
    mean_sum = None
    for f in input_files:
        img = load_nifti(str(f).split("[")[0])
        d = np.asarray(img.dataobj, dtype=np.float32)
        pos = np.all(d > 0, axis=-1)
        coverage = pos if coverage is None else coverage & pos
        m = d.mean(-1)
        mean_sum = m if mean_sum is None else mean_sum + m
        affine = img.affine
        del d
    assert coverage is not None and affine is not None and mean_sum is not None
    shape = coverage.shape
    voxel_mm = np.sqrt((affine[:3, :3] ** 2).sum(0))
    if verbose:
        print(f"  Coverage: {int(coverage.sum()):,} voxels positive in all {len(input_files)} runs")

    # -- units: ribbon patches, grown ribbon, tiles
    try:
        surfaces = resolve_ribbon_surfaces(
            args.fs_subj,
            args.white,
            args.pial,
            args.white_mesh,
            args.pial_mesh,
            args.cortex_label,
            args.hemi,
            log=print if verbose else None,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    patches = tdv.ribbon_patches(surfaces, affine, shape, coverage, args.patch_mm, device)
    depth_flat, n_units = patches.depth, patches.n_patches
    label = patches.label
    regions: list = []
    atlas = None
    if args.atlas:
        regions = _atlas_regions(args, surfaces, verbose)
        atlas = tdv.atlas_labels(
            patches, [r.members for r in regions], args.atlas_unit, keep_rest=not args.atlas_only
        )
        label, n_units = atlas.label, atlas.n_units
        if verbose:
            n_in = int(((atlas.label >= 0) & (atlas.label < atlas.n_roi_units)).sum())
            print(
                f"  Atlas: {len(regions)} ROI(s) -> {atlas.n_roi_units:,} unit(s) "
                f"({args.atlas_unit}); {n_in:,} covered ribbon voxels in them"
                + (
                    f"; {atlas.n_overlap:,} claimed by >1 ROI (first wins)"
                    if atlas.n_overlap
                    else ""
                )
            )
    n_patches = n_units
    label_vol = tdv.extend_labels(label.reshape(shape), voxel_mm, args.extend_mm)
    label_vol[~coverage] = -1
    label = label_vol.reshape(-1)
    n_ribbon = int((label >= 0).sum())
    if args.tile_mm > 0 and not args.atlas_only:
        # Coverage alone is the whole field of view in data that is positive
        # outside the head; tiles are for brain, so the brain is asked for.
        if args.tile_mask:
            brain = np.asarray(load_nifti(args.tile_mask).dataobj).reshape(shape) > 0
        else:
            from fastfuncstuff.processing.mask import automask

            brain = automask(torch.from_numpy(mean_sum)).numpy()
        rest = np.flatnonzero(coverage.reshape(-1) & brain.reshape(-1) & (label < 0))
        if rest.size:
            ijk = np.array(np.unravel_index(rest, shape)).T
            label[rest] = tdv.volume_tiles(ijk, voxel_mm, args.tile_mm) + n_units
    analysed = label >= 0
    # Units numbered densely (a patch can end up with no covered voxel).
    used, unit_of = np.unique(label[analysed], return_inverse=True)
    unit_kind = (used >= n_patches).astype(np.int8)
    if verbose:
        print(
            f"  Units: {int((unit_kind == 0).sum()):,} cortical patches ({n_ribbon:,} voxels, "
            f"ribbon + {args.extend_mm:g} mm), {int((unit_kind == 1).sum()):,} tiles "
            f"({int(analysed.sum()) - n_ribbon:,} voxels)"
        )

    # -- data, design pieces, nuisance
    load = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_array=analysed.reshape(shape),
        do_scale=True,
        device=device,
        force_cpu=True,
        verbose=verbose,
    )
    data, run_starts, n_timepoints, tr = load.data, load.run_starts, load.n_timepoints, load.tr
    del load
    dt = resolve_microtime_dt(tr, args.microtime_dt)
    try:
        offset = resolve_microtime_offset(args.microtime_offset, input_files, tr)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    onset_bin = microtime_offset_bins(offset, tr, dt)
    all_onsets = apply_onset_rounding(args, timing.all_onsets, tr, offset, verbose=verbose)
    durations = list(timing.durations)

    polort = resolve_polort(args.polort, compute_run_lengths(run_starts, n_timepoints), tr)
    blocks = collect_nuisance_blocks(args, run_starts, n_timepoints, verbose=verbose)
    nuisance = build_nuisance_per_run(
        run_starts=run_starts,
        n_timepoints=n_timepoints,
        polort=polort,
        device=device,
        blocks=blocks,
    )
    q_runs = tdv.orthonormal_nuisance(nuisance, device)
    n_nuisance = sum(q.shape[1] for q in q_runs)

    def design_for(curves: np.ndarray) -> torch.Tensor:
        return build_task_design(
            torch.as_tensor(np.atleast_2d(curves), dtype=torch.float32, device=device),
            n_timepoints,
            run_starts,
            tr=tr,
            microtime_dt=dt,
            microtime_onset=onset_bin,
            event_onsets=all_onsets,
            durations=durations,
            device=device,
        )

    def project(x: torch.Tensor) -> torch.Tensor:
        return tdv.project_runs(x.to(device=device, dtype=torch.float64), q_runs, run_starts)

    # Data projected once, kept (T, V) on the host; the GPU sees it in chunks.
    V = data.shape[0]
    y = torch.empty(n_timepoints, V, dtype=torch.float32)
    step = max(1, int(2e8 // (n_timepoints * 8)))
    for a in range(0, V, step):
        y[:, a : a + step] = project(data[a : a + step].T).float().cpu()
    del data

    lib = get_hrf_library(
        mode=args.hrf_mode,
        stim_duration=0.0,
        microtime_dt=dt,
        n_hrfs=args.n_hrfs,
        device=torch.device("cpu"),
        library_path=args.hrf_library,
    )
    shapes = np.atleast_2d(lib.cpu().numpy().astype(np.float64))
    shapes /= np.abs(shapes).max(1, keepdims=True).clip(1e-12, None)
    canon, _ = build_shape_library("canonical", dt, 32.0)
    grid = tdv.LagGrid.make(args.max_lag, args.lag_step, args.widths)
    if args.screen == "library":
        designs = project(tdv.as_bank(design_for(shapes), shapes.shape[0]).to(device))
        responsive, screen_r2, positive, _ = tdv.screen_library(
            y, designs, n_nuisance, args.p_thresh, device
        )
        del designs
    else:
        sign_cols = None
        if args.screen == "bank":
            k = args.screen_k
            canon = tdv.screen_basis(canon[0], dt, grid, k)
            sign_cols = np.arange(len(timing.condition_labels)) * k
        responsive, screen_r2, positive = tdv.screen_responsive(
            y, project(design_for(canon)), n_nuisance, args.p_thresh, device, sign_cols=sign_cols
        )
    unit_weight = {
        "responsive": None,
        "weighted": np.where(positive, screen_r2, 0.0),
        "all": np.ones(screen_r2.shape),
    }[args.unit_mean]
    if verbose:
        print(
            f"  Responsive ({args.screen} F p < {args.p_thresh:g}, positive): {int(responsive.sum()):,}"
        )

    res = tdv.fit_hrftdv(
        y,
        unit_of,
        unit_kind,
        shapes,
        design_for,
        project,
        responsive,
        screen_r2,
        grid,
        dt,
        percentiles=(args.percentiles[0], args.percentiles[1]),
        min_voxels=args.min_voxels,
        lag_step=args.pair_step,
        unit_weight=unit_weight,
        device=device,
        verbose=verbose,
    )

    roi_depth = None
    if args.roi_lag or args.pair_from == "depth":
        roi_depth = _fit_roi_depth(
            args, surfaces, affine, shape, analysed, y, unit_of, res, shapes, design_for,
            project, grid, dt, run_starts, device, verbose,
        )  # fmt: skip
        if args.pair_from == "depth":
            res = _pairs_from_depth(roi_depth, res, shapes, dt, args.pair_step, grid, verbose)

    # -- outputs
    vox = np.flatnonzero(analysed)

    def vol(values, fill=0.0) -> np.ndarray:
        out = np.full(int(np.prod(shape)), fill, np.float32)
        out[vox] = values
        return out.reshape(shape)

    entry = res.voxel_entry + 1
    save_nifti(
        np.stack([vol(entry), vol(np.nan_to_num(res.fit.r2))], -1),
        f"{pfx.stem}_hrf_index{ext}",
        affine=affine,
        brick_labels=["pair_entry", "lag_fit_r2"],
    )
    pair_lo = np.array([p.early_lag for p in res.pairs])[unit_of]
    pair_hi = np.array([p.late_lag for p in res.pairs])[unit_of]
    bricks = {
        "lag": res.fit.lag,
        "rel_lag": np.nan_to_num(res.rel_lag),
        "width": res.fit.width,
        "r2": res.fit.r2,
        "at_bound": res.fit.at_bound.astype(np.float32),
        "responsive": res.responsive.astype(np.float32),
        "unit": unit_of + 1,
        "unit_early_lag": pair_lo,
        "unit_late_lag": pair_hi,
    }
    stack = [vol(v) for v in bricks.values()] + [np.nan_to_num(depth_flat).reshape(shape)]
    save_nifti(
        np.stack(stack, -1),
        f"{pfx.stem}_tdv{ext}",
        affine=affine,
        brick_labels=[*bricks, "depth"],
    )
    mode, blend_extra = parse_hrf_mode(args.hrf_mode)
    n_lib = {"library": shapes.shape[0], "pighs": 0}.get(
        mode, shapes.shape[0] - (blend_extra or BLEND_DEFAULT_EXTRA)
    )
    labels = [f"lib{i:02d}" for i in range(n_lib)] + [
        f"pighs{i:02d}" for i in range(shapes.shape[0] - n_lib)
    ]
    table = res.library.table(labels)
    n_units_per = np.bincount(res.library.index, minlength=len(table))
    n_vox_per = np.bincount(res.voxel_entry, minlength=len(table))
    with open(f"{pfx.stem}_pairs.tsv", "w") as fh:
        cols = ["entry", "shape", "early_lag", "late_lag", "early_width", "late_width"]
        fh.write("\t".join([*cols, "n_units", "n_voxels"]) + "\n")
        for row, nu, nv in zip(table, n_units_per, n_vox_per, strict=True):
            fh.write("\t".join(str(row[c]) for c in cols) + f"\t{nu}\t{nv}\n")
    if atlas is not None:
        # Units are numbered densely over every label in use; tiles sit past the atlas's.
        roi_of_unit = np.full(used.size, -1, np.int64)
        cortical = used < atlas.n_units
        roi_of_unit[cortical] = atlas.unit_roi[used[cortical]]
        path = _write_roi_table(
            f"{pfx.stem}_roi.tsv", regions, surfaces, roi_of_unit, unit_of, res, shapes, labels, dt
        )
        if verbose:
            print(f"  Wrote {path}")
    if roi_depth is not None:
        for path in _write_roi_depth(f"{pfx.stem}_roi_depth", roi_depth, res, shapes, dt):
            if verbose:
                print(f"  Wrote {path}")
    meta = {
        "tool": "ffs_hrftdv",
        "microtime_dt": dt,
        "microtime_offset": offset,
        "basis_labels": ["early", "late"],
        "canonical_mode": "spmg1",
        "hrf_mode": args.hrf_mode,
        "input_files": [str(f) for f in input_files],
        "event_files": args.events,
        "spec": args.spec,
        "condition_labels": timing.condition_labels,
        "durations": durations,
        "max_lag": args.max_lag,
        "lag_step": args.lag_step,
        "widths": list(args.widths),
        "percentiles": list(args.percentiles),
        "pair_step": args.pair_step,
        "patch_mm": args.patch_mm,
        "extend_mm": args.extend_mm,
        "tile_mm": args.tile_mm,
        "atlas": args.atlas,
        "atlas_unit": args.atlas_unit if args.atlas else None,
        "roi_lag": bool(args.roi_lag),
        "pair_from": args.pair_from,
        "depths": list(args.depths) if roi_depth is not None else None,
        "roi_claim": args.roi_claim if args.atlas else None,
        "p_thresh": args.p_thresh,
        "screen": args.screen,
        "screen_k": args.screen_k if args.screen == "bank" else None,
        "unit_mean": args.unit_mean,
        "n_responsive": int(responsive.sum()),
        "n_units": int(unit_kind.size),
        "n_patches": int((unit_kind == 0).sum()),
        "n_fallback_units": res.n_fallback,
        "entries": table,
    }
    torch.save(
        {
            "hrf_library": torch.as_tensor(res.library.curves),
            "hrf_index": torch.as_tensor(res.voxel_entry),
            "metadata": meta,
            "duration_convolved": False,
            # The unit curves the pairs are shifted from, for plots of them.
            "unit_shapes": torch.as_tensor(shapes, dtype=torch.float32),
        },
        f"{pfx.stem}_hrf_library.pt",
    )
    Path(f"{pfx.stem}_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if roi_depth is not None:
        for path in _plot_best_roi(pfx.stem, roi_depth):
            if verbose:
                print(f"  Wrote {path}  (more ROIs: python -m fastfuncstuff.cli.hrftdv_plot)")

    dsort_files: list[str] = []
    if args.pooled_late:
        pattern = tdv.pooled_late_patterns(
            y, unit_of, responsive, res.library, design_for, project, device
        )
        regs = tdv.late_regressors(res.library, pattern, design_for)  # (T, U)
        early_meta = {**meta, "pooled_late": True, "basis_labels": ["early"]}
        torch.save(
            {
                "hrf_library": torch.as_tensor(res.library.curves[:, 0]),
                "hrf_index": torch.as_tensor(res.voxel_entry),
                "metadata": early_meta,
                "duration_convolved": False,
            },
            f"{pfx.stem}_early_hrf_library.pt",
        )
        save_nifti(
            np.stack([vol(entry), vol(np.nan_to_num(res.fit.r2))], -1),
            f"{pfx.stem}_early_hrf_index{ext}",
            affine=affine,
            brick_labels=["early_entry", "lag_fit_r2"],
        )
        bounds = [*run_starts, n_timepoints]
        for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            # Voxels outside the units keep a zero column; reml's per-voxel GLS
            # stays finite with it (tests/test_hrftdv_cli.py).
            out = np.zeros((int(np.prod(shape)), b - a), np.float32)
            out[vox] = regs[a:b, unit_of].T
            path = f"{pfx.stem}_late_dsort.run-{r + 1:02d}{ext}"
            save_nifti(out.reshape(*shape, b - a), path, affine=affine, tr=tr)
            dsort_files.append(path)
            del out
        names = timing.condition_labels
        med = np.median(pattern[unit_kind == 0], axis=0) if (unit_kind == 0).any() else pattern[0]
        if verbose:
            print(
                "  Pooled late: median cortical pattern "
                + ", ".join(f"{n} {w:+.2f}" for n, w in zip(names, med, strict=False))
            )

    if verbose:
        print()
        print(
            f"  Library: {len(table)} early/late entries for {unit_kind.size:,} units "
            f"({res.n_fallback:,} on the median pair: too few responsive voxels)"
        )
        sep = pair_hi - pair_lo
        ribbon_resp = res.responsive & (unit_kind[unit_of] == 0)
        if ribbon_resp.any():
            print(
                f"  Early-late separation in responsive ribbon voxels: median {np.median(sep[ribbon_resp]):.2f} s"
            )
            d = depth_flat[vox]
            m = ribbon_resp & np.isfinite(d) & ~res.fit.at_bound
            if m.sum() > 50:
                print("  rel_lag by depth (responsive ribbon voxels; 0 = white, 1 = pial):")
                for a in np.linspace(0, 0.8, 5):
                    k = m & (d >= a) & (d < a + 0.2 + 1e-9)
                    if k.any():
                        print(
                            f"    {a:.1f}-{a + 0.2:.1f}  n={int(k.sum()):6d}  median {np.median(res.rel_lag[k]):+.3f} s"
                        )
        print(f"  Wrote {pfx.stem}_hrf_index{ext}, _hrf_library.pt, _tdv{ext}, _pairs.tsv")
        print(f"  Next: ffs_reml ... -hrf_opt {pfx.stem}  ({time.time() - t0:.0f} s)")
        if dsort_files:
            print(
                f"     or: ffs_reml ... -hrf_opt {pfx.stem}_early -dsort {' '.join(dsort_files)} "
                "-dsort_betas yes ..."
            )


if __name__ == "__main__":
    main()
