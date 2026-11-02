"""CLI: the laminar model over overlapping cortical columns, painted back to cortex.

Command: ffs_laminascope

Usage:
    # voxel curves once (a linear estimator, so column means of them are exact)
    ffs_deconvolve -input run*.nii.gz -events ... -event_prefix task -model TENT \
        -window 0 14 -tent-smooth global -mask ribbon.nii.gz -prefix curves/sub
    ffs_laminascope -iresp_prefix curves/sub -window 0 14 -duration 1 \
        -spec comparisons.toml -fs_subj $SUBJ -responsive faces_t.nii.gz \
        -responsive_thresh 4 -depths 7 9 11 -equivolume -device cuda -prefix scope/sub
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, add_device_arg

_DESCRIPTION = """\
Run the laminar BOLD model (Havlicek & Uludag; the analysis of Faes et al. 2026, as
ffs_laminar runs it per ROI) over cortical columns made on the fly, and paint what
each column found back onto the surface and the ribbon.

Columns are geodesic disks of -column_radius on the midthickness, centred so that
every responsive vertex (-responsive) is within -column_spacing of a centre: the
responsive area sets how many there are. With the default spacing (half the radius)
an interior vertex sits in several columns that differ only in where their edges
fall, and the maps report their consensus and how far they agree.

Each column's depth curves are read from VOXEL response curves (-iresp_prefix: the
{prefix}_iresp_<label> files of ffs_deconvolve, in percent signal change). Fit them
with a linear estimator -- -tent-smooth global (one lambda), or plain TENT -- and a
column's curve equals the fit of the column's mean, so fitting voxels once and
sampling after loses nothing. Per-voxel lambda (-tent-smooth reml) is not linear and
biases weak voxels flat.

Each -spec comparison is one model space at every column: baseline and test
conditions concatenated with masked padding, every layer hypothesis inverted at
every -depths K, averaged across K, compared by free energy. Columns run batched
(-device cuda pays from ~100 columns; it saturates near 1,000 per batch).

Writes, per comparison NAME:
  {prefix}_NAME_columns.tsv   one line per column: centre, best model, P(model),
                              P(depth modulated), B per depth (sd, P != 0), R2, flags
  {prefix}_NAME.<hemi>.func.gii  per vertex: P(depth modulated) per depth, B per
                              depth, consensus winner and its agreement, columns covering
  {prefix}_NAME_ribbon.nii.gz the same maps on the -iresp grid, through each ribbon
                              voxel's owning vertex
and {prefix}_columns.npz (columns, their curves and PSFs, every fit), {prefix}.json.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from fastfuncstuff.surface.ribbon_args import add_depth_args, add_ribbon_args

    p = FfsArgumentParser(prog="ffs_laminascope", description=_DESCRIPTION)
    g = p.add_argument_group("Curves and comparisons")
    g.add_argument("-iresp_prefix", required=True, metavar="PREFIX",
                   help="PREFIX of voxel curve files PREFIX_iresp_<label>.nii[.gz|.zst].")  # fmt: skip
    g.add_argument("-window", type=float, nargs=2, required=True, metavar=("BOT", "TOP"),
                   help="Seconds after onset of the first and last curve sample (deconvolve's -window).")  # fmt: skip
    g.add_argument("-duration", type=float, required=True, metavar="SEC",
                   help="Duration of the events the curves respond to (s).")  # fmt: skip
    g.add_argument("-spec", default=None, metavar="TOML",
                   help="Comparisons to fit: [[comparison]] tables with name, baseline, test "
                   "(a label, a glob or a list, averaged) and optional models.")  # fmt: skip
    g.add_argument("-compare", nargs=3, action="append", default=None,
                   metavar=("NAME", "BASELINE", "TEST"),
                   help="One comparison on the command line (globs allowed); repeatable.")  # fmt: skip
    g.add_argument("-mask", default=None, metavar="FILE",
                   help="Voxels the reads may use (the mask the curves were fitted in).")  # fmt: skip

    c = p.add_argument_group("Columns")
    add_ribbon_args(c)
    c.add_argument("-responsive", default=None, metavar="FILE",
                   help="Volume marking responsive cortex: a mask, or a statistic with "
                   "-responsive_thresh. A column centre goes wherever it marks the ribbon at "
                   "any depth. Select on something independent of the comparisons (an omnibus "
                   "F, a localiser) -- never on the difference being tested.")  # fmt: skip
    c.add_argument("-responsive_thresh", type=float, default=None, metavar="T",
                   help="Threshold for a statistic -responsive (|value| >= T).")  # fmt: skip
    c.add_argument("-responsive_positive", action="store_true",
                   help="Threshold signed (value >= T) instead of |value|.")  # fmt: skip
    c.add_argument("-within", default=None, metavar="FILE",
                   help="Only centre columns inside this region (label, annot or volume ROI): "
                   "for trying a few places before the whole cortex.")  # fmt: skip
    c.add_argument("-regions_from", default=None, metavar="MAP",
                   help="Instead of disks: one region per connected surface cluster of this "
                   "statistic above -regions_thresh (e.g. faces > objects from a localiser), "
                   "closed along the mesh by 1 mm, at least -regions_min_mm2. The data's own "
                   "patches, often 8-10 mm or more across. Select on independent runs.")  # fmt: skip
    c.add_argument("-regions_atlas", nargs="+", default=None, metavar="FILE",
                   help="Instead of disks: one region per atlas label (.annot, .label, GIfTI "
                   "label map, or a volume atlas claimed along the column), named. The way "
                   "to have the same regions in every subject for group tests.")  # fmt: skip
    c.add_argument("-regions_thresh", type=float, default=3.1, metavar="T",
                   help="Threshold for -regions_from (value >= T; |value| with -regions_abs).")  # fmt: skip
    c.add_argument("-regions_abs", action="store_true",
                   help="Cluster |value| >= T, so negative patches (objects > faces) count too.")  # fmt: skip
    c.add_argument("-regions_min_mm2", type=float, default=20.0, metavar="MM2",
                   help="Smallest cluster kept as a region (midthickness area).")  # fmt: skip
    c.add_argument("-column_radius", type=float, default=3.0, metavar="MM",
                   help="Geodesic radius of a column on the midthickness.")  # fmt: skip
    c.add_argument("-column_spacing", type=float, default=None, metavar="MM",
                   help="Every responsive vertex is within this of a centre (default: half the "
                   "radius, so interior vertices sit in several columns).")  # fmt: skip
    c.add_argument("-min_coverage", type=float, default=0.99, metavar="FRAC",
                   help="Skip columns whose depth bins were read less than this inside the "
                   "data (FOV edge, mask): the model cannot fit a cut profile.")  # fmt: skip
    c.add_argument("-max_columns", type=int, default=None, metavar="N",
                   help="Fit a random N of the columns (a quick look).")  # fmt: skip
    c.add_argument("-seed", type=int, default=0, help="Seed of the cover order and -max_columns.")
    d = p.add_argument_group("Depth sampling")
    add_depth_args(d, ribbon=False)

    m = p.add_argument_group("Model (as ffs_laminar)")
    m.add_argument("-models", nargs="+", default=None, metavar="NAME",
                   help="Subset of the model space for every comparison (a comparison's own "
                   "'models' wins).")  # fmt: skip
    m.add_argument("-N", type=int, default=3, help="Neuronal depths.")
    m.add_argument("-modulation", nargs="+", default=["event"], metavar="WHEN",
                   help="When the test condition's modulation is on: 'event' (default, the whole "
                   "event), 'reference' (Faes et al.'s 0.1 s violation pulse, parity only) or "
                   "START DUR seconds after onset.")  # fmt: skip
    m.add_argument("-psf", choices=("measured", "none"), default="measured",
                   help="Depth PSF: each column's exact sampling matrix, or none.")  # fmt: skip
    m.add_argument("-free_energy", choices=("reference", "sum", "mean"), default="reference")
    m.add_argument("-TE", type=float, default=0.028, help="Echo time (s).")
    m.add_argument("-B0", type=float, default=7.0, help="Field strength (T).")
    m.add_argument("-max_iter", type=int, default=128, help="Variational Laplace iterations.")
    m.add_argument("-model_space", choices=("reduce", "invert"), default="reduce",
                   help="'reduce' (default): invert only the full model (every depth "
                   "modulated) and derive the other layer hypotheses by Bayesian model "
                   "reduction (Friston & Penny 2011) -- 2^N-fold fewer inversions, and nested "
                   "evidences consistent by construction (brute force left ~23%% of real "
                   "columns with a model far below one it contains). On the published data it "
                   "ranks all models as brute force does, the competitive ones within 1.3-1.6 "
                   "nats. 'invert': every model separately, as Faes et al.")  # fmt: skip
    m.add_argument("-vascular_prior", choices=("none", "local", "global"), default="none",
                   help="Empirical Bayes on the vascular parameters (s_d, al_d), with -model_space "
                   "reduce: each column's prior is centred on its neighbours' posteriors "
                   "(columns within -vascular_radius; 'global': the whole hemisphere) with the "
                   "spread that maximises the summed evidence (Friston et al. 2016). The "
                   "vasculature belongs to the tissue, not to one column, and a column's own "
                   "s_d otherwise drifts low -- the direction that gives wrong profiles.")  # fmt: skip
    m.add_argument("-vascular_from", nargs=2, default=None, metavar=("NPZ", "COMPARISON"),
                   help="Each column's prior on the vascular parameters (s_d, al_d) from another "
                   "ffs_laminascope run's posterior: its PREFIX_columns.npz and comparison name. "
                   "Columns are matched by centre vertex, so build both runs with the same "
                   "-responsive, -column_radius/-column_spacing and -seed. Use a high-SNR "
                   "localiser on separate runs: the vasculature belongs to the tissue, not "
                   "the task. Unmatched columns keep the original prior. Needs -model_space "
                   "reduce; replaces -vascular_prior.")  # fmt: skip
    m.add_argument("-vascular_radius", type=float, default=10.0, metavar="MM",
                   help="Neighbourhood of -vascular_prior local (column centres, mm).")  # fmt: skip
    m.add_argument("-rescue", type=int, default=2, metavar="N",
                   help="Restart fits that stalled (ended on rejected steps) up to N times from "
                   "their best point. On real columns half stalled somewhere without it, and a "
                   "stalled model can score hundreds of nats below a model it contains. 0 is "
                   "the published scheme (ffs_laminar's default).")  # fmt: skip
    m.add_argument("-pad_rows", type=int, default=20, metavar="N")
    m.add_argument("-pad_sd", type=float, default=0.1, metavar="PCT")
    m.add_argument("-pad_seed", type=int, default=20260918, metavar="N")
    m.add_argument("-hyperprior", choices=("reference", "rows"), default="reference")

    o = p.add_argument_group("Compute and output")
    add_device_arg(o)
    o.add_argument("-chunk", type=int, default=None, metavar="N",
                   help="Columns per batched inversion (default 1024 on a GPU, 64 on CPU).")  # fmt: skip
    o.add_argument("-prefix", required=True, help="Output prefix.")
    o.add_argument("-subject", default=None, metavar="ID",
                   help="Subject label written into the long tables (default: the prefix's "
                   "file name), so subjects' tables concatenate for group tests.")  # fmt: skip
    o.add_argument("-verb", type=int, default=1, help="0 quiet, 1 progress.")
    return p.parse_args(argv)


def _iresp_labels(prefix: str) -> dict[str, str]:
    import glob
    import re

    out = {}
    for path in sorted(glob.glob(f"{glob.escape(prefix)}_iresp_*")):
        m = re.match(re.escape(prefix) + r"_iresp_(.+?)\.nii(\.gz|\.zst)?$", path)
        if m and not m.group(1).endswith("_unconstrained"):
            out[m.group(1)] = path
    if not out:
        raise ValueError(f"no {prefix}_iresp_<label>.nii* files")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    try:
        return _run(args, t_start)
    except (OSError, ValueError) as exc:
        print(f"ffs_laminascope: {exc}", file=sys.stderr)
        return 1


def _run(args, t_start) -> int:  # noqa: C901 -- a pipeline, read top to bottom
    import json

    import torch

    from fastfuncstuff.cli.laminar import parse_modulation
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.laminar import columns as col
    from fastfuncstuff.laminar.comparisons import make_comparison, read_comparisons
    from fastfuncstuff.laminar.experiment import layer_model_names, layer_model_targets
    from fastfuncstuff.laminar.io import two_condition_batch
    from fastfuncstuff.laminar.pipeline import depth_names, reference_p0
    from fastfuncstuff.laminar.scope import fit_columns
    from fastfuncstuff.processing.roi_depth import (
        CLAIM_ANY,
        depth_list,
        options_from_args,
        read_depth_rois,
    )
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    say = print if args.verb >= 1 else (lambda *_a, **_k: None)
    modulation = parse_modulation(args.modulation)
    device = setup_device(args.device, float64_only="ffs_laminascope")

    files = _iresp_labels(args.iresp_prefix)
    labels = list(files)
    comps = read_comparisons(args.spec, labels) if args.spec else []
    for name, base, test in args.compare or []:
        comps.append(make_comparison(name, base, test, labels))
    if not comps:
        raise ValueError("give -spec or -compare")
    all_names, all_targets = layer_model_names(args.N), layer_model_targets(args.N)
    for cmp in comps:
        cmp.models = cmp.models or args.models
        bad = [mm for mm in (cmp.models or []) if mm not in all_names]
        if bad:
            raise ValueError(
                f"comparison {cmp.name}: unknown model(s) {bad}; the space is {all_names}"
            )
    used = sorted({lab for c in comps for lab in c.baseline + c.test}, key=labels.index)

    ref = load_nifti(files[used[0]])
    affine = np.asarray(ref.affine, np.float64)
    shape3 = tuple(int(n) for n in ref.shape[:3])
    n_t = int(ref.shape[3]) if len(ref.shape) > 3 else 1
    times = np.linspace(args.window[0], args.window[1], n_t)
    mask = None
    if args.mask:
        mask = np.asanyarray(load_nifti(args.mask).dataobj).reshape(shape3) != 0

    logs: list[str] = []
    surfaces = resolve_ribbon_surfaces(
        args.fs_subj, args.white, args.pial, args.white_mesh, args.pial_mesh,
        args.cortex_label, hemi=args.hemi, log=logs.append,
    )  # fmt: skip
    if args.regions_atlas and args.regions_from:
        raise ValueError("give -regions_atlas or -regions_from, not both")
    if args.regions_atlas:
        columns = col.regions_from_atlas(surfaces, args.regions_atlas)
        say(f"Regions: {columns.n_columns} atlas region(s) from {', '.join(args.regions_atlas)}")
    elif args.regions_from:
        t0 = time.time()
        columns = col.regions_from_map(surfaces, args.regions_from, args.regions_thresh,
                                       absolute=args.regions_abs, min_area_mm2=args.regions_min_mm2)  # fmt: skip
        sizes = np.asarray(columns.members.sum(axis=1)).ravel()
        say(f"Regions: {columns.n_columns} cluster(s) of {args.regions_from} >= {args.regions_thresh:g}, "
            f"{int(np.median(sizes)):,} vertices median ({time.time() - t0:.0f} s)")  # fmt: skip
    else:
        if not args.responsive:
            raise ValueError(
                "give -responsive (columns centred on responsive cortex) or -regions_from"
            )
        resp = col.responsive_vertices(args.responsive, surfaces, threshold=args.responsive_thresh,
                                       absolute=not args.responsive_positive)  # fmt: skip
        if args.within:
            notes: list[str] = []
            inside = [np.zeros(len(s.white), bool) for s in surfaces]
            for r in read_depth_rois([args.within], surfaces, notes, CLAIM_ANY):
                for h, mbr in r.members.items():
                    inside[h] |= mbr
            resp = [a & b for a, b in zip(resp, inside, strict=True)]
        say(
            "Responsive vertices: "
            + ", ".join(f"{s.name} {int(r.sum()):,}" for s, r in zip(surfaces, resp, strict=True))
        )
        t0 = time.time()
        columns = col.make_columns(surfaces, resp, radius_mm=args.column_radius,
                                   spacing_mm=args.column_spacing, seed=args.seed)  # fmt: skip
        cover = columns.coverage_count()
        say(f"Columns: {columns.n_columns:,} (radius {columns.radius_mm:g} mm, spacing "
            f"{columns.spacing_mm:g} mm) over {columns.vertex_id.size:,} vertices; columns per "
            f"vertex median {np.median(cover):.0f} ({time.time() - t0:.0f} s)")  # fmt: skip

    ks = depth_list(args)
    opts = options_from_args(args)
    t0 = time.time()
    reads = col.vertex_depth_reads(surfaces, columns, affine, shape3, ks, opts, mask=mask,
                                   psf=args.psf == "measured", device=device)  # fmt: skip
    say(f"Depth reads: K = {ks}, {reads.voxels.size:,} voxels ({time.time() - t0:.0f} s)")
    in_fov = np.ones(columns.n_columns)
    if args.regions_atlas or args.regions_from:
        columns, in_fov = col.trim_to_coverage(columns, reads, args.min_coverage)
        cut = in_fov < 0.999
        if cut.any():
            say("Regions trimmed to the FOV/mask: " + ", ".join(
                f"{columns.names[c] if columns.names else c} {100 * in_fov[c]:.0f}%"
                for c in np.flatnonzero(cut)))  # fmt: skip

    # Voxel curves of every label used, on the voxels the reads touch.
    curves_vox = {}
    for lab in used:
        img = load_nifti(files[lab])
        arr = np.asanyarray(img.dataobj)
        if tuple(arr.shape[:3]) != shape3 or (arr.shape[3] if arr.ndim > 3 else 1) != n_t:
            raise ValueError(f"{files[lab]}: grid or curve length differs from {files[used[0]]}")
        curves_vox[lab] = np.asarray(arr.reshape(-1, n_t)[reads.voxels], np.float64)
        del arr, img

    col_curves = {}  # K -> label -> (C, K, T)
    coverage = {}
    psfs = {}
    for k in ks:
        stack = np.stack([curves_vox[lab] for lab in used], axis=-1)  # (V, T, L)
        cc = col.column_curves(reads, columns, stack, k)
        col_curves[k] = {lab: cc.values[..., i] for i, lab in enumerate(used)}
        coverage[k], psfs[k] = cc.coverage, cc.psf
    cov_ok = np.all([coverage[k].min(1) >= args.min_coverage for k in ks], axis=0)
    finite = np.all(
        [np.isfinite(col_curves[k][lab]).all((1, 2)) for k in ks for lab in used], axis=0
    )
    if psfs[ks[0]] is not None:
        finite &= np.all([np.isfinite(psfs[k]).all((1, 2)) for k in ks], axis=0)
    fit_mask = cov_ok & finite
    if args.max_columns and fit_mask.sum() > args.max_columns:
        pick = np.random.default_rng(args.seed).choice(
            np.flatnonzero(fit_mask), args.max_columns, replace=False
        )
        fit_mask = np.zeros_like(fit_mask)
        fit_mask[pick] = True
    fit_idx = np.flatnonzero(fit_mask)
    say(f"Fitting {fit_idx.size:,} column(s); {int((~cov_ok).sum()):,} cut by the FOV/mask "
        f"(coverage < {args.min_coverage:g})")  # fmt: skip
    if fit_idx.size == 0:
        raise ValueError("no column has complete depth coverage")
    big = float(np.nanmax(np.abs(np.stack([col_curves[ks[0]][lab][fit_idx] for lab in used]))))
    if big > 50:
        raise ValueError(
            "curves look like data units, not percent change (fit without -dont_scale)"
        )

    chunk = args.chunk or (1024 if device.type == "cuda" else 64)
    vascular_given = None
    if args.vascular_from:
        vascular_given = _vascular_from(args.vascular_from, columns, fit_idx)
        say(
            f"Vascular priors from {args.vascular_from[0]} ({args.vascular_from[1]}): "
            f"{int(np.isfinite(vascular_given['s_d'][0]).sum()):,} of {fit_idx.size:,} columns matched"
        )
    centres = np.stack([
        0.5 * (np.asarray(surfaces[int(h)].white[v]) + np.asarray(surfaces[int(h)].pial[v]))
        for h, v in zip(columns.centre_surface, columns.centre_vertex, strict=True)
    ])  # fmt: skip
    results = {}
    out_npz: dict[str, np.ndarray] = {}
    for cmp in comps:
        names = cmp.models or all_names
        targets = [all_targets[all_names.index(mm)] for mm in names]
        data_for_k, kern_for_k = {}, {}
        for k in ks:
            base = np.mean([col_curves[k][lab][fit_idx] for lab in cmp.baseline], axis=0)
            test = np.mean([col_curves[k][lab][fit_idx] for lab in cmp.test], axis=0)
            data_for_k[k] = two_condition_batch(
                base.transpose(0, 2, 1), test.transpose(0, 2, 1), times,
                duration=args.duration, n_neuronal=args.N, gap=args.pad_rows,
                noise_sd=args.pad_sd, seed=args.pad_seed, delay_seconds=0.0,
                TE=args.TE, B0=args.B0, p0=reference_p0(), modulation=modulation,
            )  # fmt: skip
            if psfs[k] is not None:  # superficial first, as the model wants
                kern_for_k[k] = torch.as_tensor(psfs[k][fit_idx][:, ::-1, ::-1].copy())
            else:
                kern_for_k[k] = None
            out_npz[f"{cmp.name}_baseline_K{k}"] = base
            out_npz[f"{cmp.name}_test_K{k}"] = test
        say(f"\n{cmp.name}: {' + '.join(cmp.test)}  vs  {' + '.join(cmp.baseline)}  "
            f"({len(names)} models x K {ks} x {fit_idx.size:,} columns)")  # fmt: skip
        t0 = time.time()
        fits = fit_columns(data_for_k, kern_for_k, targets=targets, names=names,
                           free_energy=args.free_energy, max_iter=args.max_iter,
                           hyper_from_rows=args.hyperprior == "rows", device=device,
                           chunk=chunk, progress=args.verb >= 1, rescue=args.rescue,
                           model_space=args.model_space, vascular_prior=args.vascular_prior,
                           vascular_radius_mm=args.vascular_radius, centres=centres[fit_idx],
                           hemi=columns.centre_surface[fit_idx],
                           vascular_given=vascular_given)  # fmt: skip
        for k, ep in fits.empirical.items():
            if ep.shrink == 0.0:
                say(
                    f"  vascular prior K={k}: no pooled prior beats the columns' own "
                    f"(best summed dF {ep.grid_free_energy.max():.1f} nats) -- not applied"
                )
                continue
            say(f"  vascular prior K={k}: shrink {ep.shrink:.3g} of the prior variance; "
                f"s_d prior mean {np.percentile(ep.mean[:, 0], 5):.2f}..{np.percentile(ep.mean[:, 0], 95):.2f}")  # fmt: skip
        say(
            f"  fitted in {time.time() - t0:.0f} s ({(time.time() - t0) / fit_idx.size:.3f} s/column)"
        )
        results[cmp.name] = fits
        _write_comparison(args.prefix, cmp, fits, columns, fit_idx, surfaces, affine, shape3,
                          depth_names(args.N), device, say)  # fmt: skip
        k_mid = ks[len(ks) // 2]
        base_mid = np.mean([col_curves[k_mid][lab][fit_idx] for lab in cmp.baseline], axis=0)
        _write_long(args, cmp, fits, columns, fit_idx, surfaces, depth_names(args.N), say,
                    in_fov=in_fov[fit_idx], baseline_peak=base_mid.mean(1).max(1))  # fmt: skip
        for key in ("F", "p_model", "p_depth", "best", "B_mean", "B_sd", "B_p", "r2",
                    "stalled", "failed", "nesting", "s_d", "refit"):  # fmt: skip
            out_npz[f"{cmp.name}_{key}"] = getattr(fits, key)
        for name, (mean, sd) in fits.vascular.items():
            out_npz[f"{cmp.name}_post_{name}_mean"] = mean
            out_npz[f"{cmp.name}_post_{name}_sd"] = sd

    m = columns.members.tocsr()
    np.savez_compressed(
        f"{args.prefix}_columns.npz", centre_surface=columns.centre_surface,
        centre_vertex=columns.centre_vertex, members_data=m.data, members_indices=m.indices,
        members_indptr=m.indptr, members_shape=np.asarray(m.shape),
        vertex_surface=columns.vertex_surface, vertex_id=columns.vertex_id,
        surfaces=np.asarray([s.name for s in surfaces]), fitted=fit_idx, times=times,
        ks=np.asarray(ks), **{f"coverage_K{k}": coverage[k] for k in ks},
        **{f"psf_K{k}": psfs[k] for k in ks if psfs[k] is not None}, **out_npz,
    )  # fmt: skip
    with open(f"{args.prefix}.json", "w") as fh:
        json.dump({
            "tool": "ffs_laminascope", "argv": sys.argv, "iresp": {lab: files[lab] for lab in used},
            "comparisons": [vars(c) for c in comps], "K": ks, "columns": columns.n_columns,
            "fitted": int(fit_idx.size), "radius_mm": columns.radius_mm,
            "spacing_mm": columns.spacing_mm, "depth_method": opts.method, "psf": args.psf,
            "device": str(device), "modulation": modulation, "model_space": args.model_space, "vascular_prior": args.vascular_prior,
            "vascular_radius_mm": args.vascular_radius, "notes": logs + reads.notes,
            "timing_s": {c: r.timing for c, r in results.items()},
        }, fh, indent=2, default=str)  # fmt: skip
    say(f"\nffs_laminascope: done in {time.time() - t_start:.0f} s")
    return 0


def _vascular_from(spec, columns, fit_idx):
    """Per fitted column, (mean, variance) of s_d and al_d from another run's npz."""
    path, cmp = spec
    z = np.load(path)
    key = f"{cmp}_post_s_d_mean"
    if key not in z.files:
        raise ValueError(
            f"{path}: no posterior for comparison {cmp!r} (have: "
            + ", ".join(sorted({f.split("_post_")[0] for f in z.files if "_post_" in f}))
            + ")"
        )
    fitted = z["fitted"]
    other = {
        (int(z["centre_surface"][c]), int(z["centre_vertex"][c])): j for j, c in enumerate(fitted)
    }
    out = {}
    for name in ("s_d", "al_d"):
        mean_o, sd_o = z[f"{cmp}_post_{name}_mean"], z[f"{cmp}_post_{name}_sd"]
        mean = np.full(fit_idx.size, np.nan)
        var = np.full(fit_idx.size, np.nan)
        for i, c in enumerate(fit_idx):
            j = other.get((int(columns.centre_surface[c]), int(columns.centre_vertex[c])))
            if j is not None:
                mean[i], var[i] = mean_o[j], sd_o[j] ** 2
        out[name] = (mean, var)
    return out


def _write_long(args, cmp, fits, columns, fit_idx, surfaces, dnames, say, *, in_fov,
                baseline_peak) -> None:  # fmt: skip
    from pathlib import Path

    from fastfuncstuff.laminar.scope import write_long_table
    from fastfuncstuff.surface.mesh import vertex_areas

    areas = {}
    for h, s in enumerate(surfaces):
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        areas[h] = vertex_areas(mid, np.asarray(s.faces), mid.shape[0])
    m = columns.members.tocsr()
    n_vert, area, names, hemis = [], [], [], []
    for c in fit_idx:
        vi = m[c].indices
        n_vert.append(vi.size)
        area.append(
            float(
                sum(
                    areas[int(h)][v]
                    for h, v in zip(columns.vertex_surface[vi], columns.vertex_id[vi], strict=True)
                )
            )
        )
        hemis.append(surfaces[int(columns.centre_surface[c])].name)
        names.append(columns.names[c] if columns.names else f"{hemis[-1]}.region{c}")
    path = write_long_table(
        f"{args.prefix}_{cmp.name}_regions_long.tsv", fits,
        subject=args.subject or Path(args.prefix).name, comparison=cmp.name,
        region_ids=np.asarray(fit_idx), names=names, hemis=hemis,
        n_vertices=np.asarray(n_vert), area_mm2=np.asarray(area), depth_names=dnames,
        in_fov=in_fov, baseline_peak=baseline_peak,
    )  # fmt: skip
    say(f"  wrote {path}")


def _write_comparison(prefix, cmp, fits, columns, fit_idx, surfaces, affine, shape3,
                      dnames, device, say) -> None:  # fmt: skip
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data
    from fastfuncstuff.laminar import columns as col

    n_cols, n_d = columns.n_columns, fits.p_depth.shape[1]

    def full(a, fill=np.nan):
        out = np.full((n_cols, *a.shape[1:]), fill, dtype=np.float64)
        out[fit_idx] = a
        return out

    valid = np.zeros(n_cols, bool)
    valid[fit_idx] = fits.best >= 0
    best_full = np.full(n_cols, -1)
    best_full[fit_idx] = fits.best
    p_best = full(fits.p_model[np.arange(fit_idx.size), np.clip(fits.best, 0, None)])
    null = fits.models.index("null") if "null" in fits.models else None
    log_bf = (full(np.max(fits.F, 1) - fits.F[:, null]) if null is not None
              else np.full(n_cols, np.nan))  # fmt: skip
    per_column = np.concatenate(
        [full(fits.p_depth), full(fits.B_mean), full(fits.B_p), p_best[:, None],
         log_bf[:, None], full(fits.r2.max(1))[:, None]], axis=1,
    )  # fmt: skip
    names = ([f"p_{d}" for d in dnames] + [f"B_{d}" for d in dnames]
             + [f"PB_{d}" for d in dnames] + ["p_best", "logBF_vs_null", "r2"])  # fmt: skip
    vert, n_cov = col.paint_vertices(columns, per_column, valid)
    winner, agree = col.vertex_winner(columns, best_full, len(fits.models), valid)
    vmaps = np.concatenate([vert, winner[:, None].astype(float), agree[:, None], n_cov[:, None]], 1)
    vnames = names + ["winner", "agreement", "n_columns"]
    meta_base = {"tool": "ffs_laminascope", "comparison": cmp.name, "maps": ",".join(vnames),
                 "winner_models": ",".join(fits.models)}  # fmt: skip
    for h, s in enumerate(surfaces):
        if not np.any(columns.vertex_surface == h):
            continue
        data = col.full_surface(columns, vmaps, h, len(s.white))
        meta = dict(
            meta_base, hemi=s.name, mesh_fingerprint=mesh_fingerprint(s.faces, len(s.white))
        )
        path = f"{prefix}_{cmp.name}.{s.name}.func.gii"
        save_gifti_data(path, data, meta, time_series=False)
        say(f"  wrote {path}")
    vol = col.paint_ribbon(surfaces, columns, vmaps, affine, shape3, device=device)
    vol = np.nan_to_num(vol, nan=0.0).reshape(*shape3, -1).astype(np.float32)
    path = f"{prefix}_{cmp.name}_ribbon.nii.gz"
    save_nifti(vol, output_path=path, reference_img=None, affine=affine, brick_labels=vnames)
    say(f"  wrote {path}")

    tsv = f"{prefix}_{cmp.name}_columns.tsv"
    with open(tsv, "w") as fh:
        fh.write("column\themi\tcentre_vertex\tbest\tp_best\t"
                 + "\t".join(f"p_{mm}" for mm in fits.models) + "\t"
                 + "\t".join(f"p_mod_{d}\tB_{d}\tB_sd_{d}\tPB_{d}" for d in dnames)
                 + "\tr2_best\ts_d\tstalled\tfailed\tnesting_nats\n")  # fmt: skip
        for j, c in enumerate(fit_idx):
            b = int(fits.best[j])
            fh.write(
                f"{c}\t{surfaces[int(columns.centre_surface[c])].name}\t{columns.centre_vertex[c]}\t"
                f"{fits.models[b] if b >= 0 else 'none'}\t{fits.p_model[j, max(b, 0)]:.4g}\t"
                + "\t".join(f"{v:.4g}" for v in fits.p_model[j]) + "\t"
                + "\t".join(f"{fits.p_depth[j, i]:.4g}\t{fits.B_mean[j, i]:.4g}\t"
                            f"{fits.B_sd[j, i]:.4g}\t{fits.B_p[j, i]:.4g}" for i in range(n_d))
                + f"\t{fits.r2[j, max(b, 0)]:.4g}\t{fits.s_d[j]:.4g}\t{int(fits.stalled[j].any())}\t"
                f"{int(fits.failed[j].any())}\t{fits.nesting[j]:.3g}\n"
            )  # fmt: skip
    say(f"  wrote {tsv}")
    if n_bad := int((fits.nesting > 10).sum()):
        say(f"  ⚠️  {n_bad} column(s) with a model >10 nats below one it contains (stalled fits)")


if __name__ == "__main__":
    sys.exit(main())
