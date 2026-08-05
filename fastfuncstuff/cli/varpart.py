#!/usr/bin/env python
"""ffs_varpart - per-voxel variance partitioning for crossed factorial designs.

Consumes single-trial betas you already have (ffs_ridge, GLMsingle, anything) plus a
sidecar table describing each trial, and reports how much of the reliable response each
factor explains uniquely, what they share, and what lives in their interaction.

Method: ../fmri_wiki/concepts/Variance partitioning.md
Interface: ../fmri_wiki/software/ffs_varpart.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime

import numpy as np
import torch

try:
    from fastfuncstuff.cli_utils import parse_prefix
    from fastfuncstuff.design.trial_table import sanitize_levels
    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.stats.variance_partition import (
        build_roi_weights,
        collapse_to_rois,
        paint_rois_to_voxels,
        partition_variance,
        permutation_test,
    )
    from fastfuncstuff.utils import configure_torch_backends, get_device
except ImportError as e:  # pragma: no cover - install-time guard
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    sys.exit(1)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_varpart",
        description="Variance partitioning for fully crossed factorial designs",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # 21 tasks x 20 stimuli, 3 repeats, single-trial betas from ffs_ridge
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -mask brain.nii.gz -prefix vp

  # Parcel-level: far faster, and a higher noise ceiling resolves ranks that
  # per-voxel data cannot
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -atlas schaefer400.nii.gz -prefix vp_roi

  # With permutation inference (slow at voxel level; pair it with -atlas)
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -mask brain.nii.gz -perm 1000 -prefix vp

  # Restrict to a subset: drop the first run and one task entirely
  ffs_varpart -betas out_single_trial_betas.nii.gz \\
              -trials out_single_trial_events.tsv -factors trial_type,task \\
              -drop_trials run 01 -drop_trials task rest -prefix vp

Sidecar table: one row per volume of -betas, in the same order. Must contain the
columns named by -factors. Columns 'run', 'session' and 'repeat' are used when
present (fold construction and permutation blocks) and ignored when absent.
ffs_reml / ffs_ridge / ffs_denoise write exactly this table next to their
single-trial betas as {prefix}_single_trial_events.tsv when given BIDS -events.
Factor levels may be free text; they are sanitized into identifiers here (the
mapping back to the original labels is written to {prefix}_varpart.json).
""",
    )
    req = p.add_argument_group("Required")
    req.add_argument("-betas", required=True, help="4-D image; one volume per trial")
    req.add_argument("-trials", required=True, help="CSV/TSV sidecar, one row per volume")
    req.add_argument(
        "-factors",
        required=True,
        help="Comma-separated sidecar column names to partition over (exactly 2)",
    )
    req.add_argument("-prefix", required=True, help="Output prefix")

    opt = p.add_argument_group("Options")
    opt.add_argument(
        "-drop_trials",
        "-drop-trials",
        dest="drop_trials",
        nargs=2,
        action="append",
        metavar=("COLUMN", "LABEL"),
        default=None,
        help=(
            "Exclude every trial whose COLUMN equals LABEL, before anything else "
            "(e.g. -drop_trials run 01 -drop_trials task rest). Repeatable; each "
            "occurrence drops one COLUMN/LABEL pair. Numeric labels match regardless "
            "of zero padding and string labels regardless of case."
        ),
    )
    opt.add_argument("-mask", default=None, help="Restrict to voxels inside this mask")
    opt.add_argument(
        "-atlas",
        default=None,
        help=(
            "Collapse to ROIs before partitioning. 3-D integer label map (one ROI per "
            "non-zero value) or 4-D stack (one volume per ROI; binary or weighted, may "
            "overlap). Writes a per-ROI table plus a parcel-painted volume for figures."
        ),
    )
    opt.add_argument(
        "-max_rank",
        "-max-rank",
        dest="max_rank",
        type=int,
        default=None,
        help="Highest interaction rank to cross-validate (default: full rank)",
    )
    opt.add_argument(
        "-min_ncsnr_for_rank",
        "-min-ncsnr-for-rank",
        dest="min_ncsnr_for_rank",
        type=float,
        default=0.75,
        help=(
            "Noise-ceiling SNR below which interaction rank is reported as -1 "
            "(undetermined) instead of 0. Rank selection misses real structure long "
            "before it invents any, so without this low-SNR tissue reads as "
            "'task-invariant'. Set 0 to disable."
        ),
    )
    opt.add_argument(
        "-perm",
        type=int,
        default=0,
        help="Permutations for the Freedman-Lane null (0 = skip inference)",
    )
    opt.add_argument(
        "-perm_stats",
        "-perm-stats",
        dest="perm_stats",
        default="unique_a,unique_b,interaction",
        help="Which statistics to test; comma-separated",
    )
    opt.add_argument("-seed", type=int, default=0, help="RNG seed for permutations")
    opt.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto)")
    opt.add_argument("-quiet", action="store_true", help="Suppress progress bars")
    return p


def _read_table(path: str) -> list[dict]:
    """Read the trial sidecar, sniffing tab vs comma from the extension."""
    delim = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        raise SystemExit(f"❌ trial table is empty: {path}")
    return rows


def _column(rows: list[dict], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise SystemExit(
            f"❌ column '{name}' not found in trial table. Available: {sorted(rows[0])}"
        )
    return np.array([r[name] for r in rows])


def _optional_column(rows: list[dict], name: str) -> np.ndarray | None:
    return _column(rows, name) if name in rows[0] else None


def _label_matches(value: str, label: str) -> bool:
    """Lenient equality for -drop_trials.

    A BIDS run entity is written ``01`` in one table and ``1`` in the next, and a
    user typing ``-drop_trials run 1`` means the same run either way. Numeric
    values compare numerically; everything else compares case-insensitively on
    stripped text.
    """
    a, b = str(value).strip(), str(label).strip()
    if a.lower() == b.lower():
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _apply_drop_trials(rows: list[dict], drops: list[list[str]] | None) -> np.ndarray:
    """Return a boolean keep-mask over *rows* after applying every -drop_trials pair.

    A pair that matches nothing is an error, not a no-op: it is almost always a typo
    or the wrong column, and silently analysing the full dataset under the belief
    that a condition was excluded is the failure mode worth being loud about.
    """
    keep = np.ones(len(rows), dtype=bool)
    if not drops:
        return keep

    for column, label in drops:
        if column not in rows[0]:
            raise SystemExit(
                f"❌ -drop_trials: column '{column}' not found in trial table. "
                f"Available: {sorted(rows[0])}"
            )
        hit = np.array([_label_matches(r[column], label) for r in rows], dtype=bool)
        if not hit.any():
            values = sorted({str(r[column]) for r in rows})
            shown = values[:20] + (["..."] if len(values) > 20 else [])
            raise SystemExit(
                f"❌ -drop_trials {column} {label}: no trial has that value.\n"
                f"   Values present in '{column}': {', '.join(shown)}"
            )
        print(f"   ✂️  -drop_trials {column}={label}: dropping {int(hit.sum())} trials")
        keep &= ~hit

    if not keep.any():
        raise SystemExit("❌ -drop_trials removed every trial")
    return keep


def main() -> int:
    args = create_parser().parse_args()
    device = get_device(args.device)
    configure_torch_backends(device)

    print("=" * 70)
    print("ffs_varpart - variance partitioning")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"🖥️  Device: {device}")

    factor_names = [f.strip() for f in args.factors.replace(" ", ",").split(",") if f.strip()]
    if len(factor_names) != 2:
        raise SystemExit(
            f"❌ -factors needs exactly 2 column names, got {len(factor_names)}: {factor_names}"
        )

    all_rows = _read_table(args.trials)
    n_rows_total = len(all_rows)
    if args.drop_trials:
        print(f"\n✂️  Dropping trials ({n_rows_total} in table)")
    keep = _apply_drop_trials(all_rows, args.drop_trials)
    rows = [r for r, k in zip(all_rows, keep, strict=True) if k]

    # Levels arrive as free text ("face, inverted"), and become identifiers downstream --
    # map names, JSON keys, anything a later script indexes by. Sanitize once here, keeping
    # distinct labels distinct, and report the mapping so the raw names stay recoverable.
    factor_codes: dict[str, np.ndarray] = {}
    level_maps: dict[str, dict[str, str]] = {}
    for name in factor_names:
        raw = [str(v) for v in _column(rows, name)]
        clean, mapping = sanitize_levels(raw)
        factor_codes[name] = np.array(clean)
        level_maps[name] = mapping
        # -drop_trials makes a one-level factor easy to reach by accident; the
        # contrast builder's ValueError does not say which flag caused it.
        if len(set(clean)) < 2:
            raise SystemExit(
                f"❌ factor '{name}' has only one level ({sorted(set(clean))}) after "
                "trial selection; variance partitioning needs at least two."
            )

    run = _optional_column(rows, "run")
    session = _optional_column(rows, "session")
    repeat_col = _optional_column(rows, "repeat")
    repeat = repeat_col.astype(int) if repeat_col is not None else None

    # Session participates in fold locality the same way run does: a repeat that shares a
    # session with its training partners still shares session-level nuisance. When both
    # exist the block label is their combination, which is the stricter of the two.
    if run is not None and session is not None:
        block = np.array([f"{s}/{r}" for s, r in zip(session, run, strict=True)])
    else:
        block = run if run is not None else session

    kept_note = f" ({n_rows_total - len(rows)} dropped)" if len(rows) != n_rows_total else ""
    print(f"\n📋 Trial table: {len(rows)} rows{kept_note}")
    for name in factor_names:
        mapping = level_maps[name]
        renamed = sum(1 for k, v in mapping.items() if k != v)
        # Two raw labels sharing an identifier means they differed only in whitespace
        # and were merged into one level -- worth saying out loud, it changes the design.
        merged = len(mapping) - len(set(mapping.values()))
        notes = []
        if renamed:
            notes.append(f"{renamed} name(s) sanitized")
        if merged:
            notes.append(f"{merged} merged on whitespace")
        note = f", {'; '.join(notes)}" if notes else ""
        print(f"   • {name}: {len(np.unique(factor_codes[name]))} levels{note}")
    print(f"   • run: {'yes' if run is not None else 'absent'}")
    print(f"   • session: {'yes' if session is not None else 'absent'}")
    print(f"   • repeat: {'yes' if repeat is not None else 'derived from cell order'}")

    print(f"\n📥 Loading betas: {args.betas}")
    img = load_nifti(args.betas)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 4:
        raise SystemExit(f"❌ -betas must be 4-D (one volume per trial), got {data.ndim}-D")
    vol_shape, n_trials = data.shape[:3], data.shape[3]
    # The pairing is checked against the *full* table: -drop_trials subsets the volumes
    # here, so the file on disk still has to be one volume per un-dropped row.
    if n_trials != n_rows_total:
        raise SystemExit(
            f"❌ -betas has {n_trials} volumes but the trial table has {n_rows_total} rows.\n"
            "One row per volume is required; use -drop_trials to exclude trials rather "
            "than editing the table."
        )
    if len(rows) != n_rows_total:
        data = data[..., keep]
        n_trials = data.shape[3]
        print(f"   Kept {n_trials} of {n_rows_total} volumes after -drop_trials")

    if args.mask:
        mask = np.asanyarray(load_nifti(args.mask).dataobj).astype(bool)
        if mask.shape != vol_shape:
            raise SystemExit(f"❌ mask {mask.shape} does not match betas grid {vol_shape}")
    else:
        # Voxels that are all-zero across trials carry no information and would give a
        # zero noise ceiling; excluding them keeps them out of the FWE max-statistic.
        mask = np.any(data != 0, axis=3)

    betas = torch.as_tensor(data[mask].astype(np.float32))
    print(f"   {vol_shape} x {n_trials} trials; {betas.shape[0]:,} voxels in mask")

    roi_ids: list | None = None
    roi_spec: np.ndarray | None = None
    if args.atlas:
        atlas = np.asanyarray(load_nifti(args.atlas).dataobj)
        if atlas.shape[:3] != vol_shape:
            raise SystemExit(f"❌ atlas grid {atlas.shape[:3]} does not match betas {vol_shape}")
        roi_spec, roi_ids, roi_sizes = build_roi_weights(atlas, mask=mask)
        betas = collapse_to_rois(betas, roi_spec, roi_sizes, device=device).cpu()
        kind = "label map" if atlas.ndim == 3 else "4-D masks"
        print(f"\n🧩 Atlas ({kind}): collapsed to {betas.shape[0]} ROIs")
        print(f"   ROI sizes: min {roi_sizes.min():.0f}, median {np.median(roi_sizes):.0f}")

    print("\n🔬 Partitioning...")
    res = partition_variance(
        betas,
        factor_codes,
        repeat=repeat,
        run=block,
        max_rank=args.max_rank,
        min_ncsnr_for_rank=args.min_ncsnr_for_rank,
        device=device,
        verbose=not args.quiet,
    )

    fa, fb = factor_names
    assert res.rank_e is not None and res.rank_e_raw is not None
    assert res.shared is not None and res.interaction is not None and res.preference is not None
    d = res.diagnostics
    print("\n📊 Diagnostics")
    print(f"   balanced: {d['balanced']}  (max off-diagonal Gram {d['max_offdiag_gram']:.2e})")
    print(
        f"   cells: {d['cells_total']}, empty {d['cells_empty']}, repeats {d['repeats_min']}"
        f"-{d['repeats_max']}"
    )
    print(f"   folds: {d['n_folds']}   run locality: {d['run_locality_ok']}")
    # Shared variance is ~0 by construction under an exhaustively crossed balanced
    # design, so a non-trivial value means the balance broke, not that a real overlap
    # was found. Surface it as a check rather than a result.
    print(f"   shared |C| median: {d['shared_abs_median']:.4f}  (expected ~0)")
    if d["shared_abs_median"] > 0.02:
        print("   ⚠️  shared variance is not ~0: the design is unbalanced somewhere;")
        print("       treat the partition as approximate and check for dropped trials.")
    print(
        f"   rank undetermined (ncsnr < {args.min_ncsnr_for_rank}): "
        f"{d['rank_undetermined_frac']:.1%}"
    )

    perm_res = None
    if args.perm > 0:
        stats = tuple(s.strip() for s in args.perm_stats.split(",") if s.strip())
        print(f"\n🎲 Permutation null: {args.perm} permutations x {len(stats)} statistic(s)")
        if block is None:
            print("   ⚠️  no run/session column: permuting freely (anticonservative)")
        perm_res = permutation_test(
            betas,
            factor_codes,
            repeat=repeat,
            run=block,
            statistics=stats,
            n_perms=args.perm,
            seed=args.seed,
            device=device,
            verbose=not args.quiet,
        )

    # ── Outputs ──────────────────────────────────────────────────────────────
    maps: dict[str, torch.Tensor] = {
        f"unique_{fa}": res.unique[fa],
        f"unique_{fb}": res.unique[fb],
        "shared": res.shared,
        "interaction": res.interaction,
        "preference": res.preference,
        "rank_E": res.rank_e.float(),  # -1 where ncsnr is below the detection floor
        "rank_E_raw": res.rank_e_raw.float(),  # unmasked argmax, for diagnosing the mask
        "ncsnr": res.ncsnr,
        "noise_ceiling": res.noise_ceiling,
        f"gamma_{fa}": res.gammas[fa],
        f"gamma_{fb}": res.gammas[fb],
        "gamma_interaction": res.gammas[f"{fa}:{fb}"],
        "r2_additive": res.r2["M_add"],
        "r2_full": res.r2["M_full"],
    }
    if perm_res is not None:
        rename = {"unique_a": f"unique_{fa}", "unique_b": f"unique_{fb}"}
        for key in perm_res.p_fwe:
            base = rename.get(key, key)
            maps[f"p_unc_{base}"] = perm_res.p_uncorrected[key]
            maps[f"p_fwe_{base}"] = perm_res.p_fwe[key]

    info = parse_prefix(args.prefix)
    stem = info.stem

    names = list(maps)
    stacked = np.zeros((*vol_shape, len(maps)), dtype=np.float32)

    if roi_ids is not None:
        # The table is the quantitative output -- one row per ROI, no invented resolution.
        out_tsv = f"{stem}_roi.tsv"
        with open(out_tsv, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["roi"] + names)
            for i, rid in enumerate(roi_ids):
                w.writerow([rid] + [f"{float(maps[n][i]):.6g}" for n in names])
        print(f"\n💾 Wrote {out_tsv} ({len(roi_ids)} ROIs x {len(names)} measures)")

        # The volume is the *display* output: every voxel in a parcel carries that
        # parcel's value, so it renders on a brain without claiming within-parcel
        # structure. Same sub-bricks and grid as voxel mode, so figures and overlays
        # work identically either way.
        for i, name in enumerate(names):
            painted = paint_rois_to_voxels(maps[name], roi_spec, int(mask.sum()))
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = painted
            stacked[..., i] = vol
    else:
        for i, name in enumerate(names):
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = maps[name].cpu().numpy().astype(np.float32)
            stacked[..., i] = vol

    out_path = f"{stem}{info.nifti_ext}"
    save_nifti(stacked, output_path=out_path, affine=img.affine, brick_labels=names)
    kind = "parcel-painted" if roi_ids is not None else "voxelwise"
    print(f"💾 Wrote {out_path} ({len(maps)} sub-bricks, {kind})")
    for i, name in enumerate(names):
        print(f"   [{i:>2}] {name}")

    meta = {
        "factors": factor_names,
        "level_names": level_maps,  # sanitized identifier -> original label, per factor
        "dropped_trials": [list(d) for d in (args.drop_trials or [])],
        "n_trials_in_table": n_rows_total,
        "n_trials": n_trials,
        "n_units": int(betas.shape[0]),
        "unit": "roi" if roi_ids is not None else "voxel",
        "diagnostics": {
            k: (v if not isinstance(v, np.generic) else v.item()) for k, v in d.items()
        },
        "min_ncsnr_for_rank": args.min_ncsnr_for_rank,
        "n_perms": args.perm,
    }
    if roi_ids is not None:
        meta["roi_ids"] = [int(r) for r in roi_ids]
    with open(f"{stem}_varpart.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"💾 Wrote {stem}_varpart.json")

    print(f"\n✅ Done: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
