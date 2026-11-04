#!/usr/bin/env python
"""Paint any per-column quantity from an ffs_laminascope run onto surface and ribbon maps.

Why this exists: the per-column TSVs are unreadable at thousands of columns, and some
columns of them never reached the maps.  Runs made before the maps carried
``baseline_peak`` / ``in_fov`` can still be gated without refitting, because
``{prefix}_<cmp>_regions_long.tsv`` has both per column x depth and
``{prefix}_columns.npz`` has the column memberships the painting needs.

    python scripts/paint_laminascope_table.py \\
        -prefix laminascope/ss -comparison gender_vs_fixate \\
        -fields baseline_peak in_fov B_bma p_modulated \\
        -fs_subj $SUBJ -grid stage10.mask_epi.nii.zst -out laminascope/gate

Writes {out}_<cmp>.<hemi>.func.gii and {out}_<cmp>_ribbon.nii.gz, one brick per field
(per neuronal depth where a field varies with depth), NaN off the columns -- the same
convention the tool's own maps use, so "no column here" stays distinct from "no effect
here".

A field is read from the long table if it is there, otherwise from
{prefix}_<cmp>_columns.tsv.  Overlapping columns mean a vertex's value is the MEAN over
the columns containing it, exactly as the tool paints its own maps: a consensus of fits
that share data, not a pooled estimate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="paint_laminascope_table",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-prefix", required=True, help="The run's -prefix (finds _columns.npz etc).")
    p.add_argument("-comparison", required=True, help="Comparison NAME from the run.")
    p.add_argument("-fields", nargs="+", required=True, help="Column names to paint.")
    p.add_argument("-fs_subj", required=True, help="FreeSurfer subject (the run's -fs_subj).")
    p.add_argument("-grid", default=None,
                   help="Reference volume for the ribbon output; omit to write surfaces only.")  # fmt: skip
    p.add_argument("-out", required=True, help="Output prefix.")
    p.add_argument("-device", default=None, help="Device for the ribbon painting.")
    return p.parse_args(argv)


def _read_table(path, fields):
    """``{field: {column: value}}`` and, per field, whether it varied with depth.

    A long table has one row per column x neuronal_depth; a columns table one per
    column.  Depth-varying fields come back as ``{field: {column: [v0, v1, ...]}}``.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty")
    have = set(rows[0])
    key = "region" if "region" in have else "column"
    if key not in have:
        raise ValueError(f"{path}: no 'region' or 'column' field (have {sorted(have)})")
    depth_key = "neuronal_depth" if "neuronal_depth" in have else None
    depths = ([] if depth_key is None
              else sorted({r[depth_key] for r in rows}, key=lambda d: [r[depth_key] for r in rows].index(d)))  # fmt: skip
    out, varies = {}, {}
    for field in fields:
        if field not in have:
            continue
        per = defaultdict(dict)
        for r in rows:
            try:
                v = float(r[field])
            except (TypeError, ValueError):
                v = np.nan
            per[int(r[key])][r[depth_key] if depth_key else ""] = v
        if depth_key is None:
            out[field] = {c: [d[""]] for c, d in per.items()}
            varies[field] = [""]
            continue
        # Constant across depths (baseline_peak, in_fov, r2_best, ...) collapses to one
        # map; anything that moves with depth keeps one map per depth.
        const = all(len({v for v in d.values() if np.isfinite(v)}) <= 1 for d in per.values())
        if const:
            out[field] = {c: [next(iter(d.values()))] for c, d in per.items()}
            varies[field] = [""]
        else:
            out[field] = {c: [d.get(k, np.nan) for k in depths] for c, d in per.items()}
            varies[field] = list(depths)
    return out, varies


def main(argv=None) -> int:
    import scipy.sparse as sp

    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data
    from fastfuncstuff.laminar import columns as col
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    args = parse_args(argv)
    z = np.load(f"{args.prefix}_columns.npz")
    cs = col.ColumnSet(
        centre_surface=z["centre_surface"], centre_vertex=z["centre_vertex"],
        members=sp.csr_matrix(
            (z["members_data"], z["members_indices"], z["members_indptr"]),
            shape=tuple(int(n) for n in z["members_shape"]),
        ),
        vertex_surface=z["vertex_surface"], vertex_id=z["vertex_id"],
        radius_mm=float(z["radius_mm"]) if "radius_mm" in z.files else float("nan"),
        spacing_mm=float(z["spacing_mm"]) if "spacing_mm" in z.files else float("nan"),
    )  # fmt: skip

    wanted = list(args.fields)
    tables = [f"{args.prefix}_{args.comparison}_regions_long.tsv",
              f"{args.prefix}_{args.comparison}_columns.tsv"]  # fmt: skip
    values, varies, found = {}, {}, set()
    for path in tables:
        try:
            got, var = _read_table(path, [f for f in wanted if f not in found])
        except OSError:
            continue
        values.update(got)
        varies.update(var)
        found |= set(got)
    missing = [f for f in wanted if f not in found]
    if missing:
        print(f"not in {' or '.join(tables)}: {missing}", file=sys.stderr)
    if not found:
        return 1

    n_cols = cs.n_columns
    bricks, names = [], []
    for field in wanted:
        if field not in values:
            continue
        per = values[field]
        for i, depth in enumerate(varies[field]):
            v = np.full(n_cols, np.nan)
            for c, vals in per.items():
                if 0 <= c < n_cols:
                    v[c] = vals[i]
            bricks.append(v)
            names.append(field if depth == "" else f"{field}_{depth}")
    stack = np.stack(bricks, axis=1)
    print(f"painting {len(names)} map(s) from {sum(np.isfinite(b).sum() for b in bricks):,} "
          f"column values over {n_cols:,} columns: {', '.join(names)}")  # fmt: skip

    surfaces = resolve_ribbon_surfaces(args.fs_subj, "white", "pial", None, None, None,
                                       hemi="both", log=lambda m: None)  # fmt: skip
    vert, n_cov = col.paint_vertices(cs, stack, np.isfinite(stack).any(1))
    vmaps = np.concatenate([vert, n_cov[:, None].astype(float)], 1)
    vnames = names + ["n_columns"]
    written = []
    for h, s in enumerate(surfaces):
        if not np.any(cs.vertex_surface == h):
            continue
        data = col.full_surface(cs, vmaps, h, len(s.white))
        meta = {"tool": "paint_laminascope_table", "comparison": args.comparison,
                "maps": ",".join(vnames), "hemi": s.name,
                "mesh_fingerprint": mesh_fingerprint(s.faces, len(s.white))}  # fmt: skip
        path = f"{args.out}_{args.comparison}.{s.name}.func.gii"
        save_gifti_data(path, data, meta, time_series=False)
        written.append(path)

    if args.grid:
        import torch

        ref = load_nifti(args.grid)
        affine = np.asarray(ref.affine, np.float64)
        shape3 = tuple(int(n) for n in ref.shape[:3])
        device = torch.device(args.device) if args.device else None
        vol = col.paint_ribbon(surfaces, cs, vmaps, affine, shape3, device=device)
        vol = vol.reshape(*shape3, -1).astype(np.float32)
        j = vnames.index("n_columns")
        vol[..., j] = np.nan_to_num(vol[..., j], nan=0.0)
        path = f"{args.out}_{args.comparison}_ribbon.nii.gz"
        save_nifti(vol, output_path=path, reference_img=None, affine=affine,
                   brick_labels=vnames)  # fmt: skip
        written.append(path)
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
