"""CLI for MELODIC-style Gaussian/Gamma mixture modelling of a statistic image.

Command: ffs_util_gmm (registered as entry point in pyproject.toml)

Fits MELODIC's mixture model (one Gaussian for the null + two Gammas for the
positive and negative tails) to each sub-brick of a statistic image, then
rescales the map so the *null* has mean 0 and standard deviation 1 and reports
the posterior probability that each voxel is signal.

This is the standalone equivalent of FSL's documented trick for running
MELODIC's mixture modelling without the ICA:

    echo "1" > grot.txt
    melodic -i myZstat --ICs=myZstat --mix=grot.txt -o out --Oall \\
            --report -v --mmthresh=0

Use it when a nominally-z statistic image isn't really z — the classic case
being a GLM on temporally smooth data, where the null is not N(0, 1) and every
threshold you pick is wrong. The mixture model estimates the null empirically
and standardises against it.

Usage:
    ffs_util_gmm -input zstat.nii.gz -prefix out
    ffs_util_gmm -input melodic_IC.nii.gz -mask mask.nii.gz -prefix out -mmthresh 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_verbose_arg,
    parse_prefix,
    print_cli_header,
    setup_device,
    spinner,
)
from fastfuncstuff.decomposition import io as decomposition_io
from fastfuncstuff.decomposition.tools import batch_mixture_zscores
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_util_gmm",
        description="MELODIC-style Gaussian/Gamma mixture modelling of a statistic "
        "image: fit the null empirically, rescale to a true z, and report P(signal). "
        "GPU-accelerated; matches FSL MELODIC's mixture-model stage.",
    )
    parser.add_argument(
        "-input",
        required=True,
        help="Statistic image (3D single map, or 4D fit independently per sub-brick)",
    )
    parser.add_argument("-prefix", required=True, help="Output prefix")
    parser.add_argument(
        "-mask",
        default=None,
        help="Mask dataset (nonzero = in). Without it, voxels that are nonzero and "
        "finite in at least one sub-brick are used.",
    )
    parser.add_argument(
        "-mmthresh",
        type=float,
        default=0.5,
        help="Threshold on P(signal) for the thresholded output. MELODIC's default "
        "0.5 balances false positives against false negatives; 0 writes the "
        "corrected-but-unthresholded map. (default: 0.5)",
    )
    parser.add_argument(
        "-n_iter",
        "-n-iter",
        dest="n_iter",
        type=int,
        default=200,
        help="Maximum EM iterations (default: 200). Raising this does NOT improve "
        "MELODIC agreement — see the note in the module docstring of "
        "decomposition/tools.py.",
    )
    parser.add_argument(
        "-drop_constant",
        "-drop-constant",
        dest="drop_constant",
        action="store_true",
        default=True,
        help="Exclude zero-variance voxels from the fit, as MELODIC does. Leaving "
        "them in puts a delta spike at zero in the histogram that collapses the "
        "null Gaussian. (default: on)",
    )
    parser.add_argument(
        "-no_drop_constant",
        "-no-drop-constant",
        dest="drop_constant",
        action="store_false",
        help="Keep constant voxels in the fit (not recommended)",
    )
    parser.add_argument(
        "-save_prob",
        "-save-prob",
        dest="save_prob",
        action="store_true",
        default=True,
        help="Write the P(signal) map (default: on)",
    )
    parser.add_argument(
        "-no_save_prob",
        "-no-save-prob",
        dest="save_prob",
        action="store_false",
        help="Skip the P(signal) map",
    )
    parser.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto)")
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verb = args.verb

    device = setup_device(args.device, tf32=REGISTRATION_TF32)

    _pfx = parse_prefix(args.prefix)
    out_prefix, nii_ext = _pfx.stem, _pfx.nifti_ext
    if verb >= 1:
        print_cli_header("ffs_util_gmm", "MELODIC-style mixture modelling")
    t0 = time.time()

    # Stay in NIfTI (x, y, z[, k]) order throughout — that is what
    # save_masked_component_maps_4d and `data[mask3d]` both assume.
    with spinner(f"Loading {Path(args.input).name}"):
        img = load_nifti(args.input)
        vol = np.asarray(img.dataobj, dtype=np.float32)
    vol = np.squeeze(vol)
    if vol.ndim == 3:
        vol = vol[..., np.newaxis]
    elif vol.ndim != 4:
        raise SystemExit(f"ffs_util_gmm: expected a 3D or 4D dataset, got {vol.shape}")
    shape3d = tuple(int(s) for s in vol.shape[:3])
    n_k = int(vol.shape[3])
    affine = np.asarray(img.affine, dtype=np.float64)

    maps_np = vol.reshape(-1, n_k).T.copy()  # (K, V_all)
    del vol

    # --- Mask ---
    if args.mask is not None:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask_arr = np.squeeze(np.asarray(load_nifti(args.mask).dataobj))
        if mask_arr.ndim == 4:
            mask_arr = mask_arr[..., 0]
        if tuple(int(s) for s in mask_arr.shape) != shape3d:
            raise SystemExit(f"ffs_util_gmm: mask shape {mask_arr.shape} != data grid {shape3d}")
        mask3d = mask_arr != 0
    else:
        finite_any = np.isfinite(maps_np).all(axis=0)
        nonzero_any = (maps_np != 0).any(axis=0)
        mask3d = (finite_any & nonzero_any).reshape(shape3d)
        if verb >= 1:
            print(f"  Auto-mask (nonzero & finite): {int(mask3d.sum()):,} voxels")

    flat_mask = mask3d.reshape(-1)
    comp_kv = maps_np[:, flat_mask]  # (K, V)

    if args.drop_constant:
        # "No data here", not "zero variance across maps" — the sub-bricks are
        # independent statistic maps, not a timeseries, and a 3D input has only
        # one of them. Exactly-zero or non-finite is what MELODIC excludes.
        bad = ~np.isfinite(comp_kv).all(axis=0) | (comp_kv == 0).all(axis=0)
        n_drop = int(bad.sum())
        if n_drop:
            keep3d = mask3d.copy()
            keep3d[mask3d] = ~bad
            mask3d = keep3d
            comp_kv = comp_kv[:, ~bad]
            if verb >= 1:
                print(
                    f"  Mask updated: dropped {n_drop:,} constant voxels "
                    f"({100.0 * n_drop / len(bad):.2f}% of mask)"
                )

    n_vox = comp_kv.shape[1]
    if n_vox < 100:
        raise SystemExit(f"ffs_util_gmm: only {n_vox} voxels after masking — check -mask")
    if verb >= 1:
        print(f"  Fitting {n_k} map(s) x {n_vox:,} voxels on {device}")

    t_fit = time.time()
    z_t, p_t, meta = batch_mixture_zscores(
        torch.from_numpy(comp_kv), device=device, verbose=verb >= 1, n_iter=int(args.n_iter)
    )
    z_maps = z_t.cpu().numpy().astype(np.float32)
    p_maps = p_t.cpu().numpy().astype(np.float32)
    if verb >= 1:
        n_conv = sum(1 for m in meta if m.get("converged", False))
        n_fb = sum(1 for m in meta if m.get("gmm_fallback", False))
        print(
            f"  GGM done in {time.time() - t_fit:.2f}s: {n_conv}/{n_k} converged, {n_fb} fallback"
        )

    thresh_z = z_maps.copy()
    thresh_z[p_maps < float(args.mmthresh)] = 0.0

    # --- Save ---
    saves = [
        (z_maps, f"{out_prefix}_zmaps{nii_ext}"),
        (thresh_z, f"{out_prefix}_thresh_zmaps{nii_ext}"),
    ]
    if args.save_prob:
        saves.append((p_maps, f"{out_prefix}_probmap{nii_ext}"))
    for arr, fname in saves:
        decomposition_io.save_masked_component_maps_4d(
            components_kv=arr,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(fname),
        )
        if verb >= 1:
            print(f"  Wrote {fname}")

    # The mask actually fitted, after any constant-voxel pruning.
    save_nifti(
        mask3d.astype(np.float32),
        output_path=f"{out_prefix}_mask{nii_ext}",
        affine=affine,
    )
    if verb >= 1:
        print(f"  Wrote {out_prefix}_mask{nii_ext}")

    # MMstats: one row per map, MELODIC's (noise, pos, neg) x (mean, var, prop).
    stats_rows = np.array(
        [
            [
                m["mu_noise"],
                m["sigma_noise"] ** 2,
                m["pi_noise"],
                m["mu_signal_pos"],
                m["sigma_signal_pos"] ** 2,
                m["pi_pos"],
                m["mu_signal_neg"],
                m["sigma_signal_neg"] ** 2,
                m["pi_neg"],
            ]
            for m in meta
        ],
        dtype=np.float64,
    )
    np.savetxt(
        f"{out_prefix}_mmstats.txt",
        stats_rows,
        fmt="%.8f",
        header="mu_noise var_noise pi_noise mu_pos var_pos pi_pos mu_neg var_neg pi_neg",
    )
    with open(f"{out_prefix}_mmstats.json", "w") as fh:
        json.dump(
            {
                "input": str(args.input),
                "n_maps": n_k,
                "n_voxels": int(n_vox),
                "mmthresh": float(args.mmthresh),
                "components": meta,
            },
            fh,
            indent=2,
            default=float,
        )
    if verb >= 1:
        print(f"  Wrote {out_prefix}_mmstats.txt / .json")
        print(f"\nDone in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
