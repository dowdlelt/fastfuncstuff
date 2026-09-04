#!/usr/bin/env python3
"""
ffs_clustsim — Monte-Carlo cluster-size thresholds (3dClustSim).

Simulates noise-only volumes with a prescribed spatial autocorrelation,
finds the largest null cluster at each per-voxel threshold, and turns the
distribution of that maximum into the cluster-size table AFNI's viewer
reads out of a stats dataset's header.

Unlike ``ffs_perm`` this needs no permutable design, so it applies to an
ordinary first-level GLM::

    ffs_clustsim -mask mask.nii.gz -acf 0.6 3.0 5.0 -niter 10000 \\
                 -prefix CStemp -refit stats.nii.gz

``-acf_from`` closes the loop in one process: the ACF is estimated off a
residual dataset with the same 3dFWHMx port ffs_reml uses, so there is no
3dFWHMx → parse-a-.1D → 3dClustSim → 3drefit round trip::

    ffs_clustsim -acf_from errts.nii.gz -mask mask.nii.gz -refit stats.nii.gz \\
                 -prefix CStemp

The threshold is a z, not a t: the fields are renormalised to unit
standard deviation, so one table serves every sub-brick regardless of its
degrees of freedom.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

# Sidedness spellings: the hyphenated form is what the NIML `thresholding`
# attribute carries; the bare form names the file and the 3drefit attribute.
_SIDED_ATTR = {"1-sided": "1sided", "2-sided": "2sided", "bi-sided": "bisided"}
_SIDED_FROM_CLI = {"1sided": "1-sided", "2sided": "2-sided", "bisided": "bi-sided"}


def build_parser() -> FfsArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_clustsim",
        description=__doc__,
        formatter_class=FfsHelpFormatter,
    )

    inp = p.add_argument_group("Inputs")
    inp.add_argument(
        "-mask",
        required=True,
        help="Mask dataset.  Simulated clusters are confined to it, and its "
        "grid and voxel size set the simulation geometry.",
    )
    smooth = inp.add_mutually_exclusive_group(required=True)
    smooth.add_argument(
        "-acf",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        help="Mixed-model ACF parameters from 3dFWHMx -acf (or ffs_reml "
        "-save_acf):  ACF(r) = a*exp(-r^2/2b^2) + (1-a)*exp(-r/c).",
    )
    smooth.add_argument(
        "-acf_from",
        metavar="RESID",
        help="Estimate the ACF from this residual dataset (errts) instead of "
        "taking it as a number.  Uses the same 3dFWHMx port as ffs_reml.",
    )
    smooth.add_argument(
        "-fwhm",
        type=float,
        help="Pure-Gaussian smoothness in mm, as an equivalent ACF.  Real fMRI "
        "residuals have a heavier tail than a Gaussian, so -acf is the "
        "defensible choice; this exists for comparison against old results.",
    )

    sim = p.add_argument_group("Simulation")
    sim.add_argument(
        "-niter",
        type=int,
        default=10000,
        help="Monte-Carlo iterations.  Below ~2000 the tail of the table is too noisy to trust.",
    )
    sim.add_argument(
        "-seed",
        type=int,
        default=None,
        help="Random seed (default: nondeterministic).  Reproduces a run "
        "exactly only together with -batch, since the automatic batch size "
        "depends on how much memory is free at the time.",
    )
    sim.add_argument(
        "-batch",
        type=int,
        default=None,
        help="Volumes simulated per batch.  Default: from free memory on the "
        "target device.  Lower it if the GPU is shared.",
    )
    sim.add_argument(
        "-cpu_cluster",
        action="store_true",
        help="Cluster on CPU worker processes even when simulating on a GPU.  "
        "The two agree exactly; this is the fallback if the fused kernels "
        "are unavailable.",
    )
    sim.add_argument(
        "-pthr",
        nargs="+",
        type=float,
        default=None,
        help="Per-voxel uncorrected p thresholds (rows of the table).",
    )
    sim.add_argument(
        "-athr",
        nargs="+",
        type=float,
        default=None,
        help="Family-wise corrected alphas (columns of the table).",
    )
    sim.add_argument(
        "-LOTS",
        action="store_true",
        help="Use AFNI's larger 29-pthr x 10-athr grid.",
    )
    sim.add_argument(
        "-NN",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[1, 2, 3],
        help="Connectivities to tabulate: 1=faces, 2=+edges, 3=+corners.",
    )
    sim.add_argument(
        "-sided",
        nargs="+",
        choices=("1sided", "2sided", "bisided"),
        default=["1sided", "2sided", "bisided"],
        help="Thresholding schemes.  bisided clusters each sign separately; "
        "2sided lets opposite-sign voxels join one cluster.",
    )

    out = p.add_argument_group("Output")
    out.add_argument(
        "-prefix",
        required=True,
        help="Output prefix for the .1D tables, the NIML files and the mask blob.",
    )
    out.add_argument(
        "-refit",
        metavar="DSET",
        default=None,
        help="Inject the tables into this stats dataset's AFNI header so the "
        "viewer reports cluster significance.  Done in-script; 3drefit is "
        "not required, though the equivalent script is still written.",
    )
    out.add_argument(
        "-nodec",
        action="store_true",
        help="Round cluster sizes up to whole voxels in the .1D tables.",
    )

    misc = p.add_argument_group("Misc")
    try:
        from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg

        add_device_arg(misc)
        add_verbose_arg(misc)
    except ImportError:  # pragma: no cover
        misc.add_argument("-device", default=None)
        misc.add_argument("-verb", type=int, default=1)
    misc.add_argument(
        "-jobs",
        "-j",
        type=int,
        default=None,
        help="Worker processes for the clustering pass.  Default: cpu_count-1.",
    )
    return p


# ---------------------------------------------------------------------------
# .1D table, in 3dClustSim's layout
# ---------------------------------------------------------------------------


def _prob6(p: float) -> str:
    """AFNI ``prob6``: a p-value in exactly 6 characters (column headers)."""
    if p >= 0.00010:
        return f"{p:7.5f}"[1:]
    dec = int(0.9999 - math.log10(p))
    return f"{p * 10.0**dec:4.1f}e-{dec:1d}"[1:]


def _prob9(p: float) -> str:
    """AFNI ``prob9``: a p-value in exactly 9 characters (row labels)."""
    if p >= 0.00010:
        return f"{p:9.6f}"
    dec = int(0.9999 - math.log10(p))
    return f"{p * 10.0**dec:6.3f}e-{dec:1d}"


def write_1d_table(
    path: Path,
    table: np.ndarray,
    *,
    nn: int,
    sidedness: str,
    pthr: tuple[float, ...],
    athr: tuple[float, ...],
    shape: tuple[int, int, int],
    voxmm: tuple[float, float, float],
    mask_count: int,
    commandline: str,
    nodec: bool,
) -> None:
    """Write one ``ppp.NN{n}_{sided}.1D``.

    Byte-for-byte 3dClustSim's layout, down to ``prob9``/``prob6`` and the
    ``%7.1f`` cells, because these files are read by eye and by afni_proc.
    """
    n_total = int(np.prod(shape))
    in_mask = " in mask" if mask_count < n_total else ""
    lines = [
        f"# {commandline}",
        f"# {sidedness} thresholding",
        "# Grid: {}x{}x{} {:.2f}x{:.2f}x{:.2f} mm^3 ({} voxels{})".format(
            *shape, *voxmm, mask_count, in_mask
        ),
        "#",
        "# CLUSTER SIZE THRESHOLD(pthr,alpha) in Voxels",
        f"# -NN {nn}  | alpha = Prob(Cluster >= given size)",
        "#  pthr  |" + "".join(f" {_prob6(a)}" for a in athr),
        "# ------ |" + " ------" * len(athr),
    ]
    for i, p in enumerate(pthr):
        cells = ""
        for v in table[i]:
            if nodec:
                cells += f"{int(v):7d}"  # already rounded by gumbel_extent_table
            elif v <= 9999.9:
                cells += f"{v:7.1f}"
            else:
                cells += f"{v:7.0f}"
        lines.append(f"{_prob9(p)} {cells}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from fastfuncstuff.cli_utils import parse_prefix, setup_device
    from fastfuncstuff.io.afni import load_afni_mask, load_nifti
    from fastfuncstuff.stats.clustsim import (
        ACF,
        DEFAULT_CS_ATHR,
        DEFAULT_CS_PTHR,
        LOTS_ATHR,
        LOTS_PTHR,
        acf_fwhm,
        gumbel_extent_table,
        random_field_grid,
        simulate_cluster_null,
    )
    from fastfuncstuff.stats.niml import (
        resolve_mask_idcode,
        run_refit,
        write_clustsim_niml,
        write_mask_b64,
    )

    t_start = time.time()
    verb = getattr(args, "verb", 1)
    device = setup_device(args.device)

    # ── Geometry ───────────────────────────────────────────────────────────
    mask = load_afni_mask(args.mask)
    ref = load_nifti(args.mask)
    voxmm = tuple(abs(float(z)) for z in ref.header.get_zooms()[:3])
    shape = tuple(int(s) for s in mask.shape)
    n_mask = int(mask.sum())
    if n_mask == 0:
        print("[ffs_clustsim] mask is empty", file=sys.stderr)
        return 1

    # ── Smoothness ─────────────────────────────────────────────────────────
    if args.acf is not None:
        acf = ACF(*args.acf)
    elif args.fwhm is not None:
        acf = ACF.from_fwhm(args.fwhm)
    else:
        acf = _estimate_acf(args.acf_from, mask, shape, voxmm, device, verb)

    pthr = tuple(args.pthr) if args.pthr else (LOTS_PTHR if args.LOTS else DEFAULT_CS_PTHR)
    athr = tuple(args.athr) if args.athr else (LOTS_ATHR if args.LOTS else DEFAULT_CS_ATHR)
    nns = tuple(sorted(set(args.NN)))
    sideds = tuple(_SIDED_FROM_CLI[s] for s in dict.fromkeys(args.sided))

    grid = random_field_grid(shape, voxmm, acf)
    if verb >= 1:
        print(
            f"[ffs_clustsim] {n_mask} voxels in mask "
            f"({100.0 * n_mask / np.prod(shape):.2f}% of {shape[0]}x{shape[1]}x{shape[2]})\n"
            f"[ffs_clustsim] ACF({acf.a:.2f},{acf.b:.2f},{acf.c:.2f}) => "
            f"FWHM={acf_fwhm(acf):.2f}mm => pads to {grid[0]}x{grid[1]}x{grid[2]}\n"
            f"[ffs_clustsim] {args.niter} iterations on {device}",
            file=sys.stderr,
        )

    # ── Simulate ───────────────────────────────────────────────────────────
    null = simulate_cluster_null(
        mask,
        voxmm,
        acf,
        n_iter=args.niter,
        pthr=pthr,
        athr=athr,
        nns=nns,
        sideds=sideds,
        device=device,
        n_jobs=args.jobs,
        batch=args.batch,
        seed=args.seed,
        on_device=False if args.cpu_cluster else None,
        verbose=verb >= 1,
    )

    # ── Tables ─────────────────────────────────────────────────────────────
    prefix = parse_prefix(args.prefix)
    out_dir = Path(prefix.stem).parent
    base = Path(prefix.stem).name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_line = " ".join(["ffs_clustsim", *sys.argv[1:]])

    mask_b64 = out_dir / f"{base}.mask"
    mask_count = write_mask_b64(mask_b64, mask)
    mask_idcode = resolve_mask_idcode(args.mask)
    mask_name = str(Path(args.mask).resolve())

    niml_files: dict[tuple[int, str], Path] = {}
    for sided in sideds:
        for nn in nns:
            table = gumbel_extent_table(
                null.max_extent[(sided, nn)], athr, args.niter, nodec=args.nodec
            )
            tag = _SIDED_ATTR[sided]
            write_1d_table(
                out_dir / f"{base}.NN{nn}_{tag}.1D",
                table,
                nn=nn,
                sidedness=sided,
                pthr=pthr,
                athr=athr,
                shape=shape,
                voxmm=voxmm,
                mask_count=mask_count,
                commandline=cmd_line,
                nodec=args.nodec,
            )
            niml_path = out_dir / f"{base}.NN{nn}_{tag}.niml"
            write_clustsim_niml(
                niml_path,
                table,
                nn=nn,
                sidedness=sided,
                commandline=cmd_line,
                nxyz=shape,
                dxyz=voxmm,
                pthr=pthr,
                athr=athr,
                n_perms=args.niter,
                mask_count=mask_count,
                mask_idcode=mask_idcode,
                mask_name=mask_name,
            )
            niml_files[(nn, tag)] = niml_path

    if verb >= 1:
        _print_summary(null, nns, sideds, pthr, athr, args.niter, gumbel_extent_table)

    # ── Attach to the stats dataset ────────────────────────────────────────
    if args.refit is not None:
        ok = run_refit(
            stat_path=Path(args.refit),
            niml_files=niml_files,
            mask_b64_path=mask_b64,
            write_script_path=out_dir / f"{base}.3drefit.cmd",
            verbose=verb >= 1,
        )
        if verb >= 1 and ok:
            print(f"[ffs_clustsim] cluster tables inserted into {args.refit}", file=sys.stderr)

    if verb >= 1:
        print(f"[ffs_clustsim] done in {time.time() - t_start:.1f}s", file=sys.stderr)
    return 0


def _estimate_acf(resid_path, mask, shape, voxmm, device, verb):
    """Estimate the ACF off a residual dataset, the way 3dFWHMx -acf does."""
    import torch

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.stats.clustsim import ACF
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    img = load_nifti(resid_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise SystemExit(f"-acf_from needs a 4-D residual dataset, got shape {data.shape}")
    if data.shape[:3] != shape:
        raise SystemExit(f"-acf_from grid {data.shape[:3]} does not match the mask grid {shape}")
    resid = torch.from_numpy(data[mask])  # (V, T)
    del data
    est = estimate_fwhmx_run(
        resid,
        torch.from_numpy(mask),
        shape,
        voxmm,
        device=device,
        progress=verb >= 1,
    )
    if verb >= 1:
        print(
            f"[ffs_clustsim] estimated ACF from {Path(resid_path).name}: "
            f"a={est.a:.4f} b={est.b:.4f} c={est.c:.4f}  FWHM={est.fwhm:.2f}mm",
            file=sys.stderr,
        )
    return ACF(est.a, est.b, est.c)


def _print_summary(null, nns, sideds, pthr, athr, niter, table_fn):
    """Echo the NN1 table, the one people actually read off the terminal."""
    sided = sideds[0]
    nn = nns[0]
    table = table_fn(null.max_extent[(sided, nn)], athr, niter)
    print(f"\n# NN{nn} {sided} — cluster size threshold (voxels)", file=sys.stderr)
    print("#  pthr  | " + " ".join(f"{a:.5f}"[1:].rjust(6) for a in athr), file=sys.stderr)
    print("# ------ | " + " ".join("------" for _ in athr), file=sys.stderr)
    for i, p in enumerate(pthr):
        print(
            f" {p:.6f} " + " ".join(f"{v:6.1f}" for v in table[i]),
            file=sys.stderr,
        )
    print("", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
