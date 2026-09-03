#!/usr/bin/env python3
"""ffs_util_restore — give a wrongly-removed component back to a denoised series."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_device_arg,
    add_verbose_arg,
    parse_prefix,
    print_cli_header,
    print_cli_section,
    setup_device,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_util_restore",
        description=__doc__,
        formatter_class=FfsHelpFormatter,
        epilog="""
The workflow this closes:

  ffs_nordic -input-magn bold.nii.gz -input-phase ph.nii.gz \\
             -prefix NORD -events run_events.tsv
      # -events implies -save_task_loss, so NORD_taskloss is the REMOVED field

  ffs_ica -input NORD_taskloss.nii.gz -prefix LOSS -num_comps auto
      # model order comes from the eigenspectrum. The removed field is the
      # Marchenko-Pastur BULK -- what is left once the top components are taken
      # out -- which is exactly what that estimator is built for, and it reports
      # ~0 components when the denoiser removed only noise.

  ffs_util_restore -denoised NORD.nii.gz -removed NORD_taskloss.nii.gz \\
                   -maps LOSS_ica_maps.nii.gz \\
                   -timecourses LOSS_ica_timecourses.1D \\
                   -events run_events.tsv -prefix NORD_restored

Selection is familywise-controlled across components, not per-voxel: a component
either fits the task in its TIME COURSE or it does not, and that is one test per
component against its own phase-randomised null.
""",
    )
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-denoised", required=True, metavar="FILE", help="The denoised series to add back into."
    )
    io_group.add_argument(
        "-removed",
        required=True,
        metavar="FILE",
        help="The removed field the components were decomposed from "
        "(ffs_nordic's {prefix}_taskloss).",
    )
    io_group.add_argument(
        "-maps",
        required=True,
        metavar="FILE",
        help="4-D component spatial maps, one sub-brick each (ffs_ica's _ica_maps).",
    )
    io_group.add_argument(
        "-timecourses",
        "-mixing",
        required=True,
        metavar="FILE",
        dest="timecourses",
        help="(T, K) component time courses (ffs_ica's _ica_timecourses.1D).",
    )
    io_group.add_argument("-prefix", required=True, help="Output prefix.")

    sel = parser.add_argument_group("Component selection")
    sel.add_argument(
        "-components",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="0-based component indices to restore. Overrides -events selection; give "
        "both to score the components you picked and see the numbers behind them.",
    )
    sel.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv for this run. Scores every component's TIME COURSE "
        "against the task and, unless -components says otherwise, restores the ones "
        "that survive familywise control. The null is each component phase-randomised "
        "against its own amplitude spectrum -- a component's autocorrelation is what "
        "makes a spurious fit possible -- and the threshold is the (1-alpha) quantile "
        "of the MAX z across components per draw, so a 60-component decomposition does "
        "not flag three by chance.",
    )
    sel.add_argument(
        "-event_ignore",
        "-event-ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude from the design.",
    )
    sel.add_argument(
        "-event_cols",
        "-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom -events column names. Default: onset duration trial_type.",
    )
    sel.add_argument(
        "-tr",
        type=float,
        default=None,
        metavar="SEC",
        help="Repetition time for the design. Default: the -denoised header pixdim[4].",
    )
    sel.add_argument(
        "-task_polort",
        "-task-polort",
        type=int,
        default=None,
        metavar="DEG",
        help="Legendre drift degree removed before the fit. "
        "Default: AFNI's 1 + floor(run_seconds/150).",
    )
    sel.add_argument(
        "-alpha",
        type=float,
        default=0.05,
        metavar="A",
        help="Familywise error rate for the component selection. Default 0.05.",
    )
    sel.add_argument(
        "-surrogates",
        type=int,
        default=2000,
        metavar="N",
        help="Phase-randomised draws per component for the null. Default 2000.",
    )
    sel.add_argument(
        "-dry_run",
        "-dry-run",
        action="store_true",
        help="Score and report, write nothing. Use it to look before restoring.",
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def _load(path: str):
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(path)
    return img, np.ascontiguousarray(img.get_fdata(dtype=np.float32))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.components is None and args.events is None:
        print("ERROR: give -components, or -events to select them automatically")
        sys.exit(1)

    device = setup_device(args.device)
    prefix_info = parse_prefix(args.prefix)
    ext = prefix_info.nifti_ext or ".nii.gz"

    print_cli_header("ffs_util_restore", "Return removed components to a denoised series")

    den_img, den = _load(args.denoised)
    _, rem = _load(args.removed)
    _, maps4d = _load(args.maps)
    mixing = np.atleast_2d(np.loadtxt(args.timecourses))
    if mixing.shape[0] != den.shape[3] and mixing.shape[1] == den.shape[3]:
        mixing = mixing.T  # a (K, T) file is the other common spelling; accept it
    n_k = mixing.shape[1]
    if maps4d.ndim != 4 or maps4d.shape[3] != n_k:
        print(
            f"ERROR: {n_k} time courses but the maps file holds "
            f"{maps4d.shape[3] if maps4d.ndim == 4 else 'a 3-D volume'}"
        )
        sys.exit(1)
    maps_kv = torch.as_tensor(maps4d.reshape(-1, n_k).T.copy())
    mixing_tk = torch.as_tensor(mixing.astype(np.float32))
    print(f"  {n_k} components, {den.shape[3]} frames, grid {den.shape[:3]}")

    fit = None
    if args.events is not None:
        from fastfuncstuff.cli.task_events import task_design_from_events
        from fastfuncstuff.stats.task_coupling import component_task_fit, default_polort

        n_t = den.shape[3]
        tr = args.tr if args.tr is not None else float(den_img.header["pixdim"][4])
        if tr <= 0:
            print(f"ERROR: the header gives TR={tr:g}s; pass -tr SEC")
            sys.exit(1)
        design, labels = task_design_from_events(args, n_t, tr, torch.device("cpu"))
        polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
        print_cli_section("Component task fit")
        print(
            f"  {len(labels)} condition(s) ({', '.join(labels)}), TR {tr:g}s, "
            f"polort {polort}, alpha {args.alpha:g} familywise"
        )
        fit = component_task_fit(
            mixing_tk, design, polort=polort, n_surrogates=args.surrogates, alpha=args.alpha
        )
        flagged = list(fit["flagged"])
        for c in flagged[:20]:
            print(
                f"    comp {c:3d}: R2 {float(fit['r2'][c]):.3f}  z {float(fit['z'][c]):+.2f}  "
                f"p {float(fit['p'][c]):.4g}"
            )
        print(
            f"  {len(flagged)} component(s) fit the task above z_cut "
            f"{float(fit['z_cut']):.2f}" + (f": {flagged}" if flagged else " — nothing to restore")
        )

    indices = args.components if args.components is not None else list(fit["flagged"])
    if not indices:
        print("\nNothing selected; no output written.")
        return

    print_cli_section("Restore")
    from fastfuncstuff.denoise.restore import restore_components

    result = restore_components(
        torch.as_tensor(den).to(device),
        torch.as_tensor(rem).to(device),
        maps_kv.to(device),
        mixing_tk.to(device),
        list(indices),
    )
    for c, g, v in zip(result.indices, result.gammas, result.var_returned, strict=True):
        print(f"    comp {c:3d}: amplitude {g:+.4g}, {100 * v:.2f}% of the removed variance")
    print(
        f"  restored {len(result.indices)} component(s), "
        f"{100 * result.var_returned_total:.2f}% of the removed field's variance, "
        f"{result.dof_returned} degrees of freedom handed back"
    )
    if args.dry_run:
        print("\n  -dry_run: nothing written.")
        return

    from fastfuncstuff.io.afni import save_nifti

    out_path = Path(f"{prefix_info.stem}{ext}")
    save_nifti(
        result.restored.cpu().numpy().astype(np.float32),
        output_path=out_path,
        reference_img=args.denoised,
    )
    meta = {
        "denoised": args.denoised,
        "removed": args.removed,
        "maps": args.maps,
        "timecourses": args.timecourses,
        "restored_components": result.indices,
        "gammas": result.gammas.tolist(),
        "variance_share_of_removed": result.var_returned.tolist(),
        "variance_share_of_removed_total": result.var_returned_total,
        "dof_returned": result.dof_returned,
        "selection": "events" if args.components is None else "explicit",
    }
    if fit is not None:
        meta["z_cut"] = float(fit["z_cut"])
        meta["component_r2"] = [float(x) for x in np.asarray(fit["r2"])]
    meta_path = Path(f"{prefix_info.stem}_restore.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  Restored series: {out_path}")
    print(f"  Metadata: {meta_path}")
    print(
        f"  NOTE {result.dof_returned} degrees of freedom went back into the data; "
        "carry that into ffs_reml -adjust_dof / ffs_util_updatedof."
    )


if __name__ == "__main__":
    main()
