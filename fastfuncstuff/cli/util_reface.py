"""CLI for defacing / refacing a T1w (AFNI @afni_refacer_run, GPU).

Command: ffs_util_reface (registered as entry point in pyproject.toml)

Aligns the head to AFNI's MNI template, carries the refacer shell back onto the
subject, and zeroes (deface) or replaces (reface, reface_plus) the face. Same
template, shell, modes, flags and output names as @afni_refacer_run; the affine
search -- nearly all of AFNI's ~7 minutes -- is ffs_allineate's.

Usage:
    ffs_util_reface -input sub-01_T1w.nii.gz -mode_reface_plus -prefix sub-01_acq-reface_T1w
    ffs_util_reface -input T1w.nii.gz -mode_all -prefix T1w_anon -anonymize_output
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter, option_string_value
from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_device_arg,
    add_verbose_arg,
    collect_batch_jobs,
    find_afni_file,
    parse_prefix,
    print_cli_footer,
    print_cli_header,
    print_cli_section,
    run_batch_jobs,
    setup_device,
    spinner,
)
from fastfuncstuff.processing.reface import (
    REFACE_MODES,
    SHELL_NAME,
    TEMPLATE_HEAD_BRICK,
    TEMPLATE_NAME,
    TEMPLATE_WEIGHT_BRICK,
    apply_shell,
    shell_gain,
    shell_on_subject,
    widen_keep,
)
from fastfuncstuff.utils import REGISTRATION_TF32

_EPILOG = """\
One -mode_* flag is required, as for @afni_refacer_run.

Outputs (PREFIX's extension picks the format; .nii.gz by default):
  one mode:  PREFIX.nii.gz, and the shell on the subject grid as
             PREFIX.face.nii.gz (deface/reface) or PREFIX.face_plus.nii.gz
  -mode_all: PREFIX.deface / .reface / .reface_plus, and both shells

The default template and shell are the ones AFNI installs
(MNI152_2009_template_SSW.nii.gz, afni_refacer_shell_sym_1.0.nii.gz), found
beside the afni on $PATH, in $AFNI_ATLAS_PATH or ~/.afni/atlases.

Differences from @afni_refacer_run:
  - No 3dUnifize before the fit: lpa is a local correlation, which the
    bias field barely moves, and it costs 30 s.
  - No QC montage images.
"""


def parse_args(
    argv: list[str] | None = None, namespace: argparse.Namespace | None = None
) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_util_reface",
        description="Deface or reface a T1w through AFNI's template-space face shell "
        "(GPU port of @afni_refacer_run).",
        epilog=_EPILOG,
        formatter_class=FfsHelpFormatter,
    )
    io = parser.add_argument_group("input/output")
    io.add_argument("-input", default=None, help="T1w to deface (the first volume of a 4D)")
    io.add_argument(
        "-prefix", default=None, help="Output prefix (see Outputs below) [required unless -batch]"
    )
    add_batch_args(
        io,
        tool="ffs_util_reface",
        what="defacings",
        example="-input sub-01/anat/sub-01_T1w.nii.gz -prefix sub-01/anat/sub-01_acq-reface_T1w",
        skip_note="every output the mode writes",
    )
    modes = parser.add_argument_group("mode")
    mex = modes.add_mutually_exclusive_group()
    mode_help = {
        "deface": "Zero the face and ears (and everything outside the head)",
        "reface": "Replace the face and ears with the shell's synthetic face",
        "reface_plus": "Replace the face, ears and skull",
    }
    for m in REFACE_MODES:
        mex.add_argument(
            f"-mode_{m}", dest="mode", action="store_const", const=m, help=mode_help[m]
        )
    mex.add_argument("-mode_all", dest="mode", action="store_const", const="all", help="All three")

    ref = parser.add_argument_group("template and shell")
    ref.add_argument(
        "-template",
        default=None,
        help=f"Template with the skull on in sub-brick {TEMPLATE_HEAD_BRICK} and the "
        f"fit weight in {TEMPLATE_WEIGHT_BRICK} (default: AFNI's {TEMPLATE_NAME})",
    )
    ref.add_argument(
        "-shell",
        default=None,
        help=f"Refacer shell in the template's space (default: AFNI's {SHELL_NAME})",
    )

    al = parser.add_argument_group("alignment")
    al.add_argument("-cost", default="lpa", help="ffs_allineate cost for the template fit")
    al.add_argument(
        "-dof",
        type=int,
        choices=(9, 12),
        default=9,
        help="Fit shift+rotate+scale (9, AFNI's choice) or a full affine (12). "
        "12 matches the template better but its shear can bend the shell's inner "
        "edge toward the brain: on one T1 it took the back of the shell onto the "
        "skull's inner table, where 9 kept AFNI's clearance or more",
    )
    ref.add_argument(
        "-margin",
        type=float,
        default=0.0,
        metavar="MM",
        help="Safety knob for when the shell sits too close to the brain on some "
        "head: keep the subject's own voxels for the first MM of the shell's "
        "replaced region next to the brain, moving the shell's inner edge MM "
        "further out (e.g. 3). Leaves a thin band of the subject's inner skull "
        "in place, not face. 0 = AFNI's shell",
    )
    option_string_value(
        al.add_argument(
            "-allineate_opts",
            default="",
            metavar="'OPTS'",
            help="More ffs_allineate flags for the fit, quoted, e.g. '-slow'",
        )
    )
    al.add_argument(
        "-1Dmatrix_save",
        dest="matrix_save",
        default=None,
        help="Save the template->subject matrix (.aff12.1D, base = template)",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "-anonymize_output",
        action="store_true",
        help="Strip header extensions (the AFNI history holds paths and user@host) "
        "and the descrip/aux_file text fields",
    )
    out.add_argument(
        "-no_face", action="store_true", help="Do not write the face/face_plus shell volumes"
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv, namespace or argparse.Namespace())


def _modes(args: argparse.Namespace) -> tuple[str, ...]:
    return REFACE_MODES if args.mode == "all" else (args.mode,)


def _output_paths(args: argparse.Namespace) -> dict[str, str]:
    """``{what: path}`` for every file the run writes, keyed by mode or shell name."""
    p = parse_prefix(args.prefix)
    ext = p.nifti_ext
    paths: dict[str, str] = {}
    if args.mode == "all":
        for m in REFACE_MODES:
            paths[m] = f"{p.stem}.{m}{ext}"
    else:
        paths[args.mode] = f"{p.stem}{ext}"
    if not args.no_face:
        if args.mode in ("all", "reface_plus"):
            paths["face_plus"] = f"{p.stem}.face_plus{ext}"
        if args.mode in ("all", "reface", "deface"):
            paths["face"] = f"{p.stem}.face{ext}"
    return paths


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    outs = list(_output_paths(args).values())
    return outs + ([args.matrix_save] if args.matrix_save else [])


def _validate(args: argparse.Namespace) -> None:
    missing = [f"-{f}" for f in ("input", "prefix") if getattr(args, f) is None]
    if args.mode is None:
        missing.append("a -mode_* flag")
    if missing:
        raise SystemExit("ffs_util_reface: missing " + ", ".join(missing))


def _resolve(path: str | None, name: str, flag: str) -> str:
    if path is not None:
        return path
    found = find_afni_file(name)
    if found is None:
        raise SystemExit(
            f"ffs_util_reface: cannot find AFNI's {name} (looked beside afni on $PATH, in "
            f"$AFNI_ATLAS_PATH and ~/.afni/atlases); pass {flag} FILE"
        )
    return found


@lru_cache(maxsize=2)
def _load_reference(path: str) -> tuple[torch.Tensor, dict]:
    """A template or shell, read once per process: a -batch reuses it for every subject."""
    from fastfuncstuff.processing.io import load_image

    return load_image(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    if args.batch is not None or args.batch_run:
        run_batch_jobs(
            tool="ffs_util_reface",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=device,
            parse_line=lambda line, base: parse_args(shlex.split(line), base),
            defaults=args,
            dispatch=_dispatch_run,
            validate=_validate,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
            verb=args.verb,
        )
        return
    try:
        _validate(args)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    _dispatch_run(args, device)


def _dispatch_run(args: argparse.Namespace, device: torch.device) -> None:
    from fastfuncstuff.cli.allineate import build_align_config
    from fastfuncstuff.cli.allineate import parse_args as allineate_parse
    from fastfuncstuff.processing.affine import save_matrix_1D, voxel_matrix_to_dicom
    from fastfuncstuff.processing.allineate import allineate
    from fastfuncstuff.processing.io import load_image, save_image

    verb = args.verb
    t0 = time.time()
    if verb >= 1:
        print_cli_header("ffs_util_reface", "Deface / reface a T1w through a template shell")
    template_path = _resolve(args.template, TEMPLATE_NAME, "-template")
    shell_path = _resolve(args.shell, SHELL_NAME, "-shell")

    with spinner(f"Loading {Path(args.input).name}", enabled=verb >= 1):
        subject, sh = load_image(args.input)
    if subject.ndim == 4:
        subject = subject[0]
    subject = subject.to(device)
    with spinner("Loading template and shell", enabled=verb >= 1):
        template, th = _load_reference(template_path)
        shell, shh = _load_reference(shell_path)
    if template.ndim != 4 or template.shape[0] <= TEMPLATE_WEIGHT_BRICK:
        raise SystemExit(
            f"{template_path}: need sub-bricks {TEMPLATE_HEAD_BRICK} (head) and "
            f"{TEMPLATE_WEIGHT_BRICK} (weight); see -help"
        )
    head = template[TEMPLATE_HEAD_BRICK].to(device)
    if verb >= 1:
        print_cli_section("Inputs", leading_blank=False)
        print(f"  Input:    {args.input} {tuple(subject.shape)}")
        print(f"  Template: {template_path}")
        print(f"  Shell:    {shell_path}")

    # The fit goes through ffs_allineate's own parser, so -allineate_opts means
    # exactly what it means there and the defaults cannot drift apart. AFNI's
    # recipe: lpa, source automask, the template's weight brick as the cost mask.
    al_args = allineate_parse(
        [
            *("-base", template_path, "-source", args.input, "-prefix", "unused.nii"),
            *("-cost", args.cost, "-automask", "-noautoweight"),
            *(("-shift_rotate_scale",) if args.dof == 9 else ()),
            *("-base_mask", f"{template_path}[{TEMPLATE_WEIGHT_BRICK}]"),
            *("-verb", str(max(verb - 1, 0))),
            *shlex.split(args.allineate_opts),
        ]
    )
    config, _, _ = build_align_config(al_args, head, th, device)
    if verb >= 1:
        print_cli_section("Aligning the head to the template")
    t1 = time.time()
    matrix, warped = allineate(head, subject, config, base_header=th, source_header=sh)
    del warped
    dicom = voxel_matrix_to_dicom(matrix, th["affine"], sh["affine"])
    if verb >= 1:
        print(f"  {config.dof} fit, cost {config.cost}: {time.time() - t1:.1f}s")
    if args.matrix_save:
        save_matrix_1D(matrix, args.matrix_save, th["affine"], sh["affine"])

    shell_subj = shell_on_subject(
        shell.to(device), shh["affine"], dicom, sh["affine"], tuple(subject.shape)
    )
    gain = shell_gain(subject, shell_subj)
    # Tensor order is (z, y, x); NIfTI zooms are (x, y, z).
    zooms = tuple(float(z) for z in sh["header"].get_zooms()[:3])
    voxel_size = (zooms[2], zooms[1], zooms[0])
    # After the gain, so the synthetic face is scaled exactly as AFNI scales it.
    shell_subj = widen_keep(shell_subj, args.margin, voxel_size)

    # A short input comes back short, as @afni_refacer_run does, when it fits.
    short_in = np.dtype(sh["header"].get_data_dtype()) == np.int16
    paths = _output_paths(args)
    if verb >= 1:
        print_cli_section("Outputs")
    for mode in _modes(args):
        out = apply_shell(subject, shell_subj, mode, gain, voxel_size)
        dtype = np.int16 if short_in and float(out.max()) < 32000 else np.float32
        with spinner(f"Writing {Path(paths[mode]).name}", enabled=verb >= 1):
            save_image(out, paths[mode], sh, dtype=dtype, anonymize=args.anonymize_output)
        if verb >= 1:
            print(f"  {mode}: {paths[mode]}")
    for key, brick in (("face_plus", 0), ("face", 1)):
        if key in paths:
            save_image(shell_subj[brick], paths[key], sh, anonymize=args.anonymize_output)
            if verb >= 1:
                print(f"  {key}: {paths[key]}")
    if verb >= 1:
        print(f"  Shell gain: {gain:.4g}")
        print_cli_footer("ffs_util_reface", elapsed_seconds=time.time() - t0)


if __name__ == "__main__":
    main()
