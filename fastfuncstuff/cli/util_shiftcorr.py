"""CLI for partition-axis inter-echo shift correction (ffs_util_shiftcorr).

Command: ffs_util_shiftcorr (registered as entry point in pyproject.toml)

Usage:
    ffs_util_shiftcorr -input e1.nii e2.nii e3.nii -axis IS \\
        -echo_times 7.61 21.71 35.81 -ordering ascending -prefix sc.nii.gz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, spinner
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.locomoco import resolve_pe_axis
from fastfuncstuff.processing.shiftcorr import apply_shift, estimate_shifts, save_shift_tables

# Siemens phase images are stored as int16 over ±4096 -> radians.
_DEFAULT_PHASE_SCALE = math.pi / 4096.0


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show each arg's default AND keep the epilog's hand-formatted layout."""


_EPILOG = """\
what this corrects

  A 3-D-encoded volume is acquired in one shot, so a steady frequency change over
  that shot (drift, breathing, a warming gradient) translates the image rigidly
  along the SLOW phase-encode / partition axis. The translation grows with echo
  time, so every echo lands somewhere different:

      shift_e(t) = m(t) . TE_e     [voxels]

  With -echo_times, m(t) is fit per timepoint and applied THROUGH THE ORIGIN, so
  echo 1 is corrected too and the echo-common part (the fit's intercept) is left
  for rigid motion correction. Without -echo_times the raw cumulative inter-echo
  shifts are applied and echo 1 is the fixed reference.

outputs

  -prefix out.nii.gz     e1_out.nii.gz, e2_out.nii.gz, ... (and eN_out_phase...
                         when -phase was given)
  -save_shifts stem      stem_shifts_xcorr.1D    (T x E) raw cumulative estimate
                         stem_shifts_applied.1D  (T x E) what was corrected for
                         stem_corr.1D            (T x E-1) peak correlation, QC
                         stem_te_fit.1D          (T x 3) slope, intercept, drift Hz

examples

  # Multi-echo timeseries, ascending partition order, TEs in ms:
  ffs_util_shiftcorr -input e1.nii e2.nii e3.nii -axis IS \\
      -echo_times 7.61 21.71 35.81 -ordering ascending \\
      -prefix sc.nii.gz -save_shifts sc

  # Single 3-D volumes with phase (the reference script's native case):
  ffs_util_shiftcorr -input m_e1.nii m_e2.nii m_e3.nii \\
      -phase p_e1.nii p_e2.nii p_e3.nii -axis z \\
      -echo_times 7.61 21.71 35.81 -prefix sc.nii.gz
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_util_shiftcorr",
        description="Estimate and correct the TE-dependent partition-axis shift of "
        "multi-echo 3-D EPI/GRE, by whole-volume inter-echo cross-correlation and a "
        "sinc-exact Fourier shift.",
        epilog=_EPILOG,
        formatter_class=_HelpFormatter,
    )
    p.add_argument(
        "-input",
        "-i",
        dest="input_file",
        nargs="+",
        required=True,
        help="Magnitude images, ONE PER ECHO in echo order. Each may be a single "
        "3-D volume or a 4-D timeseries (shifts are then estimated per TR).",
    )
    p.add_argument(
        "-phase",
        dest="phase_file",
        nargs="+",
        default=None,
        help="Matching phase images (same count/order as -input). The correction is "
        "then applied to the complex data and both magnitude and phase are written.",
    )
    p.add_argument(
        "-phase_scale",
        "-phase-scale",
        dest="phase_scale",
        type=float,
        default=_DEFAULT_PHASE_SCALE,
        help="Multiplier converting stored phase values to radians.",
    )
    p.add_argument(
        "-axis",
        required=True,
        help="Partition (slow phase-encode) axis: a direction code AP/PA/LR/RL/IS/SI "
        "or an axis letter x/y/z (i/j/k). This is the axis the shift lives along.",
    )
    p.add_argument(
        "-echo_times",
        "-echo-times",
        dest="echo_times",
        nargs="+",
        type=float,
        default=None,
        help="Echo times in MILLISECONDS, one per -input. Enables the TE regression "
        "(recommended): the fitted line through the origin corrects every echo "
        "including the first. Omit to apply the raw cumulative shifts instead.",
    )
    p.add_argument(
        "-ordering",
        choices=["ascending", "descending", "unknown"],
        default="unknown",
        help="Partition view ordering. A known ordering fixes the sign of the drift "
        "and halves the search: ascending searches [-max,0], descending [0,+max].",
    )
    p.add_argument(
        "-max_shift",
        "-max-shift",
        dest="max_shift",
        type=float,
        default=5.0,
        help="Search half-range in voxels.",
    )
    p.add_argument(
        "-coarse_step",
        "-coarse-step",
        dest="coarse_step",
        type=float,
        default=0.25,
        help="Coarse grid spacing (voxels) for the correlation search.",
    )
    p.add_argument(
        "-fine_step",
        "-fine-step",
        dest="fine_step",
        type=float,
        default=0.005,
        help="Refinement grid spacing (voxels) around the coarse peak.",
    )
    p.add_argument(
        "-corr_extent",
        "-corr-extent",
        dest="corr_extent",
        choices=["full", "inner_half"],
        default="full",
        help="Which voxels the correlation sees. 'full' uses the whole volume, "
        "tapering only the outermost few partitions that a trial shift fills with "
        "replicated content. 'inner_half' reproduces the reference script's "
        "central-half crop, which exists there only because it does not pad before "
        "shifting — it assumes the anatomy is centred on the partition axis.",
    )
    p.add_argument(
        "-weight",
        choices=["none", "signal"],
        default="none",
        help="Voxel weighting for the correlation. 'none' is a plain Pearson r over "
        "the whole volume — the default, and sufficient because the patch IS the "
        "volume. 'signal' softly weights by mean echo-1 intensity, biasing toward "
        "brain without the hard edges a mask would introduce.",
    )
    p.add_argument(
        "-prefix",
        default=None,
        help="Output prefix; each echo is written with an eN_ prefix on the basename. "
        "Omit to estimate only (with -save_shifts).",
    )
    p.add_argument(
        "-save_shifts",
        "-save-shifts",
        dest="save_shifts",
        default=None,
        help="Stem for the shift/QC tables (see the output list above).",
    )
    p.add_argument("-device", default=None, help="Compute device (cuda/cpu/mps).")
    add_verbose_arg(p)
    return p.parse_args(argv)


def _select_device(arg_device: str | None) -> torch.device:
    """Resolve the -device flag (or auto-detect)."""
    if arg_device:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sibling(path: str, prefix: str) -> str:
    """Return ``path`` with ``prefix`` prepended to its basename (dir preserved)."""
    import os

    d, base = os.path.split(path)
    return os.path.join(d, prefix + base)


def _load_echo(mag_path: str, phase_path: str | None, phase_scale: float):
    """Load one echo as real magnitude or, with phase, as a complex volume."""
    with spinner(f"Loading {Path(mag_path).name}"):
        mag, hdr = load_image(mag_path)
    if phase_path is None:
        return mag, hdr, None
    with spinner(f"Loading {Path(phase_path).name}"):
        pha, pha_hdr = load_image(phase_path)
    if pha.shape != mag.shape:
        print(
            f"Error: phase {phase_path} shape {tuple(pha.shape)} does not match "
            f"magnitude {mag_path} {tuple(mag.shape)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    comp = torch.polar(mag.float(), pha.float() * phase_scale)
    return comp, hdr, pha_hdr


def _validate(args: argparse.Namespace) -> None:
    n_e = len(args.input_file)
    if n_e < 2:
        print(
            "Error: -input needs at least 2 echoes to estimate an inter-echo shift.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.phase_file is not None and len(args.phase_file) != n_e:
        print(
            f"Error: -phase has {len(args.phase_file)} files but -input has {n_e}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.echo_times is not None and len(args.echo_times) != n_e:
        print(
            f"Error: -echo_times has {len(args.echo_times)} values but -input has {n_e} echoes.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.prefix is None and args.save_shifts is None:
        print("Error: nothing to do — pass -prefix and/or -save_shifts.", file=sys.stderr)
        sys.exit(1)
    if args.max_shift <= 0:
        print("Error: -max_shift must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.fine_step <= 0 or args.coarse_step <= 0:
        print("Error: -coarse_step and -fine_step must be positive.", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for ffs_util_shiftcorr."""
    args = parse_args(argv)
    _validate(args)
    device = _select_device(args.device)
    verb = args.verb
    axis = resolve_pe_axis(args.axis)
    tes = np.asarray(args.echo_times, dtype=np.float64) if args.echo_times else None
    phase_files = args.phase_file or [None] * len(args.input_file)

    t0 = time.time()
    if verb >= 1:
        print(
            f"ffs_util_shiftcorr: device={device}, partition axis {args.axis} (voxel axis {axis})"
        )

    echoes, headers, phase_headers = [], [], []
    for mag_path, pha_path in zip(args.input_file, phase_files, strict=True):
        data, hdr, pha_hdr = _load_echo(mag_path, pha_path, args.phase_scale)
        if echoes and data.shape != echoes[0].shape:
            print(
                f"Error: {mag_path} shape {tuple(data.shape)} does not match the "
                f"first echo {tuple(echoes[0].shape)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        echoes.append(data)
        headers.append(hdr)
        phase_headers.append(pha_hdr)

    n_t = echoes[0].shape[0] if echoes[0].ndim == 4 else 1
    if verb >= 1:
        print(f"Input: {len(echoes)} echoes, {tuple(echoes[0].shape)} ({n_t} volume(s))")

    # The correlation runs on magnitude either way — the search is over geometry,
    # and phase would only add its own (wrapped) structure to the metric.
    mag_echoes = [e.abs() if e.is_complex() else e for e in echoes]
    est = estimate_shifts(
        mag_echoes,
        axis,
        tes=tes,
        ordering=args.ordering,
        max_shift=args.max_shift,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        weight=None if args.weight == "none" else args.weight,
        extent=args.corr_extent,
        device=device,
        verb=verb,
    )
    del mag_echoes

    if args.save_shifts is not None:
        save_shift_tables(est, args.save_shifts, tes, verb)

    if args.prefix is not None:
        for i, data in enumerate(echoes):
            tag = f"e{i + 1}_"
            out = apply_shift(data, est.applied[:, i], axis, device=device, disable_pbar=verb == 0)
            mag_path = parse_prefix(_sibling(args.prefix, tag)).as_file()
            save_image(out.abs() if out.is_complex() else out, mag_path, header_info=headers[i])
            if verb >= 1:
                print(f"Saved: {mag_path}")
            if out.is_complex():
                pha_path = parse_prefix(_sibling(args.prefix, f"{tag}phase_")).as_file()
                save_image(
                    torch.angle(out) / args.phase_scale,
                    pha_path,
                    header_info=phase_headers[i] or headers[i],
                )
                if verb >= 1:
                    print(f"Saved: {pha_path}")
            del out

    if verb >= 1:
        print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
