"""``ffs_locomoco`` — residual non-linear motion correction via GPU optical flow.

Estimates the frame-to-frame residual displacement that rigid motion correction
leaves behind in single-echo EPI (mostly along the phase-encode axis) by treating
each slice's time course as a movie and running batched optical flow against a
reference frame. Writes:

  * ``{prefix}_warp.nii.gz``      — per-frame 5-D DICOM-mm warp for ``ffs_nwarp``
  * ``{prefix}_locomoco.nii.gz``  — the non-linear-motion-corrected series
  * ``{prefix}_flowdir.nii.gz``   — 4-D per-voxel flow direction (deg 0–360),
                                    scrub it like a timeseries with a cyclic LUT
  * ``{prefix}_flowmag.nii.gz``   — 4-D per-voxel flow magnitude (voxels)
  * ``{prefix}_flow.mp4``         — a contact-sheet movie of the flow, colored by
                                    the circular-phase wheel (hue = direction).
                                    mp4 via system ffmpeg if present, else gif.

See ``processing/locomoco.py`` for the method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _axis_from_token(tok: str) -> int:
    m = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    if tok not in m:
        raise argparse.ArgumentTypeError(f"axis must be x/y/z or 0/1/2, got '{tok}'")
    return m[tok]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_locomoco",
        description="Residual non-linear (PE-axis) motion correction via GPU optical flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("Input/Output")
    io.add_argument("-input", "-i", required=True, help="4D motion-corrected NIfTI series")
    io.add_argument("-prefix", "-o", required=True, help="Output stem (no extension)")
    io.add_argument(
        "-pe_dir",
        "-pe",
        required=True,
        help="Phase-encode direction: AP/PA/LR/RL/IS/SI, or an axis letter x/y/z. "
        "The residual motion is corrected along this axis.",
    )
    io.add_argument(
        "-slice_axis",
        "-slice",
        default="z",
        type=_axis_from_token,
        help="Through-plane (slice-select) axis to cut the movie along (x/y/z). "
        "Must differ from the PE axis; default z suits axial EPI.",
    )
    io.add_argument("-ext", default=".nii.gz", choices=(".nii.gz", ".nii"), help="Output extension")

    flow = p.add_argument_group("Optical flow")
    flow.add_argument(
        "-ref",
        default="mean",
        help="Reference frame: mean | median | first | <frame index>.",
    )
    flow.add_argument(
        "-smooth",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian blur (voxels) applied before flow only (robustness); 0 = off.",
    )
    flow.add_argument(
        "-pe_only",
        action="store_true",
        help="Constrain the flow to the PE axis (1 DOF, more robust) instead of full 2-D.",
    )
    flow.add_argument("-levels", type=int, default=3, help="Pyramid levels.")
    flow.add_argument("-iters", type=int, default=4, help="Warping iterations per level.")
    flow.add_argument(
        "-window",
        type=float,
        default=2.0,
        metavar="SIGMA",
        help="Gaussian window sigma (voxels) for the Lucas-Kanade normal equations.",
    )

    out = p.add_argument_group("Outputs")
    out.add_argument("-no_warp", action="store_true", help="Skip the per-frame warp file.")
    out.add_argument("-no_corrected", action="store_true", help="Skip the corrected series.")
    out.add_argument("-no_movie", action="store_true", help="Skip the flow movie.")
    out.add_argument(
        "-no_flowmap",
        action="store_true",
        help="Skip the 4-D flow direction (deg) + magnitude (vox) NIfTIs.",
    )
    out.add_argument(
        "-movie_format",
        default=None,
        choices=("gif", "mp4"),
        help="Flow-movie container. Default: mp4 if a system ffmpeg is on PATH "
        "(small h264), else gif.",
    )
    out.add_argument("-fps", type=int, default=10, help="Flow-movie frame rate.")
    out.add_argument(
        "-flow_max",
        type=float,
        default=None,
        metavar="VOX",
        help="Fix the magnitude→color scaling (voxels); default = 99th percentile.",
    )

    hw = p.add_argument_group("Hardware")
    hw.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto).")
    return p


def _find_ffmpeg() -> str | None:
    """System ffmpeg binary path, if available (for small/fast h264 mp4)."""
    import shutil

    return shutil.which("ffmpeg")


def _write_mp4_ffmpeg(frames: np.ndarray, path: str, fps: int, ffmpeg: str) -> None:
    """Pipe RGB frames straight to the system ffmpeg → h264 mp4 (no pip deps)."""
    import subprocess

    # h264 + yuv420p needs even dimensions; pad the sheet if odd.
    t, h, w, _ = frames.shape
    ph, pw = h + (h % 2), w + (w % 2)
    if (ph, pw) != (h, w):
        padded = np.full((t, ph, pw, 3), 255, np.uint8)
        padded[:, :h, :w] = frames
        frames = padded
        h, w = ph, pw
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        path,
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])


def _write_movie(frames: np.ndarray, path: str, fps: int, fmt: str) -> str:
    """Write ``(T, H, W, 3)`` uint8 frames as a movie; return the actual path."""
    if fmt == "mp4":
        ffmpeg = _find_ffmpeg()
        if ffmpeg is not None:
            _write_mp4_ffmpeg(frames, path, fps, ffmpeg)
            return path
        gif_path = str(Path(path).with_suffix(".gif"))
        print(f"  ⚠️  no ffmpeg on PATH; writing GIF instead: {gif_path}")
        path = gif_path

    import imageio.v2 as imageio

    imageio.mimwrite(path, list(frames), duration=1000.0 / max(fps, 1), loop=0)
    return path


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.processing.locomoco import estimate_residual_flow, resolve_pe_axis

    pe_axis = resolve_pe_axis(args.pe_dir)
    slice_axis = args.slice_axis
    if pe_axis == slice_axis:
        print(
            f"❌ PE axis ({pe_axis}) and -slice_axis ({slice_axis}) coincide. The PE "
            "direction must lie in the slice plane — pick a different -slice_axis.",
            file=sys.stderr,
        )
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_nifti(args.input)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 4:
        print(f"❌ -input must be 4D, got shape {data.shape}", file=sys.stderr)
        return 2
    affine = img.affine.copy()

    print(f"🌀 ffs_locomoco: {args.input}  shape={data.shape}  device={device}")
    print(
        f"   PE axis={pe_axis} ({args.pe_dir}), slice axis={slice_axis}, "
        f"ref={args.ref}, smooth={args.smooth}, {'1-D PE' if args.pe_only else '2-D'} flow"
    )

    result = estimate_residual_flow(
        data,
        pe_axis,
        slice_axis,
        ref_mode=args.ref,
        smooth_sigma=args.smooth,
        n_levels=args.levels,
        n_iters=args.iters,
        window_sigma=args.window,
        pe_only=args.pe_only,
        device=device,
    )

    ext = args.ext
    print("💾 Writing outputs...")
    if not args.no_warp:
        from fastfuncstuff.processing.medic import save_medic_warp

        warp_path = save_medic_warp(
            result.pe_displacement(), pe_axis, affine, args.prefix, nii_ext=ext, as_5d=True
        )
        print(f"  • warp (5D DICOM-mm, ffs_nwarp): {warp_path}")

    if not args.no_corrected:
        corr_path = f"{args.prefix}_locomoco{ext}"
        save_nifti(result.corrected_series().numpy(), corr_path, affine=affine)
        print(f"  • corrected series: {corr_path}")

    if not args.no_flowmap:
        dir_path = f"{args.prefix}_flowdir{ext}"
        save_nifti(result.flow_direction_deg().numpy(), dir_path, affine=affine)
        print(f"  • flow direction 4D (deg 0–360, scrub like a timeseries): {dir_path}")
        mag_path = f"{args.prefix}_flowmag{ext}"
        save_nifti(result.flow_magnitude().numpy(), mag_path, affine=affine)
        print(f"  • flow magnitude 4D (voxels): {mag_path}")

    if not args.no_movie:
        fmt = args.movie_format or ("mp4" if _find_ffmpeg() else "gif")
        frames = result.flow_movie(max_mag=args.flow_max)
        movie_path = f"{args.prefix}_flow.{fmt}"
        actual = _write_movie(frames, movie_path, args.fps, fmt)
        print(f"  • flow movie (circular-phase wheel): {actual}")

    print("✅ ffs_locomoco complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
