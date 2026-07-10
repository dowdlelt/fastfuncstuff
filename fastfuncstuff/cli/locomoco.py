"""``ffs_locomoco`` — residual non-linear motion correction via GPU optical flow.

Estimates the frame-to-frame residual displacement that rigid motion correction
leaves behind in single-echo EPI (mostly along the phase-encode axis) by treating
each slice's time course as a movie and running batched optical flow against a
reference frame. Writes:

  * ``{prefix}_warp.nii.gz``      — per-frame 5-D DICOM-mm warp for ``ffs_nwarp``
  * ``{prefix}_locomoco.nii.gz``  — the non-linear-motion-corrected series
  * ``{prefix}_flow.nii.gz``      — 4-D signed PE flow (voxels; sign = direction),
                                    scrub it like a timeseries
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


def _split_prefix(prefix: str) -> tuple[str, str]:
    """Split an output prefix into (stem, nii_ext); default .nii.gz if none given."""
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if prefix.endswith(ext):
            return prefix[: -len(ext)], ext
    return prefix, ".nii.gz"


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_locomoco",
        description="Residual non-linear (PE-axis) motion correction via GPU optical flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("Input/Output")
    io.add_argument("-input", "-i", required=True, help="4D motion-corrected NIfTI series")
    io.add_argument(
        "-prefix",
        "-o",
        required=True,
        help="Output stem. A trailing .nii.gz/.nii.zst/.nii is stripped and sets the "
        "output format for the NIfTI outputs (default .nii.gz).",
    )
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
    flow = p.add_argument_group("Optical flow")
    flow.add_argument(
        "-ref",
        default="mean",
        help="Reference: static mean | median | first | <frame index>, or PROGRESSIVE "
        "first_mean / first_median — frame t registers to the running mean/median of "
        "the already-corrected earlier frames (a bootstrapped template; frame 0 is the "
        "seed). Progressive modes are sequential and slower.",
    )
    flow.add_argument(
        "-do_blur",
        type=float,
        default=0.0,
        metavar="FWHM_MM",
        help="In-plane Gaussian blur (FWHM mm) applied to frames BEFORE flow only, "
        "for robustness on noisy data; 0 = off. Does not blur the corrected output.",
    )
    flow.add_argument(
        "-full_2d",
        action="store_true",
        help="Estimate full 2-D flow. Default is PE-only (1 DOF along the PE axis, "
        "more robust) — the correction and warp use only the PE component either way, "
        "so 2-D mainly enriches the direction movie.",
    )
    flow.add_argument(
        "-levels",
        type=int,
        default=3,
        help="Coarse-to-fine pyramid levels. The flow is solved on a stack of "
        "images each halved in size; the coarsest catches large displacements "
        "(1 px there = 2^(levels-1) px full-res), finer levels refine. More = "
        "handles bigger motion, but risks aliasing on small slices.",
    )
    flow.add_argument(
        "-iters",
        type=int,
        default=4,
        help="Refinement iterations per pyramid level: warp by the current flow, "
        "recompute the update, add it, repeat. More = better convergence for "
        "larger motion, at linear cost. 4 suits sub-voxel residual motion.",
    )
    flow.add_argument(
        "-window",
        type=float,
        default=2.0,
        metavar="SIGMA",
        help="Lucas-Kanade neighbourhood: gradients are pooled over a Gaussian of "
        "this sigma (voxels), assuming locally-constant flow. Larger = smoother, "
        "better-conditioned warp but blurs fine detail; smaller = sharper, noisier.",
    )

    mask = p.add_argument_group("Masking")
    mask.add_argument(
        "-no_automask",
        action="store_true",
        help="Disable brain-mask soft-gating of the flow. By default the flow is "
        "multiplied by a dilated, feathered automask of the time-mean so optical "
        "flow's wild guesses in the pure-noise air outside the head fade to zero "
        "(they don't corrupt the data, they just look bad and jump around).",
    )
    mask.add_argument(
        "-automask_dilate",
        type=int,
        default=4,
        metavar="VOX",
        help="Dilate the automask by this many voxels before feathering — a safety "
        "margin so real motion at the brain edge is kept.",
    )
    mask.add_argument(
        "-automask_sigma",
        type=float,
        default=3.0,
        metavar="SIGMA_VOX",
        help="Gaussian feather (voxels) on the dilated mask, so the flow decays "
        "smoothly to zero near the edge instead of at a hard boundary; 0 = hard edge.",
    )

    out = p.add_argument_group("Outputs")
    out.add_argument("-no_warp", action="store_true", help="Skip the per-frame warp file.")
    out.add_argument("-no_corrected", action="store_true", help="Skip the corrected series.")
    out.add_argument("-no_movie", action="store_true", help="Skip the flow movie.")
    out.add_argument(
        "-no_flow",
        action="store_true",
        help="Skip the 4-D signed PE flow map (voxels; sign = direction).",
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

    stem, ext = _split_prefix(args.prefix)

    img = load_nifti(args.input)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 4:
        print(f"❌ -input must be 4D, got shape {data.shape}", file=sys.stderr)
        return 2
    affine = img.affine.copy()

    # -do_blur is FWHM in mm (repo convention); convert to an in-plane voxel sigma
    # for the pre-flow Gaussian. In-plane voxel size = mean of the two non-slice axes.
    smooth_sigma = 0.0
    if args.do_blur > 0:
        vox = np.linalg.norm(affine[:3, :3], axis=0)  # per-axis mm
        in_plane = [a for a in (0, 1, 2) if a != slice_axis]
        inplane_mm = float(np.mean([vox[in_plane[0]], vox[in_plane[1]]]))
        smooth_sigma = (args.do_blur / 2.35482) / max(inplane_mm, 1e-6)

    pe_only = not args.full_2d
    automask = not args.no_automask
    mask_desc = (
        f"on (dilate {args.automask_dilate}, σ {args.automask_sigma:.1f} vox)"
        if automask
        else "off"
    )
    print(f"🌀 ffs_locomoco: {args.input}  shape={data.shape}  device={device}")
    print(
        f"   PE axis={pe_axis} ({args.pe_dir}), slice axis={slice_axis}, ref={args.ref}, "
        f"do_blur={args.do_blur}mm (σ={smooth_sigma:.2f}vox), "
        f"{'1-D PE' if pe_only else '2-D'} flow, automask={mask_desc}"
    )

    result = estimate_residual_flow(
        data,
        pe_axis,
        slice_axis,
        ref_mode=args.ref,
        smooth_sigma=smooth_sigma,
        n_levels=args.levels,
        n_iters=args.iters,
        window_sigma=args.window,
        pe_only=pe_only,
        automask=automask,
        automask_dilate=args.automask_dilate,
        automask_sigma=args.automask_sigma,
        device=device,
    )

    print("💾 Writing outputs...")
    if not args.no_warp:
        from fastfuncstuff.processing.medic import save_medic_warp

        warp_path = save_medic_warp(
            result.pe_displacement(), pe_axis, affine, stem, nii_ext=ext, as_5d=True
        )
        print(f"  • warp (5D DICOM-mm, ffs_nwarp): {warp_path}")

    if not args.no_corrected:
        corr_path = f"{stem}_locomoco{ext}"
        save_nifti(result.corrected_series().numpy(), corr_path, affine=affine)
        print(f"  • corrected series: {corr_path}")

    if not args.no_flow:
        flow_path = f"{stem}_flow{ext}"
        save_nifti(result.pe_displacement().numpy(), flow_path, affine=affine)
        print(
            f"  • signed PE flow 4D (voxels, ± = direction; scrub like a timeseries): {flow_path}"
        )

    if not args.no_movie:
        fmt = args.movie_format or ("mp4" if _find_ffmpeg() else "gif")
        frames = result.flow_movie(max_mag=args.flow_max)
        movie_path = f"{stem}_flow.{fmt}"
        actual = _write_movie(frames, movie_path, args.fps, fmt)
        print(f"  • flow movie (circular-phase wheel): {actual}")

    print("✅ ffs_locomoco complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
