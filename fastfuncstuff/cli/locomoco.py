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


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show each arg's default AND keep the epilog's hand-formatted layout."""


_EPILOG = """\
backends (all estimate the SAME residual PE-axis shift — pick on speed/quality;
numbers are mean |err| recovering known shifts on a 0.8 mm real brain):

  flow   pyramidal Lucas-Kanade optical flow. Most precise (~0.006 vox), slowest.
         reads: -full_2d -levels -iters -window
  phase  phase-correlation searchlight — shift from the FFT phase-ramp along PE.
         Fastest, near-flow accuracy on real tissue (~0.013 vox).
         reads: -patch -stride -iters -max_shift
  xcorr  magnitude cross-correlation searchlight — slide along PE, peak local corr.
         Robust, single-shot (~0.028 vox).
         reads: -window -max_shift -xcorr_step

which flag feeds which backend:

  flag           flow   phase  xcorr   meaning
  -ref            ·      ·      ·       reference frame (all)
  -do_blur        ·      ·      ·       pre-blur noisy frames (all)
  -full_2d        ·      -      -       2-D vs PE-only flow
  -levels         ·      -      -       optical-flow pyramid levels
  -iters          ·      ·      -       flow: LK passes / phase: warp-refine passes
  -window         ·      -      ·       flow: LK window / xcorr: searchlight radius
  -max_shift      -      ·      ·       search bound (voxels)
  -xcorr_step     -      -      ·       xcorr trial spacing (sub-voxel knob)
  -patch          -      ·      -       phase FFT patch side
  -stride         -      ·      -       phase patch spacing

tuning (turning the knobs):

  -window     up = smoother, more robust, but blurs fine local shifts; down =
              sharper, follows small structure, noisier. 2 is a good middle.
  -max_shift  set just above the biggest residual shift you expect (this data is
              sub- to a few voxels). Smaller = faster xcorr (fewer trials) + a
              tighter phase no-wrap band; too small clips real motion.
  -patch      bigger = less phase leakage per pass (needs fewer -iters), coarser
              field; smaller = finer field but wants more -iters. 16 default.
  -stride     smaller = denser, smoother field, more FFTs; ~patch/2 = NORDIC-style
              overlap. 8 default.
  -iters      more = better convergence for larger motion (flow AND phase), linear
              cost; xcorr ignores it (single-shot).
  -xcorr_step xcorr's sub-voxel knob (its answer to -iters): 0.5 default; 0.25 doubles
              the trials for a bit more precision. Peak is always 5-point-parabola fit.
  -levels     more pyramid levels handle bigger motion but risk aliasing on thin
              slices.

dual phase-encode (rare — e.g. 3-D EPI encoded on two in-plane axes):

  Give two -pe_dir (e.g. -pe_dir AP IS). Both in-plane axes are estimated and BOTH
  warp components saved; -slice_axis is forced to the third (un-encoded) axis so
  both PE axes lie in the slice plane (AP+IS -> slice on L-R). flow does this
  natively; phase reads one phase-ramp per axis; xcorr searches the axes separably
  (no O(trials²) grid). The single signed flow map splits into a magnitude
  (_flowmag) + angle (_flowang, degrees) pair, since a 2-D vector has no single
  signed scalar.

examples:

  # default: optical flow, PE-only, automask on
  ffs_locomoco -input moco.nii.gz -prefix sub -pe_dir AP

  # dual phase-encode: both AP and IS, slice forced to L-R
  ffs_locomoco -i moco.nii.gz -o sub -pe_dir AP IS -backend phase

  # phase backend, denser field + more refine passes
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -backend phase -patch 12 -stride 4 -iters 6

  # xcorr, tighter search + smaller searchlight for sharp local distortion
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -backend xcorr -max_shift 2 -window 1.5

  # progressive reference + blur first for noisy data
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -ref first_mean -do_blur 2
"""


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_locomoco",
        description="Residual non-linear (PE-axis) motion correction via GPU optical "
        "flow, phase-correlation, or cross-correlation searchlights. See the bottom of "
        "this help for which -flag applies to which -backend and how to tune each.",
        formatter_class=_HelpFormatter,
        epilog=_EPILOG,
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
        nargs="+",
        metavar="DIR",
        help="Phase-encode direction(s): AP/PA/LR/RL/IS/SI or an axis letter x/y/z. "
        "Motion is corrected along this axis. Give TWO (e.g. -pe_dir AP IS) for a "
        "dual-phase-encode acquisition — both in-plane axes are estimated and both "
        "warps saved (-slice_axis is then forced to the remaining, third axis).",
    )
    io.add_argument(
        "-slice_axis",
        "-slice",
        default="z",
        type=_axis_from_token,
        help="Through-plane (slice-select) axis to cut the movie along (x/y/z). Must "
        "differ from the PE axis; default z suits axial EPI. Ignored for dual -pe_dir "
        "(the slice axis is fixed to the one axis not phase-encoded).",
    )
    est = p.add_argument_group("Estimation — all backends")
    est.add_argument(
        "-backend",
        default="flow",
        choices=("flow", "phase", "xcorr"),
        help="Displacement estimator (all measure the same PE shift): 'flow' "
        "(default) pyramidal Lucas-Kanade optical flow — most precise, slowest; "
        "'phase' phase-correlation searchlight (FFT phase-ramp along PE) — fastest, "
        "near-flow accuracy; 'xcorr' magnitude cross-correlation searchlight (slide "
        "along PE, peak local correlation) — robust, single-shot. See epilog for the "
        "flag→backend map and tuning.",
    )
    est.add_argument(
        "-ref",
        default="mean",
        help="[all] Reference: static mean | median | first | <frame index>, or "
        "PROGRESSIVE first_mean / first_median — frame t registers to the running "
        "mean/median of the already-corrected earlier frames (a bootstrapped template; "
        "frame 0 is the seed). Progressive modes are sequential and slower.",
    )
    est.add_argument(
        "-do_blur",
        type=float,
        default=0.0,
        metavar="FWHM_MM",
        help="[all] In-plane Gaussian blur (FWHM mm) applied to frames BEFORE "
        "estimation only, for robustness on noisy data; 0 = off. All three backends "
        "tested noise-robust to ~15%%, so usually leave at 0. Never blurs the output.",
    )

    flow = p.add_argument_group("Optical-flow backend (-backend flow)")
    flow.add_argument(
        "-full_2d",
        action="store_true",
        help="[flow only] Estimate full 2-D flow. Default is PE-only (1 DOF along the "
        "PE axis, more robust) — the correction and warp use only the PE component "
        "either way, so 2-D mainly enriches the direction movie.",
    )
    flow.add_argument(
        "-levels",
        type=int,
        default=3,
        help="[flow only] Coarse-to-fine pyramid levels. The flow is solved on a stack "
        "of images each halved in size; the coarsest catches large displacements "
        "(1 px there = 2^(levels-1) px full-res), finer levels refine. More = handles "
        "bigger motion, but risks aliasing on thin slices.",
    )

    tune = p.add_argument_group("Shared tuning (applies to the backends tagged below)")
    tune.add_argument(
        "-iters",
        type=int,
        default=4,
        help="[flow, phase] Refinement iterations. flow: LK warp-and-update passes per "
        "pyramid level. phase: whole-field warp-and-re-read passes that cancel the "
        "single-patch leakage bias. More = better convergence for larger motion, "
        "linear cost. xcorr ignores it (single-shot).",
    )
    tune.add_argument(
        "-window",
        type=float,
        default=2.0,
        metavar="SIGMA",
        help="[flow, xcorr] Neighbourhood Gaussian sigma (voxels). flow: the LK "
        "gradient-pooling window (locally-constant-flow assumption). xcorr: the "
        "searchlight radius the local correlation is measured over. Larger = smoother, "
        "more robust, blurs fine local shifts; smaller = sharper, noisier. phase "
        "ignores it (its window is -patch).",
    )

    search = p.add_argument_group("Searchlight backends (-backend phase / xcorr)")
    search.add_argument(
        "-max_shift",
        type=float,
        default=3.0,
        metavar="VOX",
        help="[phase, xcorr] Largest PE shift to search for (voxels). Set just above "
        "the biggest residual shift you expect (sub- to a few voxels here). Smaller = "
        "faster xcorr (fewer trial offsets) and a tighter phase no-wrap band; too "
        "small clips real motion. flow ignores it (pyramid handles range).",
    )
    search.add_argument(
        "-xcorr_step",
        type=float,
        default=0.5,
        metavar="VOX",
        help="[xcorr only] Trial-offset spacing (voxels) of the correlation search — "
        "xcorr's sub-voxel knob, like -iters is for the others. The peak is fit by a "
        "5-point parabola either way; a finer step (e.g. 0.25) samples the curve more "
        "densely for a touch more accuracy at ~2× the trials. 0.5 is the sweet spot.",
    )
    search.add_argument(
        "-patch",
        type=int,
        default=16,
        help="[phase only] Side (voxels) of the square searchlight FFT'd along PE. "
        "Bigger = less boundary leakage per pass (needs fewer -iters), coarser field; "
        "smaller = finer field but wants more -iters.",
    )
    search.add_argument(
        "-stride",
        type=int,
        default=8,
        help="[phase only] Spacing (voxels) between patch centres; < patch gives "
        "NORDIC-style overlap. Smaller = denser, smoother field, more FFTs (slower); "
        "~patch/2 is a good overlap.",
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

    pe_axes = [resolve_pe_axis(d) for d in args.pe_dir]
    if len(pe_axes) > 2:
        print(f"❌ -pe_dir takes 1 or 2 directions, got {len(pe_axes)}.", file=sys.stderr)
        return 2
    dual = len(pe_axes) == 2
    slice_axis = args.slice_axis
    if dual:
        if pe_axes[0] == pe_axes[1]:
            print(f"❌ the two -pe_dir must be different axes, got {args.pe_dir}.", file=sys.stderr)
            return 2
        third = next(a for a in (0, 1, 2) if a not in pe_axes)  # the un-encoded axis
        if slice_axis != third:
            print(
                f"ℹ️  dual -pe_dir: forcing -slice_axis to {third} (the axis not phase-"
                f"encoded) so both PE axes lie in the slice plane.",
            )
        slice_axis = third
        pe_axis = pe_axes[0]  # representative; dual estimates both in-plane axes
    else:
        pe_axis = pe_axes[0]
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

    from fastfuncstuff.cli_utils import spinner

    with spinner(f"Loading {Path(args.input).name}"):
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
    if dual:
        mode = "2-D dual-PE"
    elif args.backend == "flow":
        mode = f"{'1-D PE' if pe_only else '2-D'} flow"
    else:
        mode = args.backend
    print(f"🌀 ffs_locomoco: {args.input}  shape={data.shape}  device={device}")
    print(
        f"   PE {args.pe_dir} (axes {pe_axes}), slice axis={slice_axis}, backend={args.backend}, "
        f"ref={args.ref}, do_blur={args.do_blur}mm (σ={smooth_sigma:.2f}vox), "
        f"{mode}, automask={mask_desc}"
    )

    result = estimate_residual_flow(
        data,
        pe_axis,
        slice_axis,
        ref_mode=args.ref,
        backend=args.backend,
        smooth_sigma=smooth_sigma,
        n_levels=args.levels,
        n_iters=args.iters,
        window_sigma=args.window,
        pe_only=pe_only,
        dual=dual,
        max_shift=args.max_shift,
        trial_step=args.xcorr_step,
        patch=args.patch,
        stride=args.stride,
        automask=automask,
        automask_dilate=args.automask_dilate,
        automask_sigma=args.automask_sigma,
        device=device,
    )

    print("💾 Writing outputs...")
    if not args.no_warp:
        from fastfuncstuff.processing.medic import save_medic_warp

        comps = result.warp_components()  # [(nifti_axis, disp), ...]
        (primary_axis, primary_disp), *rest = comps
        with spinner(f"Writing {Path(stem).name}_warp{ext}"):
            warp_path = save_medic_warp(
                primary_disp,
                primary_axis,
                affine,
                stem,
                nii_ext=ext,
                as_5d=True,
                extra_components=[(d, a) for a, d in rest],
            )
        axes_note = f"axes {[a for a, _ in comps]}" if dual else f"axis {primary_axis}"
        print(f"  • warp (5D DICOM-mm, ffs_nwarp, {axes_note}): {warp_path}")

    if not args.no_corrected:
        corr_path = f"{stem}_locomoco{ext}"
        with spinner(f"Writing {Path(corr_path).name}"):
            save_nifti(result.corrected_series().numpy(), corr_path, affine=affine)
        print(f"  • corrected series: {corr_path}")

    if not args.no_flow:
        if dual:
            # No single signed scalar holds a 2-D vector — split into magnitude + angle.
            mag_path, ang_path = f"{stem}_flowmag{ext}", f"{stem}_flowang{ext}"
            with spinner(f"Writing {Path(mag_path).name}"):
                save_nifti(result.flow_magnitude().numpy(), mag_path, affine=affine)
            with spinner(f"Writing {Path(ang_path).name}"):
                save_nifti(result.flow_angle().numpy(), ang_path, affine=affine)
            print(f"  • flow magnitude 4D (voxels): {mag_path}")
            print(f"  • flow angle 4D (degrees 0–360; direction of motion): {ang_path}")
        else:
            flow_path = f"{stem}_flow{ext}"
            with spinner(f"Writing {Path(flow_path).name}"):
                save_nifti(result.pe_displacement().numpy(), flow_path, affine=affine)
            print(
                f"  • signed PE flow 4D (voxels, ± = direction; scrub like a series): {flow_path}"
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
