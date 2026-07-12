"""``ffs_locomoco`` — residual non-linear motion correction via GPU optical flow.

Estimates the frame-to-frame residual displacement that rigid motion correction
leaves behind in single-echo EPI (mostly along the phase-encode axis) by treating
each slice's time course as a movie and running batched optical flow against a
reference frame. Writes:

  * ``{prefix}_warp``             — per-frame DICOM-mm warp for ``ffs_nwarp``; a
                                    5-D ``.nii.gz`` file (default) or a folder of
                                    numbered 4-D frames (``-warp_format folder``)
  * ``{prefix}_locomoco.nii.gz``  — the non-linear-motion-corrected series
  * ``{prefix}_locomoco_mean.nii.gz`` — temporal mean of the corrected series
                                    (with ``-save_mean``); a registration target
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

accuracy (trade time for exactness — all backends):

  -warp_interp bicubic   faithful resampler for the estimation iterations (removes
                         bilinear damping); helped flow on local-field tests, neutral
                         for phase/xcorr. Final warp is wsinc5 via ffs_nwarp regardless.
  -refine N              rebuild the reference from the corrected series (sharp, motion
                         removed) and re-register N more times; converges the template
                         out of its bias (measurably tighter frame alignment).
  -jacobian              conserve PE signal — stretched regions dim, compressed brighten
                         (J = det(I+∇disp)). Off by default; for data with real B0
                         pile-up, not a purely geometric shift.
  -workhard / -superhard presets over the above + more iters + denser search
                         (~3-5× / ~15-30× time). Explicit flags override the preset.

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
        "-raw_input",
        "-raw",
        default=None,
        help="Pre-moco RAW 4D series (same grid as -input). Giving this AND -moco_matrix "
        "switches on the rotation-aware estimator: residual distortion is measured in "
        "each frame's native orientation (PE genuinely axis-aligned) and reprojected to "
        "the reference frame by the head rotation, so a head tilt's off-PE-axis leakage "
        "is recovered. -input is still the moco'd series (used for the drift anchor, "
        "reusing moco's resample). Without these two flags the plain estimator runs.",
    )
    io.add_argument(
        "-moco_matrix",
        "-moco_mat",
        default=None,
        help="Per-volume moco matrices (.aff12.1D, one 12-value row per frame, DICOM — "
        "e.g. ffs_moco -1Dmatrix_save). Required with -raw_input for rotation-aware mode.",
    )
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
        default=None,
        help="[all] Reference: static mean | median | max | first | <frame index>, or "
        "PROGRESSIVE first_mean / first_median — frame t registers to the running "
        "mean/median of the already-corrected earlier frames (a bootstrapped template; "
        "frame 0 is the seed). Progressive modes are sequential and slower. 'max' takes "
        "the temporal maximum, which fills slices later frames rotate out of the FoV and "
        "is a high-signal target. Default: max for rotation-aware mode, mean otherwise.",
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

    acc = p.add_argument_group("Accuracy (trade time for exactness)")
    acc.add_argument(
        "-warp_interp",
        default="bilinear",
        choices=("bilinear", "bicubic"),
        help="[all] Resampler for the estimation iterations and the correction. "
        "'bicubic' removes the bilinear damping bias so the iterations converge to the "
        "true shift (biggest gain on smooth data); costs a little more per warp.",
    )
    acc.add_argument(
        "-refine",
        type=int,
        default=0,
        metavar="ROUNDS",
        help="[plain path only] Outer reference-refinement rounds. After the first estimate, "
        "rebuild the reference from the corrected series (motion removed → sharp) and "
        "re-register the original frames against it, converging the template out of "
        "its bias. 1–3 tightens the recovered values; 0 = off.",
    )
    acc.add_argument(
        "-jacobian",
        action="store_true",
        help="[plain path only] Modulate the corrected series by the PE Jacobian to conserve "
        "signal: where distortion stacked voxels (compression) the correction stretches "
        "them and dims accordingly, and vice-versa. Off by default (pure geometric "
        "correction); affects the corrected image, not the saved warp geometry.",
    )
    acc.add_argument(
        "-workhard",
        action="store_true",
        help="Preset: crank the accuracy knobs (bicubic warp, more iters, 1 refine "
        "round, denser search) for ~3–5× the time. Explicit flags still override.",
    )
    acc.add_argument(
        "-superhard",
        action="store_true",
        help="Preset: maximum accuracy (bicubic warp, many iters, 3 refine rounds, "
        "finest search) for ~15–30× the time. Explicit flags still override.",
    )

    rot = p.add_argument_group("Rotation-aware residual motion (needs -raw_input + -moco_matrix)")
    rot.add_argument(
        "-fuse",
        choices=("auto", "on", "off"),
        default="auto",
        help="How the absolute per-frame anchor pins the neighbour-differential chain: "
        "'auto' engages the per-voxel smoother only when the measured chain-vs-anchor "
        "drift exceeds -fuse_thresh; 'on' always; 'off' uses the chain alone (the anchor "
        "just fixes its per-voxel DC level). The drift is always printed as a diagnostic.",
    )
    rot.add_argument(
        "-fuse_thresh",
        type=float,
        default=0.05,
        metavar="VOX",
        help="Drift (RMS chain−anchor, in-brain, voxels) above which -fuse auto engages "
        "the smoother. Below it the sequential chain is trusted on its own.",
    )
    rot.add_argument(
        "-fuse_weight",
        type=float,
        default=1.0,
        metavar="W",
        help="Anchor weight in the smoother when engaged (relative to the unit-weighted "
        "differentials): higher = trust the absolute anchor more, less high-frequency detail.",
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
    out.add_argument(
        "-warp_format",
        "-warp-format",
        choices=["5d", "folder"],
        default="5d",
        help="Warp on-disk format: '5d' = one {prefix}_warp file (nx,ny,nz,T,3); "
        "'folder' = a {prefix}_warp/ directory of per-frame 4D files. Both are "
        "consumed by ffs_nwarp -nwarp. [default: %(default)s]",
    )
    out.add_argument("-no_corrected", action="store_true", help="Skip the corrected series.")
    out.add_argument(
        "-save_mean",
        "-save-mean",
        action="store_true",
        help="Also write the temporal mean of the corrected series "
        "({prefix}_locomoco_mean), e.g. for use as a registration target. "
        "Independent of -no_corrected.",
    )
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


# Presets bump the levers that helped on spatially-varying (local stretch/squish)
# field tests: bicubic warp (better for flow, neutral elsewhere), more convergence
# iterations, reference-refinement rounds, and denser search.
_PRESET_DEFAULTS = {
    "iters": 4,
    "levels": 3,
    "stride": 8,
    "xcorr_step": 0.5,
    "warp_interp": "bilinear",
    "refine": 0,
}
_PRESETS = {
    "workhard": {
        "iters": 8,
        "levels": 4,
        "stride": 4,
        "xcorr_step": 0.25,
        "warp_interp": "bicubic",
        "refine": 1,
    },
    "superhard": {
        "iters": 16,
        "levels": 5,
        "stride": 2,
        "xcorr_step": 0.25,
        "warp_interp": "bicubic",
        "refine": 3,
    },
}


def _apply_preset(args) -> str | None:
    """Fill accuracy knobs from -workhard/-superhard, but let explicit flags win.

    A knob still sitting at its documented default is taken from the preset; a knob
    the user set to a non-default value is left alone. Returns the preset name (or None).
    """
    name = "superhard" if args.superhard else "workhard" if args.workhard else None
    if name is None:
        return None
    for knob, val in _PRESETS[name].items():
        if getattr(args, knob) == _PRESET_DEFAULTS[knob]:  # user left it default → preset sets it
            setattr(args, knob, val)
    return name


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    preset = _apply_preset(args)

    # Rotation-aware mode (both raw + matrices) defaults its reference to the temporal
    # MAX (fills FoV dropout, high-signal anchor); the plain path stays on the mean.
    rotaware = args.raw_input is not None or args.moco_matrix is not None
    if args.ref is None:
        args.ref = "max" if rotaware else "mean"

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
    acc_desc = f"warp={args.warp_interp}, refine={args.refine}, jacobian={'on' if args.jacobian else 'off'}"
    if preset:
        acc_desc = f"[{preset}] " + acc_desc
    print(f"🌀 ffs_locomoco: {args.input}  shape={data.shape}  device={device}")
    print(
        f"   PE {args.pe_dir} (axes {pe_axes}), slice axis={slice_axis}, backend={args.backend}, "
        f"ref={args.ref}, do_blur={args.do_blur}mm (σ={smooth_sigma:.2f}vox), "
        f"{mode}, automask={mask_desc}"
    )
    print(f"   accuracy: {acc_desc}")

    if rotaware:
        if args.raw_input is None or args.moco_matrix is None:
            print(
                "❌ rotation-aware mode needs BOTH -raw_input and -moco_matrix.",
                file=sys.stderr,
            )
            return 2
        if dual:
            print(
                "❌ rotation-aware mode is single phase-encode only (one -pe_dir).", file=sys.stderr
            )
            return 2
        # The idea-1 accuracy knobs live on the moco-frame estimator; rotation-aware has
        # its own convergence path (neighbour differential + anchor smoother, tuned by
        # -fuse and -iters), so these are inert here. Warn rather than silently ignore.
        if args.refine or args.jacobian or preset:
            print(
                "ℹ️  rotation-aware: -refine, -jacobian, and the -workhard/-superhard "
                "reference-refinement rounds are moco-frame (idea-1) knobs and do NOT apply "
                "here — use -fuse / -fuse_weight and -iters to tune this path."
            )
        from fastfuncstuff.processing.affine import dicom_matrix_to_voxel
        from fastfuncstuff.processing.locomoco import estimate_residual_flow_rotaware

        with spinner(f"Loading raw {Path(args.raw_input).name}"):
            raw = np.asarray(load_nifti(args.raw_input).get_fdata(dtype=np.float32))
        if raw.shape != data.shape:
            print(
                f"❌ -raw_input {raw.shape} must match -input {data.shape} (same grid, same T).",
                file=sys.stderr,
            )
            return 2
        # Load per-volume DICOM matrices (reference→raw), derive the voxel-space stack.
        lines = [
            l.strip()
            for l in Path(args.moco_matrix).read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        m12 = np.array([[float(x) for x in ln.split()[:12]] for ln in lines], dtype=np.float32)
        if m12.shape != (data.shape[3], 12):
            print(
                f"❌ -moco_matrix has {m12.shape[0]} rows of {m12.shape[1] if m12.ndim > 1 else '?'}; "
                f"expected {data.shape[3]} rows of 12.",
                file=sys.stderr,
            )
            return 2
        mats_dicom = torch.eye(4).repeat(data.shape[3], 1, 1)
        mats_dicom[:, :3, :] = torch.from_numpy(m12.reshape(-1, 3, 4))
        mats_vox = dicom_matrix_to_voxel(mats_dicom, affine, affine)
        print(
            f"   rotation-aware: raw={args.raw_input}, fuse={args.fuse} (thresh {args.fuse_thresh} vox)"
        )
        result = estimate_residual_flow_rotaware(
            raw,
            data,
            mats_vox,
            mats_dicom,
            affine,
            pe_axis,
            slice_axis,
            ref_mode=args.ref,
            backend=args.backend,
            smooth_sigma=smooth_sigma,
            n_levels=args.levels,
            n_iters=args.iters,
            window_sigma=args.window,
            pe_only=pe_only,
            max_shift=args.max_shift,
            trial_step=args.xcorr_step,
            patch=args.patch,
            stride=args.stride,
            warp_interp=args.warp_interp,
            fuse=args.fuse,
            fuse_thresh=args.fuse_thresh,
            fuse_weight=args.fuse_weight,
            automask=automask,
            automask_dilate=args.automask_dilate,
            automask_sigma=args.automask_sigma,
            device=device,
        )
    else:
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
            warp_interp=args.warp_interp,
            refine_rounds=args.refine,
            jacobian=args.jacobian,
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
        as_5d = args.warp_format == "5d"
        with spinner(f"Writing {Path(stem).name}_warp{ext}"):
            warp_path = save_medic_warp(
                primary_disp,
                primary_axis,
                affine,
                stem,
                nii_ext=ext,
                as_5d=as_5d,
                extra_components=[(d, a) for a, d in rest],
            )
        axes_note = f"axes {[a for a, _ in comps]}" if dual else f"axis {primary_axis}"
        fmt_note = "5D DICOM-mm" if as_5d else "folder of 4D frames, DICOM-mm"
        print(f"  • warp ({fmt_note}, ffs_nwarp, {axes_note}): {warp_path}")

    if not args.no_corrected:
        corr_path = f"{stem}_locomoco{ext}"
        with spinner(f"Writing {Path(corr_path).name}"):
            save_nifti(result.corrected_series().numpy(), corr_path, affine=affine)
        print(f"  • corrected series: {corr_path}")

    if args.save_mean:
        mean_path = f"{stem}_locomoco_mean{ext}"
        # Temporal mean of the corrected series ((nx, ny, nz, T) -> mean over T).
        with spinner(f"Writing {Path(mean_path).name}"):
            save_nifti(result.corrected_series().mean(dim=-1).numpy(), mean_path, affine=affine)
        print(f"  • corrected mean: {mean_path}")

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
        with spinner(f"Writing {Path(movie_path).name}"):
            actual = _write_movie(frames, movie_path, args.fps, fmt)
        print(f"  • flow movie (circular-phase wheel): {actual}")

    print("✅ ffs_locomoco complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
