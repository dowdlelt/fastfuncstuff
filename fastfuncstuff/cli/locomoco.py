"""``ffs_locomoco`` — residual non-linear motion correction via GPU optical flow.

Estimates the frame-to-frame residual displacement that rigid motion correction
leaves behind in EPI — the part that lives along the encode axes and so is not a
rigid body move — by treating the time course as a movie and registering each frame
to a reference with batched optical flow, phase correlation, cross correlation or
qwarp patches.

Up to TWO encode axes are corrected:

  * ``-pe_dir1`` the PRIMARY phase encode. In-plane distortion, NOT echo-time
    dependent (every echo shifts alike). Slicewise unless ``-is_3depi``.
  * ``-pe_dir2`` the PARTITION / 2nd phase encode of a 3-D EPI. Echo-time DEPENDENT
    (echo ``e`` shifts by ``TE_e/TE_1`` times the shared field).

Given both, they are solved simultaneously and independently — no ratio between them
is assumed, since the two artifacts plausibly have different physical sources — and
their measured relationship is written out as a diagnostic rather than presumed.

Writes the corrected series, the equivalent per-frame warp for ``ffs_nwarp``, a
signed flow map per encode axis, and (for two axes) the separability and coupling
diagnostics. Run ``ffs_locomoco -h`` for the case table and the full output guide;
see ``processing/locomoco.py`` for the method.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_device_arg,
    collect_batch_jobs,
    print_cli_footer,
    print_cli_header,
    print_cli_section,
    print_cli_subsection,
    run_batch_jobs,
    setup_device,
)
from fastfuncstuff.utils import REGISTRATION_TF32


def _axis_from_token(tok: str) -> int:
    m = {"x": 0, "y": 1, "z": 2, "i": 0, "j": 1, "k": 2, "0": 0, "1": 1, "2": 2}
    key = tok.strip().lstrip("-").lower()
    if key not in m:
        raise argparse.ArgumentTypeError(f"axis must be x/y/z, i/j/k or 0/1/2, got '{tok}'")
    return m[key]


def _split_prefix(prefix: str) -> tuple[str, str]:
    """Split an output prefix into (stem, nii_ext); default .nii.gz if none given."""
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if prefix.endswith(ext):
            return prefix[: -len(ext)], ext
    return prefix, ".nii.gz"


_EPILOG = """\
WHICH CASE ARE YOU IN?  — set the encode axes; the rest has working defaults.

  what you are correcting                        flags
  ----------------------------------------------------------------------------
  2-D multi-slice, primary-PE distortion         -pe_dir1 AP
  2-D multi-slice MULTI-ECHO, primary PE         -pe_dir1 AP -me_flat_scaling
      (one shift per slice; every echo sees          -echo_times ...   (2+ -input)
       the same one, so the echoes are extra
       evidence for it -- the MEDIC-comparable
       case, from magnitude alone)
  3-D EPI, primary-PE distortion                 -pe_dir1 AP -is_3depi
  3-D multi-echo, primary-PE distortion          -pe_dir1 AP -me_3depi -me_flat_scaling
      (same shift on every echo)                     -echo_times ...
  3-D EPI, partition wiggle, single echo         -pe_dir2 IS
  3-D multi-echo, partition wiggle               -pe_dir2 IS -me_3depi -echo_times ...
      (shift scales with TE)
  3-D EPI, BOTH, single echo                     -pe_dir1 AP -pe_dir2 IS
  3-D multi-echo, BOTH                           -pe_dir1 AP -pe_dir2 IS -me_3depi
      (flat on PE, TE-scaled on partition)           -echo_times ...

  Multi-echo is triggered by passing several -input with -echo_times. 2-D multi-slice
  is the default there, exactly as on the single-echo path; -me_3depi / -is_3depi opts
  into the 3-D solve. Any echo count from 2 upward works on every path.

THE TWO ENCODE AXES

  -pe_dir1  PRIMARY phase encode (a.k.a. -pe_dir / -pe). In-plane distortion.
            NOT echo-time dependent — every echo shifts by the same amount.
            Slicewise unless -is_3depi.
  -pe_dir2  PARTITION / 2nd phase encode (a.k.a. -partition_dir). 3-D EPI only,
            so it implies -is_3depi. Echo-time DEPENDENT — echo e shifts by
            TE_e/TE_1 times the shared field. Can be given ALONE (e.g. MEDIC or a
            fieldmap already fixed the primary axis), or with -pe_dir1.

  Given both, the two fields are solved SIMULTANEOUSLY and INDEPENDENTLY. No ratio
  between them is assumed, because the two artifacts plausibly arise from different
  physical sources. Whether they nonetheless move together is something this tool
  MEASURES rather than presumes — see _locomoco_coupling.txt below.

  How well the two axes can be told apart differs by case. MULTI-ECHO is the easy
  one: the axes scale differently with TE (primary PE flat, partition TE-scaled), so
  the echo axis itself separates them — which is also why -me_flat_scaling with two
  axes warns: it makes both laws identical and throws that advantage away.
  SINGLE-ECHO has only image structure to go on — the two shifts separate where the
  local neighbourhood holds edges of more than one orientation, and go ambiguous
  along a locally straight edge. _locomoco_sep maps exactly that, per voxel
  (1 = cleanly separable, 0 = ambiguous). Read the two flow maps with it open.

  `-pe_dir AP IS` (two values) is shorthand for `-pe_dir1 AP -pe_dir2 IS`. Unlike
  the explicit -pe_dir2 it does NOT imply 3-D, so an existing 2-D multi-slice
  dual-PE command line keeps its old meaning.

READING THE OUTPUTS  (3D = one value/voxel, 4D = a time series)

 the deliverable —

  _locomoco.nii.gz   (4D)  THE CORRECTED SERIES. The residual non-linear motion that
      rigid moco left behind is resampled out. This is what you want if you just want
      corrected images. With -jacobian the intensities are also signal-conserved
      (stretched regions dim, compressed brighten); without it, geometry only.
      Skip with -no_corrected.

  _locomoco_mean.nii.gz  [-save_mean]  Temporal mean of the corrected series — a
      sharp, motion-reduced registration target.

  _locomoco_max / _min.nii.gz  [-save_max / -save_min]  Coverage images: max is the
      union of every voxel ever imaged (edges the mean dims because motion took them
      out of the FoV), min is 0 wherever any single volume lost the voxel.

 the transform —

  _warp / _warp.nii.gz     The SAME correction as a per-frame DICOM-mm displacement
      field for ffs_nwarp, instead of pre-resampled data. Use it to fold this step
      into ONE interpolation with your other transforms, e.g.
      ffs_nwarp -nwarp 'sub_warp moco.aff12.1D' = rigid-then-nonlinear in a single
      resample (less blur than applying _locomoco to already-resampled data). 5-D
      (nx,ny,nz,T,3) by default, or per-frame 4-D files with -warp_format folder.
      With two encode axes both components ride in the one 5-D file. You want EITHER
      this or _locomoco, rarely both. Skip with -no_warp.

 the diagnostics —

  _flow.nii.gz       (4D)  SIGNED residual displacement per frame, in VOXELS; sign =
      direction along the encode axis. This is literally how much motion rigid moco
      MISSED, per voxel per frame. Scrub it like a series.
      QC: it should be spatially STRUCTURED (largest at tissue/air boundaries and
      where PE distortion lives), coherent frame-to-frame, and ~0 in static tissue —
      not salt-and-pepper. Big coherent values = real residual motion caught;
      noise-like everywhere = little to correct (or SNR too low to trust). It is NOT
      the warp (voxels vs DICOM-mm) — don't feed it to ffs_nwarp. Skip with -no_flow.

  _flow_pe1 / _flow_pe2.nii.gz  (4D)  With TWO encode axes, _flow splits into one
      signed map per axis: pe1 = primary phase encode, pe2 = partition. Two signed
      maps rather than a magnitude/angle pair, because these are two distinct
      artifacts along two named axes, not two halves of one vector.

  _locomoco_sep.nii.gz  (4D)  [two axes, -backend flow]  Per-voxel SEPARABILITY of
      the two axes: 1 where their gradients are orthogonal over the pooling window
      (the split between axes is well determined), 0 on a straight edge (the split is
      arbitrary and the regulariser picked it). Low sep is not a failure — it is the
      map telling you where not to believe the per-axis split.

  _locomoco_coupling.txt / _coupling_r.nii.gz / _coupling_r.1D  [two axes]
      THE COUPLING MEASUREMENT. Correlation r between the two fields, the free
      least-squares ratio kappa (d2 ~ kappa*d1) with its R2, plus r per voxel (3D)
      and per frame (1D). High |r| with high kappa R2 = evidence the primary-PE and
      partition wiggles share one off-resonance source seen through two effective
      dwell times; low |r| = separate mechanisms. Computed only over voxels that
      moved AND were separable, so it reports a finding rather than the regulariser.

  _taskr_pe1.nii.gz / _taskrms_pe1.nii.gz / _taskr_data.nii.gz +
  _locomoco_taskcoupling.txt   [-events]  IS THE FIELD READING BOLD AS MOTION?
      Every backend closes a brightness-constancy data term, so a strong block design
      hands it an intensity change on the very edges it tracks. _taskr is the SIGNED
      partial correlation between the field and each condition (conditions on the 4th
      axis); _taskrms is the task-explained part in VOXELS. Both are DESCRIPTIVE -- a
      few blocks is ~2 degrees of freedom, so no surrogate can make one voxel's r
      significant, and the report says so rather than pretending otherwise.

      THE VERDICT IS kappa, in the report: a brightness-constancy estimator explains
      an intensity change dI by a shift d with g*d = dI, so if it is absorbing BOLD
      then (PE gradient x beta_field) should EQUAL beta_data across voxels. kappa near
      +/-1 with a real R2 = contamination; near 0 = the field's task response is not
      the BOLD response, i.e. real task-correlated motion you must NOT remove. Unlike
      a correlation, kappa survives negative BOLD and the gradient flipping across an
      edge -- both cancel in the ratio.

      Add -detask to ACT on it: the task-locked part is removed from the field and
      every output (warp, corrected series, flow, PCs, movie) is derived from the
      cleaned field, with the removed part written as _flow_*_taskpart. Drift is fitted
      but NOT subtracted -- a drifting displacement is real motion. The diagnosis above
      always measures the ORIGINAL field, so the report and the fix read as one story.

  _flow.mp4 / .gif   THE QUICK-LOOK — contact-sheet movie of _flow, colored by a
      circular-phase wheel (hue = direction, brightness = magnitude). Fastest QC
      there is: coherent within-brain flow pulsing with the time course vs. random
      speckle. mp4 via system ffmpeg if present, else gif. Skip with -no_movie.

  _locomoco_pcs.1D   [-want_pcs N]  DENOISING REGRESSORS — top-N temporal PCs of the
      warp (unit variance; variance-explained in the header). Add them as nuisance
      columns in your GLM.

  legacy 2-D dual-PE (two in-plane axes on multi-slice data): the two components ARE
      one in-plane vector there, so they are written as _flowmag.nii.gz (voxels) +
      _flowang.nii.gz (degrees 0-360) instead of _flow_pe1/_flow_pe2.

  Note: -backend flow/phase/xcorr changes precision and speed, not WHICH files you
  get. -jacobian only changes _locomoco intensities, never the warp/flow geometry.

BACKENDS  (all estimate the SAME residual shift — pick on speed/quality; numbers are
mean |err| recovering known shifts on a 0.8 mm real brain)

  flow   pyramidal Lucas-Kanade optical flow. Most precise (~0.006 vox), slowest.
         The only backend with a 3-D two-axis joint solve and a _sep map.
         reads: -full_2d -levels -iters -window
  phase  phase-correlation searchlight — shift from the FFT phase-ramp along PE.
         Fastest, near-flow accuracy on real tissue (~0.013 vox). 2-D only.
         reads: -patch -stride -iters -max_shift
  xcorr  magnitude cross-correlation searchlight — slide along PE, peak local corr.
         Robust, single-shot (~0.028 vox). Two axes are searched separably.
         reads: -window -max_shift -xcorr_step
  qwarp  fine nonlinear patches own the whole field. Two encode axes are solved as
         ONE coupled Gauss-Newton system per patch (not two independent fits).
         reads: -qwarp_minpatch -qwarp_levels -qwarp_iters -qwarp_cost -qwarp_optimizer

which flag feeds which backend:

  flag           flow   phase  xcorr   meaning
  -ref            *      *      *       reference frame (all)
  -do_blur        *      *      *       pre-blur noisy frames (all)
  -hpf_spatial    *      *      *       estimate on spatial high-pass (all, experimental)
  -match          *      *      -       gain-invariant estimation (xcorr already is)
  -full_2d        *      -      -       2-D vs PE-only flow
  -levels         *      -      -       optical-flow pyramid levels
  -iters          *      *      -       flow: LK passes / phase: warp-refine passes
  -window         *      -      *       flow: LK window / xcorr: searchlight radius
  -max_shift      -      *      *       search bound (voxels)
  -xcorr_step     -      -      *       xcorr trial spacing (sub-voxel knob)
  -patch          -      *      -       phase FFT patch side
  -stride         -      *      -       phase patch spacing

TUNING  (turning the knobs)

  -window     up = smoother, more robust, but blurs fine local shifts; down =
              sharper, follows small structure, noisier. 2 is a good middle.
  -max_shift  set just above the biggest residual shift you expect (this data is
              sub- to a few voxels). Smaller = faster xcorr (fewer trials) + a
              tighter phase no-wrap band; too small clips real motion. With two
              axes it is also the flow solver's trust region.
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

ACCURACY  (trade time for exactness — all backends)

  -warp_interp bicubic   faithful resampler for the estimation iterations (removes
                         bilinear damping); helped flow on local-field tests, neutral
                         for phase/xcorr. Final warp is wsinc5 via ffs_nwarp regardless.
                         lanczos is 1-D only, so it is unavailable with two axes.
  -refine N              rebuild the reference from the corrected series (sharp, motion
                         removed) and re-register N more times; converges the template
                         out of its bias (measurably tighter frame alignment).
  -jacobian              conserve PE signal — stretched regions dim, compressed brighten
                         (J = det(I+grad disp)). Off by default; for data with real B0
                         pile-up, not a purely geometric shift.
  -workhard / -superhard presets over the above + more iters + denser search
                         (~3-5x / ~15-30x time). Explicit flags override the preset.

EXAMPLES

  # 2-D multi-slice, optical flow, PE-only, automask on
  ffs_locomoco -input moco.nii.gz -prefix sub -pe_dir1 AP

  # 3-D EPI: primary PE and partition solved together, single echo
  ffs_locomoco -i moco.nii.gz -o sub -pe_dir1 AP -pe_dir2 IS -refine 2

  # partition only — MEDIC already corrected the primary axis
  ffs_locomoco -i medic_out.nii.gz -o sub -pe_dir2 IS

  # 2-D multi-slice multi-echo: one per-slice PE field, pooled over echoes
  ffs_locomoco -i e1.nii.gz e2.nii.gz e3.nii.gz -o sub -pe_dir1 AP \\
      -echo_times 12 30 48 -me_flat_scaling -refine 2

  # multi-echo 3-D EPI, TE-scaled partition wiggle
  ffs_locomoco -i e1.nii.gz e2.nii.gz e3.nii.gz -o sub -pe_dir2 IS \\
      -me_3depi -echo_times 12 30 48

  # multi-echo, BOTH axes — the well-posed case: flat on PE, TE-scaled on partition
  ffs_locomoco -i e1.nii.gz e2.nii.gz e3.nii.gz -o sub -pe_dir1 AP -pe_dir2 IS \\
      -me_3depi -echo_times 12 30 48 -refine 2

  # phase backend, denser field + more refine passes
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -backend phase -patch 12 -stride 4 -iters 6

  # xcorr, tighter search + smaller searchlight for sharp local distortion
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -backend xcorr -max_shift 2 -window 1.5

  # progressive reference + blur first for noisy data
  ffs_locomoco -i moco.nii.gz -o sub -pe AP -ref first_mean -do_blur 2

  # two axes with a fine nonlinear polish (Gauss-Newton is the default optimizer)
  ffs_locomoco -i moco.nii.gz -o sub -pe_dir1 AP -pe_dir2 IS -final_qwarp

  # let qwarp own the whole two-axis field instead of the searchlight
  ffs_locomoco -i moco.nii.gz -o sub -pe_dir1 AP -pe_dir2 IS \\
      -backend qwarp -qwarp_levels 4 -qwarp_minpatch 9
"""


# Appended to every flag that accepts one value per encode axis. One sentence, written
# once: two-axis runs would otherwise need a separate -pe2_ twin of each of these flags,
# and the help would say the same thing six times.
PERAXIS = (
    "\nTakes 1 or 2 values: one applies to BOTH encode axes, two set the primary PE and "
    "the partition separately (a 3-D EPI's partition field is small, local and patchy "
    "where the primary PE field is smooth, so they want different scales). On a "
    "partition-only run (-pe_dir2 alone) a pair still means PE1 PE2, and the second "
    "value is the one used."
)


# (dest, flag spelling, cast) for every flag that takes one value per encode axis.
_AXIS_FLAGS = (
    ("levels", "-levels", int),
    ("window", "-window", float),
    ("max_shift", "-max_shift", float),
    ("xcorr_step", "-xcorr_step", float),
    ("reg_sigma", "-reg_sigma", float),
    ("qwarp_minpatch", "-qwarp_minpatch", int),
)


def _axis_txt(vals) -> str:
    """``2`` or ``2/4`` — a per-axis flag value as it should read in a settings line."""
    v = list(vals) if isinstance(vals, (list, tuple)) else [vals]
    out = "/".join(f"{x:g}" for x in v)
    return out


def _resolve_axis_flags(args, n_axes: int, partition_only: bool) -> int | None:
    """Collapse every per-axis flag to exactly one value per encode axis, in place.

    Done once, here, so that no estimator has to re-decide what a bare ``-window 2``
    means: from this point on every one of these flags is a list as long as the run's
    encode-axis list, in the same order.
    """
    for dest, flag, cast in _AXIS_FLAGS:
        raw = getattr(args, dest)
        vals = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if not 1 <= len(vals) <= 2:
            print(f"❌ {flag} takes 1 or 2 values (PE1 [PE2]), got {len(vals)}.", file=sys.stderr)
            return 2
        if len(vals) == 1:
            vals = vals * n_axes
        elif n_axes == 1:
            # A pair always reads PE1 PE2, even when only one of them is being solved:
            # a -pe_dir2-only run IS the partition, so it takes the second value. Said
            # out loud, because silently dropping half of what the user typed is how a
            # run ends up tuned for the axis it isn't correcting.
            keep = 1 if partition_only else 0
            print(
                f"   ⚠ {flag} was given two values but this run solves one encode axis; "
                f"using the {'partition' if partition_only else 'primary PE'} value "
                f"({vals[keep]:g})."
            )
            vals = [vals[keep]]
        setattr(args, dest, [cast(v) for v in vals])
    return None


def create_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_locomoco",
        description="Residual non-linear (PE-axis) motion correction via GPU optical "
        "flow, phase-correlation, or cross-correlation searchlights. See the notes below "
        "for how to read each output, which -flag applies to which -backend, and how to "
        "tune each.",
        formatter_class=FfsHelpFormatter,
        epilog=_EPILOG,
    )
    io = p.add_argument_group("Input/Output")
    io.add_argument(
        "-input",
        "-i",
        nargs="+",
        help="4D motion-corrected NIfTI series [required unless -batch]. Pass ONE for normal single-echo use; pass "
        "SEVERAL (with -me_3depi and -echo_times) for multi-echo 3-D EPI — the echoes are "
        "jointly corrected by one shared partition-direction field scaled per echo.",
    )
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
        help="Output stem [required unless -batch]. A trailing .nii.gz/.nii.zst/.nii is stripped and sets the "
        "output format for the NIfTI outputs (default .nii.gz).",
    )
    io.add_argument(
        "-pe_dir",
        "-pe_dir1",
        "-pe",
        default=None,
        nargs="+",
        metavar="DIR",
        dest="pe_dir",
        help="PRIMARY phase-encode direction: AP/PA/LR/RL/IS/SI or an axis letter x/y/z. "
        "This is the in-plane distortion axis, and it is NOT echo-time dependent (every "
        "echo shifts by the same amount). Giving TWO directions here is shorthand for "
        "'-pe_dir1 A -pe_dir2 B'.",
    )
    io.add_argument(
        "-pe_dir2",
        "-partition_dir",
        "-partition",
        default=None,
        metavar="DIR",
        dest="pe_dir2",
        help="PARTITION (2nd phase-encode) direction, for 3-D EPI. Implies -is_3depi. The "
        "partition wiggle is echo-time DEPENDENT (echo e shifts by TE_e/TE_1 times the "
        "shared field) — that is what distinguishes it from -pe_dir1, and with multiple "
        "echoes it is what lets the two be told apart. Can be given alone (e.g. after "
        "MEDIC has already corrected the primary axis) or together with -pe_dir1 to solve "
        "both at once. One of -pe_dir1 / -pe_dir2 is required.",
    )
    io.add_argument(
        "-slice_axis",
        "-slice",
        default="z",
        type=_axis_from_token,
        help="Through-plane (slice-select) axis to cut the movie along (x/y/z). Must "
        "differ from the PE axis; default z suits axial EPI. Ignored for dual -pe_dir "
        "(the slice axis is fixed to the one axis not phase-encoded), and only a display "
        "hint under -is_3dacq (no slicing is done).",
    )
    io.add_argument(
        "-is_3dacq",
        "-is_3depi",
        "-3d",
        action="store_true",
        dest="is_3dacq",
        help="Data is 3-D-acquired EPI (single-shot / 3-D EPI), not 2-D multi-slice. Then "
        "there are no per-slice fields, so residual distortion is estimated as ONE 3-D PE "
        "field (3-D pooling + through-plane regularisation) instead of slice-by-slice — "
        "strictly better than averaging the two valid perpendicular cuts. Works for the "
        "flow and xcorr backends (phase has no 3-D path yet), plain and rotation-aware.",
    )
    me = p.add_argument_group("Multi-echo 3-D EPI (-me_3depi)")
    me.add_argument(
        "-me_3depi",
        "-me",
        action="store_true",
        help="Multi-echo 3-D EPI: pass one 4D series per echo to -input and the matching TEs "
        "to -echo_times. The partition-direction wiggle is one shared field w(r,t) whose "
        "magnitude scales per echo (echo e is warped by alpha_e·w), so all echoes are jointly "
        "estimated and land back on a common grid ('a voxel is a voxel across echoes'). Writes "
        "a per-echo warp + corrected series, and the learned alpha vs echo-time diagnostic. "
        "Implies a single 3-D solve along -pe_dir; backend flow or xcorr.",
    )
    me.add_argument(
        "-echo_times",
        "-tes",
        nargs="+",
        type=float,
        default=None,
        help="Echo times in ms, one per -input (multi-echo only). Seeds the per-echo scaling "
        "and is the reference for the linear-in-TE diagnostic.",
    )
    me.add_argument(
        "-me_fixed_scaling",
        action="store_true",
        help="Enforce TE-linearity (alpha_e = TE_e/TE_1, not learned) while pooling ALL echoes "
        "into ONE informed search for the shared field — the best-of-both: every echo's SNR, "
        "the hard linear constraint, no cross-echo compromise. flow: image-space pooled LK; "
        "xcorr: a shared-parameter searchlight (every echo trial-shifted by alpha_e·s at once, "
        "SNR-weighted). Use when linearity is already established; omit (default) to instead LEARN "
        "alpha and data-check the scaling.",
    )
    me.add_argument(
        "-me_flat_scaling",
        "-me_flat",
        action="store_true",
        help="Like -me_fixed_scaling but FLAT: every echo shifts the SAME amount "
        "(alpha_e = 1, not TE-scaled) while still pooling all echoes' signal into one "
        "informed search. For acquisitions whose partition wiggle is TE-independent. "
        "Combine with -me_estimate_from to apply one echo's field unchanged to the rest.",
    )
    me.add_argument(
        "-me_interecho",
        "-me_ie",
        action="store_true",
        help="Inter-echo mode: align the echo STACK within each TR instead of aligning each "
        "echo across time. Registers every echo to its lower-TE neighbour (nearest contrast, a "
        "short ΔTE-sized reach — no temporal template), pools all adjacent pairs per TR under the "
        "linear-in-TE scaling, and corrects each echo onto echo 1's frame. Echo 1 (shortest TE) is "
        "the assumed-undistorted anchor. Does NOT remove echo 1's own residual wiggle — follow "
        "with a temporal pass if needed. Ignores the -me_*_scaling / -me_estimate_from flags. "
        "Registers echoes to each other WITHIN a TR, so it needs no temporal template and can run "
        "BEFORE motion correction (unlike the temporal modes, which require moco'd input).",
    )
    me.add_argument(
        "-me_interecho_refine",
        "-me_ie_refine",
        nargs="?",
        type=int,
        const=1,
        default=0,
        metavar="N",
        help="After -me_interecho, run a TEMPORAL joint pass on the inter-echo-corrected "
        "stack (N internal -refine rounds; bare flag = 1). Recovers what the inter-echo pass "
        "cannot see: its per-pair dropout masks are eroded and gated, so the brain RIM — where "
        "the late echo has dropped out but the early ones have not — gets no estimate, and a "
        "dimming rim can even read as shrinkage. The second pass has a temporal template "
        "instead of a cross-echo one, needs no dropout mask, and any leftover rim shift is now "
        "a large error in the EARLY echoes, which is what lets it pull the edges back. It also "
        "removes the leftover TE_1·g that inter-echo leaves on EVERY echo (it only ever "
        "removed the differences between echoes). ONE combined solve over the whole stack — "
        "pooled searchlight / pooled LK — per scaling law; see -me_refine_scaling. Uses "
        "the SAME backend / -window / -max_shift / -xcorr_step as the first pass, plus -ref "
        "(which the plain inter-echo mode ignores). The two fields are composed (not summed) "
        "and the output is resampled once from the raw data.",
    )
    me.add_argument(
        "-me_refine_scaling",
        choices=("affine", "step", "flat", "te", "learn"),
        default="affine",
        help="[-me_interecho_refine] Per-echo scaling model for the refine pass.\n"
        "What inter-echo leaves behind has TWO parts, and they scale with TE differently:\n"
        "  (a) TE_1·g — echo 1's OWN distortion. Inter-echo aligns the echoes to each other,"
        " never to undistorted anatomy, so this survives identically on every echo. FLAT.\n"
        "  (b) (TE_e−TE_1)·delta — the ladder error, e.g. the pass found 1.0 vox where the"
        " truth was 1.5. The shortfall grows with TE and is zero at the anchor. STEP.\n"
        "  affine  run both as two pooled solves, spanning the whole family.\n"
        "  step    the ladder law alone.\n"
        "  flat    echo 1's own distortion alone.\n"
        "  te      plain TE-proportional.\n"
        "  learn   fit alpha from the data. The only mode that is NOT a single pooled solve —"
        " rank-1 factoring needs every echo estimated separately first — and the way to"
        " data-check which law actually dominates.",
    )
    me.add_argument(
        "-me_match",
        choices=("none", "meanstd", "localnorm", "gradmag", "ngf"),
        default="localnorm",
        help="[-me_interecho -backend flow] Intensity matching applied to each ECHO PAIR before"
        " the LK solve. The cross-TE counterpart of -match (which matches frames over TIME, and"
        " is ignored on the multi-echo path).\n"
        "Optical flow assumes brightness constancy, which consecutive echoes violate by"
        " construction — T2* dims the later echo everywhere. Unmatched, that decay step is read"
        " as displacement and the field diverges.\n"
        "  none       the raw residual.\n"
        "  localnorm  local z-score both sides.\n"
        "  ngf        BETA. Signed unit-gradient component along the encode axis,\n"
        "             invariant to a local multiplicative gain (drift 0.09%% under a\n"
        "             1.25x gain, against 3.32%% for localnorm). It did NOT reduce task\n"
        "             contamination on 0.8mm data: gradient orientation is preserved\n"
        "             only where the response varies slowly against the anatomy, and at\n"
        "             submillimetre it varies on the same scale. See -match_eta_q\n"
        "             before concluding. 3-D solve only.\n"
        "  gradmag    locally-normalized gradient magnitude — edges only, the most"
        " contrast-agnostic.\n"
        "  meanstd    one global rescale.\n"
        "Neighbourhood size is -me_match_sigma. Inert for -backend xcorr (correlation is"
        " already scale-invariant).",
    )
    me.add_argument(
        "-me_match_sigma",
        type=float,
        default=2.0,
        metavar="VOX",
        help="[-me_match localnorm/gradmag] Gaussian sigma, in VOXELS, of the neighbourhood the"
        " local mean and scale are measured over.\n"
        "A CONTRAST scale, independent of the motion scale -window: matching finishes on the"
        " echo pair before the LK solve starts, which then pools gradients over -window on the"
        " matched data.\n"
        "Much tighter than -match_sigma (6) on purpose. -match cancels a smooth"
        " illumination-like drift, so it wants a wide mean; the T2* difference between two"
        " echoes tracks tissue and follows anatomy closely, so the mean has to be local enough"
        " to follow it. Widen it if the late echo has dropped out over a region larger than the"
        " window.",
    )
    me.add_argument(
        "-me_estimate_from",
        "-me_from",
        default=None,
        help="RECOMMENDED once TE-linearity is established. Estimate the shared field on ONE"
        " echo and scale to the rest by the TE ratio — no joint solve, no per-echo passes.\n"
        "  last   the largest TE, whose shifts are the easiest to detect.\n"
        "  mid    the middle echo.\n"
        "  first  the shortest TE.\n"
        "  <N>    a 1-based echo index.\n"
        "Runs the full single-echo -is_3dacq estimator on that echo (-refine / -superhard and"
        " the rest all apply); every other echo's warp is (TE_e/TE_k)·w. Much faster and often"
        " steadier than the joint solve. Omit it and use the joint path only when you also want"
        " to DATA-CHECK the scaling.",
    )
    est = p.add_argument_group("Estimation — all backends")
    est.add_argument(
        "-backend",
        default="flow",
        choices=("flow", "phase", "xcorr", "qwarp"),
        help="Displacement estimator. All four measure the same PE shift.\n"
        "  flow   pyramidal Lucas-Kanade optical flow. Most precise, slowest."
        " Tuned by -levels / -iters / -window.\n"
        "  phase  phase-correlation searchlight (FFT phase ramp along PE). Fastest, and close"
        " to flow in accuracy. Its window is -patch, not -window.\n"
        "  xcorr  magnitude cross-correlation searchlight: slide along PE, take the peak local"
        " correlation. Robust and single-shot; already scale-invariant, so the"
        " intensity-matching flags are inert for it.\n"
        "  qwarp  no flow estimate at all — the joint TE-scaled qwarp owns the whole field."
        " Its per-patch ncc is invariant to an affine intensity change inside the patch, so"
        " it needs none of the -match machinery;"
        " registering the raw frames to a temporal reduction of themselves (-ref picks the"
        " reduction, -refine rebuilds it from the corrected series and re-solves). For input"
        " that is ALREADY motion corrected; for residual motion use -backend flow"
        " -final_qwarp.\n"
        "See the epilog for tuning.",
    )
    est.add_argument(
        "-ref",
        default=None,
        help="[all] What every frame is registered TO. Three families:\n"
        "  STATIC  mean | median | max | first | <frame index> — one template for the whole"
        " run. 'max' takes the temporal maximum, which fills slices that later frames rotate"
        " out of the FoV and is a high-signal target.\n"
        "  PROGRESSIVE  first_mean | first_median — frame t registers to the running"
        " mean/median of the already-corrected EARLIER frames (a bootstrapped template, frame"
        " 0 the seed). Sequential, so slower.\n"
        "  CONDITION-PAIRED  paired | paired_mean | paired_median — needs -events and a TR."
        " Frames are binned by predicted BOLD state and each registers to the template of its"
        " OWN bin, so the task response is common-mode within the pair and cancels. This is"
        " PREVENTION for the contamination -detask cures, with no HRF fit and nothing assumed"
        " about how the response enters; see -task_bin_width. 3-D solve only, and not with"
        " -backend qwarp. 'max' has no paired form — a max over one bin's frames is a noise"
        " envelope, not a template.\n"
        "Also selects the aggregation for -refine, and (when given explicitly) the qwarp"
        " template. Default: max for rotation-aware mode, mean otherwise.",
    )
    est.add_argument(
        "-first_n",
        "-first-n",
        type=int,
        default=None,
        metavar="N",
        help="[all] Build the -ref aggregate (mean/median/max) from only the FIRST N "
        "frames, not the whole run. The best-of-both for a run whose later frames drift "
        "(e.g. a slow time-stretch): the SNR/FoV-fill of an aggregate without the bad late "
        "frames polluting the template. Honoured for the initial AND every refine "
        "reference. Default: all frames.",
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
    est.add_argument(
        "-hpf_spatial",
        "-hpf-spatial",
        type=float,
        default=0.0,
        metavar="MM",
        help="[all, EXPERIMENTAL] Spatial high-pass (Gaussian sigma, mm) applied to the "
        "ESTIMATION frames only: subtracts a blurred copy so smooth non-motion "
        "intensity changes (drift, the respiration B0 modulation riding along with a "
        "sub-voxel PE shift) stay out of the flow, while the edges that encode the "
        "shift survive — a more purely geometric target. The correction still "
        "resamples the RAW series (true intensities preserved). Works for slicewise, "
        "3-D-acq, and -me_3depi paths. 0 = off. Not supported with rotation-aware mode. "
        "No clear advantage over the plain path observed yet — a knob to experiment with.",
    )
    est.add_argument(
        "-match",
        "-tmatch",
        choices=("none", "meanstd", "localnorm", "gradmag", "ngf"),
        default="none",
        help="[flow, phase] Intensity matching applied to the ESTIMATION frames only, before"
        " -hpf_spatial and -do_blur. The corrected output still resamples the RAW series"
        " either way.\n"
        "Reach for this when frame intensity varies OVER THE RUN: the pre-steady-state ramp"
        " (first frames brighter until T1 saturation settles), or any other non-motion"
        " fluctuation. Optical flow assumes a voxel keeps its intensity as it moves, so it"
        " reads a brightness change as displacement — on one 1.2 mm run the LK field for"
        " frame 0 came out 4.5x the steady-state median and spatially incoherent.\n"
        "  none       the raw residual.\n"
        "  localnorm  local z-score both sides. Cancels a MULTIPLICATIVE gain, which"
        " -hpf_spatial cannot (it only subtracts). The usual choice.\n"
        "  ngf        signed unit-gradient component along the encode axis. Invariant\n"
        "             to a local multiplicative gain, so a BOLD response cannot enter\n"
        "             the solve as brightness through the INTERIOR of a responding\n"
        "             region; its boundary is still an edge no intensity transform\n"
        "             removes. 3-D solve only.\n"
        "  gradmag    locally-normalized gradient magnitude — edges only, the most"
        " contrast-agnostic.\n"
        "  meanstd    one global rescale per frame. Near-useless for the ramp, whose gain"
        " varies 0.94-1.39 across tissue.\n"
        "Neighbourhood size is -match_sigma; it does not interact with -window.\n"
        "This is really a FLOW fix. LK works from the raw residual moving−fixed, so a gain"
        " enters the solve as if it were displacement and there is nothing in the cost to"
        " remove it. The correlation-based estimators are already immune, because a Pearson"
        " r is invariant to any affine intensity change within its window: -backend xcorr"
        " normalizes inside its own searchlight, and the qwarp paths (-final_qwarp,"
        " -backend qwarp) score ncc per patch on the RAW data and never see this flag."
        " -backend phase is in between — the phase-only normalization handles a global"
        " gain, but not one that varies across the volume.\n"
        "Applies to single-echo AND multi-echo runs (each echo is matched against itself,"
        " over TIME). The cross-TE counterpart is -me_match. Not supported with"
        " rotation-aware mode.",
    )
    est.add_argument(
        "-pe_null",
        "-pe-null",
        default=None,
        metavar="DIR",
        help="BETA — read it, do not trust it to fix anything yet.\n"
        "Solves the UN-ENCODED axis alongside the encode axes as a null channel: an EPI\n"
        "residual cannot be displaced along the readout direction, so whatever is\n"
        "estimated there is spurious, measured on the same voxels and the same\n"
        "estimator as the encode axes. The part of each encode axis that the null\n"
        "predicts over TIME is then regressed out per voxel. The null field is never\n"
        "applied to the data; it is written as {prefix}_flow_null.\n"
        "WHAT ACTUALLY HAPPENS on real data: the 3-axis solve is markedly less\n"
        "determined than the 2-axis one (separability 0.978 -> 0.880 on a 0.8mm run),\n"
        "so real encode-plane motion leaks into the null through the aperture — it came\n"
        "out 5x LARGER than the encode axes, the refine loop diverged a pass earlier,\n"
        "and the correction moved task coupling only 8.9x -> 7.8x. Treat the written\n"
        "field as a diagnostic of what the estimator invents; the regression is not\n"
        "yet a reliable fix.\n"
        "The direction cannot be assumed (a partition-only run leaves two candidates),\n"
        "so name it: -pe_null RL for a readout along x. 3-D single-echo path only.",
    )
    est.add_argument(
        "-pe_null_skip",
        "-pe-null-skip",
        type=int,
        default=0,
        metavar="N",
        help="[-pe_null] Drop the first N frames from the SLOPE FIT (they are still\n"
        "corrected). Pre-steady-state frames are outliers in the null and the encode\n"
        "axes at once, and a least-squares slope is dominated by outliers, so a few\n"
        "such frames set the coefficient for the whole run. Measured on a 0.8mm run:\n"
        "frame 0 sat 11%% above the run mean with a per-voxel gain against steady state\n"
        "of 0.856-1.454, and the flow rms there was 1.86x the steady-state level,\n"
        "decaying over ~6 frames. -match localnorm reduces this and cannot remove it —\n"
        "the gain is tissue-dependent, so it varies at tissue boundaries, which is\n"
        "sub-window structure a local z-score leaves behind. Try 5-10.",
    )
    est.add_argument(
        "-pe_null_min_r2",
        "-pe-null-min-r2",
        type=float,
        default=0.0,
        metavar="R2",
        help="[-pe_null] Leave a voxel alone unless the null channel explains this\n"
        "fraction of the encode axis's variance over time. 0 corrects everywhere.\n"
        "Raise it when the null channel is noisy: a slope fitted to noise injects the\n"
        "displacement the flag exists to remove.",
    )
    est.add_argument(
        "-match_eta_q",
        "-match-eta-q",
        type=float,
        default=0.5,
        metavar="Q",
        help="[-match ngf] Quantile of the volume's own squared gradient magnitude that\n"
        "sets the eta floor. The knob that decides what counts as an edge: normalising\n"
        "promotes EVERY gradient to unit length, so a floor taken from the median of a\n"
        "mostly-flat volume hands noise-level structure the same weight LK gives a real\n"
        "boundary. Raise it toward 0.9 when ngf comes out noisier than localnorm.",
    )
    est.add_argument(
        "-match_sigma",
        "-match-sigma",
        type=float,
        default=6.0,
        metavar="VOX",
        help="[-match localnorm/gradmag] Gaussian sigma, in VOXELS, of the neighbourhood the"
        " local mean and scale are measured over.\n"
        "This is a CONTRAST scale, not a motion scale. It wants to be wide enough that the"
        " local mean tracks the illumination-like drift you are cancelling rather than the"
        " anatomy you are trying to match: too small and the z-score flattens the very edges"
        " the estimator tracks, too large and it stops following a spatially varying gain."
        " The default is deliberately wider than the flow -window.\n"
        "It does NOT interact with -window. Matching runs to completion on the frames first;"
        " the estimator then pools gradients over -window on the already-matched data. Two"
        " independent knobs on two different stages.",
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
        nargs="+",
        default=[3],
        help="[flow only] Coarse-to-fine pyramid levels. The flow is solved on a stack "
        "of images each halved in size; the coarsest catches large displacements "
        "(1 px there = 2^(levels-1) px full-res), finer levels refine. More = handles "
        "bigger motion, but risks aliasing on thin slices.\n"
        "Per axis this is the number of levels, COUNTING FROM THE COARSEST, that the axis "
        "stays in the joint solve: the pyramid is as deep as the larger value, and the "
        "axis with the smaller one drops out of the 2x2 normal equations below it and "
        "keeps the field it had. `-levels 2 4` refines both down two levels, then "
        "refines the partition alone on the two finest — which is what you want when the "
        "primary PE field has no structure left at that scale and fitting it there just "
        "leaks noise into the axis that does." + PERAXIS,
    )

    tune = p.add_argument_group("Shared tuning (applies to the backends tagged below)")
    tune.add_argument(
        "-iters",
        type=int,
        default=4,
        help="[flow, phase] Refinement iterations INSIDE one estimate (not the outer"
        " -refine reference loop).\n"
        "  flow   LK warp-and-update passes per pyramid level.\n"
        "  phase  whole-field warp-and-re-read passes, which cancel the single-patch"
        " leakage bias.\n"
        "  xcorr  ignores it; it is single-shot.\n"
        "More = better convergence for larger motion, at linear cost.",
    )
    tune.add_argument(
        "-window",
        type=float,
        nargs="+",
        default=[2.0],
        metavar="SIGMA",
        help="[flow, xcorr] Neighbourhood Gaussian sigma, in VOXELS, over which the"
        " DISPLACEMENT is pooled.\n"
        "  flow   the LK gradient-pooling window — the scale over which flow is assumed"
        " locally constant.\n"
        "  xcorr  the searchlight radius the local correlation is measured over.\n"
        "  phase  ignores it; its window is -patch.\n"
        "Larger = smoother and more robust, but blurs fine local shifts; smaller = sharper"
        " and noisier. This is a MOTION scale, and is independent of the contrast scales"
        " -match_sigma / -me_match_sigma: intensity matching finishes before the estimator"
        " runs, and the estimator then pools over -window on the matched data." + PERAXIS,
    )

    search = p.add_argument_group("Searchlight backends (-backend phase / xcorr)")
    search.add_argument(
        "-max_shift",
        type=float,
        nargs="+",
        default=[3.0],
        metavar="VOX",
        help="Largest PE shift to allow (voxels). Set just above the biggest residual "
        "shift you expect (sub- to a few voxels here). Smaller = faster xcorr (fewer "
        "trial offsets) and a tighter phase no-wrap band; too small clips real motion. "
        "phase and xcorr search within it; flow clamps its accumulated field to it, "
        "which is what stops a textureless slab-end slice random-walking to hundreds "
        "of voxels. All three also use it as the refine divergence threshold." + PERAXIS,
    )
    search.add_argument(
        "-search_min_steps",
        "-search-min-steps",
        type=int,
        default=5,
        metavar="N",
        help="[xcorr 3-D first-peak] Minimum trial-offset samples swept per side before the "
        "adaptive early-stop can fire — enough points to know a voxel is really rising (one "
        "point at ±0.5 vox isn't). Clamped to the search range, so a tiny -max_shift / coarse "
        "-xcorr_step just searches what's there. Default 5.",
    )
    search.add_argument(
        "-argmax",
        action="store_true",
        help="[xcorr 3-D] Use the classic global-argmax peak over the full ±max_shift grid "
        "instead of the default first-peak finder. The first-peak finder sweeps outward "
        "from zero, takes the first real peak nearest zero (no-shift-biased, ignores "
        "later oscillation humps, never rails), and stops once no voxel is still rising — "
        "usually faster and cleaner. Flip to -argmax to compare or if a peak is genuinely "
        "far from zero and multi-modal.",
    )
    search.add_argument(
        "-xcorr_step",
        type=float,
        nargs="+",
        default=[0.5],
        metavar="VOX",
        help="[xcorr only] Trial-offset spacing (voxels) of the correlation search — "
        "xcorr's sub-voxel knob, like -iters is for the others. The peak is fit by a "
        "5-point parabola either way; a finer step (e.g. 0.25) samples the curve more "
        "densely for a touch more accuracy at ~2× the trials. 0.5 is the sweet spot." + PERAXIS,
    )
    search.add_argument(
        "-reg_sigma",
        "-reg-sigma",
        type=float,
        nargs="+",
        default=[1.5],
        metavar="VOX",
        help="[xcorr] Confidence-weighted spatial smoothing of the searchlight field "
        "(Gaussian sigma, voxels). The displacement field is physically smooth, so each "
        "voxel borrows from its high-confidence neighbours (peak quality × prominence "
        "over no-shift); high-confidence voxels keep their own estimate. Fixes lone "
        "railed/spurious peaks without blurring real structure. 0 = off. Default 1.5." + PERAXIS,
    )
    search.add_argument(
        "-noshift_margin",
        "-noshift-margin",
        type=float,
        default=0.0,
        metavar="CORR",
        help="[xcorr] Optional HARD no-shift guard: a peak beating the zero-shift "
        "correlation by less than this (normalised-corr units) is zeroed outright and "
        "filled from neighbours via -reg_sigma. OFF by default (0), because residual "
        "motion is itself small — its prominence over no-shift is small, so a hard guard "
        "risks zeroing real sub-voxel shifts. The no-shift prior is already applied SOFTLY "
        "(confidence ∝ prominence weights the -reg_sigma smoother). Enable (e.g. 0.03) "
        "only on very noisy data where you would rather drop uncertain voxels.",
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

    qw = p.add_argument_group(
        "Nonlinear qwarp (-final_qwarp polish / -backend qwarp)",
        "A PE-only nonlinear warp under the joint TE-scaled objective. It either POLISHES\n"
        "the residual a flow/phase/xcorr estimate could not resolve (-final_qwarp; the\n"
        "total field is w+r), or OWNS the whole field on its own (-backend qwarp, no flow\n"
        "pass at all). Every -qwarp_* knob below applies to both.",
    )
    qw.add_argument(
        "-final_qwarp",
        action="store_true",
        help="After a flow/xcorr/phase estimate, POLISH the residual with a few fine PE-only "
        "nonlinear qwarp levels under the joint TE-scaled objective (one shared residual field "
        "r, every echo scored at alpha_e·r). Removes the sub-voxel wiggle the search can't "
        "resolve; total field is w+r. Configured by the -qwarp_* flags below. Needs the "
        "corrected series (not compatible with -no_corrected). Use -backend qwarp instead to "
        "let qwarp own the whole field. Works single-echo too; for 2-D multi-slice data the "
        "patches are 2-D slicewise (see -qwarp_3d).",
    )
    qw.add_argument(
        "-qwarp_3d",
        "-qwarp-3d",
        action="store_true",
        dest="qwarp_3d",
        help="Use isotropic 3-D qwarp patches on 2-D multi-slice data. By default (2-D "
        "acquisition, no -is_3dacq) the patches are 2-D and one slice thick, matching the "
        "slicewise estimator: each slice is acquired at its own instant, so a 3-D patch would "
        "smooth the residual field across acquisition times. Set this only if you know the "
        "residual field really is smooth through plane. Implied by -is_3dacq / -me_3depi.",
    )
    qw.add_argument(
        "-qwarp_minpatch",
        "-final_qwarp_minpatch",
        type=int,
        nargs="+",
        default=[7],
        dest="qwarp_minpatch",
        help="Finest qwarp patch size (voxels), in-plane only when the patches are 2-D "
        "slicewise.\n"
        "The default suits the -final_qwarp polish; try 9 for -backend qwarp, which has the"
        " whole field to find rather than a residual.\n"
        "Per axis this is the FINEST patch that axis is still solved at: it drops out of "
        "the coupled per-patch Gauss-Newton system below its own value while the other "
        "keeps refining. `-qwarp_minpatch 13 7` stops the smooth primary-PE field at 13 "
        "and takes the partition down to 7." + PERAXIS,
    )
    qw.add_argument(
        "-qwarp_levels",
        "-final_qwarp_levels",
        type=int,
        default=2,
        dest="qwarp_levels",
        help="Number of qwarp levels; the coarsest is ~minpatch/0.75^(n-1).\n"
        "The default suits the -final_qwarp polish; try 4 for -backend qwarp, which needs more"
        " reach starting from scratch.",
    )
    qw.add_argument(
        "-qwarp_iters",
        "-final_qwarp_iters",
        type=int,
        default=10,
        dest="qwarp_iters",
        help="Per-patch optimizer iterations. The GN solver converges in a few steps, so this"
        " is a generous cap, not a dial.",
    )
    qw.add_argument(
        "-qwarp_cost",
        default="ncc",
        choices=("ncc", "lpa", "lpc"),
        help="qwarp patch cost.\n"
        "  ncc  weighted Pearson. Right for the same-contrast job this is, and the only cost"
        " the Gauss-Newton optimizer supports.\n"
        "  lpa  AFNI-faithful blok-local Pearson, absolute value. Autodiff Adam only.\n"
        "  lpc  AFNI-faithful blok-local Pearson, signed. Autodiff Adam only.",
    )
    qw.add_argument(
        "-qwarp_optimizer",
        default="gn",
        choices=("gn", "adam"),
        help="qwarp per-patch optimizer.\n"
        "  gn    Gauss-Newton with an analytic image-gradient Jacobian. No autograd, and it"
        " converges in a few steps (~5x faster). Needs -qwarp_cost ncc, and falls back to adam"
        " otherwise.\n"
        "  adam  the autodiff optimizer. Required for the lpa/lpc costs.",
    )
    qw.add_argument(
        "-qwarp_compile",
        "-qwarp-compile",
        action="store_true",
        dest="qwarp_compile",
        help="Experimental (CUDA only, off by default): torch.compile the per-frame qwarp "
        "building blocks. The plan geometry is identical across frames, so the compile "
        "warmup is paid once and amortized over the whole series. Applies to both the "
        "-final_qwarp polish and -backend qwarp. Benchmark before trusting on a new box; "
        "falls back to eager if inductor is unavailable.",
    )
    qw.add_argument(
        "-qwarp_refine",
        "-qwarp-refine",
        type=int,
        default=None,
        dest="qwarp_refine",
        help="Re-run the qwarp registration N extra times, each against a template rebuilt "
        "from the previous pass's corrected series (re-solved from seed 0, never seeded "
        "with the previous field). The first template is built from data that still "
        "carries the distortion, so it is blurred by it and biases the field LOW — the "
        "same reason the estimator has -refine. Defaults to -refine under -backend qwarp "
        "(where no flow pass runs, so -refine has nothing else to do) and to 0 for the "
        "-final_qwarp polish (whose template the flow refine already sharpened). Each "
        "pass costs a full qwarp sweep.",
    )
    acc = p.add_argument_group("Accuracy (trade time for exactness)")
    acc.add_argument(
        "-warp_interp",
        default="auto",
        choices=("auto", "bilinear", "bicubic", "lanczos"),
        help="[all] Resampler for the estimation iterations and the correction.\n"
        "  auto      Lanczos for 1-D/2-D PE warps, bicubic for rotation-aware 3-D.\n"
        "  bilinear  cheapest, and damps the signal it moves.\n"
        "  bicubic   removes the bilinear damping bias, so the iterations converge to the true"
        " shift. Biggest gain on smooth data; costs a little more per warp.\n"
        "  lanczos   separable windowed sinc over the active PE axis/axes (half-width"
        " -warp_radius): 1-D for single PE, 2-D for dual PE on CUDA. Preserves sub-voxel signal"
        " that trilinear blurs out of the CORRECTED output and the refine template. ~2x per"
        " warp, and it passes more thermal noise — verify on your data.\n"
        "This is about OUTPUT fidelity, not shift accuracy: the shift estimate itself is set by"
        " the pooling window (-window), not by the resampler.",
    )
    acc.add_argument(
        "-warp_radius",
        type=int,
        default=3,
        metavar="A",
        help="[all] Lanczos half-width for -warp_interp lanczos (taps = 2·A): 3 = "
        "Lanczos-3, 5 ≈ wsinc5, 7 ≈ heptic. Larger = sharper but more noise/ringing on "
        "thermal-noise data. Ignored unless -warp_interp lanczos.",
    )
    acc.add_argument(
        "-refine",
        type=int,
        default=0,
        metavar="ROUNDS",
        help="[plain path + -backend qwarp] Outer reference-refinement rounds (the max cap when "
        "-converge is set). After the first estimate, rebuild the reference from the "
        "corrected series (motion removed → sharp, and aggregated per -ref: a max stays "
        "FoV-filled) and re-register the original frames against it. Each pass prints its "
        "step size (RMS displacement change, in-brain). 1–3 tightens the values; 0 = off. "
        "Under -backend qwarp there is no flow pass, so this drives the QWARP passes "
        "instead (override with -qwarp_refine).",
    )
    acc.add_argument(
        "-converge",
        type=float,
        default=0.0,
        metavar="VOX",
        help="[plain path only] Stop the -refine loop early once a pass's step size (RMS "
        "displacement change vs the previous pass, in-brain voxels) falls below this — so "
        "you spend passes only while they still move the estimate. 0 = off (run all "
        "-refine rounds). Try ~0.02; the printed per-pass Δ tells you where it plateaus.",
    )
    acc.add_argument(
        "-converge_rel",
        "-converge-rel",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="[plain path only] RELATIVE convergence: stop the -refine loop once a pass "
        "shrinks the step by less than this fraction of the previous pass (the improvement "
        "itself has plateaued — 'changing by about the same amount each time'), even if the "
        "absolute step is still non-trivial. e.g. 0.05 = stop when a pass gains <5%% over "
        "the last. Either -converge or -converge_rel can fire. 0 = off.",
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
    mask.add_argument(
        "-nocoverage",
        "-no_coverage",
        "-no-coverage",
        dest="nocoverage",
        action="store_true",
        help="Disable the temporal data-coverage restriction. By default the gate also "
        "excludes voxels that are not acquired in EVERY frame — the zero wedge that "
        "rigid motion correction leaves at the FoV edge sits in a different place each "
        "frame, and a voxel that is bright in one frame and exactly zero in the next "
        "makes a shift estimator invent an enormous displacement to explain it.",
    )
    mask.add_argument(
        "-coverage_erode",
        "-coverage-erode",
        type=int,
        default=1,
        metavar="VOX",
        help="Voxels to peel off the data-coverage boundary (default 1), dropping the "
        "ramp of partial-value voxels that interpolation leaves just inside the empty "
        "wedge. The feather width is peeled on top of this automatically.",
    )

    task = p.add_argument_group("Task-coupling diagnostic (-events)")
    task.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv for THIS run — switches on the task-coupling "
        "diagnostic. Every backend here closes a brightness-constancy data term, so a "
        "strong block design gives it an intensity change it cannot distinguish from a "
        "sub-voxel shift. This measures how much of the estimated field the task "
        "explains (over a circular-shift null), and whether that lands on top of the "
        "BOLD response — which is what separates 'BOLD leaked into the estimate' from "
        "'the subject moved with the task'.\n"
        "Writes, per encode axis, {prefix}_taskr_pe1/pe2 — the field's SIGNED partial "
        "correlation, labelled and tagged fico so AFNI thresholds it — and _taskrms_*, "
        "its task-explained part in voxels.\n"
        "For the DATA it writes {prefix}_taskfit_data and, on the corrected series, "
        "{prefix}_taskfit_data_after: one bucket each, sub-bricks alternating "
        "{cond}_Coef (percent signal change) and {cond}_Correl, so you view the "
        "amplitude and threshold on the sub-brick after it. Both are written whenever "
        "-events is given, fix or no fix. Inspect the after fit to judge the impact of "
        "contamination — changes in the amplitude of the response, or blurring.\n"
        "The fico p is NOMINAL — no autocorrelation correction — so it is there to "
        "threshold and colour the maps, not to be reported.\n"
        "Diagnostic only: nothing about the correction changes.",
    )
    task.add_argument(
        "-event_ignore",
        "-event-ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude from the task design (e.g. fixation).",
    )
    task.add_argument(
        "-event_cols",
        "-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom -events column names. Default: onset duration trial_type.",
    )
    task.add_argument(
        "-tr",
        type=float,
        default=None,
        metavar="SEC",
        help="Repetition time for the task design. Default: the NIfTI header pixdim[4].",
    )
    task.add_argument(
        "-task_polort",
        "-task-polort",
        type=int,
        default=None,
        metavar="DEG",
        help="Legendre drift degree removed from the field AND the design before the "
        "fit, so the R2 is partial and a slow drift in the field cannot be charged to a "
        "long block. Default: AFNI's 1 + floor(run_seconds/150).",
    )
    task.add_argument(
        "-task_bin_width",
        "-task-bin-width",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="[-ref paired] Bin size as a fraction of each condition's peak-to-peak "
        "swing. 0.2 (default) gives five levels: baseline, three slope steps, peak. "
        "Narrower bins match the state more tightly but thin the average that suppresses "
        "the residual field — that average is what makes the template cleaner than the "
        "frames it came from, so this is the real trade.",
    )
    task.add_argument(
        "-task_bins",
        "-task-bins",
        type=int,
        default=None,
        metavar="N",
        help="[-ref paired] Explicit level count per condition; overrides -task_bin_width.",
    )
    task.add_argument(
        "-task_min_frames",
        "-task-min-frames",
        type=int,
        default=4,
        metavar="N",
        help="[-ref paired] Bins with fewer frames than this are merged into the nearest "
        "bin in state space. A template averaged over two frames suppresses the residual "
        "field by only 1/sqrt(2) and hands its own noise to every frame registered "
        "against it. Default 4.",
    )
    task.add_argument(
        "-write_pc_maps",
        "-write-pc-maps",
        nargs="?",
        type=int,
        const=-1,
        default=None,
        metavar="N",
        help="Write the warp PCs' SPATIAL loadings for eyeballing: one 4-D file per\n"
        "encode axis ({prefix}_pcmap_pe1 / _pcmap_pe2), component on the 4th axis.\n"
        "Bare = every component; N = the top N by variance.\n"
        "One file PER AXIS because the temporal basis is shared but each component has\n"
        "its own loading on each axis — sub-brick k of the two files is the same\n"
        "component seen two ways. Loadings are raw and signed; the brick label carries\n"
        "the variance share, and the task enrichment too when -events is given.\n"
        "Works without -events (unscored). The temporal side is -want_pcs, which writes\n"
        "the matching time courses as .1D.",
    )
    task.add_argument(
        "-warp_recon",
        "-warp-recon",
        default=None,
        metavar="SPEC",
        help="BETA. REBUILD the field from its principal components before any output is\n"
        "derived. SPEC: 'pcs' (all components — lossless, identity on its own),\n"
        "'pcs:N' (top N by variance), 'pcs:0.F' (components reaching that FRACTION of\n"
        "the variance). Independent of -want_pcs, which only chooses how many PC time\n"
        "courses are WRITTEN.\n"
        "This is DENOISING by truncation. To REJECT task-loaded components, use\n"
        "-detask ica: it projects them out of the full-rank field, which is a\n"
        "de-tasking operation and lives with the other -detask modes.\n"
        "PCA cannot find contamination on its own, and this is a property of the\n"
        "method rather than a tuning problem: it orders by VARIANCE, and on a real\n"
        "contaminated run the task part was 0.7%% of the field's — no principal\n"
        "component exceeded 1.25x enrichment while the field itself was 8.8x at the\n"
        "tail. Multi-echo rebuilds echo 1 only (truncation has no shared time courses\n"
        "to reuse across echoes) and says so.",
    )
    task.add_argument(
        "-reject",
        nargs="?",
        type=float,
        const=2.0,
        default=None,
        metavar="E",
        help="[-warp_recon] Also drop any component whose SPATIAL weights are more than\n"
        "E-fold concentrated on activated tissue (default %(default)s when given bare,\n"
        "1.0 = spread like the brain). Needs -events.\n"
        "Scored on WHERE the weights live, not on correlation with the design: a block\n"
        "design is ~2 degrees of freedom, so a temporal correlation cannot be\n"
        "thresholded, while an energy share against the active mask has a no-relation\n"
        "value of 1.0 by construction. Per COMPONENT PER AXIS — the temporal basis is\n"
        "shared but each component loads separately on each encode axis.",
    )
    task.add_argument(
        "-reject_surrogates",
        "-reject-surrogates",
        type=int,
        default=2000,
        metavar="N",
        help="[-detask ica] Phase-randomised surrogates for the TEMPORAL criterion's\n"
        "null. More is a tighter familywise threshold and a better effective-DoF\n"
        "estimate; the cost is negligible next to the decomposition. [default: %(default)s]",
    )
    task.add_argument(
        "-detask",
        nargs="?",
        const="field",
        default=None,
        metavar="MODE",
        help="MODE is 'field' (bare -detask), 'filter[:N]' or 'fit[:D]'.\n\n"
        "  filter[:N]  NOTCH the design's frequency band out of the images the\n"
        "              ESTIMATOR sees, before it runs, then resample the raw input with\n"
        "              the resulting field — so the field is never contaminated and\n"
        "              nothing bleeds through the solve into other frames. The band is\n"
        "              every line carrying at least 1%% of the strongest; N widens it by\n"
        "              N bins either side (for amplitude drift across blocks, NOT for\n"
        "              HRF width — a wider HRF narrows the design's band).\n"
        "              THE WORKING FIX, and only for a near-periodic BLOCK design.\n"
        "              Onset jitter past ~0.5s of a 20s period costs bins fast (1 bin\n"
        "              at 0.25s, 7 at 1s, 14 at 2s), the run needs a whole number of\n"
        "              cycles, over 15%% of the spectrum warns and over 50%% is refused.\n"
        "              A jittered event-related design is broadband and cannot be\n"
        "              notched — that is a limit of the METHOD, not evidence such a\n"
        "              design is free of the contamination. Nothing here establishes\n"
        "              that; use -events without -detask to measure it first.\n"
        "              Several conditions are fine: the cost scales with the number of\n"
        "              distinct PERIODS, not conditions. Multi-echo is supported: one\n"
        "              basis notches every echo, since the band is a property of the\n"
        "              DESIGN and the echoes pool into one shared field.\n\n"
        "  fit[:D]     PROJECT THE DESIGN out of the images the ESTIMATOR sees, then\n"
        "              resample the raw input with the resulting field — same place in\n"
        "              the pipeline as 'filter', but the cut is made in DESIGN space\n"
        "              instead of frequency space. Costs ONE degree of freedom per\n"
        "              regressor rather than two per notched line: on a real 5-condition\n"
        "              18s-block run the notch took 43 bins = 86 DoF where the design\n"
        "              spans 5. No periodicity requirement, so this is the only\n"
        "              estimator-side option for a JITTERED or broadband design.\n"
        "              D adds that many time derivatives of every regressor (1 = the\n"
        "              temporal derivative, absorbing a few hundred ms of latency\n"
        "              mismatch; 2 adds curvature) at one more DoF per regressor each.\n"
        "              WHAT IT DOES NOT DO: only the part of the response the canonical\n"
        "              shape explains is removed, so latency/width mismatch leaves task\n"
        "              variance behind — and a contamination bad enough to matter also\n"
        "              distorts the very fit being projected out, leaving a share of the\n"
        "              artifact proportional to how bad the problem was. A mitigation,\n"
        "              not a proof: read the enrichment diagnostic afterwards.\n\n"
        "  ica[:N]     DECOMPOSE the field into independent components and PROJECT the\n"
        "              task-loaded ones out of the FULL-RANK field. Rejection is\n"
        "              implied — that is the whole operation. Bare 'ica' SWEEPS for the\n"
        "              rank (it has an interior optimum: 20 components missed the task\n"
        "              source, 60 found it, the full 119 over-split it again); ica:N\n"
        "              or ica:0.F fixes it.\n"
        "              TWO criteria, and a component failing EITHER is dropped:\n"
        "                SPATIAL   — its weights are >E-fold concentrated on activated\n"
        "                            tissue (-reject E, default 2.0). No null needed:\n"
        "                            1.0 is 'spread like the brain' by construction.\n"
        "                            Blind to a task-locked component driving FEW\n"
        "                            voxels, whose energy lives elsewhere.\n"
        "                TEMPORAL  — the design explains its TIME COURSE better than\n"
        "                            its own phase-randomised surrogates do (omnibus\n"
        "                            R² on the whole design, familywise-corrected\n"
        "                            across components; -reject_surrogates sets N).\n"
        "                            This is what catches the sparse case.\n"
        "              The temporal test MEASURES its own effective DoF and DECLINES\n"
        "              when it has no power, saying so — a component whose own spectrum\n"
        "              coincides with the design's can fit it by accident. Measured:\n"
        "              broadband components vs a periodic block design give ~195 DoF\n"
        "              (usable), narrowband ones at that frequency give ~5 (declines).\n"
        "              Multi-echo: ONE decomposition on the shared field, projected per\n"
        "              echo — exact, since the projection is linear in the field.\n\n"
        "  field       REMOVE the task-locked part of the field, keeping drift and everything "
        "else, and derive EVERY output from the cleaned field — warp, corrected series, "
        "flow maps, PCs, movie. The diagnostic above still measures the ORIGINAL field "
        "(measuring the fix's own output would always say 'no task'), and the part that "
        "was removed is written as {prefix}_flow_*_taskpart so nothing is hidden. "
        "Polynomials go into the fit but NOT the subtraction — a slowly drifting "
        "displacement is real residual motion, and the warp PCs routinely carry a "
        "poly-like component that IS that motion. The corrected series is re-resampled "
        "from the RAW input, never from the already-corrected one. Not wired for the "
        "multi-echo path (one shared field scaled per echo).",
    )
    task.add_argument(
        "-task_thresh",
        "-task-thresh",
        type=float,
        default=None,
        metavar="R",
        help="Absolute |r| cut on the DATA's task map that defines 'where the task is'. "
        "The headline number is then how much of the FIELD's task-locked displacement "
        "falls inside that mask, against the share of voxels it occupies — 1.0x means "
        "the field's task coupling is spread like the brain and has nothing to do with "
        "where the task is. Default: use -task_top_frac instead.",
    )
    task.add_argument(
        "-task_top_frac",
        "-task-top-frac",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of voxels, ranked by the DATA's own task-R2, that count as "
        "'where the data responds' — the stratum the report headlines and the verdict "
        "is judged on. A whole-brain median is mostly non-responding tissue and buries "
        "a real effect confined to active cortex. Default 0.1.",
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
        "-allow_neg",
        action="store_true",
        help="Keep negative values in the corrected series/mean. By default they are clamped "
        "to 0 (fMRI magnitude data is non-negative; wsinc5/lanczos resampling rings negative "
        "near sharp edges). Does not affect the signed flow/warp/PC outputs.",
    )
    out.add_argument(
        "-want_pcs",
        "-want-pcs",
        nargs="?",
        type=int,
        const=5,
        default=None,
        metavar="N",
        help="Also save the top-N temporal PCs of the warp as {prefix}_locomoco_pcs.1D — "
        "structured residual-motion regressors that are strong denoising nuisances (the "
        "same thing ffs_util_pcwarp extracts post-hoc, here in-line). Bare flag = 5 PCs; "
        "give a number for more/fewer. Default: off.",
    )
    out.add_argument(
        "-save_mean",
        "-save-mean",
        action="store_true",
        help="Also write the temporal mean of the corrected series "
        "({prefix}_locomoco_mean), e.g. for use as a registration target. "
        "Independent of -no_corrected.",
    )
    out.add_argument(
        "-save_max",
        "-save-max",
        dest="save_max",
        action="store_true",
        help="Also write the temporal MAX of the corrected series "
        "({prefix}_locomoco_max): motion carries edge voxels out of the FoV, where "
        "the resampler leaves 0, so the max is the union of everything ever imaged "
        "— a fuller alignment target than the mean, which dims those edges.",
    )
    out.add_argument(
        "-save_min",
        "-save-min",
        dest="save_min",
        action="store_true",
        help="Also write the temporal MIN of the corrected series "
        "({prefix}_locomoco_min): 0 wherever ANY volume lost the voxel, so >0 is "
        "exactly the region with complete data at every timepoint.",
    )
    out.add_argument(
        "-save_first_last",
        "-save-first-last",
        dest="save_first_last",
        action="store_true",
        help="Save the first & last corrected volumes as one switchable file "
        "({prefix}_locomoco_firstlast) — flip between them in a viewer to see how "
        "well the correction worked.",
    )
    out.add_argument(
        "-save_first_last_diff",
        "-save-first-last-diff",
        dest="save_first_last_diff",
        action="store_true",
        help="Like -save_first_last but the file also carries a third volume, the "
        "difference (last - first): {prefix}_locomoco_firstlastdiff. The signed "
        "difference is kept un-clamped.",
    )
    out.add_argument(
        "-save_tsnr",
        "-save-tsnr",
        dest="save_tsnr",
        action="store_true",
        help="Save a temporal-SNR map (temporal mean / temporal std) of the "
        "corrected series ({prefix}_locomoco_tsnr) — a QC map of where the "
        "correction left clean signal.",
    )
    out.add_argument(
        "-save_initial",
        "-save-initial",
        dest="save_initial",
        action="store_true",
        help="Also emit the requested QC files (first/last, diff, tSNR) for the "
        "ORIGINAL uncorrected data ({prefix}_orig_firstlast, {prefix}_orig_tsnr), "
        "for a before/after comparison. Defaults to plain first/last when no other "
        "QC flag is given.",
    )
    out.add_argument(
        "-save_confidence",
        "-save-confidence",
        action="store_true",
        help="[xcorr] Also write the searchlight confidence map "
        "({prefix}_locomoco_confidence, 4-D) — per-voxel peak quality × prominence, the "
        "weight the field-regularizer (-reg_sigma) trusts. High where the shift is "
        "well-determined, ~0 in dropout/noise. A diagnostic for where the warp is real.",
    )
    out.add_argument(
        "-save_corr_curve",
        "-save-corr-curve",
        nargs="?",
        type=int,
        const=-1,
        default=None,
        metavar="FRAME",
        help="[xcorr] Also write the full per-voxel correlation-vs-offset "
        "SEARCH LANDSCAPE for one frame ({prefix}_locomoco_corrcurve, 4-D: x,y,z,offset). "
        "The 4th axis is the trial shift from -max_shift to +max_shift (step -xcorr_step); "
        "scrub it like a timeseries to see each voxel's correlation curve and judge whether "
        "the peak is clean/unimodal. Bare flag = middle frame; give a frame index to pick. "
        "The offsets are printed and written to {prefix}_locomoco_corrcurve_offsets.1D.",
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
    add_device_arg(
        hw,
        extra="On Apple Silicon, auto uses CPU; pass mps explicitly to experiment with Metal.",
    )
    add_batch_args(
        p,
        tool="ffs_locomoco",
        what="residual-motion estimates",
        example="-input run1.nii.gz -prefix nlmoco_run1.nii.gz -pe_dir y",
        skip_note="-prefix (its _warp and any -save_mean/-save_max/-save_min images)",
    )
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
    "warp_interp": "auto",
    "refine": 0,
}
_PRESETS = {
    "workhard": {
        "iters": 8,
        "levels": 4,
        "stride": 4,
        "xcorr_step": 0.25,
        "refine": 1,
    },
    "superhard": {
        "iters": 16,
        "levels": 5,
        "stride": 2,
        "xcorr_step": 0.25,
        "refine": 3,
    },
}


def _resolve_warp_interp(requested: str, *, rotaware: bool) -> str:
    """Resolve the CLI's geometry-aware high-fidelity default."""
    if requested != "auto":
        return requested
    return "bicubic" if rotaware else "lanczos"


def _warp_kernel_label(interp: str, radius: int) -> str:
    """User-facing name for the actual correction resampler."""
    return f"Lanczos-{radius}" if interp == "lanczos" else interp


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


def _resolve_tr(args, img) -> float | None:
    """TR for the task design: -tr wins, else the header's pixdim[4].

    Returns None (with a message) rather than guessing — a wrong TR misaligns every
    block, and the failure is silent in the worst way: the design still builds, still
    fits, and reports "no coupling".

    An implausible header value is REFUSED rather than used. Plenty of real EPI headers
    carry a pixdim[4] in the wrong units or left at a slice time (0.0534 on the run that
    exposed this), which would make a 120-frame run 6.4 seconds long and push every
    event past the end of it.
    """
    if args.tr is not None:
        return float(args.tr)
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 0.0

    def _refuse(why: str) -> None:
        print(f"  ⚠️  task coupling skipped: {why}")
        if args.detask_field:
            # Never let a requested FIX vanish quietly just because the diagnostic
            # could not run — the outputs would silently be the uncleaned field.
            print("  ⚠️  -detask NOT applied — the outputs below are the ORIGINAL field.")

    if tr <= 0:
        _refuse("no TR in the header — pass -tr SEC.")
        return None
    if not 0.1 <= tr <= 20.0:
        _refuse(
            f"header TR is {tr:g}s, which is not a plausible fMRI TR (expected "
            "0.1-20 s). 3-D EPI often puts the shot time in pixdim[4] rather than the "
            "volume TR — pass -tr SEC explicitly."
        )
        return None
    return tr


def _task_design_from_events(args, n_timepoints: int, tr: float, device):
    """Convolved (T, K) task design for the coupling diagnostic, plus condition labels.

    Goes through the same ``parse_bids_events`` -> ``build_task_design`` path every
    other ffs GLM uses, so a design that works for -events elsewhere works here.
    """
    import torch

    from fastfuncstuff.design.bids_events import parse_bids_events
    from fastfuncstuff.design.builder import spm_canonical_hrf
    from fastfuncstuff.design.matrices import build_task_design, commensurate_microtime_dt

    onsets, durations, labels = parse_bids_events(
        event_files=list(args.events),
        event_ignore=args.event_ignore,
        event_cols=tuple(args.event_cols) if args.event_cols else None,
        n_runs=1,
    )
    # Catch a wrong TR HERE, where the numbers are still interpretable, rather than
    # letting an all-zero design fall through to an empty condition list far downstream.
    run_seconds = n_timepoints * tr
    latest = max(
        (float(o.max()) for cond in onsets for o in cond if len(o)),
        default=0.0,
    )
    if latest >= run_seconds:
        raise ValueError(
            f"every event starts at or after the end of the run: last onset "
            f"{latest:g}s, run length {run_seconds:g}s ({n_timepoints} frames x "
            f"{tr:g}s TR). The TR is almost certainly wrong — pass -tr SEC."
        )

    dt = commensurate_microtime_dt(tr)
    hrf = torch.tensor(spm_canonical_hrf(tr=dt), dtype=torch.float64, device=device)
    design = build_task_design(
        hrf_bases=hrf,
        n_timepoints=n_timepoints,
        run_starts=[0],
        tr=tr,
        microtime_dt=dt,
        event_onsets=onsets,
        durations=durations,
        device=device,
    )
    keep = [k for k in range(design.shape[1]) if float(design[:, k].abs().max()) > 0]
    if not keep:
        raise ValueError(
            f"the task design is all zeros for every condition at TR {tr:g}s over "
            f"{n_timepoints} frames. Check -tr and that the events file covers this run."
        )
    if len(keep) < design.shape[1]:
        dropped = [labels[k] for k in range(design.shape[1]) if k not in keep]
        print(f"  ⚠️  dropping condition(s) with no events in this run: {', '.join(dropped)}")
        design = design[:, keep]
        labels = [labels[k] for k in keep]
    return design, labels


def parse_ref_mode(ref: str) -> tuple[bool, str, str]:
    """Split ``-ref`` into ``(paired?, within-reference statistic, display label)``.

    ``-ref`` names two independent things: WHICH frames form a reference, and WHAT
    statistic reduces them. ``paired[_mean|_median]`` follows the compound-name shape
    the progressive ``first_mean`` / ``first_median`` modes already established, so the
    CLI stays one flag while the library keeps taking a plain ``ref_mode`` alongside an
    orthogonal ``paired_bins``.

    The label is returned separately because every banner must keep the name the user
    typed: reporting the split-off statistic as ``ref=median`` would hide that the run
    was paired at all.

    There is deliberately no ``paired_max``. A temporal max fills FoV dropout across a
    whole run; over one bin's ~15 frames it is a noise envelope, not a template.
    """
    paired = ref.startswith("paired")
    if not paired:
        return False, ref, ref
    return True, ("median" if ref.endswith("_median") else "mean"), ref


def _paired_bins_for(args, n_timepoints: int, tr: float, device):
    """Per-frame task-state bins for ``-ref paired``, or None with a printed reason.

    Built BEFORE the estimation, because the bins choose the reference each frame is
    registered against — unlike the diagnostic, which only reads the field afterwards.
    """
    from fastfuncstuff.design.binning import design_state_bins, format_bin_report

    design, labels = _task_design_from_events(args, n_timepoints, tr, device)
    bin_of, info = design_state_bins(
        design,
        bin_width=args.task_bin_width,
        n_bins=args.task_bins,
        min_frames=args.task_min_frames,
    )
    print(format_bin_report(info, labels))
    if info["n_bins"] < 2:
        print(
            "  ⚠️  -ref paired: every frame landed in ONE bin, which is just the plain "
            "reference. Widen the design or lower -task_bin_width."
        )
        return None
    return bin_of


def parse_detask(value):
    """``-detask`` MODE -> ``(clean_field, notch_widen, fit_deriv)``.

    ``None`` -> (False, None, None); ``field`` -> (True, None, None); ``filter`` /
    ``filter:N`` -> (False, N, None); ``fit`` / ``fit:D`` -> (False, None, D). The
    modes are alternatives, not a pipeline: both estimator-side cuts (filter, fit)
    remove the task before the estimator runs, so there is nothing left for the field
    projection to take out, and running two would only spend the degrees of freedom
    twice.
    """
    if value is None:
        return False, None, None
    text = str(value).strip().lower()
    if text == "field":
        return True, None, None
    if text == "ica" or text.startswith("ica:"):
        # Validated here so a bad rank fails at parse time alongside the other modes;
        # the spec itself is re-read by parse_detask_ica where it is used.
        parse_detask_ica(text)
        return False, None, None
    if text == "filter":
        return False, 0, None
    if text.startswith("filter:"):
        tail = text.split(":", 1)[1]
        try:
            widen = int(tail)
        except ValueError:
            raise ValueError(
                f"-detask filter:N needs an integer widening in bins, got {tail!r}"
            ) from None
        if widen < 0:
            raise ValueError(f"-detask filter:N needs N >= 0, got {widen}")
        return False, widen, None
    if text == "fit":
        return False, None, 0
    if text.startswith("fit:"):
        tail = text.split(":", 1)[1]
        try:
            deriv = int(tail)
        except ValueError:
            raise ValueError(
                f"-detask fit:D needs an integer derivative count, got {tail!r}"
            ) from None
        if deriv < 0:
            raise ValueError(f"-detask fit:D needs D >= 0, got {deriv}")
        return False, None, deriv
    raise ValueError(
        f"-detask MODE must be 'field', 'filter[:N]', 'fit[:D]' or 'ica[:N]', got {value!r}"
    )


def parse_detask_ica(value):
    """``-detask ica`` SPEC -> ``(n_components, variance_fraction)`` or ``("sweep", None)``.

    Lives beside :func:`parse_detask` rather than inside it because the ICA mode is the
    only one whose parameter is a RANK, and the rank has an interior optimum: measured,
    20 components missed the task source (1.75x enrichment), 60 found it (2.79x), and
    the full 119 over-split it back down (2.27x). Bare ``ica`` therefore SEARCHES for
    the rank instead of guessing one -- a variance fraction was the original default
    and is a bad one, resolving to 105 of 119 components on a real run, deep in the
    over-splitting regime.
    """
    text = str(value).strip().lower()
    _, _, tail = text.partition(":")
    if tail in ("", "sweep", "all"):
        return "sweep", None
    try:
        num = float(tail)
    except ValueError:
        raise ValueError(
            f"-detask ica:N needs an integer rank, a variance fraction below 1, or "
            f"'sweep', got {tail!r}"
        ) from None
    if num <= 0:
        raise ValueError(f"-detask ica:N needs a positive value, got {tail!r}")
    if num < 1:
        return None, num
    if float(num).is_integer():
        return int(num), None
    raise ValueError(f"-detask ica:N needs an integer rank or a fraction, got {tail!r}")


def _detask_pre_estimation(args) -> bool:
    """Is a mode active that changes what the ESTIMATOR is shown (rather than the field)?

    Both ``filter`` and ``fit`` cut the task out of the images before the solve, so
    every site that re-derives the corrected series from the untouched input has to
    fire for either of them. Keeping the test in one place is what stops a third mode
    from being wired into three of the four gates.
    """
    return args.detask_widen is not None or args.detask_fit is not None


def _detask_mode_name(args) -> str:
    """Which estimator-side mode is active, for messages that must name it."""
    return "fit" if args.detask_fit is not None else "filter"


def _pe_axis_name(label: str, axis: int, args) -> str:
    """``pe1 (primary PE AP, axis 1)`` — the filename stem first, then what it means.

    Every task-coupling number is reported twice, once per encode axis, and the two
    blocks used to be headed "PE displacement" and "partition displacement" with the
    ``pe1``/``pe2`` suffix that names the files on disk appearing nowhere. Lead with
    the suffix so a block and a ``_taskr_pe2.nii.gz`` are obviously the same thing.
    """
    letter = "xyz"[axis] if 0 <= axis < 3 else "?"
    # A -pe_dir2-only run still labels its single field "pe1" (pe_displacements has no
    # second axis to report), but that field IS the partition -- the same case the
    # setup path handles by resolving pe_axis from pe_dir2. Calling it "primary PE"
    # here would name it the one thing it is not.
    partition_only = not args.pe_dir and args.pe_dir2 is not None
    if label == "pe1" and not partition_only:
        direction = args.pe_dir[0] if args.pe_dir else None
        kind = "primary PE"
    else:
        direction = args.pe_dir2 or (
            args.pe_dir[1] if args.pe_dir and len(args.pe_dir) > 1 else None
        )
        kind = "partition"
    dir_txt = f" {direction}" if direction else ""
    return f"{label} ({kind}{dir_txt}, axis {axis} = {letter})"


def _coupling_stataux(n_t: int, polort: int, n_sub: int) -> dict:
    """AFNI ``fico`` parameters for a drift-partial correlation map, per sub-brick.

    SAMPLES = timepoints, FIT-PARAMETERS = 1 (the single regressor each sub-brick is
    correlated against), ORT-PARAMETERS = polort+1 (the Legendre drift columns removed
    from both sides first, degrees 0..polort).

    AFNI reads these three straight into ``correl_t2p(rho, nsam, nfit, nort)``
    (mri_stats.c), which is ``incbeta(1 - rho^2, (nsam-nfit-nort)/2, nfit/2)`` — the
    multiple-correlation null with ``nsam-nfit-nort`` residual dof. Our ``r`` is a
    correlation between two vectors already projected out of a (polort+1)-dimensional
    drift subspace, with one fitted parameter, so that is exactly T-(polort+1)-1. AFNI
    has no R-squared stat type at all; ``fico`` is the only correlation code, and it is
    the right one here because it keeps the SIGN that an R-squared would throw away.

    That "under independence" is the whole caveat. fMRI residuals are autocorrelated and
    nothing here corrects for it, so the p AFNI derives from this is NOMINAL and
    anticonservative; :mod:`fastfuncstuff.stats.task_coupling` deliberately makes no
    significance claim and this does not change that. The tag is here so the maps
    threshold and colour like every other functional overlay, not so the p can be
    reported.
    """
    from fastfuncstuff.io.afni import stat_type_to_stataux

    return {i: stat_type_to_stataux("fico", (n_t, 1, polort + 1)) for i in range(n_sub)}


def _save_map(path, arr, labels, affine, stataux=None):
    """Write one labelled map: 4-D with named sub-bricks, 3-D if there is only one."""
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    a = arr.float()
    # A single-condition map has always been written 3-D here; keep that spelling.
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a.squeeze(-1)
    n_sub = 1 if a.ndim == 3 else a.shape[-1]
    with spinner(f"Writing {Path(path).name}"):
        save_nifti(
            a.numpy(),
            path,
            affine=affine,
            brick_labels=list(labels[:n_sub]),
            brick_stataux=stataux,
        )


def _save_task_map(path, arr, labels, affine, *, stataux_polort=None, n_t=0):
    """A correlation-only map (the FIELD's coupling), every sub-brick tagged ``fico``."""
    a = arr.float()
    n_sub = 1 if (a.ndim == 3 or a.shape[-1] == 1) else a.shape[-1]
    stataux = None if stataux_polort is None else _coupling_stataux(n_t, stataux_polort, n_sub)
    _save_map(path, a, labels, affine, stataux)


def _save_task_fit(path, tc, psc, labels, affine, polort, n_t):
    """One AFNI-style bucket per condition: amplitude, then the stat to threshold it on.

    Sub-bricks alternate ``{cond}_Coef`` (percent signal change) and ``{cond}_Correl``
    (the signed partial correlation, tagged ``fico``), so AFNI's usual "view the beta,
    threshold on the sub-brick after it" gesture works without opening two datasets and
    keeping their indices lined up by hand.
    """
    import torch

    bricks, names, stataux = [], [], {}
    stat = _coupling_stataux(n_t, polort, 1)[0]
    for k, lb in enumerate(labels):
        bricks += [psc[..., k].float(), tc.r[..., k].float()]
        names += [f"{lb}_Coef", f"{lb}_Correl"]
        stataux[2 * k + 1] = stat
    _save_map(path, torch.stack(bricks, dim=-1), names, affine, stataux)


def _psc_betas(tc, design, reference, mask):
    """Condition betas as PERCENT SIGNAL CHANGE of each voxel's own temporal mean.

    ``tc.beta`` is map units per unit of regressor, so the response a condition actually
    produces is ``beta × the regressor's peak-to-trough swing``; dividing by the voxel
    mean and scaling by 100 puts every condition, voxel and dataset on the one axis a
    reader can judge — a 2% response is a 2% response whatever the scanner's arbitrary
    intensity units were. Nothing extra is fitted: the betas come out of the same solve
    as ``r``, at no additional cost.
    """
    import torch

    x = torch.as_tensor(np.asarray(design), dtype=torch.float32)
    swing = (x.amax(dim=0) - x.amin(dim=0)).clamp(min=1e-12)
    base = reference.float().abs()
    ok = (mask > 0) & (base > 0)
    out = torch.zeros(*tc.beta.shape[:3], tc.beta.shape[-1], dtype=torch.float32)
    for k in range(out.shape[-1]):
        out[..., k] = torch.where(
            ok,
            100.0 * tc.beta[..., k].float() * float(swing[k]) / base.clamp(min=1e-12),
            torch.zeros_like(base),
        )
    return out


def _write_task_diagnostics(result, data, stem, ext, affine, args, tr):
    """Measure how task-locked the estimated field is, and whether that is BOLD.

    The statistic, the null and the strata are explained in
    :mod:`fastfuncstuff.stats.task_coupling`. Shared by the single- and multi-echo
    paths: both expose ``pe_displacements()``, so each encode axis is scored on its
    own — the primary-PE and partition fields can be contaminated to different
    degrees and one pooled number would hide it.
    """
    import numpy as np
    import torch

    from fastfuncstuff.processing.locomoco import _brain_mask_from
    from fastfuncstuff.stats.task_coupling import (
        co_location,
        contamination_slope,
        default_polort,
        enrichment_curve,
        format_task_coupling_report,
        pe_gradient,
        responding_mask,
        task_coupling,
        task_enrichment,
    )

    n_t = data.shape[3]
    design, labels = _task_design_from_events(args, n_t, tr, torch.device("cpu"))
    polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
    print(
        f"  Task coupling: {len(labels)} condition(s) ({', '.join(labels)}), "
        f"TR {tr:g}s, polort {polort}"
    )

    series = torch.as_tensor(np.ascontiguousarray(data))
    reference = series.mean(dim=3)
    mask = _brain_mask_from(reference.abs())

    kwargs = dict(polort=polort, mask=mask, labels=labels)
    # The BOLD response itself, measured the same way — the reference the field's
    # coupling is compared AGAINST, and what defines "where the data responds".
    data_tc = task_coupling(series, design, **kwargs)
    resp, quiet, cut = responding_mask(data_tc.r, mask, args.task_top_frac, thresh=args.task_thresh)
    how = "cut" if args.task_thresh is not None else f"top {args.task_top_frac * 100:.0f}%"
    print(
        f"  Active mask: {int(resp.sum())} voxels with data |r| > {cut:.3f} ({how}) "
        f"= {100 * int(resp.sum()) / max(1, int((mask > 0).sum())):.1f}% of the brain"
    )

    report: list[str] = []
    pending: list[tuple] = []
    for label, axis, field in result.pe_displacements():
        tc = task_coupling(field, design, **kwargs)
        coloc = co_location(tc.r, data_tc.r, mask)
        r_sum, q_sum = tc.summarize(resp), tc.summarize(quiet)
        name = _pe_axis_name(label, axis, args)
        # The gradient must be along THIS axis: a partition-direction displacement
        # produces an intensity change through the partition-direction gradient, and
        # using the primary-PE gradient for it would test the wrong relation.
        conds = r_sum.get("conditions") or []
        if not conds:
            raise ValueError("no voxel in the active mask carries temporal variance in the field")
        best_k = max(range(len(conds)), key=lambda k: conds[k]["abs_r_median"])
        slope = contamination_slope(
            tc.beta, data_tc.beta, pe_gradient(reference, axis), resp, condition=best_k
        )
        enrich = task_enrichment(tc, resp, mask)
        curve = enrichment_curve(tc, resp, mask)
        report.append(
            format_task_coupling_report(
                tc,
                data_tc,
                coloc,
                units="voxels",
                label=name,
                responding=r_sum,
                quiet=q_sum,
                top_frac=args.task_top_frac,
                slope=slope,
                enrichment=enrich,
                active_thresh=cut,
                curve=curve,
            )
        )
        best = r_sum["conditions"][best_k]
        tail = curve[-1] if curve else None
        tail_txt = (
            f"tail enrichment {tail['enrichment']:.2f}x of {tail['ceiling']:.0f}x "
            f"(field |r| > {tail['r_cut']:.2f})"
            if tail
            else f"enrichment {enrich['enrichment']:.2f}x"
        )
        # Each axis gets its own banner: the two findings lines per axis ran together
        # into one block, and pe1 vs pe2 is the first thing a reader is looking for.
        print_cli_subsection(name)
        print(
            f"  {tail_txt}, |r| {best['abs_r_median']:.3f} med / "
            f"{best['abs_r_p95']:.3f} p95 in the active mask ({best['label']}), "
            f"kappa {slope['kappa']:+.3f}"
        )
        # The verdict sentence, and only that. The full stratum table goes to the .txt:
        # it was echoed here in full for every axis, which buried the one line that
        # actually says what to do.
        print(f"  {report[-1].rstrip().splitlines()[-1].strip()}")
        # One 4-D file per axis, conditions on the 4th axis — a viewer scrubs the
        # conditions, and one file per condition would flood the output directory.
        # Queued, not written here: the findings above are what a reader scans, and
        # interleaving file names between them is what made this block unreadable.
        pending += [
            (f"{stem}_taskr_{label}{ext}", tc.r, labels, polort),
            (f"{stem}_taskrms_{label}{ext}", tc.task_rms, [f"{label} task rms (vox)"], None),
        ]

    Path(f"{stem}_locomoco_taskcoupling.txt").write_text(("\n" + "-" * 70 + "\n\n").join(report))
    print()
    for path, arr, brick_labels, stat_polort in pending:
        _save_task_map(path, arr, brick_labels, affine, stataux_polort=stat_polort, n_t=n_t)
    # The data side: the task response as it stands BEFORE any correction, as one
    # bucket of alternating amplitude and stat. `r` says where the response is, the PSC
    # beta says how big it is, and the amplitude is the half that moves when a
    # contaminated field drags a responding voxel toward the mean or blurs it.
    psc_before = _psc_betas(data_tc, design, reference, mask)
    _save_task_fit(f"{stem}_taskfit_data{ext}", data_tc, psc_before, labels, affine, polort, n_t)
    args._task_labels = labels
    # Everything the data-side "after" needs, so it can run at the point the corrected
    # series already exists rather than paying for a second 4-D warp to get one.
    args._task_after = {
        "design": design,
        "polort": polort,
        "resp": resp,
        "mask": mask,
        "labels": labels,
        "before": data_tc,
        "before_psc": psc_before,
        "reference": reference,
        "affine": affine,
        "ext": ext,
    }
    return design, polort, resp, mask


def _pc_task_correlation(components, design, polort) -> float:
    """|corr| of the top temporal PC of the field with the drift-residualized task.

    The number that says whether the nuisance regressors are safe to use. Measured
    against the DETRENDED regressor because the GLM these PCs enter carries
    polynomials too, so the slow component the task shares with drift is not the
    task's to claim.
    """
    import torch

    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.stats.task_coupling import _orthonormal_basis

    n_t = components[0][1].shape[-1]
    m = torch.cat([c.reshape(-1, n_t) for _, c in components], dim=0).double()
    m = m - m.mean(dim=1, keepdim=True)
    keep = m.norm(dim=1) > 0
    if not bool(keep.any()):
        return 0.0
    pc = torch.linalg.svd(m[keep], full_matrices=False)[2][0]
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=pc.device, dtype=torch.float64)
    )
    x = torch.as_tensor(design, dtype=torch.float64)[:, 0]
    x = x - q_n @ (q_n.T @ x)
    denom = float(pc.norm() * x.norm())
    return abs(float(pc.dot(x) / denom)) if denom > 0 else 0.0


def parse_warp_recon(value):
    """``-warp_recon`` SPEC -> ``(count, variance_fraction, method)``.

    ``pcs`` keeps every principal component (exact), ``pcs:N`` the top N, ``pcs:0.8``
    however many reach 80% of the variance. ``ica`` runs FastICA over a PCA reduction
    that defaults to 95% of the variance, ``ica:N`` / ``ica:0.F`` set that rank
    explicitly.

    A value below 1 is read as a FRACTION, which is unambiguous because a count below
    1 is meaningless, and it is the more portable request -- the number of components
    worth keeping is a property of the run, not a constant.
    """
    if value is None:
        return None, None, None
    text = str(value).strip().lower()
    method, _, tail = text.partition(":")
    if method == "ica":
        raise ValueError(
            "-warp_recon ica moved to '-detask ica' (rejection implied). -warp_recon "
            "reconstructs from principal components; ICA never reconstructed at all — "
            "it PROJECTS the task-loaded time courses out of the full-rank field, "
            "which is a de-tasking operation and belongs with the other three."
        )
    if method != "pcs":
        raise ValueError(f"-warp_recon SPEC must start with 'pcs', got {value!r}")
    # ICA has no natural "all": it needs a rank to reduce to, and the full rank
    # over-splits (measured: 60 components found the task source, 119 lost it again).
    if tail == "sweep":
        raise ValueError("-warp_recon sweep is only defined for -detask ica")
    if tail in ("", "all"):
        return None, None, method
    try:
        num = float(tail)
    except ValueError:
        raise ValueError(
            f"-warp_recon {method}:N needs an integer, a variance fraction below 1, or "
            f"'all', got {tail!r}"
        ) from None
    if num <= 0:
        raise ValueError(f"-warp_recon {method}:N needs a positive value, got {tail!r}")
    if num < 1:
        return None, num, method
    if float(num).is_integer():
        return int(num), None, method
    raise ValueError(
        f"-warp_recon {method}:{tail} is ambiguous: below 1 is a variance fraction, "
        "1 or more must be a whole number of components."
    )


def _free_device_cache(device) -> None:
    """Hand the caching allocator's unused blocks back before a large allocation.

    Freeing a tensor returns it to PyTorch's pool, not to the driver, so a later
    allocation of a DIFFERENT size can still fail with plenty nominally free. The warp
    decomposition and the projection that follows are both whole-field sized, which is
    exactly the case where this matters -- see [[VRAM debugging]].
    """
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def _report_vram(device, where: str) -> None:
    """One line of device memory, so an OOM can be attributed rather than guessed at."""
    if device is None or getattr(device, "type", None) != "cuda":
        return
    free_b, total_b = torch.cuda.mem_get_info(device)
    gb = 1024.0**3
    print(
        f"  VRAM {where}: {(total_b - free_b) / gb:.1f} GB used of {total_b / gb:.1f} GB "
        f"({free_b / gb:.1f} GB free; torch reserves "
        f"{torch.cuda.memory_reserved(device) / gb:.1f} GB)"
    )


def _ica_rank_sweep(comps, resp, mask, ranks, device):
    """Best ICA rank by peak task enrichment — the rank is not guessable a priori.

    Measured on one real field: 20 components missed the task source entirely (1.75x),
    60 found it (2.79x), 105 and 119 lost it again to over-splitting (1.42x, 2.27x).
    There is an interior optimum and no variance rule locates it -- 95% of the variance
    landed on 105, deep in the failing regime -- so it is searched for instead.

    Coarse grid then a local refine rather than a trisection: the criterion is not
    guaranteed unimodal, and one ICA fit is seconds, so a grid that cannot be fooled is
    cheaper than a bisection that can.

    The grid is weighted toward LOW ranks and self-extends when the winner lands on an
    edge. An argmax on the boundary means the optimum was never searched, not that the
    boundary is optimal -- seen on a real run, where the old 0.15 floor (rank 33) won
    outright with enrichment falling monotonically above it.

    A bias worth knowing when reading the numbers. ``peak`` is a MAX over the k
    components, so a larger k gets more draws and is favoured by chance alone -- the
    criterion is biased UPWARD in rank. So a sweep in which enrichment *falls* with rank
    is stronger evidence for a low optimum than it looks, because the bias points the
    other way. It also means peak enrichment is not comparable across ranks in absolute
    terms; it is used to rank candidates, not to quantify contamination. The enrichment
    reported in the diagnostic, which is computed on the field rather than on a
    max-over-components, is the number to quote.
    """
    from fastfuncstuff.processing.locomoco import warp_ica_basis
    from fastfuncstuff.stats.task_coupling import map_enrichment

    def score(rank):
        got = warp_ica_basis(comps, n_components=rank, pca_components=rank, device=device)
        # Each fit builds TWO (T, S) copies of the whole field on the device -- 5.2 GB
        # at 160x160x114x225 in float32 -- and the returned basis/loadings are already
        # on the CPU. Without this the caching allocator holds every rank's working set
        # until the process exits, and the sweep alone can fill a 16 GB card.
        _free_device_cache(device)
        if got is None:
            return 0.0, None
        basis, loadings, means, var = got
        k = basis.shape[1]
        peak = max(
            map_enrichment(load[..., i], resp, mask)["enrichment"]
            for _ax, load in loadings
            for i in range(k)
        )
        return peak, (basis, loadings, means, var)

    seen: dict[int, float] = {}
    best_rank, best_peak, best_got = ranks[0], -1.0, None

    def _try(rank, tag=""):
        nonlocal best_rank, best_peak, best_got
        if rank in seen or rank < 2:
            return
        peak, got = score(rank)
        seen[rank] = peak
        print(f"    rank {rank:4d}: peak enrichment {peak:.2f}x{tag}")
        if peak > best_peak:
            best_rank, best_peak, best_got = rank, peak, got

    for rank in ranks:
        _try(rank)

    # An argmax on the EDGE of the grid means the grid is wrong, not that the edge is
    # optimal -- the optimum is outside what was searched. Extend outward until the
    # winner is interior or the bound is reached. Measured on a real run this mattered:
    # the grid floor (rank 33 of a 0.15-0.8 span) won outright with enrichment falling
    # monotonically above it, so the true optimum was never in the search at all.
    #
    # Extending DOWN is nearly free -- a low-rank ICA is the fastest fit and the
    # smallest allocation -- which is the other reason not to just widen the fixed grid
    # and pay for the high ranks that were already losing.
    for _ in range(5):
        order = sorted(seen)
        if best_rank == order[0] and best_rank > 2:
            cand = max(2, best_rank // 2)
        elif best_rank == order[-1] and best_rank < ranks[-1]:
            cand = min(ranks[-1], best_rank + max(1, best_rank - order[-2]))
        else:
            break
        if cand in seen:
            break
        _try(cand, "  (edge)")

    # One refine pass halfway to each neighbour of the winner.
    order = sorted(seen)
    i = order.index(best_rank)
    refine = []
    if i > 0:
        refine.append((order[i - 1] + best_rank) // 2)
    if i < len(order) - 1:
        refine.append((best_rank + order[i + 1]) // 2)
    for rank in refine:
        _try(rank, "  (refine)")
    edge = " — AT THE GRID EDGE, treat as a lower bound" if best_rank == min(seen) else ""
    print(f"    → rank {best_rank} wins at {best_peak:.2f}x{edge}")
    return best_rank, best_got


def _write_task_after(result, design, polort, resp, mask, stem, ext, affine, args):
    """Re-measure task coupling on the FIXED field and write the after maps.

    The diagnostic above deliberately measures the field AS ESTIMATED -- scoring the
    fix's own output would always report success. But that leaves nothing to compare
    against, so the same measurement is repeated on the cleaned field and written with
    an ``_after`` suffix. Two ``_taskr`` maps of the same run, before and after, are
    what actually answer "did it work".
    """
    from fastfuncstuff.stats.task_coupling import (
        enrichment_curve,
        task_coupling,
        task_enrichment,
    )

    print_cli_subsection("TASK COUPLING AFTER THE FIX")
    pending: list[tuple] = []
    for label, axis, field in result.pe_displacements():
        name = _pe_axis_name(label, axis, args)
        tc = task_coupling(field, design, polort=polort, mask=mask, labels=args._task_labels)
        summary = tc.summarize(resp)
        curve = enrichment_curve(tc, resp, mask)
        tail = curve[-1] if curve else None
        best = max(summary["conditions"], key=lambda c: c["abs_r_median"])
        tail_txt = (
            f"tail enrichment {tail['enrichment']:.2f}x of {tail['ceiling']:.0f}x"
            if tail
            else f"enrichment {task_enrichment(tc, resp, mask)['enrichment']:.2f}x"
        )
        print_cli_subsection(name)
        print(
            f"  {tail_txt}, |r| {best['abs_r_median']:.3f} med / "
            f"{best['abs_r_p95']:.3f} p95 in the active mask"
        )
        pending.append((f"{stem}_taskr_{label}_after{ext}", tc.r))
    print()
    n_t = int(np.asarray(design).shape[0])
    for path, arr in pending:
        _save_task_map(path, arr, args._task_labels, affine, stataux_polort=polort, n_t=n_t)


def _write_data_task_after(corrected, stem, args):
    """Re-measure the task fit on the CORRECTED DATA, before/after.

    The field-side ``_after`` above answers "did the fix clean the field". This measures
    what the whole correction did to the task response itself. It runs whenever
    ``-events`` is given, with or without a fix: the comparison is a diagnostic either
    way, and it is how damage becomes visible rather than inferred.

    Both halves are reported because they are not redundant. A contaminated field drags
    responding voxels toward the mean and blurs; both cut the AMPLITUDE while ``|r|``
    can barely move. The maps are written as one paired bucket per state so the two can
    be compared voxel-wise, not just as the medians printed here.
    """
    st = getattr(args, "_task_after", None)
    if st is None:
        return
    if corrected is None:
        print(
            "  ⚠️  task fit AFTER not measured: no corrected series was materialized "
            "(-no_corrected)."
        )
        return

    from fastfuncstuff.stats.task_coupling import task_coupling

    print_cli_subsection("TASK FIT BEFORE → AFTER (data, medians in the active mask)")
    ext, affine, resp = st["ext"], st["affine"], st["resp"]
    series = torch.as_tensor(np.ascontiguousarray(np.asarray(corrected))).float()
    after = task_coupling(
        series,
        st["design"],
        polort=st["polort"],
        mask=st["mask"],
        labels=st["labels"],
    )
    # PSC is referenced to the ORIGINAL mean, not the corrected one: a changed baseline
    # would move every percentage without any response having changed.
    psc_after = _psc_betas(after, st["design"], st["reference"], st["mask"])
    psc_before = st["before_psc"]
    n_t = int(np.asarray(st["design"]).shape[0])

    sel = resp > 0
    for k, lb in enumerate(st["labels"]):
        r0 = st["before"].r[..., k][sel].abs()
        r1 = after.r[..., k][sel].abs()
        b0 = psc_before[..., k][sel].abs()
        b1 = psc_after[..., k][sel].abs()
        d_r = float(r1.median()) - float(r0.median())
        d_b = float(b1.median()) - float(b0.median())
        pct = 100.0 * d_b / float(b0.median()) if float(b0.median()) > 0 else 0.0
        print(
            f"  {lb}: |r| {float(r0.median()):.3f} → {float(r1.median()):.3f} "
            f"({d_r:+.3f}), amplitude {float(b0.median()):.3f} → "
            f"{float(b1.median()):.3f} %sig ({pct:+.1f}%)",
            flush=True,
        )
    _save_task_fit(
        f"{stem}_taskfit_data_after{ext}",
        after,
        psc_after,
        st["labels"],
        affine,
        st["polort"],
        n_t,
    )


def _warp_recon_stage(result, data, args, resp, mask, device, design=None, polort=2):
    """Rebuild the field from a warp decomposition, dropping task-loaded components.

    ``resp``/``mask`` come from the task diagnostic, so ``-reject`` inherits exactly the
    active mask the coupling report was computed on -- the alternative, a second
    threshold chosen here, would let the report and the fix disagree about where the
    task is.
    """
    import numpy as np

    from fastfuncstuff.processing.locomoco import (
        pc_reconstruct_result,
        warp_ica_basis,
        warp_pc_basis,
        warp_project_out,
        warp_reconstruct,
    )
    from fastfuncstuff.stats.task_coupling import map_enrichment

    if args.detask_ica:
        n_pcs, var_frac = parse_detask_ica(args.detask)
        method = "ica"
    else:
        n_pcs, var_frac, method = parse_warp_recon(args.warp_recon)
    comps = [(ax, f) for _, ax, f in result.pe_displacements()]

    if method == "ica" and n_pcs == "sweep":
        if resp is None:
            print("  ⚠️  -detask ica sweep needs -events to score ranks; using 60.")
            n_pcs, got = 60, warp_ica_basis(comps, 60, 60, device=device)
        else:
            n_t = comps[0][1].shape[-1]
            top = max(4, n_t - 1)
            # Weighted toward LOW ranks: over-splitting is the failure mode that has
            # actually been observed (60 found the task source, 119 lost it), the
            # optimum has landed on the old 0.15 floor on real data, and a low-rank fit
            # is both the fastest and the smallest allocation. The edge-extension below
            # covers whatever this still misses.
            grid = sorted({max(2, int(top * f)) for f in (0.05, 0.1, 0.15, 0.3, 0.5, 0.75)})
            print_cli_subsection("ICA RANK SWEEP — peak task enrichment over components")
            n_pcs, got = _ica_rank_sweep(comps, resp, mask, grid, device)
    elif method == "ica":
        got = warp_ica_basis(
            comps,
            n_components=n_pcs,
            pca_components=n_pcs if var_frac is None else var_frac,
            device=device,
        )
    else:
        got = warp_pc_basis(comps, n_pcs=None if var_frac is not None else n_pcs, device=device)
    if got is None:
        print("  ⚠️  the warp is empty — nothing to decompose.")
        return result
    basis, loadings, means, var = got

    if method == "pcs" and var_frac is not None:
        # Cumulative over the FULL basis, so the fraction means what it says rather
        # than a fraction of some earlier truncation.
        n_pcs = int(torch.searchsorted(torch.cumsum(var, 0), float(var_frac)).item()) + 1
        n_pcs = max(1, min(n_pcs, basis.shape[1]))
        basis, var = basis[:, :n_pcs].contiguous(), var[:n_pcs]
        loadings = [(ax, load[..., :n_pcs].contiguous()) for ax, load in loadings]
        print(
            f"  -warp_recon pcs:{var_frac:g} → {n_pcs} component(s) reach "
            f"{float(var.sum()):.1%} of the variance"
        )
    k = basis.shape[1]
    label_of = {ax: lbl for lbl, ax, _ in result.pe_displacements()}

    # ── the TEMPORAL criterion, scored once on the SHARED basis ──────────────────
    # The time courses are shared across encode axes, so this is one test per
    # component, not one per component per axis. It complements the spatial score
    # rather than replacing it: an energy share cannot see a task-locked component
    # that drives few voxels (its energy lives elsewhere), and a temporal fit cannot
    # see a widespread component that is not time-locked. Either firing is enough.
    temporal = None
    reject_on = args.reject is not None or args.detask_ica
    if reject_on and design is not None:
        from fastfuncstuff.stats.task_coupling import component_task_fit

        try:
            temporal = component_task_fit(
                basis, design, polort=polort, n_surrogates=args.reject_surrogates
            )
        except ValueError as exc:
            print(f"  ⚠️  temporal criterion unavailable: {exc}")
    if temporal is not None:
        if temporal["informative"]:
            print(
                f"  temporal criterion: {temporal['eff_dof']:.0f} effective DoF, a "
                f"component needs R²>{temporal['r2_needed']:.3f} to be flagged "
                f"(familywise α={temporal['alpha']:g} over {k} components, "
                f"{temporal['n_surrogates']} surrogates)"
            )
        else:
            # Saying WHY beats printing "none flagged", which a reader would take as
            # evidence of a clean field rather than an absent measurement.
            print(f"  ⚠️  temporal criterion declines: {temporal['uninformative_reason']}.")
            print("      The spatial criterion below is carrying the decision alone.")

    keep, dropped, best = {}, {}, 0.0
    temporal_bad = set(temporal["flagged"]) if temporal else set()
    for axis, load in loadings:
        idx = list(range(k))
        if reject_on and resp is not None:
            cut = args.reject if args.reject is not None else 2.0
            scores = [map_enrichment(load[..., i], resp, mask)["enrichment"] for i in range(k)]
            best = max(best, max(scores, default=0.0))
            bad = {i for i in range(k) if scores[i] > cut} | temporal_bad
            idx = [i for i in range(k) if i not in bad]
            dropped[axis] = [(i, scores[i]) for i in sorted(bad)]
        keep[axis] = idx

    if method == "ica":
        head = f"WARP TASK REJECTION — ICA rank {k}, projected from the FULL-rank field"
    else:
        span = f"top {k}" if n_pcs is not None else f"all {k}"
        head = f"WARP RECONSTRUCTION — {span} principal components"
    print_cli_subsection(head)
    cut_txt = f"{args.reject:g}" if args.reject is not None else "2"
    for axis, _load in loadings:
        lbl = label_of.get(axis, f"axis{axis}")
        drops = dropped.get(axis, [])
        if not reject_on:
            print(f"  • {lbl}: {len(keep[axis])} of {k} components kept")
        elif drops:
            # WHICH criterion fired, per component. The two find different things, so
            # collapsing them into one count would hide the fact that (say) every drop
            # came from the temporal test and the spatial one saw nothing -- which is
            # the expected picture for a sparse response and worth reading directly.
            def _tag(i: int, e: float) -> str:
                sp = e > (args.reject if args.reject is not None else 2.0)
                tp = i in temporal_bad
                if sp and tp:
                    return f"#{i} ({e:.1f}x, R²={float(temporal['r2'][i]):.2f})"
                if sp:
                    return f"#{i} ({e:.1f}x)"
                return f"#{i} (R²={float(temporal['r2'][i]):.2f})"

            det = ", ".join(_tag(i, e) for i, e in drops[:6])
            # The index is into the SHARED basis, so the same number on both axes is one
            # component scoring differently through two spatial loadings, not two finds.
            more = f" +{len(drops) - 6} more" if len(drops) > 6 else ""
            n_sp = sum(1 for i, e in drops if e > (args.reject if args.reject is not None else 2.0))
            n_tp = sum(1 for i, _e in drops if i in temporal_bad)
            print(
                f"  • {lbl}: dropped {len(drops)} task-loaded component(s) "
                f"({n_sp} spatial >{cut_txt}x, {n_tp} temporal) — {det}{more}"
            )
        else:
            both = (
                "either criterion"
                if temporal_bad or (temporal and temporal["informative"])
                else f">{cut_txt}x"
            )
            print(f"  • {lbl}: no component met {both} — nothing dropped")

    if method == "ica":
        # Reject by PROJECTION on the full-rank field, not by rebuilding from the kept
        # components: an ICA rank is fixed before the rotation, so a reconstruction
        # discards whatever the PCA reduction dropped even when nothing is rejected --
        # measured, 21.5% of a real field's rms for zero benefit. The decomposition's
        # only job is to name the bad time courses.
        bad_tc = {
            axis: basis[:, [i for i, _e in dropped.get(axis, [])]] for axis, _load in loadings
        }
        if any(v.shape[1] for v in bad_tc.values()):
            # The decomposition is done and its outputs live on the CPU; the projection
            # that follows is whole-field sized. Give the pool back first.
            _free_device_cache(device)
            if os.environ.get("FFS_DEBUG_VRAM"):
                _report_vram(device, "before the task projection")
            rebuilt = warp_project_out(comps, bad_tc, device=device)
        else:
            rebuilt = [(ax, f.float()) for ax, f in comps]
    else:
        rebuilt = warp_reconstruct(basis, loadings, means, keep)
    # What actually left the field, measured rather than inferred from a variance
    # ratio. Under ica this is purely what was rejected -- the projection touches
    # nothing else. Under pcs it also carries the truncation loss, which is why the two
    # are labelled differently rather than sharing one number a reader would misread.
    what = "removed" if method == "ica" else "removed + truncated away"
    for (axis, new_f), (_, old_f) in zip(rebuilt, comps, strict=True):
        lbl = label_of.get(axis, f"axis{axis}")
        o = old_f.float()
        resid = float((new_f - o).pow(2).mean().sqrt())
        print(
            f"    {lbl}: rms {float(o.std()):.4f} → {float(new_f.std()):.4f} vox; "
            f"{what} {resid:.4f} vox "
            f"({resid / max(float(o.std()), 1e-9) * 100:.1f}% of the field's rms)"
        )

    if reject_on and not any(dropped.values()):
        # A no-op here is a RESULT, not silence. PCA maximises variance, so a
        # contamination that is a small share of the field's energy cannot dominate any
        # component -- measured on a 0.8mm checkerboard run, the contaminated voxels
        # were 0.65% of the mask carrying 0.70% of the field's energy, and no principal
        # component scored above 1.25x while the field itself was 8.8x enriched at the
        # tail. ICA is not variance-ordered and does better on that run (2.8-3.0x), so
        # it is the thing to try before giving up on a decomposition.
        extra = (
            " Try -detask ica, which is not variance-ordered and reached 2.8-3.0x "
            "where PCA reached 1.25x on a real contaminated run, and which also applies "
            "the TEMPORAL criterion a sparse component can still trip."
            if method == "pcs"
            else ""
        )
        print(
            f"    ⓘ  no component's weights are concentrated on active tissue "
            f"(strongest {best:.2f}x vs the {cut_txt}x cut). If the diagnostic "
            "above says CONTAMINATION, the response is likely too SPARSE to isolate: "
            "it is a small share of the field's variance." + extra + " Otherwise use "
            "-detask filter (block designs) or -detask field."
        )
    if reject_on and any(dropped.values()) and resp is not None:
        # The design correlation of what was dropped, as corroboration. It cannot be
        # thresholded (a block design is ~2 DoF) but a dropped component whose time
        # course tracks the design is much easier to believe.
        design = getattr(args, "_task_design", None)
        if design is not None:
            x = np.asarray(design)[:, 0]
            x = x - x.mean()
            for axis, drops in dropped.items():
                if not drops:
                    continue
                lbl = label_of.get(axis, f"axis{axis}")
                cs = []
                for i, _e in drops[:4]:
                    tc = basis[:, i].numpy()
                    tc = tc - tc.mean()
                    d = np.linalg.norm(tc) * np.linalg.norm(x)
                    cs.append(f"#{i} |r|={abs(float(tc @ x / d)) if d > 0 else 0:.2f}")
                print(f"    {lbl} dropped components vs the design: {', '.join(cs)}")

    if any(dropped.values()):
        design = getattr(args, "_task_design", None)
        cols, names = [], []
        if design is not None:
            d = np.asarray(design)[:, 0].astype(float)
            cols.append((d - d.mean()) / max(d.std(), 1e-12))
            names.append("design")
        # ONE column per component, not per (component, axis). The temporal basis is
        # shared across the encode axes -- only the spatial loading differs -- so a
        # component rejected on both axes has the SAME time course twice, and writing
        # it twice reads as two findings when it is one.
        who: dict[int, list[str]] = {}
        for axis, drops in dropped.items():
            for i, _e in drops:
                who.setdefault(i, []).append(label_of.get(axis, f"axis{axis}"))
        for i in sorted(who):
            tc = basis[:, i].numpy().astype(float)
            cols.append((tc - tc.mean()) / max(tc.std(), 1e-12))
            names.append(f"ic{i:02d}_{'+'.join(sorted(who[i]))}")
        rpath = f"{args._stem}_locomoco_rejected.1D"
        # z-scored, and the design shipped in column 1: the reason to plot this is to
        # see whether what was removed tracks the task, and that comparison is unreadable
        # if the columns sit at different scales or the reference lives in another file.
        header = (
            "# z-scored; column 1 is the task design for comparison\n"
            "# one column per COMPONENT (the temporal basis is shared across encode\n"
            "# axes; the suffix names the axes it was rejected from)\n# "
            + "  ".join(f"{n:>12s}" for n in names)
        )
        np.savetxt(rpath, np.stack(cols, axis=1), fmt="%12.6f", header=header, comments="")

    out, note = pc_reconstruct_result(
        result,
        data,
        keep,
        rebuilt=rebuilt,
        warp_interp=args.warp_interp,
        warp_radius=args.warp_radius,
        device=device,
    )
    if note:
        print(f"  ⚠️  component decomposition: {note}")
    # The bad time courses travel back to the caller so a MULTI-ECHO run can apply the
    # SAME rejection to every echo without decomposing each one. The projection is
    # linear and each echo's field is a scaled copy of the shared field, so projecting
    # per echo with one shared set of time courses is exactly equivalent to cleaning
    # the shared field and rescaling -- and far cheaper than E decompositions that
    # would each find slightly different components.
    return out, (bad_tc if method == "ica" else None)


def _me_task_stage(result, datas, echo_mean, stem, ext, affine, args, tr, device):
    """Diagnose the shared multi-echo field, then optionally hand back a de-tasked one.

    The multi-echo counterpart of :func:`_task_stage`, and it runs BEFORE the outputs
    are written for the same reason that one does: the diagnostic must see the field as
    estimated, while any fix has to land before the warps, corrected series, flow maps
    and PCs are derived from it.

    Why the fix can be applied per echo at all, which is what the "not wired" refusal
    this replaces assumed it could not be. Echo ``e``'s field is ``alpha_e * w`` (or, on
    a composed run, an affine-in-TE combination of two shared fields), and every de-task
    here is a LINEAR operator applied along time. So ``P(alpha_e * w) = alpha_e * P(w)``
    exactly: projecting each echo's own field with the shared operator is identical to
    cleaning ``w`` and rescaling, with no approximation. What would be wrong is
    DECOMPOSING each echo separately -- E decompositions find E slightly different
    component sets -- so the decomposition happens once, on echo 1 (whose alpha is 1 on
    both axes, making its per-echo fields the shared fields), and only the resulting
    time courses are reused.

    The corrected series is re-derived from each echo's RAW input, never from the series
    the estimator already warped, so nothing stacks a second interpolation.
    """
    from fastfuncstuff.processing.locomoco import (
        detask_result,
        pc_reconstruct_result,
        warp_project_out,
    )

    try:
        design, polort, resp, mask = _write_task_diagnostics(
            result.per_echo[0], echo_mean, stem, ext, affine, args, tr
        )
    except (ValueError, RuntimeError) as exc:
        # A diagnostic must never destroy a finished fit -- the estimation above can be
        # many minutes of GPU time. Any requested fix is refused loudly rather than
        # skipped silently.
        print(f"  ⚠️  task coupling skipped: {exc}")
        for flag, on in (
            ("-detask", args.detask_field),
            ("-detask ica", args.detask_ica),
            ("-warp_recon", bool(args.warp_recon)),
        ):
            if on:
                print(f"  ⚠️  {flag} NOT applied — the outputs below are the ORIGINAL field.")
        return result

    args._task_design = design
    args._stem = stem
    if args.write_pc_maps is not None:
        _write_pc_maps(result.per_echo[0], stem, ext, affine, args, resp, mask, device)

    # ── component rejection / reconstruction, decomposed ONCE on the shared field ──
    if args.warp_recon or args.detask_ica:
        cleaned0, bad_tc = _warp_recon_stage(
            result.per_echo[0], datas[0], args, resp, mask, device, design=design, polort=polort
        )
        result.per_echo[0] = cleaned0
        if bad_tc is not None and any(v.shape[1] for v in bad_tc.values()):
            for j in range(1, len(result.per_echo)):
                res_j = result.per_echo[j]
                comps_j = [(ax, f) for _, ax, f in res_j.pe_displacements()]
                rebuilt = warp_project_out(comps_j, bad_tc, device=device)
                out_j, note = pc_reconstruct_result(
                    res_j,
                    datas[j],
                    {},  # ignored when `rebuilt` is supplied
                    rebuilt=rebuilt,
                    warp_interp=args.warp_interp,
                    warp_radius=args.warp_radius,
                    device=device,
                )
                if note:
                    print(f"  ⚠️  echo {j + 1}: {note}")
                result.per_echo[j] = out_j
            print(
                f"  applied the same rejection to all {len(result.per_echo)} echoes "
                "(one shared decomposition, projected per echo)"
            )
        elif args.warp_recon and len(result.per_echo) > 1:
            # -warp_recon pcs TRUNCATES rather than projecting, so there is no shared
            # time-course set to reuse and the other echoes would silently keep their
            # full-rank fields. Refusing beats shipping a mixed-rank result.
            print(
                "  ⚠️  -warp_recon pcs rebuilt echo 1 only — truncation has no shared "
                "time courses to reuse. Use -detask ica for a multi-echo rejection."
            )

    if not args.detask_field:
        if args.warp_recon or args.detask_ica:
            _write_task_after(
                result.per_echo[0], design, polort, resp, mask, stem, ext, affine, args
            )
        return result

    print_cli_subsection("DE-TASKING — every output below comes from the CLEANED field")
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    for j, res_j in enumerate(result.per_echo):
        cleaned, removed, note = detask_result(
            res_j,
            datas[j],
            design,
            polort,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            device=device,
        )
        if note:
            print(f"  ⚠️  echo {j + 1}: {note}")
        if j == 0:
            # Only echo 1's removed part is written: the others are its alpha-scaled
            # copies by construction, so E files would be one finding shown E times.
            for (label, _, _new), (_, _, task_part) in zip(
                cleaned.pe_displacements(), removed, strict=True
            ):
                path = f"{stem}_flow_{label}_taskpart{ext}"
                with spinner(f"Writing {Path(path).name}"):
                    save_nifti(task_part.float().numpy(), path, affine=affine)
        result.per_echo[j] = cleaned

    _write_task_after(result.per_echo[0], design, polort, resp, mask, stem, ext, affine, args)
    return result


def _write_pc_maps(result, stem, ext, affine, args, resp, mask, device):
    """Write the warp PCs' SPATIAL loadings, one 4-D file per encode axis.

    One file per axis, not one shared file: the temporal basis is common but every
    component carries its own loading on each encode axis, and those are the maps that
    differ. Sub-brick k of ``_pcmap_pe1`` and of ``_pcmap_pe2`` are the same temporal
    component seen on the two axes, so they can be scrubbed side by side.

    Loadings are written RAW and signed, not normalised per component: the amplitude is
    the information -- it says which components carry the motion -- and a viewer scales
    each sub-brick on its own anyway. The variance share rides in the brick label.
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.processing.locomoco import warp_pc_basis
    from fastfuncstuff.stats.task_coupling import map_enrichment

    want = args.write_pc_maps
    got = warp_pc_basis(
        [(ax, f) for _, ax, f in result.pe_displacements()],
        n_pcs=None if want in (None, -1) else want,
        device=device,
    )
    if got is None:
        print("  ⚠️  -write_pc_maps: the warp is empty — no components to write.")
        return
    _u, loadings, _means, var = got
    k = loadings[0][1].shape[-1]
    label_of = {ax: lbl for lbl, ax, _ in result.pe_displacements()}

    scored = resp is not None and mask is not None
    rows = []
    for axis, load in loadings:
        lbl = label_of.get(axis, f"axis{axis}")
        scores = (
            [map_enrichment(load[..., i], resp, mask)["enrichment"] for i in range(k)]
            if scored
            else [float("nan")] * k
        )
        rows.append((lbl, scores))
        names = [
            f"PC{i:02d} {var[i]:.1%}" + (f" {scores[i]:.2f}x" if scored else "") for i in range(k)
        ]
        path = f"{stem}_pcmap_{lbl}{ext}"
        with spinner(f"Writing {Path(path).name}"):
            save_nifti(load.float().numpy(), path, affine=affine, brick_labels=names)

    if scored:
        # A companion table so a map can be matched to its number without reading
        # brick labels one at a time.
        tpath = f"{stem}_pcmap_scores.1D"
        header = "# component  variance  " + "  ".join(lbl for lbl, _ in rows)
        lines = [header]
        for i in range(k):
            cells = "  ".join(f"{sc[i]:8.3f}" for _, sc in rows)
            lines.append(f"{i:9d}  {float(var[i]):8.5f}  {cells}")
        Path(tpath).write_text("\n".join(lines) + "\n")


def _task_stage(result, data, stem, ext, affine, args, tr, device):
    """Diagnose the ORIGINAL field, then optionally hand back a de-tasked one.

    Ordering is the whole point of this function existing. The diagnostic must see the
    field as estimated — measuring after the fix would report the fix's own output and
    always say "no task" — while ``-detask`` has to land BEFORE the warp, corrected
    series, flow maps, PCs and movie are written, so that every output describes one
    field rather than a mix of cleaned and uncleaned.
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.processing.locomoco import detask_result

    try:
        design, polort, resp, mask = _write_task_diagnostics(
            result, data, stem, ext, affine, args, tr
        )
    except (ValueError, RuntimeError) as exc:
        # A diagnostic must never destroy a finished fit. The estimation above can be
        # minutes of GPU time; losing it to a bad -tr or an empty mask is the wrong
        # trade. -detask is refused loudly, because silently skipping the fix the user
        # asked for would be worse than the crash.
        print(f"  ⚠️  task coupling skipped: {exc}")
        if args.detask_field:
            print("  ⚠️  -detask NOT applied — the outputs below are the ORIGINAL field.")
        if args.warp_recon or args.detask_ica:
            print(
                "  ⚠️  the component decomposition was NOT applied — the outputs below "
                "are the ORIGINAL field."
            )
        return result

    args._task_design = design
    args._stem = stem
    if args.write_pc_maps is not None:
        _write_pc_maps(result, stem, ext, affine, args, resp, mask, device)

    # PC reconstruction runs BEFORE -detask: it rebuilds the field, so a field
    # projection afterwards acts on what was actually kept.
    if args.warp_recon or args.detask_ica:
        result, _bad_tc = _warp_recon_stage(
            result, data, args, resp, mask, device, design=design, polort=polort
        )

    if not args.detask_field:
        if args.warp_recon or args.detask_ica:
            _write_task_after(result, design, polort, resp, mask, stem, ext, affine, args)
        return result

    print_cli_subsection("DE-TASKING — every output below comes from the CLEANED field")
    raw_components = [(ax, f) for _, ax, f in result.pe_displacements()]
    cleaned, removed, note = detask_result(
        result,
        data,
        design,
        polort,
        warp_interp=args.warp_interp,
        warp_radius=args.warp_radius,
        device=device,
    )

    for (label, _, new_field), (_, _, task_part) in zip(
        cleaned.pe_displacements(), removed, strict=True
    ):
        path = f"{stem}_flow_{label}_taskpart{ext}"
        with spinner(f"Writing {Path(path).name}"):
            save_nifti(task_part.float().numpy(), path, affine=affine)
        raw = dict(raw_components)[
            next(ax for lbl, ax, _ in result.pe_displacements() if lbl == label)
        ]
        rms_before = float((raw**2).mean().sqrt())
        rms_after = float((new_field**2).mean().sqrt())
        pct = 100 * (1 - rms_after / rms_before) if rms_before > 0 else 0.0
        print(
            f"  • {label}: rms {rms_before:.4f} -> {rms_after:.4f} vox ({pct:.1f}% removed); "
            f"drift KEPT (polort {polort} fitted, not subtracted)"
        )
        print(f"    removed component: {path}")

    if args.want_pcs is not None:
        before = _pc_task_correlation(raw_components, design, polort)
        after = _pc_task_correlation(
            [(ax, f) for _, ax, f in cleaned.pe_displacements()], design, polort
        )
        print(
            f"  • top warp PC vs task: |r| {before:.3f} -> {after:.3f}  "
            "(a task-correlated nuisance regressor removes real BOLD in a GLM)"
        )
    if note:
        print(f"  ⚠️  {note}; the flow/warp/PCs below ARE cleaned.")
    else:
        print("    corrected series re-resampled from the RAW input with the cleaned field.")
    return cleaned


def _qwarp_template_args(args, qwarp_backend: bool) -> tuple[str, int]:
    """``(ref_mode, refine)`` for the qwarp template — the two knobs it used to pin.

    The template was hard-wired to a temporal median built once. ``-ref`` now reaches it,
    but only when the user actually typed one: the resolved default is ``mean`` (``max``
    rotation-aware) and a median template is the more robust choice for registration, so
    defaulting to ``args.ref`` would silently change every existing command. ``-refine``
    drives the qwarp passes under ``-backend qwarp``, where no flow pass runs and it is
    otherwise inert; the ``-final_qwarp`` polish stays at one pass unless asked, since the
    flow refine already sharpened the template it registers against.
    """
    ref_mode = args.ref if getattr(args, "ref_explicit", False) else "median"
    if args.qwarp_refine is not None:
        refine = max(0, int(args.qwarp_refine))
    else:
        refine = max(0, int(args.refine)) if qwarp_backend else 0
    return ref_mode, refine


def _brain_mask_for_coupling(data, shape) -> torch.Tensor | None:
    """Binary (nx,ny,nz) brain mask from the mean volume, or None if it can't be built.

    The coupling statistics need a mask of where the DATA is, not of where the estimated
    flow happens to be large — the two disagree badly, because the feathered gate leaves
    more residual flow outside the head than inside it. Built on the CPU: it is one
    automask over a mean volume, next to nothing beside the solve that just ran.
    """
    if data is None:
        return None
    try:
        from fastfuncstuff.processing.mask import automask

        ref = torch.from_numpy(np.ascontiguousarray(np.asarray(data).mean(axis=3))).float()
        if tuple(ref.shape) != tuple(shape):
            return None
        m = automask(ref.permute(2, 1, 0).contiguous(), device=torch.device("cpu"))
        return m.permute(2, 1, 0).contiguous().bool()
    except Exception:
        # A diagnostic mask is never worth failing the run over; fall back to the caller's.
        return None


def _write_dual_axis_diagnostics(result, stem: str, ext: str, affine, args, data=None) -> None:
    """Write the two-encode-axis extras: the separability map and the coupling report.

    Both exist to answer the question the joint solve deliberately does NOT assume an
    answer to. The two fields are estimated with no ratio tying them together, so:

    * ``_sep`` says WHERE the data could tell the axes apart at all (an aperture-
      ambiguous voxel's split between the axes is arbitrary, and a coupling measured
      over such voxels would be an artifact of the regulariser, not a finding).
    * ``_coupling.txt`` / ``_coupling_r.1D`` say whether, where they COULD be told
      apart, they nonetheless moved together — which is the evidence for or against the
      two artifacts sharing a physical source.
    """
    if result.pe_axis2 is None:
        return
    import numpy as np

    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    if result.sep_map is not None:
        spath = f"{stem}_locomoco_sep{ext}"
        with spinner(f"Writing {Path(spath).name}"):
            save_nifti(result.sep_map.numpy(), spath, affine=affine)

    # Restrict the coupling stats to BRAIN. This used to gate on displacement energy
    # ("voxels that actually moved"), which is backwards on real data: the feathered
    # automask leaves more residual flow in air and at the FoV edge than the brain's
    # genuinely sub-voxel wiggle, so a 10%-of-max energy threshold kept 59% of air and
    # only 39% of brain on a 2 mm ME run — it selected the noise it meant to exclude.
    comps = result.pe_displacements()
    d1, d2 = comps[0][2], comps[1][2]
    mask = _brain_mask_for_coupling(data, d1.shape[:3])
    if mask is None:
        energy = (d1.abs() + d2.abs()).mean(dim=3)
        mask = energy > 0.1 * float(energy.max()) if float(energy.max()) > 0 else None
    if result.sep_map is not None and mask is not None:
        # and to voxels where the split between axes was actually determined
        mask = mask & (result.sep_map.mean(dim=3) > 0.2)
    if mask is not None and not bool(mask.any()):
        mask = None
    c = result.coupling(mask)
    if c is None:
        return

    rpath = f"{stem}_locomoco_coupling_r{ext}"
    # Unmeasured voxels are NaN, not 0: r = 0 is a legitimate result ("measured, and the
    # two fields are unrelated here"), so filling the out-of-mask half with zeros made a
    # perfectly ordinary map look like it had failed over half the volume.
    r_vox = c["r_per_voxel"].clone()
    if mask is not None:
        r_vox[~mask] = float("nan")
    with spinner(f"Writing {Path(rpath).name}"):
        save_nifti(r_vox.numpy(), rpath, affine=affine)
    frame_path = f"{stem}_locomoco_coupling_r.1D"
    np.savetxt(
        frame_path,
        c["r_per_frame"].numpy(),
        fmt="%.6f",
        header="per-frame spatial correlation between the primary-PE and partition fields",
    )
    txt = (
        f"ffs_locomoco two-axis coupling report\n"
        f"  primary PE axis : {comps[0][1]}   rms {c['rms1']:.4f} vox\n"
        f"  partition axis  : {comps[1][1]}   rms {c['rms2']:.4f} vox\n"
        f"  correlation r   : {c['r']:+.4f}\n"
        f"  ratio kappa     : {c['kappa']:+.4f} vox/vox   (R2 {c['kappa_r2']:.4f})\n"
        f"\n"
        f"  The two fields were solved INDEPENDENTLY -- no ratio was imposed. A high |r|\n"
        f"  with a high kappa R2 is evidence the primary-PE and partition wiggles share\n"
        f"  one off-resonance source seen through two effective dwell times. A low |r|\n"
        f"  says they are separate mechanisms. Read it alongside the _sep map: voxels\n"
        f"  where the axes were not separable were excluded from these statistics.\n"
    )
    cpath = f"{stem}_locomoco_coupling.txt"
    Path(cpath).write_text(txt)
    cov = (
        f"{int(mask.sum())} voxels ({100.0 * float(mask.float().mean()):.1f}% of the FoV);"
        " elsewhere NaN"
        if mask is not None
        else "whole FoV"
    )
    print(f"    coverage: {cov}")
    print(
        f"    r = {c['r']:+.4f}, kappa = {c['kappa']:+.4f} vox/vox (R² {c['kappa_r2']:.3f})"
        f"  [{'coupled' if abs(c['r']) > 0.5 else 'largely independent'}]"
    )


def _write_xcorr_diagnostics(result, stem, ext, affine, args) -> None:
    """Write -save_confidence / -save_corr_curve maps — shared by single- and multi-echo.

    Both LocomocoResult and MultiEchoLocomocoResult carry ``confidence`` / ``corr_curve``
    / ``corr_offsets`` (populated only by the xcorr searchlight), so one writer serves
    every path. No-ops with an explanatory note when the running backend produced none.
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    if args.save_confidence:
        conf = getattr(result, "confidence", None)
        if conf is not None:
            p = f"{stem}_locomoco_confidence{ext}"
            with spinner(f"Writing {Path(p).name}"):
                save_nifti(conf.numpy(), p, affine=affine)
        else:
            print(
                "  • -save_confidence: no map — needs -backend xcorr (the flow/phase "
                "backends and the learn-alpha joint solve have no per-voxel search quality)."
            )

    if args.save_corr_curve is not None:
        cc = getattr(result, "corr_curve", None)
        co = getattr(result, "corr_offsets", None)
        if cc is not None and co is not None:
            p = f"{stem}_locomoco_corrcurve{ext}"
            with spinner(f"Writing {Path(p).name}"):
                save_nifti(cc.numpy(), p, affine=affine)
            offs = co.tolist()
            op = f"{stem}_locomoco_corrcurve_offsets.1D"
            with open(op, "w") as f:
                f.write("# ffs_locomoco corr-curve trial offsets (voxels)\n")
                f.write("  " + "  ".join(f"{o:g}" for o in offs) + "\n")
            off_str = ", ".join(f"{o:g}" for o in offs)
            print(f"    offset axis (vox): [{off_str}]  → also {op}")
        else:
            print("  • -save_corr_curve: no landscape — needs -backend xcorr.")


def _write_warp_pcs(components, stem, n_pcs) -> None:
    """Write the top-N temporal warp PCs as {stem}_locomoco_pcs.1D — shared by both paths.

    Recomputed from the in-memory warp, so it is independent of -no_warp (the warp need
    not be written to disk to extract its denoising regressors). ``components`` is the
    ``[(nifti_axis, disp)]`` list — from ``result.warp_components()`` for a single echo,
    or ``[(pe_axis, w_field)]`` for the shared multi-echo field (every echo's warp is a
    scalar multiple of it, so they share these temporal PCs).
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.processing.locomoco import warp_time_pcs

    with spinner("Computing warp PCs"):
        scores, var = warp_time_pcs(components, n_pcs=n_pcs, device=None)
    if scores is None or var is None:
        print("  • warp PCs: skipped (warp is all-zero)")
        return
    pcs_path = f"{stem}_locomoco_pcs.1D"
    var_pct = " ".join(f"{v * 100:.2f}%" for v in var.tolist())
    with open(pcs_path, "w") as f:
        f.write(f"# ffs_locomoco warp temporal PCs — {scores.shape[1]} PCs, unit variance\n")
        f.write(f"# Variance explained: {var_pct}\n")
        for row in scores.numpy():
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")
    print(f"    warp PCs: {scores.shape[1]} components, var {var_pct}")


def _neg_clip(arr: np.ndarray, allow_neg: bool) -> np.ndarray:
    """Clamp negatives in a magnitude image unless -allow_neg (sinc resampling rings)."""
    return arr if allow_neg else np.clip(arr, 0.0, None)


def _want_qc(args) -> bool:
    """True if any QC-output flag (first/last, diff, tSNR, initial) was requested."""
    return bool(
        args.save_first_last or args.save_first_last_diff or args.save_tsnr or args.save_initial
    )


def _want_corrected_qc(args) -> bool:
    """True if a QC output needs the corrected series (not just the raw data)."""
    return bool(args.save_first_last or args.save_first_last_diff or args.save_tsnr)


def _save_tsnr(series, out_stem, ext, affine) -> None:
    """Write a temporal-SNR map (mean / temporal std over T) for QC.

    ``series`` is ``(nx, ny, nz, T)`` with time on the LAST axis. Voxels with
    zero temporal std map to 0 (rather than inf/nan).
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    if series.ndim != 4 or series.shape[3] < 2:
        print(f"  ⚠️  tSNR skipped ({Path(out_stem).name}): need 4-D with ≥2 volumes")
        return

    mean = series.mean(axis=-1)
    std = series.std(axis=-1)
    tsnr = np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)
    path = f"{out_stem}{ext}"
    with spinner(f"Writing {Path(path).name}"):
        save_nifti(tsnr, path, affine=affine)


def _save_first_last(series, out_stem, ext, affine, *, include_diff, allow_neg) -> None:
    """Write a switchable first/last (and optional difference) 4-D file.

    ``series`` is ``(nx, ny, nz, T)`` with time on the LAST axis; ``out_stem`` is
    the full path stem (extension added here). The difference volume (last-first)
    is signed and never neg-clamped so a diverging map reads both directions.
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti

    if series.ndim != 4 or series.shape[3] < 2:
        print(f"  ⚠️  first/last skipped ({Path(out_stem).name}): need 4-D with ≥2 volumes")
        return

    vols = [_neg_clip(series[..., 0], allow_neg), _neg_clip(series[..., -1], allow_neg)]
    if include_diff:
        vols.append(series[..., -1] - series[..., 0])
    stack = np.stack(vols, axis=-1)
    path = f"{out_stem}{ext}"
    with spinner(f"Writing {Path(path).name}"):
        save_nifti(stack, path, affine=affine)


def _write_qc_diag(args, corrected, original, corr_stem, orig_stem, ext, affine) -> None:
    """Write the requested QC files (first/last, difference, tSNR) for one series.

    ``corrected`` / ``original`` are ``(nx, ny, nz, T)`` numpy arrays (corrected
    vs pre-correction). ``corr_stem`` / ``orig_stem`` are their path stems.
    """
    want_plain = args.save_first_last
    want_diff = args.save_first_last_diff

    if want_plain:
        _save_first_last(
            corrected,
            f"{corr_stem}_firstlast",
            ext,
            affine,
            include_diff=False,
            allow_neg=args.allow_neg,
        )
    if want_diff:
        _save_first_last(
            corrected,
            f"{corr_stem}_firstlastdiff",
            ext,
            affine,
            include_diff=True,
            allow_neg=args.allow_neg,
        )
    if args.save_tsnr:
        _save_tsnr(corrected, f"{corr_stem}_tsnr", ext, affine)

    if args.save_initial:
        # -save_initial mirrors whichever QC output(s) are active onto the raw
        # data; alone it means plain first/last on the raw data.
        if want_plain or not (want_diff or args.save_tsnr):
            _save_first_last(
                original,
                f"{orig_stem}_firstlast",
                ext,
                affine,
                include_diff=False,
                allow_neg=args.allow_neg,
            )
        if want_diff:
            _save_first_last(
                original,
                f"{orig_stem}_firstlastdiff",
                ext,
                affine,
                include_diff=True,
                allow_neg=args.allow_neg,
            )
        if args.save_tsnr:
            _save_tsnr(original, f"{orig_stem}_tsnr", ext, affine)


def _pe_label(args) -> str:
    """How to name the solved axis in a banner: the direction the user actually gave.

    ``-pe_dir2`` alone collapses onto ``pe_axis`` upstream, so ``args.pe_dir`` is None
    there and printing it reads as "PE None".
    """
    if args.pe_dir:
        return " ".join(args.pe_dir)
    return f"{args.pe_dir2} (partition)"


def _notch_estimation_data(datas, args, tr):
    """Band-filtered copies of the series for estimation, plus a line describing the cut.

    Raises ValueError with the reason if the design has no line to notch or is too
    broadband -- both are the tool declining to run rather than notching the data.
    """
    import numpy as np
    import torch

    from fastfuncstuff.stats.task_coupling import (
        default_polort,
        design_fit_basis,
        design_notch_bins,
        filter_task_band,
        notch_basis,
    )

    n_t = datas[0].shape[3]
    design, labels = _task_design_from_events(args, n_t, tr, torch.device("cpu"))
    polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
    if args.detask_fit is not None:
        basis = design_fit_basis(design, polort, n_deriv=args.detask_fit)
        out = [
            filter_task_band(torch.from_numpy(np.ascontiguousarray(d)), basis).numpy()
            for d in datas
        ]
        deriv = "" if not args.detask_fit else f" + {args.detask_fit} time derivative(s)"
        note = (
            f"-detask fit: projected the {len(labels)}-condition design{deriv} out of "
            f"the estimation data ({basis.shape[1]} DoF of {n_t - polort - 1}). "
            "Latency/width mismatch leaves task variance behind — read the enrichment "
            "diagnostic below rather than assuming the cut worked."
        )
        return out, note
    bins, info = design_notch_bins(design, polort, widen=args.detask_widen)
    basis = notch_basis(n_t, bins, polort)
    freqs = np.fft.rfftfreq(n_t, d=float(tr))
    out = [
        filter_task_band(torch.from_numpy(np.ascontiguousarray(d)), basis).numpy() for d in datas
    ]
    if info.get("warning"):
        print(f"   ⚠️  -detask filter: {info['warning']}")
    hz = ", ".join(f"{freqs[b]:.4f}" for b in bins)
    note = (
        f"-detask filter: notched {len(bins)} line(s) at {hz} Hz "
        f"({info['spectrum_frac'] * 100:.1f}% of the spectrum, {basis.shape[1]} DoF) "
        f"from the estimation data; {len(labels)} condition(s), polort {polort}"
    )
    return out, note


def _run_multiecho(
    args, pe_axis, slice_axis, dual, device, stem, ext, *, pe_axis2=None, slicewise=False
) -> int:
    """Multi-echo 3-D EPI: joint shared-field estimate, per-echo warp + corrected out."""
    import numpy as np

    if args.paired_ref:
        # The shared field is solved jointly across echoes against one reference per
        # echo; per-bin templates would have to be built per echo too. Refusing beats
        # silently reverting to the plain reference the user did not ask for.
        print(
            "❌ -ref paired is not wired for the multi-echo path (one shared field "
            "scaled per echo, so the templates would have to be per echo as well).",
            file=sys.stderr,
        )
        return 2

    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.processing.locomoco import estimate_residual_flow_multiecho

    if dual:
        print("❌ -me_3depi is single phase-encode only (one -pe_dir).", file=sys.stderr)
        return 2
    if args.backend == "phase":
        print("❌ -me_3depi has no 'phase' backend; use -backend flow or xcorr.", file=sys.stderr)
        return 2
    if args.raw_input is not None or args.moco_matrix is not None:
        print("❌ -me_3depi is not compatible with the rotation-aware path yet.", file=sys.stderr)
        return 2
    if not args.echo_times:
        print("❌ -me_3depi requires -echo_times (one TE in ms per -input).", file=sys.stderr)
        return 2
    if len(args.echo_times) != len(args.input):
        print(
            f"❌ -echo_times has {len(args.echo_times)} values but {len(args.input)} inputs.",
            file=sys.stderr,
        )
        return 2
    if args.me_interecho_refine and not args.me_interecho:
        print(
            "❌ -me_interecho_refine is the second pass of -me_interecho; add -me_interecho "
            "(or use -refine for the plain temporal solve).",
            file=sys.stderr,
        )
        return 2
    if args.me_interecho_refine < 0:
        print("❌ -me_interecho_refine takes a non-negative round count.", file=sys.stderr)
        return 2
    # Argument-only -backend qwarp checks: fire BEFORE the (slow) data load, not after.
    if args.backend == "qwarp":
        if args.me_interecho:
            print(
                "❌ -backend qwarp needs a temporal reference; not valid with -me_interecho.",
                file=sys.stderr,
            )
            return 2
        if args.no_corrected:
            print(
                "❌ -backend qwarp builds its reference from the corrected series; drop -no_corrected.",
                file=sys.stderr,
            )
            return 2
        if args.final_qwarp:
            print(
                "❌ -backend qwarp already owns the field; -final_qwarp is redundant. Pick one.",
                file=sys.stderr,
            )
            return 2

    datas = []
    affine = None
    for path in args.input:
        with spinner(f"Loading {Path(path).name}"):
            img = load_nifti(path)
            d = np.asarray(img.get_fdata(dtype=np.float32))
        if d.ndim != 4:
            print(f"❌ -input {path} must be 4D, got shape {d.shape}", file=sys.stderr)
            return 2
        if affine is None:
            affine = img.affine.copy()
            tr_sec = _resolve_tr(args, img) if args.events else None
        datas.append(d)

    # -detask filter/fit: cut the task out of the images the ESTIMATOR sees. The raw
    # `datas` are kept untouched -- the corrected series is re-resampled from them after
    # the solve, so the only thing the cut changes is what the estimator was allowed to
    # look at.
    est_datas, notch_note = datas, None
    if _detask_pre_estimation(args):
        # One notch basis for every echo. The band is a property of the DESIGN, not of
        # the data, and the estimator pools the echoes into one shared field -- so
        # filtering them with anything but the same basis would let the task back in
        # through whichever echo was treated differently.
        try:
            est_datas, notch_note = _notch_estimation_data(datas, args, tr_sec)
        except ValueError as exc:
            print(f"❌ -detask {_detask_mode_name(args)}: {exc}", file=sys.stderr)
            return 2
        print(f"   🔇 {notch_note}")
        if args.final_qwarp:
            # Same caveat the single-echo path carries: the polish registers raw
            # intensities, so it never sees the notch.
            print(
                "   ⚠️  the -final_qwarp stage still sees UNCUT intensities — only the "
                "flow/xcorr estimate is de-tasked."
            )

    smooth_sigma = 0.0
    hpf_sigma = 0.0
    if args.do_blur > 0 or args.hpf_spatial > 0:
        vox = np.linalg.norm(affine[:3, :3], axis=0)
        in_plane = [a for a in (0, 1, 2) if a != slice_axis]
        inplane_mm = float(np.mean([vox[in_plane[0]], vox[in_plane[1]]]))
        # -do_blur is FWHM (÷2.355 → σ); -hpf_spatial is already a σ (the removed scale).
        smooth_sigma = (args.do_blur / 2.35482) / max(inplane_mm, 1e-6) if args.do_blur > 0 else 0.0
        hpf_sigma = args.hpf_spatial / max(inplane_mm, 1e-6) if args.hpf_spatial > 0 else 0.0

    # Correlation-curve frame: bare -save_corr_curve (const -1) → middle frame.
    corr_curve_frame = None
    if args.save_corr_curve is not None:
        nt = datas[0].shape[3]
        corr_curve_frame = nt // 2 if args.save_corr_curve == -1 else args.save_corr_curve

    automask = not args.no_automask
    # main() already resolved the default; ref_explicit tells us whether the user chose it.
    if args.me_interecho and args.ref_explicit and not args.me_interecho_refine:
        print(
            f"   ℹ️  -ref '{args.ref_label}' is ignored under -me_interecho: there is no temporal "
            "template — each echo registers to its adjacent lower-TE echo at the same TR. "
            "(-me_interecho_refine adds a temporal pass, which does use it.)"
        )
    if args.me_fixed_scaling and args.me_flat_scaling:
        print(
            "❌ choose one of -me_fixed_scaling (TE ratio) or -me_flat_scaling (alpha=1).",
            file=sys.stderr,
        )
        return 2
    te_str = ", ".join(f"{t:g}" for t in args.echo_times)

    # Resolve -me_estimate_from (None → joint solve across all echoes).
    est_idx = None
    if args.me_estimate_from is not None:
        sel = str(args.me_estimate_from).lower()
        if sel == "last":
            est_idx = len(datas) - 1
        elif sel in ("mid", "middle"):
            est_idx = len(datas) // 2
        elif sel == "first":
            est_idx = 0
        else:
            try:
                est_idx = int(sel) - 1  # 1-based index
            except ValueError:
                print(
                    f"❌ -me_estimate_from must be last|mid|first|<index>, got {args.me_estimate_from!r}.",
                    file=sys.stderr,
                )
                return 2
        if not 0 <= est_idx < len(datas):
            print(
                f"❌ -me_estimate_from index resolves to echo {est_idx + 1}, outside 1..{len(datas)}.",
                file=sys.stderr,
            )
            return 2

    # -me_flat_scaling and -me_estimate_from CONTRADICT the inter-echo model (which is
    # linear-in-TE by construction, and estimates from every adjacent pair). -me_fixed_scaling
    # does not: it asks for exactly what inter-echo already imposes, so accept it — under
    # -me_interecho_refine it usefully carries that constraint into the temporal pass too.
    if args.me_interecho and (est_idx is not None or args.me_flat_scaling):
        which = "-me_estimate_from" if est_idx is not None else "-me_flat_scaling"
        print(
            f"❌ {which} contradicts -me_interecho: inter-echo pools every adjacent pair under "
            "a linear-in-TE scaling anchored at echo 1. Drop one.",
            file=sys.stderr,
        )
        return 2
    if args.me_interecho and args.me_fixed_scaling:
        note = (
            "; the refine pass has its own model (-me_refine_scaling, default affine)"
            if args.me_interecho_refine
            else " (no-op here)"
        )
        print(f"   ℹ️  -me_fixed_scaling is what -me_interecho already imposes{note}.")
    if args.me_refine_scaling != "affine" and not args.me_interecho_refine:
        print(
            "❌ -me_refine_scaling only applies to the -me_interecho_refine pass.",
            file=sys.stderr,
        )
        return 2

    # -backend qwarp: a flow -refine pass builds the refined median reference, then the
    # joint TE-scaled qwarp owns the whole field (raw echoes -> reference). The arg-only
    # validity checks already ran before the data load.
    qwarp_backend = args.backend == "qwarp"
    qwarp_ref_mode, qwarp_refine = _qwarp_template_args(args, qwarp_backend)
    # The estimator that runs is the reference-building pass (flow for the qwarp backend).
    prepass_backend = "flow" if qwarp_backend else args.backend

    print_cli_header("ffs_locomoco", "Residual nonlinear motion correction")
    print_cli_section("Configuration", leading_blank=False)
    print(
        f"   multi-echo ({'2-D slicewise' if slicewise else '3-D'}): "
        f"{len(datas)} echoes, shape={datas[0].shape}, device={device}"
    )
    if qwarp_backend:
        print("   final data resampling kernel: wsinc5 (qwarp)")
    else:
        print(
            f"   data resampling kernel: {_warp_kernel_label(args.warp_interp, args.warp_radius)}"
        )
    if args.me_interecho:
        mode, scaling = "inter-echo (align stack per TR)", "linear-in-TE (anchor=echo1)"
        if args.me_interecho_refine:
            mode += f" + temporal refine ×{args.me_interecho_refine}"
            scaling += f" → refine {args.me_refine_scaling}"
    else:
        mode = f"scaled from echo {est_idx + 1}" if est_idx is not None else "joint solve"
        if args.me_flat_scaling:
            scaling = "flat(alpha=1)"
        elif args.me_fixed_scaling or est_idx is not None:
            scaling = "fixed(TE)"
        else:
            scaling = "learned"
    if args.match != "none":
        if qwarp_backend:
            # No flow estimate runs, and qwarp registers raw data under a per-patch
            # correlation that is already invariant to a smooth gain.
            print(
                f"   ⚠️  -match {args.match} needs a flow/phase estimate; -backend qwarp "
                "registers the RAW frames, so it is IGNORED."
            )
        elif args.me_interecho and not args.me_interecho_refine:
            # The inter-echo pass registers echo to echo, never frame to frame, so there
            # is no cross-TIME comparison for -match to fix; -me_match is its knob.
            print(
                f"   ⚗️  cross-TE intensity match: {args.me_match} "
                f"(σ={args.me_match_sigma:g}vox) — -match {args.match} is inert without "
                "-me_interecho_refine (the inter-echo pass has no temporal reference)."
            )
        else:
            print(
                f"   ⚗️  estimation intensity match: {args.match} "
                f"(σ={args.match_sigma:g}vox, cross-TIME, per echo)"
            )
    if pe_axis2 is not None:
        # Two axes carry two DIFFERENT laws, and only the partition one is settable:
        # the primary PE row is pinned alpha=1 by construction. A bare "scaling=fixed(TE)"
        # reads as if it applied to both, so name the axis each law belongs to.
        scaling = f"primary PE flat(alpha=1) / partition {scaling}"
    # ref / refine are temporal-template knobs — inert under inter-echo unless its
    # temporal refine pass runs, which is a genuine template-based solve.
    ie_only = args.me_interecho and not args.me_interecho_refine
    ref_note = "ref=n/a" if ie_only else f"ref={args.ref_label}"
    if ie_only:
        refine_note = "refine=n/a"
    elif args.me_interecho:
        refine_note = f"refine={args.me_interecho_refine} (temporal pass)"
    else:
        refine_note = f"refine={args.refine}"
    if qwarp_backend:
        # qwarp owns the whole field and the reference is a plain median of the raw
        # echoes (no flow estimation), so the flow-tuning / ref / refine flags don't apply.
        print(
            f"   TEs [{te_str}] ms, PE {_pe_label(args)} (axis {pe_axis}), backend=qwarp "
            f"(no flow pass), template={qwarp_ref_mode} of raw echoes, "
            f"refine={qwarp_refine}, scaling={scaling}, "
            f"automask={'on' if automask else 'off'}"
        )
        print(
            f"   🪄 qwarp field: minpatch={_axis_txt(args.qwarp_minpatch)}, "
            f"levels={args.qwarp_levels}, "
            f"iters={args.qwarp_iters}, cost={args.qwarp_cost}, optimizer={args.qwarp_optimizer}"
        )
    else:
        print(
            f"   TEs [{te_str}] ms, PE {_pe_label(args)} (axis {pe_axis}), backend={args.backend}, "
            f"{ref_note}, mode={mode}, scaling={scaling}, "
            f"levels={_axis_txt(args.levels)}, iters={args.iters}, {refine_note}, "
            f"automask={'on' if automask else 'off'}"
        )
    if hpf_sigma > 0:
        print(f"   ⚗️  estimation spatial high-pass: {args.hpf_spatial}mm (σ={hpf_sigma:.2f}vox)")
    # The inter-echo mode has no temporal reference, so the refine bias warning is moot.
    # The qwarp backend has its own template-refine loop and its own note below.
    if qwarp_backend and qwarp_refine == 0:
        print(
            f"   ℹ️  qwarp refine=0: the template ('{qwarp_ref_mode}' of the RAW echoes) still "
            "carries the distortion, so it is blurred by it and biases the field LOW — add "
            "-refine/-qwarp_refine 2 to rebuild it from the corrected series."
        )
    if args.refine == 0 and not args.me_interecho and not qwarp_backend:
        print(
            f"   ℹ️  refine=0: the reference ('{args.ref_label}') is built from the un-corrected "
            "frames, so it still carries the wiggle and biases displacement LOW — add "
            "-refine 2/-workhard/-superhard for full magnitude."
        )

    if qwarp_backend:
        # qwarp owns the whole field, so the flow ESTIMATION is skipped entirely: the
        # reference is a plain temporal median of the raw echoes (correct when the inputs
        # are already moco'd/NORDIC'd). Use -backend flow -final_qwarp for flow refinement.
        from fastfuncstuff.processing.locomoco import make_raw_reference_me_result

        result = make_raw_reference_me_result(
            est_datas,
            args.echo_times,
            pe_axis,
            slice_axis,
            flat_scaling=args.me_flat_scaling,
        )
    elif args.me_interecho:
        from fastfuncstuff.processing.locomoco import estimate_residual_flow_me_interecho

        result = estimate_residual_flow_me_interecho(
            est_datas,
            args.echo_times,
            pe_axis,
            slice_axis,
            backend=prepass_backend,
            smooth_sigma=smooth_sigma,
            n_levels=args.levels,
            n_iters=args.iters,
            window_sigma=args.window,
            max_shift=args.max_shift,
            trial_step=args.xcorr_step,
            automask=automask,
            automask_sigma=args.automask_sigma,
            noshift_margin=args.noshift_margin,
            reg_sigma=args.reg_sigma,
            peak_mode="argmax" if args.argmax else "first_peak",
            search_min_steps=args.search_min_steps,
            save_corr_curve=corr_curve_frame,
            hpf_sigma=hpf_sigma,
            match=args.me_match,
            match_sigma=args.me_match_sigma,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            device=device,
        )
        if args.me_interecho_refine:
            from fastfuncstuff.processing.locomoco import refine_interecho_temporally

            result = refine_interecho_temporally(
                result,
                est_datas,
                args.echo_times,
                pe_axis,
                slice_axis,
                refine_rounds=args.me_interecho_refine,
                ref_mode=args.ref,
                backend=prepass_backend,
                smooth_sigma=smooth_sigma,
                n_levels=args.levels,
                n_iters=args.iters,
                window_sigma=args.window,
                max_shift=args.max_shift,
                trial_step=args.xcorr_step,
                converge=args.converge,
                converge_rel=args.converge_rel,
                first_n=args.first_n,
                automask=automask,
                automask_dilate=args.automask_dilate,
                automask_sigma=args.automask_sigma,
                coverage_erode=None if args.nocoverage else args.coverage_erode,
                scaling=args.me_refine_scaling,
                match=args.match,
                match_sigma=args.match_sigma,
                ngf_eta_q=args.match_eta_q,
                noshift_margin=args.noshift_margin,
                reg_sigma=args.reg_sigma,
                peak_mode="argmax" if args.argmax else "first_peak",
                search_min_steps=args.search_min_steps,
                warp_interp=args.warp_interp,
                warp_radius=args.warp_radius,
                hpf_sigma=hpf_sigma,
                want_corrected=not args.no_corrected,
                device=device,
            )
    elif est_idx is not None:
        from fastfuncstuff.processing.locomoco import estimate_residual_flow_me_scaled

        result = estimate_residual_flow_me_scaled(
            est_datas,
            args.echo_times,
            est_idx,
            pe_axis,
            slice_axis,
            ref_mode=args.ref,
            backend=prepass_backend,
            smooth_sigma=smooth_sigma,
            n_levels=args.levels,
            n_iters=args.iters,
            window_sigma=args.window,
            max_shift=args.max_shift,
            trial_step=args.xcorr_step,
            refine_rounds=args.refine,
            converge=args.converge,
            converge_rel=args.converge_rel,
            first_n=args.first_n,
            automask=automask,
            automask_dilate=args.automask_dilate,
            automask_sigma=args.automask_sigma,
            coverage_erode=None if args.nocoverage else args.coverage_erode,
            flat_scaling=args.me_flat_scaling,
            noshift_margin=args.noshift_margin,
            reg_sigma=args.reg_sigma,
            peak_mode="argmax" if args.argmax else "first_peak",
            search_min_steps=args.search_min_steps,
            match=args.match,
            match_sigma=args.match_sigma,
            ngf_eta_q=args.match_eta_q,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            hpf_sigma=hpf_sigma,
            device=device,
        )
    else:
        result = estimate_residual_flow_multiecho(
            est_datas,
            args.echo_times,
            pe_axis,
            slice_axis,
            ref_mode=args.ref,
            backend=prepass_backend,
            smooth_sigma=smooth_sigma,
            n_levels=args.levels,
            n_iters=args.iters,
            window_sigma=args.window,
            max_shift=args.max_shift,
            trial_step=args.xcorr_step,
            refine_rounds=args.refine,
            converge=args.converge,
            converge_rel=args.converge_rel,
            first_n=args.first_n,
            automask=automask,
            automask_dilate=args.automask_dilate,
            automask_sigma=args.automask_sigma,
            coverage_erode=None if args.nocoverage else args.coverage_erode,
            learn_scaling=not args.me_fixed_scaling,
            flat_scaling=args.me_flat_scaling,
            noshift_margin=args.noshift_margin,
            reg_sigma=args.reg_sigma,
            peak_mode="argmax" if args.argmax else "first_peak",
            search_min_steps=args.search_min_steps,
            save_corr_curve=corr_curve_frame,
            want_corrected=not args.no_corrected,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            hpf_sigma=hpf_sigma,
            match=args.match,
            match_sigma=args.match_sigma,
            ngf_eta_q=args.match_eta_q,
            pe_axis2=pe_axis2,
            slicewise=slicewise,
            device=device,
        )

    if qwarp_backend or args.final_qwarp:
        if args.final_qwarp and args.me_interecho:
            raise SystemExit(
                "-final_qwarp needs a temporal reference; not supported with -me_interecho."
            )
        if args.final_qwarp and args.no_corrected:
            raise SystemExit("-final_qwarp polishes the corrected series; drop -no_corrected.")
        from fastfuncstuff.processing.locomoco import polish_me_result

        if qwarp_backend:
            print(
                f"🪄 qwarp backend: registering raw echoes to the {qwarp_ref_mode}-of-raw "
                f"reference ({qwarp_refine + 1} pass{'es' if qwarp_refine else ''})..."
            )
        else:
            print("🪄 Polishing residual with joint TE-scaled qwarp...")
        result = polish_me_result(
            result,
            minpatch=args.qwarp_minpatch,
            n_levels=args.qwarp_levels,
            iters=args.qwarp_iters,
            cost=args.qwarp_cost,
            optimizer=args.qwarp_optimizer,
            compile=args.qwarp_compile,
            full=qwarp_backend,
            ref_mode=qwarp_ref_mode,
            refine=qwarp_refine,
            slicewise=False,  # -me_3depi is 3-D-acquired: through-plane continuity is real
            raw_datas=datas,
            device=device,
        )

    if _detask_pre_estimation(args):
        # The estimator was shown task-cut images (notched or design-projected), so the
        # series it warped carries that cut too. Re-derive it from the untouched input
        # with the same field.
        from fastfuncstuff.processing.locomoco import resample_from_raw

        # EVERY echo, not just the first. Each per-echo series was warped from the
        # task-cut copy the estimator saw, so re-deriving only echo 1 would ship one
        # raw-derived echo alongside E-1 cut ones.
        for j, res_j in enumerate(result.per_echo):
            result.per_echo[j] = resample_from_raw(
                res_j,
                datas[j],
                warp_interp=args.warp_interp,
                warp_radius=args.warp_radius,
                device=device,
                desc=f"detask resample e{j + 1}",
            )

    # Two-axis extras. Echo 1 carries alpha = 1 on BOTH axes (primary PE is flat by
    # construction, and the partition law is normalised to echo 1), so its per-echo
    # fields ARE the shared fields — the coupling measured on it is the shared-field
    # coupling, not one echo's view of it.
    # The echo MEAN, for the same reason the task diagnostic below uses it: the brain
    # mask this derives is a tissue mask, and the shortest echo gives the weakest
    # contrast to build it from.
    echo_mean = datas[0] if len(datas) == 1 else np.mean(np.stack(datas), axis=0)
    _write_dual_axis_diagnostics(result.per_echo[0], stem, ext, affine, args, echo_mean)

    if args.events and tr_sec:
        result = _me_task_stage(result, datas, echo_mean, stem, ext, affine, args, tr_sec, device)

    print_cli_section("Outputs")
    as_5d = args.warp_format == "5d"
    # The task-fit "after" is measured on the echo MEAN, because the "before" was
    # (echo_mean above): comparing a single echo against a mean would mostly report the
    # contrast difference between them. Accumulated in the loop that already
    # materializes each echo's corrected series rather than a second pass over all of
    # them.
    corr_sum = None
    for j, res in enumerate(result.per_echo):
        estem = f"{stem}_e{j + 1}"
        if not args.no_warp:
            from fastfuncstuff.processing.medic import save_medic_warp

            # EVERY component, exactly as the single-echo path does. A dual-encode or
            # rotation-aware result carries 2-3 axes, and writing only the first ships a
            # warp that silently omits one of them -- the corrected series would be
            # right while the saved warp, the thing applied to other data, is not.
            (primary_axis, primary_disp), *rest = res.warp_components()
            with spinner(f"Writing {Path(estem).name}_warp{ext}"):
                save_medic_warp(
                    primary_disp,
                    primary_axis,
                    affine,
                    estem,
                    nii_ext=ext,
                    as_5d=as_5d,
                    extra_components=[(d, a) for a, d in rest],
                )
        # Materialized once per echo for both the write and the QC maps below --
        # see the single-echo block for why the repeat call is not free.
        want_corrected = (
            not args.no_corrected
            or (_want_qc(args) and _want_corrected_qc(args))
            # Echo 1's corrected series feeds the task-fit "after" below, measured on
            # the same echo-mean the "before" was.
            or getattr(args, "_task_after", None) is not None
        )
        corrected_series = res.corrected_series() if want_corrected else None
        if corrected_series is not None and getattr(args, "_task_after", None) is not None:
            corr_sum = corrected_series.clone() if corr_sum is None else corr_sum + corrected_series
        if not args.no_corrected:
            assert corrected_series is not None
            corr_path = f"{estem}_locomoco{ext}"
            with spinner(f"Writing {Path(corr_path).name}"):
                save_nifti(
                    _neg_clip(corrected_series.numpy(), args.allow_neg),
                    corr_path,
                    affine=affine,
                )
        if not args.no_flow:
            if res.pe_axis2 is not None:
                for label, _axis, field in res.pe_displacements():
                    fpath = f"{estem}_flow_{label}{ext}"
                    with spinner(f"Writing {Path(fpath).name}"):
                        save_nifti(field.numpy(), fpath, affine=affine)
            else:
                flow_path = f"{estem}_flow{ext}"
                with spinner(f"Writing {Path(flow_path).name}"):
                    save_nifti(res.pe_displacement().numpy(), flow_path, affine=affine)
        if _want_qc(args):
            corrected = (
                corrected_series.numpy()
                if corrected_series is not None and _want_corrected_qc(args)
                else None
            )
            _write_qc_diag(
                args, corrected, datas[j], f"{estem}_locomoco", f"{estem}_orig", ext, affine
            )
        del corrected_series

    if corr_sum is not None:
        _write_data_task_after(corr_sum / len(result.per_echo), stem, args)
        del corr_sum
    elif getattr(args, "_task_after", None) is not None:
        _write_data_task_after(None, stem, args)

    # Shared scaling diagnostic: learned alpha vs echo time, and the linearity r².
    alpha_path = f"{stem}_locomoco_alpha.1D"
    with open(alpha_path, "w") as f:
        f.write("# ffs_locomoco multi-echo per-echo scaling (alpha_e · shared field)\n")
        f.write(f"# linear-in-TE r² = {result.linearity_r2:.6f}\n")
        f.write(f"# echo_TE_ms   {result.alpha_label}\n")
        for te_v, a_v in zip(result.echo_times.tolist(), result.alpha.tolist(), strict=True):
            f.write(f"  {te_v:10.4f}  {a_v:12.6f}\n")

    if args.want_pcs is not None:
        # PCs of the SHARED field w: every echo's warp is alpha_e·w, so they all share
        # these temporal regressors. In memory regardless of -no_warp.
        _write_warp_pcs([(result.pe_axis, result.w_field)], stem, args.want_pcs)

    _write_xcorr_diagnostics(result, stem, ext, affine, args)

    print_cli_footer("ffs_locomoco")
    return 0


def _parse(argv: list[str], namespace: argparse.Namespace | None = None) -> argparse.Namespace:
    """Parse one ffs_locomoco command line. Shared by the solo path and by every
    -batch manifest line, so a batched run is byte-identical to a solo one."""
    from fastfuncstuff.processing.locomoco import normalize_axis_argv

    return create_parser().parse_args(
        normalize_axis_argv(
            argv,
            {
                "-pe_dir",
                "-pe_dir1",
                "-pe",
                "-pe_dir2",
                "-partition_dir",
                "-partition",
                "-slice_axis",
                "-slice",
            },
        ),
        namespace or argparse.Namespace(),
    )


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output paths a solo run of ``args`` would write, for -batch_skip.

    The warp is the estimate itself; the lane reductions are checked alongside it
    because a working directory from before those flags existed has the warp but
    not the images the next stage reads."""
    if not args.prefix:
        return []
    stem, ext = _split_prefix(args.prefix)
    outs = [f"{stem}_warp{ext}"]
    for which in ("mean", "max", "min"):
        if getattr(args, f"save_{which}", False):
            outs.append(f"{stem}_locomoco_{which}{ext}")
    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: needs -input/-prefix."""
    missing = [f for f in ("input", "prefix") if not getattr(run_args, f, None)]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    args = _parse(raw)

    if args.batch is not None or args.batch_run:
        # A locomoco run is a real chunk of GPU work, but the flow backend's
        # torch.compile warmup is paid per PROCESS — over a session's worth of
        # runs that dominates. One process warms up once and reuses the kernels.
        run_batch_jobs(
            tool="ffs_locomoco",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=setup_device(args.device, tf32=REGISTRATION_TF32),
            parse_line=lambda line, base: _parse(shlex.split(line), base),
            defaults=args,
            # _dispatch_run reports bad requests with a nonzero return, not an
            # exception; the batch runner only counts raises, so translate.
            dispatch=_dispatch_raising,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
        )
        return 0

    try:
        _validate_batch_run(args)
    except ValueError as exc:
        print(
            f"❌ {exc} (or use -batch FILE / -batch_run ARGS).",
            file=sys.stderr,
        )
        return 2

    return _dispatch_run(args, None)


def _dispatch_raising(args: argparse.Namespace, device: torch.device) -> None:
    rc = _dispatch_run(args, device)
    if rc:
        raise ValueError(f"run exited {rc}")


def _dispatch_run(args: argparse.Namespace, device: torch.device | None) -> int:
    """Estimate one dataset's residual motion (the whole per-input body).

    ``device`` is None on the solo path (resolve it here) and pre-resolved by the
    batch runner, so the batch chooses a device once for all of its runs."""
    preset = _apply_preset(args)

    # -detask has nothing to key on without a design, and the task stage it lives in is
    # gated on -events — so a lone -detask is a silent no-op all the way through. Say so
    # here rather than let a run finish looking de-tasked.
    try:
        args.detask_field, args.detask_widen, args.detask_fit = parse_detask(args.detask)
        args.detask_ica = bool(args.detask) and str(args.detask).strip().lower().startswith("ica")
        parse_warp_recon(args.warp_recon)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    if args.reject is not None and not (args.warp_recon or args.detask_ica):
        print(
            "❌ -reject sets the SPATIAL threshold for a component decomposition; pass "
            "-detask ica (rejection implied) or -warp_recon pcs too.",
            file=sys.stderr,
        )
        return 2
    if args.reject is not None and not args.events:
        print(
            "❌ -reject needs -events: it scores each component against WHERE the task "
            "response is, which the design defines.",
            file=sys.stderr,
        )
        return 2
    if args.detask and not args.events:
        print(
            "❌ -detask needs -events (the task-locked part is defined by the design); "
            "nothing would be removed.",
            file=sys.stderr,
        )
        return 2

    # Multi-echo 3-D EPI is a single 3-D-acquired solve, so (like -is_3dacq) the PE
    # direction is allowed to coincide with the slice axis.
    me_mode = args.me_3depi or len(args.input) > 1

    # Rotation-aware mode (both raw + matrices) defaults its reference to the temporal
    # MAX (fills FoV dropout, high-signal anchor); the plain path stays on the mean.
    rotaware = args.raw_input is not None or args.moco_matrix is not None
    # Stash explicitness BEFORE defaulting: the multi-echo path runs later and can only
    # tell "the user asked for this reference" from "nobody set one" via this flag.
    args.ref_explicit = args.ref is not None
    if args.ref is None:
        args.ref = "max" if rotaware else "mean"
    # "paired[_stat]" selects the BINS; the statistic within a bin is an independent
    # axis, so it is split off here and the library keeps taking a plain ref_mode.
    # Same compound-name shape as the existing first_mean / first_median.
    args.paired_ref, args.ref, args.ref_label = parse_ref_mode(args.ref)

    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.processing.locomoco import estimate_residual_flow, resolve_pe_axis

    # ── which encode axes are we solving? ────────────────────────────────────
    # -pe_dir1 is the PRIMARY (in-plane) phase encode; -pe_dir2 the PARTITION. Two values
    # on -pe_dir is shorthand for both. The explicit -pe_dir2 flag implies 3-D acquisition
    # (a partition direction is a 3-D concept); the two-value shorthand does NOT, so an
    # existing 2-D multi-slice `-pe_dir AP IS` command line keeps its old meaning.
    if not args.pe_dir and args.pe_dir2 is None:
        print(
            "❌ give at least one encode direction: -pe_dir1 (primary phase encode) "
            "and/or -pe_dir2 (partition).",
            file=sys.stderr,
        )
        return 2
    null_axis = resolve_pe_axis(args.pe_null) if args.pe_null else None
    pe1_list = [resolve_pe_axis(d) for d in (args.pe_dir or [])]
    if len(pe1_list) > 2:
        print(f"❌ -pe_dir takes 1 or 2 directions, got {len(pe1_list)}.", file=sys.stderr)
        return 2
    if len(pe1_list) == 2 and args.pe_dir2 is not None:
        print(
            "❌ give EITHER two directions to -pe_dir (shorthand) OR -pe_dir2, not both.",
            file=sys.stderr,
        )
        return 2
    partition_only = False
    if len(pe1_list) == 2:
        pe_axis, pe_axis2 = pe1_list
    elif not pe1_list:
        # -pe_dir2 alone: a single-axis solve along the PARTITION direction (e.g. the
        # primary axis is already corrected, by MEDIC or a fieldmap). Downstream this is
        # an ordinary one-axis run; only the labelling and the multi-echo scaling law
        # differ, so collapse it onto pe_axis and remember which axis it really is.
        pe_axis = resolve_pe_axis(args.pe_dir2)
        pe_axis2 = None
        partition_only = True
        args.is_3dacq = True
    else:
        pe_axis = pe1_list[0]
        pe_axis2 = resolve_pe_axis(args.pe_dir2) if args.pe_dir2 is not None else None
        if pe_axis2 is not None:
            args.is_3dacq = True
    two_axes = pe_axis2 is not None
    if two_axes and pe_axis == pe_axis2:
        print(
            f"❌ the primary PE and partition directions must differ, both resolved to "
            f"axis {pe_axis}.",
            file=sys.stderr,
        )
        return 2
    # `dual` = the LEGACY 2-D multi-slice two-in-plane-axis mode (one in-plane vector).
    # `dual3d` = the 3-D joint solve: two physically distinct artifacts, solved
    # simultaneously and independently.
    dual = two_axes and not args.is_3dacq
    dual3d = two_axes and args.is_3dacq
    pe_axes = [pe_axis, pe_axis2] if two_axes else [pe_axis]
    if (rc := _resolve_axis_flags(args, len(pe_axes), partition_only)) is not None:
        return rc
    args.warp_interp = _resolve_warp_interp(args.warp_interp, rotaware=rotaware)

    if me_mode:
        if two_axes and (args.me_interecho or args.me_estimate_from is not None):
            which = "-me_interecho" if args.me_interecho else "-me_estimate_from"
            print(
                f"❌ {which} is single encode-axis only; drop -pe_dir2 (or the second "
                f"-pe_dir) to use it.",
                file=sys.stderr,
            )
            return 2
        me_is_3d = args.me_3depi or args.is_3dacq
        if me_is_3d and args.pe_dir2 is None and not two_axes and not args.me_flat_scaling:
            # -pe_dir used to mean the PARTITION axis under -me_3depi. It now means the
            # primary PE axis everywhere. Remapping silently would point an existing
            # command line at a different physical axis and still produce plausible
            # output, so make the rename explicit instead. -me_flat_scaling is exempt:
            # it explicitly asks for a TE-INDEPENDENT solve, which is what the primary
            # PE axis is, so `-pe_dir X -me_flat_scaling` is an unambiguous request.
            print(
                f"❌ -me_3depi: -pe_dir now means the PRIMARY phase-encode axis "
                f"(it used to mean the partition axis). For the TE-scaled partition "
                f"wiggle this tool has always corrected, say:\n"
                f"      -pe_dir2 {args.pe_dir[0]}      (or -partition_dir {args.pe_dir[0]})\n"
                f"   For a TE-INDEPENDENT primary-PE solve across echoes, keep -pe_dir "
                f"and add -me_flat_scaling.",
                file=sys.stderr,
            )
            return 2

    # Rotation-aware synthesis is a genuine arbitrary 3-D warp and still has no
    # Lanczos implementation. Dual PE is only a 2-D warp and now has a fused CUDA path.
    if args.warp_interp == "lanczos":
        if rotaware:
            print(
                "❌ -warp_interp lanczos does not yet support rotation-aware 3-D warps; use "
                "bilinear/bicubic.",
                file=sys.stderr,
            )
            return 2
    slice_axis = args.slice_axis
    if args.is_3dacq and args.backend == "phase":
        print(
            "❌ -is_3dacq has no 'phase' backend yet; use -backend flow or xcorr.", file=sys.stderr
        )
        return 2
    if two_axes:
        # Both encode axes must lie in the display plane, so we cut along the third.
        # For dual3d this is only a display/movie choice (the solve is one 3-D pass and
        # re-derives the same axis itself); for the legacy 2-D dual it is structural.
        third = next(a for a in (0, 1, 2) if a not in pe_axes)
        if slice_axis != third:
            which = "-pe_dir1/-pe_dir2" if dual3d else "dual -pe_dir"
            print(
                f"ℹ️  {which}: forcing -slice_axis to {third} (the un-encoded axis) so "
                f"both encode axes lie in the slice plane.",
            )
        slice_axis = third
    # Under -is_3dacq / -me_3depi the slice axis is only a display hint, PE==slice ok.
    elif pe_axis == slice_axis and not args.is_3dacq and not me_mode:
        print(
            f"❌ PE axis ({pe_axis}) and -slice_axis ({slice_axis}) coincide. The PE "
            "direction must lie in the slice plane — pick a different -slice_axis.",
            file=sys.stderr,
        )
        return 2

    if device is None:
        device = setup_device(args.device, tf32=REGISTRATION_TF32)

    stem, ext = _split_prefix(args.prefix)

    from fastfuncstuff.cli_utils import spinner

    # Multi-echo: several inputs, one shared field per encode axis, scaled per echo.
    # 2-D multi-slice is the default; -me_3depi / -is_3depi opts into the 3-D solve,
    # exactly as -is_3depi does on the single-echo path.
    if me_mode:
        return _run_multiecho(
            args,
            pe_axis,
            slice_axis,
            dual,
            device,
            stem,
            ext,
            pe_axis2=pe_axis2 if dual3d else None,
            slicewise=not (args.me_3depi or args.is_3dacq),
        )

    # Single-echo qwarp: same idea as ME with E=1 -- no echo scaling, just register each
    # frame to the refined median via ncc. -backend qwarp lets qwarp own the field;
    # -final_qwarp polishes the estimator's residual.
    qwarp_backend = args.backend == "qwarp"
    if qwarp_backend or args.final_qwarp:
        which = "backend" if qwarp_backend else "polish"
        if rotaware:
            print(f"❌ qwarp {which} is not supported with rotation-aware mode.", file=sys.stderr)
            return 2
        if args.no_corrected:
            print(
                f"❌ qwarp {which} builds its reference from the corrected series; drop -no_corrected.",
                file=sys.stderr,
            )
            return 2
    qwarp_ref_mode, qwarp_refine = _qwarp_template_args(args, qwarp_backend)
    # The estimator that runs is the reference-building pass (flow for the qwarp backend).
    prepass_backend = "flow" if qwarp_backend else args.backend
    # -backend qwarp owns the whole field, so the flow estimation is skipped (qwarp+dual
    # / qwarp+rotaware already errored above, so reaching here means neither is set).
    skip_flow_qwarp = qwarp_backend and not rotaware
    if len(args.input) != 1:
        print(
            f"❌ single-echo mode takes ONE -input (got {len(args.input)}); use -me_3depi for "
            "multi-echo.",
            file=sys.stderr,
        )
        return 2
    args.input = args.input[0]

    with spinner(f"Loading {Path(args.input).name}"):
        img = load_nifti(args.input)
        data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 4:
        print(f"❌ -input must be 4D, got shape {data.shape}", file=sys.stderr)
        return 2
    affine = img.affine.copy()
    tr_sec = _resolve_tr(args, img) if args.events else None

    # -detask filter/fit: the estimator sees a task-cut copy, `data` stays raw so the
    # corrected series can be re-derived from it after the solve.
    est_data = data
    if _detask_pre_estimation(args):
        try:
            (est_data,), note = _notch_estimation_data([data], args, tr_sec)
        except ValueError as exc:
            print(f"❌ -detask {_detask_mode_name(args)}: {exc}", file=sys.stderr)
            return 2
        print(f"   🔇 {note}")
        if args.final_qwarp or args.backend == "qwarp":
            # The polish registers raw intensities against a raw template; -match never
            # reached it either. Say so rather than let a partially-filtered run read as
            # a clean one.
            print(
                "   ⚠️  the qwarp stage still sees UNCUT intensities — only the "
                "flow/xcorr estimate is de-tasked."
            )

    paired_bins = None
    if args.paired_ref:
        if not (args.is_3dacq or dual3d):
            print(
                "❌ -ref paired needs the 3-D solve: pass -is_3dacq, or two encode axes "
                "(-pe_dir1/-pe_dir2). The 2-D slicewise path builds its reference per "
                "slice and has not been converted.",
                file=sys.stderr,
            )
            return 2
        if qwarp_backend:
            print(
                "❌ -ref paired does not apply to -backend qwarp, which owns the field "
                "and registers to a median of the raw series. Use -backend flow/xcorr "
                "(optionally with -final_qwarp).",
                file=sys.stderr,
            )
            return 2
        if not (args.events and tr_sec):
            print(
                "❌ -ref paired needs -events and a usable TR to know each frame's task state.",
                file=sys.stderr,
            )
            return 2
        try:
            paired_bins = _paired_bins_for(args, data.shape[3], tr_sec, torch.device("cpu"))
        except ValueError as exc:
            print(f"❌ -ref paired: {exc}", file=sys.stderr)
            return 2

    # Correlation-curve frame (xcorr diagnostics): bare -save_corr_curve (const -1) → middle.
    corr_curve_frame = None
    if args.save_corr_curve is not None:
        corr_curve_frame = (
            data.shape[3] // 2 if args.save_corr_curve == -1 else args.save_corr_curve
        )

    # -do_blur is FWHM in mm (repo convention); convert to an in-plane voxel sigma
    # for the pre-flow Gaussian. In-plane voxel size = mean of the two non-slice axes.
    smooth_sigma = 0.0
    hpf_sigma = 0.0
    if args.do_blur > 0 or args.hpf_spatial > 0:
        vox = np.linalg.norm(affine[:3, :3], axis=0)  # per-axis mm
        in_plane = [a for a in (0, 1, 2) if a != slice_axis]
        inplane_mm = float(np.mean([vox[in_plane[0]], vox[in_plane[1]]]))
        # -do_blur is FWHM (÷2.355 → σ); -hpf_spatial is already a σ (the smoothing
        # scale removed by the unsharp subtraction), so no FWHM conversion.
        smooth_sigma = (args.do_blur / 2.35482) / max(inplane_mm, 1e-6) if args.do_blur > 0 else 0.0
        hpf_sigma = args.hpf_spatial / max(inplane_mm, 1e-6) if args.hpf_spatial > 0 else 0.0

    if args.match != "none" and rotaware:
        raise SystemExit(
            "❌ -match is not supported with rotation-aware mode yet.",
        )
    if args.hpf_spatial > 0 and rotaware:
        print(
            "❌ -hpf_spatial is not supported with rotation-aware mode yet.",
            file=sys.stderr,
        )
        return 2

    pe_only = not args.full_2d
    automask = not args.no_automask
    cover_desc = "no coverage" if args.nocoverage else f"coverage erode {args.coverage_erode}"
    mask_desc = (
        f"on (dilate {args.automask_dilate}, σ {args.automask_sigma:.1f} vox, {cover_desc})"
        if automask
        else "off"
    )
    if dual3d:
        mode = f"2-axis joint {args.backend}"
    elif dual:
        mode = "2-D dual-PE"
    elif args.backend == "flow":
        mode = f"{'1-D PE' if pe_only else '2-D'} flow"
    else:
        mode = args.backend
    acc_desc = f"refine={args.refine}, jacobian={'on' if args.jacobian else 'off'}"
    if preset:
        acc_desc = f"[{preset}] " + acc_desc
    print_cli_header("ffs_locomoco", "Residual nonlinear motion correction")
    print_cli_section("Configuration", leading_blank=False)
    print(f"   input: {args.input}, shape={data.shape}, device={device}")
    print(f"   data resampling kernel: {_warp_kernel_label(args.warp_interp, args.warp_radius)}")
    if dual3d:
        axes_desc = (
            f"primary PE {args.pe_dir[0]} (axis {pe_axis}) + partition "
            f"{args.pe_dir2 or args.pe_dir[1]} (axis {pe_axis2})"
        )
    elif dual:
        axes_desc = f"dual in-plane PE {args.pe_dir} (axes {pe_axes})"
    elif partition_only:
        axes_desc = f"partition {args.pe_dir2} (axis {pe_axis}), primary PE assumed corrected"
    else:
        axes_desc = f"PE {args.pe_dir[0]} (axis {pe_axis})"
    print(
        f"   {axes_desc}, slice axis={slice_axis}, backend={args.backend}, "
        f"ref={args.ref_label}, do_blur={args.do_blur}mm (σ={smooth_sigma:.2f}vox), "
        f"{mode}, automask={mask_desc}"
    )
    if hpf_sigma > 0:
        print(f"   ⚗️  estimation spatial high-pass: {args.hpf_spatial}mm (σ={hpf_sigma:.2f}vox)")
    if args.match != "none":
        note = " — inert here, qwarp scores ncc on the RAW data" if qwarp_backend else ""
        print(f"   ⚗️  estimation intensity match: {args.match} (σ={args.match_sigma:g}vox){note}")
    print(f"   accuracy: {acc_desc}")
    if qwarp_backend:
        print(
            f"   🪄 qwarp field (E=1, ncc to {qwarp_ref_mode} of raw, refine={qwarp_refine}, "
            f"no flow pass): "
            f"minpatch={_axis_txt(args.qwarp_minpatch)}, levels={args.qwarp_levels}, "
            f"iters={args.qwarp_iters}, optimizer={args.qwarp_optimizer}"
        )

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
        # Rotation-aware converges via the anchor/fuse, not reference-refinement, so only
        # -refine / -jacobian (and the refine rounds inside the presets) are inert here.
        # The presets' extra iters / levels / search density DO feed the estimator.
        if args.refine or args.jacobian or preset:
            print(
                "ℹ️  rotation-aware: -refine and -jacobian (and the refine rounds inside "
                "-workhard/-superhard) don't apply — convergence is the anchor/fuse. The "
                "presets' extra iters/levels/search DO apply; tune further with -fuse / -iters."
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
            est_data,
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
            warp_radius=args.warp_radius,
            fuse=args.fuse,
            fuse_thresh=args.fuse_thresh,
            fuse_weight=args.fuse_weight,
            first_n=args.first_n,
            is_3dacq=args.is_3dacq,
            automask=automask,
            automask_dilate=args.automask_dilate,
            automask_sigma=args.automask_sigma,
            coverage_erode=None if args.nocoverage else args.coverage_erode,
            noshift_margin=args.noshift_margin,
            reg_sigma=args.reg_sigma,
            peak_mode="argmax" if args.argmax else "first_peak",
            search_min_steps=args.search_min_steps,
            device=device,
        )
    elif skip_flow_qwarp:
        # -backend qwarp owns the whole field, so skip the flow estimation entirely: the
        # reference is a plain temporal median of the raw series (correct for already
        # moco'd/NORDIC'd input). Use -backend flow -final_qwarp for flow refinement.
        from fastfuncstuff.processing.locomoco import make_raw_reference_result

        result = make_raw_reference_result(
            data, pe_axis, slice_axis, is_3dacq=args.is_3dacq, dual=dual
        )
    else:
        result = estimate_residual_flow(
            est_data,
            pe_axis,
            slice_axis,
            ref_mode=args.ref,
            backend=prepass_backend,
            smooth_sigma=smooth_sigma,
            n_levels=args.levels,
            n_iters=args.iters,
            window_sigma=args.window,
            pe_only=pe_only,
            dual=dual,
            null_axis=null_axis,
            null_min_r2=args.pe_null_min_r2,
            null_skip=args.pe_null_skip,
            max_shift=args.max_shift,
            trial_step=args.xcorr_step,
            patch=args.patch,
            stride=args.stride,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            refine_rounds=args.refine,
            converge=args.converge,
            converge_rel=args.converge_rel,
            first_n=args.first_n,
            jacobian=args.jacobian,
            is_3dacq=args.is_3dacq,
            pe_axis2=pe_axis2 if dual3d else None,
            automask=automask,
            automask_dilate=args.automask_dilate,
            automask_sigma=args.automask_sigma,
            coverage_erode=None if args.nocoverage else args.coverage_erode,
            noshift_margin=args.noshift_margin,
            reg_sigma=args.reg_sigma,
            peak_mode="argmax" if args.argmax else "first_peak",
            search_min_steps=args.search_min_steps,
            save_corr_curve=corr_curve_frame,
            hpf_sigma=hpf_sigma,
            match=args.match,
            match_sigma=args.match_sigma,
            ngf_eta_q=args.match_eta_q,
            paired_bins=paired_bins,
            device=device,
        )

    if qwarp_backend or args.final_qwarp:
        # Reuse the joint machinery at E=1 (alpha=[1], no scaling): wrap the single result,
        # register raw frames (backend) or the corrected series (polish) to its median.
        from fastfuncstuff.processing.locomoco import MultiEchoLocomocoResult, polish_me_result

        print(
            f"🪄 qwarp backend: registering frames to the {qwarp_ref_mode}-of-raw "
            f"reference ({qwarp_refine + 1} pass{'es' if qwarp_refine else ''})..."
            if qwarp_backend
            else "🪄 Polishing residual with qwarp..."
        )
        # On the skip-flow path there is no estimated field (the placeholder canonical
        # tensors make pe_displacement() invalid); qwarp owns the field, so seed zeros.
        if skip_flow_qwarp:
            w_seed = torch.zeros(result.orig_shape)
            w_seed1 = torch.zeros(result.orig_shape) if dual3d else None
        elif dual3d:
            comps = result.pe_displacements()
            w_seed1, w_seed = comps[0][2], comps[1][2]
        else:
            w_seed, w_seed1 = result.pe_displacement(), None
        wrapped = MultiEchoLocomocoResult(
            per_echo=[result],
            alpha=torch.tensor([1.0]),
            echo_times=torch.tensor([1.0]),
            w_field=w_seed,
            # Single echo: pe_axis names the axis the (trivial) scaling law applies to,
            # which for a two-axis run is the partition — matching the ME convention.
            pe_axis=pe_axis2 if dual3d else result.pe_axis,
            linearity_r2=1.0,
            w_field_pe1=w_seed1,
            pe_axis1=pe_axis if dual3d else None,
        )
        polished = polish_me_result(
            wrapped,
            minpatch=args.qwarp_minpatch,
            n_levels=args.qwarp_levels,
            iters=args.qwarp_iters,
            cost=args.qwarp_cost,
            optimizer=args.qwarp_optimizer,
            compile=args.qwarp_compile,
            full=qwarp_backend,
            ref_mode=qwarp_ref_mode,
            refine=qwarp_refine,
            # 2-D multi-slice: each slice is its own acquisition instant, so the qwarp
            # patches stay 2-D like the estimator. -is_3dacq is one shot -> 3-D patches.
            slicewise=not (args.is_3dacq or args.qwarp_3d),
            # Polish takes raw too: it warps it once by the composed w+r instead of
            # resampling the already-corrected series a second time.
            raw_datas=[data],
            device=device,
        )
        result = polished.per_echo[0]

    if _detask_pre_estimation(args):
        # The estimator was shown task-cut images (notched or design-projected), so the
        # series it warped carries that cut too. Re-derive it from the untouched input
        # with the same field.
        from fastfuncstuff.processing.locomoco import resample_from_raw

        result = resample_from_raw(
            result,
            data,
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            device=device,
            desc="detask resample",
        )

    print_cli_section("Outputs")

    # Task coupling runs BEFORE any output: it must measure the ORIGINAL field (running
    # it after -detask would report the fix's own output and always say "no task"), and
    # -detask has to replace the field before the warp/corrected series are written.
    if args.events and tr_sec:
        result = _task_stage(result, data, stem, ext, affine, args, tr_sec, device)
    elif args.write_pc_maps is not None:
        # Without -events there is no active mask, so the maps go out unscored. Looking
        # at them is the point; the score is the optional part.
        _write_pc_maps(result, stem, ext, affine, args, None, None, device)

    if not args.no_warp:
        from fastfuncstuff.processing.medic import save_medic_warp

        comps = result.warp_components()  # [(nifti_axis, disp), ...]
        (primary_axis, primary_disp), *rest = comps
        as_5d = args.warp_format == "5d"
        with spinner(f"Writing {Path(stem).name}_warp{ext}"):
            save_medic_warp(
                primary_disp,
                primary_axis,
                affine,
                stem,
                nii_ext=ext,
                as_5d=as_5d,
                extra_components=[(d, a) for a, d in rest],
            )

    if args.want_pcs is not None:
        _write_warp_pcs(result.warp_components(), stem, args.want_pcs)

    # One materialization for every consumer below. On the estimator paths that
    # hold the series in canonical axis order, corrected_series() is a permute +
    # contiguous -- a full copy of the 4-D series (a GB for a long single-echo
    # run), so calling it once per output is what made writing cost more than
    # estimating. _neg_clip is out-of-place, so sharing one tensor is exact.
    want_corrected = (
        not args.no_corrected
        or args.save_mean
        or args.save_max
        or args.save_min
        or (_want_qc(args) and _want_corrected_qc(args))
        # The task-fit "after" measures the corrected DATA, so it needs the series even
        # under -no_corrected. One warp, and it is what answers whether the correction
        # helped or hurt the thing the scan was collected for.
        or getattr(args, "_task_after", None) is not None
    )
    corrected_series = result.corrected_series() if want_corrected else None
    _write_data_task_after(corrected_series, stem, args)

    if not args.no_corrected:
        assert corrected_series is not None
        corr_path = f"{stem}_locomoco{ext}"
        with spinner(f"Writing {Path(corr_path).name}"):
            save_nifti(
                _neg_clip(corrected_series.numpy(), args.allow_neg),
                corr_path,
                affine=affine,
            )

    # Temporal reductions of the corrected series ((nx, ny, nz, T) -> over T).
    # max/min are coverage images, not contrast images — see the -save_max help.
    for want, which, reduce in (
        (args.save_mean, "mean", lambda s: s.mean(dim=-1)),
        (args.save_max, "max", lambda s: s.amax(dim=-1)),
        (args.save_min, "min", lambda s: s.amin(dim=-1)),
    ):
        if not want:
            continue
        assert corrected_series is not None
        out_path = f"{stem}_locomoco_{which}{ext}"
        with spinner(f"Writing {Path(out_path).name}"):
            save_nifti(
                _neg_clip(reduce(corrected_series).numpy(), args.allow_neg),
                out_path,
                affine=affine,
            )

    if _want_qc(args):
        # Works even with -no_corrected; only the QC maps that read the corrected
        # series get it.
        corrected = (
            corrected_series.numpy()
            if corrected_series is not None and _want_corrected_qc(args)
            else None
        )
        _write_qc_diag(args, corrected, data, f"{stem}_locomoco", f"{stem}_orig", ext, affine)
    del corrected_series

    if not args.no_flow:
        if result.pe_axis2 is not None:
            # Two physically distinct artifacts: write each as its OWN signed map. The
            # magnitude/angle pair below is for the legacy in-plane-vector case, where the
            # two components really are one vector; here they are not.
            for label, _axis, field in result.pe_displacements():
                fpath = f"{stem}_flow_{label}{ext}"
                with spinner(f"Writing {Path(fpath).name}"):
                    save_nifti(field.numpy(), fpath, affine=affine)
            if result.null_field is not None:
                # The un-encoded axis, as estimated and BEFORE it was regressed out.
                # It is the diagnostic the whole flag rests on: an axis that cannot
                # physically move, so whatever is here is what the estimator invented.
                npath = f"{stem}_flow_null{ext}"
                with spinner(f"Writing {Path(npath).name}"):
                    save_nifti(result.null_field.numpy(), npath, affine=affine)
        elif dual:
            # No single signed scalar holds a 2-D vector — split into magnitude + angle.
            mag_path, ang_path = f"{stem}_flowmag{ext}", f"{stem}_flowang{ext}"
            with spinner(f"Writing {Path(mag_path).name}"):
                save_nifti(result.flow_magnitude().numpy(), mag_path, affine=affine)
            with spinner(f"Writing {Path(ang_path).name}"):
                save_nifti(result.flow_angle().numpy(), ang_path, affine=affine)
        else:
            flow_path = f"{stem}_flow{ext}"
            with spinner(f"Writing {Path(flow_path).name}"):
                save_nifti(result.pe_displacement().numpy(), flow_path, affine=affine)
            print(
                f"  • signed PE flow 4D (voxels, ± = direction; scrub like a series): {flow_path}"
            )

    _write_dual_axis_diagnostics(result, stem, ext, affine, args, data)

    _write_xcorr_diagnostics(result, stem, ext, affine, args)

    if not args.no_movie:
        fmt = args.movie_format or ("mp4" if _find_ffmpeg() else "gif")
        frames = result.flow_movie(max_mag=args.flow_max)
        movie_path = f"{stem}_flow.{fmt}"
        with spinner(f"Writing {Path(movie_path).name}"):
            _write_movie(frames, movie_path, args.fps, fmt)

    print_cli_footer("ffs_locomoco")
    return 0


if __name__ == "__main__":
    sys.exit(main())
