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
        choices=("none", "meanstd", "localnorm", "gradmag"),
        default="localnorm",
        help="[-me_interecho -backend flow] Intensity matching applied to each ECHO PAIR before"
        " the LK solve. The cross-TE counterpart of -match (which matches frames over TIME, and"
        " is ignored on the multi-echo path).\n"
        "Optical flow assumes brightness constancy, which consecutive echoes violate by"
        " construction — T2* dims the later echo everywhere. Unmatched, that decay step is read"
        " as displacement and the field diverges.\n"
        "  none       the raw residual.\n"
        "  localnorm  local z-score both sides.\n"
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
        choices=("none", "meanstd", "localnorm", "gradmag"),
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
        default=2.0,
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
        " runs, and the estimator then pools over -window on the matched data.",
    )

    search = p.add_argument_group("Searchlight backends (-backend phase / xcorr)")
    search.add_argument(
        "-max_shift",
        type=float,
        default=3.0,
        metavar="VOX",
        help="Largest PE shift to allow (voxels). Set just above the biggest residual "
        "shift you expect (sub- to a few voxels here). Smaller = faster xcorr (fewer "
        "trial offsets) and a tighter phase no-wrap band; too small clips real motion. "
        "phase and xcorr search within it; flow clamps its accumulated field to it, "
        "which is what stops a textureless slab-end slice random-walking to hundreds "
        "of voxels. All three also use it as the refine divergence threshold.",
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
        default=0.5,
        metavar="VOX",
        help="[xcorr only] Trial-offset spacing (voxels) of the correlation search — "
        "xcorr's sub-voxel knob, like -iters is for the others. The peak is fit by a "
        "5-point parabola either way; a finer step (e.g. 0.25) samples the curve more "
        "densely for a touch more accuracy at ~2× the trials. 0.5 is the sweet spot.",
    )
    search.add_argument(
        "-reg_sigma",
        "-reg-sigma",
        type=float,
        default=1.5,
        metavar="VOX",
        help="[xcorr] Confidence-weighted spatial smoothing of the searchlight field "
        "(Gaussian sigma, voxels). The displacement field is physically smooth, so each "
        "voxel borrows from its high-confidence neighbours (peak quality × prominence "
        "over no-shift); high-confidence voxels keep their own estimate. Fixes lone "
        "railed/spurious peaks without blurring real structure. 0 = off. Default 1.5.",
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
        default=7,
        dest="qwarp_minpatch",
        help="Finest qwarp patch size (voxels), in-plane only when the patches are 2-D "
        "slicewise.\n"
        "The default suits the -final_qwarp polish; try 9 for -backend qwarp, which has the"
        " whole field to find rather than a residual.",
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
        "'the subject moved with the task'. Writes {prefix}_taskr2_* maps and a report. "
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
        "-detask",
        action="store_true",
        help="REMOVE the task-locked part of the field, keeping drift and everything "
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
        if args.detask:
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

    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.processing.locomoco import _brain_mask_from
    from fastfuncstuff.stats.task_coupling import (
        co_location,
        contamination_slope,
        default_polort,
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
    for label, axis, field in result.pe_displacements():
        tc = task_coupling(field, design, **kwargs)
        coloc = co_location(tc.r, data_tc.r, mask)
        r_sum, q_sum = tc.summarize(resp), tc.summarize(quiet)
        name = "PE displacement" if label == "pe1" else "partition displacement"
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
        report.append(
            format_task_coupling_report(
                tc,
                data_tc,
                coloc,
                units="voxels",
                label=f"{name} (axis {axis})",
                responding=r_sum,
                quiet=q_sum,
                top_frac=args.task_top_frac,
                slope=slope,
                enrichment=enrich,
                active_thresh=cut,
            )
        )
        best = r_sum["conditions"][best_k]
        print(
            f"  • {name}: ENRICHMENT {enrich['enrichment']:.2f}x "
            f"({enrich['energy_share'] * 100:.1f}% of task-locked displacement in "
            f"{enrich['voxel_share'] * 100:.1f}% of voxels); |r| {best['abs_r_median']:.3f} "
            f"there ({best['label']}); kappa {slope['kappa']:+.3f} (R² {slope['r2']:.3f})"
        )
        # One 4-D file per axis, conditions on the 4th axis — a viewer scrubs the
        # conditions, and one file per condition would flood the output directory.
        for kind, arr in (("taskr", tc.r), ("taskrms", tc.task_rms)):
            path = f"{stem}_{kind}_{label}{ext}"
            with spinner(f"Writing {Path(path).name}"):
                save_nifti(
                    arr.float().squeeze(-1).numpy()
                    if kind == "taskr" and arr.shape[-1] == 1
                    else arr.float().numpy(),
                    path,
                    affine=affine,
                )
        print(f"    {stem}_taskr_{label}{ext} · {stem}_taskrms_{label}{ext}")

    dpath = f"{stem}_taskr_data{ext}"
    with spinner(f"Writing {Path(dpath).name}"):
        dr = data_tc.r.float()
        save_nifti((dr.squeeze(-1) if dr.shape[-1] == 1 else dr).numpy(), dpath, affine=affine)
    print(f"  • data coupling r (the BOLD response, for co-location): {dpath}")

    tpath = f"{stem}_locomoco_taskcoupling.txt"
    Path(tpath).write_text(("\n" + "-" * 70 + "\n\n").join(report))
    print(f"  • task-coupling report: {tpath}")
    for block in report:
        print()
        print("\n".join("    " + ln for ln in block.rstrip().splitlines()))
    return design, polort


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
        design, polort = _write_task_diagnostics(result, data, stem, ext, affine, args, tr)
    except (ValueError, RuntimeError) as exc:
        # A diagnostic must never destroy a finished fit. The estimation above can be
        # minutes of GPU time; losing it to a bad -tr or an empty mask is the wrong
        # trade. -detask is refused loudly, because silently skipping the fix the user
        # asked for would be worse than the crash.
        print(f"  ⚠️  task coupling skipped: {exc}")
        if args.detask:
            print("  ⚠️  -detask NOT applied — the outputs below are the ORIGINAL field.")
        return result

    if not args.detask:
        return result

    print("  ── de-tasking (everything below is derived from the CLEANED field) ──")
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
        print(
            f"  • axis separability 4D (1 = axes cleanly separable, 0 = aperture-"
            f"ambiguous): {spath}"
        )

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
    print(f"  • per-voxel coupling r 3D [{cov}]: {rpath}")
    print(f"  • per-frame coupling r: {frame_path}")
    print(f"  • coupling report: {cpath}")
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
            print(f"  • searchlight confidence map 4D: {p}")
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
            print(f"  • per-voxel corr landscape 4D: {p}")
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
    print(f"  • warp PCs ({scores.shape[1]}, denoising regressors, var {var_pct}): {pcs_path}")


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
    print(f"  • tSNR: {path}")


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
    print(f"  • {'first/last/diff' if include_diff else 'first/last'}: {path}")


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
    if args.detask:
        # The late warning fires only after the whole solve; a run can be many minutes of
        # work before the user learns nothing was de-tasked. Say it up front too.
        print(
            "   ⚠️  -detask is not wired for the multi-echo path — the outputs will NOT "
            "be de-tasked (the task diagnostic is still written)."
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
            f"   🪄 qwarp field: minpatch={args.qwarp_minpatch}, levels={args.qwarp_levels}, "
            f"iters={args.qwarp_iters}, cost={args.qwarp_cost}, optimizer={args.qwarp_optimizer}"
        )
    else:
        print(
            f"   TEs [{te_str}] ms, PE {_pe_label(args)} (axis {pe_axis}), backend={args.backend}, "
            f"{ref_note}, mode={mode}, scaling={scaling}, "
            f"levels={args.levels}, iters={args.iters}, {refine_note}, "
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
            datas,
            args.echo_times,
            pe_axis,
            slice_axis,
            flat_scaling=args.me_flat_scaling,
        )
    elif args.me_interecho:
        from fastfuncstuff.processing.locomoco import estimate_residual_flow_me_interecho

        result = estimate_residual_flow_me_interecho(
            datas,
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
                datas,
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
            datas,
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
            warp_interp=args.warp_interp,
            warp_radius=args.warp_radius,
            hpf_sigma=hpf_sigma,
            device=device,
        )
    else:
        result = estimate_residual_flow_multiecho(
            datas,
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

    print_cli_section("Outputs")
    as_5d = args.warp_format == "5d"
    for j, res in enumerate(result.per_echo):
        estem = f"{stem}_e{j + 1}"
        if not args.no_warp:
            from fastfuncstuff.processing.medic import save_medic_warp

            axis, disp = res.warp_components()[0]
            with spinner(f"Writing {Path(estem).name}_warp{ext}"):
                warp_path = save_medic_warp(disp, axis, affine, estem, nii_ext=ext, as_5d=as_5d)
            print(f"  • echo {j + 1} warp (ffs_nwarp, axis {axis}): {warp_path}")
        # Materialized once per echo for both the write and the QC maps below --
        # see the single-echo block for why the repeat call is not free.
        want_corrected = not args.no_corrected or (_want_qc(args) and _want_corrected_qc(args))
        corrected_series = res.corrected_series() if want_corrected else None
        if not args.no_corrected:
            assert corrected_series is not None
            corr_path = f"{estem}_locomoco{ext}"
            with spinner(f"Writing {Path(corr_path).name}"):
                save_nifti(
                    _neg_clip(corrected_series.numpy(), args.allow_neg),
                    corr_path,
                    affine=affine,
                )
            print(f"  • echo {j + 1} corrected series: {corr_path}")
        if not args.no_flow:
            if res.pe_axis2 is not None:
                names = {"pe1": "primary PE", "pe2": "partition"}
                for label, axis, field in res.pe_displacements():
                    fpath = f"{estem}_flow_{label}{ext}"
                    with spinner(f"Writing {Path(fpath).name}"):
                        save_nifti(field.numpy(), fpath, affine=affine)
                    print(
                        f"  • echo {j + 1} signed {names[label]} flow 4D "
                        f"(voxels, axis {axis}): {fpath}"
                    )
            else:
                flow_path = f"{estem}_flow{ext}"
                with spinner(f"Writing {Path(flow_path).name}"):
                    save_nifti(res.pe_displacement().numpy(), flow_path, affine=affine)
                print(f"  • echo {j + 1} signed PE flow 4D (voxels): {flow_path}")
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

    # Two-axis extras. Echo 1 carries alpha = 1 on BOTH axes (primary PE is flat by
    # construction, and the partition law is normalised to echo 1), so its per-echo
    # fields ARE the shared fields — the coupling measured on it is the shared-field
    # coupling, not one echo's view of it.
    _write_dual_axis_diagnostics(result.per_echo[0], stem, ext, affine, args, datas[0])

    if args.events and tr_sec:
        # Echo 1's series and the shared field: the coupling question is about the ONE
        # field every echo is corrected by, not each echo's scaled copy of it.
        _write_task_diagnostics(result.per_echo[0], datas[0], stem, ext, affine, args, tr_sec)
        if args.detask:
            # The multi-echo correction is one SHARED field scaled per echo; de-tasking
            # it means re-deriving every echo's warp and series from the cleaned w, which
            # is not wired yet. Refusing beats silently cleaning one echo's copy.
            print(
                "  ⚠️  -detask is not wired for the multi-echo path (the shared field is "
                "scaled per echo); the outputs below are NOT de-tasked."
            )

    # Shared scaling diagnostic: learned alpha vs echo time, and the linearity r².
    alpha_path = f"{stem}_locomoco_alpha.1D"
    with open(alpha_path, "w") as f:
        f.write("# ffs_locomoco multi-echo per-echo scaling (alpha_e · shared field)\n")
        f.write(f"# linear-in-TE r² = {result.linearity_r2:.6f}\n")
        f.write(f"# echo_TE_ms   {result.alpha_label}\n")
        for te_v, a_v in zip(result.echo_times.tolist(), result.alpha.tolist(), strict=True):
            f.write(f"  {te_v:10.4f}  {a_v:12.6f}\n")
    print(f"  • per-echo scaling + linearity: {alpha_path}")

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
            f"minpatch={args.qwarp_minpatch}, levels={args.qwarp_levels}, "
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
            data,
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

    print_cli_section("Outputs")

    # Task coupling runs BEFORE any output: it must measure the ORIGINAL field (running
    # it after -detask would report the fix's own output and always say "no task"), and
    # -detask has to replace the field before the warp/corrected series are written.
    if args.events and tr_sec:
        result = _task_stage(result, data, stem, ext, affine, args, tr_sec, device)

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
    )
    corrected_series = result.corrected_series() if want_corrected else None

    if not args.no_corrected:
        assert corrected_series is not None
        corr_path = f"{stem}_locomoco{ext}"
        with spinner(f"Writing {Path(corr_path).name}"):
            save_nifti(
                _neg_clip(corrected_series.numpy(), args.allow_neg),
                corr_path,
                affine=affine,
            )
        print(f"  • corrected series: {corr_path}")

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
        print(f"  • corrected {which}: {out_path}")

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
            names = {"pe1": "primary PE", "pe2": "partition"}
            for label, axis, field in result.pe_displacements():
                fpath = f"{stem}_flow_{label}{ext}"
                with spinner(f"Writing {Path(fpath).name}"):
                    save_nifti(field.numpy(), fpath, affine=affine)
                print(f"  • signed {names[label]} flow 4D (voxels, axis {axis}): {fpath}")
        elif dual:
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

    _write_dual_axis_diagnostics(result, stem, ext, affine, args, data)

    _write_xcorr_diagnostics(result, stem, ext, affine, args)

    if not args.no_movie:
        fmt = args.movie_format or ("mp4" if _find_ffmpeg() else "gif")
        frames = result.flow_movie(max_mag=args.flow_max)
        movie_path = f"{stem}_flow.{fmt}"
        with spinner(f"Writing {Path(movie_path).name}"):
            actual = _write_movie(frames, movie_path, args.fps, fmt)
        print(f"  • flow movie (circular-phase wheel): {actual}")

    print_cli_footer("ffs_locomoco")
    return 0


if __name__ == "__main__":
    sys.exit(main())
