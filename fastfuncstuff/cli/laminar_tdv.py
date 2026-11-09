"""ffs_laminar_tdv (EXPERIMENTAL): remove ascending-vein transport from GE-BOLD runs.

Command: ffs_laminar_tdv (registered as entry point in pyproject.toml)

Usage:
    ffs_laminar_tdv -input run1.nii.gz run2.nii.gz run3.nii.gz \\
        -fs_subj $SUBJECTS_DIR/sub01 -repeated_runs -prefix ltdv/sub01
    ffs_laminar_tdv -input rest.nii.gz -fs_subj $SUBJECTS_DIR/sub01 \\
        -ridge 0.1 -prefix ltdv/rest

Not a validated preprocessing step: the vascular physiology is fixed (nothing
is calibrated on your data), and on ds003427 the corrected GE moved slightly
AWAY from SE. See the wiki's Laminar timeseries deveining note.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_bids_events_args,
    add_device_arg,
    add_microtime_offset_arg,
    add_round_onsets_arguments,
    add_verbose_arg,
)
from fastfuncstuff.surface.ribbon_args import add_ribbon_args

_EPILOG = """\
what it does

  Ascending veins carry deoxygenated blood from deep cortex to the surface, so
  a GE-BOLD response at one depth also appears, later and smeared, at every
  depth above it.  This tool estimates that carried-up part from the data and
  subtracts it.  What stays is each depth's response to its OWN blood flow
  (its local BOLD, with the normal HRF shape and its own venous part), plus
  every voxel's residual, untouched.  It does not estimate neural activity.

    1. columns   ribbon voxels -> geodesic patches of the midthickness mesh
                 (-patch_mm), through each voxel's owning vertex, as in
                 ffs_hrftdv: a patch is every depth under a few mm of cortex.
                 Or -depth/-gmfrac (segmentation depths, patches grown through
                 GM voxels), or your own -columns labels.
    2. profiles  each run -> percent signal change, Legendre drift removed
                 (-polort), averaged into -depths bins per patch.
    3. model     the Havlicek-Uludag laminar vascular model (as in ffs_laminar),
                 linearised about rest, gives one depth x depth transfer matrix
                 per frequency: how a blood-flow change at depth n shows up in
                 BOLD at depth k, with the ascending-vein delay.  Its parameters
                 are FIXED: published priors, or -vascular_json.
    4. inverse   per patch and frequency, a ridge inverse finds the depth
                 blood-flow histories that explain the profiles (-ridge).
    5. split     model(all depths' inputs) - model(each depth's own input) =
                 the transported part; model(own input) = the local part.  Both
                 are read out at every voxel's depth.
    6. voxels    (-removal voxel, the default) each voxel regresses its own
                 data on its column's local part (free), transported part (a
                 gain, pulled toward 1) and that part's time derivative (a
                 timing shift, pulled toward 0).  It loses gain x transport +
                 shift x derivative; its local part and residual stay.  How
                 hard the pull is, per voxel, is chosen by leave-one-run-out
                 (-gain_lambdas).  -removal column skips this: every voxel at a
                 depth loses the same column transport.
       Each voxel's mean and drift are kept exactly.

  Why per voxel: the column describes what drains through a depth and when; a
  vein voxel carries several times more of it than the column mean, and the
  column alone takes from it what it takes from its neighbours.  Because the
  local part competes for the same variance, a gain only takes what the
  voxel's own local response cannot explain.  The column profiles are
  precision-weighted (a voxel weighs min(1, (median SD at its depth / its SD)^2)),
  so veins do not define their column.

  Coverage: the ribbon, plus a shell up to -extend_mm beyond it (pial veins sit
  just outside pial).  Shell voxels take the column and depth of their nearest
  ribbon voxel and only get voxel gains; they do not build the columns.
  Subcortical and other voxels, those no run observed (see -coverage), and
  patches that do not reach every model depth pass through bit-identical.

ridge regularization (-ridge)

  Each LAMBDA is a ridge penalty relative to the patch's own model scale:
  0.001 trusts the data almost fully (sharp but noisy), 1 shrinks hard.  The
  penalty also rises with frequency, by 1 + (f / -cutoff_hz)^4, so fluctuations
  faster than the slow vascular model can explain are not inverted.  More
  shrinkage means a smaller transport estimate, so LESS is removed: when the
  data are unsure the tool corrects less, not more.

  With -removal voxel the gains rescale the transport per voxel, so -ridge
  shapes the column estimate more than it sets the amount removed.

  Several values: each patch picks its own (-ridge_cv).

    split  each patch's voxels go into two halves (whole -ridge_cv_mm cubes,
           so neighbours sharing resampled noise stay together); the inverse
           of one half at each LAMBDA predicts the other half, same run, same
           moment, and both ways are pooled.  Any schedule works, rest too,
           and every fluctuation the halves share counts as signal -- what
           the correction removes.  A half is twice as noisy as the patch, so
           it is fit at twice the LAMBDA being scored.  split_leak in the
           bucket is the halves' correlation above the model's band: near 0
           when their noise is independent; high values mean shared noise is
           being read as signal, favouring too little shrinkage.
    loro   predict each run from the mean of the others.  Needs every run to
           have the same task timing (-repeated_runs), and rewards only the
           task-locked part: on repeated-run data it mostly picks the
           largest LAMBDA, i.e. smoothing.

targets (-target)

  local  keep each depth's response to its own blood flow, including the
         ascending-vein signal it makes locally; remove only what drains in
         from below.  The default, and the usual goal.
  micro  keep only the venule (microvascular) part: also strips each depth's
         local ascending-vein signal.  Larger change; closer in spirit to SE.

events (optional; any schedule, each run its own)

  With -events or -onsets, the fixed physiology can be chosen from a grid
  (-cv_p0) instead of taken on trust.  Each run's events, through the model's
  own stimulus -> CBF response, give every depth a CBF source with a free
  amplitude per depth and condition; the vascular model turns those into a
  predicted GE profile.  Leave-one-run-out: amplitudes fit on the other runs
  predict the held-out run, whose schedule may differ.  The candidate with the
  best held-out R² pooled over all patches is used for the correction, which
  itself still uses no events, so non-task fluctuations are corrected too.
  Pooled, because the score is weak per patch (below).

  What it can tell apart: parameters that change response SHAPE (t0v, the tau_*)
  separate clearly.  s_d, the drainage strength, barely does: free amplitudes
  per depth absorb most of it, so its curve is shallow (simulated, 0.4 vs 0.8:
  ~0.001 R²) and only pooling over many patches resolves it.  The score assumes
  every depth's own response has one shape; a genuinely later superficial
  response is read as drainage.  Events do not choose -ridge: a design-fitted
  signal has no noise left for the ridge to trade against.

voxel gains (-gain_lambdas)

  Each LAMBDA pulls a voxel's gain toward 1 and shift toward 0 with a strength
  relative to its own transport regressor's energy: 1e4 is the column model,
  0.01 a nearly free fit.  Every voxel keeps the LAMBDA whose fit on the other
  runs best predicts each held-out run; any schedule works, events are not
  used, and at least 2 runs are needed.  gain_cv in the bucket is the held-out
  R² gained over the column model; 0 means the voxel looks like its column.

physiology (-vascular_json)

  {"p0": {"s_d": 0.6, "w_v": 0.4},    baseline values (laminar/params.py:P0)
   "parameters": {"t0v": 0.2}}        scalar log-deviations from those

  s_d, the slope of ascending-vein blood volume toward pial, matters most: it
  sets how much of each depth's signal reaches the depths above.  Others: V0t
  (total CBV), w_v (venule vs vein share), t0v (venule transit, s), E0v/E0d,
  al_v/al_d (Grubb), nr, tau_* (flow-volume lag, s).

inputs and space

  -input runs share one grid and length; TR from the header (or -tr).  -te and
  -b0 come from the first run's BIDS sidecar (EchoTime, MagneticFieldStrength)
  unless given; the tool refuses to guess them.  Surfaces must be in the
  scanner space of -input (an -fs_subj aligned to the functional data, or
  meshes already moved there via -white_mesh/-pial_mesh), as for ffs_hrftdv.

outputs

  {prefix}_corrected_run-NN.nii.gz    input with the transport removed, input units
  {prefix}_removed_psc_run-NN.nii.gz  what was removed, % signal change
  {prefix}_fit_psc_run-NN.nii.gz      (-save_fit) the full model reconstruction:
                                      a smoothing-only control
  {prefix}_ltdv.nii.gz                bucket: ridge, ge_cv_r2, noise_gain,
                                      removed_rms, valid, shell, patch, depth
                                      (+ voxel: gain_local, gain_transport,
                                      gain_shift, gain_lambda, gain_cv)
                                      (+ events: phys_cv_r2, phys_best)
                                      (+ -ridge_cv split: split_leak)
  {prefix}_physiology_cv.tsv          (events) each candidate's pooled held-out R²
  {prefix}_metadata.json

  ridge is each patch's LAMBDA; ge_cv_r2 the held-out R² of the full model
  (held-out runs, or held-out voxel halves with -ridge_cv split; NaN for a
  patch too small to split, which takes the majority's LAMBDA; never SE); noise_gain the white-noise RMS gain of the
  correction (1 = noise unchanged); removed_rms the RMS of the removed series
  (% signal change, all runs); depth is equivolume, 0 at white, 1 at pial
  (a shell voxel's is its nearest ribbon voxel's).  -save_fit with -removal
  voxel writes each voxel's own fit (local + gain x transport + shift).
  phys_cv_r2 is each patch's held-out R² under the chosen physiology, and
  phys_best the candidate (row of the TSV) that patch alone would have chosen:
  a map of how much the choice varies, not a per-patch correction.

what to expect

  EXPERIMENTAL.  In simulation with the model's own physiology it removes deep
  -> superficial leakage and keeps superficial-only responses (96-98% of
  amplitude).  On ds003427 (7T, segmentation depths) the corrected GE moved
  slightly away from SE (median voxel r with SE .100 -> .083), most likely
  because the fixed physiology does not fit those data (that was the column-only
  version).  With voxel gains on 7T data (0.8 mm, 6 runs), the top 1% of voxels
  by task amplitude lost ~0.3 of their PSC SD instead of ~0.1 (shell: 0.27
  instead of untouched), while typical voxels barely changed.  Check it against an
  independent reference (VASO, SE) before using it.  The correction changes
  the noise covariance a later GLM sees.
"""


def create_parser() -> FfsArgumentParser:
    parser = FfsArgumentParser(
        prog="ffs_laminar_tdv",
        description="EXPERIMENTAL: remove ascending-vein transport from laminar GE-BOLD "
        "timeseries with a linearised vascular model",
        formatter_class=FfsHelpFormatter,
        epilog=_EPILOG,
    )
    req = parser.add_argument_group("Inputs")
    req.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="GE-BOLD runs, one file per run, same grid. Runs may differ in length "
        "with one -ridge value and no -events (each run is inverted on its own)",
    )
    req.add_argument("-prefix", required=True, help="Output prefix")
    req.add_argument("-tr", "-TR", type=float, default=None, help="TR in seconds (default: header)")
    req.add_argument(
        "-te",
        type=float,
        default=None,
        metavar="SEC",
        help="GE echo time in seconds (default: EchoTime from the first run's sidecar)",
    )
    req.add_argument(
        "-b0",
        type=float,
        default=None,
        metavar="TESLA",
        help="Field strength (default: MagneticFieldStrength from the sidecar)",
    )

    ev = parser.add_argument_group("Events (optional: cross-validate the physiology)")
    ev.add_argument("-onsets", nargs="+", default=None, help="AFNI onset files, one per condition")
    ev.add_argument("-durations", nargs="+", default=None, help="Durations for -onsets (s)")
    add_bids_events_args(ev, events_help="BIDS *_events.tsv files, one per run")
    add_round_onsets_arguments(ev, hyphen=False)
    ev.add_argument(
        "-cv_p0",
        nargs="+",
        action="append",
        default=None,
        metavar=("NAME", "VALUE"),
        help="A physiology grid to score: a P0 baseline name and its candidate values "
        "(-cv_p0 s_d 0.2 0.4 0.8). Repeat for more names; every combination is scored. "
        "Default with events: s_d 0.2 0.4 0.8 x t0v 0.5 1 2. See 'events' below",
    )
    ev.add_argument("-microtime_dt", type=float, default=0.1, help="Microtime step (s)")
    add_microtime_offset_arg(ev)

    surf = parser.add_argument_group("Cortical ribbon (surfaces; the usual input)")
    add_ribbon_args(surf)
    surf.add_argument(
        "-patch_mm",
        type=float,
        default=5.0,
        help="Geodesic radius of a cortical patch: the column whose depths are inverted "
        "together. Bigger = less noise per profile, more blurring of real differences "
        "between neighbouring columns",
    )

    surf.add_argument(
        "-extend_mm",
        type=float,
        default=2.0,
        help="Also correct voxels up to this far outside the ribbon (into WM and past "
        "pial, where pial veins sit), each with its nearest ribbon voxel's column and "
        "depth. Voxel removal only; 0 = ribbon only",
    )
    surf.add_argument(
        "-ribbon_frac",
        type=float,
        default=0.1,
        metavar="F",
        help="A voxel joins the ribbon when at least this share of it lies between "
        "white and pial (64 points per voxel), so voxels the ribbon only clips, on the "
        "pial or the WM side, are kept. Its depth and column are measured at its GM part "
        "and its share weights it in the column profiles. 0 = the old voxel-centre rule",
    )
    surf.add_argument(
        "-coverage",
        choices=["auto", "each", "all"],
        default="auto",
        help="Which voxels a run must observe (positive at every volume) to be corrected. "
        "each: a voxel is corrected in every run that saw it, patches are built per run "
        "from that run's voxels, and voxel gains are chosen across the runs that saw it "
        "(a voxel seen once takes the column model) -- for slabs that move between runs. "
        "all: only voxels every run saw. auto: each when it can be (one -ridge value, no "
        "-events, which compare runs sample by sample), else all",
    )

    vol = parser.add_argument_group("Volume depth (instead of surfaces)")
    vol.add_argument(
        "-depth",
        default=None,
        metavar="DSET",
        help="Scalar depth map on the -input grid (e.g. from a segmentation)",
    )
    vol.add_argument(
        "-depth_order",
        choices=("pial_to_white", "white_to_pial"),
        default=None,
        help="Which end of -depth is 0",
    )
    vol.add_argument(
        "-gmfrac",
        default=None,
        metavar="DSET",
        help="GM fraction on the -input grid; voxels below -gm_threshold are left out "
        "(depth 0 is also the background value of most depth maps)",
    )
    vol.add_argument("-gm_threshold", type=float, default=0.5, help="Minimum -gmfrac")
    vol.add_argument(
        "-columns",
        default=None,
        metavar="DSET",
        help="Integer column labels on the -input grid (0 = background) instead of "
        "-patch_mm patches. Works with surfaces too",
    )

    model = parser.add_argument_group("Vascular model")
    model.add_argument(
        "-depths",
        type=int,
        default=5,
        help="Model depths (= depth bins per patch). Each needs voxels in every patch; "
        "more than the data's independent depth samples only adds correlated bins",
    )
    model.add_argument(
        "-target",
        choices=("local", "micro"),
        default="local",
        help="What to keep: each depth's own response incl. its local vein (local), or "
        "venules only (micro). See 'targets' below",
    )
    model.add_argument(
        "-transfer_mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help="static uses the zero-frequency mixing at every frequency (no vein delay): "
        "a control for whether the timing part of the model helps",
    )
    model.add_argument(
        "-vascular_json",
        default=None,
        metavar="JSON",
        help="Override the fixed physiology. See 'physiology' below",
    )

    vox = parser.add_argument_group("Voxel removal")
    vox.add_argument(
        "-removal",
        choices=("voxel", "column"),
        default="voxel",
        help="voxel: each voxel scales its column's transport by its own gain (needs "
        ">= 2 runs). column: every voxel at a depth loses the same column transport",
    )
    vox.add_argument(
        "-gain_lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 1e4],
        metavar="LAMBDA",
        help="How hard a voxel's gain is pulled toward the column model (1e4 = the "
        "column, 0.01 = free); each voxel picks one by leave-one-run-out. See 'voxel "
        "gains' below",
    )

    inv = parser.add_argument_group("Inverse")
    inv.add_argument(
        "-ridge",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        metavar="LAMBDA",
        help="Ridge strength(s), relative to each patch's model scale; several are "
        "chosen between per patch (-ridge_cv). See 'ridge regularization' below",
    )
    inv.add_argument(
        "-ridge_cv",
        choices=["auto", "loro", "split"],
        default="auto",
        help="How several -ridge values are chosen between. loro: predict each run from "
        "the others (needs -repeated_runs). split: predict half of each patch's voxels "
        "from the other half, same run, same moment -- any schedule, rest included. "
        "auto: loro with -repeated_runs, else split",
    )
    inv.add_argument(
        "-ridge_cv_mm",
        type=float,
        default=2.0,
        help="-ridge_cv split: voxels go to a half in cubes this size, so neighbours that "
        "share resampled noise stay together (see split_leak in the bucket)",
    )
    inv.add_argument(
        "-ridge_cv_repeats",
        type=int,
        default=2,
        help="-ridge_cv split: random half assignments averaged",
    )
    inv.add_argument(
        "-repeated_runs",
        action="store_true",
        help="State that every run has the same task timing, which leave-one-run-out "
        "selection of several -ridge values relies on",
    )
    inv.add_argument(
        "-cutoff_hz",
        type=float,
        default=0.1,
        help="Frequency where the ridge penalty starts rising steeply (Hz)",
    )
    inv.add_argument(
        "-pad_seconds",
        type=float,
        default=60.0,
        help="Reflection padding at each run end against FFT wraparound (s); the first "
        "and last this-many seconds are the least trustworthy",
    )

    nuis = parser.add_argument_group("Nuisance and output")
    nuis.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Legendre drift order (default: 1 + floor(run seconds / 150)). The drift "
        "is removed before the inverse and kept exactly in the output",
    )
    nuis.add_argument(
        "-save_fit",
        action="store_true",
        help="Also write {prefix}_fit_psc_run-NN, the full model reconstruction",
    )
    add_device_arg(nuis)
    add_verbose_arg(nuis)
    return parser


def _acquisition(args, first: str) -> tuple[float, float]:
    from fastfuncstuff.autoproc.bids import sidecar_path

    side = sidecar_path(Path(first.split("[")[0]))
    meta = json.loads(side.read_text()) if side.exists() else {}
    te = args.te if args.te is not None else meta.get("EchoTime")
    b0 = args.b0 if args.b0 is not None else meta.get("MagneticFieldStrength")
    if te is None or b0 is None:
        missing = " and ".join(n for n, v in (("-te", te), ("-b0", b0)) if v is None)
        sys.exit(f"ERROR: give {missing}: not in {side.name}" if side.exists() else
                 f"ERROR: give {missing}: no sidecar {side.name}")  # fmt: skip
    if not 0 < float(te) < 0.5:
        sys.exit(f"ERROR: echo time {te} is not in seconds")
    return float(te), float(b0)


def _volume_units(args, header, load_image):
    """GM mask and depth (0 = pial) in load_image's (Z, Y, X) order."""
    depth, dh = load_image(args.depth)
    frac, fh = load_image(args.gmfrac)
    if depth.ndim != 3 or frac.shape != depth.shape or not _same_grid(dh, fh):
        sys.exit("ERROR: -depth and -gmfrac must share a 3D grid")
    if not _same_grid(header, dh) or depth.shape != header["shape"]:
        sys.exit("ERROR: -depth must be on the -input grid")
    d = depth.numpy()
    if args.depth_order == "white_to_pial":
        d = 1 - d
    mask = (frac.numpy() >= args.gm_threshold) & np.isfinite(d) & (d >= 0) & (d <= 1)
    return mask, np.where(mask, d, np.nan)


DEFAULT_CV_GRID = (("s_d", (0.2, 0.4, 0.8)), ("t0v", (0.5, 1.0, 2.0)))


def _physiology_grid(spec: list[list[str]] | None) -> list[dict[str, float]]:
    """-cv_p0 NAME V ... (repeatable) -> every combination, as P0 overrides."""
    import itertools
    from dataclasses import fields

    from fastfuncstuff.laminar.params import P0

    names = {f.name for f in fields(P0) if f.type in ("float", float)}
    axes = []
    for item in spec or []:
        if len(item) < 2:
            sys.exit("ERROR: -cv_p0 needs a name and at least one value")
        if item[0] not in names:
            sys.exit(f"ERROR: -cv_p0 {item[0]}: not a P0 baseline ({', '.join(sorted(names))})")
        try:
            axes.append((item[0], tuple(float(v) for v in item[1:])))
        except ValueError:
            sys.exit(f"ERROR: -cv_p0 {item[0]}: values must be numbers")
    axes = axes or list(DEFAULT_CV_GRID)
    return [dict(zip([a for a, _ in axes], combo, strict=True))
            for combo in itertools.product(*[v for _, v in axes])]  # fmt: skip


def _design_builder(args, timing, files, n_t, tr, verbose):
    """Per-run (T, C) regressors for a CBF response curve, as ffs_hrftdv builds them."""
    import torch

    from fastfuncstuff.cli_utils import (
        apply_onset_rounding,
        microtime_offset_bins,
        resolve_microtime_dt,
        resolve_microtime_offset,
    )
    from fastfuncstuff.design.matrices import build_task_design

    dt = resolve_microtime_dt(tr, args.microtime_dt)
    try:
        offset = resolve_microtime_offset(args.microtime_offset, files, tr)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    onset_bin = microtime_offset_bins(offset, tr, dt)
    onsets = apply_onset_rounding(args, timing.all_onsets, tr, offset, verbose=verbose)
    durations = list(timing.durations)
    starts = [i * n_t for i in range(len(files))]

    def design_for(curve):
        x = build_task_design(
            curve.float()[None],
            n_t * len(files),
            starts,
            tr=tr,
            microtime_dt=dt,
            microtime_onset=onset_bin,
            event_onsets=onsets,
            durations=durations,
            device=torch.device("cpu"),
        )
        return list(x.split(n_t, 0))

    design_for.microtime_dt = dt  # ty: ignore[unresolved-attribute]
    return design_for


def _write_cv_table(path: str, grid: list[dict[str, float]], cv) -> list[dict]:
    best_per_patch = np.argmax(cv.patch_r2, axis=0)
    rows = []
    for i, (values, r2) in enumerate(zip(grid, cv.pooled_r2, strict=True)):
        rows.append({"candidate": i + 1, **values, "pooled_r2": float(r2),
                     "patch_share": float(np.mean(best_per_patch == i))})  # fmt: skip
    cols = list(rows[0])
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write(
                "\t".join(f"{row[c]:g}" if isinstance(row[c], float) else str(row[c]) for c in cols)
                + "\n"
            )
    return rows


def _same_grid(a: dict, b: dict) -> bool:
    return bool(np.allclose(a["affine"], b["affine"], atol=1e-4))


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args(argv)
    t0 = time.time()
    surfaces_given = bool(args.fs_subj or args.white_mesh or args.pial_mesh)
    volume_given = bool(args.depth or args.gmfrac)
    if surfaces_given == volume_given:
        sys.exit("ERROR: give surfaces (-fs_subj, or -white_mesh and -pial_mesh) or "
                 "-depth/-gmfrac/-depth_order, not both")  # fmt: skip
    if volume_given and not (args.depth and args.gmfrac and args.depth_order):
        sys.exit("ERROR: -depth needs -gmfrac and -depth_order")
    if args.fs_subj is None and surfaces_given and not (args.white_mesh and args.pial_mesh):
        sys.exit("ERROR: give -fs_subj, or -white_mesh and -pial_mesh")
    if args.depths < 2 or not 0 < args.gm_threshold <= 1:
        sys.exit("ERROR: need -depths >= 2 and 0 < -gm_threshold <= 1")
    if any(x <= 0 for x in args.ridge):
        sys.exit("ERROR: -ridge values must be positive")
    if any(x <= 0 for x in args.gain_lambdas) or args.extend_mm < 0:
        sys.exit("ERROR: -gain_lambdas must be positive and -extend_mm >= 0")
    if args.removal == "voxel" and len(args.input) < 2:
        sys.exit("ERROR: -removal voxel picks each voxel's gain by leave-one-run-out: "
                 "give >= 2 runs, or -removal column")  # fmt: skip
    if args.onsets and args.events:
        sys.exit("ERROR: give -onsets/-durations or -events, not both")
    if args.onsets and args.durations is None:
        sys.exit("ERROR: -durations is required with -onsets")
    timed = bool(args.onsets or args.events)
    if args.cv_p0 and not timed:
        sys.exit("ERROR: -cv_p0 scores physiologies from events: give -events or -onsets")
    if args.ridge_cv == "auto":
        args.ridge_cv = "loro" if args.repeated_runs else "split"
    if len(args.ridge) > 1 and args.ridge_cv == "loro" and not args.repeated_runs:
        sys.exit("ERROR: -ridge_cv loro predicts each run from the others, which needs runs "
                 "with the same task timing: add -repeated_runs, or use -ridge_cv split")  # fmt: skip
    if args.ridge_cv_mm <= 0 or args.ridge_cv_repeats < 1:
        sys.exit("ERROR: -ridge_cv_mm must be positive and -ridge_cv_repeats >= 1")

    import torch

    from fastfuncstuff.cli_utils import parse_input_files, parse_prefix, setup_device
    from fastfuncstuff.design.polort import max_polort, resolve_polort
    from fastfuncstuff.laminar.params import ModelSpec, zero_params
    from fastfuncstuff.laminar.pipeline import reference_p0
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.laminar_tdv import devein_volume_runs

    verbose = args.verb >= 1
    print("ffs_laminar_tdv is EXPERIMENTAL: model physiology, not validated on real data")
    device = setup_device(args.device)
    files = [str(f) for f in parse_input_files(args.input)]
    pfx = parse_prefix(args.prefix)
    ext = ".nii.gz" if pfx.nifti_ext.endswith(".gii") else pfx.nifti_ext
    Path(pfx.stem).parent.mkdir(parents=True, exist_ok=True)
    te, b0 = _acquisition(args, files[0])
    grid = _physiology_grid(args.cv_p0) if timed else []
    timing = None
    if timed:
        from fastfuncstuff.cli_utils import (
            bids_event_kwargs,
            check_bids_event_args,
            parse_timing_spec,
        )

        check_bids_event_args(args, has_events=bool(args.events))
        if len(files) < 2:
            sys.exit("ERROR: physiology cross-validation needs at least 2 runs")
        try:
            timing = parse_timing_spec(
                events=args.events,
                onsets=args.onsets,
                durations_arg=args.durations,
                n_runs=len(files),
                **bids_event_kwargs(args),
                input_files=files,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"ERROR: {exc}")

    # Units need only the grid, so each run is read once here and reduced to
    # the candidate voxels; the full series is read again only to write it.
    first, header = load_image(files[0])
    if first.ndim != 4:
        sys.exit(f"ERROR: {files[0]} is not a time series")
    header["shape"] = tuple(first.shape[1:])
    n_t = first.shape[0]
    tr = args.tr or float(header["header"].get_zooms()[3])
    gm_frac_vol = None
    if surfaces_given:
        from fastfuncstuff.processing.hrftdv import ribbon_patches
        from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

        try:
            surfaces = resolve_ribbon_surfaces(
                args.fs_subj,
                args.white,
                args.pial,
                args.white_mesh,
                args.pial_mesh,
                args.cortex_label,
                args.hemi,
                log=print if verbose else None,
            )
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        xyz = header["shape"][::-1]
        patches = ribbon_patches(
            surfaces, header["affine"], xyz, np.ones(xyz, bool), args.patch_mm, device,
            min_gm_fraction=args.ribbon_frac or None,
        )  # fmt: skip
        if patches.gm_frac is not None:
            gm_frac_vol = np.ascontiguousarray(patches.gm_frac.reshape(xyz).T)
        # Surface code is flat C-order over (X, Y, Z); load_image is (Z, Y, X).
        labels_vol = np.ascontiguousarray(patches.label.reshape(xyz).T)
        depth_vol = np.ascontiguousarray(1 - patches.depth.reshape(xyz).T)
        candidates = labels_vol >= 0
    else:
        candidates, depth_vol = _volume_units(args, header, load_image)
        labels_vol = None
    if args.columns:
        cols, ch = load_image(args.columns)
        if cols.shape != header["shape"] or not _same_grid(header, ch):
            sys.exit("ERROR: -columns must be on the -input grid")
        if not torch.isfinite(cols).all() or not torch.equal(cols, cols.round()):
            sys.exit("ERROR: -columns labels must be finite integers")
        labels_vol = cols.numpy().astype(np.int64) - 1
        candidates &= labels_vol >= 0

    from fastfuncstuff.processing.hrftdv import nearest_labelled

    spacing = np.sqrt(np.square(header["affine"][:3, :3]).sum(0))[::-1]  # (Z, Y, X)
    extend = args.extend_mm if args.removal == "voxel" else 0.0
    shell_c = (
        nearest_labelled(np.where(candidates, 0, -1), spacing, extend).reshape(candidates.shape)
        >= 0
    )
    cand_idx = np.flatnonzero(candidates | shell_c)
    series, seen, lengths = [], [], []
    for i, path in enumerate(files):
        d, h = (first, header) if i == 0 else load_image(path)
        if d.ndim != 4 or tuple(d.shape[1:]) != header["shape"] or not _same_grid(header, h):
            sys.exit(f"ERROR: {path}: every run must share the first run's grid")
        lengths.append(int(d.shape[0]))
        if args.tr is None and abs(float(h["header"].get_zooms()[3]) - tr) > 1e-5:
            sys.exit(f"ERROR: {path}: TR differs from the first run's")
        s = d.reshape(d.shape[0], -1)[:, cand_idx].T.contiguous()
        seen.append((torch.isfinite(s).all(1) & (s.amin(1) > 0)).numpy())
        series.append(s)
        del d
    del first
    each_ok = timing is None and len(args.ridge) == 1
    coverage = args.coverage if args.coverage != "auto" else ("each" if each_ok else "all")
    if coverage == "each" and not each_ok:
        sys.exit("ERROR: -coverage each solves every run on its own voxels, which needs one "
                 "-ridge value and no -events (both compare runs sample by sample)")  # fmt: skip
    covered = np.logical_or.reduce(seen) if coverage == "each" else np.logical_and.reduce(seen)
    if len(set(lengths)) > 1:
        # Each run's inverse stands alone at one ridge strength; only comparing runs
        # sample by sample (event CV, choosing between strengths) needs one length.
        if timing is not None:
            sys.exit(f"ERROR: runs differ in length ({', '.join(map(str, lengths))} volumes); "
                     "-events scores the physiology against one schedule across runs. "
                     "Drop -events (fixed physiology) or give runs of one length")  # fmt: skip
        if len(args.ridge) > 1:
            sys.exit(f"ERROR: runs differ in length ({', '.join(map(str, lengths))} volumes); "
                     "choosing between several -ridge values compares runs sample by sample. "
                     "Give one -ridge value")  # fmt: skip
    ok = np.zeros(candidates.size, bool)
    ok[cand_idx[covered]] = True
    ribbon = candidates & ok.reshape(candidates.shape)
    if not ribbon.any():
        sys.exit(
            f"ERROR: no ribbon voxels positive throughout {'any' if coverage == 'each' else 'every'} run"
        )
    lab_vol = np.full(candidates.shape, -1, np.int64)
    if labels_vol is None:
        from fastfuncstuff.processing.laminar_tdv import depth_patches

        lab_vol[ribbon] = depth_patches(ribbon, depth_vol, spacing, args.patch_mm)
    else:
        lab_vol[ribbon] = labels_vol[ribbon]
    # Shell voxels inherit the nearest covered ribbon voxel's column and depth.
    src = nearest_labelled(lab_vol, spacing, extend).reshape(candidates.shape)
    shell = shell_c & ok.reshape(candidates.shape) & (src >= 0)
    lab_vol[shell] = lab_vol.reshape(-1)[src[shell]]
    depth_vol = np.where(shell, depth_vol.reshape(-1)[np.maximum(src, 0)], depth_vol)
    mask = ribbon | shell
    rows = np.searchsorted(cand_idx, np.flatnonzero(mask))
    runs = [s[torch.from_numpy(rows)] for s in series]
    present = [p[rows] for p in seen]
    if all(p.all() for p in present):
        present = None
    elif verbose:
        n_all = int(np.logical_and.reduce(present).sum())
        print(f"  Coverage each: {mask.sum():,} voxels seen by some run, {n_all:,} by all "
              f"{len(runs)}; per run {', '.join(f'{int(p.sum()):,}' for p in present)}")  # fmt: skip
    del series
    depth, labels, in_columns = depth_vol[mask], lab_vol[mask], ribbon[mask]
    # Shell voxels never build columns, so only the ribbon's shares matter here.
    voxel_weight = None if gm_frac_vol is None else np.where(in_columns, gm_frac_vol[mask], 1.0)
    split_blocks = None
    if args.ridge_cv == "split" and len(args.ridge) > 1:
        from fastfuncstuff.processing.hrftdv import volume_tiles

        ijk = np.array(np.unravel_index(np.flatnonzero(mask), mask.shape)).T
        split_blocks = volume_tiles(ijk, spacing, args.ridge_cv_mm)

    polort = max_polort(resolve_polort(args.polort, lengths, tr, verbose=verbose))
    overrides = json.loads(Path(args.vascular_json).read_text()) if args.vascular_json else {}
    baselines = reference_p0()
    try:
        baselines = replace(baselines, **overrides.get("p0", {}))
    except TypeError as exc:
        sys.exit(f"ERROR: -vascular_json p0: {exc}")
    spec = ModelSpec(
        N=args.depths, K=args.depths, TR=tr, TE=te, B0=b0, p0=baselines, device=device,
        dtype=torch.float32,
    )  # fmt: skip
    params = zero_params(spec)
    for name, val in overrides.get("parameters", {}).items():
        if name not in params or params[name].ndim:
            sys.exit(f"ERROR: -vascular_json: {name} is not a scalar vascular parameter")
        params[name] = torch.as_tensor(val, device=device, dtype=spec.dtype)
    columns, cv = None, None
    if timing is not None:
        from fastfuncstuff.processing.laminar_tdv import column_profiles, physiology_crossval

        columns = column_profiles(
            runs, depth, labels, args.depths, polort, device, use=in_columns,
            robust=args.removal == "voxel",
        )  # fmt: skip
        design_for = _design_builder(args, timing, files, n_t, tr, verbose)
        candidates = [replace(baselines, **values) for values in grid]
        cv = physiology_crossval(
            columns, candidates, spec, design_for, params=params,
            microtime_dt=design_for.microtime_dt, verbose=verbose,
        )  # fmt: skip
        baselines = candidates[cv.best]
        spec = replace(spec, p0=baselines)
        rows = _write_cv_table(f"{pfx.stem}_physiology_cv.tsv", grid, cv)
        if verbose:
            print("  Physiology, leave-one-run-out from events (pooled held-out R²):")
            for row in rows:
                star = " <- used" if row["candidate"] == cv.best + 1 else ""
                vals = ", ".join(f"{k} {v:g}" for k, v in grid[row["candidate"] - 1].items())
                print(f"    {row['candidate']:3d}  {vals:28s} {row['pooled_r2']:.4f}  "
                      f"(best in {row['patch_share']:.0%} of patches){star}")  # fmt: skip
    if verbose:
        print(
            f"  {len(runs)} runs x {'/'.join(map(str, sorted(set(lengths))))} volumes, TR {tr:g} s, TE {te * 1e3:g} ms, {b0:g} T; "
            f"{int(in_columns.sum()):,} ribbon + {int((~in_columns).sum()):,} shell voxels in "
            f"{len(np.unique(labels)):,} patches; {args.removal} removal; "
            f"target {args.target}, {args.transfer_mode} transfer, "
            + ("events-chosen physiology" if cv is not None else "fixed physiology")
        )
    result = devein_volume_runs(
        runs,
        depth,
        labels,
        spec,
        params=params,
        target=args.target,
        transfer_mode=args.transfer_mode,
        strengths=tuple(args.ridge),
        pad_seconds=args.pad_seconds,
        cutoff_hz=args.cutoff_hz,
        polort=polort,
        columns=columns,
        removal=args.removal,
        in_columns=in_columns,
        gain_lambdas=tuple(args.gain_lambdas),
        ridge_cv=args.ridge_cv,
        split_blocks=split_blocks,
        split_repeats=args.ridge_cv_repeats,
        present=present,
        voxel_weight=voxel_weight,
        verbose=verbose,
    )
    del runs

    flat = np.flatnonzero(mask)
    for i, path in enumerate(files):
        d, h = load_image(path)
        d = d.contiguous()  # so the flat view below writes into d
        d.view(lengths[i], -1)[:, flat] = result.corrected[i].T
        result.corrected[i] = torch.empty(0)
        save_image(d, f"{pfx.stem}_corrected_run-{i + 1:02d}{ext}", h)
        del d
        outs = [("removed_psc", result.removed_psc)]
        if args.save_fit:
            outs.append(("fit_psc", result.smooth_psc))
        for suffix, arrs in outs:
            out = torch.zeros(lengths[i], *header["shape"])
            out.view(lengths[i], -1)[:, flat] = arrs[i].T
            save_image(out, f"{pfx.stem}_{suffix}_run-{i + 1:02d}{ext}", h)
            del out
    removed_rms = torch.cat(result.removed_psc, -1).square().mean(-1).sqrt().numpy()
    bricks = {
        "ridge": result.regularization,
        "ge_cv_r2": result.cv_r2,
        "noise_gain": result.noise_gain,
        "removed_rms": removed_rms,
        "valid": result.valid,
        "shell": ~in_columns,
        "patch": labels + 1,
        "depth": 1 - depth,  # 0 white .. 1 pial, as ffs_hrftdv writes it
        **({} if voxel_weight is None else {"gm_frac": np.where(in_columns, voxel_weight, 0.0)}),
    }
    if result.gains is not None:
        for i, term in enumerate(("local", "transport", "shift")):
            bricks[f"gain_{term}"] = result.gains.beta[:, i]
        bricks["gain_lambda"] = result.gains.lam
        bricks["gain_cv"] = result.gains.cv_gain
    if result.split_leak is not None:
        bricks["split_leak"] = result.split_leak
    if cv is not None and columns is not None:
        patch_r2, patch_best = np.zeros(len(depth), np.float32), np.zeros(len(depth), np.float32)
        r2 = cv.patch_r2
        for u, ix in enumerate(columns.members):
            patch_r2[ix] = r2[cv.best, u]
            patch_best[ix] = int(np.argmax(r2[:, u])) + 1
        bricks["phys_cv_r2"], bricks["phys_best"] = patch_r2, patch_best
    bucket = torch.zeros(len(bricks), *header["shape"])
    for i, values in enumerate(bricks.values()):
        bucket[i][torch.from_numpy(mask)] = torch.as_tensor(
            np.nan_to_num(np.asarray(values, np.float32))
        )
    save_image(bucket, f"{pfx.stem}_ltdv{ext}", header, brick_labels=list(bricks))
    meta = {
        "tool": "ffs_laminar_tdv",
        "experimental": True,
        **{k: v for k, v in vars(args).items() if k != "verb"},
        "input_files": files,
        "actual_tr": tr,
        "te": te,
        "b0": b0,
        "polort": polort,
        "units": "surfaces" if surfaces_given else "volume depth",
        "p0": asdict(baselines),
        "n_valid": int(result.valid.sum()),
        "n_ribbon": int(mask.sum()),
        "n_patches": int(len(np.unique(labels))),
        "depth_convention": "ltdv depth brick: 0 = white, 1 = pial",
        "calibration": (
            "physiology chosen from events by pooled leave-one-run-out design prediction"
            if cv is not None
            else "fixed physiology"
        ),
        "physiology_cv": None
        if cv is None
        else [
            {**values, "pooled_r2": float(r2)}
            for values, r2 in zip(grid, cv.pooled_r2, strict=True)
        ],
    }
    Path(f"{pfx.stem}_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    if verbose:
        v = result.valid
        print(f"\n  Corrected {int(v.sum()):,}/{int(mask.sum()):,} ribbon + shell voxels "
              f"({int(mask.sum() - v.sum()):,} in patches missing a depth: unchanged)")  # fmt: skip
        if v.any():
            if len(args.ridge) > 1:
                chosen = result.regularization[v]
                print("  Ridge chosen: " + ", ".join(
                    f"{r:g} {np.mean(chosen == np.float32(r)):.0%}" for r in args.ridge))  # fmt: skip
                print(f"  Held-out R² of the model (median): {np.nanmedian(result.cv_r2[v]):.3f}")
            print(f"  Noise gain (median): {np.median(result.noise_gain[v]):.3f}")
            g = result.gains
            print("  Removed RMS (% signal) by depth, 0 = white, 1 = pial"
                  + (";  transport gain median / 95th pct" if g is not None else "") + ":")  # fmt: skip
            wp = 1 - depth
            groups = [(f"{a:.1f}-{a + 0.2:.1f}", in_columns & (wp >= a) & (wp < a + 0.2 + 1e-9))
                      for a in np.linspace(0, 0.8, 5)] + [("shell", ~in_columns)]  # fmt: skip
            for name, k in groups:
                k = v & k
                if not k.any():
                    continue
                line = f"    {name:7s}  n={int(k.sum()):7d}  median {np.median(removed_rms[k]):.3f}"
                if g is not None:
                    gt = g.beta[k, 1]
                    line += f"   gain {np.median(gt):5.2f} / {np.percentile(gt, 95):5.2f}"
                print(line)
            if g is not None:
                free = g.lam[v] < max(args.gain_lambdas)
                print(f"  Own gain beat the column model (held out) in {free.mean():.0%} of "
                      f"voxels; median held-out R² gained there "
                      f"{np.median(g.cv_gain[v][free]) if free.any() else 0:.3f}")  # fmt: skip
        print(f"  Wrote {pfx.stem}_corrected_run-NN{ext}, _removed_psc_run-NN, _ltdv{ext} "
              f"({time.time() - t0:.0f} s)")  # fmt: skip


if __name__ == "__main__":
    main()
