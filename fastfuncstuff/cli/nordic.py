#!/usr/bin/env python3
"""CLI for NORDIC-style denoising (ffs_nordic)."""

from __future__ import annotations

import argparse
import json
import sys
import time

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
from fastfuncstuff.denoise.nordic import NordicConfig, run_nordic, run_nordic_multiecho
from fastfuncstuff.denoise.nordic_sweep import run_nordic_factor_sweep


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_nordic",
        description="NORDIC-style denoising for magnitude-only or complex (magnitude+phase) fMRI data.",
        formatter_class=FfsHelpFormatter,
        epilog="""
Examples:
  # Closest to MATLAB call
  ffs_nordic -input-magn sub-08_bold.nii.gz \
             -input-phase sub-08_phase.nii.gz \
             -prefix NORDIC_sub-08_bold \
             -temporal-phase 1 \
             -phase-filter-width 10 \
             -noise-volume-last 3 \
             -nordic

  # Magnitude-only mode
  ffs_nordic -input-magn sub-08_bold.nii.gz \
             -prefix NORDIC_sub-08_bold_magonly \
             -magnitude-only
""",
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-input-magn",
        "-input_magn",
        nargs="+",
        required=True,
        help="Input magnitude NIfTI file(s). Pass one per echo (>=2) to enable "
        "the multi-echo cross-echo signal-rescue path.",
    )
    io_group.add_argument(
        "-input-phase",
        "-input_phase",
        nargs="+",
        default=None,
        help="Input phase NIfTI file(s); one per magnitude file. Required unless -magnitude-only.",
    )
    io_group.add_argument("-prefix", required=True, help="Output prefix")
    io_group.add_argument(
        "-make-complex-nii",
        action="store_true",
        help="Write separate magnitude and phase outputs",
    )
    io_group.add_argument(
        "-save-gfactor-map",
        action="store_true",
        help="Save estimated g-factor proxy map",
    )
    io_group.add_argument(
        "-save-residual-map",
        action="store_true",
        help="Save denoising residual map (magnitude of complex difference)",
    )
    io_group.add_argument(
        "-save-num-comps",
        "-save_num_comps",
        action="store_true",
        help="Save per-voxel count of components removed (patch-averaged, fractional)",
    )
    io_group.add_argument(
        "-add-mean",
        "-add_mean",
        action="store_true",
        help="Add the raw magnitude's per-voxel temporal mean onto the saved residual "
        "(mean + removed noise), so it survives downstream resampling like real data. "
        "Requires -save-residual-map.",
    )
    io_group.add_argument(
        "-no-resid-qc",
        "-no_resid_qc",
        dest="resid_qc",
        action="store_false",
        help="Disable the multi-echo cross-echo residual correlation QC maps (on by default)",
    )

    algo_group = parser.add_argument_group("Algorithm")
    algo_group.add_argument(
        "-temporal-phase",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Temporal phase correction mode",
    )
    algo_group.add_argument(
        "-phase-filter-width",
        type=float,
        default=10.0,
        help="Phase low-pass strength",
    )
    algo_group.add_argument(
        "-noise-volume-last",
        type=int,
        default=0,
        help="Number of trailing volumes used as noise-only",
    )
    algo_group.add_argument(
        "-factor-error",
        type=float,
        default=1.0,
        help="NORDIC threshold scaling (>1 higher floor, <1 lower floor)",
    )
    algo_group.add_argument(
        "-retain-dof",
        "-retain_dof",
        type=float,
        default=None,
        metavar="N|FRAC",
        help="Cap denoising to preserve degrees of freedom: keep at least this many "
        "components per patch (remove at most K-N). Integer (>=1) = absolute min kept; "
        "float in (0,1) = fraction of timepoints kept. Bounds the numcomps map so the "
        "GLM keeps >= N residual DoF and stats stay valid without a post-hoc adjustment.",
    )
    algo_group.add_argument(
        "-nordic",
        action="store_true",
        help="Use NORDIC thresholding (default if MP not selected)",
    )
    algo_group.add_argument(
        "-mp",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="MP mode (1/2 enables MP-PCA style thresholding)",
    )
    algo_group.add_argument(
        "-magnitude-only",
        action="store_true",
        help="Ignore phase input and denoise magnitude as complex-with-zero-phase",
    )
    algo_group.add_argument(
        "-per-echo-gfactor",
        "-per_echo_gfactor",
        action="store_true",
        help=(
            "Multi-echo: estimate each echo's own g-factor (and thermal sigma) "
            "instead of sharing echo 1's. Use when thermal sigma is not "
            "TE-invariant; costs one g-factor pass per echo (default: share echo 1)"
        ),
    )
    algo_group.add_argument(
        "-kernel-size-pca",
        nargs=3,
        type=int,
        default=None,
        metavar=("KX", "KY", "KZ"),
        help="Patch size for main LLR denoising",
    )
    algo_group.add_argument(
        "-kernel-size-gfactor",
        nargs=3,
        type=int,
        default=[14, 14, 1],
        metavar=("KX", "KY", "KZ"),
        help="Patch size for g-factor proxy estimation",
    )
    algo_group.add_argument(
        "-gfactor-nvols",
        type=int,
        default=90,
        help="Number of volumes for g-factor proxy estimation",
    )
    algo_group.add_argument(
        "-patch-overlap",
        type=int,
        default=2,
        help="Patch overlap divisor for main pass (MATLAB default: 2)",
    )
    algo_group.add_argument(
        "-gfactor-patch-overlap",
        type=int,
        default=2,
        help="Patch overlap divisor for g-factor pass",
    )
    algo_group.add_argument(
        "-use-magn-for-gfactor",
        action="store_true",
        help="Estimate g-factor from magnitude-only data (MATLAB use_magn_for_gfactor)",
    )
    algo_group.add_argument(
        "-phase-slice-average",
        action="store_true",
        help="Enable mean-phase removal per slice (MATLAB phase_slice_average_for_kspace_centering=1)",
    )

    me_group = parser.add_argument_group("Multi-echo rescue (>=2 echoes)")
    me_group.add_argument(
        "-no-rescue",
        "-no_rescue",
        dest="rescue",
        action="store_false",
        help="Disable the cross-echo signal-rescue guard (denoise each echo "
        "independently). Default: rescue on for multi-echo input.",
    )
    me_group.add_argument(
        "-rescue-band",
        "-rescue_band",
        type=float,
        default=0.25,
        help="Fraction of each echo's kill set (top singular values, nearest the "
        "threshold) tested for rescue. Larger = more components considered.",
    )
    me_group.add_argument(
        "-rescue-alpha",
        "-rescue_alpha",
        type=float,
        default=0.05,
        help="Per-patch false-rescue rate. The rescue threshold is the "
        "(1 - alpha) quantile of the all-thermal-noise null.",
    )

    sweep_group = parser.add_argument_group("Factor-sweep diagnostic (single-echo)")
    sweep_group.add_argument(
        "-factor-sweep",
        "-factor_sweep",
        action="store_true",
        help="Sweep the threshold factor and emit residual voxel-to-voxel correlation "
        "diagnostic plots/table (single-echo, NORDIC threshold). Off by default.",
    )
    sweep_group.add_argument(
        "-factor-sweep-range",
        "-factor_sweep_range",
        nargs=3,
        type=float,
        default=None,
        metavar=("LO", "HI", "N"),
        help="Factor window as LO HI N (e.g. 0.5 2.0 9). Default: 9 even steps over "
        "0.75-1.25 (lands on 1.0). Overridden by -factor-sweep-values.",
    )
    sweep_group.add_argument(
        "-factor-sweep-values",
        "-factor_sweep_values",
        nargs="+",
        type=float,
        default=None,
        help="Explicit factor values to sweep (overrides -factor-sweep-range).",
    )
    sweep_group.add_argument(
        "-factor-sweep-max-voxels",
        "-factor_sweep_max_voxels",
        type=int,
        default=40000,
        help="Per-mask voxel cap for the correlation (random subsample). 0 = all voxels.",
    )
    sweep_group.add_argument(
        "-save-imgs",
        "-save_imgs",
        action="store_true",
        help="Save the generated automask and the top_pairs voxel mask as NIfTIs.",
    )
    sweep_group.add_argument(
        "-save-factor-img",
        "-save_factor_img",
        action="store_true",
        help="Save a 4D patch-averaged #components-removed image; sub-bricks step the "
        "factor 0.1..5.0 (shows how the keep/kill floor moves with factor).",
    )
    sweep_group.add_argument(
        "-save-eigen-img",
        "-save_eigen_img",
        action="store_true",
        help="Save a 4D patch-averaged singular-value spectrum image (sub-bricks = "
        "component rank); shows where factor*lambda lands on each patch's spectrum.",
    )

    task_group = parser.add_argument_group("Task-leak diagnostic (-events)")
    task_group.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv for this run — switches on the task-leak diagnostic.\n"
        "NORDIC decides what to discard from a patch's singular-value spectrum alone; "
        "nothing in that decision knows the task, so a component carrying a real BOLD "
        "response can fall under the threshold. This measures how much of the design "
        "survives in what was thrown away, and — the part that separates 'we removed "
        "some task variance' from 'we removed noise that happens to fit' — whether it "
        "lands on the tissue that actually responds.\n"
        "The series it scores is |input| - |denoised| (see -save-task-loss), not the "
        "modulus of the removed complex field.\n"
        "The headline is an ENRICHMENT: the share of the loss series' task-locked "
        "energy that lands inside the responding mask, over the share of voxels that "
        "mask occupies. Its no-relation value is 1.0 by construction, so it needs no "
        "surrogate — which matters, because a block design admits none. The mask is "
        "placed on one half of the run and the energy scored on the other; a mask drawn "
        "from the whole run shares noise with the series it scores and reads ~3.9x on "
        "data with no task in it at all.\n"
        "Writes {prefix}_taskfit_input, _taskfit_kept and _taskfit_lost: one AFNI "
        "bucket each, full_model_R then {cond}_Coef (percent signal change) and "
        "{cond}_Correl, from one joint fit over the whole design — read them the way "
        "you read 3dDeconvolve's. Plus {prefix}_taskleak.txt with the report.\n"
        "Diagnostic only: the denoised output is not changed. -retain-dof is the knob "
        "if the answer is bad. NOTE the input is not motion-corrected at this stage, so "
        "the per-voxel fits are worse than a preprocessed run's — read the enrichment, "
        "not the R maps.",
    )
    task_group.add_argument(
        "-event_ignore",
        "-event-ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude from the task design (e.g. fixation).",
    )
    task_group.add_argument(
        "-event_cols",
        "-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom -events column names. Default: onset duration trial_type.",
    )
    task_group.add_argument(
        "-tr",
        type=float,
        default=None,
        metavar="SEC",
        help="Repetition time for the task design. Default: the input header pixdim[4].",
    )
    task_group.add_argument(
        "-task_polort",
        "-task-polort",
        type=int,
        default=None,
        metavar="DEG",
        help="Legendre drift degree removed from both series AND the design before the "
        "fit. Default: AFNI's 1 + floor(run_seconds/150).",
    )
    task_group.add_argument(
        "-task_top_frac",
        "-task-top-frac",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of voxels, ranked by the DENOISED data's own full-model R, that "
        "count as 'where the task is' — the stratum the report headlines. Default 0.1.",
    )
    task_group.add_argument(
        "-task_thresh",
        "-task-thresh",
        type=float,
        default=None,
        metavar="R",
        help="Absolute cut on the denoised data's full-model R that defines 'where the "
        "task is'. Default: use -task_top_frac instead.",
    )
    task_group.add_argument(
        "-task_fdr_q",
        "-task-fdr-q",
        type=float,
        default=0.05,
        metavar="Q",
        help="Benjamini-Hochberg q for the per-patch design-overlap test. Each patch is "
        "asked whether the directions it DISCARDED carry more of the design than a "
        "uniformly random subspace of the same dimension would - a null that is exact "
        "under 'this patch holds only thermal noise' (a white Wishart's eigenvectors "
        "are Haar and independent of its eigenvalues) and that never touches the "
        "design, which is what makes it usable where a shift or phase-randomised "
        "surrogate is not. The patch, not the voxel, is the test unit: every voxel in a "
        "patch is reconstructed through the same discarded subspace, so a voxelwise FDR "
        "would count each patch once per voxel. Writes {prefix}_taskpatch (excess "
        "overlap, and 1-q). Default 0.05.",
    )
    task_group.add_argument(
        "-save-task-loss",
        "-save_task_loss",
        action="store_true",
        dest="save_task_loss",
        help="Write {prefix}_taskloss: the signed magnitude-domain loss series, "
        "|input| - |denoised| frame by frame. This is what every GLM downstream of "
        "NORDIC actually loses, and it keeps its sign — unlike -save-residual-map, "
        "which is the modulus of the removed complex field and so is half-rectified "
        "and offset by the Rician mean of the thermal noise it mostly consists of. "
        "Implied by -events; on its own it is the series to hand to an ICA of what "
        "was discarded.",
    )

    perf_group = parser.add_argument_group("Performance")
    add_device_arg(
        perf_group,
        extra="MPS supports Gram-eigh patches; direct complex SVD falls back to CPU.",
    )
    perf_group.add_argument(
        "-svd-batch-size",
        type=int,
        default=512,
        help="Number of patches per batched SVD call (tune for GPU memory vs speed)",
    )
    perf_group.add_argument(
        "-decomp-method",
        choices=["auto", "svd", "eigh"],
        default="auto",
        help="Decomposition method: auto (eigh when M/N>=2), svd, or eigh",
    )
    add_verbose_arg(perf_group, default=1)

    return parser.parse_args(argv)


def _resolve_task_design(args, magnitude_file: str):
    """Convolved design + drift degree for the per-patch test, built before the run.

    The patch test lives inside the LLR loop -- it needs the discarded singular vectors,
    which exist only there -- so unlike the -events fits this cannot be done afterwards
    from the saved files. Header read only; the design is trimmed to the denoised frame
    count inside the library.
    """
    from fastfuncstuff.cli.task_events import task_design_from_events
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.stats.task_coupling import default_polort

    hdr = load_nifti(magnitude_file).header
    n_t = int(hdr.get_data_shape()[3])
    tr = args.tr if args.tr is not None else float(hdr["pixdim"][4])
    if tr <= 0:
        raise ValueError(
            f"the input header gives TR={tr:g}s, which cannot build a design - pass -tr SEC."
        )
    design, labels = task_design_from_events(args, n_t, tr, torch.device("cpu"))
    polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
    return design.cpu().numpy(), labels, polort, tr


def _print_patch_test(summary: dict, label: str) -> list[str]:
    """Per-patch verdict: was the discarded subspace aligned with the design?

    The z leads, not the FDR count, because that is where the dynamic range is.
    NORDIC routinely keeps 3 of 120 components, so "is the design in what was thrown
    away" is trivially yes for the design AND for every random direction, and the
    upper-tail test has almost nothing to separate. The SIGNED distance from the null
    still separates cleanly: measured across the synthetic runs, a task NORDIC
    preserved reads a median z of -17, no task at all reads +0.3, and a task too weak
    to survive the threshold reads +0.6 to +0.9.
    """
    n_in = summary["n_patches_in_brain"]
    n_sig = summary["n_patches_significant"]
    z = summary["z_median_in_brain"]
    if z < -2.0:
        verdict = "the response was preferentially KEPT"
    elif z > 2.0:
        verdict = "removal was selectively aligned with the design"
    else:
        verdict = "the design went out with the noise like any other direction"
    return [
        f"  PATCH TEST{label}: median z {z:+.2f} over {n_in} in-brain patches -- {verdict}",
        "    z is how far the design sits inside what each patch DISCARDED, against "
        "random directions",
        "    drawn in its own drift-orthogonal subspace. Negative = preferentially "
        "kept; ~0 = treated",
        "    like any other direction; positive = discarded selectively.",
        f"    {n_sig} of {n_in} patches positive at q <= {summary['fdr_q']:g} "
        f"(BH-FDR, one-sided upper tail)",
        f"    null: {summary['n_null_frames']} random "
        f"{summary['n_design_columns']}-column frames; patches discarded "
        f"{summary['mean_removed_dim']:.1f} of {summary['n_components']} components "
        f"on average",
    ]


def _split_half_enrichment(source, lost, design, mask, tr, args, labels):
    """Enrichment with the mask chosen on frames the score never sees.

    The whole difficulty of this diagnostic is that every candidate mask shares noise
    with the series being scored. ``input = kept + lost``, so ranking voxels by the
    input's task fit lets the loss series help pick the mask it is then scored in;
    measured on a run with NO task planted at all that alone reads 3.9x, which is not a
    bias to caveat but a number that would be read as a leak. Ranking on the kept series
    has the mirror flaw with the opposite sign, and a block design admits no surrogate
    to calibrate either — a circular shift only negates it, and a phase-randomised one
    sits at the design's own frequency (both on the record in
    :mod:`fastfuncstuff.stats.task_coupling`).

    Splitting the run in time removes the shared noise instead of correcting for it.
    The mask comes from one contiguous half's input fit, the task energy is measured on
    the OTHER half of the loss series, and the two halves swap roles so neither is
    privileged. Real response is in both halves, so a real leak still concentrates;
    thermal noise is not, so the selection cannot manufacture one. Contiguous halves
    rather than odd/even frames: interleaving would leave each half's noise correlated
    with the other's through the autocorrelation the split is there to break.

    Returns ``(mean_enrichment, per_half)`` — or ``(None, [])`` when neither half can
    place a mask, which happens when nothing in the run responds.
    """
    import torch

    from fastfuncstuff.stats.task_coupling import (
        default_polort,
        map_enrichment,
        responding_mask,
        task_coupling,
    )

    n_t = source.shape[3]
    half = n_t // 2
    halves = [slice(0, half), slice(half, n_t)]
    out = []
    for rank_i, score_i in ((0, 1), (1, 0)):
        rank_sl, score_sl = halves[rank_i], halves[score_i]
        d_rank = torch.as_tensor(design)[rank_sl]
        d_score = torch.as_tensor(design)[score_sl]
        try:
            po_r = args.task_polort or default_polort(int(d_rank.shape[0]), tr)
            po_s = args.task_polort or default_polort(int(d_score.shape[0]), tr)
            rank_tc = task_coupling(
                source[..., rank_sl], d_rank, polort=po_r, mask=mask, labels=labels
            )
            resp, _, _ = responding_mask(
                rank_tc.r_full, mask, args.task_top_frac, thresh=args.task_thresh
            )
            score_tc = task_coupling(
                lost[..., score_sl], d_score, polort=po_s, mask=mask, labels=labels
            )
        except ValueError:
            # A half with too few blocks to separate the design from its own drift, or
            # with no voxel responding in it, contributes nothing rather than a number.
            continue
        out.append(map_enrichment(score_tc.task_rms, resp, mask)["enrichment"])
    if not out:
        return None, []
    return sum(out) / len(out), out


def _task_leak_report(outputs, magnitude_file: str, args, label: str) -> None:
    """Is the task in what NORDIC threw away, and is it where the task lives?

    Three fits of one design — the input, the series that was kept, and the signed
    magnitude loss — all in the same magnitude units, so their task-explained rms is
    directly comparable and the ratios are the numbers this exists to produce.

    A ratio alone is not evidence. A K-column design projected onto ANY series keeps
    ``sqrt(K/df)`` of its norm, so a pure-noise loss series still reports a non-zero
    task rms; ``TaskCoupling.chance_share`` is that floor. What carries the argument is
    the enrichment: the share of the loss series' task-locked energy that falls inside
    the responding mask, over the share of voxels that mask occupies. Its no-relation
    value is 1.0 by construction, which is what makes it readable on data this stage
    has not yet motion-corrected.

    **Which voxels count as responding is biased whichever series ranks them, so both
    bounds are reported.** ``input = kept + lost``, so ranking on the input lets the
    loss series help choose the mask it is then scored in, and a run whose own task fit
    is weak inflates the enrichment toward the noise in ``lost``. Ranking on the kept
    series has the mirror flaw and is worse: a leak shows up precisely in the voxels
    where the response did NOT survive, so selecting on what survived selects against
    the effect being looked for -- and when the threshold strips a patch outright there
    is no surviving response left to rank at all.

    There is no surrogate to calibrate this with. A circular-shift null is invalid for
    a block design outright (a P/2 shift only negates it, and task rms is a magnitude),
    and a phase-randomized one sits at the design's own frequency; both are on the
    record in :mod:`fastfuncstuff.stats.task_coupling`. So the two masks are reported
    as a bracket around the true value. Where they agree the answer is solid. Where
    they disagree widely the input's task fit is too weak for any mask drawn from it
    to decide, and the unsupervised route -- an ICA of the loss series -- is the one
    to take.
    """
    from pathlib import Path

    from fastfuncstuff.cli.task_events import psc_betas, save_task_fit, task_design_from_events
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.stats.task_coupling import (
        default_polort,
        responding_mask,
        task_coupling,
    )

    if outputs.task_loss_file is None:
        raise RuntimeError("the task-leak diagnostic needs the loss series but none was written")

    src = load_nifti(magnitude_file)
    tr = args.tr if args.tr is not None else float(src.header["pixdim"][4])
    if tr <= 0:
        raise ValueError(
            f"the input header gives TR={tr:g}s, which cannot build a design — pass -tr SEC."
        )

    kept_img = load_nifti(str(outputs.magnitude_file))
    kept = torch.as_tensor(np.ascontiguousarray(kept_img.get_fdata(dtype=np.float32)))
    lost = torch.as_tensor(
        np.ascontiguousarray(load_nifti(str(outputs.task_loss_file)).get_fdata(dtype=np.float32))
    )
    affine = kept_img.affine
    n_t = kept.shape[3]

    design, labels = task_design_from_events(args, n_t, tr, torch.device("cpu"))
    polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
    # Before the two fits, not after: a wrong -tr shows up here as a design that never
    # lines up with the run, and the numbers below are uninterpretable until it is right.
    print(
        f"  Design{label}: {len(labels)} condition(s) ({', '.join(labels)}), "
        f"TR {tr:g}s, polort {polort}, {n_t} frames"
    )

    # kept + lost IS the input magnitude, by construction of the loss series -- cheaper
    # and exactly consistent with re-reading the input file, which may still carry the
    # trailing noise volumes NORDIC trimmed. Three float32 volumes is the peak here,
    # and this runs after the GPU work is done.
    source = kept + lost
    reference = source.mean(dim=3)
    mask = reference > 0.1 * float(reference.max())
    kwargs = dict(polort=polort, mask=mask, labels=labels)
    src_tc = task_coupling(source, design, **kwargs)
    kept_tc = task_coupling(kept, design, **kwargs)
    lost_tc = task_coupling(lost, design, **kwargs)

    # "Where does the data respond" is a whole-model question, so it is ranked on
    # r_full, and on the INPUT -- see the docstring for why not on what survived.
    resp, _quiet, cut = responding_mask(
        src_tc.r_full, mask, args.task_top_frac, thresh=args.task_thresh
    )
    # The headline. Reported instead of a whole-run enrichment, not alongside one: see
    # _split_half_enrichment for why every mask drawn from the whole run is biased by
    # noise it shares with the series it scores.
    enrich, per_half = _split_half_enrichment(source, lost, design, mask, tr, args, labels)
    del source
    src_sum = src_tc.summarize(resp)
    kept_sum, lost_sum = kept_tc.summarize(resp), lost_tc.summarize(resp)
    # Two ratios, answering different questions. The amplitude one is in magnitude units
    # on both sides, so it says how big what left is next to what stayed. The share is
    # the lost series' task-explained rms over its OWN rms, which is the only form
    # comparable to chance_share -- that is a fraction of norm, not an amplitude, and
    # printing it beside a task rms invites exactly the wrong reading.
    amp_ratio = lost_sum["task_rms_median"] / max(1e-12, src_sum["task_rms_median"])
    kept_ratio = kept_sum["task_rms_median"] / max(1e-12, src_sum["task_rms_median"])
    lost_share = lost_sum["task_rms_median"] / max(1e-12, lost_sum["total_rms_median"])

    stem, _, ext = str(outputs.task_loss_file).partition("_taskloss")
    for name, tc in (("input", src_tc), ("kept", kept_tc), ("lost", lost_tc)):
        save_task_fit(
            f"{stem}_taskfit_{name}{ext}",
            tc,
            psc_betas(tc, design, reference, mask),
            labels,
            affine,
            polort + 1,
            n_t,
        )

    how = "cut" if args.task_thresh is not None else f"top {args.task_top_frac * 100:.0f}%"
    lines = [
        f"  summary mask : {int(resp.sum())} voxels, whole-run input R > {cut:.3f} "
        f"({how}) = {100 * int(resp.sum()) / max(1, int((mask > 0).sum())):.1f}% of the brain",
        "                 descriptive only -- the three lines below are selected on it "
        "and read high",
        "                 because of that. The enrichment is not.",
        f"  input: full-model R {src_sum['r_full_median']:.3f} med, task rms "
        f"{src_sum['task_rms_median']:.4g}",
        f"  kept : full-model R {kept_sum['r_full_median']:.3f} med, task rms "
        f"{kept_sum['task_rms_median']:.4g} = {100 * kept_ratio:.1f}% of the input's",
        f"  lost : full-model R {lost_sum['r_full_median']:.3f} med, task rms "
        f"{lost_sum['task_rms_median']:.4g} = {100 * amp_ratio:.1f}% of the input's",
        f"  lost : task-explained share of its OWN rms {lost_share:.3f}, against "
        f"{lost_tc.chance_share:.3f} that any {len(labels)}-column design keeps from "
        f"pure noise",
        "",
        "  ENRICHMENT of the lost series' task energy on responding tissue",
        (
            f"    {enrich:.2f}x   mask from one half of the run, energy scored on the "
            f"other  ({' and '.join(f'{e:.2f}x' for e in per_half)})"
            if enrich is not None
            else "    n/a    neither half of the run places a mask: nothing in it "
            "responds to the design"
        ),
        "",
        "  1.0x means what was removed is spread like the brain and has nothing to do with where",
        "  the task is - the reassuring answer, and one a share above chance does NOT contradict,",
        "  because a design projected onto pure noise keeps some of it everywhere. Well above 1.0x",
        "  means real response left with the noise; -retain-dof caps how much any patch "
        "may remove.",
        "  The mask is chosen on frames the score never sees, so the 1.0x floor is real and not an",
        "  artefact of selection - a whole-run mask reads ~3.9x on data with no task in it at all.",
        "  Power is the cost: half a run's blocks place the mask, so a weak leak attenuates toward",
        "  1.0x rather than showing up. An ICA of the loss series is the unsupervised next step.",
    ]
    print("\n".join(lines))
    header = (
        f"Task-leak diagnostic{label}\n"
        f"  design         : {len(labels)} condition(s) ({', '.join(labels)}), "
        f"TR {tr:g}s, polort {polort}, {n_t} frames"
    )
    Path(f"{stem}_taskleak.txt").write_text(
        header + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    magn_files: list[str] = list(args.input_magn)
    phase_files: list[str] | None = list(args.input_phase) if args.input_phase is not None else None
    n_echoes = len(magn_files)

    if not args.magnitude_only and phase_files is None:
        print("ERROR: -input-phase is required unless -magnitude-only is set")
        sys.exit(1)
    if phase_files is not None and len(phase_files) != n_echoes:
        print(
            f"ERROR: got {n_echoes} magnitude file(s) but {len(phase_files)} phase file(s) "
            "(need one phase per echo)"
        )
        sys.exit(1)
    if args.add_mean and not args.save_residual_map:
        print("ERROR: -add-mean only affects the residual map; also pass -save-residual-map")
        sys.exit(1)
    if args.retain_dof is not None and args.retain_dof <= 0:
        print("ERROR: -retain-dof must be positive (integer components or a fraction in (0,1))")
        sys.exit(1)
    # The diagnostic scores the loss series, so -events has to produce it.
    save_task_loss = args.save_task_loss or args.events is not None
    task_design = None
    if args.events is not None:
        # Built up front, before any denoising: the per-patch test runs INSIDE the LLR
        # loop (it needs the discarded singular vectors), so a design resolved
        # afterwards would be too late, and a bad -tr should fail in a second rather
        # than after the whole run.
        task_design, task_labels, task_polort, task_tr = _resolve_task_design(args, magn_files[0])
        print(
            f"Task design: {len(task_labels)} condition(s) ({', '.join(task_labels)}), "
            f"TR {task_tr:g}s, polort {task_polort}"
        )

    # Resolve the factor-sweep window: explicit values win, else LO HI N range.
    sweep_values: tuple[float, ...] | None = None
    if args.factor_sweep_values is not None:
        sweep_values = tuple(args.factor_sweep_values)
    elif args.factor_sweep_range is not None:
        lo, hi, n = args.factor_sweep_range
        sweep_values = tuple(float(x) for x in np.linspace(lo, hi, int(n)))
    sweep_max_voxels = (
        None if args.factor_sweep_max_voxels in (0, None) else args.factor_sweep_max_voxels
    )

    # The sweep path also handles the cache-only diagnostic images.
    run_sweep_path = (
        args.factor_sweep or args.save_imgs or args.save_factor_img or args.save_eigen_img
    )
    if run_sweep_path and args.events is not None:
        print(
            "ERROR: -events is a diagnostic on the denoised run; it does not apply to "
            "the -factor-sweep path (which never writes one)."
        )
        sys.exit(1)
    if run_sweep_path and n_echoes > 1:
        print("ERROR: -factor-sweep / -save-*-img are single-echo only (pass one -input-magn).")
        sys.exit(1)

    prefix_info = parse_prefix(args.prefix)
    prefix = prefix_info.stem

    device_spec = args.device
    if (
        device_spec is None or str(device_spec).lower() == "auto"
    ) and not torch.cuda.is_available():
        device_spec = "cpu"
    device = setup_device(device_spec)

    print_cli_header("ffs_nordic", "NORDIC-style denoising")
    print(f"Input magnitude: {magn_files}")
    print(f"Input phase: {phase_files}")
    print(f"Output prefix: {prefix}")
    print(f"Device: {device}")
    if n_echoes > 1:
        print(f"Multi-echo: {n_echoes} echoes, rescue={'on' if args.rescue else 'off'}")

    cfg = NordicConfig(
        temporal_phase=args.temporal_phase,
        phase_filter_width=args.phase_filter_width,
        noise_volume_last=args.noise_volume_last,
        factor_error=args.factor_error,
        nordic=(True if (args.nordic or args.mp == 0) else False),
        mp_mode=args.mp,
        magnitude_only=args.magnitude_only,
        kernel_size_pca=tuple(args.kernel_size_pca) if args.kernel_size_pca is not None else None,
        kernel_size_gfactor=tuple(args.kernel_size_gfactor),
        gfactor_nvols=args.gfactor_nvols,
        patch_overlap=max(1, args.patch_overlap),
        gfactor_patch_overlap=max(1, args.gfactor_patch_overlap),
        use_magn_for_gfactor=args.use_magn_for_gfactor,
        phase_slice_average=args.phase_slice_average,
        save_gfactor_map=args.save_gfactor_map,
        save_residual_map=args.save_residual_map,
        add_mean=args.add_mean,
        retain_dof=args.retain_dof,
        save_num_comps=args.save_num_comps,
        save_task_loss=save_task_loss,
        task_design=task_design,
        task_polort=task_polort if args.events is not None else 1,
        task_fdr_q=args.task_fdr_q,
        make_complex_nii=args.make_complex_nii,
        nifti_ext=prefix_info.nifti_ext,
        svd_batch_size=args.svd_batch_size,
        decomp_method=args.decomp_method,
        rescue=args.rescue,
        rescue_band=args.rescue_band,
        rescue_alpha=args.rescue_alpha,
        per_echo_gfactor=args.per_echo_gfactor,
        resid_qc=args.resid_qc,
        factor_sweep=args.factor_sweep,
        factor_sweep_values=sweep_values,
        factor_sweep_max_voxels=sweep_max_voxels,
        factor_sweep_save_imgs=args.save_imgs,
        factor_sweep_save_factor_img=args.save_factor_img,
        factor_sweep_save_eigen_img=args.save_eigen_img,
        verbose=args.verb >= 1,
    )

    t0 = time.time()
    if run_sweep_path:
        summary = run_nordic_factor_sweep(
            magnitude_file=magn_files[0],
            phase_file=phase_files[0] if phase_files is not None else None,
            output_prefix=prefix,
            config=cfg,
            device=device,
        )
        elapsed = time.time() - t0
        print("\nDone (factor sweep)" if args.factor_sweep else "\nDone (diagnostic images)")
        sf = summary.get("suggested_factor")
        if sf is not None:
            print(f"  Liftoff factor (null held up to): {sf.get('liftoff_factor')}")
        for key, path in summary.get("outputs", {}).items():
            print(f"  {key}: {path}")
        print(f"  Elapsed: {elapsed:.1f} s")
        return

    if n_echoes > 1:
        all_outputs = run_nordic_multiecho(
            magnitude_files=magn_files,
            phase_files=phase_files,  # type: ignore[arg-type]
            output_prefix=prefix,
            config=cfg,
            device=device,
        )
    else:
        all_outputs = [
            run_nordic(
                magnitude_file=magn_files[0],
                phase_file=phase_files[0] if phase_files is not None else None,
                output_prefix=prefix,
                config=cfg,
                device=device,
            )
        ]
    elapsed = time.time() - t0

    print("\nDone")
    for i, outputs in enumerate(all_outputs):
        if n_echoes > 1:
            print(f"  [echo {i + 1}]")
        print(f"  Magnitude output: {outputs.magnitude_file}")
        if outputs.phase_file is not None:
            print(f"  Phase output: {outputs.phase_file}")
        if outputs.gfactor_file is not None:
            print(f"  G-factor output: {outputs.gfactor_file}")
        if outputs.residual_file is not None:
            print(f"  Residual output: {outputs.residual_file}")
        if outputs.task_loss_file is not None:
            print(f"  Task-loss output: {outputs.task_loss_file}")
        if outputs.task_patch_file is not None:
            print(f"  Patch-test output: {outputs.task_patch_file}")
        if outputs.num_comps_file is not None:
            print(f"  Num-comps output: {outputs.num_comps_file}")
        if outputs.recfactor_file is not None:
            print(f"  Rec-factor output: {outputs.recfactor_file}")
        print(f"  Metadata: {outputs.metadata_file}")
    print(f"  Elapsed: {elapsed:.1f} s")

    if args.events is not None:
        print_cli_section("Task-leak diagnostic")
        for i, outputs in enumerate(all_outputs):
            label = f" [echo {i + 1}]" if n_echoes > 1 else ""
            _task_leak_report(outputs, magn_files[i], args, label)
            with open(outputs.metadata_file) as f:
                summary = json.load(f).get("task_patch_test")
            if summary is not None:
                report = _print_patch_test(summary, label)
                print("")
                print("\n".join(report))
                stem = str(outputs.task_loss_file).partition("_taskloss")[0]
                with open(f"{stem}_taskleak.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(report) + "\n")


if __name__ == "__main__":
    main()
