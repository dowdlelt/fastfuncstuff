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
        "-task_rescue",
        "-task-rescue",
        nargs="?",
        const="ica",
        default=None,
        metavar="MODE",
        help="ACT on the component scan: put back the removed components whose time "
        "course follows the task. MODE is 'ica' (the default, and currently the only "
        "one) — bare -task_rescue means -task_rescue ica.\n"
        "The measurement runs either way; this is the ffs_locomoco -detask of this tool, "
        "an opt-in pass that can only move in one direction. Nothing here can discard "
        "what the threshold kept, so the trade is a bounded amount of returned noise "
        "against a response that would otherwise be gone.\n"
        "It is also the single-echo sibling of -rescue: that one protects a component "
        "from removal when it has a correlated partner in another echo (thermal noise is "
        "the only thing independent across echoes); this one protects it when the task "
        "explains its time course, which needs no second echo.\n"
        "The primary output then CARRIES the rescue. What was added is written beside it "
        "as {prefix}_taskrescued, so the un-rescued series is exactly the output minus "
        "that file and nothing is hidden. Each restored component hands back one degree "
        "of freedom, reported so it can go into ffs_reml -adjust_dof. The task fit is "
        "re-run on the rescued series and printed BEFORE -> AFTER.\n"
        "Requires -events.",
    )
    task_group.add_argument(
        "-task_comps",
        "-task-comps",
        default="auto",
        metavar="N|auto|est|off",
        help="Component count for the scan of the removed field. 'auto' (default) uses a "
        "fixed floor of 10; 'est' runs the model-order estimator and takes max(est, 10); "
        "an integer forces the order; 'off' skips the scan (and with it -task_rescue).\n"
        "'auto' does not estimate because the estimate has never once changed the answer "
        "and is not cheap: on the removed field the estimator returns 1 (four synthetic "
        "constructions and a real 96x112x65x350 run), so max(est, floor) is the floor "
        "every time — and getting that 1 costs a whole-volume ACF smoothness pass, "
        "measured at 68 s on that run's geometry.\n"
        "The floor is why this is not just the estimator. That estimator is conservative "
        "on the removed field, which is what makes it trustworthy for 'was anything "
        "structured taken out' — and is also its limit here: a component NORDIC "
        "discarded was, by definition, at the thermal-noise level inside its own patch, "
        "so pooled over the brain it can be too small a share of the removed variance to "
        "earn a component of its own. Measured on synthetic runs it returned k=1 every "
        "time, and one component pooling the whole removed field is not a thing a leak "
        "can be found in.\n"
        "The floor is SMALL (10) on purpose, and forcing N HIGHER usually costs "
        "detections rather than buying them: the leak's z is set by the data and barely "
        "moves with the order, while the familywise cut grows with the number of "
        "components tested. Measured on a planted over-removal, three seeds: flagged 3/3 "
        "at k=5 and k=10, 0/3 at k=15, 30 and 60, with no false positive at any order on "
        "runs with no task in them.",
    )
    task_group.add_argument(
        "-task_alpha",
        "-task-alpha",
        type=float,
        default=0.05,
        metavar="A",
        help="Familywise error rate for the component scan's selection. Default 0.05.",
    )
    task_group.add_argument(
        "-task_surrogates",
        "-task-surrogates",
        type=int,
        default=2000,
        metavar="N",
        help="Phase-randomised draws per component for the component scan's null. Default 2000.",
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


def parse_task_comps(value) -> str | int:
    """``-task_comps`` -> ``"auto"``, ``"est"``, ``"off"``, or a forced component count."""
    text = str(value).strip().lower()
    if text in ("auto", "est", "off"):
        return text
    try:
        k = int(text)
    except ValueError:
        raise ValueError(
            f"-task_comps takes an integer, 'auto', 'est' or 'off', got {value!r}"
        ) from None
    if k < 1:
        raise ValueError(f"-task_comps needs at least one component, got {k}")
    return k


def parse_task_rescue(value) -> str:
    """``-task_rescue`` MODE -> the mode name. Only ``ica`` exists today.

    Mode-valued rather than a bare switch because this is the same slot as
    ``ffs_locomoco -detask``: an opt-in pass over a measurement that ran anyway, and
    the place a second approach would go.
    """
    mode = str(value).strip().lower()
    if mode != "ica":
        raise ValueError(f"-task_rescue MODE must be 'ica', got {value!r}")
    return mode


def _scan_order(args, lost_vt, lost, mask, voxdims, n_t, device) -> tuple[int, dict]:
    """Component count for the scan: forced, or the estimator under a small floor.

    The estimator is not wrong when it says one -- the removed field IS the
    Marchenko-Pastur bulk and mostly thermal noise, which is the regime it is built for
    and the answer it should give. The floor is about what this particular scan is for.
    A component NORDIC discarded was by definition at the thermal-noise level inside its
    own patch, so pooled over the brain a real leak can be too small a share of the
    removed variance to earn a component of its own; measured across the synthetic runs
    the estimate came back k=1 every time, and one component pooling the whole removed
    field is not a thing a leak can be found in.

    Why the floor is SMALL, which is the counter-intuitive half. Raising the order does
    not buy sensitivity here, it spends it: the leak's z is set by the data and barely
    moves with k, while the familywise cut grows with the number of components tested.
    Measured on a planted over-removal, three seeds, z of the leak against the cut::

        k       leak z        z_cut        flagged     no-task z (3 seeds)
        5      4.34-4.51    3.07-3.59        3/3         -0.09 to 0.75
        10     4.08-4.25    3.83-3.97        3/3          0.24 to 2.30
        15     3.99-4.19    4.06-4.30        0/3          1.18 to 2.11
        30       4.05         4.60           0/3
        60       4.14         5.15           0/3

    Ten detects every planted leak with margin, flags nothing on runs with no task in
    them, and still gives a leak room to separate from the bulk. Thirty -- the order a
    "look harder" instinct reaches for -- misses all three.
    """
    from fastfuncstuff.cli_utils import spinner
    from fastfuncstuff.decomposition.model_order import effective_sample_size_from_resels
    from fastfuncstuff.decomposition.tools import estimate_ica_component_count
    from fastfuncstuff.decomposition.workflow import estimate_smoothness_resels_acf

    floor = max(2, min(10, n_t // 8))
    if isinstance(args.task_comps, int):
        return int(args.task_comps), {"mode": "explicit"}
    if args.task_comps != "est":
        # The estimator is not run at all by default, because its answer has never once
        # changed the outcome and it is not cheap. On the removed field it returns k=1 --
        # four synthetic constructions and a real 96x112x65x350 run -- so max(est, floor)
        # IS the floor every time; and getting that 1 costs a whole-volume ACF smoothness
        # pass, measured at 68 s on that real run's geometry. Sixty-eight seconds for a
        # number the floor overrides is not a default. `-task_comps est` runs it.
        return floor, {"mode": "floor", "floor": int(floor)}
    n_in = int(mask.sum())
    # On the user's device, not pinned to the CPU: this is a full 4-D pass and it was
    # the single biggest cost in the scan (about 7 s on a small synthetic run, and it
    # scales with the dataset).
    with spinner("Estimating the removed field's smoothness"):
        resels, _, _ = estimate_smoothness_resels_acf(
            lost.numpy(), voxdims, mask=mask.numpy(), device=device
        )
    n_eff = effective_sample_size_from_resels(n_in, resels, floor=n_t)
    k_est, _, diag = estimate_ica_component_count(
        data_vox_t=lost_vt.to(device),
        method="auto",
        max_auto_components=max(2, int(round(0.66 * n_t))),
        auto_min_components=min(5, max(2, n_t // 4)),
        auto_var_threshold=0.90,
        use_mp_prior=True,
        n_eff=n_eff,
        device=device,
        verbose=False,
    )
    diag = dict(diag, k_estimated=int(k_est), floor=int(floor), resels=float(resels), n_eff=n_eff)
    return max(int(k_est), floor), diag


def _task_component_scan(outputs, args, label: str, device, state) -> dict | None:
    """Decompose what was removed and ask which components follow the task.

    This is the temporal half of the ``-events`` diagnostic and it runs by default,
    because it has power exactly where the spatial half does not. The enrichment needs
    a leak big enough to move a mask placed on half the run; a component that is
    genuinely task-locked but lives in a handful of voxels does not move it, and shows
    up here instead. The cost is a smoothness estimate and one small ICA -- seconds
    against a denoising run -- so measuring is the default and ``-task_comps off`` is
    the escape.

    Nothing is changed unless ``-task_rescue`` is given. That pass is the single-echo
    sibling of ``-rescue``: that one protects a component from removal when it has a
    correlated partner in another echo, on the argument that thermal noise is the only
    thing independent across echoes; this one protects it when the task explains its
    time course, which needs no second echo. It can only ever make the denoising remove
    **less** -- nothing here can discard what the threshold kept -- which is the
    asymmetry that lets it sit behind one flag rather than in a separate tool, the same
    shape as ``ffs_locomoco -detask``.

    Selection is familywise across components, one test per component against its own
    phase-randomised null. The per-voxel question does not arise: a component either
    follows the task in time or it does not.
    """
    from pathlib import Path

    import numpy as np

    from fastfuncstuff.decomposition.ica import FastICA
    from fastfuncstuff.denoise.restore import restore_components
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.stats.task_coupling import component_task_fit, component_variance_in_data

    # Both series come from the report that ran just above rather than off disk again:
    # each is a gigabyte on a real run, and re-reading cost a decompress plus an F-to-C
    # reorder for data already in hand.
    kept, lost = state["kept"], state["lost"]
    n_t = kept.shape[3]
    # The SAME brain mask the report used -- one definition of "brain" per run, and the
    # decomposition then sees tissue rather than whatever an intensity cut let through.
    mask = state["mask"].clone()
    # Constant voxels score a perfect fit and have poisoned two decompositions in this
    # codebase already (the MELODIC/GGM parity hunt, the denoisatorial PC selection).
    mask &= lost.std(dim=3) > 1e-8
    n_in = int(mask.sum())
    if n_in < n_t:
        print(f"  Component scan{label}: only {n_in} usable voxels for {n_t} frames; skipped.")
        return None

    lost_vt = lost.reshape(-1, n_t)[mask.reshape(-1)]  # (V', T)
    k, num_diag = _scan_order(args, lost_vt, lost, mask, state["voxdims"], n_t, device)
    est = num_diag.get("k_estimated")
    how = {
        "explicit": f"{k} forced",
        "floor": f"{k} (floor; -task_comps est to estimate)",
    }.get(str(num_diag.get("mode")), f"{k} = max(est {est}, floor {num_diag.get('floor')})")
    print(f"  COMPONENTS : {how}, over {n_in} vox x {n_t} frames")

    ica = FastICA(n_components=k, device=device, random_state=0, verbose=False)
    ica.fit(lost_vt.T.to(device))  # (T, V') -- mixing_ comes back (T, k)
    assert ica.mixing_ is not None and ica.components_ is not None  # fit() populates both
    mixing = ica.mixing_.detach().cpu()
    comps = ica.components_.detach().cpu()  # (k, V')

    design, polort = state["design"], state["polort"]
    fit = component_task_fit(
        mixing, design, polort=polort, n_surrogates=args.task_surrogates, alpha=args.task_alpha
    )
    flagged = [int(c) for c in fit["flagged"]]
    # What the flagged components are, as a share of what the denoising threw away. The
    # z above says a component follows the task; this says whether it is worth anything.
    shares = component_variance_in_data(
        mixing, lost, polort=polort, design=design, mask=mask, device=device
    )
    summary = {
        "n_components": int(k),
        "flagged_components": flagged,
        "z_cut": float(fit["z_cut"]),
        "eff_dof": float(fit["eff_dof"]),
        "informative": bool(fit["informative"]),
        "model_order_mode": str(num_diag.get("mode", "?")),
        "model_order_estimated": est,
        "rescued": False,
    }
    dof_txt = f"{fit['eff_dof_reported']:.0f}{'+' if fit['eff_dof_capped'] else ''} eff DoF"
    if fit["informative"]:
        print(
            f"               R2>{fit['r2_needed']:.3f} to flag "
            f"(fwe a={fit['alpha']:g} over {k}, {dof_txt})"
        )
    else:
        # Saying WHY beats printing "none flagged", which a reader would take as
        # evidence that nothing task-locked was removed rather than as an absent
        # measurement. A 20 s periodic block design lands here every time: a component
        # merely sharing its spectrum fits it by construction.
        print(
            f"               ⚠️  no power ({fit['uninformative_reason']}) -- the "
            "enrichment decides alone"
        )
        summary["uninformative_reason"] = str(fit["uninformative_reason"])
        return summary
    for c in flagged:
        print(
            f"               #{c}: R2 {float(fit['r2'][c]):.3f} z {float(fit['z'][c]):+.2f} "
            f"p {float(fit['p'][c]):.4g}; {100 * float(shares['var_data'][c]):.2f}% of "
            f"what was removed, {100 * float(shares['var_task'][c]):.1f}% of its task variance"
        )
    if flagged:
        summary["var_of_removed_flagged"] = [float(shares["var_data"][c]) for c in flagged]
        summary["var_of_removed_task_flagged"] = [float(shares["var_task"][c]) for c in flagged]
        print(
            f"               {len(flagged)} of {k} follow the task "
            f"(z_cut {float(fit['z_cut']):.2f})"
        )
    else:
        print(f"               none of {k} follows the task (z_cut {float(fit['z_cut']):.2f})")
    if not args.task_rescue:
        if flagged:
            print("               DIAGNOSTIC ONLY -- -task_rescue puts these back")
        return summary
    if not flagged:
        print("               -task_rescue: nothing to put back, output unchanged")
        return summary

    # Scatter the masked maps back to the full grid, so a voxel outside the mask can
    # never receive anything.
    maps_kv = torch.zeros(k, int(np.prod(kept.shape[:3])), dtype=torch.float32)
    maps_kv[:, mask.reshape(-1)] = comps.float()
    result = restore_components(kept, lost, maps_kv, mixing.float(), flagged)

    added = (result.restored - kept).numpy().astype(np.float32)
    stem, _, ext = str(outputs.task_loss_file).partition("_taskloss")
    added_path = f"{stem}_taskrescued{ext}"
    save_nifti(added, output_path=added_path, reference_img=str(outputs.magnitude_file))
    # The primary output now carries the rescue; what was added is written beside it, so
    # the un-rescued series is exactly output minus this file and nothing is hidden.
    save_nifti(
        result.restored.numpy().astype(np.float32),
        output_path=str(outputs.magnitude_file),
        reference_img=str(outputs.magnitude_file),
    )
    print(
        f"               put back {len(flagged)} comp(s), "
        f"{100 * result.var_returned_total:.2f}% of the removed variance, "
        f"{result.dof_returned} DoF -> {Path(added_path).name}"
    )
    print(
        f"               carry {result.dof_returned} DoF into ffs_reml -adjust_dof / "
        "ffs_util_updatedof"
    )
    summary |= {
        "rescued": True,
        "mode": args.task_rescue,
        "gammas": result.gammas.tolist(),
        "variance_share_of_removed_total": result.var_returned_total,
        "dof_returned": result.dof_returned,
        "added_file": added_path,
    }
    del lost, lost_vt, maps_kv, kept
    _task_fit_after(result.restored, state, args, label)
    return summary


def _task_fit_after(restored, state, args, label: str) -> None:
    """Re-measure the task fit on the RESCUED series, kept -> rescued.

    The scan above says a component follows the task; this says what putting it back did
    to the response, which is the question the whole diagnostic is about. It is the same
    before/after ``ffs_locomoco`` prints on its corrected data, and for the same reason:
    damage -- or repair -- becomes visible rather than inferred.

    Both halves are reported because they are not redundant. Restoring a component that
    carries response raises the AMPLITUDE in the voxels it lives in; ``|r|`` can move
    either way, since what came back carries its own noise into the residual.
    """
    from fastfuncstuff.cli.task_events import psc_betas, save_task_fit
    from fastfuncstuff.stats.task_coupling import task_coupling

    print(f"  TASK FIT KEPT -> RESCUED{label} (medians on the responding mask)")
    design, labels, polort, mask = state["design"], state["labels"], state["polort"], state["mask"]
    after = task_coupling(
        restored, design, polort=polort, mask=mask, labels=labels, device=state["device"]
    )
    psc_after = psc_betas(after, design, state["reference"], mask)
    sel = state["resp"] > 0
    for k, lb in enumerate(labels):
        r0 = state["kept_r"][..., k][sel].abs()
        r1 = after.r[..., k][sel].abs()
        b0 = state["kept_psc"][..., k][sel].abs()
        b1 = psc_after[..., k][sel].abs()
        base = float(b0.median())
        pct = 100.0 * (float(b1.median()) - base) / base if base > 0 else 0.0
        print(
            f"    {lb}: |r| {float(r0.median()):.3f} -> {float(r1.median()):.3f} "
            f"({float(r1.median()) - float(r0.median()):+.3f}), amplitude "
            f"{base:.3f} -> {float(b1.median()):.3f} %sig ({pct:+.1f}%)",
            flush=True,
        )
    stem, ext = state["stem"], state["ext"]
    save_task_fit(
        f"{stem}_taskfit_rescued{ext}",
        after,
        psc_after,
        labels,
        state["affine"],
        polort + 1,
        int(state["n_t"]),
    )


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
        f"  PATCH TEST : median z {z:+.2f} over {n_in} patches, {n_sig} positive at "
        f"q <= {summary['fdr_q']:g} -- {verdict}",
        f"               (patches discarded {summary['mean_removed_dim']:.1f} of "
        f"{summary['n_components']} components; negative z = preferentially kept)",
    ]


def _split_half_enrichment(source, lost, design, mask, tr, args, labels, device):
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
                source[..., rank_sl],
                d_rank,
                polort=po_r,
                mask=mask,
                labels=labels,
                device=device,
            )
            resp, _, _ = responding_mask(
                rank_tc.r_full, mask, args.task_top_frac, thresh=args.task_thresh
            )
            score_tc = task_coupling(
                lost[..., score_sl],
                d_score,
                polort=po_s,
                mask=mask,
                labels=labels,
                device=device,
            )
        except ValueError:
            # A half with too few blocks to separate the design from its own drift, or
            # with no voxel responding in it, contributes nothing rather than a number.
            continue
        out.append(map_enrichment(score_tc.task_rms, resp, mask)["enrichment"])
    if not out:
        return None, []
    return sum(out) / len(out), out


def _brain_mask(reference: torch.Tensor, device) -> torch.Tensor:
    """A real automask on the magnitude mean, not an intensity cut.

    What this replaced was ``reference > 0.1 * reference.max()``, which on an EPI keeps
    whatever is bright: ghosts, neck, skull marrow, the rim where the coil sensitivity
    peaks. It matters in both directions here. The enrichment divides by the share of
    voxels the responding mask occupies, so junk in the denominator moves the headline;
    and the component scan decomposes every voxel in the mask, so junk gets its own
    independent components to be found in.

    Falls back to the intensity cut if the automask comes back empty or absurd -- this
    is a diagnostic, and it must not take a finished denoising down with it.
    """
    from fastfuncstuff.processing.mask import automask

    cut = reference > 0.1 * float(reference.max())
    try:
        mask = automask(reference, device=device).to(torch.bool).cpu()
    except (RuntimeError, ValueError) as exc:
        print(f"  ⚠️  automask failed ({exc}); falling back to an intensity cut")
        return cut
    n, n_cut = int(mask.sum()), int(cut.sum())
    if n < 0.1 * n_cut:
        print(f"  ⚠️  automask kept only {n} voxels against {n_cut}; using the intensity cut")
        return cut
    return mask


def _task_leak_report(outputs, magnitude_file: str, args, label: str, device) -> dict:
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
    # nibabel hands back an F-CONTIGUOUS array, so something has to reorder it -- and
    # np.ascontiguousarray does that single-threaded: 4.15 s for one gigabyte, against
    # 0.80 s for torch's threaded copy of the same array. Two volumes, so ~7 s of a run's
    # CPU time went on a memory layout.
    kept = torch.as_tensor(kept_img.get_fdata(dtype=np.float32)).contiguous()
    lost = torch.as_tensor(
        load_nifti(str(outputs.task_loss_file)).get_fdata(dtype=np.float32)
    ).contiguous()
    affine = kept_img.affine
    n_t = kept.shape[3]

    design, labels = task_design_from_events(args, n_t, tr, torch.device("cpu"))
    polort = args.task_polort if args.task_polort is not None else default_polort(n_t, tr)
    # Before the two fits, not after: a wrong -tr shows up here as a design that never
    # lines up with the run, and the numbers below are uninterpretable until it is right.
    print(
        f"  design     : {len(labels)} condition(s) ({', '.join(labels)}), "
        f"TR {tr:g}s, polort {polort}, {n_t} frames"
    )

    # kept + lost IS the input magnitude, by construction of the loss series -- cheaper
    # and exactly consistent with re-reading the input file, which may still carry the
    # trailing noise volumes NORDIC trimmed. Three float32 volumes is the peak here,
    # and this runs after the GPU work is done.
    source = kept + lost
    reference = source.mean(dim=3)
    mask = _brain_mask(reference, device)
    # The series stay in host memory (three float32 volumes of a real run is more than
    # a consumer card holds) and task_coupling streams them to `device` in chunks. Left
    # on the CPU this is five full passes over a 250k-voxel, 350-frame dataset and it
    # dominated the tool's wall clock -- minutes after a 52 s denoise.
    kwargs = dict(polort=polort, mask=mask, labels=labels, device=device)
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
    enrich, per_half = _split_half_enrichment(source, lost, design, mask, tr, args, labels, device)
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
    # The verdict is the whole point of the line and it is binary. Above 1.0 the removed
    # series' task energy sits preferentially on responding tissue, which is a leak;
    # at or below it there is nothing task-shaped in what was removed, and how far below
    # carries no meaning -- the statistic has no informative range under its own null.
    if enrich is None:
        enrich_txt = "n/a  (neither half of the run places a mask)"
    elif enrich <= 1.0:
        enrich_txt = (
            f"{enrich:.2f}x  (halves {', '.join(f'{e:.2f}' for e in per_half)})"
            "  -- no task structure in what was removed"
        )
    else:
        enrich_txt = (
            f"{enrich:.2f}x  (halves {', '.join(f'{e:.2f}' for e in per_half)})"
            "  -- SOME TASK in what was removed, concentrated where the response is"
        )
    # Terse on stdout, long-form in the file. Every number a reader acts on is here;
    # what each one MEANS is in -help and in {prefix}_taskleak.txt, which is where a
    # reader who needs it goes once, rather than scrolling past it on every run.
    lines = [
        f"  mask       : {int(resp.sum())} vox ({how}, input R > {cut:.3f}) = "
        f"{100 * int(resp.sum()) / max(1, int((mask > 0).sum())):.1f}% of brain",
        f"  input      : R {src_sum['r_full_median']:.3f}, task rms "
        f"{src_sum['task_rms_median']:.4g}",
        f"  kept       : R {kept_sum['r_full_median']:.3f}, task rms "
        f"{kept_sum['task_rms_median']:.4g} ({100 * kept_ratio:.1f}% of input)",
        f"  lost       : R {lost_sum['r_full_median']:.3f}, task rms "
        f"{lost_sum['task_rms_median']:.4g} ({100 * amp_ratio:.1f}% of input); task "
        f"share {lost_share:.3f} vs {lost_tc.chance_share:.3f} chance",
        f"  ENRICHMENT : {enrich_txt}",
    ]
    if enrich is not None and enrich > 1.0:
        lines.append("               -retain_dof caps how much any patch may remove")
    print("\n".join(lines))

    detail = [
        "",
        "How to read this (see also `ffs_nordic -help`):",
        "  The mask is DESCRIPTIVE only. The input/kept/lost lines are selected on it and",
        "  read high because of that. The enrichment is not.",
        "",
        "  ENRICHMENT is the share of the lost series' task-locked energy landing inside the",
        "  responding mask, over the share of voxels that mask occupies. Its no-relation value",
        "  is 1.0 by construction, so it needs no surrogate -- which matters, because a block",
        "  design admits none. Well above 1.0x means real response left with the noise;",
        "  -retain_dof caps how much any patch may remove.",
        "",
        "  The mask comes from one half of the run and the energy is scored on the other, so",
        "  the 1.0x floor is real and not an artefact of selection: a whole-run mask reads",
        "  ~3.9x on data with no task in it at all. The cost is power -- half a run's blocks",
        "  place the mask, so a weak leak attenuates toward 1.0x rather than showing up. That",
        "  is why the component scan runs too: it asks the same question in TIME, where a leak",
        "  confined to a few voxels still shows.",
        "",
        "  NOTE the input is not motion-corrected at this stage, so the per-voxel fits are",
        "  worse than a preprocessed run's -- read the enrichment, not the R maps.",
    ]
    header = (
        f"Task-leak diagnostic{label}\n"
        f"  design         : {len(labels)} condition(s) ({', '.join(labels)}), "
        f"TR {tr:g}s, polort {polort}, {n_t} frames"
    )
    Path(f"{stem}_taskleak.txt").write_text(
        header + "\n" + "\n".join(lines + detail) + "\n", encoding="utf-8"
    )
    # Everything the component scan and its after-fit need, so neither re-derives the
    # design, the mask or the responding stratum -- a second threshold chosen there
    # would let the two halves of one report disagree about where the task is.
    return {
        "design": design,
        "labels": labels,
        "polort": polort,
        "mask": mask,
        "resp": resp,
        "reference": reference,
        "kept_r": kept_tc.r,
        "kept_psc": psc_betas(kept_tc, design, reference, mask),
        "affine": affine,
        "stem": stem,
        "ext": ext,
        "n_t": n_t,
        "device": device,
        # Handed to the component scan so it need not re-read a gigabyte apiece.
        "kept": kept,
        "lost": lost,
        "voxdims": tuple(float(v) for v in kept_img.header.get_zooms()[:3]),
    }


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
    try:
        args.task_comps = parse_task_comps(args.task_comps)
        if args.task_rescue is not None:
            args.task_rescue = parse_task_rescue(args.task_rescue)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if args.task_rescue and args.events is None:
        print("ERROR: -task_rescue needs -events to know what the task is")
        sys.exit(1)
    if args.task_rescue and args.task_comps == "off":
        print("ERROR: -task_rescue acts on the component scan that -task_comps off disables")
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
        print("")
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

    if args.events is not None:
        print_cli_section("Task-leak diagnostic")
        for i, outputs in enumerate(all_outputs):
            label = f" [echo {i + 1}]" if n_echoes > 1 else ""
            state = _task_leak_report(outputs, magn_files[i], args, label, device)
            with open(outputs.metadata_file) as f:
                summary = json.load(f).get("task_patch_test")
            if summary is not None:
                report = _print_patch_test(summary, label)
                print("\n".join(report))
                stem = str(outputs.task_loss_file).partition("_taskloss")[0]
                with open(f"{stem}_taskleak.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(report) + "\n")
            if args.task_comps != "off":
                scan = _task_component_scan(outputs, args, label, device, state)
                if scan is not None:
                    with open(outputs.metadata_file) as f:
                        meta = json.load(f)
                    meta["task_component_scan"] = scan
                    with open(outputs.metadata_file, "w") as f:
                        json.dump(meta, f, indent=2)

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
            if args.events is not None:
                stem = str(outputs.task_loss_file).partition("_taskloss")[0]
                print(f"  Task-leak report: {stem}_taskleak.txt")
        if outputs.task_patch_file is not None:
            print(f"  Patch-test output: {outputs.task_patch_file}")
        if outputs.num_comps_file is not None:
            print(f"  Num-comps output: {outputs.num_comps_file}")
        if outputs.recfactor_file is not None:
            print(f"  Rec-factor output: {outputs.recfactor_file}")
        print(f"  Metadata: {outputs.metadata_file}")
    print(f"  Elapsed: {elapsed:.1f} s")
    # A blank final line: without it the next run of a batch or a loop starts on the
    # line this one ended, and the two runs read as one.
    print("")


if __name__ == "__main__":
    main()
