"""Shared task-coupling diagnostic plumbing for the ``-events`` CLIs.

Every tool that asks "is the task in this map?" — ``ffs_locomoco`` on a displacement
field, ``ffs_nordic`` on the residual it threw away — needs the same three things: a
convolved design built from BIDS events the way every other ffs GLM builds one, and
the two AFNI bucket writers that make the answer thresholdable in the viewer.

The statistics themselves live in :mod:`fastfuncstuff.stats.task_coupling`; this
module is only the CLI-facing wrapper around them (argparse ``args`` in, files out).
"""

from __future__ import annotations

from pathlib import Path


def task_design_from_events(args, n_timepoints: int, tr: float, device):
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


def coupling_stataux(n_t: int, n_ort: int, n_sub: int, n_fit: int = 1) -> dict:
    """AFNI ``fico`` parameters for a drift-partial correlation map, per sub-brick.

    SAMPLES = timepoints, FIT-PARAMETERS = 1 (the single regressor each sub-brick is
    correlated against), ORT-PARAMETERS = every column removed from both sides first --
    the polort+1 Legendre drift columns, plus any nuisance regressors carried in the
    model, so a fit that spends degrees of freedom on warp PCs is not credited with the
    dof of one that did not.

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

    return {i: stat_type_to_stataux("fico", (n_t, n_fit, n_ort)) for i in range(n_sub)}


def save_labelled_map(path, arr, labels, affine, stataux=None):
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


def stack_bricks(path, bricks, affine):
    """Write ``[(name, 3-D map, stataux|None), ...]`` as one bucket, in order."""
    import torch

    names = [b[0] for b in bricks]
    stataux = {i: b[2] for i, b in enumerate(bricks) if b[2] is not None}
    save_labelled_map(
        path, torch.stack([b[1].float() for b in bricks], dim=-1), names, affine, stataux
    )


def full_model_brick(tc, n_ort, n_t):
    """The lead sub-brick of every task map: R of the WHOLE design, tagged ``fico``.

    "How well does this voxel fit the model" is the question a reader actually has, and
    the per-condition correlations below it cannot answer it.  Each of those is
    MARGINAL -- fitted against one regressor with the other K-1 responses left in the
    residual -- so on a run with several conditions it is capped at
    ``corr(x_k, sum_j x_j)`` no matter how strong the response is.  Measured on an
    8-condition rapid event-related run, a voxel carrying a NOISELESS copy of the full
    task response reads 0.11..0.31 per condition and 1.000 here.

    ``nfit`` is the design's rank, so AFNI's ``correl_t2p`` gets the multiple-
    correlation null with the right numerator dof rather than a simple-r one.
    """
    return (
        "full_model_R",
        tc.r_full,
        coupling_stataux(n_t, n_ort, 1, n_fit=tc.n_fit)[0],
    )


def save_task_map(path, tc, labels, affine, *, n_ort=None, n_t=0):
    """The FIELD's coupling: full-model R, then the per-condition MARGINAL r.

    Marginal is deliberate on this side -- "does the field follow THIS regressor" is
    the contamination question, and letting correlated conditions share credit is the
    honest reading of it.  ``full_model_R`` leads because it is the one that answers
    "does the field fit the model at all".
    """
    stat = coupling_stataux(n_t, n_ort, 1)[0] if n_ort is not None else None
    bricks = [] if n_ort is None else [full_model_brick(tc, n_ort, n_t)]
    bricks += [(lb, tc.r[..., k], stat) for k, lb in enumerate(labels)]
    stack_bricks(path, bricks, affine)


def save_task_fit(path, tc, psc, labels, affine, n_ort, n_t):
    """The DATA's fit, as an AFNI bucket: full-model R, then amplitude/stat per condition.

    ``full_model_R`` first, then ``{cond}_Coef`` (percent signal change) and
    ``{cond}_Correl`` alternating, so AFNI's usual "view the beta, threshold on the
    sub-brick after it" gesture works without opening two datasets and keeping their
    indices lined up by hand.

    Both per-condition halves come from the JOINT fit -- one multiple regression over
    the whole design -- which is what makes them comparable to the betas and t-stats
    of the GLM this run is headed for.  ``nort`` absorbs the other ``n_fit - 1``
    regressors, so the partial correlation is tagged with the dof it actually has.
    """
    stat = coupling_stataux(n_t, n_ort + tc.n_fit - 1, 1)[0]
    bricks = [full_model_brick(tc, n_ort, n_t)]
    for k, lb in enumerate(labels):
        bricks += [(f"{lb}_Coef", psc[..., k], None), (f"{lb}_Correl", tc.r_joint[..., k], stat)]
    stack_bricks(path, bricks, affine)


def psc_betas(tc, design, reference, mask):
    """Joint condition betas as PERCENT SIGNAL CHANGE of each voxel's temporal mean.

    ``tc.beta_joint`` is map units per unit of regressor from ONE multiple regression
    over the whole design, so the response a condition actually produces is
    ``beta × the regressor's peak-to-trough swing``; dividing by the voxel mean and
    scaling by 100 puts every condition, voxel and dataset on the one axis a reader can
    judge — a 2% response is a 2% response whatever the scanner's arbitrary intensity
    units were. Nothing extra is fitted: the joint betas come out of the same solve as
    ``r``, at no additional cost.

    The MARGINAL ``tc.beta`` is not used here. It answers a different question, and as
    an amplitude it is simply wrong whenever conditions are correlated: each condition
    is credited with whatever the others explain too.
    """
    import numpy as np
    import torch

    x = torch.as_tensor(np.asarray(design), dtype=torch.float32)
    swing = (x.amax(dim=0) - x.amin(dim=0)).clamp(min=1e-12)
    base = reference.float().abs()
    ok = (mask > 0) & (base > 0)
    out = torch.zeros(*tc.beta_joint.shape[:3], tc.beta_joint.shape[-1], dtype=torch.float32)
    for k in range(out.shape[-1]):
        out[..., k] = torch.where(
            ok,
            100.0 * tc.beta_joint[..., k].float() * float(swing[k]) / base.clamp(min=1e-12),
            torch.zeros_like(base),
        )
    return out
