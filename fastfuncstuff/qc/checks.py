"""Reusable QC checks: each adds figures and metrics to a :class:`QCRecord`.

Checks are by *domain* (motion, an EPI series, an alignment), not by tool, so
ffs_moco, ffs_locomoco and ffs_allineate draw the same picture of the same thing.
Tools call them with arrays already in memory; nothing here reloads a file.

Warning limits are deliberately few and each states its reason. Most metrics
carry none: what counts as bad depends on the acquisition, and the cross-subject
table (``ffs_qc_table``) flags robust outliers among comparable runs instead.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import plots, render, style
from .plots import Trace
from .record import QCRecord

# AFNI's own pre-steady-state rule (3dToutcount in afni_proc): more than 40% of
# voxels outlying in the first volume means it was acquired before T1 settled.
PRE_STEADY_LIMIT = 0.4
# A censor fraction above this leaves too little data to trust a run's estimates;
# a common exclusion rule in afni_proc-based studies (often 10-20%).
CENSOR_FRAC_WARN = 0.15


def _status(flag: bool) -> str:
    return "warn" if flag else "ok"


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def motion_check(
    rec: QCRecord,
    step: int,
    afni_params: np.ndarray,
    *,
    tr: float | None = None,
    enorm: np.ndarray | None = None,
    fd: np.ndarray | None = None,
    max_displacement: np.ndarray | None = None,
    outlier_frac: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    motion_limit: float | None = None,
    motion_metric: str = "enorm",
    outlier_limit: float | None = None,
    subtitle: str = "",
) -> None:
    """Six parameters, per-TR motion, outlier fraction; censored TRs shaded.

    ``afni_params`` is ``(nt, 6)`` in 3dvolreg order (roll pitch yaw dS dL dP,
    degrees and mm) — :func:`fastfuncstuff.processing.ffs_moco.to_afni_motion`.
    """
    p = np.asarray(afni_params, dtype=float)
    nt = p.shape[0]
    rec.metric("n_vols", nt, "", "volumes in the run")
    if tr:
        rec.metric("tr", tr, "s", "repetition time")
    rot_range = float(np.ptp(p[:, :3], axis=0).max()) if nt else 0.0
    tra_range = float(np.ptp(p[:, 3:], axis=0).max()) if nt else 0.0
    rec.metric("rot_range", rot_range, "deg", "largest peak-to-peak rotation (any axis)")
    rec.metric("trans_range", tra_range, "mm", "largest peak-to-peak translation (any axis)")
    stats: list[tuple[str, str | None]] = []
    if enorm is not None:
        rec.metric(
            "enorm_mean",
            float(np.mean(enorm)),
            "mm",
            "mean Euclidean norm of the motion derivative (AFNI average motion per TR)",
        )
        rec.metric("enorm_max", float(np.max(enorm)), "mm", "largest single-TR motion (enorm)")
        stats.append((f"mean enorm {np.mean(enorm):.3f} mm", None))
    if fd is not None:
        rec.metric("fd_mean", float(np.mean(fd)), "mm", "mean framewise displacement (Power)")
        rec.metric("fd_max", float(np.max(fd)), "mm", "largest framewise displacement")
        stats.append((f"mean FD {np.mean(fd):.3f} mm", None))
    if max_displacement is not None:
        rec.metric(
            "maxdisp",
            float(np.max(max_displacement)),
            "mm",
            "largest displacement of any brain voxel from the base",
        )
    if keep is not None:
        k = np.asarray(keep)
        frac = float((k == 0).mean()) if k.size else 0.0
        rec.metric("n_censored", int((k == 0).sum()), "TRs", "time points censored")
        rec.metric(
            "censor_frac", frac, "", "fraction of time points censored", warn_gt=CENSOR_FRAC_WARN
        )
        stats.append((f"censored {100 * frac:.1f}%", _status(frac > CENSOR_FRAC_WARN)))
    if outlier_frac is not None:
        of = np.asarray(outlier_frac, dtype=float)
        rec.metric(
            "outlier_mean", float(of.mean()), "", "mean per-TR outlier fraction (3dToutcount)"
        )
        rec.metric(
            "outlier_vol0",
            float(of[0]),
            "",
            "outlier fraction of the first volume; above 0.4 = pre-steady-state",
            warn_gt=PRE_STEADY_LIMIT,
        )
        if of[0] > PRE_STEADY_LIMIT:
            stats.append(("pre-steady-state vol 0", "warn"))

    roll, pitch, yaw = (p[:, i] for i in range(3))
    d_s, d_l, d_p = (p[:, i] for i in range(3, 6))
    panels = [
        (
            "rotation (deg)",
            [
                Trace("roll (I-S)", roll, style.BLUE),
                Trace("pitch (R-L)", pitch, style.ORANGE),
                Trace("yaw (A-P)", yaw, style.AQUA),
            ],
        ),
        (
            "translation (mm)",
            [
                Trace("dS", d_s, style.BLUE),
                Trace("dL", d_l, style.ORANGE),
                Trace("dP", d_p, style.AQUA),
            ],
        ),
    ]
    for name, trace in (("enorm", enorm), ("fd", fd)):
        if trace is not None:
            lim = motion_limit if motion_metric == name else None
            label = "enorm (mm)" if name == "enorm" else "FD (mm)"
            panels.append((label, [Trace(name, np.asarray(trace), style.VIOLET, lim)]))
    if outlier_frac is not None:
        panels.append(
            (
                "outlier fraction",
                [Trace("outliers", np.asarray(outlier_frac), style.MAGENTA, outlier_limit)],
            )
        )
    body = plots.stacked_traces(panels, tr=tr, keep=keep)
    cap = (
        "Look for: smooth, small parameter traces; spikes in enorm/FD that line up with "
        "outlier spikes are real head motion. Shaded bands are censored time points. "
        "Slow drifts are normal; sudden steps are the subject moving."
    )
    img = render.compose(
        body, f"Motion · {rec.label}", subtitle or f"{rec.tool}  ·  {nt} volumes", stats, cap
    )
    path = rec.figure_path(step, "motion")
    render.save_png(img, path)
    rec.add_figure(step, "motion", "Motion parameters", path)


# ---------------------------------------------------------------------------
# One EPI series: mean, tSNR, carpet
# ---------------------------------------------------------------------------


def _legendre(nt: int, degree: int, device, dtype):
    """Orthonormal linear (+ quadratic) drift columns, mean removed. Built in float64
    on the CPU (a tiny QR) and only then moved and cast for the device."""
    import torch

    from fastfuncstuff import policy_linalg as plinalg

    t = torch.linspace(-1, 1, nt, dtype=torch.float64)
    cols = [t] + ([1.5 * t * t - 0.5] if degree >= 2 else [])
    a = torch.stack(cols, 1)
    q, _ = plinalg.qr(a - a.mean(0))
    return q.to(dtype).to(device)


def epi_check(
    rec: QCRecord,
    step: int,
    series,
    affine,
    *,
    tr: float | None = None,
    keep: np.ndarray | None = None,
    motion: np.ndarray | None = None,
    motion_name: str = "enorm",
    device=None,
    groups: Mapping[str, np.ndarray] | None = None,
    what: str = "corrected",
) -> None:
    """Mean + tSNR montage and a carpet plot for one 4-D series.

    ``groups`` maps a name to a boolean ``(z, y, x)`` mask for ordering carpet
    rows (tissue classes when a segmentation is at hand). Without it the brain
    splits into its outer shell (where motion and edge effects show first) and
    its interior.
    """
    import torch

    from fastfuncstuff.processing.mask import automask

    x = series if isinstance(series, torch.Tensor) else torch.from_numpy(np.asarray(series))
    nt = x.shape[0]
    dev = torch.device(device) if device is not None else x.device
    mean, std = _slice_moments(x, dev)
    mask_t = automask(mean, device=torch.device("cpu")).cpu().bool()
    mask = mask_t.numpy()
    tsnr = torch.where(std > 0, mean / std, torch.zeros_like(mean)).numpy()
    # Outside the head tSNR is meaningless (and on low-noise backgrounds huge).
    tsnr = np.where(mask, tsnr, 0.0).astype(np.float32)
    nvox = int(mask.sum())
    med_tsnr = float(np.median(tsnr[mask])) if nvox else float("nan")
    rec.metric(
        "tsnr_median", med_tsnr, "", f"median detrended tSNR in the automask ({what} series)"
    )
    rec.metric("mask_voxels", nvox, "voxels", "automask size")

    # DVARS and global signal over the mask, from the slice-wise accumulators.
    dvars, gsig = _masked_traces(x, mask, dev)
    m_in = float(mean.numpy()[mask].mean()) if nvox else 1.0
    dvars = np.concatenate([[0.0], 100.0 * np.sqrt(dvars / max(nvox, 1)) / max(m_in, 1e-12)])
    gsig = gsig / max(nvox, 1)
    gpct = 100.0 * (gsig / max(gsig.mean(), 1e-12) - 1.0)
    rec.metric(
        "dvars_mean",
        float(dvars[1:].mean()) if nt > 1 else 0.0,
        "%",
        "mean DVARS (RMS of TR-to-TR change in the mask, % of mean signal)",
    )

    # Carpet voxels: subsample evenly beyond ~60k so a 1 mm brain stays cheap; rows
    # are block averages anyway.
    if groups is None:
        from fastfuncstuff.processing.mask import _erode_6conn

        core = _erode_6conn(mask_t, 2).numpy()
        groups = {"shell": mask & ~core, "interior": core}
    flat = {k: np.flatnonzero(v.reshape(-1) & mask.reshape(-1)) for k, v in groups.items()}
    total = sum(len(v) for v in flat.values())
    stride = max(1, total // 60000)
    flat = {k: v[::stride] for k, v in flat.items()}
    allidx = np.concatenate([v for v in flat.values()]) if total else np.zeros(0, int)
    gcor = float("nan")
    if allidx.size:
        ts = x.reshape(nt, -1)[:, torch.from_numpy(allidx)].float().numpy()
        pos, gl = 0, []
        for k, v in flat.items():
            gl.append((k, np.arange(pos, pos + len(v))))
            pos += len(v)
        rows, blocks = plots.carpet_rows(ts, gl)
        z = plots._zscore_detrended(ts)
        u = z / np.sqrt(max(nt, 1))
        gcor = float(np.sum(u.mean(axis=1) ** 2))
        rec.metric(
            "gcor",
            gcor,
            "",
            "global correlation: mean correlation between all voxel pairs (AFNI GCOR)",
        )
    else:
        rows, blocks = np.zeros((1, nt)), [("mask", 0, 1)]

    # --- figure 1: mean + tSNR montage --------------------------------------
    ras_mean, zooms = render.ras_volume(mean, affine)
    ras_tsnr, _ = render.ras_volume(tsnr, affine)
    ras_mask, _ = render.ras_volume(mask.astype(np.float32), affine)
    ras_mask = ras_mask > 0.5
    grid = render.Grid(ras_mean, zooms, ras_mask, n_cut=7)
    lo, hi = render.window(ras_mean, ras_mask)
    t_hi = float(np.percentile(ras_tsnr[ras_mask], 99)) if ras_mask.any() else 1.0
    rows_img, letters = [], []
    for plane in ("axial", "coronal", "sagittal"):
        asp = grid.aspect(plane)
        mean_panels = [
            render.panel(render.colorize(s, lo, hi), asp, 220) for s in grid.slices(ras_mean, plane)
        ]
        tsnr_panels = [
            render.panel(render.colorize(s, 0, t_hi, "inferno"), asp, 220)
            for s in grid.slices(ras_tsnr, plane)
        ]
        rows_img += [(f"mean\n{plane}", mean_panels), (f"tSNR\n{plane}", tsnr_panels)]
        letters += [grid.side_letter(plane)] * 2
    body = render.montage(rows_img, side_letters=letters)
    cap = (
        f"tSNR = mean / detrended temporal std, scaled 0 to {t_hi:.0f} (99th percentile). "
        "Look for: tSNR highest in cortex and falling off smoothly toward the centre; dark "
        "bands or stripes along slices (slice dropout, multiband leakage), signal loss near "
        "sinuses, or bright rims at the brain edge (motion)."
    )
    img = render.compose(
        body,
        f"Mean & tSNR · {rec.label}",
        f"{rec.tool}  ·  {what} series  ·  {nvox:,} automask voxels",
        [(f"median tSNR {med_tsnr:.1f}", None)],
        cap,
    )
    path = rec.figure_path(step, "tsnr")
    render.save_png(img, path)
    rec.add_figure(step, "tsnr", "Mean and tSNR", path)

    # --- figure 2: carpet -----------------------------------------------------
    traces = []
    if motion is not None:
        traces.append(Trace(f"{motion_name} (mm)", np.asarray(motion, float), style.VIOLET))
    # DVARS has no value at the first volume; NaN leaves a gap rather than a fake dip.
    traces.append(Trace("DVARS (%)", np.concatenate([[np.nan], dvars[1:]]), style.ORANGE))
    traces.append(Trace("global (%)", gpct, style.AQUA))
    body = plots.carpet(rows, blocks, traces, tr=tr, keep=keep)
    cap = (
        "Each row is a group of voxels (detrended, z-scored, grey = +/-2 SD); each column a "
        "volume. Look for: vertical bands across all rows (motion, global signal changes) that "
        "line up with the traces above; horizontal stripes (bad voxels/slices); a shell "
        "brighter than the interior during motion."
    )
    img = render.compose(
        body,
        f"Carpet · {rec.label}",
        f"{rec.tool}  ·  {what} series  ·  rows: " + ", ".join(b[0] for b in blocks),
        [
            (f"GCOR {gcor:.3f}", None),
            (f"mean DVARS {dvars[1:].mean() if nt > 1 else 0:.2f}%", None),
        ],
        cap,
    )
    path = rec.figure_path(step + 1, "carpet")
    render.save_png(img, path)
    rec.add_figure(step + 1, "carpet", "Carpet plot", path)


def _slice_moments(x, dev) -> tuple:
    """Per-voxel mean and detrended temporal std, one slice on the device at a time.

    The trend (linear + quadratic Legendre) is removed before the std: raw std
    counts scanner drift as noise and understates tSNR on long runs. Slices go to
    the device one at a time, so the series is never duplicated there.
    """
    import torch

    nt, nz = x.shape[0], x.shape[1]
    q = _legendre(nt, 2, dev, torch.float32) if nt > 3 else None
    mean = torch.empty(x.shape[1:], dtype=torch.float32)
    std = torch.zeros_like(mean)
    for z in range(nz):
        s = x[:, z].to(dev).to(torch.float32).reshape(nt, -1)
        m = s.mean(0)
        c = s - m
        if q is not None:
            c = c - q @ (q.T @ c)
            sd = c.pow(2).sum(0).div(nt - 1 - q.shape[1]).sqrt()
        else:
            sd = c.pow(2).mean(0).sqrt()
        mean[z] = m.reshape(mean.shape[1:]).cpu()
        std[z] = sd.reshape(mean.shape[1:]).cpu()
    return mean, std


def _masked_traces(x, mask: np.ndarray, dev) -> tuple[np.ndarray, np.ndarray]:
    """Per-TR sum of squared TR-to-TR differences and per-TR signal sum, in the mask."""
    import torch

    nt, nz = x.shape[0], x.shape[1]
    dsq = torch.zeros(max(nt - 1, 0), dtype=torch.float64, device=dev)
    gsum = torch.zeros(nt, dtype=torch.float64, device=dev)
    sel = torch.from_numpy(mask.reshape(nz, -1))
    for z in range(nz):
        if not bool(sel[z].any()):
            continue
        s = x[:, z].reshape(nt, -1)[:, sel[z]].to(dev).to(torch.float32)
        gsum += s.sum(1, dtype=torch.float64)
        if nt > 1:
            dsq += (s[1:] - s[:-1]).pow(2).sum(1, dtype=torch.float64)
    return dsq.cpu().numpy(), gsum.cpu().numpy()
