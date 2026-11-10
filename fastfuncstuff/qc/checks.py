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


# ---------------------------------------------------------------------------
# Alignment: two images that should now share one space
# ---------------------------------------------------------------------------

# Edge colours: bright steps of two categorical hues, so they read on grey.
EDGE_A = "#eb6834"  # edges of the base, drawn over the source
EDGE_B = "#1baf7a"  # edges of the source, drawn over the base


def _support(vol: np.ndarray) -> np.ndarray:
    """Where an image has signal: automask, or nonzero when the automask is empty
    (a skull-stripped or thin-slab image can starve AFNI's clip level)."""
    import torch

    from fastfuncstuff.processing.mask import automask

    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    try:
        m = automask(t, device=torch.device("cpu")).cpu().numpy().astype(bool)
    except Exception:
        m = np.zeros(vol.shape, bool)
    if m.sum() < 64:
        m = np.isfinite(vol) & (vol > 0)
    return m


def alignment_check(
    rec: QCRecord,
    step: int,
    base,
    moved,
    affine,
    *,
    base_name: str = "base",
    moved_name: str = "source",
    n_cut: int = 6,
    subtitle: str = "",
) -> None:
    """Edges of each image over the other, and an animated flicker between them.

    ``base`` and ``moved`` are 3-D ``(z, y, x)`` arrays on the SAME grid (the
    moved image already resampled onto the base). Cuts are placed inside the moved
    image's coverage, so a slab is checked where it has data, not where the base
    happens to be big.
    """
    b = base.detach().cpu().numpy() if hasattr(base, "detach") else np.asarray(base)
    m = moved.detach().cpu().numpy() if hasattr(moved, "detach") else np.asarray(moved)
    b = np.asarray(b, np.float32)
    m = np.asarray(m, np.float32)
    if b.ndim == 4:
        b = b.mean(0)
    if m.ndim == 4:
        m = m.mean(0)
    mb, mm = _support(b), _support(m)
    both = mb & mm
    dice = float(2 * both.sum() / max(mb.sum() + mm.sum(), 1))
    rec.metric(
        "mask_dice",
        dice,
        "",
        f"Dice overlap of the {base_name} and {moved_name} masks (AFNI's anat/EPI mask Dice)",
    )
    corr = float(np.corrcoef(b[both], m[both])[0, 1]) if both.sum() > 10 else float("nan")
    rec.metric(
        "corr",
        corr,
        "",
        "Pearson r in the overlap; negative is expected across contrasts (T1w vs T2*)",
    )
    cover = float(both.sum() / max(mm.sum(), 1))
    rec.metric(
        "coverage",
        cover,
        "",
        f"fraction of the {moved_name} mask that lands inside the {base_name} mask",
        warn_lt=0.9,
    )

    ras_b, zooms = render.ras_volume(b, affine)
    ras_m, _ = render.ras_volume(m, affine)
    ras_mm, _ = render.ras_volume(mm.astype(np.float32), affine)
    grid = render.Grid(ras_b, zooms, ras_mm > 0.5, n_cut=n_cut)
    ras_both, _ = render.ras_volume(both.astype(np.float32), affine)
    wb = render.window(ras_b, ras_both > 0.5, 1, 99.5)
    wm = render.window(ras_m, ras_both > 0.5, 1, 99.5)
    ca, cb = style.hex_rgb(EDGE_A), style.hex_rgb(EDGE_B)

    rows, letters = [], []
    plain_b, plain_m = [], []
    for plane in ("axial", "coronal", "sagittal"):
        asp = grid.aspect(plane)
        sb, sm = grid.slices(ras_b, plane), grid.slices(ras_m, plane)
        over_m, over_b, pb, pm = [], [], [], []
        for x_b, x_m in zip(sb, sm, strict=True):
            gb = render.colorize(x_b, *wb)
            gm = render.colorize(x_m, *wm)
            pb.append(render.panel(gb, asp, 220))
            pm.append(render.panel(gm, asp, 220))
            e_b, e_m = render.edges_2d(x_b, asp), render.edges_2d(x_m, asp)
            a, c = gm.copy(), gb.copy()
            a[e_b] = ca
            c[e_m] = cb
            # Edges are one pixel wide in data space; nearest-neighbour keeps them crisp.
            over_m.append(render.panel(a, asp, 220, smooth=False))
            over_b.append(render.panel(c, asp, 220, smooth=False))
        rows += [
            (f"{moved_name}\n+ {base_name} edges · {plane}", over_m),
            (f"{base_name}\n+ {moved_name} edges · {plane}", over_b),
        ]
        letters += [grid.side_letter(plane)] * 2
        plain_b.append((f"{plane}", pb))
        plain_m.append((f"{plane}", pm))

    stats = [(f"Dice {dice:.3f}", None), (f"coverage {100 * cover:.1f}%", _status(cover < 0.9))]
    body = render.montage(rows, side_letters=letters)
    cap = (
        f"Orange: edges of the {base_name} drawn on the {moved_name}. Green: edges of the "
        f"{moved_name} on the {base_name}. Rows go axial, coronal, sagittal. Look for: edges "
        "that sit on the matching boundary (ventricles, the brain outline, the cortical "
        "ribbon) in every plane; a consistent offset in one direction is a misalignment, a "
        "local mismatch near sinuses or the brain edge is usually distortion."
    )
    title = f"Alignment · {rec.label}"
    sub = subtitle or f"{rec.tool}  ·  {moved_name} → {base_name}"
    img = render.compose(body, title, sub, stats, cap)
    path = rec.figure_path(step, "edges")
    render.save_png(img, path)
    rec.add_figure(step, "edges", "Alignment edges", path)

    letters3 = [grid.side_letter(p) for p in ("axial", "coronal", "sagittal")]
    frames = []
    for which, rws in ((base_name, plain_b), (moved_name, plain_m)):
        body = render.montage(rws, side_letters=letters3)
        frames.append(
            render.compose(
                body,
                title,
                f"{sub}  ·  showing: {which.upper()}",
                stats,
                f"Animated: alternates {base_name} and {moved_name} on identical cuts. "
                "Look for: anatomy that stays still while only the contrast changes. Any "
                "boundary that jumps between frames is misaligned there. Frame 1 is the "
                f"{base_name} in viewers that do not animate.",
            )
        )
    path = rec.figure_path(step + 1, "flicker")
    render.save_apng(frames, path, 800)
    rec.add_figure(step + 1, "flicker", "Alignment flicker (animated)", path)


# ---------------------------------------------------------------------------
# A stack of images that should all line up (runs, sessions, fieldmaps)
# ---------------------------------------------------------------------------

STACK_CORR_WARN = 0.9  # a frame this far from its peers is misaligned or broken
MAX_STACK_FRAMES = 40


def _head_mask(vol: np.ndarray) -> np.ndarray:
    """Cheap torch-free support mask for display: above 10% of the 98th percentile."""
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        return np.zeros(vol.shape, bool)
    return np.isfinite(vol) & (vol > 0.1 * np.percentile(v, 98))


def stack_check(
    rec: QCRecord,
    step: int,
    frames_zyx: np.ndarray,
    affine,
    labels: list[str],
    *,
    what: str = "",
    n_cut: int = 5,
) -> None:
    """Every frame against the stack's median: a ranked similarity chart and a flicker.

    ``frames_zyx`` is ``(n, z, y, x)``: images a pipeline stage claims to have put
    in one space (autoproc's ``stageNN.QC.*`` stacks). The median frame is the
    reference: robust to one bad frame, and its edges are drawn on every frame of
    the flicker, so a misaligned run is visible even when paused.
    """
    x = np.asarray(frames_zyx, np.float32)[:MAX_STACK_FRAMES]
    n = x.shape[0]
    labels = (list(labels) + [f"#{i}" for i in range(len(labels), n)])[:n]
    ref = np.median(x, axis=0)
    mask = _head_mask(ref)
    corr = np.array(
        [
            float(np.corrcoef(f[mask], ref[mask])[0, 1]) if mask.sum() > 10 else float("nan")
            for f in x
        ]
    )
    worst = int(np.nanargmin(corr)) if np.isfinite(corr).any() else 0
    rec.metric("n_frames", n, "", "images in the stack")
    rec.metric(
        "corr_min",
        float(np.nanmin(corr)),
        "",
        "lowest frame-to-median spatial correlation",
        warn_lt=STACK_CORR_WARN,
    )
    rec.metric(
        "corr_median", float(np.nanmedian(corr)), "", "median frame-to-median spatial correlation"
    )
    rec.info["worst_frame"] = labels[worst]

    # --- ranked chart ---------------------------------------------------------
    from matplotlib.figure import Figure

    order = np.argsort(corr)
    with style.mpl_style():
        fig = Figure(figsize=(style.WIDTH / 100, 0.6 + 0.32 * n), dpi=100)
        ax = fig.add_subplot(111)
        ypos = np.arange(n)
        # A dot plot, not bars: the interesting range is near 1, and a bar needs a
        # zero baseline to be honest.
        colors = [style.STATUS["warn"] if corr[i] < STACK_CORR_WARN else style.BLUE for i in order]
        ax.hlines(ypos, 1.0, corr[order], color=style.GRID, lw=1.2)
        ax.scatter(
            corr[order], ypos, s=46, c=colors, zorder=3, edgecolors=style.SURFACE, linewidths=0.8
        )
        ax.set_yticks(ypos, [labels[i] for i in order], fontsize=9)
        lo = min(float(np.nanmin(corr)) - 0.03, STACK_CORR_WARN - 0.03)
        ax.set_xlim(lo, 1.005)
        ax.set_ylim(n - 0.5, -0.5)  # worst at the top
        ax.axvline(STACK_CORR_WARN, color=style.INK2, lw=1, ls=(0, (4, 3)))
        for yy, i in zip(ypos, order, strict=True):
            ax.annotate(
                f"{corr[i]:.3f}",
                (corr[i], yy),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=style.INK2,
            )
        ax.set_xlabel("spatial correlation with the median frame")
        ax.grid(axis="y", visible=False)
        fig.subplots_adjust(left=0.28, right=0.97, top=0.98, bottom=0.6 / (0.6 + 0.32 * n) + 0.02)
        body = render.figure_image(fig)
    stats = [
        (f"worst {labels[worst]}: {corr[worst]:.3f}", _status(corr[worst] < STACK_CORR_WARN)),
    ]
    title = f"{what or 'Stack'} · {rec.label}"
    img = render.compose(
        body,
        title,
        f"{n} images that should share one space",
        stats,
        "Each dot is one image's spatial correlation with the median of all of them, worst "
        f"first. Look for: one image well below the rest (below {STACK_CORR_WARN}, yellow) — "
        "misaligned, or a broken run. Open the flicker to see where.",
    )
    path = rec.figure_path(step, "ranking")
    render.save_png(img, path)
    rec.add_figure(step, "ranking", f"{what or 'Stack'} similarity ranking", path)

    # --- flicker with the median's edges --------------------------------------
    ras_ref, zooms = render.ras_volume(ref, affine)
    ras_mask, _ = render.ras_volume(mask.astype(np.float32), affine)
    grid = render.Grid(ras_ref, zooms, ras_mask > 0.5, n_cut=n_cut)
    planes = ("axial", "coronal", "sagittal")
    ref_edges = {
        p: [render.edges_2d(s, grid.aspect(p)) for s in grid.slices(ras_ref, p)] for p in planes
    }
    color = style.hex_rgb(EDGE_A)
    letters = [grid.side_letter(p) for p in planes]
    pics = []
    for i in range(n):
        ras_f, _ = render.ras_volume(x[i], affine)
        win = render.window(ras_f, ras_mask > 0.5)
        rows = []
        for p in planes:
            asp = grid.aspect(p)
            panels = []
            for s, e in zip(grid.slices(ras_f, p), ref_edges[p], strict=True):
                g = render.colorize(s, *win)
                g[e] = color
                panels.append(render.panel(g, asp, 200, smooth=False))
            rows.append((p, panels))
        body = render.montage(rows, side_letters=letters)
        st = _status(corr[i] < STACK_CORR_WARN)
        pics.append(
            render.compose(
                body,
                title,
                f"frame {i + 1}/{n}:  {labels[i]}",
                [(f"r = {corr[i]:.3f}", st)],
                "Animated: steps through every image with the MEDIAN image's edges (orange) "
                "fixed on top. Look for: anatomy that stays under the same edges in every "
                "frame; a frame whose anatomy slides out from under them is misaligned.",
            )
        )
    path = rec.figure_path(step + 1, "flicker")
    render.save_apng(pics, path, 600)
    rec.add_figure(step + 1, "flicker", f"{what or 'Stack'} flicker (animated)", path)


# ---------------------------------------------------------------------------
# GLM: the design and whether the task shows up at all
# ---------------------------------------------------------------------------

# 1d_tool.py -show_cormat_warnings starts warning at |r| 0.4; past 0.5 a task
# regressor shares enough with a nuisance one that its beta is partly motion.
TASK_NUISANCE_R_WARN = 0.5


def column_roles(design_info: dict) -> list[str]:
    """'task' / 'nuisance' / 'drift' per design column.

    AFNI's ColumnGroups say it directly (-1 polort, 0 baseline, >0 stimulus);
    designs built in-process carry stim/nuisance index lists instead, and their
    polynomial columns are named ``Run#kPol#j`` as in AFNI.
    """
    import re

    labels = list(design_info.get("column_labels") or [])
    p = int(np.asarray(design_info["matrix"]).shape[1])
    labels += [f"col{i}" for i in range(len(labels), p)]
    groups = design_info.get("column_groups")
    if groups is not None and len(groups) == p:
        return ["drift" if g < 0 else "nuisance" if g == 0 else "task" for g in groups]
    stim = set(design_info.get("stim_indices") or [])
    out = []
    for i, lab in enumerate(labels):
        if re.search(r"Pol#\d+", lab):
            out.append("drift")
        elif i in stim:
            out.append("task")
        else:
            out.append("nuisance")
    return out


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # Blue <-> red through the dark neutral: no hue at zero.
    return LinearSegmentedColormap.from_list("qc_div", [style.BLUE, "#383835", style.RED])


def glm_check(
    rec: QCRecord,
    step: int,
    design_info: dict,
    *,
    fstat=None,
    affine=None,
    fstat_label: str = "Full_Fstat",
) -> None:
    """Design-matrix picture, regressor correlations, and the omnibus F map.

    Correlations are taken after removing each run's drift columns (the fit does
    the same), so two regressors that merely share a slow trend do not look
    collinear. ``fstat`` is the omnibus statistic as a ``(z, y, x)`` array.
    """
    X = np.asarray(design_info["matrix"], dtype=np.float64)
    n, p = X.shape
    labels = list(design_info.get("column_labels") or [])
    labels += [f"col{i}" for i in range(len(labels), p)]
    roles = column_roles(design_info)
    good = design_info.get("good_list")
    keep = np.ones(n, bool)
    if good is not None and len(good):
        keep[:] = False
        keep[np.asarray(good, int)] = True
    Xg = X[keep]
    drift = [i for i, r in enumerate(roles) if r == "drift"]
    task = [i for i, r in enumerate(roles) if r == "task"]
    nuis = [i for i, r in enumerate(roles) if r == "nuisance"]
    dof = int(keep.sum()) - p
    rec.metric("n_timepoints", int(keep.sum()), "TRs", "time points in the fit (after censoring)")
    rec.metric("n_regressors", p, "", "design columns")
    rec.metric("n_task", len(task), "", "task columns")
    rec.metric("dof", dof, "", "residual degrees of freedom", warn_lt=20)
    sd = Xg.std(axis=0)
    nz = sd > 0
    cond = (
        float(np.linalg.cond(Xg[:, nz] / np.where(Xg[:, nz].std(0) > 0, Xg[:, nz].std(0), 1)))
        if nz.any()
        else float("nan")
    )
    rec.metric("cond", cond, "", "condition number of the column-normalised design")

    # Residualise on drift, then correlate task + nuisance columns.
    show = task + nuis
    R = np.zeros((len(show), len(show)))
    if show:
        A = Xg[:, show]
        if drift:
            D = Xg[:, drift]
            A = A - D @ np.linalg.lstsq(D, A, rcond=None)[0]
        s = A.std(axis=0)
        A = (A - A.mean(0)) / np.where(s > 0, s, 1)
        R = (A.T @ A) / max(A.shape[0], 1)
    t_pos = list(range(len(task)))
    n_pos = list(range(len(task), len(show)))
    max_tn = float(np.abs(R[np.ix_(t_pos, n_pos)]).max()) if t_pos and n_pos else 0.0
    rec.metric(
        "task_nuisance_r",
        max_tn,
        "",
        "largest |r| between a task and a nuisance regressor (drift removed)",
        warn_gt=TASK_NUISANCE_R_WARN,
    )
    if len(t_pos) > 1:
        Rt = np.abs(R[np.ix_(t_pos, t_pos)])
        np.fill_diagonal(Rt, 0)
        rec.metric(
            "task_task_r",
            float(Rt.max()),
            "",
            "largest |r| between two task regressors (drift removed)",
        )

    # --- figure: design matrix + correlation matrix --------------------------
    from matplotlib.figure import Figure

    with style.mpl_style():
        fig = Figure(figsize=(style.WIDTH / 100, 7.8), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.25)
        ax = fig.add_subplot(gs[0])
        cols = task + nuis
        M = X[:, cols]
        rng = np.abs(M).max(axis=0)
        M = M / np.where(rng > 0, rng, 1)
        ax.imshow(
            M, aspect="auto", cmap=_diverging_cmap(), vmin=-1, vmax=1, interpolation="nearest"
        )
        ax.grid(False)
        for rs in (design_info.get("run_starts") or [])[1:]:
            ax.axhline(rs - 0.5, color=style.INK2, lw=0.8)
        if len(task):
            ax.axvline(len(task) - 0.5, color=style.INK, lw=1.4)
        if not keep.all():
            bad = np.flatnonzero(~keep)
            ax.plot(
                np.full(bad.size, -0.9), bad, "_", color=style.STATUS["fail"], ms=8, clip_on=False
            )
        if len(cols) <= 40:
            ax.set_xticks(
                range(len(cols)), [labels[i] for i in cols], rotation=70, ha="right", fontsize=7
            )
        else:
            ax.set_xticks([])
        ax.set_ylabel("time point")
        ax.set_title(
            f"design: {len(task)} task | {len(nuis)} nuisance  (+{len(drift)} drift, not shown)"
        )
        ax2 = fig.add_subplot(gs[1])
        if show:
            # The diagonal is 1 by definition; blank it so it does not out-shout the
            # off-diagonal cells that carry the information.
            shown = R.copy()
            np.fill_diagonal(shown, np.nan)
            cm = _diverging_cmap().with_extremes(bad=style.SURFACE)
            ax2.imshow(shown, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
            if len(show) <= 30:
                ax2.set_xticks(
                    range(len(show)), [labels[i] for i in show], rotation=70, ha="right", fontsize=7
                )
                ax2.set_yticks(range(len(show)), [labels[i] for i in show], fontsize=7)
                for a in range(len(show)):
                    for b in range(len(show)):
                        if a != b and abs(R[a, b]) >= 0.4:
                            ax2.text(
                                b,
                                a,
                                f"{R[a, b]:.1f}",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color=style.INK,
                            )
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])
            if t_pos and n_pos:
                ax2.axhline(len(task) - 0.5, color=style.INK, lw=1.2)
                ax2.axvline(len(task) - 0.5, color=style.INK, lw=1.2)
        ax2.grid(False)
        ax2.set_title("correlation (drift removed); |r| >= 0.4 printed")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.2)
        body = render.figure_image(fig)
    stats = [
        (f"dof {dof}", _status(dof < 20)),
        (f"max task–nuisance |r| {max_tn:.2f}", _status(max_tn > TASK_NUISANCE_R_WARN)),
    ]
    img = render.compose(
        body,
        f"Design · {rec.label}",
        f"{rec.tool}  ·  {n} time points, {p} columns, {len(design_info.get('run_starts') or [0])} run(s)",
        stats,
        "Left: each column scaled to +/-1 (blue negative, red positive), time running down, run "
        "boundaries as lines, task columns left of the bold line; red ticks at the left edge are "
        "censored time points. Right: regressor correlations after drift removal. Look for: task "
        "columns that correlate with motion (betas partly motion), and task columns that "
        "correlate with each other (betas hard to separate).",
    )
    path = rec.figure_path(step, "design")
    render.save_png(img, path)
    rec.add_figure(step, "design", "Design matrix", path)

    # --- figure: omnibus F ---------------------------------------------------
    if fstat is None:
        return
    f = fstat.detach().cpu().numpy() if hasattr(fstat, "detach") else np.asarray(fstat)
    f = np.nan_to_num(np.asarray(f, np.float32))
    mask = f != 0
    if not mask.any():
        return
    p99 = float(np.percentile(f[mask], 99))
    rec.metric("fstat_p99", p99, "", f"99th percentile of {fstat_label} in the fitted voxels")
    ras_f, zooms = render.ras_volume(f, affine)
    ras_m, _ = render.ras_volume(mask.astype(np.float32), affine)
    grid = render.Grid(ras_f, zooms, ras_m > 0.5, n_cut=7)
    rows, letters = [], []
    for plane in ("axial", "coronal", "sagittal"):
        asp = grid.aspect(plane)
        panels = []
        for s, m in zip(grid.slices(ras_f, plane), grid.slices(ras_m, plane), strict=True):
            rgb = render.colorize(s, 0, p99, "inferno")
            # A faint silhouette where the fit ran, so the map has a brain to sit in.
            rgb[(m > 0.5) & (s <= 0.02 * p99)] = style.hex_rgb("#2c2c2a")
            rgb[m <= 0.5] = 0
            panels.append(render.panel(rgb, asp, 220, smooth=False))
        rows.append((plane, panels))
        letters.append(grid.side_letter(plane))
    body = render.montage(rows, side_letters=letters)
    img = render.compose(
        body,
        f"Task response · {rec.label}",
        f"{rec.tool}  ·  {fstat_label}, scaled 0 to its 99th percentile ({p99:.1f})",
        [(f"F p99 {p99:.1f}", None)],
        "The omnibus F: how much all task regressors together explain, per voxel. Look for: "
        "high values in the regions the task should drive (visual cortex for visual stimuli, "
        "motor cortex for button presses). Uniformly high F everywhere, or along the brain "
        "edge, suggests the task is entangled with motion.",
    )
    path = rec.figure_path(step + 1, "fstat")
    render.save_png(img, path)
    rec.add_figure(step + 1, "fstat", "Omnibus F map", path)


# ---------------------------------------------------------------------------
# Distortion correction: a reversed phase-encode pair before and after
# ---------------------------------------------------------------------------


def _arr(x) -> np.ndarray:
    a = x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.asarray(a, np.float32)


def distortion_check(
    rec: QCRecord,
    step: int,
    raw: list,
    corrected: list,
    affine,
    *,
    field_hz=None,
    fold_frac: float | None = None,
    names: tuple[str, str] = ("blip up", "blip down"),
) -> None:
    """Flicker between the two blips, raw rows above corrected rows; the field map.

    In the raw rows the anatomy jumps along the phase-encode axis between frames
    (opposite distortion); in the corrected rows it should stand still. That one
    animation is the whole check, so it comes first.
    """
    r = [_arr(v) for v in raw[:2]]
    c = [_arr(v) for v in corrected[:2]]
    ref = 0.5 * (c[0] + c[1])
    mask = _head_mask(ref)

    def _r(a, b):
        return float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 10 else float("nan")

    r_before, r_after = _r(r[0], r[1]), _r(c[0], c[1])
    rec.metric("pair_r_before", r_before, "", "spatial r between the two blips before correction")
    rec.metric("pair_r_after", r_after, "", "spatial r between the two blips after correction")
    rec.metric(
        "pair_r_gain",
        r_after - r_before,
        "",
        "how much correction brought the pair together (negative = made it worse)",
        warn_lt=0.0,
    )
    if fold_frac is not None:
        rec.metric(
            "fold_frac",
            fold_frac,
            "",
            "fraction of voxels where the warp folds (Jacobian < 0)",
            warn_gt=0.0,
        )
    stats = [
        (f"pair r {r_before:.3f} → {r_after:.3f}", _status(r_after < r_before)),
    ]

    ras_ref, zooms = render.ras_volume(ref, affine)
    ras_mask, _ = render.ras_volume(mask.astype(np.float32), affine)
    grid = render.Grid(ras_ref, zooms, ras_mask > 0.5, n_cut=6)
    planes = ("axial", "sagittal")
    ras_r = [render.ras_volume(v, affine)[0] for v in r]
    ras_c = [render.ras_volume(v, affine)[0] for v in c]
    # One window for both blips of a row: the flicker should change geometry, not
    # brightness.
    w_raw = render.window(np.concatenate([ras_r[0][ras_mask > 0.5], ras_r[1][ras_mask > 0.5]]))
    w_cor = render.window(np.concatenate([ras_c[0][ras_mask > 0.5], ras_c[1][ras_mask > 0.5]]))
    ref_edges = {
        p: [render.edges_2d(s, grid.aspect(p)) for s in grid.slices(ras_ref, p)] for p in planes
    }
    color = style.hex_rgb(EDGE_A)
    frames = []
    for k in range(2):
        rows, letters = [], []
        for p in planes:
            asp = grid.aspect(p)
            for tag, vol, win in (("raw", ras_r[k], w_raw), ("corrected", ras_c[k], w_cor)):
                panels = []
                for s, e in zip(grid.slices(vol, p), ref_edges[p], strict=True):
                    g = render.colorize(s, *win)
                    g[e] = color
                    panels.append(render.panel(g, asp, 200, smooth=False))
                rows.append((f"{tag}\n{p}", panels))
                letters.append(grid.side_letter(p))
        body = render.montage(rows, side_letters=letters)
        frames.append(
            render.compose(
                body,
                f"Distortion correction · {rec.label}",
                f"{rec.tool}  ·  showing: {names[k].upper()}",
                stats,
                "Animated: alternates the two phase-encode directions, with the corrected mean's "
                "edges (orange) fixed on top. Look for: raw rows that jump along the phase-encode "
                "axis between frames (expected) and corrected rows that stand still under the "
                "edges. Corrected rows that still jump, or stretch near the sinuses, mean the "
                "field is wrong there.",
            )
        )
    path = rec.figure_path(step, "pair")
    render.save_apng(frames, path, 700)
    rec.add_figure(step, "pair", "Blip pair flicker (animated)", path)

    if field_hz is None:
        return
    f = _arr(field_hz)
    lim = float(np.percentile(np.abs(f[mask]), 99)) if mask.any() else 1.0
    lim = max(lim, 1e-3)
    rec.metric("field_abs_p99", lim, "Hz", "99th percentile of |field| in the head")
    ras_f, _ = render.ras_volume(f, affine)
    cmap = "RdBu_r"
    rows, letters = [], []
    for p in ("axial", "coronal", "sagittal"):
        asp = grid.aspect(p)
        panels = []
        for s, m, e in zip(
            grid.slices(ras_f, p),
            grid.slices(ras_mask, p),
            [render.edges_2d(x, asp) for x in grid.slices(ras_ref, p)],
            strict=True,
        ):
            g = render.colorize(s, -lim, lim, cmap)
            g[m <= 0.5] = (g[m <= 0.5] * 0.35).astype(np.uint8)
            g[e] = (20, 20, 20)
            panels.append(render.panel(g, asp, 200, smooth=False))
        rows.append((p, panels))
        letters.append(grid.side_letter(p))
    body = render.montage(rows, side_letters=letters)
    img = render.compose(
        body,
        f"Field map · {rec.label}",
        f"{rec.tool}  ·  off-resonance in Hz, blue negative / red positive, +/-{lim:.0f} Hz (99th pct)",
        [(f"|field| p99 {lim:.0f} Hz", None)]
        + (
            [(f"folds {100 * fold_frac:.3f}%", _status(fold_frac > 0))]
            if fold_frac is not None
            else []
        ),
        "Dark lines are the corrected mean's edges; outside the head is dimmed. Look for: a smooth "
        "field that is strongest above the sinuses and ear canals (orbitofrontal, inferior "
        "temporal). Speckle, stripes along slices, or strong field deep in the brain are fit "
        "errors.",
    )
    path = rec.figure_path(step + 1, "field")
    render.save_png(img, path)
    rec.add_figure(step + 1, "field", "Field map", path)
