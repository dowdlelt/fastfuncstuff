"""Reusable QC checks: each adds figures and metrics to a :class:`QCRecord`.

Checks are by *domain* (motion, an EPI series, an alignment), not by tool, so
ffs_moco, ffs_locomoco and ffs_allineate draw the same picture of the same thing.
Tools call them with arrays already in memory; nothing here reloads a file.

Warning limits are deliberately few and each states its reason. Most metrics
carry none: what counts as bad depends on the acquisition, and the cross-subject
table (``ffs_qc_table``) flags robust outliers among comparable runs instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from . import plots, render, style
from .plots import Trace
from .record import QCRecord, clean_label

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

    for name, trace in (("enorm", enorm), ("fd", fd)):
        if trace is not None:
            rec.add_series(name, np.asarray(trace, float))
    if keep is not None:
        rec.add_series("keep", np.asarray(keep))
    if motion_limit is not None:
        rec.info["motion_limit"] = motion_limit
        rec.info["motion_metric"] = motion_metric

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


_ORDER_WORDS = {
    "pc": "sorted within each group by loading on the first principal time course (PC1)",
    "gs": "sorted within each group by correlation with the global signal",
    "none": "in mask order",
}


def _carpet_caption(order: str, extra: str) -> str:
    return (
        f"Rows: neighbouring voxels (detrended, z-scored) averaged, {_ORDER_WORDS[order]}; "
        "ONE grey scale for the whole image (+/-2 robust SD), so rows whose voxels share a "
        "fluctuation stand out against the averaged-down noise. " + extra
    )


def _pc1_trace(cp) -> list:
    if cp.pc1 is None:
        return []
    return [
        Trace(f"PC1 ({100 * cp.pc1_frac:.0f}% var)", cp.pc1 * np.sqrt(len(cp.pc1)), style.GREEN)
    ]


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
    carpet_order: str = "pc",
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
        cp = plots.carpet_rows(ts, gl, order=carpet_order)
        gcor = cp.gcor
        rec.metric(
            "pc1_frac",
            cp.pc1_frac,
            "",
            "variance fraction of the first principal time course (carpet voxels)",
        )
        rec.metric(
            "gcor",
            gcor,
            "",
            "global correlation: mean correlation between all voxel pairs (AFNI GCOR)",
        )
    else:
        cp = plots.Carpet(np.zeros((1, nt)), [("mask", 0, 1)], None, 0.0, float("nan"))

    # --- figure 1: mean + tSNR montage --------------------------------------
    ras_mean, zooms = render.ras_volume(mean, affine)
    ras_tsnr, _ = render.ras_volume(tsnr, affine)
    ras_mask, _ = render.ras_volume(mask.astype(np.float32), affine)
    ras_mask = ras_mask > 0.5
    grid = render.Grid(ras_mean, zooms, ras_mask, n_cut=7)
    lo, hi = render.window(ras_mean, ras_mask)
    t_p99 = float(np.percentile(ras_tsnr[ras_mask], 99)) if ras_mask.any() else 1.0
    planes: dict[str, np.ndarray] = {}
    for plane in TSNR_PLANES:
        # The mean is stored already windowed (uint8): it is only ever drawn grey.
        m = [render.colorize(s_, lo, hi)[..., 0] for s_ in grid.slices(ras_mean, plane)]
        planes[f"{plane}_mean"] = np.stack(m)
        planes[f"{plane}_tsnr"] = np.stack(grid.slices(ras_tsnr, plane)).astype(np.float16)
        planes[f"{plane}_aspect"] = np.asarray(grid.aspect(plane))
    meta = {
        "label": rec.label,
        "tool": rec.tool,
        "what": what,
        "nvox": nvox,
        "med_tsnr": med_tsnr,
        "t_p99": t_p99,
        "file": rec.figure_path(step, "tsnr").name,
        "letters": {p_: grid.side_letter(p_) for p_ in TSNR_PLANES},
    }
    # A sidecar with exactly what the figure is drawn from, so ffs_qc_report can
    # redraw every run's tSNR on ONE scale once all runs exist (no run knows the
    # others' range when it is written).
    np.savez_compressed(rec.sidecar_path("tsnr"), meta=json.dumps(meta), **planes)
    path = rec.qc_dir / meta["file"]
    render.save_png(tsnr_figure(meta, planes, t_p99), path)
    rec.add_figure(step, "tsnr", "Mean and tSNR", path)

    # --- figure 2: carpet -----------------------------------------------------
    traces = []
    if motion is not None:
        traces.append(Trace(f"{motion_name} (mm)", np.asarray(motion, float), style.VIOLET))
    # DVARS has no value at the first volume; NaN leaves a gap rather than a fake dip.
    traces.append(Trace("DVARS (%)", np.concatenate([[np.nan], dvars[1:]]), style.ORANGE))
    traces.append(Trace("global (%)", gpct, style.AQUA))
    traces += _pc1_trace(cp)
    body = plots.carpet(cp.rows, cp.blocks, traces, tr=tr, keep=keep)
    cap = _carpet_caption(
        carpet_order,
        "Look for: vertical bands (motion, breathing, global changes) that line up with the "
        "traces above; with PC order, a band concentrated at the top of a group is the "
        "structure PC1 captured. Horizontal stripes are bad voxels or slices; a shell band "
        "brighter than the interior during motion is motion.",
    )
    img = render.compose(
        body,
        f"Carpet · {rec.label}",
        f"{rec.tool}  ·  {what} series  ·  rows: " + ", ".join(b[0] for b in cp.blocks),
        [
            (f"GCOR {gcor:.3f}", None),
            (f"mean DVARS {dvars[1:].mean() if nt > 1 else 0:.2f}%", None),
        ],
        cap,
    )
    path = rec.figure_path(step + 1, "carpet")
    render.save_png(img, path)
    rec.add_figure(step + 1, "carpet", "Carpet plot", path)


TSNR_PLANES = ("axial", "coronal", "sagittal")


def tsnr_figure(meta: dict, planes: Mapping[str, np.ndarray], t_hi: float, shared: str = ""):
    """The mean + tSNR montage of one run, tSNR drawn 0..``t_hi``, with its colour bar.

    ``shared`` names the runs the scale is shared with; empty means the run's own
    99th percentile.
    """
    rows_img, letters = [], []
    for plane in TSNR_PLANES:
        asp = tuple(float(a) for a in planes[f"{plane}_aspect"])
        mean_panels = [
            render.panel(np.repeat(m[..., None], 3, axis=2), asp, 220)
            for m in planes[f"{plane}_mean"]
        ]
        tsnr_panels = [
            render.panel(render.colorize(t.astype(np.float32), 0, t_hi, "inferno"), asp, 220)
            for t in planes[f"{plane}_tsnr"]
        ]
        rows_img += [(f"mean\n{plane}", mean_panels), (f"tSNR\n{plane}", tsnr_panels)]
        letters += [meta["letters"][plane]] * 2
    scale = shared or f"this run's 99th percentile ({meta['t_p99']:.0f})"
    bar = render.colorbar(
        render.lut("inferno"),
        0.0,
        t_hi,
        "tSNR",
        ticks=np.linspace(0, t_hi, 5),
        fmt="{:.0f}",
        note=f"0 to {t_hi:.0f}: {scale}",
    )
    body = render.vstack([render.montage(rows_img, side_letters=letters), bar])
    look = (
        "Shared scale: a run darker than its neighbours has lower tSNR, one brighter has higher. "
        if shared
        else ""
    )
    cap = (
        f"tSNR = mean / detrended temporal std, scaled 0 to {t_hi:.0f} ({scale}). {look}"
        "Look for: tSNR highest in cortex and falling off smoothly toward the centre; dark "
        "bands or stripes along slices (slice dropout, multiband leakage), signal loss near "
        "sinuses, or bright rims at the brain edge (motion)."
    )
    return render.compose(
        body,
        f"Mean & tSNR · {meta['label']}",
        f"{meta['tool']}  ·  {meta['what']} series  ·  {meta['nvox']:,} automask voxels",
        [(f"median tSNR {meta['med_tsnr']:.1f}", None)],
        cap,
    )


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
    """Every frame against the stack's median: similarity numbers and a flicker.

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
    # Per frame, for the session wheel: which RUN a poor frame is.
    rec.info["frames"] = "\n".join(labels)
    rec.add_series("corr", np.nan_to_num(corr, nan=0.0))

    # No chart of the correlations: a ranked dot plot said less than the table row
    # and the flicker already do, so the numbers live in the record (and the
    # session wheel reads them per run) and the flicker is the figure.
    title = f"{what or 'Stack'} · {rec.label}"

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
                f"frame {i + 1}/{n}:  {labels[i]}  ·  {n} images that should share one space",
                [(f"r with the median = {corr[i]:.3f}", st)],
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
# Display threshold for GLM maps: the conventional uncorrected voxelwise p that
# afni_proc's review driver and most first-look QC use. Not an inference threshold.
GLM_P_THR = 0.001
# Coef colour range, in the bucket's units (percent signal change when scaled):
# +/-7 holds nearly every real task effect without washing small ones out.
COEF_LIM = 7.0


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
    fstat_dof: tuple[float, float] | None = None,
    underlay=None,
    underlay_affine=None,
    coef=None,
    tstat=None,
    t_dof: float | None = None,
    coef_label: str = "",
    coef_why: str = "",
    p_thr: float = GLM_P_THR,
    coef_lim: float = COEF_LIM,
) -> None:
    """Design-matrix picture, regressor correlations, the omnibus F and one task Coef.

    Correlations are taken after removing each run's drift columns (the fit does
    the same), so two regressors that merely share a slow trend do not look
    collinear. ``fstat`` is the omnibus statistic as a ``(z, y, x)`` array, drawn
    at ``p_thr`` over ``underlay`` (any grid; the stats are resampled onto it).
    ``coef`` / ``tstat`` are one estimate and its t, the Coef drawn where the t
    passes ``p_thr`` two-sided. Missing dofs fall back to the design's own (task
    columns, residual dof), which ignores any -adjust_dof.
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

    # --- figure: omnibus F, thresholded on the anatomy ----------------------
    if fstat is None:
        return
    f = _arr(fstat)
    f = np.nan_to_num(f)
    mask = f != 0
    if not mask.any():
        return
    from scipy import stats as sps

    d1, d2 = fstat_dof if fstat_dof else (max(len(task), 1), max(dof, 1))
    f_thr = float(sps.f.isf(p_thr, d1, d2))
    p99 = float(np.percentile(f[mask], 99))
    frac = float((f[mask] >= f_thr).mean())
    rec.metric("fstat_p99", p99, "", f"99th percentile of {fstat_label} in the fitted voxels")
    rec.metric(
        "fstat_frac_sig",
        frac,
        "",
        f"fraction of fitted voxels with {fstat_label} at p < {p_thr:g} (uncorrected)",
    )
    ov = _Overlay(mask, affine, underlay, underlay_affine)
    f_hi = max(p99, 1.5 * f_thr)
    body = ov.montage(
        f,
        f,
        f_thr,
        f_thr,
        f_hi,
        render.POS_LUT,
        colorbar=render.colorbar(
            render.POS_LUT,
            f_thr,
            f_hi,
            "F",
            ticks=np.linspace(f_thr, f_hi, 5),
            fmt="{:.1f}",
            note=f"F({d1:g}, {d2:g}) shown from p < {p_thr:g} (F >= {f_thr:.2f}) to the 99th "
            f"percentile; weaker voxels hidden",
        ),
    )
    img = render.compose(
        body,
        f"Task response · {rec.label}",
        f"{rec.tool}  ·  {fstat_label} at p < {p_thr:g} uncorrected, on {ov.under_name}",
        [(f"{100 * frac:.1f}% of voxels p<{p_thr:g}", None), (f"F p99 {p99:.1f}", None)],
        "The omnibus F: how much all task regressors together explain, per voxel, shown "
        f"where p < {p_thr:g} (uncorrected) over the anatomy. Look for: clusters in the "
        "regions the task should drive (visual cortex for visual stimuli, motor cortex for "
        "button presses). F everywhere, or a rim along the brain edge, suggests the task is "
        "entangled with motion; nothing anywhere suggests wrong timing or a failed run.",
    )
    path = rec.figure_path(step + 1, "fstat")
    render.save_png(img, path)
    rec.add_figure(step + 1, "fstat", "Omnibus F map", path)

    # --- figure: one Coef, thresholded by its t ------------------------------
    if coef is None or tstat is None:
        return
    c = np.nan_to_num(_arr(coef))
    t = np.nan_to_num(_arr(tstat))
    t_thr = float(sps.t.isf(p_thr / 2, t_dof if t_dof else max(dof, 1)))
    sig = mask & (np.abs(t) >= t_thr)
    rec.info["coef_shown"] = coef_label
    rec.metric(
        "coef_frac_sig",
        float(sig.sum() / mask.sum()),
        "",
        f"fraction of fitted voxels where the shown Coef's |t| passes p < {p_thr:g} (two-sided)",
    )
    lut = render.diverging_overlay_lut()
    body = ov.montage(
        c,
        t,
        t_thr,
        -coef_lim,
        coef_lim,
        lut,
        two_sided=True,
        colorbar=render.colorbar(
            lut,
            -coef_lim,
            coef_lim,
            "Coef",
            ticks=np.linspace(-coef_lim, coef_lim, 5),
            note=f"{coef_label}, shown where |t| >= {t_thr:.2f} (p < {p_thr:g} two-sided, "
            f"dof {t_dof if t_dof else dof:g}); colour saturates at +/-{coef_lim:g}",
        ),
    )
    img = render.compose(
        body,
        f"Task effect · {rec.label}",
        "  ·  ".join(x for x in (rec.tool, coef_label, coef_why) if x),
        [(f"{100 * sig.sum() / mask.sum():.1f}% of voxels |t| sig", None)],
        "One task estimate in its own units (red-yellow positive, blue-cyan negative), drawn "
        f"only where its t passes p < {p_thr:g} two-sided. Look for: the sign you expect "
        "where you expect it (activation positive in the driven region, deactivation in the "
        "default network); large coefficients along the edge or in ventricles are motion or "
        "draining veins, not neurons.",
    )
    path = rec.figure_path(step + 2, "coef")
    render.save_png(img, path)
    rec.add_figure(step + 2, "coef", "Task coefficient", path)


# Carpet voxels are capped by array size, not count: three voxel-by-time arrays
# (data, fit, residual) live at once, and a long concatenated series would make a
# fixed voxel count cost gigabytes. Rows are block averages of ~700 either way.
CARPET_MAX_VOXELS = 30000
CARPET_MAX_ELEMENTS = 24e6
WHEEL_SIDECAR = "wheel"


def _shell_interior(mask: np.ndarray) -> dict[str, np.ndarray]:
    import torch

    from fastfuncstuff.processing.mask import _erode_6conn

    core = _erode_6conn(torch.from_numpy(mask), 2).numpy()
    return {"shell": mask & ~core, "interior": core}


def residual_check(
    rec: QCRecord,
    step: int,
    data,
    flat_index: np.ndarray,
    mask: np.ndarray,
    affine,
    design_info: dict,
    *,
    tr: float | None = None,
    device=None,
    tissue: Mapping[str, np.ndarray] | None = None,
    fstat: np.ndarray | None = None,
    run_labels: list[str] | None = None,
) -> None:
    """What the model removed and what it left: tSNR in vs out, and three carpets.

    ``data`` is the series the fit used, time last, in whatever layout the tool
    holds it; ``flat_index`` is a ``(z, y, x)`` volume giving each voxel's row in
    ``data.reshape(-1, T)``, so nothing is transposed or copied to get here.
    ``mask`` is the fitted voxels on that ``(z, y, x)`` grid. ``tissue`` maps a
    name to a boolean mask (from a segmentation on this grid) for the carpet
    bands; without it the mask splits into shell and interior.

    Residuals are OLS on the shared design (see :mod:`fastfuncstuff.qc.residuals`
    for why not the whitened ones), and both noise levels are dof-corrected.
    """
    from . import residuals as rs

    x = np.asarray(design_info["matrix"], dtype=np.float64)
    n_time = x.shape[0]
    roles = column_roles(design_info)
    drift = [i for i, r in enumerate(roles) if r == "drift"]
    task = [i for i, r in enumerate(roles) if r == "task"]
    nuis = [i for i, r in enumerate(roles) if r != "task"]
    keep = np.ones(n_time, bool)
    good = design_info.get("good_list")
    if good is not None and len(good):
        keep[:] = False
        keep[np.asarray(good, int)] = True
    run_starts = [int(s) for s in (design_info.get("run_starts") or [0])]
    n_runs = len(run_starts)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return
    voxels = np.asarray(flat_index)[mask]

    st = rs.residual_stats(data, voxels, x, drift, run_starts, keep, device)
    t_in, t_res = st.tsnr_in(), st.tsnr_res()
    live = (t_in > 0) & (t_res > 0)
    if not live.any():
        return
    med_in, med_res = float(np.median(t_in[live])), float(np.median(t_res[live]))
    gain = float(np.median(t_res[live] / t_in[live]))
    rec.metric(
        "tsnr_input_median",
        med_in,
        "",
        "median tSNR of the GLM's input (mean / SD after drift only, dof-corrected)",
    )
    rec.metric(
        "tsnr_resid_median",
        med_res,
        "",
        "median tSNR of the residuals (mean / residual SD, dof-corrected)",
    )
    rec.metric(
        "tsnr_gain",
        gain,
        "x",
        "median per-voxel ratio residual tSNR / input tSNR; 1 = the regressors removed "
        "no more variance than their degrees of freedom cost",
    )
    run_in = np.array([np.median(v[live]) for v in st.run_tsnr(st.run_sd_in)])
    run_res = np.array([np.median(v[live]) for v in st.run_tsnr(st.run_sd_res)])
    empty = st.run_kept == 0
    run_in[empty] = np.nan
    run_res[empty] = np.nan

    # --- groups: tissue classes when a segmentation is at hand ---------------
    groups = {k: np.asarray(v, bool) & mask for k, v in (tissue or {}).items()}
    groups = {k: v for k, v in groups.items() if v.any()}
    by_tissue = bool(groups)
    if not groups:
        groups = _shell_interior(mask)
    pos_in_mask = np.full(mask.shape, -1, np.int64)
    pos_in_mask[mask] = np.arange(int(mask.sum()))
    gpos = {k: pos_in_mask[v] for k, v in groups.items()}

    # --- figure: input vs residual tSNR --------------------------------------
    markers = [
        (
            k,
            float(np.median(t_in[p][live[p]])) if live[p].any() else 0.0,
            float(np.median(t_res[p][live[p]])) if live[p].any() else 0.0,
            style.SERIES[i % len(style.SERIES)],
        )
        for i, (k, p) in enumerate(gpos.items())
    ]
    for k, p in gpos.items():
        if live[p].any():
            rec.metric(
                f"tsnr_gain_{clean_label(k)}",
                float(np.median(t_res[p][live[p]] / t_in[p][live[p]])),
                "x",
                f"tsnr_gain within {k}",
            )
    body = plots.density_scatter(
        t_in[live],
        t_res[live],
        "input tSNR (after drift)",
        "residual tSNR (after the full model)",
        markers=markers,
    )
    img = render.compose(
        body,
        f"tSNR before / after the model · {rec.label}",
        f"{rec.tool}  ·  {int(live.sum()):,} fitted voxels  ·  rank {st.rank_full} design "
        f"({st.rank_drift} drift), {st.n_kept} time points",
        [(f"input {med_in:.1f}", None), (f"residual {med_res:.1f}", None), (f"x{gain:.2f}", None)],
        "Each voxel once: tSNR of the series the model was given (drift removed) against tSNR "
        "of what it left. Both SDs are divided by their own degrees of freedom, so the dashed "
        "diagonal is what a model that explains nothing would give, however many columns it "
        "has. Dots are the medians of the named voxel groups. Look for: the cloud lifted off "
        "the diagonal where the regressors earn their place (more in grey matter than white "
        "when it is task, everywhere when it is motion or physiology). A cloud ON the diagonal "
        "means the regressors removed nothing; below it, they cost more than they explain.",
    )
    path = rec.figure_path(step, "tsnr_gain")
    render.save_png(img, path)
    rec.add_figure(step, "tsnr_gain", "tSNR before and after the model", path)

    # --- carpet voxels: one subsample for the carpets and the held-out fit ---
    total = sum(len(p) for p in gpos.values())
    cap = int(min(CARPET_MAX_VOXELS, max(CARPET_MAX_ELEMENTS / max(n_time, 1), 2000)))
    stride = max(1, -(-total // cap))
    sub = {k: p[::stride] for k, p in gpos.items()}
    allpos = np.concatenate(list(sub.values()))
    flat = np.asarray(data).reshape(-1, n_time)
    ts = np.ascontiguousarray(flat[voxels[allpos]], dtype=np.float32).T  # (time, voxels)
    d, f, r = rs.fit_parts(ts, x, drift, keep)
    pct = 100.0 / np.maximum(np.abs(st.mean[allpos]), 1e-12)
    d, f, r = d * pct, f * pct, r * pct
    if fstat is not None:
        score = np.nan_to_num(np.asarray(fstat)[mask][allpos]).astype(np.float32)
        order_words = "sorted within each group by the omnibus F, strongest at the top"
    else:
        sd = r.std(axis=0)
        z = r / np.where(sd > 0, sd, 1.0)
        pc, _ = plots._first_pc(z)
        score = z.T @ pc
        score = -score if score.sum() < 0 else score
        order_words = "sorted within each group by loading on the residual's first PC"
    gl, pos = [], 0
    for k, p in sub.items():
        gl.append((k, np.arange(pos, pos + len(p))))
        pos += len(p)
    cp_d = plots.carpet_rows(d, gl, standardize=False, score=score)
    cp_f = plots.carpet_rows(f, gl, standardize=False, score=score, scale=cp_d.scale)
    cp_r = plots.carpet_rows(r, gl, standardize=False, score=score, scale=cp_d.scale)
    rec.metric(
        "resid_gcor", cp_r.gcor, "", "GCOR of the residuals (mean pairwise voxel correlation)"
    )
    rec.metric(
        "resid_pc1_frac",
        cp_r.pc1_frac,
        "",
        "variance fraction of the residuals' first principal time course",
    )
    traces = [Trace("global (%)", r.mean(axis=1), style.AQUA)]
    traces += _pc1_trace(cp_r)
    body = plots.carpet(
        [("data", cp_d.rows), ("fitted", cp_f.rows), ("residual", cp_r.rows)],
        cp_d.blocks,
        traces,
        tr=tr,
        keep=keep,
        run_starts=run_starts,
    )
    bands = ", ".join(b[0] for b in cp_d.blocks)
    img = render.compose(
        body,
        f"Model carpet · {rec.label}",
        f"{rec.tool}  ·  data = fitted + residual, drift removed  ·  traces: the residual's "
        f"global mean and first PC  ·  rows: {bands}"
        + ("" if by_tissue else "  (no segmentation given)"),
        [(f"residual GCOR {cp_r.gcor:.3f}", None), (f"PC1 {100 * cp_r.pc1_frac:.0f}%", None)],
        f"Three carpets on ONE grey scale (+/-2 x {cp_d.scale:.2f}% signal) and the same rows: "
        f"neighbouring voxels averaged, {order_words}. Units are percent of each voxel's mean, "
        "not z-scores, so a tissue or a panel that fluctuates less is visibly dimmer. Yellow "
        "lines are run boundaries; red ticks are censored time points, whose residual is what "
        "the kept points' fit could not explain. Look for: the task as bands in 'fitted' at "
        "the top of grey matter and NOT in 'residual' (bands left there are task the model "
        "missed: wrong timing or response shape); vertical bands through every tissue in "
        "'residual' (motion, breathing, a global change the nuisance regressors did not "
        "catch); one run brighter than the rest.",
    )
    path = rec.figure_path(step + 1, "resid_carpet")
    render.save_png(img, path)
    rec.add_figure(step + 1, "resid_carpet", "Data, fit and residual carpets", path)

    # --- per run: how well the other runs predict it -------------------------
    run_r2 = np.full(n_runs, np.nan)
    try:
        run_r2 = rs.heldout_r2_by_run(ts.T, x, run_starts, task, nuis, keep, device)
    except Exception as err:  # a summary number must never fail a finished fit
        print(f"  QC: held-out R² per run skipped ({err})")
    labels = list(run_labels or [])
    labels += [f"run-{i + 1:02d}" for i in range(len(labels), n_runs)]
    rec.add_series("run_tsnr_input", np.nan_to_num(run_in, nan=0.0))
    rec.add_series("run_tsnr_resid", np.nan_to_num(run_res, nan=0.0))
    if np.isfinite(run_r2).any():
        rec.add_series("run_heldout_r2", np.nan_to_num(run_r2, nan=0.0))
        rec.metric(
            "heldout_r2_min",
            float(np.nanmin(run_r2)),
            "",
            "lowest per-run held-out R² (other runs' task betas predicting this run, in "
            "the voxels those runs rank in the top 5%)",
        )

    # --- sidecar: everything the across-run wheel draws for this fit ---------
    ras_t, zooms = render.ras_volume(_fill(mask, t_res), affine)
    ras_m, _ = render.ras_volume(mask.astype(np.float32), affine)
    grid = render.Grid(ras_t, zooms, ras_m > 0.5, n_cut=3)
    frac = rec.metrics.get("fstat_frac_sig")
    meta = {
        "label": rec.label,
        "tool": rec.tool,
        "runs": labels[:n_runs],
        "run_kept": [int(v) for v in st.run_kept],
        "run_tsnr_input": [_num(v) for v in run_in],
        "run_tsnr_resid": [_num(v) for v in run_res],
        "run_heldout_r2": [_num(v) for v in run_r2],
        "tsnr_input_median": med_in,
        "tsnr_resid_median": med_res,
        "frac_sig": None if frac is None else frac.value,
        "n_task": len(task),
        "letter": grid.side_letter("axial"),
    }
    np.savez_compressed(
        rec.sidecar_path(WHEEL_SIDECAR),
        meta=json.dumps(meta),
        tsnr=np.stack(grid.slices(ras_t, "axial")).astype(np.float16),
        aspect=np.asarray(grid.aspect("axial")),
    )


def _num(v) -> float | None:
    return float(v) if np.isfinite(v) else None


def _fill(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    vol = np.zeros(mask.shape, np.float32)
    vol[mask] = values
    return vol


class _Overlay:
    """Stat maps on the EPI grid drawn over an anatomy on its own grid.

    The stats are resampled onto the underlay (linear, then thresholded), so the
    picture has the anatomy's resolution. Without an underlay the fit mask is a
    dark silhouette on the stats' own grid.
    """

    def __init__(self, mask, affine, underlay=None, underlay_affine=None, n_cut: int = 7):
        self.affine = np.eye(4) if affine is None else np.asarray(affine, float)
        if underlay is not None:
            self.under = np.nan_to_num(_arr(underlay))
            self.under_aff = (
                self.affine if underlay_affine is None else np.asarray(underlay_affine, float)
            )
            self.under_name = "the anatomy"
        else:
            self.under = None
            self.under_aff = self.affine
            self.under_name = "the fit mask"
        m = self._on_under(mask.astype(np.float32)) > 0.5
        ras_m, zooms = render.ras_volume(m.astype(np.float32), self.under_aff)
        self.ras_mask = ras_m > 0.5
        if self.under is not None:
            ras_u, _ = render.ras_volume(self.under, self.under_aff)
            head = _head_mask(ras_u) | self.ras_mask
            self.grid = render.Grid(ras_u, zooms, head, n_cut=n_cut)
            self.win = render.window(ras_u, head)
        else:
            ras_u = np.where(self.ras_mask, 0.17, 0.0).astype(np.float32)
            self.grid = render.Grid(ras_u, zooms, self.ras_mask, n_cut=n_cut)
            self.win = (0.0, 1.0)
        self.ras_under = ras_u

    def _on_under(self, vol: np.ndarray, order: int = 1) -> np.ndarray:
        if self.under is None:
            return vol
        same = vol.shape == self.under.shape and np.allclose(self.affine, self.under_aff, atol=1e-4)
        if same:
            return vol
        return render.resample_like(vol, self.affine, self.under.shape, self.under_aff, order)

    def montage(self, stat, gate, thr, lo, hi, table, colorbar, two_sided=False):
        """Three planes of ``stat`` over the underlay, painted inside the fit mask where
        ``gate`` passes ``thr`` (``|gate|`` when two-sided), ``colorbar`` beneath."""
        ras_s, _ = render.ras_volume(self._on_under(stat), self.under_aff)
        ras_g = (
            ras_s if gate is stat else render.ras_volume(self._on_under(gate), self.under_aff)[0]
        )
        show = (np.abs(ras_g) if two_sided else ras_g) >= thr
        show = show & self.ras_mask
        rows, letters = [], []
        for plane in ("axial", "coronal", "sagittal"):
            asp = self.grid.aspect(plane)
            panels = []
            for u, s_, sh in zip(
                self.grid.slices(self.ras_under, plane),
                self.grid.slices(ras_s, plane),
                self.grid.slices(show.astype(np.float32), plane),
                strict=True,
            ):
                base = render.colorize(u, *self.win)
                panels.append(
                    render.panel(render.overlay(base, s_, sh > 0.5, lo, hi, table), asp, 220)
                )
            rows.append((plane, panels))
            letters.append(self.grid.side_letter(plane))
        return render.vstack([render.montage(rows, side_letters=letters), colorbar])


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


# ---------------------------------------------------------------------------
# Denoising: what came out should look like noise
# ---------------------------------------------------------------------------


def denoise_check(
    rec: QCRecord,
    step: int,
    before,
    after,
    affine,
    *,
    tr: float | None = None,
    device=None,
    extra_maps: dict | None = None,
    carpet_order: str = "pc",
) -> None:
    """tSNR before and after on one scale, the removed std, and a carpet of what
    was removed.

    ``before`` / ``after`` are ``(t, z, y, x)`` on one grid. The removed series
    (before - after) is formed one slice at a time, so no third copy of the run
    exists. A denoiser that works removes something spatially flat and temporally
    white; structure in the removed carpet (bands, a response that follows the
    task) is signal lost. ``extra_maps`` adds rows (e.g. NORDIC's g-factor).
    """
    import torch

    from fastfuncstuff.processing.mask import automask

    b = before if isinstance(before, torch.Tensor) else torch.from_numpy(np.asarray(before))
    a = after if isinstance(after, torch.Tensor) else torch.from_numpy(np.asarray(after))
    nt = min(b.shape[0], a.shape[0])
    b, a = b[:nt], a[:nt]
    dev = torch.device(device) if device is not None else torch.device("cpu")
    mean_b, std_b = _slice_moments(b, dev)
    mean_a, std_a = _slice_moments(a, dev)
    mask_t = automask(mean_b, device=torch.device("cpu")).cpu().bool()
    mask = mask_t.numpy()
    nz = b.shape[1]
    rem_sd = torch.zeros_like(mean_b)
    for z in range(nz):
        d = (b[:, z].to(dev).float() - a[:, z].to(dev).float()).reshape(nt, -1)
        rem_sd[z] = d.std(0).reshape(rem_sd.shape[1:]).cpu()
    tb = torch.where(std_b > 0, mean_b / std_b, 0).numpy()
    ta = torch.where(std_a > 0, mean_a / std_a, 0).numpy()
    tb, ta = np.where(mask, tb, 0), np.where(mask, ta, 0)
    rem = np.where(mask, rem_sd.numpy(), 0)
    med_b = float(np.median(tb[mask])) if mask.any() else float("nan")
    med_a = float(np.median(ta[mask])) if mask.any() else float("nan")
    rec.metric("tsnr_median_before", med_b, "", "median detrended tSNR before denoising")
    rec.metric("tsnr_median_after", med_a, "", "median detrended tSNR after denoising")
    rec.metric("tsnr_gain", med_a / med_b if med_b > 0 else float("nan"), "x", "after / before")
    sdb = std_b.numpy()
    frac = (
        float(np.median((rem[mask] / np.maximum(sdb[mask], 1e-12)) ** 2))
        if mask.any()
        else float("nan")
    )
    rec.metric(
        "removed_var_frac", frac, "", "median fraction of each voxel's temporal variance removed"
    )

    # Removed-series carpet + GCOR: correlated removal is structure, not noise.
    flat = np.flatnonzero(mask.reshape(-1))
    flat = flat[:: max(1, flat.size // 60000)]
    gcor = float("nan")
    cp = plots.Carpet(np.zeros((1, nt)), [("mask", 0, 1)], None, 0.0, float("nan"))
    if flat.size:
        idx = torch.from_numpy(flat)
        d = (b.reshape(nt, -1)[:, idx].float() - a.reshape(nt, -1)[:, idx].float()).numpy()
        cp = plots.carpet_rows(d, [("removed", np.arange(d.shape[1]))], order=carpet_order)
        gcor = cp.gcor
        rec.metric(
            "removed_pc1_frac",
            cp.pc1_frac,
            "",
            "variance fraction of the removed series' first principal time course; white noise gives ~1/T",
        )
        rec.metric(
            "removed_gcor",
            gcor,
            "",
            "GCOR of the removed series: ~0 for white noise, higher when the removal is structured",
        )
        gtrace = d.mean(axis=1)
    else:
        gtrace = np.zeros(nt)

    ras_b, zooms = render.ras_volume(tb, affine)
    ras_a, _ = render.ras_volume(ta, affine)
    ras_r, _ = render.ras_volume(rem, affine)
    ras_m, _ = render.ras_volume(mask.astype(np.float32), affine)
    grid = render.Grid(ras_b, zooms, ras_m > 0.5, n_cut=7)
    # Before and after share ONE scale: here the change is the point.
    t_hi = float(np.percentile(ras_a[ras_m > 0.5], 99)) if (ras_m > 0.5).any() else 1.0
    r_hi = float(np.percentile(ras_r[ras_m > 0.5], 99)) if (ras_m > 0.5).any() else 1.0
    extra = {k: render.ras_volume(_arr(v), affine)[0] for k, v in (extra_maps or {}).items()}
    rows_img, letters = [], []
    for plane in ("axial", "coronal"):
        asp = grid.aspect(plane)
        for name, vol, lo, hi, cmap in (
            ("tSNR before", ras_b, 0, t_hi, "inferno"),
            ("tSNR after", ras_a, 0, t_hi, "inferno"),
            ("removed SD", ras_r, 0, r_hi, "gray"),
            *((k, v, *render.window(v, ras_m > 0.5), "viridis") for k, v in extra.items()),
        ):
            rows_img.append(
                (
                    f"{name}\n{plane}",
                    [
                        render.panel(render.colorize(s, lo, hi, cmap), asp, 200)
                        for s in grid.slices(vol, plane)
                    ],
                )
            )
            letters.append(grid.side_letter(plane))
    body = render.montage(rows_img, side_letters=letters)
    img = render.compose(
        body,
        f"Denoising · {rec.label}",
        f"{rec.tool}  ·  tSNR rows share one scale (0 to {t_hi:.0f})",
        [
            (f"tSNR {med_b:.1f} → {med_a:.1f}", None),
            (f"removed {100 * frac:.0f}% of variance", None),
        ],
        "Look for: tSNR after brighter everywhere and the removed SD flat across the brain, "
        "highest where g-factor (coil geometry) is worst. Anatomy, vessels or activation visible "
        "in the removed SD means signal was taken out with the noise.",
    )
    path = rec.figure_path(step, "tsnr")
    render.save_png(img, path)
    rec.add_figure(step, "tsnr", "tSNR before/after and removed noise", path)

    body = plots.carpet(
        cp.rows, cp.blocks, [Trace("mean removed", gtrace, style.ORANGE), *_pc1_trace(cp)], tr=tr
    )
    img = render.compose(
        body,
        f"Removed signal · {rec.label}",
        f"{rec.tool}  ·  input minus output, detrended and z-scored per voxel",
        [(f"removed GCOR {gcor:.3f}", None)],
        _carpet_caption(
            carpet_order,
            "What the denoiser took out. Look for: uniform TV-static. Vertical bands, a PC1 "
            "that follows the task, or horizontal stripes mean structured signal was removed "
            "along with the thermal noise.",
        ),
    )
    path = rec.figure_path(step + 1, "removed")
    render.save_png(img, path)
    rec.add_figure(step + 1, "removed", "Removed signal carpet", path)
