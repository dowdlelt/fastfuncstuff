"""RF spike (k-space spike) detection and repair from magnitude images.

A spike during one slice's readout is a single bright k-space sample. In the
image it becomes a herringbone over the whole slice, in one volume only. The
image is all we have here (magnitude, no raw k-space), but its 2D FFT is a
usable pseudo-k-space: the herringbone folds back into a compact blob (plus its
Hermitian mirror, because a magnitude image is real), sitting on a 1/f-shaped
spectrum that is otherwise very stable over time at every k-point.

Detection, per slice:
    1. log|FFT2| of every volume.
    2. Remove slow change: subtract a running temporal median at each k-point,
       then each volume's median over k. The second step absorbs a global
       intensity change (non-steady-state volumes, drift), which is a constant
       offset in log space.
    3. Robust z per k-point over time (median/MAD).
    4. 3x3 box over k. A spike is a coherent blob of adjacent k-points; noise
       is single-pixel speckle.
    5. Re-standardise each frame's blob map by its own median/MAD over k.
       Motion and inflow raise z over a broad area of the frame (an edge is a
       line through the k centre), a spike raises one blob, so this step is
       what keeps the start-of-run motion volume from flagging every slice.
    6. Max over k outside a central ellipse (``exclude_radius`` of k-max,
       where motion and global signal change live, ~5% of k-space at 0.25),
       then a robust z of that max against the same slice's other volumes.
       A (slice, volume) above the threshold is a hit.

Sensitivity floor: the log compresses a spike to log(amplitude / noise), so a
sharp, weak spike (under ~10x the per-k noise of the magnitude image) does
not clear a 3x3 blob score. Spikes in real magnitude data arrive spread into
blobs by the reconstruction, which is what the blob score is built for. A
linear-ratio residual was tried: it lifts spikes and noise tails alike and
separated the known true/false hits worse (1.9x vs 2.4x).

Repair (``clean``, method ``kclip``): for a hit frame, take its complex
spectrum's deviation from the same slice's neighbours (linear in time between
the nearest non-hit volumes) and clip that deviation, k-point by k-point, to
``clip`` times the k-point's normal deviation (the median over the run of
``|K[t] - (K[t-1] + K[t+1]) / 2|``), phase kept. A k-point behaving normally
is left alone; one the spike inflated is pulled back to the normal range,
wherever it is. A blob mask does not do this: magnitude reconstruction
multiplies the herringbone by the head, which smears each spike into a band
around the blob and a streak through the centre that sit below any blob
threshold (a blob patch removed ~55% of the excess power on the test runs,
clipping at 1.5 ~100%, leaving hit frames at the noise of a normal frame).
DC and its eight neighbours are never touched. ``slice`` replaces the whole
slice by the same temporal interpolation: complete, but it also flattens the
frame's real fluctuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

DEFAULT_THRESHOLD = 10.0
DEFAULT_WINDOW = 9
DEFAULT_EXCLUDE_RADIUS = 0.25
DEFAULT_CLIP = 1.5
REPAIR_METHODS = ("kclip", "slice")


@dataclass
class SpikeScan:
    """Per-(slice, volume) scores; arrays are (n_slices, n_volumes)."""

    score: np.ndarray  # slice-normalised blob score; the thresholded quantity
    raw: np.ndarray  # max 3x3 blob z before slice normalisation
    k_y: np.ndarray  # peak k offset from centre, rows (ny axis)
    k_x: np.ndarray  # peak k offset from centre, columns (nx axis)
    threshold: float
    hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))  # (slice, vol)
    repair: dict[tuple[int, int], str] = field(default_factory=dict)  # (slice, vol) -> method
    clipped_frac: dict[tuple[int, int], float] = field(default_factory=dict)  # k-points shrunk
    residual: np.ndarray | None = None  # (slice, vol) still above threshold after cleaning
    residual_max: float = 0.0


def _running_median_t(x: Tensor, window: int) -> Tensor:
    """Median over a centred temporal window along dim 0, edges replicated."""
    h = window // 2
    t = x.shape[0]
    if t < 3:
        return x.median(dim=0, keepdim=True).values.expand_as(x)
    h = min(h, (t - 1) // 2)
    xp = torch.cat([x[:1].expand(h, *x.shape[1:]), x, x[-1:].expand(h, *x.shape[1:])], dim=0)
    # unfold puts the window last; median materialises (T, ..., W) once per chunk
    return xp.unfold(0, 2 * h + 1, 1).median(dim=-1).values


def _blob(z: Tensor) -> Tensor:
    """3x3 box over the last two (k) axes with wraparound, scaled to unit variance."""
    lead = z.shape[:-2]
    z4 = z.reshape(-1, 1, *z.shape[-2:])
    z4 = F.pad(z4, (1, 1, 1, 1), mode="circular")
    return (F.avg_pool2d(z4, 3, stride=1) * 3.0).reshape(*lead, *z.shape[-2:])


def _center_mask(ny: int, nx: int, radius: float, device: torch.device) -> Tensor:
    """True inside the central ellipse of ``radius`` (fraction of k-max per axis)."""
    yy = (torch.arange(ny, device=device) - ny // 2) / (ny / 2)
    xx = (torch.arange(nx, device=device) - nx // 2) / (nx / 2)
    return (yy[:, None] ** 2 + xx[None, :] ** 2) < radius**2


def _kspace_z(slab: Tensor, window: int, exclude_radius: float) -> tuple[Tensor, Tensor]:
    """(spectrum, blob z) for a (T, S, ny, nx) slab; the spectrum is fftshift-ed complex.

    Scoring ignores the blob z inside the central ellipse (:func:`_peak`).
    """
    spec = torch.fft.fftshift(torch.fft.fft2(slab), dim=(-2, -1))
    lk = torch.log(spec.abs() + 1e-3)
    r = lk - _running_median_t(lk, window)
    del lk
    r = r - r.flatten(-2).median(dim=-1).values[..., None, None]
    mad = r.abs().median(dim=0, keepdim=True).values * 1.4826 + 1e-6
    z = r / mad
    del r, mad
    ny, nx = z.shape[-2:]
    center = _center_mask(ny, nx, exclude_radius, z.device)
    zb = _blob(z)
    del z
    outer = zb[..., ~center]
    med = outer.median(dim=-1).values
    # floor the frame spread so a frame quieter than the run cannot inflate its own peak
    spread = (outer - med[..., None]).abs().median(dim=-1).values * 1.4826
    del outer
    zb = (zb - med[..., None, None]) / spread.clamp_min(0.3)[..., None, None]
    return spec, zb


def _peak(zb: Tensor, center: Tensor) -> tuple[Tensor, Tensor]:
    """Max blob z and its flat index outside the central ellipse, per (T, S)."""
    return zb.masked_fill(center, float("-inf")).flatten(-2).max(dim=-1)


def _slice_chunk(nt: int, nz: int, ny: int, nx: int, window: int, device: torch.device) -> int:
    from fastfuncstuff.memory import bytes_per_slice_spikehunt, get_available_memory

    per = bytes_per_slice_spikehunt(nt, ny, nx, window)
    avail = get_available_memory(device, empty_cache=False)
    return int(max(1, min(nz, avail // max(1, per))))


def _robust_z_over_t(raw: Tensor) -> Tensor:
    """Robust z of each row of (S, T) against that row's own median/MAD."""
    med = raw.median(dim=1, keepdim=True).values
    mad = (raw - med).abs().median(dim=1, keepdim=True).values * 1.4826
    return (raw - med) / mad.clamp_min(1e-6)


def _neighbours(hit_vols: list[int], nt: int) -> dict[int, tuple[int, int, float]]:
    """For each hit volume: nearest non-hit volume below and above, and the weight on the upper.

    Interpolating between non-hit volumes bridges a run of consecutive hits
    instead of smearing it into itself. A hit at either end of the run copies
    its one neighbour.
    """
    hit_set = set(hit_vols)
    clean_t = np.asarray([t for t in range(nt) if t not in hit_set])
    out: dict[int, tuple[int, int, float]] = {}
    if clean_t.size == 0:
        return out
    for t in hit_vols:
        j = int(np.searchsorted(clean_t, t))
        lo = int(clean_t[j - 1]) if j > 0 else int(clean_t[j])
        hi = int(clean_t[j]) if j < clean_t.size else int(clean_t[j - 1])
        out[t] = (lo, hi, 0.0 if hi == lo else (t - lo) / (hi - lo))
    return out


def _repair_slice(
    spec: Tensor,
    hit_vols: list[int],
    out_slice: Tensor,
    method: str,
    clip: float,
    protect: Tensor,
) -> dict[int, float]:
    """Repair the hit volumes of one slice in ``out_slice`` (T, ny, nx) in place.

    ``spec`` is the slice's (T, ny, nx) fftshift-ed spectrum of the unrepaired
    data. Returns the fraction of k-points each repair changed.
    """
    nt = spec.shape[0]
    nbrs = _neighbours(hit_vols, nt)
    if method == "slice":
        src = out_slice.clone()
        for t, (lo, hi, w) in nbrs.items():
            out_slice[t] = (1.0 - w) * src[lo] + w * src[hi]
        return {t: 1.0 for t in nbrs}
    # normal deviation magnitude per k-point: median over the run of the
    # second-difference residual, which is what a clean frame's deviation from
    # its two neighbours' mean looks like
    normal = (spec[1:-1] - 0.5 * (spec[:-2] + spec[2:])).abs().median(dim=0).values
    limit = clip * normal
    fracs: dict[int, float] = {}
    for t, (lo, hi, w) in nbrs.items():
        ref = (1.0 - w) * spec[lo] + w * spec[hi]
        dev = spec[t] - ref
        mag = dev.abs()
        shrink = torch.where(protect, 1.0, (limit / mag.clamp_min(1e-12)).clamp(max=1.0))
        fracs[t] = float((shrink < 1.0).float().mean())
        k = ref + dev * shrink
        out_slice[t] = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).real.to(
            out_slice.dtype
        )
    return fracs


def hunt_spikes(
    data: Tensor,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    exclude_radius: float = DEFAULT_EXCLUDE_RADIUS,
    clean: bool = False,
    out: Tensor | None = None,
    method: str = "kclip",
    clip: float = DEFAULT_CLIP,
    compute_device: torch.device | None = None,
    progress: bool = True,
) -> SpikeScan:
    """Score every (slice, volume) of a (T, Z, Y, X) magnitude series for RF spikes.

    Slices are taken along Z (the NIfTI third axis, the one ``SliceTiming``
    indexes); the FFT runs over the in-plane (Y, X) axes. ``data`` may live on
    the CPU while ``compute_device`` is a GPU: slabs of slices are streamed.

    With ``clean=True`` the hits are repaired into ``out`` (default: a copy of
    ``data``; pass ``data`` itself to repair in place) with ``method`` (see
    the module docstring). The returned scores are those of the input;
    ``residual`` lists the hits that still score above the threshold when the
    repaired data is rescored.
    """
    if data.ndim != 4:
        raise ValueError(f"hunt_spikes needs (T, Z, Y, X), got {tuple(data.shape)}")
    nt, nz, ny, nx = data.shape
    if nt < 5:
        raise ValueError(f"hunt_spikes needs at least 5 volumes, got {nt}")
    dev = compute_device or data.device
    if method not in REPAIR_METHODS:
        raise ValueError(f"method must be one of {REPAIR_METHODS}, got {method!r}")
    if clean and out is None:
        out = data.clone()

    raw = torch.zeros(nz, nt, dtype=torch.float32)
    k_y = torch.zeros(nz, nt, dtype=torch.int64)
    k_x = torch.zeros(nz, nt, dtype=torch.int64)
    scores = torch.zeros(nz, nt, dtype=torch.float32)
    repair: dict[tuple[int, int], str] = {}
    clipped_frac: dict[tuple[int, int], float] = {}
    residual: list[tuple[int, int]] = []
    residual_max = 0.0
    center = _center_mask(ny, nx, exclude_radius, dev)
    # DC and its immediate neighbours are the slice's mean and slowest shading:
    # never changed
    iy = (torch.arange(ny, device=dev) - ny // 2).abs()
    ix = (torch.arange(nx, device=dev) - nx // 2).abs()
    protect = (iy[:, None] <= 1) & (ix[None, :] <= 1)

    chunk = _slice_chunk(nt, nz, ny, nx, window, dev)
    starts = range(0, nz, chunk)
    for s0 in tqdm(
        starts, desc="spikehunt", unit="slab", leave=True, disable=not progress or len(starts) < 2
    ):
        s1 = min(nz, s0 + chunk)
        # copy=True: on the same device and dtype .to() aliases, and cleaning
        # repairs the slab in place
        slab = data[:, s0:s1].to(dev, torch.float32, copy=clean)
        spec, zb = _kspace_z(slab, window, exclude_radius)
        best, arg = _peak(zb, center)  # (T, S)
        raw_c = best.T.contiguous()  # (S, T)
        sc = _robust_z_over_t(raw_c)
        raw[s0:s1] = raw_c.cpu()
        scores[s0:s1] = sc.cpu()
        k_y[s0:s1] = (arg // nx - ny // 2).T.cpu()
        k_x[s0:s1] = (arg % nx - nx // 2).T.cpu()
        if clean:
            assert out is not None
            flagged = sc > threshold
            by_slice: dict[int, list[int]] = {}
            for s, t in flagged.nonzero().tolist():
                by_slice.setdefault(s, []).append(t)
            for s, vols in by_slice.items():
                fracs = _repair_slice(spec[:, s], vols, slab[:, s], method, clip, protect)
                for t, f in fracs.items():
                    repair[(s0 + s, t)] = method
                    clipped_frac[(s0 + s, t)] = f
                out[:, s0 + s] = slab[:, s].to(out.device, out.dtype)
            if by_slice:
                # rescore only the repaired slices: the others are unchanged
                rows = sorted(by_slice)
                del spec, zb
                spec, zb = _kspace_z(slab[:, rows], window, exclude_radius)
                sc2 = _robust_z_over_t(_peak(zb, center)[0].T.contiguous())
                fl2 = flagged[rows]
                for i, t in (fl2 & (sc2 > threshold)).nonzero().tolist():
                    residual.append((s0 + rows[i], t))
                residual_max = max(residual_max, float(sc2[fl2].max()))
        del slab, spec, zb

    score_np = scores.numpy()
    hits = np.argwhere(score_np > threshold)
    return SpikeScan(
        score=score_np,
        raw=raw.numpy(),
        k_y=k_y.numpy(),
        k_x=k_x.numpy(),
        threshold=threshold,
        hits=hits,
        repair=repair,
        clipped_frac=clipped_frac,
        residual=np.asarray(residual, dtype=int).reshape(-1, 2) if clean else None,
        residual_max=residual_max,
    )


# ---------------------------------------------------------------------------
# Timing: hits -> acquisition events
# ---------------------------------------------------------------------------


@dataclass
class SpikeEvent:
    """One excitation's worth of hits: same volume, same slice acquisition time."""

    volume: int
    time: float  # seconds from the first volume's start (vol * TR + slice time)
    within_tr: float  # slice acquisition time within the TR
    slices: list[int]
    score: float  # max over the event's slices


@dataclass
class AcquisitionWindow:
    tr: float
    slice_times: np.ndarray | None  # (n_slices,) seconds, None when unknown
    ta: float  # readout window per TR (== tr when no slice timing / no gap)
    # volumes dropped before the scanned series (moco -skip_first): reported
    # volume numbers and times count from the scan's first volume
    first_volume: int = 0

    @property
    def gap(self) -> float:
        return max(0.0, self.tr - self.ta)


def acquisition_window(
    tr: float, slice_times: list[float] | np.ndarray | None, first_volume: int = 0
) -> AcquisitionWindow:
    """TR, slice times and the readout window TA.

    TA is the last excitation time plus one excitation spacing (the smallest
    positive step between distinct slice times): the time the last slice group
    is still being read. A sequence with a delay in TR (a silent gap) has
    TA < TR, and a spike in the gap lands in no slice and cannot be seen.
    """
    if slice_times is None or len(slice_times) == 0:
        return AcquisitionWindow(tr=tr, slice_times=None, ta=tr, first_volume=first_volume)
    st = np.asarray(slice_times, dtype=np.float64)
    uniq = np.unique(np.round(st, 6))
    step = float(np.diff(uniq).min()) if uniq.size > 1 else tr
    ta = min(tr, float(uniq.max()) + step)
    return AcquisitionWindow(tr=tr, slice_times=st, ta=ta, first_volume=first_volume)


def group_events(scan: SpikeScan, window: AcquisitionWindow) -> list[SpikeEvent]:
    """Merge hits that share a volume and an excitation time (multiband partners)."""
    st = window.slice_times
    groups: dict[tuple[int, float], list[int]] = {}
    for s, t in scan.hits.tolist():
        key = (t, round(float(st[s]), 4) if st is not None and s < len(st) else float("nan"))
        if st is None:
            key = (t, float(s))  # no timing: every slice is its own event
        groups.setdefault(key, []).append(s)
    events = []
    for (t, _), slices in groups.items():
        within = float(st[slices[0]]) if st is not None and slices[0] < len(st) else 0.0
        vol = t + window.first_volume
        events.append(
            SpikeEvent(
                volume=vol,
                time=vol * window.tr + within,
                within_tr=within,
                slices=sorted(slices),
                score=float(max(scan.score[s, t] for s in slices)),
            )
        )
    events.sort(key=lambda e: e.time)
    return events


def sidecar_for(path: str):
    """The BIDS JSON beside a NIfTI under the same name, or None."""
    import re
    from pathlib import Path

    p = Path(path)
    cand = p.parent / (re.sub(r"\.nii(\.gz|\.zst)?$", "", p.name) + ".json")
    return cand if cand.is_file() else None


def resolve_timing(
    input_path: str, tpattern: str | None, tr_flag: float | None
) -> tuple[float, list[float] | None, str]:
    """(TR, slice times or None, source label) for reporting spike times.

    Slice times come from ``tpattern`` (BIDS JSON or one offset per line), else
    the JSON sidecar beside the input. TR: flag, else the JSON's
    ``RepetitionTime``, else the NIfTI header.
    """
    import json
    from pathlib import Path

    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.processing.slicetime import load_slice_timing

    src = tpattern or sidecar_for(input_path)
    slice_times, json_tr, label = None, None, "none"
    if src is not None:
        src = str(src)
        if src.endswith(".json"):
            meta = json.loads(Path(src).read_text())
            json_tr = meta.get("RepetitionTime")
            if "SliceTiming" in meta:
                slice_times = [float(v) for v in meta["SliceTiming"]]
        else:
            slice_times = load_slice_timing(src)
        label = Path(src).name
    tr = tr_flag or (float(json_tr) if json_tr else None) or get_tr_from_file(input_path)
    return float(tr), slice_times, label


def print_report(
    scan: SpikeScan, events: list[SpikeEvent], acq: AcquisitionWindow, nt: int, indent: str = "  "
) -> None:
    """Terminal summary: readout window, counts, then one line per event."""
    gap = (
        f" ({acq.gap:.3f} s silent gap per TR: spikes there are invisible)"
        if acq.gap > 1e-6
        else ""
    )
    print(f"{indent}TR {acq.tr:.3f} s, readout window TA {acq.ta:.3f} s{gap}")
    if len(scan.hits) == 0:
        print(f"{indent}no spikes above {scan.threshold:g}")
        return
    n_vols = len({e.volume for e in events})
    read_min = nt * acq.ta / 60.0
    print(
        f"{indent}{len(events)} spike events ({len(scan.hits)} slice hits) in {n_vols} of {nt} "
        f"volumes; {len(events) / read_min:.1f} events per minute of readout"
    )
    if acq.slice_times is None:
        print(f"{indent}(no slice timing: times are volume onsets)")
    print(f"{indent}{'vol':>5} {'time_s':>9} {'in_TR_s':>8} {'score':>6}  slices")
    for e in events:
        sl = ",".join(str(x) for x in e.slices)
        print(f"{indent}{e.volume:5d} {e.time:9.3f} {e.within_tr:8.3f} {e.score:6.1f}  {sl}")


def write_hits_tsv(path: str, scan: SpikeScan, window: AcquisitionWindow) -> None:
    """One row per hit (slice, volume), time-sorted."""
    st = window.slice_times
    rows = []
    for s, t in scan.hits.tolist():
        within = float(st[s]) if st is not None and s < len(st) else float("nan")
        vol = t + window.first_volume
        time = vol * window.tr + (within if np.isfinite(within) else 0.0)
        rows.append(
            (
                time,
                f"{vol}\t{s}\t{time:.4f}\t{within:.4f}\t{scan.score[s, t]:.2f}\t{scan.raw[s, t]:.2f}"
                f"\t{scan.k_y[s, t]}\t{scan.k_x[s, t]}\t{scan.repair.get((s, t), 'none')}"
                f"\t{scan.clipped_frac.get((s, t), float('nan')):.4f}",
            )
        )
    rows.sort(key=lambda r: r[0])
    hdr = "volume\tslice\ttime_s\twithin_tr_s\tscore\traw_blob_z\tk_y\tk_x\trepair\tclipped_frac"
    with open(path, "w") as f:
        f.write(hdr + "\n")
        for _, line in rows:
            f.write(line + "\n")


def write_score_tsv(path: str, scan: SpikeScan) -> None:
    """The full score matrix, one row per volume, one column per slice."""
    nz, nt = scan.score.shape
    with open(path, "w") as f:
        f.write("\t".join(f"s{s}" for s in range(nz)) + "\n")
        for t in range(nt):
            f.write("\t".join(f"{scan.score[s, t]:.2f}" for s in range(nz)) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _residual_image(data: Tensor, s: int, t: int, window: int) -> np.ndarray:
    nt = data.shape[0]
    h = window // 2
    lo, hi = max(0, t - h), min(nt, t + h + 1)
    nb = data[lo:hi, s].float()
    return (data[t, s].float() - nb.median(dim=0).values).cpu().numpy()


def plot_spikehunt(
    path: str,
    scan: SpikeScan,
    window_acq: AcquisitionWindow,
    data: Tensor,
    cleaned: Tensor | None = None,
    *,
    title: str = "",
    n_gallery: int = 8,
    window: int = DEFAULT_WINDOW,
) -> None:
    """Score heatmap, per-volume trace, within-TR timing, and a hit gallery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nt = scan.score.shape[1]
    order = np.argsort(scan.score.ravel())[::-1]
    top = [np.unravel_index(i, scan.score.shape) for i in order[:n_gallery]]
    top = [(int(s), int(t)) for s, t in top if scan.score[s, t] > scan.threshold]
    rows_g = 3 if cleaned is not None else 2
    n_g = max(1, len(top))
    fig = plt.figure(figsize=(max(12, 1.5 * n_g), 7.0 + 1.6 * rows_g))
    gs = fig.add_gridspec(3 + rows_g, max(n_g, 4), height_ratios=[2.2, 1.0, 1.2] + [1.2] * rows_g)

    v0 = window_acq.first_volume
    nz = scan.score.shape[0]
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(
        np.clip(scan.score, 0, max(3 * scan.threshold, 1)),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        origin="lower",
        extent=(v0 - 0.5, v0 + nt - 0.5, -0.5, nz - 0.5),
    )
    if len(scan.hits):
        ax.scatter(
            scan.hits[:, 1] + v0,
            scan.hits[:, 0],
            s=14,
            facecolors="none",
            edgecolors="cyan",
            lw=0.7,
        )
    ax.set_ylabel("slice")
    ax.set_title(f"{title}  k-space spike score (circles: > {scan.threshold:g})", fontsize=10)
    # colorbar in an inset outside the axes: fig.colorbar(ax=...) would shrink the
    # heatmap and break its column alignment with the per-volume trace below
    fig.colorbar(im, cax=ax.inset_axes((1.005, 0.0, 0.012, 1.0)))

    ax2 = fig.add_subplot(gs[1, :], sharex=ax)
    ax2.plot(np.arange(nt) + v0, scan.score.max(axis=0), lw=0.8, color="0.2")
    ax2.axhline(scan.threshold, color="tab:red", lw=0.8, ls="--")
    ax2.set_ylabel("max over\nslices")
    ax2.set_xlabel("volume")
    ax2.set_xlim(v0 - 0.5, v0 + nt - 0.5)

    ax3 = fig.add_subplot(gs[2, :])
    st = window_acq.slice_times
    if st is not None and len(scan.hits):
        within = np.array([st[s] for s, _ in scan.hits.tolist() if s < len(st)])
        bins = np.linspace(0, window_acq.tr, 41)
        ax3.hist(within, bins=bins, color="tab:purple")
        ax3.axvspan(window_acq.ta, window_acq.tr, color="0.85", label="no readout (gap)")
        ax3.set_xlim(0, window_acq.tr)
        if window_acq.gap > 0:
            ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlabel("acquisition time within TR (s)")
    ax3.set_ylabel("hits")

    for j, (s, t) in enumerate(top):
        res = _residual_image(data, s, t, window)
        v = float(np.percentile(np.abs(res), 99.5)) or 1.0
        a = fig.add_subplot(gs[3, j])
        a.imshow(res, origin="lower", cmap="gray", vmin=-v, vmax=v)
        a.set_title(f"s{s} v{t + v0}  {scan.score[s, t]:.0f}", fontsize=7)
        b = fig.add_subplot(gs[4, j])
        b.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(res)))), origin="lower", cmap="magma")
        if cleaned is not None:
            c = fig.add_subplot(gs[5, j])
            rc = _residual_image(cleaned, s, t, window)
            c.imshow(rc, origin="lower", cmap="gray", vmin=-v, vmax=v)
            c.set_title(scan.repair.get((s, t), ""), fontsize=7)
            c.set_xticks([])
            c.set_yticks([])
            if j == 0:
                c.set_ylabel("cleaned", fontsize=8)
        for a_ in (a, b):
            a_.set_xticks([])
            a_.set_yticks([])
        if j == 0:
            a.set_ylabel("img - local median", fontsize=8)
            b.set_ylabel("|FFT| of that", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
