"""Breathing and heartbeat from 3D EPI phase: the field sampled once per partition cell.

BETA. Validated only against same-session 2D runs and internal consistency, not against
recordings (wiki: *Partition-encode field sampling in 3D EPI*).

A 3D EPI volume is filled shot by shot along the partition (kz) axis, so each kz line is
acquired at its own time in the TR. A field change between shots puts a phase
``2 pi TE df(t_line)`` on that line, common to the whole line. Transforming the complex image
along the partition axis only (x, y stay in image space) and comparing each line with the
run-mean k-space reads that phase back from reconstructed, coil-combined images:
``c(v, kz) = sum_xy K_v conj(K_ref)``. With two or more echoes the line's field is the TE
slope of its phase, with the intercept dropped, so phase that does not grow with TE (coil
phase, motion) leaves. The lines of one CAIPI cell (``Rz`` rows, finished within S
consecutive shots; Stirnberg & Stoecker 2020) share one acquisition time; the cells step
across the volume, so a run becomes a field trace at ``n_cells / TR`` -- 5-24 Hz on the
OHBMPILOT protocols, against one sample per TR.

On those 8 runs (7 timeable): breathing came out at its true rate where the volume rate
aliases it (two sessions within 0.2/min of same-session 2D runs), and the heartbeat at TR
<= 1.7 s within 2/min of the 2D heart rate; nothing convincing at TR 2.65-3.5 s.
Known limits: GRAPPA has already mixed acquired lines into the filled ones (timing per cell
is blurred), the coil combination is only approximately linear, and the fill order
(ascending or descending in the file's partition index) is chosen from the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy import signal

from fastfuncstuff.memory import partition_fft_volume_chunk

from . import physio

# Rows this close to the k-space centre carry the bulk image change (any intensity or
# phase change of the whole object), not a line's own field.
CENTRE_EXCLUDE = 2
# Window (s) for the slow trend removed before choosing the fill direction.
TREND_S = 60.0


@dataclass
class PartitionTraces:
    """Field change per partition cell, in acquisition order."""

    times: np.ndarray  # (nt, n_cells) acquisition time of each cell (s, from volume 0)
    field: (
        np.ndarray
    )  # (nt, n_cells) field change vs the run mean (Hz; rad/2pi with one echo and no TE)
    halves: np.ndarray  # (2, nt, n_cells) the same from two halves of the image
    rows_per_cell: int
    order: str  # "ascending" / "descending" in the file's (fftshifted) partition index,
    # chosen by boundary continuity (_boundary_ratio)
    order_agrees: bool  # the spectral-line second opinion picked the same order
    boundary_ratio: float  # mean step across volume boundaries / within volumes (1 = seamless)
    centre_offset: float  # time within the TR at which the k-space centre cell was acquired
    centre_cell: int  # its column in ``times``
    sweep_change: np.ndarray  # (nt,) field change across each volume's partition sweep (Hz)
    n_echoes: int
    # (ne, nt, nz) whole-image line products sum_xy K_v conj(K_ref), fftshifted partition
    # index; kept for kz_diagnostic (a few MB at most)
    line_products: np.ndarray | None = None

    @property
    def cell_rate(self) -> float:
        return self.times.shape[1] / float(np.median(np.diff(self.times[:, 0])))


def _phase_map(phase: np.ndarray) -> tuple[float, float]:
    """Linear map of a raw phase series onto [-pi, pi] by its own range (as excitation_traces)."""
    lo, hi = float(phase.min()), float(phase.max())
    if not hi > lo:
        raise ValueError("phase has no range; expected raw scanner units or radians")
    return lo, 2 * np.pi / (hi - lo)


def _line_products(
    mag: np.ndarray, phase: np.ndarray, masks: torch.Tensor, device: torch.device, axis: int
) -> np.ndarray:
    """``sum_xy K_v conj(K_ref)`` per volume and partition line, for each x-y mask.

    Returns (n_masks, nt, nz) complex128. Two passes over the volumes (the reference,
    then the products), chunked by the memory planner."""
    nt = mag.shape[-1]
    nvox = int(np.prod(mag.shape[:3]))
    chunk = partition_fft_volume_chunk(nvox, nt, device)
    lo, scale = _phase_map(phase)

    def kspace(a: int, b: int) -> torch.Tensor:
        m = torch.from_numpy(np.ascontiguousarray(mag[..., a:b], dtype=np.float32)).to(device)
        p = torch.from_numpy(np.ascontiguousarray(phase[..., a:b], dtype=np.float32)).to(device)
        z = torch.polar(m, (p - lo) * scale - np.pi)
        return torch.fft.fftshift(torch.fft.fft(z, dim=axis), dim=axis)

    ref = None
    for a in range(0, nt, chunk):
        s = kspace(a, min(nt, a + chunk)).sum(-1)
        ref = s if ref is None else ref + s
    assert ref is not None
    ref = (ref / nt).conj()
    out = np.zeros((masks.shape[0], nt, mag.shape[axis]), np.complex128)
    for a in range(0, nt, chunk):
        b = min(nt, a + chunk)
        # partition axis moved last of the three spatial axes; masks are over the other two
        prod = (kspace(a, b) * ref[..., None]).movedim(axis, 2)
        out[:, a:b] = torch.einsum("xyzt,mxy->mtz", prod, masks.to(prod.dtype)).cpu().numpy()
        del prod
    return out


def _te_field(c: np.ndarray, tes: np.ndarray | None) -> np.ndarray:
    """Field (Hz) per (volume, line) from per-echo line products ``c`` (ne, nt, nz).

    Two or more echoes: weighted least-squares TE slope with an intercept (weights |c|^2,
    normalised over echoes) -- the intercept takes phase that does not grow with TE. One
    echo: phase / (2 pi TE), or phase / 2 pi without a TE."""
    ph = np.unwrap(np.angle(c), axis=1)
    ph -= ph.mean(1, keepdims=True)
    if c.shape[0] == 1:
        te = float(tes[0]) if tes is not None else 1.0
        return ph[0] / (2 * np.pi * te)
    assert tes is not None
    w = np.abs(c) ** 2
    w /= np.maximum(w.sum(0, keepdims=True), 1e-300)
    t = tes[:, None, None]
    x0, x1, x2 = w.sum(0), (w * t).sum(0), (w * t * t).sum(0)
    y0, y1 = (w * ph).sum(0), (w * ph * t).sum(0)
    return (x0 * y1 - x1 * y0) / np.maximum(x0 * x2 - x1 * x1, 1e-300) / (2 * np.pi)


def _cells(field: np.ndarray, weight: np.ndarray, rows: int) -> np.ndarray:
    """Weighted mean over each cell's rows (fftshifted index), centre rows left out:
    (nt, nz) -> (nt, n_cells)."""
    nz = field.shape[1]
    k = np.arange(nz) - nz // 2
    ncell = max(1, nz // rows)
    cell = np.minimum(np.arange(nz) // rows, ncell - 1)
    use = np.abs(k) >= CENTRE_EXCLUDE
    out = np.zeros((field.shape[0], ncell))
    for j in range(ncell):
        m = use & (cell == j)
        if not m.any():
            m = cell == j
        out[:, j] = field[:, m] @ (weight[m] / weight[m].sum())
    return out


def _boundary_ratio(series: np.ndarray, ncell: int, fs: float) -> float:
    """Mean |step| across volume boundaries over mean |step| within volumes, after the
    slow trend is removed (1 = seamless). A wrong fill order reverses every volume and the
    boundary steps stand out: it chose the expected order on all 8 OHBMPILOT runs, by wide
    margins. It needs content away from the volume Nyquist rate: a trace that is breathing
    alone, near Nyquist, looks the same reversed (simulated: 0.91 vs 0.98); a heartbeat or
    the run's other fluctuations break the tie. :func:`_line_concentration` is the second
    opinion, and the CLI warns when they disagree."""
    win = min(series.size // 4 * 2 + 1, int(TREND_S * fs) | 1)
    x = series - signal.savgol_filter(series, max(win, 5), 2)
    d = np.abs(np.diff(x))
    at = (np.arange(1, x.size) % ncell) == 0
    return float(d[at].mean() / max(d[~at].mean(), 1e-30))


def _line_concentration(series: np.ndarray, fs: float) -> float:
    """Share of the 0.05-2 Hz power in the strongest spectral line (+-2 bins). Stitched in
    the true order a rhythm is one line; reversed, its power scatters over alias images.
    Second opinion only: a rhythm at the volume Nyquist frequency looks the same either
    way, and breathing sat there on two OHBMPILOT runs (where this picked the wrong order
    and a wrong heartbeat followed)."""
    x = signal.detrend(series)
    f, P = signal.periodogram(x * np.hanning(x.size), fs=fs)
    band = (f > 0.05) & (f < min(2.0, 0.45 * fs))
    j = np.flatnonzero(band)[np.argmax(P[band])]
    return float(P[max(j - 2, 0) : j + 3].sum() / P[band].sum())


def partition_field_traces(
    magnitudes: Sequence[np.ndarray],
    phases: Sequence[np.ndarray],
    tr: float,
    rows_per_cell: int,
    echo_times: Sequence[float] | None = None,
    device: torch.device | None = None,
    split_axis: int = 0,
    partition_axis: int = 2,
) -> PartitionTraces:
    """The field change per partition cell, from 3D EPI magnitude and phase.

    Parameters
    ----------
    magnitudes, phases : sequences of (nx, ny, nz, nt), one per echo
        Raw reconstructed series (phase in scanner units or radians).
    tr : float
        Volume TR (s) -- not the per-shot TR some 3D sidecars store.
    rows_per_cell : int
        Partition lines finished together: the partition acceleration Rz of the CAIPI
        pattern (1 without CAIPI or partition acceleration).
    echo_times : sequence of float (s), optional
        Required with more than one echo.
    split_axis : int
        In-plane axis split into two halves for an independent second estimate (the one
        NOT phase-encoded, so a ghost does not put one voxel in both halves).
    """
    device = device or torch.device("cpu")
    ne = len(magnitudes)
    if len(phases) != ne:
        raise ValueError(f"{len(phases)} phase echoes for {ne} magnitude echoes")
    if ne > 1 and (echo_times is None or len(echo_times) != ne):
        raise ValueError(f"{ne} echoes need as many echo_times")
    tes = np.asarray(echo_times, float) if echo_times is not None else None
    shape = magnitudes[0].shape
    nt, nz = shape[-1], shape[partition_axis]
    inplane = [d for d in range(3) if d != partition_axis]
    n0, n1 = shape[inplane[0]], shape[inplane[1]]
    ones = torch.ones((n0, n1), device=device)
    coord = torch.arange((n0, n1)[inplane.index(split_axis)], device=device)
    half = coord < (n0, n1)[inplane.index(split_axis)] // 2
    h0 = (
        half[:, None].expand(n0, n1)
        if inplane.index(split_axis) == 0
        else half[None].expand(n0, n1)
    )
    masks = torch.stack([ones, h0.float(), (~h0).float()])
    c = np.stack(
        [
            _line_products(m, p, masks, device, partition_axis)
            for m, p in zip(magnitudes, phases, strict=True)
        ],
        1,
    )  # (3 masks, ne, nt, nz)
    weight = np.abs(c[0]).mean((0, 1))
    fields = [_cells(_te_field(c[i], tes), weight, rows_per_cell) for i in range(3)]
    ncell = fields[0].shape[1]
    fs = ncell / tr
    orders = {"ascending": np.arange(ncell), "descending": np.arange(ncell)[::-1]}
    ratios = {o: _boundary_ratio(fields[0][:, c].ravel(), ncell, fs) for o, c in orders.items()}
    lines = {o: _line_concentration(fields[0][:, c].ravel(), fs) for o, c in orders.items()}
    order = min(ratios, key=lambda o: ratios[o])
    cols, ratio = orders[order], ratios[order]
    agrees = max(lines, key=lambda o: lines[o]) == order
    times = np.arange(nt)[:, None] * tr + (np.arange(ncell)[None] + 0.5) * tr / ncell
    field, halves = fields[0][:, cols], np.stack([fields[1][:, cols], fields[2][:, cols]])
    centre_cell = int(np.flatnonzero(cols == min((nz // 2) // rows_per_cell, ncell - 1))[0])
    tc = times[0] - times[0].mean()
    sweep = (field - field.mean(1, keepdims=True)) @ tc / (tc @ tc) * tr
    return PartitionTraces(
        times=times, field=field, halves=halves, rows_per_cell=rows_per_cell, order=order,
        order_agrees=agrees, boundary_ratio=ratio, centre_offset=float(times[0, centre_cell]),
        centre_cell=centre_cell,
        sweep_change=sweep, n_echoes=ne, line_products=c[0],
    )  # fmt: skip


@dataclass
class KzDiagnostic:
    """The strongest volume-rate rhythm, read per partition line and echo.

    A field change that is static within a volume would give every kz line one temporal
    phase; a field sampled line by line across the TR gives a phase that advances across
    kz by ~2 pi f tau(kz). Its span, its straightness, and whether the echoes agree (and
    scale with TE) are the evidence that the partition axis carries timing at all."""

    kz: np.ndarray  # (nz,) partition index about the k-space centre
    freqs: np.ndarray  # (nf,) Hz, volume-rate spectrum
    spectrum: np.ndarray  # (nf,) last echo's high-passed line phase, |c|-weighted over kz
    alias_hz: float  # its strongest peak above 0.02 Hz
    amplitude: np.ndarray  # (nz, ne) complex: each line's component at alias_hz
    highpassed: np.ndarray  # (nt, nz) last echo's high-passed line phase (rad)
    phase_span_rad: float  # unwrapped angle change of the last echo across the inner lines
    nonlinearity: float  # share of the inner lines' complex A(kz) a straight line misses
    echo_agreement: float  # |mean exp(i (angle A_e(n-1) - angle A_e(n)))|, inner lines
    amp_ratio: list[float]  # median |A_e| / |A_last| per echo (TE ratio when it is field)
    peak_over_median: float
    inner: np.ndarray  # (nz,) bool: 2 <= |kz| <= 0.4 nz, the lines the numbers use


def kz_diagnostic(pt: PartitionTraces, tr: float) -> KzDiagnostic:
    """The partition-axis evidence behind the cell timing (the 3D research figure's
    numbers). Needs ``pt.line_products``. Inner lines only: the centre rows carry the
    bulk image change and the outermost ones little signal."""
    if pt.line_products is None:
        raise ValueError("PartitionTraces has no line_products")
    c = np.moveaxis(pt.line_products, 0, -1)  # (nt, nz, ne)
    nt, nz, ne = c.shape
    el = ne - 1
    k = np.arange(nz) - nz // 2
    ph = np.unwrap(np.angle(c), axis=0)
    ph -= ph.mean(0)
    win = min(31, (nt // 4) * 2 + 1)
    hp = ph - signal.savgol_filter(ph, max(win, 5), 2, axis=0)
    f = np.fft.rfftfreq(nt, tr)
    P = np.abs(np.fft.rfft(hp * np.hanning(nt)[:, None, None], axis=0)) ** 2
    w = np.abs(c).mean(0)
    pk = P[:, :, el] @ (w[:, el] / w[:, el].sum())
    band = f > 0.02
    fa = float(f[band][np.argmax(pk[band])])
    A = (hp * np.exp(-2j * np.pi * fa * np.arange(nt) * tr)[:, None, None]).sum(0)
    inner = (np.abs(k) <= int(0.4 * nz)) & (np.abs(k) >= CENTRE_EXCLUDE)
    a = A[inner, el]
    X = np.stack([np.ones(inner.sum()), k[inner]], 1).astype(complex)
    res = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    nonlin = float((np.abs(res) ** 2).sum() / max((np.abs(a - a.mean()) ** 2).sum(), 1e-300))
    ang = np.unwrap(np.angle(a))
    prev = A[inner, max(el - 1, 0)]
    return KzDiagnostic(
        kz=k, freqs=f, spectrum=pk, alias_hz=fa, amplitude=A, highpassed=hp[:, :, el],
        phase_span_rad=float(ang[-1] - ang[0]), nonlinearity=nonlin,
        echo_agreement=float(np.abs(np.mean(np.exp(1j * np.angle(prev * a.conj()))))),
        amp_ratio=[float(np.median(np.abs(A[inner, e]) / np.abs(a))) for e in range(ne)],
        peak_over_median=float(pk[band].max() / np.median(pk[band])), inner=inner,
    )  # fmt: skip


def excitation_view(pt: PartitionTraces, n_slices: int) -> physio.ExcitationTraces:
    """The cell trace as ``physio.ExcitationTraces``, so rates, phases and regressors come
    from the same code as 2D. Every slice is assigned the k-space centre cell: a 3D voxel
    is a k-space mixture over the whole volume whose contrast is set by the low partition
    lines, so the centre's acquisition time is the instant it stands for."""
    group = np.full(n_slices, pt.centre_cell)
    return physio.ExcitationTraces(
        times=pt.times, group=group, magnitude=pt.field, phase=pt.field, n_voxels=0
    )


def _uniform_series(times: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return physio._uniform(*physio._sorted_series(times, x))


def breathing_cycle(pt: PartitionTraces, order: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """(trace with the breathing cycle removed, breathing phase per cell, breathing rate
    per minute). Breathing and the heartbeat share one field trace here, and the breathing
    waveform's harmonics fall in the cardiac band (2 x 20/min = 40/min took the cardiac
    track on B20R1); its cycle, cos/sin of the phase to ``order``, is fitted and removed
    first, as the 2D detector does per voxel."""
    t, x = _uniform_series(pt.times, pt.field)
    rt, rf, _ = physio.peak_rate(t, x, physio.RESP_BAND)
    phi = physio.cycle_phase(t, x, np.interp(t, rt, rf))
    ph = np.interp(pt.times.ravel(), t, phi).reshape(pt.times.shape)
    cols = [np.ones(ph.size)] + [
        f(m * ph.ravel()) for m in range(1, order + 1) for f in (np.cos, np.sin)
    ]
    X = np.stack(cols, 1)
    y = pt.field.ravel()
    clean = (y - X @ np.linalg.lstsq(X, y, rcond=None)[0]).reshape(pt.field.shape)
    return clean, ph, float(np.median(rf) * 60)


def _peak_prominence(t: np.ndarray, x: np.ndarray, band, avoid) -> tuple[float, float]:
    """(frequency, prominence) of the strongest whole-run peak in ``band`` outside ``avoid``:
    peak power over the median power of the allowed band."""
    fs = physio._fs(t)
    xb = physio._bandpass(x, band[0] * 0.8, min(band[1] * 1.2, 0.45 * fs), fs)
    f, P = signal.periodogram(xb * np.hanning(xb.size), fs=fs, nfft=4 * xb.size)
    ok = (f >= band[0]) & (f <= band[1])
    for fa in avoid:
        fc, hw = fa if isinstance(fa, tuple) else (fa, 0.03)
        ok &= np.abs(f - fc) > hw
    if not ok.any():
        return float("nan"), 0.0
    j = np.flatnonzero(ok)[np.argmax(P[ok])]
    return float(f[j]), float(P[j] / np.median(P[ok]))


@dataclass
class Cardiac3D:
    """Heartbeat from the cell trace, judged by its cell timing."""

    rate_per_min: float
    found: bool
    prominence: float  # strongest cardiac-band peak at the true cell times
    shuffled_max: float  # its maximum over the derangements of the cell times
    phase: np.ndarray  # (nt, n_cells) cycle phase
    trace: np.ndarray  # (nt, n_cells) breathing-removed trace it was read from
    n_shuffles: int
    # False when the 1/TR lines and their breathing sidebands leave none of the cardiac band
    # open (TR 3.5 s: lines 0.29 Hz apart); found is then False and the rate NaN
    testable: bool = True


def cardiac(pt: PartitionTraces, tr: float, n_shuffles: int = 99, seed: int = 0) -> Cardiac3D:
    """The heartbeat in the breathing-removed cell trace, and whether it is real.

    A heartbeat lines up across cells only under their true acquisition times: stitched
    in the right order its line is coherent, with the cells' times deranged (one
    permutation without fixed points, the same in every volume) it smears. Anything the
    same for every cell of a volume -- the 1/TR comb, volume-locked drift -- is unchanged
    by the derangement. For each derangement the whole search is repeated (strongest
    in-band peak, its prominence), so a noise peak chosen at the true times gets no
    advantage. Found = the true times beat every derangement (the 2D judge's logic;
    wiki: Slice-time alignment of the heartbeat)."""
    clean, _, resp_rate = breathing_cycle(pt)
    avoid = physio.cardiac_avoid(tr, resp_rate / 60)
    t, x = _uniform_series(pt.times, clean)
    f0, prom = _peak_prominence(t, x, physio.CARDIAC_BAND, avoid)
    if np.isnan(f0):
        return Cardiac3D(
            rate_per_min=float("nan"), found=False, prominence=0.0, shuffled_max=float("nan"),
            phase=np.zeros_like(pt.times), trace=clean, n_shuffles=0, testable=False,
        )  # fmt: skip
    rng = np.random.default_rng(seed)
    nt, nc = pt.times.shape
    offs = pt.times[0] - pt.times[0, 0]
    null = []
    for _ in range(n_shuffles):
        perm = rng.permutation(nc)
        while np.any(perm == np.arange(nc)):
            perm = rng.permutation(nc)
        tt = pt.times[:, :1] + offs[perm][None]
        ts, xs = _uniform_series(tt, clean)
        null.append(_peak_prominence(ts, xs, physio.CARDIAC_BAND, avoid)[1])
    tc, fr, _ = physio.track_rate(t, x, physio.CARDIAC_BAND, 20.0, avoid=avoid)
    phi = physio.cycle_phase(t, x, np.interp(t, tc, fr))
    phase = np.interp(pt.times.ravel(), t, phi).reshape(pt.times.shape)
    nmax = float(max(null)) if null else float("nan")
    return Cardiac3D(
        rate_per_min=float(np.median(fr) * 60), found=bool(prom > nmax), prominence=prom,
        shuffled_max=nmax, phase=phase, trace=clean, n_shuffles=n_shuffles,
    )  # fmt: skip
