"""Cardiac and respiratory noise estimated from the fMRI data itself.

A 2-D EPI volume is not one sample of the heartbeat: every excitation is. With
multiband, the slices sharing an excitation are acquired together and the
excitations are spread through the TR, so reading them in acquisition order gives
a physiological trace at ``n_excitations / TR`` -- 17 Hz on an MB5, 85-slice, 1 s
acquisition -- while the volume rate aliases cardiac. Measured on TASKFORCE
against a belt and pulse oximeter (wiki: *Physiological noise from the data*):
respiration reads best from **phase** (the field follows lung volume), cardiac
from **magnitude** (inflow and pulsatility), coherence 0.9-0.99 for both.

Pipeline, each step a reduction or a small batched solve:

1. :func:`excitation_traces` -- one sample per (volume, excitation) from magnitude
   and, when given, phase. Each voxel's own mean and slow trend are removed first,
   so the different baselines of different slices do not stitch into a comb.
2. :func:`track_rate` -- the fundamental frequency over time, per band.
3. :func:`cycle_phase` -- the instantaneous cardiac / respiratory phase.
4. :func:`slice_regressors` -- RETROICOR terms and RVT, evaluated at every
   slice's own acquisition time, in AFNI's slice-major ``slibase`` layout.

The traces are global means over hundreds of thousands of voxels, so a voxel's
own noise contributes a negligible share of the regressor it is later fitted with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import signal

from fastfuncstuff import policy_linalg as plinalg
from fastfuncstuff.stats.task_coupling import _orthonormal_basis
from fastfuncstuff.utils import accum_dtype, linalg_device

from ..glm.core import construct_polynomial_matrix
from ..io.physio_bids import cmrr_volume_zero, read_cmrr_log  # noqa: F401  (moved; old import path)

CARDIAC_BAND = (0.6, 2.0)  # Hz: 36-120 bpm
RESP_BAND = (0.1, 0.6)  # Hz: 6-36 breaths per minute


@dataclass
class ExcitationTraces:
    """Physiological traces at the excitation rate."""

    times: np.ndarray  # (nt, G) acquisition time of each excitation (s, from volume 0)
    group: np.ndarray  # (nslices,) excitation index of each slice, in time order
    magnitude: np.ndarray  # (nt, G) mean fractional change over brain voxels
    phase: np.ndarray | None  # (nt, G) mean phase change (rad) over stable voxels
    n_voxels: int = 0
    # (nt, G, 10) second-order field fit to each excitation's phase-change maps:
    # 1, x, y, z, x2, y2, z2, xy, xz, yz (coordinates in 100 mm units). See
    # field_breathing_trace for why the gradients carry breath depth.
    field_coef: np.ndarray | None = None
    # (nt, G) cardiac trace from the vascular voxels (refine_cardiac), when made
    cardiac: np.ndarray | None = None
    # (nt, G) cardiac phase the detector settled on, when it should be used as is
    # rather than re-tracked from ``cardiac`` (the constant-modulus path)
    cardiac_phase: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def series(self, which: str, uniform: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(times, values) of one trace as a single series in acquisition order.

        ``uniform=True`` resamples it onto an even grid at the run's mean excitation
        rate, which every spectral step needs: excitations are not evenly spaced
        (TASKFORCE: 0.0575 / 0.06 s apart, then a gap before the next volume), and
        treating them as uniform at 1/median spacing (16.7 Hz, true 17.0) put every
        frequency 2 % low -- heart rates 1.3-1.7/min under the reviewed beats.
        """
        vals = {"magnitude": self.magnitude, "phase": self.phase, "weighted": self.cardiac}[which]
        if vals is None:
            raise ValueError(f"no {which} trace on this run")
        order = np.argsort(self.times.ravel(), kind="stable")
        t, x = self.times.ravel()[order], vals.ravel()[order]
        return _uniform(t, x) if uniform else (t, x)


def excitation_groups(slice_times: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Slices sharing an acquisition time form one excitation.

    Returns ``group`` (excitation index per slice, numbered in time order) and the
    excitation times. Times closer than ``tol`` seconds are one excitation, since
    slice timing written to a sidecar is rounded.
    """
    st = np.asarray(slice_times, dtype=float)
    order = np.argsort(st, kind="stable")
    times: list[float] = []
    group = np.zeros(len(st), dtype=int)
    for k in order:
        if not times or st[k] - times[-1] > tol:
            times.append(float(st[k]))
        group[k] = len(times) - 1
    return group, np.array(times)


def _fs(t: np.ndarray) -> float:
    """Mean sampling rate of a series (correct for an uneven one, unlike 1/median)."""
    return (len(t) - 1) / float(t[-1] - t[0])


def _uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x`` resampled linearly onto an even grid at the series' mean rate, same span.
    Columns of a 2-D ``x`` are resampled together."""
    tu = np.linspace(t[0], t[-1], len(t))
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    return tu, np.stack([np.interp(tu, t, c) for c in x.T], 1)


def _slice_rows(a: np.ndarray, k: int, device: torch.device, np_dtype=np.float32) -> torch.Tensor:
    """Slice ``k`` of an (nx, ny, nz, nt) array as a (nx*ny, nt) tensor on ``device``,
    voxels x-fastest (index them with ``mask[:, :, k].reshape(-1, order="F")``).

    Copied from the host in the array's own memory order and reordered on the device:
    a NIfTI series is Fortran-ordered, and gathering a slice into C order on the host
    took 1.7 s per run against 0.24 s this way."""
    nt = a.shape[-1]
    s = a[:, :, k, :]
    if a.flags.f_contiguous:
        host = np.ascontiguousarray(s.reshape(-1, nt, order="F").T, dtype=np_dtype)
        return torch.from_numpy(host).to(device).T
    host = np.ascontiguousarray(s, dtype=np_dtype)
    return torch.from_numpy(host).to(device).permute(1, 0, 2).reshape(-1, nt)


def _put_slice_rows(out: np.ndarray, k: int, rows: torch.Tensor) -> None:
    """Inverse of `_slice_rows`: write (nx*ny, nt) x-fastest rows into ``out[:, :, k, :]``."""
    nx, ny, _, nt = out.shape
    host = rows.T.contiguous().cpu().numpy()
    out[:, :, k, :] = host.reshape(nt, ny, nx).transpose(2, 1, 0)


def _running_mean(z: torch.Tensor, n: int) -> torch.Tensor:
    """Centred running mean of length ``n`` (odd) along the last axis, the window
    shortened at the ends."""
    h = n // 2
    c = torch.cat([torch.zeros_like(z[:, :1]), z.cumsum(1)], 1)
    idx = torch.arange(z.shape[1], device=z.device)
    hi, lo = (idx + h + 1).clamp_max(z.shape[1]), (idx - h).clamp_min(0)
    return (c[:, hi] - c[:, lo]) / (hi - lo).to(z.real.dtype)


def trailing_noise_volumes(mean_per_volume: np.ndarray, frac: float = 0.2) -> int:
    """Count volumes at the end of a run whose mean signal collapses.

    Noise scans acquired for NORDIC (no RF) are appended to the series; they carry
    no physiology and, left in, put a step into every detrended trace.
    """
    ref = float(np.median(mean_per_volume))
    n = 0
    for v in mean_per_volume[::-1]:
        if v < frac * ref:
            n += 1
        else:
            break
    return n


def _drift_degree(nt: int, tr: float) -> int:
    """AFNI's run-length rule for the polynomial drift order."""
    return 1 + int(nt * tr / 150.0)


def excitation_traces(
    magnitude: np.ndarray | Sequence[np.ndarray],
    slice_times: np.ndarray,
    tr: float,
    phase: np.ndarray | Sequence[np.ndarray] | None = None,
    device: torch.device | None = None,
    mask: np.ndarray | None = None,
    stability: float = 0.7,
    comb: bool = True,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_s: float = 60.0,
) -> ExcitationTraces:
    """One physiological sample per (volume, excitation).

    Parameters
    ----------
    magnitude, phase : (nx, ny, nz, nt)
        Raw slices -- before any motion correction or slice timing, so that every
        sample is still at its own acquisition time. ``phase`` may be raw scanner
        units; only its *change* over time is used, so no spatial unwrap is needed.
        Multi-echo: a list of each, one per echo, shortest TE first (see
        :func:`echo_views`).
    slice_times : (nz,)
        Acquisition offset of each slice within the TR (s); slice ``k`` is the
        file's third axis.
    mask : (nx, ny, nz) bool, optional
        Voxels to average. Default: brighter than a quarter of the 98th percentile
        of the mean image (each echo's own image, multi-echo).
    stability : float
        Phase voxels must have a running-mean phasor (``reference_s`` window) whose
        magnitude averages above this: steady enough for a change to be read. Slow
        drift and one-off steps do not count against a voxel.
    reference_s : float
        Window (s) of the running-mean phasor each voxel's phase change is read
        against.

    Notes
    -----
    Magnitude: each voxel's mean and Legendre drift are removed and the residual
    divided by its mean, then averaged over the excitation's voxels. Phase: each
    voxel's change against its own running-mean phasor, ``angle(z_t * conj(zref_t))``,
    drift removed, averaged with weight ``mean magnitude**2`` (phase noise scales as
    1/magnitude). Multi-echo: every echo's voxels join the magnitude average, and
    the phase is read from the middle echo alone. Fitting each voxel's phase change
    across echoes (through the origin, inverse-variance weights) tied the middle echo
    on breathing timing over 24 ds006926 runs and lost breath depth (field RVT vs
    belt) in 16 of them; the first echo alone was worst (the likely reason, untested:
    phase that does not grow with TE, which a fit across TE amplifies on short echoes).
    """
    device = device or torch.device("cpu")
    mags = [magnitude] if isinstance(magnitude, np.ndarray) else list(magnitude)
    phases = [] if phase is None else [phase] if isinstance(phase, np.ndarray) else list(phase)
    if phases and len(phases) != len(mags):
        raise ValueError(f"{len(phases)} phase echoes for {len(mags)} magnitude echoes")
    _, _, nz, nt = mags[0].shape
    group, gtimes = excitation_groups(slice_times)
    G = len(gtimes)
    if mask is None:
        masks = [_bright_mask(m) for m in mags]
    else:
        masks = [mask] * len(mags)
    deg = _drift_degree(nt, tr)
    ref_len = min(nt, max(3, int(round(reference_s / tr)) | 1))
    L = construct_polynomial_matrix(nt, deg, device=device)
    proj = L @ plinalg.pinv(L)  # (nt, nt) projector onto the drift space
    per_slice_m = np.zeros((nt, nz))
    per_slice_p = np.zeros((nt, nz))
    count_m = np.zeros(nz)
    count_p = np.zeros(nz)
    pe = len(mags) // 2  # the echo the phase is read from: the middle one
    if phases:
        # The run's own range onto [-pi, pi], as complex.scale_phase_to_radians does,
        # but with one min/max for every slice so all slices share the mapping.
        lo, hi = float(phases[pe].min()), float(phases[pe].max())
        if not hi > lo:
            raise ValueError("phase has no range; expected raw scanner units or radians")
    nx, ny = mags[0].shape[:2]
    gx, gy = np.meshgrid(
        (np.arange(nx) - nx / 2) * voxel_size[0] / 100.0,
        (np.arange(ny) - ny / 2) * voxel_size[1] / 100.0,
        indexing="ij",
    )
    gx_t = torch.from_numpy(gx.reshape(-1, order="F").astype(np.float32)).to(device)
    gy_t = torch.from_numpy(gy.reshape(-1, order="F").astype(np.float32)).to(device)
    field_A = np.zeros((nz, 10, 10))
    field_b = np.zeros((nz, 10, nt))
    for k in range(nz):
        sels = [torch.from_numpy(mk[:, :, k].reshape(-1, order="F")).to(device) for mk in masks]
        mus, sums = [], []
        for mag_e, sel_e in zip(mags, sels, strict=True):
            if int(sel_e.sum()) < 20:
                mus.append(None)
                continue
            m = _slice_rows(mag_e, k, device)[sel_e]
            mu_e = m.mean(1)
            # Centred before the projection: the constant is in the drift space, so the
            # result is the same, but TF32 matmuls (on by default on CUDA) keep ~3
            # digits, which on raw intensities is the size of the cardiac pulsation.
            mc = m - mu_e[:, None]
            frac = (mc - mc @ proj.T) / mu_e.clamp_min(1e-6)[:, None]
            sums.append((frac.mean(0).cpu().numpy(), float(m.shape[0])))
            mus.append(mu_e)
            del m, mc, frac
        if not sums:
            continue
        count_m[k] = sum(n for _, n in sums)
        per_slice_m[:, k] = sum(x * (n / count_m[k]) for x, n in sums)
        sel, mu = sels[pe], mus[pe]
        if phases and mu is not None:
            p = _slice_rows(phases[pe], k, device)[sel]
            p = (p - lo) * (2 * np.pi / (hi - lo)) - np.pi
            z = torch.polar(torch.ones_like(p), p)
            # Each voxel against its own running-mean phasor, not its whole-run mean:
            # a global phase step (a head movement, a frequency adjustment) left every
            # voxel's run mean below the stability cut and dropped 99.7 % of the brain
            # on one TASKFORCE run. Breathing is far faster than the window and passes.
            zref = _running_mean(z, ref_len)
            keep = zref.abs().mean(1) > stability
            if int(keep.sum()) < 20:
                continue
            zr = zref[keep]
            dphi = torch.angle(z[keep] * torch.conj(zr / zr.abs().clamp_min(1e-6)))
            dphi = dphi - dphi @ proj.T
            w = mu[keep] ** 2
            per_slice_p[:, k] = ((w[:, None] * dphi).sum(0) / w.sum()).cpu().numpy()
            count_p[k] = float(keep.sum())
            # second-order field over this slice's stable voxels (weighted like the mean)
            xk, yk = gx_t[sel][keep], gy_t[sel][keep]
            zk = torch.full_like(xk, (k - nz / 2) * voxel_size[2] / 100.0)
            bas = torch.stack(
                [
                    torch.ones_like(xk),
                    xk,
                    yk,
                    zk,
                    xk * xk,
                    yk * yk,
                    zk * zk,
                    xk * yk,
                    xk * zk,
                    yk * zk,
                ],
                1,
            )
            wn = (w / w.sum())[:, None]
            field_A[k] = (bas.T @ (wn * bas)).cpu().double().numpy() * count_p[k]
            field_b[k] = (bas.T @ (wn * dphi)).cpu().double().numpy() * count_p[k]
    mag_t = np.zeros((nt, G))
    ph_t = np.zeros((nt, G)) if phases else None
    field_t = np.zeros((nt, G, 10)) if phases else None
    if field_t is not None:
        for g in range(G):
            ks = np.flatnonzero((group == g) & (count_p > 0))
            if ks.size == 0:
                continue
            Ag = field_A[ks].sum(0)
            Ag = Ag + 1e-6 * np.trace(Ag) / 10 * np.eye(
                10
            )  # one excitation's few slices barely fix z2
            field_t[:, g] = np.linalg.solve(Ag, field_b[ks].sum(0)).T
    for g in range(G):
        s = group == g
        if count_m[s].sum() > 0:
            mag_t[:, g] = per_slice_m[:, s] @ (count_m[s] / count_m[s].sum())
        if ph_t is not None and count_p[s].sum() > 0:
            ph_t[:, g] = per_slice_p[:, s] @ (count_p[s] / count_p[s].sum())
    if comb:
        mag_t = remove_tr_comb(mag_t, tr)
    times = np.arange(nt)[:, None] * tr + gtimes[None, :]
    return ExcitationTraces(times, group, mag_t, ph_t, int(count_m.sum()), field_coef=field_t)


def _bright_mask(magnitude: np.ndarray) -> np.ndarray:
    """Voxels brighter than a quarter of the mean image's 98th percentile."""
    mean_img = magnitude.mean(-1)
    return mean_img > 0.25 * np.percentile(mean_img[mean_img > 0], 98)


def stack_echoes(
    n_echoes: int, load: Callable[[int], np.ndarray], split_axis: int
) -> tuple[np.ndarray, int]:
    """Echoes side by side in one 4-D array, and the axis they are tiled along.

    The cardiac detector then treats every echo's voxels as more voxels: each echo is
    another reading of the same excitation, and the vessels it shows best differ
    (inflow on the first echo, dephasing on the later ones), so its top voxels are
    drawn from all echoes rather than from one fixed combination. Tiled along the
    in-plane axis the found-a-heartbeat halves test does NOT split, so both halves keep
    their own side of every echo. ``load(e)`` returns echo ``e``; each is copied in and
    dropped, so the peak is the stack plus one echo. Fortran order, as NIfTI is read.
    """
    axis = 1 - split_axis
    first = load(0)
    shape = list(first.shape)
    n = shape[axis]
    shape[axis] = n * n_echoes
    out = np.empty(shape, dtype=first.dtype, order="F")
    idx: list = [slice(None)] * first.ndim
    for e in range(n_echoes):
        a = first if e == 0 else load(e)
        if a.shape != first.shape:
            raise ValueError(f"echo {e + 1} shape {a.shape} differs from echo 1 {first.shape}")
        if (
            not np.can_cast(a.dtype, out.dtype, "same_kind")
            or a.dtype.itemsize > out.dtype.itemsize
        ):
            out = out.astype(np.result_type(out.dtype, a.dtype, np.float32), order="F")
        idx[axis] = slice(e * n, (e + 1) * n)
        out[tuple(idx)] = a
        del a
    return out, axis


def echo_views(stacked: np.ndarray, n_echoes: int, axis: int) -> list[np.ndarray]:
    """The echoes of :func:`stack_echoes` as views (no copy)."""
    n = stacked.shape[axis] // n_echoes
    idx = [slice(None)] * stacked.ndim
    views = []
    for e in range(n_echoes):
        idx[axis] = slice(e * n, (e + 1) * n)
        views.append(stacked[tuple(idx)])
    return views


def field_breathing_trace(traces: ExcitationTraces) -> tuple[np.ndarray, np.ndarray]:
    """A trace whose breath-to-breath amplitude follows breath DEPTH.

    The mean phase gives breathing timing but not depth (breath-by-breath RVT from it
    vs a belt: median r +0.20 after RRF convolution over 91 TASKFORCE runs). Breathing
    changes lung volume, which reaches the head as a field gradient, not just an
    offset; the first principal component of the x/y/z gradient coefficients in the
    respiratory band matched the belt's RVT at median +0.71 (middle half of runs
    +0.56 to +0.79; +0.85 on a breath-hold run). The offset is left out: it also
    carries scanner field drift. Sign is arbitrary (depth is peak minus trough).
    """
    if traces.field_coef is None or traces.phase is None:
        raise ValueError("no field fit: the run was given without phase")
    order = np.argsort(traces.times.ravel(), kind="stable")
    t, G3 = _uniform(
        traces.times.ravel()[order], traces.field_coef[:, :, 1:4].reshape(-1, 3)[order]
    )
    G3 = (G3 - G3.mean(0)) / np.maximum(G3.std(0), 1e-12)
    _, x_ph = traces.series("phase", uniform=True)
    _, rf, _ = peak_rate(t, x_ph, RESP_BAND)
    f0 = float(np.median(rf))
    fs = _fs(t)
    sos = signal.butter(2, [0.6 * f0, min(1.6 * f0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    _, _, vt = np.linalg.svd(signal.sosfiltfilt(sos, G3, axis=0), full_matrices=False)
    return t, G3 @ vt[0]


def remove_tr_comb(trace: np.ndarray, tr: float, cutoff: float = 0.08) -> np.ndarray:
    """Remove the TR-periodic pattern a stitched magnitude trace carries.

    Shared slow fluctuations (drift residue, global signal, task) are seen with a
    different gain by each excitation's slices; stitched in time order that is a
    slow signal times a pattern repeating every TR -- peaks at 1/TR and harmonics,
    right inside the cardiac band at heart rates near 60 bpm. Each excitation's
    trace is regressed on the low-passed volume-rate mean, which carries the slow
    signal but (below ``cutoff`` Hz) none of the physiology.
    """
    trace = np.asarray(trace, dtype=float)
    nt = trace.shape[0]
    if nt < 30:
        return trace
    vol_mean = trace.mean(1)
    b, a = signal.butter(2, cutoff, fs=1.0 / tr)
    slow = signal.filtfilt(b, a, vol_mean)
    slow = slow - slow.mean()
    den = float(slow @ slow)
    if den <= 0:
        return trace
    beta = (slow @ (trace - trace.mean(0))) / den
    return trace - np.outer(slow, beta)


def notch_tr_harmonics(
    t: np.ndarray, x: np.ndarray, tr: float, band: tuple[float, float]
) -> np.ndarray:
    """Notch the k/TR lines that fall inside ``band`` (narrow, Q=30, zero phase).

    :func:`remove_tr_comb` takes out the part of the comb carried by slow signals,
    but a stitched magnitude trace can keep a sharp line at 2/TR that the heart-rate
    tracker locks onto (a TASKFORCE rest run: 47 bpm off). A narrow notch costs a
    heart rate sitting exactly on a TR harmonic a little power and nothing else.
    """
    fs = _fs(t)
    out = np.asarray(x, dtype=float)
    k = 1
    while k / tr <= band[1] * 1.2:
        f0 = k / tr
        if f0 >= band[0] * 0.8 and f0 < 0.45 * fs:
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            out = signal.filtfilt(b, a, out)
        k += 1
    return out


def _bandpass(x: np.ndarray, lo: float, hi: float, fs: float, order: int = 3) -> np.ndarray:
    # second-order sections: as (b, a) a ~1 Hz band at a 10 kHz recording rate (ds003192)
    # has its poles on z = 1 and filtfilt refuses it
    sos = signal.butter(order, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - x.mean())


def track_rate(
    t: np.ndarray,
    x: np.ndarray,
    band: tuple[float, float],
    window: float,
    step: float = 2.0,
    avoid: tuple[float | tuple[float, float], ...] = (),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fundamental frequency over time within ``band``.

    Sliding Hann-windowed periodograms (zero-padded 16x, parabolic peak refinement),
    each window's peak searched near the previous one so a harmonic or a noise
    burst cannot capture the track. Frequencies within 0.03 Hz of any in ``avoid``
    (the TR harmonics) are never chosen; an entry ``(f, half_width)`` sets its own. Returns window centres, frequencies (Hz) and
    the peak prominence (in-band peak power over in-band median, on the whole-run
    spectrum) used to choose between magnitude and phase.
    """
    fs = _fs(t)
    xb = _bandpass(x, band[0] * 0.8, band[1] * 1.2, fs)
    n = int(window * fs)
    s = max(1, int(step * fs))
    f_all, P_all = signal.periodogram(xb * np.hanning(len(xb)), fs=fs, nfft=4 * len(xb))

    def allowed(f: np.ndarray) -> np.ndarray:
        ok = (f >= band[0]) & (f <= band[1])
        for fa in avoid:
            f0, w = fa if isinstance(fa, tuple) else (fa, 0.03)
            ok &= np.abs(f - f0) > w
        return ok

    inb = allowed(f_all)
    prominence = float(P_all[inb].max() / np.median(P_all[inb]))
    f0_run = float(f_all[inb][np.argmax(P_all[inb])])
    cen, est = [], []
    prev = f0_run
    starts = range(0, max(1, len(xb) - n), s)
    m = min(n, len(xb))
    # every window's periodogram in one batched call (each row as the one-window call)
    segs = np.stack([xb[i : i + m] for i in starts]) * np.hanning(m)
    f, P_all_w = signal.periodogram(segs, fs=fs, nfft=16 * m, axis=-1)
    inb_w = (f >= band[0]) & (f <= band[1])
    ok_w = allowed(f)
    for i, P in zip(starts, P_all_w, strict=True):
        near = inb_w & (np.abs(f - prev) <= 0.25 * prev)
        if not near.any():
            near = inb_w
        j = int(np.flatnonzero(near)[np.argmax(P[near])])
        if ok_w[j]:
            fj = float(np.interp(_refine_peak(P, j), np.arange(len(f)), f))
        else:
            # The strongest nearby power sits on a TR harmonic: a comb line, or a heart
            # rate passing through k/TR (60/min at TR 1 s is ordinary). Hold the last
            # estimate. Taking the best allowed bin instead picked the shoulder of the
            # real peak, whose parabola is not around a maximum, and one TASKFORCE track
            # flew from 63 to ~110/min and stayed there.
            fj = prev
        est.append(fj)
        cen.append(float(t[min(i + n // 2, len(t) - 1)]))
        prev = fj
    return np.array(cen), np.array(est), prominence


def _refine_peak(P: np.ndarray, j: int) -> float:
    """Sub-bin position of a spectral peak: a parabola through log power at j-1, j, j+1,
    only where j is a local maximum and at most half a bin. Off a maximum the vertex is
    meaningless; at zero curvature the old unguarded form moved 1e15 bins."""
    if not 0 < j < len(P) - 1:
        return float(j)
    a, b, c = np.log(P[j - 1 : j + 2] + 1e-30)
    den = a - 2 * b + c
    if b < a or b < c or den >= 0:
        return float(j)
    return j + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def peak_rate(
    t: np.ndarray, x: np.ndarray, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cycle-by-cycle rate from peak intervals, for a slow and irregular rhythm.

    A periodogram long enough to resolve breathing (tens of seconds) blurs a rate
    that changes breath to breath and lets slow drift compete at the band's low edge;
    on a TASKFORCE rest run it wandered 5-10 per minute against reviewed belt peaks
    near 10.7. Counting peaks of the trace band-passed around the run's own rate is
    what physio_calc.py does on the belt. Returns (interval mid-times, rates in Hz,
    whole-run peak prominence).

    The run's rate, which sets the band the peaks are counted in, is the peak of the
    derivative's spectrum (power x f^2): a plain argmax let slow power at the band's low
    edge (task blocks, 1/f drift) win, and on two TASKFORCE task runs whose phase trace
    was coherent with the belt (1.00) it reported 6/min against 15-17 true.
    """
    fs = _fs(t)
    xb = _bandpass(x, band[0] * 0.8, band[1] * 1.2, fs)
    f_all, P_all = signal.periodogram(xb * np.hanning(len(xb)), fs=fs, nfft=4 * len(xb))
    inb = (f_all >= band[0]) & (f_all <= band[1])
    f0 = float(f_all[inb][np.argmax((P_all * f_all**2)[inb])])
    prominence = float(P_all[inb].max() / np.median(P_all[inb]))
    xc = _bandpass(x, 0.6 * f0, 1.6 * f0, fs)
    pk, _ = signal.find_peaks(xc, distance=max(1, int(0.6 * fs / f0)), prominence=0.3 * xc.std())
    if pk.size < 3:
        return np.array([t[len(t) // 2]]), np.array([f0]), prominence
    tp = t[pk]
    rate = 1.0 / np.diff(tp)
    rate = signal.medfilt(rate, 3) if rate.size >= 3 else rate  # one missed/extra peak
    return 0.5 * (tp[1:] + tp[:-1]), np.clip(rate, band[0], band[1]), prominence


def cycle_phase(t: np.ndarray, x: np.ndarray, rate_f: np.ndarray) -> np.ndarray:
    """Instantaneous cycle phase (rad, unwrapped) of a physiological trace.

    The trace is band-passed around the run's tracked rate and its analytic signal
    taken; the band is wide enough (0.6-1.6x the median rate) to follow a drifting
    rate without letting in the second harmonic. Used for respiration too: Glover's
    amplitude-histogram phase (AFNI RetroTS's choice) was tried and explained less
    variance on 4 of 5 TASKFORCE runs, and reading the phase earlier (the phase trace
    lags the belt by 1.2-1.7 s) lost variance at every lead tried.
    """
    fs = _fs(t)
    f0 = float(np.median(rate_f))
    xb = _bandpass(x, 0.6 * f0, 1.6 * f0, fs)
    return np.unwrap(np.angle(signal.hilbert(xb)))


def respiration_volume(
    t: np.ndarray, x: np.ndarray, rate_t: np.ndarray, rate_f: np.ndarray
) -> np.ndarray:
    """Respiration volume per time (RVT-like): breath depth times breathing rate.

    Depth is the envelope of the band-passed respiratory trace (Hilbert magnitude),
    rate the tracked fundamental -- the same quantity as Birn's (max - min)/period.
    """
    fs = _fs(t)
    f0 = float(np.median(rate_f))
    xb = _bandpass(x, 0.6 * f0, 1.6 * f0, fs)
    env = np.abs(signal.hilbert(xb))
    # smooth to one breath, so the envelope is a breath-by-breath depth, not a ripple
    b, a = signal.butter(2, min(0.45 * fs, 0.8 * f0), fs=fs)
    env = signal.filtfilt(b, a, env)
    rate = np.interp(t, rate_t, rate_f)
    return env * rate


def _depth_rvt(traces: ExcitationTraces, t_r, x_r, rt, rf) -> np.ndarray:
    """RVT at ``t_r``: breath by breath from the field-gradient trace when the run has
    phase (it follows depth; :func:`field_breathing_trace`), else envelope x rate of the
    respiratory trace."""
    if traces.field_coef is not None and traces.phase is not None:
        tf, xf = field_breathing_trace(traces)
        bt, br = breath_rvt(tf, xf, float(np.median(rf)))
        return np.interp(t_r, bt, br)
    return respiration_volume(t_r, x_r, rt, rf)


def _cardiac_input(traces, card_from, t_c, x_c, tr, f_resp):
    """The cardiac trace as the tracker should see it, and the lines it must avoid.

    The weighted trace (:func:`refine_cardiac`) gets the full avoid set it was built
    with, breathing sidebands of k/TR included; the mean traces keep the k/TR-only
    set they were validated with -- on TASKFORCE (TR 1 s, breathing 0.2-0.3 Hz) the
    sidebands sit at 71-77/min, where real heart rates were."""
    if card_from == "weighted":
        avoid = cardiac_avoid(tr, f_resp)
        t_c, x_c = _cardiac_series(traces.times, traces.cardiac, avoid)
        return t_c, x_c, avoid
    harmonics = tuple(k / tr for k in range(1, int(CARDIAC_BAND[1] * tr * 1.2) + 2))
    if card_from == "magnitude":
        x_c = notch_tr_harmonics(t_c, x_c, tr, CARDIAC_BAND)
    return t_c, x_c, harmonics


def _sorted_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times.ravel(), kind="stable")
    return times.ravel()[order], values.ravel()[order]


def slice_regressors(
    traces: ExcitationTraces,
    card_from: str,
    resp_from: str,
    rvt_shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
    order: int = 2,
    card_window: float = 20.0,
) -> tuple[np.ndarray, list[str], dict]:
    """Per-slice RETROICOR + RVT regressors, AFNI ``slibase`` layout (slice-major).

    For slice ``k`` of volume ``f`` the regressors are evaluated at that slice's own
    acquisition time. Rates and cycle phases are estimated on the traces resampled to
    an even grid, then read back at each excitation's true time. Columns per slice,
    matching ``physio_calc.py``:
    ``rvt00..rvtNN, card.c1, card.s1, card.c2, card.s2, resp.c1, resp.s1, resp.c2,
    resp.s2``. Returns ``(matrix (nt, nslices*m), column labels, diagnostics)``.
    """
    t_c, x_c = traces.series(card_from, uniform=True)
    t_r, x_r = traces.series(resp_from, uniform=True)
    tr = float(np.median(np.diff(traces.times[:, 0])))
    rt, rf, rprom = peak_rate(t_r, x_r, RESP_BAND)
    if card_from == "weighted" and traces.cardiac_phase is not None:
        # The detector's own phase: re-tracking its trace with the band-pass tracker
        # would bring back the very failure the constant-modulus path avoids.
        t_c, phi_c = _uniform(*_sorted_series(traces.times, traces.cardiac_phase))
        ct, cf, cprom = t_c, np.gradient(phi_c, t_c) / (2 * np.pi), float("nan")
    else:
        t_c, x_c, harmonics = _cardiac_input(traces, card_from, t_c, x_c, tr, float(np.median(rf)))
        ct, cf, cprom = track_rate(t_c, x_c, CARDIAC_BAND, card_window, avoid=harmonics)
        phi_c = cycle_phase(t_c, x_c, cf)
    phi_r = cycle_phase(t_r, x_r, rf)
    rv = _depth_rvt(traces, t_r, x_r, rt, rf)

    nt, G = traces.times.shape
    t_true = traces.times  # (nt, G): each excitation's own acquisition time

    def per_excitation(tu: np.ndarray, v: np.ndarray, shift: float = 0.0) -> np.ndarray:
        """A series on the even grid ``tu``, read at every excitation's true time."""
        return np.interp(t_true.ravel() - shift, tu, v).reshape(nt, G)

    pc, pr = per_excitation(t_c, phi_c), per_excitation(t_r, phi_r)
    rvx = per_excitation(t_r, rv)
    rv_shifted = [per_excitation(t_r, rv, s) for s in rvt_shifts]
    labels_one = [f"rvt{i:02d}" for i in range(len(rvt_shifts))]
    labels_one += [f"card.{cs}{m}" for m in range(1, order + 1) for cs in ("c", "s")]
    labels_one += [f"resp.{cs}{m}" for m in range(1, order + 1) for cs in ("c", "s")]
    nsl = traces.group.size
    cols = []
    labels = []
    for k in range(nsl):
        g = traces.group[k]
        block = [rv_shifted[i][:, g] for i in range(len(rvt_shifts))]
        for m in range(1, order + 1):
            block += [np.cos(m * pc[:, g]), np.sin(m * pc[:, g])]
        for m in range(1, order + 1):
            block += [np.cos(m * pr[:, g]), np.sin(m * pr[:, g])]
        cols.extend(block)
        labels.extend(f"s{k:03d}.{lab}" for lab in labels_one)
    diag = {
        "cardiac_from": card_from,
        "respiration_from": resp_from,
        "cardiac_rate_hz": (ct, cf),
        "respiration_rate_hz": (rt, rf),
        "cardiac_prominence": cprom,
        "respiration_prominence": rprom,
        "rv": rvx,
        # the continuous models, for regressors at other times (volume_regressors)
        "series": {"t_card": t_c, "phi_card": phi_c, "t_resp": t_r, "phi_resp": phi_r, "rv": rv},
    }
    return np.stack(cols, 1), labels, diag


# ---------------------------------------------------------------------------
# Cardiac from the vascular voxels
# ---------------------------------------------------------------------------

# Measured on 5 TASKFORCE runs against a pulse oximeter (wiki: *Physiological noise
# from the data*, 2026-11-17). Fraction of each excitation's voxels that vote: the
# vascular ones. 1 % locked to the pulse at 0.96-1.00; 5 % lost two runs and 10 %
# three, because every extra voxel brings a noisy weight.
REFINE_TOP = 0.01
REFINE_BLOCK_S = 20.0  # weights fitted on alternate blocks of this length
REFINE_ITER = 8
REFINE_HIGHPASS_HZ = 0.05
# Found or not: a heartbeat is sampled at each slice's own acquisition time, so the best
# vessel voxels' cardiac phases line up across excitations only under the TRUE slice
# times; drift, task, spin history and denoising residue are the same in every slice of
# a volume and line up no better under shuffled ones. The kept answer must out-align
# every one of SLICE_PERMUTATIONS shuffles of itself for each of the (up to)
# PERMUTED_ANSWERS distinct answers tried -- Bonferroni for "the best of several". Tested against
# recordings on TASKFORCE, two Philips sites (TR 2 s sequential, MB4) and an MB4 slab:
# no wrong find; misses come out not found (wiki: Slice-time alignment of the heartbeat).
SLICE_PERMUTATIONS = 30
# non-harmonic multiples of the cardiac phase, at least 0.2 from an integer
HARMONIC_CONTROLS = tuple(k + d / 10 for k in (1, 2, 3) for d in range(2, 9))
PERMUTED_ANSWERS = 4
# Per-excitation ICA start (used when the mean-trace start is not found).
ICA_COMPONENTS = 20
ICA_CANDIDATES = 14  # aliased frequencies tried, in order of how many excitations agree
# Below this, an aliased frequency can be a real slow signal (vasomotion ~0.1 Hz, drift
# residue): it must then out-stitch its own slow reading to count as cardiac.
ICA_SLOW_HZ = 0.15


@dataclass
class CardiacRefinement:
    """Result of :func:`refine_cardiac`."""

    trace: np.ndarray  # (nt, G) cardiac signal at each excitation
    found: bool
    split_half: float  # median 30 s phase locking between two independent halves
    heart_rate_per_min: float
    weight: np.ndarray  # (nx, ny, nz) cardiac signal-to-residual per voxel
    n_voxels: int
    start_rate_per_min: float
    alias_coherence: float = 0.0  # brain-mean trace vs the chosen alias (diagnostic)
    alias_ratio: float = 0.0  # slice alignment of the chosen alias over the next-best
    start: str = "mean"  # which start was kept: "mean", "ica@<aliased Hz>"
    tried: list = field(
        default_factory=list
    )  # (start, rate, alignment, its shuffles' max, harmonic, control max); nan if untested
    tried_phase: list = field(default_factory=list)  # each start's final (nt, G) phase
    alignment: float = 0.0  # vessel phases lined up under the true slice times, 0-1
    alignment_null: float = 0.0  # the best of the slice-time shuffles
    harmonic: float = 0.0  # the same alignment at twice the cardiac phase
    harmonic_null: float = 0.0  # the best at non-harmonic multiples (HARMONIC_CONTROLS)
    testable: bool = True  # False with a single excitation time: nothing to shuffle
    resolvability: dict = field(default_factory=dict)  # slice_resolvability of the acquisition
    # "constant_modulus" (physio_cm.detect) or "refine" (the rebuild-and-re-read starts);
    # for the former, alignment/null/harmonic hold its fundamental+harmonic statistic
    method: str = "refine"
    phase: np.ndarray | None = None  # (nt, G) phase to use as is (constant_modulus only)
    # multi-echo: each echo's share of the summed squared weight (diagnostic)
    echo_share: tuple[float, ...] = ()


def cardiac_avoid(
    tr: float, f_resp: float | None = None, band: tuple[float, float] = CARDIAC_BAND
) -> tuple[tuple[float, float], ...]:
    """Frequencies the cardiac tracker must not choose, as (Hz, half-width) pairs.

    ``k/TR``, widened to 0.05 Hz: the voxels are high-passed at
    ``REFINE_HIGHPASS_HZ`` and a heartbeat that close to k/TR aliases below it in
    every voxel, where it cannot be told from slow drift. Left open, the shoulder
    of the comb captured the track on HiHiFaces (87/min, 1/TR 89.6).
    ``k/TR +- f_resp`` and ``+- 2 f_resp``: breathing modulates the TR pattern. On
    HiHiFaces those lines (72, 81, 98/min) out-scored everything else in the band,
    in magnitude and phase alike, and per-voxel respiratory RETROICOR did not remove
    them.
    """
    out: list[tuple[float, float]] = []
    reach = band[1] * 1.2 + (2 * f_resp if f_resp else 0.0)
    k = 1
    while k / tr <= reach:
        out.append((k / tr, 0.05))
        if f_resp:
            for m in (1, 2):
                out += [(k / tr - m * f_resp, 0.03), (k / tr + m * f_resp, 0.03)]
        k += 1
    return tuple(a for a in out if band[0] * 0.8 - 0.05 < a[0] < band[1] * 1.2 + 0.05)


def _cardiac_series(
    times: np.ndarray, x: np.ndarray, avoid, band: tuple[float, float] = CARDIAC_BAND
) -> tuple[np.ndarray, np.ndarray]:
    """An (nt, G) trace on an even grid, band-passed and notched at ``avoid``."""
    order = np.argsort(times.ravel(), kind="stable")
    tu, xu = _uniform(times.ravel()[order], x.ravel()[order])
    fs = _fs(tu)
    xb = _bandpass(xu, band[0], band[1], fs)
    for f0, _ in avoid:
        if band[0] * 0.8 < f0 < min(band[1] * 1.2, 0.45 * fs):
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            xb = signal.filtfilt(b, a, xb)
    return tu, xb


def _trace_phase(times: np.ndarray, x: np.ndarray, avoid, window: float = 20.0):
    """Cardiac cycle phase of an (nt, G) trace at every excitation, and its rate track."""
    tu, xb = _cardiac_series(times, x, avoid)
    ct, cf, prom = track_rate(tu, xb, CARDIAC_BAND, window, avoid=avoid)
    phi = cycle_phase(tu, xb, cf)
    return np.interp(times.ravel(), tu, phi).reshape(times.shape), ct, cf, prom


def _cardiac_voxels(
    magnitude: np.ndarray,
    slice_times: np.ndarray,
    tr: float,
    mask: np.ndarray,
    resp_phase: np.ndarray | None,
    highpass: float,
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """Each excitation's voxels as host (nv, nt) float32: fractional change with drift,
    slow fluctuations below ``highpass`` Hz and breathing (RETROICOR to 2nd order at
    the excitation's own times) projected out, and the (x, y, z) of every row.

    Built on the CPU in float64: per slice it is a small product, and on CUDA the
    default TF32 matmul keeps ~3 digits, the size of a pulsation on a raw intensity.
    """
    nx, _, nz, nt = magnitude.shape
    group, gtimes = excitation_groups(slice_times)
    cpu = torch.device("cpu")
    L = construct_polynomial_matrix(nt, _drift_degree(nt, tr), device=cpu, dtype=torch.float64)
    n = torch.arange(nt, dtype=torch.float64)
    ks = [k for k in range(1, nt) if k / (2 * nt * tr) < highpass]
    C = (
        torch.stack([torch.cos(torch.pi * k * (n + 0.5) / nt) for k in ks], 1)
        if ks
        else n.new_zeros((nt, 0))
    )
    bases = []
    for g in range(len(gtimes)):
        cols = [L, C]
        if resp_phase is not None:
            p = torch.from_numpy(np.asarray(resp_phase[:, g], dtype=np.float64))
            cols.append(torch.stack([p.cos(), p.sin(), (2 * p).cos(), (2 * p).sin()], 1))
        bases.append(_orthonormal_basis(torch.cat(cols, 1)))
    ys: list[list[torch.Tensor]] = [[] for _ in gtimes]
    where: list[list[np.ndarray]] = [[] for _ in gtimes]
    for k in range(nz):
        sel = mask[:, :, k].reshape(-1, order="F")
        if int(sel.sum()) < 20:
            continue
        m = _slice_rows(magnitude, k, cpu, np.float64)[torch.from_numpy(sel)]
        mu = m.mean(1, keepdim=True)
        y = (m - mu) / mu.clamp_min(1e-6)
        Q = bases[group[k]]
        y = y - (y @ Q) @ Q.T
        ys[group[k]].append(y.float())
        flat = np.flatnonzero(sel)
        where[group[k]].append(np.column_stack([flat % nx, flat // nx, np.full(flat.size, k)]))
    Ys = [torch.cat(y) if y else torch.zeros((0, nt)) for y in ys]
    idx = [np.concatenate(w) if w else np.zeros((0, 3), int) for w in where]
    return Ys, idx


@dataclass
class _Member:
    """One fit inside a shared pass: its rows of each excitation and its result."""

    rows: list  # per excitation: index tensor on the store device, or None for all
    phase: np.ndarray
    trace: np.ndarray | None = None
    rates: np.ndarray | None = None
    snr: list | None = None


def _store_voxels(
    Ys: list[torch.Tensor], device: torch.device
) -> tuple[list[torch.Tensor], torch.dtype]:
    """The voxel series where the passes read them: resident on ``device`` in the
    accumulator dtype when they fit (float64 on CUDA keeps TF32 away from a pulsation
    the size of its rounding), otherwise left on the host for streaming."""
    from fastfuncstuff.memory import get_available_memory

    dt = accum_dtype(device)
    need = sum(Y.numel() for Y in Ys) * torch.empty(0, dtype=dt).element_size()
    if device.type != "cpu" and need > get_available_memory(device, empty_cache=False):
        return Ys, dt
    return [Y.to(device=device).to(dt) for Y in Ys], dt


def _matmul_rows(Y: torch.Tensor, X: torch.Tensor, device: torch.device, dt: torch.dtype):
    """``Y @ X`` on ``device``; a host ``Y`` is streamed in chunks."""
    if Y.device == device:
        return Y @ X
    from fastfuncstuff.memory import bytes_per_voxel_cardiac_refine, get_available_memory

    per = bytes_per_voxel_cardiac_refine(Y.shape[1])
    chunk = int(max(1024, get_available_memory(device, empty_cache=False) // per))
    return torch.cat([Y[i : i + chunk].to(device).to(dt) @ X for i in range(0, Y.shape[0], chunk)])


def _refine_pass(
    Ys: list[torch.Tensor],
    times: np.ndarray,
    members: list[_Member],
    avoid,
    tr: float,
    device: torch.device,
    dt: torch.dtype,
    top: float = REFINE_TOP,
    block_s: float = REFINE_BLOCK_S,
    n_iter: int = REFINE_ITER,
) -> list[_Member]:
    """Iterate: fit every voxel to the current cardiac phase, rebuild the trace from
    the best-fitting voxels, re-read its phase -- for several members at once.

    Each voxel is fitted with cos/sin of the phase at its excitation's times, so the
    pair absorbs the voxel's own delay and sign. A member's new trace at a sample is
    the weighted least-squares estimate of (cos, sin) from its top ``top`` voxels of
    the excitation, weights 1/residual variance. Weights are fitted on alternate
    ``block_s`` blocks and applied to the others, so no voxel's noise feeds its own
    weight. Members (the whole brain and its two halves) are row subsets of the same
    voxels, so one product per excitation serves all of them.
    """
    nt, G = times.shape
    nb = max(2, int(round(nt * tr / block_s)))
    blk = (np.arange(nt) * nb // nt) % 2
    train = torch.from_numpy(np.stack([blk != 0, blk != 1], 1).astype(np.float64)).to(device, dt)
    test = [torch.from_numpy(blk == f).to(device) for f in (0, 1)]
    n_train = train.sum(0)
    ss = [_matmul_rows(Y * Y if Y.device == device else Y, train, device, dt) if Y.device == device
          else torch.cat([(Y[i:i + 65536].to(device).to(dt) ** 2) @ train for i in range(0, Y.shape[0], 65536)])
          if Y.shape[0] else torch.zeros((0, 2), device=device, dtype=dt)
          for Y in Ys]  # fmt: skip
    M = len(members)
    for _ in range(n_iter):
        traces = np.zeros((M, nt, G))
        snrs: list[list[torch.Tensor]] = [[] for _ in range(M)]
        for g, Y in enumerate(Ys):
            nv = Y.shape[0]
            if nv == 0:
                for m in range(M):
                    snrs[m].append(torch.zeros(0))
                continue
            Xs = []
            for mem in members:
                X = torch.from_numpy(
                    np.stack([np.cos(mem.phase[:, g]), np.sin(mem.phase[:, g])], 1)
                )
                X = X.to(dt).to(device)
                Xs += [X * train[:, 0:1], X * train[:, 1:2]]
            P = _matmul_rows(Y, torch.cat(Xs, 1), device, dt)  # (nv, 4M)
            for m, mem in enumerate(members):
                rows = mem.rows[g] if mem.rows is not None else None
                Pm = P if rows is None else P[rows]
                ssm = ss[g] if rows is None else ss[g][rows]
                nvm = Pm.shape[0]
                snr_m = torch.zeros(nvm, device=device, dtype=dt)
                for f in (0, 1):
                    Xf = Xs[2 * m + f]
                    a, b, d = (
                        (Xf[:, 0] ** 2).sum(),
                        (Xf[:, 0] * Xf[:, 1]).sum(),
                        (Xf[:, 1] ** 2).sum(),
                    )
                    ridge = 1e-3 * (a + d) + 1e-12
                    det = (a + ridge) * (d + ridge) - b * b
                    Xty = Pm[:, 2 * (2 * m + f) : 2 * (2 * m + f) + 2]
                    B = torch.stack([(d + ridge) * Xty[:, 0] - b * Xty[:, 1],
                                     (a + ridge) * Xty[:, 1] - b * Xty[:, 0]], 1) / det  # fmt: skip
                    expl = a * B[:, 0] ** 2 + 2 * b * B[:, 0] * B[:, 1] + d * B[:, 1] ** 2
                    rss = (ssm[:, f] - 2 * (B * Xty).sum(1) + expl).clamp_min(1e-30)
                    snr = expl / rss
                    snr_m += 0.5 * snr
                    k = min(nvm, max(10, int(np.ceil(top * nvm))))
                    keep = snr.topk(k).indices
                    Bk = B[keep]
                    W = Bk / (rss[keep] / n_train[f])[:, None]
                    src = keep if rows is None else rows[keep]
                    Yk = Y[src] if Y.device == device else Y[src.cpu()].to(device)
                    rhs = W.T @ Yk[:, test[f]].to(dt)  # (2, n_test)
                    m00, m01, m11 = (
                        (Bk[:, 0] * W[:, 0]).sum(),
                        (Bk[:, 0] * W[:, 1]).sum(),
                        (Bk[:, 1] * W[:, 1]).sum(),
                    )
                    eps = 1e-9 * (m00 + m11) + 1e-30
                    th0 = ((m11 + eps) * rhs[0] - m01 * rhs[1]) / (
                        (m00 + eps) * (m11 + eps) - m01 * m01
                    )
                    traces[m, blk == f, g] = th0.cpu().numpy()
                snrs[m].append(snr_m.cpu())
        for m, mem in enumerate(members):
            mem.phase, ct, cf, _ = _trace_phase(times, traces[m], avoid)
            mem.trace, mem.rates, mem.snr = traces[m], np.stack([ct, cf]), snrs[m]
    return members


def mean_trace_coherence(mean_trace: np.ndarray, phase: np.ndarray, tr: float) -> float:
    """|<brain-mean trace, exp(i phase)>| over the norms.

    The unit-weight stitch keeps the timing across excitations that a per-voxel fit
    absorbs, so it tells the aliases of a cardiac phase apart even where it is too
    weak, or too cancelled by mixed vessel signs, to serve as a start."""
    x = remove_tr_comb(mean_trace, tr)
    x = x - x.mean(0)
    z = np.exp(1j * phase)
    return float(np.abs((x * np.conj(z)).sum()) / np.sqrt((x**2).sum() * z.size))


def _alias_phases(times: np.ndarray, phase: np.ndarray, tr: float) -> dict[float, np.ndarray]:
    """The phase under every alias with a rate in the cardiac band (widened to 150/min):
    f + k/TR and k/TR - f. Only these are indistinguishable to a per-voxel fit."""
    span = times[-1].mean() - times[0].mean()
    f0 = (phase[-1].mean() - phase[0].mean()) / (2 * np.pi * span)
    out = {}
    for k in range(-3, 4):
        for sgn in (1, -1):
            fr = sgn * f0 + k / tr
            if CARDIAC_BAND[0] <= fr <= 2.5:
                out[round(fr * 60, 1)] = sgn * phase + 2 * np.pi * k * times / tr
    return out


def _split_locking(a: np.ndarray, b: np.ndarray, times: np.ndarray, tr: float) -> float:
    """Median phase locking of two (nt, G) phases over 30 s windows."""
    G = times.shape[1]
    d = (a - b).ravel()[np.argsort(times.ravel(), kind="stable")]
    w = max(2, int(30.0 * G / tr))
    return float(
        np.median(
            [abs(np.exp(1j * d[i : i + w]).mean()) for i in range(0, max(1, d.size - w), w // 2)]
        )
    )


def _stitched_power(cols: np.ndarray, times: np.ndarray, cands) -> float:
    """Peak power of excitation columns stitched at their own times, within 0.03 Hz of
    any candidate frequency (a direct DFT at the true sample times)."""
    fr = np.concatenate([np.arange(c - 0.03, c + 0.03, 0.002) for c in cands])
    x = sum(
        (cols[:, g, None] * np.exp(-2j * np.pi * fr[None] * times[:, g, None])).sum(0)
        for g in range(cols.shape[1])
    )
    return float((np.abs(x) ** 2).max())


def _voxel_fits(Ys: list[torch.Tensor], phase: np.ndarray, device: torch.device) -> list:
    """Each voxel's complex cardiac coefficient and fit signal-to-residual under
    ``phase`` (nt, G), excitation by excitation: (z, snr) on ``device``, or None."""
    dt = accum_dtype(device)
    out = []
    for g, Y in enumerate(Ys):
        if Y.shape[0] == 0:
            out.append(None)
            continue
        X = torch.from_numpy(np.stack([np.cos(phase[:, g]), np.sin(phase[:, g])], 1))
        X = X.to(dt).to(device)
        Yd = Y.to(device).to(dt)
        Yd = Yd - Yd.mean(1, keepdim=True)
        XtX = X.T @ X
        B = (Yd @ X) @ plinalg.inv(XtX)
        expl = ((B @ XtX) * B).sum(1)
        rss = ((Yd * Yd).sum(1) - expl).clamp_min(1e-30)
        out.append((torch.complex(B[:, 0], -B[:, 1]), expl / rss))
    return out


def slice_alignment(fits: list, top: float = REFINE_TOP) -> float:
    """How well the best voxels' cardiac phases line up, pooled over excitations: the
    SNR-weighted mean of exp(2i * phase) (doubled, so a vessel's sign does not matter),
    0-1. Each voxel's phase holds its own delay, so only a heartbeat sampled at the
    slice times it was fitted at lines up across excitations; anything the same in
    every slice of a volume turns with the slice time instead (wiki: Slice-time
    alignment of the heartbeat)."""
    num = torch.zeros((), dtype=torch.complex128)
    den = 0.0
    for ft in fits:
        if ft is None:
            continue
        z, snr = ft
        k = min(snr.numel(), max(10, int(np.ceil(top * snr.numel()))))
        sel = snr.topk(k).indices
        u = z[sel] / z[sel].abs().clamp_min(1e-30)
        num = num + (snr[sel] * u * u).sum().cpu()
        den += float(snr[sel].sum())
    return float(num.abs() / den) if den > 0 else 0.0


def _row_moments(Y: torch.Tensor, device: torch.device, dt: torch.dtype):
    """Per-row sum and sum of squares of ``Y`` on ``device``; a host ``Y`` streamed."""
    if Y.device == device:
        return Y.sum(1), (Y * Y).sum(1)
    from fastfuncstuff.memory import bytes_per_voxel_cardiac_refine, get_available_memory

    chunk = int(
        max(1024, get_available_memory(device, empty_cache=False)
            // bytes_per_voxel_cardiac_refine(Y.shape[1]))
    )  # fmt: skip
    parts = [Y[i : i + chunk].to(device).to(dt) for i in range(0, Y.shape[0], chunk)]
    return torch.cat([q.sum(1) for q in parts]), torch.cat([(q * q).sum(1) for q in parts])


def slice_alignments(
    Ys: list[torch.Tensor], phases, device: torch.device, top: float = REFINE_TOP
) -> np.ndarray:
    """``slice_alignment(_voxel_fits(Ys, phase))`` for many phases in one pass over the
    voxel store: every phase's cos/sin is a column of one design per excitation, so a
    null of hundreds of deranged clocks costs a few matmuls, not hundreds of re-reads
    of the store. Same estimator: per-voxel least squares on the mean-removed series,
    top ``top`` voxels by fit SNR, SNR-weighted doubled-angle mean."""
    phases = list(phases)
    P = len(phases)
    if P == 0:
        return np.zeros(0)
    from fastfuncstuff.memory import get_available_memory

    nt, G = phases[0].shape
    num = torch.zeros(P, dtype=torch.complex128)
    den = torch.zeros(P, dtype=torch.float64)
    n_max = max(Y.shape[0] for Y in Ys)
    dt = accum_dtype(device)
    width = torch.empty(0, dtype=dt).element_size()
    # per phase: its (nt, G) clock on the device, cos/sin, and ~10 (n, .) temporaries
    per_phase = (max(1, n_max) * 10 + nt * (G + 4)) * width
    step = int(max(1, min(P, get_available_memory(device, empty_cache=False) // per_phase)))
    moments = []
    for Y in Ys:
        if Y.shape[0] == 0:
            moments.append(None)
            continue
        ysum, ysq = _row_moments(Y, device, dt)
        mean = ysum / nt
        moments.append((mean, ysq - mean * ysum))  # centred sum of squares
    for a in range(0, P, step):
        # one upload per chunk; the trig runs on the device (on the host, cos/sin of
        # strided per-excitation slices cost 10x the matmuls)
        ph = torch.from_numpy(np.stack(phases[a : a + step])).to(dt).to(device)  # (p, nt, G)
        p_ = ph.shape[0]
        for g, Y in enumerate(Ys):
            if moments[g] is None:
                continue
            mean, ssc = moments[g]
            n = Y.shape[0]
            k = min(n, max(10, int(np.ceil(top * n))))
            q = ph[:, :, g].T  # (nt, p)
            C, S = torch.cos(q), torch.sin(q)
            YX = _matmul_rows(Y, torch.cat([C, S], 1), device, dt)  # (n, 2p)
            u = YX[:, :p_] - mean[:, None] * C.sum(0)
            v = YX[:, p_:] - mean[:, None] * S.sum(0)
            cc, cs, ss = (C * C).sum(0), (C * S).sum(0), (S * S).sum(0)
            det = cc * ss - cs * cs
            b0 = (ss * u - cs * v) / det
            b1 = (cc * v - cs * u) / det
            expl = b0 * u + b1 * v
            snr = expl / (ssc[:, None] - expl).clamp_min(1e-30)
            del u, v, YX, expl
            w, idx = snr.topk(k, dim=0)  # (k, p)
            z0, z1 = b0.gather(0, idx), -b1.gather(0, idx)
            r2 = (z0 * z0 + z1 * z1).clamp_min(1e-60)
            re, im = (z0 * z0 - z1 * z1) / r2, 2 * z0 * z1 / r2  # z^2 / |z|^2
            num[a : a + p_] += torch.complex(
                (w * re).sum(0).cpu().double(), (w * im).sum(0).cpu().double()
            )
            den[a : a + p_] += w.sum(0).cpu().double()
        del ph
    out = (num.abs() / den.clamp_min(1e-300)).numpy()
    out[den.numpy() <= 0] = 0.0
    return out


def _alias_offsets(times, phase, tr, avoid=()) -> dict[float, tuple[int, int]]:
    """The (sign, k) of every alias of ``phase`` -- sign * phase + 2 pi k t / TR -- whose
    rate is in the cardiac band and outside ``avoid``, keyed by that rate per minute.
    Near k/TR a slow signal, the same in every slice of a volume, re-aliases onto a
    rate that lines up like a heartbeat (pure noise did, at 58.6/min with TR 1 s)."""
    span = times[-1].mean() - times[0].mean()
    f0 = (phase[-1].mean() - phase[0].mean()) / (2 * np.pi * span)
    out = {}
    for k in range(-3, 4):
        for sgn in (1, -1):
            fr = sgn * f0 + k / tr
            if CARDIAC_BAND[0] <= fr <= CARDIAC_BAND[1] and all(abs(fr - c) >= w for c, w in avoid):
                out[round(fr * 60, 1)] = (sgn, k)
    return out


def _best_alias(Ys, times, phase, offsets, tr, device) -> tuple[float, np.ndarray | None, dict]:
    """The alias in ``offsets`` whose vessel voxels line up best: (score, phase, scores)."""
    phases = {h: sgn * phase + 2 * np.pi * k * times / tr for h, (sgn, k) in offsets.items()}
    scores = dict(zip(phases, slice_alignments(Ys, phases.values(), device), strict=True))
    if not scores:
        return 0.0, None, {}
    h = max(scores, key=scores.get)
    return scores[h], phases[h], scores


def _shuffled_alignment(Ys, times, phase, offsets, tr, device, n: int, rng) -> np.ndarray:
    """The best-alias alignment of ``phase`` with the excitations' sample times shuffled
    among them, choosing among the same alias offsets as the answer did: the null a
    heartbeat must beat. Every excitation moves (a derangement): with few excitations,
    a shuffle that swaps two neighbours is nearly the truth, not a null."""
    G = phase.shape[1]
    if n == 0 or not offsets:
        return np.zeros(n)
    phases = []
    for _ in range(n):
        perm = rng.permutation(G)
        while np.any(perm == np.arange(G)):
            perm = rng.permutation(G)
        phases += [sgn * phase[:, perm] + 2 * np.pi * k * times / tr for sgn, k in offsets.values()]
    return slice_alignments(Ys, phases, device).reshape(n, len(offsets)).max(1)


def _harmonic_alignment(Ys, phase, device) -> tuple[float, float]:
    """Slice alignment at the second harmonic (twice ``phase``), and the best of the
    same alignment at the non-harmonic multiples ``HARMONIC_CONTROLS``.

    A pulse is not a sinusoid: its waveform carries a second harmonic, sampled at the
    same slice times, that lines up too. Slow physical signals that impersonate a
    heartbeat through an alias line up at the fundamental only. A control multiple has
    no physical relation to the candidate but the same slice timing and voxels, so the
    controls are this run's own floor for chance alignment: a signal without a harmonic
    beats all of them 1 time in len(HARMONIC_CONTROLS) + 1. Shuffling the excitations
    is no floor here -- it only rearranges the excitations' few angles, and the best
    of 30 often lines a real harmonic up again (median 0.27 under true heartbeats,
    against 0.15 under impostors). Wiki: Slice-time alignment of the heartbeat."""
    a = slice_alignments(Ys, [m * phase for m in (2, *HARMONIC_CONTROLS)], device)
    return float(a[0]), float(a[1:].max())


def slice_resolvability(
    slice_times: np.ndarray, tr: float, rates_per_min=(50.0, 75.0, 100.0)
) -> dict:
    """How sharply this acquisition's excitation times can test a heartbeat, from the
    sidecar alone (wiki: Slice-time alignment of the heartbeat).

    A perfect heartbeat lines up at slope kappa = 0; its profile over kappa is
    |mean_g exp(4 pi i kappa s_g / TR)|. ``alias_leak`` is that profile's height at the
    other integer kappa (how confusable alias steps are); ``volume_leak`` its height
    where a signal the same in every slice of a volume would sit (kappa = rate x TR),
    for the given heart rates. Near 0: sharp. TASKFORCE (17 excitations) 0.00 / 0.17;
    ds004645 (5 excitations at TR 0.227) 0.40 / 0.81, where the test is blunt."""
    s = np.unique(np.round(np.asarray(slice_times, float), 6)) / tr

    def psf(k):
        return float(np.abs(np.exp(4j * np.pi * k * s).mean()))

    return dict(
        excitations=int(s.size),
        alias_leak=max(psf(k) for k in (-3, -2, -1, 1, 2, 3)),
        volume_leak=max(psf(r * tr / 60) for r in rates_per_min),
    )


def _crossed_phase(times: np.ndarray, trace: np.ndarray, avoid) -> np.ndarray:
    """The cardiac phase at the odd excitations read from the even excitations' traces
    only, and vice versa. A phase tracked through all excitations at their true times
    is shaped by each excitation's own voxels, so on pure noise it partly manufactures
    the cross-excitation agreement :func:`slice_alignment` measures; crossing removes
    that. Each half still spans the TR, so the phase stays well sampled."""
    nt, G = times.shape
    if G < 2:
        return _trace_phase(times, trace, avoid)[0]
    out = np.zeros((nt, G))
    halves = (np.arange(0, G, 2), np.arange(1, G, 2))
    for src, dst in (halves, halves[::-1]):
        tu, xb = _cardiac_series(times[:, src], trace[:, src], avoid)
        _, cf, _ = track_rate(tu, xb, CARDIAC_BAND, 20.0, avoid=avoid)
        phi = cycle_phase(tu, xb, cf)
        out[:, dst] = np.interp(times[:, dst].ravel(), tu, phi).reshape(nt, dst.size)
    return out


def _phase_rate(times: np.ndarray, phase: np.ndarray) -> float:
    """Mean rate (per minute) of an unwrapped (nt, G) phase over the run."""
    span = times[-1].mean() - times[0].mean()
    return float(abs(phase[-1].mean() - phase[0].mean()) / (2 * np.pi * span) * 60)


def _ica_starts(Ys: list[torch.Tensor], times: np.ndarray, tr: float, avoid, device: torch.device):
    """Cardiac starts from spatial ICA per excitation, best-supported first.

    Every slice of an excitation is sampled at one instant, and the heartbeat aliases to
    the SAME volume-rate frequency in every excitation, whatever its slice time. So the
    aliased cardiac frequency is where most excitations have a component concentrated,
    and each excitation's component there is its piece of the trace. Signs come from
    neighbours in acquisition time (0.05 s apart, so a heartbeat correlates positively):
    choosing them to maximise in-band power turned slow signals into 1/TR combs.
    Yields (label, start phase). Breathing and vasomotion are not excluded by band --
    excluding breathing's harmonics hid a heartbeat aliased onto 2x the breathing rate --
    they are tried and fail the same test as any wrong start.
    """
    from fastfuncstuff.decomposition.ica import FastICA

    nt, G = times.shape
    fs = 1.0 / tr
    # one component count for every excitation, so they stack; an excitation with too
    # few voxels for it (slices outside the mask) contributes silent columns
    K = min(ICA_COMPONENTS, nt - 1, max(1, max(Y.shape[0] for Y in Ys) - 1))
    tcs = []
    for Y in Ys:
        if Y.shape[0] <= K:
            tcs.append(np.zeros((nt, K)))
            continue
        y = Y.to(device=device, dtype=torch.float32)
        y = y / (y.std(1, keepdim=True) + 1e-9)
        ica = FastICA(
            n_components=K, pca_components=K, random_state=0, device=device, verbose=False
        )
        T = ica.fit_transform(y.T).cpu().numpy().astype(np.float64)  # spatial ICA: time courses
        tcs.append(T / T.std(0, keepdims=True).clip(1e-12))
    nseg = min(128, nt)
    f, Pw = signal.welch(np.stack(tcs), fs=fs, nperseg=nseg, axis=1)  # (G, F, K)
    Pn = Pw / Pw.sum(1, keepdims=True).clip(1e-300)
    score = np.where(f > 1.0 / (nt * tr) * 2, Pn.max(2).sum(0), 0.0)
    for i in [j for j in np.argsort(score)[::-1] if score[j] > 0][:ICA_CANDIDATES]:
        band = np.abs(f - f[i]) <= 2 * (f[1] - f[0])
        cols = np.stack([tcs[g][:, int(np.argmax(Pn[g][band].sum(0)))] for g in range(G)], 1)
        for g in range(1, G):
            r = np.corrcoef(cols[:, g - 1], cols[:, g])[0, 1]
            cols[:, g] *= -1.0 if r < 0 else 1.0
        if f[i] < ICA_SLOW_HZ:
            card = [k * fs + s * f[i] for k in range(4) for s in (-1, 1)]
            card = [c for c in card if CARDIAC_BAND[0] <= c <= CARDIAC_BAND[1]]
            if not card or _stitched_power(cols, times, card) <= _stitched_power(
                cols, times, [f[i]]
            ):
                continue
        yield f"ica@{f[i]:.3f}", _trace_phase(times, cols, avoid)[0]


def refine_cardiac(
    magnitude: np.ndarray, *args, n_echoes: int = 1, echo_axis: int | None = None, **kwargs
) -> CardiacRefinement:
    """:func:`_refine_cardiac`, with a stacked multi-echo weight map folded back to one
    volume (see there)."""
    out = _refine_cardiac(magnitude, *args, n_echoes=n_echoes, echo_axis=echo_axis, **kwargs)
    if n_echoes > 1:
        assert echo_axis is not None
        per = np.stack(echo_views(out.weight, n_echoes, echo_axis))
        sq = (per.astype(np.float64) ** 2).sum(axis=tuple(range(1, per.ndim)))
        out.echo_share = tuple(float(x) for x in sq / max(float(sq.sum()), 1e-30))
        out.weight = np.take_along_axis(per, np.abs(per).argmax(0)[None], 0)[0]
    return out


def _refine_cardiac(
    magnitude: np.ndarray,
    traces: ExcitationTraces,
    slice_times: np.ndarray,
    tr: float,
    resp_from: str = "phase",
    mask: np.ndarray | None = None,
    split_axis: int = 1,
    device: torch.device | None = None,
    top: float = REFINE_TOP,
    n_iter: int = REFINE_ITER,
    ica: bool = True,
    constant_modulus: bool = True,
    n_echoes: int = 1,
    echo_axis: int | None = None,
) -> CardiacRefinement:
    """A cardiac trace from the voxels that carry the heartbeat, not from all of them.

    The magnitude trace averages every brain voxel, but pulsation lives in a small
    fraction of them (the vessels), each with its own sign and delay, so the mean
    dilutes and partly cancels it. From a start, :func:`_refine_pass` finds the voxels
    that follow it and rebuilds the trace from them, excitation by excitation (all
    slices acquired at one instant are fitted at that instant).

    Every start (the brain-mean trace, then per-excitation ICA, :func:`_ica_starts`) is
    refined and judged by slice timing (wiki: Slice-time alignment of the heartbeat):
    the alias whose vessel phases line up best across excitations is kept, and found
    means that alignment beats slice-time shuffles of itself (Bonferroni over the
    answers tested) AND its second harmonic beats its own shuffles -- slow physical
    signals aliased into the band pass the first and fail the second. One excitation
    time: untestable. Two brain halves (split along ``split_axis``, the in-plane axis
    NOT phase-encoded) are refitted from the kept answer as a diagnostic only.

    Multi-echo: ``magnitude`` is :func:`stack_echoes` output (``n_echoes`` tiled along
    ``echo_axis``); every echo's voxels are candidates, the default mask is each
    echo's own, and the returned weight map is per voxel the largest over echoes,
    with ``echo_share`` saying how much of the weight each echo carried.
    """
    device = device or torch.device("cpu")
    if n_echoes > 1 and (echo_axis is None or echo_axis == split_axis):
        raise ValueError("stacked echoes need echo_axis, and not the split axis")
    nt, G = traces.times.shape
    times = traces.times
    if G < 2:  # one excitation time: no within-volume samples, nothing to shuffle
        return CardiacRefinement(
            trace=np.zeros((nt, G)), found=False, split_half=0.0, heart_rate_per_min=float("nan"),
            weight=np.zeros(magnitude.shape[:3], np.float32), n_voxels=0,
            start_rate_per_min=float("nan"), start="none", testable=False,
            resolvability=slice_resolvability(slice_times, tr),
        )  # fmt: skip
    if mask is None:
        if n_echoes > 1:
            assert echo_axis is not None
            views = echo_views(magnitude[..., :nt], n_echoes, echo_axis)
            mask = np.concatenate([_bright_mask(v) for v in views], axis=echo_axis)
        else:
            mask = _bright_mask(magnitude[..., :nt])
    if resp_from == "phase" and traces.phase is None:
        resp_from = "magnitude"
    t_r, x_r = traces.series(resp_from, uniform=True)
    _, rf, _ = peak_rate(t_r, x_r, RESP_BAND)
    phi_r = cycle_phase(t_r, x_r, rf)
    resp_phase = np.interp(times.ravel(), t_r, phi_r).reshape(times.shape)
    avoid = cardiac_avoid(tr, float(np.median(rf)))
    host, idx = _cardiac_voxels(
        magnitude[..., :nt], slice_times, tr, mask, resp_phase, REFINE_HIGHPASS_HZ
    )
    Ys, dt = _store_voxels(host, device)
    n_ax = magnitude.shape[split_axis]
    halves = [
        [torch.from_numpy(np.flatnonzero((i[:, split_axis] < n_ax // 2) == (side == 0))).to(device)  # indexes P, computed on device
         for g, i in enumerate(idx)]
        for side in (0, 1)
    ]  # fmt: skip
    cm_tried: list = []
    cm_phases: list = []
    if constant_modulus:
        # Frequency-free starts first (physio_cm): they find heartbeats that alias near
        # 0 or Nyquist, where the spectral starts below fail. Not found -> the old path.
        from .physio_cm import detect, split_half_locking, split_masks, temporal_grams

        # one pass builds both halves' Grams; the whole brain's is their sum
        half_grams = temporal_grams(magnitude[..., :nt], split_masks(mask, split_axis), device)
        cm = detect(magnitude[..., :nt], mask, Ys, times, tr, device, np.random.default_rng(1),
                    band=CARDIAC_BAND, shuffles=SLICE_PERMUTATIONS,
                    controls=HARMONIC_CONTROLS, gram=half_grams[0] + half_grams[1])  # fmt: skip
        cm_tried, cm_phases = list(cm.tried), list(cm.tried_phase)
        if not cm.found:
            del half_grams
        if cm.found and cm.phase is not None:
            fits = _voxel_fits(Ys, cm.phase, device)
            n_vox = int(sum(Y.shape[0] for Y in Ys))
            Ys = halves = []
            weight = np.zeros(magnitude.shape[:3], np.float32)
            for ft, i in zip(fits, idx, strict=True):
                if ft is not None and i.size:
                    weight[i[:, 0], i[:, 1], i[:, 2]] = ft[1].float().cpu().numpy()
            del fits
            return CardiacRefinement(
                trace=np.cos(cm.phase), found=True,
                split_half=split_half_locking(magnitude[..., :nt], mask, split_axis, cm.phase,
                                              times, tr, device, grams=half_grams),
                heart_rate_per_min=cm.rate_per_min, weight=weight, n_voxels=n_vox,
                start_rate_per_min=cm.rate_per_min, start=cm.label, tried=cm_tried,
                tried_phase=cm_phases, alignment=cm.statistic, alignment_null=cm.floor,
                harmonic=cm.harmonic, harmonic_null=cm.harmonic_null, testable=True,
                resolvability=slice_resolvability(slice_times, tr),
                method="constant_modulus", phase=cm.phase,
            )  # fmt: skip

    def judge(start: np.ndarray):
        # A loop, not recursion: a closure that calls itself is a reference cycle, and it
        # kept the GPU-resident voxel store alive until the garbage collector happened
        # to run -- 15 GB over a loop of runs in one process.
        for realias in (True, False):
            mem = _refine_pass(
                Ys, times, [_Member(None, start)], avoid, tr, device, dt, top, n_iter=n_iter
            )
            offsets = _alias_offsets(times, mem[0].phase, tr, avoid)
            score, chosen, scores = _best_alias(Ys, times, mem[0].phase, offsets, tr, device)
            if chosen is None:
                return None, mem[0], 0.0, 0.0, {}
            if not (
                realias and abs(_phase_rate(times, chosen) - _phase_rate(times, mem[0].phase)) > 3.0
            ):
                break
            start = chosen  # the refinement cannot tell aliases apart: rerun on the chosen one
        order = sorted(scores, key=lambda h: -scores[h])
        ratio = scores[order[0]] / max(scores[order[1]], 1e-12) if len(order) > 1 else np.inf
        # offsets re-expressed relative to the chosen alias, for its shuffle null
        return chosen, mem[0], float(ratio), score, _alias_offsets(times, chosen, tr, avoid)

    start_mean, _, cf0, _ = _trace_phase(times, traces.magnitude, avoid)
    starts = [("mean", start_mean)] + (
        list(_ica_starts(Ys, times, tr, avoid, device)) if ica else []
    )
    answers = []  # [label, phase, member, alias ratio, alignment, rate, shuffles, alias offsets, harmonic, its null]
    for label, start in starts:
        phase, mem, ratio, _, _ = judge(start)
        if phase is None:
            continue  # every alias in a blind zone
        # scored, and shuffled, on the crossed phase (see _crossed_phase)
        crossed = _crossed_phase(times, mem.trace, avoid)
        offs = _alias_offsets(times, crossed, tr, avoid)
        score, crossed, _ = _best_alias(Ys, times, crossed, offs, tr, device)
        if crossed is None:
            continue
        offs = _alias_offsets(times, crossed, tr, avoid)
        answers.append(
            [
                label,
                crossed,
                mem,
                ratio,
                score,
                _phase_rate(times, crossed),
                None,
                offs,
                np.nan,
                np.nan,
            ]
        )
    if not answers:  # nothing outside the blind zones: report the mean start, unfound
        mem = _refine_pass(
            Ys, times, [_Member(None, start_mean)], avoid, tr, device, dt, top, n_iter=n_iter
        )[0]
        answers.append(
            [
                "mean",
                mem.phase,
                mem,
                0.0,
                0.0,
                _phase_rate(times, mem.phase),
                None,
                {},
                np.nan,
                np.nan,
            ]
        )
    # the slice-time shuffle null, for each distinct answer (best first)
    rng = np.random.default_rng(0)
    tested: list[float] = []
    for a in sorted(answers, key=lambda a: -a[4]):
        if G < 2 or len(tested) >= PERMUTED_ANSWERS:
            break
        if any(abs(a[5] - r) < 1.5 for r in tested):
            continue
        tested.append(a[5])
        a[6] = _shuffled_alignment(Ys, times, a[1], a[7], tr, device, SLICE_PERMUTATIONS, rng)
        a[8], a[9] = _harmonic_alignment(Ys, a[1], device)
    nulls = [a for a in answers if a[6] is not None]
    if nulls:
        # a heartbeat's second harmonic lines up too; among those, the most clearly
        # slice-locked (with none, the most slice-locked is reported, unfound)
        pulsed = [a for a in nulls if a[8] > a[9]] or nulls
        kept = max(pulsed, key=lambda a: a[4] - a[6].max())
        # Several answers were tried, so the kept one must beat as many shuffles OF ITS
        # OWN as were drawn for all of them (Bonferroni). Pooling the others' shuffles
        # instead let a badly converged answer -- whose shuffles can repair it -- set
        # the floor for a good one.
        extra = _shuffled_alignment(Ys, times, kept[1], kept[7], tr, device,
                                    SLICE_PERMUTATIONS * (len(nulls) - 1), rng)  # fmt: skip
        floor = (
            float(np.concatenate([kept[6], extra]).max()) if extra.size else float(kept[6].max())
        )
        ok = kept[4] > floor and kept[8] > kept[9]
    else:
        kept, floor, ok = max(answers, key=lambda a: a[4]), float("nan"), False
    label, phase, full, ratio, score, rate, _, _, harm, harm_null = kept
    # diagnostics for the kept answer only: two brain halves fitted from it, and the
    # brain-mean trace's coherence with it
    hm = _refine_pass(Ys, times, [_Member(halves[0], phase), _Member(halves[1], phase)],
                      avoid, tr, device, dt, top, n_iter=n_iter)  # fmt: skip
    split = _split_locking(hm[0].phase, hm[1].phase, times, tr)
    coherence = mean_trace_coherence(traces.magnitude, phase, tr)
    n_vox = int(sum(Y.shape[0] for Y in Ys))
    Ys = halves = []  # drop the device-resident store now, not when the caller lets go
    weight = np.zeros(magnitude.shape[:3], np.float32)
    for g, i in enumerate(idx):
        if i.size:
            weight[i[:, 0], i[:, 1], i[:, 2]] = full.snr[g].numpy()
    return CardiacRefinement(
        trace=full.trace,
        found=bool(ok),
        split_half=float(split),
        heart_rate_per_min=rate,
        weight=weight,
        n_voxels=n_vox,
        start_rate_per_min=float(np.median(cf0) * 60),
        alias_coherence=float(coherence),
        alias_ratio=float(ratio),
        start=label,
        tried=cm_tried
        + [
            (
                a[0],
                a[5],
                a[4],
                float(a[6].max()) if a[6] is not None else float("nan"),
                float(a[8]),
                float(a[9]),
            )
            for a in answers
        ],
        tried_phase=cm_phases + [a[1] for a in answers],
        alignment=float(score),
        alignment_null=float(floor),
        harmonic=float(harm),
        harmonic_null=float(harm_null),
        testable=G >= 2,
        resolvability=slice_resolvability(slice_times, tr),
    )


def volume_regressors(
    diag: dict,
    times: np.ndarray,
    rvt_shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
    order: int = 2,
    cardiac: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Regressors for data that every voxel samples at the same ``times``.

    After a slice-timing-corrected resample every voxel stands for the instant
    ``f*TR + tzero``, so the physiological model is evaluated there, once, and used as
    a GLM nuisance like the motion parameters (PhysIO's placement: Kasper 2017 samples
    every regressor at the slice-timing reference slice). Columns are the slibase
    columns in the same order, so at a slice acquired at ``tzero`` they coincide.

    Cardiac survives slice timing although it is aliased at the volume rate. Slice
    timing shifts the aliased component by the wrong phase, but that error is a
    constant per slice and harmonic as long as the harmonic does not alias across a
    multiple of the volume rate during the run, and a voxel's free cos/sin amplitudes
    absorb a constant phase. Jones 2008 found one regressor set per volume within
    0.2 % of slice-specific ones in subject data. Measured on 19 TASKFORCE runs (wiki
    note, 2026-11-15; belt + pulse): cardiac at tzero after slice timing is significant
    (p < 0.001) in 15 % of voxels, against 17 % for each slice's own-time terms on raw
    slices and 16 % for those terms pushed through the same slice-timing operator.
    """
    s = diag["series"]
    cols, labels = [], []
    for i, sh in enumerate(rvt_shifts):
        cols.append(np.interp(times - sh, s["t_resp"], s["rv"]))
        labels.append(f"rvt{i:02d}")
    phases = [("resp", np.interp(times, s["t_resp"], s["phi_resp"]))]
    if cardiac:
        phases.insert(0, ("card", np.interp(times, s["t_card"], s["phi_card"])))
    for name, ph in phases:
        for m in range(1, order + 1):
            cols += [np.cos(m * ph), np.sin(m * ph)]
            labels += [f"{name}.c{m}", f"{name}.s{m}"]
    return np.stack(cols, 1), labels


def choose_source(
    traces: ExcitationTraces, band: tuple[float, float], window: float
) -> tuple[str, dict]:
    """The trace to read a band from, plus each trace's peak prominence as a diagnostic.

    Respiration comes from phase and cardiac from magnitude whenever phase exists:
    measured on TASKFORCE against belt and pulse oximeter, and in line with PREPAIR.
    Choosing by peak prominence instead was tried and fails -- on a rest run it picked
    phase for cardiac and tracked a heart rate 14 bpm off, because a narrow line in
    the phase spectrum out-scored the heartbeat.
    """
    scores = {}
    for which in ("magnitude", "phase"):
        if which == "phase" and traces.phase is None:
            continue
        t, x = traces.series(which, uniform=True)
        scores[which] = track_rate(t, x, band, window)[2]
    if traces.phase is None:
        return "magnitude", scores
    return ("phase" if band == RESP_BAND else "magnitude"), scores


def _drift_basis(nt: int, tr: float) -> torch.Tensor:
    """Orthonormal drift basis, on the CPU: it and the per-slice bases built from it
    are tiny dense factorizations, which lose to launch latency on a GPU."""
    cpu = torch.device("cpu")
    L = construct_polynomial_matrix(nt, _drift_degree(nt, tr), device=cpu, dtype=torch.float64)
    return _orthonormal_basis(L)


def _beyond_drift(UL: torch.Tensor, Xk: np.ndarray, device: torch.device) -> torch.Tensor:
    """Orthonormal basis of the regressors' part orthogonal to the drift basis ``UL``.

    With the two orthogonal, a fit of [drift, regressors] is the sum of two
    projections, so per-slice fits are a few small products on the slice's voxels
    instead of a least-squares solve with thousands of right-hand sides (136 ms vs
    3 ms per call on CPU). The SVD drops a regressor the drift spans, as lstsq did."""
    X = torch.from_numpy(np.asarray(Xk, dtype=np.float64))
    if X.shape[1] == 0:
        return X.to(device)
    Xo = X - UL @ (UL.T @ X)
    return _orthonormal_basis(Xo, scale=float(plinalg.svdvals(X)[0])).to(device)


def regress_out(
    data: np.ndarray,
    X: np.ndarray,
    tr: float,
    device: torch.device | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each voxel's physiological fit, slice by slice (AFNI's ``ricor`` block).

    ``data`` is (nx, ny, nz, nt) raw slices; ``X`` the slice-major slibase matrix
    (nt, nz * m). Each voxel is fitted with its own slice's ``m`` regressors together
    with Legendre drift, and only the regressors' part is subtracted -- the mean and
    the drift stay, since motion correction and the GLM model those themselves.
    The regressors are first made orthogonal to the drift basis, so a slow one (RVT)
    cannot take drift it does not own and the subtraction leaves mean and drift exact. Returns the cleaned series and the fraction of each voxel's drift-removed
    variance the regressors took (a QC map; 0 outside ``mask``).

    The regression spends ``m`` degrees of freedom per voxel that a later GLM does
    not see, as with afni_proc's ``ricor`` block.
    """
    device = linalg_device(device or torch.device("cpu"))  # float64 throughout
    nx, ny, nz, nt = data.shape
    m = X.shape[1] // nz
    if X.shape != (nt, nz * m):
        raise ValueError(f"regressors {X.shape} do not match {nt} volumes x {nz} slices")
    UL_cpu = _drift_basis(nt, tr)
    UL = UL_cpu.to(device)
    out = np.array(data, dtype=np.float32, copy=True)
    removed = np.zeros((nx, ny, nz), dtype=np.float32)
    if mask is None:
        mask = np.ones((nx, ny, nz), dtype=bool)
    for k in range(nz):
        sel_np = mask[:, :, k].reshape(-1, order="F")
        if not sel_np.any():
            continue
        sel = torch.from_numpy(sel_np).to(device)
        rows = _slice_rows(out, k, device)
        Y = rows[sel].T.to(torch.float64)
        # Regressors made orthogonal to the drift basis (afni_proc 3dDetrends them):
        # subtracting their fit then cannot move a voxel's mean or drift.
        UX = _beyond_drift(UL_cpu, X[:, k * m : (k + 1) * m], device)
        base = Y - UL @ (UL.T @ Y)  # drift-removed signal, before the physio fit
        fit = UX @ (UX.T @ base)
        frac = (fit * (2 * base - fit)).sum(0) / (base**2).sum(0).clamp_min(1e-12)
        rows = rows.clone()
        rows[sel] = (Y - fit).T.float()
        _put_slice_rows(out, k, rows)
        rm = np.zeros(nx * ny, dtype=np.float32)
        rm[sel_np] = frac.float().cpu().numpy()
        removed[:, :, k] = rm.reshape(nx, ny, order="F")
    return out, removed


def explained_maps(
    data: np.ndarray,
    X: np.ndarray,
    tr: float,
    groups: dict[str, list[int]],
    device: torch.device | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per voxel, the share of drift-removed variance each regressor group explains
    beyond all the others (unique R^2), on raw slices.

    ``X`` is the slice-major slibase matrix (nt, nz * m); ``groups`` names column
    indices within one slice's ``m`` (e.g. ``{"cardiac": [5, 6, 7, 8]}``). Each voxel
    is fitted with its own slice's regressors, so the maps are in the acquisition
    grid and overlay the raw mean image directly. The map's shape is the check
    PhysIO uses (Kasper 2017): cardiac on vessels, ventricles and the brain edge,
    respiration broad; a wrong trace leaves a flat or speckled map. Edges also catch
    motion-locked fits, since raw voxels are scanner locations. Returns maps
    (nx, ny, nz), 0 outside ``mask``, plus ``"all"``.
    """
    device = linalg_device(device or torch.device("cpu"))  # float64 throughout
    nx, ny, nz, nt = data.shape
    m = X.shape[1] // nz
    if X.shape != (nt, nz * m):
        raise ValueError(f"regressors {X.shape} do not match {nt} volumes x {nz} slices")
    UL_cpu = _drift_basis(nt, tr)
    UL = UL_cpu.to(device)
    if mask is None:
        mask = np.ones((nx, ny, nz), dtype=bool)
    out = {g: np.zeros((nx, ny, nz), dtype=np.float32) for g in ("all", *groups)}

    def explained(U, R):
        return ((U.T @ R) ** 2).sum(0)

    for k in range(nz):
        sel = mask[:, :, k].reshape(-1, order="F")
        if sel.sum() < 2:
            continue
        Y = _slice_rows(data, k, device)[torch.from_numpy(sel).to(device)].T.to(torch.float64)
        Xk = X[:, k * m : (k + 1) * m]
        R = Y - UL @ (UL.T @ Y)
        s0 = (R**2).sum(0).clamp_min(1e-12)
        full = explained(_beyond_drift(UL_cpu, Xk, device), R)
        vals = {"all": full / s0}
        for g, cols in groups.items():
            rest = [c for c in range(m) if c not in cols]
            vals[g] = (full - explained(_beyond_drift(UL_cpu, Xk[:, rest], device), R)) / s0
        for g, v in vals.items():
            plane = np.zeros(nx * ny, dtype=np.float32)
            plane[sel] = v.clamp_min(0).float().cpu().numpy()
            out[g][:, :, k] = plane.reshape(nx, ny, order="F")
    return out


# ---------------------------------------------------------------------------
# Standard regressor sets (TAPAS / physio_calc style), from published definitions.
# TAPAS is GPL-3 and this repository GPL-2-or-later, so nothing here is ported from
# it; TAPAS is used only as a black-box reference in validation.
# ---------------------------------------------------------------------------


def cardiac_response(t: np.ndarray) -> np.ndarray:
    """Cardiac response function (Chang, Cunningham & Glover 2009, NeuroImage 44:857)."""
    t = np.asarray(t, dtype=float)
    return 0.6 * t**2.7 * np.exp(-t / 1.6) - 16.0 / np.sqrt(2 * np.pi * 9) * np.exp(
        -0.5 * (t - 12) ** 2 / 9
    )


def respiration_response(t: np.ndarray) -> np.ndarray:
    """Respiration response function (Birn, Smith, Jones & Bandettini 2008, NeuroImage 40:644)."""
    t = np.asarray(t, dtype=float)
    return 0.6 * t**2.1 * np.exp(-t / 1.6) - 0.0023 * t**3.54 * np.exp(-t / 4.25)


def find_beats(
    t: np.ndarray, x: np.ndarray, band: tuple[float, float], f0: float | None = None
) -> np.ndarray:
    """Times of cycle peaks in a trace band-passed around its own rate.

    Used for the beat-to-beat cardiac phase and heart rate. ``f0`` (Hz) is the
    run's rate; estimated from the whole-run spectrum when not given.
    """
    fs = _fs(t)
    if f0 is None:
        xb = _bandpass(x, band[0], band[1], fs)
        f, P = signal.periodogram(xb * np.hanning(len(xb)), fs=fs, nfft=4 * len(xb))
        inb = (f >= band[0]) & (f <= band[1])
        f0 = float(f[inb][np.argmax(P[inb])])
    xc = _bandpass(x, max(band[0] * 0.8, 0.6 * f0), min(band[1] * 1.2, 1.8 * f0), fs)
    pk, _ = signal.find_peaks(xc, distance=max(1, int(0.6 * fs / f0)), prominence=0.3 * xc.std())
    return t[pk]


def beat_phase(peaks: np.ndarray, times: np.ndarray) -> np.ndarray:
    """RETROICOR cycle phase: 2*pi*k at the k-th peak, linear in time between peaks
    (Glover, Li & Ress 2000). Unwrapped; extrapolated at the ends by the end intervals."""
    k = np.arange(peaks.size, dtype=float)
    ph = np.interp(times, peaks, k)
    lo, hi = times < peaks[0], times > peaks[-1]
    ph[lo] = (times[lo] - peaks[0]) / (peaks[1] - peaks[0])
    ph[hi] = k[-1] + (times[hi] - peaks[-1]) / (peaks[-1] - peaks[-2])
    return 2 * np.pi * ph


def beat_rate(peaks: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Instantaneous rate (per minute) from peak intervals, at ``times``."""
    mid = 0.5 * (peaks[1:] + peaks[:-1])
    rate = 60.0 / np.diff(peaks)
    rate = signal.medfilt(rate, 3) if rate.size >= 3 else rate
    return np.interp(times, mid, rate)


def _convolve_at(
    t_src: np.ndarray, x_src: np.ndarray, times: np.ndarray, kernel, dt: float = 0.1
) -> np.ndarray:
    """Convolve a slow series with a response function on a fine grid, sample at ``times``.

    The series is held at its first value before it starts, so the response does
    not open with a step from zero."""
    t0, t1 = min(t_src[0], times[0]) - 40.0, max(t_src[-1], times[-1])
    grid = np.arange(t0, t1 + dt, dt)
    x = np.interp(grid, t_src, x_src)
    x = x - x.mean()
    h = kernel(np.arange(0, 40.0, dt))
    y = np.convolve(x, h)[: grid.size] * dt
    return np.interp(times, grid, y)


@dataclass
class PhysioModel:
    """Everything the regressors are built from, wherever the traces came from."""

    card_peaks: np.ndarray  # beat times (s, from volume 0)
    resp_t: np.ndarray
    resp_phase: np.ndarray  # unwrapped respiratory phase at resp_t
    rvt: np.ndarray  # breath-by-breath RVT (Birn 2006) at resp_t
    source: str = ""


def respiration_phase_histogram(t: np.ndarray, x: np.ndarray, bins: int = 100) -> np.ndarray:
    """Respiratory phase by amplitude histogram equalisation (Glover, Li & Ress 2000).

    For a belt recording: each sample's phase is pi times the fraction of the run's
    samples below its amplitude, signed by whether the chest is filling or emptying,
    so a shallow breath reaches a smaller phase than a deep one. The sign is read
    from the trace low-passed at 1 Hz -- the raw derivative flips on every ripple.
    Measured against TAPAS PhysIO's own phase on a TASKFORCE run: circular agreement
    0.98 (0.99 on its own filtered trace). For a data-derived trace the Hilbert phase
    explained more variance (:func:`cycle_phase`).
    """
    fs = _fs(t)
    sos = signal.butter(2, [0.05, min(1.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    r = signal.sosfiltfilt(sos, x - x.mean())
    r = r - r.min()
    hist, edges = np.histogram(r, bins=bins)
    frac = np.interp(r, edges, np.concatenate([[0.0], np.cumsum(hist) / hist.sum()]))
    lp = signal.sosfiltfilt(signal.butter(2, min(1.0, 0.45 * fs), fs=fs, output="sos"), r)
    sign = np.sign(np.gradient(lp))
    sign[sign == 0] = 1.0
    return np.pi * frac * sign


def breath_rvt(t: np.ndarray, x: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Respiration volume per time, breath by breath (Birn et al. 2006, NeuroImage 31:1536).

    Depth of each breath (its peak minus the trough level there) over its period, at
    the peak times. Matches AFNI physio_calc.py's RVT on a TASKFORCE belt recording at
    r = 0.89. TAPAS PhysIO's "Hilbert" RVT is a different quantity: it and
    physio_calc agree only at r = 0.62 on the same recording, so there is no single
    reference to match; this is the classic definition. Returns (peak times, RVT).
    """
    fs = _fs(t)
    sos = signal.butter(2, [0.05, min(1.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x - x.mean())
    dist = max(1, int(0.6 * fs / rate_hz))
    pk, _ = signal.find_peaks(xb, distance=dist, prominence=0.3 * xb.std())
    tr, _ = signal.find_peaks(-xb, distance=dist, prominence=0.3 * xb.std())
    if pk.size < 3 or tr.size < 2:
        return np.array([t[0], t[-1]]), np.zeros(2)
    tp = t[pk]
    depth = xb[pk] - np.interp(tp, t[tr], xb[tr])
    return tp, depth / np.gradient(tp)


def model_from_series(
    t_c,
    x_c,
    t_r,
    x_r,
    card_f0: float | None = None,
    resp_phase: str = "histogram",
    source: str = "",
) -> PhysioModel:
    """Build the model from a cardiac and a respiratory trace (recorded or data-derived).

    ``resp_phase``: "histogram" (Glover; the standard for a belt) or "hilbert"
    (better for the data-derived trace)."""
    peaks = find_beats(t_c, x_c, CARDIAC_BAND, card_f0)
    rt, rf, _ = peak_rate(t_r, x_r, RESP_BAND)
    phase = (
        respiration_phase_histogram(t_r, x_r)
        if resp_phase == "histogram"
        else cycle_phase(t_r, x_r, rf)
    )
    bt, rvt = breath_rvt(t_r, x_r, float(np.median(rf)))
    return PhysioModel(peaks, t_r, np.unwrap(phase), np.interp(t_r, bt, rvt), source)


def model_from_traces(
    traces: ExcitationTraces, card_from: str = "magnitude", resp_from: str = "phase"
) -> PhysioModel:
    """The model from the data's excitation-rate traces (cardiac notched at k/TR)."""
    t_c, x_c = traces.series(card_from, uniform=True)
    t_r, x_r = traces.series(resp_from, uniform=True)
    tr = float(np.median(np.diff(traces.times[:, 0])))
    f_resp = float(np.median(peak_rate(t_r, x_r, RESP_BAND)[1]))
    t_c, x_c, harmonics = _cardiac_input(traces, card_from, t_c, x_c, tr, f_resp)
    _, cf, _ = track_rate(t_c, x_c, CARDIAC_BAND, 20.0, avoid=harmonics)
    model = model_from_series(
        t_c,
        x_c,
        t_r,
        x_r,
        card_f0=float(np.median(cf)),
        resp_phase="hilbert",
        source=f"data ({card_from}/{resp_from})",
    )
    if traces.field_coef is not None and traces.phase is not None:
        _, rf, _ = peak_rate(t_r, x_r, RESP_BAND)
        model.rvt = _depth_rvt(traces, t_r, x_r, None, rf)
    return model


def standard_regressors(
    model: PhysioModel,
    times: np.ndarray,
    card_order: int = 3,
    resp_order: int = 4,
    inter_order: int = 1,
    hrv: bool = True,
    rvt: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """RETROICOR (cardiac, respiratory, interaction) + HRV*CRF + RVT*RRF at ``times``.

    The set TAPAS PhysIO builds with orders c/r/cr and its HRV and RVT models; the
    defaults are its common c3 r4 cr1. Interaction terms are cos/sin of
    m*phi_c +- n*phi_r (Harvey et al. 2008, JMRI 28:1337).
    """
    phc = beat_phase(model.card_peaks, times)
    phr = np.interp(times, model.resp_t, model.resp_phase)
    cols, labels = [], []
    for m in range(1, card_order + 1):
        cols += [np.cos(m * phc), np.sin(m * phc)]
        labels += [f"card.c{m}", f"card.s{m}"]
    for m in range(1, resp_order + 1):
        cols += [np.cos(m * phr), np.sin(m * phr)]
        labels += [f"resp.c{m}", f"resp.s{m}"]
    for m in range(1, inter_order + 1):
        for sgn, name in ((1, "p"), (-1, "m")):
            ph = m * phc + sgn * m * phr
            cols += [np.cos(ph), np.sin(ph)]
            labels += [f"cxr.c{m}{name}", f"cxr.s{m}{name}"]
    if hrv:
        fine = np.arange(model.card_peaks[0], model.card_peaks[-1], 0.1)
        cols.append(_convolve_at(fine, beat_rate(model.card_peaks, fine), times, cardiac_response))
        labels.append("hrv.crf")
    if rvt:
        cols.append(_convolve_at(model.resp_t, model.rvt, times, respiration_response))
        labels.append("rvt.rrf")
    return np.stack(cols, 1), labels
