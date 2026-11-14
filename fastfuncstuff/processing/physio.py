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

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import signal

from fastfuncstuff import policy_linalg as plinalg

from ..glm.core import construct_polynomial_matrix

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
    notes: list[str] = field(default_factory=list)

    def series(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """(times, values) of one trace as a single series in acquisition order."""
        vals = self.magnitude if which == "magnitude" else self.phase
        if vals is None:
            raise ValueError("no phase trace: the run was given without phase")
        order = np.argsort(self.times.ravel(), kind="stable")
        return self.times.ravel()[order], vals.ravel()[order]


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
    magnitude: np.ndarray,
    slice_times: np.ndarray,
    tr: float,
    phase: np.ndarray | None = None,
    device: torch.device | None = None,
    mask: np.ndarray | None = None,
    stability: float = 0.7,
    comb: bool = True,
) -> ExcitationTraces:
    """One physiological sample per (volume, excitation).

    Parameters
    ----------
    magnitude, phase : (nx, ny, nz, nt)
        Raw slices -- before any motion correction or slice timing, so that every
        sample is still at its own acquisition time. ``phase`` may be raw scanner
        units; only its *change* over time is used, so no spatial unwrap is needed.
    slice_times : (nz,)
        Acquisition offset of each slice within the TR (s); slice ``k`` is the
        file's third axis.
    mask : (nx, ny, nz) bool, optional
        Voxels to average. Default: brighter than a quarter of the 98th percentile
        of the mean image.
    stability : float
        Phase voxels must have a mean phasor magnitude above this (their phase is
        steady enough over the run for a change to be read).

    Notes
    -----
    Magnitude: each voxel's mean and Legendre drift are removed and the residual
    divided by its mean, then averaged over the excitation's voxels. Phase: each
    voxel's change against its own temporal complex mean, ``angle(z_t * conj(zbar))``,
    drift removed, averaged with weight ``mean magnitude**2`` (phase noise scales as
    1/magnitude).
    """
    device = device or torch.device("cpu")
    _, _, nz, nt = magnitude.shape
    group, gtimes = excitation_groups(slice_times)
    G = len(gtimes)
    mean_img = magnitude.mean(-1)
    if mask is None:
        thr = 0.25 * np.percentile(mean_img[mean_img > 0], 98)
        mask = mean_img > thr
    deg = _drift_degree(nt, tr)
    L = construct_polynomial_matrix(nt, deg, device=device)
    proj = L @ plinalg.pinv(L)  # (nt, nt) projector onto the drift space
    per_slice_m = np.zeros((nt, nz))
    per_slice_p = np.zeros((nt, nz))
    count_m = np.zeros(nz)
    count_p = np.zeros(nz)
    if phase is not None:
        # The run's own range onto [-pi, pi], as complex.scale_phase_to_radians does,
        # but with one min/max for every slice so all slices share the mapping.
        lo, hi = float(phase.min()), float(phase.max())
        if not hi > lo:
            raise ValueError("phase has no range; expected raw scanner units or radians")
    for k in range(nz):
        sel = torch.from_numpy(mask[:, :, k].reshape(-1)).to(device)
        if int(sel.sum()) < 20:
            continue
        m = (
            torch.from_numpy(np.ascontiguousarray(magnitude[:, :, k, :]))
            .to(device)
            .reshape(-1, nt)[sel]
        )
        mu = m.mean(1)
        frac = (m - m @ proj.T) / mu.clamp_min(1e-6)[:, None]
        per_slice_m[:, k] = frac.mean(0).cpu().numpy()
        count_m[k] = float(m.shape[0])
        if phase is not None:
            p = (
                torch.from_numpy(np.ascontiguousarray(phase[:, :, k, :]))
                .to(device)
                .reshape(-1, nt)[sel]
            )
            p = (p - lo) * (2 * np.pi / (hi - lo)) - np.pi
            z = torch.polar(torch.ones_like(p), p)
            zbar = z.mean(1)
            keep = zbar.abs() > stability
            if int(keep.sum()) < 20:
                continue
            dphi = torch.angle(z[keep] * torch.conj(zbar[keep] / zbar[keep].abs())[:, None])
            dphi = dphi - dphi @ proj.T
            w = mu[keep] ** 2
            per_slice_p[:, k] = ((w[:, None] * dphi).sum(0) / w.sum()).cpu().numpy()
            count_p[k] = float(keep.sum())
    mag_t = np.zeros((nt, G))
    ph_t = np.zeros((nt, G)) if phase is not None else None
    for g in range(G):
        s = group == g
        if count_m[s].sum() > 0:
            mag_t[:, g] = per_slice_m[:, s] @ (count_m[s] / count_m[s].sum())
        if ph_t is not None and count_p[s].sum() > 0:
            ph_t[:, g] = per_slice_p[:, s] @ (count_p[s] / count_p[s].sum())
    if comb:
        mag_t = remove_tr_comb(mag_t, tr)
    times = np.arange(nt)[:, None] * tr + gtimes[None, :]
    return ExcitationTraces(times, group, mag_t, ph_t, int(count_m.sum()))


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
    fs = 1.0 / float(np.median(np.diff(t)))
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
    b, a = signal.butter(order, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs)
    return signal.filtfilt(b, a, x - x.mean())


def track_rate(
    t: np.ndarray,
    x: np.ndarray,
    band: tuple[float, float],
    window: float,
    step: float = 2.0,
    avoid: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fundamental frequency over time within ``band``.

    Sliding Hann-windowed periodograms (zero-padded 16x, parabolic peak refinement),
    each window's peak searched near the previous one so a harmonic or a noise
    burst cannot capture the track. Frequencies within 0.03 Hz of any in ``avoid``
    (the TR harmonics) are never chosen. Returns window centres, frequencies (Hz) and
    the peak prominence (in-band peak power over in-band median, on the whole-run
    spectrum) used to choose between magnitude and phase.
    """
    fs = 1.0 / float(np.median(np.diff(t)))
    xb = _bandpass(x, band[0] * 0.8, band[1] * 1.2, fs)
    n = int(window * fs)
    s = max(1, int(step * fs))
    f_all, P_all = signal.periodogram(xb * np.hanning(len(xb)), fs=fs, nfft=4 * len(xb))

    def allowed(f: np.ndarray) -> np.ndarray:
        ok = (f >= band[0]) & (f <= band[1])
        for fa in avoid:
            ok &= np.abs(f - fa) > 0.03
        return ok

    inb = allowed(f_all)
    prominence = float(P_all[inb].max() / np.median(P_all[inb]))
    f0_run = float(f_all[inb][np.argmax(P_all[inb])])
    cen, est = [], []
    prev = f0_run
    for i in range(0, max(1, len(xb) - n), s):
        seg = xb[i : i + n]
        f, P = signal.periodogram(seg * np.hanning(len(seg)), fs=fs, nfft=16 * len(seg))
        near = allowed(f) & (np.abs(f - prev) <= 0.25 * prev)
        if not near.any():
            near = allowed(f)
        j = int(np.flatnonzero(near)[np.argmax(P[near])])
        if 0 < j < len(P) - 1:  # parabolic refinement on log power
            a_, b_, c_ = np.log(P[j - 1 : j + 2] + 1e-30)
            j = j + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-30)
        fj = float(np.interp(j, np.arange(len(f)), f))
        est.append(fj)
        cen.append(float(t[min(i + n // 2, len(t) - 1)]))
        prev = fj
    return np.array(cen), np.array(est), prominence


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
    """
    fs = 1.0 / float(np.median(np.diff(t)))
    xb = _bandpass(x, band[0] * 0.8, band[1] * 1.2, fs)
    f_all, P_all = signal.periodogram(xb * np.hanning(len(xb)), fs=fs, nfft=4 * len(xb))
    inb = (f_all >= band[0]) & (f_all <= band[1])
    f0 = float(f_all[inb][np.argmax(P_all[inb])])
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
    fs = 1.0 / float(np.median(np.diff(t)))
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
    fs = 1.0 / float(np.median(np.diff(t)))
    f0 = float(np.median(rate_f))
    xb = _bandpass(x, 0.6 * f0, 1.6 * f0, fs)
    env = np.abs(signal.hilbert(xb))
    # smooth to one breath, so the envelope is a breath-by-breath depth, not a ripple
    b, a = signal.butter(2, min(0.45 * fs, 0.8 * f0), fs=fs)
    env = signal.filtfilt(b, a, env)
    rate = np.interp(t, rate_t, rate_f)
    return env * rate


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
    acquisition time, which is exactly one of the excitation samples -- so no
    interpolation in time is involved. Columns per slice, matching ``physio_calc.py``:
    ``rvt00..rvtNN, card.c1, card.s1, card.c2, card.s2, resp.c1, resp.s1, resp.c2,
    resp.s2``. Returns ``(matrix (nt, nslices*m), column labels, diagnostics)``.
    """
    t_c, x_c = traces.series(card_from)
    t_r, x_r = traces.series(resp_from)
    tr = float(np.median(np.diff(traces.times[:, 0])))
    harmonics = tuple(k / tr for k in range(1, int(CARDIAC_BAND[1] * tr * 1.2) + 2))
    if card_from == "magnitude":
        x_c = notch_tr_harmonics(t_c, x_c, tr, CARDIAC_BAND)
    ct, cf, cprom = track_rate(t_c, x_c, CARDIAC_BAND, card_window, avoid=harmonics)
    rt, rf, rprom = peak_rate(t_r, x_r, RESP_BAND)
    phi_c = cycle_phase(t_c, x_c, cf)
    phi_r = cycle_phase(t_r, x_r, rf)
    rv = respiration_volume(t_r, x_r, rt, rf)

    nt, G = traces.times.shape
    order_idx = np.argsort(traces.times.ravel(), kind="stable")
    rank = np.empty_like(order_idx)
    rank[order_idx] = np.arange(order_idx.size)

    def per_excitation(v: np.ndarray) -> np.ndarray:  # series in time order -> (nt, G)
        return v[rank].reshape(nt, G)

    pc, pr, rvx = per_excitation(phi_c), per_excitation(phi_r), per_excitation(rv)
    rv_shifted = [per_excitation(np.interp(t_r - s, t_r, rv)) for s in rvt_shifts]
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


def volume_regressors(
    diag: dict,
    times: np.ndarray,
    rvt_shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
    order: int = 2,
    cardiac: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Regressors for data that every voxel samples at the same ``times``.

    After a slice-timing-corrected resample every voxel stands for the instant
    ``f*TR + tzero``, so the physiological model is evaluated there, once, and used as
    a GLM nuisance like the motion parameters. Measured on ImaginationStimulation:
    respiration + RVT as nuisance after the resample raised held-out R2 (top 0.5 %:
    33.5 -> 35.7, 87 % of voxels better), while the same terms regressed out of the
    raw slices before motion correction LOWERED it (to 31.0) -- on raw slices they
    mostly fit breathing-locked apparent motion, which motion correction and slice
    timing then remove themselves, so the subtraction became structured noise.

    Cardiac is off by default: at the volume rate it is aliased, and slice timing
    cannot shift an aliased component correctly (it acts at the alias frequency).
    Measured +0.1 there -- harmless, useless.
    """
    s = diag["series"]
    cols, labels = [], []
    for i, sh in enumerate(rvt_shifts):
        cols.append(np.interp(times - sh, s["t_resp"], s["rv"]))
        labels.append(f"rvt{i:02d}")
    phr = np.interp(times, s["t_resp"], s["phi_resp"])
    for m in range(1, order + 1):
        cols += [np.cos(m * phr), np.sin(m * phr)]
        labels += [f"resp.c{m}", f"resp.s{m}"]
    if cardiac:
        phc = np.interp(times, s["t_card"], s["phi_card"])
        for m in range(1, order + 1):
            cols += [np.cos(m * phc), np.sin(m * phc)]
            labels += [f"card.c{m}", f"card.s{m}"]
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
        t, x = traces.series(which)
        scores[which] = track_rate(t, x, band, window)[2]
    if traces.phase is None:
        return "magnitude", scores
    return ("phase" if band == RESP_BAND else "magnitude"), scores


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
    device = device or torch.device("cpu")
    nx, ny, nz, nt = data.shape
    m = X.shape[1] // nz
    if X.shape != (nt, nz * m):
        raise ValueError(f"regressors {X.shape} do not match {nt} volumes x {nz} slices")
    L = construct_polynomial_matrix(nt, _drift_degree(nt, tr), device=device, dtype=torch.float64)
    out = np.array(data, dtype=np.float32, copy=True)
    removed = np.zeros((nx, ny, nz), dtype=np.float32)
    if mask is None:
        mask = np.ones((nx, ny, nz), dtype=bool)
    for k in range(nz):
        sel = mask[:, :, k].reshape(-1)
        if not sel.any():
            continue
        Y = torch.from_numpy(out[:, :, k, :].reshape(-1, nt)[sel].T.astype(np.float64)).to(device)
        Xk = torch.from_numpy(X[:, k * m : (k + 1) * m].astype(np.float64)).to(device)
        # Regressors made orthogonal to the drift basis (afni_proc 3dDetrends them):
        # subtracting their fit then cannot move a voxel's mean or drift.
        Xk = Xk - L @ plinalg.lstsq(L, Xk).solution
        A = torch.cat([L, Xk], 1)
        beta = plinalg.lstsq(A, Y).solution
        fit = Xk @ beta[L.shape[1] :]
        base = Y - L @ beta[: L.shape[1]]  # drift-removed signal, before the physio fit
        frac = (fit * (2 * base - fit)).sum(0) / (base**2).sum(0).clamp_min(1e-12)
        cleaned = (Y - fit).T.float().cpu().numpy()
        plane = out[:, :, k, :].reshape(-1, nt)
        plane[sel] = cleaned
        out[:, :, k, :] = plane.reshape(nx, ny, nt)
        rm = np.zeros(nx * ny, dtype=np.float32)
        rm[sel] = frac.float().cpu().numpy()
        removed[:, :, k] = rm.reshape(nx, ny)
    return out, removed


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
    fs = 1.0 / float(np.median(np.diff(t)))
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
    fs = 1.0 / float(np.median(np.diff(t)))
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
    fs = 1.0 / float(np.median(np.diff(t)))
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
    t_c, x_c = traces.series(card_from)
    t_r, x_r = traces.series(resp_from)
    tr = float(np.median(np.diff(traces.times[:, 0])))
    harmonics = tuple(k / tr for k in range(1, int(CARDIAC_BAND[1] * tr * 1.2) + 2))
    if card_from == "magnitude":
        x_c = notch_tr_harmonics(t_c, x_c, tr, CARDIAC_BAND)
    _, cf, _ = track_rate(t_c, x_c, CARDIAC_BAND, 20.0, avoid=harmonics)
    return model_from_series(
        t_c, x_c, t_r, x_r, card_f0=float(np.median(cf)), source=f"data ({card_from}/{resp_from})"
    )


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


def read_cmrr_log(path) -> tuple[np.ndarray, np.ndarray]:
    """(times in s, values) from a CMRR physio log (RESP / PULS / ECG channel 1),
    placed by each sample's own 2.5 ms tick -- gaps cost nothing."""
    t, v = [], []
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit() and p[1] in ("RESP", "PULS", "ECG1", "ECG"):
            t.append(int(p[0]))
            v.append(float(p[2]))
    return np.asarray(t, dtype=float) * 0.0025, np.asarray(v, dtype=float)


def cmrr_volume_zero(info_path) -> float:
    """Start of the first acquisition (s, same clock as the logs) from a CMRR Info log."""
    first = None
    for line in open(info_path):
        p = line.split()
        if len(p) == 5 and all(x.isdigit() for x in p):
            t = int(p[2]) * 0.0025
            first = t if first is None else min(first, t)
    if first is None:
        raise ValueError(f"no acquisition rows in {info_path}")
    return first
