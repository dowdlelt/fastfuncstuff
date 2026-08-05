"""Partition-axis apparent-shift correction for multi-echo 3-D EPI / GRE.

A 3-D-encoded volume is one shot: every partition (the slow phase-encode axis) is
acquired at a different moment inside the same TR. A steady frequency change over
that shot — scanner drift, breathing, a warming gradient — shows up in the image
not as blur but as a *rigid translation along the partition axis*, and because the
accumulated phase error scales with echo time, that translation is DIFFERENT for
every echo of a multi-echo readout::

    shift_e(t) = m(t) · TE_e            [voxels]

That is exactly the assumption ffs_moco's multi-echo mode violates: it estimates
one rigid pose from one echo and applies it to all of them. Removing the
TE-dependent part first restores "a voxel is the same voxel across echoes", and
leaves the echo-COMMON bulk translation (the fit's intercept, discarded here) to
rigid motion correction where it belongs. Doing it before :mod:`locomoco` matters
even more: a whole-volume translation of a couple of voxels is not residual
non-linear motion, and feeding it to a searchlight estimator just corrupts the
local fields.

Estimation is a global inter-echo cross-correlation — deliberately one giant patch
covering the whole volume, which is why it needs no mask and no regularisation to
be stable (contrast with locomoco, where small patches force both). Consecutive
echoes are correlated as a function of a trial sub-voxel shift, the peak is found
per timepoint, and the pairwise answers are cumulated to give each echo's shift
relative to echo 1. A line is then fit through those against TE and applied
THROUGH THE ORIGIN, so echo 1 is corrected too.

Sub-voxel shifting is the Fourier shift theorem, not interpolation: one phase ramp
in k-space along the partition axis. It is sinc-exact, so a fractional shift adds
no smoothing (a linear interpolator would low-pass the data and, worse, inflate the
correlation at integer shifts — the search would lock onto whole voxels).

Sign convention throughout: a shift ``d`` PUSHES image content toward +axis, i.e.
``out(x) = in(x - d)``. This matches the reference implementation this module was
built against (R. Stirnberg, ``gre3d_slicedriftcorrection.py``) so the saved shift
traces are directly comparable. Note it is the opposite of
:func:`~fastfuncstuff.processing.locomoco._fourier_shifter`, whose scalar ``s`` is
a pull.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.memory import get_available_memory

# View ordering constrains the SIGN of the shift: partitions encoded in a fixed
# order accumulate the drift phase monotonically, so a known ordering halves the
# search and keeps a noisy timepoint from flipping sign. "unknown" searches both.
ORDERING_SIGN: dict[str, tuple[float, float]] = {
    "ascending": (-1.0, 0.0),
    "descending": (0.0, 1.0),
    "unknown": (-1.0, 1.0),
}


def axis_to_dim(axis: int, ndim: int) -> int:
    """NIfTI voxel axis (0/1/2) -> tensor dim for our ``(…, nz, ny, nx)`` layout."""
    return ndim - 1 - axis


def search_bounds(ordering: str, max_shift: float) -> tuple[float, float]:
    """(lo, hi) shift search bounds in voxels for a view ordering."""
    try:
        lo, hi = ORDERING_SIGN[ordering]
    except KeyError:
        raise ValueError(
            f"unknown view ordering '{ordering}'; use ascending / descending / unknown"
        ) from None
    return lo * max_shift, hi * max_shift


@dataclass
class ShiftEstimate:
    """Per-timepoint, per-echo partition-axis shifts, in voxels.

    Attributes:
        pairwise: ``(T, E)`` shift of echo e relative to echo e-1 (column 0 is 0).
        cumulative: ``(T, E)`` cumulative sum of ``pairwise`` — each echo relative
            to echo 1, which is what the raw cross-correlation actually measures.
        corr: ``(T, E)`` peak correlation of each pair (column 0 is NaN). The
            quality trace: a value that dives on some timepoints means the shift
            there is not trustworthy.
        slope: ``(T,)`` fitted voxels per ms of TE, or None without echo times.
        intercept: ``(T,)`` fitted echo-common offset. DISCARDED when applying —
            it is a bulk translation and rigid moco removes it properly (this term
            is reported only as a diagnostic).
        applied: ``(T, E)`` the shifts actually corrected for.
    """

    pairwise: np.ndarray
    cumulative: np.ndarray
    corr: np.ndarray
    slope: np.ndarray | None
    intercept: np.ndarray | None
    applied: np.ndarray

    @property
    def frequency_drift_hz(self) -> np.ndarray | None:
        """Total frequency change over the shot, per timepoint, in Hz.

        One voxel of partition-direction displacement corresponds to one cycle of
        phase accumulated across the shot, so voxels-per-second-of-TE reads
        directly as Hz. Sign follows the reference: a negative slope (content
        pushed toward -axis with TE) is a positive frequency change.
        """
        if self.slope is None:
            return None
        return -self.slope * 1.0e3  # voxels/ms -> voxels/s == Hz


# ---------------------------------------------------------------------------
# Fourier shifting
# ---------------------------------------------------------------------------


def _replicate_pad(x: Tensor, dim: int, pad: int) -> Tensor:
    """Edge-replicate ``pad`` samples on both ends of ``dim``.

    The FFT shift is circular, so without this the trailing partitions fold into
    the leading ones. Replicating rather than zero-padding (what the reference
    does) avoids manufacturing a hard step at the array edge, which would ring.
    """
    if pad <= 0:
        return x
    n = x.shape[dim]
    lo = x.narrow(dim, 0, 1).expand(*[pad if i == dim else -1 for i in range(x.ndim)])
    hi = x.narrow(dim, n - 1, 1).expand(*[pad if i == dim else -1 for i in range(x.ndim)])
    return torch.cat([lo, x, hi], dim=dim)


class _ShiftBank:
    """Forward transform of one batch, reusable for any number of trial shifts.

    The forward FFT is the expensive part and does not depend on the shift, so it
    is computed once here; each :meth:`shift` is a phase multiply plus an inverse
    transform. Real input uses ``rfft`` (half the spectrum, half the memory).
    """

    def __init__(self, data: Tensor, dim: int, pad: int):
        self.dim = dim
        self.pad = pad
        self.n = data.shape[dim]
        self.dtype = data.dtype
        padded = _replicate_pad(data, dim, pad)
        self.npad = padded.shape[dim]
        self.complex_in = padded.is_complex()
        if self.complex_in:
            self.spec = torch.fft.fft(padded, dim=dim)
            freq = torch.fft.fftfreq(self.npad, device=data.device)
        else:
            self.spec = torch.fft.rfft(padded.float(), dim=dim)
            freq = torch.fft.rfftfreq(self.npad, device=data.device)
        shape = [1] * padded.ndim
        shape[dim] = freq.numel()
        # -2πk/N: multiplying the spectrum by exp(-i·k·d) yields in(x - d), a push
        # of the content toward +axis by d voxels.
        self.k = (-2.0 * math.pi * freq).reshape(shape)

    def shift(self, d: Tensor | float) -> Tensor:
        """Shift by ``d`` voxels (scalar, or one value per batch element)."""
        if not torch.is_tensor(d):
            d = torch.as_tensor(float(d), device=self.spec.device)
        d = d.reshape(list(d.shape) + [1] * (self.spec.ndim - d.ndim))
        ramp = torch.polar(torch.ones_like(self.k), self.k * d)
        if self.complex_in:
            out = torch.fft.ifft(self.spec * ramp, dim=self.dim)
        else:
            out = torch.fft.irfft(self.spec * ramp, n=self.npad, dim=self.dim)
        out = out.narrow(self.dim, self.pad, self.n) if self.pad else out
        return out.to(self.dtype)


# ---------------------------------------------------------------------------
# Correlation search
# ---------------------------------------------------------------------------


def _inner_half(x: Tensor, dim: int) -> Tensor:
    """The central half of ``dim`` — the reference's wrap-around guard.

    Even with padding, the outermost partitions are the least trustworthy part of
    a shifted volume, and they are also where a 3-D EPI's slab profile rolls off.
    Correlating only the middle half keeps the metric on real signal.
    """
    n = x.shape[dim]
    return x.narrow(dim, n // 4, max(1, (3 * n) // 4 - n // 4))


def _corr(a: Tensor, b: Tensor, w: Tensor | None) -> Tensor:
    """(Weighted) Pearson r between ``a`` and ``b``, batched over dim 0.

    ``w`` is a static per-voxel weight broadcastable over the batch. Weighting is
    optional and off by default: over the whole volume the correlation is already
    dominated by brain, and a hard mask would bias the metric toward edges, where
    the late echoes have lost signal.
    """
    flat = (a.flatten(1), b.flatten(1))
    if w is None:
        a_c = flat[0] - flat[0].mean(1, keepdim=True)
        b_c = flat[1] - flat[1].mean(1, keepdim=True)
        num = (a_c * b_c).sum(1)
        den = a_c.pow(2).sum(1).sqrt() * b_c.pow(2).sum(1).sqrt()
    else:
        wf = w.flatten().unsqueeze(0)
        sw = wf.sum()
        a_c = flat[0] - (wf * flat[0]).sum(1, keepdim=True) / sw
        b_c = flat[1] - (wf * flat[1]).sum(1, keepdim=True) / sw
        num = (wf * a_c * b_c).sum(1)
        den = (wf * a_c.pow(2)).sum(1).sqrt() * (wf * b_c.pow(2)).sum(1).sqrt()
    return num / den.clamp_min(1e-12)


def _parabola_peak(curve: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
    """Sub-step peak of ``curve`` ``(B, S)`` sampled at ``offsets`` ``(B, S)`` or ``(S,)``.

    Three-point parabola through the sampled maximum and its neighbours. The
    correlation-vs-shift curve is smooth and locally quadratic near its peak (it is
    a sinc-exact shift, so no interpolation ripple), which is what makes a coarse
    grid plus this refinement as accurate as the reference's Brent search at a
    fraction of the evaluations — and, unlike Brent, fully batched over time.
    """
    b, s = curve.shape
    if offsets.ndim == 1:
        offsets = offsets.unsqueeze(0).expand(b, s)
    idx = curve.argmax(dim=1)
    ic = idx.clamp(1, s - 2)
    rows = torch.arange(b, device=curve.device)
    y0, y1, y2 = curve[rows, ic - 1], curve[rows, ic], curve[rows, ic + 1]
    x0, x1, x2 = offsets[rows, ic - 1], offsets[rows, ic], offsets[rows, ic + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = torch.where(denom.abs() > 1e-12, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom))
    # A parabola through three samples cannot legitimately peak outside them; if it
    # claims to, the sampled max is on the search boundary and is the honest answer.
    delta = delta.clamp(-1.0, 1.0)
    step = torch.where(delta >= 0, x2 - x1, x1 - x0)
    peak_x = x1 + delta * step
    peak_y = y1 + 0.25 * (y0 - y2) * delta
    on_edge = (idx == 0) | (idx == s - 1)
    return torch.where(on_edge, offsets[rows, idx], peak_x), torch.where(
        on_edge, curve[rows, idx], peak_y
    )


def _search_chunk(
    ref: Tensor,
    mov: Tensor,
    dim: int,
    lo: float,
    hi: float,
    coarse_step: float,
    fine_step: float,
    weight: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Peak-correlation shift for one time chunk. ``ref``/``mov`` are ``(B, …)``."""
    pad = int(math.ceil(max(abs(lo), abs(hi)))) + 1
    bank = _ShiftBank(mov, dim, pad)
    ref_s = _inner_half(ref, dim)
    # The weight is a single volume; give it the batch axis so `dim` addresses the
    # same spatial axis it does in ref/mov.
    w_s = _inner_half(weight.unsqueeze(0), dim).squeeze(0) if weight is not None else None

    n_coarse = max(3, int(round((hi - lo) / coarse_step)) + 1)
    coarse = torch.linspace(lo, hi, n_coarse, device=ref.device)
    curve = torch.stack(
        [_corr(ref_s, _inner_half(bank.shift(float(s)), dim), w_s) for s in coarse], dim=1
    )
    center, _ = _parabola_peak(curve, coarse)

    # Local refinement: the grid is now per-timepoint, which costs exactly the same
    # as a shared one (the ramp is already a broadcast multiply).
    half = int(math.ceil(coarse_step / (2.0 * fine_step)))
    fine_off = torch.arange(-half, half + 1, device=ref.device, dtype=ref.dtype) * fine_step
    trials = (center.unsqueeze(1) + fine_off.unsqueeze(0)).clamp(lo, hi)
    curve_f = torch.stack(
        [
            _corr(ref_s, _inner_half(bank.shift(trials[:, j]), dim), w_s)
            for j in range(trials.shape[1])
        ],
        dim=1,
    )
    peak, corr = _parabola_peak(curve_f, trials)
    return peak.clamp(lo, hi), corr


def _time_chunk(n_vox: int, device: torch.device, n_t: int) -> int:
    """Timepoints per chunk. The spectrum, the ramp product, and the inverse
    transform are all live at once, so budget generously per voxel."""
    bytes_per_t = n_vox * 48
    avail = get_available_memory(device)
    return int(max(1, min(n_t, avail // max(1, bytes_per_t))))


def estimate_pair_shift(
    ref: Tensor,
    mov: Tensor,
    axis: int,
    *,
    ordering: str = "unknown",
    max_shift: float = 5.0,
    coarse_step: float = 0.25,
    fine_step: float = 0.005,
    weight: Tensor | None = None,
    device: torch.device | None = None,
    desc: str = "shift search",
    disable_pbar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-voxel partition-axis shift of ``mov`` relative to ``ref``, per timepoint.

    Both are ``(T, nz, ny, nx)`` (or 3-D, treated as ``T=1``) on any device;
    ``axis`` is the NIfTI voxel axis of the partition direction. Returns
    ``(shift, corr)``, each ``(T,)`` — the shift being the value that, PUSHED onto
    ``mov``, best matches ``ref``.
    """
    lo, hi = search_bounds(ordering, max_shift)
    if ref.ndim == 3:  # a lone volume is just T=1
        ref, mov = ref.unsqueeze(0), mov.unsqueeze(0)
    if ref.shape != mov.shape:
        raise ValueError(f"shape mismatch: ref {tuple(ref.shape)} vs mov {tuple(mov.shape)}")
    device = device or ref.device
    dim = axis_to_dim(axis, ref.ndim)
    n_t = ref.shape[0]
    n_vox = int(np.prod(ref.shape[1:]))
    step = _time_chunk(n_vox, device, n_t)
    if weight is not None:
        weight = weight.to(device=device, dtype=torch.float32)

    shifts = np.zeros(n_t, dtype=np.float64)
    corrs = np.zeros(n_t, dtype=np.float64)
    for start in tqdm(
        range(0, n_t, step),
        desc=desc,
        unit="chunk",
        leave=True,
        disable=disable_pbar or n_t <= step,
        ncols=80,
    ):
        stop = min(start + step, n_t)
        r = ref[start:stop].to(device=device, dtype=torch.float32)
        m = mov[start:stop].to(device=device, dtype=torch.float32)
        s, c = _search_chunk(r, m, dim, lo, hi, coarse_step, fine_step, weight)
        shifts[start:stop] = s.double().cpu().numpy()
        corrs[start:stop] = c.double().cpu().numpy()
        del r, m, s, c
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return shifts, corrs


def apply_shift(
    data: Tensor,
    shifts: np.ndarray | Tensor,
    axis: int,
    *,
    device: torch.device | None = None,
    disable_pbar: bool = True,
) -> Tensor:
    """Push ``data`` by a per-timepoint shift along ``axis``. Returns a CPU tensor.

    ``data`` is ``(T, nz, ny, nx)`` (or 3-D with a scalar/1-element ``shifts``) and
    may be complex, in which case magnitude AND phase are shifted coherently.
    """
    squeeze = data.ndim == 3
    if squeeze:
        data = data.unsqueeze(0)
    device = device or data.device
    dim = axis_to_dim(axis, data.ndim)
    s = torch.as_tensor(np.asarray(shifts, dtype=np.float64).reshape(-1))
    if s.numel() != data.shape[0]:
        raise ValueError(f"got {s.numel()} shifts for {data.shape[0]} volumes")
    pad = int(math.ceil(float(s.abs().max().item()))) + 1
    n_t = data.shape[0]
    n_vox = int(np.prod(data.shape[1:]))
    step = _time_chunk(n_vox, device, n_t)

    out = torch.empty_like(data, device="cpu")
    for start in tqdm(
        range(0, n_t, step),
        desc="apply shift",
        unit="chunk",
        leave=True,
        disable=disable_pbar or n_t <= step,
        ncols=80,
    ):
        stop = min(start + step, n_t)
        block = data[start:stop].to(device=device)
        if not block.is_complex():
            block = block.float()
        bank = _ShiftBank(block, dim, pad)
        out[start:stop] = bank.shift(s[start:stop].to(device=device, dtype=torch.float32)).cpu()
        del block, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# TE regression
# ---------------------------------------------------------------------------


def te_regression(cumulative: np.ndarray, tes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-timepoint least-squares fit of ``cumulative`` shift against echo time.

    ``cumulative`` is ``(T, E)`` voxels, ``tes`` is ``(E,)`` ms. Returns
    ``(slope, intercept)``, each ``(T,)``, with slope in voxels per ms. The fit
    INCLUDES an intercept (the data has one: echo 1's own uncorrected shift sets
    the level) but callers apply ``slope · TE`` only.
    """
    tes = np.asarray(tes, dtype=np.float64)
    if cumulative.shape[1] != tes.size:
        raise ValueError(f"{cumulative.shape[1]} echoes but {tes.size} echo times")
    if tes.size < 2:
        raise ValueError("TE regression needs at least 2 echo times")
    x = tes - tes.mean()
    sxx = float((x * x).sum())
    slope = (cumulative * x[None, :]).sum(axis=1) / sxx
    intercept = cumulative.mean(axis=1) - slope * tes.mean()
    return slope, intercept


def signal_weight(volume: Tensor) -> Tensor:
    """Soft per-voxel weight from mean intensity, normalised to a [0, 1] peak.

    Deliberately NOT a mask. The inter-echo correlation loses signal at brain
    edges in the late echoes, so a hard boundary would let the edge decide the
    shift; a smooth intensity weight just tilts the metric toward tissue.
    """
    mean = volume.float().mean(dim=0) if volume.ndim == 4 else volume.float()
    return (mean / mean.max().clamp_min(1e-12)).clamp_min(0.0)


def estimate_shifts(
    echoes: Iterable[Tensor],
    axis: int,
    *,
    tes: np.ndarray | None = None,
    ordering: str = "unknown",
    max_shift: float = 5.0,
    coarse_step: float = 0.25,
    fine_step: float = 0.005,
    weight: Tensor | str | None = None,
    device: torch.device | None = None,
    verb: int = 1,
) -> ShiftEstimate:
    """Full estimate over the per-echo series, in echo order and all the same shape.

    Correlates consecutive echoes, cumulates, and — when ``tes`` (ms) is given —
    fits the TE line and applies it through the origin so every echo including the
    first is corrected. Without ``tes`` the raw cumulative shifts are applied and
    echo 1 is left alone.

    ``echoes`` is consumed lazily and only two are ever held at once, so a caller
    with the data on disk can pass a generator that loads one echo at a time
    instead of holding the whole multi-echo series in RAM. ``weight`` may be a
    per-voxel tensor, ``"signal"`` to derive one from the first echo (see
    :func:`signal_weight`), or None for an unweighted whole-volume Pearson r.
    """
    lo, hi = search_bounds(ordering, max_shift)
    shifts: list[np.ndarray] = []
    corrs: list[np.ndarray] = []
    prev: Tensor | None = None
    for e, cur in enumerate(echoes):
        if prev is None:
            if weight == "signal":
                weight = signal_weight(cur)
            n_t = cur.shape[0] if cur.ndim == 4 else 1
        else:
            if cur.shape != prev.shape:
                raise ValueError(
                    f"echo {e + 1} shape {tuple(cur.shape)} does not match "
                    f"echo {e} {tuple(prev.shape)}"
                )
            s, c = estimate_pair_shift(
                prev,
                cur,
                axis,
                ordering=ordering,
                max_shift=max_shift,
                coarse_step=coarse_step,
                fine_step=fine_step,
                weight=weight if isinstance(weight, Tensor) else None,
                device=device,
                desc=f"  xcorr echo {e + 1} vs {e}",
                disable_pbar=verb == 0,
            )
            shifts.append(s)
            corrs.append(c)
            if verb >= 1:
                edge = float(np.mean((np.abs(s - lo) < fine_step) | (np.abs(s - hi) < fine_step)))
                note = f"  [!] {edge:.0%} of volumes hit the search bound" if edge > 0.02 else ""
                print(
                    f"  echo {e + 1} vs {e}: shift {s.mean():+.4f} ± {s.std():.4f} vox  "
                    f"(r = {np.nanmean(c):.4f}){note}"
                )
        prev = cur
    n_e = len(shifts) + 1
    if n_e < 2:
        raise ValueError("inter-echo shift estimation needs at least 2 echoes")

    pairwise = np.zeros((n_t, n_e), dtype=np.float64)
    corr = np.full((n_t, n_e), np.nan, dtype=np.float64)
    for e in range(1, n_e):
        pairwise[:, e] = shifts[e - 1]
        corr[:, e] = corrs[e - 1]
    cumulative = np.cumsum(pairwise, axis=1)

    slope = intercept = None
    if tes is not None:
        slope, intercept = te_regression(cumulative, np.asarray(tes, dtype=np.float64))
        applied = slope[:, None] * np.asarray(tes, dtype=np.float64)[None, :]
        if verb >= 1:
            hz = -slope * 1.0e3
            print(
                f"  TE fit: {slope.mean() * 1e3:+.4e} vox per s of TE "
                f"→ {hz.mean():+.3f} Hz drift (range {hz.min():+.3f} … {hz.max():+.3f})"
            )
    else:
        applied = cumulative.copy()
    return ShiftEstimate(pairwise, cumulative, corr, slope, intercept, applied)


# ---------------------------------------------------------------------------
# Composition with a rigid motion-correction transform
# ---------------------------------------------------------------------------


def fold_shift_into_matrices(matrices_vox: Tensor, shifts: np.ndarray, axis: int) -> Tensor:
    """Compose a per-volume partition-axis shift into voxel-space pull matrices.

    ``matrices_vox`` is ``(nt, 4, 4)`` mapping OUTPUT voxel coordinates to input
    voxel coordinates (what the resampler samples at). The shift correction wants
    ``corrected(x) = raw(x - d)``, so sampling the corrected volume at ``c`` means
    sampling the raw volume at ``c - d·ê_axis`` — one subtraction on the
    translation entry of row ``axis``, valid whatever rotation the matrix carries.

    Folding rather than resampling twice is the point: the shift reaches the
    output through the SAME interpolation the motion correction already pays for,
    and the resulting matrix is literally the total transform, so a saved
    ``.aff12.1D`` describes exactly what happened to the data.
    """
    out = matrices_vox.clone()
    d = torch.as_tensor(np.asarray(shifts, dtype=np.float64).reshape(-1), dtype=out.dtype)
    if d.numel() != out.shape[0]:
        raise ValueError(f"got {d.numel()} shifts for {out.shape[0]} matrices")
    out[:, axis, 3] -= d
    return out


# ---------------------------------------------------------------------------
# QC tables
# ---------------------------------------------------------------------------


def save_shift_tables(
    est: ShiftEstimate, stem: str, tes: np.ndarray | None = None, verb: int = 1
) -> None:
    """Write the shift / correlation / TE-fit traces as plain ``.1D`` text.

    Four files under ``stem``: the raw cumulative cross-correlation estimate, the
    shifts actually applied, the per-pair peak correlation (the quality trace —
    watch for timepoints where it dives), and the TE fit with its discarded
    intercept and the frequency drift in Hz.
    """
    n_e = est.applied.shape[1]
    echo_cols = " ".join(f"e{i + 1}" for i in range(n_e))

    def _write(path: str, arr: np.ndarray, header: str) -> None:
        np.savetxt(path, arr, fmt="%12.6f", header=header)
        if verb >= 1:
            print(f"Saved: {path}")

    _write(
        f"{stem}_shifts_xcorr.1D",
        est.cumulative,
        f"cumulative inter-echo shift correction [voxels]; rows = volumes, cols = {echo_cols}",
    )
    _write(
        f"{stem}_shifts_applied.1D",
        est.applied,
        f"shift correction applied [voxels]; rows = volumes, cols = {echo_cols}",
    )
    _write(
        f"{stem}_corr.1D",
        est.corr[:, 1:],
        "peak correlation per consecutive echo pair; rows = volumes, cols = "
        + " ".join(f"e{i + 1}-e{i}" for i in range(1, n_e)),
    )
    if est.slope is not None and est.intercept is not None:
        te_str = " ".join(f"{t:g}" for t in np.asarray(tes).ravel()) if tes is not None else "?"
        _write(
            f"{stem}_te_fit.1D",
            np.stack([est.slope, est.intercept, est.frequency_drift_hz], axis=1),
            f"TE fit over TEs [{te_str}] ms; rows = volumes, cols = "
            "slope[vox/ms] intercept[vox, NOT applied] frequency_drift[Hz]",
        )


def shift_table_paths(stem: str, with_te_fit: bool) -> list[str]:
    """Paths :func:`save_shift_tables` writes — for batch-skip bookkeeping."""
    paths = [f"{stem}_shifts_xcorr.1D", f"{stem}_shifts_applied.1D", f"{stem}_corr.1D"]
    if with_te_fit:
        paths.append(f"{stem}_te_fit.1D")
    return paths
