"""Frequency-free cardiac starts: the heartbeat as a constant-modulus 2-D subspace.

At volume rate the heartbeat is c_n = exp(i phi(n TR)). Its phase is aliased and
smeared by heart-rate variability (it folds at 0 and Nyquist in a real signal), so
spectral-peak starts miss it, but its modulus is constant. Every voxel carries
Re(a_v c_n) with a free complex gain (sign, delay, slice-time rotation), so the pair
(cos, sin) spans a 2-D temporal subspace of the whole-brain data whose radius is flat.
We look for that subspace in whitened temporal principal components; no band, rate or
waveform enters. Wiki: Physiological noise from the data.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.glm.core import construct_polynomial_matrix

CM_DEGREE = 4  # Legendre drift removed before the search (a k/TR heartbeat is slow)
CM_RESTARTS = 64
CM_ITERS = 400
CM_STEP = 0.02
CM_KEEP = 3


def cm_components(nt: int) -> int:
    """Principal components searched. 40 overfit 120-volume runs (flat-radius pairs
    made of noise); 12 missed a heartbeat that sat beyond PC 12."""
    return int(min(20, max(4, nt // 6)))


def temporal_gram(magnitude: np.ndarray, mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """(nt, nt) float64 Gram of the voxel series (fractional change, Legendre drift
    out, unit variance), accumulated slice by slice: memory is O(nt^2 + one slice),
    never the whole masked volume in float64."""
    nt = magnitude.shape[-1]
    L = construct_polynomial_matrix(nt, CM_DEGREE, device=device, dtype=torch.float64)
    Q, _ = torch.linalg.qr(L.cpu())
    Q = Q.to(device)
    G = torch.zeros((nt, nt), dtype=torch.float64, device=device)
    for k in range(magnitude.shape[2]):
        sel = mask[:, :, k]
        if not sel.any():
            continue
        y = torch.from_numpy(np.asarray(magnitude[:, :, k][sel], dtype=np.float32))
        y = y.to(device).to(torch.float64)
        y = y - (y @ Q) @ Q.T
        y = y / (y.std(1, keepdim=True) + 1e-9)
        G += y.T @ y
    return G


def constant_modulus_traces(
    G: torch.Tensor, n_keep: int = CM_KEEP, seed: int = 0
) -> list[tuple[float, np.ndarray]]:
    """The ``n_keep`` most constant-modulus complex traces from the Gram ``G``, best
    first, as (E[r^4] / E[r^2]^2, c). The ratio is 2 for a Gaussian pair, 1 for a pure
    rotating phasor. Distinct subspaces only (smallest principal cosine < 0.9)."""
    nt = G.shape[0]
    k = cm_components(nt)
    _, evecs = torch.linalg.eigh(G.cpu())
    X = (evecs[:, -k:].T.flip(0) * math.sqrt(nt)).to(G.device)  # whitened (k, nt)
    # Starts: every pair of principal axes (the heartbeat is often nearly one, and the
    # landscape on ~100 volumes has many shallow minima random starts can all miss),
    # plus random ones.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ii, jj = torch.triu_indices(k, k, 1)
    pairs = torch.zeros((ii.numel(), k, 2), dtype=torch.float64)
    pairs[torch.arange(ii.numel()), ii, 0] = 1.0
    pairs[torch.arange(ii.numel()), jj, 1] = 1.0
    W = torch.cat([pairs, torch.randn((CM_RESTARTS, k, 2), generator=gen, dtype=torch.float64)])
    W, _ = torch.linalg.qr(W.to(G.device))
    for _ in range(CM_ITERS):
        Z = W.transpose(1, 2) @ X  # (R, 2, nt)
        r2 = (Z * Z).sum(1, keepdim=True)
        grad = (X.unsqueeze(0) * r2) @ Z.transpose(1, 2) * (4 / nt)  # d E[r^4] / dW
        sym = W.transpose(1, 2) @ grad
        grad = grad - W @ (sym + sym.transpose(1, 2)) / 2
        W, _ = torch.linalg.qr(W - CM_STEP * grad)
    Z = W.transpose(1, 2) @ X
    r2 = (Z * Z).sum(1)
    ratio = ((r2 * r2).mean(1) / r2.mean(1) ** 2).cpu().numpy()
    W, Z = W.cpu(), Z.cpu().numpy()
    out: list[tuple[float, np.ndarray]] = []
    kept: list[torch.Tensor] = []
    for i in np.argsort(ratio):
        if any(torch.linalg.svdvals(W[i].T @ V).min() >= 0.9 for V in kept):
            continue
        kept.append(W[i])
        out.append((float(ratio[i]), Z[i, 0] + 1j * Z[i, 1]))
        if len(out) == n_keep:
            break
    return out


def unwrap_around_rate(steps: np.ndarray, tr: float, window_s: float = 60.0) -> np.ndarray:
    """Per-volume phase steps lifted around a local circular-mean step, not around 0.

    A complex trace's aliased rate has no fold, but plain unwrapping slips a cycle
    whenever a step crosses +-pi -- exactly where a heartbeat near (k + 1/2)/TR sits.
    Lifting around the local mean keeps heart-rate drift continuous while it stays
    within 1/(2 TR) of that mean (15/min at TR 2 s)."""
    w = max(3, int(round(window_s / tr)) | 1)
    u = np.exp(1j * steps)
    # unwrapped: a reference sitting near pi would otherwise flip sign between
    # neighbours and carry the slip it exists to prevent
    ref = np.unwrap(np.angle(np.convolve(u, np.ones(w) / w, mode="same")))
    return ref + np.angle(u * np.exp(-1j * ref))


def excitation_phase(c: np.ndarray, times: np.ndarray, tr: float) -> np.ndarray:
    """(nt, G) phase at the excitation times from a volume-rate complex trace, on the
    alias step nearest mid-band (the alias is chosen afterwards from slice timing).
    The trace belongs to no single slice; it is placed at each volume's mean time."""
    steps = unwrap_around_rate(np.angle(c[1:] * np.conj(c[:-1])), tr)
    theta = np.angle(c[0]) + np.concatenate([[0.0], np.cumsum(steps)])
    tv = times.mean(1)
    f_alias = (theta[-1] - theta[0]) / (2 * np.pi * (tv[-1] - tv[0]))
    step = int(np.round((1.3 - f_alias) * tr))
    slope = np.gradient(theta, tv)
    phase = np.interp(times.ravel(), tv, theta).reshape(times.shape)
    lo, hi = times < tv[0], times > tv[-1]  # extrapolate, don't clamp
    phase[lo] = theta[0] + slope[0] * (times[lo] - tv[0])
    phase[hi] = theta[-1] + slope[-1] * (times[hi] - tv[-1])
    return phase + 2 * np.pi * step * times / tr


def complete_aliases(times: np.ndarray, phase: np.ndarray, tr: float, band) -> list:
    """Every (sign, step) whose rate sign * f + step / TR is in ``band``. Keys are the
    transformation, not a rounded rate, so mirrored clocks with one mean survive."""
    span = times[-1].mean() - times[0].mean()
    f = (phase[-1].mean() - phase[0].mean()) / (2 * np.pi * span)
    out = []
    for sign in (1, -1):
        first = math.ceil((band[0] - sign * f) * tr - 1e-10)
        last = math.floor((band[1] - sign * f) * tr + 1e-10)
        out += [(sign, step) for step in range(first, last + 1)]
    return out


@dataclass
class ConstantModulusDetection:
    """Result of :func:`detect`. ``tried`` rows: (label, rate/min, statistic, its
    shuffles' best, harmonic alignment, non-harmonic best)."""

    found: bool
    phase: np.ndarray | None  # (nt, G) cardiac phase at every excitation
    label: str
    rate_per_min: float
    alignment: float  # fundamental slice alignment of the kept alias
    harmonic: float
    harmonic_null: float
    statistic: float  # fundamental + harmonic alignment
    floor: float  # best shuffled statistic over every start (Bonferroni)
    tried: list = field(default_factory=list)
    tried_phase: list = field(default_factory=list)


def _alignment(Ys, phase, device) -> float:
    from fastfuncstuff.processing.physio import _voxel_fits, slice_alignment

    return slice_alignment(_voxel_fits(Ys, phase, device))


def _alias_rows(Ys, times, start, tr, device, band):
    """(phase, fundamental alignment, harmonic alignment) for every alias of ``start``."""
    rows = []
    for sign, step in complete_aliases(times, start, tr, band):
        phase = sign * start + 2 * np.pi * step * times / tr
        rows.append((phase, _alignment(Ys, phase, device), _alignment(Ys, 2 * phase, device)))
    return rows


def _derange(G: int, rng) -> np.ndarray:
    perm = rng.permutation(G)
    while np.any(perm == np.arange(G)):
        perm = rng.permutation(G)
    return perm


def _null(Ys, times, start, tr, device, band, n, rng) -> np.ndarray:
    """Best alias statistic after each of ``n`` derangements of the excitation times."""
    G = times.shape[1]
    return np.array([
        max(r[1] + r[2] for r in _alias_rows(Ys, times, start[:, _derange(G, rng)], tr,
                                             device, band) or [(None, 0.0, 0.0)])
        for _ in range(n)
    ] or [0.0])  # fmt: skip


def detect(
    magnitude: np.ndarray,
    mask: np.ndarray,
    Ys: list[torch.Tensor],
    times: np.ndarray,
    tr: float,
    device: torch.device,
    rng: np.random.Generator,
    band: tuple[float, float] = (0.6, 2.0),
    shuffles: int = 30,
    controls: tuple[float, ...] = (),
) -> ConstantModulusDetection:
    """The heartbeat from constant-modulus starts, judged by slice timing.

    Each start's alias is the one whose vessel phases line up best at the fundamental
    AND the second harmonic (slice alignment of phi plus of 2 phi). The fundamental
    alone could not tell a heartbeat's aliases apart once voxel delays scatter its
    phases (Rockland CAP: 0.37 for the true alias and two wrong ones); a pulse's
    harmonic lines up only on the true one. Found = that statistic beats every
    derangement of the excitation times, re-choosing among the same aliases (30 per
    start tried, all drawn for the kept start: Bonferroni), and the harmonic beats the non-harmonic
    multiples ``controls`` at the kept alias. The phase is not re-tracked: the
    rebuild-and-re-read refinement dragged correct starts onto a wrong rate.
    """
    nt, G = times.shape
    answers = []
    for i, (ratio, c) in enumerate(constant_modulus_traces(temporal_gram(magnitude, mask, device))):
        start = excitation_phase(c, times, tr)
        rows = _alias_rows(Ys, times, start, tr, device, band)
        if not rows:
            continue
        phase, a1, a2 = max(rows, key=lambda r: r[1] + r[2])
        null = _null(Ys, times, start, tr, device, band, shuffles, rng)
        hnull = max((_alignment(Ys, m * phase, device) for m in controls), default=0.0)
        rate = float(abs(phase[-1].mean() - phase[0].mean())
                     / (2 * np.pi * (times[-1].mean() - times[0].mean())) * 60)  # fmt: skip
        answers.append(dict(label=f"cm{i}@{ratio:.2f}", phase=phase, a1=a1, a2=a2, start=start,
                            stat=a1 + a2, null=null, hnull=hnull, rate=rate))  # fmt: skip
    if not answers:
        return ConstantModulusDetection(False, None, "none", float("nan"), 0.0, 0.0, 0.0, 0.0,
                                        float("nan"))  # fmt: skip
    pulsed = [a for a in answers if a["a2"] > a["hnull"]] or answers
    kept = max(pulsed, key=lambda a: a["stat"] - a["null"].max())
    # Bonferroni over the starts tried: more derangements of the KEPT start, not the
    # other starts' nulls -- a start that is no heartbeat (drift the same in every
    # slice) survives derangement, and its high null rejected correct picks (69 of 171 runs).
    extra = _null(Ys, times, kept["start"], tr, device, band, shuffles * (len(answers) - 1), rng)
    floor = float(np.concatenate([kept["null"], extra]).max())
    found = bool(kept["stat"] > floor and kept["a2"] > kept["hnull"])
    return ConstantModulusDetection(
        found=found, phase=kept["phase"], label=kept["label"], rate_per_min=kept["rate"],
        alignment=kept["a1"], harmonic=kept["a2"], harmonic_null=kept["hnull"],
        statistic=kept["stat"], floor=floor,
        tried=[(a["label"], a["rate"], a["stat"], float(a["null"].max()), a["a2"], a["hnull"])
               for a in answers],
        tried_phase=[a["phase"] for a in answers],
    )  # fmt: skip


def split_half_locking(magnitude, mask, split_axis, phase, times, tr, device) -> float:
    """Diagnostic: each brain half's own best constant-modulus clock, on the alias
    closest to ``phase``; their median 30 s phase locking."""
    from fastfuncstuff.processing.physio import _split_locking

    n = mask.shape[split_axis]
    coord = np.arange(n).reshape([-1 if a == split_axis else 1 for a in range(3)])
    halves = []
    for side in (coord < n // 2, coord >= n // 2):
        m = mask & side
        if m.sum() < 100:
            return float("nan")
        _, c = constant_modulus_traces(temporal_gram(magnitude, m, device), n_keep=1)[0]
        start = excitation_phase(c, times, tr)
        cands = [s * start + 2 * np.pi * k * times / tr
                 for s, k in complete_aliases(times, start, tr, (0.3, 3.0))]  # fmt: skip
        halves.append(max(cands, key=lambda q: abs(np.exp(1j * (q - phase)).mean())))
    return _split_locking(halves[0], halves[1], times, tr)
