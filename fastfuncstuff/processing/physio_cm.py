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

from fastfuncstuff import policy_linalg as plinalg
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


def temporal_grams(
    magnitude: np.ndarray, masks: list[np.ndarray], device: torch.device
) -> list[torch.Tensor]:
    """(nt, nt) float64 Gram of the voxel series (fractional change, Legendre drift
    out, unit variance) inside each mask, in one pass, slice by slice: memory is
    O(nt^2 + one slice), never the whole masked volume in float64. Each voxel adds its
    own outer product, so the Gram of a union is the sum of its parts' Grams."""
    nt = magnitude.shape[-1]
    L = construct_polynomial_matrix(nt, CM_DEGREE, device=device, dtype=torch.float64)
    Q, _ = plinalg.qr(L.cpu())
    Q = Q.to(device)
    grams = [torch.zeros((nt, nt), dtype=torch.float64, device=device) for _ in masks]
    union = np.logical_or.reduce(masks)
    for k in range(magnitude.shape[2]):
        sel = union[:, :, k]
        if not sel.any():
            continue
        y = torch.from_numpy(np.asarray(magnitude[:, :, k][sel], dtype=np.float32))
        y = y.to(device).to(torch.float64)
        y = y - (y @ Q) @ Q.T
        y = y / (y.std(1, keepdim=True) + 1e-9)
        for G, m in zip(grams, masks, strict=True):
            rows = torch.from_numpy(m[:, :, k][sel]).to(device)
            if bool(rows.all()):
                G += y.T @ y
            elif bool(rows.any()):
                G += y[rows].T @ y[rows]
    return grams


def temporal_gram(magnitude: np.ndarray, mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return temporal_grams(magnitude, [mask], device)[0]


def split_masks(mask: np.ndarray, split_axis: int) -> list[np.ndarray]:
    """The brain mask's two halves along ``split_axis`` (a partition of ``mask``)."""
    n = mask.shape[split_axis]
    coord = np.arange(n).reshape([-1 if a == split_axis else 1 for a in range(3)])
    return [mask & (coord < n // 2), mask & (coord >= n // 2)]


def constant_modulus_traces(
    G: torch.Tensor, n_keep: int = CM_KEEP, seed: int = 0
) -> list[tuple[float, np.ndarray]]:
    """The ``n_keep`` most constant-modulus complex traces from the Gram ``G``, best
    first, as (E[r^4] / E[r^2]^2, c). The ratio is 2 for a Gaussian pair, 1 for a pure
    rotating phasor. Distinct subspaces only (smallest principal cosine < 0.9)."""
    nt = G.shape[0]
    k = cm_components(nt)
    _, evecs = plinalg.eigh(G)
    X = evecs[:, -k:].T.flip(0) * math.sqrt(nt)  # whitened (k, nt)
    # Starts: every pair of principal axes (the heartbeat is often nearly one, and the
    # landscape on ~100 volumes has many shallow minima random starts can all miss),
    # plus random ones.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ii, jj = torch.triu_indices(k, k, 1)
    pairs = torch.zeros((ii.numel(), k, 2), dtype=torch.float64)
    pairs[torch.arange(ii.numel()), ii, 0] = 1.0
    pairs[torch.arange(ii.numel()), jj, 1] = 1.0
    W = torch.cat([pairs, torch.randn((CM_RESTARTS, k, 2), generator=gen, dtype=torch.float64)])
    W, _ = plinalg.qr(W.to(G.device))
    for _ in range(CM_ITERS):
        Z = W.transpose(1, 2) @ X  # (R, 2, nt)
        r2 = (Z * Z).sum(1, keepdim=True)
        grad = X @ (Z * r2).transpose(1, 2) * (4 / nt)  # d E[r^4] / dW, (R, k, 2)
        sym = W.transpose(1, 2) @ grad
        grad = grad - W @ (sym + sym.transpose(1, 2)) / 2
        W = _polar(W - CM_STEP * grad)
    Z = W.transpose(1, 2) @ X
    r2 = (Z * Z).sum(1)
    ratio = ((r2 * r2).mean(1) / r2.mean(1) ** 2).cpu().numpy()
    W, Z = W.cpu(), Z.cpu().numpy()
    out: list[tuple[float, np.ndarray]] = []
    kept: list[torch.Tensor] = []
    for i in np.argsort(ratio):
        if any(plinalg.svdvals(W[i].T @ V).min() >= 0.9 for V in kept):
            continue
        kept.append(W[i])
        out.append((float(ratio[i]), Z[i, 0] + 1j * Z[i, 1]))
        if len(out) == n_keep:
            break
    return out


def _polar(M: torch.Tensor) -> torch.Tensor:
    """Nearest orthonormal (k, 2) frames, M (M^T M)^(-1/2), with the 2 x 2 inverse
    square root in closed form: the retraction without a batched QR, whose small-matrix
    solver launches dominated the search on the GPU (8 ms of every 9 ms step)."""
    A = M.transpose(1, 2) @ M
    a, b, d = A[:, 0, 0], A[:, 0, 1], A[:, 1, 1]
    s = (a * d - b * b).clamp_min(1e-300).sqrt()
    t = (a + d + 2 * s).sqrt()
    # sqrt(A) = (A + s I) / t, so its inverse is t (A + s I)^-1
    inv = torch.stack([torch.stack([d + s, -b], -1), torch.stack([-b, a + s], -1)], -2)
    inv = inv * (t / ((a + s) * (d + s) - b * b))[:, None, None]
    return M @ inv


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


def _alignments(Ys, phases, device) -> np.ndarray:
    from fastfuncstuff.processing.physio import slice_alignments

    return slice_alignments(Ys, phases, device)


def _alias_phases(times, start, tr, band) -> list[np.ndarray]:
    return [sign * start + 2 * np.pi * step * times / tr
            for sign, step in complete_aliases(times, start, tr, band)]  # fmt: skip


def _alias_rows(Ys, times, start, tr, device, band):
    """(phase, fundamental alignment, harmonic alignment) for every alias of ``start``."""
    phases = _alias_phases(times, start, tr, band)
    a = _alignments(Ys, phases + [2 * q for q in phases], device)
    return list(zip(phases, a[: len(phases)], a[len(phases) :], strict=True))


def _derange(G: int, rng) -> np.ndarray:
    perm = rng.permutation(G)
    while np.any(perm == np.arange(G)):
        perm = rng.permutation(G)
    return perm


def _null(Ys, times, start, tr, device, band, n, rng) -> np.ndarray:
    """Best alias statistic after each of ``n`` derangements of the excitation times,
    all scored in one batched pass. A derangement keeps the run's mean rate, so every
    deranged clock has the same alias set as ``start``."""
    G = times.shape[1]
    A = len(complete_aliases(times, start, tr, band))
    if n == 0 or A == 0:
        return np.zeros(1)
    phases = []
    for _ in range(n):
        q = _alias_phases(times, start[:, _derange(G, rng)], tr, band)
        phases += q + [2 * x for x in q]
    a = _alignments(Ys, phases, device).reshape(n, 2, A)
    return (a[:, 0] + a[:, 1]).max(1)


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
    gram: torch.Tensor | None = None,
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
    if gram is None:
        gram = temporal_gram(magnitude, mask, device)
    for i, (ratio, c) in enumerate(constant_modulus_traces(gram)):
        start = excitation_phase(c, times, tr)
        rows = _alias_rows(Ys, times, start, tr, device, band)
        if not rows:
            continue
        phase, a1, a2 = max(rows, key=lambda r: r[1] + r[2])
        null = _null(Ys, times, start, tr, device, band, shuffles, rng)
        hnull = float(_alignments(Ys, [m * phase for m in controls], device).max(initial=0.0))
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


def split_half_locking(magnitude, mask, split_axis, phase, times, tr, device, grams=None) -> float:
    """Diagnostic: each brain half's own best constant-modulus clock, on the alias
    closest to ``phase``; their median 30 s phase locking. ``grams``: the halves'
    Grams if already built (:func:`temporal_grams` on :func:`split_masks`)."""
    from fastfuncstuff.processing.physio import _split_locking

    masks = split_masks(mask, split_axis)
    if any(m.sum() < 100 for m in masks):
        return float("nan")
    if grams is None:
        grams = temporal_grams(magnitude, masks, device)
    halves = []
    for G in grams:
        _, c = constant_modulus_traces(G, n_keep=1)[0]
        start = excitation_phase(c, times, tr)
        cands = [s * start + 2 * np.pi * k * times / tr
                 for s, k in complete_aliases(times, start, tr, (0.3, 3.0))]  # fmt: skip
        halves.append(max(cands, key=lambda q: abs(np.exp(1j * (q - phase)).mean())))
    return _split_locking(halves[0], halves[1], times, tr)
