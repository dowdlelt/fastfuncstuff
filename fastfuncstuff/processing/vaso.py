"""SS-SI VASO: split the interleaved nulled/BOLD series, align it in time, BOLD-correct it.

A slab-selective slice-inversion (SS-SI) VASO run interleaves two contrasts in
one 4-D file: a *nulled* readout at the blood-nulling time after an inversion,
then a *not-nulled* (BOLD) readout one volume-TR later, repeating every
``pair_tr``. The VASO signal itself is the nulled image divided by the BOLD
image at the same moment, which cancels the BOLD (T2*) weighting the nulled
readout also carries (Huber et al. 2014).

The pieces here are what ``ffs_moco -vaso`` composes:

1. :func:`split_streams` -- drop the non-steady-state volumes and split the
   series into the two contrasts, so each is motion-corrected against a base of
   its own contrast.
2. :func:`shift_series` -- move the BOLD stream onto the nulled stream's
   acquisition times (the nulled stream is the anchor, so it is never
   interpolated).
3. :func:`bold_correct` -- nulled / BOLD, with LN_BOCO's guards.
4. :func:`compare_motion` -- the two streams see the same head, a volume-TR
   apart, so their motion estimates must agree once that offset is removed.
   Disagreement means one stream's registration failed.

The timing is never read from the image header: a 3-D VASO header holds a shot
or volume TR, not the pair TR, so the caller supplies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.processing.slicetime import _row_chunk, shift_timeseries

NULLED = "nulled"
BOLD = "bold"
CONTRASTS = (NULLED, BOLD)

# LN_BOCO's ceiling for nulled/BOLD. True VASO ratios sit near 1; values far
# above come from a BOLD denominator near the noise floor, not from blood volume.
BOCO_MAX_RATIO = 5.0


@dataclass
class VasoSplit:
    """The two contrasts of an interleaved VASO series, after trimming."""

    nulled: Tensor  # (n_pairs, nz, ny, nx)
    bold: Tensor  # (n_pairs, nz, ny, nx); pair k is acquired after nulled[k]
    n_discarded: int  # raw volumes dropped from the start (trim + parity fix)
    n_tail: int  # raw volumes dropped from the end (an unpaired last volume)

    @property
    def n_pairs(self) -> int:
        return int(self.nulled.shape[0])


def split_streams(data: Tensor, first: str = NULLED, trim: int = 4) -> VasoSplit:
    """Trim ``trim`` raw volumes and split the remainder into nulled/BOLD pairs.

    ``first`` is the contrast of raw volume 0. Pairs always start on a nulled
    volume, because the nulled stream is the time anchor and BOLD follows it
    within a pair: if the trim leaves a BOLD volume first, one more is dropped
    (and reported in ``n_discarded``). An unpaired last volume is dropped too.
    Nothing is copied; the streams are strided views of ``data``.
    """
    if first not in CONTRASTS:
        raise ValueError(f"first must be one of {CONTRASTS}, got {first!r}")
    if trim < 0:
        raise ValueError(f"trim must be >= 0, got {trim}")
    nt = int(data.shape[0])
    # The contrast at raw index i alternates, starting from `first`.
    start = trim
    if (start % 2 == 0) != (first == NULLED):
        start += 1
    n_pairs = (nt - start) // 2
    if n_pairs < 2:
        raise ValueError(
            f"{nt} volumes leave {max(n_pairs, 0)} nulled/BOLD pair(s) after dropping "
            f"{start}; need at least 2"
        )
    stop = start + 2 * n_pairs
    return VasoSplit(
        nulled=data[start:stop:2],
        bold=data[start + 1 : stop : 2],
        n_discarded=start,
        n_tail=nt - stop,
    )


def anchor_time(n_discarded: int, first: str, pair_tr: float, bold_delay: float) -> float:
    """Seconds from raw volume 0 to raw volume ``n_discarded``, a nulled volume.

    This is when output volume 0 was acquired, relative to the first acquired
    volume -- what event onsets timed to the start of the run must be reduced by.
    """
    if first == NULLED:
        return n_discarded // 2 * pair_tr
    # Raw 0 is BOLD; the next nulled readout is the rest of the pair later.
    return (n_discarded - 1) // 2 * pair_tr + (pair_tr - bold_delay)


def volume_means(data: Tensor) -> np.ndarray:
    """Mean of each volume over a rough brain mask (above the time-mean's 60th percentile).

    A whole-FOV mean is dominated by air, which swamps the few-percent contrast
    differences this is used to read. The mask comes from the time-mean, so the
    non-steady-state volumes don't shape it.
    """
    flat = data.reshape(data.shape[0], -1).float()
    ref = flat.mean(dim=0)
    mask = ref >= torch.quantile(ref[:: max(1, ref.numel() // 200_000)], 0.6)
    return flat[:, mask].mean(dim=1).double().numpy()


def trailing_noise_volumes(means: np.ndarray, frac: float = 0.5) -> int:
    """Count trailing volumes that are noise maps, not images.

    Some VASO protocols append receiver-noise acquisitions (no excitation) to the
    run, at roughly a quarter of the image signal. Whether a run has them varies
    run to run, so a fixed tail trim cannot be right; anything at the end below
    ``frac`` of the dimmer stream's level is one. Left in, they become the last
    "pairs" and corrupt both the registration and the division.
    """
    means = np.asarray(means, dtype=float)
    level = min(float(np.median(means[0::2])), float(np.median(means[1::2])))
    n = 0
    while n < means.size and means[means.size - 1 - n] < frac * level:
        n += 1
    return n


def detect_steady_state(means: np.ndarray, tol: float = 0.01) -> int:
    """Raw volumes to drop so both interleaved streams have reached steady state.

    Each parity stream is judged against its own level (the median of its second
    half): it is steady from the first index after which every value stays
    within ``tol`` of that level, or within 4 robust SDs when the volume-to-volume
    noise is larger than ``tol``. The two streams are judged separately because
    they settle to different levels; averaging them would hide the alternation.
    """
    means = np.asarray(means, dtype=float)
    first_steady = []
    for parity in (0, 1):
        s = means[parity::2]
        if s.size < 4:
            raise ValueError(f"too few volumes ({means.size}) to judge steady state")
        tail = s[s.size // 2 :]
        level = float(np.median(tail))
        noise = 1.4826 * float(np.median(np.abs(tail - level)))
        bound = max(tol * abs(level), 4.0 * noise)
        bad = np.flatnonzero(np.abs(s - level) > bound)
        # Only an excursion before the tail counts: a late spike is motion, not
        # the approach to steady state.
        bad = bad[bad < s.size // 2]
        k = int(bad.max()) + 1 if bad.size else 0
        first_steady.append(2 * k + parity)
    return max(first_steady)


def brightness_check(nulled: Tensor, bold: Tensor) -> tuple[float, float]:
    """Brain-mean signal of the two streams over a shared mask.

    In tissue the nulled readout is darker than the BOLD one -- that is the
    premise of the division -- so a nulled stream that comes out brighter means
    the parity is wrong and the VASO will be BOLD/nulled, an inverted contrast.
    """
    n = nulled.float().mean(dim=0).reshape(-1)
    b = bold.float().mean(dim=0).reshape(-1)
    both = (n + b) / 2
    mask = both >= torch.quantile(both[:: max(1, both.numel() // 200_000)], 0.6)
    return float(n[mask].mean()), float(b[mask].mean())


def shift_series(
    data: Tensor,
    shift: float,
    method: str = "heptic",
    device: torch.device | None = None,
    verb: int = 1,
) -> Tensor:
    """Resample a 4-D series in time so that ``out[k] = data(k - shift)``.

    ``shift`` is in samples. A BOLD stream acquired ``delay`` seconds after the
    nulled stream it is paired with goes onto the nulled times with
    ``shift = delay / pair_tr``.

    Edges: a centred kernel near either end reaches past the data, and any
    padding (zeros, or the replicated edge 3dUpsample's clamped indices amount to)
    puts a kink in what it interpolates -- a few percent of error in the outer
    three samples of a heptic. Those outputs are instead computed from a
    one-sided Lagrange stencil of the same order that stays inside the data,
    which is as exact as the interior. An output that needs a value from outside
    the acquired span -- the first one for a positive shift -- is the nearest
    acquired volume, as in the LayNii pipeline; it is an extrapolation either
    way, and a copy cannot overshoot.

    The default heptic is the 7th-order Lagrange interpolant 3dUpsample uses,
    which the published VASO pipelines run. Returns a CPU float32 tensor.
    """
    nt = int(data.shape[0])
    out = torch.empty(data.shape, dtype=torch.float32)
    if abs(shift) < 1e-9:
        out.copy_(data)
        return out
    device = device or torch.device("cpu")
    pad = 12 + math.ceil(abs(shift))
    flat_in = data.reshape(nt, -1)
    flat_out = out.reshape(nt, -1)
    n_vox = flat_in.shape[1]
    chunk = _row_chunk(n_vox, nt + 2 * pad, device)
    for v0 in tqdm(
        range(0, n_vox, chunk),
        desc="  temporal shift",
        unit="chunk",
        leave=True,
        disable=verb < 1 or n_vox <= chunk,
    ):
        rows = flat_in[:, v0 : v0 + chunk].T.to(device).float()
        padded = torch.cat([rows[:, :1].expand(-1, pad), rows, rows[:, -1:].expand(-1, pad)], dim=1)
        shifted = shift_timeseries(padded, shift, method=method)[:, pad : pad + nt]
        flat_out[:, v0 : v0 + chunk] = shifted.T.float().cpu()

    order = _EDGE_ORDER.get(method.lower(), 7)
    reach = _KERNEL_REACH.get(method.lower(), 4)
    src = np.arange(nt) - shift
    for k, p in enumerate(src):
        if p < 0:
            out[k] = data[0]
        elif p > nt - 1:
            out[k] = data[nt - 1]
        elif math.floor(p) - reach < 0 or math.floor(p) + reach > nt - 1:
            nodes, w = _one_sided_stencil(p, nt, order)
            acc = torch.zeros(data.shape[1:], dtype=torch.float64)
            for j, wj in zip(nodes, w, strict=True):
                acc += wj * data[j].double()
            out[k] = acc.float()
    return out


# Polynomial order of the one-sided edge stencil, and how far (in samples) each
# kernel reaches from floor(p) -- outputs closer than that to an end are redone.
# Non-polynomial kernels (sinc, Fourier) get heptic edges.
_EDGE_ORDER = {"linear": 1, "cubic": 3, "quintic": 5, "heptic": 7}
_KERNEL_REACH = {
    "linear": 1,
    "cubic": 2,
    "quintic": 3,
    "heptic": 4,
    "wsinc5": 5,
    "wsinc9": 9,
    "fourier": 16,
}


def _one_sided_stencil(p: float, nt: int, order: int) -> tuple[list[int], list[float]]:
    """Lagrange nodes and weights for position ``p``, shifted to lie inside [0, nt-1]."""
    n_pts = min(order + 1, nt)
    s0 = min(max(math.floor(p) - (n_pts - 1) // 2, 0), nt - n_pts)
    nodes = list(range(s0, s0 + n_pts))
    weights = []
    for j in nodes:
        w = 1.0
        for m in nodes:
            if m != j:
                w *= (p - m) / (j - m)
        weights.append(w)
    return nodes, weights


def extrapolated(nt: int, shift: float) -> np.ndarray:
    """Mask of outputs that :func:`shift_series` fills by copying an edge volume."""
    src = np.arange(nt) - shift
    return (src < 0) | (src > nt - 1)


def bold_correct(nulled: Tensor, bold: Tensor, max_ratio: float = BOCO_MAX_RATIO) -> Tensor:
    """VASO = nulled / BOLD, as LN_BOCO computes it.

    Where either input is <= 0 (outside the brain, or a resampling edge) the
    result is 0; the ratio is clipped to ``[0, max_ratio]``; non-finite values
    become 0. The two series must already share acquisition times.
    """
    if nulled.shape != bold.shape:
        raise ValueError(f"nulled {tuple(nulled.shape)} and BOLD {tuple(bold.shape)} differ")
    n = nulled.float()
    b = bold.float()
    valid = (n > 0) & (b > 0)
    ratio = torch.where(valid, n / torch.where(valid, b, torch.ones_like(b)), 0.0)
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return ratio.clamp_(0.0, max_ratio)


MOTION_LABELS = ("roll", "pitch", "yaw", "dS", "dL", "dP")


def framewise_displacement(motion: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Power FD from AFNI-ordered motion (roll pitch yaw in degrees, then mm)."""
    m = np.asarray(motion, dtype=float)
    d = np.abs(np.diff(m, axis=0))
    fd = d[:, 3:].sum(axis=1) + radius * np.deg2rad(d[:, :3]).sum(axis=1)
    return np.concatenate([[0.0], fd])


@dataclass
class MotionComparison:
    """Agreement between the nulled and BOLD motion estimates (AFNI order)."""

    r_raw: np.ndarray  # (6,) correlation per parameter, BOLD at its own times
    r_aligned: np.ndarray  # (6,) correlation after shifting BOLD onto nulled times
    rms_diff: np.ndarray  # (6,) RMS of (nulled - aligned BOLD), offset removed
    # (the aligned statistics skip any copied edge volume)
    max_diff: np.ndarray  # (6,) max |nulled - aligned BOLD|, offset removed
    offset: np.ndarray  # (6,) mean(nulled - aligned BOLD): pose between the two bases
    range_nulled: np.ndarray  # (6,) peak-to-peak of the nulled estimate
    fd_nulled: np.ndarray  # (n,)
    fd_bold: np.ndarray  # (n,)
    bold_aligned: np.ndarray  # (n, 6) BOLD motion on the nulled times


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)


def compare_motion(
    motion_nulled: np.ndarray, motion_bold: np.ndarray, shift: float, method: str = "heptic"
) -> MotionComparison:
    """Compare the two streams' motion after putting BOLD's on the nulled times.

    Each stream is registered to its own base, so the two traces differ by a
    constant -- the head's pose change between the two base volumes -- which is
    reported as ``offset`` and removed before the differences are measured.
    The BOLD trace is moved in time with the same kernel as the BOLD data, so
    ``r_aligned`` against ``r_raw`` shows how much of the disagreement was only
    the acquisition offset.
    """
    mn = np.asarray(motion_nulled, dtype=float)
    mb = np.asarray(motion_bold, dtype=float)
    if mn.shape != mb.shape:
        raise ValueError(f"motion shapes differ: {mn.shape} vs {mb.shape}")
    aligned = (
        shift_series(torch.from_numpy(mb).float()[:, :, None, None], shift, method, verb=0)
        .squeeze(-1)
        .squeeze(-1)
        .double()
        .numpy()
    )
    # The copied edge volume is not an estimate of the motion at that time.
    keep = ~extrapolated(mn.shape[0], shift)
    diff = (mn - aligned)[keep]
    offset = diff.mean(axis=0)
    resid = diff - offset
    return MotionComparison(
        r_raw=_corr(mn, mb),
        r_aligned=_corr(mn[keep], aligned[keep]),
        rms_diff=np.sqrt((resid**2).mean(axis=0)),
        max_diff=np.abs(resid).max(axis=0),
        offset=offset,
        range_nulled=np.ptp(mn, axis=0),
        fd_nulled=framewise_displacement(mn),
        fd_bold=framewise_displacement(aligned),
        bold_aligned=aligned,
    )


def format_motion_comparison(cmp: MotionComparison) -> str:
    """Tab-separated table: one row per parameter, then FD summary rows."""
    lines = ["param\tr_raw\tr_aligned\trms_diff\tmax_diff\toffset\trange_nulled"]
    for i, name in enumerate(MOTION_LABELS):
        lines.append(
            f"{name}\t{cmp.r_raw[i]:.3f}\t{cmp.r_aligned[i]:.3f}\t{cmp.rms_diff[i]:.4f}"
            f"\t{cmp.max_diff[i]:.4f}\t{cmp.offset[i]:.4f}\t{cmp.range_nulled[i]:.4f}"
        )
    fd_r = _corr(cmp.fd_nulled[:, None], cmp.fd_bold[:, None])[0]
    lines.append(
        f"FD\t\t{fd_r:.3f}\t\t\tmean_nulled={cmp.fd_nulled.mean():.4f}"
        f"\tmean_bold={cmp.fd_bold.mean():.4f}"
    )
    return "\n".join(lines)


@dataclass
class VasoResult:
    """Everything ``ffs_moco -vaso`` writes, on the nulled stream's time grid."""

    split: VasoSplit
    nulled: Tensor  # (n_pairs, ...) motion-corrected nulled stream (the time anchor)
    bold: Tensor  # (n_pairs, ...) motion-corrected BOLD, shifted onto the nulled times
    bold_unshifted: Tensor  # (n_pairs, ...) motion-corrected BOLD at its own times
    vaso: Tensor  # (n_pairs, ...) nulled / BOLD
    motion_nulled: np.ndarray  # (n_pairs, 6) AFNI order
    motion_bold: np.ndarray  # (n_pairs, 6) AFNI order, BOLD's own times
    comparison: MotionComparison
    shift: float  # samples: bold_delay / pair_tr
    brightness: tuple[float, float]  # brain mean (nulled, BOLD) before correction


def run_vaso(
    data: Tensor,
    config,
    pair_tr: float,
    bold_delay: float,
    first: str = NULLED,
    trim: int = 4,
    tinterp: str = "heptic",
    header_info: dict | None = None,
    device: torch.device | None = None,
) -> VasoResult:
    """Split, motion-correct each contrast, align BOLD to nulled in time, BOLD-correct.

    ``config`` is the :class:`~fastfuncstuff.processing.ffs_moco.MocoConfig` both
    streams are registered with; ``config.base_index`` indexes each stream (its
    own pair index), so each contrast is aligned to a base of the same contrast.
    The temporal shift comes after the spatial correction: interpolating in time
    across volumes that are not yet in register would mix tissue between voxels.
    """
    from fastfuncstuff.processing.ffs_moco import moco, to_afni_motion

    if not 0 < bold_delay < pair_tr:
        raise ValueError(f"bold_delay {bold_delay:g}s must lie inside the pair TR {pair_tr:g}s")
    split = split_streams(data, first=first, trim=trim)
    if not 0 <= config.base_index < split.n_pairs:
        raise ValueError(f"base index {config.base_index} outside 0..{split.n_pairs - 1}")
    brightness = brightness_check(split.nulled, split.bold)

    verb = config.verb
    results = {}
    for name, stream in ((NULLED, split.nulled), (BOLD, split.bold)):
        if verb >= 1:
            print(f"\n  -- {name} stream: {split.n_pairs} volumes --")
        results[name] = moco(stream.contiguous(), config, header_info=header_info)

    shift = bold_delay / pair_tr
    nulled = results[NULLED].aligned.float().cpu()
    bold_unshifted = results[BOLD].aligned.float().cpu()
    bold = shift_series(bold_unshifted, shift, method=tinterp, device=device, verb=verb)
    motion_nulled = to_afni_motion(results[NULLED].params)
    motion_bold = to_afni_motion(results[BOLD].params)
    return VasoResult(
        split=split,
        nulled=nulled,
        bold=bold,
        bold_unshifted=bold_unshifted,
        vaso=bold_correct(nulled, bold),
        motion_nulled=motion_nulled,
        motion_bold=motion_bold,
        comparison=compare_motion(motion_nulled, motion_bold, shift, method=tinterp),
        shift=shift,
        brightness=brightness,
    )
