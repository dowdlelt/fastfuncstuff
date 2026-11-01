"""Reference-anchored phase unwrapping of a whole time series on the GPU.

MEDIC (warpkit) and ROMEO's ``-t epi`` mode spend nearly all their time unwrapping
every frame spatially with ROMEO's region growing, which is sequential and cannot
use a GPU. A frame differs from its neighbours in time by a small field change,
so this module does the spatial unwrap ONCE and anchors every frame to it:

1. **Reference.** The temporal-mean complex image (high SNR, typical anatomy),
   unwrapped by warpkit's own per-frame routine (MCPC-3D-S phase offset + C++ ROMEO
   + global-mode correction) for multi-echo, or ROMEO 3-D for a single echo. Where
   the field is steep, motion decoheres the mean (later echoes drop out), and its
   MCPC-3D-S offset and 2π counts go wrong; a few ROMEO'd single frames
   (``anchor_frames``) supply the offset (their circular mean) and outvote the
   reference's shortest-echo count.
2. **Change, per voxel, all frames at once.** Each frame's wrapped difference from
   the reference. The shortest echo's change is wrap-free up to ``1000/(2 TE1)`` Hz.
   Where a voxel's change really leaves that range (motion through a steep field,
   drift; a time-varying voxel), the anchored value flips by 2π between frames; the
   change is then followed along time instead (``temporal_path``), but only in
   voxels whose frame-to-frame steps are small, so noise cannot random-walk. The
   mean cannot speak for such a voxel, so its level is set by the anchor frames at
   their own time points. The frame-wide field change (breathing, a breath-hold,
   drift) is fitted per frame as a low-order polynomial on those smooth voxels, and
   every other voxel wraps only its residual about it: a breath-hold moved the
   whole head past half a period at TE 36 ms, wrapping a quarter of the voxels too
   noisy to follow.
3. **Echo ladder.** Each later echo's 2π count from the weighted through-origin fit
   of the earlier echoes' change (warpkit's temporal-consistency step, per voxel),
   then once more on the total phase: rounding the reference and the change
   separately lets their residuals add past π. From the third echo on, the count is
   then checked against a line WITH intercept, which no offset error can bias.
4. **Field.** Magnitude²-weighted fit through the origin (warpkit's convention).

Everything per voxel, so the series is streamed in slabs along its slowest spatial
axis: each slab holds every frame, which is what the temporal path needs.

:func:`svd_filter_fields` is warpkit's border-aware + global SVD field denoising,
ported to torch (warpkit is MIT; Van et al. 2026, doi:10.1162/IMAG.a.1262).
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.utils import linalg_device

from ..memory import get_available_memory

TWO_PI = 2.0 * np.pi

# A good voxel's shortest-echo change moves far less than π/2 between frames (noise
# ~1/SNR rad); in a noise-dominated voxel ~half of the steps exceed it.
_ROUGH_STEP = np.pi / 2


@dataclass
class UnwrapResult:
    field_hz: np.ndarray  # (x, y, z, t), unfiltered, 0 outside the mask
    mask: np.ndarray  # (x, y, z) int8: 0 outside, 2 core, 1 dilated border (warpkit's)
    phase_offset: np.ndarray  # (x, y, z) radians removed before unwrapping (0 for 1 echo)
    temporal_path: np.ndarray  # (x, y, z) bool, voxels whose change was followed in time
    unwrapped: np.ndarray | None = None  # (x, y, z, e, t) radians, offset removed


def raw_phase_to_radians(phase: np.ndarray, frame: int = 0) -> np.ndarray:
    """Map raw scanner phase onto [-π, π) the way warpkit does (min/max mode over echoes).

    ``phase`` is ``(x, y, z, e, t)``; the range is read from one frame per echo and
    the most common min and max win, so a single odd echo cannot set the scale.
    """
    from scipy.stats import mode

    lo = mode([float(phase[..., e, frame].min()) for e in range(phase.shape[3])]).mode
    hi = mode([float(phase[..., e, frame].max()) for e in range(phase.shape[3])]).mode
    out = (phase - lo) * np.float32(TWO_PI / (hi - lo)) - np.float32(np.pi)
    return out.astype(np.float32, copy=False)


def _wrap(x: Tensor) -> Tensor:
    return torch.remainder(x + np.pi, TWO_PI) - np.pi


def _slab_axis(a: np.ndarray) -> int:
    """The spatial axis that is slowest in memory: slabs along it are (near-)contiguous
    whether the series came C-ordered from np.stack or as a view of (t, z, y, x)."""
    return int(np.argmax([abs(st) for st in a.strides[:3]]))


def _slab_width(n_plane: int, ne: int, nt: int, device: torch.device) -> int:
    """Planes per slab: phase + magnitude + ~6 working copies of (e, t) per voxel."""
    per_plane = max(1, n_plane * ne * nt * 4 * 8)
    return max(1, int(get_available_memory(device) // per_plane))


def signal_frames(mag: np.ndarray) -> np.ndarray:
    """Frames carrying signal: noise volumes (scanner off, a few % of the median
    intensity) are often appended to a run and must not set the reference."""
    level = mag[..., 0, :].mean(axis=(0, 1, 2))
    return level > 0.3 * np.median(level)


def _complex_mean(
    pv: np.ndarray | Tensor, mv: np.ndarray | Tensor, keep: np.ndarray, device: torch.device
) -> np.ndarray:
    """Temporal-mean complex image over the ``keep`` frames, in the slab views' layout."""
    out = np.empty(tuple(pv.shape[:4]), np.complex64)
    sx = _slab_width(int(np.prod(pv.shape[1:3])), pv.shape[3], pv.shape[4], device)
    k = torch.from_numpy(keep).to(device)
    for a in range(0, pv.shape[0], sx):
        p = _to_device(pv[a : a + sx], device)
        m = _to_device(mv[a : a + sx], device)
        out[a : a + sx] = torch.polar(m, p)[..., k].mean(-1).cpu().numpy()
    return out


def _to_device(slab: np.ndarray | Tensor, device: torch.device) -> Tensor:
    if isinstance(slab, Tensor):
        return slab
    return torch.from_numpy(np.ascontiguousarray(slab)).to(device)


def _single_echo_mask(mag_mean: np.ndarray, dilation: int, device: torch.device) -> np.ndarray:
    """Core automask (2) + ``dilation`` voxels of border (1), warpkit's mask convention."""
    import torch.nn.functional as F

    from .mask import automask

    vol = torch.from_numpy(np.ascontiguousarray(mag_mean.transpose(2, 1, 0))).float().to(device)
    core = automask(vol, dilate_extra=0, device=device).bool()
    x = core.float()[None, None]
    for _ in range(max(0, dilation)):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    m = core.to(torch.int8) + (x[0, 0] > 0.5).to(torch.int8)
    return m.cpu().numpy().transpose(2, 1, 0).copy()


def _romeo_volume(
    phase: np.ndarray, mag: np.ndarray, tes: np.ndarray, se_mask: np.ndarray | None, dilation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROMEO one ``(x, y, z, e)`` volume: (unwrapped, mask 0/1/2, offset), no ladder.

    Multi-echo: warpkit's per-frame routine (MCPC-3D-S offset + ROMEO + global mode).
    Single echo: ROMEO 3-D in ``se_mask``; the offset cannot be separated, so it stays in.
    """
    if phase.shape[3] >= 2:
        from warpkit.unwrap import unwrap_phase

        work = phase.astype(np.float32).copy()  # warpkit removes the offset in place
        u, mask = unwrap_phase(
            work,
            mag.astype(np.float32),
            tes.astype(np.float32),
            np.ones(mag.shape[:3], bool),
            automask=True,
            automask_dilation=dilation,
        )
        offset = np.angle(np.exp(1j * (phase[..., 0] - u[..., 0]))).astype(np.float32)
        offset[mask == 0] = 0
        return u.astype(np.float32), mask.astype(np.int8), offset
    from warpkit.warpkit_cpp import romeo_unwrap3d

    assert se_mask is not None
    u = romeo_unwrap3d(
        phase=np.ascontiguousarray(phase[..., 0], np.float32),
        weights="romeo",
        mag=np.ascontiguousarray(mag[..., 0], np.float32),
        mask=se_mask > 0,
        correct_global=True,
    )
    u = np.asarray(u, np.float32)[..., None]
    u[se_mask == 0] = 0
    return u, se_mask, np.zeros(se_mask.shape, np.float32)


def romeo_voxel_quality(
    mags: Sequence[np.ndarray], phases_rad: Sequence[np.ndarray], tes: Sequence[float]
) -> np.ndarray:
    """ROMEO's per-voxel quality, ``(x, y, z)`` in [0, 1], of a multi-echo series.

    Scored on the temporal means (magnitude, and the unwrapped phase -- the score reads
    only wrapped neighbour differences, so it is re-wrapped first): phase coherence with
    the neighbours, agreement of the phase gradient across echoes, and magnitude
    coherence. High where the phase is spatially smooth and echo-consistent; low where
    dephasing, partial volume or a steep field leave it incoherent -- exactly where a
    phase gradient cannot be trusted to say how the phase moves when the tissue does.
    """
    from warpkit.warpkit_cpp import romeo_voxelquality

    if len(mags) < 2:
        raise ValueError("ROMEO's voxel quality needs 2+ echoes.")
    mag = np.stack([np.asarray(m, np.float32).mean(axis=3) for m in mags], axis=3)
    ph = np.stack([np.asarray(p, np.float32).mean(axis=3) for p in phases_rad], axis=3)
    ph = np.angle(np.exp(1j * ph)).astype(np.float32)
    q = romeo_voxelquality(
        np.ascontiguousarray(ph), np.asarray(tes, np.float32), np.ascontiguousarray(mag)
    )
    return np.clip(np.nan_to_num(np.asarray(q, np.float32)), 0.0, 1.0)


def unwrap_reference(
    ref_phase: np.ndarray,
    ref_mag: np.ndarray,
    tes: np.ndarray,
    *,
    anchors: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    automask_dilation: int = 3,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatially unwrap the reference ``(x, y, z, e)`` volume: (unwrapped, mask, offset).

    ROMEO on the reference alone can settle on a different 2π count than ROMEO on
    the individual frames where the field is steep: motion there decoheres the
    temporal mean. Each of ``anchors`` (single frames' phase, magnitude) is ROMEO'd
    too, carried onto the reference by its wrapped difference, and where most of
    them put the shortest echo in another count, the reference follows the majority.
    Later echoes then come from a per-voxel ladder (warpkit's global-mode correction
    is one count per echo for the whole volume).

    Also returns each anchor's shortest-echo unwrapped phase on the reference's
    offset, ``(x, y, z, n_anchors)``, NaN outside that frame's mask (or for an anchor
    ROMEO failed on): the per-time check for voxels whose field moves too far for
    the reference to speak for them.
    """
    ne = ref_phase.shape[3]
    se_mask = None if ne >= 2 else _single_echo_mask(ref_mag[..., 0], automask_dilation, device)
    from concurrent.futures import ThreadPoolExecutor

    def romeo_or_none(pm):
        try:
            return _romeo_volume(pm[0], pm[1], tes, se_mask, automask_dilation)
        except (IndexError, ValueError) as exc:  # e.g. warpkit finds no head in it
            warnings.warn(f"anchor frame skipped, ROMEO failed: {exc}", stacklevel=2)
            return None

    # One pool for the reference and its anchors: ROMEO releases the GIL around its
    # C++ kernel, and at 1.25 mm whole-brain one volume takes ~20 s, so running the
    # reference first and the anchors after would double the wall time.
    with ThreadPoolExecutor(min(len(anchors) + 1, os.cpu_count() or 1)) as ex:
        ref_job = ex.submit(_romeo_volume, ref_phase, ref_mag, tes, se_mask, automask_dilation)
        done = list(ex.map(romeo_or_none, anchors))
        u, mask, offset = ref_job.result()

    # The vote is full-volume elementwise work over every anchor: on the device it
    # is ~free, in numpy it cost more than all the ROMEO runs (single echo, 1.25 mm).
    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    ut, off, ph_t = dev(u), dev(offset), dev(ref_phase)
    inside_t = dev(mask > 0)
    te_t = dev(tes.astype(np.float32))
    pairs = [(a, d) for a, d in zip(anchors, done, strict=True) if d is not None]
    if ne >= 2 and pairs:
        # MCPC-3D-S on the decohered mean is biased where later echoes drop out;
        # the frames' own offsets (circular mean) are what warpkit removes.
        zc = torch.zeros_like(off)
        zs = torch.zeros_like(off)
        for _a, (_u, ma, oa) in pairs:
            o, m = dev(oa), dev(ma > 0)
            zc += torch.cos(o) * m
            zs += torch.sin(o) * m
        off = torch.where(zc * zc + zs * zs > 0, torch.atan2(zs, zc), off)
        # keep each echo's 2pi count, re-base its fraction on the new offset
        ut = ut + _wrap((ph_t - off[..., None]) - ut)
    if pairs:
        ref0 = ph_t[..., 0] - off
        counts = torch.stack(
            [
                torch.where(
                    dev(ma > 0),
                    torch.round(
                        (dev(ua[..., 0]) + _wrap(ref0 - (dev(p[..., 0]) - dev(oa))) - ut[..., 0])
                        / TWO_PI
                    ),
                    torch.nan,
                )
                for (p, _m), (ua, ma, oa) in pairs
            ],
            -1,
        )
        vote = torch.nanmedian(counts, -1).values
        agree = (counts == vote[..., None]).sum(-1) > 0.5 * len(pairs)
        move = torch.where(inside_t & agree & torch.isfinite(vote), vote, 0.0)
        # an echo-1 count is 1000/TE1 Hz of field: carry it to every echo, then re-ladder
        ut = ut + (TWO_PI * move)[..., None] * (te_t / te_t[0])
        del counts
    w = dev(ref_mag.astype(np.float32)) ** 2
    for e in range(1, ne):
        num = (w[..., :e] * ut[..., :e] * te_t[:e]).sum(-1)
        den = (w[..., :e] * te_t[:e] ** 2).sum(-1).clamp_min(1e-12)
        ut[..., e] += TWO_PI * torch.round((num / den * te_t[e] - ut[..., e]) / TWO_PI)
    ut = ut * inside_t[..., None]
    at_anchor = torch.full((*mask.shape, len(anchors)), torch.nan, device=device)
    for i, d in enumerate(done):
        if d is not None:
            ua, ma, oa = d
            at_anchor[..., i] = torch.where(
                dev(ma > 0), dev(ua[..., 0]) + _wrap(dev(oa) - off), torch.nan
            )
    return (
        ut.float().cpu().numpy(),
        mask,
        off.float().cpu().numpy(),
        at_anchor.cpu().numpy(),
    )


def _gather(slab: np.ndarray | Tensor, shape: tuple[int, ...], idx: Tensor) -> Tensor:
    """A slab, flattened to voxels, on ``idx``'s device and restricted to ``idx``."""
    return _to_device(slab, idx.device).reshape(shape)[idx]


def _offset_free_step(u: Tensor, w: Tensor, te: Tensor) -> Tensor:
    """-1, 0 or +1 for the last echo of ``u`` (V, e, t): whichever 2π step leaves the
    smallest weighted residual about a line WITH intercept through all its echoes.

    The through-origin prediction inherits any error in the removed phase offset,
    scaled by TE_e/TE_1 - 1 (4x for a typical third echo), which is enough to flip a
    low-signal voxel's count in the odd frame; a free intercept is blind to it.
    """
    t = te[None, :, None]
    sw, st, stt = w.sum(1), (w * t).sum(1), (w * t * t).sum(1)
    det = (sw * stt - st * st).clamp_min(1e-12)
    best = torch.full_like(u[:, 0], float("inf"))
    step = torch.zeros_like(u[:, 0])
    for k in (0.0, -1.0, 1.0):
        y = u.clone()
        y[:, -1] += TWO_PI * k
        sy, sty, syy = (w * y).sum(1), (w * t * y).sum(1), (w * y * y).sum(1)
        a = (stt * sy - st * sty) / det
        b = (sw * sty - st * sy) / det
        ssr = syy - a * sy - b * sty
        better = ssr < best - 1e-6 * syy.abs()
        best = torch.where(better, ssr, best)
        step = torch.where(better, torch.full_like(step, k), step)
    return step


def _time_path(c: Tensor, rough_max: float) -> tuple[Tensor, Tensor]:
    """Follow each voxel's change ``c`` (V, t) along time: (path centred within half a
    period of the reference by its median, smooth enough to trust)."""
    step = _wrap(torch.diff(c, dim=1))
    u = torch.cat([c[:, :1], c[:, :1] + torch.cumsum(step, 1)], 1)
    u = u - TWO_PI * torch.round(u.median(1, keepdim=True).values / TWO_PI)
    smooth = (step.abs() > _ROUGH_STEP).float().mean(1) <= rough_max
    return u, smooth


def _poly_basis(coords: Tensor, order: int) -> Tensor:
    """Low-order 3-D Legendre polynomial on coordinates scaled to [-1, 1]: (V, k)."""
    x, y, z = coords.unbind(1)
    cols = [torch.ones_like(x)]
    if order >= 1:
        cols += [x, y, z]
    if order >= 2:
        cols += [(3 * x * x - 1) / 2, (3 * y * y - 1) / 2, (3 * z * z - 1) / 2, x * y, x * z, y * z]
    return torch.stack(cols, 1)


def _slab_coords(a: int, b: int, dims: tuple[int, int, int], idx: Tensor) -> Tensor:
    """[-1, 1] coordinates of the slab voxels ``idx`` (slab = planes a..b of axis 0)."""
    d0, d1, d2 = dims
    g = torch.stack(
        torch.meshgrid(
            torch.arange(a, b, device=idx.device),
            torch.arange(d1, device=idx.device),
            torch.arange(d2, device=idx.device),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)[idx]
    scale = torch.tensor([max(d0 - 1, 1), max(d1 - 1, 1), max(d2 - 1, 1)], device=idx.device)
    return g.float() / scale * 2 - 1


def _anchor_slab(
    P: Tensor,
    M: Tensor,
    uref: Tensor,
    offset: Tensor,
    inside: Tensor,
    te: Tensor,
    *,
    temporal_path: bool,
    rough_max: float,
    at_anchor: Tensor | None = None,
    anchor_times: Tensor | None = None,
    predicted: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One slab, voxels flattened: P, M (V, e, t). Returns (U (V,e,t), field (V,t), path (V,)).

    ``predicted`` (V, t): the shortest echo's expected change from the frame-wide field
    fit; voxels not followed in time wrap only their residual about it.
    """
    ne = P.shape[1]
    D = _wrap(P - offset[:, None, None] - uref[:, :, None])
    c = D[:, 0]
    path = torch.zeros(c.shape[0], dtype=torch.bool, device=c.device)
    if predicted is not None:
        c = predicted + _wrap(c - predicted)
    if temporal_path and c.shape[1] > 2:
        u, smooth = _time_path(c, rough_max)
        moved = (torch.round((u - c) / TWO_PI) != 0).any(1)
        path = moved & smooth & inside
        if at_anchor is not None and anchor_times is not None and at_anchor.shape[1]:
            # A voxel that moves this far decoheres the temporal mean, so the
            # reference's count there is unreliable. Set the path's level by majority
            # against the ROMEO'd frames at their own time points instead.
            k = torch.round((uref[:, :1] + u[:, anchor_times] - at_anchor) / TWO_PI)
            valid = torch.isfinite(k)
            vote = torch.nanmedian(torch.where(valid, k, torch.nan), 1).values
            agree = ((k == vote[:, None]) & valid).sum(1) > 0.5 * valid.sum(1)
            u = u - TWO_PI * torch.where(path & agree, torch.nan_to_num(vote), 0.0)[:, None]
        c = torch.where(path[:, None], u, c)
    w = M * M
    dphi = torch.empty_like(D)
    dphi[:, 0] = c
    for e in range(1, ne):
        sl = (w[:, :e] * dphi[:, :e] * te[None, :e, None]).sum(1) / (
            w[:, :e] * te[None, :e, None] ** 2
        ).sum(1).clamp_min(1e-12)
        dphi[:, e] = D[:, e] + TWO_PI * torch.round((sl * te[e] - D[:, e]) / TWO_PI)
    U = uref[:, :, None] + dphi
    for e in range(1, ne):
        sl = (w[:, :e] * U[:, :e] * te[None, :e, None]).sum(1) / (
            w[:, :e] * te[None, :e, None] ** 2
        ).sum(1).clamp_min(1e-12)
        U[:, e] += TWO_PI * torch.round((sl * te[e] - U[:, e]) / TWO_PI)
        if e >= 2:
            U[:, e] += TWO_PI * _offset_free_step(U[:, : e + 1], w[:, : e + 1], te[: e + 1])
    U = U * inside[:, None, None]
    field = (w * U * te[None, :, None]).sum(1) / (w * te[None, :, None] ** 2).sum(1).clamp_min(
        1e-12
    )
    return U, field * (1000.0 / TWO_PI), path


def _predicted_change(a: int, b: int, dims, idx: Tensor, coef: Tensor) -> Tensor:
    """The frame-wide field fit at the slab voxels ``idx``, centred on its time median."""
    order = {1: 0, 4: 1, 10: 2}[coef.shape[0]]
    h = _poly_basis(_slab_coords(a, b, tuple(dims), idx), order) @ coef
    return h - h.median(1, keepdim=True).values


def _fit_frame_field(
    pv: np.ndarray,
    uv: np.ndarray,
    ofv: np.ndarray,
    mkv: np.ndarray,
    sx: int,
    order: int,
    rough_max: float,
    device: torch.device,
) -> Tensor:
    """Per-frame low-order polynomial of the shortest echo's change, (k, t).

    Fitted on the voxels whose change can be followed along time (smooth), so it
    captures the frame-wide field (breathing, breath-holds, drift) even where it
    moves by more than half a period; slab views as in :func:`unwrap_timeseries`.
    """
    device = linalg_device(device)  # float64 throughout; Metal has none
    d0, d1, d2 = pv.shape[:3]
    nt = pv.shape[-1]
    k = _poly_basis(torch.zeros(1, 3), order).shape[1]
    xtx = torch.zeros(k, k, dtype=torch.float64, device=device)
    xty = torch.zeros(k, nt, dtype=torch.float64, device=device)
    for a in range(0, d0, sx):
        b = min(d0, a + sx)
        n = (b - a) * d1 * d2
        inside = torch.from_numpy(np.ascontiguousarray(mkv[a:b] == 2)).to(device).reshape(n)
        idx = inside.nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        p0 = _gather(pv[a:b, :, :, 0], (n, nt), idx)
        c = _wrap(
            p0
            - _gather(ofv[a:b], (n,), idx)[:, None]
            - _gather(uv[a:b, :, :, 0], (n,), idx)[:, None]
        )
        u, smooth = _time_path(c, rough_max)
        y = (u - u.median(1, keepdim=True).values)[smooth].to(device).to(torch.float64)
        x = (
            _poly_basis(_slab_coords(a, b, (d0, d1, d2), idx[smooth]), order)
            .to(device)
            .to(torch.float64)
        )
        xtx += x.T @ x
        xty += x.T @ y
    ridge = 1e-6 * torch.trace(xtx) / k * torch.eye(k, dtype=xtx.dtype, device=device)
    return torch.linalg.solve(xtx + ridge, xty).float()


def unwrap_timeseries(
    phase: np.ndarray,
    mag: np.ndarray,
    tes: Sequence[float],
    *,
    device: torch.device,
    temporal_path: bool = True,
    rough_max: float = 0.05,
    anchor_frames: int = 8,
    frame_field_order: int | None = 2,
    automask_dilation: int = 3,
    return_unwrapped: bool = False,
    verbose: bool = True,
) -> UnwrapResult:
    """Unwrap ``(x, y, z, e, t)`` phase (radians, [-π, π)) against one reference.

    ``mag`` matches ``phase``; ``tes`` in ms, shortest first. ``rough_max`` is the
    largest fraction of frame-to-frame steps above π/2 a voxel may have and still
    have its shortest-echo change followed in time (see the module docstring).
    ``frame_field_order``: polynomial order of the per-frame field change fitted on
    those smooth voxels; every other voxel wraps only its residual about it, so a
    breath-hold that moves the whole head by more than half a period cannot wrap
    them. None turns it off.
    """
    if phase.shape != mag.shape or phase.ndim != 5:
        raise ValueError(
            f"phase/mag must share an (x, y, z, e, t) shape: {phase.shape} vs {mag.shape}"
        )
    te_np = np.asarray(tes, np.float32)
    if te_np.size != phase.shape[3]:
        raise ValueError(f"{te_np.size} echo times for {phase.shape[3]} echoes")
    if np.any(np.diff(te_np) <= 0):
        raise ValueError(f"echo times must increase: {te_np.tolist()}")
    nx, ny, nz, ne, nt = phase.shape
    t0 = time.perf_counter()
    keep = signal_frames(mag)
    ax = _slab_axis(phase)
    pv: np.ndarray | Tensor = np.moveaxis(phase, ax, 0)
    mv: np.ndarray | Tensor = np.moveaxis(mag, ax, 0)
    # Three passes read the whole series (complex mean, frame-wide field, main);
    # when it fits comfortably, copy it to the device once instead of three times.
    if device.type != "cpu" and 2 * phase.size * 4 <= 0.35 * get_available_memory(device):
        pv, mv = _to_device(pv, device), _to_device(mv, device)
    zbar = np.moveaxis(_complex_mean(pv, mv, keep, device), 0, ax)
    good = np.flatnonzero(keep)
    picks = good[
        np.unique(np.linspace(0, good.size - 1, min(anchor_frames, good.size)).round().astype(int))
    ]
    uref, mask, offset, at_anchor = unwrap_reference(
        np.angle(zbar).astype(np.float32),
        np.abs(zbar).astype(np.float32),
        te_np,
        anchors=[(phase[..., t], mag[..., t]) for t in picks] if anchor_frames > 0 else (),
        automask_dilation=automask_dilation,
        device=device,
    )
    del zbar
    t_ref = time.perf_counter() - t0

    field = np.zeros((nx, ny, nz, nt), np.float32)
    path_map = np.zeros((nx, ny, nz), bool)
    unwrapped = np.zeros(phase.shape, np.float32) if return_unwrapped else None
    te = torch.from_numpy(te_np).to(device)
    views = [np.moveaxis(a, ax, 0) for a in (uref, offset, mask, field, path_map, at_anchor)]
    uv, ofv, mkv, fv, pmv, av = views
    n_anchor = at_anchor.shape[-1]
    t_anchor = torch.from_numpy(picks[:n_anchor]).to(device) if n_anchor else None
    unv = np.moveaxis(unwrapped, ax, 0) if unwrapped is not None else None
    d1, d2 = pv.shape[1:3]
    sx = _slab_width(d1 * d2, ne, nt, device)
    slabs = range(0, pv.shape[0], sx)
    dims = pv.shape[:3]
    coef = None
    if frame_field_order is not None and temporal_path and nt > 2:
        coef = _fit_frame_field(pv, uv, ofv, mkv, sx, frame_field_order, rough_max, device)
    for a in tqdm(
        slabs, desc="  anchored unwrap", leave=True, disable=not verbose or len(slabs) < 2
    ):
        b = min(pv.shape[0], a + sx)
        n = (b - a) * d1 * d2
        inside = torch.from_numpy(np.ascontiguousarray(mkv[a:b] > 0)).to(device).reshape(n)
        idx = inside.nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        U, f, p = _anchor_slab(
            _gather(pv[a:b], (n, ne, nt), idx),
            _gather(mv[a:b], (n, ne, nt), idx),
            _gather(uv[a:b], (n, ne), idx),
            _gather(ofv[a:b], (n,), idx),
            torch.ones(idx.numel(), dtype=torch.bool, device=device),
            te,
            temporal_path=temporal_path,
            rough_max=rough_max,
            at_anchor=_gather(av[a:b], (n, n_anchor), idx) if n_anchor else None,
            anchor_times=t_anchor,
            predicted=None if coef is None else _predicted_change(a, b, dims, idx, coef),
        )
        full = torch.zeros(n, nt, device=device)
        full[idx] = f
        fv[a:b] = full.reshape(b - a, d1, d2, nt).cpu().numpy()
        pm = torch.zeros(n, dtype=torch.bool, device=device)
        pm[idx] = p
        pmv[a:b] = pm.reshape(b - a, d1, d2).cpu().numpy()
        if unv is not None:
            full = torch.zeros(n, ne, nt, device=device)
            full[idx] = U
            unv[a:b] = full.reshape(b - a, d1, d2, ne, nt).cpu().numpy()
        del U, f, full
    if verbose:
        dt = time.perf_counter() - t0
        n_in = int((mask > 0).sum())
        print(
            f"  anchored unwrap: {nt} frames x {ne} echo(es) in {dt:.1f}s "
            f"(reference {t_ref:.1f}s); {n_in:,} voxels, "
            f"{int(path_map.sum()):,} followed in time"
        )
    return UnwrapResult(
        field_hz=field,
        mask=mask,
        phase_offset=offset,
        temporal_path=path_map,
        unwrapped=unwrapped,
    )


def _gaussian_symmetric(vols: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian of ``(T, x, y, z)`` volumes, scipy's ``reflect`` edges, truncate 4."""
    r = int(4.0 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=vols.device, dtype=vols.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    out = vols
    for ax in (1, 2, 3):
        n = out.shape[ax]
        idx = torch.arange(-r, n + r, device=vols.device)
        # scipy 'reflect' = (d c b a | a b c d | d c b a): mirror including the edge sample
        period = 2 * n
        idx = torch.remainder(idx, period)
        idx = torch.where(idx >= n, period - 1 - idx, idx)
        padded = out.index_select(ax, idx)
        moved = padded.movedim(ax, -1)
        sh = moved.shape
        conv = torch.nn.functional.conv1d(moved.reshape(-1, 1, sh[-1]), k[None, None])
        out = conv.reshape(*sh[:-1], n).movedim(-1, ax)
    return out


def _low_rank(a: Tensor, k: int) -> Tensor:
    """Rank-``k`` reconstruction of a tall (V, T) matrix via its T x T Gram matrix."""
    from ..utils import to_factor_f64

    g = to_factor_f64(a.T @ a)
    _, vecs = torch.linalg.eigh(g)
    v = vecs[:, -k:].to(a.device, a.dtype)
    return (a @ v) @ v.T


def svd_filter_fields(
    field: np.ndarray,
    mask: np.ndarray,
    voxel_size: float,
    *,
    border_filt: tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    device: torch.device,
) -> np.ndarray:
    """warpkit's field denoising: border voxels (mask 1) from a smoothed low-rank fit,
    then every masked voxel from a rank-``svd_filt`` fit. ``mask`` is 3-D or 4-D."""
    nt = field.shape[-1]
    m4 = mask if mask.ndim == 4 else np.broadcast_to(mask[..., None], field.shape)
    union = (m4 > 0).any(-1)
    out = field.copy()
    flat_union = torch.from_numpy(union).to(device)
    border = torch.from_numpy(np.ascontiguousarray((m4 == 1)[union])).to(device)  # (V, T)
    inside = torch.from_numpy(np.ascontiguousarray((m4 > 0)[union])).to(device)
    f = torch.from_numpy(out).to(device)  # (x, y, z, T)
    if m4.max() == 2 and nt >= max(border_filt):
        sigma = (4.0 / voxel_size) / 2.355
        sm = _gaussian_symmetric(f.permute(3, 0, 1, 2), sigma).permute(1, 2, 3, 0)
        vals = f[flat_union]
        rec = _low_rank(sm[flat_union], border_filt[0])
        vals = torch.where(border, rec, vals)
        rec = _low_rank(vals, border_filt[1])
        vals = torch.where(border, rec, vals)
        f[flat_union] = vals
        del sm
    if nt >= svd_filt:
        vals = f[flat_union]
        rec = _low_rank(vals, svd_filt)
        f[flat_union] = torch.where(inside, rec, vals)
    return f.cpu().numpy()


def remove_offset_drift(
    unwrapped: np.ndarray,
    mag: np.ndarray,
    tes: Sequence[float],
    mask: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    """Remove the head-wide drift of the phase at TE 0 from ``unwrapped``, in place.

    The receiver phase is static in principle, but on some runs it drifts (one 3-echo
    run: 0.08 rad SD, head-wide). A drifting offset is constant across echoes in
    radians, so in Hz it falls as 1/TE: it biases the field map and reads as a
    TE-dependent partition shift downstream. Per volume, the per-voxel intercept of
    phase vs TE (magnitude²-weighted, minus its temporal mean) is fitted over ``mask``
    by a first-order spatial polynomial, which is subtracted from every echo. The
    static offset (already removed by the reference) is untouched.

    ``unwrapped``/``mag``: ``(x, y, z, e, t)``; needs 2+ echoes. Returns the per-volume
    polynomial coefficients ``(4, t)`` [rad] (offset, x, y, z on [-1, 1] coordinates).
    """
    device = linalg_device(device)  # float64 throughout; Metal has none
    te = torch.as_tensor(np.asarray(tes, dtype=np.float32), device=device)
    if te.numel() < 2:
        raise ValueError("the offset drift needs at least two echoes")
    shape = unwrapped.shape[:3]
    idx = np.flatnonzero(mask)
    nt = unwrapped.shape[-1]
    ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float32)
    basis = torch.from_numpy(
        np.c_[np.ones(len(idx)), ijk / np.maximum(np.array(shape) - 1, 1) * 2 - 1].astype(
            np.float32
        )
    ).to(device)
    xtx = torch.zeros(4, 4, dtype=torch.float64, device=device)
    xty = torch.zeros(4, nt, dtype=torch.float64, device=device)
    flat_u = unwrapped.reshape(-1, *unwrapped.shape[3:])
    flat_m = mag.reshape(-1, *mag.shape[3:])
    chunk = max(1, int(2e8 // max(1, unwrapped.shape[3] * nt * 4)))
    t_ = te[None, :, None]
    for a in range(0, idx.size, chunk):
        sel = idx[a : a + chunk]
        u = torch.from_numpy(np.ascontiguousarray(flat_u[sel])).to(device)  # V, e, t
        w = torch.from_numpy(np.ascontiguousarray(flat_m[sel])).to(device) ** 2
        sw, st, stt = w.sum(1), (w * t_).sum(1), (w * t_ * t_).sum(1)
        sy, sty = (w * u).sum(1), (w * t_ * u).sum(1)
        icpt = (stt * sy - st * sty) / (sw * stt - st * st).clamp_min(1e-12)  # V, t
        icpt = icpt - icpt.mean(1, keepdim=True)
        b = basis[a : a + chunk]
        xtx += (b.T @ b).double()
        xty += (b.T @ icpt).double()
    coef = torch.linalg.solve(xtx, xty).float()  # 4, t
    # subtract the fitted drift from every echo, plane by plane along x (cheap, in place)
    gx, gy, gz = (torch.linspace(-1, 1, n, device=device) for n in shape)
    for i in range(shape[0]):
        d = coef[0][None, None] + gx[i] * coef[1][None, None] + gy[:, None, None] * coef[2]
        d = d + gz[None, :, None] * coef[3]  # y, z, t
        d = d * torch.from_numpy(np.ascontiguousarray(mask[i])).to(device)[..., None]
        unwrapped[i] -= d.cpu().numpy()[:, :, None, :]
    return coef.cpu().numpy()


def field_from_unwrapped(
    unwrapped: np.ndarray, mag: np.ndarray, tes: Sequence[float], *, device: torch.device
) -> np.ndarray:
    """Magnitude²-weighted field through the origin [Hz], ``(x, y, z, t)``."""
    te = torch.as_tensor(np.asarray(tes, dtype=np.float32), device=device)
    out = np.zeros(unwrapped.shape[:3] + unwrapped.shape[4:], np.float32)
    for i in range(unwrapped.shape[0]):
        u = torch.from_numpy(np.ascontiguousarray(unwrapped[i])).to(device)
        w = torch.from_numpy(np.ascontiguousarray(mag[i])).to(device) ** 2
        t_ = te[None, None, :, None]
        f = (w * u * t_).sum(2) / (w * t_ * t_).sum(2).clamp_min(1e-12)
        out[i] = (f * (1000.0 / TWO_PI)).cpu().numpy()
    return out
