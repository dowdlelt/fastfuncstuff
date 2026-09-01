"""LOcal COrrection of MOtion via optical-flow COnfabulation (``ffs_locomoco``).

Single-echo EPI has residual, spatially non-linear frame-to-frame displacement
that rigid motion correction cannot remove — mostly along the phase-encode (PE)
axis (motion × B0-inhomogeneity interaction). Multi-echo MEDIC estimates this
per-voxel from phase; single-echo data has no phase to lean on. This tool instead
treats the already-moco'd magnitude series as a *movie* and estimates the
residual displacement with GPU optical flow.

Idea 1 (this file): the PE axis lies IN the acquired slice plane, so each slice is
a 2-D movie through time. Optical flow of frame ``t`` against a reference frame
gives a dense per-pixel displacement; its PE component is the residual motion we
could not correct. Assembled over slices and frames it becomes a per-frame warp
``(nx, ny, nz, T)`` along the PE axis — exactly what :func:`medic.save_medic_warp`
writes for ``ffs_nwarp`` (5-D DICOM-mm), so the correction composes into the same
single-resample chain as MEDIC / qwarp. Frame == reference gets a zero warp.

Outputs: the per-frame **warp**, a **nonlinear-motion-corrected** series (the
flow applied along PE), a 4-D **signed PE-flow** map (one scalar per voxel per
frame; sign = motion direction, scrub like a timeseries), and a **flow movie** —
a per-frame contact sheet of all slices, colored by the classic optical-flow /
circular-phase wheel (hue = direction, saturation = magnitude) so residual motion
is eyeballable. Optical flow invents wild displacements in the pure-noise air
outside the head; a feathered brain **automask** soft-gates the flow to zero there.

The optical-flow backend here is a dependency-free, batched, pyramidal
Lucas-Kanade solver (pure Torch, runs on the GPU), well suited to the
small-displacement residual-motion regime and trivially batched across the
(time × slice) movie frames. It is isolated in :func:`optical_flow_lk_2d` so a
future NVIDIA-hardware / RAFT backend can slot in behind the same signature.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .locomoco_interp_triton import shift1d_lanczos_triton, shift2d_triton
except Exception:  # pragma: no cover - Triton is optional and CUDA-only
    shift1d_lanczos_triton = shift2d_triton = None
from tqdm import tqdm

# Latched when a fused launch actually fails on this machine.
_fused_shift_unavailable = False


def _try_fused(fn, *args):
    """Run a fused CUDA shift, or return None so the caller takes the portable path.

    A GPU or driver that cannot build the kernel must cost one failed attempt,
    not abort the run and not retry once per frame.
    """
    global _fused_shift_unavailable
    if fn is None or _fused_shift_unavailable:
        return None
    try:
        return fn(*args)
    except AssertionError:
        raise  # a failed assertion is a bug here, not a missing GPU capability
    except Exception as exc:  # pragma: no cover - needs a Triton-hostile GPU
        _fused_shift_unavailable = True
        print(f"** fused CUDA shift unavailable ({type(exc).__name__}: {exc}); using PyTorch")
        return None


# Phase-encode direction letters -> NIfTI spatial axis (x=0, y=1, z=2). Only the
# AXIS matters for building the movie / placing the warp component; the sign of
# the displacement is data-driven (it falls out of the optical flow).
PE_DIR_TO_AXIS: dict[str, int] = {
    "LR": 0,
    "RL": 0,
    "x": 0,
    "i": 0,
    "AP": 1,
    "PA": 1,
    "y": 1,
    "j": 1,
    "IS": 2,
    "SI": 2,
    "z": 2,
    "k": 2,
}


# ── low-level optical-flow primitives (batched, GPU) ──────────────────────────
# The searchlight blurs tens of thousands of times per run, always with the same
# handful of (sigma, device, dtype) triples — recomputing the kernel each call was a
# measurable chunk of the solve. Cache the read-only kernel; callers never mutate it.
_KERNEL_CACHE: dict[tuple[float, torch.device, torch.dtype], torch.Tensor] = {}


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (sigma, device, dtype)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        _KERNEL_CACHE[key] = k
    return k


def _blur2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a batch of 2-D images ``(B, H, W)``."""
    if sigma <= 0:
        return img
    k = _gaussian_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img.unsqueeze(1)
    # ``replicate`` (not ``reflect``) so a blur radius larger than the plane still
    # pads: automask crops edge slices to tiny in-brain bounding boxes (e.g. 10x7),
    # and reflect requires pad < dim. Matches _gaussian_blur3d.
    #
    # One 4-sided pad, not one per axis. Replicate padding along W copies whole
    # columns and the vertical pass is per-column, so the columns the horizontal
    # pass sees are the same either way -- verified bit-identical (maxdiff 0.0).
    # It blurs 2r extra columns in the vertical pass, but that costs less than
    # the second pad's kernel launch: 57.5 -> 50.6 us on CUDA at (60, 64, 64).
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return x.squeeze(1)


def _spatial_highpass(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Unsharp spatial high-pass of a batch of 2-D images ``(B, H, W)``.

    ``img − blur(img, sigma)``: strips spatially-smooth intensity offsets (drift,
    slow BOLD, the respiration B0 modulation that rides along with a sub-voxel PE
    shift) while keeping the edges, where the shift is actually encoded. Optical
    flow's brightness-constancy assumption reads any non-motion intensity change as
    displacement, so feeding it the high-passed frames gives a more purely geometric
    target. Estimation only — the correction still resamples the raw series, so the
    output keeps its true intensities. Experimental (``-hpf_spatial``).
    """
    if sigma <= 0:
        return img
    return img - _blur2d(img, sigma)


def _gaussian_blur3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3-D Gaussian blur of a single ``(D, H, W)`` volume."""
    if sigma <= 0:
        return vol
    k = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
    r = (k.numel() - 1) // 2
    x = vol[None, None]
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1))
    x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1))
    x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1))
    return x[0, 0]


def _spatial_gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradients (d/dW, d/dH) of ``(B, H, W)`` images."""
    gx = torch.zeros_like(img)
    gy = torch.zeros_like(img)
    gx[:, :, 1:-1] = 0.5 * (img[:, :, 2:] - img[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (img[:, 2:, :] - img[:, :-2, :])
    return gx, gy


@functools.lru_cache(maxsize=32)
def _plane_meshgrid(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``(ys, xs)`` index grids for an ``(h, w)`` plane.

    The estimation loop warps the same handful of plane shapes (one per pyramid
    level) tens of thousands of times, and the grid is a pure function of shape
    plus device/dtype. Cached entries are read-only by construction — every
    consumer builds a new tensor from them rather than writing in place.
    """
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return ys, xs


def _warp2d(
    img: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    mode: str = "bilinear",
    radius: int = 3,
) -> torch.Tensor:
    """Sample ``img`` at ``(x + u, y + v)`` (u along W, v along H). ``(B, H, W)``.

    ``mode`` is the grid_sample interpolator (``bilinear`` or ``bicubic``). Bilinear
    damps a fractionally-shifted image, which biases the fixed point the estimation
    iterations converge to; ``bicubic`` resamples faithfully so they land on the true
    displacement — the accuracy win is largest on smooth data. ``lanczos`` is a 1-D
    (single-axis) resampler with no 2-D grid_sample equivalent, and the estimate is set by
    the pooling window anyway, so a 2-D estimation warp falls back to bilinear; the 1-D PE
    correction takes the true lanczos path in :func:`_correct_pe`.
    """
    needs_grad = torch.is_grad_enabled() and (
        img.requires_grad or u.requires_grad or v.requires_grad
    )
    if (
        img.device.type == "cuda"
        and img.dtype == torch.float32
        and mode in ("bicubic", "lanczos")
        and not needs_grad
    ):
        fused = _try_fused(shift2d_triton, img, v, u, 1, 2, mode, radius)
        if fused is not None:
            return fused
    if mode == "lanczos":
        return _shift2d_high_order(img, v, u, 1, 2, mode="lanczos", radius=radius)
    _, h, w = img.shape
    ys, xs = _plane_meshgrid(h, w, img.device, img.dtype)
    gxn = 2.0 * (xs.unsqueeze(0) + u) / max(w - 1, 1) - 1.0
    gyn = 2.0 * (ys.unsqueeze(0) + v) / max(h - 1, 1) - 1.0
    grid = torch.stack([gxn, gyn], dim=-1)
    out = F.grid_sample(
        img.unsqueeze(1), grid, mode=mode, padding_mode="border", align_corners=True
    )
    return out.squeeze(1)


def _warp2d_pe(
    img: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    mode: str,
    radius: int,
    *,
    pe_is_u: bool,
    two_dimensional: bool,
) -> torch.Tensor:
    """Use the warp's active dimensionality rather than the image tensor rank."""
    if mode == "lanczos" and not two_dimensional:
        return _shift1d_windowed_sinc(img, u if pe_is_u else v, 1 if pe_is_u else 0, radius)
    return _warp2d(img, u, v, mode, radius)


def optical_flow_lk_2d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    reg: float = 1e-3,
    pe_only_axis: int | None = None,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    max_shift: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense pyramidal Lucas-Kanade optical flow for a batch of 2-D image pairs.

    Solves, per pixel, for the displacement ``(u, v)`` (u along W, v along H) that
    pulls ``moving`` onto ``fixed`` (``moving(x + flow) ≈ fixed(x)``) — the sign
    convention for undoing motion by resampling ``moving`` at ``x + flow``.

    ``pe_only_axis`` (0 = W/columns, 1 = H/rows) constrains the flow to one axis
    (residual EPI motion is ~1-D along PE): a 1-DOF, more robust estimate.
    Returns ``(u, v)``; a constrained-out component is all zeros.

    ``max_shift`` bounds the accumulated displacement, and is the same guard the phase
    backend has always applied for the same reason (see :func:`phase_correlation_flow_2d`).
    The LK step is ``-blur(Ip·It) / (blur(Ip²) + reg)``; where the image has no structure
    the denominator collapses to ``reg`` (1e-3) and a single iteration can move a pixel
    hundreds of voxels. Nothing pulls it back, and n_iters × n_levels of that is a random
    walk with no bound — the mechanism behind a measured 3621-voxel displacement on a
    130-voxel axis, in the ramp-in slices at the end of a slab where the tissue signal is
    a third of mid-slab and there is nothing to track. Clamping per level keeps the
    unbounded case at the physical limit the caller already declared via ``-max_shift``.
    """
    fixed_pyr, moving_pyr = [fixed], [moving]
    for _ in range(n_levels - 1):
        if min(fixed_pyr[-1].shape[1:]) < 8:
            break
        fixed_pyr.append(F.avg_pool2d(_blur2d(fixed_pyr[-1], 1.0).unsqueeze(1), 2).squeeze(1))
        moving_pyr.append(F.avg_pool2d(_blur2d(moving_pyr[-1], 1.0).unsqueeze(1), 2).squeeze(1))

    u = torch.zeros_like(fixed_pyr[-1])
    v = torch.zeros_like(u)

    for level in range(len(fixed_pyr) - 1, -1, -1):
        fx_l, mv_l = fixed_pyr[level], moving_pyr[level]
        lh, lw = fx_l.shape[1], fx_l.shape[2]
        if u.shape[1] != lh or u.shape[2] != lw:
            sy, sx = lh / u.shape[1], lw / u.shape[2]
            u = (
                F.interpolate(
                    u.unsqueeze(1), size=(lh, lw), mode="bilinear", align_corners=True
                ).squeeze(1)
                * sx
            )
            v = (
                F.interpolate(
                    v.unsqueeze(1), size=(lh, lw), mode="bilinear", align_corners=True
                ).squeeze(1)
                * sy
            )

        for _ in range(n_iters):
            mv_w = _warp2d_pe(
                mv_l,
                u,
                v,
                warp_interp,
                warp_radius,
                pe_is_u=pe_only_axis == 0,
                two_dimensional=pe_only_axis is None,
            )
            it = mv_w - fx_l
            ix, iy = _spatial_gradients(mv_w)
            if pe_only_axis is not None:
                ip = ix if pe_only_axis == 0 else iy
                step = -_blur2d(ip * it, window_sigma) / (_blur2d(ip * ip, window_sigma) + reg)
                if pe_only_axis == 0:
                    u = u + step
                else:
                    v = v + step
            else:
                a11 = _blur2d(ix * ix, window_sigma) + reg
                a22 = _blur2d(iy * iy, window_sigma) + reg
                a12 = _blur2d(ix * iy, window_sigma)
                b1 = -_blur2d(ix * it, window_sigma)
                b2 = -_blur2d(iy * it, window_sigma)
                det = a11 * a22 - a12 * a12
                u = u + (a22 * b1 - a12 * b2) / det
                v = v + (a11 * b2 - a12 * b1) / det
            if max_shift is not None:
                u = u.clamp(-max_shift, max_shift)
                v = v.clamp(-max_shift, max_shift)

    return u, v


def _searchlight_field_and_conf(
    best_i: torch.Tensor,
    best_val: torch.Tensor,
    zero_val: torch.Tensor,
    ym2: torch.Tensor,
    ym1: torch.Tensor,
    y0: torch.Tensor,
    yp1: torch.Tensor,
    yp2: torch.Tensor,
    *,
    nd: int,
    r: float,
    trial_step: float,
    noshift_margin: float,
    reg_sigma: float,
    blur,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a searchlight's running-peak state into a regularised field + confidence.

    The shared tail of every xcorr searchlight (:func:`_xcorr_shift_1d`,
    :func:`xcorr_search_flow_3d`, :func:`xcorr_search_flow_3d_multiecho`). A bare
    per-voxel argmax of a noisy correlation curve is fragile: it can latch onto a
    spurious far peak, and it treats every voxel as independent even though the true
    displacement field is spatially smooth. Four steps harden it:

    1. **Sub-voxel peak** by a 5-point least-squares parabola (3-point near an edge) —
       the original behaviour.
    2. **Confidence** per voxel = peak quality (``best_val``) × prominence over the
       no-shift correlation (``best_val − zero_val``), knocked down ×0.1 where the peak
       railed at the ``±max_shift`` search boundary (a railed peak is almost always
       dropout/noise, not real motion — residual shifts are small by construction). This
       is the *soft* no-shift prior: a peak barely above no-shift earns little trust and
       is out-voted by its neighbours in step 4, without being destroyed.
    3. **Optional hard no-shift guard** (``noshift_margin > 0``, OFF by default): a peak
       beating the zero-shift correlation by less than ``noshift_margin`` is zeroed
       (displacement and confidence) so a neighbour fills it in. Off by default because
       residual motion is itself small — its prominence is small, so a hard threshold
       risks zeroing real sub-voxel shifts. Enable only on very noisy data.
    4. **Confidence-weighted smoothing**: ``blur(c·field)/blur(c)`` with a small
       relative floor on ``c`` — high-confidence voxels flow into ambiguous ones (the
       field is physically smooth, so we *borrow from trustworthy neighbours* rather
       than blur the data itself), and it degrades gracefully to a plain blur where
       confidence is uniform. This is the workhorse. ``reg_sigma <= 0`` disables it.

    Returns ``(field, conf)`` — ``conf`` doubles as the saved diagnostic quality map.
    """
    b5 = (-2.0 * ym2 - ym1 + yp1 + 2.0 * yp2) / 10.0
    a5 = (5.0 * (4.0 * ym2 + ym1 + yp1 + 4.0 * yp2) - 10.0 * (ym2 + ym1 + y0 + yp1 + yp2)) / 70.0
    vtx5 = torch.where(a5.abs() > 1e-9, -b5 / (2.0 * a5), torch.zeros_like(a5)).clamp(-1.0, 1.0)
    den3 = ym1 - 2.0 * y0 + yp1
    vtx3 = torch.where(den3.abs() > 1e-6, 0.5 * (ym1 - yp1) / den3, torch.zeros_like(den3))
    vtx3 = vtx3.clamp(-1.0, 1.0)
    can5 = (best_i >= 2) & (best_i <= nd - 3)
    can3 = (best_i >= 1) & (best_i <= nd - 2)
    sub = torch.where(can5, vtx5, torch.where(can3, vtx3, torch.zeros_like(vtx5)))
    field = (-r + best_i.to(best_val.dtype) * trial_step) + sub * trial_step

    prominence = (best_val - zero_val).clamp_min(0.0)
    at_edge = (best_i <= 0) | (best_i >= nd - 1)
    conf = best_val.clamp_min(0.0) * prominence
    conf = torch.where(at_edge, conf * 0.1, conf)

    if noshift_margin > 0.0:
        keep = prominence >= noshift_margin
        field = torch.where(keep, field, torch.zeros_like(field))
        conf = torch.where(keep, conf, torch.zeros_like(conf))

    if reg_sigma > 0.0:
        # Relative floor: keeps blur(c) well clear of 0 (no magnitude shrinkage) and
        # makes the smoother fall back to a plain Gaussian where confidence is flat.
        cw = conf + 1e-3 * conf.amax()
        field = blur(cw * field) / blur(cw).clamp_min(1e-12)

    return field, conf


def _side_first_peak(stack: torch.Tensor):
    """First peak of a per-voxel correlation curve swept outward from zero.

    ``stack`` is ``(K, …)`` — the correlation at offsets ``0, ±step, ±2·step, …`` on ONE
    side, index 0 being no-shift. The *first* peak (nearest zero) is where the curve stops
    rising: everything past it (later oscillation humps) is ignored, and if the curve falls
    immediately the peak is at 0 (that side contributes no shift). Connectivity to zero is
    automatic — we only climb from index 0. Returns ``(peak_idx, ym1, y0, yp1)``: the
    integer offset index of the peak and the three samples around it for a sub-voxel vertex.
    """
    kk = stack.shape[0]
    if kk == 1:
        z = torch.zeros_like(stack[0], dtype=torch.long)
        return z, stack[0], stack[0], stack[0]
    rising = (stack[1:] > stack[:-1]).to(stack.dtype)  # (K-1, …)
    # cumprod stays 1 while every step so far rose, 0 once any step fell → the count of
    # leading rises IS the first-peak index (0 if it never rose).
    peak_idx = torch.cumprod(rising, dim=0).sum(0).long()

    def gather(idx):
        return stack.gather(0, idx.clamp(0, kk - 1).unsqueeze(0)).squeeze(0)

    y0 = gather(peak_idx)
    return peak_idx, gather(peak_idx - 1), y0, gather(peak_idx + 1)


def _first_peak_field_and_conf(
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    trial_step: float,
    noshift_margin: float,
    reg_sigma: float,
    ambiguity_frac: float,
    max_offset: float,
    blur,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rule-based peak finder replacing argmax — the first REAL peak nearest zero.

    ``pos`` / ``neg`` are ``(K, …)`` correlation stacks swept outward from zero (offsets
    ``0,+step,…`` and ``0,−step,…``; both share index 0 = no-shift). Per voxel:

    - find the first peak on each side (:func:`_side_first_peak`), sub-voxel by a 3-point
      vertex — 0 is on each peak's slope by construction (no-shift prior baked in);
    - the winning side is the higher peak; a peak counts only if it beats no-shift by
      ``noshift_margin``;
    - a **bilateral** rise where the losing side is comparably prominent (within
      ``ambiguity_frac``) is degenerate/spurious → default to **zero**; a clear winner is
      taken. Later oscillation humps never enter — only the first peak on each side does.

    Then the same confidence-weighted smoothing as :func:`_searchlight_field_and_conf`.
    """
    corr0 = pos[0]
    pi, pym1, py0, pyp1 = _side_first_peak(pos)
    ni, nym1, ny0, nyp1 = _side_first_peak(neg)

    def subvox(ym1, y0, yp1, idx):
        den = ym1 - 2.0 * y0 + yp1
        s = torch.where(den.abs() > 1e-6, 0.5 * (ym1 - yp1) / den, torch.zeros_like(den))
        return torch.where(idx >= 1, s.clamp(-1.0, 1.0), torch.zeros_like(s))

    pos_field = (pi.to(corr0.dtype) + subvox(pym1, py0, pyp1, pi)) * trial_step  # ≥ 0
    neg_field = -(ni.to(corr0.dtype) + subvox(nym1, ny0, nyp1, ni)) * trial_step  # ≤ 0
    prom_p = (py0 - corr0).clamp_min(0.0)
    prom_n = (ny0 - corr0).clamp_min(0.0)

    win_pos = py0 >= ny0
    win_val = torch.where(win_pos, py0, ny0)
    win_field = torch.where(win_pos, pos_field, neg_field)
    win_prom = torch.where(win_pos, prom_p, prom_n)
    lose_prom = torch.where(win_pos, prom_n, prom_p)
    # Bilateral & comparable → ambiguous (spurious/degenerate) → keep zero.
    ambiguous = (lose_prom > noshift_margin) & (lose_prom >= (1.0 - ambiguity_frac) * win_prom)
    credible = (win_prom > noshift_margin) & (~ambiguous)

    field = torch.where(credible, win_field, torch.zeros_like(win_field))
    # The edge parabola can extrapolate a hair past the searched range — a shift beyond
    # max_shift is unphysical, so clamp before smoothing (keeps rails from bleeding out).
    field = field.clamp(-max_offset, max_offset)
    conf = torch.where(credible, win_val.clamp_min(0.0) * win_prom, torch.zeros_like(win_val))
    if reg_sigma > 0.0:
        amax = conf.amax(dim=tuple(range(1, conf.ndim)), keepdim=True)
        cw = conf + 1e-3 * amax
        field = blur(cw * field) / blur(cw).clamp_min(1e-12)
    return field, conf


def _sweep_first_peak(
    ncc,
    r,
    trial_step,
    noshift_margin,
    reg_sigma,
    ambiguity_frac,
    blur,
    curve_out,
    min_rising_frac: float = 0.001,
    min_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outward-from-zero sweep + first-peak finder (the B+C path).

    ``ncc(s)`` returns the per-voxel local correlation at offset ``s``. Sweeps
    ``s = 0, ±step, ±2·step, …`` outward, ALWAYS at least ``min_steps`` samples per side (so
    the curve is characterized before the rising test can stop it — one point at ±0.5 vox is
    not enough to know a voxel is really rising), then **grows the range only while more than
    a FRACTION of voxels is still rising** — so a *coherent* large-shift region (which keeps
    that fraction up) IS searched to completion, and the sweep only stops once the still-rising
    set drops below ``min_rising_frac`` (≈14³ voxels of a whole-brain volume). The leftover
    stragglers below that are scattered noise/dropout (handled by reg/mask), so clamping them
    where they are is safe. This is the data-adaptive speedup: mostly-small motion searches
    ~±min_steps instead of ±r. A plain "any voxel still rising" test never fires across
    millions of voxels — some noise voxel is always climbing — hence the fraction. When
    ``curve_out`` is requested the full ±r range is swept so the saved landscape is complete.
    """
    corr0 = ncc(0.0)
    pos = [corr0]
    neg = [corr0]
    pos_rising = torch.ones_like(corr0, dtype=torch.bool)
    neg_rising = torch.ones_like(corr0, dtype=torch.bool)
    kmax = int(round(r / trial_step))
    # At least min_steps samples per side before the rising test can fire, but never more
    # than the search range itself (tiny max_shift / coarse step ⇒ few points, don't error).
    k_min = min(kmax, max(1, min_steps))
    n_vox = corr0.numel()
    early = curve_out is None
    for k in range(1, kmax + 1):
        s = k * trial_step
        cp, cn = ncc(s), ncc(-s)
        pos_rising = pos_rising & (cp > pos[-1])
        neg_rising = neg_rising & (cn > neg[-1])
        pos.append(cp)
        neg.append(cn)
        if early and k >= k_min:
            frac = float((pos_rising | neg_rising).sum()) / n_vox
            if frac < min_rising_frac:
                break
    if curve_out is not None:
        for cc in reversed(neg[1:]):
            curve_out.append(cc.detach().cpu())
        curve_out.append(corr0.detach().cpu())
        for cc in pos[1:]:
            curve_out.append(cc.detach().cpu())
    return _first_peak_field_and_conf(
        torch.stack(pos),
        torch.stack(neg),
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        ambiguity_frac=ambiguity_frac,
        max_offset=r,
        blur=blur,
    )


def _xcorr_shift_1d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pe_is_u: bool,
    max_shift: float,
    window_sigma: float,
    trial_step: float,
    interp: str,
    eps: float = 1e-4,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    curve_out: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-voxel pull shift along ONE in-plane axis by local correlation search.

    Slide ``moving`` along the axis (``pe_is_u`` → W/columns, else H/rows) over trial
    offsets spanning ``±max_shift`` in ``trial_step`` steps; at each, measure local
    normalized correlation with ``fixed`` under a ``window_sigma`` Gaussian searchlight.
    Per voxel the peak offset, refined by a 5-point least-squares parabola (peak ±2,
    robust to a noisy correlation curve), is the pull displacement. Trials are swept
    with a streaming running-peak, so memory is O(frame×slice), not O(trials×…).

    Returns ``(field, conf)``; ``noshift_margin`` / ``reg_sigma`` are the searchlight
    robustness knobs and ``conf`` the quality map — see
    :func:`_searchlight_field_and_conf`.
    """
    import math

    r = float(max(1, int(math.ceil(max_shift))))
    device, dtype = fixed.device, fixed.dtype
    b, h, w = fixed.shape

    def _win(x: torch.Tensor) -> torch.Tensor:
        return _blur2d(x, window_sigma)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    def _shift(img: torch.Tensor, d: float) -> torch.Tensor:
        if pe_is_u:
            gx = 2.0 * (xs + d) / max(w - 1, 1) - 1.0
            gy = 2.0 * ys / max(h - 1, 1) - 1.0
        else:
            gx = 2.0 * xs / max(w - 1, 1) - 1.0
            gy = 2.0 * (ys + d) / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None].expand(b, h, w, 2)
        return F.grid_sample(
            img[:, None], grid, mode=interp, padding_mode="border", align_corners=True
        )[:, 0]

    mean_f = _win(fixed)  # fixed moments precomputed once (only `moving` slides)
    var_f = (_win(fixed * fixed) - mean_f * mean_f).clamp_min(eps)

    offsets = torch.arange(-r, r + 1e-6, trial_step, device=device, dtype=dtype)
    nd = int(offsets.numel())

    # Streaming running-peak keeping the FOUR neighbours (±2) needed for a 5-point
    # least-squares parabola — more robust than the 3-point analytic vertex and free
    # (no extra trials). `need` counts how many right neighbours a fresh peak still
    # awaits; prev1/prev2 are the trailing two correlations for its left neighbours.
    z = torch.zeros((b, h, w), device=device, dtype=dtype)
    best_val = torch.full((b, h, w), float("-inf"), device=device, dtype=dtype)
    best_i = torch.zeros((b, h, w), device=device, dtype=torch.long)
    zero_val = z.clone()  # correlation at no-shift (s≈0) — the no-shift-guard reference
    ym2, ym1, y0, yp1, yp2 = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    need = torch.zeros((b, h, w), device=device, dtype=torch.long)
    prev1: torch.Tensor | None = None
    prev2: torch.Tensor | None = None
    for i in range(nd):
        s = float(offsets[i])
        mw = _shift(moving, s)
        mean_m = _win(mw)
        var_m = (_win(mw * mw) - mean_m * mean_m).clamp_min(eps)
        corr = (_win(fixed * mw) - mean_f * mean_m) / torch.sqrt(var_f * var_m)
        if curve_out is not None:
            curve_out.append(corr.detach().cpu())
        if abs(s) < trial_step * 0.5 + 1e-6:
            zero_val = corr
        # Capture right neighbours (first the +1, then the +2) for awaiting peaks.
        yp1 = torch.where(need == 2, corr, yp1)
        yp2 = torch.where(need == 1, corr, yp2)
        need = torch.where(need > 0, need - 1, need)
        newp = corr > best_val
        ym2 = torch.where(newp, prev2 if prev2 is not None else corr, ym2)
        ym1 = torch.where(newp, prev1 if prev1 is not None else corr, ym1)
        y0 = torch.where(newp, corr, y0)
        best_val = torch.where(newp, corr, best_val)
        best_i = torch.where(newp, torch.full_like(best_i, i), best_i)
        need = torch.where(newp, torch.full_like(need, 2), need)  # await this peak's ±1,±2
        prev2 = prev1
        prev1 = corr

    field, conf = _searchlight_field_and_conf(
        best_i,
        best_val,
        zero_val,
        ym2,
        ym1,
        y0,
        yp1,
        yp2,
        nd=nd,
        r=r,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        blur=lambda x: _blur2d(x, reg_sigma),
    )
    return field, conf


def xcorr_search_flow_2d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    pe_is_u: bool,
    max_shift: float = 3.0,
    window_sigma: float = 2.0,
    trial_step: float = 0.5,
    interp: str = "bicubic",
    eps: float = 1e-4,
    dual: bool = False,
    n_passes: int = 3,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    conf_out: list[torch.Tensor] | None = None,
    curve_out: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local cross-correlation searchlight backend, same ``(fixed, moving) -> (u, v)``.

    Residual EPI motion is a small local translation; we do the literal thing —
    slide ``moving`` and take the per-voxel offset of peak local correlation (a true
    alignment search, so accurate single-shot). See :func:`_xcorr_shift_1d`.

    Single-PE searches one axis (``pe_is_u`` selects it). ``dual`` (two PE axes)
    searches BOTH, but *separably* — a 1-D search along each axis, alternating and
    warping by the running estimate for ``n_passes`` (converges in ~3) — to avoid the
    O(trials²) blow-up of a joint 2-D offset grid. ``conf_out`` / ``curve_out`` capture
    the single-PE searchlight quality / landscape (no-ops for the separable dual pass).
    """
    if not dual:
        s, conf = _xcorr_shift_1d(
            fixed,
            moving,
            pe_is_u,
            max_shift,
            window_sigma,
            trial_step,
            interp,
            eps,
            noshift_margin,
            reg_sigma,
            curve_out=curve_out,
        )
        if conf_out is not None:
            conf_out.append(conf)
        z = torch.zeros_like(s)
        return (s, z) if pe_is_u else (z, s)

    u = torch.zeros_like(fixed)
    v = torch.zeros_like(fixed)
    for _ in range(n_passes):
        mw = _warp2d_pe(
            moving,
            u,
            v,
            warp_interp,
            warp_radius,
            pe_is_u=pe_is_u,
            two_dimensional=dual,
        )
        du, _ = _xcorr_shift_1d(
            fixed,
            mw,
            True,
            max_shift,
            window_sigma,
            trial_step,
            interp,
            eps,
            noshift_margin,
            reg_sigma,
        )
        u = u + du
        mw = _warp2d_pe(
            moving,
            u,
            v,
            warp_interp,
            warp_radius,
            pe_is_u=pe_is_u,
            two_dimensional=dual,
        )
        dv, _ = _xcorr_shift_1d(
            fixed,
            mw,
            False,
            max_shift,
            window_sigma,
            trial_step,
            interp,
            eps,
            noshift_margin,
            reg_sigma,
        )
        v = v + dv
    return u, v


def _phase_slope_dense(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pe_is_u: bool,
    patch: int,
    stride: int,
    max_shift: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """One-shot dense PE shift from the FFT phase ramp of overlapping patches.

    Tiles the slice into ``patch``×``patch`` squares (stride ``stride``), 1-D FFTs
    each *along PE*, and reads the shift off the slope of the cross-power-spectrum
    phase (summed over the perpendicular lines, fit over low-k bins so it never
    wraps for ``|shift| <= max_shift``). Returns the dense pull field ``(B, H, W)``
    (patch-centre values bilinearly interpolated up). Biased low on its own — a
    local patch's shift isn't circular — so callers wrap it in warping iterations.
    """
    import math

    b, h, w = fixed.shape
    device, dtype = fixed.device, fixed.dtype
    p = patch
    pad = p // 2
    lh = (h + 2 * pad - p) // stride + 1
    lw = (w + 2 * pad - p) // stride + 1
    ppe = p  # patch extent along PE (patches are p×p; PE runs along dim=1 below)
    kmax = max(1, min(p // 2 - 1, int(p / (2.0 * max_shift))))

    # The working set holds a handful of (frames, kmax, perpendicular, positions)
    # complex tensors; at dense settings that is still GiBs for a long series.
    # Process the frame batch in chunks sized to fit — the per-patch estimate is
    # independent across frames, so this is exact.
    per_frame = kmax * max(h, w) * max(lh, lw) * 48  # ~3 complex working tensors
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device)
        budget = int(0.4 * free_b)
    else:
        budget = 2 * 1024**3
    chunk = max(1, min(b, budget // max(1, per_frame)))

    out = torch.empty(b, h, w, device=device, dtype=dtype)
    ks = torch.arange(1, kmax + 1, device=device, dtype=dtype)

    # Only bins 1..kmax of the per-patch PE spectrum are ever read, so the patches
    # themselves never have to exist: bin k of every patch along PE is a strided
    # correlation of the image with exp(-2*pi*i*k*j/p), and the per-patch sum over
    # the perpendicular lines is a box sum with the same window and stride. That
    # replaces a p*p-per-patch materialization plus a full FFT with two small
    # strided convolutions, and its cost barely moves with the frame count: at
    # 360 frames (patch 16, stride 8) 2.06 ms -> 0.54 ms, agreeing with the
    # patch-and-FFT form to 4e-7 relative. Below ~100 frames the two convolution
    # launches dominate and the direct form is quicker, so keep both.
    j = torch.arange(p, device=device, dtype=dtype)
    theta = (-2.0 * math.pi / p) * ks[:, None] * j[None, :]  # (kmax, p)
    dft_kernel = torch.cat([torch.cos(theta), torch.sin(theta)])[:, None, :]  # (2*kmax, 1, p)
    box_kernel = torch.ones(1, 1, p, device=device, dtype=dtype)
    # Which form is quicker is set by the direct path's patch buffer against the
    # convolution form's fixed ~0.5 ms of launch overhead: the convolutions barely
    # notice the frame count (0.53 ms at 64 frames, 0.54 at 360) while the patch
    # materialization scales with it (0.23 -> 2.06 ms). 16 MiB of patches is where
    # the two met on the measured shapes.
    batched_bins = b * lh * lw * p * p * 4 > 16 * 1024**2

    def _patches(x: torch.Tensor, n: int) -> torch.Tensor:
        # Tensor.unfold builds the tiling as strided views and lets the ordinary
        # copy kernel materialize it; F.unfold routes the same bytes through
        # im2col, which is ~5x slower here (9.6 ms -> 2.0 ms for the unfold+FFT
        # chain at 360 frames, patch 16, stride 8) for a bit-identical result.
        tiles = x[:, 0].unfold(1, p, stride).unfold(2, p, stride)  # (n, lh, lw, p, p)
        return tiles.reshape(n * lh * lw, p, p)

    def _pe_bins(x: torch.Tensor) -> torch.Tensor:
        """Bins 1..kmax of the windowed PE spectrum: ``(n, perp, kmax, positions)``."""
        n, q, pe = x.shape
        y = F.conv1d(x.reshape(n * q, 1, pe), dft_kernel, stride=stride)  # (n*q, 2*kmax, S)
        y = y.reshape(n, q, 2 * kmax, -1)
        return torch.complex(y[:, :, :kmax], y[:, :, kmax:])

    def _cross_by_convolution(f: torch.Tensor, m: torch.Tensor, n: int) -> torch.Tensor:
        # Lay the PE axis last, perpendicular second: (n, perp, PE).
        if not pe_is_u:
            f, m = f.transpose(1, 2), m.transpose(1, 2)
        # One global offset keeps the analytic DC cancellation (sum of the kernel
        # over a full period is zero for k >= 1) from losing precision on the large
        # positive means EPI carries. A per-patch mean would be equivalent: any
        # constant contributes only to bin 0.
        offset = f.mean()
        prod = _pe_bins(f - offset) * _pe_bins(m - offset).conj()  # (n, perp, kmax, S)
        n_, q_, k_, s_ = prod.shape
        flat = prod.permute(0, 2, 3, 1).reshape(n_ * k_ * s_, 1, q_)
        summed = torch.complex(
            F.conv1d(flat.real, box_kernel, stride=stride),
            F.conv1d(flat.imag, box_kernel, stride=stride),
        )  # (n*kmax*S, 1, T)
        t_ = summed.shape[-1]
        summed = summed.reshape(n_, k_, s_, t_)
        # Patch positions enumerate H then W, matching the unfold tiling; S runs
        # along PE, T along the perpendicular axis, so which is which flips with
        # the PE axis.
        order = (0, 3, 2, 1) if pe_is_u else (0, 2, 3, 1)
        return summed.permute(*order).reshape(n * lh * lw, k_)

    for b0 in range(0, b, chunk):
        b1 = min(b0 + chunk, b)
        cb = b1 - b0
        fpad = F.pad(fixed[b0:b1, None], (pad, pad, pad, pad), mode="reflect")[:, 0]
        mpad = F.pad(moving[b0:b1, None], (pad, pad, pad, pad), mode="reflect")[:, 0]
        if batched_bins:
            cross_k = _cross_by_convolution(fpad, mpad, cb)
        else:
            fp, mp = _patches(fpad[:, None], cb), _patches(mpad[:, None], cb)
            if pe_is_u:  # PE along W (dim=2) -> move it to dim=1 for the FFT
                fp, mp = fp.transpose(1, 2), mp.transpose(1, 2)
            fp = fp - fp.mean(dim=(1, 2), keepdim=True)
            mp = mp - mp.mean(dim=(1, 2), keepdim=True)
            # Patches are real and only bins 1..kmax are read, so the negative-
            # frequency half of a full complex FFT is pure waste: rfft returns bins
            # 0..ppe//2, and kmax <= ppe//2 - 1 by construction.
            cross = (torch.fft.rfft(fp, dim=1) * torch.fft.rfft(mp, dim=1).conj()).sum(dim=2)
            del fp, mp
            cross_k = cross[:, 1 : kmax + 1]
        ang = torch.angle(cross_k)
        wts = cross_k.abs()
        slope = (wts * ks * ang).sum(dim=1) / (wts * ks * ks).sum(dim=1).clamp_min(eps)
        # slope*ppe/2π is already the pull displacement (−Δ for moving=fixed(x+Δ)).
        shift = (slope * ppe / (2.0 * math.pi)).clamp(-max_shift, max_shift).reshape(cb, lh, lw)
        out[b0:b1] = F.interpolate(
            shift[:, None], size=(h, w), mode="bilinear", align_corners=True
        )[:, 0]
        del cross_k, ang, wts, slope, shift
    return out


def phase_correlation_flow_2d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    pe_is_u: bool,
    patch: int = 16,
    stride: int = 8,
    max_shift: float = 3.0,
    n_iters: int = 5,
    dual: bool = False,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-correlation searchlight backend (FFT phase ramp along PE).

    An alternative to :func:`optical_flow_lk_2d` / :func:`xcorr_search_flow_2d`,
    same ``(fixed, moving) -> (u, v)`` contract. A translation along an axis is a
    linear phase ramp of the FFT along that axis (the shift theorem), so the local
    shift reads straight off the cross-spectrum phase — the clean, "treat it as a
    vector" route. A single windowed patch reads the shift low (its content isn't
    circular), so we warp ``moving`` by the running estimate and re-read the
    residual, ``n_iters`` times, until the leakage bias vanishes. The accumulated
    field is the pull displacement (``moving(x+flow) ≈ fixed(x)``).

    Single-PE estimates one component (``pe_is_u`` selects it). ``dual`` (two PE
    axes) estimates BOTH u and v — one phase-ramp per axis inside the same warping
    loop; the FFT route makes this near-free (no combinatorial search).
    """
    want_u = dual or pe_is_u
    want_v = dual or not pe_is_u
    u = torch.zeros_like(fixed)
    v = torch.zeros_like(fixed)
    for _ in range(n_iters):
        mw = _warp2d_pe(
            moving,
            u,
            v,
            warp_interp,
            warp_radius,
            pe_is_u=pe_is_u,
            two_dimensional=dual,
        )
        # Clamp the ACCUMULATED field, not just each increment: this is residual
        # motion (small), so max_shift bounds the total displacement. Otherwise n_iters
        # increments each capped at max_shift let low-signal patches random-walk to
        # n_iters·max_shift — the "enormous displacement" blow-up, which the refine
        # loop then compounds by rebuilding its reference from the over-warped series.
        if want_u:
            u = (u + _phase_slope_dense(fixed, mw, True, patch, stride, max_shift)).clamp(
                -max_shift, max_shift
            )
        if want_v:
            v = (v + _phase_slope_dense(fixed, mw, False, patch, stride, max_shift)).clamp(
                -max_shift, max_shift
            )
    return u, v


def resolve_pe_axis(pe_dir: str) -> int:
    """Map a PE-direction token to a NIfTI axis.

    Accepts axis letters (x/y/z or the i/j/k voxel convention) and anatomical
    direction codes (AP/PA/LR/RL/IS/SI). A leading dash is tolerated so a habit
    of writing ``-pe_dir -j`` resolves the same as ``-pe_dir j``. Axis letters are
    case-insensitive; direction codes are matched uppercase.
    """
    key = pe_dir.strip().lstrip("-")
    for cand in (key, key.lower(), key.upper()):
        if cand in PE_DIR_TO_AXIS:
            return PE_DIR_TO_AXIS[cand]
    raise ValueError(
        f"Unknown -pe_dir '{pe_dir}'. Use axis letters x/y/z or i/j/k, "
        "or direction codes AP/PA/LR/RL/IS/SI."
    )


# Axis letters written with a leading dash (``-pe_dir -j``) look like option flags
# to argparse and get swallowed before the nargs value can consume them. Rewrite
# them to the bare token, but ONLY when they directly follow an axis-taking option
# so the ``-i`` alias for ``-input`` (and any other real short flag) is untouched.
_DASHED_AXIS_TOKEN = {f"-{c}": c for c in ("i", "j", "k", "x", "y", "z")}


def normalize_axis_argv(argv: list[str], axis_opts: set[str]) -> list[str]:
    """Un-dash axis letters that follow one of ``axis_opts`` (e.g. ``-pe_dir``).

    ``axis_opts`` are the option strings whose value(s) are axis tokens. Only the
    run of dashed axis letters immediately after such an option is rewritten, so
    ``-pe_dir -j`` becomes ``-pe_dir j`` while an unrelated ``-i input.nii`` is left
    alone.
    """
    out: list[str] = []
    expect = False
    for tok in argv:
        if tok in axis_opts:
            out.append(tok)
            expect = True
            continue
        if expect and tok in _DASHED_AXIS_TOKEN:
            out.append(_DASHED_AXIS_TOKEN[tok])
            continue  # stay in the run: -pe_dir accepts two axes (nargs="+")
        expect = False
        out.append(tok)
    return out


_AXIS_LETTER = {0: "x", 1: "y", 2: "z"}


def _geometry_report(
    shape, pe_axis: int, slice_axis: int, *, is_3dacq: bool, dual: bool = False
) -> str:
    """One line describing which axes/planes the correlation actually works on.

    The 2-D vs 3-D distinction is physical: a 2-D (multiband) acquisition takes each
    slice separately, so the search is confined to the ONE in-plane view that contains
    PE — we report that plane and how many slices we sweep along the slice axis. A 3-D
    acquisition is one coherent volume, so BOTH planes that contain the PE axis "see"
    the partition displacement and the single 3-D solve pools them — we report both.
    """
    dims = [int(shape[0]), int(shape[1]), int(shape[2])]
    L = _AXIS_LETTER
    if is_3dacq:
        if dual:
            # Two encode axes: the informative statement is which plane they span and
            # which axis is left un-encoded, not which planes "view" a single axis.
            other = next(a for a in (0, 1, 2) if a not in (pe_axis, slice_axis))
            enc = sorted((pe_axis, other))
            return (
                f"geometry: 3-D joint solve — primary PE axis {pe_axis} ({L[pe_axis]}, "
                f"{dims[pe_axis]} vox) + partition axis {other} ({L[other]}, {dims[other]} "
                f"vox), spanning the {L[enc[0]]}×{L[enc[1]]} plane "
                f"({dims[enc[0]]}×{dims[enc[1]]}); axis {slice_axis} ({L[slice_axis]}) is "
                f"un-encoded. Both fields are solved at once, with no ratio assumed "
                f"between them"
            )
        o0, o1 = (a for a in (0, 1, 2) if a != pe_axis)
        return (
            f"geometry: 3-D solve — PE axis {pe_axis} ({L[pe_axis]}, {dims[pe_axis]} vox) is "
            f"coherent through the volume; pooling both planes that view it: "
            f"{L[pe_axis]}×{L[o0]} ({dims[pe_axis]}×{dims[o0]}) and "
            f"{L[pe_axis]}×{L[o1]} ({dims[pe_axis]}×{dims[o1]})"
        )
    a0, a1 = sorted(a for a in (0, 1, 2) if a != slice_axis)
    pe_kind = "2 in-plane PE axes" if dual else f"PE axis {pe_axis} ({L[pe_axis]})"
    return (
        f"geometry: 2-D slicewise — {pe_kind} in the {L[a0]}×{L[a1]} plane "
        f"({dims[a0]}×{dims[a1]}); {dims[slice_axis]} slices along axis {slice_axis} "
        f"({L[slice_axis]})"
    )


def _validate_estimation_inputs(
    shape,
    pe_axis: int,
    slice_axis: int,
    *,
    is_3dacq: bool,
    max_shift: float,
    trial_step: float,
    window_sigma: float,
    verbose: bool = True,
) -> None:
    """Guard the estimation geometry/knobs — shared sanity for every path.

    Raises on the impossible (out-of-range axes, PE == slice in a 2-D acquisition,
    non-positive search knobs); warns on the merely risky (a PE axis too thin for the
    requested ``max_shift``, or a degenerate single-slice 2-D run) so a bad run is
    caught up front rather than producing a quietly-wrong field.
    """
    dims = [int(shape[0]), int(shape[1]), int(shape[2])]
    for name, ax in (("pe_axis", pe_axis), ("slice_axis", slice_axis)):
        if ax not in (0, 1, 2):
            raise ValueError(f"{name} must be 0, 1 or 2 (got {ax}).")
    if not is_3dacq and pe_axis == slice_axis:
        raise ValueError(
            f"-pe_axis ({pe_axis}) must differ from -slice_axis ({slice_axis}): in a 2-D "
            "acquisition PE must lie inside the slice plane to be visible to 2-D flow "
            "(use -is_3dacq if this is a 3-D-acquired series)."
        )
    if max_shift <= 0:
        raise ValueError(f"-max_shift must be > 0 (got {max_shift}).")
    if trial_step <= 0:
        raise ValueError(f"-xcorr_step must be > 0 (got {trial_step}).")
    if window_sigma < 0:
        raise ValueError(f"-window must be >= 0 (got {window_sigma}).")
    if verbose:
        pe_n = dims[pe_axis]
        if pe_n < 2 * max_shift + 1:
            print(
                f"   ⚠ PE axis {pe_axis} is only {pe_n} vox — thinner than the ±{max_shift:g} "
                "search span; the field near the edges may rail. Lower -max_shift if so."
            )
        if not is_3dacq and dims[slice_axis] < 2:
            print(
                f"   ⚠ only {dims[slice_axis]} slice along axis {slice_axis}: a 2-D run on a "
                "single slice — fine, but -is_3dacq is usually meant here."
            )


# ── flow → color (Middlebury / circular-phase wheel) ──────────────────────────
def flow_to_rgb(u: np.ndarray, v: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Color a 2-D flow ``(H, W)`` by direction (hue) and magnitude (saturation).

    Zero flow is white; hue winds through the color wheel with direction so the
    dominant displacement direction reads off instantly (a circular-phase map).
    Vectorized over arbitrary leading dims: ``u, v`` shaped ``(..., H, W)`` ->
    uint8 ``(..., H, W, 3)`` in a single conversion.
    """
    from matplotlib.colors import hsv_to_rgb

    ang = np.arctan2(v, u)  # [-π, π]
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99)) or 1.0
    hue = (ang / (2 * np.pi)) % 1.0
    sat = np.clip(mag / max_mag, 0.0, 1.0)
    hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
    return (hsv_to_rgb(hsv) * 255).astype(np.uint8)


def _color_wheel(size: int) -> np.ndarray:
    """Small circular color-wheel legend, uint8 ``(size, size, 3)`` on white."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    u = (xx - cx) / cx
    v = (yy - cy) / cy
    rgb = flow_to_rgb(u, v, max_mag=1.0)
    outside = (u * u + v * v) > 1.0
    rgb[outside] = 255
    return rgb


def _montage(slices_rgb: np.ndarray, pad: int = 1) -> np.ndarray:
    """Tile ``(nS, H, W, 3)`` slice images into a near-square contact sheet."""
    ns, h, w, _ = slices_rgb.shape
    cols = int(np.ceil(np.sqrt(ns)))
    rows = int(np.ceil(ns / cols))
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i in range(ns):
        r, c = divmod(i, cols)
        y0 = r * (h + pad) + pad
        x0 = c * (w + pad) + pad
        sheet[y0 : y0 + h, x0 : x0 + w] = slices_rgb[i]
    return sheet


# ── result container + orchestration ──────────────────────────────────────────
@dataclass
class LocomocoResult:
    """Per-frame residual-motion estimate and the outputs derived from it.

    ``u_canon`` / ``v_canon`` are the in-plane flow in canonical (T, nSlice, H, W)
    layout; ``corrected_canon`` is the PE-corrected series in the same layout.
    Geometry fields let the accessors map back to NIfTI ``(nx, ny, nz, T)``.
    """

    u_canon: torch.Tensor
    v_canon: torch.Tensor
    corrected_canon: torch.Tensor
    perm: list[int]
    pe_flow_is_u: bool
    pe_axis: int
    slice_axis: int
    orig_shape: tuple[int, int, int, int]
    a0: int = 0  # in-plane NIfTI axis carrying v (rows/H)
    a1: int = 1  # in-plane NIfTI axis carrying u (cols/W)
    dual: bool = False  # two PE axes: both u and v are real displacements
    pe_axis2: int | None = None  # partition (2nd PE) axis, dual runs only
    # Per-voxel separability of the two axes, (nx,ny,nz,T); see optical_flow_lk_3d_axes.
    sep_map: torch.Tensor | None = None
    # ── rotation-aware (idea 2) extras — None for the plain idea-1 path ──
    reproject_weights: torch.Tensor | None = None  # (T, 3) per-frame axis weights
    corrected_nifti: torch.Tensor | None = None  # precomputed (nx,ny,nz,T) 3-D-warp series
    drift_rms: float | None = None  # chain-vs-anchor drift (in-brain), voxels
    drift_max: float | None = None
    fused: bool = False  # whether the anchor smoother was engaged
    # ── xcorr searchlight diagnostics — None unless the xcorr backend ran ──
    confidence: torch.Tensor | None = None  # (nx,ny,nz,T) per-voxel peak quality map
    corr_curve: torch.Tensor | None = None  # (nx,ny,nz,nd) per-voxel corr vs offset, one frame
    corr_offsets: torch.Tensor | None = None  # (nd,) trial offsets (voxels) for corr_curve

    def _to_nifti(self, canon: torch.Tensor) -> torch.Tensor:
        inv = [0, 0, 0, 0]
        for new_pos, old_axis in enumerate(self.perm):
            inv[old_axis] = new_pos
        return canon.permute(inv).contiguous()

    def pe_displacement(self) -> torch.Tensor:
        """Per-frame SIGNED PE displacement ``(nx, ny, nz, T)`` in voxel units.

        The single-PE flow map: one signed scalar per voxel per frame — sign is the
        direction of residual motion along the PE axis, magnitude is how far. Scrub
        it like a timeseries. Also the content of the saved warp (before mm). For a
        dual-PE run use :meth:`warp_components` / :meth:`flow_magnitude` instead.
        """
        return self._to_nifti(self.u_canon if self.pe_flow_is_u else self.v_canon)

    def warp_components(self) -> list[tuple[int, torch.Tensor]]:
        """``(nifti_axis, signed_displacement)`` pairs to write into the warp.

        Single-PE: one entry on the PE axis. Dual-PE: two — u on ``a1``, v on ``a0``.
        Rotation-aware (idea 2): the reference-frame PE magnitude ``p`` reprojected
        onto all three axes by the per-frame head rotation — mostly PE, with the
        small IS/LR leakage a head tilt throws off-axis. Each axis component is
        ``p * reproject_weights[:, axis]`` (already in voxel-pull units, so it feeds
        :func:`save_medic_warp` exactly like the single-PE case). Reduces to the
        single-PE entry when the rotation is identity (weight 1 on PE, 0 elsewhere).
        """
        if self.reproject_weights is not None:
            p = self.pe_displacement()  # (nx, ny, nz, T) ref-frame PE magnitude
            w = self.reproject_weights.to(p.dtype)  # (T, 3)
            return [(a, p * w[:, a]) for a in (0, 1, 2)]
        if self.dual:
            return [
                (self.a1, self._to_nifti(self.u_canon)),
                (self.a0, self._to_nifti(self.v_canon)),
            ]
        return [(self.pe_axis, self.pe_displacement())]

    def pe_displacements(self) -> list[tuple[str, int, torch.Tensor]]:
        """``(label, nifti_axis, signed_displacement)`` per encode axis, PRIMARY first.

        The dual-run counterpart of :meth:`pe_displacement`, and what the CLI writes as
        ``_flow_pe1`` / ``_flow_pe2``. Two SIGNED 4-D maps beat the
        ``_flowmag``/``_flowang`` pair here: the sign along each named physical axis is
        the quantity of interest (which way the partition wiggle went this frame), and a
        magnitude/angle pair buries that in a polar coordinate nobody wants to scrub.
        Magnitude/angle stay for the legacy 2-D slicewise dual case, where the two axes
        are two halves of one in-plane vector rather than two separate artifacts.
        """
        if not self.dual or self.pe_axis2 is None:
            return [("pe1", self.pe_axis, self.pe_displacement())]
        u = self._to_nifti(self.u_canon)
        v = self._to_nifti(self.v_canon)

        def pick(ax: int) -> torch.Tensor:
            return u if ax == self.a1 else v

        return [
            ("pe1", self.pe_axis, pick(self.pe_axis)),
            ("pe2", self.pe_axis2, pick(self.pe_axis2)),
        ]

    def coupling(self, mask: torch.Tensor | None = None) -> dict | None:
        """Measured relationship between the two fields, or None for a single-axis run.

        See :func:`dual_field_coupling` — a diagnostic only; no ratio between the two
        fields was assumed while solving them.
        """
        if not self.dual or self.pe_axis2 is None:
            return None
        comps = self.pe_displacements()
        return dual_field_coupling(comps[0][2], comps[1][2], mask)

    def flow_magnitude(self) -> torch.Tensor:
        """Per-frame displacement magnitude ``sqrt(u²+v²)`` ``(nx,ny,nz,T)`` (voxels)."""
        return self._to_nifti((self.u_canon**2 + self.v_canon**2).sqrt())

    def flow_angle(self) -> torch.Tensor:
        """Per-frame displacement direction in degrees [0,360) ``(nx,ny,nz,T)``.

        The dual-PE companion to the magnitude map (no single signed scalar can hold
        a 2-D vector): scrub the two together, or feed the direction to a movie.
        """
        ang = torch.atan2(self.v_canon, self.u_canon) * (180.0 / np.pi)
        return self._to_nifti(ang % 360.0)

    def corrected_series(self) -> torch.Tensor:
        """Motion-corrected 4-D series ``(nx, ny, nz, T)``.

        Rotation-aware runs precompute a genuine 3-D warp of the moco'd series (the
        correction has a through-plane component a per-slice 2-D warp can't hold), so
        return that directly; the idea-1 path maps its canonical 2-D-warp series back.
        """
        if self.corrected_nifti is not None:
            return self.corrected_nifti
        return self._to_nifti(self.corrected_canon)

    def flow_movie(
        self, max_mag: float | None = None, add_legend: bool = True, max_tile: int = 64
    ) -> np.ndarray:
        """Per-frame contact-sheet flow movie ``(T, Hs, Ws, 3)`` uint8.

        Each frame tiles all slices, colored by the circular-phase flow wheel;
        ``max_mag`` fixes the magnitude→saturation scaling (default: 99th pct of
        the whole run, so brightness is comparable across frames). Slices bigger
        than ``max_tile`` are block-averaged down first — the flow field is smooth,
        so this only shrinks the movie (and the color step) without hiding motion.
        """
        u = self.u_canon.numpy()
        v = self.v_canon.numpy()
        hh, ww = u.shape[2], u.shape[3]
        f = max(1, int(np.ceil(max(hh, ww) / max_tile)))
        if f > 1:
            # Block-mean downsample (crop to a multiple of f, then reshape-mean).
            ch, cw = (hh // f) * f, (ww // f) * f
            u = (
                u[:, :, :ch, :cw]
                .reshape(u.shape[0], u.shape[1], ch // f, f, cw // f, f)
                .mean((3, 5))
            )
            v = (
                v[:, :, :ch, :cw]
                .reshape(v.shape[0], v.shape[1], ch // f, f, cw // f, f)
                .mean((3, 5))
            )
        if max_mag is None:
            mag = np.sqrt(u * u + v * v)
            max_mag = float(np.percentile(mag, 99)) or 1.0
        nt = u.shape[0]
        frames = []
        for t in range(nt):
            # One vectorized color conversion for all slices of this frame.
            slabs = flow_to_rgb(u[t], v[t], max_mag)  # (nS, H, W, 3)
            sheet = _montage(slabs)
            if add_legend:
                wheel = _color_wheel(max(24, sheet.shape[0] // 12))
                wh = wheel.shape[0]
                sheet[:wh, :wh] = wheel
            frames.append(sheet)
        return np.stack(frames, 0)


def _temporal_coverage(data: np.ndarray, erode: int, device: torch.device) -> torch.Tensor:
    """Voxels holding real data in **every** frame, as a (nz, ny, nx) bool tensor.

    :func:`mask.data_coverage_mask` answers "did the scanner put a number here" for one
    volume. Over a series the question is per-frame and the answer must be the
    intersection: a voxel that is acquired in most frames and an exact zero in the rest
    has no business driving a displacement in any of them.

    That case is the norm, not an edge case, because locomoco's input has usually been
    through rigid motion correction, which resamples with zero fill. Each frame carries
    its own zero wedge at the FoV boundary, in a slightly different place. The time-MEAN
    is nonzero right across that band — it averages the frames that were covered — so an
    automask built from the mean happily includes it, and the flow then sees a voxel that
    is bright in one frame and exactly zero in the next. There is no displacement that
    explains that, but a shift estimator will invent an enormous one trying: on a 9.4T
    288-frame run this produced in-brain motion of 0.10 vox alongside a whole-volume max
    of 3621 vox, and poisoned the reference that the refine passes were rebuilt from.

    Intersecting per frame rather than testing ``min(t) != 0`` is deliberate: the min is
    only equivalent for non-negative data, and NORDIC/detrended series are signed, where
    a negative minimum would hide an exactly-zero frame. The loop also avoids
    materializing a full-size boolean copy of the series.
    """
    from .mask import _dilate_6conn

    cover = np.ones(data.shape[:3], dtype=bool)
    for t in range(data.shape[3]):
        v = data[..., t]
        np.logical_and(cover, np.isfinite(v) & (v != 0), out=cover)
    c = torch.from_numpy(cover).permute(2, 1, 0).contiguous().to(device)  # (nz, ny, nx)
    if bool(c.all()) or erode <= 0:
        return c  # full-FoV series: nothing to protect, and erosion would only eat brain
    # Peel by DILATING THE VOID rather than eroding the coverage. The two differ at the
    # volume boundary: mask._erode_6conn zero-pads, so it treats every face of the FoV as
    # a coverage edge and peels inward from all six — which on a thin-slab acquisition
    # (few slices) can consume the volume outright, and always throws away good voxels on
    # data that has no void near the boundary. Growing the void leaves the FoV edge alone.
    return ~(_dilate_6conn(~c, iterations=int(erode)) > 0)


def _dilate_inplane(m: torch.Tensor, iterations: int) -> torch.Tensor:
    """4-connected dilation within each ``(H, W)`` plane of a ``(nSlice, H, W)`` stack."""
    if iterations <= 0:
        return m
    kernel = torch.zeros(1, 1, 3, 3, device=m.device, dtype=torch.float32)
    kernel[0, 0, 1, 1] = kernel[0, 0, 0, 1] = kernel[0, 0, 2, 1] = 1
    kernel[0, 0, 1, 0] = kernel[0, 0, 1, 2] = 1
    x = m.float()[:, None]  # slices are the batch: no growth ACROSS planes
    for _ in range(iterations):
        x = (F.conv2d(x, kernel, padding=1) > 0.5).float()
    return x[:, 0]


def _blur_inplane(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur within each ``(H, W)`` plane of a ``(nSlice, H, W)`` stack."""
    if sigma <= 0:
        return m
    k = _gaussian_kernel1d(sigma, m.device, m.dtype)
    r = (k.numel() - 1) // 2
    x = m[:, None]
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return x[:, 0]


def _build_soft_mask(
    data: np.ndarray,
    slice_axis: int,
    a0: int,
    a1: int,
    dilate: int,
    sigma: float,
    device: torch.device,
    coverage_erode: int | None = 1,
    in_plane: bool = True,
) -> torch.Tensor:
    """Feathered gate in canonical ``(nSlice, H, W)`` layout, values in [0, 1].

    Two distinct restrictions, multiplied:

    * a 3dAutomask on the time-mean, dilated by ``dilate`` voxels (safety margin so
      real brain-edge motion survives) — "where is the head", which kills the wild
      displacements optical flow invents in the pure-noise air outside it;
    * the temporal data coverage (:func:`_temporal_coverage`) when ``coverage_erode``
      is not None — "where is there a number in every frame", which kills the ones it
      invents at a no-data boundary *inside* the automask.

    The whole thing is Gaussian-feathered by ``sigma`` so the flow decays smoothly to
    zero instead of at a hard edge. Coverage is eroded by a further ``ceil(sigma)``
    first: feathering a hard 1→0 step leaves the gate at ~0.5 *on* the boundary and
    non-trivial for ~sigma voxels into the void, so without the extra peel half the
    ramp would sit over voxels that have no data.

    ``in_plane`` (the slicewise estimators) does the dilation and the feather WITHIN
    each slice instead of in 3-D, and this is the difference between a usable gate and
    a broken one at the ends of the slab. The margin exists to protect brain-edge
    motion, which for a 2-D slicewise solve is an in-plane concept; letting it grow
    along the slice axis instead lets it *seed* mask on slices where 3dAutomask found
    no brain at all. Measured on a 9.4T 80-slice run: 3dAutomask claims 0 voxels on
    z=0 and z=1, and the 3-D dilate-by-4 handed the estimator 2075 and 2467 of them —
    slabs of pure ramp-in signal (slice mean 12 and 45 vs 208 mid-slab) with no
    correspondence to track. The flow went to 3621 voxels there and poisoned the
    reference; 1.86M of the 2.27M grossly displaced voxel-frames were on those two
    slices. A 3-D *feather* leaks the same way at reduced amplitude, which is still
    hundreds of voxels of displacement, so both have to be in-plane.
    """
    from .mask import automask

    def _canon(v):  # (nz, ny, nx) -> (nSlice, H, W)
        return v.permute(2, 1, 0).permute(slice_axis, a0, a1).contiguous()

    ref = torch.from_numpy(np.ascontiguousarray(data.mean(axis=3)))  # (nx, ny, nz)
    vol_zyx = ref.permute(2, 1, 0).contiguous().to(device).float()  # automask wants (nz,ny,nx)
    m = automask(vol_zyx, dilate_extra=0 if in_plane else dilate, device=device).float()
    if in_plane:
        m = _dilate_inplane(_canon(m), dilate)
    # Coverage is intersected AFTER the dilation, never before: the safety margin is
    # allowed to grow over brain the automask missed, but not back over voxels that
    # hold no data — dilating first and masking second would simply undo it.
    if coverage_erode is not None:
        peel = int(coverage_erode) + int(np.ceil(max(0.0, sigma)))
        cov = _temporal_coverage(data, peel, device).float()
        m = m * (_canon(cov) if in_plane else cov)
    if in_plane:
        return _blur_inplane(m, sigma).cpu()
    return _canon(_gaussian_blur3d(m, sigma)).cpu()


def _jacobian_det(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of the pull map ``(row,col) → (row+v, col+u)``.

    ``u`` (along W/col), ``v`` (along H/row) are canonical ``(T, nS, H, W)`` fields.
    ``J = (1 + ∂v/∂H)(1 + ∂u/∂W) − (∂v/∂W)(∂u/∂H)`` by central differences. Multiplying
    the corrected series by J conserves PE signal: stretched regions (J>1) brighten,
    compressed regions (J<1) dim. Clamped positive to keep folds from exploding.
    """

    def d_w(x: torch.Tensor) -> torch.Tensor:  # ∂/∂W (columns, last axis)
        d = torch.zeros_like(x)
        d[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
        return d

    def d_h(x: torch.Tensor) -> torch.Tensor:  # ∂/∂H (rows, second-to-last axis)
        d = torch.zeros_like(x)
        d[..., 1:-1, :] = 0.5 * (x[..., 2:, :] - x[..., :-2, :])
        return d

    j = (1.0 + d_h(v)) * (1.0 + d_w(u)) - d_w(v) * d_h(u)
    return j.clamp_min(0.1)


def _correct_pe(
    moving_raw: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    pe_flow_is_u: bool,
    dual: bool = False,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
) -> torch.Tensor:
    """Warp ``moving_raw`` ``(nt, H, W)`` by the estimated displacement (the warp we save).

    Single-PE keeps only the PE-axis component; ``dual`` (two PE axes) uses both.
    A single-PE correction is a 1-D shift along one in-plane axis, so ``lanczos`` routes it
    through the windowed-sinc gather along that axis; ``dual`` is a genuine 2-D warp and uses
    the tensor-product Lanczos kernel over both active axes.
    """
    if warp_interp == "lanczos" and not dual:
        pe_shift, ax = (u, 1) if pe_flow_is_u else (v, 0)  # u→W (axis 1), v→H (axis 0)
        return _shift1d_windowed_sinc(moving_raw, pe_shift, ax, radius=warp_radius)
    if dual:
        return _warp2d(moving_raw, u, v, warp_interp, warp_radius)
    uc = u if pe_flow_is_u else torch.zeros_like(u)
    vc = torch.zeros_like(v) if pe_flow_is_u else v
    return _warp2d(moving_raw, uc, vc, warp_interp, warp_radius)


def _estimate_static(
    vol,
    ref_mode,
    pe_flow_is_u,
    smooth_sigma,
    flow_fn,
    device,
    dual=False,
    warp_interp="bilinear",
    warp_radius=3,
    ref_override=None,
    first_n=None,
    diag=None,
    hpf_sigma=0.0,
    match="none",
    match_sigma=6.0,
    gate=None,
):
    """Fixed reference: batch the flow over ALL frames at once, looping slices.

    ``ref_override`` (``(nS, H, W)``) bypasses ``ref_mode`` — used by the outer
    reference-refinement loop, which re-registers against the corrected-data mean.
    ``first_n`` windows the reference aggregate to the early frames.

    ``diag`` (a dict) opts into the xcorr searchlight diagnostics: it is filled with
    ``conf`` ``(nt, nS, H, W)`` and, when ``diag["curve_frame"]`` is set, ``curve``
    ``(nd, nS, H, W)`` for that frame — both in canonical layout, empty for flow/phase.

    ``gate`` ``(nS, H, W)`` scales the flow down outside the head before the
    correction is resampled. Applying it here rather than to the returned series is
    what keeps the correction a single pass: gating afterwards means re-resampling
    every slice from the raw data with the gated field.
    """
    nt, ns, hh, ww = vol.shape
    win = _time_window(vol, 0, first_n)
    if ref_override is not None:
        ref = ref_override
    elif ref_mode == "mean":
        ref = win.mean(dim=0)
    elif ref_mode == "median":
        ref = win.median(dim=0).values
    elif ref_mode == "max":
        # Temporal max fills slices that later drop out of the FoV (head moved out)
        # and is a high-signal target; brighter/noisier per voxel than the mean.
        ref = win.max(dim=0).values
    elif ref_mode == "first":
        ref = vol[0]
    else:
        try:
            ref = vol[int(ref_mode)]
        except ValueError as e:
            raise ValueError(
                f"-ref must be mean|median|max|first|first_mean|first_median|<index>, got '{ref_mode}'"
            ) from e

    u_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    v_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    corrected = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    curve_frame = diag.get("curve_frame") if diag is not None else None
    conf_all = torch.zeros(nt, ns, hh, ww) if diag is not None else None
    curve_all: torch.Tensor | None = None  # (nd, nS, H, W), lazily sized once nd is known
    for s in tqdm(range(ns), desc="locomoco slices", unit="slice", leave=True, disable=ns < 2):
        moving_raw = vol[:, s].to(device).float()
        fixed_raw = ref[s].to(device).unsqueeze(0).expand(nt, hh, ww).contiguous().float()
        # Estimation source (optionally high-passed) is distinct from moving_raw, which
        # _correct_pe resamples — the correction keeps the raw intensities.
        est_fixed = _spatial_highpass(_match_prep2d(fixed_raw, match, match_sigma), hpf_sigma)
        est_moving = _spatial_highpass(_match_prep2d(moving_raw, match, match_sigma), hpf_sigma)
        fixed = _blur2d(est_fixed, smooth_sigma) if smooth_sigma > 0 else est_fixed
        moving = _blur2d(est_moving, smooth_sigma) if smooth_sigma > 0 else est_moving
        conf_acc: list[torch.Tensor] | None = [] if diag is not None else None
        curve_acc: list[torch.Tensor] | None = [] if curve_frame is not None else None
        u, v = flow_fn(fixed, moving, conf_out=conf_acc, curve_out=curve_acc)
        if gate is not None:
            gate_s = gate[s].to(device)
            u, v = u * gate_s, v * gate_s
        corrected[:, s] = _correct_pe(
            moving_raw, u, v, pe_flow_is_u, dual, warp_interp, warp_radius
        ).cpu()
        u_all[:, s] = u.cpu()
        v_all[:, s] = v.cpu()
        if conf_all is not None and conf_acc:  # xcorr populated it (flow/phase leave it empty)
            conf_all[:, s] = conf_acc[0].cpu()
        if curve_acc:  # per-offset (nt, H, W); keep only the target frame's landscape
            stack = torch.stack(curve_acc, 0)[:, curve_frame]  # (nd, H, W)
            if curve_all is None:
                curve_all = torch.zeros(stack.shape[0], ns, hh, ww)
            curve_all[:, s] = stack
    if diag is not None:
        diag["conf"] = (
            conf_all if (conf_all is not None and float(conf_all.abs().sum()) > 0) else None
        )
        diag["curve"] = curve_all
    return u_all, v_all, corrected


def _estimate_cumulative(
    vol,
    ref_mode,
    pe_flow_is_u,
    smooth_sigma,
    flow_fn,
    device,
    dual=False,
    warp_interp="bilinear",
    warp_radius=3,
    hpf_sigma=0.0,
    match="none",
    match_sigma=6.0,
):
    """Progressive reference: frame t registers to the running mean/median of the
    already-corrected frames 0..t-1 (frame 0 is the seed, zero warp). Sequential in
    time, so batched over SLICES per frame instead of over time.
    """
    nt, ns, hh, ww = vol.shape
    u_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    v_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    corrected = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)

    f0 = vol[0].to(device).float()  # (ns, hh, ww)
    corrected[0] = f0.cpu()
    running_sum = f0.clone()  # running Σ of corrected frames (for the mean)
    # The median needs every corrected frame; keep them on the GPU (same footprint
    # as the input). The mean only needs the running sum.
    buf = None
    if ref_mode == "first_median":
        buf = torch.zeros(nt, ns, hh, ww, dtype=torch.float32, device=device)
        buf[0] = f0

    for t in tqdm(range(1, nt), desc="locomoco frames", unit="frame", leave=True, disable=nt < 3):
        if ref_mode == "first_median":
            ref = buf[:t].median(dim=0).values
        else:  # first_mean
            ref = running_sum / t
        moving_raw = vol[t].to(device).float()
        est_fixed = _spatial_highpass(_match_prep2d(ref, match, match_sigma), hpf_sigma)
        est_moving = _spatial_highpass(_match_prep2d(moving_raw, match, match_sigma), hpf_sigma)
        fixed = _blur2d(est_fixed, smooth_sigma) if smooth_sigma > 0 else est_fixed
        moving = _blur2d(est_moving, smooth_sigma) if smooth_sigma > 0 else est_moving
        u, v = flow_fn(fixed, moving)
        corr = _correct_pe(moving_raw, u, v, pe_flow_is_u, dual, warp_interp, warp_radius)
        corrected[t] = corr.cpu()
        u_all[t] = u.cpu()
        v_all[t] = v.cpu()
        running_sum = running_sum + corr
        if buf is not None:
            buf[t] = corr
    return u_all, v_all, corrected


def _build_flow_fn(
    backend: str,
    *,
    pe_flow_is_u: bool,
    dual: bool,
    pe_only: bool,
    n_levels: int,
    n_iters: int,
    window_sigma: float,
    warp_interp: str,
    warp_radius: int,
    patch: int,
    stride: int,
    max_shift: float,
    trial_step: float,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
):
    """Build the per-pair ``(fixed, moving) -> (u, v)`` estimator for a backend.

    A single place mapping ``-backend`` to a closure over its tuning knobs, shared
    by the plain (idea-1) and rotation-aware (idea-2) orchestrators so they can never
    drift apart. Single-PE returns just the PE component; ``dual`` returns both. The
    closure optionally emits the xcorr searchlight quality (``conf_out``) and one-frame
    correlation landscape (``curve_out``) — no-ops for the flow / phase backends.
    """
    if backend == "flow":
        # dual (both axes) = full 2-D flow; single respects pe_only.
        pe_only_axis = None if dual else ((0 if pe_flow_is_u else 1) if pe_only else None)

        def flow_fn(fx, mv, **_unused) -> tuple[torch.Tensor, torch.Tensor]:
            return optical_flow_lk_2d(
                fx,
                mv,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
                pe_only_axis=pe_only_axis,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                max_shift=max_shift,
            )
    elif backend == "phase":

        def flow_fn(fx, mv, **_unused) -> tuple[torch.Tensor, torch.Tensor]:
            return phase_correlation_flow_2d(
                fx,
                mv,
                pe_is_u=pe_flow_is_u,
                patch=patch,
                stride=stride,
                max_shift=max_shift,
                n_iters=n_iters,
                dual=dual,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
            )
    elif backend == "xcorr":

        def flow_fn(fx, mv, conf_out=None, curve_out=None) -> tuple[torch.Tensor, torch.Tensor]:
            return xcorr_search_flow_2d(
                fx,
                mv,
                pe_is_u=pe_flow_is_u,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                dual=dual,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                conf_out=conf_out,
                curve_out=curve_out,
            )
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected flow | phase | xcorr.")
    return flow_fn


def estimate_residual_flow(
    data: np.ndarray,
    pe_axis: int,
    slice_axis: int,
    *,
    ref_mode: str = "mean",
    backend: str = "flow",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    pe_only: bool = True,
    dual: bool = False,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    patch: int = 16,
    stride: int = 8,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    refine_rounds: int = 0,
    converge: float = 0.0,
    converge_rel: float = 0.0,
    first_n: int | None = None,
    jacobian: bool = False,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    coverage_erode: int | None = 1,
    is_3dacq: bool = False,
    pe_axis2: int | None = None,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    save_corr_curve: int | None = None,
    hpf_sigma: float = 0.0,
    match: str = "none",
    match_sigma: float = 6.0,
    ngf_eta_q: float = 0.5,
    paired_bins: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> LocomocoResult:
    """Estimate per-frame residual PE-axis motion of a 4-D series, slicewise.

    ``data`` is ``(nx, ny, nz, T)`` (already motion-corrected). Slices are taken
    orthogonal to ``slice_axis`` (so PE lies in-plane); each slice's time course is
    a 2-D movie registered frame-by-frame to a reference. The WARP/correction/flow-
    map use only the ``pe_axis`` component (residual EPI motion is a PE-axis
    displacement, MEDIC-style).

    ``backend`` picks the per-frame estimator, all three measuring the same PE
    displacement by different routes:

    - ``flow`` (default): pyramidal Lucas-Kanade optical flow. ``pe_only`` (default)
      constrains it to 1 DOF along PE; False gives full 2-D flow (richer direction
      movie). ``n_levels`` / ``n_iters`` / ``window_sigma`` tune it.
    - ``phase``: phase-correlation searchlight — the shift read off the FFT phase
      ramp along PE over ``patch``×``patch`` squares (stride ``stride``), refined by
      ``n_iters`` warping passes. The clean "shift = phase vector" route.
    - ``xcorr``: magnitude cross-correlation searchlight — slide ``moving`` along PE
      over ``±max_shift`` and take the per-voxel offset of peak local (``window_sigma``
      Gaussian) correlation, sub-voxel by parabolic fit. Robust, single-shot.

    ``max_shift`` bounds the phase/xcorr search (residual motion is sub- to a few
    voxels).

    ``pe_axis2`` (3-D-acquired data only, i.e. with ``is_3dacq``) is the PARTITION /
    2nd phase-encode axis. Giving it solves the primary-PE and partition fields
    simultaneously and independently — see :func:`_run_3dacq_plain`. It is a different
    thing from ``dual`` below: ``dual`` is a 2-D multi-slice acquisition encoded on two
    in-plane axes, where the two components are two halves of one in-plane vector;
    ``pe_axis2`` is one 3-D acquisition carrying two physically distinct artifacts.

    ``dual`` (two PE axes, e.g. a 3-D-EPI acquisition phase-encoded on both in-plane
    axes) estimates BOTH in-plane components and warps/saves both — ``slice_axis``
    must be the third axis so both PE axes lie in the slice plane. flow does this
    natively (full 2-D), phase reads one phase-ramp per axis, xcorr searches the two
    axes separably (no O(n²) grid). The single signed PE map is then replaced by a
    magnitude + angle pair.

    ``ref_mode``: static ``mean`` | ``median`` | ``first`` | ``<int>``, or a
    PROGRESSIVE ``first_mean`` / ``first_median`` — frame t registers to the
    running mean/median of the already-corrected frames before it (a bootstrapped
    template; frame 0 is the seed). Progressive modes are sequential and slower;
    ``first_median`` also holds the corrected series on the GPU.

    Accuracy levers (trade time for exactness of the recovered value):
    ``warp_interp`` selects linear, cubic-convolution, or Lanczos resampling for both
    estimation and correction. Lanczos preserves the high-frequency structure that
    bilinear interpolation removes from later refine templates. ``refine_rounds`` re-registers against the
    corrected-data mean that many extra times, converging the reference template out
    of its bias. ``jacobian`` scales the corrected series by the PE Jacobian so signal
    is conserved (stretched regions dim, compressed regions brighten).

    ``hpf_sigma`` (voxels, experimental) spatially high-passes ONLY the frames fed to
    the estimator (``img − blur(img, hpf_sigma)``), keeping smooth non-motion
    intensity changes (drift, respiration B0) out of the flow while preserving the
    edges that encode the shift; the correction still resamples the raw series.

    ``match`` is the stronger form of the same idea and applies to the same
    estimation-only frames: ``localnorm`` divides out a local SCALE as well as a local
    mean, so a multiplicative gain change cancels. Reach for it when frame intensity
    varies over the run — a pre-steady-state ramp being the loud case, where T1
    saturation gives the first frames a gain that varies across TISSUE (measured
    0.94–1.39 on one 1.2mm run) and so survives any per-frame rescale. Only the
    brightness-constancy backends need it: ``xcorr`` normalizes inside its own
    correlation window and is already immune.

    ``automask`` (off here; on by the CLI) soft-gates the flow by a feathered brain
    mask — a dilated 3dAutomask of the time-mean, Gaussian-blurred by
    ``automask_sigma`` voxels — so optical flow's wild guesses in the pure-noise air
    outside the head fade to zero instead of showing up as huge displacements. The
    corrected series is rebuilt from the masked flow (so it stays consistent).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _validate_estimation_inputs(
        data.shape,
        pe_axis,
        slice_axis,
        is_3dacq=is_3dacq,
        max_shift=max_shift,
        trial_step=trial_step,
        window_sigma=window_sigma,
        verbose=verbose,
    )

    if is_3dacq:
        # 3-D acquisition: no per-slice fields, so estimate the whole 3-D PE field at
        # once (slice_axis is only a display hint for the movie/flow-map layout).
        return _run_3dacq_plain(
            data,
            pe_axis,
            slice_axis,
            ref_mode=ref_mode,
            backend=backend,
            smooth_sigma=smooth_sigma,
            n_levels=n_levels,
            n_iters=n_iters,
            window_sigma=window_sigma,
            max_shift=max_shift,
            trial_step=trial_step,
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            first_n=first_n,
            automask=automask,
            automask_dilate=automask_dilate,
            automask_sigma=automask_sigma,
            coverage_erode=coverage_erode,
            noshift_margin=noshift_margin,
            reg_sigma=reg_sigma,
            peak_mode=peak_mode,
            search_min_steps=search_min_steps,
            save_corr_curve=save_corr_curve,
            warp_interp=warp_interp,
            warp_radius=warp_radius,
            hpf_sigma=hpf_sigma,
            match=match,
            match_sigma=match_sigma,
            ngf_eta_q=ngf_eta_q,
            pe_axis2=pe_axis2,
            paired_bins=paired_bins,
            device=device,
            verbose=verbose,
        )

    if paired_bins is not None:
        raise ValueError(
            "the condition-paired reference is wired for the 3-D solve only "
            "(-is_3dacq, or two encode axes). The 2-D slicewise path builds its "
            "reference per slice and has not been converted."
        )

    orig_shape = (data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    in_plane = sorted(a for a in (0, 1, 2) if a != slice_axis)
    a0, a1 = in_plane  # a0 -> H (rows, v), a1 -> W (cols, u)
    pe_flow_is_u = pe_axis == a1

    perm = [3, slice_axis, a0, a1]
    vol = torch.from_numpy(np.ascontiguousarray(data)).permute(perm).contiguous()
    nt, ns, hh, ww = vol.shape
    if verbose:
        print(
            f"🌀 locomoco {_geometry_report(orig_shape, pe_axis, slice_axis, is_3dacq=False, dual=dual)}"
        )

    flow_fn = _build_flow_fn(
        backend,
        pe_flow_is_u=pe_flow_is_u,
        dual=dual,
        pe_only=pe_only,
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        warp_interp=warp_interp,
        warp_radius=warp_radius,
        patch=patch,
        stride=stride,
        max_shift=max_shift,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
    )

    # xcorr searchlight diagnostics collector (conf always; curve for one frame if asked).
    # Progressive references don't collect (rare mode) — diag stays None there.
    diag: dict | None = None
    if backend == "xcorr" and ref_mode not in ("first_mean", "first_median"):
        cf = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
        diag = {"curve_frame": cf}

    # The gate is built BEFORE the first estimate and re-applied after every one,
    # including inside the refine loop. Gating only the final field is too late: refine
    # rebuilds its next reference from the *corrected* series, so one pass's ungated
    # edge flow warps the template that the next pass registers everything against, and
    # the damage spreads inward from the no-data boundary into good brain. See
    # _temporal_coverage for the failure this fixes.
    soft = None
    if automask:
        soft = _build_soft_mask(
            data,
            slice_axis,
            a0,
            a1,
            automask_dilate,
            automask_sigma,
            device,
            coverage_erode=coverage_erode,
        )

    def _gate(u, v, corr):
        """Zero the flow outside the gate and rebuild the corrected series from it.

        Outside the head (and outside data coverage) the displacement becomes ~0, so
        the resample is the identity there and the voxels pass through untouched
        instead of being yanked by a phantom warp.

        Only the progressive-reference path needs this: `_estimate_static` takes the
        gate directly and applies it before it resamples, so the fixed-reference path
        never builds the corrected series twice.
        """
        if soft is None:
            return u, v, corr
        u, v = u * soft[None], v * soft[None]
        for s in range(ns):
            corr[:, s] = _correct_pe(
                vol[:, s].to(device).float(),
                u[:, s].to(device),
                v[:, s].to(device),
                pe_flow_is_u,
                dual,
                warp_interp,
                warp_radius,
            ).cpu()
        return u, v, corr

    progressive = ref_mode in ("first_mean", "first_median")
    if progressive:
        u_all, v_all, corrected = _estimate_cumulative(
            vol,
            ref_mode,
            pe_flow_is_u,
            smooth_sigma,
            flow_fn,
            device,
            dual,
            warp_interp,
            warp_radius,
            hpf_sigma=hpf_sigma,
            match=match,
            match_sigma=match_sigma,
        )
    else:
        u_all, v_all, corrected = _estimate_static(
            vol,
            ref_mode,
            pe_flow_is_u,
            smooth_sigma,
            flow_fn,
            device,
            dual,
            warp_interp,
            warp_radius,
            first_n=first_n,
            diag=diag,
            hpf_sigma=hpf_sigma,
            match=match,
            match_sigma=match_sigma,
            gate=soft,
        )
    if progressive:
        u_all, v_all, corrected = _gate(u_all, v_all, corrected)
    # Outer reference-refinement (shared engine): rebuild the reference from the
    # corrected series and re-register, converging the template out of its bias. The
    # aggregate honours -ref and -first_n; the step is the in-brain rms of the stacked
    # (u, v) change.
    if refine_rounds > 0:
        brain = _brain_mask_from(corrected.abs().mean(dim=0))  # (nS, H, W)

        def _est_2d(new_ref):
            u, v, corr = _estimate_static(
                vol,
                ref_mode,
                pe_flow_is_u,
                smooth_sigma,
                flow_fn,
                device,
                dual,
                warp_interp,
                warp_radius,
                ref_override=new_ref,
                diag=diag,
                hpf_sigma=hpf_sigma,
                match=match,
                match_sigma=match_sigma,
                gate=soft,
            )
            return (u, v), corr

        # Frames per step-rms chunk, sized so the difference stays around 64 MiB
        # however long the run is. Differencing the whole series at once, as the
        # step used to, costs a pair of full-size buffers to then keep only the
        # in-brain voxels; chunking is both smaller and measurably faster (1.6 s
        # -> 0.8 s at 360x65x112x104).
        rms_chunk = max(1, (64 * 1024**2) // max(1, ns * hh * ww * 4))

        def _brain_rms_2d(disp, prev):
            total, count = 0.0, 0
            for new_c, old_c in zip(disp, prev, strict=True):
                for t0 in range(0, new_c.shape[0], rms_chunk):
                    sl = slice(t0, t0 + rms_chunk)
                    d = (new_c[sl] - old_c[sl])[:, brain]
                    total += float(d.pow(2).sum())
                    count += d.numel()
            return (total / count) ** 0.5 if count else 0.0

        pair, corrected = _refine_loop(
            _est_2d,
            (u_all, v_all),
            corrected,
            reduce_ref=lambda c: _refine_reduce(c, ref_mode, 0, first_n),
            brain_rms=_brain_rms_2d,
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            max_shift=max_shift,
            verbose=verbose,
        )
        u_all, v_all = pair

    if jacobian:
        # Conserve signal along PE: where the unwarp stretches a region (Jacobian > 1)
        # scale intensity up, where it compresses (< 1) scale down. J = det(I + ∇disp)
        # over the in-plane (H, W) axes; single-PE reduces to 1 + ∂PE-disp/∂PE.
        corrected = corrected * _jacobian_det(u_all, v_all)

    result = LocomocoResult(
        u_canon=u_all,
        v_canon=v_all,
        corrected_canon=corrected,
        perm=perm,
        pe_flow_is_u=pe_flow_is_u,
        pe_axis=pe_axis,
        slice_axis=slice_axis,
        orig_shape=orig_shape,
        a0=a0,
        a1=a1,
        dual=dual,
    )
    # Surface the xcorr searchlight diagnostics (canonical → NIfTI layout, like the flow).
    if diag is not None:
        if diag.get("conf") is not None:
            result.confidence = result._to_nifti(diag["conf"])
        if diag.get("curve") is not None:
            result.corr_curve = result._to_nifti(diag["curve"])
            rr = float(max(1, int(np.ceil(max_shift))))
            result.corr_offsets = torch.arange(-rr, rr + 1e-6, trial_step)
    if verbose:
        # Restrict the median to the moving voxels so it stays meaningful: the
        # masked-out background is exact zero and the feather ramp is near-zero —
        # both would drag a whole-volume median to ~0 and hide the real motion.
        # Prefer the brain core (mask weight ~1); fall back to nonzero when unmasked.
        # dual: report vector magnitude; single: the PE component.
        ap = (u_all**2 + v_all**2).sqrt() if dual else (u_all if pe_flow_is_u else v_all).abs()
        if soft is not None:
            sel = ap[:, soft > 0.5]
            region = "in-brain"
        else:
            sel = ap[ap > 0]
            region = "nonzero"
        med = float(sel.median()) if sel.numel() else 0.0
        # Report the max over the SAME set as the median. A whole-volume max is
        # dominated by whatever the gate is about to zero anyway, so it read as a
        # catastrophe (3621 vox) on runs whose actual output was fine.
        mx = float(sel.abs().max()) if sel.numel() else 0.0
        axes = f"axes {a0},{a1}" if dual else f"axis {pe_axis}"
        print(
            f"🌀 locomoco: {nt} frames × {ns} slices, PE {axes}, ref={ref_mode} "
            f"({'2-D dual-PE' if dual else '1-D PE' if pe_only else '2-D'} flow); "
            f"|disp| median {med:.3f} vox ({region}), max {mx:.3f} vox ({region})"
        )
    return result


# ── rotation-aware residual motion (idea 2) ───────────────────────────────────
# The plain path above estimates residual PE motion in the moco'd (reference) frame
# and constrains it to the PE axis. That is only exact when the head did not rotate:
# rigid moco resamples every volume by R_t, so the physical distortion — fixed along
# SCANNER phase-encode — points along ``R_t · PE`` in the reference frame, not PE. A
# head tilt (nod) throws part of it out of the acquired slice plane, invisible to a
# 2-D slice movie. This path removes that assumption:
#
#   1. Estimate the DIFFERENTIAL distortion between neighbouring frames, with the
#      target left in its NATIVE orientation (PE genuinely axis-aligned, δR tiny so
#      2-D flow is valid), forward and backward, averaged on the reference grid.
#   2. Anchor to an ABSOLUTE per-frame estimate (each raw frame vs the reference
#      template, registered in the frame's native orientation) via a per-voxel
#      smoother, so the differential chain cannot drift (see the drift diagnostic).
#   3. Reproject the reference-frame PE magnitude by ``R_t`` — a per-frame, per-axis
#      scalar reweight — so the saved 5-D warp lives in the reference frame and
#      composes as ``moco-then-nonlinear`` in ffs_nwarp, exactly like the plain path
#      (to which it reduces when R_t = I). The IS/LR leakage of a tilt is SYNTHESISED
#      from the known rotation, not measured by the (2-D) flow.
#
# Inputs: the RAW (pre-moco) series for the native-frame estimation, the moco'd
# series for the anchor (reuses moco's existing resample — no extra interpolation),
# and the per-volume moco matrices (voxel + DICOM).


def detask_result(
    result,
    data: np.ndarray,
    design,
    polort: int,
    *,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    device: torch.device | None = None,
):
    """Remove the task-locked part of a field and RE-DERIVE the corrected series from it.

    Returns ``(cleaned_result, removed_components, note)``.  ``note`` is None when the
    corrected series was genuinely rebuilt, or a string explaining why it could not be
    (rotation-aware runs reproject the field through per-frame head rotations, so the
    correction is not a plain per-axis pull and re-deriving it here would be wrong).

    Cleaning goes through the NIfTI-order view and permutes back, rather than acting on
    ``u_canon`` directly: the canonical layout does not put time on a fixed axis across
    the slicewise and 3-D paths, and assuming it does silently multiplies the design
    against a spatial axis.  Writing the result back into the canonical fields means
    every derived product (``pe_displacements``, ``warp_components``, the PCs, the warp)
    inherits the cleaned field with no second code path to keep in sync.

    The resample is redone from the RAW input rather than adjusted from the existing
    corrected series: warping a warped series would stack a second interpolation, which
    is the trap :doc:`the interpolation-stacking audit </locomoco>` already caught once
    in the qwarp polish.
    """
    from dataclasses import replace

    from fastfuncstuff.stats.task_coupling import project_task_out

    def _clean(canon: torch.Tensor):
        nifti = result._to_nifti(canon)  # (nx, ny, nz, T) — time last, always
        keep, drop = project_task_out(nifti, design, polort)
        back = list(result.perm)
        return keep.permute(back).contiguous(), drop.permute(back).contiguous()

    (u_clean, u_task), (v_clean, v_task) = _clean(result.u_canon), _clean(result.v_canon)
    cleaned = replace(result, u_canon=u_clean, v_canon=v_clean)
    removed = replace(result, u_canon=u_task, v_canon=v_task).pe_displacements()

    if result.reproject_weights is not None:
        return (
            cleaned,
            removed,
            (
                "rotation-aware mode: the correction reprojects the field through each "
                "frame's head rotation, so the corrected series was NOT re-derived"
            ),
        )

    return (
        resample_from_raw(
            cleaned,
            data,
            warp_interp=warp_interp,
            warp_radius=warp_radius,
            device=device,
            desc="detask resample",
        ),
        removed,
        None,
    )


def pc_reconstruct_result(
    result,
    data: np.ndarray,
    keep: dict,
    n_pcs=None,
    *,
    rebuilt=None,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    device: torch.device | None = None,
):
    """Rebuild the field from a subset of its warp PCs and re-derive the corrected series.

    Returns ``(rebuilt_result, note)``; ``note`` is None on success or the reason the
    corrected series could not be re-derived.

    Writes back through the NIfTI-order view and permutes into the canonical fields,
    exactly as :func:`detask_result` does and for the same reason: the canonical layout
    does not put time on a fixed axis across the slicewise and 3-D paths, so acting on
    ``u_canon`` directly would silently mix a spatial axis into the reconstruction.
    Going through the canonical fields also means every derived product -- the warp,
    the PCs, the flow maps, the movie -- inherits the rebuilt field with no second code
    path to keep in sync.
    """
    from dataclasses import replace

    if rebuilt is None:
        comps = [(ax, f) for _, ax, f in result.pe_displacements()]
        rebuilt = warp_pc_reconstruct(comps, keep, n_pcs=n_pcs, device=device)
    if rebuilt is None:
        return result, "the warp is empty -- nothing to reconstruct"

    by_axis = dict(rebuilt)
    back = list(result.perm)

    def _to_canon(axis):
        return by_axis[axis].permute(back).contiguous()

    if result.dual and result.pe_axis2 is not None:
        u_axis = result.pe_axis if result.pe_axis == result.a1 else result.pe_axis2
        v_axis = result.pe_axis2 if u_axis == result.pe_axis else result.pe_axis
        out = replace(result, u_canon=_to_canon(u_axis), v_canon=_to_canon(v_axis))
    else:
        out = replace(result, u_canon=_to_canon(result.pe_axis))

    if result.reproject_weights is not None:
        return out, (
            "rotation-aware mode: the correction reprojects the field through each "
            "frame's head rotation, so the corrected series was NOT re-derived"
        )
    return (
        resample_from_raw(
            out,
            data,
            warp_interp=warp_interp,
            warp_radius=warp_radius,
            device=device,
            desc="pc recon resample",
        ),
        None,
    )


def resample_from_raw(
    result,
    data: np.ndarray,
    *,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    device: torch.device | None = None,
    desc: str = "resample",
):
    """Re-derive the corrected series from the RAW input using ``result``'s field.

    Two callers need this and for the same reason: the series the estimator produced
    is not the one to keep. ``-detask`` changed the field after the fact, and
    ``-detask filter`` estimated on a band-filtered copy of the data. Both must warp
    the untouched input, never adjust the series the estimator already warped —
    warping a warped series stacks a second interpolation, which is the trap the
    interpolation-stacking audit caught in the qwarp polish.
    """
    from dataclasses import replace

    if device is None:
        device = torch.device("cpu")
    comps = result.pe_displacements()
    axes = [ax for _, ax, _ in comps]
    fields = [f for _, _, f in comps]
    series = torch.from_numpy(np.ascontiguousarray(data)).float()
    corrected = torch.zeros_like(series)
    for t in tqdm(range(series.shape[3]), desc=desc, unit="frame", leave=True):
        corrected[..., t] = (
            _shift3d_axes(
                series[..., t].to(device)[None],
                [f[..., t].to(device)[None] for f in fields],
                axes,
                mode=warp_interp,
                radius=warp_radius,
            )[0]
            .cpu()
            .float()
        )
    return replace(result, corrected_nifti=corrected)


def _inv_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def compute_reproject_weights(
    matrices_dicom: torch.Tensor, affine: np.ndarray, pe_axis: int
) -> torch.Tensor:
    """Per-frame, per-axis weights that reproject the PE magnitude to reference space.

    The reference-frame displacement of a ``p``-voxel scanner-PE distortion is
    ``R_t⁻¹ · (p · e_pe)`` in mm (a rotation, so done in metric space, not voxels).
    Written per NIfTI axis in the voxel-pull units :func:`save_medic_warp` expects,
    that is ``p * w[t, a]`` with ``w[t, a] = (R_tᵀ · e_pe_dicom)[a] / ijk2dicom[a, a]``,
    where ``R_t`` is the DICOM rotation of the moco pull matrix (reference→raw) and
    ``e_pe_dicom`` the raw PE axis in DICOM mm. Reduces to ``w[:, pe]=1`` and ``0``
    elsewhere when ``R_t = I`` — i.e. exactly the plain single-PE warp.
    """
    from .nwarpforge import compute_cardinal_affine

    cardinal = compute_cardinal_affine(np.asarray(affine, dtype=np.float64))  # ijk→RAS
    dsign = np.array([-1.0, -1.0, 1.0])  # RAS→DICOM negates x, y
    ijk2dicom = cardinal[:3, :3] * dsign[:, None]  # ijk→DICOM linear part
    e_pe = ijk2dicom[:, pe_axis]  # (3,) raw PE direction in DICOM mm (incl. voxel size)
    diag = np.array([ijk2dicom[a, a] for a in range(3)])  # (3,)
    R = matrices_dicom[:, :3, :3].detach().cpu().numpy().astype(np.float64)  # (T,3,3)
    c = np.einsum("tji,j->ti", R, e_pe)  # R_tᵀ @ e_pe → ref-frame DICOM direction
    w = c / diag[None, :]
    return torch.from_numpy(w).float()


def pe_tilt_degrees(matrices_dicom: torch.Tensor, affine: np.ndarray, pe_axis: int) -> torch.Tensor:
    """Per-frame angle (deg) the reference PE axis is tilted by each head rotation.

    The distortion is off-axis exactly insofar as ``R_t`` rotates the PE unit vector
    away from itself: ``θ_t = arccos(êᵀ R_t ê)``. This is the "does idea 2 matter for
    this ``pe_dir`` and this subject" readout — rotation ABOUT PE leaves ``ê`` fixed
    (θ≈0), rotations perpendicular to PE tilt it. Axis-general: which physical
    rotations count depends on ``pe_axis``, and this measures the net effect directly.
    """
    from .nwarpforge import compute_cardinal_affine

    cardinal = compute_cardinal_affine(np.asarray(affine, dtype=np.float64))
    dsign = np.array([-1.0, -1.0, 1.0])
    ijk2dicom = cardinal[:3, :3] * dsign[:, None]
    e = ijk2dicom[:, pe_axis]
    e = e / np.linalg.norm(e)
    R = matrices_dicom[:, :3, :3].detach().cpu().numpy().astype(np.float64)
    cos = np.clip(np.einsum("i,tij,j->t", e, R, e), -1.0, 1.0)
    return torch.from_numpy(np.degrees(np.arccos(cos))).float()


def _fuse_tridiag(
    fd: torch.Tensor, anchor: torch.Tensor, w_anchor: float, w_diff: float = 1.0
) -> torch.Tensor:
    """Per-voxel least-squares fuse of a differential chain and absolute anchors.

    Minimise ``Σ_t w_diff (p_t − p_{t−1} − fd_t)² + Σ_t w_anchor (p_t − anchor_t)²``
    for each voxel — a symmetric tridiagonal system, identical across voxels (only
    the RHS differs), solved by a batched Thomas sweep. ``w_anchor`` sets how hard
    the absolute estimate pins the level and kills accumulated drift; ``fd_t`` is the
    forward difference into frame ``t`` (``fd[0]`` unused). ``fd``/``anchor`` are
    ``(T, ...)``; returns ``p`` ``(T, ...)``.
    """
    T = fd.shape[0]
    shape = fd.shape[1:]
    # The system is strictly diagonally dominant. Keep the large voxel fields
    # float32 on accelerators, while computing its shared Thomas coefficients
    # once in CPU float64. This avoids consumer-CUDA double throughput and makes
    # the rotation-aware path available on MPS without weakening the sensitive
    # scalar recurrence. CPU retains its full float64 reference path.
    work_dtype = torch.float64 if fd.device.type == "cpu" else torch.float32
    fdf = fd.reshape(T, -1).to(work_dtype)  # (T, N)
    af = anchor.reshape(T, -1).to(work_dtype)
    wa, wd = float(w_anchor), float(w_diff)

    # Constant tridiagonal: diag b_k = wa + wd*(#adjacent steps); off-diag = -wd.
    b64 = torch.full((T,), wa + 2.0 * wd, dtype=torch.float64)
    b64[0] = wa + wd
    b64[-1] = wa + wd
    # RHS: wa*anchor_k + wd*(fd_k − fd_{k+1}); boundaries drop the missing step term.
    rhs = wa * af
    rhs[:-1] = rhs[:-1] - wd * fdf[1:]  # − wd * fd_{k+1}
    rhs[1:] = rhs[1:] + wd * fdf[1:]  # + wd * fd_k

    # Thomas sweep (sub/super diagonals are the constant −wd).
    cp64 = torch.zeros(T, dtype=torch.float64)
    denom64 = torch.empty(T, dtype=torch.float64)
    denom64[0] = b64[0]
    cp64[0] = -wd / b64[0]
    for k in range(1, T):
        denom64[k] = b64[k] - (-wd) * cp64[k - 1]
        cp64[k] = -wd / denom64[k]
    cp = cp64.to(device=fd.device, dtype=work_dtype)
    denom = denom64.to(device=fd.device, dtype=work_dtype)

    dp = torch.zeros_like(rhs)
    dp[0] = rhs[0] / denom[0]
    for k in range(1, T):
        dp[k] = (rhs[k] - (-wd) * dp[k - 1]) / denom[k]
    p = torch.zeros_like(rhs)
    p[-1] = dp[-1]
    for k in range(T - 2, -1, -1):
        p[k] = dp[k] - cp[k] * p[k + 1]
    return p.reshape(T, *shape).float()


def _warp3d_pull(vol: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    """Sample ``vol`` (nx,ny,nz) at ``(x+dx, y+dy, z+dz)`` — a voxel-space pull warp."""
    nx, ny, nz = vol.shape
    xs, ys, zs = torch.meshgrid(
        torch.arange(nx, device=vol.device, dtype=vol.dtype),
        torch.arange(ny, device=vol.device, dtype=vol.dtype),
        torch.arange(nz, device=vol.device, dtype=vol.dtype),
        indexing="ij",
    )
    gx = 2.0 * (xs + dx) / max(nx - 1, 1) - 1.0
    gy = 2.0 * (ys + dy) / max(ny - 1, 1) - 1.0
    gz = 2.0 * (zs + dz) / max(nz - 1, 1) - 1.0
    # grid_sample wants (N,D,H,W,3) with the last-axis order (x_w, y_h, z_d); our
    # (nx,ny,nz) maps to (W=nx, H=ny, D=nz), so permute to (nz,ny,nx) and stack (gx,gy,gz).
    grid = torch.stack([gx, gy, gz], dim=-1).permute(2, 1, 0, 3).unsqueeze(0)
    out = F.grid_sample(
        vol.permute(2, 1, 0)[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0, 0].permute(2, 1, 0).contiguous()


def estimate_residual_flow_rotaware(
    raw_data: np.ndarray,
    moco_data: np.ndarray,
    matrices_vox: torch.Tensor,
    matrices_dicom: torch.Tensor,
    affine: np.ndarray,
    pe_axis: int,
    slice_axis: int,
    *,
    ref_mode: str = "max",
    backend: str = "flow",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    pe_only: bool = True,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    patch: int = 16,
    stride: int = 8,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    fuse: str = "auto",
    fuse_thresh: float = 0.05,
    fuse_weight: float = 1.0,
    first_n: int | None = None,
    is_3dacq: bool = False,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    coverage_erode: int | None = 1,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> LocomocoResult:
    """Rotation-aware residual PE motion: neighbour-differential + anchor, in ref space.

    ``raw_data`` (pre-moco) and ``moco_data`` (its motion-corrected series) are both
    ``(nx, ny, nz, T)`` on the same grid; ``matrices_vox``/``matrices_dicom`` are the
    per-volume moco pull matrices (reference→raw) in voxel and DICOM space. Returns a
    :class:`LocomocoResult` whose warp is a reference-frame 5-D field (mostly PE, with
    the rotation-synthesised IS/LR leakage), reducing to the plain path at zero rotation.

    ``fuse`` — ``auto`` (engage the anchor smoother only if the measured chain-vs-anchor
    drift exceeds ``fuse_thresh`` voxels), ``on`` (always), or ``off`` (chain only, the
    anchor just fixes the per-voxel DC offset). ``fuse_weight`` is the anchor weight
    when engaged.
    """
    from .affine import apply_affine

    # 3-D acquisition ignores the slice decomposition (slice_axis is only a display hint);
    # otherwise PE must lie in the slice plane to be visible to the 2-D flow.
    if is_3dacq and pe_axis == slice_axis:
        slice_axis = next(a for a in (0, 1, 2) if a != pe_axis)
    elif pe_axis == slice_axis:
        raise ValueError(f"-pe_axis ({pe_axis}) must differ from -slice_axis ({slice_axis}).")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx, ny, nz, nt = raw_data.shape
    _validate_estimation_inputs(
        raw_data.shape,
        pe_axis,
        slice_axis,
        is_3dacq=is_3dacq,
        max_shift=max_shift,
        trial_step=trial_step,
        window_sigma=window_sigma,
        verbose=verbose,
    )
    in_plane = sorted(a for a in (0, 1, 2) if a != slice_axis)
    a0, a1 = in_plane
    pe_flow_is_u = pe_axis == a1
    perm_sp = [slice_axis, a0, a1]  # (nx,ny,nz) → canonical (nS, H, W)
    inv_sp = _inv_perm(perm_sp)
    perm4 = [3, slice_axis, a0, a1]
    if verbose:
        print(
            f"🌀 locomoco {_geometry_report(raw_data.shape, pe_axis, slice_axis, is_3dacq=is_3dacq)}"
            "  (rotation-aware)"
        )

    Mv = matrices_vox.to(device=device, dtype=torch.float32)  # (T,4,4) ref→raw
    Mv_inv = torch.linalg.inv(Mv.double()).float()  # raw→ref

    flow_fn = _build_flow_fn(
        backend,
        pe_flow_is_u=pe_flow_is_u,
        dual=False,
        pe_only=pe_only,
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        warp_interp=warp_interp,
        warp_radius=warp_radius,
        patch=patch,
        stride=stride,
        max_shift=max_shift,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
    )

    raw = torch.from_numpy(np.ascontiguousarray(raw_data)).float()  # (nx,ny,nz,T) on CPU
    moco = torch.from_numpy(np.ascontiguousarray(moco_data)).float()
    out_shape = (nz, ny, nx)  # apply_affine works in (nz,ny,nx)

    def _to_src(v_xyz: torch.Tensor) -> torch.Tensor:
        return v_xyz.permute(2, 1, 0).contiguous()  # (nx,ny,nz)→(nz,ny,nx)

    def _from_src(v_zyx: torch.Tensor) -> torch.Tensor:
        return v_zyx.permute(2, 1, 0).contiguous()  # (nz,ny,nx)→(nx,ny,nz)

    # 3-D-acquired EPI: estimate the PE field on the whole volume (no slice movie), so
    # the same neighbour/anchor machinery below works unchanged — only _pe_flow differs.
    flow3d = (
        _build_flow3d_fn(
            backend,
            pe_axis,
            n_levels=n_levels,
            n_iters=n_iters,
            window_sigma=window_sigma,
            max_shift=max_shift,
            trial_step=trial_step,
            noshift_margin=noshift_margin,
            reg_sigma=reg_sigma,
            peak_mode=peak_mode,
            search_min_steps=search_min_steps,
        )
        if is_3dacq
        else None
    )

    def _pe_flow(fixed_xyz: torch.Tensor, moving_xyz: torch.Tensor) -> torch.Tensor:
        """PE-component flow (fixed vs moving), both (nx,ny,nz) → (nx,ny,nz) on ref/native grid."""
        if flow3d is not None:
            fx = _blur3d_b(fixed_xyz[None], smooth_sigma)[0] if smooth_sigma > 0 else fixed_xyz
            mv = _blur3d_b(moving_xyz[None], smooth_sigma)[0] if smooth_sigma > 0 else moving_xyz
            return flow3d(fx[None], mv[None])[0]
        fc = fixed_xyz.permute(perm_sp).contiguous()  # (nS,H,W)
        mc = moving_xyz.permute(perm_sp).contiguous()
        if smooth_sigma > 0:
            fc, mc = _blur2d(fc, smooth_sigma), _blur2d(mc, smooth_sigma)
        u, v = flow_fn(fc, mc)
        pe = u if pe_flow_is_u else v
        return pe.permute(inv_sp).contiguous()

    def _resample(v_xyz: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Resample a (nx,ny,nz) volume by a voxel matrix (base→source) onto the same grid."""
        return _from_src(apply_affine(_to_src(v_xyz), mat, output_shape=out_shape))

    # ── 1. bidirectional differential chain, accumulated on the reference grid ──
    g_ref = torch.zeros(nt, nx, ny, nz)  # forward differences D_t − D_{t-1}, ref grid
    for t in tqdm(
        range(1, nt), desc="locomoco neighbour", unit="frame", leave=True, disable=nt < 3
    ):
        rt = raw[..., t].to(device)
        rtm = raw[..., t - 1].to(device)
        # Both passes estimate the forward difference p_t − p_{t-1} (= −(s_t − s_{t-1}),
        # since the stored p = −distortion, matching the anchor/plain convention where
        # _pe_flow(fixed, moving) ≈ shift(fixed) − shift(moving)).
        # forward: target = frame t native; bring t-1 into t's frame (M_{t-1}∘M_t⁻¹).
        mv_fwd = _resample(rtm, Mv[t - 1] @ Mv_inv[t])
        gA = -_pe_flow(rt, mv_fwd)  # _pe_flow = s_t − s_{t-1} → negate; on t's grid
        # backward: target = frame t-1 native; bring t into (t-1)'s frame.
        mv_bwd = _resample(rt, Mv[t] @ Mv_inv[t - 1])
        gB = _pe_flow(rtm, mv_bwd)  # _pe_flow = s_{t-1} − s_t = p_t − p_{t-1}; on (t-1)'s grid
        gA_ref = _resample(gA, Mv[t])  # t-grid → ref grid
        gB_ref = _resample(gB, Mv[t - 1])  # (t-1)-grid → ref grid
        g_ref[t] = (0.5 * (gA_ref + gB_ref)).cpu()

    # ── 2. absolute anchor: each raw frame vs the reference template, native frame ──
    # Temporal MAX is the rotation-aware default: it fills slices that later frames
    # rotate/translate out of the FoV (blank in the mean, biasing the anchor) and is a
    # high-signal registration target — worth the brighter/noisier per-voxel estimate.
    ref_moco = _select_ref_vol(moco, ref_mode, first_n)
    templ_src = _to_src(ref_moco.to(device))
    anchor = torch.zeros(nt, nx, ny, nz)
    for t in tqdm(range(nt), desc="locomoco anchor", unit="frame", leave=True, disable=nt < 3):
        templ_t = _from_src(
            apply_affine(templ_src, Mv_inv[t], output_shape=out_shape)
        )  # ref→t frame
        a_t = _pe_flow(templ_t, raw[..., t].to(device))  # raw_t → template : D_t (on t grid)
        anchor[t] = _resample(a_t, Mv[t]).cpu()  # t-grid → ref grid

    # ── 3. drift diagnostic + fuse ──
    chain = torch.cumsum(g_ref, dim=0)  # p_chain, DC arbitrary
    chain = chain - chain.mean(0, keepdim=True) + anchor.mean(0, keepdim=True)  # match anchor DC
    resid = chain - anchor
    brain = ref_moco.abs().cpu() > (0.1 * float(ref_moco.abs().max()))
    sel = resid[:, brain] if brain.any() else resid
    drift_rms = float(sel.pow(2).mean().sqrt()) if sel.numel() else 0.0
    drift_max = float(sel.abs().max()) if sel.numel() else 0.0

    engage = fuse == "on" or (fuse == "auto" and drift_rms > fuse_thresh)
    if engage:
        p = _fuse_tridiag(g_ref, anchor, w_anchor=fuse_weight)
    else:
        p = chain  # chain, DC-pinned to the anchor
    p = p.permute(1, 2, 3, 0).contiguous()  # (nx,ny,nz,T)

    # ── automask soft-gate (same feathered brain mask as the plain path) ──
    if automask:
        soft = _build_soft_mask(
            moco_data,
            slice_axis,
            a0,
            a1,
            automask_dilate,
            automask_sigma,
            device,
            coverage_erode=coverage_erode,
        )
        soft_xyz = soft.permute(_inv_perm(perm_sp)).contiguous()  # canonical→(nx,ny,nz)
        p = p * soft_xyz[..., None]

    # ── reproject to reference-frame warp components + build corrected series ──
    weights = compute_reproject_weights(matrices_dicom, affine, pe_axis)  # (T,3)
    corrected = torch.zeros(nx, ny, nz, nt)
    for t in range(nt):
        pf = p[..., t].to(device)
        dcomp = [pf * float(weights[t, a]) for a in range(3)]  # voxel-pull disp per axis
        corrected[..., t] = _warp3d_pull(
            moco[..., t].to(device), dcomp[0], dcomp[1], dcomp[2]
        ).cpu()

    # Pack into a LocomocoResult: store p as the PE flow so the flow map / movie work;
    # warp_components() reprojects it via the weights; corrected is precomputed 3-D.
    p_canon = p.permute(perm4).contiguous()  # (T,nS,H,W)
    u_canon = p_canon if pe_flow_is_u else torch.zeros_like(p_canon)
    v_canon = torch.zeros_like(p_canon) if pe_flow_is_u else p_canon

    if verbose:
        tilt = pe_tilt_degrees(matrices_dicom, affine, pe_axis)
        print(
            f"🌀 locomoco rot-aware: {nt} frames, PE axis {pe_axis}, backend={backend}, "
            f"ref={ref_mode}"
        )
        print(
            f"   PE-axis tilt from head rotation: median {float(tilt.median()):.2f}° / "
            f"max {float(tilt.max()):.2f}° (≈0 ⇒ off-axis correction is negligible for this pe_dir)"
        )
        print(
            f"   drift(chain−anchor): rms {drift_rms:.3f} / max {drift_max:.3f} vox → "
            f"{'FUSED (anchor smoother)' if engage else 'chain (drift below threshold)'}"
        )

    return LocomocoResult(
        u_canon=u_canon,
        v_canon=v_canon,
        corrected_canon=torch.zeros_like(p_canon),  # unused; corrected_nifti takes over
        perm=perm4,
        pe_flow_is_u=pe_flow_is_u,
        pe_axis=pe_axis,
        slice_axis=slice_axis,
        orig_shape=(nx, ny, nz, nt),
        a0=a0,
        a1=a1,
        dual=False,
        reproject_weights=weights,
        corrected_nifti=corrected,
        drift_rms=drift_rms,
        drift_max=drift_max,
        fused=engage,
    )


# ── 3D-acquired EPI (idea 3: -is_3dacq) ───────────────────────────────────────
# 2-D multi-slice EPI acquires each slice at its own instant with its own field, so
# residual distortion MUST be estimated slice-by-slice (each slice an independent 2-D
# movie). 3-D EPI acquires the whole volume at once: one coherent field, smooth through
# the partition direction. So the slice decomposition isn't just unnecessary — it throws
# away real 3-D coupling. Estimate the single 3-D PE-displacement field directly, with
# 3-D pooling and through-plane regularisation. This also dissolves the "which of the two
# valid perpendicular cuts do I use" question: the 3-D solve uses all axes at once
# (strictly better than running the two cuts and averaging their 2-D marginals).


def _blur3d_b(vol: torch.Tensor, sigma: float, skip_axis: int | None = None) -> torch.Tensor:
    """Separable 3-D Gaussian blur of a batch of volumes ``(B, X, Y, Z)``.

    ``skip_axis`` leaves one axis unblurred — the slice axis of a 2-D multi-slice
    acquisition, where each slice is sampled at its own instant so pooling across
    slices would average over acquisition times rather than over signal.
    """
    if sigma <= 0:
        return vol
    k = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
    r = (k.numel() - 1) // 2
    x = vol.unsqueeze(1)
    pads = ((0, 0, 0, 0, r, r), (0, 0, r, r, 0, 0), (r, r, 0, 0, 0, 0))
    views = ((1, 1, -1, 1, 1), (1, 1, 1, -1, 1), (1, 1, 1, 1, -1))
    for ax in range(3):
        if ax == skip_axis:
            continue
        x = F.conv3d(F.pad(x, pads[ax], mode="replicate"), k.view(*views[ax]))
    return x.squeeze(1)


def _pyr_down3d(vol: torch.Tensor, skip_axis: int | None = None) -> torch.Tensor:
    """One pyramid step on ``(B, X, Y, Z)``, not downsampling ``skip_axis``."""
    ks = [2, 2, 2]
    if skip_axis is not None:
        ks[skip_axis] = 1
    return F.avg_pool3d(_blur3d_b(vol, 1.0, skip_axis).unsqueeze(1), tuple(ks)).squeeze(1)


def _pyr_min_extent(shape, skip_axis: int | None) -> int:
    """Smallest extent among the axes a pyramid step would actually shrink.

    The depth guard must ignore an un-downsampled slice axis: a 20-slice 2-D run has
    plenty of in-plane room to pyramid, and letting ``nz=20`` cap the depth would
    silently give a slicewise solve less motion reach than the equivalent 3-D one.
    """
    dims = [int(d) for i, d in enumerate(shape) if i != skip_axis]
    return min(dims) if dims else 0


def _spatial_highpass3d(
    vol: torch.Tensor, sigma: float, skip_axis: int | None = None
) -> torch.Tensor:
    """Unsharp 3-D spatial high-pass of a batch of volumes ``(B, X, Y, Z)``.

    The 3-D twin of :func:`_spatial_highpass` for the 3-D-acquisition paths — strips
    spatially-smooth non-motion intensity changes (drift, respiration B0) from the
    frames fed to the flow while keeping the edges that encode the shift. Estimation
    only; the correction still resamples the raw series. Experimental (``-hpf_spatial``).

    ``skip_axis`` keeps the high-pass in-plane for a 2-D multi-slice run, so a slice is
    never high-passed against its neighbours' (differently-timed) intensities.
    """
    if sigma <= 0:
        return vol
    return vol - _blur3d_b(vol, sigma, skip_axis)


MATCH_MODES = ("none", "meanstd", "localnorm", "gradmag", "ngf")


def _ngf_component(
    vol: torch.Tensor,
    axis: int,
    skip_axis: int | None = None,
    eta_frac: float = 0.5,
    eta_q: float = 0.5,
) -> torch.Tensor:
    """Unit-gradient component along ``axis`` — the encode-axis slice of the NGF vector.

    ``g_axis / sqrt(|grad|^2 + eta^2)``, the component of the normalised gradient field
    (Haber & Modersitzki) that a 1-DOF PE-axis solve can actually use.
    :func:`fastfuncstuff.processing.metrics.ngf_volume_cost` is the volume-to-volume
    COST built on the same quantity; LK needs an image, not a cost, so what goes
    through ``-match`` is the component itself.

    Why this and not ``localnorm`` for BOLD contamination. A task response is
    approximately a local multiplicative gain ``c``, and ``grad(cS) = c*grad(S) +
    S*grad(c)``. Inside a responding region ``grad(c) ~ 0``, so the gradient DIRECTION
    is untouched however large ``c`` is, and this transform is exactly invariant
    there. ``localnorm`` is invariant to a local mean and scale, which covers the same
    interior, but it still passes through the new EDGE that the boundary of the
    response creates, and that edge is what the estimator tracks.

    The honest limit: at that boundary ``S*grad(c)`` is not zero either, and it is
    proportional to the mean signal rather than to the anatomical contrast, so where
    anatomy is weak it can dominate the direction. Gradient orientation buys the
    region interior, not the boundary. No intensity-based transform buys the boundary
    — see the wiki note ``Frame brightness and brightness constancy``.

    Signed, unlike ``gradmag``: an edge keeps its polarity instead of folding both
    flanks into one ridge, so LK sees one zero crossing where gradmag shows two.

    ``eta_frac`` / ``eta_q`` set that floor as a fraction of the ``eta_q`` quantile of
    the volume's own squared gradient magnitude, and they are the knob that decides
    what counts as an edge at all. It matters more than it looks: normalisation
    promotes EVERY gradient to unit length, so a floor set from the median of a volume
    that is mostly low-gradient tissue hands noise-level structure the same weight LK
    gives a real boundary. Raising ``eta_q`` toward the upper decile restricts unit
    treatment to genuine edges and lets the rest fall away, which is the behaviour to
    reach for when ngf estimates come out noisier than localnorm rather than cleaner.
    """
    axes = [a for a in (0, 1, 2) if a != skip_axis]
    if axis not in axes:
        raise ValueError(
            f"-match ngf needs the gradient along encode axis {axis}, but that axis is "
            f"the excluded slice axis. Use -match localnorm for this acquisition."
        )
    b = _blur3d_b(vol, 1.0, skip_axis)
    sq = torch.zeros_like(b)
    grads = {}
    for a in axes:
        grads[a] = _grad_axis_3d(b, a)
        sq = sq + grads[a] * grads[a]
    # eta floors the denominator so flat, noisy tissue cannot contribute a
    # full-strength random orientation. Scaled from each volume's OWN gradient
    # magnitude: across TE the later echo's gradients are uniformly smaller, and a
    # shared constant would floor one echo to noise and leave the other unfloored.
    ref = torch.zeros(sq.shape[0], device=sq.device, dtype=sq.dtype)
    flat = sq.reshape(sq.shape[0], -1)
    for i in range(flat.shape[0]):
        v = flat[i]
        # Subsampled before the quantile: this runs per frame per pyramid level, and a
        # full sort of a 0.8mm volume is not worth an exact floor.
        if v.numel() > 1_000_000:
            v = v[:: v.numel() // 1_000_000 + 1]
        pos = v[v > 0]
        ref[i] = torch.quantile(pos, eta_q) if pos.numel() else flat.new_tensor(1.0)
    eta2 = (eta_frac**2) * ref.reshape(-1, 1, 1, 1)
    return grads[axis] / (sq + eta2).clamp(min=1e-12).sqrt()


def _match_prep(
    vol: torch.Tensor,
    mode: str,
    sigma: float = 2.0,
    skip_axis: int | None = None,
    pe_axis: int | None = None,
    ngf_eta_q: float = 0.5,
) -> torch.Tensor:
    """Make brightness constancy approximately true for a cross-contrast ``(B,X,Y,Z)`` pair.

    Same modes and semantics as :func:`fastfuncstuff.processing.optiwarp.prep_intensity`
    (written for optiwarp's ``(D,H,W)`` convention); this is the batched form locomoco's
    kernels take. LK's residual ``moving − fixed`` only encodes displacement when both
    images sit on one intensity scale. Across TE they do not — T2* decay dims the later
    echo everywhere, and that intensity step is gradient-shaped, so the Gauss-Newton
    solve reads it as displacement and runs away. Removing a LOCAL mean/scale (or
    dropping to gradient magnitude) takes the decay out and leaves the geometry.

    The same machinery serves the cross-TIME job (``-match``): a frame whose brightness
    drifts over the run violates constancy exactly as a later echo does.

    ``skip_axis`` leaves one axis out of the local neighbourhood (and, for ``gradmag``,
    out of the gradient), for a 2-D multi-slice acquisition where each slice is sampled
    at its own instant.
    """
    if mode == "none":
        return vol
    if mode == "ngf":
        if pe_axis is None:
            raise ValueError(
                "-match ngf needs the encode axis, which this estimation path does not "
                "pass through. It is wired for the 3-D solve; use -match localnorm on "
                "the 2-D slicewise path."
            )
        return _ngf_component(vol, pe_axis, skip_axis, eta_q=ngf_eta_q)
    if mode == "gradmag":
        b = _blur3d_b(vol, 1.0, skip_axis)
        axes = [a for a in (0, 1, 2) if a != skip_axis]
        sq = torch.zeros_like(b)
        for a in axes:
            g = _grad_axis_3d(b, a)
            sq = sq + g * g
        mag = sq.clamp(min=1e-12).sqrt()
        return _match_prep(mag, "localnorm", sigma, skip_axis)
    if mode == "meanstd":
        dims = (1, 2, 3)
        mu = vol.mean(dim=dims, keepdim=True)
        sd = vol.std(dim=dims, keepdim=True).clamp(min=1e-6)
        return (vol - mu) / sd
    if mode == "localnorm":
        mu = _blur3d_b(vol, sigma, skip_axis)
        var = _blur3d_b(vol * vol, sigma, skip_axis) - mu * mu
        return (vol - mu) / var.clamp(min=1e-12).sqrt()
    raise ValueError(f"unknown match mode {mode!r}; choose from {MATCH_MODES}")


def _sd_floor(sd: torch.Tensor, frac: float) -> torch.Tensor:
    """``frac`` x the 90th-percentile local sd, as a floor for a local-z-score divide.

    Subsampled before the quantile: ``torch.quantile`` refuses tensors past ~16M
    elements, and a whole slice's time course clears that on a long run.
    """
    flat = sd.reshape(-1)
    if flat.numel() > 1_000_000:
        flat = flat[:: flat.numel() // 1_000_000 + 1]
    return frac * torch.quantile(flat, 0.9)


def _match_prep2d(
    img: torch.Tensor, mode: str, sigma: float = 6.0, floor_frac: float = 0.1
) -> torch.Tensor:
    """2-D twin of :func:`_match_prep` for the slicewise ``(B, H, W)`` estimation path.

    Same modes; what it defends against differs. ``_match_prep`` matches two ECHOES
    whose contrast differs by construction. This matches FRAMES of one series whose
    intensity moves — the pre-steady-state ramp being the loud case: on a 1.2mm run
    here frame 0 was only 8% brighter overall, but its gain relative to steady state
    ran 0.94-1.39 ACROSS the brain (T1 saturation is tissue-dependent), so no per-frame
    rescale touches it. LK reads that gain as displacement; ``localnorm`` divides out a
    local scale as well as a local mean, which ``hpf_sigma`` (subtract only) cannot.

    Fully gain-invariant estimation already exists — ``-backend xcorr`` normalizes
    inside its correlation window — so this is for callers who want the LK backend.

    ``floor_frac`` floors the local sd before the divide. Without it the division
    rescales pure air noise to unit variance and the flow chases it: measured mean
    |v| in air 0.66 vs 0.40 voxels, with the in-brain result unchanged either way.
    """
    if mode == "none":
        return img
    if mode == "gradmag":
        b = _blur2d(img, 1.0)
        gx, gy = _spatial_gradients(b)
        mag = (gx * gx + gy * gy).clamp(min=1e-12).sqrt()
        return _match_prep2d(mag, "localnorm", sigma, floor_frac)
    if mode == "meanstd":
        dims = (1, 2)
        mu = img.mean(dim=dims, keepdim=True)
        sd = img.std(dim=dims, keepdim=True).clamp(min=1e-6)
        return (img - mu) / sd
    if mode == "localnorm":
        mu = _blur2d(img, sigma)
        sd = (_blur2d(img * img, sigma) - mu * mu).clamp(min=0).sqrt()
        return (img - mu) / sd.clamp(min=_sd_floor(sd, floor_frac))
    raise ValueError(f"unknown match mode {mode!r}; choose from {MATCH_MODES}")


def _grad_axis_3d(vol: torch.Tensor, axis: int) -> torch.Tensor:
    """Central-difference gradient of ``(B, X, Y, Z)`` along spatial ``axis`` (0/1/2)."""
    dim = axis + 1
    n = vol.shape[dim]
    d = torch.zeros_like(vol)
    if n >= 3:
        upper = vol.narrow(dim, 2, n - 2)
        lower = vol.narrow(dim, 0, n - 2)
        d.narrow(dim, 1, n - 2).copy_(0.5 * (upper - lower))
    return d


def _shift1d_windowed_sinc(vol: torch.Tensor, shift, axis: int, radius: int = 3) -> torch.Tensor:
    """Resample a batched volume (``(B,X,Y,Z)``) or slice stack (``(B,H,W)``) along ``axis``
    ONLY — ``axis`` is the 0-based spatial axis, batch is dim 0 — by a scalar or per-voxel
    ``shift`` with a Lanczos (windowed-sinc) kernel of half-width ``radius``: the high-fidelity
    resampler for sub-voxel PE-axis warps (single-PE correction is a 1-D shift along one axis).

    A pure axis shift is 1-D interpolation, so a windowed sinc along that ONE axis reaches
    wsinc/heptic-grade fidelity at ``2*radius`` taps — cheap next to a 3-D tensor-product sinc,
    and sinc-exact for a uniform shift as ``radius→∞``. Trilinear resampling low-pass-filters
    as it warps, blurring sub-voxel structure out of the CORRECTED output (and the refine
    template built from it); a windowed sinc preserves it. The correction and template are
    where this bites — the LK estimate itself is dominated by the pooling window, not the
    interpolator, so this is an output-FIDELITY tool, not a shift-accuracy one. The Lanczos
    window tames the Gibbs ringing / noise amplification a raw truncated sinc would add on
    thermal-noise data. Weights are DC-normalised so a flat region keeps its intensity (no
    brightness ripple). Sign matches :func:`_shift3d_axis` (a pull by ``s``: ``out[i] ≈
    vol[i + s]``); taps are border-clamped like its ``padding_mode="border"``.
    """
    dim = axis + 1  # batch is dim 0; spatial ``axis`` (0-based) → tensor dim. Any rank ≥ 2.
    needs_grad = torch.is_grad_enabled() and (
        vol.requires_grad or (isinstance(shift, torch.Tensor) and shift.requires_grad)
    )
    if (
        vol.device.type == "cuda"
        and vol.dtype == torch.float32
        and isinstance(shift, torch.Tensor)
        and not needs_grad
    ):
        fused = _try_fused(shift1d_lanczos_triton, vol, shift, dim, radius)
        if fused is not None:
            return fused
    n = vol.shape[dim]
    ishape = [1] * vol.ndim
    ishape[dim] = n
    idx = torch.arange(n, device=vol.device, dtype=vol.dtype).reshape(ishape)
    coord = (idx + shift).expand(vol.shape)  # scalar or (B,X,Y,Z) field → full grid
    base = torch.floor(coord)
    frac = coord - base
    base = base.to(torch.long)
    a = float(radius)
    num = torch.zeros_like(vol)
    den = torch.zeros_like(vol)
    # 2·radius taps span the Lanczos support [coord-a, coord+a]; the window is zero at ±a.
    for j in range(-(radius - 1), radius + 1):
        tap = (base + j).clamp_(0, n - 1)
        x = frac - j
        w = torch.sinc(x) * torch.sinc(x / a)  # Lanczos = sinc(x)·sinc(x/a)
        num = num + w * vol.gather(dim, tap)
        den = den + w
    return num / den


def _shift2d_high_order(
    vol: torch.Tensor,
    shift0: torch.Tensor,
    shift1: torch.Tensor,
    dim0: int,
    dim1: int,
    *,
    mode: str,
    radius: int = 3,
) -> torch.Tensor:
    """Portable tensor-product cubic/Lanczos over two active dimensions.

    This is the CPU/MPS and differentiable fallback for the fused CUDA kernel.
    It samples only the dimensions the displacement can move, irrespective of
    whether ``vol`` stores 2-D slices or 3-D EPI volumes.
    """
    if mode not in ("bicubic", "lanczos"):
        raise ValueError(f"unsupported 2-D high-order mode {mode!r}")
    vol = vol.contiguous()
    shape = vol.shape
    size0, size1 = shape[dim0], shape[dim1]
    stride0, stride1 = vol.stride(dim0), vol.stride(dim1)
    flat = vol.reshape(-1)
    idx = torch.arange(flat.numel(), device=vol.device)
    p0 = torch.div(idx, stride0, rounding_mode="floor") % size0
    p1 = torch.div(idx, stride1, rounding_mode="floor") % size1
    row = idx - p0 * stride0 - p1 * stride1
    c0 = p0.to(vol.dtype) + shift0.expand(shape).reshape(-1)
    c1 = p1.to(vol.dtype) + shift1.expand(shape).reshape(-1)
    b0, b1 = c0.floor().long(), c1.floor().long()
    f0, f1 = c0 - b0, c1 - b1

    if mode == "lanczos":
        offsets = range(-(radius - 1), radius + 1)

        def weights(frac, k):
            x = frac - k
            return torch.sinc(x) * torch.sinc(x / float(radius))

    else:
        offsets = range(-1, 3)

        def weights(frac, k):
            a = -0.75
            if k == -1:
                x = frac + 1.0
                return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
            if k == 0:
                x = frac
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            if k == 1:
                x = 1.0 - frac
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            x = 2.0 - frac
            return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

    w0 = [weights(f0, k) for k in offsets]
    w1 = [weights(f1, k) for k in offsets]
    acc = torch.zeros_like(flat)
    for j, k0 in enumerate(offsets):
        t0 = (b0 + k0).clamp(0, size0 - 1)
        for k, k1 in enumerate(offsets):
            t1 = (b1 + k1).clamp(0, size1 - 1)
            acc = acc + w0[j] * w1[k] * flat[(row + t0 * stride0 + t1 * stride1).long()]
    if mode == "lanczos":
        acc = acc / (torch.stack(w0).sum(0) * torch.stack(w1).sum(0))
    return acc.reshape(shape)


def _shift3d_axis(
    vol: torch.Tensor, shift, axis: int, mode: str = "bilinear", radius: int = 3
) -> torch.Tensor:
    """Sample ``(B,X,Y,Z)`` at ``coord + shift`` along ``axis`` ONLY (PE pull warp).

    ``shift`` is a scalar or a ``(B,X,Y,Z)`` field. grid_sample's 3-D mode is trilinear
    (``bilinear``) or nearest — no bicubic in 3-D — so a bicubic request degrades here.

    Scalar fast path: a uniform shift along one axis is *exactly* 1-D linear interpolation
    there (trilinear reduces to it — the other two axes sit on integer grid), so we skip
    building the full 3-D coordinate grid and the 3-D sampler. The xcorr searchlight applies
    ``nd`` scalar shifts per frame, so this is the hot per-offset op. Border-clamp + linear
    weight reproduce ``grid_sample(padding_mode="border", align_corners=True)`` bit-for-bit.
    """
    if mode == "lanczos":
        return _shift1d_windowed_sinc(vol, shift, axis, radius)

    if isinstance(shift, (int, float)) and mode != "nearest":
        import math

        dim = axis + 1
        n = vol.shape[dim]
        k = math.floor(shift)
        frac = float(shift) - k
        idx = torch.arange(n, device=vol.device)
        i0 = (idx + k).clamp(0, n - 1)
        v0 = vol.index_select(dim, i0)
        if frac == 0.0:
            return v0.contiguous()
        i1 = (idx + k + 1).clamp(0, n - 1)
        v1 = vol.index_select(dim, i1)
        return v0 * (1.0 - frac) + v1 * frac

    b, X, Y, Z = vol.shape
    dev, dt = vol.device, vol.dtype
    xs, ys, zs = torch.meshgrid(
        torch.arange(X, device=dev, dtype=dt),
        torch.arange(Y, device=dev, dtype=dt),
        torch.arange(Z, device=dev, dtype=dt),
        indexing="ij",
    )
    n = (X, Y, Z)[axis]
    comps = [
        (2.0 * xs / max(X - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
        (2.0 * ys / max(Y - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
        (2.0 * zs / max(Z - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
    ]
    comps[axis] = comps[axis] + shift * (2.0 / max(n - 1, 1))
    grid = torch.stack([comps[2], comps[1], comps[0]], dim=-1)  # (B,X,Y,Z,3): last = (W=Z,H=Y,D=X)
    if mode == "bicubic":
        mode = "bilinear"
    out = F.grid_sample(
        vol.unsqueeze(1), grid, mode=mode, padding_mode="border", align_corners=True
    )
    return out.squeeze(1)


def _shift3d_axes(
    vol: torch.Tensor,
    shifts: list[torch.Tensor],
    axes: list[int],
    mode: str = "bilinear",
    radius: int = 3,
) -> torch.Tensor:
    """Sample ``(B,X,Y,Z)`` at ``coord + shift`` along ONE OR TWO axes simultaneously.

    The multi-axis generalisation of :func:`_shift3d_axis`. A single axis delegates
    straight back to it, keeping the scalar fast path and the lanczos (windowed-sinc)
    resampler. Two axes are a genuine 2-D tensor-product interpolation evaluated in
    one pass; applying two 1-D gathers in sequence would resample twice and read the
    second displacement at the wrong location.
    """
    if len(axes) == 1:
        return _shift3d_axis(vol, shifts[0], axes[0], mode=mode, radius=radius)

    b, X, Y, Z = vol.shape
    dev, dt = vol.device, vol.dtype
    xs, ys, zs = torch.meshgrid(
        torch.arange(X, device=dev, dtype=dt),
        torch.arange(Y, device=dev, dtype=dt),
        torch.arange(Z, device=dev, dtype=dt),
        indexing="ij",
    )
    n = (X, Y, Z)
    comps = [
        (2.0 * xs / max(X - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
        (2.0 * ys / max(Y - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
        (2.0 * zs / max(Z - 1, 1) - 1.0)[None].expand(b, X, Y, Z),
    ]
    for ax, sh in zip(axes, shifts, strict=True):
        comps[ax] = comps[ax] + sh * (2.0 / max(n[ax] - 1, 1))
    grid = torch.stack([comps[2], comps[1], comps[0]], dim=-1)  # last = (W=Z, H=Y, D=X)
    needs_grad = torch.is_grad_enabled() and (
        vol.requires_grad or any(sh.requires_grad for sh in shifts)
    )
    if (
        vol.device.type == "cuda"
        and vol.dtype == torch.float32
        and mode in ("bicubic", "lanczos")
        and not needs_grad
    ):
        fused = _try_fused(
            shift2d_triton, vol, shifts[0], shifts[1], axes[0] + 1, axes[1] + 1, mode, radius
        )
        if fused is not None:
            return fused
    if mode in ("bicubic", "lanczos"):
        return _shift2d_high_order(
            vol,
            shifts[0],
            shifts[1],
            axes[0] + 1,
            axes[1] + 1,
            mode=mode,
            radius=radius,
        )
    out = F.grid_sample(
        vol.unsqueeze(1), grid, mode=mode, padding_mode="border", align_corners=True
    )
    return out.squeeze(1)


def _fourier_shifter(moving: torch.Tensor, axis: int, pad: int):
    """Build a scalar-shift function for ``(B,X,Y,Z)`` via the Fourier shift theorem.

    Sinc-exact sub-voxel resampling: unlike linear interpolation (a low-pass filter that
    inflates the local correlation at fractional shifts and makes the corr curve oscillate
    with a 1-voxel period), a global scalar shift is one phase ramp in k-space and does not
    blur. The forward FFT along ``axis`` is computed ONCE here; each returned ``shift(s)``
    is just a phase multiply + inverse FFT. ``pad`` replicate-pads the axis first so the
    FFT's circular wrap-around doesn't fold the opposite edge in (cropped back after). Sign
    matches :func:`_shift3d_axis` — a pull by ``s`` (``out[i] ≈ moving[i + s]``).
    """
    import math

    dim = axis + 1
    n = moving.shape[dim]
    if pad > 0:
        spec = [0, 0, 0, 0, 0, 0]  # F.pad order (Zl,Zr, Yl,Yr, Xl,Xr) for 4-D (B,X,Y,Z)
        spec[2 * (2 - axis)] = pad
        spec[2 * (2 - axis) + 1] = pad
        moving = F.pad(moving.unsqueeze(1), spec, mode="replicate").squeeze(1)
    npad = moving.shape[dim]
    k = torch.fft.fftfreq(npad, device=moving.device) * (2.0 * math.pi)
    shape = [1, 1, 1, 1]
    shape[dim] = npad
    k = k.reshape(shape)
    mov_fft = torch.fft.fft(moving, dim=dim)  # precomputed once

    def shift(s: float) -> torch.Tensor:
        out = torch.fft.ifft(mov_fft * torch.exp(1j * k * s), dim=dim).real.to(moving.dtype)
        return out.narrow(dim, pad, n).contiguous() if pad > 0 else out.contiguous()

    return shift


def optical_flow_lk_3d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pe_axis: int,
    *,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    reg: float = 1e-3,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
) -> torch.Tensor:
    """1-DOF (PE-axis) pyramidal Lucas-Kanade on 3-D volumes ``(B, X, Y, Z)``.

    The 3-D analogue of :func:`optical_flow_lk_2d` with ``pe_only``: only the PE-axis
    gradient enters, the pooling window is a 3-D Gaussian, and warping/pyramids are
    volumetric. ``warp_interp="lanczos"`` (half-width ``warp_radius``) resamples the
    iteration warp with a windowed sinc instead of trilinear — higher fidelity, though the
    estimate is set mostly by the pooling window (the interpolator matters far more for the
    correction/template than for the shift itself). Returns the PE-axis pull displacement
    ``(B, X, Y, Z)``.
    """
    fpyr, mpyr = [fixed], [moving]
    for _ in range(n_levels - 1):
        if min(fpyr[-1].shape[1:]) < 8:
            break
        fpyr.append(F.avg_pool3d(_blur3d_b(fpyr[-1], 1.0).unsqueeze(1), 2).squeeze(1))
        mpyr.append(F.avg_pool3d(_blur3d_b(mpyr[-1], 1.0).unsqueeze(1), 2).squeeze(1))

    disp = torch.zeros_like(fpyr[-1])
    for lvl in range(len(fpyr) - 1, -1, -1):
        fx, mv = fpyr[lvl], mpyr[lvl]
        if disp.shape[1:] != fx.shape[1:]:
            scale = fx.shape[pe_axis + 1] / disp.shape[pe_axis + 1]
            disp = (
                F.interpolate(
                    disp.unsqueeze(1),
                    size=tuple(fx.shape[1:]),
                    mode="trilinear",
                    align_corners=True,
                ).squeeze(1)
                * scale
            )
        for _ in range(n_iters):
            mw = _shift3d_axis(mv, disp, pe_axis, mode=warp_interp, radius=warp_radius)
            it = mw - fx
            ip = _grad_axis_3d(mw, pe_axis)
            step = -_blur3d_b(ip * it, window_sigma) / (_blur3d_b(ip * ip, window_sigma) + reg)
            disp = disp + step
    return disp


def xcorr_search_flow_3d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pe_axis: int,
    *,
    max_shift: float = 3.0,
    window_sigma: float = 2.0,
    trial_step: float = 0.5,
    eps: float = 1e-4,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    fourier_shift: bool = True,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    ambiguity_frac: float = 0.5,
    curve_out: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-D cross-correlation searchlight along PE — the "big block" xcorr for 3-D EPI.

    Slide the whole ``moving`` volume along the PE axis over ``±max_shift`` and take the
    per-voxel local correlation under a 3-D Gaussian searchlight (``window_sigma``).

    ``fourier_shift`` (default) resamples each trial shift with the sinc-exact Fourier
    shift theorem instead of linear interpolation: linear interp low-pass-filters the
    image at fractional shifts, spuriously inflating the local correlation there and
    making the corr curve oscillate with a 1-voxel period. The Fourier shift does not blur.

    ``peak_mode`` picks how the shift is read off the per-voxel curve:

    - ``"first_peak"`` (default): sweep offsets OUTWARD from zero and take the first real
      peak nearest zero (:func:`_first_peak_field_and_conf`) — no-shift-biased, immune to
      later oscillation humps, and the sweep stops once no voxel is still rising, so a run
      of mostly-small shifts searches ~±1 instead of ±max_shift (a data-adaptive speedup).
    - ``"argmax"``: the classic global-max + 5-point parabola over the full ``±max_shift``
      grid (:func:`_searchlight_field_and_conf`).

    Returns ``(field, conf)``. If ``curve_out`` is given the per-offset correlation is
    appended (``-r … +r`` order) — and the outward sweep runs the full range so the saved
    landscape is complete.
    """
    import math

    r = float(max(1, int(math.ceil(max_shift))))

    def _win(x: torch.Tensor) -> torch.Tensor:
        return _blur3d_b(x, window_sigma)

    _shifter = (
        _fourier_shifter(moving, pe_axis, pad=int(r))
        if fourier_shift
        else (lambda s: _shift3d_axis(moving, s, pe_axis))
    )
    mean_f = _win(fixed)
    var_f = (_win(fixed * fixed) - mean_f * mean_f).clamp_min(eps)

    def _ncc(s: float) -> torch.Tensor:
        mw = _shifter(s)
        mean_m = _win(mw)
        var_m = (_win(mw * mw) - mean_m * mean_m).clamp_min(eps)
        return (_win(fixed * mw) - mean_f * mean_m) / torch.sqrt(var_f * var_m)

    if peak_mode == "first_peak":
        return _sweep_first_peak(
            _ncc,
            r,
            trial_step,
            noshift_margin,
            reg_sigma,
            ambiguity_frac,
            lambda x: _blur3d_b(x, reg_sigma),
            curve_out,
            min_steps=search_min_steps,
        )

    offsets = torch.arange(-r, r + 1e-6, trial_step, device=fixed.device, dtype=fixed.dtype)
    nd = int(offsets.numel())
    z = torch.zeros_like(fixed)
    best_val = torch.full_like(fixed, float("-inf"))
    best_i = torch.zeros_like(fixed, dtype=torch.long)
    zero_val = z.clone()
    ym2, ym1, y0, yp1, yp2 = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    need = torch.zeros_like(fixed, dtype=torch.long)
    prev1: torch.Tensor | None = None
    prev2: torch.Tensor | None = None
    for i in range(nd):
        s = float(offsets[i])
        corr = _ncc(s)
        if curve_out is not None:
            curve_out.append(corr.detach().cpu())
        if abs(s) < trial_step * 0.5 + 1e-6:
            zero_val = corr
        yp1 = torch.where(need == 2, corr, yp1)
        yp2 = torch.where(need == 1, corr, yp2)
        need = torch.where(need > 0, need - 1, need)
        newp = corr > best_val
        ym2 = torch.where(newp, prev2 if prev2 is not None else corr, ym2)
        ym1 = torch.where(newp, prev1 if prev1 is not None else corr, ym1)
        y0 = torch.where(newp, corr, y0)
        best_val = torch.where(newp, corr, best_val)
        best_i = torch.where(newp, torch.full_like(best_i, i), best_i)
        need = torch.where(newp, torch.full_like(need, 2), need)
        prev2 = prev1
        prev1 = corr

    return _searchlight_field_and_conf(
        best_i,
        best_val,
        zero_val,
        ym2,
        ym1,
        y0,
        yp1,
        yp2,
        nd=nd,
        r=r,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        blur=lambda x: _blur3d_b(x, reg_sigma),
    )


def _build_flow3d_fn(
    backend: str,
    pe_axis: int,
    *,
    n_levels: int,
    n_iters: int,
    window_sigma: float,
    max_shift: float,
    trial_step: float,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
):
    """Build a 3-D ``(fixed, moving) -> disp`` PE estimator, ``(B,X,Y,Z)`` in and out.

    The returned ``f`` optionally appends the xcorr searchlight quality map to ``conf_out``
    and the per-offset correlation stack to ``curve_out`` (both no-ops for the flow
    backend, which has no per-voxel search) — so single-echo callers get the same
    confidence / correlation-landscape diagnostics as the multi-echo paths.
    """
    if backend == "flow":

        def f(fx, mv, **_unused) -> torch.Tensor:  # flow has no searchlight conf/curve
            return optical_flow_lk_3d(
                fx, mv, pe_axis, n_levels=n_levels, n_iters=n_iters, window_sigma=window_sigma
            )
    elif backend == "xcorr":

        def f(fx, mv, conf_out=None, curve_out=None) -> torch.Tensor:
            field, conf = xcorr_search_flow_3d(
                fx,
                mv,
                pe_axis,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                peak_mode=peak_mode,
                search_min_steps=search_min_steps,
                curve_out=curve_out,
            )
            if conf_out is not None:
                conf_out.append(conf)
            return field
    elif backend == "phase":
        raise ValueError(
            "phase backend has no 3-D (-is_3dacq) path yet; use -backend flow or xcorr."
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected flow | xcorr (phase: no 3-D path)."
        )
    return f


def _build_flow3d_axes_fn(
    backend: str,
    axes: list[int],
    alphas: torch.Tensor,
    *,
    n_levels: int,
    n_iters: int,
    window_sigma: float,
    max_shift: float,
    trial_step: float,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    xcorr_passes: int = 3,
    sep_floor: float = 1e-2,
    slicewise_axis: int | None = None,
):
    """Build a ``(fixed_list, moving_list) -> list[field]`` estimator over 1-2 axes.

    The axes-aware counterpart of :func:`_build_flow3d_fn`, dispatching to
    :func:`optical_flow_lk_3d_axes` / :func:`xcorr_search_flow_3d_axes`. One axis and one
    echo reproduces the single-axis builder; the returned ``f`` fills ``conf_out`` /
    ``curve_out`` / ``sep_out`` with whatever the chosen backend can supply.

    The flow backend takes a ``max_shift`` trust region for TWO axes but not for one.
    That asymmetry is deliberate and matches the existing 2-D pair: 1-DOF
    :func:`optical_flow_lk_3d` has no clamp, 2-DOF :func:`optical_flow_lk_2d` does. A
    coupled 2×2 inverse has a divergence mode the scalar update simply does not have —
    where ``sep`` sits near its floor the inverse amplifies, so the dual path needs the
    bound the 1-DOF path never did. Clamping the single-axis case here instead would
    silently change every existing ``-is_3dacq`` run.
    """
    if backend == "flow":

        def f(fixed_list, moving_list, conf_out=None, curve_out=None, sep_out=None):
            return optical_flow_lk_3d_axes(
                fixed_list,
                moving_list,
                axes,
                alphas,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                max_disp=max_shift if len(axes) == 2 else None,
                sep_floor=sep_floor,
                sep_out=sep_out,
                slicewise_axis=slicewise_axis,
            )
    elif backend == "xcorr":

        def f(fixed_list, moving_list, conf_out=None, curve_out=None, sep_out=None):
            fields, confs = xcorr_search_flow_3d_axes(
                fixed_list,
                moving_list,
                axes,
                alphas,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                peak_mode=peak_mode,
                search_min_steps=search_min_steps,
                n_passes=xcorr_passes,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                curve_out=curve_out,
                slicewise_axis=slicewise_axis,
            )
            if conf_out is not None:
                conf_out.extend(confs)
            return fields
    elif backend == "phase":
        raise ValueError("phase backend has no 3-D path yet; use -backend flow or xcorr.")
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected flow | xcorr (phase: no 3-D path)."
        )
    return f


def dual_field_coupling(
    d1: torch.Tensor,
    d2: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict:
    """Measure how related two independently-solved displacement fields turned out.

    ``d1`` / ``d2`` are ``(nx,ny,nz,T)`` signed displacements along the primary PE and
    partition axes. NOTHING here feeds back into the estimate — the two fields are solved
    without any assumed ratio precisely so that this measurement means something. If the
    primary-PE and partition wiggles share a physical source (one off-resonance field
    δ(r,t) seen through two different effective dwell times) the fields will come back
    proportional; if they are separate mechanisms they will not.

    ``mask`` is an optional ``(nx,ny,nz)`` bool restricting every statistic to brain —
    air voxels carry no displacement and would otherwise inflate every correlation toward
    whatever the gating left behind.

    Returns a dict with:

    - ``r`` — Pearson r over all in-mask voxels × frames. The headline number.
    - ``kappa`` — the free least-squares ratio ``d2 ≈ kappa·d1``, in voxels per voxel.
      Only interpretable alongside ``r``: a slope through an uncorrelated cloud is noise.
    - ``kappa_r2`` — variance of ``d2`` explained by ``kappa·d1``.
    - ``r_per_frame`` — ``(T,)`` spatial correlation frame by frame. A field pair that is
      coupled by physics is coupled in EVERY frame; a run-average r driven by a handful of
      big-motion frames is a different (and weaker) claim.
    - ``r_per_voxel`` — ``(nx,ny,nz)`` temporal correlation, so the coupling can be seen
      to be spatially structured (or not).
    - ``rms1`` / ``rms2`` — in-mask rms of each field (voxels), for scale.
    """
    if d1.shape != d2.shape:
        raise ValueError(f"fields must share a shape, got {tuple(d1.shape)} vs {tuple(d2.shape)}")
    m = (
        torch.ones(d1.shape[:3], dtype=torch.bool, device=d1.device)
        if mask is None
        else mask.to(d1.device).bool()
    )
    a = d1[m]  # (nvox, T)
    b = d2[m]

    def _pearson(x: torch.Tensor, y: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        xc = x - x.mean(dim=dim, keepdim=dim is not None)
        yc = y - y.mean(dim=dim, keepdim=dim is not None)
        num = (xc * yc).sum(dim=dim)
        den = (xc.pow(2).sum(dim=dim) * yc.pow(2).sum(dim=dim)).sqrt()
        return num / den.clamp_min(1e-12)

    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    denom = flat_a.pow(2).sum().clamp_min(1e-12)
    kappa = float((flat_a * flat_b).sum() / denom)
    resid = flat_b - kappa * flat_a
    var_b = (flat_b - flat_b.mean()).pow(2).sum().clamp_min(1e-12)
    return {
        "r": float(_pearson(flat_a, flat_b)),
        "kappa": kappa,
        "kappa_r2": float(1.0 - resid.pow(2).sum() / var_b),
        "r_per_frame": _pearson(a, b, dim=0).cpu(),  # (T,)
        "r_per_voxel": _scatter_masked(_pearson(a, b, dim=1), m),  # (nx,ny,nz)
        "rms1": float(a.pow(2).mean().sqrt()),
        "rms2": float(b.pow(2).mean().sqrt()),
    }


def _scatter_masked(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Put a per-in-mask-voxel vector back into a full ``mask``-shaped volume (0 outside)."""
    out = torch.zeros(mask.shape, dtype=vals.dtype, device=vals.device)
    out[mask] = vals
    return out.cpu()


def _time_window(series: torch.Tensor, dim: int, first_n: int | None) -> torch.Tensor:
    """Restrict ``series`` to its first ``first_n`` frames along ``dim`` (None = all).

    Lets a ``mean``/``max`` reference be built from just the early frames — the win for a
    run whose later frames drift (e.g. a slow time-stretch) that would otherwise pollute
    the aggregate, without giving up the SNR/FoV-fill of aggregating over many frames.
    """
    if first_n and first_n < series.shape[dim]:
        return series.narrow(dim, 0, max(1, first_n))
    return series


def _select_ref_vol(vol4d: torch.Tensor, ref_mode: str, first_n: int | None = None) -> torch.Tensor:
    """Reduce a ``(nx,ny,nz,T)`` series to a single reference volume per ``ref_mode``."""
    v = _time_window(vol4d, 3, first_n)
    if ref_mode in ("mean", "first_mean"):
        return v.mean(dim=3)
    if ref_mode in ("median", "first_median"):
        return v.median(dim=3).values
    if ref_mode == "max":
        return v.max(dim=3).values
    if ref_mode == "first":
        return v[..., 0]
    return vol4d[..., int(ref_mode)]  # explicit index ignores first_n


def _refine_reduce(
    corrected: torch.Tensor, ref_mode: str, dim: int, first_n: int | None = None
) -> torch.Tensor:
    """Aggregate the corrected series into a refine reference, HONOURING ``-ref``.

    ``mean`` / ``median`` / ``max`` are respected (so a ``max`` reference stays FoV-filled
    through refinement, not silently reverted to the mean); ``first`` / ``<index>`` /
    progressive fall back to the mean — a single frame is a poor, noisy refine template.
    ``first_n`` windows the aggregate to the early frames, same as the initial reference.
    """
    c = _time_window(corrected, dim, first_n)
    if ref_mode in ("median", "first_median"):
        return c.median(dim=dim).values
    if ref_mode == "max":
        return c.max(dim=dim).values
    return c.mean(dim=dim)


def _refine_reduce_label(ref_mode: str) -> str:
    """Human name for what :func:`_refine_reduce` will actually do with ``ref_mode``."""
    if ref_mode in ("median", "first_median"):
        return "median"
    if ref_mode == "max":
        return "max"
    return "mean"


def paired_templates(
    series: torch.Tensor, bin_of: torch.Tensor, ref_mode: str, dim: int = 3
) -> torch.Tensor:
    """One reference per task-state bin: ``(n_bins, ...)`` stacked along a new axis 0.

    The condition-paired reference. Each frame is later registered to the template of
    its OWN bin, so the BOLD response is common-mode within the pair and cancels — the
    estimator never sees the intensity change it would otherwise read as displacement.

    ``max`` falls back to the mean here. A per-bin max over ~15 frames is a noise
    envelope, not a template; the FoV-filling argument that justifies ``-ref max``
    globally does not survive being applied to a tenth of the frames.
    """
    n_bins = int(bin_of.max()) + 1
    outs = []
    for b in range(n_bins):
        sel = series.index_select(dim, torch.nonzero(bin_of == b, as_tuple=True)[0])
        if ref_mode in ("median", "first_median"):
            outs.append(sel.median(dim=dim).values)
        else:
            outs.append(sel.mean(dim=dim))
    return torch.stack(outs)


def _brain_mask_from(ref_mag: torch.Tensor, thresh_frac: float = 0.1) -> torch.Tensor:
    """Coarse brain mask (``|ref| > frac·max``) so the refine step size ignores air noise."""
    m = ref_mag > thresh_frac * float(ref_mag.max())
    return m if bool(m.any()) else torch.ones_like(ref_mag, dtype=torch.bool)


def _refine_converged(
    step: float, prev_step: float | None, converge: float, converge_rel: float
) -> str | None:
    """Should the refine loop stop after this pass? Return the reason, or None to continue.

    ``converge`` is an ABSOLUTE floor on the per-pass step (voxels). ``converge_rel`` is a
    RELATIVE floor: stop once a pass shrinks the step by less than this fraction of the
    previous pass — i.e. the improvement itself has plateaued ("changing by about the same
    amount each time"), even if the absolute step is still non-trivial. Either can fire.
    """
    if converge > 0 and step < converge:
        return f"Δ {step:.4f} < {converge} vox"
    if converge_rel > 0 and prev_step is not None and prev_step > 0:
        if (prev_step - step) / prev_step < converge_rel:
            return f"step shrank <{converge_rel * 100:.0f}%/pass ({prev_step:.4f}→{step:.4f})"
    return None


def _refine_loop(
    estimate,
    disp,
    corrected,
    *,
    reduce_ref,
    brain_rms,
    refine_rounds: int,
    converge: float,
    converge_rel: float,
    max_shift: float,
    verbose: bool,
):
    """Outer reference-refinement loop — shared by the 2-D, 3-D and multi-echo paths.

    The reference defines "zero motion", so a blurred/biased one biases every shift.
    After the first estimate, rebuild the reference from the corrected series (motion
    removed → sharp) via ``reduce_ref`` and re-``estimate`` against it, converging the
    template out of its bias.

    - ``estimate(new_ref) -> (disp, corrected)`` re-runs the per-frame estimator.
    - ``reduce_ref(corrected) -> new_ref`` aggregates the corrected series (honouring
      ``-ref`` / ``-first_n``) into the next reference.
    - ``brain_rms(disp, prev) -> float`` is the in-brain rms of the change between two
      iterates — the step size. The caller measures it rather than receiving a
      difference, because ``disp`` is opaque here (a ``(u, v)`` pair, a stacked field,
      a plain ``(…,T)`` one) and because differencing whole multi-GiB series to then
      keep only the in-brain voxels is the wrong order of operations.

    A pass whose step grows markedly (>1.5× the previous, or > ``max_shift`` outright) is
    compounding an over-warped reference: roll it back and stop. ``-converge`` /
    ``-converge_rel`` stop early once the step (or its improvement) plateaus.
    """
    if refine_rounds <= 0:
        return disp, corrected
    prev = disp
    prev_step: float | None = None
    for i in range(refine_rounds):
        saved = (disp, corrected)  # roll back to here if this pass diverges
        disp, corrected = estimate(reduce_ref(corrected))
        step = brain_rms(disp, prev)
        if verbose:
            print(f"   refine pass {i + 1}/{refine_rounds}: Δdisp rms {step:.4f} vox (in-brain)")
        if step > max_shift or (prev_step is not None and step > 1.5 * prev_step):
            disp, corrected = saved
            if verbose:
                print(
                    f"   ⚠ refine pass {i + 1} diverged (Δ {step:.4f} vox) — "
                    f"reverting to pass {i} and stopping."
                )
            break
        prev = disp
        reason = _refine_converged(step, prev_step, converge, converge_rel)
        prev_step = step
        if reason is not None:
            if verbose:
                print(f"   ✓ converged ({reason}) — stopping refine at pass {i + 1}.")
            break
    return disp, corrected


def _run_3dacq_plain(
    data: np.ndarray,
    pe_axis: int,
    display_slice: int,
    *,
    ref_mode: str,
    backend: str,
    smooth_sigma: float,
    n_levels: int,
    n_iters: int,
    window_sigma: float,
    max_shift: float,
    trial_step: float,
    refine_rounds: int,
    converge: float,
    converge_rel: float,
    first_n: int | None,
    automask: bool,
    automask_dilate: int,
    automask_sigma: float,
    coverage_erode: int | None = 1,
    device: torch.device,
    verbose: bool,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    save_corr_curve: int | None = None,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    hpf_sigma: float = 0.0,
    match: str = "none",
    match_sigma: float = 6.0,
    ngf_eta_q: float = 0.5,
    pe_axis2: int | None = None,
    xcorr_passes: int = 3,
    sep_floor: float = 1e-2,
    paired_bins: torch.Tensor | None = None,
) -> LocomocoResult:
    """Plain (moco-frame) residual motion for 3-D-acquired EPI: a single 3-D solve.

    ``pe_axis`` is the primary phase-encode axis. Passing ``pe_axis2`` (the partition /
    2nd phase-encode axis) solves BOTH simultaneously and independently — see
    :func:`optical_flow_lk_3d_axes` for why that is a joint 2x2 rather than two
    restricted solves, and :func:`dual_field_coupling` for the diagnostic that measures,
    after the fact, whether the two fields turned out related.

    ``paired_bins`` (a per-frame task-state bin id) switches on the condition-paired
    reference: one template per bin, each frame registered to its own. Refine rebuilds
    the templates per bin too, which is where the repeats pay off — each bin's average
    suppresses the residual field it was built from, so the templates get geometrically
    cleaner while staying matched in BOLD state.
    """
    nx, ny, nz, nt = data.shape
    dual = pe_axis2 is not None
    if dual and pe_axis2 == pe_axis:
        raise ValueError(f"the two encode axes must differ, got pe_axis=pe_axis2={pe_axis}")
    axes = [pe_axis] if not dual else [pe_axis, pe_axis2]
    if dual:
        # Both encode axes must lie in the display plane, so the un-encoded third axis is
        # the one we cut along — the same forcing the 2-D dual-PE path applies.
        disp_slice = next(a for a in (0, 1, 2) if a not in axes)
    else:
        disp_slice = (
            display_slice
            if display_slice != pe_axis
            else next(a for a in (0, 1, 2) if a != pe_axis)
        )
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    if verbose:
        print(
            f"🌀 locomoco "
            f"{_geometry_report(data.shape, pe_axis, disp_slice, is_3dacq=True, dual=dual)}"
        )

    vol4d = torch.from_numpy(np.ascontiguousarray(data)).float()
    # Single echo: no per-echo scaling to apply, so every axis gets alpha = 1.
    alphas = torch.ones(len(axes), 1)
    flow3d = _build_flow3d_axes_fn(
        backend,
        axes,
        alphas,
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        max_shift=max_shift,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        peak_mode=peak_mode,
        search_min_steps=search_min_steps,
        warp_interp=warp_interp,
        warp_radius=warp_radius,
        xcorr_passes=xcorr_passes,
        sep_floor=sep_floor,
    )

    # xcorr searchlight diagnostics, filled by the LAST estimate pass (see _refine_loop):
    # per-voxel confidence over the whole series, and the correlation landscape of one frame.
    xc = backend == "xcorr"
    conf_field = torch.zeros(nx, ny, nz, nt) if xc else None
    curve_frame = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
    curve_box: dict[str, torch.Tensor | None] = {"curve": None, "offsets": None}

    # Estimation prep: optional intensity match and spatial high-pass (raw kept for the
    # resample), then the existing estimation blur. Identity at the defaults. v is a
    # single (X,Y,Z) vol.
    def _prep(v: torch.Tensor) -> torch.Tensor:
        x = _spatial_highpass3d(
            _match_prep(v[None], match, match_sigma, pe_axis=pe_axis, ngf_eta_q=ngf_eta_q),
            hpf_sigma,
        )
        if smooth_sigma > 0:
            x = _blur3d_b(x, smooth_sigma)
        return x[0]

    # Dual runs also carry the per-voxel separability of the two axes (flow backend).
    sep_field = torch.zeros(nx, ny, nz, nt) if (dual and backend == "flow") else None

    def _estimate(ref_vol: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        # One reference, or one per task-state bin. The prep runs per TEMPLATE, not per
        # frame, so a paired run costs n_bins preps instead of 1 — not n_frames.
        if paired_bins is None:
            fxb_all = [_prep(ref_vol)]

            def _fixed(_t: int) -> torch.Tensor:
                return fxb_all[0]
        else:
            fxb_all = [_prep(ref_vol[b].to(device)) for b in range(ref_vol.shape[0])]

            def _fixed(t: int) -> torch.Tensor:
                return fxb_all[int(paired_bins[t])]

        disps = [torch.zeros(nx, ny, nz, nt) for _ in axes]
        corrected = torch.zeros(nx, ny, nz, nt)
        for t in tqdm(range(nt), desc="locomoco 3D", unit="frame", leave=True, disable=nt < 3):
            mv = vol4d[..., t].to(device)
            mvb = _prep(mv)
            fxb = _fixed(t)
            conf_acc: list[torch.Tensor] | None = [] if xc else None
            curve_acc: list[torch.Tensor] | None = [] if (xc and t == curve_frame) else None
            sep_acc: list[torch.Tensor] | None = [] if sep_field is not None else None
            d = flow3d(
                [fxb[None]], [mvb[None]], conf_out=conf_acc, curve_out=curve_acc, sep_out=sep_acc
            )
            corrected[..., t] = _shift3d_axes(
                mv[None], d, axes, mode=warp_interp, radius=warp_radius
            )[0].cpu()
            for k in range(len(axes)):
                disps[k][..., t] = d[k][0].cpu()
            if conf_field is not None and conf_acc:
                # Dual: a voxel is only as trustworthy as its WEAKER axis, so the two
                # searchlight qualities combine by min. `sep_map` is where the dual-only
                # ambiguity (which axis was indeterminate) is recorded.
                c = conf_acc[0][0]
                for extra in conf_acc[1:]:
                    c = torch.minimum(c, extra[0])
                conf_field[..., t] = c.cpu()
            if sep_acc:
                sep_field[..., t] = sep_acc[0][0].cpu()
            if curve_acc is not None:
                curve_box["curve"] = (
                    torch.stack(curve_acc, 0)[:, 0].permute(1, 2, 3, 0).contiguous()
                )
                rr = float(max(1, int(np.ceil(max_shift))))
                curve_box["offsets"] = torch.arange(-rr, rr + 1e-6, trial_step)
        return disps, corrected

    # Gate before the first refine pass, not after the last: refine rebuilds its next
    # reference from `corrected`, so ungated no-data flow would be baked into the
    # template every pass. Same reasoning as the 2-D path — see _temporal_coverage.
    soft_xyz = None
    if automask:
        soft = _build_soft_mask(
            data,
            disp_slice,
            a0,
            a1,
            automask_dilate,
            automask_sigma,
            device,
            coverage_erode=coverage_erode,
            in_plane=False,  # one 3-D solve: the margin/feather are 3-D concepts here
        )
        soft_xyz = soft.permute(_inv_perm([disp_slice, a0, a1])).contiguous()

    def _gate3d(d: list[torch.Tensor], corr):
        if soft_xyz is None:
            return d, corr
        d = [x * soft_xyz[..., None] for x in d]
        for t in range(nt):
            corr[..., t] = _shift3d_axes(
                vol4d[..., t].to(device)[None],
                [x[..., t].to(device)[None] for x in d],
                axes,
                mode=warp_interp,
                radius=warp_radius,
            )[0].cpu()
        return d, corr

    def _estimate_gated(ref_vol):
        return _gate3d(*_estimate(ref_vol))

    def _initial_ref() -> torch.Tensor:
        if paired_bins is None:
            return _select_ref_vol(vol4d, ref_mode, first_n).to(device)
        # -first_n is deliberately ignored for a paired reference: windowing to the early
        # frames would drop whole task states, and a bin is only as good as its repeats.
        return paired_templates(vol4d, paired_bins, ref_mode)

    disp, corrected = _estimate_gated(_initial_ref())
    # Reference-refinement (the -refine / -workhard / -superhard knob, in 3-D) via the
    # shared engine: rebuild the reference from the corrected series and re-register.
    # The aggregate honours -ref and -first_n; the step is the in-brain rms of the field.
    if refine_rounds > 0:
        brain = _brain_mask_from(corrected.abs().mean(dim=3))  # (nx, ny, nz)
        disp, corrected = _refine_loop(
            _estimate_gated,
            disp,
            corrected,
            reduce_ref=(
                (lambda c: _refine_reduce(c, ref_mode, 3, first_n).to(device))
                if paired_bins is None
                else (lambda c: paired_templates(c, paired_bins, ref_mode))
            ),
            # The step is the total move across BOTH axes, so a dual run converges only
            # when neither component is still shifting.
            brain_rms=lambda d, p: float(
                sum(float((a[brain] - b[brain]).pow(2).mean()) for a, b in zip(d, p, strict=True))
                ** 0.5
            ),
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            max_shift=max_shift,
            verbose=verbose,
        )

    # Canonical layout: u carries axis a1, v carries a0. Single-PE fills only the PE
    # component and leaves the other zero; dual fills both with real displacements.
    canon = [d.permute(perm4).contiguous() for d in disp]
    by_axis = dict(zip(axes, canon, strict=True))
    zero = torch.zeros_like(canon[0])
    u_canon = by_axis.get(a1, zero)
    v_canon = by_axis.get(a0, zero)
    if verbose:
        for k, ax in enumerate(axes):
            ap = disp[k].abs()
            sel = ap[ap > 0]
            med = float(sel.median()) if sel.numel() else 0.0
            label = "PE" if k == 0 else "partition"
            print(
                f"🌀 locomoco 3D-acq: {nt} frames, {label} axis {ax}, backend={backend}, "
                f"ref={ref_mode} (single 3-D solve, no slicing); |disp| median {med:.3f} vox, "
                f"max {float(ap.max()):.3f} vox"
            )
        if sep_field is not None:
            s = sep_field[sep_field > 0]
            if s.numel():
                print(
                    f"   axis separability: median {float(s.median()):.3f} "
                    f"(1 = axes cleanly separable, 0 = aperture-ambiguous)"
                )
    return LocomocoResult(
        u_canon=u_canon,
        v_canon=v_canon,
        corrected_canon=torch.zeros_like(canon[0]),
        perm=perm4,
        pe_flow_is_u=pe_flow_is_u,
        pe_axis=pe_axis,
        slice_axis=disp_slice,
        orig_shape=(nx, ny, nz, nt),
        a0=a0,
        a1=a1,
        dual=dual,
        pe_axis2=pe_axis2,
        sep_map=sep_field,
        corrected_nifti=corrected,
        confidence=conf_field,
        corr_curve=curve_box["curve"],
        corr_offsets=curve_box["offsets"],
    )


# ── multi-echo 3-D EPI (idea 4: -me_3depi) ────────────────────────────────────
# Multi-echo 3-D-EPI acquires the same volume at several echo times. The residual
# partition-direction (2nd phase-encode / slice) wiggle -is3dacq corrects is one
# shared dynamic off-resonance field, but its magnitude SCALES with echo time:
# echo e's displacement is ``alpha_e · w(r,t)`` — same spatial pattern, same time
# course, one scalar per echo (physically a global readout property, so the whole
# spatial pattern lives in the shared field w and the echo dependence is a single
# number → a rank-1 family across the echo axis).
#
# Estimating each echo alone and warping it by its own field re-breaks the "a voxel
# is the same voxel across echoes" rule (independent noise per echo). The joint
# solve keeps it: fit ONE w(r,t) pooled across echoes and correct echo e by
# ``alpha_e · w``, so every echo lands back on the same undistorted grid. Pooling
# also complements the echoes' strengths — early echoes are high-SNR but low-
# displacement (insensitive), late echoes low-SNR but high-displacement (sensitive)
# — the shared-field LK weights each echo by ``alpha_e² · gradient²`` automatically.
#
# ``alpha`` is LEARNED from the data (rank-1 factor of the per-echo estimates),
# then reported against the echo times: if alpha_e/TE_e is constant the scaling is
# linear as expected; a sign flip flags an alternating-blip scheme. -echo_times
# seeds it. Static differential distortion across echoes (later echoes more warped
# at rest) is a fieldmap/topup job and stays out of scope — this is the dynamic
# residual only.


def optical_flow_lk_3d_multiecho(
    fixed_list: list[torch.Tensor],
    moving_list: list[torch.Tensor],
    alpha: torch.Tensor,
    pe_axis: int,
    *,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    reg: float = 1e-3,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    match: str = "none",
    match_sigma: float = 2.0,
    ngf_eta_q: float = 0.5,
    max_disp: float | None = None,
) -> torch.Tensor:
    """Shared-field 1-DOF (PE-axis) pyramidal LK across echoes ``(B,X,Y,Z)``.

    ``fixed_list`` / ``moving_list`` are the E per-echo reference / moving volumes;
    ``alpha`` is the ``(E,)`` per-echo scaling. Solves for ONE displacement ``w`` such
    that echo ``e`` is warped by ``alpha_e·w``, pooling the Gauss-Newton normal
    equations over echoes: ``step = -Σ_e α_e·⟨∇·r⟩ / (Σ_e α_e²·⟨∇²⟩ + reg)`` under a
    Gaussian window. Reduces to :func:`optical_flow_lk_3d` for a single echo with
    ``alpha=[1]``. Returns the shared PE-axis pull displacement ``w`` ``(B,X,Y,Z)``.

    ``match`` (see :func:`_match_prep`) preprocesses both sides when fixed and moving
    are NOT the same contrast — mandatory for inter-echo pairs, pointless for the
    temporal modes where each echo is matched to its own template. ``max_disp`` bounds
    ``|w|`` after every Gauss-Newton step: LK has no built-in trust region, so a pair
    whose residual is dominated by contrast rather than geometry otherwise diverges to
    tens of voxels instead of returning a merely-wrong small number.
    """
    if match != "none":
        fixed_list = [
            _match_prep(f, match, match_sigma, pe_axis=pe_axis, ngf_eta_q=ngf_eta_q)
            for f in fixed_list
        ]
        moving_list = [
            _match_prep(m, match, match_sigma, pe_axis=pe_axis, ngf_eta_q=ngf_eta_q)
            for m in moving_list
        ]
    e = len(fixed_list)
    fpyr = [[f] for f in fixed_list]
    mpyr = [[m] for m in moving_list]
    for _ in range(n_levels - 1):
        if min(fpyr[0][-1].shape[1:]) < 8:
            break
        for j in range(e):
            fpyr[j].append(F.avg_pool3d(_blur3d_b(fpyr[j][-1], 1.0).unsqueeze(1), 2).squeeze(1))
            mpyr[j].append(F.avg_pool3d(_blur3d_b(mpyr[j][-1], 1.0).unsqueeze(1), 2).squeeze(1))
    nlev = len(fpyr[0])

    disp = torch.zeros_like(fpyr[0][-1])
    for lvl in range(nlev - 1, -1, -1):
        fx0 = fpyr[0][lvl]
        if disp.shape[1:] != fx0.shape[1:]:
            scale = fx0.shape[pe_axis + 1] / disp.shape[pe_axis + 1]
            disp = (
                F.interpolate(
                    disp.unsqueeze(1),
                    size=tuple(fx0.shape[1:]),
                    mode="trilinear",
                    align_corners=True,
                ).squeeze(1)
                * scale
            )
        for _ in range(n_iters):
            num = torch.zeros_like(disp)
            den = torch.zeros_like(disp)
            for j in range(e):
                a = float(alpha[j])
                mw = _shift3d_axis(
                    mpyr[j][lvl], a * disp, pe_axis, mode=warp_interp, radius=warp_radius
                )
                it = mw - fpyr[j][lvl]
                ip = _grad_axis_3d(mw, pe_axis)
                num = num + a * _blur3d_b(ip * it, window_sigma)
                den = den + (a * a) * _blur3d_b(ip * ip, window_sigma)
            disp = disp - num / (den + reg)
            if max_disp is not None:
                disp = disp.clamp(-max_disp, max_disp)
    return disp


# ── the unified axes × echoes solver ──────────────────────────────────────────
# Every 3-D-acquired locomoco case is one call to :func:`optical_flow_lk_3d_axes`
# with a different (axes, alphas) table — the seven cases in the CLI help are not
# seven algorithms, they are seven rows:
#
#   case                                   axes        alphas
#   3-D, primary PE                        [pe1]       [[1]]
#   3-D ME, primary PE (TE-independent)    [pe1]       [[1, 1, …, 1]]
#   3-D, partition, single echo            [pe2]       [[1]]
#   3-D ME, partition (TE-dependent)       [pe2]       [[TE_e/TE_1]]
#   3-D, primary + partition, single echo  [pe1, pe2]  [[1], [1]]
#   3-D ME, primary + partition            [pe1, pe2]  [[1, …], [TE_e/TE_1]]
#
# The two fields are solved SIMULTANEOUSLY and INDEPENDENTLY: no ratio between them
# is assumed, because the two artifacts plausibly have different physical sources
# (a dwell-time shift along the primary PE axis is TE-independent, while the
# partition-direction wiggle scales with TE). Whether the two recovered fields turn
# out correlated is a measurement this code is built to make, not an input to it.
#
# Identifiability differs sharply between the single- and multi-echo dual cases:
#
#   * MULTI-ECHO dual is well posed. The two axes carry DIFFERENT per-echo scaling
#     laws, so the echo axis itself separates them — the pooled normal equations
#     stay well conditioned even where the local image gradient is degenerate.
#   * SINGLE-ECHO dual has only geometry to lean on. The two components separate
#     wherever the pooling window contains edges of differing orientation, and
#     collapse into each other on a locally straight edge (the aperture problem).
#     That is not a bug to hide: `sep_out` returns the per-voxel separability so the
#     ambiguity is visible on a map instead of silently redistributed between axes.


def optical_flow_lk_3d_axes(
    fixed_list: list[torch.Tensor],
    moving_list: list[torch.Tensor],
    axes: list[int],
    alphas: torch.Tensor,
    *,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    reg: float = 1e-3,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    match: str = "none",
    match_sigma: float = 2.0,
    ngf_eta_q: float = 0.5,
    max_disp: float | None = None,
    sep_floor: float = 1e-2,
    sep_out: list[torch.Tensor] | None = None,
    slicewise_axis: int | None = None,
) -> list[torch.Tensor]:
    """Shared-field pyramidal LK over 1-2 encode axes and 1-N echoes, ``(B,X,Y,Z)``.

    Solves for one shared field ``w_k`` per entry of ``axes`` such that echo ``e`` is
    warped along axis ``k`` by ``alphas[k, e] · w_k``. ``alphas`` is ``(n_axes, E)``.
    Returns one ``(B,X,Y,Z)`` field per axis, in ``axes`` order.

    The Gauss-Newton normal equations are pooled over echoes AND coupled across axes::

        A[k,l] = Σ_e α_k,e·α_l,e·⟨g_k·g_l⟩        b[k] = -Σ_e α_k,e·⟨g_k·r⟩

    under a 3-D Gaussian window ``⟨·⟩``. One axis reduces exactly to the scalar update
    ``-Σ_e α_e⟨g·r⟩ / (Σ_e α_e²⟨g²⟩ + reg)``, i.e. to
    :func:`optical_flow_lk_3d_multiecho`, and at ``E=1, alpha=[[1]]`` to
    :func:`optical_flow_lk_3d`.

    Two axes solve the 2×2 directly rather than alternating two 1-DOF solves. That
    distinction is the whole point: two restricted solves each absorb the other axis's
    shift wherever the local edge is oblique, and averaging them does not undo it.

    Conditioning is handled in the NORMALISED coordinate ``sep = 1 - ⟨g1g2⟩²/(⟨g1²⟩⟨g2²⟩)``
    — the pooled Gram matrix's determinant divided by the product of its diagonal, so
    ``sep ∈ [0, 1]``: 1 where the two axes' gradients are orthogonal over the window
    (perfectly separable), 0 on a straight edge (fully ambiguous). Clamping ``sep`` at
    ``sep_floor`` bounds the step where the data cannot tell the axes apart, and is
    scale-free in a way a raw determinant floor is not (the determinant carries image
    contrast units, so any absolute floor would be a different constraint per dataset).
    ``sep_out``, if given, receives the map.

    Pooling over echoes needs no explicit SNR weighting: ``g_e`` and the residual both
    scale with echo amplitude ``s_e``, so the step is ``T + Σ s_e·N_e / Σ s_e²``, whose
    noise variance is ``σ²/Σ s_e²`` — already the inverse-variance-optimal combination
    under constant noise across echoes. Every echo with signal left lowers the variance.
    An explicit per-echo weight was implemented and measured: it changed nothing, and was
    removed rather than left as a dead knob.

    ``slicewise_axis`` makes the solve 2-D multi-slice: the pooling window, the pyramid
    blur and the pyramid downsampling all skip that axis, so each slice is solved from
    its own data alone. That is the right geometry when every slice is acquired at its
    own instant — pooling across slices would be averaging over acquisition times, not
    over signal. The echoes still pool normally, because all echoes of a given slice
    ARE sampled together: that is exactly the multi-echo win in a 2-D acquisition, more
    evidence per slice-time without smearing across slice-times.
    """
    if len(axes) not in (1, 2):
        raise ValueError(f"optical_flow_lk_3d_axes takes 1 or 2 axes, got {axes}")
    if len(axes) == 2 and axes[0] == axes[1]:
        raise ValueError(f"the two axes must differ, got {axes}")
    if slicewise_axis is not None and slicewise_axis in axes:
        raise ValueError(
            f"slicewise_axis ({slicewise_axis}) must differ from every encode axis "
            f"({axes}): the encode directions have to lie inside the slice plane."
        )
    if alphas.shape[0] != len(axes) or alphas.shape[1] != len(fixed_list):
        raise ValueError(
            f"alphas must be (n_axes, n_echoes) = ({len(axes)}, {len(fixed_list)}), "
            f"got {tuple(alphas.shape)}"
        )
    if match != "none":
        # The joint solve takes ONE image pair for both encode axes, so an
        # axis-dependent match (ngf) has to pick one: the primary encode axis, whose
        # displacement dominates and whose BOLD contamination is the reason the mode
        # exists. The partition solve then runs on a rendering oriented for its
        # neighbour -- acceptable, and the task-coupling diagnostic reports the two
        # axes separately so the cost of that choice is visible rather than assumed.
        fixed_list = [
            _match_prep(f, match, match_sigma, pe_axis=axes[0], ngf_eta_q=ngf_eta_q)
            for f in fixed_list
        ]
        moving_list = [
            _match_prep(m, match, match_sigma, pe_axis=axes[0], ngf_eta_q=ngf_eta_q)
            for m in moving_list
        ]

    n_ax, n_echo = len(axes), len(fixed_list)
    fpyr = [[f] for f in fixed_list]
    mpyr = [[m] for m in moving_list]
    for _ in range(n_levels - 1):
        if _pyr_min_extent(fpyr[0][-1].shape[1:], slicewise_axis) < 8:
            break
        for j in range(n_echo):
            fpyr[j].append(_pyr_down3d(fpyr[j][-1], slicewise_axis))
            mpyr[j].append(_pyr_down3d(mpyr[j][-1], slicewise_axis))
    nlev = len(fpyr[0])

    disp = [torch.zeros_like(fpyr[0][-1]) for _ in range(n_ax)]
    sep_map: torch.Tensor | None = None
    for lvl in range(nlev - 1, -1, -1):
        fx0 = fpyr[0][lvl]
        if disp[0].shape[1:] != fx0.shape[1:]:
            for k, ax in enumerate(axes):
                # Each component rescales by ITS OWN axis ratio, not a shared one: an
                # anisotropic pyramid step would otherwise skew the two components
                # relative to each other.
                scale = fx0.shape[ax + 1] / disp[k].shape[ax + 1]
                disp[k] = (
                    F.interpolate(
                        disp[k].unsqueeze(1),
                        size=tuple(fx0.shape[1:]),
                        mode="trilinear",
                        align_corners=True,
                    ).squeeze(1)
                    * scale
                )
        for _ in range(n_iters):
            a = [[torch.zeros_like(disp[0]) for _ in range(n_ax)] for _ in range(n_ax)]
            b = [torch.zeros_like(disp[0]) for _ in range(n_ax)]
            for j in range(n_echo):
                shifts = [float(alphas[k, j]) * disp[k] for k in range(n_ax)]
                mw = _shift3d_axes(mpyr[j][lvl], shifts, axes, mode=warp_interp, radius=warp_radius)
                it = mw - fpyr[j][lvl]
                g = [_grad_axis_3d(mw, ax) for ax in axes]
                for k in range(n_ax):
                    ak = float(alphas[k, j])
                    b[k] = b[k] - ak * _blur3d_b(g[k] * it, window_sigma, slicewise_axis)
                    for m in range(k, n_ax):
                        am = float(alphas[m, j])
                        a[k][m] = a[k][m] + (ak * am) * _blur3d_b(
                            g[k] * g[m], window_sigma, slicewise_axis
                        )
            if n_ax == 1:
                disp[0] = disp[0] + b[0] / (a[0][0] + reg)
            else:
                a11 = a[0][0] + reg
                a22 = a[1][1] + reg
                a12 = a[0][1]
                prod = a11 * a22
                sep = (1.0 - (a12 * a12) / prod).clamp(0.0, 1.0)
                sep_map = sep
                det = prod * sep.clamp_min(sep_floor)
                disp[0] = disp[0] + (a22 * b[0] - a12 * b[1]) / det
                disp[1] = disp[1] + (a11 * b[1] - a12 * b[0]) / det
            if max_disp is not None:
                for k in range(n_ax):
                    disp[k] = disp[k].clamp(-max_disp, max_disp)
    if sep_out is not None and sep_map is not None:
        sep_out.append(sep_map)
    return disp


def xcorr_search_flow_3d_multiecho(
    fixed_list: list[torch.Tensor],
    moving_list: list[torch.Tensor],
    alpha: torch.Tensor,
    pe_axis: int,
    *,
    max_shift: float = 3.0,
    window_sigma: float = 2.0,
    trial_step: float = 0.5,
    weights: list[torch.Tensor] | None = None,
    eps: float = 1e-4,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    fourier_shift: bool = True,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    ambiguity_frac: float = 0.5,
    curve_out: list[torch.Tensor] | None = None,
    slicewise_axis: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared-parameter 3-D xcorr searchlight across echoes — TE-linearity enforced.

    ONE displacement is searched for all echoes at once under the hard constraint
    ``disp_e = alpha_e · w``. The search variable ``s`` is the shift of the largest-alpha
    echo ``m`` (biggest, most detectable displacement, and ``s`` stays within
    ``max_shift``); echo ``e`` is trial-shifted by ``(alpha_e/alpha_m)·s`` and the
    SNR-weighted sum of the per-echo local normalised correlations is maximised per voxel
    (streaming running-peak, sub-voxel 5-point parabola — the single-echo
    :func:`xcorr_search_flow_3d` machinery, pooled). This is the searchlight's robustness
    with every echo's SNR folded into one informed search, rather than searching each echo
    alone and averaging the answers.

    ``weights`` are per-echo pooling weights ``(B,X,Y,Z)`` (default: each echo's local
    signal energy ``var_f`` — a per-voxel SNR proxy that fades out dropout regions in the
    late echoes). Returns ``(w, conf)``: the ECHO-1-scaled shared displacement
    ``w = s/alpha_m`` ``(B,X,Y,Z)`` (so ``alpha_e·w`` recovers each echo's shift, matching
    the joint path) and the pooled searchlight quality map. ``noshift_margin`` /
    ``reg_sigma`` are the robustness knobs — see :func:`_searchlight_field_and_conf`.
    If ``curve_out`` is given, the pooled correlation at every trial offset (the search
    variable ``s`` = echo-``m`` shift) is appended to it (``nd`` tensors, offset order).
    """
    import math

    e = len(fixed_list)
    m = int(torch.argmax(alpha))
    ratio = [float(alpha[j] / alpha[m]) for j in range(e)]  # ≤ 1
    r = float(max(1, int(math.ceil(max_shift))))

    def _win(x: torch.Tensor) -> torch.Tensor:
        # slicewise: the searchlight is a 2-D disc in the slice plane, never a 3-D ball
        return _blur3d_b(x, window_sigma, slicewise_axis)

    # One sinc-exact shifter per echo (forward FFT precomputed once); echo j is shifted by
    # ratio[j]·s at each offset. See :func:`_fourier_shifter` / xcorr_search_flow_3d.
    shifters = [
        _fourier_shifter(moving_list[j], pe_axis, pad=int(r)) if fourier_shift else None
        for j in range(e)
    ]
    mean_f = [_win(f) for f in fixed_list]
    var_f = [
        (_win(f * f) - mf * mf).clamp_min(eps) for f, mf in zip(fixed_list, mean_f, strict=True)
    ]
    wgt = weights if weights is not None else var_f
    wsum = torch.stack(wgt, 0).sum(0).clamp_min(eps)

    def _pooled_ncc(s: float) -> torch.Tensor:
        pooled = torch.zeros_like(fixed_list[0])
        for j in range(e):
            sj = ratio[j] * s
            sh = shifters[j]
            mw = sh(sj) if sh is not None else _shift3d_axis(moving_list[j], sj, pe_axis)
            mean_m = _win(mw)
            var_m = (_win(mw * mw) - mean_m * mean_m).clamp_min(eps)
            corr = (_win(fixed_list[j] * mw) - mean_f[j] * mean_m) / torch.sqrt(var_f[j] * var_m)
            pooled = pooled + wgt[j] * corr
        return pooled / wsum

    if peak_mode == "first_peak":
        s_field, conf = _sweep_first_peak(
            _pooled_ncc,
            r,
            trial_step,
            noshift_margin,
            reg_sigma,
            ambiguity_frac,
            lambda x: _blur3d_b(x, reg_sigma, slicewise_axis),
            curve_out,
            min_steps=search_min_steps,
        )
        return s_field / float(alpha[m]), conf  # echo-1 scale (alpha_1 = 1)

    offsets = torch.arange(
        -r, r + 1e-6, trial_step, device=fixed_list[0].device, dtype=fixed_list[0].dtype
    )
    nd = int(offsets.numel())
    z = torch.zeros_like(fixed_list[0])
    best_val = torch.full_like(z, float("-inf"))
    best_i = torch.zeros_like(z, dtype=torch.long)
    zero_val = z.clone()
    ym2, ym1, y0, yp1, yp2 = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    need = torch.zeros_like(z, dtype=torch.long)
    prev1: torch.Tensor | None = None
    prev2: torch.Tensor | None = None
    for i in range(nd):
        s = float(offsets[i])
        pooled = _pooled_ncc(s)
        if curve_out is not None:
            curve_out.append(pooled.detach().cpu())
        if abs(s) < trial_step * 0.5 + 1e-6:
            zero_val = pooled
        yp1 = torch.where(need == 2, pooled, yp1)
        yp2 = torch.where(need == 1, pooled, yp2)
        need = torch.where(need > 0, need - 1, need)
        newp = pooled > best_val
        ym2 = torch.where(newp, prev2 if prev2 is not None else pooled, ym2)
        ym1 = torch.where(newp, prev1 if prev1 is not None else pooled, ym1)
        y0 = torch.where(newp, pooled, y0)
        best_val = torch.where(newp, pooled, best_val)
        best_i = torch.where(newp, torch.full_like(best_i, i), best_i)
        need = torch.where(newp, torch.full_like(need, 2), need)
        prev2 = prev1
        prev1 = pooled

    # The shared search variable is echo m's shift; regularise/guard it, then rescale
    # to echo-1 units. conf is the pooled (SNR-weighted) searchlight quality map.
    s_field, conf = _searchlight_field_and_conf(
        best_i,
        best_val,
        zero_val,
        ym2,
        ym1,
        y0,
        yp1,
        yp2,
        nd=nd,
        r=r,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        blur=lambda x: _blur3d_b(x, reg_sigma, slicewise_axis),
    )
    return s_field / float(alpha[m]), conf  # echo-1 scale (alpha_1 = 1)


def xcorr_search_flow_3d_axes(
    fixed_list: list[torch.Tensor],
    moving_list: list[torch.Tensor],
    axes: list[int],
    alphas: torch.Tensor,
    *,
    max_shift: float = 3.0,
    window_sigma: float = 2.0,
    trial_step: float = 0.5,
    weights: list[torch.Tensor] | None = None,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    fourier_shift: bool = True,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    n_passes: int = 3,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    curve_out: list[torch.Tensor] | None = None,
    slicewise_axis: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The xcorr counterpart of :func:`optical_flow_lk_3d_axes` — 1-2 axes, 1-N echoes.

    One axis delegates straight to :func:`xcorr_search_flow_3d_multiecho`. Two axes are
    searched SEPARABLY, alternating for ``n_passes``: a joint 2-D offset grid would cost
    O(trials²) per voxel, which the 2-D dual-PE path already declined to pay for the same
    reason.

    Each half-pass is a true block-coordinate step, not an increment: the moving volumes
    are pre-warped by the OTHER axis's current estimate only, so the search along this
    axis sees the other one already corrected and re-solves its own component in full.
    Assigning the result (rather than accumulating a residual search on top of a
    partly-warped image) keeps every pass a search over the original intensities, so the
    sub-voxel parabola fit never compounds its own quantisation, and the estimate stays
    inside ``max_shift`` by construction.

    Returns ``(fields, confs)``, one per axis in ``axes`` order, each in echo-1 units
    (``alphas[k, e] · fields[k]`` is echo ``e``'s shift along ``axes[k]``).

    ``curve_out`` (the correlation landscape) is only captured for the single-axis case —
    under alternation there is no single landscape, each axis has one per pass.
    """
    if len(axes) not in (1, 2):
        raise ValueError(f"xcorr_search_flow_3d_axes takes 1 or 2 axes, got {axes}")
    if len(axes) == 2 and axes[0] == axes[1]:
        raise ValueError(f"the two axes must differ, got {axes}")
    if slicewise_axis is not None and slicewise_axis in axes:
        raise ValueError(
            f"slicewise_axis ({slicewise_axis}) must differ from every encode axis "
            f"({axes}): the encode directions have to lie inside the slice plane."
        )
    if alphas.shape[0] != len(axes) or alphas.shape[1] != len(fixed_list):
        raise ValueError(
            f"alphas must be (n_axes, n_echoes) = ({len(axes)}, {len(fixed_list)}), "
            f"got {tuple(alphas.shape)}"
        )

    def _search(mv: list[torch.Tensor], k: int, curves):
        return xcorr_search_flow_3d_multiecho(
            fixed_list,
            mv,
            alphas[k],
            axes[k],
            max_shift=max_shift,
            window_sigma=window_sigma,
            trial_step=trial_step,
            weights=weights,
            noshift_margin=noshift_margin,
            reg_sigma=reg_sigma,
            fourier_shift=fourier_shift,
            peak_mode=peak_mode,
            search_min_steps=search_min_steps,
            curve_out=curves,
            slicewise_axis=slicewise_axis,
        )

    if len(axes) == 1:
        w, conf = _search(moving_list, 0, curve_out)
        return [w], [conf]

    n_echo = len(fixed_list)
    w = [torch.zeros_like(fixed_list[0]) for _ in axes]
    confs = [torch.zeros_like(fixed_list[0]) for _ in axes]
    for _ in range(n_passes):
        for k in (0, 1):
            other = 1 - k
            mv = [
                _shift3d_axes(
                    moving_list[j],
                    [float(alphas[other, j]) * w[other]],
                    [axes[other]],
                    mode=warp_interp,
                    radius=warp_radius,
                )
                for j in range(n_echo)
            ]
            w[k], confs[k] = _search(mv, k, None)
    return w, confs


@dataclass
class MultiEchoLocomocoResult:
    """Joint multi-echo result: one shared field, per-echo scaled warps.

    ``per_echo[e]`` is a full :class:`LocomocoResult` for echo ``e`` whose warp is
    ``alpha[e]·w_field`` and whose corrected series is that echo warped by it — so the
    CLI writes per-echo warps/corrected exactly like the single-echo path. ``alpha`` /
    ``echo_times`` and ``linearity_r2`` (fit of ``alpha`` vs ``TE`` through the origin)
    are the shared diagnostics.
    """

    per_echo: list[LocomocoResult]
    alpha: torch.Tensor  # (E,) learned per-echo scaling
    echo_times: torch.Tensor  # (E,) ms
    w_field: torch.Tensor  # (nx,ny,nz,T) shared canonical PE displacement (voxels)
    pe_axis: int
    linearity_r2: float
    confidence: torch.Tensor | None = None  # (nx,ny,nz,T) searchlight quality map, if xcorr
    corr_curve: torch.Tensor | None = None  # (nx,ny,nz,nd) per-voxel corr vs offset, one frame
    corr_offsets: torch.Tensor | None = None  # (nd,) the trial offsets (voxels) for corr_curve
    # How alpha is normalised, for the diagnostic header. Only alpha·w is determined, so
    # the divisor is a display choice — and ÷echo1 is unreadable when echo 1 is the
    # inter-echo anchor (alpha_1 ≈ 0 blows every other entry up).
    alpha_label: str = "alpha(÷echo1)"
    # ── two encode axes: the PRIMARY phase-encode field alongside the partition one ──
    # `w_field` / `pe_axis` / `alpha` always describe the axis the per-echo scaling law
    # applies to — the partition axis in a dual run. The primary-PE field is flat across
    # echoes by construction (alpha = 1), so it needs no scaling vector of its own.
    w_field_pe1: torch.Tensor | None = None  # (nx,ny,nz,T) primary-PE displacement
    pe_axis1: int | None = None


def _rank1_factor_echoes(disps: torch.Tensor, alpha_init: torch.Tensor, n_iter: int = 8):
    """Rank-1 factor per-echo displacement stacks into ``alpha`` (E,) and ``w`` (R,).

    ``disps`` is ``(E, R)`` (R = voxels·frames flattened). Power iteration on the model
    ``disps_e ≈ alpha_e · w``: alternately ``w = Σ α_e d_e / Σ α_e²`` and
    ``alpha_e = ⟨d_e, w⟩ / ⟨w, w⟩``, seeded from ``alpha_init`` (the echo times). The
    product ``alpha_e·w`` is what the output uses, so the overall scale is arbitrary;
    we normalise ``alpha`` by its first entry for an interpretable TE comparison.
    """
    alpha = alpha_init.clone().float()
    w = torch.zeros(disps.shape[1], dtype=disps.dtype)
    for _ in range(n_iter):
        denom = max(float((alpha * alpha).sum()), 1e-12)
        w = (alpha[:, None] * disps).sum(0) / denom
        ww = float((w * w).sum())
        if ww < 1e-12:
            break
        alpha = (disps * w[None, :]).sum(1) / ww
    a0 = float(alpha[0]) if abs(float(alpha[0])) > 1e-12 else 1.0
    return alpha / a0, w


def estimate_residual_flow_multiecho(
    datas: list[np.ndarray],
    echo_times: list[float],
    pe_axis: int,
    slice_axis: int,
    *,
    ref_mode: str = "mean",
    backend: str = "flow",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    refine_rounds: int = 0,
    converge: float = 0.0,
    converge_rel: float = 0.0,
    first_n: int | None = None,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    coverage_erode: int | None = 1,
    learn_scaling: bool = True,
    flat_scaling: bool = False,
    alpha_override: torch.Tensor | None = None,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    save_corr_curve: int | None = None,
    want_corrected: bool = True,
    warp_interp: str = "bilinear",
    warp_radius: int = 3,
    hpf_sigma: float = 0.0,
    match: str = "none",
    match_sigma: float = 6.0,
    ngf_eta_q: float = 0.5,
    pe_axis2: int | None = None,
    xcorr_passes: int = 3,
    sep_floor: float = 1e-2,
    slicewise: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Joint residual encode-axis motion for multi-echo EPI.

    ``datas`` is the list of E moco'd 4-D series (one per echo, identical grid + T);
    ``echo_times`` the matching TEs in ms. ``pe_axis`` is the corrected direction (the
    slice/partition axis for 3-D EPI, so PE==slice is fine). By default all echoes are
    treated as one 3-D-acquired series (no per-slice fields).

    ``slicewise=True`` is the 2-D MULTI-SLICE case instead: one field per slice, solved
    from that slice's own data, with ``slice_axis`` excluded from every pooling window
    and from the pyramid. Echoes still pool — every echo of a given slice is acquired at
    the same instant, so they are independent looks at ONE displacement. That is the
    whole multi-echo win here: more evidence per slice-time, without smearing across
    slice-times. A 2-D acquisition has no partition direction, so this is a single-axis,
    TE-INDEPENDENT solve — pass ``flat_scaling=True``.

    Learns a shared field ``w(r,t)`` and per-echo scaling ``alpha`` under the model
    ``disp_e = alpha_e · w``:

    1. Estimate each echo independently (3-D LK / xcorr) → per-echo displacement.
    2. Rank-1 factor across echoes (seeded by the echo times) → ``alpha`` + initial ``w``.
       ``learn_scaling=False`` skips this and fixes ``alpha_e = TE_e/TE_0``;
       ``flat_scaling=True`` instead fixes ``alpha_e = 1`` (every echo shifts the SAME
       amount — for acquisitions whose partition wiggle is TE-independent) while still
       pooling all echoes' signal.
    3. ``flow`` backend: solve ``w`` with the shared-field pooled LK
       (:func:`optical_flow_lk_3d_multiecho`) so the SNR/sensitivity pooling happens in
       image space. ``xcorr``: keep the factored ``w``.
    4. ``refine_rounds`` (``flow`` only): rebuild each echo's reference from its corrected
       series (motion removed → sharp) and re-solve, converging the template out of its
       motion-blur bias — the same knob as the single-echo ``-refine`` and the dominant
       lever on recovered MAGNITUDE (a blurred initial reference biases displacement low).

    ``pe_axis2`` adds the PARTITION axis alongside ``pe_axis`` (then read as the PRIMARY
    phase encode) and solves both shared fields at once. The two carry DIFFERENT, known
    per-echo scaling laws — primary PE is flat (``alpha=1``, TE-independent), partition
    is ``TE_e/TE_1`` — and that difference is what makes the two-axis multi-echo case
    well posed where the single-echo one is not: the echo axis separates the axes even
    where the local image gradient cannot. Only the PARTITION alpha is ever learned; the
    primary axis stays pinned flat, because its TE-independence is the physical premise
    the whole decomposition rests on and freeing it would let the two laws collapse onto
    each other. See :func:`optical_flow_lk_3d_axes`.

    Returns per-echo :class:`LocomocoResult`\\ s (warp ``alpha_e·w``, echo warped by it)
    plus the shared ``alpha`` / linearity diagnostics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = len(datas)
    if e < 2:
        raise ValueError(f"multi-echo needs ≥2 echoes, got {e}.")
    if len(echo_times) != e:
        raise ValueError(f"-echo_times has {len(echo_times)} values but there are {e} inputs.")
    shp = datas[0].shape
    for k, d in enumerate(datas):
        if d.shape != shp:
            raise ValueError(
                f"echo {k} shape {d.shape} != echo 0 shape {shp} (need identical grid+T)."
            )
    if len(shp) != 4:
        raise ValueError(f"each -input must be 4D, got {shp}.")
    if any(t <= 0 for t in echo_times):
        raise ValueError(f"-echo_times must all be > 0 ms, got {echo_times}.")
    nx, ny, nz, nt = shp

    # Geometry first: whether this is a 2-D multi-slice or a 3-D solve decides what the
    # shared validation should even be checking (PE == slice is fine in 3-D, fatal in 2-D).
    dual = pe_axis2 is not None
    if dual and pe_axis2 == pe_axis:
        raise ValueError(f"the two encode axes must differ, got pe_axis=pe_axis2={pe_axis}")
    if slicewise and dual:
        raise ValueError("a 2-D multi-slice acquisition has no partition direction; drop pe_axis2.")
    axes = [pe_axis] if not dual else [pe_axis, pe_axis2]
    if dual:
        disp_slice = next(a for a in (0, 1, 2) if a not in axes)
    elif slicewise:
        if slice_axis == pe_axis:
            raise ValueError(
                f"slicewise needs the PE axis inside the slice plane, but slice_axis and "
                f"pe_axis are both {pe_axis}."
            )
        disp_slice = slice_axis
    else:
        disp_slice = (
            slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
        )
    # The axis every pooling window and pyramid step must leave alone (None = 3-D solve).
    sw_axis = disp_slice if slicewise else None

    _validate_estimation_inputs(
        shp,
        pe_axis,
        slice_axis,
        is_3dacq=not slicewise,
        max_shift=max_shift,
        trial_step=trial_step,
        window_sigma=window_sigma,
        verbose=verbose,
    )

    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    n_ax = len(axes)
    if verbose:
        print(
            f"🌀 locomoco "
            f"{_geometry_report(shp, pe_axis, disp_slice, is_3dacq=not slicewise, dual=dual)}"
            f"  × {e} echoes"
        )

    te = torch.tensor([float(x) for x in echo_times], dtype=torch.float32)
    vols = [torch.from_numpy(np.ascontiguousarray(d)).float() for d in datas]

    # Estimation prep: optional spatial high-pass (raw kept for the resample), then the
    # existing estimation blur. Identity when both sigmas are 0. v is a single (X,Y,Z) vol.
    def _prep(v: torch.Tensor) -> torch.Tensor:
        # Cross-TIME matching, applied per echo and to both sides. Each echo is normalised
        # against itself, so the TE relationship the joint solve depends on is untouched:
        # `alphas` scales the DISPLACEMENT field, never the intensities.
        x = _spatial_highpass3d(
            _match_prep(v[None], match, match_sigma, sw_axis, pe_axis=pe_axis, ngf_eta_q=ngf_eta_q),
            hpf_sigma,
            sw_axis,
        )
        if smooth_sigma > 0:
            x = _blur3d_b(x, smooth_sigma, sw_axis)
        return x[0]

    refs = [_prep(_select_ref_vol(v, ref_mode, first_n).to(device)) for v in vols]

    # A fixed scaling (TE ratio, flat, or an explicit law) skips learning; all still pool
    # every echo into one solve.
    learn = learn_scaling and not flat_scaling and alpha_override is None

    # Single-echo backend estimator — used to LEARN alpha (rank-1 factor of one per-echo
    # pass) and, for xcorr, to solve the shared field itself. Persistent bars (leave=True)
    # so a long run stays visible.
    flow3d = _build_flow3d_axes_fn(
        backend,
        axes,
        torch.ones(n_ax, 1),  # one echo at a time: no cross-echo scaling to apply
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        max_shift=max_shift,
        trial_step=trial_step,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        peak_mode=peak_mode,
        search_min_steps=search_min_steps,
        warp_interp=warp_interp,
        warp_radius=warp_radius,
        xcorr_passes=xcorr_passes,
        sep_floor=sep_floor,
        slicewise_axis=sw_axis,
    )

    # Confidence map from the pooled searchlight (fixed/flat-scaling xcorr path only — the
    # LK and learn-alpha paths have no single per-voxel searchlight quality). Filled by the
    # final _pooled_xcorr call and surfaced on the result for a `-save_confidence` diag.
    pooled_conf = torch.zeros(nx, ny, nz, nt)
    have_pooled_conf = False
    # Per-voxel separability of the two axes (flow backend only) — see optical_flow_lk_3d_axes.
    sep_field = torch.zeros(nx, ny, nz, nt) if (dual and backend == "flow") else None
    curve_frame = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
    corr_curve: torch.Tensor | None = None
    corr_offsets: torch.Tensor | None = None

    def _per_echo_estimate(cur_refs: list[torch.Tensor], tag: str) -> torch.Tensor:
        """Each echo solved ALONE, ``(n_ax, e, nx, ny, nz, nt)`` — the input to alpha learning."""
        out = torch.zeros(n_ax, e, nx, ny, nz, nt)
        for j in range(e):
            for t in tqdm(
                range(nt), desc=f"me {tag} e{j + 1}/{e}", unit="frame", leave=True, disable=nt < 3
            ):
                mv = vols[j][..., t].to(device)
                mvb = _prep(mv)
                fields = flow3d([cur_refs[j][None]], [mvb[None]])
                for k in range(n_ax):
                    out[k, j, ..., t] = fields[k][0].cpu()
        return out

    def _project(flat: torch.Tensor, k: int) -> torch.Tensor:
        """Least-squares project axis ``k``'s per-echo estimates (e, R) onto its alpha."""
        a = alphas[k]
        return (a[:, None] * flat).sum(0) / max(float((a * a).sum()), 1e-12)

    def _joint_lk(cur_refs: list[torch.Tensor], tag: str) -> list[torch.Tensor]:
        out = [torch.zeros(nx, ny, nz, nt) for _ in axes]
        for t in tqdm(range(nt), desc=f"me {tag}", unit="frame", leave=True, disable=nt < 3):
            movs = [_prep(vols[j][..., t].to(device)) for j in range(e)]
            sep_acc: list[torch.Tensor] | None = [] if (dual and sep_field is not None) else None
            fields = optical_flow_lk_3d_axes(
                [r[None] for r in cur_refs],
                [m[None] for m in movs],
                axes,
                alphas,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                max_disp=max_shift if dual else None,
                sep_floor=sep_floor,
                sep_out=sep_acc,
                slicewise_axis=sw_axis,
            )
            for k in range(n_ax):
                out[k][..., t] = fields[k][0].cpu()
            if sep_acc and sep_field is not None:
                sep_field[..., t] = sep_acc[0][0].cpu()
        return out

    def _pooled_xcorr(cur_refs: list[torch.Tensor], tag: str) -> list[torch.Tensor]:
        nonlocal have_pooled_conf, corr_curve, corr_offsets
        out = [torch.zeros(nx, ny, nz, nt) for _ in axes]
        for t in tqdm(range(nt), desc=f"me {tag}", unit="frame", leave=True, disable=nt < 3):
            movs = [_prep(vols[j][..., t].to(device)) for j in range(e)]
            # The correlation landscape is a single-axis concept; under alternation each
            # axis has one per pass, so it is captured only for the one-axis case.
            curve_acc: list[torch.Tensor] | None = [] if (t == curve_frame and not dual) else None
            w_be, c_be = xcorr_search_flow_3d_axes(
                [r[None] for r in cur_refs],
                [m[None] for m in movs],
                axes,
                alphas,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                peak_mode=peak_mode,
                search_min_steps=search_min_steps,
                n_passes=xcorr_passes,
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                curve_out=curve_acc,
                slicewise_axis=sw_axis,
            )
            for k in range(n_ax):
                out[k][..., t] = w_be[k][0].cpu()
            # A voxel is only as trustworthy as its weaker axis.
            conf = c_be[0][0]
            for extra in c_be[1:]:
                conf = torch.minimum(conf, extra[0])
            pooled_conf[..., t] = conf.cpu()
            if curve_acc is not None:
                corr_curve = torch.stack(curve_acc, 0)[:, 0].permute(1, 2, 3, 0).contiguous()
                rr = float(max(1, int(np.ceil(max_shift))))
                corr_offsets = torch.arange(-rr, rr + 1e-6, trial_step)
        have_pooled_conf = True
        return out

    def _solve_w(cur_refs: list[torch.Tensor], tag: str) -> list[torch.Tensor]:
        # flow: image-space shared-field pooled LK. xcorr with FIXED alpha: shared-parameter
        # searchlight pooling every echo into one informed search (the "best of both"). xcorr
        # while LEARNING alpha: per-echo estimate + project (the per-echo fields are needed to
        # learn the ratios anyway), warp-space pooling.
        if backend == "flow":
            return _joint_lk(cur_refs, tag)
        if not learn:
            return _pooled_xcorr(cur_refs, tag)
        per = _per_echo_estimate(cur_refs, tag)
        return [_project(per[k].reshape(e, -1), k).reshape(nx, ny, nz, nt) for k in range(n_ax)]

    def _corrected_echo(cur_w: list[torch.Tensor], j: int) -> torch.Tensor:
        out = torch.zeros(nx, ny, nz, nt)
        for t in range(nt):
            out[..., t] = _shift3d_axes(
                vols[j][..., t].to(device)[None],
                [float(alphas[k, j]) * cur_w[k][..., t].to(device)[None] for k in range(n_ax)],
                axes,
                mode=warp_interp,
                radius=warp_radius,
            )[0].cpu()
        return out

    # Phase 1: learn alpha, then the initial shared field. When learning, factor ONE
    # per-echo pass; for xcorr reuse that very pass as the initial w (no second sweep).
    # flat_scaling pins alpha to 1 (TE-independent shift); else the TE ratio. An explicit
    # alpha_override supplies a law neither covers — e.g. the inter-echo ladder
    # alpha ∝ (TE_e − TE_1), which is zero at the anchor echo rather than at TE = 0.
    if alpha_override is not None:
        if alpha_override.shape != (e,):
            raise ValueError(f"alpha_override must be ({e},), got {tuple(alpha_override.shape)}.")
        partition_alpha = alpha_override.float().clone()
    else:
        partition_alpha = torch.ones(e) if flat_scaling else te / float(te[0])
    if dual:
        # The two laws must DIFFER — that difference is what separates the axes across
        # echoes. Primary PE is pinned flat (TE-independent by construction); only the
        # partition row is ever learned.
        alphas = torch.stack([torch.ones(e), partition_alpha])
        # Identical laws collapse the advantage: if the partition scales flat too, the
        # echo axis carries no information distinguishing the axes and the solve degrades
        # to the (ill-posed on straight edges) single-echo dual case with more SNR. Worth
        # saying out loud, because the run still "works" and the numbers still look fine.
        spread = float((partition_alpha / partition_alpha.abs().max().clamp_min(1e-12)).std())
        if spread < 1e-3 and verbose:
            print(
                "   ⚠ both encode axes have the SAME per-echo scaling law, so the echo "
                "axis cannot separate them — the split between axes now rests on image "
                "structure alone (check _locomoco_sep). Drop -me_flat_scaling to let the "
                "partition axis scale with TE."
            )
    else:
        alphas = partition_alpha[None]
    if learn:
        disp0 = _per_echo_estimate(refs, "init")
        # Learn only the partition row; row 0 of a dual solve stays flat.
        k_learn = n_ax - 1
        learned, _ = _rank1_factor_echoes(disp0[k_learn].reshape(e, -1), alphas[k_learn])
        alphas[k_learn] = learned
        w = (
            _joint_lk(refs, "joint")
            if backend == "flow"
            else [_project(disp0[k].reshape(e, -1), k).reshape(nx, ny, nz, nt) for k in range(n_ax)]
        )
    else:
        w = _solve_w(refs, "joint" if backend == "flow" else "solve")
    alpha = alphas[-1]  # the reported/diagnosed law is the partition one

    # Gate every solve, not just the last: _reduce_me derives each echo's next reference
    # from w, so ungated no-data flow would go straight into the refined template.
    soft_xyz = None
    if automask:
        # One shared mask from the across-echo mean (all echoes share geometry).
        mean_series = np.mean(np.stack([np.asarray(d) for d in datas], 0), 0)
        soft = _build_soft_mask(
            mean_series,
            disp_slice,
            a0,
            a1,
            automask_dilate,
            automask_sigma,
            device,
            coverage_erode=coverage_erode,
            # 2-D multi-slice: margin/feather are in-plane concepts, per slice.
            in_plane=slicewise,
        )
        soft_xyz = soft.permute(_inv_perm([disp_slice, a0, a1])).contiguous()

    def _gate_me(fields: list[torch.Tensor]) -> list[torch.Tensor]:
        if soft_xyz is None:
            return fields
        return [f * soft_xyz[..., None] for f in fields]

    w = _gate_me(w)

    # Phase 2: reference-refinement (the -refine knob; applies to BOTH backends — it is
    # the estimator-agnostic template sharpening). The initial reference is the mean/median
    # of the still-DISTORTED frames, so it is motion-blurred and biases every shift LOW —
    # the dominant cause of under-displacement (and the main thing -superhard turns on).
    # Rebuild each echo's reference from its corrected series (sharp) and re-solve, with the
    # divergence guard / convergence checks of the single-echo 3-D path.
    if refine_rounds > 0:
        # State is w itself (corrected is derived per echo on demand); the shared engine
        # carries corrected == w. brain is None until w has signal, so brain_rms falls
        # back to the whole field. Each solve's tqdm label tracks the pass number.
        tot = torch.stack([f.abs() for f in w], 0).sum(0)
        brain = _brain_mask_from(tot.mean(dim=3)) if float(tot.max()) > 0 else None
        del tot
        _pass = {"n": 0}

        def _reduce_me(cur_w):
            return [
                _prep(_refine_reduce(_corrected_echo(cur_w, j), ref_mode, 3, first_n).to(device))
                for j in range(e)
            ]

        def _est_me(new_refs):
            _pass["n"] += 1
            w_new = _gate_me(_solve_w(new_refs, f"refine {_pass['n']}/{refine_rounds}"))
            return w_new, w_new

        def _brain_rms_me(disp, prev):
            # Total move across every axis: a dual run converges only when neither
            # component is still shifting.
            acc = 0.0
            for a, b in zip(disp, prev, strict=True):
                d = (a[brain] - b[brain]) if brain is not None else (a - b)
                if d.numel():
                    acc += float(d.pow(2).mean())
            return acc**0.5

        w, _ = _refine_loop(
            _est_me,
            w,
            w,
            reduce_ref=_reduce_me,
            brain_rms=_brain_rms_me,
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            max_shift=max_shift,
            verbose=verbose,
        )

    # Build per-echo results: echo e warped by alpha_e · w.
    per_echo: list[LocomocoResult] = []
    sep_canon = None if sep_field is None else sep_field
    for j in range(e):
        # Materializing the corrected 4-D series (a per-frame warp of every echo) is
        # pure waste when the caller won't write it (-no_corrected); the warp field and
        # diagnostics don't need it. Refine builds its own corrected internally above.
        corrected = _corrected_echo(w, j) if want_corrected else None
        canon = [
            (alphas[k, j] * w[k]).contiguous().permute(perm4).contiguous() for k in range(n_ax)
        ]
        by_axis = dict(zip(axes, canon, strict=True))
        zero = torch.zeros_like(canon[0])
        per_echo.append(
            LocomocoResult(
                u_canon=by_axis.get(a1, zero),
                v_canon=by_axis.get(a0, zero),
                corrected_canon=torch.zeros_like(canon[0]),
                perm=perm4,
                pe_flow_is_u=pe_flow_is_u,
                pe_axis=pe_axis,
                slice_axis=disp_slice,
                orig_shape=(nx, ny, nz, nt),
                a0=a0,
                a1=a1,
                dual=dual,
                pe_axis2=pe_axis2,
                sep_map=sep_canon,
                corrected_nifti=corrected,
            )
        )

    # Linearity of alpha vs TE (through the origin): r² of the 1-parameter fit — only
    # meaningful when alpha was LEARNED; an imposed scaling (TE ratio or flat) is 1 by fiat.
    if learn:
        slope = float((alpha * te).sum() / (te * te).sum().clamp(min=1e-12))
        resid = alpha - slope * te
        ss_res = float((resid * resid).sum())
        ss_tot = float(((alpha - alpha.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    else:
        r2 = 1.0

    if verbose:
        ap = w[-1].abs()
        sel = ap[ap > 0]
        med = float(sel.median()) if sel.numel() else 0.0
        a_str = ", ".join(f"{float(x):.3f}" for x in alpha)
        te_str = ", ".join(f"{float(x):.0f}" for x in te)
        if learn:
            tag = f"·  learned, linear-in-TE r²={r2:.4f}"
        elif flat_scaling:
            tag = "·  FLAT scaling (alpha=1, all echoes shift equally)"
        else:
            tag = "·  fixed to TE ratio"
        which = f"partition axis {pe_axis2}" if dual else f"PE axis {pe_axis}"
        print(
            f"🌀 locomoco ME-3D: {e} echoes (TE {te_str} ms), {nt} frames, {which}, "
            f"backend={backend}; shared |w| median {med:.3f} vox, max {float(ap.max()):.3f} vox"
        )
        print(f"   alpha (÷echo1) = [{a_str}]  {tag}")
        if dual:
            p1 = w[0].abs()
            s1 = p1[p1 > 0]
            print(
                f"   primary PE axis {pe_axis} (alpha pinned FLAT, TE-independent): "
                f"|w| median {float(s1.median()) if s1.numel() else 0.0:.3f} vox, "
                f"max {float(p1.max()):.3f} vox"
            )
            if sep_field is not None:
                sv = sep_field[sep_field > 0]
                if sv.numel():
                    print(
                        f"   axis separability: median {float(sv.median()):.3f} "
                        f"(multi-echo separates the axes by their differing TE laws)"
                    )

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=w[-1],
        pe_axis=axes[-1],
        linearity_r2=r2,
        confidence=pooled_conf if have_pooled_conf else None,
        corr_curve=corr_curve,
        corr_offsets=corr_offsets,
        w_field_pe1=w[0] if dual else None,
        pe_axis1=pe_axis if dual else None,
    )


def estimate_residual_flow_me_scaled(
    datas: list[np.ndarray],
    echo_times: list[float],
    estimate_idx: int,
    pe_axis: int,
    slice_axis: int,
    *,
    ref_mode: str = "median",
    backend: str = "xcorr",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    refine_rounds: int = 0,
    converge: float = 0.0,
    converge_rel: float = 0.0,
    first_n: int | None = None,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    coverage_erode: int | None = 1,
    flat_scaling: bool = False,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    warp_interp: str = "lanczos",
    warp_radius: int = 3,
    hpf_sigma: float = 0.0,
    match: str = "none",
    match_sigma: float = 6.0,
    ngf_eta_q: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Estimate the shared partition field on ONE echo, scale to the rest by TE ratio.

    The ME 3-D-EPI partition wiggle scales linearly with echo time, so once that is
    established there is nothing to learn jointly: run the full single-echo ``-is_3dacq``
    estimator (including ``-refine``) on the echo where the shift is easiest to see —
    typically the LAST (largest displacement) or a middle echo — and every other echo's
    warp is ``(TE_e / TE_k) · w``. Much cheaper than the joint solve (one echo's data, one
    pass) and often steadier (that echo's own SNR, no cross-echo compromise).

    ``estimate_idx`` is the 0-based echo to estimate on. The shared field ``w`` is stored
    echo-1-scaled and ``alpha`` is ``TE / TE_0`` (same convention as the joint path), so
    ``linearity_r2`` is 1.0 by construction (the scaling is applied, not fitted).
    ``flat_scaling=True`` applies the SAME field to every echo (``alpha_e = 1``) — for
    acquisitions whose partition wiggle is TE-independent.

    ``warp_interp`` controls both the selected echo's correction/refine template and
    the one-pass application of its scaled field to every other echo. Lanczos is the
    default so none of those corrected series inherits linear-interpolation blur.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = len(datas)
    if len(echo_times) != e:
        raise ValueError(f"-echo_times has {len(echo_times)} values but there are {e} inputs.")
    if not 0 <= estimate_idx < e:
        raise ValueError(f"estimate_idx {estimate_idx} out of range for {e} echoes.")
    te = torch.tensor([float(x) for x in echo_times], dtype=torch.float32)

    # Reuse the whole single-echo 3-D-acq estimator on the chosen echo (refine, automask,
    # ref modes — all of it). Its PE displacement IS the echo-k field.
    res_k = estimate_residual_flow(
        datas[estimate_idx],
        pe_axis,
        slice_axis,
        ref_mode=ref_mode,
        backend=backend,
        smooth_sigma=smooth_sigma,
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        max_shift=max_shift,
        trial_step=trial_step,
        refine_rounds=refine_rounds,
        converge=converge,
        converge_rel=converge_rel,
        first_n=first_n,
        automask=automask,
        automask_dilate=automask_dilate,
        automask_sigma=automask_sigma,
        coverage_erode=coverage_erode,
        is_3dacq=True,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        peak_mode=peak_mode,
        search_min_steps=search_min_steps,
        warp_interp=warp_interp,
        warp_radius=warp_radius,
        hpf_sigma=hpf_sigma,
        match=match,
        match_sigma=match_sigma,
        ngf_eta_q=ngf_eta_q,
        device=device,
        verbose=verbose,
    )
    if flat_scaling:
        # Same shift for every echo: alpha=1, w is the echo-k field applied unchanged.
        alpha = torch.ones(e)
        w = res_k.pe_displacement()
    else:
        alpha = te / float(te[0])
        # Rescale the echo-k field to echo-1 scale so w/alpha match the joint-path
        # convention; the product alpha_e·w = w_k·(TE_e/TE_k) is unchanged either way.
        w = res_k.pe_displacement() * float(te[0] / te[estimate_idx])
    nx, ny, nz, nt = w.shape
    perm4, pe_flow_is_u = res_k.perm, res_k.pe_flow_is_u
    disp_slice, a0, a1 = res_k.slice_axis, res_k.a0, res_k.a1

    per_echo: list[LocomocoResult] = []
    for j in range(e):
        if j == estimate_idx:
            per_echo.append(res_k)  # its own warp already == alpha_k·w
            continue
        disp_e = (alpha[j] * w).contiguous()
        vol_j = torch.from_numpy(np.ascontiguousarray(datas[j])).float()
        corrected = torch.zeros(nx, ny, nz, nt)
        for t in range(nt):
            corrected[..., t] = _shift3d_axis(
                vol_j[..., t].to(device)[None],
                disp_e[..., t].to(device)[None],
                pe_axis,
                mode=warp_interp,
                radius=warp_radius,
            )[0].cpu()
        p_canon = disp_e.permute(perm4).contiguous()
        u_canon = p_canon if pe_flow_is_u else torch.zeros_like(p_canon)
        v_canon = torch.zeros_like(p_canon) if pe_flow_is_u else p_canon
        per_echo.append(
            LocomocoResult(
                u_canon=u_canon,
                v_canon=v_canon,
                corrected_canon=torch.zeros_like(p_canon),
                perm=perm4,
                pe_flow_is_u=pe_flow_is_u,
                pe_axis=pe_axis,
                slice_axis=disp_slice,
                orig_shape=(nx, ny, nz, nt),
                a0=a0,
                a1=a1,
                dual=False,
                corrected_nifti=corrected,
            )
        )

    if verbose:
        ap = w.abs()
        sel = ap[ap > 0]
        med = float(sel.median()) if sel.numel() else 0.0
        te_str = ", ".join(f"{float(x):.0f}" for x in te)
        a_str = ", ".join(f"{float(x):.3f}" for x in alpha)
        scale_note = "FLAT (alpha=1, all echoes equal)" if flat_scaling else "TE-linear"
        print(
            f"🌀 locomoco ME-3D (scaled from echo {estimate_idx + 1}, "
            f"TE {float(te[estimate_idx]):.0f} ms): {e} echoes (TE {te_str} ms), {nt} frames; "
            f"|w| median {med:.3f} vox, max {float(ap.max()):.3f} vox"
        )
        print(f"   alpha (÷echo1) = [{a_str}]  ·  {scale_note} scaling applied (not fitted)")

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=w,
        pe_axis=pe_axis,
        linearity_r2=1.0,
    )


def make_raw_reference_me_result(
    datas: list[np.ndarray],
    echo_times: list[float],
    pe_axis: int,
    slice_axis: int,
    *,
    flat_scaling: bool = False,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Zero-motion multi-echo container for the ``-backend qwarp`` direct path.

    Builds exactly what :func:`polish_me_result` (``full=True``) needs WITHOUT running
    the optical-flow estimation: each echo's "corrected" series IS the raw echo, so the
    qwarp reference (temporal median of the corrected series) is a plain median of raw,
    and qwarp then owns the whole field (raw echoes -> that median). Use when the inputs
    are already motion-corrected (NORDIC'd / moco'd) so a flow-refined template buys
    nothing -- it skips the full estimation pass. For residual motion, run the flow
    estimate instead (``-backend flow -final_qwarp``).

    Geometry is the same deterministic derivation the estimator uses; ``alpha`` is the
    fixed TE ratio (or 1 for flat scaling). The canonical flow tensors are unused on this
    path (only ``corrected_nifti`` + the geometry fields are read), so they are tiny
    placeholders rather than full-size zeros.
    """
    e = len(datas)
    shp = datas[0].shape
    nx, ny, nz, nt = (int(s) for s in shp)
    disp_slice = slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    te = torch.tensor([float(x) for x in echo_times], dtype=torch.float32)
    alpha = torch.ones(e) if flat_scaling else te / float(te[0])

    if verbose:
        print(
            f"🌀 locomoco {_geometry_report(shp, pe_axis, disp_slice, is_3dacq=True)}  × {e} echoes"
        )
        a_str = ", ".join(f"{float(x):.3f}" for x in alpha)
        tag = "FLAT (alpha=1)" if flat_scaling else "fixed to TE ratio"
        print(
            f"   ⏭️  skipping flow estimate (qwarp owns the field); alpha (÷echo1) = [{a_str}]  {tag}"
        )

    ph = torch.zeros((nt, 1, 1, 1))  # placeholder canonical tensors — unused on this path
    per_echo = [
        LocomocoResult(
            u_canon=ph,
            v_canon=ph,
            corrected_canon=ph,
            perm=perm4,
            pe_flow_is_u=pe_flow_is_u,
            pe_axis=pe_axis,
            slice_axis=disp_slice,
            orig_shape=(nx, ny, nz, nt),
            a0=a0,
            a1=a1,
            corrected_nifti=torch.from_numpy(np.ascontiguousarray(datas[j])).float(),
        )
        for j in range(e)
    ]
    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=torch.zeros(nx, ny, nz, nt),
        pe_axis=pe_axis,
        linearity_r2=1.0,
    )


def make_raw_reference_result(
    data: np.ndarray,
    pe_axis: int,
    slice_axis: int,
    *,
    is_3dacq: bool = False,
    dual: bool = False,
    verbose: bool = True,
) -> LocomocoResult:
    """Zero-motion single-echo result for the ``-backend qwarp`` direct path.

    Single-echo analogue of :func:`make_raw_reference_me_result`: builds what the E=1
    qwarp wrap needs WITHOUT running :func:`estimate_residual_flow`. ``corrected_nifti``
    is the raw series, so the qwarp reference is a plain temporal median of raw and qwarp
    owns the whole field. The geometry MUST match the estimator it replaces so the saved
    warp keeps its orientation: the 3-D-acq path mirrors :func:`_run_3dacq_plain` (the
    ``disp_slice`` fallback when slice==PE), the 2-D path mirrors the slicewise branch of
    :func:`estimate_residual_flow` (``slice_axis`` used directly).
    """
    orig_shape = tuple(int(s) for s in data.shape)
    if is_3dacq:
        disp_slice = (
            slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
        )
    else:
        disp_slice = slice_axis
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm = [3, disp_slice, a0, a1]
    if verbose:
        print(
            f"🌀 locomoco {_geometry_report(orig_shape, pe_axis, disp_slice, is_3dacq=is_3dacq, dual=dual)}"
        )
        print("   ⏭️  skipping flow estimate (qwarp owns the field)")
    nt = orig_shape[3]
    ph = torch.zeros((nt, 1, 1, 1))  # placeholder canonical tensors — unused on this path
    return LocomocoResult(
        u_canon=ph,
        v_canon=ph,
        corrected_canon=ph,
        perm=perm,
        pe_flow_is_u=pe_flow_is_u,
        pe_axis=pe_axis,
        slice_axis=disp_slice,
        orig_shape=orig_shape,
        a0=a0,
        a1=a1,
        dual=dual,
        corrected_nifti=torch.from_numpy(np.ascontiguousarray(data)).float(),
    )


def _rewarp_raw_single_pass(
    raw_datas: list[np.ndarray],
    w_axes: list[torch.Tensor],
    axes: list[int],
    alpha_ch: torch.Tensor,
    device: torch.device,
    verbose: bool,
) -> list[torch.Tensor]:
    """Warp the RAW echoes by the TOTAL field in one wsinc5 pass, ``(nx, ny, nz, T)`` each.

    The qwarp polish registers the CORRECTED series, so its own warped output has been
    resampled twice: once by the estimator (historically bilinear) and
    again by qwarp. The second pass cannot put back what the first one blurred, and it
    also leaves the saved series describing a slightly different transform from the saved
    warp, which is the additive total ``w + r``. Applying that total to the raw data
    instead costs one extra resample of the raw and removes both problems.
    """
    from .interp import warp_image

    nx, ny, nz, nt = w_axes[0].shape
    out = []
    for j, raw_j in enumerate(raw_datas):
        raw = torch.from_numpy(np.ascontiguousarray(raw_j)).float()  # (nx, ny, nz, T)
        dst = torch.empty(nx, ny, nz, nt)
        chans = [torch.zeros(nz, ny, nx, device=device) for _ in range(3)]
        for t in tqdm(
            range(nt),
            desc=f"qwarp resample (echo {j + 1})",
            unit="frame",
            leave=True,
            disable=nt < 2 or not verbose,
        ):
            for c in chans:
                c.zero_()
            for k, ax in enumerate(axes):
                chans[ax] = (float(alpha_ch[ax, j]) * w_axes[k][..., t]).permute(2, 1, 0).to(device)
            src = raw[..., t].permute(2, 1, 0).contiguous().to(device)
            dst[..., t] = warp_image(src, *chans, mode="wsinc5").permute(2, 1, 0).cpu()
        out.append(dst.contiguous())
    return out


def polish_me_result(
    result: MultiEchoLocomocoResult,
    *,
    minpatch: int = 7,
    n_levels: int = 2,
    iters: int = 10,
    cost: str = "ncc",
    optimizer: str = "gn",
    compile: bool = False,
    full: bool = False,
    slicewise: bool = True,
    raw_datas: list[np.ndarray] | None = None,
    ref_mode: str = "median",
    refine: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Nonlinear joint TE-scaled ``qwarp`` of a multi-echo locomoco result.

    Two modes against the SAME reference -- the temporal median of the (refined)
    corrected series, a sharp motion-removed template:

    * ``full=False`` (**polish**, ``-final_qwarp``): register the *corrected* series
      (seed 0) to the reference, finding the residual ``r`` the estimator's search
      couldn't resolve; total field is ``w + r``. Sign-safe (no seeding with ``w``).
      Given ``raw_datas`` the returned series is the RAW data warped once by ``w + r``
      rather than qwarp's twice-resampled output; see
      :func:`_rewarp_raw_single_pass`.
    * ``full=True`` (**backend**, ``-backend qwarp``): register the *raw* echoes
      (``raw_datas``, seed 0) to the reference -- qwarp owns the whole field, output is
      just the qwarp field (the flow pass only built the reference). Needs a fuller
      cascade (more levels) since it starts from the full distortion.

    ``slicewise=True`` (the default, for 2-D multi-slice EPI) makes the qwarp patches
    2-D: one voxel thick along the slice axis, in-plane basis only. That matches the
    estimator, which solves each slice independently because each slice is acquired at
    its own instant -- a 3-D patch would smooth the field across ``minpatch`` slices,
    i.e. across acquisition times. Pass ``slicewise=False`` for 3-D-acquired EPI
    (``-is_3dacq`` / ``-me_3depi``), where the volume is one shot and through-plane
    continuity is real.

    ``ref_mode`` picks the temporal reduction that builds the template (``median`` --
    robust to a few bad frames -- ``mean``, or ``max``); ``first``/index fall back to the
    mean, as in :func:`_refine_reduce`, because one frame is a noisy template.

    ``refine`` runs the whole registration more than once, rebuilding the template from
    the series the previous pass corrected. Same idea as the estimator's ``-refine``: the
    first template is built from data that still carries the distortion, so it is blurred
    by it and biases the field LOW. Each pass re-solves from seed 0 against the sharper
    template -- never seeded with the previous field, so a pass can walk a bad step back
    instead of compounding it.

    Both are JOINT: one shared PE field, every echo scored at ``alpha_e·field``.
    Returns a new :class:`MultiEchoLocomocoResult` the CLI writes exactly like the
    estimator's own output. (The single composed warp is ``alpha_e·field``; writing it
    as one field is a downstream concern.)
    """
    from .warp import QwarpConfig, qwarp_pe_scaled_polish_series

    ref = result.per_echo[0]
    e = len(result.per_echo)
    w = result.w_field  # (nx, ny, nz, T) shared field, echo-1 scale
    nx, ny, nz, nt = w.shape
    alpha = result.alpha
    pe_axis = result.pe_axis
    # Two encode axes: the primary-PE field rides alongside, with its own (flat) law.
    dual = result.w_field_pe1 is not None and result.pe_axis1 is not None
    axes = [result.pe_axis1, pe_axis] if dual else [pe_axis]
    w_axes = [result.w_field_pe1, w] if dual else [w]
    # Per-CHANNEL scaling: channel `pe_axis1` is flat across echoes, channel `pe_axis`
    # carries the TE law. A single alpha_e would apply the TE law to both.
    alpha_ch = torch.ones(3, e)
    alpha_ch[pe_axis] = alpha
    if dual:
        alpha_ch[result.pe_axis1] = torch.ones(e)

    # Reference = temporal median of the corrected series (refined template). The
    # moving data is the corrected series (polish) or the raw echoes (full backend).
    dev = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    red = _refine_reduce_label(ref_mode)
    if verbose:
        print(
            f"   ⏳ building reference (temporal {red} of {nt} frames × {e} echoes) "
            "and preparing the series…",
            flush=True,
        )
    corr_series = [result.per_echo[j].corrected_series().float() for j in range(e)]
    if full:
        if raw_datas is None:
            raise ValueError("polish_me_result(full=True) needs raw_datas (the raw echoes).")
        source_series = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(raw_datas[j])).float().permute(2, 1, 0, 3)
                for j in range(e)
            ]
        ).contiguous()
    else:
        source_series = torch.stack([c.permute(2, 1, 0, 3).contiguous() for c in corr_series])
    # One seed per encode axis; zero either way (residual for the polish, the full field
    # from raw for the backend).
    seed_series = torch.zeros(len(axes), nz, ny, nx, nt)

    # Gauss-Newton on the ncc cost: analytic image-gradient Jacobian (no autograd),
    # converges in a few steps. Starts near the solution (polish) or from the full
    # distortion (backend, needs more levels), always from seed 0.
    cfg = QwarpConfig(
        minpatch=minpatch,
        cost_method=cost,
        verb=0,
        batch_optimizer_iters=iters,
        optimizer=optimizer if cost == "ncc" else "adam",
        compile=compile,
    )
    # slice_axis and pe_axis are both NIfTI axes, which the qwarp grid labels the same
    # way (channel 0=x); the (z,y,x) storage order is handled inside the plan.
    slicewise_axis = ref.slice_axis if slicewise else None

    # The template starts from the series the estimator corrected (the raw echoes on the
    # full-backend path, where no estimate ran) and is REBUILT from each pass's output.
    template = corr_series
    w_axes_new: list[torch.Tensor] = []
    r_axes: list[torch.Tensor] = []
    warped_nifti: list[torch.Tensor] = []
    for it in range(int(refine) + 1):
        # The temporal reduction is a full sort per voxel for the median -- slow on CPU;
        # run it on the GPU (each echo transiently, freed after) when we have one.
        base_echoes = torch.stack(
            [_refine_reduce(c.to(dev), ref_mode, 3).permute(2, 1, 0).contiguous() for c in template]
        )
        warped, resid = qwarp_pe_scaled_polish_series(
            base_echoes,
            source_series,
            seed_series,
            axes,  # displacement channel labels are shared between NIfTI-xyz and qwarp grid
            alpha=alpha_ch,
            config=cfg,
            n_levels=n_levels,
            device=device,
            show_progress=verbose,
            slicewise_axis=slicewise_axis,
        )
        del base_echoes
        # (A, nz, ny, nx, T) -> one (nx, ny, nz, T) field per encode axis, echo-1 scale.
        r_axes = [resid[k].permute(2, 1, 0, 3).contiguous() for k in range(len(axes))]
        # Polish adds to the estimator field; the full backend IS the field.
        prev = w_axes_new
        w_axes_new = [r if full else wk + r for wk, r in zip(w_axes, r_axes, strict=True)]
        warped_nifti = [warped[j].permute(2, 1, 0, 3).contiguous() for j in range(e)]
        del warped, resid
        if it < int(refine):
            # Next pass registers the SAME source against a template built from this
            # pass's output — re-solved from seed 0, never seeded with the field above.
            template = warped_nifti
        if verbose and prev:
            d = torch.stack([(a - b) for a, b in zip(w_axes_new, prev, strict=True)])
            moved = d[d != 0]
            rms = float(moved.pow(2).mean().sqrt()) if moved.numel() else 0.0
            print(f"   qwarp refine pass {it}/{int(refine)}: Δfield rms {rms:.4f} vox")
    del template, source_series, seed_series
    w_new = w_axes_new[-1]
    if not full and raw_datas is not None:
        # Prefer one resample of the raw over qwarp's second pass on already-corrected
        # data -- see :func:`_rewarp_raw_single_pass`. Only the polish needs this; the
        # full backend already registers raw.
        del warped_nifti
        warped_nifti = _rewarp_raw_single_pass(
            raw_datas, w_axes_new, list(axes), alpha_ch, dev, verbose
        )

    perm4, pe_flow_is_u = ref.perm, ref.pe_flow_is_u
    a0, a1, disp_slice = ref.a0, ref.a1, ref.slice_axis
    per_echo: list[LocomocoResult] = []
    for j in range(e):
        canon = [
            (float(alpha_ch[ax, j]) * wk).contiguous().permute(perm4).contiguous()
            for ax, wk in zip(axes, w_axes_new, strict=True)
        ]
        by_axis = dict(zip(axes, canon, strict=True))
        zero = torch.zeros_like(canon[0])
        per_echo.append(
            LocomocoResult(
                u_canon=by_axis.get(a1, zero),
                v_canon=by_axis.get(a0, zero),
                corrected_canon=torch.zeros_like(canon[0]),
                perm=perm4,
                pe_flow_is_u=pe_flow_is_u,
                pe_axis=axes[0] if dual else pe_axis,
                slice_axis=disp_slice,
                orig_shape=(nx, ny, nz, nt),
                a0=a0,
                a1=a1,
                dual=dual,
                pe_axis2=axes[-1] if dual else None,
                sep_map=ref.sep_map,
                corrected_nifti=warped_nifti[j],
            )
        )

    if verbose:
        rr = r_axes[-1].abs()
        sel = rr[rr > 0]
        med = float(sel.median()) if sel.numel() else 0.0
        tag = "field |w|" if full else "residual |r|"
        geom = (
            f"2-D {minpatch}×{minpatch} patches, slicewise along {'xyz'[ref.slice_axis]}"
            if slicewise
            else f"3-D {minpatch}³ patches"
        )
        print(
            f"🪄 qwarp {'backend' if full else 'polish'}: {tag} median {med:.3f} vox, "
            f"max {float(rr.max()):.3f} vox ({geom}, {n_levels} levels, "
            f"{cost}/{optimizer if cost == 'ncc' else 'adam'})"
        )

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=result.echo_times,
        w_field=w_new,
        pe_axis=pe_axis,
        linearity_r2=result.linearity_r2,
        w_field_pe1=w_axes_new[0] if dual else None,
        pe_axis1=axes[0] if dual else None,
    )


def estimate_residual_flow_me_interecho(
    datas: list[np.ndarray],
    echo_times: list[float],
    pe_axis: int,
    slice_axis: int,
    *,
    backend: str = "xcorr",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    automask: bool = False,
    automask_sigma: float = 3.0,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    save_corr_curve: int | None = None,
    hpf_sigma: float = 0.0,
    match: str = "localnorm",
    match_sigma: float = 2.0,
    ngf_eta_q: float = 0.5,
    warp_interp: str = "lanczos",
    warp_radius: int = 3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Inter-echo per-TR alignment — align the echo stack, not each echo across time.

    The temporal modes reach across *time*: each echo's frame is registered to that
    echo's own temporal template. This one reaches across *TE* instead — a shorter,
    easier reach. Within each TR, register every echo ``n`` to its lower-TE neighbour
    ``n-1``: same brain, nearest contrast, and only a ``ΔTE``-sized partition shift
    between them. Because that shift is ``ΔTE_n · g(r,t)`` (the same linear-in-TE field),
    all consecutive pairs are POOLED per TR into one per-ms field ``g`` (the shared-field
    kernels, with each pair scaled by its ``ΔTE``), and echo ``n`` is corrected onto echo
    1's frame by ``(TE_n − TE_1) · g``. Echo 1 (shortest TE, assumed ~undistorted) is the
    anchor, left unchanged.

    Because it registers echoes to each other *within* a TR, it needs no temporal
    template and can run BEFORE motion correction — unlike the temporal modes, which
    register each frame to a moco'd template and so require motion-corrected input.

    Needs no temporal averaging and exploits the strong same-TR inter-echo correlation
    (two real volumes per estimate), but it does NOT remove echo 1's own (small) wiggle
    that is common to every echo — follow with a temporal pass (the joint / scaled modes)
    if that residual matters. ``w`` is stored at the echo1→echo2 step scale and ``alpha``
    counts steps from echo 1 (``alpha_1 = 0``); ``linearity_r2`` is 1.0 (imposed).

    The final correction is a one-axis resample of each raw echo. ``warp_interp``
    defaults to Lanczos so a later temporal refine starts from a sharp corrected
    stack rather than a stack already softened by linear interpolation.

    ``backend="flow"`` registers images of DIFFERENT contrast (echo n vs n−1 differ by a
    T2* decay factor everywhere), which breaks LK's brightness-constancy assumption, so
    the pairs are intensity-matched first (``match``, default local z-scoring) and each
    Gauss-Newton step is clamped so the echo1→echo2 step field stays within ``max_shift``
    — the same bound the xcorr searchlight obeys. ``match="none"`` restores the raw
    (divergent) behaviour. ``xcorr`` needs neither: correlation is scale-invariant.

    ``automask`` gates out signal-dropout regions: later echoes lose signal where the
    shorter echo still has it, and a searchlight there rails at ``max_shift``. For each
    pair the *later* echo's temporal mean is automasked and eroded by the searchlight
    radius (``ceil(window_sigma)`` voxels), so we only search where the later echo has
    valid signal; those masks weight each pair's xcorr, and their feathered union gates
    the output ``w``. The model is therefore only trusted out to echo 2's mask floor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = len(datas)
    if e < 2:
        raise ValueError(f"inter-echo needs ≥2 echoes, got {e}.")
    if len(echo_times) != e:
        raise ValueError(f"-echo_times has {len(echo_times)} values but there are {e} inputs.")
    shp = datas[0].shape
    for k, d in enumerate(datas):
        if d.shape != shp:
            raise ValueError(f"echo {k} shape {d.shape} != echo 0 shape {shp}.")
    if len(shp) != 4:
        raise ValueError(f"each -input must be 4D, got {shp}.")
    if backend not in ("flow", "xcorr"):
        raise ValueError(f"inter-echo backend must be flow | xcorr, got {backend!r}.")
    nx, ny, nz, nt = shp
    _validate_estimation_inputs(
        shp,
        pe_axis,
        slice_axis,
        is_3dacq=True,
        max_shift=max_shift,
        trial_step=trial_step,
        window_sigma=window_sigma,
        verbose=verbose,
    )

    te = torch.tensor([float(x) for x in echo_times], dtype=torch.float32)
    dte = te[1:] - te[:-1]  # (E-1,) consecutive-pair gaps = per-pair scaling
    if float(dte.min()) <= 0:
        raise ValueError("-echo_times must be strictly increasing for inter-echo mode.")
    denom = float(te[1] - te[0])  # echo1→echo2 gap: the unit w is stored in

    disp_slice = slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    if verbose:
        print(
            f"🌀 locomoco {_geometry_report(shp, pe_axis, disp_slice, is_3dacq=True)}  "
            f"× {e} echoes (inter-echo)"
        )
        if backend == "flow":
            print(
                f"   cross-TE contrast: match={match} (σ={match_sigma:g} vox), "
                f"GN step clamp |w| ≤ {max_shift:g} vox"
            )

    vols = [torch.from_numpy(np.ascontiguousarray(d)).float() for d in datas]

    # Dropout masking: late echoes lose signal in high-susceptibility regions, where
    # correlating echo n against n-1 is pure noise and the xcorr search rails at
    # ±max_shift. So restrict each pair to where its LATER (moving) echo has signal —
    # a per-echo automask ERODED by the searchlight radius so no window reaches into
    # dropout. Pair j is then weighted by echo (j+1)'s mask; where a late echo has
    # dropped out but an earlier one survives, the earlier pair still carries the shared
    # field (the linear model lets us apply alpha_e·w to the dropped-out echo anyway).
    # The warp's validity floor is thus echo 2's mask (the union of the moving-echo masks).
    pair_masks: list[torch.Tensor] | None = None
    gate: torch.Tensor | None = None
    if automask:
        from .mask import _erode_6conn
        from .mask import automask as _automask

        erode_vox = max(1, int(np.ceil(window_sigma)))
        pair_masks = []
        for j in range(e - 1):
            ref_zyx = vols[j + 1].mean(dim=3).permute(2, 1, 0).contiguous().to(device)  # later echo
            m = _automask(ref_zyx, dilate_extra=0, device=device)
            m = _erode_6conn(m, erode_vox)
            pair_masks.append(m.permute(2, 1, 0).contiguous().float())  # (nx, ny, nz)
        union = torch.stack(pair_masks, 0).amax(0)  # ≈ echo-2 mask (largest, least dropout)
        gate = _gaussian_blur3d(union, automask_sigma).cpu()  # feather the warp to 0 at edges
        if verbose:
            frac = float(union.mean())
            print(
                f"   dropout mask: per-pair later-echo automask eroded {erode_vox} vox "
                f"(window {window_sigma:g}); warp valid in {frac * 100:.0f}% of FoV (echo-2 floor)"
            )

    w = torch.zeros(nx, ny, nz, nt)
    conf = torch.zeros(nx, ny, nz, nt) if backend == "xcorr" else None
    curve_frame = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
    corr_curve: torch.Tensor | None = None
    corr_offsets: torch.Tensor | None = None

    # Estimation prep: optional spatial high-pass (raw kept for the resample), then the
    # existing estimation blur. Identity when both sigmas are 0. v is a single (X,Y,Z) vol.
    def _prep(v: torch.Tensor) -> torch.Tensor:
        x = _spatial_highpass3d(v[None], hpf_sigma)
        if smooth_sigma > 0:
            x = _blur3d_b(x, smooth_sigma)
        return x[0]

    for t in tqdm(range(nt), desc="me interecho", unit="frame", leave=True, disable=nt < 3):
        ims = [_prep(vols[j][..., t].to(device)) for j in range(e)]
        fixed_list = [ims[j][None] for j in range(e - 1)]  # echo n-1 (lower TE)
        moving_list = [ims[j + 1][None] for j in range(e - 1)]  # echo n
        if backend == "flow":
            # LK has no per-pair weighting yet; dropout is handled by gating the output.
            # g is per-ms, so the max_shift bound (an echo1→echo2 step) divides by denom.
            g = optical_flow_lk_3d_multiecho(
                fixed_list,
                moving_list,
                dte,
                pe_axis,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
                match=match,
                match_sigma=match_sigma,
                ngf_eta_q=ngf_eta_q,
                max_disp=max_shift / denom,
            )[0]
        else:
            weights = [pm[None] for pm in pair_masks] if pair_masks is not None else None
            curve_acc: list[torch.Tensor] | None = [] if t == curve_frame else None
            g_be, c_be = xcorr_search_flow_3d_multiecho(
                fixed_list,
                moving_list,
                dte,
                pe_axis,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                weights=weights,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                peak_mode=peak_mode,
                search_min_steps=search_min_steps,
                curve_out=curve_acc,
            )
            g = g_be[0]
            if conf is not None:
                conf[..., t] = c_be[0].cpu()
            if curve_acc is not None:
                # (nd,B,X,Y,Z) → (X,Y,Z,nd): a 4-D per-voxel search landscape for one TR.
                corr_curve = torch.stack(curve_acc, 0)[:, 0].permute(1, 2, 3, 0).contiguous()
                rr = float(max(1, int(np.ceil(max_shift))))
                corr_offsets = torch.arange(-rr, rr + 1e-6, trial_step)
        # g is the per-ms field (both kernels return s/alpha_max = per-unit-alpha);
        # store w at the echo1→echo2 step so |w| is an interpretable inter-echo shift.
        w[..., t] = (denom * g).cpu()

    if gate is not None:
        # Zero (feathered) the warp outside the signal region so masked-out dropout
        # voxels — where the search railed — carry no correction.
        w = w * gate[..., None]
        if conf is not None:
            conf = conf * gate[..., None]

    # alpha counts echo1→echo2 steps: (TE_e − TE_1)/(TE_2 − TE_1). alpha_1 = 0 (anchor).
    alpha = (te - float(te[0])) / denom
    per_echo: list[LocomocoResult] = []
    for j in range(e):
        disp_e = (alpha[j] * w).contiguous()
        corrected = torch.zeros(nx, ny, nz, nt)
        for t in range(nt):
            corrected[..., t] = _shift3d_axis(
                vols[j][..., t].to(device)[None],
                disp_e[..., t].to(device)[None],
                pe_axis,
                mode=warp_interp,
                radius=warp_radius,
            )[0].cpu()
        p_canon = disp_e.permute(perm4).contiguous()
        u_canon = p_canon if pe_flow_is_u else torch.zeros_like(p_canon)
        v_canon = torch.zeros_like(p_canon) if pe_flow_is_u else p_canon
        per_echo.append(
            LocomocoResult(
                u_canon=u_canon,
                v_canon=v_canon,
                corrected_canon=torch.zeros_like(p_canon),
                perm=perm4,
                pe_flow_is_u=pe_flow_is_u,
                pe_axis=pe_axis,
                slice_axis=disp_slice,
                orig_shape=(nx, ny, nz, nt),
                a0=a0,
                a1=a1,
                dual=False,
                corrected_nifti=corrected,
            )
        )

    if verbose:
        ap = w.abs()
        sel = ap[ap > 0]
        med = float(sel.median()) if sel.numel() else 0.0
        te_str = ", ".join(f"{float(x):.0f}" for x in te)
        a_str = ", ".join(f"{float(x):.2f}" for x in alpha)
        amax = float(ap.max())
        print(
            f"🌀 locomoco ME-3D inter-echo: {e} echoes (TE {te_str} ms), {nt} frames, "
            f"PE axis {pe_axis}, backend={backend}; echo1→echo2 step |w| median {med:.3f} vox, "
            f"max {amax:.3f} vox (echo 1 = anchor, uncorrected)"
        )
        print(f"   alpha (echo1→echo2 steps) = [{a_str}]  ·  pooled {e - 1} adjacent-echo pairs/TR")
        # Per-echo CUMULATIVE correction = alpha_e · w — this is where the TE scaling shows:
        # each echo's applied displacement is its step count times the shared per-step field.
        disp_str = "  ".join(f"e{j + 1}={float(alpha[j]) * med:.3f}" for j in range(e))
        print(f"   applied |disp| median (vox): {disp_str}   (= alpha_e · step field)")

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=w,
        pe_axis=pe_axis,
        linearity_r2=1.0,
        confidence=conf,
        corr_curve=corr_curve,
        corr_offsets=corr_offsets,
    )


def _nonzero_median(mag: torch.Tensor) -> float:
    """Median of the non-zero entries — a magnitude summary a masked-out field can't dilute."""
    sel = mag[mag > 0]
    return float(sel.median()) if sel.numel() else 0.0


def _affine_in_te_r2(disps: torch.Tensor, te: torch.Tensor) -> float:
    """How well ``disp_e(r,t) = A(r,t) + TE_e·B(r,t)`` explains a per-echo warp stack.

    ``disps`` is ``(E, nx, ny, nz, T)``. The TE model every multi-echo mode enforces is
    a per-echo SCALING of shared fields, so the per-echo warps must lie in a 2-D family
    spanned by ``1`` and ``TE`` — proportional-to-TE for a temporal solve (echo 1 moves
    too), proportional-to-(TE − TE_1) for an inter-echo solve (echo 1 is the anchor), and
    the composition of the two is the general affine case. r² is the variance-weighted
    fraction explained across the whole field, so it is a genuine model check: 1.0 means
    the echoes really are one field scaled, well below 1 means something has broken the
    coupling and the echoes are drifting apart.
    """
    e = disps.shape[0]
    if e < 3:
        return 1.0  # two echoes always fit a 2-parameter model exactly
    x = torch.stack([torch.ones_like(te), te], dim=1).double()  # (E, 2)
    resid_op = torch.eye(e, dtype=torch.float64) - x @ torch.linalg.pinv(x.T @ x) @ x.T
    centre = torch.eye(e, dtype=torch.float64) - 1.0 / e
    ss_res = ss_tot = 0.0
    for t in range(disps.shape[-1]):  # per frame: bounds the temporary to one volume
        y = disps[..., t].reshape(e, -1).double()
        ss_res += float((resid_op @ y).pow(2).sum())
        ss_tot += float((centre @ y).pow(2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0


# Per-echo scaling laws available to the inter-echo refine pass, and the sequence of
# pooled solves each one runs. The residual left by the inter-echo pass has TWO parts —
# the anchor leftover TE_1·g (flat: echo 1's own distortion, which inter-echo can never
# see) and the ladder error (TE_e − TE_1)·δ from an imperfectly estimated g (zero at the
# anchor, growing with TE). No single per-echo scaling spans both, so the default runs
# one pooled solve per component: 'step' first (it dominates the late echoes, which carry
# the largest, best-conditioned signal), then 'flat'. Together they span the full
# affine-in-TE family the residual actually lives in.
_REFINE_LAWS: dict[str, tuple[str, ...]] = {
    "affine": ("step", "flat"),
    "step": ("step",),
    "flat": ("flat",),
    "te": ("te",),
    "learn": ("learn",),
}
_LAW_BLURB = {
    "step": "ladder error ∝ (TE_e − TE₁)",
    "flat": "anchor leftover TE₁·g, flat across echoes",
    "te": "∝ TE_e",
    "learn": "learned α",
}


def _refine_alpha(law: str, te: torch.Tensor) -> torch.Tensor | None:
    """Per-echo scaling vector for a refine law; ``None`` for ``learn`` (fitted from data)."""
    if law == "learn":
        return None
    if law == "step":  # the inter-echo ladder: zero at the anchor, growing with TE
        return (te - float(te[0])) / float(te[1] - te[0])
    if law == "flat":
        return torch.ones_like(te)
    if law == "te":
        return te / float(te[0])
    raise ValueError(f"unknown refine law {law!r}")


def _orthogonalize_alpha(a: torch.Tensor, used: list[torch.Tensor]) -> torch.Tensor:
    """Gram-Schmidt a scaling vector against the laws already solved for.

    Sequential solves only add up to the joint one if their per-echo scalings are
    orthogonal. ``step`` = [0,1,2,3,4] and ``flat`` = [1,1,1,1,1] are far from it
    (cos ≈ 0.82), so a naive step-then-flat pair lets the first solve absorb part of the
    second's component and leaves it there: against a purely common-mode residual the
    step pass takes ``Σα/Σα²`` of it, and echo 1 ends up ~⅓ corrected. Replacing the
    second law by its component orthogonal to the first makes one round of each EXACT —
    it spans the same affine family (only the basis changes), so the composed warp is
    unaffected, but the split between the two solves is no longer biased.
    """
    out = a.clone()
    for p in used:
        denom = float((p * p).sum())
        if denom > 1e-12:
            out = out - float((out * p).sum()) / denom * p
    scale = float(out.abs().max())
    return out / scale if scale > 1e-12 else out


def refine_interecho_temporally(
    result: MultiEchoLocomocoResult,
    datas: list[np.ndarray],
    echo_times: list[float],
    pe_axis: int,
    slice_axis: int,
    *,
    refine_rounds: int = 1,
    ref_mode: str = "mean",
    backend: str = "flow",
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    max_shift: float = 3.0,
    trial_step: float = 0.5,
    converge: float = 0.0,
    converge_rel: float = 0.0,
    first_n: int | None = None,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    coverage_erode: int | None = 1,
    scaling: str = "affine",
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    search_min_steps: int = 5,
    warp_interp: str = "lanczos",
    warp_radius: int = 3,
    hpf_sigma: float = 0.0,
    match: str = "none",
    match_sigma: float = 6.0,
    ngf_eta_q: float = 0.5,
    want_corrected: bool = True,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Second pass: a temporal joint solve seeded by the inter-echo correction.

    The inter-echo solve is blind exactly where its own masking makes it blind. Every
    pair is gated by the LATER echo's eroded automask, so at the brain edge — where the
    late echo has dropped out but the early echoes have not — there is no estimate, and
    the feathered gate rolls the field to zero. Worse, the surviving edge windows see a
    later echo that is *dimmer at its rim*, which a displacement search reads as
    shrinkage. Inside the brain it is the better estimator (two real volumes per
    estimate, a short ΔTE reach, no template); at the rim it is not.

    So run :func:`estimate_residual_flow_multiecho` on the inter-echo-CORRECTED series.
    That pass has a temporal template rather than a cross-echo one, needs no dropout
    mask, and sees the edge voxels the inter-echo pass had to throw away; because the
    stack is already aligned, what is left for it is echo 1's own (common-mode) wiggle
    plus the rim the first pass could not reach. Any real edge displacement is now a
    large error in the EARLY echoes too, which is what makes the temporal pass able to
    pull it back.

    Nothing is masked to "the voxels the first pass missed" — every solve here is a full
    joint pass over the whole volume and every echo, with the inter-echo result acting as
    an INITIALISATION rather than a boundary. A gated-region-only refine would just draw a
    second seam inside the first one.

    ``scaling`` picks the model for the leftover, which has two components and therefore
    is not describable by any single per-echo scaling:

    - ``TE_1·g`` — echo 1's own distortion. Inter-echo aligns echoes to each other, never
      to undistorted anatomy, so this survives on EVERY echo identically (flat). With
      ``TE_1 = 7.45`` ms against ~13 ms spacing it is over half an inter-echo step.
    - ``(TE_e − TE_1)·δ`` — the ladder error from estimating ``ĝ`` instead of ``g``. If
      the true step was 1.5 vox and the pass found 1.0, echo 3 was moved 2.0 instead of
      3.0 and echo 2 was moved 1.0 instead of 1.5: the shortfall grows with TE and is
      exactly zero at the anchor. Note it scales as ``TE_e − TE_1``, not ``TE_e``.

    So the default ``"affine"`` runs TWO pooled solves — ``"step"`` (``∝ TE_e − TE_1``)
    then ``"flat"`` — which together span the affine family the residual lives in. Each
    is a genuine single solve over the whole stack (one shared-parameter searchlight with
    every echo trial-shifted by ``alpha_e·s`` at once, or one pooled Gauss-Newton), and
    each round re-derives its input from the RAW data through the composed warp, so the
    stack the second solve sees is one interpolation deep, not two.

    Single-law modes ``"step"`` / ``"flat"`` / ``"te"`` run just that component. ``"learn"``
    fits alpha from the data instead and is the only mode that is NOT one pooled solve:
    rank-1 factoring needs each echo estimated separately first, then pools in warp space.
    Use it to data-check which law the leftover actually follows.

    Fields are composed rather than summed — all are pull warps along the same axis, so
    ``total(x) = new(x) + old(x + new(x))``. The per-echo warps returned are exact; the
    returned ``alpha`` / ``w_field`` are a rank-1 summary of them (a single ``alpha·w``
    cannot hold a two-component field), and ``linearity_r2`` is the affine-in-TE model
    check from :func:`_affine_in_te_r2`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = len(datas)
    nx, ny, nz, nt = datas[0].shape
    proto = result.per_echo[0]

    if scaling not in _REFINE_LAWS:
        raise ValueError(f"refine scaling must be one of {tuple(_REFINE_LAWS)}, got {scaling!r}.")
    te = result.echo_times
    laws = _REFINE_LAWS[scaling]

    if verbose:
        # Name the actual solve, because "joint" means different things per mode: only the
        # fixed-alpha paths are ONE search over all echoes. Learning alpha costs a per-echo
        # pass (that is what factors the ratios) and pools in warp space after.
        how = (
            "per-echo passes + rank-1 projection (learning alpha)"
            if "learn" in laws and backend != "flow"
            else f"{len(laws)} pooled solve(s), each over all echoes at once"
        )
        print(
            f"🔁 inter-echo refine: temporal pass over all {e} echoes on the corrected stack "
            f"— {how} (ref={ref_mode}, backend={backend}, refine={refine_rounds}, "
            f"scaling={scaling}: {' then '.join(_LAW_BLURB[law] for law in laws)})"
        )

    vols = [torch.from_numpy(np.ascontiguousarray(d)).float() for d in datas]
    # Running state: the exact per-echo warp so far (seeded from the inter-echo pass) and
    # the series it produces from the RAW data. Re-deriving the series from `vols` each
    # round keeps the corrected stack one interpolation deep no matter how many laws run.
    totals = torch.stack([(result.alpha[j] * result.w_field).contiguous() for j in range(e)], 0)
    cur = [r.corrected_series() for r in result.per_echo]

    def _apply(j: int, disp: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(nx, ny, nz, nt)
        for t in range(nt):
            out[..., t] = _shift3d_axis(
                vols[j][..., t].to(device)[None],
                disp[..., t].to(device)[None],
                pe_axis,
                mode=warp_interp,
                radius=warp_radius,
            )[0].cpu()
        return out

    res_rf = None
    used_alphas: list[torch.Tensor] = []
    for law in laws:
        a_law = _refine_alpha(law, te)
        if a_law is not None:
            a_law = _orthogonalize_alpha(a_law, used_alphas)
            used_alphas.append(a_law)
            if verbose and len(used_alphas) > 1:
                print(
                    f"   {law} law orthogonalized against the previous solve → alpha = "
                    f"[{', '.join(f'{float(x):.3f}' for x in a_law)}]"
                )
        res_rf = estimate_residual_flow_multiecho(
            [c.numpy() for c in cur],
            echo_times,
            pe_axis,
            slice_axis,
            ref_mode=ref_mode,
            backend=backend,
            smooth_sigma=smooth_sigma,
            n_levels=n_levels,
            n_iters=n_iters,
            window_sigma=window_sigma,
            max_shift=max_shift,
            trial_step=trial_step,
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            first_n=first_n,
            automask=automask,
            automask_dilate=automask_dilate,
            automask_sigma=automask_sigma,
            coverage_erode=coverage_erode,
            learn_scaling=a_law is None,
            flat_scaling=False,
            alpha_override=a_law,
            noshift_margin=noshift_margin,
            reg_sigma=reg_sigma,
            peak_mode=peak_mode,
            search_min_steps=search_min_steps,
            want_corrected=False,
            warp_interp=warp_interp,
            warp_radius=warp_radius,
            hpf_sigma=hpf_sigma,
            match=match,
            match_sigma=match_sigma,
            ngf_eta_q=ngf_eta_q,
            device=device,
            verbose=verbose,
        )
        # Compose this round's field onto the running total: both are pull warps along the
        # same axis, so total(x) = new(x) + old(x + new(x)) — NOT a sum.
        for j in range(e):
            d_new = res_rf.alpha[j] * res_rf.w_field
            for t in range(nt):
                nw = d_new[..., t].to(device)[None]
                old = _shift3d_axis(totals[j, ..., t].to(device)[None], nw, pe_axis)
                totals[j, ..., t] = (nw + old)[0].cpu()
            cur[j] = _apply(j, totals[j])

    assert res_rf is not None  # _REFINE_LAWS never maps to an empty sequence
    per_echo: list[LocomocoResult] = []
    for j in range(e):
        corrected = cur[j] if want_corrected else None
        p_canon = totals[j].permute(proto.perm).contiguous()
        u_canon = p_canon if proto.pe_flow_is_u else torch.zeros_like(p_canon)
        v_canon = torch.zeros_like(p_canon) if proto.pe_flow_is_u else p_canon
        per_echo.append(
            LocomocoResult(
                u_canon=u_canon,
                v_canon=v_canon,
                corrected_canon=torch.zeros_like(p_canon),
                perm=proto.perm,
                pe_flow_is_u=proto.pe_flow_is_u,
                pe_axis=pe_axis,
                slice_axis=proto.slice_axis,
                orig_shape=(nx, ny, nz, nt),
                a0=proto.a0,
                a1=proto.a1,
                dual=False,
                corrected_nifti=corrected,
            )
        )

    te = result.echo_times
    alpha, w_flat = _rank1_factor_echoes(totals.reshape(e, -1), te / float(te[0]))
    # Re-normalise onto the LARGEST echo. _rank1_factor_echoes divides by alpha_1, which is
    # the one echo guaranteed to be near-zero here (the inter-echo anchor barely moves), so
    # its default scaling reports a readable field as alpha = [1, 17, 33, ...].
    amax = float(alpha.abs().max())
    if amax > 1e-12:
        alpha, w_flat = alpha / amax, w_flat * amax
    w = w_flat.reshape(nx, ny, nz, nt)
    # The TE model check for the COMPOSED warp. Each pass is a per-echo scaling of one
    # shared field, but with different TE anchors — inter-echo ∝ (TE − TE_1) (echo 1 is
    # the anchor), temporal ∝ TE (echo 1 moves too) — so the sum is affine in TE, not
    # proportional to it. r² of the affine fit is therefore the invariant to watch: it
    # says the echoes are still ONE field scaled by echo time. A single alpha·w rank-1
    # summary cannot represent two components, so its own r² is reported separately and
    # is expected to sit below 1 whenever both passes contributed.
    r2 = _affine_in_te_r2(totals, te)

    if verbose:
        a_str = ", ".join(f"{float(x):.3f}" for x in alpha)
        for j in range(e):
            ie_med = _nonzero_median((result.alpha[j] * result.w_field).abs())
            tot_med = _nonzero_median(totals[j].abs())
            print(
                f"   echo {j + 1}: |disp| median {ie_med:.3f} → {tot_med:.3f} vox "
                f"(max {float(totals[j].abs().max()):.3f})"
            )
        rf_law = " ⊕ ".join(_LAW_BLURB[law] for law in laws)
        print(
            f"   composed warp is affine-in-TE (inter-echo ∝ TE−TE₁ ⊕ {rf_law}): "
            f"r²={r2:.5f}  ← 1.0 = the per-echo TE scaling still holds exactly"
        )
        print(
            f"   rank-1 summary alpha (÷largest) = [{a_str}] — the saved alpha/w only; the "
            "per-echo warps written out are the exact composed fields."
        )

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=w,
        pe_axis=pe_axis,
        linearity_r2=r2,
        confidence=res_rf.confidence,
        corr_curve=res_rf.corr_curve,
        corr_offsets=res_rf.corr_offsets,
        alpha_label="alpha(÷largest echo)",
    )


def _warp_matrix(components, device=None, dtype=torch.float64):
    """``(x, mean, xc, shapes, axes, widths)`` — the axes' fields as one ``(T, S)`` matrix.

    Shared by the PCA and ICA decompositions so both see exactly the same columns in
    the same order; the per-axis split on the way out is then just a column range.
    """
    mats, shapes, axes = [], [], []
    for axis, disp in components:
        if float(disp.abs().max()) == 0.0:
            continue
        t = disp.shape[-1]
        mats.append(disp.contiguous().reshape(-1, t).T.contiguous())  # (T, spatial)
        shapes.append(tuple(disp.shape[:3]))
        axes.append(axis)
    if not mats:
        return None
    widths = [m.shape[1] for m in mats]
    x = torch.cat(mats, dim=1).to(dtype)
    if device is not None:
        x = x.to(device)
    mean = x.mean(dim=0, keepdim=True)
    return x, mean, x - mean, shapes, axes, widths


def _split_loadings(load, mean, shapes, axes, widths, k):
    """``(k, S)`` loadings and a ``(1, S)`` mean back into per-axis 4-D / 3-D blocks."""
    loadings, means, start = [], [], 0
    for axis, shape, w in zip(axes, shapes, widths, strict=True):
        block = load[:, start : start + w].T.reshape(*shape, k)
        loadings.append((axis, block.contiguous().cpu()))
        means.append(mean[0, start : start + w].reshape(shape).contiguous().cpu())
        start += w
    return loadings, means


def warp_reconstruct(basis, loadings, means, keep):
    """``mean + basis[:, keep] @ loading[keep]`` per axis — one formula for PCA and ICA.

    The two decompositions differ only in what the pair means. For PCA ``basis`` is the
    orthonormal temporal factor and the loading is its projection, so the product is an
    orthogonal projection onto the kept components. For ICA ``basis`` is the mixing
    matrix and the loading is the independent spatial map, so the product is the same
    sum of outer products with a non-orthogonal basis. Either way, reconstruction is
    "add back the components you kept", and the per-voxel temporal MEAN is always
    restored -- it is not a component, and dropping it would move every voxel by its
    own average displacement.
    """
    out = []
    for (axis, load), mean in zip(loadings, means, strict=True):
        k_all = load.shape[-1]
        idx = sorted(set(keep.get(axis, range(k_all))))
        shape = tuple(load.shape[:3])
        n_t = basis.shape[0]
        if idx:
            sel = torch.tensor(idx, dtype=torch.long)
            recon = (load.reshape(-1, k_all)[:, sel] @ basis[:, sel].T).reshape(*shape, n_t)
        else:
            recon = torch.zeros(*shape, n_t, dtype=load.dtype)
        out.append((axis, (recon + mean[..., None]).float()))
    return out


def warp_project_out(components, bad_timecourses, device=None):
    """Regress a set of time courses out of the FULL-rank field, per axis.

    The alternative to reconstructing from kept components, and strictly better when
    the goal is rejection rather than denoising. A decomposition is used only to FIND
    the offending time courses; removing them is then a projection on the untruncated
    data, so everything the decomposition did not model survives untouched. Rebuilding
    from an ICA basis instead discards whatever the PCA reduction dropped -- measured on
    a real run, a 95%-variance ICA cost 21.5% of the field's rms while rejecting
    nothing at all.

    ``bad_timecourses`` is ``{axis: (T, m) tensor}``; an axis with no entry is returned
    unchanged. Per axis because a component can be contaminated on one encode axis and
    clean on the other.

    Least squares, not dot products: independent components are not orthogonal, so
    subtracting each one's projection separately would remove their shared part more
    than once.
    """
    out = []
    for axis, disp in components:
        bad = bad_timecourses.get(axis)
        if bad is None or bad.shape[1] == 0:
            out.append((axis, disp.float()))
            continue
        shape = tuple(disp.shape[:3])
        n_t = disp.shape[-1]
        x = disp.reshape(-1, n_t).T.to(torch.float64)  # (T, S)
        if device is not None:
            x = x.to(device)
        a = bad.to(dtype=torch.float64, device=x.device)
        # Centre before projecting and restore after: the per-voxel temporal mean is
        # not part of any component, and a time course with any DC left in it would
        # otherwise drag the mean displacement with it.
        mean = x.mean(dim=0, keepdim=True)
        xc = x - mean
        q, _ = torch.linalg.qr(a - a.mean(dim=0, keepdim=True))
        clean = xc - q @ (q.T @ xc) + mean
        out.append((axis, clean.T.reshape(*shape, n_t).contiguous().float().cpu()))
    return out


def warp_ica_basis(components, n_components=None, pca_components=0.95, device=None):
    """Independent spatial components of a per-frame warp, in the same shape as the PCs.

    Returns ``(mixing (T,k), loadings [(axis, (nx,ny,nz,k))], means, var_ratio_or_None)``
    -- deliberately the tuple :func:`warp_pc_basis` returns, so :func:`warp_reconstruct`
    and the rejection scorer take either without a branch.

    Why ICA is here at all. PCA orders by VARIANCE, so contamination worth a fraction
    of a percent of the field cannot surface in any component. Measured on a 0.8mm
    checkerboard run whose field was 8.8x task-enriched at its tail: no principal
    component exceeded 1.25x enrichment or 0.07 correlation with the design, while the
    best INDEPENDENT component reached 2.8-3.0x enrichment with a timecourse
    correlating 0.68 with the design. Independence is not variance, and that is the
    whole difference.

    The rank matters and has an interior optimum -- 20 components missed it (1.75x),
    60 found it (2.79x), the full 119 over-split it back down (2.27x). ``pca_components``
    is therefore a real knob and not a formality; it defaults to a variance fraction
    rather than a count so it travels between runs.
    """
    from ..decomposition.ica import FastICA

    built = _warp_matrix(components, device=device, dtype=torch.float32)
    if built is None:
        return None
    x, mean, _xc, shapes, axes, widths = built
    ica = FastICA(
        n_components=n_components,
        pca_components=pca_components,
        random_state=0,
        device=device,
    )
    ica.fit_transform(x)
    if ica.components_ is None or ica.mixing_ is None:
        raise RuntimeError("FastICA returned no components for the warp field")
    load = ica.components_.to(torch.float64).cpu()  # (k, S) independent spatial maps
    mixing = ica.mixing_.to(torch.float64).cpu()  # (T, k) their time courses
    k = load.shape[0]
    loadings, means = _split_loadings(load, mean.double().cpu(), shapes, axes, widths, k)
    # ICA components are not variance-ordered and carry no eigenvalue, so the share of
    # the field each one explains is measured directly rather than read off a spectrum.
    energy = (load**2).sum(dim=1) * (mixing**2).sum(dim=0)
    var = energy / energy.sum().clamp(min=1e-30)
    return mixing, loadings, means, var


def warp_pc_basis(components, n_pcs=None, device=None):
    """Shared temporal PCs of a per-frame warp, WITH the per-axis spatial loadings.

    ``warp_time_pcs`` returns the scores alone, which is all a nuisance regressor
    needs; reconstruction and rejection need the loadings too, and both need them
    split back per encode axis.

    The temporal basis is deliberately SHARED across the axes -- the same
    concatenation ``warp_time_pcs`` already does. Measured on a two-axis 0.8mm run, a
    per-axis basis buys 0.09 points of explained variance on the primary PE and 0.22
    on the partition, because the dominant temporal modes (respiration, drift) are
    common even when the two fields' spatial patterns are not. The coupling report on
    that run says the axes are "largely independent", which is about the per-voxel
    displacement RATIO and is not in conflict.

    Sharing the basis costs nothing and constrains nothing: every component carries
    its OWN spatial loading per axis, so a component can be dropped from one axis and
    kept on the other. Returns ``(scores (T,k), loadings [(axis, (nx,ny,nz,k))],
    means [(nx,ny,nz)], var_ratio (k,))``, or None if the warp is empty.
    """
    built = _warp_matrix(components, device=device)
    if built is None:
        return None
    x, mean, xc, shapes, axes, widths = built
    n_t = x.shape[0]
    # Economy SVD on (T, S) with T ~ 100 and S in the millions: the left factor is what
    # we want and torch computes it without ever forming an S x S covariance.
    u, sv, _ = torch.linalg.svd(xc, full_matrices=False)
    k_max = int(min(n_t - 1, min(widths), sv.numel()))
    k = k_max if n_pcs is None else max(1, min(int(n_pcs), k_max))
    u = u[:, :k].contiguous()
    var = (sv[:k] ** 2) / (sv**2).sum().clamp(min=1e-30)

    load = u.T @ xc  # (k, S) -- component loadings over the pooled columns
    loadings, means = _split_loadings(load, mean, shapes, axes, widths, k)
    return u.cpu(), loadings, means, var.cpu()


def warp_pc_reconstruct(components, keep, n_pcs=None, device=None):
    """Rebuild each axis's 4-D field from a subset of the shared temporal PCs.

    ``keep`` is ``{axis: [component indices]}`` -- per axis, because a component can be
    contaminated on one encode axis and clean on the other. The per-voxel temporal MEAN
    is always restored: it is not a principal component, and dropping it would move
    every voxel by its own average displacement.
    """
    got = warp_pc_basis(components, n_pcs=n_pcs, device=device)
    if got is None:
        return None
    u, loadings, means, _var = got
    return warp_reconstruct(u, loadings, means, keep)


def warp_time_pcs(
    components: list[tuple[int, torch.Tensor]],
    n_pcs: int = 5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Temporal principal components of a per-frame warp — nuisance regressors.

    The residual-motion warp is a structured spatiotemporal field; its dominant
    temporal patterns make excellent denoising regressors (the same thing
    :mod:`ffs_util_pcwarp` extracts post-hoc, computed here in-line from the warp we
    already have). ``components`` is the ``[(nifti_axis, disp(nx,ny,nz,T))]`` list from
    :meth:`LocomocoResult.warp_components`; all-zero axes are dropped. Returns
    ``(scores (T, k), explained_variance_ratio (k,))`` — scores normalised to unit
    variance for direct use as .1D regressors — or ``(None, None)`` if the warp is empty.
    """
    from ..decomposition.pca import PCA

    mats = []
    for _axis, disp in components:
        if float(disp.abs().max()) == 0.0:
            continue
        t = disp.shape[-1]
        mats.append(disp.contiguous().reshape(-1, t).T.contiguous())  # (T, spatial)
    if not mats:
        return None, None
    mat = torch.cat(mats, dim=1).float()
    nt, nfeat = mat.shape
    k = max(1, min(n_pcs, nt - 1, nfeat))
    pca = PCA(n_components=k, device=device)
    scores = pca.fit_transform(mat.to(device) if device is not None else mat)
    sc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    return (scores / sc_std).cpu(), pca.explained_variance_ratio_[:k].cpu()
