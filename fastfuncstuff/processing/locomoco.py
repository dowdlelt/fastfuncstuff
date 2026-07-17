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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Phase-encode direction letters -> NIfTI spatial axis (x=0, y=1, z=2). Only the
# AXIS matters for building the movie / placing the warp component; the sign of
# the displacement is data-driven (it falls out of the optical flow).
PE_DIR_TO_AXIS: dict[str, int] = {
    "LR": 0,
    "RL": 0,
    "x": 0,
    "AP": 1,
    "PA": 1,
    "y": 1,
    "IS": 2,
    "SI": 2,
    "z": 2,
}


# ── low-level optical-flow primitives (batched, GPU) ──────────────────────────
def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


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
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return x.squeeze(1)


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


def _warp2d(
    img: torch.Tensor, u: torch.Tensor, v: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Sample ``img`` at ``(x + u, y + v)`` (u along W, v along H). ``(B, H, W)``.

    ``mode`` is the grid_sample interpolator (``bilinear`` or ``bicubic``). Bilinear
    damps a fractionally-shifted image, which biases the fixed point the estimation
    iterations converge to; ``bicubic`` resamples faithfully so they land on the true
    displacement — the accuracy win is largest on smooth data.
    """
    _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    gxn = 2.0 * (xs.unsqueeze(0) + u) / max(w - 1, 1) - 1.0
    gyn = 2.0 * (ys.unsqueeze(0) + v) / max(h - 1, 1) - 1.0
    grid = torch.stack([gxn, gyn], dim=-1)
    out = F.grid_sample(
        img.unsqueeze(1), grid, mode=mode, padding_mode="border", align_corners=True
    )
    return out.squeeze(1)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense pyramidal Lucas-Kanade optical flow for a batch of 2-D image pairs.

    Solves, per pixel, for the displacement ``(u, v)`` (u along W, v along H) that
    pulls ``moving`` onto ``fixed`` (``moving(x + flow) ≈ fixed(x)``) — the sign
    convention for undoing motion by resampling ``moving`` at ``x + flow``.

    ``pe_only_axis`` (0 = W/columns, 1 = H/rows) constrains the flow to one axis
    (residual EPI motion is ~1-D along PE): a 1-DOF, more robust estimate.
    Returns ``(u, v)``; a constrained-out component is all zeros.
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
            mv_w = _warp2d(mv_l, u, v, warp_interp)
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
    ncc, r, trial_step, noshift_margin, reg_sigma, ambiguity_frac, blur, curve_out
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outward-from-zero sweep + first-peak finder (the B+C path).

    ``ncc(s)`` returns the per-voxel local correlation at offset ``s``. Sweeps
    ``s = 0, ±step, ±2·step, …`` and **grows the range only while some voxel is still
    rising** — once none is, every voxel has passed its first peak, so we stop (the
    data-adaptive speedup: mostly-small shifts search ~±1, not ±r). When ``curve_out`` is
    requested the full ±r range is swept so the saved landscape is complete.
    """
    corr0 = ncc(0.0)
    pos = [corr0]
    neg = [corr0]
    pos_rising = torch.ones_like(corr0, dtype=torch.bool)
    neg_rising = torch.ones_like(corr0, dtype=torch.bool)
    kmax = int(round(r / trial_step))
    early = curve_out is None
    for k in range(1, kmax + 1):
        s = k * trial_step
        cp, cn = ncc(s), ncc(-s)
        pos_rising = pos_rising & (cp > pos[-1])
        neg_rising = neg_rising & (cn > neg[-1])
        pos.append(cp)
        neg.append(cn)
        if early and not bool((pos_rising | neg_rising).any()):
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
        best_i, best_val, zero_val, ym2, ym1, y0, yp1, yp2,
        nd=nd, r=r, trial_step=trial_step, noshift_margin=noshift_margin,
        reg_sigma=reg_sigma, blur=lambda x: _blur2d(x, reg_sigma),
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
            fixed, moving, pe_is_u, max_shift, window_sigma, trial_step, interp, eps,
            noshift_margin, reg_sigma, curve_out=curve_out,
        )
        if conf_out is not None:
            conf_out.append(conf)
        z = torch.zeros_like(s)
        return (s, z) if pe_is_u else (z, s)

    u = torch.zeros_like(fixed)
    v = torch.zeros_like(fixed)
    for _ in range(n_passes):
        mw = _warp2d(moving, u, v, warp_interp)
        du, _ = _xcorr_shift_1d(
            fixed, mw, True, max_shift, window_sigma, trial_step, interp, eps,
            noshift_margin, reg_sigma,
        )
        u = u + du
        mw = _warp2d(moving, u, v, warp_interp)
        dv, _ = _xcorr_shift_1d(
            fixed, mw, False, max_shift, window_sigma, trial_step, interp, eps,
            noshift_margin, reg_sigma,
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
    ppe = p  # patch extent along PE (patches are p×p; PE axis moved to dim=1 below)

    # The cross-power spectrum holds several (frames·lh·lw, p, p) complex/real
    # tensors at once; at dense settings (small stride) that is many GiB for the
    # whole time series. Process the frame batch in chunks sized to fit — the
    # per-patch estimate is independent across frames, so this is exact.
    per_frame = lh * lw * p * p * 40  # ~5 working tensors (real+complex) per patch elt
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device)
        budget = int(0.4 * free_b)
    else:
        budget = 2 * 1024**3
    chunk = max(1, min(b, budget // max(1, per_frame)))

    out = torch.empty(b, h, w, device=device, dtype=dtype)
    kmax = max(1, min(p // 2 - 1, int(p / (2.0 * max_shift))))
    ks = torch.arange(1, kmax + 1, device=device, dtype=dtype)

    def _patches(x: torch.Tensor, n: int) -> torch.Tensor:
        cols = F.unfold(x, kernel_size=p, stride=stride)  # (n, p*p, L)
        return cols.transpose(1, 2).reshape(n * lh * lw, p, p)

    for b0 in range(0, b, chunk):
        b1 = min(b0 + chunk, b)
        cb = b1 - b0
        fpad = F.pad(fixed[b0:b1, None], (pad, pad, pad, pad), mode="reflect")
        mpad = F.pad(moving[b0:b1, None], (pad, pad, pad, pad), mode="reflect")
        fp, mp = _patches(fpad, cb), _patches(mpad, cb)
        if pe_is_u:  # PE along W (dim=2) -> move it to dim=1 for the FFT
            fp, mp = fp.transpose(1, 2), mp.transpose(1, 2)
        fp = fp - fp.mean(dim=(1, 2), keepdim=True)
        mp = mp - mp.mean(dim=(1, 2), keepdim=True)

        cross = (torch.fft.fft(fp, dim=1) * torch.fft.fft(mp, dim=1).conj()).sum(dim=2)  # (N, ppe)
        del fp, mp
        ang = torch.angle(cross[:, 1 : kmax + 1])
        wts = cross[:, 1 : kmax + 1].abs()
        slope = (wts * ks * ang).sum(dim=1) / (wts * ks * ks).sum(dim=1).clamp_min(eps)
        # slope*ppe/2π is already the pull displacement (−Δ for moving=fixed(x+Δ)).
        shift = (slope * ppe / (2.0 * math.pi)).clamp(-max_shift, max_shift).reshape(cb, lh, lw)
        out[b0:b1] = F.interpolate(
            shift[:, None], size=(h, w), mode="bilinear", align_corners=True
        )[:, 0]
        del cross, ang, wts, slope, shift
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
        mw = _warp2d(moving, u, v, warp_interp)
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
    """Map a PE-direction token (AP/PA/LR/RL/IS/SI or x/y/z) to a NIfTI axis."""
    key = pe_dir.strip()
    if key not in PE_DIR_TO_AXIS:
        raise ValueError(
            f"Unknown -pe_dir '{pe_dir}'. Use axis letters x/y/z or AP/PA/LR/RL/IS/SI."
        )
    return PE_DIR_TO_AXIS[key]


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


def _build_soft_mask(
    data: np.ndarray,
    slice_axis: int,
    a0: int,
    a1: int,
    dilate: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Feathered brain mask in canonical ``(nSlice, H, W)`` layout, values in [0, 1].

    A 3dAutomask on the time-mean, dilated by ``dilate`` voxels (safety margin so
    real brain-edge motion survives) and Gaussian-feathered by ``sigma`` voxels so
    the flow decays smoothly to zero instead of at a hard edge. Multiplying the
    flow by this kills the wild displacements optical flow invents in the pure-noise
    air outside the head, without clipping a hard boundary through the brain.
    """
    from .mask import automask

    ref = torch.from_numpy(np.ascontiguousarray(data.mean(axis=3)))  # (nx, ny, nz)
    vol_zyx = ref.permute(2, 1, 0).contiguous().to(device).float()  # automask wants (nz,ny,nx)
    m = automask(vol_zyx, dilate_extra=dilate, device=device).float()
    m = _gaussian_blur3d(m, sigma)
    mask_nifti = m.permute(2, 1, 0)  # back to (nx, ny, nz)
    # Reorder to the canonical spatial layout used for the flow (slice, H=a0, W=a1).
    return mask_nifti.permute(slice_axis, a0, a1).contiguous().cpu()


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
) -> torch.Tensor:
    """Warp ``moving_raw`` by the estimated displacement (the warp we save).

    Single-PE keeps only the PE-axis component; ``dual`` (two PE axes) uses both.
    """
    if dual:
        return _warp2d(moving_raw, u, v, warp_interp)
    uc = u if pe_flow_is_u else torch.zeros_like(u)
    vc = torch.zeros_like(v) if pe_flow_is_u else v
    return _warp2d(moving_raw, uc, vc, warp_interp)


def _estimate_static(
    vol,
    ref_mode,
    pe_flow_is_u,
    smooth_sigma,
    flow_fn,
    device,
    dual=False,
    warp_interp="bilinear",
    ref_override=None,
    first_n=None,
    diag=None,
):
    """Fixed reference: batch the flow over ALL frames at once, looping slices.

    ``ref_override`` (``(nS, H, W)``) bypasses ``ref_mode`` — used by the outer
    reference-refinement loop, which re-registers against the corrected-data mean.
    ``first_n`` windows the reference aggregate to the early frames.

    ``diag`` (a dict) opts into the xcorr searchlight diagnostics: it is filled with
    ``conf`` ``(nt, nS, H, W)`` and, when ``diag["curve_frame"]`` is set, ``curve``
    ``(nd, nS, H, W)`` for that frame — both in canonical layout, empty for flow/phase.
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
        fixed = _blur2d(fixed_raw, smooth_sigma) if smooth_sigma > 0 else fixed_raw
        moving = _blur2d(moving_raw, smooth_sigma) if smooth_sigma > 0 else moving_raw
        conf_acc: list[torch.Tensor] | None = [] if diag is not None else None
        curve_acc: list[torch.Tensor] | None = [] if curve_frame is not None else None
        u, v = flow_fn(fixed, moving, conf_out=conf_acc, curve_out=curve_acc)
        corrected[:, s] = _correct_pe(moving_raw, u, v, pe_flow_is_u, dual, warp_interp).cpu()
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
        diag["conf"] = conf_all if (conf_all is not None and float(conf_all.abs().sum()) > 0) else None
        diag["curve"] = curve_all
    return u_all, v_all, corrected


def _estimate_cumulative(
    vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_fn, device, dual=False, warp_interp="bilinear"
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
        fixed = _blur2d(ref, smooth_sigma) if smooth_sigma > 0 else ref
        moving = _blur2d(moving_raw, smooth_sigma) if smooth_sigma > 0 else moving_raw
        u, v = flow_fn(fixed, moving)
        corr = _correct_pe(moving_raw, u, v, pe_flow_is_u, dual, warp_interp)
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
    refine_rounds: int = 0,
    converge: float = 0.0,
    converge_rel: float = 0.0,
    first_n: int | None = None,
    jacobian: bool = False,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    is_3dacq: bool = False,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    save_corr_curve: int | None = None,
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
    ``warp_interp`` (``bilinear`` | ``bicubic``) is the resampler for the estimation
    iterations and the correction — ``bicubic`` removes the bilinear damping bias so
    iterations converge to the true shift. ``refine_rounds`` re-registers against the
    corrected-data mean that many extra times, converging the reference template out
    of its bias. ``jacobian`` scales the corrected series by the PE Jacobian so signal
    is conserved (stretched regions dim, compressed regions brighten).

    ``automask`` (off here; on by the CLI) soft-gates the flow by a feathered brain
    mask — a dilated 3dAutomask of the time-mean, Gaussian-blurred by
    ``automask_sigma`` voxels — so optical flow's wild guesses in the pure-noise air
    outside the head fade to zero instead of showing up as huge displacements. The
    corrected series is rebuilt from the masked flow (so it stays consistent).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _validate_estimation_inputs(
        data.shape, pe_axis, slice_axis, is_3dacq=is_3dacq, max_shift=max_shift,
        trial_step=trial_step, window_sigma=window_sigma, verbose=verbose,
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
            noshift_margin=noshift_margin,
            reg_sigma=reg_sigma,
            peak_mode=peak_mode,
            save_corr_curve=save_corr_curve,
            device=device,
            verbose=verbose,
        )

    orig_shape = (data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    in_plane = sorted(a for a in (0, 1, 2) if a != slice_axis)
    a0, a1 = in_plane  # a0 -> H (rows, v), a1 -> W (cols, u)
    pe_flow_is_u = pe_axis == a1

    perm = [3, slice_axis, a0, a1]
    vol = torch.from_numpy(np.ascontiguousarray(data)).permute(perm).contiguous()
    nt, ns = vol.shape[0], vol.shape[1]
    if verbose:
        print(f"🌀 locomoco {_geometry_report(orig_shape, pe_axis, slice_axis, is_3dacq=False, dual=dual)}")

    flow_fn = _build_flow_fn(
        backend,
        pe_flow_is_u=pe_flow_is_u,
        dual=dual,
        pe_only=pe_only,
        n_levels=n_levels,
        n_iters=n_iters,
        window_sigma=window_sigma,
        warp_interp=warp_interp,
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

    progressive = ref_mode in ("first_mean", "first_median")
    if progressive:
        u_all, v_all, corrected = _estimate_cumulative(
            vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_fn, device, dual, warp_interp
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
            first_n=first_n,
            diag=diag,
        )
    # Outer reference-refinement (shared engine): rebuild the reference from the
    # corrected series and re-register, converging the template out of its bias. The
    # aggregate honours -ref and -first_n; the step is the in-brain rms of the stacked
    # (u, v) change.
    if refine_rounds > 0:
        brain = _brain_mask_from(corrected.abs().mean(dim=0))  # (nS, H, W)

        def _est_2d(new_ref):
            u, v, corr = _estimate_static(
                vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_fn, device, dual,
                warp_interp, ref_override=new_ref, diag=diag,
            )
            return torch.stack([u, v]), corr

        def _brain_rms_2d(delta):
            d = delta[:, :, brain]  # (2, nt, N_brain)
            return float(d.pow(2).mean().sqrt()) if d.numel() else 0.0

        stacked, corrected = _refine_loop(
            _est_2d,
            torch.stack([u_all, v_all]),
            corrected,
            reduce_ref=lambda c: _refine_reduce(c, ref_mode, 0, first_n),
            brain_rms=_brain_rms_2d,
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            max_shift=max_shift,
            verbose=verbose,
        )
        u_all, v_all = stacked[0], stacked[1]

    soft = None
    if automask:
        soft = _build_soft_mask(data, slice_axis, a0, a1, automask_dilate, automask_sigma, device)
        u_all = u_all * soft[None]
        v_all = v_all * soft[None]
        # Rebuild the corrected series from the masked flow: outside the head the
        # displacement is now ~0, so the resample is identity and the noise passes
        # through untouched instead of getting yanked by a phantom warp.
        for s in range(ns):
            moving_raw = vol[:, s].to(device).float()
            corrected[:, s] = _correct_pe(
                moving_raw,
                u_all[:, s].to(device),
                v_all[:, s].to(device),
                pe_flow_is_u,
                dual,
                warp_interp,
            ).cpu()

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
        axes = f"axes {a0},{a1}" if dual else f"axis {pe_axis}"
        print(
            f"🌀 locomoco: {nt} frames × {ns} slices, PE {axes}, ref={ref_mode} "
            f"({'2-D dual-PE' if dual else '1-D PE' if pe_only else '2-D'} flow); "
            f"|disp| median {med:.3f} vox ({region}), max {float(ap.max()):.3f} vox"
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
    fdf = fd.reshape(T, -1).double()  # (T, N)
    af = anchor.reshape(T, -1).double()
    wa, wd = float(w_anchor), float(w_diff)

    # Constant tridiagonal: diag b_k = wa + wd*(#adjacent steps); off-diag = -wd.
    b = torch.full((T,), wa + 2.0 * wd, dtype=torch.float64, device=fd.device)
    b[0] = wa + wd
    b[-1] = wa + wd
    # RHS: wa*anchor_k + wd*(fd_k − fd_{k+1}); boundaries drop the missing step term.
    rhs = wa * af
    rhs[:-1] = rhs[:-1] - wd * fdf[1:]  # − wd * fd_{k+1}
    rhs[1:] = rhs[1:] + wd * fdf[1:]  # + wd * fd_k

    # Thomas sweep (sub/super diagonals are the constant −wd).
    cp = torch.zeros(T, dtype=torch.float64, device=fd.device)
    dp = torch.zeros_like(rhs)
    cp[0] = -wd / b[0]
    dp[0] = rhs[0] / b[0]
    for k in range(1, T):
        m = b[k] - (-wd) * cp[k - 1]
        cp[k] = -wd / m
        dp[k] = (rhs[k] - (-wd) * dp[k - 1]) / m
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
    fuse: str = "auto",
    fuse_thresh: float = 0.05,
    fuse_weight: float = 1.0,
    first_n: int | None = None,
    is_3dacq: bool = False,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
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
        raw_data.shape, pe_axis, slice_axis, is_3dacq=is_3dacq, max_shift=max_shift,
        trial_step=trial_step, window_sigma=window_sigma, verbose=verbose,
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
            moco_data, slice_axis, a0, a1, automask_dilate, automask_sigma, device
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


def _blur3d_b(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3-D Gaussian blur of a batch of volumes ``(B, X, Y, Z)``."""
    if sigma <= 0:
        return vol
    k = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
    r = (k.numel() - 1) // 2
    x = vol.unsqueeze(1)
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1))
    x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1))
    x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1))
    return x.squeeze(1)


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


def _shift3d_axis(vol: torch.Tensor, shift, axis: int, mode: str = "bilinear") -> torch.Tensor:
    """Sample ``(B,X,Y,Z)`` at ``coord + shift`` along ``axis`` ONLY (PE pull warp).

    ``shift`` is a scalar or a ``(B,X,Y,Z)`` field. grid_sample's 3-D mode is trilinear
    (``bilinear``) or nearest — no bicubic in 3-D — so a bicubic request degrades here.

    Scalar fast path: a uniform shift along one axis is *exactly* 1-D linear interpolation
    there (trilinear reduces to it — the other two axes sit on integer grid), so we skip
    building the full 3-D coordinate grid and the 3-D sampler. The xcorr searchlight applies
    ``nd`` scalar shifts per frame, so this is the hot per-offset op. Border-clamp + linear
    weight reproduce ``grid_sample(padding_mode="border", align_corners=True)`` bit-for-bit.
    """
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
) -> torch.Tensor:
    """1-DOF (PE-axis) pyramidal Lucas-Kanade on 3-D volumes ``(B, X, Y, Z)``.

    The 3-D analogue of :func:`optical_flow_lk_2d` with ``pe_only``: only the PE-axis
    gradient enters, the pooling window is a 3-D Gaussian, and warping/pyramids are
    volumetric. Returns the PE-axis pull displacement ``(B, X, Y, Z)``.
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
            mw = _shift3d_axis(mv, disp, pe_axis)
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
            _ncc, r, trial_step, noshift_margin, reg_sigma, ambiguity_frac,
            lambda x: _blur3d_b(x, reg_sigma), curve_out,
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
        best_i, best_val, zero_val, ym2, ym1, y0, yp1, yp2,
        nd=nd, r=r, trial_step=trial_step, noshift_margin=noshift_margin,
        reg_sigma=reg_sigma, blur=lambda x: _blur3d_b(x, reg_sigma),
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
    - ``brain_rms(Δdisp) -> float`` is the in-brain rms of a displacement change — the
      step size, layout-specific so each caller supplies its own (its ``disp`` may be a
      stacked ``(2,…)`` u/v or a plain ``(…,T)`` field).

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
        step = brain_rms(disp - prev)
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
    device: torch.device,
    verbose: bool,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    save_corr_curve: int | None = None,
) -> LocomocoResult:
    """Plain (moco-frame) residual PE motion for 3-D-acquired EPI: a single 3-D solve."""
    nx, ny, nz, nt = data.shape
    disp_slice = (
        display_slice if display_slice != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
    )
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    if verbose:
        print(f"🌀 locomoco {_geometry_report(data.shape, pe_axis, disp_slice, is_3dacq=True)}")

    vol4d = torch.from_numpy(np.ascontiguousarray(data)).float()
    flow3d = _build_flow3d_fn(
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
    )

    # xcorr searchlight diagnostics, filled by the LAST estimate pass (see _refine_loop):
    # per-voxel confidence over the whole series, and the correlation landscape of one frame.
    xc = backend == "xcorr"
    conf_field = torch.zeros(nx, ny, nz, nt) if xc else None
    curve_frame = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
    curve_box: dict[str, torch.Tensor | None] = {"curve": None, "offsets": None}

    def _estimate(ref_vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fxb = _blur3d_b(ref_vol[None], smooth_sigma)[0] if smooth_sigma > 0 else ref_vol
        disp = torch.zeros(nx, ny, nz, nt)
        corrected = torch.zeros(nx, ny, nz, nt)
        for t in tqdm(range(nt), desc="locomoco 3D", unit="frame", leave=True, disable=nt < 3):
            mv = vol4d[..., t].to(device)
            mvb = _blur3d_b(mv[None], smooth_sigma)[0] if smooth_sigma > 0 else mv
            conf_acc: list[torch.Tensor] | None = [] if xc else None
            curve_acc: list[torch.Tensor] | None = [] if (xc and t == curve_frame) else None
            d = flow3d(fxb[None], mvb[None], conf_out=conf_acc, curve_out=curve_acc)[0]
            corrected[..., t] = _shift3d_axis(mv[None], d[None], pe_axis)[0].cpu()
            disp[..., t] = d.cpu()
            if conf_field is not None and conf_acc:
                conf_field[..., t] = conf_acc[0][0].cpu()
            if curve_acc is not None:
                curve_box["curve"] = torch.stack(curve_acc, 0)[:, 0].permute(1, 2, 3, 0).contiguous()
                rr = float(max(1, int(np.ceil(max_shift))))
                curve_box["offsets"] = torch.arange(-rr, rr + 1e-6, trial_step)
        return disp, corrected

    disp, corrected = _estimate(_select_ref_vol(vol4d, ref_mode, first_n).to(device))
    # Reference-refinement (the -refine / -workhard / -superhard knob, in 3-D) via the
    # shared engine: rebuild the reference from the corrected series and re-register.
    # The aggregate honours -ref and -first_n; the step is the in-brain rms of the field.
    if refine_rounds > 0:
        brain = _brain_mask_from(corrected.abs().mean(dim=3))  # (nx, ny, nz)
        disp, corrected = _refine_loop(
            _estimate,
            disp,
            corrected,
            reduce_ref=lambda c: _refine_reduce(c, ref_mode, 3, first_n).to(device),
            brain_rms=lambda delta: float(delta[brain].pow(2).mean().sqrt()),
            refine_rounds=refine_rounds,
            converge=converge,
            converge_rel=converge_rel,
            max_shift=max_shift,
            verbose=verbose,
        )

    if automask:
        soft = _build_soft_mask(data, disp_slice, a0, a1, automask_dilate, automask_sigma, device)
        soft_xyz = soft.permute(_inv_perm([disp_slice, a0, a1])).contiguous()
        disp = disp * soft_xyz[..., None]
        for t in range(nt):
            corrected[..., t] = _shift3d_axis(
                vol4d[..., t].to(device)[None], disp[..., t].to(device)[None], pe_axis
            )[0].cpu()

    p_canon = disp.permute(perm4).contiguous()
    u_canon = p_canon if pe_flow_is_u else torch.zeros_like(p_canon)
    v_canon = torch.zeros_like(p_canon) if pe_flow_is_u else p_canon
    if verbose:
        ap = disp.abs()
        sel = ap[ap > 0]
        med = float(sel.median()) if sel.numel() else 0.0
        print(
            f"🌀 locomoco 3D-acq: {nt} frames, PE axis {pe_axis}, backend={backend}, "
            f"ref={ref_mode} (single 3-D solve, no slicing); |disp| median {med:.3f} vox, "
            f"max {float(ap.max()):.3f} vox"
        )
    return LocomocoResult(
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
) -> torch.Tensor:
    """Shared-field 1-DOF (PE-axis) pyramidal LK across echoes ``(B,X,Y,Z)``.

    ``fixed_list`` / ``moving_list`` are the E per-echo reference / moving volumes;
    ``alpha`` is the ``(E,)`` per-echo scaling. Solves for ONE displacement ``w`` such
    that echo ``e`` is warped by ``alpha_e·w``, pooling the Gauss-Newton normal
    equations over echoes: ``step = -Σ_e α_e·⟨∇·r⟩ / (Σ_e α_e²·⟨∇²⟩ + reg)`` under a
    Gaussian window. Reduces to :func:`optical_flow_lk_3d` for a single echo with
    ``alpha=[1]``. Returns the shared PE-axis pull displacement ``w`` ``(B,X,Y,Z)``.
    """
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
                mw = _shift3d_axis(mpyr[j][lvl], a * disp, pe_axis)
                it = mw - fpyr[j][lvl]
                ip = _grad_axis_3d(mw, pe_axis)
                num = num + a * _blur3d_b(ip * it, window_sigma)
                den = den + (a * a) * _blur3d_b(ip * ip, window_sigma)
            disp = disp - num / (den + reg)
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
    ambiguity_frac: float = 0.5,
    curve_out: list[torch.Tensor] | None = None,
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
        return _blur3d_b(x, window_sigma)

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
            _pooled_ncc, r, trial_step, noshift_margin, reg_sigma, ambiguity_frac,
            lambda x: _blur3d_b(x, reg_sigma), curve_out,
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
        best_i, best_val, zero_val, ym2, ym1, y0, yp1, yp2,
        nd=nd, r=r, trial_step=trial_step, noshift_margin=noshift_margin,
        reg_sigma=reg_sigma, blur=lambda x: _blur3d_b(x, reg_sigma),
    )
    return s_field / float(alpha[m]), conf  # echo-1 scale (alpha_1 = 1)


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
    learn_scaling: bool = True,
    flat_scaling: bool = False,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
    save_corr_curve: int | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MultiEchoLocomocoResult:
    """Joint residual partition-direction motion for multi-echo 3-D EPI.

    ``datas`` is the list of E moco'd 4-D series (one per echo, identical grid + T);
    ``echo_times`` the matching TEs in ms. ``pe_axis`` is the corrected direction (the
    slice/partition axis for 3-D EPI, so PE==slice is fine). All echoes are treated as
    one 3-D-acquired series (no per-slice fields).

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
    _validate_estimation_inputs(
        shp, pe_axis, slice_axis, is_3dacq=True, max_shift=max_shift,
        trial_step=trial_step, window_sigma=window_sigma, verbose=verbose,
    )

    disp_slice = slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    pe_flow_is_u = pe_axis == a1
    perm4 = [3, disp_slice, a0, a1]
    if verbose:
        print(f"🌀 locomoco {_geometry_report(shp, pe_axis, disp_slice, is_3dacq=True)}  × {e} echoes")

    te = torch.tensor([float(x) for x in echo_times], dtype=torch.float32)
    vols = [torch.from_numpy(np.ascontiguousarray(d)).float() for d in datas]
    refs = [_select_ref_vol(v, ref_mode, first_n).to(device) for v in vols]
    if smooth_sigma > 0:
        refs = [_blur3d_b(r[None], smooth_sigma)[0] for r in refs]

    # A fixed scaling (TE ratio or flat) skips learning; both still pool all echoes.
    learn = learn_scaling and not flat_scaling

    # Single-echo backend estimator — used to LEARN alpha (rank-1 factor of one per-echo
    # pass) and, for xcorr, to solve the shared field itself. Persistent bars (leave=True)
    # so a long run stays visible.
    flow3d = _build_flow3d_fn(
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
    )

    # Confidence map from the pooled searchlight (fixed/flat-scaling xcorr path only — the
    # LK and learn-alpha paths have no single per-voxel searchlight quality). Filled by the
    # final _pooled_xcorr call and surfaced on the result for a `-save_confidence` diag.
    pooled_conf = torch.zeros(nx, ny, nz, nt)
    have_pooled_conf = False
    curve_frame = None if save_corr_curve is None else max(0, min(int(save_corr_curve), nt - 1))
    corr_curve: torch.Tensor | None = None
    corr_offsets: torch.Tensor | None = None

    def _per_echo_estimate(cur_refs: list[torch.Tensor], tag: str) -> torch.Tensor:
        out = torch.zeros(e, nx, ny, nz, nt)
        for j in range(e):
            for t in tqdm(
                range(nt), desc=f"me {tag} e{j + 1}/{e}", unit="frame", leave=True, disable=nt < 3
            ):
                mv = vols[j][..., t].to(device)
                mvb = _blur3d_b(mv[None], smooth_sigma)[0] if smooth_sigma > 0 else mv
                out[j, ..., t] = flow3d(cur_refs[j][None], mvb[None])[0].cpu()
        return out

    def _project(flat: torch.Tensor) -> torch.Tensor:
        """Least-squares project the per-echo estimates (e, R) onto the current alpha."""
        return (alpha[:, None] * flat).sum(0) / max(float((alpha * alpha).sum()), 1e-12)

    def _joint_lk(cur_refs: list[torch.Tensor], tag: str) -> torch.Tensor:
        out = torch.zeros(nx, ny, nz, nt)
        for t in tqdm(range(nt), desc=f"me {tag}", unit="frame", leave=True, disable=nt < 3):
            movs = [vols[j][..., t].to(device) for j in range(e)]
            if smooth_sigma > 0:
                movs = [_blur3d_b(m[None], smooth_sigma)[0] for m in movs]
            out[..., t] = optical_flow_lk_3d_multiecho(
                [r[None] for r in cur_refs],
                [m[None] for m in movs],
                alpha,
                pe_axis,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
            )[0].cpu()
        return out

    def _pooled_xcorr(cur_refs: list[torch.Tensor], tag: str) -> torch.Tensor:
        nonlocal have_pooled_conf, corr_curve, corr_offsets
        out = torch.zeros(nx, ny, nz, nt)
        for t in tqdm(range(nt), desc=f"me {tag}", unit="frame", leave=True, disable=nt < 3):
            movs = [vols[j][..., t].to(device) for j in range(e)]
            if smooth_sigma > 0:
                movs = [_blur3d_b(m[None], smooth_sigma)[0] for m in movs]
            curve_acc: list[torch.Tensor] | None = [] if t == curve_frame else None
            w_be, c_be = xcorr_search_flow_3d_multiecho(
                [r[None] for r in cur_refs],
                [m[None] for m in movs],
                alpha,
                pe_axis,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                noshift_margin=noshift_margin,
                reg_sigma=reg_sigma,
                peak_mode=peak_mode,
                curve_out=curve_acc,
            )
            out[..., t] = w_be[0].cpu()
            pooled_conf[..., t] = c_be[0].cpu()
            if curve_acc is not None:
                corr_curve = torch.stack(curve_acc, 0)[:, 0].permute(1, 2, 3, 0).contiguous()
                rr = float(max(1, int(np.ceil(max_shift))))
                corr_offsets = torch.arange(-rr, rr + 1e-6, trial_step)
        have_pooled_conf = True
        return out

    def _solve_w(cur_refs: list[torch.Tensor], tag: str) -> torch.Tensor:
        # flow: image-space shared-field pooled LK. xcorr with FIXED alpha: shared-parameter
        # searchlight pooling every echo into one informed search (the "best of both"). xcorr
        # while LEARNING alpha: per-echo estimate + project (the per-echo fields are needed to
        # learn the ratios anyway), warp-space pooling.
        if backend == "flow":
            return _joint_lk(cur_refs, tag)
        if not learn:
            return _pooled_xcorr(cur_refs, tag)
        return _project(_per_echo_estimate(cur_refs, tag).reshape(e, -1)).reshape(nx, ny, nz, nt)

    def _corrected_echo(cur_w: torch.Tensor, j: int) -> torch.Tensor:
        disp_e = alpha[j] * cur_w
        out = torch.zeros(nx, ny, nz, nt)
        for t in range(nt):
            out[..., t] = _shift3d_axis(
                vols[j][..., t].to(device)[None], disp_e[..., t].to(device)[None], pe_axis
            )[0].cpu()
        return out

    # Phase 1: learn alpha, then the initial shared field. When learning, factor ONE
    # per-echo pass; for xcorr reuse that very pass as the initial w (no second sweep).
    # flat_scaling pins alpha to 1 (TE-independent shift); else the TE ratio.
    alpha = torch.ones(e) if flat_scaling else te / float(te[0])
    if learn:
        disp0 = _per_echo_estimate(refs, "init").reshape(e, -1)
        alpha, _ = _rank1_factor_echoes(disp0, alpha)
        w = (
            _joint_lk(refs, "joint")
            if backend == "flow"
            else _project(disp0).reshape(nx, ny, nz, nt)
        )
    else:
        w = _solve_w(refs, "joint" if backend == "flow" else "solve")

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
        brain = _brain_mask_from(w.abs().mean(dim=3)) if float(w.abs().max()) > 0 else None
        _pass = {"n": 0}

        def _reduce_me(cur_w):
            refs = [
                _refine_reduce(_corrected_echo(cur_w, j), ref_mode, 3, first_n).to(device)
                for j in range(e)
            ]
            if smooth_sigma > 0:
                refs = [_blur3d_b(r[None], smooth_sigma)[0] for r in refs]
            return refs

        def _est_me(new_refs):
            _pass["n"] += 1
            w_new = _solve_w(new_refs, f"refine {_pass['n']}/{refine_rounds}")
            return w_new, w_new

        def _brain_rms_me(delta):
            d = delta[brain] if brain is not None else delta
            return float(d.pow(2).mean().sqrt()) if d.numel() else 0.0

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

    if automask:
        # One shared mask from the across-echo mean (all echoes share geometry).
        mean_series = np.mean(np.stack([np.asarray(d) for d in datas], 0), 0)
        soft = _build_soft_mask(
            mean_series, disp_slice, a0, a1, automask_dilate, automask_sigma, device
        )
        soft_xyz = soft.permute(_inv_perm([disp_slice, a0, a1])).contiguous()
        w = w * soft_xyz[..., None]

    # Build per-echo results: echo e warped by alpha_e · w.
    per_echo: list[LocomocoResult] = []
    for j in range(e):
        disp_e = (alpha[j] * w).contiguous()
        corrected = _corrected_echo(w, j)
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
        ap = w.abs()
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
        print(
            f"🌀 locomoco ME-3D: {e} echoes (TE {te_str} ms), {nt} frames, PE axis {pe_axis}, "
            f"backend={backend}; shared |w| median {med:.3f} vox, max {float(ap.max()):.3f} vox"
        )
        print(f"   alpha (÷echo1) = [{a_str}]  {tag}")

    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=alpha,
        echo_times=te,
        w_field=w,
        pe_axis=pe_axis,
        linearity_r2=r2,
        confidence=pooled_conf if have_pooled_conf else None,
        corr_curve=corr_curve,
        corr_offsets=corr_offsets,
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
    flat_scaling: bool = False,
    noshift_margin: float = 0.0,
    reg_sigma: float = 1.5,
    peak_mode: str = "first_peak",
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
        is_3dacq=True,
        noshift_margin=noshift_margin,
        reg_sigma=reg_sigma,
        peak_mode=peak_mode,
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
                vol_j[..., t].to(device)[None], disp_e[..., t].to(device)[None], pe_axis
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
    save_corr_curve: int | None = None,
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
        shp, pe_axis, slice_axis, is_3dacq=True, max_shift=max_shift,
        trial_step=trial_step, window_sigma=window_sigma, verbose=verbose,
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
    for t in tqdm(range(nt), desc="me interecho", unit="frame", leave=True, disable=nt < 3):
        ims = [vols[j][..., t].to(device) for j in range(e)]
        if smooth_sigma > 0:
            ims = [_blur3d_b(im[None], smooth_sigma)[0] for im in ims]
        fixed_list = [ims[j][None] for j in range(e - 1)]  # echo n-1 (lower TE)
        moving_list = [ims[j + 1][None] for j in range(e - 1)]  # echo n
        if backend == "flow":
            # LK has no per-pair weighting yet; dropout is handled by gating the output.
            g = optical_flow_lk_3d_multiecho(
                fixed_list,
                moving_list,
                dte,
                pe_axis,
                n_levels=n_levels,
                n_iters=n_iters,
                window_sigma=window_sigma,
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
                vols[j][..., t].to(device)[None], disp_e[..., t].to(device)[None], pe_axis
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
