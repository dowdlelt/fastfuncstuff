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
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
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


def _xcorr_shift_1d(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    pe_is_u: bool,
    max_shift: float,
    window_sigma: float,
    trial_step: float,
    interp: str,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Sub-voxel pull shift along ONE in-plane axis by local correlation search.

    Slide ``moving`` along the axis (``pe_is_u`` → W/columns, else H/rows) over trial
    offsets spanning ``±max_shift`` in ``trial_step`` steps; at each, measure local
    normalized correlation with ``fixed`` under a ``window_sigma`` Gaussian searchlight.
    Per voxel the peak offset, refined by a 5-point least-squares parabola (peak ±2,
    robust to a noisy correlation curve), is the pull displacement. Trials are swept
    with a streaming running-peak, so memory is O(frame×slice), not O(trials×…).
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
    ym2, ym1, y0, yp1, yp2 = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    need = torch.zeros((b, h, w), device=device, dtype=torch.long)
    prev1: torch.Tensor | None = None
    prev2: torch.Tensor | None = None
    for i in range(nd):
        mw = _shift(moving, float(offsets[i]))
        mean_m = _win(mw)
        var_m = (_win(mw * mw) - mean_m * mean_m).clamp_min(eps)
        corr = (_win(fixed * mw) - mean_f * mean_m) / torch.sqrt(var_f * var_m)
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

    # 5-point LS quadratic (x=-2..2) vertex; fall back to 3-point near an edge.
    b5 = (-2.0 * ym2 - ym1 + yp1 + 2.0 * yp2) / 10.0
    a5 = (5.0 * (4.0 * ym2 + ym1 + yp1 + 4.0 * yp2) - 10.0 * (ym2 + ym1 + y0 + yp1 + yp2)) / 70.0
    vtx5 = torch.where(a5.abs() > 1e-9, -b5 / (2.0 * a5), torch.zeros_like(a5)).clamp(-1.0, 1.0)
    den3 = ym1 - 2.0 * y0 + yp1
    vtx3 = torch.where(den3.abs() > 1e-6, 0.5 * (ym1 - yp1) / den3, torch.zeros_like(den3))
    vtx3 = vtx3.clamp(-1.0, 1.0)
    can5 = (best_i >= 2) & (best_i <= nd - 3)
    can3 = (best_i >= 1) & (best_i <= nd - 2)
    sub = torch.where(can5, vtx5, torch.where(can3, vtx3, torch.zeros_like(vtx5)))
    return (-r + best_i.to(dtype) * trial_step) + sub * trial_step


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local cross-correlation searchlight backend, same ``(fixed, moving) -> (u, v)``.

    Residual EPI motion is a small local translation; we do the literal thing —
    slide ``moving`` and take the per-voxel offset of peak local correlation (a true
    alignment search, so accurate single-shot). See :func:`_xcorr_shift_1d`.

    Single-PE searches one axis (``pe_is_u`` selects it). ``dual`` (two PE axes)
    searches BOTH, but *separably* — a 1-D search along each axis, alternating and
    warping by the running estimate for ``n_passes`` (converges in ~3) — to avoid the
    O(trials²) blow-up of a joint 2-D offset grid.
    """
    if not dual:
        s = _xcorr_shift_1d(
            fixed, moving, pe_is_u, max_shift, window_sigma, trial_step, interp, eps
        )
        z = torch.zeros_like(s)
        return (s, z) if pe_is_u else (z, s)

    u = torch.zeros_like(fixed)
    v = torch.zeros_like(fixed)
    for _ in range(n_passes):
        mw = _warp2d(moving, u, v, warp_interp)
        u = u + _xcorr_shift_1d(fixed, mw, True, max_shift, window_sigma, trial_step, interp, eps)
        mw = _warp2d(moving, u, v, warp_interp)
        v = v + _xcorr_shift_1d(fixed, mw, False, max_shift, window_sigma, trial_step, interp, eps)
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
    fpad = F.pad(fixed[:, None], (pad, pad, pad, pad), mode="reflect")
    mpad = F.pad(moving[:, None], (pad, pad, pad, pad), mode="reflect")
    lh = (fpad.shape[2] - p) // stride + 1
    lw = (fpad.shape[3] - p) // stride + 1

    def _patches(x: torch.Tensor) -> torch.Tensor:
        cols = F.unfold(x, kernel_size=p, stride=stride)  # (B, p*p, L)
        return cols.transpose(1, 2).reshape(b * lh * lw, p, p)

    fp, mp = _patches(fpad), _patches(mpad)
    if pe_is_u:  # PE along W (dim=2) -> move it to dim=1 for the FFT
        fp, mp = fp.transpose(1, 2), mp.transpose(1, 2)
    fp = fp - fp.mean(dim=(1, 2), keepdim=True)
    mp = mp - mp.mean(dim=(1, 2), keepdim=True)
    ppe = fp.shape[1]

    cross = (torch.fft.fft(fp, dim=1) * torch.fft.fft(mp, dim=1).conj()).sum(dim=2)  # (N, ppe)
    kmax = max(1, min(ppe // 2 - 1, int(ppe / (2.0 * max_shift))))
    ks = torch.arange(1, kmax + 1, device=device, dtype=dtype)
    ang = torch.angle(cross[:, 1 : kmax + 1])
    wts = cross[:, 1 : kmax + 1].abs()
    slope = (wts * ks * ang).sum(dim=1) / (wts * ks * ks).sum(dim=1).clamp_min(eps)
    # slope*ppe/2π is already the pull displacement (−Δ for moving=fixed(x+Δ)).
    shift = (slope * ppe / (2.0 * math.pi)).clamp(-max_shift, max_shift).reshape(b, lh, lw)
    return F.interpolate(shift[:, None], size=(h, w), mode="bilinear", align_corners=True)[:, 0]


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
        if want_u:
            u = u + _phase_slope_dense(fixed, mw, True, patch, stride, max_shift)
        if want_v:
            v = v + _phase_slope_dense(fixed, mw, False, patch, stride, max_shift)
    return u, v


def resolve_pe_axis(pe_dir: str) -> int:
    """Map a PE-direction token (AP/PA/LR/RL/IS/SI or x/y/z) to a NIfTI axis."""
    key = pe_dir.strip()
    if key not in PE_DIR_TO_AXIS:
        raise ValueError(
            f"Unknown -pe_dir '{pe_dir}'. Use axis letters x/y/z or AP/PA/LR/RL/IS/SI."
        )
    return PE_DIR_TO_AXIS[key]


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
        """
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
        """Motion-corrected 4-D series ``(nx, ny, nz, T)``."""
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
):
    """Fixed reference: batch the flow over ALL frames at once, looping slices.

    ``ref_override`` (``(nS, H, W)``) bypasses ``ref_mode`` — used by the outer
    reference-refinement loop, which re-registers against the corrected-data mean.
    """
    nt, ns, hh, ww = vol.shape
    if ref_override is not None:
        ref = ref_override
    elif ref_mode == "mean":
        ref = vol.mean(dim=0)
    elif ref_mode == "median":
        ref = vol.median(dim=0).values
    elif ref_mode == "first":
        ref = vol[0]
    else:
        try:
            ref = vol[int(ref_mode)]
        except ValueError as e:
            raise ValueError(
                f"-ref must be mean|median|first|first_mean|first_median|<index>, got '{ref_mode}'"
            ) from e

    u_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    v_all = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    corrected = torch.zeros(nt, ns, hh, ww, dtype=torch.float32)
    for s in tqdm(range(ns), desc="locomoco slices", unit="slice", leave=True, disable=ns < 2):
        moving_raw = vol[:, s].to(device).float()
        fixed_raw = ref[s].to(device).unsqueeze(0).expand(nt, hh, ww).contiguous().float()
        fixed = _blur2d(fixed_raw, smooth_sigma) if smooth_sigma > 0 else fixed_raw
        moving = _blur2d(moving_raw, smooth_sigma) if smooth_sigma > 0 else moving_raw
        u, v = flow_fn(fixed, moving)
        corrected[:, s] = _correct_pe(moving_raw, u, v, pe_flow_is_u, dual, warp_interp).cpu()
        u_all[:, s] = u.cpu()
        v_all[:, s] = v.cpu()
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
    jacobian: bool = False,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
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
    if pe_axis == slice_axis:
        raise ValueError(
            f"-pe_axis ({pe_axis}) must differ from -slice_axis ({slice_axis}): PE must "
            "lie inside the slice plane to be visible to 2-D flow."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_shape = (data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    in_plane = sorted(a for a in (0, 1, 2) if a != slice_axis)
    a0, a1 = in_plane  # a0 -> H (rows, v), a1 -> W (cols, u)
    pe_flow_is_u = pe_axis == a1

    perm = [3, slice_axis, a0, a1]
    vol = torch.from_numpy(np.ascontiguousarray(data)).permute(perm).contiguous()
    nt, ns = vol.shape[0], vol.shape[1]

    # Backend: a per-pair (fixed, moving) -> (u, v) estimator plugged into the same
    # slice/frame machinery. Optical flow (default), phase-correlation, or magnitude
    # cross-correlation. Single-PE returns just the PE component; dual returns both.
    if backend == "flow":
        # dual (both axes) = full 2-D flow; single respects pe_only.
        pe_only_axis = None if dual else ((0 if pe_flow_is_u else 1) if pe_only else None)

        def flow_fn(fx: torch.Tensor, mv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        def flow_fn(fx: torch.Tensor, mv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        def flow_fn(fx: torch.Tensor, mv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return xcorr_search_flow_2d(
                fx,
                mv,
                pe_is_u=pe_flow_is_u,
                max_shift=max_shift,
                window_sigma=window_sigma,
                trial_step=trial_step,
                dual=dual,
                warp_interp=warp_interp,
            )
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected flow | phase | xcorr.")

    progressive = ref_mode in ("first_mean", "first_median")
    est = _estimate_cumulative if progressive else _estimate_static
    u_all, v_all, corrected = est(
        vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_fn, device, dual, warp_interp
    )
    # Outer reference-refinement: the reference defines "zero", so a blurred/biased
    # one biases every shift. Rebuild it from the corrected series (motion removed →
    # sharp template) and re-register the ORIGINAL frames against it, converging the
    # template. Progressive refs already self-refine over time, so start them there.
    for _ in range(max(0, refine_rounds)):
        new_ref = corrected.mean(dim=0)  # (nS, H, W) sharpened template
        u_all, v_all, corrected = _estimate_static(
            vol,
            ref_mode,
            pe_flow_is_u,
            smooth_sigma,
            flow_fn,
            device,
            dual,
            warp_interp,
            ref_override=new_ref,
        )

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
