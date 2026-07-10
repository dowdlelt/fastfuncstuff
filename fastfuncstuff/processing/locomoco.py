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


def _warp2d(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample ``img`` at ``(x + u, y + v)`` (u along W, v along H). ``(B, H, W)``."""
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
        img.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True
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
            mv_w = _warp2d(mv_l, u, v)
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

    def _to_nifti(self, canon: torch.Tensor) -> torch.Tensor:
        inv = [0, 0, 0, 0]
        for new_pos, old_axis in enumerate(self.perm):
            inv[old_axis] = new_pos
        return canon.permute(inv).contiguous()

    def pe_displacement(self) -> torch.Tensor:
        """Per-frame SIGNED PE displacement ``(nx, ny, nz, T)`` in voxel units.

        This is the single flow map: one signed scalar per voxel per frame — the
        sign is the direction of residual motion along the PE axis, the magnitude
        is how far. Scrub it like a timeseries (a diverging +/- colormap). Also the
        content of the saved warp (before mm conversion).
        """
        return self._to_nifti(self.u_canon if self.pe_flow_is_u else self.v_canon)

    def corrected_series(self) -> torch.Tensor:
        """PE-corrected 4-D series ``(nx, ny, nz, T)``."""
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


def _correct_pe(
    moving_raw: torch.Tensor, u: torch.Tensor, v: torch.Tensor, pe_flow_is_u: bool
) -> torch.Tensor:
    """Warp ``moving_raw`` along the PE axis only (the single-axis warp we save)."""
    uc = u if pe_flow_is_u else torch.zeros_like(u)
    vc = torch.zeros_like(v) if pe_flow_is_u else v
    return _warp2d(moving_raw, uc, vc)


def _estimate_static(vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_kw, device):
    """Fixed reference: batch the flow over ALL frames at once, looping slices."""
    nt, ns, hh, ww = vol.shape
    if ref_mode == "mean":
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
        u, v = optical_flow_lk_2d(fixed, moving, **flow_kw)
        corrected[:, s] = _correct_pe(moving_raw, u, v, pe_flow_is_u).cpu()
        u_all[:, s] = u.cpu()
        v_all[:, s] = v.cpu()
    return u_all, v_all, corrected


def _estimate_cumulative(vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_kw, device):
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
        u, v = optical_flow_lk_2d(fixed, moving, **flow_kw)
        corr = _correct_pe(moving_raw, u, v, pe_flow_is_u)
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
    smooth_sigma: float = 0.0,
    n_levels: int = 3,
    n_iters: int = 4,
    window_sigma: float = 2.0,
    pe_only: bool = True,
    automask: bool = False,
    automask_dilate: int = 4,
    automask_sigma: float = 3.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> LocomocoResult:
    """Estimate per-frame residual motion of a 4-D series via slicewise optical flow.

    ``data`` is ``(nx, ny, nz, T)`` (already motion-corrected). Slices are taken
    orthogonal to ``slice_axis`` (so PE lies in-plane); each slice's time course is
    a 2-D movie. Optical flow of every frame against a reference gives the in-plane
    displacement; the WARP/correction/flow-map use only the ``pe_axis`` component
    (residual EPI motion is a PE-axis displacement, MEDIC-style). ``pe_only``
    (default) constrains the flow itself to the PE axis (1 DOF, more robust); set
    it False for full 2-D flow (e.g. a richer direction movie).

    ``ref_mode``: static ``mean`` | ``median`` | ``first`` | ``<int>``, or a
    PROGRESSIVE ``first_mean`` / ``first_median`` — frame t registers to the
    running mean/median of the already-corrected frames before it (a bootstrapped
    template; frame 0 is the seed). Progressive modes are sequential and slower;
    ``first_median`` also holds the corrected series on the GPU.

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

    pe_only_axis = (0 if pe_flow_is_u else 1) if pe_only else None
    flow_kw = dict(
        n_levels=n_levels, n_iters=n_iters, window_sigma=window_sigma, pe_only_axis=pe_only_axis
    )
    est = _estimate_cumulative if ref_mode in ("first_mean", "first_median") else _estimate_static
    u_all, v_all, corrected = est(vol, ref_mode, pe_flow_is_u, smooth_sigma, flow_kw, device)

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
                moving_raw, u_all[:, s].to(device), v_all[:, s].to(device), pe_flow_is_u
            ).cpu()

    result = LocomocoResult(
        u_canon=u_all,
        v_canon=v_all,
        corrected_canon=corrected,
        perm=perm,
        pe_flow_is_u=pe_flow_is_u,
        pe_axis=pe_axis,
        slice_axis=slice_axis,
        orig_shape=orig_shape,
    )
    if verbose:
        pe = result.pe_displacement()
        print(
            f"🌀 locomoco: {nt} frames × {ns} slices, PE axis {pe_axis}, ref={ref_mode} "
            f"({'1-D PE' if pe_only else '2-D'} flow); "
            f"|PE disp| median {pe.abs().median():.3f} vox, max {pe.abs().max():.3f} vox"
        )
    return result
