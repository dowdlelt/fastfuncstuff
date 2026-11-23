"""
Batched 3D Gaussian smoothing for [[ffs_perm]]'s ``-vsmooth`` (pseudo-t).

Implements **mask-aware** separable Gaussian smoothing — the standard
``3dBlurInMask``-style correction so out-of-mask zeros don't bleed into
mask voxels.  ``smooth_var_per_perm`` is the public entry point: takes a
``[P, V_in_mask]`` variance matrix, scatters to a 4-D ``[P, X, Y, Z]``
volume, smooths each perm independently with a shared kernel, and
gathers back to ``[P, V_in_mask]``.

Performance: separable 1-D passes along each axis, each a shifted-slab
accumulation (``blur_axis``) that works on any memory layout.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

# FWHM ↔ sigma constant: FWHM = sigma · 2√(2 ln 2)
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def fwhm_mm_to_sigma_vox(
    fwhm_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert isotropic FWHM in mm to per-axis Gaussian σ in voxels."""
    sx = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[0]
    sy = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[1]
    sz = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[2]
    return (sx, sy, sz)


def _gaussian_kernel_1d(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Truncated 1-D Gaussian kernel, normalised to sum to 1.

    Truncation radius = 4σ (matches scipy.ndimage default ``truncate=4.0``).
    """
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return k / k.sum()


_fused_unavailable = False


def _fused_blur(x: torch.Tensor):
    """The one-pass CUDA kernel for ``x``, or None for the shift-add path.

    float32 on CUDA only, and never under autograd: the kernel has no backward,
    and CPU/MPS keep the layout-agnostic path below.  ``FFS_NO_FUSED_BLUR=1``
    forces the shift-add path.
    """
    global _fused_unavailable
    if (
        _fused_unavailable
        or x.device.type != "cuda"
        or x.dtype != torch.float32
        or (x.requires_grad and torch.is_grad_enabled())
        or os.environ.get("FFS_NO_FUSED_BLUR")
    ):
        return None
    try:
        from fastfuncstuff.stats.blur_triton import blur_axis_cuda
    except Exception as exc:  # pragma: no cover - Triton is optional
        _fused_unavailable = True
        print(f"** fused blur kernel unavailable ({exc}); using the shift-add blur")
        return None
    return blur_axis_cuda


def blur_axis(
    x: torch.Tensor, kernel: torch.Tensor, axis: int, *, reflect: bool = False
) -> torch.Tensor:
    """Correlate ``x`` with a symmetric 1-D ``kernel`` along ``axis``, any layout.

    Shifted-slab accumulation -- one strided ``add_`` per tap -- rather than
    ``conv1d``/``conv3d``. A convolution wants the filtered axis contiguous, so
    on a ``(x, y, z, t)`` run it costs a transpose per axis per volume; and
    oneDNN's single-channel ``conv3d`` measured 7x slower than this on CPU.
    Every op here is elementwise over whatever strides ``x`` has, so a C- or
    F-ordered array is blurred in place of its layout, on any device.  A float32
    CUDA tensor goes to a one-pass Triton kernel instead (``blur_triton``): one
    read and one write per axis rather than ~3 per tap, 14x on a large run.

    ``reflect`` mirrors without repeating the edge sample (torch's
    ``reflect``), which needs ``radius < x.shape[axis]``; otherwise the
    border is zero-padded, as ``conv1d(padding=r)`` would.
    """
    n = x.shape[axis]
    r = (int(kernel.numel()) - 1) // 2
    if reflect and r >= n:
        raise ValueError(f"reflect padding needs radius < {n}, got {r}")
    fused = _fused_blur(x)
    if fused is not None:
        return fused(x, kernel, axis, reflect=reflect)
    taps = [float(w) for w in kernel.tolist()]
    out = x * taps[r]
    for j in range(1, r + 1):
        w = taps[r + j]
        if j < n:
            out.narrow(axis, 0, n - j).add_(x.narrow(axis, j, n - j), alpha=w)
            out.narrow(axis, j, n - j).add_(x.narrow(axis, 0, n - j), alpha=w)
        if reflect:
            # Past the high edge, i + j lands on 2(n-1) - (i + j); past the
            # low edge, i - j lands on j - i. Both are j samples, mirrored.
            out.narrow(axis, n - j, j).add_(x.narrow(axis, n - 1 - j, j).flip(axis), alpha=w)
            out.narrow(axis, 0, j).add_(x.narrow(axis, 1, j).flip(axis), alpha=w)
    return out


def gaussian3d_batched(
    vol: torch.Tensor,  # [P, X, Y, Z]
    sigma_vox: tuple[float, float, float],
) -> torch.Tensor:
    """Separable 3-D Gaussian, batched over the leading dim.

    Padding is reflect so the mask-normalised denominator stays well
    behaved at the brain boundary.
    """
    x = vol
    for ax, s in enumerate(sigma_vox):
        if s > 0:
            x = blur_axis(x, _gaussian_kernel_1d(s, x.dtype, x.device), ax + 1, reflect=True)
    return x


def blur_in_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    fwhm_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> torch.Tensor:
    """``(N, A, B, C)`` blurred inside ``mask``; voxels outside it are kept as they were.

    Normalised by the blurred mask, so a voxel at the mask's edge is an average
    of mask voxels only, not of them and the zeros (or noise) beyond -- AFNI's
    ``3dBlurInMask -preserve``. ``voxel_size_mm`` is per spatial axis of ``x``,
    in ``x``'s own axis order.
    """
    # Reflect padding needs a kernel narrower than the volume; cap it there.
    sigma = tuple(
        min(s, (n - 1) / 4.0)
        for s, n in zip(fwhm_mm_to_sigma_vox(fwhm_mm, voxel_size_mm), x.shape[1:], strict=True)
    )
    m = mask.to(x.dtype)
    num = gaussian3d_batched(torch.cat([x * m, m[None]]), sigma)  # type: ignore[arg-type]
    return torch.where(mask, num[:-1] / num[-1].clamp(min=1e-6), x)


def smooth_var_per_perm(
    var_pv: torch.Tensor,  # [P, V_in_mask], float32
    mask: np.ndarray,  # [X, Y, Z] bool
    sigma_vox: tuple[float, float, float],
    perm_chunk: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mask-aware Gaussian smoothing of a per-perm variance matrix.

    Mask-aware = divide ``smooth(var · 1_mask)`` by ``smooth(1_mask)``
    so out-of-mask zeros don't dilute voxels near the boundary.  This is
    the same correction used by ``3dBlurInMask``.

    Streams in chunks of ``perm_chunk`` perms so peak memory stays
    bounded regardless of P. ``None`` (the default) sizes the chunk from the
    memory module, capped at 256 perms.
    """
    p, v_mask = var_pv.shape
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    nx, ny, nz = mask.shape
    flat_idx = torch.from_numpy(np.flatnonzero(mask.ravel())).to(dev, dtype=torch.int64)
    assert flat_idx.numel() == v_mask, "mask voxel count must equal V_in_mask"
    n_total = nx * ny * nz

    # Pre-compute mask-smooth denominator once (1-channel, broadcast).
    mask_3d = torch.from_numpy(mask.astype(np.float32)).to(dev).unsqueeze(0)
    mask_smooth = gaussian3d_batched(mask_3d, sigma_vox).clamp_min(1e-10)  # [1, X, Y, Z]

    if perm_chunk is None:
        from fastfuncstuff.memory import estimate_chunk_size

        # A perm holds ~6 full-grid float32 volumes at the peak (the scatter
        # target, the blur's per-axis outputs, the normalised copy, the gather):
        # the GLM model's (5*T + R)*4 bytes with T = R = grid size is 24 bytes a
        # voxel. Per-perm results are independent, so the chunk never moves a bit.
        perm_chunk = estimate_chunk_size(
            p, n_total, n_total, dev, operation="glm", min_chunk_size=1, max_chunk_size=256
        )
    perm_chunk = max(1, int(perm_chunk))

    out = torch.empty_like(var_pv)
    for p0 in range(0, p, perm_chunk):
        p1 = min(p0 + perm_chunk, p)
        pc = p1 - p0
        chunk_cpu = var_pv[p0:p1]
        chunk = chunk_cpu.to(dev, non_blocking=True)
        vol_flat = torch.zeros((pc, n_total), dtype=chunk.dtype, device=dev)
        vol_flat[:, flat_idx] = chunk
        vol = vol_flat.view(pc, nx, ny, nz)
        smoothed = gaussian3d_batched(vol, sigma_vox)  # [Pc, X, Y, Z]
        smoothed = smoothed / mask_smooth  # mask-aware
        gathered = smoothed.view(pc, -1)[:, flat_idx]
        out[p0:p1] = gathered.cpu() if dev.type != "cpu" else gathered
    return out
