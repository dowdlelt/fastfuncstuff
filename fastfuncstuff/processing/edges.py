"""Thin edge maps, for alignment QC overlays and as a general image transform.

The recipe follows what AFNI's ``@djunct_edgy_align_check`` (the edge overlay in
``@SSwarper`` / ``afni_proc.py`` QC) feeds ``@chauffeur_afni``: a small median filter
to knock out speckle, a gradient-magnitude edge detector, then a weak threshold so
only the clearly-not-noise boundaries remain. AFNI's detector is ``3dedge3``
(Monga-Deriche recursive filtering with non-maximum suppression); this uses the
same structure -- Gaussian derivative, then suppression along the gradient -- which
gives the same one-voxel-thin ridges without the recursive filter.

Thinness is the point of the overlay. A raw gradient magnitude is a band several
voxels wide, and a misalignment of one voxel is invisible inside a band that wide.

2-D vs 3-D for display. A 3-D edge map cut by a slice is not an outline: where a
surface runs nearly parallel to the slice, suppression along its (through-plane)
gradient keeps the whole tangent sheet, and the cut shows a filled patch. Overlays
should find edges *in the displayed plane* -- pass a 2-D image.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fastfuncstuff.memory import get_available_memory

from .cost import _separable_smooth_3d
from .mask import _quantile


def _median_cross(img: Tensor) -> Tensor:
    """Median over the face-neighbour cross: 5 pixels in 2-D, 7 voxels in 3-D
    (``3dMedianFilter -irad 1.01``)."""
    nd = img.ndim
    padded = F.pad(img[None, None], (1, 1) * nd, mode="replicate")[0, 0]
    centre = (slice(1, -1),) * nd
    stack = [padded[centre]]
    for axis in range(nd):
        for lo, hi in ((None, -2), (2, None)):
            idx = list(centre)
            idx[axis] = slice(lo, hi)
            stack.append(padded[tuple(idx)])
    return torch.stack(stack).median(dim=0).values


def _suppress_non_maxima(mag: Tensor, step: tuple[Tensor, ...]) -> Tensor:
    """Zero every element that is not a maximum of ``mag`` along its own gradient.

    ``step`` is the unit gradient direction in index units, one tensor per axis in
    array order. Streamed in slabs along the first axis: the sample grid is ``ndim``
    full-size float tensors per neighbour.
    """
    nd = mag.ndim
    lead, rest = mag.shape[0], mag.shape[1:]
    device = mag.device
    per_row = 1
    for n in rest:
        per_row *= n
    # Grid (ndim) + a sampled neighbour + the keep mask, per element, float32.
    bytes_per_elem = 4 * (nd + 3)
    slab = max(1, min(lead, get_available_memory(device) // max(1, bytes_per_elem * per_row)))
    rest_axes = torch.meshgrid(
        *(torch.arange(n, device=device, dtype=mag.dtype) for n in rest), indexing="ij"
    )
    sizes = [max(n - 1, 1) for n in mag.shape]
    src = mag[None, None]
    out = torch.zeros_like(mag)
    for a0 in range(0, lead, slab):
        a1 = min(lead, a0 + slab)
        lead_axis = torch.arange(a0, a1, device=device, dtype=mag.dtype).view(-1, *([1] * (nd - 1)))
        coords = (lead_axis, *rest_axes)
        local = [s[a0:a1] for s in step]
        keep = torch.ones_like(mag[a0:a1], dtype=torch.bool)
        for sign in (1.0, -1.0):
            # grid_sample wants the last axis first (x, y[, z]).
            grid = torch.stack(
                [
                    2.0 * (coords[ax] + sign * local[ax]) / sizes[ax] - 1.0
                    for ax in reversed(range(nd))
                ],
                dim=-1,
            )[None]
            neighbour = F.grid_sample(
                src, grid, mode="bilinear", padding_mode="border", align_corners=True
            )[0, 0]
            # >= on one side and > on the other: a plateau two voxels wide keeps one.
            keep &= mag[a0:a1] >= neighbour if sign > 0 else mag[a0:a1] > neighbour
        out[a0:a1] = torch.where(keep, mag[a0:a1], out[a0:a1])
    return out


def edge_map(
    img: Tensor,
    *,
    sigma: float = 1.0,
    spacing: tuple[float, ...] | None = None,
    median: bool = True,
    thin: bool = True,
    threshold: float = 0.1,
    mask: Tensor | None = None,
) -> Tensor:
    """Edge strength of a 2-D or 3-D image: non-zero only on retained edges.

    Args:
        img: (ny, nx) or (nz, ny, nx) image.
        sigma: Gaussian pre-smoothing in voxels before differentiating. Larger keeps
            only coarser boundaries. 0 differentiates the raw (median-filtered) image.
        spacing: Voxel size per axis in array order, in mm. The gradient is taken
            per mm so an anisotropic grid does not favour edges across its thick axis.
            Default: isotropic.
        median: Apply the face-neighbour median first, as AFNI's QC does.
        thin: Non-maximum suppression along the gradient -- one-voxel-wide ridges.
        threshold: Drop edges weaker than this fraction of the 99th percentile of
            edge strength. Relative, so it is independent of the image's units.
        mask: Optional region of the same shape; edges outside it are zeroed.

    Returns:
        Float tensor shaped like ``img``: gradient magnitude on kept edges, 0 elsewhere.
    """
    nd = img.ndim
    if nd not in (2, 3):
        raise ValueError(f"edge_map needs a 2-D or 3-D image, got shape {tuple(img.shape)}")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    spacing = tuple(float(s) or 1.0 for s in (spacing or (1.0,) * nd))
    if len(spacing) != nd:
        raise ValueError(f"spacing needs {nd} values, got {len(spacing)}")

    v = img.float()
    if median:
        v = _median_cross(v)
    if sigma > 0:
        # The smoother is 3-D; a singleton leading axis is skipped, so 2-D rides along.
        v = _separable_smooth_3d(v if nd == 3 else v[None], sigma)
        v = v if nd == 3 else v[0]

    grads = torch.gradient(v, spacing=spacing)
    del v
    mag = torch.sqrt(sum(g * g for g in grads))  # type: ignore[arg-type]

    if thin:
        # The suppression walks one voxel along the gradient, which is a direction
        # in index space: a per-mm slope converts to per-voxel by multiplying back.
        per_voxel = [g * s for g, s in zip(grads, spacing, strict=True)]
        del grads
        norm = torch.sqrt(sum(p * p for p in per_voxel))  # type: ignore[arg-type]
        norm = norm.clamp_min(torch.finfo(mag.dtype).tiny)
        mag = _suppress_non_maxima(mag, tuple(p / norm for p in per_voxel))
    else:
        del grads

    if mask is not None:
        mag = mag * (mask > 0)

    nonzero = mag[mag > 0]
    if threshold > 0 and nonzero.numel():
        floor = threshold * _quantile(nonzero, 0.99)
        mag = torch.where(mag >= floor, mag, torch.zeros_like(mag))
    return mag
