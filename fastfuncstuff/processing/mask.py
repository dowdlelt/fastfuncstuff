"""Fast GPU automasking for brain volumes.

Creates binary brain masks using intensity thresholding, morphological
operations (via 3D max/min pooling), connected-component extraction,
and hole filling -- all on GPU without scipy.

Key function:
    automask(vol) -> binary mask (nz, ny, nx) bool tensor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .weight import _clip_level


def automask(
    vol: Tensor,
    clip_frac: float = 0.3,
    dilate_extra: int = 2,
    device: torch.device | None = None,
) -> Tensor:
    """Create a binary brain mask from a 3D volume.

    Algorithm:
        1. Threshold at clip_frac * clip_level
        2. Erode 1 iteration (remove thin noise bridges)
        3. Keep largest connected component (iterative GPU dilation from seed)
        4. Dilate to recover eroded edges + extra padding
        5. Fill interior holes via border flood-fill

    Args:
        vol: (nz, ny, nx) float tensor.
        clip_frac: Fraction of clip level to use as threshold (default 0.3).
        dilate_extra: Extra dilation iterations beyond erosion recovery (default 2).
        device: Device to run on. Defaults to vol.device.

    Returns:
        (nz, ny, nx) bool tensor -- binary brain mask.
    """
    if device is not None:
        vol = vol.to(device)

    # Step 1: intensity threshold
    clip = _clip_level(vol, frac=0.5)
    threshold = clip * clip_frac
    mask = vol.abs() > threshold

    # Step 2: erode to remove thin bridges / noise
    eroded = _erode_3d(mask, iterations=1)

    # Step 3: keep largest connected component
    largest = _largest_component_gpu(eroded, vol=vol)

    # Step 4: dilate back (1 to undo erosion + extra)
    dilated = _dilate_3d(largest, iterations=1 + dilate_extra)

    # Intersect with the original liberal threshold to avoid expanding into
    # background -- use a slightly more liberal threshold for the boundary
    liberal_mask = vol.abs() > (threshold * 0.5)
    dilated = dilated & liberal_mask

    # Step 5: fill interior holes
    filled = _fill_holes_3d(dilated)

    return filled


def _erode_3d(mask: Tensor, iterations: int = 1) -> Tensor:
    """Morphological erosion via min-pooling (max-pool on negated mask).

    Args:
        mask: (nz, ny, nx) bool tensor.
        iterations: Number of erosion steps.

    Returns:
        (nz, ny, nx) bool tensor.
    """
    if iterations <= 0:
        return mask
    # min-pool = negate -> max-pool -> negate
    x = (~mask).float()[None, None]  # (1, 1, D, H, W)
    for _ in range(iterations):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x[0, 0] < 0.5  # invert back


def _dilate_3d(mask: Tensor, iterations: int = 1) -> Tensor:
    """Morphological dilation via max-pooling.

    Args:
        mask: (nz, ny, nx) bool tensor.
        iterations: Number of dilation steps.

    Returns:
        (nz, ny, nx) bool tensor.
    """
    if iterations <= 0:
        return mask
    x = mask.float()[None, None]  # (1, 1, D, H, W)
    for _ in range(iterations):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x[0, 0] > 0.5


def _largest_component_gpu(mask: Tensor, vol: Tensor | None = None) -> Tensor:
    """Keep largest connected component via iterative dilation from seed.

    Seed is placed at the argmax of vol * mask (brightest voxel inside the
    mask). From there, iterative 3x3x3 dilation AND-ed with the mask
    grows the connected region. Converges when no new voxels are added.

    Args:
        mask: (nz, ny, nx) bool tensor.
        vol: (nz, ny, nx) optional volume for seed selection. If None,
             seed is placed at center of mass of mask.

    Returns:
        (nz, ny, nx) bool tensor -- largest connected component.
    """
    if mask.sum() == 0:
        return mask

    # Choose seed: brightest voxel inside mask
    if vol is not None:
        vals = vol.abs() * mask.float()
        seed_idx = vals.reshape(-1).argmax()
    else:
        seed_idx = mask.float().reshape(-1).argmax()

    nz, ny, nx = mask.shape
    seed = torch.zeros_like(mask, dtype=torch.float32)
    seed.reshape(-1)[seed_idx] = 1.0

    mask_f = mask.float()
    seed_5d = seed[None, None]  # (1, 1, D, H, W)
    mask_5d = mask_f[None, None]

    prev_count = 1
    check_every = 10
    max_iter = max(nz, ny, nx)
    for i in range(max_iter):  # worst case: diagonal of volume
        # Dilate seed
        seed_5d = F.max_pool3d(seed_5d, kernel_size=3, stride=1, padding=1)
        # AND with mask
        seed_5d = seed_5d * mask_5d
        # Check convergence periodically to avoid GPU sync overhead
        if (i + 1) % check_every == 0 or i == max_iter - 1:
            new_count = int((seed_5d > 0.5).sum().item())
            if new_count == prev_count:
                break  # converged
            prev_count = new_count

    return seed_5d[0, 0] > 0.5


def _fill_holes_3d(mask: Tensor) -> Tensor:
    """Fill interior holes by flood-filling background from border, then inverting.

    Any background voxel not reachable from the border is an interior hole
    and gets filled.

    Args:
        mask: (nz, ny, nx) bool tensor.

    Returns:
        (nz, ny, nx) bool tensor with holes filled.
    """
    nz, ny, nx = mask.shape

    # Background = ~mask. Seed flood from all border voxels.
    bg = (~mask).float()
    seed = torch.zeros_like(bg)

    # Mark all border voxels that are background as seeds
    seed[0, :, :] = bg[0, :, :]
    seed[-1, :, :] = bg[-1, :, :]
    seed[:, 0, :] = bg[:, 0, :]
    seed[:, -1, :] = bg[:, -1, :]
    seed[:, :, 0] = bg[:, :, 0]
    seed[:, :, -1] = bg[:, :, -1]

    # Iterative dilation of seed AND-ed with background
    seed_5d = seed[None, None]
    bg_5d = bg[None, None]

    check_every = 10
    max_iter = max(nz, ny, nx)
    prev_count = int((seed_5d > 0.5).sum().item())
    for i in range(max_iter):
        seed_5d = F.max_pool3d(seed_5d, kernel_size=3, stride=1, padding=1)
        seed_5d = seed_5d * bg_5d
        # Check convergence periodically to avoid GPU sync overhead
        if (i + 1) % check_every == 0 or i == max_iter - 1:
            new_count = int((seed_5d > 0.5).sum().item())
            if new_count == prev_count:
                break
            prev_count = new_count

    # Exterior background = reachable from border
    exterior = seed_5d[0, 0] > 0.5
    # Filled mask = everything that is NOT exterior background
    return ~exterior
