"""Fast GPU automasking for brain volumes (AFNI-compatible).

Implements the 3dAutomask algorithm on GPU using PyTorch:
    1. Iterative clip-level estimation (THD_cliplevel)
    2. Threshold → initial binary mask
    3. Peel erosion with neighbor-count threshold (THD_mask_erodemany)
    4. Largest connected component (6-connectivity, THD_mask_clust)
    5. Small hole fill (opposite-side, THD_mask_fillin_once)
    6. Large hole fill (distance-based, THD_mask_fillin_completely)
    7. Interior hole fill (invert → cluster → invert)

Reference: https://github.com/afni/afni/blob/master/src/thd_automask.c

Key function:
    automask(vol) -> binary mask (nz, ny, nx) bool tensor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Clip-level estimation (matches THD_cliplevel)
# ---------------------------------------------------------------------------

def _cliplevel(vol: Tensor, mfrac: float = 0.5) -> float:
    """Estimate intensity clip level using AFNI's iterative median algorithm.

    The algorithm finds a threshold that separates tissue from background by
    iteratively computing the median of above-threshold voxels and lowering
    the threshold to ``mfrac * median``.

    This matches THD_cliplevel() in AFNI's thd_cliplevel.c.

    Parameters
    ----------
    vol : Tensor
        3D volume.
    mfrac : float
        Fraction of median to use as clip level (AFNI default: 0.5).
    """
    v = vol.reshape(-1).float()
    pos = v[v > 0]
    if pos.numel() < 224:
        return 1.0

    # Initial cut: include upper ~65% of positive voxels
    # AFNI uses sqrt(mean(x^2)) as initial rough estimate, then scales by
    # the histogram position corresponding to ~35th percentile.
    # We approximate this with a percentile approach.
    ncut = float(pos.quantile(0.35).item())
    if ncut <= 0:
        ncut = float(pos.quantile(0.50).item())

    # Iterative convergence: median above cut → new cut = mfrac * median
    for _ in range(66):
        above = pos[pos >= ncut]
        if above.numel() < 10:
            break
        median_val = float(above.median().item())
        new_cut = mfrac * median_val
        if abs(new_cut - ncut) < 0.01 * ncut:
            ncut = new_cut
            break
        ncut = new_cut

    return ncut


# ---------------------------------------------------------------------------
# 6-connectivity morphological operations (matching AFNI)
# ---------------------------------------------------------------------------

def _dilate_6conn(mask: Tensor, iterations: int = 1) -> Tensor:
    """Dilate with 6-connectivity (face neighbors only).

    Uses a 3D convolution with a cross-shaped kernel instead of max_pool3d
    (which gives 26-connectivity).
    """
    if iterations <= 0:
        return mask
    # 6-connectivity kernel: center + 6 face neighbors
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1  # center
    kernel[0, 0, 0, 1, 1] = 1  # -z
    kernel[0, 0, 2, 1, 1] = 1  # +z
    kernel[0, 0, 1, 0, 1] = 1  # -y
    kernel[0, 0, 1, 2, 1] = 1  # +y
    kernel[0, 0, 1, 1, 0] = 1  # -x
    kernel[0, 0, 1, 1, 2] = 1  # +x

    x = mask.float()[None, None]
    for _ in range(iterations):
        x = F.conv3d(x, kernel, padding=1)
        x = (x > 0.5).float()
    return x[0, 0] > 0.5


# 18-connectivity kernel for neighbor counting (matching AFNI's NN2)
def _count_neighbors_18(mask: Tensor) -> Tensor:
    """Count number of set 18-neighbors for each voxel.

    AFNI uses 18-connectivity (NN2: face + edge neighbors) for its
    peel/erosion threshold check.
    """
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    # 6 face neighbors
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1
    # 12 edge neighbors
    kernel[0, 0, 0, 0, 1] = 1
    kernel[0, 0, 0, 2, 1] = 1
    kernel[0, 0, 2, 0, 1] = 1
    kernel[0, 0, 2, 2, 1] = 1
    kernel[0, 0, 0, 1, 0] = 1
    kernel[0, 0, 0, 1, 2] = 1
    kernel[0, 0, 2, 1, 0] = 1
    kernel[0, 0, 2, 1, 2] = 1
    kernel[0, 0, 1, 0, 0] = 1
    kernel[0, 0, 1, 0, 2] = 1
    kernel[0, 0, 1, 2, 0] = 1
    kernel[0, 0, 1, 2, 2] = 1

    x = mask.float()[None, None]
    counts = F.conv3d(x, kernel, padding=1)
    return counts[0, 0]


def _peel_once(mask: Tensor, peelthr: int = 17) -> Tensor:
    """Remove mask voxels with fewer than peelthr of 18 neighbors set.

    Matches AFNI's THD_mask_erodemany single-pass logic:
    voxels with < peelthr neighbors (out of 18) are cleared.
    """
    counts = _count_neighbors_18(mask)
    # Keep only voxels that have enough neighbors
    return mask & (counts >= peelthr)


def _peel(mask: Tensor, peelcount: int = 1, peelthr: int = 17) -> Tensor:
    """AFNI-style erosion: peel voxels with < peelthr/18 neighbors.

    Matches THD_mask_erodemany.
    """
    for _ in range(peelcount):
        mask = _peel_once(mask, peelthr)
    return mask


# ---------------------------------------------------------------------------
# Connected component (6-connectivity, matching AFNI's THD_mask_clust)
# ---------------------------------------------------------------------------

def _largest_component_6conn(mask: Tensor, vol: Tensor | None = None) -> Tensor:
    """Keep largest connected component using 6-connectivity.

    Uses iterative dilation from a seed (brightest voxel inside mask)
    with a 6-connectivity kernel, AND-ed with the mask at each step.
    """
    if mask.sum() == 0:
        return mask

    # Seed at brightest voxel
    if vol is not None:
        vals = vol.abs() * mask.float()
        seed_idx = int(vals.reshape(-1).argmax().item())
    else:
        seed_idx = int(mask.float().reshape(-1).argmax().item())

    nz, ny, nx = mask.shape
    seed = torch.zeros(nz * ny * nx, device=mask.device, dtype=torch.float32)
    seed[seed_idx] = 1.0
    seed = seed.view(nz, ny, nx)

    # 6-connectivity kernel
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1

    mask_5d = mask.float()[None, None]
    seed_5d = seed[None, None]

    prev_count = 1
    check_every = 10
    max_iter = nz + ny + nx  # worst case: L1 diagonal
    for i in range(max_iter):
        seed_5d = F.conv3d(seed_5d, kernel, padding=1)
        seed_5d = (seed_5d > 0.5).float() * mask_5d
        if (i + 1) % check_every == 0 or i == max_iter - 1:
            new_count = int((seed_5d > 0.5).sum().item())
            if new_count == prev_count:
                break
            prev_count = new_count

    return seed_5d[0, 0] > 0.5


# ---------------------------------------------------------------------------
# Hole filling (matches THD_mask_fillin_once / THD_mask_fillin_completely)
# ---------------------------------------------------------------------------

def _fillin_once(mask: Tensor, nside: int = 1) -> Tensor:
    """Fill voxels that have mask on opposite sides within distance nside.

    Matches THD_mask_fillin_once: a background voxel is filled if for any
    axis (x, y, z), there is a set voxel within nside steps in the positive
    AND negative direction along that axis.
    """
    m = mask.clone()
    bg = ~mask

    for axis in range(3):
        for d in range(1, nside + 1):
            sz = mask.shape[axis]
            if 2 * d >= sz:
                continue

            # For voxel at position i (in range [d, sz-d)):
            #   neighbor at i+d  →  pos_slice
            #   neighbor at i-d  →  neg_slice
            #   the voxel itself →  center_slice
            pos_slice = [slice(None)] * 3
            pos_slice[axis] = slice(2 * d, sz)         # i+d for i in [d, sz-d)
            neg_slice = [slice(None)] * 3
            neg_slice[axis] = slice(0, sz - 2 * d)     # i-d for i in [d, sz-d)
            center_slice = [slice(None)] * 3
            center_slice[axis] = slice(d, sz - d)       # i in [d, sz-d)

            # Voxels that have a set neighbor at +d AND -d along this axis
            has_both = mask[tuple(pos_slice)] & mask[tuple(neg_slice)]
            # Only fill background voxels
            fill_zone = has_both & bg[tuple(center_slice)]
            m[tuple(center_slice)] |= fill_zone

    return m


def _fillin_completely(mask: Tensor, nside: int) -> Tensor:
    """Iterate fillin_once until no new voxels are added.

    Matches THD_mask_fillin_completely.
    """
    for _ in range(100):
        new_mask = _fillin_once(mask, nside)
        added = int(new_mask.sum().item()) - int(mask.sum().item())
        mask = new_mask
        if added == 0:
            break
    return mask


def _fill_holes_3d(mask: Tensor) -> Tensor:
    """Fill interior holes by flood-filling background from border.

    Matches AFNI's approach: invert → keep largest component (exterior) → invert.
    Any background region not connected to the border is filled.
    """
    nz, ny, nx = mask.shape

    bg = (~mask).float()
    seed = torch.zeros_like(bg)

    # Mark all border voxels that are background as seeds
    seed[0, :, :] = bg[0, :, :]
    seed[-1, :, :] = bg[-1, :, :]
    seed[:, 0, :] = bg[:, 0, :]
    seed[:, -1, :] = bg[:, -1, :]
    seed[:, :, 0] = bg[:, :, 0]
    seed[:, :, -1] = bg[:, :, -1]

    # 6-connectivity flood fill from border
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1

    seed_5d = seed[None, None]
    bg_5d = bg[None, None]

    check_every = 10
    max_iter = nz + ny + nx
    prev_count = int((seed_5d > 0.5).sum().item())
    for i in range(max_iter):
        seed_5d = F.conv3d(seed_5d, kernel, padding=1)
        seed_5d = (seed_5d > 0.5).float() * bg_5d
        if (i + 1) % check_every == 0 or i == max_iter - 1:
            new_count = int((seed_5d > 0.5).sum().item())
            if new_count == prev_count:
                break
            prev_count = new_count

    exterior = seed_5d[0, 0] > 0.5
    return ~exterior


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def automask(
    vol: Tensor,
    clip_frac: float = 0.5,
    dilate_extra: int = 0,
    peelcount: int = 1,
    peelthr: int = 17,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tensor:
    """Create a binary brain mask from a 3D volume (AFNI-compatible).

    Follows the 3dAutomask algorithm:
        1. Compute clip level via iterative median (THD_cliplevel)
        2. Threshold to create initial mask
        3. Peel erosion: remove voxels with < peelthr/18 neighbors
        4. Keep largest 6-connected component
        5. Fill small holes (opposite-side fillin, nside=1, x3)
        6. Fill large holes (distance-based, ~1.6% of volume dims)
        7. Peel once more + recluster
        8. Fill interior holes (flood from border)

    Parameters
    ----------
    vol : Tensor
        (nz, ny, nx) float tensor — 3D volume.
    clip_frac : float
        Fraction parameter for THD_cliplevel (default 0.5, matching AFNI).
    dilate_extra : int
        Extra dilation iterations after the algorithm (default 0).
    peelcount : int
        Number of peel iterations (AFNI default: 1).
    peelthr : int
        Minimum 18-neighbors to survive peeling (AFNI default: 17).
    device : torch.device, optional
        Device to run on. Defaults to vol.device.

    Returns
    -------
    Tensor
        (nz, ny, nx) bool tensor — binary brain mask.
    """
    if device is not None:
        vol = vol.to(device)

    nz, ny, nx = vol.shape

    # Step 1: clip level (matches THD_cliplevel)
    clip = _cliplevel(vol, mfrac=clip_frac)
    if verbose:
        print(f"  automask: shape=({nz},{ny},{nx}) clip={clip:.4f} "
              f"range=[{vol.min().item():.4f}, {vol.max().item():.4f}]")

    # Step 2: threshold
    mask = vol.abs() >= clip
    if verbose:
        print(f"  automask: after threshold: {int(mask.sum().item()):,} voxels")

    # Step 3: peel erosion (THD_mask_erodemany)
    mask = _peel(mask, peelcount=peelcount, peelthr=peelthr)
    if verbose:
        print(f"  automask: after peel: {int(mask.sum().item()):,} voxels")

    # Step 4: largest connected component (6-connectivity)
    mask = _largest_component_6conn(mask, vol=vol)
    if verbose:
        print(f"  automask: after cluster: {int(mask.sum().item()):,} voxels")

    # Step 5: small hole fill (3 rounds of fillin_once with nside=1)
    for _ in range(3):
        mask = _fillin_once(mask, nside=1)
    if verbose:
        print(f"  automask: after small fill: {int(mask.sum().item()):,} voxels")

    # Step 6: large hole fill (nside = ~1.6% of each dim, take max)
    nside_large = max(
        round(0.016 * nx),
        round(0.016 * ny),
        round(0.016 * nz),
        1,
    )
    mask = _fillin_completely(mask, nside=nside_large)
    if verbose:
        print(f"  automask: after large fill (nside={nside_large}): {int(mask.sum().item()):,} voxels")

    # Step 7: fill interior holes (flood from border)
    mask = _fill_holes_3d(mask)
    if verbose:
        print(f"  automask: after hole fill: {int(mask.sum().item()):,} voxels")

    # Optional extra dilation
    if dilate_extra > 0:
        mask = _dilate_6conn(mask, iterations=dilate_extra)
        if verbose:
            print(f"  automask: after dilate({dilate_extra}): {int(mask.sum().item()):,} voxels")

    return mask
