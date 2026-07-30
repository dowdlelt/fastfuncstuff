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


def _quantile(x: Tensor, q: float) -> float:
    """Quantile that tolerates large tensors.

    ``torch.quantile`` refuses inputs above 2**24 elements ("input tensor is too
    large"), which a full-resolution anatomical easily exceeds. For those we fall
    back to ``kthvalue`` (nearest-rank, no interpolation) — plenty precise for a
    clip-level estimate.
    """
    n = x.numel()
    if n <= (1 << 24):
        return float(x.quantile(q).item())
    k = min(n, max(1, int(round(q * (n - 1))) + 1))
    return float(x.kthvalue(k).values.item())


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
    ncut = _quantile(pos, 0.35)
    if ncut <= 0:
        ncut = _quantile(pos, 0.50)

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


def _dilate_6conn(mask: Tensor, iterations: int = 2) -> Tensor:
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


def _erode_6conn(mask: Tensor, iterations: int = 1) -> Tensor:
    """Erode with 6-connectivity: a voxel survives only if it AND all 6 face neighbors
    are set. The morphological dual of :func:`_dilate_6conn` (voxels within
    ``iterations`` of the boundary — including the FoV edge — are peeled)."""
    if iterations <= 0:
        return mask
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
        s = F.conv3d(x, kernel, padding=1)  # zero-pad → FoV-edge voxels lose neighbours
        x = (s >= 6.5).float()  # all 7 (self + 6 faces) set
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

    Erode-only. This is the *first half* of THD_mask_erodemany; see
    :func:`erode_many` for the faithful peel-then-redilate version.
    """
    for _ in range(peelcount):
        mask = _peel_once(mask, peelthr)
    return mask


def _count_neighbors_18_replicate(mask: Tensor) -> Tensor:
    """18-neighbor count with edge voxels replicated, as AFNI counts them.

    AFNI clamps the neighbor index at the volume face (``if(ii==0) im=0``), so a
    boundary voxel sees itself in place of the missing neighbor. Zero-padding
    instead makes every boundary voxel look under-connected, which erodes a shell
    off any mask that reaches the matrix edge.
    """
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    for dz, dy, dx in [
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 1, 0),
        (1, 1, 2),  # 6 face
        (0, 0, 1),
        (0, 2, 1),
        (2, 0, 1),
        (2, 2, 1),
        (0, 1, 0),
        (0, 1, 2),
        (2, 1, 0),
        (2, 1, 2),
        (1, 0, 0),
        (1, 0, 2),
        (1, 2, 0),
        (1, 2, 2),  # 12 edge
    ]:
        kernel[0, 0, dz, dy, dx] = 1
    x = F.pad(mask.float()[None, None], (1,) * 6, mode="replicate")
    return F.conv3d(x, kernel)[0, 0]


def erode_many(mask: Tensor, npeel: int = 1, peelthr: int = 17) -> Tensor:
    """Peel ``npeel`` layers off a mask, then re-dilate — AFNI ``THD_mask_erodemany``.

    Each pass marks (simultaneously, not sequentially) every set voxel with fewer
    than ``peelthr`` of 18 neighbours set, recording the layer it fell in, then
    removes them. The re-dilate pass then walks layers back outward and restores
    any peeled voxel still touching a survivor — more than one neighbour for the
    outer layers, at least one for the innermost.

    The round trip is what makes this a *shape* filter rather than an erosion: a
    solid boundary comes back, a one-voxel-thick bridge or speck does not. Skip
    the re-dilate and every result shrinks by a full shell.
    """
    if npeel < 1 or mask.numel() < 27:
        return mask

    thr = min(18, peelthr)
    layer = torch.zeros(mask.shape, dtype=torch.int16, device=mask.device)
    cur = mask.clone()
    for pp in range(1, npeel + 1):
        newly = cur & (_count_neighbors_18_replicate(cur) < thr)
        layer[newly] = pp
        cur = cur & ~newly

    for pp in range(npeel, 0, -1):
        # The innermost layer only needs one surviving neighbour; outer layers
        # need two, so the mask cannot regrow along a single-voxel filament.
        bth = 0 if pp == npeel else 1
        counts = _count_neighbors_18_replicate(cur)
        cur = cur | ((layer >= pp) & ~cur & (counts > bth))

    return cur


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
            pos_slice[axis] = slice(2 * d, sz)  # i+d for i in [d, sz-d)
            neg_slice = [slice(None)] * 3
            neg_slice[axis] = slice(0, sz - 2 * d)  # i-d for i in [d, sz-d)
            center_slice = [slice(None)] * 3
            center_slice[axis] = slice(d, sz - d)  # i in [d, sz-d)

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
    dilate_extra: int = 2,
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
        Extra dilation iterations after the algorithm (default 1).
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

    # A single NaN poisons the whole thing: the clip level comes out NaN, every
    # `vol >= clip` comparison is False, and the mask is empty -- a silent 0%-brain
    # answer rather than an error. NaN is not brain, so treat it as background.
    if not bool(torch.isfinite(vol).all()):
        vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: clip level (matches THD_cliplevel)
    clip = _cliplevel(vol, mfrac=clip_frac)
    if verbose:
        print(
            f"  automask: shape=({nz},{ny},{nx}) clip={clip:.4f} "
            f"range=[{vol.min().item():.4f}, {vol.max().item():.4f}]"
        )

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
        print(
            f"  automask: after large fill (nside={nside_large}): {int(mask.sum().item()):,} voxels"
        )

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


def data_coverage_mask(
    vol: Tensor,
    erode: int = 1,
    device: torch.device | None = None,
) -> Tensor:
    """Voxels where ``vol`` actually holds acquired data: finite and not exactly zero.

    Distinct from :func:`automask`: this is not "where is the brain" but "where did
    the scanner (and any resampling since) put a real number". A volume that has been
    rotated onto another grid — say by ``ffs_allineate`` after the subject turned
    their head out of the FoV — carries a hard zero wedge where the source had no
    data. A registration metric evaluated across that wedge sees a step edge and
    happily stretches real tissue into it, so callers intersect this with their own
    weight/mask to keep the metric inside the shared support of both images.

    NaN/Inf count as no-data, and in practice are the commonest spelling of it: any
    upstream step that divides by the data (a scaling or normalisation) turns the
    exact-zero rim into NaN, so a volume can hold a fully empty slab and not contain
    a single zero. Testing ``!= 0`` alone silently passes every one of those voxels
    through as valid data.

    ``erode`` peels the coverage boundary (6-connectivity) to drop the ramp of
    partial-value voxels that linear/sinc resampling leaves one voxel inside the
    empty wedge — nonzero, but a blend of tissue and nothing. A volume that is finite
    and nonzero throughout has no wedge to protect, and is returned all-true without
    erosion so this is a no-op on full-FoV data.
    """
    if device is not None:
        vol = vol.to(device)

    cover = torch.isfinite(vol) & (vol != 0)
    if bool(cover.all()):
        return cover
    return _erode_6conn(cover, iterations=erode)


def cross_fill_no_data(
    fixed: Tensor,
    moving: Tensor,
    fixed_cover: Tensor | None,
    moving_cover: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Make a pair of images safe for a registration metric to compare.

    Returns ``(fixed_metric, moving_metric, cover)``: copies of the two images with
    each one's no-data region filled from the other, plus the shared support.

    Excluding a no-data region from the metric is not enough on its own, and the
    reason is easy to miss: exclusion stops the warp being *rewarded* for reaching
    into the void, but it leaves the void's edge in plain view. Every local window
    within the metric's radius of the boundary still straddles a cliff between tissue
    and nothing, and that cliff is a strong feature the warp will try to align to
    something. Filling the void from the other image makes the pair agree exactly
    there, so the cliff is gone and the local gradient is ~0 -- the metric becomes
    genuinely *indifferent* to the region rather than being fenced out of it.

    Measured on a clipped 9.4T pair (max |dz| in the six slices above the source's
    data floor): 12.81 unrestricted, 8.89 with brain masks, 7.99 adding coverage
    exclusion, 0.69 adding this fill. The hard edge, not the weighting, is what drives
    the artifact.

    Callers must use the returned images for the METRIC ONLY and keep the originals
    for the final resample -- filling the output would fabricate anatomy into a saved
    image. Pass only *data coverage* here, never a brain automask: an automask
    boundary is real anatomy, and cross-filling across it would splice one image's
    skull into the other's, which a cross-modal metric would follow.
    """
    f_metric, m_metric = fixed, moving
    if fixed_cover is not None:
        fixed_cover = fixed_cover.to(fixed.device) > 0
        f_metric = torch.where(fixed_cover, fixed, moving)
    if moving_cover is not None:
        moving_cover = moving_cover.to(moving.device) > 0
        m_metric = torch.where(moving_cover, moving, fixed)

    if fixed_cover is not None and moving_cover is not None:
        cover = fixed_cover & moving_cover
    else:
        cover = fixed_cover if moving_cover is None else moving_cover
    return f_metric, m_metric, cover
