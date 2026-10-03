"""Fast GPU automasking for brain volumes (AFNI-compatible).

Implements 3dAutomask (mri_automask_image + -dilate), bit-identical to AFNI:
    1. Gradual (octant-blended) histogram clip level (THD_cliplevel_gradual)
    2. Threshold, largest 6-connected cluster
    3. Peel + re-dilate (THD_mask_erodemany), recluster
    4. Small and distance-based hole fill (THD_mask_fillin_once / _completely)
    5. Final erode + recluster, interior hole fill
    6. Optional -dilate (THD_mask_dilate + fill-in)

Reference: https://github.com/afni/afni/blob/master/src/thd_automask.c

Key function:
    automask(vol, dilate_extra=0) -> binary mask (nz, ny, nx) bool tensor
"""

from __future__ import annotations

import numpy
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


# ---------------------------------------------------------------------------
# 6-connectivity dilation and flood fill
# ---------------------------------------------------------------------------

# How often the flood fill polls for convergence.  Each poll is a device
# synchronisation; each extra iteration is one cheap dilation.  Overshooting by
# a few dilations beats syncing after every one -- 4 measured fastest on both a
# 96x96x60 EPI and a 160x160x120 anatomical.
_FLOOD_CHECK_EVERY = 4


def _dilate_6conn_once(x: Tensor) -> Tensor:
    """One 6-connectivity dilation of a 5-D {0,1} volume.

    Three axis-aligned max-pools, unioned, are exactly the 6-neighbourhood, and
    about half the cost of the cross-shaped ``conv3d`` that computes the same
    thing through a multiply-accumulate plus cuDNN algorithm selection.  The two
    forms agree voxel-for-voxel; the test suite pins that.
    """
    grown = torch.maximum(
        torch.maximum(
            F.max_pool3d(x, (3, 1, 1), 1, (1, 0, 0)),
            F.max_pool3d(x, (1, 3, 1), 1, (0, 1, 0)),
        ),
        F.max_pool3d(x, (1, 1, 3), 1, (0, 0, 1)),
    )
    return torch.maximum(grown, x)


def _flood_fill_6conn(seed: Tensor, allowed: Tensor) -> Tensor:
    """Grow *seed* through *allowed* under 6-connectivity until it stops.

    Both arguments are 3-D {0,1} float volumes.  Returns the grown region as a
    bool volume.  The iteration bound is the worst-case L1 diameter, which a
    head-shaped region never approaches.

    On the GPU this is an iterated dilation: one cheap kernel per L1 step, all
    of it parallel.  On the CPU the same loop is a disaster -- ~60 passes over
    the whole volume, seconds for one EPI automask -- so there we label the
    components in a single pass instead and keep the ones the seed reaches.
    Both paths return the same voxels; the test suite pins that.
    """
    nz, ny, nx = allowed.shape
    if allowed.device.type == "cpu":
        return _flood_fill_6conn_labelled(seed, allowed)
    current = seed[None, None]
    allowed_5d = allowed[None, None]
    max_iter = nz + ny + nx
    previous = -1.0
    for i in range(max_iter):
        current = _dilate_6conn_once(current) * allowed_5d
        if (i + 1) % _FLOOD_CHECK_EVERY == 0 or i == max_iter - 1:
            count = float(current.sum().item())
            if count == previous:
                break
            previous = count
    return current[0, 0] > 0.5


def _flood_fill_6conn_labelled(seed: Tensor, allowed: Tensor) -> Tensor:
    """Single-pass CPU flood fill: label *allowed*, keep the labels *seed* reaches.

    A seed voxel outside *allowed* still seeds its in-region neighbours, exactly
    as one dilation step of the iterative form would, so the entry set is taken
    after a single dilation rather than from the raw seed.
    """
    from scipy import ndimage

    entry = (_dilate_6conn_once(seed[None, None])[0, 0] * allowed) > 0.5
    labels, _ = ndimage.label(
        allowed.numpy() > 0.5, structure=ndimage.generate_binary_structure(3, 1)
    )
    reached = numpy.unique(labels[entry.numpy()])
    reached = reached[reached != 0]
    return torch.from_numpy(numpy.isin(labels, reached)).to(seed.device)


def _dilate_6conn(mask: Tensor, iterations: int = 2) -> Tensor:
    """Dilate with 6-connectivity (face neighbors only).

    Uses a 3D convolution with a cross-shaped kernel instead of max_pool3d
    (which gives 26-connectivity).
    """
    if iterations <= 0:
        return mask
    x = mask.float()[None, None]
    for _ in range(iterations):
        x = _dilate_6conn_once(x)
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


def largest_cluster_6conn(mask: Tensor) -> Tensor:
    """Keep the largest 6-connected component of a binary mask (``THD_mask_clust``).

    The *largest* cluster wins, not the one holding the brightest voxel. Seeding on
    the brightest voxel is the same thing on most data and catastrophically not on
    some: on a 0.8 mm CBV run the brightest voxel in the thresholded volume sat in a
    236-voxel speck, so the automask returned 276 voxels instead of the head's 1.6 M
    — and everything gated by it (locomoco's flow, its coupling report) came back
    empty without complaining.

    Labelling is exact rather than iterative: a flood fill on the GPU costs one pass
    per unit of geodesic diameter (hundreds, for a head), so for a single 3-D mask
    the transfer plus an exact CPU labelling is both cheaper and more correct.
    """
    if not bool(mask.any()):
        return mask
    from scipy import ndimage

    labels, n = ndimage.label(
        mask.detach().cpu().numpy(), structure=ndimage.generate_binary_structure(3, 1)
    )
    if n <= 1:
        return mask
    counts = numpy.bincount(labels.ravel())
    counts[0] = 0
    return torch.from_numpy(labels == int(counts.argmax())).to(device=mask.device)


# ---------------------------------------------------------------------------
# Hole filling (matches THD_mask_fillin_once / THD_mask_fillin_completely)
# ---------------------------------------------------------------------------


def _fill_holes_3d(mask: Tensor) -> Tensor:
    """Fill interior holes by flood-filling background from border.

    Matches AFNI's approach: invert → keep largest component (exterior) → invert.
    Any background region not connected to the border is filled.
    """
    bg = (~mask).float()
    seed = torch.zeros_like(bg)

    # Mark all border voxels that are background as seeds
    seed[0, :, :] = bg[0, :, :]
    seed[-1, :, :] = bg[-1, :, :]
    seed[:, 0, :] = bg[:, 0, :]
    seed[:, -1, :] = bg[:, -1, :]
    seed[:, :, 0] = bg[:, :, 0]
    seed[:, :, -1] = bg[:, :, -1]

    # 6-connectivity flood fill from the border: whatever the background cannot
    # reach from outside is an interior hole.
    return ~_flood_fill_6conn(seed, bg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _afni_cliplevel(im: numpy.ndarray, mfrac: float = 0.5) -> float:
    """``THD_cliplevel`` on a float image, bin for bin (10000-bin histogram)."""
    if mfrac <= 0.0 or mfrac >= 0.99:
        mfrac = 0.5
    nhist = 10000
    v = im.ravel().astype(numpy.float32)
    fac = float(v.max()) if v.size else 0.0
    if fac < 1.0e-100:
        return 0.0
    sfac = nhist / fac
    pos = v[v > 0.0]
    kk = (sfac * pos.astype(numpy.float64) + 0.499).astype(numpy.int64)
    kk = kk[kk <= nhist]
    npos = kk.size
    if npos <= 222:
        return 0.0
    hist = numpy.bincount(kk, minlength=nhist + 1)
    dsum = float((kk.astype(numpy.float64) ** 2).sum())

    # Start at the cut holding the upper 65% of positive voxels.
    qq = int(0.65 * npos)
    ib = int(numpy.rint(0.5 * numpy.sqrt(dsum / npos)))
    ii, acc = nhist - 1, 0
    while ii >= ib and acc < qq:
        acc += int(hist[ii])
        ii -= 1
    ncut = ii
    # cut = mfrac * median of everything at or above the cut, to a fixed point.
    cum_from_top = numpy.cumsum(hist[:nhist][::-1])[::-1]  # count in [i, nhist)
    for _ in range(66):
        nabove = int(cum_from_top[ncut]) if ncut < nhist else 0
        nhalf = nabove // 2
        ii, acc = ncut, 0
        while ii < nhist and acc < nhalf:
            acc += int(hist[ii])
            ii += 1
        nold = ncut
        ncut = int(mfrac * ii)
        if ncut == nold:
            break
    return float(numpy.float32(ncut / sfac))


def _afni_cliplevel_gradual(im: numpy.ndarray, mfrac: float = 0.5) -> numpy.ndarray:
    """``THD_cliplevel_gradual``: octant clip levels about the centre of mass,
    trilinearly blended. ``im`` is (nz, ny, nx); AFNI's i/j/k are x/y/z."""
    nz, ny, nx = im.shape
    it, jt, kt = nx - 1, ny - 1, nz - 1
    w = numpy.abs(im.astype(numpy.float64))
    tot = w.sum()
    zc_, yc_, xc_ = (
        (w.sum(axis=(1, 2)) * numpy.arange(nz)).sum() / tot,
        (w.sum(axis=(0, 2)) * numpy.arange(ny)).sum() / tot,
        (w.sum(axis=(0, 1)) * numpy.arange(nx)).sum() / tot,
    )
    ic, jc, kc = (int(numpy.rint(numpy.float32(c))) for c in (xc_, yc_, zc_))
    floor_val = 0.333 * _afni_cliplevel(im, mfrac)
    di = max(int(numpy.rint(0.01 * nx)), 1)
    dj = max(int(numpy.rint(0.01 * ny)), 1)
    dk = max(int(numpy.rint(0.01 * nz)), 1)
    icm, icp = max(ic - di, 0), min(ic + di, it)
    jcm, jcp = max(jc - dj, 0), min(jc + dj, jt)
    kcm, kcp = max(kc - dk, 0), min(kc + dk, kt)
    xr = ((0, icp), (icm, it))
    yr = ((0, jcp), (jcm, jt))
    zr = ((0, kcp), (kcm, kt))
    clip = numpy.zeros((2, 2, 2))  # [z][y][x] octant
    for a in range(2):
        for b in range(2):
            for c in range(2):
                (za, zb), (ya, yb), (xa, xb) = zr[a], yr[b], xr[c]
                val = _afni_cliplevel(im[za : zb + 1, ya : yb + 1, xa : xb + 1], mfrac)
                clip[a, b, c] = max(val, floor_val)

    def frac(n_idx: int, c: int, t: int) -> numpy.ndarray:
        p0, p1 = 0.5 * c, 0.5 * (c + t)
        inv = 1.0 / (p1 - p0) if p1 > p0 else 0.0
        return numpy.clip((numpy.arange(n_idx) - p0) * inv, 0.0, 1.0)

    x1 = frac(nx, ic, it)[None, None, :]
    y1 = frac(ny, jc, jt)[None, :, None]
    z1 = frac(nz, kc, kt)[:, None, None]
    x0, y0, z0 = 1.0 - x1, 1.0 - y1, 1.0 - z1
    out = numpy.zeros(im.shape)
    for a, za in ((0, z0), (1, z1)):
        for b, yb in ((0, y0), (1, y1)):
            for c, xc in ((0, x0), (1, x1)):
                out = out + clip[a, b, c] * za * yb * xc
    return out.astype(numpy.float32)


def _afni_fillin_once(mask: numpy.ndarray, nside: int) -> tuple[numpy.ndarray, int]:
    """``THD_mask_fillin_once``: fill an unset voxel with a set voxel within
    1..nside on BOTH sides along some axis. Voxels within nside of any edge are
    never considered, and fills are applied after the sweep (simultaneously)."""
    nz, ny, nx = mask.shape
    ns = [min((n - 1) // 2, nside) for n in (nz, ny, nx)]
    if not any(ns):
        return mask, 0
    fill = numpy.zeros_like(mask)
    core = tuple(slice(s, n - s) for s, n in zip(ns, (nz, ny, nx), strict=True))
    for axis in range(3):
        s, n = ns[axis], mask.shape[axis]
        if s == 0:
            continue
        plus = numpy.zeros_like(mask)
        minus = numpy.zeros_like(mask)
        for d in range(1, s + 1):
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            src[axis], dst[axis] = slice(d, n), slice(0, n - d)
            plus[tuple(dst)] |= mask[tuple(src)]
            minus[tuple(src)] |= mask[tuple(dst)]
        fill[core] |= (plus & minus)[core]
    fill &= ~mask
    return mask | fill, int(fill.sum())


def _afni_dilate_once(mask: numpy.ndarray, nmm: int) -> numpy.ndarray:
    """3dAutomask's ``-dilate`` step: ``THD_mask_dilate(..., 3, NN2)`` adds every
    unset voxel with at least 3 of its 18 neighbours set, then
    ``THD_mask_fillin_completely`` closes what that opened."""
    t = torch.from_numpy(mask)
    m = (t | (~t & (_count_neighbors_18_replicate(t) >= 3))).numpy()
    while True:
        m, n = _afni_fillin_once(m, nmm)
        if n == 0:
            return m


def automask(
    vol: Tensor,
    clip_frac: float = 0.5,
    dilate_extra: int = 0,
    peelcount: int = 1,
    peelthr: int = 17,
    gradual: bool = True,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tensor:
    """3dAutomask: AFNI ``mri_automask_image`` step for step, then ``-dilate``.

    Bit-identical to 3dAutomask on four ds000030 runs: the exact histogram clip
    level, the spatially *gradual* clip (on by default in AFNI), the peel that
    re-dilates (:func:`erode_many`), and the clustering/fill order including the
    final erode + recluster. The toolbox's earlier automask got all four subtly
    wrong and came out about a shell tight (27,786 vs 34,797 voxels on a
    64x64x34 EPI), which is why callers that used to dilate it by 2-4 now ask
    for 1-2.

    ``dilate_extra`` is 3dAutomask's ``-dilate``: AFNI's neighbour-count
    dilation with a fill-in after each step and a final interior-hole fill, not
    a plain 6-connected grow. Default 0, as in AFNI.

    Runs on the CPU (one small volume of branchy, sequential logic -- faster
    than the old GPU version even on a 1 mm head); the mask comes back on
    ``device`` or, by default, on ``vol``'s device.
    """
    out_device = device if device is not None else vol.device
    im = vol.detach().cpu().numpy().astype(numpy.float32)
    im = numpy.nan_to_num(im, nan=0.0, posinf=0.0, neginf=0.0)
    if not (im > 0).any():
        # 3dAutomask returns the whole volume here (cliplevel 0); no signal is no brain.
        return torch.zeros(im.shape, dtype=torch.bool, device=out_device)
    if gradual:
        m = im >= _afni_cliplevel_gradual(im, clip_frac)
    else:
        m = im >= _afni_cliplevel(im, clip_frac)
    if not m.any() or min(im.shape) < 2:
        return torch.from_numpy(m).to(out_device)

    def clust(a: numpy.ndarray) -> numpy.ndarray:
        return largest_cluster_6conn(torch.from_numpy(a)).numpy()

    def erode(a: numpy.ndarray, n: int) -> numpy.ndarray:
        return erode_many(torch.from_numpy(a), npeel=n, peelthr=peelthr).numpy()

    m = clust(m)
    m = erode(m, peelcount)
    m = clust(m)
    for _ in range(3):
        m, n = _afni_fillin_once(m, 1)
        if n == 0:
            break
    nz, ny, nx = im.shape
    nmm = max(1, int(numpy.rint(0.016 * nx)), int(numpy.rint(0.016 * ny)))
    jj = int(numpy.rint(0.016 * nz))
    nmm = max(nmm, jj)
    if nmm > 1 or jj > 0:
        for ii in range(2, nmm):
            m, _ = _afni_fillin_once(m, ii)
        while True:
            m, n = _afni_fillin_once(m, nmm)
            if n == 0:
                break
    m = erode(m, 1)
    m = clust(m)
    m = ~clust(~m)  # fill every hole that does not reach the volume edge
    if verbose:
        print(f"  automask: shape=({nz},{ny},{nx}) {int(m.sum()):,} voxels")
    if dilate_extra > 0:
        nmm_d = max(1, *(int(numpy.rint(0.032 * n)) for n in (nx, ny, nz)))
        for _ in range(dilate_extra):
            m = _afni_dilate_once(m, nmm_d)
        m = ~clust(~m)
        if verbose:
            print(f"  automask: after dilate({dilate_extra}): {int(m.sum()):,} voxels")
    return torch.from_numpy(m).to(out_device)


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


def looks_skull_stripped(cover: Tensor) -> bool:
    """True when a coverage mask looks like a masked object, not missing data.

    The coverage machinery (:func:`cross_fill_no_data`, the void guard) exists for a
    clipped field of view, and on that it is right. On a skull-stripped image the zero
    background is a real edge, and treating it as no-data stops a registration pulling
    tissue in across the brain boundary -- silently, because nothing fails.

    The two look different. A clipped FoV leaves most of the grid covered (a wedge or
    slab is missing); an acquisition slab covers a box, which fills its bounding box.
    A stripped brain covers a minority of the grid and roughly half of its own bounding
    box. Hence: under 40% of the grid, and under 75% of the covered bounding box.
    """
    covered = cover > 0
    frac = float(covered.float().mean())
    if not 0.01 < frac < 0.40:
        return False
    extent = 1
    for dim in range(covered.ndim):
        others = tuple(d for d in range(covered.ndim) if d != dim)
        idx = torch.nonzero(covered.any(dim=others)).flatten()
        extent *= int(idx[-1] - idx[0] + 1)
    return int(covered.sum()) / extent < 0.75


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
