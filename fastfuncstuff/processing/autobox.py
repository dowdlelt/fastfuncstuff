"""Auto bounding box: find the box that holds the data and crop to it.

Backend for ``ffs_util_autobox`` (AFNI ``3dAutobox``). The dataset does not move
— only the matrix shrinks, with the header origin walked to the new corner — so
downstream tools see the same voxels in the same place, just fewer of the empty
ones. Cropping air off a 1mm anatomical is usually a 2-4x cut in every later
volume-sized allocation.

Algorithm (AFNI ``THD_autobbox`` -> ``MRI_autobbox``):

1. Collapse sub-bricks to a max-|value| volume.
2. Clip at ``THD_cliplevel`` (skipped by ``-noclust``, which is AFNI's coupling
   of "no clustering" to "no clipping" as well).
3. Threshold to a mask, then largest 6-connected cluster -> peel -> largest
   cluster again, which is what drops isolated bright specks outside the head.
4. First and last index along each axis that still has a set voxel.

Reference: AFNI ``src/thd_automask.c``, ``src/3dAutobox.c``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .grid import crop_affine, take_index_map
from .mask import _cliplevel, erode_many

# AFNI's own defaults: thd_automask.c static clfrac=0.5, peelcount=1.
DEFAULT_CLFRAC = 0.5
DEFAULT_PEELCOUNT = 1


def largest_cluster_6conn(mask: Tensor) -> Tensor:
    """Keep the largest 6-connected component of a binary mask.

    Faithful ``THD_mask_clust``: the *largest* cluster wins, not the one holding
    the brightest voxel (which is what :func:`~fastfuncstuff.processing.mask.
    _largest_component_6conn` finds — fine for automask, where the seed is
    inside the brain by construction, but wrong here where the whole point is to
    reject a bright speck that may outshine the head).

    Labelling is exact rather than iterative: a flood-fill on GPU costs one pass
    per unit of geodesic diameter (hundreds, for a head), so for a single 3D mask
    the transfer plus an exact CPU labelling is the cheaper and more correct path.
    """
    if not bool(mask.any()):
        return mask
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, n = ndimage.label(mask.detach().cpu().numpy(), structure=structure)
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    best = int(counts.argmax())
    return torch.from_numpy(labels == best).to(device=mask.device)


def autobox_mask(
    vol: Tensor,
    clust: bool = True,
    clip: bool = True,
    clfrac: float = DEFAULT_CLFRAC,
    peelcount: int = DEFAULT_PEELCOUNT,
) -> Tensor:
    """Binary mask whose extent defines the autobox.

    ``vol`` may be 3D or 4D; sub-bricks are collapsed by max |value| first, so a
    voxel active in any volume keeps the box open.
    """
    if vol.ndim == 4:
        v = vol.abs().amax(dim=0)
    elif vol.ndim == 3:
        v = vol.abs()
    else:
        raise ValueError(f"expected a 3D or 4D volume, got {vol.ndim}D")

    if clip:
        v = v.masked_fill(v < _cliplevel(v, clfrac), 0.0)

    mask = v != 0
    if clust and bool(mask.any()):
        mask = largest_cluster_6conn(mask)
        mask = erode_many(mask, npeel=peelcount)
        if bool(mask.any()):
            mask = largest_cluster_6conn(mask)
    return mask


def autobox_bounds(
    vol: Tensor,
    clust: bool = True,
    clip: bool = True,
    clfrac: float = DEFAULT_CLFRAC,
    peelcount: int = DEFAULT_PEELCOUNT,
) -> tuple[int, int, int, int, int, int]:
    """``(imin, imax, jmin, jmax, kmin, kmax)`` in NIfTI index order, inclusive.

    Same six numbers, same order, as ``3dAutobox -extent_ijk``.
    """
    mask = autobox_mask(vol, clust=clust, clip=clip, clfrac=clfrac, peelcount=peelcount)
    if not bool(mask.any()):
        raise ValueError("autobox found no non-zero voxels — the box is empty")

    # mask is (nz, ny, nx), so NIfTI i/j/k are tensor axes 2/1/0.
    zy = mask.any(dim=2)  # (nz, ny)
    zx = mask.any(dim=1)  # (nz, nx)
    bounds: list[int] = []
    for present in (zx.any(dim=0), zy.any(dim=0), zy.any(dim=1)):
        idx = torch.nonzero(present, as_tuple=False).reshape(-1)
        bounds.extend((int(idx[0].item()), int(idx[-1].item())))
    imin, imax, jmin, jmax, kmin, kmax = bounds
    return imin, imax, jmin, jmax, kmin, kmax


def pad_bounds(
    bounds: tuple[int, int, int, int, int, int],
    npad: int,
    shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Grow (or, for negative ``npad``, shrink) the box by ``npad`` voxels a side.

    Pass ``shape`` to clamp the result inside the input matrix — AFNI's
    ``-npad_safety_on``. Without it a positive ``npad`` may push the box past the
    original edges, and the crop zero-pads to fill it (AFNI does the same).
    """
    out = [
        bounds[0] - npad,
        bounds[1] + npad,
        bounds[2] - npad,
        bounds[3] + npad,
        bounds[4] - npad,
        bounds[5] + npad,
    ]
    if shape is not None:
        dims_xyz = (shape[2], shape[1], shape[0])
        for a in range(3):
            hi = dims_xyz[a] - 1
            out[2 * a] = min(max(out[2 * a], 0), hi)
            out[2 * a + 1] = min(max(out[2 * a + 1], 0), hi)
    imin, imax, jmin, jmax, kmin, kmax = out
    return imin, imax, jmin, jmax, kmin, kmax


def crop_to_bounds(
    data: Tensor,
    affine: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
) -> tuple[Tensor, np.ndarray]:
    """Crop (or zero-pad) ``data`` to ``bounds``, walking the affine to match.

    Equivalent to ``THD_zeropad(..., ZPAD_IJK)`` with the autobox offsets.
    Returns ``(cropped, new_affine)``.
    """
    imin, imax, jmin, jmax, kmin, kmax = bounds
    out_shape = (kmax - kmin + 1, jmax - jmin + 1, imax - imin + 1)
    if min(out_shape) <= 0:
        raise ValueError(f"empty crop box: {bounds}")

    identity = np.arange(3)
    cropped = take_index_map(
        data,
        perm=identity,
        signs=np.ones(3, dtype=int),
        offsets=np.array([imin, jmin, kmin], dtype=int),
        out_shape=out_shape,
    )
    return cropped, crop_affine(affine, (imin, jmin, kmin))
