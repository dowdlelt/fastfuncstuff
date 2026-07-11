"""Joint space-time resampling: motion + slice-timing in a single interpolation.

Implements the *application* half of Roche's 4-D registration (Roche 2011, "A
Four-Dimensional Registration Algorithm With Application to Joint Correction of
Motion and Slice Timing in fMRI"; nipy ``SpaceTimeRealign``).  The motion is
assumed already estimated (a per-volume affine in the ``ffs_nwarp`` chain); this
module folds slice-timing correction into the *same* resample so the data is
interpolated only once.

Why this is not just ``3dTshift`` then motion-correct:
    A slice's acquisition time is a property of the *scanner* slice, not the
    tissue.  When the head moves, a fixed piece of brain lands in a different
    scanner slice each frame and so picks up a different timing offset.  Shifting
    each scanner slice's whole timeseries (3dTshift) only holds if the head is
    still.  Roche's fix: after applying a frame's pose you know which scanner
    slice ``k'`` each output voxel came from, and you sample the raw 4-D data at
    the temporal coordinate ``t' = t - Delta(k')``.

Operational form used here (separable, per output frame ``j``):
    Delta(v)  = interp(slice_times, sz(v))           # per-voxel acquisition offset
    T(v)      = j + (tzero - Delta(v)) / TR           # fractional input-frame coord
    S_f       = spatial_warp(source[f]) with pose j   # same spatial map for all f
    warped(v) = sum_f K(T(v) - f) * S_f(v) / sum_f K  # per-voxel temporal kernel

``sz`` (the source-voxel k index each output voxel samples) is exactly the
scanner slice ``k'`` because we sample the *raw* source there -- whatever the rest
of the chain does downstream.  All temporal neighbours reuse pose ``j`` (the slow-
motion assumption the paper makes explicit): the raw data is sampled at
scanner coordinates derived from a single pose per output frame.

Temporal kernels are convolution kernels (functions of distance), so the per-voxel
continuous shift ``T`` is handled without a per-slice uniform-shift assumption --
which is why Fourier (a whole-slice phase rotation) has no place on this path.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .interp import warp_image
from .slicetime import _m3_window, _sinc

# Temporal kernel half-widths (number of taps on each side of the sample point).
_KERNEL_HALFWIDTH = {"linear": 1, "cubic": 2, "wsinc5": 5, "wsinc9": 9}


def temporal_kernel_weights(dist: Tensor, method: str) -> Tensor:
    """Interpolation weight for each element at (signed) frame distance ``dist``.

    ``dist = T - f`` where ``T`` is the fractional sample coordinate and ``f`` an
    integer frame.  Kernels are the standard convolution forms so the weight is a
    pure function of distance (see module docstring for why that matters).

    - ``linear``: tent, support (-1, 1).
    - ``cubic``: Keys cubic convolution with a=-0.5 (Catmull-Rom), support (-2, 2).
      Matches the temporal kernel already used by ``slicetime.temporal_resample``.
    - ``wsinc5`` / ``wsinc9``: windowed sinc, half-width 5 / 9 (min-sidelobe M3
      window), reusing the same kernel pieces as ``ffs_slicetime``.
    """
    ad = dist.abs()
    if method == "linear":
        return (1.0 - ad).clamp_min(0.0)
    if method == "cubic":
        a = -0.5
        ad2 = ad * ad
        ad3 = ad2 * ad
        w_near = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
        w_far = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
        w = torch.where(ad < 1.0, w_near, w_far)
        return torch.where(ad < 2.0, w, torch.zeros_like(w))
    if method in ("wsinc5", "wsinc9"):
        half = _KERNEL_HALFWIDTH[method]
        w = _sinc(dist) * _m3_window(dist / half)
        return torch.where(ad < half, w, torch.zeros_like(w))
    raise ValueError(f"Unknown temporal interpolation method: {method!r}")


def interp_slice_times(sz: Tensor, slice_times: Tensor) -> Tensor:
    """Acquisition-time offset at fractional slice coordinate ``sz`` (seconds).

    ``slice_times[k]`` is the offset for integer scanner slice ``k``.  A voxel that
    lands between slices gets a linearly interpolated offset (edge-clamped), the
    same convention as nipy's ``interp_slice_times``.
    """
    n = slice_times.shape[0]
    if n == 1:
        return torch.full_like(sz, float(slice_times[0]))
    szc = sz.clamp(0.0, n - 1.0)
    i0 = szc.floor().long().clamp(0, n - 2)
    frac = szc - i0.to(szc.dtype)
    return slice_times[i0] * (1.0 - frac) + slice_times[i0 + 1] * frac


def apply_spacetime_sample(
    source: Tensor,
    sx: Tensor,
    sy: Tensor,
    sz: Tensor,
    frame_idx: int,
    tr: float,
    tzero: float,
    slice_times: Tensor,
    tinterp: str = "cubic",
    interp: str = "wsinc5",
    no_neg: bool = False,
) -> Tensor:
    """Sample the 4-D ``source`` at output frame ``frame_idx`` with joint slice-timing.

    Parameters
    ----------
    source : (nt, snz, sny, snx)
        Raw (un-slice-timed) 4-D series in internal ``(t, k, j, i)`` order.
    sx, sy, sz : (onz, ony, onx)
        Absolute source-voxel coordinates each output voxel samples, already
        carrying this frame's pose + the rest of the warp chain (as produced by
        ``nwarpforge._output_to_source_voxel_coords``).  ``sz`` is the scanner
        slice index used for the timing offset.
    frame_idx : int
        Output frame ``j``.  At native TR this is also the input-frame index the
        output represents (all slices realigned to ``tzero``).
    tr : float
        Repetition time (seconds).
    tzero : float
        Reference time within the TR all slices are aligned to (seconds).
    slice_times : (snz,) tensor
        Per-slice acquisition offsets (seconds), on ``source``'s device.
    tinterp : str
        Temporal kernel: ``linear``, ``cubic`` (default), ``wsinc5``, ``wsinc9``.
    interp : str
        Spatial kernel for sampling each source frame (matches ``ffs_nwarp``
        ``-interp``; default ``wsinc5``).
    no_neg : bool
        Clamp the spatial resample at 0 (suppress ringing on non-negative data).

    Returns
    -------
    (onz, ony, onx) warped, slice-timing-corrected volume.
    """
    nt = source.shape[0]
    # Compute on the coords' device; ``source`` may live on CPU and stream frame
    # by frame (the temporal window is only ~4-6 frames), so a large 4-D series
    # never needs to sit on the GPU in full.
    device = sx.device

    # Per-voxel acquisition offset and the fractional input-frame coordinate.
    delta = interp_slice_times(sz, slice_times)
    tcoord = frame_idx + (tzero - delta) / tr  # (onz, ony, onx)

    # Absolute source coords -> displacement form expected by warp_image
    # (output voxel (i,j,k) samples source at (i+xd, j+yd, k+zd)).
    onz, ony, onx = sz.shape
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=torch.float32, device=device),
        torch.arange(ony, dtype=torch.float32, device=device),
        torch.arange(onx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    xd, yd, zd = sx - ii, sy - jj, sz - kk

    half = _KERNEL_HALFWIDTH[tinterp]
    f_lo = int(math.floor(tcoord.min().item())) - (half - 1)
    f_hi = int(math.floor(tcoord.max().item())) + half

    acc = torch.zeros((onz, ony, onx), dtype=torch.float32, device=device)
    wsum = torch.zeros_like(acc)
    for f in range(f_lo, f_hi + 1):
        w = temporal_kernel_weights(tcoord - f, tinterp)
        if not bool(torch.any(w != 0.0)):
            continue
        # Edge-extend past the series ends (nipy uses reflect; clamp is adequate
        # and never invents structure -- the weights there are already tiny).
        fc = min(max(f, 0), nt - 1)
        frame = source[fc].to(device=device, dtype=acc.dtype)  # streams from CPU if needed
        s_f = warp_image(frame, xd, yd, zd, mode=interp)
        if no_neg:
            s_f = s_f.clamp_min(0.0)
        acc += w * s_f
        wsum += w

    return acc / wsum.clamp_min(1e-8)


def apply_spacetime_sample_following(
    source: Tensor,
    frame_coords: list[tuple[Tensor, Tensor, Tensor]],
    frame_idx: int,
    tr: float,
    tzero: float,
    slice_times: Tensor,
    device: torch.device,
    tinterp: str = "cubic",
    interp: str = "wsinc5",
    no_neg: bool = False,
) -> Tensor:
    """Tissue-following joint sample: drop the slow-motion assumption.

    :func:`apply_spacetime_sample` freezes a single pose (the output frame's) and
    samples every temporal neighbour at those *same* scanner coordinates. Under
    motion that fixed location holds *different tissue* each frame, so the temporal
    interpolation mixes tissue (the same failure a fixed-scanner-slice ``3dTshift``
    has). This variant instead follows the tissue: for output voxel ``u`` it samples
    each neighbour frame ``f`` at *that frame's own pose* ``T_f(u)`` -- so every tap
    is the same anatomy -- and reads the acquisition offset from the scanner slice
    ``u`` actually lands in *in frame f*. The temporal tap distance is therefore
    per frame:

        dist_f(u) = (j - f) + (tzero - Delta(sz_f(u))) / TR

    with ``sz_f`` the scanner slice of ``T_f(u)``. It reduces exactly to the
    frozen-pose version when the motion is static (all poses equal).

    Parameters
    ----------
    frame_coords : list of (sx, sy, sz), one per input frame
        Absolute source-voxel coordinates each output voxel samples *in that
        frame's pose* (as from ``_output_to_source_voxel_coords`` composed at
        ``time_idx=f``). Kept on CPU by the caller; moved to ``device`` per tap so
        a long series never sits on the GPU in full.
    frame_idx : int
        Output frame ``j`` (aligned to ``tzero``).
    Other parameters match :func:`apply_spacetime_sample`.
    """
    nt = source.shape[0]
    onz, ony, onx = frame_coords[frame_idx][2].shape
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=torch.float32, device=device),
        torch.arange(ony, dtype=torch.float32, device=device),
        torch.arange(onx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # The per-voxel timing shift (tzero - Delta)/TR lies in (-1, 1), so a kernel of
    # half-width H reaches at most H+1 frames each side of j.
    half = _KERNEL_HALFWIDTH[tinterp]
    f_lo, f_hi = frame_idx - (half + 1), frame_idx + (half + 1)

    acc = torch.zeros((onz, ony, onx), dtype=torch.float32, device=device)
    wsum = torch.zeros_like(acc)
    for f in range(f_lo, f_hi + 1):
        fc = min(max(f, 0), nt - 1)  # edge-extend pose + data past the series ends
        sx, sy, sz = (c.to(device=device, dtype=torch.float32) for c in frame_coords[fc])
        delta = interp_slice_times(sz, slice_times)
        w = temporal_kernel_weights((frame_idx - f) + (tzero - delta) / tr, tinterp)
        if not bool(torch.any(w != 0.0)):
            continue
        frame = source[fc].to(device=device, dtype=acc.dtype)
        s_f = warp_image(frame, sx - ii, sy - jj, sz - kk, mode=interp)
        if no_neg:
            s_f = s_f.clamp_min(0.0)
        acc += w * s_f
        wsum += w

    return acc / wsum.clamp_min(1e-8)
