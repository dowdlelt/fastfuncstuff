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

Two samplers live here:

* :func:`apply_spacetime_sample` -- the *frozen-pose* form (separable, per output
  frame ``j``):
      Delta(v)  = interp(slice_times, sz(v))          # per-voxel acquisition offset
      T(v)      = j + (tzero - Delta(v)) / TR          # fractional input-frame coord
      S_f       = spatial_warp(source[f]) with pose j  # same spatial map for all f
      warped(v) = sum_f K(T(v) - f) * S_f(v) / sum_f K # per-voxel temporal kernel
  Every temporal neighbour reuses pose ``j`` -- the slow-motion assumption the
  paper makes explicit. Cheap, but under motion that fixed location holds different
  tissue each frame, so the temporal interpolation mixes tissue (the same failure a
  fixed-scanner-slice ``3dTshift`` has). Selectable with ``-frozen``.

* :class:`TissueFollowingSampler` -- the **default**. It drops the slow-motion
  assumption: each neighbour frame ``f`` is sampled at *its own* pose ``T_f(u)``,
  so every temporal tap is the same anatomy, and the acquisition offset is read
  from the scanner slice ``u`` lands in *in frame f*. Recovers the signal where the
  frozen form (and a two-step tshift-then-motion) smear the wrong tissue -- e.g. a
  brain edge sweeping in and out of a voxel. Following the tissue costs the uniform
  time grid, so the taps are slid back onto one before the kernel is applied (see
  the class docstring); measured ~20-25% slower than the frozen path.

``sz`` (the source-voxel k index each output voxel samples) is exactly the scanner
slice ``k'`` because we sample the *raw* source there. Temporal kernels are
convolution kernels (functions of distance), so the per-voxel continuous shift is
handled without a per-slice uniform-shift assumption -- which is why Fourier (a
whole-slice phase rotation) has no place on this path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ..memory import compute_moco_resample_batch_size, get_available_memory
from .interp import warp_image_multi
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
    source: Tensor | Sequence[Tensor],
    sx: Tensor,
    sy: Tensor,
    sz: Tensor,
    frame_idx: int,
    tr: float,
    tzero: float,
    slice_times: Tensor,
    tinterp: str = "cubic",
    interp: str = "wsinc5",
    no_neg: bool | Sequence[bool] = False,
) -> Tensor | list[Tensor]:
    """Sample the 4-D ``source`` at output frame ``frame_idx`` with joint slice-timing.

    Parameters
    ----------
    source : (nt, snz, sny, snx) or a sequence of such tensors
        Raw (un-slice-timed) 4-D series in internal ``(t, k, j, i)`` order. Pass a
        sequence to sample several co-registered channels (e.g. the real/imag parts
        of complex phase data) through the *same* space-time map in one pass; the
        return type mirrors the input (a list when a sequence is given). ``no_neg``
        may then be a per-channel sequence.
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
    (onz, ony, onx) warped, slice-timing-corrected volume -- or a list of them, one
    per input channel, when ``source`` is a sequence.
    """
    multi = not isinstance(source, Tensor)
    sources: tuple[Tensor, ...] = tuple(source) if multi else (source,)  # type: ignore[arg-type]
    no_neg_ch = list(no_neg) if isinstance(no_neg, (list, tuple)) else [bool(no_neg)] * len(sources)
    nt = sources[0].shape[0]
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

    accs = [torch.zeros((onz, ony, onx), dtype=torch.float32, device=device) for _ in sources]
    wsum = torch.zeros((onz, ony, onx), dtype=torch.float32, device=device)
    active: list[tuple[int, Tensor]] = []
    for f in range(f_lo, f_hi + 1):
        w = temporal_kernel_weights(tcoord - f, tinterp)
        if bool(torch.any(w != 0.0)):
            active.append((f, w))

    # Every temporal tap uses the same frozen pose. Treat taps (and optional
    # phase channels) as co-registered interpolation channels, bounded by the
    # shared frame-batch memory planner. The per-voxel temporal weights are
    # applied after spatial sampling, exactly as in the scalar-tap loop.
    tap_batch = compute_moco_resample_batch_size(onz, ony, onx, len(active), device, interp=interp)
    tap_batch = max(1, tap_batch // len(sources))
    for start in range(0, len(active), tap_batch):
        block = active[start : start + tap_batch]
        frames = [
            src[min(max(f, 0), nt - 1)].to(device=device, dtype=torch.float32)
            for f, _ in block
            for src in sources
        ]
        warped = warp_image_multi(frames, xd, yd, zd, mode=interp)
        for tap_index, (_, w) in enumerate(block):
            offset = tap_index * len(sources)
            for c in range(len(sources)):
                s_f = warped[offset + c]
                if no_neg_ch[c]:
                    s_f = s_f.clamp_min(0.0)
                accs[c] += w * s_f
            wsum += w

    outs = [acc / wsum.clamp_min(1e-8) for acc in accs]
    return outs if multi else outs[0]


class TissueFollowingSampler:
    """Sliding-window tissue-following joint sampler (drops the slow-motion assumption).

    :func:`apply_spacetime_sample` freezes a single pose (the output frame's) and
    samples every temporal neighbour at those *same* scanner coordinates. Under
    motion that fixed location holds *different tissue* each frame, so the temporal
    interpolation mixes tissue (the same failure a fixed-scanner-slice ``3dTshift``
    has). This sampler follows the tissue: for output voxel ``u`` it samples each
    neighbour frame ``f`` at *that frame's own pose* ``T_f(u)`` -- so every tap is
    the same anatomy -- and reads the acquisition offset from the scanner slice
    ``u`` actually lands in *in frame f*. Tap ``f`` therefore carries the tissue's
    value at acquisition time

        tau_f(u) = f + Delta(sz_f(u)) / TR          (TR units)

    with ``sz_f`` the scanner slice of ``T_f(u)``.

    Non-uniform taps (the bookkeeping that makes this correct).
        Following the tissue buys the right anatomy but costs the *uniform time
        grid*: ``Delta`` is read at a different scanner slice each frame, so the
        ``tau_f`` are no longer equally spaced. Combining them with a plain
        normalised kernel average (``sum w*y / sum w``) is then only zeroth-order
        accurate -- it reproduces a constant exactly but not a linear trend, and
        the error grows with the tap spread. The damage is worst where the taps
        spread most: an **interleaved** slice order plus through-plane motion of an
        odd number of slices moves ``Delta`` by ~TR/2 frame to frame, and the
        uncorrected average is then several times *worse* than the frozen path it
        is meant to improve on.

        The fix keeps the kernel on its uniform grid and moves the data instead.
        Take the nominal grid to be the output frame's own offset,
        ``tau_f^nom = f + Delta(sz_j(u))/TR`` -- exactly the frozen path's grid --
        and first-order correct each tap onto it:

            s_f    = (Delta(sz_f) - Delta(sz_j)) / TR       # tap's displacement
            y_f'   = y_f - s_f * dy/dtau                    # slid onto the grid
            out    = sum_f K((j - f) + (tzero - Delta(sz_j))/TR) * y_f' / sum_f K

        ``dy/dtau`` is a central difference over the neighbouring taps' *true*
        times (one-sided at the window edges, where the kernel weight is tiny
        anyway). The weights are back on an equally-spaced grid, so the chosen
        kernel keeps its full order, and ``sum K`` is a partition of unity again.

        Two properties fall out. Without through-plane motion ``s_f == 0``
        identically and the result is bit-for-bit the frozen/``3dTshift`` answer --
        so the correction can never regress the common case. With it, the residual
        error is O(s^2 * y'') instead of O(s * y'): measured 1-2 orders of
        magnitude better across sequential, interleaved and drifting slice
        patterns, at unchanged noise gain and no extra spatial gathers.

    GPU discipline. Only a *window* of ``2*half + 2`` frames is ever touched per
    output frame, so we keep exactly that window resident on ``device`` (per-frame
    source coordinates, and the source frames themselves), advancing one frame per
    :meth:`sample` call. Each input frame's coordinates are composed once and each
    source frame is copied to the device once as the window slides -- O(nt) work at
    O(window) memory, versus precomputing all ``nt`` coordinate fields. When the
    driver's free VRAM cannot hold the window's source frames (very large grids /
    small GPUs), source frames are streamed per tap instead (coordinates, which are
    what the algorithm needs resident, stay cached); the coordinate cache alone is
    ``3 * window`` planes -- negligible beside the ``nt`` output volumes the caller
    already accumulates.
    """

    def __init__(
        self,
        source: Tensor | Sequence[Tensor],
        coords_fn,
        output_shape: tuple[int, int, int],
        tr: float,
        tzero: float,
        slice_times: Tensor,
        device: torch.device,
        tinterp: str = "cubic",
        interp: str = "wsinc5",
        no_neg: bool | Sequence[bool] = False,
        n_out: int | None = None,
        verb: int = 0,
    ) -> None:
        # ``source`` may be several co-registered channels (e.g. real/imag of complex
        # phase data): the pose and temporal weights are shared, so we warp each
        # channel with the same per-frame map and return one volume per channel.
        self.multi = not isinstance(source, Tensor)
        self.sources: tuple[Tensor, ...] = tuple(source) if self.multi else (source,)  # type: ignore[arg-type]
        self.no_neg_ch = (
            list(no_neg)
            if isinstance(no_neg, (list, tuple))
            else [bool(no_neg)] * len(self.sources)
        )
        self.coords_fn = coords_fn  # f -> (sx, sy, sz) on ``device``
        self.tr = tr
        self.tzero = tzero
        self.slice_times = slice_times
        self.device = device
        self.tinterp = tinterp
        self.interp = interp
        self.nt = self.sources[0].shape[0]
        self.half = _KERNEL_HALFWIDTH[tinterp]

        onz, ony, onx = output_shape
        self.kk, self.jj, self.ii = torch.meshgrid(
            torch.arange(onz, dtype=torch.float32, device=device),
            torch.arange(ony, dtype=torch.float32, device=device),
            torch.arange(onx, dtype=torch.float32, device=device),
            indexing="ij",
        )
        self._coords: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        self._srcs: dict[int, tuple[Tensor, ...]] = {}
        self._warped: dict[int, list[Tensor]] = {}

        # Decide whether the window's source frames fit alongside the coordinate
        # cache and the transient warp-composition workspace. Completed output
        # frames are stashed on the host by nwarpforge, so reserving ``n_out``
        # output planes here would both understate the available cache budget and
        # obscure the memory that composition actually needs.
        window = 2 * self.half + 2
        output_plane = onz * ony * onx * 4
        source_frame_bytes = sum(
            source[0].numel() * source.element_size() for source in self.sources
        )
        coord_cache_bytes = window * 3 * output_plane
        source_cache_bytes = window * source_frame_bytes
        warped_cache_bytes = window * len(self.sources) * output_plane
        # A high-order warp composition temporarily holds several output-grid
        # fields and interpolation buffers. Keep a deliberately conservative
        # reserve so caching source frames never crowds that peak.
        compose_workspace_bytes = 16 * output_plane
        avail = get_available_memory(device)  # device-specific safe budget
        base_need = source_cache_bytes + coord_cache_bytes + compose_workspace_bytes
        self.cache_source = device.type != "cuda" or avail > base_need
        # A tap warped through frame f's own pose is independent of which output
        # frame later consumes it. Adjacent temporal windows overlap almost
        # completely, so retaining the bounded window turns O(nt*window) spatial
        # resamples into O(nt). Disable automatically when the extra output planes
        # would crowd composition or source-frame storage.
        self.cache_warped = avail > base_need + warped_cache_bytes
        if verb >= 1:
            where = "device-cached" if self.cache_source else "streamed"
            taps = "cached" if self.cache_warped else "recomputed"
            print(
                f"nwarpforge: tissue-following window={window} frames, "
                f"source {where}, warped taps {taps}"
            )

    def _clamp(self, f: int) -> int:
        return min(max(f, 0), self.nt - 1)

    def _ensure(self, fcs: set[int]) -> None:
        """Compose/load the given (clamped) frames; evict everything else."""
        for fc in list(self._coords):
            if fc not in fcs:
                del self._coords[fc]
                self._srcs.pop(fc, None)
                self._warped.pop(fc, None)
        for fc in fcs:
            if fc not in self._coords:
                self._coords[fc] = self.coords_fn(fc)
                if self.cache_source:
                    self._srcs[fc] = tuple(
                        src[fc].to(self.device, dtype=torch.float32) for src in self.sources
                    )

    def _tap(self, f: int) -> list[Tensor]:
        """The source frames at ``f`` warped to the output grid through frame ``f``'s
        own pose -- i.e. the tissue at every output voxel, as frame ``f`` saw it."""
        fc = self._clamp(f)  # edge-extend pose + data past the series ends
        if fc in self._warped:
            return self._warped[fc]
        sx, sy, sz = self._coords[fc]
        frames = (
            self._srcs[fc]
            if self.cache_source
            else tuple(src[fc].to(self.device, dtype=torch.float32) for src in self.sources)
        )
        xd, yd, zd = sx - self.ii, sy - self.jj, sz - self.kk
        # All channels share this pose -> one gather builds them together.
        warped = warp_image_multi(frames, xd, yd, zd, mode=self.interp)
        result = [
            v.clamp_min(0.0) if self.no_neg_ch[c] else v
            for c, v in enumerate(warped)  # noqa: E501
        ]
        if self.cache_warped:
            self._warped[fc] = result
        return result

    def sample(self, frame_idx: int) -> Tensor | list[Tensor]:
        """The tissue-following, slice-timing-corrected output volume at ``frame_idx``.

        Returns one volume per channel (a list) when constructed with multiple
        source channels, else a single tensor.
        """
        # The per-voxel timing shift (tzero - Delta)/TR lies in (-1, 1), so a kernel
        # of half-width H reaches at most H+1 frames each side of j.
        f_lo, f_hi = frame_idx - (self.half + 1), frame_idx + (self.half + 1)
        frames = list(range(f_lo, f_hi + 1))
        self._ensure({self._clamp(f) for f in frames})

        # Per-tap acquisition offset, and the nominal (uniform) grid: the output
        # frame's own offset, which is exactly the frozen path's time grid.
        delta = {
            f: interp_slice_times(self._coords[self._clamp(f)][2], self.slice_times) for f in frames
        }
        delta_ref = delta[frame_idx]
        weights = {
            f: temporal_kernel_weights(
                (frame_idx - f) + (self.tzero - delta_ref) / self.tr, self.tinterp
            )
            for f in frames
        }
        active = [f for f in frames if bool(torch.any(weights[f] != 0.0))]

        # Warped taps are fetched on demand and dropped as the centre advances, so
        # at most three (f-1, f, f+1) are resident -- the derivative's whole cost.
        cache: dict[int, list[Tensor]] = {}

        def tap(f: int) -> list[Tensor]:
            if f not in cache:
                cache[f] = self._tap(f)
            return cache[f]

        accs = [torch.zeros_like(self.ii) for _ in self.sources]
        wsum = torch.zeros_like(self.ii)
        for f in active:
            y = tap(f)
            w = weights[f]
            # Slide this tap from its true time onto the nominal grid (see class
            # docstring). shift == 0 without through-plane motion, which is what
            # makes the no-motion result identical to the frozen path.
            shift = (delta[f] - delta_ref) / self.tr
            fm, fp = (f - 1 if f - 1 in frames else None), (f + 1 if f + 1 in frames else None)
            if fm is not None or fp is not None:
                lo_f, hi_f = (fm if fm is not None else f), (fp if fp is not None else f)
                ylo = tap(lo_f) if lo_f != f else y
                yhi = tap(hi_f) if hi_f != f else y
                # Spacing in TR units between the two taps' TRUE acquisition times.
                dt = (hi_f - lo_f) + (delta[hi_f] - delta[lo_f]) / self.tr
                # Near-coincident taps carry no derivative information; leaving the
                # tap uncorrected there is strictly better than dividing by ~0.
                safe = dt.abs() > 1e-3
                dt = torch.where(safe, dt, torch.ones_like(dt))
                for c in range(len(self.sources)):
                    slope = torch.where(safe, (yhi[c] - ylo[c]) / dt, torch.zeros_like(dt))
                    accs[c] = accs[c] + w * (y[c] - shift * slope)
            else:
                for c in range(len(self.sources)):
                    accs[c] = accs[c] + w * y[c]
            wsum = wsum + w
            for stale in [k for k in cache if k < f]:
                del cache[stale]

        outs = [acc / wsum.clamp_min(1e-8) for acc in accs]
        return outs if self.multi else outs[0]
