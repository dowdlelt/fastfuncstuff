"""Clean series before correlating them: censor, then remove nuisance per run.

Every series a connectivity measure touches -- ROI means, voxel time courses, an
external seed -- goes through the same :class:`Cleaner`, so they share one residual
space. That matters: correlating a cleaned voxel with an uncleaned seed puts the
nuisance back into the map.

Three decisions:

* **Censored TRs leave before the fit**, not after. Fitting drift and motion with a
  spike in the rows lets the spike pull the fit, and dropping the row afterwards
  leaves that distortion in every other TR (3dTproject's ``-cenmode KILL``).
* **Per run, block-diagonal.** Concatenated runs have their own means and drifts; a
  global fit leaves a step at every boundary, and a step shared by all ROIs is a
  correlation shared by all ROIs (see [[Block-diagonal nuisance]]).
* **A passband is a set of regressors**, not an FFT. Sines and cosines at the
  stopband frequencies of each run are projected out with everything else, which
  stays correct across censored gaps (where an FFT has no evenly sampled series to
  act on) and removes the band *given* the motion regressors rather than leaving
  the filtered data correlated with unfiltered regressors again (Lindquist 2019).

Projection is linear and identical for every series, so averaging voxels into ROIs
and then cleaning gives the same ROI series as cleaning every voxel first; the CLI
uses that to clean a few hundred ROI rows instead of 10^5 voxel rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.utils import full_fp32_matmul


def stopband_bins(n_t: int, tr: float, fbot: float, ftop: float) -> np.ndarray:
    """``(n_t // 2 + 1,)`` bool over the rfft bins: True where the passband removes it.

    The run's own Fourier grid ``j / (n_t * tr)``; bin 0 (the mean) is never in the
    stopband -- it belongs to the drift model. ``ftop <= 0`` means no lowpass,
    ``fbot <= 0`` no highpass. The one definition both cleaning algorithms use.
    """
    j = np.arange(n_t // 2 + 1)
    f = j / (n_t * tr) if n_t > 0 and tr > 0 else np.zeros_like(j, dtype=float)
    stop = np.zeros(j.size, bool)
    if fbot > 0:
        stop |= f < fbot
    if ftop > 0:
        stop |= f > ftop
    stop[0] = False
    return stop


def stopband_regressors(n_t: int, tr: float, fbot: float, ftop: float) -> np.ndarray:
    """``(n_t, k)`` sines and cosines at every DFT frequency of a run outside ``[fbot, ftop]``.

    The frequencies are the run's own Fourier grid ``j / (n_t * tr)``, ``j = 1 .. n_t // 2``;
    0 Hz is left to the polynomial drift. ``ftop <= 0`` (or at/above Nyquist) means no
    lowpass; ``fbot <= 0`` means no highpass. At Nyquist (even ``n_t``) only the cosine
    exists -- the sine is identically zero.
    """
    if n_t < 2 or tr <= 0:
        return np.zeros((max(n_t, 0), 0))
    t = np.arange(n_t, dtype=np.float64)
    cols = []
    for j in np.flatnonzero(stopband_bins(n_t, tr, fbot, ftop)):
        arg = 2.0 * np.pi * j * t / n_t
        cols.append(np.cos(arg))
        if 2 * j != n_t:
            cols.append(np.sin(arg))
    return np.stack(cols, axis=1) if cols else np.zeros((n_t, 0))


def has_baseline(rows: Tensor, run_starts: Sequence[int] = (0,)) -> bool:
    """Whether ``(V, T)`` rows have a positive baseline that percent change can divide by.

    BOLD intensities do (a mean far above the fluctuation); residuals, demeaned or
    z-scored series and difference images do not, and percent change of those divides
    by ~0 -- every ROI mean is then whichever voxels happen to sit nearest zero.
    Judged per run on the median voxel: mean above 5 standard deviations, and almost
    no voxel with a non-positive mean.
    """
    if rows.numel() == 0:
        return False
    starts = list(run_starts) + [rows.shape[1]]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        seg = rows[:, a:b].float()
        mean = seg.mean(1)
        sd = seg.std(1)
        if float((mean <= 0).float().mean()) > 0.01:
            return False
        if not float(mean.median()) > 5.0 * float(sd.median()):
            return False
    return True


@dataclass
class Cleaner:
    """Per-run nuisance removal over the kept TRs, and how to apply it.

    Two exact algorithms for the same projection, chosen per run:

    * **regression** -- one orthonormal basis of ``[nuisance | stopband sines]`` on the
      kept rows, removed as ``y - Q Q^T y``. Works across censored gaps.
    * **fft** -- for a run with every TR kept: zero the stopband rfft bins, then remove
      the basis of the *band-filtered* nuisance. The DFT basis is orthogonal on a full
      run, so zeroing bins is the stopband projection, and
      ``span(stop, N) = span(stop) + span(P_band N)``: the same residual, O(T log T) per
      row instead of O(T k) with k in the hundreds. (This is also what InstaCorr did by
      hand; filtering the nuisance before projecting is what keeps it exact.)
    """

    keep: np.ndarray  # (T,) bool over the concatenated TRs
    run_cols: list[np.ndarray]  # per run: indices of its kept TRs into the full axis
    bases: list[Tensor]  # per run: (n_kept_r, k_r) orthonormal, float64 on CPU
    ranks: list[int]  # k_r: the dimensions each run's nuisance removes
    #: Per run, the rank without the stopband: ``ranks - band_free_ranks`` is what the
    #: passband alone costs (its sines overlap the drift polynomials, so it is not
    #: simply the number of stopband columns).
    band_free_ranks: list[int] | None = None
    #: Per run: the rfft bins kept (float 1/0) when the run uses the fft algorithm,
    #: in which case ``bases[r]`` spans the band-filtered nuisance; None = regression.
    band_masks: list[Tensor | None] | None = None

    @property
    def passband_cost(self) -> int:
        """Dimensions the passband removes beyond drift and regressors."""
        if self.band_free_ranks is None:
            return 0
        return int(sum(self.ranks) - sum(self.band_free_ranks))

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    @property
    def n_removed(self) -> int:
        """Dimensions removed in total; at least 1, the mean a Pearson r takes out anyway."""
        return max(1, int(sum(self.ranks)))

    @property
    def dof(self) -> int:
        """Residual degrees of freedom: kept TRs minus everything projected out."""
        return self.n_kept - self.n_removed

    @property
    def kept_run_starts(self) -> list[int]:
        """Run starts on the censored axis (runs with no kept TR are dropped)."""
        starts, pos = [], 0
        for cols in self.run_cols:
            if cols.size:
                starts.append(pos)
            pos += cols.size
        return starts

    @full_fp32_matmul()
    def apply(
        self,
        data: Tensor,
        device: torch.device | None = None,
        out_device: torch.device | None = None,
        chunk: int | None = None,
        verbose: bool = False,
    ) -> Tensor:
        """``(R, T)`` -> ``(R, n_kept)`` float32 residuals, rows streamed through ``device``.

        ``data`` may live on the host (a voxel matrix too big for the card); it is
        moved ``chunk`` rows at a time and the result is gathered on ``out_device``
        (default: ``data``'s device).

        TF32 is off for the products: the data still carry their mean (~100 in percent
        signal change) when ``y @ Q`` is formed, and TF32's ~1e-3 relative error on that
        leaves a residual ~0.5% outside the stopband -- enough to make 360 ROIs from
        114 residual dof look full rank (measured on CUDA, TASKFORCE rest).
        """
        if data.shape[1] != self.keep.size:
            raise ValueError(f"data has {data.shape[1]} TRs, the cleaner expects {self.keep.size}")
        device = device or data.device
        out_device = out_device or data.device
        n_rows = data.shape[0]
        if chunk is None:
            from fastfuncstuff.memory import estimate_chunk_size

            k_max = max(self.ranks, default=0)
            chunk = estimate_chunk_size(
                n_rows, self.keep.size, k_max, device, operation="paired_corr", n_trials=2
            )
        chunk = max(1, min(int(chunk), n_rows)) if n_rows else 1
        out = torch.empty(n_rows, self.n_kept, dtype=torch.float32, device=out_device)
        bases = [b.to(device=device, dtype=torch.float32) for b in self.bases]
        cols_t = [torch.as_tensor(c, device=device) for c in self.run_cols]
        masks = self.band_masks or [None] * len(bases)
        masks_t = [None if m is None else m.to(device) for m in masks]
        pieces = [(c, b, m) for c, b, m in zip(cols_t, bases, masks_t, strict=True) if c.numel()]
        bar = tqdm(
            range(0, n_rows, chunk),
            desc="Cleaning",
            unit="chunk",
            leave=True,
            disable=not verbose or n_rows <= chunk,
        )
        for r0 in bar:
            block = data[r0 : r0 + chunk].to(device=device, dtype=torch.float32)
            parts = []
            for cols, q, m in pieces:
                y = block.index_select(1, cols)
                if m is not None:
                    y = torch.fft.irfft(torch.fft.rfft(y, dim=1) * m, n=y.shape[1], dim=1)
                if q.shape[1]:
                    y = y - (y @ q) @ q.T
                parts.append(y)
            out[r0 : r0 + chunk] = torch.cat(parts, dim=1).to(out_device)
        return out


def _rank_safe_basis(cols: list[np.ndarray], rows: np.ndarray) -> Tensor:
    """Orthonormal float64 basis of ``cols`` on ``rows``, columns judged on shape not units."""
    from fastfuncstuff.stats.task_coupling import _orthonormal_basis

    if not cols:
        return torch.zeros(rows.size, 0, dtype=torch.float64)
    mat = np.concatenate(cols, axis=1)[rows]
    norms = np.linalg.norm(mat, axis=0)
    mat = mat[:, norms > 1e-12] / norms[norms > 1e-12]
    return _orthonormal_basis(torch.as_tensor(mat, dtype=torch.float64))


def _filtered_basis(cols: list[np.ndarray], keep_bins: np.ndarray, n_t: int) -> Tensor:
    """Orthonormal basis of the nuisance columns after the passband (rfft bins ``keep_bins``).

    A column the band removes entirely (a slow polynomial under a highpass) is dropped
    by comparing its filtered norm with its own unfiltered one -- normalising the
    remnant first would turn roundoff into a unit-length regressor of noise.
    """
    from fastfuncstuff.stats.task_coupling import _orthonormal_basis

    if not cols:
        return torch.zeros(n_t, 0, dtype=torch.float64)
    mat = np.concatenate(cols, axis=1)
    filt = np.fft.irfft(np.fft.rfft(mat, axis=0) * keep_bins[:, None], n=n_t, axis=0)
    before = np.linalg.norm(mat, axis=0)
    after = np.linalg.norm(filt, axis=0)
    alive = after > 1e-6 * np.maximum(before, 1e-300)
    if not alive.any():
        return torch.zeros(n_t, 0, dtype=torch.float64)
    filt = filt[:, alive] / after[alive]
    return _orthonormal_basis(torch.as_tensor(filt, dtype=torch.float64))


def build_cleaner(
    nuisance_per_run: Sequence[Tensor | np.ndarray | None],
    run_starts: Sequence[int],
    n_timepoints: int,
    *,
    keep: np.ndarray | None = None,
    tr: float | None = None,
    passband: tuple[float, float] | None = None,
    method: str = "auto",
) -> Cleaner:
    """Orthonormal per-run bases of ``[nuisance | stopband]`` over each run's kept TRs.

    ``nuisance_per_run`` holds one ``(run_length, k)`` matrix per run (polynomials and
    regressors, as :func:`cli_utils.assemble_per_run_nuisance` returns them) or None.
    Columns are scaled to unit norm before the rank decision, so a regressor in mm and
    one in radians are judged on shape, not units; collinear columns (a passband sine
    that a polynomial already spans, a motion column that censoring has flattened) are
    dropped rather than handed back as directions of noise.

    ``method``: ``auto`` uses the fft algorithm for every run that has a passband and
    no censored TR, regression otherwise; ``regression`` / ``fft`` force one (``fft``
    refuses a censored run). Both give the same residual and the same dof.
    """
    if method not in ("auto", "regression", "fft"):
        raise ValueError(f"method must be auto, regression or fft, got {method!r}")
    run_starts = list(run_starts)
    n_runs = len(run_starts)
    keep = np.ones(n_timepoints, bool) if keep is None else np.asarray(keep, bool).reshape(-1)
    if keep.size != n_timepoints:
        raise ValueError(f"censor has {keep.size} rows for {n_timepoints} TRs")
    if passband is not None and (tr is None or tr <= 0):
        raise ValueError("a passband needs the TR")
    if len(nuisance_per_run) != n_runs:
        raise ValueError(f"{len(nuisance_per_run)} nuisance blocks for {n_runs} runs")

    run_cols, bases, ranks, free, masks = [], [], [], [], []
    for r in range(n_runs):
        a = run_starts[r]
        b = run_starts[r + 1] if r + 1 < n_runs else n_timepoints
        n_r = b - a
        local = np.flatnonzero(keep[a:b])
        run_cols.append(a + local)
        blocks = []
        nuis = nuisance_per_run[r]
        if nuis is not None and nuis.shape[1]:
            mat = nuis.detach().cpu().numpy() if isinstance(nuis, Tensor) else np.asarray(nuis)
            if mat.shape[0] != n_r:
                raise ValueError(f"run {r + 1}: nuisance has {mat.shape[0]} rows for {n_r} TRs")
            blocks.append(mat.astype(np.float64))
        if passband is not None:
            assert tr is not None
            blocks.append(stopband_regressors(n_r, tr, passband[0], passband[1]))
        if not blocks or local.size == 0:
            bases.append(torch.zeros(local.size, 0, dtype=torch.float64))
            ranks.append(0)
            free.append(0)
            masks.append(None)
            continue

        uncensored = local.size == n_r
        if method == "fft" and passband is not None and not uncensored:
            raise ValueError(f"run {r + 1} has censored TRs; the fft algorithm needs whole runs")
        nuis_cols = blocks[:-1] if passband is not None else blocks
        free_basis = _rank_safe_basis(nuis_cols, local)
        free.append(int(free_basis.shape[1]))
        if passband is not None and uncensored and method != "regression":
            assert tr is not None
            stop = stopband_bins(n_r, tr, passband[0], passband[1])
            n_stop = 2 * int(stop.sum()) - int(n_r % 2 == 0 and bool(stop[-1]))
            q = _filtered_basis(nuis_cols, ~stop, n_r)
            masks.append(torch.as_tensor(~stop, dtype=torch.float32))
            rank = n_stop + int(q.shape[1])
        else:
            q = _rank_safe_basis(blocks, local)
            masks.append(None)
            rank = int(q.shape[1])
        if rank >= local.size:
            raise ValueError(
                f"run {r + 1}: {rank} nuisance dimensions for {local.size} kept TRs "
                "leave nothing to correlate; lower -polort, narrow the stopband, or censor less"
            )
        bases.append(q)
        ranks.append(rank)
    return Cleaner(
        keep=keep, run_cols=run_cols, bases=bases, ranks=ranks, band_free_ranks=free,
        band_masks=masks,
    )  # fmt: skip
