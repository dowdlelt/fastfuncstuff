"""Carpet plots, also called grayplots: every voxel's time course at once.

One row per voxel, time across, intensity as the value. It is the fastest way
to see what a run actually did -- a subject swallowing, a scanner spike, a
drifting coil, the global waves that motion leaves behind. Power's grayplot
made the case that this picture belongs beside every preprocessing decision.

Two things decide whether it is readable, and both are choices rather than
defaults:

* **Normalisation**, because raw BOLD counts differ tenfold between tissue
  types and the picture would be a map of tissue rather than of time.
* **Row order**, because acquisition order is spatial and the structure worth
  seeing is temporal. Sorting by correlation with the dominant component puts
  the voxels that move together next to each other, and a band that was
  invisible scattered through 900k rows becomes a stripe.

Nuisance projection is deliberately *not* a control here. Point the carpet at a
layer DERIVE already denoised, and the comparison -- raw carpet beside cleaned
carpet, same ordering -- is two windows rather than a checkbox. A quick polort
is offered because a carpet is unreadable through a linear drift.

No Qt. The widget paints what this returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

ProgressFn = Callable[[float, str], None]

#: Row orders. ``voxel`` is the raw picture; the rest group voxels that share a
#: time course, which is what turns a smear into a band.
ORDERINGS = ("voxel", "pc1", "seed", "overlay", "roi")

ORDER_LABELS = {
    "voxel": "acquisition order",
    "pc1": "corr with PC1",
    "seed": "corr with the seed voxel",
    "overlay": "overlay value",
    "roi": "corr with the overlay's ROI mean",
}

#: More rows than any screen has pixels, so binning never throws away something
#: that would have been visible.
MAX_ROWS = 1200


@dataclass(frozen=True)
class Carpet:
    """A rendered carpet, ready to be turned into pixels."""

    image: np.ndarray  # (rows, T) float32
    #: One value per row, in the same order, for the band drawn beside the
    #: carpet -- the overlay's value where each row's voxels are. ``None`` when
    #: there is no overlay to show.
    sidebar: np.ndarray | None
    n_voxels: int
    order: str
    units: str
    #: Symmetric display limit; the carpet is drawn on [-limit, +limit].
    limit: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))

    @property
    def binned(self) -> bool:
        return self.n_voxels > self.image.shape[0]

    def status(self) -> str:
        rows, nt = self.shape
        what = f"{self.n_voxels:,} voxels"
        if self.binned:
            what += f" in {rows} rows"
        return f"{what} x {nt} TR   {ORDER_LABELS.get(self.order, self.order)}   {self.units}"


def automask_from_series(data: np.ndarray, *, device: torch.device | None = None) -> np.ndarray:
    """A brain mask from a 4-D run's temporal mean, AFNI's algorithm.

    Through ``processing/mask.py:automask`` rather than a threshold invented
    here: a carpet whose rows include air is mostly a picture of air, and
    "where is the brain" is a question this toolbox already answers the same
    way ``3dAutomask`` does.
    """
    from fastfuncstuff.processing.mask import automask

    mean = torch.as_tensor(np.asarray(data, dtype=np.float32).mean(axis=3))
    return automask(mean, device=device).cpu().numpy().astype(bool)


def normalize_rows(flat: torch.Tensor, how: str) -> tuple[torch.Tensor, str, float]:
    """Put every voxel on a comparable scale. Returns (rows, units, limit).

    Percent change is the conventional carpet unit and the one to quote; z is
    the one that stays readable when the series is not BOLD -- a derived layer
    with the mean projected out has no baseline to be a percentage of.
    """
    mean = flat.mean(-1, keepdim=True)
    if how == "psc":
        # Voxels with no baseline would divide by ~0 and swamp the colour scale.
        safe = mean.abs().clamp(min=1e-6)
        rows = (flat - mean) / safe * 100.0
        return rows, "% change", 2.0
    rows = flat - mean
    sd = rows.std(-1, keepdim=True).clamp(min=1e-9)
    return rows / sd, "z", 3.0


def _first_pc(rows: torch.Tensor) -> torch.Tensor:
    """The dominant temporal component of ``(V, T)`` normalised rows.

    Randomized SVD rather than an exact eigendecomposition of the T x T
    covariance: the covariance costs V*T^2 and this costs V*T*q, and what the
    component is used for is an *ordering*. A slightly rotated first component
    reorders rows that were already adjacent.
    """
    q = min(4, min(rows.shape) - 1) or 1
    _, _, v = torch.svd_lowrank(rows, q=q, niter=2)
    return v[:, 0]


def _correlate(rows: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Correlation of every row with one time course."""
    ref = reference - reference.mean()
    ref = ref / ref.norm().clamp(min=1e-9)
    centred = rows - rows.mean(-1, keepdim=True)
    centred = centred / centred.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return centred @ ref


def _bin_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Average ordering-contiguous blocks down to at most ``max_rows``.

    Averaging *within the ordering* is the honest reduction: rows that land in
    one bin are the ones the ordering just declared alike, so the mean of them
    is the thing the band was going to show anyway. Striding instead would drop
    whole voxels and make the picture depend on where the stride happened to
    land.
    """
    n = int(rows.shape[0])
    if n <= max_rows:
        return rows
    edges = torch.linspace(0, n, max_rows + 1).round().to(torch.long)
    out = torch.empty((max_rows, rows.shape[1]), dtype=rows.dtype, device=rows.device)
    for i in range(max_rows):
        lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
        out[i] = rows[lo:hi].mean(0)
    return out


def _bin_values(values: np.ndarray, max_rows: int) -> np.ndarray:
    n = int(values.shape[0])
    if n <= max_rows:
        return values
    edges = np.linspace(0, n, max_rows + 1).round().astype(int)
    return np.array(
        [values[lo : max(hi, lo + 1)].mean() for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        dtype=np.float32,
    )


def build_carpet(
    data: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    order: str = "voxel",
    seed_series: np.ndarray | None = None,
    order_volume: np.ndarray | None = None,
    sidebar_volume: np.ndarray | None = None,
    polort: int = -1,
    normalize: str = "z",
    max_rows: int = MAX_ROWS,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> Carpet:
    """Turn a 4-D run into a carpet.

    ``order_volume`` supplies the per-voxel number that ``order="overlay"``
    sorts by, and the ROI whose mean time course ``order="roi"`` correlates
    against -- which is how "order by correlation with what the stats picked
    out" is expressed without the carpet knowing anything about statistics.
    """
    if data.ndim != 4:
        raise ValueError(f"a carpet needs a 4-D series, got shape {data.shape}")
    if order not in ORDERINGS:
        raise ValueError(f"unknown ordering {order!r}; have {ORDERINGS}")

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    device = device or torch.device("cpu")
    nx, ny, nz, nt = data.shape
    if nt < 2:
        raise ValueError("a carpet needs more than one volume")

    step(0.05, "masking")
    if mask is None:
        mask = automask_from_series(data, device=device)
    mask = np.asarray(mask, dtype=bool).reshape(nx, ny, nz)
    if not mask.any():
        raise ValueError("the mask is empty; nothing to draw")

    flat_mask = mask.reshape(-1)
    flat = torch.as_tensor(np.asarray(data, dtype=np.float32).reshape(-1, nt)[flat_mask]).to(device)

    if polort >= 0:
        step(0.25, f"detrend {polort}")
        from fastfuncstuff.viewer.derive import legendre_columns, orthonormal_basis, project_out

        basis = torch.as_tensor(
            orthonormal_basis(legendre_columns(nt, polort)), dtype=torch.float32
        ).to(device)
        flat = project_out(flat, basis, keep_mean=True)

    step(0.45, "normalising")
    rows, units, limit = normalize_rows(flat, normalize)

    step(0.65, "ordering")
    index = _order_index(
        rows, order, seed_series=seed_series, order_volume=order_volume, mask=flat_mask
    )
    rows = rows[index]

    side = None
    if sidebar_volume is not None:
        values = np.asarray(sidebar_volume, dtype=np.float32).reshape(-1)[flat_mask]
        side = _bin_values(values[index.cpu().numpy()], max_rows)

    step(0.85, "binning")
    image = _bin_rows(rows, max_rows).cpu().numpy().astype(np.float32)
    step(1.0, "ready")
    return Carpet(
        image=image,
        sidebar=side,
        n_voxels=int(rows.shape[0]),
        order=order,
        units=units,
        limit=limit,
    )


def _order_index(
    rows: torch.Tensor,
    order: str,
    *,
    seed_series: np.ndarray | None,
    order_volume: np.ndarray | None,
    mask: np.ndarray,
) -> torch.Tensor:
    """Row permutation for one ordering. Descending, so the strongest is on top."""
    if order == "voxel":
        return torch.arange(rows.shape[0], device=rows.device)

    if order == "pc1":
        key = _correlate(rows, _first_pc(rows))
    elif order == "seed":
        if seed_series is None or np.size(seed_series) != rows.shape[1]:
            raise ValueError("ordering by seed needs a seed time course of the same length")
        key = _correlate(rows, torch.as_tensor(np.asarray(seed_series, np.float32)).to(rows.device))
    elif order in ("overlay", "roi"):
        if order_volume is None:
            raise ValueError(f"ordering by {order!r} needs an overlay volume")
        values = np.asarray(order_volume, dtype=np.float32).reshape(-1)[mask]
        if values.shape[0] != rows.shape[0]:
            raise ValueError("the overlay does not cover the same voxels as the series")
        if order == "overlay":
            key = torch.as_tensor(np.nan_to_num(values)).to(rows.device)
        else:
            key = _correlate(rows, _roi_mean(rows, values))
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(order)
    return torch.argsort(key, descending=True)


def _roi_mean(rows: torch.Tensor, values: np.ndarray) -> torch.Tensor:
    """Mean time course of the strongest voxels of an overlay.

    "The top decile" rather than "above the display threshold": the carpet must
    not change every time the colour-bar slider moves, and a proportion is the
    definition that means the same thing on a t map, an F map and a beta map.
    """
    finite = np.nan_to_num(np.abs(values))
    if not finite.any():
        raise ValueError("the overlay is empty where the series is")
    cutoff = float(np.quantile(finite, 0.9))
    picked = torch.as_tensor(finite >= max(cutoff, 1e-12)).to(rows.device)
    if not bool(picked.any()):
        raise ValueError("the overlay selects no voxels")
    return rows[picked].mean(0)


__all__ = [
    "MAX_ROWS",
    "ORDERINGS",
    "ORDER_LABELS",
    "Carpet",
    "automask_from_series",
    "build_carpet",
    "normalize_rows",
]
