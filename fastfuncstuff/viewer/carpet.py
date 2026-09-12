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

Both of those, and the masking and binning around them, live in
:mod:`viewer.series` -- shared with the correlation matrix, so two windows onto
one run cannot disagree about which voxels are brain.

Nuisance projection is deliberately *not* a control here. Point the carpet at a
layer DERIVE already denoised, and the comparison -- raw carpet beside cleaned
carpet, same ordering -- is two windows rather than a checkbox. A quick polort
is offered because a carpet is unreadable through a linear drift.

No Qt. The widget paints what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.viewer.series import (
    MAX_ROWS,
    ProgressFn,
    automask_from_series,
    bin_representatives,
    bin_rows,
    bin_values,
    correlate,
    first_pc,
    normalize_rows,
    prepare_rows,
)

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
    #: ``(rows, 3)`` voxel indices, one per displayed row. A carpet has a
    #: spatial axis as much as a temporal one -- it is just scrambled by the
    #: ordering -- and without this the picture is the only place in the viewer
    #: you cannot click your way back out of. A binned row stands for many
    #: voxels; this is the one at the centre of the bin.
    voxels: np.ndarray | None = None

    def voxel_of(self, row: int) -> tuple[int, int, int] | None:
        """Where row ``row`` came from, or ``None`` if that was not recorded."""
        if self.voxels is None or not (0 <= row < self.voxels.shape[0]):
            return None
        i, j, k = (int(v) for v in self.voxels[row])
        return (i, j, k)

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
    if order not in ORDERINGS:
        raise ValueError(f"unknown ordering {order!r}; have {ORDERINGS}")

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    rows, flat_mask, units, limit = prepare_rows(
        data, mask=mask, polort=polort, normalize=normalize, device=device, progress=progress
    )

    step(0.65, "ordering")
    index = _order_index(
        rows, order, seed_series=seed_series, order_volume=order_volume, mask=flat_mask
    )
    rows = rows[index]

    side = None
    if sidebar_volume is not None:
        values = np.asarray(sidebar_volume, dtype=np.float32).reshape(-1)[flat_mask]
        side = bin_values(values[index.cpu().numpy()], max_rows)

    step(0.85, "binning")
    image = bin_rows(rows, max_rows).cpu().numpy().astype(np.float32)
    # Which voxel each drawn row is, after the same ordering and the same
    # binning the picture went through -- carried rather than recomputed, since
    # a second derivation of "row 400 is voxel X" is a second chance to be off
    # by one, and the symptom would be a crosshair that lands near the truth.
    flat_ids = np.flatnonzero(flat_mask)[index.cpu().numpy()]
    voxels = np.stack(np.unravel_index(bin_representatives(flat_ids, max_rows), data.shape[:3]), 1)
    step(1.0, "ready")
    return Carpet(
        image=image,
        sidebar=side,
        n_voxels=int(rows.shape[0]),
        order=order,
        units=units,
        limit=limit,
        voxels=voxels.astype(np.int32),
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
        key = correlate(rows, first_pc(rows))
    elif order == "seed":
        if seed_series is None or np.size(seed_series) != rows.shape[1]:
            raise ValueError("ordering by seed needs a seed time course of the same length")
        key = correlate(rows, torch.as_tensor(np.asarray(seed_series, np.float32)).to(rows.device))
    elif order in ("overlay", "roi"):
        if order_volume is None:
            raise ValueError(f"ordering by {order!r} needs an overlay volume")
        values = np.asarray(order_volume, dtype=np.float32).reshape(-1)[mask]
        if values.shape[0] != rows.shape[0]:
            raise ValueError("the overlay does not cover the same voxels as the series")
        if order == "overlay":
            key = torch.as_tensor(np.nan_to_num(values)).to(rows.device)
        else:
            key = correlate(rows, _roi_mean(rows, values))
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
