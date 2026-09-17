"""Turning a run into rows: comparable voxel time courses, however you order them.

A carpet and a correlation matrix are the same first four steps -- find the
brain, take the time courses out of it, put them on a comparable scale, reduce
them to something a screen can hold -- and differ only in what they do with the
result. Written twice they would drift, and the drift would not look like a
bug: two windows onto the same run would simply disagree about it.

The choices here are the ones that decide whether either picture means
anything:

* **Automasking**, because a picture of air is mostly a picture of air.
* **Normalisation**, because raw BOLD counts differ tenfold between tissue
  types, and without it both pictures are maps of tissue rather than of time.
* **Binning within an ordering**, because rows that land in one bin are the
  ones the ordering just declared alike -- so their mean is the thing the band
  was going to show anyway, where striding would drop whole voxels and make the
  picture depend on where the stride happened to land.

No Qt, no session, no windows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

ProgressFn = Callable[[float, str], None]

#: More rows than any screen has pixels, so binning a carpet never throws away
#: something that would have been visible.
MAX_ROWS = 1200


def automask_from_series(data: np.ndarray, *, device: torch.device | None = None) -> np.ndarray:
    """A brain mask from a 4-D run's temporal mean, AFNI's algorithm.

    Through ``processing/mask.py:automask`` rather than a threshold invented
    here: "where is the brain" is a question this toolbox already answers the
    same way ``3dAutomask`` does.
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


def first_pc(rows: torch.Tensor) -> torch.Tensor:
    """The dominant temporal component of ``(V, T)`` normalised rows.

    Randomized SVD rather than an exact eigendecomposition of the T x T
    covariance: the covariance costs V*T^2 and this costs V*T*q, and what the
    component is used for is an *ordering*. A slightly rotated first component
    reorders rows that were already adjacent.

    Routed off Metal through the measured policy, because ``svd_lowrank``'s
    range finder is a ``linalg.qr`` -- the one op in the table that does not
    merely lose on MPS but asks for a 44 GiB buffer and dies. Nothing here
    tests ``device.type``; the table is the single place that question is
    answered.
    """
    from fastfuncstuff.utils import cpu_if_mps

    where = cpu_if_mps(rows.device, "qr")
    q = min(4, min(rows.shape) - 1) or 1
    _, _, v = torch.svd_lowrank(rows.to(where), q=q, niter=2)
    return v[:, 0].to(rows.device)


def correlate(rows: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Correlation of every row with one time course."""
    ref = reference - reference.mean()
    ref = ref / ref.norm().clamp(min=1e-9)
    centred = rows - rows.mean(-1, keepdim=True)
    centred = centred / centred.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return centred @ ref


def correlation_matrix(rows: torch.Tensor) -> torch.Tensor:
    """``(K, K)`` Pearson correlation between every pair of rows.

    One normalisation then one matmul. The diagonal is forced to exactly 1
    rather than left at whatever the arithmetic produced, because a picture
    whose diagonal is 0.9999 has a visible off-white stripe down it that reads
    as a result.
    """
    centred = rows - rows.mean(-1, keepdim=True)
    centred = centred / centred.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    out = centred @ centred.T
    out = out.clamp(-1.0, 1.0)
    out.fill_diagonal_(1.0)
    return out


def bin_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Average ordering-contiguous blocks down to at most ``max_rows``."""
    n = int(rows.shape[0])
    if n <= max_rows:
        return rows
    edges = torch.linspace(0, n, max_rows + 1).round().to(torch.long)
    out = torch.empty((max_rows, rows.shape[1]), dtype=rows.dtype, device=rows.device)
    for i in range(max_rows):
        lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
        out[i] = rows[lo:hi].mean(0)
    return out


def bin_counts(n: int, max_rows: int) -> list[int]:
    """How many source rows each bin of :func:`bin_rows` averages."""
    if n <= max_rows:
        return [1] * n
    edges = np.linspace(0, n, max_rows + 1).round().astype(int)
    return [max(int(hi) - int(lo), 1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def bin_representatives(values: np.ndarray, max_rows: int) -> np.ndarray:
    """One source row per bin -- the middle one -- rather than an average.

    For a voxel index, averaging is meaningless: the mean of flat indices 4 and
    900000 is a voxel in a different lobe. A displayed row does stand for many
    voxels, and the one this picks is the row at the centre of the bin, which
    under any of the orderings is the median of whatever the ordering sorted by.
    """
    n = int(values.shape[0])
    if n <= max_rows:
        return values
    edges = np.linspace(0, n, max_rows + 1).round().astype(int)
    middles = [
        min((int(lo) + int(hi)) // 2, n - 1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return values[middles]


def bin_values(values: np.ndarray, max_rows: int) -> np.ndarray:
    """The same reduction for a per-row scalar, so a sidebar stays aligned."""
    n = int(values.shape[0])
    if n <= max_rows:
        return values
    edges = np.linspace(0, n, max_rows + 1).round().astype(int)
    return np.array(
        [values[lo : max(hi, lo + 1)].mean() for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        dtype=np.float32,
    )


def prepare_rows(
    data: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    polort: int = -1,
    normalize: str = "z",
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> tuple[torch.Tensor, np.ndarray, str, float]:
    """``(V, T)`` rows, the flat mask that selected them, their units and limit.

    The one path from a 4-D array to comparable time courses. Both the carpet
    and the correlation matrix start here, so the two cannot end up disagreeing
    about which voxels are brain or what scale they are on.
    """

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    if data.ndim != 4:
        raise ValueError(f"this needs a 4-D series, got shape {data.shape}")
    nx, ny, nz, nt = data.shape
    if nt < 2:
        raise ValueError("this needs more than one volume")

    device = device or torch.device("cpu")
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
    return rows, flat_mask, units, limit


__all__ = [
    "MAX_ROWS",
    "ProgressFn",
    "automask_from_series",
    "bin_counts",
    "bin_representatives",
    "bin_rows",
    "bin_values",
    "correlate",
    "correlation_matrix",
    "first_pc",
    "normalize_rows",
    "prepare_rows",
]
