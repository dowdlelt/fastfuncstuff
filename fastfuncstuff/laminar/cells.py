"""Column x depth response curves fitted from the timeseries, in one step.

The two-step route (``ffs_deconvolve`` -> ``ffs_laminascope -iresp_prefix``) fits
every voxel and averages the fits into cells. That is exact only while the
estimator is linear, which is why it needs one fixed lambda
([[Smooth FIR]], ``tests/test_laminascope_singlestep.py``).

Here the mixing happens first: voxel timeseries are averaged into column x depth
cells and each cell is fitted once. Averaging data and estimating once needs no
linearity argument, gives each cell its own residual variance and effective dof,
and lets the depth sampling see the real per-condition design instead of the
zero-padded iresp files. One lambda for every cell is still the rule -- the noise
level is a depth gradient, so a per-cell lambda would write that gradient into
the amplitudes, which are the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.laminar import columns as col


@dataclass
class CellCurves:
    """What :func:`fit_cell_curves` hands the laminar model, shaped as the iresp
    route's ``col_curves`` / ``coverage`` / ``psfs`` so the downstream is identical."""

    curves: dict[int, dict[str, np.ndarray]]  # K -> label -> (C, K, n_knots)
    coverage: dict[int, np.ndarray]  # K -> (C, K)
    psf: dict[int, np.ndarray | None]  # K -> (C, K, K)
    times: np.ndarray  # (n_knots,) seconds after onset
    labels: list[str]
    log10_lambda: float
    edf: dict[int, np.ndarray]  # K -> (C * K,) effective knots per cell
    r2: dict[int, np.ndarray]  # K -> (C * K,) variance explained
    notes: list[str]


def ribbon_mask(reads, shape3: tuple[int, ...]) -> np.ndarray:
    """A 3-D mask of exactly the voxels the depth reads touch, to mask at load so
    the timeseries of the rest of the brain is never resident."""
    flat = np.zeros(int(np.prod(shape3)), bool)
    flat[reads.voxels] = True
    return flat.reshape(shape3)


def rows_for_reads(reads, mask_flat: np.ndarray) -> np.ndarray:
    """Row indices into masked data for each of ``reads.voxels``, in read order.

    The loader returns one row per mask voxel in flat-volume order; the reads index
    the volume. Both are sorted, so this is a lookup, but it is checked rather than
    assumed: a silent misalignment here would shuffle the ribbon.
    """
    kept = np.flatnonzero(np.asarray(mask_flat, bool).ravel())
    rows = np.searchsorted(kept, reads.voxels)
    if rows.max(initial=-1) >= kept.size or not np.array_equal(kept[rows], reads.voxels):
        raise ValueError("the loaded mask does not contain every voxel the depth reads use")
    return rows


def _condition_slices(n_knots: int, n_conditions: int) -> list[slice]:
    return [slice(i * n_knots, (i + 1) * n_knots) for i in range(n_conditions)]


def fit_cell_curves(
    data: torch.Tensor,
    rows: np.ndarray,
    reads,
    columns,
    ks: list[int],
    design: torch.Tensor,
    n_task: int,
    penalty: np.ndarray,
    labels: list[str],
    times: np.ndarray,
    run_starts: list[int],
    *,
    lam: float | None = None,
    signal_r2: float = 0.05,
    cell_lambda: str = "global",
    device: torch.device | None = None,
    verbose: bool = True,
) -> CellCurves:
    """Mix the voxel timeseries into cells and fit each cell's response curves.

    ``data`` is ``(n_mask_voxels, n_timepoints)``; ``rows`` maps ``reads.voxels``
    onto its rows. ``lam`` fixes the relative smoothing strength; without it one
    lambda is chosen for the whole dataset by leave-one-run-out
    (:func:`~fastfuncstuff.glm.smooth_basis.choose_global_lambda`).

    ``cell_lambda`` is ``"global"``: one lambda everywhere, the only safe rule
    when amplitudes are compared across columns, which every map and group table
    this feeds does. ``"column"`` (one per column, pooled over its depth bins) is
    reserved for reading a single column in isolation and is not implemented yet;
    per-cell is never offered, because the noise level is a depth gradient.
    """
    from fastfuncstuff.glm.smooth_basis import choose_global_lambda, fit_smooth_basis

    device = device or torch.device("cpu")
    if cell_lambda != "global":
        raise ValueError(
            f"cell_lambda {cell_lambda!r} is not implemented yet; only 'global' is. "
            "A per-column lambda biases amplitudes across columns, which is what the "
            "vertex maps and the group tables compare."
        )
    notes: list[str] = []
    n_knots = len(times)
    if n_task != n_knots * len(labels):
        raise ValueError(
            f"{n_task} task columns do not divide into {len(labels)} conditions of "
            f"{n_knots} knots; per-condition windows are not supported yet"
        )

    # Voxel timeseries in read order, float64 for the mixing (a weighted mean over
    # the ribbon tiling, which the curves inherit).
    series = np.asarray(data[rows].numpy(), np.float64)

    cells: dict[int, col.ColumnCurves] = {}
    for k in ks:
        cells[k] = col.column_curves(reads, columns, series, k)
    del series

    # One lambda for the whole run, chosen on the cells of every K together: it is
    # one scalar, and the binnings are the same data seen at different resolutions.
    stacked = np.concatenate(
        [cells[k].values.reshape(-1, cells[k].values.shape[-1]) for k in ks], axis=0
    )
    finite = np.isfinite(stacked).all(1)
    if not finite.any():
        raise ValueError("no column x depth cell was read completely enough to fit")
    if lam is None:
        chosen = choose_global_lambda(
            torch.as_tensor(stacked[finite], dtype=torch.float32),
            design, n_task, penalty, run_starts,
            signal_r2=signal_r2, device=device, verbose=verbose,
        )  # fmt: skip
        lam = float(chosen.lam)
        notes.append(
            f"global lambda log10 {np.log10(max(lam, 1e-300)):.3f} from "
            f"{int(finite.sum()):,} cells ({chosen.n_signal:,} above R2 {signal_r2:g})"
        )
    del stacked, finite

    curves: dict[int, dict[str, np.ndarray]] = {}
    coverage: dict[int, np.ndarray] = {}
    psf: dict[int, np.ndarray | None] = {}
    edf: dict[int, np.ndarray] = {}
    r2: dict[int, np.ndarray] = {}
    sl = _condition_slices(n_knots, len(labels))
    for k in ks:
        cc = cells[k]
        n_c = cc.values.shape[0]
        flat = cc.values.reshape(n_c * k, -1)
        ok = np.isfinite(flat).all(1)
        betas = np.full((n_c * k, n_task), np.nan)
        this_edf = np.full(n_c * k, np.nan, np.float32)
        this_r2 = np.full(n_c * k, np.nan, np.float32)
        if ok.any():
            fit = fit_smooth_basis(
                torch.as_tensor(flat[ok], dtype=torch.float32), design, n_task, penalty,
                method="fixed", lam=lam, device=device, verbose=verbose,
            )  # fmt: skip
            betas[ok] = fit.betas.cpu().numpy().astype(np.float64)
            this_edf[ok] = fit.edf.cpu().numpy()
            this_r2[ok] = fit.r2.cpu().numpy()
        shaped = betas.reshape(n_c, k, n_task)
        curves[k] = {lab: shaped[:, :, s] for lab, s in zip(labels, sl, strict=True)}
        coverage[k], psf[k] = cc.coverage, cc.psf
        edf[k], r2[k] = this_edf, this_r2
        cells[k] = None  # type: ignore[assignment]  # the mixed timeseries can go

    return CellCurves(curves, coverage, psf, np.asarray(times, np.float64), list(labels),
                      float(np.log10(max(lam, 1e-300))), edf, r2, notes)  # fmt: skip
