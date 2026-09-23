"""InstaGLM: a GLM with the lid off, refitted while you change your mind about it.

This is not the toolbox's GLM. ``ffs_deconvolve`` and ``ffs_reml`` are, and
nothing here is meant to displace them -- they fit many runs, they model serial
correlation, they write buckets. This fits **one run and one events file** and
answers a different question: *what did that choice just do?* Step the
polynomial order up and watch the drift leave the residual. Add the motion file
and watch the spikes go. Move the HRF peak by a second and watch a beta map
brighten. The point is the derivative of the answer with respect to the model,
which is the one thing a batch GLM can never show you.

Two decisions shape everything below.

**Every column is a first-class result.** A task GLM treats nuisance as
something to be rid of, and ``glm/core.py:fit_glm`` accordingly orthogonalises
the task columns against the nuisance before fitting -- correct there, because
it makes the task betas the Frisch-Waugh ones while leaving nuisance to soak up
whatever is shared. But it means a nuisance beta out of that fit is a *marginal*
coefficient, not the joint-model one, and "how much percent signal change does
roll carry, and where" would be answered with a number that quietly includes
everything roll shares with the task. So the fit here is a plain joint OLS over
the whole matrix, no block orthogonalised against any other, and a motion column
is reported exactly the way a condition column is. That is the teaching claim:
one model, one set of coefficients, and the collinearity left visible instead of
assigned.

**The extra sums of squares come out of the inverse, not out of a second fit.**
Dropping a block *A* of columns raises the residual sum of squares by
``b_A' (G⁻¹_AA)⁻¹ b_A`` where ``G = X'X`` -- the numerator of the F test for
that block. With *A* a single column that collapses to ``b_j² / G⁻¹_jj``. So
every column's semi-partial R², and the task block's R² and F, are quadratic
forms in the betas against a matrix the size of the design. Nothing is refitted,
and a map of "what does this one regressor uniquely buy" costs a reduction over
k, not another pass over the data.

Nothing here imports Qt and nothing touches session state: arrays and paths in,
arrays out, so the whole thing runs on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: ``progress(fraction, message)`` -- called from a worker thread.
ProgressFn = Callable[[float, str], None]

#: Column groups, in the order they are assembled into the design. Task first,
#: because the column picker is a list someone reads top-down looking for a
#: condition, and an xmat's thirty drift columns sitting above it is the
#: ordering that made ``Design.display_order`` necessary in the first place.
TASK, ORT, PC, DRIFT = "task", "ort", "pc", "drift"
GROUPS = (TASK, ORT, PC, DRIFT)

#: Basis names for the SPM derivative family. The suffix goes on the condition
#: label, so ``Faces'`` is the time derivative of ``Faces`` and reads as one.
DERIVATIVE_SUFFIX = ("", "'", "''")

#: Fraction of the largest singular value below which a direction is rank
#: deficiency rather than a regressor. Matches ``derive.RANK_TOL``.
RANK_TOL = 1e-8

#: A voxel mean this close to zero has no baseline to be a percentage of.
MEAN_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One design column, and enough about it to report a beta in real units."""

    label: str
    group: str
    #: Peak-to-trough of the column itself. This is what turns a coefficient
    #: into a signal change: the regressor's own excursion is the amount of
    #: itself it actually asked the data for, so ``beta * swing`` is the swing
    #: in the data that this column accounts for -- comparable between a
    #: condition in unit-peak HRF units and a roll column in degrees, which
    #: ``beta`` alone is not.
    swing: float


@dataclass(frozen=True)
class Model:
    """A design matrix whose columns know what they are."""

    matrix: np.ndarray  # (T, k) float64
    columns: tuple[Column, ...]
    #: One line for the status bar: what is in the model right now.
    note: str
    tr: float = 0.0

    @property
    def n_time(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.columns)

    def indices(self, *groups: str) -> list[int]:
        return [i for i, c in enumerate(self.columns) if c.group in groups]

    def index_of(self, label: str) -> int | None:
        return next((i for i, c in enumerate(self.columns) if c.label == label), None)

    @property
    def swings(self) -> np.ndarray:
        return np.array([c.swing for c in self.columns], dtype=np.float64)


@dataclass(frozen=True)
class Events:
    """Stimulus timing for one run, in the shape the design builder wants."""

    #: ``onsets[condition][run]`` -- one run here, but the nesting is the
    #: toolbox's and keeping it means ``build_task_design`` takes this as is.
    onsets: list[list[np.ndarray]]
    durations: list[float]
    labels: list[str]
    source: str = ""

    @property
    def n_conditions(self) -> int:
        return len(self.labels)

    @property
    def n_events(self) -> int:
        return sum(int(run.size) for cond in self.onsets for run in cond)


# ---------------------------------------------------------------------------
# reading timing
# ---------------------------------------------------------------------------


def read_events(path: str | Path, *, n_time: int, tr: float) -> Events:
    """One events file as conditions, onsets and durations.

    BIDS ``*_events.tsv`` by preference, through the same ``parse_bids_events``
    every other ffs GLM uses, so a file that works for ``-events`` elsewhere
    works here. An AFNI timing file has no condition column, so it is one
    condition named after the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no events file at {p}")

    if _looks_like_bids(p):
        from fastfuncstuff.design.bids_events import parse_bids_events

        onsets, durations, labels = parse_bids_events(event_files=[p], n_runs=1)
    else:
        from fastfuncstuff.design.builder import parse_afni_timing_file

        runs = parse_afni_timing_file(p)
        if not runs:
            raise ValueError(f"{p.name} holds no onsets")
        # An AFNI timing file is one condition over many runs; one run is
        # wanted, so the first row is the run and the rest is not ours to guess.
        onsets, durations, labels = [[np.asarray(runs[0], dtype=float)]], [0.0], [p.stem]

    _check_onsets_fit(onsets, labels, n_time=n_time, tr=tr, source=p.name)
    return Events(onsets=onsets, durations=list(durations), labels=list(labels), source=p.name)


def _looks_like_bids(path: Path) -> bool:
    """Whether the first line names an onset column. Read, not inferred from
    the suffix: ``.1D`` and ``.tsv`` are both used for both around here."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return False
    return "onset" in header.lower().split("\t")[0:1] or "onset" in header.lower().split()[:1]


def _check_onsets_fit(
    onsets: list[list[np.ndarray]],
    labels: Sequence[str],
    *,
    n_time: int,
    tr: float,
    source: str,
) -> None:
    """Refuse a design that would be all zeros, while the numbers still mean
    something. The usual cause is a wrong TR, and the symptom downstream is an
    empty map that looks like a bad dataset rather than a bad argument."""
    if tr <= 0:
        raise ValueError(f"{source}: onsets are in seconds, so the run needs a TR")
    run_seconds = n_time * tr
    latest = max((float(o.max()) for cond in onsets for o in cond if len(o)), default=-1.0)
    if latest < 0:
        raise ValueError(f"{source}: no events in any of {', '.join(labels) or 'the file'}")
    if latest >= run_seconds:
        raise ValueError(
            f"{source}: every event starts at or after the end of the run -- last onset "
            f"{latest:g}s, run length {run_seconds:g}s ({n_time} frames x {tr:g}s TR). "
            f"The TR is almost certainly wrong."
        )


# ---------------------------------------------------------------------------
# the HRF
# ---------------------------------------------------------------------------


def hrf_bases(
    basis: str,
    *,
    microtime_dt: float,
    index: int = 0,
    delay: float = 6.0,
    dispersion: float = 1.0,
    ratio: float = 0.167,
    duration: float = 32.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """``(n_basis, n_microtime)`` impulse responses, and a suffix per basis.

    Two ways to move the shape, because they teach different things. The
    *library* is the 20 curves ffs actually fits with, so stepping its index is
    the same gesture ``-hrf_library`` makes in a batch fit. The *custom* double
    gamma is the tactile one: peak, width and undershoot as three numbers you
    can put a finger on, which is how you find out that the beta map cares a
    great deal about the first and hardly at all about the third.
    """
    from fastfuncstuff.design.hrf import (
        get_spm_canonical_hrf,
        get_spm_hrf_with_derivatives,
        load_canonical_hrf_library,
    )

    if basis == "library":
        library = load_canonical_hrf_library(microtime_dt=microtime_dt, device=device)
        k = int(np.clip(index, 0, library.shape[0] - 1))
        return library[k : k + 1], ("",)

    if basis == "custom":
        curve = get_spm_canonical_hrf(
            microtime_dt=microtime_dt,
            hrf_duration=duration,
            delay=float(delay),
            dispersion=float(dispersion),
            ratio=float(ratio),
            device=device,
        )
        curve = curve.reshape(1, -1)
        # Unit peak, matching every other basis here, so that switching between
        # them moves the shape and not the scale of the betas.
        peak = curve.abs().max()
        return (curve / peak if float(peak) > 0 else curve), ("",)

    n_basis = {"spmg1": 1, "spmg2": 2, "spmg3": 3}.get(basis)
    if n_basis is None:
        raise ValueError(f"unknown HRF basis {basis!r}")
    curves = get_spm_hrf_with_derivatives(
        microtime_dt=microtime_dt, hrf_duration=duration, n_basis=n_basis, device=device
    )
    return curves, DERIVATIVE_SUFFIX[:n_basis]


def library_size() -> int:
    """How many curves the canonical library holds, for a slider's upper bound."""
    from fastfuncstuff.design.hrf import load_canonical_hrf_library

    return int(load_canonical_hrf_library(device=torch.device("cpu")).shape[0])


# ---------------------------------------------------------------------------
# building the design
# ---------------------------------------------------------------------------


def task_columns(
    events: Events,
    *,
    n_time: int,
    tr: float,
    curves: torch.Tensor,
    suffixes: Sequence[str],
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convolved condition regressors, ``(T, n_conditions * n_basis)``.

    Through ``design/matrices.py:build_task_design``, which duration-convolves
    and peak-normalises each event before conditions are formed by addition --
    so touching and overlapping events keep their identity, and a joint basis
    set stays on its anchor's scale. Reimplementing that here would get the
    easy case right and the interesting one wrong.
    """
    from fastfuncstuff.design.matrices import build_task_design, commensurate_microtime_dt

    dt = commensurate_microtime_dt(tr)
    design = build_task_design(
        hrf_bases=curves,
        n_timepoints=n_time,
        run_starts=[0],
        tr=tr,
        microtime_dt=dt,
        event_onsets=events.onsets,
        durations=events.durations,
        device=device,
    )
    labels = [f"{name}{suffix}" for name in events.labels for suffix in (suffixes or ("",))]
    return np.asarray(design.detach().cpu().numpy(), dtype=np.float64), labels


def build_model(
    *,
    n_time: int,
    tr: float,
    task: np.ndarray | None = None,
    task_labels: Sequence[str] = (),
    polort: int = 2,
    ort: np.ndarray | None = None,
    ort_labels: Sequence[str] = (),
    pcs: np.ndarray | None = None,
) -> Model:
    """Assemble the design: task, then ortvecs, then noise PCs, then drift."""
    blocks: list[tuple[np.ndarray, list[str], str]] = []

    if task is not None and task.shape[1]:
        blocks.append((np.asarray(task, dtype=np.float64), list(task_labels), TASK))

    if ort is not None and ort.shape[1]:
        ort = np.asarray(ort, dtype=np.float64)
        names = list(ort_labels) or [f"ort#{i}" for i in range(ort.shape[1])]
        blocks.append((ort, names, ORT))

    if pcs is not None and pcs.shape[1]:
        pcs = np.asarray(pcs, dtype=np.float64)
        blocks.append((pcs, [f"PC#{i}" for i in range(pcs.shape[1])], PC))

    if polort >= 0:
        from fastfuncstuff.viewer.derive import legendre_columns

        poly = legendre_columns(n_time, polort)
        blocks.append((poly, [f"Pol#{i}" for i in range(poly.shape[1])], DRIFT))

    if not blocks:
        raise ValueError("an empty model: give events, a polort, an ortvec or PCs")

    for columns, _, _ in blocks:
        if columns.shape[0] != n_time:
            raise ValueError(
                f"a regressor block has {columns.shape[0]} rows but the run has {n_time} volumes"
            )

    matrix = np.concatenate([c for c, _, _ in blocks], axis=1)
    columns = tuple(
        Column(label=label, group=group, swing=float(np.ptp(matrix[:, j])))
        for j, (label, group) in enumerate(
            (lab, grp) for cols, labs, grp in blocks for lab in _pad(labs, cols.shape[1])
        )
    )
    return Model(matrix=matrix, columns=columns, note=_describe(blocks, polort), tr=tr)


def _pad(labels: list[str], n: int) -> list[str]:
    """Labels for n columns, however many the block actually named."""
    return [labels[i] if i < len(labels) else f"col{i}" for i in range(n)]


def _describe(blocks: list[tuple[np.ndarray, list[str], str]], polort: int) -> str:
    counts: dict[str, int] = {}
    for columns, _, group in blocks:
        counts[group] = counts.get(group, 0) + int(columns.shape[1])
    parts = []
    if TASK in counts:
        parts.append(f"{counts[TASK]} task")
    if ORT in counts:
        parts.append(f"{counts[ORT]} ort")
    if PC in counts:
        parts.append(f"{counts[PC]} PC")
    if DRIFT in counts:
        parts.append(f"polort {polort}")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# the data, arranged once
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """The run as voxels-by-time, gathered once and kept for every refit.

    The gather is the expensive half -- a mask applied to a 4-D array is a copy,
    and on a real run that is a gigabyte -- while a refit is two matmuls over
    the result. Splitting them is what makes stepping the polynomial order feel
    like a slider rather than like a job.
    """

    y: torch.Tensor  # (V, T) on the compute device
    mean: torch.Tensor  # (V,) each voxel's temporal mean
    mask: np.ndarray  # (nx, ny, nz) bool
    #: ``(nx, ny, nz)`` int32 of each voxel's row in ``y``, -1 outside the mask.
    lookup: np.ndarray
    affine: np.ndarray
    tr: float

    @property
    def shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.mask.shape
        return int(nx), int(ny), int(nz)

    @property
    def n_voxels(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.y.shape[1])

    def row(self, ijk: tuple[int, int, int]) -> int:
        """The row of a voxel, or -1 if it is outside the mask or the volume."""
        i, j, k = ijk
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return -1
        return int(self.lookup[i, j, k])

    def timecourse(self, ijk: tuple[int, int, int]) -> np.ndarray | None:
        row = self.row(ijk)
        if row < 0:
            return None
        return self.y[row].detach().cpu().numpy().astype(np.float32)

    @property
    def bytes(self) -> int:
        return int(self.y.numel() * self.y.element_size())


def _blurred_rows(
    data: np.ndarray,
    mask: np.ndarray,
    sigma_vox: tuple[float, float, float],
    device: torch.device,
    progress: ProgressFn | None = None,
) -> torch.Tensor:
    """``(V, T)`` of the run smoothed *within* the mask.

    Within, as ``3dBlurInMask`` does it: ``blur(x . m) / blur(m)``. A plain
    blur mixes the zeros outside the brain into every edge voxel, which pulls
    the rim of the map toward nothing and is indistinguishable from a real
    drop-off in activation there.

    Chunked over time rather than blurring the whole 4-D run at once, because
    a real run is a gigabyte and this is the interactive path -- the smoothed
    copy would be a second one. Peak is one chunk of volumes on top of the
    ``(V, T)`` output, which has to exist regardless.
    """
    from fastfuncstuff.memory import get_available_memory
    from fastfuncstuff.stats.smooth3d import gaussian3d_batched

    nx, ny, nz, nt = data.shape
    keep = torch.as_tensor(np.flatnonzero(mask.reshape(-1)), device=device, dtype=torch.long)
    out = torch.empty((int(keep.numel()), nt), dtype=torch.float32, device=device)

    m = torch.as_tensor(mask.astype(np.float32), device=device).unsqueeze(0)
    # Once, not per chunk: the denominator is the same for every time point.
    denom = gaussian3d_batched(m, sigma_vox).clamp_min(1e-6)

    # Four volumes per time point in flight -- the block, it masked, the blur's
    # working copy and the result -- so budget for that and keep at least one.
    per_t = 4 * nx * ny * nz * 4
    budget = get_available_memory(device, empty_cache=False)
    chunk = max(1, min(nt, int(budget // max(per_t, 1))))
    for t0 in range(0, nt, chunk):
        t1 = min(t0 + chunk, nt)
        if progress is not None:
            progress(0.05 + 0.25 * (t0 / nt), f"blurring {t0}-{t1} of {nt}")
        block = torch.as_tensor(
            np.ascontiguousarray(data[..., t0:t1], dtype=np.float32), device=device
        ).permute(3, 0, 1, 2)
        smoothed = gaussian3d_batched(block * m, sigma_vox) / denom
        out[:, t0:t1] = smoothed.reshape(t1 - t0, -1)[:, keep].T
        del block, smoothed
    return out


def prepare(
    data: np.ndarray,
    *,
    affine: np.ndarray,
    tr: float,
    mask: np.ndarray | None = None,
    blur_fwhm: float = 0.0,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> Prepared:
    """Mask a 4-D run and lay it out as ``(V, T)`` on the compute device.

    Masked, not whole: outside the brain there is no baseline, so a percent
    signal change is a division by noise and the colour scale ends up set by
    air. The mask is AFNI's, through ``series.py:automask_from_series``, which
    is the same brain this viewer's carpets find.

    ``blur_fwhm`` smooths in millimetres before gathering. The mask is found on
    the *unsmoothed* run, so turning the blur up does not quietly grow the
    brain -- which would change how many voxels are fitted as well as what
    they contain, and make two fits at two blurs not comparable.
    """
    if data.ndim != 4:
        raise ValueError(f"InstaGLM needs one 4-D run; got shape {tuple(data.shape)}")
    device = device or torch.device("cpu")
    if progress is not None:
        progress(0.05, "masking")
    if mask is None:
        from fastfuncstuff.viewer.series import automask_from_series

        mask = automask_from_series(data, device=device)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} does not match the run {data.shape[:3]}")
    if not mask.any():
        raise ValueError("the mask is empty; nothing to fit")

    lookup = np.full(mask.shape, -1, dtype=np.int32)
    lookup[mask] = np.arange(int(mask.sum()), dtype=np.int32)

    if blur_fwhm > 0:
        from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox

        zooms = tuple(
            float(np.linalg.norm(np.asarray(affine, float)[:3, i])) or 1.0 for i in range(3)
        )
        y = _blurred_rows(data, mask, fwhm_mm_to_sigma_vox(blur_fwhm, zooms), device, progress)
    else:
        if progress is not None:
            progress(0.35, "gathering")
        flat = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[3])
        y = torch.as_tensor(flat[mask.reshape(-1)])
        if device.type != "cpu":
            if progress is not None:
                progress(0.7, "to device")
            y = y.to(device)
    if progress is not None:
        progress(1.0, "ready")
    return Prepared(
        y=y,
        mean=y.mean(dim=1),
        mask=mask,
        lookup=lookup,
        affine=np.asarray(affine, dtype=float),
        tr=float(tr),
    )


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

#: What the overlay can show. ``unique R2`` is the semi-partial R² of the
#: selected column -- the variance only it explains, which is the honest answer
#: to "how much does this regressor buy" when regressors overlap.
MAPS = ("beta", "t", "R2", "unique R2", "task R2", "task F", "resid sd")

#: How a coefficient becomes a percentage. See :attr:`Column.swing`.
PSC_MODES = ("swing", "per unit")


@dataclass
class Fit:
    """One joint OLS over the whole design, and everything read off it."""

    model: Model
    prepared: Prepared
    betas: torch.Tensor  # (k, V)
    tstats: torch.Tensor  # (k, V)
    ss_unique: torch.Tensor  # (k, V) extra sum of squares for each column alone
    task_ss: torch.Tensor | None  # (V,) extra sum of squares for the task block
    rss: torch.Tensor  # (V,)
    tss: torch.Tensor  # (V,)
    sigma2: torch.Tensor  # (V,)
    dof: int
    rank: int

    @property
    def n_task(self) -> int:
        return len(self.model.indices(TASK))

    @property
    def r2(self) -> torch.Tensor:
        return 1.0 - self.rss / self.tss.clamp(min=1e-12)

    # -- maps ----------------------------------------------------------
    def values(self, kind: str, column: int = 0, psc: str = "swing") -> torch.Tensor:
        """The ``(V,)`` vector behind one map, before it is put back in a volume."""
        k = int(np.clip(column, 0, self.model.n_columns - 1))
        if kind == "beta":
            scale = self.model.columns[k].swing if psc == "swing" else 1.0
            floor = torch.as_tensor(MEAN_FLOOR, device=self.prepared.mean.device)
            return (
                self.betas[k]
                * float(scale)
                / torch.maximum(self.prepared.mean.abs(), floor)
                * 100.0
            )
        if kind == "t":
            return self.tstats[k]
        if kind == "R2":
            return self.r2
        if kind == "unique R2":
            return self.ss_unique[k] / self.tss.clamp(min=1e-12)
        if kind == "task R2":
            if self.task_ss is None:
                return torch.zeros_like(self.rss)
            return self.task_ss / self.tss.clamp(min=1e-12)
        if kind == "task F":
            if self.task_ss is None or self.n_task == 0:
                return torch.zeros_like(self.rss)
            return (self.task_ss / self.n_task) / self.sigma2.clamp(min=1e-12)
        if kind == "resid sd":
            return self.sigma2.clamp(min=0).sqrt()
        raise ValueError(f"unknown map {kind!r}; have {', '.join(MAPS)}")

    def volume(self, kind: str, column: int = 0, psc: str = "swing") -> np.ndarray:
        """One map as a ``(nx, ny, nz)`` volume, zero outside the mask."""
        flat = self.values(kind, column, psc).detach().cpu().numpy().astype(np.float32)
        out = np.zeros(self.prepared.shape, dtype=np.float32)
        out[self.prepared.mask] = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # -- one voxel -----------------------------------------------------
    def decompose(self, ijk: tuple[int, int, int], column: int = 0) -> dict[str, np.ndarray]:
        """The model pulled apart at one voxel, every line on the data's baseline.

        ``signal`` is the point of the picture: the measurement with everything
        that is not the task taken out of it, drawn against the fit that is
        meant to explain what is left. Put the drift back by dropping polort and
        the two lines separate in front of you, which is the whole lesson.

        Baselines are deliberately shared. A nuisance fit has a mean of its own,
        and subtracting it whole would drop ``signal`` to zero and leave it
        floating somewhere below a ``data`` it is supposed to sit on top of.
        """
        row = self.prepared.row(ijk)
        if row < 0:
            return {}
        x = np.asarray(self.model.matrix, dtype=np.float32)
        beta = self.betas[:, row].detach().cpu().numpy().astype(np.float32)
        data = self.prepared.y[row].detach().cpu().numpy().astype(np.float32)
        full = x @ beta

        nuisance = self.model.indices(ORT, PC, DRIFT)
        out: dict[str, np.ndarray] = {"data": data, "resid": data - full}
        if nuisance:
            taken = x[:, nuisance] @ beta[nuisance]
            out["signal"] = data - (taken - taken.mean())
            out["fit"] = full - (taken - taken.mean())
        else:
            out["signal"] = data
            out["fit"] = full

        k = int(np.clip(column, 0, self.model.n_columns - 1))
        contribution = x[:, k] * beta[k]
        out["column"] = contribution - contribution.mean() + float(data.mean())
        return out


def fit_model(
    prepared: Prepared,
    model: Model,
    *,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> Fit:
    """Solve the whole design against every masked voxel, in one pass.

    Chunked through ``memory.py:estimate_chunk_size`` rather than a number
    picked here, because the two temporaries this makes per chunk -- the
    predictions and the residuals -- are both voxels-by-time, and that is
    exactly the shape the GLM memory model already plans for.
    """
    from fastfuncstuff.memory import estimate_chunk_size
    from fastfuncstuff.utils import factor_device, pinv_f64

    device = device or prepared.y.device
    n_time, k = model.matrix.shape
    if n_time != prepared.n_time:
        raise ValueError(f"the design has {n_time} rows but the run has {prepared.n_time} volumes")
    home = prepared.y.device
    x64 = torch.as_tensor(np.asarray(model.matrix, dtype=np.float64), device=factor_device(home))
    x = x64.to(device=home, dtype=torch.float32)

    # Through an SVD of the design rather than the normal equations.
    #
    # X'X squares the condition number, and the interesting designs are exactly
    # the ill-conditioned ones: a motion file *contains* drift, so adding one to
    # a polort 2 model takes cond(X) from 7 to 133 and cond(X'X) to 1.8e4. Solved
    # in float32 -- which is what casting the float64 inverse back to the data's
    # dtype amounted to -- that left the residual no longer orthogonal to the
    # drift columns, and an uncancelled polynomial trend walked into the fit and
    # the signal trace. Measured on a real run, the residual's leak into the
    # drift subspace was 1.6e-3 against 1e-14 for an honest float64 solve.
    #
    # U is orthonormal, so the one big matmul stays perfectly conditioned even in
    # float32, and every ill-conditioned step is (k, k) and done in float64. The
    # cost is identical: the same two matmuls of the same two shapes.
    u64, sv64, vh64 = torch.linalg.svd(x64, full_matrices=False)
    keep = sv64 > sv64.max().clamp(min=1e-300) * RANK_TOL
    u64, sv64, vh64 = u64[:, keep], sv64[keep], vh64[keep]

    gram = x64.T @ x64
    ginv64 = pinv_f64(gram).to(device=x64.device)
    rank = int(torch.linalg.matrix_rank(gram, rtol=RANK_TOL))
    dof = max(int(n_time - rank), 1)
    ginv_diag = torch.diagonal(ginv64).clamp(min=1e-12).to(device=home, dtype=x.dtype)

    # V S^-1 and U, the two halves of the pseudo-inverse, kept apart so the
    # residual can be formed from U alone.
    vs = (vh64.T / sv64).to(device=home, dtype=x.dtype)
    u = u64.to(device=home, dtype=x.dtype)

    # Taking each voxel's mean out before the solve is worth another two orders
    # of magnitude (7e-8 against 6e-6), because a BOLD series sits at 11000 while
    # the signal in it is tens, and float32 has seven digits to cover both. It is
    # only legitimate when a constant is in the design's span -- polort >= 0 --
    # since otherwise the model cannot put the mean back. ``restore`` is
    # pinv(X) @ 1, the coefficients that rebuild a constant, so the betas come
    # back exactly as if the mean had never been removed.
    ones = torch.ones(n_time, dtype=torch.float64, device=x64.device)
    restore64 = vh64.T @ ((u64.T @ ones) / sv64)
    centre = bool(torch.linalg.vector_norm(x64 @ restore64 - ones) < 1e-8 * float(n_time) ** 0.5)
    restore = restore64.to(device=home, dtype=x.dtype).unsqueeze(1)

    n_voxels = prepared.n_voxels
    chunk = estimate_chunk_size(
        n_voxels=n_voxels,
        n_timepoints=n_time,
        n_regressors=k,
        device=device,
        operation="glm",
        max_chunk_size=n_voxels,
    )

    betas = torch.zeros(k, n_voxels, device=x.device, dtype=x.dtype)
    rss = torch.zeros(n_voxels, device=x.device, dtype=x.dtype)
    tss = torch.zeros(n_voxels, device=x.device, dtype=x.dtype)
    for start in range(0, n_voxels, chunk):
        stop = min(start + chunk, n_voxels)
        block = prepared.y[start:stop].T  # (T, c) -- time contiguous for the matmuls
        mean = block.mean(dim=0, keepdim=True)
        centred = block - mean
        work = centred if centre else block
        z = u.T @ work  # (r, c) -- the fit in the design's orthonormal frame
        residual = work - u @ z
        b = vs @ z
        if centre:
            b = b + restore * mean
        betas[:, start:stop] = b
        rss[start:stop] = (residual * residual).sum(dim=0)
        tss[start:stop] = (centred * centred).sum(dim=0)
        if progress is not None:
            progress(stop / n_voxels, f"fitting {k} regressors")

    sigma2 = rss / dof
    stderr = (sigma2.unsqueeze(0) * ginv_diag.unsqueeze(1)).clamp(min=1e-24).sqrt()
    # Dropping column j raises the residual sum of squares by b_j^2 / G^-1_jj.
    # That is the whole extra-sums-of-squares story for one column, and it is
    # also t_j^2 * sigma^2 -- the same number the t statistic is built from.
    ss_unique = betas * betas / ginv_diag.unsqueeze(1)

    task = model.indices(TASK)
    task_ss = None
    if task:
        # The block form of the same identity: b_A' (G^-1_AA)^-1 b_A. One
        # (k_task, k_task) inverse and a quadratic form per voxel, so the joint
        # task map costs a reduction over k rather than a second fit.
        sub = pinv_f64(ginv64[np.ix_(task, task)]).to(device=home, dtype=x.dtype)
        b_task = betas[task]
        task_ss = torch.einsum("iv,ij,jv->v", b_task, sub, b_task)

    return Fit(
        model=model,
        prepared=prepared,
        betas=betas,
        tstats=betas / stderr,
        ss_unique=ss_unique,
        task_ss=task_ss,
        rss=rss,
        tss=tss,
        sigma2=sigma2,
        dof=dof,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# noise pool principal components
# ---------------------------------------------------------------------------


def noise_pool(fit: Fit, *, brightness: float = 0.5, f_ceiling: float = 1.0) -> torch.Tensor:
    """Bright voxels the task does not explain -- GLMdenoise's pool, ``(V,)`` bool.

    Bright *and* unexplained, not either alone: dark voxels are air and carry no
    physiology, while bright voxels the task already explains are the signal the
    PCs must not be allowed to eat.

    The second half is the task **F**, not the task R². GLMdenoise's own rule is
    "keep voxels whose R² is negative", which works there because its R² is
    cross-validated and so can be. The extra sum of squares computed here is
    in-sample: it is a quadratic form in a positive semi-definite matrix, so it
    is never negative, and an R² ceiling of zero would select nothing at all.
    Worse, an in-sample task R² has an expected value of about ``k_task / T``
    under the null, so any fixed R² ceiling means something different for every
    design and run length. F is already divided by exactly that, which makes a
    ceiling of 1 say what was meant all along: *the task explains no more of
    this voxel than chance would.*
    """
    mean = fit.prepared.mean
    cut = torch.quantile(mean.float(), float(np.clip(brightness, 0.0, 0.99)))
    task_f = fit.values("task F")
    return (mean > cut) & (task_f <= float(f_ceiling))


def noise_pcs(
    prepared: Prepared,
    pool: torch.Tensor,
    nuisance: np.ndarray,
    n_components: int,
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, n)`` PC time courses from the pool, and their variance ratios.

    Through ``denoise/combinatorial.py:extract_pcs_single_run_with_variance``,
    which is the same extraction ``ffs_denoisatorial`` selects over: projection
    of the nuisance first, then unit-length normalisation per voxel, then PCA.
    A PC derived a slightly different way here would not be the PC that tool
    would have chosen, and the two would disagree about the same run.
    """
    from fastfuncstuff.denoise.combinatorial import extract_pcs_single_run_with_variance

    if n_components <= 0 or not bool(pool.any()):
        return np.zeros((prepared.n_time, 0), dtype=np.float64), np.zeros(0)
    device = device or prepared.y.device
    scores, ratios = extract_pcs_single_run_with_variance(
        run_data=prepared.y,
        noise_pool_mask=pool.to(prepared.y.device),
        nuisance=torch.as_tensor(nuisance, dtype=torch.float32, device=prepared.y.device),
        max_components=int(n_components),
        device=device,
    )
    return np.asarray(scores.detach().cpu().numpy(), dtype=np.float64), np.asarray(ratios)


__all__ = [
    "DRIFT",
    "GROUPS",
    "MAPS",
    "ORT",
    "PC",
    "PSC_MODES",
    "TASK",
    "Column",
    "Events",
    "Fit",
    "Model",
    "Prepared",
    "build_model",
    "fit_model",
    "hrf_bases",
    "library_size",
    "noise_pcs",
    "noise_pool",
    "prepare",
    "read_events",
    "task_columns",
]
