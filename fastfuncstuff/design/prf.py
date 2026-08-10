"""Population receptive-field (pRF) fitting primitives.

The compressive spatial summation (CSS) model and every convention here - the
super-grid seed list, the ``(row, column, sigma, gain, exponent)`` parameter
order, the one-based square-frame pixel coordinates, the parameter bounds, and
the staged fit - come from:

    Kay KN, Winawer J, Mezer A, Wandell BA (2013).
    Compressive spatial summation in human visual cortex.
    Journal of Neurophysiology 110(2), 481-494.
    https://doi.org/10.1152/jn.00105.2013

    analyzePRF (MATLAB), by Kendrick Kay - http://kendrickkay.net/analyzePRF/
    Copyright (c) 2014 Kendrick Kay. Licensed CC BY 3.0 Unported.

Work published using these primitives should cite Kay et al. 2013.

The grid search deliberately returns seeds instead of refining them, so callers
can validate it independently and choose an optimizer suited to their workload.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from fastfuncstuff.glm.core import orthogonalize_design


@dataclass(frozen=True)
class PRFGrid:
    """Candidate spatial/CSS parameters using MATLAB's one-based pixel coordinates."""

    parameters: torch.Tensor  # (n_candidates, 4): row, column, sigma, exponent

    @property
    def n_candidates(self) -> int:
        return self.parameters.shape[0]


@dataclass
class PRFGridFit:
    """Best super-grid candidate and linear gain for every fitted voxel."""

    candidate_index: torch.Tensor  # (n_voxels,), zero-based
    hrf_index: torch.Tensor  # (n_voxels,), zero-based
    parameters: torch.Tensor  # (n_voxels, 4): row, column, sigma, exponent
    gain: torch.Tensor  # (n_voxels,), constrained non-negative
    correlation: torch.Tensor  # (n_voxels,)
    r2: torch.Tensor  # (n_voxels,), correlation squared
    # Each HRF's own best grid candidate, kept only under track_per_hrf. The
    # winner above is the max over these; these are what seed a forced-HRF refine.
    correlation_per_hrf: torch.Tensor | None = None  # (n_voxels, n_hrfs)
    candidate_per_hrf: torch.Tensor | None = None  # (n_voxels, n_hrfs), int32
    grid_parameters: torch.Tensor | None = None  # (n_candidates, 4), the full table

    def seeds_for_hrf(self, hrf_index: torch.Tensor) -> PRFGridFit:
        """Re-seed every voxel at the given HRF's *own* best super-grid candidate.

        Forcing only ``hrf_index`` and leaving ``parameters`` at the global winner
        starts Gauss-Newton from the position the *winning* HRF preferred, which is
        not neutral: HRF delay trades against position along the sweep, so a
        non-winning HRF's grid optimum sits systematically elsewhere (about a third
        of a pixel per library step). Under a step limit and early stopping that
        biases the comparison toward whichever HRF supplied the seed. The super grid
        already scored every (candidate, HRF) pair, so the honest seed is free --
        it just has to be kept.

        Falls back to forcing the HRF alone when the per-HRF tables were not
        tracked, which keeps callers that never asked for them working unchanged.
        ``gain`` is left at the winner's value: refinement solves it by variable
        projection and never reads the seed.
        """
        if self.candidate_per_hrf is None or self.grid_parameters is None:
            return replace(self, hrf_index=hrf_index)
        rows = torch.arange(hrf_index.shape[0], device=hrf_index.device)
        candidate = self.candidate_per_hrf[rows, hrf_index].long()
        correlation = (
            self.correlation_per_hrf[rows, hrf_index]
            if self.correlation_per_hrf is not None
            else self.correlation
        )
        return replace(
            self,
            hrf_index=hrf_index,
            candidate_index=candidate,
            parameters=self.grid_parameters.to(self.parameters.device)[candidate],
            correlation=correlation,
            r2=correlation.square(),
        )


@dataclass(frozen=True)
class PRFRefinementConfig:
    """Numerical controls for bounded CSS Gauss-Newton refinement."""

    max_iter: int = 50
    damping: float = 1e-3
    step_tolerance: float = 1e-4
    min_sigma: float = 1e-3
    min_exponent: float = 1e-3
    max_line_search: int = 4
    fix_exponent: bool = False
    stagewise_exponent: bool = False


@dataclass
class PRFRefinedFit:
    """Refined CSS parameters and full-fit metrics for every voxel."""

    candidate_index: torch.Tensor
    hrf_index: torch.Tensor
    parameters: torch.Tensor  # (n_voxels, 4): row, column, sigma, exponent
    gain: torch.Tensor
    correlation: torch.Tensor
    r2: torch.Tensor  # Coefficient of determination after nuisance projection
    residual_ss: torch.Tensor
    n_iters: torch.Tensor
    converged: torch.Tensor


@dataclass
class PRFCrossValidationFit:
    """Held-out metrics from fold-local leave-one-run-out pRF fitting."""

    r2: torch.Tensor
    residual_ss: torch.Tensor
    total_ss: torch.Tensor


def stimulus_pixel_axes(
    stimulus_shape: tuple[int, int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the row and column coordinates of each aperture pixel.

    analyzePRF evaluates the Gaussian on a square ``resmx x resmx`` grid and then
    centers that square into the true ``rows x columns`` aperture (``placematrix``
    with an empty position). Every pRF coordinate — the super-grid seeds, the
    parameter bounds, and the reported eccentricity — therefore lives in the
    square frame, whose center is ``(1 + resmx) / 2`` on both axes. Working in
    the aperture's own 1..rows / 1..columns frame instead silently shifts every
    non-square fit; this offset reproduces the square frame exactly.
    """
    rows, columns = stimulus_shape
    extent = max(rows, columns)
    row_axis = torch.arange(1, rows + 1, device=device, dtype=dtype) + (extent - rows) // 2
    column_axis = torch.arange(1, columns + 1, device=device, dtype=dtype) + (extent - columns) // 2
    return row_axis, column_axis


def choose_aperture_bin_factor(length: int, target: int) -> int:
    """Pick the block-mean factor that divides ``length`` and lands nearest ``target``.

    An exact divisor keeps every output pixel an equal-area average of the same
    number of input pixels, so a 1080-pixel aperture becomes 108 (factor 10)
    rather than an uneven resampling to exactly 100. Ties prefer the smaller
    factor, i.e. the higher-resolution aperture. Divisors that would land below
    half the target are rejected: a prime length like 101 has only itself as a
    non-trivial divisor, and collapsing the axis to a single pixel is far worse
    than leaving it for the caller's resize fallback.
    """
    if length <= target or target < 1:
        return 1
    divisors = [
        factor
        for factor in range(1, length + 1)
        if length % factor == 0 and length / factor >= target / 2
    ]
    return min(divisors, key=lambda factor: (abs(length / factor - target), factor))


def downsample_aperture(movie: torch.Tensor, target: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Block-average a ``rows x columns x time`` aperture down to roughly ``target`` pixels.

    The pRF Gaussian is evaluated over every aperture pixel, so refinement cost
    and memory scale with the aperture area: a 1080x1080 stimulus is ~136x the
    work of the ~100x100 aperture analyzePRF resizes to. Block averaging turns a
    binary mask into fractional coverage, which is what the reference's
    ``imresize`` produces too. Axes whose length has no useful divisor fall back
    to an area-weighted resize, which is uneven but never leaves the aperture
    at full resolution.
    """
    rows, columns, _ = movie.shape
    row_factor = choose_aperture_bin_factor(rows, target)
    column_factor = choose_aperture_bin_factor(columns, target)
    if row_factor == 1 and column_factor == 1:
        return movie, (rows, columns)

    # A divisor of 1 on an axis that still needs shrinking means the length is
    # prime-ish; resize that axis rather than silently keeping it huge.
    stuck = (row_factor == 1 and rows > target) or (column_factor == 1 and columns > target)
    if stuck:
        frames = movie.permute(2, 0, 1).unsqueeze(1)
        resized = F.interpolate(
            frames,
            size=(min(rows, target), min(columns, target)),
            mode="area",
        )
        binned = resized[:, 0].permute(1, 2, 0).contiguous()
    else:
        binned = movie.reshape(
            rows // row_factor,
            row_factor,
            columns // column_factor,
            column_factor,
            movie.shape[2],
        ).mean(dim=(1, 3))
    return binned.contiguous(), (binned.shape[0], binned.shape[1])


def _hrf_convolve_columns(values: torch.Tensor, hrf: torch.Tensor) -> torch.Tensor:
    """Causally convolve every column of a ``(time, batch)`` matrix with one HRF.

    The HRF is shared across the batch, so this is a banded lower-triangular
    Toeplitz matrix applied to every column at once -- a plain GEMM.

    ``F.conv1d`` is the obvious spelling and was the original one, but the batch
    here is one channel per voxel-parameter (voxels x 4) against a kernel of a
    few dozen taps, which is the worst possible shape for cuDNN. With
    ``cudnn.benchmark`` on -- which ffs sets globally -- the first call probes
    algorithms needing a **3.4 GB** workspace (measured, 4,724-voxel chunk)
    purely to time them, and that one probe was the peak VRAM of the entire pRF
    fit, scaling with whatever chunk size the memory model picks. Turning
    autotuning off for the call removes the spike but picks an algorithm 15x
    slower. The GEMM has no autotune step at all, and is also 1.4-2x faster than
    the autotuned convolution and uses half the memory, measured at 300/600/1200
    timepoints. It costs ``n_timepoints`` more flops per output than the
    convolution does, so a run long enough for that to matter (far beyond 1200
    TRs) would want the FFT form instead.

    TF32 is switched off for the GEMM. It is worth ~20% here, but it makes the
    prediction differ from the convolution by ~2e-4 relative, and that is enough
    to push ill-conditioned voxels into a different local optimum of the flat
    sigma/exponent valley. Full fp32 keeps this bit-identical to the convolution
    it replaces, and is still faster than it.
    """
    n_timepoints = values.shape[0]
    positions = torch.arange(n_timepoints, device=values.device)
    lag = positions.view(-1, 1) - positions.view(1, -1)
    causal_hrf = torch.where(
        (lag >= 0) & (lag < hrf.numel()),
        hrf[lag.clamp(0, hrf.numel() - 1)],
        torch.zeros((), device=values.device, dtype=values.dtype),
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return causal_hrf @ values
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


class _HRFApplier:
    """Convolve a time-by-voxel block whose voxels may use different HRFs.

    Voxels are kept sorted by HRF, so each HRF owns a contiguous column range and
    is applied as one GEMM against a slice -- no gather, no scatter. This is what
    lets a mixed-HRF refinement run as a single full-width batch: everything
    upstream (receptive fields, neural drive, derivatives) is HRF-agnostic, and
    only this step has to know which voxel wants which HRF.

    Grouping the voxels *outside* the batch instead, one refinement per HRF, is
    what the code used to do, and it made every operation 20x smaller and purely
    launch-bound: ~100-row predictions taking as long as 4,724-row ones.
    """

    def __init__(self, hrfs: torch.Tensor, bounds: Sequence[int] | None = None) -> None:
        self.hrfs = hrfs if hrfs.ndim == 2 else hrfs.unsqueeze(0)
        if bounds is not None and len(bounds) != self.hrfs.shape[0] + 1:
            raise ValueError("bounds must hold one boundary per HRF plus a final end")
        self.bounds = None if bounds is None else [int(bound) for bound in bounds]
        self._matrix_cache: dict[int, torch.Tensor] = {}

    def with_bounds(self, bounds: Sequence[int]) -> _HRFApplier:
        """Return a view with new column boundaries, reusing the built matrices."""
        applier = _HRFApplier(self.hrfs, bounds)
        applier._matrix_cache = self._matrix_cache
        return applier

    def _matrices(self, n_timepoints: int) -> torch.Tensor:
        """Banded lower-triangular Toeplitz operators, one per HRF, cached per run length."""
        cached = self._matrix_cache.get(n_timepoints)
        if cached is None:
            positions = torch.arange(n_timepoints, device=self.hrfs.device)
            lag = positions.view(-1, 1) - positions.view(1, -1)
            valid = (lag >= 0) & (lag < self.hrfs.shape[1])
            gathered = self.hrfs[:, lag.clamp(0, self.hrfs.shape[1] - 1)]
            cached = torch.where(valid, gathered, torch.zeros((), device=self.hrfs.device))
            self._matrix_cache[n_timepoints] = cached
        return cached

    def apply(self, values: torch.Tensor, block: int = 1) -> torch.Tensor:
        """Convolve ``(time, voxels * block)`` columns, ``block`` adjacent per voxel."""
        matrices = self._matrices(values.shape[0])
        # TF32 would perturb the prediction by ~2e-4 relative, enough to push
        # ill-conditioned voxels into a different local optimum of the flat
        # sigma/exponent valley. See _hrf_convolve_columns.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            if self.bounds is None:
                return matrices[0] @ values
            convolved = torch.empty_like(values)
            for index, (start, end) in enumerate(
                zip(self.bounds[:-1], self.bounds[1:], strict=True)
            ):
                if end > start:
                    columns = slice(start * block, end * block)
                    convolved[:, columns] = matrices[index] @ values[:, columns]
            return convolved
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _as_applier(hrf: torch.Tensor | _HRFApplier) -> _HRFApplier:
    """Accept a single shared HRF or a per-voxel applier interchangeably."""
    return hrf if isinstance(hrf, _HRFApplier) else _HRFApplier(hrf)


def _group_bounds(group: torch.Tensor, n_groups: int) -> list[int]:
    """Start/end column of each group in an ascending-sorted group vector.

    One host transfer per call, which is why the boundaries are materialised as
    Python ints: reading them inside the convolution would sync per HRF per run.
    """
    counts = torch.bincount(group, minlength=n_groups)
    return [0, *torch.cumsum(counts, dim=0).tolist()]


def _subdivide_eccentricities(values: Sequence[float], steps: int) -> list[float]:
    """Insert ``steps - 1`` rings between consecutive reference eccentricities.

    The reference ring spacing is roughly geometric, so new rings are placed
    geometrically too -- linear interpolation would crowd the periphery and
    leave the fovea, where pRFs are smallest and most densely packed, coarse.
    """
    if steps <= 1:
        return list(values)
    subdivided = [float(values[0])]
    for inner, outer in zip(values[:-1], values[1:], strict=True):
        for step in range(1, steps + 1):
            fraction = step / steps
            if inner <= 0.0:
                subdivided.append(float(outer) * fraction)
            else:
                subdivided.append(float(inner) * (float(outer) / float(inner)) ** fraction)
    return subdivided


def make_analyzeprf_grid(
    stimulus_shape: tuple[int, int],
    *,
    eccentricities: Sequence[float] = (
        0.0,
        0.00551,
        0.014,
        0.0269,
        0.0459,
        0.0731,
        0.112,
        0.166,
        0.242,
        0.348,
        0.498,
        0.707,
        1.0,
    ),
    n_angles: int = 16,
    exponents: Sequence[float] = (0.5, 0.25, 0.125),
    sigma_steps_per_octave: int = 1,
    eccentricity_steps: int = 1,
    angle_mode: str = "uniform",
    sigma_mode: str = "absolute",
    size_slopes: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PRFGrid:
    """Construct the analyzePRF super-grid of Gaussian CSS candidates.

    Candidate coordinates use the MATLAB convention: row then column, both
    one-based.  The reference sigma grid is 1, 2, 4, ... pixels up to the longest
    stimulus dimension, and CSS exponents are the reference defaults.

    ``sigma_steps_per_octave`` and ``eccentricity_steps`` subdivide the sigma and
    eccentricity axes; with ``n_angles`` they control how finely the seed grid
    samples the parameter space. Density matters beyond seeding: the super-grid
    also picks the HRF, and its errors are quantization, not unidentifiability --
    the *correct* HRF is scored at an off-grid position, and that penalty can
    exceed the cost of a neighbouring HRF that happens to land on a better grid
    point. Finer sampling removes the handicap.

    ``angle_mode`` decides how the angular budget is spread over the rings.
    ``"uniform"`` is the reference's: ``n_angles`` on every ring, so arc spacing
    grows from 0.12 px at the innermost ring to 21 px at the outermost.
    ``"arc"`` scales the count with radius for constant pixel spacing.

    **Uniform measured better, and the reason is worth keeping.** At matched
    candidate counts (8.1k vs 8.7k) uniform picked the refine-selected HRF for
    84.2% of well-fit voxels against arc's 82.0%, and arc stayed behind even at
    twice the candidates. Constant *pixel* spacing is the wrong invariant: pRF
    size grows about linearly with eccentricity, so uniform angles already give
    constant spacing measured in pRF widths -- it is cortical-magnification-aware
    in the metric that matters. Making the spacing constant in pixels instead
    strips the mid-eccentricity rings down to 2-5 angles, which is genuinely too
    coarse where much of V1/V2 lives. ``"arc"`` is kept for stimulus geometries
    where that trade may differ, but it is not the default for good reason.

    ``sigma_mode`` is the one that did pay, modestly. ``"absolute"`` is the
    reference's: the same sigma ladder at every eccentricity, which spends
    candidates on combinations anatomy rules out -- a 64-pixel pRF at the fovea,
    a 1-pixel one at the far periphery. ``"slope"`` samples the slope of the
    linear size-versus-eccentricity relationship (Freeman & Simoncelli 2011)
    instead, so each ring gets the sizes plausible *there*; this is the CNI
    toolbox's parametrisation. Measured, it lifts HRF agreement 84.2% -> 85.1%
    for 22% more candidates (0.8 -> 1.0 s), and 86.0% when combined with a much
    finer angular grid. Worth having; not a transformation.
    """
    rows, columns = stimulus_shape
    if rows < 1 or columns < 1:
        raise ValueError(f"stimulus_shape must be positive, got {stimulus_shape}")
    if n_angles < 1:
        raise ValueError("n_angles must be positive")
    if sigma_steps_per_octave < 1 or eccentricity_steps < 1:
        raise ValueError("sigma_steps_per_octave and eccentricity_steps must be positive")
    if angle_mode not in ("uniform", "arc"):
        raise ValueError(f"angle_mode must be 'uniform' or 'arc', got {angle_mode!r}")
    if sigma_mode not in ("absolute", "slope"):
        raise ValueError(f"sigma_mode must be 'absolute' or 'slope', got {sigma_mode!r}")
    if sigma_mode == "slope" and (not size_slopes or any(s <= 0 for s in size_slopes)):
        raise ValueError("size_slopes must contain positive values")
    if not exponents or any(exponent <= 0 for exponent in exponents):
        raise ValueError("exponents must contain positive values")

    extent = max(rows, columns)
    n_octaves = extent.bit_length()
    sigmas = [
        float(2 ** (octave + step / sigma_steps_per_octave))
        for octave in range(n_octaves)
        for step in range(sigma_steps_per_octave)
        if 2 ** (octave + step / sigma_steps_per_octave) <= extent
    ]
    eccentricities = _subdivide_eccentricities(eccentricities, eccentricity_steps)
    largest_eccentricity = max(eccentricities) or 1.0
    # Seeds live in the square resmx frame (see stimulus_pixel_axes), so both
    # centers are (1 + resmx) / 2 even when the aperture is not square.
    center_row = center_column = (1.0 + extent) / 2.0
    candidates: list[tuple[float, float, float, float]] = []

    for eccentricity in eccentricities:
        if eccentricity == 0:
            ring_angles = 1
        elif angle_mode == "arc":
            ring_angles = max(1, round(n_angles * eccentricity / largest_eccentricity))
        else:
            ring_angles = n_angles
        if sigma_mode == "slope":
            # Sizes plausible at THIS eccentricity, floored at the smallest
            # absolute sigma so the foveal ring is not left with sigma ~ 0.
            radius = eccentricity * extent
            ring_sigmas = sorted({max(slope * radius, sigmas[0]) for slope in size_slopes})
        else:
            ring_sigmas = sigmas
        for index in range(ring_angles):
            angle = index * (2 * math.pi / ring_angles)
            row = center_row - math.sin(angle) * eccentricity * extent
            column = center_column + math.cos(angle) * eccentricity * extent
            for sigma in ring_sigmas:
                for exponent in exponents:
                    candidates.append((row, column, sigma * exponent**0.5, exponent))

    return PRFGrid(torch.tensor(candidates, device=device, dtype=dtype))


def gaussian_receptive_fields(
    parameters: torch.Tensor,
    stimulus_shape: tuple[int, int],
) -> torch.Tensor:
    """Return flattened, continuous unit-mass Gaussian receptive fields.

    The normalization is ``2*pi*sigma^2``, matching analyzePRF rather than
    renormalizing each image after its support is clipped at the aperture edge.
    """
    if parameters.ndim != 2 or parameters.shape[1] != 4:
        raise ValueError("parameters must have shape (n_candidates, 4)")
    row_axis, column_axis = stimulus_pixel_axes(
        stimulus_shape, device=parameters.device, dtype=parameters.dtype
    )
    row_grid, column_grid = torch.meshgrid(row_axis, column_axis, indexing="ij")

    center_row = parameters[:, 0, None, None]
    center_column = parameters[:, 1, None, None]
    sigma = parameters[:, 2, None, None].abs().clamp_min(torch.finfo(parameters.dtype).eps)
    squared_distance = (row_grid - center_row).square() + (column_grid - center_column).square()
    fields = torch.exp(-squared_distance / (2 * sigma.square())) / (2 * torch.pi * sigma.square())
    return fields.flatten(start_dim=1).contiguous()


def predict_prf_runwise(
    stimulus_runs: Sequence[torch.Tensor],
    receptive_fields: torch.Tensor,
    exponents: torch.Tensor,
    hrf: torch.Tensor | _HRFApplier,
) -> torch.Tensor:
    """Predict one time series per candidate without convolving across runs."""
    if receptive_fields.ndim != 2:
        raise ValueError("receptive_fields must have shape (n_candidates, n_pixels)")
    n_candidates, n_pixels = receptive_fields.shape
    if exponents.shape != (n_candidates,):
        raise ValueError("exponents must contain one value per receptive field")
    if isinstance(hrf, torch.Tensor) and (hrf.ndim != 1 or hrf.numel() == 0):
        raise ValueError("hrf must be a non-empty one-dimensional tensor")
    applier = _as_applier(hrf)

    predictions = []
    for run_index, stimulus in enumerate(stimulus_runs):
        if stimulus.ndim != 2 or stimulus.shape[1] != n_pixels:
            raise ValueError(
                f"stimulus run {run_index} must have shape (time, {n_pixels}), got {tuple(stimulus.shape)}"
            )
        neural_drive = (stimulus @ receptive_fields.T).clamp_min(0).pow(exponents)
        predictions.append(applier.apply(neural_drive))
    return torch.cat(predictions, dim=0)


def _project_per_run(
    values: torch.Tensor,
    nuisance_per_run: Sequence[torch.Tensor] | None,
    run_starts: Sequence[int],
) -> torch.Tensor:
    """Project time-by-feature values from block-diagonal per-run nuisance."""
    if nuisance_per_run is None:
        return values
    run_ends = [*run_starts[1:], values.shape[0]]
    if len(nuisance_per_run) != len(run_starts):
        raise ValueError("nuisance_per_run must contain one matrix per run")

    projected = []
    for run_index, (start, end, nuisance) in enumerate(
        zip(run_starts, run_ends, nuisance_per_run, strict=True)
    ):
        run_values = values[start:end]
        nuisance = nuisance.to(device=values.device, dtype=values.dtype)
        if nuisance.shape[0] != run_values.shape[0]:
            raise ValueError(
                f"nuisance run {run_index} has {nuisance.shape[0]} rows, expected {run_values.shape[0]}"
            )
        projected.append(orthogonalize_design(run_values, nuisance))
    return torch.cat(projected, dim=0)


def project_nuisance_per_run(
    data: torch.Tensor,
    nuisance_per_run: Sequence[torch.Tensor] | None,
    run_starts: Sequence[int],
) -> torch.Tensor:
    """Project per-run nuisance out of ``(n_voxels, n_timepoints)`` data.

    The voxel-major public spelling of the projection the fitting path uses
    internally. Anything comparing raw timecourses across runs -- a noise
    ceiling above all -- has to see the same projected data the fit is scored
    on, or shared drift gets counted as reproducible signal.
    """
    return _project_per_run(data.T, nuisance_per_run, run_starts).T


def _project_prf_derivatives(
    derivatives: torch.Tensor,
    nuisance_per_run: Sequence[torch.Tensor] | None,
    run_starts: Sequence[int],
) -> torch.Tensor:
    """Project ``(time, voxel, parameter)`` derivatives per run."""
    n_timepoints, n_voxels, n_parameters = derivatives.shape
    flattened = derivatives.reshape(n_timepoints, n_voxels * n_parameters)
    return _project_per_run(flattened, nuisance_per_run, run_starts).reshape(
        n_timepoints, n_voxels, n_parameters
    )


def _css_prediction_and_derivatives(
    stimulus_runs: Sequence[torch.Tensor],
    parameters: torch.Tensor,
    hrf: torch.Tensor | _HRFApplier,
    stimulus_shape: tuple[int, int],
    n_derivatives: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict CSS responses and analytic derivatives for one voxel batch.

    Parameters are ``(row, column, sigma, exponent)`` in stimulus-pixel units.
    The returned derivatives are ordered identically and exclude gain, which is
    solved exactly by variable projection in the GN loop.

    ``n_derivatives=3`` omits the exponent derivative for the fixed-exponent
    (linear pRF) model, which never uses it. That drops a log over the whole
    time-by-voxel block and a quarter of the derivative convolution and of the
    derivative buffer, every iteration.
    """
    if n_derivatives not in (3, 4):
        raise ValueError(f"n_derivatives must be 3 or 4, got {n_derivatives}")
    n_voxels = parameters.shape[0]
    applier = _as_applier(hrf)
    dtype, device = parameters.dtype, parameters.device
    row_axis, column_axis = stimulus_pixel_axes(stimulus_shape, device=device, dtype=dtype)
    row_grid, column_grid = torch.meshgrid(row_axis, column_axis, indexing="ij")
    row_grid = row_grid.reshape(1, -1)
    column_grid = column_grid.reshape(1, -1)

    center_row = parameters[:, 0:1]
    center_column = parameters[:, 1:2]
    sigma = parameters[:, 2:3].clamp_min(torch.finfo(dtype).eps)
    exponent = parameters[:, 3:4].clamp_min(torch.finfo(dtype).eps)
    row_delta = row_grid - center_row
    column_delta = column_grid - center_column
    squared_distance = row_delta.square() + column_delta.square()
    fields = torch.exp(-squared_distance / (2 * sigma.square())) / (2 * torch.pi * sigma.square())
    # Each spatial derivative is a (voxel, pixel) field as large as `fields`
    # itself, so they are formed and consumed one at a time: stacking all three
    # is what makes the aperture term dominate the refinement's memory model.
    del squared_distance

    predictions: list[torch.Tensor] = []
    derivatives: list[torch.Tensor] = []
    eps = torch.finfo(dtype).eps
    for stimulus in stimulus_runs:
        n_timepoints = stimulus.shape[0]
        drive = stimulus @ fields.T
        # The value must match `_css_prediction` exactly (the line search compares
        # the two objectives), so only the chain rule and the log use the eps floor.
        neural = drive.clamp_min(0).pow(exponent.T)
        positive_drive = drive.clamp_min(eps)
        # d(drive^n)/d(drive), shared by all three spatial derivatives.
        drive_chain = exponent.T * positive_drive.pow(exponent.T - 1)

        neural_derivatives = torch.empty(
            (n_timepoints, n_voxels, n_derivatives), device=device, dtype=dtype
        )
        inverse_variance = sigma.square().reciprocal()
        neural_derivatives[:, :, 0] = drive_chain * (
            stimulus @ (fields * row_delta * inverse_variance).T
        )
        neural_derivatives[:, :, 1] = drive_chain * (
            stimulus @ (fields * column_delta * inverse_variance).T
        )
        neural_derivatives[:, :, 2] = drive_chain * (
            stimulus
            @ (
                fields * ((row_delta.square() + column_delta.square()) / sigma.pow(3) - 2.0 / sigma)
            ).T
        )
        if n_derivatives == 4:
            neural_derivatives[:, :, 3] = neural * positive_drive.log()

        predictions.append(applier.apply(neural))
        derivatives.append(
            applier.apply(
                neural_derivatives.reshape(n_timepoints, n_voxels * n_derivatives),
                block=n_derivatives,
            ).view(n_timepoints, n_voxels, n_derivatives)
        )
    return torch.cat(predictions, dim=0), torch.cat(derivatives, dim=0)


def _css_prediction(
    stimulus_runs: Sequence[torch.Tensor],
    parameters: torch.Tensor,
    hrf: torch.Tensor | _HRFApplier,
    stimulus_shape: tuple[int, int],
) -> torch.Tensor:
    """Predict CSS responses without the derivatives the line search never uses."""
    fields = gaussian_receptive_fields(parameters, stimulus_shape)
    exponent = parameters[:, 3].clamp_min(torch.finfo(parameters.dtype).eps)
    return predict_prf_runwise(stimulus_runs, fields, exponent, hrf)


def _bounded_css_parameters(
    parameters: torch.Tensor,
    stimulus_shape: tuple[int, int],
    config: PRFRefinementConfig,
) -> torch.Tensor:
    """Apply analyzePRF-compatible parameter bounds after a GN update."""
    extent = float(max(stimulus_shape))
    bounded = parameters.clone()
    bounded[:, 0].clamp_(2.0 - extent, 2.0 * extent - 1.0)
    bounded[:, 1].clamp_(2.0 - extent, 2.0 * extent - 1.0)
    bounded[:, 2].clamp_(min=config.min_sigma)
    bounded[:, 3].clamp_(min=config.min_exponent)
    return bounded


def _variable_projection_sse(data: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Return the profiled residual sum of squares at the optimal non-negative gain."""
    eps = torch.finfo(data.dtype).eps
    gain = (
        (prediction * data).sum(dim=0) / prediction.square().sum(dim=0).clamp_min(eps)
    ).clamp_min(0)
    return (data - prediction * gain).square().sum(dim=0)


def _variable_projection_terms(
    data: torch.Tensor,
    prediction: torch.Tensor,
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gain, residual, variable-projection Jacobian, and SSE.

    ``data`` and ``prediction`` are time-by-voxel. The exact derivative of the
    per-voxel OLS gain is included, so GN follows the profiled CSS objective
    instead of treating gain as an additional slow nonlinear parameter.
    """
    eps = torch.finfo(data.dtype).eps
    denominator = prediction.square().sum(dim=0).clamp_min(eps)
    numerator = (prediction * data).sum(dim=0)
    unconstrained_gain = numerator / denominator
    gain = unconstrained_gain.clamp_min(0)
    residual = data - prediction * gain

    derivative_data = (derivatives * data.unsqueeze(-1)).sum(dim=0)
    derivative_prediction = (derivatives * prediction.unsqueeze(-1)).sum(dim=0)
    gain_derivative = (
        derivative_data * denominator.unsqueeze(1)
        - numerator.unsqueeze(1) * 2.0 * derivative_prediction
    ) / denominator.square().unsqueeze(1)
    jacobian = -(derivatives * gain.view(1, -1, 1) + prediction.unsqueeze(-1) * gain_derivative)
    # At the non-negative gain boundary, the profiled model is the zero predictor
    # locally; permitting a GN spatial update there creates artificial motion.
    jacobian[:, unconstrained_gain <= 0] = 0
    return gain, residual, jacobian, residual.square().sum(dim=0)


def _refine_css_batch(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    parameters: torch.Tensor,
    hrf: torch.Tensor | _HRFApplier,
    stimulus_shape: tuple[int, int],
    run_starts: Sequence[int],
    nuisance_per_run: Sequence[torch.Tensor] | None,
    config: PRFRefinementConfig,
    hrf_group: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Damped, batched GN refinement for one voxel chunk.

    ``hrf_group`` gives each voxel's index into a multi-HRF ``hrf`` applier and
    must be sorted ascending, so every HRF owns a contiguous column range. That
    lets voxels using different HRFs refine together in one full-width batch
    instead of one small batch per HRF.

    The batch shrinks as voxels finish. Voxels leave at very different rates --
    measured on real data, 99% are still iterating at iteration 10 but only 14%
    at iteration 25 and 1% at iteration 50 -- so a fixed-width batch spends over
    half its work recomputing voxels that converged or exhausted long ago. Every
    working tensor is therefore compacted to the live set as soon as any voxel
    drops out, and results are scattered back through ``live``.

    The Jacobian is also cached across rejected steps. A rejected line search
    leaves the parameters exactly where they were and changes only the damping,
    so the next sweep's Jacobian would be recomputed identically. Since ~65% of
    steps are rejected here, that recomputation was most of the Jacobian cost.
    """
    projected_data = _project_per_run(data.T, nuisance_per_run, run_starts)
    current = _bounded_css_parameters(parameters, stimulus_shape, config)
    n_voxels = current.shape[0]
    n_free = 3 if config.fix_exponent else 4

    applier = _as_applier(hrf)
    n_groups = applier.hrfs.shape[0]
    if hrf_group is not None and n_groups > 1:
        group = hrf_group
        applier = applier.with_bounds(_group_bounds(group, n_groups))
    else:
        group = None
    solve_dtype = torch.float32 if current.device.type == "mps" else torch.float64

    final_parameters = current.clone()
    n_iters = torch.full((n_voxels,), config.max_iter, device=current.device, dtype=torch.int32)
    converged = torch.zeros(n_voxels, device=current.device, dtype=torch.bool)

    # `live` maps working rows back to voxel rows; every live voxel is active.
    live = torch.arange(n_voxels, device=current.device)
    damping = torch.full((n_voxels,), config.damping, device=current.device, dtype=current.dtype)
    jtj = torch.zeros((n_voxels, n_free, n_free), device=current.device, dtype=solve_dtype)
    rhs = torch.zeros((n_voxels, n_free), device=current.device, dtype=solve_dtype)
    current_ss = torch.zeros(n_voxels, device=current.device, dtype=current.dtype)
    stale = torch.ones(n_voxels, device=current.device, dtype=torch.bool)
    identity = torch.eye(n_free, device=current.device, dtype=solve_dtype)

    for iteration in range(config.max_iter):
        refresh = torch.nonzero(stale, as_tuple=False).squeeze(1)
        if refresh.numel():
            # `refresh` keeps ascending order, so the HRF groups stay contiguous
            # within the gathered subset -- only the boundaries move.
            refresh_applier = (
                applier
                if group is None
                else applier.with_bounds(_group_bounds(group[refresh], n_groups))
            )
            prediction, derivatives = _css_prediction_and_derivatives(
                stimulus_runs,
                current[refresh],
                refresh_applier,
                stimulus_shape,
                n_derivatives=n_free,
            )
            prediction = _project_per_run(prediction, nuisance_per_run, run_starts)
            derivatives = _project_prf_derivatives(derivatives, nuisance_per_run, run_starts)
            _, residual, jacobian, refreshed_ss = _variable_projection_terms(
                projected_data[:, refresh], prediction, derivatives
            )
            # The Jacobian is already only the free parameters: the fixed-exponent
            # model never forms the exponent column.
            jtj[refresh] = torch.einsum("tbp,tbq->bpq", jacobian, jacobian).to(solve_dtype)
            rhs[refresh] = (-torch.einsum("tbp,tb->bp", jacobian, residual)).to(solve_dtype)
            current_ss[refresh] = refreshed_ss
            del prediction, derivatives, jacobian, residual

        diagonal_scale = jtj.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)
        normal = jtj + identity * (damping.to(solve_dtype) * diagonal_scale).view(-1, 1, 1)
        try:
            free_step = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1).to(current.dtype)
        except torch.linalg.LinAlgError:
            free_step = (
                (torch.linalg.pinv(normal) @ rhs.unsqueeze(-1)).squeeze(-1).to(current.dtype)
            )
        step = torch.zeros_like(current)
        step[:, :n_free] = free_step

        accepted = torch.zeros_like(converged[live])
        trial_scale = torch.ones_like(damping)
        best_parameters = current
        for _ in range(config.max_line_search):
            trial = _bounded_css_parameters(
                current + step * trial_scale.unsqueeze(1), stimulus_shape, config
            )
            trial_prediction = _project_per_run(
                _css_prediction(stimulus_runs, trial, applier, stimulus_shape),
                nuisance_per_run,
                run_starts,
            )
            trial_ss = _variable_projection_sse(projected_data, trial_prediction)
            improved = ~accepted & torch.isfinite(trial_ss) & (trial_ss < current_ss)
            best_parameters = torch.where(improved.unsqueeze(1), trial, best_parameters)
            accepted = accepted | improved
            # Each voxel keeps its FIRST accepted trial, so once every live voxel
            # has one, the remaining halvings only re-predict. Exiting is exact.
            if bool(accepted.all()):
                break
            trial_scale = torch.where(accepted, trial_scale, trial_scale * 0.5)

        effective_step = (best_parameters - current).abs().amax(dim=1)
        current = torch.where(accepted.unsqueeze(1), best_parameters, current)
        # Classic Levenberg bookkeeping: a rejected step raises the damping and is
        # retried on the next sweep. Dropping the voxel on first rejection would
        # freeze it at the point where the Gauss-Newton step was merely too long.
        damping = torch.where(
            accepted, (damping * 0.5).clamp_min(1e-8), (damping * 10.0).clamp_max(1e8)
        )
        newly_converged = accepted & (effective_step < config.step_tolerance)
        exhausted = ~accepted & (damping >= 1e8)

        final_parameters[live] = current
        n_iters[live[newly_converged]] = iteration + 1
        converged[live[newly_converged]] = True

        # Only voxels whose parameters actually moved need a new Jacobian.
        stale = accepted
        keep = torch.nonzero(~(newly_converged | exhausted), as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            break
        if keep.numel() != live.numel():
            live = live[keep]
            current = current[keep].contiguous()
            projected_data = projected_data[:, keep].contiguous()
            damping = damping[keep]
            jtj = jtj[keep]
            rhs = rhs[keep]
            current_ss = current_ss[keep]
            stale = stale[keep]
            if group is not None:
                group = group[keep]
                applier = applier.with_bounds(_group_bounds(group, n_groups))

    current = final_parameters
    projected_data = _project_per_run(data.T, nuisance_per_run, run_starts)
    final_applier = (
        _as_applier(hrf)
        if hrf_group is None or n_groups == 1
        else _as_applier(hrf).with_bounds(_group_bounds(hrf_group, n_groups))
    )
    prediction, derivatives = _css_prediction_and_derivatives(
        stimulus_runs, current, final_applier, stimulus_shape
    )
    prediction = _project_per_run(prediction, nuisance_per_run, run_starts)
    derivatives = _project_prf_derivatives(derivatives, nuisance_per_run, run_starts)
    gain, _, _, residual_ss = _variable_projection_terms(projected_data, prediction, derivatives)
    ss_total = (projected_data - projected_data.mean(dim=0, keepdim=True)).square().sum(dim=0)
    r2 = 1.0 - residual_ss / ss_total.clamp_min(torch.finfo(data.dtype).eps)
    correlation = (prediction * projected_data).sum(dim=0) / (
        torch.linalg.vector_norm(prediction, dim=0)
        * torch.linalg.vector_norm(projected_data, dim=0)
    ).clamp_min(torch.finfo(data.dtype).eps)
    return current, gain, correlation, r2, residual_ss, n_iters, converged


def refine_prf_supergrid(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid_fit: PRFGridFit,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    *,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    voxel_chunk_size: int = 20_000,
    device: torch.device | None = None,
    config: PRFRefinementConfig = PRFRefinementConfig(),
    verbose: bool = False,
) -> PRFRefinedFit:
    """Refine super-grid seeds per voxel.

    Voxels are sorted by their selected HRF and then chunked, so a chunk holds
    contiguous runs of each HRF and refines them together at full width. Chunking
    by HRF *first* -- one batch per HRF -- is the obvious implementation and was
    the original one, but it makes every batch ``n_voxels / n_hrfs`` wide and
    leaves the GPU launch-bound: with a 20-HRF library it ran ~100-row
    predictions that cost as much as the full-width ones.
    """
    if data.ndim != 2:
        raise ValueError("data must have shape (n_voxels, n_timepoints)")
    if voxel_chunk_size < 1:
        raise ValueError("voxel_chunk_size must be positive")
    if grid_fit.parameters.shape != (data.shape[0], 4):
        raise ValueError("grid_fit parameters must contain one four-parameter seed per voxel")

    device = device or data.device
    output_device = data.device
    n_voxels = data.shape[0]
    parameters = torch.empty((n_voxels, 4), device=output_device, dtype=data.dtype)
    gain = torch.empty(n_voxels, device=output_device, dtype=data.dtype)
    correlation = torch.empty(n_voxels, device=output_device, dtype=data.dtype)
    r2 = torch.empty(n_voxels, device=output_device, dtype=data.dtype)
    residual_ss = torch.empty(n_voxels, device=output_device, dtype=data.dtype)
    n_iters = torch.empty(n_voxels, device=output_device, dtype=torch.int32)
    converged = torch.empty(n_voxels, device=output_device, dtype=torch.bool)
    stimuli = [run.to(device=device, dtype=data.dtype) for run in stimulus_runs]
    hrf_library = hrf_library.to(device=device, dtype=data.dtype)

    progress = None
    if verbose:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=n_voxels, desc="pRF refinement", unit="vox", leave=True, disable=n_voxels < 1000
        )

    # Sorting by HRF makes every HRF a contiguous column range inside a chunk,
    # which is what _HRFApplier needs to apply each one as a single GEMM.
    hrf_order = torch.argsort(grid_fit.hrf_index, stable=True)
    used_hrfs, dense_group = grid_fit.hrf_index[hrf_order].unique(return_inverse=True)
    chunk_library = hrf_library[used_hrfs.to(hrf_library.device)]
    # One applier for every chunk: the Toeplitz operators are (n_hrfs, T, T) and
    # rebuilding them per chunk would repeat that for no reason.
    applier = _HRFApplier(chunk_library)

    for start in range(0, n_voxels, voxel_chunk_size):
        chunk_indices = hrf_order[start : start + voxel_chunk_size]
        chunk_group = dense_group[start : start + voxel_chunk_size].to(device)
        if config.stagewise_exponent and not config.fix_exponent:
            stage_config = replace(
                config,
                max_iter=max(1, config.max_iter // 2),
                fix_exponent=True,
                stagewise_exponent=False,
            )
            stage_parameters, _, _, _, _, stage_iters, _ = _refine_css_batch(
                data[chunk_indices].to(device=device, dtype=data.dtype),
                stimuli,
                grid_fit.parameters[chunk_indices].to(device=device, dtype=data.dtype),
                applier,
                stimulus_shape,
                run_starts,
                nuisance_per_run,
                stage_config,
                hrf_group=chunk_group,
            )
            refine_parameters = stage_parameters
        else:
            refine_parameters = grid_fit.parameters[chunk_indices].to(
                device=device, dtype=data.dtype
            )
            stage_iters = torch.zeros(chunk_indices.numel(), device=device, dtype=torch.int32)
        (
            chunk_parameters,
            chunk_gain,
            chunk_corr,
            chunk_r2,
            chunk_ss,
            chunk_iters,
            chunk_converged,
        ) = _refine_css_batch(
            data[chunk_indices].to(device=device, dtype=data.dtype),
            stimuli,
            refine_parameters,
            applier,
            stimulus_shape,
            run_starts,
            nuisance_per_run,
            replace(config, stagewise_exponent=False),
            hrf_group=chunk_group,
        )
        parameters[chunk_indices] = chunk_parameters.to(output_device)
        gain[chunk_indices] = chunk_gain.to(output_device)
        correlation[chunk_indices] = chunk_corr.to(output_device)
        r2[chunk_indices] = chunk_r2.to(output_device)
        residual_ss[chunk_indices] = chunk_ss.to(output_device)
        n_iters[chunk_indices] = (stage_iters + chunk_iters).to(output_device)
        converged[chunk_indices] = chunk_converged.to(output_device)
        if progress is not None:
            progress.update(chunk_indices.numel())
    if progress is not None:
        progress.close()

    return PRFRefinedFit(
        candidate_index=grid_fit.candidate_index,
        hrf_index=grid_fit.hrf_index,
        parameters=parameters,
        gain=gain,
        correlation=correlation,
        r2=r2,
        residual_ss=residual_ss,
        n_iters=n_iters,
        converged=converged,
    )


def refine_prf_all_hrfs(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid_fit: PRFGridFit,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    *,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    voxel_chunk_size: int = 20_000,
    device: torch.device | None = None,
    config: PRFRefinementConfig = PRFRefinementConfig(),
    keep_hrf_index: int | None = None,
    verbose: bool = False,
) -> tuple[PRFRefinedFit, torch.Tensor, PRFRefinedFit | None]:
    """Refine under every HRF and keep each voxel's best, returning the full R2 map.

    The super-grid picks its HRF against position quantized to 16 angles and
    powers-of-two sigmas, and at that resolution the HRF is very nearly
    unidentifiable: on noiseless synthetic data the grid recovers the true HRF
    about 9% of the time against a 5% chance rate. The failure is silent, because
    a wrong HRF is absorbed almost perfectly by sliding the pRF along the
    stimulus sweep -- R2 stays at 0.997 either way while position picks up a
    systematic bias of roughly a third of a pixel per HRF step.

    Refining under each HRF separates them (about 62% exact, and the residual
    error is one HRF step rather than five), because only then is position free
    enough for the timing mismatch to show up in the residual.

    Returns the best fit per voxel and an ``(n_voxels, n_hrfs)`` R2 map. Save the
    map: a voxel whose R2 is flat across the library has no HRF evidence, and its
    selected index should not be interpreted.

    ``keep_hrf_index`` additionally returns that one HRF's fit unreduced, for
    free — every HRF is refined anyway. It is how the fixed-canonical-HRF result
    is obtained alongside the selected one, so the two can be compared voxelwise
    on the same data rather than across two runs of the tool.
    """
    n_hrfs = hrf_library.shape[0]
    if keep_hrf_index is not None and not 0 <= keep_hrf_index < n_hrfs:
        raise ValueError(f"keep_hrf_index {keep_hrf_index} is outside the {n_hrfs}-HRF library")
    kept_fit: PRFRefinedFit | None = None
    r2_per_hrf = torch.full(
        (data.shape[0], n_hrfs), -torch.inf, device=data.device, dtype=data.dtype
    )
    best_fit: PRFRefinedFit | None = None

    hrf_iterator = range(n_hrfs)
    if verbose:
        from tqdm.auto import tqdm

        hrf_iterator = tqdm(hrf_iterator, desc="pRF refine per HRF", leave=True)

    for hrf_index in hrf_iterator:
        forced = grid_fit.seeds_for_hrf(torch.full_like(grid_fit.hrf_index, hrf_index))
        fit = refine_prf_supergrid(
            data,
            stimulus_runs,
            stimulus_shape,
            forced,
            hrf_library,
            run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            config=config,
        )
        r2_per_hrf[:, hrf_index] = fit.r2
        if hrf_index == keep_hrf_index:
            kept_fit = fit
        best_fit = _keep_better_fit(best_fit, fit)

    assert best_fit is not None
    return best_fit, r2_per_hrf, kept_fit


def _keep_better_fit(best: PRFRefinedFit | None, candidate: PRFRefinedFit) -> PRFRefinedFit:
    """Per-voxel elementwise max over R2 of two complete fits."""
    if best is None:
        return candidate
    improved = candidate.r2 > best.r2
    return PRFRefinedFit(
        # Seeds differ per HRF once seeds_for_hrf is in play, so the winning
        # candidate has to be selected like every other field.
        candidate_index=torch.where(improved, candidate.candidate_index, best.candidate_index),
        hrf_index=torch.where(improved, candidate.hrf_index, best.hrf_index),
        parameters=torch.where(improved.unsqueeze(1), candidate.parameters, best.parameters),
        gain=torch.where(improved, candidate.gain, best.gain),
        correlation=torch.where(improved, candidate.correlation, best.correlation),
        r2=torch.where(improved, candidate.r2, best.r2),
        residual_ss=torch.where(improved, candidate.residual_ss, best.residual_ss),
        n_iters=torch.where(improved, candidate.n_iters, best.n_iters),
        converged=torch.where(improved, candidate.converged, best.converged),
    )


def refine_prf_hrf_window(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid_fit: PRFGridFit,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    *,
    window: int = 1,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    voxel_chunk_size: int = 20_000,
    device: torch.device | None = None,
    config: PRFRefinementConfig = PRFRefinementConfig(),
    verbose: bool = False,
) -> PRFRefinedFit:
    """Refine only the HRFs within ``window`` steps of each voxel's grid choice.

    The middle ground between trusting the super-grid's HRF and refitting all of
    them. It is justified by where the grid actually errs: on real data its
    choice lands within one library step of the refine-selected HRF for 99.3% of
    voxels at R2>0.2 and 100% at R2>0.4, so a +/-1 window recovers essentially
    the whole benefit of the full search at a fraction of its cost.

    The window **slides** at the library edges rather than clamping: a voxel that
    picked HRF 0 is scored on 0, 1, 2 -- not on a degenerate two-wide window. Every
    voxel therefore gets the same number of candidates, which keeps the selection
    from being quietly weaker at the ends of the library.
    """
    n_hrfs = hrf_library.shape[0]
    width = 2 * window + 1
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    if width >= n_hrfs:
        return refine_prf_all_hrfs(
            data,
            stimulus_runs,
            stimulus_shape,
            grid_fit,
            hrf_library,
            run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            config=config,
            verbose=verbose,
        )[0]

    window_start = (grid_fit.hrf_index - window).clamp(0, n_hrfs - width)
    best_fit: PRFRefinedFit | None = None
    slots = range(width)
    if verbose:
        from tqdm.auto import tqdm

        slots = tqdm(slots, desc=f"pRF refine grid+/-{window}", leave=True)

    for slot in slots:
        forced = grid_fit.seeds_for_hrf(window_start + slot)
        fit = refine_prf_supergrid(
            data,
            stimulus_runs,
            stimulus_shape,
            forced,
            hrf_library,
            run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            config=config,
        )
        best_fit = _keep_better_fit(best_fit, fit)

    assert best_fit is not None
    return best_fit


def refine_prf_hrf_ranked(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid_fit: PRFGridFit,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    *,
    n_extra: int = 2,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    voxel_chunk_size: int = 20_000,
    device: torch.device | None = None,
    config: PRFRefinementConfig = PRFRefinementConfig(),
    verbose: bool = False,
) -> PRFRefinedFit:
    """Refine the grid's HRF plus the ``n_extra`` next-best HRFs *by grid fit*.

    The library-index window of :func:`refine_prf_hrf_window` assumes neighbouring
    indices are neighbouring shapes, which holds for a swept library but not for a
    Latin-hypercube (pighs) draw, where index order is close to arbitrary. Ranking
    by each voxel's own super-grid correlation is shape-agnostic: it picks the
    HRFs that actually came closest to fitting that voxel, whatever their position
    in the library.

    ``n_extra`` counts the *additional* candidates, so ``n_extra=2`` refines three
    HRFs per voxel -- the same budget as ``window=1``.

    Requires ``grid_fit.correlation_per_hrf``, i.e. the super grid must have run
    with ``track_per_hrf=True``.
    """
    if n_extra < 1:
        raise ValueError(f"n_extra must be positive, got {n_extra}")
    if grid_fit.correlation_per_hrf is None:
        raise ValueError(
            "refine_prf_hrf_ranked needs per-HRF grid scores; call fit_prf_supergrid "
            "with track_per_hrf=True"
        )
    n_hrfs = hrf_library.shape[0]
    width = min(n_extra + 1, n_hrfs)
    if width >= n_hrfs:
        return refine_prf_all_hrfs(
            data,
            stimulus_runs,
            stimulus_shape,
            grid_fit,
            hrf_library,
            run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            config=config,
            verbose=verbose,
        )[0]

    ranked = grid_fit.correlation_per_hrf.topk(width, dim=1).indices
    best_fit: PRFRefinedFit | None = None
    slots = range(width)
    if verbose:
        from tqdm.auto import tqdm

        slots = tqdm(slots, desc=f"pRF refine grid-{n_extra}", leave=True)

    for slot in slots:
        forced = grid_fit.seeds_for_hrf(ranked[:, slot].to(grid_fit.hrf_index.device))
        fit = refine_prf_supergrid(
            data,
            stimulus_runs,
            stimulus_shape,
            forced,
            hrf_library,
            run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            config=config,
        )
        best_fit = _keep_better_fit(best_fit, fit)

    assert best_fit is not None
    return best_fit


def summarize_hrf_selection(
    r2_per_hrf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce an ``(n_voxels, n_hrfs)`` R2 map to index, sub-step index, evidence.

    Which criterion *should* pick the HRF is unsettled -- see the wiki note
    [[pRF HRF selection]]. What is not in doubt is that a single argmax is a
    lossy summary of this map, so all three come back:

    ``index``
        Plain argmax. The provisional choice.
    ``continuous_index``
        Parabolic interpolation through the argmax and its two neighbours. The
        R2-vs-HRF curve is smooth and unimodal, so its peak generally falls
        between library entries; this recovers that without enlarging the
        library, and is far less jumpy than the discrete argmax under noise.
    ``evidence``
        Peak minus median R2 across the library. Near zero means the data do not
        distinguish the HRFs at all and the selected index carries no
        information -- the map to threshold on before believing an HRF result.
    """
    if r2_per_hrf.ndim != 2:
        raise ValueError("r2_per_hrf must have shape (n_voxels, n_hrfs)")
    n_hrfs = r2_per_hrf.shape[1]
    index = r2_per_hrf.argmax(dim=1)
    evidence = r2_per_hrf.amax(dim=1) - r2_per_hrf.median(dim=1).values

    if n_hrfs < 3:
        return index, index.to(r2_per_hrf.dtype), evidence

    # Parabolic vertex through (i-1, i, i+1); interior peaks only, since a peak
    # at either end of the library is unbracketed and its offset is undefined.
    interior = index.clamp(1, n_hrfs - 2)
    rows = torch.arange(r2_per_hrf.shape[0], device=r2_per_hrf.device)
    left = r2_per_hrf[rows, interior - 1]
    center = r2_per_hrf[rows, interior]
    right = r2_per_hrf[rows, interior + 1]
    curvature = left - 2 * center + right
    offset = torch.where(
        curvature.abs() > torch.finfo(r2_per_hrf.dtype).eps,
        0.5 * (left - right) / curvature,
        torch.zeros_like(curvature),
    ).clamp(-1.0, 1.0)
    continuous = torch.where(
        (index == interior), interior.to(offset.dtype) + offset, index.to(offset.dtype)
    )
    return index, continuous, evidence


def grid_seeds_as_fit(grid_fit: PRFGridFit) -> PRFRefinedFit:
    """Present raw super-grid seeds as a refined fit (analyzePRF ``seedmode -2``).

    The grid search scores candidates by correlation, so ``r2`` here is ``r^2``
    of the best candidate rather than a coefficient of determination: it cannot
    go negative, and it is not comparable to a refined fit's ``r2``.
    """
    return PRFRefinedFit(
        candidate_index=grid_fit.candidate_index,
        hrf_index=grid_fit.hrf_index,
        parameters=grid_fit.parameters,
        gain=grid_fit.gain,
        correlation=grid_fit.correlation,
        r2=grid_fit.r2,
        residual_ss=torch.full_like(grid_fit.gain, torch.nan),
        n_iters=torch.zeros_like(grid_fit.hrf_index, dtype=torch.int32),
        converged=torch.zeros_like(grid_fit.hrf_index, dtype=torch.bool),
    )


def _predict_selected_prfs(
    stimulus_runs: Sequence[torch.Tensor],
    parameters: torch.Tensor,
    hrf_index: torch.Tensor,
    gain: torch.Tensor,
    hrf_library: torch.Tensor,
    stimulus_shape: tuple[int, int],
    voxel_chunk_size: int,
) -> torch.Tensor:
    """Predict one selected-HRF CSS response for each voxel."""
    n_timepoints = sum(run.shape[0] for run in stimulus_runs)
    prediction = torch.empty(
        (n_timepoints, parameters.shape[0]), device=parameters.device, dtype=parameters.dtype
    )
    for selected_hrf in hrf_index.unique(sorted=True).tolist():
        indices = torch.nonzero(hrf_index == selected_hrf, as_tuple=False).squeeze(1)
        for start in range(0, indices.numel(), voxel_chunk_size):
            chunk = indices[start : start + voxel_chunk_size]
            prediction[:, chunk] = (
                _css_prediction(
                    stimulus_runs, parameters[chunk], hrf_library[selected_hrf], stimulus_shape
                )
                * gain[chunk]
            )
    return prediction


def balanced_group_halves(
    group_labels: Sequence[int],
    *,
    n_draws: int = 2,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Split runs into two disjoint halves that each cover every stimulus group.

    Returns ``(half_a, half_b)`` run-index pairs, one per draw.

    Two constraints shape this, and they pull in opposite directions. Parameter
    reliability requires the two fits to share **no** training data, which caps
    each side at half the runs. A pRF fit is only valid if its training set
    contains every stimulus type -- bars and wedges are complementary probes of
    one receptive field, not competing models, so a fit that saw only bars is
    deficient rather than alternative. Together these force one run per group
    per side (``n_g // 2`` when a group has more), which is what this builds.

    Groups with a single run cannot appear on both sides and are dropped; the
    caller is expected to say so out loud, since their stimulus is then absent
    from the reliability estimate entirely.

    Odd-sized groups leave one run unused per draw, and which one it is rotates
    with the draw. That rotation is the point of ``n_draws``: a fixed choice
    would tie the estimate to whichever runs happen to sit out, and with
    retinotopy runs acquired in order those are systematically the late,
    motion-heavy ones. Draws that come out identical are dropped rather than
    refitted, so ``n_draws`` is a ceiling and not a promise.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    labels: list[int] = []
    for label in group_labels:
        if label not in labels:
            labels.append(label)

    draws: list[tuple[list[int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for draw in range(n_draws):
        generator = torch.Generator().manual_seed(seed + draw)
        first: list[int] = []
        second: list[int] = []
        for label in labels:
            members = [index for index, value in enumerate(group_labels) if value == label]
            if len(members) < 2:
                continue
            shuffled = [
                members[position]
                for position in torch.randperm(len(members), generator=generator).tolist()
            ]
            take = len(members) // 2
            first.extend(shuffled[:take])
            second.extend(shuffled[take : 2 * take])
        if not first or not second:
            continue
        first, second = sorted(first), sorted(second)
        # Swapping the halves gives the same pair of fits, so canonicalise on
        # which side owns the lowest run index before testing for duplicates.
        if second[0] < first[0]:
            first, second = second, first
        key = (*first, -1, *second)
        if key in seen:
            continue
        seen.add(key)
        draws.append((first, second))
    return draws


def prf_parameter_maps(
    parameters: torch.Tensor,
    stimulus_shape: tuple[int, int],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Derive the reported pRF quantities from raw ``(row, column, sigma, n)``.

    One place computes these, because the saved bucket and the reliability
    analysis have to agree on them exactly -- particularly the sign convention
    relating ``y`` to ``angle``, which is silently reversible.

    ``angle`` is in degrees to match the saved bucket; circular statistics on it
    need radians.
    """
    extent = float(max(stimulus_shape))
    center = (1.0 + extent) / 2.0
    exponent = parameters[:, 3].clamp_min(torch.finfo(parameters.dtype).eps)
    # Larger row index is the upper visual field for the aperture layouts these
    # tools are given, so vertical position is row - center, not center - row.
    vertical = parameters[:, 0] - center
    horizontal = parameters[:, 1] - center
    eccentricity = torch.hypot(vertical, horizontal)
    angle = torch.rad2deg(torch.atan2(vertical, horizontal)).remainder(360.0)
    return {
        "x": horizontal * scale,
        "y": vertical * scale,
        "sigma": parameters[:, 2].abs() * scale,
        "exponent": parameters[:, 3],
        "angle": torch.where(eccentricity == 0, torch.full_like(angle, torch.nan), angle),
        "eccentricity": eccentricity * scale,
        "rfsize": parameters[:, 2].abs() / torch.sqrt(exponent) * scale,
    }


def loro_folds(n_runs: int) -> list[tuple[list[int], list[int]]]:
    """Classic leave-one-run-out fold specification."""
    return [
        ([index for index in range(n_runs) if index != held_out], [held_out])
        for held_out in range(n_runs)
    ]


def fit_prf_folds(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid: PRFGrid,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    candidate_chunk_size: int = 256,
    voxel_chunk_size: int = 20_000,
    refine_chunk_size: int | None = None,
    device: torch.device | None = None,
    refinement_config: PRFRefinementConfig = PRFRefinementConfig(),
    hrf_mode: str = "grid",
    fixed_hrf_index: torch.Tensor | None = None,
    refine: bool = True,
    return_fits: bool = False,
    verbose: bool = False,
) -> tuple[PRFCrossValidationFit, list[PRFRefinedFit]]:
    """Cross-validate with fold-local pRF fitting over explicit run partitions.

    Each fold is ``(training_runs, held_out_runs)``; every fold performs the
    complete grid-selection and GN-refinement pipeline on its training runs, so
    the CSS parameters and gain are never informed by the held-out data. Folds
    need not hold out a single run, and need not be exhaustive -- what makes the
    pooled R2 a whole-dataset number is each timepoint being held out once, and
    that is the caller's business to arrange.

    ``return_fits`` additionally hands back each fold's fitted parameters. That
    is what makes split-half reliability free: two folds with disjoint training
    sets are, between them, a cross-validation and a reliability estimate.

    ``hrf_mode`` decides how much of the fold-locality applies to the HRF, which
    is the one place where cost and rigour genuinely trade off:

    ``"grid"``
        The fold's own super-grid picks the HRF. Leak-free, and the default, but
        grid-resolution HRF selection is barely better than chance, so the
        held-out R2 scores a different (worse) model than a refine-selected fit
        reports -- read it as a conservative lower bound.
    ``"fixed"``
        Reuse ``fixed_hrf_index``, normally the full fit's per-voxel choice.
        Same cost, and it scores the model actually reported, but the HRF was
        chosen using every run including the held-out one: one discrete choice
        out of the library leaks, biasing R2 optimistic.
    ``"refine"``
        Refine under every HRF inside each fold. Leak-free *and* consistent, at
        roughly ``n_hrfs`` times the cost.
    """
    if not refine and hrf_mode == "refine":
        raise ValueError("hrf_mode='refine' has no meaning when refine=False")
    if hrf_mode not in ("grid", "fixed", "refine"):
        raise ValueError(f"hrf_mode must be 'grid', 'fixed', or 'refine', got {hrf_mode!r}")
    if hrf_mode == "fixed" and fixed_hrf_index is None:
        raise ValueError("hrf_mode='fixed' requires fixed_hrf_index")
    if not folds:
        raise ValueError("at least one fold is required")
    if data.shape[1] != sum(run.shape[0] for run in stimulus_runs):
        raise ValueError("stimulus runs must span every data timepoint")
    for training_indices, held_indices in folds:
        if not training_indices or not held_indices:
            raise ValueError("every fold needs at least one training and one held-out run")
        if set(training_indices) & set(held_indices):
            raise ValueError("a fold cannot both train on and hold out the same run")

    device = device or data.device
    output_device = data.device
    n_voxels = data.shape[0]
    run_ends = [*run_starts[1:], data.shape[1]]
    residual_ss = torch.zeros(n_voxels, device=output_device, dtype=data.dtype)
    total_ss = torch.zeros(n_voxels, device=output_device, dtype=data.dtype)
    fold_fits: list[PRFRefinedFit] = []
    fold_iterator = list(folds)
    if verbose:
        from tqdm.auto import tqdm

        fold_iterator = tqdm(fold_iterator, desc="pRF CV folds", leave=True)

    for training_indices, held_indices in fold_iterator:
        training_indices = list(training_indices)
        held_indices = list(held_indices)
        training_data = torch.cat(
            [data[:, run_starts[index] : run_ends[index]] for index in training_indices], dim=1
        )
        training_stimuli = [stimulus_runs[index] for index in training_indices]
        training_starts = [
            sum(run.shape[0] for run in training_stimuli[:index])
            for index in range(len(training_stimuli))
        ]
        training_nuisance = (
            [nuisance_per_run[index] for index in training_indices]
            if nuisance_per_run is not None
            else None
        )
        training_grid = fit_prf_supergrid(
            training_data,
            training_stimuli,
            stimulus_shape,
            grid,
            hrf_library,
            training_starts,
            nuisance_per_run=training_nuisance,
            candidate_chunk_size=candidate_chunk_size,
            voxel_chunk_size=voxel_chunk_size,
            device=device,
            track_per_hrf=refine and hrf_mode in ("refine", "fixed") and hrf_library.shape[0] > 1,
        )
        if hrf_mode == "fixed":
            assert fixed_hrf_index is not None
            # Seed at the imposed HRF's own grid optimum, not the grid winner's:
            # position and HRF delay trade off, so the winner's seed would start
            # refinement systematically off-target for every other HRF.
            training_grid = training_grid.seeds_for_hrf(
                fixed_hrf_index.to(training_grid.hrf_index.device)
            )
        refine_kwargs = dict(
            nuisance_per_run=training_nuisance,
            voxel_chunk_size=refine_chunk_size or voxel_chunk_size,
            device=device,
            config=refinement_config,
        )
        if not refine:
            # Seeds only. The super-grid is ~1 s where refinement is minutes, and
            # for comparing nuisance models against each other -- which noise
            # regressors are worth keeping -- the seeds move together with the
            # refined fit. Not a substitute for the reported fit.
            training_fit = grid_seeds_as_fit(training_grid)
        elif hrf_mode == "refine" and hrf_library.shape[0] > 1:
            training_fit, _, _ = refine_prf_all_hrfs(
                training_data,
                training_stimuli,
                stimulus_shape,
                training_grid,
                hrf_library,
                training_starts,
                **refine_kwargs,
            )
        else:
            training_fit = refine_prf_supergrid(
                training_data,
                training_stimuli,
                stimulus_shape,
                training_grid,
                hrf_library,
                training_starts,
                **refine_kwargs,
            )

        if return_fits:
            fold_fits.append(training_fit)

        held_stimuli = [
            stimulus_runs[index].to(device=device, dtype=data.dtype) for index in held_indices
        ]
        held_lengths = [run.shape[0] for run in held_stimuli]
        held_starts = [sum(held_lengths[:index]) for index in range(len(held_lengths))]
        held_nuisance = (
            [nuisance_per_run[index] for index in held_indices]
            if nuisance_per_run is not None
            else None
        )
        held_data = torch.cat(
            [data[:, run_starts[index] : run_ends[index]] for index in held_indices], dim=1
        ).T.to(device=device, dtype=data.dtype)
        held_data = _project_per_run(held_data, held_nuisance, held_starts)
        held_prediction = _predict_selected_prfs(
            held_stimuli,
            training_fit.parameters.to(device=device, dtype=data.dtype),
            training_fit.hrf_index.to(device=device),
            training_fit.gain.to(device=device, dtype=data.dtype),
            hrf_library.to(device=device, dtype=data.dtype),
            stimulus_shape,
            refine_chunk_size or voxel_chunk_size,
        )
        held_prediction = _project_per_run(held_prediction, held_nuisance, held_starts)
        residual_ss += (held_data - held_prediction).square().sum(dim=0).to(output_device)
        # Centred per held-out run, not over the concatenation: between-run mean
        # offsets are not variance the model was ever asked to explain, and
        # folding them into the denominator quietly inflates R2. With a polort
        # that includes the constant this is a no-op, but -polort -1 is legal.
        for start, length in zip(held_starts, held_lengths, strict=True):
            segment = held_data[start : start + length]
            total_ss += (
                (segment - segment.mean(dim=0, keepdim=True)).square().sum(dim=0).to(output_device)
            )

    r2 = 1.0 - residual_ss / total_ss.clamp_min(torch.finfo(data.dtype).eps)
    return PRFCrossValidationFit(r2=r2, residual_ss=residual_ss, total_ss=total_ss), fold_fits


def fit_prf_loro(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid: PRFGrid,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    **kwargs,
) -> PRFCrossValidationFit:
    """Leave-one-run-out cross-validation: :func:`fit_prf_folds` over single runs."""
    if len(stimulus_runs) < 2:
        raise ValueError("LORO pRF fitting requires at least two runs")
    cross_validation, _ = fit_prf_folds(
        data,
        stimulus_runs,
        stimulus_shape,
        grid,
        hrf_library,
        run_starts,
        loro_folds(len(stimulus_runs)),
        **kwargs,
    )
    return cross_validation


def fit_prf_supergrid(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid: PRFGrid,
    hrf_library: torch.Tensor,
    run_starts: Sequence[int],
    *,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    candidate_chunk_size: int = 256,
    voxel_chunk_size: int = 20_000,
    device: torch.device | None = None,
    track_per_hrf: bool = False,
    verbose: bool = False,
) -> PRFGridFit:
    """Fit the best CSS super-grid and HRF candidate per voxel.

    ``data`` is voxel-major while stimulus runs are time-major.  Candidate
    predictions are generated on the compute device, while result maps remain
    on the data device.  This permits CPU-backed data with CUDA voxel streaming.

    ``track_per_hrf`` additionally keeps each voxel's best candidate *under every
    HRF*, not just the winner, in ``correlation_per_hrf`` / ``candidate_per_hrf``.
    Two (n_voxels, n_hrfs) buffers and one masked update per chunk buys both the
    ability to rank HRFs by fit rather than by library index, and an unbiased seed
    for each HRF's refinement -- see :meth:`PRFGridFit.seeds_for_hrf`.
    """
    if data.ndim != 2:
        raise ValueError("data must have shape (n_voxels, n_timepoints)")
    if hrf_library.ndim != 2:
        raise ValueError("hrf_library must have shape (n_hrfs, n_hrf_samples)")
    if candidate_chunk_size < 1 or voxel_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    if not stimulus_runs:
        raise ValueError("at least one stimulus run is required")

    device = device or data.device
    n_voxels, n_timepoints = data.shape
    run_lengths = [run.shape[0] for run in stimulus_runs]
    if sum(run_lengths) != n_timepoints:
        raise ValueError("stimulus frames across runs must equal the data timepoints")
    expected_starts = [sum(run_lengths[:index]) for index in range(len(run_lengths))]
    if list(run_starts) != expected_starts:
        raise ValueError(f"run_starts must be {expected_starts}, got {list(run_starts)}")

    output_device = data.device
    best_correlation = torch.full((n_voxels,), -torch.inf, device=output_device, dtype=data.dtype)
    best_candidate = torch.zeros(n_voxels, device=output_device, dtype=torch.long)
    best_hrf = torch.zeros(n_voxels, device=output_device, dtype=torch.long)
    best_gain = torch.zeros(n_voxels, device=output_device, dtype=data.dtype)
    hrf_correlation = None
    hrf_candidate = None
    if track_per_hrf:
        hrf_correlation = torch.full(
            (n_voxels, hrf_library.shape[0]), -torch.inf, device=output_device, dtype=data.dtype
        )
        # int32 halves the buffer against long, and no super-grid comes close to
        # 2^31 candidates.
        hrf_candidate = torch.zeros(
            (n_voxels, hrf_library.shape[0]), device=output_device, dtype=torch.int32
        )

    parameters = grid.parameters.to(device=device, dtype=data.dtype)
    hrf_library = hrf_library.to(device=device, dtype=data.dtype)
    stimuli = [run.to(device=device, dtype=data.dtype) for run in stimulus_runs]

    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    if any(run.shape[1] != n_pixels for run in stimulus_runs):
        raise ValueError(f"stimulus runs must each contain {n_pixels} flattened pixels")

    eps = torch.finfo(data.dtype).eps
    # Project and unit-normalize the data exactly once. Doing it inside the
    # candidate loop repeats an O(voxels x time x nuisance) QR projection for
    # every candidate chunk and HRF, which dominates the grid search once the
    # library has more than a couple of HRFs.
    normalized_data = torch.empty_like(data)
    data_norm = torch.empty(n_voxels, device=output_device, dtype=data.dtype)
    for voxel_start in range(0, n_voxels, voxel_chunk_size):
        voxel_end = min(voxel_start + voxel_chunk_size, n_voxels)
        projected = _project_per_run(
            data[voxel_start:voxel_end].T.to(device=device, dtype=data.dtype),
            nuisance_per_run,
            run_starts,
        )
        chunk_norm = torch.linalg.vector_norm(projected, dim=0).clamp_min(eps)
        normalized_data[voxel_start:voxel_end] = (projected / chunk_norm).T.to(output_device)
        data_norm[voxel_start:voxel_end] = chunk_norm.to(output_device)

    n_candidate_chunks = (grid.n_candidates + candidate_chunk_size - 1) // candidate_chunk_size
    candidate_starts = range(0, grid.n_candidates, candidate_chunk_size)
    if verbose:
        from tqdm.auto import tqdm

        candidate_starts = tqdm(
            candidate_starts,
            total=n_candidate_chunks,
            desc="pRF super-grid",
            leave=True,
            disable=n_candidate_chunks < 2,
        )

    for candidate_start in candidate_starts:
        candidate_end = min(candidate_start + candidate_chunk_size, grid.n_candidates)
        candidate_parameters = parameters[candidate_start:candidate_end]
        fields = gaussian_receptive_fields(candidate_parameters, stimulus_shape)
        exponents = candidate_parameters[:, 3]

        for hrf_index, hrf in enumerate(hrf_library):
            predictions = _project_per_run(
                predict_prf_runwise(stimuli, fields, exponents, hrf), nuisance_per_run, run_starts
            )
            prediction_norm = torch.linalg.vector_norm(predictions, dim=0).clamp_min(eps)
            normalized_predictions = predictions / prediction_norm
            del predictions

            for voxel_start in range(0, n_voxels, voxel_chunk_size):
                voxel_end = min(voxel_start + voxel_chunk_size, n_voxels)
                output_slice = slice(voxel_start, voxel_end)
                chunk = normalized_data[output_slice].to(device=device, dtype=data.dtype)
                correlations = chunk @ normalized_predictions
                candidate_scores, candidate_offsets = correlations.max(dim=1)
                # analyzePRF's identity: the least-squares gain of the winning
                # candidate is r * |data| / |prediction|, so the selected
                # predictions never have to be gathered into a (time, voxel) block.
                gains = (
                    candidate_scores
                    * data_norm[output_slice].to(device)
                    / prediction_norm[candidate_offsets]
                ).clamp_min(0)

                if hrf_correlation is not None and hrf_candidate is not None:
                    scores_out = candidate_scores.to(output_device)
                    better = scores_out > hrf_correlation[output_slice, hrf_index]
                    hrf_correlation[output_slice, hrf_index] = torch.where(
                        better, scores_out, hrf_correlation[output_slice, hrf_index]
                    )
                    hrf_candidate[output_slice, hrf_index] = torch.where(
                        better,
                        (candidate_start + candidate_offsets).to(output_device, torch.int32),
                        hrf_candidate[output_slice, hrf_index],
                    )

                previous_scores = best_correlation[output_slice].to(device)
                improved = candidate_scores > previous_scores
                if improved.any():
                    improved_out = improved.to(output_device)
                    best_correlation[output_slice] = torch.where(
                        improved_out,
                        candidate_scores.to(output_device),
                        best_correlation[output_slice],
                    )
                    best_candidate[output_slice] = torch.where(
                        improved_out,
                        (candidate_start + candidate_offsets).to(output_device),
                        best_candidate[output_slice],
                    )
                    best_hrf[output_slice] = torch.where(
                        improved_out,
                        torch.full_like(candidate_offsets, hrf_index).to(output_device),
                        best_hrf[output_slice],
                    )
                    best_gain[output_slice] = torch.where(
                        improved_out, gains.to(output_device), best_gain[output_slice]
                    )

    best_parameters = grid.parameters.to(output_device, dtype=data.dtype)[best_candidate]
    return PRFGridFit(
        candidate_index=best_candidate,
        hrf_index=best_hrf,
        parameters=best_parameters,
        gain=best_gain,
        correlation=best_correlation,
        r2=best_correlation.square(),
        correlation_per_hrf=hrf_correlation,
        candidate_per_hrf=hrf_candidate,
        grid_parameters=grid.parameters.to(output_device, dtype=data.dtype)
        if track_per_hrf
        else None,
    )


def hashed_gaussian_tiles(
    stimulus_shape: tuple[int, int],
    *,
    n_tiles: int = 250,
    gaussians_per_tile: int = 5,
    sigma_fraction: tuple[float, float] = (0.05, 0.25),
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a random ``(n_pixels, n_tiles)`` basis of hashed Gaussian tiles.

    Each tile is the sum of ``gaussians_per_tile`` randomly placed isotropic
    Gaussians, normalised to unit volume. Overlapping several Gaussians per tile
    is the "hashing" step: it lets a few hundred tiles cover the visual field
    densely enough to localise a pRF, where a few hundred *single* Gaussians
    would leave gaps.

    This is the stimulus encoding from Bhat, Luhrs, Goebel & Senden (2021),
    'Extremely fast pRF mapping for real-time applications', NeuroImage 245:118671
    (Computational Neuroimaging Toolbox, https://github.com/ccnmaastricht/CNI_toolbox).
    The basis is deliberately random rather than a grid: it is used with ridge
    regression, where what matters is that the tiles span the space, not that any
    one of them matches a pRF.
    """
    if n_tiles < 1 or gaussians_per_tile < 1:
        raise ValueError("n_tiles and gaussians_per_tile must be positive")
    if not 0 < sigma_fraction[0] <= sigma_fraction[1]:
        raise ValueError(
            f"sigma_fraction must be an increasing positive pair, got {sigma_fraction}"
        )

    rows, columns = stimulus_shape
    extent = float(max(rows, columns))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_gaussians = n_tiles * gaussians_per_tile
    centers_row = torch.rand(n_gaussians, generator=generator) * rows
    centers_column = torch.rand(n_gaussians, generator=generator) * columns
    low, high = sigma_fraction
    sigmas = (torch.rand(n_gaussians, generator=generator) * (high - low) + low) * extent

    parameters = torch.stack(
        [
            centers_row.to(device=device, dtype=dtype),
            centers_column.to(device=device, dtype=dtype),
            sigmas.to(device=device, dtype=dtype),
            torch.ones(n_gaussians, device=device, dtype=dtype),
        ],
        dim=1,
    )
    fields = gaussian_receptive_fields(parameters, stimulus_shape)
    tiles = fields.view(n_tiles, gaussians_per_tile, -1).sum(dim=1)
    return (tiles / tiles.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)).T.contiguous()


def _screening_folds(
    run_starts: Sequence[int], n_timepoints: int, n_blocks: int
) -> list[torch.Tensor]:
    """Cross-validation folds for screening: whole runs, or time blocks within one run."""
    if len(run_starts) > 1:
        ends = [*run_starts[1:], n_timepoints]
        return [torch.arange(start, end) for start, end in zip(run_starts, ends, strict=True)]
    edges = torch.linspace(0, n_timepoints, n_blocks + 1).round().long().tolist()
    return [torch.arange(start, end) for start, end in zip(edges[:-1], edges[1:], strict=True)]


def screen_voxels_ridge(
    data: torch.Tensor,
    stimulus_runs: Sequence[torch.Tensor],
    stimulus_shape: tuple[int, int],
    hrf: torch.Tensor,
    run_starts: Sequence[int],
    *,
    nuisance_per_run: Sequence[torch.Tensor] | None = None,
    n_tiles: int = 250,
    gaussians_per_tile: int = 5,
    ridge: float = 1.0,
    n_blocks: int = 4,
    voxel_chunk_size: int = 50_000,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Cross-validated R2 of a fast *linear* pRF model, for every voxel.

    A screening statistic, not a fit. The CSS model is nonlinear and costs a
    Gauss-Newton refinement per voxel; this replaces the pRF with a ridge
    regression onto a random hashed-Gaussian basis, which is one small linear
    solve shared by every voxel and runs in seconds over a whole brain. Voxels
    that no linear combination of tiles can predict have no pRF worth refining,
    so thresholding this is a *functionally* derived mask -- far more specific
    than an anatomical or intensity mask, and the cheapest large win available
    on whole-brain data, where refinement cost is simply linear in voxels.

    Held out by run when there is more than one, otherwise by contiguous time
    blocks. Nuisance regressors are projected out fold-locally, so drift cannot
    leak into the held-out score (see the LORO principle).

    Method from Bhat et al. (2021), NeuroImage 245:118671. Their own use is the
    same: fit everything linearly, keep the top voxels, spend the real fitting
    effort there.
    """
    device = device or data.device
    output_device = data.device
    n_voxels, n_timepoints = data.shape
    tiles = hashed_gaussian_tiles(
        stimulus_shape,
        n_tiles=n_tiles,
        gaussians_per_tile=gaussians_per_tile,
        seed=seed,
        device=device,
        dtype=data.dtype,
    )
    hrf = hrf.to(device=device, dtype=data.dtype)
    design = torch.cat(
        [
            _hrf_convolve_columns(run.to(device=device, dtype=data.dtype) @ tiles, hrf)
            for run in stimulus_runs
        ],
        dim=0,
    )

    folds = _screening_folds(run_starts, n_timepoints, n_blocks)
    eps = torch.finfo(data.dtype).eps
    residual_ss = torch.zeros(n_voxels, device=output_device, dtype=data.dtype)
    total_ss = torch.zeros(n_voxels, device=output_device, dtype=data.dtype)
    fold_iterator = folds
    if verbose:
        from tqdm.auto import tqdm

        fold_iterator = tqdm(folds, desc="pRF screening", leave=True, disable=len(folds) < 2)

    for test_index in fold_iterator:
        train_mask = torch.ones(n_timepoints, dtype=torch.bool)
        train_mask[test_index] = False
        train_index = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
        test_index = test_index.to(device)

        # Fold-local nuisance: the same block-diagonal design, restricted to the
        # fold's timepoints, so drift is removed without seeing the held-out data.
        train_design = design[train_index]
        test_design = design[test_index]
        if nuisance_per_run is not None:
            nuisance = torch.block_diag(*[block.to(device) for block in nuisance_per_run])
            train_design = orthogonalize_design(train_design, nuisance[train_index])
            test_design = orthogonalize_design(test_design, nuisance[test_index])

        gram = train_design.T @ train_design
        penalty = ridge * gram.diagonal().mean().clamp_min(eps)
        normal = gram + torch.eye(n_tiles, device=device, dtype=gram.dtype) * penalty
        factor = torch.linalg.cholesky(normal.double())

        for start in range(0, n_voxels, voxel_chunk_size):
            stop = min(start + voxel_chunk_size, n_voxels)
            chunk = data[start:stop].to(device=device, dtype=data.dtype)
            train_data = chunk[:, train_index].T
            test_data = chunk[:, test_index].T
            if nuisance_per_run is not None:
                train_data = orthogonalize_design(train_data, nuisance[train_index])
                test_data = orthogonalize_design(test_data, nuisance[test_index])
            weights = torch.cholesky_solve((train_design.T @ train_data).double(), factor).to(
                data.dtype
            )
            prediction = test_design @ weights
            residual_ss[start:stop] += (
                (test_data - prediction).square().sum(dim=0).to(output_device)
            )
            centered = test_data - test_data.mean(dim=0, keepdim=True)
            total_ss[start:stop] += centered.square().sum(dim=0).to(output_device)
            del chunk, train_data, test_data, weights, prediction, centered

    # A voxel with no variance to explain (all-zero, or constant) would score a
    # perfect 1.0 from 0/0 and sail to the top of any ranking. This is the same
    # trap that made constant voxels win PC selection in ffs_denoisatorial.
    return torch.where(
        total_ss > eps,
        1.0 - residual_ss / total_ss.clamp_min(eps),
        torch.full_like(total_ss, -torch.inf),
    )


def select_noise_pc_count(
    median_r2_by_count: Sequence[float],
    tolerance: float = 0.05,
) -> int:
    """Choose how many noise PCs to keep, GLMdenoise's conservative rule.

    ``median_r2_by_count[n]`` is the cross-validated median R2 over responsive
    voxels when ``n`` components are projected out, so entry 0 is the undenoised
    model. Rather than the argmax -- that curve is noisy, and its peak sits at
    more components than the data supports -- take the FEWEST components that
    reach within ``tolerance`` of the best improvement over the undenoised fit.

    Returns 0 when no count improves on the undenoised model, which is the
    answer whenever denoising is not worth its degrees of freedom. That case is
    the point of running the sweep: a design with strong stimulus-locked signal
    can genuinely want zero components, and taking the argmax would hide it.

    Kay, Rokem, Winawer, Dougherty & Wandell (2013), 'GLMdenoise: a fast,
    automated technique for denoising task-based fMRI data', Front Neurosci 7:247.
    """
    if not median_r2_by_count:
        raise ValueError("median_r2_by_count must not be empty")
    if not 0 <= tolerance < 1:
        raise ValueError(f"tolerance must be in [0, 1), got {tolerance}")

    baseline = median_r2_by_count[0]
    improvements = [value - baseline for value in median_r2_by_count]
    best = max(improvements)
    if best <= 0:
        return 0
    target = (1.0 - tolerance) * best
    return next(count for count, gain in enumerate(improvements) if gain >= target)
