"""Research-only continuous cardiac phase fit; no recordings or waveform template.

The initial phase supplies the alias and basin. Variable projection learns each
voxel's sine/cosine coefficients while a shared cubic spline learns phase at the
actual acquisition times. This is deliberately outside the production pipeline.
Wiki rationale and limits: ``Continuous cardiac phase refinement``.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def select_voxels(series, phase, fraction=0.01):
    """Select/scale training voxels using only their fit to the initial phase."""
    selected, conditioning = [], []
    for y, p in zip(series, phase.T, strict=True):
        y = np.asarray(y, dtype=np.float64)
        x = np.column_stack((np.ones(len(p)), np.cos(p), np.sin(p)))
        beta = np.linalg.lstsq(x, y.T, rcond=None)[0].T
        residual = np.mean((y - beta @ x.T) ** 2, axis=1)
        power = np.sum(beta[:, 1:] ** 2, axis=1)
        k = min(len(y), max(10, int(np.ceil(fraction * len(y)))))
        keep = np.argsort(power / np.maximum(residual, 1e-20))[-k:]
        selected.append(y[keep] / np.sqrt(np.maximum(residual[keep, None], 1e-20)))
        b = beta[keep, 1:] / np.sqrt(np.maximum(residual[keep, None], 1e-20))
        conditioning.append(float(np.linalg.cond(b.T @ b)))
    return selected, conditioning


def fit_phase(series, times, initial, knot_s=4.0, maxiter=150):
    """Fit one smooth phase across groups, allowing free voxel amplitudes/delays.

    No cardiac acceptance decision is made here. Knot spacing controls flexibility
    and must be assessed on independent observations, not recording agreement.
    """
    times = np.asarray(times, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    knots = np.linspace(times.min(), times.max(), max(4, int(np.ceil(np.ptp(times) / knot_s)) + 1))
    basis = CubicSpline(knots, np.eye(len(knots)), bc_type="natural")(times)
    carrier = 2 * np.pi * np.polyfit(times.ravel(), initial.ravel() / (2 * np.pi), 1)[0] * times
    flat_basis = basis.reshape(-1, len(knots))
    # Crossed tracks have independent 2*pi origins. Project a common lift of the
    # demodulated phase, otherwise averaging those origins can invent half-cycles.
    residual_phase = np.angle(np.exp(1j * (initial - carrier)))
    centre = np.unwrap(np.angle(np.exp(1j * residual_phase).mean(axis=1)))
    lifted = centre[:, None] + np.angle(np.exp(1j * (residual_phase - centre[:, None])))
    start = np.linalg.lstsq(flat_basis, lifted.ravel(), rcond=None)[0]
    if maxiter == 0:
        return carrier + np.einsum("ngk,k->ng", basis, start), dict(
            success=True, iterations=0, loss=None, message="Initial phase smoothed only"
        )
    ys = [np.asarray(y, dtype=np.float64) for y in series]
    normalizer = sum(y.size for y in ys)

    def objective(c):
        phase = carrier + np.einsum("ngk,k->ng", basis, c)
        loss, gradient = 0.0, np.zeros_like(c)
        for g, y in enumerate(ys):
            p = phase[:, g]
            co, si = np.cos(p), np.sin(p)
            x = np.column_stack((np.ones(len(p)), co, si))
            beta = np.linalg.lstsq(x, y.T, rcond=None)[0].T
            residual = beta @ x.T - y
            derivative = -beta[:, 1, None] * si + beta[:, 2, None] * co
            loss += np.sum(residual**2)
            gradient += basis[:, g].T @ (2 * np.sum(residual * derivative, axis=0))
        return loss / normalizer, gradient / normalizer

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    phase = carrier + np.einsum("ngk,k->ng", basis, result.x)
    return phase, dict(
        success=bool(result.success),
        iterations=int(result.nit),
        loss=float(result.fun),
        message=str(result.message),
    )


def prediction_score(series, phase, tr, block_s=20.0):
    """Temporal cross-prediction with coefficients learned on alternate blocks.

    To test phase generalisation, series must also be spatially excluded from phase
    refinement. All voxels are scored; no selection using the candidate being scored.
    """
    nt = len(phase)
    folds = (np.arange(nt) * tr // block_s).astype(int) % 2
    improvement, total = 0.0, 0.0
    for y, p in zip(series, phase.T, strict=True):
        y = np.asarray(y, dtype=np.float64)
        x = np.column_stack((np.ones(nt), np.cos(p), np.sin(p)))
        for fold in (0, 1):
            train, test = folds != fold, folds == fold
            beta = np.linalg.lstsq(x[train], y[:, train].T, rcond=None)[0].T
            base = np.sum((y[:, test] - y[:, train].mean(axis=1, keepdims=True)) ** 2)
            rss = np.sum((y[:, test] - beta @ x[test].T) ** 2)
            improvement += base - rss
            total += base
    return float(improvement / total)
