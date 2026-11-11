"""CPU demonstration: fit TR-spaced TENT estimates in their estimator's space.

This is a controlled, noise-free isolated-response superposition experiment,
not a full DCM calibration or a model of nonlinear event-train adaptation.
Run from the repository root; JSON lines report reproducible measurements.
"""

import json

import numpy as np
import torch

from fastfuncstuff.design.matrices import make_tent_design
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.integrate import build_input, integrate
from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.pipeline import reference_p0


def main():
    torch.set_num_threads(1)
    tr, dt, top, n_scans = 2.0, 0.05, 32.0, 600
    fine_times = np.arange(round(top / dt) + 1) * dt
    knots = np.arange(0, top + tr / 2, tr)
    spec = ModelSpec(N=3, K=7, n_inputs=2, n_mod=1, TR=dt, dt=dt, p0=reference_p0())
    p = faes_priors(spec, (1, 1, 1)).pE
    p["C"] = torch.tensor([[0.0, 1.0], [0.0, 1.4], [0.0, 0.9]], dtype=torch.float64)
    u = build_input(spec, [[0.0], [0.0]], [[1.0], [1.0]], len(fine_times))
    sigma_grid = np.linspace(-0.5, 0.5, 41)
    responses = []
    for sigma in sigma_grid:
        p["sigma"] = torch.tensor(sigma, dtype=torch.float64)
        responses.append(integrate(u, p, spec, len(fine_times), delay_seconds=0.0).numpy())
    responses = np.stack(responses)
    truth = responses[len(sigma_grid) // 2]
    native = responses[:, np.rint(knots / dt).astype(int)]
    rng = np.random.default_rng(123)
    base = np.cumsum(rng.integers(4, 8, size=95)) * tr
    phases = {
        "locked": np.zeros(len(base)),
        "all_half": np.full(len(base), 0.5),
        "balanced": rng.choice([0.0, 0.5], len(base)),
        "95_percent_half": np.where(np.arange(len(base)) % 20 == 0, 0.0, 0.5),
        "quarters": rng.choice([0.0, 0.25, 0.5, 0.75], len(base)),
        "arbitrary": rng.integers(0, round(tr / dt), len(base)) * dt / tr,
    }
    nuisance = np.polynomial.legendre.legvander(np.linspace(-1, 1, n_scans), 2)
    q, _ = np.linalg.qr(nuisance)
    penalty = np.diff(np.eye(len(knots)), 2, axis=0)
    penalty = penalty.T @ penalty
    for name, phase in phases.items():
        onsets = base + tr * phase
        x = (
            make_tent_design(
                [onsets],
                bot=0,
                top=top,
                tr=tr,
                n_timepoints=n_scans,
                n_basis=len(knots),
                device=torch.device("cpu"),
            )
            .double()
            .numpy()
        )
        # Dense-grid interpolation maps ANY fine response to the acquired series;
        # the production TENT design is used only for the subsequent estimator.
        e = np.zeros((n_scans, len(fine_times)))
        for onset in onsets:
            lag = np.arange(n_scans) * tr - onset
            valid = (lag >= 0) & (lag <= top)
            e[np.flatnonzero(valid), np.rint(lag[valid] / dt).astype(int)] += 1
        xr, er = x - q @ (q.T @ x), e - q @ (q.T @ e)
        gram = xr.T @ xr
        scale = np.trace(gram) / np.trace(penalty)
        for lam in (0.0, 1.0):
            solve = (
                np.linalg.pinv(xr, rcond=1e-5)
                if lam == 0
                else np.linalg.solve(gram + lam * scale * penalty, xr.T)
            )
            op = solve @ er
            estimated = op @ truth
            naive_error = np.mean((native - estimated) ** 2, axis=(1, 2))
            aware = np.einsum("kt,ptd->pkd", op, responses)
            aware_error = np.mean((aware - estimated) ** 2, axis=(1, 2))
            sigma_naive = float(sigma_grid[naive_error.argmin()])
            sigma_aware = float(sigma_grid[aware_error.argmin()])
            cov = solve @ solve.T
            cor = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
            record = {
                "timing": name,
                "lambda": lam,
                "knot_spacing_seconds": tr,
                "design_rank": int(np.linalg.matrix_rank(xr, rtol=1e-5)),
                "n_knots": len(knots),
                "direct_sample_rmse_at_truth": float(np.sqrt(naive_error[20])),
                "operator_rmse_at_truth": float(np.sqrt(aware_error[20])),
                "sigma_true_hz": 3.0,
                "sigma_naive_hz": float(3 * np.exp(sigma_naive)),
                "sigma_aware_hz": float(3 * np.exp(sigma_aware)),
                "max_abs_offdiag_knot_noise_correlation": float(
                    np.abs(cor - np.eye(len(knots))).max()
                ),
            }
            assert sigma_aware == 0.0
            assert record["operator_rmse_at_truth"] < 1e-12
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
