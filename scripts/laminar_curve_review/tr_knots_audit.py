"""CPU audit of TR-spaced response estimates, independent analytic truth and AFNI.

Run from the repository root with:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python \
        python scripts/laminar_curve_review/tr_knots_audit.py --output /tmp/tr_knots_audit.json
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import block_diag, cholesky, solve_triangular, toeplitz
from scipy.stats import gamma

from fastfuncstuff.design.matrices import make_tent_design
from fastfuncstuff.glm.smooth_basis import (
    choose_global_lambda,
    fit_smooth_basis,
    roughness_penalty,
)

CPU = torch.device("cpu")
TR = 1.857
NTP = 240
NRUN = 3
K = 9
NDRAW = 128


def reference_design(onsets, bot, top, n=NTP):
    """Independent float64 implementation of AFNI's half-tent edge convention."""
    lag = np.arange(n)[:, None] * TR - onsets[None, :]
    dt = (top - bot) / (K - 1)
    out = np.zeros((n, K))
    for j, mid in enumerate(np.linspace(bot, top, K)):
        lo = mid - dt if j else bot - 0.00111 * dt
        hi = mid + dt if j < K - 1 else top + 0.00111 * dt
        out[:, j] = np.maximum(0, np.minimum((lag - lo) / (mid - lo), (hi - lag) / (hi - mid))).sum(
            1
        )
    return out


def hrf(t):
    """Depth-dependent analytic curves, independent of any fitted basis."""
    scale = np.array([1.1, 1.0, 0.9])
    amplitude = np.array([1.5, 1.0, 0.65])
    values = (
        gamma.pdf(t[..., None], a=6, scale=scale) - gamma.pdf(t[..., None], a=16, scale=scale) / 6
    )
    taper = 0.5 * (1 + np.cos(np.pi * np.clip((t - 7 * TR) / TR, 0, 1)))
    return values * amplitude / gamma.pdf(5 * scale, a=6, scale=scale) * taper[..., None]


def audit_case(name, phase, bot, noise_sd=0.15):
    rng = np.random.default_rng(590)
    onsets, designs, true_series = [], [], []
    for _ in range(NRUN):
        indices = np.sort(rng.choice(np.arange(5, NTP - 15), 32, replace=False))
        p = phase(rng, len(indices))
        o = (indices + p) * TR
        onsets.append(o)
        designs.append(make_tent_design([o], bot, bot + 8 * TR, TR, NTP, device=CPU).numpy())
        true_series.append(hrf(np.arange(NTP)[:, None] * TR - o[None, :]).sum(1))
    x = np.concatenate(designs)
    nuisance = block_diag(*[np.ones((NTP, 1)) for _ in range(NRUN)])
    m = np.eye(NTP * NRUN) - nuisance @ np.linalg.pinv(nuisance)
    xm = m @ x
    u, s, vh = np.linalg.svd(xm, full_matrices=False)
    rank = int((s > s[0] * 1e-6).sum())
    truth_knots = hrf(np.linspace(bot, bot + 8 * TR, K)).T
    signal = np.concatenate(true_series)
    y = signal.T[:, None, :] + noise_sd * rng.standard_normal((3, NDRAW, NTP * NRUN))
    y = y.reshape(-1, NTP * NRUN)
    packed = torch.tensor(np.column_stack([x, nuisance]), dtype=torch.float32)
    data = torch.tensor(y, dtype=torch.float32)
    penalty = roughness_penalty([K])
    selected = choose_global_lambda(data, packed, K, penalty, [0, NTP, 2 * NTP], device=CPU)
    ols = (np.linalg.pinv(xm, rcond=1e-6) @ (y @ m).T).T.reshape(3, NDRAW, K)
    fixed = (
        fit_smooth_basis(data, packed, K, penalty, method="fixed", lam=selected.lam, device=CPU)
        .betas.numpy()
        .reshape(3, NDRAW, K)
    )
    reml = (
        fit_smooth_basis(data, packed, K, penalty, method="reml", device=CPU)
        .betas.numpy()
        .reshape(3, NDRAW, K)
    )
    ref = np.concatenate([reference_design(o, bot, bot + 8 * TR) for o in onsets])
    answer = {
        "rank": rank,
        "condition": float(s[0] / s[-1]),
        "design_max_error": float(np.abs(x - ref).max()),
        "global_lambda": selected.lam,
        "global_fallback": selected.fallback,
    }
    for method, b in (("ols", ols), ("global", fixed), ("reml_smoothing", reml)):
        mean = b.mean(1)
        answer[method] = {
            "rmse": float(np.mean((b - truth_knots[:, None]) ** 2) ** 0.5),
            "bias_rms": float(np.mean((mean - truth_knots) ** 2) ** 0.5),
            "mean_peak_ratio_by_depth": (mean.max(1) / truth_knots.max(1)).tolist(),
            "first_knot_mean": mean[:, 0].tolist(),
        }
    # This is the key observation-space correction: fitting the same noiseless
    # train uses exactly the expectation of the fitted estimates for fixed lambda.
    gram = xm.T @ xm
    scaled_p = penalty * (np.trace(gram) / np.trace(penalty))
    estimate_operator = np.linalg.solve(gram + selected.lam * scaled_p, xm.T) @ m
    projected_truth = estimate_operator @ signal
    empirical_mean = fixed.mean(1).T
    production_noiseless = (
        fit_smooth_basis(
            torch.tensor(signal.T, dtype=torch.float32),
            packed,
            K,
            penalty,
            method="fixed",
            lam=selected.lam,
            device=CPU,
        )
        .betas.numpy()
        .T
    )
    answer["production_noiseless_vs_dense_projection_max"] = float(
        np.abs(production_noiseless - projected_truth).max()
    )
    answer["projected_truth_vs_sampled_truth_rms"] = float(
        np.mean((projected_truth.T - truth_knots) ** 2) ** 0.5
    )
    answer["empirical_mean_vs_projected_truth_rms"] = float(
        np.mean((empirical_mean - projected_truth) ** 2) ** 0.5
    )
    print(name, json.dumps(answer), flush=True)
    return answer


def afni_referee():
    afni = shutil.which("3dDeconvolve")
    if afni is None:
        return {"available": False}
    results = {}
    with tempfile.TemporaryDirectory(prefix="tent_referee_") as directory:
        for phase in (0.0, 0.3, 0.5):
            bot = TR / 2 if phase == 0.5 else 0.0
            top = bot + 8 * TR
            onsets = (np.array([5, 41, 133, 305]) + phase) * TR
            timing = Path(directory) / "onsets.1D"
            timing.write_text(" ".join(f"{o:.12g}" for o in onsets) + "\n")
            output = Path(directory) / "x.1D"
            cmd = [
                afni,
                "-nodata",
                "400",
                str(TR),
                "-polort",
                "-1",
                "-num_stimts",
                "1",
                "-stim_times",
                "1",
                str(timing),
                f"TENT({bot:.12g},{top:.12g},{K})",
                "-x1D",
                str(output),
                "-x1D_stop",
                "-GOFORIT",
                "10",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=directory)
            expected = np.loadtxt(output)
            got = make_tent_design([onsets], bot, top, TR, 400, device=CPU).numpy()
            results[str(phase)] = {
                "corrected_vs_afni_max": float(np.abs(got - expected).max()),
                "float64_ref_vs_afni_max": float(
                    np.abs(reference_design(onsets, bot, top, 400) - expected).max()
                ),
            }
    return {"available": True, "binary": afni, "cases": results}


def rank_checks():
    # Timing diversity pooled across conditions cannot identify a condition
    # whose own events all share one off-grid phase.
    a = np.arange(5, NTP - 15, 17) * TR
    b = (np.arange(10, NTP - 15, 19) + 0.5) * TR
    x0 = make_tent_design([a], 0, 8 * TR, TR, NTP, device=CPU).numpy()
    x1 = make_tent_design([b], 0, 8 * TR, TR, NTP, device=CPU).numpy()
    x = np.column_stack([x0, x1])
    chol = cholesky(toeplitz(0.5 ** np.arange(NTP)), lower=True)
    whitened = solve_triangular(chol, x)
    return {
        "locked_condition_rank": int(np.linalg.matrix_rank(x0, tol=1e-5)),
        "half_condition_rank": int(np.linalg.matrix_rank(x1, tol=1e-5)),
        "joint_rank_of_18": int(np.linalg.matrix_rank(x, tol=1e-5)),
        "known_ar05_whitened_rank_of_18": int(np.linalg.matrix_rank(whitened, tol=1e-5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = {
        f"constant_{p}": (lambda rng, n, p=p: np.full(n, p), 0.0) for p in (0.0, 0.1, 0.3, 0.5, 0.9)
    }
    cases.update(
        {
            "half_aligned": (lambda rng, n: np.full(n, 0.5), TR / 2),
            "mixed_halves": (lambda rng, n: rng.choice([0, 0.5], n), 0.0),
            "95percent_half": (lambda rng, n: rng.choice([0, 0.5], n, p=[0.05, 0.95]), 0.0),
            "uniform": (lambda rng, n: rng.random(n), 0.0),
            "near_half": (lambda rng, n: rng.uniform(0.49, 0.51, n), 0.0),
        }
    )
    result = {name: audit_case(name, *case) for name, case in cases.items()}
    result["afni_referee"] = afni_referee()
    result["rank_checks"] = rank_checks()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("AFNI", result["afni_referee"])


if __name__ == "__main__":
    main()
