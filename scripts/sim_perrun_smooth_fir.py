"""Per-run smooth FIR/TENT curves: accuracy vs OLS, honest validation, and
false positives / power for run-level statistics.

Estimators of each run's curve:
  ols        per-run OLS TENT (the usual approach)
  reml_run   per-run smooth fit, lambda by REML on that run alone
  shared     per-run smooth fit, lambda from a pooled REML fit over all runs,
             transferred at constant ABSOLUTE strength (relative lambda is scaled
             by trace(Gram), so lam_run = lam_pool * tr_pool / tr_run)
  shared_raw pooled relative lambda reused as-is (the transfer trap)

CPU only, ~40 s. Usage: python scripts/sim_perrun_smooth_fir.py [exp1] [val] [stats]
The group-level (across-subject) test is scripts/sim_perrun_smooth_fir_group.py.
Findings: fmri_wiki concepts/Smooth FIR.md, "Per-run curves".
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from scipy import stats
from scipy.stats import gamma

from fastfuncstuff.design.builder import legendre_polynomials
from fastfuncstuff.design.matrices import make_tent_design
from fastfuncstuff.glm.smooth_basis import fit_smooth_basis, roughness_penalty

CPU = torch.device("cpu")
TR = 2.0
T = 200
R = 8
POLORT = 2
WIN = (0.0, 20.0)
N_SEEDS = 10
N_VOX = 300
torch.set_num_threads(4)
# run-level tests need ROI-averaged curves: single-voxel CNR gives ~5% power
STATS_NULL_CNR = (1.0, 2.0, 4.0)
STATS_POWER_CNR = (1.0, 2.0)


def hrf(t):
    def g(u):
        u = np.clip(u, 0, None)
        return gamma.pdf(u, 6) - gamma.pdf(u, 16) / 6

    return g(t) / g(np.linspace(0, 20, 2001)).max()


def onsets(rng):
    # jittered continuous onsets, min gap 5 s, ~24 events in a 400 s run
    t, out = 2.0 + rng.uniform(0, 4), []
    while t < T * TR - 22:
        out.append(t)
        t += 5 + rng.exponential(10)
    return np.array(out)


def run_design(rng, n_basis):
    x = make_tent_design([onsets(rng)], *WIN, TR, T, n_basis=n_basis, device=CPU).double().numpy()
    p = legendre_polynomials(T, POLORT, normalize=True)
    return x, p


def proj_out(p, a):
    q, _ = np.linalg.qr(p)
    return a - q @ (q.T @ a)


def gram_trace(x, p):
    xp = proj_out(p, x)
    return float(np.sum(xp * xp))


def pooled_design(xs, ps):
    k = xs[0].shape[1]
    tt = sum(x.shape[0] for x in xs)
    npol = sum(p.shape[1] for p in ps)
    d = np.zeros((tt, k + npol))
    r0 = c0 = 0
    for x, p in zip(xs, ps, strict=True):
        n = x.shape[0]
        d[r0 : r0 + n, :k] = x
        d[r0 : r0 + n, k + c0 : k + c0 + p.shape[1]] = p
        r0 += n
        c0 += p.shape[1]
    return d


def smooth(y, x, p, pen, method, lam=None):
    d = torch.as_tensor(np.hstack([x, p]), dtype=torch.float32)
    f = fit_smooth_basis(
        torch.as_tensor(y, dtype=torch.float32),
        d,
        x.shape[1],
        pen,
        method=method,
        lam=lam,
        device=CPU,
    )
    return f.betas.double().numpy(), f.lam.double().numpy()


def pooled_lam(ys, xs, ps, pen):
    d = torch.as_tensor(pooled_design(xs, ps), dtype=torch.float32)
    y = torch.as_tensor(np.hstack(ys), dtype=torch.float32)
    f = fit_smooth_basis(y, d, xs[0].shape[1], pen, method="reml", device=CPU)
    return f.lam.double().numpy(), f.betas.double().numpy()


def ols(y, x, p):
    b = np.linalg.lstsq(np.hstack([x, p]), y.T, rcond=None)[0].T
    return b[:, : x.shape[1]]


def ar1(rng, shape, rho):
    e = rng.standard_normal(shape)
    if rho == 0:
        return e
    out = np.empty_like(e)
    out[..., 0] = e[..., 0]
    s = np.sqrt(1 - rho**2)
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + s * e[..., t]
    return out


def simulate(
    cnr,
    *,
    n_basis=None,
    rho=0.0,
    noise_scale=None,
    amp_scale=None,
    shift=None,
    seed0=0,
    validate=False,
):
    """Returns per-estimator curves (S*V, R, K), truth (R, K), knot times, validation stats."""
    k_times = None
    ests = {m: [] for m in ("ols", "reml_run", "shared", "shared_raw")}
    lam_log = {m: [] for m in ("reml_run", "shared")}
    val = {m: {"mse_vs_ols_target": [], "heldout_r2": []} for m in ("ols", "reml_run", "shared")}
    noise_scale = np.ones(R) if noise_scale is None else noise_scale
    amp_scale = np.ones(R) if amp_scale is None else amp_scale
    shift = np.zeros(R) if shift is None else shift
    truths = None
    for s in range(N_SEEDS):
        rng = np.random.default_rng(1000 * seed0 + s)
        xs, ps = zip(*[run_design(rng, n_basis) for _ in range(R)], strict=True)
        k = xs[0].shape[1]
        k_times = np.linspace(*WIN, k)
        pen = roughness_penalty([k], order=2)
        truths = np.stack([cnr * amp_scale[r] * hrf(k_times - shift[r]) for r in range(R)])
        ys = [
            (xs[r] @ truths[r])[None, :] + noise_scale[r] * ar1(rng, (N_VOX, T), rho)
            for r in range(R)
        ]
        tr_run = np.array([gram_trace(xs[r], ps[r]) for r in range(R)])
        lam_pool, _ = pooled_lam(ys, xs, ps, pen)
        per = {m: np.empty((N_VOX, R, k)) for m in ests}
        for r in range(R):
            per["ols"][:, r] = ols(ys[r], xs[r], ps[r])
            per["reml_run"][:, r], lr = smooth(ys[r], xs[r], ps[r], pen, "reml")
            lam_log["reml_run"].append(np.log10(lr * tr_run[r]))
            lam_abs = lam_pool * tr_run.sum()
            per["shared"][:, r], _ = smooth(ys[r], xs[r], ps[r], pen, "fixed", lam_abs / tr_run[r])
            lam_log["shared"].append(np.log10(lam_abs))
            per["shared_raw"][:, r], _ = smooth(ys[r], xs[r], ps[r], pen, "fixed", lam_pool)
        for m in ests:
            ests[m].append(per[m])
        if validate:
            # Fit on run r, score on run j != r. The shared lambda comes from a pooled
            # REML over every run except the target j (fold-local).
            ols_b = per["ols"]
            yproj = [proj_out(ps[j], ys[j].T).T for j in range(R)]
            xproj = [proj_out(ps[j], xs[j]) for j in range(R)]
            for j in range(R):
                others = [i for i in range(R) if i != j]
                lam_j, _ = pooled_lam(
                    [ys[i] for i in others], [xs[i] for i in others], [ps[i] for i in others], pen
                )
                tr_pool_j = tr_run[others].sum()
                for r in others:
                    b = {
                        "ols": ols_b[:, r],
                        "reml_run": per["reml_run"][:, r],
                        "shared": smooth(
                            ys[r], xs[r], ps[r], pen, "fixed", lam_j * tr_pool_j / tr_run[r]
                        )[0],
                    }
                    ss_tot = (yproj[j] ** 2).sum(axis=1)
                    for m, bb in b.items():
                        val[m]["mse_vs_ols_target"].append(((bb - ols_b[:, j]) ** 2).mean(axis=1))
                        res = yproj[j] - bb @ xproj[j].T
                        val[m]["heldout_r2"].append(1 - (res**2).sum(axis=1) / ss_tot)
    out = {m: np.concatenate(v) for m, v in ests.items()}
    lam_log = {m: np.concatenate(v) for m, v in lam_log.items()}
    return out, truths, k_times, val, lam_log


# ---- features --------------------------------------------------------------


def features(curves, k_times, peak_idx):
    dt = k_times[1] - k_times[0]
    auc = curves.sum(-1) * dt
    amp = curves[..., peak_idx]
    pmax = curves.max(-1)
    i = curves.argmax(-1).clip(1, curves.shape[-1] - 2)
    f0 = np.take_along_axis(curves, (i - 1)[..., None], -1)[..., 0]
    f1 = np.take_along_axis(curves, i[..., None], -1)[..., 0]
    f2 = np.take_along_axis(curves, (i + 1)[..., None], -1)[..., 0]
    c = f0 - 2 * f1 + f2
    step = np.where(c < 0, 0.5 * (f0 - f2) / np.where(c == 0, 1, c), 0).clip(-1, 1)
    lat = k_times[0] + (i + step) * dt
    return {"auc": auc, "amp@peak": amp, "peak_max": pmax, "latency": lat}


def slope_test(f):
    """Per-voxel linear trend of a feature across runs: (slope, p)."""
    x = np.arange(R) - (R - 1) / 2
    slope = (f * x).sum(-1) / (x**2).sum()
    fit = f.mean(-1, keepdims=True) + slope[:, None] * x
    s2 = ((f - fit) ** 2).sum(-1) / (R - 2)
    t = slope / np.sqrt(s2 / (x**2).sum())
    return slope, 2 * stats.t.sf(np.abs(t), R - 2)


def corr_rows(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1), 0.0)


METHODS = ("ols", "reml_run", "shared", "shared_raw")


def exp1(results):
    print("\n=== Exp 1: per-run curve accuracy (MSE vs truth, relative to OLS) ===")
    cells = [
        ("TR knots, white", {}),
        ("TR knots, AR(1) .3", {"rho": 0.3}),
        ("1 s knots (sub-TR)", {"n_basis": 21}),
    ]
    for name, kw in cells:
        print(f"\n-- {name} --")
        print(
            f"{'CNR':>5} "
            + " ".join(f"{m:>11}" for m in METHODS)
            + "  | ols MSE   log10 lam_abs reml_run/shared (median)"
        )
        for cnr in (0.0, 0.1, 0.25, 0.5, 1.0):
            t0 = time.time()
            ests, truth, kt, _, lamlog = simulate(cnr, **kw)
            mse = {m: ((ests[m] - truth[None]) ** 2).mean() for m in METHODS}
            print(
                f"{cnr:>5.2f} "
                + " ".join(f"{mse[m] / mse['ols']:>11.3f}" for m in METHODS)
                + f"  | {mse['ols']:.4f}   {np.median(lamlog['reml_run']):6.2f} / {np.median(lamlog['shared']):6.2f}"
                + f"   ({time.time() - t0:.0f}s)"
            )
            results.setdefault("exp1", {}).setdefault(name, {})[cnr] = {
                m: float(mse[m]) for m in METHODS
            }


def exp_validation(results):
    print("\n=== Exp 1b: does the validation rank estimators like the truth does? ===")
    print("fit run r, score on run j (OLS target curve / held-out data R2); diff vs OLS")
    print(
        f"{'CNR':>5} {'method':>9} {'trueMSE-d':>10} {'valMSE-d':>10} {'heldR2-d':>10} "
        f"{'win%(val)':>9} {'win%(true)':>10} {'r_pair':>7} {'r_truth':>7}"
    )
    for cnr in (0.0, 0.1, 0.25, 0.5):
        ests, truth, kt, val, _ = simulate(cnr, validate=True, seed0=7)
        mse_true = {m: ((ests[m] - truth[None]) ** 2).mean(axis=(1, 2)) for m in METHODS}
        v_ols = np.concatenate(val["ols"]["mse_vs_ols_target"])
        r2_ols = np.concatenate(val["ols"]["heldout_r2"])
        for m in ("reml_run", "shared"):
            vm = np.concatenate(val[m]["mse_vs_ols_target"])
            r2m = np.concatenate(val[m]["heldout_r2"])
            # smoothed-vs-smoothed inter-run correlation (the trap) vs corr with truth
            c = ests[m]
            pair = np.mean(
                [corr_rows(c[:, a], c[:, b]).mean() for a in range(R) for b in range(a + 1, R)]
            )
            tru = (
                np.mean(
                    [
                        corr_rows(c[:, a], np.broadcast_to(truth[a], c[:, a].shape)).mean()
                        for a in range(R)
                    ]
                )
                if cnr > 0
                else float("nan")
            )
            print(
                f"{cnr:>5.2f} {m:>9} {(mse_true[m] - mse_true['ols']).mean():>10.4f} "
                f"{(vm - v_ols).mean():>10.4f} {(r2m - r2_ols).mean():>10.4f} "
                f"{100 * ((vm - v_ols) < 0).mean():>8.0f}% {100 * ((mse_true[m] - mse_true['ols']) < 0).mean():>9.0f}% "
                f"{pair:>7.3f} {tru:>7.3f}"
            )
        c = ests["ols"]
        pair = np.mean(
            [corr_rows(c[:, a], c[:, b]).mean() for a in range(R) for b in range(a + 1, R)]
        )
        tru = (
            np.mean(
                [
                    corr_rows(c[:, a], np.broadcast_to(truth[a], c[:, a].shape)).mean()
                    for a in range(R)
                ]
            )
            if cnr > 0
            else float("nan")
        )
        print(
            f"{cnr:>5.2f} {'ols':>9} {'':>10} {'':>10} {'':>10} {'':>9} {'':>10} {pair:>7.3f} {tru:>7.3f}"
        )


def run_stats(title, key, cnr, **kw):
    ests, truth, kt, _, _ = simulate(cnr, seed0=3, **kw)
    peak_idx = int(np.argmax(hrf(kt)))
    print(f"\n-- {title} (CNR {cnr}) --  rejection rate of a linear trend across runs, alpha .05")
    names = ("auc", "amp@peak", "peak_max", "latency")
    print(
        f"{'method':>11} "
        + " ".join(f"{n:>9}" for n in names)
        + "   mean slope (amp@peak, latency)"
    )
    out = {}
    for m in METHODS:
        feats = features(ests[m], kt, peak_idx)
        rates, slopes = {}, {}
        for n in names:
            sl, p = slope_test(feats[n])
            rates[n] = float((p < 0.05).mean())
            slopes[n] = float(sl.mean())
        out[m] = rates
        print(
            f"{m:>11} "
            + " ".join(f"{100 * rates[n]:>8.1f}%" for n in names)
            + f"   {slopes['amp@peak']:+.4f}, {slopes['latency']:+.3f}s"
        )
    return out


def exp_stats(results):
    print("\n=== Exp 2: false positives -- same true curve, noise rises 1x -> 2x across runs ===")
    ns = np.linspace(1.0, 2.0, R)
    for cnr in STATS_NULL_CNR:
        results.setdefault("exp2", {})[cnr] = run_stats(
            "null, rising noise", "fp", cnr, noise_scale=ns
        )
    results["exp2_eq"] = run_stats("null, equal noise (sanity)", "fp_eq", STATS_NULL_CNR[-1])

    print("\n=== Exp 3: power -- equal noise ===")
    results.setdefault("exp3", {})
    for cnr in STATS_POWER_CNR:
        results["exp3"][f"hab_{cnr}"] = run_stats(
            "amplitude falls 30% across runs", "hab", cnr, amp_scale=np.linspace(1.0, 0.7, R)
        )
        results["exp3"][f"lat_{cnr}"] = run_stats(
            "latency grows 0 -> 2 s across runs", "lat", cnr, shift=np.linspace(0.0, 2.0, R)
        )


if __name__ == "__main__":
    which = sys.argv[1:] or ["exp1", "val", "stats"]
    results: dict = {}
    if "exp1" in which:
        exp1(results)
    if "val" in which:
        exp_validation(results)
    if "stats" in which:
        exp_stats(results)
