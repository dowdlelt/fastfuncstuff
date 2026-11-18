"""Frequency-free cardiac discovery: constant-modulus 2-D temporal subspaces.

The heartbeat seen at volume rate is c_n = exp(i phi(n TR)): its phase is aliased and
wanders with heart-rate variability, so it need not be narrowband (and folds at 0 and
Nyquist), but its modulus is constant. Real voxels carry Re(a_v c_n) with a free
complex gain (sign, delay, slice-time rotation), so the pair (cos phi_n, sin phi_n)
spans a 2-D temporal subspace of the whole-brain data. In whitened PCA coordinates x_n
we look for an orthonormal W (K x 2) minimising E[(|W^T x|^2 - 2)^2]: a 2-D independent
subspace with the most sub-Gaussian radius. No band, no template, no rate.

The recording is read only after all subspaces exist, as an evaluation overlay.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p


def legendre_detrend(Y, degree):
    nt = Y.shape[1]
    x = np.linspace(-1, 1, nt)
    L = np.polynomial.legendre.legvander(x, degree)
    Q, _ = np.linalg.qr(L)
    return Y - (Y @ Q) @ Q.T


def cm_subspaces(X, n_restarts, rng, iters=400, lr=0.02):
    """X: (K, nt) whitened. Returns list of (cost, W) sorted, distinct."""
    K, nt = X.shape
    out = []
    for _ in range(n_restarts):
        W, _ = np.linalg.qr(rng.standard_normal((K, 2)))
        for _ in range(iters):
            Z = W.T @ X  # (2, nt)
            r2 = (Z**2).sum(0)
            # minimise E[r^4] on the Stiefel manifold (E[r^2] = 2 fixed by whitening)
            G = (X * r2) @ Z.T * (4 / nt)
            G -= W @ (W.T @ G + G.T @ W) / 2  # tangent
            W, _ = np.linalg.qr(W - lr * G)
        r2 = ((W.T @ X) ** 2).sum(0)
        cost = float(np.mean(r2**2) / np.mean(r2) ** 2)  # 2 for a Gaussian pair, 1 for CM
        out.append((cost, W))
    out.sort(key=lambda t: t[0])
    distinct = []
    for cost, W in out:
        # subspace overlap: singular values of W^T W'
        if all(np.linalg.svd(W.T @ V, compute_uv=False).min() < 0.9 for _, V in distinct):
            distinct.append((cost, W))
    return distinct


def recorded_phase(peaks, t):
    k = np.arange(peaks.size)
    return np.interp(t, peaks, 2 * np.pi * k, left=np.nan, right=np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=60)
    args = ap.parse_args()
    row = next(
        ln.split() for ln in args.list.read_text().splitlines() if ln.startswith(args.label + " ")
    )
    side = json.loads(Path(row[3]).read_text())
    st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(row[1]).dataobj).astype(np.float32)
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    mag = mag[..., :nt]
    mean = mag.mean(-1)
    mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
    zidx = np.nonzero(mask)[2]
    Y = mag[mask].astype(np.float64)
    del mag
    Y = legendre_detrend(Y, args.degree)
    Y /= Y.std(1, keepdims=True) + 1e-9
    # temporal PCA (voxels are observations of the nt-dim time course)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    K = args.k
    X = Vt[:K] * np.sqrt(nt)  # whitened (K, nt), unit variance rows
    rng = np.random.default_rng(0)
    subs = cm_subspaces(X, args.restarts, rng)

    # ---- evaluation only below ----
    peaks = np.loadtxt(row[4]) / float(row[6]) + float(row[5])
    tvol = np.arange(nt) * tr
    results = []
    for rank, (cost, W) in enumerate(subs[:10]):
        Z = W.T @ X
        c = Z[0] + 1j * Z[1]
        # map: complex gain per voxel; slope of its phase vs slice time = unaliased rate
        m = Y @ np.conj(c) / nt
        best = (0.0, None, None)
        for s in np.unique(st):
            ph = recorded_phase(peaks, tvol + s)
            ok = np.isfinite(ph)
            for sgn in (1, -1):
                lock = abs(np.mean(np.exp(1j * (np.angle(c[ok]) - sgn * ph[ok]))))
                if lock > best[0]:
                    best = (float(lock), float(s), sgn)
        # rate implied by slice-time phase ramp of the strongest voxels
        w = np.abs(m) ** 2
        top = w > np.quantile(w, 0.98)
        zs = st[zidx[top]]
        rates = np.arange(0.0, 2.5, 0.005)
        # doubled angle removes voxel sign; scan unaliased rate
        coh = [
            abs(np.sum(w[top] * np.exp(2j * (np.angle(m[top]) - 2 * np.pi * f * zs))))
            / w[top].sum()
            for f in rates
        ]
        coh = np.asarray(coh)
        order = np.argsort(coh)[::-1]
        results.append(
            dict(
                rank=rank,
                cm_ratio=cost,
                lock=best[0],
                lock_slice_time=best[1],
                lock_sign=best[2],
                ramp_rate_bpm=float(rates[order[0]] * 60),
                ramp_coherence=float(coh[order[0]]),
            )
        )
        print(args.label, results[-1], flush=True)

    # reference: top PCA pairs and random pairs
    def lock_of(c):
        b = 0.0
        for s in np.unique(st):
            ph = recorded_phase(peaks, tvol + s)
            ok = np.isfinite(ph)
            for sgn in (1, -1):
                b = max(b, abs(np.mean(np.exp(1j * (np.angle(c[ok]) - sgn * ph[ok])))))
        return float(b)

    pca_pairs = [lock_of(X[i] + 1j * X[j]) for i in range(10) for j in range(i + 1, 10)]
    ij = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    for q in np.argsort(pca_pairs)[::-1][:3]:
        i, j = ij[q]
        r2 = X[i] ** 2 + X[j] ** 2
        print(
            args.label,
            "pca pair",
            ij[q],
            "lock",
            round(pca_pairs[q], 3),
            "cm_ratio",
            round(float(np.mean(r2**2) / np.mean(r2) ** 2), 3),
            flush=True,
        )
    rand = []
    for _ in range(60):
        Wr, _ = np.linalg.qr(rng.standard_normal((K, 2)))
        Z = Wr.T @ X
        rand.append(lock_of(Z[0] + 1j * Z[1]))
    ref_hr = float(60 / np.median(np.diff(peaks)))
    rec = dict(
        run=args.label,
        nt=nt,
        k=K,
        ref_hr=ref_hr,
        subspaces=results,
        pca_pair_lock_max=max(pca_pairs),
        random_lock_p95=float(np.quantile(rand, 0.95)),
        random_lock_max=max(rand),
    )
    print(
        args.label,
        "ref_hr",
        round(ref_hr, 1),
        "pca max",
        round(max(pca_pairs), 3),
        "rand p95/max",
        round(rec["random_lock_p95"], 3),
        round(rec["random_lock_max"], 3),
    )
    args.output.write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()


def cm_traces(mag, mask, k=None, n=3, degree=4, restarts=60, seed=0):
    """Image-only: the ``n`` most constant-modulus complex volume-rate traces.

    Returns [(cm_ratio, c)], best first. ``k`` defaults to min(20, nt // 6): 40 overfit
    the 120-volume Rockland runs, 12 missed sub-06's heartbeat (it sat beyond PC 12).
    """
    nt = mag.shape[-1]
    k = k or min(20, nt // 6)
    Y = legendre_detrend(mag[mask].astype(np.float64), degree)
    Y /= Y.std(1, keepdims=True) + 1e-9
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    X = Vt[:k] * np.sqrt(nt)
    subs = cm_subspaces(X, restarts, np.random.default_rng(seed))
    return [(cost, (W.T @ X)[0] + 1j * (W.T @ X)[1]) for cost, W in subs[:n]]


def unwrap_around_rate(theta_steps, tr, window_s=60.0):
    """Per-volume phase steps lifted around a local circular-mean rate, not around 0.

    The complex trace's aliased rate has no fold, but plain unwrapping slips a cycle
    whenever a step crosses +-pi -- exactly where a heartbeat near (k + 1/2)/TR sits.
    Wrapping around the local mean step instead keeps the slow HRV drift continuous
    as long as it stays within 1/(2 TR) of that mean."""
    w = max(3, int(round(window_s / tr)) | 1)
    u = np.exp(1j * theta_steps)
    ref = np.angle(np.convolve(u, np.ones(w) / w, mode="same"))
    return ref + np.angle(u * np.exp(-1j * ref))


def excitation_start(c, times, tr, band=(0.6, 2.0)):
    """An in-band (nt, G) start phase at the excitation times from a volume trace.

    The trace is referenced to the mean excitation time; the alias step chosen here is
    only a start -- refine_cardiac re-chooses the alias by slice alignment."""
    nt = c.size
    steps = unwrap_around_rate(np.angle(c[1:] * np.conj(c[:-1])), tr)
    theta = np.angle(c[0]) + np.concatenate([[0.0], np.cumsum(steps)])
    tv = times.mean(1)
    f_a = (theta[-1] - theta[0]) / (2 * np.pi * (tv[-1] - tv[0]))
    k = int(np.round((np.mean(band) - f_a) * tr))
    slope = np.gradient(theta, tv)
    flat = np.interp(times.ravel(), tv, theta).reshape(times.shape)
    # extrapolate linearly past the first/last volume centre instead of clamping
    lo, hi = times < tv[0], times > tv[-1]
    flat[lo] = theta[0] + slope[0] * (times[lo] - tv[0])
    flat[hi] = theta[-1] + slope[-1] * (times[hi] - tv[-1])
    assert nt == times.shape[0]
    return flat + 2 * np.pi * k * times / tr
