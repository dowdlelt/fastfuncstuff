import csv
import sys

import numpy as np

np.set_printoptions(linewidth=220, precision=3, suppress=True)
d = sys.argv[1]
tr = np.load(d + "/truth.npz")
dt = float(tr["dt"])
tf = np.arange(tr["fixate_hi"].shape[0]) * dt
TR, NT = 1.857, 123


def design_proj(nk, top):
    dk = top / (nk - 1)
    Xs = []
    Ys = []
    for r in (1, 2, 3):
        rows = list(csv.DictReader(open(f"{d}/task-fixate_run-{r}_events.tsv"), delimiter="\t"))
        ts = np.arange(NT) * TR
        X = np.zeros((NT, 2 * nk))
        y = np.zeros((NT, 7))
        for row in rows:
            c = 0 if row["stim_contrast"] == "hi" else 1
            lag = ts - float(row["onset"])
            X[:, c * nk : (c + 1) * nk] += np.clip(
                1 - np.abs(lag[:, None] / dk - np.arange(nk)[None]), 0, None
            )
            h = tr["fixate_" + row["stim_contrast"]]
            for k in range(7):
                y[:, k] += np.interp(lag, tf, h[:, k], left=0, right=0)
        P = np.polynomial.legendre.legvander(np.linspace(-1, 1, NT), 2)
        blk = np.zeros((NT, 9))
        blk[:, (r - 1) * 3 : (r) * 3] = P
        Xs.append(np.hstack([X, blk]))
        Ys.append(y)
    X = np.vstack(Xs)
    y = np.vstack(Ys)
    return np.linalg.lstsq(X, y, rcond=None)[0][:nk].T


def l2proj(h, nk, top):
    dk = top / (nk - 1)
    w = tf <= top
    Phi = np.clip(1 - np.abs(tf[w, None] / dk - np.arange(nk)[None]), 0, None)
    return np.linalg.lstsq(Phi, h[w], rcond=None)[0].T


h = tr["fixate_hi"]
for nk in (9, 17):
    times = np.linspace(0, 14.8, nk)
    true = np.stack([np.interp(times, tf, h[:, k]) for k in range(7)])
    a = design_proj(nk, 14.8)
    b = l2proj(h, nk, 14.8)
    print(
        f"nk={nk}  rms(noise-free design fit - point samples) {np.sqrt(((a - true) ** 2).mean()):.4f} | rms(design fit - L2 projection) {np.sqrt(((a - b) ** 2).mean()):.4f}"
    )
    print("   design fit - samples, pial", (a - true)[0][:9])
    print("   design fit - L2 proj, pial", (a - b)[0][:9])
    print("   design fit - L2 proj, mid ", (a - b)[3][:9])
