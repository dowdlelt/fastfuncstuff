"""How far is the model's own response to the real event train from a sum of isolated responses?"""

import csv
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
np.set_printoptions(linewidth=220, precision=3, suppress=True)
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.integrate import integrate
from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.pipeline import reference_p0

d = "d1"
tr = np.load(f"{d}/truth.npz")
dt = 0.05
tf = np.arange(tr["fixate_hi"].shape[0]) * dt
TR, NT, K = 1.857, 123, 7
spec = ModelSpec(N=3, K=K, n_inputs=2, n_mod=1, TR=dt, dt=dt, p0=reference_p0())
scale = 0.8247363173044623
P = {k: v.clone() for k, v in faes_priors(spec, (1, 1, 1)).pE.items()}
P["C"] = torch.tensor([[0.0, 1.0], [0.0, 1.4], [0.0, 0.9]], dtype=torch.float64) * scale
Xs = []
Yn = []
Yl = []
for nk in (17,):
    dk = 14.8 / (nk - 1)
    for r in (1, 2, 3):
        rows = list(csv.DictReader(open(f"{d}/task-fixate_run-{r}_events.tsv"), delimiter="\t"))
        n = int(round(NT * TR / dt)) + 2
        u = torch.zeros(n, 2, dtype=torch.float64)
        ts = np.arange(NT) * TR
        X = np.zeros((NT, 2 * nk))
        lin = np.zeros((NT, K))
        for row in rows:
            o = float(row["onset"])
            a = 1.0 if row["stim_contrast"] == "hi" else 0.7
            i0 = int(round(o / dt))
            u[i0 : i0 + 21, 1] = a  # 1 s, end-inclusive like build_input
            c = 0 if a == 1.0 else 1
            lag = ts - o
            X[:, c * nk : (c + 1) * nk] += np.clip(
                1 - np.abs(lag[:, None] / dk - np.arange(nk)[None]), 0, None
            )
            h = tr["fixate_" + row["stim_contrast"]]
            for k in range(K):
                lin[:, k] += np.interp(
                    lag - (i0 * dt - o), tf, h[:, k], left=0, right=0
                )  # same 50 ms onset grid as the train
        y = integrate(u, P, spec, n, delay_seconds=0.0).numpy()
        yn = np.stack([np.interp(ts, np.arange(n) * dt, y[:, k]) for k in range(K)], 1)
        Pm = np.zeros((NT, 9))
        Pm[:, (r - 1) * 3 : r * 3] = np.polynomial.legendre.legvander(np.linspace(-1, 1, NT), 2)
        Xs.append(np.hstack([X, Pm]))
        Yn.append(yn)
        Yl.append(lin)
        print(
            "run",
            r,
            "mean level of the train (pial, deep): nonlinear %.3f %.3f | linear sum %.3f %.3f | peak-to-peak %.2f %.2f"
            % (
                yn[20:, 0].mean(),
                yn[20:, -1].mean(),
                lin[20:, 0].mean(),
                lin[20:, -1].mean(),
                np.ptp(yn[20:, 0]),
                np.ptp(lin[20:, 0]),
            ),
        )
    X = np.vstack(Xs)
    bn = np.linalg.lstsq(X, np.vstack(Yn), rcond=None)[0][:nk].T
    bl = np.linalg.lstsq(X, np.vstack(Yl), rcond=None)[0][:nk].T
    print(
        "deconvolved hi curve from the NONLINEAR train / from the linear sum, ratio at each curve's peak, by depth pial..deep:"
    )
    pk = bl.argmax(1)
    print("  ", np.array([bn[k, pk[k]] / bl[k, pk[k]] for k in range(K)]))
    print("  pial  linear", bl[0][:12])
    print("  pial  nonlin", bn[0][:12])
    print("  deep  linear", bl[6][:12])
    print("  deep  nonlin", bn[6][:12])
    print(
        "  pial/deep peak ratio: linear %.3f  nonlinear %.3f"
        % (bl[0].max() / bl[6].max(), bn[0].max() / bn[6].max())
    )
