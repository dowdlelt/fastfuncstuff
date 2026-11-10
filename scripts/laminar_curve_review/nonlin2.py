"""Noise-free curves deconvolved from the model's NONLINEAR response to the real trains (fixate and gender runs)."""

import csv
import warnings

import nibabel as nib
import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.integrate import integrate
from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.pipeline import reference_p0

d = "d1"
dt = 0.05
TR, NT, K = 1.857, 123, 7
spec = ModelSpec(N=3, K=K, n_inputs=2, n_mod=1, TR=dt, dt=dt, p0=reference_p0())
scale = 0.8247363173044623
mods = {"fixate": [0, 0, 0], "gender": [0, 0, 0.6]}
for nk in (9, 17):
    dk = 14.8 / (nk - 1)
    for task, B in mods.items():
        P = {k: v.clone() for k, v in faes_priors(spec, (1, 1, 1)).pE.items()}
        P["C"] = torch.tensor([[0.0, 1.0], [0.0, 1.4], [0.0, 0.9]], dtype=torch.float64) * scale
        P["B"] = torch.diag(torch.tensor(B, dtype=torch.float64))[None]
        Xs = []
        Ys = []
        for r in (1, 2, 3):
            rows = list(csv.DictReader(open(f"{d}/task-{task}_run-{r}_events.tsv"), delimiter="\t"))
            n = int(round(NT * TR / dt)) + 2
            u = torch.zeros(n, 2, dtype=torch.float64)
            ts = np.arange(NT) * TR
            X = np.zeros((NT, 2 * nk))
            for row in rows:
                o = float(row["onset"])
                a = 1.0 if row["stim_contrast"] == "hi" else 0.7
                i0 = int(round(o / dt))
                u[i0 : i0 + 21, 1] = a
                u[i0 : i0 + 21, 0] = 1.0
                c = 0 if a == 1.0 else 1
                lag = ts - i0 * dt
                X[:, c * nk : (c + 1) * nk] += np.clip(
                    1 - np.abs(lag[:, None] / dk - np.arange(nk)[None]), 0, None
                )
            y = integrate(u, P, spec, n, delay_seconds=0.0).numpy()
            yn = np.stack([np.interp(ts, np.arange(n) * dt, y[:, k]) for k in range(K)], 1)
            Pm = np.zeros((NT, 9))
            Pm[:, (r - 1) * 3 : r * 3] = np.polynomial.legendre.legvander(np.linspace(-1, 1, NT), 2)
            Xs.append(np.hstack([X, Pm]))
            Ys.append(yn)
        b = np.linalg.lstsq(np.vstack(Xs), np.vstack(Ys), rcond=None)[0][:nk].T  # (K, nk) hi curve
        nib.save(
            nib.Nifti1Image(b[:, None, None, :].astype(np.float64), np.eye(4)),
            f"{d}/est_nl{nk}_iresp_{task}_hi.nii.gz",
        )
        print(nk, task, "pial", np.round(b[0][:9], 3))
