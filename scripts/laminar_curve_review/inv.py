"""Invert sim curves through the laminar model space; report recovery of the known truth."""

import json
import sys
import time
import warnings

import nibabel as nib
import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from fastfuncstuff.laminar.io import two_condition_batch
from fastfuncstuff.laminar.pipeline import fit_roi, reference_p0

d, name, mode, base, test = sys.argv[1:6]
extra = sys.argv[6:]
tr = np.load(f"{d}/truth.npz")
dt = float(tr["dt"])


def curves(cond):
    if name.startswith("truth"):
        nk = int(name[5:])
        times = np.linspace(0, 14.8, nk)
        h = tr[cond]
        tf = np.arange(h.shape[0]) * dt
        return np.stack([np.interp(times, tf, h[:, k]) for k in range(7)])[:, None, :], times
    est = nib.load(f"{d}/est_{name}_iresp_{cond}.nii.gz").get_fdata()[:, :, 0, :]
    return est, np.linspace(0, 14.8, est.shape[-1])


a, times = curves(base)
b, _ = curves(test)
if mode == "mean":
    sel = slice(None)
elif mode.startswith("avg"):
    n, i = map(int, mode[3:].split("_"))
    sel = slice(i * n, (i + 1) * n)
else:
    i = int(mode)
    sel = slice(i, i + 1)
A = a[:, sel].mean(1).T[None, :, ::-1].copy()
B = b[:, sel].mean(1).T[None, :, ::-1].copy()  # (1, nt, K) deep first
data = two_condition_batch(A, B, times, duration=1.0, p0=reference_p0())
data.y = data.y[0]
t0 = time.time()
fit = fit_roi({7: data}, None, rescue=2, hyper_from_rows=("rows" in extra))
out = dict(
    name=name,
    mode=mode,
    pair=f"{test}-{base}",
    best=fit.models[fit.best],
    p_depth=np.round(fit.p_depth, 3).tolist(),
    B=np.round(fit.B_mean, 3).tolist(),
    F=np.round(fit.F - fit.F.max(), 1).tolist(),
    r2=float(np.round(fit.r2[fit.best], 4)),
    post=fit.meta["posterior"],
    stalled=int(fit.stalled.sum()),
    sec=round(time.time() - t0),
)
print(json.dumps(out), flush=True)
