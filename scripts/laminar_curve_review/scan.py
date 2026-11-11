"""Real ROI: how does the fit change if the model's event is placed `shift` s later (a latency offset)?"""

import json
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from fastfuncstuff.laminar.io import model_kernel, pick_roi, read_iresp_by_k, two_condition_batch
from fastfuncstuff.laminar.pipeline import fit_roi, reference_p0
from fastfuncstuff.processing.roi_depth import read_depth_psf_by_k

pfx, roi, shift, space = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
free = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] != "-" else []
K = 7
a = pick_roi(read_iresp_by_k(f"{pfx}_iresp_fixate_hi.tsv")[K], roi)
b = pick_roi(read_iresp_by_k(f"{pfx}_iresp_gender_hi.tsv")[K], roi)
psf = read_depth_psf_by_k(f"{pfx}_depth_psf.tsv")[K]
key = [k for k in psf if roi in k][0]
kern = model_kernel(psf[key])
data = two_condition_batch(
    a.values[None],
    b.values[None],
    (a.times - shift, b.times - shift),
    duration=1.0,
    p0=reference_p0(),
)
data.y = data.y[0]
targets = None if space == "full" else [(0, 0, 0)]
names = None if space == "full" else ["null"]
t0 = time.time()
fit = fit_roi({K: data}, {K: kern}, targets=targets, names=names, rescue=2, free=free)
print(
    json.dumps(
        dict(
            pfx=pfx.split("/")[-1],
            roi=roi,
            shift=shift,
            free=free,
            best=fit.models[fit.best],
            F=np.round(fit.F, 1).tolist(),
            p_depth=np.round(fit.p_depth, 3).tolist(),
            B=np.round(fit.B_mean, 2).tolist(),
            r2=float(np.round(fit.r2[fit.best], 4)),
            post={k: round(v, 3) for k, v in fit.meta["posterior"].items()},
            stalled=int(fit.stalled.sum()),
            nrows=int(data.rows.sum()),
            sec=round(time.time() - t0),
        )
    ),
    flush=True,
)
