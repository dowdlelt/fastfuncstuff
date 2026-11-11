import dataclasses

import numpy as np
import torch

from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import ModelSpec, zero_params

torch.set_num_threads(2)
K = N = 7
dt = 0.01


def ss(TE=0.028, fix=False, B0=7.0):
    spec = ModelSpec(N=N, K=K, dt=dt, TE=TE, B0=B0)
    P = zero_params(spec)
    c = dataclasses.replace(build_cache(spec, P), n2k=torch.eye(K, dtype=torch.float64))
    if fix:
        p = spec.p0
        ep_d = p.rho_d / p.rho_t * np.exp(-TE * (p.R2s_d - p.R2s_t))
        c = dataclasses.replace(
            c, k2d=ep_d * p.r0d * 0.35 * TE * torch.ones(K, dtype=torch.float64)
        )
    x = torch.zeros(spec.n_states, dtype=torch.float64)
    x[3 * N : 4 * N] = np.log(1.6)
    with torch.no_grad():
        for t in range(int(60 / dt)):
            d = rhs(x, torch.zeros(1, dtype=torch.float64), c, N, K)
            x[4 * N :] += dt * d[4 * N :]
        return g_obs(x, P, spec, cache=c).numpy(), c


a, c = ss()
b, _ = ss(fix=True)
print("as coded (k2d uses eps_v):", np.round(a, 3))
print("paper eq.7 (eps_d):       ", np.round(b, 3))
print(
    "ratio",
    np.round(b / a, 4),
    " pial/deep as coded %.3f, paper %.3f" % (a[0] / a[-1], b[0] / b[-1]),
)
print(
    "k1v %.3f k2v %.3f k3v %.3f | k1d %.3f k2d %.3f k3d %.3f"
    % tuple(float(v.flatten()[0]) for v in (c.k1v, c.k2v, c.k3v, c.k1d, c.k2d, c.k3d))
)
for TE in (0.020, 0.024, 0.028, 0.033):
    y, _ = ss(TE=TE)
    print(
        "TE %.3f  LBR" % TE,
        np.round(y, 2),
        " shape pial/deep %.3f  scale vs 28ms %.3f" % (y[0] / y[-1], y.mean() / a.mean()),
    )
