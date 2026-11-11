import dataclasses
import os

import numpy as np
import torch

D = os.path.dirname(os.path.abspath(__file__)) + "/"
from fastfuncstuff.laminar.forward import baseline_hemodynamics, build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params

torch.set_num_threads(2)
cbf = np.loadtxt(D + "cbf.txt")
ref = np.loadtxt(D + "lbr_orig.txt")
K = cbf.shape[1]
N = K
dt = 0.01
p0 = P0(V0t=2.5, s_d=0.3)
spec = ModelSpec(N=N, K=K, n_inputs=1, n_mod=1, dt=dt, p0=p0)
P = zero_params(spec)
c = build_cache(spec, P)
c = dataclasses.replace(c, n2k=torch.eye(K, dtype=torch.float64))
x = torch.zeros(spec.n_states, dtype=torch.float64)
u = torch.zeros(2, dtype=torch.float64)[: 1 + 1]
u = (
    torch.zeros(spec.n_inputs + 0, dtype=torch.float64)
    if False
    else torch.zeros(1, dtype=torch.float64)
)
out = np.zeros_like(ref)
cb = torch.as_tensor(cbf)
with torch.no_grad():
    for t in range(cbf.shape[0]):
        x[3 * N : 4 * N] = torch.log(cb[t])
        d = (
            rhs(x, torch.zeros(2, dtype=torch.float64), c, N, K)
            if c.C.shape[-1] == 2
            else rhs(x, u, c, N, K)
        )
        x[4 * N :] = x[4 * N :] + dt * d[4 * N :]
        out[t] = g_obs(x, P, spec, cache=c).numpy()
print("dynamic: max|ours-orig| = %.3e   (peak LBR %.3f%%)" % (np.abs(out - ref).max(), ref.max()))
print("orig peak per depth", np.round(ref.max(0), 3))
# ---- steady state, analytic mass conservation ----
p0 = P0(V0t=2.5, s_d=0.4)
spec = ModelSpec(N=N, K=K, dt=dt, p0=p0)
P = zero_params(spec)
c = dataclasses.replace(build_cache(spec, P), n2k=torch.eye(K, dtype=torch.float64))
f = torch.tensor([1.6, 1.5, 1.7, 1.3, 1.8, 1.4, 1.55], dtype=torch.float64)  # depth-varying flow
x = torch.zeros(spec.n_states, dtype=torch.float64)
x[3 * N : 4 * N] = torch.log(f)
with torch.no_grad():
    for t in range(int(400 / dt)):
        d = rhs(x, torch.zeros(1, dtype=torch.float64), c, N, K)
        x[4 * N :] += dt * d[4 * N :]
vv, qv, vd, qd = [torch.exp(x[4 * N + i * K : 4 * N + (i + 1) * K]) for i in range(4)]
h = baseline_hemodynamics(spec, P)
F0v, F0d = h["F0v"], h["F0d"]
m = (f + 4 - 1) / 4
flow = torch.cumsum((F0v * f).flip(0), 0).flip(0)  # absolute flow through AV at depth k
dhb = torch.cumsum((F0v * f * (m / f)).flip(0), 0).flip(0)  # absolute dHb flux (= sum F0v*m)
fd = flow / F0d
print(
    "steady: v_v-f^al %.2e  q_v/v_v-m/f %.2e  v_d-fd^al %.2e  q_d/v_d - flux ratio %.2e"
    % (
        (vv - f**0.3).abs().max(),
        (qv / vv - m / f).abs().max(),
        (vd - fd**0.2).abs().max(),
        (qd / vd - dhb / flow).abs().max(),
    )
)
ss = np.loadtxt(D + "lbr_orig.txt.ss")
x = torch.zeros(spec.n_states, dtype=torch.float64)
x[3 * N : 4 * N] = np.log(1.6)
with torch.no_grad():
    for t in range(int(40 / dt)):
        d = rhs(x, torch.zeros(1, dtype=torch.float64), c, N, K)
        x[4 * N :] += dt * d[4 * N :]
    mine = g_obs(x, P, spec, cache=c).numpy()
print(
    "steady-state example (cbf 1.6, s_d .4): orig",
    np.round(ss, 4),
    "\n ours",
    np.round(mine, 4),
    " max diff %.2e" % np.abs(ss - mine).max(),
)
