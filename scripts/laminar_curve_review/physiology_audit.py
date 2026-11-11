"""Independent analytic checks of laminar flux and steady-state equations.

Run on CPU with the installed project environment. No reference output files
or integration scheme are needed: these checks are identities from the
physical volume/dHb conservation equations in Havlicek & Uludag (2020).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import torch

from fastfuncstuff.laminar.forward import baseline_hemodynamics, build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import ModelSpec, zero_params


def audit() -> dict[str, object]:
    torch.set_num_threads(1)
    rng = np.random.default_rng(920)
    rows = []
    for K in (3, 7, 9, 10, 11):
        spec = ModelSpec(N=K, K=K, device=torch.device("cpu"))
        P = zero_params(spec)
        c = dataclasses.replace(build_cache(spec, P), n2k=torch.eye(K, dtype=spec.dtype))
        h = baseline_hemodynamics(spec, P)
        F0v, F0d = h["F0v"].numpy(), h["F0d"].numpy()
        V0v, V0d = h["V0v"].numpy(), h["V0d"].numpy()
        f = rng.uniform(0.8, 1.9, K)
        m = 1 + (f - 1) / spec.p0.nr
        av_flow = np.cumsum((F0v * f)[::-1])[::-1]
        av_dhb_flux = np.cumsum((F0v * m)[::-1])[::-1]
        fd_ss = av_flow / F0d
        vv = f**spec.p0.al_v
        qv = vv * m / f
        vd = fd_ss**spec.p0.al_d
        qd = vd * av_dhb_flux / av_flow
        x = torch.zeros(spec.n_states, dtype=spec.dtype)
        x[3 * K : 4 * K] = torch.from_numpy(np.log(f))
        x[4 * K :] = torch.from_numpy(np.log(np.concatenate((vv, qv, vd, qd))))
        dx = rhs(x, torch.zeros(spec.n_inputs, dtype=spec.dtype), c, K, K)
        steady_residual = float(dx[4 * K :].abs().max())

        # Random nonstationary physical states; derive absolute conservation
        # without using the port's normalized ascending-vein inflow weights.
        states = rng.uniform(0.7, 1.4, (4, K))
        vv, qv, vd, qd = states
        x[4 * K :] = torch.from_numpy(np.log(states.ravel()))
        actual = rhs(x, torch.zeros(spec.n_inputs, dtype=spec.dtype), c, K, K)
        physical_actual = actual[4 * K :].numpy().reshape(4, K) * states
        fv = (V0v * vv ** (1 / spec.p0.al_v) + F0v * spec.p0.tau_v_in * f) / (
            V0v + F0v * spec.p0.tau_v_in
        )
        expected = np.zeros((4, K))
        expected[0] = F0v * (f - fv) / V0v
        expected[1] = F0v * (m - fv * qv / vv) / V0v
        flow_below = 0.0
        dhb_below = 0.0
        for k in range(K - 1, -1, -1):
            inflow = F0v[k] * fv[k] + flow_below
            fd = (V0d[k] * vd[k] ** (1 / spec.p0.al_d) + spec.p0.tau_d_in * inflow) / (
                V0d[k] + F0d[k] * spec.p0.tau_d_in
            )
            outflow = F0d[k] * fd
            out_dhb = outflow * qd[k] / vd[k]
            expected[2, k] = (inflow - outflow) / V0d[k]
            expected[3, k] = (F0v[k] * fv[k] * qv[k] / vv[k] + dhb_below - out_dhb) / V0d[k]
            flow_below, dhb_below = outflow, out_dhb
        conservation_residual = float(np.max(np.abs(physical_actual - expected)))
        rest = torch.zeros_like(x)
        baseline_bold = float(g_obs(rest, P, spec, cache=c).abs().max())
        baseline_rhs = float(rhs(rest, torch.zeros(1, dtype=spec.dtype), c, K, K).abs().max())
        assert steady_residual < 1e-12
        assert conservation_residual < 1e-12
        assert baseline_bold == 0
        assert baseline_rhs < 1e-12
        assert np.all(np.diff(F0d) < 0)
        rows.append(
            dict(
                K=K,
                steady_state_max_derivative=steady_residual,
                absolute_flux_max_residual=conservation_residual,
                baseline_bold=baseline_bold,
                baseline_rhs=baseline_rhs,
            )
        )
    return {"seed": 920, "dtype": "float64", "device": "cpu", "checks": rows}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
