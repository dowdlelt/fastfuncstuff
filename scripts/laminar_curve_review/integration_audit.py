"""Compare the two laminar Taylor steppers with a high-accuracy ODE solution.

The reference uses the same RHS with an independent SciPy integrator. The
boxcar is constructed explicitly to keep its duration and the observation
clock identical as dt changes; SPM's end-inclusive boxcars would confound
the comparison by adding a different duration at each step size.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.integrate import solve_ivp

from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
from fastfuncstuff.laminar.integrate import integrate
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params


def audit() -> dict[str, object]:
    torch.set_num_threads(1)
    spec0 = ModelSpec(
        N=3,
        K=7,
        n_inputs=2,
        n_mod=1,
        TR=0.5,
        device=torch.device("cpu"),
        p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5),
    )
    P = zero_params(spec0)
    P["C"] = torch.tensor([[0.0, 1.0]] * 3, dtype=spec0.dtype)
    P["B"][0, 2, 2] = 0.6
    P["mu"] = torch.tensor(-0.8, dtype=spec0.dtype)
    P["lam"] = torch.tensor(1.8, dtype=spec0.dtype)
    cache = build_cache(spec0, P)
    sample_times = np.arange(24) * spec0.TR
    x = np.zeros(spec0.n_states)
    observations = []
    reference_states = []
    for start, stop, active in ((0.0, 1.0, 0.0), (1.0, 2.0, 1.0), (2.0, 12.0, 0.0)):
        u = torch.full((2,), active, dtype=spec0.dtype)

        def f(t, state, _u=u):
            del t
            return rhs(torch.from_numpy(state), _u, cache, spec0.N, spec0.K).numpy()

        solution = solve_ivp(
            f, (start, stop), x, method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True
        )
        assert solution.success
        times = sample_times[(sample_times >= start) & (sample_times < stop)]
        states = solution.sol(times).T
        reference_states.append(states)
        observations.append(g_obs(torch.from_numpy(states), P, spec0, cache=cache).numpy())
        x = solution.y[:, -1]
    truth = np.concatenate(observations)
    states_truth = np.concatenate(reference_states)
    rows = []
    output_by_mode = {}
    for dt in (0.05, 0.025):
        spec = ModelSpec(**{**spec0.__dict__, "dt": dt})
        micro_times = np.arange(round(12 / dt)) * dt
        active = (micro_times >= 1.0) & (micro_times < 2.0)
        u = torch.from_numpy(np.repeat(active[:, None], 2, axis=1)).to(spec.dtype)
        output_by_mode[dt] = {}
        for mode in ("reference", "exact"):
            y, states = integrate(
                u,
                P,
                spec,
                len(sample_times),
                delay_seconds=dt,
                jacobian=mode,
                compile=False,
                return_states=True,
            )
            y = y.numpy()
            output_by_mode[dt][mode] = y
            err = y - truth
            rows.append(
                dict(
                    dt=dt,
                    mode=mode,
                    max_abs_bold_error_pct=float(np.abs(err).max()),
                    rms_bold_error_pct=float(np.sqrt(np.mean(err**2))),
                    max_error_fraction_of_peak=float(np.abs(err).max() / truth.max()),
                    peak_ratio_by_depth=(y.max(0) / truth.max(0)).tolist(),
                    max_abs_state_error=float(np.max(np.abs(states.numpy() - states_truth))),
                )
            )
    return dict(
        reference="SciPy DOP853 on the same continuous RHS, piecewise constant input",
        rtol=1e-11,
        atol=1e-13,
        K=7,
        N=3,
        sample_times=sample_times.tolist(),
        stimulus="modulation and driving inputs both 1 on [1,2) seconds; 0 elsewhere",
        truth_peak_pct=float(truth.max()),
        results=rows,
        reference_exact_max_difference_pct={
            str(dt): float(np.abs(out["reference"] - out["exact"]).max())
            for dt, out in output_by_mode.items()
        },
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
