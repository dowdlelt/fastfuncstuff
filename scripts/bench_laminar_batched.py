"""Time the batched laminar inversion across batch widths, CPU against GPU.

The published Planum Polare data (the test oracle) replicated with noise as B
"columns", one model inverted against all of them. Reports seconds per VL
iteration (one integration of B x (np+1) probe rows plus the batched E/M-step)
and, with --converge, a full run's wall time and seconds per fit.

    python scripts/bench_laminar_batched.py --device cpu --widths 1 8 32
    python scripts/bench_laminar_batched.py --device cuda --widths 8 64 256 1024 --converge

Compile is per (K, width) and is excluded: each width is warmed up with a
two-iteration run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def _problem():
    from fastfuncstuff.laminar.params import P0, ModelSpec

    o = json.loads((ROOT / "tests/laminar_oracle/laminar_modelspace_oracle.json").read_text())
    block = o["per_k"][1]
    K, ns = int(block["K"]), int(block["ns"])
    spec = ModelSpec(N=int(o["N"]), K=K, n_inputs=2, n_mod=1, dt=float(o["TR"]) / 32.0,
                     TR=float(o["TR"]), p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5))  # fmt: skip

    def t(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)

    return (spec, t(block["y"]).reshape(ns, K), t(block["u"]).reshape(-1, 2),
            t(block["mask_rows"]).reshape(-1).to(torch.bool), t(o["kernel"]).reshape(-1))  # fmt: skip


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=4, help="iterations timed per width")
    ap.add_argument("--converge", action="store_true", help="also time a full run")
    ap.add_argument("--target", type=int, nargs=3, default=[1, 0, 0])
    args = ap.parse_args()

    from fastfuncstuff.laminar.batched import variational_laplace_batched
    from fastfuncstuff.laminar.experiment import faes_priors

    device = torch.device(args.device)
    spec, y, u, rows, kernel = _problem()
    pr = faes_priors(spec, tuple(args.target))
    g = torch.Generator().manual_seed(0)
    print(f"device {device}, K {spec.K}, scans {y.shape[0]}, microsteps "
          f"{int(round(y.shape[0] * spec.TR / spec.dt))}, model {tuple(args.target)}")  # fmt: skip
    for B in args.widths:
        Y = y + 0.05 * torch.randn((B, *y.shape), generator=g, dtype=torch.float64)
        kw = dict(rows=rows, kernel=kernel, device=device)
        variational_laplace_batched(spec, pr, u, Y, max_iter=2, **kw)  # compile this width
        _sync(device)
        t0 = time.perf_counter()
        variational_laplace_batched(spec, pr, u, Y, max_iter=args.iters, **kw)
        _sync(device)
        per_it = (time.perf_counter() - t0) / args.iters
        line = (
            f"  B {B:6d}: {per_it:8.3f} s/iteration  ({1e3 * per_it / B:8.2f} ms per fit-iteration)"
        )
        if args.converge:
            t0 = time.perf_counter()
            res = variational_laplace_batched(spec, pr, u, Y, **kw)
            _sync(device)
            wall = time.perf_counter() - t0
            line += (f" | full {wall:8.1f} s, {wall / B:8.3f} s/fit, iters "
                     f"{int(res.n_iter.min())}-{int(res.n_iter.max())}, failed {int(res.failed.sum())}")  # fmt: skip
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
