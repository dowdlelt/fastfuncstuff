"""
Which null goes with which group-level statistic.

``ffs_group`` writes a bucket of (estimate, t) pairs from one of four models —
pooled OLS, Welch OLS, MEMA with one tau^2, MEMA with a tau^2 per set — and
every cluster method (ClustSim tables, ETAC) needs, for some of those t
bricks, a source of null z-maps produced by the *same* estimator.  This module
is the single map from (model, brick) to that source, so the inference code
never has to know which model it is thresholding.  See [[Group-level inference]].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from fastfuncstuff.stats.group_ttest import FITT, FIZT, Brick, t_to_z_np


@dataclass
class BrickNull:
    """The null of one t brick: bricks[brick] is its estimate, brick+1 its t."""

    label: str
    brick: int
    source: Any  # callable (start, nb) -> [nb, V] z, with n_iter, n_distinct, device
    main: bool = False


def brick_to_z(b: Brick) -> torch.Tensor:
    """The z with the same tail probability as a t (or z) brick."""
    if b.stat is None or b.stat[0] not in (FITT, FIZT):
        raise ValueError(f"{b.label} is not a t or z statistic")
    if b.stat[0] == FIZT:
        return b.data.float()
    z = t_to_z_np(b.data.detach().cpu().numpy(), b.stat[1][0])
    return torch.as_tensor(z, dtype=torch.float32, device=b.data.device)


def brick_nulls(
    design,
    *,
    model: str,
    ya: torch.Tensor,
    yb: torch.Tensor | None,
    ta: torch.Tensor | None = None,
    tb: torch.Tensor | None = None,
    n_iter: int,
    seed: int = 0,
    knha: bool = True,
    reml: str = "always",
    main_only: bool = False,
) -> list[BrickNull]:
    """A null for every t brick the model writes (just the main test if asked).

    ``ya``/``yb`` are [n, V] maps (MEMA: betas, with ``ta``/``tb`` their t).
    Brick indices follow :func:`group_ttest.run_ttest` /
    :func:`group_mema.run_mema` exactly; a test checks that.
    """
    if model == "ols":
        out = _ols_nulls(design, ya, yb, n_iter, seed)
    elif model == "mema":
        assert ta is not None
        out = _mema_nulls(design, ya, ta, yb, tb, n_iter, seed, knha, reml)
    else:
        raise ValueError(f"unknown model {model!r}")
    return [b for b in out if b.main] if main_only else out


def _ols_nulls(design, ya, yb, n_iter, seed) -> list[BrickNull]:
    from fastfuncstuff.stats.group_ttest import OlsNull, WelchNull, one_sample_matrix, t_tests

    if design.unpooled:
        assert yb is not None
        la, lb = design.label_a, design.label_b

        def one(y):
            x = one_sample_matrix(None, y.shape[0])
            return OlsNull(y.float(), x, np.array([1.0]), n_iter=n_iter, seed=seed)

        return [
            BrickNull(f"{la}-{lb}_mean", 0, WelchNull(ya.float(), yb.float(), n_iter, seed), True),
            BrickNull(f"{la}_mean", 2, one(ya)),
            BrickNull(f"{lb}_mean", 4, one(yb)),
        ]
    out = []
    for k, sp in enumerate(t_tests(design, ya, yb)):
        src = OlsNull(sp.y.float(), sp.x, sp.c, n_iter=n_iter, seed=seed, permute=sp.permute)
        out.append(BrickNull(sp.label, sp.brick, src, main=(k == 0)))
    return out


def _mema_nulls(design, ba, ta, bb, tb, n_iter, seed, knha, reml) -> list[BrickNull]:
    from fastfuncstuff.stats.group_mema import (
        TOL_U,
        MemaNull,
        mema_covariates,
        mema_design,
        variance_from_t,
    )

    la, lb = design.label_a, design.label_b
    va = variance_from_t(ba, ta)
    kw = dict(n_iter=n_iter, seed=seed, knha=knha, reml=reml)
    if design.unpooled:
        assert bb is not None and tb is not None
        y = torch.cat([ba, bb])
        v = torch.cat([va, variance_from_t(bb, tb)])
        na, nb = design.n_a, design.n_b

        def un(out, dof):
            return MemaNull(y, v, None, out, float(dof), n_a=na, **kw)

        return [
            BrickNull(la, 0, un(0, na - 1)),
            BrickNull(lb, 2, un(1, nb - 1)),
            BrickNull(f"{la}-{lb}", 4, un(2, na + nb - 2), True),
        ]
    cov = mema_covariates(design, design.center)
    if not design.two_sets:
        y, v = ba, va
        x = mema_design(design.n_a, 0, cov)
        heads = [(la, 0)]
    elif design.paired:
        assert bb is not None and tb is not None
        v = va + variance_from_t(bb, tb)
        y = torch.where(v >= TOL_U, torch.zeros_like(ba), ba - bb)
        x = mema_design(design.n_a, 0, cov)
        heads = [(f"{la}-{lb}", 0)]
    else:
        assert bb is not None and tb is not None
        y = torch.cat([ba, bb])
        v = torch.cat([va, variance_from_t(bb, tb)])
        x = mema_design(design.n_a, design.n_b, cov)
        heads = [(la, 0), (lb, 2), (f"{la}-{lb}", 4)]
    p = x.shape[1]
    e = np.eye(p)
    dof = float(x.shape[0] - (2 if design.two_sets and not design.paired else 1) - design.q)
    cvec = {la: e[0]}
    if design.two_sets and not design.paired:
        cvec = {la: e[0], lb: e[0] + e[1], f"{la}-{lb}": -e[1]}
    elif design.paired:
        cvec = {f"{la}-{lb}": e[0]}
    main_label = heads[-1][0] if design.two_sets else la
    out = [
        BrickNull(lab, idx, MemaNull(y, v, x, cvec[lab], dof, **kw), lab == main_label)
        for lab, idx in heads
    ]
    k0 = len(heads) * 2
    j0 = 2 if design.two_sets and not design.paired else 1
    for j, nm in enumerate(design.cov_names):
        out.append(BrickNull(nm, k0 + 2 * j, MemaNull(y, v, x, e[j0 + j], dof, **kw)))
    return out
