"""Empirical vascular priors across columns: shrink when columns agree, not when they differ."""

import numpy as np
import torch

from fastfuncstuff.laminar.batched import BatchedVL
from fastfuncstuff.laminar.empirical import _flat_index, empirical_vascular_prior, neighbourhoods
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.inversion import vec
from fastfuncstuff.laminar.params import ModelSpec


def _posteriors(s_d_true, sd_post=0.3, seed=0):
    """A fake full-model inversion per column: posterior over every free parameter
    with s_d observed around s_d_true with posterior sd ``sd_post``."""
    spec = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
    pr = faes_priors(spec, (1, 1, 1))
    free = torch.nonzero(vec(pr.pC) > 0).squeeze(-1)
    n, C = free.numel(), len(s_d_true)
    rng = np.random.default_rng(seed)
    Cp = torch.diag(0.1 * vec(pr.pC)[free]).expand(C, n, n).clone()
    i_sd = int(torch.nonzero(free == _flat_index(spec, "s_d"))[0])
    Cp[:, i_sd, i_sd] = sd_post**2
    Ep = vec(pr.pE).expand(C, -1).clone()
    Ep[:, _flat_index(spec, "s_d")] = torch.as_tensor(s_d_true + sd_post * rng.standard_normal(C))
    z = torch.zeros(C, dtype=torch.float64)
    run = BatchedVL(Ep_vec=Ep, Cp=Cp, Eh=torch.zeros(C, 6, dtype=torch.float64), F=z.clone(),
                    L=torch.zeros(C, 3, dtype=torch.float64),
                    y_pred=torch.zeros(C, 4, 6, dtype=torch.float64), free_idx=free,
                    n_iter=torch.ones(C, dtype=torch.long), converged=z.bool() | True,
                    stalled=z.bool(), failed=z.bool(), spec=spec)  # fmt: skip
    return run, pr, spec


def _grid_centres(n):
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2) * 3.0
    return np.c_[g, np.zeros(len(g))], np.zeros(len(g), int)


def test_columns_that_agree_are_shrunk_toward_their_neighbours():
    centres, hemi = _grid_centres(8)
    run, pr, spec = _posteriors(np.full(len(centres), -0.4))
    ep = empirical_vascular_prior(run, pr, centres, hemi, radius_mm=7.0)
    assert ep.shrink < 0.1  # a shared value: the evidence wants a tight prior
    i = _flat_index(spec, "s_d")
    noisy = run.Ep_vec[:, i].numpy()
    assert np.std(ep.mean[:, 0]) < 0.5 * np.std(noisy)
    assert abs(np.mean(ep.mean[:, 0]) - (-0.4)) < 0.1


def test_columns_that_truly_differ_keep_a_wide_prior():
    centres, hemi = _grid_centres(8)
    rng = np.random.default_rng(3)
    run, pr, _ = _posteriors(rng.normal(0, 1.5, len(centres)), sd_post=0.1)
    ep = empirical_vascular_prior(run, pr, centres, hemi, radius_mm=7.0)
    assert ep.shrink >= 0.3


def test_neighbourhoods_respect_radius_and_hemisphere():
    centres = np.array([[0, 0, 0], [5, 0, 0], [20, 0, 0], [1, 0, 0.0]])
    hemi = np.array([0, 0, 0, 1])
    nb = neighbourhoods(centres, hemi, 6.0)
    assert set(nb[0]) == {0, 1} and set(nb[2]) == {2} and set(nb[3]) == {3}
    assert set(neighbourhoods(centres, hemi, None)[0]) == {0, 1, 2}
