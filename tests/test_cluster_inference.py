"""Shared cluster-inference driver (stats/cluster_inference.py).

The driver must give the same null as the CPU Numba path ffs_perm used before
it was routed here — extent exactly, mass to float-summation tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.cluster import accumulate_cluster_null
from fastfuncstuff.stats.cluster_inference import max_fom_null, rows_source

SIDEDS = ("1-sided", "2-sided", "bi-sided")


def _null(seed=0, n=12, shape=(18, 16, 12)):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    g = [(np.arange(k) - (k - 1) / 2) / ((k - 1) / 2) for k in shape]
    gx, gy, gz = np.meshgrid(*g, indexing="ij")
    mask = (gx**2 + gy**2 + gz**2) < 0.95**2
    t = np.zeros((n, int(mask.sum())), np.float32)
    for i in range(n):
        x = gaussian_filter(rng.standard_normal(shape), 1.0)
        t[i] = (x / x[mask].std())[mask]
    return mask, t


def test_driver_matches_numba_extent_and_cc3d_mass():
    mask, t = _null()
    pthr = (0.05, 0.01)
    dof = 30
    ref_ext = accumulate_cluster_null(
        t, mask, dof, nns=(1, 2), sideds=SIDEDS, pthr=pthr, verbose=False, fast=True, n_jobs=1
    )
    ref_mass = accumulate_cluster_null(
        t, mask, dof, nns=(1, 2), sideds=SIDEDS, pthr=pthr, verbose=False, fast=False, n_jobs=1
    )
    from fastfuncstuff.stats.cluster import _t_critical

    tcrits = {s: np.array([_t_critical(p, dof, s) for p in pthr]) for s in SIDEDS}
    cpu = torch.device("cpu")
    got = max_fom_null(
        rows_source(t, cpu),
        t.shape[0],
        mask,
        tcrits,
        nns=(1, 2),
        sideds=SIDEDS,
        hpows=(0, 1),
        device=cpu,
        batch=5,  # forces several batches, including a ragged last one
        verbose=False,
    )
    for s in SIDEDS:
        for nn in (1, 2):
            np.testing.assert_array_equal(
                got[(s, nn, 0)].astype(np.int64), ref_ext.max_extent[(s, nn)]
            )
            np.testing.assert_allclose(got[(s, nn, 1)], ref_mass.max_mass[(s, nn)], rtol=1e-5)


def test_columns_follow_caller_threshold_order():
    mask, t = _null(seed=1, n=4)
    cpu = torch.device("cpu")
    kw = dict(nns=(2,), sideds=("2-sided",), device=cpu, verbose=False)
    asc = max_fom_null(rows_source(t, cpu), 4, mask, {"2-sided": np.array([1.5, 2.5])}, **kw)
    desc = max_fom_null(rows_source(t, cpu), 4, mask, {"2-sided": np.array([2.5, 1.5])}, **kw)
    np.testing.assert_array_equal(asc[("2-sided", 2, 0)], desc[("2-sided", 2, 0)][:, ::-1])


@pytest.mark.gpu
def test_cuda_null_matches_cpu_numba():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    mask, t = _null(seed=2)
    kw = dict(nns=(1, 2, 3), sideds=SIDEDS, pthr=(0.05, 0.01, 0.001), verbose=False, n_jobs=1)
    a = accumulate_cluster_null(t, mask, 25, fast=True, **kw)
    b = accumulate_cluster_null(t, mask, 25, fast=True, device=torch.device("cuda"), **kw)
    for k in a.max_extent:
        np.testing.assert_array_equal(a.max_extent[k], b.max_extent[k])
