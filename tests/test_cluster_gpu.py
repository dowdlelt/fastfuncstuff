"""Batched connected-component extents (stats/cluster_gpu.py).

The reference is always ``cluster_fast``'s per-volume union-find: these are
two independent algorithms (sorted DSU vs Shiloach-Vishkin label propagation)
that must agree exactly, which is a far stronger check than either against
itself.  Most tests run the torch fallback on CPU so they need no card.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.cluster_fast import _offsets_for_nn, cluster_extent_one_perm
from fastfuncstuff.stats.cluster_gpu import build_neighbor_table, cluster_extent_batched

SIDEDS = ("1-sided", "2-sided", "bi-sided")
NNS = (1, 2, 3)


def _setup(seed=0, shape=(20, 18, 14), n_vol=6, device="cpu"):
    rng = np.random.default_rng(seed)
    g = [(np.arange(n) - (n - 1) / 2) / ((n - 1) / 2) for n in shape]
    gx, gy, gz = np.meshgrid(*g, indexing="ij")
    mask = (gx**2 + gy**2 + gz**2) < 0.9**2
    v = int(mask.sum())
    # Smooth a little so clusters are more than single voxels.
    fields = rng.standard_normal((n_vol, v)).astype(np.float32)
    tcrits = {s: np.array([2.2, 1.6, 1.1]) for s in SIDEDS}  # descending
    return mask, fields, tcrits, torch.device(device)


def _cpu_reference(mask, fields, tcrits, nns=NNS, sideds=SIDEDS):
    shape = mask.shape
    flat_idx = np.flatnonzero(mask.ravel()).astype(np.int64)
    offsets = {nn: _offsets_for_nn(nn) for nn in nns}
    n_tot = int(np.prod(shape))
    parent = np.full(n_tot, -1, dtype=np.int64)
    size = np.zeros(n_tot, dtype=np.int64)
    n_pthr = len(next(iter(tcrits.values())))
    out = {(s, nn): np.zeros((fields.shape[0], n_pthr), np.int64) for s in sideds for nn in nns}
    stat3d = np.zeros(shape, np.float32)
    for i in range(fields.shape[0]):
        stat3d[:] = 0.0
        stat3d[mask] = fields[i]
        res = cluster_extent_one_perm(
            stat3d, flat_idx, shape, nns, sideds, tcrits, offsets, parent, size
        )
        for k, v in res.items():
            out[k][i] = v
    return out


def _run_batched(mask, fields, tcrits, device, nns=NNS, sideds=SIDEDS):
    nbr = {nn: build_neighbor_table(mask, nn, device) for nn in nns}
    tc = {s: torch.from_numpy(v).to(device, torch.float32) for s, v in tcrits.items()}
    f = torch.from_numpy(fields).to(device)
    return cluster_extent_batched(f, nbr, nns, sideds, tc)


def test_matches_the_union_find_exactly():
    mask, fields, tcrits, device = _setup()
    got = _run_batched(mask, fields, tcrits, device)
    want = _cpu_reference(mask, fields, tcrits)
    for key in want:
        np.testing.assert_array_equal(got[key].cpu().numpy(), want[key], err_msg=str(key))


def test_matches_union_find_on_a_percolating_threshold():
    """A very loose threshold makes the suprathreshold set percolate, which is
    where label propagation would need geodesic-diameter rounds without
    pointer jumping."""
    mask, fields, _, device = _setup(seed=5)
    tcrits = {s: np.array([0.5, 0.0, -0.5]) for s in SIDEDS}
    got = _run_batched(mask, fields, tcrits, device)
    want = _cpu_reference(mask, fields, tcrits)
    for key in want:
        np.testing.assert_array_equal(got[key].cpu().numpy(), want[key], err_msg=str(key))
    # Sanity: the loosest threshold really is one huge blob.
    assert want[("1-sided", 3)][:, -1].max() > 0.5 * int(mask.sum())


def test_neighbour_table_is_symmetric_and_in_bounds():
    mask, _, _, device = _setup()
    v = int(mask.sum())
    for nn in NNS:
        nbr = build_neighbor_table(mask, nn, device).numpy()
        assert nbr.shape == (v, {1: 6, 2: 18, 3: 26}[nn])
        assert nbr.max() < v and nbr.min() >= -1
        # If a is b's neighbour then b is a's.
        for a in range(0, v, 37):
            for b in nbr[a]:
                if b >= 0:
                    assert a in nbr[b], f"NN{nn}: {a}->{b} not mirrored"


def test_components_never_span_volumes():
    """The batch is labelled as one flat array; two volumes must not merge.

    Made adversarial by giving every volume the identical field — if the
    adjacency leaked across rows, all rows would report one giant cluster.
    """
    mask, fields, tcrits, device = _setup(n_vol=1)
    repeated = np.repeat(fields, 5, axis=0)
    got = _run_batched(mask, repeated, tcrits, device)
    want = _cpu_reference(mask, repeated, tcrits)
    for key in want:
        arr = got[key].cpu().numpy()
        np.testing.assert_array_equal(arr, want[key])
        # Identical inputs must give identical answers, row by row.
        assert np.all(arr == arr[0])


def test_subset_of_sidedness_and_nn_is_honoured():
    mask, fields, tcrits, device = _setup()
    got = _run_batched(mask, fields, tcrits, device, nns=(2,), sideds=("bi-sided",))
    assert set(got) == {("bi-sided", 2)}
    want = _cpu_reference(mask, fields, tcrits, nns=(2,), sideds=("bi-sided",))
    np.testing.assert_array_equal(got[("bi-sided", 2)].cpu().numpy(), want[("bi-sided", 2)])


def test_empty_suprathreshold_set_yields_zero():
    mask, fields, _, device = _setup()
    tcrits = {s: np.array([99.0, 98.0]) for s in SIDEDS}
    got = _run_batched(mask, fields, tcrits, device)
    for v in got.values():
        assert int(v.sum()) == 0


@pytest.mark.gpu
def test_triton_kernels_match_the_torch_fallback():
    """The fused kernels are the whole point of the module; they must not
    drift from the reference they were written against."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    mask, fields, tcrits, _ = _setup(seed=9, shape=(28, 26, 20), n_vol=8)
    on_cpu = _run_batched(mask, fields, tcrits, torch.device("cpu"))
    on_gpu = _run_batched(mask, fields, tcrits, torch.device("cuda"))
    for key in on_cpu:
        np.testing.assert_array_equal(
            on_gpu[key].cpu().numpy(), on_cpu[key].numpy(), err_msg=str(key)
        )


@pytest.mark.gpu
def test_on_device_driver_matches_the_worker_pool():
    """The two clustering backends must agree on identical simulated fields."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastfuncstuff.stats.clustsim import ACF, simulate_cluster_null

    mask, _, _, _ = _setup(shape=(24, 24, 18))
    kw = dict(
        n_iter=48,
        pthr=(0.01, 0.001),
        device=torch.device("cuda"),
        seed=4,
        # Pinned: the batch plan reads *free* VRAM, so leaving it automatic
        # would let the two runs draw different fields and compare nothing.
        batch=16,
        verbose=False,
    )
    gpu = simulate_cluster_null(mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), on_device=True, **kw)
    cpu = simulate_cluster_null(
        mask, (3.0, 3.0, 3.0), ACF(0.5, 3.0, 4.0), on_device=False, n_jobs=2, **kw
    )
    for key in gpu.max_extent:
        np.testing.assert_array_equal(gpu.max_extent[key], cpu.max_extent[key], err_msg=str(key))
