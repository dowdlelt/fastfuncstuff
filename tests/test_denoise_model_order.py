"""ffs_denoise's model-order count (the ffs_ica estimator applied per run to the pool)."""

import torch

from fastfuncstuff.denoise.sequential import estimate_noise_model_order_per_run

CPU = torch.device("cpu")


def _pool(rank, T=150, V=6000, n_runs=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    runs = []
    for _ in range(n_runs):
        S = torch.randn(T, rank, generator=g) * torch.linspace(20, 10, rank)
        M = torch.randn(rank, V, generator=g)
        runs.append(S @ M / V**0.5 + torch.randn(T, V, generator=g))
    return torch.cat(runs, 0).T.contiguous(), [r * T for r in range(n_runs)]


def test_recovers_the_planted_rank_in_white_noise():
    data, starts = _pool(rank=6)
    est = estimate_noise_model_order_per_run(
        data,
        starts,
        torch.ones(data.shape[0], dtype=torch.bool),
        resels_per_run=[1.0, 1.0],
        min_components=1,
        max_components=60,
        device=CPU,
    )
    assert all(abs(k - 6) <= 1 for k in est.per_run_caps), est.per_run_caps


def test_pure_noise_gives_the_floor():
    data, starts = _pool(rank=1, seed=3)
    data = torch.randn_like(data)
    est = estimate_noise_model_order_per_run(
        data,
        starts,
        torch.ones(data.shape[0], dtype=torch.bool),
        resels_per_run=[1.0, 1.0],
        min_components=1,
        max_components=60,
        device=CPU,
    )
    assert max(est.per_run_caps) <= 2, est.per_run_caps
