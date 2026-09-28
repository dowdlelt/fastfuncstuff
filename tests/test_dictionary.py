"""Sparse dictionary learning (decomposition/dictionary.py)."""

import numpy as np
import torch

from fastfuncstuff.decomposition.dictionary import dictionary_learning

CPU = torch.device("cpu")


def _sparse_problem(T=120, V=3000, K=5, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(T, K, generator=g)
    D = D / D.norm(dim=0)
    S = torch.zeros(K, V)
    # every voxel uses one or two atoms: overlapping, non-orthogonal "maps"
    first = torch.randint(K, (V,), generator=g)
    second = torch.randint(K, (V,), generator=g)
    use2 = torch.rand(V, generator=g) < 0.4
    S[first, torch.arange(V)] = torch.randn(V, generator=g).abs() + 0.5
    S[second[use2], torch.arange(V)[use2]] += torch.randn(int(use2.sum()), generator=g)
    X = D @ S + noise * torch.randn(T, V, generator=g)
    return X, D, S


def _match(A, B):
    """Best |corr| of each true column in B against the learned columns in A."""
    C = (A / A.norm(dim=0)).T @ (B / B.norm(dim=0))
    return C.abs().max(dim=0).values


def test_recovers_sparse_generative_atoms():
    X, D, _ = _sparse_problem()
    res = dictionary_learning(X, 5, alpha=0.05, n_iter=100, device=CPU)
    assert _match(res.atoms, D).min() > 0.97
    # codes are genuinely sparse: most voxels use only a couple of the 5 atoms
    assert res.density < 0.6
    assert torch.allclose(res.atoms.norm(dim=0), torch.ones(5), atol=1e-4)
    assert torch.all(res.energy[:-1] >= res.energy[1:])


def test_pca_does_not_recover_the_same_atoms():
    # The point of the method: overlapping sparse sources are not the principal axes.
    X, D, _ = _sparse_problem()
    U = torch.linalg.svd(X, full_matrices=False)[0][:, :5]
    assert _match(U, D).min() < 0.9


def test_near_zero_sparsity_spans_the_pca_subspace():
    X, _, _ = _sparse_problem(noise=0.3)
    res = dictionary_learning(X, 5, alpha=1e-6, n_iter=5, device=CPU)
    U = torch.linalg.svd(X, full_matrices=False)[0][:, :5]
    cos = torch.linalg.svdvals(U.T @ torch.linalg.qr(res.atoms)[0])
    assert cos.min() > 0.999


def test_chunked_streaming_matches_resident():
    X, _, _ = _sparse_problem(V=1000)
    a = dictionary_learning(X, 4, alpha=0.1, n_iter=10, device=CPU, chunk_size=10_000)
    b = dictionary_learning(X, 4, alpha=0.1, n_iter=10, device=CPU, chunk_size=97)
    np.testing.assert_allclose(a.atoms.abs().numpy(), b.atoms.abs().numpy(), atol=1e-4)
    np.testing.assert_allclose(a.codes.numpy(), b.codes.numpy(), atol=1e-4)
