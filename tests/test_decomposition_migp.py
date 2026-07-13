"""Tests for fastfuncstuff.decomposition.migp."""

from __future__ import annotations

import torch

from fastfuncstuff.decomposition.migp import migp_reduce


def _full_topk_subspace(runs, k, scale_by_n=True):
    """Reference: full concat + SVD → top-k right singular vectors (V, k)."""
    n = len(runs)
    s = (1.0 / n) if scale_by_n else 1.0
    full = torch.cat([r * s for r in runs], dim=0)  # (T_total, V)
    _, _, Vt = torch.linalg.svd(full, full_matrices=False)
    return Vt[:k].T  # (V, k) — basis of the principal voxel subspace


def _principal_angle_max(A: torch.Tensor, B: torch.Tensor) -> float:
    """Max principal angle between column spaces of (V, k) matrices A, B (radians)."""
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1.0, 1.0)
    # Smallest singular value → largest angle.
    return float(torch.arccos(s.min()).item())


def test_migp_preserves_topk_subspace_small():
    torch.manual_seed(0)
    T, V, k = 30, 200, 5
    # 4 runs, each with the same k underlying signal subspace + noise.
    base_signal = torch.randn(V, k) @ torch.randn(k, T) * 5.0  # (V, T)
    runs = []
    for _ in range(4):
        noise = torch.randn(V, T) * 0.5
        runs.append((base_signal + noise).T)  # (T, V)

    reduced = migp_reduce(runs, migp_n=10, migp_factor=1.0)
    # Top-k subspace from reduced (small T) and from full SVD reference.
    _, _, Vt_red = torch.linalg.svd(reduced, full_matrices=False)
    A = Vt_red[:k].T  # (V, k) from MIGP
    B = _full_topk_subspace(runs, k)  # (V, k) reference
    angle = _principal_angle_max(A, B)
    # Top-k subspace should align very tightly (well under 5 degrees).
    assert angle < 0.087, f"principal angle too large: {angle:.4f} rad"


def test_migp_default_dim():
    torch.manual_seed(1)
    runs = [torch.randn(20, 50) for _ in range(3)]
    reduced = migp_reduce(runs)
    # Default migp_n = 2*T_first - 1 = 39
    assert reduced.shape == (39, 50)


def test_migp_no_scale_matches_concat_when_no_reduction_needed():
    torch.manual_seed(2)
    runs = [torch.randn(5, 30) for _ in range(2)]
    reduced = migp_reduce(runs, migp_n=100, scale_by_n=False)
    # 2 runs × 5 = 10 rows; migp_n=100 → no reduction triggered, returns stack as-is.
    expected = torch.cat(runs, dim=0)
    assert reduced.shape == expected.shape
    assert torch.allclose(reduced, expected)


def test_migp_reduces_when_threshold_exceeded():
    torch.manual_seed(3)
    runs = [torch.randn(50, 80) for _ in range(5)]
    reduced = migp_reduce(runs, migp_n=20, migp_factor=1.0, scale_by_n=False)
    assert reduced.shape == (20, 80)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-x", "-v"])
