"""Sparse dictionary learning: temporal atoms with sparse spatial maps.

    min_{D, S}  1/2 ||X - D S||_F^2 + lam ||S||_1      s.t. ||d_k||_2 <= 1

X is (T, V): time by voxels. D (T, K) holds the temporal atoms, S (K, V) the sparse spatial
"maps" (codes). Unlike PCA/ICA the maps may overlap and are not orthogonal or independent;
each voxel is explained by a few atoms.

Alternating minimisation, full batch over voxels (chunked through the memory module):
  codes       FISTA on each voxel's code, warm-started; batched as (K, V) matmuls
  dictionary  block coordinate descent on the atoms (Mairal et al. 2010, Alg. 2), from the
              sufficient statistics A = S S^T and B = X S^T
The atoms start at the top-K principal components, so K atoms with lam -> 0 reproduce the
PCA subspace. ``alpha`` sets lam as a fraction of lam_max = max |D0^T X|, the value at which
every code would be zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from fastfuncstuff.memory import estimate_chunk_size


@dataclass
class DictionaryResult:
    atoms: Tensor  # (T, K) temporal atoms, unit norm, ordered by energy
    codes: Tensor  # (K, V) sparse spatial maps, same order
    energy: Tensor  # (K,) fraction of the reconstruction energy per atom
    lam: float
    n_iter: int
    objective: list[float] = field(default_factory=list)

    @property
    def density(self) -> float:
        """Fraction of nonzero code entries."""
        return float((self.codes != 0).float().mean())


def _soft(x: Tensor, thr: float) -> Tensor:
    return torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)


def _fista(D: Tensor, Gm: Tensor, L: float, xc: Tensor, Sc: Tensor, lam: float, n: int) -> Tensor:
    """``n`` FISTA steps on min 1/2||xc - D S||^2 + lam||S||_1, warm-started at ``Sc``."""
    Bx = D.T @ xc
    Y, t_k = Sc.clone(), 1.0
    for _ in range(n):
        S_new = _soft(Y - (Gm @ Y - Bx) / L, lam / L)
        t_new = (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5) / 2.0
        Y = S_new + ((t_k - 1.0) / t_new) * (S_new - Sc)
        Sc, t_k = S_new, t_new
    return Sc


def _top_components(X: Tensor, k: int, chunk: int) -> Tensor:
    """Top-k left singular vectors of the wide (T, V) matrix via its (T, T) Gram."""
    T, V = X.shape
    G = torch.zeros(T, T, dtype=torch.float64, device=X.device if chunk >= V else "cpu")
    for s in range(0, V, chunk):
        xc = X[:, s : s + chunk]
        G += (xc @ xc.T).to(G.device, torch.float64)
    _, U = torch.linalg.eigh(G)
    return U[:, -k:].flip(1).to(torch.float32)


def dictionary_learning(
    X: Tensor,
    n_atoms: int,
    alpha: float = 0.1,
    n_iter: int = 50,
    fista_iter: int = 25,
    tol: float = 1e-4,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> DictionaryResult:
    """Learn ``n_atoms`` temporal atoms and sparse spatial codes for ``X`` (T, V).

    ``X`` may live on the CPU; voxel chunks stream to ``device`` when it does not fit.
    """
    device = device or X.device
    T, V = X.shape
    K = int(n_atoms)
    if not 1 <= K <= min(T, V):
        raise ValueError(f"n_atoms must be in [1, {min(T, V)}], got {K}")
    if chunk_size is None:
        chunk_size = estimate_chunk_size(V, T, 5 * K, device)
    chunk = max(1, min(int(chunk_size), V))
    resident = chunk >= V
    Xd = X.to(device, torch.float32) if resident else X
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def chunks():
        for s in range(0, V, chunk):
            xc = Xd[:, s : s + chunk]
            yield s, (xc if resident else xc.to(device, torch.float32))

    D = _top_components(Xd if resident else X, K, chunk).to(device)
    lam_max = max(float((D.T @ xc).abs().max()) for _, xc in chunks())
    lam = float(alpha) * lam_max
    S = torch.zeros(K, V, device=device)
    objective: list[float] = []
    it = 0
    for it in range(1, n_iter + 1):
        # ---- sparse coding (FISTA), warm start
        Gm = D.T @ D
        L = float(torch.linalg.eigvalsh(Gm).max().clamp_min(1e-8))
        A = torch.zeros(K, K, device=device)
        B = torch.zeros(T, K, device=device)
        sq_err = 0.0
        l1 = 0.0
        for s, xc in chunks():
            Sc = _fista(D, Gm, L, xc, S[:, s : s + xc.shape[1]], lam, fista_iter)
            S[:, s : s + xc.shape[1]] = Sc
            A += Sc @ Sc.T
            B += xc @ Sc.T
            sq_err += float(((xc - D @ Sc) ** 2).sum())
            l1 += float(Sc.abs().sum())
        objective.append(0.5 * sq_err + lam * l1)
        # ---- dictionary update (block coordinate descent)
        for k in range(K):
            if A[k, k] < 1e-10:
                # dead atom: restart on a random voxel's residual
                v = int(torch.randint(V, (1,), generator=gen))
                x = (Xd if resident else X)[:, v].to(device, torch.float32)
                r = x - D @ S[:, v]
                D[:, k] = r / r.norm().clamp_min(1e-8)
                continue
            u = D[:, k] + (B[:, k] - D @ A[:, k]) / A[k, k]
            D[:, k] = u / max(1.0, float(u.norm()))
        if verbose:
            print(
                f"  dictionary iter {it:3d}: objective {objective[-1]:.5g}, code density {float((S != 0).float().mean()):.3f}"
            )
        if len(objective) > 1 and abs(objective[-2] - objective[-1]) <= tol * abs(objective[-2]):
            break
    # Final codes for the final dictionary, then order atoms by reconstruction energy.
    Gm = D.T @ D
    L = float(torch.linalg.eigvalsh(Gm).max().clamp_min(1e-8))
    for s, xc in chunks():
        S[:, s : s + xc.shape[1]] = _fista(D, Gm, L, xc, S[:, s : s + xc.shape[1]], lam, fista_iter)
    norms = D.norm(dim=0).clamp_min(1e-8)
    D = D / norms
    S = S * norms[:, None]
    energy = (S**2).sum(dim=1)
    order = torch.argsort(energy, descending=True)
    energy = energy[order] / energy.sum().clamp_min(1e-12)
    return DictionaryResult(
        atoms=D[:, order], codes=S[order], energy=energy, lam=lam, n_iter=it, objective=objective
    )
