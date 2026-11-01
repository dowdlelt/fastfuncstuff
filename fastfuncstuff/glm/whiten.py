"""Run-block prewhitening: L⁻¹ of a run-block-diagonal correlation matrix, per run.

Concatenated runs are independent realisations, so the noise correlation R is
block-diagonal by run and so are its Cholesky factor and L⁻¹. Stored densely, L⁻¹
for T = Σ n_r timepoints costs T² per (a, b) and T²·V to apply: quadratic in the
number of runs, almost all of it multiplying zeros. Stored per run it costs
Σ n_r² and Σ n_r²·V — linear in runs, and the same numbers.

Without censoring every run's block is a leading principal submatrix of ONE
Toeplitz R of the longest run's length. The Cholesky factor of a leading block is
the leading block of the factor, and likewise its inverse, so a single max-length
L⁻¹ per (a, b) serves every run: run r uses ``L⁻¹[:n_r, :n_r]``. Runs of equal
length then go through one fused GEMM. With censoring (``tau``) the blocks are no
longer Toeplitz and each run carries its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from fastfuncstuff import policy_linalg as plinalg


def run_bounds(n: int, run_starts: Sequence[int] | Tensor | None) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` for each run of an ``n``-timepoint series."""
    if run_starts is None:
        return [(0, n)]
    starts = [
        int(s) for s in (run_starts.tolist() if isinstance(run_starts, Tensor) else run_starts)
    ]
    starts = sorted(set(starts))
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    stops = [*starts[1:], n]
    return [(s, e) for s, e in zip(starts, stops, strict=True) if e > s]


class _Group:
    """Runs whitened by one shared (m, m) block.

    ``span`` is set when the runs tile a contiguous stretch in order, so the
    data can be viewed as (n_runs, m, ·) with no gather."""

    __slots__ = ("block", "bounds", "index", "span")

    def __init__(self, block: Tensor, bounds: list[tuple[int, int]]):
        self.block = block
        self.bounds = bounds
        contiguous = all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
        self.span = (bounds[0][0], bounds[-1][1]) if contiguous else None
        self.index = (
            None
            if contiguous
            else torch.cat([torch.arange(s, e, device=block.device) for s, e in bounds])
        )

    def to(self, device, dtype=None) -> _Group:
        return _Group(self.block.to(device=device, dtype=dtype or self.block.dtype), self.bounds)


class RunWhitener:
    """Apply L⁻¹ of a run-block-diagonal R without forming the T×T matrix.

    ``cols(M)`` is ``L⁻¹ @ M`` for M (T, k); ``rows(Y)`` is ``Y @ L⁻ᵀ`` for Y (V, T)
    — the two layouts the REML code whitens design and data in.
    """

    def __init__(self, groups: list[_Group], n: int):
        self.groups = groups
        self.n = n

    @property
    def device(self) -> torch.device:
        return self.groups[0].block.device

    @property
    def dtype(self) -> torch.dtype:
        return self.groups[0].block.dtype

    @property
    def nbytes(self) -> int:
        # Blocks that are views of one max-length factor share storage.
        seen: dict[int, int] = {}
        for g in self.groups:
            st = g.block.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        return sum(seen.values())

    def to(self, device, dtype=None) -> RunWhitener:
        if device == self.device and (dtype is None or dtype == self.dtype):
            return self
        # Move a shared parent storage once, then re-slice, so per-run views of
        # one max-length factor stay one tensor on the target device.
        moved: dict[int, Tensor] = {}
        groups = []
        for g in self.groups:
            base = g.block
            parent = base._base if base._base is not None else base
            key = parent.untyped_storage().data_ptr()
            if key not in moved:
                moved[key] = parent.to(device=device, dtype=dtype or parent.dtype)
            m = base.shape[0]
            p = moved[key]
            block = p[:m, :m] if p.shape[-1] != m else p
            groups.append(_Group(block, g.bounds))
        return RunWhitener(groups, self.n)

    def cols(self, M: Tensor) -> Tensor:
        """``L⁻¹ @ M`` for M of shape (T, k) (or (T,))."""
        squeeze = M.ndim == 1
        if squeeze:
            M = M.unsqueeze(1)
        out = torch.empty_like(M)
        k = M.shape[1]
        for g in self.groups:
            m, nr = g.block.shape[0], len(g.bounds)
            B = g.block.to(dtype=M.dtype)
            if g.span is not None:
                s, e = g.span
                # (m, m) @ (m, nr·k): runs side by side as one GEMM
                Mg = M[s:e].reshape(nr, m, k).permute(1, 0, 2).reshape(m, nr * k)
                out[s:e] = (B @ Mg).reshape(m, nr, k).permute(1, 0, 2).reshape(nr * m, k)
            else:
                assert g.index is not None
                idx = g.index.to(M.device)
                Mg = M.index_select(0, idx).reshape(nr, m, k).permute(1, 0, 2).reshape(m, nr * k)
                out[idx] = (B @ Mg).reshape(m, nr, k).permute(1, 0, 2).reshape(nr * m, k)
        return out.squeeze(1) if squeeze else out

    def rows(self, Y: Tensor) -> Tensor:
        """``Y @ L⁻ᵀ`` for Y of shape (V, T): each row whitened."""
        out = torch.empty_like(Y)
        v = Y.shape[0]
        for g in self.groups:
            m, nr = g.block.shape[0], len(g.bounds)
            Bt = g.block.to(dtype=Y.dtype).T
            if g.span is not None:
                s, e = g.span
                out[:, s:e] = (Y[:, s:e].reshape(v, nr, m) @ Bt).reshape(v, nr * m)
            else:
                assert g.index is not None
                idx = g.index.to(Y.device)
                Yg = Y.index_select(1, idx).reshape(v, nr, m)
                out[:, idx] = (Yg @ Bt).reshape(v, nr * m)
        return out

    def dense(self) -> Tensor:
        """The T×T L⁻¹ — for tests and debugging only."""
        eye = torch.eye(self.n, device=self.device, dtype=self.dtype)
        return self.cols(eye)


def stack_cols(whiteners: Sequence[RunWhitener], M: Tensor) -> Tensor:
    """``[W.cols(M) for W in whiteners]`` as one (n_w, T, k) tensor.

    Whiteners built together share a group layout, so each group runs as one
    batched GEMM across all of them rather than one GEMM per whitener."""
    n_w = len(whiteners)
    T, k = M.shape
    first = whiteners[0]
    same_layout = all(
        len(w.groups) == len(first.groups)
        and all(a.bounds == b.bounds for a, b in zip(w.groups, first.groups, strict=True))
        for w in whiteners
    )
    if not same_layout:
        return torch.stack([w.cols(M) for w in whiteners])
    out = torch.empty((n_w, T, k), device=M.device, dtype=M.dtype)
    for gi, g in enumerate(first.groups):
        m, nr = g.block.shape[0], len(g.bounds)
        B = torch.stack([w.groups[gi].block for w in whiteners]).to(M.device, M.dtype)
        if g.span is not None:
            s, e = g.span
            Mg = M[s:e].reshape(nr, m, k).permute(1, 0, 2).reshape(m, nr * k)
            res = torch.matmul(B, Mg)  # (n_w, m, nr·k)
            out[:, s:e] = res.reshape(n_w, m, nr, k).permute(0, 2, 1, 3).reshape(n_w, nr * m, k)
        else:
            assert g.index is not None
            idx = g.index.to(M.device)
            Mg = M.index_select(0, idx).reshape(nr, m, k).permute(1, 0, 2).reshape(m, nr * k)
            res = torch.matmul(B, Mg)
            out[:, idx] = res.reshape(n_w, m, nr, k).permute(0, 2, 1, 3).reshape(n_w, nr * m, k)
    return out


def whitener_elements(n: int, run_starts=None, tau: Tensor | None = None) -> int:
    """Stored elements per (a, b): the longest run's m² without censoring
    (every run shares its leading blocks), else Σ n_r². Memory planning uses
    this where it used to use T²."""
    lens = [e - s for s, e in run_bounds(n, run_starts)]
    return max(lens) ** 2 if tau is None else sum(m * m for m in lens)


def arma11_whiteners(
    a_grid: Tensor,
    b_grid: Tensor,
    n: int,
    *,
    run_starts=None,
    tau: Tensor | None = None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[RunWhitener], Tensor, list[tuple[float, float]]]:
    """One :class:`RunWhitener` per valid (a, b) of the grid, plus logdet(R).

    Validity filtering is ``build_arma11_covariance_batch``'s, so the pair list
    is the one the dense path produced. The Cholesky runs on ``device``.

    Returns ``(whiteners, logdet_R (n_valid,), param_list)``.
    """
    from fastfuncstuff.glm.arma import build_arma11_covariance_batch

    bounds = run_bounds(n, run_starts)
    lens = [e - s for s, e in bounds]
    eye_cache: dict[int, Tensor] = {}

    def factor(R: Tensor) -> tuple[Tensor, Tensor]:
        L = plinalg.cholesky(R)
        # +1e-10 as everywhere else in the REML code, so logdets match the dense path.
        logdiag = torch.log(torch.diagonal(L, dim1=1, dim2=2) + 1e-10)
        m = R.shape[-1]
        if m not in eye_cache:
            eye_cache[m] = torch.eye(m, device=R.device, dtype=R.dtype)
        L_inv = plinalg.solve_triangular(L, eye_cache[m].expand(R.shape[0], -1, -1), upper=False)
        return L_inv, logdiag

    if tau is None:
        m_max = max(lens)
        R, _, param_list = build_arma11_covariance_batch(a_grid, b_grid, m_max, device, dtype)
        if not param_list:
            return [], torch.empty(0, device=device, dtype=dtype), []
        L_inv, logdiag = factor(R)
        del R
        cum = torch.cumsum(logdiag, dim=1)  # logdet of every leading block at once
        logdet = 2 * torch.stack([cum[:, m - 1] for m in lens]).sum(dim=0)
        by_len: dict[int, list[tuple[int, int]]] = {}
        for b in bounds:
            by_len.setdefault(b[1] - b[0], []).append(b)
        whiteners = []
        for i in range(len(param_list)):
            # Own storage per pair (not a view of the batch), so moving one
            # whitener moves m_max² elements, not the whole grid.
            own = L_inv[i].clone()
            groups = [_Group(own if m == m_max else own[:m, :m], bs) for m, bs in by_len.items()]
            whiteners.append(RunWhitener(groups, n))
        return whiteners, logdet, param_list

    # Censored: each run's block follows its own tau, so no sharing.
    per_run = []
    param_list: list = []
    logdet = None
    for s, e in bounds:
        R, _, param_list = build_arma11_covariance_batch(
            a_grid, b_grid, e - s, device, dtype, tau=tau[s:e]
        )
        if not param_list:
            return [], torch.empty(0, device=device, dtype=dtype), []
        L_inv, logdiag = factor(R)
        del R
        per_run.append(L_inv)
        ld = 2 * logdiag.sum(dim=1)
        logdet = ld if logdet is None else logdet + ld
    assert logdet is not None
    whiteners = [
        RunWhitener([_Group(per_run[r][i].clone(), [b]) for r, b in enumerate(bounds)], n)
        for i in range(len(param_list))
    ]
    return whiteners, logdet, param_list


def arma11_whitener(
    a: float,
    b: float,
    n: int,
    *,
    run_starts=None,
    tau: Tensor | None = None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> RunWhitener:
    """The whitener for one (a, b) — the GLS step after the search has chosen it."""
    ws, _, _ = arma11_whiteners(
        torch.tensor([a], dtype=dtype, device=device),
        torch.tensor([b], dtype=dtype, device=device),
        n,
        run_starts=run_starts,
        tau=tau,
        device=device,
        dtype=dtype,
    )
    if not ws:
        raise ValueError(f"Invalid ARMA(1,1) parameters: a={a}, b={b}")
    return ws[0]
