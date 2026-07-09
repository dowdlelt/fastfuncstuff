"""Preprocessing for the BSDS ROI-timeseries data contract.

A *session* is a ``(D, N)`` array (``D`` ROIs by ``N`` timepoints). BSDS models
the covariance structure across ROIs within each latent state, so the standard
preparation is, per session and per ROI: remove low-order drift (orthogonal
Legendre polynomials, never raw monomials — see the ``[[Legendre polynomials]]``
principle) and standardise each ROI so that a state's covariance reads as a
functional-connectivity pattern rather than being dominated by amplitude scale.

Detrending is done per session (each run gets its own polynomial basis), which is
the block-diagonal-nuisance discipline applied to ROI data: drift in one run must
never leak across a run boundary.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.utils import to_tensor

# A dataset is a list of (D, N) sessions. Elements may be numpy or torch; the
# preprocessing entry point returns torch tensors on the requested device.
Sessions = list[np.ndarray] | list[torch.Tensor]


def detrend_session(
    y: torch.Tensor,
    degree: int,
    *,
    motion: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Project Legendre drift (and optional motion nuisance) out of one session.

    Both the polynomial drift and the motion regressors are projected out **per
    session** in a single orthogonalised pass, so a run's nuisance never leaks
    across a run boundary — the block-diagonal-nuisance discipline, achieved by
    processing one run at a time rather than materialising a big block-diagonal
    matrix. Projecting the combined ``[poly | motion]`` basis (not sequentially)
    handles collinearity between drift and slow motion correctly.

    Parameters
    ----------
    y : torch.Tensor
        ``(D, N)`` ROI-by-time session.
    degree : int
        Maximum Legendre degree to project out (0 = constant/demean, 1 = linear,
        ...). ``degree < 0`` skips the polynomial basis (motion may still apply).
    motion : array-like, optional
        This run's motion parameters, ``(N, K)`` (or ``(K, N)`` — auto-transposed),
        e.g. K=6 rigid-body columns. Columns are used as-is; add derivatives/squares
        upstream if wanted. ``None`` = no motion projection.
    device, dtype
        Where/what to compute in. Defaults to ``y``'s device and float32.

    Returns
    -------
    torch.Tensor
        ``(D, N)`` residual after projecting out the combined nuisance basis.
    """
    if device is None:
        device = y.device
    y = to_tensor(y, dtype=dtype, device=device)
    n = y.shape[1]

    cols: list[torch.Tensor] = []
    if degree >= 0:
        # (N, degree+1) orthogonal Legendre basis (never raw monomials).
        cols.append(construct_polynomial_matrix(n, degree, device=device, dtype=dtype))
    if motion is not None:
        m = to_tensor(motion, dtype=dtype, device=device)
        if m.ndim != 2:
            raise ValueError(f"motion must be 2-D (N, K); got shape {tuple(m.shape)}")
        if m.shape[0] != n:
            # Accept (K, N) by transposing; otherwise it's a length mismatch.
            if m.shape[1] == n:
                m = m.transpose(0, 1).contiguous()
            else:
                raise ValueError(
                    f"motion has {m.shape[0]} rows but the session has N={n} timepoints"
                )
        cols.append(m)

    if not cols:
        return y  # degree < 0 and no motion: nothing to remove

    basis = torch.cat(cols, dim=1)  # (N, degree+1+K)
    # QR gives an orthonormal Q so the projection is numerically clean even when
    # drift and motion columns are correlated.
    q, _ = torch.linalg.qr(basis, mode="reduced")
    yt = y.transpose(0, 1)  # (N, D)
    resid = yt - q @ (q.transpose(0, 1) @ yt)
    return resid.transpose(0, 1).contiguous()


def standardize_session(
    y: torch.Tensor,
    mode: str | None = "zscore",
) -> torch.Tensor:
    """Standardise each ROI (row) of one ``(D, N)`` session.

    ``mode`` is one of ``"zscore"`` (demean + unit variance — state covariance
    becomes a correlation), ``"varnorm"`` (unit variance, mean preserved),
    ``"demean"`` (subtract the mean only), or ``None`` (no-op). Constant ROIs
    (zero variance) are left demeaned rather than producing NaNs.
    """
    if mode is None:
        return y
    if mode not in ("zscore", "varnorm", "demean"):
        raise ValueError(f"unknown standardize mode: {mode!r}")

    mean = y.mean(dim=1, keepdim=True)
    if mode == "demean":
        return y - mean
    # Unbiased=False: we standardise the sample, not estimate a population sd.
    std = y.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    if mode == "varnorm":
        return y / std
    return (y - mean) / std


def preprocess_sessions(
    sessions: Sessions,
    *,
    detrend_degree: int = 1,
    standardize: str | None = "zscore",
    motion: Sessions | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    show_progress: bool | None = None,
) -> list[torch.Tensor]:
    """Detrend (+ optional motion) + standardise a list of ``(D, N)`` sessions.

    Every session must share the same ROI count ``D`` (the model learns one
    observation model across all of them); ``N`` may differ per session.

    Parameters
    ----------
    sessions : list of ``(D, N)`` array-like
        The dataset (runs/sessions = the BSDS "subjects" list).
    detrend_degree : int, default 1
        Legendre drift degree removed per session (see :func:`detrend_session`).
    standardize : str or None, default ``"zscore"``
        Per-ROI standardisation (see :func:`standardize_session`).
    motion : list of ``(N, K)`` array-like, optional
        One motion-parameter array per session (same order/count as ``sessions``),
        projected out per run together with the drift (block-diagonal by
        construction). Each must have ``N`` rows matching its session. ``None`` =
        no motion projection.
    device, dtype
        Output device/dtype. ``device=None`` keeps the current device.
    show_progress : bool or None
        Force the progress bar on/off. ``None`` shows it only for many sessions.

    Returns
    -------
    list of torch.Tensor
        Preprocessed ``(D, N)`` sessions on ``device``.
    """
    if len(sessions) == 0:
        raise ValueError("sessions is empty")
    d0 = sessions[0].shape[0]
    for i, s in enumerate(sessions):
        if s.ndim != 2:
            raise ValueError(f"session {i} must be 2-D (D, N); got shape {tuple(s.shape)}")
        if s.shape[0] != d0:
            raise ValueError(
                f"session {i} has D={s.shape[0]} but session 0 has D={d0}; "
                "all sessions must share the same ROI count"
            )
    if motion is not None and len(motion) != len(sessions):
        raise ValueError(
            f"motion has {len(motion)} entries but there are {len(sessions)} sessions; "
            "pass one motion array per run (same order)"
        )

    if show_progress is None:
        show_progress = len(sessions) >= 8
    out: list[torch.Tensor] = []
    for i, s in enumerate(
        tqdm(sessions, desc="preprocess sessions", leave=True, disable=not show_progress)
    ):
        y = to_tensor(s, dtype=dtype, device=device)
        y = detrend_session(
            y,
            detrend_degree,
            motion=None if motion is None else motion[i],
            device=device,
            dtype=dtype,
        )
        y = standardize_session(y, standardize)
        out.append(y)
    return out


def estimate_latent_dim(
    sessions: Sessions,
    energy: float = 0.9,
    *,
    min_dim: int = 1,
) -> int:
    """Auto-select the latent-factor bound from PCA energy of the data.

    Mirrors the reference ``decideOnLocalComplexityBasedOnEnergy``: per session,
    count the ROI-space principal components needed to reach ``energy`` fraction
    of the variance, and take the max across sessions (capped at ``D - 1``). ARD
    then prunes further per state, so this only sets the upper bound.
    """
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    d = int(sessions[0].shape[0])
    dims: list[int] = []
    for s in sessions:
        y = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        y = y - y.mean(axis=1, keepdims=True)
        sv = np.linalg.svd(y, compute_uv=False)  # (min(D, N),)
        ev = sv**2
        total = ev.sum()
        if total <= 0:
            dims.append(min_dim)
            continue
        cum = np.cumsum(ev) / total
        dims.append(int(np.searchsorted(cum, energy) + 1))
    return int(min(max(max(dims), min_dim), d - 1))


def concat_sessions(
    sessions: list[torch.Tensor],
) -> tuple[torch.Tensor, list[int]]:
    """Concatenate sessions along time for a group-level fit.

    Returns the ``(D, N_total)`` concatenation and the per-session lengths, so
    the group state sequence can be split back into per-session pieces.
    """
    if len(sessions) == 0:
        raise ValueError("sessions is empty")
    lengths = [int(s.shape[1]) for s in sessions]
    return torch.cat(list(sessions), dim=1), lengths
