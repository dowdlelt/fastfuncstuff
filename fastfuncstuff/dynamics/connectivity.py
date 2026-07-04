"""Per-state directed (effective) connectivity from the ROI data.

BSDS's state covariance is *undirected* functional connectivity. The reference
also fits a per-state vector-autoregression to the observed ROI time series
(``posthocVARfromData`` → ``net.ARmodel``), giving a ``D x D`` **directed**
transition ``B_s``: entry ``B_s[i, j]`` is the influence of ROI ``j`` at ``t-1``
on ROI ``i`` at ``t`` within state ``s`` (a Granger-style effective connectivity).

This reuses the same responsibility-weighted VAR estimator as the latent AR
diagnostic (:mod:`.bsds.ar`), applied to the ROI data instead of the latent
factors, and weighted by the state responsibilities.
"""

from __future__ import annotations

import torch

from fastfuncstuff.dynamics.bsds import ar

_DTYPE = torch.float64


def per_state_directed_connectivity(
    model,
    sessions: list[torch.Tensor],
    *,
    responsibilities: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a per-state directed VAR(1) to the ROI data.

    ``sessions`` are the (preprocessed) ``(D, N)`` runs; ``responsibilities``
    default to ``model.responsibilities`` (which must align to ``sessions`` — i.e.
    the runs the model was fit on). To use held-out runs, pass their
    responsibilities from :func:`~fastfuncstuff.dynamics.bsds.model.decode`.

    Returns ``B (K, D, D)`` directed transitions and ``noise_cov (K, D, D)``.
    """
    device = sessions[0].device
    sessions = [s.to(device=device, dtype=_DTYPE) for s in sessions]
    lengths = [int(s.shape[1]) for s in sessions]
    y = torch.cat(sessions, dim=1)  # (D, N)

    resp = responsibilities if responsibilities is not None else model.responsibilities
    qns = torch.cat([r.to(device=device, dtype=_DTYPE) for r in resp], dim=0)  # (N, K)
    if qns.shape[0] != y.shape[1]:
        raise ValueError(
            "responsibilities do not align with sessions "
            f"({qns.shape[0]} frames vs {y.shape[1]}); pass decode(...) responsibilities "
            "for held-out data"
        )

    k, d = model.n_states, int(y.shape[0])
    b_all = torch.empty(k, d, d, dtype=_DTYPE, device=device)
    nc_all = torch.empty(k, d, d, dtype=_DTYPE, device=device)
    for s in range(k):
        b_all[s], nc_all[s] = ar.fit_state_var(y, qns[:, s], lengths)
    return b_all, nc_all
