"""Gaussian-Gamma mixture modelling of spatial maps.

Paper-derived. No implementation is a reference for this module; see
``../fmri_wiki/notes/FSL clean-room policy.md``.

References
----------
- Beckmann, C.F. & Smith, S.M. (2004). *Probabilistic independent component analysis for
  functional magnetic resonance imaging*. IEEE TMI 23(2):137-152, §II-E.

The model
---------
An independent component's spatial map is modelled as a mixture of three classes:

- a **Gaussian** for the background -- voxels not involved in this component, whose spread
  is the residual noise level;
- a **Gamma** for the positive activation tail;
- a mirrored **Gamma** for the negative tail.

Fitted by expectation-maximisation. The useful output is the **posterior probability that
a voxel belongs to an activation class**, which is an *alternative-hypothesis* statement --
"how likely is this voxel active" -- rather than the null-hypothesis "how unlikely is this
value under noise" that a fixed z-threshold gives. The noise level comes from the map's own
background rather than being assumed, which is the entire point of the approach.

What the paper leaves to the implementer, and what we chose
-----------------------------------------------------------
The generative model is fully specified; the fit is not. Three things have to be pinned
down, and an unconstrained EM will find a degenerate answer for each:

1. **Standardisation.** The Gamma is scale-dependent, so the map has to be put on a
   standard scale first. We use the **median and MAD** rather than the mean and standard
   deviation, because the latter are contaminated by the very activation tails the model is
   trying to isolate. Note this does *not* change the returned z-scores, which are
   scale-invariant by construction (the standardising scale divides out of
   ``(x_std - mu_noise) / sigma_noise``). What it changes is the **starting point**: the
   initialisation and the separation floor below are both expressed in standardised units,
   so a scale inflated by a component's own activation puts the EM in a worse basin. The
   fit is bistable, so the basin is what matters.
2. **Separation.** Nothing stops the positive Gamma sliding down on top of the Gaussian and
   modelling the background twice. We require the activation classes to sit at least
   :data:`MIN_SEPARATION_SD` noise standard deviations from the background mean -- if a
   class is not separated from the background, it is not an activation class.
3. **Shape.** A Gamma with shape parameter below 1 has its mode at zero and turns into a
   spike that swallows near-zero voxels. Requiring shape >= :data:`MIN_GAMMA_SHAPE` keeps
   each tail component unimodal with an interior mode, which is what "an activation
   distribution" means. In mean/variance terms that is ``var <= mean**2 / MIN_GAMMA_SHAPE``.

The variance floor applies to **every** component, the background Gaussian included. This
is not decoration: with a floor only on the Gammas, a collapsing background component
divides the z-scores by something near zero and the map blows up by orders of magnitude.
See ``../fmri_wiki/concepts/Constant voxels break the mixture model.md`` -- and note that
the underlying cause there was constant voxels reaching the mixture model at all, which is
:func:`~fastfuncstuff.decomposition.varnorm.noise_std_map`'s job to prevent.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "MIN_GAMMA_SHAPE",
    "MIN_SEPARATION_SD",
    "batch_mixture_zscores",
    "fit_gaussian_gamma_mixture",
    "mixture_zscores_signed",
]

MIN_GAMMA_SHAPE = 2.0
"""Smallest Gamma shape parameter accepted for an activation class.

Above 1 the Gamma is unimodal with its mode away from zero; at or below 1 it becomes a
spike at the origin that competes with the background for near-zero voxels. 2 keeps a clear
margin from that boundary.
"""

MIN_SEPARATION_SD = 3.0
"""How far an activation class must sit from the background, in background SDs."""

_VAR_FLOOR = 1e-4
"""Floor on every component variance, background included. See the module docstring."""

_EPS = 1e-12


def _gamma_logpdf(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """Log density of a Gamma parametrised by mean and variance, evaluated at ``x >= 0``.

    shape = mean^2 / var, rate = mean / var. Values at or below zero get ``-inf`` so they
    contribute no responsibility to a tail class.
    """
    mean = mean.clamp(min=_EPS)
    var = var.clamp(min=_EPS)
    shape = (mean * mean / var).clamp(min=_EPS)
    rate = (mean / var).clamp(min=_EPS)
    xs = x.clamp(min=_EPS)
    lp = shape * torch.log(rate) + (shape - 1.0) * torch.log(xs) - rate * xs - torch.lgamma(shape)
    return torch.where(x > 0, lp, torch.full_like(lp, -float("inf")))


def _gauss_logpdf(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp(min=_EPS)
    return -0.5 * (torch.log(2.0 * torch.pi * var) + (x - mean) ** 2 / var)


def _robust_standardise(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Centre and scale each row by its median and MAD. Returns ``(z, centre, scale)``.

    MAD is rescaled by 1.4826 so that it estimates the standard deviation of a Gaussian,
    which keeps the standardised background at roughly unit variance and makes the
    initialisation below scale-free.
    """
    centre = x.median(dim=1, keepdim=True).values
    mad = (x - centre).abs().median(dim=1, keepdim=True).values * 1.4826
    # A map with more than half its voxels identical has MAD 0; fall back to the standard
    # deviation so the fit degrades rather than dividing by zero.
    fallback = x.std(dim=1, keepdim=True, unbiased=True)
    scale = torch.where(mad > _EPS, mad, fallback).clamp(min=_EPS)
    return (x - centre) / scale, centre.squeeze(1), scale.squeeze(1)


@torch.inference_mode()
def fit_gaussian_gamma_mixture(
    maps_kv: Tensor,
    n_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, Tensor]:
    """Fit the three-class mixture to each row of ``maps_kv`` ``(K, V)`` by EM.

    All K components are fitted simultaneously; every quantity below carries a leading
    component axis, so this is one batched EM rather than K sequential ones.

    Returns a dict of ``(K,)`` parameter tensors plus ``(K, V)`` posteriors, in the
    standardised frame (``centre`` and ``scale`` map back to data units).
    """
    x, centre, scale = _robust_standardise(maps_kv.to(torch.float64))
    k, v = x.shape
    dev = x.device

    # Initialisation: background at the robust centre with unit spread, tails placed
    # symmetrically out in the wings. Deliberately wide -- the fit is bistable, and a
    # narrow-background start can converge to a solution that calls most of the map signal.
    mu_n = torch.zeros(k, 1, dtype=x.dtype, device=dev)
    var_n = torch.ones(k, 1, dtype=x.dtype, device=dev)
    mu_p = torch.full((k, 1), 3.0, dtype=x.dtype, device=dev)
    var_p = torch.ones(k, 1, dtype=x.dtype, device=dev)
    mu_m = torch.full((k, 1), 3.0, dtype=x.dtype, device=dev)  # magnitude on the neg side
    var_m = torch.ones(k, 1, dtype=x.dtype, device=dev)
    pi = torch.full((k, 3), 1.0 / 3.0, dtype=x.dtype, device=dev)

    prev_ll = torch.full((k,), -float("inf"), dtype=x.dtype, device=dev)
    converged = torch.zeros(k, dtype=torch.bool, device=dev)

    for _ in range(int(n_iter)):
        # ---- E step ----
        log_n = _gauss_logpdf(x, mu_n, var_n) + torch.log(pi[:, 0:1].clamp(min=_EPS))
        log_p = _gamma_logpdf(x, mu_p, var_p) + torch.log(pi[:, 1:2].clamp(min=_EPS))
        log_m = _gamma_logpdf(-x, mu_m, var_m) + torch.log(pi[:, 2:3].clamp(min=_EPS))

        stacked = torch.stack([log_n, log_p, log_m], dim=0)  # (3, K, V)
        log_norm = torch.logsumexp(stacked, dim=0)  # (K, V)
        resp = torch.exp(stacked - log_norm.unsqueeze(0))  # (3, K, V)

        ll = log_norm.mean(dim=1)
        converged = converged | ((ll - prev_ll).abs() < tol)
        prev_ll = ll
        if bool(converged.all()):
            break

        # ---- M step ----
        nk = resp.sum(dim=2).clamp(min=_EPS)  # (3, K)
        pi = (nk / float(v)).T.clamp(min=_EPS)
        pi = pi / pi.sum(dim=1, keepdim=True)

        r_n, r_p, r_m = resp[0], resp[1], resp[2]

        mu_n = (r_n * x).sum(dim=1, keepdim=True) / nk[0].unsqueeze(1)
        var_n = (r_n * (x - mu_n) ** 2).sum(dim=1, keepdim=True) / nk[0].unsqueeze(1)
        var_n = var_n.clamp(min=_VAR_FLOOR)

        # Tail classes are fitted by moment matching on the side of the map they live on.
        for sign, (mu_ref, var_ref) in ((1.0, ("p", None)), (-1.0, ("m", None))):
            r = r_p if sign > 0 else r_m
            xs = x if sign > 0 else -x
            pos = (xs > 0).to(x.dtype)
            w = r * pos
            wn = w.sum(dim=1, keepdim=True).clamp(min=_EPS)
            m = (w * xs).sum(dim=1, keepdim=True) / wn
            s = (w * (xs - m) ** 2).sum(dim=1, keepdim=True) / wn

            # Separation: an activation class must stand clear of the background, or it is
            # just modelling the background a second time.
            floor = (
                mu_n.abs() + MIN_SEPARATION_SD * torch.sqrt(var_n)
                if sign > 0
                else (-mu_n + MIN_SEPARATION_SD * torch.sqrt(var_n))
            )
            m = torch.maximum(m, floor.clamp(min=_EPS))
            # Shape: keep the Gamma unimodal with an interior mode.
            s = s.clamp(min=_VAR_FLOOR)
            s = torch.minimum(s, m * m / MIN_GAMMA_SHAPE)

            if sign > 0:
                mu_p, var_p = m, s
            else:
                mu_m, var_m = m, s
            del mu_ref, var_ref

    # Final responsibilities at the fitted parameters.
    log_n = _gauss_logpdf(x, mu_n, var_n) + torch.log(pi[:, 0:1].clamp(min=_EPS))
    log_p = _gamma_logpdf(x, mu_p, var_p) + torch.log(pi[:, 1:2].clamp(min=_EPS))
    log_m = _gamma_logpdf(-x, mu_m, var_m) + torch.log(pi[:, 2:3].clamp(min=_EPS))
    stacked = torch.stack([log_n, log_p, log_m], dim=0)
    resp = torch.exp(stacked - torch.logsumexp(stacked, dim=0).unsqueeze(0))

    return {
        "x_std": x,
        "centre": centre,
        "scale": scale,
        "mu_noise": mu_n.squeeze(1),
        "var_noise": var_n.squeeze(1),
        "mu_pos": mu_p.squeeze(1),
        "var_pos": var_p.squeeze(1),
        "mu_neg": mu_m.squeeze(1),
        "var_neg": var_m.squeeze(1),
        "pi_noise": pi[:, 0],
        "pi_pos": pi[:, 1],
        "pi_neg": pi[:, 2],
        "p_noise": resp[0],
        "p_pos": resp[1],
        "p_neg": resp[2],
        "converged": converged,
        "log_likelihood": prev_ll,
    }


@torch.inference_mode()
def batch_mixture_zscores(
    components_kv: Tensor | np.ndarray,
    device: torch.device | None = None,
    verbose: bool = False,
    n_iter: int = 200,
) -> tuple[Tensor, Tensor, list[dict]]:
    """Signed z-scores and activation posteriors for every component map.

    ``components_kv`` is ``(K, V)``. Returns ``(z_signed, p_signal, meta_list)`` with the
    first two ``(K, V)`` float32 tensors.

    ``z_signed`` is the map expressed in units of the fitted **background** standard
    deviation -- so it says how far each voxel is from this component's own noise, not from
    an assumed one. ``p_signal`` is the posterior probability of belonging to either
    activation class, which is the quantity to threshold when you want a statement about
    activation rather than about the null.
    """
    if not isinstance(components_kv, torch.Tensor):
        components_kv = torch.as_tensor(np.asarray(components_kv), dtype=torch.float32)
    if components_kv.ndim != 2:
        raise ValueError(f"expected (K, V), got shape {tuple(components_kv.shape)}")
    if device is not None:
        components_kv = components_kv.to(device)

    fit = fit_gaussian_gamma_mixture(components_kv, n_iter=n_iter)

    sigma_n = torch.sqrt(fit["var_noise"]).clamp(min=1e-8).unsqueeze(1)
    z_signed = ((fit["x_std"] - fit["mu_noise"].unsqueeze(1)) / sigma_n).to(torch.float32)
    p_signal = (fit["p_pos"] + fit["p_neg"]).to(torch.float32)

    meta_list = []
    for i in range(components_kv.shape[0]):
        meta_list.append(
            {
                "mu_noise": float(fit["mu_noise"][i]),
                "sigma_noise": float(torch.sqrt(fit["var_noise"][i])),
                "mu_signal_pos": float(fit["mu_pos"][i]),
                "sigma_signal_pos": float(torch.sqrt(fit["var_pos"][i])),
                "mu_signal_neg": float(fit["mu_neg"][i]),
                "sigma_signal_neg": float(torch.sqrt(fit["var_neg"][i])),
                "pi_noise": float(fit["pi_noise"][i]),
                "pi_pos": float(fit["pi_pos"][i]),
                "pi_neg": float(fit["pi_neg"][i]),
                "mixing_signal": float(p_signal[i].mean()),
                "converged": bool(fit["converged"][i]),
                "data_centre": float(fit["centre"][i]),
                "data_scale": float(fit["scale"][i]),
            }
        )
    if verbose:
        n_conv = int(fit["converged"].sum())
        print(f"  GGM: {n_conv}/{components_kv.shape[0]} components converged")
    return z_signed, p_signal, meta_list


def mixture_zscores_signed(comp_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-component convenience wrapper over :func:`batch_mixture_zscores`."""
    arr = np.asarray(comp_map, dtype=np.float64).ravel()[None, :]
    z, p, meta = batch_mixture_zscores(torch.as_tensor(arr))
    return z[0].cpu().numpy(), p[0].cpu().numpy(), meta[0]
