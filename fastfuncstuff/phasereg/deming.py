"""Deming and OLS regression, vectorized across voxels.

Deming regression is the errors-in-variables generalisation of OLS.
When both the predictor (phase) and response (magnitude) have measurement
noise, OLS under-estimates the slope (attenuation bias).  Deming regression
corrects for this given the variance ratio phi = Var(eps_y) / Var(eps_x).

Special cases:
    phi = 1   -> orthogonal distance regression (total least squares)
    phi -> inf -> ordinary least squares (noise only in Y)

Relation to Menon 2002's chi-squared loss
-----------------------------------------
These are the same estimator.  Menon minimises

    L_ChiSq(b0, b1) = sum_t (S^m - b0 - b1 S^p)^2 / (sigma_m^2 + b1^2 sigma_p^2)

Deming minimises sum_t [(y - y_hat)^2/sigma_y^2 + (x - x_hat)^2/sigma_x^2] over
both the coefficients and the latent true x_hat; profiling out x_hat collapses
it to precisely L_ChiSq with delta = sigma_y^2/sigma_x^2 = phi.  So
``regression="deming"`` here IS phase regression as Menon defined it, and the
closed form below solves the exact b1^2-in-denominator objective.

Worth knowing when reading Vu & Gallant 2015: the "PR" they benchmark sPR
against is *not* quite Menon.  Their Eq. 4-10 use a denominator linear in b1
rather than quadratic, which is what lets them write b1* as a quadratic-formula
closed form.  Our Deming path is closer to the original than their baseline is.

Direction of the bias matters when choosing between the two.  Because the
denominator grows with b1, a larger slope is penalised less, so the minimiser
drifts to larger |b1| than OLS gives.  Under the errors-in-variables model that
is a *correction* — OLS is attenuated when the regressor is noisy.  Where the
EIV model does not hold, it reads as over-suppression, which is the objection
raised by Nencka & Rowe 2007 and Vu & Gallant's reason for reverting to plain
OLS.  Practical consequence: Deming removes MORE macrovascular signal than OLS.
If veins are surviving phase regression, Deming is the right choice and OLS is
the wrong one.

All functions are vectorised across voxels and operate on PyTorch tensors
for GPU acceleration.
"""

from __future__ import annotations

import torch


def deming_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    phi: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deming regression of y on x, vectorised across voxels.

    Parameters
    ----------
    x : Tensor, shape (n_timepoints, n_voxels)
        Predictor (phase time series).
    y : Tensor, shape (n_timepoints, n_voxels)
        Response (magnitude time series).
    phi : Tensor (n_voxels,) or float
        Variance ratio Var(eps_y) / Var(eps_x).  phi=1 gives orthogonal
        distance regression; large phi approaches OLS.

    Returns
    -------
    slope : Tensor (n_voxels,)
    intercept : Tensor (n_voxels,)
    """
    x_bar = x.mean(dim=0)
    y_bar = y.mean(dim=0)

    x_c = x - x_bar
    y_c = y - y_bar

    sxx = (x_c * x_c).sum(dim=0)  # (n_voxels,)
    syy = (y_c * y_c).sum(dim=0)
    sxy = (x_c * y_c).sum(dim=0)

    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=x.dtype, device=x.device)
    phi = phi.broadcast_to(sxx.shape)

    # Closed-form Deming slope (Wikipedia / scipy.odr convention):
    # With delta = phi = Var(eps_y) / Var(eps_x),
    #   beta1 = (SYY - phi*SXX + sqrt((SYY - phi*SXX)^2 + 4*phi*SXY^2))
    #           / (2 * SXY)
    # An earlier rewrite of this used (SXX - phi*SYY) and divided by
    # 2*phi*SXY, which is the same form with phi inverted — i.e. it
    # treated phi as Var(eps_x)/Var(eps_y), the opposite convention from
    # what noise.estimate_variance_ratio returns. That made Deming run as
    # inverse-OLS at fMRI scales (real phi ~ 1e5 → effective delta ~ 1e-5),
    # producing slope ≈ SYY/SXY which blows up in low-correlation voxels.
    diff = syy - phi * sxx
    discriminant = diff * diff + 4.0 * phi * sxy * sxy
    sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))

    numerator = diff + sqrt_disc
    denominator = 2.0 * sxy

    # Guard against sxy ~ 0 (no correlation -> slope = 0)
    slope = torch.where(
        denominator.abs() > 1e-30,
        numerator / denominator,
        torch.zeros_like(numerator),
    )

    intercept = y_bar - slope * x_bar

    return slope, intercept


def ols_regression(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary least squares regression of y on x, vectorised across voxels.

    Parameters
    ----------
    x : Tensor, shape (n_timepoints, n_voxels)
        Predictor (phase time series).
    y : Tensor, shape (n_timepoints, n_voxels)
        Response (magnitude time series).

    Returns
    -------
    slope : Tensor (n_voxels,)
    intercept : Tensor (n_voxels,)
    """
    x_bar = x.mean(dim=0)
    y_bar = y.mean(dim=0)

    x_c = x - x_bar
    y_c = y - y_bar

    sxx = (x_c * x_c).sum(dim=0)
    sxy = (x_c * y_c).sum(dim=0)

    slope = torch.where(
        sxx.abs() > 1e-30,
        sxy / sxx,
        torch.zeros_like(sxy),
    )
    intercept = y_bar - slope * x_bar

    return slope, intercept
