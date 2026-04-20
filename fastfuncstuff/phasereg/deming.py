"""Deming and OLS regression, vectorized across voxels.

Deming regression is the errors-in-variables generalisation of OLS.
When both the predictor (phase) and response (magnitude) have measurement
noise, OLS under-estimates the slope (attenuation bias).  Deming regression
corrects for this given the variance ratio phi = Var(eps_y) / Var(eps_x).

Special cases:
    phi = 1   -> orthogonal distance regression (total least squares)
    phi -> inf -> ordinary least squares (noise only in Y)

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

    # Closed-form Deming slope:
    # beta1 = (phi*SYY - SXX + sqrt((SXX - phi*SYY)^2 + 4*phi*SXY^2))
    #         / (2 * phi * SXY)
    diff = sxx - phi * syy
    discriminant = diff * diff + 4.0 * phi * sxy * sxy
    sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))

    numerator = phi * syy - sxx + sqrt_disc
    denominator = 2.0 * phi * sxy

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
