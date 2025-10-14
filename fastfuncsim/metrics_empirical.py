"""
Design Efficiency and Power Metrics - EMPIRICAL Implementation

Based on Das et al. (2023) "Optimizing cognitive neuroscience experiments
for separating event-related fMRI BOLD responses in non-randomized alternating designs"
and their deconv toolbox implementation.

Key differences from theoretical Liu & Frank (2004):
1. Uses GLS (Generalized Least Squares) with AR(1) prewhitening
2. Estimates temporal autocorrelation from data residuals
3. Computes empirical variance of beta estimates
4. GPU-compatible PyTorch implementation

Theory:
- Detection Power (Fd): Ability to detect activation amplitude
  Fd = 1 / trace(C * Var(β) * C')
  Higher = better detection of signal change

- Estimation Efficiency (Fe): Ability to estimate HRF shape
  Fe = 1 / trace((C ⊗ I) * Var(β_FIR) * (C ⊗ I)')
  Higher = better HRF shape recovery

Both computed with AR(1) correction for realistic fMRI temporal autocorrelation.
"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Tuple
from scipy.linalg import toeplitz
import warnings


def estimate_ar1_coefficient(
    residuals: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None
) -> float:
    """
    Estimate AR(1) coefficient from residuals

    AR(1) model: ε_t = ρ * ε_{t-1} + η_t

    Estimates ρ by OLS regression: residuals[1:] ~ residuals[:-1]

    Parameters
    ----------
    residuals : array-like, shape (n_timepoints,)
        GLM residuals
    device : torch.device, optional
        Device for computation

    Returns
    -------
    rho : float
        AR(1) coefficient (typically 0.2-0.4 for fMRI)
    """
    if device is None:
        device = torch.device('cpu')

    # Convert to numpy for scipy compatibility
    if torch.is_tensor(residuals):
        residuals = residuals.cpu().numpy()

    # OLS: residuals[t] ~ residuals[t-1]
    y = residuals[1:]
    x = residuals[:-1]

    # β = (X'X)^(-1) X'Y
    # For simple regression: β = cov(x,y) / var(x)
    if len(x) > 1:
        rho = np.corrcoef(x, y)[0, 1]  # Correlation = AR(1) coeff
        # Clip to valid range
        rho = np.clip(rho, -0.99, 0.99)
    else:
        rho = 0.0

    return float(rho)


def build_ar1_covariance_matrix(
    n_timepoints: int,
    rho: float,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Build AR(1) covariance matrix

    Σ[i,j] = ρ^|i-j|

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints
    rho : float
        AR(1) coefficient
    device : torch.device, optional
        Device for computation

    Returns
    -------
    sigma : torch.Tensor, shape (n_timepoints, n_timepoints)
        AR(1) covariance matrix
    """
    if device is None:
        device = torch.device('cpu')

    # Build Toeplitz matrix: Σ[i,j] = ρ^|i-j|
    order = np.arange(n_timepoints)
    sigma_np = toeplitz(order)
    sigma_np = rho ** sigma_np

    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    return sigma


def gls_fit(
    Y: Union[torch.Tensor, np.ndarray],
    X: Union[torch.Tensor, np.ndarray],
    sigma: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Generalized Least Squares (GLS) with known covariance

    GLS: β = (X' Σ^(-1) X)^(-1) X' Σ^(-1) Y
    Var(β) = (X' Σ^(-1) X)^(-1)

    Parameters
    ----------
    Y : array-like, shape (n_timepoints,) or (n_timepoints, n_voxels)
        Observed data
    X : array-like, shape (n_timepoints, n_regressors)
        Design matrix
    sigma : array-like, shape (n_timepoints, n_timepoints)
        Covariance matrix
    device : torch.device, optional
        Device for computation

    Returns
    -------
    results : dict
        'betas': Parameter estimates
        'var_betas': Covariance of beta estimates
        'residuals': Residuals
        'sigma_inv_sqrt': Cholesky factor of Sigma^(-1) for whitening
    """
    if device is None:
        device = torch.device('cpu')

    # Convert to tensors
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
    else:
        Y = Y.to(device)

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    else:
        sigma = sigma.to(device)

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y[:, None]

    n_timepoints, n_regressors = X.shape

    # Compute Σ^(-1) via Cholesky decomposition
    # Σ = L L', Σ^(-1) = L'^(-1) L^(-1)
    try:
        L = torch.linalg.cholesky(sigma)
        L_inv = torch.linalg.inv(L)
        sigma_inv = L_inv.T @ L_inv
        sigma_inv_sqrt = L_inv.T  # For whitening
    except:
        # Fallback: direct inversion with regularization
        sigma_reg = sigma + 1e-6 * torch.eye(n_timepoints, device=device)
        sigma_inv = torch.linalg.inv(sigma_reg)
        sigma_inv_sqrt = torch.linalg.cholesky(sigma_inv).T

    # GLS estimates: β = (X' Σ^(-1) X)^(-1) X' Σ^(-1) Y
    XtSX = X.T @ sigma_inv @ X
    XtSY = X.T @ sigma_inv @ Y

    # Add regularization for numerical stability
    XtSX_reg = XtSX + 1e-6 * torch.eye(n_regressors, device=device)

    betas = torch.linalg.solve(XtSX_reg, XtSY)

    # Variance of beta estimates: Var(β) = (X' Σ^(-1) X)^(-1)
    var_betas = torch.linalg.inv(XtSX_reg)

    # Residuals
    residuals = Y - X @ betas

    return {
        'betas': betas,
        'var_betas': var_betas,
        'residuals': residuals,
        'sigma_inv_sqrt': sigma_inv_sqrt,
        'sigma_inv': sigma_inv,
    }


def compute_detection_power_empirical(
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    contrast: Union[torch.Tensor, np.ndarray, None] = None,
    estimate_ar1: bool = True,
    rho: Optional[float] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Compute detection power using GLS with AR(1) correction

    Following Das et al. (2023) deconv implementation.

    Detection power: Fd = 1 / trace(C * Var(β) * C')

    Steps:
    1. Fit OLS to estimate AR(1) coefficient from residuals
    2. Build AR(1) covariance matrix
    3. Fit GLS with AR(1) correction
    4. Compute detection power from contrast variance

    Parameters
    ----------
    data : array-like, shape (n_timepoints,) or (n_voxels, n_timepoints)
        fMRI timeseries (single voxel or ROI average)
    design : array-like, shape (n_timepoints, n_regressors)
        Design matrix (convolved with HRF)
    contrast : array-like, shape (n_regressors,), optional
        Contrast vector (e.g., [1, 0] for first condition)
        If None, uses [1, 0, 0, ...]
    estimate_ar1 : bool, default=True
        Whether to estimate AR(1) from data
    rho : float, optional
        AR(1) coefficient (if not estimating from data)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    result : dict
        'detection_power': float, detection power (higher = better)
        'rho': float, estimated AR(1) coefficient
        'betas': torch.Tensor, GLS parameter estimates
        'var_contrast': float, variance of contrast estimate
    """
    if device is None:
        device = torch.device('cpu')

    # Convert to tensors
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    if not torch.is_tensor(design):
        design = torch.tensor(design, dtype=torch.float32, device=device)
    else:
        design = design.to(device)

    # Handle multiple voxels: average if needed
    if data.ndim == 2:
        # Average across voxels (dim=1, keeping timepoints in dim=0)
        data = data.mean(dim=1)

    n_timepoints = design.shape[0]
    n_regressors = design.shape[1]

    # Default contrast: first condition
    if contrast is None:
        contrast = torch.zeros(n_regressors, device=device)
        contrast[0] = 1.0
    elif not torch.is_tensor(contrast):
        contrast = torch.tensor(contrast, dtype=torch.float32, device=device)
    else:
        contrast = contrast.to(device)

    # Step 1: Estimate AR(1) coefficient if requested
    if estimate_ar1:
        # Fit OLS first
        ols_betas = torch.linalg.lstsq(design, data).solution
        ols_residuals = data - design @ ols_betas

        # Estimate AR(1) from residuals
        rho = estimate_ar1_coefficient(ols_residuals, device=device)
    else:
        if rho is None:
            rho = 0.3  # Default typical value for fMRI

    # Step 2: Build AR(1) covariance matrix
    sigma = build_ar1_covariance_matrix(n_timepoints, rho, device=device)

    # Step 3: Fit GLS
    gls_results = gls_fit(data, design, sigma, device=device)

    # Step 4: Compute detection power
    # Fd = 1 / trace(C * Var(β) * C')
    var_betas = gls_results['var_betas']

    # C * Var(β) * C' = scalar for 1D contrast
    var_contrast = contrast @ var_betas @ contrast

    # Detection power = 1 / variance
    detection_power = 1.0 / var_contrast.item()

    return {
        'detection_power': detection_power,
        'rho': rho,
        'betas': gls_results['betas'],
        'var_contrast': var_contrast.item(),
        'var_betas': var_betas,
    }


def compute_estimation_efficiency_empirical(
    data: Union[torch.Tensor, np.ndarray],
    onsets: Union[torch.Tensor, np.ndarray],
    n_conditions: int,
    hrf_length: int = 30,
    contrast: Union[torch.Tensor, np.ndarray, None] = None,
    estimate_ar1: bool = True,
    rho: Optional[float] = None,
    tr: float = 1.0,
    device: Optional[torch.device] = None
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Compute estimation efficiency using FIR design with GLS

    Following Das et al. (2023) deconv implementation.

    Estimation efficiency: Fe = 1 / trace((C ⊗ I) * Var(β_FIR) * (C ⊗ I)')

    where ⊗ is Kronecker product, I is identity matrix of size hrf_length

    Steps:
    1. Build FIR design matrix (shifted impulses)
    2. Estimate AR(1) from OLS residuals
    3. Fit GLS with AR(1) correction
    4. Compute efficiency with Kronecker product contrast

    Parameters
    ----------
    data : array-like, shape (n_timepoints,) or (n_voxels, n_timepoints)
        fMRI timeseries
    onsets : array-like, shape (n_timepoints, n_conditions)
        Binary onset matrix
    n_conditions : int
        Number of conditions
    hrf_length : int, default=30
        Number of FIR lags (30 TRs ≈ 30s for TR=1s)
    contrast : array-like, shape (n_conditions,), optional
        Contrast for conditions (e.g., [1, 0] for condition 1)
    estimate_ar1 : bool, default=True
        Whether to estimate AR(1)
    rho : float, optional
        AR(1) coefficient (if not estimating)
    tr : float, default=1.0
        Repetition time
    device : torch.device, optional
        Device for computation

    Returns
    -------
    result : dict
        'estimation_efficiency': float, efficiency (higher = better)
        'rho': float, AR(1) coefficient
        'betas_fir': torch.Tensor, FIR parameter estimates
        'var_contrast_fir': float, variance of contrast HRF estimate
    """
    if device is None:
        device = torch.device('cpu')

    # Convert to tensors
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    if not torch.is_tensor(onsets):
        onsets = torch.tensor(onsets, dtype=torch.float32, device=device)
    else:
        onsets = onsets.to(device)

    # Handle multiple voxels
    if data.ndim == 2:
        # Average across voxels (dim=1, keeping timepoints in dim=0)
        data = data.mean(dim=1)

    n_timepoints = len(data)

    # Default contrast: first condition
    if contrast is None:
        contrast_cond = torch.zeros(n_conditions, device=device)
        contrast_cond[0] = 1.0
    elif not torch.is_tensor(contrast):
        contrast_cond = torch.tensor(contrast, dtype=torch.float32, device=device)
    else:
        contrast_cond = contrast.to(device)

    # Step 1: Build FIR design matrix
    # X_FIR[t, condition*lag + lag] = 1 if onset at (t - lag)
    X_FIR = torch.zeros((n_timepoints, n_conditions * hrf_length), device=device)

    for cond in range(n_conditions):
        event_times = torch.where(onsets[:, cond] > 0.5)[0]

        for t in event_times:
            t = int(t.item())
            for lag in range(hrf_length):
                if t + lag < n_timepoints:
                    col_idx = cond * hrf_length + lag
                    X_FIR[t + lag, col_idx] = 1.0

    # Step 2: Estimate AR(1)
    if estimate_ar1:
        ols_betas = torch.linalg.lstsq(X_FIR, data).solution
        ols_residuals = data - X_FIR @ ols_betas
        rho = estimate_ar1_coefficient(ols_residuals, device=device)
    else:
        if rho is None:
            rho = 0.3

    # Step 3: Build AR(1) covariance
    sigma = build_ar1_covariance_matrix(n_timepoints, rho, device=device)

    # Step 4: Fit GLS
    gls_results = gls_fit(data, X_FIR, sigma, device=device)

    # Step 5: Compute efficiency with Kronecker product
    # Contrast for FIR: C ⊗ I_{hrf_length}
    # This selects all lags for the condition(s) of interest
    I_hrf = torch.eye(hrf_length, device=device)

    # Kronecker product: contrast_cond ⊗ I
    # Result shape: (hrf_length, n_conditions * hrf_length)
    contrast_fir = torch.kron(contrast_cond, I_hrf)

    # Var(contrast_fir' * β_FIR) = contrast_fir' * Var(β_FIR) * contrast_fir
    var_betas_fir = gls_results['var_betas']
    var_contrast_fir = contrast_fir @ var_betas_fir @ contrast_fir.T

    # Fe = 1 / trace(Var(HRF))
    estimation_efficiency = 1.0 / torch.trace(var_contrast_fir).item()

    return {
        'estimation_efficiency': estimation_efficiency,
        'rho': rho,
        'betas_fir': gls_results['betas'],
        'var_contrast_fir': torch.trace(var_contrast_fir).item(),
        'hrf_estimate': gls_results['betas'][:hrf_length] if n_conditions == 1 else None,
    }


def evaluate_design_empirical(
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    onsets: Union[torch.Tensor, np.ndarray],
    n_conditions: int,
    hrf_length: int = 30,
    contrast: Union[torch.Tensor, np.ndarray, None] = None,
    tr: float = 1.0,
    device: Optional[torch.device] = None
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Complete design evaluation: detection power + estimation efficiency

    Convenience function that computes both metrics with shared AR(1) estimation.

    Parameters
    ----------
    data : array-like
        fMRI timeseries
    design : array-like
        Design matrix (convolved)
    onsets : array-like
        Onset matrix (binary)
    n_conditions : int
        Number of conditions
    hrf_length : int, default=30
        FIR lags
    contrast : array-like, optional
        Contrast vector
    tr : float, default=1.0
        Repetition time
    device : torch.device, optional
        Device

    Returns
    -------
    metrics : dict
        'detection_power': float
        'estimation_efficiency': float
        'rho': float, AR(1) coefficient
        'summary': dict with key metrics
    """
    if device is None:
        device = torch.device('cpu')

    # Compute detection power
    power_results = compute_detection_power_empirical(
        data, design, contrast, estimate_ar1=True, device=device
    )

    # Compute estimation efficiency (reuse AR(1) estimate)
    efficiency_results = compute_estimation_efficiency_empirical(
        data, onsets, n_conditions, hrf_length, contrast,
        estimate_ar1=False, rho=power_results['rho'], tr=tr, device=device
    )

    return {
        'detection_power': power_results['detection_power'],
        'estimation_efficiency': efficiency_results['estimation_efficiency'],
        'rho': power_results['rho'],
        'summary': {
            'Fd': power_results['detection_power'],
            'Fe': efficiency_results['estimation_efficiency'],
            'rho_ar1': power_results['rho'],
        },
        'details': {
            'power': power_results,
            'efficiency': efficiency_results,
        }
    }
