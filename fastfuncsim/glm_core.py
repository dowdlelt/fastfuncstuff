"""
Ultra-fast GPU-accelerated GLM solver
Core engine for fastfuncsim - handles all GLM fitting regardless of design matrix type

Supports multiple GLM variants:
- OLS (Ordinary Least Squares): Fast, assumes independence
- ARMA(1,1): Accounts for temporal autocorrelation (see arma_glm.py)
- Ridge: Regularization for correlated regressors (future)
"""

import warnings
from typing import Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from .utils import get_device, optimal_chunk_size, to_tensor


class GLMResults:
    """Container for GLM results"""

    def __init__(self):
        self.betas = None  # (n_voxels, n_regressors) or (n_x, n_y, n_z, n_regressors)
        self.r2 = None  # (n_voxels,) or (n_x, n_y, n_z)
        self.r2_run = None  # (n_voxels, n_runs) if provided
        self.residuals = None  # (n_voxels, n_timepoints) - optional
        self.predicted = None  # (n_voxels, n_timepoints) - optional
        self.meanvol = None  # Mean signal across time
        self.tstats = None  # (n_voxels, n_regressors)
        self.stderr = None  # (n_voxels, n_regressors)
        self.sigma2 = None  # (n_voxels,)
        self.fstats = None  # (n_voxels,)
        self.dof = None  # Degrees of freedom used for contrasts
        self.original_shape = None  # Original spatial dimensions
        self.tr = None  # Repetition time (seconds)
        self.voxel_mask = None  # Optional boolean mask for sparse analyses
        self.full_shape = None  # Original spatial shape before masking
        self.affine = None  # Spatial affine if available

    def to_spatial(self):
        """Reshape results back to spatial dimensions if available"""
        if self.original_shape is None:
            warnings.warn("Original shape not stored, cannot reshape")
            return self

        if len(self.original_shape) == 3:
            nx, ny, nz = self.original_shape
            self.betas = self.betas.reshape(nx, ny, nz, -1)
            self.r2 = self.r2.reshape(nx, ny, nz)
            if self.r2_run is not None:
                self.r2_run = self.r2_run.reshape(nx, ny, nz, -1)
            if self.residuals is not None:
                self.residuals = self.residuals.reshape(nx, ny, nz, -1)
            if self.predicted is not None:
                self.predicted = self.predicted.reshape(nx, ny, nz, -1)
            if self.meanvol is not None:
                self.meanvol = self.meanvol.reshape(nx, ny, nz)
            if self.tstats is not None:
                self.tstats = self.tstats.reshape(nx, ny, nz, -1)
            if self.stderr is not None:
                self.stderr = self.stderr.reshape(nx, ny, nz, -1)
            if self.sigma2 is not None:
                self.sigma2 = self.sigma2.reshape(nx, ny, nz)
            if self.fstats is not None:
                self.fstats = self.fstats.reshape(nx, ny, nz)

        return self


def construct_polynomial_matrix(
    n_timepoints: int, max_degree: int, device: torch.device
) -> torch.Tensor:
    """
    Construct polynomial nuisance regressor matrix (for detrending)

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    max_degree : int
        Maximum polynomial degree (0=constant, 1=linear, etc.)
    device : torch.device
        Device to create tensor on

    Returns
    -------
    poly_matrix : torch.Tensor
        (n_timepoints, max_degree+1) polynomial matrix
    """
    if max_degree < 0:
        return torch.empty(n_timepoints, 0, device=device)

    # Create time vector normalized to [-1, 1]
    t = torch.linspace(-1, 1, n_timepoints, device=device)

    # Create polynomial terms
    poly_matrix = torch.zeros(n_timepoints, max_degree + 1, device=device)
    for degree in range(max_degree + 1):
        poly_matrix[:, degree] = t**degree

    return poly_matrix


def orthogonalize_design(X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalize design matrix X with respect to nuisance regressors Z
    Returns X_orth such that X_orth is orthogonal to Z

    Parameters
    ----------
    X : torch.Tensor
        (n_timepoints, n_regressors) design matrix
    Z : torch.Tensor
        (n_timepoints, n_nuisance) nuisance regressors

    Returns
    -------
    X_orth : torch.Tensor
        Orthogonalized design matrix
    """
    if Z.shape[1] == 0:
        return X

    # Project out Z from X: X_orth = X - Z * (Z'Z)^-1 * Z'X
    # Using QR for numerical stability
    Q, _ = torch.linalg.qr(Z)
    X_orth = X - Q @ (Q.T @ X)

    return X_orth


def fit_glm_chunk(
    data: torch.Tensor,
    design: torch.Tensor,
    want_residuals: bool = False,
    want_predicted: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Fit GLM for a chunk of voxels using efficient batched operations

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) data matrix
    design : torch.Tensor
        (n_timepoints, n_regressors) design matrix
    want_residuals : bool
        Whether to return residuals
    want_predicted : bool
        Whether to return predicted values

    Returns
    -------
    betas : torch.Tensor
        (n_voxels, n_regressors) beta coefficients
    r2 : torch.Tensor
        (n_voxels,) R² values
    ss_residual : torch.Tensor
        (n_voxels,) sum of squared residuals
    residuals : torch.Tensor or None
        (n_voxels, n_timepoints) residuals if requested
    predicted : torch.Tensor or None
        (n_voxels, n_timepoints) predicted values if requested
    """
    n_voxels, n_timepoints = data.shape

    # Solve using batched least squares: beta = (X'X)^-1 X'Y
    # Using torch.linalg.lstsq is faster than manual (X'X)^-1
    # Shape: design (T, R), data.T (T, V) -> betas (R, V)
    try:
        # lstsq solves X @ beta = Y for beta
        betas = torch.linalg.lstsq(design, data.T).solution  # (n_regressors, n_voxels)
        betas = betas.T  # (n_voxels, n_regressors)
    except RuntimeError:
        # Fallback to QR decomposition if lstsq fails
        Q, R = torch.linalg.qr(design)
        betas = torch.linalg.solve_triangular(R, Q.T @ data.T, upper=True)
        betas = betas.T

    # Compute predictions
    predicted_vals = data @ design @ betas.T if want_predicted else design @ betas.T

    # Compute R²: R² = 1 - SSE/SST
    # SST = sum((Y - mean(Y))²)
    # SSE = sum((Y - Y_pred)²)
    data_mean = data.mean(dim=1, keepdim=True)
    ss_total = ((data - data_mean) ** 2).sum(dim=1)

    residuals_vals = data - predicted_vals.T
    ss_residual = (residuals_vals**2).sum(dim=1)

    r2 = 1 - ss_residual / (
        ss_total + 1e-10
    )  # Add small epsilon to avoid division by zero
    r2 = torch.clamp(r2, 0, 1)  # Clamp to [0, 1]

    return (
        betas,
        r2,
        ss_residual,
        residuals_vals if want_residuals else None,
        predicted_vals.T if want_predicted else None,
    )


def fit_glm(
    data: Union[torch.Tensor, list],
    design: Union[torch.Tensor, list],
    tr: float,
    max_poly_degree: Optional[Union[int, list]] = None,
    extra_regressors: Optional[Union[torch.Tensor, list]] = None,
    want_residuals: bool = False,
    want_predicted: bool = False,
    want_r2_run: bool = True,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
    verbose: bool = True,
    preload_data_to_device: bool = True,
) -> GLMResults:
    """
    Fast GPU-accelerated GLM fitting

    This is the core engine that handles all GLM variants:
    - Assumed HRF: design is onsets convolved with HRF
    - FIR: design is shifted impulses for each lag
    - HRF library: call this multiple times with different HRFs, pick best R²

    Parameters
    ----------
    data : torch.Tensor or list of torch.Tensor
        fMRI data. Can be:
        - Single run: (n_voxels, n_timepoints) or (n_x, n_y, n_z, n_timepoints)
        - Multiple runs: list of tensors, each (n_voxels, n_timepoints)
    design : torch.Tensor or list of torch.Tensor
        Design matrix. Can be:
        - Single run: (n_timepoints, n_regressors)
        - Multiple runs: list of matrices, each (n_timepoints, n_regressors)
    tr : float
        Repetition time in seconds
    max_poly_degree : int or list of int, optional
        Maximum polynomial degree for detrending each run
        If None, auto-compute as round(duration_minutes/2)
    extra_regressors : torch.Tensor or list of torch.Tensor, optional
        Additional nuisance regressors (motion, GLMdenoise PCs, etc.)
        Shape: (n_timepoints, n_extra) or list per run
    want_residuals : bool
        Return residuals (uses more memory)
    want_predicted : bool
        Return predicted timecourses
    want_r2_run : bool
        Compute R² separately for each run
    device : torch.device, optional
        Computing device. If None, auto-detect.
    chunk_size : int, optional
        Number of voxels to process at once. If None, auto-compute.
    verbose : bool
        Print progress
    preload_data_to_device : bool
        If True (default), loads all voxel data onto the compute device up front (legacy behavior).
        If False, keeps data on CPU and streams chunks to the device to reduce memory usage.

    Returns
    -------
    results : GLMResults
        Object containing betas, R², etc.
    """
    # Setup device
    if device is None:
        device = get_device()

    # Handle single vs multiple runs
    is_single_run = not isinstance(data, list)
    if is_single_run:
        data = [data]
        design = [design]
        if extra_regressors is not None:
            extra_regressors = [extra_regressors]

    n_runs = len(data)

    # Convert to tensors and reshape to 2D (optionally keep on CPU for streaming)
    original_shape = None
    data_2d: list[torch.Tensor] = []
    storage_device = device if preload_data_to_device else torch.device("cpu")
    for i, d in enumerate(data):
        d_tensor = to_tensor(d, device=storage_device)
        d_tensor = d_tensor.to(torch.float32)
        if not preload_data_to_device and d_tensor.device.type != "cpu":
            d_tensor = d_tensor.cpu()

        if d_tensor.ndim == 4:  # (n_x, n_y, n_z, n_timepoints)
            if original_shape is None:
                original_shape = d_tensor.shape[:3]
            d_tensor = d_tensor.reshape(
                -1, d_tensor.shape[-1]
            )  # (n_voxels, n_timepoints)
        elif d_tensor.ndim == 2:
            pass  # Already (n_voxels, n_timepoints)
        else:
            raise ValueError(
                f"Data must be 2D (n_voxels, n_timepoints) or 4D (nx, ny, nz, nt), got {d_tensor.ndim}D"
            )
        data_2d.append(d_tensor)

    n_voxels = data_2d[0].shape[0]

    # Convert design matrices
    design_2d = [to_tensor(d, device=device) for d in design]

    # Handle polynomial detrending
    if max_poly_degree is None:
        max_poly_degree = [round((d.shape[1] * tr / 60) / 2) for d in data_2d]
    elif isinstance(max_poly_degree, int):
        max_poly_degree = [max_poly_degree] * n_runs

    # Build full design matrices with polynomials and extra regressors
    full_designs = []
    for run_idx in range(n_runs):
        n_tp = design_2d[run_idx].shape[0]

        # Start with the task design
        full_design = design_2d[run_idx]

        # Add polynomial regressors
        poly = construct_polynomial_matrix(n_tp, max_poly_degree[run_idx], device)

        # Add extra regressors if provided
        if extra_regressors is not None and extra_regressors[run_idx] is not None:
            extra = to_tensor(extra_regressors[run_idx], device=device)
            nuisance = torch.cat([poly, extra], dim=1)
        else:
            nuisance = poly

        # Orthogonalize task design with respect to nuisance
        if nuisance.shape[1] > 0:
            full_design = orthogonalize_design(full_design, nuisance)

        # Concatenate task + nuisance for final design matrix
        full_design = torch.cat([full_design, nuisance], dim=1)
        full_designs.append(full_design)

    n_task_regressors = design_2d[0].shape[1]

    # Determine chunk size
    if chunk_size is None:
        chunk_size = optimal_chunk_size(
            n_voxels, data_2d[0].shape[1], full_designs[0].shape[1], device
        )

    if verbose:
        print(
            f"Fitting GLM: {n_voxels} voxels, {n_runs} runs, {n_task_regressors} task regressors"
        )
        print(f"Processing in chunks of {chunk_size} voxels")

    # Concatenate all runs
    data_concat = torch.cat(data_2d, dim=1)  # (n_voxels, total_timepoints)
    design_concat = torch.block_diag(*full_designs)  # Block diagonal for runs

    # Also track per-run data for R² calculation
    run_boundaries = [0]
    for d in data_2d:
        run_boundaries.append(run_boundaries[-1] + d.shape[1])

    # Fit in chunks
    all_betas = []
    all_r2 = []
    all_r2_run = [] if want_r2_run else None
    all_residuals = [] if want_residuals else None
    all_predicted = [] if want_predicted else None

    n_chunks = (n_voxels + chunk_size - 1) // chunk_size

    # Pre-compute matrices for statistics
    xtx = design_concat.T @ design_concat
    ridge = 1e-6 * torch.eye(xtx.shape[0], device=device)
    xtx_reg = xtx + ridge
    xtx_inv = torch.linalg.inv(xtx_reg)

    task_beta_indices = []
    reg_offset = 0
    for run_idx in range(n_runs):
        run_n_regressors = full_designs[run_idx].shape[1]
        task_beta_indices.extend(range(reg_offset, reg_offset + n_task_regressors))
        reg_offset += run_n_regressors

    if len(task_beta_indices) == 0:
        raise ValueError("No task regressors detected; cannot compute GLM statistics")

    task_idx_tensor = torch.tensor(task_beta_indices, device=device, dtype=torch.long)
    xtx_inv_task = xtx_inv.index_select(0, task_idx_tensor).index_select(
        1, task_idx_tensor
    )
    xtx_inv_task_diag = torch.diagonal(xtx_inv_task, dim1=0, dim2=1)
    dof = design_concat.shape[0] - design_concat.shape[1]
    if dof <= 0:
        warnings.warn(
            "Non-positive degrees of freedom detected in GLM fit; statistics may be invalid"
        )
        dof = max(1, dof)

    # Precompute inverse for F-stat (avoid instability by adding ridge)
    xtx_inv_task_reg = xtx_inv_task + 1e-6 * torch.eye(
        xtx_inv_task.shape[0], device=device
    )
    xtx_inv_task_inv = torch.linalg.inv(xtx_inv_task_reg)
    n_task_params = xtx_inv_task.shape[0]

    all_sigma2: list[torch.Tensor] = []
    all_tstats: list[torch.Tensor] = []
    all_stderr: list[torch.Tensor] = []
    all_fstats: list[torch.Tensor] = []

    # Progress bar for chunks
    chunk_iterator = range(n_chunks)
    if verbose and n_chunks > 1:
        chunk_iterator = tqdm(
            chunk_iterator, desc="Processing voxel chunks", unit="chunk"
        )

    for chunk_idx in chunk_iterator:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_voxels)

        chunk_data_cpu = data_concat[start_idx:end_idx]
        chunk_data = (
            chunk_data_cpu if preload_data_to_device else chunk_data_cpu.to(device)
        )

        # Fit this chunk on the compute device
        betas_dev, r2_dev, ss_residual_dev, residuals_dev, predicted_dev = (
            fit_glm_chunk(chunk_data, design_concat, want_residuals, want_predicted)
        )

        # Extract only task regressors (ignore nuisance)
        betas_task_dev = betas_dev[:, task_beta_indices]

        sigma2_dev = torch.clamp(ss_residual_dev / dof, min=0.0)

        # Standard errors and t-stats
        stderr_dev = torch.sqrt(
            torch.clamp(sigma2_dev.unsqueeze(1), min=0.0)
            * xtx_inv_task_diag.unsqueeze(0)
        )
        tstats_dev = betas_task_dev / (stderr_dev + 1e-10)

        # F-statistics (default to t^2 when single regressor)
        if n_task_params == 1:
            fstats_dev = tstats_dev[:, 0] ** 2
        else:
            quad_dev = torch.einsum(
                "bi,ij,bj->b", betas_task_dev, xtx_inv_task_inv, betas_task_dev
            )
            fstats_dev = quad_dev / (n_task_params * sigma2_dev + 1e-10)

        # Move outputs back to CPU for aggregation
        if preload_data_to_device:
            betas_cpu = betas_task_dev
            r2_cpu = r2_dev
            sigma2_cpu = sigma2_dev
            stderr_cpu = stderr_dev
            tstats_cpu = tstats_dev
            fstats_cpu = fstats_dev
            residuals_cpu = residuals_dev if residuals_dev is not None else None
            predicted_cpu = predicted_dev if predicted_dev is not None else None
        else:
            betas_cpu = betas_task_dev.cpu()
            r2_cpu = r2_dev.cpu()
            sigma2_cpu = sigma2_dev.cpu()
            stderr_cpu = stderr_dev.cpu()
            tstats_cpu = tstats_dev.cpu()
            fstats_cpu = fstats_dev.cpu()
            residuals_cpu = residuals_dev.cpu() if residuals_dev is not None else None
            predicted_cpu = predicted_dev.cpu() if predicted_dev is not None else None

        all_betas.append(betas_cpu)
        all_r2.append(r2_cpu)

        if want_residuals and residuals_cpu is not None:
            all_residuals.append(residuals_cpu)
        if want_predicted and predicted_cpu is not None:
            all_predicted.append(predicted_cpu)

        all_sigma2.append(sigma2_cpu)
        all_tstats.append(tstats_cpu)
        all_stderr.append(stderr_cpu)
        all_fstats.append(fstats_cpu)

        # Compute per-run R² if requested
        if want_r2_run:
            r2_run = []
            for run_idx in range(n_runs):
                run_start = run_boundaries[run_idx]
                run_end = run_boundaries[run_idx + 1]
                run_data_cpu = chunk_data_cpu[:, run_start:run_end]
                if predicted_cpu is not None:
                    run_pred_cpu = predicted_cpu[:, run_start:run_end]
                else:
                    run_design = full_designs[run_idx]
                    run_betas_dev = betas_task_dev[
                        :,
                        run_idx * n_task_regressors : (run_idx + 1) * n_task_regressors,
                    ]
                    run_pred_dev = (
                        run_design[:, :n_task_regressors] @ run_betas_dev.T
                    ).T
                    run_pred_cpu = (
                        run_pred_dev if preload_data_to_device else run_pred_dev.cpu()
                    )

                run_mean = run_data_cpu.mean(dim=1, keepdim=True)
                run_ss_total = ((run_data_cpu - run_mean) ** 2).sum(dim=1)
                run_residuals_cpu = run_data_cpu - run_pred_cpu
                run_ss_residual = (run_residuals_cpu**2).sum(dim=1)
                run_r2_cpu = 1 - run_ss_residual / (run_ss_total + 1e-10)
                run_r2_cpu = torch.clamp(run_r2_cpu, 0, 1)
                r2_run.append(run_r2_cpu)

            all_r2_run.append(torch.stack(r2_run, dim=1))  # (chunk_voxels, n_runs)

        # Release GPU tensors for this chunk
        del (
            chunk_data,
            betas_dev,
            betas_task_dev,
            r2_dev,
            ss_residual_dev,
            residuals_dev,
            predicted_dev,
            sigma2_dev,
            stderr_dev,
            tstats_dev,
            fstats_dev,
        )

    # Concatenate results
    results = GLMResults()
    concat_device = device if preload_data_to_device else torch.device("cpu")
    results.betas = torch.cat(all_betas, dim=0).to(concat_device)
    results.r2 = torch.cat(all_r2, dim=0).to(concat_device)
    meanvol = data_concat.mean(dim=1)
    results.meanvol = meanvol.to(concat_device)
    results.original_shape = original_shape
    results.tstats = torch.cat(all_tstats, dim=0).to(concat_device)
    results.stderr = torch.cat(all_stderr, dim=0).to(concat_device)
    results.sigma2 = torch.cat(all_sigma2, dim=0).to(concat_device)
    results.fstats = torch.cat(all_fstats, dim=0).to(concat_device)
    results.dof = dof
    results.tr = tr

    if want_r2_run:
        results.r2_run = torch.cat(all_r2_run, dim=0).to(concat_device)
    if want_residuals:
        results.residuals = torch.cat(all_residuals, dim=0).to(concat_device)
    if want_predicted:
        results.predicted = torch.cat(all_predicted, dim=0).to(concat_device)

    if verbose:
        print(f"GLM complete. Mean R² = {results.r2.mean().item():.3f}")

    return results


def percent_bold_change(betas: torch.Tensor, meanvol: torch.Tensor) -> torch.Tensor:
    """
    Convert beta coefficients to percent BOLD change

    Parameters
    ----------
    betas : torch.Tensor
        Beta coefficients (n_voxels, n_regressors)
    meanvol : torch.Tensor
        Mean signal per voxel (n_voxels,)

    Returns
    -------
    betas_pct : torch.Tensor
        Betas in percent BOLD change
    """
    return (betas / meanvol.unsqueeze(1).abs()) * 100


def fit_glm_hrf_library(
    data: Union[torch.Tensor, list],
    design: Union[torch.Tensor, list],
    hrf_library: torch.Tensor,
    tr: float,
    **kwargs,
) -> Tuple[GLMResults, torch.Tensor, torch.Tensor]:
    """
    Fit GLM with HRF library and select best HRF per voxel

    Parameters
    ----------
    data : torch.Tensor or list
        fMRI data
    design : torch.Tensor or list
        Design matrix (NOT yet convolved with HRF)
    hrf_library : torch.Tensor
        (n_hrfs, n_timepoints) library of HRF candidates
    tr : float
        Repetition time
    **kwargs : dict
        Additional arguments passed to fit_glm

    Returns
    -------
    results : GLMResults
        Results using best HRF per voxel
    hrf_index : torch.Tensor
        (n_voxels,) index of best HRF for each voxel
    r2_all_hrfs : torch.Tensor
        (n_voxels, n_hrfs) R² for each HRF
    """
    device = kwargs.get("device", get_device())
    hrf_library = to_tensor(hrf_library, device=device)
    n_hrfs = hrf_library.shape[0]

    verbose = kwargs.get("verbose", True)
    if verbose:
        print(f"Fitting HRF library: {n_hrfs} candidates")

    # Fit GLM for each HRF
    all_r2 = []
    all_results = []

    for hrf_idx in range(n_hrfs):
        if verbose:
            print(f"  HRF {hrf_idx + 1}/{n_hrfs}")

        # Convolve design with this HRF
        # This needs to be implemented in design.py
        # For now, assume design is already per-HRF or we'll implement later

        results = fit_glm(data, design, tr, verbose=False, **kwargs)
        all_r2.append(results.r2)
        all_results.append(results)

    # Stack R² values
    r2_all_hrfs = torch.stack(all_r2, dim=1)  # (n_voxels, n_hrfs)

    # Select best HRF per voxel
    hrf_index = torch.argmax(r2_all_hrfs, dim=1)  # (n_voxels,)

    # Extract results for best HRF
    best_results = GLMResults()
    best_results.original_shape = all_results[0].original_shape

    n_voxels = r2_all_hrfs.shape[0]
    n_regressors = all_results[0].betas.shape[1]

    # Select betas and R² for best HRF per voxel
    best_results.betas = torch.zeros(n_voxels, n_regressors, device=device)
    best_results.r2 = torch.zeros(n_voxels, device=device)

    for voxel_idx in range(n_voxels):
        best_hrf = hrf_index[voxel_idx].item()
        best_results.betas[voxel_idx] = all_results[best_hrf].betas[voxel_idx]
        best_results.r2[voxel_idx] = all_results[best_hrf].r2[voxel_idx]

    best_results.meanvol = all_results[0].meanvol

    if verbose:
        print(f"Best HRF selection: mean R² = {best_results.r2.mean().item():.3f}")
        print(f"HRF usage: {torch.bincount(hrf_index, minlength=n_hrfs)}")

    return best_results, hrf_index, r2_all_hrfs
