"""
Ultra-fast GPU-accelerated GLM solver
Core engine for fastfuncsim - handles all GLM fitting regardless of design matrix type

Supports multiple GLM variants:
- OLS (Ordinary Least Squares): Fast, assumes independence
- ARMA(1,1): Accounts for temporal autocorrelation (see arma_glm.py)
- Ridge: Regularization for correlated regressors (future)
"""

from __future__ import annotations

import warnings

import torch
from tqdm.auto import tqdm

from .design import convolve_hrf_microtime
from .design_builder import legendre_polynomials
from .memory import estimate_chunk_size
from .utils import get_device, to_tensor
from .xval import compute_r2_metric


class GLMResults:
    """Container for GLM results"""

    def __init__(self):
        """Initialize an empty GLM results container.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initializes all result attributes to ``None``.
        """
        self.betas = None  # (n_voxels, n_regressors) or (n_x, n_y, n_z, n_regressors)
        self.r2 = None  # (n_voxels,) or (n_x, n_y, n_z) - total model R²
        self.r2_partial = None  # (n_voxels, n_task_regressors) - partial R² per TASK regressor
        self.r2_partial_nuisance = (
            None  # (n_voxels, n_nuisance_regressors) - partial R² per NUISANCE regressor
        )
        self.r2_semipartial = (
            None  # (n_voxels, n_task_regressors) - semi-partial R² per TASK regressor
        )
        self.r2_semipartial_nuisance = (
            None  # (n_voxels, n_nuisance_regressors) - semi-partial R² per NUISANCE regressor
        )
        self.r2_run = None  # (n_voxels, n_runs) if provided
        self.residuals = None  # (n_voxels, n_timepoints) - optional
        self.predicted = None  # (n_voxels, n_timepoints) - optional
        self.meanvol = None  # Mean signal across time
        self.tstats = None  # (n_voxels, n_regressors)
        self.stderr = None  # (n_voxels, n_regressors)
        self.sigma2 = None  # (n_voxels,)
        self.fstats = None  # (n_voxels,)
        self.dof = None  # Degrees of freedom used for contrasts
        self.xtx_inv = None  # (n_regressors, n_regressors) - needed for contrasts
        self.original_shape = None  # Original spatial dimensions
        self.tr = None  # Repetition time (seconds)
        self.voxel_mask = None  # Optional boolean mask for sparse analyses
        self.full_shape = None  # Original spatial shape before masking
        self.affine = None  # Spatial affine if available
        self.nifti_header = None  # NIfTI header for output reconstruction (from analysis module)
        self.hrf_idx = None  # Selected HRF index (from hrf_selection module)
        self.r2_per_hrf = None  # R² for each HRF in the library (from hrf_selection)
        self.trial_labels = None  # Trial condition labels (from hrf_selection module)

        # GLT contrast results (computed in-loop, not post-hoc)
        self.contrast_labels = None  # List of contrast names
        self.contrast_betas = None  # (n_voxels, n_contrasts) - c'β estimates
        self.contrast_tstats = None  # (n_voxels, n_contrasts) - t-statistics
        self.contrast_fstats = None  # (n_voxels, n_contrasts) - F-statistics (for multi-row GLTs)

    def to_spatial(self):
        """Reshape results back to spatial dimensions if available"""
        if self.original_shape is None:
            warnings.warn("Original shape not stored, cannot reshape", stacklevel=2)
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
    n_timepoints: int,
    max_degree: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construct Legendre polynomial nuisance regressor matrix (for detrending)

    Uses orthogonal Legendre polynomials instead of monomials for better
    numerical stability and interpretability. Each run gets its own polynomial
    basis (via block_diag in calling code).

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    max_degree : int
        Maximum polynomial degree (0=constant, 1=linear, etc.)
    device : torch.device
        Device to create tensor on
    dtype : torch.dtype, default=torch.float32
        Data type for the polynomial matrix

    Returns
    -------
    poly_matrix : torch.Tensor
        (n_timepoints, max_degree+1) Legendre polynomial matrix
        Each column is a Legendre polynomial P_k(t) for k=0 to max_degree
    """
    if max_degree < 0:
        return torch.empty(n_timepoints, 0, device=device, dtype=dtype)

    # Use existing Legendre polynomial implementation from design_builder
    # (uses scipy.special.eval_legendre, well-tested, AFNI-compatible)
    poly_np = legendre_polynomials(n_timepoints, max_degree, normalize=False)

    # Convert to torch tensor with specified device and dtype
    poly_matrix = torch.as_tensor(poly_np, device=device, dtype=dtype)

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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
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

    # Compute predictions and residuals
    # NOTE: design @ betas.T gives (n_timepoints, n_regressors) @ (n_regressors, n_voxels) = (n_timepoints, n_voxels)
    # This can be huge! For 332k voxels × 2880 timepoints × 4 bytes = 3.5 GB!
    # Only compute if needed, otherwise compute residuals directly

    if want_predicted or want_residuals:
        # Need full predictions
        predicted_vals = (design @ betas.T).T  # (n_voxels, n_timepoints)
        residuals_vals = data - predicted_vals
    else:
        # Just compute residuals for R² (more memory efficient)
        predicted_vals = None
        residuals_vals = data - (design @ betas.T).T  # (n_voxels, n_timepoints)

    # Compute residual sum of squares for output
    ss_residual = (residuals_vals**2).sum(dim=1)

    # Compute R² using unified function (allows negative values for poor fits)
    r2 = compute_r2_metric(data, data - residuals_vals, metric="cod")

    return (
        betas,
        r2,
        ss_residual,
        residuals_vals if want_residuals else None,
        predicted_vals if want_predicted else None,
    )


def fit_glm(
    data: torch.Tensor | list,
    design: torch.Tensor | list,
    tr: float,
    max_poly_degree: int | list | None = None,
    extra_regressors: torch.Tensor | list | None = None,
    want_residuals: bool = False,
    want_predicted: bool = False,
    want_r2_run: bool = True,
    want_r2_partial: bool = False,
    r2_partial_mode: str = "full",  # "full" or "task" - how to compute partial R²
    want_r2_semipartial: bool = False,
    r2_semipartial_mode: str = "full",  # "full" or "task" - how to compute semi-partial R²
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
    preload_data_to_device: bool = True,
    use_double: bool = False,
    glt_labels: list | None = None,
    glt_matrices: list | None = None,
    task_indices: list | None = None,
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
    use_double : bool, default=False
        If True, use float64 precision (matches AFNI exactly, ~2x memory, ~1.5x slower).
        If False, use float32 precision (faster, tiny differences from AFNI).

    Returns
    -------
    results : GLMResults
        Object containing betas, R², etc.
    """
    # Setup precision
    dtype = torch.float64 if use_double else torch.float32

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
    for _i, d in enumerate(data):
        d_tensor = to_tensor(d, device=storage_device)
        d_tensor = d_tensor.to(dtype)
        if not preload_data_to_device and d_tensor.device.type != "cpu":
            d_tensor = d_tensor.cpu()

        if d_tensor.ndim == 4:  # (n_x, n_y, n_z, n_timepoints)
            if original_shape is None:
                original_shape = d_tensor.shape[:3]
            d_tensor = d_tensor.reshape(-1, d_tensor.shape[-1])  # (n_voxels, n_timepoints)
        elif d_tensor.ndim == 2:
            pass  # Already (n_voxels, n_timepoints)
        else:
            raise ValueError(
                f"Data must be 2D (n_voxels, n_timepoints) or 4D (nx, ny, nz, nt), got {d_tensor.ndim}D"
            )
        data_2d.append(d_tensor)

    n_voxels = data_2d[0].shape[0]

    # Convert design matrices to correct dtype
    design_2d = [to_tensor(d, device=device).to(dtype) for d in design]

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
        poly = construct_polynomial_matrix(n_tp, max_poly_degree[run_idx], device, dtype)

        # Add extra regressors if provided
        if extra_regressors is not None and extra_regressors[run_idx] is not None:
            extra = to_tensor(extra_regressors[run_idx], device=device).to(dtype)
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

    # Concatenate all runs
    data_concat = torch.cat(data_2d, dim=1)  # (n_voxels, total_timepoints)
    design_concat = torch.block_diag(*full_designs)  # Block diagonal for runs

    total_timepoints = data_concat.shape[1]

    # Determine chunk size
    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=total_timepoints,
            n_regressors=design_concat.shape[1],
            device=device,
            operation="glm",
        )

        # When streaming from CPU, additional overhead from:
        # 1. Copying data chunk to GPU (temporary during transfer)
        # 2. The design_concat is already on GPU and is block-diagonal (can be large)
        # Be more conservative to avoid OOM on the canonical fit
        if not preload_data_to_device and device.type == "cuda":
            # Reduce by 8x to account for all intermediates + design matrix
            chunk_size = chunk_size // 3  # LTD, trying this to speed things up
            chunk_size = max(500, chunk_size)  # At least 500 voxels per chunk

    if verbose:
        print(f"Fitting GLM: {n_voxels} voxels, {n_runs} runs, {n_task_regressors} task regressors")
        print(f"Processing in chunks of {chunk_size} voxels")
        if not preload_data_to_device:
            print("Streaming from CPU (reduced chunk size for memory safety)")

    # Also track per-run data for R² calculation
    run_boundaries = [0]
    for d in data_2d:
        run_boundaries.append(run_boundaries[-1] + d.shape[1])

    # Fit in chunks
    all_betas = []
    all_r2 = []
    all_r2_partial = [] if want_r2_partial else None
    all_r2_partial_nuisance = [] if want_r2_partial else None  # Store nuisance partial R² too
    all_r2_semipartial = [] if want_r2_semipartial else None
    all_r2_semipartial_nuisance = (
        [] if want_r2_semipartial else None
    )  # Store nuisance semi-partial R² too
    all_r2_run = [] if want_r2_run else None
    all_residuals = [] if want_residuals else None
    all_predicted = [] if want_predicted else None

    n_chunks = (n_voxels + chunk_size - 1) // chunk_size

    # Pre-compute matrices for statistics
    xtx = design_concat.T @ design_concat
    # No ridge regularization - match AFNI behavior (will fail if matrix is singular)
    # Use Cholesky decomposition for symmetric positive definite X'X (2x faster than general inverse)
    try:
        L = torch.linalg.cholesky(xtx)
        xtx_inv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        # Fallback to general inverse if Cholesky fails (rare for well-conditioned data)
        xtx_inv = torch.linalg.inv(xtx)

    # Determine which columns are "task" vs "nuisance"
    # If task_indices provided (from AFNI StimBots/StimTops), use those
    # Otherwise use the old logic (first n_task_regressors columns)
    if task_indices is not None:
        # User explicitly specified which columns are task regressors
        task_beta_indices = task_indices
        if verbose:
            print(
                f"Using {len(task_indices)} explicitly specified task regressors (from StimBots/StimTops)"
            )
    else:
        # Default: assume first n_task_regressors are task, rest are nuisance
        task_beta_indices = []
        reg_offset = 0
        for run_idx in range(n_runs):
            run_n_regressors = full_designs[run_idx].shape[1]
            task_beta_indices.extend(range(reg_offset, reg_offset + n_task_regressors))
            reg_offset += run_n_regressors

    if len(task_beta_indices) == 0:
        raise ValueError("No task regressors detected; cannot compute GLM statistics")

    task_idx_tensor = torch.tensor(task_beta_indices, device=device, dtype=torch.long)
    xtx_inv_task = xtx_inv.index_select(0, task_idx_tensor).index_select(1, task_idx_tensor)
    xtx_inv_task_diag = torch.diagonal(xtx_inv_task, dim1=0, dim2=1)

    # For partial R² and semi-partial R² computation: need diagonal of FULL (X'X)^-1 (all regressors)
    # Also needed for F-stat when n_regressors == 1
    xtx_inv_full_diag = torch.diagonal(xtx_inv, dim1=0, dim2=1)

    dof = design_concat.shape[0] - design_concat.shape[1]
    if dof <= 0:
        warnings.warn(
            "Non-positive degrees of freedom detected in GLM fit; statistics may be invalid", stacklevel=2
        )
        dof = max(1, dof)

    # Precompute inverse for F-stat (no ridge - match AFNI behavior)
    # For Full_Fstat: need (X'X) for ALL regressors
    # For task F-stats: need (X'X) for task regressors only
    try:
        L_full = torch.linalg.cholesky(xtx_inv)
        xtx_inv_full_inv = torch.cholesky_inverse(L_full)  # This is X'X for all regressors
    except torch.linalg.LinAlgError:
        xtx_inv_full_inv = torch.linalg.inv(xtx_inv)

    try:
        L_task = torch.linalg.cholesky(xtx_inv_task)
        xtx_inv_task_inv = torch.cholesky_inverse(L_task)
    except torch.linalg.LinAlgError:
        xtx_inv_task_inv = torch.linalg.inv(xtx_inv_task)
    n_task_params = xtx_inv_task.shape[0]

    # Setup GLT contrasts (if present) - compute in-loop like ARMA
    glt_contrasts_tensor = None
    n_contrasts = 0
    all_contrast_betas: list[torch.Tensor] = []
    all_contrast_tstats: list[torch.Tensor] = []

    if glt_labels and glt_matrices:
        n_contrasts = len(glt_labels)
        if verbose:
            old_mem_gb = (n_voxels * n_task_params * n_task_params * 8) / (1024**3) / 2
            new_mem_mb = (n_voxels * n_contrasts * 8) / (1024**2)
            print(f"📊 {n_contrasts} GLT contrasts will be computed in-loop (OLS)")
            print(
                f"   Memory: {new_mem_mb:.1f} MB (vs {old_mem_gb:.1f} GB if storing full covariance)"
            )

        # Convert GLT matrices to tensors on device
        glt_contrasts_list = []
        for glt_mat in glt_matrices:
            glt_tensor = torch.as_tensor(glt_mat, dtype=dtype, device=device)
            # Check if this is a single-row contrast (t-test) or multi-row (F-test)
            if glt_tensor.ndim == 1:
                # Already 1D - single-row contrast (t-test)
                glt_contrasts_list.append(glt_tensor)
            elif glt_tensor.ndim == 2 and glt_tensor.shape[0] == 1:
                # Shape (1, n_regressors) - squeeze to 1D for single-row contrast
                glt_contrasts_list.append(glt_tensor.squeeze(0))
            else:
                # Multi-row contrast (F-test) - not yet supported
                raise NotImplementedError(
                    f"Multi-row GLT contrasts (F-tests) not yet supported in OLS. "
                    f"Got shape {glt_tensor.shape}, expected (n_regressors,) or (1, n_regressors)"
                )
        glt_contrasts_tensor = torch.stack(glt_contrasts_list)  # (n_contrasts, n_regressors_full)

        # NOTE: GLT contrasts use the FULL design (all columns)
        # We fit the full design, so contrasts can involve any regressor (task or nuisance)
        # No filtering needed here!

    all_sigma2: list[torch.Tensor] = []
    all_tstats: list[torch.Tensor] = []
    all_stderr: list[torch.Tensor] = []
    all_fstats: list[torch.Tensor] = []

    # Progress bar for chunks
    chunk_iterator = range(n_chunks)
    if verbose and n_chunks > 1:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing voxel chunks", unit="chunk")

    for chunk_idx in chunk_iterator:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_voxels)

        chunk_data_cpu = data_concat[start_idx:end_idx]
        chunk_data = chunk_data_cpu if preload_data_to_device else chunk_data_cpu.to(device)

        # Fit this chunk on the compute device
        betas_dev, r2_dev, ss_residual_dev, residuals_dev, predicted_dev = fit_glm_chunk(
            chunk_data, design_concat, want_residuals, want_predicted
        )

        # Extract only task regressors (ignore nuisance)
        betas_task_dev = betas_dev[:, task_beta_indices]

        sigma2_dev = torch.clamp(ss_residual_dev / dof, min=0.0)

        # Standard errors and t-stats FOR TASK REGRESSORS (for main output)
        stderr_dev = torch.sqrt(
            torch.clamp(sigma2_dev.unsqueeze(1), min=0.0) * xtx_inv_task_diag.unsqueeze(0)
        )
        tstats_dev = betas_task_dev / (stderr_dev + 1e-10)

        # Partial R² per regressor: r²_partial_i = t²_i / (t²_i + df)
        # Compute for ALL regressors if requested, then split
        if want_r2_partial:
            # Compute t-stats for ALL betas (task + nuisance)
            stderr_full_dev = torch.sqrt(
                torch.clamp(sigma2_dev.unsqueeze(1), min=0.0) * xtx_inv_full_diag.unsqueeze(0)
            )
            tstats_full_dev = betas_dev / (stderr_full_dev + 1e-10)
            t_squared_full_dev = tstats_full_dev**2
            r2_partial_full_dev = t_squared_full_dev / (t_squared_full_dev + dof)

            # Split into task vs nuisance
            r2_partial_task_dev = r2_partial_full_dev[:, task_beta_indices]

            # Get nuisance indices (all columns NOT in task_beta_indices)
            all_indices = set(range(betas_dev.shape[1]))
            nuisance_indices = sorted(list(all_indices - set(task_beta_indices)))

            # Always extract nuisance partial R² (for -bout output)
            r2_partial_nuisance_dev = (
                r2_partial_full_dev[:, nuisance_indices] if len(nuisance_indices) > 0 else None
            )

            if r2_partial_mode == "task" and len(nuisance_indices) > 0:
                # Rescale task partial R² by variance remaining after nuisance
                # Sum nuisance partial R² (total variance explained by nuisance)
                r2_nuisance_total = r2_partial_nuisance_dev.sum(dim=1, keepdim=True)
                # Rescale: r²_task_adjusted = r²_task / (1 - R²_nuisance)
                # Clamp to avoid division by zero if nuisance explains ~100%
                denominator = torch.clamp(1.0 - r2_nuisance_total, min=0.01)
                r2_partial_dev = r2_partial_task_dev / denominator
            else:
                # Full mode: use raw partial R² values
                r2_partial_dev = r2_partial_task_dev

        # Semi-partial R² per regressor: r²_semi_i = partial_r²_i * (1 - R²_full)
        # This gives the unique variance contribution (sums to total R²)
        if want_r2_semipartial:
            # Need to compute semi-partial R² for ALL regressors first, then split
            # If we didn't compute partial R² above, we need to do it now
            if not want_r2_partial:
                # Compute partial R² for all regressors (needed for semi-partial)
                stderr_full_dev = torch.sqrt(
                    torch.clamp(sigma2_dev.unsqueeze(1), min=0.0) * xtx_inv_full_diag.unsqueeze(0)
                )
                tstats_full_dev = betas_dev / (stderr_full_dev + 1e-10)
                t_squared_full_dev = tstats_full_dev**2
                r2_partial_full_dev = t_squared_full_dev / (t_squared_full_dev + dof)

                # Split into task vs nuisance
                r2_partial_task_dev = r2_partial_full_dev[:, task_beta_indices]

                # Get nuisance indices
                all_indices = set(range(betas_dev.shape[1]))
                nuisance_indices = sorted(list(all_indices - set(task_beta_indices)))
                r2_partial_nuisance_dev = (
                    r2_partial_full_dev[:, nuisance_indices] if len(nuisance_indices) > 0 else None
                )

            # Compute semi-partial R² from partial R²
            # Formula: r²_semi = partial_r² * (1 - R²_full)
            variance_remaining = torch.clamp(1.0 - r2_dev.unsqueeze(1), min=0.0)

            # Task semi-partial R²
            r2_semipartial_task_dev = r2_partial_task_dev * variance_remaining

            # Nuisance semi-partial R²
            r2_semipartial_nuisance_dev = (
                r2_partial_nuisance_dev * variance_remaining
                if r2_partial_nuisance_dev is not None
                else None
            )

            # Apply rescaling mode for task regressors
            if r2_semipartial_mode == "task" and r2_semipartial_nuisance_dev is not None:
                # Rescale by variance remaining after nuisance
                # First sum nuisance semi-partial R²
                r2_semi_nuisance_total = r2_semipartial_nuisance_dev.sum(dim=1, keepdim=True)
                denominator = torch.clamp(1.0 - r2_semi_nuisance_total, min=0.01)
                r2_semipartial_dev = r2_semipartial_task_dev / denominator
            else:
                # Full mode: use raw semi-partial R² values
                r2_semipartial_dev = r2_semipartial_task_dev

        # F-statistics: Test ALL task regressors
        # NOTE: Confirmed from AFNI source (3dREMLfit.c:2900-2906) that "Full_Fstat"
        # tests ONLY stimuli (task regressors), NOT baseline/nuisance regressors.
        # The "Full" GLT uses create_subset_matrix over stim_bot[jj]:stim_top[jj].
        if n_task_params == 1:
            fstats_dev = tstats_dev[:, 0] ** 2
        else:
            quad_dev = torch.einsum("bi,ij,bj->b", betas_task_dev, xtx_inv_task_inv, betas_task_dev)
            fstats_dev = quad_dev / (n_task_params * sigma2_dev + 1e-10)

        # GLT CONTRASTS (OLS): Compute in-loop using FULL betas
        # For each contrast c: compute c'β and Var(c'β) = c' (σ² (X'X)^-1) c
        if glt_contrasts_tensor is not None:
            chunk_size_actual = betas_dev.shape[0]

            # Compute c'β for all contrasts at once
            # glt_contrasts_tensor: (n_contrasts, n_regressors_full)
            # betas_dev: (chunk_size, n_regressors_full) - FULL betas including nuisance
            # Result: (chunk_size, n_contrasts)
            contrast_betas_dev = torch.mm(betas_dev, glt_contrasts_tensor.T)

            # Compute Var(c'β) = c' Var(β) c for each contrast
            # Var(β) = σ² (X'X)^-1, we have xtx_inv: (n_reg_full, n_reg_full)
            # For each voxel: Var(β_voxel) = sigma2[voxel] * xtx_inv

            # VECTORIZED: Compute all c' (X'X)^-1 c at once (faster than loop)
            # glt_contrasts_tensor: (n_contrasts, n_regressors)
            # xtx_inv: (n_regressors, n_regressors)
            # Result: (n_contrasts,) - same for all voxels
            contrast_xtx_inv_c = torch.einsum(
                "cr,rs,cs->c",
                glt_contrasts_tensor,  # (n_contrasts, n_regressors)
                xtx_inv,  # (n_regressors, n_regressors)
                glt_contrasts_tensor,  # (n_contrasts, n_regressors)
            )

            # Broadcast to all voxels: Var(c'β) = σ² * c' (X'X)^-1 c
            # sigma2_dev: (chunk_size,), contrast_xtx_inv_c: (n_contrasts,)
            # Result: (chunk_size, n_contrasts)
            contrast_vars_dev = sigma2_dev.unsqueeze(1) * contrast_xtx_inv_c.unsqueeze(0)

            # Compute t-statistics for contrasts
            contrast_se_dev = torch.sqrt(torch.clamp(contrast_vars_dev, min=0.0))
            contrast_tstats_dev = contrast_betas_dev / (contrast_se_dev + 1e-10)

            # Store (will be moved to CPU below)
            contrast_betas_chunk = contrast_betas_dev
            contrast_tstats_chunk = contrast_tstats_dev

        # Move outputs back to CPU for aggregation
        if preload_data_to_device:
            betas_cpu = betas_task_dev
            r2_cpu = r2_dev
            sigma2_cpu = sigma2_dev
            stderr_cpu = stderr_dev
            tstats_cpu = tstats_dev
            fstats_cpu = fstats_dev
            r2_partial_cpu = r2_partial_dev if want_r2_partial else None
            r2_partial_nuisance_cpu = (
                r2_partial_nuisance_dev
                if (want_r2_partial and r2_partial_nuisance_dev is not None)
                else None
            )
            r2_semipartial_cpu = r2_semipartial_dev if want_r2_semipartial else None
            r2_semipartial_nuisance_cpu = (
                r2_semipartial_nuisance_dev
                if (want_r2_semipartial and r2_semipartial_nuisance_dev is not None)
                else None
            )
            residuals_cpu = residuals_dev if residuals_dev is not None else None
            predicted_cpu = predicted_dev if predicted_dev is not None else None
        else:
            betas_cpu = betas_task_dev.cpu()
            r2_cpu = r2_dev.cpu()
            sigma2_cpu = sigma2_dev.cpu()
            stderr_cpu = stderr_dev.cpu()
            tstats_cpu = tstats_dev.cpu()
            fstats_cpu = fstats_dev.cpu()
            r2_partial_cpu = r2_partial_dev.cpu() if want_r2_partial else None
            r2_partial_nuisance_cpu = (
                r2_partial_nuisance_dev.cpu()
                if (want_r2_partial and r2_partial_nuisance_dev is not None)
                else None
            )
            r2_semipartial_cpu = r2_semipartial_dev.cpu() if want_r2_semipartial else None
            r2_semipartial_nuisance_cpu = (
                r2_semipartial_nuisance_dev.cpu()
                if (want_r2_semipartial and r2_semipartial_nuisance_dev is not None)
                else None
            )
            residuals_cpu = residuals_dev.cpu() if residuals_dev is not None else None
            predicted_cpu = predicted_dev.cpu() if predicted_dev is not None else None

        all_betas.append(betas_cpu)
        all_r2.append(r2_cpu)

        if want_r2_partial and r2_partial_cpu is not None:
            all_r2_partial.append(r2_partial_cpu)
        if want_r2_partial and r2_partial_nuisance_cpu is not None:
            all_r2_partial_nuisance.append(r2_partial_nuisance_cpu)

        if want_r2_semipartial and r2_semipartial_cpu is not None:
            all_r2_semipartial.append(r2_semipartial_cpu)
        if want_r2_semipartial and r2_semipartial_nuisance_cpu is not None:
            all_r2_semipartial_nuisance.append(r2_semipartial_nuisance_cpu)

        if want_residuals and residuals_cpu is not None:
            all_residuals.append(residuals_cpu)
        if want_predicted and predicted_cpu is not None:
            all_predicted.append(predicted_cpu)

        all_sigma2.append(sigma2_cpu)
        all_tstats.append(tstats_cpu)
        all_stderr.append(stderr_cpu)
        all_fstats.append(fstats_cpu)

        # Store contrast results
        if glt_contrasts_tensor is not None:
            if preload_data_to_device:
                all_contrast_betas.append(contrast_betas_chunk)
                all_contrast_tstats.append(contrast_tstats_chunk)
            else:
                all_contrast_betas.append(contrast_betas_chunk.cpu())
                all_contrast_tstats.append(contrast_tstats_chunk.cpu())

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

                    # Extract the appropriate design columns and betas
                    if task_indices is not None:
                        # When task_indices is provided, extract those specific columns
                        # task_indices are global indices, so use them directly for single run
                        # or compute per-run indices for multi-run
                        if n_runs == 1:
                            # Single run: use task_indices directly
                            run_design_task = run_design[:, task_beta_indices]
                            run_betas_dev = betas_task_dev  # All task betas
                        else:
                            # Multi-run: need to figure out which task_indices belong to this run
                            # This is complex - for now, use all betas (will need refinement)
                            run_design_task = run_design[:, :n_task_regressors]
                            run_betas_dev = betas_task_dev[
                                :,
                                run_idx * n_task_regressors : (run_idx + 1) * n_task_regressors,
                            ]
                    else:
                        # Original behavior: first n_task_regressors columns
                        run_design_task = run_design[:, :n_task_regressors]
                        run_betas_dev = betas_task_dev[
                            :,
                            run_idx * n_task_regressors : (run_idx + 1) * n_task_regressors,
                        ]

                    run_pred_dev = (run_design_task @ run_betas_dev.T).T
                    run_pred_cpu = run_pred_dev if preload_data_to_device else run_pred_dev.cpu()

                run_mean = run_data_cpu.mean(dim=1, keepdim=True)
                run_ss_total = ((run_data_cpu - run_mean) ** 2).sum(dim=1)
                run_residuals_cpu = run_data_cpu - run_pred_cpu
                run_ss_residual = (run_residuals_cpu**2).sum(dim=1)
                run_r2_cpu = 1 - run_ss_residual / (run_ss_total + 1e-10)
                run_r2_cpu = torch.clamp(run_r2_cpu, 0, 1)
                r2_run.append(run_r2_cpu)

            all_r2_run.append(torch.stack(r2_run, dim=1))  # (chunk_voxels, n_runs)

        # Release GPU tensors for this chunk and clear cache periodically
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

        # Clear GPU cache every 10 chunks to prevent fragmentation
        if not preload_data_to_device and device.type == "cuda" and chunk_idx % 10 == 0:
            torch.cuda.empty_cache()

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
    # Store (X'X)^-1 for contrast computation (task regressors only, matching betas)
    results.xtx_inv = xtx_inv_task.to(concat_device)

    if want_r2_run:
        results.r2_run = torch.cat(all_r2_run, dim=0).to(concat_device)
    if want_r2_partial:
        results.r2_partial = torch.cat(all_r2_partial, dim=0).to(concat_device)
        # Also store nuisance partial R² if we have any
        if all_r2_partial_nuisance and len(all_r2_partial_nuisance) > 0:
            results.r2_partial_nuisance = torch.cat(all_r2_partial_nuisance, dim=0).to(
                concat_device
            )
    if want_r2_semipartial:
        results.r2_semipartial = torch.cat(all_r2_semipartial, dim=0).to(concat_device)
        # Also store nuisance semi-partial R² if we have any
        if all_r2_semipartial_nuisance and len(all_r2_semipartial_nuisance) > 0:
            results.r2_semipartial_nuisance = torch.cat(all_r2_semipartial_nuisance, dim=0).to(
                concat_device
            )
    if want_residuals:
        results.residuals = torch.cat(all_residuals, dim=0).to(concat_device)
    if want_predicted:
        results.predicted = torch.cat(all_predicted, dim=0).to(concat_device)

    # Concatenate GLT contrast results
    if glt_contrasts_tensor is not None:
        results.contrast_labels = glt_labels
        results.contrast_betas = torch.cat(all_contrast_betas, dim=0).to(concat_device)
        results.contrast_tstats = torch.cat(all_contrast_tstats, dim=0).to(concat_device)

    if verbose:
        print(f"GLM complete. Mean R² = {results.r2.mean().item():.3f}")
        if glt_contrasts_tensor is not None:
            print(f"  Computed {n_contrasts} GLT contrasts")

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
    data: torch.Tensor | list,
    design: torch.Tensor | list,
    hrf_library: torch.Tensor,
    tr: float,
    microtime_dt: float = 0.1,
    microtime_onset: int = 0,
    n_timepoints: int | None = None,
    **kwargs,
) -> tuple[GLMResults, torch.Tensor, torch.Tensor]:
    """
    Fit GLM with HRF library and select best HRF per voxel

    Parameters
    ----------
    data : torch.Tensor or list
        fMRI data
    design : torch.Tensor or list
        Design matrix (NOT yet convolved with HRF) at microtime_dt resolution.
        Shape: (n_microtime_points, n_conditions) where
        n_microtime_points = n_timepoints * (tr / microtime_dt)
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_timepoints) library of HRF candidates at microtime_dt resolution
    tr : float
        Repetition time in seconds
    microtime_dt : float, default=0.1
        Microtime resolution in seconds. Both design and hrf_library should be
        at this resolution.
    microtime_onset : int, default=0
        Which microtime bin within each TR to sample (0-indexed).
    n_timepoints : int, optional
        Number of TR timepoints. If None, inferred from design shape.
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

    # Calculate bins per TR
    bins_per_tr = int(round(tr / microtime_dt))

    # Determine n_timepoints for output
    if n_timepoints is None:
        # Infer from design shape
        n_timepoints = design.shape[0] // bins_per_tr
    if verbose:
        print(f"  Using microtime: dt={microtime_dt}s ({bins_per_tr} bins/TR)")

    # Fit GLM for each HRF
    all_r2 = []
    all_results = []

    for hrf_idx in range(n_hrfs):
        if verbose:
            print(f"  HRF {hrf_idx + 1}/{n_hrfs}")

        # Convolve design with this HRF
        hrf = hrf_library[hrf_idx]

        # Convolve at microtime resolution and downsample to TR
        convolved_design = convolve_hrf_microtime(
            design,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            microtime_onset=microtime_onset,
            device=device,
        )

        # Fit GLM for each HRF
        fit_kwargs = kwargs.copy()
        fit_kwargs["verbose"] = False  # Suppress inner GLM progress

        results = fit_glm(data, convolved_design, tr, **fit_kwargs)
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
