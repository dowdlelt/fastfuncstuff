"""
GPU-accelerated ICA for fMRI data with stability analysis

Implements FastICA with GPU acceleration and automatic component selection
based on stability across multiple random initializations.

Key Features
------------
- GPU-accelerated FastICA using PyTorch
- PCA dimensionality reduction (required preprocessing for ICA)
- Stability-based component selection via repeated runs
- Both spatial and temporal ICA modes
- Memory-efficient processing for large fMRI datasets
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.utils import get_device, to_tensor


class _RowCenteredPCAState:
    """Lightweight PCA state from MELODIC-style row-centered PCA.

    Stores enough information for FastICA.transform() and inverse_transform()
    without depending on the column-centering PCA class.
    """

    def __init__(
        self,
        components: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        row_mean: torch.Tensor,
        device: torch.device,
    ):
        self.components_ = components  # (k, V) — whitened spatial components
        self.n_components_ = components.shape[0]
        self.explained_variance_ = eigenvalues  # (k,) — PCA eigenvalues
        self._eigenvectors = eigenvectors  # (T, k)
        self._eigenvalues = eigenvalues  # (k,)
        self._row_mean = row_mean  # (T, 1) — not used for transform, stored for reference
        self.device = device

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project new data: X @ components^T -> (T, k)."""
        X = to_tensor(X, device=self.device)
        return X @ self.components_.T


class FastICA:
    """
    GPU-accelerated FastICA for fMRI data

    Implements the FastICA algorithm using PyTorch for GPU acceleration.
    Includes PCA dimensionality reduction as a required preprocessing step.

    Parameters
    ----------
    n_components : int, optional
        Number of ICA components to extract. If None, uses number of
        PCA components from dimensionality reduction.
    pca_components : int, float, or str, optional
        Number of PCA components for dimensionality reduction:
        - float (0.0-1.0): Keep components explaining this fraction of variance
        - int: Keep exactly this many components
        - str: Advanced selection method ('mle', 'knee')
        - None: Use n_components (default: 0.85 = 85% variance)
    max_iter : int, default=200
        Maximum iterations for FastICA convergence
    tol : float, default=1e-4
        Tolerance for convergence
    fun : str, default='logcosh'
        Nonlinearity function: 'logcosh', 'exp', or 'cube'
    random_state : int, optional
        Random seed for reproducibility
    whiten : bool, default=True
        Whether to whiten data after PCA (recommended for ICA)
    device : torch.device, optional
        Device to use for computation

    Attributes
    ----------
    components_ : torch.Tensor, shape (n_components, n_features)
        ICA spatial maps (for spatial ICA) or temporal patterns
    mixing_ : torch.Tensor, shape (n_samples, n_components)
        ICA mixing matrix (timeseries for spatial ICA)
    pca_ : _RowCenteredPCAState
        Fitted PCA state (MELODIC-style row-centered)
    mean_ : torch.Tensor
        Mean used for centering
    n_iter_ : int
        Number of iterations run

    Examples
    --------
    # Spatial ICA: extract 25 independent spatial components
    >>> ica = FastICA(n_components=25, pca_components=0.85)
    >>> timeseries = ica.fit_transform(fmri_data)  # (n_timepoints, n_voxels)
    >>> spatial_maps = ica.components_  # (25, n_voxels)
    >>> timeseries = ica.mixing_  # (n_timepoints, 25)

    # Temporal ICA: transpose input
    >>> ica = FastICA(n_components=20)
    >>> components = ica.fit_transform(fmri_data.T)  # (n_voxels, n_timepoints)
    """

    def __init__(
        self,
        n_components: int | None = None,
        pca_components: int | float | str | None = 0.85,
        max_iter: int = 200,
        tol: float = 1e-4,
        fun: str = "logcosh",
        random_state: int | None = None,
        whiten: bool = True,
        device: torch.device | None = None,
    ):
        self.n_components = n_components
        self.pca_components = pca_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.random_state = random_state
        self.whiten = whiten
        self.device = device if device is not None else get_device()

        # Fitted attributes
        self.components_ = None
        self.mixing_ = None
        self.pca_ = None
        self.mean_ = None
        self.n_iter_ = None

    @torch.inference_mode()
    def fit(self, X: np.ndarray | torch.Tensor) -> FastICA:
        """
        Fit ICA on data

        Uses MELODIC-style row-centered PCA for whitening: covariance is
        computed from row-centered (spatial-mean-subtracted) data, but
        whitening is applied to the original (non-centered) data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (n_timepoints, n_voxels)

        Returns
        -------
        self : FastICA
            Fitted ICA object
        """
        # Convert to tensor
        X = to_tensor(X, device=self.device)

        n_samples, n_features = X.shape  # T, V

        # Step 1: PCA with MELODIC-style centering
        # MELODIC first removes the temporal mean (remmean(alldat,2), i.e.
        # per-voxel/column mean) before PCA.  Then cov_r row-centres on top
        # (subtracts spatial mean per timepoint).  Whitening is applied to
        # the temporally-demeaned data.  We replicate both steps.
        self.mean_ = X.mean(dim=0, keepdim=True)  # (1, V) temporal mean per voxel
        X = X - self.mean_  # temporal demean (column-centre)

        row_mean = X.mean(dim=1, keepdim=True)  # (T, 1) spatial mean per timepoint
        # Memory-efficient: (X-m)(X-m)^T = X@X^T - V*m*m^T
        cov_t = (X @ X.T - n_features * (row_mean @ row_mean.T)) / float(n_features)

        # Eigendecomposition (ascending order from eigvalsh)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_t)
        del cov_t
        # Flip to descending order
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        eigenvalues = torch.clamp(eigenvalues, min=0)

        # Determine number of PCA components to keep
        n_max = min(n_samples, n_features)
        pca_comp = self.pca_components
        if pca_comp is None:
            pca_n_components = n_max
        elif isinstance(pca_comp, float) and 0.0 < pca_comp < 1.0:
            # Variance fraction
            total_var = eigenvalues.sum()
            cumvar = torch.cumsum(eigenvalues, dim=0) / total_var
            pca_n_components = int(torch.searchsorted(cumvar, pca_comp).item()) + 1
            pca_n_components = min(pca_n_components, n_max)
        elif isinstance(pca_comp, int):
            pca_n_components = min(pca_comp, n_max)
        else:
            pca_n_components = n_max

        # Keep top k eigenvalues/vectors
        evals_k = eigenvalues[:pca_n_components]  # (k,)
        evecs_k = eigenvectors[:, :pca_n_components]  # (T, k)

        # Store for later use (transform, etc.)
        self._pca_n_components = pca_n_components
        self._pca_eigenvalues = evals_k
        self._pca_eigenvectors = evecs_k
        self._row_mean = row_mean

        # Whitening matrix: diag(1/sqrt(lambda)) @ U^T  -> (k, T)
        # Applied to original (not row-centred) data
        white = torch.diag(1.0 / torch.sqrt(evals_k + 1e-12)) @ evecs_k.T  # (k, T)
        # Dewhitening: U @ diag(sqrt(lambda))  -> (T, k)
        dewhite = evecs_k @ torch.diag(torch.sqrt(evals_k))  # (T, k)

        # Whitened spatial components: white @ X -> (k, V)
        # Applied to ORIGINAL data (not row-centred), matching MELODIC
        pca_components_all = white @ X  # (k, V)

        # Projected timecourses: X @ components^T, but also available as
        # dewhite directly (since white @ X gives components, and
        # X_pca = X @ (white @ X)^T... but simpler: X_pca = dewhite scaled)
        # Actually: X_pca_i = U_i * sqrt(lambda_i) for whitened PCA
        # The mixing matrix needs projected timeseries, compute directly:
        X_pca = X @ pca_components_all.T  # (T, k)

        # Determine number of ICA components
        n_comp_req = self.n_components
        if n_comp_req is None:
            n_components = pca_n_components
        else:
            n_components = min(int(cast(int, n_comp_req)), pca_n_components)

        # Step 2: Run FastICA on TOP PCA spatial components (spatial ICA)
        pca_components = pca_components_all[:n_components]  # (n_ica, V)

        # The whitened components already have Cov ≈ I by construction:
        # Cov = (white @ X)(white @ X)^T / V
        #     = white @ (X @ X^T / V) @ white^T
        # Since white = Λ^(-1/2) U^T and cov_r ≈ U Λ U^T (row-centred cov),
        # this is approximately I. No additional sqrt(V) scaling needed.
        W, n_iter = self._fastica(pca_components, n_components)
        self.n_iter_ = n_iter

        # Step 3: Extract ICA spatial maps
        # W @ pca_components = (n_ica, n_ica) @ (n_ica, V) = (n_ica, V)
        self.components_ = W @ pca_components

        # Step 4: Compute mixing matrix (ICA timecourses)
        # MELODIC: timecourses = dewhite @ pinv(W)
        X_pca_top = X_pca[:, :n_components]  # (T, n_ica)
        self.mixing_ = X_pca_top @ torch.linalg.pinv(W)

        # Store PCA-like state for compatibility with transform()
        self.pca_ = _RowCenteredPCAState(
            components=pca_components_all,
            eigenvectors=evecs_k,
            eigenvalues=evals_k,
            row_mean=row_mean,
            device=self.device,
        )

        return self

    def transform(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Transform data to ICA component space (get timeseries)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        S : torch.Tensor, shape (n_samples, n_components)
            ICA component timeseries
        """
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted before transform()")

        # Convert to tensor
        X = to_tensor(X, device=self.device)

        # PCA transform
        X_pca = self.pca_.transform(X)

        # ICA mixing (get timeseries)
        # Reconstruct unmixing matrix from components
        W = self.components_ @ torch.linalg.pinv(self.pca_.components_)
        S = X_pca @ W.T

        return S

    def fit_transform(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Fit ICA and return component timeseries

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        S : torch.Tensor, shape (n_samples, n_components)
            ICA component timeseries
        """
        self.fit(X)
        return self.mixing_

    def inverse_transform(self, S: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Reconstruct data from ICA components

        Parameters
        ----------
        S : array-like, shape (n_samples, n_components)
            ICA component timeseries

        Returns
        -------
        X_reconstructed : torch.Tensor, shape (n_samples, n_features)
            Reconstructed data
        """
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted before inverse_transform()")

        # Convert to tensor
        S = to_tensor(S, device=self.device)

        # Reconstruct: X = S @ components
        X_reconstructed = S @ self.components_

        return X_reconstructed

    def _fastica(
        self,
        X: torch.Tensor,
        n_components: int,
        w_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Core FastICA algorithm (Symmectric / Parallel implementation)

        Parameters
        ----------
        X : torch.Tensor, shape (n_features, n_samples)
            Whitened data (transposed)
        n_components : int
            Number of components to extract
        w_init : torch.Tensor, optional
            Initial unmixing matrix (n_components, n_features)

        Returns
        -------
        W : torch.Tensor, shape (n_components, n_features)
            Unmixing matrix
        n_iter : int
            Number of iterations
        """
        n_features, n_samples = X.shape

        # Set random seed if provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Initialize unmixing matrix
        if w_init is not None:
            W = w_init.to(self.device).type(X.dtype)
        else:
            W = torch.randn(n_components, n_features, device=self.device, dtype=X.dtype)

        # Initial symmetric decorrelation
        W = self._symmetric_decorrelation(W)

        # Get combined nonlinearity function (returns g and g_prime_mean in-place)
        g_and_gprime = self._get_nonlinearity_combined(self.fun)

        # Optimization: pre-compute float n_samples for division
        scale = 1.0 / n_samples

        # FastICA iterations
        for n_iter in range(self.max_iter):  # noqa: B007
            W_old = W.clone()

            # 1. Linear projection
            # W: (n_comp, n_feat), X: (n_feat, n_samp) -> wx: (n_comp, n_samp)
            wx = W @ X

            # 2. Apply nonlinearity in-place, get g_wx and g_prime_mean
            # g_wx is wx modified in-place; g_prime_mean is (n_comp, 1)
            g_wx, g_prime_mean = g_and_gprime(wx)

            # 3. Update rule (Symmetric)
            # (g_wx @ X.T) / N -> (n_comp, n_samp) @ (n_samp, n_feat) -> (n_comp, n_feat)
            term1 = (g_wx @ X.T) * scale
            del g_wx  # free (n_comp, n_samp) before allocating term2

            # Term 2: E[g'(w^T x)] w -> g_prime_mean * W
            W = term1 - g_prime_mean * W

            # 4. Symmetric Decorrelation
            W = self._symmetric_decorrelation(W)

            # 5. Check convergence
            # Distance between subspaces: 1 - min(abs(diag(W @ W_old.T)))
            # Ideally W @ W_old.T should be Identity (or Permutation/Sign matrix)
            # Since we track trajectory of W, we check correlation of rows
            # abs(diag(W @ W_old.T)) is correlation of w_new_i with w_old_i
            lim = torch.abs(torch.diag(W @ W_old.T))
            delta = torch.max(1 - torch.abs(lim))

            if delta < self.tol:
                break

        return W, n_iter + 1

    @staticmethod
    def _symmetric_decorrelation(W: torch.Tensor) -> torch.Tensor:
        """
        Symmetric decorrelation (whitening) of weight matrix

        W_new = (W @ W.T)^(-1/2) @ W

        Parameters
        ----------
        W : torch.Tensor, shape (n_components, n_features)
            Weight matrix

        Returns
        -------
        W_decorr : torch.Tensor, shape (n_components, n_features)
            Decorrelated weight matrix
        """
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        return U @ Vt

    @staticmethod
    def _get_nonlinearity(fun: str):
        """
        Get nonlinearity function and its derivative

        Parameters
        ----------
        fun : str
            Function name: 'logcosh', 'exp', or 'cube'

        Returns
        -------
        g : callable
            Nonlinearity function
        g_prime : callable
            Derivative of nonlinearity
        """
        if fun == "logcosh":
            alpha = 1.0

            def g_logcosh(x):
                return torch.tanh(alpha * x)

            def g_prime_logcosh(x):
                return alpha * (1 - torch.tanh(alpha * x) ** 2)

            return g_logcosh, g_prime_logcosh

        elif fun == "exp":

            def g_exp(x):
                exp_x = torch.exp(-(x**2) / 2)
                return x * exp_x

            def g_prime_exp(x):
                exp_x = torch.exp(-(x**2) / 2)
                return (1 - x**2) * exp_x

            return g_exp, g_prime_exp

        elif fun == "cube":

            def g_cube(x):
                return x**3

            def g_prime_cube(x):
                return 3 * x**2

            return g_cube, g_prime_cube

        elif fun == "pow3":
            # MELODIC default: skewness-based contrast with custom factors
            # Update: W = 3*E[X*u^2] - E[u]*W
            def g_pow3(x):
                return 3.0 * x**2

            def g_prime_pow3(x):
                return x  # E[u], not 2u — MELODIC's custom factor

            return g_pow3, g_prime_pow3

        else:
            raise ValueError(
                f"Unknown nonlinearity: '{fun}'. "
                "Use 'logcosh', 'exp', 'cube', or 'pow3'"
            )

    @staticmethod
    def _get_nonlinearity_combined(fun: str):
        """Return a function that computes g(x) in-place and g_prime_mean.

        Memory-efficient: computes g and mean(g') together, reusing
        intermediates and avoiding full-size temporary allocations.

        Returns
        -------
        g_and_gprime : callable
            Takes x (n_comp, n_samp), modifies x in-place to g(x),
            returns (g_x, g_prime_mean) where g_prime_mean is (n_comp, 1).
        """
        if fun == "logcosh":

            def g_logcosh_combined(x):
                # In-place tanh (alpha=1.0)
                x.tanh_()
                # g_prime = 1 - tanh^2; compute mean without full materialization
                # Use chunked reduction to avoid (n_comp, n_samp) temporary
                n_samp = x.shape[1]
                chunk = min(n_samp, 200_000)
                sq_sum = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                for j in range(0, n_samp, chunk):
                    sq_sum += x[:, j : j + chunk].pow(2).sum(dim=1, keepdim=True)
                g_prime_mean = 1.0 - sq_sum / n_samp
                return x, g_prime_mean

            return g_logcosh_combined

        elif fun == "exp":

            def g_exp_combined(x):
                # exp(-x^2/2) in-place
                x.pow_(2).mul_(-0.5).exp_()  # x is now exp(-x_orig^2/2)
                # Need x_orig * exp_x for g, but we lost x_orig...
                # We need a different approach: compute from scratch with one temp
                # Actually, we can't recover x_orig after in-place pow_.
                # Use a buffer for exp_x instead.
                raise NotImplementedError(
                    "exp nonlinearity not yet optimized for memory. Use logcosh."
                )

            return g_exp_combined

        elif fun == "cube":

            def g_cube_combined(x):
                # g = x^3, g_prime = 3x^2
                # Compute g_prime_mean = 3 * mean(x^2) first (before modifying x)
                n_samp = x.shape[1]
                chunk = min(n_samp, 200_000)
                sq_sum = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                for j in range(0, n_samp, chunk):
                    sq_sum += x[:, j : j + chunk].pow(2).sum(dim=1, keepdim=True)
                g_prime_mean = 3.0 * sq_sum / n_samp
                x.pow_(3)
                return x, g_prime_mean

            return g_cube_combined

        elif fun == "pow3":

            def g_pow3_combined(x):
                # MELODIC's pow3 (default nonlinearity), NOT standard g/g'.
                # MELODIC update: W = 3*E[X*u^2] - E[u]*W
                # To achieve this through our generic loop (term1 - g_prime_mean * W):
                #   term1 = (g(wx) @ X^T) / N, so g(u) = 3*u^2 → term1 = 3*E[X*u^2]
                #   g_prime_mean = E[u] (not 2*E[u])
                n_samp = x.shape[1]
                chunk = min(n_samp, 200_000)
                x_sum = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                for j in range(0, n_samp, chunk):
                    x_sum += x[:, j : j + chunk].sum(dim=1, keepdim=True)
                g_prime_mean = x_sum / n_samp  # E[u], not 2*E[u]
                x.pow_(2).mul_(3.0)  # g(u) = 3*u^2
                return x, g_prime_mean

            return g_pow3_combined

        else:
            raise ValueError(
                f"Unknown nonlinearity: '{fun}'. "
                "Use 'logcosh', 'exp', 'cube', or 'pow3'"
            )

    @torch.inference_mode()
    def compute_variance_explained(
        self,
        X: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute variance explained by each ICA component

        ICA components are not ordered by variance, but it's useful to know
        how much variance each component explains for interpretation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data (same as used for fit)

        Returns
        -------
        variance_explained : torch.Tensor, shape (n_components,)
            Variance explained by each component
        variance_ratio : torch.Tensor, shape (n_components,)
            Fraction of total variance explained by each component
        """
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted first")

        # Convert to tensor
        X = to_tensor(X, device=self.device)

        # Mean-center data
        X_centered = X - X.mean(dim=0)

        # Total variance
        total_var = (X_centered**2).sum()

        # Vectorized: mixing_ (T, K) @ components_ (K, V) reconstructs per-component
        # Variance per component = sum of squared reconstruction per component
        # = sum_t sum_v (mixing_[t, k] * components_[k, v])^2
        # = sum_v (mixing_[:, k]^2.sum()) * (components_[k, v]^2)
        # = (mixing_^2).sum(0) * (components_^2).sum(1)
        mix_sq_sum = (self.mixing_**2).sum(dim=0)  # (K,)
        comp_sq_sum = (self.components_**2).sum(dim=1)  # (K,)
        variance_explained = mix_sq_sum * comp_sq_sum

        # Compute ratios
        variance_ratio = variance_explained / total_var

        return variance_explained, variance_ratio

    def to_dict(self) -> dict:
        """Export ICA parameters to dictionary"""
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted first")

        return {
            "components": self.components_.cpu().numpy(),
            "mixing": self.mixing_.cpu().numpy(),
            "pca": self.pca_.to_dict(),
            "n_components": self.n_components,
            "n_iter": self.n_iter_,
        }


def ica_stability_analysis(
    X: np.ndarray | torch.Tensor,
    n_components: int,
    pca_components: int | float | str | None = 0.85,
    n_runs: int = 100,
    n_jobs: int = 1,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """
    Analyze ICA stability across multiple random initializations

    Runs ICA many times with different random seeds and measures the
    stability/reproducibility of extracted components. More stable
    components indicate more reliable decomposition.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        fMRI data
    n_components : int
        Number of ICA components to extract
    pca_components : int, float, or str, optional
        PCA dimensionality reduction (default: 0.85)
    n_runs : int, default=100
        Number of ICA runs with different random seeds
    n_jobs : int, default=1
        Number of parallel jobs (currently unused, runs sequentially)
    device : torch.device, optional
        Computation device
    verbose : bool, default=True
        Show progress bar

    Returns
    -------
    results : dict
        Dictionary with:
        - 'components_list': list of component matrices from each run
        - 'stability_scores': stability score for each component
        - 'mean_components': average spatial maps across runs
        - 'std_components': standard deviation across runs

    Examples
    --------
    >>> # Test stability for 25 components
    >>> results = ica_stability_analysis(fmri_data, n_components=25, n_runs=100)
    >>> print(f"Mean stability: {results['stability_scores'].mean():.3f}")
    >>>
    >>> # Find most stable components
    >>> stable_idx = results['stability_scores'] > 0.8
    >>> print(f"Highly stable components: {stable_idx.sum()}")
    """
    device = device if device is not None else get_device()

    # Convert to tensor
    X = to_tensor(X, device=device)

    # Run ICA multiple times
    components_list = []

    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc="Running ICA stability analysis")

    for i in iterator:
        ica = FastICA(
            n_components=n_components,
            pca_components=pca_components,
            random_state=i,
            device=device,
        )
        ica.fit(X)
        components_list.append(ica.components_.cpu().numpy())

    # Convert to array
    components_array = np.array(components_list)  # (n_runs, n_components, n_features)

    # Compute stability: correlation between runs
    # For each component, find best match across runs and compute correlation
    stability_scores = _compute_component_stability(components_array)

    return {
        "components_list": components_list,
        "stability_scores": stability_scores,
        "n_runs": n_runs,
    }


def _compute_component_stability(components_array: np.ndarray) -> np.ndarray:
    """
    Compute stability scores for ICA components

    Measures how consistently each component appears across runs by
    computing pairwise correlations and finding best matches.

    Parameters
    ----------
    components_array : np.ndarray, shape (n_runs, n_components, n_features)
        Components from multiple ICA runs

    Returns
    -------
    stability_scores : np.ndarray, shape (n_components,)
        Stability score for each component (0-1, higher is more stable)
    """
    n_runs, n_components, n_features = components_array.shape

    # Use first run as reference
    reference = components_array[0]  # (n_components, n_features)

    stability_scores = np.zeros(n_components)

    for comp_idx in range(n_components):
        ref_comp = reference[comp_idx]

        # Compute correlation with all components in all other runs
        correlations = []

        for run_idx in range(1, n_runs):
            run_comps = components_array[run_idx]  # (n_components, n_features)

            # Compute correlation with each component
            corr_matrix = np.corrcoef(ref_comp.reshape(1, -1), run_comps)[0, 1:]

            # Take best match (accounting for sign flip)
            best_corr = np.max(np.abs(corr_matrix))
            correlations.append(best_corr)

        # Stability = mean of best correlations across runs
        stability_scores[comp_idx] = np.mean(correlations)

    return stability_scores


def select_n_components_by_stability(
    X: np.ndarray | torch.Tensor,
    n_components_range: range | list[int],
    pca_components: int | float | str | None = 0.85,
    n_runs: int = 50,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """
    Automatically select number of ICA components based on stability

    Runs ICA stability analysis for different numbers of components and
    selects the number that produces the most stable decomposition.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        fMRI data
    n_components_range : range or list of int
        Range of component numbers to test (e.g., range(10, 50, 5))
    pca_components : int, float, or str, optional
        PCA dimensionality reduction (default: 0.85)
    n_runs : int, default=50
        Number of ICA runs per component number
    min_stability : float, default=0.7
        Minimum acceptable stability score
    device : torch.device, optional
        Computation device
    verbose : bool, default=True
        Show progress

    Returns
    -------
    results : dict
        Dictionary with:
        - 'optimal_n_components': recommended number of components
        - 'stability_by_n_components': dict mapping n_components to mean stability
        - 'all_results': dict of full stability analysis for each n_components

    Examples
    --------
    >>> # Test range of components
    >>> results = select_n_components_by_stability(
    ...     fmri_data,
    ...     n_components_range=range(15, 35, 5),
    ...     n_runs=50
    ... )
    >>> print(f"Optimal: {results['optimal_n_components']} components")
    >>> print(f"Stability: {results['stability_by_n_components']}")
    """
    device = device if device is not None else get_device()

    stability_by_n_components = {}
    all_results = {}

    if verbose:
        print(f"Testing {len(list(n_components_range))} different component numbers...")

    for n_comp in n_components_range:
        if verbose:
            print(f"\nTesting n_components={n_comp}...")

        results = ica_stability_analysis(
            X,
            n_components=n_comp,
            pca_components=pca_components,
            n_runs=n_runs,
            device=device,
            verbose=verbose,
        )

        mean_stability = results["stability_scores"].mean()
        stability_by_n_components[n_comp] = mean_stability
        all_results[n_comp] = results

        if verbose:
            print(f"  Mean stability: {mean_stability:.3f}")
            n_stable = (results["stability_scores"] > min_stability).sum()
            print(f"  Components with stability > {min_stability}: {n_stable}/{n_comp}")

    # Select optimal n_components (highest mean stability)
    optimal_n_components = max(
        stability_by_n_components, key=lambda k: stability_by_n_components[k]
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Optimal n_components: {optimal_n_components}")
        print(f"Stability: {stability_by_n_components[optimal_n_components]:.3f}")

    return {
        "optimal_n_components": optimal_n_components,
        "stability_by_n_components": stability_by_n_components,
        "all_results": all_results,
    }
