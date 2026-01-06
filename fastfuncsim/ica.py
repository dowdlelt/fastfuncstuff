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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .pca import PCA
from .utils import get_device, to_tensor


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
    pca_ : PCA
        Fitted PCA object used for dimensionality reduction
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
        n_components: Optional[int] = None,
        pca_components: Optional[Union[int, float, str]] = 0.85,
        max_iter: int = 200,
        tol: float = 1e-4,
        fun: str = 'logcosh',
        random_state: Optional[int] = None,
        whiten: bool = True,
        device: Optional[torch.device] = None,
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

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> 'FastICA':
        """
        Fit ICA on data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : FastICA
            Fitted ICA object
        """
        # Convert to tensor
        X = to_tensor(X, device=self.device)

        n_samples, n_features = X.shape

        # Step 1: PCA dimensionality reduction
        self.pca_ = PCA(
            n_components=self.pca_components,
            whiten=self.whiten,
            device=self.device
        )
        X_pca = self.pca_.fit_transform(X)

        # Determine number of ICA components
        if self.n_components is None:
            n_components = self.pca_.n_components_
        else:
            n_components = min(self.n_components, self.pca_.n_components_)

        # Step 2: Run FastICA on TOP PCA spatial components (spatial ICA)
        # PCA components are ordered by variance explained (highest first)
        # Select only the TOP n_components for ICA
        # Example: 100 PCA components → take first 50 for 50 ICA components
        pca_components_all = self.pca_.components_  # (n_pca_components, n_voxels)
        pca_components = pca_components_all[:n_components]  # Take TOP n_components only!

        # For SPATIAL ICA: pass components with voxels as "samples"
        # Our _fastica expects (n_features, n_samples), so pass (n_ica, n_voxels)
        # This makes: n_features=n_ica, n_samples=n_voxels
        # FastICA finds independent rows = independent PC patterns = SPATIAL ICA
        # Following nilearn's CanICA approach
        W, n_iter = self._fastica(pca_components, n_components)  # Input: (n_ica, n_voxels)
        self.n_iter_ = n_iter

        # Step 3: Extract ICA spatial maps
        # W has shape (n_ica, n_ica) - unmixing matrix in TOP PC space
        # Spatial ICA maps: W @ pca_components (top PCs)
        # = (n_ica, n_ica) @ (n_ica, n_voxels) = (n_ica, n_voxels)
        self.components_ = W @ pca_components  # (n_components, n_voxels)

        # Step 4: Compute mixing matrix (ICA timecourses)
        # We have: X_pca with (n_timepoints, n_pca_total)
        # But ICA only used the FIRST n_components PCs
        # So use: X_pca[:, :n_components]
        X_pca_top = X_pca[:, :n_components]  # (n_timepoints, n_ica)
        # Model: X_pca_top ≈ mixing @ W
        # Solving for mixing: mixing = X_pca_top @ pinv(W)
        # Shapes: (n_timepoints, n_ica) @ (n_ica, n_ica) = (n_timepoints, n_ica)
        self.mixing_ = X_pca_top @ torch.linalg.pinv(W)

        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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

    def inverse_transform(self, S: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor, int]:
        """
        Core FastICA algorithm

        Parameters
        ----------
        X : torch.Tensor, shape (n_features, n_samples)
            Whitened data (transposed)
        n_components : int
            Number of components to extract

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

        # Initialize unmixing matrix randomly
        W = torch.randn(n_components, n_features, device=self.device, dtype=X.dtype)

        # Orthogonalize rows
        W = self._symmetric_decorrelation(W)

        # Get nonlinearity functions
        g, g_prime = self._get_nonlinearity(self.fun)

        # FastICA iterations
        for n_iter in range(self.max_iter):
            W_old = W.clone()

            # Update each component
            for i in range(n_components):
                w = W[i, :].unsqueeze(0)  # (1, n_features)

                # Compute projection
                w_X = w @ X  # (1, n_samples)

                # Apply nonlinearity
                g_wx = g(w_X)  # (1, n_samples)
                g_prime_wx = g_prime(w_X)  # (1, n_samples)

                # Update rule
                w_new = (X * g_wx).mean(dim=1, keepdim=True).T - g_prime_wx.mean() * w

                # Decorrelate with previous components
                w_new = w_new - (W[:i] @ w_new.T).T @ W[:i]

                # Normalize
                w_new = w_new / torch.norm(w_new)

                W[i, :] = w_new.squeeze()

            # Check convergence
            delta = torch.max(torch.abs(torch.abs(torch.diag(W @ W_old.T)) - 1))

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
        if fun == 'logcosh':
            alpha = 1.0

            def g(x):
                return torch.tanh(alpha * x)

            def g_prime(x):
                return alpha * (1 - torch.tanh(alpha * x) ** 2)

        elif fun == 'exp':
            def g(x):
                exp_x = torch.exp(-x ** 2 / 2)
                return x * exp_x

            def g_prime(x):
                exp_x = torch.exp(-x ** 2 / 2)
                return (1 - x ** 2) * exp_x

        elif fun == 'cube':
            def g(x):
                return x ** 3

            def g_prime(x):
                return 3 * x ** 2

        else:
            raise ValueError(f"Unknown nonlinearity: '{fun}'. Use 'logcosh', 'exp', or 'cube'")

        return g, g_prime

    def compute_variance_explained(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        total_var = (X_centered ** 2).sum()

        # Reconstruct each component separately
        variance_explained = torch.zeros(self.components_.shape[0], device=self.device)

        for i in range(self.components_.shape[0]):
            # Reconstruct using only this component
            reconstruction = self.mixing_[:, i:i+1] @ self.components_[i:i+1, :]

            # Variance explained = variance of reconstruction
            var_i = (reconstruction ** 2).sum()
            variance_explained[i] = var_i

        # Compute ratios
        variance_ratio = variance_explained / total_var

        return variance_explained, variance_ratio

    def to_dict(self) -> Dict:
        """Export ICA parameters to dictionary"""
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted first")

        return {
            'components': self.components_.cpu().numpy(),
            'mixing': self.mixing_.cpu().numpy(),
            'pca': self.pca_.to_dict(),
            'n_components': self.n_components,
            'n_iter': self.n_iter_,
        }


def ica_stability_analysis(
    X: Union[np.ndarray, torch.Tensor],
    n_components: int,
    pca_components: Optional[Union[int, float, str]] = 0.85,
    n_runs: int = 100,
    n_jobs: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
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

    # Compute mean and std components
    mean_components = components_array.mean(axis=0)
    std_components = components_array.std(axis=0)

    return {
        'components_list': components_list,
        'stability_scores': stability_scores,
        'mean_components': mean_components,
        'std_components': std_components,
        'n_runs': n_runs,
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
    X: Union[np.ndarray, torch.Tensor],
    n_components_range: Union[range, List[int]],
    pca_components: Optional[Union[int, float, str]] = 0.85,
    n_runs: int = 50,
    min_stability: float = 0.7,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
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

        mean_stability = results['stability_scores'].mean()
        stability_by_n_components[n_comp] = mean_stability
        all_results[n_comp] = results

        if verbose:
            print(f"  Mean stability: {mean_stability:.3f}")
            n_stable = (results['stability_scores'] > min_stability).sum()
            print(f"  Components with stability > {min_stability}: {n_stable}/{n_comp}")

    # Select optimal n_components (highest mean stability)
    optimal_n_components = max(stability_by_n_components, key=stability_by_n_components.get)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimal n_components: {optimal_n_components}")
        print(f"Stability: {stability_by_n_components[optimal_n_components]:.3f}")

    return {
        'optimal_n_components': optimal_n_components,
        'stability_by_n_components': stability_by_n_components,
        'all_results': all_results,
    }
