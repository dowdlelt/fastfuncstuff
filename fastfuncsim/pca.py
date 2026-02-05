"""
GPU-accelerated PCA for fMRI data

Implements fast PCA using PyTorch for GPU acceleration, with flexible
component selection based on variance explained or fixed number of components.

Key Features
------------
- GPU-accelerated SVD via PyTorch
- Flexible n_components specification:
  - float (0-1): Keep components explaining this % of variance (e.g., 0.85 = 85%)
  - int: Keep exactly this many components
  - str: Advanced selection methods (e.g., 'mle', 'knee')
- Memory-efficient chunked processing for large datasets
- Support for both spatial and temporal PCA
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch

from .utils import get_device, to_tensor


class PCA:
    """
    GPU-accelerated PCA for fMRI data

    This implementation uses PyTorch's SVD for fast GPU-accelerated PCA,
    with automatic component selection based on variance explained.

    Parameters
    ----------
    n_components : int, float, or str, optional
        Number of components to keep:
        - float (0.0-1.0): Keep components explaining this fraction of variance
        - int: Keep exactly this many components
        - str: Advanced selection method ('mle', 'knee')
        - None: Keep all components (default)
    whiten : bool, default=False
        If True, components are divided by sqrt(explained_variance)
    device : torch.device, optional
        Device to use for computation. If None, auto-detects GPU.

    Attributes
    ----------
    components_ : torch.Tensor, shape (n_components, n_features)
        Principal components (eigenvectors)
    explained_variance_ : torch.Tensor, shape (n_components,)
        Variance explained by each component
    explained_variance_ratio_ : torch.Tensor, shape (n_components,)
        Fraction of variance explained by each component
    mean_ : torch.Tensor, shape (n_features,)
        Per-feature mean (used for centering)
    n_components_ : int
        Actual number of components kept
    n_samples_ : int
        Number of samples used for fitting
    n_features_ : int
        Number of features in input data

    Examples
    --------
    # Keep components explaining 85% of variance
    >>> pca = PCA(n_components=0.85)
    >>> components = pca.fit_transform(fmri_data)  # (n_timepoints, n_voxels)
    >>> print(f"Kept {pca.n_components_} components")

    # Keep exactly 50 components
    >>> pca = PCA(n_components=50)
    >>> components = pca.fit_transform(fmri_data)

    # Transform new data using fitted PCA
    >>> new_components = pca.transform(new_data)

    # Reconstruct from components
    >>> reconstructed = pca.inverse_transform(components)

    Notes
    -----
    For fMRI data, the input should typically be:
    - Spatial PCA: (n_timepoints, n_voxels) - reduces spatial dimensions
    - Temporal PCA: (n_voxels, n_timepoints) - reduces temporal dimensions

    Data is automatically mean-centered before PCA.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float, str]] = None,
        whiten: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.device = device if device is not None else get_device()

        # Fitted attributes (set by fit())
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> PCA:
        """
        Fit PCA on data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : PCA
            Fitted PCA object
        """
        # Convert to tensor on device
        X = to_tensor(X, device=self.device)

        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Center data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # Compute SVD using memory-efficient covariance approach when n_features >> n_samples
        # This is the GLMdenoise/GLMsingle approach: svd(X @ X^T) instead of svd(X)
        # For fMRI noise pools: (400 timepoints, 260k voxels) → (400, 400) covariance
        # Memory: O(n_samples^2) instead of O(n_samples * n_features)
        #
        # Math: X X^T = U S^2 U^T, so we get same U (PC timecourses) from covariance SVD
        # Then compute loadings V from V = X^T U S^-1 if needed
        if n_features > 10 * n_samples:  # n_features >> n_samples (typical for fMRI)
            # Covariance approach: SVD on (n_samples, n_samples) instead of (n_samples, n_features)
            cov = X_centered @ X_centered.T  # (n_samples, n_samples)
            U, S_squared, _ = torch.linalg.svd(cov, full_matrices=False)
            S = torch.sqrt(torch.clamp(S_squared, min=0))  # Eigenvalues → singular values

            # Explained variance: S^2 / (n_samples - 1)
            explained_variance = (S**2) / (n_samples - 1)
            total_variance = explained_variance.sum()
            explained_variance_ratio = explained_variance / total_variance

            # Determine number of components to keep BEFORE computing loadings
            n_components = self._select_n_components(
                explained_variance_ratio, n_samples, n_features
            )

            # Compute loadings (V^T) ONLY for selected components to save memory
            # V = X^T U S^-1, so Vt = S^-1 * U^T @ X
            S_inv = 1.0 / torch.clamp(S[:n_components], min=1e-10)
            Vt = S_inv[:, None] * (U[:, :n_components].T @ X_centered)  # (n_components, n_features)
        else:
            # Standard approach for moderate-sized data
            # X = U @ S @ V^T
            # Components are rows of V^T (columns of V)
            # Scores are U @ S
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

            # Explained variance: S^2 / (n_samples - 1)
            explained_variance = (S**2) / (n_samples - 1)
            total_variance = explained_variance.sum()
            explained_variance_ratio = explained_variance / total_variance

            # Determine number of components to keep
            n_components = self._select_n_components(
                explained_variance_ratio, n_samples, n_features
            )

            # Keep selected components
            Vt = Vt[:n_components]

        # Store selected components
        self.components_ = Vt
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.n_components_ = n_components

        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transform data to PC space

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_transformed : torch.Tensor, shape (n_samples, n_components)
            Transformed data (PC scores)
        """
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform()")

        # Convert to tensor on device
        X = to_tensor(X, device=self.device)

        # Center and project
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T

        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_)

        return X_transformed

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Fit PCA and transform data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        X_transformed : torch.Tensor, shape (n_samples, n_components)
            Transformed data (PC scores)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transform data back to original space

        Parameters
        ----------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data (PC scores)

        Returns
        -------
        X_reconstructed : torch.Tensor, shape (n_samples, n_features)
            Reconstructed data in original space
        """
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before inverse_transform()")

        # Convert to tensor on device
        X_transformed = to_tensor(X_transformed, device=self.device)

        if self.whiten:
            X_transformed = X_transformed * torch.sqrt(self.explained_variance_)

        # Project back and add mean
        X_reconstructed = X_transformed @ self.components_ + self.mean_

        return X_reconstructed

    def _select_n_components(
        self, explained_variance_ratio: torch.Tensor, n_samples: int, n_features: int
    ) -> int:
        """
        Determine number of components to keep

        Parameters
        ----------
        explained_variance_ratio : torch.Tensor
            Fraction of variance explained by each component
        n_samples : int
            Number of samples
        n_features : int
            Number of features

        Returns
        -------
        n_components : int
            Number of components to keep
        """
        max_components = min(n_samples, n_features)

        if self.n_components is None:
            # Keep all components
            return max_components

        elif isinstance(self.n_components, int):
            # Keep fixed number of components
            if self.n_components > max_components:
                raise ValueError(
                    f"n_components={self.n_components} is too large. "
                    f"Maximum is min(n_samples, n_features) = {max_components}"
                )
            return self.n_components

        elif isinstance(self.n_components, float):
            # Keep components explaining this fraction of variance
            if not 0.0 < self.n_components < 1.0:
                raise ValueError(
                    f"n_components={self.n_components} must be between 0 and 1 "
                    f"when specified as float (fraction of variance)"
                )

            cumsum_variance = torch.cumsum(explained_variance_ratio, dim=0)
            n_components = int(torch.searchsorted(cumsum_variance, self.n_components).item() + 1)

            # Ensure at least 1 component
            return max(1, min(n_components, max_components))

        elif isinstance(self.n_components, str):
            # Advanced selection methods
            if self.n_components == "mle":
                return self._select_mle(explained_variance_ratio, n_samples, n_features)
            elif self.n_components == "knee":
                return self._select_knee(explained_variance_ratio)
            else:
                raise ValueError(
                    f"Unknown n_components string: '{self.n_components}'. Supported: 'mle', 'knee'"
                )

        else:
            raise ValueError(
                f"n_components must be int, float, str, or None. Got {type(self.n_components)}"
            )

    def _select_mle(
        self, explained_variance_ratio: torch.Tensor, n_samples: int, n_features: int
    ) -> int:
        """
        Select number of components using Minka's MLE method

        Based on: Minka, T. P. "Automatic choice of dimensionality for PCA"
        """
        # TODO: Implement Minka's MLE method
        # For now, fall back to keeping 95% variance
        cumsum_variance = torch.cumsum(explained_variance_ratio, dim=0)
        n_components = int(torch.searchsorted(cumsum_variance, 0.95).item() + 1)
        return max(1, min(n_components, min(n_samples, n_features)))

    def _select_knee(self, explained_variance_ratio: torch.Tensor) -> int:
        """
        Select number of components using knee/elbow detection

        Finds the "elbow" in the scree plot where variance explained
        drops off sharply.
        """
        # Simple knee detection: find max second derivative
        variance_curve = explained_variance_ratio.cpu().numpy()

        # Compute second derivative
        first_deriv = np.diff(variance_curve)
        second_deriv = np.diff(first_deriv)

        # Find knee (max curvature)
        knee_idx = np.argmax(np.abs(second_deriv)) + 2  # +2 to account for two diffs

        return max(1, knee_idx)

    def get_explained_variance_cumsum(self) -> torch.Tensor:
        """
        Get cumulative explained variance

        Returns
        -------
        cumsum : torch.Tensor, shape (n_components_,)
            Cumulative sum of explained variance ratios
        """
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("PCA must be fitted first")

        return torch.cumsum(self.explained_variance_ratio_, dim=0)

    def to_dict(self) -> Dict:
        """
        Export PCA parameters to dictionary

        Returns
        -------
        params : dict
            Dictionary with PCA parameters (converted to numpy arrays)
        """
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted first")

        return {
            "components": self.components_.cpu().numpy(),
            "explained_variance": self.explained_variance_.cpu().numpy(),
            "explained_variance_ratio": self.explained_variance_ratio_.cpu().numpy(),
            "mean": self.mean_.cpu().numpy(),
            "n_components": self.n_components_,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "whiten": self.whiten,
        }

    @classmethod
    def from_dict(cls, params: Dict, device: Optional[torch.device] = None) -> PCA:
        """
        Create PCA object from dictionary

        Parameters
        ----------
        params : dict
            Dictionary with PCA parameters (from to_dict())
        device : torch.device, optional
            Device to load tensors to

        Returns
        -------
        pca : PCA
            PCA object with loaded parameters
        """
        device = device if device is not None else get_device()

        pca = cls(n_components=params["n_components"], whiten=params["whiten"], device=device)

        pca.components_ = to_tensor(params["components"], device=device)
        pca.explained_variance_ = to_tensor(params["explained_variance"], device=device)
        pca.explained_variance_ratio_ = to_tensor(params["explained_variance_ratio"], device=device)
        pca.mean_ = to_tensor(params["mean"], device=device)
        pca.n_components_ = params["n_components"]
        pca.n_samples_ = params["n_samples"]
        pca.n_features_ = params["n_features"]

        return pca


def explained_variance_analysis(
    X: Union[np.ndarray, torch.Tensor],
    max_components: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Analyze explained variance across different numbers of components

    Fits PCA with all components and returns variance statistics to help
    determine appropriate n_components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to analyze
    max_components : int, optional
        Maximum components to analyze (default: min(n_samples, n_features))
    device : torch.device, optional
        Device for computation

    Returns
    -------
    analysis : dict
        Dictionary with:
        - 'explained_variance': variance per component
        - 'explained_variance_ratio': fraction of variance per component
        - 'cumulative_variance': cumulative variance explained
        - 'n_components_80': components needed for 80% variance
        - 'n_components_85': components needed for 85% variance
        - 'n_components_90': components needed for 90% variance
        - 'n_components_95': components needed for 95% variance

    Examples
    --------
    >>> analysis = explained_variance_analysis(fmri_data)
    >>> print(f"Components for 85% variance: {analysis['n_components_85']}")
    >>> plt.plot(analysis['cumulative_variance'])
    >>> plt.xlabel('Number of components')
    >>> plt.ylabel('Cumulative variance explained')
    """
    device = device if device is not None else get_device()

    # Fit PCA with all components
    pca = PCA(n_components=None, device=device)
    pca.fit(X)

    # Limit to max_components if specified
    if max_components is not None:
        max_components = min(max_components, pca.n_components_)
    else:
        max_components = pca.n_components_

    # Get variance statistics
    explained_var = pca.explained_variance_[:max_components].cpu().numpy()
    explained_var_ratio = pca.explained_variance_ratio_[:max_components].cpu().numpy()
    cumsum_var = np.cumsum(explained_var_ratio)

    # Find components needed for different variance thresholds
    thresholds = [0.80, 0.85, 0.90, 0.95]
    n_components_for_threshold = {}

    for threshold in thresholds:
        n_comp = int(np.searchsorted(cumsum_var, threshold) + 1)
        n_comp = min(n_comp, max_components)
        key = f"n_components_{int(threshold * 100)}"
        n_components_for_threshold[key] = n_comp

    return {
        "explained_variance": explained_var,
        "explained_variance_ratio": explained_var_ratio,
        "cumulative_variance": cumsum_var,
        **n_components_for_threshold,
    }
