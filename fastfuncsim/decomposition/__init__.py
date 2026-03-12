"""Dimensionality reduction: PCA, FastICA, ICASSO stability analysis."""

from fastfuncsim.decomposition.ica import FastICA
from fastfuncsim.decomposition.icasso import icasso, icasso_auto_select
from fastfuncsim.decomposition.pca import PCA

__all__ = [
    "PCA",
    "FastICA",
    "icasso",
    "icasso_auto_select",
]
