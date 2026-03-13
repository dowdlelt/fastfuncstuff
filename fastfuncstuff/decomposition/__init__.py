"""Dimensionality reduction: PCA, FastICA, ICASSO stability analysis."""

from fastfuncstuff.decomposition.ica import FastICA
from fastfuncstuff.decomposition.icasso import icasso, icasso_auto_select
from fastfuncstuff.decomposition.pca import PCA

__all__ = [
    "PCA",
    "FastICA",
    "icasso",
    "icasso_auto_select",
]
