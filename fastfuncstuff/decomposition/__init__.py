"""Dimensionality reduction: PCA, FastICA, InfoMax ICA, ICASSO stability analysis."""

from fastfuncstuff.decomposition.ica import FastICA, InfoMaxICA, create_ica
from fastfuncstuff.decomposition.icasso import icasso, icasso_auto_select, icasso_plot
from fastfuncstuff.decomposition.pca import PCA

__all__ = [
    "PCA",
    "FastICA",
    "InfoMaxICA",
    "create_ica",
    "icasso",
    "icasso_auto_select",
    "icasso_plot",
]
