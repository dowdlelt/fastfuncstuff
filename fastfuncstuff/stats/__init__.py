"""Statistical utilities for neuroimaging analysis."""

from .fdr import (
    add_fdrcurves_to_nifti,
    compute_fdr_curve,
    fdr_qvalues,
    stat_to_pvalue,
)
from .spatial import (
    consistency_report,
    optimal_matching,
    spatial_correlation,
    spatial_correlation_matrix,
)

__all__ = [
    "add_fdrcurves_to_nifti",
    "compute_fdr_curve",
    "consistency_report",
    "fdr_qvalues",
    "optimal_matching",
    "spatial_correlation",
    "spatial_correlation_matrix",
    "stat_to_pvalue",
]
