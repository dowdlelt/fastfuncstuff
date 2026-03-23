"""Statistical utilities for neuroimaging analysis."""

from .spatial import (
    consistency_report,
    optimal_matching,
    spatial_correlation,
    spatial_correlation_matrix,
)

__all__ = [
    "consistency_report",
    "optimal_matching",
    "spatial_correlation",
    "spatial_correlation_matrix",
]
