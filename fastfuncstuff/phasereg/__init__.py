"""Phase regression for macrovascular BOLD suppression.

Implements voxel-wise regression of magnitude on phase to remove
signal from oriented macrovasculature (pial veins, cerebral veins),
preserving microvascular BOLD.

References
----------
Menon RS (2002). Postacquisition suppression of large-vessel BOLD signals
    in high-resolution fMRI. Magn Reson Med 47:1-9.
Stanley OW et al (2021). Effects of phase regression on high-resolution
    functional MRI of the primary visual cortex. NeuroImage 117631.
Knudsen L et al (2023). Improved sensitivity and microvascular weighting
    of 3T laminar fMRI with GE-BOLD using NORDIC and phase regression.
    NeuroImage 271:120011.
Vu AT, Gallant JL (2015). Using a novel source-localized phase regressor
    technique for evaluation of the vascular contribution to semantic
    category area localization in BOLD fMRI. Front Neurosci 9:411.
"""

from fastfuncstuff.phasereg.core import PhaseRegResult, phase_regress
from fastfuncstuff.phasereg.deming import deming_regression, ols_regression
from fastfuncstuff.phasereg.noise import estimate_variance_ratio
from fastfuncstuff.phasereg.spr import build_neighbor_index, select_phase_donor

__all__ = [
    "PhaseRegResult",
    "phase_regress",
    "deming_regression",
    "ols_regression",
    "estimate_variance_ratio",
    "build_neighbor_index",
    "select_phase_donor",
]
