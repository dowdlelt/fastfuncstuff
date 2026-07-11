"""qwarp_torch: GPU-accelerated nonlinear image warping.

A PyTorch reimplementation of AFNI's 3dQwarp, providing piecewise polynomial
nonlinear registration using Hermite basis functions over overlapping patches
with multi-level refinement and GPU-parallel batch optimization.
"""

__version__ = "0.2.0"

from .allineate import AffineAlignConfig, allineate
from .apply import apply_warp, compose_warps, invert_warp
from .basis import HermiteCubic, HermiteQuintic
from .cost import clipped_pearson_correlation, lpa_correlation, lpc_correlation, pearson_correlation
from .io import (
    load_image,
    load_warp_field,
    load_warp_series,
    save_image,
    save_warp_field,
    save_warp_series,
)
from .mask import automask
from .memory import estimate_gpu_memory_gb, print_memory_report
from .nwarpforge import (
    AffineTransform,
    NonlinearWarp,
    compose_chain,
    compose_matrix_then_warp,
    compose_warp_then_matrix,
    compose_warp_then_warp,
    compute_cardinal_affine,
    load_affine_1D,
    load_warp,
    nwarpforge,
    prepare_warp_for_grid,
)
from .warp import QwarpConfig, qwarp
