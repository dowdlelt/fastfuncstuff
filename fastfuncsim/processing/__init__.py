"""qwarp_torch: GPU-accelerated nonlinear image warping.

A PyTorch reimplementation of AFNI's 3dQwarp, providing piecewise polynomial
nonlinear registration using Hermite basis functions over overlapping patches
with multi-level refinement and GPU-parallel batch optimization.
"""

__version__ = "0.2.0"

from .warp import qwarp, QwarpConfig
from .basis import HermiteCubic, HermiteQuintic
from .cost import pearson_correlation, clipped_pearson_correlation, lpa_correlation
from .apply import apply_warp, compose_warps, invert_warp
from .io import load_image, save_image, save_warp_field, load_warp_field
from .memory import estimate_gpu_memory_gb, print_memory_report
from .allineate import allineate, AffineAlignConfig
from .cost import lpc_correlation
from .mask import automask
from .nwarpforge import (
    nwarpforge,
    load_affine_1D,
    load_warp,
    compose_chain,
    compose_warp_then_matrix,
    compose_matrix_then_warp,
    compose_warp_then_warp,
    AffineTransform,
    NonlinearWarp,
)
