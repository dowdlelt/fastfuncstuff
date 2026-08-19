"""qwarp_torch: GPU-accelerated nonlinear image warping.

A PyTorch reimplementation of AFNI's 3dQwarp, providing piecewise polynomial
nonlinear registration using Hermite basis functions over overlapping patches
with multi-level refinement and GPU-parallel batch optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0"

# Eager re-exports meant that touching any single submodule -- e.g. a CLI that
# only wants `processing.io.load_image` -- also executed allineate (scipy.optimize)
# and warp, adding ~0.3 s to every startup. They resolve on first attribute access
# instead (PEP 562), so `from fastfuncstuff.processing import qwarp` still works
# and costs what it always did *when it is used*.
_LAZY: dict[str, str] = {
    "AffineAlignConfig": ".allineate",
    "allineate": ".allineate",
    "apply_warp": ".apply",
    "compose_warps": ".apply",
    "invert_warp": ".apply",
    "HermiteCubic": ".basis",
    "HermiteQuintic": ".basis",
    "clipped_pearson_correlation": ".cost",
    "lpa_correlation": ".cost",
    "lpc_correlation": ".cost",
    "pearson_correlation": ".cost",
    "load_image": ".io",
    "load_warp_field": ".io",
    "load_warp_series": ".io",
    "save_image": ".io",
    "save_warp_field": ".io",
    "save_warp_series": ".io",
    "automask": ".mask",
    "estimate_gpu_memory_gb": ".memory",
    "print_memory_report": ".memory",
    "AffineTransform": ".nwarpforge",
    "NonlinearWarp": ".nwarpforge",
    "compose_chain": ".nwarpforge",
    "compose_matrix_then_warp": ".nwarpforge",
    "compose_warp_then_matrix": ".nwarpforge",
    "compose_warp_then_warp": ".nwarpforge",
    "compute_cardinal_affine": ".nwarpforge",
    "load_affine_1D": ".nwarpforge",
    "load_warp": ".nwarpforge",
    "nwarpforge": ".nwarpforge",
    "prepare_warp_for_grid": ".nwarpforge",
    "QwarpConfig": ".warp",
    "qwarp": ".warp",
}

if TYPE_CHECKING:
    from .allineate import AffineAlignConfig, allineate
    from .apply import apply_warp, compose_warps, invert_warp
    from .basis import HermiteCubic, HermiteQuintic
    from .cost import (
        clipped_pearson_correlation,
        lpa_correlation,
        lpc_correlation,
        pearson_correlation,
    )
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


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)
