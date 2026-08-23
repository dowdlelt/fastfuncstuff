"""Design matrix construction, HRF generation, and HRF selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eager re-exports meant that importing *any* design submodule -- including the
# tiny, dependency-free `design.trim` that cli_utils needs -- executed builder
# and matrices, pulling scipy.stats in for ~0.4 s on every CLI startup. They
# resolve on first attribute access instead (PEP 562), so `from
# fastfuncstuff.design import build_glm_design` still works and costs what it
# always did *when it is used*.
_LAZY: dict[str, str] = {
    "create_onset_matrix_microtime": "fastfuncstuff.design.builder",
    "create_flobs_library": "fastfuncstuff.design.hrf",
    "create_pighs_library": "fastfuncstuff.design.hrf",
    "flobs_halfcos": "fastfuncstuff.design.hrf",
    "get_canonical_hrf": "fastfuncstuff.design.hrf",
    "get_canonical_hrf_library": "fastfuncstuff.design.hrf",
    "get_hrf_library": "fastfuncstuff.design.hrf",
    "pighs_halfcos": "fastfuncstuff.design.hrf",
    "build_event_design_microtime": "fastfuncstuff.design.matrices",
    "build_glm_design": "fastfuncstuff.design.matrices",
    "convolve_hrf": "fastfuncstuff.design.matrices",
    "convolve_hrf_microtime": "fastfuncstuff.design.matrices",
    "generate_random_onsets": "fastfuncstuff.design.matrices",
    "make_fir_design": "fastfuncstuff.design.matrices",
    "make_singletrialdesign": "fastfuncstuff.design.matrices",
}

if TYPE_CHECKING:
    from fastfuncstuff.design.builder import create_onset_matrix_microtime
    from fastfuncstuff.design.hrf import (
        create_flobs_library,
        create_pighs_library,
        flobs_halfcos,
        get_canonical_hrf,
        get_canonical_hrf_library,
        get_hrf_library,
        pighs_halfcos,
    )
    from fastfuncstuff.design.matrices import (
        build_event_design_microtime,
        build_glm_design,
        convolve_hrf,
        convolve_hrf_microtime,
        generate_random_onsets,
        make_fir_design,
        make_singletrialdesign,
    )


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


# hrf_selection has heavy deps (glm.core) -- import lazily to avoid circular imports
# Access via fastfuncstuff.design.hrf_selection directly

__all__ = sorted(_LAZY)
