"""Design matrix construction, HRF generation, and HRF selection."""

from fastfuncstuff.design.builder import (
    create_onset_matrix_microtime,
)
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
    build_glm_design,
    convolve_hrf,
    convolve_hrf_microtime,
    generate_random_onsets,
    make_fir_design,
    make_singletrialdesign,
)

# hrf_selection has heavy deps (glm.core) -- import lazily to avoid circular imports
# Access via fastfuncstuff.design.hrf_selection directly

__all__ = [
    "build_glm_design",
    "convolve_hrf",
    "generate_random_onsets",
    "make_fir_design",
    "make_singletrialdesign",
    "get_canonical_hrf",
    "get_canonical_hrf_library",
    "get_hrf_library",
    "pighs_halfcos",
    "create_pighs_library",
    "flobs_halfcos",
    "create_flobs_library",
    "convolve_hrf_microtime",
    "create_onset_matrix_microtime",
]
