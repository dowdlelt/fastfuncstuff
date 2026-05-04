"""Benchmark stages registry."""

from __future__ import annotations

from . import (
    align,
    automask,
    build_design,
    crossalign,
    glm,
    glm_im,
    glm_im_reml,
    glm_tent,
    glmsingle_denoise,
    glmsingle_hrf,
    glmsingle_matlab,
    glmsingle_prep,
    glmsingle_ridge,
    ica,
    ica_single,
    ica_solver,
    ica_trace,
    moco,
    nordic,
    sauna,
    slicetime,
    warp,
)

ALL_STAGES = [
    moco, slicetime, crossalign, align, warp, glm, build_design, ica, ica_single,
    ica_trace, ica_solver,
    automask, nordic, sauna,
    glmsingle_prep, glmsingle_matlab, glmsingle_hrf, glmsingle_denoise, glmsingle_ridge,
    glm_tent, glm_im, glm_im_reml,
]

STAGE_MAP = {s.name: s for s in ALL_STAGES}


def get_stages(names: list[str] | None = None) -> list:
    """Get stage modules by name. None = all stages."""
    if names is None:
        return list(ALL_STAGES)
    result = []
    for n in names:
        if n not in STAGE_MAP:
            raise ValueError(f"Unknown stage: {n!r}. Available: {list(STAGE_MAP)}")
        result.append(STAGE_MAP[n])
    return result
