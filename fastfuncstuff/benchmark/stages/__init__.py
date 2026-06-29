"""Benchmark stages registry."""

from __future__ import annotations

from . import (
    align,
    automask,
    build_design,
    crossalign,
    glm,
    glm_fdr,
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
    ica_single_trace,
    ica_solver,
    ica_trace,
    moco,
    nordic,
    phasereg,
    sauna,
    slicetime,
    warp,
)

ALL_STAGES = [
    moco, slicetime, crossalign, align, warp, glm, build_design, ica, ica_single,
    ica_single_trace, ica_trace, ica_solver,
    automask, nordic, sauna, phasereg,
    glmsingle_prep, glmsingle_matlab, glmsingle_hrf, glmsingle_denoise, glmsingle_ridge,
    glm_tent, glm_fdr, glm_im, glm_im_reml,
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


def expand_with_deps(names: list[str]) -> list[str]:
    """Expand a stage-name list to include upstream ``requires``, in run order.

    Walks each stage's ``requires`` transitively and returns the union ordered
    by ``ALL_STAGES`` (so upstream stages run first). Used by ``--with-deps`` so
    ``-stages warp`` can pull in moco/crossalign/align automatically instead of
    failing INCOMPLETE on their missing outputs.
    """
    wanted: set[str] = set()

    def visit(n: str) -> None:
        if n in wanted or n not in STAGE_MAP:
            return
        wanted.add(n)
        for dep in getattr(STAGE_MAP[n], "requires", []):
            visit(dep)

    for n in names:
        visit(n)
    return [s.name for s in ALL_STAGES if s.name in wanted]


def unsatisfied_deps(names: list[str]) -> dict[str, list[str]]:
    """Map each requested stage to its ``requires`` that are NOT in the request.

    Lets the runner warn up front that a subset run depends on stages it isn't
    running (whose outputs may be missing) before doing any expensive work.
    """
    requested = set(names)
    out: dict[str, list[str]] = {}
    for n in names:
        stage = STAGE_MAP.get(n)
        if stage is None:
            continue
        missing = [d for d in getattr(stage, "requires", []) if d not in requested]
        if missing:
            out[n] = missing
    return out
