"""Benchmark stages registry."""

from __future__ import annotations

from . import align, glm, ica, moco, slicetime, warp

ALL_STAGES = [moco, slicetime, align, warp, glm, ica]

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
