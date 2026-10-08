"""One statistics mask for a hemisphere from every run's projection.

The vertex twin of autoproc's stage10b volume masks. Each run's projection already
writes ``mask`` = cortex label AND that run's footprints fully inside the EPI in every
frame; across runs that is an intersection (a vertex a single run lost has a hole in
its series). The automask half -- dark where the signal dropped out -- is AFNI's clip
level (``THD_cliplevel``, what 3dAutomask thresholds at) applied to the mean of the
runs' vertex means, over the vertices still in. Cortex has no background, so the
level sits at ``clfrac`` of the median of the brighter cortex: it drops dropout (OFC,
temporal poles near the sinuses), not ordinary intensity bias.
"""

from __future__ import annotations

import numpy as np

from fastfuncstuff.processing.mask import _afni_cliplevel

__all__ = ["combine_run_masks"]


def combine_run_masks(
    masks: list[np.ndarray], means: list[np.ndarray] | None = None, clfrac: float = 0.5
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """``(mask, meanall, clip)``: runs' masks intersected, then clipped on the mean.

    ``clfrac`` 0 skips the clip (coverage AND cortex only); ``clip`` is the level used.
    """
    stack = np.stack([np.asarray(m).reshape(-1) > 0 for m in masks])
    mask: np.ndarray = np.asarray(stack.all(axis=0))
    if not means:
        return mask, None, 0.0
    if len(means) != len(masks):
        raise ValueError(f"{len(means)} means for {len(masks)} masks")
    meanall = np.mean([np.asarray(m, np.float32).reshape(-1) for m in means], axis=0)
    clip = 0.0
    if clfrac > 0 and mask.any():
        clip = _afni_cliplevel(np.where(mask, meanall, 0.0).astype(np.float32), clfrac)
        mask &= meanall >= clip
    return mask, meanall.astype(np.float32), clip
