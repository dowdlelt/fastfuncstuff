"""Group frames by their predicted BOLD state, for a condition-paired reference.

A brightness-constancy motion estimator cannot tell "this boundary got brighter" from
"it moved", so registering every frame to one template hands it the full task response
as apparent displacement.  Registering each frame to a template built from *frames in
the same task state* makes the response common-mode within the pair, where it cancels
exactly — no HRF fit, no projection, no assumption that the contamination is linear.

The binning is on the **convolved design**, not on the condition labels, so the sloped
parts of the response get their own bins.  That matters more than it sounds: at TR 2.5 s
with 50 s blocks a single TR at a block transition carries up to 41% of the full
ON-vs-OFF swing, and a two-state ON/OFF split puts those frames in whichever bin their
label says rather than the one their signal is actually in.

The trade is bins against repeats.  Every extra bin sharpens the state match and thins
the average that suppresses the residual field, so ``min_frames`` merges bins that got
too few frames rather than letting a two-frame template through.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["design_state_bins", "format_bin_report"]


def _unit_scale(design: torch.Tensor) -> torch.Tensor:
    """Each column mapped to [0, 1] by its own peak-to-peak.

    Per column, so ``bin_width`` means "this fraction of THIS condition's swing" and a
    weak condition is not binned into a single bin by a strong one's range.
    """
    lo = design.min(dim=0, keepdim=True).values
    span = (design.max(dim=0, keepdim=True).values - lo).clamp(min=1e-12)
    return (design - lo) / span


def design_state_bins(
    design: torch.Tensor | np.ndarray,
    *,
    bin_width: float = 0.2,
    n_bins: int | None = None,
    min_frames: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Assign each frame to a bin of predicted BOLD state.

    Parameters
    ----------
    design : (T, K)
        The HRF-convolved design.  Multi-condition designs are quantized per column and
        grouped on the resulting tuple, which is exact for one condition and sensible
        for a simple factorial one; it is not intended for a rapid event-related design
        where nearly every frame lands in its own state.
    bin_width : float
        Bin size as a fraction of each condition's peak-to-peak swing.  0.2 gives five
        levels: baseline, three slope steps, peak.  Ignored when ``n_bins`` is given.
    n_bins : int, optional
        Explicit level count per condition; overrides ``bin_width``.
    min_frames : int
        Bins with fewer frames than this are merged into the nearest occupied bin in
        state space.  A template averaged over two frames suppresses the residual field
        by 1/sqrt(2) and would hand its own noise to every frame registered against it.

    Returns
    -------
    bin_of : (T,) int64
        Consecutive bin id per frame.
    info : dict
        ``n_bins``, ``counts``, ``states`` (mean state vector per bin), ``n_merged``.
    """
    x = torch.as_tensor(np.asarray(design), dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_t = x.shape[0]
    levels = int(n_bins) if n_bins else max(1, int(round(1.0 / max(bin_width, 1e-6))))
    if levels < 2:
        raise ValueError(
            f"binning needs at least 2 levels, got {levels} "
            f"(bin_width={bin_width}, n_bins={n_bins})"
        )

    z = _unit_scale(x)
    # The top of the range must land in the last bin rather than one past it.
    idx = (z * levels).floor().clamp(0, levels - 1).to(torch.int64)

    keys = [tuple(int(v) for v in row) for row in idx]
    order: dict[tuple, int] = {}
    for k in keys:
        order.setdefault(k, len(order))
    bin_of = torch.tensor([order[k] for k in keys], dtype=torch.int64)

    # Mean state per bin, used both as the report and as the merge metric.
    def _states(b: torch.Tensor, n: int) -> torch.Tensor:
        return torch.stack([z[b == i].mean(dim=0) for i in range(n)])

    n_merged = 0
    n = len(order)
    while n > 1:
        counts = torch.bincount(bin_of, minlength=n)
        small = int(counts.argmin())
        if int(counts[small]) >= min_frames:
            break
        states = _states(bin_of, n)
        d = (states - states[small]).pow(2).sum(dim=1)
        d[small] = float("inf")
        target = int(d.argmin())
        bin_of[bin_of == small] = target
        # Re-pack ids so they stay consecutive.
        uniq = sorted(set(int(v) for v in bin_of))
        remap = {old: i for i, old in enumerate(uniq)}
        bin_of = torch.tensor([remap[int(v)] for v in bin_of], dtype=torch.int64)
        n = len(uniq)
        n_merged += 1

    counts = torch.bincount(bin_of, minlength=n)
    return bin_of, {
        "n_bins": n,
        "counts": counts.tolist(),
        "states": _states(bin_of, n).tolist(),
        "n_merged": n_merged,
        "n_timepoints": n_t,
        "levels_requested": levels,
    }


def format_bin_report(info: dict, labels: list[str] | None = None) -> str:
    """One line per bin: how many frames, and what state they share."""
    lines = [
        f"  paired reference: {info['n_bins']} bins from {info['n_timepoints']} frames "
        f"({info['levels_requested']} levels requested"
        + (f", {info['n_merged']} merged for sparsity" if info["n_merged"] else "")
        + ")"
    ]
    for i, (c, st) in enumerate(zip(info["counts"], info["states"], strict=True)):
        state = ", ".join(
            f"{(labels[k] if labels and k < len(labels) else f'c{k + 1}')}={v:.2f}"
            for k, v in enumerate(st)
        )
        lines.append(f"    bin {i}: {c:>4} frames   state {state}")
    return "\n".join(lines)
