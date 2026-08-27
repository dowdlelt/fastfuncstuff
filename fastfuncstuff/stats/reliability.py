"""Split-half reliability and noise-ceiling primitives.

Held-out R2 answers "does the model predict unseen data". It does not answer
"is this parameter estimate stable", and for some parameters the two questions
have different answers. Lage-Castellanos, Valente, Senden & De Martino (2020),
'Investigating the reliability of population receptive field size estimates
using fMRI', Front Neurosci 14:825, measured five pRF estimation methods whose
prediction accuracy was identical to two decimal places (0.36) while their
split-half reliability of pRF *size* ranged from 0.39 to 0.81. Prediction
accuracy simply cannot see the difference, which is why these primitives exist
separately from the cross-validation ones.

Two quantities live here:

**Reliability** is the agreement between parameters estimated from two disjoint
halves of the data. It needs no repeated stimuli -- only two fits that share no
timepoints -- so it is available for any design.

**Noise ceiling** is the fraction of a voxel's variance that is reproducible at
all, estimated from repeats of an *identical* design. It upper-bounds any
model's R2, which is what makes a low R2 interpretable: 0.3 against a ceiling of
0.35 is a good fit in bad data, and 0.3 against 0.8 is a bad model. Unlike
reliability it is not always computable, so callers must treat it as optional.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Rank ``values`` ascending, averaging the ranks within each tie group.

    Ordinal ranks (``argsort(argsort(x))``) are the cheap spelling and are wrong
    here: pRF parameters pile up on their bounds -- sigma at the lower clamp,
    exponent at ``exptlowerbound`` -- so ties are common rather than incidental,
    and breaking them by array order manufactures agreement or disagreement out
    of nothing but voxel indexing.
    """
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = torch.argsort(values)
    _, inverse, counts = torch.unique(
        values[order], return_inverse=True, return_counts=True, sorted=True
    )
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    # One-based ranks start+1 .. end average to (start + end + 1) / 2.
    group_rank = (starts + ends + 1).to(values.dtype) / 2.0
    ranks = torch.empty_like(values)
    ranks[order] = group_rank[inverse]
    return ranks


def pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Plain Pearson correlation between two one-dimensional tensors."""
    if first.shape != second.shape:
        raise ValueError("inputs must have the same shape")
    if first.numel() < 2:
        return float("nan")
    a = first.to(torch.float64) - first.to(torch.float64).mean()
    b = second.to(torch.float64) - second.to(torch.float64).mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator <= 0:
        return float("nan")
    return float((a @ b) / denominator)


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Rank correlation, used for the parameters with non-normal distributions.

    Eccentricity and pRF size are strongly skewed across a visual-cortex ROI, so
    a Pearson correlation of them is dominated by the peripheral tail. The
    reference analysis uses Spearman for exactly this reason.
    """
    if first.numel() < 2:
        return float("nan")
    return pearson_correlation(average_ranks(first), average_ranks(second))


def circular_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Jammalamadaka-Sarma circular correlation for angles **in radians**.

    Polar angle wraps, so a linear correlation reports near-zero agreement for
    two estimates that differ only by which side of 0/2*pi they landed on. This
    is the coefficient the reference analysis uses (Mardia 1975).
    """
    if first.shape != second.shape:
        raise ValueError("inputs must have the same shape")
    if first.numel() < 2:
        return float("nan")
    a = first.to(torch.float64)
    b = second.to(torch.float64)
    mean_a = torch.atan2(a.sin().mean(), a.cos().mean())
    mean_b = torch.atan2(b.sin().mean(), b.cos().mean())
    sin_a = (a - mean_a).sin()
    sin_b = (b - mean_b).sin()
    denominator = torch.sqrt(sin_a.square().sum() * sin_b.square().sum())
    if denominator <= 0:
        return float("nan")
    return float((sin_a * sin_b).sum() / denominator)


def spearman_brown(reliability: float, length_ratio: float = 2.0) -> float:
    """Project a half-length reliability up to the full-length one.

    Split-half reliability is measured between two fits that each saw half the
    data, so it understates the reliability of the fit actually reported, which
    saw all of it. This is the standard correction; ``length_ratio`` is how many
    times longer the reported fit is than one half.
    """
    if reliability != reliability:  # NaN
        return reliability
    denominator = 1.0 + (length_ratio - 1.0) * reliability
    if denominator == 0:
        return float("nan")
    return length_ratio * reliability / denominator


def split_half_noise_ceiling(
    data: torch.Tensor,
    repeat_groups: Sequence[Sequence[int]],
    run_starts: Sequence[int],
    n_timepoints: int,
) -> torch.Tensor:
    """Per-voxel upper bound on R2, from repeats of an identical design.

    ``repeat_groups`` lists sets of run indices whose stimulus is bit-identical,
    so their expected responses are equal and any disagreement is noise. For
    ``y = s + e`` with independent noise across repeats, the correlation between
    two repeats is ``var(s) / var(y)`` -- which is exactly the largest R2 any
    model can reach on a single run. So the mean pairwise correlation *is* the
    ceiling, in the same units as the reported R2, with no further correction.

    **Pass nuisance-projected data.** Shared drift is reproducible across
    repeats and would be counted as signal, inflating the ceiling toward 1 for
    every voxel with a slow trend in it.

    Groups whose runs differ in length are skipped: unequal lengths mean the
    designs were not in fact identical, whatever the caller believed. Voxels
    with no usable group come back NaN rather than 0, so "no ceiling here" is
    distinguishable from "nothing is reproducible here".
    """
    if data.ndim != 2:
        raise ValueError("data must have shape (n_voxels, n_timepoints)")
    run_ends = [*run_starts[1:], n_timepoints]
    totals = torch.zeros(data.shape[0], device=data.device, dtype=torch.float64)
    counts = 0

    for group in repeat_groups:
        segments = [data[:, run_starts[index] : run_ends[index]] for index in group]
        lengths = {segment.shape[1] for segment in segments}
        if len(group) < 2 or len(lengths) != 1:
            continue
        centered = [
            (segment.to(torch.float64) - segment.to(torch.float64).mean(dim=1, keepdim=True))
            for segment in segments
        ]
        norms = [torch.linalg.vector_norm(segment, dim=1) for segment in centered]
        for first in range(len(centered)):
            for second in range(first + 1, len(centered)):
                denominator = norms[first] * norms[second]
                correlation = torch.where(
                    denominator > 0,
                    (centered[first] * centered[second]).sum(dim=1) / denominator.clamp_min(1e-30),
                    torch.zeros_like(denominator),
                )
                totals += correlation
                counts += 1

    if counts == 0:
        return torch.full((data.shape[0],), torch.nan, device=data.device, dtype=data.dtype)
    # Negative mean correlation means nothing reproduces; clamping at 0 keeps the
    # ceiling usable as a divisor without inventing signal.
    return (totals / counts).clamp_min(0.0).to(data.dtype)


def identical_design_labels(stimulus_runs: Sequence[torch.Tensor], atol: float = 0.0) -> list[int]:
    """Label each run so that runs with the same expected response share a label.

    Detected rather than declared. A user-supplied stimulus grouping says "these
    runs probe the same thing", which is the right unit for cross-validation but
    is weaker than what a noise ceiling needs: clockwise and counter-clockwise
    wedges belong to one cross-validation group and have completely different
    expected timecourses. Comparing the runs directly answers the stricter
    question and cannot disagree with the data.

    Takes aperture movies (pRF) or per-run design-matrix blocks (GLM) -- both are
    "the thing whose equality makes two runs repeats of each other".

    ``atol`` is 0 by default, i.e. bit-identical. Raise it for designs that were
    *convolved* rather than replayed: identical event timing can still produce
    last-bit differences, and a tolerance of ~1e-6 accepts those without
    accepting a genuinely different design (whose columns differ by order 1).
    """
    labels: list[int] = []
    representatives: list[torch.Tensor] = []
    for run in stimulus_runs:
        for slot, representative in enumerate(representatives):
            if representative.shape != run.shape:
                continue
            same = (
                torch.equal(representative, run)
                if atol <= 0
                else bool(torch.allclose(representative, run, rtol=0.0, atol=atol))
            )
            if same:
                labels.append(slot)
                break
        else:
            representatives.append(run)
            labels.append(len(representatives) - 1)
    return labels


def identical_design_groups(
    stimulus_runs: Sequence[torch.Tensor], atol: float = 0.0
) -> list[list[int]]:
    """Run-index sets whose expected response is the same, singletons dropped."""
    labels = identical_design_labels(stimulus_runs, atol=atol)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return [members for members in groups.values() if len(members) > 1]
