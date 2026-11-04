"""Minimum-statistic conjunctions (Nichols, Brett, Andersson, Wager & Poline 2005).

A conjunction "A AND B AND ..." is significant at a voxel when *every* component
is: its t is the voxelwise minimum of the component t-stats, thresholded with the
ordinary t null. That tests the conjunction null (not all effects are real), which
is the valid one; the global-null reading of the same map is not what it means.

Components are conditions or design columns (fit as the one-row GLT ``+1*name``,
i.e. exactly what a ``SYM: +name`` contrast gives, spread over a multi-column
basis the way ``-gltsym`` does) or labels of contrasts the fit already computes.
They therefore ride the GLT machinery on every route (OLS, ARMA/REML, per-voxel
HRF, -dsort snapshot) and share the GLM's DoF, which the bucket header carries.

The coefficient written beside the t is the voxelwise minimum of the component
*coefficients*, not the coefficient of the argmin-t component: a size that every
component reaches is the honest "how big is the shared effect" number, though the
two minima can come from different components.

Only the positive conjunction is tested. For "less than zero in all" negate each
component (``-A``): the minimum of the negated statistics is the negated maximum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# Internal component GLTs are computed in the fit and dropped before anything is
# written; the prefix only has to never collide with a user contrast label.
_INTERNAL_PREFIX = "__conj__"


@dataclass(frozen=True)
class ConjunctionPlan:
    """Resolved conjunctions over a GLT list.

    ``n_user`` leading GLTs are the user's contrasts and are kept; the rest are
    internal component GLTs. Each conjunction is ``(label, ((glt_index, sign), ...))``.
    """

    n_user: int
    conjunctions: tuple[tuple[str, tuple[tuple[int, float], ...]], ...]


def split_components(components: str | Sequence[str]) -> list[str]:
    """``'A B,-C'`` or ``['A', '-C']`` → ``['A', 'B', '-C']``."""
    if isinstance(components, str):
        components = [components]
    out: list[str] = []
    for c in components:
        out.extend(tok for tok in c.replace(",", " ").split() if tok)
    return out


def plan_conjunctions(
    conjunctions: Sequence[tuple[str, Sequence[str]]],
    design_info: dict,
) -> ConjunctionPlan:
    """Resolve conjunctions against a design and append their component GLTs.

    ``design_info`` is the dict ``read_afni_design_matrix`` returns (or the one
    ffs_reml builds from events): ``glt_labels`` / ``glt_matrices`` are extended
    in place with one internal t-test GLT per condition component, and the plan
    is stored as ``design_info["glt_conjunctions"]``. Raises ``ValueError``
    listing every problem at once.
    """
    from fastfuncstuff.design.builder import glt_rows_to_matrix

    glt_labels = list(design_info.get("glt_labels") or [])
    glt_matrices = list(design_info.get("glt_matrices") or [])
    column_labels = list(design_info.get("column_labels") or [])
    n_columns = int(np.asarray(design_info["matrix"]).shape[1])
    if len(column_labels) != n_columns:
        column_labels = [""] * n_columns
    stim_ranges = {
        str(label): (int(bot), int(top))
        for label, bot, top in zip(
            design_info.get("stim_labels") or [],
            design_info.get("stim_bots") or [],
            design_info.get("stim_tops") or [],
            strict=False,
        )
    }
    n_user = len(glt_labels)
    user_index = {label: i for i, label in enumerate(glt_labels)}
    internal_index: dict[str, int] = {}
    internal_labels: list[str] = []
    internal_matrices: list[np.ndarray] = []

    problems: list[str] = []
    seen = set(glt_labels)
    plans = []
    for label, raw_components in conjunctions:
        components = split_components(raw_components)
        if label in seen:
            problems.append(f"'{label}' is already a contrast or conjunction label")
        seen.add(label)
        if len(components) < 2:
            problems.append(f"'{label}' needs at least two components (got {components})")
            continue
        resolved = []
        for comp in components:
            sign = -1.0 if comp.startswith("-") else 1.0
            name = comp.lstrip("+-")
            if name in user_index:
                mat = np.asarray(glt_matrices[user_index[name]])
                if mat.ndim == 2 and mat.shape[0] > 1:
                    problems.append(f"'{label}': contrast '{name}' is an F-test, not a t-test")
                    continue
                resolved.append((user_index[name], sign))
                continue
            if name not in internal_index:
                try:
                    row = glt_rows_to_matrix([{name: (1.0, None)}], column_labels, stim_ranges)
                except ValueError:
                    problems.append(f"'{label}': unknown component '{name}'")
                    continue
                internal_index[name] = n_user + len(internal_labels)
                internal_labels.append(f"{_INTERNAL_PREFIX}{name}")
                internal_matrices.append(row)
            resolved.append((internal_index[name], sign))
        if len(resolved) == len(components) and len(set(resolved)) < 2:
            problems.append(f"'{label}' needs at least two distinct components")
        plans.append((label, tuple(resolved)))

    if problems:
        known = list(stim_ranges) + glt_labels
        raise ValueError(
            "conjunction: " + "; ".join(problems) + f". Known names: {', '.join(known)}"
        )
    design_info["glt_labels"] = glt_labels + internal_labels
    design_info["glt_matrices"] = glt_matrices + internal_matrices
    plan = ConjunctionPlan(n_user=n_user, conjunctions=tuple(plans))
    design_info["glt_conjunctions"] = plan
    return plan


def _as_2d(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t.unsqueeze(1) if t.ndim == 1 else t


def _combine(values: torch.Tensor, plan: ConjunctionPlan, signed: bool) -> torch.Tensor:
    cols = []
    for _, comps in plan.conjunctions:
        stack = torch.stack([values[:, i] * (sign if signed else 1.0) for i, sign in comps], dim=1)
        cols.append(stack.amin(dim=1))
    return torch.cat([values[:, : plan.n_user], torch.stack(cols, dim=1)], dim=1)


def apply_conjunctions(results, plan: ConjunctionPlan | None) -> None:
    """Replace a fit's internal component GLTs with its conjunction columns.

    Idempotent, and a no-op without a plan or contrasts, so every writer can call
    it on whichever result object (REML, OLS, -dsort snapshot) it is about to
    write. Coef and t take the signed minimum; a partial / semi-partial R², which
    is unsigned, takes the plain minimum (the weakest component's share).
    """
    if plan is None or results is None:
        return
    if getattr(results, "contrast_betas", None) is None:
        return
    if getattr(results, "_conjunctions_applied", False):
        return
    results.contrast_betas = _combine(_as_2d(results.contrast_betas), plan, signed=True)
    results.contrast_tstats = _combine(_as_2d(results.contrast_tstats), plan, signed=True)
    for attr in ("contrast_r2_partial", "contrast_r2_semipartial"):
        value = getattr(results, attr, None)
        if value is not None:
            setattr(results, attr, _combine(_as_2d(value), plan, signed=False))
    labels = list(getattr(results, "contrast_labels", None) or [])[: plan.n_user]
    results.contrast_labels = labels + [label for label, _ in plan.conjunctions]
    results._conjunctions_applied = True


def conjoin_by_label(
    plan: ConjunctionPlan,
    glt_labels: Sequence[str],
    coef_by_label: dict[str, np.ndarray],
    tstat_by_label: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The same minimum statistic as :func:`apply_conjunctions`, for a caller
    that holds its GLT results per label instead of as one column block.

    ``glt_labels`` is the extended GLT list the plan indexes into (user
    contrasts then ``__conj__*`` components). Returns
    ``(label, coef, tstat)`` per conjunction, skipping any whose components
    are not all present — a component that failed to compute cannot be
    silently treated as passing.
    """
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, comps in plan.conjunctions:
        names = [glt_labels[i] for i, _ in comps]
        if any(n not in coef_by_label or n not in tstat_by_label for n in names):
            continue
        signs = [sign for _, sign in comps]
        coef = np.minimum.reduce([coef_by_label[n] * s for n, s in zip(names, signs, strict=True)])
        tstat = np.minimum.reduce(
            [tstat_by_label[n] * s for n, s in zip(names, signs, strict=True)]
        )
        out.append((label, coef, tstat))
    return out
