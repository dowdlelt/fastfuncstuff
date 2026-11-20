"""What each nuisance regressor, and each group of them, did in the fit (ffs_reml
-nuisance_summary).

Groups come from the column labels (motion ``mot_demean.r01[3]`` -> ``mot_demean``,
ffs_physio slibase ``card.c1`` -> ``card``, ``Run#2Pol#1`` -> ``polort``), or from
``-nuisance_group NAME REGEX``. Per group, and then per column of it, the bucket holds
the size of the fitted signal first and its significance last, as AFNI orders a bucket:

- ``PSC``: temporal SD of the fitted signal (X_S b_S), percent of the voxel mean.
- ``R2unique``: share of the voxel's variance lost if the set were dropped,
  q F (1 - R^2) / dof. The F is whitened and R^2 is not, so this is the usual
  approximation, not an exact refit.
- ``R2partial``: share of what everything else leaves that the set takes,
  q F / (q F + dof); exact in the whitened model. Tagged Beta(q/2, dof/2).
- ``Fstat`` (group, Wald test of the whole set) / ``Tstat`` (column), with AFNI
  stat parameters so a viewer thresholds by p.

Columns also lead with their ``Coef``. Drift (polort) gets the group entry only, and
tests degree >= 1: the run means are left out.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

_POLORT = re.compile(r"^(Run#\d+Pol#\d+|r\d+_poly\d+)$")
# a run's constant: its mean is not nuisance work, and with it the drift test would
# report the signal level (R2 measured about the mean cannot hold it)
_RUN_MEAN = re.compile(r"^(Run#\d+Pol#0|r\d+_poly0)$")


@dataclass
class NuisanceSet:
    name: str
    columns: list[int]  # full-design indices (dsort column k = n_design + k)
    labels: list[str]
    per_column: bool = True
    column_sets: list[int] = field(default_factory=list)  # filled by build_nuisance_sets


def group_name(label: str) -> str:
    """Default group of a column label."""
    if _POLORT.match(label):
        return "polort"
    s = re.sub(r"\[\d+\]$", "", label)
    s = re.sub(r"#\d+$", "", s)
    s = re.sub(r"_run\d+$", "", s)  # per-run -dsort columns
    s = re.sub(r"[_.]?\d+$", "", s)
    if "." in s:  # name.r01 (AFNI per-run ortvec), card.c (physio harmonic)
        s = s.split(".")[0]
    return s or label


def build_nuisance_sets(
    column_labels: Sequence[str],
    stim_indices: Sequence[int],
    dsort_labels: Sequence[str] = (),
    overrides: Sequence[tuple[str, str]] = (),
) -> tuple[list[NuisanceSet], list[list[int]]]:
    """Group the non-task columns, in design order.

    Returns the groups and the flat ``column_sets`` for the fit: each group, then (when
    ``per_column``) each of its columns alone. ``overrides`` are (name, regex): every
    nuisance column whose label matches joins that group, before the defaults apply.
    """
    stim = set(int(i) for i in stim_indices)
    n = len(column_labels)
    cols = [
        (i, str(lab))
        for i, lab in enumerate(column_labels)
        if i not in stim and not _RUN_MEAN.match(str(lab))
    ]
    cols += [(n + k, str(lab)) for k, lab in enumerate(dsort_labels)]
    pats = [(name, re.compile(rx)) for name, rx in overrides]
    groups: dict[str, NuisanceSet] = {}
    for i, lab in cols:
        name = next((nm for nm, rx in pats if rx.search(lab)), None) or group_name(lab)
        g = groups.setdefault(name, NuisanceSet(name, [], [], per_column=name != "polort"))
        g.columns.append(i)
        g.labels.append(lab)
    sets: list[list[int]] = []
    for g in groups.values():
        g.column_sets = [len(sets)]
        sets.append(list(g.columns))
        if g.per_column:
            for c in g.columns:
                g.column_sets.append(len(sets))
                sets.append([c])
    return list(groups.values()), sets


def nuisance_summary_bricks(
    groups: Sequence[NuisanceSet],
    set_fstats: np.ndarray,
    set_sd: np.ndarray,
    set_coef: np.ndarray,
    r2: np.ndarray,
    mean: np.ndarray,
    dof: int,
) -> tuple[np.ndarray, list[str], dict[int, tuple[str, tuple[float, ...]]]]:
    """(values (n_vox, n_bricks), brick labels, {brick: (AFNI stat type, params)}).

    The ``set_*`` arrays are the fit's, (n_vox, n_sets), indexed by each group's
    ``column_sets``; a column's t is sign(coef) sqrt(F) of its one-column set. ``mean``
    is each voxel's temporal mean (PSC denominator; PSC 0 where it is not positive)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(mean > 0, 100.0 / mean, 0.0)
    one_minus_r2 = np.clip(1.0 - r2, 0.0, 1.0)
    out: list[np.ndarray] = []
    labels: list[str] = []
    stat: dict[int, tuple[str, tuple[float, ...]]] = {}

    def add(x, lab, st=None):
        if st is not None:
            stat[len(out)] = st
        out.append(np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        labels.append(lab)

    for g in groups:
        si = g.column_sets[0]
        q = len(g.columns)
        qF = q * set_fstats[:, si]
        add(set_sd[:, si] * pct, f"{g.name}_PSC")
        add(qF * one_minus_r2 / dof, f"{g.name}_R2unique")
        add(qF / (qF + dof), f"{g.name}_R2partial", ("fibt", (q / 2, dof / 2)))
        add(set_fstats[:, si], f"{g.name}_Fstat", ("fift", (q, dof)))
        if not g.per_column:
            continue
        for lab, si in zip(g.labels, g.column_sets[1:], strict=True):
            b = set_coef[:, si]
            t = np.sign(b) * np.sqrt(np.clip(set_fstats[:, si], 0, None))
            add(b, f"{lab}_Coef")
            add(set_sd[:, si] * pct, f"{lab}_PSC")
            add(t * t * one_minus_r2 / dof, f"{lab}_R2unique")
            add(t * t / (t * t + dof), f"{lab}_R2partial", ("fibt", (0.5, dof / 2)))
            add(t, f"{lab}_Tstat", ("fitt", (dof,)))
    return np.stack(out, 1), labels, stat
