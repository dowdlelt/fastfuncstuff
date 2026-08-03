"""Per-run data validity: which runs actually carry usable data for each voxel.

A voxel can be solidly inside the brain mask and still have no data in *some*
runs — partial coverage that shifts between sessions, a subject who moved on the
last run, a dropped slab. Left alone this is silent and wrong: the dead rows say
"response = 0 here" while the task regressor is nonzero, so the beta is diluted
toward zero by the fraction of dead runs, while the residual sum of squares those
rows contribute (also zero) deflates ``sigma2`` by the *same* fraction. The
t-statistic barely moves. A beta that is 35% too small still reports t = 4.2.

This module is the single detector both consumers share:

* the **guard** (default, ``ffs_reml -no_guard`` to disable) intersects the mask
  down to voxels valid in every run, which is always statistically correct but
  throws away real data at the edges;
* ``-handle_missing`` keeps those voxels by partitioning them into *families*
  that share a run-validity pattern and fitting each family against its own
  censored design (see :func:`build_families`).

Detection runs on **raw data, before blur and before scaling**. Blur smears
nonzero data into the dead voxels and destroys the exact-zero signature outright;
scaling a zero-mean run is a no-op that preserves it but tells us nothing new.

The rules, in order of trustworthiness:

1. **Constant** — a run whose timeseries never changes carries no information,
   whatever its level. This is the primary rule because it is scale-invariant
   (it compares std against the run's own level) and because an all-zero run is
   just its most common special case.
2. **Exact zeros** — catches the run that starts with data and drops to zero
   partway through, which is *not* constant and which a per-run polynomial
   cannot absorb (the step blows up the residual instead). An exact ``0.0`` in
   float fMRI data inside a brain mask is essentially never accidental; it is
   the zero-fill of a resample that went out of bounds.
3. **Negative** — physically impossible in magnitude EPI, so a run gone negative
   is an artifact. Applied at run level, never per sample: sinc/wsinc5 ringing
   puts isolated negative samples in perfectly good cortex. It also
   **self-disables** when the input clearly is not magnitude data (a detrended
   or mean-centered input is ~50% negative by construction), because a default-on
   rule that can empty the mask is worse than the bug it fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# A run whose relative std falls below this carries no usable signal. Compared
# against the run's own level so it is scale-invariant (raw counts or percent
# signal change both work).
CONSTANT_REL_TOL = 1e-6

# Fraction of a run's samples that must be negative before the run is called bad.
NEGATIVE_FRAC = 0.25

# If more than this fraction of all in-mask samples are negative, the input is
# not magnitude data and the negative rule is meaningless — disable it.
NEGATIVE_AUTODISABLE_FRAC = 0.05


@dataclass
class RunValidity:
    """Per-voxel, per-run validity plus why each run was rejected."""

    valid: torch.Tensor  # (n_voxels, n_runs) bool — run carried usable data
    constant: torch.Tensor  # (n_voxels, n_runs) bool — rejected as constant
    zeros: torch.Tensor  # (n_voxels, n_runs) bool — rejected for exact zeros
    negative: torch.Tensor  # (n_voxels, n_runs) bool — rejected as negative
    negative_rule_active: bool  # False if auto-disabled (input is not magnitude)

    @property
    def n_runs(self) -> int:
        return int(self.valid.shape[1])

    @property
    def all_runs_valid(self) -> torch.Tensor:
        """(n_voxels,) bool — the guard mask: usable in *every* run."""
        return self.valid.all(dim=1)

    @property
    def any_run_valid(self) -> torch.Tensor:
        """(n_voxels,) bool — salvageable by ``-handle_missing``."""
        return self.valid.any(dim=1)

    def summary(self) -> str:
        n_vox = int(self.valid.shape[0])
        n_full = int(self.all_runs_valid.sum())
        n_partial = int(self.any_run_valid.sum()) - n_full
        n_dead = n_vox - n_full - n_partial
        parts = [
            f"{n_full:,} voxels valid in all {self.n_runs} runs",
            f"{n_partial:,} partial",
            f"{n_dead:,} dead",
        ]
        why = []
        if bool(self.constant.any()):
            why.append(f"constant={int(self.constant.sum()):,}")
        if bool(self.zeros.any()):
            why.append(f"zeros={int(self.zeros.sum()):,}")
        if bool(self.negative.any()):
            why.append(f"negative={int(self.negative.sum()):,}")
        elif not self.negative_rule_active:
            why.append("negative=off")
        tail = f" (voxel-runs rejected: {', '.join(why)})" if why else ""
        return ", ".join(parts) + tail


def _reduce_run(
    run: torch.Tensor, constant_rel_tol: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-voxel (is_constant, has_zeros, negative_fraction) for one run."""
    run = run.float()
    mean = run.mean(dim=1)
    std = run.std(dim=1)
    # Scale-invariant: compare std against the run's own level. The absolute
    # floor catches a run that is constant *at* zero, where the relative test
    # degenerates.
    level = mean.abs().clamp_min(1e-12)
    is_constant = (std <= constant_rel_tol * level) | (std <= 1e-12)
    has_zeros = (run == 0).any(dim=1)
    neg_frac = (run < 0).float().mean(dim=1)
    return is_constant, has_zeros, neg_frac


class RunValidityAccumulator:
    """Streaming :func:`detect_run_validity` — observe one run at a time.

    ``ffs_reml`` blurs each run as it loads it, so the raw concatenated dataset
    never exists in memory. Detection has to happen inside that loop, on the raw
    run, before the blur smears nonzero data into the dead voxels and destroys
    the exact-zero signature.
    """

    def __init__(
        self,
        n_voxels: int,
        n_runs: int,
        *,
        constant_rel_tol: float = CONSTANT_REL_TOL,
        negative_frac: float = NEGATIVE_FRAC,
        check_negative: bool = True,
    ) -> None:
        shape = (n_voxels, n_runs)
        self.constant = torch.zeros(shape, dtype=torch.bool)
        self.zeros = torch.zeros(shape, dtype=torch.bool)
        self.neg_frac = torch.zeros(shape, dtype=torch.float32)
        self.constant_rel_tol = constant_rel_tol
        self.negative_frac = negative_frac
        self.check_negative = check_negative
        self._seen: set[int] = set()

    def observe_run(self, run_data: torch.Tensor, run_idx: int) -> None:
        """Record one **raw** run, shaped (n_voxels, n_run_timepoints)."""
        if run_data.shape[1] == 0:
            return
        c, z, n = _reduce_run(run_data.cpu(), self.constant_rel_tol)
        self.constant[:, run_idx] = c
        self.zeros[:, run_idx] = z
        self.neg_frac[:, run_idx] = n
        self._seen.add(int(run_idx))

    def finalize(self, *, mask: torch.Tensor | None = None, verbose: bool = True) -> RunValidity:
        missing = set(range(self.constant.shape[1])) - self._seen
        if missing:
            raise RuntimeError(f"run validity: never observed run(s) {sorted(missing)}")
        return _finalize_validity(
            self.constant,
            self.zeros,
            self.neg_frac,
            mask=mask,
            negative_frac=self.negative_frac,
            check_negative=self.check_negative,
            verbose=verbose,
        )


def _finalize_validity(
    constant: torch.Tensor,
    zeros: torch.Tensor,
    neg_frac_per_run: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    negative_frac: float,
    check_negative: bool,
    verbose: bool,
) -> RunValidity:
    negative = torch.zeros_like(constant)
    # The negative rule only means anything on magnitude data. If the input is
    # broadly negative it has been detrended/centered and the rule would delete
    # the brain -- turn it off and say so.
    negative_rule_active = check_negative
    if check_negative:
        if mask is not None:
            m = mask.reshape(-1).bool()
            global_neg = float(neg_frac_per_run[m].mean()) if bool(m.any()) else 0.0
        else:
            global_neg = float(neg_frac_per_run.mean())
        if global_neg > NEGATIVE_AUTODISABLE_FRAC:
            negative_rule_active = False
            if verbose:
                print(
                    f"  ⚠️  {global_neg:.0%} of in-mask samples are negative — this is not "
                    "magnitude data (detrended or mean-centered?), so the negative "
                    "run-validity rule is disabled. Constancy and zero checks still apply."
                )
        else:
            negative = neg_frac_per_run >= negative_frac

    result = RunValidity(
        valid=~(constant | zeros | negative),
        constant=constant,
        zeros=zeros,
        negative=negative,
        negative_rule_active=negative_rule_active,
    )
    if verbose:
        print(f"  Run validity: {result.summary()}")
    return result


def detect_run_validity(
    data: torch.Tensor,
    run_starts: list[int],
    *,
    mask: torch.Tensor | None = None,
    constant_rel_tol: float = CONSTANT_REL_TOL,
    negative_frac: float = NEGATIVE_FRAC,
    check_negative: bool = True,
    chunk_voxels: int = 200_000,
    verbose: bool = True,
) -> RunValidity:
    """Classify every (voxel, run) as carrying usable data or not.

    Parameters
    ----------
    data : (n_voxels, n_timepoints) tensor
        **Raw** data — before blur, before percent-signal scaling.
    run_starts : list[int]
        Start index of each run in the concatenated timeline.
    mask : (n_voxels,) bool tensor, optional
        Restricts the negative-rule auto-disable statistic to in-brain voxels.
        Out-of-brain air is full of near-zero constant voxels that would
        otherwise dominate. Detection itself still runs everywhere.
    constant_rel_tol, negative_frac : float
        See module docstring.
    check_negative : bool
        Disable the negative rule outright (it also self-disables, see above).
    chunk_voxels : int
        Voxel-axis chunk for the reductions, so whole-dataset inputs never
        materialise a second copy.

    Returns
    -------
    RunValidity
    """
    if data.ndim != 2:
        raise ValueError(f"data must be (n_voxels, n_timepoints), got {tuple(data.shape)}")
    n_voxels, n_total = data.shape
    starts = [int(s) for s in run_starts] or [0]
    ends = starts[1:] + [n_total]
    n_runs = len(starts)

    shape = (n_voxels, n_runs)
    constant = torch.zeros(shape, dtype=torch.bool)
    zeros = torch.zeros(shape, dtype=torch.bool)

    # The cheap per-run reductions. Everything accumulated is (n_voxels, n_runs)
    # -- a few MB even at whole-dataset scale -- so only the slices are big.
    neg_frac_per_run = torch.zeros(shape, dtype=torch.float32)
    for v0 in range(0, n_voxels, chunk_voxels):
        v1 = min(v0 + chunk_voxels, n_voxels)
        for r, (s, e) in enumerate(zip(starts, ends, strict=True)):
            run = data[v0:v1, s:e]
            if run.shape[1] == 0:
                continue
            c, z, n = _reduce_run(run, constant_rel_tol)
            constant[v0:v1, r], zeros[v0:v1, r], neg_frac_per_run[v0:v1, r] = c, z, n

    return _finalize_validity(
        constant,
        zeros,
        neg_frac_per_run,
        mask=mask,
        negative_frac=negative_frac,
        check_negative=check_negative,
        verbose=verbose,
    )


def _pattern_label(pattern: torch.Tensor) -> str:
    kept = [str(i) for i, k in enumerate(pattern.tolist()) if k]
    return "runs " + ",".join(kept)


@dataclass
class MissingFamily:
    """A set of voxels sharing one run-validity pattern, fitted together."""

    pattern: torch.Tensor  # (n_runs,) bool — which runs this family keeps
    voxel_indices: torch.Tensor  # (n_family_voxels,) long, into the masked voxel axis
    good_list: list[int]  # retained timepoints in the concatenated timeline

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_indices.numel())

    @property
    def n_runs_kept(self) -> int:
        return int(self.pattern.sum())

    def label(self) -> str:
        return _pattern_label(self.pattern)


def censor_design(
    design: torch.Tensor,
    good_list: list[int],
    zero_tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict a design to retained rows and drop the columns that go all-zero.

    Censoring a whole run zeroes that run's block-diagonal polynomial columns.
    Leaving them in is the classic rank-deficiency trap ([[Block-diagonal
    nuisance]]) — the normal equations go singular and the betas explode — so
    they must come out before the fit.

    Returns
    -------
    design_censored : (n_kept, n_surviving_cols) tensor
    keep_cols : (n_regressors,) bool tensor
    """
    rows = torch.as_tensor(good_list, dtype=torch.long, device=design.device)
    sub = design.index_select(0, rows)
    keep_cols = (sub.abs() > zero_tol).any(dim=0)
    return sub[:, keep_cols], keep_cols


def task_survival(
    design: torch.Tensor,
    task_indices: list[int] | None,
    good_list: list[int],
) -> torch.Tensor:
    """Fraction of each task regressor's design mass that survives censoring.

    A column can be nonzero after censoring and still be worthless — a stimulus
    regressor keeping 2 of 40 trials is estimable and garbage. Comparing retained
    sum-of-squares against the full column's is the continuous version of "does
    this voxel still see every event type", and it is what lets a family be
    rejected before it produces a wild beta.

    Returns ``(n_task,)`` in [0, 1]; an all-ones tensor when there are no task
    columns to check.
    """
    if not task_indices:
        return torch.ones(0, dtype=torch.float64)
    rows = torch.as_tensor(good_list, dtype=torch.long, device=design.device)
    cols = torch.as_tensor(task_indices, dtype=torch.long, device=design.device)
    full = design.index_select(1, cols).double()
    kept = full.index_select(0, rows)
    total = full.pow(2).sum(dim=0)
    retained = kept.pow(2).sum(dim=0)
    return torch.where(total > 0, retained / total, torch.ones_like(total))


# Retained design mass at or below this counts as "this regressor is simply not
# present in the surviving runs" rather than "present but weakly sampled".
ESTIMABLE_EPS = 1e-9


def task_runs_present(
    design: torch.Tensor,
    task_indices: list[int] | None,
    good_list: list[int],
    run_starts: list[int],
    n_timepoints: int,
) -> torch.Tensor:
    """How many *surviving* runs each task regressor actually has events in.

    Distinct from :func:`task_survival`: mass answers "how much of this
    regressor is left", this answers "in how many runs is it observed at all".
    A design where a condition appears in one run only can retain 100% of its
    mass from a single surviving run, and a design where it appears everywhere
    can retain 6% and still be observed in every run that survived.
    """
    if not task_indices:
        return torch.zeros(0, dtype=torch.long)
    starts = [int(s) for s in run_starts] or [0]
    ends = starts[1:] + [int(n_timepoints)]
    keep = set(int(t) for t in good_list)
    cols = torch.as_tensor(task_indices, dtype=torch.long, device=design.device)
    sub = design.index_select(1, cols).double()

    counts = torch.zeros(len(task_indices), dtype=torch.long)
    for s, e in zip(starts, ends, strict=True):
        rows = [t for t in range(s, e) if t in keep]
        if not rows:
            continue
        idx = torch.as_tensor(rows, dtype=torch.long, device=design.device)
        present = sub.index_select(0, idx).pow(2).sum(dim=0) > ESTIMABLE_EPS
        counts += present.cpu().long()
    return counts


def censored_design_is_full_rank(design: torch.Tensor, good_list: list[int]) -> bool:
    """Can the censored design be fitted at all?

    Retained mass per column is necessary but not sufficient: dropping runs can
    make two task regressors collinear with each other even though both still
    carry plenty of mass, and that is a rank deficiency the per-column test
    cannot see. One SVD per family is nothing next to the fit it guards.
    """
    sub, _ = censor_design(design, good_list)
    if sub.numel() == 0 or sub.shape[1] == 0:
        return False
    return int(torch.linalg.matrix_rank(sub.double())) == int(sub.shape[1])


def build_families(
    validity: RunValidity,
    run_starts: list[int],
    n_timepoints: int,
    *,
    candidate_mask: torch.Tensor | None = None,
    min_family_voxels: int = 50,
    max_families: int = 32,
    min_runs: int = 1,
    design: torch.Tensor | None = None,
    task_indices: list[int] | None = None,
    min_task_mass: float = 0.0,
    min_task_runs: int = 1,
    verbose: bool = True,
) -> tuple[list[MissingFamily], torch.Tensor]:
    """Partition partially-valid voxels into families sharing a run pattern.

    Voxels are grouped by their *run-level* validity bitmask, so a mid-run
    dropout is rounded **up** to "this whole run is censored". That is
    conservative — it discards the good half of one run — but it caps the number
    of distinct patterns at ``2^n_runs`` instead of letting a TR-level keep
    pattern generate a near-unique family per voxel. Each family costs a full
    REML fit with its own autocorrelation cache (which is keyed on timepoint
    count and so cannot be shared), making family count the dominant cost.

    When ``design`` and ``task_indices`` are given, two *different* things are
    checked and it matters that they are not confused:

    **Estimability — a hard constraint, never tunable.** If a condition is only
    ever presented in run 1 and a family censors run 1, that condition has no
    data at all: its column is identically zero, its beta is undefined, and any
    contrast touching it is meaningless. The same applies when censoring makes
    two task regressors collinear, which per-column mass cannot see, so the
    censored design is also rank-checked. Families failing either are always
    rejected. Requiring every task regressor to survive has a useful side
    effect: each surviving family carries the full task column set, so contrasts
    stay defined and the bucket layout is uniform. Only nuisance columns drop.

    **Precision — a soft preference, off by default.** A condition observed in
    every run but retaining only 24% of its mass *is* estimable; it is simply
    estimated from less data. That is not a reason to discard the voxel, because
    the dof accounting already reports it honestly: fewer degrees of freedom, a
    larger standard error, and a z-score at the voxel's true dof. So
    ``min_task_mass`` defaults to 0 (no floor) and ``min_task_runs`` to 1
    (observed at all). Raise them only if you would rather drop weakly-sampled
    voxels than read an honest, unimpressive statistic — the usual reason being
    that the *Coef* map still shows a large noisy value that people threshold on
    by eye.

    Families that are too small, keep too few runs, are unestimable, fall below a
    requested precision floor, or fall outside ``max_families`` are *not*
    returned; their voxels come back in the ``demoted`` mask for the caller to
    drop from the analysis mask instead.

    Returns
    -------
    families : list[MissingFamily]
        Sorted largest-first.
    demoted : (n_voxels,) bool tensor
        Voxels that were candidates but did not earn a family.
    """
    starts = [int(s) for s in run_starts] or [0]
    ends = starts[1:] + [int(n_timepoints)]

    candidates = validity.any_run_valid & ~validity.all_runs_valid
    if candidate_mask is not None:
        candidates = candidates & candidate_mask.reshape(-1).bool()
    cand_idx = candidates.nonzero(as_tuple=True)[0]

    demoted = torch.zeros(validity.valid.shape[0], dtype=torch.bool)
    if cand_idx.numel() == 0:
        return [], demoted

    # Same primitive glm/arma.py uses to group voxels by (a, b): unique rows plus
    # an inverse index. Not a list of lists -- a (n_patterns, n_runs) bool tensor.
    patterns, inverse = torch.unique(validity.valid[cand_idx], dim=0, return_inverse=True)

    order = sorted(
        range(int(patterns.shape[0])),
        key=lambda p: int((inverse == p).sum()),
        reverse=True,
    )

    families: list[MissingFamily] = []
    n_rejected_task = 0
    for p in order:
        members = cand_idx[inverse == p]
        pattern = patterns[p]
        n_kept_runs = int(pattern.sum())
        too_small = int(members.numel()) < min_family_voxels
        too_few_runs = n_kept_runs < min_runs
        over_budget = len(families) >= max_families
        if too_small or too_few_runs or over_budget:
            demoted[members] = True
            continue
        good_list = [
            t
            for r, (s, e) in enumerate(zip(starts, ends, strict=True))
            if bool(pattern[r])
            for t in range(s, e)
        ]
        if design is not None and task_indices:
            reason = None
            mass = task_survival(design, task_indices, good_list)

            # (1) Estimability — always enforced, whatever the thresholds say.
            if bool((mass <= ESTIMABLE_EPS).any()):
                dead = [task_indices[i] for i, m in enumerate(mass.tolist()) if m <= ESTIMABLE_EPS]
                reason = (
                    f"task regressor(s) {dead} have NO data in the surviving runs "
                    "— unestimable, so every contrast touching them would be meaningless"
                )
            elif not censored_design_is_full_rank(design, good_list):
                reason = (
                    "the censored design is rank-deficient (censoring made "
                    "regressors collinear), so the fit is not identifiable"
                )
            else:
                # (2) Precision — only if the user asked for a floor.
                n_runs_seen = task_runs_present(
                    design, task_indices, good_list, run_starts, n_timepoints
                )
                if min_task_runs > 1 and bool((n_runs_seen < min_task_runs).any()):
                    reason = (
                        f"a task regressor is observed in only "
                        f"{int(n_runs_seen.min())} surviving run(s) "
                        f"(-guard_min_task_runs {min_task_runs})"
                    )
                elif min_task_mass > 0.0 and bool((mass < min_task_mass).any()):
                    reason = (
                        f"a task regressor retains only {float(mass.min()):.0%} of its "
                        f"design mass (-guard_min_task_mass {min_task_mass:.0%}). It is "
                        "estimable — just from less data — so consider lowering or "
                        "removing this floor and reading the z-score instead"
                    )

            if reason is not None:
                demoted[members] = True
                n_rejected_task += 1
                if verbose:
                    print(
                        f"    ✗ {_pattern_label(pattern)}: rejected — {reason}; "
                        f"{int(members.numel()):,} voxels demoted to the guard"
                    )
                continue
        families.append(MissingFamily(pattern=pattern, voxel_indices=members, good_list=good_list))

    if verbose:
        n_rescued = sum(f.n_voxels for f in families)
        n_demoted = int(demoted.sum())
        print(
            f"  Missing-data families: {len(families)} pattern(s) covering "
            f"{n_rescued:,} voxels; {n_demoted:,} voxels demoted to the guard"
        )
        for f in families:
            print(f"    • {f.label()}: {f.n_voxels:,} voxels, {len(f.good_list)} TRs")
    return families, demoted


def run_inclusion_map(
    validity: RunValidity,
    families: list[MissingFamily],
    final_mask: torch.Tensor,
) -> torch.Tensor:
    """(n_voxels, n_runs) float 0/1 — which runs fed each voxel's fit.

    This is the bookkeeping artifact ``ffs_reml -save_runmask`` writes and
    ``ffs_util_updatedof -adjust_dof_set`` consumes: a per-run dof cost must only
    be charged to a voxel for the runs it actually used.
    """
    n_voxels, n_runs = validity.valid.shape
    inc = torch.zeros((n_voxels, n_runs), dtype=torch.float32)
    m = final_mask.reshape(-1).bool()
    # Guard-mask voxels used every run; family voxels used their pattern.
    inc[m & validity.all_runs_valid] = 1.0
    for fam in families:
        inc[fam.voxel_indices] = fam.pattern.to(torch.float32)
    inc[~m] = 0.0
    return inc


def dof_loss_map(
    n_voxels: int,
    families: list[MissingFamily],
    family_dofs: list[int],
    dof_full: int,
    final_mask: torch.Tensor,
) -> torch.Tensor:
    """(n_voxels,) float — degrees of freedom each voxel lost to censoring.

    Fed to ``-adjust_dof`` so the z-scores come out at each voxel's true dof.
    Zero for guard-mask voxels, which lost nothing.

    This is deliberately built from each family's **fitted** dof rather than from
    the censored timepoint count: censoring a run also drops that run's
    block-diagonal polynomial columns, so the design shrinks alongside the data
    and the loss is ``timepoints_dropped - columns_dropped``, not the first term
    alone. Predicting it from run lengths overstates the loss by the polort width
    of every censored run.
    """
    if len(families) != len(family_dofs):
        raise ValueError(f"{len(families)} families but {len(family_dofs)} dofs")
    loss = torch.zeros(n_voxels, dtype=torch.float32)
    for fam, dof in zip(families, family_dofs, strict=True):
        loss[fam.voxel_indices] = float(dof_full - int(dof))
    loss[~final_mask.reshape(-1).bool()] = 0.0
    return loss


def fit_and_merge_families(
    results,
    data: torch.Tensor,
    design: torch.Tensor,
    families: list[MissingFamily],
    run_starts: list[int],
    n_timepoints: int,
    *,
    fit_kwargs: dict,
    tr: float,
    method: str = "arma11",
    verbose: bool = True,
) -> torch.Tensor:
    """Refit each family against its own censored design, in place on ``results``.

    ``results`` is the main fit over the whole masked voxel set. Family voxels
    were fitted there too, with their dead rows still in, which is exactly the
    wrong answer this module exists to prevent — so every family's rows are
    overwritten here by a fit that drops those rows. Fitting them twice is a
    small waste (family voxels are edge data by construction) in exchange for
    leaving the main path completely untouched.

    Why the design is re-derived per family rather than reused: censoring a whole
    run zeroes that run's block-diagonal polynomial columns, and leaving them in
    is not merely inelegant — ``torch.linalg.cholesky`` fails outright on the
    singular normal equations ("the leading minor of order 8 is not
    positive-definite"). They have to be dropped. Because the guard only admits
    families where every *task* regressor survives, the dropped columns are
    always nuisance, so the fitted output keeps the same column layout and can be
    written into the same bucket.

    ``method="ols"`` runs the same partition through :func:`fit_glm` instead.
    It is strictly the simpler case: with no ARMA noise model there is no
    ``(a, b)`` grid to re-search and no ``tau`` to carry across the censoring
    gaps, so a family is just least squares on fewer rows.

    Returns
    -------
    dof_loss : (n_masked_voxels,) float tensor
        ``main_dof - family_dof`` per voxel, 0 outside families. This is the map
        ``-adjust_dof`` needs to put every voxel's z-score at its true dof.
    """
    from fastfuncstuff.glm.arma import (
        OLS_VOXEL_SCATTER_ATTRS,
        TIME_AXIS_ATTRS,
        VOXEL_SCATTER_ATTRS,
        build_censor_run_info,
        fit_glm_arma11,
    )

    is_ols = method == "ols"
    scatter_attrs = OLS_VOXEL_SCATTER_ATTRS if is_ols else VOXEL_SCATTER_ATTRS
    n_masked = int(data.shape[0])
    dof_loss = torch.zeros(n_masked, dtype=torch.float32)
    if not families:
        return dof_loss

    dof_full = int(results.dof)
    if verbose:
        print(f"\n🔁 Refitting {len(families)} missing-data family/families (censored)")

    for fam_i, fam in enumerate(families):
        rows = torch.as_tensor(fam.good_list, dtype=torch.long)
        sub_design, keep_cols = censor_design(design, fam.good_list)
        sub_run_starts, sub_tau = build_censor_run_info(
            run_starts, n_timepoints, good_list=fam.good_list
        )
        sub_data = data[fam.voxel_indices][:, rows]

        if verbose:
            print(
                f"  ─── family {fam_i + 1}/{len(families)} ({fam.label()}): "
                f"{fam.n_voxels:,} voxels, {len(fam.good_list)} TRs, "
                f"{int((~keep_cols).sum())} nuisance column(s) dropped ───"
            )

        sub_kwargs = dict(fit_kwargs)
        # Contrasts are defined against the FULL design; drop the same columns so
        # the GLT rows still line up with the censored design.
        for key in ("glt_matrices",):
            mats = sub_kwargs.get(key)
            if mats:
                keep_np = keep_cols.cpu().numpy()
                sub_kwargs[key] = [np.asarray(m).reshape(-1)[keep_np] for m in mats]
        # Column indices refer to positions in the full design; remap them onto
        # the censored one. All task columns survive by construction (see
        # build_families), so this is a pure re-indexing, never a loss.
        if sub_kwargs.get("task_indices"):
            remap = torch.cumsum(keep_cols.long(), 0) - 1
            sub_kwargs["task_indices"] = [
                int(remap[i]) for i in sub_kwargs["task_indices"] if bool(keep_cols[i])
            ]

        if is_ols:
            from fastfuncstuff.glm.core import fit_glm

            # OLS has no noise model, so the censoring gaps need no tau and the
            # run boundaries only matter to whatever built the design.
            fam_results = fit_glm(sub_data, sub_design, tr=tr, **sub_kwargs)
        else:
            fam_results = fit_glm_arma11(
                sub_data,
                sub_design,
                tr=tr,
                run_starts=sub_run_starts,
                tau=sub_tau,
                **sub_kwargs,
            )

        _scatter_family(results, fam_results, fam, rows, scatter_attrs, TIME_AXIS_ATTRS)
        dof_loss[fam.voxel_indices] = float(dof_full - int(fam_results.dof))
        if verbose:
            print(
                f"      dof {dof_full} → {fam_results.dof} (lost {dof_full - int(fam_results.dof)})"
            )

    return dof_loss


def _scatter_family(results, fam_results, fam, rows, scatter_attrs, time_attrs) -> None:
    """Overwrite the family's rows in the main results with its censored refit."""
    idx = fam.voxel_indices
    for attr in scatter_attrs:
        dest = getattr(results, attr, None)
        src = getattr(fam_results, attr, None)
        if dest is None or src is None:
            continue
        src = src.cpu()
        if attr in time_attrs:
            # Censored fits are short along time: place each retained sample back
            # at its original index and leave the censored TRs at zero, so an
            # errts volume stays aligned with the input timeline.
            block = torch.zeros((src.shape[0], dest.shape[1]), dtype=dest.dtype, device=dest.device)
            block[:, rows] = src.to(dest.dtype)
            dest[idx] = block
            continue
        if tuple(dest.shape[1:]) != tuple(src.shape[1:]):
            # Shape disagreement means the censored design changed a
            # non-time axis we cannot align; skip rather than write nonsense.
            continue
        dest[idx] = src.to(dest.dtype)


def subset_run_validity(validity: RunValidity, mask: torch.Tensor) -> RunValidity:
    """Restrict a volume-indexed RunValidity to a masked voxel subset.

    The detector runs over the whole volume (it is what decides the mask), but
    the fit works on masked voxels, so families must be indexed the same way the
    fitted arrays are.
    """
    m = mask.reshape(-1).bool()
    return RunValidity(
        valid=validity.valid[m],
        constant=validity.constant[m],
        zeros=validity.zeros[m],
        negative=validity.negative[m],
        negative_rule_active=validity.negative_rule_active,
    )
