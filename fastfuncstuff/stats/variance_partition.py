"""Per-voxel variance partitioning for fully crossed factorial designs.

Method and rationale: ``../fmri_wiki/concepts/Variance partitioning.md``.

Given single-trial betas from an exhaustively crossed design (the motivating case:
21 tasks x 20 stimuli, 3 repeats per cell), split the reliable response into what only
factor A explains, what only factor B explains, what they share, and what lives in their
interaction -- all cross-validated over held-out repeats.

Three facts about this design shape the whole implementation:

1. **Shared variance is zero by construction.** Exhaustive crossing makes the factor
   column spaces orthogonal, so a nonzero ``C`` means the design lost its balance
   downstream (censoring, dropped trials). It is reported as a *diagnostic*, not a result.

2. **The saturated-vs-additive comparison is an estimation-noise comparison, not a
   degrees-of-freedom one.** At 3 repeats under leave-one-repeat-out, the additive model
   averages ~40 trials per estimated level while the saturated model averages 2. That is
   ~20x the prediction noise, and it drives the measured interaction to zero even when it
   is real and large. Unregularized cell means are unusable at realistic repeat counts, so
   every band gets its own shrinkage.

3. **That shrinkage is free, because of (1).** Balance makes ``X^T X`` block diagonal, so
   banded ridge decouples exactly -- no joint hyperparameter grid. With orthonormal
   contrast codes it collapses further to ``X^T X = n*I``, and each band's ridge is a
   single scalar ``gamma_b = n/(n + lambda_b)`` multiplying that band's OLS coefficients.
   That scalar is exactly the fractional-ridge ``frac`` of :mod:`fastfuncstuff.glm.ridge`.

Because predictions are *linear* in the per-band gammas, the optimal gammas come from a
tiny per-voxel least-squares solve rather than a grid search. By default each outer fold's
gammas are selected from inner predictions confined to its training data, so the reported
R2 is genuinely held out. Setting nested_gamma to false restores the original faster
procedure: one gamma is selected on the assembled reporting-fold predictions, making raw
R2 mildly optimistic. The permutation null re-runs whichever selection mode produced the
observed statistic.

Two consequences of the gammas being *clamped* to [0, 1] are worth knowing before reading
the maps. First, the clamp is what makes shared variance ``C`` depart from exactly zero on
a balanced design: if a band's unconstrained optimum lands outside [0, 1] in one nested
model but not in another, the R2 differences stop telling a perfectly additive story. That
is a much more common source of small nonzero ``C`` than lost balance, so read ``C`` against
the Gram diagnostic, not on its own. Second, every model that *includes* a band pays the
same clamp, so the partition differences stay comparable.

The interaction band gets a second, structural regularizer on top of its gamma: the cell
means with the additive part stripped form a matrix ``E``, and its useful rank is small.
Both the hard rank sweep and the nuclear (singular-value soft-threshold) sweep run under
the *same* fitted gammas as the reported models, so ``rank_r2[:, 0]`` is ``R2(M_add)`` and
the curve is directly comparable to it -- see :func:`partition_variance`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import get_device

# Off-diagonal Gram mass above this fraction of the diagonal means the bands are no longer
# orthogonal and the closed-form decoupling in this module does not hold.
BALANCE_TOL = 1e-6

# Noise ceiling below which ratio-valued maps (preference) are reported as 0: an oracle
# model could not reach 1% of the variance there, so the ratio is noise over noise.
NC_FLOOR_FOR_RATIO = 0.01

# Sums of squares below this are a constant voxel; dividing by them makes noise.
_SS_FLOOR = 1e-12


@dataclass
class FactorDesign:
    """Orthonormal contrast coding of a crossed factorial design.

    ``bands`` maps a band name to its trial-level columns. Band names are the factor name
    for a main effect and ``"a:b"`` for an interaction; the intercept is separate because
    it is never shrunk (it is estimated from every trial and is effectively noiseless).
    """

    factor_names: list[str]
    levels: list[list]
    codes: Tensor  # (n_trials, n_factors) int64 level indices
    contrasts: list[Tensor]  # per factor, (n_levels, n_levels - 1) orthonormal
    bands: dict[str, Tensor]  # band name -> (n_trials, n_cols)
    band_order: list[str]
    band_members: dict[str, tuple[int, ...]]  # band name -> factor indices it spans
    balanced: bool
    max_offdiag: float
    cell_counts: Tensor  # (n_levels_0, n_levels_1, ...) trials per cell

    @property
    def main_bands(self) -> list[str]:
        """Main-effect band names, in factor order."""
        return [n for n in self.band_order if len(self.band_members[n]) == 1]

    @property
    def pair_bands(self) -> list[str]:
        """Two-way interaction band names -- the ones with an SVD structure to report."""
        return [n for n in self.band_order if len(self.band_members[n]) == 2]

    def bands_involving(self, factor: int) -> list[str]:
        return [n for n in self.band_order if factor in self.band_members[n]]

    @property
    def n_trials(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)


@dataclass
class VarPartResult:
    """Per-voxel partition maps plus the diagnostics needed to trust them."""

    r2: dict[str, Tensor] = field(default_factory=dict)  # model name -> (n_voxels,)
    unique: dict[str, Tensor] = field(default_factory=dict)  # factor name -> (n_voxels,)
    # band name -> (n_voxels,) that band's unique contribution to the FULL model. Main
    # effects, two-way interactions and higher-order terms all appear here; ``unique`` is
    # the same idea restricted to the additive model, so factors stay comparable.
    band_unique: dict[str, Tensor] = field(default_factory=dict)
    shared: Tensor | None = None
    interaction: Tensor | None = None
    gammas: dict[str, Tensor] = field(default_factory=dict)  # band name -> (n_voxels,)
    preference: Tensor | None = None  # two factors only

    # Two-way interaction structure, keyed by interaction band name. Higher-order bands get
    # variance and shrinkage but no structure: their coefficient array is a tensor, and
    # rank needs a CP/Tucker decomposition rather than an SVD.
    pair_rank_e: dict[str, Tensor] = field(default_factory=dict)  # -1 = below the SNR floor
    pair_rank_e_raw: dict[str, Tensor] = field(default_factory=dict)  # same, no SNR mask
    pair_rank_r2: dict[str, Tensor] = field(default_factory=dict)  # (n_voxels, max_rank + 1)
    pair_nuclear_tau: dict[str, Tensor] = field(default_factory=dict)
    pair_nuclear_gain: dict[str, Tensor] = field(default_factory=dict)
    # band -> factor -> (n_voxels,) |cos| between the leading singular vector and that
    # parent factor's main effect. 1 = pure multiplicative gain on that factor's profile,
    # 0 = reorganisation. Zeroed where the rank is below 1.
    pair_gain_alignment: dict[str, dict[str, Tensor]] = field(default_factory=dict)

    # Flat aliases for the single-interaction (two-factor) case; empty above that.
    rank_e: Tensor | None = None
    rank_e_raw: Tensor | None = None
    rank_r2: Tensor | None = None
    nuclear_r2: Tensor | None = None
    nuclear_tau: Tensor | None = None
    nuclear_gain: Tensor | None = None
    gain_alignment: dict[str, Tensor] = field(default_factory=dict)

    ncsnr: Tensor | None = None
    noise_ceiling: Tensor | None = None
    heldout_sst: Tensor | None = None
    diagnostics: dict = field(default_factory=dict)


def _orthonormal_contrasts(n_levels: int, dtype: torch.dtype) -> Tensor:
    """Orthonormal contrast basis for one factor: columns span the sum-to-zero subspace.

    Dummy/treatment coding would give neither orthonormal columns nor orthogonality to the
    intercept, and the closed-form band decoupling this module relies on collapses without
    both. QR of ``[1, e_1, ..., e_{L-1}]`` puts the constant in the first Q column, so
    dropping it leaves an orthonormal basis orthogonal to the intercept.
    """
    if n_levels < 2:
        raise ValueError(f"factor needs >= 2 levels, got {n_levels}")
    m = torch.eye(n_levels, dtype=dtype)
    m[:, 0] = 1.0
    q, _ = torch.linalg.qr(m)
    return q[:, 1:].contiguous()


def build_factor_design(
    factor_codes: dict[str, np.ndarray],
    dtype: torch.dtype = torch.float64,
) -> FactorDesign:
    """Build orthonormal contrast bands for a crossed factorial design.

    Parameters
    ----------
    factor_codes
        Maps factor name -> per-trial label array (any dtype; labels are factorized).
        Insertion order sets band order, so it also sets output map naming.
    dtype
        Contrast construction runs in float64 by default -- the Gram is compared against a
        tight tolerance to decide the fast path, and float32 QR noise alone can exceed it.
    """
    if len(factor_codes) < 2:
        raise ValueError("variance partitioning needs at least 2 factors")

    names = list(factor_codes.keys())
    levels: list[list] = []
    code_cols: list[Tensor] = []
    for name in names:
        labels = np.asarray(factor_codes[name])
        uniq, idx = np.unique(labels, return_inverse=True)
        levels.append(list(uniq))
        code_cols.append(torch.as_tensor(idx, dtype=torch.int64))
    codes = torch.stack(code_cols, dim=1)

    contrasts = [_orthonormal_contrasts(len(lv), dtype) for lv in levels]

    # One band per non-empty subset of factors: k main effects, then every interaction.
    # An interaction band is the elementwise (row-wise Khatri-Rao) product of its members'
    # columns. Because each factor's contrast columns sum to zero over its own levels, a
    # product band is automatically orthogonal to every band it is not a superset of --
    # which is what makes the whole partition decouple under balance, at any k.
    bands: dict[str, Tensor] = {}
    band_order: list[str] = []
    band_members: dict[str, tuple[int, ...]] = {}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(range(len(names)), size):
            name = ":".join(names[f] for f in subset)
            cols = contrasts[subset[0]][codes[:, subset[0]], :]
            for f in subset[1:]:
                nxt = contrasts[f][codes[:, f], :]
                cols = (cols.unsqueeze(2) * nxt.unsqueeze(1)).reshape(cols.shape[0], -1)
            bands[name] = cols.contiguous()
            band_order.append(name)
            band_members[name] = subset

    shape = tuple(len(lv) for lv in levels)
    flat = torch.zeros(codes.shape[0], dtype=torch.int64)
    for f in range(len(names)):
        flat = flat * shape[f] + codes[:, f]
    cell_counts = torch.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)

    # Balance check. Under an exhaustively crossed balanced design every band Gram is
    # n*I and cross-band Grams vanish; departures mean censoring or dropped trials broke
    # the orthogonality everything downstream assumes.
    full = torch.cat(
        [torch.ones(codes.shape[0], 1, dtype=dtype)] + [bands[n] for n in band_order], dim=1
    )
    gram = full.T @ full
    diag = torch.diagonal(gram)
    offdiag = gram - torch.diag(diag)
    max_offdiag = float(offdiag.abs().max() / diag.abs().max())
    balanced = bool(max_offdiag < BALANCE_TOL) and bool(cell_counts.min() > 0)

    return FactorDesign(
        factor_names=names,
        levels=levels,
        codes=codes,
        contrasts=contrasts,
        bands=bands,
        band_order=band_order,
        band_members=band_members,
        balanced=balanced,
        max_offdiag=max_offdiag,
        cell_counts=cell_counts,
    )


def flat_cell_index(design: FactorDesign) -> Tensor:
    """Row-major flat index of each trial's cell in the k-dimensional level grid."""
    shape = [len(lv) for lv in design.levels]
    flat = torch.zeros(design.n_trials, dtype=torch.int64)
    for f in range(design.n_factors):
        flat = flat * shape[f] + design.codes[:, f]
    return flat


def derive_repeat_index(design: FactorDesign) -> np.ndarray:
    """Number each trial by how many times its cell has already occurred.

    Used when the sidecar has no explicit ``repeat`` column. Row order is trial order, so
    this reproduces "1st/2nd/3rd presentation of this cell".
    """
    flat = flat_cell_index(design).numpy()
    seen: dict[int, int] = {}
    out = np.empty(len(flat), dtype=np.int64)
    for i, cell in enumerate(flat):
        out[i] = seen.get(int(cell), 0)
        seen[int(cell)] = out[i] + 1
    return out


def build_repeat_folds(
    repeat: np.ndarray,
    run: np.ndarray | None = None,
    cell: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Leave-one-repeat-out folds, refusing any fold that leaks a run across the split.

    Every cell is trained and tested exactly once per fold, so the folds stay balanced.
    The constraint that matters is run locality: if two repeats of a cell share a run, then
    run-level nuisance (drift, motion, alertness, shared noise PCs) is present in both the
    training and test side of *that cell*, which inflates every model equally and reads as
    signal. Same trap as fold-local nuisance projection in LORO cross-validation.

    ``cell`` is what makes that check mean what the paragraph above says. Without it the
    test degrades to "does any run appear on both sides", which every interleaved design
    fails by construction: a run holding trials from many cells necessarily contributes
    training trials (other cells) and test trials (this cell) at once, and that is harmless
    -- the two sets share no cell mean. Pass the per-trial cell identity and the check
    becomes per (cell, run), which is the leak that actually inflates prediction.
    """
    repeat = np.asarray(repeat)
    uniq = np.unique(repeat)
    folds = []
    for r in uniq:
        test = np.flatnonzero(repeat == r)
        train = np.flatnonzero(repeat != r)
        folds.append((train, test))

    diag: dict = {"n_folds": len(folds), "repeat_levels": [int(u) for u in uniq]}
    if run is not None:
        run = np.asarray(run).astype(str)
        keys = (
            np.array(
                [f"{c}\x00{r}" for c, r in zip(np.asarray(cell).astype(str), run, strict=True)]
            )
            if cell is not None
            else run
        )
        leaks = []
        for fi, (train, test) in enumerate(folds):
            shared = np.intersect1d(np.unique(keys[train]), np.unique(keys[test]))
            if shared.size:
                entry: dict = {"fold": fi, "n_leaks": int(shared.size)}
                if cell is None:
                    entry["runs"] = [str(s) for s in shared.tolist()[:10]]
                else:
                    entry["examples"] = [
                        f"cell {s.split(chr(0))[0]} in run {s.split(chr(0))[1]}"
                        for s in shared.tolist()[:10]
                    ]
                leaks.append(entry)
        diag["run_leaks"] = leaks
        diag["run_locality_ok"] = not leaks
    else:
        diag["run_locality_ok"] = None
    return folds, diag


def detect_run_nesting(
    factor_codes: dict[str, np.ndarray],
    run: np.ndarray | None,
) -> dict[str, dict[str, int]]:
    """Factors that never vary within a run -- i.e. that are *nested* in run.

    A factor whose level is fixed for a whole run is confounded with everything else that
    is fixed for a whole run: residual drift, motion regime, arousal, physiological state,
    coil warm-up, and the participant's position in the session. No analysis of these data
    can separate the two, because within the data they are the same regressor.

    Returns factor name -> {level: number of runs at that level}, empty when nothing is
    nested. The run count per level is the effective sample size for that factor's main
    effect: with three runs per task, the task effect rests on three independent
    observations, not on the hundreds of trials that carry it.
    """
    if run is None:
        return {}
    run_s = np.asarray(run).astype(str)
    uniq_runs = np.unique(run_s)
    out: dict[str, dict[str, int]] = {}
    for name, labels in factor_codes.items():
        lab = np.asarray(labels).astype(str)
        if any(np.unique(lab[run_s == r]).size > 1 for r in uniq_runs):
            continue
        out[name] = {str(lv): int(np.unique(run_s[lab == lv]).size) for lv in np.unique(lab)}
    return out


def _nesting_warning(nested: dict[str, dict[str, int]], factor_names: list[str]) -> str:
    """The paragraph a user needs to read before believing a run-nested factor's map."""
    lines = []
    for name, per_level in nested.items():
        other = [f for f in factor_names if f != name]
        other_name = other[0] if other else "the other factor"
        counts = sorted(set(per_level.values()))
        n_runs = counts[0] if len(counts) == 1 else min(counts)
        lines.append(
            f"'{name}' never varies within a run: it is NESTED in run, with "
            f"{n_runs} run(s) per level.\n"
            f"       * unique_{name} is confounded with every run-level effect (drift, "
            f"motion regime,\n"
            f"         arousal, session order). Nothing in these data can separate them.\n"
            f"       * It is ALSO biased downward: per-run polynomial/nuisance regressors "
            f"in the\n"
            f"         single-trial fit remove between-run variance, which is where this "
            f"factor's\n"
            f"         main effect lives. The two biases do not cancel and neither is "
            f"measurable.\n"
            f"       * What survives: unique_{other_name} and the interaction. An additive "
            f"per-run\n"
            f"         offset is constant across {other_name}, so it lands ENTIRELY in "
            f"'{name}'s main\n"
            f"         effect and contributes exactly zero to the other two. Read those.\n"
            f"       * Inference for unique_{name} switches to whole-run permutation, which "
            f"supplies\n"
            f"         the correct error term (between-run, n={n_runs} per level) but "
            f"cannot undo the\n"
            f"         confound. Counterbalance level order across the session, or accept "
            f"it as a limit."
        )
    return "\n   ".join(lines)


def cell_labels(design: FactorDesign) -> np.ndarray:
    """Per-trial ``"levelA|levelB|..."`` cell identity, for readable locality diagnostics."""
    codes = design.codes.numpy()
    levels = design.levels
    return np.array(["|".join(str(levels[f][c]) for f, c in enumerate(row)) for row in codes])


def _check_run_locality(fold_diag: dict, strict: bool, verbose: bool) -> None:
    """Warn (or refuse, under *strict*) when a cell's repeats share a run.

    Repeating a cell inside one run is a deliberate design choice in plenty of
    experiments, so this is not an error by default. What it costs is honesty about
    magnitude: run-level nuisance then sits on both sides of that cell's train/test
    split, inflating held-out R² and the noise ceiling. It inflates every band alike,
    so the partition *ratios* -- which is what the tool is for -- are far less affected
    than the absolute numbers. Pass ``strict_run_locality=True`` to make it fatal when
    you expected repeats to be spread across runs and want to be told they are not.
    """
    leaks = fold_diag.get("run_leaks")
    if not leaks:
        return

    total = sum(int(item.get("n_leaks", 0)) for item in leaks)
    detail = "; ".join(
        f"fold {item['fold']}: {item.get('n_leaks')} "
        f"({', '.join(item.get('examples', item.get('runs', []))[:3])}...)"
        for item in leaks[:4]
    )
    if strict:
        raise ValueError(
            f"fold construction leaks {total} (cell, run) pairs across the train/test "
            f"split: {detail}. A cell has two repeats in the same run, so run-level "
            "nuisance sits on both sides of that cell and inflates held-out R² and the "
            "noise ceiling. Drop -strict_run_locality if the repeats-within-run are "
            "deliberate; the partition ratios survive it, the absolute numbers are "
            "optimistic."
        )
    if verbose:
        print(
            f"   ⚠️  {total} (cell, run) pairs have repeats inside one run "
            f"({detail}). Held-out R² and ncsnr are inflated by shared run nuisance; "
            "the partition ratios are much less affected."
        )


def _solve_gammas_from_gram(gram: Tensor, rhs: Tensor) -> Tensor:
    """Clamped least-squares shrinkage from precomputed normal equations.

    Leading dimensions are free, so this serves both the per-model solve (one system per
    voxel) and the rank / nuclear sweeps (one system per voxel per variant).
    """
    n = gram.shape[-1]
    eye = torch.eye(n, dtype=gram.dtype, device=gram.device)
    # Jitter keeps the solve defined for voxels whose bands are identically zero -- which
    # is exactly the rank-0 variant, where the interaction column is structurally absent.
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    reg = gram + 1e-8 * scale[..., None, None] * eye
    gam = torch.linalg.solve(reg, rhs.unsqueeze(-1)).squeeze(-1)
    return gam.clamp(0.0, 1.0)


def _ss_res_from_gram(gram: Tensor, rhs: Tensor, sst: Tensor, gam: Tensor) -> Tensor:
    """Residual sum of squares of ``target - P @ gam``, from the normal equations.

    ``||t - P g||^2 = ||t||^2 - 2 g'(P't) + g'(P'P)g``. Uses the *unjittered* Gram, so the
    result is exact for the clamped gammas rather than for the regularised solve.
    """
    quad = torch.einsum("...b,...bc,...c->...", gam, gram, gam)
    return sst - 2.0 * (gam * rhs).sum(dim=-1) + quad


def _solve_gammas(
    partials: Tensor,  # (n_vox, n_test, n_bands)
    target: Tensor,  # (n_vox, n_test)
    active: list[int],
) -> Tensor:
    """Least-squares per-band shrinkage, clamped to [0, 1].

    Held-out prediction is linear in the gammas, so the optimum is a (n_active x n_active)
    normal-equation solve per voxel -- no grid search. n_active is at most 3 here.
    Clamping is exact whenever the unconstrained optimum is interior, which it is wherever
    the band carries signal; the boundary cases are voxels where a band is pure noise
    (gamma -> 0) or already unshrunk (gamma -> 1).
    """
    p = partials[:, :, active]
    gram = torch.einsum("vtb,vtc->vbc", p, p)
    rhs = torch.einsum("vtb,vt->vb", p, target)
    return _solve_gammas_from_gram(gram, rhs)


def _build_nested_fold_solvers(
    band_mats: dict[str, Tensor],
    band_order: list[str],
    folds: list[tuple[Tensor, Tensor]],
) -> list[list[tuple[Tensor, Tensor, Tensor]]]:
    """Precompute inner solvers for outer-fold gamma selection."""
    out: list[list[tuple[Tensor, Tensor, Tensor]]] = []
    for outer_i, (outer_train, _) in enumerate(folds):
        inner: list[tuple[Tensor, Tensor, Tensor]] = []
        for inner_i, (_, inner_val) in enumerate(folds):
            if inner_i == outer_i:
                continue
            inner_train = outer_train[~torch.isin(outer_train, inner_val)]
            if inner_train.numel() == 0 or inner_val.numel() == 0:
                continue
            x = torch.cat(
                [torch.ones(len(inner_train), 1, device=inner_train.device)]
                + [band_mats[n][inner_train] for n in band_order],
                dim=1,
            )
            inner.append((inner_train, inner_val, torch.linalg.pinv(x.double()).float()))
        if not inner:
            raise ValueError(
                "nested gamma selection needs at least 3 repeat folds; use "
                "nested_gamma=False to restore reporting-fold gamma selection"
            )
        out.append(inner)
    return out


def _nested_partials(
    y: Tensor,
    band_mats: dict[str, Tensor],
    band_order: list[str],
    band_slices: dict[str, slice],
    inner_ops: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Predictions made wholly inside one outer training set."""
    partial_blocks = []
    target_blocks = []
    for train, val, solver in inner_ops:
        coef = y[:, train] @ solver.T
        target_blocks.append(y[:, val] - coef[:, 0:1])
        block = torch.empty(y.shape[0], len(val), len(band_order), dtype=y.dtype, device=y.device)
        for bi, name in enumerate(band_order):
            sl = band_slices[name]
            cb = coef[:, 1 + sl.start : 1 + sl.stop]
            block[:, :, bi] = cb @ band_mats[name][val].T
        partial_blocks.append(block)
    return torch.cat(partial_blocks, dim=1), torch.cat(target_blocks, dim=1)


def _abs_cos(x: Tensor, y: Tensor) -> Tensor:
    """|cos| between corresponding rows. Sign-free, because singular vectors have no sign."""
    num = (x * y).sum(dim=-1).abs()
    den = x.norm(dim=-1) * y.norm(dim=-1)
    return num / den.clamp_min(1e-12)


# Layout of the sweep accumulator's last axis. The first five entries are fixed; the rest
# are the swept band's inner products against each OTHER band's partial, two per band.
_ACC_RR, _ACC_RS, _ACC_SS, _ACC_RT, _ACC_ST = range(5)
_ACC_FIXED = 5


def _acc_width(n_other: int) -> int:
    return _ACC_FIXED + 2 * n_other


def _sweep_systems(
    acc: Tensor,  # (..., _acc_width(n_other)) gathered sweep accumulators
    gm: Tensor,  # (n_vox, n_other, n_other) Gram of the un-swept bands, over all folds
    rh: Tensor,  # (n_vox, n_other) their right-hand side
    tau: Tensor,  # (...) soft-threshold, broadcastable against acc[..., 0]
    fold_dim: int | None,
) -> tuple[Tensor, Tensor]:
    """Assemble the full normal equations with ONE band replaced by a swept predictor.

    The swept predictor is ``R_r - tau * S_r``; *acc* carries the inner products of ``R_r``
    and ``S_r`` against each other, against every un-swept band's partial, and against the
    target. Everything else in the system is independent of the sweep, so it is passed in
    once as *gm* / *rh*. The swept band lands at the LAST index of the returned system.

    When *fold_dim* is given the swept terms are summed over it first, which is what lets
    each fold use its own prefix length ``r`` under one shared per-voxel threshold.
    """
    n_other = gm.shape[-1]
    qrr = acc[..., _ACC_RR]
    qrs = acc[..., _ACC_RS]
    qss = acc[..., _ACC_SS]
    qrt = acc[..., _ACC_RT]
    qst = acc[..., _ACC_ST]
    qrm = acc[..., _ACC_FIXED : _ACC_FIXED + n_other]
    qsm = acc[..., _ACC_FIXED + n_other :]

    c_jj = qrr - 2.0 * tau * qrs + tau * tau * qss
    c_jt = qrt - tau * qst
    c_jm = qrm - tau[..., None] * qsm
    if fold_dim is not None:
        c_jj = c_jj.sum(dim=fold_dim)
        c_jt = c_jt.sum(dim=fold_dim)
        c_jm = c_jm.sum(dim=fold_dim)

    # Broadcast the (sweep-invariant) block for the other bands across the variant axis.
    lead = c_jj.shape  # (n_vox, n_variants)
    pad = (1,) * (len(lead) - 1)
    n = n_other + 1
    gram = torch.zeros(*lead, n, n, dtype=c_jj.dtype, device=c_jj.device)
    gram[..., :n_other, :n_other] = gm.reshape((lead[0],) + pad + (n_other, n_other))
    gram[..., :n_other, n_other] = c_jm
    gram[..., n_other, :n_other] = c_jm
    gram[..., n_other, n_other] = c_jj

    rhs = torch.empty(*lead, n, dtype=c_jj.dtype, device=c_jj.device)
    rhs[..., :n_other] = rh.reshape((lead[0],) + pad + (n_other,))
    rhs[..., n_other] = c_jt
    return gram, rhs


def _r2_from_pred(y: Tensor, pred: Tensor) -> Tensor:
    """Coefficient of determination against the voxel's own mean over held-out trials."""
    ss_res = ((y - pred) ** 2).sum(dim=1)
    ss_tot = ((y - y.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _compute_ncsnr(betas: Tensor, cell_flat: Tensor, n_cells: int) -> tuple[Tensor, Tensor]:
    """NSD-style noise ceiling from repeat-to-repeat variability.

    Split-half with Spearman-Brown is unavailable at 3 repeats (there is no even split),
    so the variance-components estimator is the one that works at realistic repeat counts.
    """
    n_vox = betas.shape[0]
    dev, dt = betas.device, betas.dtype
    counts = torch.bincount(cell_flat, minlength=n_cells).to(dt)
    sums = torch.zeros(n_vox, n_cells, dtype=dt, device=dev)
    sums.index_add_(1, cell_flat, betas)
    means = sums / counts.clamp_min(1)
    resid = betas - means[:, cell_flat]

    # Within-cell variance, pooled with the right dof (sum of n_c - 1 over cells).
    dof = float((counts - 1).clamp_min(0).sum())
    noise_var = (resid**2).sum(dim=1) / max(dof, 1.0)
    total_var = betas.var(dim=1, unbiased=True)
    signal_var = (total_var - noise_var).clamp_min(0.0)
    ncsnr = torch.sqrt(signal_var) / torch.sqrt(noise_var.clamp_min(1e-12))

    n_rep = float(counts[counts > 0].float().mean())
    nc = ncsnr**2 / (ncsnr**2 + 1.0 / n_rep)
    return ncsnr, nc


def partition_variance(
    betas: Tensor | np.ndarray,
    factor_codes: dict[str, np.ndarray],
    repeat: np.ndarray | None = None,
    run: np.ndarray | None = None,
    max_rank: int | None = None,
    min_ncsnr_for_rank: float = 0.75,
    min_interaction_frac_ceiling: float = 0.02,
    n_nuclear_taus: int = 11,
    strict_run_locality: bool = False,
    nested_gamma: bool = True,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> VarPartResult:
    """Partition per-voxel variance across two crossed factors.

    Parameters
    ----------
    betas
        (n_voxels, n_trials) single-trial estimates. Column order must match the trial
        table row order. These are *consumed*, not estimated -- see ffs_ridge / GLMsingle.
    factor_codes
        Factor name -> per-trial labels. Exactly two factors.
    repeat
        Per-trial repeat index. Derived from cell-occurrence order when omitted.
    run
        Per-trial run label. Used only to verify fold locality; omitted means unchecked.
    max_rank
        Highest interaction rank to cross-validate. Defaults to the full rank of the
        interaction matrix.
    min_ncsnr_for_rank
        Noise-ceiling SNR below which the interaction rank is reported as ``-1``
        (undetermined) rather than ``0``. Rank selection misses real structure long
        before it invents any, so an unmasked map turns low-SNR tissue into a false
        "task-invariant" region. The 0.75 default is where a rank-1 interaction starts
        being missed >50% of the time on synthetic 20x21x3 data; raise it for a stricter
        map, set it to 0 to disable masking.
    min_interaction_frac_ceiling
        Detection floor for the interaction rank: the rank curve must improve on the
        additive model by at least this fraction of the voxel's noise ceiling before a
        nonzero rank is reported. Guards against reading a rank off a curve that is flat
        because the interaction band was shrunk away entirely. This is a detection
        threshold, not a significance test -- for that, run the permutation null on the
        ``interaction`` statistic.
    n_nuclear_taus
        Grid size for the singular-value soft-threshold (nuclear) sweep, which is the
        continuous counterpart of the hard rank sweep. Thresholds are fractions of each
        voxel's leading singular value, spanning 0 (full interaction) to 1 (none). Set to
        0 to skip it.

    nested_gamma
        Select a separate gamma for each outer fold using inner predictions made only
        from that outer fold's training trials. Enabled by default so held-out R2 is
        untouched by hyperparameter selection. Requires at least three repeat folds.
        Disable for the original faster reporting-fold procedure, whose R2 is mildly
        optimistic.

    Returns
    -------
    VarPartResult
        Partition maps, per-band shrinkage maps, interaction rank, noise ceiling, and the
        diagnostics that decide whether any of it is trustworthy.
    """
    if device is None:
        device = get_device()

    design = build_factor_design(factor_codes)
    n_trials = design.n_trials

    betas_t = torch.as_tensor(np.asarray(betas)) if not isinstance(betas, Tensor) else betas
    if betas_t.shape[1] != n_trials:
        raise ValueError(
            f"betas has {betas_t.shape[1]} trials but the factor table has {n_trials}; "
            "one row per volume is required, with excluded trials dropped from both."
        )
    betas_t = betas_t.to(torch.float32)
    n_vox = betas_t.shape[0]

    if repeat is None:
        repeat = derive_repeat_index(design)
    folds, fold_diag = build_repeat_folds(repeat, run, cell=cell_labels(design))
    _check_run_locality(fold_diag, strict=strict_run_locality, verbose=verbose)

    nested = detect_run_nesting(factor_codes, run)
    if nested and verbose:
        print(f"\n   ⚠️  RUN-NESTED FACTOR\n   {_nesting_warning(nested, design.factor_names)}")

    names = design.factor_names
    n_levels = [len(lv) for lv in design.levels]
    n_cells = int(np.prod(n_levels))

    band_order = design.band_order
    n_bands = len(band_order)
    main_bands = design.main_bands
    pair_bands = design.pair_bands
    band_mats = {k: v.to(device=device, dtype=torch.float32) for k, v in design.bands.items()}
    band_slices = {}
    off = 0
    for name in band_order:
        w = band_mats[name].shape[1]
        band_slices[name] = slice(off, off + w)
        off += w

    # Per-pair SVD geometry. A two-way interaction's coefficient matrix is (ia, ib), so its
    # rank is capped by the SMALLER factor -- a 2-level factor forces every interaction it
    # takes part in to rank <= 1, which is worth knowing before reading a rank map that can
    # only ever say 0 or 1. There, gain_align is the informative output, not rank.
    pair_dims = {}
    pair_rank = {}
    for name in pair_bands:
        f, g = design.band_members[name]
        ia, ib = n_levels[f] - 1, n_levels[g] - 1
        pair_dims[name] = (f, g, ia, ib)
        mr = min(ia, ib) if max_rank is None else int(min(max_rank, min(ia, ib)))
        pair_rank[name] = mr

    # Model set. Under exhaustive crossing every band is orthogonal to every other, so
    # "unique variance" is just each band's own contribution and there is nothing shared to
    # apportion -- which is why this generalises to any number of factors while classical
    # commonality analysis (2^k - 1 overlapping terms) does not. Each band's unique variance
    # is a drop-one comparison against the full model; each factor's is a drop-one against
    # the additive model, so the factors stay comparable to each other on main effects alone.
    models: dict[str, list[str]] = {"M0": [], "M_add": list(main_bands), "M_full": list(band_order)}
    for name in main_bands:
        models[f"M_{name}"] = [name]
        models[f"M_add-{name}"] = [n for n in main_bands if n != name]
    for name in band_order:
        models[f"M_full-{name}"] = [n for n in band_order if n != name]
    band_index = {name: i for i, name in enumerate(band_order)}

    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=n_trials,
            n_regressors=off + 1,
            device=device,
            operation="xval",
        )

    r2_out = {m: torch.zeros(n_vox) for m in models}
    gam_out = {b: torch.zeros(n_vox) for b in band_order}
    rank_r2_out = {p: torch.zeros(n_vox, pair_rank[p] + 1) for p in pair_bands}
    nuclear_r2_out = {p: torch.zeros(n_vox) for p in pair_bands}
    nuclear_tau_out = {p: torch.zeros(n_vox) for p in pair_bands}
    nuclear_gain_out = {p: torch.zeros(n_vox) for p in pair_bands}
    heldout_sst_out = torch.zeros(n_vox)
    # Alignment is per (pair band, parent factor): the left singular vector against one
    # parent's main effect, the right against the other's.
    align_out = {(p, side): torch.zeros(n_vox) for p in pair_bands for side in (0, 1)}

    n_folds = len(folds)
    n_taus = int(max(n_nuclear_taus, 0))
    tau_fracs = torch.linspace(0.0, 1.0, max(n_taus, 1), device=device)

    cell_flat = flat_cell_index(design).to(device)

    fold_t = [
        (
            torch.as_tensor(tr, dtype=torch.long, device=device),
            torch.as_tensor(te, dtype=torch.long, device=device),
        )
        for tr, te in folds
    ]
    nested_solvers = (
        _build_nested_fold_solvers(band_mats, band_order, fold_t) if nested_gamma else None
    )

    # Per-fold band pseudoinverses. Under leave-one-repeat-out on a balanced design the
    # training set keeps n-1 repeats of every cell, so it stays balanced and each band's
    # solve is a scaled transpose; pinv is used anyway so imbalance degrades gracefully
    # instead of silently returning the wrong coefficients.
    fold_solvers = []
    for tr, _ in fold_t:
        design_tr = torch.cat(
            [torch.ones(len(tr), 1, device=device)] + [band_mats[n][tr] for n in band_order], dim=1
        )
        fold_solvers.append(torch.linalg.pinv(design_tr.double()).float())

    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for c0 in tqdm(
        range(0, n_vox, chunk_size),
        total=n_chunks,
        desc="varpart",
        leave=True,
        disable=not verbose or n_chunks < 2,
    ):
        c1 = min(c0 + chunk_size, n_vox)
        y = betas_t[c0:c1].to(device)
        nvc = c1 - c0

        partials = torch.zeros(nvc, n_trials, n_bands, device=device)
        target = torch.zeros(nvc, n_trials, device=device)

        # Sweep accumulators, one set per two-way interaction band. For every prefix length
        # r of that band's singular expansion we keep the inner products of two running
        # partial sums,
        #     R_r = sum_{k<r} s_k c_k    (hard rank-r truncation)
        #     S_r = sum_{k<r} c_k        (the linear-in-tau correction)
        # against each other, against every OTHER band's partial, and against the target.
        # Every soft-threshold predictor is exactly ``R_r - tau * S_r`` with
        # r = #{s_k > tau}, because the singular values arrive sorted so the surviving set
        # is always a prefix. A handful of scalars per (fold, rank) therefore reconstruct
        # the exact normal equations for BOTH sweeps, at any threshold, without ever
        # materialising a (n_vox_chunk, n_trials, n_variants) prediction tensor -- the
        # peak-memory term the earlier implementation had to design around. The fold axis
        # is kept because a single per-voxel threshold selects a different prefix length in
        # each fold.
        n_other = n_bands - 1
        acc = {
            p: torch.zeros(nvc, n_folds, pair_rank[p] + 1, _acc_width(n_other), device=device)
            for p in pair_bands
        }
        sv_all = {
            p: torch.zeros(nvc, n_folds, max(pair_rank[p], 1), device=device) for p in pair_bands
        }
        align = {k: torch.zeros(nvc, device=device) for k in align_out}

        for fi, ((tr, te), solver) in enumerate(zip(fold_t, fold_solvers, strict=True)):
            coef = y[:, tr] @ solver.T  # (nvc, 1 + sum band widths)
            intercept = coef[:, 0:1]
            tgt_fold = y[:, te] - intercept
            target[:, te] = tgt_fold

            for name in band_order:
                sl = band_slices[name]
                cf = coef[:, 1 + sl.start : 1 + sl.stop]
                partials[:, te, band_index[name]] = cf @ band_mats[name][te].T

            for pname in pair_bands:
                f, g, ia, ib = pair_dims[pname]
                mr = pair_rank[pname]
                # rank(E) == rank(B_I) because the contrast bases are full column rank, so
                # the SVD runs on the (ia, ib) coefficient matrix rather than the
                # (n_a, n_b) cell-mean matrix -- same answer, smaller decomposition.
                sl = band_slices[pname]
                b_i = coef[:, 1 + sl.start : 1 + sl.stop].reshape(nvc, ia, ib)
                u, s, vh = torch.linalg.svd(b_i, full_matrices=False)
                sv_all[pname][:, fi, : min(mr, s.shape[1])] = s[:, :mr]

                # A multiplicative gain model, m = mu + a_s * (1 + g_t), leaves an
                # interaction a_s * g_t: rank 1, with its LEFT singular vector parallel to
                # the first parent's main effect. Generic rank-1 reorganisation points
                # somewhere else. One dot product per fold separates the two -- the
                # difference between "this factor rescales that profile" and "rewrites it".
                fname, gname = names[f], names[g]
                cf_f = coef[:, 1 + band_slices[fname].start : 1 + band_slices[fname].stop]
                cf_g = coef[:, 1 + band_slices[gname].start : 1 + band_slices[gname].stop]
                align[(pname, 0)] += _abs_cos(u[:, :, 0], cf_f)
                align[(pname, 1)] += _abs_cos(vh[:, 0, :], cf_g)

                f_te = band_mats[fname][te]
                g_te = band_mats[gname][te]
                other = [band_index[n] for n in band_order if n != pname]
                p_other = partials[:, te, :][:, :, other]  # (nvc, n_te, n_other)
                a_p = acc[pname]
                run_r = torch.zeros_like(tgt_fold)
                run_s = torch.zeros_like(tgt_fold)
                for k in range(mr):
                    # c_k[t] = (u_k . f_t)(v_k . g_t). An interaction band is the elementwise
                    # Kronecker product of its parents' bands by construction, so a rank-1
                    # term costs two (ia x n_test) products instead of one (ia*ib x n_test)
                    # product -- an order of magnitude less work per variant.
                    ck = (u[:, :, k] @ f_te.T) * (vh[:, k, :] @ g_te.T)
                    run_r = run_r + s[:, k : k + 1] * ck
                    run_s = run_s + ck
                    a_p[:, fi, k + 1, _ACC_RR] = (run_r * run_r).sum(dim=1)
                    a_p[:, fi, k + 1, _ACC_RS] = (run_r * run_s).sum(dim=1)
                    a_p[:, fi, k + 1, _ACC_SS] = (run_s * run_s).sum(dim=1)
                    a_p[:, fi, k + 1, _ACC_RT] = (run_r * tgt_fold).sum(dim=1)
                    a_p[:, fi, k + 1, _ACC_ST] = (run_s * tgt_fold).sum(dim=1)
                    a_p[:, fi, k + 1, _ACC_FIXED : _ACC_FIXED + n_other] = torch.einsum(
                        "vt,vtm->vm", run_r, p_other
                    )
                    a_p[:, fi, k + 1, _ACC_FIXED + n_other :] = torch.einsum(
                        "vt,vtm->vm", run_s, p_other
                    )
                del u, s, vh, b_i, run_r, run_s, p_other
            del coef

        nested_inner = (
            [
                _nested_partials(y, band_mats, band_order, band_slices, nested_solvers[fi])
                for fi in range(n_folds)
            ]
            if nested_gamma and nested_solvers is not None
            else None
        )

        gam_full = torch.zeros(nvc, n_bands, device=device)
        gam_full_fold: Tensor | None = None
        for mname, active_names in models.items():
            if not active_names:
                pred = torch.zeros_like(target)
            else:
                active = [band_index[n] for n in active_names]
                if nested_gamma:
                    assert nested_inner is not None
                    gam_folds = torch.stack(
                        [_solve_gammas(*nested_inner[fi], active) for fi in range(n_folds)],
                        dim=1,
                    )
                    pred = torch.zeros_like(target)
                    for fi, (_, te) in enumerate(fold_t):
                        pred[:, te] = torch.einsum(
                            "vtb,vb->vt", partials[:, te][:, :, active], gam_folds[:, fi]
                        )
                    if mname == "M_full":
                        gam_full_fold = gam_folds
                        gam_full = gam_folds.mean(dim=1)
                else:
                    gam = _solve_gammas(partials, target, active)
                    pred = torch.einsum("vtb,vb->vt", partials[:, :, active], gam)
                    if mname == "M_full":
                        gam_full = gam
                if mname == "M_full":
                    for k, n in enumerate(active_names):
                        gam_out[n][c0:c1] = gam_full[:, k].cpu()
            r2_out[mname][c0:c1] = _r2_from_pred(target, pred).cpu()

        sst = (target * target).sum(dim=1)
        ss_tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        heldout_sst_out[c0:c1] = ss_tot.cpu()

        for pname in pair_bands:
            mr = pair_rank[pname]
            other = [band_index[n] for n in band_order if n != pname]
            p_other = partials[:, :, other]
            gm = torch.einsum("vtb,vtc->vbc", p_other, p_other)
            rh = torch.einsum("vtb,vt->vb", p_other, target)
            # Reorder M_full's gammas to match the swept system, which puts the swept band
            # last.
            gam_s = torch.cat(
                [gam_full[:, other], gam_full[:, band_index[pname] : band_index[pname] + 1]], dim=1
            )
            gam_s = gam_s.unsqueeze(1)

            # Hard rank sweep, evaluated under M_full's gammas so the curve and the reported
            # models are the same predictor family: rank_r2[:, mr] IS R2(M_full), and
            # rank_r2[:, 0] is the full model with this band removed.
            #
            # The gammas are deliberately NOT re-fitted per rank. They are selected on
            # inner folds by default (or reporting folds in compatibility mode), then held
            # fixed across variants. Re-fitting would give the sweep a free scalar per
            # variant and additive voxels then climb to spurious rank 2-6. Holding them fixed
            # restores the one-sided error property the rank map is read under: a pure-noise
            # component adds variance at fixed weight, so it can only ever hurt, and every
            # selection error is a miss rather than an invention.
            if nested_gamma:
                assert gam_full_fold is not None
                rank_ss_res = torch.zeros(nvc, mr + 1, device=device)
                for fi, (_, te) in enumerate(fold_t):
                    po = partials[:, te][:, :, other]
                    gm_i = torch.einsum("vtb,vtc->vbc", po, po)
                    rh_i = torch.einsum("vtb,vt->vb", po, target[:, te])
                    gram_i, rhs_i = _sweep_systems(
                        acc[pname][:, fi], gm_i, rh_i, torch.zeros((), device=device), None
                    )
                    gf = gam_full_fold[:, fi]
                    gf = torch.cat(
                        [gf[:, other], gf[:, band_index[pname] : band_index[pname] + 1]],
                        dim=1,
                    )
                    rank_ss_res += _ss_res_from_gram(
                        gram_i, rhs_i, (target[:, te] ** 2).sum(dim=1)[:, None], gf[:, None]
                    )
                rank_r2_out[pname][c0:c1] = (1.0 - rank_ss_res / (ss_tot[:, None] + 1e-12)).cpu()
            else:
                gram, rhs = _sweep_systems(
                    acc[pname].sum(dim=1), gm, rh, torch.zeros((), device=device), None
                )
                rank_r2_out[pname][c0:c1] = (
                    1.0
                    - _ss_res_from_gram(gram, rhs, sst[:, None], gam_s) / (ss_tot[:, None] + 1e-12)
                ).cpu()

            if n_taus > 0 and mr > 0:
                # Nuclear (singular-value soft-threshold) sweep: a continuous relaxation of
                # the hard rank truncation. Hard truncation keeps a noisy singular value at
                # full size or discards it entirely; soft thresholding shrinks all of them,
                # which is the better-behaved estimator when the singular values are
                # themselves noisy. Thresholds are per-voxel fractions of that voxel's
                # leading singular value, so one grid spans every scale in the brain.
                sv = sv_all[pname]
                scale = sv[:, :, 0].mean(dim=1).clamp_min(1e-12)  # (nvc,)
                tau = tau_fracs[None, :] * scale[:, None]  # (nvc, n_taus)
                # Per fold, the surviving prefix length under this voxel's threshold.
                r_sel = (sv[:, :, None, :] > tau[:, None, :, None]).sum(dim=-1)  # (nvc,F,T)
                idx = r_sel.unsqueeze(-1).expand(-1, -1, -1, _acc_width(n_other))
                acc_sel = torch.gather(acc[pname], 2, idx)  # (nvc, F, T, w)
                if nested_gamma:
                    assert gam_full_fold is not None
                    nuclear_ss_res = torch.zeros(nvc, n_taus, device=device)
                    for fi, (_, te) in enumerate(fold_t):
                        po = partials[:, te][:, :, other]
                        gm_i = torch.einsum("vtb,vtc->vbc", po, po)
                        rh_i = torch.einsum("vtb,vt->vb", po, target[:, te])
                        gram_i, rhs_i = _sweep_systems(
                            acc_sel[:, fi], gm_i, rh_i, tau, fold_dim=None
                        )
                        gf = gam_full_fold[:, fi]
                        gf = torch.cat(
                            [gf[:, other], gf[:, band_index[pname] : band_index[pname] + 1]],
                            dim=1,
                        )
                        nuclear_ss_res += _ss_res_from_gram(
                            gram_i,
                            rhs_i,
                            (target[:, te] ** 2).sum(dim=1)[:, None],
                            gf[:, None],
                        )
                    r2_t = 1.0 - nuclear_ss_res / (ss_tot[:, None] + 1e-12)
                else:
                    gram_t, rhs_t = _sweep_systems(acc_sel, gm, rh, tau[:, None, :], fold_dim=1)
                    r2_t = 1.0 - _ss_res_from_gram(gram_t, rhs_t, sst[:, None], gam_s) / (
                        ss_tot[:, None] + 1e-12
                    )
                # Referenced to this family's own no-interaction endpoint (tau = s1, every
                # singular value thresholded away), not to R2(M_add), so the gain is >= 0 by
                # construction and is not contaminated by a gamma difference between models.
                gain_t = r2_t - r2_t[:, -1:]
                best_t = gain_t.max(dim=1, keepdim=True).values
                # Largest threshold reaching 95% of the best gain -- the mirror of the rank
                # sweep's smallest-rank rule, and it fails the same way without it: where the
                # interaction was shrunk away the curve is flat and a bare argmax returns
                # tau = 0 ("no shrinkage needed") for a voxel that has no interaction at all.
                ok = gain_t >= 0.95 * best_t
                sel = (n_taus - 1) - ok.flip(1).float().argmax(dim=1)
                nuclear_r2_out[pname][c0:c1] = r2_t.gather(1, sel[:, None]).squeeze(1).cpu()
                nuclear_tau_out[pname][c0:c1] = tau_fracs[sel].cpu()
                nuclear_gain_out[pname][c0:c1] = gain_t.gather(1, sel[:, None]).squeeze(1).cpu()
                del gain_t, best_t, ok, sel, r2_t, acc_sel, idx, r_sel
            del gm, rh, p_other

        for key, val in align.items():
            align_out[key][c0:c1] = (val / n_folds).cpu()

        del partials, target, acc, sv_all, y

    ncsnr, noise_ceiling = _compute_ncsnr(betas_t.to(device), cell_flat, n_cells)
    ncsnr = ncsnr.cpu()
    ceiling_cpu = noise_ceiling.cpu()
    obtainable = ceiling_cpu > NC_FLOOR_FOR_RATIO

    # Rank selection wants to be one-sided in its errors: below ncsnr ~0.75 a real rank-1
    # interaction collapses to rank 0, and that miss is acceptable, but an *invented* rank
    # is not -- it paints structure onto tissue that has none.
    #
    # A bare argmax over the curve does invent, for a reason that only shows up once the
    # curve is evaluated under fitted gammas: wherever the held-out data gives no support
    # for the interaction band at all, gamma_interaction shrinks to 0, every rank predicts
    # identically, and the curve is flat to within float32 noise. argmax over a flat curve
    # returns whichever rank the rounding favoured -- on synthetic additive data that put
    # 8/20 voxels at rank 2-6 off improvements of 1e-5. Two guards fix it:
    #
    #   parsimony  take the SMALLEST rank reaching 95% of the best improvement, not the
    #              argmax. A flat curve then collapses to 0, and a genuine rank-1 voxel
    #              whose curve wobbles upward afterwards still reads 1.
    #   detection  require the improvement to be worth something against what was
    #              obtainable here. An absolute R2 floor would be unfair to low-ceiling
    #              tissue, so the bar is a fraction of the noise ceiling -- the same
    #              normalisation the *_frac_ceiling maps use.
    #
    # rank_e_raw is this selection WITHOUT the ncsnr mask, so it isolates what that mask
    # removed -- it is not the naive argmax, which differs in three ways at once and would
    # diagnose nothing.
    rank_e_all: dict[str, Tensor] = {}
    rank_e_raw_all: dict[str, Tensor] = {}
    for pname in pair_bands:
        curve = rank_r2_out[pname]
        improvement = curve - curve[:, :1]
        best_gain = improvement.max(dim=1).values
        # The ceiling has to be clamped before dividing, and a clamped denominator would let
        # a meaningless gain clear the bar wherever the ceiling is essentially zero -- which
        # is exactly the tissue this guard exists to protect. Require an obtainable ceiling.
        detected = obtainable & (
            best_gain / ceiling_cpu.clamp_min(NC_FLOOR_FOR_RATIO) > min_interaction_frac_ceiling
        )
        parsimonious = (improvement >= 0.95 * best_gain[:, None]).float().argmax(dim=1)
        raw = torch.where(detected, parsimonious, torch.zeros_like(parsimonious))
        rank_e_raw_all[pname] = raw
        # The ncsnr floor is a *sensitivity* mask, not a detection one: unmasked, the map
        # prints "task-invariant" over exactly the low-SNR territory (white matter, dropout,
        # edges), which is a spatial artifact that reads as a finding. Those voxels are
        # marked -1 ("cannot tell here"), never 0.
        rank_e_all[pname] = torch.where(ncsnr >= min_ncsnr_for_rank, raw, torch.full_like(raw, -1))

        # Gain vs reorganisation is only a question where there is a rank-1-or-more
        # interaction to describe; below that the leading singular vector is fitting noise
        # and its alignment is uniformly distributed, which reads as structure on a map.
        # Same reasoning as the ncsnr floor on rank_E itself.
        has_interaction = (rank_e_all[pname] >= 1).to(torch.float32)
        for side in (0, 1):
            align_out[(pname, side)] = align_out[(pname, side)] * has_interaction

        # The nuclear sweep answers the same question by a different route, so it takes the
        # same detection gate: below it there is no interaction to describe, the gain is
        # zero and the threshold is "everything removed".
        nuclear_gain_out[pname] = torch.where(
            detected, nuclear_gain_out[pname], torch.zeros_like(nuclear_gain_out[pname])
        )
        nuclear_tau_out[pname] = torch.where(
            detected, nuclear_tau_out[pname], torch.ones_like(nuclear_tau_out[pname])
        )

    # Each factor's unique main-effect variance, and each band's unique contribution in the
    # context of the full model. Under exhaustive crossing these are the same idea applied
    # at two levels, and there is nothing "shared" to apportion between them.
    unique = {f: r2_out["M_add"] - r2_out[f"M_add-{f}"] for f in main_bands}
    band_unique = {b: r2_out["M_full"] - r2_out[f"M_full-{b}"] for b in band_order}
    interaction = r2_out["M_full"] - r2_out["M_add"]

    # Balance diagnostic, generalised: under orthogonality the factor-alone models add up to
    # the additive model exactly, so any residual is lost balance (or the gamma clamp -- see
    # the module docstring). At k = 2 this is the classical commonality C.
    shared = sum((r2_out[f"M_{f}"] for f in main_bands), torch.zeros(n_vox)) - r2_out["M_add"]

    # Preference is a two-way ratio and only defined for two factors. Both uniquenesses are
    # held-out R2 differences that go genuinely negative in noise, so a guard on
    # |denominator| is not enough: a denominator of -1e-3 passes it and flips the sign for no
    # reason, painting a confident "purely A-driven" over tissue where neither factor
    # explains anything. Require a POSITIVE denominator and an obtainable ceiling.
    preference = None
    interpretable = torch.ones(n_vox, dtype=torch.bool)
    if len(main_bands) == 2:
        u_a, u_b = unique[main_bands[0]], unique[main_bands[1]]
        denom = u_a + u_b
        interpretable = (denom > 1e-8) & obtainable
        preference = torch.where(interpretable, (u_b - u_a) / denom, torch.zeros_like(denom))

    diagnostics = {
        "balanced": design.balanced,
        "max_offdiag_gram": design.max_offdiag,
        "fast_path": design.balanced,
        "cells_total": int(design.cell_counts.numel()),
        "cells_empty": int((design.cell_counts == 0).sum()),
        "repeats_min": int(design.cell_counts.min()),
        "repeats_max": int(design.cell_counts.max()),
        "n_factors": len(main_bands),
        "n_levels": dict(zip(main_bands, n_levels, strict=True)),
        "bands": list(band_order),
        "max_rank_per_pair": dict(pair_rank),
        "shared_abs_median": float(shared.abs().median()),
        "nested_gamma": nested_gamma,
        "min_ncsnr_for_rank": min_ncsnr_for_rank,
        "min_interaction_frac_ceiling": min_interaction_frac_ceiling,
        "rank_undetermined_frac_per_pair": {
            p: float((rank_e_all[p] < 0).float().mean()) for p in pair_bands
        },
        "n_nuclear_taus": n_taus,
        "factors_nested_in_run": nested,
        "preference_uninterpretable_frac": float((~interpretable).float().mean()),
        **fold_diag,
    }
    if len(pair_bands) == 1:
        diagnostics["rank_undetermined_frac"] = diagnostics["rank_undetermined_frac_per_pair"][
            pair_bands[0]
        ]
    if not design.balanced:
        diagnostics["warning"] = (
            "design is not balanced: band orthogonality is broken, so the closed-form "
            "decoupling and the C~0 expectation no longer hold. Treat the partition as "
            "approximate and inspect shared variance."
        )

    pair_align = {
        p: {names[pair_dims[p][0]]: align_out[(p, 0)], names[pair_dims[p][1]]: align_out[(p, 1)]}
        for p in pair_bands
    }

    result = VarPartResult(
        r2=r2_out,
        unique=unique,
        band_unique=band_unique,
        shared=shared,
        interaction=interaction,
        gammas=gam_out,
        preference=preference,
        pair_rank_e=rank_e_all,
        pair_rank_e_raw=rank_e_raw_all,
        pair_rank_r2=rank_r2_out,
        pair_nuclear_tau=nuclear_tau_out if n_taus > 0 else {},
        pair_nuclear_gain=nuclear_gain_out if n_taus > 0 else {},
        pair_gain_alignment=pair_align,
        ncsnr=ncsnr,
        noise_ceiling=noise_ceiling.cpu(),
        heldout_sst=heldout_sst_out,
        diagnostics=diagnostics,
    )

    # Two-factor convenience aliases. With one interaction band there is no ambiguity about
    # which one "the" rank map refers to, so the flat fields stay populated and every
    # existing caller keeps working; above two factors they are empty and the pair_* dicts
    # are the interface.
    if len(pair_bands) == 1:
        only = pair_bands[0]
        result.rank_e = rank_e_all[only]
        result.rank_e_raw = rank_e_raw_all[only]
        result.rank_r2 = rank_r2_out[only]
        result.gain_alignment = pair_align[only]
        if n_taus > 0:
            result.nuclear_r2 = nuclear_r2_out[only]
            result.nuclear_tau = nuclear_tau_out[only]
            result.nuclear_gain = nuclear_gain_out[only]
    return result


# ---------------------------------------------------------------------------
# Permutation inference
#
# The statistic is a difference of cross-validated R2 with per-voxel hyperparameter
# selection. It has no tractable null distribution, so inference is by permutation.
#
# What to permute is the subtle part. Shuffling task labels across trials would destroy
# the very balance the partition depends on -- the permuted design would have unequal cell
# counts, band orthogonality would break, and the null statistic would be computed under a
# different estimator variance than the observed one. Permuting *level names* is a no-op
# (the model is invariant to level naming). So the labels stay fixed and the data moves:
# Freedman-Lane permutes the residuals of the reduced model and adds the reduced fit back,
# which nulls the effect under test while preserving everything else and leaving the design
# untouched.
#
#     statistic       reduced model       what the null destroys
#     unique(A)       intercept + B       any effect of A
#     unique(B)       intercept + A       any effect of B
#     interaction     intercept + A + B   the interaction, main effects preserved
#
# Exchangeability blocks are runs: single-trial beta noise carries run-level structure
# (shared drift, motion, noise PCs), so free permutation across runs is anticonservative.
#
# Gamma selection re-runs inside every permutation using the same nested or compatibility
# procedure as the observed statistic.
# ---------------------------------------------------------------------------


@dataclass
class _CellOps:
    """Cell-space operators for the balanced fast path.

    Every model here depends on the data only through per-cell means, so the whole
    computation compresses from n_trials to n_cells columns. The interaction projection is
    never formed directly: the four projections are complementary, so

        P_interaction = I - P_intercept - P_A - P_B

    which replaces a 380-column band with a subtraction. That is what makes a permutation
    null affordable -- the direct route costs roughly ten times more per permutation.
    """

    n_cells: int
    band_order: list[str]
    # Cell-space orthonormal basis per band. The highest-order band is stored as None and
    # obtained by complement instead -- see the class docstring.
    bases: dict[str, Tensor | None]
    complement: str
    cell_of_trial: Tensor  # (n_trials,)
    folds: list[tuple[Tensor, Tensor, Tensor, float]]
    n_trials: int


def _build_cell_ops(
    design: FactorDesign,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> _CellOps:
    n_levels = [len(lv) for lv in design.levels]
    n_cells = int(np.prod(n_levels))
    cell_of_trial = flat_cell_index(design).to(device)

    # Each cell's level on each factor. The flat index is row-major, so the last factor
    # varies fastest and decoding runs backwards.
    cell_codes: list[Tensor] = [torch.zeros(n_cells, dtype=torch.long, device=device)] * len(
        n_levels
    )
    rem = torch.arange(n_cells, device=device)
    for f in reversed(range(len(n_levels))):
        cell_codes[f] = rem % n_levels[f]
        rem = rem // n_levels[f]

    # Cell-level orthonormal bases. A band's columns are constant across every factor it
    # does not involve, so its cell-space norm picks up a sqrt of those factors' level
    # counts. The top-order band is left out: the projections are complementary, so
    # P_top = I - P_intercept - sum(P_other), which replaces the widest band with a
    # subtraction and is what makes a permutation null affordable.
    complement = design.band_order[-1]
    bases: dict[str, Tensor | None] = {}
    for name in design.band_order:
        if name == complement:
            bases[name] = None
            continue
        members = design.band_members[name]
        cols = design.contrasts[members[0]].to(device=device, dtype=torch.float32)[
            cell_codes[members[0]], :
        ]
        for f in members[1:]:
            nxt = design.contrasts[f].to(device=device, dtype=torch.float32)[cell_codes[f], :]
            cols = (cols.unsqueeze(2) * nxt.unsqueeze(1)).reshape(n_cells, -1)
        outside = float(np.prod([n_levels[f] for f in range(len(n_levels)) if f not in members]))
        bases[name] = (cols / outside**0.5).contiguous()

    fold_ops = []
    for train, test in folds:
        tr = torch.as_tensor(train, dtype=torch.long, device=device)
        te = torch.as_tensor(test, dtype=torch.long, device=device)
        tr_cells = cell_of_trial[tr]
        te_cells = cell_of_trial[te]
        if te_cells.numel() != n_cells or torch.unique(te_cells).numel() != n_cells:
            raise ValueError(
                "the cell-space fast path needs exactly one held-out trial per cell per "
                "fold; this design does not have equal repeats across all cells"
            )
        # Order the held-out trials by cell so column j always means cell j.
        order = torch.argsort(te_cells)
        counts = torch.bincount(tr_cells, minlength=n_cells)
        if int(counts.min()) != int(counts.max()):
            raise ValueError("unequal training repeats per cell; fast path unavailable")
        fold_ops.append((tr, tr_cells, te[order], float(counts[0])))

    return _CellOps(
        n_cells=n_cells,
        band_order=list(design.band_order),
        bases=bases,
        complement=complement,
        cell_of_trial=cell_of_trial,
        folds=fold_ops,
        n_trials=design.n_trials,
    )


def _cellspace_partials(y: Tensor, ops: _CellOps) -> tuple[Tensor, Tensor]:
    """Per-band held-out predictions and targets, concatenated over folds.

    Returns ``(partials, target)`` of shapes ``(n_vox, n_folds * n_cells, n_bands)`` and
    ``(n_vox, n_folds * n_cells)``, with bands in ``ops.band_order``.
    """
    n_vox = y.shape[0]
    nc = ops.n_cells
    n_folds = len(ops.folds)
    n_bands = len(ops.band_order)
    partials = torch.empty(n_vox, n_folds * nc, n_bands, device=y.device, dtype=y.dtype)
    target = torch.empty(n_vox, n_folds * nc, device=y.device, dtype=y.dtype)

    for fi, (tr, tr_cells, te_by_cell, n_rep_train) in enumerate(ops.folds):
        m = torch.zeros(n_vox, nc, device=y.device, dtype=y.dtype)
        m.index_add_(1, tr_cells, y[:, tr])
        m = m / n_rep_train
        mu = m.mean(dim=1, keepdim=True)
        sl = slice(fi * nc, (fi + 1) * nc)
        rest = m - mu
        for bi, name in enumerate(ops.band_order):
            basis = ops.bases[name]
            if basis is None:
                continue
            p = (m @ basis) @ basis.T
            partials[:, sl, bi] = p
            rest = rest - p
        partials[:, sl, ops.band_order.index(ops.complement)] = rest
        target[:, sl] = y[:, te_by_cell] - mu

    return partials, target


@dataclass
class _TrialOps:
    """Trial-space equivalent of :class:`_CellOps`, for designs that are not balanced.

    The cell-space engine compresses every model to per-cell means, which is only valid
    when each cell has the same number of repeats and every fold holds out exactly one of
    each. Unequal repeats -- the normal result of dropping trials, or of a design that
    simply did not run every cell the same number of times -- break both assumptions.

    This path fits the same nested models with an explicit pseudo-inverse per fold, which
    is what :func:`partition_variance` already does for the observed statistic. It costs
    roughly an order of magnitude more per permutation because the design is n_trials wide
    instead of n_cells, and the fold solvers depend only on the design, so they are built
    once and reused across every permutation.
    """

    band_mats: dict[str, Tensor]
    band_order: list[str]
    band_slices: dict[str, slice]
    folds: list[tuple[Tensor, Tensor, Tensor]]  # (train idx, test idx, solver)
    reduced_solvers: dict[tuple[int, ...], tuple[Tensor, Tensor]]
    n_trials: int
    cell_of_trial: Tensor
    n_cells: int


def _build_trial_ops(
    design: FactorDesign,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> _TrialOps:
    band_order = list(design.band_order)
    band_mats = {k: v.to(device=device, dtype=torch.float32) for k, v in design.bands.items()}
    band_slices: dict[str, slice] = {}
    off = 0
    for name in band_order:
        w = band_mats[name].shape[1]
        band_slices[name] = slice(off, off + w)
        off += w

    fold_ops = []
    for train, test in folds:
        tr = torch.as_tensor(train, dtype=torch.long, device=device)
        te = torch.as_tensor(test, dtype=torch.long, device=device)
        x_tr = torch.cat(
            [torch.ones(len(tr), 1, device=device)] + [band_mats[n][tr] for n in band_order], dim=1
        )
        fold_ops.append((tr, te, torch.linalg.pinv(x_tr.double()).float()))

    # Freedman-Lane reduced fits, on the whole dataset. Under imbalance a reduced model is
    # not a sub-vector of the full fit, so each one gets its own solve. Every statistic's
    # reduced model is "the full band set minus the effect under test", so those are the
    # only subsets that ever get asked for.
    reduced: dict[tuple[int, ...], tuple[Tensor, Tensor]] = {}
    for drop in range(len(band_order)):
        bands = tuple(i for i in range(len(band_order)) if i != drop)
        x = torch.cat(
            [torch.ones(design.n_trials, 1, device=device)]
            + [band_mats[band_order[i]] for i in bands],
            dim=1,
        )
        reduced[bands] = (x, torch.linalg.pinv(x.double()).float())
    for f in range(design.n_factors):
        involved = {band_order.index(n) for n in design.bands_involving(f)}
        bands = tuple(i for i in range(len(band_order)) if i not in involved)
        if bands in reduced:
            continue
        cols = [torch.ones(design.n_trials, 1, device=device)]
        cols += [band_mats[band_order[i]] for i in bands]
        x = torch.cat(cols, dim=1)
        reduced[bands] = (x, torch.linalg.pinv(x.double()).float())

    return _TrialOps(
        band_mats=band_mats,
        band_order=band_order,
        band_slices=band_slices,
        folds=fold_ops,
        reduced_solvers=reduced,
        n_trials=design.n_trials,
        cell_of_trial=flat_cell_index(design).to(device),
        n_cells=int(np.prod([len(lv) for lv in design.levels])),
    )


def _trialspace_partials(y: Tensor, ops: _TrialOps) -> tuple[Tensor, Tensor]:
    """Per-band held-out predictions and targets, in trial space.

    Same contract as :func:`_cellspace_partials` -- ``(n_vox, n_trials, n_bands)`` partials
    and ``(n_vox, n_trials)`` targets, in ``ops.band_order`` -- so :func:`_partition_stats`
    consumes either without knowing which engine produced it. Leave-one-repeat-out folds
    partition the trials, so every column gets written exactly once.
    """
    n_vox = y.shape[0]
    partials = torch.zeros(n_vox, ops.n_trials, len(ops.band_order), device=y.device, dtype=y.dtype)
    target = torch.zeros(n_vox, ops.n_trials, device=y.device, dtype=y.dtype)

    for tr, te, solver in ops.folds:
        coef = y[:, tr] @ solver.T
        target[:, te] = y[:, te] - coef[:, 0:1]
        for bi, name in enumerate(ops.band_order):
            sl = ops.band_slices[name]
            cb = coef[:, 1 + sl.start : 1 + sl.stop]
            partials[:, te, bi] = cb @ ops.band_mats[name][te].T

    return partials, target


def _reduced_fit_trials(y: Tensor, ops: _TrialOps, bands: tuple[int, ...]) -> Tensor:
    """Whole-dataset fitted values of a reduced model, per trial (see Freedman-Lane note)."""
    x, pinv = ops.reduced_solvers[tuple(sorted(bands))]
    return (y @ pinv.T) @ x.T


@dataclass
class _StatSpec:
    """One permutation statistic: which models make it, and what the null must destroy.

    ``full`` and ``reduced`` are band-index tuples; the statistic is
    ``R2(full) - R2(reduced)``, and Freedman-Lane permutes the residuals of ``reduced``.
    """

    full: tuple[int, ...]
    reduced: tuple[int, ...]
    factor: str | None  # set when the statistic is a factor's unique main-effect variance


def build_stat_specs(design: FactorDesign) -> dict[str, _StatSpec]:
    """Statistic registry for a k-factor design.

    Names are stable and explicit: ``unique_<factor>`` for a factor's main-effect variance
    (against the additive model, so factors stay comparable), ``band_<band>`` for a band's
    contribution to the full model, and ``interaction`` for every interaction band at once.
    The two-factor names ``unique_a`` / ``unique_b`` are kept as aliases so existing callers
    and saved scripts keep working.
    """
    order = list(design.band_order)
    idx = {n: i for i, n in enumerate(order)}
    mains = design.main_bands
    all_bands = tuple(range(len(order)))
    add = tuple(idx[n] for n in mains)

    specs: dict[str, _StatSpec] = {}
    for f, name in enumerate(mains):
        without = tuple(i for i in add if i != idx[name])
        specs[f"unique_{name}"] = _StatSpec(full=add, reduced=without, factor=name)
        if len(mains) == 2:
            specs["unique_a" if f == 0 else "unique_b"] = specs[f"unique_{name}"]
    for name in order:
        drop = tuple(i for i in all_bands if i != idx[name])
        specs[f"band_{name}"] = _StatSpec(full=all_bands, reduced=drop, factor=None)
    specs["interaction"] = _StatSpec(full=all_bands, reduced=add, factor=None)
    return specs


def _partition_stats(
    partials: Tensor,
    target: Tensor,
    specs: dict[str, _StatSpec],
    keys: tuple[str, ...] | None = None,
) -> dict[str, Tensor]:
    """Cross-validated R2 for every model a requested statistic needs, then the statistics.

    Only the models actually referenced get fitted, which matters inside a permutation loop:
    the full registry is 2^k + k models but any one null needs two of them.
    """
    wanted = tuple(specs) if keys is None else keys
    needed: set[tuple[int, ...]] = set()
    for key in wanted:
        needed.add(specs[key].full)
        needed.add(specs[key].reduced)

    r2: dict[tuple[int, ...], Tensor] = {}
    for active in needed:
        if not active:
            pred = torch.zeros_like(target)
        else:
            gam = _solve_gammas(partials, target, list(active))
            pred = torch.einsum("vtb,vb->vt", partials[:, :, list(active)], gam)
        r2[active] = _r2_from_pred(target, pred)

    out: dict[str, Tensor] = {}
    for key in wanted:
        spec = specs[key]
        out[key] = r2[spec.full] - r2[spec.reduced]
    return out


def _partition_stats_nested(
    y: Tensor,
    ops: _TrialOps,
    nested_solvers: list[list[tuple[Tensor, Tensor, Tensor]]],
    specs: dict[str, _StatSpec],
    keys: tuple[str, ...],
) -> dict[str, Tensor]:
    """Statistics with gamma selected strictly inside each outer training fold."""
    partials, target = _trialspace_partials(y, ops)
    nested_inner = [
        _nested_partials(y, ops.band_mats, ops.band_order, ops.band_slices, nested_solvers[fi])
        for fi in range(len(ops.folds))
    ]
    needed: set[tuple[int, ...]] = set()
    for key in keys:
        needed.update((specs[key].full, specs[key].reduced))

    r2: dict[tuple[int, ...], Tensor] = {}
    for active_t in needed:
        if not active_t:
            pred = torch.zeros_like(target)
        else:
            active = list(active_t)
            pred = torch.zeros_like(target)
            for fi, (_, te, _) in enumerate(ops.folds):
                gam = _solve_gammas(*nested_inner[fi], active)
                pred[:, te] = torch.einsum("vtb,vb->vt", partials[:, te][:, :, active], gam)
        r2[active_t] = _r2_from_pred(target, pred)

    return {key: r2[specs[key].full] - r2[specs[key].reduced] for key in keys}


def _reduced_fit_cells(y: Tensor, ops: _CellOps, bands: tuple[int, ...]) -> Tensor:
    """Fitted values of a reduced model, in cell space, using every repeat.

    Freedman-Lane needs the reduced fit on the *whole* dataset (not fold-local): the fold
    structure belongs to the statistic, not to the null.
    """
    n_vox = y.shape[0]
    m = torch.zeros(n_vox, ops.n_cells, device=y.device, dtype=y.dtype)
    m.index_add_(1, ops.cell_of_trial, y)
    counts = torch.bincount(ops.cell_of_trial, minlength=ops.n_cells).to(y.dtype)
    m = m / counts.clamp_min(1)
    mu = m.mean(dim=1, keepdim=True)
    fit = mu.expand(-1, ops.n_cells).clone()
    keep = {ops.band_order[i] for i in bands}
    if ops.complement in keep:
        # The top-order band has no stored basis, so build it by complement: everything
        # except the intercept and the bands that were left out.
        fit = m.clone()
        for name in ops.band_order:
            basis = ops.bases[name]
            if name in keep or basis is None:
                continue
            fit = fit - (m @ basis) @ basis.T
        return fit
    for name in keep:
        basis = ops.bases[name]
        assert basis is not None
        fit = fit + (m @ basis) @ basis.T
    return fit


def _within_block_permutation(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(blocks.shape[0])
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        perm[idx] = rng.permutation(idx)
    return perm


@dataclass
class _RunExchange:
    """Trial-slot correspondence across runs, for whole-run permutation.

    Within-run permutation cannot test a run-nested factor. A factor that is constant
    within a run contributes a component of the residual that is *also* constant within a
    run, and any vector constant on a block is exactly invariant under permutation within
    that block -- so every permuted dataset retains the full effect, the null lands on top
    of the observed statistic, and every p-value comes back at ~1. The test does not merely
    lose power; it stops being a test.

    The valid exchangeability unit for such a factor is the run itself. Whole-run
    permutation relabels runs and carries each run's trials to the matching slots of
    another run, matching on the level of the factor that *does* vary within a run plus
    its occurrence index. That destroys the nested factor's association with the data while
    preserving within-run structure, and it charges the statistic the correct error term:
    between-run variance, on as many runs as there are, not between-trial variance on
    hundreds of trials.

    It corrects the error term. It does not undo the confound -- if every level of the
    factor sat at a fixed position in the session, session-order effects are still
    indistinguishable from the factor, and no permutation scheme can help.
    """

    runs: list[str]
    run_of_trial: np.ndarray  # (n_trials,) index into runs
    slots: np.ndarray  # (n_runs, n_slots) trial index for each (run, slot)
    slot_of_trial: np.ndarray  # (n_trials,)


def build_run_exchange(run: np.ndarray, within_run_labels: np.ndarray) -> _RunExchange:
    """Match trial slots across runs so whole runs can be swapped. See :class:`_RunExchange`."""
    run_s = np.asarray(run).astype(str)
    lab = np.asarray(within_run_labels).astype(str)
    runs = [str(r) for r in np.unique(run_s)]
    run_index = {r: i for i, r in enumerate(runs)}

    key_of_trial = np.empty(len(run_s), dtype=object)
    per_run: list[dict] = [{} for _ in runs]
    for r in runs:
        seen: dict[str, int] = {}
        for t in np.flatnonzero(run_s == r):
            k = lab[t]
            c = seen.get(k, 0)
            seen[k] = c + 1
            key_of_trial[t] = (k, c)
            per_run[run_index[r]][(k, c)] = int(t)

    keysets = {frozenset(d) for d in per_run}
    if len(keysets) != 1:
        raise ValueError(
            "runs do not have matching trial structure, so whole-run permutation is not "
            "defined: at least one run holds a different multiset of the within-run "
            "factor's levels than another. Equalise the runs (or drop the odd ones) to "
            "test a run-nested factor."
        )
    keys = sorted(keysets.pop())
    slot_index = {k: i for i, k in enumerate(keys)}
    slots = np.array([[d[k] for k in keys] for d in per_run], dtype=np.int64)
    slot_of_trial = np.array([slot_index[k] for k in key_of_trial], dtype=np.int64)
    run_of = np.array([run_index[r] for r in run_s], dtype=np.int64)
    return _RunExchange(runs=runs, run_of_trial=run_of, slots=slots, slot_of_trial=slot_of_trial)


def _whole_run_permutation(ex: _RunExchange, rng: np.random.Generator) -> np.ndarray:
    """One draw: relabel the runs, then map every trial onto its slot in the new run."""
    sigma = rng.permutation(len(ex.runs))
    return ex.slots[sigma[ex.run_of_trial], ex.slot_of_trial]


@dataclass
class PermutationResult:
    """Observed statistics with uncorrected and FWE-corrected permutation p-values."""

    observed: dict[str, Tensor] = field(default_factory=dict)
    p_uncorrected: dict[str, Tensor] = field(default_factory=dict)
    p_fwe: dict[str, Tensor] = field(default_factory=dict)
    null_max: dict[str, Tensor] = field(default_factory=dict)
    n_perms: int = 0
    diagnostics: dict = field(default_factory=dict)


# Reduced model (as band indices) whose residuals get permuted to null each statistic.
# The reduced model per statistic is derived from the design now (see build_stat_specs),
# because at k factors "everything except the effect under test" is no longer a short list.


def permutation_test(
    betas: Tensor | np.ndarray,
    factor_codes: dict[str, np.ndarray],
    repeat: np.ndarray | None = None,
    run: np.ndarray | None = None,
    statistics: tuple[str, ...] = ("unique_a", "unique_b", "interaction"),
    n_perms: int = 1000,
    seed: int = 0,
    strict_run_locality: bool = False,
    nested_gamma: bool = True,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> PermutationResult:
    """Freedman-Lane permutation null for the partition statistics.

    Parameters
    ----------
    statistics
        Any of ``"unique_a"`` (unique variance of the first factor), ``"unique_b"``, and
        ``"interaction"``. Each gets its own null, because each has a different reduced
        model.
    n_perms
        Permutations per statistic. FWE p-values are floored at ``1/(n_perms + 1)``.
    run
        Exchangeability blocks. Omitting it permutes freely across all trials, which is
        anticonservative whenever run-level noise structure exists -- so it warns.

    Returns
    -------
    PermutationResult
        Observed statistic, uncorrected p, max-statistic FWE p, and the null max
        distribution, per requested statistic.
    """
    if device is None:
        device = get_device()

    design = build_factor_design(factor_codes)
    specs = build_stat_specs(design)
    bad = set(statistics) - set(specs)
    if bad:
        raise ValueError(f"unknown statistics {sorted(bad)}; expected {sorted(specs)}")

    betas_t = torch.as_tensor(np.asarray(betas)) if not isinstance(betas, Tensor) else betas
    if betas_t.shape[1] != design.n_trials:
        raise ValueError(
            f"betas has {betas_t.shape[1]} trials but the factor table has {design.n_trials}"
        )
    betas_t = betas_t.to(torch.float32)
    n_vox = betas_t.shape[0]

    if repeat is None:
        repeat = derive_repeat_index(design)
    folds, fold_diag = build_repeat_folds(repeat, run, cell=cell_labels(design))
    _check_run_locality(fold_diag, strict=strict_run_locality, verbose=verbose)

    # Balanced designs get the cell-space engine, which is what makes 200k voxels x 1000
    # permutations affordable. Imbalance (unequal repeats per cell, usually from dropped
    # trials) invalidates its compression, so fall back to the same pseudo-inverse engine
    # partition_variance uses for the observed statistic. Refusing instead would be
    # over-strict: what imbalance costs is the exact orthogonality of the bands, which
    # makes the *partition* approximate -- it does not make the null invalid, because the
    # null is computed by the identical estimator under the identical design.
    ops: _CellOps | _TrialOps
    if design.balanced and not nested_gamma:
        ops = _build_cell_ops(design, folds, device)
        partials_of = _cellspace_partials

        def reduced_fit(y: Tensor, bands: tuple[int, ...]) -> Tensor:
            cell_ops = ops
            assert isinstance(cell_ops, _CellOps)
            return _reduced_fit_cells(y, cell_ops, bands)[:, cell_ops.cell_of_trial]
    else:
        ops = _build_trial_ops(design, folds, device)
        partials_of = _trialspace_partials

        def reduced_fit(y: Tensor, bands: tuple[int, ...]) -> Tensor:
            trial_ops = ops
            assert isinstance(trial_ops, _TrialOps)
            return _reduced_fit_trials(y, trial_ops, bands)

        if verbose and not design.balanced:
            print(
                f"   ⚠️  design is not balanced (max off-diagonal Gram "
                f"{design.max_offdiag:.2e}); using the general trial-space permutation "
                "engine, ~10x slower per permutation. The partition itself is approximate "
                "under imbalance — check shared |C| before reading the p-values."
            )

    if isinstance(ops, _TrialOps):
        for spec in specs.values():
            bands = tuple(sorted(spec.reduced))
            if bands in ops.reduced_solvers:
                continue
            x = torch.cat(
                [torch.ones(design.n_trials, 1, device=device)]
                + [ops.band_mats[ops.band_order[i]] for i in bands],
                dim=1,
            )
            ops.reduced_solvers[bands] = (x, torch.linalg.pinv(x.double()).float())

    if run is None:
        blocks = np.zeros(design.n_trials, dtype=np.int64)
        import warnings

        warnings.warn(
            "no run labels: permuting freely across all trials. Single-trial beta noise "
            "carries run-level structure, so this is anticonservative. Pass run= to block.",
            stacklevel=2,
        )
    else:
        blocks = np.asarray(run)

    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=design.n_trials,
            n_regressors=ops.n_cells,
            device=device,
            operation="xval",
        )

    if nested_gamma:
        trial_ops = ops
        assert isinstance(trial_ops, _TrialOps)
        fold_pairs = [(tr, te) for tr, te, _ in trial_ops.folds]
        nested_perm_solvers = _build_nested_fold_solvers(
            trial_ops.band_mats, trial_ops.band_order, fold_pairs
        )

        def stats_of(y: Tensor, keys: tuple[str, ...]) -> dict[str, Tensor]:
            return _partition_stats_nested(y, trial_ops, nested_perm_solvers, specs, keys)

    else:

        def stats_of(y: Tensor, keys: tuple[str, ...]) -> dict[str, Tensor]:
            return _partition_stats(*partials_of(y, ops), specs, keys)

    # Observed statistics, computed by the same engine the permutations use so that the
    # comparison is exact rather than merely close.
    observed: dict[str, Tensor] = {k: torch.zeros(n_vox) for k in statistics}
    for c0 in range(0, n_vox, chunk_size):
        c1 = min(c0 + chunk_size, n_vox)
        chunk = betas_t[c0:c1].to(device)
        stats = stats_of(chunk, statistics)
        for key in statistics:
            observed[key][c0:c1] = stats[key].cpu()

    rng = np.random.default_rng(seed)

    # A factor nested in run needs the run, not the trial, as its exchangeability unit --
    # within-run permutation leaves a run-constant effect exactly untouched and returns
    # p ~ 1 for everything. See :class:`_RunExchange`.
    nested = detect_run_nesting(factor_codes, run)
    scheme_of = {
        key: ("whole_run" if specs[key].factor in nested else "within_run") for key in statistics
    }
    exchange: _RunExchange | None = None
    if "whole_run" in scheme_of.values():
        free = [f for f in design.factor_names if f not in nested]
        if not free:
            raise ValueError(
                "both factors are nested in run, so every run is a single cell: there is "
                "no within-run variation left to permute against and no valid null exists "
                "for either main effect. Only the interaction is testable, and only if a "
                "factor varies within a run."
            )
        assert run is not None  # nesting is only detected when run labels exist
        exchange = build_run_exchange(run, factor_codes[free[0]])
        if verbose:
            swapped = [k for k, v in scheme_of.items() if v == "whole_run"]
            print(
                f"   ⚠️  {', '.join(sorted(nested))} nested in run: null for "
                f"{', '.join(sorted(swapped))} switches to WHOLE-RUN permutation over "
                f"{len(exchange.runs)} runs. This gives the correct (between-run) error "
                "term; it cannot undo the confound with session-level effects."
            )

    result = PermutationResult(
        observed=observed,
        n_perms=n_perms,
        diagnostics={
            "n_blocks": int(np.unique(blocks).size),
            "blocked_by_run": run is not None,
            "balanced": design.balanced,
            "max_offdiag_gram": design.max_offdiag,
            "engine": "cell-space" if isinstance(ops, _CellOps) else "trial-space",
            "nested_gamma": nested_gamma,
            "factors_nested_in_run": nested,
            "permutation_scheme": scheme_of,
            **fold_diag,
        },
    )

    # All permutations are drawn up front and shared across chunks. Sharing across chunks
    # is required, not an optimisation: the max-statistic null is a maximum over voxels
    # *within* a permutation, so every chunk has to see the same shuffle.
    def _draw(scheme: str) -> Tensor:
        if scheme == "whole_run":
            assert exchange is not None
            rows = [_whole_run_permutation(exchange, rng) for _ in range(n_perms)]
        else:
            rows = [_within_block_permutation(blocks, rng) for _ in range(n_perms)]
        return torch.stack([torch.as_tensor(r, dtype=torch.long) for r in rows]).to(device)

    perms_for = {scheme: _draw(scheme) for scheme in sorted(set(scheme_of.values()))}

    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for key in statistics:
        reduced = specs[key].reduced
        perms = perms_for[scheme_of[key]]
        # Running max over voxels, kept on device so the inner loop never synchronises.
        null_max = torch.full((n_perms,), float("-inf"), device=device)
        count_ge = torch.zeros(n_vox, dtype=torch.int32, device=device)
        obs_dev = observed[key].to(device)

        # Chunks outer, permutations inner. The reduced fit and residuals do not depend on
        # the permutation, so this computes them once per chunk instead of once per
        # (chunk, permutation) -- and keeps the chunk resident on the device across all
        # permutations rather than re-uploading it n_perms times.
        with tqdm(
            total=n_chunks * n_perms,
            desc=f"perm[{key}]",
            leave=True,
            disable=not verbose,
        ) as bar:
            for c0 in range(0, n_vox, chunk_size):
                c1 = min(c0 + chunk_size, n_vox)
                chunk = betas_t[c0:c1].to(device)
                fit_trials = reduced_fit(chunk, reduced)
                resid = chunk - fit_trials
                obs_chunk = obs_dev[c0:c1]
                for p in range(n_perms):
                    y_star = fit_trials + resid[:, perms[p]]
                    stats = stats_of(y_star, (key,))
                    s = stats[key]
                    null_max[p] = torch.maximum(null_max[p], s.max())
                    count_ge[c0:c1] += (s >= obs_chunk).to(torch.int32)
                    bar.update(1)
                del chunk, fit_trials, resid

        null_max_cpu = null_max.cpu()
        obs = observed[key]
        result.p_uncorrected[key] = (count_ge.cpu().float() + 1.0) / (n_perms + 1)
        result.p_fwe[key] = ((null_max_cpu[None, :] >= obs[:, None]).sum(dim=1).float() + 1.0) / (
            n_perms + 1
        )
        result.null_max[key] = null_max_cpu

    return result


# ---------------------------------------------------------------------------
# ROI collapsing
# ---------------------------------------------------------------------------


# ── Classical ANOVA ───────────────────────────────────────────────────────────


@dataclass
class AnovaResult:
    """Per-voxel classical factorial ANOVA over the same bands as the partition.

    Every field except ``df`` and ``diagnostics`` is a ``(n_voxels,)`` tensor, keyed by
    band name where it is a dict.
    """

    ss: dict[str, Tensor] = field(default_factory=dict)  # Type-III sums of squares
    df: dict[str, int] = field(default_factory=dict)
    f: dict[str, Tensor] = field(default_factory=dict)
    p: dict[str, Tensor] = field(default_factory=dict)
    eta2: dict[str, Tensor] = field(default_factory=dict)  # SS_b / SS_total
    eta2_partial: dict[str, Tensor] = field(default_factory=dict)  # SS_b / (SS_b + SS_err)
    omega2: dict[str, Tensor] = field(default_factory=dict)
    r2_full: Tensor | None = None
    r2_full_adj: Tensor | None = None
    f_full: Tensor | None = None  # saturated model against pure error
    p_full: Tensor | None = None
    ss_total: Tensor | None = None
    ss_error: Tensor | None = None  # pure (within-cell) error
    ss_resid: Tensor | None = None  # full-model residual
    noise_ceiling: Tensor | None = None  # 1 - ss_error/ss_total
    df_error: int = 0
    df_resid: int = 0
    diagnostics: dict = field(default_factory=dict)


def _column_space(x: Tensor) -> tuple[Tensor, int]:
    """Orthonormal basis for the column space of ``x``, plus its numerical rank.

    Residual sums of squares come from ``||y||^2 - ||y Q||^2`` rather than from an explicit
    fit, which is both cheaper and stable when dropped trials leave the design rank
    deficient -- the normal case here, not an edge case.
    """
    u, s, _ = torch.linalg.svd(x.double(), full_matrices=False)
    tol = float(s.max()) * max(x.shape) * torch.finfo(torch.float64).eps if s.numel() else 0.0
    rank = int((s > tol).sum())
    return u[:, :rank].contiguous().float(), rank


def anova_partition(
    betas: Tensor | np.ndarray,
    factor_codes: dict[str, np.ndarray],
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> AnovaResult:
    """Classical in-sample factorial ANOVA on the same bands :func:`partition_variance` uses.

    This is the textbook analysis, offered as an *independent check* on the cross-validated
    partition and as a number that is easy to explain to a reader. It shares the design
    construction and nothing else: no folds, no shrinkage, no permutation. Sums of squares
    are Type III (each band's residual increase when it alone is dropped from the saturated
    model), which is what makes it well defined on the unbalanced, missing-cell designs that
    dropping trials produces. On a perfectly balanced design the bands are orthogonal and
    Type III coincides with the simple projection partition.

    The error term is **pure error**: the within-cell scatter of replicate trials. It does
    not assume the model is correct, and with a saturated model on complete cells it equals
    the full-model residual exactly (checked in ``diagnostics``). ``1 - ss_error/ss_total``
    is therefore an ANOVA-native noise ceiling, independent of the ``ncsnr`` estimator that
    :func:`partition_variance` reports.

    **Read the effect sizes knowing what inflates them.** In-sample R2 rises with degrees of
    freedom whether or not there is signal: a band with ``p_b`` columns explains about
    ``p_b / (n_trials - 1)`` of the variance under a pure-noise null. On a 20x21x3 design the
    interaction band has 380 df against 1260 trials, so it collects ~0.30 of the variance
    from nothing at all, and ``r2_full`` collects ~0.33. ``eta2`` carries that inflation in
    full; ``omega2`` subtracts its expectation and is the honest one to quote. Neither is a
    substitute for the cross-validated partition, which is the estimate that survives being
    shown new data.

    One consequence of Type III worth knowing before reading the table: the band ``eta2``
    values sum to ``r2_full`` only when the design is balanced. Under imbalance the bands
    overlap, each one's drop-one SS includes variance another band could also have claimed,
    and the sum overshoots. That overshoot is itself the imbalance diagnostic -- the ANOVA
    counterpart of the partition's ``shared`` map.

    Parameters
    ----------
    betas
        (n_voxels, n_trials) single-trial estimates, same convention as
        :func:`partition_variance`.
    factor_codes
        Factor name -> per-trial labels. Two or more factors.

    Returns
    -------
    AnovaResult
        Per-band SS/df/F/p and effect sizes, plus the model and pure-error terms.
    """
    from scipy.stats import f as f_dist

    if device is None:
        device = get_device()

    design = build_factor_design(factor_codes)
    n_trials = design.n_trials

    betas_t = torch.as_tensor(np.asarray(betas)) if not isinstance(betas, Tensor) else betas
    if betas_t.shape[1] != n_trials:
        raise ValueError(
            f"betas has {betas_t.shape[1]} trials but the factor table has {n_trials}; "
            "one row per volume is required, with excluded trials dropped from both."
        )
    betas_t = betas_t.to(torch.float32)
    n_vox = betas_t.shape[0]
    band_order = list(design.band_order)

    ones = torch.ones(n_trials, 1, dtype=torch.float64)
    band_cols = {n: design.bands[n].double() for n in band_order}
    q_full, rank_full = _column_space(torch.cat([ones] + [band_cols[n] for n in band_order], dim=1))
    q_drop: dict[str, Tensor] = {}
    df_band: dict[str, int] = {}
    for name in band_order:
        keep = [band_cols[n] for n in band_order if n != name]
        q, rank = _column_space(torch.cat([ones] + keep, dim=1))
        q_drop[name] = q.to(device)
        df_band[name] = rank_full - rank
    q_full = q_full.to(device)

    # Pure error: replicate scatter within each cell. Model-free, so it stays a valid F
    # denominator even where the saturated model is not the truth. Cells seen once
    # contribute nothing and cost no df.
    cell_flat = flat_cell_index(design).to(device)
    occupied, cell_of_trial = torch.unique(cell_flat, return_inverse=True)
    n_occupied = int(occupied.numel())
    counts = torch.bincount(cell_of_trial, minlength=n_occupied).clamp_min(1).float()
    df_error = n_trials - n_occupied
    df_resid = n_trials - rank_full
    if df_error <= 0:
        raise ValueError(
            "the ANOVA needs replicate trials for its pure-error term, but every cell "
            "occurs exactly once; there is nothing to test against"
        )

    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=n_trials,
            n_regressors=rank_full + 1,
            device=device,
            operation="glm",
        )

    ss_band = {n: torch.zeros(n_vox) for n in band_order}
    ss_total_out = torch.zeros(n_vox)
    ss_error_out = torch.zeros(n_vox)
    ss_resid_out = torch.zeros(n_vox)

    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for c0 in tqdm(
        range(0, n_vox, chunk_size),
        total=n_chunks,
        desc="anova",
        leave=True,
        disable=not verbose or n_chunks < 2,
    ):
        c1 = min(c0 + chunk_size, n_vox)
        y = betas_t[c0:c1].to(device)
        nvc = c1 - c0

        centred = y - y.mean(dim=1, keepdim=True)
        ss_total = (centred * centred).sum(dim=1)
        ss_total_out[c0:c1] = ss_total.cpu()

        cell_sum = torch.zeros(nvc, n_occupied, device=device)
        cell_sum.index_add_(1, cell_of_trial, y)
        cell_mean = cell_sum / counts
        resid_pure = y - cell_mean[:, cell_of_trial]
        ss_error_out[c0:c1] = (resid_pure * resid_pure).sum(dim=1).cpu()
        del cell_sum, cell_mean, resid_pure

        ss_y = (y * y).sum(dim=1)
        fit_full = ((y @ q_full) ** 2).sum(dim=1)
        ss_resid_full = ss_y - fit_full
        ss_resid_out[c0:c1] = ss_resid_full.cpu()
        for name in band_order:
            ss_resid_drop = ss_y - ((y @ q_drop[name]) ** 2).sum(dim=1)
            ss_band[name][c0:c1] = (ss_resid_drop - ss_resid_full).clamp_min(0).cpu()
        del y, centred, ss_y, fit_full, ss_resid_full

    ms_error = ss_error_out / df_error
    ss_model = (ss_total_out - ss_resid_out).clamp_min(0)
    df_model = max(rank_full - 1, 1)
    f_full = (ss_model / df_model) / ms_error.clamp_min(_SS_FLOOR)
    res = AnovaResult(
        df=df_band,
        r2_full=1.0 - ss_resid_out / ss_total_out.clamp_min(_SS_FLOOR),
        r2_full_adj=1.0
        - (ss_resid_out / max(df_resid, 1)) / (ss_total_out / (n_trials - 1)).clamp_min(_SS_FLOOR),
        f_full=f_full,
        p_full=torch.as_tensor(
            f_dist.sf(f_full.double().numpy(), df_model, df_error), dtype=torch.float32
        ),
        ss_total=ss_total_out,
        ss_error=ss_error_out,
        ss_resid=ss_resid_out,
        noise_ceiling=1.0 - ss_error_out / ss_total_out.clamp_min(_SS_FLOOR),
        df_error=df_error,
        df_resid=df_resid,
    )
    for name in band_order:
        ss_b = ss_band[name]
        db = max(df_band[name], 1)
        f_stat = (ss_b / db) / ms_error.clamp_min(_SS_FLOOR)
        res.ss[name] = ss_b
        res.f[name] = f_stat
        res.p[name] = torch.as_tensor(
            f_dist.sf(f_stat.double().numpy(), db, df_error), dtype=torch.float32
        )
        res.eta2[name] = ss_b / ss_total_out.clamp_min(_SS_FLOOR)
        res.eta2_partial[name] = ss_b / (ss_b + ss_error_out).clamp_min(_SS_FLOOR)
        # Omega-squared subtracts the variance a band of this width collects from noise
        # alone. Without it a 380-column interaction band reads as the dominant effect in
        # every voxel in the brain, white matter included.
        res.omega2[name] = (ss_b - db * ms_error) / (ss_total_out + ms_error).clamp_min(_SS_FLOOR)

    res.diagnostics = {
        "balanced": design.balanced,
        "max_offdiag_gram": design.max_offdiag,
        "n_trials": n_trials,
        "cells_occupied": n_occupied,
        "cells_total": int(np.prod([len(lv) for lv in design.levels])),
        "rank_full": rank_full,
        "df_error": df_error,
        "df_resid": df_resid,
        "saturated": rank_full == n_occupied,
        "df_null_r2_full": (rank_full - 1) / (n_trials - 1),
        "df_null_eta2": {n: df_band[n] / (n_trials - 1) for n in band_order},
        "df_model": df_model,
        "bands": band_order,
    }
    return res


def build_roi_weights(
    atlas: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Turn a parcellation into per-ROI voxel weights.

    Accepts either form an atlas usually ships as:

    * **3-D integer label map** — one ROI per distinct non-zero value. ROIs are
      disjoint, so collapsing is a grouped mean.
    * **4-D stack** — one volume per ROI, values used as weights. Binary masks and
      probabilistic/partial-volume maps both work, and ROIs may overlap.

    Returns ``(roi_index_or_weights, roi_ids, roi_sizes)``. For the 3-D case the first
    element is a ``(n_vox,)`` int array giving each voxel's ROI slot (``-1`` = unassigned),
    which lets the collapse run as an ``index_add`` instead of materialising an
    ``(n_rois, n_vox)`` matrix. For the 4-D case it is that ``(n_rois, n_vox)`` weight
    matrix, which is affordable because overlapping atlases carry far fewer ROIs.
    """
    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    if atlas.ndim == 3:
        flat = atlas.reshape(-1)
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        ids = [int(v) for v in np.unique(flat) if v != 0]
        if not ids:
            raise ValueError("atlas contains no non-zero labels")
        lookup = {v: i for i, v in enumerate(ids)}
        slot = np.full(flat.shape[0], -1, dtype=np.int64)
        for v, i in lookup.items():
            slot[flat == v] = i
        sizes = np.bincount(slot[slot >= 0], minlength=len(ids)).astype(np.float64)
        return slot, ids, sizes

    if atlas.ndim == 4:
        n_rois = atlas.shape[3]
        w = atlas.reshape(-1, n_rois).T.astype(np.float64)  # (n_rois, n_vox_full)
        if mask is not None:
            w = w[:, mask.reshape(-1)]
        w = np.clip(w, 0.0, None)
        sizes = w.sum(axis=1)
        if not np.any(sizes > 0):
            raise ValueError("4-D atlas has no ROI with positive weight inside the mask")
        ids = list(range(1, n_rois + 1))
        return w, ids, sizes

    raise ValueError(f"atlas must be 3-D (labels) or 4-D (per-ROI masks), got {atlas.ndim}-D")


def collapse_to_rois(
    betas: Tensor,
    roi_spec: np.ndarray,
    roi_sizes: np.ndarray,
    device: torch.device | None = None,
    chunk_size: int = 20000,
) -> Tensor:
    """Average single-trial betas within each ROI.

    Returns ``(n_rois, n_trials)``. Empty ROIs come back as zeros rather than NaN so a
    partial atlas cannot poison downstream reductions.

    Collapsing before partitioning is not only a speed lever. Averaging voxels raises the
    noise ceiling roughly as sqrt(n_voxels) for signal shared across the ROI, which
    directly relaxes the ncsnr floor that rank selection needs -- an interaction rank that
    is undetectable per voxel is often resolvable per parcel.
    """
    if device is None:
        device = betas.device
    n_trials = betas.shape[1]
    sizes = torch.as_tensor(roi_sizes, dtype=torch.float32, device=device).clamp_min(1e-12)

    if roi_spec.ndim == 1:
        # Disjoint labels: grouped sum via index_add, no (n_rois, n_vox) matrix.
        slot = torch.as_tensor(roi_spec, dtype=torch.long, device=device)
        n_rois = len(roi_sizes)
        out = torch.zeros(n_rois, n_trials, device=device, dtype=torch.float32)
        keep = slot >= 0
        out.index_add_(0, slot[keep], betas.to(device)[keep].float())
        return out / sizes[:, None]

    # Weighted (possibly overlapping) ROIs: chunk the voxel axis so the weight matrix
    # never has to be resident in full alongside the data.
    w_full = torch.as_tensor(roi_spec, dtype=torch.float32)
    n_rois, n_vox = w_full.shape
    out = torch.zeros(n_rois, n_trials, device=device, dtype=torch.float32)
    for c0 in range(0, n_vox, chunk_size):
        c1 = min(c0 + chunk_size, n_vox)
        out += w_full[:, c0:c1].to(device) @ betas[c0:c1].to(device).float()
    return out / sizes[:, None]


def paint_rois_to_voxels(
    values: Tensor | np.ndarray,
    roi_spec: np.ndarray,
    n_voxels: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Broadcast per-ROI values back onto voxels, for display.

    The result carries no more spatial information than the ROI table it came from --
    every voxel in a parcel gets the same number. It exists so parcel results can be
    rendered on a brain; do not read within-parcel structure into it, because there
    isn't any.

    Overlapping 4-D atlases are resolved by weighted average of the ROIs covering each
    voxel, which reduces to the exact ROI value wherever coverage is disjoint and binary.
    Voxels no ROI covers get *fill*.
    """
    vals = np.asarray(values.cpu() if isinstance(values, Tensor) else values, dtype=np.float64)

    if roi_spec.ndim == 1:
        out = np.full(n_voxels, fill, dtype=np.float32)
        assigned = roi_spec >= 0
        out[assigned] = vals[roi_spec[assigned]].astype(np.float32)
        return out

    w = np.asarray(roi_spec, dtype=np.float64)  # (n_rois, n_vox)
    denom = w.sum(axis=0)
    out = np.full(n_voxels, fill, dtype=np.float32)
    covered = denom > 0
    out[covered] = ((vals @ w)[covered] / denom[covered]).astype(np.float32)
    return out
