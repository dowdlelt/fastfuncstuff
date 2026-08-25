#!/usr/bin/env python
"""ffs_varpart - per-voxel variance partitioning for crossed factorial designs.

Consumes single-trial betas you already have (ffs_ridge, GLMsingle, anything) plus a
sidecar table describing each trial, and reports how much of the reliable response each
factor explains uniquely, what they share, and what lives in their interaction.

Method: ../fmri_wiki/concepts/Variance partitioning.md
Interface: ../fmri_wiki/software/ffs_varpart.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime

import numpy as np
import torch

try:
    from fastfuncstuff.cli_utils import parse_prefix, setup_device
    from fastfuncstuff.design.trial_table import (
        canonicalize_label,
        level_identifier,
        sanitize_levels,
    )
    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.stats.variance_partition import (
        anova_partition,
        build_roi_weights,
        collapse_to_rois,
        paint_rois_to_voxels,
        partition_variance,
        permutation_test,
    )
except ImportError as e:  # pragma: no cover - install-time guard
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    sys.exit(1)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


# Noise ceiling below which the *_frac_ceiling ratios are reported as 0: an oracle model
# could not reach 1% of the variance there, so the ratio divides noise by noise.
CEILING_FLOOR = 0.01


def _summarize_effects(
    effect_maps: dict[str, torch.Tensor],
    heldout_sst: torch.Tensor,
    noise_ceiling: torch.Tensor,
) -> list[dict[str, str | int | float]]:
    """Pool unit-wise CV-R2 effects without averaging ratios.

    Voxel mode treats voxels as units; atlas mode treats parcels as units. Each unit is
    weighted by its held-out total sum of squares, so pooling reconstructs the explained
    sum of squares over the complete mask rather than giving a flat voxel or parcel the
    same influence as a strongly varying one.
    """
    sst = heldout_sst.detach().cpu().to(torch.float64)
    ceiling = noise_ceiling.detach().cpu().to(torch.float64)
    finite_base = torch.isfinite(sst) & torch.isfinite(ceiling) & (sst > 0)
    reliable = finite_base & (ceiling > CEILING_FLOOR)
    obtainable_ss = (sst[reliable] * ceiling[reliable]).sum()

    rows: list[dict[str, str | int | float]] = []
    for effect, values in effect_maps.items():
        val = values.detach().cpu().to(torch.float64)
        finite = finite_base & torch.isfinite(val)
        effect_reliable = reliable & torch.isfinite(val)
        pooled = (
            (sst[finite] * val[finite]).sum() / sst[finite].sum()
            if finite.any()
            else torch.tensor(float("nan"))
        )
        pooled_frac = (
            (sst[effect_reliable] * val[effect_reliable]).sum() / obtainable_ss
            if effect_reliable.any() and obtainable_ss > 0
            else torch.tensor(float("nan"))
        )
        ratios = val[effect_reliable] / ceiling[effect_reliable]
        if ratios.numel():
            q25, median, q75 = torch.quantile(
                ratios, torch.tensor([0.25, 0.5, 0.75], dtype=ratios.dtype)
            )
            positive_frac = (val[effect_reliable] > 0).to(torch.float64).mean()
        else:
            q25 = median = q75 = positive_frac = torch.tensor(float("nan"))
        rows.append(
            {
                "effect": effect,
                "pooled_cv_r2": float(pooled),
                "pooled_frac_ceiling": float(pooled_frac),
                "median_frac_ceiling": float(median),
                "q25_frac_ceiling": float(q25),
                "q75_frac_ceiling": float(q75),
                "positive_frac_reliable": float(positive_frac),
                "n_units": int(finite.sum()),
                "n_reliable": int(effect_reliable.sum()),
            }
        )
    return rows


def _anova_rows(res, band_order: list[str]) -> list[dict[str, str | int | float]]:
    """Pool the per-unit ANOVA into one row per band, plus a saturated-model row.

    Pooling is on sums of squares, not on ratios: the effect sizes are recomputed from
    mask-wide SS totals so a flat voxel does not carry the same weight as a strongly
    varying one. That matches how ``_summarize_effects`` pools the cross-validated R2, so
    the two tables can be read side by side.
    """
    ss_tot = float(res.ss_total.sum())
    ss_err = float(res.ss_error.sum())
    ms_err = ss_err / res.df_error

    def row(effect, dfb, ss_b, per_unit_eta2, per_unit_omega2, per_unit_p):
        eta2 = ss_b / ss_tot if ss_tot > 0 else float("nan")
        omega2 = (ss_b - dfb * ms_err) / (ss_tot + ms_err) if ss_tot > 0 else float("nan")
        med = lambda t: float(t.detach().cpu().to(torch.float64).median())  # noqa: E731
        return {
            "effect": effect,
            "df": dfb,
            "pooled_eta2": eta2,
            "pooled_omega2": omega2,
            "pooled_F": (ss_b / dfb) / ms_err if ms_err > 0 else float("nan"),
            "median_eta2": med(per_unit_eta2),
            "median_omega2": med(per_unit_omega2),
            "frac_units_p05": float((per_unit_p < 0.05).to(torch.float64).mean()),
            # What a band this wide collects from noise alone: E[SS_b] = df_b * sigma^2
            # under the null, with sigma^2 estimated by the replicate scatter. This is the
            # term omega2 subtracts, printed next to eta2 so the inflation is visible
            # rather than argued. A df-only fraction would overstate it wherever other
            # bands already explain much of the variance.
            "eta2_null_expected": dfb * ms_err / ss_tot if ss_tot > 0 else float("nan"),
        }

    rows = [
        row(
            band,
            res.df[band],
            float(res.ss[band].sum()),
            res.eta2[band],
            res.omega2[band],
            res.p[band],
        )
        for band in band_order
    ]
    ss_model = float((res.ss_total - res.ss_resid).clamp_min(0).sum())
    omega2_full = (
        res.r2_full * res.ss_total - res.diagnostics["df_model"] * (res.ss_error / res.df_error)
    ) / (res.ss_total + res.ss_error / res.df_error).clamp_min(1e-12)
    rows.append(
        row(
            "r2_full",
            res.diagnostics["df_model"],
            ss_model,
            res.r2_full,
            omega2_full,
            res.p_full,
        )
    )
    return rows


def _dominance_rows(
    effect_maps: dict[str, torch.Tensor],
    heldout_sst: torch.Tensor,
    noise_ceiling: torch.Tensor,
) -> tuple[list[dict[str, str | int | float]], int, int]:
    """Count reliable units whose largest effect is each effect.

    Pooled variance answers "how much of the brain does this effect explain"; it does
    not answer "where does it win". A factor can carry a small share of the total and
    still be the dominant effect over a sizeable, coherent territory -- which is the
    thing a reader is actually looking for in the preference map. Returns the rows plus
    (units with a winner, reliable units); the gap between those two is units where
    every effect is negative, i.e. where nothing beat the unit's own mean.
    """
    sst = heldout_sst.detach().cpu().to(torch.float64)
    ceiling = noise_ceiling.detach().cpu().to(torch.float64)
    reliable = torch.isfinite(sst) & (sst > 0) & torch.isfinite(ceiling) & (ceiling > CEILING_FLOOR)
    names = list(effect_maps)
    if not names or not bool(reliable.any()):
        return [], 0, int(reliable.sum())

    stack = torch.stack([effect_maps[n].detach().cpu().to(torch.float64) for n in names])
    stack = torch.nan_to_num(stack, nan=float("-inf"))
    winner = stack.argmax(dim=0)
    # A unit where every effect is negative has no winner: the model did worse than the
    # unit's own mean there, and naming the least-bad effect "dominant" is reading noise.
    has_winner = reliable & (stack.max(dim=0).values > 0)
    total = int(has_winner.sum())
    rows: list[dict[str, str | int | float]] = []
    for i, name in enumerate(names):
        n = int((has_winner & (winner == i)).sum())
        rows.append(
            {
                "effect": name,
                "n_dominant": n,
                "frac_dominant": (n / total) if total else float("nan"),
            }
        )
    return rows, total, int(reliable.sum())


def _significance_rows(
    p_uncorrected: dict[str, torch.Tensor],
    p_fwe: dict[str, torch.Tensor],
    names: dict[str, str],
) -> list[dict[str, str | int | float]]:
    """Count units passing each threshold, per tested statistic.

    The p-maps already say which unit is significant; nobody can read a count off a
    volume, and with -atlas the count *is* the headline result ("33 of 400 parcels").
    ``names`` maps the permutation key to the map name it was written under.
    """
    rows: list[dict[str, str | int | float]] = []
    for key, fwe_t in p_fwe.items():
        unc = p_uncorrected[key].detach().cpu().to(torch.float64)
        fwe = fwe_t.detach().cpu().to(torch.float64)
        rows.append(
            {
                "effect": names.get(key, key),
                "n_units": int(fwe.numel()),
                "n_sig_unc_p05": int((unc < 0.05).sum()),
                "n_sig_fwe_p05": int((fwe < 0.05).sum()),
                "n_sig_fwe_p01": int((fwe < 0.01).sum()),
                "frac_sig_fwe_p05": float((fwe < 0.05).to(torch.float64).mean()),
                "min_p_fwe": float(fwe.min()),
                "min_p_unc": float(unc.min()),
            }
        )
    return rows


def _significant_unit_ids(
    p_fwe: torch.Tensor,
    effect: torch.Tensor,
    unit_ids: list,
    alpha: float = 0.05,
) -> list:
    """Unit ids passing FWE *alpha*, strongest effect first.

    Ordered by effect size rather than by p, because the max-statistic null saturates:
    at 1000 permutations every parcel well past the null shares p = 1/1001, and sorting
    on a tied column would hand back an arbitrary order.
    """
    fwe = p_fwe.detach().cpu().to(torch.float64)
    val = effect.detach().cpu().to(torch.float64)
    hits = torch.nonzero(fwe < alpha, as_tuple=False).flatten()
    order = torch.argsort(val[hits], descending=True)
    return [unit_ids[int(i)] for i in hits[order]]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_varpart",
        description="Variance partitioning for fully crossed factorial designs",
        formatter_class=_HelpFormatter,
        epilog="""
OUTPUTS
=======
  {prefix}.nii.gz         one sub-brick per measure below (parcel-painted with -atlas)
  {prefix}_roi.tsv        with -atlas: the same measures, one row per ROI
  {prefix}_summary.tsv    pooled whole-mask variance summary, one row per effect
  {prefix}_anova.tsv      with -anova: classical Type-III ANOVA, one row per band
  {prefix}_significance.tsv     with -perm: significant-unit counts per effect
  {prefix}_significant_rois.tsv with -perm -atlas: every ROI at pFWE < 0.05
  {prefix}_varpart.json   factors, level-name mapping, dropped trials, diagnostics,
                          and the significance counts

WHAT THE SUMMARY BLOCKS SAY
===========================
Two tables are printed at the end and written alongside the maps, because the
question "so how much of the brain does task explain" is not answerable by eye
from a volume.

Overall variance summary -- one row per effect, over the whole mask:
  CV-R2       Pooled held-out R2: the explained sum of squares summed over units,
              divided by the total. Units are weighted by their own held-out SST,
              so a flat voxel does not count as much as a strongly varying one.
              This is a total, NOT the average of the per-voxel R2 values.
  obtainable  The same pooled quantity as a fraction of what the noise ceiling
              ever allowed. "task explains 12% of the obtainable signal."
  median [IQR]  The per-unit distribution of effect/noise_ceiling, over reliable
              units only. Pooled and median disagree when the effect is
              concentrated: a strong effect in a few high-variance units gives a
              large pooled value and a small median.
  positive    Fraction of reliable units where the effect is above 0 at all.
  dominant    Fraction of reliable units where this effect is the LARGEST of the
              partition pieces. This is the "territory" number -- an effect can
              carry a small share of total variance and still win in a large,
              coherent region. Blank for `shared` and `r2_full`, which do not
              compete. Units where every piece is negative have no winner and are
              excluded from the denominator (the counts are in the reported line).
  partition check   The pieces summed against r2_full. They are not required to
              match exactly -- the per-band shrinkage is clamped to [0, 1] -- but
              a large delta means read the pieces as an ordering rather than as a
              decomposition. Under a balanced crossed design it is ~0.

Significance (with -perm) -- one row per tested statistic:
  p<.05 unc   Units passing the uncorrected null. Expect ~5% of units by chance;
              compare the count against that before reading anything into it.
  pFWE<.05    Units passing the max-statistic family-wise null. With -atlas this
  pFWE<.01    is the number to report: "33 of 400 parcels, pFWE < 0.05".
  min pFWE    The smallest p any unit reached. It cannot go below 1/(N+1), so with
              too few permutations a column of zeros means "not enough
              permutations", not "no effect" -- the tool says so when that bites.
With -atlas the significant ROI ids are listed per effect, strongest effect first
(NOT smallest p first: the max-stat null saturates and ties every strong parcel at
the same floor), and written to {prefix}_significant_rois.tsv with their values.

Everything is computed on HELD-OUT repeats (leave-one-repeat-out CV). "Explains"
always means predicted out of sample, never fitted. Below, A and B are your two
-factors, in the order you gave them.

Band shrinkage is selected inside each outer fold's training data by default,
so the reporting repeat is untouched by coefficient OR hyperparameter fitting.
This costs roughly 3x for the observed maps. -no_nested_gamma restores the
original faster reporting-fold selection; its raw R2 is mildly optimistic.
Permutation always repeats the selected mode and may cost more than 3x because
nested selection needs the general trial-space engine.


TWO FACTORS OR MORE? The names below describe the two-factor case, which is what
most of the maps are about. -factors accepts any number; see "Three or more
factors" near the bottom for what changes.

--- The R2 family: fraction of held-out variance explained ------------------
R2 = 1 - SS_residual/SS_total, NOT a squared correlation. 1.0 = perfect
prediction, 0.0 = no better than the unit's own mean, NEGATIVE = worse than the
mean (routine in noise -- it is a real value, not a bug).

  r2_additive   R2 of the model A + B, no interaction term.
  r2_full       R2 of the saturated model A + B + A:B. Not guaranteed above
                r2_additive: the interaction band is estimated from ~2 trials per
                cell, so when there is no interaction its shrinkage goes to 0 and
                the two models coincide.

--- The partition: same R2 units, so these add up ---------------------------
  unique_A      R2(A+B) - R2(B alone). Variance only A can explain.
  unique_B      R2(A+B) - R2(A alone).
  shared        R2(A) + R2(B) - R2(A+B). Under exhaustive crossing this is ~0 BY
                CONSTRUCTION -- the factors are orthogonal, so there is nothing to
                share. Read it as a BALANCE CHECK, not a result: a nonzero value
                means trials were dropped or censored unevenly.
  interaction   r2_full - r2_additive. Response specific to a particular (A, B)
                combination, beyond what A and B contribute separately.
  preference    (unique_B - unique_A) / (unique_B + unique_A). Dimensionless,
                -1 = purely A-driven, +1 = purely B-driven, 0 = equal. Being a
                RATIO it is largely insensitive to how reliable the voxel is, so
                it survives low SNR far better than the raw R2 maps. Usually the
                map to look at first. Reported as 0 where the denominator is not
                POSITIVE (both uniquenesses go negative in noise, and a denominator
                of -1e-3 would otherwise flip the sign for no reason) or where the
                noise ceiling is below 0.01.

Note on "adds up": these are CROSS-VALIDATED commonality measures, not classical
variance components. Uniquenesses can be negative, and the four pieces need not
sum exactly to r2_full -- the per-band shrinkage is clamped to [0, 1], and a band
that hits a boundary in one nested model but not another breaks exact additivity.
That clamp, not lost balance, is the usual source of a small nonzero `shared`.

--- Fraction of the OBTAINABLE variance -------------------------------------
Held-out R2 is capped by noise_ceiling, not by 1, and that ceiling swings wildly
across the brain. These divide by it, turning "how much variance" into "how much
of what was ever gettable here":

  unique_A_frac_ceiling      unique_A / noise_ceiling
  unique_B_frac_ceiling      unique_B / noise_ceiling
  interaction_frac_ceiling   interaction / noise_ceiling
  r2_full_frac_ceiling       r2_full / noise_ceiling -- "how close to the best
                             anyone could do is this model, here?"

  1.0 = the whole obtainable signal. 0.4 = 40% of it, whether the raw R2 was 0.02
  or 0.5. Reported as 0 wherever noise_ceiling <= 0.01, since dividing by a
  ceiling that low is noise over noise. Values slightly ABOVE 1 happen and are not
  a bug: the ceiling is itself estimated from a handful of repeats, so it lands
  low in some units.

--- Interaction structure ---------------------------------------------------
  rank_E        Cross-validated rank of the interaction matrix E (the cell means
                with the additive part stripped out). 0 = additive, nothing beyond
                A + B. 1 = ONE pattern, scaled: e.g. every level of B rescales the
                same A-profile (a gain change). >1 = REORGANIZATION: different
                levels of B reshape the A-profile in genuinely different ways.
                -1 = UNDETERMINED (ncsnr below -min_ncsnr_for_rank). -1 means
                "cannot tell here", never "no interaction here".
  rank_E_raw    The naive argmax of the rank curve, with neither the SNR mask nor
                the detection floor applied. Compare against rank_E to see what the
                guards removed; do not interpret it on its own.
  gain_align_A  |cos| in 0..1 between the leading singular vector of the interaction
  gain_align_B  and that factor's own main effect. This is what separates the two
                ways a rank-1 interaction can arise. A pure multiplicative GAIN,
                m = mu + a_s*(1 + g_t), leaves an interaction a_s*g_t whose left
                singular vector is PARALLEL to the A main effect -- so
                gain_align_A near 1 means "B rescales A's response profile". Near 0
                means REORGANISATION: the interaction pattern is unrelated to the
                main effect, i.e. B rewrites the profile rather than scaling it.
                Zeroed where rank_E < 1, since below that the leading singular
                vector is fitting noise and its alignment is uniform.
  interaction_nuclear
                Interaction strength under a singular-value SOFT threshold instead
                of a hard rank cut -- the continuous version of rank_E, and the
                better-behaved estimator when the singular values are themselves
                noisy. Same R2 units as `interaction`, >= 0 by construction.
  nuclear_tau   The selected threshold, as a fraction of the voxel's leading
                singular value. 0 = no shrinkage needed (strong, clean
                interaction), 1 = everything thresholded away (no interaction).
                A diagnostic for interaction_nuclear, like the gamma maps.

Both rank_E and interaction_nuclear are chosen by maximising held-out R2 over a
grid, on the same folds they are reported on, so they carry SELECTION optimism:
read them as an ordering and a structure description, not as unbiased variance.
The permutation null on `interaction` is the significance test.

--- Reliability: properties of the DATA, not of the model -------------------
  ncsnr         Noise-ceiling SNR = sd(signal) / sd(noise), estimated from
                repeat-to-repeat variability of the same cell. Dimensionless,
                >= 0. 0 = no reliable signal at all; 1 = signal and noise are the
                same size; >1.5 = strong. Low ncsnr caps everything else.
  noise_ceiling ncsnr^2 / (ncsnr^2 + 1/n_repeats), in 0..1. The R2 a PERFECT model
                would get on this unit given its trial-to-trial noise and this
                many repeats -- the best anyone could do. Always read r2_full
                against it: r2_full = 0.10 is poor against a ceiling of 0.80 and
                essentially perfect against a ceiling of 0.12.

--- Shrinkage: diagnostic, not a result -------------------------------------
  gamma_A       Per-band shrinkage in 0..1, gamma = n/(n + lambda), fitted per
  gamma_B       unit. 1 = that band kept at full strength (held-out data supports
  gamma_inter.. it), 0 = shrunk away entirely (no out-of-sample evidence for it).
                gamma_interaction near 0 over most of the brain is the EXPECTED
                picture at 3 repeats, not a failure. Useful for asking "did this
                band get used at all here?" before believing its R2.

--- Inference (only with -perm N) -------------------------------------------
  oneminusp_unc_<stat>   1 - uncorrected p, per unit.
  oneminusp_fwe_<stat>   1 - family-wise p, from the max-statistic null across
                         all units (voxels, or ROIs with -atlas).
                Stored as 1 - p so SIGNIFICANT IS THE HIGH END: threshold at 0.95
                for p < 0.05, 0.99 for p < 0.01. The null is Freedman-Lane on the
                reduced model's residuals, permuted within run blocks, with the
                per-band shrinkage re-fitted inside every permutation so the null
                absorbs the selection optimism.

--- Three or more factors ---------------------------------------------------
-factors takes any number of columns. Exhaustive crossing makes every band
orthogonal to every other, so the partition gets SIMPLER with more factors, not
harder: there is nothing shared to apportion, and each effect is just its own
band's contribution. With k factors there are 2^k - 1 bands -- k main effects,
every two-way interaction, and every higher-order term.

What changes in the outputs:

  band_<band>   Each band's unique contribution to the full model, in R2 units.
                Written for k > 2 only (with two factors the single interaction
                band's value IS `interaction`). ':' becomes '_x_' in map names,
                so the stim x task band is `band_stim_x_task`.
  rank_E_<pair> The interaction-structure maps are now PER two-way interaction,
  gain_align_*  suffixed with the pair. Higher-order bands get variance and a
                gamma but no structure: their coefficients form a tensor, and
                rank there needs a CP/Tucker decomposition rather than an SVD.
  preference    Not written. It is a two-way ratio and has no k-way meaning.

Why bother splitting a factor out rather than folding it into the levels of
another? Because collapsed, two different effects are indistinguishable. If
16 stimuli x 2 noise levels are coded as 32 "stimuli", then "noise level rescales
the stimulus profile" and "task reorganises the stimulus profile" both read as
one big stimulus x task interaction. Split out, the first lands in stim:noise and
the second in stim:task, and you can see which is which.

The one thing you must NOT do is average over a factor you care about. Two trials
that differ in noise level but share a cell label make that difference WITHIN-cell
variance -- which is exactly what ncsnr calls noise. The ceiling drops, and every
*_frac_ceiling map inflates. Either give the factor its own column or bake it into
the level name; never let it collapse into repeats.

Note that a 2-level factor has ONE contrast column, so every interaction it takes
part in is rank <= 1 by construction. rank_E carries no information there -- it
can only say 0 or 1 -- and gain_align is the map that distinguishes "this factor
scales the other's profile" from "it reshapes it".

--- If a factor is locked to run (READ THIS) --------------------------------
A factor that never varies within a run -- one task per run, say -- is NESTED in
run, and ffs_varpart says so loudly rather than refusing. What it means:

  * unique_<that factor> is confounded with EVERY run-level effect: residual
    drift, motion regime, arousal, physiological state, position in the session.
    Within these data the factor and "which run this was" are the same regressor,
    and no analysis can separate them. Cross-validation does not help -- it will
    happily confirm a stable run-level trend as reproducible "task" signal.
  * It is also biased the OTHER way: per-run polynomial/nuisance regressors in the
    single-trial fit remove between-run variance, which is exactly where a
    run-locked factor's main effect lives. Both biases are present, they do not
    cancel, and neither is measurable from the data.
  * What survives intact: the OTHER factor's unique variance, and the whole
    interaction family (interaction, rank_E, gain_align_*). An additive per-run
    offset is constant across the within-run factor, so it lands entirely in the
    nested factor's main effect and contributes exactly ZERO to those. The
    interaction analysis is the part of this tool that stays trustworthy.
  * Inference switches automatically: the null for the nested factor becomes
    WHOLE-RUN permutation, which supplies the correct error term (between-run,
    n = runs per level). Within-run permutation is not merely weak here, it is
    powerless -- a run-constant effect is exactly invariant under shuffling inside
    a run, so it survives into every permuted dataset and the test detects nothing.

Two things fix this at the scanner, not in software: counterbalance run ORDER
across levels so session trends do not align with the factor, and randomise the
within-run trial order so within-run drift does not alias onto the other factor.

Examples:
  # 21 tasks x 20 stimuli, 3 repeats, single-trial betas from ffs_ridge
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -mask brain.nii.gz -prefix vp

  # Parcel-level: far faster, and a higher noise ceiling resolves ranks that
  # per-voxel data cannot
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -atlas schaefer400.nii.gz -prefix vp_roi

  # With permutation inference (slow at voxel level; pair it with -atlas)
  ffs_varpart -betas trials.nii.gz -trials trials.csv -factors stim,task \\
              -mask brain.nii.gz -perm 1000 -prefix vp

  # Three factors: 16 stimuli x 2 noise levels x 3 tasks. Splits "noise rescales
  # the stimulus profile" (stim:noise) from "task reorganises it" (stim:task)
  ffs_varpart -betas trials.nii.gz -trials trials.tsv \\
              -factors stim,task,noise -mask brain.nii.gz -prefix vp3

  # Test one specific band
  ffs_varpart -betas trials.nii.gz -trials trials.tsv -factors stim,task,noise \\
              -perm 1000 -perm_stats band_stim:noise -atlas schaefer400.nii.gz \\
              -prefix vp3

  # Restrict to a subset: drop the first run and one task entirely
  ffs_varpart -betas out_single_trial_betas.nii.gz \\
              -trials out_single_trial_events.tsv -factors trial_type,task \\
              -drop_trials run 01 -drop_trials task rest -prefix vp

Sidecar table: one row per volume of -betas, in the same order. Must contain the
columns named by -factors. Columns 'run', 'session' and 'repeat' are used when
present (fold construction and permutation blocks) and ignored when absent.
ffs_reml / ffs_ridge / ffs_denoise write exactly this table next to their
single-trial betas as {prefix}_single_trial_events.tsv when given BIDS -events.
Factor levels may be free text; they are sanitized into identifiers here (the
mapping back to the original labels is written to {prefix}_varpart.json).
""",
    )
    req = p.add_argument_group("Required")
    req.add_argument("-betas", required=True, help="4-D image; one volume per trial")
    req.add_argument("-trials", required=True, help="CSV/TSV sidecar, one row per volume")
    req.add_argument(
        "-factors",
        required=True,
        help="Comma-separated sidecar column names to partition over (exactly 2)",
    )
    req.add_argument("-prefix", required=True, help="Output prefix")

    opt = p.add_argument_group("Options")
    opt.add_argument(
        "-drop_trials",
        "-drop-trials",
        dest="drop_trials",
        nargs=2,
        action="append",
        metavar=("COLUMN", "LABEL"),
        default=None,
        help=(
            "Exclude every trial whose COLUMN equals LABEL, before anything else "
            "(e.g. -drop_trials run 01 -drop_trials task rest). Repeatable; each "
            "occurrence drops one COLUMN/LABEL pair. Numeric labels match regardless "
            "of zero padding and string labels regardless of case."
        ),
    )
    opt.add_argument("-mask", default=None, help="Restrict to voxels inside this mask")
    opt.add_argument(
        "-atlas",
        default=None,
        help=(
            "Collapse to ROIs before partitioning. 3-D integer label map (one ROI per "
            "non-zero value) or 4-D stack (one volume per ROI; binary or weighted, may "
            "overlap). Writes a per-ROI table plus a parcel-painted volume for figures."
        ),
    )
    opt.add_argument(
        "-max_rank",
        "-max-rank",
        dest="max_rank",
        type=int,
        default=None,
        help="Highest interaction rank to cross-validate (default: full rank)",
    )
    opt.add_argument(
        "-min_ncsnr_for_rank",
        "-min-ncsnr-for-rank",
        dest="min_ncsnr_for_rank",
        type=float,
        default=0.75,
        help=(
            "Noise-ceiling SNR below which interaction rank is reported as -1 "
            "(undetermined) instead of 0. Rank selection misses real structure long "
            "before it invents any, so without this low-SNR tissue reads as "
            "'task-invariant'. Set 0 to disable."
        ),
    )
    opt.add_argument(
        "-min_interaction_frac_ceiling",
        "-min-interaction-frac-ceiling",
        dest="min_interaction_frac_ceiling",
        type=float,
        default=0.02,
        help=(
            "Detection floor for rank_E: the rank curve must beat the additive model by "
            "this fraction of the voxel's noise ceiling before any nonzero rank is "
            "reported. Without it, voxels whose interaction was shrunk away entirely have "
            "a flat curve and the argmax reads float noise as structure."
        ),
    )
    opt.add_argument(
        "-nuclear_taus",
        "-nuclear-taus",
        dest="nuclear_taus",
        type=int,
        default=11,
        help=(
            "Grid size for the singular-value soft-threshold sweep, the continuous "
            "counterpart of the hard rank sweep (0 = skip it)"
        ),
    )
    gamma_mode = opt.add_mutually_exclusive_group()
    gamma_mode.add_argument(
        "-nested_gamma",
        "-nested-gamma",
        dest="nested_gamma",
        action="store_true",
        default=True,
        help=(
            "Select each outer fold's band shrinkage using inner folds confined to its "
            "training data (default; honest held-out R2)"
        ),
    )
    gamma_mode.add_argument(
        "-no_nested_gamma",
        "-no-nested-gamma",
        "-reporting_fold_gamma",
        "-reporting-fold-gamma",
        dest="nested_gamma",
        action="store_false",
        help=(
            "Compatibility/fast mode: select one gamma on the reporting-fold predictions. "
            "R2 is mildly optimistic, but this avoids the roughly 3x nested-CV cost."
        ),
    )
    opt.add_argument(
        "-perm",
        type=int,
        default=0,
        help="Permutations for the Freedman-Lane null (0 = skip inference)",
    )
    opt.add_argument(
        "-perm_stats",
        "-perm-stats",
        dest="perm_stats",
        default=None,
        help=(
            "Which statistics to test; comma-separated. Default: unique_<factor> for every "
            "factor plus 'interaction'. Also available: band_<band> for any single band, "
            "e.g. band_stim:task or band_stim:task:noise"
        ),
    )
    opt.add_argument(
        "-strict_run_locality",
        "-strict-run-locality",
        dest="strict_run_locality",
        action="store_true",
        help=(
            "Fail instead of warn when a cell has two repeats inside one run. Repeats "
            "within a run are a legitimate design; they only mean run-level nuisance sits "
            "on both sides of that cell's train/test split, inflating held-out R² and the "
            "noise ceiling (the partition ratios are much less affected). Use this when "
            "you expected repeats to be spread across runs and want to be told they are not."
        ),
    )
    opt.add_argument(
        "-anova",
        action="store_true",
        help=(
            "Also run a classical in-sample factorial ANOVA over the same bands and write "
            "{prefix}_anova.tsv plus anova_* sub-bricks. Independent of the cross-validated "
            "partition: Type-III sums of squares, F-tested against pure within-cell error. "
            "Costs a second or two. Its R² is NOT held out and rises with degrees of "
            "freedom whether or not there is signal, so read omega2 rather than eta2 when "
            "quoting a number"
        ),
    )
    opt.add_argument("-seed", type=int, default=0, help="RNG seed for permutations")
    opt.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto)")
    opt.add_argument("-quiet", action="store_true", help="Suppress progress bars")
    return p


def _read_table(path: str) -> list[dict]:
    """Read the trial sidecar, sniffing tab vs comma from the extension."""
    delim = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        raise SystemExit(f"❌ trial table is empty: {path}")
    return rows


def _column(rows: list[dict], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise SystemExit(
            f"❌ column '{name}' not found in trial table. Available: {sorted(rows[0])}"
        )
    return np.array([r[name] for r in rows])


def _optional_column(rows: list[dict], name: str) -> np.ndarray | None:
    return _column(rows, name) if name in rows[0] else None


def _label_matches(value: str, label: str) -> bool:
    """Lenient equality for -drop_trials.

    Three spellings of the same level all have to match, because all three are things a
    user legitimately has in front of them:

    - the raw table text, modulo whitespace and case (``"Where is  this shown"``);
    - the sanitized identifier that appears in the JSON ``level_names`` and in map
      names (``Where_is_this_shown``);
    - a number written with different zero padding (``run 1`` vs ``run 01``).

    Anything stricter drops *some* trials of a level and leaves the whitespace-variant
    ones in, which is worse than not dropping at all -- it unbalances the design silently.
    """
    a, b = canonicalize_label(value), canonicalize_label(label)
    if a.lower() == b.lower():
        return True
    if level_identifier(a).lower() == level_identifier(b).lower():
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _apply_drop_trials(rows: list[dict], drops: list[list[str]] | None) -> np.ndarray:
    """Return a boolean keep-mask over *rows* after applying every -drop_trials pair.

    A pair that matches nothing is an error, not a no-op: it is almost always a typo
    or the wrong column, and silently analysing the full dataset under the belief
    that a condition was excluded is the failure mode worth being loud about.
    """
    keep = np.ones(len(rows), dtype=bool)
    if not drops:
        return keep

    for column, label in drops:
        if column not in rows[0]:
            raise SystemExit(
                f"❌ -drop_trials: column '{column}' not found in trial table. "
                f"Available: {sorted(rows[0])}"
            )
        hit = np.array([_label_matches(r[column], label) for r in rows], dtype=bool)
        if not hit.any():
            values = sorted({str(r[column]) for r in rows})
            shown = values[:20] + (["..."] if len(values) > 20 else [])
            raise SystemExit(
                f"❌ -drop_trials {column} {label}: no trial has that value.\n"
                f"   Values present in '{column}': {', '.join(shown)}"
            )
        matched = sorted({str(r[column]) for r, h in zip(rows, hit, strict=True) if h})
        # More than one distinct raw value means the label matched whitespace or case
        # variants of itself. That is the intent, but say so -- it is also how a user
        # discovers their table has variants at all.
        variants = f"  [matched {len(matched)} spellings: {', '.join(repr(m) for m in matched)}]"
        print(
            f"   ✂️  -drop_trials {column}={label}: dropping {int(hit.sum())} trials"
            + (variants if len(matched) > 1 else "")
        )
        keep &= ~hit

    if not keep.any():
        raise SystemExit("❌ -drop_trials removed every trial")
    return keep


def main() -> int:
    args = create_parser().parse_args()
    device = setup_device(args.device)

    print("=" * 70)
    print("ffs_varpart - variance partitioning")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"🖥️  Device: {device}")

    factor_names = [f.strip() for f in args.factors.replace(" ", ",").split(",") if f.strip()]
    if len(factor_names) < 2:
        raise SystemExit(
            f"❌ -factors needs at least 2 column names, got {len(factor_names)}: {factor_names}"
        )

    all_rows = _read_table(args.trials)
    n_rows_total = len(all_rows)
    if args.drop_trials:
        print(f"\n✂️  Dropping trials ({n_rows_total} in table)")
    keep = _apply_drop_trials(all_rows, args.drop_trials)
    rows = [r for r, k in zip(all_rows, keep, strict=True) if k]

    # Levels arrive as free text ("face, inverted"), and become identifiers downstream --
    # map names, JSON keys, anything a later script indexes by. Sanitize once here, keeping
    # distinct labels distinct, and report the mapping so the raw names stay recoverable.
    factor_codes: dict[str, np.ndarray] = {}
    level_maps: dict[str, dict[str, str]] = {}
    for name in factor_names:
        raw = [str(v) for v in _column(rows, name)]
        clean, mapping = sanitize_levels(raw)
        factor_codes[name] = np.array(clean)
        level_maps[name] = mapping
        # -drop_trials makes a one-level factor easy to reach by accident; the
        # contrast builder's ValueError does not say which flag caused it.
        if len(set(clean)) < 2:
            raise SystemExit(
                f"❌ factor '{name}' has only one level ({sorted(set(clean))}) after "
                "trial selection; variance partitioning needs at least two."
            )

    run = _optional_column(rows, "run")
    session = _optional_column(rows, "session")
    repeat_col = _optional_column(rows, "repeat")
    repeat = repeat_col.astype(int) if repeat_col is not None else None

    # Session participates in fold locality the same way run does: a repeat that shares a
    # session with its training partners still shares session-level nuisance. When both
    # exist the block label is their combination, which is the stricter of the two.
    if run is not None and session is not None:
        block = np.array([f"{s}/{r}" for s, r in zip(session, run, strict=True)])
    else:
        block = run if run is not None else session

    kept_note = f" ({n_rows_total - len(rows)} dropped)" if len(rows) != n_rows_total else ""
    print(f"\n📋 Trial table: {len(rows)} rows{kept_note}")
    for name in factor_names:
        mapping = level_maps[name]
        renamed = sum(1 for k, v in mapping.items() if k != v)
        # Two raw labels sharing an identifier means they differed only in whitespace
        # and were merged into one level -- worth saying out loud, it changes the design.
        merged = len(mapping) - len(set(mapping.values()))
        notes = []
        if renamed:
            notes.append(f"{renamed} name(s) sanitized")
        if merged:
            notes.append(f"{merged} merged on whitespace")
        note = f", {'; '.join(notes)}" if notes else ""
        print(f"   • {name}: {len(np.unique(factor_codes[name]))} levels{note}")
    print(f"   • run: {'yes' if run is not None else 'absent'}")
    print(f"   • session: {'yes' if session is not None else 'absent'}")
    print(f"   • repeat: {'yes' if repeat is not None else 'derived from cell order'}")

    print(f"\n📥 Loading betas: {args.betas}")
    img = load_nifti(args.betas)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 4:
        raise SystemExit(f"❌ -betas must be 4-D (one volume per trial), got {data.ndim}-D")
    vol_shape, n_trials = data.shape[:3], data.shape[3]
    # The pairing is checked against the *full* table: -drop_trials subsets the volumes
    # here, so the file on disk still has to be one volume per un-dropped row.
    if n_trials != n_rows_total:
        raise SystemExit(
            f"❌ -betas has {n_trials} volumes but the trial table has {n_rows_total} rows.\n"
            "One row per volume is required; use -drop_trials to exclude trials rather "
            "than editing the table."
        )
    if len(rows) != n_rows_total:
        data = data[..., keep]
        n_trials = data.shape[3]
        print(f"   Kept {n_trials} of {n_rows_total} volumes after -drop_trials")

    if args.mask:
        mask = np.asanyarray(load_nifti(args.mask).dataobj).astype(bool)
        if mask.shape != vol_shape:
            raise SystemExit(f"❌ mask {mask.shape} does not match betas grid {vol_shape}")
    else:
        # Voxels that are all-zero across trials carry no information and would give a
        # zero noise ceiling; excluding them keeps them out of the FWE max-statistic.
        mask = np.any(data != 0, axis=3)

    betas = torch.as_tensor(data[mask].astype(np.float32))
    print(f"   {vol_shape} x {n_trials} trials; {betas.shape[0]:,} voxels in mask")

    roi_ids: list | None = None
    roi_spec: np.ndarray | None = None
    if args.atlas:
        atlas = np.asanyarray(load_nifti(args.atlas).dataobj)
        if atlas.shape[:3] != vol_shape:
            raise SystemExit(f"❌ atlas grid {atlas.shape[:3]} does not match betas {vol_shape}")
        roi_spec, roi_ids, roi_sizes = build_roi_weights(atlas, mask=mask)
        betas = collapse_to_rois(betas, roi_spec, roi_sizes, device=device).cpu()
        kind = "label map" if atlas.ndim == 3 else "4-D masks"
        print(f"\n🧩 Atlas ({kind}): collapsed to {betas.shape[0]} ROIs")
        print(f"   ROI sizes: min {roi_sizes.min():.0f}, median {np.median(roi_sizes):.0f}")

    print("\n🔬 Partitioning...")
    res = partition_variance(
        betas,
        factor_codes,
        repeat=repeat,
        run=block,
        max_rank=args.max_rank,
        min_ncsnr_for_rank=args.min_ncsnr_for_rank,
        min_interaction_frac_ceiling=args.min_interaction_frac_ceiling,
        n_nuclear_taus=args.nuclear_taus,
        nested_gamma=args.nested_gamma,
        strict_run_locality=args.strict_run_locality,
        device=device,
        verbose=not args.quiet,
    )

    anova_res = None
    if args.anova:
        anova_res = anova_partition(betas, factor_codes, device=device, verbose=not args.quiet)

    assert res.shared is not None and res.interaction is not None
    d = res.diagnostics
    print("\n📊 Diagnostics")
    print(f"   balanced: {d['balanced']}  (max off-diagonal Gram {d['max_offdiag_gram']:.2e})")
    print(
        f"   cells: {d['cells_total']}, empty {d['cells_empty']}, repeats {d['repeats_min']}"
        f"-{d['repeats_max']}"
    )
    locality = d["run_locality_ok"]
    if locality is None:
        locality_note = "no run/session column"
    elif locality:
        locality_note = "ok (no cell repeats within a run)"
    else:
        n_leaks = sum(int(item.get("n_leaks", 0)) for item in d.get("run_leaks", []))
        locality_note = f"{n_leaks} cell-run repeats — R² and ncsnr inflated, ratios ok"
    print(f"   folds: {d['n_folds']}   run locality: {locality_note}")
    # Shared variance is ~0 by construction under an exhaustively crossed balanced
    # design, so a non-trivial value means the balance broke, not that a real overlap
    # was found. Surface it as a check rather than a result.
    print(f"   shared |C| median: {d['shared_abs_median']:.4f}  (expected ~0)")
    if d["shared_abs_median"] > 0.02:
        print("   ⚠️  shared variance is not ~0: the design is unbalanced somewhere;")
        print("       treat the partition as approximate and check for dropped trials.")
    for pair, frac in d["rank_undetermined_frac_per_pair"].items():
        print(f"   rank undetermined for {pair} (ncsnr < {args.min_ncsnr_for_rank}): {frac:.1%}")
    print(
        f"   gamma selection: {'nested inner folds' if args.nested_gamma else 'reporting folds (legacy)'}"
    )
    nested = d.get("factors_nested_in_run") or {}
    if nested:
        # partition_variance already printed the full explanation; keep a one-line marker
        # here so it survives in the diagnostics block a user scrolls back to.
        for name, per_level in nested.items():
            n_runs = min(per_level.values())
            print(
                f"   ⚠️  '{name}' is NESTED in run ({n_runs} runs/level): unique_{name} is "
                f"confounded with run-level nuisance — read unique/interaction of the "
                f"within-run factor instead."
            )

    perm_res = None
    if args.perm > 0:
        if args.perm_stats:
            stats = tuple(s.strip() for s in args.perm_stats.split(",") if s.strip())
        else:
            stats = tuple(f"unique_{n}" for n in factor_names) + ("interaction",)
        print(f"\n🎲 Permutation null: {args.perm} permutations x {len(stats)} statistic(s)")
        if block is None:
            print("   ⚠️  no run/session column: permuting freely (anticonservative)")
        perm_res = permutation_test(
            betas,
            factor_codes,
            repeat=repeat,
            run=block,
            statistics=stats,
            n_perms=args.perm,
            seed=args.seed,
            strict_run_locality=args.strict_run_locality,
            device=device,
            nested_gamma=args.nested_gamma,
            verbose=not args.quiet,
        )

    # ── Outputs ──────────────────────────────────────────────────────────────
    # ':' separates a band's factors, but it is not usable in an AFNI sub-brick label
    # selector, so band names become '_x_' in map names ("stim_x_task").
    def tag(band: str) -> str:
        return band.replace(":", "_x_")

    maps: dict[str, torch.Tensor] = {}
    for name in factor_names:
        maps[f"unique_{name}"] = res.unique[name]
    maps["shared"] = res.shared
    maps["interaction"] = res.interaction
    if res.preference is not None:
        maps["preference"] = res.preference

    # Per-band contribution to the full model. With two factors these repeat information
    # already in unique_*/interaction; above two they are the primary result, because that
    # is where "which of the seven effects is this voxel carrying" is a real question.
    # With two factors the single interaction band's unique variance IS `interaction`, so
    # these would just duplicate it. Above two factors they are the primary result.
    if len(factor_names) > 2:
        for band in d["bands"]:
            maps[f"band_{tag(band)}"] = res.band_unique[band]

    for pair, rank in res.pair_rank_e.items():
        suffix = "" if len(res.pair_rank_e) == 1 else f"_{tag(pair)}"
        maps[f"rank_E{suffix}"] = rank.float()  # -1 where ncsnr is below the floor
        maps[f"rank_E_raw{suffix}"] = res.pair_rank_e_raw[pair].float()
        for fac, align in res.pair_gain_alignment[pair].items():
            maps[f"gain_align{suffix}_{fac}"] = align
        if pair in res.pair_nuclear_gain:
            maps[f"interaction_nuclear{suffix}"] = res.pair_nuclear_gain[pair]
            maps[f"nuclear_tau{suffix}"] = res.pair_nuclear_tau[pair]

    maps["ncsnr"] = res.ncsnr
    maps["noise_ceiling"] = res.noise_ceiling
    for band in d["bands"]:
        maps[f"gamma_{tag(band)}"] = res.gammas[band]
    maps["r2_additive"] = res.r2["M_add"]
    maps["r2_full"] = res.r2["M_full"]

    # Held-out R² is capped by the noise ceiling, not by 1, and the ceiling varies enormously
    # across the brain. 0.02 against a ceiling of 0.05 is most of what was ever obtainable
    # there; the same 0.02 against 0.60 is marginal. Dividing turns "how much variance" into
    # "how much of the obtainable variance", which is the comparison a reader is making by
    # eye anyway -- and doing it by eye across a map is not possible.
    ceiling = maps["noise_ceiling"]
    zero = torch.zeros((), dtype=ceiling.dtype, device=ceiling.device)
    # Below this an oracle model could not explain 1% of the variance, so the ratio is
    # noise/noise and would paint garbage over exactly the tissue with nothing in it.
    obtainable = ceiling > CEILING_FLOOR
    frac_sources = [f"unique_{n}" for n in factor_names] + ["interaction", "r2_full"]
    frac_sources += [k for k in maps if k.startswith("interaction_nuclear")]
    for source in frac_sources:
        frac = torch.where(obtainable, maps[source] / ceiling.clamp_min(CEILING_FLOOR), zero)
        maps[f"{source}_frac_ceiling"] = frac

    if anova_res is not None:
        for band in anova_res.diagnostics["bands"]:
            maps[f"anova_eta2_{tag(band)}"] = anova_res.eta2[band]
            maps[f"anova_omega2_{tag(band)}"] = anova_res.omega2[band]
            maps[f"anova_F_{tag(band)}"] = anova_res.f[band]
            maps[f"oneminusp_anova_{tag(band)}"] = 1.0 - anova_res.p[band]
        maps["anova_r2_full"] = anova_res.r2_full
        maps["anova_r2_full_adj"] = anova_res.r2_full_adj
        # Replicate scatter is a model-free ceiling; it should track ncsnr's, and where the
        # two disagree one of the two estimators is being told something the other is not.
        maps["anova_noise_ceiling"] = anova_res.noise_ceiling

    perm_map_names: dict[str, str] = {}
    if perm_res is not None:
        # Stored as 1 - p so that "significant" is the *high* end: threshold the map at
        # 0.95 for p < 0.05 and every viewer's one-sided threshold slider does the right
        # thing. Raw p-values invert that (small = interesting), which makes every overlay
        # a two-step operation and is easy to get backwards. The library still returns
        # true p-values; only the written maps are complemented. Name avoids a leading
        # digit so it stays usable in AFNI sub-brick label selectors.
        rename = {}
        if len(factor_names) == 2:
            rename = {
                "unique_a": f"unique_{factor_names[0]}",
                "unique_b": f"unique_{factor_names[1]}",
            }
        for key in perm_res.p_fwe:
            base = tag(rename.get(key, key))
            perm_map_names[key] = base
            maps[f"oneminusp_unc_{base}"] = 1.0 - perm_res.p_uncorrected[key]
            maps[f"oneminusp_fwe_{base}"] = 1.0 - perm_res.p_fwe[key]

    info = parse_prefix(args.prefix)
    stem = info.stem

    assert res.heldout_sst is not None and res.noise_ceiling is not None
    # The pieces the partition splits the full model into: with two factors that is the
    # two uniquenesses plus the interaction, above two it is the 2^k - 1 bands (which
    # already include every main effect, so unique_* would double-count them here).
    if len(factor_names) > 2:
        partition_maps = {n: maps[n] for n in maps if n.startswith("band_")}
    else:
        partition_maps = {f"unique_{name}": maps[f"unique_{name}"] for name in factor_names}
        partition_maps["interaction"] = maps["interaction"]
    summary_maps = dict(partition_maps)
    for name in factor_names:
        summary_maps.setdefault(f"unique_{name}", maps[f"unique_{name}"])
    summary_maps["shared"] = maps["shared"]
    summary_maps["r2_full"] = maps["r2_full"]
    summary_rows = _summarize_effects(summary_maps, res.heldout_sst, res.noise_ceiling)

    dominance, n_with_winner, n_reliable = _dominance_rows(
        partition_maps, res.heldout_sst, res.noise_ceiling
    )
    dom_by_effect = {str(r["effect"]): r for r in dominance}
    for row in summary_rows:
        dom = dom_by_effect.get(str(row["effect"]))
        # shared and r2_full are not competitors in the partition, so they have no
        # dominance count -- an empty cell, not a zero, which would read as "never wins".
        row["n_dominant"] = dom["n_dominant"] if dom else ""
        row["frac_dominant"] = dom["frac_dominant"] if dom else ""

    summary_tsv = f"{stem}_summary.tsv"
    summary_fields = list(summary_rows[0])
    with open(summary_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields, delimiter="	")
        writer.writeheader()
        writer.writerows(summary_rows)

    unit_label = "parcels" if roi_ids is not None else "voxels"
    print(f"\n📈 Overall variance summary ({unit_label})")
    print(
        "   effect                       CV-R²   obtainable   median [IQR]"
        "        positive   dominant"
    )
    for row in summary_rows:
        dom = f"{float(row['frac_dominant']):>7.1%}" if row["frac_dominant"] != "" else "      —"
        print(
            f"   {str(row['effect']):<28} "
            f"{float(row['pooled_cv_r2']):>7.3f}   "
            f"{float(row['pooled_frac_ceiling']):>9.1%}   "
            f"{float(row['median_frac_ceiling']):>6.1%} "
            f"[{float(row['q25_frac_ceiling']):>6.1%}, "
            f"{float(row['q75_frac_ceiling']):>6.1%}]   "
            f"{float(row['positive_frac_reliable']):>7.1%}   {dom}"
        )
    print(
        f"   reliable units: {n_reliable}/{summary_rows[0]['n_units']} "
        f"(noise ceiling > {CEILING_FLOOR:g}); "
        f"{n_with_winner} with a positive effect to be dominant"
    )

    # The partition is only meaningful if its pieces reconstruct the full model. They do
    # not have to sum EXACTLY -- per-band shrinkage is clamped to [0, 1], and a band that
    # hits the boundary in one nested model but not another breaks additivity -- so print
    # the residual rather than asserting on it. A large gap means read the pieces as
    # ordering, not as a decomposition.
    pooled = {str(r["effect"]): float(r["pooled_cv_r2"]) for r in summary_rows}
    parts = [pooled[n] for n in partition_maps] + (
        [pooled["shared"]] if len(factor_names) == 2 else []
    )
    total, full = sum(parts), pooled["r2_full"]
    print(
        f"   partition check: pieces sum to {total:+.4f} vs r2_full {full:+.4f} "
        f"(Δ {total - full:+.4f})"
    )
    print(f"💾 Wrote {summary_tsv}")

    if anova_res is not None:
        anova_rows = _anova_rows(anova_res, anova_res.diagnostics["bands"])
        anova_tsv = f"{stem}_anova.tsv"
        with open(anova_tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(anova_rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(anova_rows)

        ad = anova_res.diagnostics
        print(f"\n🧮 Classical ANOVA ({unit_label}, in-sample, Type III, pure-error F)")
        print(
            "   effect                          df       η²       ω²"
            "        F   p<.05 units   η² expected at null"
        )
        for row in anova_rows:
            print(
                f"   {str(row['effect']):<28} "
                f"{int(row['df']):>5}  "
                f"{float(row['pooled_eta2']):>7.3f}  "
                f"{float(row['pooled_omega2']):>7.3f}  "
                f"{float(row['pooled_F']):>7.2f}   "
                f"{float(row['frac_units_p05']):>10.1%}   "
                f"{float(row['eta2_null_expected']):>18.3f}"
            )
        print(
            f"   pure error: {ad['df_error']} df from replicate scatter"
            f" ({ad['cells_occupied']}/{ad['cells_total']} cells occupied,"
            f" saturated={ad['saturated']})"
        )
        # The last column is what a band of this width collects from noise alone. When eta2
        # is not clearly above it, the band explained nothing and omega2 is the honest read.
        print(
            "   η² is in-sample and inflates with df — compare each row against the last "
            "column,\n   and quote ω² (their difference, rescaled) rather than η² outside "
            "this table."
        )
        band_eta2 = sum(float(r["pooled_eta2"]) for r in anova_rows[:-1])
        full_eta2 = float(anova_rows[-1]["pooled_eta2"])
        print(
            f"   partition check: bands sum to {band_eta2:+.4f} vs r2_full {full_eta2:+.4f} "
            f"(Δ {band_eta2 - full_eta2:+.4f}; exactly 0 only under balance)"
        )
        print(f"💾 Wrote {anova_tsv}")

    sig_rows: list[dict[str, str | int | float]] = []
    sig_units: dict[str, list] = {}
    if perm_res is not None:
        sig_rows = _significance_rows(perm_res.p_uncorrected, perm_res.p_fwe, perm_map_names)
        sig_tsv = f"{stem}_significance.tsv"
        with open(sig_tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sig_rows[0]), delimiter="	")
            writer.writeheader()
            writer.writerows(sig_rows)

        n_units = int(sig_rows[0]["n_units"])
        print(f"\n🎯 Significance ({args.perm} permutations, {n_units} {unit_label})")
        print("   effect                       p<.05 unc     pFWE<.05     pFWE<.01   min pFWE")
        for row in sig_rows:
            print(
                f"   {str(row['effect']):<28} "
                f"{int(row['n_sig_unc_p05']):>9,}   "
                f"{int(row['n_sig_fwe_p05']):>10,}   "
                f"{int(row['n_sig_fwe_p01']):>10,}   "
                f"{float(row['min_p_fwe']):>8.4f}"
            )
        # The smallest reachable p is 1/(N+1): with 18 permutations nothing can clear
        # 0.05 no matter how strong the effect, and a table of zeros then means "too few
        # permutations", not "no effect".
        floor_p = 1.0 / (args.perm + 1)
        if floor_p > 0.05:
            print(
                f"   ⚠️  {args.perm} permutations put the smallest possible p at "
                f"{floor_p:.3f}: nothing can reach p < 0.05. Use -perm 1000 or more."
            )
        elif floor_p > 0.01:
            print(f"   note: p is floored at {floor_p:.4f}; pFWE < 0.01 is unreachable here.")

        if roi_ids is not None:
            for key, base in perm_map_names.items():
                ids = _significant_unit_ids(perm_res.p_fwe[key], maps[base], roi_ids)
                sig_units[base] = ids
                shown = ", ".join(str(i) for i in ids[:12])
                more = f" … (+{len(ids) - 12} more)" if len(ids) > 12 else ""
                print(
                    f"   {base}: {len(ids)}/{len(roi_ids)} parcels pFWE<0.05"
                    + (f" — {shown}{more}" if ids else "")
                )
            roi_sig_tsv = f"{stem}_significant_rois.tsv"
            with open(roi_sig_tsv, "w", newline="") as fh:
                w = csv.writer(fh, delimiter="	")
                w.writerow(["effect", "roi", "value", "p_unc", "p_fwe"])
                for key, base in perm_map_names.items():
                    index = {rid: i for i, rid in enumerate(roi_ids)}
                    for rid in sig_units[base]:
                        i = index[rid]
                        w.writerow(
                            [
                                base,
                                rid,
                                f"{float(maps[base][i]):.6g}",
                                f"{float(perm_res.p_uncorrected[key][i]):.6g}",
                                f"{float(perm_res.p_fwe[key][i]):.6g}",
                            ]
                        )
            print(f"💾 Wrote {roi_sig_tsv} (parcels with pFWE < 0.05, strongest first)")
        print(f"💾 Wrote {sig_tsv}")

    names = list(maps)
    stacked = np.zeros((*vol_shape, len(maps)), dtype=np.float32)

    if roi_ids is not None:
        # The table is the quantitative output -- one row per ROI, no invented resolution.
        out_tsv = f"{stem}_roi.tsv"
        with open(out_tsv, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["roi"] + names)
            for i, rid in enumerate(roi_ids):
                w.writerow([rid] + [f"{float(maps[n][i]):.6g}" for n in names])
        print(f"\n💾 Wrote {out_tsv} ({len(roi_ids)} ROIs x {len(names)} measures)")

        # The volume is the *display* output: every voxel in a parcel carries that
        # parcel's value, so it renders on a brain without claiming within-parcel
        # structure. Same sub-bricks and grid as voxel mode, so figures and overlays
        # work identically either way.
        for i, name in enumerate(names):
            painted = paint_rois_to_voxels(maps[name], roi_spec, int(mask.sum()))
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = painted
            stacked[..., i] = vol
    else:
        for i, name in enumerate(names):
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = maps[name].cpu().numpy().astype(np.float32)
            stacked[..., i] = vol

    out_path = f"{stem}{info.nifti_ext}"
    save_nifti(stacked, output_path=out_path, affine=img.affine, brick_labels=names)
    kind = "parcel-painted" if roi_ids is not None else "voxelwise"
    print(f"💾 Wrote {out_path} ({len(maps)} sub-bricks, {kind})")
    for i, name in enumerate(names):
        print(f"   [{i:>2}] {name}")
    if perm_res is not None:
        print("   p-maps are stored as 1 - p: threshold at 0.95 for p < 0.05.")

    meta = {
        "factors": factor_names,
        "level_names": level_maps,  # sanitized identifier -> original label, per factor
        "dropped_trials": [list(d) for d in (args.drop_trials or [])],
        "n_trials_in_table": n_rows_total,
        "n_trials": n_trials,
        "n_units": int(betas.shape[0]),
        "unit": "roi" if roi_ids is not None else "voxel",
        "diagnostics": {
            k: (v if not isinstance(v, np.generic) else v.item()) for k, v in d.items()
        },
        "nested_gamma": args.nested_gamma,
        "min_ncsnr_for_rank": args.min_ncsnr_for_rank,
        "n_perms": args.perm,
    }
    if anova_res is not None:
        meta["anova"] = {
            "per_band": anova_rows,
            "diagnostics": {
                k: (v if not isinstance(v, np.generic) else v.item())
                for k, v in anova_res.diagnostics.items()
            },
        }
    if perm_res is not None:
        meta["significance"] = {
            "alpha": 0.05,
            "per_effect": sig_rows,
            "significant_rois": sig_units,  # empty without -atlas: voxel indices are not ids
        }
        meta["p_map_convention"] = "oneminusp_* sub-bricks store 1 - p (threshold 0.95 for p<0.05)"
        meta["perm_diagnostics"] = {
            k: (v if not isinstance(v, np.generic) else v.item())
            for k, v in perm_res.diagnostics.items()
        }
    if roi_ids is not None:
        meta["roi_ids"] = [int(r) for r in roi_ids]
    with open(f"{stem}_varpart.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"💾 Wrote {stem}_varpart.json")

    print(f"\n✅ Done: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
