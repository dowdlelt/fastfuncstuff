"""CLI for standalone noise ceilings — bring your own data.

Command: ffs_util_noiseceiling (registered as entry point in pyproject.toml)

The GLM tools compute their ceiling inside their own cross-validation, where
the folds and the nuisance projection are already settled. This is for
everything else: betas from another package, a repeated-stimulus design nobody
ran a GLM on, a ceiling you want to recompute with different repeat groupings.

Two modes, matching the two spaces a ceiling can live in:

**Timeseries** (``-input`` + ``-identical_sets``) is the estimator ffs_pyrf
uses. Runs sharing a label are asserted to have seen an identical stimulus, so
their expected responses are equal and any disagreement between them is noise.
The grouping is declared rather than detected, because whether two movies are
"the same" is a fact about the experiment, not about the data:

    -identical_sets 1 1 1 2 2 2      # runs 1-3 saw one movie, runs 4-6 another

**Beta space** (``-betas`` + ``-trial_table``) is the NSD/GLMsingle estimator.
It needs the per-trial betas and a table saying which condition and run each
belongs to -- exactly the ``_single_trial_events.tsv`` that ffs_ridge,
ffs_denoise, ffs_hrfopt and ffs_reml already write next to their betas, so
their output is directly consumable. Any TSV with ``condition`` and
``run_index`` columns works.

Pass ``-xval_r2`` in either mode to also get the explainable-R2 map, but only
if that R2 came from the same data and the same folds -- the ratio is
meaningless otherwise, which is why it is opt-in rather than automatic.

Usage:
    ffs_util_noiseceiling -input run*.nii.gz -identical_sets 1 1 1 2 2 2 \\
                          -prefix movie_ceiling

    ffs_util_noiseceiling -betas ridge_single_trial_betas.nii.gz \\
                          -trial_table ridge_single_trial_events.tsv \\
                          -prefix ridge_ceiling
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, print_cli_header
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.stats.noise_ceiling import (
    beta_space_ceiling,
    mean_train_repeats,
    ncsnr,
    ncsnr_noise_ceiling,
)
from fastfuncstuff.stats.reliability import split_half_noise_ceiling


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_util_noiseceiling",
        description="Standalone per-voxel noise ceilings from repeated runs or single-trial betas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repeated movie: runs 1-3 saw movie A, runs 4-6 saw movie B
  ffs_util_noiseceiling -input run*.nii.gz -identical_sets 1 1 1 2 2 2 \\
                        -prefix movie_ceiling

  # Bring your own betas, using the table ffs_ridge wrote alongside them
  ffs_util_noiseceiling -betas sub01_single_trial_betas.nii.gz \\
                        -trial_table sub01_single_trial_events.tsv \\
                        -prefix sub01_ceiling

  # ...and turn an existing held-out R2 into an explainable-R2 map
  ffs_util_noiseceiling -betas betas.nii.gz -trial_table trials.tsv \\
                        -xval_r2 sub01_xval_r2.nii.gz -prefix sub01_ceiling

Notes:
  - -identical_sets means BIT-IDENTICAL stimulus, not "same condition". Two runs
    of the same task with different trial orders are NOT an identical set; use
    the beta-space mode for those.
  - Runs in a group must have the same number of timepoints. Groups that do not
    are skipped with a warning, since unequal length means the designs were not
    in fact identical.
  - Nuisance-project (detrend) the timeseries first. Shared drift reproduces
    across repeats and would be counted as signal, inflating the ceiling toward
    1 everywhere.
        """,
    )
    ts = parser.add_argument_group("Timeseries mode (repeated identical runs)")
    ts.add_argument(
        "-input",
        nargs="+",
        default=None,
        help="Run files, one per run, in the order -identical_sets labels them. "
        "Detrend these first; see Notes.",
    )
    ts.add_argument(
        "-identical_sets",
        "-identical-sets",
        dest="identical_sets",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="One label per input run; runs sharing a label saw a bit-identical "
        "stimulus. E.g. '1 1 1 2 2 2' = two sets of three repeats. Labels are "
        "arbitrary tokens; singleton groups are ignored.",
    )

    beta = parser.add_argument_group("Beta-space mode (bring your own betas)")
    beta.add_argument(
        "-betas",
        default=None,
        help="4D single-trial betas, one volume per trial, in the trial table's order.",
    )
    beta.add_argument(
        "-trial_table",
        "-trial-table",
        dest="trial_table",
        default=None,
        metavar="TSV",
        help="Table with 'condition' and 'run_index' columns, one row per trial "
        "-- the {prefix}_single_trial_events.tsv the ffs GLM tools write.",
    )
    beta.add_argument(
        "-zscore_by_run",
        "-zscore-by-run",
        dest="zscore_by_run",
        action="store_true",
        help="Z-score betas per run before estimating the ceiling. Set this if "
        "the cross-validation you are comparing against did (ffs_ridge and "
        "ffs_denoise do by default), or the ceiling will bound a different "
        "quantity than the R2 and the explainable fraction can exceed 1.",
    )
    beta.add_argument(
        "-nsd_form",
        "-nsd-form",
        dest="nsd_form",
        action="store_true",
        help="Report the published NSD ceiling ncsnr^2/(ncsnr^2+1), which assumes "
        "a noiseless predictor, instead of the fold-matched form that accounts "
        "for the training average's own noise. Use for comparison against "
        "published NSD maps; the default is the honest divisor for a CV R2.",
    )

    common = parser.add_argument_group("Common")
    common.add_argument("-prefix", required=True, help="Output prefix.")
    common.add_argument("-mask", default=None, help="Restrict to voxels inside this mask.")
    common.add_argument(
        "-xval_r2",
        "-xval-r2",
        dest="xval_r2",
        default=None,
        help="An existing held-out R2 map to divide by the ceiling, producing "
        "{prefix}_explainable_r2. Only meaningful if it was computed on THIS "
        "data with the same folds.",
    )
    add_verbose_arg(common)
    return parser


def _index_labels(labels: list[str]) -> torch.Tensor:
    """Map free-text labels to indices by first appearance.

    First-appearance rather than sorted order, so the mapping is reproducible
    and does not depend on how the labels happen to sort.
    """
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        if label not in seen:
            seen[label] = len(seen)
        out.append(seen[label])
    return torch.tensor(out)


def _read_trial_order_txt(path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read ``{prefix}_single_trial_order.txt``, the table the ffs tools always write.

    The TSV sidecar only appears when the run used ``-events``, so this
    whitespace format is what most ffs output actually ships with. Its columns
    are ``trial_index condition run`` where the condition carries a per-trial
    suffix (``cond3_001``) and the run is ``run1``-style; both are normalised
    here so the caller sees the same thing either format gives.

    Returns ``None`` if this is not that format, so the caller can try the TSV.
    """
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines or not lines[0].startswith("#"):
        return None
    if "trial_index" not in lines[0] or "condition" not in lines[0]:
        return None

    conditions: list[str] = []
    runs: list[str] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 3:
            continue
        # Strip the trailing _NNN occurrence counter: 'cond3_001' and
        # 'cond3_002' are two trials of ONE condition, and treating them as
        # separate conditions would leave every condition unrepeated and the
        # ceiling unestimable.
        conditions.append(re.sub(r"_\d+$", "", fields[1]))
        runs.append(fields[2])
    if not conditions:
        return None
    return _index_labels(conditions), _index_labels(runs)


def _read_trial_table(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition and run ids per trial, from either table the ffs tools write."""
    parsed = _read_trial_order_txt(path)
    if parsed is not None:
        return parsed

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        print(f"ERROR: {path} has no rows")
        sys.exit(1)
    for required in ("condition", "run_index"):
        if required not in rows[0]:
            print(
                f"ERROR: {path} has no '{required}' column (found: {list(rows[0])}).\n"
                "  Expected either a *_single_trial_events.tsv (condition, run_index)\n"
                "  or a *_single_trial_order.txt (# trial_index condition run)."
            )
            sys.exit(1)

    conditions = _index_labels([row["condition"] for row in rows])
    runs = torch.tensor([int(float(row["run_index"])) for row in rows])
    return conditions, runs


def _load_volume(path: str) -> tuple[np.ndarray, np.ndarray, object, tuple[int, ...]]:
    image = load_nifti(path)
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    shape = data.shape[:3]
    flat = data.reshape(-1, data.shape[3]) if data.ndim == 4 else data.reshape(-1, 1)
    return flat, image.affine, image.header, shape


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    timeseries_mode = args.input is not None
    beta_mode = args.betas is not None
    if timeseries_mode == beta_mode:
        print("ERROR: give exactly one of -input (with -identical_sets) or -betas.")
        sys.exit(1)
    if timeseries_mode and not args.identical_sets:
        print("ERROR: -input requires -identical_sets to say which runs repeat.")
        sys.exit(1)
    if beta_mode and not args.trial_table:
        print("ERROR: -betas requires -trial_table to say which condition each trial is.")
        sys.exit(1)

    print_cli_header("ffs_util_noiseceiling", "standalone per-voxel noise ceilings")
    prefix_info = parse_prefix(args.prefix)
    out_prefix = prefix_info.stem
    extension = prefix_info.nifti_ext

    mask_flat = None
    if args.mask:
        mask_data, _, _, _ = _load_volume(args.mask)
        mask_flat = mask_data[:, 0] > 0

    if timeseries_mode:
        if len(args.identical_sets) != len(args.input):
            print(
                f"ERROR: -identical_sets has {len(args.identical_sets)} labels but "
                f"-input has {len(args.input)} runs; they must correspond one to one."
            )
            sys.exit(1)

        segments = []
        affine = header = None
        shape: tuple[int, ...] = ()
        for path in args.input:
            flat, affine, header, shape = _load_volume(path)
            segments.append(torch.from_numpy(flat))
        run_lengths = [segment.shape[1] for segment in segments]
        run_starts = [int(sum(run_lengths[:i])) for i in range(len(run_lengths))]
        data = torch.cat(segments, dim=1)
        del segments

        groups: dict[str, list[int]] = {}
        for index, label in enumerate(args.identical_sets):
            groups.setdefault(str(label), []).append(index)
        repeat_groups = [members for members in groups.values() if len(members) > 1]
        dropped = [label for label, members in groups.items() if len(members) < 2]
        if not repeat_groups:
            print("ERROR: no label in -identical_sets appears on two or more runs.")
            sys.exit(1)
        print(f"  Runs: {len(args.input)}   repeat groups: {len(repeat_groups)}")
        for label, members in groups.items():
            if len(members) > 1:
                print(f"    '{label}': runs {[m + 1 for m in members]}")
        if dropped:
            print(f"  Ignoring singleton label(s): {', '.join(dropped)}")

        work = data if mask_flat is None else data[torch.from_numpy(mask_flat)]
        ceiling_masked = split_half_noise_ceiling(work, repeat_groups, run_starts, data.shape[1])
        ceiling = torch.full((data.shape[0],), torch.nan)
        if mask_flat is None:
            ceiling = ceiling_masked
        else:
            ceiling[torch.from_numpy(mask_flat)] = ceiling_masked
        ncsnr_map = None
    else:
        betas_flat, affine, header, shape = _load_volume(args.betas)
        condition_ids, run_ids = _read_trial_table(args.trial_table)
        if betas_flat.shape[1] != condition_ids.numel():
            print(
                f"ERROR: -betas has {betas_flat.shape[1]} volumes but -trial_table has "
                f"{condition_ids.numel()} rows; they must correspond one to one."
            )
            sys.exit(1)

        betas = torch.from_numpy(betas_flat)
        work = betas if mask_flat is None else betas[torch.from_numpy(mask_flat)]

        n_runs = int(run_ids.max()) + 1
        cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
        print(f"  Trials: {condition_ids.numel()}   conditions: {int(condition_ids.max()) + 1}")
        print(f"  Runs: {n_runs}")

        if args.nsd_form:
            from fastfuncstuff.stats.noise_ceiling import zscore_betas_by_run

            scored = zscore_betas_by_run(work, run_ids) if args.zscore_by_run else work
            result = ncsnr_noise_ceiling(scored, condition_ids, n_train_repeats=None)
            ncsnr_masked = ncsnr(scored, condition_ids)
            print("  Ceiling form: NSD ncsnr^2/(ncsnr^2+1) (noiseless predictor)")
        else:
            bundle = beta_space_ceiling(
                betas=work,
                condition_ids=condition_ids,
                run_ids=run_ids,
                cv_splits=cv_splits,
                zscore_by_run=args.zscore_by_run,
            )
            result = bundle.result
            ncsnr_masked = bundle.ncsnr_map
            repeats = mean_train_repeats(condition_ids, run_ids, cv_splits)
            print(f"  Ceiling form: fold-matched (m={repeats:.1f} training trials/condition)")

        def _scatter(values: torch.Tensor) -> torch.Tensor:
            if mask_flat is None:
                return values
            full = torch.full((betas.shape[0],), torch.nan)
            full[torch.from_numpy(mask_flat)] = values
            return full

        ceiling = _scatter(result.ceiling)
        ncsnr_map = _scatter(ncsnr_masked)

    finite = ceiling[torch.isfinite(ceiling)]
    usable = finite[finite >= 0.01]
    print()
    if usable.numel():
        print(f"  {usable.numel():,} voxels with ceiling >= 0.01, median {usable.median():.4f}")
    else:
        print("  No voxel reached a ceiling of 0.01 — nothing reproduces in this data.")

    def _save(values: torch.Tensor, name: str) -> None:
        volume = values.detach().cpu().numpy().astype(np.float32).reshape(shape)
        path = f"{out_prefix}_{name}{extension}"
        save_nifti(volume, output_path=path, affine=affine, header=header)
        print(f"  • {path}")

    print()
    print("Writing outputs...")
    _save(ceiling, "noise_ceiling")
    if ncsnr_map is not None:
        _save(ncsnr_map, "ncsnr")

    if args.xval_r2:
        xval_flat, _, _, xval_shape = _load_volume(args.xval_r2)
        if tuple(xval_shape) != tuple(shape):
            print(
                f"  ⚠️  -xval_r2 grid {xval_shape} does not match {shape}; "
                "explainable_r2 not written."
            )
        else:
            xval = torch.from_numpy(xval_flat[:, 0])
            # NaN where the ceiling is too near zero to divide by: see
            # CeilingResult.explainable_r2 for why that is not clamped instead.
            defined = torch.isfinite(ceiling) & (ceiling >= 0.01)
            explainable = torch.where(defined, xval / ceiling.clamp_min(0.01), torch.nan)
            _save(explainable, "explainable_r2")

    print()
    print("✅ Done.")


if __name__ == "__main__":
    main()
