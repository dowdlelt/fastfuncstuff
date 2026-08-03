#!/usr/bin/env python3
"""
ffs_perm — nonparametric permutation tests for single-trial (or any) betas.

Two test families, picked automatically from how you specify the data:

* **1-sample** sign-flip on a single condition's trials.
* **2-sample** label-swap between two conditions, or one-vs-all.

Outputs a single 4-D stat dataset with AFNI-style sub-bricks, plus a
3dClustSim-style cluster table injected into its header (when ``3drefit``
is on PATH) so AFNI's viewer reports cluster significance interactively.

Run-block (within-run) exchangeability is the default whenever events
files are supplied — single-trial fMRI's repeated-measures structure
demands restricted exchangeability (Nichols & Holmes 2002).

This is v1.  Paired tests, TFCE, Freedman-Lane covariates, and Welch
unequal-variance are punted to v2; pooled-variance is the conservative
choice for unequal-N ``-onevsall`` designs in the meantime.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from fastfuncstuff.cli_utils import (
        add_verbose_arg,
        parse_device_arg,
        parse_prefix,
        spinner,
    )
    from fastfuncstuff.io.afni import load_afni_mask, load_nifti, save_nifti
    from fastfuncstuff.stats.cluster import (
        DEFAULT_NN,
        DEFAULT_SIDED,
        ClusterNull,
        accumulate_cluster_null,
        compute_observed_cluster_masks,
        max_abs_t_per_perm,
        p_to_t,
        voxelwise_fwe_p,
    )
    from fastfuncstuff.stats.niml import (
        resolve_mask_idcode,
        run_refit,
        write_clustsim_niml,
        write_mask_b64,
    )
    from fastfuncstuff.stats.perm_io import (
        load_events,
        select_one_sample,
        select_one_vs_all,
        select_two_sample,
    )
    from fastfuncstuff.stats.permutation import (
        generate_label_swaps,
        generate_sign_flips,
        one_sample_t_perm,
        two_sample_t_perm,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _CleanHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Hide ``-foo_bar`` aliases when a ``-foo-bar`` form exists.

    Both forms remain accepted by argparse; only the dash form is shown in
    the help and usage line — keeps the help readable while still honouring
    the project's "accept both" convention.
    """

    def _filter(self, option_strings: list[str]) -> list[str]:
        dash = [s for s in option_strings if "_" not in s]
        return dash or list(option_strings)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        opts = self._filter(action.option_strings)
        if action.nargs == 0:
            return ", ".join(opts)
        args_str = self._format_args(action, self._get_default_metavar_for_optional(action))
        return ", ".join(f"{o} {args_str}" for o in opts)

    def _format_actions_usage(self, actions, groups):
        # Drop underscore aliases from the usage line by rewriting each
        # action's option_strings to its filtered form for one render.
        saved = [a.option_strings for a in actions]
        try:
            for a in actions:
                a.option_strings = self._filter(a.option_strings)
            return super()._format_actions_usage(actions, groups)
        finally:
            for a, orig in zip(actions, saved, strict=False):
                a.option_strings = orig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_perm",
        description="GPU-accelerated nonparametric permutation testing with "
        "AFNI-style cluster correction.",
        formatter_class=_CleanHelpFormatter,
    )

    inp = p.add_argument_group("Inputs")
    inp.add_argument(
        "-input",
        "-i",
        help="Single 4D NIfTI of single-trial (or any-trial) betas, concatenated in run order.",
    )
    inp.add_argument(
        "-input-a",
        "-input_a",
        nargs="+",
        default=None,
        help="(Group/2-file mode) 4D inputs for group A.",
    )
    inp.add_argument(
        "-input-b",
        "-input_b",
        nargs="+",
        default=None,
        help="(Group/2-file mode) 4D inputs for group B. If omitted, "
        "-input-a alone is a 1-sample test against zero.",
    )
    inp.add_argument(
        "-events",
        nargs="+",
        default=None,
        help="BIDS *_events.tsv files, one per run, in the same order as "
        "the trials concatenated in -input.",
    )
    inp.add_argument(
        "-mask",
        help="3D mask NIfTI (or AFNI HEAD/BRIK).  Voxels outside the mask are not tested.",
    )

    sel = p.add_argument_group("Selection (events mode)")
    sel.add_argument(
        "-col",
        help="Events column to read.  Use with -test for 1- or 2-sample selection.",
    )
    sel.add_argument(
        "-test",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Label(s) in -col to test.  One label = 1-sample test on those "
        "trials.  Two labels = 2-sample test between them.",
    )
    sel.add_argument(
        "-onevsall",
        "-one_vs_all",
        nargs=2,
        metavar=("COL", "LABEL"),
        default=None,
        help="2-sample: trials with COL==LABEL vs all other trials.",
    )
    sel.add_argument(
        "-drop-values",
        "-drop_values",
        nargs="+",
        default=(),
        metavar="VAL",
        help="Values in the selection column to exclude entirely (e.g. 'fixation' 'baseline').",
    )

    perm = p.add_argument_group("Permutation")
    perm.add_argument(
        "-n-perms",
        "-n_perms",
        type=int,
        default=1000,
        help="Number of permutations including the identity (default 1000). "
        "Set higher with GPU to push uncorrected p resolution.",
    )
    perm.add_argument(
        "-seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default 0).",
    )
    perm.add_argument(
        "-no-blocks",
        "-no_blocks",
        dest="use_blocks",
        action="store_false",
        default=True,
        help="Disable within-run exchangeability (default: ON when events are provided).",
    )
    perm.add_argument(
        "-welch",
        action="store_true",
        default=False,
        help="(2-sample only) Use Welch's unequal-variance t. Recommended "
        "when group sizes or variances differ (notably -onevsall).",
    )
    perm.add_argument(
        "-vsmooth",
        "-vsmooth_mm",
        "-vsmooth-mm",
        type=float,
        default=0.0,
        metavar="FWHM_MM",
        help="Variance smoothing: Gaussian-smooth the per-perm variance "
        "estimate at this FWHM (mm) and emit pseudo-t bricks "
        "(t_unc_pseudo, t_fwe_pseudo) alongside the regular ones. "
        "Cluster table for the pseudo branch uses empirical tcrits "
        "from the permutation null (pseudo-t is not Student-t). "
        "0 disables.  Matches randomise -v in spirit.",
    )

    clust = p.add_argument_group("Clustering")
    clust.add_argument(
        "-nn",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Restrict cluster tables to one NN connectivity.  Default: "
        "compute all of NN1/NN2/NN3.",
    )
    clust.add_argument(
        "-sided",
        choices=("1-sided", "2-sided", "bi-sided", "all"),
        default="all",
        help="Cluster-table sidedness; 'all' = emit tables for 1-sided, "
        "2-sided, and bi-sided.  (Default 'all'; matches 3dClustSim "
        "-both -bisided.)",
    )
    clust.add_argument(
        "-with-mass",
        "-with_mass",
        dest="with_mass",
        action="store_true",
        default=False,
        help="Also compute cluster-mass tables (slower path: cc3d per "
        "threshold).  Default off — fast single-pass DSU computes "
        "extent only.  AFNI's viewer only uses extent anyway.",
    )

    out = p.add_argument_group("Output")
    out.add_argument(
        "-prefix",
        required=True,
        help="Output prefix (extension optional; .nii.gz is default).",
    )
    out.add_argument(
        "-save-clust-masks",
        "-save_clust_masks",
        action="store_true",
        help="Also write the per-(NN,pthr,metric,sided) surviving-cluster "
        "label maps as <prefix>_clust_masks.nii.gz.  Useful to compare "
        "extent vs mass thresholding.",
    )
    out.add_argument(
        "-no-refit",
        "-no_refit",
        dest="run_refit",
        action="store_false",
        default=True,
        help="Do not call 3drefit even if available.  Still writes the "
        "NIML files and a refit.sh script.",
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument(
        "-device",
        default="cuda",
        help="Compute device (cuda / cpu / mps).  Default cuda.",
    )
    misc.add_argument(
        "-jobs",
        "-j",
        type=int,
        default=None,
        help="Process-pool workers for the cluster null pass.  "
        "Default: os.cpu_count() - 1.  Set 1 to disable parallelism.",
    )
    add_verbose_arg(misc)

    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_4d(path: str | Path) -> tuple[np.ndarray, object]:
    """Load a 4D NIfTI; return ``(data[X,Y,Z,T], img)``."""
    img = load_nifti(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    if data.ndim != 4:
        raise ValueError(f"{path} is not 3D or 4D ({data.shape})")
    return data, img


def _load_mask(path: str | Path | None, shape3d: tuple[int, int, int]) -> np.ndarray:
    """Return a boolean 3D mask of ``shape3d``.  All-True if ``path`` is None."""
    if path is None:
        return np.ones(shape3d, dtype=bool)
    try:
        m = load_afni_mask(path)
    except Exception:
        m = np.asarray(load_nifti(path).dataobj) > 0
    m = np.asarray(m).astype(bool)
    if m.shape != shape3d:
        raise ValueError(f"mask shape {m.shape} != data shape {shape3d}")
    return m


def _resolve_selection(args, events):
    """Return ('one', OneSampleSelection) or ('two', TwoSampleSelection)."""
    n_modes = sum(
        x is not None
        for x in (
            args.test,
            args.onevsall,
        )
    )
    if n_modes != 1:
        raise SystemExit("Specify exactly one of: -col/-test or -onevsall.  See -h.")

    if args.onevsall is not None:
        col, label = args.onevsall
        return "two", select_one_vs_all(events, col, label, tuple(args.drop_values))

    if not args.col:
        raise SystemExit("-test requires -col.")
    labels = args.test
    if len(labels) == 1:
        return "one", select_one_sample(events, args.col, labels[0])
    if len(labels) == 2:
        return "two", select_two_sample(events, args.col, labels[0], labels[1])
    raise SystemExit(f"-test takes 1 or 2 labels (got {len(labels)}).")


def _stack_inputs(paths: list[str]) -> tuple[np.ndarray, object]:
    """Concatenate 4D inputs along the trial axis."""
    arrs = []
    img0 = None
    for p in paths:
        d, im = _load_4d(p)
        if img0 is None:
            img0 = im
            shape3d = d.shape[:3]
        elif d.shape[:3] != shape3d:
            raise ValueError(f"shape mismatch: {p} is {d.shape[:3]}, expected {shape3d}")
        arrs.append(d)
    return np.concatenate(arrs, axis=3), img0


def _run_perm_one_sample(y_nv, n_perms, seed, args):
    rng = np.random.default_rng(seed)
    signs = generate_sign_flips(y_nv.shape[0], n_perms, rng)
    return signs, one_sample_t_perm(
        y_nv,
        signs,
        device=args.device,
        show_progress=args.verb >= 1,
        keep_perm_data=args.vsmooth > 0,
    )


def _run_perm_two_sample(y_nv, group, blocks, n_perms, seed, args):  # noqa: D401
    rng = np.random.default_rng(seed)
    block_arg = blocks if args.use_blocks else None
    swaps = generate_label_swaps(group, n_perms, rng, blocks=block_arg)
    return swaps, two_sample_t_perm(
        y_nv,
        swaps,
        device=args.device,
        show_progress=args.verb >= 1,
        welch=args.welch,
        keep_perm_data=args.vsmooth > 0,
    )


def _compute_pseudo_t(pstats, test_kind: str, args, mask, voxel_size_mm) -> torch.Tensor:
    """Recompute per-perm pseudo-t after smoothing each perm's variance map.

    Returns a ``[P, V_in_mask]`` float32 tensor.  Caller is responsible
    for stripping the identity row from the null distribution.
    """
    from fastfuncstuff.stats.smooth3d import (
        fwhm_mm_to_sigma_vox,
        smooth_var_per_perm,
    )

    sigma_vox = fwhm_mm_to_sigma_vox(args.vsmooth, voxel_size_mm)
    extras = pstats.extras
    if test_kind == "one":
        n = pstats.dof + 1  # one-sample N = dof + 1
        m = extras["perm_means"]  # [P, V] cpu
        sum_y2 = extras["sum_y2"]  # [V] cpu
        var = (sum_y2[None, :] - n * m * m) / (n - 1)
        var = var.clamp_min(1e-30)
        var_smooth = smooth_var_per_perm(var, mask, sigma_vox, device=args.device)
        return (m * float(np.sqrt(n)) / torch.sqrt(var_smooth.clamp_min(1e-30))).float()

    # two-sample
    mA = extras["perm_mA"]
    mB = extras["perm_mB"]
    if args.welch:
        nA = extras["_nA"]
        nB = extras["_nB"]
        var_A_s = smooth_var_per_perm(extras["perm_varA"], mask, sigma_vox, device=args.device)
        var_B_s = smooth_var_per_perm(extras["perm_varB"], mask, sigma_vox, device=args.device)
        denom = torch.sqrt((var_A_s / nA + var_B_s / nB).clamp_min(1e-30))
    else:
        pool_factor = float(extras["pool_factor"])
        var_s = smooth_var_per_perm(extras["perm_var"], mask, sigma_vox, device=args.device)
        denom = torch.sqrt((var_s * pool_factor).clamp_min(1e-30))
    return ((mA - mB) / denom).float()


def _build_stat_dataset(
    test_kind: str,
    pstats,
    mask: np.ndarray,
    sidedness_fwe: str,
    pseudo_t_pv: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[str], list[int], int]:
    """Assemble the 4D output volume.

    Returns ``(volume, labels, stat_brick_indices, dof)`` where
    ``stat_brick_indices`` lists the 0-based sub-brick indices that hold
    a Student t (so 3drefit can attach ``fitt`` stat params).

    When ``pseudo_t_pv`` is given, two extra bricks are appended:
    ``t_unc_pseudo`` and ``t_fwe_pseudo``.  These are NOT true Student-t —
    the value at each voxel is the empirical permutation p (uncorrected
    or max-stat FWE) re-mapped to a t with the nominal DoF, just so the
    AFNI viewer threshold slider behaves monotonically.  The cluster
    table attached for the pseudo branch uses empirical tcrits derived
    from the permutation null (see :func:`empirical_tcrits`).
    """
    from fastfuncstuff.stats.cluster import uncorrected_p_from_perms

    t = pstats.t.numpy()  # [P, V_in_mask]
    obs_t = t[0]
    null_max = max_abs_t_per_perm(pstats.t, sidedness_fwe)
    fwe_p = voxelwise_fwe_p(obs_t, null_max, sidedness_fwe)
    fwe_t = p_to_t(fwe_p, pstats.dof, sidedness_fwe)

    shape3d = mask.shape

    def to_vol(vec_1d: np.ndarray) -> np.ndarray:
        v = np.zeros(shape3d, dtype=np.float32)
        v[mask] = vec_1d
        return v

    if test_kind == "one":
        bricks = [to_vol(pstats.mean.numpy()), to_vol(obs_t), to_vol(fwe_t)]
        labels = ["mean", "t_unc", "t_fwe"]
        stat_brick_indices = [1, 2]
    else:
        bricks = [
            to_vol(pstats.extras["meanA"].numpy()),
            to_vol(pstats.extras["meanB"].numpy()),
            to_vol(pstats.mean.numpy()),  # diff = A - B
            to_vol(obs_t),
            to_vol(fwe_t),
        ]
        labels = ["meanA", "meanB", "diff", "t_unc", "t_fwe"]
        stat_brick_indices = [3, 4]

    if pseudo_t_pv is not None:
        pseudo_obs = pseudo_t_pv[0].numpy()
        pseudo_p_unc = uncorrected_p_from_perms(pseudo_t_pv.numpy(), sidedness_fwe)
        pseudo_null_max = max_abs_t_per_perm(pseudo_t_pv, sidedness_fwe)
        pseudo_p_fwe = voxelwise_fwe_p(pseudo_obs, pseudo_null_max, sidedness_fwe)
        # Map empirical p back to a t-value with the nominal DoF for
        # viewer thresholding (pseudo-t is not Student-t; this is a
        # monotonic remap only).
        t_unc_pseudo = p_to_t(pseudo_p_unc, pstats.dof, sidedness_fwe)
        t_fwe_pseudo = p_to_t(pseudo_p_fwe, pstats.dof, sidedness_fwe)
        bricks += [to_vol(t_unc_pseudo), to_vol(t_fwe_pseudo)]
        idx0 = len(labels)
        labels += ["t_unc_pseudo", "t_fwe_pseudo"]
        stat_brick_indices += [idx0, idx0 + 1]

    vol4d = np.stack(bricks, axis=-1)
    return vol4d, labels, stat_brick_indices, int(pstats.dof)


def _accumulate_cluster_null(
    pstats,
    mask: np.ndarray,
    args,
    save_masks: bool,
    pseudo_t_pv: torch.Tensor | None = None,
    tcrits_override: dict[str, np.ndarray] | None = None,
) -> tuple[ClusterNull, dict | None]:
    """Build the null in parallel; capture observed masks on the main process.

    When ``pseudo_t_pv`` is given, the cluster null is built from the
    pseudo-t permutation matrix instead of the parametric t, using the
    provided ``tcrits_override`` (empirical pseudo-t critical values).
    """
    nns = (args.nn,) if args.nn else DEFAULT_NN
    sideds = DEFAULT_SIDED if args.sided == "all" else (args.sided,)
    t_for_cluster = pseudo_t_pv.numpy() if pseudo_t_pv is not None else pstats.t.numpy()

    obs_masks: dict | None = None
    if save_masks:
        obs_masks = compute_observed_cluster_masks(
            t_for_cluster[0],
            mask,
            pstats.dof,
            nns=nns,
            sideds=sideds,
        )

    null = accumulate_cluster_null(
        t_for_cluster,
        mask,
        dof=pstats.dof,
        nns=nns,
        sideds=sideds,
        n_jobs=args.jobs,
        verbose=args.verb >= 1,
        fast=not args.with_mass,
        tcrits_override=tcrits_override,
    )
    return null, obs_masks


def main() -> None:
    args = build_parser().parse_args()
    parse_device_arg(args.device)  # validates + normalises
    prefix = parse_prefix(args.prefix)
    t_start = time.time()

    # ── Inputs ─────────────────────────────────────────────────────────────
    if args.input is not None and (args.input_a is not None or args.input_b is not None):
        raise SystemExit(
            "Use either -input (events mode) or -input-a/-input-b (group mode), not both."
        )

    test_kind: str

    if args.input is not None:
        with spinner(f"Loading {Path(args.input).name}"):
            data4d, _ = _load_4d(args.input)
        # No events / no selection → default 1-sample test on every trial.
        # Run-block exchangeability is moot for sign flips, so this is well-defined
        # even without events.
        if not args.events and args.test is None and args.onevsall is None:
            import types

            n = data4d.shape[3]
            test_kind = "one"
            selection = types.SimpleNamespace(
                indices=np.arange(n, dtype=np.int64),
                blocks=np.zeros(n, dtype=np.int64),
                label="mean",
            )
        else:
            if not args.events:
                raise SystemExit(
                    "-col/-test and -onevsall require -events (one TSV per run). "
                    "Omit all three to run a 1-sample test on every trial."
                )
            events = load_events(args.events)
            if len(events) != data4d.shape[3]:
                raise SystemExit(
                    f"Events row count ({len(events)}) does not match "
                    f"4D input trial count ({data4d.shape[3]}).  Events must be "
                    "in run order matching the input."
                )
            if args.test is None and args.onevsall is None:
                # Events provided but no selection — default to 1-sample on all
                # trials, with run-blocks (harmless for sign-flip but kept for
                # documentation consistency).
                import types

                n = data4d.shape[3]
                test_kind = "one"
                selection = types.SimpleNamespace(
                    indices=np.arange(n, dtype=np.int64),
                    blocks=events.run_idx.astype(np.int64),
                    label="mean",
                )
            else:
                test_kind, selection = _resolve_selection(args, events)
    else:
        # Group / two-file mode
        if args.input_a is None:
            raise SystemExit("Specify -input or -input-a.")
        import types

        data_a, _ = _stack_inputs(args.input_a)
        if args.input_b is None:
            data4d = data_a
            test_kind = "one"
            n = data4d.shape[3]
            selection = types.SimpleNamespace(
                indices=np.arange(n, dtype=np.int64),
                blocks=np.zeros(n, dtype=np.int64),
                label="mean",
            )
        else:
            data_b, _ = _stack_inputs(args.input_b)
            data4d = np.concatenate([data_a, data_b], axis=3)
            n_a = data_a.shape[3]
            n_b = data_b.shape[3]
            group = np.concatenate(
                [
                    np.ones(n_a, dtype=np.int8),
                    np.zeros(n_b, dtype=np.int8),
                ]
            )
            test_kind = "two"
            selection = types.SimpleNamespace(
                indices=np.arange(n_a + n_b, dtype=np.int64),
                group=group,
                blocks=np.zeros(n_a + n_b, dtype=np.int64),  # no run info → single block
                label_a="A",
                label_b="B",
            )

    # ── Mask ───────────────────────────────────────────────────────────────
    # -mask is optional (_load_mask returns all-True for None), but the spinner
    # label used to evaluate Path(args.mask) unconditionally and died on None
    # before _load_mask was ever reached.
    if args.mask:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask = _load_mask(args.mask, data4d.shape[:3])
    else:
        mask = _load_mask(None, data4d.shape[:3])
        if args.verb >= 1:
            print("[ffs_perm] no -mask given: testing every voxel", file=sys.stderr)
    v_in_mask = int(mask.sum())
    if v_in_mask == 0:
        raise SystemExit("Mask is empty.")

    # ── Subset trials, reshape to [N, V] ───────────────────────────────────
    sel = data4d[..., selection.indices]  # [X,Y,Z,N_sel]
    y_nv = sel.reshape(-1, sel.shape[-1])[mask.ravel()].T.astype(np.float32, copy=False)
    # y_nv is [N_sel, V_in_mask]
    n_trials = y_nv.shape[0]

    if args.verb >= 1:
        print(
            f"[ffs_perm] trials selected: {n_trials}, voxels in mask: {v_in_mask}", file=sys.stderr
        )

    # ── Permutation pass ───────────────────────────────────────────────────
    t0 = time.time()
    if test_kind == "one":
        _, pstats = _run_perm_one_sample(y_nv, args.n_perms, args.seed, args)
        sidedness_fwe = "2-sided"
    else:
        _, pstats = _run_perm_two_sample(
            y_nv,
            selection.group,
            selection.blocks,
            args.n_perms,
            args.seed,
            args,
        )
        sidedness_fwe = "2-sided"
    if args.verb >= 1:
        print(f"[ffs_perm] perm stat pass: {time.time() - t0:.2f}s", file=sys.stderr)

    # ── (Optional) variance smoothing → pseudo-t per perm ─────────────────
    pseudo_t_pv: torch.Tensor | None = None
    pseudo_tcrits: dict[str, np.ndarray] | None = None
    if args.vsmooth > 0:
        t0 = time.time()
        ref_img = load_nifti(args.input or args.input_a[0])
        zooms = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
        pseudo_t_pv = _compute_pseudo_t(pstats, test_kind, args, mask, zooms)
        # Empirical tcrits for the pseudo cluster table — pseudo-t is NOT
        # Student-t, so we can't use student_t.isf(pthr, dof) here.
        from fastfuncstuff.stats.cluster import (
            DEFAULT_PTHR,
            DEFAULT_SIDED,
            empirical_tcrits,
        )

        sideds_for_pseudo = DEFAULT_SIDED if args.sided == "all" else (args.sided,)
        pseudo_tcrits = {
            s: empirical_tcrits(pseudo_t_pv.numpy(), DEFAULT_PTHR, s) for s in sideds_for_pseudo
        }
        if args.verb >= 1:
            print(f"[ffs_perm] vsmooth + pseudo-t pass: {time.time() - t0:.2f}s", file=sys.stderr)

    # ── Build stat dataset ─────────────────────────────────────────────────
    vol4d, labels, stat_brick_indices, dof = _build_stat_dataset(
        test_kind,
        pstats,
        mask,
        sidedness_fwe,
        pseudo_t_pv=pseudo_t_pv,
    )
    stat_path = Path(prefix.with_suffix("stats"))
    with spinner(f"Writing {stat_path.name}"):
        save_nifti(vol4d, stat_path, reference_img=args.input or args.input_a[0])

    # ── Cluster null + observed masks ──────────────────────────────────────
    # When pseudo-t is available, cluster correction is built from the
    # pseudo-t null with empirical tcrits.  The parametric t branch is
    # still emitted in the stat dataset (t_unc / t_fwe) but is no longer
    # the one driving the AFNI viewer's cluster panel.
    t0 = time.time()
    null, obs_masks = _accumulate_cluster_null(
        pstats,
        mask,
        args,
        save_masks=args.save_clust_masks,
        pseudo_t_pv=pseudo_t_pv,
        tcrits_override=pseudo_tcrits,
    )
    if args.verb >= 1:
        print(f"[ffs_perm] cluster null pass: {time.time() - t0:.2f}s", file=sys.stderr)

    # ── Write NIML tables + mask blob ──────────────────────────────────────
    niml_dir = stat_path.parent
    prefix_base = Path(prefix.stem).name  # filename portion only (drop parent dir)
    niml_files_extent: dict[tuple[int, str], Path] = {}
    cmd_line = " ".join(sys.argv)

    # Use header info for nxyz / dxyz.
    ref_img = load_nifti(args.input or args.input_a[0])
    zooms = ref_img.header.get_zooms()[:3]
    nxyz = tuple(int(x) for x in mask.shape)
    dxyz = tuple(float(z) for z in zooms)

    mask_b64 = niml_dir / f"{prefix_base}.mask"  # AFNI 3dClustSim uses .mask
    mask_count = write_mask_b64(mask_b64, mask)
    mask_idcode = resolve_mask_idcode(args.mask)
    mask_name = str(Path(args.mask).resolve()) if args.mask else "<inline>"

    # File basenames + 3drefit attribute names use no-hyphen sidedness
    # ("1sided", "2sided", "bisided"); only the `thresholding` value inside
    # the NIML keeps the hyphen.  This matches AFNI 3dClustSim exactly.
    _attr_name = {"1-sided": "1sided", "2-sided": "2sided", "bi-sided": "bisided"}

    for sided in null.sideds:
        for nn in null.nns:
            ext_table = null.extent_table(sided, nn)
            sided_attr = _attr_name[sided]
            ext_path = niml_dir / f"{prefix_base}.NN{nn}_{sided_attr}.niml"
            write_clustsim_niml(
                ext_path,
                ext_table,
                nn=nn,
                sidedness=sided,
                commandline=cmd_line,
                nxyz=nxyz,
                dxyz=dxyz,
                pthr=null.pthr,
                athr=null.athr,
                n_perms=null.n_perms,
                mask_count=mask_count,
                mask_idcode=mask_idcode,
                mask_name=mask_name,
            )
            niml_files_extent[(nn, sided_attr)] = ext_path
            if args.with_mass:
                mass_table = null.mass_table(sided, nn)
                mass_path = niml_dir / f"{prefix_base}.NN{nn}_{sided_attr}_mass.niml"
                write_clustsim_niml(
                    mass_path,
                    mass_table,
                    nn=nn,
                    sidedness=sided,
                    commandline=cmd_line + "  [mass]",
                    nxyz=nxyz,
                    dxyz=dxyz,
                    pthr=null.pthr,
                    athr=null.athr,
                    n_perms=null.n_perms,
                    mask_count=mask_count,
                    mask_idcode=mask_idcode,
                    mask_name=mask_name,
                )

    # ── Optional: save observed cluster-label masks 4D ────────────────────
    if obs_masks is not None and obs_masks:
        masks_path = niml_dir / f"{prefix_base}_clust_masks{prefix.nifti_ext}"
        keys = sorted(obs_masks.keys(), key=lambda k: (k[3], k[0], k[1]))
        stack = np.stack([obs_masks[k] for k in keys], axis=-1).astype(np.int32)
        with spinner(f"Writing {masks_path.name}"):
            save_nifti(stack, masks_path, reference_img=args.input or args.input_a[0])
        labels_mask = [f"NN{nn}_p{pth:g}_{metric}_{sided}" for (nn, pth, metric, sided) in keys]
        labels_path = masks_path.with_suffix(".labels.txt")
        labels_path.write_text("\n".join(labels_mask) + "\n")
        if args.verb >= 1:
            print(f"[ffs_perm] wrote {masks_path} and {labels_path}", file=sys.stderr)

    # ── 3drefit script (always written) + optional auto-run ───────────────
    from fastfuncstuff.stats.niml import build_refit_commands

    refit_script = niml_dir / f"{prefix_base}_refit.sh"
    _cmds = build_refit_commands(
        stat_path,
        niml_files_extent,
        mask_b64,
        brick_labels=labels,
        stat_brick_indices=stat_brick_indices,
        dof=dof,
    )
    _script_body = "#!/usr/bin/env bash\nset -e\n"
    for _c in _cmds:
        _script_body += (
            " ".join(
                ("'" + a.replace("'", "'\\''") + "'") if any(ch in a for ch in " \t\"'$`\\") else a
                for a in _c
            )
            + "\n"
        )
    refit_script.write_text(_script_body)
    refit_script.chmod(0o755)

    if args.run_refit:
        run_refit(
            stat_path=stat_path,
            niml_files=niml_files_extent,
            mask_b64_path=mask_b64,
            brick_labels=labels,
            stat_brick_indices=stat_brick_indices,
            dof=dof,
            write_script_path=refit_script,
            verbose=args.verb >= 1,
        )

    if args.verb >= 1:
        print(
            f"[ffs_perm] done in {time.time() - t_start:.1f}s. Output: {stat_path}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
