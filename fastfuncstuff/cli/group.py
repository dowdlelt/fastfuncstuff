#!/usr/bin/env python3
"""
ffs_group — group-level tests on one map per subject (3dttest++).

One-sample, two-sample (pooled or ``-unpooled`` Welch), and ``-paired`` tests,
with numeric covariates centred the 3dttest++ way.  The output bucket has
3dttest++'s sub-brick order, labels and degrees of freedom, so it drops into
an AFNI workflow unchanged::

    ffs_group -setA vis s01 s01.nii.gz'[vis#0_Coef]' s02 s02.nii.gz'[vis#0_Coef]' ... \\
              -mask group_mask.nii.gz -prefix vis_ttest

    ffs_group -setA ctl/*.nii.gz -setB pat/*.nii.gz -labelA ctl -labelB pat \\
              -covariates covs.txt -center SAME -prefix ctl_vs_pat

Sets come in 3dttest++'s two forms: the long form (``-setA LABEL subj dset
subj dset ...``) names every subject, which ``-covariates`` matches on; the
short form (``-setA dset dset ...``) labels each subject by its file name,
truncated to 12 characters as 3dttest++ does.  Sub-brick selectors may be
indices (``[3]``) or labels (``[vis#0_Coef]``).

The design behind the tool (model x inference, ETAC, MEMA) is in the wiki
note [[Group-level inference]].
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter, add_device_arg

_NIFTI_SUFFIXES = (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK", ".BRIK.gz", ".mgz")


def build_parser() -> FfsArgumentParser:
    p = FfsArgumentParser(prog="ffs_group", description=__doc__, formatter_class=FfsHelpFormatter)

    inp = p.add_argument_group("Inputs")
    inp.add_argument(
        "-setA",
        nargs="+",
        required=True,
        metavar="ITEM",
        help="First set: 'LABEL subj dset subj dset ...' (long form) or 'dset dset ...'.",
    )
    inp.add_argument(
        "-setB",
        nargs="+",
        metavar="ITEM",
        help="Second set, same forms.  Gives a two-sample (or -paired) test of A-B.",
    )
    inp.add_argument("-labelA", help="Set A's label for the short form (default SetA).")
    inp.add_argument("-labelB", help="Set B's label for the short form (default SetB).")
    inp.add_argument(
        "-brick",
        metavar="SEL",
        help="Sub-brick to read from every dataset that has no [..] of its own: an "
        "index or a label, e.g. -brick 'vis#0_Coef'.  Saves quoting a selector on "
        "every file of a glob.",
    )
    inp.add_argument(
        "-mask",
        help="Analysis mask.  Default: every voxel nonzero in at least one input.",
    )

    test = p.add_argument_group("Test")
    pair = test.add_mutually_exclusive_group()
    pair.add_argument(
        "-paired",
        action="store_true",
        help="Paired test: subject k of A with subject k of B (a one-sample test of A-B).",
    )
    pair.add_argument(
        "-unpooled",
        action="store_true",
        help="Separate variances for A and B (Welch, Satterthwaite dof).  Every "
        "t brick is written as z, as in 3dttest++.",
    )
    test.add_argument(
        "-covariates",
        metavar="FILE",
        help="Table: header line, then 'label value value ...' per subject.  Rows "
        "are matched to the set labels; extra rows are ignored, a missing one is an "
        "error.  Numeric columns only.  With two sets each set gets its own slopes, "
        "and the A-B block reports their differences.",
    )
    test.add_argument(
        "-center",
        choices=("DIFF", "SAME", "NONE"),
        type=str.upper,
        help="Covariate centering: DIFF removes each set's own mean, SAME the "
        "mean over both sets, NONE nothing (the mean bricks are then the value "
        "at covariate = 0).  Default DIFF for -model ols (3dttest++), SAME for "
        "-model mema (3dMEMA).",
    )
    test.add_argument("-toz", action="store_true", help="Write t statistics as z.")

    model = p.add_argument_group("Model")
    model.add_argument(
        "-model",
        choices=("ols", "mema"),
        default="ols",
        help="ols: the 3dttest++ t-test on one map per subject.  mema: the 3dMEMA "
        "mixed-effects model, which weights each subject by its own precision "
        "(beta and t per subject) plus a between-subject variance tau^2 per voxel.  "
        "MEMA's two-set model has one tau^2 and shared covariate slopes; its "
        "difference is reported as A-B (3dMEMA itself writes B-A).",
    )
    model.add_argument(
        "-mema_reml",
        choices=("always", "shortcut", "never"),
        default="always",
        help="Where MEMA estimates tau^2 by REML.  3dMEMA uses REML only where a "
        "moments-estimate |z| > 1.3 and keeps the moments estimate elsewhere, which "
        "puts a seam in the map at 1.3; 'always' removes it (cost: a few REML "
        "iterations per voxel).  'shortcut' reproduces 3dMEMA to float precision.",
    )
    model.add_argument(
        "-no_HKtest",
        action="store_true",
        help="MEMA: skip the Hartung-Knapp variance adjustment (3dMEMA's default is on).",
    )

    inf = p.add_argument_group("Cluster inference (permutation; -model ols)")
    inf.add_argument(
        "-clustsim",
        action="store_true",
        help="Attach 3dClustSim-format cluster-size tables built from a "
        "permutation null (3dttest++ -Clustsim); ffs_viewer and AFNI then report "
        "each cluster's corrected alpha as you move the threshold.  A dataset holds "
        "one table set, so the output bucket's tables are for its main test (A-B, "
        "or the one-sample mean); every other t brick is also written to its own "
        "PREFIX.<label> file (e.g. PREFIX.G1_mean, PREFIX.G1-G2_age) carrying "
        "tables from that test's own null.  The .1D/.niml side files go in "
        "PREFIX.clustsim/.  Works for every model: with -model mema each "
        "permutation refits tau^2 (minutes rather than seconds at full brain size).",
    )
    inf.add_argument(
        "-etac",
        action="store_true",
        help="Global ETAC (3dttest++ -ETAC): several p-thresholds, figures of merit "
        "and blurs thresholded jointly at one family-wise FPR.  Writes "
        "PREFIX.ETACmask.global.SIDE.Nperc (survivors), PREFIX.ETACmaskALL... "
        "(one volume per blur x pthr sub-test) and PREFIX.ETAC.thresholds.json.",
    )
    inf.add_argument(
        "-etac_blur",
        nargs="+",
        type=float,
        metavar="FWHM",
        help="Extra Gaussian blurs (mm FWHM, inside the mask), one ETAC case "
        "each; 0 is the unblurred input.  Default: 0 only.  Blur adds in "
        "quadrature to what the inputs already have.  With -clustsim, each "
        "nonzero blur also writes PREFIX.B<fwhm> -- the same test on the blurred "
        "inputs -- carrying its own ClustSim tables (3dttest++'s layout); the "
        "main output's tables are for the unblurred data.",
    )
    inf.add_argument(
        "-etac_pthr",
        nargs="+",
        type=float,
        metavar="P",
        help="Voxelwise p-thresholds (default 0.010 0.009 ... 0.001, 3dXClustSim's).",
    )
    inf.add_argument(
        "-etac_hpow",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=[2],
        help="Figures of merit: sum |z|^h over the cluster; 0 is size (default 2).",
    )
    inf.add_argument("-etac_nn", type=int, choices=(1, 2, 3), default=2, help="Connectivity.")
    inf.add_argument(
        "-etac_sid",
        type=int,
        choices=(1, 2),
        default=2,
        help="2: two-sided, clusters join both signs (one mask, 2sid).  1: "
        "one-sided, positive and negative masks (1pos, 1neg) each at the FPR.",
    )
    inf.add_argument(
        "-etac_fpr",
        nargs="+",
        type=float,
        default=[5.0],
        metavar="PCT",
        help="Family-wise false-positive goal(s) in percent (default 5).",
    )
    inf.add_argument(
        "-etac_splitfrac",
        type=float,
        default=0.0,
        help="Set thresholds on the first fraction of iterations and measure the "
        "FPR on the rest (0.2-0.8); 0, the default as in AFNI, uses all for both.",
    )
    inf.add_argument(
        "-niter",
        type=int,
        default=10080,
        help="Permutations for -clustsim / -etac (default 10080, as 3dttest++).",
    )
    inf.add_argument("-seed", type=int, default=0, help="Permutation seed.")

    out = p.add_argument_group("Output")
    out.add_argument("-prefix", required=True, help="Output bucket (.nii.gz by default).")
    out.add_argument("-overwrite", action="store_true", help="Replace an existing output.")

    misc = p.add_argument_group("Execution")
    add_device_arg(misc)
    misc.add_argument("-verb", type=int, default=1, choices=(0, 1, 2), help="Verbosity.")
    return p


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------


def _split_selector(spec: str) -> tuple[str, str | None]:
    m = re.match(r"^(.*?)'?\[([^\]]*)\]'?$", spec)
    return (m.group(1), m.group(2)) if m else (spec, None)


def _exists(spec: str) -> bool:
    path, _ = _split_selector(spec)
    return Path(path).exists() or Path(path + ".HEAD").exists()


def short_label(spec: str) -> str:
    """3dttest++'s short-form subject label: the file prefix, 12 characters."""
    name = Path(_split_selector(spec)[0]).name
    for suf in _NIFTI_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    name = re.sub(r"\+(orig|tlrc|acpc)$", "", name)
    return name[:12]


def with_brick(spec: str, brick: str | None) -> str:
    """Apply the ``-brick`` default to a spec that has no selector of its own."""
    if brick is None or _split_selector(spec)[1] is not None:
        return spec
    return f"{spec}[{brick}]"


def parse_set(
    items: list[str],
    default_label: str,
    label_flag: str | None,
    want_t: bool = False,
    brick: str | None = None,
):
    """``(set_label, [subject labels], [[spec] or [beta, t]])`` from a -setX value.

    Long form ``LABEL subj d [t] subj d [t] ...``; short form ``d [t] d [t] ...``
    with each subject labelled by its first dataset's file name.  With
    ``want_t`` (MEMA) a subject's t-stat may be given explicitly — the dataset
    right after its beta — or left out, and then it is found in the beta's own
    bucket (:func:`find_tstat`).
    """
    items = list(items)
    long_form = not _exists(items[0])
    if long_form:
        label, rest = items[0], items[1:]
        if label_flag:
            raise ValueError(f"set {label!r} is in long form: drop -label{default_label[-1]}")
    else:
        label, rest = label_flag or default_label, items

    subj: list[str] = []
    groups: list[list[str]] = []
    k = 0
    while k < len(rest):
        if long_form:
            if k + 1 >= len(rest):
                raise ValueError(f"set {label!r}: subject {rest[k]!r} has no dataset")
            name, k = rest[k], k + 1
        else:
            name = None
        d = rest[k]
        if not _exists(d):
            raise ValueError(f"set {label!r}: no such dataset {d!r}")
        grp = [with_brick(d, brick)]
        k += 1
        if want_t and _is_tstat(grp[0]):
            raise ValueError(
                f"set {label!r}: {grp[0]} is marked as a t statistic; MEMA needs the "
                "coefficient (its t-stat is found from it)"
            )
        if want_t and k < len(rest) and _exists(rest[k]):
            # Short form pairs only a t from the beta's own file: two adjacent
            # betas from different subjects must never be read as (beta, t).
            same_file = _split_selector(rest[k])[0] == _split_selector(d)[0]
            if long_form or (same_file and _is_tstat(rest[k])):
                grp.append(rest[k])
                k += 1
        subj.append(name if name is not None else short_label(d))
        groups.append(grp)
    if not groups:
        raise ValueError(f"set {label!r} is empty")
    if want_t:
        groups = [g if len(g) == 2 else [g[0], find_tstat(g[0])] for g in groups]
    return label, subj, groups


def _brick_meta(path: str):
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.headers import read_brick_labels, read_brick_stataux

    img = load_nifti(path)
    return read_brick_labels(img), read_brick_stataux(img), img


def _resolve_index(spec: str) -> tuple[str, int | None, list[str], dict]:
    """``(path, sub-brick index or None, labels, stataux)`` of a spec."""
    import numpy as np

    path, sel = _split_selector(spec)
    labels, aux, img = _brick_meta(path)
    if sel is None:
        n = int(np.prod(img.shape[3:])) if len(img.shape) > 3 else 1
        return path, (0 if n == 1 else None), labels, aux
    if re.fullmatch(r"\s*\d+\s*", sel):
        return path, int(sel), labels, aux
    if sel in labels:
        return path, labels.index(sel), labels, aux
    raise ValueError(f"{path}: no sub-brick labelled {sel!r}")


def _is_tstat(spec: str) -> bool:
    """Whether a spec names a t-statistic brick (stat code fitt, or *Tstat label)."""
    try:
        _, idx, labels, aux = _resolve_index(spec)
    except (ValueError, OSError):
        return False
    if idx is None:
        return False
    if idx in aux:
        return aux[idx][0] == 3
    return idx < len(labels) and labels[idx].endswith("Tstat")


def find_tstat(beta_spec: str) -> str:
    """The t-stat brick that goes with a coefficient brick, from the same bucket.

    By label first (``X_Coef`` → ``X_Tstat``, the 3dDeconvolve / 3dREMLfit /
    ffs_reml convention), else the next sub-brick if the header says it is a t
    statistic.  Anything else is refused — guessing a brick is how a group
    analysis silently runs on the wrong numbers.
    """
    path, idx, labels, aux = _resolve_index(beta_spec)
    if idx is None:
        raise ValueError(f"{beta_spec}: several sub-bricks; select the coefficient (or use -brick)")
    lab = labels[idx] if idx < len(labels) else ""
    if lab.endswith("_Coef"):
        want = lab[: -len("_Coef")] + "_Tstat"
        if want in labels:
            return f"{path}[{labels.index(want)}]"
    nxt = idx + 1
    if aux.get(nxt, (None,))[0] == 3:
        return f"{path}[{nxt}]"
    found = f"label {lab!r}" if lab else f"sub-brick {idx}"
    raise ValueError(
        f"{beta_spec}: cannot find its t-stat ({found}; no matching _Tstat label and "
        f"sub-brick {nxt} is not marked as a t statistic).  Give it explicitly after the beta."
    )


def load_volume(spec: str):
    """One 3-D volume ``(data, img)``; label selectors resolved against the header."""
    import numpy as np

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.headers import read_brick_labels

    path, sel = _split_selector(spec)
    if sel is not None and not re.fullmatch(r"[\d,.$()\s]+", sel):
        img = load_nifti(path)
        labels = read_brick_labels(img)
        if sel not in labels:
            raise ValueError(f"{path}: no sub-brick labelled {sel!r} (have {labels[:8]}...)")
        idx = labels.index(sel)
        data = np.asanyarray(img.dataobj)
        data = data.reshape(*data.shape[:3], -1)[..., idx]
        return np.asarray(data, dtype=np.float32), img
    img = load_nifti(spec)
    data = np.asanyarray(img.dataobj)
    if data.ndim > 3 and int(np.prod(data.shape[3:])) != 1:
        raise ValueError(f"{spec}: {int(np.prod(data.shape[3:]))} volumes; select one with [..]")
    return np.asarray(data.reshape(data.shape[:3]), dtype=np.float32), img


def read_covariates(path: str, labels: list[str]):
    """``(names, [n, q] values)`` for the given subject labels, in their order."""
    import numpy as np

    lines = [ln.split() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln[0].startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"{path}: needs a header line and at least one row")
    names = tuple(lines[0][1:])
    table = {}
    for row in lines[1:]:
        if len(row) != len(names) + 1:
            raise ValueError(
                f"{path}: row {row[0]!r} has {len(row) - 1} values, header {len(names)}"
            )
        try:
            table[row[0]] = [float(v) for v in row[1:]]
        except ValueError as err:
            raise ValueError(
                f"{path}: row {row[0]!r} is not all numeric (dataset covariates are not supported)"
            ) from err
    missing = [lab for lab in labels if lab not in table]
    if missing:
        raise ValueError(f"{path}: no row for subject(s) {missing[:5]}")
    return names, np.array([table[lab] for lab in labels], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except ValueError as err:
        print(f"❌ ERROR: {err}", file=sys.stderr)
        sys.exit(1)


def run(args) -> None:
    import numpy as np
    import torch

    from fastfuncstuff.cli_utils import parse_prefix, setup_device
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.stats.group_ttest import TtestDesign, run_ttest, to_z_bricks

    prefix = parse_prefix(args.prefix)
    args.prefix_stem = prefix.stem
    out_path = Path(prefix.stem + prefix.nifti_ext)
    if out_path.exists() and not args.overwrite:
        raise ValueError(f"{out_path} exists (use -overwrite)")
    if (args.paired or args.unpooled) and not args.setB:
        raise ValueError("-paired and -unpooled need -setB")
    mema = args.model == "mema"
    if not mema and (args.no_HKtest or args.mema_reml != "always"):
        raise ValueError("-no_HKtest and -mema_reml apply to -model mema only")
    center = args.center or ("SAME" if mema else "DIFF")
    if mema and any(b > 0 for b in (args.etac_blur or ())):
        raise ValueError(
            "-etac_blur > 0 is OLS-only: blurring a beta changes its within-subject "
            "variance in a way its t map does not record"
        )
    if not args.etac and args.etac_pthr is not None:
        raise ValueError("-etac_pthr needs -etac")
    if not (args.etac or args.clustsim) and args.etac_blur is not None:
        raise ValueError("-etac_blur needs -etac or -clustsim")
    la, subj_a, dsets_a = parse_set(args.setA, "SetA", args.labelA, mema, args.brick)
    lb, subj_b, dsets_b = (None, [], [])
    if args.setB:
        lb, subj_b, dsets_b = parse_set(args.setB, "SetB", args.labelB, mema, args.brick)
        if lb == la:
            raise ValueError(f"both sets are labelled {la!r}")

    device = setup_device(args.device)
    # vols[s][j]: subject s, dataset j (j=0 the map/beta, j=1 MEMA's t).
    vols_a = [[load_volume(d) for d in grp] for grp in dsets_a]
    vols_b = [[load_volume(d) for d in grp] for grp in dsets_b]
    ref_img = vols_a[0][0][1]
    shape = vols_a[0][0][0].shape
    for grp, vols in zip(dsets_a + dsets_b, vols_a + vols_b, strict=True):
        for spec, (v, _) in zip(grp, vols, strict=True):
            if v.shape != shape:
                raise ValueError(f"{spec}: grid {v.shape} differs from {shape}")

    if args.mask:
        mask_vol, _ = load_volume(args.mask)
        if mask_vol.shape != shape:
            raise ValueError(f"mask grid {mask_vol.shape} differs from the data {shape}")
        mask = mask_vol != 0
    else:
        mask = np.zeros(shape, dtype=bool)
        for vols in vols_a + vols_b:
            mask |= vols[0][0] != 0
    n_vox = int(mask.sum())
    if n_vox == 0:
        raise ValueError("empty mask")

    # MEMA's REML wants float64 (on MPS that means the CPU: linalg_device).
    from fastfuncstuff.utils import linalg_device

    dev = linalg_device(device) if mema else device
    dt = torch.float64 if mema else torch.float32

    def stack(vols, j):
        arr = np.stack([v[j][0][mask] for v in vols])
        return torch.as_tensor(arr, device=dev).to(dt)

    cov_a = cov_b = None
    names: tuple[str, ...] = ()
    if args.covariates:
        names, cov_a = read_covariates(args.covariates, subj_a)
        if subj_b:
            _, cov_b = read_covariates(args.covariates, subj_b)

    design = TtestDesign(
        n_a=len(vols_a),
        n_b=len(vols_b),
        label_a=la,
        label_b=lb or "SetB",
        paired=args.paired,
        unpooled=args.unpooled,
        cov_a=cov_a,
        cov_b=cov_b,
        cov_names=names,
        center=center,
    )
    if args.verb:
        kind = (
            ("paired" if args.paired else "Welch" if args.unpooled else "two-sample")
            if design.two_sets
            else "one-sample"
        )
        cov = f", covariates {', '.join(names)} (center {center})" if names else ""
        print(
            f"[ffs_group] {args.model} {kind}: {la} n={design.n_a}"
            + (f", {lb} n={design.n_b}" if design.two_sets else "")
            + f", {n_vox} voxels{cov}",
            file=sys.stderr,
        )

    if mema:
        from fastfuncstuff.stats.group_mema import run_mema

        bricks, fit = run_mema(
            design,
            stack(vols_a, 0),
            stack(vols_a, 1),
            stack(vols_b, 0) if vols_b else None,
            stack(vols_b, 1) if vols_b else None,
            knha=not args.no_HKtest,
            reml=args.mema_reml,
        )
        if args.toz:
            bricks = to_z_bricks(bricks)
        if args.verb and args.mema_reml != "never":
            n_bad = int((~fit.converged).sum())
            msg = f"[ffs_group] REML on {int(fit.reml.sum())}/{n_vox} voxels"
            if n_bad:
                msg += f"; {n_bad} did not converge in 200 iterations and keep the moments tau^2"
            print(msg, file=sys.stderr)
    else:
        bricks = run_ttest(design, stack(vols_a, 0), stack(vols_b, 0) if vols_b else None,
                           toz=args.toz)  # fmt: skip

    out = np.zeros((*shape, len(bricks)), dtype=np.float32)
    for kb, b in enumerate(bricks):
        out[..., kb][mask] = b.data.detach().cpu().numpy()
    stataux = {kb: b.stat for kb, b in enumerate(bricks) if b.stat is not None}
    save_nifti(
        out,
        out_path,
        affine=ref_img.affine,
        header=ref_img.header,
        brick_labels=[b.label for b in bricks],
        brick_stataux=stataux,
    )
    if args.verb:
        print(f"[ffs_group] wrote {out_path} ({len(bricks)} sub-bricks)", file=sys.stderr)

    if args.clustsim or args.etac:
        voxdims = tuple(float(abs(z)) for z in ref_img.header.get_zooms()[:3])
        ya, yb = stack(vols_a, 0), stack(vols_b, 0) if vols_b else None
        ta = stack(vols_a, 1) if mema else None
        tb = stack(vols_b, 1) if mema and vols_b else None
        model_kw = dict(model=args.model, ta=ta, tb=tb, knha=not args.no_HKtest,
                        reml=args.mema_reml)  # fmt: skip
        if args.clustsim:
            _clustsim(args, design, ya, yb, model_kw, mask, voxdims, bricks, out_path, prefix,
                      ref_img)  # fmt: skip
        if args.etac:
            _run_etac(args, design, ya, yb, model_kw, mask, voxdims, prefix.stem,
                      prefix.nifti_ext, ref_img)  # fmt: skip


def _clustsim(args, design, ya, yb, model_kw, mask, voxdims, bricks, out_path, prefix,
              ref_img) -> None:  # fmt: skip
    """Tables for every t brick: the main test in the bucket, the rest in their own files.

    A dataset carries one ClustSim table set, so the bucket's tables describe
    its main test; every other (estimate, t) pair goes to PREFIX.<label>, whose
    name says which test its tables belong to.  Each null comes from the same
    estimator as its brick (pooled / Welch OLS, MEMA refitting tau^2).
    """
    import numpy as np

    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.stats.group_inference import brick_nulls

    if args.verb and model_kw["model"] == "mema":
        print("[ffs_group] MEMA null: tau^2 refitted for every permutation", file=sys.stderr)
    nulls = brick_nulls(design, ya=ya, yb=yb, n_iter=args.niter, seed=args.seed, **model_kw)
    for bn in nulls:
        if bn.main:
            path = out_path
        else:
            pair = bricks[bn.brick : bn.brick + 2]
            safe = re.sub(r"[^A-Za-z0-9._+-]", "_", pair[0].label.replace(":b", ""))
            path = Path(f"{prefix.stem}.{safe}{prefix.nifti_ext}")
            out = np.zeros((*mask.shape, 2), dtype=np.float32)
            for kb, b in enumerate(pair):
                out[..., kb][mask] = b.data.detach().cpu().numpy()
            save_nifti(out, path, affine=ref_img.affine, header=ref_img.header,
                       brick_labels=[b.label for b in pair],
                       brick_stataux={1: pair[1].stat} if pair[1].stat else None)  # fmt: skip
        _attach_tables(args, bn.source, mask, voxdims, path)
        if args.verb and not bn.main:
            print(f"[ffs_group] {path}: {bn.label} with its own ClustSim tables", file=sys.stderr)
    for fwhm in sorted(set(args.etac_blur or ())):
        if fwhm > 0:
            _clustsim_blur_case(args, design, ya, yb, mask, voxdims, fwhm, prefix, ref_img)


def _attach_tables(args, source, mask, voxdims, path) -> None:
    from fastfuncstuff.stats.clustsim import attach_clustsim_tables
    from fastfuncstuff.stats.group_ttest import PermutationFields

    _warn_distinct(args, source.n_distinct)
    # The .1D / .niml / .mask / 3drefit side files (18 per table set) go in one
    # PREFIX.clustsim/ folder; the tables themselves live in each NIfTI header.
    path = Path(path)
    # Only the NIfTI extension: Path.suffixes also counts ".G1_mean", which
    # once named every per-brick table set "PREFIX.CSim" and let them overwrite.
    name = re.sub(r"\.nii(\.gz|\.zst)?$", "", path.name)
    folder = Path(str(args.prefix_stem) + ".clustsim")
    attach_clustsim_tables(
        mask,
        voxdims,
        None,
        prefix=folder / f"{name}.CSim",
        refit=path,
        n_iter=args.niter,
        device=source.device,
        seed=args.seed,
        commandline=" ".join(sys.argv),
        verbose=args.verb >= 1,
        simulator=PermutationFields(source, mask),
    )


def _clustsim_blur_case(args, design, ya, yb, mask, voxdims, fwhm, prefix, ref_img) -> None:
    """PREFIX.B<fwhm>: the test re-run on blurred inputs, with its own tables."""
    import numpy as np

    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.stats.blur_to import MaskedGaussian
    from fastfuncstuff.stats.etac import case_label
    from fastfuncstuff.stats.group_ttest import run_ttest

    g = MaskedGaussian(mask, voxdims, fwhm, device=ya.device)
    ya_b = g.rows(ya)
    yb_b = g.rows(yb) if yb is not None else None
    bricks = run_ttest(design, ya_b, yb_b, toz=args.toz)
    path = Path(f"{prefix.stem}.{case_label(fwhm)}{prefix.nifti_ext}")
    out = np.zeros((*mask.shape, len(bricks)), dtype=np.float32)
    for kb, b in enumerate(bricks):
        out[..., kb][mask] = b.data.detach().cpu().numpy()
    save_nifti(out, path, affine=ref_img.affine, header=ref_img.header,
               brick_labels=[b.label for b in bricks],
               brick_stataux={kb: b.stat for kb, b in enumerate(bricks) if b.stat})  # fmt: skip
    from fastfuncstuff.stats.group_inference import brick_nulls

    (bn,) = brick_nulls(design, model="ols", ya=ya_b, yb=yb_b, n_iter=args.niter,
                        seed=args.seed, main_only=True)  # fmt: skip
    _attach_tables(args, bn.source, mask, voxdims, path)


def _warn_distinct(args, n_distinct: float) -> None:
    if args.verb and n_distinct < args.niter:
        print(
            f"[ffs_group] note: only {n_distinct:.0f} distinct relabelings exist for "
            f"this design; {args.niter} iterations repeat some of them",
            file=sys.stderr,
        )


def _run_etac(args, design, ya, yb, model_kw, mask, voxdims, stem, ext, ref_img) -> None:
    import json

    import numpy as np

    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.stats.etac import DEFAULT_PTHR, EtacOptions, run_group_etac

    opts = EtacOptions(
        pthr=tuple(args.etac_pthr) if args.etac_pthr else DEFAULT_PTHR,
        hpows=tuple(sorted(set(args.etac_hpow))),
        nn=args.etac_nn,
        sided=args.etac_sid,
        fpr=tuple(args.etac_fpr),
        splitfrac=args.etac_splitfrac,
    )
    blurs = tuple(args.etac_blur) if args.etac_blur else (0.0,)
    res = run_group_etac(design, ya, yb, mask, voxdims, blurs=blurs, opts=opts,
                         n_iter=args.niter, seed=args.seed, device=ya.device,
                         verbose=args.verb >= 1, **model_kw)  # fmt: skip
    _warn_distinct(args, res.n_distinct)
    shape = mask.shape
    sub_labels = [f"{c}_p{p:.4g}" for c in res.case_labels for p in opts.pthr]
    for (goal, side), (union, per) in res.masks.items():
        tag = f"global.{side}.{goal:g}perc"
        vol = np.zeros(shape, dtype=np.float32)
        vol[mask] = union
        save_nifti(vol, f"{stem}.ETACmask.{tag}{ext}", affine=ref_img.affine)
        allv = np.zeros((*shape, per.shape[0]), dtype=np.float32)
        for k in range(per.shape[0]):
            allv[..., k][mask] = per[k]
        save_nifti(allv, f"{stem}.ETACmaskALL.{tag}{ext}", affine=ref_img.affine,
                   brick_labels=sub_labels)  # fmt: skip
        if args.verb:
            print(f"[ffs_group] ETAC {side} FPR {goal:g}%: {int(union.sum())} voxels survive",
                  file=sys.stderr)  # fmt: skip
    record = {
        "n_iter": res.n_iter,
        "seed": args.seed,
        "nn": opts.nn,
        "sided": opts.sided,
        "min_clust": opts.min_clust,
        "hpow": list(opts.hpows),
        "pthr": list(opts.pthr),
        "zthr": [float(z) for z in opts.zthr],
        "cases": list(res.case_labels),
        "goals": [
            {
                "fpr_goal": t.fpr_goal,
                "fpr_achieved": t.fpr,
                "tfrac": t.tfrac,
                "thresholds": {
                    c: {f"{p:.4g}": {f"h{h}": float(t.thr[ic, ip, ih])
                                     for ih, h in enumerate(opts.hpows)}
                        for ip, p in enumerate(opts.pthr)}
                    for ic, c in enumerate(res.case_labels)
                },
            }
            for t in res.thresholds
        ],
    }  # fmt: skip
    Path(f"{stem}.ETAC.thresholds.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
