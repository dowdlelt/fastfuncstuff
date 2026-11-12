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
        default="DIFF",
        type=str.upper,
        help="Covariate centering: DIFF removes each set's own mean, SAME the "
        "mean over both sets, NONE nothing (the mean bricks are then the value "
        "at covariate = 0).",
    )
    test.add_argument("-toz", action="store_true", help="Write t statistics as z.")

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


def parse_set(items: list[str], default_label: str, label_flag: str | None):
    """``(set_label, [subject labels], [dataset specs])`` from a -setX value."""
    if not _exists(items[0]):
        # Long form: LABEL then subject/dataset pairs.
        label, rest = items[0], items[1:]
        if label_flag:
            raise ValueError(f"set {label!r} is in long form: drop -label{default_label[-1]}")
        if len(rest) < 2 or len(rest) % 2:
            raise ValueError(
                f"set {label!r}: long form needs 'LABEL subj dset subj dset ...' "
                f"(got {len(rest)} items after the label)"
            )
        subj, dsets = rest[0::2], rest[1::2]
        missing = [d for d in dsets if not _exists(d)]
        if missing:
            raise ValueError(f"set {label!r}: no such dataset {missing[0]!r}")
        return label, list(subj), list(dsets)
    missing = [d for d in items if not _exists(d)]
    if missing:
        raise ValueError(f"no such dataset {missing[0]!r}")
    return label_flag or default_label, [short_label(d) for d in items], list(items)


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
    from fastfuncstuff.stats.group_ttest import TtestDesign, run_ttest

    prefix = parse_prefix(args.prefix)
    out_path = Path(prefix.stem + prefix.nifti_ext)
    if out_path.exists() and not args.overwrite:
        raise ValueError(f"{out_path} exists (use -overwrite)")
    if (args.paired or args.unpooled) and not args.setB:
        raise ValueError("-paired and -unpooled need -setB")

    la, subj_a, dsets_a = parse_set(args.setA, "SetA", args.labelA)
    lb, subj_b, dsets_b = (None, [], [])
    if args.setB:
        lb, subj_b, dsets_b = parse_set(args.setB, "SetB", args.labelB)
        if lb == la:
            raise ValueError(f"both sets are labelled {la!r}")

    device = setup_device(args.device)
    vols_a = [load_volume(d) for d in dsets_a]
    vols_b = [load_volume(d) for d in dsets_b]
    ref_img = vols_a[0][1]
    shape = vols_a[0][0].shape
    for spec, (v, _) in zip(dsets_a + dsets_b, vols_a + vols_b, strict=True):
        if v.shape != shape:
            raise ValueError(f"{spec}: grid {v.shape} differs from {shape}")

    if args.mask:
        mask_vol, _ = load_volume(args.mask)
        if mask_vol.shape != shape:
            raise ValueError(f"mask grid {mask_vol.shape} differs from the data {shape}")
        mask = mask_vol != 0
    else:
        mask = np.zeros(shape, dtype=bool)
        for v, _ in vols_a + vols_b:
            mask |= v != 0
    n_vox = int(mask.sum())
    if n_vox == 0:
        raise ValueError("empty mask")

    def stack(vols):
        return torch.as_tensor(np.stack([v[mask] for v, _ in vols]), device=device)

    ya = stack(vols_a)
    yb = stack(vols_b) if vols_b else None

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
        center=args.center,
    )
    if args.verb:
        kind = (
            ("paired" if args.paired else "Welch" if args.unpooled else "two-sample")
            if design.two_sets
            else "one-sample"
        )
        cov = f", covariates {', '.join(names)} (center {args.center})" if names else ""
        print(
            f"[ffs_group] {kind}: {la} n={design.n_a}"
            + (f", {lb} n={design.n_b}" if design.two_sets else "")
            + f", {n_vox} voxels{cov}",
            file=sys.stderr,
        )

    bricks = run_ttest(design, ya, yb, toz=args.toz)

    out = np.zeros((*shape, len(bricks)), dtype=np.float32)
    for k, b in enumerate(bricks):
        out[..., k][mask] = b.data.detach().cpu().numpy()
    stataux = {k: b.stat for k, b in enumerate(bricks) if b.stat is not None}
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


if __name__ == "__main__":
    main()
