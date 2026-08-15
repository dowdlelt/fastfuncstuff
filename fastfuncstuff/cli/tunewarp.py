"""CLI for searching nonlinear registration settings.

Command: ffs_tunewarp (registered as entry point in pyproject.toml)

Answers "what settings work for data like this", not "what is the best warp for
this one pair". You give it a few already-affine-aligned subjects, it fits every
backend across its own parameter grid, scores each result with functionals the
backend did not optimise plus a deformation-regularity check, and prints a
ranked table you pick a number out of.

It is a dev tool for building defaults, so it is allowed to take a while — but
it is not allowed to fill your disk. Trial outputs are scored and deleted;
only the numbers and the exact command survive. When a row looks interesting,
``-reproduce N`` re-runs it and keeps the images so you can actually look.

Usage:
    # search (one base, many sources)
    ffs_tunewarp -type MNI_T1 -base mni.nii.gz -source s1.nii.gz s2.nii.gz s3.nii.gz \\
                 -out tune_mni

    # look at what has been tried
    ffs_tunewarp -out tune_mni -list

    # rebuild config 7's warps and keep them
    ffs_tunewarp -out tune_mni -reproduce 7

Sources must be affine-aligned to the base. Pass ``-allineate`` to have that
done as step 0 (cached in {out}/affine/, so re-runs skip it), or pre-align them
yourself with ffs_allineate. Either way it is a one-time cost per subject and is
deliberately not part of the search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device
from fastfuncstuff.processing.tunespec import BACKENDS, RECIPES, parse_fix, with_overrides
from fastfuncstuff.processing.tunestore import (
    TrialStore,
    format_convergence,
    format_knob_effects,
    format_reproduce,
    format_results_table,
    knob_effects,
)
from fastfuncstuff.processing.tunewarp import (
    AdaptivePlan,
    SubjectPair,
    affine_align,
    enumerate_configs,
    reproduce,
    run_adaptive,
    run_search,
)
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_tunewarp",
        description="Search nonlinear registration settings across backends and "
        "subjects, judged by functionals the backend did not optimise plus a "
        "deformation-regularity gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument("-out", required=True, help="Working directory (holds the trial table)")
    parser.add_argument(
        "-type",
        dest="recipe",
        choices=sorted(RECIPES),
        default=None,
        help="Recipe: what to tune and how to judge it, for a kind of registration",
    )
    parser.add_argument("-base", nargs="+", default=None, help="Base/target image(s)")
    parser.add_argument("-source", nargs="+", default=None, help="Affine-aligned source image(s)")
    parser.add_argument(
        "-backend",
        nargs="+",
        default=None,
        choices=sorted(BACKENDS),
        help="Restrict the search to these backends (default: all of them)",
    )
    search = parser.add_argument_group("Search strategy")
    search.add_argument(
        "-search",
        choices=("adaptive", "grid"),
        default="adaptive",
        help="adaptive (default): a surrogate proposes settings, each is screened "
        "on one subject and only survivors are confirmed on more, and a range is "
        "extended when the best setting sits on its end. grid: the full factorial, "
        "every config on every subject — exhaustive, reproducible, and much slower.",
    )
    search.add_argument(
        "-budget",
        type=int,
        default=60,
        help="Adaptive only: fits to spend per backend (default: 60)",
    )
    search.add_argument(
        "-screen",
        type=int,
        default=1,
        help="Adaptive only: subjects a fresh candidate is tried on (default: 1)",
    )
    search.add_argument(
        "-confirm",
        type=int,
        default=2,
        help="Adaptive only: further subjects a promising candidate earns (default: 2)",
    )
    search.add_argument(
        "-batch",
        type=int,
        default=4,
        help="Adaptive only: candidates proposed per surrogate refit (default: 4)",
    )
    search.add_argument(
        "-no_expand",
        "-no-expand",
        action="store_true",
        help="Adaptive only: keep every knob inside its listed range. Off by "
        "default because an optimum on a range edge means the range is wrong.",
    )
    search.add_argument("-seed", type=int, default=0, help="Adaptive only: RNG seed (default: 0)")
    search.add_argument(
        "-max_configs",
        type=int,
        default=None,
        help="Cap the configs tried per backend. For a smoke test — a real run "
        "wants the full grid, since the point is that nothing is eliminated "
        "before it has been tried.",
    )
    parser.add_argument(
        "-allineate",
        action="store_true",
        help="Run the affine alignment as step 0, once per subject, instead of "
        "requiring pre-aligned sources. Uses the recipe's cost (lpa/lpc) and "
        "caches into {out}/affine/, so a re-run skips it.",
    )
    parser.add_argument(
        "-fix",
        nargs="+",
        default=None,
        metavar="KEY=VAL",
        help="Pin a knob instead of searching it, e.g. -fix formwarp.total_var=1.0. "
        "Spends the budget on what you do not already know.",
    )
    parser.add_argument(
        "-tune",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Search a knob the recipe leaves alone, e.g. -tune formwarp.iters",
    )
    parser.add_argument(
        "-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Kill any single fit that exceeds this many seconds",
    )

    act = parser.add_argument_group("Inspecting results")
    act.add_argument("-list", action="store_true", help="Print the table and exit")
    act.add_argument(
        "-reproduce",
        type=int,
        default=None,
        metavar="N",
        help="Re-run config N and KEEP its outputs, so they can be looked at",
    )
    act.add_argument("-top", type=int, default=25, help="Rows to show (default: 25)")
    act.add_argument(
        "-convergence",
        action="store_true",
        help="Per-level iteration report: whether each backend was starved of "
        "iterations or over-ran and fell back to an earlier iterate.",
    )
    act.add_argument(
        "-effects",
        action="store_true",
        help="Per-knob report: what each level scored, how often it folded, and "
        "whether every subject agrees. This is the output you build a default on.",
    )

    add_device_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args(argv)


def _epilog() -> str:
    lines = ["Recipes:", ""]
    for name, r in sorted(RECIPES.items()):
        lines.append(f"  {name:10s} {r.describe}")
        lines.append(f"  {'':10s}   optimize={r.optimize}, pairing={r.pairing}")
        if r.notes:
            for chunk in _wrap(r.notes, 66):
                lines.append(f"  {'':10s}   {chunk}")
        lines.append("")
    lines += [
        "Examples (MNI tuning, start to finish):",
        "",
        "  # 1. search everything, affine included (-allineate does step 0 for you)",
        "  ffs_tunewarp -type MNI_T1 -allineate \\",
        "               -base MNI152_2009_template.nii.gz \\",
        "               -source sub-*/SUMA/brain.nii.gz \\",
        "               -out tune_mni",
        "",
        "  # 2. or spend the budget only on what you do not already know",
        "  ffs_tunewarp -type MNI_T1 ... -out tune_mni \\",
        "               -fix qwarp.minpatch=13 qwarp.hfactor_q=0.5",
        "",
        "  # 2b. exhaustive instead, when you want every cell of the factorial",
        "  ffs_tunewarp -type MNI_T1 ... -out tune_mni -search grid",
        "",
        "  # 3. read the answer",
        "  ffs_tunewarp -out tune_mni -effects      # what each knob does",
        "  ffs_tunewarp -out tune_mni -list         # ranked configs",
        "",
        "  # 4. look at the winner",
        "  ffs_tunewarp -out tune_mni -reproduce 23",
        "",
        "Pairing:",
        "  one_base   -base ONE -source A B C     (each source vs the same base)",
        "  paired     -base A B C -source a b c   (paired BY POSITION, never sorted)",
        "",
        "The table's score is a mean rank across cost functionals, computed within",
        "each subject and averaged. Lower is better. A PASS config always outranks",
        "a MARGINAL one and a MARGINAL always outranks a FAIL, whatever the score:",
        "a better similarity number never buys its way past a folded warp.",
        "",
        "The jury is chosen, not just filtered. A recipe's contrast regime decides",
        "which functionals are meaningful at all -- the signed ones (lss/lpc/lpc+)",
        "reward anti-correlation, so on same-modality data they rank the WORST warp",
        "first -- and excluding the optimised cost excludes its whole family, since",
        "lpa/lpa+/lpc/lpc+ are all the same number wearing different signs.",
        "",
        "Expect the ranking to sit on the fold boundary. Less regularization always",
        "scores better, because folding is what buys the score, so the top row is",
        "normally the least regularization that is still legal rather than an",
        "optimum. -effects is the safer read: it separates 'how good when it works'",
        "from 'how often does it work'.",
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def _strip_ext(path: str) -> str:
    name = Path(path).name
    for ext in (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def subject_names(paths: list[str]) -> list[str]:
    """Short, **unique** labels for a set of source paths.

    Uniqueness is not cosmetic. Consensus ranks are computed within a subject, so
    two subjects sharing a label would be pooled and ranked against each other as
    if they were the same brain, and `-reproduce` would write one over the other.

    Filenames first (``sub-01_T1w.nii.gz`` -> ``sub-01_T1w``); the parent
    directory is folded in only when filenames collide, which is the FreeSurfer
    layout where every subject owns a ``brain.nii.gz``.
    """
    stems = [_strip_ext(p) for p in paths]
    if len(set(stems)) == len(stems):
        return stems

    # Keep only the path components that actually differ between subjects. The
    # FreeSurfer layout is sub-XXXX/SUMA/brain.nii.gz, where both the filename
    # and its immediate parent are identical for everyone — looking one level up
    # is not enough, and the distinguishing component can be any depth away.
    comps = [[*Path(p).parts[:-1], s] for p, s in zip(paths, stems, strict=True)]
    width = max(len(c) for c in comps)
    aligned = [[""] * (width - len(c)) + c for c in comps]  # right-align
    differing = [i for i in range(width) if len({row[i] for row in aligned}) > 1]
    if differing:
        names = ["_".join(row[i] for i in differing if row[i]) for row in aligned]
        if len(set(names)) == len(names):
            return names

    # Pathological (the same path twice) — stay unique anyway.
    return [f"{s}#{i}" for i, s in enumerate(stems)]


def _build_pairs(args: argparse.Namespace, pairing: str) -> list[SubjectPair]:
    """Turn -base/-source into subject pairs, per the recipe's pairing rule."""
    bases, sources = args.base or [], args.source or []
    if not bases or not sources:
        raise SystemExit("-base and -source are required to run a search")
    names = subject_names(sources)

    if pairing == "one_base":
        if len(bases) != 1:
            raise SystemExit(
                f"recipe pairing is 'one_base' but {len(bases)} bases were given. "
                "Pass one -base and many -source."
            )
        return [SubjectPair(n, bases[0], s) for n, s in zip(names, sources, strict=True)]

    if len(bases) != len(sources):
        raise SystemExit(
            f"recipe pairing is 'paired' but got {len(bases)} bases and "
            f"{len(sources)} sources. They pair BY POSITION, so the counts must match."
        )
    return [SubjectPair(n, b, s) for n, b, s in zip(names, bases, sources, strict=True)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    store = TrialStore(out / "trials.json")

    if args.list or args.effects or args.convergence:
        if args.recipe:
            store.compute_consensus(RECIPES[args.recipe].panel())
        if args.effects:
            print(format_knob_effects(knob_effects(store)))
        if args.convergence:
            print(format_convergence(store))
        if args.list:
            print(format_results_table(store.results(), limit=args.top))
        return 0

    if args.reproduce is not None:
        if not store.trials:
            raise SystemExit(f"no trials recorded in {store.path}")
        print(format_reproduce(store, args.reproduce, out / "kept"))
        written = reproduce(store, args.reproduce, out, timeout=args.timeout, verb=args.verb)
        if written:
            print(f"\nKept {len(written)} output(s) under {out / 'kept'}")
        return 0

    if not args.recipe:
        raise SystemExit("-type is required to run a search (see -help for recipes)")
    fixed = parse_fix(args.fix or [])
    recipe = with_overrides(RECIPES[args.recipe], fixed, args.tune)
    pairs = _build_pairs(args, recipe.pairing)
    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    backends = args.backend or list(recipe.backends)

    if args.verb >= 1:
        panel = recipe.panel()
        print(f"Recipe {recipe.name}: {recipe.describe}")
        print(f"  optimize={recipe.optimize}, contrast={recipe.contrast}")
        print(f"  judged by {len(panel)} functional(s): {', '.join(panel)}")
        if args.search == "adaptive":
            total = args.budget * len(backends)
            print(
                f"  {len(pairs)} subject(s), {len(backends)} backend(s), "
                f"adaptive, <= {total} fits "
                f"(screen {args.screen}, confirm {args.confirm})"
            )
        else:
            total = sum(
                len(enumerate_configs(recipe, b, args.max_configs, fixed)) * len(pairs)
                for b in backends
            )
            print(f"  {len(pairs)} subject(s), {len(backends)} backend(s), grid, {total} fits")
            # The recipes tune the iteration schedule as well as the regularization,
            # and a full factorial over both is thousands of configs per backend. Say
            # so up front rather than after a day of fitting.
            if total > 2000:
                print(
                    f"\n  WARNING: {total} fits at ~10 s each is roughly "
                    f"{total * 10 / 3600:.0f} hours. The full factorial is no longer a "
                    "practical\n  way to search this space. Use -search adaptive "
                    "(the default), or narrow the\n  grid with -fix / -max_configs."
                )
        if fixed:
            print("  pinned: " + ", ".join(f"{k}={v}" for k, v in sorted(fixed.items())))
        print(f"  trial outputs are discarded after scoring; table in {store.path}")

    if args.allineate:
        if args.verb >= 1:
            print(f"\nStep 0: affine ({recipe.optimize}) for {len(pairs)} subject(s)")
        pairs = affine_align(pairs, recipe, out, device=device, verb=args.verb)

    if args.search == "adaptive":
        plan = AdaptivePlan(
            budget=args.budget,
            screen=args.screen,
            confirm=args.confirm,
            batch=args.batch,
            expand=not args.no_expand,
            seed=args.seed,
        )
        run_adaptive(
            pairs,
            recipe,
            store,
            plan,
            backends=backends,
            fixed=fixed,
            device=device,
            verb=args.verb,
        )
    else:
        run_search(
            pairs,
            recipe,
            store,
            backends=backends,
            max_configs=args.max_configs,
            fixed=fixed,
            device=device,
            verb=args.verb,
        )

    store.compute_consensus(recipe.panel())
    store.save()
    print("\n" + format_results_table(store.results(), limit=args.top))
    print("\nPer-knob effects:\n")
    print(format_knob_effects(knob_effects(store)))
    print("\nConvergence:\n")
    print(format_convergence(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
