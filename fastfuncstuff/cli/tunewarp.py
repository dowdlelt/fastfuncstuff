"""CLI for searching nonlinear registration settings.

Command: ffs_tunewarp (registered as entry point in pyproject.toml)

Answers "what settings work for data like this", not "what is the best warp for
this one pair". You give it a few already-affine-aligned subjects, it fits every
backend across its own parameter grid, scores each result with functionals the
backend did not optimise plus a deformation-regularity check, and prints the
accuracy/smoothness trade-off you pick a row out of.

It is a dev tool for building defaults, so it is allowed to take a while — but
it is not allowed to fill your disk. Trial outputs are scored and deleted;
only the numbers and the exact command survive. When a row looks interesting,
``-reproduce N`` re-runs it on every subject and keeps the images, alongside the
base and the affine-aligned source, so you can actually look.

**A tuning directory has a long life.** The expected shape is not one big run:
it is a small run early in a study to get a direction, and more subjects folded
in later to sharpen it. So pointing a second invocation at the same ``-out``
picks up where the last one left off rather than starting over — the ladders are
rebuilt from the trials already recorded, screening goes to whichever subjects
the table knows least about (which puts a newly added brain first without being
told it is new), values that folded every time they were tried are dropped, and a
backend stops once the frontier stops moving. That last one makes ``-budget`` a
ceiling rather than a promise: asking for 300 fits on a space that is already
mapped costs about 60.

Resuming assumes the *engines* have not changed under the stored trials. The run
records its commit and says so up front when they differ; it is a warning rather
than a refusal, because only you can judge whether a given commit moved numbers.

Usage:
    # start a study (one base, many sources)
    ffs_tunewarp -type MNI_T1 -base mni.nii.gz -source s1.nii.gz s2.nii.gz \
                 -out tune_mni

    # later: two more subjects, same directory. Prior fits are reused; the new
    # brains are screened first; it stops when it stops learning.
    ffs_tunewarp -type MNI_T1 -base mni.nii.gz \
                 -source s1.nii.gz s2.nii.gz s3.nii.gz s4.nii.gz \
                 -out tune_mni -budget 200

    # read the answer
    ffs_tunewarp -out tune_mni -list        # ranked table + the frontier
    ffs_tunewarp -out tune_mni -bands       # what backing off the winner costs
    ffs_tunewarp -out tune_mni -effects     # what each knob does
    ffs_tunewarp -out tune_mni -importance  # which knobs to stop searching

    # once the corner is found: fill in the room beside it rather than chase it
    ffs_tunewarp -type MNI_T1 -base mni.nii.gz -source s1.nii.gz s2.nii.gz \
                 -out tune_mni -explore -patience 0 -budget 60

    # look at a row, on every subject
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

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_deterministic_arg,
    add_device_arg,
    add_verbose_arg,
    enable_determinism,
    setup_device,
)
from fastfuncstuff.processing.tunespec import BACKENDS, RECIPES, parse_fix, with_overrides
from fastfuncstuff.processing.tunestore import (
    TrialStore,
    format_bands,
    format_convergence,
    format_export,
    format_guide,
    format_importance,
    format_iteration_advice,
    format_knob_effects,
    format_level_gains,
    format_reproduce,
    format_results_table,
    format_resume,
    format_runs,
    knob_effects,
    knob_importance,
    recommend_iterations,
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
    parser = FfsArgumentParser(
        prog="ffs_tunewarp",
        description="Search nonlinear registration settings across backends and "
        "subjects, judged by functionals the backend did not optimise plus a "
        "deformation-regularity gate.",
        formatter_class=FfsHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument(
        "-out", required=True, metavar="DIR", help="Working directory (holds the trial table)"
    )
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
        default=2,
        help="Adaptive only: subjects a fresh candidate is tried on (default: 2). "
        "One is not enough: on a 7T epi2epi run the same config's rank moved a "
        "median of 10 places (and up to 43, of ~150) depending on which brain it "
        "met, so a single screen promotes and kills candidates by which subject "
        "came up. Two costs twice the screening -- which is the cheap half -- and "
        "buys a candidate that survived disagreement.",
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
    search.add_argument(
        "-note",
        default="",
        metavar="TEXT",
        help="One line describing what this data actually is, e.g. 'MP2RAGE 7T T1, "
        "0.8mm, 5 healthy adults'. Stored with the run and carried into -export and "
        "-guide. A preset is a claim that settings suit data of a KIND, and the "
        "shape and voxel size recorded automatically do not say which kind.",
    )
    search.add_argument(
        "-patience",
        type=int,
        default=3,
        help="Adaptive only: stop a backend after this many rounds fail to grow the "
        "accuracy/smoothness frontier by -tol (default: 3; 0 spends the whole budget). "
        "This makes -budget a ceiling rather than a promise, so asking for more fits "
        "than the space can use costs nothing.",
    )
    search.add_argument(
        "-tol",
        type=float,
        default=0.02,
        help="Adaptive only: relative growth in the frontier's dominated area, per "
        "round, that counts as progress (default: 0.02). A judgement dial, not a "
        "measured constant: nothing can tell you how much better is worth another ten "
        "fits. Raise it to stop sooner on a rough answer, lower it to keep refining.",
    )
    search.add_argument(
        "-explore",
        nargs="?",
        type=int,
        const=5,
        default=0,
        metavar="N",
        help="Adaptive only: spend the budget filling in the settings BESIDE the "
        "winner instead of chasing it. The score range from the best result down to "
        "the median is cut into N bands (default: 5 when the flag is given bare), "
        "and each round asks for the smoothest field that scores in one of them. "
        "Use it once a search has already found its corner: the default acquisition "
        "sweeps the accuracy/smoothness trade at a random weight and cannot be SENT "
        "anywhere, so it keeps re-measuring the end it likes and the less-aggressive "
        "warps beside the optimum stay thinly sampled. "
        "Pair it with -patience 0 -- filling a band adds a small frontier point at "
        "a time, which the default convergence test can read as no progress.",
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
        "-plot",
        nargs="?",
        const="",
        default=None,
        metavar="FILE.png",
        help="Write the accuracy/smoothness frontier as a scatter: bending energy "
        "against score, one marker per config (shape=backend, area=seconds per "
        "fit, colour=grade, the number inside is the id -reproduce takes). "
        "Written automatically after a search; pass this with -list to redraw an "
        "existing table, optionally to a path of your choosing "
        "[default: OUT/frontier.png]",
    )
    act.add_argument(
        "-no_plot",
        "-no-plot",
        action="store_true",
        help="Skip the frontier PNG a finished search would otherwise write",
    )
    act.add_argument(
        "-runs",
        action="store_true",
        help="What produced the trials in this directory -- code, machine and data -- "
        "and how the batches differ from each other.",
    )
    act.add_argument(
        "-importance",
        action="store_true",
        help="Rank the knobs by how much they actually moved the score, so the next "
        "run can pin the ones that did not.",
    )
    act.add_argument(
        "-guide",
        action="store_true",
        help="Emit the alignment recommendation this run supports, as a document: "
        "settings, whether nonlinear was worth it, timings and caveats.",
    )
    act.add_argument(
        "-export",
        action="store_true",
        help="Print the winning configs as Preset source to paste into tunespec, so "
        "the run's conclusion becomes the default that -type applies.",
    )
    act.add_argument(
        "-convergence",
        action="store_true",
        help="Per-level iteration report: what ceiling each backend actually needs, "
        "and whether it was starved or over-ran and fell back to an earlier iterate.",
    )
    act.add_argument(
        "-bands",
        nargs="?",
        type=int,
        const=5,
        default=0,
        metavar="N",
        help="Per backend, the SMOOTHEST setting found at each of N score levels "
        "(default: 5). The ranked table names the winner and the frontier names the "
        "choices worth making across backends; this names what backing off the "
        "winner costs and buys, for one backend at a time. A level holding a single "
        "config is one the search barely visited -- see -explore.",
    )
    act.add_argument(
        "-effects",
        action="store_true",
        help="Per-knob report: what each level scored, how often it folded, and "
        "whether every subject agrees. This is the output you build a default on.",
    )

    add_deterministic_arg(parser)

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
        "A study, start to finish:",
        "",
        "  # 1. early in the study: a few subjects, affine included",
        "  ffs_tunewarp -type MNI_T1 -allineate \\",
        "               -base MNI152_2009_template.nii.gz \\",
        "               -source sub-00{1,2,3}/SUMA/brain.nii.gz \\",
        "               -out tune_mni",
        "",
        "  # 2. read what it found",
        "  ffs_tunewarp -out tune_mni -list         # ranked configs + the frontier",
        "  ffs_tunewarp -out tune_mni -effects      # what each knob does",
        "  ffs_tunewarp -out tune_mni -importance   # which knobs never mattered",
        "",
        "  # 3. look at a row you like -- on every subject, next to its inputs",
        "  ffs_tunewarp -out tune_mni -reproduce 23",
        "",
        "  # 4. LATER: more subjects, same directory. Earlier fits are reused, the",
        "  #    new brains are screened first, and it stops when it stops learning,",
        "  #    so a generous -budget costs only what the space can actually use.",
        "  ffs_tunewarp -type MNI_T1 -allineate \\",
        "               -base MNI152_2009_template.nii.gz \\",
        "               -source sub-*/SUMA/brain.nii.gz \\",
        "               -out tune_mni -budget 200",
        "",
        "  # 5. spend the budget only on what you do not already know",
        "  ffs_tunewarp -type MNI_T1 ... -out tune_mni \\",
        "               -fix qwarp.minpatch=13 qwarp.hfactor_q=0.5",
        "",
        "  # 5b. exhaustive instead, when you want every cell of the factorial",
        "  ffs_tunewarp -type MNI_T1 ... -out tune_mni -search grid",
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
        "Reading the table:",
        "",
        "  Every functional in the jury improves with overfit, and the gate fails only",
        "  on FOLDING -- det(J) > 0 is a topology check, not a smoothness one. So the",
        "  top row is the loosest field that is still legal, never an optimum, and the",
        "  'score' column on its own cannot tell you otherwise.",
        "",
        "  'bend' (bending energy) and 'jacmin' are printed next to it for that reason,",
        "  and the FRONTIER table below the ranking is the one to read: it lists the",
        "  configs nothing else beats on both similarity and smoothness, smoothest",
        "  first. Walk up it and stop where the score stops being worth the roughness.",
        "  A 10x jump in 'bend' for a few hundredths of score is the ranking paying for",
        "  detail that is not anatomy.",
        "",
        "  PINNED means the warp came back resting on the solver's own anti-fold floor,",
        "  so the damping is what kept it legal rather than its regularization. Those",
        "  are demoted and kept off the frontier: the same settings fold outright on a",
        "  subject where the guard cannot hold.",
        "",
        "  Mind the 'n' column. Screening deliberately spends few fits on most",
        "  candidates, so a row measured on one brain and one measured on four are not",
        "  comparable; -reproduce a row before believing it.",
        "",
        "  -effects is the safer read for a knob: it separates 'how good when it works'",
        "  from 'how often does it work'.",
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


def _write_plot(store, path, recipe: str | None) -> None:
    """Draw the frontier, and let a missing matplotlib be a note rather than a failure.

    A run that produced a good table and no picture is still a successful run, so
    this never raises: the fits are the expensive part and they are already done.
    """
    from ..processing.tuneplot import plot_frontier

    try:
        written = plot_frontier(store.results(), path, recipe=recipe or "")
    except ImportError:
        print("  (no frontier plot: matplotlib is not installed)")
        return
    if written:
        print(f"\nFrontier plot: {written}")
    else:
        print("  (no frontier plot: no config in this table produced a warp)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if getattr(args, "deterministic", False):
        enable_determinism(getattr(args, "verb", 1))
    out = Path(args.out)
    store = TrialStore(out / "trials.json")

    if (
        args.list
        or args.plot is not None
        or args.effects
        or args.bands
        or args.convergence
        or args.export
        or args.runs
        or args.importance
        or args.guide
    ):
        if args.recipe:
            store.compute_consensus(RECIPES[args.recipe].panel())
        if args.runs:
            print(format_runs(store))
        if args.importance:
            print(format_importance(knob_importance(store)))
        if args.guide:
            if not args.recipe:
                raise SystemExit("-guide needs -type, since a recommendation is per recipe")
            print(format_guide(store, args.recipe))
        if args.effects:
            print(format_knob_effects(knob_effects(store)))
        if args.bands:
            print(format_bands(store.results(), args.bands))
        if args.export:
            if not args.recipe:
                raise SystemExit("-export needs -type, since a preset is keyed by recipe")
            print(format_export(store, args.recipe))
        if args.convergence:
            print(format_level_gains(store))
            print()
            print(format_iteration_advice(recommend_iterations(store)))
            print()
            print(format_convergence(store))
        if args.list:
            print(format_results_table(store.results(), limit=args.top))
        if args.plot is not None:
            _write_plot(store, args.plot or out / "frontier.png", args.recipe)
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

    store.begin_run(
        device=str(device),
        recipe=recipe.name,
        contrast=recipe.contrast,
        optimize=recipe.optimize,
        panel=recipe.panel(),
        search=args.search,
        note=args.note,
    )

    # After begin_run, so `warnings()` can compare the earlier runs against the one
    # about to happen -- and before any fitting, because "these trials were scored by
    # a different build" is worth knowing while it can still change your mind.
    if args.verb >= 1 and (resume := format_resume(store, [p.name for p in pairs])):
        print("\n" + resume)
        for w in store.warnings():
            print(f"  ! {w}")

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
            patience=args.patience,
            tol=args.tol,
            explore=args.explore,
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
    if (warn := store.warnings()) and args.verb < 1:
        # Already said before the fits when verbose; repeated here only for a quiet
        # run, where this is the first and last chance to say it.
        print("\nThis directory holds earlier runs that may not be comparable:")
        for w in warn:
            print(f"  - {w}")
    print("\n" + format_results_table(store.results(), limit=args.top))
    if not args.no_plot:
        _write_plot(store, args.plot or out / "frontier.png", recipe.name)
    print("\nPer-knob effects:\n")
    print(format_knob_effects(knob_effects(store)))
    print("\nIteration ceilings:\n")
    print(format_iteration_advice(recommend_iterations(store)))
    print("\nConvergence:\n")
    print(format_convergence(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
