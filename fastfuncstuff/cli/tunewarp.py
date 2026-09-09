"""CLI for searching nonlinear registration settings.

Command: ffs_tunewarp (registered as entry point in pyproject.toml)

Answers "what settings work for data like this", not "what is the best warp for
this one pair". It fits every backend across its own parameter grid, scores each
result with evidence the backend did not optimise, and prints the
accuracy/smoothness trade-off you pick a row out of.

**What counts as evidence is the choice that matters**, and there are three
answers, in increasing order of how much they can be trusted and how much they
cost:

* **Image similarity** (`-type MNI_T1`, `epi2t1`, `epi2epi`). A jury of cost
  functionals the fit did not use. Cheap, always available, and structurally
  limited: every one of them improves monotonically with overfit, and the
  regularity gate fails only on folding, so the top row is by construction the
  loosest field that has not yet inverted.
* **Manual segmentations, pairwise** (`-type cohort_T1 -cohort DIR`). Subject
  A's tracing carried through the warp and compared against subject B's own
  tracing. The referee is a human, and it has no obligation to reward overfit.
* **Manual segmentations in one common space** (`-type common_T1 -cohort DIR
  -base template`). Cross-subject label agreement, which is the quantity a group
  analysis actually needs. N fits per config rather than N(N-1) -- but both sides
  move when the settings change, so read it alongside the pairwise mode rather
  than instead of it.

Two things are load-bearing and easy to miss.

**`-holdout` is on by default.** Everything else in the table is in-sample: the
surrogate proposed those configs *because* of how they scored on those brains.
A quarter of the subjects are reserved, the search never sees them, and the
finalists are fit on them once at the end. Split by subject, never by pair.

**Settings are in millimetres.** Smoothing sigmas, patch sizes and step caps are
searched and stored in mm and converted to voxels only at the engine and the
printed command, so a study at 0.7 mm and a study at 1 mm are comparable and a
preset transfers between them.

It is a dev tool for building defaults, so it is allowed to take a while -- but
it is not allowed to fill your disk. Trial outputs are scored and deleted; only
the numbers and the exact command survive. When a row looks interesting,
``-reproduce N`` re-runs it keeping the images, and for a common-space run
``-diagnostics N`` writes the overlap-probability volume and the per-label
tables behind its score.

``-diag_only NAME`` runs those same diagnostics on segmentations *another tool*
already warped into the common space, so a head-to-head against AFNI, ANTs or
FSL is measured by the same instrument rather than by two papers' worth of
argument that the instruments matched. ``-collect`` concatenates every method's
tables for pandas.

**A tuning directory has a long life.** The expected shape is not one big run:
it is a small run early in a study to get a direction, and more subjects folded
in later to sharpen it. So pointing a second invocation at the same ``-out``
picks up where the last one left off -- ladders rebuilt from the trials already
recorded, screening sent to whichever subjects the table knows least about, the
pair window slid so a resumed cohort study covers new ground, values that folded
every time dropped, and held-out fits already recorded skipped.

Resuming assumes the *engines* have not changed under the stored trials. The run
records its commit and says so up front when they differ; it also says so if the
held-out split moved, which would quietly stop those numbers being out-of-sample.

Sources must be affine-aligned to the base. Pass ``-allineate`` to have that done
as step 0 (cached in {out}/affine/, matrices included so a segmentation reaches
the base grid in one gather), or pre-align them yourself with ffs_allineate.
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
from fastfuncstuff.processing.cohort import (
    TEST,
    TRAIN,
    describe_cohort,
    discover_cohort,
    pairwise,
    panel_size,
    rotation_start,
    split_subjects,
)
from fastfuncstuff.processing.tunespec import BACKENDS, RECIPES, parse_fix, with_overrides
from fastfuncstuff.processing.tunestore import (
    TrialStore,
    format_bands,
    format_convergence,
    format_export,
    format_guide,
    format_holdout,
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
    DIAGNOSTIC_METRICS,
    AdaptivePlan,
    SubjectPair,
    affine_align,
    collect_diagnostics,
    diagnose_warped,
    enumerate_configs,
    evaluate_holdout,
    group_diagnostics,
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

    coh = parser.add_argument_group("Cohort (pairwise, judged on segmentations)")
    coh.add_argument(
        "-cohort",
        metavar="DIR",
        default=None,
        help="A directory of subjects to fit PAIRWISE instead of -base/-source. "
        "Each image is paired with its segmentation (see -label_suffix) and every "
        "pair is a real test: subject A's tracing is carried through the warp and "
        "compared against subject B's own tracing, so the referee is a human "
        "rather than the intensities the fit was driven by.",
    )
    coh.add_argument(
        "-label_suffix",
        "-label-suffix",
        default="_seg",
        metavar="SUFFIX",
        help="How a segmentation is named beside its image (default: _seg, i.e. "
        "na01.nii.gz -> na01_seg.nii.gz). A subject with no match is still fit, "
        "and simply cannot be judged on anatomy.",
    )
    coh.add_argument(
        "-pairs",
        default=None,
        metavar="N|all",
        help="Pairs the search may draw on, per run [default: budget/2]. Chosen "
        "round-robin, so every subject appears equally often as a base and as a "
        "source and no single unusual brain can dominate a small panel; the window "
        "slides half a panel each run, so a resumed study reuses half its pairs "
        "(sharpening what they measure) and draws half fresh (so the settings are "
        "not tuned to one fixed set of brains). The default is arithmetic, not "
        "taste: the search compares settings by z-scoring each fit against the "
        "other fits on the SAME pair, which needs at least two, and total fits "
        "equal the budget -- so a pool above budget/2 makes a growing share of "
        "your fits invisible to the search. 'all' is every ordered pair.",
    )
    coh.add_argument(
        "-holdout",
        type=float,
        default=0.25,
        metavar="FRAC|N",
        help="Subjects reserved from the search entirely (default: 0.25; 0 disables). "
        "The search never sees them, and after it finishes the settings it chose "
        "are fit on them once -- which is the only number in the table that is not "
        "in-sample. Split by SUBJECT, never by pair: pairs A->B and A->C share a "
        "brain, so a held-out pair reusing a training subject measures a setting "
        "that was partly chosen on that same brain.",
    )
    coh.add_argument(
        "-holdout_configs",
        "-holdout-configs",
        type=int,
        default=5,
        metavar="N",
        help="Finalists re-fit on the held-out subjects (default: 5). Deliberately "
        "few: a held-out set used to CHOOSE among many candidates stops being held "
        "out. It answers 'does the chosen setting transfer', not 'which is best'.",
    )
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
    act.add_argument(
        "-diagnostics",
        "-diag",
        type=int,
        default=None,
        metavar="N",
        help="Common-space runs: re-fit config N on the cohort and write everything "
        "behind its score into {out}/diag/configN/ -- a 4D overlap-probability "
        "volume with one frame per label (the AFNI-style picture: bright in a "
        "parcel's core, fading at its edge, and the width of that fade is the "
        "registration's real error bar), the agreement and consensus-label maps, "
        "and per-label / per-pair / per-subject tables. Use it on the winner.",
    )
    act.add_argument(
        "-diag_only",
        "-diag-only",
        default=None,
        metavar="NAME",
        help="Score labels somebody ELSE already warped into the common space, and "
        "call the result NAME (e.g. -diag_only 'AFNI 3dQwarp'). No fitting: point "
        "-cohort at a directory of segmentations another tool produced, and it "
        "writes the same tables and the same overlap volume an ffs config gets. "
        "That sameness is the point -- a tool comparison is only worth reading if "
        "the instrument is identical on both sides, and here that is not an "
        "argument but the same function. Combine with -collect for the head-to-head.",
    )
    act.add_argument(
        "-metrics",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Image functionals to record beside the label scores in a diagnostics "
        f"run [default: {' '.join(DIAGNOSTIC_METRICS)}]. 'all' takes every metric "
        "meaningful for the contrast. With -diag_only these need -base and the "
        "warped IMAGES beside the labels: we rank on the labels, but the other "
        "tools were tuned on these, so a comparison that showed only Dice would be "
        "answering a different question from the one they were optimising.",
    )
    act.add_argument(
        "-warp_suffix",
        "-warp-suffix",
        default=None,
        metavar="SUFFIX",
        help="With -diag_only, also read each method's own displacement field "
        "(e.g. -warp_suffix _WARP finds na01_WARP.nii.gz) and grade it for "
        "folding, compression and bending with the SAME code that grades ours. "
        "This is how 'how much deformation is normal?' becomes a measurement: the "
        "bounds in warpqc are thresholds on a continuum, and only warps the field "
        "already accepts can say where that continuum sits.",
    )
    act.add_argument(
        "-warp_units",
        "-warp-units",
        choices=("mm", "voxel"),
        default="mm",
        help="Units of the fields read by -warp_suffix (default: mm, which is what "
        "AFNI and ANTs write; ours are voxel). Check jac_neg_frac in the output -- "
        "a field read under the wrong convention reports implausible folding "
        "rather than failing, so that column is the check that it was understood.",
    )
    act.add_argument(
        "-method",
        default=None,
        metavar="NAME",
        help="Name this result carries in the tables [default: 'ffs <backend> "
        "c<id>' for -diagnostics, the -diag_only argument otherwise]. It is the "
        "column a head-to-head pivots on, so every row you want to compare must "
        "have a distinct one.",
    )
    act.add_argument(
        "-collect",
        action="store_true",
        help="Concatenate every method's tables under {out}/diag/ into all_*.tsv, "
        "ready for pandas. Each row already carries a 'method' column, so this is a "
        "concatenation, not a join: a method with a different label set or a missing "
        "subject just contributes the rows it has.",
    )
    act.add_argument(
        "-save_subject_labels",
        "-save-subject-labels",
        action="store_true",
        help="With -diagnostics, also write each subject's transported segmentation",
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
        judged = "labels" if r.labels else "intensities"
        shape = "whole cohort per config" if r.group else "one pair at a time"
        lines.append(f"  {name:10s} {r.describe}")
        lines.append(f"  {'':10s}   optimize={r.optimize}, pairing={r.pairing}")
        lines.append(f"  {'':10s}   judged on {judged}, scored {shape}")
        if r.notes:
            for chunk in _wrap(r.notes, 66):
                lines.append(f"  {'':10s}   {chunk}")
        lines.append("")
    lines += [
        "THREE WAYS TO USE THIS",
        "",
        "1. Tune against a target, judged on image similarity (the original mode)",
        "",
        "  ffs_tunewarp -type MNI_T1 -allineate -base template.nii.gz \\",
        "               -source sub-00{1,2,3}/brain.nii.gz -out tune_mni",
        "",
        "2. Tune against manual segmentations, PAIRWISE within a labelled cohort",
        "",
        "  ffs_tunewarp -type cohort_T1 -cohort labelled_brains/ -out tune_pairs",
        "",
        "  Each pair is a real test: subject A's tracing carried through the warp,",
        "  compared against subject B's OWN tracing. The referee is a human, not the",
        "  intensities the fit was driven by -- which matters because every intensity",
        "  functional improves monotonically with overfit and the gate only catches",
        "  folding, so a similarity ranking always crowns the loosest legal field.",
        "",
        "3. Tune (or evaluate) a whole cohort in ONE common space",
        "",
        "  ffs_tunewarp -type common_T1 -allineate -cohort labelled_brains/ \\",
        "               -base MNI152_2009_template.nii.gz -out tune_mni",
        "",
        "  N fits per config instead of N(N-1), scoring cross-subject label",
        "  agreement -- which is what a group analysis actually needs. Read it",
        "  ALONGSIDE mode 2, not instead of it: both sides move when the settings",
        "  change, so a config that drives every brain harder onto the template can",
        "  raise this by making the errors agree rather than by making them small.",
        "",
        "COHORT LAYOUT",
        "",
        "  -cohort DIR expects  na01.nii.gz + na01_seg.nii.gz  per subject",
        "  (-label_suffix changes '_seg'). A subject with no tracing is still fit;",
        "  it simply cannot be judged on anatomy.",
        "",
        "HELD-OUT VALIDATION  (-holdout, on by default at 0.25)",
        "",
        "  Everything in the search table is IN-SAMPLE: the surrogate proposed those",
        "  configs because of how they scored on those brains, and the ladders grew",
        "  toward them. -holdout reserves subjects the search never sees, and after",
        "  it finishes the finalists are fit on them once. That is the only number",
        "  in the output that is not in-sample.",
        "",
        "  Split by SUBJECT, never by pair -- pairs A->B and A->C share a brain, so",
        "  a held-out pair reusing a training subject measures a setting that was",
        "  partly chosen on that same brain. Membership is a hash threshold, so",
        "  adding subjects later leaves everyone on the side they were already on.",
        "",
        "  Read the gap, not the ordering, and read it against the BASELINE's own",
        "  shift: that row involves no settings, so it measures how much harder the",
        "  held-out brains are. Anything beyond it is optimism.",
        "",
        "DIAGNOSTICS: what is behind one config's score",
        "",
        "  ffs_tunewarp -out tune_mni -type common_T1 -diagnostics 14 -allineate \\",
        "               -cohort brains/ -base MNI152_2009_template.nii.gz",
        "",
        "  Re-fits config 14 on the WHOLE cohort and writes a 4D overlap-probability",
        "  volume (one frame per label: bright in a parcel's core, fading at its edge",
        "  -- that fade is the registration's error bar), agreement and consensus-",
        "  label maps, and per-label / per-pair / per-subject tables.",
        "",
        "HEAD-TO-HEAD AGAINST AFNI, ANTs, FSL, SPM",
        "",
        "  Warp the cohort with the other tool, apply its warp to that subject's",
        "  segmentation, and drop the results in one directory per tool:",
        "",
        "     ants_out/                     what it is            needed for",
        "       na01.nii.gz                 warped image          lpa/mi/lncc",
        "       na01_seg.nii.gz             warped labels         Dice, overlap",
        "       na01_Warp.nii.gz            the displacement      folding, det(J), bend",
        "       na02.nii.gz  ...            (same three per subject)",
        "",
        "  Names must agree on the subject stem: <subj>, <subj>_seg, <subj><warp>.",
        "  Labels alone are enough; the image and the warp each unlock more columns.",
        "",
        "  ffs_tunewarp -out cmp -diag_only 'ANTs SyN' -cohort ants_out/ \\",
        "               -base MNI152_2009_template.nii.gz \\",
        "               -warp_suffix _Warp -warp_units mm",
        "",
        "  ffs_tunewarp -out cmp -diag_only 'AFNI 3dQwarp' -cohort afni_out/ \\",
        "               -base MNI152_2009_template.nii.gz \\",
        "               -warp_suffix _WARP -warp_units mm",
        "",
        "  Then put OUR winners on the same axes -- note -method, or they collide",
        "  in the one column the comparison pivots on:",
        "",
        "  ffs_tunewarp -out cmp -type common_T1 -diagnostics 89 -allineate \\",
        "               -cohort brains/ -base MNI152_2009_template.nii.gz \\",
        "               -method 'ffs optiwarp_gradient'",
        "",
        "  ffs_tunewarp -out cmp -collect        # -> cmp/diag/all_*.tsv",
        "",
        "     import pandas as pd",
        "     df = pd.read_csv('cmp/diag/all_summary.tsv', sep='\\t')",
        "     df.plot.scatter('bend_max', 'dice_mean')     # every tool, one axis",
        "",
        "  SAME SUBJECTS, ALWAYS. Cross-subject Dice is a mean over pairs of one",
        "  particular set, so a method scored on 16 brains cannot be compared with",
        "  one scored on 10. -diagnostics re-runs on the whole cohort for exactly",
        "  this reason; scores lifted out of the tuning table are 'train' or",
        "  'held out' subsets and are NOT comparable to an external run.",
        "",
        "  -warp_units is mm for AFNI and ANTs, voxel for ours. Check jac_neg_frac:",
        "  a field read under the wrong convention reports implausible folding",
        "  instead of failing, so that column is the check that it was understood.",
        "",
        "WHAT THE COMPARISON CAN SETTLE",
        "",
        "  Only FOLDING is known to be wrong -- tissue cannot turn inside out. The",
        "  compression bound (1st-pct det(J) < 0.25) is a convention, and bending",
        "  energy has no bound at all; both are printed because they describe a warp,",
        "  not because a number has been shown to be too big. Warping different",
        "  brains onto one template is a large-deformation problem, so more",
        "  deformation may simply be the job.",
        "",
        "  Scoring warps the field already accepts is what turns those conventions",
        "  into measurements. If ANTs and AFNI sit where a setting of ours sits,",
        "  that setting is ordinary.",
        "",
        "UNITS: MILLIMETRES, NOT VOXELS",
        "",
        "  Smoothing sigmas, patch sizes and step caps are searched, stored and",
        "  exported in mm, and converted to voxels for the engine and the printed",
        "  command. A ladder learned at 1 mm would otherwise ask a different",
        "  question at 0.7 mm -- minpatch 13 is a 13 mm patch on one dataset and a",
        "  9.1 mm patch on the other, and only one of those is the anatomical scale",
        "  the finding was about.",
        "",
        "RESUMING",
        "",
        "  Pointing a second run at the same -out picks up where the last one left",
        "  off: ladders rebuilt from the stored trials, screening sent to whichever",
        "  subjects the table knows least about, values that always folded dropped,",
        "  and held-out fits already recorded skipped. For a pairwise cohort the",
        "  pair window also slides half a panel, so a resumed study covers new",
        "  ground instead of re-measuring one fixed set of brains.",
        "",
        "  It assumes the ENGINES have not changed under the stored trials. The run",
        "  records its commit and warns when they differ -- and warns if the",
        "  held-out split moved, which would make those numbers no longer",
        "  out-of-sample.",
        "",
        "READING THE TABLE",
        "",
        "  'score' is a rank WITHIN this run (lower better) and does not transfer;",
        "  the absolute metric column beside it is the one that means anything",
        "  outside the run. A PASS always outranks MARGINAL and MARGINAL always",
        "  outranks FAIL, whatever the score: a better similarity number never buys",
        "  its way past a folded warp.",
        "",
        "  The '1-dice' / '1-xdice' columns are stored lower-is-better like every",
        "  other metric here. Dice itself is 1 minus what is printed.",
        "",
        "  Every functional in an intensity jury improves with overfit, and the gate",
        "  fails only on FOLDING -- det(J) > 0 is a topology check, not a smoothness",
        "  one. So the top row is the loosest field that is still legal.",
        "",
        "  Whether that is a PROBLEM is not something this tool knows. Folding is the",
        "  only thing established to be wrong; 'bend' has no bound at all and the",
        "  compression thresholds are conventions, not measurements. Warping",
        "  different brains onto a template is a large-deformation problem, so more",
        "  deformation may simply be the job being done. The FRONTIER maps the trade",
        "  so you can see what a setting costs and buys -- it does not say where to",
        "  stop. Calibrate that against tools whose output is already accepted:",
        "  ffs_tunewarp -diag_only NAME -cohort their_output/ -warp_suffix _WARP",
        "",
        "  PINNED means the warp came back resting on the solver's own anti-fold",
        "  floor, so the damping is what kept it legal rather than its",
        "  regularization. Those are demoted and kept off the frontier.",
        "",
        "  Mind the 'n' column. Screening deliberately spends few fits on most",
        "  candidates, so a row measured on one brain and one measured on four are",
        "  not comparable; -reproduce a row before believing it.",
        "",
        "  -effects is the safer read for a knob: it separates 'how good when it",
        "  works' from 'how often does it work'.",
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


def _find_warps(root: str, suffix: str, subjects) -> dict[str, str]:
    """Match each subject to its own displacement field by name.

    Matched on the subject stem rather than by position, because a method that
    failed on one brain simply has no warp for it -- and pairing by position
    would then silently attribute every subsequent field to the wrong subject.
    """
    found: dict[str, str] = {}
    for s in subjects:
        for cand in sorted(Path(root).glob(f"{s.name}{suffix}*")):
            if cand.is_file():
                found[s.name] = str(cand)
                break
    return found


def _slug(name: str) -> str:
    """A directory name from a method name, keeping it recognisable.

    'AFNI 3dQwarp' -> 'AFNI_3dQwarp'. The full name stays in every table's method
    column and in meta.json, so nothing depends on this being reversible.
    """
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in name.strip()]
    return "".join(keep).strip("_") or "method"


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


def _template_pairs(subjects, base: str, split: str, verb: int = 1) -> list[SubjectPair]:
    """One pair per subject, all against the same template."""
    side = [s for s in subjects if s.split == split]
    if verb >= 1:
        print(f"  {len(side)} {split} subject(s) -> {base}")
    return [SubjectPair(s.name, base, s.image, None, s.labels, s.split) for s in side]


def _build_cohort_pairs(
    args: argparse.Namespace, n_run: int = 0, verb: int = 1, group: bool = False
) -> tuple[list[SubjectPair], list[SubjectPair]]:
    """Discover a labelled cohort and split it into training and held-out pairs."""
    subjects = discover_cohort(args.cohort, label_suffix=args.label_suffix)
    subjects = split_subjects(subjects, args.holdout, seed=args.seed)
    if verb >= 1:
        print(describe_cohort(subjects, []))

    # A group recipe warps every subject to ONE base, so there are no pairs to
    # choose: the "panel" is the cohort itself and every member of it is fit for
    # every config.
    if group:
        if not args.base or len(args.base) != 1:
            raise SystemExit(
                "-type common_T1 warps a cohort into one common space, so it needs "
                "exactly one -base (the template), e.g. "
                "-base MNI152_2009_template_SSW.nii.gz'[0]'"
            )
        return (
            _template_pairs(subjects, args.base[0], TRAIN, verb),
            _template_pairs(subjects, args.base[0], TEST, verb=0),
        )

    n_train = sum(s.split == TRAIN for s in subjects)
    pool = n_train * (n_train - 1)
    if args.pairs is None:
        n_pairs = panel_size(pool, args.budget) if args.search == "adaptive" else min(pool, 4)
    elif str(args.pairs).lower() == "all":
        n_pairs = None
    else:
        n_pairs = int(args.pairs)

    # The window slides between runs so a resumed study stops re-measuring one
    # fixed panel. The HELD-OUT panel deliberately does not move: its whole value
    # is being the same yardstick every time the study is reopened.
    start = rotation_start(n_run, n_pairs) if n_pairs else 0
    train = pairwise(subjects, n_pairs, split=TRAIN, start=start)
    # The held-out panel is a fixed round-robin decided BEFORE any of it is fit, so
    # capping it costs nothing in honesty -- what would cost is choosing which
    # pairs to keep after seeing them. Two offsets means every held-out subject
    # appears twice as a base and twice as a source; the full ordered set of six
    # subjects is thirty pairs per config, which is half an hour a finalist for a
    # number that has already stopped moving.
    n_test = sum(s.split == TEST for s in subjects)
    test = pairwise(subjects, 2 * n_test, split=TEST) if n_test else []
    args._held_out = [s.name for s in subjects if s.split == TEST]
    if verb >= 1:
        if start:
            print(f"  panel window starts at pair {start} of {pool} (run {n_run + 1})")
        if test:
            print(f"  {len(test)} held-out pair(s), fit after the search")
        missing = [s.name for s in subjects if s.labels is None]
        if missing:
            print(f"  ! no segmentation for: {', '.join(missing)}")
    return train, test


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
            if any(t.split == TEST for t in store.trials):
                print("\n" + format_holdout(store))
        if args.plot is not None:
            _write_plot(store, args.plot or out / "frontier.png", args.recipe)
        return 0

    if args.collect:
        written = collect_diagnostics(out / "diag")
        if not written:
            raise SystemExit(f"no method tables under {out / 'diag'} (see -diagnostics/-diag_only)")
        print(f"Collected {len(written)} table(s) in {out / 'diag'}:")
        for w in written:
            print(f"  {w.name}")
        return 0

    if args.diag_only:
        if not args.cohort:
            raise SystemExit("-diag_only needs -cohort DIR: the labels to score")
        subjects = discover_cohort(
            args.cohort,
            label_suffix=args.label_suffix,
            labels_only=True,
            ignore_suffixes=(args.warp_suffix,) if args.warp_suffix else (),
        )
        target = out / "diag" / _slug(args.diag_only)
        print(f"\nScoring {len(subjects)} pre-warped segmentation(s) as {args.diag_only!r}")
        contrast = RECIPES[args.recipe].contrast if args.recipe else "same"
        warps = _find_warps(args.cohort, args.warp_suffix, subjects) if args.warp_suffix else None
        if args.warp_suffix and not warps:
            raise SystemExit(f"no warp files matching *{args.warp_suffix}* in {args.cohort}")
        written = diagnose_warped(
            subjects,
            target,
            args.method or args.diag_only,
            base=args.base[0] if args.base else None,
            metrics=args.metrics,
            contrast=contrast,
            warps=warps,
            warp_units=args.warp_units,
            device=setup_device(args.device, tf32=REGISTRATION_TF32),
            save_subject_labels=args.save_subject_labels,
            verb=args.verb,
        )
        print(f"\nWrote {len(written)} file(s) to {target}")
        print(open(target / "summary.tsv").read())
        return 0

    if args.diagnostics is not None:
        if not args.recipe:
            raise SystemExit("-diagnostics needs -type, since it re-runs a fit")
        recipe = with_overrides(RECIPES[args.recipe], parse_fix(args.fix or []), args.tune)
        if not recipe.group:
            raise SystemExit(
                f"-diagnostics is for common-space recipes; -type {recipe.name} scores "
                "one pair at a time, so use -reproduce to keep its outputs instead."
            )
        row = next((r for r in store.results(split=None) if r.config_id == args.diagnostics), None)
        if row is None:
            raise SystemExit(f"no config {args.diagnostics} in {store.path}")
        pairs, held_out = _build_cohort_pairs(args, len(store.runs), args.verb, group=True)
        device = setup_device(args.device, tf32=REGISTRATION_TF32)
        if args.allineate:
            pairs = affine_align(pairs, recipe, out, device=device, verb=args.verb)
            if held_out:
                held_out = affine_align(held_out, recipe, out, device=device, verb=args.verb)
        target = out / "diag" / f"config{args.diagnostics:04d}"
        # Every backend's winner must carry a distinct name or they collide in the
        # method column and the head-to-head pivot silently averages them.
        method = args.method or f"ffs {row.backend} c{row.config_id}"
        print(f"\nConfig {args.diagnostics} as {method!r}: {row.backend} {row.label()}")
        written = group_diagnostics(
            pairs + held_out,
            recipe,
            row.config,
            row.backend,
            target,
            device=device,
            save_subject_labels=args.save_subject_labels,
            metrics=args.metrics,
            method=method,
            verb=args.verb,
        )
        print(f"\nWrote {len(written)} file(s) to {target}")
        for w in written:
            print(f"  {w.name}")
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
    held_out: list[SubjectPair] = []
    if args.cohort:
        # len(store.runs) is the number of runs BEFORE this one, so a fresh study
        # is run 0 and each resume slides the panel window along.
        pairs, held_out = _build_cohort_pairs(args, len(store.runs), args.verb, group=recipe.group)
    else:
        pairs = _build_pairs(args, recipe.pairing)
    if recipe.labels:
        # A common-space run compares the cohort's transported tracings against
        # each other, so the target needs none of its own -- MNI has none.
        traced = (p.has_source_labels if recipe.group else p.has_labels for p in pairs)
        if not all(traced):
            raise SystemExit(
                f"-type {recipe.name} is judged on segmentations, but some subjects "
                "have none. Check -label_suffix, or pick a recipe that judges on "
                "intensities."
            )
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
        held_out=list(getattr(args, "_held_out", [])),
    )

    # After begin_run, so `warnings()` can compare the earlier runs against the one
    # about to happen -- and before any fitting, because "these trials were scored by
    # a different build" is worth knowing while it can still change your mind.
    if args.verb >= 1 and (resume := format_resume(store, [p.name for p in pairs])):
        print("\n" + resume)
        for w in store.warnings():
            print(f"  ! {w}")

    if args.allineate:
        # Held-out subjects need it too. They are fit after the search, against
        # the same base, so a native-grid source there is a shape mismatch that
        # only shows up an hour in.
        n_all = len(pairs) + len(held_out)
        if args.verb >= 1:
            print(f"\nStep 0: affine ({recipe.optimize}) for {n_all} subject(s)")
        pairs = affine_align(pairs, recipe, out, device=device, verb=args.verb)
        if held_out:
            held_out = affine_align(held_out, recipe, out, device=device, verb=args.verb)

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

    if held_out:
        evaluate_holdout(
            held_out,
            recipe,
            store,
            n_configs=args.holdout_configs,
            device=device,
            verb=args.verb,
        )

    if (warn := store.warnings()) and args.verb < 1:
        # Already said before the fits when verbose; repeated here only for a quiet
        # run, where this is the first and last chance to say it.
        print("\nThis directory holds earlier runs that may not be comparable:")
        for w in warn:
            print(f"  - {w}")
    print("\n" + format_results_table(store.results(), limit=args.top))
    if held_out:
        print("\n" + format_holdout(store))
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
