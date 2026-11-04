"""CLI: the laminar model over overlapping cortical columns, painted back to cortex.

Command: ffs_laminascope

Usage:
    # one step: the cells are fitted from the timeseries
    ffs_laminascope -input run*.nii.gz -events *_events.tsv -window 0 14 \
        -spec comparisons.toml -fs_subj $SUBJ -responsive faces_t.nii.gz \
        -responsive_thresh 4 -depths 7 9 11 -equivolume -device cuda -prefix scope/sub

    # or two steps, from voxel curves fitted once by a LINEAR estimator
    ffs_deconvolve -input run*.nii.gz -events ... -event_prefix task -model TENT \
        -window 0 14 -tent-smooth global -mask ribbon.nii.gz -prefix curves/sub
    ffs_laminascope -iresp_prefix curves/sub -window 0 14 -duration 1 \
        -spec comparisons.toml -fs_subj $SUBJ -responsive faces_t.nii.gz \
        -responsive_thresh 4 -depths 7 9 11 -equivolume -device cuda -prefix scope/sub
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, add_device_arg
from fastfuncstuff.cli_utils import add_bids_events_args, add_ortvec_arguments

_DESCRIPTION = """\
Run the laminar BOLD model (Havlicek & Uludag; the analysis of Faes et al. 2026, as
ffs_laminar runs it per ROI) over cortical columns made on the fly, and paint what
each column found back onto the surface and the ribbon.

Columns are geodesic disks of -column_radius on the midthickness, centred so that
every responsive vertex (-responsive) is within -column_spacing of a centre: the
responsive area sets how many there are. With the default spacing (half the radius)
an interior vertex sits in several columns that differ only in where their edges
fall, and the maps report their consensus and how far they agree.

Each column's depth curves come from one of two routes. With -input, the voxel
timeseries is averaged into column x depth cells and each cell is fitted here: an
estimate of averaged data, so no linearity is assumed, the knots follow the events,
and every cell carries its own residual variance. With -iresp_prefix the curves are
read from the {prefix}_iresp_<label> files of ffs_deconvolve (percent signal change)
and averaged; that is exact ONLY for a linear estimator -- -tent-smooth global (one
lambda) or a fixed lambda -- because only then does a column's curve equal the fit of
the column's mean. Per-voxel lambda (-tent-smooth reml) is not linear: it shrinks
noisy voxels harder, and since the noise level is a depth gradient that lands in the
amplitudes as a fake depth profile.

Each -spec comparison is one model space at every column: baseline and test
conditions concatenated with masked padding, every layer hypothesis inverted at
every -depths K, averaged across K, compared by free energy. Columns run batched
(-device cuda pays from ~100 columns; it saturates near 1,000 per batch).

Writes, per comparison NAME:
  {prefix}_NAME_columns.tsv   one line per column: centre, best model, P(model),
                              P(depth modulated), B per depth (sd, P != 0), R2, flags
  {prefix}_NAME.<hemi>.func.gii  per vertex: P(depth modulated) per depth, B per
                              depth, consensus winner and its agreement, columns covering
  {prefix}_NAME_ribbon.nii.gz the same maps on the -iresp grid, through each ribbon
                              voxel's owning vertex
and {prefix}_columns.npz (columns, their curves and PSFs, every fit), {prefix}.json.
"""


_EPILOG = """\
START HERE: FOUR BRICKS, ONE MASK

Mask every map with the 'gate' brick and read two others.  Nothing below this
paragraph is needed to look at a result:

    R=myprefix_gender_vs_fixate_ribbon.nii.gz
    3dcalc -a "$R[gate]" -b "$R[p_mod_superficial]" -expr 'a*b' -prefix p_sup.nii.gz
    3dcalc -a "$R[gate]" -b "$R[B_superficial]" -expr 'a*b' -prefix B_sup.nii.gz

p_mod_<depth> is P(this depth is modulated), 0 to 1, marginal over the whole model
space -- the map to threshold, at 0.75 or 0.95 as you would any posterior
probability.  B_<depth> is how much, and its sign says which way (positive B
LOWERS self-inhibition).  gate is 1 where the column responded at all, was read
whole, and its fit was sound; it is 0 or NaN everywhere else, and it is the
tool's own verdict, printed as a count when the maps are written.

THE SENTENCE THE MAPS SUPPORT

For one gated location, read five numbers and say this:

  "In gender vs fixate, this column responded at 1.8% peak; the conditions
   differed with 1,000:1 evidence over no difference (logBF 7); the modulation
   was at the SUPERFICIAL depth (p 1.00, and 0.39 middle, 0.39 deep); it was
   POSITIVE, B = +0.92 Hz against a self-inhibition of sigma Hz, i.e. gender
   lowered that depth's self-inhibition -- more recurrent excitation, a larger
   neuronal response -- by 0.92 of the 1/s units sigma is in; the best single
   model was 'superficial', which won only p_best 0.38 of the posterior, so the
   DEPTH statement is the marginal, not the winner."

That is the whole content: WHERE (which column), WHETHER (logBF over null),
WHICH DEPTH (p_mod_<depth>), WHICH WAY AND HOW MUCH (B in Hz, self-inhibition),
and HOW SURE the single best story is (p_best).  There is no "what the effect
was" beyond that: the model has one modulatory parameter per depth and it enters
as the depth's self-inhibition, so "gender differed from fixate here" can only
come out as "the neuronal gain at this depth changed by B".  Attention,
adaptation and sharper tuning are indistinguishable in it.

The winner brick is a DEPTH BITMASK, not a model number: bit 1 superficial,
2 middle, 4 deep, so 5 = superficial+deep, 7 = all three, 0 = null.  The legend
is printed when the maps are written and sits in the .gii metadata
(winner_bits).  The per-column TSV spells the name out instead.

Two bricks are NOT thresholds, however statistical they look.  logBF_vs_null is
high nearly everywhere there is a response -- median 24.5 nats over 903 real
columns, where 3 nats is the conventional floor -- because "these conditions
differ at SOME depth" is almost always true once a column responds at all.  It
rules things out, it does not select.  And p_best is how narrowly ONE of 2^N
hypotheses won: 0.3 is a near-tie, and its median on those same columns was 0.5.
Neither is a p-value, and no voxel-wise correction is valid here (the columns
overlap by construction -- see WHAT NOT TO BELIEVE).

WHERE THE MAPS EXIST, AND WHERE THEY DO NOT

The painted extent is NOT the -responsive mask.  Centres are placed so every
responsive vertex is within -column_spacing of one, but each column is a disk of
-column_radius, so the maps cover the responsive set DILATED by the radius --
measured on a real run, 15,924 responsive vertices became 66,062 covered ones
(4.1x) at -column_radius 6, i.e. 35% of the ribbon.  That is the design (a
column is a neighbourhood, not a vertex), and it means the edge of the coloured
blob carries no information about where the response was.  The startup line
reports the factor; threshold the maps, never read their boundary.

If your localiser already exists on the surface, give the .gii instead of the
volume -- one per hemisphere, matched by metadata, then filename, then vertex
count.  Nothing is sampled, so the "any of 6 depth reads" rule below does not
apply and the map you looked at is the map that places the columns.

A cluster filter is the cheap fix, and it is worth using.  From a VOLUME, ONE
suprathreshold voxel touching the ribbon makes a vertex responsive (the vertex
passes if ANY of 6 nearest-voxel reads at fracs 0.1-0.85 does), and one
responsive vertex is enough to place a column.  On the run above, |z| >= 3 gave 5,038 clusters of
which only 191 held 5 or more voxels -- and those specks accounted for 14% of the
responsive vertices, 58% of the painted area and 37% of the columns fitted:

    filter        resp vtx   columns   covered vtx
    none            15,924     1,576        66,062  (4.1x)
    NN1 >= 2        14,543     1,149        36,834  (2.5x)
    NN1 >= 5        13,720       990        27,911  (2.0x)
    NN1 >= 10       13,373       916        25,440  (1.9x)

-responsive_cluster_size 5 -responsive_cluster_nn 1 is the knee.  It is off by
default so a run stays comparable with earlier ones, and the census is printed
either way so the cost is visible without running the job twice.  On a surface
-responsive the size counts VERTICES over mesh edges (NN does not apply) and the
note reports the kept area in mm2, which is the number worth calibrating on.

Also check which sub-brick -responsive actually read.  Brick 0 of an AFNI stats
bucket is Full_Fstat and brick 1 is Full_Zstat -- the OMNIBUS, every condition
against baseline, which is "responded to the experiment" and not the contrast
you meant to localise.  Both are non-negative, so -responsive_positive silently
does nothing on them; the tool now warns when the brick it read has no negative
values.  A signed contrast brick is usually several further along
(3dinfo -label).

Outside the columns there is no estimate, and the maps say so with NaN rather
than 0 -- B, p_mod_<depth> and logBF_vs_null are all legitimately near 0 INSIDE a
column, so a zero fill would make "no column here" unreadable against "no effect
here".  n_columns is the exception and keeps its 0: that is a real count, and
n_columns > 0 is the coverage mask if your viewer wants one.

READING THE MAPS: WHAT THE GATE IS MADE OF

The 'gate' brick is steps 1-3 below, already applied; this is what it asserts and
how to tighten it.  Each step decides whether the next one is about physiology or
about noise.

1. IS THERE A RESPONSE?  baseline_peak (percent, the peak of the depth-averaged
   baseline curve).  A column with no response gives the model nothing to
   explain and its B is noise however confidently flagged -- measured: an atlas
   region with no event response still produced flagged depths.  -gate_baseline
   sets it (default 0.5%) and it is the one worth setting from your own
   distribution, since it is in percent and therefore interpretable; on 903 real
   columns the percentiles were 0.10 / 0.37 / 0.69 / 1.00 / 1.40.  Judging a
   laminar effect in tissue that did not respond is the commonest way to read
   this tool wrong.

2. WAS THE COLUMN READ?  in_fov, the share of its ribbon inside the mask/FOV.
   A slab cuts V1 and V2 in half; a cut depth profile is not a shallow profile,
   it is a different measurement.  -min_coverage already drops the worst (the
   "cut by the FOV/mask" count on startup); in_fov grades what survived.

3. IS THE FIT SOUND?  Three flags, any of which makes a ranking a report about
   the optimiser:
     nesting_nats   a model scoring below one it CONTAINS.  Impossible in
                    principle (the smaller model is the larger with B pinned at
                    0), so it is a local optimum.  >10 nats is counted and
                    warned about.  -model_space reduce (the default) took this
                    from 23% of real columns to 0%; brute-force -model_space
                    invert is where the 23% came from.
     stalled        ended on rejected steps with the predicted gain collapsed.
                    -rescue cut these from 47% to 9% on real columns; it is
                    worth having on.
     failed / r2    no usable inversion at all / how much of the fitted rows the
                    winner explains.

4. IS THERE ANY MODULATION AT ALL?  logBF_vs_null, the best model's evidence over
   the null.  The conventional reading applies -- ~3 nats is about 20:1 -- and
   below it the honest statement is "no evidence these conditions differ at any
   depth".  But it is a floor, not a selector: on 903 real columns its median was
   24.5 nats and it removed only 52 of the 547 that passed steps 1-3, so a map of
   it looks significant everywhere and says almost nothing.  It is left out of
   'gate' for that reason; add it (ispositive(logBF_vs_null-3)) if you want the
   published convention stated explicitly.

5. ONLY THEN, WHERE.  Prefer p_mod_<depth> -- P(depth modulated), marginalised over
   the whole model space -- to the winner map.  The winner is one of 2^N
   hypotheses and p_best says how narrowly it won; a p_best of 0.3 is a near-tie
   dressed as an answer.

   Mind the two families of probability in the per-column TSV.  p_mod_<depth> is
   the MARGINAL, P(this depth is modulated) over the whole space, and is what the
   maps carry.  p_<MODEL> -- p_null, p_superficial+deep, ... -- is one model's
   posterior, and a model can be near 0 while its depth is near 1 (measured on a
   real column: p_superficial 6e-20, p_mod_superficial 0.84, because the depth's
   support was spread over the five other models containing it).  Maps written
   before 2026-11-04 named the marginal bricks p_<depth>, which collides with the
   TSV's model column of the same name; they hold the marginal.  B_<depth> is the WINNER's modulation; the long table's
   B_bma is model-averaged, which is what a group test should carry, because a
   depth that some models pin at 0 contributes 0 rather than dropping out.

   B modulates self-inhibition (A_kk = -sigma + u*B_k), so positive B LOWERS
   inhibition; sigma - B_bma below 0 is net self-excitation while the modulation
   is on.  P(depth modulated) ignores the sign, so a comparison is
   non-directional however its sides are ordered.

WHAT NOT TO BELIEVE

  The columns are not independent, by construction.  At the default spacing
  (half the radius) an interior vertex sits in several columns differing only in
  where their edges fall.  That redundancy is the point -- agreement and
  n_columns report the consensus -- but it means no vertex-wise or voxel-wise
  multiple-comparison correction is valid on these maps.  They are a smoothed,
  deliberately overlapping summary, not a test image.

  The middle depth is the least identified, everywhere.  Measured across atlas
  regions and contrasts: P(modulated) near 0.5 with a large posterior sd.  A
  middle-depth claim needs more than this tool.

  Low s_d is the dangerous direction.  Underestimating the ascending vein's
  baseline-CBV slope gives qualitatively WRONG laminar profiles; overestimating
  is safe (Uludag & Havlicek 2021).  Real columns have come out with a 5th
  percentile near -2.8, i.e. the bad direction.  Look at the s_d map before
  trusting a depth assignment, and treat 3 mm columns as unlikely to identify it
  at all -- pooling it was tried (-vascular_prior) and the evidence declined.

  sigma is not comparable across designs and can be bimodal WITHIN a region
  (~0.1 vs ~3.5 Hz observed in one region).  Check its distribution before any
  group test on sigma, and never compare a block design's sigma with an event
  design's.

  Absolute F moves with the padding draw (~1 nat; the model comparison ~0.2)
  under -hyperprior reference, the parity default.  Rankings within a run are
  safe; F across runs with different -pad_seed is not.

  -free_energy reference (the default) ranks on the FIRST K only, as
  spm_dcm_bpa leaves it, while the parameters are averaged over all K.  The
  per-K evidence is in {prefix}_columns.npz (per_k_F), so this is checkable
  without refitting.

  Identifiability at this scale is unproven.  The model's own authors declined
  to explore dynamic invertibility and warned it depends on SNR, ROI size and
  within-ROI variability; a 6 mm column has far less than a whole ROI.

WHAT REPLICATED, MEASURED

Split-half on one subject at 6 mm columns: a stimulus-contrast comparison
replicates in the data (profile r 0.63, 0.74 in face patches) and in the model,
attenuated (P and B r 0.2-0.26 against a null of 0.07).  TASK contrasts did not
replicate at 6 mm in one subject.  The model tracking the data's own reliability
is the system working; a single subject's single-run map is not the unit of
inference.  Inference is ACROSS SUBJECTS, as in the published analysis: use
{prefix}_NAME_regions_long.tsv (one line per region x depth, with in_fov and
baseline_peak to gate on) and test B_bma, p_modulated or sigma_test_hz across a
cohort.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from fastfuncstuff.surface.ribbon_args import add_depth_args, add_ribbon_args

    p = FfsArgumentParser(prog="ffs_laminascope", description=_DESCRIPTION, epilog=_EPILOG)
    g = p.add_argument_group("Curves and comparisons")
    g.add_argument("-iresp_prefix", default=None, metavar="PREFIX",
                   help="PREFIX of voxel curve files PREFIX_iresp_<label>.nii[.gz|.zst], from "
                   "ffs_deconvolve with a LINEAR estimator (-tent-smooth global or a fixed "
                   "lambda). Mutually exclusive with -input, which fits the curves here.")  # fmt: skip
    g.add_argument("-window", nargs="+", default=None, metavar="WINDOW",
                   help="Seconds after onset of the first and last curve sample, as "
                   "ffs_deconvolve's -window: '0 14' or '0,14' (every condition), or "
                   "'0,14 0,24' (one per condition, in -labels / trial_type order). A 3 s and "
                   "a 10 s stimulus need different lengths of response, and the two blocks are "
                   "then fitted at their own lengths. BOT must be 0: the model puts the event "
                   "on the first sample. With -input and no -window, each condition's window is "
                   "estimated from its own event duration.")  # fmt: skip
    g.add_argument("-duration", type=float, default=None, nargs="+", metavar="SEC",
                   help="Duration of the events the curves respond to (s): one value, or two "
                   "(BASELINE TEST) when the compared conditions' events differ in length. "
                   "With -input it defaults to the events' own durations. With -modulation "
                   "event the modulatory input follows the TEST event's length.")  # fmt: skip
    g.add_argument("-spec", default=None, metavar="TOML",
                   help="Comparisons to fit: [[comparison]] tables with name, baseline, test "
                   "(a label, a glob or a list, averaged) and optional models.")  # fmt: skip
    g.add_argument("-compare", nargs=3, action="append", default=None,
                   metavar=("NAME", "BASELINE", "TEST"),
                   help="One comparison on the command line (globs allowed); repeatable.")  # fmt: skip
    g.add_argument("-mask", default=None, metavar="FILE",
                   help="Voxels the reads may use (the mask the curves were fitted in).")  # fmt: skip

    ts = p.add_argument_group(
        "Timeseries input (fit the curves here, instead of -iresp_prefix)",
        "Mixes the voxel timeseries into column x depth cells and fits each cell once, "
        "rather than fitting voxels and averaging their fits. Averaging the data and "
        "estimating once needs no linearity argument, so the fit is not restricted to a "
        "linear estimator, and every cell gets its own residual variance and effective dof.",
    )
    ts.add_argument("-input", nargs="+", default=None, metavar="FILE",
                    help="Input runs (wildcards fine). Mutually exclusive with -iresp_prefix.")  # fmt: skip
    ts.add_argument("-onsets", nargs="+", default=None, metavar="FILE",
                    help="AFNI timing files, one per condition, one row per run.")  # fmt: skip
    add_bids_events_args(ts, events_help="BIDS events TSV files, one per run. Conditions "
                         "come from trial_type; the comparisons in -spec name them.")  # fmt: skip
    ts.add_argument("-labels", nargs="+", default=None, metavar="LABEL",
                    help="Condition labels, overriding trial_type / timing-file names.")  # fmt: skip
    ts.add_argument("-tr", type=float, default=None, metavar="SEC", help="Override the TR.")
    ts.add_argument("-polort", default="A", metavar="N",
                    help="Per-run polynomial drift degree, block-diagonal: 'A' (default) is "
                    "AFNI's 1 + floor(duration/150) per run, an integer is fixed, -1 none.")  # fmt: skip
    ts.add_argument("-do_scale", dest="do_scale", action="store_true", default=True,
                    help="Scale each VOXEL to percent signal change before mixing (default): "
                    "the laminar model wants percent, and scaling per voxel rather than per "
                    "cell keeps a column's voxels equally weighted whatever their intensity.")  # fmt: skip
    ts.add_argument("-dont_scale", dest="do_scale", action="store_false",
                    help="Leave the data in its own units (the model will refuse them).")  # fmt: skip
    ts.add_argument("-do_blur", type=float, default=None, metavar="FWHM",
                    help="Spatial blur (mm FWHM) per run before masking. Rarely wanted here: "
                    "it mixes depths, which is the thing being measured.")  # fmt: skip
    ts.add_argument("-drop_first", type=int, default=0, metavar="N",
                    help="Drop N volumes from the start of every run (shifts onsets).")  # fmt: skip
    ts.add_argument("-drop_last", type=int, default=0, metavar="N")
    ts.add_argument("-load_threads", type=int, default=None, metavar="N")
    ts.add_argument("-microtime_offset", type=float, default=None, metavar="SEC",
                    help="Within-TR time each volume was sampled at (default: the header's).")  # fmt: skip
    add_ortvec_arguments(ts)
    ts.add_argument("-tent_lambda", type=float, default=None, metavar="LAM",
                    help="Fixed relative smoothing strength for the cell fits. Default: one "
                    "lambda chosen for the whole dataset by leave-one-run-out. Give a value "
                    "when a condition's events all sit in one run, where LORO cannot choose.")  # fmt: skip
    ts.add_argument("-cell_lambda", choices=("global",), default="global",
                    help="Scope of the smoothing strength. Only 'global' (one lambda for every "
                    "cell) is implemented: the noise level is a depth gradient, so a per-cell "
                    "lambda writes that gradient into the amplitudes, and a per-column one "
                    "biases amplitudes between the columns the maps compare.")  # fmt: skip
    ts.add_argument("-smooth_penalty", default="diff2", metavar="SPEC",
                    help="Penalty shape for the cell fits: diffN or gp:SECONDS.")  # fmt: skip
    ts.add_argument("-smooth_signal_r2", type=float, default=0.05, metavar="R2",
                    help="Cells that count as signal when choosing the global lambda.")  # fmt: skip

    c = p.add_argument_group("Columns")
    add_ribbon_args(c)
    c.add_argument("-responsive", nargs="+", default=None, metavar="FILE",
                   help="Map marking responsive cortex: a mask, or a statistic with "
                   "-responsive_thresh. Either a VOLUME (file[N] picks a brick) or one .gii "
                   "per hemisphere, which needs no sampling and is the better input when you "
                   "have it. A column centre goes wherever it marks the ribbon at "
                   "any depth. Select on something independent of the comparisons (an omnibus "
                   "F, a localiser) -- never on the difference being tested.")  # fmt: skip
    c.add_argument("-responsive_thresh", type=float, default=None, metavar="T",
                   help="Threshold for a statistic -responsive (|value| >= T).")  # fmt: skip
    c.add_argument("-responsive_positive", action="store_true",
                   help="Threshold signed (value >= T) instead of |value|.")  # fmt: skip
    c.add_argument("-responsive_cluster_size", type=int, default=None, metavar="N",
                   help="Drop suprathreshold clusters smaller than N voxels before any vertex "
                   "is read. A lone voxel at |z| >= 3 is mostly a false positive, and ONE "
                   "responsive vertex is enough to place a column -- which then claims every "
                   "vertex within -column_radius and costs a model inversion. Measured on a "
                   "real localiser at |z| >= 3: 5,038 clusters, of which only 191 held 5 or "
                   "more voxels -- and the specks were 14%% of the responsive vertices but "
                   "58%% of the painted area and 37%% of the columns fitted. 5 at NN1 is the knee "
                   "(10 and 20 barely move it). Default: no filter, so a run stays comparable "
                   "with earlier ones, and the speck census is printed either way.")  # fmt: skip
    c.add_argument("-responsive_cluster_nn", type=int, default=1, choices=(1, 2, 3),
                   help="Connectivity for -responsive_cluster_size on a VOLUME -responsive, "
                   "as AFNI's NN: 1 faces, 2 faces + edges, 3 faces + edges + corners. A "
                   "surface -responsive ignores it -- mesh adjacency is the only connectivity "
                   "there, and the size is then counted in vertices (the area kept is "
                   "reported in mm2).")  # fmt: skip
    c.add_argument("-within", default=None, metavar="FILE",
                   help="Only centre columns inside this region (label, annot or volume ROI): "
                   "for trying a few places before the whole cortex.")  # fmt: skip
    c.add_argument("-regions_from", default=None, metavar="MAP",
                   help="Instead of disks: one region per connected surface cluster of this "
                   "statistic above -regions_thresh (e.g. faces > objects from a localiser), "
                   "closed along the mesh by 1 mm, at least -regions_min_mm2. The data's own "
                   "patches, often 8-10 mm or more across. Select on independent runs.")  # fmt: skip
    c.add_argument("-regions_atlas", nargs="+", default=None, metavar="FILE",
                   help="Instead of disks: one region per atlas label (.annot, .label, GIfTI "
                   "label map, or a volume atlas claimed along the column), named. The way "
                   "to have the same regions in every subject for group tests.")  # fmt: skip
    c.add_argument("-regions_thresh", type=float, default=3.1, metavar="T",
                   help="Threshold for -regions_from (value >= T; |value| with -regions_abs).")  # fmt: skip
    c.add_argument("-regions_abs", action="store_true",
                   help="Cluster |value| >= T, so negative patches (objects > faces) count too.")  # fmt: skip
    c.add_argument("-regions_min_mm2", type=float, default=20.0, metavar="MM2",
                   help="Smallest cluster kept as a region (midthickness area).")  # fmt: skip
    c.add_argument("-column_radius", type=float, default=3.0, metavar="MM",
                   help="Geodesic radius of a column on the midthickness.")  # fmt: skip
    c.add_argument("-column_spacing", type=float, default=None, metavar="MM",
                   help="Every responsive vertex is within this of a centre (default: half the "
                   "radius, so interior vertices sit in several columns).")  # fmt: skip
    c.add_argument("-min_coverage", type=float, default=0.99, metavar="FRAC",
                   help="Skip columns whose depth bins were read less than this inside the "
                   "data (FOV edge, mask): the model cannot fit a cut profile.")  # fmt: skip
    c.add_argument("-gate_baseline", type=float, default=0.5, metavar="PCT",
                   help="Peak baseline response, in percent signal change, below which a column "
                   "fails the 'gate' brick of the maps.  Nothing is skipped and no other output "
                   "changes: the fit still runs and is still written, because the cost of a "
                   "column is paid before this is known.  0 passes every fitted column.  A "
                   "column with no response gives the model nothing to explain, so its depth "
                   "probabilities are a decomposition of noise -- set this from your own "
                   "baseline_peak distribution (0.5 kept 587 of 903 real columns).")  # fmt: skip
    c.add_argument("-max_columns", type=int, default=None, metavar="N",
                   help="Fit a random N of the columns (a quick look).")  # fmt: skip
    c.add_argument("-seed", type=int, default=0, help="Seed of the cover order and -max_columns.")
    d = p.add_argument_group("Depth sampling")
    add_depth_args(d, ribbon=False)

    m = p.add_argument_group("Model (as ffs_laminar)")
    m.add_argument("-models", nargs="+", default=None, metavar="NAME",
                   help="Subset of the model space for every comparison (a comparison's own "
                   "'models' wins).")  # fmt: skip
    m.add_argument("-N", type=int, default=3, help="Neuronal depths.")
    m.add_argument("-modulation", nargs="+", default=["event"], metavar="WHEN",
                   help="When the test condition's modulation is on: 'event' (default, the whole "
                   "event), 'reference' (Faes et al.'s 0.1 s violation pulse, parity only) or "
                   "START DUR seconds after onset.")  # fmt: skip
    m.add_argument("-psf", choices=("measured", "none"), default="measured",
                   help="Depth PSF: each column's exact sampling matrix, or none.")  # fmt: skip
    m.add_argument("-free_energy", choices=("reference", "sum", "mean"), default="reference")
    m.add_argument("-TE", type=float, default=0.028, help="Echo time (s).")
    m.add_argument("-B0", type=float, default=7.0, help="Field strength (T).")
    m.add_argument("-max_iter", type=int, default=128, help="Variational Laplace iterations.")
    m.add_argument("-model_space", choices=("reduce", "invert"), default="reduce",
                   help="'reduce' (default): invert only the full model (every depth "
                   "modulated) and derive the other layer hypotheses by Bayesian model "
                   "reduction (Friston & Penny 2011) -- 2^N-fold fewer inversions, and nested "
                   "evidences consistent by construction (brute force left ~23%% of real "
                   "columns with a model far below one it contains). On the published data it "
                   "ranks all models as brute force does, the competitive ones within 1.3-1.6 "
                   "nats. 'invert': every model separately, as Faes et al.")  # fmt: skip
    m.add_argument("-vascular_prior", choices=("none", "local", "global"), default="none",
                   help="Empirical Bayes on the vascular parameters (s_d, al_d), with -model_space "
                   "reduce: each column's prior is centred on its neighbours' posteriors "
                   "(columns within -vascular_radius; 'global': the whole hemisphere) with the "
                   "spread that maximises the summed evidence (Friston et al. 2016). The "
                   "vasculature belongs to the tissue, not to one column, and a column's own "
                   "s_d otherwise drifts low -- the direction that gives wrong profiles.")  # fmt: skip
    m.add_argument("-vascular_from", nargs=2, default=None, metavar=("NPZ", "COMPARISON"),
                   help="Each column's prior on the vascular parameters (s_d, al_d) from another "
                   "ffs_laminascope run's posterior: its PREFIX_columns.npz and comparison name. "
                   "Columns are matched by centre vertex, so build both runs with the same "
                   "-responsive, -column_radius/-column_spacing and -seed. Use a high-SNR "
                   "localiser on separate runs: the vasculature belongs to the tissue, not "
                   "the task. Unmatched columns keep the original prior. Needs -model_space "
                   "reduce; replaces -vascular_prior.")  # fmt: skip
    m.add_argument("-vascular_radius", type=float, default=10.0, metavar="MM",
                   help="Neighbourhood of -vascular_prior local (column centres, mm).")  # fmt: skip
    m.add_argument("-rescue", type=int, default=2, metavar="N",
                   help="Restart fits that stalled (ended on rejected steps) up to N times from "
                   "their best point. On real columns half stalled somewhere without it, and a "
                   "stalled model can score hundreds of nats below a model it contains. 0 is "
                   "the published scheme (ffs_laminar's default).")  # fmt: skip
    m.add_argument("-pad_rows", type=int, default=20, metavar="N")
    m.add_argument("-pad_sd", type=float, default=0.1, metavar="PCT")
    m.add_argument("-pad_seed", type=int, default=20260918, metavar="N")
    m.add_argument("-hyperprior", choices=("reference", "rows"), default="reference")

    o = p.add_argument_group("Compute and output")
    add_device_arg(o)
    o.add_argument("-chunk", type=int, default=None, metavar="N",
                   help="Columns per batched inversion (default 1024 on a GPU, 64 on CPU).")  # fmt: skip
    o.add_argument("-prefix", required=True, help="Output prefix.")
    o.add_argument("-subject", default=None, metavar="ID",
                   help="Subject label written into the long tables (default: the prefix's "
                   "file name), so subjects' tables concatenate for group tests.")  # fmt: skip
    o.add_argument("-verb", type=int, default=1, help="0 quiet, 1 progress.")
    return p.parse_args(argv)


def _iresp_labels(prefix: str) -> dict[str, str]:
    import glob
    import re

    out = {}
    for path in sorted(glob.glob(f"{glob.escape(prefix)}_iresp_*")):
        m = re.match(re.escape(prefix) + r"_iresp_(.+?)\.nii(\.gz|\.zst)?$", path)
        if m and not m.group(1).endswith("_unconstrained"):
            out[m.group(1)] = path
    if not out:
        raise ValueError(f"no {prefix}_iresp_<label>.nii* files")
    return out


def _header_tr(img) -> float:
    """TR from a NIfTI header, for sizing the knot grid."""
    tr = float(getattr(img.header, "get_zooms", lambda: (0, 0, 0, 0))()[3]) if img.ndim > 3 else 0.0
    if tr <= 0:
        raise ValueError("the input has no TR in its header; give -tr")
    return tr


def _resolve_timing(args, say):
    """Event timing through the one shared parser, so -events / -onsets behave here
    exactly as in every other GLM tool."""
    from fastfuncstuff.cli_utils import (
        bids_event_kwargs,
        check_bids_event_args,
        parse_input_files,
        parse_timing_spec,
    )

    input_files = parse_input_files(args.input)
    if bool(args.onsets) == bool(args.events):
        raise ValueError("give -onsets or -events with -input")
    check_bids_event_args(args)
    timing = parse_timing_spec(
        events=args.events, onsets=args.onsets, durations_arg=None,
        n_runs=len(input_files), **bids_event_kwargs(args), input_files=input_files,
        verbose=args.verb >= 1, allow_missing_durations=True,
    )  # fmt: skip
    if args.labels:
        if len(args.labels) != timing.n_conditions:
            raise ValueError(
                f"-labels count ({len(args.labels)}) does not match the "
                f"{timing.n_conditions} condition(s) {timing.condition_labels}"
            )
        timing.condition_labels = list(args.labels)
    say(f"Conditions: {', '.join(timing.condition_labels)} over {len(input_files)} run(s)")
    return timing, input_files


def _resolve_windows(args, labels: list[str], tr: float | None, per_label, n_file, say):
    """Each condition's knot grid, seconds after its own onset.

    Per-condition windows are the point: a 3 s and a 10 s stimulus need different
    lengths of response. BOT must be 0 for every condition -- the model places the
    event on the response's first sample, so a later start would be modelled as if
    it began at the onset (see laminar/io.py:_scan_interval).

    ``tr`` is the knot spacing when we are choosing the knots (``-input``). With
    ``-iresp_prefix`` it is ``None`` and ``n_file`` maps each label to its file's
    sample count; the spacing then comes from the longest window over its own file's
    samples. Reading it from the header would be wrong as well as fragile -- the
    header carries the data's TR, which is the knot spacing only when deconvolve put
    one knot per TR, and ffs_deconvolve does not always write one at all.
    """
    from fastfuncstuff.design.hrf import compute_windows_from_durations, parse_tent_windows

    if args.window is not None:
        windows = parse_tent_windows(args.window, len(labels))
        if tr is None:
            # The condition with the longest window: its file is never shorter than
            # its own window, so it fixes the spacing for all of them.
            i_long = max(range(len(labels)), key=lambda i: windows[i][1])
            n_long = n_file[labels[i_long]]
            if n_long < 3:
                raise ValueError(f"{labels[i_long]}: {n_long} curve sample(s); that is not a curve")
            tr = windows[i_long][1] / (n_long - 1)
        # Positional against the condition order, as ffs_deconvolve's -window is, and
        # that order is trial_type's (alphabetical) unless -labels set it. Said out
        # loud every time: giving the long condition the short window truncates its
        # response, and nothing downstream can notice.
        if len({w for w in windows}) > 1:
            say("Windows as given, per condition: " + ", ".join(
                f"{lab} 0-{top:g} s" for lab, (_b, top) in zip(labels, windows, strict=True)))  # fmt: skip
    elif per_label is not None:
        windows = compute_windows_from_durations([per_label[lab] for lab in labels], tr)
        say("Windows from the events' own durations: " + ", ".join(
            f"{lab} {per_label[lab]:g} s event -> 0-{top:g} s"
            for lab, (_bot, top) in zip(labels, windows, strict=True)))  # fmt: skip
    else:
        raise ValueError("give -window (the curves do not carry it)")
    assert tr is not None  # -input always has one; the iresp route derived it above

    out = {}
    for lab, (bot, top) in zip(labels, windows, strict=True):
        if abs(bot) > 1e-6 * max(tr, 1.0):
            raise ValueError(
                f"condition {lab}: -window starts at {bot:g} s, but the model puts the event on "
                "the response's first sample; use a window that starts at 0"
            )
        n = int(round((top - bot) / tr)) + 1
        if n < 3:
            raise ValueError(
                f"condition {lab}: window 0-{top:g} s is only {n} knot(s) at TR {tr:g}; "
                "the laminar model needs a curve"
            )
        out[lab] = np.linspace(0.0, bot + (n - 1) * tr, n)
    return out


def _event_durations(timing, labels: list[str]) -> dict[str, float]:
    """Each condition's own event length, from the events that defined it."""
    if not timing.durations_given:
        raise ValueError("-onsets without durations in the files: give -duration")
    return {lab: float(timing.durations[labels.index(lab)]) for lab in labels}


def _comparison_times(cmp, times: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(baseline, test)`` sample grids for one comparison.

    The two SIDES may differ -- that is the point of per-condition windows -- but the
    labels averaged within one side must share a window, or there is no grid their
    mean is sampled on.
    """
    out = []
    for side, labs in (("baseline", cmp.baseline), ("test", cmp.test)):
        widths = {lab: len(times[lab]) for lab in labs}
        if len(set(widths.values())) > 1:
            raise ValueError(
                f"comparison {cmp.name}: its {side} averages conditions sampled over "
                "different windows ("
                + ", ".join(f"{k} {v} knots" for k, v in sorted(widths.items()))
                + "); give one -window for them, or compare them separately"
            )
        out.append(times[labs[0]])
    return out[0], out[1]


def _comparison_duration(cmp, args, per_label: dict[str, float] | None) -> list[float]:
    """``[baseline, test]`` event lengths for one comparison.

    A comparison's sides may each average several labels, so a side is only
    well defined when its labels share a duration -- averaged curves of events of
    different lengths are not a response to any one event.
    """
    if args.duration is not None:
        given = [float(v) for v in args.duration]
        return given * 2 if len(given) == 1 else given
    if per_label is None:
        raise ValueError("give -duration with -iresp_prefix (the curves do not carry it)")
    out = []
    for side, labs in (("baseline", cmp.baseline), ("test", cmp.test)):
        vals = {lab: per_label[lab] for lab in labs}
        if max(vals.values()) - min(vals.values()) > 1e-6:
            raise ValueError(
                f"comparison {cmp.name}: its {side} averages conditions whose events differ "
                "in length ("
                + ", ".join(f"{k} {v:g}s" for k, v in sorted(vals.items()))
                + "); the averaged curve is not a response to one event, so give -duration"
            )
        out.append(next(iter(vals.values())))
    return out


def _iresp_cells(args, files, used, reads, columns, ks, shape3, times):
    """Column x depth curves by averaging per-voxel fits -- exact only because the
    curves came from a linear estimator (see -iresp_prefix).

    With per-condition windows ffs_deconvolve writes each condition at its OWN
    length, so the files legitimately differ; requiring one length is what used to
    refuse them. Its per-voxel-window branch instead pads every condition to the
    longest window, so a file may also be longer than its window asks for -- the
    extra samples are zeros and are trimmed, since they would otherwise enter the
    model as a response that returns to baseline and holds.
    """
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.laminar import columns as col

    curves_vox = {}
    for lab in used:
        img = load_nifti(files[lab])
        arr = np.asanyarray(img.dataobj)
        n_lab = arr.shape[3] if arr.ndim > 3 else 1
        keep = len(times[lab])
        if tuple(arr.shape[:3]) != shape3:
            raise ValueError(f"{files[lab]}: grid {arr.shape[:3]} is not {shape3}")
        if n_lab < keep:
            raise ValueError(f"{files[lab]}: {n_lab} sample(s), but its window asks for {keep}")
        curves_vox[lab] = np.asarray(arr.reshape(-1, n_lab)[reads.voxels][:, :keep], np.float64)
        del arr, img

    # Conditions may now have different lengths, so each is mixed on its own.
    col_curves, coverage, psfs = {}, {}, {}
    for k in ks:
        col_curves[k] = {}
        for lab in used:
            cc = col.column_curves(reads, columns, curves_vox[lab], k)
            col_curves[k][lab] = cc.values
            coverage[k], psfs[k] = cc.coverage, cc.psf
    return col_curves, coverage, psfs


def _fit_cells(args, timing, input_files, reads, columns, ks, times, labels,
               affine, shape3, device, say):  # fmt: skip
    """Mix the voxel timeseries into column x depth cells and fit each cell once."""
    import torch

    from fastfuncstuff.cli_utils import (
        append_nuisance_blocks,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
    )
    from fastfuncstuff.design.builder import build_per_run_task_designs
    from fastfuncstuff.design.polort import polort_per_run, resolve_polort
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.glm.smooth_basis import penalty_matrix
    from fastfuncstuff.laminar import cells as cl

    cpu = torch.device("cpu")
    # Mask at load to exactly the ribbon the reads touch: the rest of the brain's
    # timeseries is never resident ([[RAM-careful policy]]).
    t0 = time.time()
    load = load_and_preprocess_runs(
        input_files=input_files, tr=args.tr, mask_array=cl.ribbon_mask(reads, shape3),
        blur_fwhm=args.do_blur, do_scale=args.do_scale, device=device, force_cpu=True,
        verbose=args.verb >= 1, load_threads=args.load_threads,
        drop_first=args.drop_first, drop_last=args.drop_last,
    )  # fmt: skip
    if tuple(load.volume_shape) != shape3:
        raise ValueError(f"{input_files[0]}: grid {load.volume_shape} is not {shape3}")
    tr = float(load.tr)
    n_tp_per_run = [
        b - a
        for a, b in zip(
            load.run_starts, list(load.run_starts[1:]) + [load.n_timepoints], strict=True
        )  # fmt: skip
    ]
    say(f"Loaded {load.n_voxels:,} ribbon voxel(s) x {load.n_timepoints:,} TR "
        f"(TR {tr:g}, {len(n_tp_per_run)} run(s), {time.time() - t0:.0f} s)")  # fmt: skip
    rows = cl.rows_for_reads(reads, load.mask_flat)

    # Each condition's knots are its own window at the data's TR.
    widths = {lab: len(times[lab]) for lab in labels}
    for lab in labels:
        spacing = float(times[lab][1] - times[lab][0])
        if abs(spacing - tr) > 1e-6 * max(tr, 1.0):
            raise ValueError(
                f"condition {lab}: the window gives knots spaced {spacing:g} s, but the "
                f"data's TR is {tr:g} s"
            )
    offset = args.microtime_offset
    if offset is None:
        offset = 0.0
    task = build_per_run_task_designs(
        timing.all_onsets, n_tp_per_run, tr, basis="TENT",
        condition_labels=labels,
        fir_window_s=[(float(times[lab][0]), float(times[lab][-1])) for lab in labels],
        microtime_offset=float(offset), device=cpu,
    )  # fmt: skip
    n_task = task.per_run[0].shape[1]

    degrees = polort_per_run(
        resolve_polort(None, n_tp_per_run, tr, verbose=False)
        if str(args.polort).strip().upper() == "A"
        else int(args.polort),
        len(n_tp_per_run),
    )
    # Per-run nuisance on the block diagonal ([[Block-diagonal nuisance]]).
    nuis_per_run = [
        construct_polynomial_matrix(n, max(d, 0), cpu) if d >= 0 else torch.zeros((n, 0))
        for n, d in zip(n_tp_per_run, degrees, strict=True)
    ]
    blocks = collect_nuisance_blocks(
        args, run_starts=list(load.run_starts), n_timepoints=load.n_timepoints,
        verbose=args.verb >= 1,
    )  # fmt: skip
    if blocks:
        nuis_per_run = append_nuisance_blocks(
            nuis_per_run, blocks, list(load.run_starts), load.n_timepoints
        )
    n_nuis = max(m.shape[1] for m in nuis_per_run)
    design = torch.zeros((load.n_timepoints, n_task + n_nuis * len(n_tp_per_run)))
    for i, (start, n) in enumerate(zip(load.run_starts, n_tp_per_run, strict=True)):
        sl = slice(start, start + n)
        design[sl, :n_task] = task.per_run[i]
        design[sl, n_task + i * n_nuis : n_task + i * n_nuis + nuis_per_run[i].shape[1]] = (
            nuis_per_run[i]
        )
    say(f"Design: {n_task} task column(s) ("
        + ", ".join(f"{lab} {w}" for lab, w in widths.items())
        + f" knots), {n_nuis * len(n_tp_per_run)} nuisance (polort {degrees})")  # fmt: skip

    penalty = penalty_matrix(
        args.smooth_penalty, [widths[lab] for lab in labels], [tr] * len(labels)
    )
    t0 = time.time()
    out = cl.fit_cell_curves(
        load.data, rows, reads, columns, ks, design, n_task, penalty, labels, times,
        list(load.run_starts), lam=args.tent_lambda, signal_r2=args.smooth_signal_r2,
        cell_lambda=args.cell_lambda, device=device, verbose=args.verb >= 1,
    )  # fmt: skip
    n_cells = sum(columns.n_columns * k for k in ks)
    say(f"Cell fits: {n_cells:,} column x depth cell(s) over K = {ks} "
        f"({time.time() - t0:.0f} s)")  # fmt: skip
    for k in ks:
        good = np.isfinite(out.edf[k])
        if good.any():
            say(f"  K={k}: effective knots {np.median(out.edf[k][good]):.1f} of {n_task}, "
                f"median R2 {np.nanmedian(out.r2[k][good]):.3f}")  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    try:
        return _run(args, t_start)
    except (OSError, ValueError) as exc:
        print(f"ffs_laminascope: {exc}", file=sys.stderr)
        return 1


def _run(args, t_start) -> int:  # noqa: C901 -- a pipeline, read top to bottom
    import json

    import torch

    from fastfuncstuff.cli.laminar import parse_modulation
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.laminar import columns as col
    from fastfuncstuff.laminar.comparisons import make_comparison, read_comparisons
    from fastfuncstuff.laminar.experiment import layer_model_names, layer_model_targets
    from fastfuncstuff.laminar.io import two_condition_batch
    from fastfuncstuff.laminar.pipeline import depth_names, reference_p0
    from fastfuncstuff.laminar.scope import fit_columns
    from fastfuncstuff.processing.roi_depth import (
        CLAIM_ANY,
        depth_list,
        options_from_args,
        read_depth_rois,
    )
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    say = print if args.verb >= 1 else (lambda *_a, **_k: None)
    modulation = parse_modulation(args.modulation)
    device = setup_device(args.device, float64_only="ffs_laminascope")

    if bool(args.input) == bool(args.iresp_prefix):
        raise ValueError("give -input (fit the curves here) or -iresp_prefix (read them)")
    timing = None
    files: dict[str, str] = {}
    if args.input:
        timing, input_files = _resolve_timing(args, say)
        labels = list(timing.condition_labels)
    else:
        files = _iresp_labels(args.iresp_prefix)
        labels = list(files)
    comps = read_comparisons(args.spec, labels) if args.spec else []
    for name, base, test in args.compare or []:
        comps.append(make_comparison(name, base, test, labels))
    if not comps:
        raise ValueError("give -spec or -compare")
    all_names, all_targets = layer_model_names(args.N), layer_model_targets(args.N)
    for cmp in comps:
        cmp.models = cmp.models or args.models
        bad = [mm for mm in (cmp.models or []) if mm not in all_names]
        if bad:
            raise ValueError(
                f"comparison {cmp.name}: unknown model(s) {bad}; the space is {all_names}"
            )
    used = sorted({lab for c in comps for lab in c.baseline + c.test}, key=labels.index)

    ref = load_nifti(input_files[0] if args.input else files[used[0]])
    affine = np.asarray(ref.affine, np.float64)
    shape3 = tuple(int(n) for n in ref.shape[:3])
    # Each condition's own event length, so a comparison whose sides differ can drive
    # the model's two events differently and sample them over different windows.
    per_label = _event_durations(timing, labels) if args.input else None
    n_file = (
        {}
        if args.input
        else {
            lab: (
                int(load_nifti(files[lab]).shape[3]) if len(load_nifti(files[lab]).shape) > 3 else 1
            )
            for lab in labels
        }
    )
    # The knots are ours to choose with -input; with -iresp_prefix every file was
    # written at the LONGEST window and zero-padded, so a condition's own window
    # selects its real knots out of that padding, at the spacing those files imply.
    tr_hint = (float(args.tr) if args.tr else _header_tr(ref)) if args.input else None
    times = _resolve_windows(args, labels, tr_hint, per_label, n_file, say)
    mask = None
    if args.mask:
        mask = np.asanyarray(load_nifti(args.mask).dataobj).reshape(shape3) != 0

    logs: list[str] = []
    surfaces = resolve_ribbon_surfaces(
        args.fs_subj, args.white, args.pial, args.white_mesh, args.pial_mesh,
        args.cortex_label, hemi=args.hemi, log=logs.append,
    )  # fmt: skip
    if args.regions_atlas and args.regions_from:
        raise ValueError("give -regions_atlas or -regions_from, not both")
    if args.regions_atlas:
        columns = col.regions_from_atlas(surfaces, args.regions_atlas)
        say(f"Regions: {columns.n_columns} atlas region(s) from {', '.join(args.regions_atlas)}")
    elif args.regions_from:
        t0 = time.time()
        columns = col.regions_from_map(surfaces, args.regions_from, args.regions_thresh,
                                       absolute=args.regions_abs, min_area_mm2=args.regions_min_mm2)  # fmt: skip
        sizes = np.asarray(columns.members.sum(axis=1)).ravel()
        say(f"Regions: {columns.n_columns} cluster(s) of {args.regions_from} >= {args.regions_thresh:g}, "
            f"{int(np.median(sizes)):,} vertices median ({time.time() - t0:.0f} s)")  # fmt: skip
    else:
        if not args.responsive:
            raise ValueError(
                "give -responsive (columns centred on responsive cortex) or -regions_from"
            )
        resp_notes: list[str] = []
        resp = col.responsive_vertices(args.responsive, surfaces, threshold=args.responsive_thresh,
                                       absolute=not args.responsive_positive,
                                       cluster_size=args.responsive_cluster_size,
                                       cluster_nn=args.responsive_cluster_nn,
                                       notes=resp_notes)  # fmt: skip
        for note in resp_notes:
            say(note)
        logs += resp_notes
        if args.within:
            notes: list[str] = []
            inside = [np.zeros(len(s.white), bool) for s in surfaces]
            for r in read_depth_rois([args.within], surfaces, notes, CLAIM_ANY):
                for h, mbr in r.members.items():
                    inside[h] |= mbr
            resp = [a & b for a, b in zip(resp, inside, strict=True)]
        say(
            "Responsive vertices: "
            + ", ".join(f"{s.name} {int(r.sum()):,}" for s, r in zip(surfaces, resp, strict=True))
        )
        t0 = time.time()
        columns = col.make_columns(surfaces, resp, radius_mm=args.column_radius,
                                   spacing_mm=args.column_spacing, seed=args.seed)  # fmt: skip
        cover = columns.coverage_count()
        n_resp = int(sum(int(r.sum()) for r in resp))
        # A disk of -column_radius around every centre reaches well past the responsive
        # vertices the centres were placed on, and the painted maps cover all of it.
        say(f"Columns: {columns.n_columns:,} (radius {columns.radius_mm:g} mm, spacing "
            f"{columns.spacing_mm:g} mm) over {columns.vertex_id.size:,} vertices; columns per "
            f"vertex median {np.median(cover):.0f} ({time.time() - t0:.0f} s)")  # fmt: skip
        if n_resp:
            say(f"  the maps therefore cover {columns.vertex_id.size / n_resp:.1f}x the "
                f"{n_resp:,} responsive vertices (every vertex within {columns.radius_mm:g} mm "
                "of a centre); threshold them, do not read their extent as the mask")  # fmt: skip

    ks = depth_list(args)
    opts = options_from_args(args)
    t0 = time.time()
    reads = col.vertex_depth_reads(surfaces, columns, affine, shape3, ks, opts, mask=mask,
                                   psf=args.psf == "measured", device=device)  # fmt: skip
    say(f"Depth reads: K = {ks}, {reads.voxels.size:,} voxels ({time.time() - t0:.0f} s)")
    in_fov = np.ones(columns.n_columns)
    if args.regions_atlas or args.regions_from:
        columns, in_fov = col.trim_to_coverage(columns, reads, args.min_coverage)
        cut = in_fov < 0.999
        if cut.any():
            say("Regions trimmed to the FOV/mask: " + ", ".join(
                f"{columns.names[c] if columns.names else c} {100 * in_fov[c]:.0f}%"
                for c in np.flatnonzero(cut)))  # fmt: skip

    if args.input:
        cell = _fit_cells(args, timing, input_files, reads, columns, ks, times, labels,
                          affine, shape3, device, say)  # fmt: skip
        col_curves, coverage, psfs = cell.curves, cell.coverage, cell.psf
        logs += cell.notes
        for note in cell.notes:
            say(f"  {note}")
    else:
        col_curves, coverage, psfs = _iresp_cells(
            args, files, used, reads, columns, ks, shape3, times
        )

    cov_ok = np.all([coverage[k].min(1) >= args.min_coverage for k in ks], axis=0)
    finite = np.all(
        [np.isfinite(col_curves[k][lab]).all((1, 2)) for k in ks for lab in used], axis=0
    )
    if psfs[ks[0]] is not None:
        finite &= np.all([np.isfinite(psfs[k]).all((1, 2)) for k in ks], axis=0)
    fit_mask = cov_ok & finite
    if args.max_columns and fit_mask.sum() > args.max_columns:
        pick = np.random.default_rng(args.seed).choice(
            np.flatnonzero(fit_mask), args.max_columns, replace=False
        )
        fit_mask = np.zeros_like(fit_mask)
        fit_mask[pick] = True
    fit_idx = np.flatnonzero(fit_mask)
    say(f"Fitting {fit_idx.size:,} column(s); {int((~cov_ok).sum()):,} cut by the FOV/mask "
        f"(coverage < {args.min_coverage:g})")  # fmt: skip
    if fit_idx.size == 0:
        raise ValueError("no column has complete depth coverage")
    # Per condition: with per-condition windows they have different lengths.
    big = max(float(np.nanmax(np.abs(col_curves[ks[0]][lab][fit_idx]))) for lab in used)
    if big > 50:
        raise ValueError(
            "curves look like data units, not percent change (fit without -dont_scale)"
        )

    chunk = args.chunk or (1024 if device.type == "cuda" else 64)
    vascular_given = None
    if args.vascular_from:
        vascular_given = _vascular_from(args.vascular_from, columns, fit_idx)
        say(
            f"Vascular priors from {args.vascular_from[0]} ({args.vascular_from[1]}): "
            f"{int(np.isfinite(vascular_given['s_d'][0]).sum()):,} of {fit_idx.size:,} columns matched"
        )
    centres = np.stack([
        0.5 * (np.asarray(surfaces[int(h)].white[v]) + np.asarray(surfaces[int(h)].pial[v]))
        for h, v in zip(columns.centre_surface, columns.centre_vertex, strict=True)
    ])  # fmt: skip
    if args.duration is not None and len(args.duration) not in (1, 2):
        raise ValueError(f"-duration takes one value or two, got {len(args.duration)}")

    results = {}
    out_npz: dict[str, np.ndarray] = {}
    durations_used: dict[str, list[float]] = {}
    for cmp in comps:
        names = cmp.models or all_names
        targets = [all_targets[all_names.index(mm)] for mm in names]
        duration = _comparison_duration(cmp, args, per_label)
        durations_used[cmp.name] = duration
        t_base, t_test = _comparison_times(cmp, times)
        data_for_k, kern_for_k = {}, {}
        for k in ks:
            base = np.mean([col_curves[k][lab][fit_idx] for lab in cmp.baseline], axis=0)
            test = np.mean([col_curves[k][lab][fit_idx] for lab in cmp.test], axis=0)
            data_for_k[k] = two_condition_batch(
                base.transpose(0, 2, 1), test.transpose(0, 2, 1), (t_base, t_test),
                duration=duration, n_neuronal=args.N, gap=args.pad_rows,
                noise_sd=args.pad_sd, seed=args.pad_seed, delay_seconds=0.0,
                TE=args.TE, B0=args.B0, p0=reference_p0(), modulation=modulation,
            )  # fmt: skip
            if psfs[k] is not None:  # superficial first, as the model wants
                kern_for_k[k] = torch.as_tensor(psfs[k][fit_idx][:, ::-1, ::-1].copy())
            else:
                kern_for_k[k] = None
            out_npz[f"{cmp.name}_baseline_K{k}"] = base
            out_npz[f"{cmp.name}_test_K{k}"] = test
        dur_note = (f"{duration[0]:g} s" if duration[0] == duration[1]
                    else f"{duration[0]:g} s baseline / {duration[1]:g} s test")  # fmt: skip
        if len(t_base) != len(t_test):
            dur_note += f", windows 0-{t_base[-1]:g} / 0-{t_test[-1]:g} s"
        say(f"\n{cmp.name}: {' + '.join(cmp.test)}  vs  {' + '.join(cmp.baseline)}  "
            f"({len(names)} models x K {ks} x {fit_idx.size:,} columns, events {dur_note})")  # fmt: skip
        t0 = time.time()
        fits = fit_columns(data_for_k, kern_for_k, targets=targets, names=names,
                           free_energy=args.free_energy, max_iter=args.max_iter,
                           hyper_from_rows=args.hyperprior == "rows", device=device,
                           chunk=chunk, progress=args.verb >= 1, rescue=args.rescue,
                           model_space=args.model_space, vascular_prior=args.vascular_prior,
                           vascular_radius_mm=args.vascular_radius, centres=centres[fit_idx],
                           hemi=columns.centre_surface[fit_idx],
                           vascular_given=vascular_given)  # fmt: skip
        for k, ep in fits.empirical.items():
            if ep.shrink == 0.0:
                say(
                    f"  vascular prior K={k}: no pooled prior beats the columns' own "
                    f"(best summed dF {ep.grid_free_energy.max():.1f} nats) -- not applied"
                )
                continue
            say(f"  vascular prior K={k}: shrink {ep.shrink:.3g} of the prior variance; "
                f"s_d prior mean {np.percentile(ep.mean[:, 0], 5):.2f}..{np.percentile(ep.mean[:, 0], 95):.2f}")  # fmt: skip
        say(
            f"  fitted in {time.time() - t0:.0f} s ({(time.time() - t0) / fit_idx.size:.3f} s/column)"
        )
        results[cmp.name] = fits
        k_mid = ks[len(ks) // 2]
        base_mid = np.mean([col_curves[k_mid][lab][fit_idx] for lab in cmp.baseline], axis=0)
        baseline_peak = base_mid.mean(1).max(1)
        _write_comparison(args.prefix, cmp, fits, columns, fit_idx, surfaces, affine, shape3,
                          depth_names(args.N), device, say, in_fov=in_fov[fit_idx],
                          baseline_peak=baseline_peak, gate_baseline=args.gate_baseline)  # fmt: skip
        _write_long(args, cmp, fits, columns, fit_idx, surfaces, depth_names(args.N), say,
                    in_fov=in_fov[fit_idx], baseline_peak=baseline_peak)  # fmt: skip
        for key in ("F", "p_model", "p_depth", "best", "B_mean", "B_sd", "B_p", "r2",
                    "stalled", "failed", "nesting", "s_d", "refit"):  # fmt: skip
            out_npz[f"{cmp.name}_{key}"] = getattr(fits, key)
        for name, (mean, sd) in fits.vascular.items():
            out_npz[f"{cmp.name}_post_{name}_mean"] = mean
            out_npz[f"{cmp.name}_post_{name}_sd"] = sd

    m = columns.members.tocsr()
    np.savez_compressed(
        f"{args.prefix}_columns.npz", centre_surface=columns.centre_surface,
        centre_vertex=columns.centre_vertex, members_data=m.data, members_indices=m.indices,
        members_indptr=m.indptr, members_shape=np.asarray(m.shape),
        vertex_surface=columns.vertex_surface, vertex_id=columns.vertex_id,
        surfaces=np.asarray([s.name for s in surfaces]), fitted=fit_idx,
        labels=np.asarray(labels), ks=np.asarray(ks),
        **{f"times_{lab}": times[lab] for lab in labels},
        **{f"coverage_K{k}": coverage[k] for k in ks},
        **{f"psf_K{k}": psfs[k] for k in ks if psfs[k] is not None},
        # Kept for readers written against one shared grid; absent when they differ.
        **({"times": times[labels[0]]}
           if len({len(t) for t in times.values()}) == 1 else {}),
        **out_npz,
    )  # fmt: skip
    with open(f"{args.prefix}.json", "w") as fh:
        json.dump({
            "tool": "ffs_laminascope", "argv": sys.argv,
            **({"input": input_files, "events": args.events, "onsets": args.onsets,
                "windows": {lab: [0.0, float(t[-1])] for lab, t in times.items()},
                "durations": durations_used, "tr": tr_hint,
                "log10_lambda": cell.log10_lambda, "cell_lambda": args.cell_lambda,
                "smooth_penalty": args.smooth_penalty, "polort": args.polort,
                "do_scale": args.do_scale}
               if args.input else {"iresp": {lab: files[lab] for lab in used},
                                   "windows": {lab: [0.0, float(t[-1])]
                                               for lab, t in times.items()},
                                   "durations": durations_used}),
            "comparisons": [vars(c) for c in comps], "K": ks, "columns": columns.n_columns,
            "fitted": int(fit_idx.size), "radius_mm": columns.radius_mm,
            "spacing_mm": columns.spacing_mm, "depth_method": opts.method, "psf": args.psf,
            "responsive": args.responsive, "responsive_thresh": args.responsive_thresh,
            "responsive_cluster_size": args.responsive_cluster_size,
            "responsive_cluster_nn": args.responsive_cluster_nn,
            "device": str(device), "modulation": modulation, "model_space": args.model_space, "vascular_prior": args.vascular_prior,
            "vascular_radius_mm": args.vascular_radius, "notes": logs + reads.notes,
            "timing_s": {c: r.timing for c, r in results.items()},
        }, fh, indent=2, default=str)  # fmt: skip
    say(f"\nffs_laminascope: done in {time.time() - t_start:.0f} s")
    return 0


def _vascular_from(spec, columns, fit_idx):
    """Per fitted column, (mean, variance) of s_d and al_d from another run's npz."""
    path, cmp = spec
    z = np.load(path)
    key = f"{cmp}_post_s_d_mean"
    if key not in z.files:
        raise ValueError(
            f"{path}: no posterior for comparison {cmp!r} (have: "
            + ", ".join(sorted({f.split("_post_")[0] for f in z.files if "_post_" in f}))
            + ")"
        )
    fitted = z["fitted"]
    other = {
        (int(z["centre_surface"][c]), int(z["centre_vertex"][c])): j for j, c in enumerate(fitted)
    }
    out = {}
    for name in ("s_d", "al_d"):
        mean_o, sd_o = z[f"{cmp}_post_{name}_mean"], z[f"{cmp}_post_{name}_sd"]
        mean = np.full(fit_idx.size, np.nan)
        var = np.full(fit_idx.size, np.nan)
        for i, c in enumerate(fit_idx):
            j = other.get((int(columns.centre_surface[c]), int(columns.centre_vertex[c])))
            if j is not None:
                mean[i], var[i] = mean_o[j], sd_o[j] ** 2
        out[name] = (mean, var)
    return out


def _write_long(args, cmp, fits, columns, fit_idx, surfaces, dnames, say, *, in_fov,
                baseline_peak) -> None:  # fmt: skip
    from pathlib import Path

    from fastfuncstuff.laminar.scope import write_long_table
    from fastfuncstuff.surface.mesh import vertex_areas

    areas = {}
    for h, s in enumerate(surfaces):
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        areas[h] = vertex_areas(mid, np.asarray(s.faces), mid.shape[0])
    m = columns.members.tocsr()
    n_vert, area, names, hemis = [], [], [], []
    for c in fit_idx:
        vi = m[c].indices
        n_vert.append(vi.size)
        area.append(
            float(
                sum(
                    areas[int(h)][v]
                    for h, v in zip(columns.vertex_surface[vi], columns.vertex_id[vi], strict=True)
                )
            )
        )
        hemis.append(surfaces[int(columns.centre_surface[c])].name)
        names.append(columns.names[c] if columns.names else f"{hemis[-1]}.region{c}")
    path = write_long_table(
        f"{args.prefix}_{cmp.name}_regions_long.tsv", fits,
        subject=args.subject or Path(args.prefix).name, comparison=cmp.name,
        region_ids=np.asarray(fit_idx), names=names, hemis=hemis,
        n_vertices=np.asarray(n_vert), area_mm2=np.asarray(area), depth_names=dnames,
        in_fov=in_fov, baseline_peak=baseline_peak,
    )  # fmt: skip
    say(f"  wrote {path}")


def _write_comparison(prefix, cmp, fits, columns, fit_idx, surfaces, affine, shape3,
                      dnames, device, say, *, in_fov, baseline_peak, gate_baseline) -> None:  # fmt: skip
    """The per-column maps and table.

    ``baseline_peak`` (percent, the depth-averaged baseline curve's peak) and ``in_fov``
    travel with every map and not only the region table: a column with no response has
    nothing for the model to explain, and its B is noise however confidently flagged.
    """
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data
    from fastfuncstuff.laminar import columns as col

    n_cols, n_d = columns.n_columns, fits.p_depth.shape[1]

    def full(a, fill=np.nan):
        out = np.full((n_cols, *a.shape[1:]), fill, dtype=np.float64)
        out[fit_idx] = a
        return out

    valid = np.zeros(n_cols, bool)
    valid[fit_idx] = fits.best >= 0
    best_full = np.full(n_cols, -1)
    best_full[fit_idx] = fits.best
    p_best = full(fits.p_model[np.arange(fit_idx.size), np.clip(fits.best, 0, None)])
    null = fits.models.index("null") if "null" in fits.models else None
    log_bf = (full(np.max(fits.F, 1) - fits.F[:, null]) if null is not None
              else np.full(n_cols, np.nan))  # fmt: skip
    # ONE brick to mask with.  The gating order was documented and nobody could apply it:
    # assembling four bricks by hand in 3dcalc is a different skill from reading a map, and
    # the thresholds are the tool's own claims, not the reader's.  So the tool states them.
    bp_full, fov_full = full(baseline_peak), full(np.asarray(in_fov, float))
    sound = np.zeros(n_cols, bool)
    sound[fit_idx] = ~(fits.stalled.any((1, 2)) | fits.failed.any((1, 2)))
    gate = (valid & sound & (bp_full >= gate_baseline) & (fov_full >= 0.9)
            & (full(fits.nesting) <= 3.0))  # fmt: skip
    per_column = np.concatenate(
        [full(fits.p_depth), full(fits.B_mean), full(fits.B_p), p_best[:, None],
         log_bf[:, None], full(fits.r2.max(1))[:, None], bp_full[:, None],
         fov_full[:, None], full(fits.nesting)[:, None],
         np.where(valid, gate, np.nan)[:, None]], axis=1,
    )  # fmt: skip
    names = ([f"p_mod_{d}" for d in dnames] + [f"B_{d}" for d in dnames]
             + [f"PB_{d}" for d in dnames]
             + ["p_best", "logBF_vs_null", "r2", "baseline_peak", "in_fov", "nesting_nats",
                "gate"])  # fmt: skip
    say(f"  gate: {int(gate.sum()):,} of {fit_idx.size:,} fitted column(s) pass "
        f"(baseline_peak >= {gate_baseline:g}%, in_fov >= 0.9, nesting <= 3 nats, "
        "no stalled or failed fit); mask every other map with it")  # fmt: skip
    vert, n_cov = col.paint_vertices(columns, per_column, valid)
    winner, agree = col.vertex_winner(columns, best_full, len(fits.models), valid)
    # The winner is painted as a BITMASK over depths, not as the model's index in a list:
    # a volume carries no legend, so an index is unreadable in any viewer, while bit 2^i
    # per modulated depth is self-describing and arithmetic (5 = superficial+deep, and
    # "superficial modulated" is simply bit 0 set).
    bits = np.array([sum(1 << i for i, d in enumerate(dnames) if d in m.split("+"))
                     for m in fits.models])  # fmt: skip
    win_bits = np.where(winner >= 0, bits[np.clip(winner, 0, None)], np.nan)
    say("  winner brick is a depth bitmask: "
        + ", ".join(f"{1 << i}={d}" for i, d in enumerate(dnames))
        + f" (so {sum(1 << i for i in range(len(dnames)))}=all, 0=null)")  # fmt: skip
    vmaps = np.concatenate([vert, win_bits[:, None], agree[:, None], n_cov[:, None]], 1)
    vnames = names + ["winner", "agreement", "n_columns"]
    meta_base = {"tool": "ffs_laminascope", "comparison": cmp.name, "maps": ",".join(vnames),
                 "winner_models": ",".join(fits.models),
                 "winner_bits": ",".join(f"{b}={m}" for b, m in zip(bits, fits.models, strict=True)),
                 "depths": ",".join(dnames)}  # fmt: skip
    for h, s in enumerate(surfaces):
        if not np.any(columns.vertex_surface == h):
            continue
        data = col.full_surface(columns, vmaps, h, len(s.white))
        meta = dict(
            meta_base, hemi=s.name, mesh_fingerprint=mesh_fingerprint(s.faces, len(s.white))
        )
        path = f"{prefix}_{cmp.name}.{s.name}.func.gii"
        save_gifti_data(path, data, meta, time_series=False)
        say(f"  wrote {path}")
    vol = col.paint_ribbon(surfaces, columns, vmaps, affine, shape3, device=device)
    # Outside the columns there is no estimate, and B / p_depth / logBF are all legitimately
    # ~0 INSIDE them: zero-filling made "no column here" unreadable against "no effect here",
    # which is the whole question a thresholded map is asked.  NaN is what the .func.gii
    # already carries.  n_columns keeps its 0, which is a real count and not a missing value.
    vol = vol.reshape(*shape3, -1).astype(np.float32)
    if "n_columns" in vnames:
        j = vnames.index("n_columns")
        vol[..., j] = np.nan_to_num(vol[..., j], nan=0.0)
    path = f"{prefix}_{cmp.name}_ribbon.nii.gz"
    save_nifti(vol, output_path=path, reference_img=None, affine=affine, brick_labels=vnames)
    say(f"  wrote {path}")

    tsv = f"{prefix}_{cmp.name}_columns.tsv"
    with open(tsv, "w") as fh:
        fh.write("column\themi\tcentre_vertex\tbest\tp_best\t"
                 + "\t".join(f"p_{mm}" for mm in fits.models) + "\t"
                 + "\t".join(f"p_mod_{d}\tB_{d}\tB_sd_{d}\tPB_{d}" for d in dnames)
                 + "\tr2_best\ts_d\tbaseline_peak\tin_fov\tlogBF_vs_null"
                 + "\tstalled\tfailed\tnesting_nats\tgate\n")  # fmt: skip
        for j, c in enumerate(fit_idx):
            b = int(fits.best[j])
            fh.write(
                f"{c}\t{surfaces[int(columns.centre_surface[c])].name}\t{columns.centre_vertex[c]}\t"
                f"{fits.models[b] if b >= 0 else 'none'}\t{fits.p_model[j, max(b, 0)]:.4g}\t"
                + "\t".join(f"{v:.4g}" for v in fits.p_model[j]) + "\t"
                + "\t".join(f"{fits.p_depth[j, i]:.4g}\t{fits.B_mean[j, i]:.4g}\t"
                            f"{fits.B_sd[j, i]:.4g}\t{fits.B_p[j, i]:.4g}" for i in range(n_d))
                + f"\t{fits.r2[j, max(b, 0)]:.4g}\t{fits.s_d[j]:.4g}\t"
                f"{baseline_peak[j]:.4g}\t{in_fov[j]:.4g}\t{log_bf[c]:.4g}\t"
                f"{int(fits.stalled[j].any())}\t"
                f"{int(fits.failed[j].any())}\t{fits.nesting[j]:.3g}\t{int(gate[c])}\n"
            )  # fmt: skip
    say(f"  wrote {tsv}")
    if n_bad := int((fits.nesting > 10).sum()):
        say(f"  ⚠️  {n_bad} column(s) with a model >10 nats below one it contains (stalled fits)")


if __name__ == "__main__":
    sys.exit(main())
