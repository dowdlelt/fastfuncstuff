"""Declarative search space for the nonlinear registration backends.

The backbone of `ffs_tunewarp` (see [[automatic registration tuning]]): one
table describing what each backend exposes, what values are worth trying, and
which of them a given kind of registration should actually tune. The search, the
help text, the recipes and the reproduce-this-trial machinery all read *this*
rather than hardcoding flags, so adding a knob is a one-line change in one file.

Two things are deliberately separated:

* the **backend spec** — what the tool can do, which is a property of the tool;
* the **recipe** — what is worth tuning for a kind of data, which is a property
  of the problem.

That split matters because the right answer is not global. Warping a T1 to MNI
is a negotiation between two genuinely different objects, where heavy
regularization is usually right. Warping EPI to EPI is the *same brain*, where
small details carry the signal and over-smoothing throws away the thing you
were trying to fix. A recipe encodes that; a default cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


# Below this sigma a discrete Gaussian kernel degenerates: `cost._gauss_kernel_1d`
# floors its radius at 1 voxel, and at sigma <= ~0.15 both wings underflow to zero,
# leaving [0, 1, 0] -- an identity convolution. 0.25 is the first sigma whose wings
# survive in float32, so it is the smallest value that is *not* just a slow no-op.
GAUSS_SIGMA_RESOLUTION = 0.25


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob on one backend.

    ``values`` is the coarse grid actually searched. These are explicit lists
    rather than (min, max, n) because the useful settings are rarely uniformly
    spaced — regularization matters most near zero, patch sizes are odd
    integers, and iteration counts are schedules rather than scalars.
    """

    name: str  # dotted key, e.g. "formwarp.total_var"
    flag: str  # the CLI flag it renders to
    values: tuple[Any, ...]
    default: Any
    role: str  # "regularization" | "schedule" | "metric" | "effort" | "model"
    help: str = ""
    # Field name on the backend's config dataclass, when it differs from the CLI
    # spelling. The search drives the library in-process, so it needs both: the
    # attribute to set, and the flag to print for a reproducible command line.
    attr: str = ""
    # "x" renders a tuple schedule as 100x70x40 for the command line.
    fmt: str = ""
    # Hard (min, max) the adaptive search may never expand past, independent of
    # the listed grid. A magnitude gets a zero floor for free; this is for the
    # knobs where going further is *meaningless* rather than merely unusual --
    # below minpatch 5 an 8-voxel patch carries 24 cubic parameters, so the fit is
    # underdetermined and "passes" only because the box constraint bounds it.
    bounds: tuple[float | None, float | None] = (None, None)
    # Smallest magnitude that is distinguishable from zero for this knob. Not a
    # bound -- 0.0 itself stays legal -- but the search may not propose a value in
    # the open interval (0, resolution), because the backend cannot act on one.
    #
    # Bug of record: every Gaussian-sigma knob has such a zone. `_gauss_kernel_1d`
    # floors its radius at 1 voxel, so for sigma <= ~0.15 both wings underflow and
    # the kernel is exactly [0, 1, 0] -- bit-identical to sigma=0, at the price of
    # a conv3d. On a 7T epi2epi run the adaptive search bisected [0, 0.5] down to
    # 0.0078 and spent ~15 fits there; `-effects` then reported total_sigma=0.0078
    # as the best level with "100% of subjects agree", which was the other knobs'
    # effect wearing a dead knob's name.
    resolution: float | None = None

    @property
    def config_attr(self) -> str:
        return self.attr or self.key

    @property
    def backend(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def key(self) -> str:
        return self.name.split(".", 1)[1]

    def render(self, value: Any) -> list[str]:
        """This parameter as command-line tokens."""
        if isinstance(value, bool):
            return [self.flag] if value else []
        if self.fmt == "x" and isinstance(value, (list, tuple)):
            return [self.flag, "x".join(str(v) for v in value)]
        if isinstance(value, (list, tuple)):
            return [self.flag, *[str(v) for v in value]]
        return [self.flag, str(value)]


@dataclass(frozen=True)
class BackendSpec:
    """A registration backend: its command, its knobs, and its fixed arguments."""

    name: str
    command: str
    params: tuple[ParamSpec, ...]
    # Metric flag differs per tool; recipes set the value, this says how to pass it.
    metric_flag: str = ""
    warp_flag: str = "-save_warp"
    notes: str = ""

    def param(self, key: str) -> ParamSpec:
        for p in self.params:
            if p.key == key:
                return p
        raise KeyError(f"{self.name} has no tunable parameter {key!r}")

    def keys(self) -> list[str]:
        return [p.key for p in self.params]


# --- qwarp ------------------------------------------------------------------
# Under the Gauss-Newton solver the fine levels are half the fit: measured on a
# 193^3 T1->MNI pair, 25.1 s/fit at -minpatch 25 against 44.9 at 5. (The earlier
# note here claimed 45.0 vs 45.7 and predates the solver change.) That time is
# not detail-resolving -- see the -minpatch docstring for what it actually buys.

QWARP = BackendSpec(
    name="qwarp",
    command="ffs_qwarp",
    metric_flag="-cost",
    params=(
        ParamSpec(
            "qwarp.penfac",
            "-penfac",
            (0.0, 0.008, 0.033, 0.1, 0.3),
            0.033,
            "regularization",
            "Jacobian-energy penalty. The knob that decides folding.",
            attr="penalty_factor",
        ),
        ParamSpec(
            "qwarp.minpatch",
            "-minpatch",
            (5, 9, 13, 19, 25),
            13,
            "schedule",
            "Finest patch size in voxels. NOT simply 'smaller resolves more detail': "
            "on a 193^3 T1->MNI grid (5 subjects x 3 hfactor_q, every fit) the levels "
            "from 19 to 13 over-COMPRESS the field -- 1st pct det(J) falls 0.274 -> "
            "0.200, under the 0.25 gate -- and the levels below 9 relax it back to "
            "0.274, trading image match for regularity (lncc -0.677 at 13, -0.634 at "
            "5). Stopping anywhere in 19..9 failed on 15 of 15 fits; 25 and 5 passed "
            "on 15 of 15. Run the fine levels or stop above them, never between. "
            "Floored at AFNI's 5; below that the patch has fewer voxels than the "
            "basis has parameters.",
            bounds=(5, None),
        ),
        ParamSpec(
            "qwarp.workhard",
            "-workhard",
            ((0, 2), (0, -1)),
            None,
            "effort",
            "Double-pass optimization over a level range.",
        ),
        ParamSpec(
            "qwarp.hfactor_q",
            "-hfactor_q",
            (0.35, 0.5, 0.7),
            0.5,
            "regularization",
            "Shrinks the per-patch displacement bound as patches get finer.",
        ),
    ),
)

# --- formwarp (ANTs SyN) ----------------------------------------------------
# total_var defaults to 0.0 because that is ANTs' own SyN[0.1,3,0] -- it
# regularizes the update field and not the total. Faithful, and on T1->MNI it
# permits folding (measured: 9879 folded voxels at 0.0, zero at 1.0), which is
# exactly the sort of thing a recipe should settle per data type.

FORMWARP = BackendSpec(
    name="formwarp",
    command="ffs_formwarp",
    metric_flag="-metric",
    params=(
        ParamSpec(
            "formwarp.total_var",
            "-total_var",
            (0.0, 0.5, 1.0, 2.0, 3.0),
            0.0,
            "regularization",
            "Elastic (total-field) smoothing. Off by default, per ANTs.",
            resolution=GAUSS_SIGMA_RESOLUTION,
        ),
        ParamSpec(
            "formwarp.update_var",
            "-update_var",
            (1.0, 2.0, 3.0, 4.0),
            3.0,
            "regularization",
            "Fluid (update-field) smoothing.",
            resolution=GAUSS_SIGMA_RESOLUTION,
        ),
        ParamSpec(
            "formwarp.grad_step",
            "-grad_step",
            (0.1, 0.25, 0.5),
            0.25,
            "effort",
            "Per-iteration step size, in voxels.",
        ),
        ParamSpec(
            "formwarp.iters",
            "-iters",
            ((100, 70, 40), (300, 210, 120), (600, 420, 240)),
            (300, 210, 120),
            "effort",
            "Per-level iteration CEILING. Not searched by default and rarely worth "
            "searching: it bounds the solver rather than shaping its answer. Set it "
            "high and read recommend_iterations() instead.",
            attr="iterations",
            fmt="x",
        ),
        ParamSpec(
            "formwarp.conv_window",
            "-conv_window",
            (5, 10, 20),
            10,
            "schedule",
            "Trailing-window size for early stopping. 0 (disable) is deliberately not "
            "in the ladder: with the ceiling set high it would run every iteration for "
            "no quality gain, and whether stopping is premature is already answerable "
            "from best_iter vs iters_run.",
            attr="convergence_window",
        ),
        ParamSpec(
            "formwarp.conv_threshold",
            "-conv_thresh",
            (1e-6, 1e-5, 1e-4),
            1e-6,
            "schedule",
            "Convergence slope threshold; larger stops sooner.",
            attr="convergence_threshold",
        ),
    ),
)

# --- optiwarp (optical flow) ------------------------------------------------
# One backend per force model: they do not share a parameter set, and screening
# them against each other on defaults would be exactly the mistake this tool
# exists to avoid.

_OW_SHARED = (
    ParamSpec(
        "optiwarp.total_sigma",
        "-total_sigma",
        (0.0, 0.5, 1.0, 2.0),
        1.0,
        "regularization",
        "Elastic (total-field) smoothing. On T1->MNI, 0.0 and 0.5 fold on every "
        "subject and 1.0 is the lowest that does not -- i.e. the default is "
        "already the best legal value, and lower merely scores better by folding.",
        resolution=GAUSS_SIGMA_RESOLUTION,
    ),
    ParamSpec(
        "optiwarp.update_sigma",
        "-update_sigma",
        (0.5, 1.0, 2.0),
        1.0,
        "regularization",
        "Fluid (update-field) smoothing.",
        resolution=GAUSS_SIGMA_RESOLUTION,
    ),
    ParamSpec(
        "optiwarp.match",
        "-match",
        ("localnorm", "gradmag", "meanstd"),
        "localnorm",
        "metric",
        "Intensity prep before the flow solve. A wrong match negates the force, "
        "so this is checked before any regularization knob.",
    ),
    ParamSpec(
        "optiwarp.iters",
        "-iters",
        ((100, 70, 40), (300, 210, 120), (600, 420, 240)),
        (300, 210, 120),
        "effort",
        "Per-level iteration CEILING. Not searched by default and rarely worth "
        "searching: it bounds the solver rather than shaping its answer. Set it high "
        "and read recommend_iterations() instead.",
        attr="iterations",
        fmt="x",
    ),
    ParamSpec(
        "optiwarp.conv_window",
        "-conv_window",
        (5, 10, 20),
        10,
        "schedule",
        "Trailing-window size for early stopping. 0 (disable) is deliberately not in "
        "the ladder: with the ceiling set high it would run every iteration for no "
        "quality gain, and whether stopping is premature is already answerable from "
        "best_iter vs iters_run.",
        attr="convergence_window",
    ),
    ParamSpec(
        "optiwarp.conv_threshold",
        "-conv_thresh",
        (1e-6, 1e-5, 1e-4),
        1e-6,
        "schedule",
        "Convergence slope threshold; larger stops sooner.",
        attr="convergence_threshold",
    ),
    ParamSpec(
        "optiwarp.max_step",
        "-max_step",
        (0.5, 1.0, 2.0),
        1.0,
        "effort",
        "Cap on the per-iteration displacement, in voxels.",
    ),
)

_OW_FORCE = {
    "demons": (
        ParamSpec(
            "optiwarp.demons_noise",
            "-demons_noise",
            (0.5, 1.0, 2.0),
            1.0,
            "model",
            "Intensity difference at which the demons force is damped.",
        ),
    ),
    "lk": (
        ParamSpec(
            "optiwarp.lk_radius", "-lk_radius", (1, 2, 3), 2, "model", "LK window half-width."
        ),
        ParamSpec(
            "optiwarp.lk_reg",
            "-lk_reg",
            (0.001, 0.01, 0.1),
            0.01,
            "model",
            "Ridge on the LK structure tensor.",
        ),
    ),
    "hs": (
        ParamSpec(
            "optiwarp.hs_alpha",
            "-hs_alpha",
            (0.5, 1.0, 2.0),
            1.0,
            "model",
            "Horn-Schunck smoothness weight.",
        ),
        ParamSpec(
            "optiwarp.hs_iters",
            "-hs_iters",
            (10, 20, 40),
            20,
            "model",
            "Jacobi iterations per flow solve.",
        ),
    ),
}


def _optiwarp(force: str) -> BackendSpec:
    return BackendSpec(
        name=f"optiwarp_{force}",
        command="ffs_optiwarp",
        metric_flag="-metric",
        params=_OW_SHARED + _OW_FORCE[force],
        notes=f"-force {force}",
    )


BACKENDS: dict[str, BackendSpec] = {
    b.name: b
    for b in (
        QWARP,
        FORMWARP,
        _optiwarp("demons"),
        _optiwarp("lk"),
        _optiwarp("hs"),
    )
}

# The per-patch solver the tuner drives qwarp with. Gauss-Newton because the tuner
# is throughput-bound and comparing settings, not producing a final warp: on a
# 193^3 fit it took lpa from 96.7s to 15.9s while scoring slightly *better*, which
# is what lets qwarp take part in a budget at all. Defined once and used both to
# configure the in-process run and to render the reproducible command, so the two
# cannot drift -- the failure mode that hid `-conv_threshold` for a whole run.
QWARP_TUNE_OPTIMIZER = "gn"

# Fixed arguments a backend always gets: the force model for the flow engines, and
# the solver for qwarp. Kept out of the tunable set.
BACKEND_FIXED_ARGS: dict[str, list[str]] = {
    "qwarp": ["-optimizer", QWARP_TUNE_OPTIMIZER],
    "optiwarp_demons": ["-force", "demons"],
    "optiwarp_lk": ["-force", "lk"],
    "optiwarp_hs": ["-force", "hs"],
}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """What to tune, and how to judge it, for one kind of registration."""

    name: str
    describe: str
    optimize: str  # in-loop cost/metric, pinned (removes a whole search axis)
    evaluate_exclude: tuple[str, ...]  # extra functionals barred from voting
    pairing: str  # "one_base" (many sources vs one base) | "paired"
    tune: tuple[str, ...]  # dotted parameter keys worth searching
    # "same" | "cross" modality. Decides which functionals are even *meaningful*
    # judges: the signed ones (lss/lpc/lpc+) reward anti-correlation, so on a
    # same-modality pair they rank the worst warp first. This is not a nuance —
    # left unset it silently inverted three of fourteen votes on a T1->MNI run.
    contrast: str = "same"
    notes: str = ""
    backends: tuple[str, ...] = field(default_factory=lambda: tuple(BACKENDS))

    def panel(self) -> list[str]:
        """The metrics allowed to judge fits produced under this recipe.

        ``grid`` is on because the tuner scores whole volumes, so the
        neighbourhood metrics (lncc/ngf/mind) are available to it. A caller that
        only has scattered in-mask values must ask for ``grid=False`` instead.
        """
        from .metrics import panel_for

        return panel_for(self.optimize, self.contrast, tuple(self.evaluate_exclude), grid=True)


# The **stopping rule**, which is a real choice: it decides when "good enough" has
# been reached, and getting it wrong costs quality in one direction and time in the
# other.
#
# The iteration *ceiling* is deliberately NOT here. It is not a parameter of the same
# kind — it sets the stage rather than shaping the answer. Too few iterations is
# always wrong, and too many is free (early stopping ends the level, best-restore
# means exhaustion cannot return a worse warp), so there is no interior optimum to
# find and laddering over 60/100/160 asks a question with a known answer: more. The
# ceiling is instead set high in the engine defaults and *measured* — see
# `tunestore.recommend_iterations`, which reads the observed usage back and reports
# what the ceiling should be, rather than spending fits discovering it.
_ALL_EFFORT = (
    "qwarp.workhard",
    "formwarp.conv_window",
    "formwarp.conv_threshold",
    "optiwarp.conv_window",
    "optiwarp.conv_threshold",
)

_ALL_REG = (
    "qwarp.penfac",
    "qwarp.minpatch",
    "formwarp.total_var",
    "formwarp.update_var",
    "optiwarp.total_sigma",
    "optiwarp.update_sigma",
)

RECIPES: dict[str, Recipe] = {
    "MNI_T1": Recipe(
        name="MNI_T1",
        describe="T1 to an MNI template: same modality, different brains",
        optimize="lpa",
        # The descriptor metrics are barred from the same-modality juries. They cost
        # far more to evaluate than the rest of the panel, and what they buy is
        # invariance to a contrast inversion -- which is worth nothing when both
        # images are T1s. They stay in the registry and stay available to `-tune`
        # and to the cross-modal recipe; this is a default, not a verdict on them.
        evaluate_exclude=("mind", "mindssc"),
        contrast="same",
        pairing="one_base",
        tune=_ALL_REG + _ALL_EFFORT + ("qwarp.hfactor_q", "formwarp.grad_step"),
        notes=(
            "Measured on 5 FreeSurfer brains -> MNI152_2009 (480 fits, -metric "
            "lpa). optiwarp total_sigma 1.0 is the floor: 0.0 and 0.5 fold on "
            "100% of subjects for all three force models, and demons at "
            "1.0/0.5 is both near-best and 4.75x cheaper than formwarp. "
            "formwarp wants total_var 0.0-0.5 and update_var 3-4, and folds "
            "almost never (1 of 300). Score and gate point OPPOSITE ways on "
            "every regularization knob -- the top scorer is always the least "
            "regularization that is still legal -- so read the ranking as the "
            "fold boundary, not as an optimum. With the tool default -metric cc "
            "the same settings fold ~9900 voxels: on same-modality data the "
            "metric matters more than the regularization."
        ),
    ),
    "epi2t1": Recipe(
        name="epi2t1",
        describe="BOLD to that subject's own T1: cross-modal, same brain",
        optimize="lpc",
        evaluate_exclude=(),
        contrast="cross",
        pairing="paired",
        tune=_ALL_REG + _ALL_EFFORT + ("optiwarp.match",),
        notes=(
            "Cross-modal, so the metric is the fragile part: -match gradmag is "
            "the only optiwarp prep that survives a contrast inversion. The "
            "residual distortion is largely along the phase-encode axis, so a "
            "warp free in all three directions is over-parameterised."
        ),
    ),
    "epi2epi": Recipe(
        name="epi2epi",
        describe="BOLD to BOLD: same modality, same brain",
        optimize="lpa",
        # The descriptor metrics are barred from the same-modality juries. They cost
        # far more to evaluate than the rest of the panel, and what they buy is
        # invariance to a contrast inversion -- which is worth nothing when both
        # images are T1s. They stay in the registry and stay available to `-tune`
        # and to the cross-modal recipe; this is a default, not a verdict on them.
        evaluate_exclude=("mind", "mindssc"),
        contrast="same",
        pairing="paired",
        tune=_ALL_REG + _ALL_EFFORT + ("optiwarp.max_step", "formwarp.grad_step"),
        notes=(
            "The same brain twice, so the right answer is close to identity and "
            "small details carry the signal. Expect the optimum to sit at much "
            "LOWER regularization than MNI_T1 -- over-smoothing here discards "
            "exactly the residual misalignment you were trying to remove, and "
            "the tolerance for folding should be tighter, not looser, because "
            "there is no legitimate reason for large volume change."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Presets: what a tuning run concluded, in a form the tools can apply
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """Settings a tuning run measured as best for one (recipe, backend) pair.

    This is the point of the tuner. A search that ends in a table someone reads
    once has not changed anything; the finding has to become the setting the tool
    uses. So a run's conclusion lands here as data, `ffs_<backend> -type <recipe>`
    applies it, and the help text is generated from it — one fact, three uses.

    ``provenance`` is mandatory prose because a preset without it is unfalsifiable.
    Every one of these is "best on the data we happened to try", and the next person
    needs to know whether their data resembles it before trusting the numbers.
    """

    recipe: str
    backend: str
    config: dict[str, Any]
    provenance: str
    caveat: str = ""

    def describe(self) -> str:
        settings = " ".join(f"{k}={v}" for k, v in sorted(self.config.items())) or "(defaults)"
        out = f"{settings}\n    measured on: {self.provenance}"
        if self.caveat:
            out += f"\n    caveat: {self.caveat}"
        return out


# Keyed (recipe, backend). Grown by running ffs_tunewarp and pasting what
# `-export` prints; deliberately hand-committed rather than written at runtime, so
# that a default change is a reviewable diff with a provenance line attached.
PRESETS: dict[tuple[str, str], Preset] = {
    ("MNI_T1", "optiwarp_demons"): Preset(
        recipe="MNI_T1",
        backend="optiwarp_demons",
        config={"total_sigma": 1.0, "update_sigma": 0.5},
        provenance=(
            "5 FreeSurfer brains -> MNI152_2009_template, 480 fits, -metric lpa, "
            "2026-08-15. Near-best consensus and 4.75x cheaper than formwarp "
            "(2.4 s/fit vs 11.4)."
        ),
        caveat=(
            "Measured BEFORE the local fold guard existed, when total_sigma below "
            "1.0 folded on every subject. The guard makes lower values legal, so "
            "this floor should be re-measured rather than assumed."
        ),
    ),
    ("MNI_T1", "formwarp"): Preset(
        recipe="MNI_T1",
        backend="formwarp",
        config={"total_var": 0.5, "update_var": 4.0, "grad_step": 0.5},
        provenance=(
            "5 FreeSurfer brains -> MNI152_2009_template, 300 fits, -metric lpa, "
            "2026-08-15. total_var 0.0 scored better but was the only level that "
            "ever folded, so the shipped value is the best that never did."
        ),
        caveat=(
            "Same pre-fold-guard caveat as optiwarp. formwarp was otherwise the "
            "robust choice here: 299 PASS / 1 MARGINAL / 0 FAIL over 60 configs."
        ),
    ),
}


def preset_for(recipe: str, backend: str) -> Preset | None:
    """The tuned settings for this pair, if a run has ever concluded any."""
    return PRESETS.get((recipe, backend))


def preset_config_for_cli(recipe: str, backend: str) -> dict[str, Any]:
    """A preset as ``{cli_dest: value}``, ready to push onto parsed arguments.

    Keyed by the CLI's own attribute name rather than the ParamSpec key, because
    that is what the caller has: an argparse namespace.

    Values are converted to the form that namespace already holds, which is *after*
    argparse's ``type=`` ran. Schedules are the trap: the tuner stores ``iters`` as
    a tuple, but the CLI takes it as an ``"300x210x120"`` string and parses it
    downstream, so handing back the tuple would crash the very tool the preset
    exists to configure.
    """
    preset = preset_for(recipe, backend)
    if preset is None:
        return {}
    spec = BACKENDS[backend]
    out: dict[str, Any] = {}
    for key, value in preset.config.items():
        param = spec.param(key)
        if param.fmt == "x" and isinstance(value, (list, tuple)):
            out[param.flag.lstrip("-")] = "x".join(f"{v:g}" for v in value)
        else:
            out[param.flag.lstrip("-")] = value
    return out


def describe_presets(backend: str) -> str:
    """The ``-type`` help text for one backend, generated from the presets."""
    rows = [(r, p) for (r, b), p in sorted(PRESETS.items()) if b == backend]
    if not rows:
        return "No tuned presets exist for this backend yet."
    lines = []
    for recipe, preset in rows:
        lines.append(f"  {recipe}: {preset.describe()}")
    return "\n".join(lines)


def find_param(dotted: str) -> ParamSpec:
    """Look up a parameter by its dotted name, e.g. ``optiwarp.total_sigma``.

    ``optiwarp.*`` resolves against any one force model, since the three share
    the parameter set that name refers to.
    """
    prefix = dotted.split(".", 1)[0]
    candidates = [prefix] if prefix in BACKENDS else [b for b in BACKENDS if b.startswith(prefix)]
    for backend in candidates:
        for p in BACKENDS[backend].params:
            if p.name == dotted:
                return p
    known = sorted({p.name for s in BACKENDS.values() for p in s.params})
    raise KeyError(f"unknown parameter {dotted!r}. Known: {', '.join(known)}")


def parse_fix(items: list[str]) -> dict[str, Any]:
    """Parse ``-fix backend.key=value`` into ``{dotted_name: typed_value}``.

    The value is coerced to the type the parameter's own grid uses, so
    ``-fix formwarp.total_var=1`` pins a float and not the string "1".
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"-fix expects backend.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        p = find_param(dotted.strip())
        out[p.name] = _coerce(raw.strip(), p)
    return out


def _coerce(raw: str, p: ParamSpec) -> Any:
    proto = p.default if p.default is not None else p.values[0]
    if isinstance(proto, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(proto, (tuple, list)):
        parts = raw.replace(",", "x").split("x")
        return tuple(int(v) if str(proto[0]).isdigit() else float(v) for v in parts)
    if isinstance(proto, int) and not isinstance(proto, bool):
        return int(raw)
    if isinstance(proto, float):
        return float(raw)
    return raw


def fixed_for(fixed: dict[str, Any], backend: str) -> dict[str, Any]:
    """The subset of ``-fix`` values that apply to this backend, keyed by param key."""
    prefix = "optiwarp" if backend.startswith("optiwarp") else backend
    keys = {p.key for p in BACKENDS[backend].params}
    return {
        name.split(".", 1)[1]: v
        for name, v in fixed.items()
        if name.split(".", 1)[0] == prefix and name.split(".", 1)[1] in keys
    }


def with_overrides(
    recipe: Recipe, fix: dict[str, Any] | None = None, tune: list[str] | None = None
) -> Recipe:
    """A copy of ``recipe`` with extra knobs tuned and fixed ones dropped.

    Fixing a parameter removes it from the search *and* pins its value, which is
    the point: it is how you spend the budget on the knobs you do not already
    understand.
    """
    keys = list(recipe.tune)
    for dotted in tune or []:
        p = find_param(dotted)
        if p.name not in keys:
            keys.append(p.name)
    fixed_names = set(fix or {})
    keys = [k for k in keys if k not in fixed_names]
    return replace(recipe, tune=tuple(keys))


def resolve_tunable(recipe: Recipe, backend: str) -> list[ParamSpec]:
    """The parameters this recipe wants searched on this backend."""
    spec = BACKENDS[backend]
    # optiwarp.* keys apply to all three force-model backends.
    prefix = "optiwarp" if backend.startswith("optiwarp") else backend
    wanted = {k.split(".", 1)[1] for k in recipe.tune if k.split(".", 1)[0] == prefix}
    return [p for p in spec.params if p.key in wanted]


def render_command(
    backend: str,
    base: str,
    source: str,
    prefix: str,
    config: dict[str, Any],
    recipe: Recipe | None = None,
    save_warp: bool = True,
) -> list[str]:
    """Build the full command line for one trial.

    The same function produces the command that is *run* and the command that is
    *reported*, so what the user pastes to reproduce a trial is the thing that
    actually ran — not a reconstruction of it.
    """
    spec = BACKENDS[backend]
    cmd = [spec.command, "-base", base, "-source", source, "-prefix", prefix]
    cmd += BACKEND_FIXED_ARGS.get(backend, [])
    if recipe is not None and spec.metric_flag:
        cmd += [spec.metric_flag, _metric_for(spec, recipe)]
    for key in sorted(config):
        cmd += spec.param(key).render(config[key])
    if save_warp:
        cmd.append(spec.warp_flag)
    return cmd


def qwarp_cost_for(optimize: str) -> str:
    """The recipe's cost in qwarp's own vocabulary.

    One function so the fit and the reproduce command cannot disagree: "ls" is
    the allineate spelling of the whole-image Pearson that qwarp calls pearclp,
    and a recipe that pins it must run and print the same thing.
    """
    return "pearclp" if optimize == "ls" else optimize


def _metric_for(spec: BackendSpec, recipe: Recipe) -> str:
    """Translate the recipe's cost into what this backend calls it.

    qwarp speaks the allineate cost vocabulary; the flow/SyN tools have their own
    shorter list and have no lpc, so a cross-modal recipe falls back to the
    metric that behaves most like it.
    """
    if spec.name == "qwarp":
        return qwarp_cost_for(recipe.optimize)
    if recipe.optimize in ("lpa", "lpc"):
        return recipe.optimize
    return "cc"
