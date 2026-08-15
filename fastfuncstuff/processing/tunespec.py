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
# Cost lives in the coarse levels, not the fine ones: -minpatch buys much less
# time back than you would expect (measured 45.0s vs 45.7s), so schedule knobs
# here are about quality, not speed.

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
            "Finest patch size in voxels; smaller resolves more detail.",
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
        ),
        ParamSpec(
            "formwarp.update_var",
            "-update_var",
            (1.0, 2.0, 3.0, 4.0),
            3.0,
            "regularization",
            "Fluid (update-field) smoothing.",
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
            ((60, 40, 20), (100, 70, 40), (160, 120, 80)),
            (100, 70, 40),
            "effort",
            "Per-level iteration ceilings. Read alongside the recorded best_iter: a "
            "level whose best iterate is its last was starved; one whose best is "
            "early over-ran and fell back.",
            attr="iterations",
            fmt="x",
        ),
        ParamSpec(
            "formwarp.conv_window",
            "-conv_window",
            (0, 5, 10, 20),
            10,
            "schedule",
            "Trailing-window size for early stopping; 0 runs every iteration.",
            attr="convergence_window",
        ),
        ParamSpec(
            "formwarp.conv_threshold",
            "-conv_threshold",
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
    ),
    ParamSpec(
        "optiwarp.update_sigma",
        "-update_sigma",
        (0.5, 1.0, 2.0),
        1.0,
        "regularization",
        "Fluid (update-field) smoothing.",
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
        ((60, 40, 20), (100, 70, 40), (160, 120, 80)),
        (100, 70, 40),
        "effort",
        "Per-level iteration ceilings. Read alongside the recorded best_iter: a level "
        "whose best iterate is its last was starved, one whose best is early over-ran.",
        attr="iterations",
        fmt="x",
    ),
    ParamSpec(
        "optiwarp.conv_window",
        "-conv_window",
        (0, 5, 10, 20),
        10,
        "schedule",
        "Trailing-window size for early stopping; 0 runs every iteration.",
        attr="convergence_window",
    ),
    ParamSpec(
        "optiwarp.conv_threshold",
        "-conv_threshold",
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

# The fixed argument that selects the force model, kept out of the tunable set.
BACKEND_FIXED_ARGS: dict[str, list[str]] = {
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
        """The functionals allowed to judge fits produced under this recipe."""
        from .allcost import judge_panel

        return judge_panel(self.optimize, self.contrast, tuple(self.evaluate_exclude))


# Iteration budget and stopping rule. Conceptually separate from regularization —
# these decide how long the solver looks, not how stiff the answer is — but they
# interact with it, so they are searched jointly rather than pinned. The recorded
# LevelStats is what makes the result readable: an optimum at the top of the iters
# ladder with `starved` set means the ladder is short, not that more is better.
_ALL_EFFORT = (
    "qwarp.workhard",
    "formwarp.iters",
    "formwarp.conv_window",
    "formwarp.conv_threshold",
    "optiwarp.iters",
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
        evaluate_exclude=(),
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
        evaluate_exclude=(),
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


def _metric_for(spec: BackendSpec, recipe: Recipe) -> str:
    """Translate the recipe's cost into what this backend calls it.

    qwarp speaks the allineate cost vocabulary; the flow/SyN tools have their own
    shorter list and have no lpc, so a cross-modal recipe falls back to the
    metric that behaves most like it.
    """
    if spec.name == "qwarp":
        return recipe.optimize
    if recipe.optimize in ("lpa", "lpc"):
        return recipe.optimize
    return "cc"
