"""Run and score nonlinear-registration trials — the engine of `ffs_tunewarp`.

The search drives the backend **libraries in-process**, not their CLIs. Each
image is loaded once and stays resident on the GPU; a trial is a function call
that returns tensors, which are scored in memory and dropped. Nothing a trial
produces reaches the disk.

That is both the space answer and most of the speed answer. A warp field on a
193^3 grid is ~90 MB, so a few hundred fits would be hundreds of gigabytes; and
shelling out would pay process startup, NIfTI compression, and a reload of every
volume, per trial, for data whose entire useful content is one row of a table.

What is recorded instead is the exact equivalent command line. Nothing is kept,
but anything can be rebuilt: ``reproduce()`` re-runs a config through the real
CLI with its outputs kept, so the winner can be looked at.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .allineate import _voxdims_from_header
from .io import load_image
from .mask import automask
from .metrics import MetricInputs, evaluate_metrics
from .tuneopt import Observation, SearchSpace, config_key, propose
from .tunespec import (
    BACKENDS,
    QWARP_TUNE_OPTIMIZER,
    Recipe,
    fixed_for,
    render_command,
    resolve_tunable,
)
from .tunestore import BASELINE, TrialStore
from .warpqc import (
    FAIL,
    FAILED_MARGIN,
    UNCONSTRAINED_MARGIN,
    gate_margin,
    pad_mask_to_field,
    regularity_cautions,
    regularity_margin,
    regularity_verdict,
    warp_regularity,
)
from .weight import compute_weight_image


@dataclass
class SubjectPair:
    """One base/source pair to fit. ``name`` is what shows up in the report."""

    name: str
    base: str
    source: str


# --- in-process backend drivers ---------------------------------------------
#
# Each returns (warped_image, (xd, yd, zd)) as tensors, all on the GPU. The
# config dict is keyed by ParamSpec.key; ``config_attr`` maps that onto the
# backend's own dataclass field where the two spellings differ.


def _apply_config(cfg_obj: Any, backend: str, config: dict[str, Any]) -> None:
    spec = BACKENDS[backend]
    for key, value in config.items():
        setattr(cfg_obj, spec.param(key).config_attr, value)


def _run_qwarp(base, source, config, recipe, device):
    from .warp import QwarpConfig, qwarp

    cfg = QwarpConfig(verb=0, optimizer=QWARP_TUNE_OPTIMIZER)
    if recipe is not None:
        cfg.cost_method = "pearclp" if recipe.optimize == "ls" else recipe.optimize
    _apply_config(cfg, "qwarp", config)
    # The pyramid levels are a trajectory: a run at minpatch 5 passes through every
    # coarser patch size on the way. Recording what each level cost and bought means
    # one run answers "was going finer worth it?" for all of them.
    levels: list[dict] = []
    warped, xd, yd, zd = qwarp(base, source, config=cfg, device=device, level_log=levels)
    return warped, (xd, yd, zd), [_QwarpLevel(lv) for lv in levels]


class _QwarpLevel:
    """Adapts a qwarp level record to the ``.as_dict()`` the trial store expects."""

    def __init__(self, rec: dict):
        self._rec = rec

    def as_dict(self) -> dict:
        return dict(self._rec)


def _run_formwarp(base, source, config, recipe, device):
    from .formwarp import SynConfig, formwarp

    cfg = SynConfig(verb=0)
    if recipe is not None:
        cfg.metric = _syn_metric(recipe.optimize)
    _apply_config(cfg, "formwarp", config)
    res = formwarp(base, source, config=cfg, device=device)
    return res.warped, res.fwd, res.levels


def _run_optiwarp(force: str):
    def run(base, source, config, recipe, device):
        from .optiwarp import OptiwarpConfig, optiwarp

        cfg = OptiwarpConfig(verb=0, force=force)
        if recipe is not None:
            cfg.metric = _syn_metric(recipe.optimize)
        _apply_config(cfg, f"optiwarp_{force}", config)
        res = optiwarp(base, source, config=cfg, device=device)
        return res.warped, res.fwd, res.levels

    return run


def _syn_metric(optimize: str) -> str:
    """The SyN/flow tools take a shorter metric list than the allineate costs."""
    return optimize if optimize in ("lpa", "lpc", "pearson", "mse", "cc") else "cc"


DRIVERS = {
    "qwarp": _run_qwarp,
    "formwarp": _run_formwarp,
    "optiwarp_demons": _run_optiwarp("demons"),
    "optiwarp_lk": _run_optiwarp("lk"),
    "optiwarp_hs": _run_optiwarp("hs"),
}


class Referee:
    """Scores a warped image + its field against one base. Caches the base setup.

    The base-side work — weight image, brain mask, voxel dims — is identical for
    every trial against that base, and is the expensive part of scoring, so it is
    built once per base rather than once per trial.
    """

    def __init__(self, base_path: str, device: torch.device):
        self.device = device
        self.base, self.header = load_image(base_path, device=device)
        if self.base.ndim == 4:
            self.base = self.base[..., 0]
        self.voxdims = _voxdims_from_header(self.header)
        self.weight = compute_weight_image(
            self.base,
            edge_fraction=0.05,
            median_radius=2.25,
            clusterize=True,
            hist_cliplevel=True,
        )
        self.brain = automask(self.base, device=device)

    def score(
        self, warped: torch.Tensor, field: tuple | None, panel: list[str] | None = None
    ) -> dict[str, Any]:
        """Score an in-memory result: similarity, then deformation regularity.

        Goes through the metric registry rather than the AFNI cost list alone, so
        the neighbourhood metrics are scored too. The referee holds the volumes,
        which is exactly what those need and what a flattened cost input cannot
        provide.
        """
        inp = MetricInputs(
            base=self.base, moving=warped, weight=self.weight, voxdims=self.voxdims, overlap=1.0
        )
        scores = evaluate_metrics(inp, panel)
        del inp

        grade, reasons, cautions, qc = "pass", [], [], {}
        # A field-less result (an affine-only backend) has nothing to fold, so it
        # gets the best margin rather than a missing one — absent evidence of a
        # boundary is not evidence of being on the wrong side of it.
        margin = clearance = UNCONSTRAINED_MARGIN
        if field is not None:
            xd, yd, zd = field
            mask = pad_mask_to_field(self.brain, tuple(xd.shape))
            w = warp_regularity(xd, yd, zd, mask=mask, voxdims=self.voxdims)
            grade, reasons = regularity_verdict(w)
            cautions = regularity_cautions(w)
            margin = regularity_margin(w)
            clearance = gate_margin(w)
            qc = w.as_dict()
        return {
            "scores": scores,
            "grade": grade,
            "reasons": reasons,
            "cautions": cautions,
            "warpqc": qc,
            "margin": margin,
            "gate_margin": clearance,
        }


def affine_align(
    pairs: list[SubjectPair],
    recipe: Recipe,
    out_dir: Path,
    device: torch.device | None = None,
    verb: int = 1,
) -> list[SubjectPair]:
    """Step 0: affine-align every source to its base, and cache the result.

    The nonlinear search assumes the pair already agrees in the affine sense, so
    this has to happen once per subject — but it is not part of the search, and
    nobody wants to hand-run it ten times.

    Unlike trial outputs, these *are* written to disk, for two reasons: they are
    deterministic inputs rather than per-trial noise (so a re-run should skip
    them, not redo ~25 s per subject), and `-reproduce` shells out to the real
    CLI, which needs a path to the aligned volume — a command line pointing at
    the unaligned source would reproduce the wrong thing.
    """
    from .allineate import AffineAlignConfig
    from .allineate import allineate as run_allineate
    from .io import save_image

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = out_dir / "affine"
    cache.mkdir(parents=True, exist_ok=True)

    out: list[SubjectPair] = []
    for pair in pairs:
        dst = cache / f"{pair.name.replace('/', '_')}.nii.gz"
        if dst.exists():
            if verb >= 1:
                print(f"  {pair.name}: affine cached", flush=True)
            out.append(SubjectPair(pair.name, pair.base, str(dst)))
            continue

        base, base_header = load_image(pair.base, device=device)
        source, source_header = load_image(pair.source, device=device)
        if base.ndim == 4:
            base = base[..., 0]
        if source.ndim == 4:
            source = source[..., 0]
        cfg = AffineAlignConfig(cost=recipe.optimize, device=str(device), verb=0)
        t0 = time.time()
        _, warped = run_allineate(base, source, cfg, base_header, source_header)
        save_image(warped, str(dst), header_info=base_header)
        if verb >= 1:
            print(f"  {pair.name}: affine {time.time() - t0:.1f}s -> {dst.name}", flush=True)
        out.append(SubjectPair(pair.name, pair.base, str(dst)))
        del base, source, warped
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def enumerate_configs(
    recipe: Recipe,
    backend: str,
    max_configs: int | None = None,
    fixed: dict[str, Any] | None = None,
) -> list[dict]:
    """The full factorial over this recipe's tunable knobs for this backend.

    Full, not sampled: the whole point of the per-backend search is that no
    setting is eliminated before it has been tried. Pruning happens *within* a
    backend after the fact, never across backends beforehand.

    ``fixed`` (from ``-fix``) is merged into every config, so a pinned knob still
    shows up in the recorded settings and the reproducible command line.
    """
    pins = fixed_for(fixed or {}, backend)
    params = [p for p in resolve_tunable(recipe, backend) if p.key not in pins]
    if not params:
        return [dict(pins)]
    grids = [[(p.key, v) for v in p.values] for p in params]
    configs = [{**pins, **dict(combo)} for combo in product(*grids)]
    return configs[:max_configs] if max_configs else configs


def run_trial(
    backend: str,
    pair: SubjectPair,
    config: dict[str, Any],
    recipe: Recipe,
    referee: Referee,
    store: TrialStore,
    source: torch.Tensor,
) -> None:
    """Fit once in memory, score it, record the numbers, drop the tensors.

    The equivalent command line is recorded even though it was never executed —
    it is what ``reproduce()`` runs, and what a user pastes to get this result
    outside the tool.
    """
    # Only the recipe's panel is scored, not every metric in the registry. The
    # neighbourhood metrics are far more expensive than the AFNI functionals, and
    # scoring a metric that is barred from voting buys nothing.
    prefix = f"{backend}_c{store.config_id(backend, config):04d}.nii.gz"
    cmd = render_command(backend, pair.base, pair.source, prefix, config, recipe)

    t0 = time.time()
    try:
        warped, field, levels = DRIVERS[backend](
            referee.base, source, config, recipe, referee.device
        )
        outcome = referee.score(warped, field, recipe.panel())
        outcome["levels"] = [lv.as_dict() for lv in levels]
        del warped, field
    except (RuntimeError, ValueError) as exc:
        # A backend that blows up on a setting is a fact about that setting, not
        # a reason to abandon the search — record it and move on.
        outcome = {
            "grade": FAIL,
            "reasons": [f"{type(exc).__name__}: {exc}"[:300]],
            "margin": FAILED_MARGIN,
        }
    seconds = time.time() - t0

    store.add(backend, pair.name, config, cmd, seconds=seconds, **outcome)
    if referee.device.type == "cuda":
        torch.cuda.empty_cache()


def score_baseline(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    referees: dict[str, Referee],
    sources: dict[str, torch.Tensor],
) -> None:
    """Score every subject's *input*, unwarped, as the do-nothing row.

    Without it a run reports which candidate won but not whether any of them beat
    leaving the data alone -- and "nonlinear bought this much on data like this" is
    the statement a recommendation is actually made of. Cheap: one scoring pass per
    subject, no fit.
    """
    panel = recipe.panel()
    for pair in pairs:
        referee = referees[pair.base]
        source = sources[pair.source]
        outcome = referee.score(source, None, panel)
        store.add(
            BASELINE,
            pair.name,
            {},
            [],
            seconds=0.0,
            **outcome,
        )


def describe_pairs(pairs: list[SubjectPair], referees: dict[str, Referee]) -> dict[str, Any]:
    """The data's own properties, for the run record.

    A preset claims some settings suit data *of a kind*; resolution, matrix and how
    much of the volume is brain are what let the next person decide whether their
    data is that kind.
    """
    ref = referees[pairs[0].base]
    return {
        "subjects": [p.name for p in pairs],
        "base": pairs[0].base,
        "shape": tuple(int(v) for v in ref.base.shape),
        "voxdims": tuple(float(v) for v in ref.voxdims),
        "n_mask_voxels": int(ref.brain.sum()),
    }


def run_search(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    backends: list[str] | None = None,
    max_configs: int | None = None,
    fixed: dict[str, Any] | None = None,
    device: torch.device | None = None,
    verb: int = 1,
    progress=None,
) -> None:
    """Search every backend over its own full grid, on every subject."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = backends or list(recipe.backends)
    for backend in names:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; have {', '.join(BACKENDS)}")

    # Every image is loaded exactly once for the whole search, and the base-side
    # referee setup (weight image, brain mask) once per distinct base. An
    # MNI-style run shares a single referee across every subject and trial.
    referees: dict[str, Referee] = {}
    sources: dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.base not in referees:
            referees[pair.base] = Referee(pair.base, device)
        if pair.source not in sources:
            vol, _ = load_image(pair.source, device=device)
            sources[pair.source] = vol[..., 0] if vol.ndim == 4 else vol

    if store.runs:
        # The data's own properties are only knowable once the images are open, so
        # the run record is completed here rather than at begin_run().
        for k, v in describe_pairs(pairs, referees).items():
            setattr(store.runs[-1], k, v)
    if not any(t.backend == BASELINE for t in store.trials):
        score_baseline(pairs, recipe, store, referees, sources)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    plan = [(b, enumerate_configs(recipe, b, max_configs, fixed)) for b in names]
    total = sum(len(cfgs) for _, cfgs in plan) * len(pairs)
    # A full search is hundreds of fits over tens of minutes; a bar that shows
    # which backend is up and how many fits remain is the difference between
    # "it is working" and "is it stuck".
    bar = (
        tqdm(total=total, desc="tuning", unit="fit", leave=True, file=sys.stderr)
        if tqdm is not None and verb >= 1 and total > 1
        else None
    )

    for backend, configs in plan:
        if bar is not None:
            bar.set_description(backend)
        elif verb >= 1:
            print(f"\n{backend}: {len(configs)} configs x {len(pairs)} subjects", flush=True)
        for config in configs:
            for pair in pairs:
                run_trial(
                    backend,
                    pair,
                    config,
                    recipe,
                    referees[pair.base],
                    store,
                    sources[pair.source],
                )
                if bar is not None:
                    last = store.trials[-1]
                    bar.set_postfix_str(f"{last.grade}", refresh=False)
                    bar.update(1)
                if progress is not None:
                    progress()
            # Save after every config, not every backend. A slow backend (qwarp
            # is minutes per fit) would otherwise leave the table empty for the
            # whole run, so a search you interrupt tells you nothing.
            store.compute_consensus(recipe.panel())
            store.save()

    if bar is not None:
        bar.close()


@dataclass
class AdaptivePlan:
    """How the adaptive search spends its fits.

    ``budget`` is per backend and counted in *fits*, which is the currency that
    matters — a fit is tens of seconds and everything else here is microseconds.
    """

    budget: int = 60
    screen: int = 1  # subjects a fresh candidate is tried on
    confirm: int = 2  # further subjects a survivor earns
    batch: int = 4  # candidates proposed per surrogate refit
    expand: bool = True  # grow a ladder when the incumbent sits on its end
    seed: int = 0


def panel_scores(trials: list, panel: list[str]) -> dict[int, float]:
    """Per-trial score for the surrogate: the panel mean, z-scored within subject.

    Two things force this rather than the report's consensus rank. The rank is
    *relative to the field*, so it changes underneath the surrogate every time a
    trial is added — an unusable regression target. And the adaptive search
    deliberately measures different configs on different numbers of subjects, so
    the target has to be comparable across subjects; z-scoring within a subject
    removes exactly the between-brain offset that would otherwise dominate, and
    makes a config screened on one brain comparable to one confirmed on three.

    Subjects with fewer than two scored trials are skipped: a z-score needs a
    spread, and inventing one would feed the surrogate a confident zero.
    """
    by_subject: dict[str, list] = {}
    for t in trials:
        if t.scores:
            by_subject.setdefault(t.subject, []).append(t)

    out: dict[int, float] = {}
    for ts in by_subject.values():
        if len(ts) < 2:
            continue
        usable = [c for c in panel if all(c in t.scores for t in ts)]
        totals: dict[int, list[float]] = {t.trial_id: [] for t in ts}
        for cost in usable:
            vals = [t.scores[cost] for t in ts]
            mean = statistics.fmean(vals)
            sd = statistics.pstdev(vals)
            if sd <= 0:
                continue
            for t, v in zip(ts, vals, strict=True):
                totals[t.trial_id].append((v - mean) / sd)
        for tid, zs in totals.items():
            if zs:
                out[tid] = statistics.fmean(zs)
    return out


def _observations(store: TrialStore, backend: str, panel: list[str]) -> list[Observation]:
    """The store's trials for one backend, in the form the optimiser consumes."""
    trials = [t for t in store.trials if t.backend == backend]
    scores = panel_scores(trials, panel)
    return [
        Observation(config=t.config, score=scores[t.trial_id], margin=t.margin)
        for t in trials
        if t.trial_id in scores
    ]


def _incumbent(observations: list[Observation]) -> dict | None:
    """Best config so far: feasible ones first, then by mean score."""
    if not observations:
        return None
    agg: dict[tuple, list[Observation]] = {}
    for o in observations:
        agg.setdefault(config_key(o.config), []).append(o)
    ranked = sorted(
        agg.values(),
        key=lambda os: (
            0 if min(o.margin for o in os) > 0 else 1,
            statistics.fmean([o.score for o in os]),
        ),
    )
    return ranked[0][0].config


def run_adaptive(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    plan: AdaptivePlan,
    backends: list[str] | None = None,
    fixed: dict[str, Any] | None = None,
    device: torch.device | None = None,
    verb: int = 1,
) -> None:
    """Search by surrogate instead of by grid: propose, screen, confirm, expand.

    The loop, per backend:

    1. Fit a GP to the scores seen so far and a second GP to the regularity
       margins, and propose a batch by expected improvement times probability of
       feasibility.
    2. **Screen** each candidate on one subject. This is where the savings are:
       most candidates are answered by one fit, and a folded setting is answered
       by its first.
    3. **Confirm** only the survivors on further subjects, so the fits that cost
       the most go to the settings that might actually become the default.
    4. **Expand** any ladder the incumbent is sitting on the end of, because an
       optimum at a range edge is a statement about the range, not the optimum.

    The screening subject rotates between batches. Within a batch it is held
    fixed so the candidates are compared against each other on the same brain,
    but a screen that never rotated would tune to one head.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = backends or list(recipe.backends)
    for backend in names:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; have {', '.join(BACKENDS)}")

    panel = recipe.panel()
    referees: dict[str, Referee] = {}
    sources: dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.base not in referees:
            referees[pair.base] = Referee(pair.base, device)
        if pair.source not in sources:
            vol, _ = load_image(pair.source, device=device)
            sources[pair.source] = vol[..., 0] if vol.ndim == 4 else vol

    if store.runs:
        # The data's own properties are only knowable once the images are open, so
        # the run record is completed here rather than at begin_run().
        for k, v in describe_pairs(pairs, referees).items():
            setattr(store.runs[-1], k, v)
    if not any(t.backend == BASELINE for t in store.trials):
        score_baseline(pairs, recipe, store, referees, sources)

    rng = np.random.default_rng(plan.seed)
    for backend in names:
        space = SearchSpace.from_params(
            resolve_tunable(recipe, backend), fixed_for(fixed or {}, backend)
        )
        if verb >= 1:
            print(f"\n{backend}: budget {plan.budget} fits over {len(space.axes)} knob(s)")

        spent = 0
        round_no = 0
        while spent < plan.budget:
            obs = _observations(store, backend, panel)
            batch = propose(space, obs, plan.batch, rng)
            if not batch:
                # The lattice is exhausted. Growing it is the only way forward,
                # and if nothing can grow the space is genuinely finished.
                grown = space.grow_toward(_incumbent(obs) or {}) if plan.expand else []
                if not grown:
                    if verb >= 1:
                        print("  space exhausted")
                    break
                if verb >= 1:
                    print(f"  expanded: {', '.join(grown)}")
                continue

            # One screening brain per round, rotated, so within-round comparisons
            # are like-for-like without the search fitting itself to one head.
            screen_pairs = [pairs[(round_no + i) % len(pairs)] for i in range(plan.screen)]
            round_no += 1

            for config in batch:
                if spent >= plan.budget:
                    break
                for pair in screen_pairs:
                    run_trial(
                        backend,
                        pair,
                        config,
                        recipe,
                        referees[pair.base],
                        store,
                        sources[pair.source],
                    )
                    spent += 1
                last = store.trials[-1]
                if verb >= 1:
                    label = " ".join(f"{k}={v}" for k, v in sorted(config.items()))
                    print(
                        f"  [{spent:>3}/{plan.budget}] screen {last.grade:8s} {label}", flush=True
                    )

                if not _promising(store, backend, panel, config):
                    continue
                rest = [p for p in pairs if p not in screen_pairs]
                rng.shuffle(rest)  # type: ignore[arg-type]
                for pair in rest[: plan.confirm]:
                    if spent >= plan.budget:
                        break
                    run_trial(
                        backend,
                        pair,
                        config,
                        recipe,
                        referees[pair.base],
                        store,
                        sources[pair.source],
                    )
                    spent += 1
                if verb >= 1:
                    print(
                        f"  [{spent:>3}/{plan.budget}] confirmed on {min(plan.confirm, len(rest))}"
                    )

            if plan.expand:
                incumbent = _incumbent(_observations(store, backend, panel)) or {}
                grown = space.grow_toward(incumbent)
                if grown and verb >= 1:
                    print(f"  expanded: {', '.join(grown)}")
                # Extend where the incumbent is pinned against an end, subdivide
                # where it sits between two rungs. Without the second the search can
                # reach a listed value but nothing between two of them, and the gaps
                # here are large: measured 130-290x the run-to-run noise.
                refined = space.refine_around(incumbent)
                if refined and verb >= 1:
                    print(f"  refined: {', '.join(refined)}")

            store.compute_consensus(panel)
            store.save()

        store.compute_consensus(panel)
        store.save()


def _promising(store: TrialStore, backend: str, panel: list[str], config: dict) -> bool:
    """Does a screened candidate earn confirmation fits on more subjects?

    A folded screen is an immediate no — confirming it would only establish more
    precisely how broken it is, and the surrogate already learned what it needed
    from the one fit. Otherwise the bar is the lower half of the feasible
    candidates seen so far, with the first few waved through so the comparison
    has something to be a comparison against.
    """
    key = config_key(config)
    obs = _observations(store, backend, panel)
    mine = [o for o in obs if config_key(o.config) == key]
    if not mine or min(o.margin for o in mine) <= 0:
        return False
    feasible = sorted(o.score for o in obs if o.margin > 0)
    if len(feasible) < 4:
        return True
    return statistics.fmean([o.score for o in mine]) <= feasible[len(feasible) // 2]


def reproduce(
    store: TrialStore,
    config_id: int,
    work_dir: Path,
    timeout: float | None = None,
    verb: int = 1,
) -> list[Path]:
    """Re-run every fit of one config, keeping the outputs so they can be looked at."""
    out_root = work_dir / "kept" / f"config{config_id:04d}"
    out_root.mkdir(parents=True, exist_ok=True)
    written = []
    for subject, cmd in store.commands_for(config_id):
        cmd = list(cmd)
        # Redirect -prefix into the keep directory, leaving the rest verbatim.
        pi = cmd.index("-prefix")
        target = out_root / f"{subject}_{Path(cmd[pi + 1]).name}"
        cmd[pi + 1] = str(target)
        if verb >= 1:
            print(f"  {subject}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, timeout=timeout, check=False)
        written.append(target)
    return written
