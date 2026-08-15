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

import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

from .allcost import build_cost_inputs, evaluate_all_costs
from .allineate import _voxdims_from_header
from .io import load_image
from .mask import automask
from .tunespec import BACKENDS, Recipe, fixed_for, render_command, resolve_tunable
from .tunestore import TrialStore
from .warpqc import FAIL, pad_mask_to_field, regularity_verdict, warp_regularity
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

    cfg = QwarpConfig(verb=0)
    if recipe is not None:
        cfg.cost_method = "pearclp" if recipe.optimize == "ls" else recipe.optimize
    _apply_config(cfg, "qwarp", config)
    warped, xd, yd, zd = qwarp(base, source, config=cfg, device=device)
    return warped, (xd, yd, zd)


def _run_formwarp(base, source, config, recipe, device):
    from .formwarp import SynConfig, formwarp

    cfg = SynConfig(verb=0)
    if recipe is not None:
        cfg.metric = _syn_metric(recipe.optimize)
    _apply_config(cfg, "formwarp", config)
    res = formwarp(base, source, config=cfg, device=device)
    return res.warped, res.fwd


def _run_optiwarp(force: str):
    def run(base, source, config, recipe, device):
        from .optiwarp import OptiwarpConfig, optiwarp

        cfg = OptiwarpConfig(verb=0, force=force)
        if recipe is not None:
            cfg.metric = _syn_metric(recipe.optimize)
        _apply_config(cfg, f"optiwarp_{force}", config)
        res = optiwarp(base, source, config=cfg, device=device)
        return res.warped, res.fwd

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

    def score(self, warped: torch.Tensor, field: tuple | None) -> dict[str, Any]:
        """Score an in-memory result: similarity, then deformation regularity."""
        inp = build_cost_inputs(self.base, warped, self.weight, self.voxdims, 1.0, "tohd")
        scores = evaluate_all_costs(inp)
        del inp

        grade, reasons, qc = "pass", [], {}
        if field is not None:
            xd, yd, zd = field
            mask = pad_mask_to_field(self.brain, tuple(xd.shape))
            w = warp_regularity(xd, yd, zd, mask=mask, voxdims=self.voxdims)
            grade, reasons = regularity_verdict(w)
            qc = w.as_dict()
        return {"scores": scores, "grade": grade, "reasons": reasons, "warpqc": qc}


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
    prefix = f"{backend}_c{store.config_id(backend, config):04d}.nii.gz"
    cmd = render_command(backend, pair.base, pair.source, prefix, config, recipe)

    t0 = time.time()
    try:
        warped, field = DRIVERS[backend](referee.base, source, config, recipe, referee.device)
        outcome = referee.score(warped, field)
        del warped, field
    except (RuntimeError, ValueError) as exc:
        # A backend that blows up on a setting is a fact about that setting, not
        # a reason to abandon the search — record it and move on.
        outcome = {"grade": FAIL, "reasons": [f"{type(exc).__name__}: {exc}"[:300]]}
    seconds = time.time() - t0

    store.add(backend, pair.name, config, cmd, seconds=seconds, **outcome)
    if referee.device.type == "cuda":
        torch.cuda.empty_cache()


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
            store.compute_consensus(exclude=recipe.evaluate_exclude)
            store.save()

    if bar is not None:
        bar.close()


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
