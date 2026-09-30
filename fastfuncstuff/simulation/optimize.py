"""Search for the best realization of a design: evolution over run plans.

A recipe (units, counts, gap distributions) has many realizations -- orders
and gap draws -- and they differ: the worst shuffle of a block design needed
17% more signal than the best. Best-of-N random draws is the baseline; this
is an elitist evolutionary search (Wager & Nichols 2003; Kao et al. 2009 do
the same with a multi-objective fitness) over :class:`~.experiment.RunPlan`s:

- **mutations** keep the recipe exact: swap two units (each keeps its own
  gaps, so the run's gap distribution and total content are unchanged), or
  swap the trailing gaps of two units drawn from the same distribution;
- **crossover** takes each run whole from one parent or the other;
- **constraints**: at most ``max_repeat`` consecutive units of one kind.

The fitness is any function of a realization, lower better -- the analytic
scores of :class:`~.power.RealizationScorer`, optionally averaged over
several HRFs so the search cannot overfit one assumed response (an optimized
design that concentrates its power in one frequency band is the known
failure of design optimization). Every run also records best-of-N over as
many random draws, so the search has to beat that to be worth it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from .experiment import ExperimentSpec, Realization, RunPlan, assemble, draw_plans


def _trailing_key(spec: ExperimentSpec, ui: int) -> Any:
    """What a unit's trailing gap is drawn from: its last item's OFF, or -isi."""
    items = spec.units[ui].items
    last = items[-1]
    return ("item", ui, len(items) - 1) if last.off is not None else "isi"


def longest_repeat(plan: RunPlan) -> int:
    """Longest run of consecutive entries of the same unit."""
    best = cur = 1
    for a, b in zip(plan.entries, plan.entries[1:], strict=False):
        cur = cur + 1 if a[0] == b[0] else 1
        best = max(best, cur)
    return best if plan.entries else 0


def mutate(spec: ExperimentSpec, plans: list[RunPlan], rng: np.random.Generator) -> None:
    """One recipe-preserving change, in place: swap two units, or two same-kind trailing gaps."""
    run = plans[int(rng.integers(len(plans)))]
    n = len(run.entries)
    if n < 2:
        return
    if rng.random() < 0.6:
        a, b = rng.choice(n, 2, replace=False)
        run.entries[a], run.entries[b] = run.entries[b], run.entries[a]
        return
    keys = [_trailing_key(spec, ui) for ui, _ in run.entries]
    a = int(rng.integers(n))
    same = [k for k in range(n) if k != a and keys[k] == keys[a]]
    if not same:
        return
    b = int(rng.choice(same))
    (ua, ga), (ub, gb) = run.entries[a], run.entries[b]
    ga, gb = list(ga), list(gb)
    ga[-1], gb[-1] = gb[-1], ga[-1]
    run.entries[a], run.entries[b] = (ua, ga), (ub, gb)


def evolve(
    spec: ExperimentSpec,
    fitness: Callable[[Realization], float],
    population: int = 40,
    generations: int = 50,
    seed: int = 0,
    max_repeat: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evolve realizations of ``spec`` toward low ``fitness``; compare against best-of-N.

    Returns
        'best'          : the best Realization found
        'best_fitness'  : its fitness
        'history'       : best fitness after each generation
        'evaluations'   : fitness calls made by the search, per generation (cumulative)
        'random_best'   : best-of-N over as many fresh random draws, at the same
                          cumulative evaluation counts -- the baseline to beat
        'random_median' : median fitness of a random draw (what -ndesigns samples)
        'random_best_realization' : the best-of-N draw itself, for held-out checks
        'n_rejected'    : children refused by the max_repeat constraint
    """
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    counts, _ = draw_plans(spec, seed)

    def ok(plans: list[RunPlan]) -> bool:
        return max_repeat is None or all(longest_repeat(p) <= max_repeat for p in plans)

    def score(plans: list[RunPlan], s: int) -> float:
        if not ok(plans):
            return np.inf
        v = fitness(assemble(spec, counts, plans, s))
        return float(v) if np.isfinite(v) else np.inf

    pop = []
    draws = 0
    while len(pop) < population and draws < population * 20:
        _, plans = draw_plans(spec, seed + draws)
        draws += 1
        if ok(plans):
            pop.append((score(plans, seed), plans))
    if not pop:
        raise ValueError(f"no realization met max_repeat {max_repeat} in {draws} draws")
    # Random draws seen so far are the start of the best-of-N baseline too.
    random_scores = [f for f, _ in pop]
    random_top = min(pop, key=lambda x: x[0])
    n_evals = len(pop)
    history, evals, random_best = [], [], []
    rejected = 0
    for _ in tqdm(range(generations), desc="generations", leave=True, disable=not progress):
        children = []
        for _ in range(population):
            # Tournament of two, with replacement: -max_repeat can leave a tiny
            # population, and drawing two distinct members then failed.
            a, b = rng.integers(len(pop), size=2)
            parent = pop[a] if pop[a][0] <= pop[b][0] else pop[b]
            child = copy.deepcopy(parent[1])
            other = pop[int(rng.integers(len(pop)))][1]
            for r in range(len(child)):
                if rng.random() < 0.5:
                    child[r] = copy.deepcopy(other[r])
            for _ in range(int(rng.integers(1, 4))):
                mutate(spec, child, rng)
            if not ok(child):
                rejected += 1
                continue
            children.append((score(child, seed), child))
        n_evals += len(children)
        pop = sorted(pop + children, key=lambda x: x[0])[:population]
        # The baseline gets the same budget: that many more fresh random draws.
        while len(random_scores) < n_evals:
            _, plans = draw_plans(spec, seed + 10_000_000 + len(random_scores))
            random_scores.append(score(plans, seed))
            if random_scores[-1] < random_top[0]:
                random_top = (random_scores[-1], plans)
        history.append(pop[0][0])
        evals.append(n_evals)
        random_best.append(float(np.min(random_scores[:n_evals])))
    finite = [v for v in random_scores if np.isfinite(v)]
    return {
        "best": assemble(spec, counts, pop[0][1], seed),
        "best_fitness": pop[0][0],
        "history": history,
        "evaluations": evals,
        "random_best": random_best,
        "random_median": float(np.median(finite)) if finite else np.inf,
        "n_rejected": rejected,
        "random_best_realization": assemble(spec, counts, random_top[1], seed),
    }


ROBUST_HRFS = ("lib:0", "lib:6", "lib:13", "lib:19")  # fast to slow, beside the fitted one
HELD_OUT_HRFS = ("lib:2", "lib:9", "lib:16")  # never trained on: the generalization check


def make_fitness(
    tr: float,
    contrasts: dict[str, Any],
    noise: list[dict[str, Any]],
    pattern: Any,
    objective: str,
    ref_noise: str,
    hrfs: list[str],
    alpha: float = 0.001,
    poly_degree: int | None = None,
    mean_response: float = 1.0,
    trial_sd: float = 0.5,
    target: float = 0.8,
) -> Callable[[Realization], float]:
    """Mean over ``hrfs`` of the objective: a contrast's detection, 'shape' or 'trials'.

    Averaging over HRF shapes is what keeps the search honest: optimized for
    SPMG1 alone, the best design of a rapid event experiment was 9% better than
    best-of-N under SPMG1 and *worse than a median random draw* under a fast
    library HRF. Averaged over five shapes it won on every held-out one.
    """
    from .power import RealizationScorer

    scorers = [
        RealizationScorer(
            tr,
            contrasts,
            noise,
            pattern,
            h,
            alpha,
            target,
            poly_degree=poly_degree,
            mean_response=mean_response,
            trial_sd=trial_sd,
        )
        for h in hrfs
    ]

    def one(sc: Any, real: Realization) -> float:
        s = sc.score(real, shape=objective == "shape", single=objective == "trials")
        if s is None:
            return np.inf
        if objective == "shape":
            return s["shape_sd"][ref_noise]
        if objective == "trials":
            return s["unreliability"][ref_noise]
        return s["needed"][(ref_noise, objective)]

    return lambda real: float(np.mean([one(sc, real) for sc in scorers]))
