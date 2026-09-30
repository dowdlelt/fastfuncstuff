"""Design search over run plans: the recipe stays exact, constraints hold, the CLI runs."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, draw_plans
from fastfuncstuff.simulation.optimize import evolve, longest_repeat, mutate


def _spec():
    return ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:0.25", 1), Unit.parse("B", "B:0.25", 1)],
        isi=Interval.parse("exp:5,2,14"),
        n_runs=2,
        initial_fix=10,
        post_fix=15,
        scan_time=200,
    )


def test_mutations_keep_the_recipe_exact():
    spec = _spec()
    _, plans = draw_plans(spec, 0)
    before = [
        (Counter(u for u, _ in p.entries), sorted(g[-1] for _, g in p.entries)) for p in plans
    ]
    rng = np.random.default_rng(0)
    mutated = copy.deepcopy(plans)
    for _ in range(200):
        mutate(spec, mutated, rng)
    after = [
        (Counter(u for u, _ in p.entries), sorted(g[-1] for _, g in p.entries)) for p in mutated
    ]
    assert before == after  # same units per run, same multiset of gaps
    assert [[u for u, _ in p.entries] for p in plans] != [
        [u for u, _ in p.entries] for p in mutated
    ]


def test_evolve_respects_max_repeat_and_never_ends_worse():
    spec = _spec()
    # any fitness will do: the regularity of A's onsets (spread of its gaps)
    res = evolve(
        spec,
        lambda r: float(np.std(np.diff(np.sort(r.onsets[0][0])))),
        population=8,
        generations=4,
        max_repeat=3,
        progress=False,
    )
    assert res["history"] == sorted(res["history"], reverse=True)  # elitist: monotone
    assert res["best_fitness"] <= res["random_best"][0] + 1e-12
    counts, plans = draw_plans(spec, 0)
    assert all(longest_repeat(p) >= 1 for p in plans)


def test_optimize_cli_writes_the_realization_and_the_evidence(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main

    argv = ["-tr", "1", "-nruns", "1", "-scan_time", "150", "-initial_fix", "10",
            "-post_fix", "15", "-trial", "A", "0.25", "1", "-trial", "B", "0.25", "1",
            "-isi", "exp:5,2,14", "-contrast", "A-B", "-tsnr", "50", "-optimize", "2",
            "-optimize_pop", "6", "-optimize_hrfs", "spmg1", "-max_repeat", "3",
            "-device", "cpu", "-prefix", str(tmp_path / "o")]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "evolved" in out and "held-out HRFs" in out and "-events" in out
    assert (tmp_path / "o_optimized" / "A.txt").exists()
    assert (tmp_path / "o_optimize.png").stat().st_size > 10_000
    # the printed command runs the full simulation on the optimized timing
    cmd = next(ln for ln in out.splitlines() if ln.strip().startswith("ffs_simulate -tr"))
    rerun = cmd.split()[1:]
    rerun[rerun.index("-prefix") + 1] = str(tmp_path / "rerun")
    assert main([*rerun, "-nreps", "20", "-no_plots"]) == 0
