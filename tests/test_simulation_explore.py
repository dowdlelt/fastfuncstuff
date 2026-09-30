"""Design-space exploration: placeholders, sampling, the Pareto front, the CLI."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.simulation.explore import find_axes, pareto_front, render, sample, shortlist


def test_placeholders_become_axes_and_render_back():
    argv = ["-tr", "1", "-isi", "exp:[3.0-8.0],[1-3],12", "-null", "0.25", "[0-40%]",
            "-order", "{random,permuted_block}"]  # fmt: skip
    axes = find_axes(argv)
    assert [a.label for a in axes] == ["isi.1", "isi.2", "null.1", "order.1"]
    isi_mean, isi_min, null, order = axes
    assert not isi_mean.integer and isi_min.integer and null.percent and order.is_choice
    configs = sample(axes, 40, seed=1)
    for c in configs:
        out = render(argv, axes, c)
        assert "[" not in " ".join(out) and "{" not in " ".join(out)
        assert out[3].startswith("exp:") and out[3].endswith(",12")
        assert 3.0 <= float(c["isi.1"]) <= 8.0 and c["isi.2"] in {"1", "2", "3"}
        assert c["null.1"].endswith("%") and c["order.1"] in {"random", "permuted_block"}
    # Latin hypercube: every tenth of a range is drawn once in 10 samples
    lh = sample(axes[:1], 10, seed=2)
    assert sorted(int((float(c["isi.1"]) - 3) / 0.5) for c in lh) == list(range(10))


def test_bad_placeholders_are_refused():
    with pytest.raises(ValueError, match="low < high"):
        find_axes(["-isi", "[5-3]"])
    with pytest.raises(ValueError, match="two or more"):
        find_axes(["-order", "{random}"])


def test_pareto_front_and_shortlist():
    x = np.array([1.0, 2.0, 3.0, 1.5, 2.5, np.nan])
    y = np.array([3.0, 2.0, 1.0, 3.5, 2.5, 0.0])
    front = pareto_front(x, y)
    assert front.tolist() == [True, True, True, False, False, False]
    assert shortlist(x, front, 2) == [0, 2]  # the two ends, best detection first


def test_explore_writes_a_small_set_of_outputs(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main

    base = ["-tr", "1", "-nruns", "1", "-initial_fix", "10", "-post_fix", "15",
            "-trial", "A", "0.25", "1", "-null", "0.25", "[0-30%]",
            "-isi", "exp:[3.0-6.0],1,10", "-tsnr", "50", "-device", "cpu"]  # fmt: skip
    argv = [*base, "-scan_time", "200", "-explore", "12", "-explore_keep", "2",
            "-explore_pick", "3", "-prefix", str(tmp_path / "x")]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Pareto front" in out and "recipe:" in out and "what matters" in out
    names = {p.name for p in tmp_path.iterdir()}
    assert names <= {"x_explore.tsv", "x_explore.png", "x_explore_summary.txt", "x_explore_best"}
    assert len((tmp_path / "x_explore.tsv").read_text().splitlines()) == 13
    assert any((tmp_path / "x_explore_best").rglob("A.txt"))
    # the recipe line is a runnable command with every placeholder filled in
    recipe = next(ln for ln in out.splitlines() if "recipe:" in ln)
    assert "[" not in recipe and "-explore" not in recipe

    # one budget: time or trials, never both
    both = [*argv, "-num_events", "40"]
    assert main(both) == 1
    assert "one budget" in capsys.readouterr().err


def test_trials_objective_runs_in_explore_and_optimize(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main

    base = ["-tr", "1", "-nruns", "2", "-scan_time", "120", "-initial_fix", "10",
            "-post_fix", "15", "-trial", "A", "0.25", "1", "-tsnr", "50",
            "-objective", "trials", "-device", "cpu"]  # fmt: skip
    explore = [*base, "-isi", "exp:[3.0-9.0],1,14", "-explore", "6", "-explore_keep", "1",
               "-explore_pick", "2", "-prefix", str(tmp_path / "e")]  # fmt: skip
    assert main(explore) == 0
    assert "1 - reliability" in capsys.readouterr().out
    optimize = [*base, "-isi", "exp:4,1,14", "-optimize", "1", "-optimize_pop", "4",
                "-optimize_hrfs", "spmg1", "-prefix", str(tmp_path / "o")]  # fmt: skip
    assert main(optimize) == 0
    assert "reliability" in capsys.readouterr().out
