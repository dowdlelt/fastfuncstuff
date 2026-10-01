"""Design-space exploration: placeholders, sampling, the Pareto front, the CLI."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.simulation.explore import find_axes, pareto_front, render, sample, shortlist


def test_placeholders_become_axes_and_render_back():
    argv = ["-tr", "1", "-isi", "exp:[3.0-8.0],[1-3],12", "-null", "0.25", "[0-40%]",
            "-order", "{random,permuted_block}"]  # fmt: skip
    axes = find_axes(argv)
    assert [a.label for a in axes] == ["isi_mean", "isi_min", "null_share", "order"]
    isi_mean, isi_min, null, order = axes
    assert not isi_mean.integer and isi_min.integer and null.percent and order.is_choice
    configs = sample(axes, 40, seed=1)
    for c in configs:
        out = render(argv, axes, c)
        assert "[" not in " ".join(out) and "{" not in " ".join(out)
        assert out[3].startswith("exp:") and out[3].endswith(",12")
        assert 3.0 <= float(c["isi_mean"]) <= 8.0 and c["isi_min"] in {"1", "2", "3"}
        assert c["null_share"].endswith("%") and c["order"] in {"random", "permuted_block"}
    # Latin hypercube: every tenth of a range is drawn once in 10 samples
    lh = sample(axes[:1], 10, seed=2)
    assert sorted(int((float(c["isi_mean"]) - 3) / 0.5) for c in lh) == list(range(10))


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


def test_pareto_ties_do_not_keep_dominated_points():
    assert pareto_front(np.array([1, 1, 1, 2]), np.array([2, 1, 1, 1])).tolist() == [
        False,
        True,
        True,
        False,
    ]


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
    assert names <= {
        "x_explore.tsv", "x_explore.png", "x_explore_liu.png", "x_explore_summary.txt",
        "x_explore_best",
    }  # fmt: skip
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


def test_detection_objective_is_the_mean_over_contrasts(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import RealizationScorer

    noise = [{"label": "t50", "tsnr": 50.0, "phys_fraction": 0.5, "tau": 6.0}]
    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:1", 10), Unit.parse("B", "B:1", 10)],
        isi=Interval.parse("exp:4,2,10"),
        post_fix=15,
    )
    sc = RealizationScorer(1.0, {"A": [1, 0], "A-B": [1, -1]}, noise).score(realize(spec, 0))
    need = sc["needed"]
    assert need[("t50", "detection")] == pytest.approx(
        (need[("t50", "A")] + need[("t50", "A-B")]) / 2
    )

    argv = ["-tr", "1", "-nruns", "1", "-scan_time", "150", "-initial_fix", "10", "-post_fix",
            "15", "-trial", "A", "0.25", "1", "-trial", "B", "0.25", "1",
            "-isi", "exp:[3.0-6.0],1,12", "-contrast", "A", "-contrast", "A-B", "-tsnr", "50",
            "-objective", "efficiency", "-explore", "5", "-explore_keep", "1", "-explore_pick", "2",
            "-device", "cpu", "-prefix", str(tmp_path / "d")]  # fmt: skip
    assert main(argv) == 0
    assert "detection, mean over contrasts" in capsys.readouterr().out


def test_axes_are_named_by_what_they_control_and_edges_are_flagged():
    from fastfuncstuff.simulation.explore import at_edges

    argv = ["-trial", "E1", "0.25", "[1-3]", "-trial", "E2", "0.25", "[1-3]",
            "-null", "0.25:uniform:[2-4],6", "[0-50%]", "-initial_fix", "[5-15]"]  # fmt: skip
    axes = find_axes(argv)
    assert [a.label for a in axes] == [
        "E1_count", "E2_count", "null_gap_low", "null_share", "initial_fix",
    ]  # fmt: skip
    configs = [{a.label: a.value(0.01) for a in axes}] * 3
    edges = {label for label, _, _ in at_edges(axes, configs, [0, 1, 2])}
    assert "initial_fix" in edges and "null_share" not in edges  # 0% has nothing below it


def test_shape_diff_objective_runs(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main

    argv = ["-tr", "1", "-nruns", "1", "-scan_time", "150", "-initial_fix", "10", "-post_fix",
            "15", "-trial", "E1", "0.25", "1", "-isi", "exp:[3.0-6.0],1,12", "-tsnr", "80",
            "-objective", "shape_diff", "-explore", "3", "-explore_designs", "1",
            "-explore_keep", "1", "-explore_pick", "1", "-device", "cpu", "-no_plots",
            "-prefix", str(tmp_path / "s")]  # fmt: skip
    assert main(argv) == 0
    assert "library steps two shapes must be apart" in capsys.readouterr().out


def test_combined_objective_in_explore(tmp_path, capsys):
    from fastfuncstuff.cli.simulate import main

    argv = ["-tr", "1", "-nruns", "1", "-scan_time", "150", "-initial_fix", "10", "-post_fix",
            "15", "-trial", "E1", "0.25", "1", "-isi", "exp:[3.0-9.0],1,14", "-tsnr", "60",
            "-objective", "detection=1,shape=0.5", "-explore", "5", "-explore_designs", "1",
            "-explore_keep", "2", "-explore_pick", "3", "-device", "cpu", "-no_plots",
            "-prefix", str(tmp_path / "c")]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "combined: detection x1 + shape x0.5" in out and "combined 0." in out
    header = (tmp_path / "c_explore.tsv").read_text().splitlines()[0].split("\t")
    assert "combined" in header


def test_worker_pool_scores_exactly_like_the_serial_loop():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit
    from fastfuncstuff.simulation.explore import score_configs
    from fastfuncstuff.simulation.power import RealizationScorer

    specs = [
        ExperimentSpec(
            tr=1.5, units=[Unit.parse("A", "A:1", n)], isi=Interval.parse(f"exp:{m},2,10"),
            post_fix=12,
        )
        for n, m in ((10, 4), (12, 5), (14, 3), (9, 6))
    ]  # fmt: skip
    noise = [{"label": "t50", "tsnr": 50.0}]
    scorer = RealizationScorer(1.5, {"A": [1]}, noise)
    serial = score_configs(specs, scorer, 50, "t50", progress=False)
    pooled = score_configs(specs, scorer, 50, "t50", progress=False, jobs=2)
    # Workers run single-threaded BLAS: the same numbers up to reduction order.
    for a, b in zip(pooled, serial, strict=True):
        assert a["needed"] == pytest.approx(b["needed"], rel=1e-6)
        assert a["shape_sd"] == pytest.approx(b["shape_sd"], rel=1e-6)


def test_refine_rescores_the_designs_near_the_front(tmp_path, capsys):
    import csv

    from fastfuncstuff.cli.simulate import main

    argv = ["-tr", "1", "-nruns", "1", "-initial_fix", "10", "-post_fix", "15",
            "-trial", "A", "0.25", "1", "-null", "0.25", "[0-30%]",
            "-isi", "exp:[3.0-6.0],1,10", "-tsnr", "50", "-scan_time", "200",
            "-explore", "20", "-explore_refine", "6", "-explore_keep", "2",
            "-explore_pick", "2", "-jobs", "1", "-device", "cpu", "-prefix", str(tmp_path / "x")]  # fmt: skip
    assert main(argv) == 0
    assert "refining" in capsys.readouterr().out
    rows = list(csv.DictReader((tmp_path / "x_explore.tsv").open(), delimiter="\t"))
    refined = [r for r in rows if r["realizations"] == "6"]
    assert refined and len(refined) < len(rows)
    assert all(r["realizations"] == "6" for r in rows if r["on_front"] == "1")
