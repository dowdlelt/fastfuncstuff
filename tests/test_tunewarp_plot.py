"""Tests for the tunewarp frontier PNG.

A plot fails quietly: it writes a file whichever way the mapping is wrong, and
nobody checks the pixels. So these test the parts that carry meaning -- every
backend has a symbol, every config keeps its id, and a config that cannot be
placed on the axes is left off rather than drawn at a made-up position.
"""

from __future__ import annotations

import pytest

from fastfuncstuff.processing.tuneplot import BACKEND_MARKERS, plot_frontier
from fastfuncstuff.processing.tunespec import BACKENDS
from fastfuncstuff.processing.tunestore import BASELINE, RunMeta, TrialStore

pytest.importorskip("matplotlib")


def _store(tmp_path, backends=("qwarp", "formwarp")):
    s = TrialStore(tmp_path / "t.json")
    s.runs.append(
        RunMeta(
            run_id=1,
            started="2026-08-23T10:00:00",
            commit="abc1234",
            recipe="MNI_T1",
            contrast="same",
            optimize="lpa",
            panel=["ls", "lncc"],
            search="adaptive",
            subjects=["s1", "s2"],
            shape=(40, 40, 40),
            voxdims=(1.0, 1.0, 1.0),
            n_mask_voxels=1000,
        )
    )
    for subj in ("s1", "s2"):
        s.add(BASELINE, subj, {}, [], scores={"ls": 0.60, "lncc": -0.05}, seconds=0.0)
    for b in backends:
        for i, reg in enumerate((0.1, 1.0, 3.0)):
            for subj in ("s1", "s2"):
                s.add(
                    b,
                    subj,
                    {"reg": reg},
                    [],
                    scores={"ls": 0.30 + 0.02 * i, "lncc": -0.20 + 0.01 * i},
                    seconds=10.0 * (i + 1),
                    grade="pass",
                    gate_margin=0.2,
                    warpqc={"bending_energy": 0.01 * (3 - i), "jac_min": 0.4},
                )
    s.compute_consensus(["ls", "lncc"])
    return s


def test_every_backend_has_a_symbol():
    """A backend added to the search must not silently share a marker with another.

    The plot encodes the tool as the shape, so an unmapped backend would fall back
    to the same catch-all glyph as the next one and two tools would read as one.
    """
    assert set(BACKEND_MARKERS) == set(BACKENDS)
    assert len(set(BACKEND_MARKERS.values())) == len(BACKEND_MARKERS)


def test_writes_a_png_and_labels_every_config(tmp_path):
    store = _store(tmp_path)
    out = tmp_path / "sub" / "frontier.png"  # a directory that does not exist yet
    assert plot_frontier(store.results(), out) == str(out)
    assert out.stat().st_size > 5000


def test_config_ids_are_all_drawn(tmp_path):
    """The id is what `-reproduce N` takes, so every plotted config must show one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = _store(tmp_path)
    results = store.results()
    plot_frontier(results, tmp_path / "f.png")
    # plot_frontier closes its own figure; re-draw into a fresh one to inspect.
    drawn = {
        t.get_text() for t in _last_axes(plt, plot_frontier, results, tmp_path / "f2.png").texts
    }
    expected = {str(r.config_id) for r in results if not r.is_baseline}
    assert expected <= drawn


def _last_axes(plt, fn, results, path):
    """Run the plotter with figure closing disabled, and hand back its axes."""
    closed = plt.close
    plt.close = lambda *a, **k: None
    try:
        fn(results, path)
        return plt.gcf().axes[0]
    finally:
        plt.close = closed
        closed("all")


def test_a_table_with_no_warps_draws_nothing(tmp_path):
    """A run where nothing produced a field is not an error, and not a picture."""
    s = TrialStore(tmp_path / "t.json")
    s.runs.append(
        RunMeta(
            run_id=1,
            started="2026-08-23T10:00:00",
            commit="abc1234",
            recipe="MNI_T1",
            contrast="same",
            optimize="lpa",
            panel=["ls"],
            search="grid",
            subjects=["s1"],
            shape=(10, 10, 10),
            voxdims=(1.0, 1.0, 1.0),
            n_mask_voxels=10,
        )
    )
    s.add(BASELINE, "s1", {}, [], scores={"ls": 0.6}, seconds=0.0)
    s.compute_consensus(["ls"])
    out = tmp_path / "none.png"
    assert plot_frontier(s.results(), out) is None
    assert not out.exists()
