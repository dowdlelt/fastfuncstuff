"""ffs_util_eventcheck: dangers, fixes and the finest usable knot spacing."""

import numpy as np

from fastfuncstuff.cli.util_eventcheck import main

RNG = np.random.default_rng(5)


def _timing(tmp_path, phase_fn, n_runs=3, tr=2.0):
    rows = []
    for _ in range(n_runs):
        base = np.sort(RNG.choice(np.arange(3, 180), 25, replace=False)) * tr
        rows.append(" ".join(f"{t:.2f}" for t in phase_fn(base)))
    path = tmp_path / "stim.1D"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _run(capsys, *argv):
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def test_one_shared_mid_tr_phase_is_flagged_with_the_aligned_fix(tmp_path, capsys):
    out = _run(capsys, "-onsets", _timing(tmp_path, lambda b: b + 1.0), "-TR", "2")
    assert "SINGULAR" in out
    assert "-window 1 17 -tent-n-basis 9" in out  # knots on the mid-TR samples


def test_locked_plus_mid_tr_timing_offers_half_tr_knots(tmp_path, capsys):
    stim = _timing(tmp_path, lambda b: b + RNG.choice([0.0, 1.0], b.size))
    out = _run(capsys, "-onsets", stim, "-TR", "2")
    assert "Finest without smoothing: 1 s" in out


def test_logging_noise_does_not_masquerade_as_spread_timing(tmp_path, capsys):
    jitter = lambda b: b + RNG.uniform(-0.004, 0.004, b.size)  # noqa: E731
    out = _run(capsys, "-onsets", _timing(tmp_path, jitter), "-TR", "2")
    assert "Onsets already sit on the samples" in out
    assert "stay with TR knots" in out
