"""Ortvec entries: the grammar, and the arithmetic behind derivatives and bands."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer import ortvec


def test_an_entry_peels_transforms_from_the_right_only():
    assert ortvec.parse_entry("a.1D") == ("a.1D", [])
    assert ortvec.parse_entry("a.1D:deriv:deriv") == ("a.1D", ["deriv", "deriv"])
    path, ops = ortvec.parse_entry("/d/a.1D:cols=0,2:band=0.01-nyq")
    assert path == "/d/a.1D"
    assert ops == [(0, 2), ortvec.Band(0.01, None)]
    # A colon that belongs to the path stays in it.
    assert ortvec.parse_entry("C:/data/a.1D:deriv") == ("C:/data/a.1D", ["deriv"])


def test_a_band_spec_round_trips_without_an_exponent():
    """``1e-05`` has a dash in it, and the dash is the band separator."""
    band = ortvec.Band(0.00001234, 0.1)
    assert "e" not in band.spec()
    assert ortvec.parse_entry(f"a.1D:{band.spec()}")[1] == [ortvec.Band(0.00001234, 0.1)]


def test_the_bands_of_a_split_sum_to_the_column():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(137, 4)), axis=0) + 5.0
    tr = 1.5
    entries = ortvec.split_entries("m.1D", [0.02, 0.1], n_columns=4)
    total = np.zeros_like(x)
    for entry in entries:
        out, _ = ortvec.apply_ops(x, [f"c{i}" for i in range(4)], ortvec.parse_entry(entry)[1], tr)
        total += out
    assert len(entries) == 3
    assert np.allclose(total, x)


def test_a_band_holds_only_its_frequencies():
    tr, n = 1.0, 200
    t = np.arange(n) * tr
    slow = np.cos(2 * np.pi * 0.02 * t)
    fast = np.cos(2 * np.pi * 0.3 * t)
    x = (slow + fast)[:, None]
    low = ortvec.band_columns(x, tr, ortvec.Band(0.0, 0.1))[:, 0]
    high = ortvec.band_columns(x, tr, ortvec.Band(0.1, None))[:, 0]
    assert np.corrcoef(low, slow)[0, 1] > 0.99
    assert np.corrcoef(high, fast)[0, 1] > 0.99


def test_a_derivative_is_the_cli_backward_difference():
    x = np.cumsum(np.ones((10, 2)), axis=0)
    out, labels = ortvec.apply_ops(x, ["roll", "yaw"], ["deriv"], 2.0)
    assert labels == ["roll'", "yaw'"]
    assert out[0].tolist() == [0.0, 0.0]
    assert np.allclose(out[1:], 1.0)


def test_labels_say_what_was_done():
    x = np.ones((50, 2))
    _, labels = ortvec.apply_ops(x, ["roll", "yaw"], [(1,), ortvec.Band(0.0, 0.05), "deriv"], 2.0)
    assert labels == ["yaw <0.05Hz'"]
    assert ortvec.describe_entry("/d/m.1D:cols=1:band=0-0.05:deriv") == "m.1D  [1]  <0.05Hz  d/dt"


def test_a_partial_split_keeps_the_unsplit_columns():
    entries = ortvec.split_entries("m.1D", [0.05], n_columns=3, columns=[1])
    assert entries == ["m.1D:cols=1:band=0-0.05", "m.1D:cols=1:band=0.05-nyq", "m.1D:cols=0,2"]


def test_a_column_that_is_not_there_is_an_error():
    with pytest.raises(IndexError, match="cols"):
        ortvec.apply_ops(np.ones((5, 2)), ["a", "b"], [(3,)], 1.0)


def test_spectra_are_shares_of_variance():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 3)) * [1.0, 10.0, 100.0] + 7.0
    freqs, power = ortvec.column_spectra(x, 2.0)
    assert freqs[0] > 0 and freqs.size == 63
    assert np.allclose(power.sum(axis=0), 1.0)
