"""``ffs_hrfopt -save_selected_tr``: the selected HRF on a comparable grid.

``{prefix}_selected_hrfs.nii.gz`` is written at ``-microtime_dt`` (0.1 s),
which is right for reuse but awkward to hold against a FIR estimate sampled
every TR.  This writes the same curves decimated onto the FIR lag spacing, so
the pair lines up voxel-for-voxel with ffs_librarian's ``-save_fir_volume``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fastfuncstuff.cli.hrfopt import _selected_tr_dt


def _args(spec, tr=2.0):
    return SimpleNamespace(save_selected_tr=spec, tr=tr)


def test_unset_means_do_not_write_it():
    assert _selected_tr_dt(_args(None)) is None


def test_bare_flag_uses_the_run_tr():
    # The bare flag has to mean the TR: that is the spacing a TR-locked FIR
    # uses, and lining up with it is the whole point of the option.
    assert _selected_tr_dt(_args("tr", tr=2.0)) == 2.0
    assert _selected_tr_dt(_args("tr", tr=1.5)) == 1.5


def test_explicit_seconds_override_the_tr():
    # Sub-TR FIR windows exist, so the spacing is not always the TR.
    assert _selected_tr_dt(_args("0.5", tr=2.0)) == 0.5


@pytest.mark.parametrize("spec", ["abc", "0", "-1"])
def test_bad_values_exit(spec):
    with pytest.raises(SystemExit):
        _selected_tr_dt(_args(spec))


def test_decimation_matches_the_fir_sample_instants():
    # Decimate, do not resample.  A FIR estimate IS the response at those
    # instants, so taking the matching samples is the like-for-like
    # comparison; interpolating first would smooth the fine curve into
    # something the FIR never claimed.
    dt, tr, n = 0.1, 2.0, 320
    t = np.arange(n) * dt
    curve = np.sin(t / 4.0)
    step = max(1, int(round(tr / dt)))
    decimated = curve[::step]
    np.testing.assert_allclose(decimated, np.sin(np.arange(decimated.size) * tr / 4.0), atol=1e-12)
    assert decimated.size == int(np.ceil(n / step))


def test_flag_is_registered_with_bare_and_valued_forms():
    from fastfuncstuff.cli.hrfopt import create_parser

    base = ["-input", "a.nii", "-prefix", "p", "-onsets", "o.1D", "-durations", "2"]
    assert create_parser().parse_args(base).save_selected_tr is None
    assert create_parser().parse_args([*base, "-save_selected_tr"]).save_selected_tr == "tr"
    assert create_parser().parse_args([*base, "-save-selected-tr", "1.5"]).save_selected_tr == "1.5"
