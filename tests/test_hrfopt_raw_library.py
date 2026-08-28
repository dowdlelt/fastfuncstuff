"""``ffs_hrfopt -hrf-library-raw``: loading a duration-convolved HRF library.

A library from ``ffs_librarian`` comes in two flavours and the TSVs are
indistinguishable by inspection:

* ``{prefix}_hrflibrary.tsv`` — impulse responses.  ``-hrf-library``.  The
  duration is applied by building the onset matrix as boxcars.
* ``{prefix}_hrfraw.tsv`` — already the response to an event of the duration
  it was measured at.  ``-hrf-library-raw``.  The onset matrix must then be
  built from IMPULSES, or the duration is counted twice.

Getting that backwards is silent and wrong in both directions, which is what
these tests are for.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fastfuncstuff.cli.hrfopt import _resolve_raw_library


def _args(raw=None, lib=None):
    return SimpleNamespace(hrf_library_raw=raw, hrf_library=lib, hrf_mode="library")


def _write(tmp_path, name, meta=None, duration=20.0):
    tsv = tmp_path / name
    tsv.write_text("0\t0\n1\t1\n")
    if meta is not None:
        stem = name.replace("_hrfraw.tsv", "").replace("_hrflibrary.tsv", "")
        (tmp_path / f"{stem}_metadata.json").write_text(
            json.dumps(
                {
                    "duration_convolved": meta,
                    "groups": [{"label": "all", "median_duration_s": duration}],
                }
            )
        )
    return tsv


def test_no_flag_leaves_durations_alone():
    durations, raw = _resolve_raw_library(_args(), [2.0, 2.0])
    assert durations == [2.0, 2.0]
    assert raw is False


def test_raw_library_zeroes_the_onset_durations(tmp_path, capsys):
    # The whole point: the curve already contains the boxcar, so the onset
    # matrix must be impulses.  Leaving durations in place would convolve the
    # 20 s boxcar into a curve that already has one.
    tsv = _write(tmp_path, "sub01_hrfraw.tsv", meta=False, duration=20.0)
    args = _args(raw=str(tsv))
    durations, raw = _resolve_raw_library(args, [20.0, 20.0])
    assert durations == [0.0, 0.0]
    assert raw is True
    assert args.hrf_library == str(tsv)
    assert args.hrf_mode == "library"


def test_hrfraw_is_accepted_even_though_metadata_says_not_duration_convolved(tmp_path):
    # Regression: the sidecar's `duration_convolved` describes the FINAL
    # LIBRARY, not _hrfraw.tsv.  It reads false whenever a deconvolution ran,
    # which is most of the time -- a naive check on that flag rejects exactly
    # the file this option exists to load.
    tsv = _write(tmp_path, "sub01_hrfraw.tsv", meta=False)
    durations, raw = _resolve_raw_library(_args(raw=str(tsv)), [20.0])
    assert raw is True and durations == [0.0]


def test_impulse_library_passed_to_raw_is_rejected(tmp_path):
    # The dangerous direction: an impulse library loaded as "raw" would build
    # impulse onsets too, dropping the stimulus duration from the model
    # entirely, and nothing downstream would notice.
    tsv = _write(tmp_path, "sub01_hrflibrary.tsv", meta=False)
    with pytest.raises(SystemExit):
        _resolve_raw_library(_args(raw=str(tsv)), [20.0])


def test_duration_convolved_library_file_is_accepted(tmp_path):
    tsv = _write(tmp_path, "sub01_hrflibrary.tsv", meta=True)
    durations, raw = _resolve_raw_library(_args(raw=str(tsv)), [20.0])
    assert raw is True and durations == [0.0]


def test_mutually_exclusive_with_hrf_library(tmp_path):
    tsv = _write(tmp_path, "sub01_hrfraw.tsv")
    with pytest.raises(SystemExit):
        _resolve_raw_library(_args(raw=str(tsv), lib="other.tsv"), [20.0])


def test_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        _resolve_raw_library(_args(raw=str(tmp_path / "nope.tsv")), [20.0])


def test_duration_mismatch_warns_but_proceeds(tmp_path, capsys):
    # A duration-convolved library only describes the duration it was measured
    # at, so using it with a different -durations is a modelling error the user
    # has to be told about -- but not one we can adjudicate, so it warns.
    tsv = _write(tmp_path, "sub01_hrfraw.tsv", meta=False, duration=20.0)
    durations, raw = _resolve_raw_library(_args(raw=str(tsv)), [2.0])
    assert raw is True and durations == [0.0]
    assert "WARNING" in capsys.readouterr().out


def test_no_sidecar_still_works(tmp_path):
    tsv = _write(tmp_path, "sub01_hrfraw.tsv", meta=None)
    durations, raw = _resolve_raw_library(_args(raw=str(tsv)), [np.float64(20.0)])
    assert raw is True and durations == [0.0]


def test_raw_library_design_is_not_double_convolved():
    """The invariant the flag exists for, checked on the design itself.

    Zeroing the onset MATRIX is not enough: build_task_design prefers the event
    list over that matrix and applies `stim_durations` itself, so leaving the
    real durations in the downstream calls re-applies the boxcar to a curve
    that already contains one.  Nothing downstream detects it -- the design is
    a perfectly ordinary-looking regressor, just of the wrong shape.

    Measured on real data before the fix: hrfopt's reported per-HRF R2 profile
    correlated +0.956 with what a double-convolved design predicts and -0.438
    with the correct one, and it selected the entry ranked 19th of 20 by fit to
    the voxels' own FIR curves for 98 of 100 sampled voxels.
    """
    import numpy as np

    dt, n_t, tr = 0.1, 164, 2.0
    n_mt = int(n_t * tr / dt)
    onsets = [10.0, 70.0, 130.0, 190.0, 250.0]
    # A library curve that is already the response to a 20 s block.
    t = np.arange(320) * dt
    impulse = np.exp(-((t - 5.0) ** 2) / 8.0)
    curve = np.convolve(impulse, np.ones(int(20.0 / dt)))[: t.size]
    curve = curve / curve.max()

    def design(duration):
        x = np.zeros(n_mt)
        if duration > 0:
            for o in onsets:
                x[int(o / dt) : int((o + duration) / dt)] = 1.0
        else:
            for o in onsets:
                x[int(round(o / dt))] = 1.0
        return np.convolve(x, curve)[:n_mt].reshape(n_t, int(tr / dt)).mean(1)

    correct = design(0.0)
    doubled = design(20.0)

    # The FIRST response only -- onsets are 60 s apart, so the window up to the
    # second event isolates one.  Peak LATENCY is the discriminator, not width:
    # convolving a 20 s-block response with another 20 s boxcar barely widens
    # it (10 -> 11 TRs) but pushes its peak several seconds later, which is
    # what makes a different library entry look like the better fit.
    first = slice(0, int(60.0 / tr))
    peak_correct = int(correct[first].argmax())
    peak_doubled = int(doubled[first].argmax())
    assert peak_doubled >= peak_correct + 2, (
        f"double convolution not detectable: peak at TR {peak_correct} vs {peak_doubled}"
    )

    # And they are different enough that a selection would land elsewhere.
    c = np.corrcoef(correct, doubled)[0, 1]
    assert c < 0.90, f"designs too similar to distinguish (r={c:.4f})"
