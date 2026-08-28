"""``-hrf-library-raw`` wherever a custom HRF library is accepted.

ffs_librarian emits two library files that look identical -- impulse responses
in ``{prefix}_hrflibrary.tsv``, duration-convolved curves in
``{prefix}_hrfraw.tsv`` -- so which flag was used is the only record of which
kind is in hand.  Getting it wrong is silent in both directions: the -raw form
on an impulse library drops the stimulus duration from the model, and the plain
form on a duration-convolved library applies it twice.

One resolver serves every tool so the two cannot drift apart.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fastfuncstuff.cli_utils import resolve_hrf_library_spec


def _args(lib=None, raw=None):
    return SimpleNamespace(hrf_library=lib, hrf_library_raw=raw)


def _write(tmp_path, name, duration_convolved=None, duration=20.0):
    tsv = tmp_path / name
    tsv.write_text("0\t0\n1\t1\n")
    if duration_convolved is not None:
        stem = name.replace("_hrfraw.tsv", "").replace("_hrflibrary.tsv", "")
        (tmp_path / f"{stem}_metadata.json").write_text(
            json.dumps(
                {
                    "duration_convolved": duration_convolved,
                    "groups": [{"label": "all", "median_duration_s": duration}],
                }
            )
        )
    return str(tsv)


def test_no_library_passes_durations_through():
    path, durations, is_raw = resolve_hrf_library_spec(_args(), [2.0, 3.0])
    assert path is None and durations == [2.0, 3.0] and is_raw is False


def test_plain_library_passes_durations_through(tmp_path):
    lib = _write(tmp_path, "s_hrflibrary.tsv", duration_convolved=False)
    path, durations, is_raw = resolve_hrf_library_spec(_args(lib=lib), [20.0])
    assert path == lib and durations == [20.0] and is_raw is False


def test_raw_library_zeroes_the_model_durations(tmp_path):
    raw = _write(tmp_path, "s_hrfraw.tsv", duration_convolved=False)
    path, durations, is_raw = resolve_hrf_library_spec(_args(raw=raw), [20.0, 20.0])
    assert path == raw and durations == [0.0, 0.0] and is_raw is True


def test_hrfraw_accepted_though_metadata_says_not_duration_convolved(tmp_path):
    # The sidecar flag describes the FINAL LIBRARY, not _hrfraw.tsv, and reads
    # false whenever a deconvolution ran -- most of the time.  A naive check on
    # it rejects exactly the file the option exists to load.
    raw = _write(tmp_path, "s_hrfraw.tsv", duration_convolved=False)
    _, durations, is_raw = resolve_hrf_library_spec(_args(raw=raw), [20.0])
    assert is_raw is True and durations == [0.0]


def test_impulse_library_passed_as_raw_is_refused(tmp_path):
    lib = _write(tmp_path, "s_hrflibrary.tsv", duration_convolved=False)
    with pytest.raises(SystemExit):
        resolve_hrf_library_spec(_args(raw=lib), [20.0])


def test_both_flags_together_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        resolve_hrf_library_spec(_args(lib="a.tsv", raw=_write(tmp_path, "s_hrfraw.tsv")), [20.0])


def test_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        resolve_hrf_library_spec(_args(raw=str(tmp_path / "nope.tsv")), [20.0])


def test_duration_mismatch_warns_but_proceeds(tmp_path, capsys):
    raw = _write(tmp_path, "s_hrfraw.tsv", duration_convolved=False, duration=20.0)
    _, durations, is_raw = resolve_hrf_library_spec(_args(raw=raw), [2.0])
    assert is_raw is True and durations == [0.0]
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("tool", ["ridge", "denoise", "hrfopt"])
def test_every_library_tool_registers_both_flags(tool):
    import importlib

    parser = importlib.import_module(f"fastfuncstuff.cli.{tool}").create_parser()
    registered = set()
    for action in parser._actions:
        registered.update(action.option_strings)
    assert "-hrf-library" in registered, f"{tool} lost -hrf-library"
    assert "-hrf-library-raw" in registered, f"{tool} has no -hrf-library-raw"
