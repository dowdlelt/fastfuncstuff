"""``ffs_hrfopt -oso_mode``: the flag surface and its refusals.

The library side is covered by ``test_onset_offset.py`` and
``test_hrf_selection_oso.py``; this file is about what the CLI accepts, what it
refuses, and which of those refusals must happen BEFORE the data is loaded.
"""

from __future__ import annotations

from fastfuncstuff.cli.hrfopt import create_parser
from fastfuncstuff.design.onset_offset import DEFAULT_VIF_MAX


def _parse(extra):
    parser = create_parser()
    return parser.parse_args(["-input", "r1.nii.gz", "-events", "e1.tsv", "-prefix", "out", *extra])


def test_default_is_off():
    args = _parse([])
    assert args.oso_mode == "off"
    assert args.oso_gain is False
    # None, not the constant: the default is resolved where the design modules
    # are already imported, so building the parser stays cheap.
    assert args.oso_vif_max is None


def test_modes_parse():
    for mode in ("off", "joint", "staged"):
        assert _parse(["-oso_mode", mode]).oso_mode == mode


def test_dash_and_underscore_spellings_are_twins():
    """FfsArgumentParser derives the -foo-bar twin of every -foo_bar flag."""
    assert _parse(["-oso-mode", "joint"]).oso_mode == "joint"
    assert _parse(["-oso-vif-max", "3"]).oso_vif_max == 3.0
    assert _parse(["-oso-gain"]).oso_gain is True


def test_vif_max_override():
    assert _parse(["-oso_vif_max", "2.5"]).oso_vif_max == 2.5
    assert DEFAULT_VIF_MAX == 5.0


def test_help_mentions_the_gate():
    """A user reading -help must learn the mode needs long events."""
    text = create_parser().format_help()
    assert "-oso_mode" in text
    assert "gated" in text or "gate" in text
    assert "waveshape" in text
