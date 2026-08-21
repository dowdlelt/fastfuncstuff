"""Tests for static shell-completion generation.

The generator reaches into a CLI's parser by monkeypatching
``ArgumentParser.parse_args``, and it classifies each flag by metavar/dest/type.
Both are the kind of heuristic that rots silently -- a completion that offers
nothing looks like "shell completion is just like that" rather than a bug.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

import pytest

from fastfuncstuff.completion import (
    OptionSpec,
    _completion_kind,
    describe,
    load_parser,
    render_bash,
    render_fish,
)


def _action(**kwargs) -> argparse.Action:
    parser = argparse.ArgumentParser()
    return parser.add_argument("-x", **kwargs)


class TestCompletionKind:
    def test_choices_win(self):
        assert _completion_kind(_action(choices=["a", "b"])) == "choices"

    def test_file_metavar(self):
        assert _completion_kind(_action(metavar="FILE")) == "file"

    def test_dir_metavar(self):
        assert _completion_kind(_action(metavar="DIR")) == "dir"

    def test_device_dest_is_special_cased(self):
        parser = argparse.ArgumentParser()
        action = parser.add_argument("-device", metavar="SPEC")
        assert _completion_kind(action) == "device"

    def test_numeric_type_never_completes_files(self):
        """-round_onsets THRESHOLD matches the 'onsets' dest hint but is a float."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("-round_onsets", type=float, metavar="THRESHOLD")
        assert _completion_kind(action) == "none"

    def test_explicit_non_file_metavar_outranks_dest_hint(self):
        """A chosen metavar is a statement that the argument is not a path."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("-flobs_prior_weight", metavar="VALUE")
        assert _completion_kind(action) == "none"

    def test_dest_hint_applies_when_there_is_no_metavar(self):
        parser = argparse.ArgumentParser()
        action = parser.add_argument("-mask")
        assert _completion_kind(action) == "file"


class TestDescribe:
    def test_store_true_takes_no_value(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-flag", action="store_true")
        spec = next(s for s in describe(parser) if "-flag" in s.option_strings)
        assert spec.takes_value is False

    def test_aliases_are_kept_together(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-do_blur", "-do-blur", dest="do_blur", type=float, metavar="FWHM")
        spec = next(s for s in describe(parser) if "-do_blur" in s.option_strings)
        assert spec.option_strings == ["-do_blur", "-do-blur"]

    def test_help_is_collapsed_to_one_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-x2", help="line one\n    and  line   two")
        spec = next(s for s in describe(parser) if "-x2" in s.option_strings)
        assert "\n" not in spec.help
        assert "  " not in spec.help


class TestLoadParser:
    def test_recovers_parser_from_a_parse_args_that_actually_parses(self):
        """Half the CLIs parse inside parse_args(); the hook must still work."""
        parser = load_parser("fastfuncstuff.cli.deconvolve")
        assert parser is not None
        flags = {opt for spec in describe(parser) for opt in spec.option_strings}
        assert {"-input", "-prefix", "-model", "-ortvec_glob"} <= flags

    def test_parse_args_is_restored_afterwards(self):
        """The monkeypatch must not leak into the rest of the process."""
        before = argparse.ArgumentParser.parse_args
        load_parser("fastfuncstuff.cli.deconvolve")
        assert argparse.ArgumentParser.parse_args is before
        # And a real parse still works.
        p = argparse.ArgumentParser()
        p.add_argument("-a")
        assert p.parse_args(["-a", "1"]).a == "1"

    def test_unknown_module_returns_none(self):
        assert load_parser("fastfuncstuff.cli.does_not_exist") is None


class TestRenderFish:
    def test_single_dash_long_flags_use_old_style(self):
        """AFNI-style -input is fish's -o, not -l; -l would complete nothing."""
        text = render_fish("t", [OptionSpec(option_strings=["-input"], takes_value=True)])
        assert "-o input" in text
        assert "-l input" not in text

    def test_gnu_long_and_short_forms(self):
        text = render_fish(
            "t",
            [
                OptionSpec(option_strings=["--cpu"]),
                OptionSpec(option_strings=["-h"]),
            ],
        )
        assert "-l cpu" in text
        assert "-s h" in text

    def test_help_text_with_quote_is_escaped(self):
        text = render_fish("t", [OptionSpec(option_strings=["-x"], help="don't break the line")])
        assert r"don\'t" in text


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_generated_bash_is_syntactically_valid():
    parser = load_parser("fastfuncstuff.cli.deconvolve")
    assert parser is not None
    script = render_bash("ffs_deconvolve", describe(parser))
    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(not shutil.which("fish"), reason="fish not available")
def test_generated_fish_loads_and_completes(tmp_path):
    parser = load_parser("fastfuncstuff.cli.deconvolve")
    assert parser is not None
    path = tmp_path / "ffs_deconvolve.fish"
    path.write_text(render_fish("ffs_deconvolve", describe(parser)))

    proc = subprocess.run(
        ["fish", "-c", f"source {path}; complete -C'ffs_deconvolve -model '"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    offered = {line.split("\t")[0] for line in proc.stdout.splitlines()}
    assert {"TENT", "TENTzero", "FIR", "CSPLIN"} <= offered
