"""Tests for the shared CLI help/parser layer.

The alias contract is invisible when it works and silent when it breaks: a flag
that accepts only one of its two spellings looks like a typo on the user's end,
not a bug on ours. 646 of the toolbox's 1113 multi-word flags were in that state
before FfsArgumentParser, so these tests are the thing keeping it from drifting
back.
"""

from __future__ import annotations

import argparse
import contextlib
import io

import pytest

from fastfuncstuff.cli_help import (
    FfsArgumentParser,
    FfsHelpFormatter,
    canonical_option_strings,
    spelling_variants,
    suggest,
)


class TestSpellingVariants:
    def test_underscore_and_hyphen_are_variants_of_each_other(self):
        assert spelling_variants("-event_ignore") == ["-event-ignore"]
        assert spelling_variants("-event-ignore") == ["-event_ignore"]

    def test_a_single_word_flag_has_no_variant(self):
        assert spelling_variants("-input") == []
        assert spelling_variants("--help") == []

    def test_a_mixed_name_yields_both_uniform_forms(self):
        assert spelling_variants("-hrf-n_shapes") == ["-hrf-n-shapes", "-hrf_n_shapes"]

    def test_the_leading_dashes_survive(self):
        assert spelling_variants("--long_flag") == ["--long-flag"]


class TestAliasContract:
    def test_a_flag_registered_one_way_parses_the_other(self):
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-event_ignore")
        parser.add_argument("-single-trials", action="store_true")
        assert parser.parse_args(["-event-ignore", "x"]).event_ignore == "x"
        assert parser.parse_args(["-single_trials"]).single_trials is True

    def test_hidden_aliases_stay_out_of_help_and_usage(self):
        """The whole point of not touching option_strings.

        -help, the usage line and the completion generator all read
        option_strings; an auto-added spelling must be typeable and invisible.
        """
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-event_ignore", metavar="LABEL")
        parser.parse_known_args([])  # aliases materialise here
        text = parser.format_help()
        assert "-event_ignore" in text
        assert "-event-ignore" not in text
        assert [a.option_strings for a in parser._actions if a.dest == "event_ignore"] == [
            ["-event_ignore"]
        ]

    def test_a_real_flag_is_never_shadowed(self):
        """If both spellings are already taken by different flags, hands off."""
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-a_b", dest="first")
        parser.add_argument("-a-b", dest="second")
        parser.parse_known_args([])
        assert parser.parse_args(["-a_b", "x"]).first == "x"
        assert parser.parse_args(["-a-b", "y"]).second == "y"

    def test_declared_aliases_still_work(self):
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-drop_first", "-skip_first", type=int)
        assert parser.parse_args(["-skip-first", "3"]).drop_first == 3


class TestAbbreviation:
    def _error(self, parser, argv) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            parser.parse_args(argv)
        return err.getvalue()

    def test_two_spellings_of_one_flag_are_not_ambiguous(self):
        """argparse counts option STRINGS, not flags, when judging a prefix."""
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-drop_first", "-drop-first", type=int)
        assert parser.parse_args(["-drop_f", "3"]).drop_first == 3

    def test_a_genuine_conflict_still_reports_one_name_per_flag(self):
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-drop_first", "-drop-first", type=int)
        parser.add_argument("-drop_last", "-drop-last", type=int)
        message = self._error(parser, ["-dro", "3"])
        assert "ambiguous" in message
        assert "-drop_first" in message and "-drop_last" in message
        # ...and NOT the four-name version argparse would print unaided.
        assert "-drop-first" not in message

    def test_a_hidden_alias_never_wins_an_abbreviation(self):
        """Adding 552 hidden spellings must not create a new ambiguity."""
        parser = FfsArgumentParser(prog="demo")
        parser.add_argument("-alpha_one", type=int)
        parser.parse_known_args([])
        assert parser.parse_args(["-alpha_o", "1"]).alpha_one == 1


class TestHelpFormatter:
    def _help_for(self, **kwargs) -> str:
        parser = FfsArgumentParser(prog="demo", formatter_class=FfsHelpFormatter)
        parser.add_argument("-x", **kwargs)
        return parser.format_help()

    def test_none_and_false_defaults_are_not_printed(self):
        assert "default" not in self._help_for(default=None, help="a thing")
        assert "default" not in self._help_for(action="store_true", help="a thing")

    def test_a_default_the_author_already_wrote_is_not_doubled(self):
        text = self._help_for(default=20, type=int, help="how many (default: 20)")
        assert text.count("default: 20") == 1

    def test_store_false_does_not_borrow_its_partners_default(self):
        parser = FfsArgumentParser(prog="demo", formatter_class=FfsHelpFormatter)
        parser.add_argument("-no_thing", dest="thing", action="store_false", help="turn it off")
        assert "default" not in parser.format_help()

    def test_a_real_default_is_printed(self):
        assert "default: 7" in self._help_for(default=7, type=int, help="how many")

    def test_newlines_in_help_survive(self):
        text = self._help_for(help="pick one:\n  a   the first\n  b   the second")
        assert "a   the first" in text
        assert "b   the second" in text

    def test_spelling_variants_collapse_but_real_aliases_print(self):
        parser = FfsArgumentParser(prog="demo", formatter_class=FfsHelpFormatter)
        parser.add_argument("-max_comps", "-max-comps", "-max_pcs", type=int, help="n")
        text = parser.format_help()
        assert "-max_comps, -max_pcs" in text
        assert "-max-comps" not in text


def test_canonical_option_strings_keeps_the_first_spelling():
    assert canonical_option_strings(["-drop_first", "-drop-first", "-skip_first"]) == [
        "-drop_first",
        "-skip_first",
    ]


def test_canonical_option_strings_catches_a_dropped_separator():
    """-nocoverage vs -no_coverage: ten tools shipped both as separate entries."""
    assert canonical_option_strings(["-no_coverage", "-nocoverage"]) == ["-no_coverage"]


def test_suggest_returns_the_action_so_it_can_wrap_add_argument():
    parser = FfsArgumentParser(prog="demo")
    action = suggest(parser.add_argument("-device", metavar="SPEC"), ("auto", "cpu"))
    assert action.ffs_suggest == ["auto", "cpu"]
    assert action.dest == "device"
    # A hint is not a constraint: anything still parses.
    assert parser.parse_args(["-device", "cuda,0"]).device == "cuda,0"


def test_the_default_formatter_is_the_shared_one():
    """A new tool gets the house style without having to remember it."""
    assert FfsArgumentParser(prog="demo").formatter_class is FfsHelpFormatter
    assert argparse.ArgumentParser(prog="demo").formatter_class is not FfsHelpFormatter
