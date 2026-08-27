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

from fastfuncstuff.cli_help import suggest
from fastfuncstuff.completion import (
    OptionSpec,
    _completion_kind,
    describe,
    load_parser,
    render_bash,
    render_fish,
    render_zsh,
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

    def test_unannotated_flag_falls_back_to_files(self):
        """ffs_tunewarp -out ./so<TAB> used to beep: no type, no metavar."""
        parser = argparse.ArgumentParser()
        assert _completion_kind(parser.add_argument("-out")) == "file"
        assert _completion_kind(parser.add_argument("-mask")) == "file"

    def test_metavar_extension_is_read_as_a_path(self):
        parser = argparse.ArgumentParser()
        assert (
            _completion_kind(parser.add_argument("-useweight", metavar="WEIGHT.nii.gz")) == "file"
        )
        assert _completion_kind(parser.add_argument("-aff12", metavar="MOCO.aff12.1D")) == "file"
        assert _completion_kind(parser.add_argument("-plot", metavar="FILE.png")) == "file"

    def test_tuple_metavar_completes_files_when_any_slot_is_one(self):
        """-stim TIMING_FILE HRF_MODEL LABEL: the shell cannot tell which slot."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument(
            "-stim", nargs=3, metavar=("TIMING_FILE", "HRF_MODEL", "LABEL")
        )
        assert _completion_kind(action) == "file"

    def test_alternation_metavar_splits_on_pipe(self):
        parser = argparse.ArgumentParser()
        assert _completion_kind(parser.add_argument("-adjust_dof", metavar="MAP|N")) == "file"
        assert _completion_kind(parser.add_argument("-work_dxyz", metavar="auto|off|MM")) == "none"


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

    def test_hyphen_underscore_variants_collapse_to_one_offer(self):
        """Both spellings parse; only one should clutter the completion list."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-hrf-shapes", "-hrf_shapes", dest="hrf_shapes")
        spec = next(s for s in describe(parser) if "-hrf-shapes" in s.option_strings)
        assert spec.option_strings == ["-hrf-shapes", "-hrf_shapes"]  # still MATCHED
        assert spec.display_strings == ["-hrf-shapes"]  # only one OFFERED

    def test_renamed_flags_offer_only_the_primary_name(self):
        """Aliases are accepted, not suggested -- however different they look.

        ffs_fitbasis accepts five spellings of -hrf; listing all five makes the
        user read the completion list five times to find one flag. The other
        names still parse, and -help still prints them.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-hrf-shapes", "-hrf_shapes", "-shift-shapes", "-shift_shapes", dest="hrf_shapes"
        )
        spec = next(s for s in describe(parser) if "-hrf-shapes" in s.option_strings)
        assert spec.display_strings == ["-hrf-shapes"]
        assert "-shift-shapes" in spec.option_strings  # still MATCHED

    def test_underscore_primary_is_not_hidden_by_its_hyphen_alias(self):
        """Why the rule is "the first spelling", not "drop underscores".

        -drop_first is the documented name and -drop-first only its alias, so
        a blanket underscore filter would offer the alias and hide the flag.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-drop_first", "-drop-first", "-skip_first", "-skip-first", dest="drop_first", type=int
        )
        spec = next(s for s in describe(parser) if "-drop_first" in s.option_strings)
        assert spec.display_strings == ["-drop_first"]

    def test_dropped_separator_variants_collapse_too(self):
        """-nocoverage vs -no_coverage: normalising _ to - alone misses these.

        Ten tools had a pair like this reach the generated completions as two
        entries, because the old key only replaced underscores with hyphens.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-no_coverage", "-nocoverage", dest="no_coverage", action="store_true")
        specs = [s for s in describe(parser) if "-no_coverage" in s.option_strings]
        assert len(specs) == 1
        assert specs[0].display_strings == ["-no_coverage"]

    def test_alias_registered_as_a_second_argument_is_merged(self):
        """ffs_moco registers -chain-init as its own add_argument, not an alias.

        Per-action deduping cannot see that the two are one flag, so both
        reached the completion list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-chain_init", dest="chain_init", action="store_true")
        parser.add_argument("-chain-init", dest="chain_init", action="store_true")
        specs = [s for s in describe(parser) if "chain" in s.option_strings[0]]
        assert len(specs) == 1
        assert specs[0].display_strings == ["-chain_init"]
        assert "-chain-init" in specs[0].option_strings  # still MATCHED

    def test_suppressed_flags_are_not_offered(self):
        """A hidden flag used to be offered with "==SUPPRESS==" as its help."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-secret", help=argparse.SUPPRESS)
        assert not [s for s in describe(parser) if "-secret" in s.option_strings]

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
def test_bash_offers_one_spelling_but_still_matches_aliases():
    """The split that makes deduping safe: suggest one, match all.

    What to complete AFTER a flag has to work for whichever alias the user
    typed, even though only one spelling is ever suggested.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hrf-select", "-hrf_select", dest="hrf_select", choices=["xval", "full", "none"]
    )
    script = render_bash("ffs_demo", describe(parser))
    suggest = [ln for ln in script.splitlines() if "compgen -W" in ln and "$_opts" not in ln]
    assert len(suggest) == 1
    assert "-hrf-select" in suggest[0]
    assert "-hrf_select" not in suggest[0], "alias leaked into the suggestion list"
    # ...but both spellings still select the choice list.
    assert "-hrf-select|-hrf_select)" in script


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


def test_generated_zsh_quoting_is_balanced():
    """The hazard in an _arguments spec is quoting, and bash grades that.

    zsh is not installed everywhere, but the two shells share the word grammar,
    so ``bash -n`` catches an unbalanced quote or a broken continuation -- which
    is what ffs help text full of ``[``, ``]``, ``:`` and apostrophes threatens.
    """
    parser = load_parser("fastfuncstuff.cli.allineate")
    assert parser is not None
    script = render_zsh("ffs_allineate", describe(parser))
    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_zsh_escapes_the_characters_that_end_a_spec():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cost", choices=["lpa", "lpc"], help="[BETA] pick: don't guess")
    script = render_zsh("ffs_demo", describe(parser))
    line = next(ln for ln in script.splitlines() if "-cost" in ln)
    assert r"\[BETA\]" in line
    assert r"pick\:" in line
    assert "(lpa lpc)" in line


@pytest.mark.skipif(not shutil.which("zsh"), reason="zsh not available")
def test_generated_zsh_loads_and_completes(tmp_path):
    parser = load_parser("fastfuncstuff.cli.deconvolve")
    assert parser is not None
    (tmp_path / "_ffs_deconvolve").write_text(render_zsh("ffs_deconvolve", describe(parser)))
    script = (
        f"fpath=({tmp_path} $fpath); autoload -Uz compinit; compinit -u; "
        "autoload -Uz _ffs_deconvolve; print OK"
    )
    proc = subprocess.run(["zsh", "-f", "-c", script], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_hinted_values_are_offered():
    """suggest() covers the flags choices= is too strict for."""
    parser = argparse.ArgumentParser()
    action = parser.add_argument("-work_dxyz", metavar="auto|off|MM", default="auto")
    suggest(action, ("auto", "off"))
    spec = next(s for s in describe(parser) if "-work_dxyz" in s.option_strings)
    assert spec.completes == "choices"
    assert spec.choices == ["auto", "off"]


def test_a_hint_outranks_the_metavar_guess():
    """The author's hint beats every heuristic, including type= and metavar."""
    parser = argparse.ArgumentParser()
    suggest(parser.add_argument("-polort", type=int, metavar="N"), (0, 3))
    spec = next(s for s in describe(parser) if "-polort" in s.option_strings)
    assert spec.completes == "choices"  # not "none", which type=int would give
    assert spec.choices == ["0", "3"]


def test_device_still_completes_for_hand_rolled_flags():
    """32 tools register -device themselves instead of via add_device_arg.

    The dest fallback is what keeps those completing until they are migrated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", default="auto", help="cuda | cpu | mps")
    spec = next(s for s in describe(parser) if "-device" in s.option_strings)
    assert spec.completes == "device"
