"""Static shell-completion generation for the ``ffs_*`` CLIs.

**Why static.** A dynamic completer (argcomplete) re-enters the tool on every
TAB press. Importing any ``ffs_*`` CLI costs ~2.4 s even after the lazy-import
work, because the parsers are built at module scope and torch comes along for
the ride. Two and a half seconds per keystroke is not a completion system. So
the parsers are introspected **once**, offline, and rendered to plain shell
scripts that cost nothing at TAB time.

**Getting at the parsers.** Only about half the CLIs expose a function that
returns an ``ArgumentParser``; the rest name it ``parse_args`` and actually
parse inside it, returning a ``Namespace`` (or exiting, when required arguments
are missing). Rather than refactor sixty entry points, :func:`load_parser`
temporarily replaces ``ArgumentParser.parse_args`` with a hook that raises the
parser back out. The parser is fully built by the time anything calls
``parse_args`` on it, so what we catch is complete.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import sys
import warnings
from dataclasses import dataclass, field

# Metavars and dest names that should complete to filenames. The ffs tools take
# NIfTI (.nii/.nii.gz/.nii.zst), AFNI (.HEAD/.BRIK), .1D regressors and BIDS
# .tsv events, so the completion stays unfiltered rather than guessing an
# extension list and hiding the file the user actually wants.
_FILE_METAVARS = {
    "FILE",
    "TSV",
    "PATH",
    "YAML",
    "OUTPUT",
    "PREFIX",
    "PATTERN",
    "DSET",
    "IMAGE",
    "MASK",
    "CACHE_JSON",
    "FILE.H5",
    "LABELS",
    "MATRIX",
    "JSON",
}
_FILE_DEST_HINTS = (
    "input",
    "mask",
    "events",
    "onsets",
    "prefix",
    "ortvec",
    "matrix",
    "base",
    "source",
    "target",
    "anat",
    "warp",
    "cache",
    "config",
    "atlas",
    "weight",
)
_DIR_METAVARS = {"DIR", "DIRECTORY", "DATA_DIR", "OUT_DIR"}
# -device is registered once, by cli_utils.add_device_arg, so one word list
# covers the whole toolbox. The "cuda,N" / "cpu,N" spec forms are free-form
# and cannot be enumerated, so they are offered as bare prefixes.
_DEVICE_WORDS = ("auto", "cpu", "cuda", "mps")


@dataclass
class OptionSpec:
    """One flag, reduced to what a shell needs to complete it."""

    option_strings: list[str]
    help: str = ""
    choices: list[str] = field(default_factory=list)
    takes_value: bool = False
    completes: str = "none"  # "none" | "file" | "dir" | "choices" | "device"


class _ParserGrabbedError(Exception):
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


def load_parser(module_name: str) -> argparse.ArgumentParser | None:
    """Build (but never run) a CLI module's parser.

    ``module_name`` is the dotted path, e.g. ``fastfuncstuff.cli.deconvolve``.
    Returns ``None`` when no parser could be recovered.
    """
    original_parse = argparse.ArgumentParser.parse_args
    original_parse_known = argparse.ArgumentParser.parse_known_args
    original_argv = sys.argv

    def _hook(self, *_args, **_kwargs):
        raise _ParserGrabbedError(self)

    argparse.ArgumentParser.parse_args = _hook  # type: ignore[method-assign]
    argparse.ArgumentParser.parse_known_args = _hook  # type: ignore[method-assign]
    # A bare argv makes the "no arguments -> print help and exit" branch that a
    # few tools have fire before they ever reach parse_args; give them one.
    sys.argv = [module_name.rsplit(".", 1)[-1], "-h"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                module = importlib.import_module(module_name)
            except Exception:
                return None

            for name in (
                "parse_args",
                "create_parser",
                "build_parser",
                "_build_parser",
                "get_parser",
                "make_parser",
                "main",
            ):
                fn = getattr(module, name, None)
                if not callable(fn):
                    continue
                try:
                    signature = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                if any(
                    p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
                    for p in signature.parameters.values()
                ):
                    continue
                try:
                    # Tools print banners and help at parser-build time; swallow it.
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        result = fn()
                except _ParserGrabbedError as grabbed:
                    return grabbed.parser
                except (SystemExit, Exception):
                    continue
                if isinstance(result, argparse.ArgumentParser):
                    return result
            return None
    finally:
        argparse.ArgumentParser.parse_args = original_parse  # type: ignore[method-assign]
        argparse.ArgumentParser.parse_known_args = original_parse_known  # type: ignore[method-assign]
        sys.argv = original_argv


def _completion_kind(action: argparse.Action) -> str:
    if action.choices:
        return "choices"

    dest = (action.dest or "").lower()
    if dest == "device":
        # Every tool takes -device through the same parser; the spec forms
        # (cuda,0 / cpu,8) are documented in cli_utils.parse_device_arg.
        return "device"

    # A numeric flag never wants a filename, whatever its dest is called
    # (-round_onsets THRESHOLD would otherwise match the "onsets" hint below).
    if action.type in (int, float):
        return "none"

    names: list[str] = []
    metavar = action.metavar
    if isinstance(metavar, str):
        names.append(metavar.upper())
    elif isinstance(metavar, tuple):
        names.extend(str(m).upper() for m in metavar)

    if any(n in _DIR_METAVARS for n in names):
        return "dir"
    if any(n in _FILE_METAVARS for n in names):
        return "file"
    # The dest-name fallback only applies when there is no metavar to go on.
    # An explicit metavar the author chose (THRESHOLD, VALUE, SECONDS) is a
    # statement that the argument is not a path, and outranks the guess.
    if not names and any(hint in dest for hint in _FILE_DEST_HINTS):
        return "file"
    return "none"


def describe(parser: argparse.ArgumentParser) -> list[OptionSpec]:
    """Flatten a parser into :class:`OptionSpec`, groups and all."""
    specs: list[OptionSpec] = []
    seen: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue  # positionals: shells fall back to file completion anyway
        key = action.option_strings[0]
        if key in seen:
            continue
        seen.add(key)
        takes_value = action.nargs != 0 and not isinstance(
            action, argparse._StoreTrueAction | argparse._StoreFalseAction | argparse._HelpAction
        )
        # argparse stores the first help line verbatim; shells want one short line.
        help_text = " ".join((action.help or "").split())
        if len(help_text) > 90:
            help_text = help_text[:87].rstrip() + "..."
        specs.append(
            OptionSpec(
                option_strings=list(action.option_strings),
                help=help_text,
                choices=[str(c) for c in (action.choices or [])],
                takes_value=bool(takes_value),
                completes=_completion_kind(action) if takes_value else "none",
            )
        )
    return specs


def _shell_quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def render_bash(prog: str, specs: list[OptionSpec]) -> str:
    """A single ``complete -F`` function for one tool."""
    func = "_" + prog.replace("-", "_")
    all_flags = " ".join(opt for s in specs for opt in s.option_strings)

    file_flags, dir_flags, opaque_flags = [], [], []
    choice_cases = []
    for spec in specs:
        if not spec.takes_value:
            continue
        pattern = "|".join(spec.option_strings)
        if spec.completes == "none":
            # A number or free text. Offering the directory listing here is
            # worse than offering nothing -- it looks like a suggestion.
            opaque_flags.extend(spec.option_strings)
        if spec.completes == "device":
            choice_cases.append(
                f'        {pattern})\n            _opts="{" ".join(_DEVICE_WORDS)}"\n            ;;'
            )
        elif spec.completes == "choices":
            choice_cases.append(
                f'        {pattern})\n            _opts="{" ".join(spec.choices)}"\n            ;;'
            )
        elif spec.completes == "file":
            file_flags.extend(spec.option_strings)
        elif spec.completes == "dir":
            dir_flags.extend(spec.option_strings)

    parts = [
        f"{func}() {{",
        "    local cur prev",
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        '    prev="${COMP_WORDS[COMP_CWORD-1]}"',
        "    local _opts=",
        "",
        '    case "$prev" in',
    ]
    parts.extend(choice_cases)
    if file_flags:
        parts.append(f"        {'|'.join(file_flags)})")
        parts.append('            COMPREPLY=( $(compgen -f -- "$cur") )')
        parts.append("            compopt -o filenames 2>/dev/null")
        parts.append("            return 0")
        parts.append("            ;;")
    if dir_flags:
        parts.append(f"        {'|'.join(dir_flags)})")
        parts.append('            COMPREPLY=( $(compgen -d -- "$cur") )')
        parts.append("            compopt -o filenames 2>/dev/null")
        parts.append("            return 0")
        parts.append("            ;;")
    if opaque_flags:
        parts.append(f"        {'|'.join(opaque_flags)})")
        parts.append("            COMPREPLY=()")
        parts.append("            return 0")
        parts.append("            ;;")
    parts.append("    esac")
    parts.append("")
    parts.append('    if [[ -n "$_opts" ]]; then')
    parts.append('        COMPREPLY=( $(compgen -W "$_opts" -- "$cur") )')
    parts.append("        return 0")
    parts.append("    fi")
    parts.append("")
    parts.append('    if [[ "$cur" == -* ]]; then')
    parts.append(f'        COMPREPLY=( $(compgen -W "{all_flags}" -- "$cur") )')
    parts.append("    else")
    parts.append('        COMPREPLY=( $(compgen -f -- "$cur") )')
    parts.append("        compopt -o filenames 2>/dev/null")
    parts.append("    fi")
    parts.append("}")
    parts.append(f"complete -F {func} {prog}")
    return "\n".join(parts) + "\n"


def _fish_flag_args(option: str) -> str:
    """Map an option string onto fish's short / old-style-long / GNU-long forms.

    The ffs tools use AFNI-style single-dash long flags (``-hrf-library``),
    which is fish's "old style long" -- ``-o``, not ``-l``. Getting this wrong
    silently completes nothing.
    """
    if option.startswith("--"):
        return f"-l {option[2:]}"
    if len(option) == 2:
        return f"-s {option[1]}"
    return f"-o {option[1:]}"


def render_fish(prog: str, specs: list[OptionSpec]) -> str:
    lines = [f"# fish completions for {prog}", f"complete -c {prog} -e", ""]
    for spec in specs:
        for option in spec.option_strings:
            bits = [f"complete -c {prog}", _fish_flag_args(option)]
            if spec.takes_value:
                bits.append("-r")
                if spec.completes == "device":
                    bits.append("-f")
                    bits.append(f'-a "{" ".join(_DEVICE_WORDS)}"')
                elif spec.completes == "choices":
                    bits.append("-f")
                    bits.append(f'-a "{" ".join(spec.choices)}"')
                elif spec.completes == "dir":
                    bits.append('-a "(__fish_complete_directories)"')
                elif spec.completes == "file":
                    bits.append("-F")
                else:
                    bits.append("-f")  # numeric / free-form: do not offer files
            else:
                bits.append("-f")
            if spec.help:
                bits.append(f"-d '{_shell_quote(spec.help)}'")
            lines.append(" ".join(bits))
    return "\n".join(lines) + "\n"


RENDERERS = {"bash": render_bash, "fish": render_fish}
