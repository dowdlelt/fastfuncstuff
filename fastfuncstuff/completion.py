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
import re
import sys
import warnings
from dataclasses import dataclass, field

# Metavars that name a path. The ffs tools take NIfTI (.nii/.nii.gz/.nii.zst),
# AFNI (.HEAD/.BRIK), .1D regressors and BIDS .tsv events, so the completion
# stays unfiltered rather than guessing an extension list and hiding the file
# the user actually wants.
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
    "MAP",
    "NPZ",
    "SOURCE",
    "SPEC",
    "WARP",
    "XMAT",
}
# Suffixes that make a metavar self-evidently a filename -- MASK.nii.gz,
# MOCO.aff12.1D, FILE.1D, WARP.nii.gz. Authors write these instead of a bare
# FILE precisely because the extension is the useful part; the completion
# should read them rather than treat them as unrecognised.
_FILE_METAVAR_SUFFIXES = (
    ".1D",
    ".NII",
    ".NII.GZ",
    ".NII.ZST",
    ".HEAD",
    ".BRIK",
    ".H5",
    ".JSON",
    ".TSV",
    ".TXT",
    ".NPZ",
    ".TOML",
    ".YAML",
    ".PNG",
    ".CSV",
)
_DIR_METAVARS = {"DIR", "DIRECTORY", "DATA_DIR", "OUT_DIR"}
# -device is registered once, by cli_utils.add_device_arg, so one word list
# covers the whole toolbox. The "cuda,N" / "cpu,N" spec forms are free-form
# and cannot be enumerated, so they are offered as bare prefixes.
_DEVICE_WORDS = ("auto", "cpu", "cuda", "mps")


def _display_strings(option_strings: list[str]) -> list[str]:
    """One spelling per NAME, keeping the order argparse was given.

    Every ffs flag accepts both ``-foo-bar`` and ``-foo_bar``, and several
    carry a renamed predecessor as well.  Offering all of them doubles the
    completion list (measured on ffs_fitbasis: 71 flags, 132 spellings)
    without adding a single new thing the user can do.

    Deduping on the hyphen-normalised key rather than "drop anything with an
    underscore" matters: flags like ``-drop_first`` are *documented* with the
    underscore and only alias to ``-drop-first``, so a blanket rule would
    hide the primary name and surface the alias.  Keeping the first spelling
    of each distinct name keeps whatever argparse was told is primary, and
    still lists genuinely different names (``-drop_first`` and
    ``-skip_first``) separately.
    """
    seen: set[str] = set()
    out: list[str] = []
    for opt in option_strings:
        key = opt.replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        out.append(opt)
    return out


@dataclass
class OptionSpec:
    """One flag, reduced to what a shell needs to complete it."""

    option_strings: list[str]
    """Every accepted spelling — what a shell must MATCH against."""
    help: str = ""
    choices: list[str] = field(default_factory=list)
    takes_value: bool = False
    completes: str = "none"  # "none" | "file" | "dir" | "choices" | "device"

    @property
    def display_strings(self) -> list[str]:
        """The spellings a shell should OFFER.  See :func:`_display_strings`."""
        return _display_strings(self.option_strings)


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


def _metavar_names(action: argparse.Action) -> list[str]:
    """Every token in an action's metavar, upper-cased.

    Tuple metavars (``-stim TIMING_FILE HRF_MODEL LABEL``) and alternation
    metavars (``-adjust_dof MAP|N``) both describe more than one thing; a shell
    completing after the flag cannot tell which slot it is filling, so any
    path-shaped token is enough to offer filenames.
    """
    metavar = action.metavar
    if isinstance(metavar, str):
        raw = [metavar]
    elif isinstance(metavar, tuple):
        raw = [str(m) for m in metavar]
    else:
        return []
    return [tok.upper() for part in raw for tok in re.split(r"[|\s]+", part) if tok]


def _looks_like_path(name: str) -> bool:
    return name in _FILE_METAVARS or "FILE" in name or name.endswith(_FILE_METAVAR_SUFFIXES)


def _completion_kind(action: argparse.Action) -> str:
    """What a shell should offer after this flag.

    The default is **file** completion, not silence.  This toolbox is mostly
    paths in and paths out: of the flags that take a value, the ones that are
    provably not paths are provably so (``type=int``/``float``, or an explicit
    metavar the author chose to say VALUE / SECONDS / LABEL).  Everything left
    over -- ``-out``, ``-Rbeta``, ``-dfile``, ``-master`` -- is a path far more
    often than not, and completing nothing there is indistinguishable from a
    broken completion script (bug of record: ``ffs_tunewarp -out ./so<TAB>``
    just beeped).

    So an *unrecognised* argument falls back to files, which is also what bash
    does with no completion installed at all.  To suppress that for a genuinely
    free-form flag, give it ``type=`` or a metavar -- both are worth having in
    ``-help`` anyway.
    """
    if action.choices:
        return "choices"

    dest = (action.dest or "").lower()
    if dest == "device":
        # Every tool takes -device through the same parser; the spec forms
        # (cuda,0 / cpu,8) are documented in cli_utils.parse_device_arg.
        return "device"

    # A numeric flag never wants a filename, whatever its metavar says
    # (-round_onsets THRESHOLD is a float, not a path).
    if action.type in (int, float):
        return "none"

    names = _metavar_names(action)
    if names:
        if any(n in _DIR_METAVARS for n in names):
            return "dir"
        if any(_looks_like_path(n) for n in names):
            return "file"
        # An explicit metavar the author chose (THRESHOLD, VALUE, SECONDS)
        # that names nothing path-shaped is a statement that the argument is
        # not a path, and outranks the file default.
        return "none"

    return "file"


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
    all_flags = " ".join(opt for s in specs for opt in s.display_strings)

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
        # NB: the case patterns and the file/dir lists below deliberately use
        # every spelling, not just the offered one -- what to complete AFTER a
        # flag has to work for whichever alias the user actually typed.
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
        # Only the offered spellings: fish has no hidden completion, so
        # registering an alias is the same act as suggesting it.  Typing an
        # unregistered alias in full still works, it just falls back to fish's
        # default argument completion.
        for option in spec.display_strings:
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
