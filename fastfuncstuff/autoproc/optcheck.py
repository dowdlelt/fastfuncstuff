"""Validate ``-<stage>_opts`` strings against the target tool's own parser.

A typo in an override (``-nodric``) otherwise surfaces two hours in, when the
generated script finally reaches that stage and the tool exits on an unknown
argument. The tools' parsers are the only authority on what is spelled right, so
we import the CLI module and ask it, rather than maintaining a second list.

Only flag *names* are checked — values, arity and choices are the tool's job.
Nothing is imported unless the user actually passed an override, so the common
path pays nothing (importing a CLI module pulls in torch: ~3 s once).
"""

from __future__ import annotations

import argparse
import difflib
import functools
import importlib
import re

# Stage key → the cli module whose parser owns that stage's flags. "unwrap" is
# ROMEO (external, not ours) and has no parser to ask.
STAGE_TOOL: dict[str, str] = {
    "nordic": "nordic",
    "tshift": "slicetime",
    "moco": "moco",
    "locomoco": "locomoco",
    "blip": "blipflip",
    "b0fmap": "util_b0fmap",
    "xrun": "allineate",
    "xrun_nl": "formwarp",
    "xfmap": "allineate",
    "xfmap_nl": "formwarp",
    "xses": "allineate",
    "xses_nl": "formwarp",
    "anat": "allineate",
    "segment": "segment",
    "segment_fstpm": "segment",
    "nwarp": "nwarp",
    "glm": "reml",
}

_PARSER_FACTORIES = ("build_parser", "create_parser", "make_parser", "get_parser")
# A value may legitimately start with '-': negative numbers, ranges like
# "-0.9,0.9,65". Those are skipped; everything else that starts with '-' is
# treated as a flag (including AFNI-style digit-leading ones: -1Dfile, -1Dmatrix_save).
_NUMERIC = re.compile(r"^[0-9.,eE+-]+$")
_FLAGLIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _is_flag(token: str) -> bool:
    body = token[1:]
    if not token.startswith("-") or not body or _NUMERIC.match(body):
        return False
    return bool(_FLAGLIKE.match(body))


class _CapturedParserError(Exception):
    """Carries the parser out of the module's own parse_args()."""

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


@functools.cache
def _grab_parser(module: str) -> argparse.ArgumentParser | None:
    """The parser for ``fastfuncstuff.cli.<module>``, or None if it can't be had.

    Most CLIs build their parser inside ``parse_args()`` rather than exposing a
    factory, so when there is no factory we call that entry point with a spy in
    place of ``ArgumentParser.parse_args`` and catch the parser on its way in."""
    try:
        mod = importlib.import_module(f"fastfuncstuff.cli.{module}")
    except Exception:
        return None
    for name in _PARSER_FACTORIES:
        factory = getattr(mod, name, None)
        if callable(factory):
            try:
                return factory()
            except Exception:
                return None

    entry = getattr(mod, "parse_args", None) or getattr(mod, "main", None)
    if entry is None:
        return None
    real = argparse.ArgumentParser.parse_args

    def spy(self, *_a, **_k):
        raise _CapturedParserError(self)

    argparse.ArgumentParser.parse_args = spy
    try:
        entry([])
    except _CapturedParserError as c:
        return c.parser
    except TypeError:  # main() with no argv parameter
        try:
            entry()
        except _CapturedParserError as c:
            return c.parser
        except Exception:
            return None
    except Exception:
        return None
    finally:
        argparse.ArgumentParser.parse_args = real
    return None


def check_opts(key: str, opts: str) -> list[str]:
    """Errors for flags in ``opts`` that the stage's tool does not accept.

    Empty when everything is spelled right, the stage has no ffs parser (ROMEO),
    or the parser could not be obtained — a validator that cannot validate must
    not block script generation."""
    module = STAGE_TOOL.get(key)
    if module is None or not opts.strip():
        return []
    parser = _grab_parser(module)
    if parser is None:
        return []
    known = set(parser._option_string_actions)
    errors = []
    for token in opts.split():
        if not _is_flag(token) or token in known:
            continue
        near = difflib.get_close_matches(token, known, n=3, cutoff=0.6)
        hint = f" — did you mean {', '.join(near)}?" if near else ""
        errors.append(f"-{key}_opts: ffs_{module} has no flag '{token}'{hint}")
    return errors
