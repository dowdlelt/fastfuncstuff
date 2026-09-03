"""Every tool's -help must actually render.

argparse expands help text with ``help_string % params``, so a bare ``%`` in any
help string is a *runtime* TypeError that nothing catches until a user types
-help. Introduced exactly that way while documenting a measurement ("a 13% span
instead of 95%") -- the flag parsed fine and only -help broke.

Rendering every parser is also a cheap smoke test for the rest of the help
machinery: a bad ``choices``, a default that cannot be formatted, a metavar that
does not match its nargs all surface here too.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import fastfuncstuff.cli as cli_pkg

# Modules that are helpers rather than tools; they define no parser to render.
_NOT_TOOLS = {"__init__", "cli_help", "cli_utils", "completion"}


def _tool_modules() -> list[str]:
    return sorted(
        m.name
        for m in pkgutil.iter_modules(cli_pkg.__path__)
        if not m.ispkg and m.name not in _NOT_TOOLS and not m.name.startswith("_")
    )


def _build_parser(mod):
    """Return the module's parser, or None if it does not expose one plainly."""
    for attr in ("build_parser", "_build_parser", "make_parser", "get_parser"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                return None
    return None


@pytest.mark.parametrize("name", _tool_modules())
def test_help_renders_without_raising(name):
    mod = importlib.import_module(f"fastfuncstuff.cli.{name}")
    parser = _build_parser(mod)
    if parser is None:
        pytest.skip(f"{name} exposes no zero-argument parser factory")
    text = parser.format_help()
    assert text, f"{name} rendered an empty help"
    # A literal % that survived into the rendered text is the bug above waiting
    # to happen in a *different* argparse path (e.g. usage lines).
    assert "%(" not in text, f"{name} help leaked an unexpanded format spec"


# ---------------------------------------------------------------------------
# Static check, because most tools build their parser inside main() and the
# render test above can only reach the ones exposing a factory.
# ---------------------------------------------------------------------------

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

CLI_DIR = Path(__file__).resolve().parents[1] / "fastfuncstuff" / "cli"


def _bare_percent_positions(text: str) -> list[int]:
    """Offsets of a '%' that argparse would try to interpret as a format spec.

    ``%%`` is an escaped literal and ``%(name)s`` is argparse's own substitution;
    anything else raises TypeError inside ``help_string % params``.
    """
    bad, i = [], 0
    while i < len(text):
        if text[i] == "%":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == "%":
                i += 2
                continue
            if nxt != "(":
                bad.append(i)
        i += 1
    return bad


def _help_strings(tree: ast.AST):
    """(lineno, value) for every literal help= passed to add_argument."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    yield kw.value.lineno, kw.value.value


@pytest.mark.parametrize("path", sorted(CLI_DIR.glob("*.py")), ids=lambda p: p.stem)
def test_no_bare_percent_in_help_text(path):
    """A literal '%' in help= breaks -help at runtime and nothing else catches it."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        (lineno, txt[max(0, pos - 30) : pos + 20])
        for lineno, txt in _help_strings(tree)
        for pos in _bare_percent_positions(txt)
    ]
    assert not offenders, f"{path.name}: bare '%' in help text (write '%%'):\n" + "\n".join(
        f"  line {ln}: ...{frag}..." for ln, frag in offenders
    )
