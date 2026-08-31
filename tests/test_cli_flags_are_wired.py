"""Flags that were parsed, stored, and then never read.

argparse makes a dead flag completely silent: it appears in -help, it accepts
its value, and the tool proceeds as if the default had been given.  Four were
found this way -- ffs_xval_r2 -R2method and -data_chunk_size, ffs_pathfinder
-cv_metric, ffs_tps -cv-method -- so the check is now mechanical: every
declared option name has to appear somewhere in the module body that isn't its
own declaration.

It is a coarse check on purpose.  A name that appears only in the parser is
certainly dead; one that appears elsewhere might still be misused, which is
what the per-tool tests are for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[1] / "fastfuncstuff" / "cli"

TOOLS = ["xval_r2", "tps", "pathfinder"]

# Flags whose whole job is to be inspected by the help/completion machinery,
# or that argparse itself consumes.
EXEMPT = {"help", "version"}


def _declared_dests(tree: ast.AST) -> dict[str, int]:
    """{dest: lineno} for every add_argument call in the module."""
    dests: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        dest = None
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                dest = kw.value.value
        if dest is None:
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            long = [f for f in flags if isinstance(f, str) and f.startswith("-")]
            if not long:
                continue
            dest = max(long, key=len).lstrip("-").replace("-", "_")
        if dest not in EXEMPT:
            dests.setdefault(dest, node.lineno)
    return dests


@pytest.mark.parametrize("tool", TOOLS)
def test_every_declared_flag_is_read_somewhere(tool):
    source = (CLI_DIR / f"{tool}.py").read_text()
    tree = ast.parse(source)
    declared = _declared_dests(tree)

    # Attribute reads off any object: args.foo, opts.foo, ...
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }
    # ... plus getattr(args, "foo") and any bare mention as a keyword name.
    read |= {
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg
    }

    dead = sorted(f"-{d} (line {ln})" for d, ln in declared.items() if d not in read)
    assert not dead, f"{tool}.py declares flags nothing reads: {', '.join(dead)}"
