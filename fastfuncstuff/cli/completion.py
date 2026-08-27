#!/usr/bin/env python3
"""ffs_completion — generate static shell completions for the ffs_* toolbox.

Run once (after install, or after adding a flag); the generated scripts cost
nothing at TAB time. See :mod:`fastfuncstuff.completion` for why the parsers are
introspected offline rather than completed dynamically.

    ffs_completion -shell fish -o ~/.config/fish/completions
    ffs_completion -shell bash -o ~/.local/share/bash-completion/completions
    ffs_completion -shell bash -tool ffs_deconvolve      # to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.completion import RENDERERS, describe, load_parser


def _entry_points() -> dict[str, str]:
    """Map ``ffs_*`` console-script name -> CLI module path.

    Read from the installed distribution metadata so the list cannot drift from
    ``pyproject.toml``; falls back to parsing the file when running from a
    source tree that was never installed.
    """
    try:
        from importlib.metadata import distribution

        eps = distribution("fastfuncstuff").entry_points
        found = {ep.name: ep.value.split(":")[0] for ep in eps if ep.group == "console_scripts"}
        if found:
            return found
    except Exception:
        pass

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return {}
    import tomllib

    data = tomllib.loads(pyproject.read_text())
    scripts = data.get("project", {}).get("scripts", {})
    return {name: target.split(":")[0] for name, target in scripts.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_completion",
        description="Generate static bash/fish completions for the ffs_* tools.",
        formatter_class=FfsHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-shell",
        choices=sorted(RENDERERS),
        required=True,
        help="Which shell to emit completions for.",
    )
    parser.add_argument(
        "-o",
        "-outdir",
        dest="outdir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to write one completion file per tool. Omit to print to "
            "stdout (use with -tool). fish wants ~/.config/fish/completions; "
            "bash wants ~/.local/share/bash-completion/completions."
        ),
    )
    parser.add_argument(
        "-tool",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Limit to these console-script names (default: every ffs_* tool).",
    )
    parser.add_argument(
        "-verb",
        type=int,
        default=1,
        metavar="LEVEL",
        help="0=silent, 1=normal.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    entry_points = _entry_points()
    if not entry_points:
        print("ERROR: could not enumerate ffs_* entry points.", file=sys.stderr)
        return 1

    names = sorted(args.tool) if args.tool else sorted(entry_points)
    unknown = [n for n in names if n not in entry_points]
    if unknown:
        print(f"ERROR: unknown tool(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    render = RENDERERS[args.shell]
    outdir = Path(args.outdir).expanduser() if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, []
    for name in names:
        parser = load_parser(entry_points[name])
        if parser is None:
            skipped.append(name)
            continue
        text = render(name, describe(parser))
        if outdir is None:
            sys.stdout.write(text)
        else:
            # fish autoloads <cmd>.fish; bash-completion autoloads <cmd>.
            path = outdir / (f"{name}.fish" if args.shell == "fish" else name)
            path.write_text(text)
            if args.verb >= 1:
                print(f"  {path}")
        written += 1

    if args.verb >= 1 and outdir is not None:
        print(f"\nWrote {written} completion file(s) to {outdir}")
        if args.shell == "fish":
            print("fish picks these up automatically on the next prompt.")
        else:
            print("bash: ensure bash-completion is enabled, then start a new shell.")
    if skipped:
        print(f"WARNING: no parser recovered for: {', '.join(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
