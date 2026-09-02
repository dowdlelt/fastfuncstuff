"""Execute an installed FFS console script in-process under the profiler."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path

from .profiling import capture_profile


def _load_console_script(name: str) -> Callable[[], object]:
    matches = entry_points(group="console_scripts", name=name)
    if not matches:
        raise RuntimeError(f"FFS console script is not installed: {name}")
    return next(iter(matches)).load()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    def invoke() -> object:
        console_main = _load_console_script(Path(command[0]).name)
        previous = sys.argv
        sys.argv = command
        try:
            return console_main()
        finally:
            sys.argv = previous

    try:
        result = capture_profile(
            invoke,
            args.output,
            command=command,
            label=args.label,
            stage=args.stage,
            trace=args.trace,
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
