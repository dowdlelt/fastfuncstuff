"""Execute an installed FFS console script in-process under a speed audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import MODES, capture_audit
from .profile_cli import _load_console_script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    console_main = _load_console_script(Path(command[0]).name)

    def invoke() -> object:
        previous = sys.argv
        sys.argv = command
        try:
            return console_main()
        finally:
            sys.argv = previous

    try:
        result = capture_audit(
            invoke,
            args.output,
            mode=args.mode,
            command=command,
            label=args.label,
            stage=args.stage,
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
