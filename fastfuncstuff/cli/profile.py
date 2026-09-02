"""Profile one FFS console command from the current working directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.benchmark.profile_cli import _load_console_script
from fastfuncstuff.benchmark.profiling import (
    BenchmarkProfiler,
    aggregate_stage,
    capture_profile,
    command_argv,
)
from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_profile",
        description=(
            "Profile one FFS command. The command runs in-process so the report "
            "includes its Python call stack and PyTorch operations."
        ),
        formatter_class=FfsHelpFormatter,
    )
    parser.add_argument(
        "-trace",
        action="store_true",
        help="Collect the full PyTorch trace instead of the bounded compact sample.",
    )
    parser.add_argument(
        "-output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Profile output parent directory. Default: ./profiles.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="FFS command after --")
    args = parser.parse_args(argv)
    args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.command:
        parser.error("a command is required after --")
    if command_argv(args.command) is None:
        parser.error("command must be a directly runnable ffs_* CLI (not a shell pipeline)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = [str(part) for part in args.command]
    executable = Path(command[0]).name
    parent = args.output_dir or (Path.cwd() / "profiles")
    benchmark = BenchmarkProfiler.create(
        parent.parent if parent.name == "profiles" else parent, trace=args.trace
    )
    output_dir = benchmark.root
    output_path = output_dir / "invocation.json"
    console_main = _load_console_script(executable)

    def invoke() -> object:
        previous = sys.argv
        sys.argv = command
        try:
            return console_main()
        finally:
            sys.argv = previous

    try:
        result = capture_profile(
            invoke,
            output_path,
            command=command,
            label=executable,
            stage=executable,
            trace=args.trace,
        )
    except SystemExit as exc:
        result = int(exc.code or 0)
    summary = aggregate_stage(output_dir)
    print(f"Profile written to {output_dir}")
    print(f"  Tool time: {summary['tool_seconds']:.3f}s")
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
