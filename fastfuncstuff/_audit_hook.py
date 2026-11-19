"""Audit any ``ffs_*`` process, not only the ones ``ffs_benchmark`` launches.

``FFS_AUDIT=idle`` (or ``ops``) makes every ffs CLI started with it in its
environment record the matching :mod:`fastfuncstuff.benchmark.audit` report
from package import to interpreter exit, and write it to::

    $FFS_AUDIT_DIR/<mode>/<tool>/invocations/<stamp>-<pid>.json

(``FFS_AUDIT_DIR`` defaults to ``~/.cache/fastfuncstuff/audits``). The layout
is the one ``ffs_benchmark -audit`` writes, with the tool name as the stage, so
``python -m fastfuncstuff.benchmark.audit <dir>/<mode>`` summarises it and
``scripts/replay_op_census.py <dir>/ops`` replays it. Because the variable is
inherited, one ``FFS_AUDIT=idle ffs_autoproc ...`` audits every tool autoproc
runs as a subprocess, each under its own name.

Nothing happens -- and nothing heavy is imported -- unless the variable is set
and the process is an ``ffs_*`` console script.
"""

from __future__ import annotations

import os
import sys


def maybe_install() -> None:
    mode = os.environ.get("FFS_AUDIT", "").strip().lower()
    if not mode:
        return
    tool = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if not tool.startswith("ffs_"):
        return
    try:
        _install(mode, tool)
    except Exception as exc:  # an audit must never take the tool down with it
        print(f"FFS_AUDIT: not recording ({type(exc).__name__}: {exc})", file=sys.stderr)


def _install(mode: str, tool: str) -> None:
    import atexit
    import time
    from datetime import datetime
    from pathlib import Path

    from fastfuncstuff.benchmark import audit

    if mode not in audit.MODES:
        raise ValueError(f"FFS_AUDIT must be one of {audit.MODES}, got {mode!r}")
    base = Path(
        os.environ.get("FFS_AUDIT_DIR") or Path.home() / ".cache" / "fastfuncstuff" / "audits"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = base / mode / tool / "invocations" / f"{stamp}-{os.getpid()}.json"
    session = audit.AuditSession(mode, command=list(sys.argv), label=tool, stage=tool)
    session.start()
    started = time.monotonic()

    def finish() -> None:
        # -help, usage errors and header reads are not workloads; a report for
        # each would bury the real ones.
        if time.monotonic() - started < 2:
            return
        session.stop(output, error=None)
        print(f"FFS_AUDIT: wrote {output}", file=sys.stderr)

    atexit.register(finish)
