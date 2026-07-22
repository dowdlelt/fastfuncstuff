"""ffs_autoproc: scan a BIDS tree and emit a readable, resumable ffs pipeline
script (the ffs analogue of afni_proc.py). See ``cli/autoproc.py`` for the CLI.
"""

from __future__ import annotations

from fastfuncstuff.autoproc.bids import Subject, scan_subject
from fastfuncstuff.autoproc.plan import Options, Plan, build_plan

__all__ = ["Options", "Plan", "Subject", "build_plan", "scan_subject"]
