"""Convert FSL 3-column onset files into BIDS ``*_events.tsv``, one per run.

Written for ds004539 (Dresbach et al. 2023, event-related VASO), whose
derivatives ship FSL files at ``derivatives/sub-*/ses-*/events/``::

    sub-06_ses-001_task-eventStim_run-001.txt                  one condition
    sub-12_ses-001_task-eventStimRandom_run-001_visual.txt     one file per condition
    sub-06_ses-001_task-eventStim_run-001_longITI.txt          subset of the run file

Naming rules:
- a file without a trailing label is the run's only condition, ``trial_type``
  given by ``-single_label`` (default ``stimulus``);
- a trailing label names a condition, and a run's condition files are merged
  into one events.tsv;
- labels in ``-skip`` (default ``longITI shortITI``) are re-cuts of the run
  file, not conditions, and are ignored.

The onsets are copied unchanged. In ds004539 the stimulus log's clock resets
to 0 at the first scanner trigger, so they are relative to the first acquired
volume, as BIDS requires. A third column other than 1 is kept as a ``modulation``
column rather than dropped.

Usage:
    python scripts/fsl3col_to_bids_events.py DERIVATIVES_DIR OUT_DIR [-subjects sub-06 ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(
    r"^(?P<run>sub-[^_]+_ses-[^_]+_task-[^_]+_run-[^_]+)(?:_(?P<label>[^_.]+))?\.txt$"
)


def read_fsl3col(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"{path}:{i}: expected 3 columns, got {len(parts)}")
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return rows


def collect(deriv: Path, subjects: list[str] | None, skip: set[str], single_label: str):
    """{run stem: [(onset, duration, magnitude, trial_type), ...]}"""
    runs: dict[str, list[tuple[float, float, float, str]]] = defaultdict(list)
    kinds: dict[str, set[str]] = defaultdict(set)  # run -> {"single", "multi"}
    for path in sorted(deriv.glob("sub-*/ses-*/events/*.txt")):
        m = NAME.match(path.name)
        if m is None:
            print(f"  skip (name not understood): {path}", file=sys.stderr)
            continue
        run, label = m["run"], m["label"]
        if subjects and run.split("_")[0] not in subjects:
            continue
        if label in skip:
            continue
        kinds[run].add("multi" if label else "single")
        trial_type = label or single_label
        runs[run].extend((o, d, a, trial_type) for o, d, a in read_fsl3col(path))
    for run, k in kinds.items():
        if len(k) > 1:
            raise ValueError(f"{run}: both an unlabelled file and condition files")
    return runs


def write_events(path: Path, rows: list[tuple[float, float, float, str]]) -> None:
    rows = sorted(rows, key=lambda r: (r[0], r[3]))
    modulated = any(abs(a - 1.0) > 1e-9 for _, _, a, _ in rows)
    header = ["onset", "duration", "trial_type"] + (["modulation"] if modulated else [])
    lines = ["\t".join(header)]
    for o, d, a, tt in rows:
        cells = [f"{o:.3f}", f"{d:.3f}", tt] + ([f"{a:g}"] if modulated else [])
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("deriv", type=Path, help="derivatives dir holding sub-*/ses-*/events/")
    ap.add_argument("out", type=Path, help="output dir for *_events.tsv (flat)")
    ap.add_argument("-subjects", nargs="+", default=None, help="e.g. sub-06 sub-12")
    ap.add_argument("-skip", nargs="+", default=["longITI", "shortITI"])
    ap.add_argument("-single_label", default="stimulus")
    args = ap.parse_args(argv)

    runs = collect(args.deriv, args.subjects, set(args.skip), args.single_label)
    if not runs:
        sys.exit(f"no FSL event files found under {args.deriv}/sub-*/ses-*/events/")
    args.out.mkdir(parents=True, exist_ok=True)
    for run, rows in sorted(runs.items()):
        out = args.out / f"{run}_events.tsv"
        write_events(out, rows)
        types = sorted({r[3] for r in rows})
        print(f"  {out.name}: {len(rows)} events ({', '.join(types)})")


if __name__ == "__main__":
    main()
