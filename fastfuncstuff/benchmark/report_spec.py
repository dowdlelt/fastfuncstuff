"""What a check promotes into the benchmark report: its headline metrics and tool names.

Kept free of plotting imports, since every check imports it. The report draws everything
else (sections, captions, hosts, architectures, code versions) from the worklist itself,
so new units, datasets and machines need no change here; a metric left out of a spec still
appears, under the check's "all metrics" table.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    """One headline metric. ``{tool}`` in ``key`` expands over the spec's tools; a key
    without it compares two tools already (ffs against the reference) and is drawn per
    dataset."""

    key: str
    label: str
    better: str = "lower"  # lower | higher | none
    line: float | None = None  # a pass threshold or a meaningful level, drawn dashed
    line_label: str = ""
    kind: str = "value"  # value | rate (a 0/1 per unit, drawn as the share of units)
    log: bool = False
    note: str = ""  # one sentence for the caption: what the metric measures


@dataclass(frozen=True)
class ReportSpec:
    summary: str  # one sentence: what is compared with what
    tools: dict[str, str]  # metric suffix -> display name, in plot order
    timing: dict[str, str]  # timing side -> display name, in plot order
    references: tuple[str, ...] = ()  # metric suffixes / timing sides that are references
    metrics: tuple[Metric, ...] = ()
    # (ffs sides, reference sides): '+' sums sides, for a chain against one reference tool
    speedup: tuple[tuple[str, str], ...] = ()
    difficulty: tuple[str, str] | None = None  # (metric key, axis label)
    # Entry modules of the tools under test (CLI + check): a unit is drawn hollow when a file
    # they import changed between its run's commit and the newest run's.
    code: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)  # free notes for the section
