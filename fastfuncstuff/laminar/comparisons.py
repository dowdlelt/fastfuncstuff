"""Which condition pairs ``ffs_laminascope`` fits: a small TOML of comparisons.

The laminar model compares TWO conditions -- a baseline and a test whose
difference the modulatory input marks -- and every comparison is a whole model
space over every column. A design with several tasks and levels has far more
pairs than are worth fitting, so the pairs are named, not enumerated::

    [[comparison]]
    name = "gender_vs_fixate"
    baseline = "task-fixate_*"      # an iresp label, a glob, or a list of either
    test = "task-gender_*"

    [[comparison]]
    name = "hi_vs_lo"
    baseline = "*lo"
    test = "*hi"
    models = ["null", "superficial", "middle", "deep"]   # optional subset

Each side's labels are averaged with equal weight. That is exact, because the
curves come from a linear estimator: the average of the curves is the curve
of the averaged condition. The modulation B is signed and P(depth modulated)
ignores the sign, so a comparison is non-directional however its sides are
ordered. Only the baseline/test roles in the model's input differ.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass


@dataclass
class Comparison:
    name: str
    baseline: list[str]  # resolved iresp labels
    test: list[str]
    models: list[str] | None = None


def _patterns(value, where: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"{where}: give a label, a glob, or a list of them")


def resolve(patterns: list[str], labels: list[str], where: str) -> list[str]:
    """The labels ``patterns`` name, in ``labels`` order; every pattern must match."""
    out: list[str] = []
    for pat in patterns:
        hit = [lab for lab in labels if fnmatch.fnmatchcase(lab, pat)]
        if not hit:
            raise ValueError(f"{where}: {pat!r} matches no condition (have {', '.join(labels)})")
        out += [h for h in hit if h not in out]
    return [lab for lab in labels if lab in out]


def read_comparisons(path: str, labels: list[str]) -> list[Comparison]:
    """Comparisons from a TOML file, sides resolved against the available ``labels``."""
    import tomllib

    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    entries = doc.get("comparison")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: no [[comparison]] tables")
    out = []
    for i, e in enumerate(entries):
        where = f"{path} comparison {i + 1}"
        unknown = set(e) - {"name", "baseline", "test", "models"}
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
        for key in ("name", "baseline", "test"):
            if key not in e:
                raise ValueError(f"{where}: needs '{key}'")
        out.append(
            make_comparison(e["name"], e["baseline"], e["test"], labels, e.get("models"), where)
        )
    names = [c.name for c in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: comparison names repeat")
    return out


def make_comparison(name, baseline, test, labels, models=None, where="-compare") -> Comparison:
    if not isinstance(name, str) or not name or any(c in name for c in "/ \t"):
        raise ValueError(f"{where}: name must be a single word, got {name!r}")
    base = resolve(_patterns(baseline, f"{where} baseline"), labels, f"{where} baseline")
    tst = resolve(_patterns(test, f"{where} test"), labels, f"{where} test")
    if set(base) & set(tst):
        raise ValueError(f"{where}: {sorted(set(base) & set(tst))} on both sides")
    if models is not None:
        models = _patterns(models, f"{where} models")
    return Comparison(name, base, tst, models)


__all__ = ["Comparison", "make_comparison", "read_comparisons", "resolve"]
