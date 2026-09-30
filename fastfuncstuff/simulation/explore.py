"""Explore a design space: what do all plausible designs of this experiment look like?

Ranges and choices are written inside the ordinary ffs_simulate flags:

    -isi "exp:[3-8],[1-3],[8-16]"      a range per number
    -null 0.25 "[0-40%]"               a share of blank trials
    -order "{random,permuted_block}"   a choice

``[a-b]`` draws whole numbers when both ends are whole ("[3-8]") and any
value otherwise ("[3.0-8.0]"); a trailing % keeps the percent sign. Every
placeholder is one axis. Configurations are drawn by Latin hypercube over the
ranges (uniform over the choices), each is realized a few times and scored
analytically (:class:`~.power.RealizationScorer`: detection per contrast and
response-shape estimation), and the result is the cloud of designs, its
Pareto front, and for a shortlist the best realization to actually run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_PLACEHOLDER = re.compile(r"\[\s*([0-9.]+)\s*-\s*([0-9.]+)\s*(%?)\s*\]|\{([^{}]+)\}")


@dataclass
class Axis:
    """One placeholder: where it sits in argv, and what it draws."""

    label: str  # e.g. "isi.1" -- the flag, then which placeholder of that flag
    token: int  # index in argv
    span: tuple[int, int]  # character span inside that token
    low: float = 0.0
    high: float = 0.0
    integer: bool = False
    percent: bool = False
    choices: list[str] = field(default_factory=list)

    @property
    def is_choice(self) -> bool:
        return bool(self.choices)

    def value(self, u: float) -> str:
        """The value at quantile ``u`` in [0, 1), as it is written into argv."""
        if self.is_choice:
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.integer:
            v: float = int(np.floor(self.low + u * (self.high - self.low + 1)))
            v = min(v, self.high)
        else:
            v = self.low + u * (self.high - self.low)
        return f"{v:.4g}" + ("%" if self.percent else "")

    def numeric(self, text: str) -> float | str:
        """A drawn value back as a number (for plotting), or the choice itself."""
        return text if self.is_choice else float(text.rstrip("%"))


def find_axes(argv: list[str]) -> list[Axis]:
    """Every [a-b] / {x,y} placeholder in argv, labelled by its flag."""
    axes: list[Axis] = []
    flag, seen = "arg", {}
    for i, tok in enumerate(argv):
        if tok.startswith("-") and not _is_number(tok):
            flag = tok.lstrip("-")
            continue
        for m in _PLACEHOLDER.finditer(tok):
            seen[flag] = seen.get(flag, 0) + 1
            label = f"{flag}.{seen[flag]}"
            if m.group(4) is not None:
                choices = [c.strip() for c in m.group(4).split(",") if c.strip()]
                if len(choices) < 2:
                    raise ValueError(f"{tok!r}: a choice needs two or more options")
                axes.append(Axis(label, i, m.span(), choices=choices))
                continue
            lo, hi = float(m.group(1)), float(m.group(2))
            if hi <= lo:
                raise ValueError(f"{tok!r}: a range needs low < high")
            whole = "." not in m.group(1) and "." not in m.group(2)
            axes.append(Axis(label, i, m.span(), lo, hi, whole, bool(m.group(3))))
    return axes


def _is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def sample(axes: list[Axis], n: int, seed: int = 0) -> list[dict[str, str]]:
    """``n`` configurations: a Latin hypercube over the axes (strata shuffled per axis)."""
    rng = np.random.default_rng(seed)
    cols = {a.label: (rng.permutation(n) + rng.random(n)) / n for a in axes}
    return [{a.label: a.value(float(cols[a.label][k])) for a in axes} for k in range(n)]


def render(argv: list[str], axes: list[Axis], values: dict[str, str]) -> list[str]:
    """argv with every placeholder replaced by its value in ``values``."""
    out = list(argv)
    by_token: dict[int, list[Axis]] = {}
    for a in axes:
        by_token.setdefault(a.token, []).append(a)
    for i, group in by_token.items():
        tok = out[i]
        for a in sorted(group, key=lambda a: a.span[0], reverse=True):
            tok = tok[: a.span[0]] + values[a.label] + tok[a.span[1] :]
        out[i] = tok
    return out


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask of points no other point beats on both (lower is better on each)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    front = np.zeros(len(x), dtype=bool)
    best_y = np.inf
    for k in np.argsort(np.where(ok, x, np.inf), kind="stable"):
        if not ok[k]:
            break
        if y[k] < best_y:
            front[k], best_y = True, y[k]
    return front


def shortlist(x: np.ndarray, front: np.ndarray, k: int) -> list[int]:
    """Up to ``k`` front designs spread along it, best detection first."""
    idx = sorted(np.flatnonzero(front), key=lambda i: x[i])
    if len(idx) <= k:
        return idx
    return [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, k)]


def score_configs(
    specs: list[Any],
    scorer: Any,
    n_realizations: int,
    ref_noise: str,
    seed: int = 0,
    progress: bool = True,
    single_all: bool = False,
) -> list[dict[str, Any]]:
    """Median scores over ``n_realizations`` for each spec (None: the config was refused).

    Keys: 'needed' {contrast: effect at ``ref_noise``}, 'worst' {contrast: max
    over realizations}, 'shape_sd', 'xi', 'lss_sd', 'lsa_sd', 'leakage',
    'minutes', 'counts', 'dropped'. Single-trial scores cost 20-65 ms a
    realization, so they come from the first realization only unless
    ``single_all`` (when they are what is being optimized).
    """
    from tqdm import tqdm

    from .experiment import realize

    out: list[dict[str, Any]] = []
    for spec in tqdm(specs, desc="designs", leave=True, disable=not progress or len(specs) < 2):
        if spec is None:
            out.append({})
            continue
        scores = []
        for r in range(n_realizations):
            try:
                sc = scorer.score(realize(spec, seed + r), single=single_all or r == 0)
            except ValueError:
                sc = None
            if sc is not None:
                scores.append(sc)
        if not scores:
            out.append({})
            continue
        names = [*scorer.names, "detection"]  # 'detection': the mean over contrasts
        per = {c: [s["needed"][(ref_noise, c)] for s in scores] for c in names}
        out.append(
            {
                "needed": {c: float(np.median(v)) for c, v in per.items()},
                "worst": {c: float(np.max(v)) for c, v in per.items()},
                "shape_sd": float(np.median([s["shape_sd"][ref_noise] for s in scores])),
                "xi": float(np.median([s["xi"] for s in scores])),
                **{
                    key: float(
                        np.median(
                            [
                                s[key][ref_noise] if isinstance(s[key], dict) else s[key]
                                for s in scores
                                if key in s
                            ]
                        )
                    )
                    for key in ("lss_sd", "lsa_sd", "leakage", "unreliability", "ridge_frac")
                    if any(key in s for s in scores)
                },
                "minutes": float(np.median([s["minutes"] for s in scores])),
                "counts": scores[0]["counts"],
                "dropped": float(np.mean([s["n_dropped"] for s in scores])),
            }
        )
    return out


def best_realization(
    spec: Any,
    scorer: Any,
    n: int,
    ref_noise: str,
    objective: str,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]] | None:
    """Of ``n`` realizations of ``spec``, the one best on ``objective``.

    ``objective`` is a contrast (its detection), 'shape' (estimation) or
    'trials' (single-trial unreliability: 1 - the better of LSS and ridge).
    """
    from .experiment import realize

    best = None
    for r in range(n):
        real = realize(spec, seed + r)
        sc = scorer.score(real, single=objective == "trials")
        if sc is None:
            continue
        v = (
            sc["shape_sd"][ref_noise]
            if objective == "shape"
            else sc["unreliability"][ref_noise]
            if objective == "trials"
            else sc["needed"][(ref_noise, objective)]
        )
        if np.isfinite(v) and (best is None or v < best[0]):
            best = (v, real, sc)
    return None if best is None else (best[1], best[2])
