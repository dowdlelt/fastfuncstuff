"""ROI x depth response curves from the toolbox, in the shape the laminar model inverts.

``ffs_deconvolve`` (and ``ffs_reml``) on ROIs sampled by depth write tidy curve tables:
``{prefix}_iresp_{condition}.tsv``, one line per time x ROI x depth, depth 1 deepest.
The model wants a ``(n_scans, K)`` matrix in percent signal change, **superficial
depth first**, and -- in the published design ([[Faes 2026]]) -- two conditions'
responses concatenated with white-noise padding that is masked out of the fit,
driven by one event per condition and a modulatory input marking the second. This
module does that conversion and nothing else.

The layout copies the reference driver (``make_modelspace_oracle.m``): each
condition's block is one zero baseline row followed by the response, the event
sits on the response's first sample, the modulatory input is a 0.1 s pulse ending
one sample after the second event, and the baseline and padding rows are masked.

Units are the caller's responsibility to have right at the source: the curves must
be percent signal change (``ffs_deconvolve``'s default since it scales; not with
``-dont_scale``). A curve with a mean far above 10 is refused as data units.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.laminar.params import P0, ModelSpec


@dataclass
class DepthCurves:
    """One ROI's response, ``values`` ``(n_times, K)`` with **depth 1 (deepest) first**,
    as the toolbox writes it."""

    times: np.ndarray
    values: np.ndarray
    roi: str
    name: str
    hemi: str
    coverage: np.ndarray | None = None


def read_iresp_by_k(path: str | Path) -> dict[int, dict[tuple[str, str, str], DepthCurves]]:
    """A tidy ``_iresp_<condition>.tsv`` as ``{K: {(roi, name, hemi): DepthCurves}}``.

    A table from a multi-count run (``-depths 7 9 11``) has a ``K`` column and holds
    several counts; one without has a single count, K being its number of depths.
    """
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows or "depth" not in rows[0]:
        raise ValueError(f"{path}: not an ROI x depth curve table (no 'depth' column)")
    value_col = list(rows[0])[-1]
    groups: dict[tuple[int, str, str, str], list[dict]] = {}
    for r in rows:
        k_col = int(r["K"]) if "K" in r else 0
        groups.setdefault((k_col, r["roi"], r["name"], r.get("hemi", "-")), []).append(r)
    out: dict[int, dict] = {}
    for (k_col, rid, name, hemi), rs in groups.items():
        times = np.array(sorted({float(r["time"]) for r in rs}))
        depths = sorted({int(r["depth"]) for r in rs})
        if depths != list(range(1, len(depths) + 1)):
            raise ValueError(f"{path}: ROI {name} has depths {depths}, not 1..K")
        if k_col and k_col != len(depths):
            raise ValueError(f"{path}: ROI {name} says K = {k_col} but has {len(depths)} depths")
        t_ix = {t: i for i, t in enumerate(times)}
        vals = np.full((len(times), len(depths)), np.nan)
        cov = np.ones(len(depths))
        for r in rs:
            vals[t_ix[float(r["time"])], int(r["depth"]) - 1] = float(r[value_col])
            if "coverage" in r:
                cov[int(r["depth"]) - 1] = float(r["coverage"])
        if np.isnan(vals).any():
            raise ValueError(f"{path}: ROI {name} is missing time x depth entries")
        out.setdefault(len(depths), {})[(rid, name, hemi)] = DepthCurves(
            times, vals, rid, name, hemi, cov
        )
    return out


def read_iresp_table(path: str | Path) -> dict[tuple[str, str, str], DepthCurves]:
    """Every ROI's curves in a tidy ``_iresp_<condition>.tsv`` (keyed ``(roi, name, hemi)``);
    the table must hold one depth count (see :func:`read_iresp_by_k`)."""
    by_k = read_iresp_by_k(path)
    if len(by_k) > 1:
        raise ValueError(f"{path}: holds depth counts {sorted(by_k)}; use read_iresp_by_k")
    return next(iter(by_k.values()))


def pick_roi(
    table: dict[tuple[str, str, str], DepthCurves], roi: str, hemi: str | None = None
) -> DepthCurves:
    """The ROI whose id or name is ``roi`` (and hemisphere, when two share a name)."""
    hits = [c for (rid, name, h), c in table.items() if roi in (rid, name) and hemi in (None, h)]
    if not hits:
        have = ", ".join(sorted({f"{n} ({h})" for (_, n, h) in table}))
        raise ValueError(f"no ROI {roi!r}{f' in {hemi}' if hemi else ''}; have {have}")
    if len(hits) > 1:
        raise ValueError(f"ROI {roi!r} is in several hemispheres: give hemi")
    return hits[0]


@dataclass
class LaminarData:
    """Everything one inversion needs, in the model's orientation."""

    y: torch.Tensor  # (n_scans, K), percent change, superficial first
    rows: torch.Tensor  # (n_scans,) bool, False on baselines and padding
    u: torch.Tensor  # (n_micro, 2): modulatory (second condition), then driving
    spec: ModelSpec
    #: Keyword arguments for ``integrate`` / ``variational_laplace``.
    integrate_kwargs: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _times_pair(
    times: np.ndarray | Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """``(first, second)`` sample grids from one grid or two."""
    arr = np.asarray(times, dtype=object) if isinstance(times, (list, tuple)) else None
    if arr is not None and arr.dtype == object:
        if len(times) != 2:  # type: ignore[arg-type]
            raise ValueError(f"times takes one grid or two, got {len(times)}")  # type: ignore[arg-type]
        return np.asarray(times[0], float), np.asarray(times[1], float)  # type: ignore[index]
    one = np.asarray(times, float)
    if one.ndim == 2:
        if one.shape[0] != 2:
            raise ValueError(f"times takes one grid or two, got {one.shape[0]}")
        return one[0], one[1]
    return one, one


def _scan_interval(t_first: np.ndarray, t_second: np.ndarray) -> float:
    """The shared sample spacing, which is the model's scan interval.

    Both grids must start at the event: it is placed on the response's first sample
    (``onset_a = tr``), so a window that starts later would be modelled as if it
    started at the onset. Refused rather than shifted -- the model integrates from
    rest, so a curve whose beginning was never sampled is a tail it cannot generate,
    not a delay it can absorb.
    """
    steps = []
    for t in (t_first, t_second):
        d = np.diff(t)
        if d.size == 0 or not np.allclose(d, d[0], rtol=1e-6):
            raise ValueError("curve samples must be evenly spaced (uniform TENT knots)")
        steps.append(float(d[0]))
    if not np.isclose(steps[0], steps[1], rtol=1e-6):
        raise ValueError(
            f"the two conditions are sampled {steps[0]:g} s and {steps[1]:g} s apart; the "
            "scan interval of the model is the knot spacing, so it must be the same"
        )
    tr = steps[0]
    for name, t in (("baseline", t_first), ("test", t_second)):
        if abs(float(t[0])) > 1e-6 * max(tr, 1.0):
            raise ValueError(
                f"the {name} curves start {float(t[0]):g} s after the event, but the model "
                "puts the event on the first sample; refit with a window that starts at 0"
            )
    return tr


def _duration_pair(duration: float | Sequence[float]) -> tuple[float, float]:
    """``(baseline, test)`` event lengths from one value or two."""
    if isinstance(duration, (int, float)):
        pair = (float(duration), float(duration))
    else:
        vals = [float(v) for v in duration]
        if len(vals) == 1:
            pair = (vals[0], vals[0])
        elif len(vals) == 2:
            pair = (vals[0], vals[1])
        else:
            raise ValueError(f"duration takes one value or two, got {len(vals)}")
    if min(pair) <= 0:
        raise ValueError(f"event durations must be positive, got {pair}")
    return pair


def two_condition_data(
    first: DepthCurves,
    second: DepthCurves,
    *,
    duration: float | Sequence[float],
    n_neuronal: int = 3,
    gap: int = 20,
    noise_sd: float = 0.1,
    seed: int = 20260918,
    delay_seconds: float | None = 0.0,
    TE: float = 0.028,
    B0: float = 7.0,
    min_coverage: float = 0.99,
    p0=None,
    dtype: torch.dtype = torch.float64,
    modulation: str | tuple[float, float] = "event",
) -> LaminarData:
    """Faes et al.'s design from two conditions' curves of one ROI.

    ``duration`` (s) is the event the curves respond to -- one value, or
    ``(baseline, test)`` when the conditions' events differ. The scan interval of the
    model is the curves' own sample spacing (the TENT knot spacing), which must be
    uniform and the same for both. ``delay_seconds`` is where in each sample the
    prediction is read: TENT knots sit at exact times after the event, so 0 by
    default; ``None`` restores the reference's slice-timing ``TR / 2``.
    """
    for c in (first, second):
        if c.values.shape[1] != first.values.shape[1]:
            raise ValueError("the two conditions have different numbers of depths")
        if c.coverage is not None and (c.coverage < min_coverage).any():
            raise ValueError(
                f"ROI {c.name}: some depths were read only {100 * c.coverage.min():.0f}% "
                "inside the EPI; a cut depth profile cannot be fitted"
            )
        if np.abs(c.values).mean() > 10:
            raise ValueError(
                f"ROI {c.name}: curves look like data units, not percent change "
                "(rerun ffs_deconvolve without -dont_scale)"
            )
    # The two may be sampled over different WINDOWS (a 3 s and a 10 s stimulus need
    # different lengths of response); the spacing and the start are checked downstream.
    data = two_condition_batch(
        first.values[None], second.values[None], (first.times, second.times),
        duration=duration,
        n_neuronal=n_neuronal, gap=gap, noise_sd=noise_sd, seed=seed,
        delay_seconds=delay_seconds, TE=TE, B0=B0, p0=p0, dtype=dtype, modulation=modulation,
    )  # fmt: skip
    data.y = data.y[0]
    data.meta.update({"roi": first.name, "hemi": first.hemi})
    return data


def two_condition_batch(
    first: np.ndarray,
    second: np.ndarray,
    times: np.ndarray | Sequence[np.ndarray],
    *,
    duration: float | Sequence[float],
    n_neuronal: int = 3,
    gap: int = 20,
    noise_sd: float = 0.1,
    seed: int = 20260918,
    delay_seconds: float | None = 0.0,
    TE: float = 0.028,
    B0: float = 7.0,
    p0=None,
    dtype: torch.dtype = torch.float64,
    modulation: str | tuple[float, float] = "event",
) -> LaminarData:
    """:func:`two_condition_data` for many regions sharing one design: ``first`` and
    ``second`` ``(B, n_times, K)``, depth 1 (deepest) first. ``y`` comes back
    ``(B, n_scans, K)``; ``u``, ``rows`` and ``spec`` are shared, as
    :func:`~fastfuncstuff.laminar.batched.variational_laplace_batched` wants. Every
    region gets the same padding draw, as separate calls with one seed would.

    ``duration`` is the length of the events the curves respond to: one value for
    both, or ``(baseline, test)`` when the two conditions' events differ. The
    modulatory input of ``modulation="event"`` follows the TEST event's own length,
    since that is the event it modulates.

    ``times`` is one sample grid for both conditions, or ``(first, second)`` when
    they were sampled over different windows -- a 3 s and a 10 s stimulus need
    different lengths of response. The two must share a sample SPACING, which is the
    model's scan interval, and both must start at the event. The blocks are then
    concatenated at their own lengths and ``rows`` marks each one's extent, so the
    second event's onset follows the first block's length rather than a shared one.

    ``modulation`` is when the modulatory input is on, relative to the second event:
    ``"event"`` (default) for the whole event, which is what a condition difference
    means when the conditions are simply different stimuli or tasks. ``"reference"``
    is the published pulse, 0.1 s ending one sample after the event onset. In Faes et
    al. that is the fourth tone of a 1.6 s quartet at TR 1.6, the moment of the
    prediction violation: a property of that paradigm, needed only for parity.
    ``(start, duration)`` gives seconds after the second onset."""
    t_first, t_second = _times_pair(times)
    tr = _scan_interval(t_first, t_second)
    if first.shape[1] != len(t_first) or second.shape[1] != len(t_second):
        raise ValueError(
            f"curves of {first.shape[1]} and {second.shape[1]} samples do not match their "
            f"time grids of {len(t_first)} and {len(t_second)}"
        )
    n_b, _, k = first.shape
    block = len(t_first) + 1  # one zero baseline row, then the first response

    rng = np.random.default_rng(seed)
    pad = noise_sd * rng.standard_normal((gap, k))
    flip = slice(None, None, -1)  # depth 1 (deepest) first -> superficial first
    zero = np.zeros((n_b, 1, k))
    y = np.concatenate(
        [zero, first[:, :, flip], np.broadcast_to(pad, (n_b, gap, k)), zero, second[:, :, flip]],
        axis=1,
    )
    rows = np.zeros(y.shape[1], bool)
    rows[1:block] = True
    rows[block + gap + 1 :] = True  # the rest is the second response, whatever its length

    spec = ModelSpec(
        N=n_neuronal, K=k, n_inputs=2, n_mod=1, TE=TE, B0=B0, TR=tr, dtype=dtype,
        p0=p0 if p0 is not None else P0(),  # e.g. pipeline.reference_p0(): the driver's baselines
    )  # fmt: skip
    from fastfuncstuff.laminar.integrate import build_input

    onset_a = tr  # the event on the first response sample
    onset_b = (block + gap + 1) * tr
    n_micro = int(round(y.shape[1] * tr / spec.dt))
    dur_first, dur_test = _duration_pair(duration)
    if modulation == "event":
        m_start, m_dur = 0.0, dur_test
    elif modulation == "reference":
        m_start, m_dur = tr - 0.1, 0.1
    elif isinstance(modulation, str):
        raise ValueError(
            f"modulation must be 'event', 'reference' or (start, duration), not {modulation!r}"
        )
    else:
        m_start, m_dur = float(modulation[0]), float(modulation[1])
    if m_dur <= 0:
        raise ValueError("the modulatory input needs a positive duration")
    u = build_input(
        spec,
        [[onset_b + m_start], [onset_a, onset_b]],
        [[m_dur], [dur_first, dur_test]],
        n_micro,
    )
    kwargs = {} if delay_seconds is None else {"delay_seconds": float(delay_seconds)}
    return LaminarData(
        y=torch.as_tensor(y, dtype=dtype),
        rows=torch.as_tensor(rows),
        u=u,
        spec=spec,
        integrate_kwargs=kwargs,
        meta={"TR": tr, "K": k, "block": block, "gap": gap},
    )


def read_depth_psf(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """A single-count ``*_depth_psf.tsv`` as ``{(name, hemi): M}``, each ``(K, K)`` with
    depth 1 deepest first and the weights as written."""
    from fastfuncstuff.processing.roi_depth import read_depth_psf_by_k

    by_k = read_depth_psf_by_k(str(path))
    if len(by_k) > 1:
        raise ValueError(f"{path}: holds depth counts {sorted(by_k)}; use read_depth_psf_by_k")
    return next(iter(by_k.values()))


def model_kernel(
    matrix_deep_first: np.ndarray, *, normalise: bool = True, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """A ``(K, K)`` depth sampling matrix from
    :func:`processing.roi_depth.depth_sampling_matrices` (depth 1 deepest, as the
    toolbox writes it) in the model's superficial-first order, ready for ``kernel=``."""
    m = np.asarray(matrix_deep_first, np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        raise ValueError("a depth sampling matrix must be a finite (K, K) array")
    if normalise:  # the reference's convention: no dilution by white matter or CSF
        s = m.sum(axis=1, keepdims=True)
        if (s <= 0).any():
            raise ValueError("a depth sampling matrix row sums to zero")
        m = m / s
    return torch.as_tensor(m[::-1, ::-1].copy(), dtype=dtype)


__all__ = [
    "DepthCurves",
    "LaminarData",
    "model_kernel",
    "pick_roi",
    "read_depth_psf",
    "read_iresp_by_k",
    "read_iresp_table",
    "two_condition_data",
]
