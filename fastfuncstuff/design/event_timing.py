"""How well event timing supports a FIR / TENT response estimate.

A TENT knot grid is only as good as the spread of event onsets relative to
the samples.  With knots every TR and every event half a TR off the samples,
the alternating (+,-,+,-) pattern on the knots crosses zero at every sample
and is invisible to the data: least squares fills it with noise and the
estimated response comes out as an up-down saw.  Offsets spread across the
TR remove that blind spot, and a spread wide enough lets the knots go finer
than the TR -- the resolution random timing buys and TR-rounding throws away.

The measures here are properties of the design alone, so they can be read
before any data are fitted:

* **phase** of each onset within the sample grid, ``((onset - t0) / TR) mod 1``
  (0 = on a sample, 0.5 = midway between two);
* **noise gain** of each knot: standard error per unit noise, scaled by
  ``sqrt(n_events)`` so an isolated TR-locked FIR knot scores 1;
* **amplification**: the worst direction's gain relative to what rounding the
  same events onto the samples (plain FIR at TR knots) would give.  The
  up-down artefact is a large amplification, not a large typical gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.design.builder import build_per_run_task_designs, legendre_polynomials

#: Worst-direction amplification (vs TR-rounded FIR) above which the estimate
#: is called noisy, and above which it is called unstable.  At 5x the up-down
#: pattern dominates typical single-subject noise levels in simulation.
NOISY_AMPLIFICATION = 2.5
UNSTABLE_AMPLIFICATION = 5.0
#: Relative eigenvalue below which a knot combination is taken as unobservable.
RANK_TOL = 1e-9


@dataclass
class DesignGain:
    """Noise behaviour of one FIR/TENT design (all conditions jointly)."""

    knot_dt: float
    n_basis_per_condition: list[int]
    identifiable: bool
    #: Per condition: median over knots of the per-knot noise gain.
    median_gain: list[float]
    #: Per condition: noise gain of the worst knot combination.
    worst_gain: list[float]
    #: Worst gain over conditions divided by the TR-rounded FIR's worst gain.
    amplification: float = float("nan")

    @property
    def status(self) -> str:
        if not self.identifiable:
            return "unidentifiable"
        if self.amplification > UNSTABLE_AMPLIFICATION:
            return "unstable"
        if self.amplification > NOISY_AMPLIFICATION:
            return "noisy"
        return "ok"


@dataclass
class TimingReport:
    """Everything ``ffs_util_eventcheck`` reports about one event set."""

    tr: float
    microtime_offset: float
    precision: float
    window: tuple[float, float]
    condition_labels: list[str]
    n_events: list[int]
    #: Onset phase within the sample grid, per event (all conditions), in [0, 1).
    phases: np.ndarray
    #: Mean resultant length of the phases (1 = all identical, 0 = uniform).
    phase_concentration: float
    #: Circular mean phase in [0, 1).
    mean_phase: float
    #: mean((1 - 2 * phase)^2): 1 when every event is on a sample, 0 when every
    #: event is midway.  The TR-knot grid sees its alternating pattern only
    #: through this; 1/3 for uniformly spread phases.
    alternation_visibility: float
    #: Onset shift (s) that rounding to the nearest sample would apply.
    rounding_shift_max: float
    rounding_shift_rms: float
    #: One entry per candidate knot spacing, coarsest (= TR) first.
    grids: list[DesignGain] = field(default_factory=list)


def onset_phases(
    onsets: np.ndarray, tr: float, microtime_offset: float = 0.0, precision: float = 0.0
) -> np.ndarray:
    """Phase of each onset within the sample grid, in ``[0, 1)``.

    ``precision`` (s) quantizes onsets first: logged times like 10.2467 s carry
    digits no stimulus clock delivers, and they would read as timing
    diversity that is not there.
    """
    t = np.asarray(onsets, dtype=np.float64).ravel() - microtime_offset
    if precision > 0:
        t = np.round(t / precision) * precision
    return np.mod(t / tr, 1.0) % 1.0


def design_gain(
    per_run_designs: list[torch.Tensor],
    n_basis_per_condition: list[int],
    n_events_per_condition: list[int],
    polort: int = 2,
    knot_dt: float = float("nan"),
) -> DesignGain:
    """Per-knot and worst-direction noise gain of a task design.

    The design is projected against each run's Legendre drift (the part the
    GLM spends on baseline) before ``(X'X)^-1`` is formed, so the numbers are
    what a fit with that ``polort`` actually gets.  Gains are scaled by
    ``sqrt(n_events)``: an isolated, TR-locked FIR knot scores 1.
    """
    blocks = []
    for design in per_run_designs:
        x = design.detach().to("cpu", torch.float64).numpy()
        if polort >= 0 and x.shape[0] > polort + 1:
            q, _ = np.linalg.qr(legendre_polynomials(x.shape[0], polort))
            x = x - q @ (q.T @ x)
        blocks.append(x)
    x = np.concatenate(blocks, axis=0)
    gram = x.T @ x
    evals, evecs = np.linalg.eigh(gram)
    top = max(float(evals[-1]), 1e-300)
    keep = evals > RANK_TOL * top
    identifiable = bool(keep.all())
    cov = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T

    median_gain, worst_gain = [], []
    start = 0
    for n_basis, n_ev in zip(n_basis_per_condition, n_events_per_condition, strict=True):
        block = cov[start : start + n_basis, start : start + n_basis]
        start += n_basis
        scale = max(n_ev, 1)
        median_gain.append(float(np.sqrt(scale * np.median(np.clip(np.diag(block), 0, None)))))
        worst_gain.append(float(np.sqrt(scale * max(np.linalg.eigvalsh(block)[-1], 0.0))))
    return DesignGain(
        knot_dt=knot_dt,
        n_basis_per_condition=list(n_basis_per_condition),
        identifiable=identifiable,
        median_gain=median_gain,
        worst_gain=worst_gain,
    )


def _grid_design(
    onsets: list[list[np.ndarray]],
    n_tp: list[int],
    tr: float,
    window: tuple[float, float],
    knot_dt: float,
    microtime_offset: float,
) -> tuple[list[torch.Tensor], list[int]]:
    bot, top = window
    n_knots = int(round((top - bot) / knot_dt)) + 1
    result = build_per_run_task_designs(
        onsets,
        n_tp,
        tr,
        basis="TENT",
        fir_window_s=[(bot, bot + (n_knots - 1) * knot_dt)] * len(onsets),
        tent_n_basis=n_knots,
        microtime_offset=microtime_offset,
        device=torch.device("cpu"),
    )
    return result.per_run, list(result.n_basis_per_condition)


def rounded_to_samples(
    onsets: list[list[np.ndarray]], tr: float, microtime_offset: float
) -> list[list[np.ndarray]]:
    """Every onset moved to its nearest sample (what TR-rounding would do)."""
    return [
        [
            np.round((np.asarray(o, dtype=np.float64) - microtime_offset) / tr) * tr
            + microtime_offset
            for o in runs
        ]
        for runs in onsets
    ]


def check_event_timing(
    onsets: list[list[np.ndarray]],
    n_timepoints_per_run: list[int],
    tr: float,
    *,
    window: tuple[float, float],
    microtime_offset: float = 0.0,
    precision: float = 0.05,
    max_subdivision: int = 4,
    polort: int = 2,
    condition_labels: list[str] | None = None,
) -> TimingReport:
    """Phase statistics plus noise gains for knots at TR, TR/2, ... TR/max_subdivision.

    ``onsets`` is ``[condition][run]`` of run-relative onset times (s).  The
    design at each spacing is a TENT over ``window``; amplifications are all
    relative to the same events rounded onto the samples with TR knots (the
    plain-FIR fallback), so they read as "what this grid costs over rounding".
    """
    if condition_labels is None:
        condition_labels = [f"cond{i}" for i in range(len(onsets))]
    quantized = [
        [
            np.round(np.asarray(o, dtype=np.float64) / precision) * precision
            if precision > 0
            else np.asarray(o, dtype=np.float64)
            for o in runs
        ]
        for runs in onsets
    ]
    n_events = [int(sum(np.asarray(o).size for o in runs)) for runs in quantized]
    all_t = np.concatenate([np.asarray(o).ravel() for runs in quantized for o in runs] or [[]])
    phases = onset_phases(all_t, tr, microtime_offset)
    angle = np.exp(2j * np.pi * phases)
    resultant = complex(angle.mean()) if phases.size else 0j
    shift = np.minimum(phases, 1.0 - phases) * tr

    report = TimingReport(
        tr=tr,
        microtime_offset=microtime_offset,
        precision=precision,
        window=window,
        condition_labels=list(condition_labels),
        n_events=n_events,
        phases=phases,
        phase_concentration=float(abs(resultant)),
        mean_phase=float(np.mod(np.angle(resultant) / (2 * np.pi), 1.0)),
        alternation_visibility=float(np.mean((1.0 - 2.0 * phases) ** 2)) if phases.size else 0.0,
        rounding_shift_max=float(shift.max()) if shift.size else 0.0,
        rounding_shift_rms=float(np.sqrt(np.mean(shift**2))) if shift.size else 0.0,
    )

    ref_designs, ref_basis = _grid_design(
        rounded_to_samples(quantized, tr, microtime_offset),
        n_timepoints_per_run,
        tr,
        window,
        tr,
        microtime_offset,
    )
    reference = design_gain(ref_designs, ref_basis, n_events, polort, knot_dt=tr)
    ref_worst = max(reference.worst_gain) if reference.identifiable else float("nan")

    for m in range(1, max_subdivision + 1):
        dt = tr / m
        designs, n_basis = _grid_design(
            quantized, n_timepoints_per_run, tr, window, dt, microtime_offset
        )
        gain = design_gain(designs, n_basis, n_events, polort, knot_dt=dt)
        gain.amplification = max(gain.worst_gain) / ref_worst if gain.identifiable else float("inf")
        report.grids.append(gain)
    return report


def finest_usable_grid(report: TimingReport, allow: tuple[str, ...] = ("ok",)) -> DesignGain | None:
    """The smallest knot spacing whose status is in ``allow``, if any."""
    usable = [g for g in report.grids if g.status in allow]
    return min(usable, key=lambda g: g.knot_dt) if usable else None


def aligned_knot_start(report: TimingReport) -> float | None:
    """TENT ``bot`` that puts knots ON the samples when every event shares a phase.

    With one common phase ``p`` the samples after an event sit at lags
    ``(1 - p) * TR + k * TR``; knots there turn TENT into an exact FIR at the
    true lags -- no rounding shift, no amplification.  ``None`` when phases
    are spread (then no single shift helps) or already on the samples.
    """
    if report.phase_concentration < 0.99:
        return None
    p = report.mean_phase
    if min(p, 1.0 - p) * report.tr < 0.02:
        return None
    return (1.0 - p) * report.tr
