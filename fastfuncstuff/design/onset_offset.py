"""Onset/offset (OSO) expansion of a block design.

Three regressors per condition, all built from the *same* impulse response
``h``:

    SUS = h (x) boxcar(duration)    ON = h (x) impulse @ onset
    OFF = h (x) impulse @ (onset + duration)

which is Gonzalez-Castillo's decomposition of a block response into a sustained
part and two transients.  Two indices fall out of the betas for free:

    w = b_SUS / (|b_SUS| + |b_ON| + |b_OFF|)       the waveshape index
    a = (b_ON - b_OFF) / (|b_ON| + |b_OFF|)        the accumulator/adapter axis

The expansion needs nothing but the event list and the durations already in the
design, and it reuses the condition machinery wholesale: an ON column is just
another condition whose events are the block onsets with duration 0, so
``build_task_design`` builds all three without knowing OSO exists.  That also
means each of the three arrives peak-normalised to 1.0 -- see the anchor rule in
``matrices.build_event_design_microtime`` -- so the three betas are in
comparable units and ``w`` is computable without a scaling fudge.

> This is a better *model* for block designs and a fast index.  It is NOT a
> within-block dynamics estimator: b_OFF conflates "still accumulating at
> offset" with "rebounds after offset", so ``a`` agrees with an FIR slope for
> positively-sustained voxels and is misled for negatively-sustained ones.  See
> ``../fmri_wiki/concepts/Block response dynamics and the waveshape index.md``.

## Why the gate exists

SUS, ON and OFF are separable only when the block is long relative to the HRF.
Measured on the canonical library, worst case over HRF candidates:

    duration  TR    r(SUS, ON)     max VIF
      20 s   2.0   0.06 - 0.20        1.1
      10 s   2.0   0.12 - 0.48        1.7
       4 s   2.0   0.50 - 0.78        5.9
       2 s   1.0   0.84 - 0.90         71
     0.5 s   1.0        0.99         ~1e4

Below a couple of seconds the three columns are the *same* regressor: the fit
does not fail, it returns three enormous cancelling betas and a meaningless
``w``.  So the expansion is gated per condition on the design's own
conditioning, which lets a mixed design put OSO on its 20 s blocks and leave its
0.5 s trials alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.design.matrices import build_task_design

# A triple whose worst column exceeds this is refused.  4 s blocks at TR 2 land
# at 5.9 and 10 s blocks at 1.7, so the threshold sits in the gap between "the
# transients are separately estimable" and "they are the sustained regressor
# again".
DEFAULT_VIF_MAX = 5.0

ON_SUFFIX = "_ON"
OFF_SUFFIX = "_OFF"


@dataclass
class OSOPlan:
    """Which conditions get onset/offset columns, and where those columns land.

    ``groups`` maps an original condition index to its (SUS, ON, OFF) positions
    in the *expanded* condition list; conditions that failed the gate are absent
    from it and keep their single column.
    """

    mode: str = "off"
    enabled: list[bool] = field(default_factory=list)
    max_vif: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    source: list[int] = field(default_factory=list)
    groups: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    vif_max: float = DEFAULT_VIF_MAX

    @property
    def active(self) -> bool:
        """True when at least one condition actually gets the extra columns."""
        return self.mode != "off" and any(self.enabled)

    @property
    def n_expanded(self) -> int:
        return len(self.source)

    def to_metadata(self) -> dict:
        """JSON-safe summary for the metadata sidecar."""
        return {
            "mode": self.mode,
            "vif_max": self.vif_max,
            "enabled": list(self.enabled),
            "max_vif": [None if not np.isfinite(v) else float(v) for v in self.max_vif],
            "reasons": list(self.reasons),
            "expanded_labels": list(self.labels),
            "groups": {str(k): list(v) for k, v in self.groups.items()},
        }


def column_vifs(design: torch.Tensor) -> torch.Tensor:
    """Variance inflation factor per column, against every other column.

    An intercept is implied by mean-centring rather than added, matching what
    the polynomial nuisance block does to the design downstream.  A constant
    column has no variance to inflate and returns ``inf``, which is the right
    answer for the gate: it is exactly the degenerate case (every block offset
    falling past the end of its run) that must be refused.
    """
    X = design.to(dtype=torch.float64)
    X = X - X.mean(dim=0, keepdim=True)
    n_cols = X.shape[1]
    vifs = torch.full((n_cols,), float("inf"), dtype=torch.float64)
    for j in range(n_cols):
        own = X[:, j]
        ss_tot = float((own**2).sum())
        if ss_tot <= 0:
            continue
        if n_cols == 1:
            vifs[j] = 1.0
            continue
        others = torch.cat([X[:, :j], X[:, j + 1 :]], dim=1)
        beta = torch.linalg.lstsq(others, own.unsqueeze(1)).solution
        ss_res = float(((own.unsqueeze(1) - others @ beta) ** 2).sum())
        # Relative, not `<= 0`: an exactly duplicated column leaves a residual
        # around 1e-31 rather than zero, and a VIF of 2e31 is infinity wearing
        # a hat.  Comparing against ss_tot keeps the test scale-free.
        vifs[j] = float("inf") if ss_res <= ss_tot * 1e-12 else ss_tot / ss_res
    return vifs


def offset_events(cond_runs: list[np.ndarray], duration: float) -> list[np.ndarray]:
    """Block-offset times for one condition: every onset shifted by duration.

    Offsets past the end of a run are left in place, not dropped:
    ``build_event_design_microtime`` samples them to zero on its own, and
    filtering here would need the run lengths for no gain.
    """
    return [np.asarray(run, dtype=np.float64).ravel() + float(duration) for run in cond_runs]


def expand_events(
    event_onsets: list[list[np.ndarray]],
    durations: list[float],
    condition_labels: list[str] | None,
    enabled: list[bool],
) -> tuple[list[list[np.ndarray]], list[float], list[str], dict[int, tuple[int, int, int]]]:
    """Expand the condition list so SUS/ON/OFF are ordinary conditions.

    Returns the expanded ``(event_onsets, durations, labels, groups)``.  Every
    caller that builds a design must use all three of the first values together;
    passing expanded events with unexpanded durations silently drops the boxcar.
    """
    if len(event_onsets) != len(durations):
        raise ValueError("event_onsets and durations must have one entry per condition")
    if len(enabled) != len(event_onsets):
        raise ValueError("enabled must have one entry per condition")
    labels = (
        list(condition_labels) if condition_labels else [f"cond{i}" for i in range(len(durations))]
    )
    if len(labels) < len(event_onsets):
        raise ValueError("condition_labels must cover every condition")

    out_events: list[list[np.ndarray]] = []
    out_durations: list[float] = []
    out_labels: list[str] = []
    groups: dict[int, tuple[int, int, int]] = {}

    for cond_idx, (cond_runs, duration) in enumerate(zip(event_onsets, durations, strict=True)):
        sus_col = len(out_events)
        out_events.append(cond_runs)
        out_durations.append(float(duration))
        out_labels.append(labels[cond_idx])
        if not enabled[cond_idx]:
            continue
        out_events.append([np.asarray(run, dtype=np.float64).ravel() for run in cond_runs])
        out_durations.append(0.0)
        out_labels.append(f"{labels[cond_idx]}{ON_SUFFIX}")
        out_events.append(offset_events(cond_runs, duration))
        out_durations.append(0.0)
        out_labels.append(f"{labels[cond_idx]}{OFF_SUFFIX}")
        groups[cond_idx] = (sus_col, sus_col + 1, sus_col + 2)

    return out_events, out_durations, out_labels, groups


def plan_onset_offset(
    mode: str,
    event_onsets: list[list[np.ndarray]] | None,
    durations: list[float] | None,
    condition_labels: list[str] | None,
    hrf_library: torch.Tensor,
    n_timepoints: int,
    run_starts: list[int],
    tr: float,
    microtime_dt: float,
    microtime_onset: int = 0,
    vif_max: float = DEFAULT_VIF_MAX,
    n_probe_hrfs: int = 5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> OSOPlan:
    """Decide, per condition, whether onset/offset columns are estimable.

    The triple is built for a spread of library candidates and judged on its own
    conditioning (SUS/ON/OFF against each other, nothing else).  Judging it
    inside the full multi-condition design instead would let one hopeless
    condition inflate the VIF of a perfectly good one and disable it too.

    Conditioning varies smoothly but not monotonically with the candidate -- at
    4 s the slowest HRF is worst, at 2 s a middling one is -- so a spread is
    probed and the worst case decides.
    """
    n_conditions = len(durations) if durations is not None else 0
    plan = OSOPlan(
        mode=mode,
        enabled=[False] * n_conditions,
        max_vif=[float("nan")] * n_conditions,
        reasons=[""] * n_conditions,
        vif_max=float(vif_max),
    )

    if mode == "off" or n_conditions == 0:
        plan.labels = list(condition_labels) if condition_labels else []
        plan.source = list(range(n_conditions))
        return plan

    if event_onsets is None:
        raise ValueError(
            "-oso_mode needs the event list; it cannot be recovered from a sampled onset matrix"
        )

    if device is None:
        device = hrf_library.device
    library = hrf_library if hrf_library.ndim == 2 else hrf_library.unsqueeze(0)
    n_hrfs = int(library.shape[0])
    probe = sorted(
        set(int(i) for i in np.linspace(0, n_hrfs - 1, min(n_probe_hrfs, n_hrfs)).round())
    )

    for cond_idx in range(n_conditions):
        duration = float(durations[cond_idx])
        if duration <= 0:
            plan.reasons[cond_idx] = "impulse events (duration 0)"
            continue

        cond_events = [event_onsets[cond_idx]]
        worst = 0.0
        empty_column = False
        for hrf_idx in probe:
            triple_events, triple_durations, _, _ = expand_events(
                cond_events, [duration], [f"c{cond_idx}"], [True]
            )
            design = build_task_design(
                library[hrf_idx],
                n_timepoints,
                run_starts,
                tr=tr,
                microtime_dt=microtime_dt,
                microtime_onset=microtime_onset,
                event_onsets=triple_events,
                durations=triple_durations,
                device=device,
            )
            design = design.cpu()
            # An identically-zero column is its own failure, and a distinct one:
            # it means every block offset fell outside its run, not that the
            # transients are confounded with the sustained response.
            if float(design.abs().max(dim=0).values.min()) <= 0.0:
                empty_column = True
                worst = float("inf")
                break
            worst = max(worst, float(column_vifs(design).max()))
            if not np.isfinite(worst):
                break
        plan.max_vif[cond_idx] = worst
        if empty_column:
            plan.reasons[cond_idx] = "a column is empty (offsets fall outside the runs)"
        elif not np.isfinite(worst):
            plan.reasons[cond_idx] = "SUS/ON/OFF are perfectly collinear"
        elif worst > vif_max:
            plan.reasons[cond_idx] = f"VIF {worst:.1f} > {vif_max:.1f} (events too short)"
        else:
            plan.enabled[cond_idx] = True
            plan.reasons[cond_idx] = f"VIF {worst:.1f}"

    _, _, plan.labels, plan.groups = expand_events(
        event_onsets, list(durations), condition_labels, plan.enabled
    )
    plan.source = []
    for cond_idx in range(n_conditions):
        plan.source.extend([cond_idx] * (3 if plan.enabled[cond_idx] else 1))

    if verbose:
        print()
        print(f"Onset/offset expansion (-oso_mode {mode}):")
        for cond_idx in range(n_conditions):
            label = plan.labels[plan.source.index(cond_idx)]
            mark = "ON+OFF added" if plan.enabled[cond_idx] else "skipped"
            print(
                f"  {label:<20s} {float(durations[cond_idx]):6.2f}s  {mark:<12s} "
                f"{plan.reasons[cond_idx]}"
            )
        if not plan.active:
            print("  No condition qualified -- the fit is the ordinary sustained-only model.")
        print()

    return plan


def waveshape_maps(
    betas: torch.Tensor,
    plan: OSOPlan,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Waveshape index ``w`` and asymmetry ``a`` per OSO condition.

    ``betas`` is (n_voxels, n_task_columns) in expanded-condition order.  Keys
    are ``"{label}_w"`` and ``"{label}_a"``; conditions without OSO columns
    contribute nothing.
    """
    maps: dict[str, torch.Tensor] = {}
    for cond_idx, (sus, on, off) in plan.groups.items():
        b_sus, b_on, b_off = betas[:, sus], betas[:, on], betas[:, off]
        label = plan.labels[sus] if sus < len(plan.labels) else f"cond{cond_idx}"
        denom_w = b_sus.abs() + b_on.abs() + b_off.abs()
        maps[f"{label}_w"] = torch.where(
            denom_w > eps, b_sus / denom_w.clamp(min=eps), torch.zeros_like(b_sus)
        )
        denom_a = b_on.abs() + b_off.abs()
        maps[f"{label}_a"] = torch.where(
            denom_a > eps, (b_on - b_off) / denom_a.clamp(min=eps), torch.zeros_like(b_sus)
        )
    return maps
