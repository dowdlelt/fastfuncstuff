"""Subject-level figures drawn across runs when a QC folder is stitched.

A tool writes one run at a time and never knows the others' ranges, so anything
that must put every run on one scale is drawn here, by ``ffs_qc_report``, from
what the records (per-TR traces) and their sidecars (tSNR slices) kept. Running
the report again simply redraws them, so a run added later joins the scale.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .record import RECORDS_DIR, clean_label, metric_flagged, natural_key, warn_text

SUMMARY_LABEL = "runs"
MOTION_MEASURES = (("enorm", "enorm (mm / TR)"), ("fd", "FD (mm)"))


def motion_summary(qc_dir: Path, records: list[dict]) -> list[dict]:
    """One raincloud figure per tool whose records carry per-TR motion; returns
    pseudo-records (no metrics) naming the figures for the page."""
    from . import plots, render

    by_tool: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if any(k in rec.get("series", {}) for k, _ in MOTION_MEASURES):
            by_tool[rec["tool"]].append(rec)
    out = []
    for tool, recs in by_tool.items():
        recs = sorted(recs, key=lambda r: natural_key(r["label"]))
        panels = []
        stats = []
        for key, ylabel in MOTION_MEASURES:
            runs = []
            limit = None
            for r in recs:
                ser = r.get("series", {})
                if key not in ser:
                    continue
                v = np.array([np.nan if x is None else x for x in ser[key]], float)
                keep = ser.get("keep")
                cens = None if keep is None else np.asarray(keep) == 0
                runs.append(plots.RunValues(r["label"], v, cens))
                if r.get("info", {}).get("motion_metric") == key:
                    limit = r["info"].get("motion_limit")
            if runs:
                panels.append((ylabel, runs, limit))
                means = [float(np.nanmean(rv.values)) for rv in runs]
                worst = int(np.argmax(means))
                stats.append(
                    (f"highest mean {key}: {runs[worst].label} ({means[worst]:.3f})", None)
                )
        if not panels:
            continue
        body = plots.raincloud(panels)
        n = len(recs)
        img = render.compose(
            body,
            f"Motion across runs · {n} run{'s' if n != 1 else ''}",
            f"{tool}  ·  every time point of every run, one y scale per measure",
            stats[:1],
            "Each run: a half violin (distribution), a box (quartiles, line = median, yellow "
            "dot = mean) and every time point as a dot, red where it was censored. All runs "
            "share one y axis, so a run that moved more than the others stands out. The axis "
            "stops at the pooled 99.5th percentile; larger spikes sit on the top edge as "
            "triangles with their count. Look for: one run sitting higher or wider than the "
            "rest, or many red dots — open that run's motion figure.",
        )
        order = recs[0].get("order", 90)
        path = qc_dir / f"{order:02d}.{tool}.00-motion_summary.{SUMMARY_LABEL}.png"
        render.save_png(img, path)
        out.append(
            {
                "tool": tool,
                "label": SUMMARY_LABEL,
                "order": order,
                "info": {},
                "metrics": {},
                "figures": [
                    {
                        "step": 0,
                        "check": "motion_summary",
                        "title": "Motion across runs",
                        "file": path.name,
                    }
                ],
            }
        )
    return out


def shared_tsnr(qc_dir: Path) -> int:
    """Redraw every run's tSNR montage on one scale per (tool, series kind).

    The shared top is the highest run's 99th percentile: nothing saturates, and a
    run with poor tSNR reads visibly darker than the rest. Returns the number of
    figures redrawn (groups of one run keep their own scale).
    """
    from .checks import tsnr_figure
    from .render import save_png

    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for p in sorted((qc_dir / RECORDS_DIR).glob("*.tsnr.npz"), key=lambda p: natural_key(p.name)):
        try:
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                planes = {k: z[k] for k in z.files if k != "meta"}
        except (OSError, ValueError, KeyError):
            continue
        groups[(meta["tool"], meta["what"])].append((meta, planes))
    n = 0
    for items in groups.values():
        if len(items) < 2:
            continue
        p99 = [m["t_p99"] for m, _ in items]
        t_hi = float(max(p99))
        best = items[int(np.argmax(p99))][0]["label"]
        shared = f"shared by all {len(items)} runs (highest run 99th pct, {best})"
        for meta, planes in items:
            save_png(tsnr_figure(meta, planes, t_hi, shared), qc_dir / meta["file"])
            n += 1
    return n


# ---------------------------------------------------------------------------
# The session wheel
# ---------------------------------------------------------------------------

WHEEL_TOOL = "session"
_TASK = re.compile(r"task-([A-Za-z0-9]+)")


def _task_and_variant(label: str) -> tuple[str, str]:
    """``stage12.blur6.task-faces`` -> (``faces``, ``stage12.blur6``). The variant is
    whatever names the fit besides its task: two fits of one task (a blur, a
    phase-regressed refit) are two wheels, never two sectors of one."""
    m = _TASK.search(label)
    if not m:
        return label, ""
    return m.group(1), (label[: m.start()] + label[m.end() :]).strip("._- ")


def _flag_lines(rec: dict, who: str) -> list[str]:
    from .report import fmt_value

    return [
        f"{who}: {rec['tool']} {name} {fmt_value(m['value'])} ({warn_text(m.get('warn'))})"
        for name, m in rec["metrics"].items()
        if metric_flagged(m)
    ]


def _series(rec: dict, key: str) -> np.ndarray | None:
    v = rec.get("series", {}).get(key)
    return None if v is None else np.array([np.nan if x is None else x for x in v], float)


def _quantiles(table, i: int) -> tuple[float, ...] | None:
    """Run ``i``'s tSNR quantiles from a sidecar (absent before they were stored)."""
    if not table or i >= len(table) or any(v is None for v in table[i]):
        return None
    return tuple(float(v) for v in table[i])


def session_wheel(qc_dir: Path, records: list[dict]) -> list[dict]:
    """One wheel per fitted variant, from the GLM sidecars and every run-level record.

    A run is the GLM's: its label is the input file's. Another tool's record
    belongs to a run when one label contains the other, which is how autoproc
    names things (``ses-01.task-x.run-01`` inside ``stage10.final.ses-01...``)
    and what a hand-run pipeline gets by keeping the BIDS entities in its names.
    """
    from . import render, wheel
    from .checks import STACK_CORR_WARN, WHEEL_SIDECAR

    fits: dict[str, list[tuple[str, dict, np.ndarray, np.ndarray]]] = defaultdict(list)
    for p in sorted(
        (qc_dir / RECORDS_DIR).glob(f"*.{WHEEL_SIDECAR}.npz"), key=lambda p: natural_key(p.name)
    ):
        try:
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                tsnr, aspect = z["tsnr"].astype(np.float32), z["aspect"]
        except (OSError, ValueError, KeyError):
            continue
        task, variant = _task_and_variant(meta["label"])
        fits[variant].append((task, meta, tsnr, aspect))
    if not fits:
        return []
    motion = [r for r in records if any(k in r.get("series", {}) for k, _ in MOTION_MEASURES)]
    stacks = [r for r in records if r["tool"] == "stack" and r.get("info", {}).get("frames")]
    others = [r for r in records if r["tool"] not in ("reml", "stack", WHEEL_TOOL)]
    out = []
    for variant, items in sorted(fits.items(), key=lambda kv: natural_key(kv[0])):
        items.sort(key=lambda it: natural_key(it[0]))
        tasks, flags, slots = [], [], {}
        for task, meta, _, _ in items:
            runs = []
            for i, label in enumerate(meta["runs"]):
                who = f"{task} {wheel.short_run_label(label, i)}"
                # The longest motion label inside this run's name is its own.
                mine = max(
                    (r for r in motion if r["label"] in label),
                    key=lambda r: len(r["label"]),
                    default=None,
                )
                frag = mine["label"] if mine else label
                run = wheel.WheelRun(
                    label,
                    tsnr_in=meta["run_tsnr_input"][i],
                    tsnr_res=meta["run_tsnr_resid"][i],
                    r2=meta["run_heldout_r2"][i],
                    tsnr_in_q=_quantiles(meta.get("run_tsnr_input_q"), i),
                    tsnr_res_q=_quantiles(meta.get("run_tsnr_resid_q"), i),
                )
                if mine is not None:
                    key = mine.get("info", {}).get("motion_metric") or "enorm"
                    run.motion = _series(mine, key if key in mine["series"] else "fd")
                    run.motion_limit = mine.get("info", {}).get("motion_limit")
                    run.outlier = _series(mine, "outlier")
                    run.outlier_limit = mine.get("info", {}).get("outlier_limit")
                for rec in others:
                    if frag not in rec["label"] and rec["label"] not in label:
                        continue
                    slot = rec["tool"]
                    slots[slot] = rec.get("order", 90)
                    lines = _flag_lines(rec, who)
                    flags += lines
                    if lines or run.status.get(slot) != "warn":
                        run.status[slot] = "warn" if lines else "ok"
                for rec in stacks:
                    frames = rec["info"]["frames"].split("\n")
                    corr = _series(rec, "corr")
                    hit = [k for k, f in enumerate(frames) if f and (f in label or label in f)]
                    if corr is None or not hit or hit[0] >= corr.size:
                        continue
                    stage = rec["label"].split(".QC.")[-1]
                    slot = f"aligned:{stage}"
                    slots[slot] = rec.get("order", 55) + 0.5
                    low = corr[hit[0]] < STACK_CORR_WARN
                    run.status[slot] = "warn" if low else "ok"
                    if low:
                        flags.append(
                            f"{who}: {stage} stack r {corr[hit[0]]:.3f} (< {STACK_CORR_WARN:g})"
                        )
                runs.append(run)
            tasks.append(wheel.WheelTask(task, runs, meta.get("frac_sig")))
        slot_names = [k for k, _ in sorted(slots.items(), key=lambda kv: (kv[1], kv[0]))]
        # Centre: the fits share the final grid, so their maps average; when one
        # is on another grid the fit with the most data stands for the session.
        shapes = {it[2].shape for it in items}
        big = max(items, key=lambda it: sum(it[1]["run_kept"]))
        cuts = np.mean([it[2] for it in items], axis=0) if len(shapes) == 1 else big[2]
        centre = cuts[cuts.shape[0] // 2]
        kept = [sum(it[1]["run_kept"]) for it in items]
        med = float(np.average([it[1]["tsnr_resid_median"] for it in items], weights=kept))
        n_runs = sum(len(t.runs) for t in tasks)
        headline = [
            f"residual tSNR {med:.0f}",
            f"{n_runs} run{'s' if n_runs != 1 else ''} in {len(tasks)} "
            f"task{'s' if len(tasks) != 1 else ''}, {sum(kept):,} volumes kept",
        ]
        body = wheel.wheel(
            tasks, slot_names, centre, tuple(float(a) for a in big[3]), headline, flags
        )
        name = variant or "all"
        img = render.compose(
            body,
            f"Session wheel · {name}",
            "every run of every fitted task  ·  fixed axes: the same drawing for every subject",
            [(f"{len(flags)} to check", "warn" if flags else "ok")],
            "One sector per run, grouped by task; read from the rim to the centre. Rim: how the "
            "head moved. Then each tool's verdict, then tSNR before and after the model, then "
            "how well the other runs predict this one, then how much of the brain the task "
            "reaches, and in the middle the tSNR that is left. Every axis is fixed, so two "
            "subjects' wheels differ only where their data does. Look for: a sector that breaks "
            "the pattern of its neighbours (motion spilling past the dashed arc, a short tSNR "
            "bar sitting low, a missing fit bar), and whether the same run is the odd one in "
            "several rings: that run is the one to open.",
        )
        path = qc_dir / f"00.{WHEEL_TOOL}.00-wheel.{clean_label(name)}.png"
        render.save_png(img, path)
        out.append(
            {
                "tool": WHEEL_TOOL,
                "label": name,
                "order": 0,
                "info": {},
                "metrics": {},
                "figures": [
                    {"step": 0, "check": "wheel", "title": "Session wheel", "file": path.name}
                ],
            }
        )
    return out
