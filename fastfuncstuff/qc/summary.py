"""Subject-level figures drawn across runs when a QC folder is stitched.

A tool writes one run at a time and never knows the others' ranges, so anything
that must put every run on one scale is drawn here, by ``ffs_qc_report``, from
what the records (per-TR traces) and their sidecars (tSNR slices) kept. Running
the report again simply redraws them, so a run added later joins the scale.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .record import RECORDS_DIR, natural_key

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
