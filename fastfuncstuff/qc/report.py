"""One QC folder -> a subject table (TSV + text) and an HTML page of its PNGs.

The PNGs carry the story on their own; the page only stitches them together in
folder order with a table of the numbers on top, so it costs nothing to build
and nothing is lost by skipping it and scrolling the folder instead.
"""

from __future__ import annotations

import html
import math
import os
from pathlib import Path

from .record import load_records, metric_flagged, natural_key, warn_text

TSV_NAME = "qc_review.tsv"
TXT_NAME = "qc_review.txt"
HTML_NAME = "index.html"


def fmt_value(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "NA"
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1000:
            return f"{v:.0f}"
        if a >= 10:
            return f"{v:.2f}"
        return f"{v:.4g}"
    return str(v)


def table_rows(records: list[dict]) -> tuple[list[str], list[dict]]:
    """Wide rows, one per label, with ``tool.metric`` columns in pipeline order."""
    columns: list[str] = []
    by_label: dict[str, dict] = {}
    for rec in records:
        row = by_label.setdefault(rec["label"], {"label": rec["label"]})
        for name, m in rec["metrics"].items():
            col = f"{rec['tool']}.{name}"
            if col not in columns:
                columns.append(col)
            row[col] = m
    labels = sorted(by_label, key=natural_key)
    return columns, [by_label[k] for k in labels]


def write_tables(qc_dir: Path, records: list[dict]) -> tuple[Path, Path]:
    columns, rows = table_rows(records)
    tsv = qc_dir / TSV_NAME
    with open(tsv, "w") as f:
        f.write("\t".join(["label", *columns]) + "\n")
        for row in rows:
            vals = [fmt_value(row[c]["value"]) if c in row else "NA" for c in columns]
            f.write("\t".join([row["label"], *vals]) + "\n")
    # AFNI ss_review style: one "key : value" block per record, readable in a terminal.
    txt = qc_dir / TXT_NAME
    lines: list[str] = []
    for rec in records:
        lines.append(f"== {rec['tool']} : {rec['label']}")
        for k, v in rec.get("info", {}).items():
            lines.append(f"  {k:<22s}: {v}")
        width = max((len(k) for k in rec["metrics"]), default=10)
        for name, m in rec["metrics"].items():
            unit = f" {m['unit']}" if m.get("unit") else ""
            flag = f"   ** WARN ({warn_text(m.get('warn'))})" if metric_flagged(m) else ""
            lines.append(f"  {name:<{width}s} : {fmt_value(m['value'])}{unit}{flag}")
        lines.append("")
    txt.write_text("\n".join(lines))
    return tsv, txt


_CSS = """
:root{color-scheme:dark;--bg:#1a1a19;--panel:#20201f;--grid:#383835;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--warn:#fab219}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "DejaVu Sans",system-ui,sans-serif}
nav{position:fixed;top:0;left:0;bottom:0;width:230px;overflow:auto;padding:16px;background:var(--panel);border-right:1px solid var(--grid)}
nav a{display:block;color:var(--ink2);text-decoration:none;padding:2px 0}
nav a.tool{color:var(--ink);font-weight:bold;margin-top:10px}
nav a:hover{color:var(--ink)}
main{margin-left:230px;padding:16px 24px;max-width:1700px}
h1{font-size:22px;margin:4px 0 2px} h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--grid);padding-bottom:4px}
h3{font-size:14px;color:var(--ink2);margin:18px 0 6px}
.sub{color:var(--muted)}
.flags{background:var(--panel);border-left:4px solid var(--warn);padding:8px 12px;margin:12px 0}
table{border-collapse:collapse;font-size:12px;margin:6px 0 12px;display:block;overflow-x:auto}
th,td{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:normal;position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child{text-align:left}
td.warn{color:var(--warn);font-weight:bold}
img{width:100%;height:auto;display:block;margin:0 0 14px;border-radius:4px}
@media (max-width:900px){nav{display:none}main{margin-left:0}}
"""


def write_html(qc_dir: Path, records: list[dict], title: str = "") -> Path:
    """Stitch every PNG in the folder (records' figures first, in order) into one page."""
    e = html.escape
    title = title or qc_dir.resolve().parent.name or "QC"
    pngs = sorted((p.name for p in qc_dir.glob("*.png")), key=natural_key)
    claimed: set[str] = set()

    # Group figures by tool, then by check step, runs side by side in label order.
    tools: dict[str, dict] = {}
    for rec in records:
        t = tools.setdefault(rec["tool"], {"order": rec["order"], "checks": {}, "records": []})
        t["records"].append(rec)
        for fig in rec.get("figures", []):
            if fig["file"] not in pngs:
                continue
            key = (fig["step"], fig["check"])
            t["checks"].setdefault(key, {"title": fig["title"], "files": []})["files"].append(
                fig["file"]
            )
            claimed.add(fig["file"])

    flagged = []
    for rec in records:
        for name, m in rec["metrics"].items():
            if metric_flagged(m):
                flagged.append(
                    f"{rec['tool']}.{name} = {fmt_value(m['value'])} ({warn_text(m.get('warn'))}) in {rec['label']}"
                )

    nav, body = [], []
    body.append(
        f"<h1>{e(title)}</h1><div class='sub'>{len(records)} records · {len(pngs)} figures · {e(str(qc_dir))}</div>"
    )
    if flagged:
        body.append(
            "<div class='flags'><b>Check:</b><br>" + "<br>".join(e(f) for f in flagged) + "</div>"
        )
    for tool, t in sorted(tools.items(), key=lambda kv: (kv[1]["order"], kv[0])):
        anchor = f"t-{tool}"
        nav.append(f"<a class='tool' href='#{anchor}'>{e(tool)}</a>")
        body.append(f"<h2 id='{anchor}'>{e(tool)}</h2>")
        body.append(_metrics_table(t["records"]))
        for (step, check), c in sorted(t["checks"].items()):
            cid = f"{anchor}-{step}-{check}"
            nav.append(f"<a href='#{cid}'>{e(c['title'])}</a>")
            body.append(f"<h3 id='{cid}'>{e(c['title'])}</h3>")
            for fn in sorted(c["files"], key=natural_key):
                body.append(f"<img loading='lazy' src='{e(fn)}' alt='{e(fn)}'>")
    others = [p for p in pngs if p not in claimed]
    if others:
        nav.append("<a class='tool' href='#t-other'>other</a>")
        body.append("<h2 id='t-other'>other figures</h2>")
        body += [f"<h3>{e(p)}</h3><img loading='lazy' src='{e(p)}' alt='{e(p)}'>" for p in others]
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{e(title)} QC</title><style>{_CSS}</style></head><body>"
        f"<nav><b>{e(title)}</b>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>"
    )
    out = qc_dir / HTML_NAME
    out.write_text(page)
    return out


def _metrics_table(records: list[dict]) -> str:
    e = html.escape
    cols: list[str] = []
    meta: dict[str, dict] = {}
    for rec in records:
        for k, m in rec["metrics"].items():
            if k not in cols:
                cols.append(k)
                meta[k] = m
    if not cols:
        return ""
    head = "".join(
        f"<th title='{e(meta[c].get('desc', ''))}'>{e(c)}{(' (' + e(meta[c]['unit']) + ')') if meta[c].get('unit') else ''}</th>"
        for c in cols
    )
    rows = []
    for rec in sorted(records, key=lambda r: natural_key(r["label"])):
        if not rec["metrics"]:  # a cross-run summary: figures only
            continue
        cells = []
        for c in cols:
            m = rec["metrics"].get(c)
            if m is None:
                cells.append("<td>NA</td>")
            elif metric_flagged(m):
                cells.append(
                    f"<td class='warn' title='limit {e(warn_text(m.get('warn')))}'>&#9888; {e(fmt_value(m['value']))}</td>"
                )
            else:
                cells.append(f"<td>{e(fmt_value(m['value']))}</td>")
        rows.append(f"<tr><td>{e(rec['label'])}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>run</th>{head}</tr>{''.join(rows)}</table>"


def build_report(qc_dir: str | os.PathLike, title: str = "", html_page: bool = True) -> dict:
    from .summary import motion_summary, shared_tsnr

    d = Path(qc_dir)
    records = load_records(d)
    tsv, txt = write_tables(d, records)
    # Cross-run figures: drawn before the page so it stitches them in.
    summaries = motion_summary(d, records)
    out = {"records": len(records), "tsv": tsv, "txt": txt, "tsnr_shared": shared_tsnr(d)}
    out["summaries"] = [f["file"] for s in summaries for f in s["figures"]]
    if html_page:
        out["html"] = write_html(d, records + summaries, title)
    return out
