"""Host sets: run chosen units on another machine, bring the results back.

A host set is a self-contained bench root: its own ``worklist.sqlite`` holding copies of
units from the main worklist (same ids, keys and inputs), the manifests they need, and a
``meta`` table naming it. On the other host it runs like any root
(``FFS_BENCH_ROOT=hostset work -run ...``): units fetch their own inputs, outputs are kept
(only downloaded inputs are pruned), and every unit records the host it ran on.

Synced back, ``work -ingest DIR`` copies it under ``ROOT/hosts/<label>-<uuid8>/`` and
registers it; ingesting the same set again replaces it, so a re-sync after more units
finished is safe. The main worklist's own results are never touched: each host set stays
its own database, and ``-results`` reads them all side by side.
"""

from __future__ import annotations

import json
import shutil
import socket
import time
import uuid
from pathlib import Path

from .worklist import Worklist


def select_units(
    wl: Worklist,
    checks: list[str],
    n: int,
    manifests: dict[str, dict],
    datasets: list[str],
    include_pending: bool = False,
    bundle_local: bool = False,
    skip_ids: set[int] = frozenset(),  # type: ignore[assignment]
) -> tuple[list, list[str]]:
    """Up to ``n`` units per check, distinct datasets first, leaving out ``skip_ids``.

    Finished units by default: the point is a comparison with what this host already
    measured; ``include_pending`` adds units not run here yet. Datasets on a local tree
    cannot be fetched elsewhere, so they are skipped unless ``bundle_local`` copies their
    inputs into the host set."""
    picked, skipped = [], []
    states = ("done", "pending") if include_pending else ("done",)
    for check in checks:
        rows = wl.db.execute(
            f"SELECT * FROM units WHERE check_name=? AND status IN ({','.join('?' * len(states))})"
            " ORDER BY dataset, id",
            (check, *states),
        ).fetchall()
        by_ds: dict[str, list] = {}
        for r in rows:
            if (datasets and r["dataset"] not in datasets) or r["id"] in skip_ids:
                continue
            m = manifests.get(r["dataset"])
            if m is None or (m["source"]["kind"] != "s3" and not bundle_local):
                if r["dataset"] not in skipped:
                    skipped.append(r["dataset"])
                continue
            by_ds.setdefault(r["dataset"], []).append(r)
        chosen: list = []
        depth = 0
        while len(chosen) < n and any(len(v) > depth for v in by_ds.values()):
            for ds in sorted(by_ds):
                if len(chosen) < n and len(by_ds[ds]) > depth:
                    chosen.append(by_ds[ds][depth])
            depth += 1
        picked += chosen
    return picked, skipped


def export_hostset(
    wl: Worklist,
    manifests_dir: Path,
    out: Path,
    checks: list[str],
    n: int,
    label: str,
    datasets: list[str] = (),  # type: ignore[assignment]
    include_pending: bool = False,
    bundle_local: bool = False,
    prune_outputs: bool = False,
    repeat_elsewhere: bool = False,
) -> Path:
    import yaml

    from .census import load_manifests

    out = Path(out)
    if (out / "worklist.sqlite").exists():
        raise FileExistsError(f"{out} already holds a worklist; pick a new directory")
    manifests = {m["dataset_id"]: m for m in load_manifests(manifests_dir, curated=False)}
    # A unit another host already finished adds nothing on the next one: sets grow by new units.
    # A new architecture is the exception: re-timing the same units is what it is for.
    elsewhere = set() if repeat_elsewhere else done_on_hosts(wl)
    units, skipped = select_units(
        wl, checks, n, manifests, list(datasets), include_pending, bundle_local, elsewhere
    )
    if not units:
        raise ValueError(f"no exportable units for {checks}")
    (out / "manifests").mkdir(parents=True)
    hs = Worklist(out / "worklist.sqlite")
    with hs._tx():
        for u in units:
            hs.db.execute(
                "INSERT INTO units (id, check_name, dataset, unit_key, inputs, added_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (u["id"], u["check_name"], u["dataset"], u["unit_key"], u["inputs"], time.time()),
            )
    bundled_mb = 0.0
    for ds in sorted({u["dataset"] for u in units}):
        m = manifests[ds]
        if m["source"]["kind"] == "s3":
            shutil.copy2(manifests_dir / f"{ds}.yaml", out / "manifests" / f"{ds}.yaml")
            continue
        # Local tree: copy each unit's inputs (and same-stem sidecars) under DIR/data/<ds>/ and
        # point the manifest there; ${FFS_BENCH_ROOT} is expanded where the set runs.
        src_root = Path(m["source"]["uri"])
        for u in units:
            if u["dataset"] != ds:
                continue
            for rel in json.loads(u["inputs"]):
                for f in [src_root / rel, *(src_root / rel).parent.glob(_stem(rel) + ".json")]:
                    dst = out / "data" / ds / f.relative_to(src_root)
                    if f.is_file() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dst)
                        bundled_mb += f.stat().st_size / 1e6
        m = {
            **m,
            "source": {**m["source"], "uri": f"${{FFS_BENCH_ROOT}}/data/{ds}", "bundled": True},
        }
        (out / "manifests" / f"{ds}.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
    hs.set_meta(
        hostset=uuid.uuid4().hex,
        label=label,
        exported_from=socket.gethostname(),
        exported_at=time.strftime("%Y-%m-%d %H:%M"),
        checks=json.dumps(checks),
        prune_outputs="yes" if prune_outputs else "no",
    )
    hs.close()
    by_check: dict[str, int] = {}
    for u in units:
        by_check[u["check_name"]] = by_check.get(u["check_name"], 0) + 1
    print(f"{out}: {len(units)} units  " + "  ".join(f"{c} {k}" for c, k in by_check.items()))
    if bundled_mb:
        print(f"bundled local inputs: {bundled_mb:.0f} MB under {out / 'data'}")
    if skipped:
        print(f"skipped local-only datasets (use -bundle_local): {' '.join(skipped)}")
    return out


def done_on_hosts(wl: Worklist) -> set[int]:
    """Ids of units finished in any ingested host set."""
    ids: set[int] = set()
    for h in hostsets(wl):
        f = Path(h["path"]) / "worklist.sqlite"
        if f.exists():
            hs = Worklist(f)
            ids |= {r[0] for r in hs.db.execute("SELECT id FROM units WHERE status='done'")}
            hs.close()
    return ids


def _stem(rel: str) -> str:
    name = Path(rel).name
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _link_or_copy(src: str, dst: str) -> None:
    """Hard-link a host-set file into its ingested home; copy across filesystems.

    Unit outputs are never rewritten in place (rsync and the runner both write a new file
    and rename), so a link costs nothing and cannot alias a later change. The worklist is
    the exception: ingest rewrites its artifact paths, so it is always a real copy."""
    import os

    if Path(dst).exists():
        if os.path.samefile(src, dst):
            return
        os.unlink(dst)
    if not Path(src).name.startswith("worklist.sqlite"):
        try:
            os.link(src, dst)
            return
        except OSError:  # another filesystem, or links not supported
            pass
    shutil.copy2(src, dst)


def ingest(wl: Worklist, root: Path, src: Path) -> Path:
    """Register a synced host set under ROOT/hosts/ (files hard-linked, not duplicated);
    returns where it lives."""
    src = Path(src)
    hs = Worklist(src / "worklist.sqlite")
    hid, label = hs.meta("hostset"), hs.meta("label") or "host"
    hs.close()
    if not hid:
        raise ValueError(f"{src} is not a host set (no hostset id in its meta table)")
    dest = Path(root) / "hosts" / f"{label}-{hid[:8]}"
    if src.resolve() != dest.resolve():
        shutil.copytree(
            src,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("cache"),
            copy_function=_link_or_copy,
        )
    hs = Worklist(dest / "worklist.sqlite")
    # Artifact paths were written on the other host; point them at the copy.
    for a in hs.db.execute("SELECT unit_id, path FROM artifacts").fetchall():
        if "/units/" in a["path"]:
            new = str(dest / "units" / a["path"].split("/units/", 1)[1])
            hs.db.execute(
                "UPDATE artifacts SET path=? WHERE unit_id=? AND path=?",
                (new, a["unit_id"], a["path"]),
            )
    mismatched = [
        u["id"]
        for u in hs.db.execute("SELECT id, check_name, dataset, unit_key FROM units")
        if wl.db.execute(
            "SELECT 1 FROM units WHERE id=? AND check_name=? AND dataset=? AND unit_key=?",
            (u["id"], u["check_name"], u["dataset"], u["unit_key"]),
        ).fetchone()
        is None
    ]
    counts = dict(
        hs.db.execute("SELECT status, COUNT(*) FROM units GROUP BY status").fetchall()  # type: ignore[arg-type]
    )
    host = {
        r["key"]: r["value"]
        for r in hs.db.execute(
            "SELECT key, value FROM provenance WHERE side IN ('host', 'worker')"
            " AND key IN ('hostname', 'cpu_model', 'n_physical_cores') LIMIT 3"
        )
    }
    hs.close()
    wl.db.execute(
        "INSERT OR REPLACE INTO hostsets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            hid,
            label,
            str(dest),
            host.get("hostname"),
            host.get("cpu_model"),
            counts.get("done", 0),
            counts.get("failed", 0),
            time.time(),
        ),
    )
    print(
        f"ingested {label} ({hid[:8]}) from {host.get('hostname') or '?'}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f" -> {dest}"
    )
    if mismatched:
        print(f"warning: units {mismatched} do not match the main worklist (ids reused?)")
    return dest


def hostsets(wl: Worklist) -> list:
    return wl.db.execute("SELECT * FROM hostsets ORDER BY ingested_at").fetchall()
