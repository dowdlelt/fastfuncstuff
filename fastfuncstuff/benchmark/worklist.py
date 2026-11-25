"""Benchmark worklist: one SQLite file that holds what to run, what ran, and what it scored.

The unit of work is one check on one slice of one dataset (a run, a session, a subject
pair). "More subjects" means inserting rows; a runner claims the next pending unit,
fetches its inputs, runs reference and ffs, scores, prunes, and records everything here.
Rows are never deleted when files are pruned, so any result can be traced to the exact
inputs, versions and command that produced it, and regenerated from them.

Parallel workers (work -workers) each own their cores and thread budget, so timings stay
clean; they write a unit's rows between long tool runs, so SQLite's single writer is no limit.
Results from other machines arrive as host sets (hostset.py): separate databases registered
in ``hostsets``, never merged into these tables.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY,
    check_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    unit_key TEXT NOT NULL,          -- canonical JSON: {"sub":..,"ses":..,"task":..,"run":..}
    inputs TEXT NOT NULL,            -- JSON list of dataset-relative paths
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending | running | awaiting_ref | done | failed
    claimed_by TEXT,
    claimed_at REAL,
    finished_at REAL,
    error TEXT,
    added_at REAL NOT NULL,
    UNIQUE (check_name, dataset, unit_key)
);
CREATE TABLE IF NOT EXISTS provenance (
    unit_id INTEGER NOT NULL REFERENCES units(id),
    side TEXT NOT NULL,              -- ref | ffs | inputs | host
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (unit_id, side, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    unit_id INTEGER NOT NULL REFERENCES units(id),
    judge TEXT NOT NULL,             -- reference | truth | heldout
    metric TEXT NOT NULL,
    value REAL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (unit_id, judge, metric)
);
CREATE TABLE IF NOT EXISTS timings (
    unit_id INTEGER NOT NULL REFERENCES units(id),
    side TEXT NOT NULL,              -- ref | ffs
    device TEXT NOT NULL,            -- cpu | cuda | mps ; the reference is cpu
    seconds REAL NOT NULL,
    threads INTEGER,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (unit_id, side, device)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,            -- a host set's identity: hostset (uuid), label, ...
    value TEXT
);
CREATE TABLE IF NOT EXISTS hostsets (
    uuid TEXT PRIMARY KEY,           -- one row per ingested host set; re-ingest replaces it
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    hostname TEXT,
    cpu_model TEXT,
    n_done INTEGER,
    n_failed INTEGER,
    ingested_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    unit_id INTEGER NOT NULL REFERENCES units(id),
    path TEXT NOT NULL,
    bytes INTEGER,
    retention TEXT NOT NULL,         -- keep | regenerable
    pruned_at REAL,
    PRIMARY KEY (unit_id, path)
);
"""


def canonical_key(key: dict[str, Any]) -> str:
    return json.dumps({k: key[k] for k in sorted(key) if key[k] not in (None, "")}, sort_keys=True)


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class Worklist:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, **values: str) -> None:
        with self._tx():
            for k, v in values.items():
                self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v))

    @contextmanager
    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    # -- planning ----------------------------------------------------------

    def add_units(
        self, check: str, dataset: str, units: Iterable[tuple[dict[str, Any], list[str]]]
    ) -> int:
        """Insert (key, inputs) units; existing ones are left untouched. Returns # new."""
        now = time.time()
        n = 0
        with self._tx():
            for key, inputs in units:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO units (check_name, dataset, unit_key, inputs, added_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (check, dataset, canonical_key(key), json.dumps(inputs), now),
                )
                n += cur.rowcount
        return n

    # -- running -----------------------------------------------------------

    def claim_next(
        self,
        check: str | None = None,
        dataset: str | None = None,
        known_checks: Iterable[str] | None = None,
    ) -> sqlite3.Row | None:
        """Atomically take the next pending unit (optionally of one check / dataset).

        Round-robin over (check, dataset): every pair's first pending unit goes before any
        pair's second, so a night cut off at its stop time still covers every dataset and
        check instead of finishing one and never reaching the rest. ``known_checks`` leaves
        units of checks this runner cannot load (planned by newer code) pending."""
        where, args = ["status = 'pending'"], []
        if check:
            where.append("check_name = ?")
            args.append(check)
        if dataset:
            where.append("dataset = ?")
            args.append(dataset)
        if known_checks is not None:
            names = list(known_checks)
            where.append(f"check_name IN ({','.join('?' * len(names))})")
            args += names
        with self._tx():
            row = self.db.execute(
                # Rank over all of a pair's units, not just the pending ones, or the rank
                # restarts at 1 after every claim and the order collapses back to by-id.
                # A check's very first unit goes ahead of everything: a newly added check
                # is smoke-tested at the next unit boundary, not hours into the night.
                "SELECT * FROM (SELECT *,"
                " ROW_NUMBER() OVER (PARTITION BY check_name, dataset ORDER BY id) AS rr,"
                " ROW_NUMBER() OVER (PARTITION BY check_name ORDER BY id) AS cr FROM units)"
                f" WHERE {' AND '.join(where)} ORDER BY cr > 1, rr, id LIMIT 1",
                args,
            ).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE units SET status='running', claimed_by=?, claimed_at=?, error=NULL"
                " WHERE id=?",
                (worker_id(), time.time(), row["id"]),
            )
        return self.db.execute("SELECT * FROM units WHERE id=?", (row["id"],)).fetchone()

    def finish(self, unit_id: int, status: str = "done", error: str | None = None) -> None:
        self.db.execute(
            "UPDATE units SET status=?, finished_at=?, error=? WHERE id=?",
            (status, time.time(), error, unit_id),
        )

    def reset(self, status: str = "failed", check: str | None = None) -> int:
        """Return units in ``status`` (e.g. failed, or running after a crash) to pending."""
        q, args = "UPDATE units SET status='pending', claimed_by=NULL WHERE status=?", [status]
        if check:
            q += " AND check_name=?"
            args.append(check)
        return self.db.execute(q, args).rowcount

    # -- recording ---------------------------------------------------------

    def record_provenance(self, unit_id: int, side: str, values: dict[str, Any]) -> None:
        with self._tx():
            for k, v in values.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO provenance VALUES (?, ?, ?, ?)",
                    (unit_id, side, k, v if isinstance(v, str) else json.dumps(v)),
                )

    def record_metrics(self, unit_id: int, judge: str, values: dict[str, float | None]) -> None:
        now = time.time()
        with self._tx():
            for k, v in values.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (unit_id, judge, k, None if v is None else float(v), now),
                )

    def record_timing(
        self, unit_id: int, side: str, device: str, seconds: float, threads: int | None = None
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO timings VALUES (?, ?, ?, ?, ?, ?)",
            (unit_id, side, device, seconds, threads, time.time()),
        )

    def record_artifact(self, unit_id: int, path: Path, retention: str) -> None:
        p = Path(path)
        size = p.stat().st_size if p.exists() else None
        self.db.execute(
            "INSERT OR REPLACE INTO artifacts (unit_id, path, bytes, retention) VALUES (?, ?, ?, ?)",
            (unit_id, str(p), size, retention),
        )

    def prune(self, unit_id: int | None = None, under: Path | None = None) -> int:
        """Delete regenerable files (rows stay, stamped), only those below ``under`` if
        given (a host set keeps its outputs and drops just the input cache). Returns bytes
        freed."""
        q = "SELECT unit_id, path, bytes FROM artifacts WHERE retention='regenerable' AND pruned_at IS NULL"
        args: list[Any] = []
        if unit_id is not None:
            q += " AND unit_id=?"
            args.append(unit_id)
        # Parallel workers share the input cache: a file another running unit reads stays
        # until the last unit using it is pruned.
        busy = {
            (u["dataset"], rel)
            for u in self.db.execute(
                "SELECT dataset, inputs FROM units WHERE status='running' AND id IS NOT ?",
                (unit_id,),
            )
            for rel in json.loads(u["inputs"])
        }
        freed = 0
        for row in self.db.execute(q, args).fetchall():
            p = Path(row["path"])
            if any(p.as_posix().endswith(f"/{ds}/{rel}") for ds, rel in busy):
                continue
            if under is not None and Path(under) not in p.parents:
                continue
            if p.is_file():
                p.unlink()
                freed += row["bytes"] or 0
            self.db.execute(
                "UPDATE artifacts SET pruned_at=? WHERE unit_id=? AND path=?",
                (time.time(), row["unit_id"], row["path"]),
            )
        return freed

    # -- reading -----------------------------------------------------------

    def status(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT check_name, dataset, status, COUNT(*) AS n FROM units"
            " GROUP BY check_name, dataset, status ORDER BY check_name, dataset, status"
        ).fetchall()

    def results(self, check: str) -> list[dict[str, Any]]:
        """One dict per finished unit: key, metrics (judge.metric), timings (side.device)."""
        out = []
        for u in self.db.execute(
            "SELECT * FROM units WHERE check_name=? AND status='done' ORDER BY id", (check,)
        ):
            row: dict[str, Any] = {
                "unit_id": u["id"],
                "dataset": u["dataset"],
                **json.loads(u["unit_key"]),
            }
            for m in self.db.execute("SELECT * FROM metrics WHERE unit_id=?", (u["id"],)):
                row[f"{m['judge']}.{m['metric']}"] = m["value"]
            for t in self.db.execute("SELECT * FROM timings WHERE unit_id=?", (u["id"],)):
                row[f"seconds.{t['side']}.{t['device']}"] = t["seconds"]
            out.append(row)
        return out
