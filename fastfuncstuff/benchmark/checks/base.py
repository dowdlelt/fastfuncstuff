"""What every check shares: dataset qualification, unit sampling, timed commands, provenance."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import census


@dataclass
class UnitResult:
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)  # judge -> {m: v}
    timings: list[tuple[str, str, float, int | None]] = field(default_factory=list)
    artifacts: list[tuple[Path, str]] = field(default_factory=list)  # (path, keep|regenerable)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)  # side -> {k: v}


def _stable_rank(*parts: str) -> str:
    """Order that does not reshuffle when the dataset grows (a rank cut would)."""
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def timed(cmd: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> float:
    """Run ``cmd``, send output to ``log``, return wall seconds; raise with the log tail."""
    t = time.monotonic()
    with open(log, "a") as f:
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        r = subprocess.run(
            cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}
        )
    dt = time.monotonic() - t
    if r.returncode != 0:
        tail = "".join(open(log).readlines()[-15:])
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}:\n{tail}")
    return dt


@lru_cache(maxsize=1)
def ffs_version() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        sha, dirty = "", ""
    return {"ffs_sha": sha, "ffs_dirty": "yes" if dirty else "no"}


@lru_cache(maxsize=8)
def afni_version(prog: str = "3dvolreg") -> str:
    try:
        out = subprocess.run([prog, "-ver"], capture_output=True, text=True).stdout
    except OSError:
        return ""
    return out.split("version=")[-1].split()[0] if "version=" in out else out.strip()[:80]


class Check:
    name = ""
    description = ""
    needs: list[list[str]] = [[]]  # alternatives; tags as in census.CORE_NEEDS
    sampling = "breadth"  # breadth | depth | session
    reference = ""  # the reference tool, for reports

    # -- planning ----------------------------------------------------------

    def qualifies(self, manifest: dict, acq: dict) -> bool:
        return any(census._qualifies(acq, manifest, need) for need in self.needs)

    def unit_inputs(self, files: list[census.BidsName], index: census.Index) -> list[str]:
        """Dataset-relative inputs for one run, given every image of that run."""
        raise NotImplementedError

    def plan(self, manifest: dict, index: census.Index, n: int) -> list[tuple[dict, list[str]]]:
        """Breadth sampling: up to ``n`` runs per qualifying acquisition, distinct subjects
        first, in a hash order that is stable as the dataset grows."""
        wanted = {a["id"] for a in manifest["acquisitions"] if self.qualifies(manifest, a)}
        runs: dict[str, dict[tuple, list[census.BidsName]]] = {}
        for nm in index.names:
            if nm.ext not in census._IMAGE_EXT or nm.datatype != "func":
                continue
            aid = census.acquisition_id(nm)
            if aid not in wanted:
                continue
            rk = tuple(nm.ent.get(k) for k in ("sub", "ses", "run"))
            runs.setdefault(aid, {}).setdefault(rk, []).append(nm)
        units = []
        ds = manifest["dataset_id"]
        for aid, by_run in sorted(runs.items()):
            order = sorted(by_run, key=lambda rk: _stable_rank(ds, aid, *map(str, rk)))
            picked, seen_sub = [], set()
            for rk in order:  # first pass: one run per subject
                if len(picked) < n and rk[0] not in seen_sub:
                    picked.append(rk)
                    seen_sub.add(rk[0])
            for rk in order:  # then fill with further runs
                if len(picked) < n and rk not in picked:
                    picked.append(rk)
            for rk in picked:
                files = by_run[rk]
                rep = files[0]
                key = {
                    "acq": aid,
                    "sub": rep.ent.get("sub"),
                    "ses": rep.ent.get("ses"),
                    "run": rep.ent.get("run"),
                }
                units.append((key, self.unit_inputs(files, index)))
        return units

    # -- running -----------------------------------------------------------

    def run(self, key: dict, inputs: dict[str, Path], work: Path, device: str) -> UnitResult:
        raise NotImplementedError
