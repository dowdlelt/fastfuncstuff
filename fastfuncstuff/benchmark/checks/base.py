"""What every check shares: dataset qualification, unit sampling, timed commands, provenance."""

from __future__ import annotations

import hashlib
import json
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


def timed(
    cmd: list[str],
    cwd: Path,
    log: Path,
    env: dict[str, str] | None = None,
    timeout_s: float = 3600.0,
) -> float:
    """Run ``cmd``, send output to ``log``, return wall seconds; raise with the log tail.

    The command gets its own process group and is killed as a group at ``timeout_s``:
    MATLAB (and AFNI scripts) fork helpers, and a hung child holding the parent open is
    exactly how an unattended night stalls on one unit.
    """
    t = time.monotonic()
    with open(log, "a") as f:
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **(env or {})},
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            f.write(f"!! killed after {timeout_s:.0f} s\n")
            raise TimeoutError(
                f"{cmd[0]} exceeded {timeout_s:.0f} s (process group killed)"
            ) from None
        except BaseException:
            kill_group(proc)
            raise
    dt = time.monotonic() - t
    if rc != 0:
        tail = "".join(open(log).readlines()[-15:])
        raise RuntimeError(f"{cmd[0]} exited {rc}:\n{tail}")
    with open(log, "a") as f:
        f.write(f"# wall {dt:.3f} s\n")
    return dt


def timed_cached(
    cmd: list[str],
    stamp: Path,
    outputs: list[Path],
    cwd: Path,
    log: Path,
    timeout_s: float = 3600.0,
) -> float:
    """``timed``, but a rerun whose ``stamp`` records this exact command and whose
    ``outputs`` exist returns the recorded seconds instead of fitting again.

    For fits that take hours (3dQwarp at minpatch 9): a unit that fails while scoring
    should not pay for the fit twice, and the timing must be the fit's, not a cache hit.
    """
    import json

    if stamp.exists() and all(o.exists() for o in outputs):
        rec = json.loads(stamp.read_text())
        if rec.get("cmd") == cmd:
            with open(log, "a") as f:
                f.write(f"$ (reused, {rec['seconds']:.1f} s recorded) {shlex.join(cmd)}\n")
            return float(rec["seconds"])
    dt = timed(cmd, cwd, log, timeout_s=timeout_s)
    stamp.write_text(json.dumps({"cmd": cmd, "seconds": dt}))
    return dt


NOISE_FRAC = 0.25


def volume_means(image: Path) -> list[float]:
    """Whole-volume mean of every volume (3dmaskave streams it: ~4 s for 353 volumes)."""
    out = subprocess.run(["3dmaskave", "-quiet", str(image)], capture_output=True, text=True,
                         check=True, timeout=1800).stdout  # fmt: skip
    return [float(ln.split()[0]) for ln in out.splitlines() if ln.strip()]


def trailing_noise_count(means: list[float], frac: float = NOISE_FRAC) -> int:
    if len(means) < 4:
        return 0
    med = sorted(means)[len(means) // 2]
    k = 0
    while k < len(means) - 1 and means[-1 - k] < frac * med:
        k += 1
    return k


def kill_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left after ``grace_s``."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


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
    timeout_s = 3600.0  # per command; a hung tool is killed and the unit marked failed
    # Set by the runner per unit from the manifest's curated block (census.unit_curation):
    # {"sidecar": {BIDS key: value}, "noise_volumes": int | None, "source": str}.
    curation: dict = {}

    # -- inputs ------------------------------------------------------------

    def read_meta(self, inputs: dict[str, Path], exclude: tuple[str, ...] = ("_physio",)) -> dict:
        """The run's sidecars merged shallowest first (BIDS inheritance), then the curated
        overrides: a private tree whose sidecar is wrong (MassedEvents: RepetitionTime is
        the 3D shot TR) is corrected in the manifest, never in the data."""
        meta: dict = {}
        for rel, p in inputs.items():
            if rel.endswith(".json") and not any(x in rel for x in exclude):
                meta.update(json.loads(p.read_text()))
        meta.update(self.curation.get("sidecar") or {})
        return meta

    def file_overrides(self, rel: str) -> dict:
        """Curated ``sidecar_files`` rules matching this image's BIDS entities (``dir-AP``)
        or suffix; later rules win."""
        name = Path(rel).name.split(".")[0]
        parts = name.split("_")
        ents = dict(p.split("-", 1) for p in parts[:-1] if "-" in p)
        ents["suffix"] = parts[-1]
        out: dict = {}
        for rule in self.curation.get("sidecar_files") or []:
            if all(ents.get(k) == str(v) for k, v in (rule.get("match") or {}).items()):
                out.update(rule.get("set") or {})
        return out

    def trim_noise(
        self, image: Path, work: Path, log: Path, res: UnitResult, also: tuple[Path, ...] = ()
    ) -> tuple[Path, tuple[Path, ...]]:
        """Drop trailing noise volumes (RF-off scans some private protocols append: HiHiFaces
        3, at ~8% of the run's mean) from ``image`` and the same count from ``also`` (its
        phase). A curated count wins; otherwise trailing volumes under NOISE_FRAC of the
        median volume mean are taken as noise. Both tools then get the trimmed copy."""
        curated = self.curation.get("noise_volumes")
        means = volume_means(image)
        detected = trailing_noise_count(means)
        k = int(curated) if curated is not None else detected
        res.provenance["noise_volumes"] = {
            "dropped": k, "detected": detected, "curated": curated, "n_volumes": len(means),
            "rule": f"trailing volume mean < {NOISE_FRAC} x median (when not curated)",
        }  # fmt: skip
        if k <= 0:
            return image, also
        if k >= len(means) - 2:
            raise ValueError(f"{k} noise volumes of {len(means)}: nothing left to compare")
        out = []
        for i, src in enumerate((image, *also)):
            dst = work / f"input{i}_trim{k}.nii.gz"
            timed(["3dTcat", "-overwrite", "-prefix", str(dst),
                   f"{src}[0..{len(means) - k - 1}]"], work, log)  # fmt: skip
            res.artifacts.append((dst, "regenerable"))
            out.append(dst)
        return out[0], tuple(out[1:])

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
