"""Dataset census: what a BIDS dataset offers the benchmark, read from the files.

Builds the acquisition part of a dataset manifest (field strength, voxel size, TR,
multiband, echoes, phase, recorded physio, events, fieldmaps, anatomy) from sidecars
and one sampled NIfTI header per acquisition group. Nothing is typed by hand, because
a hand-typed TR or voxel size is exactly the kind of fact that drifts from the data.

Works on a local tree or on OpenNeuro's public S3 bucket without credentials and
without downloading images: listings come from the S3 REST API, headers from a ranged
GET of the first 64 KiB of a ``.nii.gz``.

The manifest is consumed by the benchmark: checks declare what they need ("2-D EPI
with phase and recorded cardiac") and qualify datasets by ``offers``.
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

OPENNEURO_BUCKET = "openneuro.org"
_HEADER_BYTES = 65536
_SKIP_TOP = {"derivatives", "sourcedata", "code", "stimuli", "models"}
_IMAGE_EXT = (".nii.gz", ".nii")
_FUNC_SUFFIXES = {"bold", "cbv", "asl", "phase"}  # legacy BIDS: _phase beside _bold

# Entities that identify one recording rather than an acquisition protocol.
_RUN_ENTITIES = {"sub", "ses", "run"}


# ---------------------------------------------------------------------------
# File sources
# ---------------------------------------------------------------------------


class LocalSource:
    """A BIDS tree on disk. ``root`` is the directory holding ``sub-*``."""

    kind = "local"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.uri = str(self.root)

    def list_files(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == ".":
                dirnames[:] = [d for d in dirnames if d not in _SKIP_TOP and not d.startswith(".")]
            else:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    size = os.stat(p).st_size  # follows symlinks: unfetched annex files raise
                except OSError:
                    continue
                rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
                out[rel.replace(os.sep, "/")] = size
        return out

    def top_dirs(self, name: str) -> list[str]:
        d = self.root / name
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def read(self, rel: str, max_bytes: int | None = None) -> bytes:
        with open(self.root / rel, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)


class S3Source:
    """An OpenNeuro dataset on the public S3 bucket, read anonymously."""

    kind = "s3"

    def __init__(self, dataset_id: str, bucket: str = OPENNEURO_BUCKET):
        self.dataset_id = dataset_id
        self.bucket = bucket
        self.prefix = dataset_id.rstrip("/") + "/"
        self.uri = f"s3://{bucket}/{dataset_id}"
        # Path-style: the bucket name has a dot, which breaks virtual-host TLS.
        self._host = f"https://s3.amazonaws.com/{bucket}"

    def _list(self, prefix: str, delimiter: str | None = None) -> tuple[dict[str, int], list[str]]:
        files: dict[str, int] = {}
        prefixes: list[str] = []
        token = None
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            url = f"{self._host}/?{urllib.parse.urlencode(q)}"
            with urllib.request.urlopen(url, timeout=60) as r:
                root = ET.fromstring(r.read())
            for c in root.findall("s3:Contents", ns):
                key = c.findtext("s3:Key", default="", namespaces=ns)
                files[key[len(self.prefix) :]] = int(
                    c.findtext("s3:Size", default="0", namespaces=ns)
                )
            for p in root.findall("s3:CommonPrefixes", ns):
                prefixes.append(p.findtext("s3:Prefix", default="", namespaces=ns))
            if root.findtext("s3:IsTruncated", default="false", namespaces=ns) != "true":
                break
            token = root.findtext("s3:NextContinuationToken", namespaces=ns)
        return files, prefixes

    def list_files(self) -> dict[str, int]:
        files, _ = self._list(self.prefix)
        return {
            k: v
            for k, v in files.items()
            if k and k.split("/", 1)[0] not in _SKIP_TOP and not k.startswith(".")
        }

    def top_dirs(self, name: str) -> list[str]:
        _, prefixes = self._list(f"{self.prefix}{name}/", delimiter="/")
        return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)

    def read(self, rel: str, max_bytes: int | None = None) -> bytes:
        url = f"{self._host}/{urllib.parse.quote(self.prefix + rel)}"
        req = urllib.request.Request(url)
        if max_bytes is not None:
            req.add_header("Range", f"bytes=0-{max_bytes - 1}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()


def find_bids_root(path: Path, max_depth: int = 3) -> Path:
    """The directory under ``path`` that holds ``sub-*`` folders (lab trees nest it)."""
    path = Path(path)
    frontier = [path]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            try:
                kids = [k for k in d.iterdir() if k.is_dir() and not k.name.startswith(".")]
            except OSError:
                continue
            if any(k.name.startswith("sub-") for k in kids):
                return d
            nxt.extend(k for k in kids if k.name not in _SKIP_TOP)
        frontier = nxt
    raise FileNotFoundError(f"no sub-* directories within {max_depth} levels of {path}")


def open_source(spec: str) -> LocalSource | S3Source:
    """``dsNNNNNN`` (OpenNeuro S3), ``s3://openneuro.org/dsNNNNNN``, or a local path."""
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[5:].partition("/")
        return S3Source(prefix.strip("/"), bucket=bucket)
    if re.fullmatch(r"ds\d{6}", spec) and not Path(spec).exists():
        return S3Source(spec)
    return LocalSource(find_bids_root(Path(spec)))


# ---------------------------------------------------------------------------
# BIDS names and inheritance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BidsName:
    rel: str
    entities: tuple[tuple[str, str], ...]
    suffix: str
    ext: str

    @property
    def ent(self) -> dict[str, str]:
        return dict(self.entities)

    @property
    def dir(self) -> str:
        return self.rel.rsplit("/", 1)[0] if "/" in self.rel else ""

    @property
    def datatype(self) -> str:
        parts = self.rel.split("/")
        return parts[-2] if len(parts) >= 2 else ""


def parse_name(rel: str) -> BidsName | None:
    base = rel.rsplit("/", 1)[-1]
    if "." not in base:
        return None
    dot = base.index(".")
    stem, ext = base[:dot], base[dot:]
    parts = stem.split("_")
    if not parts or "-" in parts[-1]:
        return None
    ents = []
    for p in parts[:-1]:
        k, sep, v = p.partition("-")
        if not sep:
            return None
        ents.append((k, v))
    return BidsName(rel, tuple(ents), parts[-1], ext)


def _applies(meta: BidsName, target: BidsName) -> bool:
    """BIDS inheritance: same suffix, entities a subset, directory an ancestor."""
    if meta.suffix != target.suffix:
        return False
    if meta.dir and not (target.dir == meta.dir or target.dir.startswith(meta.dir + "/")):
        return False
    t = target.ent
    return all(t.get(k) == v for k, v in meta.entities)


class Index:
    """Parsed listing with sidecar lookup."""

    def __init__(self, files: dict[str, int]):
        self.sizes = files
        self.names = [n for n in (parse_name(r) for r in sorted(files)) if n is not None]
        self.by_ext: dict[str, list[BidsName]] = defaultdict(list)
        for n in self.names:
            self.by_ext[n.ext].append(n)

    def matching(self, target: BidsName, ext: str, suffix: str | None = None) -> list[BidsName]:
        """Metadata files of ``ext`` that apply to ``target``, shallowest first."""
        probe = (
            target if suffix is None else BidsName(target.rel, target.entities, suffix, target.ext)
        )
        hits = [m for m in self.by_ext.get(ext, []) if _applies(m, probe)]
        hits.sort(key=lambda m: (m.dir.count("/") + bool(m.dir), len(m.entities)))
        return hits


def _sidecar(src, index: Index, target: BidsName, cache: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for m in index.matching(target, ".json"):
        if m.rel not in cache:
            try:
                cache[m.rel] = json.loads(src.read(m.rel))
            except (OSError, ValueError):
                cache[m.rel] = {}
        if isinstance(cache[m.rel], dict):
            merged.update(cache[m.rel])
    return merged


# ---------------------------------------------------------------------------
# NIfTI header from the first bytes
# ---------------------------------------------------------------------------


def header_from_bytes(raw: bytes, gz: bool) -> dict[str, Any] | None:
    """Shape and voxel size from the start of a (possibly truncated) NIfTI file."""
    import nibabel as nib

    if gz:
        try:
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
        except zlib.error:
            return None
    if len(raw) < 348:
        return None
    sizeof = int.from_bytes(raw[:4], "little")
    if sizeof not in (348, 540):
        sizeof = int.from_bytes(raw[:4], "big")
    cls = nib.Nifti2Header if sizeof == 540 else nib.Nifti1Header
    try:
        hdr = cls.from_fileobj(io.BytesIO(raw[:sizeof]))
    except Exception:
        return None
    shape = [int(s) for s in hdr.get_data_shape()]
    zooms = [round(float(z), 4) for z in hdr.get_zooms()]
    return {
        "matrix": shape[:3],
        "n_vols": shape[3] if len(shape) > 3 else 1,
        "voxel_mm": zooms[:3],
    }


def read_header(src, rel: str) -> dict[str, Any] | None:
    try:
        raw = src.read(rel, max_bytes=_HEADER_BYTES)
    except OSError:
        return None
    return header_from_bytes(raw, gz=rel.endswith(".gz"))


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def _multiband(meta: dict[str, Any]) -> int | None:
    mb = meta.get("MultibandAccelerationFactor")
    if isinstance(mb, (int, float)) and mb >= 1:
        return int(mb)
    st = meta.get("SliceTiming")
    if isinstance(st, list) and st:
        n_unique = len({round(float(t), 4) for t in st})
        if n_unique:
            return max(1, len(st) // n_unique)
    return None


def _physio_columns(src, index: Index, physio: BidsName, cache: dict) -> set[str]:
    meta = _sidecar(src, index, BidsName(physio.rel, physio.entities, "physio", ".json"), cache)
    kinds = set()
    for c in meta.get("Columns", []) or []:
        c = str(c).lower()
        if any(k in c for k in ("cardiac", "pulse", "ppg", "ecg", "puls")):
            kinds.add("cardiac")
        elif "resp" in c:
            kinds.add("respiratory")
    return kinds


def _group_key(n: BidsName) -> tuple:
    """Acquisition identity: everything but subject/session/run, echo and part."""
    suffix = "bold" if n.suffix == "phase" else n.suffix
    return tuple((k, v) for k, v in n.entities if k not in _RUN_ENTITIES | {"echo", "part"}) + (
        ("_suffix", suffix),
    )


def acquisition_id(n: BidsName) -> str:
    """The manifest's name for the acquisition a file belongs to."""
    key = _group_key(n)
    ent = "_".join(f"{k}-{v}" for k, v in key if not k.startswith("_"))
    return f"{ent}_{dict(key)['_suffix']}"


def _is_phase(n: BidsName) -> bool:
    return n.ent.get("part") == "phase" or n.suffix == "phase"


def census(src) -> dict[str, Any]:
    """Describe what the dataset at ``src`` offers. Returns a manifest dict."""
    files = src.list_files()
    index = Index(files)
    cache: dict[str, Any] = {}

    desc = {}
    if "dataset_description.json" in files:
        try:
            desc = json.loads(src.read("dataset_description.json"))
        except (OSError, ValueError):
            desc = {}

    images = [n for n in index.names if n.ext in _IMAGE_EXT]
    subjects = sorted({n.ent["sub"] for n in images if "sub" in n.ent})

    physio_files = [n for n in index.names if n.suffix == "physio" and n.ext == ".tsv.gz"]
    events_files = [n for n in index.names if n.suffix == "events" and n.ext == ".tsv"]

    groups: dict[tuple, list[BidsName]] = defaultdict(list)
    for n in images:
        if n.datatype == "func" and n.suffix in _FUNC_SUFFIXES:
            groups[_group_key(n)].append(n)

    acquisitions: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        ent = {k: v for k, v in key if not k.startswith("_")}
        suffix = dict(key)["_suffix"]
        mags = [m for m in members if not _is_phase(m)]
        phases = [m for m in members if _is_phase(m)]
        sample = (mags or members)[0]
        meta = _sidecar(src, index, sample, cache)

        echoes = sorted(
            {m.ent.get("echo", "1") for m in mags}, key=lambda e: int(e) if e.isdigit() else 0
        )
        te_ms = []
        for e in echoes:
            em = next(m for m in mags if m.ent.get("echo", "1") == e)
            et = _sidecar(src, index, em, cache).get("EchoTime")
            te_ms.append(round(float(et) * 1000, 2) if isinstance(et, (int, float)) else None)

        # One recording = one (sub, ses, run) of this acquisition.
        recordings = {tuple(m.ent.get(k) for k in ("sub", "ses", "run")) for m in mags}
        rec_key = lambda n: tuple(n.ent.get(k) for k in ("sub", "ses", "run"))  # noqa: E731

        physio_kinds: set[str] = set()
        n_physio = 0
        for rec in recordings:
            rep = next(m for m in mags if rec_key(m) == rec)
            hits = [
                p
                for p in physio_files
                if all(p.ent.get(k) == v for k, v in rep.entities if k not in {"echo", "part"})
            ]
            if hits:
                n_physio += 1
                if not physio_kinds:
                    for h in hits:
                        physio_kinds |= _physio_columns(src, index, h, cache)

        n_events = 0
        if events_files:
            for rec in recordings:
                rep = next(m for m in mags if rec_key(m) == rec)
                if index.matching(
                    BidsName(
                        rep.rel,
                        tuple((k, v) for k, v in rep.entities if k not in {"echo", "part"}),
                        rep.suffix,
                        rep.ext,
                    ),
                    ".tsv",
                    suffix="events",
                ):
                    n_events += 1

        hdr = read_header(src, sample.rel)
        mb = _multiband(meta)
        st = meta.get("SliceTiming")
        acq_type = meta.get("MRAcquisitionType")
        if acq_type is None and isinstance(st, list):
            acq_type = "2D" if len({round(float(t), 4) for t in st}) > 1 else "3D"
        field_t = meta.get("MagneticFieldStrength")

        task = ent.get("task", "")
        offers = []
        if n_events:
            offers.append("events")
        if "rest" in task.lower():
            offers.append("rest")
        elif not n_events:
            offers.append("no_events")  # a task run whose timing isn't in BIDS yet
        if phases:
            offers.append("phase")
        if len(echoes) > 1:
            offers.append("multi_echo")
        if "cardiac" in physio_kinds:
            offers.append("physio_cardiac")
        if "respiratory" in physio_kinds:
            offers.append("physio_resp")
        if isinstance(st, list) and st:
            offers.append("slice_timing")
        if acq_type in ("2D", "3D"):
            offers.append(f"epi_{acq_type.lower()}")
        if suffix == "cbv" or "vaso" in ent.get("acq", "").lower():
            offers.append("vaso")
        if mb and mb > 1:
            offers.append("multiband")

        acquisitions.append(
            {
                "id": acquisition_id(sample),
                "suffix": suffix,
                "entities": ent,
                "n_subjects": len({r[0] for r in recordings}),
                "n_runs": len(recordings),
                "gb": round(sum(files[m.rel] for m in members) / 1e9, 2),
                "field_T": field_t,
                "manufacturer": meta.get("Manufacturer"),
                "model": meta.get("ManufacturersModelName"),
                "tr_s": meta.get("RepetitionTime"),
                "te_ms": te_ms,
                "multiband": mb,
                "inplane_accel": meta.get("ParallelReductionFactorInPlane"),
                "acq_type": acq_type,
                "pe_dir": meta.get("PhaseEncodingDirection"),
                **(hdr or {}),
                "n_runs_physio": n_physio,
                "n_runs_events": n_events,
                "offers": offers,
                "sampled": sample.rel,
                "_meta_keys": sorted(meta),
            }
        )

    anat: dict[str, int] = defaultdict(int)
    fmap: dict[str, int] = defaultdict(int)
    for n in images:
        if n.datatype == "anat":
            anat[n.suffix] += 1
        elif n.datatype == "fmap":
            fmap[n.suffix] += 1

    ds_offers = sorted({o for a in acquisitions for o in a["offers"]})
    if anat.get("T1w") or anat.get("UNIT1"):
        ds_offers.append("anat_t1")
    if anat.get("UNIT1") or anat.get("MP2RAGE"):
        ds_offers.append("anat_mp2rage")
    if fmap.get("epi"):
        ds_offers.append("fmap_epi")
    if fmap.get("phasediff") or fmap.get("fieldmap") or fmap.get("phase1"):
        ds_offers.append("fmap_gre")

    if isinstance(src, S3Source):
        dataset_id = src.dataset_id
    else:
        m = re.search(r"(ds\d{6})", str(src.root))
        dataset_id = m.group(1) if m else _local_name(src.root)

    return {
        "dataset_id": dataset_id,
        "name": desc.get("Name", ""),
        "license": desc.get("License", ""),
        "source": {"kind": src.kind, "uri": src.uri, "private": not isinstance(src, S3Source)},
        "census_date": date.today().isoformat(),
        "n_subjects": len(subjects),
        "gb_total": round(sum(files.values()) / 1e9, 1),
        "offers": ds_offers,
        "acquisitions": acquisitions,
        "anat": dict(anat),
        "fmap": dict(fmap),
        "derivatives": src.top_dirs("derivatives"),
        "bids_issues": bids_issues(files, index, acquisitions, desc, physio_files),
    }


# Sidecar fields a benchmark check reads. Missing ones silently drop a dataset from checks.
_REQUIRED_FUNC_META = {
    "RepetitionTime": "every check",
    "MagneticFieldStrength": "paper axes",
    "EchoTime": "multi-echo, physio, phase checks",
    "SliceTiming": "slicetime, physio, slice-pose moco (2-D EPI)",
    "PhaseEncodingDirection": "blipflip, b0fmap, physio field direction",
    "TotalReadoutTime": "blipflip / topup",
    "Manufacturer": "paper tables",
}


def bids_issues(
    files: dict[str, int],
    index: Index,
    acquisitions: list[dict],
    desc: dict,
    physio: list[BidsName],
) -> list[str]:
    """What this tree needs before checks can find everything in it. Empty = nothing seen."""
    out = []
    if not desc:
        out.append("no dataset_description.json at the root (Name, BIDSVersion)")
    unparsed = [r for r in files if r.endswith(_IMAGE_EXT) and parse_name(r) is None]
    if unparsed:
        out.append(f"{len(unparsed)} images with non-BIDS names, e.g. {unparsed[0]}")
    loose = [
        n.rel
        for n in index.names
        if n.ext in _IMAGE_EXT and n.datatype not in {"func", "anat", "fmap", "dwi", "perf", "beh"}
    ]
    if loose:
        out.append(f"{len(loose)} images outside func/anat/fmap folders, e.g. {loose[0]}")
    no_task = [
        n.rel
        for n in index.names
        if n.datatype == "func" and n.ext in _IMAGE_EXT and "task" not in n.ent
    ]
    if no_task:
        out.append(f"{len(no_task)} func images without task-, e.g. {no_task[0]}")
    for a in acquisitions:
        missing = [
            k
            for k in _REQUIRED_FUNC_META
            if a.get("_meta_keys") is not None and k not in a["_meta_keys"]
        ]
        if "TotalReadoutTime" in missing and "EffectiveEchoSpacing" in a.get("_meta_keys", ()):
            missing.remove("TotalReadoutTime")
        if a["acq_type"] == "3D" and "SliceTiming" in missing:
            missing.remove("SliceTiming")
        if missing:
            out.append(f"{a['id']}: sidecar lacks {', '.join(missing)}")
        task = a["entities"].get("task", "")
        if task and "events" not in a["offers"] and "rest" not in task.lower():
            out.append(f"{a['id']}: task run with no _events.tsv")
    stray_physio = [
        r
        for r in files
        if re.search(r"(puls|resp|ecg|physio|\.log$|\.resp$|\.puls$)", r, re.I)
        and r.endswith((".log", ".resp", ".puls", ".ecg", ".txt", ".1D"))
    ]
    if stray_physio and not physio:
        out.append(
            f"{len(stray_physio)} physio-looking files not as _physio.tsv.gz + .json, e.g. {stray_physio[0]}"
        )
    return out


def _local_name(root: Path) -> str:
    """A lab tree's name: skip generic folder names like ``subjects`` or ``data``."""
    generic = {"subjects", "data", "rawdata", "bids", "output", "nii", "nifti"}
    p = Path(root)
    while p.name.lower() in generic and p.parent != p:
        p = p.parent
    return p.name


# ---------------------------------------------------------------------------
# Manifests and the gaps report
# ---------------------------------------------------------------------------


_CURATED_TEMPLATE: dict[str, Any] = {
    "exclude": None,  # a reason string drops the whole dataset from checks
    "exclude_acquisitions": {},  # acquisition id -> reason
    "set": {},  # acquisition field -> value for every acquisition (e.g. field_T a sidecar lacks)
    "set_acquisitions": {},  # acquisition id -> {field: value}
    "source": "",  # where any 'set' value comes from (paper, scanner protocol): required with 'set'
    "notes": "",
}


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write the census; a re-census keeps the hand-edited ``curated`` block."""
    import yaml

    for a in manifest["acquisitions"]:
        a.pop("_meta_keys", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['dataset_id']}.yaml"
    curated = dict(_CURATED_TEMPLATE)
    if path.exists():
        old = yaml.safe_load(path.read_text()) or {}
        curated.update(old.get("curated") or {})
    body = {k: v for k, v in manifest.items() if k != "curated"}
    with open(path, "w") as f:
        f.write(
            f"# Generated by benchmark.census on {manifest['census_date']}; "
            "hand edits go in 'curated:' only (kept across re-census).\n"
        )
        yaml.safe_dump(body, f, sort_keys=False, default_flow_style=None, width=100)
        yaml.safe_dump(
            {"curated": curated}, f, sort_keys=False, default_flow_style=False, width=100
        )
    return path


def apply_curation(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest as checks see it: exclusions dropped, curated values set."""
    cur = {**_CURATED_TEMPLATE, **(manifest.get("curated") or {})}
    if (cur["set"] or cur["set_acquisitions"]) and not cur["source"]:
        raise ValueError(f"{manifest['dataset_id']}: curated 'set' values need a 'source'")
    m = dict(manifest)
    m["excluded"] = cur["exclude"]
    kept, dropped = [], {}
    for a in manifest["acquisitions"]:
        if a["id"] in (cur["exclude_acquisitions"] or {}):
            dropped[a["id"]] = cur["exclude_acquisitions"][a["id"]]
            continue
        a = {**a, **(cur["set"] or {}), **((cur["set_acquisitions"] or {}).get(a["id"]) or {})}
        kept.append(a)
    m["acquisitions"] = kept
    m["excluded_acquisitions"] = dropped
    return m


def load_manifests(out_dir: Path, curated: bool = True) -> list[dict[str, Any]]:
    import yaml

    raw = [yaml.safe_load(p.read_text()) for p in sorted(Path(out_dir).glob("*.yaml"))]
    return [apply_curation(m) for m in raw] if curated else raw


@dataclass
class Axis:
    name: str
    bins: list[tuple[str, Any]] = field(default_factory=list)


def _bin(value: float | None, edges: list[tuple[str, float]]) -> str:
    if value is None:
        return "?"
    for label, hi in edges:
        if value < hi:
            return label
    return edges[-1][0]


_TR_BINS = [("TR<0.8", 0.8), ("0.8-1.5", 1.5), ("1.5-2.5", 2.5), ("TR>=2.5", 1e9)]
_VOX_BINS = [("<1.2mm", 1.2), ("1.2-1.8", 1.8), ("1.8-2.6", 2.6), (">=2.6mm", 1e9)]


def acquisition_axes(acq: dict[str, Any]) -> dict[str, str]:
    """The paper's summary axes for one acquisition."""
    f = acq.get("field_T")
    vox = acq.get("voxel_mm") or []
    iso = (vox[0] * vox[1] * vox[2]) ** (1 / 3) if len(vox) == 3 else None
    return {
        "field": "?"
        if not f
        else ("9.4T" if f >= 9 else "7T" if f >= 6.5 else "3T" if f >= 2.5 else f"{f}T"),
        "tr": _bin(acq.get("tr_s"), _TR_BINS),
        "voxel": _bin(iso, _VOX_BINS),
        "echoes": "ME" if "multi_echo" in acq["offers"] else "SE",
        "readout": {"epi_2d": "2D", "epi_3d": "3D"}.get(
            next((o for o in acq["offers"] if o.startswith("epi_")), ""), "?"
        ),
    }


# Provisional requirements of the paper's core tools, until each check declares its own.
CORE_NEEDS: dict[str, list[list[str]]] = {
    "moco": [[]],
    "slicetime": [["slice_timing"]],
    "physio": [["epi_2d", "physio_cardiac"], ["epi_2d", "phase", "physio_resp"]],
    "nordic": [["phase"]],
    "reml/deconvolve": [["events"]],
    "hrfopt/denoise/ridge": [["events"]],
    "ica": [["rest"]],
    "fc": [["rest"]],
    "pyrf": [["events", "@task:^(ret|prf|wedge|ring|bars?$)"]],
    "locomoco": [[]],
    "t2smap": [["multi_echo"]],
    "medic": [["multi_echo", "phase"]],
    "blipflip": [["@ds:fmap_epi"]],
    "b0fmap/phase_unwrap": [["@ds:fmap_gre"], ["phase", "multi_echo"]],
    "allineate/qwarp/nwarp/bbr": [["@ds:anat_t1"]],
    "segment": [["@ds:anat_t1"]],
    "group/clustsim": [["events", "@nsub:10"]],
}


def _qualifies(acq: dict, ds: dict, need: list[str]) -> bool:
    for tag in need:
        if tag.startswith("@ds:"):
            if tag[4:] not in ds["offers"]:
                return False
        elif tag.startswith("@task:"):
            pat = tag[6:]
            if not re.search(pat, acq["entities"].get("task", ""), re.I):
                return False
        elif tag.startswith("@nsub:"):
            if acq["n_subjects"] < int(tag[6:]):
                return False
        elif tag not in acq["offers"]:
            return False
    return True


def gaps_report(manifests: list[dict[str, Any]]) -> str:
    """Plain-text grid: which datasets cover which axis cell, and which tools they feed."""
    lines = []
    rows = []
    skipped = [ds for ds in manifests if ds.get("excluded")]
    for ds in manifests:
        if ds.get("excluded"):
            continue
        for acq in ds["acquisitions"]:
            ax = acquisition_axes(acq)
            rows.append((ds, acq, ax))

    lines.append("## Acquisitions\n")
    hdr = f"{'dataset':<24} {'acquisition':<44} {'sub':>4} {'runs':>5} {'field':>5} {'TR':>6} {'vox mm':>14} {'MB':>3} {'TE ms':<20} offers"
    lines.append(hdr)
    for ds, acq, ax in rows:
        vox = "x".join(f"{v:g}" for v in (acq.get("voxel_mm") or [])) or "?"
        te = ",".join(f"{t:g}" for t in acq["te_ms"] if t is not None) or "?"
        tr = f"{acq['tr_s']:g}" if isinstance(acq.get("tr_s"), (int, float)) else "?"
        lines.append(
            f"{ds['dataset_id'][:24]:<24} {acq['id'][:44]:<44} {acq['n_subjects']:>4} {acq['n_runs']:>5} "
            f"{ax['field']:>5} {tr:>6} {vox:>14} {acq.get('multiband') or '?'!s:>3} {te[:20]:<20} "
            f"{' '.join(o for o in acq['offers'] if o not in ('slice_timing',))}"
        )

    lines.append("\n## Coverage: field × TR (datasets per cell)\n")
    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ds, _acq, ax in rows:
        cells[(ax["field"], ax["tr"])].add(ds["dataset_id"])
    fields = sorted({k[0] for k in cells})
    trs = [b[0] for b in _TR_BINS] + (["?"] if any(k[1] == "?" for k in cells) else [])
    lines.append(f"{'':<6}" + "".join(f"{t:>12}" for t in trs))
    for f in fields:
        lines.append(f"{f:<6}" + "".join(f"{len(cells.get((f, t), ())):>12}" for t in trs))

    lines.append("\n## Coverage: field × voxel\n")
    cells2: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ds, _acq, ax in rows:
        cells2[(ax["field"], ax["voxel"])].add(ds["dataset_id"])
    voxs = [b[0] for b in _VOX_BINS]
    lines.append(f"{'':<6}" + "".join(f"{v:>12}" for v in voxs))
    for f in fields:
        lines.append(f"{f:<6}" + "".join(f"{len(cells2.get((f, v), ())):>12}" for v in voxs))

    lines.append("\n## Core tools: qualifying datasets (axes covered)\n")
    for tool, alternatives in CORE_NEEDS.items():
        hits: dict[str, set[str]] = defaultdict(set)
        for ds, acq, ax in rows:
            if any(_qualifies(acq, ds, need) for need in alternatives):
                hits[ds["dataset_id"]].add(f"{ax['field']}/{ax['tr']}/{ax['voxel']}/{ax['echoes']}")
        if not hits:
            lines.append(f"{tool:<28} NONE")
            continue
        fields_cov = sorted({c.split("/")[0] for v in hits.values() for c in v})
        lines.append(
            f"{tool:<28} {len(hits):>2} datasets  fields {','.join(fields_cov)}  : {', '.join(sorted(hits))}"
        )
    excl_acq = [
        (ds["dataset_id"], a, r)
        for ds in manifests
        for a, r in (ds.get("excluded_acquisitions") or {}).items()
    ]
    if skipped or excl_acq:
        lines.append("\n## Excluded (curated)\n")
        lines += [f"{ds['dataset_id']}: {ds['excluded']}" for ds in skipped]
        lines += [f"{d} {a}: {r}" for d, a, r in excl_acq]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.cli_help import FfsArgumentParser

    p = FfsArgumentParser(
        prog="python -m fastfuncstuff.benchmark.census",
        description="Census BIDS datasets (local path or OpenNeuro dsNNNNNN) into benchmark manifests.",
    )
    p.add_argument("-datasets", nargs="*", default=[], help="Local paths, dsNNNNNN, or s3:// URIs.")
    p.add_argument("-out", type=Path, required=True, help="Manifest directory.")
    p.add_argument(
        "-gaps", action="store_true", help="Print the coverage report for every manifest in -out."
    )
    args = p.parse_args(argv)

    for spec in args.datasets:
        try:
            src = open_source(spec)
            m = census(src)
        except Exception as e:  # one bad tree must not stop a batch census
            print(f"FAILED {spec}: {type(e).__name__}: {e}")
            continue
        path = write_manifest(m, args.out)
        print(
            f"{spec} -> {path} ({len(m['acquisitions'])} acquisitions, {m['n_subjects']} subjects)"
        )
    if args.gaps:
        print(gaps_report(load_manifests(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
