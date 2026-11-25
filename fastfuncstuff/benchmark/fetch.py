"""Bring a unit's input files to local disk: local trees are read in place, S3 is cached.

Only downloaded copies are ever handed to the pruner; a private dataset's original files
are never registered as artifacts, so pruning cannot touch them.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from .census import LocalSource, S3Source


def source_from_manifest(manifest: dict) -> LocalSource | S3Source:
    src = manifest["source"]
    if src["kind"] == "s3":
        bucket, _, prefix = src["uri"][5:].partition("/")
        return S3Source(prefix.strip("/"), bucket=bucket)
    # a bundled host set names its data relative to the root it runs from
    return LocalSource(Path(os.path.expandvars(src["uri"])))


def fetch(manifest: dict, rels: list[str], cache_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """Local path per input, and the list of files this call downloaded (prunable)."""
    src = source_from_manifest(manifest)
    paths: dict[str, Path] = {}
    downloaded: list[Path] = []
    for rel in rels:
        if isinstance(src, LocalSource):
            p = src.root / rel
            if not p.exists():
                raise FileNotFoundError(p)
            paths[rel] = p
            continue
        dest = Path(cache_dir) / manifest["dataset_id"] / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Per-process name: parallel workers may fetch the same file; the rename is atomic.
            part = dest.with_name(f"{dest.name}.part{os.getpid()}")
            url = f"{src._host}/{urllib.parse.quote(src.prefix + rel)}"
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            part.replace(dest)
            downloaded.append(dest)
        paths[rel] = dest
    return paths, downloaded


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
