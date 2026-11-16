"""GIfTI per-vertex data: ``.func.gii`` time series and ``.shape.gii`` maps.

One data array per time point, which is what SUMA, Workbench, FreeSurfer and
nibabel all read as a series. Every file carries the **mesh fingerprint** (vertex
and face counts plus a hash of the faces): per-vertex data means nothing on another
mesh, and topology edits (CHEDI delete/split) renumber vertices, so a reader can
refuse a mismatch instead of silently mis-pairing data and mesh.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HISTORY_KEY",
    "GiftiHeader",
    "gifti_header",
    "with_history",
    "load_gifti_data",
    "load_gifti_surface",
    "mesh_fingerprint",
    "save_gifti_data",
    "save_gifti_series_external",
    "save_gifti_surface",
]


#: File-level metadata key holding the provenance trail, one entry per line.
HISTORY_KEY = "history"


def with_history(meta: dict[str, str] | None) -> dict[str, str]:
    """``meta`` with this command appended to its history.

    The same ``[user@host: ctime] commandline`` entry AFNI's HISTORY_NOTE gets
    on every NIfTI ffs writes, as a plain newline-separated metadata value: a
    file written from another (``save_nifti_data_as_gifti`` carries its input's
    metadata) continues that file's trail.
    """
    from fastfuncstuff.io.afni import _history_commandline, _history_prefix

    out = {k: str(v) for k, v in (meta or {}).items()}
    entry = f"[{_history_prefix()}] {_history_commandline()}"
    prior = out.get(HISTORY_KEY, "").rstrip("\n")
    out[HISTORY_KEY] = f"{prior}\n{entry}" if prior else entry
    return out


def mesh_fingerprint(faces: np.ndarray, n_vertices: int | None = None) -> str:
    """``V<n>_F<m>_<sha1 of the int32 faces, 12 hex>``."""
    f = np.ascontiguousarray(np.asarray(faces, np.int32))
    v = int(f.max()) + 1 if n_vertices is None else int(n_vertices)
    return f"V{v}_F{f.shape[0]}_{hashlib.sha1(f.tobytes()).hexdigest()[:12]}"


def save_gifti_data(
    path: str | os.PathLike,
    data: np.ndarray,
    meta: dict[str, str] | None = None,
    time_series: bool | None = None,
) -> None:
    """Write ``(V,)`` or ``(V, T)`` float32 data, one array per column.

    ``time_series`` picks the intent (NIFTI_INTENT_TIME_SERIES vs NONE); by default a
    2-D array is a series. ``meta`` goes into the file-level metadata as strings.
    """
    import nibabel as nib

    d = np.asarray(data, np.float32)
    cols = d[:, None] if d.ndim == 1 else d
    series = d.ndim == 2 if time_series is None else time_series
    intent = "NIFTI_INTENT_TIME_SERIES" if series else "NIFTI_INTENT_NONE"
    # A series is written as plain base64: gzip saves ~11% on real surface data
    # (float noise does not compress) and costs ~7 ms a frame, serially, while the
    # GPU that produced it sits idle -- 3 s of a 460-frame file. Small maps keep it.
    encoding = "B64BIN" if series else "B64GZ"
    arrays = [
        nib.gifti.GiftiDataArray(
            np.ascontiguousarray(cols[:, t]),
            intent=intent,
            datatype="NIFTI_TYPE_FLOAT32",
            encoding=encoding,
        )
        for t in range(cols.shape[1])
    ]
    img = nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(with_history(meta)))
    nib.save(img, os.fspath(path))


@dataclass(frozen=True)
class GiftiHeader:
    """What a ``.gii`` holds, read without decoding its data."""

    n_arrays: int
    n_vertices: int  # Dim0 of the first array
    meta: dict[str, str]


def gifti_header(path: str | os.PathLike) -> GiftiHeader:
    """Array count, vertex count and file-level metadata of a ``.gii``, cheaply.

    nibabel parses and base64-decodes the whole file to answer any of these: about
    1.5 s for a 390 MB series, and one ffs_pyrf fit used to ask some 26 times. The
    file-level MetaData precedes the first DataArray, so it is parsed on its own;
    the arrays are counted by their opening tags.
    """
    import mmap
    import re
    import xml.etree.ElementTree as ET

    with (
        open(os.fspath(path), "rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        first = mm.find(b"<DataArray")
        if first < 0:
            raise ValueError(f"{path} has no DataArray")
        head = bytes(mm[:first])
        tag = bytes(mm[first : mm.find(b">", first) + 1])
        n_arrays, at = 0, first
        while at >= 0:
            n_arrays += 1
            at = mm.find(b"<DataArray", at + len(b"<DataArray"))
    dim0 = re.search(rb'Dim0\s*=\s*"(\d+)"', tag)
    if dim0 is None:
        raise ValueError(f"{path}: first DataArray has no Dim0")
    meta: dict[str, str] = {}
    start = head.find(b"<MetaData")
    if start >= 0:
        stop = head.find(b"</MetaData>", start)
        block = head[start : stop + len(b"</MetaData>")] if stop >= 0 else b"<MetaData/>"
        for md in ET.fromstring(block).iter("MD"):
            name, value = md.find("Name"), md.find("Value")
            if name is not None and name.text is not None:
                meta[name.text] = "" if value is None or value.text is None else value.text
    return GiftiHeader(n_arrays=n_arrays, n_vertices=int(dim0.group(1)), meta=meta)


@dataclass(frozen=True)
class _Array:
    data: np.ndarray
    intent: int
    meta: dict[str, str]


@dataclass(frozen=True)
class _Gifti:
    darrays: list[_Array]
    meta: dict[str, str]


def _md_dict(block: bytes) -> dict[str, str]:
    import xml.etree.ElementTree as ET

    out: dict[str, str] = {}
    for md in ET.fromstring(block).iter("MD"):
        name, value = md.find("Name"), md.find("Value")
        if name is not None and name.text is not None:
            out[name.text] = "" if value is None or value.text is None else value.text
    return out


def _read_gifti(path: str | os.PathLike) -> _Gifti | None:
    """The data arrays of a base64 ``.gii``, decoded straight from the file.

    nibabel streams the base64 text through expat's Python callbacks before it
    decodes it; on a 190 MB series that is about 60% of a 1.2 s load. Slicing each
    ``<Data>`` element out of a memory map and decoding it in one call gives the
    same arrays, bit for bit. Returns None for anything else it does not handle
    (ASCII or external-file encodings, unknown types), and the caller falls back to
    nibabel, so the result never depends on which reader ran.
    """
    import base64
    import mmap
    import re
    import zlib

    import nibabel as nib

    codes = nib.nifti1.data_type_codes
    attr = re.compile(rb'(\w+)\s*=\s*"([^"]*)"')
    with (
        open(os.fspath(path), "rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        first = mm.find(b"<DataArray")
        if first < 0:
            return None
        head = bytes(mm[:first])
        start = head.find(b"<MetaData")
        file_meta: dict[str, str] = {}
        if start >= 0:
            stop = head.find(b"</MetaData>", start)
            if stop >= 0:
                file_meta = _md_dict(head[start : stop + len(b"</MetaData>")])
        arrays: list[_Array] = []
        external: dict[str, np.memmap] = {}
        at = first
        while at >= 0:
            tag_end = mm.find(b">", at)
            attrs = {k.decode(): v.decode() for k, v in attr.findall(bytes(mm[at : tag_end + 1]))}
            next_array = mm.find(b"<DataArray", tag_end)
            end = next_array if next_array >= 0 else len(mm)
            try:
                dtype = np.dtype(codes.dtype[attrs["DataType"]])
            except KeyError:
                return None
            dtype = dtype.newbyteorder(
                "<" if attrs.get("Endian", "LittleEndian") == "LittleEndian" else ">"
            )
            ndim = int(attrs.get("Dimensionality", "1"))
            shape = tuple(int(attrs[f"Dim{k}"]) for k in range(ndim))
            order = "F" if attrs.get("ArrayIndexingOrder") == "ColumnMajorOrder" else "C"
            encoding = attrs.get("Encoding")
            if encoding == "ExternalFileBinary":
                name = attrs.get("ExternalFileName", "")
                ext = os.path.join(os.path.dirname(os.fspath(path)), name)
                if not name or not os.path.isfile(ext):
                    return None
                if ext not in external:
                    external[ext] = np.memmap(ext, dtype=np.uint8, mode="r")
                offset = int(attrs.get("ExternalFileOffset", "0"))
                count = int(np.prod(shape)) * dtype.itemsize
                block = external[ext][offset : offset + count]
                if block.size != count:
                    return None
                data = np.array(block).view(dtype).reshape(shape, order=order)
            elif encoding in ("Base64Binary", "GZipBase64Binary"):
                # Exactly <Data>: the coordinate system's <DataSpace> also starts
                # "<Data".
                open_data = mm.find(b"<Data>", tag_end, end)
                close_data = mm.find(b"</Data>", open_data, end)
                if open_data < 0 or close_data < 0:
                    return None
                raw = base64.b64decode(mm[open_data + len(b"<Data>") : close_data])
                if encoding == "GZipBase64Binary":
                    raw = zlib.decompress(raw)
                data = np.frombuffer(bytearray(raw), dtype=dtype).reshape(shape, order=order)
            else:
                return None
            meta_start = mm.find(b"<MetaData", tag_end, end)
            meta: dict[str, str] = {}
            if meta_start >= 0:
                meta_stop = mm.find(b"</MetaData>", meta_start, end)
                if meta_stop >= 0:
                    meta = _md_dict(bytes(mm[meta_start : meta_stop + len(b"</MetaData>")]))
            intent = nib.nifti1.intent_codes.code[attrs.get("Intent", "NIFTI_INTENT_NONE")]
            arrays.append(_Array(data=data, intent=int(intent), meta=meta))
            at = next_array
    return _Gifti(darrays=arrays, meta=file_meta)


def _load_gifti(path: str | os.PathLike):
    import nibabel as nib

    fast = _read_gifti(path)
    if fast is not None:
        return fast
    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    return img


def load_gifti_data(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, str]]:
    """``(V, T)`` (or ``(V,)`` for one array) and the file-level metadata."""
    img = _load_gifti(path)
    cols = [np.asarray(a.data, np.float32) for a in img.darrays]
    data = cols[0] if len(cols) == 1 else np.stack(cols, axis=1)
    return data, dict(img.meta)


def save_gifti_series_external(
    path: str | os.PathLike, data: np.ndarray, meta: dict[str, str] | None = None
) -> str:
    """A ``(V, T)`` float32 series as a GIfTI whose arrays live in a raw ``.dat`` file.

    For intermediates ffs writes and reads back itself (the merged hemispheres):
    base64 costs a full encode on write and a GIL-bound decode on every read, about
    1 s each for a 390 MB run, where raw binary is a plain copy. ExternalFileBinary
    is part of the GIfTI standard and nibabel and Workbench read it; nibabel cannot
    write it, hence this writer. Frames are stored one after another, so each array
    is one contiguous block. Returns the ``.dat`` path, which must travel with the
    ``.gii``.
    """
    import xml.etree.ElementTree as ET

    d = np.asarray(data, np.float32)
    if d.ndim != 2:
        raise ValueError(f"expected (vertices, frames), got {d.shape}")
    n_vertices, n_frames = d.shape
    out = os.fspath(path)
    dat = (out[: -len(".gii")] if out.endswith(".gii") else out) + ".dat"
    with open(dat, "wb") as fh:
        np.ascontiguousarray(d.T, dtype="<f4").tofile(fh)
    root = ET.Element("GIFTI", Version="1.0", NumberOfDataArrays=str(n_frames))
    md = ET.SubElement(root, "MetaData")
    for key, value in with_history(meta).items():
        entry = ET.SubElement(md, "MD")
        ET.SubElement(entry, "Name").text = key
        ET.SubElement(entry, "Value").text = value
    ET.SubElement(root, "LabelTable")
    for t in range(n_frames):
        array = ET.SubElement(
            root,
            "DataArray",
            Intent="NIFTI_INTENT_TIME_SERIES",
            DataType="NIFTI_TYPE_FLOAT32",
            ArrayIndexingOrder="RowMajorOrder",
            Dimensionality="1",
            Dim0=str(n_vertices),
            Encoding="ExternalFileBinary",
            Endian="LittleEndian",
            ExternalFileName=os.path.basename(dat),
            ExternalFileOffset=str(t * n_vertices * 4),
        )
        ET.SubElement(array, "MetaData")
        ET.SubElement(array, "Data")
    body = ET.tostring(root, encoding="unicode")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">\n')
        fh.write(body)
    return dat


def save_gifti_surface(
    path: str | os.PathLike,
    vertices: np.ndarray,
    faces: np.ndarray,
    meta: dict[str, str] | None = None,
) -> None:
    """A ``.surf.gii`` in scanner mm (NIFTI_XFORM_SCANNER_ANAT), as SUMA and Workbench read."""
    import nibabel as nib

    xform = nib.gifti.GiftiCoordSystem(dataspace=1, xformspace=1, xform=np.eye(4))
    pts = nib.gifti.GiftiDataArray(
        np.asarray(vertices, np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
        coordsys=xform,
    )
    tri = nib.gifti.GiftiDataArray(
        np.asarray(faces, np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"
    )
    m = {"mesh_fingerprint": mesh_fingerprint(faces, len(vertices)), **(meta or {})}
    img = nib.gifti.GiftiImage(darrays=[pts, tri], meta=nib.gifti.GiftiMetaData(with_history(m)))
    nib.save(img, os.fspath(path))


def load_gifti_surface(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """``(vertices, faces, metadata)`` of a ``.surf.gii``."""
    import nibabel as nib

    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    pts = img.agg_data("NIFTI_INTENT_POINTSET")
    tri = img.agg_data("NIFTI_INTENT_TRIANGLE")
    return np.asarray(pts, np.float64), np.asarray(tri, np.int64), dict(img.meta)


# --- GIfTI as an in-memory NIfTI ----------------------------------------------
#
# AFNI's answer to surface data is to treat it as a 1-D volume (V x 1 x 1), and
# so is ours: io/afni.load_nifti turns a .gii into a (V, 1, 1, T) image (NIfTI-2:
# NIfTI-1 dims stop at 32,767, onavg-ico64 has 40,962 vertices) and save_nifti
# turns one back, so every tool that reads and writes through those two runs on
# surface data unchanged. The GIfTI metadata (mesh fingerprint, geometry file)
# rides in a NIfTI comment extension and comes back out on save.

SURFACE_EXT_TAG = "ffs-surface:"
_NIFTI_ECODE_COMMENT = 6
#: AFNI stat codes that are also NIfTI intent codes (fitt, fift, fizt, ...).
_STAT_INTENT = {
    2: "NIFTI_INTENT_CORREL",
    3: "NIFTI_INTENT_TTEST",
    4: "NIFTI_INTENT_FTEST",
    5: "NIFTI_INTENT_ZSCORE",
    6: "NIFTI_INTENT_CHISQ",
}


def data_geometry(
    data_path: str | os.PathLike,
    meta: dict[str, str] | None,
    n_vertices: int,
    surf_geom: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of the mesh surface data live on: ``surf_geom``, else the
    data's ``geometry`` metadata (relative to the data file), checked against the
    data's vertex count and mesh fingerprint. ValueError when it cannot be found or
    does not match."""
    meta = meta or {}
    geom = None if surf_geom is None else os.fspath(surf_geom)
    if geom is None and meta.get("geometry"):
        geom = os.path.join(os.path.dirname(os.fspath(data_path)), meta["geometry"])
    if geom is None or not os.path.exists(geom):
        raise ValueError(
            "this needs the mesh the surface data live on: pass -surf_geom "
            f"(the ?h.midthickness.surf.gii); looked for {geom!r}"
        )
    vertices, faces, _ = load_gifti_surface(geom)
    if len(vertices) != n_vertices:
        raise ValueError(f"{geom} has {len(vertices)} vertices, the data {n_vertices}")
    want = meta.get("mesh_fingerprint")
    if want and want != mesh_fingerprint(faces, len(vertices)):
        raise ValueError(f"{geom} is not the mesh the data were sampled on")
    return vertices, faces


def is_gifti(path: str | os.PathLike) -> bool:
    return os.fspath(path).lower().endswith(".gii")


def _surface_json(ext) -> str | None:
    """The JSON text of our surface extension, or None for any other extension."""
    if ext.get_code() != _NIFTI_ECODE_COMMENT:
        return None
    raw = ext.get_content()
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return text[len(SURFACE_EXT_TAG) :] if text.startswith(SURFACE_EXT_TAG) else None


def surface_meta(header) -> dict[str, str] | None:
    """The GIfTI metadata a surface image carries, or None for a volume."""
    import json

    for ext in getattr(header, "extensions", []):
        text = _surface_json(ext)
        if text is not None:
            return json.loads(text)
    return None


def set_surface_meta(header, meta: dict[str, str]) -> None:
    import json

    import nibabel as nib

    keep = [e for e in header.extensions if _surface_json(e) is None]
    header.extensions.clear()
    header.extensions.extend(keep)
    blob = (SURFACE_EXT_TAG + json.dumps(meta, sort_keys=True)).encode()
    header.extensions.append(nib.nifti1.Nifti1Extension(_NIFTI_ECODE_COMMENT, blob))


def gifti_as_nifti(path: str | os.PathLike):
    """A ``.gii`` data file as a ``(V, 1, 1, T)`` (or ``(V, 1, 1)``) NIfTI-2 image."""
    import nibabel as nib

    img = _load_gifti(path)
    if any(a.intent in (1008, 1009) for a in img.darrays):  # POINTSET / TRIANGLE
        raise ValueError(f"{path} is a surface geometry file, not per-vertex data")
    cols = [np.asarray(a.data, np.float32).reshape(-1) for a in img.darrays]
    data = cols[0][:, None, None] if len(cols) == 1 else np.stack(cols, axis=1)[:, None, None, :]
    out = nib.Nifti2Image(np.ascontiguousarray(data), np.eye(4))
    meta = dict(img.meta)
    # The mesh path is relative to this file; outputs written elsewhere rebase it
    # (save_nifti_data_as_gifti), which needs to know where it really is.
    if meta.get("geometry") and not os.path.isabs(meta["geometry"]):
        here = os.path.dirname(os.path.abspath(os.fspath(path)))
        meta["geometry"] = os.path.normpath(os.path.join(here, meta["geometry"]))
    labels = [str(dict(a.meta).get("Name", "")) for a in img.darrays]
    if any(labels):
        meta["brick_labels"] = "\t".join(labels)
    set_surface_meta(out.header, meta)
    # Labels and stat codes also go into a real AFNI extension, so a tool that reads
    # buckets through the AFNI attributes (ffs_util_concalc) reads a surface bucket
    # the same way.
    stat = {}
    for k, a in enumerate(img.darrays):
        am = dict(a.meta)
        if "StatCode" in am:
            stat[k] = (
                int(am["StatCode"]),
                tuple(float(x) for x in am.get("StatParams", "").split()),
            )
    if any(labels) or stat:
        from fastfuncstuff.io.afni import _set_afni_brick_labels, _set_afni_brick_stataux

        if any(labels):
            _set_afni_brick_labels(out.header, labels)
        if stat:
            _set_afni_brick_stataux(out.header, stat, len(cols))
    tr = meta.get("TR_seconds")
    if tr and data.ndim == 4:
        out.header.set_xyzt_units(xyz="mm", t="sec")
        out.header["pixdim"][4] = float(tr)
    return out


def save_nifti_data_as_gifti(
    path: str | os.PathLike,
    data: np.ndarray,
    header=None,
    brick_labels: list[str] | None = None,
    brick_stataux: dict[int, tuple[int, tuple[float, ...]]] | None = None,
    tr: float | None = None,
) -> None:
    """Write ``(V, 1, 1[, K])`` data to a ``.gii``, one array per sub-brick.

    Brick labels become each array's ``Name``; a stat sub-brick gets the matching
    NIfTI intent and its AFNI code and parameters (``StatCode``/``StatParams``) in the
    array metadata, so the viewer can threshold by p. The input's GIfTI metadata
    (mesh fingerprint, geometry) is carried from ``header``.
    """
    import nibabel as nib

    d = np.asarray(data)
    if d.ndim < 3 or d.shape[1:3] != (1, 1):
        raise ValueError(
            f"a .gii output needs surface data (V, 1, 1[, K]); got {d.shape} -- "
            "is the input a volume?"
        )
    cols = d.reshape(d.shape[0], -1).astype(np.float32)
    meta = dict(surface_meta(header) or {}) if header is not None else {}
    meta.pop("brick_labels", None)
    if os.path.isabs(meta.get("geometry", "")):
        try:
            meta["geometry"] = os.path.relpath(
                meta["geometry"], os.path.dirname(os.path.abspath(os.fspath(path)))
            )
        except ValueError:  # another Windows drive: only an absolute path reaches it
            pass
    if tr is not None:
        meta["TR_seconds"] = f"{tr:g}"
    series = cols.shape[1] > 1 and not brick_labels and not brick_stataux
    arrays = []
    for k in range(cols.shape[1]):
        am = {}
        intent = "NIFTI_INTENT_TIME_SERIES" if series else "NIFTI_INTENT_NONE"
        if brick_labels is not None and k < len(brick_labels):
            am["Name"] = brick_labels[k]
        if brick_stataux and k in brick_stataux:
            code, params = brick_stataux[k]
            intent = _STAT_INTENT.get(int(code), intent)
            am["StatCode"] = str(int(code))
            am["StatParams"] = " ".join(f"{p:g}" for p in params)
        arrays.append(
            nib.gifti.GiftiDataArray(
                np.ascontiguousarray(cols[:, k]),
                intent=intent,
                datatype="NIFTI_TYPE_FLOAT32",
                meta=nib.gifti.GiftiMetaData(am),
            )
        )
    img = nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(with_history(meta)))
    nib.save(img, os.fspath(path))
