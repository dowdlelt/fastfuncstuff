"""Dataset introspection: everything ``3dinfo`` reports, read from the header alone.

The point of this module is that asking "how many volumes does this run have?"
must not cost a decompress. :func:`read_info` pulls the NIfTI header (and the
AFNI extension riding behind it) off the front of the file — a few KB even for a
multi-GB ``.nii.zst`` — and derives the AFNI-facing quantities from it: orient
code, obliquity, signed voxel steps, origin, spatial extent, slice timing.

Coordinate conventions
----------------------
NIfTI affines are RAS+ (``+x`` = Right, ``+y`` = Anterior, ``+z`` = Superior).
AFNI reports in its DICOM-ish frame where ``+x`` = Left and ``+y`` = Posterior,
so every AFNI-facing number here goes through :data:`AFNI_FROM_RAS`. Getting this
backwards silently flips the sign of ``-o3`` / ``-extent`` / ``-d3``, which is
why the conversion lives in exactly one place.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fastfuncstuff.io.headers import (
    _afni_ext_text,
    _read_leading_bytes,
    _read_nifti_header_with_ext,
    _resolve_indices,
    header_tr,
    parse_subbrick_selector,
    read_brick_labels,
    read_brick_stataux,
)
from fastfuncstuff.io.labels import parse_atlas_points, parse_dtable, read_value_labels

# RAS+ (NIfTI) → AFNI DICOM order (x = Left+, y = Posterior+, z = Superior+).
AFNI_FROM_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# NIfTI datatype code → the name AFNI's -datum prints.
_DATUM_NAMES: dict[str, str] = {
    "uint8": "byte",
    "int8": "byte",
    "int16": "short",
    "uint16": "short",
    "int32": "int",
    "uint32": "int",
    "int64": "int",
    "float32": "float",
    "float64": "double",
    "complex64": "complex",
    "complex128": "complex",
}

# NIfTI sform/qform codes → AFNI template-space names, used when the dataset
# carries no AFNI extension to say so itself.
_SPACE_FROM_CODE: dict[int, str] = {0: "ORIG", 1: "ORIG", 2: "ORIG", 3: "TLRC", 4: "MNI", 5: "MNI"}


def _view_from_code(code: int, space: str) -> str:
    """AFNI's +orig/+acpc/+tlrc view for a NIfTI dataset.

    Port of ``thd_niftiread.c:NIFTI_code_to_view``: the xform code decides,
    and the AFNI extension's space name is consulted only for code 2. The
    extension saying "MNI_2009c_asym" over a code of 1 is still +orig — which
    is exactly the half-converted state this field exists to make visible.
    """
    if code in (3, 4, 5):
        return "+tlrc"
    if code == 2:
        if space in ("", "ORIG"):
            return "+orig"
        return "+acpc" if space == "ACPC" else "+tlrc"
    return "+orig"


_XFORM_NAMES: dict[int, str] = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}

_SLICE_ORDER_NAMES: dict[int, str] = {
    0: "unknown",
    1: "seq+z",
    2: "seq-z",
    3: "alt+z",
    4: "alt-z",
    5: "alt+z2",
    6: "alt-z2",
}


@dataclass
class DatasetInfo:
    """Everything ``ffs_info`` can report about one dataset, header-only.

    ``shape`` is always length 4 (``nv`` = 1 for a lone volume) so callers never
    have to branch on dimensionality. Coordinate fields (``origin``, ``extent``,
    ``signed_steps``) are in the AFNI frame; ``affine`` stays RAS+.
    """

    path: Path
    iname: str
    exists: bool
    selector: list[int] | None = None

    storage: str = "UNKNOWN"  # NIFTI / NIFTI-2 / AFNI / BRIK
    compression: str | None = None  # "gzip" / "zstd" / None
    file_bytes: int = 0
    datum: str = "unknown"
    itemsize: int = 0  # bytes per voxel on disk, for the uncompressed-size line

    shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    zooms: tuple[float, float, float] = (0.0, 0.0, 0.0)  # abs voxel size (-ad3)
    signed_steps: tuple[float, float, float] = (0.0, 0.0, 0.0)  # AFNI-frame (-d3)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # AFNI coords of ijk (0,0,0)
    extent: tuple[float, float, float, float, float, float] = (0.0,) * 6

    tr: float = 0.0
    slice_timing: list[float] | None = None
    slice_order: str | None = None
    #: Time origin of the series (s): NIfTI ``toffset`` / AFNI ttorg.  After
    #: slice-timing correction it is the ``-tzero`` every slice was moved to.
    time_offset: float = 0.0

    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    orient: str = "???"
    space: str = "ORIG"
    view: str = "+orig"
    obliquity: float = 0.0
    qform_code: int = 0
    sform_code: int = 0

    labels: list[str] = field(default_factory=list)
    #: ``{value: LabelEntry}`` for a label volume that names its regions in its
    #: own header. Only the header route is taken here: a sidecar table is
    #: found by name, and "the .txt next to run1.nii.gz" is as likely to be a
    #: stimulus timing file as a LUT -- so that lookup waits until something
    #: has decided the volume really is labels.
    value_labels: dict = field(default_factory=dict)
    #: ``BRICK_STATAUX``: ``{sub_brick: (afni_stat_code, params)}``. What lets
    #: a reader turn a t or an F into a p without being told the DoF -- the
    #: bucket already says. Empty when the dataset carries no stat metadata.
    stataux: dict[int, tuple[int, tuple[float, ...]]] = field(default_factory=dict)
    history: str = ""
    #: AFNI's dataset IDCODE (``AFN_...``) when the header carries one, else "".
    #: NIfTI keeps it as the AFNI extension's ``self_idcode`` (or ``IDCODE_STRING``);
    #: a file that never passed through AFNI or ffs has none.
    idcode: str = ""
    #: A GIfTI file's own metadata (mesh fingerprint, subject, mesh, TR...);
    #: empty for a volume.
    surface_meta: dict[str, str] = field(default_factory=dict)
    #: For a GIfTI: "mesh" (vertices + triangles), "series" or "maps"; else "".
    surface_kind: str = ""
    #: Triangles in a GIfTI mesh.
    n_faces: int = 0
    descrip: str = ""
    scl_slope: float = 1.0
    scl_inter: float = 0.0

    @property
    def prefix(self) -> str:
        return self.path.name

    @property
    def is_oblique(self) -> bool:
        # AFNI calls anything under 0.01 degrees plumb (thd_coords.c rounds the
        # reported angle to 3 decimals and 3dinfo -is_oblique tests > 0).
        return self.obliquity > 0.0005

    @property
    def is_nifti(self) -> bool:
        return self.storage.startswith("NIFTI")

    @property
    def n_volumes(self) -> int:
        return self.shape[3]

    @property
    def fov(self) -> tuple[float, float, float]:
        nx, ny, nz = self.shape[:3]
        return (nx * self.zooms[0], ny * self.zooms[1], nz * self.zooms[2])

    @property
    def duration(self) -> float:
        """Series duration in seconds (0 for a non-time-series dataset)."""
        return self.tr * self.shape[3]


def afni_orient_code(affine: np.ndarray) -> str:
    """AFNI 3-letter orientation code (e.g. ``"LPI"``) for a RAS+ affine.

    AFNI names the side each axis starts *from*; nibabel's ``aff2axcodes`` names
    the side it points *to*. The codes are therefore exact opposites — an image
    nibabel calls RAS is ``LPI`` to AFNI.
    """
    flip = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    return "".join(flip[c] for c in nib.aff2axcodes(affine))


def obliquity_deg(affine: np.ndarray) -> float:
    """Degrees the voxel axes are tilted away from the cardinal directions.

    Port of ``THD_compute_oblique_angle`` (afni/src/thd_coords.c): for each voxel
    axis take the cosine to its dominant cardinal direction, and report the worst
    one. Absolute values throughout, so RAS-vs-DICOM sign flips don't matter.
    """
    r = np.asarray(affine, dtype=np.float64)[:3, :3]
    norms = np.linalg.norm(r, axis=0)
    if not np.all(norms > 0):
        return 0.0
    fig_merit = float(np.min(np.max(np.abs(r), axis=0) / norms))
    return float(np.degrees(np.arccos(min(fig_merit, 1.0))))


def _signed_steps(afni_affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel step along each axis, signed by its dominant AFNI-frame direction."""
    out = []
    for col in range(3):
        vec = afni_affine[:3, col]
        norm = float(np.linalg.norm(vec))
        sign = float(np.sign(vec[int(np.argmax(np.abs(vec)))])) or 1.0
        out.append(sign * norm)
    return (out[0], out[1], out[2])


def cardinal_affine(affine: np.ndarray) -> np.ndarray:
    """Deobliqued affine: each voxel axis snapped to its nearest cardinal direction.

    AFNI keeps the raw oblique matrix as ``ijk_to_dicom_real`` but does its
    coordinate arithmetic — including the extents ``3dinfo`` reports — with this
    cardinalized ``ijk_to_dicom``. The origin is preserved; only the 3×3 changes.
    """
    r = np.asarray(affine, dtype=np.float64)[:3, :3]
    out = np.zeros((4, 4), dtype=np.float64)
    out[3, 3] = 1.0
    out[:3, 3] = np.asarray(affine, dtype=np.float64)[:3, 3]
    for col in range(3):
        vec = r[:, col]
        dominant = int(np.argmax(np.abs(vec)))
        out[dominant, col] = np.sign(vec[dominant]) * float(np.linalg.norm(vec))
    return out


def _extent(afni_affine: np.ndarray, shape: tuple[int, int, int]) -> tuple[float, ...]:
    """Min/max AFNI coordinate along each axis, over voxel *centers* (AFNI's rule).

    Computed on the *cardinal* affine: for an oblique dataset AFNI reports the
    plumb bounding box of the deobliqued grid, not the tilted corners.
    """
    afni_affine = cardinal_affine(afni_affine)
    ni, nj, nk = (max(n - 1, 0) for n in shape)
    corners = np.array(
        [[i, j, k, 1.0] for i in (0, ni) for j in (0, nj) for k in (0, nk)], dtype=np.float64
    )
    xyz = (afni_affine @ corners.T)[:3]
    return tuple(
        float(v) for pair in zip(xyz.min(axis=1), xyz.max(axis=1), strict=True) for v in pair
    )


def _compression_of(path: Path) -> str | None:
    name = path.name
    if name.endswith(".zst"):
        return "zstd"
    if name.endswith(".gz"):
        return "gzip"
    return None


def _afni_atr(ext_text: str, name: str) -> str | None:
    """Value of one AFNI-XML attribute (``<AFNI_atr atr_name="…">…</AFNI_atr>``)."""
    import re

    m = re.search(rf'atr_name\s*=\s*"{re.escape(name)}"[^>]*>(.*?)</AFNI_atr>', ext_text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip()


def _idcode_from(ext_text: str) -> str:
    """The dataset IDCODE out of an AFNI NIfTI extension, or "" if it has none."""
    import re

    val = _afni_atr(ext_text, "IDCODE_STRING")
    if val:
        return val
    m = re.search(r'self_idcode\s*=\s*"([^"]+)"', ext_text)
    return m.group(1) if m else ""


def _decode_history(text: str) -> str:
    """AFNI-XML history → plain text: XML entities out, escaped newlines real."""
    import html

    return html.unescape(text).replace("\\n", "\n").strip()


def _slice_timing_from(header: Any, ext_text: str, n_slices: int) -> list[float] | None:
    """Per-slice acquisition offsets, AFNI extension first, then NIfTI slice fields."""
    taxis = _afni_atr(ext_text, "TAXIS_OFFSETS")
    if taxis:
        vals = [float(x) for x in taxis.split()]
        if vals:
            return vals
    try:
        times = header.get_slice_times()
    except Exception:  # nibabel raises for the (common) unset case
        return None
    if times is None or all(t is None for t in times):
        return None
    return [0.0 if t is None else float(t) for t in times][:n_slices]


def _time_offset_from(header: Any, ext_text: str, units: Any) -> float:
    """Series time origin: the AFNI extension's ttorg, else NIfTI ``toffset``."""
    floats = _afni_atr(ext_text, "TAXIS_FLOATS")
    if floats and floats.split():
        return float(floats.split()[0])
    toff = float(header["toffset"]) if np.isfinite(header["toffset"]) else 0.0
    if units and len(units) > 1 and units[1] == "msec":
        toff /= 1000.0
    elif units and len(units) > 1 and units[1] == "usec":
        toff /= 1e6
    return toff


def sample_time_offset(info: DatasetInfo) -> float | None:
    """Within-TR time (s) every voxel of this series was sampled at, if uniform.

    A slice-time-corrected series carries no per-slice offsets (or identical
    ones) and a time origin at its ``-tzero``; the sample time is origin +
    the shared offset.  ``None`` when slices still have distinct acquisition
    times: there is no single sample time to give a design.
    """
    st = info.slice_timing
    if st and max(st) - min(st) > 1e-6:
        return None
    return info.time_offset + (st[0] if st else 0.0)


def read_info(path: str | Path) -> DatasetInfo:
    """Describe a dataset from its header alone (no image payload is read).

    Accepts ``.nii`` / ``.nii.gz`` / ``.nii.zst``, AFNI ``.HEAD`` / ``.BRIK``,
    FreeSurfer ``.mgz`` / ``.mgh`` and GIfTI data (``.func.gii`` / ``.shape.gii``),
    with an optional AFNI ``[selector]`` suffix — the selector adjusts the
    reported volume count exactly as it would on a real load.
    """
    raw_arg = str(path)
    clean, indices = parse_subbrick_selector(raw_arg)
    p = Path(clean)
    if not p.exists():
        return DatasetInfo(path=p, iname=raw_arg, exists=False, selector=indices)

    if p.suffix in (".HEAD", ".BRIK") or p.name.endswith(".BRIK.gz"):
        return _read_afni_info(p, raw_arg, indices)
    if p.name.lower().endswith((".mgz", ".mgh")):
        return _read_mgh_info(p, raw_arg, indices)
    if p.name.lower().endswith(".gii"):
        return _read_gifti_info(p, raw_arg, indices)
    return _read_nifti_info(p, raw_arg, indices)


def _read_gifti_info(p: Path, iname: str, indices: list[int] | None) -> DatasetInfo:
    """A surface dataset as AFNI's 1-D volume ``(V, 1, 1, n_arrays)``, from the XML
    alone: the arrays are base64 inline (a 460-frame series is ~75 MB), so this
    counts and labels them without decoding any."""
    import re

    raw = p.read_bytes()
    n_arr = raw.count(b"<DataArray")
    dim0 = re.search(rb'Dim0="(\d+)"', raw)
    info = DatasetInfo(path=p, iname=iname, exists=True, selector=indices)
    info.storage = "GIFTI"
    info.file_bytes = len(raw)
    info.datum = "float"
    info.itemsize = 4
    nv = _apply_selector(n_arr, indices)
    info.shape = (int(dim0.group(1)) if dim0 else 0, 1, 1, nv)
    import html

    # <MD><Name>k</Name><Value>v</Value></MD>, with or without CDATA (writers differ)
    cdata = rb"(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?"
    pairs = [
        (html.unescape(k.decode()), html.unescape(v.decode()))
        for k, v in re.findall(
            # DOTALL: a value can span lines (history does), and a pair that
            # failed to match would shift every file-level key after it.
            rb"<Name>" + cdata + rb"</Name>\s*<Value>" + cdata + rb"</Value>",
            raw,
            re.DOTALL,
        )
    ]
    first_array = raw.find(b"<DataArray")
    head = re.findall(rb"<Name>" + cdata + rb"</Name>", raw[: max(first_array, 0)])
    file_md = dict(pairs[: len(head)])  # file-level metadata precedes the first array
    try:
        info.tr = float(file_md.get("TR_seconds", 0.0))
    except ValueError:
        info.tr = 0.0
    info.surface_meta = file_md
    info.history = file_md.get("history", "")
    intents = re.findall(rb'Intent="(\w+)"', raw)
    dims = re.findall(rb'Dim0="(\d+)"', raw)
    if b"NIFTI_INTENT_POINTSET" in intents:
        info.surface_kind = "mesh"
        tri = intents.index(b"NIFTI_INTENT_TRIANGLE") if b"NIFTI_INTENT_TRIANGLE" in intents else -1
        info.n_faces = int(dims[tri]) if 0 <= tri < len(dims) else 0
        info.shape = (int(dims[intents.index(b"NIFTI_INTENT_POINTSET")]), 1, 1, 1)
    else:
        info.surface_kind = "series" if b"NIFTI_INTENT_TIME_SERIES" in intents else "maps"
    labels = [v for k, v in pairs[len(head) :] if k == "Name"]
    if labels and len(labels) == n_arr:
        info.labels = (
            labels if indices is None else [labels[i] for i in _resolve_indices(indices, n_arr)]
        )
    return info


def _apply_selector(n_volumes: int, indices: list[int] | None) -> int:
    if indices is None:
        return n_volumes
    return len(_resolve_indices(indices, n_volumes))


def _common(
    info: DatasetInfo, affine: np.ndarray, shape3: tuple[int, int, int], zooms: tuple[float, ...]
) -> None:
    """Fill the geometry block shared by the NIfTI and AFNI readers."""
    afni_affine = AFNI_FROM_RAS @ affine
    info.affine = affine
    info.zooms = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    info.signed_steps = _signed_steps(afni_affine)
    info.origin = tuple(float(v) for v in afni_affine[:3, 3])  # type: ignore[assignment]
    info.extent = _extent(afni_affine, shape3)  # type: ignore[assignment]
    info.orient = afni_orient_code(affine)
    info.obliquity = obliquity_deg(affine)


def _read_nifti_info(p: Path, iname: str, indices: list[int] | None) -> DatasetInfo:
    hdr = _read_nifti_header_with_ext(p)
    ext_text = _afni_ext_text(hdr)

    dims = tuple(int(d) for d in hdr.get_data_shape())
    shape3 = (
        (dims + (1, 1, 1))[0],
        (dims + (1, 1, 1))[1],
        (dims + (1, 1, 1))[2],
    )
    # dim[5] carries the sub-brick count for AFNI buckets written as "3D+t of 1".
    n_vol = dims[3] if len(dims) > 3 else 1
    if len(dims) > 4 and dims[4] > 1 and n_vol == 1:
        n_vol = dims[4]
    n_vol = _apply_selector(n_vol, indices)

    zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    affine = np.asarray(hdr.get_best_affine(), dtype=np.float64)

    info = DatasetInfo(
        path=p,
        iname=iname,
        exists=True,
        selector=indices,
        storage="NIFTI" if isinstance(hdr, nib.Nifti1Header) else "NIFTI-2",
        compression=_compression_of(p),
        file_bytes=p.stat().st_size,
        datum=_DATUM_NAMES.get(np.dtype(hdr.get_data_dtype()).name, str(hdr.get_data_dtype())),
        itemsize=int(np.dtype(hdr.get_data_dtype()).itemsize),
        shape=(shape3[0], shape3[1], shape3[2], n_vol),
        qform_code=int(hdr["qform_code"]),
        sform_code=int(hdr["sform_code"]),
        descrip=bytes(hdr["descrip"]).decode("latin-1", "ignore").rstrip("\x00").strip(),
    )
    _common(info, affine, shape3, zooms)

    units = hdr.get_xyzt_units()
    tr = header_tr(hdr, warn=False) or 0.0
    info.tr = tr if n_vol > 1 or tr > 0 else 0.0

    info.slice_timing = _slice_timing_from(hdr, ext_text, shape3[2])
    info.time_offset = _time_offset_from(hdr, ext_text, units)
    slice_code = int(hdr["slice_code"]) if "slice_code" in hdr.keys() else 0
    info.slice_order = _SLICE_ORDER_NAMES.get(slice_code) if slice_code else None

    space = _afni_atr(ext_text, "TEMPLATE_SPACE")
    info.space = space or _SPACE_FROM_CODE.get(
        int(hdr["sform_code"]) or int(hdr["qform_code"]), "ORIG"
    )
    info.view = _view_from_code(int(hdr["sform_code"]) or int(hdr["qform_code"]), info.space)
    info.labels = read_brick_labels(hdr)
    info.stataux = read_brick_stataux(hdr)
    info.value_labels = read_value_labels(hdr)
    info.history = _decode_history(_afni_atr(ext_text, "HISTORY_NOTE") or "")
    info.idcode = _idcode_from(ext_text)
    # scl_slope 0 is the NIfTI spelling of "no scaling", not a zeroing scale factor.
    slope = float(hdr["scl_slope"]) if np.isfinite(hdr["scl_slope"]) else 1.0
    info.scl_slope = slope or 1.0
    info.scl_inter = float(hdr["scl_inter"]) if np.isfinite(hdr["scl_inter"]) else 0.0
    return info


def _read_mgh_info(p: Path, iname: str, indices: list[int] | None) -> DatasetInfo:
    """FreeSurfer MGH/MGZ via nibabel. A .mgz is gzipped, so this reads its
    (small) header from the front of the stream, not the voxels."""
    img: Any = nib.load(str(p))
    hdr: Any = img.header
    dims = tuple(int(d) for d in img.shape)
    shape3 = (dims + (1, 1, 1))[:3]
    n_vol = _apply_selector(dims[3] if len(dims) > 3 else 1, indices)
    dtype = np.dtype(hdr.get_data_dtype())
    info = DatasetInfo(
        path=p,
        iname=iname,
        exists=True,
        selector=indices,
        storage="MGH",
        compression="gzip" if p.name.lower().endswith(".mgz") else None,
        file_bytes=p.stat().st_size,
        datum=_DATUM_NAMES.get(dtype.name, str(dtype)),
        itemsize=int(dtype.itemsize),
        shape=(shape3[0], shape3[1], shape3[2], n_vol),
    )
    zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    _common(info, np.asarray(img.affine, dtype=np.float64), shape3, zooms)
    # MGH stores the acquisition TR (ms) even on a 3-D T1, where it is not a
    # volume spacing; it only means one for a series.
    info.tr = float(hdr["tr"]) / 1000.0 if n_vol > 1 else 0.0
    return info


def _read_afni_info(p: Path, iname: str, indices: list[int] | None) -> DatasetInfo:
    """AFNI HEAD/BRIK via nibabel's reader (the .HEAD is plain text and small)."""
    head = p if p.suffix == ".HEAD" else Path(str(p).split(".BRIK")[0] + ".HEAD")
    # nib.load's stub returns the FileBasedImage base; an AFNI .HEAD always
    # resolves to AFNIImage at runtime, which carries shape/affine/header.
    img: Any = nib.load(str(head))
    hdr: Any = img.header
    dims = tuple(int(d) for d in img.shape)
    shape3 = (dims + (1, 1, 1))[:3]
    n_vol = _apply_selector(dims[3] if len(dims) > 3 else 1, indices)
    zooms = tuple(float(z) for z in hdr.get_zooms()[:3])

    dtype = hdr.get_data_dtype()
    brik = next((c for c in head.parent.glob(head.stem + ".BRIK*")), None)
    info = DatasetInfo(
        path=p,
        iname=iname,
        exists=True,
        selector=indices,
        storage="BRIK",
        compression=_compression_of(brik) if brik else None,
        file_bytes=brik.stat().st_size if brik else head.stat().st_size,
        datum=_DATUM_NAMES.get(np.dtype(dtype).name, str(dtype)),
        itemsize=int(np.dtype(dtype).itemsize),
        shape=(shape3[0], shape3[1], shape3[2], n_vol),
    )
    _common(info, np.asarray(img.affine, dtype=np.float64), shape3, zooms)

    hinfo = getattr(hdr, "info", {}) or {}
    info.tr = float(hinfo.get("TAXIS_FLOATS", [0, 0])[1]) if "TAXIS_FLOATS" in hinfo else 0.0
    offsets = hinfo.get("TAXIS_OFFSETS")
    info.slice_timing = [float(x) for x in offsets] if offsets else None
    if "TAXIS_FLOATS" in hinfo:
        info.time_offset = float(hinfo["TAXIS_FLOATS"][0])
    info.space = str(hinfo.get("TEMPLATE_SPACE", "ORIG"))
    # A BRIK wears its view in the filename; the AFNI extension's SCENE_DATA
    # agrees, but the name is what AFNI itself dispatches on.
    info.view = next((v for v in ("+tlrc", "+acpc", "+orig") if v in head.name), "+orig")
    labs = hinfo.get("BRICK_LABS")
    info.labels = str(labs).split("~") if labs else []
    points = hinfo.get("ATLAS_LABEL_TABLE")
    dtable = hinfo.get("VALUE_LABEL_DTABLE")
    if points:
        info.value_labels = parse_atlas_points(str(points))
    elif dtable:
        info.value_labels = parse_dtable(str(dtable))
    info.history = str(hinfo.get("HISTORY_NOTE", "") or "")
    info.idcode = str(hinfo.get("IDCODE_STRING", "") or "").strip()
    return info


def _read_range(path: Path, start: int, length: int) -> bytes:
    """Bytes ``[start, start+length)`` of the *uncompressed* stream.

    Uncompressed files seek; compressed ones decompress from the front and stop
    at the end of the range. Reading one volume out of a 400-volume ``.nii.zst``
    therefore costs a fraction of the file rather than all of it — which is the
    difference between a 6-second and a sub-second ``-vis``.
    """
    sp = str(path)
    if not (sp.endswith(".gz") or sp.endswith(".zst")):
        with open(sp, "rb") as fh:
            fh.seek(start)
            return fh.read(length)
    return _read_leading_bytes(path, start + length)[start:]


def read_volume(path: str | Path, index: int = 0) -> tuple[np.ndarray, DatasetInfo]:
    """One 3-D volume of a NIfTI, reading only as far into the file as it sits.

    Returns the volume in ``(i, j, k)`` index order along with the dataset's
    :class:`DatasetInfo`. AFNI HEAD/BRIK falls back to a full read.
    """
    info = read_info(path)
    if not info.exists:
        raise FileNotFoundError(f"File not found: {path}")

    clean, _ = parse_subbrick_selector(str(path))
    p = Path(clean)
    if not info.is_nifti:
        from fastfuncstuff.io.afni import load_nifti

        arr = np.asanyarray(load_nifti(clean).dataobj)
        return (arr[..., index] if arr.ndim > 3 else arr), info

    hdr = _read_nifti_header_with_ext(p)
    dtype = np.dtype(hdr.get_data_dtype())
    nx, ny, nz, nv = info.shape
    n_vox = nx * ny * nz
    index = int(np.clip(index, 0, max(nv - 1, 0)))
    start = int(hdr["vox_offset"]) + index * n_vox * dtype.itemsize
    raw = _read_range(p, start, n_vox * dtype.itemsize)
    if len(raw) < n_vox * dtype.itemsize:
        raise ValueError(f"Truncated read: {p} volume {index}")

    vol = np.frombuffer(raw, dtype=dtype).reshape((nx, ny, nz), order="F").astype(np.float32)
    if info.scl_slope != 1.0 or info.scl_inter != 0.0:
        vol = vol * info.scl_slope + info.scl_inter
    return vol, info


@contextmanager
def _payload_stream(path: Path, zstd_threads: int | None) -> Generator[Any]:
    """The uncompressed byte stream of a NIfTI file, decoded as it is read.

    Same decoders as a full load: zstd through the command ``load_nifti`` uses
    (pzstd decodes a multi-frame file on every core, and either way the work
    is in another process, not under this one's GIL), gzip through rapidgzip
    when it is installed.
    """
    sp = str(path)
    if sp.endswith(".zst"):
        from fastfuncstuff.io.afni import _zstd_decode_command

        proc = subprocess.Popen(
            _zstd_decode_command(sp, zstd_threads),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            yield proc.stdout
        finally:
            assert proc.stdout is not None
            proc.stdout.close()
            proc.kill()
            proc.wait()
        return
    if sp.endswith(".gz"):
        try:
            import rapidgzip  # ty: ignore[unresolved-import]  # compiled, no stubs
        except ImportError:
            import gzip

            with gzip.open(sp, "rb") as fh:
                yield fh
            return
        fh = rapidgzip.open(sp, parallelization=os.cpu_count() or 1)
        try:
            yield fh
        finally:
            fh.close()  # an open rapidgzip file aborts the interpreter at exit
        return
    with open(sp, "rb") as fh:
        yield fh


def _read_into(stream: Any, view: memoryview) -> int:
    """Fill ``view`` from ``stream``; fewer bytes only at the end of the stream."""
    got = 0
    while got < len(view):
        n = stream.readinto(view[got:])
        if not n:
            break
        got += n
    return got


def iter_volumes(
    path: str | Path, *, into: np.ndarray | None = None, zstd_threads: int | None = None
) -> Iterator[np.ndarray]:
    """Every volume of a NIfTI in file order, each yielded as soon as it is decoded.

    One pass through the file, as a full load makes, but a consumer can use
    volume 0 while volume 300 is still compressed. Each volume is float32
    ``(i, j, k)`` with the header's scaling applied. Without ``into`` it is a
    buffer reused for the next volume -- copy what you keep. ``into``, a
    float32 ``(t, k, j, i)`` array, receives every volume in place (a native
    float32 file is decoded straight into it) and each yield is a view of it.
    Raises ``ValueError`` on a dataset that is not a single NIfTI file, and on
    a truncated one.
    """
    info = read_info(path)
    if not info.exists:
        raise FileNotFoundError(f"File not found: {path}")
    if not info.is_nifti or info.selector is not None:
        raise ValueError(f"not a whole NIfTI file: {path}")
    p = Path(parse_subbrick_selector(str(path))[0])
    hdr = _read_nifti_header_with_ext(p)
    dtype = np.dtype(hdr.get_data_dtype())
    nx, ny, nz, nv = info.shape
    if into is not None and (into.shape != (nv, nz, ny, nx) or into.dtype != np.float32):
        raise ValueError(f"into must be float32 {(nv, nz, ny, nx)}, not {into.dtype} {into.shape}")
    scaled = info.scl_slope != 1.0 or info.scl_inter != 0.0
    # A float32 file in this machine's byte order needs no conversion, so its
    # bytes can land where the volume will live: one copy fewer per volume.
    direct = into is not None and dtype == np.dtype(np.float32) and dtype.isnative
    raw = None if direct else bytearray(nx * ny * nz * dtype.itemsize)
    scratch = np.empty((nz, ny, nx), np.float32) if into is None else None
    with _payload_stream(p, zstd_threads) as stream:
        skip = int(hdr["vox_offset"])
        while skip:
            chunk = stream.read(min(skip, 1 << 20))
            if not chunk:
                raise ValueError(f"Truncated read: {p} header")
            skip -= len(chunk)
        for index in range(nv):
            out = into[index] if into is not None else scratch
            assert out is not None
            target = memoryview(out).cast("B") if direct else memoryview(raw)
            if _read_into(stream, target) < len(target):
                raise ValueError(f"Truncated read: {p} volume {index}")
            if not direct:
                # The file is (k, j, i)-major per volume, so this is a straight copy.
                src = np.frombuffer(raw, dtype=dtype).reshape(nz, ny, nx)  # type: ignore[arg-type]
                np.copyto(out, src, casting="unsafe")
            if scaled:
                out *= info.scl_slope
                out += info.scl_inter
            yield out.T


def data_range(path: str | Path, volume: int | None = None) -> tuple[float, float]:
    """Min/max voxel value — the one query that must touch the payload."""
    if volume is not None:
        arr = read_volume(path, volume)[0]
    else:
        from fastfuncstuff.io.afni import load_nifti

        arr = np.asanyarray(load_nifti(path).dataobj)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def same_grid(infos: list[DatasetInfo], tol: float = 1e-4) -> bool:
    """True when every dataset shares one voxel grid (shape *and* affine)."""
    if len(infos) < 2:
        return True
    first = infos[0]
    return all(
        i.shape[:3] == first.shape[:3] and np.allclose(i.affine, first.affine, atol=tol)
        for i in infos[1:]
    )
