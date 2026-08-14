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

import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fastfuncstuff.io.headers import (
    _afni_ext_text,
    _read_leading_bytes,
    _resolve_indices,
    parse_subbrick_selector,
    read_brick_labels,
)

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

    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    orient: str = "???"
    space: str = "ORIG"
    obliquity: float = 0.0
    qform_code: int = 0
    sform_code: int = 0

    labels: list[str] = field(default_factory=list)
    history: str = ""
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


def _decode_history(text: str) -> str:
    """AFNI-XML history → plain text: XML entities out, escaped newlines real."""
    import html

    return html.unescape(text).replace("\\n", "\n").strip()


def _read_nifti_header_with_ext(path: Path) -> nib.Nifti1Header | nib.Nifti2Header:
    """Header *plus* extensions, reading only as far as the image data starts.

    Two peeks: 544 bytes to learn ``vox_offset``, then that many bytes so
    nibabel's own extension parser sees the whole AFNI blob and nothing else.
    """
    lead = _read_leading_bytes(path, 544)
    for endian in ("<", ">"):
        (sizeof_hdr,) = struct.unpack(endian + "i", lead[:4])
        if sizeof_hdr in (348, 540):
            break
    else:
        raise ValueError(f"Not a recognizable NIfTI-1/2 header (bad sizeof_hdr): {path}")

    klass: Any = nib.Nifti1Header if sizeof_hdr == 348 else nib.Nifti2Header
    stub = klass(binaryblock=lead[:sizeof_hdr], check=False)
    vox_offset = int(stub["vox_offset"]) or sizeof_hdr + 4
    # Cap the extension read: a corrupt vox_offset should not pull the payload in.
    raw = lead if vox_offset <= len(lead) else _read_leading_bytes(path, min(vox_offset, 1 << 22))
    return klass.from_fileobj(BytesIO(raw), check=False)


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


def read_info(path: str | Path) -> DatasetInfo:
    """Describe a dataset from its header alone (no image payload is read).

    Accepts ``.nii`` / ``.nii.gz`` / ``.nii.zst`` and AFNI ``.HEAD`` / ``.BRIK``,
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
    return _read_nifti_info(p, raw_arg, indices)


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
    tr = float(hdr.get_zooms()[3]) if len(hdr.get_zooms()) > 3 else 0.0
    if units and len(units) > 1 and units[1] == "msec":
        tr /= 1000.0
    elif units and len(units) > 1 and units[1] == "usec":
        tr /= 1e6
    info.tr = tr if n_vol > 1 or tr > 0 else 0.0

    info.slice_timing = _slice_timing_from(hdr, ext_text, shape3[2])
    slice_code = int(hdr["slice_code"]) if "slice_code" in hdr.keys() else 0
    info.slice_order = _SLICE_ORDER_NAMES.get(slice_code) if slice_code else None

    space = _afni_atr(ext_text, "TEMPLATE_SPACE")
    info.space = space or _SPACE_FROM_CODE.get(
        int(hdr["sform_code"]) or int(hdr["qform_code"]), "ORIG"
    )
    info.labels = read_brick_labels(hdr)
    info.history = _decode_history(_afni_atr(ext_text, "HISTORY_NOTE") or "")
    # scl_slope 0 is the NIfTI spelling of "no scaling", not a zeroing scale factor.
    slope = float(hdr["scl_slope"]) if np.isfinite(hdr["scl_slope"]) else 1.0
    info.scl_slope = slope or 1.0
    info.scl_inter = float(hdr["scl_inter"]) if np.isfinite(hdr["scl_inter"]) else 0.0
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
    info.space = str(hinfo.get("TEMPLATE_SPACE", "ORIG"))
    labs = hinfo.get("BRICK_LABS")
    info.labels = str(labs).split("~") if labs else []
    info.history = str(hinfo.get("HISTORY_NOTE", "") or "")
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
