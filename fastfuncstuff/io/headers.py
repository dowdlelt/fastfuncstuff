"""Header-only NIfTI reading: no torch, no image payload.

Everything here is deliberately cheap to import — numpy, nibabel, stdlib — so a
question like "how many volumes does this run have?" costs milliseconds instead
of the seconds it takes to bring up the torch stack. :mod:`fastfuncstuff.io.afni`
re-exports these names, so existing callers are unaffected.

The other half of "cheap" is on-disk: :func:`_read_leading_bytes` decompresses
only as far as the header extends, even for a multi-GB ``.nii.zst``.
"""

from __future__ import annotations

import math
import re
import struct
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

_NIFTI_ECODE_AFNI = 4

# NIfTI sform/qform xform codes (nifti1.h). For a NIfTI dataset AFNI derives the
# +orig/+tlrc VIEW from these codes, not from the AFNI extension -- see
# thd_niftiread.c:NIFTI_code_to_view and io/afni.py:_resolve_space_codes.
_NIFTI_XFORM_UNKNOWN = 0
_NIFTI_XFORM_SCANNER_ANAT = 1
_NIFTI_XFORM_ALIGNED_ANAT = 2
_NIFTI_XFORM_TALAIRACH = 3
_NIFTI_XFORM_MNI_152 = 4
_NIFTI_XFORM_TEMPLATE_OTHER = 5

_XML_LABS_RE = re.compile(r'atr_name\s*=\s*"BRICK_LABS"[^>]*>\s*"([^"]+)"', re.S)
_XML_STATAUX_RE = re.compile(
    r'<AFNI_atr[^>]*atr_name\s*=\s*"BRICK_STATAUX"[^>]*>\s*([0-9eE.+\-\s]+?)\s*</AFNI_atr>',
    re.S,
)


def parse_subbrick_selector(path: str | Path) -> tuple[str, list[int] | None]:
    """Parse AFNI-style sub-brick selectors from a file path.

    Supports the following selector syntax appended to a file path:

    - ``file.nii.gz[0]``            — single volume
    - ``file.nii.gz[1,3,5]``        — specific volumes
    - ``file.nii.gz[0..10]``        — range (inclusive)
    - ``file.nii.gz[0..$]``         — range to last volume (resolved later)
    - ``file.nii.gz[0..$(2)]``      — every 2nd volume (step)
    - ``file.nii.gz[0..10(3)]``     — range with step

    Quoting with single quotes around the selector (shell-style) is stripped
    automatically: ``file.nii.gz'[0..5]'`` works the same as ``file.nii.gz[0..5]``.

    Parameters
    ----------
    path : str or Path
        File path, optionally with a ``[selector]`` suffix.

    Returns
    -------
    clean_path : str
        The file path with the selector removed.
    indices : list[int] or None
        Resolved volume indices, or *None* if no selector was present.
        A ``$`` end-point is stored as -1 and resolved at load time.
    """
    path = str(path).strip().rstrip("'")
    # Find the bracket selector — scan from the right to avoid matching
    # brackets that might be in directory names.
    bracket_start = path.rfind("[")
    if bracket_start == -1:
        return path, None

    bracket_end = path.rfind("]")
    if bracket_end == -1 or bracket_end < bracket_start:
        return path, None

    # Strip optional leading quote before '['
    clean_path = path[:bracket_start].rstrip("'")
    selector = path[bracket_start + 1 : bracket_end]

    return clean_path, _parse_selector(selector)


_1D_SELECTOR_RE = re.compile(r"(\[[^\[\]]*\]|\{[^{}]*\})\s*'?\s*$")


def parse_1d_selectors(path: str | Path) -> tuple[str, list[int] | None, list[int] | None]:
    """Parse AFNI-style ``[columns]`` and ``{rows}`` selectors off a text-file path.

    For an ASCII/1D file AFNI reads ``[..]`` as a **column** selector and
    ``{..}`` as a **row** selector — the opposite assignment to the one that
    feels natural if you come at it from ``dataset[0..100]``, where the brackets
    pick volumes. ``mri_read.c`` states it outright: "column selectors [..] and
    row selectors {..} are allowed in fname". Matching AFNI here is the whole
    point: the same string has to mean the same thing to ``3dREMLfit``, or a
    file that works in one tool silently selects something else in the other.

    Both may appear, in either order, and each accepts the full selector
    grammar :func:`parse_subbrick_selector` already supports — ``[0]``,
    ``[1,3,5]``, ``[0..7]``, ``[0..$]``, ``[0..$(2)]``.

    Examples
    --------
    - ``pcs.1D[0..7]``          — first 8 columns (e.g. the first 8 PCs)
    - ``pcs.1D[0,2,5]``         — those three columns
    - ``pcs.1D{10..$}``         — drop the first 10 rows
    - ``pcs.1D[0..7]{10..$}``   — both

    Returns
    -------
    clean_path : str
        The path with every selector removed.
    columns, rows : list[int] or None
        Resolved indices, or *None* where that selector was absent. A ``$``
        endpoint is left as the deferred marker for :func:`_resolve_indices`.
    """
    base, suffix = split_1d_selector_suffix(path)
    columns: list[int] | None = None
    rows: list[int] | None = None
    for token in re.findall(r"\[[^\[\]]*\]|\{[^{}]*\}", suffix):
        parsed = _parse_selector(token[1:-1])
        if token[0] == "[":
            columns = parsed
        else:
            rows = parsed
    return base, columns, rows


def split_1d_selector_suffix(path: str | Path) -> tuple[str, str]:
    """Split a path into ``(base, trailing selector text)`` without parsing it.

    Needed because :mod:`glob` reads ``[0..8]`` as a character class, so a
    pattern carrying a column selector matches nothing at all. A glob caller
    splits first, globs the bare pattern, then re-attaches ``suffix`` to each
    match so the loader still sees it.
    """
    text = str(path).strip()
    tokens: list[str] = []
    # Strip from the right, so both selectors come off whichever order they are
    # written in, and a directory name containing a bracket is never touched.
    while True:
        match = _1D_SELECTOR_RE.search(text)
        if match is None:
            break
        tokens.append(match.group(1))
        text = text[: match.start()].rstrip().rstrip("'").rstrip()
    return text.rstrip("'"), "".join(reversed(tokens))


def apply_1d_selectors(
    data: np.ndarray,
    columns: list[int] | None,
    rows: list[int] | None,
    source: str = "file",
) -> np.ndarray:
    """Apply parsed 1D selectors to a ``(n_rows, n_columns)`` array."""
    if rows is not None:
        try:
            data = data[_resolve_indices(rows, data.shape[0], noun="row"), :]
        except ValueError as exc:
            raise ValueError(f"{source}: {{rows}} selector: {exc}") from exc
    if columns is not None:
        try:
            data = data[:, _resolve_indices(columns, data.shape[1], noun="column")]
        except ValueError as exc:
            raise ValueError(f"{source}: [columns] selector: {exc}") from exc
    return data


def _parse_selector(selector: str) -> list[int]:
    """Parse the content inside brackets into a list of volume indices.

    ``$`` is stored as -1 to be resolved once the number of volumes is known.
    """
    selector = selector.strip()

    # Comma-separated list: 0,1,3,5
    if "," in selector:
        return [int(s.strip()) for s in selector.split(",")]

    # Range: start..end  or  start..end(step)  or  start..$(step)
    if ".." in selector:
        range_part, _, step_part = selector.partition("(")
        step = 1
        if step_part:
            step = int(step_part.rstrip(")"))

        start_str, end_str = range_part.split("..", 1)
        start = int(start_str.strip())

        end_str = end_str.strip()
        if end_str == "$" or end_str == "":
            # -1 sentinel → resolve at load time
            return _range_with_sentinel(start, -1, step)

        end = int(end_str)
        return list(range(start, end + 1, step))  # inclusive end, like AFNI

    # Single index: 0
    return [int(selector)]


def _range_with_sentinel(start: int, end: int, step: int) -> list[int]:
    """Build a range list; if *end* is -1, return a marker for deferred resolution."""
    if end == -1:
        # Store as (start, sentinel, step) encoded in a list with a negative marker.
        # Convention: [-1, start, step] — the -1 first element flags deferred.
        return [-1, start, step]
    return list(range(start, end + 1, step))


def _resolve_indices(indices: list[int], n_volumes: int, noun: str = "volume") -> list[int]:
    """Resolve deferred ``$`` selectors once the volume count is known.

    ``noun`` names the axis in error messages. The same grammar selects volumes
    from a dataset and columns from a 1D file, and "volume index 10 out of range
    for image with 10 volumes" is baffling when what you selected was a column.
    """
    if indices and indices[0] == -1 and len(indices) == 3:
        _, start, step = indices
        return list(range(start, n_volumes, step))
    # Validate explicit indices
    for i in indices:
        if i < 0:
            raise ValueError(f"Negative {noun} index {i} is not valid")
        if i >= n_volumes:
            raise ValueError(
                f"{noun} index {i} out of range: there are {n_volumes} {noun}s (0..{n_volumes - 1})"
            )
    return indices


def _read_leading_bytes(filepath: Path, n: int) -> bytes:
    """Read the first ``n`` uncompressed bytes of a NIfTI file, cheaply.

    For ``.nii.zst`` and ``.nii.gz`` this decompresses only as far as the reader
    consumes -- a few KB off the front of the stream, never the whole 4D payload.
    That is the whole point: reading a header should not cost a full decompress.
    """
    sp = str(filepath)
    if sp.endswith(".nii.zst"):
        try:
            from compression import zstd  # ty: ignore[unresolved-import]  # stdlib, 3.14+
        except ImportError:
            pass
        else:
            # In-process: ~0.15 ms a header against ~6 ms to spawn zstd, which
            # is what made reading a 650-file results directory take 8 s.
            return _zstd_leading_bytes(sp, n, zstd)
        # Stream zstd and stop as soon as we have enough. Closing the pipe early
        # sends SIGPIPE to zstd, so it never decompresses past the first block.
        proc = subprocess.Popen(
            ["zstd", "-dc", sp], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        try:
            assert proc.stdout is not None
            buf = proc.stdout.read(n)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.terminate()
            proc.wait()
        return buf
    if sp.endswith(".gz"):
        return _gzip_leading_bytes(sp, n)
    with open(sp, "rb") as fh:
        return fh.read(n)


#: Compressed bytes per read when decoding a header in process. zstd emits nothing
#: until a whole block has arrived, and a block holds up to 128 KiB, so a smaller
#: read only buys a second one.
_LEAD_CHUNK = (1 << 17) + 1024


def _zstd_leading_bytes(path: str, n: int, zstd: Any) -> bytes:
    """The first ``n`` bytes of a zstd stream, decoded straight off the file.

    The same bytes ``ZstdFile`` returns, without its two buffering layers. Frames
    are walked by hand because pzstd opens with a skippable frame that decodes to
    nothing, then writes one frame per job. A file that ends inside a frame
    raises ``EOFError``, as ``ZstdFile`` does.
    """
    out = bytearray()
    dec = zstd.ZstdDecompressor()
    pending, fed = b"", False
    with open(path, "rb", buffering=0) as fh:
        while len(out) < n:
            if dec.eof:
                pending, fed = dec.unused_data, False
                dec = zstd.ZstdDecompressor()
            if dec.needs_input and not pending:
                pending = fh.read(_LEAD_CHUNK)
                if not pending:
                    if fed:
                        raise EOFError(f"{path}: ends inside a zstd frame")
                    break
            if pending:
                fed = True
            out += dec.decompress(pending, max_length=n - len(out))
            pending = b""
    return bytes(out)


def _gzip_leading_bytes(path: str, n: int) -> bytes:
    """The first ``n`` bytes of a gzip stream, decoded straight off the file.

    ``gzip.open`` returns the same bytes at about twice the cost per header. Like
    it, this reads on through concatenated members, skips zero padding between
    them, and raises ``EOFError`` on a file that ends inside one.
    """
    import zlib

    out = bytearray()
    dec = zlib.decompressobj(31)
    pending, fed = b"", False
    with open(path, "rb", buffering=0) as fh:
        while len(out) < n:
            if not fed:
                pending = pending.lstrip(b"\x00")
            if not pending:
                pending = fh.read(1 << 16)
                if not pending:
                    if fed:
                        raise EOFError(f"{path}: ends inside a gzip member")
                    break
                continue
            # n - len(out) is never 0 here, which zlib would read as "no limit".
            out += dec.decompress(pending, n - len(out))
            fed = True
            pending = dec.unconsumed_tail
            if dec.eof:
                pending, fed = dec.unused_data, False
                dec = zlib.decompressobj(31)
    return bytes(out)


#: Bytes read on the first look at a header. AFNI extensions (history, labels,
#: stat tables) are a few KB; a second read is a second zstd spawn where the
#: stdlib module is missing, so the first one asks for plenty. Asking for less
#: saves nothing in process: 4 KB and 64 KB peeks timed the same, because zstd
#: decodes a whole block either way.
_HEADER_PEEK = 1 << 16


def _read_nifti_header_with_ext(path: Path) -> nib.Nifti1Header | nib.Nifti2Header:
    """Header *plus* extensions, reading only as far as the image data starts.

    One peek generous enough to hold the extensions of nearly every file, and a
    second only when ``vox_offset`` says they run past it; nibabel's own
    extension parser then sees the whole AFNI blob and nothing else.
    """
    lead = _read_leading_bytes(path, _HEADER_PEEK)
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
    return klass.from_fileobj(BytesIO(raw[:vox_offset]), check=False)


def read_nifti_header(
    filepath: str | Path, *, extensions: bool = False
) -> nib.Nifti1Header | nib.Nifti2Header:
    """Read just the NIfTI header, without decompressing the image payload.

    Works for ``.nii``, ``.nii.gz`` and ``.nii.zst``. Detects NIfTI-1 vs NIfTI-2
    and endianness from ``sizeof_hdr``. Any ``[selector]`` suffix is ignored here
    (selectors change the volume count, not the on-disk header); use
    :func:`nifti_shape` if you need the selector-adjusted shape.

    The default stops at the fixed 348/540 bytes, so the header carries no
    extensions. ``extensions=True`` reads on to ``vox_offset``, which anything
    after the AFNI attributes (labels, stat types, ClustSim tables) needs.
    """
    clean_path, _ = parse_subbrick_selector(str(filepath))
    filepath = Path(clean_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if filepath.name.lower().endswith(".gii"):
        # GIfTI has no NIfTI header: build the in-memory one (XML, no payload to skip).
        from fastfuncstuff.io.gifti import gifti_as_nifti

        return gifti_as_nifti(filepath).header

    if extensions:
        return _read_nifti_header_with_ext(filepath)

    # NIfTI-1 header is 348 bytes, NIfTI-2 is 540; read enough for either.
    raw = _read_leading_bytes(filepath, 544)

    # sizeof_hdr (int32 at offset 0) disambiguates version *and* endianness.
    import struct

    for endian in ("<", ">"):
        (sizeof_hdr,) = struct.unpack(endian + "i", raw[:4])
        if sizeof_hdr == 348:
            return nib.Nifti1Header(binaryblock=raw[:348])
        if sizeof_hdr == 540:
            return nib.Nifti2Header(binaryblock=raw[:540])
    raise ValueError(f"Not a recognizable NIfTI-1/2 header (bad sizeof_hdr) in {filepath}")


_TIME_UNIT_SCALE = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
PLAUSIBLE_TR_RANGE_S = (0.05, 20.0)


def header_tr(
    img_or_header: Any, *, require_units: bool = False, warn: bool = True
) -> float | None:
    """The repetition time in SECONDS from an image or its header.

    ``pixdim[4]`` alone is not a TR: NIfTI stores it in whatever unit
    ``xyzt_units`` declares, and a scanner-converted ``msec`` header reads 2000
    for a 2 s TR. Every reader goes through here so that unit is applied once.

    Returns None when there is no time axis (fewer than four zooms, or a
    non-positive step). A header whose time unit is unset or non-temporal is
    taken as seconds -- the overwhelmingly common case -- unless
    ``require_units`` is set, for callers that must tell a declared TR apart
    from NIfTI's default ``pixdim[4] = 1``; those get None instead.

    A value outside :data:`PLAUSIBLE_TR_RANGE_S` is returned but warned about:
    3-D EPI headers often hold the shot time, and a mislabelled unit is
    otherwise silent until the design is already wrong. ``warn=False`` is for
    tools that only report the value.
    """
    import math
    import warnings

    hdr = getattr(img_or_header, "header", img_or_header)
    try:
        zooms = hdr.get_zooms()
    except Exception:
        return None
    if len(zooms) < 4:
        return None
    raw = float(zooms[3])
    if not math.isfinite(raw) or raw <= 0:
        return None
    try:
        unit = hdr.get_xyzt_units()[1]
    except Exception:  # AFNI HEAD headers: nibabel already reports seconds
        unit = "sec"
    scale = _TIME_UNIT_SCALE.get(unit)
    if scale is None:
        if require_units:
            return None
        scale = 1.0
    tr = raw * scale
    lo, hi = PLAUSIBLE_TR_RANGE_S
    if warn and not lo <= tr <= hi:
        warnings.warn(
            f"header TR is {tr:g} s (pixdim[4]={raw:g}, time unit {unit!r}), outside the "
            f"plausible fMRI range {lo:g}-{hi:g} s; pass -tr explicitly if this is wrong",
            stacklevel=2,
        )
    return tr


def nifti_shape(filepath: str | Path) -> tuple[int, ...]:
    """Return a NIfTI image's shape without decompressing its payload.

    Honours ``[selector]`` suffixes (adjusts the 4th-dim volume count), so it is a
    drop-in replacement for ``load_nifti(path).shape`` on the hot paths that only
    need dimensions (run-structure timing, mask/data grid checks).
    """
    clean_path, indices = parse_subbrick_selector(str(filepath))
    hdr = read_nifti_header(clean_path)
    shape = tuple(int(d) for d in hdr.get_data_shape())
    if indices is not None:
        if len(shape) < 4:
            raise ValueError(
                f"Sub-brick selector requires a 4D image, got {len(shape)}D: {clean_path}"
            )
        n_volumes = shape[3]
        resolved = _resolve_indices(indices, n_volumes)
        shape = shape[:3] + (len(resolved),) + shape[4:]
    return shape


def _afni_ext_text(img) -> str:
    """Concatenated text of the AFNI NIfTI extension(s), or ``""`` if absent.

    Handles both plain-text (``BRICK_LABS=…\\x00``) and AFNI-XML payloads; the
    read-side counterpart to :func:`_set_afni_brick_stataux` /
    :func:`_set_afni_brick_labels`.
    """
    header = getattr(img, "header", img)
    out = []
    for ext in getattr(header, "extensions", None) or []:
        if ext.get_code() != _NIFTI_ECODE_AFNI:
            continue
        content = ext.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        out.append(content)
    return "\n".join(out)


def read_brick_stataux(img) -> dict[int, tuple[int, tuple[float, ...]]]:
    """Parse ``BRICK_STATAUX`` from a NIfTI image → ``{idx: (code, params)}``.

    Empty dict when the bucket carries no per-sub-brick stat metadata. Inverse of
    :func:`_set_afni_brick_stataux`.
    """
    m = _XML_STATAUX_RE.search(_afni_ext_text(img))
    if not m:
        return {}
    floats = [float(x) for x in m.group(1).split()]
    out: dict[int, tuple[int, tuple[float, ...]]] = {}
    i = 0
    while i + 3 <= len(floats):
        idx = int(floats[i])
        code = int(floats[i + 1])
        n_par = int(floats[i + 2])
        out[idx] = (code, tuple(floats[i + 3 : i + 3 + n_par]))
        i += 3 + n_par
    return out


def read_brick_labels(img) -> list[str]:
    """Pull sub-brick labels out of an AFNI NIfTI extension (``[]`` if absent).

    Accepts legacy plain-text (``BRICK_LABS=a~b~…``) and AFNI-XML payloads.
    """
    return _brick_labels_from_text(_afni_ext_text(img))


def _brick_labels_from_text(txt: str) -> list[str]:
    if "BRICK_LABS=" in txt and 'atr_name="BRICK_LABS"' not in txt:
        return txt.split("BRICK_LABS=")[1].split("\x00")[0].split("~")
    m = _XML_LABS_RE.search(txt)
    if m:
        return [s.strip() for s in m.group(1).split("~") if s.strip()]
    return []


@dataclass(frozen=True)
class NiftiSummary:
    """What a file list shows of a NIfTI: see :func:`peek_nifti_summary`."""

    shape: tuple[int, int, int]
    n_volumes: int
    tr: float
    labels: tuple[str, ...]


def peek_nifti_summary(path: Path) -> NiftiSummary | None:
    """Shape, volume count, TR and sub-brick labels of a NIfTI-1, or ``None``.

    The viewer's catalog needs only these. Going through nibabel and
    ``dsetinfo.read_info`` costs about 1 ms a file, most of it geometry the list
    never shows; parsed straight from the header bytes they cost a fifth of that.

    ``None`` means "not the common case": NIfTI-2, a selector in the name, an
    odd ``dim``, a non-finite TR, an extension that runs past the first read,
    or a file that will not read at all. The caller then takes the full route,
    so this never has to be right about anything unusual. Where it does answer,
    it must agree with ``read_info`` exactly, which ``tests/test_dsetinfo.py``
    checks.
    """
    if "[" in path.name:
        return None
    try:
        lead = _read_leading_bytes(path, _HEADER_PEEK)
    except Exception:
        return None
    if len(lead) < 352:
        return None
    for endian in ("<", ">"):
        if struct.unpack_from(endian + "i", lead)[0] == 348:
            break
    else:
        return None
    dim = struct.unpack_from(endian + "8h", lead, 40)
    ndim = dim[0]
    if not 1 <= ndim <= 7:
        return None
    dims = dim[1 : ndim + 1]
    if min(dims) <= 0:
        return None
    vox_offset = int(struct.unpack_from(endian + "f", lead, 108)[0]) or 352
    if not 352 <= vox_offset <= len(lead):
        return None

    # As header_tr reads it: pixdim[4] in xyzt_units' time unit, seconds when
    # the unit is unset or not a time.
    tr = 0.0
    if ndim >= 4:
        raw = struct.unpack_from(endian + "f", lead, 92)[0]
        if not math.isfinite(raw):
            return None
        if raw > 0:
            unit = {8: "sec", 16: "msec", 24: "usec"}.get(lead[123] & 0x38)
            tr = raw * _TIME_UNIT_SCALE.get(unit or "", 1.0)

    labels: list[str] = []
    if vox_offset > 352 and lead[348]:
        texts, pos = [], 352
        while pos + 8 <= vox_offset:
            esize, ecode = struct.unpack_from(endian + "2i", lead, pos)
            if esize < 16 or esize % 16 or pos + esize > vox_offset:
                return None
            if ecode == _NIFTI_ECODE_AFNI:
                texts.append(lead[pos + 8 : pos + esize].decode("utf-8", errors="ignore"))
            pos += esize
        if texts:
            labels = _brick_labels_from_text("\n".join(texts))

    shape3 = (dims + (1, 1, 1))[:3]
    n_vol = dims[3] if ndim > 3 else 1
    # dim[5] carries the sub-brick count for AFNI buckets written as "3D+t of 1".
    if ndim > 4 and dims[4] > 1 and n_vol == 1:
        n_vol = dims[4]
    return NiftiSummary(
        shape=(int(shape3[0]), int(shape3[1]), int(shape3[2])),
        n_volumes=int(n_vol),
        tr=float(tr),
        labels=tuple(labels),
    )
