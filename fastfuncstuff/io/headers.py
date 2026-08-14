"""Header-only NIfTI reading: no torch, no image payload.

Everything here is deliberately cheap to import — numpy, nibabel, stdlib — so a
question like "how many volumes does this run have?" costs milliseconds instead
of the seconds it takes to bring up the torch stack. :mod:`fastfuncstuff.io.afni`
re-exports these names, so existing callers are unaffected.

The other half of "cheap" is on-disk: :func:`_read_leading_bytes` decompresses
only as far as the header extends, even for a multi-GB ``.nii.zst``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import nibabel as nib

_NIFTI_ECODE_AFNI = 4

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


def _resolve_indices(indices: list[int], n_volumes: int) -> list[int]:
    """Resolve deferred ``$`` selectors once the volume count is known."""
    if indices and indices[0] == -1 and len(indices) == 3:
        _, start, step = indices
        return list(range(start, n_volumes, step))
    # Validate explicit indices
    for i in indices:
        if i < 0:
            raise ValueError(f"Negative volume index {i} is not valid")
        if i >= n_volumes:
            raise ValueError(f"Volume index {i} out of range for image with {n_volumes} volumes")
    return indices


def _read_leading_bytes(filepath: Path, n: int) -> bytes:
    """Read the first ``n`` uncompressed bytes of a NIfTI file, cheaply.

    For ``.nii.zst`` and ``.nii.gz`` this decompresses only as far as the reader
    consumes -- a few KB off the front of the stream, never the whole 4D payload.
    That is the whole point: reading a header should not cost a full decompress.
    """
    sp = str(filepath)
    if sp.endswith(".nii.zst"):
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
        import gzip

        with gzip.open(sp, "rb") as fh:  # decompresses lazily, only what we read
            return fh.read(n)
    with open(sp, "rb") as fh:
        return fh.read(n)


def read_nifti_header(filepath: str | Path) -> nib.Nifti1Header | nib.Nifti2Header:
    """Read just the NIfTI header, without decompressing the image payload.

    Works for ``.nii``, ``.nii.gz`` and ``.nii.zst``. Detects NIfTI-1 vs NIfTI-2
    and endianness from ``sizeof_hdr``. Any ``[selector]`` suffix is ignored here
    (selectors change the volume count, not the on-disk header); use
    :func:`nifti_shape` if you need the selector-adjusted shape.
    """
    clean_path, _ = parse_subbrick_selector(str(filepath))
    filepath = Path(clean_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

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
    txt = _afni_ext_text(img)
    if "BRICK_LABS=" in txt and 'atr_name="BRICK_LABS"' not in txt:
        return txt.split("BRICK_LABS=")[1].split("\x00")[0].split("~")
    m = _XML_LABS_RE.search(txt)
    if m:
        return [s.strip() for s in m.group(1).split("~") if s.strip()]
    return []
