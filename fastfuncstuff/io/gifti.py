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

import numpy as np

__all__ = [
    "load_gifti_data",
    "load_gifti_surface",
    "mesh_fingerprint",
    "save_gifti_data",
    "save_gifti_surface",
]


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
    arrays = [
        nib.gifti.GiftiDataArray(
            np.ascontiguousarray(cols[:, t]), intent=intent, datatype="NIFTI_TYPE_FLOAT32"
        )
        for t in range(cols.shape[1])
    ]
    img = nib.gifti.GiftiImage(
        darrays=arrays, meta=nib.gifti.GiftiMetaData({k: str(v) for k, v in (meta or {}).items()})
    )
    nib.save(img, os.fspath(path))


def load_gifti_data(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, str]]:
    """``(V, T)`` (or ``(V,)`` for one array) and the file-level metadata."""
    import nibabel as nib

    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    cols = [np.asarray(a.data, np.float32) for a in img.darrays]
    data = cols[0] if len(cols) == 1 else np.stack(cols, axis=1)
    return data, dict(img.meta)


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
    img = nib.gifti.GiftiImage(
        darrays=[pts, tri], meta=nib.gifti.GiftiMetaData({k: str(v) for k, v in m.items()})
    )
    nib.save(img, os.fspath(path))


def load_gifti_surface(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """``(vertices, faces, metadata)`` of a ``.surf.gii``."""
    import nibabel as nib

    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    pts = img.agg_data("NIFTI_INTENT_POINTSET")
    tri = img.agg_data("NIFTI_INTENT_TRIANGLE")
    return np.asarray(pts, np.float64), np.asarray(tri, np.int64), dict(img.meta)
