"""Moving a layer by hand: world transforms, followers, and the .aff12.1D bridge.

A layer is drawn through its affine, so moving one is nothing more than giving
it a different affine. The file's own affine is kept beside it
(:attr:`Layer.native_affine`), and the difference between the two is the
layer's *transform*: a 4x4 in scanner millimetres (RAS) that takes where the
header puts a point to where it is now drawn. Everything that reads
``layer.affine`` -- the renderer, the readout, the time series under the
crosshair -- therefore follows a moved layer with no special case, and a saved
layer lands where it was dragged.

A layer can *follow* another (:attr:`Layer.follows`): a statistic map riding on
the functional it was computed from, so dragging the EPI onto the anatomy
carries the activation with it. The follower gets the same world transform,
applied to its own native affine; the two may be on different grids.

The bridge to the command-line tools is :func:`to_aff12` / :func:`from_aff12`.
An ``.aff12.1D`` is AFNI's base->source matrix in DICOM millimetres, computed
from *cardinal* affines (see [[project_nwarp_oblique_matrix_interop]]), while
the viewer draws with the real, possibly oblique ones. So the conversion goes
through voxel indices, which is what both sides agree on: base voxel -> the
source voxel drawn on top of it. That is what makes a matrix saved here the
same alignment ``ffs_allineate -1Dmatrix_apply`` (or ``-1Dmatrix_init``)
produces, oblique data included.
"""

from __future__ import annotations

import numpy as np

from fastfuncstuff.viewer.layers import Layer, LayerStack

IDENTITY = np.eye(4)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def translation(d) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(d, dtype=float)
    return m


def about(center, linear: np.ndarray) -> np.ndarray:
    """A 3x3 linear map applied about a point rather than about the origin."""
    m = np.eye(4)
    m[:3, :3] = linear
    c = np.asarray(center, dtype=float)
    m[:3, 3] = c - linear @ c
    return m


def axis_rotation(axis, degrees: float) -> np.ndarray:
    """3x3 rotation about a unit axis (Rodrigues), right-handed."""
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-12)
    t = np.deg2rad(degrees)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * k + (1.0 - np.cos(t)) * (k @ k)


def euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """``Rz @ Ry @ Rx`` in degrees: pitch about R-L, roll about A-P, yaw about I-S."""
    return (
        axis_rotation((0, 0, 1), rz) @ axis_rotation((0, 1, 0), ry) @ axis_rotation((1, 0, 0), rx)
    )


def euler_angles(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_matrix`, in degrees. ``ry`` stays in [-90, 90]."""
    ry = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
    if abs(np.cos(ry)) > 1e-9:
        rx = np.arctan2(rot[2, 1], rot[2, 2])
        rz = np.arctan2(rot[1, 0], rot[0, 0])
    else:  # gimbal lock: only rx - rz is defined, so put it all in rx
        rx = np.arctan2(-rot[1, 2], rot[1, 1])
        rz = 0.0
    return (float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz)))


def polar(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``linear = R @ S``: the nearest rotation and the stretch left over.

    A hand-made transform is a pure rotation and ``S`` is the identity. One that
    came from a 12-parameter fit carries scale and shear, and splitting them off
    is what lets the rotation sliders turn it without flattening them.
    """
    u, sigma, vt = np.linalg.svd(linear)
    rot = u @ vt
    if np.linalg.det(rot) < 0:  # a reflection is not a rotation; keep it in S
        u[:, -1] *= -1
        sigma = sigma.copy()
        sigma[-1] *= -1
        rot = u @ vt
    return rot, vt.T @ np.diag(sigma) @ vt


# ---------------------------------------------------------------------------
# the rigid parameters a person adjusts
# ---------------------------------------------------------------------------

#: Parameter names, in the order :func:`decompose` returns them.
PARAMS = ("tx", "ty", "tz", "rx", "ry", "rz")


def decompose(xform: np.ndarray, pivot) -> tuple[float, ...]:
    """``(tx, ty, tz, rx, ry, rz)`` for a transform, read about a pivot.

    The translation is how far the pivot has moved, which is the number a
    person means by "I shifted it 4 mm": rotating about the pivot does not
    change it. The pivot is in the layer's *native* world coordinates -- a
    point of the image, not of the screen -- so it travels with the image.
    """
    p = np.asarray(pivot, dtype=float)
    moved = xform[:3, :3] @ p + xform[:3, 3]
    rot, _ = polar(xform[:3, :3])
    return (*(float(v) for v in moved - p), *euler_angles(rot))


def compose(params, pivot, stretch: np.ndarray | None = None) -> np.ndarray:
    """Inverse of :func:`decompose`, keeping a stretch from :func:`polar`."""
    tx, ty, tz, rx, ry, rz = (float(v) for v in params)
    linear = euler_matrix(rx, ry, rz)
    if stretch is not None:
        linear = linear @ stretch
    p = np.asarray(pivot, dtype=float)
    return translation(p + np.array([tx, ty, tz])) @ about((0, 0, 0), linear) @ translation(-p)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


def native_affine(layer: Layer) -> np.ndarray:
    return np.asarray(
        layer.affine if layer.native_affine is None else layer.native_affine, dtype=float
    )


def layer_xform(layer: Layer) -> np.ndarray:
    """The world transform currently applied to a layer; identity when unmoved."""
    if layer.native_affine is None:
        return IDENTITY.copy()
    return np.asarray(layer.affine, dtype=float) @ np.linalg.inv(native_affine(layer))


def centre_mm(layer: Layer) -> np.ndarray:
    """The centre of a layer's box in its native world coordinates."""
    mid = (np.asarray(layer.shape, dtype=float) - 1.0) / 2.0
    return (native_affine(layer) @ np.append(mid, 1.0))[:3]


def followers(stack: LayerStack, key: str) -> list[Layer]:
    return [ly for ly in stack if ly.follows == key]


def _place(stack: LayerStack, key: str, xform: np.ndarray) -> None:
    layer = stack.get(key)
    native = native_affine(layer)
    if np.allclose(xform, IDENTITY, atol=1e-10):
        stack.update(key, affine=native, native_affine=None)
    else:
        stack.update(key, affine=xform @ native, native_affine=native)


def set_xform(stack: LayerStack, key: str, xform: np.ndarray) -> list[str]:
    """Move a layer and everything following it; returns the keys moved."""
    xform = np.asarray(xform, dtype=float)
    if xform.shape != (4, 4) or not np.allclose(xform[3], (0, 0, 0, 1)):
        raise ValueError("a layer transform is a 4x4 affine with a (0, 0, 0, 1) last row")
    if abs(np.linalg.det(xform[:3, :3])) < 1e-9:
        raise ValueError("a layer transform must be invertible")
    moved = [key, *(ly.key for ly in followers(stack, key))]
    for k in moved:
        _place(stack, k, xform)
    return moved


def set_follows(stack: LayerStack, key: str, parent: str | None) -> None:
    """Attach a layer to another's transform, or detach it (``parent=None``).

    Attaching adopts the parent's transform at once, so a stat map attached to
    an EPI that has already been dragged jumps to where the EPI is drawn. One
    level only: a parent cannot itself follow, or a chain would move in an
    order nothing states.
    """
    if parent is None:
        stack.update(key, follows=None)
        return
    if parent == key:
        raise ValueError("a layer cannot follow itself")
    boss = stack.get(parent)
    if boss.follows is not None:
        raise ValueError(f"{boss.name} already follows another layer")
    if followers(stack, key):
        raise ValueError(f"{stack.get(key).name} has followers of its own")
    stack.update(key, follows=parent)
    _place(stack, key, layer_xform(boss))


# ---------------------------------------------------------------------------
# .aff12.1D
# ---------------------------------------------------------------------------


def voxel_matrix(xform: np.ndarray, base_affine: np.ndarray, source_native: np.ndarray):
    """Base voxel -> source voxel under a world transform of the source."""
    return (
        np.linalg.inv(np.asarray(source_native, dtype=float))
        @ np.linalg.inv(np.asarray(xform, dtype=float))
        @ np.asarray(base_affine, dtype=float)
    )


#: RAS <-> AFNI DICOM: negate x and y.
_DICOM = np.diag([-1.0, -1.0, 1.0, 1.0])


def _cardinal(affine: np.ndarray) -> np.ndarray:
    from fastfuncstuff.io.dsetinfo import cardinal_affine

    return cardinal_affine(np.asarray(affine, dtype=np.float64))


def to_aff12(xform: np.ndarray, base_affine: np.ndarray, source_native: np.ndarray) -> np.ndarray:
    """The ``.aff12.1D`` (4x4, DICOM mm, base->source) for a moved source.

    :func:`fastfuncstuff.processing.affine.voxel_matrix_to_dicom`'s arithmetic,
    in float64 throughout: that one builds its header matrices in float32, and a
    hand alignment should not lose a hundredth of a millimetre on the way out.
    """
    ras = (
        _cardinal(source_native)
        @ voxel_matrix(xform, base_affine, source_native)
        @ np.linalg.inv(_cardinal(base_affine))
    )
    return _DICOM @ ras @ _DICOM


def from_aff12(dicom: np.ndarray, base_affine: np.ndarray, source_native: np.ndarray) -> np.ndarray:
    """The world transform that draws the source where an ``.aff12.1D`` puts it."""
    ras = _DICOM @ np.asarray(dicom, dtype=float) @ _DICOM
    ijk = np.linalg.inv(_cardinal(source_native)) @ ras @ _cardinal(base_affine)
    return (
        np.asarray(base_affine, dtype=float)
        @ np.linalg.inv(ijk)
        @ np.linalg.inv(np.asarray(source_native, dtype=float))
    )


def save_aff12(path, xform, base_affine, source_native, *, header: str | None = None) -> None:
    from fastfuncstuff.processing.affine import save_matrix_1D

    save_matrix_1D(to_aff12(xform, base_affine, source_native), path, header=header)


def load_aff12(path, base_affine, source_native) -> np.ndarray:
    """Read the first matrix of an ``.aff12.1D``, in float64.

    Parsed here rather than by ``load_matrix_1D``, which reads into float32 and
    refuses a file holding a matrix per volume -- the first row of a moco file
    is a perfectly good starting point.
    """
    rows = [
        [float(v) for v in line.split()]
        for line in open(path).read().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows or len(rows[0]) != 12:
        raise ValueError(f"{path} does not hold a 12-number .aff12.1D row")
    dicom = np.eye(4)
    dicom[:3, :4] = np.asarray(rows[0]).reshape(3, 4)
    return from_aff12(dicom, base_affine, source_native)


__all__ = [
    "PARAMS",
    "about",
    "axis_rotation",
    "centre_mm",
    "compose",
    "decompose",
    "euler_angles",
    "euler_matrix",
    "followers",
    "from_aff12",
    "layer_xform",
    "load_aff12",
    "native_affine",
    "polar",
    "save_aff12",
    "set_follows",
    "set_xform",
    "to_aff12",
    "translation",
    "voxel_matrix",
]
