"""Voxel-grid geometry: orientation, grid derivation, crop/pad, and resampling.

The header-arithmetic layer shared by ``ffs_util_autobox`` (AFNI ``3dAutobox``)
and ``ffs_util_resample`` (AFNI ``3dresample``). Both tools change a dataset's
*matrix* without moving it in space, so both need the same three things: name the
orientation the way AFNI does, derive a target grid from a request, and move data
onto that grid.

Conventions
-----------
Volumes are ``(nz, ny, nx)`` (or ``(nt, nz, ny, nx)``) as everywhere else in this
package, i.e. the reverse of NIfTI ``(i, j, k)`` axis order. Grid geometry is
expressed as ``(shape, affine)`` with a standard NIfTI voxel->world RAS affine;
functions here take/return that shape tuple and translate internally.

AFNI's dataset axes (``THD_dataxes``) are reproduced without leaving the affine:
its ``xxdel``/``xxorg`` live in DICOM/LPS coordinates (x toward L, y toward P,
z toward S) along the dataset's own index axes, which is exactly an affine column
plus its origin under the RAS->LPS sign flip. Every formula below was checked
against ``3dresample``'s header output; see ``tests/test_grid_geometry.py``.

Reference: AFNI ``src/rickr/r_new_resam_dset.c``, ``src/thd_info.c``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .interp import _separable_resample_3d, trilinear_interpolate_multi

# AFNI orientation letters name the side each axis *starts* from, so they are the
# opposite of nibabel's axcodes (which name the side an axis points toward):
# affine ('L','A','S') is AFNI "RPI".
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "I": "S", "S": "I"}

# AFNI dataxes coordinates are DICOM/LPS: +x toward L, +y toward P, +z toward S.
_RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])

BOUND_TYPES = {"FOV": 0, "SLAB": 1, "CENT_ORIG": 2, "CENT": 3}

# 3dresample -rmode codes. The first two characters are what AFNI matches on.
RMODES = {
    "nn": "nearest",
    "li": "linear",
    "cu": "cubic",
    "bk": "blocky",
    "nearest": "nearest",
    "linear": "linear",
    "cubic": "cubic",
    "blocky": "blocky",
    # Not in 3dresample, but our separable kernels are already here and are
    # strictly better for anatomical upsampling.
    "quintic": "quintic",
    "heptic": "heptic",
    "wsinc5": "wsinc5",
}


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def afni_orient_code(affine: np.ndarray) -> str:
    """AFNI 3-letter orientation code for a NIfTI affine (e.g. ``"RPI"``)."""
    import nibabel as nib

    return "".join(_OPPOSITE[c] for c in nib.orientations.aff2axcodes(affine))


def validate_orient(orient: str) -> str:
    """Normalize and check an AFNI orientation string like ``"asl"`` -> ``"ASL"``."""
    o = orient.strip().upper()
    if len(o) != 3:
        raise ValueError(f"orientation must be 3 characters, got {orient!r}")
    pairs = [{"L", "R"}, {"A", "P"}, {"I", "S"}]
    used = [False, False, False]
    for c in o:
        for p, pair in enumerate(pairs):
            if c in pair:
                if used[p]:
                    raise ValueError(f"orientation {orient!r} repeats the {sorted(pair)} axis")
                used[p] = True
                break
        else:
            raise ValueError(f"invalid orientation character {c!r} in {orient!r}")
    return o


def reorient_grid(
    shape: tuple[int, int, int], affine: np.ndarray, orient: str
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Grid that results from relabelling ``(shape, affine)`` into ``orient``.

    A pure axis permutation and flip — no voxel moves in space, so no
    interpolation is implied. Returns ``(new_shape, new_affine)``.
    """
    import nibabel as nib

    validate_orient(orient)
    target = tuple(_OPPOSITE[c] for c in orient.upper())
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(affine),
        nib.orientations.axcodes2ornt(target),
    )
    dims_xyz = (shape[2], shape[1], shape[0])
    new_affine = np.asarray(affine, dtype=np.float64) @ nib.orientations.inv_ornt_aff(
        ornt, dims_xyz
    )
    perm = np.argsort(ornt[:, 0].astype(int))  # output axis -> source axis
    new_xyz = [dims_xyz[p] for p in perm]
    return (int(new_xyz[2]), int(new_xyz[1]), int(new_xyz[0])), new_affine


def _axis_geometry(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an affine into ``(unit directions, voxel sizes, origin)`` per axis."""
    a = np.asarray(affine, dtype=np.float64)
    r = a[:3, :3]
    vox = np.linalg.norm(r, axis=0)
    if np.any(vox <= 0):
        raise ValueError("affine has a degenerate (zero-length) axis")
    return r / vox, vox, a[:3, 3].copy()


def _afni_delta_signs(directions: np.ndarray) -> np.ndarray:
    """Sign of AFNI's ``xxdel``/``yydel``/``zzdel`` for each index axis.

    Positive means the axis is oriented R2L, A2P or I2S — AFNI's dataxes run in
    DICOM/LPS, so the sign is that of the dominant LPS component. Only the
    ``CENT`` bound type reads this (it truncates toward R/A/I), but it is what
    makes "which end gets trimmed" well defined.
    """
    lps = directions * _RAS_TO_LPS[:, None]
    dominant = np.argmax(np.abs(lps), axis=0)
    return np.sign(lps[dominant, np.arange(3)])


def _daxis_resam_preserve(dold: float, dnew: float, nold: int, btype: int) -> tuple[int, float]:
    """Port of ``daxis_resam_preserve`` — the CENT/CENT_ORIG voxel-centre rule.

    ``dold``/``dnew`` are signed deltas in AFNI's frame (same sign as each other).
    Returns ``(nnew, oshift)`` with ``oshift`` in that same signed frame.
    """
    e = 0.0001
    if abs(dnew) <= abs(dold):
        # Upsample: biggest SLAB inside the old FOV, grown symmetrically.
        nsi = math.floor(0.5 * dold / dnew - e)
        if nsi < 0:
            nsi = 0
        nnew = math.floor((nold - 1) * dold / dnew + e) + 1 + 2 * nsi
        oshift = -2.0 * nsi * dnew
    else:
        # Downsample: biggest SLAB inside the old SLAB, centred.
        nnew = math.floor((nold - 1) * dold / dnew + e) + 1
        nsi = math.floor(0.5 * ((nold - 1) * abs(dold) - (nnew - 1) * abs(dnew)) / abs(dold) + e)
        oshift = nsi * dold

    if btype == BOUND_TYPES["CENT"] and dnew < 0:
        # CENT truncates toward R/A/I rather than toward the origin: apply the
        # same shift at the far end and re-derive the origin from it.
        oshift = (nold - 1) * dold - oshift - (nnew - 1) * dnew

    return int(nnew), float(oshift)


def resample_grid(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    dxyz: tuple[float, float, float],
    bound_type: int | str = "FOV",
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Grid at new voxel sizes, keeping orientation and (approximately) coverage.

    Port of AFNI ``r_dxyz_mod_dataxes``. ``dxyz`` is in *index-axis* order
    (i, j, k) of the grid passed in, matching ``3dresample -dxyz`` (which applies
    to the axes as they stand after any reorientation).

    ``bound_type`` selects what is preserved:

    * ``FOV`` (default) — the field of view ``n*delta``; the outer voxel centres
      move in or out by half the delta difference.
    * ``SLAB`` — the outer voxel centres ``(n-1)*delta``; the FOV changes instead.
    * ``CENT`` / ``CENT_ORIG`` — original voxel centres, so an integer up/down
      sample lands exactly on old centres. They differ only in which end absorbs
      the truncation: ``CENT`` trims toward R/A/I (orientation-agnostic),
      ``CENT_ORIG`` trims toward the origin.
    """
    bt = BOUND_TYPES[bound_type.upper()] if isinstance(bound_type, str) else int(bound_type)
    if bt not in BOUND_TYPES.values():
        raise ValueError(f"invalid bound_type {bound_type!r}; choose from {sorted(BOUND_TYPES)}")

    d_new = np.asarray(dxyz, dtype=np.float64).ravel()
    if d_new.size == 1:
        d_new = np.repeat(d_new, 3)
    if d_new.size != 3 or np.any(d_new <= 0):
        raise ValueError(f"dxyz must be 3 positive values, got {dxyz!r}")

    directions, vox, origin = _axis_geometry(affine)
    signs = _afni_delta_signs(directions)
    dims_xyz = np.array([shape[2], shape[1], shape[0]], dtype=int)

    new_dims = np.zeros(3, dtype=int)
    shift = np.zeros(3, dtype=np.float64)
    for a in range(3):
        n, d, dn = int(dims_xyz[a]), float(vox[a]), float(d_new[a])
        if bt == BOUND_TYPES["FOV"]:
            length = n * d
            nn = int(length / dn + 0.499)
            # offset along the index direction, from old centre back out to new edge
            off = 0.5 * (length - d) - 0.5 * (nn - 1) * dn
        elif bt == BOUND_TYPES["SLAB"]:
            length = (n - 1) * d
            nn = int(length / dn + 0.499 + 1)
            off = 0.5 * length - 0.5 * (nn - 1) * dn
        else:
            s = float(signs[a])
            nn, oshift = _daxis_resam_preserve(s * d, s * dn, n, bt)
            off = oshift * s  # back into the index-direction frame
        new_dims[a] = max(1, nn)
        shift += directions[:, a] * off

    new_affine = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = directions * d_new
    new_affine[:3, 3] = origin + shift
    return (int(new_dims[2]), int(new_dims[1]), int(new_dims[0])), new_affine


def grid_extent_rai(shape: tuple[int, int, int], affine: np.ndarray) -> tuple[float, ...]:
    """Corner-voxel-centre extent in AFNI DICOM order: ``(R, L, A, P, I, S)``.

    Matches ``THD_dset_extent`` — the min/max over the two extreme index corners
    in LPS, which is what ``3dAutobox -extent`` prints.
    """
    a = np.asarray(affine, dtype=np.float64)
    nz, ny, nx = shape
    c0 = a[:3, 3]
    c1 = a[:3, :3] @ np.array([nx - 1, ny - 1, nz - 1], dtype=np.float64) + c0
    lps0, lps1 = c0 * _RAS_TO_LPS, c1 * _RAS_TO_LPS
    lo, hi = np.minimum(lps0, lps1), np.maximum(lps0, lps1)
    return (
        float(lo[0]),
        float(hi[0]),
        float(lo[1]),
        float(hi[1]),
        float(lo[2]),
        float(hi[2]),
    )


# ---------------------------------------------------------------------------
# Grid-to-grid mapping
# ---------------------------------------------------------------------------


def voxel_map(src_affine: np.ndarray, out_affine: np.ndarray) -> np.ndarray:
    """4x4 mapping output voxel indices -> source voxel indices (both NIfTI ijk)."""
    return np.linalg.inv(np.asarray(src_affine, dtype=np.float64)) @ np.asarray(
        out_affine, dtype=np.float64
    )


def as_index_map(
    matrix: np.ndarray, rot_tol: float = 1e-4, shift_tol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decompose ``matrix`` into ``(perm, signs, offsets)`` if it is exact.

    Returns ``None`` unless the map is a signed axis permutation with whole-voxel
    offsets — i.e. unless the output grid can be filled by permuting, flipping
    and cropping/padding the source with no interpolation at all. Reorienting,
    cropping and master-grids-that-already-match all land here.

    ``src_index[perm[a]] = signs[a] * out_index[a] + offsets[perm[a]]``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    rot, trans = m[:3, :3], m[:3, 3]
    perm = np.zeros(3, dtype=int)
    signs = np.zeros(3, dtype=int)
    for a in range(3):
        col = rot[:, a]
        row = int(np.argmax(np.abs(col)))
        if abs(abs(col[row]) - 1.0) > rot_tol:
            return None
        if np.abs(np.delete(col, row)).max() > rot_tol:
            return None
        perm[a] = row
        signs[a] = 1 if col[row] > 0 else -1
    if len(set(perm.tolist())) != 3:
        return None
    if np.abs(trans - np.round(trans)).max() > shift_tol:
        return None
    return perm, signs, np.round(trans).astype(int)


def take_index_map(
    data: Tensor,
    perm: np.ndarray,
    signs: np.ndarray,
    offsets: np.ndarray,
    out_shape: tuple[int, int, int],
) -> Tensor:
    """Fill an output grid by permute/flip/crop/pad — the exact, lossless path.

    Regions of the output that fall outside the source are zero, which is what
    both ``THD_zeropad`` (autobox with a positive ``-npad``) and the resamplers
    (outside the FOV) produce. ``data`` may be ``(nz,ny,nx)`` or ``(nt,nz,ny,nx)``.
    """
    nd = data.ndim
    if nd not in (3, 4):
        raise ValueError(f"expected a 3D or 4D volume, got {nd}D")

    # NIfTI axis a lives at tensor axis nd-1-a. Permute so output axis a draws
    # from source axis perm[a].
    order = list(range(nd - 3)) + [nd - 1 - int(perm[a]) for a in (2, 1, 0)]
    out = data.permute(order)

    flip_axes = [nd - 1 - a for a in range(3) if signs[a] < 0]
    if flip_axes:
        out = torch.flip(out, flip_axes)

    # After the flip, output index o maps to position `start + o` along the axis.
    src_slices: list[slice] = [slice(None)] * (nd - 3)
    dst_slices: list[slice] = [slice(None)] * (nd - 3)
    out_xyz = (out_shape[2], out_shape[1], out_shape[0])
    for a in (2, 1, 0):
        length = out.shape[nd - 1 - a]
        off = int(offsets[int(perm[a])])
        start = off if signs[a] > 0 else length - 1 - off
        n_out = int(out_xyz[a])
        lo, hi = max(0, start), min(length, start + n_out)
        if hi <= lo:
            return torch.zeros(
                data.shape[: nd - 3] + tuple(out_shape), dtype=data.dtype, device=data.device
            )
        src_slices.append(slice(lo, hi))
        dst_slices.append(slice(lo - start, hi - start))

    result = torch.zeros(
        data.shape[: nd - 3] + tuple(out_shape), dtype=data.dtype, device=data.device
    )
    result[tuple(dst_slices)] = out[tuple(src_slices)]
    return result


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _blocky_prewarp(coords: Tensor) -> Tensor:
    """Rewrite coordinates so plain linear interpolation becomes AFNI ``-rmode Bk``.

    Blocky weights are ``(1-B(f), B(f))`` with ``B(f) = 8f^4`` below the midpoint
    and ``1-8(1-f)^4`` above — they still sum to 1 over the same two taps, so
    blocky is exactly linear evaluated at a warped fraction. Reusing the linear
    kernel keeps this a few lines instead of a fourth interpolator.
    """
    base = torch.floor(coords)
    f = coords - base
    warped = torch.where(f < 0.5, 8.0 * f.pow(4), 1.0 - 8.0 * (1.0 - f).pow(4))
    return base + warped


def _afni_cubic_edges(source: Tensor, sx: Tensor, sy: Tensor, sz: Tensor, out: Tensor) -> Tensor:
    """Drop to linear/NN in the outer shell, the way AFNI's cubic resampler does.

    The 4-tap cubic stencil spans ``[-1, +2]``, so it does not fit within one
    voxel of either end of an axis. AFNI uses linear there and NN on the very
    last voxel, rather than clamping the stencil (which is what our separable
    kernels do everywhere else). Without this, a grown FOV differs from
    ``3dresample -rmode Cu`` by several percent of range in that shell — small
    in volume, but exactly where a subsequent autobox or mask reads.
    """
    snz, sny, snx = source.shape[1:]
    xf, yf, zf = sx.reshape(-1), sy.reshape(-1), sz.reshape(-1)

    nn = (xf < 0) | (yf < 0) | (zf < 0) | (xf >= snx - 1) | (yf >= sny - 1) | (zf >= snz - 1)
    lin = ~nn & (
        (xf < 1) | (yf < 1) | (zf < 1) | (xf >= snx - 2) | (yf >= sny - 2) | (zf >= snz - 2)
    )
    in_fov = (
        (xf >= -0.5)
        & (xf <= snx - 0.5)
        & (yf >= -0.5)
        & (yf <= sny - 0.5)
        & (zf >= -0.5)
        & (zf <= snz - 0.5)
    )
    nn &= in_fov
    lin &= in_fov

    flat = out.reshape(out.shape[0], -1)
    if bool(nn.any()):
        i = nn.nonzero(as_tuple=True)[0]
        flat[:, i] = source[
            :,
            zf[i].round().long().clamp(0, snz - 1),
            yf[i].round().long().clamp(0, sny - 1),
            xf[i].round().long().clamp(0, snx - 1),
        ]
    if bool(lin.any()):
        i = lin.nonzero(as_tuple=True)[0]
        flat[:, i] = trilinear_interpolate_multi(source, xf[i], yf[i], zf[i]).T
    return flat.reshape(out.shape)


def _sample_batch(source: Tensor, sx: Tensor, sy: Tensor, sz: Tensor, interp: str) -> Tensor:
    """Sample ``(C, nz, ny, nx)`` at shared coordinates -> ``(C, *sx.shape)``.

    All kernels return 0 outside the source FOV, as AFNI's warp-on-demand does.
    """
    n_ch = source.shape[0]
    out_shape = sx.shape
    snz, sny, snx = source.shape[1:]

    if interp == "nearest":
        xi = sx.reshape(-1).round().long()
        yi = sy.reshape(-1).round().long()
        zi = sz.reshape(-1).round().long()
        oob = (xi < 0) | (xi >= snx) | (yi < 0) | (yi >= sny) | (zi < 0) | (zi >= snz)
        vals = source[:, zi.clamp(0, snz - 1), yi.clamp(0, sny - 1), xi.clamp(0, snx - 1)]
        vals[:, oob] = 0.0
        return vals.reshape((n_ch, *out_shape))

    if interp in ("linear", "blocky"):
        if interp == "blocky":
            sx, sy, sz = _blocky_prewarp(sx), _blocky_prewarp(sy), _blocky_prewarp(sz)
        xf, yf, zf = sx.reshape(-1), sy.reshape(-1), sz.reshape(-1)
        # grid_sample clamps at the border, which reproduces AFNI's NN fallback on
        # the outermost voxel; anything past the half-voxel FOV edge is zeroed.
        vals = trilinear_interpolate_multi(source, xf, yf, zf).T  # (C, N)
        oob = (
            (xf < -0.5)
            | (xf > snx - 0.5)
            | (yf < -0.5)
            | (yf > sny - 0.5)
            | (zf < -0.5)
            | (zf > snz - 0.5)
        )
        vals = vals.clone()
        vals[:, oob] = 0.0
        return vals.reshape((n_ch, *out_shape))

    if n_ch == 1:
        out = _separable_resample_3d(source[0], sx, sy, sz, interp)[None]
    else:
        out = _separable_resample_3d(source, sx, sy, sz, interp)
    if interp == "cubic":
        out = _afni_cubic_edges(source, sx, sy, sz, out)
    return out


def resample_to_grid(
    data: Tensor,
    src_affine: np.ndarray,
    out_shape: tuple[int, int, int],
    out_affine: np.ndarray,
    interp: str = "nearest",
    verbose: int = 0,
) -> Tensor:
    """Move ``data`` onto the grid ``(out_shape, out_affine)``.

    Takes the exact permute/flip/crop path whenever the two grids differ only by
    axis order and whole-voxel offsets (reorients, crops, matching masters), so
    those stay bit-exact regardless of ``interp``. Otherwise interpolates, time-
    chunked to fit the device.
    """
    interp = RMODES.get(interp.lower(), interp.lower())
    if interp not in set(RMODES.values()):
        raise ValueError(f"unknown interpolation mode {interp!r}")

    m = voxel_map(src_affine, out_affine)
    exact = as_index_map(m)
    if exact is not None:
        return take_index_map(data, *exact, out_shape)

    is_4d = data.ndim == 4
    src = data if is_4d else data[None]
    nt = src.shape[0]
    device = src.device
    onz, ony, onx = out_shape

    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=torch.float32, device=device),
        torch.arange(ony, dtype=torch.float32, device=device),
        torch.arange(onx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    sx = mt[0, 0] * ii + mt[0, 1] * jj + mt[0, 2] * kk + mt[0, 3]
    sy = mt[1, 0] * ii + mt[1, 1] * jj + mt[1, 2] * kk + mt[1, 3]
    sz = mt[2, 0] * ii + mt[2, 1] * jj + mt[2, 2] * kk + mt[2, 3]
    del kk, jj, ii

    from fastfuncstuff.memory import compute_moco_resample_batch_size

    batch = compute_moco_resample_batch_size(
        onz, ony, onx, nt, device, interp=interp if interp in ("cubic", "wsinc5") else "linear"
    )
    out = torch.zeros((nt, onz, ony, onx), dtype=src.dtype, device=device)

    chunks = range(0, nt, batch)
    bar = None
    if verbose >= 1 and nt > 1 and len(range(0, nt, batch)) > 1:
        from tqdm import tqdm

        bar = tqdm(total=nt, desc=f"resample ({interp})", unit="vol", leave=True)
    for t0 in chunks:
        t1 = min(nt, t0 + batch)
        out[t0:t1] = _sample_batch(src[t0:t1], sx, sy, sz, interp)
        if bar is not None:
            bar.update(t1 - t0)
    if bar is not None:
        bar.close()

    return out if is_4d else out[0]


def crop_affine(affine: np.ndarray, offsets_xyz: tuple[int, int, int]) -> np.ndarray:
    """Affine of a grid whose voxel (0,0,0) sits at source index ``offsets_xyz``.

    Negative offsets describe a zero-padded (grown) grid. Only the origin moves —
    the dataset stays exactly where it was in space.
    """
    a = np.asarray(affine, dtype=np.float64).copy()
    a[:3, 3] = a[:3, :3] @ np.asarray(offsets_xyz, dtype=np.float64) + a[:3, 3]
    return a
