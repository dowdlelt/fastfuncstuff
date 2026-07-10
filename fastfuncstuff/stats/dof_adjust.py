"""Degrees-of-freedom adjustment for statistical buckets (post-NORDIC).

NORDIC denoising removes noise components, which costs residual degrees of
freedom that the GLM never sees: ``ffs_reml`` labels its t/F sub-bricks with the
*model* dof, so AFNI reports p-values that are too optimistic. If you carry the
per-voxel count of removed components through the pipeline, this module rewrites
the statistics with the corrected dof.

For each statistical sub-brick it computes ``new_dof = dof - adjustment`` (a
scalar, or a per-voxel integer map) and converts the statistic to a **z-score**
at that dof, matching AFNI's ``THD_stat_to_zscore`` exactly (``mri_stats.c``):

* t-stat  → ``z = sign(t) · qginv(P(|T| > |t|; dof)/2)`` = signed, one-tailed.
* F-stat  → ``z = qginv(P(F > f; dfn, dfd)/2)`` (AFNI treats F as 2-sided);
  only the **denominator** (error) dof is reduced by NORDIC, dfn is unchanged.
* ``qginv`` is the inverse upper-tail normal, clamped at 13 sigma like AFNI.

The z-score is inserted **after** each stat sub-brick, so
``Fstat, Coef1, Tstat1, …`` becomes ``Fstat, Zstat, Coef1, Tstat1, Zstat1, …``
and the new bricks are tagged as AFNI z-scores (``fizt``). If the bucket already
carries z-score sub-bricks (a prior run), they are recomputed in place from the
preceding stat brick's dof — so re-running with a new total adjustment updates
rather than duplicates.

Where ``new_dof <= 0`` the statistic is undefined; those voxels are clamped to
``dof = 1`` (so nothing crashes) and reported in an ``invalid`` mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# AFNI FUNC stat codes (thd_statpval.c / BRICK_STATAUX).
STAT_TTEST = 3  # fitt: params = (dof,)
STAT_FTEST = 4  # fift: params = (dof_num, dof_den)
STAT_ZSCORE = 5  # fizt: no params

_Z_CLAMP = 13.0  # AFNI qginv cuts off at 13 sigma


def _qginv(p: np.ndarray) -> np.ndarray:
    """Inverse upper-tail normal (AFNI ``qginv``): z with P(Z > z) = p.

    Clamped to ``±13`` like AFNI so saturated tails give a finite z instead of
    inf. Input is clipped to (0, 1).
    """
    from scipy import stats

    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    z = stats.norm.isf(p)
    return np.clip(z, -_Z_CLAMP, _Z_CLAMP)


def t_to_z(t: np.ndarray, dof: np.ndarray | float) -> np.ndarray:
    """Signed t-statistic → z-score at ``dof`` (AFNI ``student_t2z``).

    ``z = sign(t) · qginv(P(|T| > |t|)/2)`` — preserves the one-tailed p and the
    sign. ``dof`` may be a scalar or an array broadcastable to ``t``.
    """
    from scipy import stats

    t = np.asarray(t, dtype=np.float64)
    one_tailed_p = stats.t.sf(np.abs(t), np.asarray(dof, dtype=np.float64))
    return np.sign(t) * _qginv(one_tailed_p)


def f_to_z(f: np.ndarray, dof_num: np.ndarray | float, dof_den: np.ndarray | float) -> np.ndarray:
    """F-statistic → z-score (AFNI ``fstat_t2z``): ``z = qginv(P(F>f)/2)``.

    AFNI treats F as two-sided (halves the tail before inverting). NORDIC only
    reduces the denominator (error) dof; ``dof_num`` is unchanged.
    """
    from scipy import stats

    f = np.asarray(f, dtype=np.float64)
    p = 0.5 * stats.f.sf(
        f, np.asarray(dof_num, dtype=np.float64), np.asarray(dof_den, dtype=np.float64)
    )
    return _qginv(p)


def _z_label(stat_label: str) -> str:
    """Derive a z-score sub-brick label from its parent stat label."""
    for a, b in (
        ("_Tstat", "_Zstat"),
        ("Tstat", "Zstat"),
        ("_Fstat", "_Zstat"),
        ("Fstat", "Zstat"),
    ):
        if a in stat_label:
            return stat_label.replace(a, b, 1)
    return stat_label + "_Zstat"


@dataclass
class DofAdjustResult:
    """Output of :func:`adjust_stats_dof`."""

    data: np.ndarray  # (X, Y, Z, S') with z sub-bricks
    stataux: dict[int, tuple[int, tuple[float, ...]]]
    labels: list[str] | None
    invalid: np.ndarray  # (X, Y, Z) bool: voxels where some new dof <= 0
    updated_in_place: bool  # True if the bucket already had z sub-bricks
    n_stat_bricks: int


def _resolve_adjustment(
    dof_adjust: float | np.ndarray, volume_shape: tuple[int, int, int]
) -> np.ndarray | float:
    """Round the adjustment to integers; broadcast a scalar, validate a map."""
    if np.isscalar(dof_adjust):
        return float(round(float(dof_adjust)))
    arr = np.asarray(dof_adjust)
    if arr.shape[:3] != volume_shape:
        raise ValueError(f"-adjust_dof map shape {arr.shape[:3]} != stats volume {volume_shape}")
    return np.rint(arr.reshape(volume_shape)).astype(np.float64)


def adjust_stats_dof(
    data: np.ndarray,
    stataux: dict[int, tuple[int, tuple[float, ...]]],
    labels: list[str] | None,
    dof_adjust: float | np.ndarray,
    *,
    verbose: bool = True,
) -> DofAdjustResult:
    """Recompute z-scores at a reduced dof and (re)insert them into the bucket.

    Args:
        data: (X, Y, Z, S) statistical bucket.
        stataux: ``{sub_brick_index: (afni_code, params)}`` (from BRICK_STATAUX).
        labels: per-sub-brick labels (or None).
        dof_adjust: scalar dof to subtract everywhere, or a per-voxel (X, Y, Z)
            map (rounded to int).

    Returns:
        :class:`DofAdjustResult`.
    """
    if data.ndim != 4:
        raise ValueError(f"expected 4D bucket (X,Y,Z,S), got shape {data.shape}")
    vol_shape = data.shape[:3]
    n_sub = data.shape[3]
    adj = _resolve_adjustment(dof_adjust, vol_shape)
    invalid = np.zeros(vol_shape, dtype=bool)

    def _z_for(stat_vol: np.ndarray, code: int, params: tuple[float, ...]) -> np.ndarray:
        nonlocal invalid
        if code == STAT_TTEST:
            dof0 = float(params[0])
            new_dof = dof0 - adj
        elif code == STAT_FTEST:
            dof0 = float(params[1])  # denominator (error) dof
            new_dof = dof0 - adj
        else:
            raise ValueError(f"cannot z-convert stat code {code}")
        invalid |= np.asarray(new_dof <= 0) & np.ones(vol_shape, dtype=bool)
        dof_use = np.maximum(new_dof, 1.0)
        if code == STAT_TTEST:
            return t_to_z(stat_vol, dof_use).astype(np.float32)
        return f_to_z(stat_vol, float(params[0]), dof_use).astype(np.float32)

    has_z = any(code == STAT_ZSCORE for code, _ in stataux.values())
    n_stat = sum(1 for code, _ in stataux.values() if code in (STAT_TTEST, STAT_FTEST))

    if has_z:
        # Update mode: overwrite each existing z brick from the stat before it.
        out = data.astype(np.float32, copy=True)
        for i in range(n_sub):
            code, _params = stataux.get(i, (0, ()))
            if code != STAT_ZSCORE:
                continue
            pcode, pparams = stataux.get(i - 1, (0, ()))
            if pcode in (STAT_TTEST, STAT_FTEST):
                out[..., i] = _z_for(data[..., i - 1], pcode, pparams)
        result = DofAdjustResult(
            data=out,
            stataux=dict(stataux),
            labels=list(labels) if labels is not None else None,
            invalid=invalid,
            updated_in_place=True,
            n_stat_bricks=n_stat,
        )
    else:
        # Insert mode: append a z brick after each stat brick.
        planes: list[np.ndarray] = []
        new_labels: list[str] | None = [] if labels is not None else None
        new_stataux: dict[int, tuple[int, tuple[float, ...]]] = {}
        for i in range(n_sub):
            j = len(planes)
            planes.append(data[..., i].astype(np.float32, copy=False))
            if new_labels is not None:
                new_labels.append(labels[i] if i < len(labels) else f"sub{i:02d}")
            if i in stataux:
                new_stataux[j] = stataux[i]
            code, params = stataux.get(i, (0, ()))
            if code in (STAT_TTEST, STAT_FTEST):
                planes.append(_z_for(data[..., i], code, params))
                new_stataux[len(planes) - 1] = (STAT_ZSCORE, ())
                if new_labels is not None:
                    new_labels.append(_z_label(labels[i] if i < len(labels) else f"sub{i:02d}"))
        result = DofAdjustResult(
            data=np.stack(planes, axis=-1),
            stataux=new_stataux,
            labels=new_labels,
            invalid=invalid,
            updated_in_place=False,
            n_stat_bricks=n_stat,
        )

    if verbose:
        mode = "updated in place" if result.updated_in_place else "inserted z-scores"
        n_bad = int(result.invalid.sum())
        adj_desc = f"scalar {adj:g}" if np.isscalar(adj) else "per-voxel map"
        print(
            f"  DoF adjust ({adj_desc}): {n_stat} stat brick(s) → z, {mode}; "
            f"{result.data.shape[3]} sub-bricks out"
        )
        if n_bad:
            print(
                f"  ⚠️  {n_bad:,} voxel(s) hit new dof <= 0 → INVALID statistics "
                f"(clamped to dof=1; see the invalid map)"
            )
    return result


# ---------------------------------------------------------------------------
# File-level operation (shared by ffs_util_updatedof and ffs_reml -adjust_dof)
# ---------------------------------------------------------------------------


def resolve_dof_adjust_arg(arg: str | float, expected_shape=None) -> float | np.ndarray:
    """Interpret an ``-adjust_dof`` argument as either a scalar or a map path.

    A value that parses as a number is a scalar dof to subtract everywhere;
    otherwise it is a path to a 3-D NIfTI of per-voxel dof loss.
    """
    try:
        return float(arg)  # a plain number
    except (TypeError, ValueError):
        pass
    from fastfuncstuff.io.afni import load_nifti

    m = np.asarray(load_nifti(str(arg)).get_fdata(dtype=np.float32))
    if m.ndim == 4 and m.shape[3] == 1:
        m = m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"-adjust_dof map must be 3D, got shape {m.shape}")
    if expected_shape is not None and m.shape != tuple(expected_shape):
        raise ValueError(f"-adjust_dof map shape {m.shape} != stats volume {tuple(expected_shape)}")
    return m


def _default_invalid_path(output_path: str) -> str:
    stem = str(output_path)
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem + "_invalid_dof.nii.gz"


def update_dof_in_file(
    input_path: str,
    dof_adjust: float | np.ndarray,
    output_path: str,
    *,
    invalid_path: str | None = None,
    verbose: bool = True,
) -> DofAdjustResult:
    """Read a stats bucket, adjust its dof, and write the z-augmented result.

    Reuses the AFNI extension readers/writers in :mod:`fastfuncstuff.io.afni`, so
    the output carries updated ``BRICK_STATAUX``/``BRICK_LABS`` that AFNI reads
    natively. Writes an ``*_invalid_dof`` mask when any voxel's new dof <= 0.
    """
    from fastfuncstuff.io.afni import (
        load_nifti,
        read_brick_labels,
        read_brick_stataux,
        save_nifti,
    )

    img = load_nifti(str(input_path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 3:
        data = data[..., None]
    stataux = read_brick_stataux(img)
    if not any(c in (STAT_TTEST, STAT_FTEST, STAT_ZSCORE) for c, _ in stataux.values()):
        raise ValueError(
            f"{input_path}: no AFNI t/F stat metadata (BRICK_STATAUX) found — "
            "cannot adjust dof. Was this bucket written by ffs_reml with AFNI labels?"
        )
    labels = read_brick_labels(img) or None

    result = adjust_stats_dof(data, stataux, labels, dof_adjust, verbose=verbose)

    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else None
    save_nifti(
        result.data,
        output_path,
        affine=img.affine,
        tr=tr,
        brick_labels=result.labels,
        brick_stataux=result.stataux,
    )
    if verbose:
        print(f"  wrote {result.data.shape[3]}-brick stats → {output_path}")
    if result.invalid.any():
        ipath = invalid_path or _default_invalid_path(output_path)
        save_nifti(result.invalid.astype(np.float32), ipath, affine=img.affine)
        if verbose:
            print(f"  wrote invalid-dof mask → {ipath}")
    return result
