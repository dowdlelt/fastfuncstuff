"""Turn cortex into contiguous ROIs for BSDS.

BSDS keys on the covariance structure *across* ROIs, so the parcellation you feed
it decides what "functional connectivity" can even mean. The routes here, in
rough order of how well they respect that:

- **atlas** — project an existing label image (Schaefer/Gordon/HCP-MMP) and
  average within each parcel. Contiguous, functionally-derived group boundaries.
  The easy default when you already have aligned ROIs.
- **ward** — data-driven, contiguity-constrained agglomerative clustering on the
  subject's own voxel timeseries. Parcel borders land at *this* subject's FC
  transitions, which is ideal for a densely-sampled individual. Needs
  scikit-learn (an optional/test dependency here).
- **rena** — recursive nearest-neighbour agglomeration; a much faster
  approximation of ward. Delegates to nilearn if installed.
- **voronoi** — random geodesic-ish tiling (k-means on voxel coordinates).
  Contiguous but its borders ignore function, so it dilutes state contrast. Keep
  it as a robustness/baseline null, not the primary choice.

Library functions take arrays; NIfTI loading lives in the CLI. A parcellated
result is a ``(D, N)`` ROI-by-time array plus the integer label of each parcel.
"""

from __future__ import annotations

import warnings

import numpy as np

# Above this many parcels the per-state DxD covariance BSDS estimates gets large
# and the factor-analysis parameter-efficiency argument starts to erode.
_MANY_PARCELS = 400


def _check_parcel_count(d: int) -> None:
    if d > _MANY_PARCELS:
        warnings.warn(
            f"{d} parcels is large for BSDS; the per-state {d}x{d} covariance "
            "may be poorly conditioned. The papers used ~10-30 ROIs; prefer the "
            "tens-to-low-hundreds range.",
            stacklevel=2,
        )


def _inmask(bold4d: np.ndarray, mask3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V, T)`` in-mask timeseries and ``(V, 3)`` voxel coordinates."""
    if bold4d.ndim != 4:
        raise ValueError(f"bold must be 4-D (X, Y, Z, T); got {bold4d.ndim}-D")
    if mask3d.shape != bold4d.shape[:3]:
        raise ValueError(
            f"mask shape {mask3d.shape} does not match bold spatial shape {bold4d.shape[:3]}"
        )
    m = mask3d.astype(bool)
    coords = np.argwhere(m)  # (V, 3)
    ts = bold4d[m]  # (V, T)
    return ts.astype(np.float64, copy=False), coords


def aggregate_by_labels(
    ts_vt: np.ndarray,
    labels_v: np.ndarray,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse ``(V, T)`` voxel timeseries into ``(D, N)`` parcel timeseries.

    ``labels_v`` is a length-``V`` integer array; each unique value ``> 0`` is one
    parcel (0 = unassigned/background). ``aggregate`` is ``"mean"`` or ``"pca"``
    (first principal component, sign-fixed to positive mean loading).

    Returns ``(parcel_ids, ts (D, N))`` with ``parcel_ids`` sorted ascending.
    """
    labels_v = np.asarray(labels_v)
    ids = np.array(sorted(int(u) for u in np.unique(labels_v) if u > 0))
    if ids.size == 0:
        raise ValueError("no positive labels found; nothing to parcellate")
    _check_parcel_count(ids.size)

    n = ts_vt.shape[1]
    out = np.empty((ids.size, n), dtype=np.float64)
    for i, lab in enumerate(ids):
        block = ts_vt[labels_v == lab]  # (v_i, T)
        if aggregate == "mean":
            out[i] = block.mean(axis=0)
        elif aggregate == "pca":
            centered = block - block.mean(axis=1, keepdims=True)
            # First right singular vector = first temporal PC of the parcel.
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            pc = vh[0]
            # Sign is arbitrary; orient so it correlates positively with the mean.
            if np.dot(pc, block.mean(axis=0)) < 0:
                pc = -pc
            out[i] = pc
        else:
            raise ValueError(f"unknown aggregate: {aggregate!r}")
    return ids, out


def parcellate_atlas(
    bold4d: np.ndarray,
    atlas3d: np.ndarray,
    *,
    mask3d: np.ndarray | None = None,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Average voxel timeseries within each label of an atlas image.

    ``atlas3d`` is an integer label volume aligned to ``bold4d``. Labels ``> 0``
    become parcels; a ``mask3d`` (if given) further restricts which voxels count.
    """
    if atlas3d.shape != bold4d.shape[:3]:
        raise ValueError(
            f"atlas shape {atlas3d.shape} does not match bold spatial shape {bold4d.shape[:3]}"
        )
    m = np.ones(bold4d.shape[:3], dtype=bool) if mask3d is None else mask3d.astype(bool)
    m &= atlas3d > 0
    ts, coords = _inmask(bold4d, m)
    labels_v = atlas3d[m].astype(np.int64)
    return aggregate_by_labels(ts, labels_v, aggregate)


def parcellate_voronoi(
    bold4d: np.ndarray,
    mask3d: np.ndarray,
    n_parcels: int,
    *,
    seed: int = 0,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Random contiguous tiling: k-means on in-mask voxel *coordinates*.

    Borders ignore function; this is the robustness/baseline null. Uses
    ``scipy.cluster.vq.kmeans2`` so it needs no optional dependency.
    """
    from scipy.cluster.vq import kmeans2

    ts, coords = _inmask(bold4d, mask3d)
    coords = coords.astype(np.float64)
    rng = np.random.default_rng(seed)
    init = coords[rng.choice(coords.shape[0], size=n_parcels, replace=False)]
    _, labels = kmeans2(coords, init, minit="matrix", seed=seed)
    return aggregate_by_labels(ts, labels + 1, aggregate)  # +1: reserve 0 for bg


def _grid_adjacency(mask3d: np.ndarray):
    """6-connectivity adjacency (scipy sparse COO) over in-mask voxels.

    Rows/cols index in-mask voxels in ``np.argwhere`` order, matching
    :func:`_inmask`, so the matrix can be passed straight to scikit-learn's
    ``AgglomerativeClustering(connectivity=...)``.
    """
    from scipy import sparse

    m = mask3d.astype(bool)
    # Dense index -> compact in-mask index (or -1 outside the mask).
    idx = -np.ones(m.shape, dtype=np.int64)
    coords = np.argwhere(m)
    idx[m] = np.arange(coords.shape[0])

    rows: list[int] = []
    cols: list[int] = []
    for axis in range(3):
        # Voxels adjacent along `axis`: shift the mask by one and intersect.
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        both = m[tuple(sl_a)] & m[tuple(sl_b)]
        a = idx[tuple(sl_a)][both]
        b = idx[tuple(sl_b)][both]
        rows.extend(a.tolist())
        cols.extend(b.tolist())
        rows.extend(b.tolist())
        cols.extend(a.tolist())
    v = coords.shape[0]
    data = np.ones(len(rows), dtype=np.int8)
    return sparse.coo_matrix((data, (rows, cols)), shape=(v, v)).tocsr()


def parcellate_ward(
    bold4d: np.ndarray,
    mask3d: np.ndarray,
    n_parcels: int,
    *,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguity-constrained Ward clustering of voxel timeseries.

    Data-driven, functionally-defined, contiguous parcels — the "local
    correlations to contiguous ROIs" route. Requires scikit-learn.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:  # pragma: no cover - exercised via install state
        raise ImportError(
            "parcellate_ward needs scikit-learn: `pip install scikit-learn` "
            "(or install the `[dynamics]` extra)."
        ) from exc

    ts, _ = _inmask(bold4d, mask3d)  # (V, T)
    adj = _grid_adjacency(mask3d)
    model = AgglomerativeClustering(
        n_clusters=n_parcels,
        connectivity=adj,
        linkage="ward",
    )
    labels = model.fit_predict(ts)  # cluster on each voxel's timeseries
    return aggregate_by_labels(ts, labels + 1, aggregate)


def parcellate_rena(
    bold4d: np.ndarray,
    mask3d: np.ndarray,
    n_parcels: int,
    *,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """ReNA (recursive nearest-neighbour agglomeration); a fast Ward approximation.

    Delegates to nilearn's :class:`~nilearn.regions.ReNA` if available.
    """
    try:
        from nilearn.regions import ReNA
    except ImportError as exc:  # pragma: no cover - exercised via install state
        raise ImportError(
            "parcellate_rena needs nilearn: `pip install nilearn` "
            "(or install the `[dynamics]` extra). For a scikit-learn-only route "
            "use parcellate_ward."
        ) from exc

    import nibabel as nib

    affine = np.eye(4)
    mask_img = nib.Nifti1Image(mask3d.astype(np.int8), affine)
    bold_img = nib.Nifti1Image(np.asarray(bold4d, dtype=np.float32), affine)
    rena = ReNA(mask_img=mask_img, n_clusters=n_parcels)
    rena.fit(bold_img)
    labels_full = np.asarray(rena.labels_)  # per in-mask voxel
    ts, _ = _inmask(bold4d, mask3d)
    return aggregate_by_labels(ts, labels_full + 1, aggregate)
