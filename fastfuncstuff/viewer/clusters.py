"""Clusterize: turn a thresholded map into a list of things you can point at.

A statistic map on screen is a picture; a cluster table is a set of objects with
sizes, peaks and coordinates. The step between them is one connected-component
pass, and everything people actually do with a stat map -- report it, jump to
the biggest blob, seed a correlation from it, ask whether it would have survived
correction -- lives on the far side of it.

Two things here are deliberately not conveniences:

* **The threshold is the layer's own.** Clusterizing at a threshold other than
  the one on screen produces a table that does not describe the picture beside
  it, and the two disagreeing is worse than either alone.
* **Significance comes from the dataset or not at all.** If the bucket carries
  ClustSim tables -- which ``ffs_clustsim`` and ``ffs_reml -clustsim`` attach,
  in AFNI's own format -- each cluster gets the corrected alpha it earned. If
  it does not, the column says so. A cluster-size threshold borrowed from
  somebody else's smoothness is worse than no threshold at all, because it
  looks like a result.

No Qt. The window lists what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.viewer.layers import SignMode

#: How a layer's sign mode is clustered. ``BOTH`` is bi-sided -- the positive
#: and negative tails are labelled separately -- rather than clustering |v|,
#: because a blob that straddles zero is two findings, not one.
SIDEDNESS = {
    SignMode.POS: "1-sided",
    SignMode.NEG: "1-sided",
    SignMode.BOTH: "bi-sided",
}


@dataclass(frozen=True)
class Cluster:
    """One connected suprathreshold blob."""

    index: int
    n_voxels: int
    volume_mm3: float
    #: The extremum by magnitude, reported with its sign -- so a negative
    #: cluster's peak reads as negative rather than as a large positive number.
    peak: float
    peak_ijk: tuple[int, int, int]
    peak_xyz: tuple[float, float, float]
    #: Centre of mass, weighted by |value|, which is what 3dclust reports.
    com_ijk: tuple[float, float, float]
    com_xyz: tuple[float, float, float]
    mean: float
    #: Corrected alpha from an attached ClustSim table, or ``None`` when the
    #: dataset carries none.
    alpha: float | None = None

    @property
    def com_voxel(self) -> tuple[int, int, int]:
        i, j, k = (int(round(c)) for c in self.com_ijk)
        return (i, j, k)


@dataclass(frozen=True)
class ClusterTable:
    """Every cluster in one map at one threshold, biggest first."""

    clusters: tuple[Cluster, ...]
    labels: np.ndarray  # (nx, ny, nz) int32, 0 outside every cluster
    threshold: float
    sidedness: str
    nn: int
    min_voxels: int
    #: Per-voxel p the threshold corresponds to, when the sub-brick says what
    #: test it is. ``None`` for a plain intensity map -- and then no alpha
    #: either, because a ClustSim row is indexed by exactly this.
    pthr: float | None = None
    #: ``(strictest, loosest)`` alpha the simulation covered, when one was
    #: used. An alpha at either end is a bound rather than a measurement, and
    #: this is what lets a report print the "<" instead of a false precision.
    alpha_range: tuple[float, float] | None = None
    #: Why there are no alphas, when there are none. Empty when there are.
    note: str = ""

    def __len__(self) -> int:
        return len(self.clusters)

    def __iter__(self):
        return iter(self.clusters)

    @property
    def n_voxels(self) -> int:
        return int(sum(c.n_voxels for c in self.clusters))

    def find(self, index: int) -> Cluster | None:
        return next((c for c in self.clusters if c.index == index), None)

    def summary(self) -> str:
        text = f"{len(self.clusters)} clusters, {self.n_voxels:,} voxels"
        text += f"   thr {self.threshold:.4g}"
        if self.pthr is not None:
            text += f" (p {self.pthr:.2g})"
        text += f"   NN{self.nn} {self.sidedness}"
        return text


def clusterize(
    values: np.ndarray,
    *,
    stat: np.ndarray | None = None,
    threshold: float,
    sign_mode: SignMode = SignMode.BOTH,
    nn: int = 1,
    min_voxels: int = 1,
    affine: np.ndarray | None = None,
    voxel_mm3: float = 1.0,
    table=None,
    pthr: float | None = None,
) -> ClusterTable:
    """Label a thresholded volume and measure every blob.

    ``stat`` is what gets thresholded and ``values`` is what gets measured, so
    the normal stats case works: colour by the coefficient, cut on its t, and
    report the coefficient's peak inside the blob the t defined. They are the
    same array for a plain intensity map.

    ``table`` is a :class:`~fastfuncstuff.stats.clustsim.ClustSimTable` from the
    dataset's own header; with it, and with a ``pthr`` to index it by, every
    cluster gets a corrected alpha.
    """
    from fastfuncstuff.stats.cluster import cluster_map

    volume = np.nan_to_num(np.asarray(values, dtype=np.float32))
    if volume.ndim != 3:
        raise ValueError(f"clusterizing needs a 3-D volume, got shape {volume.shape}")
    if threshold <= 0:
        raise ValueError("clusterizing needs a threshold above zero")
    cut = volume if stat is None else np.nan_to_num(np.asarray(stat, dtype=np.float32))
    if cut.shape != volume.shape:
        raise ValueError(
            f"the threshold sub-brick is {cut.shape} and the displayed one is {volume.shape}"
        )

    # A negative-only map is the positive case on a flipped volume: one code
    # path for the labelling, and the peaks are still read off the original so
    # their signs are the data's rather than the trick's.
    work = -cut if sign_mode is SignMode.NEG else cut
    sidedness = SIDEDNESS[sign_mode]
    labels, sizes, _masses = cluster_map(work, threshold, sidedness=sidedness, nn=nn)

    keep = [i + 1 for i, n in enumerate(sizes) if int(n) >= max(int(min_voxels), 1)]
    # Renumber so the table's indices are the label map's, biggest first -- a
    # row numbered 3 has to be the voxels drawn as 3, or "make ROIs" and the
    # list stop agreeing.
    keep.sort(key=lambda label: -int(sizes[label - 1]))
    renumber = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for position, label in enumerate(keep):
        renumber[label] = position + 1
    labels = renumber[labels]

    note = ""
    if table is None:
        note = "no ClustSim table in this dataset"
    elif pthr is None:
        note = "the threshold has no p, so no ClustSim row applies"

    clusters = _measure_all(
        volume,
        labels,
        len(keep),
        affine=affine,
        voxel_mm3=voxel_mm3,
        alpha_of=(
            (lambda n: table.alpha_for(pthr, n)) if table is not None and pthr is not None else None
        ),
    )
    return ClusterTable(
        clusters=clusters,
        labels=labels.astype(np.int32),
        threshold=float(threshold),
        sidedness=sidedness,
        nn=int(nn),
        min_voxels=int(min_voxels),
        pthr=pthr,
        alpha_range=None if table is None or pthr is None else table.alpha_range,
        note=note,
    )


def _measure_all(
    volume: np.ndarray,
    labels: np.ndarray,
    n: int,
    *,
    affine: np.ndarray | None,
    voxel_mm3: float,
    alpha_of,
) -> tuple[Cluster, ...]:
    """Measure every cluster in one pass over the suprathreshold voxels.

    One pass, not one per cluster: a mask per label is O(clusters x voxels),
    and a stat map at a loose threshold with no minimum size has tens of
    thousands of one-voxel clusters -- that product was minutes of frozen
    window on a whole brain.
    """
    if n == 0:
        return ()
    flat = labels.reshape(-1)
    where = np.flatnonzero(flat)
    lab = flat[where].astype(np.int64)
    values = volume.reshape(-1)[where].astype(np.float64)
    magnitude = np.abs(values)
    coords = np.stack(np.unravel_index(where, labels.shape), 1).astype(np.float64)

    counts = np.bincount(lab, minlength=n + 1)
    sums = np.bincount(lab, weights=values, minlength=n + 1)
    mass = np.bincount(lab, weights=magnitude, minlength=n + 1)
    com = np.stack(
        [np.bincount(lab, weights=coords[:, a] * magnitude, minlength=n + 1) for a in range(3)], 1
    )
    # Sorting by label, then by descending magnitude, puts each cluster's peak
    # first in its run -- the same argmax the per-cluster version took.
    order = np.lexsort((-magnitude, lab))
    firsts = order[np.searchsorted(lab[order], np.arange(1, n + 1))]

    out = []
    for index in range(1, n + 1):
        count = int(counts[index])
        at = int(firsts[index - 1])
        peak_ijk = tuple(int(v) for v in coords[at])
        centre = tuple(float(v) for v in com[index] / max(float(mass[index]), 1e-12))
        out.append(
            Cluster(
                index=index,
                n_voxels=count,
                volume_mm3=float(count) * float(voxel_mm3),
                peak=float(values[at]),
                peak_ijk=peak_ijk,  # type: ignore[arg-type]
                peak_xyz=_to_mm(peak_ijk, affine),
                com_ijk=centre,  # type: ignore[arg-type]
                com_xyz=_to_mm(centre, affine),
                mean=float(sums[index]) / max(count, 1),
                alpha=None if alpha_of is None else alpha_of(count),
            )
        )
    return tuple(out)


def _to_mm(ijk, affine: np.ndarray | None) -> tuple[float, float, float]:
    if affine is None:
        return (float(ijk[0]), float(ijk[1]), float(ijk[2]))
    x, y, z = (np.asarray(affine, dtype=float) @ np.array([*ijk, 1.0]))[:3]
    return (float(x), float(y), float(z))


def rois_from_clusters(table: ClusterTable, *, name: str = "clusters", source: str = ""):
    """Turn a cluster table into an ROI set, which is what makes it usable.

    This is the whole point of the window: a clusterize result and an atlas are
    the same object, so the moment clusters exist they can seed a correlation,
    name the readout, or become the rows of a matrix -- without a single line
    of code that knows clusters from parcels.
    """
    from fastfuncstuff.io.labels import LabelEntry
    from fastfuncstuff.viewer.rois import rois_from_labels

    names = {c.index: LabelEntry(c.index, f"C{c.index}") for c in table.clusters}
    return rois_from_labels(table.labels, name=name, names=names, source=source)


__all__ = [
    "SIDEDNESS",
    "Cluster",
    "ClusterTable",
    "clusterize",
    "rois_from_clusters",
]
