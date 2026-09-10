"""Run and score nonlinear-registration trials — the engine of `ffs_tunewarp`.

The search drives the backend **libraries in-process**, not their CLIs. Each
image is loaded once and stays resident on the GPU; a trial is a function call
that returns tensors, which are scored in memory and dropped. Nothing a trial
produces reaches the disk.

That is both the space answer and most of the speed answer. A warp field on a
193^3 grid is ~90 MB, so a few hundred fits would be hundreds of gigabytes; and
shelling out would pay process startup, NIfTI compression, and a reload of every
volume, per trial, for data whose entire useful content is one row of a table.

What is recorded instead is the exact equivalent command line. Nothing is kept,
but anything can be rebuilt: ``reproduce()`` re-runs a config through the real
CLI with its outputs kept, so the winner can be looked at.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..io.dsetinfo import read_info
from .affine import load_matrix_1D, save_matrix_1D
from .allineate import _voxdims_from_header
from .io import load_image, save_image
from .mask import automask
from .metrics import METRICS, MetricInputs, cross_subject_dice, evaluate_metrics
from .tuneopt import (
    Observation,
    SearchSpace,
    config_key,
    frontier_hypervolume,
    in_band,
    propose,
    scalarize_balanced,
    score_bands,
)
from .tunespec import (
    BACKENDS,
    QWARP_TUNE_OPTIMIZER,
    Recipe,
    config_in_voxel_units,
    fixed_for,
    qwarp_cost_for,
    render_command,
    resolve_tunable,
)
from .tunestore import BASELINE, COHORT, GRADE_ORDER, TrialStore
from .warpqc import (
    FAIL,
    FAILED_MARGIN,
    UNCONSTRAINED_MARGIN,
    gate_margin,
    pad_mask_to_field,
    regularity_cautions,
    regularity_margin,
    regularity_verdict,
    warp_regularity,
)
from .weight import compute_weight_image


@dataclass
class SubjectPair:
    """One base/source pair to fit. ``name`` is what shows up in the report.

    The label paths are optional and are what turn the run into a scored-on-
    anatomy study: ``source_labels`` is carried through the trial's own field and
    compared against ``base_labels``, which no similarity functional can see.
    """

    name: str
    base: str
    source: str
    base_labels: str | None = None
    source_labels: str | None = None
    # Which side of a held-out split this pair belongs to. Recorded on every trial
    # so the ranking can be built from the training pairs alone and the held-out
    # ones read separately -- a config's score on data that helped choose it is
    # not evidence that it transfers.
    split: str = "train"
    # Path to the affine that put `source` on the base grid, when one was run.
    # Kept so a segmentation can go from its NATIVE grid to the base in a single
    # gather, rather than through the affine and then the field. It is a real
    # AFNI ``.aff12.1D`` (DICOM mm, base->source), so ffs_nwarp / 3dNwarpApply
    # can consume it directly; `native_source` is what turns it back into the
    # base-voxel -> source-voxel matrix the label transport wants.
    source_affine: str | None = None
    # The pre-alignment source, kept because `source` is rewritten to the
    # affine-aligned copy on the base grid once step 0 has run.
    native_source: str | None = None

    @property
    def has_labels(self) -> bool:
        """Both sides traced — what a pairwise comparison against the base needs."""
        return self.base_labels is not None and self.source_labels is not None

    @property
    def has_source_labels(self) -> bool:
        """The moving side traced, which is all a common-space run needs.

        A template has no segmentation of its own and does not need one: the
        cohort's transported tracings are compared against *each other*, not
        against the target.
        """
        return self.source_labels is not None


# --- in-process backend drivers ---------------------------------------------
#
# Each returns (warped_image, (xd, yd, zd)) as tensors, all on the GPU. The
# config dict is keyed by ParamSpec.key; ``config_attr`` maps that onto the
# backend's own dataclass field where the two spellings differ.


def _apply_config(cfg_obj: Any, backend: str, config: dict[str, Any]) -> None:
    spec = BACKENDS[backend]
    for key, value in config.items():
        setattr(cfg_obj, spec.param(key).config_attr, value)


def _run_qwarp(base, source, config, recipe, device):
    from .warp import QwarpConfig, qwarp

    cfg = QwarpConfig(verb=0, optimizer=QWARP_TUNE_OPTIMIZER)
    if recipe is not None:
        cfg.cost_method = qwarp_cost_for(recipe.optimize)
    _apply_config(cfg, "qwarp", config)
    # The pyramid levels are a trajectory: a run at minpatch 5 passes through every
    # coarser patch size on the way. Recording what each level cost and bought means
    # one run answers "was going finer worth it?" for all of them.
    levels: list[dict] = []
    warped, xd, yd, zd = qwarp(base, source, config=cfg, device=device, level_log=levels)
    return warped, (xd, yd, zd), [_QwarpLevel(lv) for lv in levels]


class _QwarpLevel:
    """Adapts a qwarp level record to the ``.as_dict()`` the trial store expects."""

    def __init__(self, rec: dict):
        self._rec = rec

    def as_dict(self) -> dict:
        return dict(self._rec)


def _run_formwarp(base, source, config, recipe, device):
    from .formwarp import SynConfig, formwarp

    cfg = SynConfig(verb=0)
    if recipe is not None:
        cfg.metric = _syn_metric(recipe.optimize)
    _apply_config(cfg, "formwarp", config)
    res = formwarp(base, source, config=cfg, device=device)
    return res.warped, res.fwd, res.levels


def _run_optiwarp(force: str):
    def run(base, source, config, recipe, device):
        from .optiwarp import OptiwarpConfig, optiwarp

        cfg = OptiwarpConfig(verb=0, force=force)
        if recipe is not None:
            cfg.metric = _syn_metric(recipe.optimize)
        _apply_config(cfg, f"optiwarp_{force}", config)
        res = optiwarp(base, source, config=cfg, device=device)
        return res.warped, res.fwd, res.levels

    return run


def _syn_metric(optimize: str) -> str:
    """The SyN/flow tools take a shorter metric list than the allineate costs."""
    return optimize if optimize in ("lpa", "lpc", "pearson", "mse", "cc") else "cc"


DRIVERS = {
    "qwarp": _run_qwarp,
    "formwarp": _run_formwarp,
    "optiwarp_demons": _run_optiwarp("demons"),
    "optiwarp_lk": _run_optiwarp("lk"),
    "optiwarp_hs": _run_optiwarp("hs"),
    "optiwarp_gradient": _run_optiwarp("gradient"),
}


class Referee:
    """Scores a warped image + its field against one base. Caches the base setup.

    The base-side work — weight image, brain mask, voxel dims — is identical for
    every trial against that base, and is the expensive part of scoring, so it is
    built once per base rather than once per trial.
    """

    def __init__(self, base_path: str, device: torch.device, labels_path: str | None = None):
        self.device = device
        self.base, self.header = load_image(base_path, device=device)
        if self.base.ndim == 4:
            # load_image returns 4D as (nt, nz, ny, nx), so the first VOLUME is
            # [0]. `[..., 0]` takes an x-slice and quietly hands the rest of the
            # tool a stack of slices shaped like a volume.
            self.base = self.base[0]
        self.voxdims = _voxdims_from_header(self.header)
        self.weight = compute_weight_image(
            self.base,
            edge_fraction=0.05,
            median_radius=2.25,
            clusterize=True,
            hist_cliplevel=True,
        )
        self.brain = automask(self.base, device=device)
        self.labels: torch.Tensor | None = None
        if labels_path is not None:
            seg, _ = load_image(labels_path, device=device)
            if seg.ndim == 4:
                seg = seg[0]
            if tuple(seg.shape) != tuple(self.base.shape):
                raise ValueError(
                    f"segmentation {labels_path} is {tuple(seg.shape)} but its base is "
                    f"{tuple(self.base.shape)}; labels must be on the base's own grid"
                )
            self.labels = seg.round()
        self._qwarp_padding: tuple[int, int, int, int, int, int] | None = None

    # --- residency ----------------------------------------------------------
    #
    # A pairwise cohort has as many referees as subjects, and each holds three
    # volumes; at 0.7 mm that is ~230 MB apiece, so sixteen of them plus a fit's
    # own working set does not fit on a consumer card. Moving the tensors to the
    # host and back costs a PCIe copy, against rebuilding a weight image and an
    # automask -- which is the expensive part of constructing one of these.

    def offload(self) -> None:
        """Park this referee's volumes on the host, keeping them built."""
        if self.base.device.type == "cpu":
            return
        cpu = torch.device("cpu")
        self.base = self.base.to(cpu)
        self.weight = self.weight.to(cpu)
        self.brain = self.brain.to(cpu)
        if self.labels is not None:
            self.labels = self.labels.to(cpu)

    def attach(self) -> None:
        """Bring them back onto the compute device."""
        if self.base.device == self.device:
            return
        self.base = self.base.to(self.device)
        self.weight = self.weight.to(self.device)
        self.brain = self.brain.to(self.device)
        if self.labels is not None:
            self.labels = self.labels.to(self.device)

    @property
    def qwarp_padding(self) -> tuple[int, int, int, int, int, int]:
        """The padding qwarp derives from this base, computed once per referee.

        Every trial warps the same ``self.base``, so the support box -- and the
        cliplevel histogram behind it -- is a property of the referee, not of the
        trial.
        """
        if self._qwarp_padding is None:
            from .warp import _compute_support_padding

            self._qwarp_padding = _compute_support_padding(self.base)
        return self._qwarp_padding

    def _lower_padding(self, field_shape: tuple[int, ...]) -> tuple[int, int, int] | None:
        """Where the base sits inside a field grid, or None when they agree.

        qwarp estimates on a padded grid and returns the field on it, while the
        SyN/flow backends return one the size of the base. Both go through here so
        that anything resampled by a trial's field -- the regularity mask, a
        segmentation -- lands back on the base's own voxels.
        """
        if tuple(field_shape) == tuple(self.brain.shape):
            return None
        px0, px1, py0, py1, pz0, pz1 = self.qwarp_padding
        expected = (
            self.brain.shape[0] + pz0 + pz1,
            self.brain.shape[1] + py0 + py1,
            self.brain.shape[2] + px0 + px1,
        )
        if tuple(field_shape) != expected:
            raise ValueError(f"qwarp field shape {tuple(field_shape)} != planned {expected}")
        return (px0, py0, pz0)

    def transport_labels_through(
        self, labels: torch.Tensor, field: tuple, affine: torch.Tensor | None
    ) -> torch.Tensor:
        """Carry a NATIVE segmentation to the base grid through affine then field.

        One gather, from the source's own voxels. The field says where each base
        voxel came from in the affine-aligned volume; the affine says where that
        location is in the source's native grid. Composing the two coordinates and
        sampling once is the only way to avoid paying nearest-neighbour boundary
        loss twice -- and that loss is the same order as the differences between
        the configs being ranked, so it is not a rounding detail.

        ``affine is None`` means the pair was already on a common grid (a cohort
        like NIREP), and this reduces to the field alone.
        """
        if affine is None:
            return self.transport_labels(labels, field)

        from .interp import nearest_resample_3d

        xd, yd, zd = field
        lower = self._lower_padding(tuple(xd.shape))
        kk, jj, ii = torch.meshgrid(
            torch.arange(xd.shape[0], dtype=torch.float32, device=xd.device),
            torch.arange(xd.shape[1], dtype=torch.float32, device=xd.device),
            torch.arange(xd.shape[2], dtype=torch.float32, device=xd.device),
            indexing="ij",
        )
        # A padded field is indexed from the padded origin, so undo the shift
        # before handing coordinates to a matrix that speaks base voxels.
        if lower is not None:
            px0, py0, pz0 = lower
            ii, jj, kk = ii - px0, jj - py0, kk - pz0
        x, y, z = ii + xd, jj + yd, kk + zd
        del ii, jj, kk

        m = affine.to(x.device, torch.float32)
        xs = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]
        ys = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]
        zs = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + m[2, 3]
        del x, y, z
        out = nearest_resample_3d(labels, xs, ys, zs).round()
        del xs, ys, zs
        if lower is None:
            return out
        from .warp import _crop_padding

        return _crop_padding(out, self.qwarp_padding, tuple(self.brain.shape))

    def transport_labels(self, labels: torch.Tensor, field: tuple) -> torch.Tensor:
        """Carry a segmentation through a trial's field, nearest-neighbour.

        One gather, from the source's own labels straight onto the base grid. The
        trap this exists to avoid is resampling twice -- NN through an affine and
        NN again through the field loses about a voxel of boundary each time, and
        that noise is the same order as the differences between the configs being
        ranked. Pairs whose affine is baked into the cached aligned volume already
        satisfy this; pairs that are natively aligned (a cohort on a common grid)
        need no affine at all.
        """
        from .interp import warp_image
        from .warp import _crop_padding, _pad_volume_faces

        xd, yd, zd = field
        lower = self._lower_padding(tuple(xd.shape))
        if lower is None:
            return warp_image(labels, xd, yd, zd, mode="nearest").round()
        padding = self.qwarp_padding
        padded = _pad_volume_faces(labels, padding)
        warped = warp_image(padded, xd, yd, zd, mode="nearest")
        del padded
        return _crop_padding(warped, padding, tuple(self.brain.shape)).round()

    def score(
        self,
        warped: torch.Tensor,
        field: tuple | None,
        panel: list[str] | None = None,
        moving_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Score an in-memory result: similarity, then deformation regularity.

        Goes through the metric registry rather than the AFNI cost list alone, so
        the neighbourhood metrics are scored too. The referee holds the volumes,
        which is exactly what those need and what a flattened cost input cannot
        provide.
        """
        inp = MetricInputs(
            base=self.base,
            moving=warped,
            weight=self.weight,
            base_labels=self.labels,
            moving_labels=moving_labels,
            voxdims=self.voxdims,
            overlap=1.0,
        )
        scores = evaluate_metrics(inp, panel)
        del inp

        grade, reasons, cautions, qc = "pass", [], [], {}
        # A field-less result (an affine-only backend) has nothing to fold, so it
        # gets the best margin rather than a missing one — absent evidence of a
        # boundary is not evidence of being on the wrong side of it.
        margin = clearance = UNCONSTRAINED_MARGIN
        if field is not None:
            xd, yd, zd = field
            lower_padding = self._lower_padding(tuple(xd.shape))
            mask = pad_mask_to_field(self.brain, tuple(xd.shape), lower_padding_xyz=lower_padding)
            w = warp_regularity(xd, yd, zd, mask=mask, voxdims=self.voxdims)
            grade, reasons = regularity_verdict(w)
            cautions = regularity_cautions(w)
            margin = regularity_margin(w)
            clearance = gate_margin(w)
            qc = w.as_dict()
        return {
            "scores": scores,
            "grade": grade,
            "reasons": reasons,
            "cautions": cautions,
            "warpqc": qc,
            "margin": margin,
            "gate_margin": clearance,
        }


def _score_group_baseline(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    volumes: CohortVolumes,
) -> None:
    """Cross-subject agreement with the affine alone, as the row to beat.

    The number every config has to improve on. Without it the table says which
    settings won but not whether the nonlinear step bought anything at all, and
    on a template that is exactly the question -- a lot of cross-subject overlap
    is already there from the affine.
    """
    intensity = [n for n in recipe.scored() if not METRICS[n].group]
    segs, per_subject = [], {}
    for pair in pairs:
        referee = volumes.referee(pair)
        source = volumes.source(pair)
        seg = volumes.source_labels(pair)
        if seg is not None:
            # No field, so the labels arrive by the affine alone -- which for a
            # pair already on a common grid means untouched.
            segs.append(_affine_only_labels(referee, seg, volumes.source_affine(pair)))
        for k, v in referee.score(source, None, intensity)["scores"].items():
            per_subject.setdefault(k, []).append(v)

    scores = {k: statistics.fmean(v) for k, v in per_subject.items() if v}
    if len(segs) >= 2:
        summary = cross_subject_dice(segs)
        scores["xdice"] = 1.0 - summary["mean"]
        scores["xdice_q25"] = 1.0 - summary["q25"]
    del segs
    store.add(BASELINE, COHORT, {}, [], seconds=0.0, split=pairs[0].split, scores=scores)


def _affine_only_labels(
    referee: Referee, labels: torch.Tensor, affine: torch.Tensor | None
) -> torch.Tensor:
    """The source segmentation on the base grid with no nonlinear warp."""
    zero = torch.zeros(referee.brain.shape, device=labels.device)
    return referee.transport_labels_through(labels, (zero, zero, zero), affine).to(torch.uint8)


class CohortVolumes:
    """Everything the trials read, loaded once and kept resident a few at a time.

    A one-base study has one referee and a handful of sources, and holding all of
    them on the GPU forever is both simplest and free. A pairwise cohort does not
    work that way: every subject is a base *and* a source, so sixteen brains means
    sixteen referees (base + weight + mask, ~230 MB each at 0.7 mm) plus sixteen
    sources and sixteen segmentations. That is ~5 GB of standing residency against
    a measured 6.1 GB qwarp working set on a 16 GB card, and the fit is what should
    get the memory.

    So: decode from disk exactly once (host side), and let only ``capacity`` pairs'
    worth sit on the compute device, least-recently-used first out. An evicted
    referee is *parked*, not destroyed -- its weight image and automask are the
    expensive part and they survive the trip to the host.
    """

    def __init__(self, device: torch.device, capacity: int = 0):
        self.device = device
        self.capacity = max(1, capacity)
        self._referees: dict[str, Referee] = {}
        self._sources: dict[str, torch.Tensor] = {}
        self._labels: dict[str, torch.Tensor] = {}
        self._affines: dict[str, torch.Tensor] = {}
        self._live_referees: list[str] = []  # LRU order, oldest first
        self._live_volumes: list[tuple[dict, str]] = []

    # --- construction -------------------------------------------------------

    @classmethod
    def open(
        cls, pairs: list[SubjectPair], device: torch.device, capacity: int = 0
    ) -> CohortVolumes:
        """Prepare a cohort, sizing residency from what a fit will need."""
        vols = cls(device, capacity or _residency_capacity(pairs, device))
        return vols

    # --- access -------------------------------------------------------------

    def referee(self, pair: SubjectPair) -> Referee:
        ref = self._referees.get(pair.base)
        if ref is None:
            ref = Referee(pair.base, self.device, pair.base_labels)
            self._referees[pair.base] = ref
        else:
            ref.attach()
        self._touch(self._live_referees, pair.base)
        while len(self._live_referees) > self.capacity:
            self._referees[self._live_referees.pop(0)].offload()
        return ref

    def source(self, pair: SubjectPair) -> torch.Tensor:
        return self._volume(self._sources, pair.source)

    def source_labels(self, pair: SubjectPair) -> torch.Tensor | None:
        if pair.source_labels is None:
            return None
        return self._volume(self._labels, pair.source_labels, integer=True)

    def source_affine(self, pair: SubjectPair) -> torch.Tensor | None:
        """The cached base->source matrix, or None when the pair shares a grid.

        The file on disk is DICOM mm (so AFNI and the rest of ffs can read it);
        the label transport wants base voxels -> source voxels, so it is
        converted back through the two grids' headers on load.
        """
        if pair.source_affine is None or pair.native_source is None:
            return None
        m = self._affines.get(pair.source_affine)
        if m is None:
            m = load_matrix_1D(
                pair.source_affine,
                base_affine=read_info(pair.base).affine,
                source_affine=read_info(pair.native_source).affine,
            ).float()
            self._affines[pair.source_affine] = m
        return m

    def _volume(self, store: dict, path: str, integer: bool = False) -> torch.Tensor:
        vol = store.get(path)
        if vol is None:
            loaded, _ = load_image(path)
            if loaded.ndim == 4:
                loaded = loaded[0]
            vol = loaded.round() if integer else loaded
            store[path] = vol
        if vol.device != self.device:
            vol = vol.to(self.device)
            store[path] = vol
        self._touch(self._live_volumes, (store, path))
        while len(self._live_volumes) > 2 * self.capacity:
            old_store, old_path = self._live_volumes.pop(0)
            old_store[old_path] = old_store[old_path].to("cpu")
        return vol

    @staticmethod
    def _touch(order: list, key: Any) -> None:
        if key in order:
            order.remove(key)
        order.append(key)

    # --- description --------------------------------------------------------

    def describe(self, pairs: list[SubjectPair]) -> dict[str, Any]:
        """The data's own properties, for the run record.

        A preset claims some settings suit data *of a kind*; resolution, matrix and
        how much of the volume is brain are what let the next person decide whether
        their data is that kind.
        """
        ref = self.referee(pairs[0])
        return {
            "subjects": [p.name for p in pairs],
            "base": pairs[0].base,
            "shape": tuple(int(v) for v in ref.base.shape),
            "voxdims": tuple(float(v) for v in ref.voxdims),
            "n_mask_voxels": int(ref.brain.sum()),
        }


def _residency_capacity(pairs: list[SubjectPair], device: torch.device) -> int:
    """How many pairs may stay on the device beside a fit, from the memory model.

    Sized against what a fit actually needs rather than a constant, because the
    same tool runs on a 64^3 phantom and a 0.7 mm cohort. On the CPU there is no
    eviction worth doing -- the host is where an evicted tensor would go anyway --
    so everything stays resident.
    """
    if device.type != "cuda" or not pairs:
        return len(pairs) or 1

    from ..io.headers import nifti_shape
    from ..memory import get_available_memory

    try:
        shape = nifti_shape(pairs[0].base)[:3]
    except (OSError, ValueError, RuntimeError):
        return 2
    nvox = int(np.prod(shape))
    # base + weight + mask + source, plus two segmentations when they are carried.
    per_pair = nvox * 4 * (4 + 2 * any(p.has_labels for p in pairs))
    # The fit is the point; residency gets what a fit's working set does not want.
    # `get_available_memory` already applies the caching allocator's safety factor.
    spare = get_available_memory(device) - _fit_working_set(shape)
    return max(1, min(len(pairs), int(spare // max(per_pair, 1))))


def _fit_working_set(shape: tuple[int, ...]) -> int:
    from ..memory import estimate_nonlinear_memory_bytes

    try:
        return estimate_nonlinear_memory_bytes(tuple(int(v) for v in shape[:3]), "qwarp")
    except (ValueError, RuntimeError):  # pragma: no cover - defensive
        return 0


def affine_align(
    pairs: list[SubjectPair],
    recipe: Recipe,
    out_dir: Path,
    device: torch.device | None = None,
    verb: int = 1,
) -> list[SubjectPair]:
    """Step 0: affine-align every source to its base, and cache the result.

    The nonlinear search assumes the pair already agrees in the affine sense, so
    this has to happen once per subject — but it is not part of the search, and
    nobody wants to hand-run it ten times.

    Unlike trial outputs, these *are* written to disk, for two reasons: they are
    deterministic inputs rather than per-trial noise (so a re-run should skip
    them, not redo ~25 s per subject), and `-reproduce` shells out to the real
    CLI, which needs a path to the aligned volume — a command line pointing at
    the unaligned source would reproduce the wrong thing.
    """
    from .allineate import AffineAlignConfig
    from .allineate import allineate as run_allineate

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = out_dir / "affine"
    cache.mkdir(parents=True, exist_ok=True)

    out: list[SubjectPair] = []
    for pair in pairs:
        stem = pair.name.replace("/", "_")
        dst = cache / f"{stem}.nii.gz"
        mat_path = cache / f"{stem}.aff12.1D"
        if dst.exists() and mat_path.exists():
            if _migrate_legacy_matrix(mat_path, pair) and verb >= 1:
                print(f"  {pair.name}: affine cache converted to AFNI mm", flush=True)
            elif verb >= 1:
                print(f"  {pair.name}: affine cached", flush=True)
            out.append(_affine_pair(pair, dst, mat_path))
            continue

        base, base_header = load_image(pair.base, device=device)
        source, source_header = load_image(pair.source, device=device)
        if base.ndim == 4:
            base = base[0]
        if source.ndim == 4:
            source = source[0]
        cfg = AffineAlignConfig(cost=recipe.optimize, device=str(device), verb=0)
        t0 = time.time()
        matrix, warped = run_allineate(base, source, cfg, base_header, source_header)
        save_image(warped, str(dst), header_info=base_header)
        # The matrix, not just the resampled image. A segmentation must reach the
        # base grid in ONE nearest-neighbour gather -- through the affine and the
        # trial's field composed -- because NN twice loses about a voxel of label
        # boundary each time, and that is the same order as the differences
        # between the configs being ranked. Composing needs the matrix.
        #
        # Written in AFNI's own format rather than the base-voxel matrix
        # allineate returns: a file named .aff12.1D that is neither 12 numbers
        # nor DICOM mm cannot be handed to ffs_nwarp or 3dNwarpApply, and a
        # head-to-head against another tool is exactly what these get used for.
        save_matrix_1D(
            matrix,
            mat_path,
            base_affine=base_header["affine"],
            source_affine=source_header["affine"],
        )
        if verb >= 1:
            print(f"  {pair.name}: affine {time.time() - t0:.1f}s -> {dst.name}", flush=True)
        out.append(_affine_pair(pair, dst, mat_path))
        del base, source, warped
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _migrate_legacy_matrix(mat_path: Path, pair: SubjectPair) -> bool:
    """Rewrite a pre-AFNI-format cache file in place; True if it was converted.

    Early runs wrote allineate's raw base-voxel -> source-voxel 4x4 under the
    ``.aff12.1D`` name. Those caches are still correct, just unusable outside
    tunewarp, and recomputing them costs ~25 s per subject for nothing.
    """
    vals = np.loadtxt(mat_path, ndmin=2)
    if vals.size == 12 and vals.shape[0] <= 3:
        return False
    if vals.shape != (4, 4):
        raise ValueError(f"Unrecognised affine cache {mat_path} with shape {vals.shape}")
    save_matrix_1D(
        torch.from_numpy(vals).float(),
        mat_path,
        base_affine=read_info(pair.base).affine,
        source_affine=read_info(pair.source).affine,
    )
    return True


def _affine_pair(pair: SubjectPair, dst: Path, mat_path: Path) -> SubjectPair:
    """The pair rewritten to use the cached aligned image, keeping NATIVE labels.

    The labels deliberately do not follow the image through the affine. They stay
    on the source's own grid and are carried across in one gather at scoring time,
    which is the whole point of saving the matrix.
    """
    aligned = SubjectPair(
        pair.name,
        pair.base,
        str(dst),
        pair.base_labels,
        pair.source_labels,
        pair.split,
    )
    aligned.source_affine = str(mat_path)
    aligned.native_source = pair.source
    return aligned


def enumerate_configs(
    recipe: Recipe,
    backend: str,
    max_configs: int | None = None,
    fixed: dict[str, Any] | None = None,
) -> list[dict]:
    """The full factorial over this recipe's tunable knobs for this backend.

    Full, not sampled: the whole point of the per-backend search is that no
    setting is eliminated before it has been tried. Pruning happens *within* a
    backend after the fact, never across backends beforehand.

    ``fixed`` (from ``-fix``) is merged into every config, so a pinned knob still
    shows up in the recorded settings and the reproducible command line.
    """
    pins = fixed_for(fixed or {}, backend)
    params = [p for p in resolve_tunable(recipe, backend) if p.key not in pins]
    if not params:
        return [dict(pins)]
    grids = [[(p.key, v) for v in p.values] for p in params]
    configs = [{**pins, **dict(combo)} for combo in product(*grids)]
    return configs[:max_configs] if max_configs else configs


def run_trial(
    backend: str,
    pair: SubjectPair,
    config: dict[str, Any],
    recipe: Recipe,
    volumes: CohortVolumes,
    store: TrialStore,
) -> None:
    """Fit once in memory, score it, record the numbers, drop the tensors.

    The equivalent command line is recorded even though it was never executed —
    it is what ``reproduce()`` runs, and what a user pastes to get this result
    outside the tool.
    """
    # Only what the recipe asks for is scored, not every metric in the registry.
    # The neighbourhood metrics are far more expensive than the AFNI functionals,
    # and scoring one that is barred from voting AND unread buys nothing.
    prefix = f"{backend}_c{store.config_id(backend, config):04d}.nii.gz"
    referee = volumes.referee(pair)
    source = volumes.source(pair)
    # `config` is in millimetres, which is what the table, the surrogate and any
    # preset speak; the engines count in voxels. The conversion happens here and
    # in the rendered command, and nowhere else.
    engine_config = config_in_voxel_units(backend, config, referee.voxdims)
    cmd = render_command(
        backend, pair.base, pair.source, prefix, config, recipe, voxdims=referee.voxdims
    )

    t0 = time.time()
    try:
        warped, field, levels = DRIVERS[backend](
            referee.base, source, engine_config, recipe, referee.device
        )
        moving_labels = None
        seg = volumes.source_labels(pair)
        if seg is not None and field is not None:
            moving_labels = referee.transport_labels_through(
                seg, field, volumes.source_affine(pair)
            )
        outcome = referee.score(warped, field, recipe.scored(), moving_labels)
        outcome["levels"] = [lv.as_dict() for lv in levels]
        del warped, field, moving_labels
    except (RuntimeError, ValueError) as exc:
        # A backend that blows up on a setting is a fact about that setting, not
        # a reason to abandon the search — record it and move on.
        outcome = {
            "grade": FAIL,
            "reasons": [f"{type(exc).__name__}: {exc}"[:300]],
            "margin": FAILED_MARGIN,
        }
    seconds = time.time() - t0

    store.add(backend, pair.name, config, cmd, seconds=seconds, split=pair.split, **outcome)
    if referee.device.type == "cuda":
        torch.cuda.empty_cache()


def run_group_trial(
    backend: str,
    pairs: list[SubjectPair],
    config: dict[str, Any],
    recipe: Recipe,
    volumes: CohortVolumes,
    store: TrialStore,
    bar=None,
) -> int:
    """Warp the whole cohort into the common space with one config, score the set.

    Cross-subject agreement is a property of the *cohort*, not of any one fit, so
    unlike :func:`run_trial` this cannot record a number until every subject has
    been warped. That makes the search's atom N fits rather than one, and is why
    the screen/confirm structure does not apply here: there is no partial answer
    to screen on. Returns the fits spent.

    One row is written per config rather than one per subject. The per-subject
    facts that still matter -- did any of them fold, how long did they take -- are
    aggregated the way ConfigResult already aggregates across subjects: the grade
    is the WORST grade, because a setting that folds on one brain in ten is not a
    setting that works, and the reason names which brain.

    The transported segmentations are the only thing kept between fits. At sixteen
    subjects on a 1 mm template that is ~150 MB as uint8, against dropping the
    warped images and the fields as usual.
    """
    prefix = f"{backend}_c{store.config_id(backend, config):04d}.nii.gz"
    segs: list[torch.Tensor] = []
    grade, reasons, cautions, qc = "pass", [], [], {}
    margin = clearance = UNCONSTRAINED_MARGIN
    seconds = 0.0
    levels: list[dict] = []
    intensity = [n for n in recipe.scored() if not METRICS[n].group]
    per_subject: dict[str, list[float]] = {}
    worst: list[dict] = []
    cmd: list[str] = []
    spent = 0

    for pair in pairs:
        referee = volumes.referee(pair)
        engine_config = config_in_voxel_units(backend, config, referee.voxdims)
        if not cmd:
            cmd = render_command(
                backend, pair.base, pair.source, prefix, config, recipe, voxdims=referee.voxdims
            )
        t0 = time.time()
        try:
            warped, field, lv = DRIVERS[backend](
                referee.base, volumes.source(pair), engine_config, recipe, referee.device
            )
            seg = volumes.source_labels(pair)
            if seg is not None and field is not None:
                segs.append(
                    referee.transport_labels_through(seg, field, volumes.source_affine(pair)).to(
                        torch.uint8
                    )
                )
            one: dict[str, Any] = referee.score(warped, field, intensity)
            levels = [x.as_dict() for x in lv]
            del warped, field
        except (RuntimeError, ValueError) as exc:
            one = {
                "grade": FAIL,
                "reasons": [f"{pair.name}: {type(exc).__name__}: {exc}"[:300]],
                "margin": FAILED_MARGIN,
                "scores": {},
            }
        seconds += time.time() - t0
        spent += 1
        if bar is not None:
            bar.set_postfix_str(f"{pair.name} {one['grade']}", refresh=False)
            bar.update(1)

        for k, v in one.get("scores", {}).items():
            per_subject.setdefault(k, []).append(v)
        if GRADE_ORDER.get(one["grade"], 3) > GRADE_ORDER.get(grade, 3):
            grade = one["grade"]
            reasons = [f"{pair.name}: {r}" for r in one.get("reasons", [])]
        cautions += [c for c in one.get("cautions", []) if c not in cautions]
        # Keep the regularity record of the subject closest to failing, not of the
        # first one whose GRADE got worse -- when nothing folds, no grade ever gets
        # worse and the qc was silently never recorded at all. That left every
        # group row reporting bend 0 and jacmin 1, which blanks the frontier, the
        # Pareto marking and the roughness half of the surrogate's objective.
        if one.get("margin", UNCONSTRAINED_MARGIN) <= margin:
            qc = dict(one.get("warpqc", {}))
        margin = min(margin, one.get("margin", UNCONSTRAINED_MARGIN))
        clearance = min(clearance, one.get("gate_margin", UNCONSTRAINED_MARGIN))
        worst.append(one.get("warpqc", {}))
        if referee.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = {k: statistics.fmean(v) for k, v in per_subject.items() if v}
    # The two numbers the table ranks roughness on are taken across the WHOLE
    # cohort rather than from one subject: a config is as rough as its roughest
    # brain and squashes as hard as its most compressed one, and those need not be
    # the same subject.
    rough = [w for w in worst if w]
    if rough:
        qc["bending_energy"] = max(float(w.get("bending_energy", 0.0)) for w in rough)
        qc["jac_min"] = min(float(w.get("jac_min", 1.0)) for w in rough)
    if len(segs) >= 2:
        summary = cross_subject_dice(segs)
        scores["xdice"] = 1.0 - summary["mean"]
        scores["xdice_q25"] = 1.0 - summary["q25"]
        qc = {**qc, "n_labels": summary["n_labels"], "n_pairs": summary["n_pairs"]}
    del segs
    if volumes.device.type == "cuda":
        torch.cuda.empty_cache()

    store.add(
        backend,
        COHORT,
        config,
        cmd,
        seconds=seconds,
        split=pairs[0].split,
        scores=scores,
        grade=grade,
        reasons=reasons,
        cautions=cautions,
        warpqc=qc,
        margin=margin,
        gate_margin=clearance,
        levels=levels,
    )
    return spent


def score_baseline(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    volumes: CohortVolumes,
) -> None:
    """Score every subject's *input*, unwarped, as the do-nothing row.

    Without it a run reports which candidate won but not whether any of them beat
    leaving the data alone -- and "nonlinear bought this much on data like this" is
    the statement a recommendation is actually made of. Cheap: one scoring pass per
    subject, no fit.
    """
    if recipe.group:
        _score_group_baseline(pairs, recipe, store, volumes)
        return
    scored = recipe.scored()
    for pair in pairs:
        referee = volumes.referee(pair)
        source = volumes.source(pair)
        # The identity "warp" transports the labels unchanged, so the do-nothing
        # row carries the Dice the affine alone already bought -- which is the
        # number every config has to beat to have been worth running.
        outcome = referee.score(source, None, scored, volumes.source_labels(pair))
        store.add(
            BASELINE,
            pair.name,
            {},
            [],
            seconds=0.0,
            split=pair.split,
            **outcome,
        )


def run_search(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    backends: list[str] | None = None,
    max_configs: int | None = None,
    fixed: dict[str, Any] | None = None,
    device: torch.device | None = None,
    verb: int = 1,
    progress=None,
) -> None:
    """Search every backend over its own full grid, on every subject."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = backends or list(recipe.backends)
    for backend in names:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; have {', '.join(BACKENDS)}")

    # Every image is decoded exactly once for the whole search, and the base-side
    # referee setup (weight image, brain mask) once per distinct base. An
    # MNI-style run shares a single referee across every subject and trial; a
    # pairwise cohort has one per subject and lets the cache decide who stays
    # resident.
    volumes = CohortVolumes.open(pairs, device)

    if store.runs:
        # The data's own properties are only knowable once the images are open, so
        # the run record is completed here rather than at begin_run().
        for k, v in volumes.describe(pairs).items():
            setattr(store.runs[-1], k, v)
    if not any(t.backend == BASELINE for t in store.trials):
        score_baseline(pairs, recipe, store, volumes)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    plan = [(b, enumerate_configs(recipe, b, max_configs, fixed)) for b in names]
    total = sum(len(cfgs) for _, cfgs in plan) * len(pairs)
    # A full search is hundreds of fits over tens of minutes; a bar that shows
    # which backend is up and how many fits remain is the difference between
    # "it is working" and "is it stuck".
    bar = (
        tqdm(total=total, desc="tuning", unit="fit", leave=True, file=sys.stderr)
        if tqdm is not None and verb >= 1 and total > 1
        else None
    )

    for backend, configs in plan:
        if bar is not None:
            bar.set_description(backend)
        elif verb >= 1:
            print(f"\n{backend}: {len(configs)} configs x {len(pairs)} subjects", flush=True)
        for config in configs:
            for pair in pairs:
                run_trial(backend, pair, config, recipe, volumes, store)
                if bar is not None:
                    last = store.trials[-1]
                    bar.set_postfix_str(f"{last.grade}", refresh=False)
                    bar.update(1)
                if progress is not None:
                    progress()
            # Save after every config, not every backend. A slow backend (qwarp
            # is minutes per fit) would otherwise leave the table empty for the
            # whole run, so a search you interrupt tells you nothing.
            store.compute_consensus(recipe.panel())
            store.save()

    if bar is not None:
        bar.close()


@dataclass
class AdaptivePlan:
    """How the adaptive search spends its fits.

    ``budget`` is per backend and counted in *fits*, which is the currency that
    matters — a fit is tens of seconds and everything else here is microseconds.
    """

    budget: int = 60
    screen: int = 1  # subjects a fresh candidate is tried on
    confirm: int = 2  # further subjects a survivor earns
    batch: int = 4  # candidates proposed per surrogate refit
    expand: bool = True  # grow a ladder when the incumbent sits on its end
    seed: int = 0
    # Rounds that may pass without meaningfully enlarging the accuracy/smoothness
    # frontier before the backend is called done. 0 disables the check.
    patience: int = 3
    # What "meaningfully" means: relative growth in the frontier's dominated area
    # over one round. Unlike the fold-guard floor this is a judgement dial and not a
    # derived constant -- there is no measurement that says how much better is worth
    # another ten fits. Measured on a 7T epi2epi study, a converged backend still
    # crept up 0.3-0.6% per four fits, so 2% per round is "still learning something"
    # and below it is decimal places.
    tol: float = 0.02
    # Bands of the score range to fill in deliberately instead of chasing the best
    # setting. 0 is off. See tuneopt.score_bands for what a band is and why the
    # default search cannot reach them.
    explore: int = 0


def panel_scores(trials: list, panel: list[str]) -> dict[int, float]:
    """Per-trial score for the surrogate: the panel mean, z-scored within subject.

    Two things force this rather than the report's consensus rank. The rank is
    *relative to the field*, so it changes underneath the surrogate every time a
    trial is added — an unusable regression target. And the adaptive search
    deliberately measures different configs on different numbers of subjects, so
    the target has to be comparable across subjects; z-scoring within a subject
    removes exactly the between-brain offset that would otherwise dominate, and
    makes a config screened on one brain comparable to one confirmed on three.

    Subjects with fewer than two scored trials are skipped: a z-score needs a
    spread, and inventing one would feed the surrogate a confident zero.
    """
    by_subject: dict[str, list] = {}
    for t in trials:
        if t.scores:
            by_subject.setdefault(t.subject, []).append(t)

    out: dict[int, float] = {}
    for ts in by_subject.values():
        if len(ts) < 2:
            continue
        usable = [c for c in panel if all(c in t.scores for t in ts)]
        totals: dict[int, list[float]] = {t.trial_id: [] for t in ts}
        for cost in usable:
            vals = [t.scores[cost] for t in ts]
            mean = statistics.fmean(vals)
            sd = statistics.pstdev(vals)
            if sd <= 0:
                continue
            for t, v in zip(ts, vals, strict=True):
                totals[t.trial_id].append((v - mean) / sd)
        for tid, zs in totals.items():
            if zs:
                out[tid] = statistics.fmean(zs)
    return out


def _observations(store: TrialStore, backend: str, panel: list[str]) -> list[Observation]:
    """The store's trials for one backend, in the form the optimiser consumes."""
    trials = [t for t in store.trials if t.backend == backend]
    scores = panel_scores(trials, panel)
    return [
        Observation(
            config=t.config,
            score=scores[t.trial_id],
            margin=t.margin,
            roughness=float(t.warpqc.get("bending_energy", 0.0)),
        )
        for t in trials
        if t.trial_id in scores
    ]


def _incumbent(
    observations: list[Observation], band: tuple[float, float] | None = None
) -> dict | None:
    """Best config so far: feasible ones first, then by the balanced frontier trade.

    With ``band`` set the question is instead "the best config *at this level*",
    which is the smoothest feasible one inside the band -- exploring rounds must
    refine around where they are working, not around the global winner.

    Which config this is decides where the ladders grow and subdivide, so ranking
    it on similarity alone does not merely mis-report a winner -- it aims the *next*
    batch. On a 7T epi2epi run the similarity-best config was the roughest one in
    the study, so every expansion pushed the regularization ladders further toward
    zero, and the search spent its budget refining the corner it should have been
    backing out of. Scoring the incumbent at an even similarity/roughness weight
    keeps the ladders centred on the trade rather than on one end of it.
    """
    if band is not None:
        inside = in_band(observations, band)
        # Falling back to the global incumbent would aim the ladders at the corner
        # again, which is the one thing an exploring round is not for. An empty band
        # simply gets no refinement this round; the next batch to land in it will
        # give the ladders something to centre on.
        if not inside:
            return None
        return min(inside, key=lambda c: (c[2], c[1]))[0]
    if not observations:
        return None
    balanced = scalarize_balanced(observations)
    agg: dict[tuple, list[float]] = {}
    feasible: dict[tuple, bool] = {}
    configs: dict[tuple, dict] = {}
    for o, v in zip(observations, balanced, strict=True):
        k = config_key(o.config)
        agg.setdefault(k, []).append(v)
        feasible[k] = feasible.get(k, True) and o.margin > 0
        configs[k] = o.config
    best = min(agg, key=lambda k: (0 if feasible[k] else 1, statistics.fmean(agg[k])))
    return configs[best]


def _fit_bar(total: int, desc: str, verb: int):
    """A per-fit progress bar, or None when there is nothing worth watching.

    A fit on a cohort volume is tens of seconds, and the adaptive loop only
    printed a line once a candidate had been screened -- so a run spent minutes
    at a time showing nothing at all, which is indistinguishable from being
    stuck. The bar is the one thing that separates those two states, and its
    final line doubles as the timing record for the run.
    """
    if verb < 1 or total <= 1:
        return None
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return None
    return tqdm(total=total, desc=desc, unit="fit", leave=True, file=sys.stderr)


def _say(bar, text: str) -> None:
    """Print without tearing the bar apart."""
    if bar is None:
        print(text, flush=True)
    else:
        bar.write(text)


def run_adaptive(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    plan: AdaptivePlan,
    backends: list[str] | None = None,
    fixed: dict[str, Any] | None = None,
    device: torch.device | None = None,
    verb: int = 1,
) -> None:
    """Search by surrogate instead of by grid: propose, screen, confirm, expand.

    The loop, per backend:

    1. Fit a GP to the scores seen so far and a second GP to the regularity
       margins, and propose a batch by expected improvement times probability of
       feasibility.
    2. **Screen** each candidate on one subject. This is where the savings are:
       most candidates are answered by one fit, and a folded setting is answered
       by its first.
    3. **Confirm** only the survivors on further subjects, so the fits that cost
       the most go to the settings that might actually become the default.
    4. **Expand** any ladder the incumbent is sitting on the end of, because an
       optimum at a range edge is a statement about the range, not the optimum.

    Within a batch the screening brains are held fixed so the candidates are
    compared against each other on the same head; between batches they move to
    whichever subjects this backend has the least evidence about.

    **Resuming is the normal case, not a special one.** A tuning study has a long
    life: run it early to get a direction, add subjects later to sharpen it. So a
    second invocation against the same directory picks up where the first left off
    rather than starting over --

    * the ladders are rebuilt from the configs already in the store, so the
      subdivisions the earlier run paid fits for are not re-derived;
    * screening goes to the subjects with the least evidence, which puts a newly
      added brain first without being told it is new;
    * ladder values that folded every time they were tried are dropped;
    * and the backend stops once several rounds pass without a new point on the
      frontier, so a large ``-budget`` is a ceiling rather than a promise.

    All of which assumes the *engines* have not changed underneath the stored
    trials. `TrialStore.warnings` is what catches that; it is a warning rather
    than a refusal because only the operator can judge whether a given commit
    moved the numbers.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = backends or list(recipe.backends)
    for backend in names:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; have {', '.join(BACKENDS)}")

    panel = recipe.panel()
    volumes = CohortVolumes.open(pairs, device)

    if store.runs:
        # The data's own properties are only knowable once the images are open, so
        # the run record is completed here rather than at begin_run().
        for k, v in volumes.describe(pairs).items():
            setattr(store.runs[-1], k, v)
    if not any(t.backend == BASELINE for t in store.trials):
        score_baseline(pairs, recipe, store, volumes)

    rng = np.random.default_rng(plan.seed)
    for backend in names:
        space = SearchSpace.from_params(
            resolve_tunable(recipe, backend), fixed_for(fixed or {}, backend)
        )
        if verb >= 1:
            print(f"\n{backend}: budget {plan.budget} fits over {len(space.axes)} knob(s)")

        # Everything the store already knows about this backend, folded into the
        # ladders before the first proposal. See SearchSpace.seed_from.
        prior = _observations(store, backend, panel)
        if prior:
            seeded = space.seed_from(prior)
            pruned = space.prune_infeasible(prior)
            if verb >= 1:
                print(f"  resuming from {len(prior)} earlier fit(s)")
                if seeded:
                    print(f"  ladders restored: {', '.join(seeded)}")
                if pruned:
                    print(f"  dropped (folded every time): {', '.join(pruned)}")

        spent = 0
        round_no = 0
        stale = 0
        best_hv = frontier_hypervolume(prior)
        # A group recipe spends N fits per config, so a budget that is not a
        # multiple of N cannot be hit exactly; round the bar up to what will
        # actually be spent rather than showing a total the loop will overshoot.
        atom = len(pairs) if recipe.group else 1
        planned = -(-plan.budget // atom) * atom
        bar = _fit_bar(planned, backend, verb)
        while spent < plan.budget:
            obs = _observations(store, backend, panel)
            # Bands are recomputed every round rather than fixed at the start: they
            # are quantiles of what has been measured, so a round that finds a new
            # best moves the whole ladder with it instead of filling a room whose
            # floor has since dropped.
            bands = score_bands(obs, plan.explore) if plan.explore else []
            band = bands[round_no % len(bands)] if bands else None
            if band is not None and verb >= 1:
                held = len(in_band(obs, band))
                k = round_no % len(bands)
                _say(
                    bar,
                    f"  exploring band {k + 1}/{len(bands)} "
                    f"(score {band[0]:+.3f}..{band[1]:+.3f}, {held} config(s) there)",
                )
            batch = propose(space, obs, plan.batch, rng, band=band)
            if not batch:
                # The lattice is exhausted. Growing it is the only way forward,
                # and if nothing can grow the space is genuinely finished.
                grown = space.grow_toward(_incumbent(obs) or {}) if plan.expand else []
                if not grown:
                    if verb >= 1:
                        _say(bar, "  space exhausted")
                    break
                if verb >= 1:
                    _say(bar, f"  expanded: {', '.join(grown)}")
                continue

            # Within a round the screening brains are held fixed, so the batch's
            # candidates are compared against each other on the same head.
            screen_pairs = _screen_pairs(pairs, store, backend, plan.screen, round_no)
            round_no += 1

            for config in batch:
                if spent >= plan.budget:
                    break
                label = " ".join(f"{k}={v}" for k, v in sorted(config.items()))

                # A group recipe has no partial answer to screen on: cross-subject
                # agreement only exists once the whole cohort is in the common
                # space. So the atom is N fits, and screen/confirm do not apply.
                if recipe.group:
                    spent += run_group_trial(backend, pairs, config, recipe, volumes, store, bar)
                    if verb >= 1:
                        _say(
                            bar,
                            f"  [{spent:>3}/{planned}] cohort {store.trials[-1].grade:8s} {label}",
                        )
                    continue

                for pair in screen_pairs:
                    run_trial(backend, pair, config, recipe, volumes, store)
                    spent += 1
                    if bar is not None:
                        bar.set_postfix_str(f"screen {store.trials[-1].grade}", refresh=False)
                        bar.update(1)
                last = store.trials[-1]
                if verb >= 1:
                    _say(bar, f"  [{spent:>3}/{plan.budget}] screen {last.grade:8s} {label}")

                if not _promising(store, backend, panel, config):
                    continue
                rest = [p for p in pairs if p not in screen_pairs]
                rng.shuffle(rest)  # type: ignore[arg-type]
                for pair in rest[: plan.confirm]:
                    if spent >= plan.budget:
                        break
                    run_trial(backend, pair, config, recipe, volumes, store)
                    spent += 1
                    if bar is not None:
                        bar.set_postfix_str(f"confirm {store.trials[-1].grade}", refresh=False)
                        bar.update(1)
                if verb >= 1:
                    _say(
                        bar,
                        f"  [{spent:>3}/{plan.budget}] confirmed on {min(plan.confirm, len(rest))}",
                    )

            if plan.expand:
                incumbent = _incumbent(_observations(store, backend, panel), band) or {}
                grown = space.grow_toward(incumbent)
                if grown and verb >= 1:
                    _say(bar, f"  expanded: {', '.join(grown)}")
                # Extend where the incumbent is pinned against an end, subdivide
                # where it sits between two rungs. Without the second the search can
                # reach a listed value but nothing between two of them, and the gaps
                # here are large: measured 130-290x the run-to-run noise.
                refined = space.refine_around(incumbent)
                if refined and verb >= 1:
                    _say(bar, f"  refined: {', '.join(refined)}")

            store.compute_consensus(panel)
            store.save()

            # A round that put nothing new on the frontier did not change the answer;
            # it measured the inside of a trade-off that is already mapped. Several in
            # a row is the search telling you it is done, and spending the rest of the
            # budget past that point buys decimal places nobody reads.
            if plan.patience > 0:
                hv = frontier_hypervolume(_observations(store, backend, panel))
                grew = hv > best_hv * (1.0 + plan.tol)
                stale = 0 if grew else stale + 1
                best_hv = max(best_hv, hv)
                if stale >= plan.patience:
                    if verb >= 1:
                        _say(
                            bar,
                            f"  converged: {stale} rounds without growing the frontier "
                            f"by {plan.tol:.0%} ({spent}/{plan.budget} fits used)",
                        )
                    break

        if bar is not None:
            bar.close()
        store.compute_consensus(panel)
        store.save()


def evaluate_holdout(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    n_configs: int = 5,
    device: torch.device | None = None,
    verb: int = 1,
) -> list[int]:
    """Fit the settings the search chose on the subjects it never saw.

    This is the only honest number a tuning run produces. Everything else in the
    table is in-sample: the surrogate proposed those configs *because* of how they
    scored on those brains, the ladders grew toward them, and the winner is by
    construction the config that best fits this particular draw of subjects. Run
    it again on brains that took no part in any of that and the score moves --
    usually down, and how far down is the thing a reader of the preset needs.

    Only the finalists are re-fit, because the point is not another ranking. A
    held-out set used to *choose* between many candidates stops being held out;
    it can answer "does the chosen setting transfer", once, for a few candidates.
    Passing rows first, in the training table's own order, so the config a user
    would actually take is always among them.

    Returns the config ids that were evaluated.
    """
    if not pairs:
        return []
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ranked = [r for r in store.results(split="train") if not r.is_baseline]
    # The frontier rows are the real choices on the accuracy/smoothness trade, so
    # they earn a slot even when a rougher config outranks them.
    chosen: list[Any] = [r for r in ranked if r.grade == "pass"][:n_configs]
    for r in ranked:
        if r.pareto and r not in chosen and len(chosen) < n_configs + 2:
            chosen.append(r)
    if not chosen:
        if verb >= 1:
            print("  no passing config to check on the held-out subjects")
        return []

    volumes = CohortVolumes.open(pairs, device)
    done = {(t.config_id, t.subject) for t in store.trials if t.split == "test"}
    baselined = {t.subject for t in store.trials if t.backend == BASELINE and t.split == "test"}
    if recipe.group:
        if COHORT not in baselined:
            score_baseline(pairs, recipe, store, volumes)
    else:
        fresh = [p for p in pairs if p.name not in baselined]
        if fresh:
            score_baseline(fresh, recipe, store, volumes)

    # Resuming is the normal case for a tuning directory, and a held-out fit is
    # the most expensive kind here -- every finalist against every held-out pair.
    # Repeating one buys nothing: the config, the pair and the engine are all the
    # same, so it would record the number that is already in the table.
    if recipe.group:
        return _holdout_group(pairs, recipe, store, chosen, volumes, verb)
    todo = [(r, p) for r in chosen for p in pairs if (r.config_id, p.name) not in done]
    already = len(chosen) * len(pairs) - len(todo)
    if verb >= 1:
        print(
            f"\nHeld out: {len(chosen)} config(s) x {len(pairs)} pair(s) on "
            f"{len({p.base for p in pairs})} unseen subject(s)"
            + (f" ({already} already recorded)" if already else "")
        )
    bar = _fit_bar(len(todo), "held out", verb)
    seen: set[int] = set()
    for r, pair in todo:
        run_trial(r.backend, pair, r.config, recipe, volumes, store)
        if bar is not None:
            bar.set_postfix_str(f"cfg {r.config_id}", refresh=False)
            bar.update(1)
        if r.config_id not in seen:
            seen.add(r.config_id)
            if verb >= 1:
                _say(bar, f"  [{r.config_id:>3}] {r.backend} {r.label()}")
        store.save()
    if bar is not None:
        bar.close()

    store.compute_consensus(recipe.panel())
    store.save()
    return [r.config_id for r in chosen]


def _holdout_group(
    pairs: list[SubjectPair],
    recipe: Recipe,
    store: TrialStore,
    chosen: list,
    volumes: CohortVolumes,
    verb: int,
) -> list[int]:
    """The finalists re-run on the held-out subjects, scored among themselves.

    The held-out cohort is scored as its own set, never pooled with the training
    one: agreement between a training brain and a held-out brain would be partly
    in-sample, and the number this table exists to give is the one that is not.
    """
    todo = [
        r
        for r in chosen
        if (r.config_id, COHORT)
        not in {(t.config_id, t.subject) for t in store.trials if t.split == "test"}
    ]
    if verb >= 1:
        print(
            f"\nHeld out: {len(todo)} config(s) x {len(pairs)} unseen subject(s)"
            + (f" ({len(chosen) - len(todo)} already recorded)" if len(chosen) > len(todo) else "")
        )
    bar = _fit_bar(len(todo) * len(pairs), "held out", verb)
    for r in todo:
        run_group_trial(r.backend, pairs, r.config, recipe, volumes, store, bar)
        if verb >= 1:
            _say(bar, f"  [{r.config_id:>3}] {r.backend} {r.label()}")
        store.save()
    if bar is not None:
        bar.close()
    store.compute_consensus(recipe.panel())
    store.save()
    return [r.config_id for r in chosen]


def _screen_pairs(
    pairs: list[SubjectPair], store: TrialStore, backend: str, n: int, round_no: int
) -> list[SubjectPair]:
    """Which brains to screen this round: the ones this backend knows least about.

    A plain rotation is right on a fresh run and wrong on every resume. Adding a
    subject to an existing study is the normal way this tool gets used, and under
    a rotation the new brain -- the only one carrying information the store does
    not already have -- waits its turn behind four that have been screened for
    hundreds of fits. Ordering by how little evidence a subject has puts it first
    automatically, and needs no flag to say "this one is new".

    Ties break on the rotation, so a fresh run where every subject has nothing
    behaves exactly as it did before, and a balanced study keeps cycling.
    """
    seen: dict[str, int] = dict.fromkeys((p.name for p in pairs), 0)
    for t in store.trials:
        if t.backend == backend and t.subject in seen:
            seen[t.subject] += 1
    order = sorted(
        range(len(pairs)), key=lambda i: (seen[pairs[i].name], (i - round_no) % len(pairs))
    )
    return [pairs[i] for i in order[:n]]


def _promising(store: TrialStore, backend: str, panel: list[str], config: dict) -> bool:
    """Does a screened candidate earn confirmation fits on more subjects?

    A folded screen is an immediate no — confirming it would only establish more
    precisely how broken it is, and the surrogate already learned what it needed
    from the one fit. Otherwise the bar is the lower half of the feasible
    candidates seen so far, with the first few waved through so the comparison
    has something to be a comparison against.
    """
    key = config_key(config)
    obs = _observations(store, backend, panel)
    mine = [o for o in obs if config_key(o.config) == key]
    if not mine or min(o.margin for o in mine) <= 0:
        return False
    feasible = sorted(o.score for o in obs if o.margin > 0)
    if len(feasible) < 4:
        return True
    return statistics.fmean([o.score for o in mine]) <= feasible[len(feasible) // 2]


# What a diagnostics run scores the warped IMAGES with, on top of the labels.
# One per family rather than the whole registry: the AFNI functionals agree with
# each other at rank correlation >= 0.96 on same-modality data, so fourteen of
# them is one judge counted fourteen times, and each costs a pass over the volume.
DIAGNOSTIC_METRICS = ("lpa", "ls", "mi", "nmi", "lncc", "ngf", "mse")


def resolve_diagnostic_metrics(names: Sequence[str] | None, contrast: str = "same") -> list[str]:
    """The intensity metrics to record beside the label scores.

    ``None`` is the curated default; ``["all"]`` is every metric meaningful for
    this contrast. Label and group metrics are dropped here whatever is asked
    for -- they are scored from the segmentations, not from one warped image
    against the base, and asking for them at this point is a category error
    rather than a request.
    """
    from .metrics import METRICS, panel_for

    if names is None:
        wanted = list(DIAGNOSTIC_METRICS)
    elif len(names) == 1 and names[0] == "all":
        wanted = panel_for(None, contrast, grid=True)
    else:
        wanted = list(names)
    for n in wanted:
        if n not in METRICS:
            raise ValueError(f"unknown metric {n!r}; see ffs_util_cost -help for the list")
    return [n for n in wanted if not METRICS[n].needs_labels and not METRICS[n].group]


def group_diagnostics(
    pairs: list[SubjectPair],
    recipe: Recipe,
    config: dict[str, Any],
    backend: str,
    out_dir: Path,
    device: torch.device | None = None,
    save_subject_labels: bool = False,
    method: str = "ffs",
    metrics: Sequence[str] | None = None,
    verb: int = 1,
) -> list[Path]:
    """Re-fit one config on the cohort and write everything behind its score.

    The table says a setting agrees better; this says *where*, and is what makes
    the claim checkable by looking rather than by trusting a mean. Written:

    ``overlap_prob.nii.gz`` -- 4-D, one frame per label, each voxel the fraction
    of the cohort placing that parcel there. This is the AFNI overlap-probability
    picture: bright in the core of a region and fading at its edge, and the width
    of that fade is the registration's real error bar.

    ``agreement.nii.gz`` and ``consensus_labels.nii.gz`` -- the same stack reduced
    to how much the cohort agrees at each voxel, and to which parcel wins there.

    ``per_label.tsv`` / ``per_pair.tsv`` / ``per_subject.tsv`` -- the numbers, so a
    finding can be read off rather than eyeballed. Per-label carries how many
    subjects *have* each label, which is what separates a parcel the tracers left
    out from one the warp misplaced.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = resolve_diagnostic_metrics(metrics, recipe.contrast)
    volumes = CohortVolumes.open(pairs, device)

    segs: list[torch.Tensor] = []
    names: list[str] = []
    rows: list[dict] = []
    header = None
    bar = _fit_bar(len(pairs), "diagnostics", verb)
    for pair in pairs:
        referee = volumes.referee(pair)
        header = referee.header
        t0 = time.time()
        warped, field, _ = DRIVERS[backend](
            referee.base,
            volumes.source(pair),
            config_in_voxel_units(backend, config, referee.voxdims),
            recipe,
            referee.device,
        )
        one = referee.score(warped, field, metrics)
        seg = volumes.source_labels(pair)
        if seg is not None and field is not None:
            segs.append(
                referee.transport_labels_through(seg, field, volumes.source_affine(pair)).to(
                    torch.uint8
                )
            )
            names.append(pair.name)
        rows.append(
            {
                "subject": pair.name,
                "grade": one["grade"],
                "bend": float(one["warpqc"].get("bending_energy", 0.0)),
                "jacmin": float(one["warpqc"].get("jac_min", 1.0)),
                "seconds": time.time() - t0,
                "metrics": dict(one.get("scores", {})),
            }
        )
        del warped, field
        if bar is not None:
            bar.update(1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if bar is not None:
        bar.close()

    return write_diagnostics(
        out_dir,
        method,
        segs,
        names,
        rows,
        header,
        meta={"backend": backend, "config": config, "recipe": recipe.name},
        save_subject_labels=save_subject_labels,
    )


def write_diagnostics(
    out_dir: Path,
    method: str,
    segs: list[torch.Tensor],
    names: list[str],
    rows: list[dict],
    header: Any = None,
    meta: dict | None = None,
    save_subject_labels: bool = False,
) -> list[Path]:
    """Everything behind one method's cohort agreement, as pictures and tables.

    Split from the fitting on purpose: the scorer must not care whether the
    segmentations arrived from an ffs trial or from AFNI, ANTs or FSL applying
    its own warp. A head-to-head comparison only means anything if every method
    is measured by the *same* instrument, and the cheapest way to guarantee that
    is to have exactly one.

    Tables are plain TSV carrying a ``method`` column and no comment lines, so a
    directory of methods concatenates in pandas without special-casing. What the
    columns mean lives in ``meta.json``, not wedged into the data file.
    """
    from .metrics import cross_subject_detail, label_overlap_stack

    if len(segs) < 2:
        raise ValueError("diagnostics need at least two transported segmentations")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = cross_subject_detail(segs, names)
    labels = detail["labels"]
    written: list[Path] = []

    stack = label_overlap_stack(segs, labels)
    written.append(
        _save(stack, out_dir / "overlap_prob.nii.gz", header, [f"label{k}" for k in labels])
    )
    agree = stack.max(dim=0).values
    written.append(_save(agree, out_dir / "agreement.nii.gz", header))
    # argmax over a stack that is zero everywhere outside any parcel would name
    # label 1 for the whole background, so the winner is masked to where somebody
    # actually drew something.
    winner = torch.tensor(labels, dtype=torch.float32)[stack.argmax(dim=0)]
    written.append(_save(winner * (agree > 0), out_dir / "consensus_labels.nii.gz", header))
    if save_subject_labels:
        for seg, name in zip(segs, names, strict=True):
            written.append(_save(seg.float(), out_dir / f"labels_{name}.nii.gz", header))

    written.append(_write_label_table(out_dir / "per_label.tsv", method, detail, stack))
    written.append(_write_pair_table(out_dir / "per_pair.tsv", method, detail))
    written.append(_write_subject_table(out_dir / "per_subject.tsv", method, rows, detail, names))
    written.append(_write_summary(out_dir / "summary.tsv", method, detail, stack, rows))

    payload = {
        "method": method,
        "n_subjects": len(segs),
        "n_labels": len(labels),
        "labels": labels,
        "subjects": names,
        "written": [w.name for w in written],
        **(meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    written.append(out_dir / "meta.json")
    del segs, stack
    return written


def _save(vol: torch.Tensor, path: Path, header, brick_labels: list[str] | None = None) -> Path:
    save_image(vol.detach().cpu(), str(path), header_info=header, brick_labels=brick_labels)
    return path


def _write_label_table(path: Path, method: str, detail: dict, stack: torch.Tensor) -> Path:
    """Per parcel: how well the cohort agrees on it, and how big it is.

    ``n_present`` sits beside the Dice because the same absence depresses both a
    parcel the tracers left out and one the warp misplaced -- only the count says
    which happened.
    """
    per_pair, vols = detail["per_pair_label"], detail["volumes"]
    head = [
        "method",
        "label",
        "n_present",
        "dice_mean",
        "dice_sd",
        "dice_min",
        "dice_max",
        "vol_mean_vox",
        "vol_cv",
        "vox_full_overlap",
        "vox_half_overlap",
        "peak_overlap",
    ]
    lines = ["\t".join(head)]
    for k, lab in enumerate(detail["labels"]):
        col = per_pair[:, k]
        col = col[~torch.isnan(col)]
        v = vols[:, k]
        v = v[v > 0]
        frame = stack[k]
        lines.append(
            "\t".join(
                [
                    method,
                    str(lab),
                    str(int(detail["present"][k])),
                    _num(col.mean() if col.numel() else None),
                    _num(col.std() if col.numel() > 1 else 0.0),
                    _num(col.min() if col.numel() else None),
                    _num(col.max() if col.numel() else None),
                    _num(v.mean() if v.numel() else 0.0, 0),
                    _num(v.std() / v.mean() if v.numel() > 1 else 0.0, 3),
                    str(int((frame >= 0.999).sum())),
                    str(int((frame >= 0.5).sum())),
                    _num(frame.max(), 3),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_pair_table(path: Path, method: str, detail: dict) -> Path:
    """Every subject pair's agreement, which is what a mean is hiding."""
    per_pair = detail["per_pair_label"]
    lines = ["\t".join(["method", "subject_a", "subject_b", "dice_mean", "dice_q25"])]
    for i, (a, b) in enumerate(detail["pairs"]):
        row = per_pair[i]
        row = row[~torch.isnan(row)]
        if row.numel() == 0:
            continue
        q25 = torch.quantile(row.float(), 0.25)
        lines.append(f"{method}\t{a}\t{b}\t{_num(row.mean())}\t{_num(q25)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_subject_table(
    path: Path, method: str, rows: list[dict], detail: dict, names: list[str]
) -> Path:
    """Each subject's agreement with the rest, beside its own warp quality.

    The column that earns its place is ``dice_vs_others``: a cohort mean hides
    the one brain that is simply different, and that brain is usually the reason
    a setting looks worse than it is.
    """
    per_pair, pairs = detail["per_pair_label"], detail["pairs"]
    mine: dict[str, list[float]] = {n: [] for n in names}
    for i, (a, b) in enumerate(pairs):
        row = per_pair[i]
        row = row[~torch.isnan(row)]
        if row.numel() == 0:
            continue
        mine[a].append(float(row.mean()))
        mine[b].append(float(row.mean()))
    # Whatever intensity functionals were scored become columns, in a stable
    # order, so two methods scored with different -metrics still concatenate --
    # pandas fills the gaps rather than refusing to line them up.
    extra = sorted({k for r in rows for k in r.get("metrics", {})})
    # The regularity detail only exists when a displacement field was available
    # -- ours always, another tool's only with -warp_suffix -- so these columns
    # appear when there is something to put in them and are omitted otherwise.
    reg = [k for k in _REGULARITY_COLUMNS if any(k in r for r in rows)]
    head = ["method", "subject", "dice_vs_others", "grade", "bend", "jacmin"]
    lines = ["\t".join(head + reg + ["seconds"] + extra)]
    for r in rows:
        vals = mine.get(r["subject"], [])
        cells = [
            method,
            r["subject"],
            _num(statistics.fmean(vals) if vals else None),
            str(r.get("grade", "")),
            _num(r.get("bend"), 5),
            _num(r.get("jacmin")),
        ]
        cells += [_num(r.get(k), 5 if k == "jac_neg_frac" else 4) for k in reg]
        cells.append(_num(r.get("seconds"), 1))
        cells += [_num(r.get("metrics", {}).get(k)) for k in extra]
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_summary(
    path: Path, method: str, detail: dict, stack: torch.Tensor, rows: list[dict]
) -> Path:
    """One row for one method -- concatenate a directory of these and you have the plot.

    Deliberately a file rather than a printed line: the whole point of the
    head-to-head is that every method's row is produced by the same code, so the
    comparison cannot be an artefact of who summarised what.
    """
    per_label = detail["per_label"]
    good = per_label[~torch.isnan(per_label)]
    bends = [r["bend"] for r in rows if r.get("bend") is not None]
    jacs = [r["jacmin"] for r in rows if r.get("jacmin") is not None]
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    head = [
        "method",
        "n_subjects",
        "n_labels",
        "n_pairs",
        "dice_mean",
        "dice_q25",
        "dice_median",
        "dice_worst_label",
        "mean_agreement",
        "bend_max",
        "jacmin_min",
        "seconds_total",
    ]
    row = [
        method,
        str(len(detail["volumes"])),
        str(len(detail["labels"])),
        str(len(detail["pairs"])),
        _num(good.mean() if good.numel() else None),
        _num(torch.quantile(good.float(), 0.25) if good.numel() else None),
        _num(torch.quantile(good.float(), 0.5) if good.numel() else None),
        _num(good.min() if good.numel() else None),
        _num(stack.max(dim=0).values[stack.max(dim=0).values > 0].mean()),
        _num(max(bends) if bends else None, 5),
        _num(min(jacs) if jacs else None),
        _num(sum(secs) if secs else None, 1),
    ]
    # The intensity functionals averaged over subjects, so one row per method
    # carries both halves: what we rank on, and what the other tools optimise.
    # Worst case across the cohort: a method is as extreme as its most extreme
    # subject, so compression takes the min and everything else the max.
    for k in [c for c in _REGULARITY_COLUMNS if any(c in r for r in rows)]:
        vals = [r[k] for r in rows if k in r]
        head.append(k + ("_min" if k == "jac_p01" else "_max"))
        row.append(_num(min(vals) if k == "jac_p01" else max(vals), 5))
    for k in sorted({k for r in rows for k in r.get("metrics", {})}):
        vals = [r["metrics"][k] for r in rows if k in r.get("metrics", {})]
        head.append(k)
        row.append(_num(statistics.fmean(vals) if vals else None))
    path.write_text("\t".join(head) + "\n" + "\t".join(row) + "\n")
    return path


# Regularity detail beyond the two headline numbers. jac_p01 is what the gate
# actually tests -- jac_min is one worst voxel out of millions and is only good
# for spotting a warp resting on the solver's own guard. jac_neg_frac is the
# only one that can say a field is WRONG rather than merely extreme, and it
# doubles as the check that an external field was read in the right units.
_REGULARITY_COLUMNS = ("jac_p01", "jac_p99", "jac_neg_frac", "disp_p99_mm")


def _num(value: Any, places: int = 4) -> str:
    """A number for a TSV cell, or an empty cell -- never the string 'nan'.

    pandas reads a blank as NaN and reads 'nan' as NaN too, but a blank cannot be
    mistaken for a label by anything else that opens the file.
    """
    if value is None:
        return ""
    v = float(value)
    if v != v:
        return ""
    return f"{v:.{places}f}"


def diagnose_warped(
    subjects: list,
    out_dir: Path,
    method: str,
    base: str | None = None,
    metrics: Sequence[str] | None = None,
    contrast: str = "same",
    warps: dict[str, str] | None = None,
    warp_units: str = "mm",
    device: torch.device | None = None,
    save_subject_labels: bool = False,
    verb: int = 1,
) -> list[Path]:
    """Score somebody else's result: labels ALREADY in the common space.

    No fitting, no warping, no assumptions about who produced them. Point it at a
    directory of segmentations that AFNI, ANTs, FSL or SPM has already carried
    into the template, name the method, and it writes the same tables and the
    same overlap volume that an ffs config gets.

    That sameness is the entire point. A comparison between tools is only worth
    reading if the instrument is identical on both sides, and every published
    registration comparison has to argue that it was. Here it is not an argument:
    there is one scorer and both sides call it.

    ``base`` turns on the intensity half of the table. Given the template and the
    warped IMAGES beside the labels, every method also gets scored on the
    functionals other tools optimise -- lpa, mutual information, local
    correlation -- against the same base, with the same weight image and the same
    mask. We rank on the labels, but a table that only carried Dice would be
    answering a different question from the one the other tools were tuned for,
    and the comparison is more honest for showing both.

    ``warps`` turns on the regularity half. Given each method's own displacement
    field, every method is graded for folding, compression and bending by the
    same code that grades ours -- which is what makes "how much deformation is
    normal?" an empirical question instead of a constant somebody chose. The
    bounds in :mod:`warpqc` are explicitly documented as thresholds on a
    continuum with real anatomical variation on both sides, and nothing here has
    ever measured where that continuum sits for output the field accepts.

    ``warp_units`` is "mm" for AFNI and ANTs fields, "voxel" for ours. Watch
    ``jac_neg_frac``: a field read under the wrong convention does not fail
    quietly, it reports implausible folding, so that column doubles as the check
    that the field was understood at all.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = resolve_diagnostic_metrics(metrics, contrast)
    referee = Referee(base, device) if base else None
    if referee is not None and verb >= 1:
        print(f"  scoring images against {Path(base).name}: {', '.join(metrics)}")

    segs, names, rows = [], [], []
    for s in subjects:
        if s.labels is None:
            raise ValueError(
                f"{s.name} has no segmentation. -diag_only scores labels that are "
                "already in the common space; every subject needs one."
            )
        seg, header = load_image(s.labels, device=device)
        if seg.ndim == 4:
            seg = seg[0]
        segs.append(seg.round().to(torch.uint8))
        names.append(s.name)
        # No warp was fitted here, so there is no regularity to report. Left empty
        # rather than filled with a neutral-looking number: a blank reads as
        # "not measured", and 1.0 would read as "measured, and perfect".
        row: dict[str, Any] = {
            "subject": s.name,
            "grade": "",
            "bend": None,
            "jacmin": None,
            "seconds": None,
            "metrics": {},
        }
        if referee is not None and s.image:
            img, _ = load_image(s.image, device=device)
            if img.ndim == 4:
                img = img[0]
            if tuple(img.shape) != tuple(referee.base.shape):
                raise ValueError(
                    f"{s.name}: warped image is {tuple(img.shape)} but the base is "
                    f"{tuple(referee.base.shape)}. -diag_only scores what a tool "
                    "already produced, so the images must be on the base's grid."
                )
            # No field to check -- somebody else's warp -- so this is the
            # similarity half only. The regularity columns stay blank.
            row["metrics"] = referee.score(img, None, metrics)["scores"]
            del img
        warp_path = (warps or {}).get(s.name)
        if warp_path:
            row.update(_warp_quality(warp_path, referee, warp_units, device))
        rows.append(row)
        if verb >= 1:
            extra = f" + {Path(warp_path).name}" if warp_path else ""
            print(f"  {s.name}: {Path(s.labels).name}{extra}", flush=True)

    shapes = {tuple(x.shape) for x in segs}
    if len(shapes) > 1:
        raise ValueError(
            f"the segmentations are on {len(shapes)} different grids ({shapes}). "
            "-diag_only compares them voxel to voxel, so they must already share "
            "the common space."
        )
    return write_diagnostics(
        out_dir,
        method,
        segs,
        names,
        rows,
        header,
        meta={
            "mode": "diag_only",
            "base": base,
            "metrics": metrics,
            "inputs": [s.labels for s in subjects],
        },
        save_subject_labels=save_subject_labels,
    )


def _warp_quality(
    path: str, referee: Referee | None, units: str, device: torch.device
) -> dict[str, Any]:
    """Grade somebody else's displacement field with our own regularity code."""
    from .io import load_warp_field

    xd, yd, zd, hdr = load_warp_field(path, device=device)
    voxdims = _voxdims_from_header(hdr)
    if units == "mm":
        xd, yd, zd = xd / voxdims[0], yd / voxdims[1], zd / voxdims[2]
    elif units != "voxel":
        raise ValueError(f"warp_units must be 'mm' or 'voxel', got {units!r}")

    mask = None
    if referee is not None and tuple(xd.shape) == tuple(referee.brain.shape):
        mask = referee.brain
    qc = warp_regularity(xd, yd, zd, mask=mask, voxdims=voxdims)
    grade, _ = regularity_verdict(qc)
    del xd, yd, zd
    return {
        "grade": grade,
        "bend": qc.bending_energy,
        "jacmin": qc.jac_min,
        "jac_p01": qc.jac_p01,
        "jac_p99": qc.jac_p99,
        "jac_neg_frac": qc.jac_neg_frac,
        "disp_p99_mm": qc.disp_p99_mm,
    }


def collect_diagnostics(root: Path) -> list[Path]:
    """Concatenate every method's tables under ``root`` into one file each.

    The head-to-head, as four dataframes. Each table already carries a ``method``
    column, so this is a concatenation and not a join -- nothing has to line up,
    and a method with a different label set or a missing subject simply
    contributes the rows it has.
    """
    written = []
    for name in ("summary", "per_label", "per_pair", "per_subject"):
        parts = sorted(root.glob(f"*/{name}.tsv"))
        if not parts:
            continue
        header, body = None, []
        for part in parts:
            lines = part.read_text().splitlines()
            if not lines:
                continue
            header = header or lines[0]
            body += [ln for ln in lines[1:] if ln.strip()]
        if header is None:
            continue
        dest = root / f"all_{name}.tsv"
        dest.write_text("\n".join([header, *body]) + "\n")
        written.append(dest)
    return written


def reproduce(
    store: TrialStore,
    config_id: int,
    work_dir: Path,
    timeout: float | None = None,
    verb: int = 1,
) -> list[Path]:
    """Re-run a config on **every** subject, keeping the outputs so they can be looked at.

    Every subject, not every recorded trial. Adaptive search screens most candidates
    on one brain, so the config you most want to eyeball -- an interesting row that
    was never confirmed -- is exactly the one whose recorded trials are a single fit.
    ``commands_for`` synthesises the missing ones from the same rendered command.
    """
    out_root = work_dir / "kept" / f"config{config_id:04d}"
    out_root.mkdir(parents=True, exist_ok=True)
    written = []
    for subject, cmd in store.commands_for(config_id):
        cmd = list(cmd)
        # Redirect -prefix into the keep directory, leaving the rest verbatim.
        pi = cmd.index("-prefix")
        target = out_root / f"{subject}_{Path(cmd[pi + 1]).name}"
        cmd[pi + 1] = str(target)
        if verb >= 1:
            print(f"  {subject}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, timeout=timeout, check=False)
        written.append(target)
        written += _keep_inputs(cmd, out_root, subject, verb)
    return written


def _keep_inputs(cmd: list[str], out_root: Path, subject: str, verb: int) -> list[Path]:
    """Drop this trial's base and affine-aligned source beside its output.

    A warped volume on its own answers nothing -- the question is always "closer
    to the target than the input was?", which needs all three open at once. They
    are *materialised* rather than copied because the recorded paths carry sub-brick
    selectors and .zst compression (``..._unwarped.nii.zst[0]``), so a copy would
    hand back a 4D archive to sub-brick by hand. What lands here is the exact
    volume the fit saw.

    Named with the same ``{subject}_`` prefix as the output, which is what keeps
    five subjects' identically-named stage files from colliding in one directory.
    """
    out: list[Path] = []
    for flag, tag in (("-base", "base"), ("-source", "source_lin")):
        try:
            src = cmd[cmd.index(flag) + 1]
        except (ValueError, IndexError):  # pragma: no cover - malformed command
            continue
        dest = out_root / f"{subject}_{tag}.nii.gz"
        out.append(dest)
        if dest.exists():
            continue  # same inputs for every config; write them once
        try:
            vol, hdr = load_image(src)
            if vol.ndim == 4:
                vol = vol[0]
            save_image(vol, dest, hdr)
        except (OSError, ValueError, RuntimeError) as exc:
            if verb >= 1:
                print(f"    (could not keep {flag} {src}: {exc})", flush=True)
            out.pop()
    return out
