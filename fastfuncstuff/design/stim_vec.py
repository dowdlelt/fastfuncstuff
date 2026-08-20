"""Continuous TR-locked stimulus vectors (``-stim_event_vec`` / ``-stim_vec``).

A stimulus vector is a regressor supplied as a column of numbers on the TR grid
rather than as a list of onsets: an oscillating background, a motion-energy
trace, a luminance timecourse. Structurally it loads like an ``-ortvec``
(one full-length file, or one file per run, with the same trim handling), but
statistically it is a *stimulus* -- it gets a beta, a t-stat, and it is fit on
the training runs during cross-validation rather than projected out of them.

That distinction is the whole point: an ``-ortvec`` is removed, a stim vector is
modelled. Placing the columns in ``stim_indices`` is all it takes -- every
consumer (``glm/xval.py``, ``glm/reml_xval.py``, the AFNI bucket writers) already
partitions on that list.

Two entry points:

- ``-stim_event_vec LABEL[:mod] VEC ...`` -- a neural-level input, convolved
  here with the design's HRF (see
  :func:`fastfuncstuff.design.matrices.convolve_tr_locked`) and peak-normalised.
- ``-stim_vec LABEL[:mod] VEC ...`` -- already convolved, used verbatim. Not
  rescaled: "pre-convolved" means the caller supplied the exact regressor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Modifiers applied to the raw vector, per run, BEFORE convolution. The
# derivative arms delegate to cli_utils.apply_nuisance_transform so AFNI's
# 1d_tool.py definition is not forked into a second implementation.
STIM_VEC_MODS = ("none", "abs", "deriv", "deriv_back", "deriv_fwd", "deriv_abs")

_DERIV_MODS = {"deriv": "deriv", "deriv_back": "deriv_back", "deriv_fwd": "deriv_fwd"}


def split_label_mod(label: str) -> tuple[str, str]:
    """``"background:abs"`` -> ``("background", "abs")``; bare label -> "none".

    Same suffix convention as the ``-ortvec`` family
    (:func:`fastfuncstuff.cli_utils.split_label_transform`). A modifier welded
    to its own label cannot drift out of sync the way a parallel
    ``-stim_vec_mod`` list ordered by position can.
    """
    if ":" not in label:
        return label, "none"
    base, _, mod = label.rpartition(":")
    if mod not in STIM_VEC_MODS:
        raise ValueError(
            f"unknown modifier {mod!r} in stim vector label {label!r} "
            f"(known: {', '.join(STIM_VEC_MODS)})"
        )
    if not base:
        raise ValueError(f"stim vector label {label!r} has a modifier but no label")
    return base, mod


def apply_stim_vec_mod(arr: np.ndarray, mod: str) -> np.ndarray:
    """Apply one modifier to ONE run's block of a stim vector.

    ``abs`` is for inputs where sign is meaningless -- a background contrast
    that oscillates through zero drives the same neural response either side of
    it, so the rectified curve is the regressor, not the raw one.
    ``deriv_abs`` is the derivative rectified (``|d[t]|``), for "amount of
    change" regardless of direction.

    Per run, not per concatenated timecourse: differencing across a run boundary
    turns the between-run offset into a spike that then eats real signal.
    """
    if mod in (None, "", "none"):
        return np.asarray(arr, dtype=np.float64)
    from fastfuncstuff.cli_utils import apply_nuisance_transform

    a = np.asarray(arr, dtype=np.float64)
    if mod == "abs":
        return np.abs(a)
    if mod in _DERIV_MODS:
        return apply_nuisance_transform(a, _DERIV_MODS[mod])
    if mod == "deriv_abs":
        return np.abs(apply_nuisance_transform(a, "deriv"))
    raise ValueError(f"unknown stim vector modifier {mod!r} (known: {STIM_VEC_MODS})")


@dataclass
class StimVecBlock:
    """One labelled stimulus-vector block, full length, modifier already applied.

    ``values`` is ``(n_timepoints, n_columns)`` spanning every run concatenated
    -- a stim vector is a single shared regressor, never block-diagonal, because
    it describes one stimulus that gets one beta across the whole experiment.
    """

    label: str
    mod: str
    values: np.ndarray
    preconvolved: bool
    sources: list[str] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return int(self.values.shape[1])

    def column_names(self, n_basis: int = 1) -> list[str]:
        """AFNI-style ``LABEL#k`` names, one per emitted design column.

        ``#k`` indexes within the stim class, which is what
        ``write_afni_xmat`` splits on to recover the group label, so a
        multi-column vector (or a multi-basis expansion of one) stays a single
        stim class with a single F-test.
        """
        return [f"{self.label}#{i}" for i in range(self.n_columns * n_basis)]

    def describe(self) -> str:
        kind = "pre-convolved" if self.preconvolved else "to convolve"
        where = ", ".join(self.sources) if self.sources else "?"
        return f"{self.label}: {self.n_columns} column(s), mod={self.mod}, {kind}, from {where}"


def _load_per_run_segments(
    paths: list[str | Path],
    run_starts: list[int],
    n_timepoints: int,
    label: str,
    trim=None,
) -> tuple[list[np.ndarray], list[str]]:
    """Read one full-length file, or exactly one file per run, into run blocks.

    Both spellings must produce the identical concatenated column -- the per-run
    form is a convenience for people whose stimulus is generated alongside each
    run, not a different model.
    """
    from fastfuncstuff.cli_utils import run_lengths_from_starts
    from fastfuncstuff.design.hrf_selection import load_nuisance_file
    from fastfuncstuff.design.trim import trim_run_series

    lengths = run_lengths_from_starts(run_starts, n_timepoints)
    n_runs = len(run_starts)

    if len(paths) == 1:
        arr = load_nuisance_file(paths[0])
        src = [str(paths[0])]
        if arr.shape[0] == n_timepoints:
            offsets = np.cumsum([0] + lengths[:-1])
            return [arr[o : o + n] for o, n in zip(offsets, lengths, strict=True)], src
        if trim is not None and trim.active:
            untrimmed = [n + trim.total for n in lengths]
            if arr.shape[0] == sum(untrimmed):
                blocks, off = [], 0
                for i, n_un in enumerate(untrimmed):
                    blocks.append(
                        trim_run_series(arr[off : off + n_un], lengths[i], trim, paths[0])
                    )
                    off += n_un
                return blocks, src
        raise ValueError(
            f"-stim vector {label!r}: {paths[0]} has {arr.shape[0]} rows, but the design "
            f"has {n_timepoints} timepoints across {n_runs} run(s)"
            + (
                f" (an untrimmed file would have {sum(n + trim.total for n in lengths)} rows)"
                if trim is not None and trim.active
                else ""
            )
        )

    if len(paths) != n_runs:
        raise ValueError(
            f"-stim vector {label!r}: got {len(paths)} files but the design has {n_runs} "
            f"run(s). Pass either one full-length file or exactly one file per run, "
            f"in run order."
        )

    blocks, src = [], []
    for run_idx, (path, expected) in enumerate(zip(paths, lengths, strict=True)):
        arr = load_nuisance_file(path)
        if trim is not None and trim.active:
            arr = trim_run_series(arr, expected, trim, path)
        elif arr.shape[0] != expected:
            raise ValueError(
                f"-stim vector {label!r}: {path} has {arr.shape[0]} rows, but run "
                f"{run_idx + 1} has {expected} timepoints"
            )
        blocks.append(arr)
        src.append(str(path))
    return blocks, src


def load_stim_vec_block(
    label_spec: str,
    paths: list[str | Path],
    run_starts: list[int],
    n_timepoints: int,
    *,
    preconvolved: bool,
    trim=None,
) -> StimVecBlock:
    """Load one ``-stim_event_vec`` / ``-stim_vec`` block and apply its modifier."""
    label, mod = split_label_mod(label_spec)
    if not paths:
        raise ValueError(f"-stim vector {label!r}: no vector files given")
    blocks, sources = _load_per_run_segments(paths, run_starts, n_timepoints, label, trim=trim)

    widths = {b.shape[1] for b in blocks}
    if len(widths) != 1:
        raise ValueError(
            f"-stim vector {label!r}: runs have differing column counts {sorted(widths)}; "
            f"a stim vector is one shared regressor and must be the same width everywhere"
        )

    modded = [apply_stim_vec_mod(b, mod) for b in blocks]
    values = np.vstack(modded).astype(np.float64, copy=False)
    return StimVecBlock(
        label=label,
        mod=mod,
        values=values,
        preconvolved=preconvolved,
        sources=sources,
    )


def collect_stim_vec_blocks(
    args,
    run_starts: list[int],
    n_timepoints: int,
    *,
    trim=None,
    verbose: bool = False,
) -> list[StimVecBlock]:
    """Translate ``args.stim_event_vec`` / ``args.stim_vec`` into blocks.

    Both flags are repeatable and each occurrence is ``[LABEL, VEC, VEC, ...]``.
    Plugs into any CLI that called :func:`add_stim_vec_arguments`.
    """
    blocks: list[StimVecBlock] = []
    seen: set[str] = set()
    for dest, preconvolved in (("stim_event_vec", False), ("stim_vec", True)):
        for entry in getattr(args, dest, None) or []:
            if len(entry) < 2:
                raise ValueError(
                    f"-{dest} needs a LABEL followed by at least one vector file (got {entry!r})"
                )
            block = load_stim_vec_block(
                entry[0],
                list(entry[1:]),
                run_starts,
                n_timepoints,
                preconvolved=preconvolved,
                trim=trim,
            )
            if block.label in seen:
                raise ValueError(
                    f"stim vector label {block.label!r} used more than once; labels name "
                    f"the output sub-bricks and must be unique"
                )
            seen.add(block.label)
            blocks.append(block)
            if verbose:
                print(f"  -{dest}: {block.describe()}")
    return blocks


def build_stim_vec_design(
    blocks: list[StimVecBlock],
    *,
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    hrf_bases=None,
    microtime_onset: int = 0,
    run_starts: list[int] | None = None,
    device=None,
):
    """Assemble the stim-vector design block.

    ``hrf_bases`` is ``(n_basis, n_hrf_bins)`` at ``microtime_dt`` -- the same
    basis set the event columns were built from, so a vector rides the design's
    HRF (including a per-voxel one, when the caller loops HRFs and passes a
    different basis each time). Required only if some block needs convolving.

    Returns ``(design, column_labels, groups)`` where ``groups`` is
    ``[(label, bot, top), ...]`` with indices relative to this block's own
    column 0 -- the caller offsets them by however many task columns precede.
    """
    import torch

    from fastfuncstuff.design.matrices import convolve_tr_locked

    if not blocks:
        empty = torch.zeros((n_timepoints, 0), device=device)
        return empty, [], []

    needs_hrf = any(not b.preconvolved for b in blocks)
    if needs_hrf and hrf_bases is None:
        raise ValueError("build_stim_vec_design: -stim_event_vec needs hrf_bases")
    if hrf_bases is not None:
        hrf_bases = torch.as_tensor(hrf_bases)
        if hrf_bases.ndim == 1:
            hrf_bases = hrf_bases.unsqueeze(0)
    n_basis = int(hrf_bases.shape[0]) if (needs_hrf and hrf_bases is not None) else 1

    columns: list[torch.Tensor] = []
    labels: list[str] = []
    groups: list[tuple[str, int, int]] = []
    offset = 0
    for block in blocks:
        raw = torch.as_tensor(block.values, dtype=torch.get_default_dtype(), device=device)
        if block.preconvolved:
            # Used verbatim: the caller already decided the shape and the scale.
            piece = raw
            block_basis = 1
        else:
            # Interleave column-major within the block (col0 basis0, col0 basis1,
            # ...), matching how the event design orders condition x basis.
            per_basis = [
                convolve_tr_locked(
                    raw,
                    hrf_bases[b],
                    n_timepoints=n_timepoints,
                    tr=tr,
                    microtime_dt=microtime_dt,
                    microtime_onset=microtime_onset,
                    run_starts=run_starts,
                    normalize_peak=True,
                    device=device,
                )
                for b in range(n_basis)
            ]
            piece = torch.cat(
                [per_basis[b][:, c : c + 1] for c in range(raw.shape[1]) for b in range(n_basis)],
                dim=1,
            )
            block_basis = n_basis
        columns.append(piece)
        labels.extend(block.column_names(n_basis=block_basis))
        groups.append((block.label, offset, offset + piece.shape[1] - 1))
        offset += piece.shape[1]

    return torch.cat(columns, dim=1), labels, groups


def add_stim_vec_arguments(parser_or_group) -> None:
    """Register ``-stim_event_vec`` / ``-stim_vec`` on a parser or argument group."""
    mod_note = (
        " LABEL may carry a modifier applied per run BEFORE convolution: "
        "LABEL:abs (rectify -- for inputs where sign is meaningless), "
        "LABEL:deriv / :deriv_back / :deriv_fwd (as 1d_tool.py), "
        "LABEL:deriv_abs (rectified derivative)."
    )
    parser_or_group.add_argument(
        "-stim_event_vec",
        "-stim-event-vec",
        dest="stim_event_vec",
        action="append",
        nargs="+",
        metavar="LABEL VEC",
        help=(
            "Continuous TR-locked stimulus vector, convolved here with the design's HRF. "
            "One full-length file, or one file per run in run order. Gets a beta and a "
            "t-stat like any condition, and is fit (not projected out) during "
            "cross-validation; never split into single trials. Peak-normalised after "
            "convolution. Repeatable: "
            "-stim_event_vec background:abs bg.1D -stim_event_vec luminance lum.1D" + mod_note
        ),
    )
    parser_or_group.add_argument(
        "-stim_vec",
        "-stim-vec",
        dest="stim_vec",
        action="append",
        nargs="+",
        metavar="LABEL VEC",
        help=(
            "As -stim_event_vec, but the vector is ALREADY convolved and is used "
            "verbatim (no HRF, no rescaling). Repeatable." + mod_note
        ),
    )


def resolve_stim_vec_hrf(
    hrf_model_name: str,
    *,
    is_fir_model: bool,
    n_basis: int | None,
    microtime_dt: float,
    device=None,
):
    """The HRF basis set a ``-stim_event_vec`` should ride, given the design's model.

    Returns ``(bases, note)`` with ``bases`` shaped ``(n_basis, n_hrf_bins)``.

    SPMG2/SPMG3 give the vector the same derivative columns the events get, so
    a background whose latency differs slightly from canonical is still
    absorbed. FIR/TENT have no basis analogue for a continuous input -- an FIR
    design models a *response to an onset*, and a vector has no onsets -- so
    those fall back to SPMG1 and say so rather than silently doing something
    surprising.
    """
    from fastfuncstuff.design.hrf import (
        get_hrf_library,
        get_spm_hrf_with_derivatives,
        get_spmg1_hrf,
    )

    name = (hrf_model_name or "SPMG1").upper()
    if is_fir_model:
        hrf = get_spmg1_hrf(
            microtime_dt=microtime_dt, stim_duration=0.0, normalize_peak=True, device=device
        )
        return hrf.unsqueeze(0), (
            f"{name} has no basis analogue for a continuous input; "
            "stim vectors use the SPMG1 canonical HRF"
        )
    if name in ("SPMG2", "SPMG3"):
        n = int(n_basis or (2 if name == "SPMG2" else 3))
        bases = get_spm_hrf_with_derivatives(
            microtime_dt=microtime_dt, hrf_duration=32.0, n_basis=n, device=device
        )
        return bases, f"{name} canonical + {n - 1} derivative(s)"
    if name in ("GLMSINGLE", "SINGLE"):
        hrf = get_hrf_library(
            mode="glmsingle", stim_duration=0.0, microtime_dt=microtime_dt, device=device
        )
        return hrf.reshape(1, -1), "GLMsingle canonical HRF"
    hrf = get_spmg1_hrf(
        microtime_dt=microtime_dt, stim_duration=0.0, normalize_peak=True, device=device
    )
    return hrf.unsqueeze(0), "SPMG1 canonical HRF"


def append_stim_vecs_to_task_design(
    blocks: list[StimVecBlock],
    *,
    task_design=None,
    designs_by_hrf: dict | None = None,
    hrf_library=None,
    hrf_model_name: str = "SPMG1",
    is_fir_model: bool = False,
    n_basis: int | None = None,
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    run_starts: list[int] | None = None,
    microtime_onset: int = 0,
    device=None,
    verbose: bool = True,
):
    """Append stim-vector columns to a task design, single or per-voxel HRF.

    Every CLI that builds a task design does the same three things with stim
    vectors -- resolve the HRF, convolve, concatenate on the right -- so it
    lives here once rather than four times. Callers pass exactly one of
    ``task_design`` or ``designs_by_hrf``, matching what
    :func:`fastfuncstuff.cli_utils.build_task_design_from_args` returned.

    In per-voxel HRF mode each design is extended with the vectors convolved
    against *that voxel group's* HRF: a background is a neural input like any
    other, so it should ride the same haemodynamics the events do.

    Returns ``(task_design, designs_by_hrf, labels, groups)`` with ``groups`` as
    absolute ``(label, bot, top)`` column ranges into the returned task block.
    """
    import torch

    if not blocks:
        return task_design, designs_by_hrf, [], []
    if (task_design is None) == (designs_by_hrf is None):
        raise ValueError("pass exactly one of task_design / designs_by_hrf")

    if task_design is not None:
        reference = task_design
    else:
        assert designs_by_hrf is not None  # guaranteed by the exclusivity check above
        reference = next(iter(designs_by_hrf.values()))
    offset = int(reference.shape[1])

    common = dict(
        n_timepoints=n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        run_starts=run_starts,
        device=device,
    )

    if designs_by_hrf is not None:
        if hrf_library is None:
            raise ValueError("per-voxel HRF mode needs hrf_library to convolve stim vectors")
        labels: list[str] = []
        groups: list[tuple[str, int, int]] = []
        for hrf_idx in list(designs_by_hrf.keys()):
            vec_design, labels, rel_groups = build_stim_vec_design(
                blocks, hrf_bases=hrf_library[int(hrf_idx)].reshape(1, -1), **common
            )
            designs_by_hrf[hrf_idx] = torch.cat([designs_by_hrf[hrf_idx], vec_design], dim=1)
            groups = rel_groups
        if verbose:
            print(f"  Stim vectors convolved with each of {len(designs_by_hrf)} voxel HRFs")
    else:
        bases, note = resolve_stim_vec_hrf(
            hrf_model_name,
            is_fir_model=is_fir_model,
            n_basis=n_basis,
            microtime_dt=microtime_dt,
            device=device,
        )
        vec_design, labels, groups = build_stim_vec_design(blocks, hrf_bases=bases, **common)
        task_design = torch.cat([task_design, vec_design], dim=1)
        if verbose and any(not b.preconvolved for b in blocks):
            print(f"  HRF for -stim_event_vec: {note}")

    groups = [(label, offset + bot, offset + top) for label, bot, top in groups]
    if verbose:
        print(f"  Stim vector columns: {len(labels)}")
    return task_design, designs_by_hrf, labels, groups


def append_stim_vecs_to_single_trial_design(
    blocks: list[StimVecBlock],
    st_design,
    *,
    hrf_library=None,
    hrf_model_name: str = "SPMG1",
    is_fir_model: bool = False,
    n_basis: int | None = None,
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    run_starts: list[int] | None = None,
    microtime_onset: int = 0,
    device=None,
    verbose: bool = True,
):
    """Append stim-vector columns to a single-trial design.

    A stim vector is never split into trials -- it describes a continuous input,
    so there is nothing to split. It rides along as extra always-present task
    columns so the trial betas are estimated with the background already
    accounted for, and it is dropped again before anything trial-indexed
    (``trial_condition_ids``, the trial table, the saved beta series) sees the
    result. Callers must therefore keep the pre-append column count and slice
    the fitted betas back to it.

    Accepts the 2-D ``(n_t, n_trials)`` design and the per-voxel-HRF 3-D
    ``(n_hrfs, n_t, n_trials)`` stack that ``create_single_trial_design``
    returns; in the 3-D case each HRF slab gets the vectors convolved with that
    HRF.

    Returns ``(st_design, labels)``.
    """
    import torch

    if not blocks:
        return st_design, []

    common = dict(
        n_timepoints=n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        run_starts=run_starts,
        device=device,
    )

    if st_design.dim() == 3:
        if hrf_library is None:
            raise ValueError("per-voxel HRF single-trial design needs hrf_library")
        slabs, labels = [], []
        for hrf_idx in range(st_design.shape[0]):
            vec_design, labels, _ = build_stim_vec_design(
                blocks, hrf_bases=hrf_library[hrf_idx].reshape(1, -1), **common
            )
            slabs.append(torch.cat([st_design[hrf_idx], vec_design], dim=1))
        out = torch.stack(slabs, dim=0)
    else:
        bases, note = resolve_stim_vec_hrf(
            hrf_model_name,
            is_fir_model=is_fir_model,
            n_basis=n_basis,
            microtime_dt=microtime_dt,
            device=device,
        )
        vec_design, labels, _ = build_stim_vec_design(blocks, hrf_bases=bases, **common)
        out = torch.cat([st_design, vec_design], dim=1)
        if verbose and any(not b.preconvolved for b in blocks):
            print(f"  HRF for -stim_event_vec: {note}")

    if verbose:
        print(f"  + {len(labels)} stim vector column(s) (not trials)")
    return out, labels


def bucket_labels_from_groups(groups: list[tuple[str, int, int]]) -> list[str]:
    """Bare per-column names for writers that append their own ``#0`` suffix.

    ``column_names`` produces the AFNI ``LABEL#k`` form, which is right for a
    design's ``column_labels`` but doubles up in writers that build
    ``f"{label}#0_Coef"`` themselves (ffs_denoise's bucket). Derived from the
    returned groups so the widths cannot drift from the design that was built.
    """
    out: list[str] = []
    for label, bot, top in groups:
        width = top - bot + 1
        out.extend([label] if width == 1 else [f"{label}_{i}" for i in range(width)])
    return out


def stim_vec_total_columns(blocks: list[StimVecBlock] | None, n_basis: int = 1) -> int:
    """How many design columns a set of blocks will contribute.

    Lets callers size output tensors before the design is built. Pre-convolved
    blocks contribute their raw width; ``-stim_event_vec`` blocks are expanded
    by the HRF basis count.
    """
    if not blocks:
        return 0
    return sum(b.n_columns * (1 if b.preconvolved else n_basis) for b in blocks)


def stim_vec_bucket_labels(blocks: list[StimVecBlock] | None, n_basis: int = 1) -> list[str]:
    """Bare per-column names straight from the blocks (see
    :func:`bucket_labels_from_groups`, which derives the same thing from a
    built design)."""
    if not blocks:
        return []
    out: list[str] = []
    for block in blocks:
        width = block.n_columns * (1 if block.preconvolved else n_basis)
        out.extend([block.label] if width == 1 else [f"{block.label}_{i}" for i in range(width)])
    return out


def residualize_stim_vecs(
    data,
    design,
    vec_design,
    *,
    device=None,
    chunk_size: int | None = None,
):
    """Remove ``span(vec_design)`` from data and design, globally (not per run).

    Stage two of an exact Frisch-Waugh split for the unpenalized case. Given
    data and design that have ALREADY had the per-run nuisance projected out
    (and ``vec_design`` residualized the same way), removing the vectors here
    leaves exactly the residual of a joint regression on
    ``[nuisance | stim vectors]`` -- the sequential form of the same projector.
    Fitting the trials on what comes out therefore yields the trial betas of the
    joint model in which the stim vectors are *unpenalized*, which is the point:
    a background regressor should soak up its variance at full strength rather
    than being shrunk alongside the trials.

    Global, not per run, because a stim vector is one shared regressor with one
    beta across the experiment; projecting it per run would silently give it a
    beta per run.

    Returns ``(data_r, design_r, q, r)`` -- the QR of the vector block, kept so
    the vector betas can be recovered afterwards. Both factors are needed:
    ``q`` alone gives coefficients in the *orthonormalised* basis, which is not
    the vector's own units.
    """
    import torch

    vec = torch.as_tensor(vec_design)
    if device is not None:
        vec = vec.to(device)
    q, r = torch.linalg.qr(vec.to(torch.float32))

    design_t = torch.as_tensor(design).to(q.device)
    design_r = design_t - q @ (q.T @ design_t)

    data_t = torch.as_tensor(data)
    n_voxels = data_t.shape[0]
    step = chunk_size or max(1, n_voxels)
    data_r = torch.empty_like(data_t)
    for c0 in range(0, n_voxels, step):
        c1 = min(c0 + step, n_voxels)
        block = data_t[c0:c1].to(q.device)
        data_r[c0:c1] = (block.T - q @ (q.T @ block.T)).T.to(data_t.device)
    return data_r, design_r, q, r


def recover_stim_vec_betas(data, design, betas, q, r, *, chunk_size: int | None = None):
    """Back-substitute the stim-vector betas after an unpenalized ridge fit.

    ``b_vec = pinv(V)(y - X b_trials)`` -- the remaining half of the
    Frisch-Waugh split. ``q`` is orthonormal, so the pinv is just ``q.T``.

    ``V = QR``, so the coefficient on ``Q`` is ``R c`` and the answer is
    ``c = R^-1 Q'(y - X b)``. Skipping the ``R`` solve returns the coefficient
    in the orthonormalised basis instead -- the right map, in the wrong units.

    ``data`` and ``design`` must be the stage-one (nuisance-projected) pair,
    NOT the stage-two output: ``q`` is orthogonal to the residualized design, so
    passing that instead silently drops the ``b``-dependent term and returns
    ``q.T y`` -- a number that ignores the trials entirely.

    Cheap (a handful of vector columns), and without it the flag would model the
    background without ever reporting it.
    """
    import torch

    n_voxels = data.shape[0]
    step = chunk_size or max(1, n_voxels)
    out = torch.zeros(n_voxels, q.shape[1], device="cpu")
    design_dev = torch.as_tensor(design).to(q.device)
    for c0 in range(0, n_voxels, step):
        c1 = min(c0 + step, n_voxels)
        y = torch.as_tensor(data[c0:c1]).to(q.device)  # (chunk, n_t)
        b = torch.as_tensor(betas[c0:c1]).to(q.device)  # (chunk, n_cols)
        resid = y - b @ design_dev.T
        coef_q = q.T @ resid.T  # (k, chunk), in the orthonormal basis
        out[c0:c1] = torch.linalg.solve_triangular(r, coef_q, upper=True).T.cpu()
    return out
