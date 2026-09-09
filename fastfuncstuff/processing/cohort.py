"""Labelled cohorts and the pairs a tuning study is run on.

A tuning study asks "what settings suit data of this kind", and the honest
version of that question has three parts this module answers.

**Who is in the cohort.** A directory of subjects, each an image and (optionally)
a manual segmentation on the same grid. Discovery is by convention rather than a
manifest because the manifest is the thing nobody updates.

**Which pairs to fit.** With N subjects there are N(N-1) ordered pairs, and at
tens of seconds a fit the exhaustive set is a day per config. A round-robin over
offsets gives a balanced subset instead -- every subject appears equally often as
a base and as a source -- so a panel of any size is unbiased with respect to who
is in it.

**What was never used to choose.** See :func:`split_subjects`. The split is by
*subject*, never by pair, and pairs never cross it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .tunewarp import SubjectPair

# Extensions the loader understands, longest first so ".nii.gz" wins over ".gz".
_EXTS = (".nii.gz", ".nii.zst", ".nii", ".HEAD")

TRAIN = "train"
TEST = "test"


@dataclass(frozen=True)
class CohortSubject:
    """One brain: an image, and the tracing that judges warps into or out of it."""

    name: str
    image: str
    labels: str | None = None
    split: str = TRAIN


def _strip_ext(name: str) -> str:
    for ext in _EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def discover_cohort(
    root: str | Path,
    pattern: str = "*.nii.gz",
    label_suffix: str = "_seg",
) -> list[CohortSubject]:
    """Find subjects in a directory, pairing each image with its segmentation.

    An image is any file matching ``pattern`` whose stem does *not* end in
    ``label_suffix``; its labels are the file with that suffix inserted, if one
    exists. So ``na01.nii.gz`` + ``na01_seg.nii.gz`` is one subject, and a cohort
    with no segmentations at all is still a cohort -- it simply cannot be judged
    on anatomy.

    Sorted by name, which is what makes the round-robin panels and the split
    reproducible from the directory alone.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"cohort directory does not exist: {root}")

    images = [
        p for p in sorted(root.glob(pattern)) if not _strip_ext(p.name).endswith(label_suffix)
    ]
    if not images:
        raise ValueError(
            f"no images matching {pattern!r} in {root} (after dropping {label_suffix})"
        )

    out: list[CohortSubject] = []
    for img in images:
        stem = _strip_ext(img.name)
        ext = img.name[len(stem) :]
        seg = img.with_name(f"{stem}{label_suffix}{ext}")
        out.append(CohortSubject(stem, str(img), str(seg) if seg.exists() else None))
    return out


def split_subjects(
    subjects: list[CohortSubject], holdout: float = 0.0, seed: int = 0
) -> list[CohortSubject]:
    """Reserve some subjects from the search, so the winner can be checked honestly.

    Splitting by **subject** rather than by pair is the whole content of this
    function. A pair is not an independent observation: pairs A->B and A->C share
    a brain, its geometry, its intensity quirks and whatever the scanner did to it
    that day, so a "held-out" pair that reuses a training subject is measuring a
    setting that was chosen partly on that same brain. Both members of a test pair
    come from the test side, which is why the pair builders below never cross the
    split.

    That costs pairs quadratically -- holding out a quarter of sixteen subjects
    leaves 132 training pairs and 12 test pairs rather than 180 and 60 -- and the
    trade is worth it. Twelve honest pairs answer "does this setting transfer to a
    brain nobody tuned on"; sixty contaminated ones answer nothing.

    Deterministic given ``seed``: adding a subject to a study must not reshuffle
    who was held out, or the earlier fits are no longer train-only.
    """
    if holdout <= 0 or len(subjects) < 4:
        return [CohortSubject(s.name, s.image, s.labels, TRAIN) for s in subjects]

    n = len(subjects)
    fraction = holdout if holdout < 1 else min(holdout / n, 1.0)

    # A THRESHOLD on the subject's own hash, not a cut through a ranking. Both are
    # deterministic, but only this one is stable as the cohort grows: membership
    # depends on the subject and the seed alone, so adding brains leaves everyone
    # already placed where they were. Under a rank cut the newcomers compete for
    # the same slots, and a subject can be moved from train to test after the
    # search has already spent fits on it -- which retroactively contaminates the
    # held-out set, silently, in the direction that flatters the result.
    #
    # The price is that the count is approximately rather than exactly the
    # requested fraction. That is the cheaper thing to give up.
    ranked = sorted(subjects, key=lambda s: _hash_fraction(s.name, seed))
    test = {s.name for s in subjects if _hash_fraction(s.name, seed) < fraction}

    # Both sides need two subjects to make any pair at all. Topping up from the
    # same ranking keeps the choice deterministic and keeps the threshold's
    # stability wherever it was already satisfied.
    for s in ranked:
        if len(test) >= 2:
            break
        test.add(s.name)
    for s in reversed(ranked):
        if n - len(test) >= 2:
            break
        test.discard(s.name)

    return [
        CohortSubject(s.name, s.image, s.labels, TEST if s.name in test else TRAIN)
        for s in subjects
    ]


def _hash_fraction(name: str, seed: int) -> float:
    """A stable pseudo-random number in [0, 1) for one subject under one seed.

    ``hash()`` is salted per interpreter run, so it cannot be used: the split
    would differ between two invocations of the same command.
    """
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def panel_size(n_pool: int, budget: int) -> int:
    """The largest panel whose every pair still informs the search.

    The pairs are the measuring instrument, not the thing being selected, and the
    surrogate compares configs by z-scoring each trial *within its pair* -- which
    is what stops a ranking of settings from becoming a ranking of brains, since
    raw overlap varies more between pairs than between the settings under test.

    A z-score needs a spread, so a pair carrying a single trial has no usable one
    and every trial on it is dropped from the surrogate: recorded in the table,
    invisible to the search. Total trials equal the budget, so a pool larger than
    budget/2 guarantees that a growing share of the fits stop informing anything.
    That is arithmetic rather than taste, which is why it is a default and not a
    constant someone has to know.
    """
    return max(2, min(n_pool, budget // 2))


def pairwise(
    subjects: list[CohortSubject],
    n_pairs: int | None = None,
    split: str | None = None,
    start: int = 0,
) -> list[SubjectPair]:
    """A balanced set of ordered pairs from one side of the cohort.

    Round-robin over offsets: with the subjects in a fixed order, offset k pairs
    each subject i with subject i+k. One offset gives N pairs in which every
    subject appears exactly once as base and once as source; taking offsets
    1, 2, 3... in turn keeps that property at every size. Compared with sampling
    pairs at random, this cannot happen to over-represent the one brain that is
    unlike the others -- which on a cohort of sixteen is the difference between a
    panel and an anecdote.

    ``start`` slides that enumeration, cycling, so successive runs of one study
    can draw overlapping panels rather than the same one forever. A window that
    does not divide evenly by the subject count is balanced to within one
    appearance per subject.

    ``n_pairs=None`` means every ordered pair. Direction matters and both are
    kept: A->B and B->A are different fits with different answers, and their
    disagreement is the raw material for the inverse-consistency check.
    """
    pool = [s for s in subjects if split is None or s.split == split]
    n = len(pool)
    if n < 2:
        raise ValueError(
            f"a pairwise panel needs at least 2 subjects, got {n}"
            + (f" on the {split} side" if split else "")
        )

    total = n * (n - 1)
    wanted = total if n_pairs is None else min(n_pairs, total)
    pairs: list[SubjectPair] = []
    for step in range(wanted):
        k = (start + step) % total
        base, source = pool[k % n], pool[(k % n + 1 + k // n) % n]
        pairs.append(
            SubjectPair(
                name=f"{source.name}_to_{base.name}",
                base=base.image,
                source=source.image,
                base_labels=base.labels,
                source_labels=source.labels,
                split=base.split,
            )
        )
    return pairs


def rotation_start(n_run: int, panel: int) -> int:
    """Where run ``n_run`` (0-based) should begin its round-robin window.

    Half a panel per run, so each run reuses half of the last one and draws half
    fresh. The carried-over pairs pick up a second and third trial, which sharpens
    the within-pair calibration every config is scored against and keeps configs
    from different runs measured partly on the same instrument -- resume pools
    them all into one surrogate, so that matters. The fresh half is what stops a
    study from tuning its parameters to one fixed set of brains, which is what a
    stationary panel does no matter how much budget is spent on it.
    """
    return max(1, panel // 2) * max(0, n_run)


def describe_cohort(subjects: list[CohortSubject], pairs: list[SubjectPair]) -> str:
    """A few lines for the run log: who is in, who is held out, what will be fit."""
    train = [s for s in subjects if s.split == TRAIN]
    test = [s for s in subjects if s.split == TEST]
    labelled = sum(s.labels is not None for s in subjects)
    lines = [
        f"Cohort: {len(subjects)} subject(s), {labelled} with segmentations",
        f"  train ({len(train)}): {_names(train)}",
    ]
    if test:
        lines.append(f"  held out ({len(test)}): {_names(test)}")
        lines.append("  held-out pairs are fit ONLY after the search, on the settings it chose")
    n_train = sum(p.split == TRAIN for p in pairs)
    if pairs:
        lines.append(f"  {n_train} training pair(s) of a possible {len(train) * (len(train) - 1)}")
    return "\n".join(lines)


def _names(subjects: list[CohortSubject], limit: int = 8) -> str:
    names = [s.name for s in subjects]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", ... (+{len(names) - limit})"


def natural_key(name: str) -> tuple:
    """Sort ``na2`` before ``na10``, so a printed cohort reads in subject order."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))
