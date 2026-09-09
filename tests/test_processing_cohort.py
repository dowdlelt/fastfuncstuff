"""Tests for cohort discovery, subject-level splits and pairwise panels.

The bugs these guard against are all silent ones: a leak that inflates a
held-out score, a split that re-deals itself when a subject is added, a panel
that happens to over-sample one brain.
"""

from __future__ import annotations

import pytest

from fastfuncstuff.processing.cohort import (
    TEST,
    TRAIN,
    CohortSubject,
    discover_cohort,
    pairwise,
    split_subjects,
)


def _cohort(n: int, labelled: bool = True) -> list[CohortSubject]:
    return [
        CohortSubject(
            f"na{i:02d}", f"na{i:02d}.nii.gz", f"na{i:02d}_seg.nii.gz" if labelled else None
        )
        for i in range(1, n + 1)
    ]


class TestDiscovery:
    def _write(self, tmp_path, names, seg=True):
        for n in names:
            (tmp_path / f"{n}.nii.gz").write_bytes(b"x")
            if seg:
                (tmp_path / f"{n}_seg.nii.gz").write_bytes(b"x")
        return tmp_path

    def test_pairs_each_image_with_its_segmentation(self, tmp_path):
        self._write(tmp_path, ["na01", "na02"])
        subs = discover_cohort(tmp_path)
        assert [s.name for s in subs] == ["na01", "na02"]
        assert subs[0].labels is not None and subs[0].labels.endswith("na01_seg.nii.gz")

    def test_segmentations_are_not_mistaken_for_subjects(self, tmp_path):
        self._write(tmp_path, ["na01", "na02"])
        assert len(discover_cohort(tmp_path)) == 2

    def test_a_subject_without_labels_is_still_a_subject(self, tmp_path):
        self._write(tmp_path, ["na01"], seg=True)
        self._write(tmp_path, ["na02"], seg=False)
        subs = discover_cohort(tmp_path)
        assert len(subs) == 2
        assert [s.labels is None for s in subs] == [False, True]

    def test_an_empty_directory_says_so(self, tmp_path):
        with pytest.raises(ValueError, match="no images"):
            discover_cohort(tmp_path)


class TestSplit:
    def test_holdout_reserves_roughly_the_requested_fraction(self):
        """Approximately, not exactly: membership is a threshold on each subject's
        own hash, which is what makes it stable when the cohort grows."""
        out = split_subjects(_cohort(16), 0.25, seed=0)
        n_test = sum(s.split == TEST for s in out)
        assert 2 <= n_test <= 8
        assert sum(s.split == TRAIN for s in out) == 16 - n_test

    def test_the_fraction_is_right_on_average_over_seeds(self):
        held = [
            sum(s.split == TEST for s in split_subjects(_cohort(16), 0.25, seed=k))
            for k in range(40)
        ]
        assert 3.0 <= sum(held) / len(held) <= 5.0

    def test_no_holdout_leaves_everyone_training(self):
        assert all(s.split == TRAIN for s in split_subjects(_cohort(16), 0.0))

    def test_a_grown_cohort_keeps_everyone_on_the_side_they_were_on(self):
        """A re-deal would retroactively contaminate every fit already spent."""
        small = split_subjects(_cohort(12), 0.25, seed=0)
        big = split_subjects(_cohort(16), 0.25, seed=0)
        sides = {s.name: s.split for s in big}
        for s in small:
            assert sides[s.name] == s.split, s.name

    def test_both_sides_always_have_enough_to_pair(self):
        for n in (4, 5, 6):
            out = split_subjects(_cohort(n), 0.9, seed=1)
            assert sum(s.split == TEST for s in out) >= 2
            assert sum(s.split == TRAIN for s in out) >= 2

    def test_an_integer_holdout_is_read_as_a_count(self):
        """3 of 16 is the same request as 0.1875 of them, and lands in the same place."""
        by_count = split_subjects(_cohort(16), 3, seed=0)
        by_fraction = split_subjects(_cohort(16), 3 / 16, seed=0)
        assert [s.split for s in by_count] == [s.split for s in by_fraction]


class TestPairwise:
    def test_pairs_never_cross_the_split(self):
        """The leak this whole design exists to prevent: a 'held-out' pair whose
        other half is a brain the search tuned on."""
        subs = split_subjects(_cohort(16), 0.25, seed=0)
        train = {s.name for s in subs if s.split == TRAIN}
        test = {s.name for s in subs if s.split == TEST}
        for p in pairwise(subs, None, split=TEST):
            src, base = p.name.split("_to_")
            assert src in test and base in test
        for p in pairwise(subs, None, split=TRAIN):
            src, base = p.name.split("_to_")
            assert src in train and base in train

    def test_a_panel_uses_every_subject_equally(self):
        """A panel that over-samples one unusual brain is an anecdote."""
        subs = _cohort(8)
        pairs = pairwise(subs, 8)
        assert len(pairs) == 8
        as_base = [p.base for p in pairs]
        as_source = [p.source for p in pairs]
        assert len(set(as_base)) == 8
        assert len(set(as_source)) == 8

    def test_both_directions_are_kept(self):
        pairs = pairwise(_cohort(4), None)
        names = {p.name for p in pairs}
        assert len(pairs) == 12
        assert "na02_to_na01" in names and "na01_to_na02" in names

    def test_no_subject_is_paired_with_itself(self):
        for p in pairwise(_cohort(6), None):
            assert p.base != p.source

    def test_labels_travel_with_the_pair(self):
        p = pairwise(_cohort(4), 1)[0]
        assert p.has_labels
        assert p.base_labels.endswith("_seg.nii.gz")

    def test_an_unlabelled_cohort_pairs_without_labels(self):
        p = pairwise(_cohort(4, labelled=False), 1)[0]
        assert not p.has_labels

    def test_requesting_more_pairs_than_exist_is_capped(self):
        assert len(pairwise(_cohort(4), 999)) == 12

    def test_one_subject_cannot_make_a_panel(self):
        with pytest.raises(ValueError, match="at least 2"):
            pairwise(_cohort(1), None)
