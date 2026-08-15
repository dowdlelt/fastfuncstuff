"""Tests for tunespec.py / tunestore.py — the ffs_tunewarp search space and bookkeeping."""

import pytest

from fastfuncstuff.processing.tunespec import (
    BACKENDS,
    RECIPES,
    Recipe,
    render_command,
    resolve_tunable,
)
from fastfuncstuff.processing.tunestore import (
    TrialStore,
    format_reproduce,
    format_results_table,
)
from fastfuncstuff.processing.warpqc import FAIL, MARGINAL, PASS


class TestBackendSpecs:
    def test_every_backend_has_a_regularization_knob(self):
        """Regularization is the knob the search leans on; none may lack one."""
        for name, spec in BACKENDS.items():
            roles = {p.role for p in spec.params}
            assert "regularization" in roles, f"{name} exposes no regularization"

    def test_force_models_are_separate_backends(self):
        """They do not share a parameter set, so they must not be screened as one."""
        for force in ("demons", "lk", "hs"):
            assert f"optiwarp_{force}" in BACKENDS
        assert BACKENDS["optiwarp_demons"].keys() != BACKENDS["optiwarp_lk"].keys()

    def test_defaults_are_inside_the_searched_values(self):
        for spec in BACKENDS.values():
            for p in spec.params:
                if p.default is not None:
                    assert p.default in p.values, f"{p.name} default not in its grid"

    def test_param_lookup_rejects_unknown_keys(self):
        with pytest.raises(KeyError):
            BACKENDS["qwarp"].param("not_a_knob")


class TestRendering:
    def test_scalar_flag(self):
        p = BACKENDS["formwarp"].param("total_var")
        assert p.render(1.0) == ["-total_var", "1.0"]

    def test_tuple_flag_expands(self):
        """-workhard takes two values, so it must not render as one token."""
        p = BACKENDS["qwarp"].param("workhard")
        assert p.render((0, -1)) == ["-workhard", "0", "-1"]

    def test_command_includes_fixed_force_arg(self):
        cmd = render_command("optiwarp_lk", "b.nii", "s.nii", "o.nii", {})
        assert "-force" in cmd and "lk" in cmd

    def test_command_is_reproducible_verbatim(self):
        cfg = {"total_var": 1.0, "update_var": 2.0}
        a = render_command("formwarp", "b.nii", "s.nii", "o.nii", cfg)
        b = render_command("formwarp", "b.nii", "s.nii", "o.nii", cfg)
        assert a == b
        assert a[:7] == ["ffs_formwarp", "-base", "b.nii", "-source", "s.nii", "-prefix", "o.nii"]

    def test_recipe_pins_the_metric(self):
        cmd = render_command("qwarp", "b", "s", "o", {}, recipe=RECIPES["MNI_T1"])
        assert "-cost" in cmd and "lpa" in cmd

    def test_save_warp_can_be_suppressed(self):
        """Space discipline: a scored-and-discarded trial need not write a warp."""
        cmd = render_command("formwarp", "b", "s", "o", {}, save_warp=False)
        assert "-save_warp" not in cmd


class TestRecipes:
    def test_every_recipe_excludes_the_metric_it_optimises(self):
        """A cost must never be allowed to vote on the fit it produced.

        Enforced on the assembled panel rather than on a hand-written exclusion
        list, because the list was the weaker check: it let the optimised cost
        keep voting through a sibling computed from the same quantity.
        """
        for r in RECIPES.values():
            assert r.optimize not in r.panel(), r.name

    def test_no_recipe_is_judged_by_a_meaningless_functional(self):
        """Signed costs reward anti-correlation; on same-modality data that
        ranks the worst warp first."""
        from fastfuncstuff.processing.allcost import SIGNED_COSTS

        for r in RECIPES.values():
            if r.contrast == "same":
                assert not set(r.panel()) & set(SIGNED_COSTS), r.name

    def test_every_recipe_declares_a_known_contrast_regime(self):
        for r in RECIPES.values():
            assert r.contrast in ("same", "cross"), r.name
            assert len(r.panel()) >= 3, r.name

    def test_epi2epi_and_mni_differ(self):
        """Same brain vs different brains are different problems."""
        assert RECIPES["epi2epi"].pairing == "paired"
        assert RECIPES["MNI_T1"].pairing == "one_base"

    def test_resolve_tunable_maps_optiwarp_keys_to_all_forces(self):
        r = RECIPES["epi2t1"]
        for force in ("demons", "lk", "hs"):
            keys = [p.key for p in resolve_tunable(r, f"optiwarp_{force}")]
            assert "match" in keys and "total_sigma" in keys

    def test_resolve_tunable_ignores_other_backends_keys(self):
        r = Recipe("t", "", "lpa", ("lpa",), "paired", ("qwarp.penfac",))
        assert [p.key for p in resolve_tunable(r, "qwarp")] == ["penfac"]
        assert resolve_tunable(r, "formwarp") == []


class TestTrialStore:
    def _store(self, tmp_path):
        s = TrialStore(tmp_path / "trials.json")
        for subj in ("s1", "s2"):
            s.add(
                "formwarp",
                subj,
                {"total_var": 0.0},
                ["cmd", "a"],
                scores={"consensus": 0.1},
                grade=FAIL,
                reasons=["folding: ..."],
                seconds=20.0,
            )
            s.add(
                "formwarp",
                subj,
                {"total_var": 1.0},
                ["cmd", "b"],
                scores={"consensus": 0.4},
                grade=PASS,
                seconds=19.0,
            )
        return s

    def test_same_config_across_subjects_shares_an_id(self, tmp_path):
        s = self._store(tmp_path)
        ids = {t.config_id for t in s.trials if t.config["total_var"] == 0.0}
        assert len(ids) == 1

    def test_passing_config_outranks_a_better_scoring_failure(self, tmp_path):
        """The inversion this tool exists to prevent: score never beats a fold."""
        s = self._store(tmp_path)
        top = s.results()[0]
        assert top.grade == PASS
        assert top.config == {"total_var": 1.0}

    def test_worst_subject_sets_the_grade(self, tmp_path):
        s = TrialStore(tmp_path / "t.json")
        s.add("qwarp", "s1", {}, ["c"], scores={"consensus": 0.5}, grade=PASS)
        s.add("qwarp", "s2", {}, ["c"], scores={"consensus": 0.5}, grade=MARGINAL)
        assert s.results()[0].grade == MARGINAL

    def test_spread_is_reported(self, tmp_path):
        s = TrialStore(tmp_path / "t.json")
        s.add("qwarp", "s1", {}, ["c"], scores={"consensus": 0.2}, grade=PASS)
        s.add("qwarp", "s2", {}, ["c"], scores={"consensus": 0.4}, grade=PASS)
        r = s.results()[0]
        assert r.score_mean == pytest.approx(0.3)
        assert r.score_spread > 0

    def test_roundtrip(self, tmp_path):
        s = self._store(tmp_path)
        s.save()
        again = TrialStore(tmp_path / "trials.json")
        assert len(again.trials) == len(s.trials)
        assert again.results()[0].config == s.results()[0].config

    def test_reproduce_returns_the_exact_commands(self, tmp_path):
        s = self._store(tmp_path)
        best = s.results()[0]
        text = format_reproduce(s, best.config_id, tmp_path)
        assert "cmd b" in text
        assert text.count("#") == 2  # one per subject

    def test_reproduce_unknown_id_is_reported(self, tmp_path):
        s = self._store(tmp_path)
        assert "No trial" in format_reproduce(s, 999, tmp_path)

    def test_table_renders_and_offers_reproduce(self, tmp_path):
        s = self._store(tmp_path)
        text = format_results_table(s.results())
        assert "-reproduce" in text
        assert "formwarp" in text

    def test_empty_table(self):
        assert "no results" in format_results_table([])


class TestSubjectNames:
    """Labels must be unique: consensus ranks *within* a subject, and -reproduce
    writes per subject, so a collision silently merges two brains into one."""

    def test_distinct_filenames_use_the_stem(self):
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(["a/sub-01_T1w.nii.gz", "a/sub-02_T1w.nii.gz"])
        assert got == ["sub-01_T1w", "sub-02_T1w"]

    def test_flat_directory_does_not_collide(self):
        """The bug: every source sharing a parent dir collapsed to that dir's name."""
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(["p3/aff_A.nii.gz", "p3/aff_B.nii.gz"])
        assert got == ["aff_A", "aff_B"]
        assert len(set(got)) == 2

    def test_freesurfer_layout_uses_the_distinguishing_component(self):
        """sub-XXX/SUMA/brain.nii.gz: filename AND immediate parent both collide,
        so the useful component is two levels up."""
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(
            [
                "/data/sub-ME486585/SUMA/brain.nii.gz",
                "/data/sub-ME641463/SUMA/brain.nii.gz",
                "/data/sub-ME733720/SUMA/brain.nii.gz",
            ]
        )
        assert got == ["sub-ME486585", "sub-ME641463", "sub-ME733720"]

    def test_unequal_path_depths_still_resolve(self):
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(["a/sub-1/brain.nii.gz", "deep/tree/sub-2/brain.nii.gz"])
        assert len(set(got)) == 2

    def test_identical_paths_stay_unique(self):
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(["x/b.nii.gz", "x/b.nii.gz"])
        assert len(set(got)) == 2

    def test_handles_zst_and_afni_extensions(self):
        from fastfuncstuff.cli.tunewarp import subject_names

        got = subject_names(["d/one.nii.zst", "d/two.HEAD"])
        assert got == ["one", "two"]


class TestOverrides:
    def test_parse_fix_coerces_to_the_grid_type(self):
        from fastfuncstuff.processing.tunespec import parse_fix

        got = parse_fix(["formwarp.total_var=1", "qwarp.minpatch=9", "optiwarp.match=gradmag"])
        assert got["formwarp.total_var"] == 1.0
        assert isinstance(got["formwarp.total_var"], float)
        assert got["qwarp.minpatch"] == 9
        assert got["optiwarp.match"] == "gradmag"

    def test_parse_fix_handles_schedule_tuples(self):
        from fastfuncstuff.processing.tunespec import parse_fix

        assert parse_fix(["formwarp.iters=160x120x80"])["formwarp.iters"] == (160, 120, 80)

    def test_parse_fix_rejects_junk(self):
        from fastfuncstuff.processing.tunespec import parse_fix

        with pytest.raises(SystemExit):
            parse_fix(["formwarp.total_var"])
        with pytest.raises(KeyError):
            parse_fix(["formwarp.nope=1"])

    def test_fixing_shrinks_the_grid_and_pins_the_value(self):
        from fastfuncstuff.processing.tunespec import parse_fix, with_overrides
        from fastfuncstuff.processing.tunewarp import enumerate_configs

        base = RECIPES["MNI_T1"]
        fix = parse_fix(["formwarp.total_var=1.0"])
        r = with_overrides(base, fix, None)
        full = enumerate_configs(base, "formwarp")
        pinned = enumerate_configs(r, "formwarp", fixed=fix)
        assert len(pinned) < len(full)
        assert all(c["total_var"] == 1.0 for c in pinned)

    def test_tune_adds_a_knob_the_recipe_skips(self):
        from fastfuncstuff.processing.tunespec import with_overrides
        from fastfuncstuff.processing.tunewarp import enumerate_configs

        # max_step is an epi2epi knob; MNI_T1 leaves it alone, which is what makes it
        # a valid subject here. (formwarp.iters used to serve this role, but the
        # recipes now tune the iteration schedule by default.)
        base = RECIPES["MNI_T1"]
        r = with_overrides(base, None, ["optiwarp.max_step"])
        assert "optiwarp.max_step" not in base.tune
        assert len(enumerate_configs(r, "optiwarp_demons")) > len(
            enumerate_configs(base, "optiwarp_demons")
        )

    def test_optiwarp_fix_reaches_every_force_model(self):
        from fastfuncstuff.processing.tunespec import fixed_for, parse_fix

        fix = parse_fix(["optiwarp.total_sigma=1.0"])
        for force in ("demons", "lk", "hs"):
            assert fixed_for(fix, f"optiwarp_{force}") == {"total_sigma": 1.0}

    def test_fix_for_one_backend_does_not_leak_to_another(self):
        from fastfuncstuff.processing.tunespec import fixed_for, parse_fix

        fix = parse_fix(["formwarp.total_var=1.0"])
        assert fixed_for(fix, "qwarp") == {}


class TestKnobEffects:
    def _store(self, tmp_path):
        """Two subjects; total_var lower is better on both, consistently."""
        from fastfuncstuff.processing.tunestore import TrialStore

        s = TrialStore(tmp_path / "t.json")
        for subj, offset in (("s1", 0.0), ("s2", 5.0)):
            for tv, base in ((0.0, 1.0), (1.0, 3.0), (2.0, 5.0)):
                for uv in (1.0, 3.0):
                    s.add(
                        "formwarp",
                        subj,
                        {"total_var": tv, "update_var": uv},
                        ["c"],
                        scores={"consensus": base + offset + (0.5 if uv == 1.0 else 0.0)},
                        grade=PASS,
                    )
        return s

    def test_direction_and_full_consistency(self, tmp_path):
        from fastfuncstuff.processing.tunestore import knob_effects

        eff = {e.key: e for e in knob_effects(self._store(tmp_path))}
        assert eff["total_var"].direction == "lower is better"
        assert eff["total_var"].consistency == 1.0

    def test_consistency_is_marginal_on_both_sides(self, tmp_path):
        """The bug: comparing a pooled marginal to each subject's best JOINT
        config reported 0% agreement for a knob that every subject agreed on."""
        from fastfuncstuff.processing.tunestore import knob_effects

        eff = {e.key: e for e in knob_effects(self._store(tmp_path))}
        # update_var=3.0 wins on average for both subjects, even though the best
        # joint config is (total_var=0.0, update_var=3.0).
        assert eff["update_var"].direction == "higher is better"
        assert eff["update_var"].consistency == 1.0

    def test_disagreement_is_reported(self, tmp_path):
        from fastfuncstuff.processing.tunestore import TrialStore, knob_effects

        s = TrialStore(tmp_path / "d.json")
        for subj, best in (("s1", 0.0), ("s2", 2.0)):  # opposite optima
            for tv in (0.0, 2.0):
                s.add(
                    "formwarp",
                    subj,
                    {"total_var": tv},
                    ["c"],
                    scores={"consensus": 1.0 if tv == best else 9.0},
                    grade=PASS,
                )
        eff = {e.key: e for e in knob_effects(s)}
        assert eff["total_var"].consistency == 0.5

    def test_pinned_knob_is_skipped(self, tmp_path):
        from fastfuncstuff.processing.tunestore import TrialStore, knob_effects

        s = TrialStore(tmp_path / "p.json")
        for subj in ("s1", "s2"):
            s.add(
                "formwarp", subj, {"total_var": 1.0}, ["c"], scores={"consensus": 1.0}, grade=PASS
            )
        assert knob_effects(s) == []

    def test_fold_rate_is_reported_per_level(self, tmp_path):
        from fastfuncstuff.processing.tunestore import TrialStore, knob_effects

        s = TrialStore(tmp_path / "f.json")
        for tv, grade in ((0.0, FAIL), (1.0, PASS)):
            for subj in ("s1", "s2"):
                s.add(
                    "formwarp",
                    subj,
                    {"total_var": tv},
                    ["c"],
                    scores={"consensus": 1.0},
                    grade=grade,
                )
        levels = {v: fold for v, _, fold, _ in knob_effects(s)[0].levels}
        assert levels[0.0] == 1.0 and levels[1.0] == 0.0
